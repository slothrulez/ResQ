from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np

from resq.network import RoadNetwork
from resq.scenario import Ambulance, Incident


class InfeasibleInstance(ValueError):
    pass


@dataclass(frozen=True)
class DispatchProblem:
    network: RoadNetwork
    ambulances: tuple[Ambulance, ...]
    incidents: tuple[Incident, ...]
    c: np.ndarray
    K: tuple[tuple[int, int], ...]
    c_max: float
    chat: np.ndarray
    P: float

    @classmethod
    def build(
        cls,
        network: RoadNetwork,
        ambulances: Sequence[Ambulance],
        incidents: Sequence[Incident],
    ) -> DispatchProblem:
        ambulances = tuple(ambulances)
        incidents = tuple(incidents)
        n_a, n_i = len(ambulances), len(incidents)
        if n_a < n_i:
            raise InfeasibleInstance(
                f"need at least as many ambulances as incidents ({n_a} < {n_i})"
            )
        if n_i == 0:
            raise ValueError("at least one incident is required")

        costs = np.full((n_a, n_i), np.inf, dtype=float)
        for a, ambulance in enumerate(ambulances):
            for i, incident in enumerate(incidents):
                distance = network.travel_time(ambulance.base_node, incident.node)
                if np.isfinite(distance):
                    costs[a, i] = float(incident.severity) * distance

        for i, incident in enumerate(incidents):
            if not np.isfinite(costs[:, i]).any():
                raise InfeasibleInstance(
                    f"incident {incident.id!r} has no reachable ambulance"
                )

        if not _has_perfect_matching(np.isfinite(costs)):
            raise InfeasibleInstance("no perfect ambulance-to-incident assignment exists")

        variables = tuple(
            (a, i)
            for a in range(n_a)
            for i in range(n_i)
            if np.isfinite(costs[a, i])
        )
        c_max = float(max(costs[a, i] for a, i in variables))
        normalized = np.full_like(costs, np.inf)
        for a, i in variables:
            normalized[a, i] = costs[a, i] / c_max
        penalty = 1.0 + sum(
            max(normalized[a, i] for a, i in variables if i == incident_index)
            for incident_index in range(n_i)
        )
        return cls(
            network=network,
            ambulances=ambulances,
            incidents=incidents,
            c=costs,
            K=variables,
            c_max=c_max,
            chat=normalized,
            P=float(penalty),
        )

    @property
    def n_qubits(self) -> int:
        return len(self.K)

    @property
    def normalized_costs(self) -> np.ndarray:
        return self.chat


@dataclass(frozen=True)
class QUBO:
    constant: float
    linear: dict[int, float]
    quadratic: dict[tuple[int, int], float]

    @property
    def n(self) -> int:
        return len(self.linear)


@dataclass(frozen=True)
class ConstraintReport:
    incident_counts: dict[str, int]
    ambulance_counts: dict[str, int]
    violations: tuple[str, ...]
    valid: bool


def build_qubo(problem: DispatchProblem) -> QUBO:
    index = {pair: k for k, pair in enumerate(problem.K)}
    constant = problem.P * len(problem.incidents)
    linear = {
        k: float(problem.chat[a, i] - problem.P)
        for k, (a, i) in enumerate(problem.K)
    }
    quadratic: dict[tuple[int, int], float] = {}

    for k, (a, i) in enumerate(problem.K):
        for other_k, (other_a, other_i) in enumerate(problem.K[k + 1 :], k + 1):
            if i == other_i and a != other_a:
                quadratic[(k, other_k)] = 2.0 * problem.P
            elif a == other_a and i != other_i:
                quadratic[(k, other_k)] = problem.P
    return QUBO(float(constant), linear, quadratic)


def qubo_energy(qubo: QUBO, x: Sequence[int | bool]) -> float:
    bits = _validate_x(x, qubo.n)
    energy = qubo.constant
    energy += sum(value * bits[k] for k, value in qubo.linear.items())
    energy += sum(value * bits[k] * bits[l] for (k, l), value in qubo.quadratic.items())
    return float(energy)


def objective_minutes(problem: DispatchProblem, x: Sequence[int | bool]) -> float:
    bits = _validate_x(x, len(problem.K))
    return float(
        sum(problem.c[a, i] * bits[k] for k, (a, i) in enumerate(problem.K))
    )


def check_constraints(problem: DispatchProblem, x: Sequence[int | bool]) -> ConstraintReport:
    bits = _validate_x(x, len(problem.K))
    incident_counts = {incident.id: 0 for incident in problem.incidents}
    ambulance_counts = {ambulance.id: 0 for ambulance in problem.ambulances}
    for bit, (a, i) in zip(bits, problem.K):
        if bit:
            incident_counts[problem.incidents[i].id] += 1
            ambulance_counts[problem.ambulances[a].id] += 1

    violations = [
        f"incident {incident_id} assigned {count} ambulances, expected exactly one"
        for incident_id, count in incident_counts.items()
        if count != 1
    ]
    violations.extend(
        f"ambulance {ambulance_id} assigned {count} incidents, expected at most one"
        for ambulance_id, count in ambulance_counts.items()
        if count > 1
    )
    return ConstraintReport(
        incident_counts=incident_counts,
        ambulance_counts=ambulance_counts,
        violations=tuple(violations),
        valid=not violations,
    )


def decode_bitstring(problem: DispatchProblem, bitstring: str) -> list[int]:
    if len(bitstring) != len(problem.K):
        raise ValueError(
            f"bitstring length {len(bitstring)} does not match {len(problem.K)} variables"
        )
    if any(character not in "01" for character in bitstring):
        raise ValueError("bitstring must contain only '0' and '1'")
    return [int(bitstring[len(problem.K) - 1 - k]) for k in range(len(problem.K))]


def assignment(
    problem: DispatchProblem, x: Sequence[int | bool]
) -> list[tuple[str, str]]:
    bits = _validate_x(x, len(problem.K))
    return [
        (problem.ambulances[a].id, problem.incidents[i].id)
        for bit, (a, i) in zip(bits, problem.K)
        if bit
    ]


def _validate_x(x: Sequence[int | bool], expected_length: int) -> list[int]:
    bits = list(x)
    if len(bits) != expected_length:
        raise ValueError(f"x length {len(bits)} does not match {expected_length} variables")
    if any(bit not in (0, 1, False, True) for bit in bits):
        raise ValueError("x must contain only binary values")
    return [int(bit) for bit in bits]


def _has_perfect_matching(reachable: np.ndarray) -> bool:
    n_ambulances, n_incidents = reachable.shape
    matched_ambulance = [-1] * n_ambulances

    def augment(incident_index: int, seen: set[int]) -> bool:
        for ambulance_index in range(n_ambulances):
            if not reachable[ambulance_index, incident_index] or ambulance_index in seen:
                continue
            seen.add(ambulance_index)
            if matched_ambulance[ambulance_index] == -1 or augment(
                matched_ambulance[ambulance_index], seen
            ):
                matched_ambulance[ambulance_index] = incident_index
                return True
        return False

    return all(augment(incident_index, set()) for incident_index in range(n_incidents))
