from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from time import perf_counter

import numpy as np
from scipy.optimize import linear_sum_assignment

from qrescue.qubo import (
    DispatchProblem,
    QUBO,
    assignment,
    build_qubo,
    check_constraints,
    objective_minutes,
    qubo_energy,
)


@dataclass(frozen=True)
class ExactResult:
    x: tuple[int, ...]
    bitstring: str
    assignment: tuple[tuple[str, str], ...]
    energy: float
    objective_minutes: float
    runtime_seconds: float


@dataclass(frozen=True)
class BaselineResult:
    x: tuple[int, ...]
    bitstring: str
    assignment: tuple[tuple[str, str], ...]
    objective_minutes: float
    runtime_seconds: float


def brute_force(problem: DispatchProblem) -> ExactResult:
    if problem.n_qubits > 20:
        raise ValueError(
            f"brute force refuses {problem.n_qubits} variables; maximum is 20"
        )
    started = perf_counter()
    qubo = build_qubo(problem)
    best_x: tuple[int, ...] | None = None
    best_energy = float("inf")
    for state in product((0, 1), repeat=problem.n_qubits):
        energy = qubo_energy(qubo, state)
        if energy < best_energy:
            best_x = tuple(state)
            best_energy = energy
    if best_x is None or not check_constraints(problem, best_x).valid:
        raise RuntimeError("QUBO ground state was not a feasible assignment")
    bitstring = "".join(str(bit) for bit in reversed(best_x))
    return ExactResult(
        x=best_x,
        bitstring=bitstring,
        assignment=tuple(assignment(problem, best_x)),
        energy=best_energy,
        objective_minutes=objective_minutes(problem, best_x),
        runtime_seconds=perf_counter() - started,
    )


def hungarian(problem: DispatchProblem) -> BaselineResult:
    started = perf_counter()
    finite_costs = problem.c[np.isfinite(problem.c)]
    if not finite_costs.size:
        raise ValueError("assignment matrix has no reachable pairs")
    sentinel = (float(finite_costs.max()) + 1.0) * (len(problem.incidents) + 1)
    costs = np.where(np.isfinite(problem.c), problem.c, sentinel)
    rows, columns = linear_sum_assignment(costs)
    if len(columns) != len(problem.incidents) or np.any(
        ~np.isfinite(problem.c[rows, columns])
    ):
        raise ValueError("Hungarian assignment selected an unreachable sentinel")

    x = [0] * len(problem.K)
    pair_to_index = {pair: index for index, pair in enumerate(problem.K)}
    chosen: list[tuple[str, str]] = []
    for ambulance_index, incident_index in zip(rows, columns):
        x[pair_to_index[(int(ambulance_index), int(incident_index))]] = 1
        chosen.append(
            (
                problem.ambulances[int(ambulance_index)].id,
                problem.incidents[int(incident_index)].id,
            )
        )
    x_tuple = tuple(x)
    bitstring = "".join(str(bit) for bit in reversed(x_tuple))
    return BaselineResult(
        x=x_tuple,
        bitstring=bitstring,
        assignment=tuple(chosen),
        objective_minutes=objective_minutes(problem, x_tuple),
        runtime_seconds=perf_counter() - started,
    )
