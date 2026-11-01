from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from collections.abc import Iterable
from typing import Any

from qrescue.classical import BaselineResult, ExactResult, brute_force, hungarian
from qrescue.network import RoadNetwork
from qrescue.qaoa import QAOAResult, run_qaoa
from qrescue.qubo import DispatchProblem, QUBO, build_qubo
from qrescue.scenario import DEMO_CLOSURE, Scenario, demo_scenario


@dataclass
class Prepared:
    network: RoadNetwork
    problem: DispatchProblem
    qubo: QUBO
    routing_time_seconds: float
    qubo_time_seconds: float


@dataclass
class Report:
    scenario: Scenario
    closure_edges: tuple[tuple[str, str], ...]
    problem: DispatchProblem
    qubo: QUBO
    qaoa: QAOAResult
    exact: ExactResult
    baseline: BaselineResult
    routes: dict[str, list[dict[str, Any]]]
    stage_seconds: dict[str, float]


def prepare(
    scenario: Scenario,
    closure: Iterable[tuple[str, str]] | None = None,
) -> Prepared:
    closure_edges = tuple(closure or ())
    network = scenario.network.closed(closure_edges) if closure_edges else scenario.network

    started = perf_counter()
    problem = DispatchProblem.build(network, scenario.ambulances, scenario.incidents)
    routing_time = perf_counter() - started

    started = perf_counter()
    qubo = build_qubo(problem)
    qubo_time = perf_counter() - started
    return Prepared(
        network=network,
        problem=problem,
        qubo=qubo,
        routing_time_seconds=routing_time,
        qubo_time_seconds=qubo_time,
    )


def run_classical(
    prepared: Prepared,
) -> tuple[ExactResult, BaselineResult]:
    return brute_force(prepared.problem), hungarian(prepared.problem)


def routes_for(
    network: RoadNetwork,
    scenario: Scenario,
    chosen_assignment: Iterable[tuple[str, str]],
) -> list[dict[str, Any]]:
    ambulance_by_id = {ambulance.id: ambulance for ambulance in scenario.ambulances}
    incident_by_id = {incident.id: incident for incident in scenario.incidents}
    routes = []
    for ambulance_id, incident_id in chosen_assignment:
        try:
            ambulance = ambulance_by_id[ambulance_id]
            incident = incident_by_id[incident_id]
        except KeyError as error:
            raise ValueError(f"unknown assignment id {error.args[0]!r}") from error
        routes.append(
            {
                "ambulance_id": ambulance_id,
                "incident_id": incident_id,
                "nodes": network.route(ambulance.base_node, incident.node),
                "travel_minutes": network.travel_time(
                    ambulance.base_node, incident.node
                ),
            }
        )
    return routes


def solve(
    scenario: Scenario | None = None,
    closure: Iterable[tuple[str, str]] | None = None,
    reps: int = 2,
    shots: int = 4096,
    seed: int = 7,
) -> Report:
    scenario = scenario or demo_scenario()
    closure_edges = tuple(closure or ())
    prepared = prepare(scenario, closure_edges)

    started = perf_counter()
    qaoa_result = run_qaoa(prepared.problem, reps=reps, shots=shots, seed=seed)
    qaoa_runtime = perf_counter() - started

    exact, baseline = run_classical(prepared)
    if qaoa_result.counts:
        qaoa_result.prob_of_exact_optimum = (
            qaoa_result.counts.get(exact.bitstring, 0) / sum(qaoa_result.counts.values())
        )

    assignments: dict[str, tuple[tuple[str, str], ...]] = {
        "qaoa": (
            qaoa_result.best_valid.assignment
            if qaoa_result.best_valid is not None
            else ()
        ),
        "exact_validation": exact.assignment,
        "hungarian_baseline": baseline.assignment,
    }
    routes = {
        label: routes_for(prepared.network, scenario, chosen)
        for label, chosen in assignments.items()
    }
    return Report(
        scenario=scenario,
        closure_edges=closure_edges,
        problem=prepared.problem,
        qubo=prepared.qubo,
        qaoa=qaoa_result,
        exact=exact,
        baseline=baseline,
        routes=routes,
        stage_seconds={
            "routing": prepared.routing_time_seconds,
            "qubo": prepared.qubo_time_seconds,
            "qaoa": qaoa_runtime,
            "exact_validation": exact.runtime_seconds,
            "hungarian_baseline": baseline.runtime_seconds,
        },
    )


def report_to_dict(report: Report) -> dict[str, Any]:
    return {
        "scenario": {
            "ambulances": [
                {
                    "id": ambulance.id,
                    "name": ambulance.name,
                    "base_node": ambulance.base_node,
                }
                for ambulance in report.scenario.ambulances
            ],
            "incidents": [
                {
                    "id": incident.id,
                    "node": incident.node,
                    "severity": incident.severity,
                    "description": incident.description,
                }
                for incident in report.scenario.incidents
            ],
        },
        "closure_edges": [list(edge) for edge in report.closure_edges],
        "problem": {
            "n_qubits": report.problem.n_qubits,
            "K": [list(pair) for pair in report.problem.K],
            "c": report.problem.c.tolist(),
            "chat": report.problem.chat.tolist(),
            "c_max": report.problem.c_max,
            "P": report.problem.P,
        },
        "qubo": {
            "constant": report.qubo.constant,
            "linear": report.qubo.linear,
            "quadratic": {
                f"{k},{l}": value for (k, l), value in report.qubo.quadratic.items()
            },
        },
        "qaoa": {
            "status": report.qaoa.status,
            "message": report.qaoa.message,
            "best_valid": _candidate_to_dict(report.qaoa.best_valid),
            "optimal_params": report.qaoa.optimal_params,
            "final_expected_energy": report.qaoa.final_expected_energy,
            "n_qubits": report.qaoa.n_qubits,
            "reps": report.qaoa.reps,
            "circuit_depth": report.qaoa.circuit_depth,
            "two_qubit_gate_count": report.qaoa.two_qubit_gate_count,
            "optimiser_iterations": report.qaoa.optimiser_iterations,
            "optimiser_evals": report.qaoa.optimiser_evals,
            "shots": report.qaoa.shots,
            "seed": report.qaoa.seed,
            "valid_fraction": report.qaoa.valid_fraction,
            "prob_of_exact_optimum": report.qaoa.prob_of_exact_optimum,
            "top_k": report.qaoa.top_k,
            "counts": report.qaoa.counts,
            "runtime_seconds": report.qaoa.runtime_seconds,
            "backend_name": report.qaoa.backend_name,
            "backend_method": report.qaoa.backend_method,
        },
        "exact_validation": {
            "x": report.exact.x,
            "bitstring": report.exact.bitstring,
            "assignment": report.exact.assignment,
            "energy": report.exact.energy,
            "objective_minutes": report.exact.objective_minutes,
            "runtime_seconds": report.exact.runtime_seconds,
        },
        "hungarian_baseline": {
            "x": report.baseline.x,
            "bitstring": report.baseline.bitstring,
            "assignment": report.baseline.assignment,
            "objective_minutes": report.baseline.objective_minutes,
            "runtime_seconds": report.baseline.runtime_seconds,
        },
        "routes": report.routes,
        "stage_seconds": report.stage_seconds,
    }


def _candidate_to_dict(candidate: Any) -> dict[str, Any] | None:
    if candidate is None:
        return None
    return {
        "bitstring": candidate.bitstring,
        "x": candidate.x,
        "assignment": candidate.assignment,
        "objective_minutes": candidate.objective_minutes,
        "qubo_energy": candidate.qubo_energy,
    }


def demo_closure() -> tuple[tuple[str, str], ...]:
    return DEMO_CLOSURE
