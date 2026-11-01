from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from qrescue.classical import BaselineResult, ExactResult, brute_force, hungarian
from qrescue.qaoa import QAOAResult, run_qaoa
from qrescue.qubo import DispatchProblem, QUBO, build_qubo
from qrescue.scenario import DEMO_CLOSURE, Scenario, demo_scenario


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


def solve(
    scenario: Scenario | None = None,
    closure: tuple[tuple[str, str], ...] | None = None,
    reps: int = 2,
    shots: int = 4096,
    seed: int = 7,
) -> Report:
    scenario = scenario or demo_scenario()
    closure_edges = tuple(closure or ())
    network = (
        scenario.network.closed(closure_edges) if closure_edges else scenario.network
    )
    timings: dict[str, float] = {}

    started = perf_counter()
    for ambulance in scenario.ambulances:
        for incident in scenario.incidents:
            network.travel_time(ambulance.base_node, incident.node)
            network.route(ambulance.base_node, incident.node)
    timings["routing"] = perf_counter() - started

    started = perf_counter()
    problem = DispatchProblem.build(network, scenario.ambulances, scenario.incidents)
    timings["problem"] = perf_counter() - started

    started = perf_counter()
    qubo = build_qubo(problem)
    timings["qubo"] = perf_counter() - started

    started = perf_counter()
    qaoa_result = run_qaoa(problem, reps=reps, shots=shots, seed=seed)
    timings["qaoa"] = perf_counter() - started

    started = perf_counter()
    exact = brute_force(problem)
    timings["exact_validation"] = perf_counter() - started
    if qaoa_result.counts:
        qaoa_result.prob_of_exact_optimum = (
            qaoa_result.counts.get(exact.bitstring, 0) / sum(qaoa_result.counts.values())
        )

    started = perf_counter()
    baseline = hungarian(problem)
    timings["hungarian_baseline"] = perf_counter() - started

    ambulance_by_id = {ambulance.id: ambulance for ambulance in scenario.ambulances}
    incident_by_id = {incident.id: incident for incident in scenario.incidents}
    assignments: dict[str, tuple[tuple[str, str], ...]] = {
        "qaoa": (
            qaoa_result.best_valid.assignment
            if qaoa_result.best_valid is not None
            else ()
        ),
        "exact_validation": exact.assignment,
        "hungarian_baseline": baseline.assignment,
    }
    routes: dict[str, list[dict[str, Any]]] = {}
    for label, chosen in assignments.items():
        routes[label] = []
        for ambulance_id, incident_id in chosen:
            ambulance = ambulance_by_id[ambulance_id]
            incident = incident_by_id[incident_id]
            routes[label].append(
                {
                    "ambulance_id": ambulance_id,
                    "incident_id": incident_id,
                    "nodes": network.route(ambulance.base_node, incident.node),
                    "travel_minutes": network.travel_time(
                        ambulance.base_node, incident.node
                    ),
                }
            )
    return Report(
        scenario=scenario,
        closure_edges=closure_edges,
        problem=problem,
        qubo=qubo,
        qaoa=qaoa_result,
        exact=exact,
        baseline=baseline,
        routes=routes,
        stage_seconds=timings,
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
