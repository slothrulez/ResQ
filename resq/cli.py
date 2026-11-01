from __future__ import annotations

import argparse
import json

from resq.pipeline import report_to_dict, solve
from resq.scenario import DEMO_CLOSURE, demo_scenario


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid QAOA ambulance dispatch demo")
    parser.add_argument("--closure", action="store_true", help="apply the named road closure")
    parser.add_argument(
        "--three-incidents", action="store_true", help="run the 3-ambulance, 3-incident variant"
    )
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", action="store_true", help="emit a JSON report")
    args = parser.parse_args()

    scenario = demo_scenario(three_incidents=args.three_incidents)
    report = solve(
        scenario=scenario,
        closure=DEMO_CLOSURE if args.closure else None,
        reps=args.reps,
        shots=args.shots,
        seed=args.seed,
    )
    if args.json:
        print(json.dumps(report_to_dict(report), indent=2))
    else:
        _print_report(report)


def _print_report(report) -> None:
    print("ResQ")
    print(
        f"Scenario: {len(report.scenario.ambulances)} ambulances, "
        f"{len(report.scenario.incidents)} incidents"
    )
    print(f"QUBO variables/qubits: {report.problem.n_qubits}; penalty P={report.problem.P:.6f}")
    if report.closure_edges:
        print(f"Road closure: {report.closure_edges}")
    else:
        print("Road closure: none")

    qaoa = report.qaoa
    print("\nQAOA (quantum, simulated)")
    print(f"  Status: {qaoa.status}")
    if qaoa.message:
        print(f"  Message: {qaoa.message}")
    print(
        f"  Backend: {qaoa.backend_name} ({qaoa.backend_method}); "
        f"reps={qaoa.reps}, shots={qaoa.shots}, seed={qaoa.seed}"
    )
    print(
        f"  Circuit: depth={qaoa.circuit_depth}, 2-qubit gates={qaoa.two_qubit_gate_count}; "
        f"COBYLA iterations={qaoa.optimiser_iterations}, evaluations={qaoa.optimiser_evals}"
    )
    print(
        f"  Expected energy={_format_number(qaoa.final_expected_energy)}; "
        f"valid fraction={qaoa.valid_fraction:.6f}; "
        f"exact-optimum probability={_format_number(qaoa.prob_of_exact_optimum)}"
    )
    if qaoa.best_valid is None:
        print("  Best valid sampled assignment: none")
    else:
        print(
            f"  Best valid sampled assignment: {qaoa.best_valid.assignment}; "
            f"weighted minutes={qaoa.best_valid.objective_minutes:.3f}; "
            f"bitstring={qaoa.best_valid.bitstring}"
        )
    print(f"  Runtime: {qaoa.runtime_seconds:.3f}s")
    print("  Top sampled bitstrings")
    _print_sample_table(qaoa.top_k)

    print("\nEXACT validation (classical)")
    print(
        f"  Assignment: {report.exact.assignment}; "
        f"weighted minutes={report.exact.objective_minutes:.3f}; "
        f"QUBO energy={report.exact.energy:.6f}"
    )
    print(f"  Runtime: {report.exact.runtime_seconds:.3f}s")

    print("\nCLASSICAL baseline (Hungarian)")
    print(
        f"  Assignment: {report.baseline.assignment}; "
        f"weighted minutes={report.baseline.objective_minutes:.3f}"
    )
    print(f"  Runtime: {report.baseline.runtime_seconds:.3f}s")
    print("\nCLASSICAL routing (Dijkstra) — routes for selected assignments")
    for label, routes in report.routes.items():
        print(f"  {label}:")
        if not routes:
            print("    no assignment available")
        for route in routes:
            print(
                f"    {route['ambulance_id']} -> {route['incident_id']}: "
                f"{route['nodes']} ({route['travel_minutes']:.1f} min)"
            )

    print("\nStage runtimes")
    for stage, seconds in report.stage_seconds.items():
        print(f"  {stage}: {seconds:.3f}s")
    print("\nNo quantum-advantage claim is made; QAOA uses a classical simulator.")


def _format_number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def _print_sample_table(samples) -> None:
    headers = ("bitstring", "count", "probability", "valid", "minutes", "assignment")
    rows = []
    for sample in samples:
        chosen = sample["assignment"]
        assignment_text = (
            "—"
            if chosen is None
            else ", ".join(f"{ambulance}→{incident}" for ambulance, incident in chosen)
        )
        minutes = (
            "n/a"
            if sample["objective_minutes"] is None
            else f"{sample['objective_minutes']:.3f}"
        )
        rows.append(
            (
                sample["bitstring"],
                str(sample["count"]),
                f"{sample['probability']:.6f}",
                str(sample["valid"]),
                minutes,
                assignment_text,
            )
        )
    widths = tuple(
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    )
    print("    " + " | ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("    " + "-+-".join("-" * width for width in widths))
    for row in rows:
        print("    " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(row)))


if __name__ == "__main__":
    main()
