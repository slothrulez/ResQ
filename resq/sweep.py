"""QAOA parameter sweep: run multiple depths and compare results."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from resq.classical import brute_force
from resq.qaoa import QAOAResult, run_qaoa
from resq.qubo import DispatchProblem


@dataclass(frozen=True)
class SweepRow:
    """Result of a single QAOA run within a sweep."""

    reps: int
    status: str
    valid_fraction: float
    prob_of_exact_optimum: float | None
    best_valid_minutes: float | None
    exact_optimum_minutes: float
    matches_exact: bool | None
    circuit_depth: int | None
    two_qubit_gates: int | None
    runtime_seconds: float


@dataclass(frozen=True)
class SweepResult:
    """Collected results for a full depth sweep."""

    rows: tuple[SweepRow, ...]
    exact_optimum_minutes: float
    total_runtime_seconds: float
    shots: int
    seed: int


def run_sweep(
    problem: DispatchProblem,
    *,
    reps_values: tuple[int, ...] = (1, 2, 3),
    shots: int = 4096,
    seed: int = 7,
    maxiter: int = 40,
) -> SweepResult:
    """Run QAOA at each depth in *reps_values* and collect comparative results.

    Returns a :class:`SweepResult` with one :class:`SweepRow` per depth.
    """
    sweep_start = perf_counter()
    exact = brute_force(problem)
    rows: list[SweepRow] = []

    for reps in reps_values:
        result: QAOAResult = run_qaoa(
            problem,
            reps=reps,
            shots=shots,
            seed=seed,
            maxiter=maxiter,
            optimal_bitstring=exact.bitstring,
        )
        best_minutes: float | None = None
        matches: bool | None = None
        if result.best_valid is not None:
            best_minutes = result.best_valid.objective_minutes
            matches = best_minutes == exact.objective_minutes

        rows.append(
            SweepRow(
                reps=reps,
                status=result.status,
                valid_fraction=result.valid_fraction,
                prob_of_exact_optimum=result.prob_of_exact_optimum,
                best_valid_minutes=best_minutes,
                exact_optimum_minutes=exact.objective_minutes,
                matches_exact=matches,
                circuit_depth=result.circuit_depth,
                two_qubit_gates=result.two_qubit_gate_count,
                runtime_seconds=result.runtime_seconds,
            )
        )

    return SweepResult(
        rows=tuple(rows),
        exact_optimum_minutes=exact.objective_minutes,
        total_runtime_seconds=perf_counter() - sweep_start,
        shots=shots,
        seed=seed,
    )


def sweep_to_dicts(sweep: SweepResult) -> list[dict[str, Any]]:
    """Serialise a sweep result to a list of plain dictionaries."""
    return [
        {
            "reps": row.reps,
            "status": row.status,
            "valid_fraction": row.valid_fraction,
            "prob_of_exact_optimum": row.prob_of_exact_optimum,
            "best_valid_minutes": row.best_valid_minutes,
            "exact_optimum_minutes": row.exact_optimum_minutes,
            "matches_exact": row.matches_exact,
            "circuit_depth": row.circuit_depth,
            "two_qubit_gates": row.two_qubit_gates,
            "runtime_seconds": row.runtime_seconds,
        }
        for row in sweep.rows
    ]

