from __future__ import annotations

from resq.classical import brute_force
from resq.isolation import run_qaoa_isolated
from resq.qaoa import run_qaoa
from resq.qubo import DispatchProblem
from resq.scenario import demo_scenario
from tests._crash_target import crash_worker


def demo_problem() -> DispatchProblem:
    scenario = demo_scenario()
    return DispatchProblem.build(
        scenario.network, scenario.ambulances, scenario.incidents
    )


def test_isolated_qaoa_matches_seeded_in_process_result():
    problem = demo_problem()
    exact = brute_force(problem)
    kwargs = {
        "reps": 1,
        "shots": 1024,
        "seed": 7,
        "maxiter": 20,
        "optimal_bitstring": exact.bitstring,
    }

    expected = run_qaoa(problem, **kwargs)
    actual = run_qaoa_isolated(problem, **kwargs)

    assert actual.status == expected.status
    assert (
        actual.best_valid.bitstring if actual.best_valid else None
    ) == (expected.best_valid.bitstring if expected.best_valid else None)
    assert actual.counts == expected.counts


def test_worker_crash_returns_error_and_next_call_uses_fresh_worker():
    problem = demo_problem()

    crashed = run_qaoa_isolated(problem, target=crash_worker)

    assert crashed.status == "error"
    assert "QAOA worker process crashed" in crashed.message

    recovered = run_qaoa_isolated(
        problem,
        reps=1,
        shots=64,
        seed=7,
        maxiter=1,
    )
    assert recovered.status in {"ok", "no_valid_sample"}
