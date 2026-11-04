"""Tests for the QAOA parameter sweep feature."""

from __future__ import annotations

import json
import sys

import pytest

from resq import cli
from resq.classical import brute_force
from resq.qubo import DispatchProblem
from resq.scenario import DEMO_CLOSURE, demo_scenario
from resq import sweep as sweep_module
from resq.sweep import SweepResult, SweepRow, run_sweep, sweep_to_dicts


def _make_problem(closure: bool = False) -> DispatchProblem:
    scenario = demo_scenario()
    network = scenario.network.closed(DEMO_CLOSURE) if closure else scenario.network
    return DispatchProblem.build(network, scenario.ambulances, scenario.incidents)


def test_sweep_returns_one_row_per_depth():
    problem = _make_problem()
    depths = (1, 2)
    sweep = run_sweep(problem, reps_values=depths, shots=64, seed=7, maxiter=4)
    assert isinstance(sweep, SweepResult)
    assert len(sweep.rows) == len(depths)
    for row, expected_reps in zip(sweep.rows, depths):
        assert isinstance(row, SweepRow)
        assert row.reps == expected_reps


def test_sweep_exact_optimum_matches_brute_force():
    problem = _make_problem()
    exact = brute_force(problem)
    sweep = run_sweep(problem, reps_values=(1,), shots=64, seed=7, maxiter=4)
    assert sweep.exact_optimum_minutes == exact.objective_minutes


def test_sweep_valid_fraction_is_between_zero_and_one():
    problem = _make_problem()
    sweep = run_sweep(problem, reps_values=(1, 2), shots=64, seed=7, maxiter=4)
    for row in sweep.rows:
        assert 0.0 <= row.valid_fraction <= 1.0


def test_sweep_records_runtime():
    problem = _make_problem()
    sweep = run_sweep(problem, reps_values=(1,), shots=64, seed=7, maxiter=4)
    assert sweep.total_runtime_seconds > 0
    for row in sweep.rows:
        assert row.runtime_seconds > 0


def test_sweep_to_dicts_round_trips_fields():
    problem = _make_problem()
    sweep = run_sweep(problem, reps_values=(1,), shots=64, seed=7, maxiter=4)
    dicts = sweep_to_dicts(sweep)
    assert len(dicts) == 1
    d = dicts[0]
    assert d["reps"] == 1
    assert d["exact_optimum_minutes"] == sweep.exact_optimum_minutes
    assert "valid_fraction" in d
    assert "runtime_seconds" in d


def test_sweep_json_output_is_machine_readable(monkeypatch, capsys):
    expected = SweepResult(
        rows=(
            SweepRow(
                reps=1,
                status="ok",
                valid_fraction=1.0,
                prob_of_exact_optimum=0.5,
                best_valid_minutes=34.0,
                exact_optimum_minutes=34.0,
                matches_exact=True,
                circuit_depth=12,
                two_qubit_gates=8,
                runtime_seconds=0.1,
            ),
        ),
        exact_optimum_minutes=34.0,
        total_runtime_seconds=0.1,
        shots=4096,
        seed=7,
    )
    monkeypatch.setattr(sweep_module, "run_sweep", lambda *args, **kwargs: expected)
    monkeypatch.setattr(sys, "argv", ["resq.cli", "--sweep", "--json"])

    cli.main()

    assert json.loads(capsys.readouterr().out) == sweep_to_dicts(expected)


@pytest.mark.quantum
def test_sweep_best_valid_is_feasible_when_found():
    problem = _make_problem()
    sweep = run_sweep(problem, reps_values=(1, 2, 3), shots=4096, seed=7, maxiter=40)
    for row in sweep.rows:
        if row.best_valid_minutes is not None:
            assert row.best_valid_minutes >= row.exact_optimum_minutes
            assert row.matches_exact is not None


def test_sweep_with_closure():
    problem = _make_problem(closure=True)
    sweep = run_sweep(problem, reps_values=(1,), shots=64, seed=7, maxiter=4)
    assert sweep.exact_optimum_minutes == 68
    assert len(sweep.rows) == 1

