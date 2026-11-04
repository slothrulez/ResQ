"""Hybrid quantum-classical ambulance dispatch research prototype."""

from resq.analysis import ClosureAnalysisReport, ClosureImpact, analyze_closures
from resq.qubo import DispatchProblem, InfeasibleInstance, build_qubo
from resq.scenario import demo_scenario
from resq.sweep import SweepResult, run_sweep

__all__ = [
    "DispatchProblem",
    "InfeasibleInstance",
    "build_qubo",
    "demo_scenario",
    "SweepResult",
    "run_sweep",
    "analyze_closures",
    "ClosureImpact",
    "ClosureAnalysisReport",
]
