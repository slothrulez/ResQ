"""Hybrid quantum-classical ambulance dispatch research prototype."""

from qrescue.qubo import DispatchProblem, InfeasibleInstance, build_qubo
from qrescue.scenario import demo_scenario

__all__ = ["DispatchProblem", "InfeasibleInstance", "build_qubo", "demo_scenario"]
