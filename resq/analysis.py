"""Bridge closure impact analysis for ambulance dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from resq.classical import brute_force
from resq.qubo import DispatchProblem
from resq.scenario import Scenario, demo_scenario

NAMED_BRIDGES: tuple[tuple[tuple[str, str], str], ...] = (
    (("Vyttila Jn", "Maradu"), "Venduruthy Bridge"),
    (("Thevara", "Willingdon Island"), "Thevara Bridge"),
    (("Kaloor", "Kakkanad"), "Kaloor-Kakkanad Bridge"),
)


@dataclass(frozen=True)
class ClosureImpact:
    bridge_name: str
    edge: tuple[str, str]
    objective_minutes: float
    cost_increase_minutes: float
    cost_increase_percent: float
    assignment: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ClosureAnalysisReport:
    baseline_minutes: float
    baseline_assignment: tuple[tuple[str, str], ...]
    impacts: tuple[ClosureImpact, ...]


def analyze_closures(
    scenario: Scenario | None = None,
    three_incidents: bool = False,
) -> ClosureAnalysisReport:
    """Evaluate dispatch cost impact of closing each backwater bridge."""
    scenario = scenario or demo_scenario(three_incidents=three_incidents)

    base_problem = DispatchProblem.build(
        scenario.network, scenario.ambulances, scenario.incidents
    )
    base_exact = brute_force(base_problem)
    base_mins = base_exact.objective_minutes

    impacts: list[ClosureImpact] = []
    for edge, bridge_name in NAMED_BRIDGES:
        closed_net = scenario.network.closed((edge,))
        closed_problem = DispatchProblem.build(
            closed_net, scenario.ambulances, scenario.incidents
        )
        exact = brute_force(closed_problem)
        diff = exact.objective_minutes - base_mins
        pct = (diff / base_mins) * 100.0 if base_mins > 0 else 0.0
        impacts.append(
            ClosureImpact(
                bridge_name=bridge_name,
                edge=edge,
                objective_minutes=exact.objective_minutes,
                cost_increase_minutes=diff,
                cost_increase_percent=pct,
                assignment=exact.assignment,
            )
        )

    impacts.sort(key=lambda imp: imp.cost_increase_minutes, reverse=True)

    return ClosureAnalysisReport(
        baseline_minutes=base_mins,
        baseline_assignment=base_exact.assignment,
        impacts=tuple(impacts),
    )


def closure_report_to_dicts(report: ClosureAnalysisReport) -> dict[str, Any]:
    return {
        "baseline_minutes": report.baseline_minutes,
        "baseline_assignment": report.baseline_assignment,
        "impacts": [
            {
                "bridge_name": imp.bridge_name,
                "edge": list(imp.edge),
                "objective_minutes": imp.objective_minutes,
                "cost_increase_minutes": imp.cost_increase_minutes,
                "cost_increase_percent": round(imp.cost_increase_percent, 1),
                "assignment": imp.assignment,
            }
            for imp in report.impacts
        ],
    }
