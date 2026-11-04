"""Tests for bridge closure impact analysis."""

from __future__ import annotations

from resq.analysis import (
    NAMED_BRIDGES,
    ClosureAnalysisReport,
    ClosureImpact,
    analyze_closures,
    closure_report_to_dicts,
)
from resq.scenario import demo_scenario


def test_analyze_closures_returns_correct_types():
    report = analyze_closures()
    assert isinstance(report, ClosureAnalysisReport)
    assert report.baseline_minutes == 34.0
    assert len(report.impacts) == len(NAMED_BRIDGES)
    for imp in report.impacts:
        assert isinstance(imp, ClosureImpact)
        assert imp.objective_minutes >= report.baseline_minutes
        assert imp.cost_increase_minutes >= 0.0


def test_venduruthy_bridge_is_most_critical():
    report = analyze_closures()
    most_critical = report.impacts[0]
    assert most_critical.bridge_name == "Venduruthy Bridge"
    assert most_critical.cost_increase_minutes == 34.0
    assert most_critical.cost_increase_percent == 100.0


def test_analyze_closures_three_incidents():
    report = analyze_closures(three_incidents=True)
    assert report.baseline_minutes == 49.0
    most_critical = report.impacts[0]
    assert most_critical.bridge_name == "Venduruthy Bridge"
    assert most_critical.cost_increase_minutes == 48.0
    assert round(most_critical.cost_increase_percent, 1) == 98.0


def test_closure_report_to_dicts_structure():
    report = analyze_closures()
    d = closure_report_to_dicts(report)
    assert "baseline_minutes" in d
    assert "impacts" in d
    assert len(d["impacts"]) == len(NAMED_BRIDGES)
    first = d["impacts"][0]
    assert first["bridge_name"] == "Venduruthy Bridge"
    assert first["cost_increase_percent"] == 100.0
