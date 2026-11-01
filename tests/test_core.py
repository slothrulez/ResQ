from __future__ import annotations

from itertools import product

import networkx as nx
import numpy as np
import pytest

from qrescue.classical import brute_force, hungarian
from qrescue.ising import qubo_to_ising
from qrescue.network import RoadNetwork
from qrescue.pipeline import prepare, routes_for, run_classical
from qrescue.qaoa import (
    aggregate_samples,
    build_qaoa_circuit,
    run_qaoa,
)
from qrescue.qubo import (
    DispatchProblem,
    InfeasibleInstance,
    assignment,
    build_qubo,
    check_constraints,
    decode_bitstring,
    objective_minutes,
    qubo_energy,
)
from qrescue.scenario import (
    DEMO_CLOSURE,
    Ambulance,
    EAST_BANK,
    Incident,
    Scenario,
    WEST_BANK,
    demo_scenario,
)


def make_problem(three_incidents: bool = False, closure: bool = False) -> DispatchProblem:
    scenario = demo_scenario(three_incidents=three_incidents)
    network = scenario.network.closed(DEMO_CLOSURE) if closure else scenario.network
    return DispatchProblem.build(network, scenario.ambulances, scenario.incidents)


def set_pairs(problem: DispatchProblem, pairs: list[tuple[int, int]]) -> list[int]:
    x = [0] * len(problem.K)
    for pair in pairs:
        x[problem.K.index(pair)] = 1
    return x


def test_qubo_energy_matches_normalized_objective_and_penalty_for_all_states():
    problem = make_problem()
    qubo = build_qubo(problem)
    for state in product((0, 1), repeat=problem.n_qubits):
        report = check_constraints(problem, state)
        energy = qubo_energy(qubo, state)
        if report.valid:
            assert energy == pytest.approx(objective_minutes(problem, state) / problem.c_max)
        else:
            assert energy >= problem.P - 1e-10


def test_ising_diagonal_matches_qubo_for_every_basis_state():
    problem = make_problem()
    qubo = build_qubo(problem)
    operator, offset = qubo_to_ising(qubo, problem.n_qubits)
    diagonal = np.diag(operator.to_matrix())
    for basis_index, energy in enumerate(diagonal):
        bitstring = format(basis_index, f"0{problem.n_qubits}b")
        x = decode_bitstring(problem, bitstring)
        assert energy.real + offset == pytest.approx(qubo_energy(qubo, x))
        assert energy.imag == pytest.approx(0.0)


def test_constraint_checker_for_valid_and_invalid_assignments():
    problem = make_problem(three_incidents=True)
    valid = set_pairs(problem, [(0, 0), (1, 1), (2, 2)])
    report = check_constraints(problem, valid)
    assert report.valid
    assert report.incident_counts == {"inc-1": 1, "inc-2": 1, "inc-3": 1}
    assert report.ambulance_counts == {"amb-1": 1, "amb-2": 1, "amb-3": 1}

    missing_incident = [0] * len(problem.K)
    assert not check_constraints(problem, missing_incident).valid

    double_assigned_incident = set_pairs(problem, [(0, 0), (1, 0), (2, 1)])
    report = check_constraints(problem, double_assigned_incident)
    assert not report.valid
    assert report.incident_counts["inc-1"] == 2

    ambulance_with_two_incidents = set_pairs(problem, [(0, 0), (0, 1), (2, 2)])
    report = check_constraints(problem, ambulance_with_two_incidents)
    assert not report.valid
    assert report.ambulance_counts["amb-1"] == 2


def test_little_endian_decoding_and_invalid_bitstrings():
    problem = make_problem()
    n = len(problem.K)
    assert decode_bitstring(problem, "0" * (n - 1) + "1") == [1, 0, 0, 0, 0, 0]
    assert decode_bitstring(problem, "1" + "0" * (n - 1)) == [0, 0, 0, 0, 0, 1]
    with pytest.raises(ValueError, match="length"):
        decode_bitstring(problem, "0")
    with pytest.raises(ValueError, match="only"):
        decode_bitstring(problem, "0" * (n - 1) + "x")


def _random_problem(seed: int) -> DispatchProblem:
    rng = np.random.default_rng(seed)
    graph = nx.complete_graph([str(index) for index in range(6)])
    for u, v in graph.edges:
        graph[u][v]["minutes"] = float(rng.uniform(10.0, 1000.0))
    network = RoadNetwork(graph)
    ambulances = tuple(
        Ambulance(f"a{i}", f"Unit {i}", str(i)) for i in range(3)
    )
    incidents = tuple(
        Incident(f"i{i}", str(i + 3), int(rng.integers(1, 4)), "synthetic")
        for i in range(3)
    )
    return DispatchProblem.build(network, ambulances, incidents)


@pytest.mark.parametrize(
    "problem_factory",
    [
        lambda: make_problem(),
        lambda: make_problem(closure=True),
        lambda: make_problem(three_incidents=True),
        *[lambda seed=seed: _random_problem(seed) for seed in range(20)],
    ],
)
def test_brute_force_and_hungarian_agree(problem_factory):
    problem = problem_factory()
    exact = brute_force(problem)
    baseline = hungarian(problem)
    assert exact.assignment == baseline.assignment
    assert exact.objective_minutes == pytest.approx(baseline.objective_minutes)
    assert check_constraints(problem, exact.x).valid


@pytest.mark.parametrize(
    "problem_factory",
    [
        lambda: make_problem(),
        lambda: make_problem(closure=True),
        lambda: make_problem(three_incidents=True),
        *[lambda seed=seed: _random_problem(seed) for seed in range(20)],
    ],
)
def test_qubo_ground_state_is_valid(problem_factory):
    problem = problem_factory()
    exact = brute_force(problem)
    assert check_constraints(problem, exact.x).valid


def test_qaoa_circuit_dimensions_parameters_and_measurements():
    problem = make_problem()
    reps = 2
    circuit = build_qaoa_circuit(problem, reps=reps)
    sampling_circuit = build_qaoa_circuit(problem, reps=reps, measure=True)
    assert circuit.num_qubits == len(problem.K)
    assert circuit.num_parameters == 2 * reps
    assert sampling_circuit.num_qubits == len(problem.K)
    assert sampling_circuit.num_clbits == len(problem.K)
    assert any(instruction.operation.name == "measure" for instruction in sampling_circuit.data)


@pytest.mark.quantum
def test_aer_qaoa_execution_is_seeded_and_reports_valid_samples():
    problem = make_problem()
    exact = brute_force(problem)
    kwargs = {
        "reps": 2,
        "shots": 128,
        "seed": 31,
        "maxiter": 8,
        "optimal_bitstring": exact.bitstring,
    }
    first = run_qaoa(problem, **kwargs)
    second = run_qaoa(problem, **kwargs)
    assert first.status in {"ok", "no_valid_sample"}
    assert sum(first.counts.values()) == kwargs["shots"]
    assert first.counts == second.counts
    assert first.optimal_params == second.optimal_params
    assert first.optimiser_iterations > 0
    assert first.optimiser_evals > 0
    if first.status == "ok":
        assert first.best_valid is not None
        assert check_constraints(problem, first.best_valid.x).valid
        assert first.best_valid.objective_minutes == pytest.approx(
            objective_minutes(problem, first.best_valid.x)
        )
        assert first.best_valid.objective_minutes >= exact.objective_minutes


def test_invalid_sample_aggregation_never_substitutes_a_classical_answer():
    problem = make_problem()
    summary = aggregate_samples(
        problem,
        {"0" * len(problem.K): 4, "1" * len(problem.K): 6},
    )
    assert summary.status == "no_valid_sample"
    assert summary.best_valid is None
    assert summary.valid_fraction == 0.0


def test_closure_changes_unique_optimum_and_recomputes_costs():
    scenario = demo_scenario()
    closed_network = scenario.network.closed(DEMO_CLOSURE)
    open_problem = DispatchProblem.build(
        scenario.network, scenario.ambulances, scenario.incidents
    )
    closed_problem = DispatchProblem.build(
        closed_network, scenario.ambulances, scenario.incidents
    )
    open_exact = brute_force(open_problem)
    closed_exact = brute_force(closed_problem)
    assert open_exact.assignment != closed_exact.assignment
    assert open_exact.objective_minutes == 34
    assert closed_exact.objective_minutes == 68
    assert not np.array_equal(open_problem.c, closed_problem.c)
    assert np.isfinite(closed_problem.c).all()

    for three_incidents, expected_open, expected_closed in ((False, 34, 68), (True, 49, 97)):
        scenario = demo_scenario(three_incidents=three_incidents)
        open_problem = DispatchProblem.build(
            scenario.network, scenario.ambulances, scenario.incidents
        )
        closed_network = scenario.network.closed(DEMO_CLOSURE)
        closed_problem = DispatchProblem.build(
            closed_network, scenario.ambulances, scenario.incidents
        )
        for problem, expected in (
            (open_problem, expected_open),
            (closed_problem, expected_closed),
        ):
            assert np.isfinite(problem.c).all()
            exact = brute_force(problem)
            assert exact.objective_minutes == expected
            feasible_costs = [
                objective_minutes(problem, state)
                for state in product((0, 1), repeat=problem.n_qubits)
                if check_constraints(problem, state).valid
            ]
            assert feasible_costs.count(min(feasible_costs)) == 1

    open_by_incident = {incident_id: ambulance_id for ambulance_id, incident_id in open_exact.assignment}
    closed_by_incident = {
        incident_id: ambulance_id for ambulance_id, incident_id in closed_exact.assignment
    }
    open_ambulance = next(
        ambulance for ambulance in scenario.ambulances
        if ambulance.id == open_by_incident["inc-2"]
    )
    closed_ambulance = next(
        ambulance for ambulance in scenario.ambulances
        if ambulance.id == closed_by_incident["inc-2"]
    )
    open_path = scenario.network.route(open_ambulance.base_node, "Aluva")
    closed_path = closed_network.route(closed_ambulance.base_node, "Aluva")
    assert open_path is not None and closed_path is not None
    open_bridge_names = {
        scenario.network.graph[u][v].get("name")
        for u, v in zip(open_path, open_path[1:])
    }
    closed_bridge_names = {
        closed_network.graph[u][v].get("name")
        for u, v in zip(closed_path, closed_path[1:])
    }
    assert "Venduruthy Bridge" in open_bridge_names
    assert "Thevara Bridge" in closed_bridge_names


def test_scenario_is_mesh_with_three_named_backwater_bridges():
    scenario = demo_scenario()
    graph = scenario.network.graph
    assert len(graph) == 12
    assert all(2 <= degree <= 4 for _, degree in graph.degree())
    crossing_edges = [
        (u, v, data)
        for u, v, data in graph.edges(data=True)
        if (u in WEST_BANK and v in EAST_BANK) or (v in WEST_BANK and u in EAST_BANK)
    ]
    assert len(crossing_edges) == 3
    assert {data["name"] for _, _, data in crossing_edges} == {
        "Venduruthy Bridge",
        "Thevara Bridge",
        "Kaloor-Kakkanad Bridge",
    }


def test_prepare_supports_unreachable_pairs_with_a_smaller_variable_set():
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        (
            ("a0", "i0", 2),
            ("a0", "a1", 1),
            ("a1", "i1", 2),
            ("a2", "i1", 3),
        ),
        weight="minutes",
    )
    network = RoadNetwork(graph)
    scenario = Scenario(
        network=network,
        ambulances=(
            Ambulance("a0", "A0", "a0"),
            Ambulance("a1", "A1", "a1"),
            Ambulance("a2", "A2", "a2"),
        ),
        incidents=(
            Incident("i0", "i0", 2, "first"),
            Incident("i1", "i1", 1, "second"),
        ),
        coordinates={node: (float(index), 0.0) for index, node in enumerate(graph)},
    )
    open_prepared = prepare(scenario)
    closed_prepared = prepare(scenario, closure=(("a0", "a1"),))
    assert open_prepared.problem.n_qubits == 6
    assert closed_prepared.problem.n_qubits == 3
    exact, baseline = run_classical(closed_prepared)
    assert exact.assignment == baseline.assignment
    assert len(routes_for(closed_prepared.network, scenario, exact.assignment)) == 2


def test_prepare_raises_for_infeasible_and_nonexistent_closure_edges():
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        (
            ("a0", "i0", 2),
            ("a0", "a1", 1),
            ("a1", "i1", 2),
            ("a2", "i1", 3),
        ),
        weight="minutes",
    )
    scenario = Scenario(
        network=RoadNetwork(graph),
        ambulances=(
            Ambulance("a0", "A0", "a0"),
            Ambulance("a1", "A1", "a1"),
            Ambulance("a2", "A2", "a2"),
        ),
        incidents=(
            Incident("i0", "i0", 2, "first"),
            Incident("i1", "i1", 1, "second"),
        ),
        coordinates={node: (float(index), 0.0) for index, node in enumerate(graph)},
    )
    with pytest.raises(InfeasibleInstance, match="no reachable ambulance"):
        prepare(
            scenario,
            closure=(("a1", "i1"), ("a2", "i1")),
        )
    with pytest.raises(ValueError, match="does not exist"):
        prepare(scenario, closure=(("unknown", "edge"),))


def test_infeasible_problem_is_rejected_before_quantum_execution():
    graph = nx.Graph()
    graph.add_edge("a", "i", minutes=1)
    network = RoadNetwork(graph)
    ambulances = (Ambulance("a0", "A", "a"),)
    incidents = (
        Incident("i0", "i", 1, "first"),
        Incident("i1", "i", 1, "second"),
    )
    with pytest.raises(InfeasibleInstance, match="ambulances as incidents"):
        DispatchProblem.build(network, ambulances, incidents)


def test_quantum_backend_exception_returns_error_without_substitute(monkeypatch):
    import qrescue.qaoa as qaoa_module

    def fail_backend(*args, **kwargs):
        raise RuntimeError("injected Aer failure")

    monkeypatch.setattr(qaoa_module, "AerSimulator", fail_backend)
    result = run_qaoa(make_problem(), reps=1, shots=8, seed=7, maxiter=1)
    assert result.status == "error"
    assert result.best_valid is None
    assert "injected Aer failure" in result.message
