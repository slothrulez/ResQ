from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import qaoa_ansatz
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2, SamplerV2
from scipy.optimize import minimize

from qrescue.ising import qubo_to_ising
from qrescue.qubo import (
    DispatchProblem,
    assignment,
    build_qubo,
    check_constraints,
    decode_bitstring,
    objective_minutes,
)


@dataclass(frozen=True)
class Candidate:
    bitstring: str
    x: tuple[int, ...]
    assignment: tuple[tuple[str, str], ...]
    objective_minutes: float
    qubo_energy: float


@dataclass(frozen=True)
class SampleSummary:
    status: str
    best_valid: Candidate | None
    valid_fraction: float
    prob_of_exact_optimum: float | None
    top_k: tuple[dict[str, Any], ...]


@dataclass
class QAOAResult:
    status: str
    best_valid: Candidate | None
    optimal_params: tuple[float, ...] | None
    final_expected_energy: float | None
    n_qubits: int
    reps: int
    circuit_depth: int | None
    two_qubit_gate_count: int | None
    optimiser_iterations: int
    optimiser_evals: int
    shots: int
    seed: int
    valid_fraction: float
    prob_of_exact_optimum: float | None
    top_k: tuple[dict[str, Any], ...]
    counts: dict[str, int]
    runtime_seconds: float
    backend_name: str
    backend_method: str
    message: str | None = None


def build_qaoa_circuit(
    problem: DispatchProblem, reps: int = 2, measure: bool = False
) -> QuantumCircuit:
    qubo = build_qubo(problem)
    operator, _ = qubo_to_ising(qubo, problem.n_qubits)
    circuit = qaoa_ansatz(operator, reps=reps)
    if measure:
        measured = QuantumCircuit(problem.n_qubits, problem.n_qubits)
        measured.compose(circuit, inplace=True)
        measured.measure(range(problem.n_qubits), range(problem.n_qubits))
        return measured
    return circuit


def aggregate_samples(
    problem: DispatchProblem,
    counts: dict[str, int],
    optimal_bitstring: str | None = None,
    top_k: int = 10,
) -> SampleSummary:
    total_shots = sum(counts.values())
    if total_shots <= 0:
        return SampleSummary("no_valid_sample", None, 0.0, None, ())

    ranked: list[dict[str, Any]] = []
    best_valid: Candidate | None = None
    valid_shots = 0
    for bitstring, count in counts.items():
        x = _decode_counts_bitstring(problem, bitstring)
        report = check_constraints(problem, x)
        objective = objective_minutes(problem, x)
        is_valid = report.valid
        ranked.append(
            {
                "bitstring": bitstring.replace(" ", ""),
                "count": int(count),
                "probability": count / total_shots,
                "valid": is_valid,
                "objective_minutes": objective if is_valid else None,
                "assignment": assignment(problem, x) if is_valid else None,
            }
        )
        if is_valid:
            valid_shots += count
            candidate = Candidate(
                bitstring=bitstring.replace(" ", ""),
                x=tuple(x),
                assignment=tuple(assignment(problem, x)),
                objective_minutes=objective,
                qubo_energy=_qubo_energy_for_problem(problem, x),
            )
            if best_valid is None or (
                candidate.objective_minutes,
                candidate.bitstring,
            ) < (best_valid.objective_minutes, best_valid.bitstring):
                best_valid = candidate

    ranked.sort(key=lambda row: (-row["count"], row["bitstring"]))
    exact_probability = (
        counts.get(optimal_bitstring, 0) / total_shots
        if optimal_bitstring is not None
        else None
    )
    return SampleSummary(
        status="ok" if best_valid is not None else "no_valid_sample",
        best_valid=best_valid,
        valid_fraction=valid_shots / total_shots,
        prob_of_exact_optimum=exact_probability,
        top_k=tuple(ranked[:top_k]),
    )


def run_qaoa(
    problem: DispatchProblem,
    reps: int = 2,
    shots: int = 4096,
    seed: int = 7,
    maxiter: int = 40,
    optimal_bitstring: str | None = None,
) -> QAOAResult:
    started = perf_counter()
    result_base = {
        "n_qubits": problem.n_qubits,
        "reps": reps,
        "circuit_depth": None,
        "two_qubit_gate_count": None,
        "optimiser_iterations": 0,
        "optimiser_evals": 0,
        "shots": shots,
        "seed": seed,
        "valid_fraction": 0.0,
        "prob_of_exact_optimum": None,
        "top_k": (),
        "counts": {},
        "runtime_seconds": 0.0,
        "backend_name": "aer_simulator",
        "backend_method": "statevector expectation; Aer sampler",
    }
    try:
        if reps < 1:
            raise ValueError("reps must be at least 1")
        if shots < 1:
            raise ValueError("shots must be at least 1")
        if maxiter < 1:
            raise ValueError("maxiter must be at least 1")

        qubo = build_qubo(problem)
        operator, offset = qubo_to_ising(qubo, problem.n_qubits)
        ansatz = qaoa_ansatz(operator, reps=reps)
        backend = AerSimulator(method="statevector", seed_simulator=seed)
        result_base["backend_name"] = backend.name
        result_base["backend_method"] = (
            "statevector Aer EstimatorV2 expectation; Aer SamplerV2"
        )
        transpiled = transpile(ansatz, backend=backend, seed_transpiler=seed)

        estimator = EstimatorV2(
            options={
                "default_precision": 0.0,
                "backend_options": {"method": "statevector", "seed_simulator": seed},
            }
        )
        parameters = list(transpiled.parameters)
        gamma_parameters = sorted(
            (p for p in parameters if "γ" in str(p) or "gamma" in str(p)),
            key=lambda p: int(str(p).split("[")[-1].rstrip("]")),
        )
        beta_parameters = sorted(
            (p for p in parameters if "β" in str(p) or "beta" in str(p)),
            key=lambda p: int(str(p).split("[")[-1].rstrip("]")),
        )
        if len(gamma_parameters) != reps or len(beta_parameters) != reps:
            raise RuntimeError(
                f"unexpected QAOA parameter set: {list(map(str, parameters))}"
            )

        objective_calls = 0

        def ordered_values(gammas: np.ndarray, betas: np.ndarray) -> list[float]:
            parameter_values = {}
            for layer, parameter in enumerate(gamma_parameters):
                parameter_values[parameter] = float(gammas[layer])
            for layer, parameter in enumerate(beta_parameters):
                parameter_values[parameter] = float(betas[layer])
            return [parameter_values[parameter] for parameter in parameters]

        def energy(theta: np.ndarray) -> float:
            nonlocal objective_calls
            objective_calls += 1
            gammas = theta[:reps]
            betas = theta[reps:]
            pub = (transpiled, operator, ordered_values(gammas, betas))
            estimate = float(np.asarray(estimator.run([pub]).result()[0].data.evs).item())
            return estimate + offset

        rng = np.random.default_rng(seed)
        starts = [
            np.tile(np.array([0.35, 0.25], dtype=float), reps),
            np.tile(np.array([1.1, 0.65], dtype=float), reps),
            np.concatenate(
                (rng.uniform(0.0, np.pi, reps), rng.uniform(0.0, np.pi / 2.0, reps))
            ),
        ]
        best_optimization = None
        total_iterations = 0
        for initial in starts:
            optimized = minimize(
                energy,
                initial,
                method="COBYLA",
                options={"maxiter": maxiter, "rhobeg": 0.5, "tol": 1e-4},
            )
            total_iterations += int(
                getattr(optimized, "nit", getattr(optimized, "nfev", 0))
            )
            if best_optimization is None or optimized.fun < best_optimization.fun:
                best_optimization = optimized

        if best_optimization is None:
            raise RuntimeError("COBYLA did not produce an optimization result")

        gammas = best_optimization.x[:reps]
        betas = best_optimization.x[reps:]
        bind_values = {}
        for layer, parameter in enumerate(gamma_parameters):
            bind_values[parameter] = float(gammas[layer])
        for layer, parameter in enumerate(beta_parameters):
            bind_values[parameter] = float(betas[layer])

        sampling_circuit = transpiled.assign_parameters(bind_values)
        sampling_circuit.measure_all()
        sampling_circuit = transpile(
            sampling_circuit,
            backend=backend,
            seed_transpiler=seed,
        )
        counts = _sample_counts(sampling_circuit, shots=shots, seed=seed)
        summary = aggregate_samples(
            problem, counts, optimal_bitstring=optimal_bitstring
        )
        operation_counts = sampling_circuit.count_ops()
        two_qubit_count = sum(
            count
            for operation, count in operation_counts.items()
            if _operation_arity(sampling_circuit, operation) == 2
        )
        return QAOAResult(
            status=summary.status,
            best_valid=summary.best_valid,
            optimal_params=tuple(float(value) for value in best_optimization.x),
            final_expected_energy=float(best_optimization.fun),
            circuit_depth=sampling_circuit.depth(),
            two_qubit_gate_count=two_qubit_count,
            optimiser_iterations=total_iterations,
            optimiser_evals=objective_calls,
            valid_fraction=summary.valid_fraction,
            prob_of_exact_optimum=summary.prob_of_exact_optimum,
            top_k=summary.top_k,
            counts=counts,
            runtime_seconds=perf_counter() - started,
            **{key: value for key, value in result_base.items() if key not in {
                "circuit_depth", "two_qubit_gate_count",
                "optimiser_iterations", "optimiser_evals", "valid_fraction",
                "prob_of_exact_optimum", "top_k", "counts", "runtime_seconds"
            }},
        )
    except Exception as error:
        return QAOAResult(
            status="error",
            best_valid=None,
            optimal_params=None,
            final_expected_energy=None,
            message=f"{type(error).__name__}: {error}",
            **{**result_base, "runtime_seconds": perf_counter() - started},
        )


def _sample_counts(circuit: QuantumCircuit, shots: int, seed: int) -> dict[str, int]:
    sampler = SamplerV2(default_shots=shots, seed=seed)
    sampled = sampler.run([circuit], shots=shots).result()[0]
    bit_array = getattr(sampled.data, "meas")
    return {str(bitstring): int(count) for bitstring, count in bit_array.get_counts().items()}


def _decode_counts_bitstring(problem: DispatchProblem, bitstring: str) -> list[int]:
    return _decode_bitstring_with_spacing(problem, bitstring)


def _decode_bitstring_with_spacing(
    problem: DispatchProblem, bitstring: str
) -> list[int]:
    return decode_bitstring(problem, bitstring.replace(" ", ""))


def _qubo_energy_for_problem(problem: DispatchProblem, x: list[int]) -> float:
    from qrescue.qubo import qubo_energy

    return qubo_energy(build_qubo(problem), x)


def _operation_arity(circuit: QuantumCircuit, name: str) -> int:
    return next(
        len(instruction.qubits)
        for instruction in circuit.data
        if instruction.operation.name == name
    )
