from __future__ import annotations

from qiskit.quantum_info import SparsePauliOp

from qrescue.qubo import QUBO


def qubo_to_ising(qubo: QUBO, n: int) -> tuple[SparsePauliOp, float]:
    if n != qubo.n:
        raise ValueError(f"n={n} does not match QUBO variable count {qubo.n}")

    z_coefficients = {k: -value / 2.0 for k, value in qubo.linear.items()}
    for (k, l), coefficient in qubo.quadratic.items():
        z_coefficients[k] -= coefficient / 4.0
        z_coefficients[l] -= coefficient / 4.0

    offset = qubo.constant
    offset += sum(value / 2.0 for value in qubo.linear.values())
    offset += sum(value / 4.0 for value in qubo.quadratic.values())

    terms: list[tuple[str, float]] = []
    for k, coefficient in z_coefficients.items():
        if coefficient:
            label = ["I"] * n
            label[n - 1 - k] = "Z"
            terms.append(("".join(label), float(coefficient)))
    for (k, l), coefficient in qubo.quadratic.items():
        if coefficient:
            label = ["I"] * n
            label[n - 1 - k] = "Z"
            label[n - 1 - l] = "Z"
            terms.append(("".join(label), float(coefficient / 4.0)))

    if not terms:
        terms.append(("I" * n, 0.0))
    return SparsePauliOp.from_list(terms).simplify(), float(offset)
