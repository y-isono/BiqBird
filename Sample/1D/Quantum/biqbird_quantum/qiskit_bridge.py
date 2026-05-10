"""Bridge between OpenFermion ``QubitOperator`` and Qiskit ``SparsePauliOp``.

OpenFermion stores a Pauli operator as a dict
    { ((qubit_index, 'X'|'Y'|'Z'), ...): complex_coefficient, ... }

Qiskit's ``SparsePauliOp`` stores it as
    SparsePauliOp(["IIIXY", "ZZZZI", ...], coeffs=[c0, c1, ...])

where each Pauli string is left-padded with 'I' so its length equals
``n_qubits``.  Note that **Qiskit uses little-endian convention**: the
*right-most* character corresponds to qubit 0.

We convert by writing each OpenFermion term as a list of 'I' / 'X' / 'Y' / 'Z'
of length ``n_qubits`` (qubit 0 at index 0), then *reversing* the string when
emitting the Qiskit label.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from openfermion.ops import QubitOperator
from qiskit.quantum_info import SparsePauliOp


def qubit_operator_to_sparse_pauli(
    qubit_op: QubitOperator,
    n_qubits: int,
) -> SparsePauliOp:
    """Convert ``QubitOperator`` -> ``SparsePauliOp`` (Qiskit little-endian).

    The all-identity term (constant offset) is included.
    Real coefficients are preserved as complex with zero imaginary part.
    """
    labels: list[str] = []
    coeffs: list[complex] = []

    for term, coeff in qubit_op.terms.items():
        # Build a list of 'I' for each qubit, then overwrite by the term spec.
        ops = ["I"] * n_qubits
        for q_idx, p_letter in term:
            if not (0 <= q_idx < n_qubits):
                raise ValueError(
                    f"qubit index {q_idx} out of range for n_qubits={n_qubits}"
                )
            ops[q_idx] = p_letter
        # Qiskit label is little-endian: right-most char = qubit 0
        label = "".join(reversed(ops))
        labels.append(label)
        coeffs.append(complex(coeff))

    if not labels:
        # Degenerate case: zero operator
        labels = ["I" * n_qubits]
        coeffs = [0.0]

    return SparsePauliOp(labels, coeffs=np.asarray(coeffs))


def sparse_pauli_to_dense(sp_op: SparsePauliOp) -> np.ndarray:
    """Materialise the Pauli operator as a dense ``2^n × 2^n`` matrix.

    Useful for small-system sanity checks against the
    OpenFermion-based ``get_sparse_operator``.
    """
    return sp_op.to_matrix()


def check_consistency(
    qubit_op: QubitOperator,
    n_qubits: int,
    *,
    tol: float = 1e-10,
) -> tuple[bool, float]:
    """Cross-check by building both representations and comparing dense matrices.

    Returns ``(ok, max_abs_diff)``.
    Only safe for small ``n_qubits`` (say, ≤ 10).
    """
    from openfermion.linalg import get_sparse_operator

    sp_op = qubit_operator_to_sparse_pauli(qubit_op, n_qubits)
    M_qiskit = sparse_pauli_to_dense(sp_op)

    M_of = get_sparse_operator(qubit_op, n_qubits=n_qubits).toarray()

    diff = np.max(np.abs(M_qiskit - M_of))
    return bool(diff < tol), float(diff)
