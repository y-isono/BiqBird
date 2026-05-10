"""Direct (classical) diagonalisation of the Hamiltonian.

This is the *reference value* against which iQPE is checked.  Two routes:

1. **Full JW route** (``lowest_eigenvalue``).  Builds the full
   ``2^n × 2^n`` JW sparse matrix and projects onto the particle-number
   sector via fancy indexing.  Easy and pedagogical, but the sparse
   build needs ``O(n_pauli_terms × 2^n)`` memory and is impractical
   beyond ``n_qubits ≈ 18``.

2. **Sector-direct route** (``lowest_eigenvalue_sector``).  Uses
   ``openfermion.linalg.get_number_preserving_sparse_operator`` to build the
   ``D × D`` matrix directly on the ``N_e``-particle sector,
   where ``D = C(2 N_DVR, N_e)`` is *much* smaller than ``2^n``.
   This is the only way to handle 22- and 28-qubit He.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from openfermion.linalg import get_number_preserving_sparse_operator

from .build_qubit_h import QubitHamiltonian, particle_number_basis


@dataclass
class DiagResult:
    energy: float
    eigenvector: np.ndarray  # in the (Hilbert-space-aligned) sector basis
    sector_indices: np.ndarray  # indices into the full 2^n_qubits Hilbert space
    n_qubits: int
    N_e: int


def lowest_eigenvalue(
    qh: QubitHamiltonian,
    N_e: int,
    *,
    method: str = "auto",
    tol: float = 1e-10,
) -> DiagResult:
    """Lowest eigenvalue using the full JW sparse and a Hamming-weight projector.

    Best for small systems (n_qubits ≲ 18).
    """
    H = qh.build_sparse()
    sec = particle_number_basis(qh.n_qubits, N_e)
    dim = sec.size

    H_csr = H.tocsr()
    H_sub = H_csr[sec, :][:, sec]

    if method == "auto":
        method = "dense" if dim <= 4096 else "eigsh"

    if method == "dense":
        H_dense = np.asarray(H_sub.todense())
        H_dense = 0.5 * (H_dense + H_dense.conj().T)
        evals, evecs = np.linalg.eigh(H_dense)
        E0 = float(evals[0].real)
        v0 = np.asarray(evecs[:, 0]).ravel()
    elif method == "eigsh":
        H_sub_sym = 0.5 * (H_sub + H_sub.conj().T).tocsc()
        evals, evecs = spla.eigsh(H_sub_sym, k=1, which="SA", tol=tol)
        E0 = float(evals[0].real)
        v0 = np.asarray(evecs[:, 0]).ravel()
    else:
        raise ValueError(f"Unknown method: {method}")

    return DiagResult(
        energy=E0,
        eigenvector=v0,
        sector_indices=sec,
        n_qubits=qh.n_qubits,
        N_e=N_e,
    )


def lowest_eigenvalue_sector(
    qh: QubitHamiltonian,
    N_e: int,
    *,
    method: str = "auto",
    tol: float = 1e-10,
) -> DiagResult:
    """Lowest eigenvalue using the sector-direct route.

    Avoids materialising the full ``2^n × 2^n`` JW matrix; goes straight from
    the FermionOperator to a particle-number-projected sparse matrix.

    The sector basis used by ``get_number_preserving_sparse_operator`` is
    *aligned with* the lexicographic ordering of bit-strings of Hamming weight
    ``N_e`` (which is exactly what :func:`particle_number_basis` returns), so
    the eigenvector index maps trivially back into the full Hilbert space.

    Memory cost: ``O(D^2 × density)`` where ``D = C(2 N_DVR, N_e)``.
    For He at 28 qubits, ``D = 378`` -> trivial.
    """
    sec = particle_number_basis(qh.n_qubits, N_e)
    dim = sec.size

    H_sec = get_number_preserving_sparse_operator(
        qh.fermion_op,
        num_qubits=qh.n_qubits,
        num_electrons=N_e,
        spin_preserving=False,
    )
    if H_sec.shape != (dim, dim):
        raise RuntimeError(
            f"Sector size mismatch: expected {dim}, got {H_sec.shape}.  "
            "Convention mismatch between particle_number_basis and "
            "get_number_preserving_sparse_operator?"
        )

    if method == "auto":
        method = "dense" if dim <= 4096 else "eigsh"

    if method == "dense":
        H_dense = np.asarray(H_sec.todense())
        H_dense = 0.5 * (H_dense + H_dense.conj().T)
        evals, evecs = np.linalg.eigh(H_dense)
        E0 = float(evals[0].real)
        v0 = np.asarray(evecs[:, 0]).ravel()
    elif method == "eigsh":
        H_sym = 0.5 * (H_sec + H_sec.conj().T).tocsc()
        evals, evecs = spla.eigsh(H_sym, k=1, which="SA", tol=tol)
        E0 = float(evals[0].real)
        v0 = np.asarray(evecs[:, 0]).ravel()
    else:
        raise ValueError(f"Unknown method: {method}")

    return DiagResult(
        energy=E0,
        eigenvector=v0,
        sector_indices=sec,
        n_qubits=qh.n_qubits,
        N_e=N_e,
    )


def cross_check(qh: QubitHamiltonian, N_e: int) -> tuple[float, float, float]:
    """Sanity check: full-route and sector-route give the same lowest eigenvalue.

    Only safe to run for small systems (n_qubits ≲ 18).
    """
    a = lowest_eigenvalue(qh, N_e=N_e)
    b = lowest_eigenvalue_sector(qh, N_e=N_e)
    return a.energy, b.energy, abs(a.energy - b.energy)
