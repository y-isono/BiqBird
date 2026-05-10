"""Build the qubit Hamiltonian from FEDVR DVR-basis integrals.

The recipe follows the design in ``Lab/BiqBird_Doc/quantum_computing_plan.md``:
treat each DVR node × spin as a fermionic mode, plug ``h_pq`` and ``V_pq`` into
the second-quantized Hamiltonian (using the DVR contraction
``V_{pqrs} = δ_{pr} δ_{qs} V_{pq}``), and Jordan-Wigner transform.

Spin-orbital ordering convention (matches OpenFermion's molecular convention):

    mode index k = 2 * p + s,   s = 0 for spin-up (alpha), s = 1 for spin-down (beta)

so spin-up and spin-down of the same spatial orbital are adjacent, and
``c_{p, s=0}`` corresponds to ``FermionOperator(((2p, 1), (2p, 0)))`` etc.

Two-electron Hamiltonian in this convention::

    H_2 = (1/2)  Σ_{p,q}  Σ_{s, s'}  V_{pq}  c†_{p,s} c†_{q,s'} c_{q,s'} c_{p,s}

In OpenFermion this turns into ``FermionOperator`` with the four-index sum
collapsed to two indices.  Note that for ``p == q`` and ``s == s'`` the
operator vanishes by Pauli exclusion (``c† c† = 0``), so we skip those terms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp
from openfermion import FermionOperator, get_sparse_operator, jordan_wigner
from openfermion.ops import QubitOperator

from .load_integrals import Integrals


def spin_orbital_index(p: int, s: int) -> int:
    """Map (spatial orbital p, spin s in {0=α, 1=β}) -> JW mode index."""
    return 2 * p + s


def build_fermion_operator(integ: Integrals) -> FermionOperator:
    """Build the second-quantized Hamiltonian as a ``FermionOperator``.

    Notes
    -----
    The DVR basis enables ``V_{pqrs} = δ_{pr} δ_{qs} V_{pq}``, so the full
    quartic two-body sum collapses into a quadratic loop over (p, q):

    .. math::
        H = \\sum_{p,q} \\sum_{s} h_{pq}\\, c^\\dagger_{p,s} c_{q,s}
            + \\tfrac{1}{2} \\sum_{p,q} \\sum_{s, s'} V_{pq}\\,
              c^\\dagger_{p,s} c^\\dagger_{q,s'} c_{q,s'} c_{p,s}

    In OpenFermion's tuple notation the action codes are
    ``1`` for creation and ``0`` for annihilation.
    """
    h = integ.h_pq
    V = integ.V_pq
    N = integ.N_DVR

    H = FermionOperator()

    # --- one-body part -----------------------------------------------------
    for p in range(N):
        for q in range(N):
            hpq = h[p, q]
            if abs(hpq) < 1e-15:
                continue
            for s in (0, 1):
                ip = spin_orbital_index(p, s)
                iq = spin_orbital_index(q, s)
                H += FermionOperator(((ip, 1), (iq, 0)), hpq)

    # --- two-body part (DVR collapsed) ------------------------------------
    # H_2 = 1/2 Σ_{p,q} V_pq Σ_{s, s'} c†_{p,s} c†_{q,s'} c_{q,s'} c_{p,s}
    for p in range(N):
        for q in range(N):
            Vpq = V[p, q]
            if abs(Vpq) < 1e-15:
                continue
            for s in (0, 1):
                for sp_ in (0, 1):
                    if p == q and s == sp_:
                        # Pauli exclusion: c†_{p,s} c†_{p,s} = 0
                        continue
                    ip_s = spin_orbital_index(p, s)
                    iq_sp = spin_orbital_index(q, sp_)
                    H += FermionOperator(
                        (
                            (ip_s, 1),
                            (iq_sp, 1),
                            (iq_sp, 0),
                            (ip_s, 0),
                        ),
                        0.5 * Vpq,
                    )

    return H


def build_qubit_operator(integ: Integrals) -> QubitOperator:
    """Jordan-Wigner transform of the fermionic Hamiltonian (``QubitOperator``)."""
    return jordan_wigner(build_fermion_operator(integ))


@dataclass
class QubitHamiltonian:
    """Bundle of representations of the same Hamiltonian.

    Attributes
    ----------
    n_qubits : int
        Total number of qubits in the spin-orbital register.
    fermion_op : FermionOperator
        Second-quantized form (still in fermion modes).
    qubit_op : QubitOperator
        After Jordan-Wigner.
    sparse : scipy.sparse.csc_matrix
        Full ``2^n_qubits × 2^n_qubits`` sparse matrix in qubit basis.
        Built lazily by :meth:`build_sparse`.
    """

    n_qubits: int
    fermion_op: FermionOperator
    qubit_op: Optional[QubitOperator] = None
    sparse: Optional[sp.spmatrix] = None
    n_terms: int = 0

    def build_sparse(self) -> sp.csc_matrix:
        """Materialise the qubit-basis Hamiltonian as a sparse matrix.

        Memory note: this allocates ``O(n_terms × 2^n_qubits)`` entries during
        accumulation but the final matrix is sparse with ``≤ 2^n_qubits``
        non-zeros per row at worst.  For Phase Q1 (n_qubits ≤ 18) it is cheap.
        """
        if self.sparse is None:
            self.sparse = get_sparse_operator(self.qubit_op, n_qubits=self.n_qubits)
        return self.sparse


def build(integ: Integrals) -> QubitHamiltonian:
    """End-to-end construction: integrals → fermion op → qubit op."""
    fermion_op = build_fermion_operator(integ)
    qubit_op = jordan_wigner(fermion_op)
    n_terms = sum(1 for _ in qubit_op.terms)
    return QubitHamiltonian(
        n_qubits=integ.n_qubits,
        fermion_op=fermion_op,
        qubit_op=qubit_op,
        n_terms=n_terms,
    )


# ---------------------------------------------------------------------------
# Particle-number projector helpers (used by direct_diag for big systems)
# ---------------------------------------------------------------------------

def particle_number_basis(n_qubits: int, N_e: int) -> np.ndarray:
    """Indices ``x ∈ [0, 2^n_qubits)`` whose bit-popcount equals ``N_e``.

    JW maps fermion mode ``k`` to qubit ``k`` with state ``|1⟩`` ↔ occupied.
    The total particle-number operator acts as bit-popcount on computational
    basis states, so the ``N_e``-particle subspace consists exactly of
    bit-strings of Hamming weight ``N_e``.
    """
    if n_qubits > 30:
        raise ValueError(
            f"n_qubits={n_qubits} too large for an explicit particle-number basis"
        )
    full = np.arange(1 << n_qubits, dtype=np.int64)
    # popcount via numpy bitwise tricks
    popcount = np.zeros_like(full)
    tmp = full.copy()
    while tmp.any():
        popcount += (tmp & 1).astype(np.int64)
        tmp >>= 1
    return np.where(popcount == N_e)[0]
