"""Trotter time evolution under the JW qubit Hamiltonian.

Strategy: split ``H = T + V`` where

- ``T`` collects all *non-diagonal* (in the computational basis) Pauli
  strings of the JW image of the kinetic plus exchange parts;
- ``V`` collects all diagonal terms (Pauli-Z products + identity).

For the second-order Trotter formula::

    U(Δt) ≈ exp(-i T Δt/2) · exp(-i V Δt) · exp(-i T Δt/2)

since ``V`` is diagonal in the computational basis, ``exp(-i V Δt) v`` is a
componentwise multiplication.  ``exp(-i T Δt/2) v`` is computed via
``scipy.sparse.linalg.expm_multiply`` on the sparse non-diagonal block.

This is accurate to ``O(Δt^3)`` per step.  Higher orders (Suzuki) are easy
to add but Q4 only requires that energy is conserved well enough to verify
"the GS we built really is a stationary state".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def split_diag_offdiag(H: sp.spmatrix) -> tuple[sp.spmatrix, np.ndarray]:
    """Return (H_offdiag, V_diag) such that H = H_offdiag + diag(V_diag)."""
    H_csr = H.tocsr().copy()
    diag = np.asarray(H_csr.diagonal()).ravel()
    H_off = H_csr - sp.diags(diag, format="csr")
    H_off.eliminate_zeros()
    return H_off, diag


@dataclass
class TrotterTrace:
    times: np.ndarray
    energies: np.ndarray
    overlaps: np.ndarray  # |<psi(t)|psi(0)>|^2


def trotter2_step(
    psi: np.ndarray,
    H_off: sp.spmatrix,
    V_diag: np.ndarray,
    dt: float,
) -> np.ndarray:
    """One second-order Trotter step under H = H_off + diag(V_diag)."""
    # exp(-i H_off dt/2)
    psi = spla.expm_multiply(-1j * 0.5 * dt * H_off, psi)
    # exp(-i V dt)  (diagonal)
    psi = np.exp(-1j * dt * V_diag) * psi
    # exp(-i H_off dt/2)
    psi = spla.expm_multiply(-1j * 0.5 * dt * H_off, psi)
    return psi


def evolve(
    H: sp.spmatrix,
    psi0: np.ndarray,
    *,
    dt: float,
    n_steps: int,
    sample_every: int = 1,
    callback: Optional[Callable[[int, np.ndarray], None]] = None,
) -> TrotterTrace:
    """Trotterise ``exp(-i H t) |psi0⟩`` for ``t ∈ [0, dt * n_steps]``.

    Records ⟨H⟩ and the overlap ``|⟨ψ₀|ψ(t)⟩|²`` at intervals of
    ``sample_every`` steps (and always at t = 0).
    """
    H_off, V_diag = split_diag_offdiag(H)
    psi = psi0.astype(np.complex128, copy=True)
    psi /= np.linalg.norm(psi)
    psi_init = psi.copy()

    sample_steps = list(range(0, n_steps + 1, sample_every))
    if sample_steps[-1] != n_steps:
        sample_steps.append(n_steps)

    times: list[float] = []
    energies: list[float] = []
    overlaps: list[float] = []

    def _record(step: int) -> None:
        Hpsi = H @ psi
        E = float(np.vdot(psi, Hpsi).real)
        ov = float(abs(np.vdot(psi_init, psi)) ** 2)
        times.append(step * dt)
        energies.append(E)
        overlaps.append(ov)
        if callback is not None:
            callback(step, psi)

    _record(0)
    next_sample_idx = 1
    for step in range(1, n_steps + 1):
        psi = trotter2_step(psi, H_off, V_diag, dt)
        if next_sample_idx < len(sample_steps) and step == sample_steps[next_sample_idx]:
            _record(step)
            next_sample_idx += 1

    return TrotterTrace(
        times=np.asarray(times),
        energies=np.asarray(energies),
        overlaps=np.asarray(overlaps),
    )
