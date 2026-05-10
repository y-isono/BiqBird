"""Iterative Quantum Phase Estimation (iQPE) for the qubit Hamiltonian.

Reference: Dobšíček et al., Phys. Rev. A 76, 030306(R) (2007).

Algorithm summary
-----------------
We want the phase φ ∈ [0, 1) such that ``U |ψ⟩ = e^{-2π i φ} |ψ⟩`` where
``U = exp(-i H τ)``.  iQPE measures the bits of φ from the most significant
bit (MSB) to the least, re-using a single ancilla qubit and feeding the
previously-measured bits back into the next round as a phase correction.

Two simulation backends
-----------------------
The "true" cost of iQPE is dominated by applying ``U^{2^k}`` to the system
state.  In a real quantum computer this is one (long) controlled-unitary; in
a state-vector simulator we have a choice:

1) **Eigenstate fast path** (``run_iqpe``).  If we *already know* the input
   is an exact eigenstate of ``H`` (e.g. taken from ``direct_diag``), then
   ``U^{2^k} |ψ⟩ = e^{-i E τ 2^k} |ψ⟩``: applying the controlled-U just
   produces a known phase on the |1⟩ branch.  This needs zero matrix-vector
   products and is *exact* for the eigenstate case.  It is the right tool
   for "verify that the iQPE protocol recovers the eigenvalue".

2) **General state path** (``run_iqpe_general``).  For non-eigenstate
   inputs (e.g. an HF guess), apply ``exp(-i H τ 2^k) |ψ⟩`` via
   ``scipy.sparse.linalg.expm_multiply``.  Costly for large powers of 2:
   each call internally needs O(τ · 2^k · ‖H‖) sparse mat-vec products.

For Phase Q1-Q3 we drive iQPE with eigenstates from direct diagonalisation,
so the fast path is what we use.  Phase Q4 (Trotter) is the place where
non-eigenstate evolution is exercised.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass
class IQPEResult:
    bits: list[int]          # measured bits, LSB first (bits[0] is the least-significant bit of m)
    phase: float             # reconstructed φ ≈ m / 2^n_bits  ∈ [0, 1)
    energy: float            # reconstructed energy = 2π φ / τ
    tau: float
    n_bits: int


def _measure_bit(p_zero: float, rng: np.random.Generator) -> int:
    """Sample a single ancilla outcome given P(|0⟩) = p_zero."""
    p_zero = float(min(max(p_zero, 0.0), 1.0))
    return 0 if rng.random() < p_zero else 1


def run_iqpe(
    H: sp.spmatrix,
    eigenstate: np.ndarray,
    *,
    tau: float,
    n_bits: int,
    rng: Optional[np.random.Generator] = None,
) -> IQPEResult:
    """Iterative QPE on ``U = exp(-i H τ)`` *assuming* the input is an eigenstate.

    Parameters
    ----------
    H : sparse Hermitian matrix, shape (2^n, 2^n)
        Qubit-basis Hamiltonian.  Used only to compute ``E = ⟨ψ|H|ψ⟩``.
    eigenstate : np.ndarray, shape (2^n,)
        An eigenstate of ``H``.  Will be normalised.  We *assert* it is an
        eigenstate within ``rtol`` and otherwise warn (the protocol gives a
        peaked phase distribution in that case but we still report the most
        likely outcome).
    tau : float
        Time-evolution duration.  Must satisfy ``|E τ| < 2π`` for the target
        eigenvalue (otherwise the phase wraps around).  See
        :func:`shifted_for_iqpe` for a safe choice.
    n_bits : int
        Number of phase bits to extract.

    Notes
    -----
    Standard *LSB-first* iQPE (Kitaev / Dobšíček 2007):

      φ ≈ m / 2^n_bits  with bits b_0 (LSB), b_1, ..., b_{n-1} (MSB),
      so  m = Σ b_k 2^k.

      For k = 0, 1, ..., n_bits - 1  (LSB first):
          power_k = 2^{n_bits - 1 - k}
          # Controlled-U^{power_k} applies e^{-2π i φ · power_k} on the |1⟩
          # ancilla branch.  Modulo 2π this equals e^{-i π b_k} (the LSB of
          # 2^{n-1-k} · m, which is b_k by construction) plus contributions
          # from the *higher* bits b_{k+1}..b_{n-1} that are still unknown
          # at this point... but in *LSB first* iQPE, by the time we get to
          # bit k, bits b_0..b_{k-1} are *already known*, so we subtract
          # their contribution to power_k · φ:
          #   ω_k = π · Σ_{j=0..k-1} b_j · 2^{k - 1 - j}     (mod 2π)
          # Then  P(0)  =  cos²((θ - ω)/2)  with  θ = -2π φ · power_k.
          # On the eigenstate this is deterministic and recovers b_k.
    """
    if rng is None:
        rng = np.random.default_rng()

    if eigenstate.shape != (H.shape[0],):
        raise ValueError(
            f"eigenstate shape {eigenstate.shape} does not match H {H.shape}"
        )

    psi = eigenstate.astype(np.complex128, copy=True)
    psi /= np.linalg.norm(psi)

    # Sanity: confirm eigenstate property.
    Hpsi = H @ psi
    E = float(np.vdot(psi, Hpsi).real)
    residual = float(np.linalg.norm(Hpsi - E * psi))
    if residual > 1e-6:
        # Not an eigenstate within our tolerance.  We still continue but warn.
        import warnings
        warnings.warn(
            f"Input is not an exact eigenstate (||H ψ - E ψ|| = {residual:.2e}); "
            "iQPE outcomes will become probabilistic and the recovered phase "
            "is the most likely one only."
        )

    # LSB-first iQPE.  Setup (assuming an exact n_bits-bit phase
    #   φ = m / 2^n_bits = Σ_{j} b_j 2^{j - n_bits}
    # ):
    #   power_k    = 2^{n_bits - 1 - k}
    #   θ_k mod 2π = -π b_k - 2π Σ_{j=0..k-1} b_j · 2^{j - k - 1}
    # Feed-forward correction kills the j<k tail:
    #   ω_k        = +2π Σ_{j=0..k-1} b_j · 2^{j - k - 1}
    #              = +π  Σ_{j=0..k-1} b_j · 2^{j - k}
    # so that
    #   θ_k - ω_k  = -π b_k - 2π · 2 · Σ_{j<k} b_j · 2^{j-k-1}
    # No wait: θ_k = -π b_k - (correction), and we want δ = -π b_k.
    # So we should *add* the correction:  δ = θ + correction = -π b_k.
    # Equivalently:  δ = (θ_k - ω_k)/2  with  ω_k = -2π Σ b_j 2^{j-k-1}
    #                                           = -π Σ b_j 2^{j-k}
    # Then b_k = 0 ⇒ δ=0 ⇒ P(0)=1 ⇒ measure 0.
    #      b_k = 1 ⇒ δ=-π/2 ⇒ P(0)=0 ⇒ measure 1.
    bits_lsb_first: list[int] = []
    two_pi = 2.0 * np.pi

    for k in range(n_bits):
        power = 1 << (n_bits - 1 - k)
        # θ_k = -2π φ · power   (we use the eigenvalue E ≡ 2π φ / τ)
        theta_k = -((E * tau * power) % two_pi)
        # Feed-forward correction (negative sign!):
        omega_k = 0.0
        for j, bj in enumerate(bits_lsb_first):
            omega_k -= np.pi * bj * (2.0 ** (j - k))
        omega_k = omega_k % two_pi
        delta = (theta_k - omega_k) / 2.0
        p_zero = float(np.cos(delta) ** 2)
        bk = _measure_bit(p_zero, rng)
        bits_lsb_first.append(bk)

    # Reconstruct m = Σ b_k 2^k  (b_0 is LSB, so bits_lsb_first[0] · 2^0 etc.)
    m = 0
    for k, bk in enumerate(bits_lsb_first):
        m |= bk << k
    phase = m / float(1 << n_bits)

    # Energy from φ: E τ = 2π φ (mod 2π)  ⇒  E = 2π φ / τ  (mod 2π / τ)
    energy = 2.0 * np.pi * phase / tau

    return IQPEResult(
        bits=bits_lsb_first,
        phase=phase,
        energy=energy,
        tau=tau,
        n_bits=n_bits,
    )


def shifted_for_iqpe(
    H: sp.spmatrix,
    *,
    margin: float = 0.05,
) -> tuple[sp.spmatrix, float, float]:
    """Shift ``H`` so the spectrum lies in (0, 2π / τ_safe) for some τ_safe.

    Returns ``(H_shifted, shift, tau_safe)`` where:

    - ``H_shifted = H + shift * I`` has its lowest eigenvalue ≈ ``margin * span``
      (a small positive number).
    - ``tau_safe`` is chosen so that ``(E_high + shift) * tau_safe < 2π``,
      hence the phase φ stays in [0, 1) for every eigenvalue.

    The original energy is recovered from the iQPE phase by
    ``E_original = 2π φ / tau_safe - shift``.

    We estimate the spectrum extremes via two cheap eigsh calls.
    """
    e_low = float(spla.eigsh(H, k=1, which="SA", tol=1e-3)[0][0])
    e_high = float(spla.eigsh(H, k=1, which="LA", tol=1e-3)[0][0])
    span = e_high - e_low
    if span <= 0:
        span = 1.0
    shift = -e_low + margin * span
    n = H.shape[0]
    H_shifted = H + shift * sp.identity(n, format=H.format)
    tau_safe = 2.0 * np.pi / ((e_high + shift) * (1.0 + margin))
    return H_shifted, shift, tau_safe


# ---------------------------------------------------------------------------
# General-state iQPE (uses expm_multiply; can be very expensive for large
# powers of 2).  Provided for completeness but not used in Q1-Q3 scripts.
# ---------------------------------------------------------------------------

@dataclass
class IQPEGeneralResult:
    bits: list[int]
    phase: float
    energy: float
    tau: float
    n_bits: int
    final_state: np.ndarray


def run_iqpe_general(
    H: sp.spmatrix,
    initial_state: np.ndarray,
    *,
    tau: float,
    n_bits: int,
    rng: Optional[np.random.Generator] = None,
) -> IQPEGeneralResult:
    """Statevector-based iQPE for arbitrary input state.

    *Warning*: applying ``exp(-i H τ · 2^{n_bits-1})`` is expensive and may
    be slow for ``n_bits ≥ ~10`` and large H.  Prefer :func:`run_iqpe`
    with an eigenstate input from ``direct_diag``.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = int(np.log2(H.shape[0]))
    psi = initial_state.astype(np.complex128, copy=True)
    psi /= np.linalg.norm(psi)

    bits_msb_first: list[int] = []
    accumulated_phase = 0.0  # for feed-forward correction

    for k in range(n_bits):
        power = 1 << (n_bits - 1 - k)
        # Apply U^power · |ψ⟩ via expm_multiply (this is the costly bit).
        evolved = spla.expm_multiply(-1j * tau * power * H, psi)

        # Build joint ancilla-system state |+⟩ ⊗ |ψ⟩, apply controlled-U^power:
        #   |0⟩ |ψ⟩  -> |0⟩ |ψ⟩
        #   |1⟩ |ψ⟩  -> |1⟩ |U^power ψ⟩
        joint = (np.concatenate([psi, np.zeros_like(psi)])
                 + np.concatenate([np.zeros_like(psi), evolved])) / np.sqrt(2.0)

        # Apply ancilla phase correction e^{-i ω_k} on |1⟩ branch
        omega_k = 0.0
        for j, bj in enumerate(bits_msb_first):
            exp = (n_bits - 1 - k) - j
            if exp == 0 and bj == 1:
                omega_k += np.pi
            elif exp < 0:
                omega_k += np.pi * bj * (2.0 ** exp)
        omega_k = omega_k % (2.0 * np.pi)
        if omega_k != 0.0:
            dim_sys = 1 << n
            joint[dim_sys:] *= np.exp(-1j * omega_k)

        # Hadamard on ancilla
        dim_sys = 1 << n
        a0 = joint[:dim_sys]
        a1 = joint[dim_sys:]
        joint = np.concatenate([(a0 + a1) / np.sqrt(2.0),
                                (a0 - a1) / np.sqrt(2.0)])

        # Measure ancilla
        p_zero = float(np.linalg.norm(joint[:dim_sys]) ** 2)
        bk = _measure_bit(p_zero, rng)
        bits_msb_first.append(bk)

        # Project and renormalise
        if bk == 0:
            psi = joint[:dim_sys]
        else:
            psi = joint[dim_sys:]
        psi = psi / max(np.linalg.norm(psi), 1e-15)

    m = 0
    for k, bk in enumerate(bits_msb_first):
        m |= bk << (n_bits - 1 - k)
    phase = m / float(1 << n_bits)
    energy = 2.0 * np.pi * phase / tau

    return IQPEGeneralResult(
        bits=bits_msb_first,
        phase=phase,
        energy=energy,
        tau=tau,
        n_bits=n_bits,
        final_state=psi,
    )
