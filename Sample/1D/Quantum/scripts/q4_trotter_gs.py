"""Phase Q4: Trotter time evolution under the (sector-restricted) qubit Hamiltonian.

We take the He ground state computed in Phase Q2 (sector matrix from
``data/he_step6.json``, n_qubits = 18, sector dim = 153) and propagate it
under the second-order Trotter scheme

    U(Δt) ≈ exp(-i T Δt/2) · exp(-i V Δt) · exp(-i T Δt/2)

with H = T (off-diagonal) + V (diagonal).  Since |ψ_GS⟩ is an eigenstate,
the *exact* time evolution should give just a global phase; the Trotter
splitting introduces ``O(Δt^3)`` per step error in the energy.

We monitor:
  1. Energy expectation ⟨ψ(t)|H|ψ(t)⟩  (should stay constant up to O(Δt^2))
  2. Overlap |⟨ψ(0)|ψ(t)⟩|²        (drifts off because of the global phase
                                      *plus* a Trotter error contribution)

The reduction at large dt confirms the Q4 deliverable:
  "the Trotterised time-evolution conserves energy on the ground state".
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from openfermion.linalg import get_number_preserving_sparse_operator

from biqbird_quantum import build_qubit_h, direct_diag, load_integrals, trotter


def main(tag: str = "he_step6") -> None:
    here = Path(__file__).resolve().parent.parent
    integ = load_integrals.load(here / "data" / f"{tag}.json")
    print(integ.summary())

    # Build sector-restricted Hamiltonian
    fermion_op = build_qubit_h.build_fermion_operator(integ)
    qh = build_qubit_h.QubitHamiltonian(
        n_qubits=integ.n_qubits, fermion_op=fermion_op
    )
    res = direct_diag.lowest_eigenvalue_sector(qh, N_e=integ.N_e)
    print(f"\n[Direct diag] E_GS = {res.energy:.10f} Ha (sector dim = {res.sector_indices.size})")

    H_sec = get_number_preserving_sparse_operator(
        fermion_op, num_qubits=integ.n_qubits,
        num_electrons=integ.N_e, spin_preserving=False,
    )
    H_sec = (0.5 * (H_sec + H_sec.conj().T)).tocsc()

    # Sanity: ⟨ψ_GS | H | ψ_GS⟩ should match res.energy
    psi0 = res.eigenvector.astype(np.complex128)
    psi0 /= np.linalg.norm(psi0)
    E0_check = float(np.vdot(psi0, H_sec @ psi0).real)
    print(f"[Sanity]    ⟨ψ_GS|H|ψ_GS⟩  = {E0_check:.10f} Ha   "
          f"(diff = {abs(E0_check - res.energy):.2e})")

    # ---- Trotter scan -----------------------------------------------------
    # We sweep over time-step sizes; for each Δt, propagate to a fixed total
    # time T and report the maximum energy excursion over the trajectory.
    T_total = 5.0
    print(f"\n[Trotter scan]  total time T = {T_total} a.u.")
    print(f"  {'Δt':>6} {'n_steps':>8} {'max ΔE':>14} {'final overlap':>15} {'time':>8}")
    print("  " + "-" * 55)

    for dt in [0.5, 0.2, 0.1, 0.05, 0.02]:
        n_steps = int(round(T_total / dt))
        sample_every = max(1, n_steps // 50)
        t0 = time.time()
        trace = trotter.evolve(
            H_sec, psi0, dt=dt, n_steps=n_steps, sample_every=sample_every,
        )
        elapsed = time.time() - t0
        max_dE = float(np.max(np.abs(trace.energies - res.energy)))
        final_overlap = float(trace.overlaps[-1])
        print(f"  {dt:>6.3f} {n_steps:>8} {max_dE:>14.3e} {final_overlap:>15.6f} {elapsed:>7.1f}s")

    print("\n[Interpretation]")
    print("  - The 2nd-order Trotter splitting has local error O(Δt^3), so")
    print("    the drift in ⟨H⟩ over the trajectory should scale ~ Δt^2.")
    print("  - 'final overlap' is |⟨ψ(0)|ψ(T)⟩|^2 ≠ 1 due to the global phase")
    print("    e^{-i E_GS T}, but |overlap| = 1 *would* hold for the exact")
    print("    propagator -- the deviation here reflects Trotter error only.")


if __name__ == "__main__":
    import sys
    tag = sys.argv[1] if len(sys.argv) > 1 else "he_step6"
    main(tag)
