"""Phase Q1 Step 3: H atom, 18 qubit (direct diag + iQPE).

JSON: data/h_step3.json (L=10, n_elements=2, n_order=5 -> N_DVR=9, n_qubits=18)

Verifies:
  1. Direct diagonalisation in the N_e = 1 sector reproduces the lowest
     eigenvalue of h_pq (since the 2-electron term contributes zero in the
     1-electron sector).
  2. iQPE on the (already classically prepared) ground eigenstate recovers
     the same energy to within the QPE bit resolution.

The reference values for soft-Coulomb 1D H at the continuum limit are
E_0 ≈ -0.66977 Ha; with N_DVR = 9 and L = 10 we expect ~ -0.668 Ha.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from biqbird_quantum import build_qubit_h, direct_diag, iqpe, load_integrals


def main() -> None:
    here = Path(__file__).resolve().parent.parent
    integ = load_integrals.load(here / "data" / "h_step3.json")
    print(integ.summary())

    # --- 1. Classical reference from the small h_pq matrix -----------------
    h_eigs = np.linalg.eigvalsh(integ.h_pq)
    print(f"\n[Classical reference] lowest eigenvalue of h_pq = {h_eigs[0]:.10f} Ha")
    print(f"                       (continuum-limit value ~ -0.66977 Ha)")

    # --- 2. Build qubit Hamiltonian ----------------------------------------
    t0 = time.time()
    qh = build_qubit_h.build(integ)
    t1 = time.time()
    print(f"\n[Qubit Hamiltonian]")
    print(f"  n_qubits      = {qh.n_qubits}")
    print(f"  n_pauli_terms = {qh.n_terms}")
    print(f"  build time    = {t1 - t0:.2f} s")

    # --- 3. Direct diagonalisation in the 1-electron sector ----------------
    t0 = time.time()
    res = direct_diag.lowest_eigenvalue(qh, N_e=integ.N_e)
    t1 = time.time()
    print(f"\n[Direct diagonalisation in N_e={integ.N_e} sector]")
    print(f"  sector dim = {res.sector_indices.size}  (full Hilbert dim = 2^{qh.n_qubits} = {1<<qh.n_qubits})")
    print(f"  E0         = {res.energy:.10f} Ha")
    print(f"  diag time  = {t1 - t0:.2f} s")

    delta_h = abs(res.energy - h_eigs[0])
    print(f"  |E0 - eig(h_pq)[0]| = {delta_h:.3e}")

    # --- 4. Lift the eigenvector into the full 2^n Hilbert space -----------
    psi_full = np.zeros(1 << qh.n_qubits, dtype=np.complex128)
    psi_full[res.sector_indices] = res.eigenvector
    psi_full /= np.linalg.norm(psi_full)

    # --- 5. iQPE on the qubit Hamiltonian ----------------------------------
    H = qh.build_sparse()
    H_shift, shift, tau = iqpe.shifted_for_iqpe(H, margin=0.05)
    print(f"\n[iQPE setup]")
    print(f"  shift  = {shift:.6f}")
    print(f"  tau    = {tau:.6f}  (==> period 2π/τ = {2*np.pi/tau:.6f} Ha)")
    print(f"  shifted lowest eigenvalue ≈ {res.energy + shift:.6f}")

    n_bits = 12  # resolution = 2π / τ / 2^n_bits in energy units
    rng = np.random.default_rng(seed=42)
    t0 = time.time()
    iqpe_result = iqpe.run_iqpe(
        H_shift, psi_full, tau=tau, n_bits=n_bits, rng=rng
    )
    t1 = time.time()
    E_iqpe = iqpe_result.energy - shift  # undo the shift
    print(f"\n[iQPE]")
    print(f"  n_bits        = {n_bits}")
    print(f"  bits (LSB..)  = {iqpe_result.bits}")
    print(f"  phase         = {iqpe_result.phase:.10f}")
    print(f"  E (raw)       = {iqpe_result.energy:.10f} Ha")
    print(f"  E (unshifted) = {E_iqpe:.10f} Ha")
    print(f"  iqpe time     = {t1 - t0:.2f} s")

    # --- 6. Comparison -----------------------------------------------------
    print(f"\n[Comparison]")
    print(f"  E_classical      = {h_eigs[0]:.10f} Ha")
    print(f"  E_direct_diag    = {res.energy:.10f} Ha")
    print(f"  E_iqpe           = {E_iqpe:.10f} Ha")
    expected_resolution = 2 * np.pi / tau / (1 << n_bits)
    print(f"  iQPE resolution  = 2π/(τ·2^n_bits) = {expected_resolution:.6e} Ha")
    print(f"  |E_iqpe - E_dd|  = {abs(E_iqpe - res.energy):.3e} Ha")

    # iQPE should match direct diag to within a few resolution units
    assert abs(E_iqpe - res.energy) < 5 * expected_resolution, (
        f"iQPE result differs from classical reference by more than 5 resolution units"
    )
    print("\nOK: iQPE recovers the classical eigenvalue within the QPE resolution.")


if __name__ == "__main__":
    main()
