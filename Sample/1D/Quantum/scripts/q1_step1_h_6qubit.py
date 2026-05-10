"""Phase Q1 Step 1: H atom, 6 qubit Hello World.

JSON: data/h_step1.json (L=5, n_elements=1, n_order=4 -> N_DVR=3, n_qubits=6)

Verifies:
  1. C++ FEDVR integrals are correctly read into Python.
  2. JW + ParticleNumberProjection direct diagonalisation reproduces the
     classical 1-electron eigenvalue obtained directly from h_pq (sanity
     for the JW machinery).
  3. (For sanity, since N_DVR=3 is too small for any meaningful physics.)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from biqbird_quantum import build_qubit_h, direct_diag, load_integrals


def main() -> None:
    here = Path(__file__).resolve().parent.parent
    integ = load_integrals.load(here / "data" / "h_step1.json")
    print(integ.summary())
    print(f"  x = {integ.x}")
    print(f"  w = {integ.w}")
    print(f"  E_nn = {integ.E_nn}")

    # Reference: lowest eigenvalue of h_pq (1-electron problem, no e-e term used)
    h_eigs = np.linalg.eigvalsh(integ.h_pq)
    print(f"\n[Classical reference] lowest eigenvalue of h_pq = {h_eigs[0]:.10f} Ha")

    # Build qubit Hamiltonian and project on N_e = 1 sector
    qh = build_qubit_h.build(integ)
    print(f"\n[Qubit Hamiltonian]")
    print(f"  n_qubits = {qh.n_qubits}")
    print(f"  n_pauli_terms = {qh.n_terms}")

    res = direct_diag.lowest_eigenvalue(qh, N_e=integ.N_e)
    print(f"\n[Direct diagonalisation in N_e={integ.N_e} sector]")
    print(f"  sector dim = {res.sector_indices.size}")
    print(f"  E0 (qubit basis) = {res.energy:.10f} Ha")

    # Add nuclear repulsion (zero for single nucleus, present for H2 etc.)
    print(f"\n[Final E_total = E_qubit + E_nn] = {res.energy + integ.E_nn:.10f} Ha")

    # Consistency check: 1-electron sector of JW(H) should == h_pq's eigenvalues
    # (electronic contribution only; in our convention E_nn is added on top)
    delta = abs(res.energy - h_eigs[0])
    print(f"\n[Consistency] |E_qubit - eig(h_pq)[0]| = {delta:.3e}")
    assert delta < 1e-9, "JW + projection should match h_pq diagonalisation for 1 electron"
    print("OK: JW machinery reproduces 1-electron eigenvalue.")


if __name__ == "__main__":
    main()
