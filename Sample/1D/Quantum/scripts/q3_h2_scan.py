"""Phase Q3: H_2 molecule, quantum-computed potential energy scan.

For each R in {1.0, 1.6, 3.0, 6.0} a.u. we read ``data/h2_R<R>.json``,
compute the FCI ground state energy via the sector-direct route, and add
the nuclear-nuclear repulsion ``E_nn`` to obtain the total potential
``E_tot(R) = E_FCI(R) + E_nn(R)``.

We compare against the classical RHF values from the existing
``H2_fedvr_HF.cpp`` (``Lab/BiqBird/Sample/1D/FEDVR/README.md`` §13):

    R     E_RHF [Ha]   E_nn      E_tot_RHF [Ha]
    1.0  -2.10159     0.70711   -1.39448
    1.6  -1.95507     0.53000   -1.42507  (equilibrium)
    3.0  -1.62724     0.31623   -1.31102
    6.0  -1.28493     0.16440   -1.12053

Quantum (FCI) should be **lower** than RHF (variational principle), and
the gap should *grow* as R increases (RHF dissociation pathology vs the
correctly-correlated FCI/CASSCF picture).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from openfermion.linalg import get_number_preserving_sparse_operator

from biqbird_quantum import build_qubit_h, direct_diag, iqpe, load_integrals


CLASSICAL_RHF = {
    "1.0": (-2.10159, 0.70711),
    "1.6": (-1.95507, 0.53000),  # equilibrium
    "3.0": (-1.62724, 0.31623),
    "6.0": (-1.28493, 0.16440),
}


def run_one(tag: str, R_label: str) -> dict:
    here = Path(__file__).resolve().parent.parent
    integ = load_integrals.load(here / "data" / f"{tag}.json")

    fermion_op = build_qubit_h.build_fermion_operator(integ)
    qh = build_qubit_h.QubitHamiltonian(
        n_qubits=integ.n_qubits, fermion_op=fermion_op
    )

    # Direct diag (FCI in DVR basis)
    res = direct_diag.lowest_eigenvalue_sector(qh, N_e=integ.N_e)

    # iQPE on the same sector matrix
    H_sec = get_number_preserving_sparse_operator(
        fermion_op, num_qubits=integ.n_qubits,
        num_electrons=integ.N_e, spin_preserving=False,
    )
    H_sec = (0.5 * (H_sec + H_sec.conj().T)).tocsc()
    H_shift, shift, tau = iqpe.shifted_for_iqpe(H_sec, margin=0.05)
    rng = np.random.default_rng(seed=42)
    iqpe_res = iqpe.run_iqpe(H_shift, res.eigenvector,
                              tau=tau, n_bits=14, rng=rng)
    E_iqpe = iqpe_res.energy - shift

    E_FCI = res.energy
    E_nn = integ.E_nn
    E_tot_quantum = E_FCI + E_nn

    rhf_R, rhf_Enn = CLASSICAL_RHF[R_label]
    E_tot_rhf = rhf_R + rhf_Enn

    return dict(
        R=R_label,
        E_FCI=E_FCI,
        E_iqpe=E_iqpe,
        E_nn=E_nn,
        E_tot_quantum=E_tot_quantum,
        E_RHF=rhf_R,
        E_tot_RHF=E_tot_rhf,
        Ecorr=E_tot_rhf - E_tot_quantum,  # positive = quantum below RHF (good)
        sector_dim=res.sector_indices.size,
    )


def main() -> None:
    print("Phase Q3: H_2 quantum potential energy scan\n")
    print(f"{'R':>5} {'sec_dim':>8} {'E_FCI':>12} {'E_iqpe':>12} {'E_nn':>10} "
          f"{'E_tot_q':>12} {'E_tot_RHF':>12} {'Ecorr':>10}")
    print("-" * 92)

    rows = []
    for R_label in ["1.0", "1.6", "3.0", "6.0"]:
        tag = f"h2_R{R_label.replace('.', '_')}"
        t0 = time.time()
        row = run_one(tag, R_label)
        elapsed = time.time() - t0
        rows.append(row)
        print(f"{row['R']:>5} {row['sector_dim']:>8} {row['E_FCI']:>12.6f} "
              f"{row['E_iqpe']:>12.6f} {row['E_nn']:>10.5f} "
              f"{row['E_tot_quantum']:>12.6f} {row['E_tot_RHF']:>12.6f} "
              f"{row['Ecorr']:>+10.5f}    [{elapsed:.1f}s]")

    print("\n[Interpretation]")
    print("  - Ecorr = E_tot_RHF - E_tot_quantum, positive means quantum (FCI) is lower.")
    print("  - At R = 1.6 (equilibrium) the correlation energy is small.")
    print("  - As R grows, RHF wrongly traps the wavefunction in a delocalised")
    print("    closed-shell ansatz; the FCI/quantum result correctly approaches")
    print("    the dissociation limit  2 E_0(H) = 2 × (-0.66977) ≈ -1.33954 Ha.")
    print("  - Hence Ecorr should *grow* with R, the textbook RHF dissociation")
    print("    pathology cured by FCI / quantum computation.")


if __name__ == "__main__":
    main()
