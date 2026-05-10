"""Phase Q2: He atom, generic driver (used for Step 6 / 7 / 8).

Usage::

    uv run python scripts/q2_he.py he_step6      # 18 qubit, target ~ -2.20 Ha
    uv run python scripts/q2_he.py he_step7      # 22 qubit, target ~ -2.234 Ha
    uv run python scripts/q2_he.py he_step8      # 28 qubit, target ~ -2.237 Ha (FCI level)

Reads ``data/<tag>.json`` exported by ``export_integrals.cpp``.

Strategy
--------
For each step we use the **sector-direct** route in
:mod:`biqbird_quantum.direct_diag` to avoid materialising the full
``2^n_qubits × 2^n_qubits`` JW matrix:

  1. Read integrals.
  2. Build the FermionOperator (no JW yet).
  3. Project to the ``N_e``-particle sector via OpenFermion's
     ``get_number_preserving_sparse_operator``: a ``D × D`` sparse with
     ``D = C(2 N_DVR, N_e)``.
  4. Diagonalise that small matrix (this *is* the FCI ground state).
  5. iQPE on the sector-restricted Hamiltonian, using the eigenstate fast
     path -- the iQPE protocol works on any Hilbert space, not specifically
     on a power-of-two register.

Step 8 (28 qubits) memory footprint:
  - sector dim D = C(28, 2) = 378
  - sector matrix     ≈ few MB
  - iQPE state vector ≈ 6 KB (sector-only)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import psutil

from biqbird_quantum import build_qubit_h, direct_diag, iqpe, load_integrals


def _mem_str() -> str:
    return f"{psutil.Process().memory_info().rss / (1024**2):.0f} MB"


def main(tag: str) -> None:
    here = Path(__file__).resolve().parent.parent
    integ = load_integrals.load(here / "data" / f"{tag}.json")
    print(integ.summary())
    print(f"  RSS = {_mem_str()}")

    # ---- 1. Build FermionOperator (no JW materialisation) ----------------
    t0 = time.time()
    fermion_op = build_qubit_h.build_fermion_operator(integ)
    print(f"\n[FermionOperator] n_modes={integ.n_qubits}, "
          f"build time={time.time()-t0:.2f}s, RSS={_mem_str()}")

    # ---- 2. Build a thin QubitHamiltonian *only* if we need JW for IQPE.
    # We bypass the heavy build_qubit_h.build() because for n_qubits >= 22
    # the full 2^n × 2^n Pauli sparse blows up memory.  Instead we'll feed
    # the sector-projected fermion sparse directly into iQPE.
    qh = build_qubit_h.QubitHamiltonian(
        n_qubits=integ.n_qubits,
        fermion_op=fermion_op,
        qubit_op=None,            # not built
        sparse=None,              # not built
        n_terms=0,
    )

    # ---- 3. Direct (sector) diagonalisation ------------------------------
    t0 = time.time()
    res = direct_diag.lowest_eigenvalue_sector(qh, N_e=integ.N_e)
    diag_time = time.time() - t0
    sec = res.sector_indices
    print(f"\n[Sector-direct diagonalisation, N_e={integ.N_e}]")
    print(f"  sector dim = {sec.size}  "
          f"(full Hilbert = 2^{integ.n_qubits} = {1<<integ.n_qubits})")
    print(f"  E0  (FCI in DVR basis) = {res.energy:.10f} Ha")
    print(f"  diag time = {diag_time:.2f} s,  RSS = {_mem_str()}")

    print(f"\n[Reference values from Sample/1D/README.md]")
    print(f"  Octopus FCI (a=1, L=8, dx=0.1) = -2.23825730 Ha")
    print(f"  RHF continuum limit            ≈ -2.22421    Ha")
    print(f"  E_corr (FCI - RHF)             ≈  0.014       Ha")

    # ---- 4. iQPE on the sector-restricted Hamiltonian --------------------
    # Re-use the sector matrix that direct_diag built.  We need a copy here
    # because shifted_for_iqpe wants to inspect spectrum extremes.
    from openfermion.linalg import get_number_preserving_sparse_operator
    H_sec = get_number_preserving_sparse_operator(
        fermion_op,
        num_qubits=integ.n_qubits,
        num_electrons=integ.N_e,
        spin_preserving=False,
    )
    H_sec = (0.5 * (H_sec + H_sec.conj().T)).tocsc()  # hermitise

    H_shift, shift, tau = iqpe.shifted_for_iqpe(H_sec, margin=0.05)

    n_bits = 14
    rng = np.random.default_rng(seed=42)
    t0 = time.time()
    iqpe_res = iqpe.run_iqpe(H_shift, res.eigenvector,
                              tau=tau, n_bits=n_bits, rng=rng)
    iqpe_time = time.time() - t0
    E_iqpe = iqpe_res.energy - shift
    expected_resolution = 2 * np.pi / tau / (1 << n_bits)

    print(f"\n[iQPE on sector matrix]")
    print(f"  shift={shift:.4f}, tau={tau:.4f}, n_bits={n_bits}")
    print(f"  resolution = 2π/(τ·2^n_bits) = {expected_resolution:.3e} Ha")
    print(f"  bits (LSB..) = {iqpe_res.bits}")
    print(f"  phase        = {iqpe_res.phase:.10f}")
    print(f"  E_iqpe       = {E_iqpe:.10f} Ha")
    print(f"  iqpe time    = {iqpe_time:.2f} s,  RSS = {_mem_str()}")

    print(f"\n[Comparison]")
    print(f"  E_direct_diag = {res.energy:.10f} Ha")
    print(f"  E_iqpe        = {E_iqpe:.10f} Ha")
    print(f"  |Δ|           = {abs(E_iqpe - res.energy):.3e} Ha")
    if abs(E_iqpe - res.energy) < 5 * expected_resolution:
        print("  OK: iQPE within 5 resolution units of direct diagonalisation.")
    else:
        print("  WARN: iQPE result outside 5 resolution units.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/q2_he.py <tag>", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])
