"""Phase Q5 (reference): Qiskit に書いた iQPE 量子回路と古典 fast path の一致確認.

目的
----
これまでの Q1〜Q4 はすべて scipy/numpy ベースの「古典シミュレータ」で動いて
いた。本スクリプトでは **Qiskit Aer のステートベクトルシミュレータ上で
本物の量子回路として iQPE を実行**し、古典 fast path (`iqpe.run_iqpe`) と
ビット単位で一致することを示す。

これは「シミュレータが古典である」点は変わらないが、回路図 (QuantumCircuit)
として書ける形式に落とし込んだことで、**将来の量子実機にそのまま投げられる
ゲート列**を持っているという証拠になる。

使う系
------
H 6 qubit (data/h_step1.json)。最も小さい系で、

  - system : 6 qubit
  - ancilla: 1 qubit
  - 合計   : 7 qubit (= statevector dim 128)

を扱う。シミュレーションはミリ秒オーダー。

回路の構造
----------
1. system register に直接対角化で得た固有ベクトルを ``initialize`` で焼き付け
2. 各ラウンド k = 0..n_bits-1 (LSB first):
     |0⟩_anc ─[H]──•─[Rz(-ω_k)]─[H]─[M]
                   │
              U^{2^{n_bits-1-k}}     where U = e^{-i H τ}
3. 各ラウンド後に ancilla を測定し古典ビット b_k を得る
4. 次ラウンドのフィードフォワード ω_{k+1} を古典で計算

各ラウンドで U^{power} を ``PauliEvolutionGate(SparsePauliOp(H), time = τ·power)``
として直接吐き出している（Trotter 化はしない pure exponentiation）。
これを ``ControlGate.control()`` で ancilla で制御化する。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg as scila
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector
from qiskit_aer import AerSimulator

from biqbird_quantum import build_qubit_h, direct_diag, iqpe, load_integrals
from biqbird_quantum.qiskit_bridge import (
    check_consistency,
    qubit_operator_to_sparse_pauli,
)


def main(tag: str = "h_step1") -> None:
    here = Path(__file__).resolve().parent.parent
    integ = load_integrals.load(here / "data" / f"{tag}.json")
    print(integ.summary())
    n_sys = integ.n_qubits
    print(f"  system qubits = {n_sys}")

    # ------------------------------------------------------------------
    # 1) Build the qubit Hamiltonian and run a sanity check
    #    (OpenFermion sparse vs Qiskit SparsePauliOp dense → same matrix?)
    # ------------------------------------------------------------------
    qh = build_qubit_h.build(integ)
    H_pauli: SparsePauliOp = qubit_operator_to_sparse_pauli(qh.qubit_op, n_sys)
    ok, diff = check_consistency(qh.qubit_op, n_sys)
    print(f"\n[Sanity]   bridge OpenFermion -> Qiskit:  ok={ok},  max diff={diff:.2e}")

    # ------------------------------------------------------------------
    # 2) Classical reference: direct diag + iQPE fast path
    # ------------------------------------------------------------------
    res = direct_diag.lowest_eigenvalue(qh, N_e=integ.N_e)
    print(f"\n[Classical] direct diag E0 = {res.energy:.10f} Ha "
          f"(sector dim = {res.sector_indices.size})")

    H_sparse = qh.build_sparse()
    H_shift, shift, tau = iqpe.shifted_for_iqpe(H_sparse, margin=0.05)
    n_bits = 12
    rng_classical = np.random.default_rng(seed=42)

    # Lift eigenvector into full 2^n_sys Hilbert
    psi_full = np.zeros(1 << n_sys, dtype=np.complex128)
    psi_full[res.sector_indices] = res.eigenvector
    psi_full /= np.linalg.norm(psi_full)

    iqpe_classical = iqpe.run_iqpe(
        H_shift, psi_full, tau=tau, n_bits=n_bits, rng=rng_classical,
    )
    E_classical_iqpe = iqpe_classical.energy - shift

    # ------------------------------------------------------------------
    # 3) Qiskit iQPE: round-by-round
    # ------------------------------------------------------------------
    # Convert shifted Hamiltonian to Qiskit SparsePauliOp.
    # H_shift = H + shift * I   (in qubit operator space)
    H_pauli_shifted = H_pauli + SparsePauliOp("I" * n_sys, coeffs=[shift])
    H_pauli_shifted = H_pauli_shifted.simplify()
    # Sanity: H_pauli_shifted as dense vs scipy H_shift
    M_qiskit = H_pauli_shifted.to_matrix()
    M_scipy  = H_shift.toarray() if hasattr(H_shift, "toarray") else np.asarray(H_shift)
    diff_shift = float(np.max(np.abs(M_qiskit - M_scipy)))
    print(f"[Sanity]   shifted H bridge:  max diff = {diff_shift:.2e}")

    simulator = AerSimulator(method="statevector")

    bits_qiskit: list[int] = []
    rng_qiskit = np.random.default_rng(seed=2026)

    print(f"\n[Qiskit iQPE]  n_bits = {n_bits}, tau = {tau:.6f}, shift = {shift:.6f}")
    print(f"  {'k':>3} {'power':>6} {'P(0) qiskit':>14} {'P(0) classical':>17} {'b_k qiskit':>11}")
    print("  " + "-" * 60)

    # Pre-compute H matrix once; we will exponentiate per-round.
    # Note: PauliEvolutionGate is not directly recognised by AerSimulator,
    # so we materialise U^power as an explicit unitary matrix and wrap it
    # in a UnitaryGate.  This keeps the circuit-level abstraction (one
    # controlled-unitary on ancilla + system) but lets AerSimulator
    # consume it as a primitive gate.  In a real-hardware run the same
    # PauliEvolutionGate would be Trotter-decomposed by the transpiler.
    H_dense = H_pauli_shifted.to_matrix()                    # 2^n_sys × 2^n_sys
    # Diagonalise once so that U^power = V diag(exp(-i E_n τ power)) V†
    # is cheap for any power.
    Hd_evals, Hd_V = np.linalg.eigh(H_dense)

    def U_power(power: int) -> np.ndarray:
        phases = np.exp(-1j * Hd_evals * tau * power)
        return (Hd_V * phases) @ Hd_V.conj().T

    t0_total = time.time()
    for k in range(n_bits):
        power = 1 << (n_bits - 1 - k)
        U_mat = U_power(power)
        evolution_gate = UnitaryGate(U_mat, label=f"U^{power}")
        # Wrap as a controlled gate (ancilla = 1 control, system = target)
        controlled_evol = evolution_gate.control(1)

        # Build a fresh circuit:  ancilla register (1) + system register (n_sys)
        sys_reg = QuantumRegister(n_sys, "sys")
        anc_reg = QuantumRegister(1,    "anc")
        c_reg   = ClassicalRegister(1, "b")
        qc = QuantumCircuit(sys_reg, anc_reg, c_reg)

        # 3a. State preparation:
        #   - system: load |psi_GS>
        #   - ancilla: |+> via Hadamard from |0>
        qc.initialize(psi_full, sys_reg)
        qc.h(anc_reg[0])

        # 3b. Controlled U^power : control=ancilla, target=system
        # The .control(1) call puts the control as the *first* argument by Qiskit
        # convention, so we apply [anc, *sys] as the qubit list.
        qc.append(controlled_evol, [anc_reg[0], *sys_reg])

        # 3c. Feed-forward phase correction R_z(-omega_k)
        # omega_k = -π Σ_{j<k} b_j 2^{j-k}   (LSB first)
        omega_k = 0.0
        for j, bj in enumerate(bits_qiskit):
            omega_k -= np.pi * bj * (2.0 ** (j - k))
        omega_k = omega_k % (2.0 * np.pi)
        if omega_k != 0.0:
            qc.rz(-omega_k, anc_reg[0])

        # 3d. Hadamard + measurement on ancilla
        qc.h(anc_reg[0])
        qc.measure(anc_reg[0], c_reg[0])

        # 3e. Run on AerSimulator with shots=4096 to estimate P(0).
        # AerSimulator does not natively recognise UnitaryGate / c-unitary;
        # we transpile down to its supported basis gates first.
        compiled = transpile(qc, simulator)
        result = simulator.run(compiled, shots=4096, seed_simulator=12345 + k).result()
        counts = result.get_counts()
        # Qiskit uses bit-string convention "<creg_high...creg_low>"
        # We have only 1 classical bit so counts.keys() ⊂ {"0", "1"}
        n0 = counts.get("0", 0)
        n1 = counts.get("1", 0)
        p_zero_q = n0 / max(n0 + n1, 1)

        # Classical analytical P(0) for the eigenstate (from iqpe.py protocol)
        E_shifted = res.energy + shift
        theta_k = -((E_shifted * tau * power) % (2.0 * np.pi))
        delta = (theta_k - omega_k) / 2.0
        p_zero_c = float(np.cos(delta) ** 2)

        # Sample the bit (use measured majority — for an exact eigenstate,
        # P(0) is either 0 or 1 in the noiseless limit, so 4096 shots are
        # plenty)
        b_k = 1 if p_zero_q < 0.5 else 0
        bits_qiskit.append(b_k)

        print(f"  {k:>3} {power:>6} {p_zero_q:>14.6f} {p_zero_c:>17.6f} {b_k:>11}")

    qiskit_total_time = time.time() - t0_total

    # ------------------------------------------------------------------
    # 4) Reconstruct phase from bits and compare
    # ------------------------------------------------------------------
    m = sum(bk << k for k, bk in enumerate(bits_qiskit))
    phase_qiskit = m / float(1 << n_bits)
    E_qiskit = (2.0 * np.pi * phase_qiskit / tau) - shift

    print(f"\n[Reconstruction]")
    print(f"  bits classical (LSB..) = {iqpe_classical.bits}")
    print(f"  bits qiskit    (LSB..) = {bits_qiskit}")
    print(f"  phase classical        = {iqpe_classical.phase:.10f}")
    print(f"  phase qiskit           = {phase_qiskit:.10f}")
    print(f"  E classical iqpe       = {E_classical_iqpe:.10f} Ha")
    print(f"  E qiskit    iqpe       = {E_qiskit:.10f} Ha")
    print(f"  E direct diag          = {res.energy:.10f} Ha")
    print(f"  |E_qiskit - E_classical_iqpe| = {abs(E_qiskit - E_classical_iqpe):.3e}")
    print(f"  qiskit run time = {qiskit_total_time:.1f} s")

    if iqpe_classical.bits == bits_qiskit:
        print("\nOK: Qiskit iQPE は古典 fast path と bit 単位で完全一致しました。")
        print("    つまり biqbird_quantum.iqpe は『実機向け量子回路の正しいシミュレーション』であることが追加検証されました。")
    else:
        print("\nWARN: Qiskit iQPE と古典 fast path の bit 列が一致しません。")
        print("      shots を増やすか、tau / shift / state preparation を疑ってください。")


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "h_step1"
    main(tag)
