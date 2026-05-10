//
// He_fd_HF_diagnostic.cpp
//
// FD He（Z=2, N_occ=1）に対し、3 つの解法
//   (A) 第一量子化 Hartree 虚時間 RK4    (He_GS.cpp 同等)
//   (B) 第二量子化 RHF 虚時間 RK4         (fedvr::rhf_imag_time)
//   (C) 第二量子化 RHF SCF                (fedvr::rhf_scf)
// を完全に同一条件で走らせ、
//
//   1. 各解で得られた収束軌道 c_A, c_B, c_C を取得
//   2. 各軌道に対し以下 2 つのエネルギー関数を適用してクロス比較：
//        E_H[c]   = compute_energies_hartree(...).Etot
//        E_RHF[c] = fedvr::rhf_energy(h, V, c)
//      → 3 軌道 × 2 関数 = 6 値を出力
//   3. 軌道間の差: ||c_a - c_b||, |<c_a|c_b>|
//
// これにより
//   - "同じ軌道で両関数が一致するか" → 数学的等価性の確認
//   - "3 つの解で軌道が一致するか"   → 収束問題か否か
// を判定する。
//
// Build:
//   EIGEN=/opt/homebrew/include/eigen3
//   g++ -std=c++17 -O2 -I"$EIGEN" -I"../FEDVR" He_fd_HF_diagnostic.cpp -o out/he_fd_diag
//   ./out/he_fd_diag
//

#include "fd_basis.hpp"
#include "hartree_fd.hpp"
#include "../FEDVR/hf_fedvr.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <iomanip>
#include <iostream>

namespace
{

// 第一量子化 Hartree 虚時間 RK4 を完全収束まで回す。
// He_GS.cpp と同じ式：F[c] = T + diag(V_ne) + diag(W[|c|^2])
struct HartreeImagResult
{
    Eigen::VectorXd c;
    double E = 0.0;
    int iterations = 0;
    bool converged = false;
};

HartreeImagResult run_hartree_imag(const fd::FDGrid &grid,
                                   const Eigen::VectorXd &V_ne,
                                   const Eigen::MatrixXd &V_pq,
                                   const Eigen::VectorXd &c_init,
                                   double dtau, int max_itr, double thresh)
{
    Eigen::VectorXd c = c_init;
    c /= c.norm();

    HartreeImagResult res;
    double prev_E = 1e30;
    for (int it = 0; it < max_itr; ++it)
    {
        const auto E = fd::compute_energies_hartree(grid.T, V_ne, V_pq, c);
        const double diff = std::abs(E.Etot - prev_E);
        if (it > 0 && diff < thresh)
        {
            res.c = c;
            res.E = E.Etot;
            res.iterations = it;
            res.converged = true;
            return res;
        }
        prev_E = E.Etot;
        fd::rk4_imag_step_hartree(grid.T, V_ne, V_pq, c, dtau);
    }
    res.c = c;
    res.E = fd::compute_energies_hartree(grid.T, V_ne, V_pq, c).Etot;
    res.iterations = max_itr;
    res.converged = false;
    return res;
}

void print_orbital_diff(const std::string &name_a, const Eigen::VectorXd &a,
                        const std::string &name_b, const Eigen::VectorXd &b)
{
    // 位相のあいまいさ（c と -c は同じ軌道）を吸収
    const double dot = a.dot(b);
    const Eigen::VectorXd b_aligned = (dot >= 0.0) ? b : Eigen::VectorXd(-b);
    const double diff = (a - b_aligned).norm();
    const double overlap = std::abs(dot);
    std::cout << "    ||" << name_a << " - " << name_b << "|| = "
              << diff << ",  |<" << name_a << "|" << name_b << ">| = "
              << overlap << "\n";
}

} // namespace

int main()
{
    // ---- パラメータ（He_GS.cpp / He_fd_HF.cpp と同じ） ----
    const double x_range = 20.0;
    const double dx = 0.4;
    const double Z = 2.0;
    const int N_occ = 1;

    // 全解法で共通のきつい収束条件を使う
    const double dtau = 0.20;
    const int max_itr = 200000;
    const double thresh = 1e-14;

    std::cout << std::scientific << std::setprecision(14);

    // ---- FD グリッド + 積分テンソル ----
    fd::FDGrid grid = fd::make_grid(x_range, dx);
    std::cout << "FD grid: x_range=" << grid.x_range
              << ", dx=" << grid.dx
              << ", N(DOF)=" << grid.N << "\n";

    Eigen::VectorXd V_ne(grid.N);
    for (int p = 0; p < grid.N; ++p)
        V_ne(p) = -Z / std::sqrt(grid.x(p) * grid.x(p) + 1.0);

    Eigen::SparseMatrix<double> h_pq = fd::build_h_pq(grid, Z);
    Eigen::MatrixXd V_pq = fd::build_V_pq(grid);

    // ---- 共通初期推定: ガウシアン ----
    Eigen::VectorXd c_init(grid.N);
    for (int p = 0; p < grid.N; ++p)
    {
        const double psi = std::exp(-grid.x(p) * grid.x(p));
        c_init(p) = std::sqrt(grid.w(p)) * psi;
    }
    c_init /= c_init.norm();

    Eigen::MatrixXd C_init(grid.N, N_occ);
    C_init.col(0) = c_init;

    std::cout << "\nCommon convergence params:"
              << "  dtau=" << dtau
              << ", max_itr=" << max_itr
              << ", thresh=" << thresh << "\n";

    // ============================================================
    // (A) 第一量子化 Hartree 虚時間
    // ============================================================
    std::cout << "\n[A] First-quantization Hartree imag-time RK4\n";
    HartreeImagResult resA = run_hartree_imag(grid, V_ne, V_pq, c_init, dtau, max_itr, thresh);
    std::cout << "    converged=" << (resA.converged ? "yes" : "NO")
              << ", itr=" << resA.iterations
              << ", E=" << resA.E << "\n";

    // ============================================================
    // (B) 第二量子化 RHF 虚時間
    // ============================================================
    std::cout << "\n[B] Second-quantization RHF imag-time RK4\n";
    fedvr::RHF_ImagTimeResult resB = fedvr::rhf_imag_time(
        h_pq, V_pq, N_occ, C_init, dtau, max_itr, thresh);
    std::cout << "    converged=" << (resB.converged ? "yes" : "NO")
              << ", itr=" << resB.iterations
              << ", E=" << resB.E_RHF << "\n";

    // ============================================================
    // (C) 第二量子化 RHF SCF
    // ============================================================
    std::cout << "\n[C] Second-quantization RHF Roothaan-Hall SCF\n";
    fedvr::RHF_SCFResult resC = fedvr::rhf_scf(
        h_pq, V_pq, N_occ, C_init, /*max_itr_scf*/ 1000, thresh);
    std::cout << "    converged=" << (resC.converged ? "yes" : "NO")
              << ", itr=" << resC.iterations
              << ", E=" << resC.E_RHF << "\n";

    // ---- 軌道を取り出す ----
    Eigen::VectorXd cA = resA.c;
    Eigen::VectorXd cB = resB.C_occ.col(0);
    Eigen::VectorXd cC = resC.C_occ.col(0);

    // 位相を A に揃える（c と -c は同じ軌道）
    if (cA.dot(cB) < 0.0) cB = -cB;
    if (cA.dot(cC) < 0.0) cC = -cC;

    // ============================================================
    // クロスエネルギー評価： 3 軌道 × 2 関数
    // ============================================================
    std::cout << "\n[Cross-energy table]  rows = orbital, cols = energy functional\n";
    std::cout << "                          E_Hartree (compute_energies_hartree)"
              << "                  E_RHF (fedvr::rhf_energy)\n";

    auto evalH = [&](const Eigen::VectorXd &c) {
        return fd::compute_energies_hartree(grid.T, V_ne, V_pq, c).Etot;
    };
    auto evalRHF = [&](const Eigen::VectorXd &c) {
        Eigen::MatrixXd C(grid.N, 1);
        C.col(0) = c;
        return fedvr::rhf_energy(h_pq, V_pq, C);
    };

    const double E_HA = evalH(cA);
    const double E_RA = evalRHF(cA);
    const double E_HB = evalH(cB);
    const double E_RB = evalRHF(cB);
    const double E_HC = evalH(cC);
    const double E_RC = evalRHF(cC);

    std::cout << "  c_A (Hartree imag)    " << E_HA << "    " << E_RA
              << "    diff=" << (E_RA - E_HA) << "\n";
    std::cout << "  c_B (RHF imag)        " << E_HB << "    " << E_RB
              << "    diff=" << (E_RB - E_HB) << "\n";
    std::cout << "  c_C (RHF SCF)         " << E_HC << "    " << E_RC
              << "    diff=" << (E_RC - E_HC) << "\n";

    // ============================================================
    // 軌道間の差
    // ============================================================
    std::cout << "\n[Orbital differences]\n";
    print_orbital_diff("c_A", cA, "c_B", cB);
    print_orbital_diff("c_A", cA, "c_C", cC);
    print_orbital_diff("c_B", cB, "c_C", cC);

    // ============================================================
    // 軌道のノルムと点値の典型値（ものさしの確認）
    // ============================================================
    std::cout << "\n[Orbital norms / max abs]\n";
    std::cout << "    ||c_A|| = " << cA.norm() << ",  max|c_A| = " << cA.cwiseAbs().maxCoeff() << "\n";
    std::cout << "    ||c_B|| = " << cB.norm() << ",  max|c_B| = " << cB.cwiseAbs().maxCoeff() << "\n";
    std::cout << "    ||c_C|| = " << cC.norm() << ",  max|c_C| = " << cC.cwiseAbs().maxCoeff() << "\n";

    // ============================================================
    // 結論ヒント
    // ============================================================
    std::cout << "\n[Diagnostic summary]\n";
    std::cout << "  E_R(c_A) - E_H(c_A) = " << (E_RA - E_HA)
              << "    (mathematical equivalence test on c_A)\n";
    std::cout << "  E_R(c_B) - E_H(c_B) = " << (E_RB - E_HB)
              << "    (mathematical equivalence test on c_B)\n";
    std::cout << "  E_R(c_C) - E_H(c_C) = " << (E_RC - E_HC)
              << "    (mathematical equivalence test on c_C)\n";
    std::cout << "  E_RHF(scf) - E_RHF(imag) on their own orbitals: "
              << (resC.E_RHF - resB.E_RHF) << "\n";
    std::cout << "  E_RHF(scf) - E_Hartree(imag) on their own orbitals: "
              << (resC.E_RHF - resA.E) << "\n";

    return 0;
}
