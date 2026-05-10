//
// He_fd_HF_imag_robust.cpp
//
// Phase 2: 虚時間 RK4 の収束パラメータを変えて、
// 「FD で虚時間 RK4 が SCF 解 (E=-2.22894) に到達できるか？」を検証する。
//
// 比較対象：fedvr::rhf_imag_time （exchange 込みの第二量子化版）
//
// 実験：
//   (1) 異なる dtau (0.20, 0.10, 0.05, 0.01) で十分長く（max_itr 大きく）回す
//   (2) より厳しい thresh (1e-14, 1e-16) を試す
//   (3) 最後の判定だけでなく、最終軌道の SCF 解との距離を出す
//   (4) 初期推定として SCF の解を入れたら自動的にそこに止まるか
//   (5) "ガウシアンより SCF 解に近い" 初期推定（少し非対称・wider）を入れる
//
// Build:
//   EIGEN=/opt/homebrew/include/eigen3
//   g++ -std=c++17 -O2 -I"$EIGEN" -I"../FEDVR" He_fd_HF_imag_robust.cpp -o out/he_fd_imag_robust
//   ./out/he_fd_imag_robust
//

#include "fd_basis.hpp"
#include "../FEDVR/hf_fedvr.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

namespace
{

// 軌道間距離（位相を揃えた上で）
double orbital_dist(const Eigen::VectorXd &a, const Eigen::VectorXd &b)
{
    const double dot = a.dot(b);
    Eigen::VectorXd b2 = (dot >= 0.0) ? b : Eigen::VectorXd(-b);
    return (a - b2).norm();
}

} // namespace

int main()
{
    const double x_range = 20.0;
    const double dx = 0.4;
    const double Z = 2.0;
    const int N_occ = 1;

    std::cout << std::scientific << std::setprecision(14);

    fd::FDGrid grid = fd::make_grid(x_range, dx);
    Eigen::SparseMatrix<double> h_pq = fd::build_h_pq(grid, Z);
    Eigen::MatrixXd V_pq = fd::build_V_pq(grid);

    // ガウシアン初期
    Eigen::MatrixXd C_init_gauss(grid.N, N_occ);
    for (int p = 0; p < grid.N; ++p)
    {
        const double psi = std::exp(-grid.x(p) * grid.x(p));
        C_init_gauss(p, 0) = std::sqrt(grid.w(p)) * psi;
    }
    C_init_gauss.col(0) /= C_init_gauss.col(0).norm();

    // ----- 基準解（SCF）を取得 -----
    fedvr::RHF_SCFResult ref = fedvr::rhf_scf(h_pq, V_pq, N_occ, C_init_gauss, 1000, 1e-14);
    std::cout << "[Reference SCF]  itr=" << ref.iterations
              << ", E_RHF=" << ref.E_RHF << "\n";
    Eigen::VectorXd c_ref = ref.C_occ.col(0);

    // ============================================================
    // Experiment 1: 異なる dtau でガウシアン初期から虚時間
    // ============================================================
    std::cout << "\n[Experiment 1] dtau scan from Gaussian init\n";
    std::cout << std::setw(8) << "dtau"
              << std::setw(12) << "max_itr"
              << std::setw(24) << "E_imag"
              << std::setw(24) << "E_imag - E_scf"
              << std::setw(20) << "||c-c_scf||"
              << std::setw(8) << "conv"
              << std::setw(10) << "itr_used\n";

    const std::vector<double> dtaus = {0.20, 0.10, 0.05, 0.01};
    const int huge_itr = 500000;
    const double tight_thresh = 1e-14;

    for (double dtau : dtaus)
    {
        fedvr::RHF_ImagTimeResult r = fedvr::rhf_imag_time(
            h_pq, V_pq, N_occ, C_init_gauss, dtau, huge_itr, tight_thresh);
        const double dist = orbital_dist(r.C_occ.col(0), c_ref);
        std::cout << std::setw(8) << dtau
                  << std::setw(12) << huge_itr
                  << std::setw(24) << r.E_RHF
                  << std::setw(24) << (r.E_RHF - ref.E_RHF)
                  << std::setw(20) << dist
                  << std::setw(8) << (r.converged ? "yes" : "NO")
                  << std::setw(10) << r.iterations << "\n";
    }

    // ============================================================
    // Experiment 2: SCF の解を初期にして虚時間を回す
    // → ちゃんと SCF 解に静止していられるか？
    // ============================================================
    std::cout << "\n[Experiment 2] imag-time starting from SCF solution itself\n";
    {
        Eigen::MatrixXd C_init_scf(grid.N, N_occ);
        C_init_scf.col(0) = c_ref;
        const double dtau = 0.05;
        fedvr::RHF_ImagTimeResult r = fedvr::rhf_imag_time(
            h_pq, V_pq, N_occ, C_init_scf, dtau, huge_itr, tight_thresh);
        const double dist = orbital_dist(r.C_occ.col(0), c_ref);
        std::cout << "    dtau=" << dtau
                  << ", E_imag=" << r.E_RHF
                  << ", E_imag-E_scf=" << (r.E_RHF - ref.E_RHF)
                  << ", ||c-c_scf||=" << dist
                  << ", conv=" << (r.converged ? "yes" : "NO")
                  << ", itr=" << r.iterations << "\n";
    }

    // ============================================================
    // Experiment 3: いろいろな初期推定からの虚時間収束先
    // ============================================================
    std::cout << "\n[Experiment 3] imag-time from various initial guesses (dtau=0.05)\n";
    auto run_from_init = [&](const std::string &name, const Eigen::VectorXd &init0) {
        Eigen::MatrixXd C(grid.N, N_occ);
        C.col(0) = init0 / init0.norm();
        fedvr::RHF_ImagTimeResult r = fedvr::rhf_imag_time(
            h_pq, V_pq, N_occ, C, 0.05, huge_itr, tight_thresh);
        const double dist = orbital_dist(r.C_occ.col(0), c_ref);
        std::cout << "  " << std::setw(28) << std::left << name << std::right
                  << "  E=" << r.E_RHF
                  << ", E-E_scf=" << (r.E_RHF - ref.E_RHF)
                  << ", ||c-c_scf||=" << dist
                  << ", itr=" << r.iterations << "\n";
    };

    // 3a: ガウシアン (alpha=1.0)
    {
        Eigen::VectorXd c(grid.N);
        for (int p = 0; p < grid.N; ++p)
            c(p) = std::sqrt(grid.w(p)) * std::exp(-grid.x(p) * grid.x(p));
        run_from_init("gauss alpha=1.0", c);
    }
    // 3b: 細いガウシアン (alpha=2.0)
    {
        Eigen::VectorXd c(grid.N);
        for (int p = 0; p < grid.N; ++p)
            c(p) = std::sqrt(grid.w(p)) * std::exp(-2.0 * grid.x(p) * grid.x(p));
        run_from_init("gauss alpha=2.0", c);
    }
    // 3c: 広いガウシアン (alpha=0.3)
    {
        Eigen::VectorXd c(grid.N);
        for (int p = 0; p < grid.N; ++p)
            c(p) = std::sqrt(grid.w(p)) * std::exp(-0.3 * grid.x(p) * grid.x(p));
        run_from_init("gauss alpha=0.3", c);
    }
    // 3d: 指数関数（H 原子っぽい）
    {
        Eigen::VectorXd c(grid.N);
        for (int p = 0; p < grid.N; ++p)
            c(p) = std::sqrt(grid.w(p)) * std::exp(-std::abs(grid.x(p)));
        run_from_init("exp(-|x|)", c);
    }
    // 3e: SCF 解そのもの
    {
        run_from_init("SCF solution", c_ref);
    }
    // 3f: SCF 解 + 小さな摂動
    {
        Eigen::VectorXd c = c_ref;
        for (int p = 0; p < grid.N; ++p)
            c(p) += 1e-3 * std::sin(grid.x(p));
        run_from_init("SCF + 1e-3 sin(x)", c);
    }
    // 3g: ランダム
    {
        Eigen::VectorXd c = Eigen::VectorXd::Random(grid.N);
        // 偶パリティに対称化
        for (int p = 0; p < grid.N; ++p)
        {
            const double xp = grid.x(p);
            c(p) = std::exp(-0.5 * xp * xp) + 0.1 * std::cos(0.3 * xp);
        }
        run_from_init("composite even", c);
    }

    // ============================================================
    // Experiment 4: SCF の解で評価される F（Fock）から、
    // SCF 解は F の最低固有ベクトル？
    // ============================================================
    std::cout << "\n[Experiment 4] is c_scf the lowest eigenvector of F[c_scf]?\n";
    {
        Eigen::MatrixXd C(grid.N, N_occ);
        C.col(0) = c_ref;
        const Eigen::MatrixXd P = fedvr::rhf_density(C);
        const Eigen::MatrixXd F = fedvr::rhf_fock(h_pq, V_pq, P);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(0.5 * (F + F.transpose()));
        const auto &evals = es.eigenvalues();
        const auto &evecs = es.eigenvectors();
        std::cout << "    eps[0]=" << evals(0) << ", eps[1]=" << evals(1)
                  << ", eps[2]=" << evals(2) << "\n";
        Eigen::VectorXd v0 = evecs.col(0);
        const double dist = orbital_dist(v0, c_ref);
        std::cout << "    ||lowest_eigvec(F[c_scf]) - c_scf|| = " << dist << "\n";
    }

    // ============================================================
    // Experiment 5: Hartree 虚時間で得た c_A についても、
    // F[c_A] の最低固有ベクトル == c_A になっているか（停留点条件）？
    // ============================================================
    std::cout << "\n[Experiment 5] is c_A (Hartree-imag-converged) a stationary point of F?\n";
    {
        // ガウシアン初期から短く虚時間で c_A を得る
        fedvr::RHF_ImagTimeResult rA = fedvr::rhf_imag_time(
            h_pq, V_pq, N_occ, C_init_gauss, 0.20, 5000, 1e-14);
        Eigen::VectorXd cA = rA.C_occ.col(0);
        Eigen::MatrixXd C(grid.N, N_occ);
        C.col(0) = cA;
        const Eigen::MatrixXd P = fedvr::rhf_density(C);
        const Eigen::MatrixXd F = fedvr::rhf_fock(h_pq, V_pq, P);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(0.5 * (F + F.transpose()));
        const auto &evals = es.eigenvalues();
        const auto &evecs = es.eigenvectors();
        std::cout << "    eps[0]=" << evals(0) << ", eps[1]=" << evals(1)
                  << ", eps[2]=" << evals(2) << "\n";
        Eigen::VectorXd v0 = evecs.col(0);
        const double dist = orbital_dist(v0, cA);
        std::cout << "    ||lowest_eigvec(F[c_A]) - c_A|| = " << dist << "\n";
        std::cout << "    E_RHF(c_A) = " << rA.E_RHF
                  << ", E_RHF(SCF) = " << ref.E_RHF
                  << ", diff = " << (rA.E_RHF - ref.E_RHF) << "\n";
        // F の最低固有ベクトルでエネルギーを測ってみる
        Eigen::MatrixXd Cv0(grid.N, N_occ);
        Cv0.col(0) = v0;
        std::cout << "    E_RHF(lowest_eigvec(F[c_A])) = "
                  << fedvr::rhf_energy(h_pq, V_pq, Cv0) << "\n";
    }

    return 0;
}
