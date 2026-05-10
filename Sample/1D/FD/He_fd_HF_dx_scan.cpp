//
// He_fd_HF_dx_scan.cpp
//
// Phase 3: dx を変えて E_imag と E_scf のズレを観察。
//
//   - DVR 縮約誤差仮説（fd_two_electron_integral_consideration.md）が正しければ、
//     E_scf - E_imag は O(dx^2) で減るはず。
//   - 実は虚時間の収束不足が主因なら、dx に依らず（あるいは dx と無関係に）
//     dtau / max_itr で決まるはず。
//
// 各 dx で
//   - SCF 解 (E_scf) を取得
//   - 虚時間を「同じ dtau, 同じ max_itr, 同じ thresh」で回した E_imag を取得
//   - ズレ = E_imag - E_scf を出力
//
// さらに比較として、虚時間を「軌道距離が thresh 以下になるまで」回した値も出す。
//
// Build:
//   EIGEN=/opt/homebrew/include/eigen3
//   g++ -std=c++17 -O2 -I"$EIGEN" -I"../FEDVR" He_fd_HF_dx_scan.cpp -o out/he_fd_dx_scan
//   ./out/he_fd_dx_scan
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
    const double Z = 2.0;
    const int N_occ = 1;

    std::cout << std::scientific << std::setprecision(8);

    const std::vector<double> dxs = {0.4, 0.2, 0.1, 0.05};

    std::cout << std::setw(6) << "dx"
              << std::setw(6) << "N"
              << std::setw(20) << "E_scf"
              << std::setw(22) << "E_imag(0.20,old)"
              << std::setw(22) << "E_imag(0.05,tight)"
              << std::setw(22) << "Delta_old=Eimag-Escf"
              << std::setw(22) << "Delta_tight=Eimag-Escf"
              << "\n";

    for (double dx : dxs)
    {
        fd::FDGrid grid = fd::make_grid(x_range, dx);
        Eigen::SparseMatrix<double> h_pq = fd::build_h_pq(grid, Z);
        Eigen::MatrixXd V_pq = fd::build_V_pq(grid);

        // Gaussian init
        Eigen::MatrixXd C_init(grid.N, N_occ);
        for (int p = 0; p < grid.N; ++p)
            C_init(p, 0) = std::sqrt(grid.w(p)) *
                           std::exp(-grid.x(p) * grid.x(p));
        C_init.col(0) /= C_init.col(0).norm();

        // SCF reference (kept tight)
        fedvr::RHF_SCFResult ref = fedvr::rhf_scf(h_pq, V_pq, N_occ, C_init, 2000, 1e-15);

        // 旧設定（README/He_fd_HF.cpp 互換）: dtau=0.20, thresh=1e-12, max_itr=5000
        fedvr::RHF_ImagTimeResult r_old = fedvr::rhf_imag_time(
            h_pq, V_pq, N_occ, C_init, 0.20, 5000, 1e-12);

        // tight 設定: dtau=0.05, thresh=1e-15, max_itr=200000
        // dx が小さいほど CFL の都合で dtau は小さくする（dx^2 程度）
        const double dtau_tight = std::min(0.05, 0.5 * dx * dx);
        fedvr::RHF_ImagTimeResult r_tight = fedvr::rhf_imag_time(
            h_pq, V_pq, N_occ, C_init, dtau_tight, 200000, 1e-15);

        const double E_scf = ref.E_RHF;
        const double E_old = r_old.E_RHF;
        const double E_tight = r_tight.E_RHF;
        std::cout << std::setw(6) << dx
                  << std::setw(6) << grid.N
                  << std::setw(20) << E_scf
                  << std::setw(22) << E_old
                  << std::setw(22) << E_tight
                  << std::setw(22) << (E_old - E_scf)
                  << std::setw(22) << (E_tight - E_scf)
                  << "\n";
    }

    std::cout << "\nNotes:\n";
    std::cout << "  - If 'Delta_old' decreases as O(dx^2), it would be a discretization-error story.\n";
    std::cout << "  - If 'Delta_tight' is much smaller than 'Delta_old' at the same dx, the gap\n";
    std::cout << "    in 'Delta_old' is a convergence artifact (early stopping by E-thresh) of imag-time RK4,\n";
    std::cout << "    not a DVR-contraction error.\n";
    return 0;
}
