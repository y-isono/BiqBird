//
// H2_fd_HF.cpp
//
// Closed-shell H2 molecule (two protons separated by R, two electrons,
// N_occ = 1) on the FD grid.  Computes the RHF potential energy curve
// E_tot(R) = E_RHF(R) + E_nn(R) over a sweep of internuclear distances R.
//
// Mirrors FEDVR/H2_fedvr_HF.cpp; uses fd::Nucleus / fd::build_h_pq for the
// multi-nucleus 1-electron Hamiltonian and reuses fedvr::rhf_scf as the
// SCF driver (the DVR-like FD basis satisfies the same algebraic
// structure as FEDVR for the second-quantization integrals).
//
// Build:
//   EIGEN=/opt/homebrew/include/eigen3
//   g++ -std=c++17 -O2 -I"$EIGEN" -I"../FEDVR" H2_fd_HF.cpp -o out/h2_fd_hf
//   ./out/h2_fd_hf > out/h2_fd_hf.csv
//

#include "fd_basis.hpp"
#include "../FEDVR/hf_fedvr.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

int main()
{
    // ---- Box / grid parameters ----
    const double x_range = 20.0;
    const double dx = 0.2;

    // ---- Electronic structure ----
    const int N_e = 2;
    const int N_occ = N_e / 2;
    const double Z = 1.0;

    // ---- SCF settings ----
    const int max_itr = 200;
    const double thresh = 1e-13;

    const std::vector<double> R_list = {
        0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8,
        2.0, 2.5, 3.0, 4.0, 5.0, 6.0
    };

    fd::FDGrid grid = fd::make_grid(x_range, dx);
    Eigen::MatrixXd V_pq = fd::build_V_pq(grid);

    std::cerr << "FD H2 RHF potential curve\n"
              << "  x_range=" << x_range << ", dx=" << grid.dx
              << ", N(DOF)=" << grid.N << "\n";

    std::cout << "# H2 RHF potential curve (FD, closed-shell, N_occ=1)\n"
              << "# x_range=" << x_range << ", dx=" << grid.dx
              << ", N(DOF)=" << grid.N << "\n"
              << "R,E_RHF,E_nn,E_tot,iters,converged\n";
    std::cout << std::scientific << std::setprecision(11);

    for (double R : R_list)
    {
        std::vector<fd::Nucleus> nuclei = {
            {Z, +0.5 * R},
            {Z, -0.5 * R}
        };

        Eigen::SparseMatrix<double> h_pq = fd::build_h_pq(grid, nuclei);

        Eigen::MatrixXd C_init(grid.N, N_occ);
        for (int p = 0; p < grid.N; ++p)
        {
            const double xp = grid.x(p);
            const double sw = std::sqrt(grid.w(p));
            const double gL = std::exp(-(xp + 0.5 * R) * (xp + 0.5 * R));
            const double gR = std::exp(-(xp - 0.5 * R) * (xp - 0.5 * R));
            C_init(p, 0) = sw * (gL + gR);
        }
        C_init.col(0) /= C_init.col(0).norm();

        fedvr::RHF_SCFResult res = fedvr::rhf_scf(
            h_pq, V_pq, N_occ, C_init, max_itr, thresh);

        const double E_nn = fd::nuclear_repulsion(nuclei);
        const double E_tot = res.E_RHF + E_nn;

        std::cout << R << ","
                  << res.E_RHF << ","
                  << E_nn << ","
                  << E_tot << ","
                  << res.iterations << ","
                  << (res.converged ? 1 : 0) << "\n";

        std::cerr << "  R=" << std::fixed << std::setprecision(2) << R
                  << "  E_RHF=" << std::scientific << std::setprecision(8) << res.E_RHF
                  << "  E_nn=" << E_nn
                  << "  E_tot=" << E_tot
                  << "  iters=" << res.iterations
                  << "\n";
    }

    return 0;
}
