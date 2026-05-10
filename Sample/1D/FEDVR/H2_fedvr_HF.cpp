//
// H2_fedvr_HF.cpp
//
// Closed-shell H2 molecule (two protons separated by R, two electrons,
// N_occ = 1) on the FEDVR grid.  Computes the RHF potential energy curve
// E_tot(R) = E_RHF(R) + E_nn(R) over a sweep of internuclear distances R.
//
// Highlights:
//   * Multi-nucleus build_h_pq from fedvr_basis.hpp:
//        V_ne(x) = - 1/sqrt((x - R/2)^2 + 1) - 1/sqrt((x + R/2)^2 + 1)
//   * Soft-Coulomb nuclear repulsion:
//        E_nn = 1 / sqrt(R^2 + 1)
//   * Closed-shell RHF (N_occ = 1) via fedvr::rhf_scf
//   * CSV output to stdout: R, E_RHF, E_nn, E_tot, iters
//
// Build:
//   EIGEN=/opt/homebrew/include/eigen3
//   g++ -std=c++17 -O2 -I"$EIGEN" H2_fedvr_HF.cpp -o out/h2_fedvr_hf
//   ./out/h2_fedvr_hf > out/h2_fedvr_hf.csv
//
// Notes on physics:
//   - Equilibrium bond length R_e ≈ 1.4 a.u. is expected (similar to 3D H2
//     with the soft-Coulomb potential pushing minima outward slightly).
//   - At large R, RHF cannot localize one electron on each H atom and
//     instead spreads them over both centres → unphysically high energy
//     ("RHF dissociation problem"), demonstrating the need for UHF or
//     post-HF methods.
//

#include "fedvr_basis.hpp"
#include "hf_fedvr.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

int main()
{
    // ---- Box / grid parameters ----
    const double L = 20.0;
    const int N_e_elem = 10;
    const int n_order = 8;

    // ---- Electronic structure ----
    const int N_e = 2;          // electrons (H2 = H + H -> 2 e-)
    const int N_occ = N_e / 2;  // closed-shell: 1 spatial orbital
    const double Z = 1.0;       // each proton

    // ---- SCF settings ----
    const int max_itr = 200;
    const double thresh = 1e-13;

    // ---- Internuclear distance scan ----
    const std::vector<double> R_list = {
        0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8,
        2.0, 2.5, 3.0, 4.0, 5.0, 6.0
    };

    // Build grid once (doesn't depend on R)
    fedvr::FEDVRGrid grid(L, N_e_elem, n_order);
    Eigen::MatrixXd V_pq = fedvr::build_V_pq(grid);

    std::cerr << "FEDVR H2 RHF potential curve\n"
              << "  L=" << L
              << ", N_e_elem=" << N_e_elem
              << ", n_order=" << n_order
              << ", N(DOF)=" << grid.N << "\n";

    // CSV header
    std::cout << "# H2 RHF potential curve (FEDVR, closed-shell, N_occ=1)\n"
              << "# L=" << L << ", N_e_elem=" << N_e_elem
              << ", n_order=" << n_order << ", N(DOF)=" << grid.N << "\n"
              << "R,E_RHF,E_nn,E_tot,iters,converged\n";
    std::cout << std::scientific << std::setprecision(11);

    // Initial guess: two Gaussians centered at the nuclei (sigma ~ 1)
    // Updated each iteration as R changes
    for (double R : R_list)
    {
        std::vector<fedvr::Nucleus> nuclei = {
            {Z, +0.5 * R},
            {Z, -0.5 * R}
        };

        Eigen::SparseMatrix<double> h_pq = fedvr::build_h_pq(grid, nuclei);

        // Initial guess: bonding combination of two Gaussians on each nucleus
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

        const double E_nn = fedvr::nuclear_repulsion(nuclei);
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
