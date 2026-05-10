//
// H2_consistency_check.cpp
//
// At a single internuclear distance R = 1.6 a.u. (near the equilibrium of
// the 1D soft-Coulomb H2 RHF curve), verify that all three solvers
// converge to the same RHF energy on each discretization (FD and FEDVR).
//
// FD case:
//   (A) First-quantization Hartree+exchange imag-time RK4
//       (fd::rk4_imag_step_rhf with multi-nucleus h_pq)
//   (B) Second-quantization RHF imag-time RK4
//       (fedvr::rhf_imag_time on FD integrals)
//   (C) Second-quantization RHF Roothaan-Hall SCF
//       (fedvr::rhf_scf on FD integrals)
//
// FEDVR case:
//   (B) fedvr::rhf_imag_time
//   (C) fedvr::rhf_scf
//   (FEDVR has no first-quantization-style multi-orbital RHF helper, so
//    the Be-style "first-quantization" path is FD-only.)
//
// Goal:
//   - All FD solvers agree to ~1e-6 Ha (or better) at R = 1.6
//   - All FEDVR solvers agree to ~1e-7 Ha at R = 1.6
//   - This confirms the multi-nucleus extension preserves the same
//     mathematical equivalence as the He / Be single-atom cases.
//
// Build:
//   EIGEN=/opt/homebrew/include/eigen3
//   g++ -std=c++17 -O2 -I"$EIGEN" -I"FD" -I"FEDVR" H2_consistency_check.cpp \
//       -o out/h2_consistency_check
//   ./out/h2_consistency_check
//

#include "FD/fd_basis.hpp"
#include "FD/hartree_fd.hpp"
#include "FEDVR/fedvr_basis.hpp"
#include "FEDVR/hf_fedvr.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

namespace
{

void print_kv(const std::string &label, double value)
{
    std::cout << "    " << std::left << std::setw(48) << label
              << " : " << std::right << std::scientific << std::setprecision(11)
              << value << "\n";
}

} // namespace

int main()
{
    const double R = 1.6;       // internuclear distance (a.u.)
    const double Z = 1.0;       // each proton
    const int N_e = 2;
    const int N_occ = N_e / 2;

    // imag-time RK4 settings (must agree to better than ~1e-6 with SCF)
    const double dtau = 0.01;
    const int max_itr_imag = 30000;
    const double thresh_imag = 1e-13;

    // SCF settings
    const int max_itr_scf = 200;
    const double thresh_scf = 1e-13;

    std::cout << std::scientific << std::setprecision(11);
    std::cout << "=== H2 consistency check at R = " << std::fixed
              << std::setprecision(2) << R << " a.u. ===\n";
    std::cout << std::scientific;

    // =============================================================
    // FD side: all three solvers
    // =============================================================
    {
        const double x_range = 20.0;
        const double dx = 0.2;
        fd::FDGrid grid = fd::make_grid(x_range, dx);

        std::vector<fd::Nucleus> nuclei = {{Z, +0.5 * R}, {Z, -0.5 * R}};
        Eigen::SparseMatrix<double> h_pq = fd::build_h_pq(grid, nuclei);
        Eigen::MatrixXd V_pq = fd::build_V_pq(grid);
        const double E_nn_fd = fd::nuclear_repulsion(nuclei);

        // initial guess: bonding combination of two Gaussians
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

        std::cout << "\n[FD]  x_range=" << x_range << ", dx=" << grid.dx
                  << ", N(DOF)=" << grid.N << "\n";

        // ---- (A) First-quantization Hartree+exchange imag-time RK4 ----
        // hartree_fd::rk4_imag_step_rhf is identical in form to FEDVR's
        // imag-time but uses the FD utilities directly.
        {
            Eigen::MatrixXd C = fd::qr_orthonormalize(C_init);
            double prev_E = 1e30;
            int iters = 0;
            for (int it = 0; it < max_itr_imag; ++it)
            {
                const double E = fd::rhf_energy(h_pq, V_pq, C);
                const double diff = std::abs(E - prev_E);
                if (it > 0 && diff < thresh_imag)
                {
                    iters = it;
                    break;
                }
                prev_E = E;
                fd::rk4_imag_step_rhf(h_pq, V_pq, C, dtau);
                iters = it + 1;
            }
            const double E_RHF = fd::rhf_energy(h_pq, V_pq, C);
            const double E_tot = E_RHF + E_nn_fd;
            std::cout << "  (A) first-quant Hartree+exchange imag-time RK4 (fd::rk4_imag_step_rhf)\n";
            print_kv("E_RHF", E_RHF);
            print_kv("E_tot = E_RHF + E_nn", E_tot);
            std::cout << "    iters = " << iters << "\n";
        }

        // ---- (B) Second-quantization RHF imag-time RK4 ----
        {
            fedvr::RHF_ImagTimeResult r = fedvr::rhf_imag_time(
                h_pq, V_pq, N_occ, C_init, dtau, max_itr_imag, thresh_imag);
            const double E_tot = r.E_RHF + E_nn_fd;
            std::cout << "  (B) second-quant RHF imag-time RK4 (fedvr::rhf_imag_time)\n";
            print_kv("E_RHF", r.E_RHF);
            print_kv("E_tot", E_tot);
            std::cout << "    iters = " << r.iterations
                      << ", converged = " << (r.converged ? "yes" : "NO") << "\n";
        }

        // ---- (C) Second-quantization RHF SCF ----
        {
            fedvr::RHF_SCFResult r = fedvr::rhf_scf(
                h_pq, V_pq, N_occ, C_init, max_itr_scf, thresh_scf);
            const double E_tot = r.E_RHF + E_nn_fd;
            std::cout << "  (C) second-quant RHF SCF (fedvr::rhf_scf)\n";
            print_kv("E_RHF", r.E_RHF);
            print_kv("E_tot", E_tot);
            std::cout << "    iters = " << r.iterations
                      << ", converged = " << (r.converged ? "yes" : "NO") << "\n";
        }

        std::cout << "  E_nn = " << E_nn_fd << "\n";
    }

    // =============================================================
    // FEDVR side: (B) and (C) (no first-quantization helper available)
    // =============================================================
    {
        const double L = 20.0;
        const int N_e_elem = 10;
        const int n_order = 8;
        fedvr::FEDVRGrid grid(L, N_e_elem, n_order);

        std::vector<fedvr::Nucleus> nuclei = {{Z, +0.5 * R}, {Z, -0.5 * R}};
        Eigen::SparseMatrix<double> h_pq = fedvr::build_h_pq(grid, nuclei);
        Eigen::MatrixXd V_pq = fedvr::build_V_pq(grid);
        const double E_nn_fedvr = fedvr::nuclear_repulsion(nuclei);

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

        std::cout << "\n[FEDVR]  L=" << L << ", N_e=" << N_e_elem
                  << ", n=" << n_order << ", N(DOF)=" << grid.N << "\n";

        {
            // FEDVR's CFL margin: the local lambda_max(T) is governed by
            // GLL spacing near element edges, which is much smaller than
            // the average dx_elem.  For (L=20, Ne=10, n=8) the historical
            // He/Be dtau=0.20 happens to work for atom-centered cases but
            // *diverges* with two off-center nuclei.  Using a small dtau
            // matched to the FD setting (0.01) keeps imag-time stable and
            // also lets it reach the SCF stationary point.
            fedvr::RHF_ImagTimeResult r = fedvr::rhf_imag_time(
                h_pq, V_pq, N_occ, C_init, /*dtau=*/0.01,
                /*max_itr=*/100000, thresh_imag);
            const double E_tot = r.E_RHF + E_nn_fedvr;
            std::cout << "  (B) second-quant RHF imag-time RK4 (fedvr::rhf_imag_time)\n";
            print_kv("E_RHF", r.E_RHF);
            print_kv("E_tot", E_tot);
            std::cout << "    iters = " << r.iterations
                      << ", converged = " << (r.converged ? "yes" : "NO") << "\n";
        }

        {
            fedvr::RHF_SCFResult r = fedvr::rhf_scf(
                h_pq, V_pq, N_occ, C_init, max_itr_scf, thresh_scf);
            const double E_tot = r.E_RHF + E_nn_fedvr;
            std::cout << "  (C) second-quant RHF SCF (fedvr::rhf_scf)\n";
            print_kv("E_RHF", r.E_RHF);
            print_kv("E_tot", E_tot);
            std::cout << "    iters = " << r.iterations
                      << ", converged = " << (r.converged ? "yes" : "NO") << "\n";
        }

        std::cout << "  E_nn = " << E_nn_fedvr << "\n";
    }

    std::cout << "\nIf the three FD solvers agree to ~1e-6 Ha and the two FEDVR\n"
              << "solvers agree to ~1e-7 Ha, the multi-nucleus extension preserves\n"
              << "the He / Be mathematical equivalence (Hartree-only = RHF Fock for\n"
              << "single-orbital closed-shell systems).\n";
    return 0;
}
