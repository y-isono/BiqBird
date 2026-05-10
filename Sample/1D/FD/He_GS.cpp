//
// He_GS.cpp
//
// Closed-shell He atom (Z=2, two electrons in a single spatial orbital)
// ground state on the finite-difference (FD) grid, in the *first-quantization*
// Hartree mean-field formulation.
//
// This is the FD counterpart of FEDVR/He_fedvr_GS.cpp.  Output should
// reproduce the legacy value E ≈ -2.22837 Ha (see legacy/README.md) up
// to floating-point convergence tolerance.
//
// The mean-field operator is
//   F[ψ] = T + diag(V_ne) + diag(W[|ψ|^2])
//   W_p  = Σ_q V_pq |c_q|^2
// with c_p = sqrt(w_p) ψ(x_p).  See hartree_fd.hpp for details.
//
// Build:
//   EIGEN=/opt/homebrew/include/eigen3
//   g++ -std=c++17 -O2 -I"$EIGEN" He_GS.cpp -o out/he_gs && ./out/he_gs
//

#include "fd_basis.hpp"
#include "hartree_fd.hpp"

#include <Eigen/Dense>
#include <iomanip>
#include <iostream>

int main()
{
    // ---- Parameters (match legacy He_GS.cpp: x_range=20, dx=0.4) ----
    const double x_range = 20.0;
    const double dx = 0.4;
    const double Z = 2.0;
    // dtau is chosen well within the CFL stability bound (lambda_max(T) ~ 1/dx^2)
    // and small enough that the energy-based convergence test reaches the true
    // RHF stationary point.  Earlier choices like dtau=0.20 satisfied the
    // energy threshold prematurely and stopped ~5e-4 Ha above the true
    // (SCF) solution; with dtau=0.01 the three solvers (this file's Hartree
    // imag-time, He_fd_HF.cpp Solver A imag-time, He_fd_HF.cpp Solver B SCF)
    // all agree to better than 1e-6 Ha.  See README §7 for details.
    const double dtau = 0.01;
    const int max_itr = 30000;
    const double thresh = 1e-13;

    std::cout << std::scientific << std::setprecision(12);

    // ---- Build FD grid + integral tensors ----
    fd::FDGrid grid = fd::make_grid(x_range, dx);
    std::cout << "FD grid: x_range=" << grid.x_range
              << ", dx=" << grid.dx
              << ", N(DOF)=" << grid.N << "\n";
    std::cout << "T nnz: " << grid.T.nonZeros() << "\n";

    Eigen::VectorXd V_ne(grid.N);
    for (int p = 0; p < grid.N; ++p)
        V_ne(p) = -Z / std::sqrt(grid.x(p) * grid.x(p) + 1.0);

    Eigen::MatrixXd V_pq = fd::build_V_pq(grid);

    // ---- Initial guess: Gaussian, projected onto FD coefficients ----
    Eigen::VectorXd c(grid.N);
    for (int p = 0; p < grid.N; ++p)
    {
        const double psi = std::exp(-grid.x(p) * grid.x(p));
        c(p) = std::sqrt(grid.w(p)) * psi;
    }
    c /= c.norm();

    std::cout << "\n[Imaginary-time RK4, Hartree mean field]"
              << "  dtau=" << dtau
              << ", max_itr=" << max_itr
              << ", thresh=" << thresh << "\n";

    double prev_E = 1e30;
    for (int it = 0; it < max_itr; ++it)
    {
        const auto E = fd::compute_energies_hartree(grid.T, V_ne, V_pq, c);
        const double diff = std::abs(E.Etot - prev_E);
        if (it % 50 == 0)
        {
            std::cout << "  itr=" << it
                      << ", E=" << E.Etot
                      << ", E1=" << E.E1
                      << ", E2=" << E.E2
                      << ", diff=" << diff
                      << "\n";
        }
        if (it > 0 && diff < thresh)
        {
            std::cout << "  converged at itr=" << it
                      << ", E=" << E.Etot << "\n";
            break;
        }
        prev_E = E.Etot;
        fd::rk4_imag_step_hartree(grid.T, V_ne, V_pq, c, dtau);
    }

    const auto E_final = fd::compute_energies_hartree(grid.T, V_ne, V_pq, c);
    std::cout << "\n[Result]\n";
    std::cout << "  E_total = " << E_final.Etot << "\n";
    std::cout << "  E_1     = " << E_final.E1 << "  (= 2 * <h_2>)\n";
    std::cout << "  E_2     = " << E_final.E2 << "  (Hartree)\n";

    constexpr double E_ref_legacy = -2.22837e+00;
    std::cout << "  E_ref (legacy/He_GS.cpp) = " << E_ref_legacy << "\n";
    std::cout << "  |E - E_ref|              = "
              << std::abs(E_final.Etot - E_ref_legacy) << "\n";

    return 0;
}
