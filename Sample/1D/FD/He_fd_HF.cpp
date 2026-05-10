//
// He_fd_HF.cpp
//
// Closed-shell He atom (Z=2, two electrons in a single spatial orbital)
// ground state on the finite-difference (FD) grid, using the
// *second-quantization* RHF machinery from FEDVR/hf_fedvr.hpp.
//
// The FEDVR closed-shell RHF solver only depends on the integral tensors
// (h_pq sparse, V_pq dense), so it works on any DVR-like basis.  The FD
// basis built by fd_basis.hpp satisfies the same DVR-like algebra
// (V_pqrs = δ_pr δ_qs V_pq), so we can reuse fedvr::rhf_imag_time and
// fedvr::rhf_scf as-is.
//
// Two solvers are run and compared (and against the first-quantization
// He_GS.cpp result):
//   Solver A : imaginary-time RK4 of MO coefficients
//   Solver B : Roothaan-Hall SCF (direct Fock diagonalization)
//
// Build:
//   EIGEN=/opt/homebrew/include/eigen3
//   g++ -std=c++17 -O2 -I"$EIGEN" -I"../FEDVR" He_fd_HF.cpp -o out/he_fd_hf
//   ./out/he_fd_hf
//

#include "fd_basis.hpp"
#include "../FEDVR/hf_fedvr.hpp"

#include <Eigen/Dense>
#include <iomanip>
#include <iostream>

int main()
{
    // ---- Parameters (match legacy He_GS.cpp: x_range=20, dx=0.4) ----
    const double x_range = 20.0;
    const double dx = 0.4;
    const double Z = 2.0;
    const int N_e = 2;
    const int N_occ = N_e / 2;

    // dtau is set well below the CFL bound and small enough that the
    // energy-based convergence test reaches the true RHF stationary point
    // (matches He_GS.cpp; see README §7 and fd_he_diagnostic_results.md).
    const double dtau_imag = 0.01;
    const int max_itr_imag = 30000;
    const double thresh_imag = 1e-13;

    const int max_itr_scf = 200;
    const double thresh_scf = 1e-13;

    std::cout << std::scientific << std::setprecision(12);

    // ---- Build FD grid + integral tensors ----
    fd::FDGrid grid = fd::make_grid(x_range, dx);
    std::cout << "FD grid: x_range=" << grid.x_range
              << ", dx=" << grid.dx
              << ", N(DOF)=" << grid.N << "\n";

    Eigen::SparseMatrix<double> h_pq = fd::build_h_pq(grid, Z);
    Eigen::MatrixXd V_pq = fd::build_V_pq(grid);
    std::cout << "h_pq nnz: " << h_pq.nonZeros() << "\n";
    std::cout << "V_pq is " << V_pq.rows() << " x " << V_pq.cols() << "\n";

    // ---- Initial guess: Gaussian ----
    Eigen::MatrixXd C_init(grid.N, N_occ);
    for (int p = 0; p < grid.N; ++p)
    {
        const double psi = std::exp(-grid.x(p) * grid.x(p));
        C_init(p, 0) = std::sqrt(grid.w(p)) * psi;
    }
    C_init.col(0) /= C_init.col(0).norm();

    // ============================================================
    // Solver A: Imaginary-time RK4 (FEDVR's solver, FD integrals)
    // ============================================================
    std::cout << "\n[Solver A: RHF imaginary-time RK4]"
              << "  dtau=" << dtau_imag
              << ", max_itr=" << max_itr_imag
              << ", thresh=" << thresh_imag << "\n";

    fedvr::RHF_ImagTimeResult resA = fedvr::rhf_imag_time(
        h_pq, V_pq, N_occ, C_init, dtau_imag, max_itr_imag, thresh_imag);

    if (resA.converged)
        std::cout << "  converged at itr=" << resA.iterations
                  << ", E_RHF = " << resA.E_RHF << "\n";
    else
        std::cout << "  NOT converged after " << resA.iterations
                  << " itr, E_RHF = " << resA.E_RHF << "\n";

    // ============================================================
    // Solver B: Roothaan-Hall SCF (FEDVR's solver, FD integrals)
    // ============================================================
    std::cout << "\n[Solver B: RHF Roothaan-Hall SCF]"
              << "  max_itr=" << max_itr_scf
              << ", thresh=" << thresh_scf << "\n";

    fedvr::RHF_SCFResult resB = fedvr::rhf_scf(
        h_pq, V_pq, N_occ, C_init, max_itr_scf, thresh_scf);

    if (resB.converged)
        std::cout << "  converged at itr=" << resB.iterations
                  << ", E_RHF = " << resB.E_RHF << "\n";
    else
        std::cout << "  NOT converged after " << resB.iterations
                  << " itr, E_RHF = " << resB.E_RHF << "\n";

    std::cout << "  Lowest 5 orbital energies:\n";
    const int show = std::min<int>(5, static_cast<int>(resB.orbital_energies.size()));
    for (int i = 0; i < show; ++i)
        std::cout << "    eps[" << i << "] = " << resB.orbital_energies(i) << "\n";

    // ============================================================
    // Comparison
    // ============================================================
    std::cout << "\n[Comparison]\n";
    std::cout << "  E_RHF (imag-time)   = " << resA.E_RHF << "\n";
    std::cout << "  E_RHF (SCF)         = " << resB.E_RHF << "\n";
    std::cout << "  |ΔE|                = " << std::abs(resA.E_RHF - resB.E_RHF) << "\n";

    constexpr double E_ref_He_GS = -2.228372523361e+00; // FD He_GS.cpp (Hartree, refactored)
    std::cout << "  E_ref (He_GS.cpp, FD Hartree) = " << E_ref_He_GS << "\n";
    std::cout << "  |E_RHF (SCF) - E_ref|         = "
              << std::abs(resB.E_RHF - E_ref_He_GS) << "\n";

    return 0;
}
