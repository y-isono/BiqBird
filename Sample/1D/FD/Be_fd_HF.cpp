//
// Be_fd_HF.cpp
//
// Closed-shell Be atom (Z=4, four electrons, N_occ=2 spatial orbitals)
// ground state on the finite-difference (FD) grid, using the
// *second-quantization* RHF machinery from FEDVR/hf_fedvr.hpp.
//
// This is the multi-orbital extension of He_fd_HF.cpp, parallel to
// FEDVR/Be_fedvr_HF.cpp.  Two solvers are run and compared, and the
// result is also compared against the FD first-quantization result
// produced by Be_GS.cpp.
//
// Build:
//   EIGEN=/opt/homebrew/include/eigen3
//   g++ -std=c++17 -O2 -I"$EIGEN" -I"../FEDVR" Be_fd_HF.cpp -o out/be_fd_hf
//   ./out/be_fd_hf
//

#include "fd_basis.hpp"
#include "../FEDVR/hf_fedvr.hpp"

#include <Eigen/Dense>
#include <iomanip>
#include <iostream>

int main()
{
    // ---- Parameters ----
    const double x_range = 20.0;
    const double dx = 0.2;
    const double Z = 4.0;
    const int N_e = 4;
    const int N_occ = N_e / 2;

    // Tightened so that Solver A and Solver B agree with Be_GS.cpp's
    // first-quantization-style imag-time result to better than ~1e-5 Ha.
    const double dtau_imag = 0.002;
    const int max_itr_imag = 200000;
    const double thresh_imag = 1e-13;

    const int max_itr_scf = 500;
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

    // ---- Initial guess: even + odd Gaussians ----
    Eigen::MatrixXd C_init(grid.N, N_occ);
    for (int p = 0; p < grid.N; ++p)
    {
        const double xp = grid.x(p);
        const double sw = std::sqrt(grid.w(p));
        const double g = std::exp(-xp * xp);
        C_init(p, 0) = sw * g;
        C_init(p, 1) = sw * xp * g;
    }
    for (int j = 0; j < N_occ; ++j)
        C_init.col(j) /= C_init.col(j).norm();

    // ============================================================
    // Solver A: Imaginary-time RK4
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
    // Solver B: Roothaan-Hall SCF
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

    if (resB.orbital_energies.size() >= N_occ + 1)
    {
        const double homo = resB.orbital_energies(N_occ - 1);
        const double lumo = resB.orbital_energies(N_occ);
        std::cout << "  HOMO        = " << homo << "\n";
        std::cout << "  LUMO        = " << lumo << "\n";
        std::cout << "  HOMO-LUMO gap = " << (lumo - homo) << "\n";
    }

    // ============================================================
    // Comparison
    // ============================================================
    std::cout << "\n[Comparison]\n";
    std::cout << "  E_RHF (imag-time)   = " << resA.E_RHF << "\n";
    std::cout << "  E_RHF (SCF)         = " << resB.E_RHF << "\n";
    std::cout << "  |ΔE|                = " << std::abs(resA.E_RHF - resB.E_RHF) << "\n";

    return 0;
}
