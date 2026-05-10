//
// Be_GS.cpp
//
// Closed-shell Be atom (Z=4, four electrons, N_occ=2 spatial orbitals)
// ground state on the finite-difference (FD) grid, in the
// *first-quantization-style* multi-orbital RHF formulation.
//
// Despite the "first-quantization" framing (we propagate the orbital
// coefficients in imaginary time RK4, in the same style as He_GS.cpp), the
// underlying Fock operator is the full closed-shell RHF (Coulomb J +
// exchange K), so the result is the *same* RHF energy as the
// second-quantization solver in Be_fd_HF.cpp.
//
// This file is the FD counterpart of FEDVR/Be_fedvr_HF.cpp's solver A
// (imaginary-time RK4) but written using fd::hartree_fd.hpp utilities.
//
// Build:
//   EIGEN=/opt/homebrew/include/eigen3
//   g++ -std=c++17 -O2 -I"$EIGEN" Be_GS.cpp -o out/be_gs && ./out/be_gs
//

#include "fd_basis.hpp"
#include "hartree_fd.hpp"

#include <Eigen/Dense>
#include <iomanip>
#include <iostream>

int main()
{
    // ---- Parameters ----
    const double x_range = 20.0;
    const double dx = 0.2; // tighter grid than He: Be is more localized
    const double Z = 4.0;
    const int N_e = 4;
    const int N_occ = N_e / 2; // closed-shell: 2 spatial orbitals

    // Match the imag-time setup of Be_fd_HF.cpp Solver A so all three
    // solvers (this first-quantization-style imag-time, the
    // second-quantization imag-time in Be_fd_HF.cpp, and the SCF in the
    // same) converge to the same RHF energy to better than ~1e-5 Ha.
    // Earlier dtau=0.05 was unstable here too (CFL margin) and stopped
    // the energy-difference test prematurely.
    const double dtau = 0.002;
    const int max_itr = 200000;
    const double thresh = 1e-13;

    std::cout << std::scientific << std::setprecision(12);

    // ---- Build FD grid + integral tensors ----
    fd::FDGrid grid = fd::make_grid(x_range, dx);
    std::cout << "FD grid: x_range=" << grid.x_range
              << ", dx=" << grid.dx
              << ", N(DOF)=" << grid.N << "\n";
    std::cout << "T nnz: " << grid.T.nonZeros() << "\n";

    Eigen::SparseMatrix<double> h_pq = fd::build_h_pq(grid, Z);
    Eigen::MatrixXd V_pq = fd::build_V_pq(grid);

    // ---- Initial guess: even (1s-like) + odd (2p-like) Gaussians ----
    Eigen::MatrixXd C_occ(grid.N, N_occ);
    for (int p = 0; p < grid.N; ++p)
    {
        const double xp = grid.x(p);
        const double sw = std::sqrt(grid.w(p));
        const double g = std::exp(-xp * xp);
        C_occ(p, 0) = sw * g;
        C_occ(p, 1) = sw * xp * g;
    }
    for (int j = 0; j < N_occ; ++j)
        C_occ.col(j) /= C_occ.col(j).norm();
    C_occ = fd::qr_orthonormalize(C_occ);

    std::cout << "\n[Imaginary-time RK4, multi-orbital RHF (Hartree+exchange)]"
              << "  dtau=" << dtau
              << ", max_itr=" << max_itr
              << ", thresh=" << thresh << "\n";

    double prev_E = 1e30;
    int iters = 0;
    for (int it = 0; it < max_itr; ++it)
    {
        const double E = fd::rhf_energy(h_pq, V_pq, C_occ);
        const double diff = std::abs(E - prev_E);
        if (it % 200 == 0)
            std::cout << "  itr=" << it << ", E=" << E << ", diff=" << diff << "\n";
        if (it > 0 && diff < thresh)
        {
            std::cout << "  converged at itr=" << it << ", E=" << E << "\n";
            iters = it;
            break;
        }
        prev_E = E;
        fd::rk4_imag_step_rhf(h_pq, V_pq, C_occ, dtau);
        iters = it + 1;
    }

    const double E_final = fd::rhf_energy(h_pq, V_pq, C_occ);
    std::cout << "\n[Result]\n";
    std::cout << "  E_RHF (multi-orbital, FD imag-time) = " << E_final << "\n";
    std::cout << "  iters = " << iters << "\n";

    return 0;
}
