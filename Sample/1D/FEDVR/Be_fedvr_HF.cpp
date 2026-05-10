//
// Be_fedvr_HF.cpp
//
// Closed-shell Restricted Hartree-Fock (RHF) ground state of the 1D
// soft-Coulomb Be atom (Z=4, four electrons, N_occ = 2 spatial orbitals),
// formulated in second quantization on the FEDVR basis.
//
// This is the natural extension of He_fedvr_HF.cpp to N_occ > 1: it uses
// the same fedvr_basis.hpp + hf_fedvr.hpp machinery without modification,
// and exercises the exchange term K_pq = V_pq P_pq for the first time
// (in He, K cancels half of J due to single-orbital occupation, so K
//  contributes "the same" as it does in any closed-shell HF — but only
//  Be and beyond reveal multi-orbital behaviour clearly).
//
//   Solver A : imaginary-time RK4 of MO coefficients
//   Solver B : Roothaan-Hall SCF (direct Fock diagonalization)
//
// Initial guess: two orthogonal Gaussian-like orbitals (even parity
// exp(-x^2) and odd parity x exp(-x^2)).  The QR step inside the solvers
// will orthonormalize them automatically.
//
// Build:
//   EIGEN=/opt/homebrew/include/eigen3
//   g++ -std=c++17 -O2 -I"$EIGEN" Be_fedvr_HF.cpp -o out/be_fedvr_hf && ./out/be_fedvr_hf
//

#include "fedvr_basis.hpp"
#include "hf_fedvr.hpp"

#include <Eigen/Dense>
#include <iomanip>
#include <iostream>

int main()
{
    // ---- Parameters ----
    const double L = 20.0;
    const int n_elements = 10;
    const int n_order = 8;
    const double Z = 4.0;
    const int N_e = 4;
    const int N_occ = N_e / 2; // closed-shell: 2 spatial orbitals for Be

    // RHF imaginary-time RK4 settings.
    // Be has stronger nuclear attraction than He -> larger spectral range
    // -> use a slightly smaller dtau than He for stability.
    const double dtau_imag = 0.002;
    const int max_itr_imag = 100000;
    const double thresh_imag = 1e-12;

    // RHF SCF settings.
    const int max_itr_scf = 500;
    const double thresh_scf = 1e-12;

    std::cout << std::scientific << std::setprecision(12);

    // ---- Build FEDVR grid ----
    fedvr::FEDVRGrid grid(L, n_elements, n_order);
    std::cout << "FEDVR grid: L=" << L
              << ", n_elements=" << n_elements
              << ", n_order=" << n_order
              << ", N(DOF)=" << grid.N << "\n";
    std::cout << "T nnz: " << grid.T.nonZeros() << "\n";

    // ---- Second-quantization integral tensors ----
    Eigen::SparseMatrix<double> h_pq = fedvr::build_h_pq(grid, Z);
    Eigen::MatrixXd V_pq = fedvr::build_V_pq(grid);

    std::cout << "h_pq nnz: " << h_pq.nonZeros() << "\n";
    std::cout << "V_pq is " << V_pq.rows() << " x " << V_pq.cols() << " (dense)\n";

    // ---- Initial guess: two orthogonal Gaussian-like orbitals ----
    //   col 0 : even parity   ~ exp(-x^2)
    //   col 1 : odd parity    ~ x exp(-x^2)
    // (the Be 1s-like ground orbital is even, the 2s/2p-like next
    //  occupied orbital is approximately odd in the 1D model)
    Eigen::MatrixXd C_init(grid.N, N_occ);
    for (int p = 0; p < grid.N; ++p)
    {
        const double xp = grid.x(p);
        const double sw = std::sqrt(grid.w(p));
        const double g = std::exp(-xp * xp);
        C_init(p, 0) = sw * g;
        C_init(p, 1) = sw * xp * g;
    }
    // QR inside the solvers will orthonormalize, but pre-normalize each
    // column for clarity.
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
    // Solver B: Roothaan-Hall SCF (direct diagonalization)
    // ============================================================
    std::cout << "\n[Solver B: RHF Roothaan-Hall SCF (Fock diagonalization)]"
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

    std::cout << "  Lowest 5 orbital energies (Fock eigenvalues):\n";
    const int show = std::min<int>(5, static_cast<int>(resB.orbital_energies.size()));
    for (int i = 0; i < show; ++i)
    {
        std::cout << "    eps[" << i << "] = " << resB.orbital_energies(i) << "\n";
    }
    // HOMO/LUMO for closed-shell: HOMO = eps[N_occ-1], LUMO = eps[N_occ]
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
    std::cout << "  E_RHF (imag-time)  = " << resA.E_RHF << "\n";
    std::cout << "  E_RHF (SCF)        = " << resB.E_RHF << "\n";
    std::cout << "  |ΔE|               = " << std::abs(resA.E_RHF - resB.E_RHF) << "\n";

    return 0;
}
