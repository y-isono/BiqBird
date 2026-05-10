//
// He_fedvr_HF.cpp
//
// Closed-shell Restricted Hartree-Fock (RHF) ground state of the 1D
// soft-Coulomb He atom (Z=2, two electrons, N_occ = 1 spatial orbital),
// formulated in second quantization on the FEDVR basis.
//
// We solve the same physical problem as ../FEDVR/He_fedvr_GS.cpp, but here
// using the second-quantization machinery in fedvr_basis.hpp + hf_fedvr.hpp.
// The two RHF solvers below should agree numerically (and with He_fedvr_GS.cpp,
// since for two electrons in a single spatial orbital, exchange exactly cancels
// half of Coulomb).
//
//   Solver A : imaginary-time RK4 of MO coefficients (matches the style of
//              H_fedvr.hpp / He_fedvr.hpp)
//   Solver B : Roothaan-Hall SCF (direct Fock diagonalization)
//
// Build:
//   EIGEN=/opt/homebrew/include/eigen3
//   g++ -std=c++17 -O2 -I"$EIGEN" He_fedvr_HF.cpp -o out/he_fedvr_hf && ./out/he_fedvr_hf
//

#include "fedvr_basis.hpp"
#include "hf_fedvr.hpp"

#include <Eigen/Dense>
#include <iomanip>
#include <iostream>

int main()
{
    // ---- Parameters (match He_fedvr_GS.cpp) ----
    const double L = 20.0;
    const int n_elements = 10;
    const int n_order = 8;
    const double Z = 2.0;
    const int N_e = 2;
    const int N_occ = N_e / 2; // closed-shell: 1 spatial orbital for He

    // RHF imaginary-time RK4 settings
    const double dtau_imag = 0.005;
    const int max_itr_imag = 50000;
    const double thresh_imag = 1e-12;

    // RHF SCF settings
    const int max_itr_scf = 200;
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

    // ---- Initial guess: Gaussian, projected onto DVR coefficients ----
    Eigen::MatrixXd C_init(grid.N, N_occ);
    for (int p = 0; p < grid.N; ++p)
    {
        const double psi = std::exp(-grid.x(p) * grid.x(p));
        C_init(p, 0) = std::sqrt(grid.w(p)) * psi;
    }
    // QR will normalize, but we also pre-normalize for clarity
    C_init.col(0) /= C_init.col(0).norm();

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

    // ============================================================
    // Comparison
    // ============================================================
    std::cout << "\n[Comparison]\n";
    std::cout << "  E_RHF (imag-time)  = " << resA.E_RHF << "\n";
    std::cout << "  E_RHF (SCF)        = " << resB.E_RHF << "\n";
    std::cout << "  |ΔE|               = " << std::abs(resA.E_RHF - resB.E_RHF) << "\n";

    // Reference value from existing Hartree-only solver (../FEDVR/He_fedvr_GS.cpp).
    // For closed-shell He on a single spatial orbital, RHF and Hartree
    // numerically coincide (exchange cancels exactly), so we expect agreement
    // at the level of solver convergence.
    constexpr double E_ref_He_fedvr_GS = -2.224120e+00; // from README §9
    std::cout << "  E_ref (He_fedvr_GS.cpp, Hartree mean field) = "
              << E_ref_He_fedvr_GS << "\n";
    std::cout << "  |E_RHF (SCF) - E_ref| = "
              << std::abs(resB.E_RHF - E_ref_He_fedvr_GS) << "\n";

    return 0;
}
