//
// He_fedvr_GS.cpp
//
// Ground state of the 1D soft-Coulomb closed-shell He atom (Z=2, two electrons,
// single spatial orbital + Hartree mean field) using FEDVR.
//
// Method: imaginary-time RK4 of the Hartree mean-field equation
//   ∂_τ ψ = -(ĥ_2 + W[|ψ|²]) ψ,   then L²-normalize each step.
// (Same model as ../FD/He_GS.cpp; no exchange.)
//
// Build:
//   EIGEN=/opt/homebrew/include/eigen3
//   g++ -std=c++17 -O2 -I"$EIGEN" He_fedvr_GS.cpp -o out/he_fedvr_gs && ./out/he_fedvr_gs
//

#include "He_fedvr.hpp"

#include <iomanip>
#include <iostream>

int main()
{
    // ---- Parameters (see ../FEDVR/README.md §8) ----
    const double L = 20.0;
    const int n_elements = 10;
    const int n_order = 8;
    const double dtau = 0.005;       // dt small enough for RK4 stability (Hmax dominates)
    const int max_itr = 50000;
    const double thresh = 1e-12;
    const double Z = 2.0;

    std::cout << std::scientific << std::setprecision(12);

    // ---- Build FEDVR grid + operators ----
    fedvr::FEDVRGrid grid(L, n_elements, n_order);
    std::cout << "FEDVR grid: L=" << L
              << ", n_elements=" << n_elements
              << ", n_order=" << n_order
              << ", N(DOF)=" << grid.N << "\n";
    std::cout << "T nnz: " << grid.T.nonZeros() << "\n";

    Eigen::VectorXd Vne = fedvr::one_electron_potential(grid, Z);
    Eigen::MatrixXd Wkernel = fedvr::build_softcoulomb_kernel(grid);

    // ---- Initial guess: gaussian, projected onto DVR coefficients ----
    Eigen::VectorXd c(grid.N);
    for (int a = 0; a < grid.N; ++a)
    {
        const double psi = std::exp(-grid.x(a) * grid.x(a));
        c(a) = std::sqrt(grid.w(a)) * psi;
    }
    c /= c.norm();

    std::cout << "\n[Imaginary-time RK4 He, Hartree mean field]"
              << "  dtau=" << dtau
              << ", max_itr=" << max_itr
              << ", thresh=" << thresh << "\n";

    double prev_E = 1e10;
    for (int it = 0; it < max_itr; ++it)
    {
        const auto E = fedvr::compute_energies(grid, Vne, Wkernel, c);
        const double diff = std::abs(E.Etot - prev_E);
        if (it % 200 == 0)
        {
            std::cout << "  itr=" << it
                      << ", E=" << E.Etot
                      << ", E1=" << E.E1
                      << ", E2=" << E.E2
                      << ", diff=" << diff
                      << ", |c|=" << c.norm()
                      << "\n";
        }
        if (diff < thresh && it > 0)
        {
            std::cout << "  converged at itr=" << it
                      << ", E=" << E.Etot
                      << ", E1=" << E.E1
                      << ", E2=" << E.E2
                      << ", diff=" << diff << "\n";
            break;
        }
        prev_E = E.Etot;
        fedvr::rk4_imag_step_He(grid, Vne, Wkernel, c, dtau);
    }

    const auto E_final = fedvr::compute_energies(grid, Vne, Wkernel, c);
    std::cout << "\n[Result]\n";
    std::cout << "  E_total = " << E_final.Etot << "\n";
    std::cout << "  E_1     = " << E_final.E1 << "  (= 2 * <h_2>)\n";
    std::cout << "  E_2     = " << E_final.E2 << "  (Hartree)\n";

    return 0;
}
