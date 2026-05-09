//
// H_fedvr_GS.cpp
//
// Ground state of the 1D soft-Coulomb hydrogen atom (Z=1) using FEDVR.
//
// We compute the ground-state energy two ways:
//   (A) direct diagonalization of the dense Hamiltonian (Eigen::SelfAdjointEigenSolver)
//   (B) imaginary-time RK4 propagation, starting from a Gaussian guess
// and verify they agree.
//
// Build:
//   EIGEN=/opt/homebrew/include/eigen3
//   g++ -std=c++17 -O2 -I"$EIGEN" H_fedvr_GS.cpp -o out/h_fedvr_gs && ./out/h_fedvr_gs
//

#include "H_fedvr.hpp"

#include <Eigen/Eigenvalues>
#include <iomanip>
#include <iostream>

int main()
{
    // ---- Parameters (see ../FEDVR/README.md §8) ----
    const double L = 20.0;
    const int n_elements = 10;
    const int n_order = 8;
    const double dtau = 0.005;
    const int max_itr = 50000;
    const double thresh = 1e-13;
    const double Z = 1.0;

    std::cout << std::scientific << std::setprecision(12);

    // ---- Build FEDVR grid ----
    fedvr::FEDVRGrid grid(L, n_elements, n_order);
    std::cout << "FEDVR grid: L=" << L
              << ", n_elements=" << n_elements
              << ", n_order=" << n_order
              << ", N(DOF)=" << grid.N << "\n";
    std::cout << "T nnz: " << grid.T.nonZeros() << "\n";

    // ---- Build potential ----
    Eigen::VectorXd V = fedvr::one_electron_potential(grid, Z);

    // ============================================================
    // (A) Direct diagonalization
    // ============================================================
    {
        Eigen::MatrixXd H = fedvr::one_electron_hamiltonian_dense(grid, Z);
        // Symmetrize numerically (SelfAdjointEigenSolver only reads lower triangle by default,
        // but our T is built symmetric up to floating noise; symmetrize for safety):
        H = 0.5 * (H + H.transpose());

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H);
        if (es.info() != Eigen::Success)
        {
            std::cerr << "Diagonalization failed!\n";
            return 1;
        }
        const auto &evals = es.eigenvalues();
        std::cout << "\n[Diagonalization]\n";
        std::cout << "  E_0 = " << evals(0) << "\n";
        std::cout << "  E_1 = " << evals(1) << "\n";
        std::cout << "  E_2 = " << evals(2) << "\n";
        std::cout << "  E_max = " << evals(evals.size() - 1) << "\n";
        // Symmetry check
        const double sym = (H - H.transpose()).cwiseAbs().maxCoeff();
        std::cout << "  max|H - H^T| = " << sym << "\n";
    }

    // ============================================================
    // (B) Imaginary-time RK4 propagation
    // ============================================================
    {
        // Initial guess: gaussian, projected onto the DVR basis as
        //   c_α = √(w_α) ψ(x_α)  with ψ(x) = exp(-x²)
        Eigen::VectorXd c(grid.N);
        for (int a = 0; a < grid.N; ++a)
        {
            const double psi = std::exp(-grid.x(a) * grid.x(a));
            c(a) = std::sqrt(grid.w(a)) * psi;
        }
        c /= c.norm();

        std::cout << "\n[Imaginary-time RK4]  dtau=" << dtau << ", max_itr=" << max_itr
                  << ", thresh=" << thresh << "\n";

        double prev_E = 1e10;
        for (int it = 0; it < max_itr; ++it)
        {
            const double E = fedvr::energy(grid, V, c);
            const double diff = std::abs(E - prev_E);
            if (it % 200 == 0)
            {
                std::cout << "  itr=" << it << ", E=" << E << ", diff=" << diff << "\n";
            }
            if (diff < thresh && it > 0)
            {
                std::cout << "  converged at itr=" << it << ", E=" << E << ", diff=" << diff << "\n";
                break;
            }
            prev_E = E;
            fedvr::rk4_imag_step(grid, V, c, dtau);
        }
        const double E_final = fedvr::energy(grid, V, c);
        std::cout << "  E_final = " << E_final << "\n";
    }

    return 0;
}
