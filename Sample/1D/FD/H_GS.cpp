//
// H_GS.cpp
//
// Ground state of the 1D soft-Coulomb hydrogen atom (Z=1) on a finite-
// difference (FD) grid.  Two methods are computed and compared:
//   (A) direct diagonalization of the dense one-electron Hamiltonian
//   (B) imaginary-time RK4 propagation from a Gaussian initial guess
//
// This is the FD counterpart of FEDVR/H_fedvr_GS.cpp.  All wavefunction
// arithmetic uses real-valued Eigen types — the soft-Coulomb GS Hamiltonian
// is real-symmetric, so its eigenfunctions can be chosen real.  TDSE-style
// extensions can switch to complex types in a separate file (see roadmap).
//
// Build:
//   EIGEN=/opt/homebrew/include/eigen3
//   g++ -std=c++17 -O2 -I"$EIGEN" H_GS.cpp -o out/h_gs && ./out/h_gs
//

#include "fd_basis.hpp"

#include <Eigen/Eigenvalues>
#include <iomanip>
#include <iostream>

namespace
{

// One-electron diagonal nuclear-attraction potential at every grid point
inline Eigen::VectorXd one_electron_potential(const fd::FDGrid &g, double Z)
{
    Eigen::VectorXd V(g.N);
    for (int p = 0; p < g.N; ++p)
        V(p) = -Z / std::sqrt(g.x(p) * g.x(p) + 1.0);
    return V;
}

// Apply H = T + diag(V) to a vector
inline Eigen::VectorXd apply_H(const Eigen::SparseMatrix<double> &T,
                               const Eigen::VectorXd &V,
                               const Eigen::VectorXd &c)
{
    Eigen::VectorXd out = (T * c).eval();
    out.array() += V.array() * c.array();
    return out;
}

// Single imaginary-time RK4 step:  ∂_τ c = -H c
// followed by L2 normalization.  Mirrors FEDVR/H_fedvr.hpp::rk4_imag_step.
inline void rk4_imag_step(const Eigen::SparseMatrix<double> &T,
                          const Eigen::VectorXd &V,
                          Eigen::VectorXd &c,
                          double dtau)
{
    auto F = [&](const Eigen::VectorXd &v) -> Eigen::VectorXd {
        return -apply_H(T, V, v);
    };

    const Eigen::VectorXd k1 = F(c);
    const Eigen::VectorXd k2 = F(c + 0.5 * dtau * k1);
    const Eigen::VectorXd k3 = F(c + 0.5 * dtau * k2);
    const Eigen::VectorXd k4 = F(c + dtau * k3);

    c += (dtau / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    c /= c.norm();
}

inline double energy(const Eigen::SparseMatrix<double> &T,
                     const Eigen::VectorXd &V,
                     const Eigen::VectorXd &c)
{
    return c.dot(apply_H(T, V, c));
}

} // namespace

int main()
{
    // ---- Parameters ----
    const double x_range = 15.0;
    const double dx = 0.2;
    const double Z = 1.0;
    const double dtau = 0.05;
    const int max_itr = 50000;
    const double thresh = 1e-13;

    std::cout << std::scientific << std::setprecision(12);

    // ---- Build FD grid ----
    fd::FDGrid grid = fd::make_grid(x_range, dx);
    std::cout << "FD grid: x_range=" << grid.x_range
              << ", dx=" << grid.dx
              << ", N(DOF)=" << grid.N << "\n";
    std::cout << "T nnz: " << grid.T.nonZeros() << "\n";

    // ---- One-electron potential ----
    Eigen::VectorXd V = one_electron_potential(grid, Z);

    // ============================================================
    // (A) Direct diagonalization
    // ============================================================
    {
        Eigen::MatrixXd H = Eigen::MatrixXd(grid.T);
        for (int p = 0; p < grid.N; ++p)
            H(p, p) += V(p);
        // Symmetrize (T is constructed symmetric up to floating-point noise).
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
        const double sym = (H - H.transpose()).cwiseAbs().maxCoeff();
        std::cout << "  max|H - H^T| = " << sym << "\n";
    }

    // ============================================================
    // (B) Imaginary-time RK4
    // ============================================================
    {
        Eigen::VectorXd c(grid.N);
        for (int p = 0; p < grid.N; ++p)
        {
            const double psi = std::exp(-grid.x(p) * grid.x(p));
            c(p) = std::sqrt(grid.w(p)) * psi;
        }
        c /= c.norm();

        std::cout << "\n[Imaginary-time RK4]  dtau=" << dtau
                  << ", max_itr=" << max_itr
                  << ", thresh=" << thresh << "\n";

        double prev_E = 1e30;
        for (int it = 0; it < max_itr; ++it)
        {
            const double E = energy(grid.T, V, c);
            const double diff = std::abs(E - prev_E);
            if (it % 200 == 0)
                std::cout << "  itr=" << it << ", E=" << E << ", diff=" << diff << "\n";
            if (it > 0 && diff < thresh)
            {
                std::cout << "  converged at itr=" << it
                          << ", E=" << E
                          << ", diff=" << diff << "\n";
                break;
            }
            prev_E = E;
            rk4_imag_step(grid.T, V, c, dtau);
        }
        const double E_final = energy(grid.T, V, c);
        std::cout << "  E_final = " << E_final << "\n";
    }

    return 0;
}
