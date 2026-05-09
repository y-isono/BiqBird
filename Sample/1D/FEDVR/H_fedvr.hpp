#pragma once
//
// H_fedvr.hpp
//
// Hydrogen-like (one-electron) FEDVR utilities for the 1D soft-Coulomb model
//   ĥ_Z(x) = -1/2 d²/dx² - Z / sqrt(x² + 1)
//
// Provides:
//   - one_electron_potential(grid, Z)            -> diagonal V vector (real)
//   - one_electron_hamiltonian_dense(grid, Z)    -> dense H matrix (real, symmetric)
//   - apply_one_electron_H(grid, V, c)           -> H * c  (using sparse T + diag V)
//   - rk4_imag_step(grid, V, c, dtau)            -> single RK4 imag-time step + L2 normalize
//

#include "fedvr_basis.hpp"
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace fedvr
{

// Diagonal one-electron potential V(x_α) = -Z / sqrt(x_α² + 1)
inline Eigen::VectorXd one_electron_potential(const FEDVRGrid &g, double Z)
{
    Eigen::VectorXd V(g.N);
    for (int a = 0; a < g.N; ++a)
        V(a) = -Z / std::sqrt(g.x(a) * g.x(a) + 1.0);
    return V;
}

// Dense one-electron Hamiltonian (small N -> OK).
inline Eigen::MatrixXd one_electron_hamiltonian_dense(const FEDVRGrid &g, double Z)
{
    Eigen::MatrixXd H = Eigen::MatrixXd(g.T);
    Eigen::VectorXd V = one_electron_potential(g, Z);
    for (int a = 0; a < g.N; ++a)
        H(a, a) += V(a);
    return H;
}

// H * c using sparse T and diagonal V.
inline Eigen::VectorXd apply_one_electron_H(const FEDVRGrid &g,
                                            const Eigen::VectorXd &V,
                                            const Eigen::VectorXd &c)
{
    // Force evaluation to a concrete VectorXd before mixing with array ops.
    Eigen::VectorXd out = (g.T * c).eval();
    out.array() += V.array() * c.array();
    return out;
}

// One imaginary-time RK4 step:  ∂_τ c = -H c
// followed by L2 normalization of c (DVR coefficients are L2-normalized).
inline void rk4_imag_step(const FEDVRGrid &g,
                          const Eigen::VectorXd &V,
                          Eigen::VectorXd &c,
                          double dtau)
{
    auto F = [&](const Eigen::VectorXd &v) -> Eigen::VectorXd {
        Eigen::VectorXd Hv = apply_one_electron_H(g, V, v);
        return -Hv;
    };

    const Eigen::VectorXd k1 = F(c);
    const Eigen::VectorXd y2 = c + (0.5 * dtau) * k1;
    const Eigen::VectorXd k2 = F(y2);
    const Eigen::VectorXd y3 = c + (0.5 * dtau) * k2;
    const Eigen::VectorXd k3 = F(y3);
    const Eigen::VectorXd y4 = c + dtau * k3;
    const Eigen::VectorXd k4 = F(y4);

    c += (dtau / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    c /= c.norm();
}

// Energy expectation <c|H|c> for L2-normalized c.
inline double energy(const FEDVRGrid &g, const Eigen::VectorXd &V, const Eigen::VectorXd &c)
{
    return c.dot(apply_one_electron_H(g, V, c));
}

} // namespace fedvr
