#pragma once
//
// He_fedvr.hpp
//
// Closed-shell He (1D soft-Coulomb model, two electrons in a single spatial
// orbital ψ) on the FEDVR grid.
//
//   ĥ_2(x) = -1/2 d²/dx² - 2 / sqrt(x² + 1)        (one-electron part, Z=2)
//   w(x1,x2) = 1 / sqrt((x1-x2)² + 1)               (electron-electron interaction)
//
// Total energy for closed-shell two-electron state with single orbital ψ:
//   E[ψ] = 2 ⟨ψ|ĥ_2|ψ⟩
//        + ∫∫ |ψ(x1)|² w(x1,x2) |ψ(x2)|² dx1 dx2
//
// In DVR coefficients (with c_α = √(w_α) ψ(x_α)):
//   density at node α : ρ_α = |c_α|² / w_α          (so ∫ρ dx ≈ Σ_α |c_α|² = 1)
//   Hartree potential : W_α = Σ_β w(x_α, x_β) w_β ρ_β
//                            = Σ_β |c_β|² / sqrt((x_α - x_β)² + 1)
//   Two-electron energy : E2 = ⟨ρ|W⟩ = Σ_α |c_α|² W_α
//
// The mean-field (closed-shell HF / TDDFT-Hartree-only) Fock-like operator
// driving the imaginary-time evolution is:
//   F[c] = T + diag(V_ne) + diag(W[|c|²])
// (no factor of 2 in front of W: closed-shell two-electron Hartree on a single
//  orbital, exact for E2 = ⟨ρ|W|ρ⟩)
//
// Note: this is the Hartree-only mean-field model (no exchange).
//

#include "H_fedvr.hpp" // brings in fedvr_basis.hpp + one_electron_potential
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>

namespace fedvr
{

// W_αβ = 1 / sqrt((x_α - x_β)² + 1)  — full N×N kernel.
// (For modest N (~80) this is cheap to keep dense.)
inline Eigen::MatrixXd build_softcoulomb_kernel(const FEDVRGrid &g)
{
    Eigen::MatrixXd W(g.N, g.N);
    for (int a = 0; a < g.N; ++a)
        for (int b = 0; b < g.N; ++b)
            W(a, b) = 1.0 / std::sqrt((g.x(a) - g.x(b)) * (g.x(a) - g.x(b)) + 1.0);
    return W;
}

// Hartree potential at every DVR node, for a given DVR-coefficient vector c.
//   W_α = Σ_β |c_β|² / sqrt((x_α - x_β)² + 1)
inline Eigen::VectorXd hartree_potential(const FEDVRGrid &g,
                                         const Eigen::MatrixXd &Wkernel,
                                         const Eigen::VectorXd &c)
{
    // |c|^2 elementwise -> contracted with Wkernel
    Eigen::VectorXd c2 = c.array().square();
    return Wkernel * c2;
}

// Apply the two-electron mean-field Hamiltonian F[c] to a vector v:
//   F[c] = T + diag(V_ne) + diag(W[|c|²])
// with c representing the current orbital coefficients (used for density).
inline Eigen::VectorXd apply_He_F(const FEDVRGrid &g,
                                  const Eigen::VectorXd &Vne,
                                  const Eigen::VectorXd &Whar,
                                  const Eigen::VectorXd &v)
{
    Eigen::VectorXd out = (g.T * v).eval();
    out.array() += (Vne.array() + Whar.array()) * v.array();
    return out;
}

// Expectation values
struct Energies
{
    double E1;    // 2 * <c | ĥ_2 | c>  (sum over both electrons)
    double E2;    // <ρ | W_kernel | ρ>  (electron-electron repulsion energy)
    double Etot;  // E1 + E2
};

inline Energies compute_energies(const FEDVRGrid &g,
                                 const Eigen::VectorXd &Vne,
                                 const Eigen::MatrixXd &Wkernel,
                                 const Eigen::VectorXd &c)
{
    // One-electron expectation per electron
    Eigen::VectorXd Hc = (g.T * c).eval();
    Hc.array() += Vne.array() * c.array();
    const double e1_per_electron = c.dot(Hc);

    // Two-electron repulsion energy
    Eigen::VectorXd c2 = c.array().square();
    const double e2 = c2.dot(Wkernel * c2);

    Energies res;
    res.E1 = 2.0 * e1_per_electron;
    res.E2 = e2;
    res.Etot = res.E1 + res.E2;
    return res;
}

// One imaginary-time RK4 step for closed-shell He:
//   ∂_τ c = -F[c] c  ;  F[c] = T + V_ne + W[|c|²]
// followed by L2 normalization.
//
// Note: the Hartree potential depends on c, so it is recomputed for each
// of the four RK4 stages.
inline void rk4_imag_step_He(const FEDVRGrid &g,
                             const Eigen::VectorXd &Vne,
                             const Eigen::MatrixXd &Wkernel,
                             Eigen::VectorXd &c,
                             double dtau)
{
    auto F_minus = [&](const Eigen::VectorXd &v) -> Eigen::VectorXd {
        const Eigen::VectorXd Whar = hartree_potential(g, Wkernel, v);
        Eigen::VectorXd Fv = apply_He_F(g, Vne, Whar, v);
        return -Fv;
    };

    const Eigen::VectorXd k1 = F_minus(c);
    const Eigen::VectorXd y2 = c + (0.5 * dtau) * k1;
    const Eigen::VectorXd k2 = F_minus(y2);
    const Eigen::VectorXd y3 = c + (0.5 * dtau) * k2;
    const Eigen::VectorXd k3 = F_minus(y3);
    const Eigen::VectorXd y4 = c + dtau * k3;
    const Eigen::VectorXd k4 = F_minus(y4);

    c += (dtau / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    c /= c.norm();
}

} // namespace fedvr
