#pragma once
//
// fd_basis.hpp
//
// 1D finite-difference (FD) "basis" for the soft-Coulomb model.
// Equispaced grid on [-x_range, x_range] with Dirichlet (zero) boundary
// conditions at both ends.
//
// Mirrors the structure of FEDVR/fedvr_basis.hpp:
//   - Grid struct exposing x[N], w[N], and the kinetic-energy matrix T
//   - build_h_pq(grid, Z) and build_V_pq(grid) for second-quantization
//
// Conventions (parallel to fedvr_basis.hpp):
//   - Wavefunction is stored as the *normalized point value*
//       c_p = sqrt(w_p) * psi(x_p),   psi(x) = Σ_p c_p φ_p(x)
//     with φ_p the FD "basis" function having φ_p(x_q) = δ_{pq}/sqrt(w_p)
//     (formally a sinc-like function on the equispaced grid; for FD we
//      treat it as an equally-weighted DVR with w_p = dx for all p).
//   - <ψ|ψ> ≈ Σ_p |c_p|^2     (so plain L2 norm of c)
//   - Local operators V(x) act diagonally:
//       V_{pq} ≈ V(x_p) δ_{pq}
//   - The two-electron integral collapses to two-index:
//       V_{pqrs} ≈ δ_{pr} δ_{qs} · w(x_p, x_q)         (DVR-like approximation)
//     Same form as in FEDVR; the only difference is that w_p = dx is
//     uniform here.
//
// Kinetic-energy operator: 2nd-order central difference,
//     T_{pq} = (1/(2 dx^2)) (2 δ_{pq} - δ_{p,q+1} - δ_{p,q-1})
// stored as a sparse symmetric tridiagonal matrix.  Dirichlet BCs are
// enforced by simply not including any sites at p < 0 or p ≥ N (the
// off-grid neighbours of the first/last site contribute 0).
//

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace fd
{

// ----------------------------------------------------------------------------
// FDGrid : equispaced grid on [-x_range, +x_range] with Dirichlet BCs.
//
// We discretize at the *interior* points
//     x_p = -x_range + (p + 1) * dx,    p = 0, 1, ..., N - 1
// so that the boundary points x = ±x_range carry zero amplitude
// (Dirichlet) and are not part of the basis.  This matches the FEDVR
// convention that endpoints are trimmed.
//
// N is determined from x_range and the *target* dx:
//     N = round(2 x_range / dx) - 1
// and dx is then re-derived as 2 x_range / (N + 1) so that endpoints fall
// exactly on ±x_range.
// ----------------------------------------------------------------------------
struct FDGrid
{
    double x_range; // half-range; domain is [-x_range, x_range]
    double dx;      // grid spacing
    int N;          // number of interior grid points (= DOF)

    Eigen::VectorXd x;             // (N) coordinate of each interior point
    Eigen::VectorXd w;             // (N) DVR-like weight; w_p = dx for all p
    Eigen::SparseMatrix<double> T; // (N x N) kinetic-energy matrix
};

// ----------------------------------------------------------------------------
// Build an FD grid + kinetic-energy matrix.
// ----------------------------------------------------------------------------
inline FDGrid make_grid(double x_range, double dx_target)
{
    if (x_range <= 0.0)
        throw std::invalid_argument("fd::make_grid: x_range must be > 0");
    if (dx_target <= 0.0)
        throw std::invalid_argument("fd::make_grid: dx_target must be > 0");

    FDGrid g;
    g.x_range = x_range;
    // N interior points + 2 Dirichlet endpoints; total intervals = N + 1.
    const int N_target = static_cast<int>(std::round(2.0 * x_range / dx_target) - 1.0);
    g.N = std::max(N_target, 1);
    g.dx = 2.0 * x_range / (g.N + 1);

    g.x = Eigen::VectorXd(g.N);
    g.w = Eigen::VectorXd::Constant(g.N, g.dx);
    for (int p = 0; p < g.N; ++p)
        g.x(p) = -x_range + (p + 1) * g.dx;

    // ---- Kinetic-energy matrix (2nd-order central finite difference) ----
    //   T_{pq} = (1/2) ∫ φ'_p φ'_q dx
    // For the equispaced FD basis with the DVR-like normalisation
    // c_p = sqrt(dx) ψ(x_p), the FD Laplacian gives
    //   T_{pp}     = +1 / dx^2
    //   T_{p,p±1}  = -1 / (2 dx^2)
    // (i.e. T = (1/(2 dx^2)) tridiag(-1, 2, -1) in the standard convention).
    //
    // Note: there is no extra DVR-weight rescaling here because w_p = dx
    // is uniform; equivalently, the FD basis functions are already
    // implicitly normalized so that ⟨φ_p|φ_q⟩ ≈ δ_{pq}.
    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(static_cast<size_t>(3 * g.N));

    const double a = 1.0 / (g.dx * g.dx);
    const double b = -0.5 / (g.dx * g.dx);
    for (int p = 0; p < g.N; ++p)
    {
        trips.emplace_back(p, p, a);
        if (p + 1 < g.N)
        {
            trips.emplace_back(p, p + 1, b);
            trips.emplace_back(p + 1, p, b);
        }
    }
    g.T.resize(g.N, g.N);
    g.T.setFromTriplets(trips.begin(), trips.end());
    g.T.makeCompressed();

    return g;
}

// ============================================================================
// Second-quantization integral tensors in the FD basis.
//
// Same logical structure as FEDVR/fedvr_basis.hpp::build_h_pq / build_V_pq:
//   h_{pq} = T_{pq} + V_ne(x_p) δ_{pq}
//   V_{pq} = w(x_p, x_q) = 1 / sqrt((x_p - x_q)^2 + 1)
// (the four-index two-electron integral is δ_{pr} δ_{qs} V_{pq} in the
//  DVR-like FD basis with uniform weight w_p = dx).
//
// These are exact analogues of the FEDVR functions and let us reuse the
// FEDVR closed-shell RHF solver (FEDVR/hf_fedvr.hpp) as-is on the FD grid.
// ============================================================================

// One-electron Hamiltonian: h_{pq} = T_{pq} + V_ne(x_p) δ_{pq}
// V_ne(x) = -Z / sqrt(x^2 + 1)   (1D soft-Coulomb nucleus at the origin)
inline Eigen::SparseMatrix<double> build_h_pq(const FDGrid &g, double Z)
{
    Eigen::SparseMatrix<double> h = g.T; // start from kinetic-energy matrix
    for (int p = 0; p < g.N; ++p)
    {
        const double Vp = -Z / std::sqrt(g.x(p) * g.x(p) + 1.0);
        h.coeffRef(p, p) += Vp;
    }
    h.makeCompressed();
    return h;
}

// Two-electron pair integrals:
//   V_{pq} = w(x_p, x_q) = 1 / sqrt((x_p - x_q)^2 + 1)
inline Eigen::MatrixXd build_V_pq(const FDGrid &g)
{
    Eigen::MatrixXd V(g.N, g.N);
    for (int p = 0; p < g.N; ++p)
    {
        for (int q = 0; q < g.N; ++q)
        {
            const double dx = g.x(p) - g.x(q);
            V(p, q) = 1.0 / std::sqrt(dx * dx + 1.0);
        }
    }
    return V;
}

} // namespace fd
