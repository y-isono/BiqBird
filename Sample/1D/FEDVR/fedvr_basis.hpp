#pragma once
//
// fedvr_basis.hpp
//
// FEDVR (Finite-Element Discrete Variable Representation) on a 1D segment [-L, L].
// - Equal-size finite elements
// - Gauss-Lobatto-Legendre (GLL) quadrature within each element
// - Lagrange interpolation polynomials as local basis
// - Bridge functions to enforce C^0 continuity across element boundaries
// - Dirichlet (zero) boundary conditions at x = -L and x = +L
//
// See ../FEDVR/README.md for the mathematical formulation.
//

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace fedvr
{

constexpr double PI = 3.14159265358979323846;

// ----------------------------------------------------------------------------
// Legendre polynomial P_n(x) and its derivative P'_n(x), via Bonnet recursion.
// Returns std::pair{P_n(x), P'_n(x)}.
// ----------------------------------------------------------------------------
inline std::pair<double, double> legendre_and_deriv(int n, double x)
{
    if (n == 0)
        return {1.0, 0.0};
    if (n == 1)
        return {x, 1.0};

    double Pkm1 = 1.0;     // P_0
    double Pk = x;         // P_1
    double dPkm1 = 0.0;    // P'_0
    double dPk = 1.0;      // P'_1
    for (int k = 1; k < n; ++k)
    {
        const double Pkp1 = ((2.0 * k + 1.0) * x * Pk - k * Pkm1) / (k + 1.0);
        // dP_{k+1} = (k+1) P_k + x dP_k  (for Legendre)
        // Equivalent recursion: (1-x^2) P'_{k+1} = (k+1)(P_k - x P_{k+1})
        // Use the simpler explicit form:
        const double dPkp1 = dPkm1 + (2.0 * k + 1.0) * Pk;
        Pkm1 = Pk;
        Pk = Pkp1;
        dPkm1 = dPk;
        dPk = dPkp1;
    }
    return {Pk, dPk};
}

// ----------------------------------------------------------------------------
// Gauss-Lobatto-Legendre nodes/weights on the reference interval [-1, 1].
// n_order = polynomial order n; total nodes = n+1, including endpoints +-1.
// Interior nodes are roots of P'_n(x) found by Newton iteration.
// Weights: w_k = 2 / [n(n+1) P_n(xi_k)^2]
// Returns (nodes, weights), both sized n+1, sorted ascending.
// ----------------------------------------------------------------------------
inline std::pair<Eigen::VectorXd, Eigen::VectorXd> gll_nodes_weights(int n_order)
{
    if (n_order < 1)
        throw std::invalid_argument("GLL order must be >= 1");

    const int N = n_order + 1; // number of nodes
    Eigen::VectorXd xi(N);
    Eigen::VectorXd w(N);

    // Endpoints
    xi(0) = -1.0;
    xi(N - 1) = 1.0;

    if (n_order == 1)
    {
        w(0) = 1.0;
        w(1) = 1.0;
        return {xi, w};
    }

    // Interior nodes: roots of P'_n on (-1, 1).
    // Initial guesses from Chebyshev-like distribution.
    for (int k = 1; k < N - 1; ++k)
    {
        // Tuned initial guess for GLL interior nodes
        double x = -std::cos((k + 0.25) * PI / n_order
                             - 3.0 / (8.0 * n_order * PI * (k + 0.25)));

        // Newton iteration: solve P'_n(x) = 0
        // We need P'_n and P''_n. Use the differential identity:
        //   (1 - x^2) P''_n - 2 x P'_n + n(n+1) P_n = 0
        //   => P''_n = [2 x P'_n - n(n+1) P_n] / (1 - x^2)
        for (int it = 0; it < 100; ++it)
        {
            auto [Pn, dPn] = legendre_and_deriv(n_order, x);
            const double d2Pn = (2.0 * x * dPn - n_order * (n_order + 1) * Pn) / (1.0 - x * x);
            const double dx = -dPn / d2Pn;
            x += dx;
            if (std::abs(dx) < 1e-15)
                break;
        }
        xi(k) = x;
    }

    // Weights: w_k = 2 / [n(n+1) P_n(xi_k)^2]
    for (int k = 0; k < N; ++k)
    {
        const auto [Pn, dPn] = legendre_and_deriv(n_order, xi(k));
        (void)dPn;
        w(k) = 2.0 / (n_order * (n_order + 1) * Pn * Pn);
    }

    return {xi, w};
}

// ----------------------------------------------------------------------------
// Lagrange differentiation matrix on GLL nodes (reference interval [-1, 1]).
// hatD(m, k) = f'_k(xi_m), where f_k is the Lagrange polynomial through nodes xi.
// Closed-form result for GLL nodes:
//   m != k          : P_n(xi_m) / [P_n(xi_k) (xi_m - xi_k)]
//   m == k == 0     : -n(n+1)/4
//   m == k == n     : +n(n+1)/4
//   otherwise (m==k internal): 0
// ----------------------------------------------------------------------------
inline Eigen::MatrixXd gll_diff_matrix(int n_order, const Eigen::VectorXd &xi)
{
    const int N = n_order + 1;
    Eigen::MatrixXd D(N, N);
    Eigen::VectorXd Pn(N);
    for (int k = 0; k < N; ++k)
    {
        Pn(k) = legendre_and_deriv(n_order, xi(k)).first;
    }
    for (int m = 0; m < N; ++m)
    {
        for (int k = 0; k < N; ++k)
        {
            if (m == k)
            {
                if (m == 0)
                    D(m, k) = -0.25 * n_order * (n_order + 1);
                else if (m == N - 1)
                    D(m, k) = 0.25 * n_order * (n_order + 1);
                else
                    D(m, k) = 0.0;
            }
            else
            {
                D(m, k) = Pn(m) / (Pn(k) * (xi(m) - xi(k)));
            }
        }
    }
    return D;
}

// ============================================================================
// FEDVRGrid: builds the 1D FEDVR basis on [-L, L] with Dirichlet BC at +-L.
//
// Public attributes after construction:
//   N           : total number of independent DVR basis functions
//                 = n_elements * n_order - 1
//   x(N)        : DVR node coordinates (sorted ascending; bridge points appear once)
//   w(N)        : DVR weights (bridge points: sum of two adjacent element weights)
//   T           : kinetic-energy matrix in the DVR basis (Eigen::SparseMatrix<double>)
//                 T_{αβ} = (1/2) ∫ φ'_α φ'_β dx, symmetric, banded with bandwidth ~ 2 n_order + 1
//
// Convention: basis functions are normalized so that <φ_α | φ_β> ≈ δ_{αβ}
//             (using the GLL quadrature induced by the same nodes).
// Wavefunction in DVR coefficients:
//   c_α = √(w_α) ψ(x_α),     ψ(x) = Σ_α c_α φ_α(x)
//   <ψ|ψ> = Σ |c_α|^2
// ============================================================================
class FEDVRGrid
{
public:
    double L;       // half-range; domain is [-L, L]
    int n_elements; // number of finite elements
    int n_order;    // GLL order per element (n_order + 1 nodes per element)
    int N;          // total DVR DOF (after Dirichlet trim and bridge merging)

    Eigen::VectorXd x; // global DVR node coordinates, size N
    Eigen::VectorXd w; // global DVR weights, size N
    Eigen::SparseMatrix<double> T; // kinetic-energy matrix, N x N

    FEDVRGrid(double L_, int n_elements_, int n_order_)
        : L(L_), n_elements(n_elements_), n_order(n_order_)
    {
        if (L <= 0.0)
            throw std::invalid_argument("FEDVRGrid: L must be > 0");
        if (n_elements < 1)
            throw std::invalid_argument("FEDVRGrid: n_elements must be >= 1");
        if (n_order < 1)
            throw std::invalid_argument("FEDVRGrid: n_order must be >= 1");

        build();
    }

private:
    // raw GLL data on reference interval [-1, 1]
    Eigen::VectorXd xi_ref_; // (n_order+1)
    Eigen::VectorXd w_ref_;  // (n_order+1)
    Eigen::MatrixXd D_ref_;  // (n_order+1, n_order+1)

    // Map a local node (element_index, k_local in [0, n_order]) to the global
    // DVR index. Endpoints of -L and +L return -1 (Dirichlet trimmed).
    int global_index_(int elem, int k_local) const
    {
        // element 0 contributes nodes 0..n_order; node 0 is -L (trimmed)
        // node n_order is shared with element 1 (bridge unless first/last element boundary at ±L)
        // We adopt the following layout for global index α:
        //   - α counts unique DVR points strictly inside (-L, L), in ascending x
        //   - Element 0 internal points (k=1..n_order-1): contiguous block
        //   - Bridge between elem 0 and elem 1 (= node n_order of elem 0 = node 0 of elem 1)
        //   - Element 1 internal points
        //   - Bridge between elem 1 and elem 2
        //   - ... etc.
        //   - Last element internal points
        //   - (right endpoint +L is trimmed)

        if (elem < 0 || elem >= n_elements)
            return -1;
        if (k_local < 0 || k_local > n_order)
            return -1;

        // Trim global endpoints
        if (elem == 0 && k_local == 0)
            return -1; // x = -L
        if (elem == n_elements - 1 && k_local == n_order)
            return -1; // x = +L

        // Each element contributes (n_order - 1) internal indices, then a bridge
        // (except after the last element).
        // Block layout per element e (for e = 0..n_elements-1):
        //   internal indices: starts at base_internal(e), length n_order-1
        //                     covers k_local = 1, 2, ..., n_order-1
        //   bridge to next  : single index at base_internal(e) + (n_order-1)
        //                     covers k_local = n_order of elem e
        //                     == k_local = 0    of elem e+1
        // For the very last element, no bridge follows.

        const int per_element_block = (n_order - 1) + 1; // internals + bridge
        // ... but the last element has no trailing bridge.

        if (k_local >= 1 && k_local <= n_order - 1)
        {
            // Internal node of element 'elem'
            return elem * per_element_block + (k_local - 1);
        }
        if (k_local == n_order)
        {
            // Right endpoint of element elem -> bridge to elem+1 (or +L, trimmed)
            if (elem == n_elements - 1)
                return -1;
            return elem * per_element_block + (n_order - 1);
        }
        if (k_local == 0)
        {
            // Left endpoint of element elem -> bridge to elem-1 (or -L, trimmed)
            if (elem == 0)
                return -1;
            // Same as right-endpoint of (elem-1)
            return (elem - 1) * per_element_block + (n_order - 1);
        }
        return -1;
    }

    void build()
    {
        // ---- Reference GLL data ----
        std::tie(xi_ref_, w_ref_) = gll_nodes_weights(n_order);
        D_ref_ = gll_diff_matrix(n_order, xi_ref_);

        // ---- Global DOF count ----
        // per_element_block = (n_order - 1) internals + 1 bridge after element
        //   total over n_elements = n_elements * n_order, then subtract the
        //   trailing bridge that doesn't exist after last element
        //   => n_elements * n_order - 1 ?  Let's recount:
        //
        // Direct count:
        //   internal nodes total : n_elements * (n_order - 1)
        //   bridge nodes         : n_elements - 1
        //   total                : n_elements * (n_order - 1) + (n_elements - 1)
        //                        = n_elements * n_order - 1
        N = n_elements * n_order - 1;

        x = Eigen::VectorXd::Zero(N);
        w = Eigen::VectorXd::Zero(N);

        const double dx_elem = 2.0 * L / n_elements;
        const double half = 0.5 * dx_elem;

        // ---- Build x and w from the layout ----
        for (int e = 0; e < n_elements; ++e)
        {
            const double xc = -L + (e + 0.5) * dx_elem; // element center
            for (int k = 0; k <= n_order; ++k)
            {
                const int g = global_index_(e, k);
                if (g < 0)
                    continue;
                const double xk = xc + half * xi_ref_(k);
                const double wk = half * w_ref_(k);

                if (k == 0 || k == n_order)
                {
                    // bridge (or trimmed; trimmed already returned -1)
                    // x is the same from both sides; just set; w accumulates.
                    x(g) = xk;
                    w(g) += wk;
                }
                else
                {
                    x(g) = xk;
                    w(g) = wk;
                }
            }
        }

        // ---- Build kinetic-energy matrix T ----
        // Element-local kinetic matrix (Lagrange basis, size n+1 x n+1):
        //   T^{(e)}_{kl} = (1/2) Σ_m w^{(e)}_m D^{(e)}_{mk} D^{(e)}_{ml}
        //   with D^{(e)} = (2/dx_elem) hatD,  w^{(e)}_m = (dx_elem/2) hat w_m
        // Then assemble into global T with bridge normalization.
        //
        // Mapping: for local k of element e -> global index g (or -1 if trimmed)
        // The DVR-basis-normalized matrix elements are:
        //   T_global[g(k), g(l)] += T^{(e)}_{kl} / sqrt(w_global[g(k)] * w_global[g(l)])
        // with the caveat that for bridge points, w_global is the sum from both
        // sides; the local Lagrange-basis weight is w^{(e)}_k. Substituting the
        // bridge-normalized basis χ_i = (f_n + f_0)/sqrt(w_left + w_right) gives
        // exactly the same formula -> the assembly rule above is uniform.

        std::vector<Eigen::Triplet<double>> trips;
        trips.reserve(static_cast<size_t>(n_elements) * (n_order + 1) * (n_order + 1));

        const double scale_D = 2.0 / dx_elem; // physical D = scale_D * hatD
        const double scale_w = 0.5 * dx_elem; // physical w = scale_w * hat_w
        // T^{(e)}_{kl} = (1/2) Σ_m (scale_w * hat_w_m)
        //              * (scale_D * hatD_{mk}) * (scale_D * hatD_{ml})
        //             = (1/2) * scale_w * scale_D^2 * Σ_m hat_w_m hatD_{mk} hatD_{ml}
        const double T_scale = 0.5 * scale_w * scale_D * scale_D;

        // Precompute reference local kinetic matrix hatT
        Eigen::MatrixXd hatT = Eigen::MatrixXd::Zero(n_order + 1, n_order + 1);
        for (int k = 0; k <= n_order; ++k)
        {
            for (int l = 0; l <= n_order; ++l)
            {
                double s = 0.0;
                for (int m = 0; m <= n_order; ++m)
                {
                    s += w_ref_(m) * D_ref_(m, k) * D_ref_(m, l);
                }
                hatT(k, l) = s; // before T_scale and DVR normalization
            }
        }

        for (int e = 0; e < n_elements; ++e)
        {
            for (int k = 0; k <= n_order; ++k)
            {
                const int gk = global_index_(e, k);
                if (gk < 0)
                    continue;
                for (int l = 0; l <= n_order; ++l)
                {
                    const int gl = global_index_(e, l);
                    if (gl < 0)
                        continue;
                    const double Tloc = T_scale * hatT(k, l);
                    // DVR basis is f_local / sqrt(w_global) for internal,
                    // χ_i = (f + f) / sqrt(w_global) for bridge -> both yield this
                    // assembly rule:
                    const double val = Tloc / std::sqrt(w(gk) * w(gl));
                    trips.emplace_back(gk, gl, val);
                }
            }
        }

        T.resize(N, N);
        T.setFromTriplets(trips.begin(), trips.end()); // duplicates summed -> bridge contributions accumulate
        T.makeCompressed();
    }
};

} // namespace fedvr
