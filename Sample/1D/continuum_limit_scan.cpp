//
// continuum_limit_scan.cpp
//
// FD と FEDVR の連続極限一致を、SCF 解で確認する検証プログラム。
//
//   - He（Z=2, N_occ=1）と Be（Z=4, N_occ=2）について、
//     * FD: dx を細かくしていく
//     * FEDVR: GLL 多項式次数 n を上げていく
//     を独立にスキャンし、両者の RHF SCF エネルギーが共通の連続極限値に
//     収束していくことを示す。
//
//   - すべて FEDVR の `fedvr::rhf_scf` ソルバーを使う（FEDVR/FD いずれも
//     同じ Roothaan-Hall 対角化スキーム）。
//
// Build:
//   EIGEN=/opt/homebrew/include/eigen3
//   g++ -std=c++17 -O2 -I"$EIGEN" -I"FD" -I"FEDVR" continuum_limit_scan.cpp -o out/continuum_limit_scan
//   ./out/continuum_limit_scan
//

#include "FD/fd_basis.hpp"
#include "FEDVR/fedvr_basis.hpp"
#include "FEDVR/hf_fedvr.hpp"

#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace
{

struct AtomSpec
{
    std::string name;
    double Z;
    int N_occ;
    double x_range;        // box half-width
    int Ne_default;        // FEDVR n_elements default
};

// FD を SCF で解いて E_RHF を返す（時間も計測）
struct ScanRow
{
    std::string scheme;        // "FD" or "FEDVR"
    std::string param_label;   // e.g. "dx=0.4" or "n=8"
    int N;                     // DOF
    double E;                  // RHF SCF energy
    int iters;
    double seconds;
};

ScanRow run_fd(const AtomSpec &atom, double dx)
{
    auto t0 = std::chrono::steady_clock::now();
    fd::FDGrid grid = fd::make_grid(atom.x_range, dx);
    Eigen::SparseMatrix<double> h_pq = fd::build_h_pq(grid, atom.Z);
    Eigen::MatrixXd V_pq = fd::build_V_pq(grid);

    // initial guess: even/odd Gaussians
    Eigen::MatrixXd C_init(grid.N, atom.N_occ);
    for (int p = 0; p < grid.N; ++p)
    {
        const double xp = grid.x(p);
        const double sw = std::sqrt(grid.w(p));
        const double g = std::exp(-xp * xp);
        C_init(p, 0) = sw * g;
        if (atom.N_occ >= 2)
            C_init(p, 1) = sw * xp * g;
    }
    for (int j = 0; j < atom.N_occ; ++j)
        C_init.col(j) /= C_init.col(j).norm();

    fedvr::RHF_SCFResult res = fedvr::rhf_scf(
        h_pq, V_pq, atom.N_occ, C_init, /*max_itr=*/500, /*thresh=*/1e-13);

    auto t1 = std::chrono::steady_clock::now();
    const double sec = std::chrono::duration<double>(t1 - t0).count();

    ScanRow row;
    row.scheme = "FD";
    {
        std::ostringstream oss;
        oss << "dx=" << dx;
        row.param_label = oss.str();
    }
    row.N = grid.N;
    row.E = res.E_RHF;
    row.iters = res.iterations;
    row.seconds = sec;
    return row;
}

ScanRow run_fedvr(const AtomSpec &atom, int Ne, int n_order)
{
    auto t0 = std::chrono::steady_clock::now();
    fedvr::FEDVRGrid grid(atom.x_range, Ne, n_order);
    Eigen::SparseMatrix<double> h_pq = fedvr::build_h_pq(grid, atom.Z);
    Eigen::MatrixXd V_pq = fedvr::build_V_pq(grid);

    Eigen::MatrixXd C_init(grid.N, atom.N_occ);
    for (int p = 0; p < grid.N; ++p)
    {
        const double xp = grid.x(p);
        const double sw = std::sqrt(grid.w(p));
        const double g = std::exp(-xp * xp);
        C_init(p, 0) = sw * g;
        if (atom.N_occ >= 2)
            C_init(p, 1) = sw * xp * g;
    }
    for (int j = 0; j < atom.N_occ; ++j)
        C_init.col(j) /= C_init.col(j).norm();

    fedvr::RHF_SCFResult res = fedvr::rhf_scf(
        h_pq, V_pq, atom.N_occ, C_init, /*max_itr=*/500, /*thresh=*/1e-13);

    auto t1 = std::chrono::steady_clock::now();
    const double sec = std::chrono::duration<double>(t1 - t0).count();

    ScanRow row;
    row.scheme = "FEDVR";
    {
        std::ostringstream oss;
        oss << "Ne=" << Ne << ",n=" << n_order;
        row.param_label = oss.str();
    }
    row.N = grid.N;
    row.E = res.E_RHF;
    row.iters = res.iterations;
    row.seconds = sec;
    return row;
}

void print_table(const std::string &title, const std::vector<ScanRow> &rows)
{
    std::cout << "\n=== " << title << " ===\n";
    std::cout << std::left
              << std::setw(8) << "scheme"
              << std::setw(20) << "param"
              << std::setw(8) << "N"
              << std::setw(22) << "E_RHF [Ha]"
              << std::setw(8) << "iters"
              << std::setw(10) << "time[s]"
              << "\n";
    std::cout << std::string(76, '-') << "\n";
    for (const auto &r : rows)
    {
        std::cout << std::left
                  << std::setw(8) << r.scheme
                  << std::setw(20) << r.param_label
                  << std::setw(8) << r.N
                  << std::scientific << std::setprecision(11) << std::setw(22) << r.E
                  << std::setw(8) << r.iters
                  << std::fixed << std::setprecision(3) << std::setw(10) << r.seconds
                  << "\n";
    }
}

void scan_atom(const AtomSpec &atom,
               const std::vector<double> &fd_dxs,
               const std::vector<int> &fedvr_orders)
{
    std::vector<ScanRow> rows;

    for (double dx : fd_dxs)
        rows.push_back(run_fd(atom, dx));
    for (int n : fedvr_orders)
        rows.push_back(run_fedvr(atom, atom.Ne_default, n));

    print_table(atom.name + "  (Z=" + std::to_string((int)atom.Z) +
                ", N_occ=" + std::to_string(atom.N_occ) +
                ", x_range=" + std::to_string((int)atom.x_range) + ")",
                rows);
}

} // namespace

int main()
{
    std::cout << "Continuum-limit scan: FD (dx) vs FEDVR (n_order)\n";
    std::cout << "All energies obtained via fedvr::rhf_scf (Roothaan-Hall SCF, thresh=1e-13).\n";

    // -------- He --------
    {
        AtomSpec He{"He", 2.0, 1, 20.0, /*Ne_default=*/10};
        std::vector<double> fd_dxs   = {0.4, 0.2, 0.1, 0.05, 0.025};
        std::vector<int> fedvr_n     = {4, 6, 8, 10, 12};
        scan_atom(He, fd_dxs, fedvr_n);
    }

    // -------- Be --------
    {
        AtomSpec Be{"Be", 4.0, 2, 20.0, /*Ne_default=*/10};
        std::vector<double> fd_dxs   = {0.2, 0.1, 0.05, 0.025};
        std::vector<int> fedvr_n     = {4, 6, 8, 10, 12};
        scan_atom(Be, fd_dxs, fedvr_n);
    }

    std::cout << "\nNote:\n";
    std::cout << "  As dx -> 0 (FD) and n_order is increased (FEDVR), both schemes\n";
    std::cout << "  should converge to the same continuum-limit RHF energy.\n";
    return 0;
}
