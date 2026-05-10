// export_integrals.cpp
//
// Build the FEDVR DVR basis on [-L, L] with given (n_elements, n_order) and
// dump the second-quantization integrals (h_pq, V_pq) plus the DVR grid (x, w)
// to a JSON file. The output is consumed by the Python side (biqbird_quantum)
// to construct a fermionic Hamiltonian, Jordan-Wigner transform it, and run
// direct diagonalization / iQPE / Trotter.
//
// Build:
//   EIGEN=/opt/homebrew/include/eigen3
//   g++ -std=c++17 -O2 -I"$EIGEN" -I"../FEDVR" \
//       export_integrals.cpp -o out/export_integrals
//
// Usage examples:
//   # H atom, 1 element, n=4 -> N_DVR=3, N_q = 6 (Phase Q1 Step 1)
//   ./out/export_integrals --L 5 --Ne 1 --n 4 --Z 1 --Nelec 1 \
//       --tag h_step1 --out data/h_step1.json
//
//   # H2 molecule, R=1.6
//   ./out/export_integrals --L 10 --Ne 2 --n 5 --Nelec 2 \
//       --H2 --R 1.6 --tag h2_step_R16 --out data/h2_step_R16.json
//
// JSON schema (see also Lab/BiqBird_Doc/quantum_computing_plan.md):
//   {
//     "model": {
//       "tag": "<string>",
//       "a": 1.0,                         // soft-Coulomb parameter
//       "L": 5.0,
//       "n_elements": 1,
//       "n_order": 4,
//       "N_DVR": 3,
//       "N_e": 1,                         // number of electrons
//       "nuclei": [{"Z": 1.0, "X": 0.0}, ...],
//       "E_nn": 0.0                       // nuclear repulsion sum (a=1 soft-Coulomb)
//     },
//     "x": [...],                          // DVR node coordinates (length N_DVR)
//     "w": [...],                          // DVR quadrature weights (length N_DVR)
//     "h_pq": [[...], ...],                // dense N_DVR x N_DVR (symmetric)
//     "V_pq": [[...], ...]                 // dense N_DVR x N_DVR (V_pq = w(x_p,x_q))
//   }
//

#include "../FEDVR/fedvr_basis.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// ----------------------------------------------------------------------------
// Minimal JSON writer (no external dependency).
// We only need to write nested objects/arrays of doubles, ints, and strings.
// ----------------------------------------------------------------------------
namespace mini_json
{

struct Writer
{
    std::ostringstream os;
    int indent_ = 0;

    void newline()
    {
        os << '\n';
        for (int i = 0; i < indent_; ++i)
            os << "  ";
    }

    Writer &begin_object()
    {
        os << '{';
        ++indent_;
        return *this;
    }
    Writer &end_object()
    {
        --indent_;
        newline();
        os << '}';
        return *this;
    }
    Writer &begin_array()
    {
        os << '[';
        return *this;
    }
    Writer &end_array()
    {
        os << ']';
        return *this;
    }

    Writer &key(const std::string &k)
    {
        newline();
        os << '"' << k << "\": ";
        return *this;
    }
    Writer &comma()
    {
        os << ',';
        return *this;
    }

    Writer &string_value(const std::string &v)
    {
        os << '"' << v << '"';
        return *this;
    }
    Writer &int_value(long long v)
    {
        os << v;
        return *this;
    }
    Writer &double_value(double v)
    {
        // Use full precision so the Python side reconstructs identical bits.
        std::ostringstream tmp;
        tmp << std::setprecision(17) << v;
        os << tmp.str();
        return *this;
    }

    Writer &vector_value(const Eigen::VectorXd &v)
    {
        os << '[';
        for (int i = 0; i < v.size(); ++i)
        {
            if (i > 0)
                os << ", ";
            std::ostringstream tmp;
            tmp << std::setprecision(17) << v(i);
            os << tmp.str();
        }
        os << ']';
        return *this;
    }
    Writer &matrix_value(const Eigen::MatrixXd &M)
    {
        os << '[';
        for (int i = 0; i < M.rows(); ++i)
        {
            if (i > 0)
                os << ',';
            os << "\n      [";
            for (int j = 0; j < M.cols(); ++j)
            {
                if (j > 0)
                    os << ", ";
                std::ostringstream tmp;
                tmp << std::setprecision(17) << M(i, j);
                os << tmp.str();
            }
            os << ']';
        }
        os << ']';
        return *this;
    }

    std::string str() const { return os.str(); }
};

} // namespace mini_json

// ----------------------------------------------------------------------------
// CLI parsing helpers (very small, since we don't want any extra dependency).
// ----------------------------------------------------------------------------
struct Args
{
    double L = 5.0;
    int Ne_elem = 1;   // n_elements
    int n_order = 4;
    double Z = 1.0;    // single-nucleus charge (used unless --H2)
    int N_e = 1;       // number of electrons
    bool H2 = false;
    double R = 1.6;    // H2 internuclear distance
    std::string out = "data/integrals.json";
    std::string tag = "untagged";
};

static void print_usage()
{
    std::cout <<
        "Usage: export_integrals [options]\n"
        "  --L      <double>   half-range of [-L, L] (default 5)\n"
        "  --Ne     <int>      number of finite elements (default 1)\n"
        "  --n      <int>      GLL order per element (default 4)\n"
        "  --Z      <double>   nuclear charge (single-atom case, default 1)\n"
        "  --Nelec  <int>      number of electrons (default 1)\n"
        "  --H2                use two H nuclei at +-R/2, Z=1 each\n"
        "  --R      <double>   H2 internuclear distance (default 1.6)\n"
        "  --tag    <string>   tag string written into JSON model.tag\n"
        "  --out    <path>     output JSON file path (default data/integrals.json)\n"
        "  --help              show this message\n";
}

static Args parse_args(int argc, char **argv)
{
    Args a;
    for (int i = 1; i < argc; ++i)
    {
        std::string s = argv[i];
        auto need = [&](const char *flag) -> std::string {
            if (i + 1 >= argc)
            {
                std::cerr << "Missing value after " << flag << "\n";
                std::exit(2);
            }
            return std::string(argv[++i]);
        };
        if (s == "--help" || s == "-h")
        {
            print_usage();
            std::exit(0);
        }
        else if (s == "--L")
            a.L = std::stod(need("--L"));
        else if (s == "--Ne")
            a.Ne_elem = std::stoi(need("--Ne"));
        else if (s == "--n")
            a.n_order = std::stoi(need("--n"));
        else if (s == "--Z")
            a.Z = std::stod(need("--Z"));
        else if (s == "--Nelec")
            a.N_e = std::stoi(need("--Nelec"));
        else if (s == "--H2")
            a.H2 = true;
        else if (s == "--R")
            a.R = std::stod(need("--R"));
        else if (s == "--out")
            a.out = need("--out");
        else if (s == "--tag")
            a.tag = need("--tag");
        else
        {
            std::cerr << "Unknown argument: " << s << "\n";
            print_usage();
            std::exit(2);
        }
    }
    return a;
}

// ----------------------------------------------------------------------------
// main
// ----------------------------------------------------------------------------
int main(int argc, char **argv)
{
    Args args = parse_args(argc, argv);

    // 1) Build FEDVR grid
    fedvr::FEDVRGrid grid(args.L, args.Ne_elem, args.n_order);
    const int N = grid.N;

    // 2) Build h_pq and V_pq depending on system geometry
    std::vector<fedvr::Nucleus> nuclei;
    if (args.H2)
    {
        nuclei.push_back({1.0, +0.5 * args.R});
        nuclei.push_back({1.0, -0.5 * args.R});
    }
    else
    {
        nuclei.push_back({args.Z, 0.0});
    }

    Eigen::SparseMatrix<double> h_sparse = fedvr::build_h_pq(grid, nuclei);
    Eigen::MatrixXd h_dense = Eigen::MatrixXd(h_sparse);
    Eigen::MatrixXd V = fedvr::build_V_pq(grid);
    double E_nn = fedvr::nuclear_repulsion(nuclei);

    // 3) Write JSON
    mini_json::Writer w;
    w.begin_object();
    {
        // model
        w.key("model");
        w.begin_object();
        {
            w.key("tag");        w.string_value(args.tag);        w.comma();
            w.key("a");          w.double_value(1.0);             w.comma();
            w.key("L");          w.double_value(args.L);          w.comma();
            w.key("n_elements"); w.int_value(args.Ne_elem);       w.comma();
            w.key("n_order");    w.int_value(args.n_order);       w.comma();
            w.key("N_DVR");      w.int_value(N);                  w.comma();
            w.key("N_e");        w.int_value(args.N_e);           w.comma();
            w.key("nuclei");
            w.begin_array();
            for (size_t i = 0; i < nuclei.size(); ++i)
            {
                if (i > 0)
                    w.comma();
                w.begin_object();
                w.key("Z"); w.double_value(nuclei[i].Z); w.comma();
                w.key("X"); w.double_value(nuclei[i].X);
                w.end_object();
            }
            w.end_array();
            w.comma();
            w.key("E_nn"); w.double_value(E_nn);
        }
        w.end_object();
        w.comma();

        w.key("x");    w.vector_value(grid.x); w.comma();
        w.key("w");    w.vector_value(grid.w); w.comma();
        w.key("h_pq"); w.matrix_value(h_dense); w.comma();
        w.key("V_pq"); w.matrix_value(V);
    }
    w.end_object();
    w.os << '\n';

    // 4) Write to disk
    {
        std::ofstream ofs(args.out);
        if (!ofs)
        {
            std::cerr << "Failed to open " << args.out << " for writing\n";
            return 1;
        }
        ofs << w.str();
    }

    // 5) Console summary
    std::cout << "Wrote integrals to " << args.out << "\n";
    std::cout << "  tag        = " << args.tag << "\n";
    std::cout << "  L          = " << args.L << "\n";
    std::cout << "  n_elements = " << args.Ne_elem << "\n";
    std::cout << "  n_order    = " << args.n_order << "\n";
    std::cout << "  N_DVR      = " << N << "  (qubits = 2 * N_DVR = " << 2 * N << ")\n";
    std::cout << "  N_e        = " << args.N_e << "\n";
    std::cout << "  nuclei     = ";
    for (size_t i = 0; i < nuclei.size(); ++i)
        std::cout << "(Z=" << nuclei[i].Z << ", X=" << nuclei[i].X << ")"
                  << (i + 1 < nuclei.size() ? ", " : "\n");
    std::cout << "  E_nn       = " << E_nn << "\n";
    return 0;
}
