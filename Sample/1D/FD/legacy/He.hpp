#include <iostream>
#include <vector>
#include <complex>

using namespace ::std;

complex<double> inner_product(const double dx, const vector<complex<double>> vector1, const vector<complex<double>> vector2)
{
    complex<double> res = 0;
    for (int i = 0; i < vector1.size(); i++)
    {
        res += conj(vector1[i]) * vector2[i];
    }

    return res * dx;
}

vector<complex<double>> scale_vector(const double c, const vector<complex<double>> vec)
/**
 * @brief = c * vector
 */
{
    vector<complex<double>> _vec(vec.size());

    for (int i = 0; i < vec.size(); i++)
    {
        _vec[i] = c * vec[i];
    }

    return _vec;
}

vector<complex<double>> scale_vector(const complex<double> c, const vector<complex<double>> vec)
/**
 * @brief = c * vector
 */
{
    vector<complex<double>> _vec(vec.size());

    for (int i = 0; i < vec.size(); i++)
    {
        _vec[i] = c * vec[i];
    }

    return _vec;
}

class spatial_orbital_1d
{
private:
    double x_range;
    double dx;
    int ngrid;
    vector<complex<double>> values;

public:
    spatial_orbital_1d(double _x_range, double _dx)
        : x_range(_x_range),
          dx(_dx)
    {
        ngrid = static_cast<int>(2 * x_range / dx) + 1;
        values.resize(ngrid);

        for (int i = 0; i < ngrid; i++)
        {
            values[i] = 0;
        }
    };

    spatial_orbital_1d(const spatial_orbital_1d &other)
        : x_range(other.x_range),
          dx(other.dx),
          ngrid(other.ngrid),
          values(other.values)
    {
    }

    const vector<complex<double>> &get_values() const { return values; }
    vector<complex<double>> &get_values() { return values; }
    double get_dx() const { return dx; }
    double get_x_range() const { return x_range; }
    int get_ngrid() const { return ngrid; }

    complex<double> &operator[](int i)
    {
        if (i < 0 || i >= ngrid)
        {
            throw out_of_range("Index out of range");
        }
        return values[i];
    }
    const complex<double> &operator[](int i) const
    {
        if (i < 0 || i >= ngrid)
        {
            throw out_of_range("Index out of range");
        }
        return values[i];
    }

    spatial_orbital_1d operator+(const spatial_orbital_1d &other) const
    {
        if (ngrid != other.ngrid)
        {
            throw invalid_argument("Orbitals must have same grid");
        }
        spatial_orbital_1d result(*this);
        for (int i = 0; i < ngrid; i++)
        {
            result[i] += other[i];
        }
        return result;
    }

    spatial_orbital_1d operator*(double scale) const
    {
        spatial_orbital_1d result(*this);
        for (int i = 0; i < ngrid; i++)
        {
            result[i] *= scale;
        }
        return result;
    }

    spatial_orbital_1d operator*(complex<double> scale) const
    {
        spatial_orbital_1d result(*this);
        for (int i = 0; i < ngrid; i++)
        {
            result[i] *= scale;
        }
        return result;
    }

    spatial_orbital_1d &operator+=(const spatial_orbital_1d &other)
    {
        if (ngrid != other.ngrid)
        {
            throw invalid_argument("Orbitals must have same grid");
        }
        for (int i = 0; i < ngrid; i++)
        {
            values[i] += other.values[i];
        }
        return *this;
    }

    spatial_orbital_1d &operator*=(double scale)
    {
        for (int i = 0; i < ngrid; i++)
        {
            values[i] *= scale;
        }
        return *this;
    }

    spatial_orbital_1d &operator*=(complex<double> scale)
    {
        for (int i = 0; i < ngrid; i++)
        {
            values[i] *= scale;
        }
        return *this;
    }

    void guess()
    {
        for (int i = 0; i < ngrid; i++)
        {
            double x = -x_range + i * dx;
            values[i] = exp(-pow(x, 2));
        }

        this->normalize();
    }
    void normalize()
    {
        complex<double> norm = inner_product(dx, values, values);
        values = scale_vector(1.0 / sqrt(real(norm)), values);
    };
};

class coulomb_interaction_1d
{
private:
    vector<complex<double>> V12;
    double dx;
    double x_range;
    int ngrid;

public:
    coulomb_interaction_1d(const double _x_range, const double _dx)
        : x_range(_x_range), dx(_dx)
    {
        ngrid = static_cast<int>(2 * x_range / dx) + 1;

        int x1, x2;
        double ngrid2 = ngrid * ngrid;
        V12.resize(ngrid2);

        for (int igrid = 0; igrid < ngrid2; igrid++)
        {
            // x1grid = igrid / ngrid;
            // x2grid = igrid % ngrid;

            double x1 = -x_range + (igrid / ngrid) * dx;
            double x2 = -x_range + (igrid % ngrid) * dx;

            double dx2 = pow(x1 - x2, 2);
            // V12 stores the *physical* electron-electron interaction
            //   w(x1, x2) = 1 / sqrt((x1 - x2)^2 + 1).
            // Discretization weights (dx, dx^2, ...) are applied at the
            // call site (Hartree potential, two-electron energy).
            V12[igrid] = 1.0 / sqrt(dx2 + 1.0);
        }
    }

    const complex<double> &operator[](int i) const
    {
        if (i < 0 || i >= ngrid * ngrid)
        {
            throw out_of_range("Index out of range");
        }
        return V12[i];
    }

    // Continuous Hartree potential
    //   W(x1) = ∫ w(x1, x2) |ψ(x2)|^2 dx2
    // discretized with the trapezoid/midpoint rule:
    //   W_α = Σ_β V12_{αβ} · |ψ_β|^2 · dx
    vector<complex<double>> hartree_potential(const spatial_orbital_1d &orb) const
    {
        vector<complex<double>> density(ngrid);
        vector<complex<double>> potential(ngrid, 0.0);

        for (int i = 0; i < ngrid; i++)
        {
            density[i] = conj(orb[i]) * orb[i];
        }

        for (int i = 0; i < ngrid * ngrid; i++)
        {
            int ix1 = i / ngrid;
            int ix2 = i % ngrid;
            potential[ix1] += V12[i] * density[ix2];
        }
        // multiply by dx for the integration over x2
        for (int i = 0; i < ngrid; i++)
        {
            potential[i] *= dx;
        }

        return potential;
    }
};

spatial_orbital_1d kinetic_energy_operator(const spatial_orbital_1d orb)
/**
 * @brief Apply kinetic energy operator (-1/2)∇² to orbitalf unction
 * @param spatial_orbital_1d
 * @return spatial_orbital_1d
 */
{
    spatial_orbital_1d _orb(orb);
    const double dx2 = orb.get_dx() * orb.get_dx();
    const int ngrid = orb.get_ngrid();

    for (int i = 0; i < ngrid; i++)
    {
        if (i == 0)
        {
            _orb[i] = -0.5 * (orb[i + 1] - 2.0 * orb[i]) / dx2;
        }
        else if (i == ngrid - 1)
        {
            _orb[i] = -0.5 * (-2.0 * orb[i] + orb[i - 1]) / dx2;
        }
        else
        {
            _orb[i] = -0.5 * (orb[i + 1] - 2.0 * orb[i] + orb[i - 1]) / dx2;
            // horb[i] = -0.5 * (-orb[i + 2] + 16.0 * orb[i + 1] - 30.0 * orb[i] + 16.0 * orb[i - 1] - orb[i - 2]) / (12 * dx2);
        }
    }

    return _orb;
};

spatial_orbital_1d hamiltonian(const spatial_orbital_1d orb, const coulomb_interaction_1d two_e_int)
/**
 * @brief Apply hamiltonian operator (-1/2)∇² - 1/sqrt(x^2+1)+ W to orbitalf unction
 * @param spatial_orbital_1d
 * @return spatial_orbital_1d
 */
{
    spatial_orbital_1d _orb(orb);
    const double dx = orb.get_dx();
    const double dx2 = dx * dx;
    const int ngrid = orb.get_ngrid();
    const double x_range = orb.get_x_range();

    // Calculate density
    vector<complex<double>> rho(ngrid);
    for (int i = 0; i < ngrid; i++)
    {
        rho[i] = conj(orb[i]) * orb[i];
    }

    // Calculate Hartree potential using V12 matrix directly
    //   W_α = (Σ_β V12_{αβ} · ρ_β) · dx     (dx is the integration weight over x2)
    vector<complex<double>> hartree(ngrid, 0.0);
    for (int i = 0; i < ngrid * ngrid; i++)
    {
        int ix = i / ngrid;
        int iy = i % ngrid;
        hartree[ix] += two_e_int[i] * rho[iy];
    }
    for (int i = 0; i < ngrid; i++)
    {
        hartree[i] *= dx;
    }

    for (int i = 0; i < ngrid; i++)
    {
        if (i == 0)
        {
            _orb[i] = -0.5 * (orb[i + 1] - 2.0 * orb[i]) / dx2;
        }
        else if (i == ngrid - 1)
        {
            _orb[i] = -0.5 * (-2.0 * orb[i] + orb[i - 1]) / dx2;
        }
        else
        {
            _orb[i] = -0.5 * (orb[i + 1] - 2.0 * orb[i] + orb[i - 1]) / dx2;
            // horb[i] = -0.5 * (-orb[i + 2] + 16.0 * orb[i + 1] - 30.0 * orb[i] + 16.0 * orb[i - 1] - orb[i - 2]) / (12 * dx2);
        }

        double x = -x_range + i * dx;
        // Z=2 for He
        _orb[i] += (-2.0 * orb[i] / sqrt(1 + x * x)) + hartree[i] * orb[i];
    }

    return _orb;
};

spatial_orbital_1d one_electron_hamiltonian(const spatial_orbital_1d orb)
/**
 * @brief Apply hamiltonian operator (-1/2)∇² - 1/sqrt(x^2+1) to orbitalf unction
 * @param spatial_orbital_1d
 * @return spatial_orbital_1d
 */
{
    spatial_orbital_1d _orb(orb);
    const double dx = orb.get_dx();
    const double dx2 = dx * dx;
    const int ngrid = orb.get_ngrid();
    const double x_range = orb.get_x_range();

    for (int i = 0; i < ngrid; i++)
    {
        if (i == 0)
        {
            _orb[i] = -0.5 * (orb[i + 1] - 2.0 * orb[i]) / dx2;
        }
        else if (i == ngrid - 1)
        {
            _orb[i] = -0.5 * (-2.0 * orb[i] + orb[i - 1]) / dx2;
        }
        else
        {
            _orb[i] = -0.5 * (orb[i + 1] - 2.0 * orb[i] + orb[i - 1]) / dx2;
            // horb[i] = -0.5 * (-orb[i + 2] + 16.0 * orb[i + 1] - 30.0 * orb[i] + 16.0 * orb[i - 1] - orb[i - 2]) / (12 * dx2);
        }

        double x = -x_range + i * dx;
        // Z=2 for He
        _orb[i] += -2.0 * orb[i] / sqrt(1 + x * x);
    }

    return _orb;
};

spatial_orbital_1d two_electron_hamiltonian(const spatial_orbital_1d orb, const coulomb_interaction_1d two_e_int)
/**
 * @brief Apply hamiltonian operator (-1/2)∇² - 1/sqrt(x^2+1) to orbitalf unction
 * @param spatial_orbital_1d
 * @return spatial_orbital_1d
 */
{
    spatial_orbital_1d _orb(orb);
    const double dx = orb.get_dx();
    const double dx2 = dx * dx;
    const int ngrid = orb.get_ngrid();
    const double x_range = orb.get_x_range();

    vector<complex<double>> hartree = two_e_int.hartree_potential(orb);

    for (int i = 0; i < ngrid; i++)
    {
        double x = -x_range + i * dx;
        _orb[i] = hartree[i] * orb[i];
    }

    return _orb;
};

complex<double> braket(const double dx, const spatial_orbital_1d orb1, const spatial_orbital_1d orb2)
{
    complex<double> res = 0;
    for (int i = 0; i < orb1.get_ngrid(); i++)
    {
        res += conj(orb1[i]) * orb2[i];
    }

    return res * dx;
}

double one_electron_energy(const spatial_orbital_1d orb)
{
    int ngrid = orb.get_ngrid();
    double dx = orb.get_dx();

    return real(braket(dx, orb, one_electron_hamiltonian(orb)));
}

// Two-electron (Hartree) energy:
//   E2 = ∫∫ |ψ(x1)|^2 w(x1, x2) |ψ(x2)|^2 dx1 dx2
//      ≈ Σ_{α, β} ρ_α · V12_{αβ} · ρ_β · (dx)^2
double two_electron_energy(const spatial_orbital_1d orb, const coulomb_interaction_1d two_e_int)
{
    int ngrid = orb.get_ngrid();
    double dx = orb.get_dx();

    vector<complex<double>> rho(ngrid);
    for (int i = 0; i < ngrid; i++)
    {
        rho[i] = conj(orb[i]) * orb[i];
    }

    vector<complex<double>> v12rho(ngrid, 0.0);
    for (int i = 0; i < ngrid * ngrid; i++)
    {
        int ix = i / ngrid;
        int iy = i % ngrid;
        v12rho[ix] += two_e_int[i] * rho[iy];
    }

    complex<double> e2 = 0.0;
    for (int i = 0; i < ngrid; i++)
    {
        e2 += v12rho[i] * rho[i];
    }

    // Two integration weights: one for x1, one for x2.
    return real(e2) * dx * dx;
}

spatial_orbital_1d rk4_gs(const double dtau, const spatial_orbital_1d orb, const coulomb_interaction_1d two_e_int)
/**
 * @brief rk4 time propagation for calc ground state
 * @param spatial_orbital_1d
 * @return spatial_orbital_1d
 */
{
    spatial_orbital_1d dorb(orb);
    spatial_orbital_1d w1(orb);
    spatial_orbital_1d w2(orb);
    const int ngrid = orb.get_ngrid();
    const double x_range = orb.get_x_range();

    dorb = hamiltonian(orb, two_e_int) * (-1.0);
    w1 = orb + dorb * (dtau / 2.0);
    w2 = orb + dorb * (dtau / 6.0);

    dorb = hamiltonian(w1, two_e_int) * (-1.0);
    w1 = orb + dorb * (dtau / 2.0);
    w2 += dorb * (dtau / 3.0);

    dorb = hamiltonian(w1, two_e_int) * (-1.0);
    w1 = orb + dorb * dtau;
    w2 += dorb * (dtau / 3.0);

    dorb = hamiltonian(w1, two_e_int) * (-1.0);
    w2 += dorb * (dtau / 6.0);

    w2.normalize();

    return w2;
};
