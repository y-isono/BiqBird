#include <iostream>
#include <vector>
#include <complex>

using namespace ::std;

// constants
double x_range = 15;
double dx = 0.2;
int ngrid = 2 * x_range / dx;
double dtau = 0.05;
double dx2 = dx * dx;

void print_vector(vector<complex<double>> vector)
{
    for (int i = 0; i < vector.size(); i++)
    {
        double x = -x_range + i * dx;
        cout << x << ", " << vector[i] << endl;
    }
};

complex<double> d2c(double a)
{
    return complex<double>(a, 0);
}

vector<complex<double>> add_vector(vector<complex<double>> wfn1, vector<complex<double>> wfn2)
/**
 * @brief = wfn1 + wfn0
 */
{
    vector<complex<double>> _wfn(ngrid);

    for (int i = 0; i < wfn1.size(); i++)
    {
        _wfn[i] = wfn1[i] + wfn2[i];
    }

    return _wfn;
}

vector<complex<double>> scale_vector(double c, vector<complex<double>> wfn)
/**
 * @brief = c * wfn
 */
{
    vector<complex<double>> _wfn(ngrid);

    for (int i = 0; i < wfn.size(); i++)
    {
        _wfn[i] = c * wfn[i];
    }

    return _wfn;
}

vector<complex<double>> scale_vector(complex<double> c, vector<complex<double>> wfn)
/**
 * @brief = c * wfn
 */
{
    vector<complex<double>> _wfn(ngrid);

    for (int i = 0; i < wfn.size(); i++)
    {
        _wfn[i] = c * wfn[i];
    }

    return _wfn;
}

vector<complex<double>> laplacian_operator(vector<complex<double>> wfn)
/**
 * @brief Apply Laplacian operator (∇²) to input wavefunction
 * @param d2wfn
 * @param wfn
 * @return n/a
 */
{
    vector<complex<double>> d2wfn(ngrid);

    // d2/dx2 u_n = (u_n+1 - 2 u_n + u_n-1)/dx^2
    for (int i = 0; i < wfn.size(); i++)
    {
        if (i == 0)
        {
            d2wfn[i] = (wfn[i + 1] - 2.0 * wfn[i]) / dx2;
        }
        else if (i == wfn.size() - 1)
        {
            d2wfn[i] = (-2.0 * wfn[i] + wfn[i - 1]) / dx2;
        }
        else
        {
            d2wfn[i] = (wfn[i + 1] - 2.0 * wfn[i] + wfn[i - 1]) / dx2;
            // hwfn[i] = -0.5 * (-wfn[i + 2] + 16.0 * wfn[i + 1] - 30.0 * wfn[i] + 16.0 * wfn[i - 1] - wfn[i - 2]) / (12 * dx2);
        }
    }

    return d2wfn;
};

vector<complex<double>> kinetic_energy_operator(vector<complex<double>> wfn)
/**
 * @brief Apply kinetic energy operator (-1/2)∇² to wavefunction
 * @param d2wfn
 * @param wfn
 * @return n/a
 */
{
    vector<complex<double>> kwfn(ngrid);
    // d2/dx2 u_n = (u_n+1 - 2 u_n + u_n-1)/dx^2
    for (int i = 0; i < wfn.size(); i++)
    {
        if (i == 0)
        {
            kwfn[i] = -0.5 * (wfn[i + 1] - 2.0 * wfn[i]) / dx2;
        }
        else if (i == wfn.size() - 1)
        {
            kwfn[i] = -0.5 * (-2.0 * wfn[i] + wfn[i - 1]) / dx2;
        }
        else
        {
            kwfn[i] = -0.5 * (wfn[i + 1] - 2.0 * wfn[i] + wfn[i - 1]) / dx2;
            // hwfn[i] = -0.5 * (-wfn[i + 2] + 16.0 * wfn[i + 1] - 30.0 * wfn[i] + 16.0 * wfn[i - 1] - wfn[i - 2]) / (12 * dx2);
        }
    }

    return kwfn;
}

vector<complex<double>> hamiltonian(vector<complex<double>> wfn)
/**
 * @brief Apply hamiltonian operator (-1/2)∇² - 1/sqrt(x^2+1) to wavefunction
 * @param hwfn
 * @param wfn
 * @return n/a
 */
{
    vector<complex<double>> hwfn(ngrid);
    for (int i = 0; i < wfn.size(); i++)
    {

        if (i == 0)
        {
            hwfn[i] = -0.5 * (wfn[i + 1] - 2.0 * wfn[i]) / dx2;
        }
        else if (i == wfn.size() - 1)
        {
            hwfn[i] = -0.5 * (-2.0 * wfn[i] + wfn[i - 1]) / dx2;
        }
        else
        {
            hwfn[i] = -0.5 * (wfn[i + 1] - 2.0 * wfn[i] + wfn[i - 1]) / dx2;
            // hwfn[i] = -0.5 * (-wfn[i + 2] + 16.0 * wfn[i + 1] - 30.0 * wfn[i] + 16.0 * wfn[i - 1] - wfn[i - 2]) / (12 * dx2);
        }

        double x = -x_range + i * dx;
        hwfn[i] += -wfn[i] / sqrt(1 + x * x);
    }

    return hwfn;
}

complex<double> inner_product(vector<complex<double>> wfn1, vector<complex<double>> wfn2)
{
    complex<double> res = 0;
    for (int i = 0; i < wfn1.size(); i++)
    {
        res += conj(wfn1[i]) * wfn2[i];
    }

    return res * dx;
}

vector<complex<double>> normalize(vector<complex<double>> wfn)
{
    vector<complex<double>> _wfn(ngrid);
    complex<double> norm = inner_product(wfn, wfn);
    _wfn = scale_vector(1.0 / sqrt(real(norm)), wfn);

    return _wfn;
}

vector<complex<double>> copy(vector<complex<double>> wfn)
{
    vector<complex<double>> _wfn(ngrid);

    for (int i = 0; i < wfn.size(); i++)
    {
        _wfn[i] = wfn[i];
    }

    return _wfn;
}

int main()
{

    int max_itr = 10000;
    vector<double> ene_list(max_itr);
    double previous_ene = 1e10;
    double thresh = 1e-15;

    vector<complex<double>> wfn(ngrid);

    // initialize
    for (int i = 0; i < ngrid; i++)
    {
        double x = -x_range + i * dx;
        wfn[i] = exp(-pow(x, 2));
    }

    wfn = normalize(wfn);
    cout << "init conplete:" << endl;

    vector<complex<double>> dwfn(ngrid);
    vector<complex<double>> w1(ngrid);
    vector<complex<double>> w2(ngrid);

    cout << "calc GS start:" << endl;
    for (int i = 0; i < max_itr; i++)
    {
        double ene = real(inner_product(wfn, hamiltonian(wfn)));
        ene_list[i] = ene;
        double diff = abs(ene - previous_ene);

        if (i % 100 == 0)
        {
            cout << "itr: " << i << ", ene: " << ene << endl;
            cout << "diff: " << diff << endl;
        }

        if (diff < thresh)
        {
            cout << "calculation end" << endl;
            cout << "itr: " << i << ", ene: " << ene << endl;
            cout << "diff: " << diff << endl;
            break;
        }

        previous_ene = ene;

        w1 = copy(wfn);
        w2 = copy(wfn);
        dwfn = scale_vector(-1.0, hamiltonian(wfn));
        w1 = add_vector(wfn, scale_vector(dtau / 2.0, dwfn));
        w2 = add_vector(w2, scale_vector(dtau / 6.0, dwfn));

        dwfn = scale_vector(-1.0, hamiltonian(w1));
        w1 = add_vector(wfn, scale_vector(dtau / 2.0, dwfn));
        w2 = add_vector(w2, scale_vector(dtau / 3.0, dwfn));

        dwfn = scale_vector(-1.0, hamiltonian(w1));
        w1 = add_vector(wfn, scale_vector(dtau, dwfn));
        w2 = add_vector(w2, scale_vector(dtau / 3.0, dwfn));

        dwfn = scale_vector(-1.0, hamiltonian(w1));
        w2 = add_vector(w2, scale_vector(dtau / 6.0, dwfn));
        wfn = normalize(w2);

        // print_vector(wfn);
        // print_vector(hamiltonian(wfn));
    }

    return 0;
}
