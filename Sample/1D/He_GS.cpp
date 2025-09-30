#include "He.hpp"

int main()
{
    double x_range = 20; // L = 160, so x_range = L/2 = 80
    double dx = 0.4;     // dx = L/(N-1) = 160/400 = 0.4
    double dtau = 0.2;   // dtau = 0.2
    int max_itr = 1000;  // maxitr = 1000
    double previous_ene = 1e10;
    double thresh = 1e-15;

    spatial_orbital_1d wfn = spatial_orbital_1d(x_range = x_range, dx = dx);
    // guess as gauss function
    wfn.guess();

    coulomb_interaction_1d two_e_int = coulomb_interaction_1d(x_range = x_range, dx = dx);

    cout << "init conplete:" << endl;

    cout << "calc GS start:" << endl;

    // Debug: Check initial parameters
    cout << "Debug: x_range = " << x_range << ", dx = " << dx << ", ngrid = " << wfn.get_ngrid() << endl;

    for (int i = 0; i < max_itr; i++)
    {
        double e1 = 2.0 * one_electron_energy(wfn);
        double e2 = two_electron_energy(wfn, two_e_int);

        double ene = e1 + e2;
        double diff = abs(ene - previous_ene);

        if (i % 100 == 0)
        {
            cout << "itr: " << i << ", ene: " << ene << endl;
            cout << "  e1 (one-electron): " << e1 << endl;
            cout << "  e2 (two-electron): " << e2 << endl;
            cout << "  diff: " << diff << endl;

            // Debug: Check normalization
            double norm = real(braket(dx, wfn, wfn));
            cout << "  norm: " << norm << endl;
        }

        if (diff < thresh)
        {
            cout << "calculation end" << endl;
            cout << "itr: " << i << ", ene: " << ene << endl;
            cout << "  e1 (one-electron): " << e1 << endl;
            cout << "  e2 (two-electron): " << e2 << endl;
            cout << "diff: " << diff << endl;
            break;
        }

        previous_ene = ene;

        wfn = rk4_gs(dtau, wfn, two_e_int);
    }
}
