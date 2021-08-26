#include "Lanczos.hpp"
#include <cmath>

#define DEBUG_READ true

using namespace std;
namespace pt = boost::property_tree;

Lanczos::Lanczos(string rootname) {
    setup_from_input(rootname);
}


void Lanczos::setup_from_input(string rootname) {
    pt::ptree root;
    pt::read_json(rootname + ".json", root);
    
    // Fill the generic values
    T = root.get<double>("T");
    n_steps = root.get<int>("n_steps");
    ignore_v3 = root.get<bool>("ignore_v3");
    ignore_v4 = root.get<bool>("ignore_v4");

    // Now read the more complex values
    pt::ptree &data = root.get_child("data");
    N = data.get<int>("n_configs");
    n_modes = data.get<int>("n_modes");
    n_syms = data.get<int>("n_syms");

    reverse_L = data.get<bool>("reverse");
    shift_value = data.get<double>("shift");
    
    n_psi = n_modes + n_modes * (n_modes + 1);

    if (DEBUG_READ) {
        cout << "[DEBUG READ] N = " << N << endl;
        cout << "[DEBUG READ] n_modes = " << n_modes << endl;
        cout << "[DEBUG READ] n_syms = " << n_syms << endl;
        cout << "[DEBUG READ] reverse_L = " << reverse_L << endl;
        cout << "[DEBUG READ] shift_value = " << shift_value << endl;
        cout << "[DEBUG READ] n_psi = " << n_psi << endl;
    }


    // Allocate the memory for the data
    N_degeneracy = (int*) malloc(sizeof(int) * n_modes);
    w = (double*) malloc(sizeof(double) * n_modes);
    nbose = (double*) malloc(sizeof(double) * n_modes);
    rho = (double *) malloc(sizeof(double) * N);
    psi = (double *) malloc(sizeof(double) * n_psi);

    Ups1 = (double*) calloc(sizeof(double), n_modes*n_modes);
    ReA1 = (double*) calloc(sizeof(double), n_modes*n_modes);

    X = (double *) malloc(sizeof(double) * N * n_modes);
    Y = (double *) malloc(sizeof(double) * N * n_modes);

    psi = (double*) malloc(sizeof(double) * n_psi);

    a = (double*) calloc(sizeof(double), n_steps);
    b = (double*) calloc(sizeof(double), n_steps);
    c = (double*) calloc(sizeof(double), n_steps);

    Qbasis = (double*) calloc(sizeof(double), n_psi * n_steps);
    Pbasis = (double*) calloc(sizeof(double), n_psi * n_steps);
    
    
    // Read the arrays from the files
    fstream file(rootname + ".ndegs");
    if (file.is_open()) {
        string line;
        int counter = 0;
        for (int k = 0; k < n_modes; ++k) {
            file >> N_degeneracy[k];
        }
    }
    file.close();

    file.open(rootname + ".freqs");
    if (file.is_open()) {
        string line;
        for (int k = 0; k < n_modes; ++k) {
            file >> w[k];
        }
    }
    file.close();


    file.open(rootname + ".rho");
    if (file.is_open()) {
        string line;
        int counter = 0;
        for (int k = 0; k < N; ++k) {
            file >> rho[k];
        }
    }
    file.close();

    file.open(rootname + ".psi");
    if (file.is_open()) {
        string line;
        for (int k = 0; k < n_psi; ++k) {
            file >> psi[k];
        }
    }
    file.close();

    // Now read the 2D arrays
    file.open(rootname + ".X.dat");
    if (file.is_open()) {
        string line;
        for (int k = 0; k < N*n_modes; ++k) {
            file >> X[k];
        }
    }
    file.close();


    file.open(rootname + ".Y.dat");
    if (file.is_open()) {
        string line;
        for (int k = 0; k < N*n_modes; ++k) {
            file >> Y[k];
        }
    }
    file.close();


    // Get the length of the degenerate space
    int n_deg_total = 0;
    for (int i = 0; i < n_modes; ++i) n_deg_total += N_degeneracy[i];
    degenerate_space = (int*) malloc(sizeof(int) * n_deg_total);


    file.open(rootname + ".degs");
    if (file.is_open()) {
        string line;
        int counter = 0;
        for (int k = 0; k < n_deg_total; ++k) {
            file >> degenerate_space[k];
        }
    }
    file.close();


    // Load the symmetries
    symmetries = (double*) calloc(sizeof(double), n_syms * n_modes * n_modes);
    int counter = 0;
    for (int i = 0; i < n_syms; ++i) {
        if (DEBUG_READ) {
            //cout << "[DEBUG READ]  Reading symmetry " << i + 1 << "/" << n_syms << " ..." <<  endl; 
        }

        file.open(rootname + ".syms" + to_string(i));
        if (file.is_open()) {
            for (int k = 0; k < n_modes*n_modes; ++k) {
                file >> symmetries[counter + k];
            }
            counter += n_modes * n_modes;
        }
        file.close();
    }

    // Update the Bose-Einstein statistics
    update_nbose();

    // Update the degenerate space to be used inside the lanczos functions
    good_deg_space = (int **) malloc(sizeof(int*) * n_modes);
    counter= 0;
    for (int i = 0; i < n_modes;++i) {
        good_deg_space[i] = (int*) malloc(sizeof(int) * N_degeneracy[i]);
        for (int j = 0; j < N_degeneracy[i]; ++j) {
        good_deg_space[i][j] = degenerate_space[counter++];
        }
    }
}

void Lanczos::update_nbose() {
    for (int i = 0; i < n_modes; ++i) {
        nbose[i] = 0;

        if (T > 0) {
            nbose[i] = 1.0 / (exp(w[i] / (T * RY_TO_K)) - 1);
        }
    }
}


void Lanczos::apply_L1(double * out_vect, bool transpose) {
    // Clean the out_vect
    int i, j, k;
    int x, y;
    int N_w2;

    N_w2 = (n_modes * (n_modes + 1)) / 2;


    for (i = 0; i < n_psi; ++i) {
        out_vect[i] = 0;
    }

    // Apply the R - R propagation
    for (i = 0; i < n_modes; ++i) {
        out_vect[i] = (psi[i] * w[i]) *w[i];
    }

    int start_Y = n_modes;
    int start_A = n_modes + N_w2;

    double X_ni, Y_ni, X1_ni, Y1_ni;
    double den;
    for (i = 0; i < N_w2; ++i) {
        get_indices_from_sym_index(i, x, y);
        
        X_ni = -w[x]*w[x];
        X_ni -= w[y]*w[y];
        den = (2*nbose[x] + 1) * (2*nbose[y] + 1);
        X_ni -= (2*w[x]*w[y]) / den;
        
        out_vect[start_Y + i] = - X_ni * psi[start_Y + i];

        Y_ni = -(8*w[x]*w[y]) / den;

        if (transpose) 
            out_vect[start_A + i] += -Y_ni * psi[start_Y + 1];
        else
            out_vect[start_Y + i] += -Y_ni * psi[start_A + i];
        
        X1_ni = -(2*nbose[x]*nbose[y] + nbose[x] + nbose[y]);
        X1_ni *= 2*nbose[x]*nbose[y] + nbose[x] + nbose[y] + 1;
        X1_ni *= 2*nbose[x]*nbose[y];
        X1_ni /= den;

        if (transpose)
            out_vect[start_Y + i] += -X1_ni * psi[start_A + i];
        else
            out_vect[start_A + i] += -X1_ni * psi[start_Y + i];
        
        Y1_ni = -w[x]*w[x] - w[y]*w[y] + 2*w[x]*w[y];
        Y1_ni /= den;

        out_vect[start_A + i] = -Y1_ni * psi[start_A + i];
    }
}


void Lanczos::apply_anharmonic(double * final_psi, bool transpose) {
    int N_w2 = (n_modes * (n_modes + 1)) / 2;
    int x, y;
    double * Y1 = (double*) calloc(sizeof(double), N_w2);
    double * R1 = (double*) calloc(sizeof(double), n_modes);

    int start_Y = n_modes;
    int start_A = n_modes + N_w2;

    double Y_wa, Y_wb, coeff_Y, ReA_w1, ReA_w2, coeff_RA;

    if (transpose) {
        for (int i = 0; i < N_w2; ++i) {
            get_indices_from_sym_index(i, x, y);

            Y_wa =  2 * w[x] / (2 * nbose[x] + 1);
            Y_wb =  2 * w[y] / (2 * nbose[y] + 1);
            coeff_Y = 2* (Y_wa*Y_wb*Y_wb + Y_wa*Y_wa*Y_wb);

            ReA_w1 = 2*w[x]*nbose[x] * (nbose[x] + 1) / (2*nbose[x] + 1);
            ReA_w2 = 2*w[y]*nbose[y] * (nbose[y] + 1) / (2*nbose[y] + 1);
            coeff_RA = 2*(Y_wa*ReA_w2*Y_wb + Y_wa*ReA_w1*Y_wb);

            Y1[i] = psi[start_Y + i] * coeff_Y;
            Y1[i] += psi[start_A + i] * coeff_RA;
        }
    } else {
        for (int i = 0; i <  N_w2; ++i)
            Y1[i] = psi[start_Y + i];
    }


    // Compute the sscha average force and potential
    double * f_pert_av = (double*) calloc(sizeof(double), n_modes);
    double * d2v_pert_av = (double*) calloc(sizeof(double), n_modes*n_modes);

    double * Y1_new = (double*) malloc(sizeof(double) * n_modes*n_modes);
    for (int i = 0; i < N_w2; ++i) {
        get_indices_from_sym_index(i, x, y);
        if (x != y && transpose) {
            Y1_new[n_modes*x + y] = Y1[i] / 2;
            Y1_new[n_modes*y + x] = Y1[i] / 2;
        } else {
            Y1_new[n_modes*x + y] = Y1[i] ;
            Y1_new[n_modes*y + x] = Y1[i] ;
        }
    }

    if (! ignore_v3) {
        get_f_average_from_Y_pert_sym(X, Y, w, Y1_new, T, n_modes, N, rho, symmetries, n_syms, N_degeneracy, good_deg_space, f_pert_av);
        get_d2v_dR2_from_R_pert_sym(X, Y, w, R1, T, n_modes, N, rho, symmetries, n_syms, N_degeneracy, good_deg_space, d2v_pert_av);
    }
    
    if (! ignore_v4) {
        get_d2v_dR2_from_Y_pert_sym(X, Y, w, Y1_new, T, n_modes, N, rho, symmetries, n_syms, N_degeneracy, good_deg_space, d2v_pert_av);
    }


    // Free memory 
    free(Y1);
    free(Y1_new);
    free(R1);


    for (int i = 0; i < n_modes; ++i) {
        final_psi[i] = -f_pert_av[i];
    }

    double pert_Y, pert_RA;

    for (int i = 0; i < N_w2; ++i) {
        get_indices_from_sym_index(i, x, y);

        Y_wa =  2 * w[x] / (2 * nbose[x] + 1);
        Y_wb =  2 * w[y] / (2 * nbose[y] + 1);

        ReA_w1 = 2*w[x]*nbose[x] * (nbose[x] + 1) / (2*nbose[x] + 1);
        ReA_w2 = 2*w[y]*nbose[y] * (nbose[y] + 1) / (2*nbose[y] + 1);

        if (transpose) {
            pert_Y = 0.5 *  d2v_pert_av[x*n_modes + y] / (Y_wa * Y_wb)
            pert_RA = 0;

            if (x != y) pert_Y *= 2;
        } else {
            pert_Y = d2v_pert_av[x*n_modes + y] * (Y_wa + Y_wb);
            pert_RA = d2v_pert_av[x*n_modes + y] * (ReA_w1 + ReA_w2);
        }

        final_psi[start_Y + i] = -pert_Y;
        final_psi[start_A + i] = -pert_RA;
        
    }

    free(f_pert_av);
    free(d2v_pert_av);
}


int get_sym_index(int a, int b) {
    int ret = 0;

    if (b > a) return get_sym_index(b, a);

    for (int i = 0; i < a; ++i) {
        ret += i + 1;
    } 
    ret += b;
    return ret;
}

void get_indices_from_sym_index(int index, int &a, int &b) {
    int i, j;
    int counter = 0;
    a = 0;
    b = 0;
    for (i= 0; i < index; ++i) {
        if (i - counter > a) { 
            a++;
            counter = i;
        }
    }
    b = index - counter;
}
