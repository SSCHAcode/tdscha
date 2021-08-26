#include "Lanczos.hpp"

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
    rho = (double *) malloc(sizeof(double) * N);
    psi = (double *) malloc(sizeof(double) * n_psi);

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
}