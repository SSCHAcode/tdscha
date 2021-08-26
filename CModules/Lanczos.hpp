#ifndef __LANCZOS__
#define __LANCZOS__
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string.h>
#include <exception>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

using namespace std;

class Lanczos {
    int N, n_syms, n_modes, n_steps;
    bool ignore_v3, ignore_v4, reverse_L;

    double T, shift_value;

    double *w, *rho, *m;
    int * N_degeneracy, *degenerate_space;

    double *X, *Y, *psi,  *symmetries;

    double *a, *b, *c;
    double *Qbasis, *Pbasis;

    int n_psi;

public:
    // Constructors
    Lanczos(string rootname);

    // Load everything from the input files
    void setup_from_input(string rootname);

    // Run the lanczos algorithm
    void run();
};

#endif

