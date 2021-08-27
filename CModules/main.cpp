#include<iostream>
#include<string.h>
#include "Lanczos.hpp"
#include <mpi.h>
#include "Utils.hpp"

using namespace std;

void PrintUsage();

int main(int argc, char * argv[]) {

    MPI_Init(&argc, &argv);

    if (argc != 2) {
        if (am_i_the_master()) PrintUsage();
        return EXIT_FAILURE;
    }

    string root_name = argv[1];

    // TODO: Initialize MPI


    // Initialize the Lanczos object
    Lanczos lanc(root_name);
    lanc.run();
}


void PrintUsage() {
    cout << "LANCZOS CALCULATOR PROGRAM" << endl;
    cout << "==========================" << endl << endl;

    cout << "Usage:" << endl;
    cout << "mpirun -np NPROC tdscha-lanczos  <rootname>" << endl << endl;

    cout << "NPROC is the number of processors" << endl;
    cout << "<rootname> is the name used to save the data from the python tdscha library." << endl;
}