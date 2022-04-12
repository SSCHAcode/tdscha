#include "Utils.hpp"

bool am_i_the_master(void) {
    int size, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) return true;
    return false;
}