CPP := mpic++
CC := mpicc
FLAGS := -g #-O3
DFLAGS := -D_MPI
MPI_FLAGS := #-I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -I/usr/lib/x86_64-linux-gnu/openmpi/include -pthread -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi 
OBJECTS := LanczosFunctions.o Lanczos.o main.o Utils.o

tdscha-lanczos.x: $(OBJECTS)
	$(CPP) $(FLAGS) $(MPI_FLAGS) $(OBJECTS) $(DFLAGS) -o tdscha-lanczos.x

LanczosFunctions.o: LanczosFunctions.c LanczosFunctions.h
	$(CPP) $(FLAGS) $(MPI_FLAGS) $(DFLAGS) -c LanczosFunctions.c -o LanczosFunctions.o

Utils.o: Utils.cpp Utils.hpp
	$(CPP) $(FLAGS) $(MPI_FLAGS) $(DFLAGS) -c Utils.cpp -o Utils.o

Lanczos.o: Lanczos.cpp Lanczos.hpp
	$(CPP) $(FLAGS) $(MPI_FLAGS) $(DFLAGS) -c Lanczos.cpp -o Lanczos.o

main.o: main.cpp
	$(CPP) $(FLAGS) $(MPI_FLAGS) $(DFLAGS) -c main.cpp -o main.o