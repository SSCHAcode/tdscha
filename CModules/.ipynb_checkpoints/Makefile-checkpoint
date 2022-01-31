CPP := /usr/bin/mpic++
CC := /usr/bin/mpicc
FLAGS :=   -O3 #(Causes a seg. fault)
OPTFLAG := #-O2
DFLAGS := -D_MPI
MPI_FLAGS := 
LIBS := 

OBJECTS := LanczosFunctions.o Lanczos.o main.o Utils.o

tdscha-lanczos.x: $(OBJECTS)
	$(CPP) $(FLAGS) $(MPI_FLAGS) $(OBJECTS) $(DFLAGS) -o tdscha-lanczos.x $(LIBS)

LanczosFunctions.o: LanczosFunctions.c LanczosFunctions.h
	$(CPP) $(FLAGS) $(OPTFLAG) $(MPI_FLAGS) $(DFLAGS) -c LanczosFunctions.c -o LanczosFunctions.o

Utils.o: Utils.cpp Utils.hpp
	$(CPP) $(FLAGS) $(MPI_FLAGS) $(DFLAGS) -c Utils.cpp -o Utils.o

Lanczos.o: Lanczos.cpp Lanczos.hpp
	$(CPP) $(FLAGS) $(MPI_FLAGS) $(DFLAGS) -c Lanczos.cpp -o Lanczos.o

main.o: main.cpp
	$(CPP) $(FLAGS) $(MPI_FLAGS) $(DFLAGS) -c main.cpp -o main.o
