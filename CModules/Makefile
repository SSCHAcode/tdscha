CPP := mpicxx
FLAGS := -O3
OBJECTS := LanczosFunctions.o Lanczos.o main.o

tdscha-lanczos.x: $(OBJECTS)
	$(CPP) $(FLAGS) $(OBJECTS) -o tdscha-lanczos.x

LanczosFunctions.o: LanczosFunctions.c 
	$(CPP) $(FLAGS) -c LanczosFunctions.c -o LanczosFunctions.o

Lanczos.o: Lanczos.cpp 
	$(CPP) $(FLAGS) -c Lanczos.cpp -o Lanczos.o

main.o: main.cpp
	$(CPP) $(FLAGS) -c main.cpp -o main.o