SRC= dependency/SVDLIBC/*s*.c
CFLAGS=-L./SVDLIBC -lsvd -lm
COPTZ=-O3 -fopenmp -g
SUPP_WARN=-Wno-write-strings

PROGRAMS = exe-LatentTree-Nytimes exe-LatentTree-Openmp exe-LatentTree-Nips
all: $(PROGRAMS)

exe-LatentTree-Nytimes:  main_nytimes.cpp Graph.cpp $(SRC)
	g++ -o $@ $^  $(CFLAGS) $(COPTZ) $(SUPP_WARN) -DNDEBUG -std=c++0x

exe-LatentTree-Openmp: main_openmp.cpp Graph.cpp $(SRC)
	g++ -o $@ $^  $(CFLAGS) $(COPTZ) $(SUPP_WARN) -DNDEBUG -std=c++0x

clean: 
	rm $(PROGRAMS)


