all: edge_cpu edge_coloring edge_omp

edge_cpu: edge_cpu.cpp Makefile graph_helper.hpp\
   	rms.hpp timer.hpp
	g++ -std=c++11 edge_cpu.cpp -o edge_cpu -O3

edge_coloring: edge_coloring.cpp Makefile graph_helper.hpp\
   	coloring.hpp rms.hpp timer.hpp
	g++ -std=c++11 edge_coloring.cpp -o edge_coloring -O3

edge_omp: edge_omp.cpp Makefile graph_helper.hpp coloring.hpp\
	rms.hpp timer.hpp
	g++ -std=c++11 edge_omp.cpp -o edge_omp -fopenmp -O3

clean:
	rm -rf *.o
	rm -f edge_cpu edge_coloring edge_omp
