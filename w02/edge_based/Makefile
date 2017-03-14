CUDA_HOME := /usr/local/cuda
INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart

NVCCFLAGS	:= -lineinfo -arch=sm_35 --ptxas-options=-v --use_fast_math
CFLAGS := -std=c++11 -Xcompiler -Wall,-Wextra

all: edge_cpu edge_coloring edge_omp edge_gpu

edge_cpu: edge_cpu.cpp Makefile graph_helper.hpp\
   	rms.hpp timer.hpp
	g++ -std=c++11 edge_cpu.cpp -o edge_cpu -O3

edge_coloring: edge_coloring.cpp Makefile graph_helper.hpp\
   	coloring.hpp rms.hpp timer.hpp
	g++ -std=c++11 edge_coloring.cpp -o edge_coloring -O3

edge_omp: edge_omp.cpp Makefile graph_helper.hpp coloring.hpp\
	rms.hpp timer.hpp
	g++ -std=c++11 edge_omp.cpp -o edge_omp -fopenmp -O3

edge_gpu: edge_gpu.cu Makefile graph_helper.hpp coloring.hpp\
	rms.hpp timer.hpp
	nvcc edge_gpu.cu -o edge_gpu $(INC) $(NVCCFLAGS) $(CFLAGS) $(LIB)

clean:
	rm -rf *.o
	rm -f edge_cpu edge_coloring edge_omp edge_gpu
