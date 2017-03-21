CUDA_HOME := /usr/local/cuda
INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart

CPP       	= clang++ 
CCFLAGS  	= -O3
CPPFLAGS 	= $(CCFLAGS) -std=c++11
OMPFLAGS 	= -I/home/ireguly/forge/clang_4/llvm/build/projects/openmp/runtime/src/ -fopenmp=libomp
OMPOFFLOAD 	= -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp-nonaliased-maps -ffp-contract=fast -Xclang -target-feature -Xclang +ptx35

NVCCFLAGS	:= -lineinfo -arch=sm_35 --ptxas-options=-v --use_fast_math
CFLAGS := -std=c++11 -Xcompiler -Wall,-Wextra

all: edge_cpu edge_coloring edge_omp edge_gpu edge_gpu_2l\
	edge_gpu_cw edge_gpu_ca edge_gpu_omp edge_gpu_omp_2l

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

edge_gpu_2l: edge_gpu_2l.cu Makefile graph_helper.hpp coloring.hpp\
	rms.hpp timer.hpp
	nvcc edge_gpu_2l.cu -o edge_gpu_2l $(INC) $(NVCCFLAGS) $(CFLAGS) $(LIB)

edge_gpu_cw: edge_gpu_cw.cu Makefile graph_helper.hpp coloring.hpp\
	rms.hpp timer.hpp
	nvcc edge_gpu_cw.cu -o edge_gpu_cw $(INC) $(NVCCFLAGS) $(CFLAGS) $(LIB)

edge_gpu_ca: edge_gpu_ca.cu Makefile graph_helper.hpp coloring.hpp\
	rms.hpp timer.hpp
	nvcc edge_gpu_ca.cu -o edge_gpu_ca $(INC) $(NVCCFLAGS) $(CFLAGS) $(LIB)

edge_gpu_omp: 	edge_gpu_omp.cpp Makefile graph_helper.hpp coloring.hpp\
	rms.hpp timer.hpp	
	$(CPP) $(CPPFLAGS) $(OMPFLAGS) $(OMPOFFLOAD) edge_gpu_omp.cpp -o edge_gpu_omp

edge_gpu_omp_2l: 	edge_gpu_omp_2l.cpp Makefile graph_helper.hpp coloring.hpp\
	rms.hpp timer.hpp	
	$(CPP) $(CPPFLAGS) $(OMPFLAGS) $(OMPOFFLOAD) edge_gpu_omp_2l.cpp -o edge_gpu_omp_2l

clean:
	rm -rf *.o
	rm -f edge_cpu edge_coloring edge_omp edge_gpu_omp edge_gpu_omp_2l
	rm -f edge_gpu edge_gpu_2l edge_gpu_cw edge_gpu_ca
