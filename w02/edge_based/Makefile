CUDA_HOME := /usr/local/cuda
INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart -lgomp

CPP       	= clang++ 
CCFLAGS  	= -O3
CPPFLAGS 	= $(CCFLAGS) -std=c++11
OMPFLAGS 	= -I/home/ireguly/forge/clang_4/llvm/build/projects/openmp/runtime/src/ -fopenmp=libomp
OMPOFFLOAD 	= -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp-nonaliased-maps -ffp-contract=fast -Xclang -target-feature -Xclang +ptx35

NVCCFLAGS	:= -lineinfo -arch=sm_35 --use_fast_math
CFLAGS := -std=c++11 -Xcompiler -Wall,-Wextra,-fopenmp

all: edge_cpu edge_omp edge_gpu

arg.o: arg.hpp arg.cpp Makefile
	g++ arg.cpp -c -o arg.o ${CPPFLAGS}

argcu.o: arg.hpp arg.cpp Makefile
	nvcc arg.cpp -DUSE_CUDA -c -o argcu.o $(INC) $(NVCCFLAGS) $(CFLAGS) $(LIB)

seqkernels.o: seqkernels.cpp Makefile arg.o
	g++ seqkernels.cpp -c -o seqkernels.o ${CPPFLAGS}

coloring.o: coloring.hpp coloring.cpp dummycoloring.cpp Makefile
	g++ coloring.cpp -c -o coloring.o ${CPPFLAGS}

simulation.o: simulation.hpp simulation.cpp Makefile
	g++ simulation.cpp -c -o simulation.o ${CPPFLAGS}

edge_cpu: edge_sim.cpp Makefile graph_helper.hpp\
   	rms.hpp timer.hpp kernels.hpp graph_gen.hpp arg.o seqkernels.o simulation.o coloring.o
	g++ -std=c++11 edge_sim.cpp arg.o seqkernels.o simulation.o coloring.o dummycoloring.cpp -o edge_cpu ${CPPFLAGS}

edge_omp: edge_sim.cpp Makefile graph_helper.hpp\
   	rms.hpp timer.hpp kernels.hpp graph_gen.hpp arg.o\
   	ompkernel.cpp simulation.o coloring.o globalcoloring.cpp
	g++ -std=c++11 edge_sim.cpp arg.o ompkernels.cpp simulation.o coloring.o globalcoloring.cpp -o edge_omp ${CPPFLAGS} -fopenmp
	
edge_gpu: edge_sim.cpp Makefile graph_helper.hpp\
	rms.hpp timer.hpp kernels.hpp graph_gen.hpp argcu.o\
	cudakernels.cu globalcoloring.cpp coloring.cpp simulation.cpp cache_calc.hpp
	nvcc edge_sim.cpp -DUSE_CUDA -DBLOCKSIZE=128 argcu.o simulation.cpp coloring.cpp\
	   	globalcoloring.cpp cudakernels.cu -o edge_gpu \
		$(INC) $(NVCCFLAGS)	$(CFLAGS) $(LIB)

clean:
	rm -rf *.o
	rm -f edge_cpu edge_coloring edge_omp edge_gpu_omp edge_gpu_omp_2l
	rm -f edge_gpu edge_gpu_2l edge_gpu_cw edge_gpu_ca
