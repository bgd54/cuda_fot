CUDA_HOME := /usr/local/cuda
INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart -lgomp

CPP       	= clang++ 
CCFLAGS  	= -O3
CPPFLAGS 	= $(CCFLAGS) -std=c++11
OMPFLAGS 	= -I/home/ireguly/forge/clang_4/llvm/build/projects/openmp/runtime/src/ -fopenmp=libomp
OMPOFFLOAD 	= -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp-nonaliased-maps -ffp-contract=fast -Xclang -target-feature -Xclang +ptx35

NVCCFLAGS	:= -lineinfo -arch=sm_35 --use_fast_math
CFLAGS := -O3 -std=c++11 -Xcompiler -fopenmp

BASECPP := arg.cpp edge_sim.cpp simulation.cpp coloring.cpp
BASEHPP := arg.hpp simulation.hpp coloring.hpp kernels.hpp timer.hpp graph_gen.hpp rms.hpp graph_helper.hpp reordering.hpp 
BASEDEF := -DBLOCKSIZE=128 -DMAX_NODE_DIM=32

all: edge_omp edge_gpu edge_gpu_2l edge_gpu_ca edge_gpu_cw edge_gpu_2l_r edge_gpu_ca_r edge_gpu_cw_r

edge_cpu: Makefile ${BASECPP} ${BASEHPP} dummy_reordering.cpp dummycoloring.cpp cache_calc.cpp\
		seqkernels.cpp 
	g++ ${CPPFLAGS} ${BASEDEF} ${BASECPP} dummy_reordering.cpp dummycoloring.cpp cache_calc.cpp\
		seqkernels.cpp -o edge_cpu

edge_omp_reord: Makefile ${BASECPP} ${BASEHPP} reordering.cpp cache_calc.cpp globalcoloring.cpp ompkernels.cpp
	g++ ${CPPFLAGS} -fopenmp ${BASEDEF} ${BASECPP} reordering.cpp globalcoloring.cpp cache_calc.cpp\
	   ompkernels.cpp -o edge_omp_reord	

edge_omp: Makefile ${BASECPP} ${BASEHPP} dummy_reordering.cpp cache_calc.cpp globalcoloring.cpp ompkernels.cpp
	g++ ${CPPFLAGS} -fopenmp ${BASEDEF} ${BASECPP} dummy_reordering.cpp globalcoloring.cpp cache_calc.cpp\
	   ompkernels.cpp -o edge_omp
	
edge_gpu: Makefile ${BASECPP} ${BASEHPP} dummy_reordering.cpp cache_calc.cpp globalcoloring.cpp cudakernels.cu
	nvcc -DUSE_CUDA ${BASEDEF}  ${BASECPP} dummy_reordering.cpp\
		globalcoloring.cpp cudakernels.cu cache_calc.cpp -o edge_gpu \
		$(INC) $(NVCCFLAGS)	$(CFLAGS) $(LIB)

edge_gpu_2l: Makefile ${BASECPP} ${BASEHPP} dummy_reordering.cpp cache_calc.cpp two_level_coloring.cpp cuda2lkernels.cu
	nvcc -DUSE_CUDA  ${BASEDEF} ${BASECPP} dummy_reordering.cpp\
		two_level_coloring.cpp cuda2lkernels.cu cache_calc.cpp -o edge_gpu_2l \
		$(INC) $(NVCCFLAGS)	$(CFLAGS) $(LIB)

edge_gpu_2l_r: Makefile ${BASECPP} ${BASEHPP} reordering.cpp cache_calc.cpp two_level_coloring.cpp cuda2lkernels.cu
	nvcc -DUSE_CUDA ${BASEDEF} ${BASECPP} reordering.cpp\
		two_level_coloring.cpp cuda2lkernels.cu cache_calc.cpp -o edge_gpu_2l_r \
		$(INC) $(NVCCFLAGS)	$(CFLAGS) $(LIB)

edge_gpu_ca: Makefile ${BASECPP} ${BASEHPP} dummy_reordering.cpp cache_calc.cpp two_level_coloring.cpp cuda_cache_kernels.cu
	nvcc -DUSE_CUDA  ${BASEDEF} ${BASECPP} dummy_reordering.cpp\
		two_level_coloring.cpp cuda_cache_kernels.cu cache_calc.cpp -o edge_gpu_ca \
		$(INC) $(NVCCFLAGS)	$(CFLAGS) $(LIB)

edge_gpu_ca_r: Makefile ${BASECPP} ${BASEHPP} reordering.cpp cache_calc.cpp two_level_coloring.cpp cuda_cache_kernels.cu
	nvcc -DUSE_CUDA  ${BASEDEF} ${BASECPP} reordering.cpp\
		two_level_coloring.cpp cuda_cache_kernels.cu cache_calc.cpp -o edge_gpu_ca_r \
		$(INC) $(NVCCFLAGS)	$(CFLAGS) $(LIB)

edge_gpu_cw:Makefile ${BASECPP} ${BASEHPP} dummy_reordering.cpp cache_calc.cpp two_level_coloring.cpp cuda_cw_kernels.cu
	nvcc -DUSE_CUDA  ${BASEDEF} ${BASECPP} dummy_reordering.cpp\
		two_level_coloring.cpp cuda_cw_kernels.cu cache_calc.cpp -o edge_gpu_cw \
	$(INC) $(NVCCFLAGS)	$(CFLAGS) $(LIB)

edge_gpu_cw_r:Makefile ${BASECPP} ${BASEHPP} reordering.cpp cache_calc.cpp two_level_coloring.cpp cuda_cw_kernels.cu
	nvcc -DUSE_CUDA ${BASEDEF} ${BASECPP} reordering.cpp\
		two_level_coloring.cpp cuda_cw_kernels.cu cache_calc.cpp -o edge_gpu_cw_r \
	$(INC) $(NVCCFLAGS)	$(CFLAGS) $(LIB)

clean:
	rm -rf *.o
	rm -f edge_cpu edge_omp edge_gpu*
