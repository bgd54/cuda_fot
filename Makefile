INC	:= -I$(CUDA_HOME)/include -I. -I ..
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart

SRC = $(wildcard *.cu)
HDR = $(wildcard *.hpp)
TGT = $(patsubst %.cu,%,$(SRC))

NVCCFLAGS	:= -lineinfo -arch=sm_35 --ptxas-options=-v --use_fast_math
NVCCFLAGS   += -std=c++11 -Xcompiler -Wall,-Wextra,-fopenmp

OPTIMIZATION_FLAGS := -g
#OPTIMIZATION_FLAGS := -O3

SCOTCH_FLAGS := -lscotch -lscotcherr -lm -I/home/software/scotch_5.1.12/include/
SCOTCH_FLAGS += -L/home/software/scotch_5.1.12/lib/

all: $(TGT)

graph: graph.cu $(HDR) reorder.cu Makefile
	nvcc graph.cu reorder.cu -o $@ $(INC) $(NVCCFLAGS) $(LIBS) $(OPTIMIZATION_FLAGS) $(SCOTCH_FLAGS) -DMY_SIZE="std::uint32_t"

test_scotch: $(HDR) reorder.cu  Makefile test_scotch.cu
	nvcc test_scotch.cu reorder.cu -o $@ $(INC) $(NVCCFLAGS) $(LIB) $(OPTIMIZATION_FLAGS) $(SCOTCH_FLAGS) -DMY_SIZE="unsigned int"

apply_reorder: apply_reorder.cu $(HDR) reorder.cu Makefile
	nvcc apply_reorder.cu reorder.cu -o $@ $(INC) $(NVCCFLAGS) $(LIB) $(OPTIMIZATION_FLAGS) $(SCOTCH_FLAGS) -DMY_SIZE="unsigned int"

%: %.cu Makefile $(HDR)
		nvcc $< -o $@ $(INC) $(NVCCFLAGS) $(LIBS) $(OPTIMIZATION_FLAGS)

clean:
		rm -f $(TGT)

debug:
		@echo $(TGT)
