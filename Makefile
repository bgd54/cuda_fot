INC	:= -I$(CUDA_HOME)/include -I. -I ..
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart

MAIN_SRC = graph.cu test_scotch.cu apply_reorder.cu
AUX_SRC =
HDR = $(wildcard *.hpp)
TGT = $(patsubst %.cu,%,$(MAIN_SRC))

NVCCFLAGS	:= -arch=sm_60 --use_fast_math
NVCCFLAGS   += -std=c++11 -Xcompiler -Wall,-Wextra,-fopenmp
#NVCCFLAGS   += --ptxas-options=-v

OPTIMIZATION_FLAGS := -g -lineinfo
#OPTIMIZATION_FLAGS := -g -pg -Xcompiler=-fno-inline -lineinfo
#OPTIMIZATION_FLAGS := -O3 -DNDEBUG

SCOTCH_FLAGS := -lscotch -lscotcherr -lm -I/home/software/scotch_5.1.12/include/
SCOTCH_FLAGS += -L/home/software/scotch_5.1.12/lib/

METIS_FLAGS := -I/home/software/parmetis-gnu/include
METIS_FLAGS += -lmetis -L/home/software/parmetis-gnu/lib/

MESH_DIM ?= 2

all: $(TGT)

%: %.cu Makefile $(HDR) $(AUX_SRC)
	nvcc $< $(AUX_SRC) -o $@ $(INC) $(METIS_FLAGS) $(NVCCFLAGS) $(LIB) $(OPTIMIZATION_FLAGS) $(SCOTCH_FLAGS) -DMY_SIZE="std::uint32_t" -DMESH_DIM_MACRO=$(MESH_DIM)

clean:
		rm -f $(TGT)

debug: Makefile $(HDR)
		@echo $(TGT)
		@echo $<
