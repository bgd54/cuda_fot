INC	:= -I$(CUDA_HOME)/include -I. -I ..
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart

MAIN_SRC = testVTK.cu
HDR = $(wildcard *.hpp)
TGT = $(patsubst %.cu,%,$(MAIN_SRC))

NVCCFLAGS	:= -lineinfo -arch=sm_35 --ptxas-options=-v --use_fast_math
NVCCFLAGS   += -std=c++11 -Xcompiler -Wall,-Wextra,-fopenmp

OPTIMIZATION_FLAGS := -g
#OPTIMIZATION_FLAGS := -O3

SCOTCH_FLAGS := -lscotch -lscotcherr -lm
SCOTCH_FLAGS += -I/home/mgiles/ireguly/software/scotch_5.1.12/include/
SCOTCH_FLAGS += -L/home/mgiles/ireguly/software/scotch_5.1.12/lib/

METIS_FLAGS := -I/home/mgiles/ireguly/software/parmetis-4.0.3-gnu/include/
METIS_FLAGS += -lmetis -L/home/mgiles/ireguly/software/parmetis-4.0.3-gnu/lib/

MESH_DIM ?= 2

all: $(TGT)

%: %.cu Makefile $(HDR)
	nvcc $< -o $@ $(INC) $(METIS_FLAGS) $(NVCCFLAGS) $(LIB) $(OPTIMIZATION_FLAGS) $(SCOTCH_FLAGS)  -DMY_SIZE="std::uint32_t" -DMESH_DIM_MACRO=$(MESH_DIM)

clean:
		rm -f $(TGT)
