INC	:= -I$(CUDA_HOME)/include -I. -I ..
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart

MAIN_SRC = graph.cu generate_grid.cu apply_reorder.cu
AUX_SRC = colouring.cu data_t.cu partition.cu
HDR = $(wildcard *.hpp) $(wildcard kernels/*.hpp)
AUX_OBJ = $(patsubst %.cu,%.o,$(AUX_SRC))
TGT = $(patsubst %.cu,%,$(MAIN_SRC))

NVCCFLAGS	:= -arch=sm_60 --use_fast_math
NVCCFLAGS   += -std=c++11 -Xcompiler -Wall,-Wextra,-fopenmp
#NVCCFLAGS   += --ptxas-options=-v

OPTIMIZATION_FLAGS := -g -lineinfo -G
#OPTIMIZATION_FLAGS := -g -pg -Xcompiler=-fno-inline -lineinfo
#OPTIMIZATION_FLAGS := -O3 -DNDEBUG

SCOTCH_FLAGS := -lscotch -lscotcherr -lm -I/home/mgiles/ireguly/software/scotch_5.1.12-gnu/include/
SCOTCH_FLAGS += -L/home/mgiles/ireguly/software/scotch_5.1.12-gnu/lib/

METIS_FLAGS := -I/home/mgiles/ireguly/software/parmetis-4.0.3-gnu/include
METIS_FLAGS += -lmetis -L/home/mgiles/ireguly/software/parmetis-4.0.3-gnu/lib/

MESH_DIM ?= 2
VERBOSE ?= no
ifeq ($(VERBOSE), no)
MACRO_VERBOSE =
else
MACRO_VERBOSE = -DVERBOSE_TEST -DUSE_TIMER_MACRO
endif

all: $(TGT)

%: %.cu Makefile $(HDR) $(AUX_OBJ)
	nvcc $< $(AUX_OBJ) -o $@ $(INC) $(METIS_FLAGS) $(NVCCFLAGS) $(LIB) $(OPTIMIZATION_FLAGS) $(SCOTCH_FLAGS) -DMY_SIZE="std::uint32_t" -DMESH_DIM_MACRO=$(MESH_DIM) $(MACRO_VERBOSE)

%.o: %.cu Makefile $(HDR)
	nvcc $< -c -o $@ $(INC) $(METIS_FLAGS) $(NVCCFLAGS) $(LIB) $(OPTIMIZATION_FLAGS) $(SCOTCH_FLAGS) -DMY_SIZE="std::uint32_t" -DMESH_DIM_MACRO=$(MESH_DIM) $(MACRO_VERBOSE)

clean:
		rm -f $(TGT)

debug: Makefile $(HDR)
		@echo $(TGT)
		@echo $<
