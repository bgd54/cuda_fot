# Customisable behaviour

# If 'yes' then pass `--ptxas-options=-v` to nvcc
PTXASV      ?=  no
# If 'yes' then use debug flags, if 'prof', use profiling flags
DEBUG       ?=  no
# If 'yes', the resulting program will print verbose test and measurement
# results
VERBOSE     ?=  no
# Scotch installation directory
SCOTCH_DIR  ?=  $(shell echo ${HOME})/software/scotch
# METIS installation directory
METIS_DIR   ?=  $(shell echo ${HOME})/software/metis


INC := -I$(CUDA_HOME)/include -I.
LIB := -L$(CUDA_HOME)/lib64 -lcudart

MAIN_SRC = graph.cu generate_grid.cu apply_reorder.cu
AUX_SRC  = colouring.cu data_t.cu partition.cu
HDR      = $(wildcard *.hpp) $(wildcard kernels/*.hpp)
TGT      = $(patsubst %.cu,%,$(MAIN_SRC))
AUX_OBJ  = $(patsubst %.cu,%.o,$(AUX_SRC))

NVCCFLAGS   := -arch=sm_60 --use_fast_math
NVCCFLAGS   += -std=c++11 -Xcompiler -Wall,-Wextra,-fopenmp
ifeq ($(PTXASV), yes)
  NVCCFLAGS += --ptxas-options=-v
endif

ifeq ($(DEBUG), yes)
  OPTIMIZATION_FLAGS := -g -G
else
ifeq ($(DEBUG), prof)
  OPTIMIZATION_FLAGS := -g -Xcompiler=-fno-inline -lineinfo
else
  OPTIMIZATION_FLAGS := -O3 -DNDEBUG -lineinfo
endif
endif

SCOTCH_FLAGS := -lscotch -lscotcherr -lm
SCOTCH_FLAGS += -I$(SCOTCH_DIR)/include/
SCOTCH_FLAGS += -L$(SCOTCH_DIR)/lib/

METIS_FLAGS := -I$(METIS_DIR)/include
METIS_FLAGS += -lmetis -L$(METIS_DIR)/lib/

MESH_DIM ?= 2
ifeq ($(VERBOSE), no)
MACRO_VERBOSE =
else
MACRO_VERBOSE = -DVERBOSE_TEST -DUSE_TIMER_MACRO
endif

.SECONDARY: $(AUX_OBJ)

all: $(TGT)

%: %.cu Makefile $(HDR) $(AUX_OBJ)
	nvcc $< $(AUX_OBJ) -o $@ $(INC) $(METIS_FLAGS) $(NVCCFLAGS) $(LIB)         \
	    $(OPTIMIZATION_FLAGS) $(SCOTCH_FLAGS) -DMY_SIZE="std::uint32_t"        \
	    -DMESH_DIM_MACRO=$(MESH_DIM) $(MACRO_VERBOSE)

%.o: %.cu Makefile $(HDR)
	nvcc $< -c -o $@ $(INC) $(METIS_FLAGS) $(NVCCFLAGS) $(LIB)                 \
	    $(OPTIMIZATION_FLAGS) $(SCOTCH_FLAGS) -DMY_SIZE="std::uint32_t"        \
	    -DMESH_DIM_MACRO=$(MESH_DIM) $(MACRO_VERBOSE)

clean:
	rm -f $(TGT)
	rm -f $(AUX_OBJ)
