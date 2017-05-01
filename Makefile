INC	:= -I$(CUDA_HOME)/include -I. -I ..
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart

SRC = $(wildcard *.cu)
HDR = $(wildcard *.hpp)
TGT = $(patsubst %.cu,%,$(SRC))

NVCCFLAGS	:= -lineinfo -arch=sm_35 --ptxas-options=-v --use_fast_math
NVCCFLAGS   += -std=c++11 -Xcompiler -Wall,-Wextra

OPTIMIZATION_FLAGS := -g
#OPTIMIZATION_FLAGS := -O3

SCOTCH_FLAGS := -lscotch -lscotcherr -lm -I/home/software/scotch_5.1.12/include/
SCOTCH_FLAGS += -L/home/software/scotch_5.1.12/lib/

all: $(TGT)

graph: graph.cu graph.hpp problem.hpp reorder.hpp reorder.cpp colouring.hpp Makefile
	nvcc graph.cu reorder.cpp -o $@ $(INC) $(NVCCFLAGS) $(LIBS) $(OPTIMIZATION_FLAGS) $(SCOTCH_FLAGS) -DMY_SIZE="std::uint32_t"

test_scotch: graph.hpp problem.hpp reorder.hpp reorder.cpp colouring.hpp Makefile test_scotch.cpp
		g++ test_scotch.cpp reorder.cpp -o $@ $(INC) -std=c++11 $(OPTIMIZATION_FLAGS) $(SCOTCH_FLAGS) -DMY_SIZE="unsigned int"

%: %.cu Makefile $(HDR)
		nvcc $< -o $@ $(INC) $(NVCCFLAGS) $(LIBS) $(OPTIMIZATION_FLAGS)

clean:
		rm -f $(TGT)

debug:
		@echo $(TGT)
