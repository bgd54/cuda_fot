INC	:= -I$(CUDA_HOME)/include -I. -I ..
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart

SRC = $(wildcard *.cu)
HDR = $(wildcard *.hpp)
TGT = $(patsubst %.cu,%,$(SRC))

NVCCFLAGS	:= -lineinfo -arch=sm_35 --ptxas-options=-v --use_fast_math
NVCCFLAGS   += -std=c++11 -Xcompiler -Wall,-Wextra

OPTIMIZATION_FLAGS := -g
#OPTIMIZATION_FLAGS := -O3

all: $(TGT)

graph: graph.cu graph.hpp problem.hpp colouring.hpp Makefile
		nvcc graph.cu -o $@ $(INC) $(NVCCFLAGS) $(LIBS) $(OPTIMIZATION_FLAGS)

test_scotch: graph.hpp problem.hpp colouring.hpp Makefile test_scotch.cpp
		g++ test_scotch.cpp -o $@ $(INC) -std=c++11 $(OPTIMIZATION_FLAGS)

%: %.cu Makefile $(HDR)
		nvcc $< -o $@ $(INC) $(NVCCFLAGS) $(LIBS) $(OPTIMIZATION_FLAGS)

clean:
		rm -f $(TGT)

debug:
		@echo $(TGT)
