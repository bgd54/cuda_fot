INC	:= -I$(CUDA_HOME)/include -I. -I ..
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart

SRC = $(wildcard *.cu)
TGT = $(patsubst %.cu,%,$(SRC))

NVCCFLAGS	:= -lineinfo -arch=sm_35 --ptxas-options=-v --use_fast_math

all: $(TGT)

graph: graph.cu graph.hpp problem.hpp colouring.hpp
		nvcc $< -o $@ $(INC) $(NVCCFLAGS) $(LIBS) -std=c++11 -Xcompiler -Wall,-Wextra -g

%: %.cu
		nvcc $< -o $@ $(INC) $(NVCCFLAGS) $(LIBS) -std=c++11 -Xcompiler -Wall,-Wextra

clean:
		rm -f $(TGT)

debug:
		@echo $(TGT)
