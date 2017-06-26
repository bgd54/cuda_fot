INC	:= -I$(CUDA_HOME)/include -I. -I ..
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart

MAIN_SRC = graph.cu test_scotch.cu apply_reorder.cu
AUX_SRC = reorder.cu
HDR = $(wildcard *.hpp)
TGT = $(patsubst %.cu,%,$(MAIN_SRC))

NVCCFLAGS	:= -lineinfo -arch=sm_35 --ptxas-options=-v --use_fast_math
NVCCFLAGS   += -std=c++11 -Xcompiler -Wall,-Wextra,-fopenmp

OPTIMIZATION_FLAGS := -g
#OPTIMIZATION_FLAGS := -O3

SCOTCH_FLAGS := -lscotch -lscotcherr -lm -I/home/software/scotch_5.1.12/include/
SCOTCH_FLAGS += -L/home/software/scotch_5.1.12/lib/

all: $(TGT)

%: %.cu Makefile $(HDR) $(AUX_SRC)
	nvcc $< $(AUX_SRC) -o $@ $(INC) $(NVCCFLAGS) $(LIB) $(OPTIMIZATION_FLAGS) $(SCOTCH_FLAGS) -DMY_SIZE="std::uint32_t"

clean:
		rm -f $(TGT)

debug: Makefile $(HDR)
		@echo $(TGT)
		@echo $<
