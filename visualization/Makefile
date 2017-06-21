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

testVTK: testVTK.cu $(HDR) Makefile
		nvcc testVTK.cu -o $@ $(INC) $(NVCCFLAGS) $(LIB) $(OPTIMIZATION_FLAGS) -DMY_SIZE="unsigned int"


%: %.cu Makefile $(HDR)
		nvcc $< -o $@ $(INC) $(NVCCFLAGS) $(LIBS) $(OPTIMIZATION_FLAGS)

clean:
		rm -f $(TGT)

debug:
		@echo $(TGT)
