#include "arg.hpp"
#include <cstdlib>
#ifdef USE_CUDA
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "helper_cuda.h"
#include "helper_string.h"
#endif

arg::arg(int set_s, int set_d, int ds, char* _data):
  set_size(set_s), set_dim(set_d), data_size(ds), data(_data){
#ifdef USE_CUDA
    int bytes = set_size*set_dim*data_size;
    checkCudaErrors( cudaMalloc((void**)&data_d, bytes) );
    checkCudaErrors( 
        cudaMemcpy(data_d, data, bytes,  cudaMemcpyHostToDevice) );
#endif
}

arg::~arg(){ 
  free(data);
#ifdef USE_CUDA
  checkCudaErrors( cudaFree(data_d) );
#endif
}

void arg::set_data(int set_s, int set_d, int ds, char* _data){
  set_size = set_s;
  set_dim = set_d;
  data_size = ds;

  free(data);
  data = _data;

#ifdef USE_CUDA
  checkCudaErrors( cudaFree(data_d) );

  int bytes = set_size*set_dim*data_size;
  checkCudaErrors( cudaMalloc((void**)&data_d, bytes) );
  checkCudaErrors( cudaMemcpy(data_d, data, bytes,  cudaMemcpyHostToDevice) );
#endif

}

void arg::update(){ 
#ifdef USE_CUDA
  int bytes = set_size*set_dim*data_size;
  checkCudaErrors( 
        cudaMemcpy(data, data_d, bytes,  cudaMemcpyDeviceToHost) );
#endif
}
void arg::updateD(){ 
#ifdef USE_CUDA
  int bytes = set_size*set_dim*data_size;
  checkCudaErrors( 
        cudaMemcpy(data_d, data, bytes,  cudaMemcpyHostToDevice) );
#endif
}


