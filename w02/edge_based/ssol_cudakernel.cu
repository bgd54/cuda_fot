#include "kernels.hpp"

__global__ void ssoln(float* old, const float* val, const int nnode, const int node_dim){
  int tid = blockDim.x*blockIdx.x+threadIdx.x;
  if(tid < nnode*node_dim){
    old[tid]=val[tid];
  }
}

void ssoln(const int nnode, const int node_dim,
   const arg& arg_node_val, arg& arg_node_old, Kernel& timer){
  float* node_old_d = (float*) arg_node_old.data_d;
  float* node_val_d = (float*) arg_node_val.data_d;
  timer.timerStart();
  ssoln<<<(nnode*node_dim-1)/BLOCKSIZE+1,BLOCKSIZE>>>(node_old_d,node_val_d, nnode, node_dim);
  checkCudaErrors( cudaDeviceSynchronize() );
  timer.timerStop();
}
