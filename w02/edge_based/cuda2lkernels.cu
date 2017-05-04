#include "helper_cuda.h"
#include "helper_string.h"
#include "kernels.hpp"
#include "ssol_cudakernel.cu"

template<int node_dim>
__global__ void iter_calc(const float* __restrict__ old, float* __restrict__ val,
    const float* __restrict__ eval, const int* __restrict__ enode, 
    const int nedge, const int* __restrict__ color, 
    const int* __restrict__ colornum, const int color_start, 
    const int nnode){

  int bIdx = blockIdx.x+color_start;
  int tid = threadIdx.x + bIdx*blockDim.x;
  
  float increment[node_dim];
  int mycolor=-1;
  if(tid < nedge){
    mycolor = color[tid];
    for(int dim=0; dim<node_dim;dim++){
    #ifdef USE_SOA
      int nodeind = enode[tid*2+0] + nnode * dim;
    #else
      int nodeind = enode[tid*2+0]*node_dim + dim;
    #endif
      increment[dim] = eval[tid]*old[nodeind];
    }
  }

  //CALC VAL
  for(int col=0; col<colornum[bIdx];++col){
    if(tid < nedge && col == mycolor){
      for(int dim=0; dim<node_dim;dim++){ 
      #ifdef USE_SOA
        int nodeind = enode[tid*2+1] + nnode * dim;
      #else
        int nodeind = enode[tid*2+1]*node_dim + dim;
      #endif
        val[nodeind] += increment[dim];
      }
    }
    __syncthreads();
  }
  //You can try overlapping loads. This should help with SoA
  //SOA, NDIM=4 6%
  //AOS, NDIM=4 0%
  /*
  for(int col=0; col<colornum[bIdx];++col){
    if(tid < nedge && col == mycolor){
      #pragma unroll node_dim
      for(int dim=0; dim<node_dim;dim++){ 
      #ifdef USE_SOA
        int nodeind = enode[tid*2+1] + nnode * dim;
      #else
        int nodeind = enode[tid*2+1]*node_dim + dim;
      #endif
         increment[dim] += val[nodeind];
      }
      #pragma unroll node_dim
      for(int dim=0; dim<node_dim;dim++){ 
      #ifdef USE_SOA
        int nodeind = enode[tid*2+1] + nnode * dim;
      #else
        int nodeind = enode[tid*2+1]*node_dim + dim;
      #endif
        val[nodeind] = increment[dim];
      }
    }
    __syncthreads();
  }*/

}

void iter_calc(const int nedge, const int nnode, const int node_dim,
   const Block_coloring& bc, const Coloring& c, const arg& arg_enode,
   const arg& arg_edge_val, arg& arg_node_val, const arg& arg_node_old,
   cacheMap& cm, Kernel& timer){

  int* enode_d = (int*) arg_enode.data_d;
  int* color_d = (int *) bc.arg_reordcolor.data_d;
  int* colornum_d = (int *) bc.arg_colornum.data_d;
  float * node_val_d = (float*) arg_node_val.data_d;
  float * node_old_d = (float*) arg_node_old.data_d;
  float * edge_val_d = (float*) arg_edge_val.data_d;

  //calc next step
  for(int col=0; col<c.colornum;col++){ 
    int start = col==0?0:c.color_offsets[col-1]; 
    int len = c.color_offsets[col]-start;

    timer.timerStart();
    iter_calc<NODE_DIM><<<len,BLOCKSIZE>>>(node_old_d,
        node_val_d, edge_val_d, enode_d, nedge, color_d,
        colornum_d, start, nnode);
    checkCudaErrors( cudaDeviceSynchronize() );
    timer.timerStop();
  }
}
