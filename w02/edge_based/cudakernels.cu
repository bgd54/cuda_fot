#include "helper_cuda.h"
#include "helper_string.h"
#include "kernels.hpp"
#include "ssol_cudakernel.cu"

template <int node_dim>
__global__ void iter_calc(const float* old, float* val,const float* eval,
    const int* enode, const int* color_reord, const int offset, 
    const int color_size, const int nedge, const int nnode){

  int tid = blockDim.x*blockIdx.x+threadIdx.x;
  int reordIdx = tid + offset;
  if(reordIdx<nedge && tid < color_size){
    int edgeIdx=color_reord[reordIdx];
    for(int dim=0; dim<node_dim;dim++){ 
#ifndef USE_SOA
      val[enode[2*edgeIdx+1]*node_dim+dim]+=
        eval[edgeIdx]*old[enode[edgeIdx*2+0]*node_dim+dim];
#else
      val[nnode*dim + enode[2*edgeIdx+1]] += 
        eval[edgeIdx]*old[nnode*dim + enode[edgeIdx*2+0]];
#endif
    }
  }
}

//Potential extra version, may help particularly for SoA:
/*
template <int node_dim>
__global__ void iter_calc(const float* old, float* val,const float* eval,
    const int* enode, const int* color_reord, const int offset, 
    const int color_size, const int nedge, const int nnode){

  int tid = blockDim.x*blockIdx.x+threadIdx.x;
  int reordIdx = tid + offset;
  float inc[node_dim];
  if(reordIdx<nedge && tid < color_size){
    int edgeIdx=color_reord[reordIdx];
    #pragma unroll
    for(int dim=0; dim<node_dim;dim++){ 
#ifndef USE_SOA
      inc[dim] = val[enode[2*edgeIdx+1]*node_dim+dim]+
        eval[edgeIdx]*old[enode[edgeIdx*2+0]*node_dim+dim];
#else
      inc[dim] = val[nnode*dim + enode[2*edgeIdx+1]] + 
        eval[edgeIdx]*old[nnode*dim + enode[edgeIdx*2+0]];
#endif
    }
    #pragma unroll
    for(int dim=0; dim<node_dim;dim++){ 
#ifndef USE_SOA
      val[enode[2*edgeIdx+1]*node_dim+dim] = inc[dim];
#else
      val[nnode*dim + enode[2*edgeIdx+1]] = inc[dim];
#endif
    }
  }
}
*/

void iter_calc(const int nedge, const int nnode, const int node_dim,
   const Block_coloring& bc, const Coloring& c, const arg& arg_enode,
   const arg& arg_edge_val, arg& arg_node_val, const arg& arg_node_old,
   cacheMap& cm, Kernel& timer){

  int* enode_d = (int*) arg_enode.data_d;
  int* color_reord_d = (int *) c.arg_color_reord.data_d;
  float * node_val_d = (float*) arg_node_val.data_d;
  float * node_old_d = (float*) arg_node_old.data_d;
  float * edge_val_d = (float*) arg_edge_val.data_d;

  //calc next step
  for(int col=0; col<c.colornum;col++){ 
    int color_offset = col==0 ? 0 : c.color_offsets[col-1];
    int color_size = c.color_offsets[col] - color_offset;
    timer.timerStart();
    iter_calc<NODE_DIM><<<(color_size-1)/BLOCKSIZE+1,BLOCKSIZE>>>(node_old_d, 
        node_val_d, edge_val_d, enode_d, color_reord_d, color_offset,
        color_size, nedge, nnode);
    checkCudaErrors( cudaDeviceSynchronize() );
    timer.timerStop();
  }
}
