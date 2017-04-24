#include "helper_cuda.h"
#include "helper_string.h"
#include "kernels.hpp"
#include "ssol_cudakernel.cu"

__global__ void iter_calc(const float* __restrict__ old, float* __restrict__ val,const float* __restrict__ eval,
    const int* __restrict__ enode, const int* __restrict__ color_reord, const int nedge,
    const int* __restrict__ color, const int* __restrict__ colornum, const int* __restrict__ blocksInColor,
    int color_start, const int node_dim){

  int tid = threadIdx.x;
  
  int bIdx = blocksInColor[blockIdx.x+color_start];
  int reordIdx = tid + bIdx*blockDim.x;
  float increment[MAX_NODE_DIM];
  int edgeIdx=0;
  int mycolor=-1;
  if(reordIdx < nedge){
    edgeIdx=color_reord[reordIdx];
    mycolor = color[reordIdx];
    for(int dim=0; dim<node_dim;dim++){ 
      increment[dim] =
        eval[edgeIdx]*old[enode[edgeIdx*2+0]*node_dim+dim];
    }
  }

  //CALC VAL
  for(int col=0; col<colornum[bIdx];++col){
    if(reordIdx < nedge && col == mycolor){
      for(int dim=0; dim<node_dim;dim++){ 
        val[enode[2*edgeIdx+1]*node_dim+dim]+= increment[dim];
      }
    }
    __syncthreads();
  }

}

void iter_calc(const int nedge, const int nnode, const int node_dim,
   const Block_coloring& bc, const Coloring& c, const arg& arg_enode,
   const arg& arg_edge_val, arg& arg_node_val, const arg& arg_node_old,
   Kernel& timer){

  int* enode_d = (int*) arg_enode.data_d;
  int* color_reord_d = (int *) bc.arg_color_reord.data_d;
  int* color_d = (int *) bc.arg_reordcolor.data_d;
  int* colornum_d = (int *) bc.arg_colornum.data_d;
  int* block_reord_d = (int *) c.arg_color_reord.data_d;
  float * node_val_d = (float*) arg_node_val.data_d;
  float * node_old_d = (float*) arg_node_old.data_d;
  float * edge_val_d = (float*) arg_edge_val.data_d;

  //calc next step
  for(int col=0; col<c.colornum;col++){ 
    int start = col==0?0:c.color_offsets[col-1]; 
    int len = c.color_offsets[col]-start;

    timer.timerStart();
    iter_calc<<<len,BLOCKSIZE>>>(node_old_d,
        node_val_d, edge_val_d, enode_d, color_reord_d, nedge, color_d,
        colornum_d, block_reord_d, start, node_dim);
    checkCudaErrors( cudaDeviceSynchronize() );
    timer.timerStop();
  }
}
