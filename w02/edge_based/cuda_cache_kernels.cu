#include "helper_cuda.h"
#include "helper_string.h"
#include "kernels.hpp"
#include "ssol_cudakernel.cu"

__global__ void iter_calc( const float* __restrict__  old, 
    float* __restrict__ val, const int node_dim ,const float* __restrict__ eval,
    const int nedge, const int* __restrict__ enode, 
    const int* __restrict__ color_reord, const int* __restrict__ threadcolors,
    const int* __restrict__ colornum, const int* __restrict__ blocksInColor,
    const int color_start, const int* __restrict__ global_to_cache, 
    const int* __restrict__ cacheOffsets,
    const int* __restrict__ global_read_to_cache,
    const int* __restrict__ cacheReadOffsets,
    const MY_IDX_TYPE* __restrict__ writeC,
    const MY_IDX_TYPE* __restrict__ readC ){

  int tid = threadIdx.x;
  extern  __shared__  float shared[];


  int bIdx = blocksInColor[blockIdx.x+color_start];
  int reordIdx = tid + bIdx*blockDim.x;

  int iwritethisIdx = -1;
  int ireadthisIdx = -1;

  if(reordIdx<nedge){
    iwritethisIdx = writeC[reordIdx];
    ireadthisIdx = readC[reordIdx];
  }
  //calc cache params
  int cache_offset = bIdx == 0? 0:cacheOffsets[bIdx-1]; 
  int cache_size = cacheOffsets[bIdx] - cache_offset;
  int read_cache_offset = bIdx == 0? 0:cacheReadOffsets[bIdx-1]; 
  int read_cache_size = cacheReadOffsets[bIdx] - read_cache_offset;
  //set pointers to cache
  float* valC = shared;
  float* oldC = shared + cache_size;
  //CACHE IN
  for (int i = 0; i < cache_size; i += blockDim.x) {
    if (i + tid < cache_size) {
      for(int dim=0; dim<node_dim; dim++)
        valC[(i + tid)*node_dim+dim] =
            val[global_to_cache[cache_offset + i + tid]*node_dim+dim];
    }
  }
  for (int i = 0; i < read_cache_size; i += blockDim.x) {
    if (i + tid < read_cache_size) {
      for(int dim=0; dim<node_dim; dim++)
        oldC[(i + tid)*node_dim+dim] = old[
              global_read_to_cache[read_cache_offset + i + tid]*node_dim+dim];
    }
  }
 
  __syncthreads();

  //CALC INCREMENT
  float increment[MAX_NODE_DIM];
  int edgeIdx=0;
  int mycolor=-1;
  if(reordIdx < nedge){
    edgeIdx=color_reord[reordIdx];
    mycolor = threadcolors[reordIdx];
    for(int dim=0; dim<node_dim;dim++){ 
      increment[dim] = eval[edgeIdx]*oldC[ireadthisIdx*node_dim+dim];
    }
  }

  //CALC VAL
  for(int col=0; col<colornum[bIdx];++col){
    if(reordIdx < nedge && col == mycolor){
      for(int dim=0; dim<node_dim;dim++){ 
        valC[iwritethisIdx*node_dim+dim]+= increment[dim];
      }
    }
    __syncthreads();
  }

  //CACHE BACK
  for (int i = 0; i < cache_size; i += blockDim.x) {
    if (i + tid < cache_size) {
      for(int dim=0; dim<node_dim; dim++)
        val[global_to_cache[cache_offset + i + tid]*node_dim+dim] = 
          valC[(i + tid)*node_dim+dim];
    }
  }
 
}


void iter_calc(const int nedge, const int nnode, const int node_dim,
   const Block_coloring& bc, const Coloring& c, const arg& arg_enode,
   const arg& arg_edge_val, arg& arg_node_val, const arg& arg_node_old,
   cacheMap& cm, Kernel& timer){

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
    //TODO shared memory calc.. 4*->worst case
    iter_calc<<<len,BLOCKSIZE,4*BLOCKSIZE*node_dim*sizeof(float)>>>(
        node_old_d, node_val_d, node_dim, edge_val_d, nedge, enode_d,
        color_reord_d, color_d, colornum_d, block_reord_d, start,
        cm.globalToCacheMap_d, cm.blockOffsets_d,
        cm.globalReadToCacheMap_d, cm.blockReadOffsets_d,
        cm.writeC_d, cm.readC_d);
    checkCudaErrors( cudaDeviceSynchronize() );
    timer.timerStop();
  }

}
