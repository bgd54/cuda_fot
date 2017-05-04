#include "helper_cuda.h"
#include "helper_string.h"
#include "kernels.hpp"
#include "ssol_cudakernel.cu"

template <int node_dim>
__global__ void iter_calc( const float* __restrict__  old, 
    float* __restrict__ val, const int nnode,
    const float* __restrict__ eval, const int nedge,
    const int* __restrict__ enode, const int* __restrict__ threadcolors,
    const int* __restrict__ colornum, const int color_start,
    const int* __restrict__ global_to_cache,
    const int* __restrict__ cacheOffsets,
    const int* __restrict__ global_read_to_cache,
    const int* __restrict__ cacheReadOffsets,
    const MY_IDX_TYPE* __restrict__ writeC,
    const MY_IDX_TYPE* __restrict__ readC ){

  int bIdx = blockIdx.x+color_start;
  int tid = threadIdx.x;
  extern  __shared__  float shared[];


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
  int max_cacge_size = cache_size > read_cache_size ? cache_size : read_cache_size;
  //set pointers to cache
  float* valC = shared;
  float* oldC = shared + cache_size*node_dim;
  //CACHE IN
  for (int i = 0; i < max_cacge_size; i += blockDim.x) {
    if (i + tid < cache_size) {
      for(int dim=0; dim<node_dim; dim++){
        //Not sure what global_to_cache looks like. But it should be sorted for each
        //block, so adjacent threads access adjacent values in memory. readC and
        // writeC of course would have to change alongside.
      #ifdef USE_SOA
        int nodeind = global_to_cache[cache_offset + i + tid]+nnode*dim;
        int cacheind = (i+tid)+cache_size*dim;
      #else
        //For AoS, adjacent threads should read the different dim positions
        int nodeind = global_to_cache[cache_offset + i + tid]*node_dim+dim;
        int cacheind = (i+tid)*node_dim + dim;
      #endif
        valC[cacheind] = val[nodeind];

      }
    }

    if (i + tid < read_cache_size) {
      for(int dim=0; dim<node_dim; dim++){
      #ifdef USE_SOA
        int nodeind = global_read_to_cache[read_cache_offset + i + tid]+nnode*dim;
        int cacheind = (i+tid)+read_cache_size*dim;
      #else
        int nodeind = global_read_to_cache[read_cache_offset + i + tid]*node_dim+dim;
        int cacheind = (i+tid)*node_dim + dim;
      #endif
        oldC[cacheind] = old[nodeind];
      }
    }
  }
/* Something like this for AoS have to double-check though
  for (int i = threadIdx.x; i < cache_size*node_dim; i+=blockDim.x) {
    int nodeind = global_to_cache[cache_offset + i/node_dim]*node_dim+dim%node_dim;
    valC[i] = val[nodeind]
  }
  */
 
  __syncthreads();

  //CALC INCREMENT
  float increment[node_dim];
  int mycolor=-1;
  if(reordIdx < nedge){
    mycolor = threadcolors[reordIdx];
    for(int dim=0; dim<node_dim;dim++){ 
    #ifdef USE_SOA
      int nodeind = ireadthisIdx + read_cache_size * dim;
    #else
      int nodeind = ireadthisIdx*node_dim + dim;
    #endif
      increment[dim] = eval[reordIdx]*oldC[nodeind];
    }
  }

  //You can use about half as much shared memory, if you do not pre-load valC,
  //but instead increment here. Perhaps an additional variant.
  //CALC VAL
  for(int col=0; col<colornum[bIdx];++col){
    if(col == mycolor){
      for(int dim=0; dim<node_dim;dim++){ 
        //val[enode[2*edgeIdx+1]*node_dim+dim] += increment[dim];
      #ifdef USE_SOA
        valC[iwritethisIdx+cache_size*dim] += increment[dim];
      #else
        valC[iwritethisIdx*node_dim+dim] += increment[dim];
      #endif
      }
    }
    __syncthreads();
  }

  //CACHE BACK
  for (int i = 0; i < cache_size; i += blockDim.x) {
    if (i + tid < cache_size) {
      for(int dim=0; dim<node_dim; dim++){
      #ifdef USE_SOA
        int nodeind = global_to_cache[cache_offset + i + tid]+nnode*dim;
        int cacheind = (i+tid)+cache_size*dim;
      #else
        int nodeind = global_to_cache[cache_offset + i + tid]*node_dim+dim;
        int cacheind = (i+tid)*node_dim + dim;
      #endif
        val[nodeind] = valC[cacheind];
      }
    }
  }
 
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
    iter_calc<NODE_DIM><<<len,BLOCKSIZE,cm.maxc*node_dim*sizeof(float)>>>(
        node_old_d, node_val_d, nnode, edge_val_d, nedge, enode_d,
        color_d, colornum_d, start,
        cm.globalToCacheMap_d, cm.blockOffsets_d,
        cm.globalReadToCacheMap_d, cm.blockReadOffsets_d,
        cm.writeC_d, cm.readC_d);
    checkCudaErrors( cudaDeviceSynchronize() );
    timer.timerStop();
  }

}
