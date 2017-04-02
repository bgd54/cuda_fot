#include "helper_cuda.h"
#include <cuda.h>
#include <stdio.h>
#include <string>
#include <string.h>
#include "graph_helper.hpp"
#include "rms.hpp"
#define TIMER_MACRO
#include "simulation.hpp"
#include "coloring.hpp"
#include "cache_calc.hpp"

using namespace std;

#define BLOCKSIZE 128

////////////////////////////////////////////////////////////////////////////////
// CPU routines
////////////////////////////////////////////////////////////////////////////////

void addTimers(Simulation &sim){
  #ifdef TIMER_MACRO
  sim.timers.push_back(timer("color"));
  sim.timers.push_back(timer("colorb"));
  sim.timers.push_back(timer("calc_cache"));
  #endif
}

////////////////////////////////////////////////////////////////////////////////
// GPU routines
////////////////////////////////////////////////////////////////////////////////
__global__ void ssoln(float* old, const float* val, const int nnode, const int node_dim){
  int tid = blockDim.x*blockIdx.x+threadIdx.x;
  if(tid < nnode*node_dim){
    old[tid]=val[tid];
  }
}

__global__ void iter_calc(const float* __restrict__  old, 
    float* __restrict__ val, const int node_dim ,const float* __restrict__ eval,
    const int nedge, const int* __restrict__ enode, 
    const int* __restrict__ color_reord, const int* __restrict__ threadcolors,
    const int* __restrict__ colornum, const int* __restrict__ blocksInColor,
    int color_start, const int* __restrict__ global_to_cache, 
    const int* __restrict__ cacheOffsets,
    const MY_IDX_TYPE* __restrict__ writeC){

  int tid = threadIdx.x;
  extern  __shared__  float shared[];


  int bIdx = blocksInColor[blockIdx.x+color_start];
  int reordIdx = tid + bIdx*blockDim.x;

  int iwritethisIdx = -1;

  if(reordIdx<nedge){
    iwritethisIdx = writeC[reordIdx];
  }
  //calc cache params
  int cache_offset = bIdx == 0? 0:cacheOffsets[bIdx-1]; 
  int cache_size = cacheOffsets[bIdx] - cache_offset;
  //CACHE IN
  for (int i = 0; i < cache_size; i += blockDim.x) {
    if (i + tid < cache_size) {
      for(int dim=0; dim<node_dim; dim++)
        shared[(i + tid)*node_dim+dim] =
            val[global_to_cache[cache_offset + i + tid]*node_dim+dim];
    }
  }
    
  __syncthreads();

  //CALC INCREMENT
  float* increment = new float[node_dim];
  if(reordIdx < nedge){
    for(int dim=0; dim<node_dim;dim++){ 
      increment[dim] = eval[color_reord[reordIdx]] *
        old[enode[color_reord[reordIdx]*2+0]*node_dim+dim];
    }
  }

  //CALC VAL
  for(int col=0; col<colornum[bIdx];++col){
    if(reordIdx < nedge && col == threadcolors[reordIdx]){
      for(int dim=0; dim<node_dim;dim++){ 
        shared[iwritethisIdx*node_dim+dim]+= increment[dim];
      }
    }
    __syncthreads();
  }
  delete[] increment;

  //CACHE BACK
  for (int i = 0; i < cache_size; i += blockDim.x) {
    if (i + tid < cache_size) {
      for(int dim=0; dim<node_dim; dim++)
        val[global_to_cache[cache_offset + i + tid]*node_dim+dim] = 
          shared[(i + tid)*node_dim+dim];
    }
  }
  
}

///___________________________________________________________________________
int main(int argc, char *argv[]){
  int niter=1000;
  int dx = 1000, dy = 2000;
  bool bidir=false;
  int node_dim = 1, edge_dim = 1;
  ///////////////////////////////////////////////////////////////////////
  //                            params
  ///////////////////////////////////////////////////////////////////////
  for(int i=1; i < argc; ++i){
    if (!strcmp(argv[i],"-niter")) niter=atoi(argv[++i]);
    else if (!strcmp(argv[i],"-dx")) dx=atoi(argv[++i]);
    else if (!strcmp(argv[i],"-dy")) dy=atoi(argv[++i]);
    else if (!strcmp(argv[i],"-bidir")) bidir=true;
    else if (!strcmp(argv[i],"-ndim")) node_dim=atoi(argv[++i]);
    else {
      fprintf(stderr,"Error: Command-line argument '%s' not recognized.\n",
          argv[i]);
      exit(-1);
    }
  }
  ///////////////////////////////////////////////////////////////////////
  //                            graph gen
  ///////////////////////////////////////////////////////////////////////

  int nnode, nedge;
  int* enode = bidir ? 
    generate_bidirected_graph(dx,dy,nedge,nnode) : 
    generate_graph(dx,dy,nedge,nnode);

  float *node_val, *node_old, *edge_val;
  
  node_val = genDataForNodes(nnode,node_dim);
  edge_val = genDataForNodes(nedge,edge_dim);
  
  node_old=(float*)malloc(nnode*node_dim*sizeof(float));
  ///////////////////////////////////////////////////////////////////////
  //                            timer
  ///////////////////////////////////////////////////////////////////////
  Simulation sim = initSimulation(nedge, nnode, node_dim);
  addTimers(sim);


  /////////////////////////////////////////////////////////
  //                        coloring
  /////////////////////////////////////////////////////////
  
  printf("start coloring\n");
  TIMER_START(sim.timers[0])
  Block_coloring c = block_coloring(enode,nedge);
  TIMER_STOP(sim.timers[0])
  printf("start coloring blocks\n");
  TIMER_START(sim.timers[1])
  Coloring bc = c.color_blocks(enode,nedge);
  TIMER_STOP(sim.timers[1])
  printf("ready\n");
  printf("calculate cacheable data\n");
  TIMER_START(sim.timers[2])
 
  cacheMap cm = genCacheMap(enode, nedge, c);

  TIMER_STOP(sim.timers[2])

  /////////////////////////////////////
  //          Device pointers
  /////////////////////////////////////
  printf("coloring ready, allocate arrays in device memory\n");
  int *enode_d, *color_reord_d, *colornum_d, *color_d;
  float *node_val_d,*node_old_d,*edge_val_d;
  int *block_reord_d;

  checkCudaErrors( cudaMalloc((void**)&enode_d, 2*nedge*sizeof(int)) );
  checkCudaErrors( cudaMalloc((void**)&color_reord_d, nedge*sizeof(int)) );
  checkCudaErrors( cudaMalloc((void**)&color_d, nedge*sizeof(int)) );
  checkCudaErrors( cudaMalloc((void**)&colornum_d, c.numblock*sizeof(int)) );
  checkCudaErrors( cudaMalloc((void**)&edge_val_d, nedge*sizeof(float)) );
  checkCudaErrors( cudaMalloc((void**)&node_old_d, nnode*node_dim*sizeof(float)) );
  checkCudaErrors( cudaMalloc((void**)&node_val_d, nnode*node_dim*sizeof(float)) );
  checkCudaErrors( cudaMalloc((void**)&block_reord_d, c.numblock*sizeof(int)) );
  
  checkCudaErrors( cudaMemcpy(enode_d, enode, 2*nedge*sizeof(int),
                              cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(color_reord_d, c.color_reord,
                               nedge*sizeof(int), cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(color_d, c.reordcolor,
                               nedge*sizeof(int), cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(colornum_d, c.colornum, c.numblock*sizeof(int),
                              cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(edge_val_d, edge_val, nedge*sizeof(float),
                              cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(node_val_d, node_val, nnode*node_dim*sizeof(float),
                              cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(block_reord_d, bc.color_reord,
                               c.numblock*sizeof(int),
                               cudaMemcpyHostToDevice) );

  ///////////////////////////////////////////////////////////
  //                      Start
  ///////////////////////////////////////////////////////////
  printf("start edge based on CPU niter: %d, nnode:%d, nedge:%d, numblock: %d\n",niter,
     nnode,nedge, c.numblock);
  //   timer
  sim.start();
  //______________________________main_loop_____________________________
  for(int i=0;i<=niter;++i){
    //save old
    sim.kernels[0].timerStart();
    ssoln<<<(nnode*node_dim-1)/BLOCKSIZE+1,BLOCKSIZE>>>(node_old_d,node_val_d, nnode, node_dim);
    checkCudaErrors( cudaDeviceSynchronize() );
    sim.kernels[0].timerStop();


    //calc next step
    for(int col=0; col<bc.colornum;col++){ 
      int start = col==0?0:bc.color_offsets[col-1]; 
      int len = bc.color_offsets[col]-start;
      sim.kernels[1].timerStart();
      //TODO shared memory calc.. 4*->worst case
      iter_calc<<<len,BLOCKSIZE,4*BLOCKSIZE*node_dim*sizeof(float)>>>(
          node_old_d, node_val_d, node_dim,  edge_val_d, nedge,  enode_d, 
          color_reord_d, color_d, colornum_d, block_reord_d, start,
          cm.globalToCacheMap_d, cm.blockOffsets_d, cm.writeC_d);
      checkCudaErrors( cudaDeviceSynchronize() );
      sim.kernels[1].timerStop();
    }

    // rms
    if(i%100==0){
      sim.kernels[2].timerStart();
      checkCudaErrors( cudaMemcpy(node_val, node_val_d, nnode*node_dim*sizeof(float),
                              cudaMemcpyDeviceToHost) );
      checkCudaErrors( cudaMemcpy(node_old, node_old_d, nnode*node_dim*sizeof(float),
                              cudaMemcpyDeviceToHost) );
      rms_calc(node_val,node_old,nnode,i,node_dim);
      sim.kernels[2].timerStop();

    }

  }
  //____________________________end main loop___________________________
  //    timer
  sim.stop();

  sim.printTiming();

  
  //free
  free(enode);
  free(node_old);
  free(node_val);
  free(edge_val);
  //cuda freee
  checkCudaErrors( cudaFree(enode_d) );
  checkCudaErrors( cudaFree(color_reord_d) );
  checkCudaErrors( cudaFree(edge_val_d) );
  checkCudaErrors( cudaFree(node_old_d) );
  checkCudaErrors( cudaFree(node_val_d) );
  checkCudaErrors( cudaFree(color_d) );
  checkCudaErrors( cudaFree(colornum_d) );
  checkCudaErrors( cudaFree(block_reord_d) );
  
  return 0;
}
