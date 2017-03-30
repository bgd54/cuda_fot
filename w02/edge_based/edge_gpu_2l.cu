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

using namespace std;

#define BLOCKSIZE 128


void addTimers(Simulation &sim){
  #ifdef TIMER_MACRO
  sim.timers.push_back(timer("color"));
  sim.timers.push_back(timer("colorb"));
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

__global__ void iter_calc(const float* old, float* val,const float* eval,
    const int* enode, const int* color_reord, const int nedge,
    const int* color, const int* colornum, const int* blocksInColor,
    int color_start, const int node_dim){

  int tid = threadIdx.x;
  
  int bIdx = blocksInColor[blockIdx.x+color_start];
  int reordIdx = tid + bIdx*blockDim.x;
  float* increment = new float[node_dim];
  if(reordIdx < nedge){
    int edgeIdx=color_reord[reordIdx];
    for(int dim=0; dim<node_dim;dim++){ 
      increment[dim]+=
        eval[edgeIdx]*old[enode[edgeIdx*2+0]*node_dim+dim];
    }
  }
  for(int col=0; col<colornum[bIdx];++col){
    if(reordIdx < nedge && col == color[reordIdx]){
      int edgeIdx=color_reord[reordIdx];
      for(int dim=0; dim<node_dim;dim++){ 
        val[enode[2*edgeIdx+1]*node_dim+dim]+= increment[dim];
      }
    }
    __syncthreads();
  }
  delete[] increment;
  //cachelt ertekek visszairasa

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
  printf("coloring ready, allocate arrays in device memory\n");
  
  /////////////////////////////////////
  //          Device pointers
  /////////////////////////////////////
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
      iter_calc<<<len,BLOCKSIZE>>>(node_old_d,
          node_val_d, edge_val_d, enode_d, color_reord_d, nedge, color_d,
          colornum_d, block_reord_d, start, node_dim);
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
