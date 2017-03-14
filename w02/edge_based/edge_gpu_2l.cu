#include "helper_cuda.h"
#include <cuda.h>
#include <stdio.h>
#include <string>
#include <string.h>
#include "graph_helper.hpp"
#include "rms.hpp"
#define TIMER_MACRO
#include "timer.hpp"
#include "coloring.hpp"

using namespace std;

#define BLOCKSIZE 128


////////////////////////////////////////////////////////////////////////////////
// GPU routines
////////////////////////////////////////////////////////////////////////////////
__global__ void ssoln(float* old, const float* val, const int nnode){
  int tid = blockDim.x*blockIdx.x+threadIdx.x;
  if(tid < nnode){
    old[tid]=val[tid];
  }
}

__global__ void iter_calc(const float* old, float* val,const float* eval,
    const int* enode, const int* color_reord, const int nedge,
    const int* color, const int* colornum, const int* blocksInColor,
    int color_start){

  int tid = threadIdx.x;
  
  int bIdx = blocksInColor[blockIdx.x+color_start];
  int reordIdx = tid + bIdx*blockDim.x;
  float increment = 0.0f;
  if(reordIdx < nedge){
    increment = 
      eval[color_reord[reordIdx]]*old[enode[color_reord[reordIdx]*2+0]];
  }
  for(int col=0; col<colornum[bIdx];++col){
    if(reordIdx < nedge && col == color[reordIdx]){
      val[enode[2*color_reord[reordIdx]+1] ] += increment;
    }
    __syncthreads();
  }
  //cachelt ertekek visszairasa

}

///___________________________________________________________________________
int main(int argc, char *argv[]){
  int niter=1000;
  int dx = 1000, dy = 2000;
  ///////////////////////////////////////////////////////////////////////
  //                            params
  ///////////////////////////////////////////////////////////////////////
  for(int i=1; i < argc; ++i){
    if (!strcmp(argv[i],"-niter")) niter=atoi(argv[++i]);
    else if (!strcmp(argv[i],"-dx")) dx=atoi(argv[++i]);
    else if (!strcmp(argv[i],"-dy")) dy=atoi(argv[++i]);
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
  int* enode = generate_graph(dx, dy, nedge, nnode);

  float *node_val, *node_old, *edge_val;
  
  node_val = genDataForNodes(nnode,1);
  edge_val = genDataForNodes(nedge,1);
  
  node_old=(float*)malloc(nnode*sizeof(float));
  ///////////////////////////////////////////////////////////////////////
  //                            timer
  ///////////////////////////////////////////////////////////////////////
#ifdef TIMER_MACRO
  timer total("total"), ssol("ssol"), iter("iter"), rms("rms"), 
        color("color1"), color2("colorb"); //TODO Attila h oldja ezt meg
#endif

  /////////////////////////////////////////////////////////
  //                        coloring
  /////////////////////////////////////////////////////////
  
  printf("start coloring\n");
  TIMER_START(color)
  Block_coloring c = block_coloring(enode,nedge,nnode);
  TIMER_STOP(color)
  printf("start coloring blocks\n");
  TIMER_START(color2)
  Coloring bc = c.color_blocks(enode,nedge);
  TIMER_STOP(color2)
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
  checkCudaErrors( cudaMalloc((void**)&node_old_d, nnode*sizeof(float)) );
  checkCudaErrors( cudaMalloc((void**)&node_val_d, nnode*sizeof(float)) );
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
  checkCudaErrors( cudaMemcpy(node_val_d, node_val, nnode*sizeof(float),
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
  TIMER_START(total)
  //______________________________main_loop_____________________________
  for(int i=0;i<=niter;++i){
    //save old
    TIMER_START(ssol)
    ssoln<<<(nnode-1)/BLOCKSIZE+1,BLOCKSIZE>>>(node_old_d,node_val_d, nnode);
    checkCudaErrors( cudaDeviceSynchronize() );
    TIMER_STOP(ssol)

    //calc next step
    for(int col=0; col<bc.colornum;col++){ 
      int start = col==0?0:bc.color_offsets[col-1]; 
      int len = bc.color_offsets[col]-start;
      TIMER_START(iter)
      iter_calc<<<len,BLOCKSIZE>>>(node_old_d,
          node_val_d, edge_val_d, enode_d, color_reord_d, nedge, color_d,
          colornum_d, block_reord_d, start);
      checkCudaErrors( cudaDeviceSynchronize() );
      TIMER_STOP(iter)
    }

    // rms
    if(i%100==0){
      TIMER_START(rms)
      checkCudaErrors( cudaMemcpy(node_val, node_val_d, nnode*sizeof(float),
                              cudaMemcpyDeviceToHost) );
      checkCudaErrors( cudaMemcpy(node_old, node_old_d, nnode*sizeof(float),
                              cudaMemcpyDeviceToHost) );

      rms_calc(node_val,node_old,nnode,i);
      TIMER_STOP(rms)
    }

  }
  //____________________________end main loop___________________________
  //    timer
  TIMER_STOP(total)

  TIMER_PRINT(ssol)
  TIMER_PRINT(iter)
  TIMER_PRINT(rms)
  TIMER_PRINT(total)
  TIMER_PRINT(color)
  TIMER_PRINT(color2)

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
