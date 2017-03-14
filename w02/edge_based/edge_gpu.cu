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

#define BLOCKSIZE 1024
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
    const int* enode, const int* color_reord, const int offset, 
    const int color_size, const int nedge){

  int tid = blockDim.x*blockIdx.x+threadIdx.x;
  int reordIdx = tid + offset;
  if(reordIdx<nedge && tid < color_size){
    val[enode[2*color_reord[reordIdx]+1]] +=
      eval[color_reord[reordIdx]]*old[enode[color_reord[reordIdx]*2+0]];
  }
}



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

  int nnode,nedge;
  int* enode = generate_graph(dx,dy,nedge,nnode);

  float* node_val,*node_old, *edge_val;
  
  node_val=genDataForNodes(nnode,1);
  edge_val=genDataForNodes(nedge,1);
  
  node_old=(float*)malloc(nnode*sizeof(float));
  ///////////////////////////////////////////////////////////////////////
  //                            timer
  ///////////////////////////////////////////////////////////////////////
#ifdef TIMER_MACRO
  timer total("total"), ssol("ssol"), iter("iter"), rms("rms"), color("color"); //TODO Attila h oldja ezt meg
#endif

  /////////////////////////////////////////////////////////
  //                        coloring
  /////////////////////////////////////////////////////////
  
  printf("start coloring\n");
  TIMER_START(color)
  Coloring c = global_coloring(enode,nedge);
  TIMER_STOP(color)
  printf("coloring ready, allocate arrays in device memory\n");
  /////////////////////////////////////
  //          Device pointers
  /////////////////////////////////////
  int *enode_d, *color_reord_d;
  float *node_val_d,*node_old_d,*edge_val_d;
  
  checkCudaErrors( cudaMalloc((void**)&enode_d, 2*nedge*sizeof(int)) );
  checkCudaErrors( cudaMalloc((void**)&color_reord_d, nedge*sizeof(int)) );
  checkCudaErrors( cudaMalloc((void**)&edge_val_d, nedge*sizeof(float)) );
  checkCudaErrors( cudaMalloc((void**)&node_old_d, nnode*sizeof(float)) );
  checkCudaErrors( cudaMalloc((void**)&node_val_d, nnode*sizeof(float)) );

  checkCudaErrors( cudaMemcpy(enode_d, enode, 2*nedge*sizeof(int),
                              cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(color_reord_d, c.color_reord,
                               nedge*sizeof(int), cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(edge_val_d, edge_val, nedge*sizeof(float),
                              cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(node_val_d, node_val, nnode*sizeof(float),
                              cudaMemcpyHostToDevice) );

  ///////////////////////////////////////////////////////////
  //                      Start
  ///////////////////////////////////////////////////////////
  printf("start edge based on CPU niter: %d, nnode:%d, nedge:%d, colornum: %d\n",niter,
     nnode,nedge, c.colornum);

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
    for(int col=0; col<c.colornum;col++){ 
      int color_offset = col==0 ? 0 : c.color_offsets[col-1];
      int color_size = c.color_offsets[col] - color_offset;
      TIMER_START(iter)
      iter_calc<<<(color_size-1)/BLOCKSIZE+1,BLOCKSIZE>>>(node_old_d, 
          node_val_d, edge_val_d, enode_d, color_reord_d, color_offset,
          color_size, nedge);
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

  //cuda freee
  checkCudaErrors( cudaFree(enode_d) );
  checkCudaErrors( cudaFree(color_reord_d) );
  checkCudaErrors( cudaFree(edge_val_d) );
  checkCudaErrors( cudaFree(node_old_d) );
  checkCudaErrors( cudaFree(node_val_d) );
  return 0;
}
