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
  sim.timers.push_back(timer("cache_calc"));
  #endif
}

////////////////////////////////////////////////////////////////////////////////
// CPU routines
////////////////////////////////////////////////////////////////////////////////

void cache_map_gen(int* enode, int nedge, int* iwillwritethis, int* icachethis,
    Block_coloring& bc){

  for(int i=0;i<nedge;++i) icachethis[i]=-1;

  for(int bIdx=0; bIdx<bc.numblock;++bIdx){
    int start= bIdx*bc.bs;
    int end= std::min((bIdx+1)*bc.bs,nedge);
    std::set<int> needtoCacheforWrite;
    for(int tid=0; tid + start < end; ++tid){
      //kigyujtom a nodeidkat amiket irni fogok szalankent es osszessegeben
      iwillwritethis[start+tid]=enode[2*bc.color_reord[start+tid]+1];
      needtoCacheforWrite.insert(iwillwritethis[start+tid]);
    }

    std::copy(needtoCacheforWrite.begin(),
        needtoCacheforWrite.end(), icachethis+start);
    for(int tid=0;tid+start<end;++tid){
      iwillwritethis[start+tid] = 
        std::find(icachethis+start, icachethis+end, iwillwritethis[start+tid])
        - (icachethis+start);
    }

  }
  /*
  for(int bIdx=0; bIdx<bc.numblock;++bIdx){
    int start= bIdx*bc.bs;
    int end= std::min((bIdx+1)*bc.bs,nedge);
    for(int tid=0; tid + start < end; ++tid){
      printf("bIdx: %3d tid: %3d i: %3d icache: %6d, iwrite: %3d, which: %6d\n", bIdx, tid, start+tid,
         icachethis[start+tid], iwillwritethis[start+tid],
         enode[2*bc.color_reord[start+tid]+1]);
    }
  }
  */

}
void cache_map_read(int* enode, int nedge, int* ireadold, int* icachethis_old,
    int* eval_ind, int* eval_cache,Block_coloring& bc){

  for(int i=0;i<nedge;++i){ 
    icachethis_old[i]=-1;
    eval_cache[i]=-1;
  }


  for(int bIdx=0; bIdx<bc.numblock;++bIdx){
    int start= bIdx*bc.bs;
    int end= std::min((bIdx+1)*bc.bs,nedge);
    std::set<int> needtoCacheforOld;
    std::set<int> needtoCacheforEval;
    for(int tid=0; tid + start < end; ++tid){
      //kigyujtom a nodeidkat amiket irni fogok szalankent es osszessegeben
      ireadold[start+tid] = enode[2*bc.color_reord[start+tid]+0];
      eval_ind[start+tid] = bc.color_reord[start+tid];
      needtoCacheforOld.insert(ireadold[start+tid]);
      needtoCacheforEval.insert(eval_ind[start+tid]);
    }

    std::copy(needtoCacheforOld.begin(),
        needtoCacheforOld.end(), icachethis_old+start);
    std::copy(needtoCacheforEval.begin(),
        needtoCacheforEval.end(), eval_cache+start);
    for(int tid=0;tid+start<end;++tid){
      ireadold[start+tid] = 
        std::find(icachethis_old+start, icachethis_old+end, ireadold[start+tid])
        - (icachethis_old+start);
      eval_ind[start+tid] = 
        std::find(eval_cache+start, eval_cache+end, eval_ind[start+tid])
        - (eval_cache+start);
    }

  }
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
    int color_start, int* cache_map, int* global_to_cache, int * cache_old_map,
    int* global_cache_old_map, int* cache_eval_map, int* global_cache_eval_map, const int node_dim){

  int tid = threadIdx.x;
  extern  __shared__  float tempval[];


  int bIdx = blocksInColor[blockIdx.x+color_start];
  int reordIdx = tid + bIdx*blockDim.x;

  int iwritethisIdx = -1;
  int iloadThis = -1;
  int ireadOld = -1;
  int ireadeval = -1;
  int icacheOld = -1;
  int icahceEval = -1;

  if(reordIdx<nedge){
    iwritethisIdx = cache_map[reordIdx];
    iloadThis = global_to_cache[reordIdx];

    ireadOld = cache_old_map[reordIdx];
    icacheOld = global_cache_old_map[reordIdx];
    
    ireadeval = cache_eval_map[reordIdx];
    icahceEval = global_cache_eval_map[reordIdx];
    
    if(iloadThis != -1){
      for(int dim=0; dim<node_dim;dim++){ 
        tempval[tid*node_dim+dim] = val[iloadThis*node_dim+dim];
      }
    }
    if(icacheOld != -1){
      for(int dim=0; dim<node_dim;dim++){ 
        tempval[tid*node_dim+blockDim.x*node_dim+dim] =
          old[icacheOld*node_dim+dim];
      }
    }
    if(icahceEval != -1) tempval[tid+2*blockDim.x*node_dim] = eval[icahceEval];
  }
  __syncthreads();


  float* increment = new float[node_dim];
  if(reordIdx < nedge){

    for(int dim=0; dim<node_dim;dim++){ 
      increment[dim] = tempval[ireadeval+2*blockDim.x*node_dim] *
        tempval[(ireadOld+blockDim.x)*node_dim+dim];
    }

      //eval[color_reord[reordIdx]]*old[enode[color_reord[reordIdx]*2+0]];
  }
  for(int col=0; col<colornum[bIdx];++col){
    if(reordIdx < nedge && col == color[reordIdx]){
      for(int dim=0; dim<node_dim;dim++){ 
        tempval[iwritethisIdx*node_dim+dim]+= increment[dim];
      }

      //val[enode[2*color_reord[reordIdx]+1] ] += increment;
    }
    __syncthreads();
  }
  //cachelt ertekek visszairasa

  if(reordIdx<nedge && iloadThis != -1){
    for(int dim=0; dim<node_dim;dim++){ 
      val[iloadThis*node_dim+dim] = tempval[tid*node_dim+dim];
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
  int* iwillwritethis, *icachethis;
  int* ireadold, *icachethis_old;
  int* ireadeval, *icachethis_eval;
  iwillwritethis      = (int*) malloc(nedge*sizeof(int));
  icachethis          = (int*) malloc(nedge*sizeof(int));
  ireadold            = (int*) malloc(nedge*sizeof(int));
  icachethis_old      = (int*) malloc(nedge*sizeof(int));
  ireadeval           = (int*) malloc(nedge*sizeof(int));
  icachethis_eval     = (int*) malloc(nedge*sizeof(int));
  
  cache_map_gen(enode, nedge, iwillwritethis, icachethis, c); 
  cache_map_read(enode, nedge, ireadold, icachethis_old,
     ireadeval, icachethis_eval, c); 
  TIMER_STOP(sim.timers[2])

  /////////////////////////////////////
  //          Device pointers
  /////////////////////////////////////
  printf("coloring ready, allocate arrays in device memory\n");
  int *enode_d, *color_reord_d, *colornum_d, *color_d;
  float *node_val_d,*node_old_d,*edge_val_d;
  int *block_reord_d;
  int *iwillwritethis_d, *icachethis_d;

  checkCudaErrors( cudaMalloc((void**)&enode_d, 2*nedge*sizeof(int)) );
  checkCudaErrors( cudaMalloc((void**)&color_reord_d, nedge*sizeof(int)) );
  checkCudaErrors( cudaMalloc((void**)&color_d, nedge*sizeof(int)) );
  checkCudaErrors( cudaMalloc((void**)&colornum_d, c.numblock*sizeof(int)) );
  checkCudaErrors( cudaMalloc((void**)&edge_val_d, nedge*sizeof(float)) );
  checkCudaErrors( cudaMalloc((void**)&node_old_d, nnode*node_dim*sizeof(float)) );
  checkCudaErrors( cudaMalloc((void**)&node_val_d, nnode*node_dim*sizeof(float)) );
  checkCudaErrors( cudaMalloc((void**)&block_reord_d, c.numblock*sizeof(int)) );
  checkCudaErrors( cudaMalloc((void**)&iwillwritethis_d, nedge*sizeof(int)) );
  checkCudaErrors( cudaMalloc((void**)&icachethis_d, nedge*sizeof(int)) );
  
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
  checkCudaErrors( cudaMemcpy(iwillwritethis_d, iwillwritethis,
                               nedge*sizeof(int), cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(icachethis_d, icachethis,
                               nedge*sizeof(int), cudaMemcpyHostToDevice) );

  //////////////////// device pointers for read caches //////////////////////

  int *ireadold_d, *icachethis_old_d;
  int *ireadeval_d, *icachethis_eval_d;
  checkCudaErrors( cudaMalloc((void**)&ireadold_d, nedge*sizeof(int)) );
  checkCudaErrors( cudaMalloc((void**)&icachethis_old_d, nedge*sizeof(int)) );
  checkCudaErrors( cudaMalloc((void**)&ireadeval_d, nedge*sizeof(int)) );
  checkCudaErrors( cudaMalloc((void**)&icachethis_eval_d, nedge*sizeof(int)) );
  checkCudaErrors( cudaMemcpy(ireadold_d, ireadold,
                               nedge*sizeof(int), cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(icachethis_old_d, icachethis_old,
                               nedge*sizeof(int), cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(ireadeval_d, ireadeval,
                               nedge*sizeof(int), cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(icachethis_eval_d, icachethis_eval,
                               nedge*sizeof(int), cudaMemcpyHostToDevice) );

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
      iter_calc<<<len,BLOCKSIZE,3*BLOCKSIZE*node_dim*sizeof(float)>>>(node_old_d, //2*nodedim+1*BS?
          node_val_d, edge_val_d, enode_d, color_reord_d, nedge, color_d,
          colornum_d, block_reord_d, start, iwillwritethis_d, icachethis_d,
          ireadold_d, icachethis_old_d, ireadeval_d, icachethis_eval_d, node_dim);
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
  free(iwillwritethis);
  free(icachethis);
  free(ireadold);
  free(icachethis_old);
  free(ireadeval);
  free(icachethis_eval);
  //cuda freee
  checkCudaErrors( cudaFree(enode_d) );
  checkCudaErrors( cudaFree(color_reord_d) );
  checkCudaErrors( cudaFree(edge_val_d) );
  checkCudaErrors( cudaFree(node_old_d) );
  checkCudaErrors( cudaFree(node_val_d) );
  checkCudaErrors( cudaFree(color_d) );
  checkCudaErrors( cudaFree(colornum_d) );
  checkCudaErrors( cudaFree(block_reord_d) );
  checkCudaErrors( cudaFree(iwillwritethis_d) );
  checkCudaErrors( cudaFree(icachethis_d) );
  checkCudaErrors( cudaFree(icachethis_old_d) );
  checkCudaErrors( cudaFree(ireadold_d) );
  checkCudaErrors( cudaFree(ireadeval_d) );
  checkCudaErrors( cudaFree(icachethis_eval_d) );
  
  
  
  return 0;
}
