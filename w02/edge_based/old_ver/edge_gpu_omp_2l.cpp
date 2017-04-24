#include <stdio.h>
#include <string>
#include <string.h>
#include "graph_helper.hpp"
#include "rms.hpp"
#define TIMER_MACRO
#include "simulation.hpp"
#include "coloring.hpp"
#include <omp.h>

using namespace std;

#define BLOCKSIZE 128
#define MAX_NODE_DIM 10

void addTimers(Simulation &sim){
  #ifdef TIMER_MACRO
  sim.timers.push_back(timer("color"));
  sim.timers.push_back(timer("colorb"));
  #endif
}

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
  
  printf("coloring\n");
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
  int *colornum_d = c.colornum, *color_d = c.reordcolor;
  int *block_reord_d = bc.color_reord;
  int * color_reord = c.color_reord;
  
#pragma omp target enter data map(to:enode[:nedge*2], edge_val[:nedge],\
    node_old[:nnode*node_dim], node_val[:nnode*node_dim])
#pragma omp target enter data map(to: color_reord[:nedge], colornum_d[:c.numblock],\
    block_reord_d[:c.numblock], color_d[:nedge])
  //   timer
  sim.start();
  //______________________________main_loop_____________________________
  for(int i=0;i<=niter;++i){
    //save old
    sim.kernels[0].timerStart();
    #pragma omp target teams distribute parallel for \
      num_teams((nnode-1)/BLOCKSIZE+1) thread_limit(BLOCKSIZE)\
      map(to:node_old[:nnode*node_dim], node_val[:nnode*node_dim])
    for(int j=0;j<nnode*node_dim;++j){
      node_old[j]=node_val[j];
    }
    sim.kernels[0].timerStop();
    
    //calc next step
    for(int col=0; col<bc.colornum;col++){
      int start = col==0 ? 0 : bc.color_offsets[col-1];
      int len = bc.color_offsets[col] -start;
      sim.kernels[1].timerStart();
      #pragma omp target teams distribute parallel for\
        num_teams(len)  thread_limit(BLOCKSIZE)\
        map(to: node_old[:nnode], node_val[:nnode], enode[:2*nedge], \
            edge_val[:nedge], color_reord[:nedge], color_d[:nedge], \
            colornum_d[:c.numblock], block_reord_d[:c.numblock]) collapse(2)
      for(int j=0; j < len; ++j){//j: tid in color
        for(int tid=0; tid<BLOCKSIZE;++tid){
          int bIdx =block_reord_d[j];
          int reordIdx = tid+bIdx*BLOCKSIZE;
          float increment[MAX_NODE_DIM];
          if(reordIdx < nedge){
            for(int dim=0; dim<node_dim; dim++)
            increment[dim] = edge_val[color_reord[reordIdx]*node_dim+dim] *
              node_old[enode[color_reord[reordIdx]*2+0]*node_dim+dim];
          }
 
          for(int col=0; col<colornum_d[bIdx];++col){
            if(reordIdx < nedge && col == color_d[reordIdx]){
              for(int dim=0; dim<node_dim; dim++)
                node_val[enode[2*color_reord[reordIdx]+1]*node_dim+dim ] += increment[dim];
            }
            //BARRIER
          }
        }

      }
      sim.kernels[1].timerStop();
    }

    //rms
    if(i%100==0){
      sim.kernels[2].timerStart();
      #pragma omp target update from(node_old[:nnode],node_val[:nnode])
      rms_calc(node_val,node_old,nnode,i,node_dim);
      sim.kernels[2].timerStop();
    }
   
  }
  //____________________________end main loop___________________________
  //    timer
  sim.stop();

  sim.printTiming();
 
  //free target memory
  #pragma omp target exit data map(delete:enode[:2*nedge], node_old[:nnode],\
      node_val[:nnode], edge_val[:nedge], color_reord[:nedge])
   
  //free
  free(enode);
  free(node_old);
  free(node_val);
  free(edge_val);

  return 0;
}

