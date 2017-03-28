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
  ///////////////////////////////////////////////////////////////////////
  //                            params
  ///////////////////////////////////////////////////////////////////////
  for(int i=1; i < argc; ++i){
    if (!strcmp(argv[i],"-niter")) niter=atoi(argv[++i]);
    else if (!strcmp(argv[i],"-dx")) dx=atoi(argv[++i]);
    else if (!strcmp(argv[i],"-dy")) dy=atoi(argv[++i]);
    else if (!strcmp(argv[i],"-bidir")) bidir=true;
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
  int* enode = bidir ? 
    generate_bidirected_graph(dx,dy,nedge,nnode) : 
    generate_graph(dx,dy,nedge,nnode);

  float* node_val,*node_old, *edge_val;
  
  node_val=genDataForNodes(nnode,1);
  edge_val=genDataForNodes(nedge,1);
  
  node_old=(float*)malloc(nnode*sizeof(float));
  ///////////////////////////////////////////////////////////////////////
  //                            timer
  ///////////////////////////////////////////////////////////////////////
  Simulation sim = initSimulation(nedge, nnode);
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
    node_old[:nnode], node_val[:nnode])
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
      map(to:node_old[:nnode], node_val[:nnode])
    for(int j=0;j<nnode;++j){
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
            colornum_d[:c.numblock], block_reord_d[:c.numblock])
      for(int j=0; j < len*BLOCKSIZE; ++j){//j: tid in color
        //problema: elm mukdik de a teamnum meg minden implementacio fuggo
        int tid  = omp_get_thread_num();
        int bIdx =block_reord_d[omp_get_team_num()];
        int reordIdx = tid+bIdx*omp_get_num_teams();
        float increment = 0.0f;
        if(reordIdx < nedge){
          increment = 
            edge_val[color_reord[reordIdx]]*node_old[enode[color_reord[reordIdx]*2+0]];
        }

        for(int col=0; col<colornum_d[bIdx];++col){
          if(reordIdx < nedge && col == color_d[reordIdx]){
            node_val[enode[2*color_reord[reordIdx]+1] ] += increment;
          }
          //eeees region cannot be closely nested in distribute parallel for
          #pragma omp barrier
        }

      }
      sim.kernels[1].timerStop();
    }

    //rms
    if(i%100==0){
      sim.kernels[2].timerStart();
      #pragma omp target update from(node_old[:nnode],node_val[:nnode])
      rms_calc(node_val,node_old,nnode,i);
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

