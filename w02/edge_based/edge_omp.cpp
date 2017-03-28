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

void addTimers(Simulation &sim){
  #ifdef TIMER_MACRO
  sim.timers.push_back(timer("color"));
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
  Coloring c = global_coloring(enode,nedge);
  TIMER_STOP(sim.timers[0])
  printf("start edge based on CPU niter: %d, nnode:%d, nedge:%d, colornum: %d\n",niter,
     nnode,nedge, c.colornum);

  //   timer
  sim.start();
  //______________________________main_loop_____________________________
  for(int i=0;i<=niter;++i){
    //save old
    sim.kernels[0].timerStart();
    for(int j=0;j<nnode;++j){
      node_old[j]=node_val[j];
    }
    sim.kernels[0].timerStop();

    //calc next step
    for(int col=0; col<c.colornum;col++){ 
      sim.kernels[1].timerStart();
      #pragma omp parallel for
      for(int j=col>0?c.color_offsets[col-1]:0;j<c.color_offsets[col];++j){
        int edgeIdx=c.color_reord[j];
        node_val[enode[2*edgeIdx+1]]+=
          edge_val[edgeIdx]*node_old[enode[edgeIdx*2+0]];
      }
      sim.kernels[1].timerStop();
    }

    //rms
    if(i%100==0){
      sim.kernels[2].timerStart();
      rms_calc(node_val,node_old,nnode,i);
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

  return 0;
}
