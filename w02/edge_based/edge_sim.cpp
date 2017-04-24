#include <stdio.h>
#include <string.h>
#include "arg.hpp"
#include "graph_gen.hpp"
#include "simulation.hpp"
#include "coloring.hpp"
#include "cache_calc.hpp"
#include "reordering.hpp"
#include "rms.hpp"
#include "kernels.hpp"

using namespace std;

#define BLOCKSIZE 128
#define MAX_NODE_DIM 10

////////////////////////////////////////////////////////////////////////////////
// CPU routines
////////////////////////////////////////////////////////////////////////////////

void addTimers(Simulation &sim){
  #ifdef TIMER_MACRO
  sim.timers.push_back(timer("reorder"));
  sim.timers.push_back(timer("color"));
  sim.timers.push_back(timer("calc_cache"));
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
  arg arg_enode, arg_node_val, arg_node_old, arg_edge_val;
  graph_generate(dx, dy, node_dim, edge_dim, bidir,
      arg_enode, arg_node_val, arg_node_old, arg_edge_val);
  int nnode = arg_node_val.set_size, nedge = arg_enode.set_size;

  ///////////////////////////////////////////////////////////////////////
  //                            timer
  ///////////////////////////////////////////////////////////////////////
  Simulation sim(nedge, nnode, node_dim);
  addTimers(sim);

  /////////////////////////////////////////////////////////
  //                        reordering
  /////////////////////////////////////////////////////////
  printf("start reordering\n");
  TIMER_START(sim.timers[0])
  //TODO reordering abstrction
  reorder();
  TIMER_STOP(sim.timers[0])

  /////////////////////////////////////////////////////////
  //                        coloring
  /////////////////////////////////////////////////////////
  printf("start coloring\n");
  Block_coloring bc;
  Coloring c;
  TIMER_START(sim.timers[1])
  coloring(arg_enode, nedge, nnode, bc, c);
  TIMER_STOP(sim.timers[1])

  /////////////////////////////////////////////////////////
  //                        cache_calc
  /////////////////////////////////////////////////////////
  printf("Calculateing cache\n");
  TIMER_START(sim.timers[2])
  cacheMap cm = genCacheMap((int*) arg_enode.data, nedge, bc);
  TIMER_STOP(sim.timers[2])
 
  printf("start simulation niter: %d, nnode: %d, nedge: %d\n",
      niter, nnode, nedge);
  //timer
  sim.start();
  //______________________________main_loop_____________________________
  for(int i=0;i<=niter;++i){
    //save old
    ssoln(nnode, node_dim, arg_node_val, arg_node_old, sim.kernels[0]);

    //calc next step
    iter_calc(nedge, nnode, node_dim, bc, c, arg_enode, arg_edge_val, arg_node_val, arg_node_old, sim.kernels[1]);

    // rms
    if(i%100==0){
      sim.kernels[2].timerStart();
      arg_node_old.update();
      arg_node_val.update();
      rms_calc((float*) arg_node_val.data, (float*) arg_node_old.data,
          nnode, i, node_dim);
      sim.kernels[2].timerStop();
    }
  }
  //____________________________end main loop___________________________
  //    timer
  sim.stop();

  sim.printTiming();
 
  return 0;
}
