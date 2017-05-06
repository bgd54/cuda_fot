#include <stdio.h>
#include <string.h>
#include "arg.hpp"
#ifdef USE_FILE
#include "graph_read.hpp"
#else
#include "graph_gen.hpp"
#endif
#include "simulation.hpp"
#include "coloring.hpp"
#include "cache_calc.hpp"
#include "reordering.hpp"
#include "rms.hpp"
#include "kernels.hpp"

using namespace std;

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

void SOA(arg &arg_data){
  float* data =(float*) arg_data.data; 
#ifdef USE_SOA
  printf("AOS/SOA conversion.\n");
  float* tmp = 
    (float*) malloc(arg_data.set_size*arg_data.set_dim*sizeof(float));
  
  for(int i=0; i<arg_data.set_size; ++i){
    for(int j=0; j<arg_data.set_dim; ++j){
      tmp[arg_data.set_size*j+i] = data[arg_data.set_dim*i+j]; 
    } 
  }

  arg_data.set_data(arg_data.set_size,
      arg_data.set_dim,arg_data.data_size,(char*)tmp);
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
  arg arg_enode, arg_node_val, arg_node_old, arg_edge_val;
#ifdef USE_FILE
  graph_generate<NODE_DIM>(arg_enode, arg_node_val, arg_node_old, arg_edge_val);
#else
  graph_generate(dx, dy, NODE_DIM, 1, bidir,
      arg_enode, arg_node_val, arg_node_old, arg_edge_val);
#endif
  int nnode = arg_node_val.set_size, nedge = arg_enode.set_size;
  
  ///////////////////////////////////////////////////////////////////////
  //                            timer
  ///////////////////////////////////////////////////////////////////////
  Simulation sim(nedge, nnode, NODE_DIM);
  addTimers(sim);

  /////////////////////////////////////////////////////////
  //                        reordering
  /////////////////////////////////////////////////////////
  printf("start reordering\n");
  TIMER_START(sim.timers[0])
  reorder( arg_enode, arg_edge_val, dx, dy, arg_node_val);
  TIMER_STOP(sim.timers[0])
  
  /////////////////////////////////////////////////////////
  //                        coloring
  /////////////////////////////////////////////////////////
  printf("start coloring\n");
  Block_coloring bc;
  Coloring c;
  TIMER_START(sim.timers[1])
  coloring(arg_enode, nedge, nnode, bc, c, arg_edge_val, arg_node_val);
  TIMER_STOP(sim.timers[1])

  ///////////////////////////////////////////////////////////////////////
  //                            AOS/SOA
  ///////////////////////////////////////////////////////////////////////
  SOA(arg_node_val); 
  SOA(arg_node_old); 

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
    ssoln(nnode, NODE_DIM, arg_node_val, arg_node_old, sim.kernels[0]);

    //calc next step
    iter_calc(nedge, nnode, NODE_DIM, bc, c, arg_enode, arg_edge_val, arg_node_val,
        arg_node_old, cm, sim.kernels[1]);

    // rms
    if(i%100==0){
      sim.kernels[2].timerStart();
      arg_node_old.update();
      arg_node_val.update();
      rms_calc((float*) arg_node_val.data, (float*) arg_node_old.data,
          nnode, i, NODE_DIM);
      sim.kernels[2].timerStop();
    }
  }
  //____________________________end main loop___________________________
  //    timer
  sim.stop();

  sim.printTiming();
 
  return 0;
}
