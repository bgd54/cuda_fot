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
#ifdef TIMER_MACRO
  timer total("total"), ssol("ssol"), iter("iter"), rms("rms"), color("color"); //TODO Attila h oldja ezt meg
#endif
  
  /////////////////////////////////////////////////////////
  //                        coloring
  /////////////////////////////////////////////////////////
  
  printf("coloring\n");
  TIMER_START(color)
  Coloring c = global_coloring(enode,nedge);
  TIMER_STOP(color)
  printf("start edge based on CPU niter: %d, nnode:%d, nedge:%d, colornum: %d\n",niter,
     nnode,nedge, c.colornum);
  /////////////////////////////////////
  //          Device pointers
  /////////////////////////////////////
  int * color_reord = c.color_reord;
  
#pragma omp target enter data map(to:enode[:nedge*2],color_reord[:nedge],\
    edge_val[:nedge], node_old[:nnode], node_val[:nnode])
  //   timer
  TIMER_START(total)
  //______________________________main_loop_____________________________
  for(int i=0;i<=niter;++i){
    //save old
    TIMER_START(ssol)
    #pragma omp target teams distribute parallel for \
      num_teams((nnode-1)/BLOCKSIZE+1) thread_limit(BLOCKSIZE)\
      map(to:node_old[:nnode], node_val[:nnode])
    for(int j=0;j<nnode;++j){
      node_old[j]=node_val[j];
    }
    TIMER_STOP(ssol)
    
    //calc next step
    for(int col=0; col<c.colornum;col++){
      int color_offset = col==0 ? 0 : c.color_offsets[col-1];
      int color_end = c.color_offsets[col];
      int color_size = color_end - color_offset;
      TIMER_START(iter)
      #pragma omp target teams distribute parallel for\
        num_teams((color_size-1)/BLOCKSIZE+1)  thread_limit(BLOCKSIZE)\
        map(to: node_old[:nnode], node_val[:nnode], enode[:2*nedge], \
            edge_val[:nedge], color_reord[:nedge])
      for(int j=color_offset; j < color_end; ++j){
        int edgeIdx=color_reord[j];
        node_val[enode[2*edgeIdx+1]]+=
          edge_val[edgeIdx]*node_old[enode[edgeIdx*2+0]];
      }
      TIMER_STOP(iter)
    }

    //rms
    if(i%100==0){
      TIMER_START(rms)
      #pragma omp target update from(node_old[:nnode],node_val[:nnode])
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

