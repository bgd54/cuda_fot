#include <stdio.h>
#include <string>
#include <string.h>
#include "graph_helper.hpp"
#include "rms.hpp"
#define TIMER_MACRO
#include "timer.hpp"
#include "coloring.hpp"

using namespace std;

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
  //                              timer
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
  printf("start edge based on CPU niter: %d, nnode:%d, nedge:%d, colornum: %d\n",niter,
     nnode,nedge, c.colornum);

  //   timer
  TIMER_START(total)
  //______________________________main_loop_____________________________
  for(int i=0;i<=niter;++i){
    //save old
    TIMER_START(ssol)
    for(int j=0;j<nnode;++j){
      node_old[j]=node_val[j];
    }
    TIMER_STOP(ssol)


    //calc next step
    for(int col=0; col<c.colornum;col++){ 
      TIMER_START(iter)
      for(int j=col>0?c.color_offsets[col-1]:0;j<c.color_offsets[col];++j){
        int edgeIdx=c.color_reord[j];
        node_val[enode[2*edgeIdx+1]]+=
          edge_val[edgeIdx]*node_old[enode[edgeIdx*2+0]];
      }
      TIMER_STOP(iter)
    }


    //rms
    if(i%100==0){
      TIMER_START(rms)
      rms_calc(node_val,node_old,nnode,i);
      TIMER_STOP(rms)
    }

  }
  //_____________________________________________________________________
  //    timer
  TIMER_STOP(total)

  TIMER_PRINT(ssol)
  TIMER_PRINT(iter)
  TIMER_PRINT(rms)
  TIMER_PRINT(total)
  TIMER_PRINT(color)
  
  //free
  free(enode);
  free(node_old);
  free(node_val);
  free(edge_val);

  return 0;
}
