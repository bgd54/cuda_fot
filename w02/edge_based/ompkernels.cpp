#include "kernels.hpp"

void ssoln(const int nnode, const int node_dim,
   const arg& arg_node_val, arg& arg_node_old, Kernel& timer){
  float* node_old = (float*) arg_node_old.data;
  float* node_val = (float*) arg_node_val.data;
  timer.timerStart();
  #pragma omp parallel for
  for(int j=0;j<nnode*node_dim;++j){
    node_old[j]=node_val[j];
  }
  timer.timerStop();
}

void iter_calc(const int nedge, const int nnode, const int node_dim,
   const Block_coloring& bc, const Coloring& c, const arg& arg_enode,
   const arg& arg_edge_val, arg& arg_node_val, const arg& arg_node_old,
   Kernel& timer){

  int* enode = (int*) arg_enode.data;
  float * node_val = (float*) arg_node_val.data;
  float * node_old = (float*) arg_node_old.data;
  float * edge_val = (float*) arg_edge_val.data;

  //calc next step
  for(int col=0; col<c.colornum;col++){ 
    timer.timerStart();
    #pragma omp parallel for
    for(int j=col>0?c.color_offsets[col-1]:0;j<c.color_offsets[col];++j){
      int edgeIdx=c.color_reord[j];
      for(int dim=0; dim<node_dim;dim++){
        node_val[enode[2*edgeIdx+1]*node_dim+dim]+=
          edge_val[edgeIdx]*node_old[enode[edgeIdx*2+0]*node_dim+dim];
      }
    }
    timer.timerStop();
  }
}
