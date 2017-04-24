#include "kernels.hpp"

void ssoln(const int nnode, const int node_dim,
   const arg& arg_node_val, arg& arg_node_old, Kernel& timer){
  float* node_old = (float*) arg_node_old.data;
  float* node_val = (float*) arg_node_val.data;
  timer.timerStart();
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

  timer.timerStart();
  for(int edgeIdx=0;edgeIdx<nedge;++edgeIdx){
    for(int dim=0; dim<node_dim;dim++){
      node_val[enode[2*edgeIdx+1]*node_dim+dim]+=
        edge_val[edgeIdx]*node_old[enode[edgeIdx*2+0]*node_dim+dim];
    }
  }
  timer.timerStop();

}
