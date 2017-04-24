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
