#include "reordering.hpp"
#include <cstdlib>
#include <cmath>
#include <stdio.h>
#include <vector>

void reorder( arg & arg_enode, arg &arg_eval, int dx, int dy, arg& arg_node_val){
  int *new_enode = (int*) malloc(
      arg_enode.set_size*arg_enode.set_dim*sizeof(int));
  float *new_eval = (float*) malloc(
      arg_eval.set_size*arg_eval.set_dim*sizeof(float));
  int *old_enode =(int*)arg_enode.data;
  float *old_eval =(float*)arg_eval.data;

  int block_dx = sqrt(BLOCKSIZE/2);
  int block_dy = block_dx/2;

  int block_num_x = dx/block_dx;
  int block_num_y = dy/block_dy;

  int edge_idx = 0;
  for(int j=0; j<block_num_y;++j){
    for(int i=0; i<block_num_x;++i){
      int start_idx = i*block_dx+j*block_dy*dx;
      for(int block_line=0; block_line<block_dy; ++block_line){
        for(int x=0; x<block_dx;++x){
          //vizszintes elek:
          int old_edge_idx = start_idx+x+block_line*dx;
          new_enode[4*edge_idx+0] = old_enode[4*old_edge_idx+0]; 
          new_enode[4*edge_idx+1] = old_enode[4*old_edge_idx+1]; 
          new_enode[4*edge_idx+2] = old_enode[4*old_edge_idx+2]; 
          new_enode[4*edge_idx+3] = old_enode[4*old_edge_idx+3]; 
          new_eval[2*edge_idx +0] = old_eval[2*old_edge_idx+0];
          new_eval[2*edge_idx +1] = old_eval[2*old_edge_idx+1];
          ++edge_idx;
          //fuggoleges elekhez eltolas es ujra eltolas: vizszintes elek szama
          old_edge_idx += dx*dy;
          new_enode[4*edge_idx+0] = old_enode[4*old_edge_idx+0]; 
          new_enode[4*edge_idx+1] = old_enode[4*old_edge_idx+1]; 
          new_enode[4*edge_idx+2] = old_enode[4*old_edge_idx+2]; 
          new_enode[4*edge_idx+3] = old_enode[4*old_edge_idx+3]; 
          new_eval[2*edge_idx +0] = old_eval[2*old_edge_idx+0];
          new_eval[2*edge_idx +1] = old_eval[2*old_edge_idx+1];
          ++edge_idx;
        }
      }
    }
  } 

  //irregular blocks:
  for(int sor=0; sor<dy; sor++){
    for(int x=sor<block_num_y*block_dy?block_num_x*block_dx:0;x<dx;++x){
      int old_edge_idx = sor*dx+x;
      new_enode[4*edge_idx+0] = old_enode[4*old_edge_idx+0]; 
      new_enode[4*edge_idx+1] = old_enode[4*old_edge_idx+1]; 
      new_enode[4*edge_idx+2] = old_enode[4*old_edge_idx+2]; 
      new_enode[4*edge_idx+3] = old_enode[4*old_edge_idx+3]; 
      new_eval[2*edge_idx +0] = old_eval[2*old_edge_idx+0];
      new_eval[2*edge_idx +1] = old_eval[2*old_edge_idx+1];
      ++edge_idx;
      //fuggoleges elekhez eltolas es ujra eltolas: bizszintes elek szama+ sorindex
      old_edge_idx += dx*dy;
      new_enode[4*edge_idx+0] = old_enode[4*old_edge_idx+0]; 
      new_enode[4*edge_idx+1] = old_enode[4*old_edge_idx+1]; 
      new_enode[4*edge_idx+2] = old_enode[4*old_edge_idx+2]; 
      new_enode[4*edge_idx+3] = old_enode[4*old_edge_idx+3]; 
      new_eval[2*edge_idx +0] = old_eval[2*old_edge_idx+0];
      new_eval[2*edge_idx +1] = old_eval[2*old_edge_idx+1];
      ++edge_idx; 
    }
  }
  if(edge_idx*2 != arg_enode.set_size) 
    printf("WARNING: edges reordered: %d/%d\n",edge_idx*2,arg_enode.set_size); 
  int nnode = arg_node_val.set_size;
  //indexelni regivel ertek uj index
  std::vector<int> nodemap(nnode,-1);
  int nodeIdx =0;
  for(int i=0; i<arg_enode.set_dim*arg_enode.set_size;++i){
    if(nodemap[new_enode[i]] == -1){
      nodemap[new_enode[i]] = nodeIdx++;
    }
    new_enode[i] = nodemap[new_enode[i]];
  }
  
  if(nodeIdx != nnode) 
    printf("WARNING: nodes reordered: %d/%d\n",nodeIdx,nnode); 
  float *new_node_val = (float*) malloc(
      arg_node_val.set_size*arg_node_val.set_dim*sizeof(float));
  float* node_val = (float*)arg_node_val.data;
  for(int i=0; i<nnode; ++i){
    for(int dim=0; dim<arg_node_val.set_dim;++dim){
      //printf("i %d dim %d nnode %d set_dim %d nodemap %d\n",i,dim,nnode,arg_node_val.set_dim,nodemap[i]);
      new_node_val[nodemap[i]*arg_node_val.set_dim+dim] = 
        node_val[i*arg_node_val.set_dim+dim]; 
    }
  }
  arg_node_val.set_data(
      nnode, arg_node_val.set_dim, arg_node_val.data_size, (char*)new_node_val
      );

  arg_enode.set_data(
      arg_enode.set_size,arg_enode.set_dim,arg_enode.data_size,(char*)new_enode
      );
  arg_eval.set_data(
      arg_eval.set_size,arg_eval.set_dim,arg_eval.data_size,(char*)new_eval
      );
}
