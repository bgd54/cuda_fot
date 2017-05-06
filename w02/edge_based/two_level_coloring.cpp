#include "coloring.hpp"
#include <vector>
#include <algorithm>

void reorder_data_after_coloring(Block_coloring &bc, Coloring& c,
    arg& arg_edge_val, arg& arg_node_val, arg& arg_enode){
  std::vector<int> new_edges(arg_enode.set_size*arg_enode.set_dim,0);
  std::vector<float> new_eval(arg_edge_val.set_size*arg_edge_val.set_dim,0);
  std::vector<float> new_nval(arg_node_val.set_size*arg_node_val.set_dim,0);
  int nedge = arg_enode.set_size;
  int* enode = (int*)arg_enode.data;
  float* eval = (float*) arg_edge_val.data;
  //reorder edges in blocks
  for(int bIdx=0; bIdx<bc.numblock; ++bIdx){
    for(int thrIdx=0; thrIdx<BLOCKSIZE && bIdx*BLOCKSIZE+thrIdx < nedge; ++thrIdx){
      //reordidx: new index, color_reord[reordidx]: old
      int reordIdx = bIdx*BLOCKSIZE+thrIdx;
      if(reordIdx*2<new_edges.size()){
        int edgeIdx = ((int*)bc.arg_color_reord.data)[reordIdx];
        if(edgeIdx<nedge){
          new_edges[2*reordIdx] = enode[2*edgeIdx+0]; 
          new_edges[2*reordIdx+1] = enode[2*edgeIdx+1]; 
          new_eval[reordIdx] = eval[edgeIdx]; 
        } else {
          printf("WARNING edgeIdx > nedge\n");  
        }
      } else {
        printf("WARNING reordIdx > nedge\n");
      }
    }
  }
  //reorder blocks 
  for(int bIdx=0; bIdx<bc.numblock; ++bIdx){
    int oldbIdx =((int*) c.arg_color_reord.data)[bIdx];
    //printf("bid:%d, oldbid:%d\n", bIdx, oldbIdx);
    for(int thrIdx=0; thrIdx<BLOCKSIZE; ++thrIdx){ 
      int reordIdx = bIdx*BLOCKSIZE+thrIdx;
      int edgeIdx = oldbIdx*BLOCKSIZE+thrIdx;
      if(reordIdx<nedge){
        enode[2*reordIdx+0] = new_edges[2*edgeIdx+0];
        enode[2*reordIdx+1] = new_edges[2*edgeIdx+1];
        eval[reordIdx] = new_eval[edgeIdx];
      }
    }
  }
  std::vector<int> nodemap(arg_node_val.set_size,-1);
  int nodeIdx =0;
  for(int i=0; i<arg_enode.set_dim*arg_enode.set_size;++i){
    if(nodemap[enode[i]] == -1){
      nodemap[enode[i]] = nodeIdx;
      ++nodeIdx;
    }
    enode[i] = nodemap[enode[i]];
  }
  if(nodeIdx != arg_node_val.set_size) 
    printf("WARNING: nodes reordered after coloring: %d/%d\n",nodeIdx,arg_node_val.set_size); 
  
  for(int i=0; i<arg_node_val.set_size; ++i){
    for(int dim=0; dim<arg_node_val.set_dim;++dim){
      new_nval[nodemap[i]*arg_node_val.set_dim+dim] = 
        ((float*)arg_node_val.data)[i*arg_node_val.set_dim+dim]; 
    }
  }
  //update args
  for(size_t i=0;i<new_nval.size();++i){
    ((float*)arg_node_val.data)[i] = new_nval[i];  
  }

  arg_enode.updateD();
  arg_edge_val.updateD();
  arg_node_val.updateD();

}


void coloring(arg& arg_enode, int nedge, int nnode, Block_coloring& bc, Coloring& c, arg& arg_eval, arg& arg_node_val){
  bc = block_coloring((int*) arg_enode.data,nedge);
  c = bc.color_blocks((int*) arg_enode.data,nedge);
  reorder_data_after_coloring(bc,c,arg_eval, arg_node_val,arg_enode);
}
