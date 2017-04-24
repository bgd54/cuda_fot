#include "graph_helper.hpp"
#include "arg.hpp"

void graph_generate(const int dx, const int dy, const int node_dim,
    const int edge_dim, const bool bidir, arg& arg_enode, arg& arg_node_val, arg& arg_node_old,
    arg& arg_edge_val){

  int nnode, nedge;
  int* enode = bidir ? 
    generate_bidirected_graph(dx,dy,nedge,nnode) : 
    generate_graph(dx,dy,nedge,nnode);

  float *node_val, *node_old, *edge_val;
  
  node_val = genDataForNodes(nnode,node_dim);
  edge_val = genDataForNodes(nedge,edge_dim);
  
  node_old=(float*)malloc(nnode*node_dim*sizeof(float));

 arg_enode.set_data(nedge, 2, sizeof(int),(char*) enode);
 arg_node_val.set_data(nnode, node_dim, sizeof(float),
     (char*) node_val);
 arg_node_old.set_data(nnode, node_dim, sizeof(float),
     (char*) node_old);
 arg_edge_val.set_data(nedge, edge_dim, sizeof(float),
     (char*) edge_val);
}
