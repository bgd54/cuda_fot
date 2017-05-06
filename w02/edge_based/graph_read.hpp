#include "graph.hpp"
#include "arg.hpp"
#include "graph_helper.hpp"
#include <iostream>
#include <fstream>
template<int node_dim>
void graph_generate(arg& arg_enode, arg& arg_node_val, arg& arg_node_old,
    arg& arg_edge_val){
///Kimenet: enode, nodeva, nodeold, edgeval
// enode - fajlbol ennek ameretei meghatarozzak edgevalt es nnode-t -> pipa
  std::ifstream fin("hydra.csv");
  Graph g(fin);
  fin.close();

  arg_enode.set_data(g.numEdges(),2,sizeof(unsigned),(char*)g.edge_list);
  g.edge_list = nullptr;
  
  float *node_val, *node_old, *edge_val;
  int nnode = g.numPoints();
  int nedge = g.numEdges();

  node_val = genDataForNodes(nnode,node_dim);
  edge_val = genDataForNodes(nedge,1);
  
  node_old=(float*)malloc(nnode*node_dim*sizeof(float));

 arg_node_val.set_data(nnode, node_dim, sizeof(float),
     (char*) node_val);
 arg_node_old.set_data(nnode, node_dim, sizeof(float),
     (char*) node_old);
 arg_edge_val.set_data(nedge, 1, sizeof(float),
     (char*) edge_val);

}
