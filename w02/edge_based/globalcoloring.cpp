#include "coloring.hpp"

void coloring(arg& arg_enode, int nedge, int nnode, Block_coloring& bc, Coloring& c, arg& arg_eval, arg& arg_node_val){
   c = global_coloring((int*)arg_enode.data,nedge,nnode);
}
