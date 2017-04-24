#include "coloring.hpp"

void coloring(arg& arg_enode, int nedge, int nnode, Block_coloring& bc, Coloring& c){
   c = global_coloring((int*)arg_enode.data,nedge,nnode);
}
