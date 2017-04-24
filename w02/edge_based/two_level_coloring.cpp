#include "coloring.hpp"

void coloring(arg& arg_enode, int nedge, int nnode, Block_coloring& bc, Coloring& c){
  bc = block_coloring((int*) arg_enode.data,nedge);
  c = bc.color_blocks((int*) arg_enode.data,nedge);
}
