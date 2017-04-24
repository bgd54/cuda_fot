#ifndef KERNELS_HPP
#define KERNELS_HPP
#include "arg.hpp"
#include "simulation.hpp"
#include "coloring.hpp"

void ssoln(const int nnode, const int node_dim,
   const arg& arg_node_val, arg& arg_node_old, Kernel& timer);


void iter_calc(const int nedge, const int nnode, const int node_dim,
   const Block_coloring& bc, const Coloring& c, const arg& arg_enode,
   const arg& arg_edge_val, arg& arg_node_val, const arg& arg_node_old,
   Kernel& timer);


#endif /* end of guard KERNELS_HPP*/
