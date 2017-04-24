#ifndef COLORING_HPP
#define COLORING_HPP

#include <vector>
#include <cstdlib>
#include <stdio.h>
#include <algorithm>
#include <set>
#include "arg.hpp"
#include <string.h>

using std::set;
using std::vector;

////////////////////////////////////////////////////////////////////////////////
//                              Global Coloring
////////////////////////////////////////////////////////////////////////////////

struct Coloring{
  int* color_reord;
  arg  arg_color_reord;
  int* color_offsets;
  int colornum;

  Coloring();
  Coloring(vector<vector<int> > _cr, vector<int> _coff, int nedge);
  void print();
  Coloring& operator=(const Coloring& c);
  Coloring(const Coloring& c);
  ~Coloring();
};


Coloring global_coloring(int* enode, int nedge, int nnode=0);

////////////////////////////////////////////////////////////////////////////////
//                                Block Coloring
////////////////////////////////////////////////////////////////////////////////
struct Block_coloring{
  int numblock, bs, nedge;
  int * colornum;
  int** color_offsets;
  int* color_reord;
  int* reordcolor;
  arg arg_colornum, arg_color_reord, arg_reordcolor;

  Block_coloring();
  Block_coloring(int _numblock, int _bs, int *_cnum, vector<vector<int> > _coff,
      int * _reord, int _nedge);

  Block_coloring& operator= (const Block_coloring& bc);

  Coloring color_blocks(int* enode, int nedge);


  ~Block_coloring();

};

Block_coloring block_coloring(int* enode, int nedge, int blockSize=128);

void coloring(arg& arg_enode, int nedge, int nnode, Block_coloring& bc, Coloring& c);

#endif
