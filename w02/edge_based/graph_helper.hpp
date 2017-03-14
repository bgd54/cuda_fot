#ifndef GRAPH_HELPER_HPP
#define GRAPH_HELPER_HPP

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <algorithm>
//Negyzetracs generalasa
//  csucsok indexelese sorfolytonosan
//  elek indexelese: vizszintesek sorfolytonosan majd a fuggolegesek sorf.
//dim_x darab el szerepel soronkent es dim_y sor van
//elkozpontu reprezentaciot ad vissza.
//  - - - - dim_x=4
// | | | | | -> fuggoleges elek soronkent: dim_x+1=5
//  - - - -
// | | | | | 
//  - - - -
// | | | | | 
//  - - - -
// | | | | | 
//  - - - - dim_y = 5
//           -> fuggoleges elekbol allo sorok szama: dim_y-1=4
int* generate_graph(const int& dim_x, const int& dim_y, int& nedge, int& nnode){
  nedge = dim_x*dim_y + (dim_x+1)*(dim_y-1); 
  nnode=(dim_x+1)*dim_y;
  int* enode =(int*) malloc(nedge*2*sizeof(int));

  for(int y=0;y<dim_y;y++){
    for(int x=0; x<dim_x;x++){
      int edgeId = y*dim_x+x;
      enode[2*edgeId+0]= edgeId+y;
      enode[2*edgeId+1]= edgeId+y+1;
    }
  }
  for(int y=0;y<dim_y-1;y++){
    for(int x=0; x<dim_x+1;x++){
      int edgeId = y*(dim_x+1)+x+dim_x*dim_y;
      enode[2*edgeId+0]= (y+1)*dim_x+x+y+1;
      enode[2*edgeId+1]= y*dim_x+x+y;
    }
  }

  return enode;
}

int* generate_bidirected_graph(const int& dim_x, const int& dim_y, int& nedge, int& nnode){
  nedge =2*(dim_x*dim_y + (dim_x+1)*(dim_y-1));
  nnode=(dim_x+1)*dim_y;
  int* enode =(int*) malloc(nedge*2*sizeof(int));

  for(int y=0;y<dim_y;y++){
    for(int x=0; x<dim_x;x++){
      int edgeId = y*dim_x+x;
      enode[4*edgeId+0]= edgeId+y;
      enode[4*edgeId+1]= edgeId+y+1;
      enode[4*edgeId+2]= edgeId+y+1;
      enode[4*edgeId+3]= edgeId+y;
    }
  }
  for(int y=0;y<dim_y-1;y++){
    for(int x=0; x<dim_x+1;x++){
      int edgeId = y*(dim_x+1)+x+dim_x*dim_y;
      enode[4*edgeId+0]= (y+1)*dim_x+x+y+1;
      enode[4*edgeId+1]= y*dim_x+x+y;
      enode[4*edgeId+2]= y*dim_x+x+y;
      enode[4*edgeId+3]= (y+1)*dim_x+x+y+1;
    }
  }

  return enode;
}

float* genDataForNodes(const int& nnode,const int& dim,
    const float& mean=0.001){
  float* nodedata= (float*) malloc(nnode*dim*sizeof(float));

  for(int i = 0; i<nnode*dim;++i){
    nodedata[i]=(i)/float(nnode)*mean;
  }

  return nodedata;

}

#endif
