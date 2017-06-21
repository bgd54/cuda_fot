#include "data_t.hpp"
#include "graph_write_VTK.hpp"
#include <vector>
#include <iostream>

using namespace std;

int main(){
  data_t<float> points(16, 2);
  for(size_t i=0; i<points.getSize(); ++i){
    points[points.getDim()*i+0] = i/4;
    points[points.getDim()*i+1] = i%4;
  }
  data_t<MY_SIZE> edge_list(24,2);
  for(MY_SIZE i=0; i< 12; ++i){
    edge_list[edge_list.getDim()*i*2+0] = i+(i)/3;
    edge_list[edge_list.getDim()*i*2+1] = i+(i)/3+1;
    edge_list[edge_list.getDim()*i*2+2] = i+(i)/3;
    edge_list[edge_list.getDim()*i*2+3] = i+(i)/3+4;
  }
  for(MY_SIZE i=0; i< 3; ++i){
    edge_list[2*(19+i*2)+0] = 3+i*4;
    edge_list[2*(19+i*2)+1] = 3+i*4+4;
  }
/*  for(MY_SIZE i=0; i< edge_list.getSize(); ++i){
    cout << edge_list[2*i] << " " << edge_list[2*i+1] << endl;
  }*/

  vector<vector<uint8_t>> edge_colors(3,vector<uint8_t>(24,0));

  for(uint8_t i=0; i<edge_colors[0].size();++i){
    edge_colors[0][i] = i;
  }
  for(uint8_t i=0; i<edge_colors[0].size();++i){
    edge_colors[1][i] = i/4;
  }
  
  for(uint8_t i=0; i<edge_colors[0].size();++i){
    edge_colors[2][i] = i%2;
  }

  writeGraphToVTKAscii("graph.vtk",points,edge_list,edge_colors);

  return 0;
}
