#include "data_t.hpp"
#include "graph.hpp"
#include "graph_write_VTK.hpp"
#include <vector>
#include <iostream>

using namespace std;

void writeGlobalColouringVTK (const std::string &filename, const data_t<float> &point_coords, const Graph &graph, MY_SIZE block_size) {
  std::vector<std::vector<std::uint16_t>> data (3);
  data[VTK_IND_THR_COL] = std::move(graph.colourEdges<true>());
  data[VTK_IND_BLK_ID].resize(graph.numEdges());
  MY_SIZE ind = 0;
  MY_SIZE num_colours = *std::max_element(data[VTK_IND_THR_COL].begin(),
      data[VTK_IND_THR_COL].end());
  for (MY_SIZE c = 0; c < num_colours; ++c) {
    for (MY_SIZE i = 0; i < graph.numEdges(); ++i) {
      if (data[VTK_IND_THR_COL][i] == c) {
        data[VTK_IND_BLK_ID][i] = ind / block_size;
        ++ind;
      }
    }
  }
  writeGraphToVTKAscii(filename, point_coords, graph.edge_to_node, data);
}

int main(){
  data_t<float> points(16, 2);
  for(size_t i=0; i<points.getSize(); ++i){
    points[points.getDim()*i+0] = i/4;
    points[points.getDim()*i+1] = i%4;
  }
  Graph graph (4,4);

  writeGlobalColouringVTK("graph.vtk",points, graph, 4);

  return 0;
}
