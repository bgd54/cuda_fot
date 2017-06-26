#include "colouring.hpp"
#include "data_t.hpp"
#include "graph_write_VTK.hpp"
#include <iostream>
#include <vector>

using namespace std;

void writeGlobalColouringVTK(const std::string &filename,
                             const data_t<float> &point_coords,
                             const Graph &graph, MY_SIZE block_size) {
  std::vector<std::vector<std::uint16_t>> data(3);
  data[VTK_IND_THR_COL] = std::move(graph.colourEdges<true>());
  data[VTK_IND_BLK_ID].resize(graph.numEdges());
  MY_SIZE ind = 0;
  // TODO optimise
  MY_SIZE num_colours = *std::max_element(data[VTK_IND_THR_COL].begin(),
                                          data[VTK_IND_THR_COL].end());
  for (MY_SIZE c = 0; c < num_colours; ++c) {
    for (MY_SIZE i = 0; i < graph.numEdges(); ++i) {
      if (data[VTK_IND_THR_COL][i] == c) {
        data[VTK_IND_BLK_ID][i] = ind / block_size;
        ++ind;
      }
    }
    if (ind % block_size != 0) {
      ind += block_size - (ind % block_size);
    }
  }
  writeGraphToVTKAscii(filename, point_coords, graph.edge_to_node, data);
}

void writeHierarchicalColouringVTK(const std::string &filename,
                                   const data_t<float> &point_coords,
                                   const Problem<> &problem,
                                   MY_SIZE block_size) {
  const HierarchicalColourMemory<> memory(block_size, problem);
  data_t<MY_SIZE> edge_list(problem.graph.numEdges(), 2);
  std::vector<std::vector<std::uint16_t>> data(3);
  MY_SIZE blk_col_ind = 0, blk_ind = 0;
  MY_SIZE edge_ind = 0;
  for (const auto &m : memory.colours) {
    data[VTK_IND_THR_COL].insert(data[VTK_IND_THR_COL].end(),
                                 m.edge_colours.begin(), m.edge_colours.end());
    data[VTK_IND_BLK_COL].insert(data[VTK_IND_BLK_COL].end(),
                                 m.edge_colours.size(), blk_col_ind++);
    assert(2 * m.edge_colours.size() == m.edge_list.size());
    for (MY_SIZE i = 0; i < m.edge_colours.size(); ++i) {
      data[VTK_IND_BLK_ID].push_back(blk_ind / block_size);
      ++blk_ind;

      MY_SIZE local_block_id = i / block_size;
      MY_SIZE offset = m.points_to_be_cached_offsets[local_block_id];
      MY_SIZE left_point =
          m.points_to_be_cached[offset + m.edge_list[2 * i + 0]];
      MY_SIZE right_point =
          m.points_to_be_cached[offset + m.edge_list[2 * i + 1]];
      edge_list[2 * edge_ind] = left_point;
      edge_list[2 * edge_ind + 1] = right_point;
      ++edge_ind;
    }
    if (blk_ind % block_size != 0) {
      blk_ind += block_size - (blk_ind % block_size);
    }
  }
  assert(edge_ind == edge_list.getSize());
  writeGraphToVTKAscii(filename, point_coords, edge_list, data);
}

int main() {
  data_t<float> points(16, 2);
  for (size_t i = 0; i < points.getSize(); ++i) {
    points[points.getDim() * i + 0] = i / 4;
    points[points.getDim() * i + 1] = i % 4;
  }
  Problem<> problem(4, 4);

  writeGlobalColouringVTK("graph_global.vtk", points, problem.graph, 4);
  writeHierarchicalColouringVTK("graph_hier.vtk", points, problem, 4);

  return 0;
}

// vim:set et sw=2 ts=2 fdm=marker:
