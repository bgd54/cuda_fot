#include "colouring.hpp"
#include "data_t.hpp"
#include "graph_write_VTK.hpp"
#include "partition.hpp"
#include <iostream>
#include <vector>

using namespace std;

void writeGlobalColouringVTK(const std::string &filename, const Graph &graph,
                             MY_SIZE block_size) {
  std::vector<std::vector<MY_SIZE>> partition =
      std::move(graph.colourEdges<>());
  std::vector<std::vector<std::uint16_t>> data(3);
  data[VTK_IND_THR_COL].resize(graph.numEdges());
  data[VTK_IND_BLK_ID].resize(graph.numEdges());
  // TODO optimise
  MY_SIZE num_colours = partition.size();
  MY_SIZE bid = 0;
  for (MY_SIZE c = 0; c < num_colours; ++c) {
    for (MY_SIZE tid = 0; tid < partition[c].size(); ++tid) {
      data[VTK_IND_THR_COL][partition[c][tid]] = c;
      if (tid % block_size == 0)
        bid++;
      data[VTK_IND_BLK_ID][partition[c][tid]] = bid;
    }
    bid++;
  }
  writeGraphToVTKAscii(filename, graph.edge_to_node, graph.point_coordinates,
                       data);
}

void writeHierarchicalColouringVTK(const std::string &filename,
                                   const Problem<> &problem) {
  const HierarchicalColourMemory<> memory(problem);
  const MY_SIZE block_size = problem.block_size;
  data_t<MY_SIZE, 2> edge_list(problem.graph.numEdges());
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
          m.points_to_be_cached[offset + m.edge_list[i]];
      MY_SIZE right_point =
          m.points_to_be_cached[offset + m.edge_list[m.edge_colours.size() + i]];
      edge_list[2 * edge_ind] = left_point;
      edge_list[2 * edge_ind + 1] = right_point;
      ++edge_ind;
    }
    if (blk_ind % block_size != 0) {
      blk_ind += block_size - (blk_ind % block_size);
    }
  }
  assert(edge_ind == edge_list.getSize());
  writeGraphToVTKAscii(filename, edge_list, problem.graph.point_coordinates,
                       data);
}

void writePartitionVTK(const std::string &filename, const Graph &graph,
                       MY_SIZE block_size) {
  std::vector<std::vector<std::uint16_t>> data(3);
  std::vector<idx_t> partition =
      partitionMetisEnh(graph.getLineGraph(), block_size, 1.016);
  data[VTK_IND_BLK_ID].resize(graph.numEdges());
  assert(partition.size() == graph.numEdges());
  for (MY_SIZE i = 0; i < graph.numEdges(); ++i) {
    data[VTK_IND_BLK_ID][i] = partition[i];
  }
  writeGraphToVTKAscii(filename, graph.edge_to_node, graph.point_coordinates,
                       data);
}

int main() {
  srand(1);
  Problem<> problem(1025, 1025, {0, 128}, true);
  /*writeGlobalColouringVTK("graph_global.vtk", problem.graph, 16);*/
  /*writeHierarchicalColouringVTK("graph_hier.vtk", problem);*/
  writePartitionVTK("graph_part.vtk", problem.graph, 128);

  return 0;
}

// vim:set et sw=2 ts=2 fdm=marker:
