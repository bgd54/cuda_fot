#include "colouring.hpp"
#include "data_t.hpp"
#include "graph_write_VTK.hpp"
#include "partition.hpp"
#include "structured_problem.hpp"
#include <iostream>
#include <vector>

using namespace std;

void writeGlobalColouringVTK(const std::string &filename,
                             const VisualisableMesh &mesh, MY_SIZE block_size) {
  std::vector<std::vector<MY_SIZE>> partition = std::move(mesh.colourCells());
  std::vector<std::vector<std::uint16_t>> data(3);
  data[VTK_IND_THR_COL].resize(mesh.numCells());
  data[VTK_IND_BLK_ID].resize(mesh.numCells());
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
  writeMeshToVTKAscii(filename, mesh.cell_to_node[0], mesh.point_coordinates,
                      data);
}

void writeHierarchicalColouringVTK(const std::string &filename,
                                   const Problem<false> &problem,
                                   const data_t &point_coordinates) {
  const HierarchicalColourMemory<false> memory(
      problem, problem.partition_vector);
  const MY_SIZE MESH_DIM = problem.mesh.cell_to_node[0].getDim();
  std::cout << "Number of block colours: " << memory.colours.size()
            << std::endl;
  data_t cell_list(data_t::create<MY_SIZE>(problem.mesh.numCells(),
                                           MESH_DIM));
  std::vector<std::vector<std::uint16_t>> data(3);
  MY_SIZE blk_col_ind = 0, blk_ind = 0;
  MY_SIZE cell_ind = 0;
  for (const auto &m : memory.colours) {
    data[VTK_IND_THR_COL].insert(data[VTK_IND_THR_COL].end(),
                                 m.cell_colours.begin(),
                                 m.cell_colours.end());
    data[VTK_IND_BLK_COL].insert(data[VTK_IND_BLK_COL].end(),
                                 m.cell_colours.size(), blk_col_ind++);
    assert(MESH_DIM * m.cell_colours.size() == m.cell_list[0].size());
    MY_SIZE local_block_id = 0;
    MY_SIZE num_cells = m.cell_colours.size();
    for (MY_SIZE i = 0; i < num_cells; ++i) {
      assert(local_block_id + 1 < m.block_offsets.size());
      assert(m.block_offsets[local_block_id] !=
             m.block_offsets[local_block_id + 1]);

      if (m.block_offsets[local_block_id + 1] <= i) {
        ++local_block_id;
        ++blk_ind;
      }
      data[VTK_IND_BLK_ID].push_back(blk_ind);

      MY_SIZE offset = m.points_to_be_cached_offsets[local_block_id];
      for (MY_SIZE j = 0; j < MESH_DIM; ++j) {
        MY_SIZE m_cell_ind = index<true>(num_cells, i, MESH_DIM, j);
        MY_SIZE point_ind =
            m.points_to_be_cached[offset + m.cell_list[0][m_cell_ind]];
        cell_list.operator[]<MY_SIZE>(MESH_DIM *cell_ind + j) = point_ind;
      }
      ++cell_ind;
    }
    ++blk_ind;
  }
  assert(cell_ind == cell_list.getSize());
  writeMeshToVTKAscii(filename, cell_list, point_coordinates, data);
}

void writePartitionVTK(const std::string &filename,
                       const VisualisableMesh &mesh, MY_SIZE block_size) {
  std::vector<std::vector<std::uint16_t>> data(3);
  std::vector<idx_t> partition =
      partitionMetisEnh(mesh.getCellToCellGraph(), block_size, 1.001);
  data[VTK_IND_BLK_ID].resize(mesh.numCells());
  assert(partition.size() == mesh.numCells());
  for (MY_SIZE i = 0; i < mesh.numCells(); ++i) {
    data[VTK_IND_BLK_ID][i] = partition[i];
  }
  writeMeshToVTKAscii(filename, mesh.cell_to_node[0], mesh.point_coordinates,
                      data);
}

int main() {
  srand(1);
  // In the future, the visualisable mesh should be extractable from the
  // StructuredProblem
  std::vector<MY_SIZE> grid_dims = {129,129,129};
  std::vector<MY_SIZE> block_dims = {8,4,4};
  Problem<false> problem = StructuredProblem<8, 1, 1, false, float>(grid_dims,
      block_dims,true);
  Grid mesh (grid_dims, 8, block_dims,true);


  writeHierarchicalColouringVTK("./vtk_hier.vtk",problem, mesh.point_coordinates);

  return 0;
}

// vim:set et sw=2 ts=2 fdm=marker:
