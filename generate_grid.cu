#include <iostream>

#include "mesh.hpp"
#include "structured_problem.hpp"
#include <fstream>

constexpr float PARTITION_TOLERANCE = 1.001;

void generateOutputs(const std::string &fname, const std::vector<MY_SIZE> &grid_dim, MY_SIZE block_size) {
  StructuredProblem<> problem (grid_dim, {0,block_size});
  {
    std::ofstream f (fname);
    problem.mesh.writeCellList(f);
  }
  problem.reorder();
  {
    std::ofstream f_gps (fname + ".gps");
    problem.mesh.writeCellList(f_gps);
  }
  problem.partition(PARTITION_TOLERANCE);
  problem.reorderToPartition();
  problem.renumberPoints();
  {
    std::ofstream f_metis(fname + ".metis");
    std::ofstream f_part(fname + ".metis_part");
    problem.mesh.writeCellList(f_metis);
    problem.writePartition(f_part);
  }
}

int main(int argc, char **argv) {
  if (argc < 2 + (MESH_DIM_MACRO == 4 ? 2 : 3)) {
    std::cerr << "Usage: " << argv[0] << " <unordered file name> <GridDim1>"
      << " <GridDim2>" << (MESH_DIM_MACRO == 4 ? "" : 
          MESH_DIM_MACRO == 8 ? " <GridDim3>" : " [<GridDim3>]") << "\n";
    return 1;
  }
  MY_SIZE grid_dim_size = std::min(argc - 2, MESH_DIM_MACRO == 4 ? 2 : 3);
  std::vector<MY_SIZE> grid_dim (grid_dim_size);
  for (size_t i = 0; i < grid_dim.size(); ++i) {
    grid_dim[i] = std::atol(argv[i + 2]);
  }
  generateOutputs(argv[1], grid_dim, 288);
  return 0;
}
