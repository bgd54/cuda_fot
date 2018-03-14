#include <algorithm>
#include <functional>
#include <iostream>
#include <vector>

#include "colouring.hpp"
#include "helper_cuda.h"
#include "kernels/res_calc.hpp"

template <bool SOA>
Problem<SOA> initProblem(const std::string &input_dir,
                         MY_SIZE block_size = DEFAULT_BLOCK_SIZE) {
  std::ifstream mesh_res(input_dir + "mesh_res");
  std::ifstream mesh_x(input_dir + "mesh_x");
  std::ifstream mesh_q(input_dir + "mesh_q");
  std::ifstream mesh_adt(input_dir + "mesh_adt");
  Problem<SOA> problem(
      std::vector<std::istream *>{&mesh_res, &mesh_x, &mesh_q, &mesh_adt},
      std::vector<MY_SIZE>{res_calc::MAPPING_DIM, res_calc::MAPPING_DIM,
                           res_calc::MAPPING_DIM, res_calc::MAPPING_DIM},
      std::vector<std::pair<MY_SIZE, unsigned>>{
          {res_calc::RES_DIM, sizeof(double)},
          {res_calc::X_DIM, sizeof(double)},
          {res_calc::Q_DIM, sizeof(double)},
          {res_calc::ADT_DIM, sizeof(double)}},
      std::vector<std::pair<MY_SIZE, unsigned>>{}, block_size);
  return problem;
}

template <bool SOA>
void readData(const std::string &input_dir, Problem<SOA> &problem) {
  std::ifstream data_res(input_dir + "data_res");
  std::ifstream data_x(input_dir + "data_x");
  std::ifstream data_q(input_dir + "data_q");
  std::ifstream data_adt(input_dir + "data_adt");
  problem.template readPointData<double>(data_res, 0);
  problem.template readPointData<double>(data_x, 1);
  problem.template readPointData<double>(data_q, 2);
  problem.template readPointData<double>(data_adt, 3);
}

template <bool SOA>
void writeData(const std::string &output_file, const Problem<SOA> &problem) {
  std::ofstream f(output_file);
  problem.template writePointData<double>(f);
}

template <bool SOA>
void writeAllData(const Problem<SOA> &problem, const std::string &output_dir,
                  bool partition) {
  const std::string fnames_id[] = {"data_res", "data_x", "data_q", "data_adt"};
  const MY_SIZE num_cells = problem.mesh.numCells();
  for (unsigned k = 0; k < problem.mesh.numMappings(); ++k) {
    const MY_SIZE num_points = problem.mesh.numPoints(k);
    const MY_SIZE dim = problem.point_weights[k].getDim();
    std::ofstream os(output_dir + "/" + fnames_id[k]);
    for (MY_SIZE i = 0; i < num_points; ++i) {
      for (MY_SIZE j = 0; j < dim; ++j) {
        const MY_SIZE ind = index<SOA>(num_points, i, dim, j);
        os << (j ? " " : "")
           << problem.point_weights[k].template operator[]<double>(ind);
      }
      os << std::endl;
    }
  }
  {
    const MY_SIZE dim = problem.mesh.cell_to_node[0].getDim();
    std::ofstream os(output_dir + "/mesh_res");
    os << problem.mesh.numPoints(0) << " " << problem.mesh.numCells()
       << std::endl;
    for (MY_SIZE i = 0; i < num_cells; ++i) {
      for (MY_SIZE j = 0; j < dim; ++j) {
        const MY_SIZE ind = index<false>(num_cells, i, dim, j);
        os << (j ? " " : "")
           << problem.mesh.cell_to_node[0].template operator[]<MY_SIZE>(ind);
      }
      os << std::endl;
    }
  }
  {
    const MY_SIZE dim = problem.mesh.cell_to_node[1].getDim();
    std::ofstream os(output_dir + "/mesh_x");
    os << problem.mesh.numPoints(1) << " " << problem.mesh.numCells()
       << std::endl;
    for (MY_SIZE i = 0; i < num_cells; ++i) {
      for (MY_SIZE j = 0; j < dim; ++j) {
        const MY_SIZE ind = index<false>(num_cells, i, dim, j);
        os << (j ? " " : "")
           << problem.mesh.cell_to_node[1].template operator[]<MY_SIZE>(ind);
      }
      os << std::endl;
    }
  }
  if (partition) {
    std::ofstream os(output_dir + "/mesh_part");
    problem.writePartition(os);
  }
}

template <bool SOA>
void runProblem(const std::string &input_dir, MY_SIZE num,
                const std::string &output_dir) {
  Problem<SOA> problem = initProblem<SOA>(input_dir + "/");
  std::string fname_base = output_dir + "/out_" + (SOA ? "SOA" : "AOS") + "_";

  readData(input_dir + "/", problem);
  problem.template loopCPUCellCentred<res_calc::StepSeq>(num);
  writeData(fname_base + "seq", problem);

  readData(input_dir + "/", problem);
  problem.template loopCPUCellCentredOMP<res_calc::StepOMP>(num);
  writeData(fname_base + "omp", problem);

  readData(input_dir + "/", problem);
  problem.template loopGPUCellCentred<res_calc::StepGPUGlobal>(num);
  writeData(fname_base + "glob", problem);

  readData(input_dir + "/", problem);
  problem.template loopGPUHierarchical<res_calc::StepGPUHierarchical>(num);
  writeData(fname_base + "hier", problem);
}

void runProblem(const std::string &input_dir, MY_SIZE num,
                const std::string &output_dir) {
  runProblem<false>(input_dir, num, output_dir);
  runProblem<true>(input_dir, num, output_dir);
}

template <bool SOA>
void testReordering(const std::string &input_dir, MY_SIZE num, bool partition) {
  std::cout << "========================================" << std::endl;
  std::cout << "Airfoil reordering test ";
  std::cout << (SOA ? "SOA" : "AOS");
  std::cout << std::endl << "Iteration: " << num;
  std::cout << " Partition: " << std::boolalpha << partition;
  std::cout << std::endl;
  std::cout << "========================================" << std::endl;
  Problem<SOA> problem1 = initProblem<SOA>(input_dir + "/");
  readData(input_dir + "/", problem1);
  Problem<SOA> problem2 = initProblem<SOA>(input_dir + "/");
  readData(input_dir + "/", problem2);

  problem1.reorder();
  if (partition) {
    problem1.partition(1.001);
    problem1.reorderToPartition();
    problem1.renumberPoints();
  }

  problem1.template loopGPUHierarchical<res_calc::StepGPUHierarchical>(num);
  problem2.template loopGPUHierarchical<res_calc::StepGPUHierarchical>(num);

  double max_diff = 0;
  const MY_SIZE num_points = problem1.mesh.numPoints(0);
  for (MY_SIZE i = 0; i < num_points; ++i) {
    for (unsigned d = 0; d < res_calc::RES_DIM; ++d) {
      const MY_SIZE ind1 = index<SOA>(
          num_points, problem1.applied_permutation[i], res_calc::RES_DIM, d);
      const MY_SIZE ind2 = index<SOA>(num_points, i, res_calc::RES_DIM, d);
      const double data1 =
          problem1.point_weights[0].template operator[]<double>(ind1);
      const double data2 =
          problem2.point_weights[0].template operator[]<double>(ind2);
      const double diff = std::abs(data1 - data2) /
                          (std::min(std::abs(data1), std::abs(data2)) + 1e-6);
      if (max_diff < diff) {
        max_diff = diff;
      }
    }
  }

  std::cout << "Test considered " << (max_diff < 1e-5 ? "PASSED" : "FAILED")
            << std::endl;
}

void testReordering(const std::string &input_dir, MY_SIZE num) {
  testReordering<false>(input_dir, num, false);
  testReordering<false>(input_dir, num, true);
  testReordering<true>(input_dir, num, false);
  testReordering<true>(input_dir, num, true);
}

template <bool SOA>
void measurement(const std::string &input_dir, MY_SIZE num,
                 MY_SIZE block_size,
                 const std::string &input_dir_gps = "",
                 const std::string &input_dir_metis = "") {

  {
    std::cout << "Running non reordered" << std::endl;
    Problem<SOA> problem = initProblem<SOA>(input_dir + "/", block_size);
    readData(input_dir + "/", problem);
    std::cout << "Data read." << std::endl;
    problem.template loopGPUHierarchical<res_calc::StepGPUHierarchical>(num);
    readData(input_dir + "/", problem);
    std::cout << "Data read." << std::endl;
    problem.template loopGPUCellCentred<res_calc::StepGPUGlobal>(num);
  }

  {
    const std::string &used_input_dir =
        input_dir_gps == "" ? input_dir : input_dir_gps;
    std::cout << "Running GPS reordered" << std::endl;
    Problem<SOA> problem = initProblem<SOA>(used_input_dir + "/", block_size);
    readData(used_input_dir + "/", problem);
    TIMER_START(timer_gps);
    if (input_dir_gps == "") {
      problem.reorder();
    }
    TIMER_PRINT(timer_gps, "reordering");
    problem.template loopGPUHierarchical<res_calc::StepGPUHierarchical>(num);
    readData(used_input_dir + "/", problem);
    if (input_dir_gps == "") {
      problem.reorder();
    }
    problem.template loopGPUCellCentred<res_calc::StepGPUGlobal>(num);
  }

  {
    const std::string &used_input_dir =
        input_dir_metis == "" ? input_dir : input_dir_metis;
    std::cout << "Running partitioned" << std::endl;
    Problem<SOA> problem = initProblem<SOA>(used_input_dir + "/", block_size);
    readData(used_input_dir + "/", problem);
    TIMER_START(timer_metis);
    if (input_dir_metis != "") {
      std::ifstream f_part(input_dir_metis + "/mesh_part");
      problem.readPartition(f_part);
      problem.reorderToPartition();
      problem.renumberPoints();
    } else {
      problem.reorder();
      problem.partition(1.001);
      problem.reorderToPartition();
      problem.renumberPoints();
    }
    TIMER_PRINT(timer_metis, "partitioning");
    problem.template loopGPUHierarchical<res_calc::StepGPUHierarchical>(num);
    readData(input_dir + "/", problem);
    if (input_dir_metis != "") {
      std::ifstream f_part(input_dir_metis + "/mesh_part");
      problem.readPartition(f_part);
      problem.reorderToPartition();
      problem.renumberPoints();
    } else {
      problem.reorder();
      problem.partition(1.001);
      problem.reorderToPartition();
      problem.renumberPoints();
    }
    problem.template loopGPUCellCentred<res_calc::StepGPUGlobal>(num);
  }
}

void measurement(const std::string &input_dir, MY_SIZE num,
                 MY_SIZE block_size,
                 const std::string &input_dir_gps = "",
                 const std::string &input_dir_metis = "") {
  std::cout << "AOS" << std::endl;
  measurement<false>(input_dir, num, block_size, input_dir_gps,
                     input_dir_metis);
  std::cout << "SOA" << std::endl;
  measurement<true>(input_dir, num, block_size, input_dir_gps, input_dir_metis);
}

void printUsageTest(const char *program_name) {
  std::cerr << "Usage: " << program_name
            << " <input_dir> <output_dir> <iteration_number>" << std::endl;
}

void printUsageMeasure(const char *program_name) {
  std::cerr << "Usage: " << program_name
            << " <input_dir> <iteration_number> <block_size>" << std::endl;
  std::cerr << "   or: " << program_name
            << " <input_dir> <gps_input_dir> <metis_input_dir>"
            << " <iteration_number> <block_size>" << std::endl;
}

int mainMeasure(int argc, char *argv[]) {
  if (argc != 4 && argc != 6) {
    printUsageMeasure(argv[0]);
    return 1;
  }
  if (argc == 4) {
    measurement(argv[1], std::atol(argv[2]), std::atol(argv[3]));
  } else {
    measurement(argv[1], std::atol(argv[4]), std::atol(argv[5]), argv[2],
                argv[3]);
  }
  return 0;
}

int mainTest(int argc, char *argv[]) {
  if (argc < 4) {
    printUsageTest(argv[0]);
    return 1;
  }
  runProblem(argv[1], std::atol(argv[3]), argv[2]);
  testReordering(argv[1], std::atol(argv[3]));
  return 0;
}

#include "visualization/graph_write_VTK.hpp"

void visualise(const std::string &input_dir, const std::string &fname,
               bool partition, MY_SIZE block_size) {
  Problem<false> problem = initProblem<false>(input_dir + "/", block_size);
  readData(input_dir + "/", problem);
  constexpr MY_SIZE MESH_DIM = 2;

  if (partition) {
    problem.reorder();
    problem.partition(1.001);
    problem.reorderToPartition();
    problem.renumberPoints();
  }

  const HierarchicalColourMemory<false> memory(problem,
                                               problem.partition_vector);
  data_t cell_list(data_t::create<MY_SIZE>(problem.mesh.numCells(), MESH_DIM));
  std::vector<std::vector<std::uint16_t>> data(3);

  MY_SIZE blk_col_ind = 0, blk_ind = 0;
  MY_SIZE cell_ind = 0;
  data_t point_coordinates(data_t::create<float>(problem.mesh.numPoints(1), 3));
  for (const auto &m : memory.colours) {
    data[VTK_IND_THR_COL].insert(data[VTK_IND_THR_COL].end(),
                                 m.cell_colours.begin(), m.cell_colours.end());
    data[VTK_IND_BLK_COL].insert(data[VTK_IND_BLK_COL].end(),
                                 m.cell_colours.size(), blk_col_ind++);
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
        MY_SIZE point_ind = m.cell_list[1][m_cell_ind];
        cell_list.operator[]<MY_SIZE>(MESH_DIM *cell_ind + j) = point_ind;
      }
      ++cell_ind;
    }
    ++blk_ind;
  }
  for (MY_SIZE i = 0; i < problem.mesh.numPoints(1); ++i) {
    point_coordinates.operator[]<float>(3 * i + 0) =
        problem.point_weights[1].operator[]<double>(MESH_DIM *i + 0);
    point_coordinates.operator[]<float>(3 * i + 1) =
        problem.point_weights[1].operator[]<double>(MESH_DIM *i + 1);
    point_coordinates.operator[]<float>(3 * i + 2) = 0;
  }
  writeMeshToVTKAscii(input_dir + "/" + fname, cell_list, point_coordinates,
                      data);
}

int mainVisualise(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <input_dir> <block_size>"
              << std::endl;
    return 1;
  }
  visualise(argv[1], "res_calc.vtk", false, std::atol(argv[2]));
  visualise(argv[1], "res_calc_part.vtk", true, std::atol(argv[2]));
  return 0;
}

void reorder(const std::string &input_dir, const std::string output_dir,
             bool partition, MY_SIZE block_size) {
  Problem<false> problem = initProblem<false>(input_dir + "/", block_size);
  readData(input_dir + "/", problem);

  problem.reorder();
  if (partition) {
    problem.partition(1.001);
    problem.reorderToPartition();
    problem.renumberPoints();
  }

  writeAllData(problem, output_dir, partition);
}

void printUsageReorder(const char *program_name) {
  std::cerr << "Usage: " << program_name
            << " <input_dir> <output_dir_GPS> <output_dir_part>"
            << " <block_size>" << std::endl;
}

int mainReorder(int argc, char *argv[]) {
  if (argc < 5) {
    printUsageReorder(argv[0]);
    return 1;
  }
  reorder(argv[1], argv[2], false, std::atol(argv[4]));
  reorder(argv[1], argv[3], true, std::atol(argv[4]));
  return 0;
}

int main(int argc, char *argv[]) { return mainMeasure(argc, argv); }
