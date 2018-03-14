#include <algorithm>
#include <functional>
#include <iomanip>
#include <iostream>
#include <vector>

#include "colouring.hpp"
#include "helper_cuda.h"
#include "kernels/getacc_scatter.hpp"

template <bool SOA>
Problem<SOA> initProblem(const std::string &input_dir,
                         MY_SIZE block_size = DEFAULT_BLOCK_SIZE) {
  std::ifstream mesh_INC(input_dir + "/mapping.dat");
  return Problem<SOA>(std::vector<std::istream *>{&mesh_INC},
                      std::vector<MY_SIZE>{getacc::MAPPING_DIM},
                      std::vector<std::pair<MY_SIZE, unsigned>>{
                          {getacc::INC_DIM, sizeof(double)}},
                      std::vector<std::pair<MY_SIZE, unsigned>>{
                          {getacc::READ_DIM, sizeof(double)},
                          {getacc::RHO_DIM, sizeof(double)},
                          {getacc::READ_DIM, sizeof(double)},
                          {getacc::READ_DIM, sizeof(double)},
                          {getacc::READ_DIM, sizeof(double)}},
                      block_size);
}

template <bool SOA>
void readData(const std::string &input_dir, Problem<SOA> &problem) {
  std::ifstream data_INC(input_dir + "/point_data_in.dat");
  std::ifstream data_cnmass(input_dir + "/cnmass.dat");
  std::ifstream data_rho(input_dir + "/rho.dat");
  std::ifstream data_cnwt(input_dir + "/cnwt.dat");
  std::ifstream data_cnfx(input_dir + "/cnfx.dat");
  std::ifstream data_cnfy(input_dir + "/cnfy.dat");
  problem.template readPointData<double>(data_INC, 0);
  problem.template readCellData<double>(data_cnmass, 0);
  problem.template readCellData<double>(data_rho, 1);
  problem.template readCellData<double>(data_cnwt, 2);
  problem.template readCellData<double>(data_cnfx, 3);
  problem.template readCellData<double>(data_cnfy, 4);
}

template <bool SOA>
void writeData(const std::string &output_file, const Problem<SOA> &problem) {
  std::ofstream f(output_file);
  const MY_SIZE num_points = problem.mesh.numPoints(0);
  const MY_SIZE dim = problem.point_weights[0].getDim();
  for (MY_SIZE i = 0; i < num_points; ++i) {
    for (MY_SIZE j = 0; j < dim; ++j) {
      const MY_SIZE ind = index<SOA>(num_points, i, dim, j);
      f << (j ? " " : "") << std::setprecision(10)
        << problem.point_weights[0].template operator[]<double>(ind);
    }
    f << std::endl;
  }
}

template <bool SOA>
void writeAllData(const Problem<SOA> &problem, const std::string &output_dir,
                  bool partition) {
  const std::string fnames_id[] = {"point_data_in.dat"};
  const std::string fnames_d[] = {"cnmass.dat", "rho.dat", "cnwt.dat",
                                  "cnfx.dat", "cnfy.dat"};
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
  const MY_SIZE dim = problem.mesh.cell_to_node[0].getDim();
  std::ofstream os(output_dir + "/mapping.dat");
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
  for (unsigned k = 0; k < problem.cell_weights.size(); ++k) {
    const MY_SIZE dim = problem.cell_weights[k].getDim();
    std::ofstream os(output_dir + "/" + fnames_d[k]);
    for (MY_SIZE i = 0; i < num_cells; ++i) {
      for (MY_SIZE j = 0; j < dim; ++j) {
        const MY_SIZE ind = index<true>(num_cells, i, dim, j);
        os << (j ? " " : "")
           << problem.cell_weights[k].template operator[]<double>(ind);
      }
      os << std::endl;
    }
  }
  if (partition) {
    std::ofstream os(output_dir + "/mesh_part");
    problem.writePartition(os);
  }
}

template <bool SOA = false>
using implementation_algorithm_t = void (Problem<SOA>::*)(MY_SIZE);

template <bool SOA>
void testKernel(const std::string &input_dir, MY_SIZE num,
                Problem<SOA> &problem,
                implementation_algorithm_t<SOA> algorithm) {
  readData(input_dir, problem);

  (problem.*algorithm)(num);

  std::ifstream data_ref(input_dir + "/point_data.dat");
  double max_diff = 0;
  const MY_SIZE num_points = problem.mesh.numPoints(0);
  for (MY_SIZE i = 0; i < num_points; ++i) {
    for (unsigned d = 0; d < getacc::INC_DIM; ++d) {
      const MY_SIZE ind_problem = index<SOA>(num_points, i, getacc::INC_DIM, d);
      const double data1 =
          problem.point_weights[0].template operator[]<double>(ind_problem);
      const double data2 = [&data_ref]() {
        double d;
        data_ref >> d;
        return d;
      }();
      const double diff = std::abs(data1 - data2) /
                          (std::min(std::abs(data1), std::abs(data2)) + 1e-6);
      if (max_diff < diff) {
        max_diff = diff;
      }
    }
  }
  writeData(input_dir + "/out_" + (SOA ? "SOA" : "AOS"), problem);
  std::cout << "Maxdiff: " << max_diff << std::endl;
  std::cout << "Test considered " << (max_diff < 1e-5 ? "PASSED" : "FAILED")
            << std::endl;
}

template <bool SOA>
void runProblem(const std::string &input_dir, MY_SIZE num,
                const std::string &output_dir) {
  Problem<SOA> problem = initProblem<SOA>(input_dir + "/");
  std::string fname_base = output_dir + "/out_" + (SOA ? "SOA" : "AOS") + "_";

  readData(input_dir + "/", problem);
  problem.template loopCPUCellCentred<getacc::StepSeq>(num);
  writeData(fname_base + "seq", problem);

  readData(input_dir + "/", problem);
  problem.template loopCPUCellCentredOMP<getacc::StepOMP>(num);
  writeData(fname_base + "omp", problem);

  readData(input_dir + "/", problem);
  problem.template loopGPUCellCentred<getacc::StepGPUGlobal>(num);
  writeData(fname_base + "glob", problem);

  readData(input_dir + "/", problem);
  problem.template loopGPUHierarchical<getacc::StepGPUHierarchical>(num);
  writeData(fname_base + "hier", problem);
}

template <bool SOA> void testKernel(const std::string &input_dir, MY_SIZE num) {
  std::cout << "========================================" << std::endl;
  std::cout << "BookLeaf implementation test ";
  std::cout << (SOA ? "SOA" : "AOS");
  std::cout << std::endl << "Iteration: " << num;
  std::cout << std::endl;
  std::cout << "========================================" << std::endl;

  Problem<SOA> problem = initProblem<SOA>(input_dir);

  std::cout << "Sequential:\n";
  testKernel(input_dir, num, problem,
             &Problem<SOA>::template loopCPUCellCentred<getacc::StepSeq>);

  std::cout << "OpenMP:\n";
  testKernel(input_dir, num, problem,
             &Problem<SOA>::template loopCPUCellCentredOMP<getacc::StepOMP>);

  std::cout << "GPU global:\n";
  testKernel(input_dir, num, problem,
             &Problem<SOA>::template loopGPUCellCentred<getacc::StepGPUGlobal>);

  std::cout << "GPU hierarchical:\n";
  testKernel(
      input_dir, num, problem,
      &Problem<SOA>::template loopGPUHierarchical<getacc::StepGPUHierarchical>);
}

void testKernel(const std::string &input_dir, MY_SIZE num) {
  testKernel<false>(input_dir, num);
  testKernel<true>(input_dir, num);
}

template <bool SOA>
void testReordering(const std::string &input_dir, MY_SIZE num, bool partition) {
  std::cout << "========================================" << std::endl;
  std::cout << "BookLeaf getacc_scatter reordering test ";
  std::cout << (SOA ? "SOA" : "AOS");
  std::cout << std::endl << "Iteration: " << num;
  std::cout << " Partition: " << std::boolalpha << partition;
  std::cout << std::endl;
  std::cout << "========================================" << std::endl;
  Problem<SOA> problem1 = initProblem<SOA>(input_dir);
  readData(input_dir, problem1);
  Problem<SOA> problem2 = initProblem<SOA>(input_dir);
  readData(input_dir, problem2);

  problem1.reorder();
  if (partition) {
    problem1.partition(1.001);
    problem1.reorderToPartition();
    problem1.renumberPoints();
  }

  problem1.template loopGPUHierarchical<getacc::StepGPUHierarchical>(num);
  problem2.template loopGPUHierarchical<getacc::StepGPUHierarchical>(num);

  double max_diff = 0;
  const MY_SIZE num_points = problem1.mesh.numPoints(0);
  for (MY_SIZE i = 0; i < num_points; ++i) {
    for (unsigned d = 0; d < getacc::INC_DIM; ++d) {
      const MY_SIZE ind1 = index<SOA>(
          num_points, problem1.applied_permutation[i], getacc::INC_DIM, d);
      const MY_SIZE ind2 = index<SOA>(num_points, i, getacc::INC_DIM, d);
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

void printUsageTest(const char *program_name) {
  std::cerr << "Usage: " << program_name
            << " <input_dir> <output_dir> <iteration_number>" << std::endl;
}

int mainTest(int argc, char *argv[]) {
  if (argc < 3) {
    printUsageTest(argv[0]);
    return 1;
  }
  testKernel(argv[1], std::atol(argv[3]));
  testReordering(argv[1], std::atol(argv[3]));
  return 0;
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
    problem.template loopGPUHierarchical<getacc::StepGPUHierarchical>(num);
    readData(input_dir + "/", problem);
    std::cout << "Data read." << std::endl;
    problem.template loopGPUCellCentred<getacc::StepGPUGlobal>(num);
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
    problem.template loopGPUHierarchical<getacc::StepGPUHierarchical>(num);
    readData(used_input_dir + "/", problem);
    if (input_dir_gps == "") {
      problem.reorder();
    }
    problem.template loopGPUCellCentred<getacc::StepGPUGlobal>(num);
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
    problem.template loopGPUHierarchical<getacc::StepGPUHierarchical>(num);
    readData(used_input_dir + "/", problem);
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
    problem.template loopGPUCellCentred<getacc::StepGPUGlobal>(num);
  }
}

void measurement(const std::string &input_dir, MY_SIZE num, MY_SIZE block_size,
                 const std::string &input_dir_gps = "",
                 const std::string &input_dir_metis = "") {
  std::cout << "AOS" << std::endl;
  measurement<false>(input_dir, num, block_size, input_dir_gps,
                     input_dir_metis);
  std::cout << "SOA" << std::endl;
  measurement<true>(input_dir, num, block_size, input_dir_gps, input_dir_metis);
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

void reorder(const std::string &input_dir, const std::string output_dir,
             bool partition, MY_SIZE block_size) {
  Problem<false> problem = initProblem<false>(input_dir, block_size);
  readData(input_dir, problem);

  problem.reorder<true>();
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

int main(int argc, char *argv[]) { return mainReorder(argc, argv); }
