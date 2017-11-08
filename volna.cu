#include <algorithm>
#include <functional>
#include <iostream>
#include <iomanip>
#include <vector>

#include "colouring.hpp"
#include "helper_cuda.h"
#include "kernels/volna.hpp"

template <bool SOA>
Problem<SOA> initProblem(const std::string &input_dir,
                         MY_SIZE block_size = DEFAULT_BLOCK_SIZE) {
  std::ifstream mesh_EC(input_dir + "/edgeToCells");
  std::ifstream mesh_EC2(input_dir + "/edgeToCells");
  Problem<SOA> problem(
      std::vector<std::istream *>{&mesh_EC,&mesh_EC2},
      std::vector<MY_SIZE>{volna::MESH_DIM, volna::MESH_DIM},
      std::vector<std::pair<MY_SIZE, unsigned>>{
          {volna::INC_DIM, sizeof(float)},
          {1, sizeof(float)}},
      std::vector<std::pair<MY_SIZE, unsigned>>{
          {volna::FLUX_DIM, sizeof(float)},
          {volna::CELL_DIM, sizeof(float)},
          {volna::CELL_DIM, sizeof(float)},
          {1, sizeof(int)}}, block_size);
  return problem;
}
template <bool SOA>
void readData(const std::string &input_dir, Problem<SOA> &problem) {
  std::ifstream data_out(input_dir + "/data_out");
  std::ifstream data_vol(input_dir + "/data_vol");
  std::ifstream data_flux(input_dir + "/data_flux");
  std::ifstream data_bathy(input_dir + "/data_bathy");
  std::ifstream data_norms(input_dir + "/data_norms");
  std::ifstream data_isBoundary(input_dir + "/data_isBoundary");
  problem.template readPointData<float>(data_out, 0);
  problem.template readPointData<float>(data_vol, 1);
  problem.template readCellData<float>(data_flux, 0);
  problem.template readCellData<float>(data_bathy, 1);
  problem.template readCellData<float>(data_norms, 2);
  problem.template readCellData<int>(data_isBoundary, 3);
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
        << problem.point_weights[0].template operator[]<float>(ind);
    }
    f << std::endl;
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

  std::ifstream data_ref(input_dir + "/volna_ref");
  double max_diff = 0;
  const MY_SIZE num_points = problem.mesh.numPoints(0);
  for (MY_SIZE i = 0; i < num_points; ++i) {
    for (unsigned d = 0; d < volna::INC_DIM; ++d) {
      const MY_SIZE ind_problem =
          index<SOA>(num_points, i, volna::INC_DIM, d);
      const float data1 =
          problem.point_weights[0].template operator[]<float>(ind_problem);
      const float data2 = [&data_ref]() {
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
  writeData(input_dir + "/out_" + (SOA ? "SOA" : "AOS"),problem);
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
  problem.template loopCPUCellCentred<volna::StepSeq>(num);
  writeData(fname_base + "seq", problem);

  /* readData(input_dir + "/", problem); */
  /* problem.template loopCPUCellCentredOMP<volna::StepOMP>(num); */
  /* writeData(fname_base + "omp", problem); */

  /* readData(input_dir + "/", problem); */
  /* problem.template loopGPUCellCentred<volna::StepGPUGlobal>(num); */
  /* writeData(fname_base + "glob", problem); */

  /* readData(input_dir + "/", problem); */
  /* problem.template loopGPUHierarchical<volna::StepGPUHierarchical>(num); */
  /* writeData(fname_base + "hier", problem); */
}

template <bool SOA> void testKernel(const std::string &input_dir, MY_SIZE num) {
  std::cout << "========================================" << std::endl;
  std::cout << "Volna implementation test ";
  std::cout << (SOA ? "SOA" : "AOS");
  std::cout << std::endl << "Iteration: " << num;
  std::cout << std::endl;
  std::cout << "========================================" << std::endl;

  Problem<SOA> problem = initProblem<SOA>(input_dir);

  std::cout << "Sequential:\n";
  testKernel(input_dir, num, problem,
             &Problem<SOA>::template loopCPUCellCentred<volna::StepSeq>);

  std::cout << "OpenMP:\n";
  testKernel(input_dir, num, problem,
             &Problem<SOA>::template loopCPUCellCentredOMP<volna::StepOMP>);

  std::cout << "GPU global:\n";
  testKernel(input_dir, num, problem,
             &Problem<SOA>::template loopGPUCellCentred<volna::StepGPUGlobal>);

  std::cout << "GPU hierarchical:\n";
  testKernel(
      input_dir, num, problem,
      &Problem<SOA>::template loopGPUHierarchical<volna::StepGPUHierarchical>);
}

void testKernel(const std::string &input_dir, MY_SIZE num) {
  testKernel<false>(input_dir, num);
  /* testKernel<true>(input_dir, num); */
}

template <bool SOA>
void testReordering(const std::string &input_dir, MY_SIZE num, bool partition) {
  std::cout << "========================================" << std::endl;
  std::cout << "Volna SpaceDiscretisation reordering test ";
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

  problem1.template loopGPUHierarchical<volna::StepGPUHierarchical>(num);
  problem2.template loopGPUHierarchical<volna::StepGPUHierarchical>(num);

  double max_diff = 0;
  const MY_SIZE num_points = problem1.mesh.numPoints(0);
  for (MY_SIZE i = 0; i < num_points; ++i) {
    for (unsigned d = 0; d < volna::INC_DIM; ++d) {
      const MY_SIZE ind1 = index<SOA>(
          num_points, problem1.applied_permutation[i], volna::INC_DIM, d);
      const MY_SIZE ind2 = index<SOA>(num_points, i, volna::INC_DIM, d);
      const float data1 =
          problem1.point_weights[0].template operator[]<float>(ind1);
      const float data2 =
          problem2.point_weights[0].template operator[]<float>(ind2);
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
            << " <input_dir> <iteration_number>" << std::endl;
}

int mainTest(int argc, char *argv[]) {
  if (argc < 2) {
    printUsageTest(argv[0]);
    return 1;
  }
  testKernel(argv[1], std::atol(argv[2]));
  testReordering(argv[1], std::atol(argv[2]));
  return 0;
}

template <bool SOA>
void measurement(const std::string &input_dir, MY_SIZE num,
                 MY_SIZE block_size) {
  {
    std::cout << "Running non reordered" << std::endl;
    Problem<SOA> problem = initProblem<SOA>(input_dir + "/", block_size);
    readData(input_dir + "/", problem);
    std::cout << "Data read." << std::endl;
    problem.template loopGPUHierarchical<volna::StepGPUHierarchical>(num);
  }

  {
    std::cout << "Running GPS reordered" << std::endl;
    Problem<SOA> problem = initProblem<SOA>(input_dir + "/", block_size);
    readData(input_dir + "/", problem);
    TIMER_START(timer_gps);
    problem.reorder();
    TIMER_PRINT(timer_gps, "reordering");
    problem.template loopGPUHierarchical<volna::StepGPUHierarchical>(num);
  }

  {
    std::cout << "Running partitioned" << std::endl;
    Problem<SOA> problem = initProblem<SOA>(input_dir + "/", block_size);
    readData(input_dir + "/", problem);
    TIMER_START(timer_metis);
    problem.reorder();
    problem.partition(1.001);
    problem.reorderToPartition();
    problem.renumberPoints();
    TIMER_PRINT(timer_metis, "partitioning");
    problem.template loopGPUHierarchical<volna::StepGPUHierarchical>(num);
  }
}

void measurement(const std::string &input_dir, MY_SIZE num,
                 MY_SIZE block_size) {
  std::cout << "AOS" << std::endl;
  measurement<false>(input_dir, num, block_size);
  std::cout << "SOA" << std::endl;
  measurement<true>(input_dir, num, block_size);
}

void printUsageMeasure(const char *program_name) {
  std::cerr << "Usage: " << program_name
            << " <input_dir> <iteration_number> <block_size>" << std::endl;
}

int mainMeasure(int argc, char *argv[]) {
  if (argc != 4) {
    printUsageMeasure(argv[0]);
    return 1;
  }
  if (argc == 4) {
    measurement(argv[1], std::atol(argv[2]), std::atol(argv[3]));
  }
  return 0;
}

int main(int argc, char *argv[]) { return mainTest(argc, argv); }
