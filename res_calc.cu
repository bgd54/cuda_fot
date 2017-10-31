#include <algorithm>
#include <functional>
#include <iostream>
#include <vector>

#include "colouring.hpp"
#include "helper_cuda.h"
#include "kernels/res_calc.hpp"

template <bool SOA> Problem<SOA> initProblem(const std::string &input_dir) {
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
      std::vector<std::pair<MY_SIZE, unsigned>>{});
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
void measurement(const std::string &input_dir, MY_SIZE num) {

  {
    std::cout << "Running non reordered" << std::endl;
    Problem<SOA> problem = initProblem<SOA>(input_dir + "/");
    readData(input_dir + "/", problem);
    std::cout << "Data read." << std::endl;
    problem.template loopGPUHierarchical<res_calc::StepGPUHierarchical>(num);
  }

  {
    std::cout << "Running GPS reordered" << std::endl;
    Problem<SOA> problem = initProblem<SOA>(input_dir + "/");
    readData(input_dir + "/", problem);
    TIMER_START(timer_gps);
    problem.reorder();
    TIMER_PRINT(timer_gps, "reordering");
    problem.template loopGPUHierarchical<res_calc::StepGPUHierarchical>(num);
  }

  {
    std::cout << "Running partitioned" << std::endl;
    Problem<SOA> problem = initProblem<SOA>(input_dir + "/");
    readData(input_dir + "/", problem);
    TIMER_START(timer_metis);
    problem.reorder();
    problem.partition(1.001);
    problem.reorderToPartition();
    problem.renumberPoints();
    TIMER_PRINT(timer_metis, "partitioning");
    problem.template loopGPUHierarchical<res_calc::StepGPUHierarchical>(num);
  }
}

void measurement(const std::string &input_dir, MY_SIZE num) {
  std::cout << "AOS" << std::endl;
  measurement<false>(input_dir, num);
  std::cout << "SOA" << std::endl;
  measurement<true>(input_dir, num);
}

void printUsageTest(const char *program_name) {
  std::cerr << "Usage: " << program_name
            << " <input_dir> <output_dir> <iteration_number>" << std::endl;
}

void printUsageMeasure(const char *program_name) {
  std::cerr << "Usage: " << program_name << " <input_dir> <iteration_number>"
            << std::endl;
}

int mainMeasure(int argc, char *argv[]) {
  if (argc < 3) {
    printUsageMeasure(argv[0]);
    return 1;
  }
  measurement(argv[1], std::atol(argv[2]));
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

int main(int argc, char *argv[]) { return mainTest(argc, argv); }
