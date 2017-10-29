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
  const MY_SIZE num_points = problem.mesh.numPoints(0);
  const MY_SIZE dim = problem.point_weights[0].getDim();
  for (MY_SIZE i = 0; i < num_points; ++i) {
    for (MY_SIZE j = 0; j < dim; ++j) {
      const MY_SIZE ind = index<SOA>(num_points, i, dim, j);
      f << (j ? " " : "")
        << problem.point_weights[0].template operator[]<double>(ind);
    }
    f << std::endl;
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

  /* readData(input_dir + "/", problem); */
  /* problem.template loopGPUHierarchical<res_calc::StepGPUHierarchical>(num);
   */
  /* writeData(fname_base + "hier", problem); */
}

void runProblem(const std::string &input_dir, MY_SIZE num,
                const std::string &output_dir) {
  runProblem<false>(input_dir, num, output_dir);
  runProblem<true>(input_dir, num, output_dir);
}

void printUsage(const char *program_name) {
  std::cerr << "Usage: " << program_name
            << " <input_dir> <output_dir> <iteration_number>" << std::endl;
}

int main(int argc, char *argv[]) {
  if (argc < 4) {
    printUsage(argv[0]);
    return 1;
  }
  runProblem(argv[1], std::atol(argv[3]), argv[2]);
  return 0;
}
