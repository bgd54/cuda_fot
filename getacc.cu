#include <algorithm>
#include <functional>
#include <iostream>
#include <vector>

#include "colouring.hpp"
#include "helper_cuda.h"
#include "kernels/getacc_scatter.hpp"

template <bool SOA> Problem<SOA> initProblem(const std::string &input_dir) {
  std::ifstream mesh_INC(input_dir + "el2node.dat");
  Problem<SOA> problem(
      std::vector<std::istream *>{&mesh_INC},
      std::vector<MY_SIZE>{getacc::MAPPING_DIM},
      std::vector<std::pair<MY_SIZE, unsigned>>{
          {getacc::INC_DIM, sizeof(double)}},
      std::vector<std::pair<MY_SIZE, unsigned>>{
	  {getacc::READ_DIM, sizeof(double)},
	  {getacc::RHO_DIM, sizeof(double)},
	  {getacc::READ_DIM, sizeof(double)},
	  {getacc::READ_DIM, sizeof(double)},
	  {getacc::READ_DIM, sizeof(double)},
	});
  return problem;
}

template <bool SOA>
void readData(const std::string &input_dir, Problem<SOA> &problem) {
  std::ifstream data_INC(input_dir + "point_data.dat");
  std::ifstream data_cnmass(input_dir + "cnmass.dat");
  std::ifstream data_rho(input_dir + "rho.dat");
  std::ifstream data_cnwt(input_dir + "cnwt.dat");
  std::ifstream data_cnfx(input_dir + "cnfx.adt");
  std::ifstream data_cnfy(input_dir + "cnfy.adt");
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

  readData(input_dir + "/", problem);
  problem.template loopGPUHierarchical<res_calc::StepGPUHierarchical>(num);
  writeData(fname_base + "hier", problem);
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
