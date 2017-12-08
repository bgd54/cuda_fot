#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <set>
#include <vector>

#include "colouring.hpp"
#include "helper_cuda.h"
#include "kernels/mine.hpp"
#include "problem.hpp"
#include "tests.hpp"

template <unsigned PointDim = 1, unsigned CellDim = 1, bool SOA = false,
          typename DataType = float>
void generateTimesWithBlockDims(MY_SIZE N, MY_SIZE M,
                                std::pair<MY_SIZE, MY_SIZE> block_dims) {
  constexpr MY_SIZE num = 500;
  MY_SIZE block_size = block_dims.first == 0
                           ? block_dims.second
                           : block_dims.first * block_dims.second * 2;
  std::cout << ":::: Generating problems with block size: " << block_dims.first
            << "x" << block_dims.second << " (= " << block_size << ")"
            << "::::" << std::endl
            << "     Point dimension: " << PointDim
            << " Cell dimension: " << CellDim << " SOA: " << std::boolalpha
            << SOA << "\n     Data type: "
            << (sizeof(DataType) == sizeof(float) ? "float" : "double")
            << std::endl;
  std::function<void(implementation_algorithm_t<SOA>)> run =
      [&](implementation_algorithm_t<SOA> algo) {
        Problem<SOA> problem(
            std::move(StructuredProblem<2, PointDim, CellDim, SOA, DataType>(
                N, M, block_dims)));
        std::cout << "--Problem created" << std::endl;
        (problem.*algo)(num);
        std::cout << "--Problem finished." << std::endl;
      };
  run(&Problem<SOA>::template loopGPUCellCentred<
      mine::StepGPUGlobal<2, PointDim, CellDim, DataType>>);
  run(&Problem<SOA>::template loopGPUHierarchical<
      mine::StepGPUHierarchical<2, PointDim, CellDim, DataType>>);
  std::cout << "Finished." << std::endl;
}

template <unsigned PointDim = 1, unsigned CellDim = 1, bool SOA = false,
          typename DataType = float>
void generateTimesWithBlockDims3D(MY_SIZE N1, MY_SIZE N2, MY_SIZE N3,
                                  std::vector<MY_SIZE> block_dims) {
  constexpr MY_SIZE num = 500;
  MY_SIZE block_size =
      StructuredProblem<8, PointDim, CellDim, SOA,
                        DataType>::calculateBlockSize(block_dims);

  std::cout << ":::: Generating problems with block size: " << block_dims[0]
            << "x" << block_dims[1] << "x" << block_dims[2]
            << " (= " << block_size << ")"
            << "::::" << std::endl
            << "     Point dimension: " << PointDim
            << " Cell dimension: " << CellDim << " SOA: " << std::boolalpha
            << SOA << "\n     Data type: "
            << (std::is_same<float, DataType>::value ? "float" : "double")
            << std::endl;
  std::function<void(implementation_algorithm_t<SOA>)> run =
      [&](implementation_algorithm_t<SOA> algo) {
        Problem<SOA> problem(
            std::move(StructuredProblem<8, PointDim, CellDim, SOA, DataType>(
                {N1, N2, N3}, block_dims)));
        std::cout << "--Problem created" << std::endl;
        (problem.*algo)(num);
        std::cout << "--Problem finished." << std::endl;
      };
  run(&Problem<SOA>::template loopGPUHierarchical<
      mine::StepGPUHierarchical<8, PointDim, CellDim, DataType>>);
  std::cout << "Finished." << std::endl;
}

template <unsigned PointDim = 1, unsigned CellDim = 1, bool SOA = false,
          typename DataType = float>
void generateTimesDifferentBlockDims(MY_SIZE N, MY_SIZE M) {
  generateTimesWithBlockDims<PointDim, CellDim, SOA, DataType>(N, M, {0, 32});
  generateTimesWithBlockDims<PointDim, CellDim, SOA, DataType>(N, M, {2, 8});
  generateTimesWithBlockDims<PointDim, CellDim, SOA, DataType>(N, M, {4, 4});
  generateTimesWithBlockDims<PointDim, CellDim, SOA, DataType>(N, M, {0, 128});
  generateTimesWithBlockDims<PointDim, CellDim, SOA, DataType>(N, M, {2, 32});
  generateTimesWithBlockDims<PointDim, CellDim, SOA, DataType>(N, M, {4, 16});
  generateTimesWithBlockDims<PointDim, CellDim, SOA, DataType>(N, M, {8, 8});
  generateTimesWithBlockDims<PointDim, CellDim, SOA, DataType>(N, M, {0, 288});
  generateTimesWithBlockDims<PointDim, CellDim, SOA, DataType>(N, M, {2, 72});
  generateTimesWithBlockDims<PointDim, CellDim, SOA, DataType>(N, M, {4, 36});
  generateTimesWithBlockDims<PointDim, CellDim, SOA, DataType>(N, M, {12, 12});
  generateTimesWithBlockDims<PointDim, CellDim, SOA, DataType>(N, M, {9, 8});
  generateTimesWithBlockDims<PointDim, CellDim, SOA, DataType>(N, M, {0, 512});
  generateTimesWithBlockDims<PointDim, CellDim, SOA, DataType>(N, M, {2, 128});
  generateTimesWithBlockDims<PointDim, CellDim, SOA, DataType>(N, M, {4, 64});
  generateTimesWithBlockDims<PointDim, CellDim, SOA, DataType>(N, M, {8, 32});
  generateTimesWithBlockDims<PointDim, CellDim, SOA, DataType>(N, M, {16, 16});
}

template <unsigned PointDim = 1, unsigned CellDim = 1, bool SOA = false,
          typename DataType = float>
void generateTimesDifferentBlockDims(MY_SIZE N1, MY_SIZE N2, MY_SIZE N3) {
  generateTimesWithBlockDims3D<PointDim, CellDim, SOA, DataType>(N1, N2, N3,
                                                                 {128, 1, 1});
  generateTimesWithBlockDims3D<PointDim, CellDim, SOA, DataType>(N1, N2, N3,
                                                                 {64, 2, 1});
  generateTimesWithBlockDims3D<PointDim, CellDim, SOA, DataType>(N1, N2, N3,
                                                                 {32, 4, 1});
  generateTimesWithBlockDims3D<PointDim, CellDim, SOA, DataType>(N1, N2, N3,
                                                                 {16, 8, 1});
  generateTimesWithBlockDims3D<PointDim, CellDim, SOA, DataType>(N1, N2, N3,
                                                                 {8, 16, 2});
  generateTimesWithBlockDims3D<PointDim, CellDim, SOA, DataType>(N1, N2, N3,
                                                                 {32, 2, 2});
  generateTimesWithBlockDims3D<PointDim, CellDim, SOA, DataType>(N1, N2, N3,
                                                                 {16, 4, 2});
  generateTimesWithBlockDims3D<PointDim, CellDim, SOA, DataType>(N1, N2, N3,
                                                                 {8, 4, 4});
  generateTimesWithBlockDims3D<PointDim, CellDim, SOA, DataType>(N1, N2, N3,
                                                                 {4, 8, 4});
}

template <unsigned MeshDim> void testReordering() {
  MY_SIZE num = 500;
  MY_SIZE N = 100, M = 200;
  constexpr unsigned TEST_DIM = 4;
  constexpr unsigned TEST_CELL_DIM = 4;
  testReordering<MeshDim, TEST_DIM, TEST_CELL_DIM, false, float>(
      num, N, M, &Problem<false>::loopCPUCellCentredOMP<
                     mine::StepOMP<MeshDim, TEST_DIM, TEST_CELL_DIM, float>>,
      &Problem<false>::loopCPUCellCentredOMP<
          mine::StepOMP<MeshDim, TEST_DIM, TEST_CELL_DIM, float>>);
  testReordering<MeshDim, TEST_DIM, TEST_CELL_DIM, true, float>(
      num, N, M, &Problem<true>::loopCPUCellCentredOMP<
                     mine::StepOMP<MeshDim, TEST_DIM, TEST_CELL_DIM, float>>,
      &Problem<true>::loopCPUCellCentredOMP<
          mine::StepOMP<MeshDim, TEST_DIM, TEST_CELL_DIM, float>>);
}

void testReordering() {
  testReordering<2>();
  testReordering<4>();
}

template <unsigned MeshDim> void testPartitioning() {
  MY_SIZE num = 500;
  MY_SIZE N = 100, M = 200;
  constexpr unsigned TEST_DIM = 4;
  constexpr unsigned TEST_CELL_DIM = 4;
  testPartitioning<MeshDim, TEST_DIM, TEST_CELL_DIM, false, float>(num, N, M);
  testPartitioning<MeshDim, TEST_DIM, TEST_CELL_DIM, true, float>(num, N, M);
}

void testPartitioning() {
  testPartitioning<2>();
  testPartitioning<4>();
}

void generateTimesDifferentBlockDims() {
  // SOA
  generateTimesDifferentBlockDims<1, 1, true, float>(1153, 1153);
  generateTimesDifferentBlockDims<2, 1, true, float>(1153, 1153);
  generateTimesDifferentBlockDims<4, 1, true, float>(1153, 1153);
  generateTimesDifferentBlockDims<8, 1, true, float>(1153, 1153);
  generateTimesDifferentBlockDims<1, 1, true, float>(1153, 1153);
  generateTimesDifferentBlockDims<2, 2, true, float>(1153, 1153);
  generateTimesDifferentBlockDims<4, 4, true, float>(1153, 1153);
  generateTimesDifferentBlockDims<8, 8, true, float>(1153, 1153);
  // AOS
  generateTimesDifferentBlockDims<1, 1, false, float>(1153, 1153);
  generateTimesDifferentBlockDims<2, 1, false, float>(1153, 1153);
  generateTimesDifferentBlockDims<4, 1, false, float>(1153, 1153);
  generateTimesDifferentBlockDims<8, 1, false, float>(1153, 1153);
  generateTimesDifferentBlockDims<1, 1, false, float>(1153, 1153);
  generateTimesDifferentBlockDims<2, 2, false, float>(1153, 1153);
  generateTimesDifferentBlockDims<4, 4, false, float>(1153, 1153);
  generateTimesDifferentBlockDims<8, 8, false, float>(1153, 1153);
  // SOA
  generateTimesDifferentBlockDims<1, 1, true, double>(1153, 1153);
  generateTimesDifferentBlockDims<2, 1, true, double>(1153, 1153);
  generateTimesDifferentBlockDims<4, 1, true, double>(1153, 1153);
  generateTimesDifferentBlockDims<8, 1, true, double>(1153, 1153);
  generateTimesDifferentBlockDims<1, 1, true, double>(1153, 1153);
  generateTimesDifferentBlockDims<2, 2, true, double>(1153, 1153);
  generateTimesDifferentBlockDims<4, 4, true, double>(1153, 1153);
  generateTimesDifferentBlockDims<8, 8, true, double>(1153, 1153);
  // AOS
  generateTimesDifferentBlockDims<1, 1, false, double>(1153, 1153);
  generateTimesDifferentBlockDims<2, 1, false, double>(1153, 1153);
  generateTimesDifferentBlockDims<4, 1, false, double>(1153, 1153);
  generateTimesDifferentBlockDims<8, 1, false, double>(1153, 1153);
  generateTimesDifferentBlockDims<1, 1, false, double>(1153, 1153);
  generateTimesDifferentBlockDims<2, 2, false, double>(1153, 1153);
  generateTimesDifferentBlockDims<4, 4, false, double>(1153, 1153);
  generateTimesDifferentBlockDims<8, 8, false, double>(1153, 1153);
}

void generateTimesDifferentBlockDims3D() {
  constexpr MY_SIZE N1 = 257, N2 = 129, N3 = 65;
  // SOA
  generateTimesDifferentBlockDims<1, 1, true, double>(N1, N2, N3);
  generateTimesDifferentBlockDims<2, 1, true, double>(N1, N2, N3);
  generateTimesDifferentBlockDims<4, 1, true, double>(N1, N2, N3);
  generateTimesDifferentBlockDims<8, 1, true, double>(N1, N2, N3);
  generateTimesDifferentBlockDims<1, 1, true, double>(N1, N2, N3);
  generateTimesDifferentBlockDims<2, 2, true, double>(N1, N2, N3);
  generateTimesDifferentBlockDims<4, 4, true, double>(N1, N2, N3);
  generateTimesDifferentBlockDims<8, 8, true, double>(N1, N2, N3);
  // AOS
  generateTimesDifferentBlockDims<1, 1, false, double>(N1, N2, N3);
  generateTimesDifferentBlockDims<2, 1, false, double>(N1, N2, N3);
  generateTimesDifferentBlockDims<4, 1, false, double>(N1, N2, N3);
  generateTimesDifferentBlockDims<8, 1, false, double>(N1, N2, N3);
  generateTimesDifferentBlockDims<1, 1, false, double>(N1, N2, N3);
  generateTimesDifferentBlockDims<2, 2, false, double>(N1, N2, N3);
  generateTimesDifferentBlockDims<4, 4, false, double>(N1, N2, N3);
  generateTimesDifferentBlockDims<8, 8, false, double>(N1, N2, N3);
}

int main(int argc, const char **argv) {
  /*generateTimesFromFile(argc, argv);*/
  /* testImplementations(); */
  /*testReordering();*/
  /*testPartitioning();*/
  /* testMultipleMapping("./test_files/mmapping/", 1); */
  /*generateTimesDifferentBlockDims();*/
  /*measurePartitioning();*/
  generateTimesDifferentBlockDims3D();
  return 0;
}

// vim:set et sw=2 ts=2 fdm=marker:
