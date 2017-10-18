#ifndef TESTS_HPP_HHJ8IWSK
#define TESTS_HPP_HHJ8IWSK

#include "kernels/mine.hpp"
#include "partition.hpp"
#include "structured_problem.hpp"
#include <iostream>

// Macros to easily change between mesh dims
#define CONCAT(a, b) a##b
#define CONCAT2(a, b) CONCAT(a, b)
#define MINE_KERNEL(kname) mine::CONCAT2(kname, MESH_DIM_MACRO)

template <unsigned PointDim = 1, unsigned CellDim = 1, bool SOA = false,
          typename DataType = float>
using implementation_algorithm_t =
    void (Problem<PointDim, CellDim, SOA, DataType>::*)(MY_SIZE);

template <unsigned PointDim = 1, unsigned CellDim = 1, bool SOA = false,
          typename DataType = float>
void testTwoImplementations(
    MY_SIZE num, MY_SIZE N, MY_SIZE M,
    implementation_algorithm_t<PointDim, CellDim, SOA, DataType> algorithm1,
    implementation_algorithm_t<PointDim, CellDim, SOA, DataType> algorithm2) {
  std::cout << "========================================" << std::endl;
  std::cout << "Two implementation test" << std::endl;
  std::cout << "PointDim: " << PointDim << ", CellDim: " << CellDim;
  std::cout << ", MeshDim: "
            << Problem<PointDim, CellDim, SOA, DataType>::MESH_DIM;
  std::cout << (SOA ? ", SOA" : ", AOS") << ", Precision: ";
  std::cout << (sizeof(DataType) == sizeof(float) ? "float" : "double");
  std::cout << std::endl << "Iteration: " << num << " size: " << N << ", " << M;
  std::cout << std::endl;
  std::cout << "========================================" << std::endl;

  std::vector<DataType> result1, result2;
  DataType maxdiff = 0;
#ifdef VERBOSE_TEST
  std::vector<MY_SIZE> not_changed, not_changed2;
  MY_SIZE ind_max = 0, dim_max = 0;
  bool single_change_in_node = false;
#endif // VERBOSE_TEST
  {
    srand(1);
    Problem<PointDim, CellDim, SOA, DataType> problem(
        StructuredProblem<PointDim, CellDim, SOA, DataType>(N, M));
    result1.resize(problem.mesh.numPoints() * PointDim);
    // save data before test
    #pragma omp parallel for
    for (MY_SIZE i = 0; i < problem.mesh.numPoints(); ++i) {
      for (MY_SIZE d = 0; d < PointDim; ++d) {
        MY_SIZE ind = index<SOA>(problem.mesh.numPoints(), i, PointDim, d);
        result1[ind] = problem.point_weights.template operator[]<DataType>(ind);
      }
    }

    // run algorithm
    (problem.*algorithm1)(num);

#ifdef VERBOSE_TEST
    DataType abs_max = 0;
#endif // VERBOSE_TEST
    for (MY_SIZE i = 0; i < problem.mesh.numPoints(); ++i) {
#ifdef VERBOSE_TEST
      MY_SIZE value_changed = PointDim;
#endif // VERBOSE_TEST
      for (MY_SIZE d = 0; d < PointDim; ++d) {
        MY_SIZE ind = index<SOA>(problem.mesh.numPoints(), i, PointDim, d);
#ifdef VERBOSE_TEST
        if (result1[ind] ==
            problem.point_weights.template operator[]<DataType>(ind)) {
          if (value_changed == PointDim)
            not_changed.push_back(i);
          value_changed--;
        }
#endif // VERBOSE_TEST
        result1[ind] = problem.point_weights.template operator[]<DataType>(ind);
#ifdef VERBOSE_TEST
        if (abs_max <
            problem.point_weights.template operator[]<DataType>(ind)) {
          abs_max = problem.point_weights.template operator[]<DataType>(ind);
          ind_max = i;
          dim_max = d;
        }
#endif // VERBOSE_TEST
      }
#ifdef VERBOSE_TEST
      if (value_changed != PointDim && value_changed != 0) {
        single_change_in_node = true;
      }
#endif // VERBOSE_TEST
    }
#ifdef VERBOSE_TEST
    std::cout << "Nodes stayed: " << not_changed.size() << "/"
              << problem.mesh.numPoints() << std::endl;
    if (single_change_in_node) {
      std::cout << "WARNING node values updated only some dimension."
                << std::endl;
    }
    for (MY_SIZE i = 0; i < 10 && i < not_changed.size(); ++i) {
      std::cout << "  " << not_changed[i] << std::endl;
    }
    std::cout << "Abs max: " << abs_max << " node: " << ind_max
              << " dim: " << dim_max << std::endl;
#endif // VERBOSE_TEST
  }

#ifdef VERBOSE_TEST
  MY_SIZE ind_diff = 0, dim_diff = 0;
  DataType max = 0;
  single_change_in_node = false;
#endif // VERBOSE_TEST
  {
    srand(1);
    Problem<PointDim, CellDim, SOA, DataType> problem{
        StructuredProblem<PointDim, CellDim, SOA, DataType>(N, M)};
    result2.resize(problem.mesh.numPoints() * PointDim);
    // save data before test
    #pragma omp parallel for
    for (MY_SIZE i = 0; i < problem.mesh.numPoints(); ++i) {
      for (MY_SIZE d = 0; d < PointDim; ++d) {
        MY_SIZE ind = index<SOA>(problem.mesh.numPoints(), i, PointDim, d);
        result2[ind] = problem.point_weights.template operator[]<DataType>(ind);
      }
    }
    // run algorithm
    (problem.*algorithm2)(num);
#ifdef VERBOSE_TEST
    DataType abs_max = 0;
#endif // VERBOSE_TEST

    for (MY_SIZE i = 0; i < problem.mesh.numPoints(); ++i) {
#ifdef VERBOSE_TEST
      MY_SIZE value_changed = PointDim;
#endif // VERBOSE_TEST
      for (MY_SIZE d = 0; d < PointDim; ++d) {
        MY_SIZE ind = index<SOA>(problem.mesh.numPoints(), i, PointDim, d);
#ifdef VERBOSE_TEST
        if (result2[ind] ==
            problem.point_weights.template operator[]<DataType>(ind)) {
          if (value_changed == PointDim)
            not_changed2.push_back(i);
          value_changed--;
        }
#endif // VERBOSE_TEST
        DataType diff =
            std::abs(problem.point_weights.template operator[]<DataType>(ind) -
                     result1[ind]) /
            std::min(result1[ind],
                     problem.point_weights.template operator[]<DataType>(ind));
        if (diff >= maxdiff) {
          maxdiff = diff;
#ifdef VERBOSE_TEST
          ind_diff = i;
          dim_diff = d;
          max = problem.point_weights.template operator[]<DataType>(ind);
#endif // VERBOSE_TEST
        }
#ifdef VERBOSE_TEST
        if (abs_max <
            problem.point_weights.template operator[]<DataType>(ind)) {
          abs_max = problem.point_weights.template operator[]<DataType>(ind);
          ind_max = i;
          dim_max = d;
        }
#endif // VERBOSE_TEST
      }
#ifdef VERBOSE_TEST
      if (value_changed != PointDim && value_changed != 0) {
        single_change_in_node = true;
      }
#endif // VERBOSE_TEST
    }
#ifdef VERBOSE_TEST
    std::cout << "Nodes stayed: " << not_changed2.size() << "/"
              << problem.mesh.numPoints() << std::endl;
    if (single_change_in_node) {
      std::cout << "WARNING node values updated only some dimension."
                << std::endl;
    }
    for (MY_SIZE i = 0; i < 10 && i < not_changed2.size(); ++i) {
      std::cout << "  " << not_changed2[i] << std::endl;
    }
    std::cout << "Abs max: " << abs_max << " node: " << ind_max
              << " dim: " << dim_max << std::endl;
    std::cout << "MAX DIFF: " << maxdiff << " node: " << ind_diff
              << " dim: " << dim_diff << std::endl;
    MY_SIZE ind =
        index<SOA>(problem.mesh.numPoints(), ind_diff, PointDim, dim_diff);
    std::cout << "Values: " << result1[ind] << " / " << max << std::endl;
#endif // VERBOSE_TEST
    std::cout << "Test considered " << (maxdiff < 0.00001 ? "PASSED" : "FAILED")
              << std::endl;
  }
}

template <unsigned PointDim = 1, unsigned CellDim = 1, bool SOA = false,
          typename DataType = float>
void testPartitioning(MY_SIZE num, MY_SIZE N, MY_SIZE M) {
  std::cout << "========================================" << std::endl;
  std::cout << "Partition test" << std::endl;
  std::cout << "PointDim: " << PointDim << ", CellDim: " << CellDim;
  std::cout << ", MeshDim: "
            << Problem<PointDim, CellDim, SOA, DataType>::MESH_DIM;
  std::cout << (SOA ? ", SOA" : ", AOS") << ", Precision: ";
  std::cout << (sizeof(DataType) == sizeof(float) ? "float" : "double");
  std::cout << std::endl << "Iteration: " << num << " size: " << N << ", " << M;
  std::cout << std::endl;
  std::cout << "========================================" << std::endl;

  std::vector<DataType> result1, result2;
  std::vector<MY_SIZE> not_changed, not_changed2;
  DataType maxdiff = 0;
  MY_SIZE ind_max = 0, dim_max = 0;
  bool single_change_in_node = false;
  {
    srand(1);
    Problem<PointDim, CellDim, SOA, DataType> problem{
        StructuredProblem<PointDim, CellDim, SOA, DataType>(N, M)};
    // std::ifstream f("/data/mgiles/asulyok/grid_4_100x200.metis");
    // Problem<PointDim, CellDim, SOA, DataType> problem(f);
    assert(problem.mesh.numPoints() == N * M);
    result1.resize(problem.mesh.numPoints() * PointDim);
    // save data before test
    #pragma omp parallel for
    for (MY_SIZE i = 0; i < problem.mesh.numPoints(); ++i) {
      for (MY_SIZE d = 0; d < PointDim; ++d) {
        MY_SIZE ind = index<SOA>(problem.mesh.numPoints(), i, PointDim, d);
        result1[ind] = problem.point_weights.template operator[]<DataType>(ind);
      }
    }

    // run algorithm
    problem.template loopCPUCellCentredOMP<MINE_KERNEL(StepSeq) < PointDim,
                                           CellDim, DataType>> (num);

    // Partition after
    problem.partition(1.01);
    // std::ifstream f_part("/data/mgiles/asulyok/grid_4_100x200.metis_part");
    // problem.readPartition(f_part);
    problem.reorderToPartition();
    problem.renumberPoints();

    DataType abs_max = 0;
    for (MY_SIZE i = 0; i < problem.mesh.numPoints(); ++i) {
      MY_SIZE value_changed = PointDim;
      for (MY_SIZE d = 0; d < PointDim; ++d) {
        MY_SIZE ind = index<SOA>(problem.mesh.numPoints(), i, PointDim, d);
        if (result1[ind] ==
            problem.point_weights.template operator[]<DataType>(ind)) {
          if (value_changed == PointDim)
            not_changed.push_back(i);
          value_changed--;
        }
        result1[ind] = problem.point_weights.template operator[]<DataType>(ind);
        if (abs_max <
            problem.point_weights.template operator[]<DataType>(ind)) {
          abs_max = problem.point_weights.template operator[]<DataType>(ind);
          ind_max = i;
          dim_max = d;
        }
      }
      if (value_changed != PointDim && value_changed != 0) {
        single_change_in_node = true;
      }
    }
    std::cout << "Nodes stayed: " << not_changed.size() << "/"
              << problem.mesh.numPoints() << std::endl;
    if (single_change_in_node) {
      std::cout << "WARNING node values updated only some dimension."
                << std::endl;
    }
    for (MY_SIZE i = 0; i < 10 && i < not_changed.size(); ++i) {
      std::cout << "  " << not_changed[i] << std::endl;
    }
    std::cout << "Abs max: " << abs_max << " node: " << ind_max
              << " dim: " << dim_max << std::endl;
  }

  MY_SIZE ind_diff = 0, dim_diff = 0;
  DataType max = 0;
  single_change_in_node = false;
  {
    srand(1);
    // std::ifstream f("/data/mgiles/asulyok/grid_4_100x200.metis");
    // Problem<PointDim, CellDim, SOA, DataType> problem(f);
    Problem<PointDim, CellDim, SOA, DataType> problem{
        StructuredProblem<PointDim, CellDim, SOA, DataType>(N, M)};
    result2.resize(problem.mesh.numPoints() * PointDim);
    // save data before test
    #pragma omp parallel for
    for (MY_SIZE i = 0; i < problem.mesh.numPoints(); ++i) {
      for (MY_SIZE d = 0; d < PointDim; ++d) {
        MY_SIZE ind = index<SOA>(problem.mesh.numPoints(), i, PointDim, d);
        result2[ind] = problem.point_weights.template operator[]<DataType>(ind);
      }
    }
    // Create partitioning
    problem.partition(1.01);
    // std::ifstream f_part("/data/mgiles/asulyok/grid_4_100x200.metis_part");
    // problem.readPartition(f_part);
    problem.reorderToPartition();
    problem.renumberPoints();

    // run algorithm
    problem.loopGPUHierarchical(num);
    DataType abs_max = 0;

    for (MY_SIZE i = 0; i < problem.mesh.numPoints(); ++i) {
      MY_SIZE value_changed = PointDim;
      for (MY_SIZE d = 0; d < PointDim; ++d) {
        MY_SIZE ind = index<SOA>(problem.mesh.numPoints(), i, PointDim, d);
        if (result2[ind] ==
            problem.point_weights.template operator[]<DataType>(ind)) {
          if (value_changed == PointDim)
            not_changed2.push_back(i);
          value_changed--;
        }
        DataType diff =
            std::abs(problem.point_weights.template operator[]<DataType>(ind) -
                     result1[ind]) /
            std::min(result1[ind],
                     problem.point_weights.template operator[]<DataType>(ind));
        if (diff >= maxdiff) {
          maxdiff = diff;
          ind_diff = i;
          dim_diff = d;
          max = problem.point_weights.template operator[]<DataType>(ind);
        }
        if (abs_max <
            problem.point_weights.template operator[]<DataType>(ind)) {
          abs_max = problem.point_weights.template operator[]<DataType>(ind);
          ind_max = i;
          dim_max = d;
        }
      }
      if (value_changed != PointDim && value_changed != 0) {
        single_change_in_node = true;
      }
    }
    std::cout << "Nodes stayed: " << not_changed2.size() << "/"
              << problem.mesh.numPoints() << std::endl;
    if (single_change_in_node) {
      std::cout << "WARNING node values updated only some dimension."
                << std::endl;
    }
    for (MY_SIZE i = 0; i < 10 && i < not_changed2.size(); ++i) {
      std::cout << "  " << not_changed2[i] << std::endl;
    }
    std::cout << "Abs max: " << abs_max << " node: " << ind_max
              << " dim: " << dim_max << std::endl;
    std::cout << "MAX DIFF: " << maxdiff << " node: " << ind_diff
              << " dim: " << dim_diff << std::endl;
    MY_SIZE ind =
        index<SOA>(problem.mesh.numPoints(), ind_diff, PointDim, dim_diff);
    std::cout << "Values: " << result1[ind] << " / " << max << std::endl;
    std::cout << "Test considered " << (maxdiff < 0.00001 ? "PASSED" : "FAILED")
              << std::endl;
  }
}

template <unsigned PointDim = 1, unsigned CellDim = 1, bool SOA = false,
          typename DataType = float>
void testReordering(
    MY_SIZE num, MY_SIZE N, MY_SIZE M,
    implementation_algorithm_t<PointDim, CellDim, SOA, DataType> algorithm1,
    implementation_algorithm_t<PointDim, CellDim, SOA, DataType> algorithm2) {
  std::cout << "========================================" << std::endl;
  std::cout << "Two implementation test" << std::endl;
  std::cout << "PointDim: " << PointDim << ", CellDim: " << CellDim;
  std::cout << ", MeshDim: "
            << Problem<PointDim, CellDim, SOA, DataType>::MESH_DIM;
  std::cout << (SOA ? ", SOA" : ", AOS") << ", Precision: ";
  std::cout << (sizeof(DataType) == sizeof(float) ? "float" : "double");
  std::cout << std::endl << "Iteration: " << num << " size: " << N << ", " << M;
  std::cout << std::endl;
  std::cout << "========================================" << std::endl;

  std::vector<DataType> result1, result2;
  std::vector<MY_SIZE> not_changed;
  DataType maxdiff = 0;
  MY_SIZE ind_max = 0, dim_max = 0;
  bool single_change_in_node = false;
  {
    srand(1);
    Problem<PointDim, CellDim, SOA, DataType> problem{
        StructuredProblem<PointDim, CellDim, SOA, DataType>(N, M)};
    // reorder first
    problem.reorder();

    result1.resize(problem.mesh.numPoints() * PointDim);
    // save data before test
    #pragma omp parallel for
    for (MY_SIZE i = 0; i < problem.mesh.numPoints(); ++i) {
      for (MY_SIZE d = 0; d < PointDim; ++d) {
        MY_SIZE ind = index<SOA>(problem.mesh.numPoints(), i, PointDim, d);
        result1[ind] = problem.point_weights.template operator[]<DataType>(ind);
      }
    }

    // run algorithm
    (problem.*algorithm1)(num);

    DataType abs_max = 0;
    for (MY_SIZE i = 0; i < problem.mesh.numPoints(); ++i) {
      MY_SIZE value_changed = PointDim;
      for (MY_SIZE d = 0; d < PointDim; ++d) {
        MY_SIZE ind = index<SOA>(problem.mesh.numPoints(), i, PointDim, d);
        if (result1[ind] ==
            problem.point_weights.template operator[]<DataType>(ind)) {
          if (value_changed == PointDim)
            not_changed.push_back(i);
          value_changed--;
        }
        result1[ind] = problem.point_weights.template operator[]<DataType>(ind);
        if (abs_max <
            problem.point_weights.template operator[]<DataType>(ind)) {
          abs_max = problem.point_weights.template operator[]<DataType>(ind);
          ind_max = i;
          dim_max = d;
        }
      }
      if (value_changed != PointDim && value_changed != 0) {
        single_change_in_node = true;
      }
    }
    std::cout << "Nodes stayed: " << not_changed.size() << "/"
              << problem.mesh.numPoints() << std::endl;
    if (single_change_in_node) {
      std::cout << "WARNING node values updated only some dimension."
                << std::endl;
    }
    for (MY_SIZE i = 0; i < 10 && i < not_changed.size(); ++i) {
      std::cout << "  " << not_changed[i] << std::endl;
    }
    std::cout << "Abs max: " << abs_max << " node: " << ind_max
              << " dim: " << dim_max << std::endl;
  }

  MY_SIZE ind_diff = 0, dim_diff = 0;
  DataType max = 0;
  single_change_in_node = false;
  {
    srand(1);
    Problem<PointDim, CellDim, SOA, DataType> problem{
        StructuredProblem<PointDim, CellDim, SOA, DataType>(N, M)};
    result2.resize(problem.mesh.numPoints() * PointDim);
    // save data before test
    #pragma omp parallel for
    for (MY_SIZE i = 0; i < problem.mesh.numPoints(); ++i) {
      for (MY_SIZE d = 0; d < PointDim; ++d) {
        MY_SIZE ind = index<SOA>(problem.mesh.numPoints(), i, PointDim, d);
        result2[ind] = problem.point_weights.template operator[]<DataType>(ind);
      }
    }
    // run algorithm
    (problem.*algorithm2)(num);

    // reorder after
    problem.reorder();

    DataType abs_max = 0;

    for (MY_SIZE i = 0; i < problem.mesh.numPoints(); ++i) {
      for (MY_SIZE d = 0; d < PointDim; ++d) {
        MY_SIZE ind = index<SOA>(problem.mesh.numPoints(), i, PointDim, d);
        DataType diff =
            std::abs(problem.point_weights.template operator[]<DataType>(ind) -
                     result1[ind]) /
            std::min(result1[ind],
                     problem.point_weights.template operator[]<DataType>(ind));
        if (diff >= maxdiff) {
          maxdiff = diff;
          ind_diff = i;
          dim_diff = d;
          max = problem.point_weights.template operator[]<DataType>(ind);
        }
        if (abs_max <
            problem.point_weights.template operator[]<DataType>(ind)) {
          abs_max = problem.point_weights.template operator[]<DataType>(ind);
          ind_max = i;
          dim_max = d;
        }
      }
    }
    std::cout << "Abs max: " << abs_max << " node: " << ind_max
              << " dim: " << dim_max << std::endl;
    std::cout << "MAX DIFF: " << maxdiff << " node: " << ind_diff
              << " dim: " << dim_diff << std::endl;
    MY_SIZE ind =
        index<SOA>(problem.mesh.numPoints(), ind_diff, PointDim, dim_diff);
    std::cout << "Values: " << result1[ind] << " / " << max << std::endl;
    std::cout << "Test considered " << (maxdiff < 0.00001 ? "PASSED" : "FAILED")
              << std::endl;
  }
}

#define TEST_TWO_IMPLEMENTATIONS(algorithm1, algorithm2, algorithm1_SOA,       \
                                 algorithm2_SOA)                               \
  {                                                                            \
    /* {{{1 */                                                                 \
    constexpr MY_SIZE num = 500;                                               \
    constexpr MY_SIZE N = 100, M = 200;                                        \
    /* float */                                                                \
    {                                                                          \
      constexpr unsigned TEST_DIM = 1;                                         \
      constexpr unsigned TEST_CELL_DIM = 1;                                    \
      using TEST_DATA_TYPE = float;                                            \
      /* {{{2 */                                                               \
      testTwoImplementations<TEST_DIM, TEST_CELL_DIM, false, TEST_DATA_TYPE>(  \
          num, N, M, algorithm1, algorithm2);                                  \
      testTwoImplementations<TEST_DIM, TEST_CELL_DIM, true, TEST_DATA_TYPE>(   \
          num, N, M, algorithm1_SOA, algorithm2_SOA);                          \
      /* 2}}} */                                                               \
    }                                                                          \
    {                                                                          \
      constexpr unsigned TEST_DIM = 2;                                         \
      constexpr unsigned TEST_CELL_DIM = 1;                                    \
      using TEST_DATA_TYPE = float;                                            \
      /* {{{2 */                                                               \
      testTwoImplementations<TEST_DIM, TEST_CELL_DIM, false, TEST_DATA_TYPE>(  \
          num, N, M, algorithm1, algorithm2);                                  \
      testTwoImplementations<TEST_DIM, TEST_CELL_DIM, true, TEST_DATA_TYPE>(   \
          num, N, M, algorithm1_SOA, algorithm2_SOA);                          \
      /* 2}}} */                                                               \
    }                                                                          \
    {                                                                          \
      constexpr unsigned TEST_DIM = 2;                                         \
      constexpr unsigned TEST_CELL_DIM = 2;                                    \
      using TEST_DATA_TYPE = float;                                            \
      /* {{{2 */                                                               \
      testTwoImplementations<TEST_DIM, TEST_CELL_DIM, false, TEST_DATA_TYPE>(  \
          num, N, M, algorithm1, algorithm2);                                  \
      testTwoImplementations<TEST_DIM, TEST_CELL_DIM, true, TEST_DATA_TYPE>(   \
          num, N, M, algorithm1_SOA, algorithm2_SOA);                          \
      /* 2}}} */                                                               \
    }                                                                          \
    {                                                                          \
      constexpr unsigned TEST_DIM = 4;                                         \
      constexpr unsigned TEST_CELL_DIM = 1;                                    \
      using TEST_DATA_TYPE = float;                                            \
      /* {{{2 */                                                               \
      testTwoImplementations<TEST_DIM, TEST_CELL_DIM, false, TEST_DATA_TYPE>(  \
          num, N, M, algorithm1, algorithm2);                                  \
      testTwoImplementations<TEST_DIM, TEST_CELL_DIM, true, TEST_DATA_TYPE>(   \
          num, N, M, algorithm1_SOA, algorithm2_SOA);                          \
      /* 2}}} */                                                               \
    }                                                                          \
    {                                                                          \
      constexpr unsigned TEST_DIM = 4;                                         \
      constexpr unsigned TEST_CELL_DIM = 4;                                    \
      using TEST_DATA_TYPE = float;                                            \
      /* {{{2 */                                                               \
      testTwoImplementations<TEST_DIM, TEST_CELL_DIM, false, TEST_DATA_TYPE>(  \
          num, N, M, algorithm1, algorithm2);                                  \
      testTwoImplementations<TEST_DIM, TEST_CELL_DIM, true, TEST_DATA_TYPE>(   \
          num, N, M, algorithm1_SOA, algorithm2_SOA);                          \
      /* 2}}} */                                                               \
    }                                                                          \
    {                                                                          \
      constexpr unsigned TEST_DIM = 8;                                         \
      constexpr unsigned TEST_CELL_DIM = 1;                                    \
      using TEST_DATA_TYPE = float;                                            \
      /* {{{2 */                                                               \
      testTwoImplementations<TEST_DIM, TEST_CELL_DIM, false, TEST_DATA_TYPE>(  \
          num, N, M, algorithm1, algorithm2);                                  \
      testTwoImplementations<TEST_DIM, TEST_CELL_DIM, true, TEST_DATA_TYPE>(   \
          num, N, M, algorithm1_SOA, algorithm2_SOA);                          \
      /* 2}}} */                                                               \
    }                                                                          \
    {                                                                          \
      constexpr unsigned TEST_DIM = 8;                                         \
      constexpr unsigned TEST_CELL_DIM = 8;                                    \
      using TEST_DATA_TYPE = float;                                            \
      /* {{{2 */                                                               \
      testTwoImplementations<TEST_DIM, TEST_CELL_DIM, false, TEST_DATA_TYPE>(  \
          num, N, M, algorithm1, algorithm2);                                  \
      testTwoImplementations<TEST_DIM, TEST_CELL_DIM, true, TEST_DATA_TYPE>(   \
          num, N, M, algorithm1_SOA, algorithm2_SOA);                          \
      /* 2}}} */                                                               \
    }                                                                          \
    /* double */                                                               \
    {                                                                          \
      constexpr unsigned TEST_DIM = 1;                                         \
      constexpr unsigned TEST_CELL_DIM = 1;                                    \
      using TEST_DATA_TYPE = double;                                           \
      /* {{{2 */                                                               \
      testTwoImplementations<TEST_DIM, TEST_CELL_DIM, false, TEST_DATA_TYPE>(  \
          num, N, M, algorithm1, algorithm2);                                  \
      testTwoImplementations<TEST_DIM, TEST_CELL_DIM, true, TEST_DATA_TYPE>(   \
          num, N, M, algorithm1_SOA, algorithm2_SOA);                          \
      /* 2}}} */                                                               \
    }                                                                          \
    {                                                                          \
      constexpr unsigned TEST_DIM = 2;                                         \
      constexpr unsigned TEST_CELL_DIM = 1;                                    \
      using TEST_DATA_TYPE = double;                                           \
      /* {{{2 */                                                               \
      testTwoImplementations<TEST_DIM, TEST_CELL_DIM, false, TEST_DATA_TYPE>(  \
          num, N, M, algorithm1, algorithm2);                                  \
      testTwoImplementations<TEST_DIM, TEST_CELL_DIM, true, TEST_DATA_TYPE>(   \
          num, N, M, algorithm1_SOA, algorithm2_SOA);                          \
      /* 2}}} */                                                               \
    }                                                                          \
    {                                                                          \
      constexpr unsigned TEST_DIM = 2;                                         \
      constexpr unsigned TEST_CELL_DIM = 2;                                    \
      using TEST_DATA_TYPE = double;                                           \
      /* {{{2 */                                                               \
      testTwoImplementations<TEST_DIM, TEST_CELL_DIM, false, TEST_DATA_TYPE>(  \
          num, N, M, algorithm1, algorithm2);                                  \
      testTwoImplementations<TEST_DIM, TEST_CELL_DIM, true, TEST_DATA_TYPE>(   \
          num, N, M, algorithm1_SOA, algorithm2_SOA);                          \
      /* 2}}} */                                                               \
    }                                                                          \
    {                                                                          \
      constexpr unsigned TEST_DIM = 4;                                         \
      constexpr unsigned TEST_CELL_DIM = 1;                                    \
      using TEST_DATA_TYPE = double;                                           \
      /* {{{2 */                                                               \
      testTwoImplementations<TEST_DIM, TEST_CELL_DIM, false, TEST_DATA_TYPE>(  \
          num, N, M, algorithm1, algorithm2);                                  \
      testTwoImplementations<TEST_DIM, TEST_CELL_DIM, true, TEST_DATA_TYPE>(   \
          num, N, M, algorithm1_SOA, algorithm2_SOA);                          \
      /* 2}}} */                                                               \
    }                                                                          \
    {                                                                          \
      constexpr unsigned TEST_DIM = 4;                                         \
      constexpr unsigned TEST_CELL_DIM = 4;                                    \
      using TEST_DATA_TYPE = double;                                           \
      /* {{{2 */                                                               \
      testTwoImplementations<TEST_DIM, TEST_CELL_DIM, false, TEST_DATA_TYPE>(  \
          num, N, M, algorithm1, algorithm2);                                  \
      testTwoImplementations<TEST_DIM, TEST_CELL_DIM, true, TEST_DATA_TYPE>(   \
          num, N, M, algorithm1_SOA, algorithm2_SOA);                          \
      /* 2}}} */                                                               \
    }                                                                          \
    {                                                                          \
      constexpr unsigned TEST_DIM = 8;                                         \
      constexpr unsigned TEST_CELL_DIM = 1;                                    \
      using TEST_DATA_TYPE = double;                                           \
      /* {{{2 */                                                               \
      testTwoImplementations<TEST_DIM, TEST_CELL_DIM, false, TEST_DATA_TYPE>(  \
          num, N, M, algorithm1, algorithm2);                                  \
      testTwoImplementations<TEST_DIM, TEST_CELL_DIM, true, TEST_DATA_TYPE>(   \
          num, N, M, algorithm1_SOA, algorithm2_SOA);                          \
      /* 2}}} */                                                               \
    }                                                                          \
    {                                                                          \
      constexpr unsigned TEST_DIM = 8;                                         \
      constexpr unsigned TEST_CELL_DIM = 8;                                    \
      using TEST_DATA_TYPE = double;                                           \
      /* {{{2 */                                                               \
      testTwoImplementations<TEST_DIM, TEST_CELL_DIM, false, TEST_DATA_TYPE>(  \
          num, N, M, algorithm1, algorithm2);                                  \
      testTwoImplementations<TEST_DIM, TEST_CELL_DIM, true, TEST_DATA_TYPE>(   \
          num, N, M, algorithm1_SOA, algorithm2_SOA);                          \
      /* 2}}} */                                                               \
    }                                                                          \
  } /* 1}}}*/

void testImplementations() {
  // std::cout << "========================================" << std::endl;
  // std::cout << "#         Sequential - OpenMP          #" << std::endl;
  // TEST_TWO_IMPLEMENTATIONS(
  //    (&Problem<TEST_DIM, TEST_CELL_DIM, false, TEST_DATA_TYPE>::
  //         loopCPUCellCentred<MINE_KERNEL(StepSeq) < TEST_DIM, TEST_CELL_DIM,
  //                            TEST_DATA_TYPE>>),
  //    (&Problem<TEST_DIM, TEST_CELL_DIM, false, TEST_DATA_TYPE>::
  //         loopCPUCellCentredOMP<MINE_KERNEL(StepOMP) < TEST_DIM,
  //         TEST_CELL_DIM,
  //                               TEST_DATA_TYPE>>),
  //    (&Problem<TEST_DIM, TEST_CELL_DIM, true, TEST_DATA_TYPE>::
  //         loopCPUCellCentred<MINE_KERNEL(StepSeq) < TEST_DIM, TEST_CELL_DIM,
  //                            TEST_DATA_TYPE>>),
  //    (&Problem<TEST_DIM, TEST_CELL_DIM, true, TEST_DATA_TYPE>::
  //         loopCPUCellCentredOMP<MINE_KERNEL(StepOMP) < TEST_DIM,
  //         TEST_CELL_DIM,
  //                               TEST_DATA_TYPE>>));

  // std::cout << "========================================" << std::endl;
  // std::cout << "#         OpenMP - GPU Global          #" << std::endl;
  // TEST_TWO_IMPLEMENTATIONS(
  //    (&Problem<TEST_DIM, TEST_CELL_DIM, false, TEST_DATA_TYPE>::
  //         loopCPUCellCentredOMP<MINE_KERNEL(StepOMP) < TEST_DIM,
  //         TEST_CELL_DIM,
  //                               TEST_DATA_TYPE>>),
  //    (&Problem<TEST_DIM, TEST_CELL_DIM, false, TEST_DATA_TYPE>::
  //         loopGPUCellCentred<MINE_KERNEL(StepGPUGlobal) < TEST_DIM,
  //                            TEST_CELL_DIM, TEST_DATA_TYPE>>),
  //    (&Problem<TEST_DIM, TEST_CELL_DIM, true, TEST_DATA_TYPE>::
  //         loopCPUCellCentredOMP<MINE_KERNEL(StepOMP) < TEST_DIM,
  //         TEST_CELL_DIM,
  //                               TEST_DATA_TYPE>>),
  //    (&Problem<TEST_DIM, TEST_CELL_DIM, true, TEST_DATA_TYPE>::
  //         loopGPUCellCentred<MINE_KERNEL(StepGPUGlobal) < TEST_DIM,
  //                            TEST_CELL_DIM, TEST_DATA_TYPE>>));

  std::cout << "========================================" << std::endl;
  std::cout << "#       OpenMP - GPU Hierarchical      #" << std::endl;
  TEST_TWO_IMPLEMENTATIONS(
      (&Problem<TEST_DIM, TEST_CELL_DIM, false, TEST_DATA_TYPE>::
           loopCPUCellCentredOMP<MINE_KERNEL(StepOMP) < TEST_DIM, TEST_CELL_DIM,
                                 TEST_DATA_TYPE>>),
      (&Problem<TEST_DIM, TEST_CELL_DIM, false, TEST_DATA_TYPE>::
           loopGPUHierarchical<MINE_KERNEL(StepGPUHierarchical) < TEST_DIM,
                               TEST_CELL_DIM, TEST_DATA_TYPE>>),
      (&Problem<TEST_DIM, TEST_CELL_DIM, true, TEST_DATA_TYPE>::
           loopCPUCellCentredOMP<MINE_KERNEL(StepOMP) < TEST_DIM, TEST_CELL_DIM,
                                 TEST_DATA_TYPE>>),
      (&Problem<TEST_DIM, TEST_CELL_DIM, true, TEST_DATA_TYPE>::
           loopGPUHierarchical<MINE_KERNEL(StepGPUHierarchical) < TEST_DIM,
                               TEST_CELL_DIM, TEST_DATA_TYPE>>));
}

#endif /* end of include guard: TESTS_HPP_HHJ8IWSK */

// vim:set et sts=2 sw=2 ts=2 fdm=marker:
