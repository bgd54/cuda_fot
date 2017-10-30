#ifndef TESTS_HPP_HHJ8IWSK
#define TESTS_HPP_HHJ8IWSK

#include "kernels/mine.hpp"
#include "kernels/mine2.hpp"
#include "partition.hpp"
#include "structured_problem.hpp"
#include <iostream>

template <bool SOA = false>
using implementation_algorithm_t = void (Problem<SOA>::*)(MY_SIZE);

template <unsigned MeshDim, unsigned PointDim, unsigned CellDim, bool SOA,
          typename DataType = float>
void testTwoImplementations(MY_SIZE num, MY_SIZE N, MY_SIZE M,
                            implementation_algorithm_t<SOA> algorithm1,
                            implementation_algorithm_t<SOA> algorithm2) {
  /* testTwoImplementations {{{1 */
  std::cout << "========================================" << std::endl;
  std::cout << "Two implementation test" << std::endl;
  std::cout << "PointDim: " << PointDim << ", CellDim: " << CellDim;
  std::cout << ", MeshDim: " << MeshDim << std::endl;
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
    Problem<SOA> problem(
        StructuredProblem<MeshDim, PointDim, CellDim, SOA, DataType>(N, M));
    result1.resize(problem.mesh.numPoints(0) * PointDim);
    // save data before test
    #pragma omp parallel for
    for (MY_SIZE i = 0; i < problem.mesh.numPoints(0); ++i) {
      for (MY_SIZE d = 0; d < PointDim; ++d) {
        MY_SIZE ind = index<SOA>(problem.mesh.numPoints(0), i, PointDim, d);
        result1[ind] =
            problem.point_weights[0].template operator[]<DataType>(ind);
      }
    }

    // run algorithm
    (problem.*algorithm1)(num);

#ifdef VERBOSE_TEST
    DataType abs_max = 0;
#endif // VERBOSE_TEST
    for (MY_SIZE i = 0; i < problem.mesh.numPoints(0); ++i) {
#ifdef VERBOSE_TEST
      MY_SIZE value_changed = PointDim;
#endif // VERBOSE_TEST
      for (MY_SIZE d = 0; d < PointDim; ++d) {
        MY_SIZE ind = index<SOA>(problem.mesh.numPoints(0), i, PointDim, d);
#ifdef VERBOSE_TEST
        if (result1[ind] ==
            problem.point_weights[0].template operator[]<DataType>(ind)) {
          if (value_changed == PointDim)
            not_changed.push_back(i);
          value_changed--;
        }
#endif // VERBOSE_TEST
        result1[ind] =
            problem.point_weights[0].template operator[]<DataType>(ind);
#ifdef VERBOSE_TEST
        if (abs_max <
            problem.point_weights[0].template operator[]<DataType>(ind)) {
          abs_max = problem.point_weights[0].template operator[]<DataType>(ind);
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
              << problem.mesh.numPoints(0) << std::endl;
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
    Problem<SOA> problem{
        StructuredProblem<MeshDim, PointDim, CellDim, SOA, DataType>(N, M)};
    result2.resize(problem.mesh.numPoints(0) * PointDim);
    // save data before test
    #pragma omp parallel for
    for (MY_SIZE i = 0; i < problem.mesh.numPoints(0); ++i) {
      for (MY_SIZE d = 0; d < PointDim; ++d) {
        MY_SIZE ind = index<SOA>(problem.mesh.numPoints(0), i, PointDim, d);
        result2[ind] =
            problem.point_weights[0].template operator[]<DataType>(ind);
      }
    }
    // run algorithm
    (problem.*algorithm2)(num);
#ifdef VERBOSE_TEST
    DataType abs_max = 0;
#endif // VERBOSE_TEST

    for (MY_SIZE i = 0; i < problem.mesh.numPoints(0); ++i) {
#ifdef VERBOSE_TEST
      MY_SIZE value_changed = PointDim;
#endif // VERBOSE_TEST
      for (MY_SIZE d = 0; d < PointDim; ++d) {
        MY_SIZE ind = index<SOA>(problem.mesh.numPoints(0), i, PointDim, d);
#ifdef VERBOSE_TEST
        if (result2[ind] ==
            problem.point_weights[0].template operator[]<DataType>(ind)) {
          if (value_changed == PointDim)
            not_changed2.push_back(i);
          value_changed--;
        }
#endif // VERBOSE_TEST
        DataType diff =
            std::abs(
                problem.point_weights[0].template operator[]<DataType>(ind) -
                result1[ind]) /
            std::min(
                result1[ind],
                problem.point_weights[0].template operator[]<DataType>(ind));
        if (diff >= maxdiff) {
          maxdiff = diff;
#ifdef VERBOSE_TEST
          ind_diff = i;
          dim_diff = d;
          max = problem.point_weights[0].template operator[]<DataType>(ind);
#endif // VERBOSE_TEST
        }
#ifdef VERBOSE_TEST
        if (abs_max <
            problem.point_weights[0].template operator[]<DataType>(ind)) {
          abs_max = problem.point_weights[0].template operator[]<DataType>(ind);
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
              << problem.mesh.numPoints(0) << std::endl;
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
        index<SOA>(problem.mesh.numPoints(0), ind_diff, PointDim, dim_diff);
    std::cout << "Values: " << result1[ind] << " / " << max << std::endl;
#endif // VERBOSE_TEST
    std::cout << "Test considered " << (maxdiff < 0.00001 ? "PASSED" : "FAILED")
              << std::endl;
  }
  /* 1}}} */
}

template <unsigned MeshDim, unsigned PointDim = 1, unsigned CellDim = 1,
          bool SOA = false, typename DataType = float>
void testPartitioning(MY_SIZE num, MY_SIZE N, MY_SIZE M) {
  /* testPartitioning {{{1 */
  std::cout << "========================================" << std::endl;
  std::cout << "Partition test" << std::endl;
  std::cout << "PointDim: " << PointDim << ", CellDim: " << CellDim;
  std::cout << ", MeshDim: " << MeshDim;
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
    Problem<SOA> problem{
        StructuredProblem<MeshDim, PointDim, CellDim, SOA, DataType>(N, M)};
    assert(problem.mesh.numPoints(0) == N * M);
    result1.resize(problem.mesh.numPoints(0) * PointDim);
    // save data before test
    #pragma omp parallel for
    for (MY_SIZE i = 0; i < problem.mesh.numPoints(0); ++i) {
      for (MY_SIZE d = 0; d < PointDim; ++d) {
        MY_SIZE ind = index<SOA>(problem.mesh.numPoints(0), i, PointDim, d);
        result1[ind] =
            problem.point_weights[0].template operator[]<DataType>(ind);
      }
    }

    // run algorithm
    problem.template loopCPUCellCentredOMP<
        mine::StepOMP<MeshDim, PointDim, CellDim, DataType>>(num);

#ifdef VERBOSE_TEST
    DataType abs_max = 0;
#endif // VERBOSE_TEST
    for (MY_SIZE i = 0; i < problem.mesh.numPoints(0); ++i) {
#ifdef VERBOSE_TEST
      MY_SIZE value_changed = PointDim;
#endif // VERBOSE_TEST
      for (MY_SIZE d = 0; d < PointDim; ++d) {
        MY_SIZE ind = index<SOA>(problem.mesh.numPoints(0), i, PointDim, d);
#ifdef VERBOSE_TEST
        if (result1[ind] ==
            problem.point_weights[0].template operator[]<DataType>(ind)) {
          if (value_changed == PointDim)
            not_changed.push_back(i);
          value_changed--;
        }
#endif // VERBOSE_TEST
        result1[ind] =
            problem.point_weights[0].template operator[]<DataType>(ind);
#ifdef VERBOSE_TEST
        if (abs_max <
            problem.point_weights[0].template operator[]<DataType>(ind)) {
          abs_max = problem.point_weights[0].template operator[]<DataType>(ind);
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
              << problem.mesh.numPoints(0) << std::endl;
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
    Problem<SOA> problem{
        StructuredProblem<MeshDim, PointDim, CellDim, SOA, DataType>(N, M)};
    result2.resize(problem.mesh.numPoints(0) * PointDim);
    // save data before test
    #pragma omp parallel for
    for (MY_SIZE i = 0; i < problem.mesh.numPoints(0); ++i) {
      for (MY_SIZE d = 0; d < PointDim; ++d) {
        MY_SIZE ind = index<SOA>(problem.mesh.numPoints(0), i, PointDim, d);
        result2[ind] =
            problem.point_weights[0].template operator[]<DataType>(ind);
      }
    }
    // Create partitioning
    problem.partition(1.01);
    problem.reorderToPartition();
    problem.renumberPoints();

    // run algorithm
    problem.template loopGPUHierarchical<
        mine::StepGPUHierarchical<MeshDim, PointDim, CellDim, DataType>>(num);

#ifdef VERBOSE_TEST
    DataType abs_max = 0;
#endif // VERBOSE_TEST

    for (MY_SIZE i = 0; i < problem.mesh.numPoints(0); ++i) {
#ifdef VERBOSE_TEST
      MY_SIZE value_changed = PointDim;
#endif // VERBOSE_TEST
      for (MY_SIZE d = 0; d < PointDim; ++d) {
        const MY_SIZE problem_ind =
            index<SOA>(problem.mesh.numPoints(0),
                       problem.applied_permutation[i], PointDim, d);
        const MY_SIZE result_ind =
            index<SOA>(problem.mesh.numPoints(0), i, PointDim, d);
#ifdef VERBOSE_TEST
        if (result2[result_ind] ==
            problem.point_weights[0].template operator[]<DataType>(
                problem_ind)) {
          if (value_changed == PointDim)
            not_changed2.push_back(i);
          value_changed--;
        }
#endif // VERBOSE_TEST
        DataType diff =
            std::abs(problem.point_weights[0].template operator[]<DataType>(
                         problem_ind) -
                     result1[result_ind]) /
            std::min(result1[result_ind],
                     problem.point_weights[0].template operator[]<DataType>(
                         problem_ind));
        if (diff >= maxdiff) {
          maxdiff = diff;
#ifdef VERBOSE_TEST
          ind_diff = i;
          dim_diff = d;
          max = problem.point_weights[0].template operator[]<DataType>(
              problem_ind);
#endif // VERBOSE_TEST
        }
#ifdef VERBOSE_TEST
        if (abs_max < problem.point_weights[0].template operator[]<DataType>(
                          problem_ind)) {
          abs_max = problem.point_weights[0].template operator[]<DataType>(
              problem_ind);
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
              << problem.mesh.numPoints(0) << std::endl;
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
        index<SOA>(problem.mesh.numPoints(0), ind_diff, PointDim, dim_diff);
    std::cout << "Values: " << result1[ind] << " / " << max << std::endl;
#endif // VERBOSE_TEST
    std::cout << "Test considered " << (maxdiff < 0.00001 ? "PASSED" : "FAILED")
              << std::endl;
  }
  /* 1}}} */
}

template <unsigned MeshDim, unsigned PointDim = 1, unsigned CellDim = 1,
          bool SOA = false, typename DataType = float>
void testReordering(MY_SIZE num, MY_SIZE N, MY_SIZE M,
                    implementation_algorithm_t<SOA> algorithm1,
                    implementation_algorithm_t<SOA> algorithm2) {
      /* testReordering {{{1 */
      std::cout << "========================================" << std::endl;
      std::cout << "Reordering test" << std::endl;
      std::cout << "PointDim: " << PointDim << ", CellDim: " << CellDim;
      std::cout << ", MeshDim: " << MeshDim << std::endl;
      std::cout << (SOA ? ", SOA" : ", AOS") << ", Precision: ";
      std::cout << (sizeof(DataType) == sizeof(float) ? "float" : "double");
      std::cout << std::endl << "Iteration: " << num << " size: " << N << ", " << M;
      std::cout << std::endl;
      std::cout << "========================================" << std::endl;
    
      std::vector<DataType> result1, result2;
      DataType maxdiff = 0;
    #ifdef VERBOSE_TEST
      std::vector<MY_SIZE> not_changed;
      MY_SIZE ind_max = 0, dim_max = 0;
      bool single_change_in_node = false;
    #endif // VERBOSE_TEST
      {
        srand(1);
        Problem<SOA> problem{
            StructuredProblem<MeshDim, PointDim, CellDim, SOA, DataType>(N, M)};
        // reorder first
        problem.reorder();
    
        result1.resize(problem.mesh.numPoints(0) * PointDim);
    // save data before test
    #pragma omp parallel for
        for (MY_SIZE i = 0; i < problem.mesh.numPoints(0); ++i) {
          for (MY_SIZE d = 0; d < PointDim; ++d) {
            MY_SIZE ind = index<SOA>(problem.mesh.numPoints(0), i, PointDim, d);
            result1[ind] =
                problem.point_weights[0].template operator[]<DataType>(ind);
          }
        }
    
        // run algorithm
        (problem.*algorithm1)(num);
    
    #ifdef VERBOSE_TEST
        DataType abs_max = 0;
    #endif // VERBOSE_TEST
        for (MY_SIZE i = 0; i < problem.mesh.numPoints(0); ++i) {
    #ifdef VERBOSE_TEST
          MY_SIZE value_changed = PointDim;
    #endif // VERBOSE_TEST
          for (MY_SIZE d = 0; d < PointDim; ++d) {
            const MY_SIZE problem_ind =
                index<SOA>(problem.mesh.numPoints(0),
                           problem.applied_permutation[i], PointDim, d);
            const MY_SIZE result_ind =
                index<SOA>(problem.mesh.numPoints(0), i, PointDim, d);
    #ifdef VERBOSE_TEST
            if (result1[result_ind] ==
                problem.point_weights[0].template operator[]<DataType>(
                    problem_ind)) {
              if (value_changed == PointDim)
                not_changed.push_back(i);
              value_changed--;
            }
    #endif // VERBOSE_TEST
            result1[result_ind] =
                problem.point_weights[0].template operator[]<DataType>(problem_ind);
    #ifdef VERBOSE_TEST
            if (abs_max < problem.point_weights[0].template operator[]<DataType>(
                              problem_ind)) {
              abs_max = problem.point_weights[0].template operator[]<DataType>(
                  problem_ind);
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
                  << problem.mesh.numPoints(0) << std::endl;
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
        Problem<SOA> problem{
            StructuredProblem<MeshDim, PointDim, CellDim, SOA, DataType>(N, M)};
        result2.resize(problem.mesh.numPoints(0) * PointDim);
    // save data before test
    #pragma omp parallel for
        for (MY_SIZE i = 0; i < problem.mesh.numPoints(0); ++i) {
          for (MY_SIZE d = 0; d < PointDim; ++d) {
            MY_SIZE ind = index<SOA>(problem.mesh.numPoints(0), i, PointDim, d);
            result2[ind] =
                problem.point_weights[0].template operator[]<DataType>(ind);
          }
        }
        // run algorithm
        (problem.*algorithm2)(num);
    
    #ifdef VERBOSE_TEST
        DataType abs_max = 0;
    #endif // VERBOSE_TEST
    
        for (MY_SIZE i = 0; i < problem.mesh.numPoints(0); ++i) {
          for (MY_SIZE d = 0; d < PointDim; ++d) {
            MY_SIZE ind = index<SOA>(problem.mesh.numPoints(0), i, PointDim, d);
            DataType diff =
                std::abs(
                    problem.point_weights[0].template operator[]<DataType>(ind) -
                    result1[ind]) /
                std::min(
                    result1[ind],
                    problem.point_weights[0].template operator[]<DataType>(ind));
            if (diff >= maxdiff) {
              maxdiff = diff;
    #ifdef VERBOSE_TEST
              ind_diff = i;
              dim_diff = d;
              max = problem.point_weights[0].template operator[]<DataType>(ind);
    #endif // VERBOSE_TEST
            }
    #ifdef VERBOSE_TEST
            if (abs_max <
                problem.point_weights[0].template operator[]<DataType>(ind)) {
              abs_max = problem.point_weights[0].template operator[]<DataType>(ind);
              ind_max = i;
              dim_max = d;
            }
    #endif // VERBOSE_TEST
          }
        }
    #ifdef VERBOSE_TEST
        std::cout << "Abs max: " << abs_max << " node: " << ind_max
                  << " dim: " << dim_max << std::endl;
        std::cout << "MAX DIFF: " << maxdiff << " node: " << ind_diff
                  << " dim: " << dim_diff << std::endl;
        MY_SIZE ind =
            index<SOA>(problem.mesh.numPoints(0), ind_diff, PointDim, dim_diff);
        std::cout << "Values: " << result1[ind] << " / " << max << std::endl;
    #endif // VERBOSE_TEST
        std::cout << "Test considered " << (maxdiff < 0.00001 ? "PASSED" : "FAILED")
                  << std::endl;
      }
      /* 1}}} */
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
      testTwoImplementations<MeshDim, TEST_DIM, TEST_CELL_DIM, false,          \
                             TEST_DATA_TYPE>(num, N, M, algorithm1,            \
                                             algorithm2);                      \
      testTwoImplementations<MeshDim, TEST_DIM, TEST_CELL_DIM, true,           \
                             TEST_DATA_TYPE>(num, N, M, algorithm1_SOA,        \
                                             algorithm2_SOA);                  \
      /* 2}}} */                                                               \
    }                                                                          \
    {                                                                          \
      constexpr unsigned TEST_DIM = 2;                                         \
      constexpr unsigned TEST_CELL_DIM = 1;                                    \
      using TEST_DATA_TYPE = float;                                            \
      /* {{{2 */                                                               \
      testTwoImplementations<MeshDim, TEST_DIM, TEST_CELL_DIM, false,          \
                             TEST_DATA_TYPE>(num, N, M, algorithm1,            \
                                             algorithm2);                      \
      testTwoImplementations<MeshDim, TEST_DIM, TEST_CELL_DIM, true,           \
                             TEST_DATA_TYPE>(num, N, M, algorithm1_SOA,        \
                                             algorithm2_SOA);                  \
      /* 2}}} */                                                               \
    }                                                                          \
    {                                                                          \
      constexpr unsigned TEST_DIM = 2;                                         \
      constexpr unsigned TEST_CELL_DIM = 2;                                    \
      using TEST_DATA_TYPE = float;                                            \
      /* {{{2 */                                                               \
      testTwoImplementations<MeshDim, TEST_DIM, TEST_CELL_DIM, false,          \
                             TEST_DATA_TYPE>(num, N, M, algorithm1,            \
                                             algorithm2);                      \
      testTwoImplementations<MeshDim, TEST_DIM, TEST_CELL_DIM, true,           \
                             TEST_DATA_TYPE>(num, N, M, algorithm1_SOA,        \
                                             algorithm2_SOA);                  \
      /* 2}}} */                                                               \
    }                                                                          \
    {                                                                          \
      constexpr unsigned TEST_DIM = 4;                                         \
      constexpr unsigned TEST_CELL_DIM = 1;                                    \
      using TEST_DATA_TYPE = float;                                            \
      /* {{{2 */                                                               \
      testTwoImplementations<MeshDim, TEST_DIM, TEST_CELL_DIM, false,          \
                             TEST_DATA_TYPE>(num, N, M, algorithm1,            \
                                             algorithm2);                      \
      testTwoImplementations<MeshDim, TEST_DIM, TEST_CELL_DIM, true,           \
                             TEST_DATA_TYPE>(num, N, M, algorithm1_SOA,        \
                                             algorithm2_SOA);                  \
      /* 2}}} */                                                               \
    }                                                                          \
    {                                                                          \
      constexpr unsigned TEST_DIM = 4;                                         \
      constexpr unsigned TEST_CELL_DIM = 4;                                    \
      using TEST_DATA_TYPE = float;                                            \
      /* {{{2 */                                                               \
      testTwoImplementations<MeshDim, TEST_DIM, TEST_CELL_DIM, false,          \
                             TEST_DATA_TYPE>(num, N, M, algorithm1,            \
                                             algorithm2);                      \
      testTwoImplementations<MeshDim, TEST_DIM, TEST_CELL_DIM, true,           \
                             TEST_DATA_TYPE>(num, N, M, algorithm1_SOA,        \
                                             algorithm2_SOA);                  \
      /* 2}}} */                                                               \
    }                                                                          \
    {                                                                          \
      constexpr unsigned TEST_DIM = 8;                                         \
      constexpr unsigned TEST_CELL_DIM = 1;                                    \
      using TEST_DATA_TYPE = float;                                            \
      /* {{{2 */                                                               \
      testTwoImplementations<MeshDim, TEST_DIM, TEST_CELL_DIM, false,          \
                             TEST_DATA_TYPE>(num, N, M, algorithm1,            \
                                             algorithm2);                      \
      testTwoImplementations<MeshDim, TEST_DIM, TEST_CELL_DIM, true,           \
                             TEST_DATA_TYPE>(num, N, M, algorithm1_SOA,        \
                                             algorithm2_SOA);                  \
      /* 2}}} */                                                               \
    }                                                                          \
    {                                                                          \
      constexpr unsigned TEST_DIM = 8;                                         \
      constexpr unsigned TEST_CELL_DIM = 8;                                    \
      using TEST_DATA_TYPE = float;                                            \
      /* {{{2 */                                                               \
      testTwoImplementations<MeshDim, TEST_DIM, TEST_CELL_DIM, false,          \
                             TEST_DATA_TYPE>(num, N, M, algorithm1,            \
                                             algorithm2);                      \
      testTwoImplementations<MeshDim, TEST_DIM, TEST_CELL_DIM, true,           \
                             TEST_DATA_TYPE>(num, N, M, algorithm1_SOA,        \
                                             algorithm2_SOA);                  \
      /* 2}}} */                                                               \
    }                                                                          \
    /* double */                                                               \
    {                                                                          \
      constexpr unsigned TEST_DIM = 1;                                         \
      constexpr unsigned TEST_CELL_DIM = 1;                                    \
      using TEST_DATA_TYPE = double;                                           \
      /* {{{2 */                                                               \
      testTwoImplementations<MeshDim, TEST_DIM, TEST_CELL_DIM, false,          \
                             TEST_DATA_TYPE>(num, N, M, algorithm1,            \
                                             algorithm2);                      \
      testTwoImplementations<MeshDim, TEST_DIM, TEST_CELL_DIM, true,           \
                             TEST_DATA_TYPE>(num, N, M, algorithm1_SOA,        \
                                             algorithm2_SOA);                  \
      /* 2}}} */                                                               \
    }                                                                          \
    {                                                                          \
      constexpr unsigned TEST_DIM = 2;                                         \
      constexpr unsigned TEST_CELL_DIM = 1;                                    \
      using TEST_DATA_TYPE = double;                                           \
      /* {{{2 */                                                               \
      testTwoImplementations<MeshDim, TEST_DIM, TEST_CELL_DIM, false,          \
                             TEST_DATA_TYPE>(num, N, M, algorithm1,            \
                                             algorithm2);                      \
      testTwoImplementations<MeshDim, TEST_DIM, TEST_CELL_DIM, true,           \
                             TEST_DATA_TYPE>(num, N, M, algorithm1_SOA,        \
                                             algorithm2_SOA);                  \
      /* 2}}} */                                                               \
    }                                                                          \
    {                                                                          \
      constexpr unsigned TEST_DIM = 2;                                         \
      constexpr unsigned TEST_CELL_DIM = 2;                                    \
      using TEST_DATA_TYPE = double;                                           \
      /* {{{2 */                                                               \
      testTwoImplementations<MeshDim, TEST_DIM, TEST_CELL_DIM, false,          \
                             TEST_DATA_TYPE>(num, N, M, algorithm1,            \
                                             algorithm2);                      \
      testTwoImplementations<MeshDim, TEST_DIM, TEST_CELL_DIM, true,           \
                             TEST_DATA_TYPE>(num, N, M, algorithm1_SOA,        \
                                             algorithm2_SOA);                  \
      /* 2}}} */                                                               \
    }                                                                          \
    {                                                                          \
      constexpr unsigned TEST_DIM = 4;                                         \
      constexpr unsigned TEST_CELL_DIM = 1;                                    \
      using TEST_DATA_TYPE = double;                                           \
      /* {{{2 */                                                               \
      testTwoImplementations<MeshDim, TEST_DIM, TEST_CELL_DIM, false,          \
                             TEST_DATA_TYPE>(num, N, M, algorithm1,            \
                                             algorithm2);                      \
      testTwoImplementations<MeshDim, TEST_DIM, TEST_CELL_DIM, true,           \
                             TEST_DATA_TYPE>(num, N, M, algorithm1_SOA,        \
                                             algorithm2_SOA);                  \
      /* 2}}} */                                                               \
    }                                                                          \
    {                                                                          \
      constexpr unsigned TEST_DIM = 4;                                         \
      constexpr unsigned TEST_CELL_DIM = 4;                                    \
      using TEST_DATA_TYPE = double;                                           \
      /* {{{2 */                                                               \
      testTwoImplementations<MeshDim, TEST_DIM, TEST_CELL_DIM, false,          \
                             TEST_DATA_TYPE>(num, N, M, algorithm1,            \
                                             algorithm2);                      \
      testTwoImplementations<MeshDim, TEST_DIM, TEST_CELL_DIM, true,           \
                             TEST_DATA_TYPE>(num, N, M, algorithm1_SOA,        \
                                             algorithm2_SOA);                  \
      /* 2}}} */                                                               \
    }                                                                          \
    {                                                                          \
      constexpr unsigned TEST_DIM = 8;                                         \
      constexpr unsigned TEST_CELL_DIM = 1;                                    \
      using TEST_DATA_TYPE = double;                                           \
      /* {{{2 */                                                               \
      testTwoImplementations<MeshDim, TEST_DIM, TEST_CELL_DIM, false,          \
                             TEST_DATA_TYPE>(num, N, M, algorithm1,            \
                                             algorithm2);                      \
      testTwoImplementations<MeshDim, TEST_DIM, TEST_CELL_DIM, true,           \
                             TEST_DATA_TYPE>(num, N, M, algorithm1_SOA,        \
                                             algorithm2_SOA);                  \
      /* 2}}} */                                                               \
    }                                                                          \
    {                                                                          \
      constexpr unsigned TEST_DIM = 8;                                         \
      constexpr unsigned TEST_CELL_DIM = 8;                                    \
      using TEST_DATA_TYPE = double;                                           \
      /* {{{2 */                                                               \
      testTwoImplementations<MeshDim, TEST_DIM, TEST_CELL_DIM, false,          \
                             TEST_DATA_TYPE>(num, N, M, algorithm1,            \
                                             algorithm2);                      \
      testTwoImplementations<MeshDim, TEST_DIM, TEST_CELL_DIM, true,           \
                             TEST_DATA_TYPE>(num, N, M, algorithm1_SOA,        \
                                             algorithm2_SOA);                  \
      /* 2}}} */                                                               \
    }                                                                          \
  } /* 1}}}*/

template <unsigned MeshDim> void testImplementations() {
  std::cout << "========================================" << std::endl;
  std::cout << "#         Sequential - OpenMP          #" << std::endl;
  TEST_TWO_IMPLEMENTATIONS(
      (&Problem<false>::loopCPUCellCentred<
          mine::StepSeq<MeshDim, TEST_DIM, TEST_CELL_DIM, TEST_DATA_TYPE>>),
      (&Problem<false>::loopCPUCellCentredOMP<
          mine::StepOMP<MeshDim, TEST_DIM, TEST_CELL_DIM, TEST_DATA_TYPE>>),
      (&Problem<true>::loopCPUCellCentred<
          mine::StepSeq<MeshDim, TEST_DIM, TEST_CELL_DIM, TEST_DATA_TYPE>>),
      (&Problem<true>::loopCPUCellCentredOMP<
          mine::StepOMP<MeshDim, TEST_DIM, TEST_CELL_DIM, TEST_DATA_TYPE>>));

  std::cout << "========================================" << std::endl;
  std::cout << "#         OpenMP - GPU Global          #" << std::endl;
  TEST_TWO_IMPLEMENTATIONS(
      (&Problem<false>::loopCPUCellCentredOMP<
          mine::StepOMP<MeshDim, TEST_DIM, TEST_CELL_DIM, TEST_DATA_TYPE>>),
      (&Problem<false>::loopGPUCellCentred<mine::StepGPUGlobal<
           MeshDim, TEST_DIM, TEST_CELL_DIM, TEST_DATA_TYPE>>),
      (&Problem<true>::loopCPUCellCentredOMP<
          mine::StepOMP<MeshDim, TEST_DIM, TEST_CELL_DIM, TEST_DATA_TYPE>>),
      (&Problem<true>::loopGPUCellCentred<mine::StepGPUGlobal<
           MeshDim, TEST_DIM, TEST_CELL_DIM, TEST_DATA_TYPE>>));

  std::cout << "========================================" << std::endl;
  std::cout << "#       OpenMP - GPU Hierarchical      #" << std::endl;
  TEST_TWO_IMPLEMENTATIONS(
      (&Problem<false>::loopCPUCellCentredOMP<
          mine::StepOMP<MeshDim, TEST_DIM, TEST_CELL_DIM, TEST_DATA_TYPE>>),
      (&Problem<false>::loopGPUHierarchical<mine::StepGPUHierarchical<
           MeshDim, TEST_DIM, TEST_CELL_DIM, TEST_DATA_TYPE>>),
      (&Problem<true>::loopCPUCellCentredOMP<
          mine::StepOMP<MeshDim, TEST_DIM, TEST_CELL_DIM, TEST_DATA_TYPE>>),
      (&Problem<true>::loopGPUHierarchical<mine::StepGPUHierarchical<
           MeshDim, TEST_DIM, TEST_CELL_DIM, TEST_DATA_TYPE>>));
}

void testImplementations() {
  testImplementations<2>();
  testImplementations<4>();
}

    /* testMultipleMapping {{{1 */
    template <bool SOA>
    void testMultipleMapping(const std::string &test_files_dir, MY_SIZE num,
                             bool reorder, bool partition,
                             implementation_algorithm_t<SOA> algorithm1,
                             implementation_algorithm_t<SOA> algorithm2) {
      // both algorithms should be of the mine2 namespace
      std::cout << "========================================" << std::endl;
      std::cout << "Multiple mapping test ";
      std::cout << (SOA ? "SOA" : "AOS");
      std::cout << std::endl << "Iteration: " << num;
      std::cout << std::endl;
      std::cout << "========================================" << std::endl;
    
      const std::string mesh0_file = test_files_dir + "mesh0";
      const std::string mesh1_file = test_files_dir + "mesh1";
      const std::string point_data0 = test_files_dir + "point_data0";
      const std::string point_data1 = test_files_dir + "point_data1";
      const std::string cell_data0 = test_files_dir + "cell_data0";
      const std::string cell_data1 = test_files_dir + "cell_data1";
      const std::string cell_data2 = test_files_dir + "cell_data2";
      constexpr unsigned MESH_DIM0 = mine2::MESH_DIM0, MESH_DIM1 = mine2::MESH_DIM1;
      constexpr unsigned POINT_DIM0 = mine2::POINT_DIM0,
                         CELL_DIM0 = mine2::CELL_DIM0,
                         POINT_DIM1 = mine2::POINT_DIM1,
                         CELL_DIM1 = mine2::CELL_DIM1, CELL_DIM2 = mine2::CELL_DIM2;
    
      std::vector<float> result1, result2;
      float maxdiff = 0;
    #ifdef VERBOSE_TEST
      std::vector<MY_SIZE> not_changed, not_changed2;
      MY_SIZE ind_max = 0, dim_max = 0;
      bool single_change_in_node = false;
    #endif // VERBOSE_TEST
      {
        srand(1);
        std::ifstream mesh0_file_stream(mesh0_file);
        std::ifstream mesh1_file_stream(mesh1_file);
        Problem<SOA> problem(
            std::vector<std::istream *>{&mesh0_file_stream, &mesh1_file_stream},
            std::vector<MY_SIZE>{MESH_DIM0, MESH_DIM1},
            std::vector<std::pair<MY_SIZE, unsigned>>{{POINT_DIM0, sizeof(float)},
                                                      {POINT_DIM1, sizeof(float)}},
            std::vector<std::pair<MY_SIZE, unsigned>>{{CELL_DIM0, sizeof(float)},
                                                      {CELL_DIM1, sizeof(float)},
                                                      {CELL_DIM2, sizeof(double)}});
        result1.resize(problem.mesh.numPoints(0) * POINT_DIM0);
        std::ifstream point_data0_stream(point_data0);
        std::ifstream point_data1_stream(point_data1);
        std::ifstream cell_data0_stream(cell_data0);
        std::ifstream cell_data1_stream(cell_data1);
        std::ifstream cell_data2_stream(cell_data2);
        problem.template readPointData<float>(point_data0_stream, 0);
        problem.template readPointData<float>(point_data1_stream, 1);
        problem.template readCellData<float>(cell_data0_stream, 0);
        problem.template readCellData<float>(cell_data1_stream, 1);
        problem.template readCellData<double>(cell_data2_stream, 2);
    // save data before test
    #pragma omp parallel for
        for (MY_SIZE i = 0; i < problem.mesh.numPoints(0); ++i) {
          for (MY_SIZE d = 0; d < POINT_DIM0; ++d) {
            MY_SIZE ind = index<SOA>(problem.mesh.numPoints(0), i, POINT_DIM0, d);
            result1[ind] = problem.point_weights[0].template operator[]<float>(ind);
          }
        }
        if (reorder) {
          problem.reorder();
          if (partition) {
            problem.partition(1.01);
            problem.reorderToPartition();
            problem.renumberPoints();
          }
        }
    
        // run algorithm
        (problem.*algorithm1)(num);
    
    #ifdef VERBOSE_TEST
        float abs_max = 0;
    #endif // VERBOSE_TEST
        for (MY_SIZE i = 0; i < problem.mesh.numPoints(0); ++i) {
    #ifdef VERBOSE_TEST
          MY_SIZE value_changed = POINT_DIM0;
    #endif // VERBOSE_TEST
          for (MY_SIZE d = 0; d < POINT_DIM0; ++d) {
            const MY_SIZE result_ind =
                index<SOA>(problem.mesh.numPoints(0), i, POINT_DIM0, d);
            const MY_SIZE problem_ind =
                reorder || partition
                    ? index<SOA>(problem.mesh.numPoints(0),
                                 problem.applied_permutation[i], POINT_DIM0, d)
                    : result_ind;
    #ifdef VERBOSE_TEST
            if (result1[result_ind] ==
                problem.point_weights[0].template operator[]<float>(problem_ind)) {
              if (value_changed == POINT_DIM0)
                not_changed.push_back(i);
              value_changed--;
            }
    #endif // VERBOSE_TEST
            result1[result_ind] =
                problem.point_weights[0].template operator[]<float>(problem_ind);
    #ifdef VERBOSE_TEST
            if (abs_max <
                problem.point_weights[0].template operator[]<float>(problem_ind)) {
              abs_max =
                  problem.point_weights[0].template operator[]<float>(problem_ind);
              ind_max = i;
              dim_max = d;
            }
    #endif // VERBOSE_TEST
          }
    #ifdef VERBOSE_TEST
          if (value_changed != POINT_DIM0 && value_changed != 0) {
            single_change_in_node = true;
          }
    #endif // VERBOSE_TEST
        }
    #ifdef VERBOSE_TEST
        std::cout << "Nodes stayed: " << not_changed.size() << "/"
                  << problem.mesh.numPoints(0) << std::endl;
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
      float max = 0;
      single_change_in_node = false;
    #endif // VERBOSE_TEST
      {
        srand(1);
        std::ifstream mesh0_file_stream(mesh0_file);
        std::ifstream mesh1_file_stream(mesh1_file);
        Problem<SOA> problem(
            std::vector<std::istream *>{&mesh0_file_stream, &mesh1_file_stream},
            {MESH_DIM0, MESH_DIM1},
            {{POINT_DIM0, sizeof(float)}, {POINT_DIM1, sizeof(float)}},
            {{CELL_DIM0, sizeof(float)},
             {CELL_DIM1, sizeof(float)},
             {CELL_DIM2, sizeof(double)}});
        result1.resize(problem.mesh.numPoints(0) * POINT_DIM0);
        std::ifstream point_data0_stream(point_data0);
        std::ifstream point_data1_stream(point_data1);
        std::ifstream cell_data0_stream(cell_data0);
        std::ifstream cell_data1_stream(cell_data1);
        std::ifstream cell_data2_stream(cell_data2);
        problem.template readPointData<float>(point_data0_stream, 0);
        problem.template readPointData<float>(point_data1_stream, 1);
        problem.template readCellData<float>(cell_data0_stream, 0);
        problem.template readCellData<float>(cell_data1_stream, 1);
        problem.template readCellData<double>(cell_data2_stream, 2);
        result2.resize(problem.mesh.numPoints(0) * POINT_DIM0);
    // save data before test
    #pragma omp parallel for
        for (MY_SIZE i = 0; i < problem.mesh.numPoints(0); ++i) {
          for (MY_SIZE d = 0; d < POINT_DIM0; ++d) {
            MY_SIZE ind = index<SOA>(problem.mesh.numPoints(0), i, POINT_DIM0, d);
            result2[ind] = problem.point_weights[0].template operator[]<float>(ind);
          }
        }
        // run algorithm
        (problem.*algorithm2)(num);
    
    #ifdef VERBOSE_TEST
        float abs_max = 0;
    #endif // VERBOSE_TEST
    
        for (MY_SIZE i = 0; i < problem.mesh.numPoints(0); ++i) {
    #ifdef VERBOSE_TEST
          MY_SIZE value_changed = POINT_DIM0;
    #endif // VERBOSE_TEST
          for (MY_SIZE d = 0; d < POINT_DIM0; ++d) {
            MY_SIZE ind = index<SOA>(problem.mesh.numPoints(0), i, POINT_DIM0, d);
    #ifdef VERBOSE_TEST
            if (result2[ind] ==
                problem.point_weights[0].template operator[]<float>(ind)) {
              if (value_changed == POINT_DIM0)
                not_changed2.push_back(i);
              value_changed--;
            }
    #endif // VERBOSE_TEST
            float diff =
                std::abs(problem.point_weights[0].template operator[]<float>(ind) -
                         result1[ind]) /
                std::min(result1[ind],
                         problem.point_weights[0].template operator[]<float>(ind));
            if (diff >= maxdiff) {
              maxdiff = diff;
    #ifdef VERBOSE_TEST
              ind_diff = i;
              dim_diff = d;
              max = problem.point_weights[0].template operator[]<float>(ind);
    #endif // VERBOSE_TEST
            }
    #ifdef VERBOSE_TEST
            if (abs_max <
                problem.point_weights[0].template operator[]<float>(ind)) {
              abs_max = problem.point_weights[0].template operator[]<float>(ind);
              ind_max = i;
              dim_max = d;
            }
    #endif // VERBOSE_TEST
          }
    #ifdef VERBOSE_TEST
          if (value_changed != POINT_DIM0 && value_changed != 0) {
            single_change_in_node = true;
          }
    #endif // VERBOSE_TEST
        }
    #ifdef VERBOSE_TEST
        std::cout << "Nodes stayed: " << not_changed2.size() << "/"
                  << problem.mesh.numPoints(0) << std::endl;
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
            index<SOA>(problem.mesh.numPoints(0), ind_diff, POINT_DIM0, dim_diff);
        std::cout << "Values: " << result1[ind] << " / " << max << std::endl;
    #endif // VERBOSE_TEST
        std::cout << "Test considered " << (maxdiff < 0.00001 ? "PASSED" : "FAILED")
                  << std::endl;
      }
    }
    /* 1}}} */
    
void testMultipleMapping(const std::string &fname, MY_SIZE num) {
  std::cout << "========================================" << std::endl;
  std::cout << "#         Sequential - OpenMP          #" << std::endl;
  testMultipleMapping<false>(
      fname, num, false, false,
      &Problem<false>::loopCPUCellCentred<mine2::StepSeq>,
      &Problem<false>::loopCPUCellCentredOMP<mine2::StepOMP>);
  testMultipleMapping<true>(
      fname, num, false, false,
      &Problem<true>::loopCPUCellCentred<mine2::StepSeq>,
      &Problem<true>::loopCPUCellCentredOMP<mine2::StepOMP>);

  std::cout << "========================================" << std::endl;
  std::cout << "#         OpenMP - GPU Global          #" << std::endl;
  testMultipleMapping<false>(
      fname, num, false, false,
      &Problem<false>::loopCPUCellCentredOMP<mine2::StepOMP>,
      &Problem<false>::loopGPUCellCentred<mine2::StepGPUGlobal>);
  testMultipleMapping<true>(
      fname, num, false, false,
      &Problem<true>::loopCPUCellCentredOMP<mine2::StepOMP>,
      &Problem<true>::loopGPUCellCentred<mine2::StepGPUGlobal>);

  std::cout << "========================================" << std::endl;
  std::cout << "#       OpenMP - GPU Hierarchical      #" << std::endl;
  testMultipleMapping<false>(
      fname, num, false, false,
      &Problem<false>::loopCPUCellCentredOMP<mine2::StepOMP>,
      &Problem<false>::loopGPUHierarchical<mine2::StepGPUHierarchical>);
  testMultipleMapping<true>(
      fname, num, false, false,
      &Problem<true>::loopCPUCellCentredOMP<mine2::StepOMP>,
      &Problem<true>::loopGPUHierarchical<mine2::StepGPUHierarchical>);

  std::cout << "========================================" << std::endl;
  std::cout << "#          Reordering (OpenMP)         #" << std::endl;
  testMultipleMapping<false>(
      fname, num, true, false,
      &Problem<false>::loopCPUCellCentredOMP<mine2::StepOMP>,
      &Problem<false>::loopCPUCellCentredOMP<mine2::StepOMP>);
  testMultipleMapping<true>(
      fname, num, true, false,
      &Problem<true>::loopCPUCellCentredOMP<mine2::StepOMP>,
      &Problem<true>::loopCPUCellCentredOMP<mine2::StepOMP>);

  std::cout << "========================================" << std::endl;
  std::cout << "#    Partitioning (GPU Hierarchical)   #" << std::endl;
  testMultipleMapping<false>(
      fname, num, true, true,
      &Problem<false>::loopGPUHierarchical<mine2::StepGPUHierarchical>,
      &Problem<false>::loopGPUHierarchical<mine2::StepGPUHierarchical>);
  testMultipleMapping<true>(
      fname, num, true, true,
      &Problem<true>::loopGPUHierarchical<mine2::StepGPUHierarchical>,
      &Problem<true>::loopGPUHierarchical<mine2::StepGPUHierarchical>);
}

#endif /* end of include guard: TESTS_HPP_HHJ8IWSK */

// vim:set et sts=2 sw=2 ts=2 fdm=marker:
