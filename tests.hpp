#ifndef TESTS_HPP_HHJ8IWSK
#define TESTS_HPP_HHJ8IWSK

#include "partition.hpp"
#include "problem.hpp"
#include <iostream>

template <unsigned PointDim = 1, unsigned EdgeDim = 1, bool SOA = false,
          typename DataType = float>
using implementation_algorithm_t =
    void (Problem<PointDim, EdgeDim, SOA, DataType>::*)(MY_SIZE, MY_SIZE);

template <unsigned PointDim = 1, unsigned EdgeDim = 1, bool SOA = false,
          typename DataType = float>
void testTwoImplementations(
    MY_SIZE num, MY_SIZE N, MY_SIZE M, MY_SIZE reset_every,
    implementation_algorithm_t<PointDim, EdgeDim, SOA, DataType> algorithm1,
    implementation_algorithm_t<PointDim, EdgeDim, SOA, DataType> algorithm2) {
  std::cout << "========================================" << std::endl;
  std::cout << "Two implementation test" << std::endl;
  std::cout << "PointDim: " << PointDim << ", EdgeDim: " << EdgeDim;
  std::cout << (SOA ? ", SOA" : ", AOS") << ", Precision: ";
  std::cout << (sizeof(DataType) == sizeof(float) ? "float" : "double");
  std::cout << std::endl << "Iteration: " << num << " size: " << N << ", " << M;
  std::cout << " reset: " << reset_every << std::endl;
  std::cout << "========================================" << std::endl;

  std::vector<DataType> result1, result2;
  std::vector<MY_SIZE> not_changed, not_changed2;
  DataType maxdiff = 0;
  MY_SIZE ind_max = 0, dim_max = 0;
  bool single_change_in_node = false;
  {
    srand(1);
    Problem<PointDim, EdgeDim, SOA, DataType> problem(N, M);
    result1.resize(problem.graph.numPoints() * PointDim);
    // save data before test
    #pragma omp parallel for
    for (MY_SIZE i = 0; i < problem.graph.numPoints(); ++i) {
      for (MY_SIZE d = 0; d < PointDim; ++d) {
        MY_SIZE ind = index<PointDim, SOA>(problem.graph.numPoints(), i, d);
        result1[ind] = problem.point_weights[ind];
      }
    }

    // run algorithm
    (problem.*algorithm1)(num, reset_every);

    DataType abs_max = 0;
    for (MY_SIZE i = 0; i < problem.graph.numPoints(); ++i) {
      MY_SIZE value_changed = PointDim;
      for (MY_SIZE d = 0; d < PointDim; ++d) {
        MY_SIZE ind = index<PointDim, SOA>(problem.graph.numPoints(), i, d);
        if (result1[ind] == problem.point_weights[ind]) {
          if (value_changed == PointDim)
            not_changed.push_back(i);
          value_changed--;
        }
        result1[ind] = problem.point_weights[ind];
        if (abs_max < problem.point_weights[ind]) {
          abs_max = problem.point_weights[ind];
          ind_max = i;
          dim_max = d;
        }
      }
      if (value_changed != PointDim && value_changed != 0) {
        single_change_in_node = true;
      }
    }
    std::cout << "Nodes stayed: " << not_changed.size() << "/"
              << problem.graph.numPoints() << std::endl;
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
    Problem<PointDim, EdgeDim, SOA, DataType> problem(N, M);
    result2.resize(problem.graph.numPoints() * PointDim);
    // save data before test
    #pragma omp parallel for
    for (MY_SIZE i = 0; i < problem.graph.numPoints(); ++i) {
      for (MY_SIZE d = 0; d < PointDim; ++d) {
        MY_SIZE ind = index<PointDim, SOA>(problem.graph.numPoints(), i, d);
        result2[ind] = problem.point_weights[ind];
      }
    }
    // run algorithm
    (problem.*algorithm2)(num, reset_every);
    DataType abs_max = 0;

    for (MY_SIZE i = 0; i < problem.graph.numPoints(); ++i) {
      MY_SIZE value_changed = PointDim;
      for (MY_SIZE d = 0; d < PointDim; ++d) {
        MY_SIZE ind = index<PointDim, SOA>(problem.graph.numPoints(), i, d);
        if (result2[ind] == problem.point_weights[ind]) {
          if (value_changed == PointDim)
            not_changed2.push_back(i);
          value_changed--;
        }
        DataType diff = std::abs(problem.point_weights[ind] - result1[ind]) /
                        std::min(result1[ind], problem.point_weights[ind]);
        if (diff >= maxdiff) {
          maxdiff = diff;
          ind_diff = i;
          dim_diff = d;
          max = problem.point_weights[ind];
        }
        if (abs_max < problem.point_weights[ind]) {
          abs_max = problem.point_weights[ind];
          ind_max = i;
          dim_max = d;
        }
      }
      if (value_changed != PointDim && value_changed != 0) {
        std::cout << std::endl;
        single_change_in_node = true;
      }
    }
    std::cout << "Nodes stayed: " << not_changed2.size() << "/"
              << problem.graph.numPoints() << std::endl;
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
        index<PointDim, SOA>(problem.graph.numPoints(), ind_diff, dim_diff);
    std::cout << "Values: " << result1[ind] << " / " << max << std::endl;
    std::cout << "Test considered " << (maxdiff < 0.00001 ? "PASSED" : "FAILED")
              << std::endl;
  }
}

template <unsigned PointDim = 1, unsigned EdgeDim = 1, bool SOA = false,
          typename DataType = float>
void testPartitioning(MY_SIZE num, MY_SIZE N, MY_SIZE M, MY_SIZE reset_every) {
  std::cout << "========================================" << std::endl;
  std::cout << "Partition test" << std::endl;
  std::cout << "PointDim: " << PointDim << ", EdgeDim: " << EdgeDim;
  std::cout << (SOA ? ", SOA" : ", AOS") << ", Precision: ";
  std::cout << (sizeof(DataType) == sizeof(float) ? "float" : "double");
  std::cout << std::endl << "Iteration: " << num << " size: " << N << ", " << M;
  std::cout << " reset: " << reset_every << std::endl;
  std::cout << "========================================" << std::endl;

  std::vector<DataType> result1, result2;
  std::vector<MY_SIZE> not_changed, not_changed2;
  DataType maxdiff = 0;
  MY_SIZE ind_max = 0, dim_max = 0;
  bool single_change_in_node = false;
  {
    srand(1);
    Problem<PointDim, EdgeDim, SOA, DataType> problem(N, M);
    result1.resize(problem.graph.numPoints() * PointDim);
    // save data before test
    #pragma omp parallel for
    for (MY_SIZE i = 0; i < problem.graph.numPoints(); ++i) {
      for (MY_SIZE d = 0; d < PointDim; ++d) {
        MY_SIZE ind = index<PointDim, SOA>(problem.graph.numPoints(), i, d);
        result1[ind] = problem.point_weights[ind];
      }
    }

    // run algorithm
    problem.loopCPUEdgeCentredOMP(num, reset_every);

    // Partition after
    problem.partition(1.01);
    problem.reorderToPartition();
    problem.renumberPoints();

    DataType abs_max = 0;
    for (MY_SIZE i = 0; i < problem.graph.numPoints(); ++i) {
      MY_SIZE value_changed = PointDim;
      for (MY_SIZE d = 0; d < PointDim; ++d) {
        MY_SIZE ind = index<PointDim, SOA>(problem.graph.numPoints(), i, d);
        if (result1[ind] == problem.point_weights[ind]) {
          if (value_changed == PointDim)
            not_changed.push_back(i);
          value_changed--;
        }
        result1[ind] = problem.point_weights[ind];
        if (abs_max < problem.point_weights[ind]) {
          abs_max = problem.point_weights[ind];
          ind_max = i;
          dim_max = d;
        }
      }
      if (value_changed != PointDim && value_changed != 0) {
        single_change_in_node = true;
      }
    }
    std::cout << "Nodes stayed: " << not_changed.size() << "/"
              << problem.graph.numPoints() << std::endl;
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
    Problem<PointDim, EdgeDim, SOA, DataType> problem(N, M);
    result2.resize(problem.graph.numPoints() * PointDim);
    // save data before test
    #pragma omp parallel for
    for (MY_SIZE i = 0; i < problem.graph.numPoints(); ++i) {
      for (MY_SIZE d = 0; d < PointDim; ++d) {
        MY_SIZE ind = index<PointDim, SOA>(problem.graph.numPoints(), i, d);
        result2[ind] = problem.point_weights[ind];
      }
    }
    // Create partitioning
    problem.partition(1.01);
    problem.reorderToPartition();
    problem.renumberPoints();

    // run algorithm
    problem.loopGPUHierarchical(num, reset_every);
    DataType abs_max = 0;

    for (MY_SIZE i = 0; i < problem.graph.numPoints(); ++i) {
      MY_SIZE value_changed = PointDim;
      for (MY_SIZE d = 0; d < PointDim; ++d) {
        MY_SIZE ind = index<PointDim, SOA>(problem.graph.numPoints(), i, d);
        if (result2[ind] == problem.point_weights[ind]) {
          if (value_changed == PointDim)
            not_changed2.push_back(i);
          value_changed--;
        }
        DataType diff = std::abs(problem.point_weights[ind] - result1[ind]) /
                        std::min(result1[ind], problem.point_weights[ind]);
        if (diff >= maxdiff) {
          maxdiff = diff;
          ind_diff = i;
          dim_diff = d;
          max = problem.point_weights[ind];
        }
        if (abs_max < problem.point_weights[ind]) {
          abs_max = problem.point_weights[ind];
          ind_max = i;
          dim_max = d;
        }
      }
      if (value_changed != PointDim && value_changed != 0) {
        std::cout << std::endl;
        single_change_in_node = true;
      }
    }
    std::cout << "Nodes stayed: " << not_changed2.size() << "/"
              << problem.graph.numPoints() << std::endl;
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
        index<PointDim, SOA>(problem.graph.numPoints(), ind_diff, dim_diff);
    std::cout << "Values: " << result1[ind] << " / " << max << std::endl;
    std::cout << "Test considered " << (maxdiff < 0.00001 ? "PASSED" : "FAILED")
              << std::endl;
  }
}

void testReordering(MY_SIZE num, MY_SIZE N, MY_SIZE M, MY_SIZE reset_every,
                    implementation_algorithm_t<> algorithm1,
                    implementation_algorithm_t<> algorithm2) {
  std::vector<float> result1;
  double rms = 0;
  {
    srand(1);
    Problem<> problem(N, M);
    /*std::ifstream f("test.in");*/
    /*Problem<> problem (f);*/
    std::cout << "Problem 1 created" << std::endl;
    problem.reorder();
    std::cout << "Problem 1 reordered" << std::endl;
    (problem.*algorithm1)(num, reset_every);
    float abs_max = 0;
    for (MY_SIZE i = 0; i < problem.graph.numPoints(); ++i) {
      result1.push_back(problem.point_weights[i]);
      abs_max = std::max(abs_max, std::abs(problem.point_weights[i]));
    }
    std::cout << "Abs max: " << abs_max << std::endl;
  }

  {
    srand(1);
    Problem<> problem(N, M);
    /*std::ifstream f("rotor37_mesh");*/
    /*Problem<> problem (f);*/
    std::cout << "Problem 2 created" << std::endl;
    (problem.*algorithm2)(num, reset_every);
    problem.reorder();
    float abs_max = 0;
    for (MY_SIZE i = 0; i < problem.graph.numPoints(); ++i) {
      rms += std::pow(problem.point_weights[i] - result1[i], 2);
      abs_max = std::max(abs_max, std::abs(problem.point_weights[i]));
    }
    std::cout << "Abs max: " << abs_max << std::endl;
    rms = std::pow(rms / result1.size(), 0.5);
    std::cout << "RMS: " << rms << std::endl;
  }
}

#endif /* end of include guard: TESTS_HPP_HHJ8IWSK */
