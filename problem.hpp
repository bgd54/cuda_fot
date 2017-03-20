#ifndef PROBLEM_HPP_CGW3IDMV
#define PROBLEM_HPP_CGW3IDMV

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "graph.hpp"
#include "timer.hpp"

struct Problem {
  Graph graph;
  float *point_weights, *edge_weights;

  /* ctor/dtor {{{1 */
  Problem(std::size_t N, std::size_t M) : graph(N, M) {
    point_weights = (float *)malloc(sizeof(float) * N * M);
    edge_weights = (float *)malloc(sizeof(float) * graph.numEdges());
    reset();
  }

  void reset() {
    for (std::size_t i = 0; i < graph.numPoints(); ++i) {
      point_weights[i] = float(rand() % 10000) / 5000.f - 1.0f;
    }
    for (std::size_t i = 0; i < graph.numEdges(); ++i) {
      edge_weights[i] = float(rand() % 10000) / 5000.f - 1.0f;
    }
  }

  ~Problem() {
    free(edge_weights);
    free(point_weights);
  }
  /* 1}}} */

  void loopGPUEdgeCentred(std::size_t num, std::size_t reset_every = 0);
  void loopGPUHierarchical(std::size_t num, std::size_t reset_every = 0);

  void stepCPUEdgeCentred(float *temp) {
    std::copy(point_weights, point_weights + graph.numPoints(), temp);
    for (std::size_t edge_ind = 0; edge_ind < graph.numEdges(); ++edge_ind) {
      temp[graph.edge_list[2 * edge_ind + 1]] +=
          edge_weights[edge_ind] * point_weights[graph.edge_list[2 * edge_ind]];
    }
    std::copy(temp, temp + graph.numPoints(), point_weights);
  }

  void loopCPUEdgeCentred(std::size_t num, std::size_t reset_every = 0) {
    float *temp = (float *)malloc(sizeof(float) * graph.numPoints());
    TIMER_START(t);
    for (std::size_t i = 0; i < num; ++i) {
      stepCPUEdgeCentred(temp);
      if (reset_every && i % reset_every == reset_every - 1) {
        reset();
      }
    }
    PRINT_BANDWIDTH(t, "loopCPUEdgeCentred",
                    (2 * graph.numPoints() + graph.numEdges()) * num);
    free(temp);
  }

  void stepCPUPointCentred(float *temp) {
    for (std::size_t point_ind = 0; point_ind < graph.numPoints();
         ++point_ind) {
      float sum = 0;
      for (std::size_t edge_ind = graph.offsets[point_ind];
           edge_ind < graph.offsets[point_ind + 1]; edge_ind += 1) {
        sum += edge_weights[graph.point_list[2 * edge_ind]] *
               point_weights[graph.point_list[2 * edge_ind + 1]];
      }
      temp[point_ind] = point_weights[point_ind] + sum;
    }
    std::copy(temp, temp + graph.numPoints(), point_weights);
  }

  void loopCPUPointCentred(std::size_t num, std::size_t reset_every = 0) {
    float *temp = (float *)malloc(sizeof(float) * graph.numPoints());
    TIMER_START(t);
    for (std::size_t i = 0; i < num; ++i) {
      stepCPUPointCentred(temp);
      if (reset_every && i % reset_every == reset_every - 1) {
        reset();
      }
    }
    PRINT_BANDWIDTH(t, "loopCPUPointCentred",
                    (2 * graph.numPoints() + graph.numEdges()) * num);
    free(temp);
  }
};

#endif /* end of include guard: PROBLEM_HPP_CGW3IDMV */
// vim:set et sw=2 ts=2 fdm=marker:
