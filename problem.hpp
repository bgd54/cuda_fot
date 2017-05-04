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
  Problem(MY_SIZE N, MY_SIZE M) : graph(N, M) {
    point_weights = (float *)malloc(sizeof(float) * N * M);
    edge_weights = (float *)malloc(sizeof(float) * graph.numEdges());
    for (MY_SIZE i = 0; i < graph.numEdges(); ++i) {
      edge_weights[i] = float(rand() % 10000) / 5000.f - 1.0f;
    }
    reset();
  }

  Problem (std::istream &is) : graph(is) {
    point_weights = (float *)malloc(sizeof(float) * graph.numPoints());
    edge_weights = (float *)malloc(sizeof(float) * graph.numEdges());
    for (MY_SIZE i = 0; i < graph.numEdges(); ++i) {
      edge_weights[i] = float(rand() % 10000) / 5000.f - 1.0f;
    }
    reset();
  }

  void reset() {
    for (MY_SIZE i = 0; i < graph.numPoints(); ++i) {
      point_weights[i] = float(rand() % 10000) / 5000.f - 1.0f;
    }
  }

  ~Problem() {
    free(edge_weights);
    free(point_weights);
  }
  /* 1}}} */

  void loopGPUEdgeCentred(MY_SIZE num, MY_SIZE reset_every = 0);
  void loopGPUHierarchical(MY_SIZE num, MY_SIZE reset_every = 0);

  void stepCPUEdgeCentred(float *temp) {
    std::copy(point_weights, point_weights + graph.numPoints(), temp);
    for (MY_SIZE edge_ind = 0; edge_ind < graph.numEdges(); ++edge_ind) {
      temp[graph.edge_list[2 * edge_ind + 1]] +=
          edge_weights[edge_ind] * point_weights[graph.edge_list[2 * edge_ind]];
    }
    std::copy(temp, temp + graph.numPoints(), point_weights);
  }

  void loopCPUEdgeCentred(MY_SIZE num, MY_SIZE reset_every = 0) {
    float *temp = (float *)malloc(sizeof(float) * graph.numPoints());
    TIMER_START(t);
    for (MY_SIZE i = 0; i < num; ++i) {
      stepCPUEdgeCentred(temp);
      if (reset_every && i % reset_every == reset_every - 1) {
        reset();
      }
    }
    PRINT_BANDWIDTH(
        t, "loopCPUEdgeCentred",
        sizeof(float) * (2 * graph.numPoints() + graph.numEdges()) * num,
        sizeof(float) * (2 * graph.numPoints() + graph.numEdges()) * num);
    free(temp);
  }

  void stepCPUPointCentred(float *temp) {
    for (MY_SIZE point_ind = 0; point_ind < graph.numPoints(); ++point_ind) {
      float sum = 0;
      for (MY_SIZE edge_ind = graph.offsets[point_ind];
           edge_ind < graph.offsets[point_ind + 1]; edge_ind += 1) {
        sum += edge_weights[graph.point_list[2 * edge_ind]] *
               point_weights[graph.point_list[2 * edge_ind + 1]];
      }
      temp[point_ind] = point_weights[point_ind] + sum;
    }
    std::copy(temp, temp + graph.numPoints(), point_weights);
  }

  void loopCPUPointCentred(MY_SIZE num, MY_SIZE reset_every = 0) {
    float *temp = (float *)malloc(sizeof(float) * graph.numPoints());
    TIMER_START(t);
    for (MY_SIZE i = 0; i < num; ++i) {
      stepCPUPointCentred(temp);
      if (reset_every && i % reset_every == reset_every - 1) {
        reset();
      }
    }
    PRINT_BANDWIDTH(
        t, "loopCPUPointCentred",
        sizeof(float) * (2 * graph.numPoints() + graph.numEdges()) * num,
        sizeof(float) * (2 * graph.numPoints() + graph.numEdges()) * num);
    free(temp);
  }

  void stepCPUEdgeCentredOMP (const std::vector<MY_SIZE> &inds, float *out) {
    #pragma omp parallel for
    for (MY_SIZE i = 0; i < inds.size(); ++i) {
      MY_SIZE ind = inds[i];
      out[graph.edge_list[2 * ind + 1]] +=
          edge_weights[ind] * point_weights[graph.edge_list[2 * ind]];
    }
  }

  void loopCPUEdgeCentredOMP (MY_SIZE num, MY_SIZE reset_every = 0) {
    float *temp = (float *)malloc(sizeof(float) * graph.numPoints());
    std::vector<std::vector<MY_SIZE>> partition = graph.colourEdges();
    MY_SIZE num_of_colours = partition.size();
    TIMER_START(t);
    #pragma omp parallel for
    for (MY_SIZE e = 0; e < graph.numPoints(); ++e) {
      temp[e] = point_weights[e];
    }
    for (MY_SIZE i = 0; i < num; ++i) {
      for (MY_SIZE c = 0; c < num_of_colours; ++c) {
        stepCPUEdgeCentredOMP(partition[c],temp);
      }
      #pragma omp parallel for
      for (MY_SIZE e = 0; e < graph.numPoints(); ++e) {
        point_weights[e] = temp[e];
      }
      TIMER_TOGGLE(t);
      if (reset_every && i % reset_every == reset_every - 1) {
        reset();
        #pragma omp parallel for
        for (MY_SIZE e = 0; e < graph.numPoints(); ++e) {
          temp[e] = point_weights[e];
        }
      }
      TIMER_TOGGLE(t);
    }
    PRINT_BANDWIDTH(t, "loopCPUEdgeCentredOMP",
        sizeof(float) * (2 * graph.numPoints() + graph.numEdges()) * num,
        sizeof(float) * (2 * graph.numPoints() + graph.numEdges()) * num
        );
    free(temp);
  }

  void reorder() { graph.reorder(edge_weights, point_weights); }
};

#endif /* end of include guard: PROBLEM_HPP_CGW3IDMV */
// vim:set et sw=2 ts=2 fdm=marker:
