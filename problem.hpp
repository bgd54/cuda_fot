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

template <unsigned Dim = 1, bool SOA = false> struct Problem {
  Graph graph;
  float *edge_weights;
  data_t<float> point_weights;

  /* ctor/dtor {{{1 */
  Problem(MY_SIZE N, MY_SIZE M) : graph(N, M), point_weights(N * M, Dim) {
    edge_weights = (float *)malloc(sizeof(float) * graph.numEdges());
    for (MY_SIZE i = 0; i < graph.numEdges(); ++i) {
      edge_weights[i] = float(rand() % 10000) / 5000.f - 1.0f;
      edge_weights[i] *= 0.0001;
    }
    reset();
  }

  Problem(std::istream &is) : graph(is), point_weights(graph.numPoints(), Dim) {
    edge_weights = (float *)malloc(sizeof(float) * graph.numEdges());
    for (MY_SIZE i = 0; i < graph.numEdges(); ++i) {
      edge_weights[i] = float(rand() % 10000) / 5000.f - 1.0f;
      edge_weights[i] *= 0.0001;
    }
    reset();
  }

  void reset() {
    for (float &w : point_weights) {
      w = float(rand() % 10000) / 5000.f - 1.0f;
      w *= 0.0001;
    }
  }

  ~Problem() { free(edge_weights); }
  /* 1}}} */

  void loopGPUEdgeCentred(MY_SIZE num, MY_SIZE reset_every = 0);
  void loopGPUHierarchical(MY_SIZE num, MY_SIZE reset_every = 0);

  void stepCPUEdgeCentred(float *temp) {
    for (MY_SIZE edge_ind = 0; edge_ind < graph.numEdges(); ++edge_ind) {
      for (MY_SIZE d = 0; d < Dim; ++d) {
        MY_SIZE w_ind_left =
            SOA
                ? d * graph.numPoints() +
                      graph.edge_to_node[graph.edge_to_node.getDim() * edge_ind]
                : graph.edge_to_node[graph.edge_to_node.getDim() * edge_ind] *
                          Dim +
                      d;
        MY_SIZE w_ind_right =
            SOA
                ? d * graph.numPoints() +
                      graph
                          .edge_to_node[graph.edge_to_node.getDim() * edge_ind +
                                        1]
                : graph.edge_to_node[graph.edge_to_node.getDim() * edge_ind +
                                     1] *
                          Dim +
                      d;
        point_weights[w_ind_right] += edge_weights[edge_ind] * temp[w_ind_left];
        point_weights[w_ind_left] += edge_weights[edge_ind] * temp[w_ind_right];
      }
    }
  }

  void loopCPUEdgeCentred(MY_SIZE num, MY_SIZE reset_every = 0) {
    float *temp = (float *)malloc(sizeof(float) * graph.numPoints() *
                                  point_weights.getDim());
    TIMER_START(t);
    for (MY_SIZE i = 0; i < num; ++i) {
      TIMER_TOGGLE(t);
      std::copy(point_weights.begin(), point_weights.end(), temp);
      TIMER_TOGGLE(t);
      stepCPUEdgeCentred(temp);
      if (reset_every && i % reset_every == reset_every - 1) {
        TIMER_TOGGLE(t);
        reset();
        TIMER_TOGGLE(t);
      }
    }
    PRINT_BANDWIDTH(
        t, "loopCPUEdgeCentred",
        sizeof(float) * (2.0 * Dim * graph.numPoints() + graph.numEdges()) *
            num,
        sizeof(float) * (2.0 * Dim * graph.numPoints() + graph.numEdges()) *
            num);
    free(temp);
  }

  void stepCPUEdgeCentredOMP(const std::vector<MY_SIZE> &inds,
                             data_t<float> &out) {
    #pragma omp parallel for collapse(2)
    for (MY_SIZE i = 0; i < inds.size(); ++i) {
      for (MY_SIZE d = 0; d < Dim; ++d) {
        MY_SIZE ind = inds[i];
        MY_SIZE w_ind_left =
            SOA
                ? d * graph.numPoints() +
                      graph.edge_to_node[graph.edge_to_node.getDim() * ind]
                : graph.edge_to_node[graph.edge_to_node.getDim() * ind] * Dim +
                      d;
        MY_SIZE w_ind_right =
            SOA
                ? d * graph.numPoints() +
                      graph.edge_to_node[graph.edge_to_node.getDim() * ind + 1]
                : graph.edge_to_node[graph.edge_to_node.getDim() * ind + 1] *
                          Dim +
                      d;
        out[w_ind_right] += edge_weights[ind] * point_weights[w_ind_left];
        out[w_ind_left] += edge_weights[ind] * point_weights[w_ind_right];
      }
    }
  }

  void loopCPUEdgeCentredOMP(MY_SIZE num, MY_SIZE reset_every = 0) {
    data_t<float> temp(point_weights.getSize(), point_weights.getDim());
    std::vector<std::vector<MY_SIZE>> partition = graph.colourEdges();
    MY_SIZE num_of_colours = partition.size();
    #pragma omp parallel for
    for (MY_SIZE e = 0; e < point_weights.getSize() * point_weights.getDim();
         ++e) {
      temp[e] = point_weights[e];
    }
    TIMER_START(t);
    for (MY_SIZE i = 0; i < num; ++i) {
      for (MY_SIZE c = 0; c < num_of_colours; ++c) {
        stepCPUEdgeCentredOMP(partition[c], temp);
      }
      TIMER_TOGGLE(t);
      #pragma omp parallel for
      for (MY_SIZE e = 0; e < point_weights.getSize() * point_weights.getDim();
           ++e) {
        point_weights[e] = temp[e];
      }
      if (reset_every && i % reset_every == reset_every - 1) {
        reset();
        #pragma omp parallel for
        for (MY_SIZE e = 0;
             e < point_weights.getSize() * point_weights.getDim(); ++e) {
          temp[e] = point_weights[e];
        }
      }
      TIMER_TOGGLE(t);
    }
    PRINT_BANDWIDTH(
        t, "loopCPUEdgeCentredOMP",
        sizeof(float) * (2.0 * Dim * graph.numPoints() + graph.numEdges()) *
            num,
        sizeof(float) * (2.0 * Dim * graph.numPoints() + graph.numEdges()) *
            num);
  }

  void reorder() { graph.reorder(edge_weights, &point_weights); }
};

#endif /* end of include guard: PROBLEM_HPP_CGW3IDMV */
// vim:set et sw=2 ts=2 fdm=marker:
