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

constexpr MY_SIZE DEFAULT_BLOCK_SIZE = 128;

template <unsigned Dim = 1, bool SOA = false, typename DataType = float>
struct Problem {
  Graph graph;
  DataType *edge_weights;
  data_t<DataType, Dim> point_weights;
  const MY_SIZE block_size; // GPU block size

  /* ctor/dtor {{{1 */
  Problem(MY_SIZE N, MY_SIZE M,
          std::pair<MY_SIZE, MY_SIZE> block_dims = {0, DEFAULT_BLOCK_SIZE},
          bool use_coordinates = false)
      : Problem({N, M}, block_dims, use_coordinates) {}

  Problem(const std::vector<MY_SIZE> &grid_dim,
          std::pair<MY_SIZE, MY_SIZE> block_dims = {0, DEFAULT_BLOCK_SIZE},
          bool use_coordinates = false)
      : graph(grid_dim, block_dims, use_coordinates),
        point_weights(graph.numPoints()), block_size{
                                              calculateBlockSize(block_dims)} {
    edge_weights = (DataType *)malloc(sizeof(DataType) * graph.numEdges());
    for (MY_SIZE i = 0; i < graph.numEdges(); ++i) {
      edge_weights[i] = DataType(rand() % 10000 + 1) / 5000.0;
      edge_weights[i] *= 0.001;
    }
    reset();
  }

  Problem(std::istream &is, MY_SIZE _block_size = DEFAULT_BLOCK_SIZE)
      : graph(is), point_weights(graph.numPoints()), block_size{_block_size} {
    edge_weights = (DataType *)malloc(sizeof(DataType) * graph.numEdges());
    for (MY_SIZE i = 0; i < graph.numEdges(); ++i) {
      edge_weights[i] = DataType(rand() % 10000 + 1) / 5000.0;
      edge_weights[i] *= 0.001;
    }
    reset();
  }

  void reset() {
    for (DataType &w : point_weights) {
      w = DataType(rand() % 10000) / 5000.f;
      w *= 0.001;
    }
  }

  ~Problem() { free(edge_weights); }
  /* 1}}} */

  void loopGPUEdgeCentred(MY_SIZE num, MY_SIZE reset_every = 0);
  void loopGPUHierarchical(MY_SIZE num, MY_SIZE reset_every = 0);

  void stepCPUEdgeCentred(DataType *temp) { /*{{{*/
    for (MY_SIZE edge_ind = 0; edge_ind < graph.numEdges(); ++edge_ind) {
      MY_SIZE ind_left_base =
          graph.edge_to_node[graph.edge_to_node.dim * edge_ind];
      MY_SIZE ind_right_base =
          graph.edge_to_node[graph.edge_to_node.dim * edge_ind + 1];
      MY_SIZE w_ind_left = 0, w_ind_right = 0;
      for (MY_SIZE d = 0; d < Dim; ++d) {
        w_ind_left = index<Dim, SOA>(graph.numPoints(), ind_left_base, d);
        w_ind_right = index<Dim, SOA>(graph.numPoints(), ind_right_base, d);

        point_weights[w_ind_right] += edge_weights[edge_ind] * temp[w_ind_left];
        point_weights[w_ind_left] += edge_weights[edge_ind] * temp[w_ind_right];
      }
    }
  } /*}}}*/

  void loopCPUEdgeCentred(MY_SIZE num, MY_SIZE reset_every = 0) { /*{{{*/
    DataType *temp = (DataType *)malloc(sizeof(DataType) * graph.numPoints() *
                                        point_weights.dim);
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
        (sizeof(DataType) * (2.0 * Dim * graph.numPoints() + graph.numEdges()) +
         2.0 * sizeof(MY_SIZE) * graph.numEdges()) *
            num,
        (sizeof(DataType) * (2.0 * Dim * graph.numPoints() + graph.numEdges()) +
         2.0 * sizeof(MY_SIZE) * graph.numEdges()) *
            num);
    free(temp);
  } /*}}}*/

  void stepCPUEdgeCentredOMP(const std::vector<MY_SIZE> &inds,
                             data_t<DataType, Dim> &out) { /*{{{*/
    #pragma omp parallel for
    for (MY_SIZE i = 0; i < inds.size(); ++i) {
      MY_SIZE ind = inds[i];
      MY_SIZE ind_left_base = graph.edge_to_node[graph.edge_to_node.dim * ind];
      MY_SIZE ind_right_base =
          graph.edge_to_node[graph.edge_to_node.dim * ind + 1];

      MY_SIZE w_ind_left = 0, w_ind_right = 0;
      for (MY_SIZE d = 0; d < Dim; ++d) {
        w_ind_left = index<Dim, SOA>(graph.numPoints(), ind_left_base, d);
        w_ind_right = index<Dim, SOA>(graph.numPoints(), ind_right_base, d);

        point_weights[w_ind_right] += edge_weights[ind] * out[w_ind_left];
        point_weights[w_ind_left] += edge_weights[ind] * out[w_ind_right];
      }
    }
  } /*}}}*/

  void loopCPUEdgeCentredOMP(MY_SIZE num, MY_SIZE reset_every = 0) { /*{{{*/
    data_t<DataType, Dim> temp(point_weights.getSize());
    std::vector<std::vector<MY_SIZE>> partition = graph.colourEdges();
    MY_SIZE num_of_colours = partition.size();
    TIMER_START(t);
    for (MY_SIZE i = 0; i < num; ++i) {
      TIMER_TOGGLE(t);
      #pragma omp parallel for
      for (MY_SIZE e = 0; e < point_weights.getSize() * point_weights.dim;
           ++e) {
        temp[e] = point_weights[e];
      }
      TIMER_TOGGLE(t);
      for (MY_SIZE c = 0; c < num_of_colours; ++c) {
        stepCPUEdgeCentredOMP(partition[c], temp);
      }
      if (reset_every && i % reset_every == reset_every - 1) {
        TIMER_TOGGLE(t);
        reset();
        TIMER_TOGGLE(t);
      }
    }
    PRINT_BANDWIDTH(
        t, "loopCPUEdgeCentredOMP",
        (sizeof(DataType) * (2.0 * Dim * graph.numPoints() + graph.numEdges()) +
         2.0 * sizeof(MY_SIZE) * graph.numEdges()) *
            num,
        (sizeof(DataType) * (2.0 * Dim * graph.numPoints() + graph.numEdges()) +
         2.0 * sizeof(MY_SIZE) * graph.numEdges()) *
            num);
  } /*}}}*/

  void reorder() {
    graph.reorderScotch<DataType, Dim, SOA>(edge_weights, &point_weights);
  }

  static MY_SIZE calculateBlockSize(std::pair<MY_SIZE, MY_SIZE> block_dims) {
    if (block_dims.first == 0) {
      return block_dims.second;
    } else if (block_dims == {9, 8}) {
      return 9 * 8 * 2 * 2;
    } else {
      return block_dims.first * block_dims.second * 2;
    }
  }
};

#endif /* end of include guard: PROBLEM_HPP_CGW3IDMV */
// vim:set et sw=2 ts=2 fdm=marker:
