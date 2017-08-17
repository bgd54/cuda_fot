#ifndef PROBLEM_HPP_CGW3IDMV
#define PROBLEM_HPP_CGW3IDMV

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "graph.hpp"
#include "partition.hpp"
#include "timer.hpp"

constexpr MY_SIZE DEFAULT_BLOCK_SIZE = 128;

template <unsigned PointDim = 1, unsigned EdgeDim = 1, bool SOA = false,
          typename DataType = float>
struct Problem {
  static_assert(
      EdgeDim == PointDim || EdgeDim == 1,
      "I know of no reason why EdgeDim should be anything but 1 or PointDim");
  Graph graph;
  data_t<DataType, EdgeDim> edge_weights;
  data_t<DataType, PointDim> point_weights;
  const MY_SIZE block_size; // GPU block size
  std::vector<MY_SIZE> partition_vector;

  /* ctor/dtor {{{1 */
  Problem(MY_SIZE N, MY_SIZE M,
          std::pair<MY_SIZE, MY_SIZE> block_dims = {0, DEFAULT_BLOCK_SIZE},
          bool use_coordinates = false)
      : Problem({N, M}, block_dims, use_coordinates) {}

  Problem(const std::vector<MY_SIZE> &grid_dim,
          std::pair<MY_SIZE, MY_SIZE> block_dims = {0, DEFAULT_BLOCK_SIZE},
          bool use_coordinates = false)
      : graph(grid_dim, block_dims, use_coordinates),
        edge_weights(graph.numEdges()),
        point_weights(graph.numPoints()), block_size{
                                              calculateBlockSize(block_dims)} {
    for (DataType &weight : edge_weights) {
      weight = DataType(rand() % 10000 + 1) / 5000.0;
      weight *= 0.001;
    }
    reset();
  }

  Problem(std::istream &graph_is, MY_SIZE _block_size = DEFAULT_BLOCK_SIZE,
          std::istream *partition_is = nullptr)
      : graph(graph_is), edge_weights(graph.numEdges()),
        point_weights(graph.numPoints()), block_size{_block_size} {
    if (partition_is != nullptr) {
      if (!(*partition_is)) {
        throw InvalidInputFile{graph.numEdges()};
      }
      partition_vector.resize(graph.numEdges());
    }
    for (MY_SIZE i = 0; i < graph.numEdges(); ++i) {
      edge_weights[i] = DataType(rand() % 10000 + 1) / 5000.0;
      edge_weights[i] *= 0.001;
      if (partition_is != nullptr) {
        *partition_is >> partition_vector[i];
        if (!(*partition_is)) {
          throw InvalidInputFile{graph.numEdges() + i};
        }
      }
    }
    reset();
  }

  void reset() {
    for (DataType &w : point_weights) {
      w = DataType(rand() % 10000) / 5000.f;
      w *= 0.001;
    }
  }

  ~Problem() {}
  /* 1}}} */

  void loopGPUEdgeCentred(MY_SIZE num, MY_SIZE reset_every = 0);
  void loopGPUHierarchical(MY_SIZE num, MY_SIZE reset_every = 0);

  void stepCPUEdgeCentred(DataType *temp) { /*{{{*/
    for (MY_SIZE edge_ind_base = 0; edge_ind_base < graph.numEdges();
         ++edge_ind_base) {
      MY_SIZE ind_left_base =
          graph.edge_to_node[graph.edge_to_node.dim * edge_ind_base];
      MY_SIZE ind_right_base =
          graph.edge_to_node[graph.edge_to_node.dim * edge_ind_base + 1];
      MY_SIZE w_ind_left = 0, w_ind_right = 0;
      for (MY_SIZE d = 0; d < PointDim; ++d) {
        w_ind_left = index<PointDim, SOA>(graph.numPoints(), ind_left_base, d);
        w_ind_right =
            index<PointDim, SOA>(graph.numPoints(), ind_right_base, d);
        MY_SIZE edge_d = EdgeDim == 1 ? 0 : d;

        MY_SIZE edge_ind =
            index<EdgeDim, true>(graph.numEdges(), edge_ind_base, edge_d);
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
    PRINT_BANDWIDTH(t, "loopCPUEdgeCentred",
                    (sizeof(DataType) * (2.0 * PointDim * graph.numPoints() +
                                         graph.numEdges()) +
                     2.0 * sizeof(MY_SIZE) * graph.numEdges()) *
                        num,
                    (sizeof(DataType) * (2.0 * PointDim * graph.numPoints() +
                                         graph.numEdges()) +
                     2.0 * sizeof(MY_SIZE) * graph.numEdges()) *
                        num);
    free(temp);
  } /*}}}*/

  void stepCPUEdgeCentredOMP(const std::vector<MY_SIZE> &inds,
                             data_t<DataType, PointDim> &out) { /*{{{*/
    #pragma omp parallel for
    for (MY_SIZE i = 0; i < inds.size(); ++i) {
      MY_SIZE ind = inds[i];
      MY_SIZE ind_left_base = graph.edge_to_node[graph.edge_to_node.dim * ind];
      MY_SIZE ind_right_base =
          graph.edge_to_node[graph.edge_to_node.dim * ind + 1];

      MY_SIZE w_ind_left = 0, w_ind_right = 0;
      for (MY_SIZE d = 0; d < PointDim; ++d) {
        w_ind_left = index<PointDim, SOA>(graph.numPoints(), ind_left_base, d);
        w_ind_right =
            index<PointDim, SOA>(graph.numPoints(), ind_right_base, d);
        MY_SIZE edge_d = EdgeDim == 1 ? 0 : d;

        MY_SIZE edge_ind = index<EdgeDim, true>(graph.numEdges(), ind, edge_d);
        point_weights[w_ind_right] += edge_weights[edge_ind] * out[w_ind_left];
        point_weights[w_ind_left] += edge_weights[edge_ind] * out[w_ind_right];
      }
    }
  } /*}}}*/

  void loopCPUEdgeCentredOMP(MY_SIZE num, MY_SIZE reset_every = 0) { /*{{{*/
    data_t<DataType, PointDim> temp(point_weights.getSize());
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
    PRINT_BANDWIDTH(t, "loopCPUEdgeCentredOMP",
                    (sizeof(DataType) * (2.0 * PointDim * graph.numPoints() +
                                         graph.numEdges()) +
                     2.0 * sizeof(MY_SIZE) * graph.numEdges()) *
                        num,
                    (sizeof(DataType) * (2.0 * PointDim * graph.numPoints() +
                                         graph.numEdges()) +
                     2.0 * sizeof(MY_SIZE) * graph.numEdges()) *
                        num);
  } /*}}}*/

  void reorder() {
    graph.reorderScotch<DataType, PointDim, EdgeDim, SOA>(&edge_weights,
                                                          &point_weights);
  }

  void partition(float tolerance, idx_t options[METIS_NOPTIONS] = NULL) {
    std::vector<idx_t> _partition_vector =
        partitionMetisEnh(graph.getLineGraph(), block_size, tolerance, options);
    partition_vector.resize(_partition_vector.size());
    std::copy(_partition_vector.begin(), _partition_vector.end(),
              partition_vector.begin());
  }

  void reorderToPartition() {
    graph.reorderToPartition<EdgeDim, DataType>(partition_vector, edge_weights);
  }

  void renumberPoints() {
    std::vector<MY_SIZE> permutation = graph.renumberPoints();
    reorderData<PointDim, SOA, DataType, MY_SIZE>(point_weights, permutation);
  }

  static MY_SIZE calculateBlockSize(std::pair<MY_SIZE, MY_SIZE> block_dims) {
    if (block_dims.first == 0) {
      return block_dims.second;
    } else if (block_dims == std::pair<MY_SIZE, MY_SIZE>{9, 8}) {
      return 9 * 8 * 2 * 2;
    } else {
      return block_dims.first * block_dims.second * 2;
    }
  }

  void writePartition(std::ostream &os) const {
    for (MY_SIZE p : partition_vector) {
      os << p << std::endl;
    }
  }
};

#endif /* end of include guard: PROBLEM_HPP_CGW3IDMV */
// vim:set et sw=2 ts=2 fdm=marker:
