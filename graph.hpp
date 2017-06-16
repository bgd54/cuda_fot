#ifndef GRAPH_HPP_35BFQORK
#define GRAPH_HPP_35BFQORK

#include "data_t.hpp"
#include "reorder.hpp"
#include <algorithm>
#include <bitset>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <ostream>
#include <tuple>
#include <vector>

struct InvalidInputFile {
  MY_SIZE line;
};

struct Graph {
  using colourset_t = std::bitset<64>;

private:
  MY_SIZE num_points, num_edges;

public:
  data_t<MY_SIZE> edge_to_node;

  /* Initialisation {{{1 */
  Graph(MY_SIZE N, MY_SIZE M, bool block = false)
      : num_edges{((N - 1) * M + N * (M - 1))}, edge_to_node(num_edges, 2) {
    // num_edges = (N - 1) * M + N * (M - 1); // vertical + horizontal
    // num_edges = 2 * ((N - 1) * M + N * (M - 1)); // to and fro
    num_points = N * M;
    if (block) {
      fillEdgeListBlock(N, M);
    } else {
      fillEdgeList(N, M);
    }
  }

  /**
   * Constructs graph from stream.
   *
   * Format:
   *   - first line: num_points and num_edges ("\d+\s+\d+")
   *   - next num_edges line: an edge, denoted by two numbers, the start- and
   *     endpoint respectively ("\d+\s+\d+")
   * If the reading is broken for some reason, the succesfully read edges are
   * kept and num_edges is set accordingly.
   */
  Graph(std::istream &is)
      : num_points{0}, num_edges{0},
        edge_to_node((is >> num_points >> num_edges, num_edges), 2) {
    if (!is) {
      throw InvalidInputFile{i};
    }
    for (MY_SIZE i = 0; i < num_edges; ++i) {
      is >> edge_to_node[2 * i] >> edge_to_node[2 * i + 1];
      if (!is) {
        throw InvalidInputFile{i};
      }
    }
  }

  ~Graph() {}

  /**
   * Grid, unidirectional: right and down
   */
  void fillEdgeList(MY_SIZE N, MY_SIZE M) {
    MY_SIZE array_ind = 0, upper_point_ind = 0, lower_point_ind = M;
    for (MY_SIZE r = 0; r < N - 1; ++r) {
      for (MY_SIZE c = 0; c < M - 1; ++c) {
        edge_to_node[array_ind++] = lower_point_ind;
        edge_to_node[array_ind++] = upper_point_ind;
        edge_to_node[array_ind++] = upper_point_ind;
        edge_to_node[array_ind++] = ++upper_point_ind;
        ++lower_point_ind;
      }
      edge_to_node[array_ind++] = lower_point_ind++;
      edge_to_node[array_ind++] = upper_point_ind++;
    }
    for (MY_SIZE c = 0; c < M - 1; ++c) {
      edge_to_node[array_ind++] = upper_point_ind;
      edge_to_node[array_ind++] = ++upper_point_ind;
    }
  }

  /**
   * Grid, bidirectional
   */
  void fillEdgeList2(MY_SIZE N, MY_SIZE M) {
    MY_SIZE array_ind = 0, upper_point_ind = 0, lower_point_ind = M;
    for (MY_SIZE r = 0; r < N - 1; ++r) {
      for (MY_SIZE c = 0; c < M - 1; ++c) {
        // up-down
        edge_to_node[array_ind++] = lower_point_ind;
        edge_to_node[array_ind++] = upper_point_ind;
        edge_to_node[array_ind++] = upper_point_ind;
        edge_to_node[array_ind++] = lower_point_ind;
        // right-left
        edge_to_node[array_ind++] = upper_point_ind;
        edge_to_node[array_ind++] = upper_point_ind + 1;
        edge_to_node[array_ind++] = upper_point_ind + 1;
        edge_to_node[array_ind++] = upper_point_ind;
        ++lower_point_ind;
        ++upper_point_ind;
      }
      // Last up-down
      edge_to_node[array_ind++] = lower_point_ind;
      edge_to_node[array_ind++] = upper_point_ind;
      edge_to_node[array_ind++] = upper_point_ind++;
      edge_to_node[array_ind++] = lower_point_ind++;
    }
    // Last horizontal
    for (MY_SIZE c = 0; c < M - 1; ++c) {
      edge_to_node[array_ind++] = upper_point_ind;
      edge_to_node[array_ind++] = upper_point_ind + 1;
      edge_to_node[array_ind++] = upper_point_ind + 1;
      edge_to_node[array_ind++] = upper_point_ind;
      ++upper_point_ind;
    }
  }

  /**
   * Grid, hard coded block-indexing
   * good for BLOCK_SIZE = 64
   */
  void fillEdgeListBlock(MY_SIZE N, MY_SIZE M) {
    assert((N - 1) % 4 == 0);
    assert((M - 1) % 4 == 0);
    // assert((2 * (N - 1) + 2 * (M - 1)) % 64 == 0);
    MY_SIZE ind = 0;
    for (MY_SIZE i = 0; i < (N - 1) / 4; ++i) {
      for (MY_SIZE j = 0; j < (M - 1) / 4; ++j) {
        for (MY_SIZE k = 0; k <= 3; ++k) {
          for (MY_SIZE l = 0; l <= 3; ++l) {
            // Down
            edge_to_node[ind++] = (4 * i + k) * M + (4 * j + l);
            edge_to_node[ind++] = (4 * i + k + 1) * M + (4 * j + l);
            // Right
            edge_to_node[ind++] = (4 * i + k) * M + (4 * j + l);
            edge_to_node[ind++] = (4 * i + k) * M + (4 * j + l + 1);
          }
        }
      }
    }
    for (MY_SIZE i = 0; i < N - 1; ++i) {
      // Right side, edges directed downwards
      edge_to_node[ind++] = i * M + (M - 1);
      edge_to_node[ind++] = (i + 1) * M + (M - 1);
    }
    for (MY_SIZE i = 0; i < M - 1; ++i) {
      // Down side, edges directed right
      edge_to_node[ind++] = (N - 1) * M + i;
      edge_to_node[ind++] = (N - 1) * M + i + 1;
    }
    std::vector<MY_SIZE> permutation = renumberPoints();
    std::for_each(edge_to_node.begin(), edge_to_node.end(),
                  [&permutation](MY_SIZE &a) { a = permutation[a]; });
    assert(ind == 2 * numEdges());
  }

  /* 1}}} */

  std::vector<std::vector<MY_SIZE>>
  colourEdges(MY_SIZE from = 0, MY_SIZE to = static_cast<MY_SIZE>(-1)) const {
    if (to > numEdges()) {
      to = numEdges();
    }
    assert(edge_to_node.getDim() == 2);
    std::vector<std::vector<MY_SIZE>> edge_partitions;
    std::vector<colourset_t> point_colours(numPoints(), 0);
    std::vector<MY_SIZE> set_sizes(64, 0);
    colourset_t used_colours;
    for (MY_SIZE i = from; i < to; ++i) {
      colourset_t occupied_colours = point_colours[edge_to_node[2 * i + 0]] |
                                     point_colours[edge_to_node[2 * i + 1]];
      colourset_t available_colours = ~occupied_colours & used_colours;
      if (available_colours.none()) {
        used_colours <<= 1;
        used_colours.set(0);
        edge_partitions.emplace_back();
        available_colours = ~occupied_colours & used_colours;
      }
      std::uint8_t colour = getAvailableColour(available_colours, set_sizes);
      edge_partitions[colour].push_back(i);
      ++set_sizes[colour];
    }
    return edge_partitions;
  }

  MY_SIZE numEdges() const { return num_edges; }

  MY_SIZE numPoints() const { return num_points; }

  /**
   * Writes the edgelist in the following format:
   *   - the first line contains two numbers separated by spaces, `numPoints()`
   *     and `numEdges()` respectively.
   *   - the following `numEdges()` lines contain two numbers, `i` and `j`,
   *     separated by spaces, and it means that there is an edge from `i` to `j`
   */
  void writeEdgeList(std::ostream &os) const {
    os << numPoints() << " " << numEdges() << std::endl;
    for (std::size_t i = 0; i < numEdges(); ++i) {
      os << edge_to_node[2 * i] << " " << edge_to_node[2 * i + 1] << std::endl;
    }
  }

  /**
   * Reorder using Scotch.
   *
   * Also reorders the edge and point data in the arguments. These must be of
   * length `numEdges()` and `numPoints()`, respectively.
   */
  void reorder(float *edge_data = nullptr,
               data_t<float> *point_data = nullptr) {
    ScotchReorder reorder(*this);
    std::vector<SCOTCH_Num> permutation = reorder.reorder();
    // Permute points
    if (point_data) {
      std::vector<float> point_tmp(numPoints());
      for (MY_SIZE i = 0; i < numPoints(); ++i) {
        point_tmp[permutation[i]] = (*point_data)[i];
      }
      std::copy(point_tmp.begin(), point_tmp.end(), point_data->begin());
    }
    // Permute edge_to_node
    std::for_each(edge_to_node.begin(), edge_to_node.end(),
                  [&permutation](MY_SIZE &a) { a = permutation[a]; });
    if (edge_data) {
      std::vector<std::tuple<MY_SIZE, MY_SIZE, float>> edge_tmp(numEdges());
      for (MY_SIZE i = 0; i < numEdges(); ++i) {
        edge_tmp[i] = std::make_tuple(
            edge_to_node[edge_to_node.getDim() * i],
            edge_to_node[edge_to_node.getDim() * i + 1], edge_data[i]);
      }
      std::sort(edge_tmp.begin(), edge_tmp.end());
      for (MY_SIZE i = 0; i < numEdges(); ++i) {
        std::tie(edge_to_node[edge_to_node.getDim() * i],
                 edge_to_node[edge_to_node.getDim() * i + 1], edge_data[i]) =
            edge_tmp[i];
      }
    } else {
      std::vector<std::tuple<MY_SIZE, MY_SIZE>> edge_tmp(numEdges());
      for (MY_SIZE i = 0; i < numEdges(); ++i) {
        edge_tmp[i] =
            std::make_tuple(edge_to_node[edge_to_node.getDim() * i],
                            edge_to_node[edge_to_node.getDim() * i + 1]);
      }
      std::sort(edge_tmp.begin(), edge_tmp.end());
      for (MY_SIZE i = 0; i < numEdges(); ++i) {
        std::tie(edge_to_node[edge_to_node.getDim() * i],
                 edge_to_node[edge_to_node.getDim() * i + 1]) = edge_tmp[i];
      }
    }
  }

  std::vector<MY_SIZE> renumberPoints() const {
    std::vector<MY_SIZE> permutation(numPoints(), numPoints());
    MY_SIZE new_ind = 0;
    for (MY_SIZE i = 0; i < 2 * numEdges(); ++i) {
      if (permutation[edge_to_node[i]] == numPoints()) {
        permutation[edge_to_node[i]] = new_ind++;
      }
    }
    // Currently not supporting isolated points
    assert(std::all_of(
        permutation.begin(), permutation.end(),
        [&permutation](MY_SIZE a) { return a < permutation.size(); }));
    return permutation;
  }

  static MY_SIZE getAvailableColour(colourset_t available_colours,
                                    const std::vector<MY_SIZE> &set_sizes) {
    assert(set_sizes.size() > 0);
    MY_SIZE colour = set_sizes.size();
    for (MY_SIZE i = 0; i < set_sizes.size(); ++i) {
      if (available_colours[i]) {
        if (colour >= set_sizes.size() || set_sizes[colour] > set_sizes[i]) {
          colour = i;
        }
      }
    }
    assert(colour < set_sizes.size());
    return colour;
  }
};

#endif /* end of include guard: GRAPH_HPP_35BFQORK */
// vim:set et sw=2 ts=2 fdm=marker:
