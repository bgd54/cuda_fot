#ifndef GRAPH_HPP_35BFQORK
#define GRAPH_HPP_35BFQORK

#include "reorder.hpp"
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <ostream>
#include <tuple>
#include <vector>

struct Graph {
private:
  MY_SIZE num_points, num_edges;

public:
  MY_SIZE *edge_list;
  MY_SIZE *offsets = nullptr, *point_list = nullptr;

  /* Initialisation {{{1 */
  Graph(MY_SIZE N, MY_SIZE M, bool block = false) {
    // num_edges = (N - 1) * M + N * (M - 1); // vertical + horizontal
    num_edges = 2 * ((N - 1) * M + N * (M - 1)); // to and fro
    num_points = N * M;
    edge_list = (MY_SIZE *)malloc(sizeof(MY_SIZE) * 2 * numEdges());
    if (block) {
      fillEdgeListBlock(N, M);
    } else {
      fillEdgeList2(N,M);
    }

    // TODO
    // offsets = (MY_SIZE *)malloc(sizeof(MY_SIZE) * (N * M + 1));
    // point_list = (MY_SIZE *)malloc(sizeof(MY_SIZE) * 2 * numEdges());
    // fillPointList(N,M);
    offsets = point_list = nullptr;
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
  Graph(std::istream &is) /*: N(0), M(0)*/ {
    is >> num_points >> num_edges;
    edge_list = (MY_SIZE *)malloc(sizeof(MY_SIZE) * 2 * numEdges());
    for (MY_SIZE i = 0; i < num_edges; ++i) {
      if (!is) {
        num_edges = i;
        break;
      }
      is >> edge_list[2 * i] >> edge_list[2 * i + 1];
    }
  }

  ~Graph() {
    free(point_list);
    free(offsets);
    free(edge_list);
  }

  /**
   * Grid, unidirectional: right and down
   */
  void fillEdgeList(MY_SIZE N, MY_SIZE M) {
    MY_SIZE array_ind = 0, upper_point_ind = 0, lower_point_ind = M;
    for (MY_SIZE r = 0; r < N - 1; ++r) {
      for (MY_SIZE c = 0; c < M - 1; ++c) {
        edge_list[array_ind++] = lower_point_ind;
        edge_list[array_ind++] = upper_point_ind;
        edge_list[array_ind++] = upper_point_ind;
        edge_list[array_ind++] = ++upper_point_ind;
        ++lower_point_ind;
      }
      edge_list[array_ind++] = lower_point_ind++;
      edge_list[array_ind++] = upper_point_ind++;
    }
    for (MY_SIZE c = 0; c < M - 1; ++c) {
      edge_list[array_ind++] = upper_point_ind;
      edge_list[array_ind++] = ++upper_point_ind;
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
        edge_list[array_ind++] = lower_point_ind;
        edge_list[array_ind++] = upper_point_ind;
        edge_list[array_ind++] = upper_point_ind;
        edge_list[array_ind++] = lower_point_ind;
        // right-left
        edge_list[array_ind++] = upper_point_ind;
        edge_list[array_ind++] = upper_point_ind + 1;
        edge_list[array_ind++] = upper_point_ind + 1;
        edge_list[array_ind++] = upper_point_ind;
        ++lower_point_ind;
        ++upper_point_ind;
      }
      // Last up-down
      edge_list[array_ind++] = lower_point_ind;
      edge_list[array_ind++] = upper_point_ind;
      edge_list[array_ind++] = upper_point_ind++;
      edge_list[array_ind++] = lower_point_ind++;
    }
    // Last horizontal
    for (MY_SIZE c = 0; c < M - 1; ++c) {
      edge_list[array_ind++] = upper_point_ind;
      edge_list[array_ind++] = upper_point_ind + 1;
      edge_list[array_ind++] = upper_point_ind + 1;
      edge_list[array_ind++] = upper_point_ind;
      ++upper_point_ind;
    }
  }

  /**
   * Grid, hard coded block-indexing
   * good for BLOCK_SIZE = 64
   */
  void fillEdgeListBlock (MY_SIZE N, MY_SIZE M) {
    assert((N-1)%4 == 0);
    assert((M-1)%4 == 0);
    assert((2*(N-1) + 2*(M-1)) % 64 == 0);
    // TODO csucsokat atszamozni
    MY_SIZE ind = 0;
    for (MY_SIZE i = 0; i < (N-1) / 4; ++i) {
      for (MY_SIZE j = 0; j < (M-1) / 4; ++j) {
        for (MY_SIZE k = 0; k <= 3; ++k) {
          for (MY_SIZE l = 0; l <= 3; ++l) {
            // Down
            edge_list[ind++] = (4*i + k) * M + (4*j + l);
            edge_list[ind++] = (4*i + k + 1) * M + (4*j + l);
            // Right
            edge_list[ind++] = (4*i + k) * M + (4*j + l);
            edge_list[ind++] = (4*i + k) * M + (4*j + l + 1);
            // Up
            edge_list[ind++] = (4*i + k + 1) * M + (4*j + l + 1);
            edge_list[ind++] = (4*i + k) * M + (4*j + l + 1);
            // Left
            edge_list[ind++] = (4*i + k + 1) * M + (4*j + l + 1);
            edge_list[ind++] = (4*i + k + 1) * M + (4*j + l);
          }
        }
      }
    }
    for (MY_SIZE i = 0; i < N-1; ++i) {
      // Left side, edges directed upwards
      edge_list[ind++] = (i + 1) * M;
      edge_list[ind++] = i * M;
      // Right side, edges directed downwards
      edge_list[ind++] = i * M + (M - 1);
      edge_list[ind++] = (i + 1) * M + (M - 1);
      // Top side, edges directed left
      edge_list[ind++] = i + 1;
      edge_list[ind++] = i;
      // Down side, edges directed right
      edge_list[ind++] = (N - 1) * M + i;
      edge_list[ind++] = (N - 1) * M + i + 1;
    }
    assert(ind == 2*numEdges());
  }

  void fillPointList(MY_SIZE N, MY_SIZE M) {
    MY_SIZE point_ind = 0, list_ind = 0, edge_ind = 0;
    MY_SIZE prev_degree = 0;
    for (MY_SIZE r = 0; r < N - 1; ++r) {
      offsets[point_ind] = prev_degree;
      ++prev_degree;
      point_list[list_ind++] = edge_ind++;
      point_list[list_ind++] = point_ind + M;
      ++point_ind;
      for (MY_SIZE c = 0; c < M - 1; ++c) {
        offsets[point_ind] = prev_degree;
        prev_degree += 2;
        point_list[list_ind++] = edge_ind++;
        point_list[list_ind++] = point_ind - 1;
        point_list[list_ind++] = edge_ind++;
        point_list[list_ind++] = point_ind + M;
        ++point_ind;
      }
    }
    offsets[point_ind++] = prev_degree;
    for (MY_SIZE c = 0; c < M - 1; ++c) {
      offsets[point_ind] = prev_degree;
      ++prev_degree;
      point_list[list_ind++] = edge_ind++;
      point_list[list_ind++] = point_ind - 1;
      ++point_ind;
    }
    offsets[point_ind] = prev_degree; // should be end of point_list
  }
  /* 1}}} */

  std::vector<std::vector<MY_SIZE>>
  colourEdges(MY_SIZE from = 0, MY_SIZE to = static_cast<MY_SIZE>(-1)) const {
    if (to > numEdges()) {
      to = numEdges();
    }
    // First fit
    // TODO optimize so the sets have roughly equal sizes
    //      ^ do we really need that in hierarchical colouring?
    std::vector<std::vector<MY_SIZE>> edge_partitions;
    std::vector<std::uint8_t> point_colours(numPoints(), 0);
    for (MY_SIZE i = from; i < to; ++i) {
      std::uint8_t colour = point_colours[edge_list[2 * i + 1]]++;
      if (colour == edge_partitions.size()) {
        edge_partitions.push_back({i});
      } else if (colour < edge_partitions.size()) {
        edge_partitions[colour].push_back(i);
      } else {
        // Wreak havoc
        std::cerr << "Something is wrong in the first fit algorithm in line "
                  << __LINE__ << std::endl;
        std::terminate();
      }
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
      os << edge_list[2 * i] << " " << edge_list[2 * i + 1] << std::endl;
    }
  }

  /**
   * Reorder using Scotch.
   *
   * Also reorders the edge and point data in the arguments. These must be of
   * length `numEdges()` and `numPoints()`, respectively.
   */
  void reorder(float *edge_data = nullptr, float *point_data = nullptr) {
    ScotchReorder reorder(*this);
    std::vector<SCOTCH_Num> permutation = reorder.reorder();
    // Permute points
    if (point_data) {
      std::vector<float> point_tmp(numPoints());
      for (MY_SIZE i = 0; i < numPoints(); ++i) {
        point_tmp[permutation[i]] = point_data[i];
      }
      std::copy(point_tmp.begin(), point_tmp.end(), point_data);
    }
    // Permute edge_list
    std::for_each(edge_list, edge_list + numEdges() * 2,
                  [&permutation](MY_SIZE &a) { a = permutation[a]; });
    if (edge_data) {
      std::vector<std::tuple<MY_SIZE, MY_SIZE, float>> edge_tmp(numEdges());
      for (MY_SIZE i = 0; i < numEdges(); ++i) {
        edge_tmp[i] = std::make_tuple(edge_list[2 * i], edge_list[2 * i + 1],
                                      edge_data[i]);
      }
      std::sort(edge_tmp.begin(), edge_tmp.end());
      for (MY_SIZE i = 0; i < numEdges(); ++i) {
        std::tie(edge_list[2 * i], edge_list[2 * i + 1], edge_data[i]) =
            edge_tmp[i];
      }
    } else {
      std::vector<std::tuple<MY_SIZE, MY_SIZE>> edge_tmp(numEdges());
      for (MY_SIZE i = 0; i < numEdges(); ++i) {
        edge_tmp[i] = std::make_tuple(edge_list[2 * i], edge_list[2 * i + 1]);
      }
      std::sort(edge_tmp.begin(), edge_tmp.end());
      for (MY_SIZE i = 0; i < numEdges(); ++i) {
        std::tie(edge_list[2 * i], edge_list[2 * i + 1]) = edge_tmp[i];
      }
    }
  }
};

#endif /* end of include guard: GRAPH_HPP_35BFQORK */
// vim:set et sw=2 ts=2 fdm=marker:
