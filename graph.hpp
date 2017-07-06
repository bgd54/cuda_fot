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
  data_t<MY_SIZE, 2> edge_to_node;
  data_t<float, 3> point_coordinates;

  /* Initialisation {{{1 */
private:
  Graph(MY_SIZE _num_points, MY_SIZE _num_edges, const MY_SIZE *_edge_to_node,
        const float *_point_coordinates = nullptr)
      : num_points{_num_points}, num_edges{_num_edges}, edge_to_node(num_edges),
        point_coordinates(num_points) {
    std::copy(_edge_to_node, _edge_to_node + 2 * num_edges,
              edge_to_node.begin());
    if (_point_coordinates) {
      std::copy(_point_coordinates,
                _point_coordinates + point_coordinates.dim * num_points,
                point_coordinates.begin());
    }
  }

public:
  Graph(MY_SIZE N, MY_SIZE M, std::pair<MY_SIZE, MY_SIZE> block_sizes = {0, 0},
        bool use_coordinates = false)
      : num_points{N * M}, num_edges{((N - 1) * M + N * (M - 1))},
        edge_to_node(num_edges),
        point_coordinates(use_coordinates ? num_points : 0) {
    // num_edges = (N - 1) * M + N * (M - 1); // vertical + horizontal
    // num_edges = 2 * ((N - 1) * M + N * (M - 1)); // to and fro
    num_points = N * M;
    if (block_sizes.first != 0) {
      fillEdgeListBlock(N, M, block_sizes.first, block_sizes.second);
    } else {
      fillEdgeList(N, M);
    }
    if (use_coordinates) {
      // TODO
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
        edge_to_node((is >> num_points >> num_edges, num_edges)),
        point_coordinates() {
    if (!is) {
      throw InvalidInputFile{0};
    }
    for (MY_SIZE i = 0; i < num_edges; ++i) {
      is >> edge_to_node[2 * i] >> edge_to_node[2 * i + 1];
      if (!is) {
        throw InvalidInputFile{i};
      }
    }
  }

  ~Graph() {}

  Graph(const Graph &) = delete;
  Graph &operator=(const Graph &) = delete;

  Graph(Graph &&other)
      : num_points{other.num_points}, num_edges{other.num_edges},
        edge_to_node{std::move(other.edge_to_node)} {
    other.num_points = 0;
    other.num_edges = 0;
  }

  Graph &operator=(Graph &&rhs) {
    std::swap(num_points, rhs.num_points);
    std::swap(num_edges, rhs.num_edges);
    std::swap(edge_to_node, rhs.edge_to_node);
    return *this;
  }

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
    if (point_coordinates.getSize() > 0) {
      for (MY_SIZE r = 0; r < N; ++r) {
        for (MY_SIZE c = 0; c < M; ++c) {
          MY_SIZE point_ind = r*M + c;
          point_coordinates[point_ind * 3 + 0] = r;
          point_coordinates[point_ind * 3 + 1] = c;
          point_coordinates[point_ind * 3 + 2] = 0;
        }
      }
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
    if (point_coordinates.getSize() > 0) {
      for (MY_SIZE r = 0; r < N; ++r) {
        for (MY_SIZE c = 0; c < M; ++c) {
          MY_SIZE point_ind = r*M + c;
          point_coordinates[point_ind * 3 + 0] = r;
          point_coordinates[point_ind * 3 + 1] = c;
          point_coordinates[point_ind * 3 + 2] = 0;
        }
      }
    }
  }

  /**
   * Grid, hard coded block-indexing
   */
  void fillEdgeListBlock(MY_SIZE N, MY_SIZE M, MY_SIZE block_h,
                         MY_SIZE block_w) {
    assert((N - 1) % block_h == 0);
    assert((M - 1) % block_w == 0);
    MY_SIZE ind = 0;
    for (MY_SIZE i = 0; i < (N - 1) / block_h; ++i) {
      for (MY_SIZE j = 0; j < (M - 1) / block_w; ++j) {
        for (MY_SIZE k = 0; k < block_h; ++k) {
          for (MY_SIZE l = 0; l < block_w; ++l) {
            // Down
            edge_to_node[ind++] = (block_h * i + k) * M + (block_w * j + l);
            edge_to_node[ind++] = (block_h * i + k + 1) * M + (block_w * j + l);
            // Right
            edge_to_node[ind++] = (block_h * i + k) * M + (block_w * j + l);
            edge_to_node[ind++] = (block_h * i + k) * M + (block_w * j + l + 1);
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
    assert(ind == 2 * numEdges());
    if (point_coordinates.getSize() > 0) {
      for (MY_SIZE r = 0; r < N; ++r) {
        for (MY_SIZE c = 0; c < M; ++c) {
          MY_SIZE point_ind = r*M + c;
          point_coordinates[point_ind * 3 + 0] = r;
          point_coordinates[point_ind * 3 + 1] = c;
          point_coordinates[point_ind * 3 + 2] = 0;
        }
      }
    }
    std::vector<MY_SIZE> permutation = renumberPoints();
    std::for_each(edge_to_node.begin(), edge_to_node.end(),
                  [&permutation](MY_SIZE &a) { a = permutation[a]; });
    if (point_coordinates.getSize() > 0) {
      reorderData<3,false,float,MY_SIZE>(point_coordinates, permutation);
    }
  }

  /* 1}}} */

  template <bool VTK = false>
  typename choose_t<VTK, std::vector<std::uint16_t>,
                    std::vector<std::vector<MY_SIZE>>>::type
  colourEdges(MY_SIZE from = 0, MY_SIZE to = static_cast<MY_SIZE>(-1)) const {
    if (to > numEdges()) {
      to = numEdges();
    }
    std::vector<std::vector<MY_SIZE>> edge_partitions;
    std::vector<colourset_t> point_colours(numPoints(), 0);
    std::vector<MY_SIZE> set_sizes(64, 0);
    std::vector<std::uint16_t> edge_colours(numEdges());
    colourset_t used_colours;
    for (MY_SIZE i = from; i < to; ++i) {
      colourset_t occupied_colours = point_colours[edge_to_node[2 * i + 0]] |
                                     point_colours[edge_to_node[2 * i + 1]];
      colourset_t available_colours = ~occupied_colours & used_colours;
      if (available_colours.none()) {
        used_colours <<= 1;
        used_colours.set(0);
        if (!VTK) {
          edge_partitions.emplace_back();
        }
        available_colours = ~occupied_colours & used_colours;
      }
      std::uint8_t colour = getAvailableColour(available_colours, set_sizes);
      if (VTK) {
        edge_colours[i] = colour;
      } else {
        edge_partitions[colour].push_back(i);
      }
      colourset_t colourset(1ull << colour);
      point_colours[edge_to_node[2 * i + 0]] |= colourset;
      point_colours[edge_to_node[2 * i + 1]] |= colourset;
      ++set_sizes[colour];
    }
    return choose_t<
        VTK, std::vector<std::uint16_t>,
        std::vector<std::vector<MY_SIZE>>>::ret_value(std::move(edge_colours),
                                                      std::move(
                                                          edge_partitions));
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

  template <typename DataType = float, unsigned DataDim = 0, bool SOA = false>
  void reorderScotch(DataType *edge_data = nullptr,
                     data_t<DataType, DataDim> *point_data = nullptr) {
    ScotchReorder reorder(*this);
    std::vector<SCOTCH_Num> permutation = reorder.reorder();
    this->template reorder<SCOTCH_Num, DataType, DataDim, SOA>(
        permutation, edge_data, point_data);
  }

  /**
   * Reorders the graph using the point permutation vector.
   *
   * Also reorders the edge and point data in the arguments. These must be of
   * length `numEdges()` and `numPoints()`, respectively.
   */
  template <typename UnsignedType, typename DataType = float,
            unsigned DataDim = 0, bool SOA = false>
  void reorder(const std::vector<UnsignedType> &point_permutation,
               DataType *edge_data = nullptr,
               data_t<DataType, DataDim> *point_data = nullptr) {
    // Permute points
    if (point_data) {
      reorderData<DataDim, SOA, DataType, UnsignedType>(*point_data,
                                                        point_permutation);
    }
    if (point_coordinates.getSize() > 0) {
      reorderData<3, false, float, UnsignedType>(point_coordinates,
          point_permutation);
    }
    // Permute edge_to_node
    std::for_each(
        edge_to_node.begin(), edge_to_node.end(),
        [&point_permutation](MY_SIZE &a) { a = point_permutation[a]; });
    if (edge_data) {
      std::vector<std::tuple<MY_SIZE, MY_SIZE, DataType>> edge_tmp(numEdges());
      for (MY_SIZE i = 0; i < numEdges(); ++i) {
        edge_tmp[i] = std::make_tuple(edge_to_node[2 * i],
                                      edge_to_node[2 * i + 1], edge_data[i]);
      }
      std::sort(edge_tmp.begin(), edge_tmp.end());
      for (MY_SIZE i = 0; i < numEdges(); ++i) {
        std::tie(edge_to_node[2 * i], edge_to_node[2 * i + 1], edge_data[i]) =
            edge_tmp[i];
      }
    } else {
      std::vector<std::tuple<MY_SIZE, MY_SIZE>> edge_tmp(numEdges());
      for (MY_SIZE i = 0; i < numEdges(); ++i) {
        edge_tmp[i] =
            std::make_tuple(edge_to_node[2 * i], edge_to_node[2 * i + 1]);
      }
      std::sort(edge_tmp.begin(), edge_tmp.end());
      for (MY_SIZE i = 0; i < numEdges(); ++i) {
        std::tie(edge_to_node[2 * i], edge_to_node[2 * i + 1]) = edge_tmp[i];
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

  Graph getLineGraph() const {
    const std::multimap<MY_SIZE, MY_SIZE> point_to_edge =
        GraphCSR<MY_SIZE>::getPointToEdge(edge_to_node);
    // TODO optimise
    std::vector<MY_SIZE> new_edge_to_point;
    for (MY_SIZE i = 0; i < numEdges(); ++i) {
      for (MY_SIZE offset = 0; offset < 2; ++offset) {
        MY_SIZE point = edge_to_node[2 * i + offset];
        const auto edge_range = point_to_edge.equal_range(point);
        for (auto it = edge_range.first; it != edge_range.second; ++it) {
          MY_SIZE other_edge = it->second;
          if (other_edge > i) {
            new_edge_to_point.push_back(i);
            new_edge_to_point.push_back(other_edge);
          }
        }
      }
    }
    return Graph(numEdges(), new_edge_to_point.size() / 2,
                 new_edge_to_point.data());
  }
};

#endif /* end of include guard: GRAPH_HPP_35BFQORK */
// vim:set et sw=2 ts=2 fdm=marker:
