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
#include <set>
#include <tuple>
#include <vector>

struct InvalidInputFile {
  std::string input_type;
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
  static MY_SIZE calcNumPoints(const std::vector<MY_SIZE> &grid_dim) {
    return grid_dim[0] * grid_dim[1] * (grid_dim.size() == 3 ? grid_dim[2] : 1);
  }
  static MY_SIZE calcNumEdges(const std::vector<MY_SIZE> &grid_dim) {
    MY_SIZE N = grid_dim[0];
    MY_SIZE M = grid_dim[1];
    return ((N - 1) * M + N * (M - 1)) *
               (grid_dim.size() == 3 ? grid_dim[2] : 1) +
           (grid_dim.size() == 3 ? (grid_dim[2] - 1) * N * M : 0);
  }

public:
  Graph(const std::vector<MY_SIZE> &grid_dim,
        std::pair<MY_SIZE, MY_SIZE> block_sizes = {0, 0},
        bool use_coordinates = false)
      : num_points{calcNumPoints(grid_dim)}, num_edges{calcNumEdges(grid_dim)},
        edge_to_node(num_edges),
        point_coordinates(use_coordinates ? num_points : 0) {
    assert(grid_dim.size() >= 2);
    // num_edges = (N - 1) * M + N * (M - 1); // vertical + horizontal
    // num_edges = 2 * ((N - 1) * M + N * (M - 1)); // to and fro
    MY_SIZE N = grid_dim[0];
    MY_SIZE M = grid_dim[1];
    if (grid_dim.size() == 2) {
      num_points = N * M;
      if (block_sizes.first != 0) {
        fillEdgeListBlock(N, M, block_sizes.first, block_sizes.second);
      } else {
        fillEdgeList(N, M);
      }
    } else {
      num_points = N * M * grid_dim[2];
      fillEdgeList3D(N, M, grid_dim[2]);
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
  Graph(std::istream &is, std::istream *coord_is = nullptr)
      : num_points{0}, num_edges{0},
        edge_to_node((is >> num_points >> num_edges, num_edges)),
        point_coordinates(coord_is == nullptr ? 0 : num_points) {
    if (!is) {
      throw InvalidInputFile{"graph input", 0};
    }
    if (coord_is != nullptr) {
      if (!(*coord_is)) {
        throw InvalidInputFile{"coordinate input", 0};
      }
    }
    for (MY_SIZE i = 0; i < num_edges; ++i) {
      is >> edge_to_node[2 * i] >> edge_to_node[2 * i + 1];
      if (!is) {
        throw InvalidInputFile{"graph input", i};
      }
    }
    if (coord_is != nullptr) {
      for (MY_SIZE i = 0; i < numPoints(); ++i) {
        *coord_is >> point_coordinates[3 * i + 0] >>
            point_coordinates[3 * i + 1] >> point_coordinates[3 * i + 2];
        if (!(*coord_is)) {
          throw InvalidInputFile{"coordinate input", i};
        }
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
        edge_to_node[array_ind++] = upper_point_ind;
        edge_to_node[array_ind++] = lower_point_ind;
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
          MY_SIZE point_ind = r * M + c;
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
          MY_SIZE point_ind = r * M + c;
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
    if (block_h == 9 && block_w == 8) {
      fillEdgeList9x8(N, M);
      return;
    }
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
          MY_SIZE point_ind = r * M + c;
          point_coordinates[point_ind * 3 + 0] = r;
          point_coordinates[point_ind * 3 + 1] = c;
          point_coordinates[point_ind * 3 + 2] = 0;
        }
      }
    }
    renumberPoints(getPointRenumberingPermutation());
  }

  void fillEdgeList3D(MY_SIZE N1, MY_SIZE N2, MY_SIZE N3) {
    MY_SIZE array_ind = 0, upper_point_ind = 0;
    MY_SIZE lower_point_ind = N2, inner_point_ind = N1 * N2;
    for (MY_SIZE l = 0; l < N3 - 1; ++l) {
      upper_point_ind = l * N1 * N2;
      lower_point_ind = upper_point_ind + N2;
      inner_point_ind = (l + 1) * N1 * N2;
      for (MY_SIZE r = 0; r < N1 - 1; ++r) {
        for (MY_SIZE c = 0; c < N2 - 1; ++c) {
          // Down
          edge_to_node[array_ind++] = lower_point_ind;
          edge_to_node[array_ind++] = upper_point_ind;
          // Deep
          edge_to_node[array_ind++] = upper_point_ind;
          edge_to_node[array_ind++] = inner_point_ind;
          // Left
          edge_to_node[array_ind++] = upper_point_ind;
          edge_to_node[array_ind++] = ++upper_point_ind;
          ++lower_point_ind;
          ++inner_point_ind;
        }
        // Left end
        edge_to_node[array_ind++] = lower_point_ind++;
        edge_to_node[array_ind++] = upper_point_ind;
        edge_to_node[array_ind++] = inner_point_ind++;
        edge_to_node[array_ind++] = upper_point_ind++;
      }
      // Down end
      for (MY_SIZE c = 0; c < N2 - 1; ++c) {
        edge_to_node[array_ind++] = upper_point_ind;
        edge_to_node[array_ind++] = upper_point_ind + 1;
        edge_to_node[array_ind++] = inner_point_ind++;
        edge_to_node[array_ind++] = upper_point_ind++;
      }
      // Down last element
      edge_to_node[array_ind++] = inner_point_ind++;
      edge_to_node[array_ind++] = upper_point_ind++;
    }

    // Last layer
    upper_point_ind = (N3 - 1) * N1 * N2;
    lower_point_ind = upper_point_ind + N2;
    for (MY_SIZE r = 0; r < N1 - 1; ++r) {
      for (MY_SIZE c = 0; c < N2 - 1; ++c) {
        // Down
        edge_to_node[array_ind++] = lower_point_ind++;
        edge_to_node[array_ind++] = upper_point_ind;
        // Left
        edge_to_node[array_ind++] = upper_point_ind;
        edge_to_node[array_ind++] = ++upper_point_ind;
      }
      // Left end
      edge_to_node[array_ind++] = lower_point_ind++;
      edge_to_node[array_ind++] = upper_point_ind++;
    }
    // Last layer Down end
    for (MY_SIZE c = 0; c < N2 - 1; ++c) {
      edge_to_node[array_ind++] = upper_point_ind;
      edge_to_node[array_ind++] = ++upper_point_ind;
    }

    // generate coordinates
    if (point_coordinates.getSize() > 0) {
      for (MY_SIZE l = 0; l < N3; ++l) {
        for (MY_SIZE r = 0; r < N1; ++r) {
          for (MY_SIZE c = 0; c < N2; ++c) {
            MY_SIZE point_ind = l * N1 * N2 + r * N2 + c;
            point_coordinates[point_ind * 3 + 0] = r;
            point_coordinates[point_ind * 3 + 1] = c;
            point_coordinates[point_ind * 3 + 2] = l;
          }
        }
      }
    }
  }

  void fillEdgeList9x8(MY_SIZE N, MY_SIZE M) {
    assert((N - 1) % 9 == 0);
    assert((M - 1) % 8 == 0);
    constexpr MY_SIZE block_h = 9, block_w = 8;
    // clang-format off
    std::array<std::array<MY_SIZE,block_w + 1>,block_h + 1> pattern = {{
      {{ 0, 1, 2, 3, 4, 5, 6, 7, 0}},
      {{ 8,16,24,32,40,48,56,64, 0}},
      {{ 9,17,25,33,41,49,57,65, 0}},
      {{10,18,26,34,42,50,58,66, 0}},
      {{11,19,27,35,43,51,59,67, 0}},
      {{12,20,28,36,44,52,60,68, 0}},
      {{13,21,29,37,45,53,61,69, 0}},
      {{14,22,30,38,46,54,62,70, 0}},
      {{15,23,31,39,47,55,63,71, 0}},
      {{ 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    }};
    // clang-format on
    MY_SIZE ind = 0, block_ind_offset = 0;
    for (MY_SIZE i = 0; i < (N - 1) / block_h; ++i) {
      for (MY_SIZE j = 0; j < (M - 1) / block_w; ++j) {
        for (MY_SIZE k = 0; k < block_w; ++k) {
          if (i != (N - 1) / block_h - 1) {
            pattern[block_h][k] = pattern[0][k] + block_h * (M - 1);
          } else {
            pattern[block_h][k] =
                (N - 1) * (M - 1) + j * block_w + k - block_ind_offset;
          }
        }
        for (MY_SIZE k = 0; k < block_h; ++k) {
          if (j != (M - 1) / block_w - 1) {
            pattern[k][block_w] = pattern[k][0] + block_h * block_w;
          } else {
            pattern[k][block_w] =
                N * M - 1 - i * block_h - k - block_ind_offset;
          }
        }
        for (MY_SIZE k = 0; k < block_h; ++k) {
          for (MY_SIZE l = 0; l < block_w; ++l) {
            const MY_SIZE point_cur = block_ind_offset + pattern[k][l];
            const MY_SIZE point_down = block_ind_offset + pattern[k + 1][l];
            const MY_SIZE point_right = block_ind_offset + pattern[k][l + 1];
            // Down
            edge_to_node[ind++] = point_cur;
            edge_to_node[ind++] = point_down;
            // Right
            edge_to_node[ind++] = point_cur;
            edge_to_node[ind++] = point_right;

            if (point_coordinates.getSize() > 0) {
              point_coordinates[point_cur * 3 + 0] = i * block_h + k;
              point_coordinates[point_cur * 3 + 1] = j * block_w + l;
              point_coordinates[point_cur * 3 + 2] = 0;
            }
          }
        }
        block_ind_offset += block_h * block_w;
      }
    }
    // edges along the edges of the grid
    MY_SIZE point_cur = (N - 1) * (M - 1);
    for (MY_SIZE i = 0; i < M - 1; ++i) {
      edge_to_node[ind++] = point_cur;
      edge_to_node[ind++] = point_cur + 1;

      if (point_coordinates.getSize() > 0) {
        point_coordinates[point_cur * 3 + 0] = N - 1;
        point_coordinates[point_cur * 3 + 1] = i;
        point_coordinates[point_cur * 3 + 2] = 0;
      }
      ++point_cur;
    }
    for (MY_SIZE i = 0; i < N - 1; ++i) {
      edge_to_node[ind++] = point_cur;
      edge_to_node[ind++] = point_cur + 1;

      if (point_coordinates.getSize() > 0) {
        point_coordinates[point_cur * 3 + 0] = N - 1 - i;
        point_coordinates[point_cur * 3 + 1] = M - 1;
        point_coordinates[point_cur * 3 + 2] = 0;
      }
      ++point_cur;
    }
    if (point_coordinates.getSize() > 0) {
      point_coordinates[point_cur * 3 + 0] = 0;
      point_coordinates[point_cur * 3 + 1] = M - 1;
      point_coordinates[point_cur * 3 + 2] = 0;
    }
    assert(ind == 2 * numEdges());
    assert(point_cur + 1 == numPoints());
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

  void writeCoordinates(std::ostream &os) const {
    assert(point_coordinates.getSize() == numPoints());
    for (MY_SIZE i = 0; i < numPoints(); ++i) {
      os << point_coordinates[3 * i + 0] << " " << point_coordinates[3 * i + 1]
         << " " << point_coordinates[3 * i + 2] << std::endl;
    }
  }

  template <typename DataType = float, unsigned DataDim = 1,
            unsigned EdgeDim = 1, bool SOA = false>
  void reorderScotch(data_t<DataType, EdgeDim> *edge_data = nullptr,
                     data_t<DataType, DataDim> *point_data = nullptr) {
    ScotchReorder reorder(*this);
    std::vector<SCOTCH_Num> permutation = reorder.reorder();
    this->template reorder<SCOTCH_Num, DataType, DataDim, EdgeDim, SOA>(
        permutation, edge_data, point_data);
  }

  /**
   * Reorders the graph using the point permutation vector.
   *
   * Also reorders the edge and point data in the arguments. These must be of
   * length `numEdges()` and `numPoints()`, respectively.
   */
  template <typename UnsignedType, typename DataType = float,
            unsigned DataDim = 1, unsigned EdgeDim = 1, bool SOA = false>
  void reorder(const std::vector<UnsignedType> &point_permutation,
               data_t<DataType, EdgeDim> *edge_data = nullptr,
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
    for (MY_SIZE i = 0; i < numEdges(); ++i) {
      edge_to_node[2 * i] = point_permutation[edge_to_node[2 * i]];
      edge_to_node[2 * i + 1] = point_permutation[edge_to_node[2 * i + 1]];
      if (edge_to_node[2 * i] > edge_to_node[2 * i + 1]) {
        std::swap(edge_to_node[2 * i], edge_to_node[2 * i + 1]);
      }
    }
    if (edge_data) {
      std::vector<std::tuple<MY_SIZE, MY_SIZE, MY_SIZE>> edge_tmp(numEdges());
      for (MY_SIZE i = 0; i < numEdges(); ++i) {
        edge_tmp[i] =
            std::make_tuple(edge_to_node[2 * i], edge_to_node[2 * i + 1], i);
      }
      std::sort(edge_tmp.begin(), edge_tmp.end());
      std::vector<MY_SIZE> inv_permutation(numEdges());
      for (MY_SIZE i = 0; i < numEdges(); ++i) {
        std::tie(edge_to_node[2 * i], edge_to_node[2 * i + 1],
                 inv_permutation[i]) = edge_tmp[i];
      }
      reorderDataInverse<EdgeDim, true>(*edge_data, inv_permutation);
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

  template <unsigned EdgeDim, class DataType>
  void reorderToPartition(std::vector<MY_SIZE> &partition_vector,
                          data_t<DataType, EdgeDim> &edge_weights) {
    assert(numEdges() == partition_vector.size());
    assert(numEdges() == edge_weights.getSize());
    std::vector<std::tuple<MY_SIZE, MY_SIZE, MY_SIZE, MY_SIZE>> tmp(numEdges());
    for (MY_SIZE i = 0; i < numEdges(); ++i) {
      tmp[i] = std::make_tuple(partition_vector[i], i, edge_to_node[2 * i],
                               edge_to_node[2 * i + 1]);
    }
    std::sort(tmp.begin(), tmp.end());
    std::vector<MY_SIZE> permutation(numEdges());
    for (MY_SIZE i = 0; i < numEdges(); ++i) {
      std::tie(partition_vector[i], permutation[i], edge_to_node[2 * i],
               edge_to_node[2 * i + 1]) = tmp[i];
    }
    reorderDataInverse<EdgeDim, true, DataType, MY_SIZE>(edge_weights,
                                                         permutation);
  }

  std::vector<MY_SIZE> getPointRenumberingPermutation() const {
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

  std::vector<MY_SIZE> renumberPoints(const std::vector<MY_SIZE> &permutation) {
    std::for_each(edge_to_node.begin(), edge_to_node.end(),
                  [&permutation](MY_SIZE &a) { a = permutation[a]; });
    if (point_coordinates.getSize() > 0) {
      reorderData<3, false, float, MY_SIZE>(point_coordinates, permutation);
    }
    return permutation;
  }

  template <bool MinimiseColourSizes = true>
  static MY_SIZE getAvailableColour(colourset_t available_colours,
                                    const std::vector<MY_SIZE> &set_sizes) {
    assert(set_sizes.size() > 0);
    MY_SIZE colour = set_sizes.size();
    for (MY_SIZE i = 0; i < set_sizes.size(); ++i) {
      if (available_colours[i]) {
        if (MinimiseColourSizes) {
          if (colour >= set_sizes.size() || set_sizes[colour] > set_sizes[i]) {
            colour = i;
          }
        } else {
          return i;
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

  std::vector<std::vector<MY_SIZE>>
  getPointToPartition(const std::vector<MY_SIZE> &partition) const {
    std::vector<std::set<MY_SIZE>> _result(num_points);
    for (MY_SIZE i = 0; i < edge_to_node.getSize(); ++i) {
      _result[edge_to_node[2 * i + 0]].insert(partition[i]);
      _result[edge_to_node[2 * i + 1]].insert(partition[i]);
    }
    std::vector<std::vector<MY_SIZE>> result(num_points);
    std::transform(_result.begin(), _result.end(), result.begin(),
                   [](const std::set<MY_SIZE> &a) {
                     return std::vector<MY_SIZE>(a.begin(), a.end());
                   });
    return result;
  }

  std::vector<MY_SIZE> getPointRenumberingPermutation2(
      const std::vector<std::vector<MY_SIZE>> &point_to_partition) const {
    std::vector<MY_SIZE> inverse_permutation(point_to_partition.size());
    data_t<MY_SIZE, 1> permutation(point_to_partition.size());
    for (MY_SIZE i = 0; i < point_to_partition.size(); ++i) {
      inverse_permutation[i] = i;
      permutation[i] = i;
    }
    std::stable_sort(
        inverse_permutation.begin(), inverse_permutation.end(),
        [&point_to_partition](MY_SIZE a, MY_SIZE b) {
          if (point_to_partition[a].size() != point_to_partition[b].size()) {
            return point_to_partition[a].size() > point_to_partition[b].size();
          } else {
            return point_to_partition[a] > point_to_partition[b];
          }
        });
    reorderData<1, false>(permutation, inverse_permutation);
    return std::vector<MY_SIZE>(permutation.begin(), permutation.end());
  }
};

#endif /* end of include guard: GRAPH_HPP_35BFQORK */
// vim:set et sw=2 ts=2 fdm=marker:
