#ifndef MESH_HPP_JLID0VDH
#define MESH_HPP_JLID0VDH

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

template <unsigned MeshDim = 2> struct Mesh {
  static_assert(MeshDim == 2 || MeshDim == 4 || MeshDim == 8,
                "Only supporting MeshDim in {2,4,8}");
  // I assume 64 colour is enough
  using colourset_t = std::bitset<64>;

  static constexpr unsigned MESH_DIM = MeshDim;

  template <unsigned _MeshDim> friend struct Mesh;

private:
  MY_SIZE num_points, num_cells;

public:
  data_t<MY_SIZE, MeshDim> cell_to_node;
  data_t<float, 3> point_coordinates;

  /* Initialisation {{{1 */
private:
  Mesh(MY_SIZE _num_points, MY_SIZE _num_cells, const MY_SIZE *_cell_to_node,
       const float *_point_coordinates = nullptr)
      : num_points{_num_points}, num_cells{_num_cells}, cell_to_node(num_cells),
        point_coordinates(num_points) {
    std::copy(_cell_to_node, _cell_to_node + MeshDim * num_cells,
              cell_to_node.begin());
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

  static MY_SIZE calcNumCells(const std::vector<MY_SIZE> &grid_dim) {
    switch (MeshDim) {
    case 2:
      return calcNumEdges(grid_dim);
    case 4:
      assert(grid_dim.size() == 2);
      return (grid_dim[0] - 1) * (grid_dim[1] - 1);
    case 8:
      assert(grid_dim.size() == 3);
      return (grid_dim[0] - 1) * (grid_dim[1] - 1) * (grid_dim[2] - 1);
    default:
      std::abort(); // Shouldn't arrive here
    }
  }

public:
  Mesh(const std::vector<MY_SIZE> &grid_dim,
       std::pair<MY_SIZE, MY_SIZE> block_sizes = {0, 0},
       bool use_coordinates = false)
      : num_points{calcNumPoints(grid_dim)}, num_cells{calcNumCells(grid_dim)},
        cell_to_node(num_cells),
        point_coordinates(use_coordinates ? num_points : 0) {
    assert(grid_dim.size() >= 2);
    // num_edges = (N - 1) * M + N * (M - 1); // vertical + horizontal
    // num_edges = 2 * ((N - 1) * M + N * (M - 1)); // to and fro
    MY_SIZE N = grid_dim[0];
    MY_SIZE M = grid_dim[1];
    if (MeshDim == 2) {
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
    } else if (MeshDim == 4) {
      assert(grid_dim.size() == 2);
      num_points = N * M;
      fillCellList(N, M);
    } else {
      assert(grid_dim.size() == 3);
      num_points = N * M * grid_dim[2];
      fillCellList3D(N, M, grid_dim[2]);
    }
  }

  /**
   * Constructs graph from stream.
   *
   * Format:
   *   - first line: num_points and num_cells ("\d+\s+\d+")
   *   - next num_cells line: an cell, denoted by MeshDim numbers, the start-
   * and
   *     endpoint respectively ("\d+\s+\d+")
   */
  Mesh(std::istream &is, std::istream *coord_is = nullptr)
      : num_points{0}, num_cells{0},
        cell_to_node((is >> num_points >> num_cells, num_cells)),
        point_coordinates(coord_is == nullptr ? 0 : num_points) {
    if (!is) {
      throw InvalidInputFile{"graph input", 0};
    }
    if (coord_is != nullptr) {
      if (!(*coord_is)) {
        throw InvalidInputFile{"coordinate input", 0};
      }
    }
    for (MY_SIZE i = 0; i < num_cells; ++i) {
      for (MY_SIZE j = 0; j < MeshDim; ++j) {
        is >> cell_to_node[MeshDim * i + j];
      }
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

  ~Mesh() {}

  Mesh(const Mesh &) = delete;
  Mesh &operator=(const Mesh &) = delete;

  Mesh(Mesh &&other)
      : num_points{other.num_points}, num_cells{other.num_cells},
        cell_to_node{std::move(other.cell_to_node)},
        point_coordinates{std::move(other.point_coordinates)} {
    other.num_points = 0;
    other.num_cells = 0;
  }

  Mesh &operator=(Mesh &&rhs) {
    std::swap(num_points, rhs.num_points);
    std::swap(num_cells, rhs.num_cells);
    std::swap(cell_to_node, rhs.cell_to_node);
    std::swap(point_coordinates, rhs.point_coordinates);
    return *this;
  }

  /**
   * Grid, unidirectional: right and down
   */
  /* fillEdgeList {{{2 */
  void fillEdgeList(MY_SIZE N, MY_SIZE M) {
    MY_SIZE array_ind = 0, upper_point_ind = 0, lower_point_ind = M;
    for (MY_SIZE r = 0; r < N - 1; ++r) {
      for (MY_SIZE c = 0; c < M - 1; ++c) {
        cell_to_node[array_ind++] = upper_point_ind;
        cell_to_node[array_ind++] = lower_point_ind;
        cell_to_node[array_ind++] = upper_point_ind;
        cell_to_node[array_ind++] = ++upper_point_ind;
        ++lower_point_ind;
      }
      cell_to_node[array_ind++] = lower_point_ind++;
      cell_to_node[array_ind++] = upper_point_ind++;
    }
    for (MY_SIZE c = 0; c < M - 1; ++c) {
      cell_to_node[array_ind++] = upper_point_ind;
      cell_to_node[array_ind++] = ++upper_point_ind;
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
  /* 2}}} */

  /**
   * Grid, bidirectional
   */
  /* fillEdgeList2 {{{2 */
  void fillEdgeList2(MY_SIZE N, MY_SIZE M) {
    MY_SIZE array_ind = 0, upper_point_ind = 0, lower_point_ind = M;
    for (MY_SIZE r = 0; r < N - 1; ++r) {
      for (MY_SIZE c = 0; c < M - 1; ++c) {
        // up-down
        cell_to_node[array_ind++] = lower_point_ind;
        cell_to_node[array_ind++] = upper_point_ind;
        cell_to_node[array_ind++] = upper_point_ind;
        cell_to_node[array_ind++] = lower_point_ind;
        // right-left
        cell_to_node[array_ind++] = upper_point_ind;
        cell_to_node[array_ind++] = upper_point_ind + 1;
        cell_to_node[array_ind++] = upper_point_ind + 1;
        cell_to_node[array_ind++] = upper_point_ind;
        ++lower_point_ind;
        ++upper_point_ind;
      }
      // Last up-down
      cell_to_node[array_ind++] = lower_point_ind;
      cell_to_node[array_ind++] = upper_point_ind;
      cell_to_node[array_ind++] = upper_point_ind++;
      cell_to_node[array_ind++] = lower_point_ind++;
    }
    // Last horizontal
    for (MY_SIZE c = 0; c < M - 1; ++c) {
      cell_to_node[array_ind++] = upper_point_ind;
      cell_to_node[array_ind++] = upper_point_ind + 1;
      cell_to_node[array_ind++] = upper_point_ind + 1;
      cell_to_node[array_ind++] = upper_point_ind;
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
  /* 2}}} */

  /**
   * Grid, hard coded block-indexing
   */
  /* fillEdgeListBlock {{{2 */
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
            cell_to_node[ind++] = (block_h * i + k) * M + (block_w * j + l);
            cell_to_node[ind++] = (block_h * i + k + 1) * M + (block_w * j + l);
            // Right
            cell_to_node[ind++] = (block_h * i + k) * M + (block_w * j + l);
            cell_to_node[ind++] = (block_h * i + k) * M + (block_w * j + l + 1);
          }
        }
      }
    }
    for (MY_SIZE i = 0; i < N - 1; ++i) {
      // Right side, edges directed downwards
      cell_to_node[ind++] = i * M + (M - 1);
      cell_to_node[ind++] = (i + 1) * M + (M - 1);
    }
    for (MY_SIZE i = 0; i < M - 1; ++i) {
      // Down side, edges directed right
      cell_to_node[ind++] = (N - 1) * M + i;
      cell_to_node[ind++] = (N - 1) * M + i + 1;
    }
    assert(ind == 2 * numCells());
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
  /* 2}}} */

  /* fillEdgeList3D {{{2 */
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
          cell_to_node[array_ind++] = lower_point_ind;
          cell_to_node[array_ind++] = upper_point_ind;
          // Deep
          cell_to_node[array_ind++] = upper_point_ind;
          cell_to_node[array_ind++] = inner_point_ind;
          // Left
          cell_to_node[array_ind++] = upper_point_ind;
          cell_to_node[array_ind++] = ++upper_point_ind;
          ++lower_point_ind;
          ++inner_point_ind;
        }
        // Left end
        cell_to_node[array_ind++] = lower_point_ind++;
        cell_to_node[array_ind++] = upper_point_ind;
        cell_to_node[array_ind++] = inner_point_ind++;
        cell_to_node[array_ind++] = upper_point_ind++;
      }
      // Down end
      for (MY_SIZE c = 0; c < N2 - 1; ++c) {
        cell_to_node[array_ind++] = upper_point_ind;
        cell_to_node[array_ind++] = upper_point_ind + 1;
        cell_to_node[array_ind++] = inner_point_ind++;
        cell_to_node[array_ind++] = upper_point_ind++;
      }
      // Down last element
      cell_to_node[array_ind++] = inner_point_ind++;
      cell_to_node[array_ind++] = upper_point_ind++;
    }

    // Last layer
    upper_point_ind = (N3 - 1) * N1 * N2;
    lower_point_ind = upper_point_ind + N2;
    for (MY_SIZE r = 0; r < N1 - 1; ++r) {
      for (MY_SIZE c = 0; c < N2 - 1; ++c) {
        // Down
        cell_to_node[array_ind++] = lower_point_ind++;
        cell_to_node[array_ind++] = upper_point_ind;
        // Left
        cell_to_node[array_ind++] = upper_point_ind;
        cell_to_node[array_ind++] = ++upper_point_ind;
      }
      // Left end
      cell_to_node[array_ind++] = lower_point_ind++;
      cell_to_node[array_ind++] = upper_point_ind++;
    }
    // Last layer Down end
    for (MY_SIZE c = 0; c < N2 - 1; ++c) {
      cell_to_node[array_ind++] = upper_point_ind;
      cell_to_node[array_ind++] = ++upper_point_ind;
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
  /* 2}}} */

  /* fillEdgeList9x8 {{{2 */
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
            cell_to_node[ind++] = point_cur;
            cell_to_node[ind++] = point_down;
            // Right
            cell_to_node[ind++] = point_cur;
            cell_to_node[ind++] = point_right;

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
      cell_to_node[ind++] = point_cur;
      cell_to_node[ind++] = point_cur + 1;

      if (point_coordinates.getSize() > 0) {
        point_coordinates[point_cur * 3 + 0] = N - 1;
        point_coordinates[point_cur * 3 + 1] = i;
        point_coordinates[point_cur * 3 + 2] = 0;
      }
      ++point_cur;
    }
    for (MY_SIZE i = 0; i < N - 1; ++i) {
      cell_to_node[ind++] = point_cur;
      cell_to_node[ind++] = point_cur + 1;

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
    assert(ind == 2 * numCells());
    assert(point_cur + 1 == numPoints());
  }
  /* 2}}} */

  /* fillCellList {{{2 */
  void fillCellList(MY_SIZE N, MY_SIZE M) {
    assert(N > 1);
    assert(M > 1);
    MY_SIZE array_ind = 0;
    for (MY_SIZE r = 0; r < N - 1; ++r) {
      for (MY_SIZE c = 0; c < M - 1; ++c) {
        MY_SIZE top_left = r * M + c;
        MY_SIZE bottom_left = (r + 1) * M + c;
        cell_to_node[array_ind++] = top_left;
        cell_to_node[array_ind++] = top_left + 1;
        cell_to_node[array_ind++] = bottom_left;
        cell_to_node[array_ind++] = bottom_left + 1;
      }
    }
    assert(array_ind == numCells() * MeshDim);
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
  /* 2}}} */

  /* fillCellList3D {{{2 */
  void fillCellList3D(MY_SIZE N1, MY_SIZE N2, MY_SIZE N3) {
    assert(N1 > 1);
    assert(N2 > 1);
    assert(N3 > 1);
    MY_SIZE array_ind = 0;
    for (MY_SIZE l = 0; l < N3 - 1; ++l) {
      for (MY_SIZE r = 0; r < N1 - 1; ++r) {
        for (MY_SIZE c = 0; c < N2 - 1; ++c) {
          MY_SIZE bottom_layer = l * N1 * N2;
          MY_SIZE top_layer = (l + 1) * N1 * N2;
          MY_SIZE inner_left_offset = r * N2 + c;
          MY_SIZE outer_left_offset = (r + 1) * N2 + c;
          cell_to_node[array_ind++] = bottom_layer + inner_left_offset;
          cell_to_node[array_ind++] = bottom_layer + inner_left_offset + 1;
          cell_to_node[array_ind++] = bottom_layer + outer_left_offset + 1;
          cell_to_node[array_ind++] = bottom_layer + outer_left_offset;
          cell_to_node[array_ind++] = top_layer + inner_left_offset;
          cell_to_node[array_ind++] = top_layer + inner_left_offset + 1;
          cell_to_node[array_ind++] = top_layer + outer_left_offset + 1;
          cell_to_node[array_ind++] = top_layer + outer_left_offset;
        }
      }
    }
    assert(array_ind == numCells() * MeshDim);
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
  /* 2}}} */

  /* 1}}} */

  template <bool VTK = false>
  typename choose_t<VTK, std::vector<std::uint16_t>,
                    std::vector<std::vector<MY_SIZE>>>::type
  colourCells(MY_SIZE from = 0, MY_SIZE to = static_cast<MY_SIZE>(-1)) const {
    if (to > numCells()) {
      to = numCells();
    }
    std::vector<std::vector<MY_SIZE>> cell_partitions;
    std::vector<colourset_t> point_colours(numPoints(), 0);
    std::vector<MY_SIZE> set_sizes(64, 0);
    std::vector<std::uint16_t> cell_colours(numCells());
    colourset_t used_colours;
    for (MY_SIZE i = from; i < to; ++i) {
      colourset_t occupied_colours;
      for (MY_SIZE j = 0; j < MeshDim; ++j) {
        occupied_colours |= point_colours[cell_to_node[MeshDim * i + j]];
      }
      colourset_t available_colours = ~occupied_colours & used_colours;
      if (available_colours.none()) {
        used_colours <<= 1;
        used_colours.set(0);
        if (!VTK) {
          cell_partitions.emplace_back();
        }
        available_colours = ~occupied_colours & used_colours;
      }
      std::uint8_t colour = getAvailableColour(available_colours, set_sizes);
      if (VTK) {
        cell_colours[i] = colour;
      } else {
        cell_partitions[colour].push_back(i);
      }
      colourset_t colourset(1ull << colour);
      for (MY_SIZE j = 0; j < MeshDim; ++j) {
        point_colours[cell_to_node[MeshDim * i + j]] |= colourset;
      }
      ++set_sizes[colour];
    }
    return choose_t<
        VTK, std::vector<std::uint16_t>,
        std::vector<std::vector<MY_SIZE>>>::ret_value(std::move(cell_colours),
                                                      std::move(
                                                          cell_partitions));
  }

  MY_SIZE numCells() const { return num_cells; }

  MY_SIZE numPoints() const { return num_points; }

  /**
   * Writes the cell list in the following format:
   *   - the first line contains two numbers separated by spaces, `numPoints()`
   *     and `numCells()` respectively.
   *   - the following `numCells()` lines contain MeshDim numbers separated
   *     by spaces: the points incident to the cell
   */
  void writeCellList(std::ostream &os) const {
    os << numPoints() << " " << numCells() << std::endl;
    for (std::size_t i = 0; i < numCells(); ++i) {
      for (unsigned j = 0; j < MeshDim; ++j) {
        os << (j > 0 ? " " : "") << cell_to_node[MeshDim * i + j];
      }
      os << std::endl;
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
            unsigned CellDim = 1, bool SOA = false>
  void reorderScotch(data_t<DataType, CellDim> *cell_data = nullptr,
                     data_t<DataType, DataDim> *point_data = nullptr) {
    ScotchReorder reorder(numPoints(), numCells(), cell_to_node);
    std::vector<SCOTCH_Num> permutation = reorder.reorder();
    this->template reorder<SCOTCH_Num, DataType, DataDim, CellDim, SOA>(
        permutation, cell_data, point_data);
  }

  /**
   * Reorders the graph using the point permutation vector.
   *
   * Also reorders the cell and point data in the arguments. These must be of
   * length `numEdges()` and `numPoints()`, respectively.
   */
  template <typename UnsignedType, typename DataType = float,
            unsigned DataDim = 1, unsigned CellDim = 1, bool SOA = false>
  void reorder(const std::vector<UnsignedType> &point_permutation,
               data_t<DataType, CellDim> *cell_data = nullptr,
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
    // Permute cell_to_node
    for (MY_SIZE i = 0; i < numCells(); ++i) {
      for (MY_SIZE j = 0; j < MeshDim; ++j) {
        cell_to_node[MeshDim * i + j] =
            point_permutation[cell_to_node[MeshDim * i + j]];
      }
    }
    std::vector<std::array<MY_SIZE, MeshDim + 1>> cell_tmp(numCells());
    for (MY_SIZE i = 0; i < numCells(); ++i) {
      cell_tmp[i][MeshDim] = i;
      std::copy(cell_to_node.begin() + MeshDim * i,
                cell_to_node.begin() + MeshDim * (i + 1), cell_tmp[i].begin());
      std::sort(cell_tmp[i].begin(), cell_tmp[i].begin() + MeshDim);
    }
    std::sort(cell_tmp.begin(), cell_tmp.end());
    std::vector<MY_SIZE> inv_permutation(numCells());
    for (MY_SIZE i = 0; i < numCells(); ++i) {
      inv_permutation[i] = cell_tmp[i][MeshDim];
    }
    if (cell_data) {
      reorderDataInverse<CellDim, true>(*cell_data, inv_permutation);
    }
    reorderDataInverse<MeshDim, false>(cell_to_node, inv_permutation);
  }

  template <unsigned CellDim, class DataType>
  void reorderToPartition(std::vector<MY_SIZE> &partition_vector,
                          data_t<DataType, CellDim> &cell_weights) {
    assert(numCells() == partition_vector.size());
    assert(numCells() == cell_weights.getSize());
    std::vector<std::array<MY_SIZE, MeshDim + 2>> tmp(numCells());
    for (MY_SIZE i = 0; i < numCells(); ++i) {
      tmp[i][0] = partition_vector[i];
      tmp[i][1] = i;
      std::copy(cell_to_node.begin() + MeshDim * i,
                cell_to_node.begin() + MeshDim * (i + 1), tmp[i].begin() + 2);
    }
    std::sort(tmp.begin(), tmp.end());
    std::vector<MY_SIZE> permutation(numCells());
    for (MY_SIZE i = 0; i < numCells(); ++i) {
      partition_vector[i] = tmp[i][0];
      permutation[i] = tmp[i][1];
      std::copy(tmp[i].begin() + 2, tmp[i].end(),
                cell_to_node.begin() + MeshDim * i);
    }
    reorderDataInverse<CellDim, true, DataType, MY_SIZE>(cell_weights,
                                                         permutation);
  }

  std::vector<MY_SIZE> getPointRenumberingPermutation() const {
    std::vector<MY_SIZE> permutation(numPoints(), numPoints());
    MY_SIZE new_ind = 0;
    for (MY_SIZE i = 0; i < MeshDim * numCells(); ++i) {
      if (permutation[cell_to_node[i]] == numPoints()) {
        permutation[cell_to_node[i]] = new_ind++;
      }
    }
    // Currently not supporting isolated points
    assert(std::all_of(
        permutation.begin(), permutation.end(),
        [&permutation](MY_SIZE a) { return a < permutation.size(); }));
    return permutation;
  }

  std::vector<MY_SIZE> renumberPoints(const std::vector<MY_SIZE> &permutation) {
    std::for_each(cell_to_node.begin(), cell_to_node.end(),
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

  Mesh<2> getCellToCellGraph() const {
    const std::multimap<MY_SIZE, MY_SIZE> point_to_cell =
        GraphCSR<MY_SIZE>::getPointToCell(cell_to_node);
    // TODO optimise
    std::vector<MY_SIZE> cell_to_cell;
    for (MY_SIZE i = 0; i < numCells(); ++i) {
      for (MY_SIZE offset = 0; offset < MeshDim; ++offset) {
        MY_SIZE point = cell_to_node[MeshDim * i + offset];
        const auto cell_range = point_to_cell.equal_range(point);
        for (auto it = cell_range.first; it != cell_range.second; ++it) {
          MY_SIZE other_cell = it->second;
          if (other_cell > i) {
            cell_to_cell.push_back(i);
            cell_to_cell.push_back(other_cell);
          }
        }
      }
    }
    return Mesh<2>(numCells(), cell_to_cell.size() / 2, cell_to_cell.data());
  }

  std::vector<std::vector<MY_SIZE>>
  getPointToPartition(const std::vector<MY_SIZE> &partition) const {
    std::vector<std::set<MY_SIZE>> _result(num_points);
    for (MY_SIZE i = 0; i < cell_to_node.getSize(); ++i) {
      for (MY_SIZE j = 0; j < MeshDim; ++j) {
        _result[cell_to_node[MeshDim * i + j]].insert(partition[i]);
      }
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

// vim:set et sw=2 ts=2 fdm=marker:
#endif /* end of include guard: MESH_HPP_JLID0VDH */
