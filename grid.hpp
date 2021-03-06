#ifndef GRID_HPP_ZBTKDSNG
#define GRID_HPP_ZBTKDSNG

#include "visualisable_mesh.hpp"

class Grid : public VisualisableMesh {
public:
  static MY_SIZE calcNumPoints(const std::vector<MY_SIZE> &grid_dim) {
    assert(grid_dim.size() >= 2);
    return grid_dim[0] * grid_dim[1] * (grid_dim.size() == 3 ? grid_dim[2] : 1);
  }

  static MY_SIZE calcNumEdges(const std::vector<MY_SIZE> &grid_dim) {
    assert(grid_dim.size() >= 2);
    MY_SIZE N = grid_dim[0];
    MY_SIZE M = grid_dim[1];
    return ((N - 1) * M + N * (M - 1)) *
               (grid_dim.size() == 3 ? grid_dim[2] : 1) +
           (grid_dim.size() == 3 ? (grid_dim[2] - 1) * N * M : 0);
  }

  static MY_SIZE calcNumCells(const std::vector<MY_SIZE> &grid_dim,
                              unsigned mesh_dim) {
    switch (mesh_dim) {
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

  Grid(const std::vector<MY_SIZE> &grid_dim, unsigned mesh_dim,
       std::vector<MY_SIZE> block_sizes = {0, 0},
       bool use_coordinates = false)
      : VisualisableMesh(calcNumPoints(grid_dim),
                         calcNumCells(grid_dim, mesh_dim), mesh_dim,
                         use_coordinates) {
    assert(mesh_dim == 2 || mesh_dim == 4 || mesh_dim == 8);
    assert(grid_dim.size() >= 2);
    MY_SIZE N = grid_dim[0];
    MY_SIZE M = grid_dim[1];
    if (mesh_dim == 2) {
      if (grid_dim.size() == 2) {
        if (block_sizes[0] != 0) {
          fillEdgeListBlock(N, M, block_sizes[0], block_sizes[1]);
        } else {
          fillEdgeList(N, M);
        }
      } else {
        fillEdgeList3D(N, M, grid_dim[2]);
      }
    } else if (mesh_dim == 4) {
      assert(grid_dim.size() == 2);
      fillCellList(N, M);
    } else {
      assert(grid_dim.size() == 3);
      if (block_sizes[0] != 0) {
        fillCellListBlock3D(N, M, grid_dim[2], block_sizes[0], block_sizes[1],
            block_sizes[2]);
      } else {
        fillCellList3D(N, M, grid_dim[2]);
      }
    }
  }

protected:
  /**
   * Grid, unidirectional: right and down
   */
  /* fillEdgeList {{{1 */
  void fillEdgeList(MY_SIZE N, MY_SIZE M) {
    MY_SIZE array_ind = 0, upper_point_ind = 0, lower_point_ind = M;
    for (MY_SIZE r = 0; r < N - 1; ++r) {
      for (MY_SIZE c = 0; c < M - 1; ++c) {
        cell_to_node[0].operator[]<MY_SIZE>(array_ind++) = upper_point_ind;
        cell_to_node[0].operator[]<MY_SIZE>(array_ind++) = lower_point_ind;
        cell_to_node[0].operator[]<MY_SIZE>(array_ind++) = upper_point_ind;
        cell_to_node[0].operator[]<MY_SIZE>(array_ind++) = ++upper_point_ind;
        ++lower_point_ind;
      }
      cell_to_node[0].operator[]<MY_SIZE>(array_ind++) = lower_point_ind++;
      cell_to_node[0].operator[]<MY_SIZE>(array_ind++) = upper_point_ind++;
    }
    for (MY_SIZE c = 0; c < M - 1; ++c) {
      cell_to_node[0].operator[]<MY_SIZE>(array_ind++) = upper_point_ind;
      cell_to_node[0].operator[]<MY_SIZE>(array_ind++) = ++upper_point_ind;
    }
    if (point_coordinates.getSize() > 0) {
      for (MY_SIZE r = 0; r < N; ++r) {
        for (MY_SIZE c = 0; c < M; ++c) {
          MY_SIZE point_ind = r * M + c;
          point_coordinates.operator[]<float>(point_ind * 3 + 0) = r;
          point_coordinates.operator[]<float>(point_ind * 3 + 1) = c;
          point_coordinates.operator[]<float>(point_ind * 3 + 2) = 0;
        }
      }
    }
  }
  /* 1}}} */

  /**
   * Grid, bidirectional
   */
  /* fillEdgeList2 {{{1 */
  void fillEdgeList2(MY_SIZE N, MY_SIZE M) {
    MY_SIZE array_ind = 0, upper_point_ind = 0, lower_point_ind = M;
    for (MY_SIZE r = 0; r < N - 1; ++r) {
      for (MY_SIZE c = 0; c < M - 1; ++c) {
        // up-down
        cell_to_node[0].operator[]<MY_SIZE>(array_ind++) = lower_point_ind;
        cell_to_node[0].operator[]<MY_SIZE>(array_ind++) = upper_point_ind;
        cell_to_node[0].operator[]<MY_SIZE>(array_ind++) = upper_point_ind;
        cell_to_node[0].operator[]<MY_SIZE>(array_ind++) = lower_point_ind;
        // right-left
        cell_to_node[0].operator[]<MY_SIZE>(array_ind++) = upper_point_ind;
        cell_to_node[0].operator[]<MY_SIZE>(array_ind++) = upper_point_ind + 1;
        cell_to_node[0].operator[]<MY_SIZE>(array_ind++) = upper_point_ind + 1;
        cell_to_node[0].operator[]<MY_SIZE>(array_ind++) = upper_point_ind;
        ++lower_point_ind;
        ++upper_point_ind;
      }
      // Last up-down
      cell_to_node[0].operator[]<MY_SIZE>(array_ind++) = lower_point_ind;
      cell_to_node[0].operator[]<MY_SIZE>(array_ind++) = upper_point_ind;
      cell_to_node[0].operator[]<MY_SIZE>(array_ind++) = upper_point_ind++;
      cell_to_node[0].operator[]<MY_SIZE>(array_ind++) = lower_point_ind++;
    }
    // Last horizontal
    for (MY_SIZE c = 0; c < M - 1; ++c) {
      cell_to_node[0].operator[]<MY_SIZE>(array_ind++) = upper_point_ind;
      cell_to_node[0].operator[]<MY_SIZE>(array_ind++) = upper_point_ind + 1;
      cell_to_node[0].operator[]<MY_SIZE>(array_ind++) = upper_point_ind + 1;
      cell_to_node[0].operator[]<MY_SIZE>(array_ind++) = upper_point_ind;
      ++upper_point_ind;
    }
    if (point_coordinates.getSize() > 0) {
      for (MY_SIZE r = 0; r < N; ++r) {
        for (MY_SIZE c = 0; c < M; ++c) {
          MY_SIZE point_ind = r * M + c;
          point_coordinates.operator[]<float>(point_ind * 3 + 0) = r;
          point_coordinates.operator[]<float>(point_ind * 3 + 1) = c;
          point_coordinates.operator[]<float>(point_ind * 3 + 2) = 0;
        }
      }
    }
  }
  /* 1}}} */

  /**
   * Grid, hard coded block-indexing
   */
  /* fillEdgeListBlock {{{1 */
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
            cell_to_node[0].operator[]<MY_SIZE>(ind++) =
                (block_h * i + k) * M + (block_w * j + l);
            cell_to_node[0].operator[]<MY_SIZE>(ind++) =
                (block_h * i + k + 1) * M + (block_w * j + l);
            // Right
            cell_to_node[0].operator[]<MY_SIZE>(ind++) =
                (block_h * i + k) * M + (block_w * j + l);
            cell_to_node[0].operator[]<MY_SIZE>(ind++) =
                (block_h * i + k) * M + (block_w * j + l + 1);
          }
        }
      }
    }
    for (MY_SIZE i = 0; i < N - 1; ++i) {
      // Right side, edges directed downwards
      cell_to_node[0].operator[]<MY_SIZE>(ind++) = i * M + (M - 1);
      cell_to_node[0].operator[]<MY_SIZE>(ind++) = (i + 1) * M + (M - 1);
    }
    for (MY_SIZE i = 0; i < M - 1; ++i) {
      // Down side, edges directed right
      cell_to_node[0].operator[]<MY_SIZE>(ind++) = (N - 1) * M + i;
      cell_to_node[0].operator[]<MY_SIZE>(ind++) = (N - 1) * M + i + 1;
    }
    assert(ind == 2 * numCells());
    if (point_coordinates.getSize() > 0) {
      for (MY_SIZE r = 0; r < N; ++r) {
        for (MY_SIZE c = 0; c < M; ++c) {
          MY_SIZE point_ind = r * M + c;
          point_coordinates.operator[]<float>(point_ind * 3 + 0) = r;
          point_coordinates.operator[]<float>(point_ind * 3 + 1) = c;
          point_coordinates.operator[]<float>(point_ind * 3 + 2) = 0;
        }
      }
    }
    Mesh::renumberPoints(Mesh::getPointRenumberingPermutation(0), 0);
  }
  /* 1}}} */

  /* fillEdgeList3D {{{1 */
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
          cell_to_node[0].operator[]<MY_SIZE>(array_ind++) = lower_point_ind;
          cell_to_node[0].operator[]<MY_SIZE>(array_ind++) = upper_point_ind;
          // Deep
          cell_to_node[0].operator[]<MY_SIZE>(array_ind++) = upper_point_ind;
          cell_to_node[0].operator[]<MY_SIZE>(array_ind++) = inner_point_ind;
          // Left
          cell_to_node[0].operator[]<MY_SIZE>(array_ind++) = upper_point_ind;
          cell_to_node[0].operator[]<MY_SIZE>(array_ind++) = ++upper_point_ind;
          ++lower_point_ind;
          ++inner_point_ind;
        }
        // Left end
        cell_to_node[0].operator[]<MY_SIZE>(array_ind++) = lower_point_ind++;
        cell_to_node[0].operator[]<MY_SIZE>(array_ind++) = upper_point_ind;
        cell_to_node[0].operator[]<MY_SIZE>(array_ind++) = inner_point_ind++;
        cell_to_node[0].operator[]<MY_SIZE>(array_ind++) = upper_point_ind++;
      }
      // Down end
      for (MY_SIZE c = 0; c < N2 - 1; ++c) {
        cell_to_node[0].operator[]<MY_SIZE>(array_ind++) = upper_point_ind;
        cell_to_node[0].operator[]<MY_SIZE>(array_ind++) = upper_point_ind + 1;
        cell_to_node[0].operator[]<MY_SIZE>(array_ind++) = inner_point_ind++;
        cell_to_node[0].operator[]<MY_SIZE>(array_ind++) = upper_point_ind++;
      }
      // Down last element
      cell_to_node[0].operator[]<MY_SIZE>(array_ind++) = inner_point_ind++;
      cell_to_node[0].operator[]<MY_SIZE>(array_ind++) = upper_point_ind++;
    }

    // Last layer
    upper_point_ind = (N3 - 1) * N1 * N2;
    lower_point_ind = upper_point_ind + N2;
    for (MY_SIZE r = 0; r < N1 - 1; ++r) {
      for (MY_SIZE c = 0; c < N2 - 1; ++c) {
        // Down
        cell_to_node[0].operator[]<MY_SIZE>(array_ind++) = lower_point_ind++;
        cell_to_node[0].operator[]<MY_SIZE>(array_ind++) = upper_point_ind;
        // Left
        cell_to_node[0].operator[]<MY_SIZE>(array_ind++) = upper_point_ind;
        cell_to_node[0].operator[]<MY_SIZE>(array_ind++) = ++upper_point_ind;
      }
      // Left end
      cell_to_node[0].operator[]<MY_SIZE>(array_ind++) = lower_point_ind++;
      cell_to_node[0].operator[]<MY_SIZE>(array_ind++) = upper_point_ind++;
    }
    // Last layer Down end
    for (MY_SIZE c = 0; c < N2 - 1; ++c) {
      cell_to_node[0].operator[]<MY_SIZE>(array_ind++) = upper_point_ind;
      cell_to_node[0].operator[]<MY_SIZE>(array_ind++) = ++upper_point_ind;
    }

    // generate coordinates
    if (point_coordinates.getSize() > 0) {
      for (MY_SIZE l = 0; l < N3; ++l) {
        for (MY_SIZE r = 0; r < N1; ++r) {
          for (MY_SIZE c = 0; c < N2; ++c) {
            MY_SIZE point_ind = l * N1 * N2 + r * N2 + c;
            point_coordinates.operator[]<float>(point_ind * 3 + 0) = r;
            point_coordinates.operator[]<float>(point_ind * 3 + 1) = c;
            point_coordinates.operator[]<float>(point_ind * 3 + 2) = l;
          }
        }
      }
    }
  }
  /* 1}}} */

  /* fillEdgeList9x8 {{{1 */
  void fillEdgeList9x8(MY_SIZE N, MY_SIZE M) {
    assert((N - 1) % 9 == 0);
    assert((M - 1) % 8 == 0);
    constexpr MY_SIZE block_h = 9, block_w = 8;
    // clang-format off
    std::array<std::array<MY_SIZE, block_w + 1>, block_h + 1> pattern = {{
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
            cell_to_node[0].operator[]<MY_SIZE>(ind++) = point_cur;
            cell_to_node[0].operator[]<MY_SIZE>(ind++) = point_down;
            // Right
            cell_to_node[0].operator[]<MY_SIZE>(ind++) = point_cur;
            cell_to_node[0].operator[]<MY_SIZE>(ind++) = point_right;

            if (point_coordinates.getSize() > 0) {
              point_coordinates.operator[]<float>(point_cur * 3 + 0) =
                  i * block_h + k;
              point_coordinates.operator[]<float>(point_cur * 3 + 1) =
                  j * block_w + l;
              point_coordinates.operator[]<float>(point_cur * 3 + 2) = 0;
            }
          }
        }
        block_ind_offset += block_h * block_w;
      }
    }
    // edges along the edges of the grid
    MY_SIZE point_cur = (N - 1) * (M - 1);
    for (MY_SIZE i = 0; i < M - 1; ++i) {
      cell_to_node[0].operator[]<MY_SIZE>(ind++) = point_cur;
      cell_to_node[0].operator[]<MY_SIZE>(ind++) = point_cur + 1;

      if (point_coordinates.getSize() > 0) {
        point_coordinates.operator[]<float>(point_cur * 3 + 0) = N - 1;
        point_coordinates.operator[]<float>(point_cur * 3 + 1) = i;
        point_coordinates.operator[]<float>(point_cur * 3 + 2) = 0;
      }
      ++point_cur;
    }
    for (MY_SIZE i = 0; i < N - 1; ++i) {
      cell_to_node[0].operator[]<MY_SIZE>(ind++) = point_cur;
      cell_to_node[0].operator[]<MY_SIZE>(ind++) = point_cur + 1;

      if (point_coordinates.getSize() > 0) {
        point_coordinates.operator[]<float>(point_cur * 3 + 0) = N - 1 - i;
        point_coordinates.operator[]<float>(point_cur * 3 + 1) = M - 1;
        point_coordinates.operator[]<float>(point_cur * 3 + 2) = 0;
      }
      ++point_cur;
    }
    if (point_coordinates.getSize() > 0) {
      point_coordinates.operator[]<float>(point_cur * 3 + 0) = 0;
      point_coordinates.operator[]<float>(point_cur * 3 + 1) = M - 1;
      point_coordinates.operator[]<float>(point_cur * 3 + 2) = 0;
    }
    assert(ind == 2 * numCells());
    assert(point_cur + 1 == numPoints(0));
  }
  /* 1}}} */

  /* fillCellList {{{1 */
  void fillCellList(MY_SIZE N, MY_SIZE M) {
    assert(N > 1);
    assert(M > 1);
    MY_SIZE array_ind = 0;
    for (MY_SIZE r = 0; r < N - 1; ++r) {
      for (MY_SIZE c = 0; c < M - 1; ++c) {
        MY_SIZE top_left = r * M + c;
        MY_SIZE bottom_left = (r + 1) * M + c;
        cell_to_node[0].operator[]<MY_SIZE>(array_ind++) = top_left;
        cell_to_node[0].operator[]<MY_SIZE>(array_ind++) = top_left + 1;
        cell_to_node[0].operator[]<MY_SIZE>(array_ind++) = bottom_left;
        cell_to_node[0].operator[]<MY_SIZE>(array_ind++) = bottom_left + 1;
      }
    }
    assert(array_ind == numCells() * cell_to_node[0].getDim());
    if (point_coordinates.getSize() > 0) {
      for (MY_SIZE r = 0; r < N; ++r) {
        for (MY_SIZE c = 0; c < M; ++c) {
          MY_SIZE point_ind = r * M + c;
          point_coordinates.operator[]<float>(point_ind * 3 + 0) = r;
          point_coordinates.operator[]<float>(point_ind * 3 + 1) = c;
          point_coordinates.operator[]<float>(point_ind * 3 + 2) = 0;
        }
      }
    }
  }
  /* 1}}} */

  /* fillCellList3D {{{1 */
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
          cell_to_node[0].operator[]<MY_SIZE>(array_ind++) =
              bottom_layer + inner_left_offset;
          cell_to_node[0].operator[]<MY_SIZE>(array_ind++) =
              bottom_layer + inner_left_offset + 1;
          cell_to_node[0].operator[]<MY_SIZE>(array_ind++) =
              bottom_layer + outer_left_offset + 1;
          cell_to_node[0].operator[]<MY_SIZE>(array_ind++) =
              bottom_layer + outer_left_offset;
          cell_to_node[0].operator[]<MY_SIZE>(array_ind++) =
              top_layer + inner_left_offset;
          cell_to_node[0].operator[]<MY_SIZE>(array_ind++) =
              top_layer + inner_left_offset + 1;
          cell_to_node[0].operator[]<MY_SIZE>(array_ind++) =
              top_layer + outer_left_offset + 1;
          cell_to_node[0].operator[]<MY_SIZE>(array_ind++) =
              top_layer + outer_left_offset;
        }
      }
    }
    assert(array_ind == numCells() * cell_to_node[0].getDim());
    // generate coordinates
    if (point_coordinates.getSize() > 0) {
      for (MY_SIZE l = 0; l < N3; ++l) {
        for (MY_SIZE r = 0; r < N1; ++r) {
          for (MY_SIZE c = 0; c < N2; ++c) {
            MY_SIZE point_ind = l * N1 * N2 + r * N2 + c;
            point_coordinates.operator[]<float>(point_ind * 3 + 0) = r;
            point_coordinates.operator[]<float>(point_ind * 3 + 1) = c;
            point_coordinates.operator[]<float>(point_ind * 3 + 2) = l;
          }
        }
      }
    }
  }
  /* 1}}} */

  /* fillCellListBlock3D {{{ */
  void fillCellListBlock3D (MY_SIZE N1, MY_SIZE N2, MY_SIZE N3,
      MY_SIZE block_size1, MY_SIZE block_size2, MY_SIZE block_size3) {
    assert(N1 > 1);
    assert(N2 > 1);
    assert(N3 > 1);
    assert(block_size1 > 0);
    assert(block_size2 > 0);
    assert(block_size3 > 0);
    assert((N1 - 1) % block_size1 == 0);
    assert((N2 - 1) % block_size2 == 0);
    assert((N3 - 1) % block_size3 == 0);
    MY_SIZE array_ind = 0;
    // Iterate over the blocks
    for (MY_SIZE l = 0; l < (N3 - 1) / block_size3; ++l) {
      for (MY_SIZE r = 0; r < (N1 - 1) / block_size1; ++r) {
        for (MY_SIZE c = 0; c < (N2 - 1) / block_size2; ++c) {
          // And go through the block
          for (MY_SIZE k3 = 0; k3 < block_size3; ++k3) {
            for (MY_SIZE k1 = 0; k1 < block_size1; ++k1) {
              for (MY_SIZE k2 = 0; k2 < block_size2; ++k2) {
                MY_SIZE bottom_layer = (l * block_size3 + k3) * N1 * N2;
                MY_SIZE top_layer = (l * block_size3 + k3 + 1) * N1 * N2;
                MY_SIZE inner_left_offset = (r * block_size1 + k1) * N2 +
                  (c * block_size2 + k2);
                MY_SIZE outer_left_offset = (r * block_size1 + k1 + 1) * N2 +
                  (c * block_size2 + k2);
                cell_to_node[0].operator[]<MY_SIZE>(array_ind++) =
                    bottom_layer + inner_left_offset;
                cell_to_node[0].operator[]<MY_SIZE>(array_ind++) =
                    bottom_layer + inner_left_offset + 1;
                cell_to_node[0].operator[]<MY_SIZE>(array_ind++) =
                    bottom_layer + outer_left_offset + 1;
                cell_to_node[0].operator[]<MY_SIZE>(array_ind++) =
                    bottom_layer + outer_left_offset;
                cell_to_node[0].operator[]<MY_SIZE>(array_ind++) =
                    top_layer + inner_left_offset;
                cell_to_node[0].operator[]<MY_SIZE>(array_ind++) =
                    top_layer + inner_left_offset + 1;
                cell_to_node[0].operator[]<MY_SIZE>(array_ind++) =
                    top_layer + outer_left_offset + 1;
                cell_to_node[0].operator[]<MY_SIZE>(array_ind++) =
                    top_layer + outer_left_offset;
              }
            }
          }
        }
      }
    }
    assert(array_ind == numCells() * cell_to_node[0].getDim());
    // generate coordinates
    if (point_coordinates.getSize() > 0) {
      for (MY_SIZE l = 0; l < N3; ++l) {
        for (MY_SIZE r = 0; r < N1; ++r) {
          for (MY_SIZE c = 0; c < N2; ++c) {
            MY_SIZE point_ind = l * N1 * N2 + r * N2 + c;
            point_coordinates.operator[]<float>(point_ind * 3 + 0) = r;
            point_coordinates.operator[]<float>(point_ind * 3 + 1) = c;
            point_coordinates.operator[]<float>(point_ind * 3 + 2) = l;
          }
        }
      }
    }
  }
  /* }}} fillCellListBlock3D */
};

// vim:set et sts=2 sw=2 ts=2 fdm=marker:
#endif /* end of include guard: GRID_HPP_ZBTKDSNG */
