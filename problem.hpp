#ifndef PROBLEM_HPP_CGW3IDMV
#define PROBLEM_HPP_CGW3IDMV

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <type_traits>
#include <vector>

#include "grid.hpp"
#include "partition.hpp"
#include "timer.hpp"

constexpr MY_SIZE DEFAULT_BLOCK_SIZE = 128;

template <unsigned PointDim = 1, unsigned CellDim = 1, bool SOA = false,
          typename DataType = float>
struct Problem {
  static_assert(
      CellDim == PointDim || CellDim == 1,
      "I know of no reason why CellDim should be anything but 1 or PointDim");
  static constexpr unsigned MESH_DIM = MESH_DIM_MACRO;
  // And because nvcc doesn't compile the above, we need the following hack:
  // using _MESH_DIM_T = std::integral_constant<unsigned, MESH_DIM_MACRO>;
  // static constexpr _MESH_DIM_T MESH_DIM {};

  Mesh<MESH_DIM> mesh;
  data_t<DataType, CellDim> cell_weights;
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
      : mesh{Grid<MESH_DIM>(grid_dim, block_dims, use_coordinates)},
        cell_weights(mesh.numCells()), point_weights(mesh.numPoints()),
        block_size{calculateBlockSize(block_dims)} {
    for (DataType &weight : cell_weights) {
      weight = DataType(rand() % 10000 + 1) / 5000.0;
      weight *= 0.001;
    }
    reset();
  }

  Problem(std::istream &mesh_is, MY_SIZE _block_size = DEFAULT_BLOCK_SIZE,
          std::istream *partition_is = nullptr)
      : mesh(mesh_is), cell_weights(mesh.numCells()),
        point_weights(mesh.numPoints()), block_size{_block_size} {
    if (partition_is != nullptr) {
      if (!(*partition_is)) {
        throw InvalidInputFile{"partition input", 0};
      }
      partition_vector.resize(mesh.numCells());
      MY_SIZE read_block_size;
      *partition_is >> read_block_size;
      if (!(*partition_is)) {
        throw InvalidInputFile{"partition input", 1};
      }
      if (read_block_size != block_size) {
        std::cerr << "Warning: block size in file (" << read_block_size
                  << ") doesn't equal used block size (" << block_size << ")"
                  << std::endl;
      }
      for (MY_SIZE i = 0; i < mesh.numCells(); ++i) {
        *partition_is >> partition_vector[i];
        if (!(*partition_is)) {
          throw InvalidInputFile{"partition input", i + 1};
        }
      }
    }
    for (DataType &weight : cell_weights) {
      weight = DataType(rand() % 10000 + 1) / 5000.0;
      weight *= 0.001;
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

  void loopGPUCellCentred(MY_SIZE num, MY_SIZE reset_every = 0);
  void loopGPUHierarchical(MY_SIZE num, MY_SIZE reset_every = 0);

  void stepCPUCellCentred(DataType *temp) { /*{{{*/
    for (MY_SIZE cell_ind_base = 0; cell_ind_base < mesh.numCells();
         ++cell_ind_base) {
      for (MY_SIZE offset = 0; offset < MESH_DIM; ++offset) {
        MY_SIZE ind_left_base =
            mesh.cell_to_node[mesh.cell_to_node.dim * cell_ind_base + offset];
        MY_SIZE ind_right_base =
            mesh.cell_to_node[mesh.cell_to_node.dim * cell_ind_base +
                              (offset == MESH_DIM - 1 ? 0 : offset + 1)];
        MY_SIZE w_ind_left = 0, w_ind_right = 0;
        for (MY_SIZE d = 0; d < PointDim; ++d) {
          w_ind_left = index<PointDim, SOA>(mesh.numPoints(), ind_left_base, d);
          w_ind_right =
              index<PointDim, SOA>(mesh.numPoints(), ind_right_base, d);
          MY_SIZE cell_d = CellDim == 1 ? 0 : d;

          MY_SIZE cell_weight_ind =
              index<CellDim, true>(mesh.numCells(), cell_ind_base, cell_d);
          point_weights[w_ind_right] +=
              cell_weights[cell_weight_ind] * temp[w_ind_left];
          point_weights[w_ind_left] +=
              cell_weights[cell_weight_ind] * temp[w_ind_right];
        }
      }
    }
  } /*}}}*/

  void loopCPUCellCentred(MY_SIZE num, MY_SIZE reset_every = 0) { /*{{{*/
    DataType *temp = (DataType *)malloc(sizeof(DataType) * mesh.numPoints() *
                                        point_weights.dim);
    TIMER_START(t);
    for (MY_SIZE i = 0; i < num; ++i) {
      TIMER_TOGGLE(t);
      std::copy(point_weights.begin(), point_weights.end(), temp);
      TIMER_TOGGLE(t);
      stepCPUCellCentred(temp);
      if (reset_every && i % reset_every == reset_every - 1) {
        TIMER_TOGGLE(t);
        reset();
        TIMER_TOGGLE(t);
      }
    }
    PRINT_BANDWIDTH(t, "loopCPUCellCentred",
                    (sizeof(DataType) * (2.0 * PointDim * mesh.numPoints() +
                                         CellDim * mesh.numCells()) +
                     1.0 * MESH_DIM * sizeof(MY_SIZE) * mesh.numCells()) *
                        num,
                    (sizeof(DataType) * (2.0 * PointDim * mesh.numPoints() +
                                         CellDim * mesh.numCells()) +
                     1.0 * MESH_DIM * sizeof(MY_SIZE) * mesh.numCells()) *
                        num);
    free(temp);
  } /*}}}*/

  void stepCPUCellCentredOMP(const std::vector<MY_SIZE> &inds,
                             data_t<DataType, PointDim> &out) { /*{{{*/
    #pragma omp parallel for
    for (MY_SIZE i = 0; i < inds.size(); ++i) {
      MY_SIZE ind = inds[i];
      for (MY_SIZE offset = 0; offset < MESH_DIM; ++offset) {

        MY_SIZE ind_left_base =
            mesh.cell_to_node[mesh.cell_to_node.dim * ind + offset];
        MY_SIZE ind_right_base =
            mesh.cell_to_node[mesh.cell_to_node.dim * ind +
                              (offset == MESH_DIM - 1 ? 0 : offset + 1)];

        MY_SIZE w_ind_left = 0, w_ind_right = 0;
        for (MY_SIZE d = 0; d < PointDim; ++d) {
          w_ind_left = index<PointDim, SOA>(mesh.numPoints(), ind_left_base, d);
          w_ind_right =
              index<PointDim, SOA>(mesh.numPoints(), ind_right_base, d);
          MY_SIZE cell_d = CellDim == 1 ? 0 : d;

          MY_SIZE cell_ind = index<CellDim, true>(mesh.numCells(), ind, cell_d);
          point_weights[w_ind_right] +=
              cell_weights[cell_ind] * out[w_ind_left];
          point_weights[w_ind_left] +=
              cell_weights[cell_ind] * out[w_ind_right];
        }
      }
    }
  } /*}}}*/

  void loopCPUCellCentredOMP(MY_SIZE num, MY_SIZE reset_every = 0) { /*{{{*/
    data_t<DataType, PointDim> temp(point_weights.getSize());
    std::vector<std::vector<MY_SIZE>> partition = mesh.colourCells();
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
        stepCPUCellCentredOMP(partition[c], temp);
      }
      if (reset_every && i % reset_every == reset_every - 1) {
        TIMER_TOGGLE(t);
        reset();
        TIMER_TOGGLE(t);
      }
    }
    PRINT_BANDWIDTH(t, "loopCPUCellCentredOMP",
                    (sizeof(DataType) * (2.0 * PointDim * mesh.numPoints() +
                                         CellDim * mesh.numCells()) +
                     1.0 * MESH_DIM * sizeof(MY_SIZE) * mesh.numCells()) *
                        num,
                    (sizeof(DataType) * (2.0 * PointDim * mesh.numPoints() +
                                         CellDim * mesh.numCells()) +
                     1.0 * MESH_DIM * sizeof(MY_SIZE) * mesh.numCells()) *
                        num);
  } /*}}}*/

  void reorder() {
    mesh.reorderScotch<DataType, PointDim, CellDim, SOA>(&cell_weights,
                                                         &point_weights);
  }

  void partition(float tolerance, idx_t options[METIS_NOPTIONS] = NULL) {
    std::vector<idx_t> _partition_vector;
    if (options == NULL) {
      idx_t _options[METIS_NOPTIONS];
      METIS_SetDefaultOptions(_options);
      _options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_VOL;
      _partition_vector = std::move(partitionMetisEnh(
          mesh.getCellToCellGraph(), block_size, tolerance, options));
    } else {
      _partition_vector = std::move(partitionMetisEnh(
          mesh.getCellToCellGraph(), block_size, tolerance, options));
    }
    partition_vector.resize(_partition_vector.size());
    std::copy(_partition_vector.begin(), _partition_vector.end(),
              partition_vector.begin());
  }

  void reorderToPartition() {
    mesh.reorderToPartition<CellDim, DataType>(partition_vector, cell_weights);
  }

  void renumberPoints() {
    std::vector<MY_SIZE> permutation = mesh.getPointRenumberingPermutation2(
        mesh.getPointToPartition(partition_vector));
    mesh.renumberPoints(permutation);
    reorderData<PointDim, SOA, DataType, MY_SIZE>(point_weights, permutation);
  }

  static MY_SIZE calculateBlockSize(std::pair<MY_SIZE, MY_SIZE> block_dims) {
    if (MESH_DIM == 2) {
      if (block_dims.first == 0) {
        return block_dims.second;
      } else if (block_dims == std::pair<MY_SIZE, MY_SIZE>{9, 8}) {
        return 9 * 8 * 2 * 2;
      } else {
        return block_dims.first * block_dims.second * 2;
      }
    } else {
      // Block dims are not yet supported with meshes
      assert(block_dims.first == 0);
      return block_dims.second;
    }
  }

  void writePartition(std::ostream &os) const {
    os << block_size << std::endl;
    for (MY_SIZE p : partition_vector) {
      os << p << std::endl;
    }
  }

  void readPartition(std::istream &partition_is) {
    if (!partition_is) {
      throw InvalidInputFile{"partition input", 0};
    }
    MY_SIZE read_block_size;
    partition_is >> read_block_size;
    if (!partition_is) {
      throw InvalidInputFile{"partition input", 1};
    }
    if (read_block_size != block_size) {
      std::cerr << "Warning: block size in file (" << read_block_size
                << ") doesn't equal to used block size (" << block_size << ")"
                << std::endl;
    }
    partition_vector.resize(mesh.numCells());
    for (MY_SIZE i = 0; i < mesh.numCells(); ++i) {
      partition_is >> partition_vector[i];
      if (!partition_is) {
        throw InvalidInputFile{"partition input", i + 1};
      }
    }
  }
};

#endif /* end of include guard: PROBLEM_HPP_CGW3IDMV */
// vim:set et sw=2 ts=2 fdm=marker:
