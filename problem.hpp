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

  Mesh mesh;
  data_t cell_weights;
  data_t point_weights;
  const MY_SIZE block_size; // GPU block size
  std::vector<MY_SIZE> partition_vector;

  /* ctor/dtor {{{1 */
protected:
  Problem(Mesh &&mesh_, MY_SIZE block_size_)
      : mesh(std::move(mesh_)),
        cell_weights(data_t::create<DataType>(mesh.numCells(), CellDim)),
        point_weights(data_t::create<DataType>(mesh.numPoints(), PointDim)),
        block_size{block_size_} {}

public:
  Problem(std::istream &mesh_is, MY_SIZE _block_size = DEFAULT_BLOCK_SIZE)
      : mesh(mesh_is, MESH_DIM),
        cell_weights(data_t::create<DataType>(mesh.numCells(), CellDim)),
        point_weights(data_t::create<DataType>(mesh.numPoints(), PointDim)),
        block_size{_block_size} {}

  ~Problem() {}

  Problem(Problem &&other)
      : mesh(std::move(other.mesh)),
        cell_weights(std::move(other.cell_weights)),
        point_weights(std::move(other.point_weights)),
        block_size(other.block_size),
        partition_vector(std::move(other.partition_vector)) {}
  /* 1}}} */

  void loopGPUCellCentred(MY_SIZE num);
  void loopGPUHierarchical(MY_SIZE num);

  template <class UserFunc> void stepCPUCellCentred(DataType *temp) { /*{{{*/
    for (MY_SIZE cell_ind_base = 0; cell_ind_base < mesh.numCells();
         ++cell_ind_base) {
      UserFunc::call(point_weights.cbegin<DataType>(), temp,
                     cell_weights.cbegin<DataType>(),
                     mesh.cell_to_node.cbegin<MY_SIZE>(), cell_ind_base);
      // for (MY_SIZE offset = 0; offset < MESH_DIM; ++offset) {
      //  MY_SIZE ind_left_base = mesh.cell_to_node.operator[]<MY_SIZE>(
      //      mesh.cell_to_node.getDim() * cell_ind_base + offset);
      //  MY_SIZE ind_right_base = mesh.cell_to_node.operator[]<MY_SIZE>(
      //      mesh.cell_to_node.getDim() * cell_ind_base +
      //      (offset == MESH_DIM - 1 ? 0 : offset + 1));
      //  MY_SIZE w_ind_left = 0, w_ind_right = 0;
      //  for (MY_SIZE d = 0; d < PointDim; ++d) {
      //    w_ind_left = index<SOA>(mesh.numPoints(), ind_left_base, PointDim,
      //    d);
      //    w_ind_right =
      //        index<SOA>(mesh.numPoints(), ind_right_base, PointDim, d);
      //    MY_SIZE cell_d = CellDim == 1 ? 0 : d;

      //    MY_SIZE cell_weight_ind =
      //        index<true>(mesh.numCells(), cell_ind_base, CellDim, cell_d);
      //    point_weights.operator[]<DataType>(w_ind_right) +=
      //        cell_weights.operator[]<DataType>(cell_weight_ind) *
      //        temp[w_ind_left];
      //    point_weights.operator[]<DataType>(w_ind_left) +=
      //        cell_weights.operator[](cell_weight_ind) * temp[w_ind_right];
      //  }
      //}
    }
  } /*}}}*/

  template <class UserFunc> void loopCPUCellCentred(MY_SIZE num) { /*{{{*/
    DataType *temp = (DataType *)malloc(sizeof(DataType) * mesh.numPoints() *
                                        point_weights.getDim());
    TIMER_START(t);
    TIMER_TOGGLE(t);
    std::copy(point_weights.begin<DataType>(), point_weights.end<DataType>(),
              temp);
    TIMER_TOGGLE(t);
    for (MY_SIZE i = 0; i < num; ++i) {
      stepCPUCellCentred<UserFunc>(temp);
      TIMER_TOGGLE(t);
      std::copy_n(temp, mesh.numPoints() * point_weights.getDim(),
                  point_weights.begin<DataType>());
      TIMER_TOGGLE(t);
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

  template <class UserFunc>
  void stepCPUCellCentredOMP(const std::vector<MY_SIZE> &inds,
                             data_t &out) { /*{{{*/
    #pragma omp parallel for
    for (MY_SIZE i = 0; i < inds.size(); ++i) {
      MY_SIZE ind = inds[i];
      UserFunc::call(point_weights.cbegin<DataType>(), out.begin<DataType>(),
                     cell_weights.cbegin<DataType>(),
                     mesh.cell_to_node.cbegin<MY_SIZE>(), ind);
      // for (MY_SIZE offset = 0; offset < MESH_DIM; ++offset) {

      //  MY_SIZE ind_left_base = mesh.cell_to_node.operator[]<MY_SIZE>(
      //      mesh.cell_to_node.getDim() * ind + offset);
      //  MY_SIZE ind_right_base = mesh.cell_to_node.operator[]<MY_SIZE>(
      //      mesh.cell_to_node.getDim() * ind +
      //      (offset == MESH_DIM - 1 ? 0 : offset + 1));

      //  MY_SIZE w_ind_left = 0, w_ind_right = 0;
      //  for (MY_SIZE d = 0; d < PointDim; ++d) {
      //    w_ind_left = index<SOA>(mesh.numPoints(), ind_left_base, PointDim,
      //    d);
      //    w_ind_right =
      //        index<SOA>(mesh.numPoints(), ind_right_base, PointDim, d);
      //    MY_SIZE cell_d = CellDim == 1 ? 0 : d;

      //    MY_SIZE cell_ind = index<true>(mesh.numCells(), ind, CellDim,
      //    cell_d);
      //    point_weights.operator[]<DataType>(w_ind_right) +=
      //        cell_weights.operator[]<DataType>(cell_ind) *
      //        out.operator[]<DataType>(w_ind_left);
      //    point_weights.operator[]<DataType>(w_ind_left) +=
      //        cell_weights.operator[]<DataType>(cell_ind) *
      //        out.operator[]<DataType>(w_ind_right);
      //  }
      //}
    }
  } /*}}}*/

  template <class UserFunc> void loopCPUCellCentredOMP(MY_SIZE num) { /*{{{*/
    data_t temp(data_t::create<DataType>(point_weights.getSize(), PointDim));
    std::vector<std::vector<MY_SIZE>> partition = mesh.colourCells();
    MY_SIZE num_of_colours = partition.size();
    TIMER_START(t);
    // Init temp
    TIMER_TOGGLE(t);
    #pragma omp parallel for
    for (MY_SIZE e = 0; e < point_weights.getSize() * point_weights.getDim();
         ++e) {
      temp.operator[]<DataType>(e) = point_weights.operator[]<DataType>(e);
    }
    TIMER_TOGGLE(t);
    for (MY_SIZE i = 0; i < num; ++i) {
      for (MY_SIZE c = 0; c < num_of_colours; ++c) {
        stepCPUCellCentredOMP<UserFunc>(partition[c], temp);
      }
      // Copy back from temp
      TIMER_TOGGLE(t);
      #pragma omp parallel for
      for (MY_SIZE e = 0; e < point_weights.getSize() * point_weights.getDim();
           ++e) {
        point_weights.operator[]<DataType>(e) = temp.operator[]<DataType>(e);
      }
      TIMER_TOGGLE(t);
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
    ScotchReorder reorder(mesh.numPoints(), mesh.numCells(), mesh.cell_to_node);
    std::vector<SCOTCH_Num> point_permutation = reorder.reorder();
    std::vector<MY_SIZE> inverse_permutation = mesh.reorder(point_permutation);
    reorderData<SOA>(point_weights, point_permutation);
    reorderDataInverse<true>(cell_weights, inverse_permutation);
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
    std::vector<MY_SIZE> inverse_permutation =
        mesh.reorderToPartition(partition_vector);
    reorderDataInverse<true>(cell_weights, inverse_permutation);
  }

  void renumberPoints() {
    std::vector<MY_SIZE> permutation = mesh.getPointRenumberingPermutation2(
        mesh.getPointToPartition(partition_vector));
    mesh.renumberPoints(permutation);
    reorderData<SOA>(point_weights, permutation);
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
