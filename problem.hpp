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

template <bool SOA = false> struct Problem {
  static constexpr unsigned MESH_DIM = MESH_DIM_MACRO;

  Mesh mesh;
  data_t cell_weights;
  data_t point_weights;
  const MY_SIZE block_size; // GPU block size
  std::vector<MY_SIZE> partition_vector;

  /* ctor/dtor {{{1 */
protected:
  Problem(Mesh &&mesh_, MY_SIZE point_dim, MY_SIZE cell_dim, unsigned type_size,
          MY_SIZE block_size_)
      : mesh(std::move(mesh_)),
        cell_weights(mesh.numCells(), cell_dim, type_size),
        point_weights(mesh.numPoints(), point_dim, type_size),
        block_size{block_size_} {}

public:
  Problem(std::istream &mesh_is, MY_SIZE point_dim, MY_SIZE cell_dim,
          unsigned type_size, MY_SIZE _block_size = DEFAULT_BLOCK_SIZE)
      : mesh(mesh_is, MESH_DIM),
        cell_weights(mesh.numCells(), cell_dim, type_size),
        point_weights(mesh.numPoints(), point_dim, type_size),
        block_size{_block_size} {}

  ~Problem() {}

  Problem(Problem &&other)
      : mesh(std::move(other.mesh)),
        cell_weights(std::move(other.cell_weights)),
        point_weights(std::move(other.point_weights)),
        block_size(other.block_size),
        partition_vector(std::move(other.partition_vector)) {}
  /* 1}}} */

  template <class UserFunc> void loopGPUCellCentred(MY_SIZE num);
  template <class UserFunc> void loopGPUHierarchical(MY_SIZE num);

  template <class UserFunc> void stepCPUCellCentred(char *temp) { /*{{{*/
    for (MY_SIZE cell_ind_base = 0; cell_ind_base < mesh.numCells();
         ++cell_ind_base) {
      UserFunc::template call<SOA>(
          point_weights.cbegin(), temp, cell_weights.cbegin(),
          mesh.cell_to_node.cbegin<MY_SIZE>(), cell_ind_base, mesh.numPoints(),
          mesh.numCells());
    }
  } /*}}}*/

  template <class UserFunc> void loopCPUCellCentred(MY_SIZE num) { /*{{{*/
    char *temp = (char *)malloc(point_weights.getTypeSize() * mesh.numPoints() *
                                point_weights.getDim());
    TIMER_START(t);
    TIMER_TOGGLE(t);
    std::copy(point_weights.begin(), point_weights.end(), temp);
    TIMER_TOGGLE(t);
    for (MY_SIZE i = 0; i < num; ++i) {
      stepCPUCellCentred<UserFunc>(temp);
      TIMER_TOGGLE(t);
      std::copy_n(temp, point_weights.getTypeSize() * mesh.numPoints() *
                            point_weights.getDim(),
                  point_weights.begin());
      TIMER_TOGGLE(t);
    }
    PRINT_BANDWIDTH(t, "loopCPUCellCentred",
                    (point_weights.getTypeSize() *
                         (2.0 * point_weights.getDim() * mesh.numPoints() +
                          cell_weights.getDim() * mesh.numCells()) +
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
      UserFunc::template call<SOA>(point_weights.cbegin(), out.begin(),
                                   cell_weights.cbegin(),
                                   mesh.cell_to_node.cbegin<MY_SIZE>(), ind,
                                   mesh.numPoints(), mesh.numCells());
    }
  } /*}}}*/

  template <class UserFunc> void loopCPUCellCentredOMP(MY_SIZE num) { /*{{{*/
    data_t temp(point_weights.getSize(), point_weights.getDim(),
                point_weights.getTypeSize());
    std::vector<std::vector<MY_SIZE>> partition = mesh.colourCells();
    MY_SIZE num_of_colours = partition.size();
    TIMER_START(t);
    // Init temp
    TIMER_TOGGLE(t);
    #pragma omp parallel for
    for (MY_SIZE e = 0;
         e < point_weights.getTypeSize() * point_weights.getSize() *
                 point_weights.getDim();
         ++e) {
      *(temp.begin() + e) = *(point_weights.begin() + e);
    }
    TIMER_TOGGLE(t);
    for (MY_SIZE i = 0; i < num; ++i) {
      for (MY_SIZE c = 0; c < num_of_colours; ++c) {
        stepCPUCellCentredOMP<UserFunc>(partition[c], temp);
      }
      // Copy back from temp
      TIMER_TOGGLE(t);
      #pragma omp parallel for
      for (MY_SIZE e = 0;
           e < point_weights.getTypeSize() * point_weights.getSize() *
                   point_weights.getDim();
           ++e) {
        *(point_weights.begin() + e) = *(temp.begin() + e);
      }
      TIMER_TOGGLE(t);
    }
    PRINT_BANDWIDTH(t, "loopCPUCellCentredOMP",
                    (point_weights.getTypeSize() *
                         (2.0 * point_weights.getDim() * mesh.numPoints() +
                          cell_weights.getDim() * mesh.numCells()) +
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

  template <class DataType> void readPointData(std::istream &is) {
    if (!is) {
      throw InvalidInputFile{"point data input", 0};
    }
    for (MY_SIZE i = 0; i < mesh.numPoints(); ++i) {
      for (MY_SIZE j = 0; j < point_weights.getDim(); ++j) {
        is >> point_weights.operator[]<DataType>(index<SOA>(
                  point_weights.getSize(), i, point_weights.getDim(), j));
      }
      if (!is) {
        throw InvalidInputFile{"point data input", i};
      }
    }
  }

  template <class DataType> void readCellData(std::istream &is) {
    if (!is) {
      throw InvalidInputFile{"cell data input", 0};
    }
    for (MY_SIZE i = 0; i < mesh.numCells(); ++i) {
      for (MY_SIZE j = 0; j < cell_weights.getDim(); ++j) {
        is >> cell_weights.operator[]<DataType>(j *cell_weights.getSize() + i);
      }
      if (!is) {
        throw InvalidInputFile{"cell data input", i};
      }
    }
  }
};

#endif /* end of include guard: PROBLEM_HPP_CGW3IDMV */
// vim:set et sw=2 ts=2 fdm=marker:
