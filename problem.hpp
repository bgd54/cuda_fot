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

  Mesh mesh;
  std::vector<data_t> point_weights;
  std::vector<data_t> cell_weights;
  const MY_SIZE block_size; // GPU block size
  std::vector<MY_SIZE> partition_vector;

  /* ctor/dtor {{{1 */
protected:
  Problem(Mesh &&mesh_,
          const std::vector<std::pair<MY_SIZE, unsigned>> &point_data_params,
          const std::vector<std::pair<MY_SIZE, unsigned>> &cell_data_params,
          MY_SIZE block_size_)
      : mesh(std::move(mesh_)), point_weights{}, cell_weights{},
        block_size{block_size_} {
    assert(point_data_params.size() == mesh.numMappings());
    for (unsigned mapping_ind = 0; mapping_ind < mesh.numMappings();
         ++mapping_ind) {
      point_weights.emplace_back(mesh.numPoints(mapping_ind),
                                 point_data_params[mapping_ind].first,
                                 point_data_params[mapping_ind].second);
    }
    for (unsigned i = 0; i < cell_data_params.size(); ++i) {
      cell_weights.emplace_back(mesh.numCells(), cell_data_params[i].first,
                                cell_data_params[i].second);
    }
  }

public:
  Problem(const std::vector<std::istream *> &mesh_is,
          const std::vector<MY_SIZE> &mesh_dim,
          const std::vector<std::pair<MY_SIZE, unsigned>> &point_data_params,
          const std::vector<std::pair<MY_SIZE, unsigned>> &cell_data_params,
          MY_SIZE _block_size = DEFAULT_BLOCK_SIZE)
      : mesh(mesh_is, mesh_dim), point_weights{}, cell_weights{},
        block_size{_block_size} {
    assert(point_data_params.size() == mesh.numMappings());
    for (unsigned mapping_ind = 0; mapping_ind < mesh.numMappings();
         ++mapping_ind) {
      point_weights.emplace_back(mesh.numPoints(mapping_ind),
                                 point_data_params[mapping_ind].first,
                                 point_data_params[mapping_ind].second);
    }
    for (unsigned i = 0; i < cell_data_params.size(); ++i) {
      cell_weights.emplace_back(mesh.numCells(), cell_data_params[i].first,
                                cell_data_params[i].second);
    }
  }

  ~Problem() {}

  Problem(Problem &&other)
      : mesh(std::move(other.mesh)),
        point_weights(std::move(other.point_weights)),
        cell_weights(std::move(other.cell_weights)),
        block_size(other.block_size),
        partition_vector(std::move(other.partition_vector)) {}
  /* 1}}} */

  template <class UserFunc> void loopGPUCellCentred(MY_SIZE num);
  template <class UserFunc> void loopGPUHierarchical(MY_SIZE num);

  template <class UserFunc> void stepCPUCellCentred(char *temp) const { /*{{{*/
    std::vector<const void *> point_data(point_weights.size());
    auto get_pointer = [](const data_t &a) { return a.cbegin(); };
    std::transform(point_weights.begin(), point_weights.end(),
                   point_data.begin(), get_pointer);
    std::vector<const void *> cell_data(cell_weights.size());
    std::transform(cell_weights.begin(), cell_weights.end(), cell_data.begin(),
                   get_pointer);
    std::vector<const MY_SIZE *> cell_to_node(mesh.numMappings());
    std::transform(mesh.cell_to_node.begin(), mesh.cell_to_node.end(),
                   cell_to_node.begin(),
                   [](const data_t &a) { return a.cbegin<MY_SIZE>(); });
    for (MY_SIZE cell_ind_base = 0; cell_ind_base < mesh.numCells();
         ++cell_ind_base) {
      UserFunc::template call<SOA>(point_data.data(), temp, cell_data.data(),
                                   cell_to_node.data(), cell_ind_base,
                                   mesh.numPoints(), mesh.numCells());
    }
  } /*}}}*/

  template <class UserFunc> void loopCPUCellCentred(MY_SIZE num) { /*{{{*/
    assert(point_weights.size() > 0);
    char *temp = (char *)malloc(point_weights[0].getTypeSize() *
                                mesh.numPoints(0) * point_weights[0].getDim());
    TIMER_START(t);
    TIMER_TOGGLE(t);
    std::copy(point_weights[0].begin(), point_weights[0].end(), temp);
    TIMER_TOGGLE(t);
    for (MY_SIZE i = 0; i < num; ++i) {
      stepCPUCellCentred<UserFunc>(temp);
      TIMER_TOGGLE(t);
      std::copy_n(temp, point_weights[0].getTypeSize() * mesh.numPoints(0) *
                            point_weights[0].getDim(),
                  point_weights[0].begin());
      TIMER_TOGGLE(t);
    }
    PRINT_BANDWIDTH(t, "loopCPUCellCentred", calcDataSize() * num);
    free(temp);
  } /*}}}*/

  template <class UserFunc>
  void stepCPUCellCentredOMP(const std::vector<MY_SIZE> &inds,
                             data_t &out) const { /*{{{*/
    std::vector<const void *> point_data(point_weights.size());
    auto get_pointer = [](const data_t &a) { return a.cbegin(); };
    std::transform(point_weights.begin(), point_weights.end(),
                   point_data.begin(), get_pointer);
    std::vector<const void *> cell_data(cell_weights.size());
    std::transform(cell_weights.begin(), cell_weights.end(), cell_data.begin(),
                   get_pointer);
    std::vector<const MY_SIZE *> cell_to_node(mesh.numMappings());
    std::transform(mesh.cell_to_node.begin(), mesh.cell_to_node.end(),
                   cell_to_node.begin(),
                   [](const data_t &a) { return a.cbegin<MY_SIZE>(); });
    #pragma omp parallel for
    for (MY_SIZE i = 0; i < inds.size(); ++i) {
      MY_SIZE ind = inds[i];
      UserFunc::template call<SOA>(point_data.data(), out.begin(),
                                   cell_data.data(), cell_to_node.data(), ind,
                                   mesh.numPoints(), mesh.numCells());
    }
  } /*}}}*/

  template <class UserFunc> void loopCPUCellCentredOMP(MY_SIZE num) { /*{{{*/
    data_t temp(point_weights[0].getSize(), point_weights[0].getDim(),
                point_weights[0].getTypeSize());
    std::vector<std::vector<MY_SIZE>> partition = mesh.colourCells();
    MY_SIZE num_of_colours = partition.size();
    TIMER_START(t);
    TIMER_TOGGLE(t);
    // Init temp
    #pragma omp parallel for
    for (MY_SIZE e = 0;
         e < point_weights[0].getTypeSize() * point_weights[0].getSize() *
                 point_weights[0].getDim();
         ++e) {
      *(temp.begin() + e) = *(point_weights[0].begin() + e);
    }
    TIMER_TOGGLE(t);
    for (MY_SIZE i = 0; i < num; ++i) {
      for (MY_SIZE c = 0; c < num_of_colours; ++c) {
        stepCPUCellCentredOMP<UserFunc>(partition[c], temp);
      }
      TIMER_TOGGLE(t);
      // Copy back from temp
      #pragma omp parallel for
      for (MY_SIZE e = 0;
           e < point_weights[0].getTypeSize() * point_weights[0].getSize() *
                   point_weights[0].getDim();
           ++e) {
        *(point_weights[0].begin() + e) = *(temp.begin() + e);
      }
      TIMER_TOGGLE(t);
    }
    PRINT_BANDWIDTH(t, "loopCPUCellCentredOMP", calcDataSize() * num);
  } /*}}}*/

  void reorder() {
    assert(mesh.numMappings() >= 1);
    ScotchReorder reorder(mesh.numPoints(0), mesh.numCells(),
                          mesh.cell_to_node[0]);
    std::vector<SCOTCH_Num> point_permutation = reorder.reorder();
    std::vector<MY_SIZE> inverse_permutation = mesh.reorder(point_permutation);
    reorderData<SOA>(point_weights[0], point_permutation);
    for (unsigned mapping_ind = 1; mapping_ind < mesh.numMappings();
         ++mapping_ind) {
      reorderData<SOA>(
          point_weights[mapping_ind],
          mesh.renumberPoints(mesh.getPointRenumberingPermutation(mapping_ind),
                              mapping_ind));
    }
    for (data_t &cw : cell_weights) {
      reorderDataInverse<true>(cw, inverse_permutation);
    }
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
    for (data_t &cw : cell_weights) {
      reorderDataInverse<true>(cw, inverse_permutation);
    }
  }

  void renumberPoints() {
    for (unsigned mapping_ind = 0; mapping_ind < mesh.numMappings();
         ++mapping_ind) {
      std::vector<MY_SIZE> permutation = mesh.getPointRenumberingPermutation2(
          mesh.getPointToPartition(partition_vector, mapping_ind));
      mesh.renumberPoints(permutation, mapping_ind);
      reorderData<SOA>(point_weights[mapping_ind], permutation);
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
      throw InvalidInputFile{"partition input", 0, 0};
    }
    MY_SIZE read_block_size;
    partition_is >> read_block_size;
    if (!partition_is) {
      throw InvalidInputFile{"partition input", 0, 1};
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
        throw InvalidInputFile{"partition input", 0, i + 1};
      }
    }
  }

  template <class DataType>
  void readPointData(std::istream &is, unsigned mapping_ind) {
    assert(mapping_ind < mesh.numMappings());
    if (!is) {
      throw InvalidInputFile{"point data input", mapping_ind, 0};
    }
    for (MY_SIZE i = 0; i < mesh.numPoints(mapping_ind); ++i) {
      for (MY_SIZE j = 0; j < point_weights[mapping_ind].getDim(); ++j) {
        is >> point_weights[mapping_ind].operator[]<DataType>(
                  index<SOA>(point_weights[mapping_ind].getSize(), i,
                             point_weights[mapping_ind].getDim(), j));
      }
      if (!is) {
        throw InvalidInputFile{"point data input", mapping_ind, i};
      }
    }
  }

  template <class DataType> void readCellData(std::istream &is, unsigned ind) {
    assert(ind < cell_weights.size());
    if (!is) {
      throw InvalidInputFile{"cell data input", ind, 0};
    }
    for (MY_SIZE i = 0; i < mesh.numCells(); ++i) {
      for (MY_SIZE j = 0; j < cell_weights[ind].getDim(); ++j) {
        is >> cell_weights[ind].operator[]<DataType>(
                  j *cell_weights[ind].getSize() + i);
      }
      if (!is) {
        throw InvalidInputFile{"cell data input", ind, i};
      }
    }
  }

  double calcDataSize() const {
    double result = 0;
    for (unsigned mapping_ind = 0; mapping_ind < mesh.numMappings();
         ++mapping_ind) {
      result += point_weights[mapping_ind].getTypeSize() *
                point_weights[mapping_ind].getDim() *
                point_weights[mapping_ind].getSize();
      result += mesh.cell_to_node[mapping_ind].getTypeSize() *
                mesh.cell_to_node[mapping_ind].getDim() * mesh.numCells();
    }
    for (unsigned i = 0; i < cell_weights.size(); ++i) {
      result += cell_weights[i].getTypeSize() * cell_weights[i].getDim() *
                cell_weights[i].getSize();
    }
    result += point_weights[0].getTypeSize() * point_weights[0].getDim() *
              point_weights[0].getSize();
    return result;
  }
};

#endif /* end of include guard: PROBLEM_HPP_CGW3IDMV */
// vim:set et sw=2 ts=2 fdm=marker:
