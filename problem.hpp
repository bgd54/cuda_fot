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

template <bool SOA, class IndirectDataParams, class DirectDataParams,
          unsigned... MeshDims>
class Problem {
protected:
  using indirect_type_generator = generate_data_set_t<IndirectDataParams>;
  using direct_type_generator = generate_data_set_t<DirectDataParams>;

public:
  using indirect_data_set_t = typename indirect_type_generator::data_set_type;
  using direct_data_set_t = typename direct_type_generator::data_set_type;
  using written_indirect_data_set_t =
      typename indirect_type_generator::written_data_set_type;
  // using written_direct_data_set_t =
  //    typename indirect_type_generator::written_data_set_type;
  using written_indirect_data_set_mapping =
      typename indirect_type_generator::written_data_set_mapping;
  static constexpr unsigned NUM_MAPPINGS = Mesh<MeshDims...>::NUM_MAPPINGS;

  Mesh<MeshDims...> mesh;
  direct_data_set_t cell_weights;
  indirect_data_set_t point_weights;
  const MY_SIZE block_size; // GPU block size
  std::vector<MY_SIZE> partition_vector{};

protected:
  Problem(Mesh<MeshDims...> &&mesh_, MY_SIZE block_size_)
      : mesh{mesh_},
        cell_weights{initMapping<direct_data_set_t>(mesh.numCells())},
        point_weights{
            initTupleFromArray<indirect_data_set_t>(mesh.numPoints())},
        block_size{block_size_} {}

public:
  Problem(MY_SIZE num_cells,
          const std::array<MY_SIZE, NUM_MAPPINGS> &num_points,
          const std::array<std::istream *, NUM_MAPPINGS> &mesh_is,
          MY_SIZE _block_size = DEFAULT_BLOCK_SIZE)
      : mesh(num_cells, num_points, mesh_is),
        cell_weights(initMapping<direct_data_set_t>(num_cells)),
        point_weights(initTupleFromArray<indirect_data_set_t>(num_points)) {}

  ~Problem() {}

  void loopGPUCellCentred(MY_SIZE num, MY_SIZE reset_every = 0);
  void loopGPUHierarchical(MY_SIZE num, MY_SIZE reset_every = 0);

  template <class LoopBodyFunc>
  void stepCPUCellCentred(written_indirect_data_set_t &point_weights_out) {
    for (MY_SIZE cell_ind = 0; cell_ind < mesh.numCells(); ++cell_ind) {
      static_assert(SOA == false, "SOA is not yet supported");
      call_with_pointers<LoopBodyFunc, written_indirect_data_set_mapping>(
          mesh.mappings, cell_ind, point_weights, point_weights_out,
          cell_weights);
      // for (MY_SIZE offset = 0; offset < MESH_DIM; ++offset) {
      //  MY_SIZE ind_left_base =
      //      mesh.cell_to_node[mesh.cell_to_node.dim * cell_ind_base + offset];
      //  MY_SIZE ind_right_base =
      //      mesh.cell_to_node[mesh.cell_to_node.dim * cell_ind_base +
      //                        (offset == MESH_DIM - 1 ? 0 : offset + 1)];
      //  MY_SIZE w_ind_left = 0, w_ind_right = 0;
      //  for (MY_SIZE d = 0; d < PointDim; ++d) {
      //    w_ind_left = index<PointDim, SOA>(mesh.numPoints(), ind_left_base,
      //    d);
      //    w_ind_right =
      //        index<PointDim, SOA>(mesh.numPoints(), ind_right_base, d);
      //    MY_SIZE cell_d = CellDim == 1 ? 0 : d;

      //    MY_SIZE cell_weight_ind =
      //        index<CellDim, true>(mesh.numCells(), cell_ind_base, cell_d);
      //    point_weights[w_ind_right] +=
      //        cell_weights[cell_weight_ind] * temp[w_ind_left];
      //    point_weights[w_ind_left] +=
      //        cell_weights[cell_weight_ind] * temp[w_ind_right];
      //  }
      //}
    }
  }

  template <class LoopBodyFunc> void loopCPUCellCentred(MY_SIZE num) { /*{{{*/
    // DataType *temp = (DataType *)malloc(sizeof(DataType) * mesh.numPoints() *
    //                                    point_weights.dim);
    //written_indirect_data_set_t point_weights_out(point_weights);
    // TODO: can't simply copy these
    TIMER_START(t);
    TIMER_TOGGLE(t);
    for (MY_SIZE i = 0; i < num; ++i) {
      TIMER_TOGGLE(t);
      stepCPUCellCentred(point_weights_out);
      // TODO copy
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
    static_assert(NUM_MAPPINGS > 0, "Assuming here that NUM_MAPPINGS > 0");
    ScotchReorder reorder(mesh.numPoints(0), mesh.numCells(),
                          std::get<0>(mesh.mappings));
    std::vector<SCOTCH_Num> permutation = reorder.reorder();
    mesh.template reorder<>(permutation, cell_data, point_data);
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
    for_each(cell_weights, ReorderDataSet{inverse_permutation});
  }

  void renumberPointsToPartition() {
    for_each(mappings, RenumberPointsToPartition<MeshDims...>{mesh});
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

protected:
  class ReorderDataSet {
    const std::vector<MY_SIZE> &inverse_permutation;

  public:
    ReorderDataSet(const std::vector<MY_SIZE> &inverse_permutation_)
        : inverse_permutation{inverse_permutation_} {}

    template <unsigned, unsigned Dim, class DataType>
    void operator()(data_t<DataType, Dim> &mapping) {
      reorderDataInverse<Dim, false>(mapping, inverse_permutation);
    }
  };

  template <class... MeshDims> class RenumberPointsToPartition {
    const Mesh<MeshDims...> &mesh;

  public:
    RenumberPointsToPartition(const Mesh<MeshDims...> &mesh_) : mesh{mesh_} {}
    template <unsigned MapInd, unsigned Dim, class DataType>
    void operator()(data_t<DataType, Dim> &data) {
      std::vector<MY_SIZE> permutation = mesh.getPointRenumberingPermutation2(
          mesh.getPointToPartition<MapInd>());
      mesh.renumberPoints<MapInd>(permutation);
      reorderData<Dim, SOA, DataType, MY_SIZE>(data, permutation);
    }
  };

};

#endif /* end of include guard: PROBLEM_HPP_CGW3IDMV */
// vim:set et sw=2 ts=2 fdm=marker:
