#ifndef STRUCTURED_PROBLEM_H
#define STRUCTURED_PROBLEM_H

#include "problem.hpp"

template <bool SOA, unsigned MeshDim, unsigned CellDim, unsigned PointDim,
          class DataType>
class StructuredProblem : Problem<SOA, DSetParam<Param<PointDim, DataType>>,
                                  DSetParam<CellDim, DataType>, MeshDim> {
public:
  using Base = Problem<SOA, DSetParam<Param<PointDim, DataType>,DSetParam<CellDim, DataType>, MeshDim>;
  StructuredProblem(MY_SIZE N, MY_SIZE M,
          std::pair<MY_SIZE, MY_SIZE> block_dims = {0, DEFAULT_BLOCK_SIZE},
          bool use_coordinates = false)
      : StructuredProblem({N, M}, block_dims, use_coordinates) {}

  StructuredProblem(const std::vector<MY_SIZE> &grid_dim,
          std::pair<MY_SIZE, MY_SIZE> block_dims = {0, DEFAULT_BLOCK_SIZE},
          bool use_coordinates = false)
      : Base(Grid<MeshDim>(grid_dim, block_dims, use_coordinates),
          calculateBlockSize(block_dims)) {
    reset();
  }

  ~StructuredProblem () {}
  
  void reset() {
    for (DataType &w : std::get<0>(this->point_weights)) {
      w = DataType(rand() % 10000) / 5000.f;
      w *= 0.001;
    }
    for (DataType &weight : std::get<0>(this->cell_weights)) {
      weight = DataType(rand() % 10000 + 1) / 5000.0;
      weight *= 0.001;
    }
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

};

#endif /* STRUCTURED_PROBLEM_H */
