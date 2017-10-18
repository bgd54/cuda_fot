#ifndef STRUCTURED_PROBLEM_H
#define STRUCTURED_PROBLEM_H

#include "problem.hpp"

template <unsigned PointDim, unsigned CellDim, bool SOA, typename DataType>
class StructuredProblem : public Problem<SOA, DataType> {
public:
  using Base = Problem<SOA, DataType>;
  StructuredProblem(
      MY_SIZE N, MY_SIZE M,
      std::pair<MY_SIZE, MY_SIZE> block_dims = {0, DEFAULT_BLOCK_SIZE},
      bool use_coordinates = false)
      : StructuredProblem({N, M}, block_dims, use_coordinates) {}

  StructuredProblem(
      const std::vector<MY_SIZE> &grid_dim,
      std::pair<MY_SIZE, MY_SIZE> block_dims = {0, DEFAULT_BLOCK_SIZE},
      bool use_coordinates = false)
      : Base(Grid(grid_dim, Base::MESH_DIM, block_dims, use_coordinates),
             PointDim, CellDim, calculateBlockSize(block_dims)) {
    reset();
  }

  ~StructuredProblem() {}

  void reset() {
    for (DataType *it = this->point_weights.template begin<DataType>();
         it != this->point_weights.template end<DataType>(); ++it) {
      DataType &w = *it;
      w = DataType(rand() % 10000) / 5000.f;
      w *= 0.001;
    }
    for (DataType *it = this->cell_weights.template begin<DataType>();
         it != this->cell_weights.template end<DataType>(); ++it) {
      DataType &weight = *it;
      weight = DataType(rand() % 10000 + 1) / 5000.0;
      weight *= 0.001;
    }
  }

  static MY_SIZE calculateBlockSize(std::pair<MY_SIZE, MY_SIZE> block_dims) {
    if (Base::MESH_DIM == 2) {
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
