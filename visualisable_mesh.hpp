#ifndef VISUALISABLE_MESH_HPP_LSIXY4T6
#define VISUALISABLE_MESH_HPP_LSIXY4T6

#include "mesh.hpp"

class VisualisableMesh : public Mesh {
public:
  data_t point_coordinates;

  VisualisableMesh(std::istream &is, unsigned mesh_dim,
                   std::istream *coord_is = nullptr)
      : Mesh({&is}, {mesh_dim}), point_coordinates(data_t::create<float>(
                                     coord_is ? numPoints(0) : 0, 3)) {
    if (coord_is != nullptr) {
      if (!(*coord_is)) {
        throw InvalidInputFile{"coordinate input", 0, 0};
      }
      for (MY_SIZE i = 0; i < numPoints(0); ++i) {
        *coord_is >> point_coordinates.operator[]<float>(3 * i + 0) >>
            point_coordinates.operator[]<float>(3 * i + 1) >>
            point_coordinates.operator[]<float>(3 * i + 2);
        if (!(*coord_is)) {
          throw InvalidInputFile{"coordinate input", 0, i};
        }
      }
    }
  }

  VisualisableMesh(const VisualisableMesh &) = delete;
  VisualisableMesh &operator=(const VisualisableMesh &) = delete;

  VisualisableMesh(VisualisableMesh &&other)
      : Mesh(std::move(other)),
        point_coordinates{std::move(other.point_coordinates)} {}

  VisualisableMesh &operator=(VisualisableMesh &&rhs) {
    Mesh::operator=(std::move(rhs));
    std::swap(point_coordinates, rhs.point_coordinates);
    return *this;
  }

  void writeCoordinates(std::ostream &os) const {
    assert(point_coordinates.getSize() == numPoints(0));
    for (MY_SIZE i = 0; i < numPoints(0); ++i) {
      os << point_coordinates.operator[]<float>(3 * i + 0) << " "
         << point_coordinates.operator[]<float>(3 * i + 1) << " "
         << point_coordinates[3 * i + 2] << std::endl;
    }
  }

  template <typename UnsignedType>
  void reorder(const std::vector<UnsignedType> &point_permutation) {
    if (point_coordinates.getSize() > 0) {
      reorderData<false>(point_coordinates, point_permutation);
    }
    return Mesh::reorder<UnsignedType>(point_permutation);
  }

  std::vector<MY_SIZE> renumberPoints(const std::vector<MY_SIZE> &permutation) {
    Mesh::renumberPoints(permutation, 0);
    if (point_coordinates.getSize() > 0) {
      reorderData<false>(point_coordinates, permutation);
    }
    return permutation;
  }

protected:
  VisualisableMesh(MY_SIZE _num_points, MY_SIZE _num_cells, MY_SIZE mesh_dim,
                   bool use_coordinates = false,
                   const MY_SIZE *_cell_to_node = nullptr)
      : Mesh({_num_points}, _num_cells, {mesh_dim},
             _cell_to_node == nullptr
                 ? std::vector<const MY_SIZE *>{}
                 : std::vector<const MY_SIZE *>{_cell_to_node}),
        point_coordinates(
            data_t::create<float>(use_coordinates ? _num_points : 0, 3)) {}
};

// vim:set et sts=2 sw=2 ts=2 fdm=marker:
#endif /* end of include guard: VISUALISABLE_MESH_HPP_LSIXY4T6 */
