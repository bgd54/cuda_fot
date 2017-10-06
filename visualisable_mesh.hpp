#ifndef VISUALISABLE_MESH_HPP_LSIXY4T6
#define VISUALISABLE_MESH_HPP_LSIXY4T6

#include "mesh.hpp"

template <unsigned MeshDim> class VisualisableMesh : public Mesh<MeshDim> {
public:
  using Base = Mesh<MeshDim>;
  using Base::cell_to_node;
  using Base::reorder;
  using Base::numPoints;
  using Base::numCells;

  data_t<float, 3> point_coordinates;

  VisualisableMesh(std::istream &is, std::istream *coord_is = nullptr)
      : Mesh<MeshDim>(is) {
    if (coord_is != nullptr) {
      if (!(*coord_is)) {
        throw InvalidInputFile{"coordinate input", 0};
      }
      for (MY_SIZE i = 0; i < numPoints(); ++i) {
        *coord_is >> point_coordinates[3 * i + 0] >>
            point_coordinates[3 * i + 1] >> point_coordinates[3 * i + 2];
        if (!(*coord_is)) {
          throw InvalidInputFile{"coordinate input", i};
        }
      }
    }
  }

  VisualisableMesh(const VisualisableMesh &) = delete;
  VisualisableMesh &operator=(const VisualisableMesh &) = delete;

  VisualisableMesh(VisualisableMesh &&other)
      : Mesh<MeshDim>(std::move(other)),
        point_coordinates{std::move(other.point_coordinates)} {}

  VisualisableMesh &operator=(VisualisableMesh &&rhs) {
    Mesh<MeshDim>::operator=(std::move(rhs));
    std::swap(point_coordinates, rhs.point_coordinates);
    return *this;
  }

  void writeCoordinates(std::ostream &os) const {
    assert(point_coordinates.getSize() == numPoints());
    for (MY_SIZE i = 0; i < numPoints(); ++i) {
      os << point_coordinates[3 * i + 0] << " " << point_coordinates[3 * i + 1]
         << " " << point_coordinates[3 * i + 2] << std::endl;
    }
  }

  template <typename UnsignedType, typename DataType = float,
            unsigned DataDim = 1, unsigned CellDim = 1, bool SOA = false>
  void reorder(const std::vector<UnsignedType> &point_permutation,
               data_t<DataType, CellDim> *cell_data = nullptr,
               data_t<DataType, DataDim> *point_data = nullptr) {
    Mesh<MeshDim>::template reorder<UnsignedType, DataType, DataDim, CellDim,
                                    SOA>(point_permutation, cell_data,
                                         point_data);
    if (point_coordinates.getSize() > 0) {
      reorderData<3, false, float, UnsignedType>(point_coordinates,
                                                 point_permutation);
    }
  }

  std::vector<MY_SIZE> renumberPoints(const std::vector<MY_SIZE> &permutation) {
    Mesh<MeshDim>::renumberPoints(permutation);
    if (point_coordinates.getSize() > 0) {
      reorderData<3, false, float, MY_SIZE>(point_coordinates, permutation);
    }
    return permutation;
  }

protected:
  VisualisableMesh(MY_SIZE _num_points, MY_SIZE _num_cells,
                   bool use_coordinates = false,
                   const MY_SIZE *_cell_to_node = nullptr)
      : Mesh<MeshDim>(_num_points, _num_cells, _cell_to_node),
        point_coordinates(use_coordinates ? _num_points : 0) {}
};

// vim:set et sw=2 ts=2 fdm=marker:
#endif /* end of include guard: VISUALISABLE_MESH_HPP_LSIXY4T6 */
