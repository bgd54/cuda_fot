#ifndef REORDER_HPP_IGDYRZTN
#define REORDER_HPP_IGDYRZTN

#include "data_t.hpp"
#include <algorithm>
#include <cassert>
#include <iterator>
#include <limits>
#include <map>
#include <scotch.h>
#include <vector>

struct ScotchError {
  int errorCode;
};

/*
 * Throw if the Scotch command `cmd` returns an error.
 */
#define SCOTCH_THROW(cmd)                                                      \
  do {                                                                         \
    int errorCode = cmd;                                                       \
    if (errorCode) {                                                           \
      /* Exterminate! Exterminate! */                                          \
      throw ScotchError{errorCode};                                            \
    }                                                                          \
  } while (0)

/**
 * A graph in compressed sparse row format
 */
/* GraphCSR {{{1 */
template <class UnsignedType> struct GraphCSR {
  const UnsignedType num_points, num_cells;

  /**
   * Creates a GraphCSR point-to-point mapping from a mesh.
   */
  explicit GraphCSR(MY_SIZE _num_points, MY_SIZE _num_cells,
                    const data_t &cell_to_node)
      : num_points(_num_points), num_cells(_num_cells) {
    // static_assert(std::size_t(std::numeric_limits<UnsignedType>::max()) >=
    // std::size_t(std::numeric_limits<MY_SIZE>::max()),
    //"GraphCSR: UnsignedType too small.");
    assert(num_cells > 0);
    assert(cell_to_node.getTypeSize() == sizeof(MY_SIZE));
    assert(cell_to_node.getSize() < std::numeric_limits<UnsignedType>::max());
    assert(static_cast<MY_SIZE>(num_cells) == cell_to_node.getSize());
    point_indices = new UnsignedType[num_points + 1];
    UnsignedType point_ind = 0;
    point_indices[0] = 0;
    std::multimap<UnsignedType, MY_SIZE> incidence =
        getPointToCell(cell_to_node);
    const unsigned mesh_dim = cell_to_node.getDim();
    for (const auto incidence_pair : incidence) {
      UnsignedType current_point = incidence_pair.first;
      MY_SIZE current_cell = incidence_pair.second;
      while (current_point != point_ind) {
        point_indices[++point_ind] = cell_endpoints.size();
      }
      bool found_current_point = false;
      for (MY_SIZE i = 0; i < mesh_dim; ++i) {
        UnsignedType other_point =
            cell_to_node.operator[]<MY_SIZE>(mesh_dim *current_cell + i);
        if (other_point != current_point) {
          if (static_cast<MY_SIZE>(point_indices[current_point]) ==
                  cell_endpoints.size() ||
              std::find(cell_endpoints.begin() + point_indices[current_point],
                        cell_endpoints.end(),
                        other_point) == cell_endpoints.end()) {
            cell_endpoints.push_back(other_point);
          }
        } else {
          /* assert(!found_current_point); */
          found_current_point = true;
        }
      }
      assert(found_current_point);
    }
    while (point_ind != num_points) {
      point_indices[++point_ind] = cell_endpoints.size();
    }
  }

  ~GraphCSR() { delete[] point_indices; }

  GraphCSR(const GraphCSR &other) = delete;
  GraphCSR &operator=(const GraphCSR &rhs) = delete;

  GraphCSR(GraphCSR &&other)
      : num_points{other.num_points}, num_cells{other.num_cells} {
    std::swap(point_indices, other.point_indices);
    std::swap(cell_endpoints, other.cell_endpoints);
  }

  const UnsignedType *pointIndices() const { return point_indices; }
  UnsignedType *pointIndices() { return point_indices; }

  const UnsignedType *cellEndpoints() const { return cell_endpoints.data(); }
  UnsignedType *cellEndpoints() { return cell_endpoints.data(); }

  UnsignedType numArcs() const { return cell_endpoints.size(); }

  /**
   * returns a multimap: from points to incident cells
   *
   * data in cell_to_node must be of type MY_SIZE
   */
  template <class Iterator>
  static std::multimap<UnsignedType, MY_SIZE>
  getPointToCell(Iterator begin, Iterator end, MY_SIZE mesh_dim) {
    std::multimap<UnsignedType, MY_SIZE> point_to_cell;
    for (MY_SIZE i = 0; begin != end; ++i, ++begin) {
      point_to_cell.insert(std::make_pair(*begin, i / mesh_dim));
    }
    return point_to_cell;
  }

  static std::multimap<UnsignedType, MY_SIZE>
  getPointToCell(const data_t &cell_to_node) {
    assert(cell_to_node.getTypeSize() == sizeof(MY_SIZE));
    const unsigned mesh_dim = cell_to_node.getDim();
    return getPointToCell(cell_to_node.cbegin<MY_SIZE>(),
                          cell_to_node.cend<MY_SIZE>(), mesh_dim);
  }

  void reorderInverse(const std::vector<MY_SIZE> &inverse_permutation);

private:
  UnsignedType *point_indices;
  std::vector<UnsignedType> cell_endpoints;
};
/* 1}}} */

class ScotchReorder {
private:
  SCOTCH_Graph graph;
  GraphCSR<SCOTCH_Num> csr;
  SCOTCH_Strat strategy;

public:
  explicit ScotchReorder(MY_SIZE num_points, MY_SIZE num_cells,
                         const data_t &cell_to_node)
      : csr(num_points, num_cells, cell_to_node) {
    SCOTCH_THROW(SCOTCH_graphInit(&graph));
    SCOTCH_THROW(SCOTCH_graphBuild(&graph, 0, csr.num_points,
                                   csr.pointIndices(), NULL, NULL, NULL,
                                   csr.numArcs(), csr.cellEndpoints(), NULL));
    SCOTCH_THROW(SCOTCH_graphCheck(&graph));
    try {
      SCOTCH_THROW(SCOTCH_stratInit(&strategy));
      SCOTCH_THROW(SCOTCH_stratGraphOrder(&strategy, strategy_string));
    } catch (ScotchError &) {
      SCOTCH_graphExit(&graph);
      throw;
    }
  }

  ~ScotchReorder() {
    SCOTCH_graphExit(&graph);
    SCOTCH_stratExit(&strategy);
  }
  ScotchReorder(const ScotchReorder &other) = delete;
  ScotchReorder &operator=(const ScotchReorder &rhs) = delete;

  std::vector<SCOTCH_Num> reorder() {
    std::vector<SCOTCH_Num> permutation(csr.num_points);
    SCOTCH_THROW(SCOTCH_graphOrder(&graph, &strategy, permutation.data(), NULL,
                                   NULL, NULL, NULL));
    return permutation;
  }

public:
  const char *strategy_string = "g";
};

template <class UnsignedType>
void GraphCSR<UnsignedType>::reorderInverse(
    const std::vector<MY_SIZE> &inverse_permutation) {
  assert(inverse_permutation.size() == num_points);
  UnsignedType *new_point_indices = new UnsignedType[num_points + 1];
  std::vector<UnsignedType> new_cell_endpoints(cell_endpoints.size());
  new_point_indices[0] = 0;
  new_point_indices[num_points] = 0;
  for (MY_SIZE i = 0; i < num_points; ++i) {
    const MY_SIZE old_point = inverse_permutation[i];
    const MY_SIZE num_endpoints =
        point_indices[old_point + 1] - point_indices[old_point];
    new_point_indices[i + 1] = new_point_indices[i] + num_endpoints;
    std::copy_n(cell_endpoints.begin() + point_indices[old_point],
                num_endpoints,
                new_cell_endpoints.begin() + new_point_indices[i]);
  }
  assert(cell_endpoints.size() == new_point_indices[num_points]);
  assert(new_cell_endpoints.size() == new_point_indices[num_points]);
  std::vector<MY_SIZE> permutation = invertPermutation(inverse_permutation);
  std::for_each(new_cell_endpoints.begin(), new_cell_endpoints.end(),
                [&permutation](UnsignedType &a) { a = permutation[a]; });
  std::swap(new_point_indices, point_indices);
  delete[] new_point_indices;
  std::swap(new_cell_endpoints, cell_endpoints);
}

#endif /* end of include guard: REORDER_HPP_IGDYRZTN */

/* vim:set et sw=2 ts=2 fdm=marker: */
