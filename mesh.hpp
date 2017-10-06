#ifndef MESH_HPP_JLID0VDH
#define MESH_HPP_JLID0VDH

#include "data_t.hpp"
#include "reorder.hpp"
#include <algorithm>
#include <bitset>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <ostream>
#include <set>
#include <tuple>
#include <vector>

struct InvalidInputFile {
  std::string input_type;
  MY_SIZE line;
};

template <unsigned MeshDim = 2> class Mesh {
public:
  static_assert(MeshDim == 2 || MeshDim == 4 || MeshDim == 8,
                "Only supporting MeshDim in {2,4,8}");
  // I assume 64 colour is enough
  using colourset_t = std::bitset<64>;

  static constexpr unsigned MESH_DIM = MeshDim;

  template <unsigned _MeshDim> friend class Mesh;

private:
  MY_SIZE num_points, num_cells;

public:
  data_t<MY_SIZE, MeshDim> cell_to_node;

  /* Initialisation {{{1 */
protected:
  Mesh(MY_SIZE _num_points, MY_SIZE _num_cells,
       const MY_SIZE *_cell_to_node = nullptr)
      : num_points{_num_points}, num_cells{_num_cells},
        cell_to_node(num_cells) {
    if (_cell_to_node) {
      std::copy(_cell_to_node, _cell_to_node + MeshDim * num_cells,
                cell_to_node.begin());
    }
  }

public:
  /**
   * Constructs graph from stream.
   *
   * Format:
   *   - first line: num_points and num_cells ("\d+\s+\d+")
   *   - next num_cells line: an cell, denoted by MeshDim numbers, the start-
   * and
   *     endpoint respectively ("\d+\s+\d+")
   */
  Mesh(std::istream &is)
      : num_points{0}, num_cells{0},
        cell_to_node((is >> num_points >> num_cells, num_cells)) {
    if (!is) {
      throw InvalidInputFile{"graph input", 0};
    }
    for (MY_SIZE i = 0; i < num_cells; ++i) {
      for (MY_SIZE j = 0; j < MeshDim; ++j) {
        is >> cell_to_node[MeshDim * i + j];
      }
      if (!is) {
        throw InvalidInputFile{"graph input", i};
      }
    }
  }

  ~Mesh() {}

  Mesh(const Mesh &) = delete;
  Mesh &operator=(const Mesh &) = delete;

  Mesh(Mesh &&other)
      : num_points{other.num_points}, num_cells{other.num_cells},
        cell_to_node{std::move(other.cell_to_node)} {
    other.num_points = 0;
    other.num_cells = 0;
  }

  Mesh &operator=(Mesh &&rhs) {
    std::swap(num_points, rhs.num_points);
    std::swap(num_cells, rhs.num_cells);
    std::swap(cell_to_node, rhs.cell_to_node);
    return *this;
  }

  /* 1}}} */

  template <bool VTK = false>
  typename choose_t<VTK, std::vector<std::uint16_t>,
                    std::vector<std::vector<MY_SIZE>>>::type
  colourCells(MY_SIZE from = 0, MY_SIZE to = static_cast<MY_SIZE>(-1)) const {
    if (to > numCells()) {
      to = numCells();
    }
    std::vector<std::vector<MY_SIZE>> cell_partitions;
    std::vector<colourset_t> point_colours(numPoints(), 0);
    std::vector<MY_SIZE> set_sizes(64, 0);
    std::vector<std::uint16_t> cell_colours(numCells());
    colourset_t used_colours;
    for (MY_SIZE i = from; i < to; ++i) {
      colourset_t occupied_colours;
      for (MY_SIZE j = 0; j < MeshDim; ++j) {
        occupied_colours |= point_colours[cell_to_node[MeshDim * i + j]];
      }
      colourset_t available_colours = ~occupied_colours & used_colours;
      if (available_colours.none()) {
        used_colours <<= 1;
        used_colours.set(0);
        if (!VTK) {
          cell_partitions.emplace_back();
        }
        available_colours = ~occupied_colours & used_colours;
      }
      std::uint8_t colour = getAvailableColour(available_colours, set_sizes);
      if (VTK) {
        cell_colours[i] = colour;
      } else {
        cell_partitions[colour].push_back(i);
      }
      colourset_t colourset(1ull << colour);
      for (MY_SIZE j = 0; j < MeshDim; ++j) {
        point_colours[cell_to_node[MeshDim * i + j]] |= colourset;
      }
      ++set_sizes[colour];
    }
    return choose_t<
        VTK, std::vector<std::uint16_t>,
        std::vector<std::vector<MY_SIZE>>>::ret_value(std::move(cell_colours),
                                                      std::move(
                                                          cell_partitions));
  }

  MY_SIZE numCells() const { return num_cells; }

  MY_SIZE numPoints() const { return num_points; }

  /**
   * Writes the cell list in the following format:
   *   - the first line contains two numbers separated by spaces, `numPoints()`
   *     and `numCells()` respectively.
   *   - the following `numCells()` lines contain MeshDim numbers separated
   *     by spaces: the points incident to the cell
   */
  void writeCellList(std::ostream &os) const {
    os << numPoints() << " " << numCells() << std::endl;
    for (std::size_t i = 0; i < numCells(); ++i) {
      for (unsigned j = 0; j < MeshDim; ++j) {
        os << (j > 0 ? " " : "") << cell_to_node[MeshDim * i + j];
      }
      os << std::endl;
    }
  }

  template <typename DataType = float, unsigned DataDim = 1,
            unsigned CellDim = 1, bool SOA = false>
  void reorderScotch(data_t<DataType, CellDim> *cell_data = nullptr,
                     data_t<DataType, DataDim> *point_data = nullptr) {
    ScotchReorder reorder(numPoints(), numCells(), cell_to_node);
    std::vector<SCOTCH_Num> permutation = reorder.reorder();
    this->template reorder<SCOTCH_Num, DataType, DataDim, CellDim, SOA>(
        permutation, cell_data, point_data);
  }

  /**
   * Reorders the graph using the point permutation vector.
   *
   * Also reorders the cell and point data in the arguments. These must be of
   * length `numEdges()` and `numPoints()`, respectively.
   */
  template <typename UnsignedType, typename DataType = float,
            unsigned DataDim = 1, unsigned CellDim = 1, bool SOA = false>
  void reorder(const std::vector<UnsignedType> &point_permutation,
               data_t<DataType, CellDim> *cell_data = nullptr,
               data_t<DataType, DataDim> *point_data = nullptr) {
    // Permute points
    if (point_data) {
      reorderData<DataDim, SOA, DataType, UnsignedType>(*point_data,
                                                        point_permutation);
    }
    // Permute cell_to_node
    for (MY_SIZE i = 0; i < numCells(); ++i) {
      for (MY_SIZE j = 0; j < MeshDim; ++j) {
        cell_to_node[MeshDim * i + j] =
            point_permutation[cell_to_node[MeshDim * i + j]];
      }
    }
    std::vector<std::array<MY_SIZE, MeshDim + 1>> cell_tmp(numCells());
    for (MY_SIZE i = 0; i < numCells(); ++i) {
      cell_tmp[i][MeshDim] = i;
      std::copy(cell_to_node.begin() + MeshDim * i,
                cell_to_node.begin() + MeshDim * (i + 1), cell_tmp[i].begin());
      std::sort(cell_tmp[i].begin(), cell_tmp[i].begin() + MeshDim);
    }
    std::sort(cell_tmp.begin(), cell_tmp.end());
    std::vector<MY_SIZE> inv_permutation(numCells());
    for (MY_SIZE i = 0; i < numCells(); ++i) {
      inv_permutation[i] = cell_tmp[i][MeshDim];
    }
    if (cell_data) {
      reorderDataInverse<CellDim, true>(*cell_data, inv_permutation);
    }
    reorderDataInverse<MeshDim, false>(cell_to_node, inv_permutation);
  }

  template <unsigned CellDim, class DataType>
  void reorderToPartition(std::vector<MY_SIZE> &partition_vector,
                          data_t<DataType, CellDim> &cell_weights) {
    assert(numCells() == partition_vector.size());
    assert(numCells() == cell_weights.getSize());
    std::vector<std::array<MY_SIZE, MeshDim + 2>> tmp(numCells());
    for (MY_SIZE i = 0; i < numCells(); ++i) {
      tmp[i][0] = partition_vector[i];
      tmp[i][1] = i;
      std::copy(cell_to_node.begin() + MeshDim * i,
                cell_to_node.begin() + MeshDim * (i + 1), tmp[i].begin() + 2);
    }
    std::sort(tmp.begin(), tmp.end());
    std::vector<MY_SIZE> permutation(numCells());
    for (MY_SIZE i = 0; i < numCells(); ++i) {
      partition_vector[i] = tmp[i][0];
      permutation[i] = tmp[i][1];
      std::copy(tmp[i].begin() + 2, tmp[i].end(),
                cell_to_node.begin() + MeshDim * i);
    }
    reorderDataInverse<CellDim, true, DataType, MY_SIZE>(cell_weights,
                                                         permutation);
  }

  std::vector<MY_SIZE> getPointRenumberingPermutation() const {
    std::vector<MY_SIZE> permutation(numPoints(), numPoints());
    MY_SIZE new_ind = 0;
    for (MY_SIZE i = 0; i < MeshDim * numCells(); ++i) {
      if (permutation[cell_to_node[i]] == numPoints()) {
        permutation[cell_to_node[i]] = new_ind++;
      }
    }
    // Currently not supporting isolated points
    assert(std::all_of(
        permutation.begin(), permutation.end(),
        [&permutation](MY_SIZE a) { return a < permutation.size(); }));
    return permutation;
  }

  std::vector<MY_SIZE> renumberPoints(const std::vector<MY_SIZE> &permutation) {
    std::for_each(cell_to_node.begin(), cell_to_node.end(),
                  [&permutation](MY_SIZE &a) { a = permutation[a]; });
    return permutation;
  }

  template <bool MinimiseColourSizes = true>
  static MY_SIZE getAvailableColour(colourset_t available_colours,
                                    const std::vector<MY_SIZE> &set_sizes) {
    assert(set_sizes.size() > 0);
    MY_SIZE colour = set_sizes.size();
    for (MY_SIZE i = 0; i < set_sizes.size(); ++i) {
      if (available_colours[i]) {
        if (MinimiseColourSizes) {
          if (colour >= set_sizes.size() || set_sizes[colour] > set_sizes[i]) {
            colour = i;
          }
        } else {
          return i;
        }
      }
    }
    assert(colour < set_sizes.size());
    return colour;
  }

  Mesh<2> getCellToCellGraph() const {
    const std::multimap<MY_SIZE, MY_SIZE> point_to_cell =
        GraphCSR<MY_SIZE>::getPointToCell(cell_to_node);
    // TODO optimise
    std::vector<MY_SIZE> cell_to_cell;
    for (MY_SIZE i = 0; i < numCells(); ++i) {
      for (MY_SIZE offset = 0; offset < MeshDim; ++offset) {
        MY_SIZE point = cell_to_node[MeshDim * i + offset];
        const auto cell_range = point_to_cell.equal_range(point);
        for (auto it = cell_range.first; it != cell_range.second; ++it) {
          MY_SIZE other_cell = it->second;
          if (other_cell > i) {
            cell_to_cell.push_back(i);
            cell_to_cell.push_back(other_cell);
          }
        }
      }
    }
    return Mesh<2>(numCells(), cell_to_cell.size() / 2, cell_to_cell.data());
  }

  std::vector<std::vector<MY_SIZE>>
  getPointToPartition(const std::vector<MY_SIZE> &partition) const {
    std::vector<std::set<MY_SIZE>> _result(num_points);
    for (MY_SIZE i = 0; i < cell_to_node.getSize(); ++i) {
      for (MY_SIZE j = 0; j < MeshDim; ++j) {
        _result[cell_to_node[MeshDim * i + j]].insert(partition[i]);
      }
    }
    std::vector<std::vector<MY_SIZE>> result(num_points);
    std::transform(_result.begin(), _result.end(), result.begin(),
                   [](const std::set<MY_SIZE> &a) {
                     return std::vector<MY_SIZE>(a.begin(), a.end());
                   });
    return result;
  }

  std::vector<MY_SIZE> getPointRenumberingPermutation2(
      const std::vector<std::vector<MY_SIZE>> &point_to_partition) const {
    std::vector<MY_SIZE> inverse_permutation(point_to_partition.size());
    data_t<MY_SIZE, 1> permutation(point_to_partition.size());
    for (MY_SIZE i = 0; i < point_to_partition.size(); ++i) {
      inverse_permutation[i] = i;
      permutation[i] = i;
    }
    std::stable_sort(
        inverse_permutation.begin(), inverse_permutation.end(),
        [&point_to_partition](MY_SIZE a, MY_SIZE b) {
          if (point_to_partition[a].size() != point_to_partition[b].size()) {
            return point_to_partition[a].size() > point_to_partition[b].size();
          } else {
            return point_to_partition[a] > point_to_partition[b];
          }
        });
    reorderData<1, false>(permutation, inverse_permutation);
    return std::vector<MY_SIZE>(permutation.begin(), permutation.end());
  }
};

// vim:set et sw=2 ts=2 fdm=marker:
#endif /* end of include guard: MESH_HPP_JLID0VDH */
