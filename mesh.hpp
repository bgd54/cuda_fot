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
  MY_SIZE mapping_ind, line;
};

class Mesh {
public:
  // I assume 64 colour is enough
  using colourset_t = std::bitset<128>;

private:
  std::vector<MY_SIZE> num_points;

public:
  std::vector<data_t> cell_to_node;

  /* Initialisation {{{1 */
protected:
  Mesh(const std::vector<MY_SIZE> &_num_points, MY_SIZE _num_cells,
       const std::vector<unsigned> &mesh_dim,
       const std::vector<const MY_SIZE *> &_cell_to_node = {})
      : num_points{_num_points}, cell_to_node{} {
    assert(_cell_to_node.empty() || _cell_to_node.size() == _num_points.size());
    assert(_num_points.size() == mesh_dim.size());
    for (size_t i = 0; i < _num_points.size(); ++i) {
      cell_to_node.emplace_back(
          data_t::create<MY_SIZE>(_num_cells, mesh_dim[i]));
    }
    for (size_t i = 0; i < _cell_to_node.size(); ++i) {
      if (_cell_to_node[i]) {
        std::copy_n(_cell_to_node[i], mesh_dim[i] * numCells(),
                    cell_to_node[i].begin<MY_SIZE>());
      }
    }
  }

public:
  /**
   * Constructs graph from stream.
   *
   * Format:
   *   - first line: num_points and num_cklls ("\d+\s+\d+")
   *   - next num_cells line: an cell, denoted by MeshDim numbers, the start-
   * and
   *     endpoint respectively ("\d+\s+\d+")
   */
  Mesh(const std::vector<std::istream *> &iis,
       const std::vector<unsigned> &mesh_dim)
      : num_points(iis.size(), 0), cell_to_node{} {
    assert(iis.size() == mesh_dim.size());
    MY_SIZE num_cells = 0;
    for (size_t mapping_ind = 0; mapping_ind < iis.size(); ++mapping_ind) {
      std::istream &is = *iis[mapping_ind];
      if (!is) {
        throw InvalidInputFile{"graph input", static_cast<MY_SIZE>(mapping_ind),
                               0};
      }
      MY_SIZE _num_points, _num_cells;
      is >> _num_points >> _num_cells;
      num_points[mapping_ind] = _num_points;
      assert(mapping_ind == 0 || num_cells == _num_cells);
      num_cells = _num_cells;
      cell_to_node.emplace_back(
          data_t::create<MY_SIZE>(num_cells, mesh_dim[mapping_ind]));
      for (MY_SIZE i = 0; i < num_cells; ++i) {
        for (MY_SIZE j = 0; j < mesh_dim[mapping_ind]; ++j) {
          is >> cell_to_node[mapping_ind].operator[]<MY_SIZE>(
                    mesh_dim[mapping_ind] * i + j);
          assert(cell_to_node[mapping_ind].operator[]<MY_SIZE>(
                     mesh_dim[mapping_ind] * i + j) < numPoints(mapping_ind));
        }
        if (!is) {
          throw InvalidInputFile{"graph input",
                                 static_cast<MY_SIZE>(mapping_ind), i + 1};
        }
      }
    }
  }

  ~Mesh() {}

  Mesh(const Mesh &) = delete;
  Mesh &operator=(const Mesh &) = delete;

  Mesh(Mesh &&other)
      : num_points{other.num_points},
        cell_to_node{std::move(other.cell_to_node)} {}

  Mesh &operator=(Mesh &&rhs) {
    std::swap(num_points, rhs.num_points);
    std::swap(cell_to_node, rhs.cell_to_node);
    return *this;
  }

  /* 1}}} */

  std::vector<std::vector<MY_SIZE>>
  colourCells(MY_SIZE from = 0, MY_SIZE to = static_cast<MY_SIZE>(-1)) const {
    if (to > numCells()) {
      to = numCells();
    }
    std::vector<std::vector<MY_SIZE>> cell_partitions;
    std::vector<std::vector<colourset_t>> point_colours{};
    for (size_t mapping_ind = 0; mapping_ind < 1; ++mapping_ind) {
      point_colours.push_back(
          std::vector<colourset_t>(numPoints(mapping_ind), 0));
    }
    std::vector<MY_SIZE> set_sizes(64, 0);
    colourset_t used_colours;
    for (MY_SIZE i = from; i < to; ++i) {
      colourset_t occupied_colours;
      for (unsigned mapping_ind = 0; mapping_ind < 1; ++mapping_ind) {
        const unsigned mesh_dim = cell_to_node[mapping_ind].getDim();
        for (MY_SIZE j = 0; j < mesh_dim; ++j) {
          occupied_colours |=
              point_colours[mapping_ind]
                           [cell_to_node[mapping_ind].operator[]<MY_SIZE>(
                               mesh_dim *i + j)];
        }
      }
      colourset_t available_colours = ~occupied_colours & used_colours;
      if (available_colours.none()) {
        used_colours <<= 1;
        used_colours.set(0);
        cell_partitions.emplace_back();
        available_colours = ~occupied_colours & used_colours;
      }
      std::uint8_t colour = getAvailableColour(available_colours, set_sizes);
      cell_partitions[colour].push_back(i);
      colourset_t colourset(1ull << colour);
      for (size_t mapping_ind = 0; mapping_ind < 1; ++mapping_ind) {
        const unsigned mesh_dim = cell_to_node[mapping_ind].getDim();
        for (MY_SIZE j = 0; j < mesh_dim; ++j) {
          point_colours[mapping_ind]
                       [cell_to_node[mapping_ind].operator[]<MY_SIZE>(
                           mesh_dim *i + j)] |= colourset;
        }
      }
      ++set_sizes[colour];
    }
    return cell_partitions;
  }

  MY_SIZE numCells() const {
    assert(!cell_to_node.empty());
    return cell_to_node[0].getSize();
  }

  MY_SIZE numPoints(unsigned mapping_ind) const {
    return num_points[mapping_ind];
  }

  const std::vector<MY_SIZE> numPoints() const { return num_points; }

  unsigned numMappings() const { return cell_to_node.size(); }

  /**
   * Writes the cell list in the following format:
   *   - the first line contains two numbers separated by spaces, `numPoints()`
   *     and `numCells()` respectively.
   *   - the following `numCells()` lines contain MeshDim numbers separated
   *     by spaces: the points incident to the cell
   */
  void writeCellList(std::ostream &os, unsigned mapping_ind) const {
    os << numPoints(mapping_ind) << " " << numCells() << std::endl;
    const unsigned mesh_dim = cell_to_node[mapping_ind].getDim();
    for (std::size_t i = 0; i < numCells(); ++i) {
      for (unsigned j = 0; j < mesh_dim; ++j) {
        os << (j > 0 ? " " : "")
           << cell_to_node[mapping_ind].operator[]<MY_SIZE>(mesh_dim *i + j);
      }
      os << std::endl;
    }
  }

  /**
   * Reorders the graph using the point permutation vector.
   */
  template <typename UnsignedType>
  std::vector<MY_SIZE>
  reorder(const std::vector<UnsignedType> &point_permutation) {
    // Permute cell_to_node
    assert(!cell_to_node.empty());
    assert(point_permutation.size() == numPoints(0));
    renumberPoints(point_permutation, 0);
    const unsigned mesh_dim = cell_to_node[0].getDim();
    std::vector<std::vector<MY_SIZE>> cell_tmp(
        numCells(), std::vector<MY_SIZE>(mesh_dim + 1));
    for (MY_SIZE i = 0; i < numCells(); ++i) {
      cell_tmp[i][mesh_dim] = i;
      std::copy(cell_to_node[0].begin<MY_SIZE>() + mesh_dim * i,
                cell_to_node[0].begin<MY_SIZE>() + mesh_dim * (i + 1),
                cell_tmp[i].begin());
      std::sort(cell_tmp[i].begin(), cell_tmp[i].begin() + mesh_dim);
    }
    std::sort(cell_tmp.begin(), cell_tmp.end());
    std::vector<MY_SIZE> inv_permutation(numCells());
    for (MY_SIZE i = 0; i < numCells(); ++i) {
      inv_permutation[i] = cell_tmp[i][mesh_dim];
    }
    for (data_t &mapping : cell_to_node) {
      reorderDataInverse<false>(mapping, inv_permutation);
    }
    return inv_permutation;
  }

  std::vector<MY_SIZE>
  reorderToPartition(std::vector<MY_SIZE> &partition_vector) {
    assert(numCells() == partition_vector.size());
    std::vector<std::array<MY_SIZE, 2>> tmp(numCells());
    for (MY_SIZE i = 0; i < numCells(); ++i) {
      tmp[i][0] = partition_vector[i];
      tmp[i][1] = i;
    }
    std::sort(tmp.begin(), tmp.end());
    std::vector<MY_SIZE> inv_permutation(numCells());
    for (MY_SIZE i = 0; i < numCells(); ++i) {
      partition_vector[i] = tmp[i][0];
      inv_permutation[i] = tmp[i][1];
    }
    for (data_t &mapping : cell_to_node) {
      reorderDataInverse<false>(mapping, inv_permutation);
    }
    return inv_permutation;
  }

  std::vector<MY_SIZE>
  getPointRenumberingPermutation(unsigned mapping_ind) const {
    std::vector<MY_SIZE> permutation(numPoints(mapping_ind),
                                     numPoints(mapping_ind));
    MY_SIZE new_ind = 0;
    for (MY_SIZE i = 0; i < cell_to_node[mapping_ind].getDim() * numCells();
         ++i) {
      if (permutation[cell_to_node[mapping_ind].operator[]<MY_SIZE>(i)] ==
          numPoints(mapping_ind)) {
        permutation[cell_to_node[mapping_ind].operator[]<MY_SIZE>(i)] =
            new_ind++;
      }
    }
    std::for_each(permutation.begin(), permutation.end(),
                  [&permutation, &new_ind](MY_SIZE &a) {
                    if (a == permutation.size()) {
                      a = new_ind++;
                    }
                  });
    assert(std::all_of(
        permutation.begin(), permutation.end(),
        [&permutation](MY_SIZE a) { return a < permutation.size(); }));
    return permutation;
  }

  template <class UnsignedType>
  const std::vector<UnsignedType> &
  renumberPoints(const std::vector<UnsignedType> &permutation,
                 unsigned mapping_ind) {
    assert(std::size_t(std::numeric_limits<UnsignedType>::max()) >=
           numPoints(mapping_ind));
    std::for_each(cell_to_node[mapping_ind].begin<MY_SIZE>(),
                  cell_to_node[mapping_ind].end<MY_SIZE>(),
                  [&permutation](MY_SIZE &a) {
                    a = permutation[a];
                    assert(a < permutation.size());
                  });
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

  Mesh getCellToCellGraph() const {
    std::set<std::pair<MY_SIZE, MY_SIZE>> cell_to_cell;
    for (unsigned mapping_ind = 0; mapping_ind < cell_to_node.size();
         ++mapping_ind) {
      const unsigned mesh_dim = cell_to_node[mapping_ind].getDim();
      const std::multimap<MY_SIZE, MY_SIZE> point_to_cell =
          GraphCSR<MY_SIZE>::getPointToCell(cell_to_node[mapping_ind]);
      for (MY_SIZE i = 0; i < numCells(); ++i) {
        for (MY_SIZE offset = 0; offset < mesh_dim; ++offset) {
          MY_SIZE point = cell_to_node[mapping_ind].operator[]<MY_SIZE>(
              mesh_dim *i + offset);
          const auto cell_range = point_to_cell.equal_range(point);
          for (auto it = cell_range.first; it != cell_range.second; ++it) {
            MY_SIZE other_cell = it->second;
            if (other_cell > i) {
              cell_to_cell.insert(std::pair<MY_SIZE, MY_SIZE>(i, other_cell));
            }
          }
        }
      }
    }
    std::vector<MY_SIZE> _cell_to_cell;
    for (std::pair<MY_SIZE, MY_SIZE> c : cell_to_cell) {
      _cell_to_cell.push_back(c.first);
      _cell_to_cell.push_back(c.second);
    }
    return Mesh({numCells()}, cell_to_cell.size(), {2}, {_cell_to_cell.data()});
  }

  std::vector<std::vector<MY_SIZE>>
  getPointToPartition(const std::vector<MY_SIZE> &partition,
                      unsigned mapping_ind) const {
    std::vector<std::set<MY_SIZE>> _result(numPoints(mapping_ind));
    const unsigned mesh_dim = cell_to_node[mapping_ind].getDim();
    for (MY_SIZE i = 0; i < cell_to_node[mapping_ind].getSize(); ++i) {
      for (MY_SIZE j = 0; j < mesh_dim; ++j) {
        _result[cell_to_node[mapping_ind].operator[]<MY_SIZE>(mesh_dim *i + j)]
            .insert(partition[i]);
      }
    }
    std::vector<std::vector<MY_SIZE>> result(numPoints(mapping_ind));
    std::transform(_result.begin(), _result.end(), result.begin(),
                   [](const std::set<MY_SIZE> &a) {
                     return std::vector<MY_SIZE>(a.begin(), a.end());
                   });
    return result;
  }

  static std::vector<MY_SIZE> getPointRenumberingPermutation2(
      const std::vector<std::vector<MY_SIZE>> &point_to_partition) {
    std::vector<MY_SIZE> inverse_permutation(point_to_partition.size());
    std::vector<MY_SIZE> permutation(point_to_partition.size());
    for (MY_SIZE i = 0; i < point_to_partition.size(); ++i) {
      inverse_permutation[i] = i;
      permutation[i] = i;
    }
    std::stable_sort(
        inverse_permutation.begin(), inverse_permutation.end(),
        [&point_to_partition](MY_SIZE a, MY_SIZE b) {
          if (point_to_partition[a].size() != point_to_partition[b].size()) {
            return point_to_partition[a].size() > point_to_partition[b].size();
          } else if (point_to_partition[b].empty()) {
            return true;
          } else if (point_to_partition[a].empty()) {
            return false;
          } else {
            return point_to_partition[a] > point_to_partition[b];
          }
        });
    reorderData<false>(permutation, 1, inverse_permutation);
    return permutation;
  }
};

// vim:set et sts=2 sw=2 ts=2 fdm=marker:
#endif /* end of include guard: MESH_HPP_JLID0VDH */
