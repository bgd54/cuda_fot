#ifndef MESH_HPP_JLID0VDH
#define MESH_HPP_JLID0VDH

#include "data_t.hpp"
#include "reorder.hpp"
#include "tuple_utils.hpp"
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
  MY_SIZE file_index, line;
};

template <unsigned... MeshDims> class Mesh {
public:
  // I assume 64 colour is enough
  using colourset_t = std::bitset<64>;

  using mappings_t = generate_mappings_t<MeshDims...>;
  static constexpr unsigned NUM_MAPPINGS = std::tuple_size<mappings_t>::value;
  static_assert(NUM_MAPPINGS > 0, "I've assumed there is at least one mapping");

  template <unsigned _MeshDim> friend class Mesh;

private:
  std::array<MY_SIZE, NUM_MAPPINGS> num_points;

public:
  mappings_t mappings;

  /* Initialisation {{{1 */
protected:
  Mesh(MY_SIZE num_cells, const std::array<MY_SIZE, NUM_MAPPINGS> &num_points_)
      : mappings{initMapping<mappings_t>(num_cells)}, num_points{num_points_} {}

public:
  /**
   * Constructs graph from stream.
   *
   * Format:
   *   - first line: num_points and num_cells ("\d+\s+\d+")
   *   - next num_cells line: an cell, denoted by MeshDim numbers, the points
   *     accessed by the cell, separated by space
   */
  Mesh(MY_SIZE num_cells, const std::array<MY_SIZE, NUM_MAPPINGS> &num_points_,
       const std::array<std::istream *, NUM_MAPPINGS> &is)
      : mappings{initMapping<mappings_t>(num_cells)} {
    for (unsigned i = 0; i < NUM_MAPPINGS; ++i) {
      if (!(*is[i])) {
        throw InvalidInputFile{"mapping input", i, 0};
      }
      num_points[i] = num_points_[i];
    }
    for_each(mappings, ReadMappingFromFile{is});
  }

  ~Mesh() {}

  Mesh(const Mesh &) = delete;
  Mesh &operator=(const Mesh &) = delete;

  Mesh(Mesh &&other) : mappings{std::move(other.mappings)} {}

  Mesh &operator=(Mesh &&rhs) {
    std::swap(mappings, rhs.mappings);
    return *this;
  }

  /* 1}}} */

  std::vector<std::vector<MY_SIZE>>
  colourCells(MY_SIZE from = 0, MY_SIZE to = static_cast<MY_SIZE>(-1)) const {
    if (to > numCells()) {
      to = numCells();
    }
    std::vector<std::vector<MY_SIZE>> cell_partitions;
    std::vector<std::vector<colourset_t>> point_colours(numPoints());
    for (unsigned i = 0; i < NUM_MAPPINGS; ++i) {
      point_colours[i].resize(num_points[i]);
    }
    std::vector<MY_SIZE> set_sizes(colourset_t{}.size(), 0);
    colourset_t used_colours;
    for (MY_SIZE i = from; i < to; ++i) {
      GatherPointColours gatherPointColours{i, point_colours};
      for_each(mappings, gatherPointColours);
      colourset_t occupied_colours = gatherPointColours.getResult;
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
      for_each(mappings, UpdatePointColours{i, colourset});
      ++set_sizes[colour];
    }
    return cell_partitions;
  }

  MY_SIZE numCells() const { return std::get<0>(mappings).getSize(); }

  MY_SIZE numPoints(unsigned mapping_ind) const {
    return num_points[mapping_ind];
  }

  const std::array<MY_SIZE, NUM_MAPPINGS> numPoints() const {
    return num_points;
  }

  /**
   * Writes the cell list in the following format:
   *   - the first line contains two numbers separated by spaces, `numPoints()`
   *     and `numCells()` respectively.
   *   - the following `numCells()` lines contain MeshDim numbers separated
   *     by spaces: the points incident to the cell
   */
  template <unsigned MappingInd> void writeCellList(std::ostream &os) const {
    os << numPoints(MappingInd) << " " << numCells() << std::endl;
    unsigned mesh_dim = std::get<MappingInd>(mappings).getSize();
    for (std::size_t i = 0; i < numCells(); ++i) {
      for (unsigned j = 0; j < mesh_dim; ++j) {
        os << (j > 0 ? " " : "")
           << std::get<MappingInd>(mappings)[mesh_dim * i + j];
      }
      os << std::endl;
    }
  }

  void renumberPointsGPS() { for_each(mappings, RenumberPointsGPS{}); }

  void reorderMappings(const std::vector<MY_SIZE> &inverse_permutation) {
    for_each(mappings, ReorderMapping{inverse_permutation});
  }

  /**
   * Reorders the mappings using the point permutation vector that corresponds
   * to the first mapping. Also renumbers the points in the first mapping.
   */
  template <typename UnsignedType>
  void reorder(const std::vector<UnsignedType> &point_permutation) {
    // Permute cell_to_node
    static_assert(NUM_MAPPINGS > 0, "Assuming here that NUM_MAPPINGS > 0");
    auto &cell_to_node = std::get<0>(mappings);
    constexpr unsigned MESH_DIM = cell_to_node.dim;
    renumberPoints<0>(point_permutation);
    std::vector<std::array<MY_SIZE, MESH_DIM + 1>> cell_tmp(numCells());
    for (MY_SIZE i = 0; i < numCells(); ++i) {
      cell_tmp[i][MESH_DIM] = i;
      std::copy(cell_to_node.begin() + MESH_DIM * i,
                cell_to_node.begin() + MESH_DIM * (i + 1), cell_tmp[i].begin());
      std::sort(cell_tmp[i].begin(), cell_tmp[i].begin() + MESH_DIM);
    }
    std::sort(cell_tmp.begin(), cell_tmp.end());
    std::vector<MY_SIZE> inv_permutation(numCells());
    for (MY_SIZE i = 0; i < numCells(); ++i) {
      inv_permutation[i] = cell_tmp[i][MESH_DIM];
    }
    reorderMappings(inv_permutation);
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
    std::vector<MY_SIZE> permutation(numCells());
    for (MY_SIZE i = 0; i < numCells(); ++i) {
      partition_vector[i] = tmp[i][0];
      permutation[i] = tmp[i][1];
    }
    reorderMappings(permutation);
    return permutation;
  }

  template <unsigned MapInd>
  std::vector<MY_SIZE> getPointRenumberingPermutation() const {
    std::vector<MY_SIZE> permutation(numPoints(MapInd), numPoints(MapInd));
    MY_SIZE new_ind = 0;
    auto &cell_to_node = std::get<MapInd>(mappings);
    unsigned mesh_dim = cell_to_node.dim;
    for (MY_SIZE i = 0; i < mesh_dim * numCells(); ++i) {
      if (permutation[cell_to_node[i]] == numPoints(MapInd)) {
        permutation[cell_to_node[i]] = new_ind++;
      }
    }
    // Currently not supporting isolated points
    assert(std::all_of(
        permutation.begin(), permutation.end(),
        [&permutation](MY_SIZE a) { return a < permutation.size(); }));
    return permutation;
  }

  template <unsigned MapInd, class IndexType>
  std::vector<IndexType>
  renumberPoints(const std::vector<IndexType> &point_permutation) {
    for_each(std::get<MapInd>(mappings).begin(),
             std::get<MapInd>(mappings).end(),
             [&](MY_SIZE &a) { a = point_permutation[a]; });
    return point_permutation;
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
    GetCellToCell reducer{};
    for_each(mappings, reducer);
    std::set<std::pair<MY_SIZE, MY_SIZE>> cell_to_cell =
        reducer.getCellToCell();
    Mesh<2> result(cell_to_cell.size() / 2, numCells());
  }

  template <unsigned MappingInd>
  std::vector<std::vector<MY_SIZE>>
  getPointToPartition(const std::vector<MY_SIZE> &partition) const {
    std::vector<std::set<MY_SIZE>> _result(numPoints(MappingInd));
    const auto &cell_to_node = std::get<MappingInd>(mappings);
    MY_SIZE mesh_dim = cell_to_node.dim;
    for (MY_SIZE i = 0; i < cell_to_node.getSize(); ++i) {
      for (MY_SIZE j = 0; j < mesh_dim; ++j) {
        _result[cell_to_node[mesh_dim * i + j]].insert(partition[i]);
      }
    }
    std::vector<std::vector<MY_SIZE>> result(num_points);
    std::transform(_result.begin(), _result.end(), result.begin(),
                   [](const std::set<MY_SIZE> &a) {
                     return std::vector<MY_SIZE>(a.begin(), a.end());
                   });
    return result;
  }

  static std::vector<MY_SIZE> getPointRenumberingPermutation2(
      const std::vector<std::vector<MY_SIZE>> &point_to_partition) {
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

protected:
  std::array<MY_SIZE, NUM_MAPPINGS>
  readFirstNumber(const std::array<std::istream *, NUM_MAPPINGS> &is) {
    std::array<MY_SIZE, NUM_MAPPINGS> result{};
    for (unsigned i = 0; i < NUM_MAPPINGS; ++i) {
      if (!(*is)) {
        throw InvalidInputFile{"mapping input", i, 0};
      }
      (*is[i]) >> result[i];
    }
  }

  class ReadMappingFromFile {
    const std::array<std::istream *, NUM_MAPPINGS> &is;

  public:
    ReadMappingFromFile(const std::array<std::istream *, NUM_MAPPINGS> &is_)
        : is{is_} {}
    template <unsigned MappingInd, unsigned Dim>
    void operator()(data_t<MY_SIZE, Dim> &mapping) {
      for (MY_SIZE j = 0; j < mapping.getSize(); ++j) {
        for (MY_SIZE k = 0; k < mapping.dim; ++k) {
          *is[MappingInd] >> mapping[mapping.dim * j + k];
        }
        if (!(*is[MappingInd])) {
          throw InvalidInputFile{"graph input", MappingInd, j};
        }
      }
    }
  };

  class GatherPointColours {
    colourset_t result{0};
    const std::vector<std::vector<MY_SIZE>> &point_colours;
    MY_SIZE cell_ind;

  public:
    GatherPointColours(MY_SIZE cell_ind_,
                       const std::vector<std::vector<MY_SIZE>> &point_colours_)
        : point_colours{point_colours_}, cell_ind{cell_ind_} {}
    colourset_t getResult() const { return result; }
    template <unsigned MappingInd, unsigned Dim>
    void operator()(data_t<MY_SIZE, Dim> &mapping) {
      for (unsigned i = 0; i < Dim; ++i) {
        result |= point_colours[MappingInd][mapping[Dim * cell_ind + i]];
      }
    }
  };

  class UpdatePointColours {
    colourset_t colour;
    std::vector<std::vector<MY_SIZE>> &point_colours;
    MY_SIZE cell_ind;

  public:
    UpdatePointColours(colourset_t colour_, MY_SIZE cell_ind_,
                       const std::vector<std::vector<MY_SIZE>> &point_colours_)
        : colour{colour_}, point_colours{point_colours_}, cell_ind{cell_ind_} {}
    template <unsigned MappingInd, unsigned Dim>
    void operator()(data_t<MY_SIZE, Dim> &mapping) {
      for (unsigned i = 0; i < Dim; ++i) {
        point_colours[MappingInd][mapping[Dim * cell_ind + i]] |= colour;
        ;
      }
    }
  };

  class RenumberPointsGPS {
  public:
    template <unsigned MappingInd, unsigned Dim>
    void operator()(data_t<MY_SIZE, Dim> &data) {
      if (MappingInd == 0) {
        return;
      }
      ScotchReorder reorder(numPoints(MapInd), numCells(), data);
      std::vector<SCOTCH_Num> point_permutation = reorder.reorder();
      for_each(data.begin(), data.end(),
               [&](MY_SIZE &a) { a = point_permutation[a]; });
    }
  };

  class ReorderMapping {
    const std::vector<MY_SIZE> &inverse_permutation;

  public:
    ReorderMapping(const std::vector<MY_SIZE> &inverse_permutation_)
        : inverse_permutation{inverse_permutation_} {
      assert(inverse_permutation.size() == numCells());
    }

    template <unsigned MapInd, unsigned Dim>
    void operator()(data_t<MY_SIZE, Dim> &mapping) {
      reorderDataInverse<Dim, false>(mapping, inverse_permutation);
    }
  };

  class GetCellToCell {
    std::set<std::pair<MY_SIZE, MY_SIZE>> cell_to_cell{};

  public:
    std::set<std::pair<MY_SIZE, MY_SIZE>> getCellToCell() const {
      return cell_to_cell;
    }
    template <unsigned, unsigned Dim>
    void operator()(data_t<MY_SIZE, Dim> &mapping) {
      const std::multimap<MY_SIZE, MY_SIZE> point_to_cell =
          GraphCSR<MY_SIZE>::getPointToCell(mapping);
      for (MY_SIZE i = 0; i < mapping.getSize(); ++i) {
        for (MY_SIZE offset = 0; offset < mapping.dim; ++offset) {
          MY_SIZE point = mapping[mapping.dim * i + offset];
          const auto cell_range = point_to_cell.equal_range(point);
          for (auto it = cell_range.first; it != cell_range.second; ++it) {
            MY_SIZE other_cell = it->second;
            if (other_cell > i) {
              cell_to_cell.insert({i, other_cell});
            }
          }
        }
      }
    }
  };
};

// vim:set et sw=2 ts=2 fdm=marker:
#endif /* end of include guard: MESH_HPP_JLID0VDH */
