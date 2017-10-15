#ifndef COLOURING_HPP_PMK0HFCY
#define COLOURING_HPP_PMK0HFCY

#include <algorithm>
#include <bitset>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <vector>

#include "problem.hpp"

template <unsigned MeshDim, unsigned PointDim = 1, unsigned CellDim = 1,
          bool SOA = false, typename DataType = float>
struct HierarchicalColourMemory {
  using colourset_t = Mesh::colourset_t;
  static_assert(Problem<PointDim, CellDim, SOA, DataType>::MESH_DIM == MeshDim,
                "Mesh dimension should be the same as in Problem");
  struct MemoryOfOneColour {
    std::vector<DataType>
        cell_weights; // restructured so it can be indexed with tid
                      // it's a vector, because it's not necessarily block_size
                      // long; also I don't want mem. management just now
    std::vector<MY_SIZE> points_to_be_cached;
    std::vector<MY_SIZE> points_to_be_cached_offsets = {0};
    // every thread caches the one with index
    // the multiple of its tid
    std::vector<MY_SIZE> cell_list; // same as before, just points to shared mem
                                    // computed from the above
    std::vector<std::uint8_t> cell_colours;     // the colour for each cell
                                                // in the block
    std::vector<std::uint8_t> num_cell_colours; // the number of cell colours
                                                // in each block
    std::vector<MY_SIZE> block_offsets = {0};   // where the is block in
                                                // the vectors above
  };
  std::vector<MemoryOfOneColour> colours;

  HierarchicalColourMemory(
      const Problem<PointDim, CellDim, SOA, DataType> &problem,
      const std::vector<MY_SIZE> &partition_vector = {}) {
    /* Algorithm:
     *   - loop through `block_size` blocks
     *     - determine the points written to (not the same as points used)
     *     - get the set of available colours
     *     - among these, choose the one with the minimal number of sets
     *     - colour all of the points above
     *     - assign block to colour:
     *       - add weights to cell_weights
     *       - add used(!) points to `points_to_be_cached` (without duplicates)
     *            (set maybe?)
     *       - add cells to cell_list
     *     - if new colour is needed, `colours.push_back()` then the same as
     *       above
     *   - done
     */
    const Mesh &mesh = problem.mesh;
    assert(mesh.cell_to_node.getTypeSize() == sizeof(MY_SIZE));
    assert(mesh.cell_to_node.getDim() == MeshDim);
    const MY_SIZE block_size = problem.block_size;
    std::vector<colourset_t> point_colours(mesh.numPoints(), 0);
    colourset_t used_colours;
    std::vector<MY_SIZE> set_sizes;
    std::vector<std::uint8_t> block_colours(mesh.numCells());
    std::vector<DataType> tmp_cell_weights(
        problem.cell_weights.template cbegin<DataType>(),
        problem.cell_weights.template cend<DataType>());
    std::vector<MY_SIZE> blocks;
    std::vector<std::pair<MY_SIZE, MY_SIZE>> partition_to_cell;
    std::vector<MY_SIZE> cell_inverse_permutation(mesh.numCells());
    if (partition_vector.size() == problem.mesh.numCells()) {
      for (MY_SIZE i = 0; i < partition_vector.size(); ++i) {
        partition_to_cell.push_back(std::make_pair(partition_vector[i], i));
      }
      std::stable_sort(partition_to_cell.begin(), partition_to_cell.end());
      MY_SIZE tid = 0;
      blocks.push_back(0);
      cell_inverse_permutation[0] = partition_to_cell[0].second;
      for (MY_SIZE i = 1; i < partition_vector.size(); ++i) {
        cell_inverse_permutation[i] = partition_to_cell[i].second;
        if (++tid == block_size) {
          tid = 0;
          blocks.push_back(i);
          continue;
        }
        if (partition_to_cell[i - 1].first != partition_to_cell[i].first) {
          tid = 0;
          blocks.push_back(i);
        }
      }
      assert(blocks.back() != partition_vector.size());
      blocks.push_back(partition_vector.size());
      std::vector<typename std::vector<DataType>::iterator> begins(CellDim);
      for (MY_SIZE i = 0; i < CellDim; ++i) {
        begins[i] =
            std::next(tmp_cell_weights.begin(), i * problem.mesh.numCells());
      }
      typename std::vector<DataType>::iterator end_first =
          std::next(tmp_cell_weights.begin(), problem.mesh.numCells());
      reorderDataInverseVectorSOA<DataType, MY_SIZE>(begins, end_first,
                                                     cell_inverse_permutation);
    } else {
      for (MY_SIZE i = 0; i < problem.mesh.numCells(); i += block_size) {
        blocks.emplace_back(i);
      }
      assert(blocks.back() != problem.mesh.numCells());
      blocks.push_back(problem.mesh.numCells());
      partition_to_cell.resize(problem.mesh.numCells());
      for (MY_SIZE i = 0; i < mesh.numCells(); ++i) {
        partition_to_cell[i] = std::make_pair(0, i);
        cell_inverse_permutation[i] = i;
      }
    }
    assert(blocks.size() >= 2);
    for (MY_SIZE i = 1; i < blocks.size(); ++i) {
      MY_SIZE block_from = blocks[i - 1];
      MY_SIZE block_to = blocks[i];
      assert(block_to != block_from);
      assert(block_to - block_from <= block_size);
      colourset_t occupied_colours =
          getOccupiedColours(mesh.cell_to_node, block_from, block_to,
                             point_colours, partition_to_cell);
      colourset_t available_colours = ~occupied_colours & used_colours;
      if (available_colours.none()) {
        // Need a new colour
        used_colours <<= 1;
        used_colours.set(0);
        set_sizes.push_back(0);
        available_colours = ~occupied_colours & used_colours;
        assert(available_colours.any());
        colours.emplace_back();
      }
      MY_SIZE colour = Mesh::getAvailableColour(available_colours, set_sizes);
      ++set_sizes[colour];
      colourBlock(block_from, block_to, colour, point_colours, problem, colours,
                  partition_to_cell, block_colours, tmp_cell_weights);
      colours[colour].block_offsets.push_back(
          colours[colour].block_offsets.back() + block_to - block_from);
    }
    copyCellWeights(problem, block_colours, tmp_cell_weights);
    cellListSOA();
  }

private:
  static colourset_t getOccupiedColours(
      const data_t &cell_to_node, MY_SIZE from, MY_SIZE to,
      const std::vector<colourset_t> &point_colours,
      const std::vector<std::pair<MY_SIZE, MY_SIZE>> &partition_to_cell) {
    colourset_t result;
    for (MY_SIZE i = from; i < to; ++i) {
      MY_SIZE cell_ind = partition_to_cell[i].second;
      for (MY_SIZE j = 0; j < MeshDim; ++j) {
        result |= point_colours[cell_to_node.operator[]<MY_SIZE>(
            MeshDim *cell_ind + j)];
      }
    }
    return result;
  }

  /*
   * Colours every point written by the cells in the block and also
   * collects all points accessed by the cells in the block.
   */
  void
  colourBlock(MY_SIZE from, MY_SIZE to, MY_SIZE colour_ind,
              std::vector<colourset_t> &point_colours,
              const Problem<PointDim, CellDim, SOA, DataType> &problem,
              std::vector<MemoryOfOneColour> &colours,
              const std::vector<std::pair<MY_SIZE, MY_SIZE>> &partition_to_cell,
              std::vector<std::uint8_t> &block_colours,
              std::vector<DataType> &tmp_cell_weights) {
    const Mesh &mesh = problem.mesh;
    const data_t &cell_to_node = mesh.cell_to_node;
    colourset_t colourset(1ull << colour_ind);
    MemoryOfOneColour &colour = colours[colour_ind];
    const MY_SIZE colour_from = colour.cell_colours.size();
    std::map<MY_SIZE, std::vector<std::pair<MY_SIZE, MY_SIZE>>>
        points_to_cells; // points -> vector of (cell_ind, point_offset)
    for (MY_SIZE i = from; i < to; ++i) {
      MY_SIZE cell_ind = partition_to_cell[i].second;
      for (MY_SIZE offset = 0; offset < MeshDim; ++offset) {
        point_colours[cell_to_node.operator[]<MY_SIZE>(MeshDim *cell_ind +
                                                       offset)] |= colourset;
        points_to_cells[cell_to_node.operator[]<MY_SIZE>(MeshDim *cell_ind +
                                                         offset)]
            .emplace_back(i - from, offset);
      }
      block_colours[i] = colour_ind;
    }
    std::vector<MY_SIZE> c_cell_list(MeshDim * (to - from));
    std::vector<MY_SIZE> points_to_be_cached;
    for (const auto &t : points_to_cells) {
      MY_SIZE point_ind = t.first;
      const std::vector<std::pair<MY_SIZE, MY_SIZE>> &cell_inds = t.second;
      for (const std::pair<MY_SIZE, MY_SIZE> c : cell_inds) {
        MY_SIZE ind = c.first;
        MY_SIZE offset = c.second;
        c_cell_list[MeshDim * ind + offset] = points_to_be_cached.size();
      }
      points_to_be_cached.push_back(point_ind);
    }
    colour.cell_list.insert(colour.cell_list.end(), c_cell_list.begin(),
                            c_cell_list.end());
    colourCells(to - from, colour, c_cell_list, points_to_be_cached.size());
    const MY_SIZE colour_to = colour.cell_colours.size();
    sortCellsByColours(colour_from, colour_to, from, to, colour_ind,
                       tmp_cell_weights);
    // permuteCachedPoints(points_to_be_cached, colour_from,
    //                                      colour_to, colour_ind);
    colour.points_to_be_cached.insert(colour.points_to_be_cached.end(),
                                      points_to_be_cached.begin(),
                                      points_to_be_cached.end());
    colour.points_to_be_cached_offsets.push_back(
        colour.points_to_be_cached.size());
  }

  void colourCells(MY_SIZE block_size, MemoryOfOneColour &block,
                   const std::vector<MY_SIZE> &cell_list, MY_SIZE num_point) {
    static std::vector<colourset_t> point_colours;
    point_colours.resize(num_point);
    memset(point_colours.data(), 0, sizeof(colourset_t) * point_colours.size());
    std::uint8_t num_cell_colours = 0;
    std::vector<MY_SIZE> set_sizes(64, 0);
    colourset_t used_colours;
    for (MY_SIZE i = 0; i < block_size; ++i) {
      colourset_t occupied_colours;
      for (MY_SIZE offset = 0; offset < MeshDim; ++offset) {
        occupied_colours |= point_colours[cell_list[MeshDim * i + offset]];
      }
      colourset_t available_colours = ~occupied_colours & used_colours;
      if (available_colours.none()) {
        ++num_cell_colours;
        used_colours <<= 1;
        used_colours.set(0);
        available_colours = ~occupied_colours & used_colours;
      }
      std::uint8_t colour = Mesh::template getAvailableColour<false>(
          available_colours, set_sizes);
      block.cell_colours.push_back(colour);
      ++set_sizes[colour];
      colourset_t colourset(1ull << colour);
      for (MY_SIZE offset = 0; offset < MeshDim; ++offset) {
        point_colours[cell_list[MeshDim * i + offset]] |= colourset;
      }
    }
    block.num_cell_colours.push_back(num_cell_colours);
  }

  void sortCellsByColours(MY_SIZE colour_from, MY_SIZE colour_to, MY_SIZE from,
                          MY_SIZE to, MY_SIZE block_colour,
                          std::vector<DataType> &tmp_cell_weights) {
    std::vector<std::tuple<std::uint8_t, std::array<MY_SIZE, MeshDim>, MY_SIZE>>
        tmp(colour_to - colour_from);
    MemoryOfOneColour &memory = colours[block_colour];
    for (MY_SIZE i = 0; i < colour_to - colour_from; ++i) {
      const MY_SIZE j = i + colour_from;
      tmp[i] = std::make_tuple(memory.cell_colours[j],
                               std::array<MY_SIZE, MeshDim>(), i);
      std::copy(memory.cell_list.begin() + MeshDim * j,
                memory.cell_list.begin() + MeshDim * (j + 1),
                std::get<1>(tmp[i]).begin());
    }
    std::stable_sort(
        tmp.begin(), tmp.end(),
        [](const std::tuple<std::uint8_t, std::array<MY_SIZE, MeshDim>, MY_SIZE>
               &a,
           const std::tuple<std::uint8_t, std::array<MY_SIZE, MeshDim>, MY_SIZE>
               &b) { return std::get<0>(a) < std::get<0>(b); });
    std::vector<MY_SIZE> inverse_permutation(colour_to - colour_from);
    for (MY_SIZE i = 0; i < colour_to - colour_from; ++i) {
      const MY_SIZE j = i + colour_from;
      std::array<MY_SIZE, MeshDim> cell_points;
      std::tie(memory.cell_colours[j], cell_points, inverse_permutation[i]) =
          tmp[i];
      std::copy(cell_points.begin(), cell_points.end(),
                memory.cell_list.begin() + MeshDim * j);
    }
    std::vector<typename std::vector<DataType>::iterator> begins(CellDim);
    for (MY_SIZE i = 0; i < CellDim; ++i) {
      MY_SIZE dim_pos =
          index<true>(tmp_cell_weights.size() / CellDim, from, CellDim, i);
      begins[i] = std::next(tmp_cell_weights.begin(), dim_pos);
    }
    typename std::vector<DataType>::iterator end_first =
        std::next(tmp_cell_weights.begin(), to);
    reorderDataInverseVectorSOA<DataType, MY_SIZE>(begins, end_first,
                                                   inverse_permutation);
  }

  void permuteCachedPoints(std::vector<MY_SIZE> &points_to_be_cached,
                           MY_SIZE colour_from, MY_SIZE colour_to,
                           MY_SIZE colour_ind) {
    std::set<MY_SIZE> seen_points;
    std::vector<MY_SIZE> &cell_list = colours[colour_ind].cell_list;
    std::vector<MY_SIZE> new_points_to_be_cached;
    new_points_to_be_cached.reserve(points_to_be_cached.size());
    std::vector<MY_SIZE> permutation(points_to_be_cached.size());
    for (MY_SIZE offset = 0; offset < MeshDim; ++offset) {
      for (MY_SIZE i = 0; i < colour_to - colour_from; ++i) {
        MY_SIZE point_ind = cell_list[MeshDim * (colour_from + i) + offset];
        assert(point_ind < permutation.size());
        auto r = seen_points.insert(point_ind);
        if (r.second) {
          permutation[point_ind] = new_points_to_be_cached.size();
          new_points_to_be_cached.push_back(points_to_be_cached[point_ind]);
        }
        cell_list[MeshDim * (colour_from + i) + offset] =
            permutation[point_ind];
      }
    }
    assert(new_points_to_be_cached.size() == points_to_be_cached.size());
    std::copy(new_points_to_be_cached.begin(), new_points_to_be_cached.end(),
              points_to_be_cached.begin());
  }

  void cellListSOA() {
    for (MemoryOfOneColour &memory : colours) {
      AOStoSOA(memory.cell_list, MeshDim);
    }
  }

  void copyCellWeights(const Problem<PointDim, CellDim, SOA, DataType> &problem,
                       const std::vector<std::uint8_t> &block_colours,
                       const std::vector<DataType> &tmp_cell_weights) {
    for (MY_SIZE i = 0; i < colours.size(); ++i) {
      colours[i].cell_weights.reserve(colours[i].cell_list.size() / MeshDim);
    }
    for (MY_SIZE d = 0; d < CellDim; ++d) {
      for (MY_SIZE i = 0; i < problem.mesh.numCells(); ++i) {
        MY_SIZE colour = block_colours[i];
        DataType weight = tmp_cell_weights[index<true>(problem.mesh.numCells(),
                                                       i, CellDim, d)];
        colours[colour].cell_weights.push_back(weight);
      }
    }
  }

public:
  /*******************
   *  Device layout  *
   *******************/
  struct DeviceMemoryOfOneColour {
    device_data_t cell_weights;
    device_data_t points_to_be_cached, points_to_be_cached_offsets;
    device_data_t cell_list;
    device_data_t cell_colours;
    device_data_t num_cell_colours;
    device_data_t block_offsets;
    MY_SIZE shared_size;

    DeviceMemoryOfOneColour(const MemoryOfOneColour &memory)
        : cell_weights(device_data_t::create(memory.cell_weights)),
          points_to_be_cached(
              device_data_t::create(memory.points_to_be_cached)),
          points_to_be_cached_offsets(
              device_data_t::create(memory.points_to_be_cached_offsets)),
          cell_list(device_data_t::create(memory.cell_list)),
          cell_colours(device_data_t::create(memory.cell_colours)),
          num_cell_colours(device_data_t::create(memory.num_cell_colours)),
          block_offsets(device_data_t::create(memory.block_offsets)) {
      shared_size = 0;
      for (MY_SIZE i = 1; i < memory.points_to_be_cached_offsets.size(); ++i) {
        shared_size = std::max<MY_SIZE>(
            shared_size, memory.points_to_be_cached_offsets[i] -
                             memory.points_to_be_cached_offsets[i - 1]);
      }
    }

    DeviceMemoryOfOneColour(const DeviceMemoryOfOneColour &other) = delete;
    DeviceMemoryOfOneColour &
    operator=(const DeviceMemoryOfOneColour &rhs) = delete;

    DeviceMemoryOfOneColour(DeviceMemoryOfOneColour &&) = default;
    DeviceMemoryOfOneColour &operator=(DeviceMemoryOfOneColour &&) = default;

    ~DeviceMemoryOfOneColour() {}
  };

  std::vector<DeviceMemoryOfOneColour> getDeviceMemoryOfOneColour() const {
    std::vector<DeviceMemoryOfOneColour> result;
    std::transform(colours.begin(), colours.end(),
                   std::inserter(result, result.end()),
                   [](const MemoryOfOneColour &memory) {
                     return DeviceMemoryOfOneColour(memory);
                   });
    return result;
  }
};

#endif /* end of include guard: COLOURING_HPP_PMK0HFCY */
// vim:set et sw=2 ts=2 fdm=marker:
