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

template <bool SOA = false> struct HierarchicalColourMemory {
  using colourset_t = Mesh::colourset_t;
  struct MemoryOfOneColour {
    std::vector<std::vector<char>>
        cell_weights; // restructured so it can be indexed with tid
    // it's a vector, because it's not necessarily block_size long
    std::vector<MY_SIZE> points_to_be_cached;
    std::vector<MY_SIZE> points_to_be_cached_offsets = {0};
    // every thread caches the one with index
    // the multiple of its tid
    std::vector<std::vector<MY_SIZE>> cell_list;
    // same as before, just points to shared mem
    // computed from the above
    std::vector<std::uint8_t> cell_colours;     // the colour for each cell
                                                // in the block
    std::vector<std::uint8_t> num_cell_colours; // the number of cell colours
                                                // in each block
    std::vector<MY_SIZE> block_offsets = {0};   // where the is block in
                                                // the vectors above
  };
  std::vector<MemoryOfOneColour> colours;

  HierarchicalColourMemory(const Problem<SOA> &problem,
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
    const MY_SIZE block_size = problem.block_size;
    std::vector<colourset_t> point_colours(mesh.numPoints(0), 0);
    colourset_t used_colours;
    std::vector<MY_SIZE> set_sizes;
    std::vector<std::uint8_t> block_colours(mesh.numCells());
    std::vector<MY_SIZE> blocks;
    std::vector<std::pair<MY_SIZE, MY_SIZE>> partition_to_cell;
    std::vector<MY_SIZE> cell_weight_inv_permutation(mesh.numCells());
    if (partition_vector.size() == problem.mesh.numCells()) {
      for (MY_SIZE i = 0; i < partition_vector.size(); ++i) {
        partition_to_cell.push_back(std::make_pair(partition_vector[i], i));
      }
      std::stable_sort(partition_to_cell.begin(), partition_to_cell.end());
      MY_SIZE tid = 0;
      blocks.push_back(0);
      cell_weight_inv_permutation[0] = partition_to_cell[0].second;
      bool warning = false;
      for (MY_SIZE i = 1; i < partition_vector.size(); ++i) {
        cell_weight_inv_permutation[i] = partition_to_cell[i].second;
        if (++tid == block_size) {
          if (!partition_to_cell[i - 1].first != partition_to_cell[i].first) {
            warning = true;
          }
          tid = 0;
          blocks.push_back(i);
          continue;
        }
        if (partition_to_cell[i - 1].first != partition_to_cell[i].first) {
          if (tid < block_size / 2) {
            std::cout << "Warning: size of this block is " << tid << std::endl;
          }
          tid = 0;
          blocks.push_back(i);
        }
      }
      if (warning) {
        std::cout
            << "Warning: new block started because of maximum block size"
            << std::endl;
      }
      assert(blocks.back() != partition_vector.size());
      blocks.push_back(partition_vector.size());
    } else {
      for (MY_SIZE i = 0; i < problem.mesh.numCells(); i += block_size) {
        blocks.emplace_back(i);
      }
      assert(blocks.back() != problem.mesh.numCells());
      blocks.push_back(problem.mesh.numCells());
      partition_to_cell.resize(problem.mesh.numCells());
      for (MY_SIZE i = 0; i < mesh.numCells(); ++i) {
        partition_to_cell[i] = std::make_pair(0, i);
        cell_weight_inv_permutation[i] = i;
      }
    }
    assert(blocks.size() >= 2);
    for (MY_SIZE i = 1; i < blocks.size(); ++i) {
      MY_SIZE block_from = blocks[i - 1];
      MY_SIZE block_to = blocks[i];
      assert(block_to != block_from);
      assert(block_to - block_from <= block_size);
      colourset_t occupied_colours =
          getOccupiedColours(mesh.cell_to_node[0], block_from, block_to,
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
        colours.back().cell_list.resize(mesh.numMappings());
        colours.back().cell_weights.resize(problem.cell_weights.size());
      }
      MY_SIZE colour = Mesh::getAvailableColour(available_colours, set_sizes);
      ++set_sizes[colour];
      colourBlock(block_from, block_to, colour, point_colours, problem, colours,
                  partition_to_cell, block_colours,
                  cell_weight_inv_permutation);
      colours[colour].block_offsets.push_back(
          colours[colour].block_offsets.back() + block_to - block_from);
    }
    copyCellWeights(problem, block_colours, cell_weight_inv_permutation);
    cellListSOA(problem.mesh);
  }

private:
  static colourset_t getOccupiedColours(
      const data_t &cell_to_node, MY_SIZE from, MY_SIZE to,
      const std::vector<colourset_t> &point_colours,
      const std::vector<std::pair<MY_SIZE, MY_SIZE>> &partition_to_cell) {
    colourset_t result;
    for (MY_SIZE i = from; i < to; ++i) {
      MY_SIZE cell_ind = partition_to_cell[i].second;
      for (MY_SIZE j = 0; j < cell_to_node.getDim(); ++j) {
        result |= point_colours[cell_to_node.operator[]<MY_SIZE>(
            cell_to_node.getDim() * cell_ind + j)];
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
              const Problem<SOA> &problem,
              std::vector<MemoryOfOneColour> &colours,
              const std::vector<std::pair<MY_SIZE, MY_SIZE>> &partition_to_cell,
              std::vector<std::uint8_t> &block_colours,
              std::vector<MY_SIZE> &cell_weight_inv_permutation) {
    const Mesh &mesh = problem.mesh;
    const data_t &cell_to_node = mesh.cell_to_node[0];
    const MY_SIZE mesh_dim = cell_to_node.getDim();
    colourset_t colourset{};
    colourset.set(colour_ind);
    MemoryOfOneColour &colour = colours[colour_ind];
    const MY_SIZE colour_from = colour.cell_colours.size();
    std::map<MY_SIZE, std::vector<std::pair<MY_SIZE, MY_SIZE>>>
        points_to_cells; // points -> vector of (cell_ind, point_offset)
    for (MY_SIZE i = from; i < to; ++i) {
      MY_SIZE cell_ind = partition_to_cell[i].second;
      for (MY_SIZE offset = 0; offset < mesh_dim; ++offset) {
        point_colours[cell_to_node.operator[]<MY_SIZE>(mesh_dim *cell_ind +
                                                       offset)] |= colourset;
        points_to_cells[cell_to_node.operator[]<MY_SIZE>(mesh_dim *cell_ind +
                                                         offset)]
            .emplace_back(i - from, offset);
      }
      block_colours[i] = colour_ind;
    }
    std::vector<MY_SIZE> c_cell_list(mesh_dim * (to - from));
    std::vector<MY_SIZE> points_to_be_cached;
    for (const auto &t : points_to_cells) {
      MY_SIZE point_ind = t.first;
      const std::vector<std::pair<MY_SIZE, MY_SIZE>> &cell_inds = t.second;
      for (const std::pair<MY_SIZE, MY_SIZE> c : cell_inds) {
        MY_SIZE ind = c.first;
        MY_SIZE offset = c.second;
        c_cell_list[mesh_dim * ind + offset] = points_to_be_cached.size();
      }
      points_to_be_cached.push_back(point_ind);
    }
    colour.cell_list[0].insert(colour.cell_list[0].end(), c_cell_list.begin(),
                               c_cell_list.end());
    for (unsigned mapping_ind = 1; mapping_ind < mesh.numMappings();
         ++mapping_ind) {
      const MY_SIZE _mesh_dim = mesh.cell_to_node[mapping_ind].getDim();
      for (MY_SIZE i = from; i < to; ++i) {
        const MY_SIZE cell_ind = partition_to_cell[i].second;
        colour.cell_list[mapping_ind].insert(
            colour.cell_list[mapping_ind].end(),
            mesh.cell_to_node[mapping_ind].cbegin<MY_SIZE>() +
                cell_ind * _mesh_dim,
            mesh.cell_to_node[mapping_ind].cbegin<MY_SIZE>() +
                (cell_ind + 1) * _mesh_dim);
      }
    }
    colourCells(to - from, colour, c_cell_list, points_to_be_cached.size(),
                mesh_dim);
    const MY_SIZE colour_to = colour.cell_colours.size();
    sortCellsByColours(colour_from, colour_to, from, colour_ind,
                       cell_weight_inv_permutation, mesh);
    // permuteCachedPoints(points_to_be_cached, colour_from,
    //                                      colour_to, colour_ind, mesh_dim);
    colour.points_to_be_cached.insert(colour.points_to_be_cached.end(),
                                      points_to_be_cached.begin(),
                                      points_to_be_cached.end());
    colour.points_to_be_cached_offsets.push_back(
        colour.points_to_be_cached.size());
  }

  void colourCells(MY_SIZE block_size, MemoryOfOneColour &block,
                   const std::vector<MY_SIZE> &cell_list, MY_SIZE num_point,
                   MY_SIZE mesh_dim) {
    static std::vector<colourset_t> point_colours;
    point_colours.resize(num_point);
    memset(point_colours.data(), 0, sizeof(colourset_t) * point_colours.size());
    std::uint8_t num_cell_colours = 0;
    std::vector<MY_SIZE> set_sizes(64, 0);
    colourset_t used_colours;
    for (MY_SIZE i = 0; i < block_size; ++i) {
      colourset_t occupied_colours;
      for (MY_SIZE offset = 0; offset < mesh_dim; ++offset) {
        occupied_colours |= point_colours[cell_list[mesh_dim * i + offset]];
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
      for (MY_SIZE offset = 0; offset < mesh_dim; ++offset) {
        point_colours[cell_list[mesh_dim * i + offset]] |= colourset;
      }
    }
    block.num_cell_colours.push_back(num_cell_colours);
  }

  void sortCellsByColours(MY_SIZE colour_from, MY_SIZE colour_to, MY_SIZE from,
                          MY_SIZE block_colour,
                          std::vector<MY_SIZE> &cell_weight_inv_permutation,
                          const Mesh &mesh) {
    std::vector<std::tuple<std::uint8_t, MY_SIZE, MY_SIZE>> tmp(colour_to -
                                                                colour_from);
    MemoryOfOneColour &memory = colours[block_colour];
    for (MY_SIZE i = 0; i < colour_to - colour_from; ++i) {
      const MY_SIZE j = i + colour_from;
      tmp[i] = std::make_tuple(memory.cell_colours[j], i,
                               cell_weight_inv_permutation[i + from]);
    }
    std::sort(tmp.begin(), tmp.end());
    std::vector<MY_SIZE> inverse_permutation(colour_to - colour_from);
    for (MY_SIZE i = 0; i < colour_to - colour_from; ++i) {
      const MY_SIZE j = i + colour_from;
      std::tie(memory.cell_colours[j], inverse_permutation[i],
               cell_weight_inv_permutation[i + from]) = tmp[i];
    }
    for (unsigned mapping_ind = 0; mapping_ind < memory.cell_list.size();
         ++mapping_ind) {
      const unsigned mesh_dim = mesh.cell_to_node[mapping_ind].getDim();
      reorderDataInverseAOS<MY_SIZE>(memory.cell_list[mapping_ind].begin() +
                                         colour_from * mesh_dim,
                                     inverse_permutation, mesh_dim);
    }
  }

  void permuteCachedPoints(std::vector<MY_SIZE> &points_to_be_cached,
                           MY_SIZE colour_from, MY_SIZE colour_to,
                           MY_SIZE colour_ind, MY_SIZE mesh_dim) {
    std::set<MY_SIZE> seen_points;
    std::vector<MY_SIZE> &cell_list = colours[colour_ind].cell_list[0];
    std::vector<MY_SIZE> new_points_to_be_cached;
    new_points_to_be_cached.reserve(points_to_be_cached.size());
    std::vector<MY_SIZE> permutation(points_to_be_cached.size());
    for (MY_SIZE offset = 0; offset < mesh_dim; ++offset) {
      for (MY_SIZE i = 0; i < colour_to - colour_from; ++i) {
        MY_SIZE point_ind = cell_list[mesh_dim * (colour_from + i) + offset];
        assert(point_ind < permutation.size());
        auto r = seen_points.insert(point_ind);
        if (r.second) {
          permutation[point_ind] = new_points_to_be_cached.size();
          new_points_to_be_cached.push_back(points_to_be_cached[point_ind]);
        }
        cell_list[mesh_dim * (colour_from + i) + offset] =
            permutation[point_ind];
      }
    }
    assert(new_points_to_be_cached.size() == points_to_be_cached.size());
    std::copy(new_points_to_be_cached.begin(), new_points_to_be_cached.end(),
              points_to_be_cached.begin());
  }

  void cellListSOA(const Mesh &mesh) {
    for (MemoryOfOneColour &memory : colours) {
      for (unsigned mapping_ind = 0; mapping_ind < memory.cell_list.size();
           ++mapping_ind) {
        AOStoSOA(memory.cell_list[mapping_ind],
                 mesh.cell_to_node[mapping_ind].getDim());
      }
    }
  }

  void
  copyCellWeights(const Problem<SOA> &problem,
                  const std::vector<std::uint8_t> &block_colours,
                  const std::vector<MY_SIZE> &cell_weight_inv_permutation) {
    for (unsigned cw_ind = 0; cw_ind < problem.cell_weights.size(); ++cw_ind) {
      const unsigned type_size = problem.cell_weights[cw_ind].getTypeSize();
      for (MY_SIZE i = 0; i < colours.size(); ++i) {
        colours[i].cell_weights[cw_ind].reserve(
            colours[i].cell_list[0].size() /
            problem.mesh.cell_to_node[0].getDim() * type_size);
      }
      for (MY_SIZE d = 0; d < problem.cell_weights[cw_ind].getDim(); ++d) {
        for (MY_SIZE i = 0; i < problem.mesh.numCells(); ++i) {
          MY_SIZE colour = block_colours[i];
          MY_SIZE glob_index = index<true>(
              problem.mesh.numCells(), cell_weight_inv_permutation[i],
              problem.cell_weights[cw_ind].getDim(), d);
          colours[colour].cell_weights[cw_ind].insert(
              colours[colour].cell_weights[cw_ind].end(),
              problem.cell_weights[cw_ind].cbegin() + glob_index * type_size,
              problem.cell_weights[cw_ind].cbegin() +
                  (glob_index + 1) * type_size);
        }
      }
    }
  }

public:
  /*******************
   *  Device layout  *
   *******************/
  struct DeviceMemoryOfOneColour {
    struct d_vector {
      std::vector<device_data_t> data;
      device_data_t ptrs;
    };

    d_vector cell_weights;
    device_data_t points_to_be_cached, points_to_be_cached_offsets;
    d_vector cell_list;
    device_data_t cell_colours;
    device_data_t num_cell_colours;
    device_data_t block_offsets;
    MY_SIZE shared_size;

    DeviceMemoryOfOneColour(const MemoryOfOneColour &memory)
        : cell_weights(initVector(memory.cell_weights)),
          points_to_be_cached(
              device_data_t::create(memory.points_to_be_cached)),
          points_to_be_cached_offsets(
              device_data_t::create(memory.points_to_be_cached_offsets)),
          cell_list(initVector(memory.cell_list)),
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

  private:
    template <typename T>
    static d_vector initVector(const std::vector<std::vector<T>> &v) {
      std::vector<const T *> collector;
      std::vector<device_data_t> data;
      for (unsigned i = 0; i < v.size(); ++i) {
        data.emplace_back(device_data_t::create(v[i]));
        collector.push_back(static_cast<const T *>(data.back()));
      }
      return {std::move(data), std::move(device_data_t::create(collector))};
    }
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
