#ifndef COLOURING_HPP_PMK0HFCY
#define COLOURING_HPP_PMK0HFCY

#include <algorithm>
#include <bitset>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <map>
#include <set>
#include <memory>
#include <numeric>
#include <vector>

#include "problem.hpp"

template <unsigned PointDim = 1, unsigned EdgeDim = 1, bool SOA = false,
          typename DataType = float>
struct HierarchicalColourMemory {
  // I assume 64 colour is enough
  using colourset_t = Graph::colourset_t;
  struct MemoryOfOneColour {
    std::vector<DataType>
        edge_weights; // restructured so it can be indexed with tid
                      // it's a vector, because it's not necessarily block_size
                      // long; also I don't want mem. management just now
    std::vector<MY_SIZE> points_to_be_cached;
    std::vector<MY_SIZE> points_to_be_cached_offsets = {0};
    // every thread caches the one with index
    // the multiple of its tid
    std::vector<MY_SIZE> edge_list; // same as before, just points to shared mem
                                    // computed from the above
    std::vector<std::uint8_t> edge_colours;     // the colour for each edge
                                                // in the block
    std::vector<std::uint8_t> num_edge_colours; // the number of edge colours
                                                // in each block
    std::vector<MY_SIZE> block_offsets = {0};   // where the is block in
                                                // the vectors above
  };
  std::vector<MemoryOfOneColour> colours;

  HierarchicalColourMemory(
      const Problem<PointDim, EdgeDim, SOA, DataType> &problem,
      const std::vector<MY_SIZE> &partition_vector = {}) {
    /* Algorithm:
     *   - loop through `block_size` blocks
     *     - determine the points written to (not the same as points used)
     *     - get the set of available colours
     *     - among these, choose the one with the minimal number of sets
     *     - colour all of the points above
     *     - assign block to colour:
     *       - add weights to edge_weights
     *       - add used(!) points to `points_to_be_cached` (without duplicates)
     *            (set maybe?)
     *       - add edges to edge_list
     *     - if new colour is needed, `colours.push_back()` then the same as
     *       above
     *   - done
     */
    const Graph &graph = problem.graph;
    const MY_SIZE block_size = problem.block_size;
    std::vector<colourset_t> point_colours(graph.numPoints(), 0);
    colourset_t used_colours;
    std::vector<MY_SIZE> set_sizes;
    std::vector<std::uint8_t> block_colours(graph.numEdges());
    std::vector<DataType> tmp_edge_weights(problem.edge_weights.cbegin(),
                                           problem.edge_weights.cend());
    std::vector<MY_SIZE> blocks;
    std::vector<std::pair<MY_SIZE, MY_SIZE>> partition_to_edge;
    std::vector<MY_SIZE> edge_inverse_permutation(graph.numEdges());
    if (partition_vector.size() == problem.graph.numEdges()) {
      for (MY_SIZE i = 0; i < partition_vector.size(); ++i) {
        partition_to_edge.push_back(std::make_pair(partition_vector[i], i));
      }
      std::stable_sort(partition_to_edge.begin(), partition_to_edge.end());
      MY_SIZE tid = 0;
      blocks.push_back(0);
      edge_inverse_permutation[0] = partition_to_edge[0].second;
      for (MY_SIZE i = 1; i < partition_vector.size(); ++i) {
        edge_inverse_permutation[i] = partition_to_edge[i].second;
        if (++tid == block_size) {
          tid = 0;
          blocks.push_back(i);
          continue;
        }
        if (partition_to_edge[i - 1].first != partition_to_edge[i].first) {
          tid = 0;
          blocks.push_back(i);
        }
      }
      assert(blocks.back() != partition_vector.size());
      blocks.push_back(partition_vector.size());
      std::array<typename std::vector<DataType>::iterator, EdgeDim> begins;
      for (MY_SIZE i = 0; i < EdgeDim; ++i) {
        begins[i] =
            std::next(tmp_edge_weights.begin(), i * problem.graph.numEdges());
      }
      typename std::vector<DataType>::iterator end_first =
          std::next(tmp_edge_weights.begin(), problem.graph.numEdges());
      reorderDataInverseVectorSOA<EdgeDim, DataType, MY_SIZE>(
          begins, end_first, edge_inverse_permutation);
    } else {
      for (MY_SIZE i = 0; i < problem.graph.numEdges(); i += block_size) {
        blocks.emplace_back(i);
      }
      assert(blocks.back() != problem.graph.numEdges());
      blocks.push_back(problem.graph.numEdges());
      partition_to_edge.resize(problem.graph.numEdges());
      for (MY_SIZE i = 0; i < graph.numEdges(); ++i) {
        partition_to_edge[i] = std::make_pair(0, i);
        edge_inverse_permutation[i] = i;
      }
    }
    assert(blocks.size() >= 2);
    for (MY_SIZE i = 1; i < blocks.size(); ++i) {
      MY_SIZE block_from = blocks[i - 1];
      MY_SIZE block_to = blocks[i];
      assert(block_to != block_from);
      assert(block_to - block_from <= block_size);
      colourset_t occupied_colours =
          getOccupiedColours(graph.edge_to_node, block_from, block_to,
                             point_colours, partition_to_edge);
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
      MY_SIZE colour = Graph::getAvailableColour(available_colours, set_sizes);
      ++set_sizes[colour];
      colourBlock(block_from, block_to, colour, point_colours, problem, colours,
                  partition_to_edge, block_colours, tmp_edge_weights);
      colours[colour].block_offsets.push_back(
          colours[colour].block_offsets.back() + block_to - block_from);
    }
    copyEdgeWeights(problem, block_colours, tmp_edge_weights);
    edgeListSOA();
  }

private:
  static colourset_t getOccupiedColours(
      const data_t<MY_SIZE, 2> &edge_to_node, MY_SIZE from, MY_SIZE to,
      const std::vector<colourset_t> &point_colours,
      const std::vector<std::pair<MY_SIZE, MY_SIZE>> &partition_to_edge) {
    colourset_t result;
    for (MY_SIZE i = from; i < to; ++i) {
      MY_SIZE edge_ind = partition_to_edge[i].second;
      MY_SIZE point_left = edge_to_node[2 * edge_ind + 0];
      MY_SIZE point_right = edge_to_node[2 * edge_ind + 1];
      result |= point_colours[point_left];
      result |= point_colours[point_right];
    }
    return result;
  }

  /*
   * Colours every point written by the edges in the block and also
   * collects all points accessed by the edges in the block.
   */
  void
  colourBlock(MY_SIZE from, MY_SIZE to, MY_SIZE colour_ind,
              std::vector<colourset_t> &point_colours,
              const Problem<PointDim, EdgeDim, SOA, DataType> &problem,
              std::vector<MemoryOfOneColour> &colours,
              const std::vector<std::pair<MY_SIZE, MY_SIZE>> &partition_to_edge,
              std::vector<std::uint8_t> &block_colours,
              std::vector<DataType> &tmp_edge_weights) {
    const Graph &graph = problem.graph;
    const data_t<MY_SIZE, 2> &edge_to_node = graph.edge_to_node;
    colourset_t colourset(1ull << colour_ind);
    MemoryOfOneColour &colour = colours[colour_ind];
    const MY_SIZE colour_from = colour.edge_colours.size();
    std::map<MY_SIZE, std::vector<std::pair<MY_SIZE, MY_SIZE>>>
        points_to_edges; // points -> vector of (edge_ind, point_offset)
    for (MY_SIZE i = from; i < to; ++i) {
      MY_SIZE edge_ind = partition_to_edge[i].second;
      MY_SIZE point_right = edge_to_node[2 * edge_ind + 1];
      MY_SIZE point_left = edge_to_node[2 * edge_ind];
      point_colours[point_right] |= colourset;
      point_colours[point_left] |= colourset;
      block_colours[i] = colour_ind;
      points_to_edges[point_right].emplace_back(i - from, 1);
      points_to_edges[point_left].emplace_back(i - from, 0);
    }
    std::vector<MY_SIZE> c_edge_list(2 * (to - from));
    std::vector<MY_SIZE> points_to_be_cached;
    for (const auto &t : points_to_edges) {
      MY_SIZE point_ind = t.first;
      const std::vector<std::pair<MY_SIZE, MY_SIZE>> &edge_inds = t.second;
      for (const std::pair<MY_SIZE, MY_SIZE> e : edge_inds) {
        MY_SIZE ind = e.first;
        MY_SIZE offset = e.second;
        c_edge_list[2 * ind + offset] = points_to_be_cached.size();
      }
      points_to_be_cached.push_back(point_ind);
    }
    colour.edge_list.insert(colour.edge_list.end(), c_edge_list.begin(),
                            c_edge_list.end());
    colourEdges(to - from, colour, c_edge_list, points_to_be_cached.size());
    const MY_SIZE colour_to = colour.edge_colours.size();
    sortEdgesByColours(colour_from, colour_to, from, to, colour_ind,
                       tmp_edge_weights);
    // permuteCachedPoints(points_to_be_cached, colour_from, colour_to,
    // colour_ind);
    colour.points_to_be_cached.insert(colour.points_to_be_cached.end(),
                                      points_to_be_cached.begin(),
                                      points_to_be_cached.end());
    colour.points_to_be_cached_offsets.push_back(
        colour.points_to_be_cached.size());
  }

  void colourEdges(MY_SIZE block_size, MemoryOfOneColour &block,
                   const std::vector<MY_SIZE> &edge_list, MY_SIZE num_point) {
    static std::vector<colourset_t> point_colours;
    point_colours.resize(num_point);
    memset(point_colours.data(), 0, sizeof(colourset_t) * point_colours.size());
    std::uint8_t num_edge_colours = 0;
    std::vector<MY_SIZE> set_sizes(64, 0);
    colourset_t used_colours;
    for (MY_SIZE i = 0; i < block_size; ++i) {
      colourset_t occupied_colours;
      occupied_colours |= point_colours[edge_list[2 * i + 0]];
      occupied_colours |= point_colours[edge_list[2 * i + 1]];
      colourset_t available_colours = ~occupied_colours & used_colours;
      if (available_colours.none()) {
        ++num_edge_colours;
        used_colours <<= 1;
        used_colours.set(0);
        available_colours = ~occupied_colours & used_colours;
      }
      std::uint8_t colour =
          Graph::getAvailableColour(available_colours, set_sizes);
      block.edge_colours.push_back(colour);
      ++set_sizes[colour];
      colourset_t colourset(1ull << colour);
      point_colours[edge_list[2 * i + 0]] |= colourset;
      point_colours[edge_list[2 * i + 1]] |= colourset;
    }
    block.num_edge_colours.push_back(num_edge_colours);
  }

  void sortEdgesByColours(MY_SIZE colour_from, MY_SIZE colour_to, MY_SIZE from,
                          MY_SIZE to, MY_SIZE block_colour,
                          std::vector<DataType> &tmp_edge_weights) {
    std::vector<std::tuple<std::uint8_t, MY_SIZE, MY_SIZE, MY_SIZE>> tmp(
        colour_to - colour_from);
    MemoryOfOneColour &memory = colours[block_colour];
    for (MY_SIZE i = 0; i < colour_to - colour_from; ++i) {
      const MY_SIZE j = i + colour_from;
      tmp[i] = std::make_tuple(memory.edge_colours[j], memory.edge_list[2 * j],
                               memory.edge_list[2 * j + 1], i);
    }
    std::stable_sort(
        tmp.begin(), tmp.end(),
        [](const std::tuple<std::uint8_t, MY_SIZE, MY_SIZE, MY_SIZE> &a,
           const std::tuple<std::uint8_t, MY_SIZE, MY_SIZE, MY_SIZE> &b) {
          return std::get<0>(a) < std::get<0>(b);
        });
    std::vector<MY_SIZE> inverse_permutation(colour_to - colour_from);
    for (MY_SIZE i = 0; i < colour_to - colour_from; ++i) {
      const MY_SIZE j = i + colour_from;
      std::tie(memory.edge_colours[j], memory.edge_list[2 * j],
               memory.edge_list[2 * j + 1], inverse_permutation[i]) = tmp[i];
    }
    std::array<typename std::vector<DataType>::iterator, EdgeDim> begins;
    for (MY_SIZE i = 0; i < EdgeDim; ++i) {
      MY_SIZE dim_pos =
          index<EdgeDim, true>(tmp_edge_weights.size() / EdgeDim, from, i);
      begins[i] = std::next(tmp_edge_weights.begin(), dim_pos);
    }
    typename std::vector<DataType>::iterator end_first =
        std::next(tmp_edge_weights.begin(), to);
    reorderDataInverseVectorSOA<EdgeDim, DataType, MY_SIZE>(
        begins, end_first, inverse_permutation);
  }

  void permuteCachedPoints(std::vector<MY_SIZE> &points_to_be_cached,
                           MY_SIZE colour_from, MY_SIZE colour_to,
                           MY_SIZE colour_ind) {
    std::set<MY_SIZE> seen_points;
    std::vector<MY_SIZE> &edge_list = colours[colour_ind].edge_list;
    std::vector<MY_SIZE> new_points_to_be_cached;
    new_points_to_be_cached.reserve(points_to_be_cached.size());
    std::vector<MY_SIZE> permutation(points_to_be_cached.size());
    for (MY_SIZE offset = 0; offset < 2; ++offset) {
      for (MY_SIZE i = 0; i < colour_to - colour_from; ++i) {
        MY_SIZE point_ind = edge_list[2 * (colour_from + i) + offset];
        assert(point_ind < permutation.size());
        auto r = seen_points.insert(point_ind);
        if (r.second) {
          permutation[point_ind] = new_points_to_be_cached.size();
          new_points_to_be_cached.push_back(points_to_be_cached[point_ind]);
        }
        edge_list[2 * (colour_from + i) + offset] = permutation[point_ind];
      }
    }
    assert(new_points_to_be_cached.size() == points_to_be_cached.size());
    std::copy(new_points_to_be_cached.begin(), new_points_to_be_cached.end(),
              points_to_be_cached.begin());
  }

  void edgeListSOA() {
    for (MemoryOfOneColour &memory : colours) {
      AOStoSOA<2>(memory.edge_list);
    }
  }

  void copyEdgeWeights(const Problem<PointDim, EdgeDim, SOA, DataType> &problem,
                       const std::vector<std::uint8_t> &block_colours,
                       const std::vector<DataType> &tmp_edge_weights) {
    for (MY_SIZE i = 0; i < colours.size(); ++i) {
      colours[i].edge_weights.reserve(colours[i].edge_list.size() / 2);
    }
    for (MY_SIZE d = 0; d < EdgeDim; ++d) {
      for (MY_SIZE i = 0; i < problem.graph.numEdges(); ++i) {
        MY_SIZE colour = block_colours[i];
        DataType weight = tmp_edge_weights[index<EdgeDim, true>(
            problem.graph.numEdges(), i, d)];
        colours[colour].edge_weights.push_back(weight);
      }
    }
  }

public:
  /*******************
  *  Device layout  *
  *******************/
  struct DeviceMemoryOfOneColour {
    device_data_t<DataType> edge_weights;
    device_data_t<MY_SIZE> points_to_be_cached, points_to_be_cached_offsets;
    device_data_t<MY_SIZE> edge_list;
    device_data_t<std::uint8_t> edge_colours;
    device_data_t<std::uint8_t> num_edge_colours;
    device_data_t<MY_SIZE> block_offsets;
    MY_SIZE shared_size;

    DeviceMemoryOfOneColour(const MemoryOfOneColour &memory)
        : edge_weights(memory.edge_weights),
          points_to_be_cached(memory.points_to_be_cached),
          points_to_be_cached_offsets(memory.points_to_be_cached_offsets),
          edge_list(memory.edge_list), edge_colours(memory.edge_colours),
          num_edge_colours(memory.num_edge_colours),
          block_offsets(memory.block_offsets) {
      shared_size = 0;
      for (MY_SIZE i = 1; i < memory.points_to_be_cached_offsets.size(); ++i) {
        shared_size =
            std::max<MY_SIZE>(shared_size,
                              memory.points_to_be_cached_offsets[i] -
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
