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
#include <vector>

#include "problem.hpp"

template <unsigned Dim = 1, bool SOA = false, typename DataType = float>
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
  };
  std::vector<MemoryOfOneColour> colours;

  HierarchicalColourMemory(const Problem<Dim, SOA, DataType> &problem) {
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
    for (MY_SIZE block_from = 0; block_from < graph.numEdges();
         block_from += block_size) {
      MY_SIZE block_to = std::min(graph.numEdges(), block_from + block_size);
      colourset_t occupied_colours = getOccupiedColours(
          graph.edge_to_node, block_from, block_to, point_colours);
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
      colourBlock(block_from, block_to, colour, point_colours, problem,
                  colours);
    }
    edgeListSOA();
  }

private:
  static colourset_t
  getOccupiedColours(const data_t<MY_SIZE, 2> &edge_to_node, MY_SIZE from,
                     MY_SIZE to,
                     const std::vector<colourset_t> &point_colours) {
    colourset_t result;
    for (MY_SIZE i = from; i < to; ++i) {
      MY_SIZE point_left = edge_to_node[2 * i + 0];
      MY_SIZE point_right = edge_to_node[2 * i + 1];
      result |= point_colours[point_left];
      result |= point_colours[point_right];
    }
    return result;
  }

  /*
   * Colours every point written by the edges in the block and also
   * collects all points accessed by the edges in the block.
   */
  void colourBlock(MY_SIZE from, MY_SIZE to, MY_SIZE colour_ind,
                   std::vector<colourset_t> &point_colours,
                   const Problem<Dim, SOA, DataType> &problem,
                   std::vector<MemoryOfOneColour> &colours) {
    const Graph &graph = problem.graph;
    const data_t<MY_SIZE, 2> &edge_to_node = graph.edge_to_node;
    colourset_t colourset(1ull << colour_ind);
    MemoryOfOneColour &colour = colours[colour_ind];
    const MY_SIZE colour_from = colour.edge_colours.size();
    std::map<MY_SIZE, std::vector<std::pair<MY_SIZE, MY_SIZE>>>
        points_to_edges; // points -> vector of (edge_ind, point_offset)
    for (MY_SIZE i = from; i < to; ++i) {
      MY_SIZE point_right = edge_to_node[2 * i + 1];
      MY_SIZE point_left = edge_to_node[2 * i];
      point_colours[point_right] |= colourset;
      point_colours[point_left] |= colourset;
      colour.edge_weights.push_back(problem.edge_weights[i]);
      points_to_edges[point_right].emplace_back(i, 1);
      points_to_edges[point_left].emplace_back(i, 0);
    }
    std::vector<MY_SIZE> c_edge_list(2 * (to - from));
    std::vector<MY_SIZE> points_to_be_cached;
    for (const auto &t : points_to_edges) {
      MY_SIZE point_ind = t.first;
      const std::vector<std::pair<MY_SIZE, MY_SIZE>> &edge_inds = t.second;
      for (const std::pair<MY_SIZE, MY_SIZE> e : edge_inds) {
        MY_SIZE ind = e.first;
        MY_SIZE offset = e.second;
        c_edge_list[2 * (ind - from) + offset] = points_to_be_cached.size();
      }
      points_to_be_cached.push_back(point_ind);
    }
    colour.edge_list.insert(colour.edge_list.end(), c_edge_list.begin(),
                            c_edge_list.end());
    colour.points_to_be_cached.insert(colour.points_to_be_cached.end(),
                                      points_to_be_cached.begin(),
                                      points_to_be_cached.end());
    colour.points_to_be_cached_offsets.push_back(
        colour.points_to_be_cached.size());
    colourEdges(to - from, colour, c_edge_list, points_to_be_cached.size());
    const MY_SIZE colour_to = colour.edge_colours.size();
    sortEdgesByColours(colour_from, colour_to, colour_ind);
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

  void sortEdgesByColours(MY_SIZE from, MY_SIZE to, MY_SIZE block_colour) {
    std::vector<std::tuple<std::uint8_t, MY_SIZE, MY_SIZE, DataType>> tmp(to -
                                                                          from);
    MemoryOfOneColour &memory = colours[block_colour];
    for (MY_SIZE i = 0; i < to - from; ++i) {
      const MY_SIZE j = i + from;
      tmp[i] =
          std::make_tuple(memory.edge_colours[j], memory.edge_list[2 * j],
                          memory.edge_list[2 * j + 1], memory.edge_weights[j]);
    }
    std::stable_sort(
        tmp.begin(), tmp.end(),
        [](const std::tuple<std::uint8_t, MY_SIZE, MY_SIZE, DataType> &a,
           const std::tuple<std::uint8_t, MY_SIZE, MY_SIZE, DataType> &b) {
          return std::get<0>(a) < std::get<0>(b);
        });
    for (MY_SIZE i = 0; i < to - from; ++i) {
      const MY_SIZE j = i + from;
      std::tie(memory.edge_colours[j], memory.edge_list[2 * j],
               memory.edge_list[2 * j + 1], memory.edge_weights[j]) = tmp[i];
    }
  }

  void edgeListSOA () {
    for (MemoryOfOneColour &memory : colours) {
      AOStoSOA<2>(memory.edge_list);
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
    MY_SIZE shared_size;

    DeviceMemoryOfOneColour(const MemoryOfOneColour &memory)
        : edge_weights(memory.edge_weights),
          points_to_be_cached(memory.points_to_be_cached),
          points_to_be_cached_offsets(memory.points_to_be_cached_offsets),
          edge_list(memory.edge_list), edge_colours(memory.edge_colours),
          num_edge_colours(memory.num_edge_colours) {
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
