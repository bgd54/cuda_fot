#ifndef COLOURING_HPP_PMK0HFCY
#define COLOURING_HPP_PMK0HFCY

#include <algorithm>
#include <bitset>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <vector>

#include "problem.hpp"

template <unsigned Dim = 1, bool SOA = false> struct HierarchicalColourMemory {
  // I assume 64 colour is enough
  typedef std::bitset<64> colourset_t;
  struct MemoryOfOneColour {
    std::vector<float>
        edge_weights; // restructured so it can be indexed with tid
                      // it's a vector, because it's not necessarily block_size
                      // long; also I don't want mem. management just now
    // std::vector<std::vector<MY_SIZE>>
    //    points_to_be_cached; // every thread caches the one with index
    //                         // the multiple of its tid
    std::vector<MY_SIZE> read_points_to_be_cached;
    std::vector<MY_SIZE> read_points_to_be_cached_offsets = {0};
    std::vector<MY_SIZE> write_points_to_be_cached;
    std::vector<MY_SIZE> write_points_to_be_cached_offsets = {0};
    std::vector<MY_SIZE> edge_list; // same as before, just points to shared mem
                                    // computed from the above
    std::vector<std::uint8_t> edge_colours;     // the colour for each edge
                                                // in the block
    std::vector<std::uint8_t> num_edge_colours; // the number of edge colours
                                                // in each block
  };
  std::vector<MemoryOfOneColour> colours;

  HierarchicalColourMemory(MY_SIZE block_size,
                           const Problem<Dim, SOA> &problem) {
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
    assert(graph.edge_to_node.getDim() == 2);
    std::vector<colourset_t> point_colours(graph.numPoints(), 0);
    colourset_t used_colours;
    std::vector<unsigned long long> set_sizes;
    for (MY_SIZE block_from = 0; block_from < graph.numEdges();
         block_from += block_size) {
      MY_SIZE block_to = std::min(graph.numEdges(), block_from + block_size);
      auto tmp = getPointsWrittenTo(graph.edge_to_node, block_from, block_to,
                                    point_colours);
      // std::vector<MY_SIZE> point_written_to = tmp.first;
      colourset_t occupied_colours = tmp.second;
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
      MY_SIZE colour = getAvailableColour(available_colours, set_sizes);
      // std::cerr<< "Allocated: " << block_from << "-" << block_to << ": " <<
      // colour
      //    <<std::endl;
      ++set_sizes[colour];
      colourBlock(block_from, block_to, colour, point_colours, problem,
                  colours);
    }
  }

private:
  static std::pair<std::vector<MY_SIZE>, colourset_t>
  getPointsWrittenTo(const data_t<MY_SIZE> &edge_to_node, MY_SIZE from,
                     MY_SIZE to,
                     const std::vector<colourset_t> &point_colours) {
    colourset_t result;
    std::vector<MY_SIZE> points;
    // TODO handle more than one dimensions/data points written to
    for (MY_SIZE i = from; i < to; ++i) {
      MY_SIZE point = edge_to_node[2 * i + 1];
      result |= point_colours[point];
      points.push_back(point);
    }
    return std::make_pair(points, result);
  }

  static MY_SIZE
  getAvailableColour(colourset_t available_colours,
                     const std::vector<unsigned long long> &set_sizes) {
    assert(set_sizes.size() > 0);
    MY_SIZE colour = set_sizes.size();
    for (MY_SIZE i = 0; i < set_sizes.size(); ++i) {
      if (available_colours[i]) {
        if (colour >= set_sizes.size() || set_sizes[colour] > set_sizes[i]) {
          colour = i;
        }
      }
    }
    assert(colour < set_sizes.size());
    return colour;
  }

  /*
   * Colours every point written by the edges in the block and also
   * collects all points accessed by the edges in the block.
   */
  void colourBlock(MY_SIZE from, MY_SIZE to, MY_SIZE colour_ind,
                   std::vector<colourset_t> &point_colours,
                   const Problem<Dim, SOA> &problem,
                   std::vector<MemoryOfOneColour> &colours) {
    // TODO handle more than one dimensions/data points written to
    const Graph &graph = problem.graph;
    const data_t<MY_SIZE> &edge_to_node = graph.edge_to_node;
    colourset_t colourset(1ull << colour_ind);
    MemoryOfOneColour &colour = colours[colour_ind];
    std::map<MY_SIZE, std::vector<MY_SIZE>> read_points_to_edges;
    std::map<MY_SIZE, std::vector<MY_SIZE>> write_points_to_edges;
    for (MY_SIZE i = from; i < to; ++i) {
      MY_SIZE point_to = edge_to_node[2 * i + 1];
      MY_SIZE point_from = edge_to_node[2 * i];
      point_colours[point_to] |= colourset;
      colour.edge_weights.push_back(problem.edge_weights[i]);
      write_points_to_edges[point_to].push_back(i);
      read_points_to_edges[point_from].push_back(i);
    }
    std::vector<MY_SIZE> c_edge_list(2 * (to - from));
    std::vector<MY_SIZE> read_points_to_be_cached;
    std::vector<MY_SIZE> write_points_to_be_cached;
    for (const auto &t : read_points_to_edges) {
      MY_SIZE point_ind = t.first;
      const std::vector<MY_SIZE> &edge_inds = t.second;
      for (MY_SIZE e : edge_inds) {
        MY_SIZE offset = 0;
        c_edge_list[2 * (e - from) + offset] = read_points_to_be_cached.size();
      }
      read_points_to_be_cached.push_back(point_ind);
    }
    for (const auto &t : write_points_to_edges) {
      MY_SIZE point_ind = t.first;
      const std::vector<MY_SIZE> &edge_inds = t.second;
      for (MY_SIZE e : edge_inds) {
        MY_SIZE offset = 1;
        c_edge_list[2 * (e - from) + offset] = write_points_to_be_cached.size();
      }
      write_points_to_be_cached.push_back(point_ind);
    }
    colour.edge_list.insert(colour.edge_list.end(), c_edge_list.begin(),
                            c_edge_list.end());
    colour.read_points_to_be_cached.insert(
        colour.read_points_to_be_cached.end(), read_points_to_be_cached.begin(),
        read_points_to_be_cached.end());
    colour.write_points_to_be_cached.insert(
        colour.write_points_to_be_cached.end(),
        write_points_to_be_cached.begin(), write_points_to_be_cached.end());
    colour.read_points_to_be_cached_offsets.push_back(
        colour.read_points_to_be_cached.size());
    colour.write_points_to_be_cached_offsets.push_back(
        colour.write_points_to_be_cached.size());
    colourEdges(from, to, problem.graph, colour);
  }

  void colourEdges(MY_SIZE from, MY_SIZE to, const Graph &graph,
                   MemoryOfOneColour &block) {
    std::vector<std::uint8_t> point_colours(graph.numPoints(), 0);
    std::uint8_t num_edge_colours = 0;
    for (MY_SIZE i = from; i < to; ++i) {
      std::uint8_t colour = point_colours[graph.edge_to_node[2 * i + 1]]++;
      num_edge_colours = std::max<std::uint8_t>(num_edge_colours, colour + 1);
      block.edge_colours.push_back(colour);
    }
    block.num_edge_colours.push_back(num_edge_colours);
  }
};

#endif /* end of include guard: COLOURING_HPP_PMK0HFCY */
// vim:set et sw=2 ts=2 fdm=marker:
