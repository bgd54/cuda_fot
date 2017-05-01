#ifndef REORDER_HPP_IGDYRZTN
#define REORDER_HPP_IGDYRZTN

#include <algorithm>
#include <limits>
#include <scotch.h>
#include <vector>

struct ScotchError {
  int errorCode;
};

class Graph;

/**
 * A graph in compressed sparse row format
 */
/* GraphCSR {{{1 */
template <class UnsignedType> struct GraphCSR {
  const UnsignedType num_points, num_edges;

  explicit GraphCSR(MY_SIZE _num_points, MY_SIZE _num_edges, MY_SIZE *edge_list)
      : num_points(_num_points), num_edges(_num_edges) {
    static_assert(std::size_t(std::numeric_limits<UnsignedType>::max()) <=
                      std::size_t(std::numeric_limits<MY_SIZE>::max()),
                  "GraphCSR: UnsignedType too small.");
    point_indices = new UnsignedType[num_points + 1];
    edge_endpoints = new UnsignedType[num_edges];
    std::vector<std::pair<UnsignedType, UnsignedType>> edges;
    for (UnsignedType i = 0; i < num_edges; ++i) {
      edges.push_back(std::make_pair(edge_list[2 * i], edge_list[2 * i + 1]));
    }
    std::sort(edges.begin(), edges.end());
    UnsignedType point_ind = 0;
    point_indices[0] = 0;
    for (UnsignedType i = 0; i < num_edges; ++i) {
      UnsignedType point = edges[i].first;
      while (point != point_ind) {
        point_indices[++point_ind] = i;
      }
      edge_endpoints[i] = edges[i].second;
    }
    while (point_ind != num_points) {
      point_indices[++point_ind] = num_edges;
    }
  }

  ~GraphCSR() {
    delete[] point_indices;
    delete[] edge_endpoints;
  }

  GraphCSR(const GraphCSR &other) = delete;
  GraphCSR &operator=(const GraphCSR &rhs) = delete;

  const UnsignedType *pointIndices() const { return point_indices; }

  const UnsignedType *edgeEndpoints() const { return edge_endpoints; }

private:
  UnsignedType *point_indices, *edge_endpoints;
};
/* 1}}} */

class ScotchReorder {
private:
  SCOTCH_Graph graph;
  const SCOTCH_Num num_points, num_edges;
  GraphCSR<SCOTCH_Num> csr;
  SCOTCH_Strat strategy;

public:
  explicit ScotchReorder(const Graph &_graph);

  ~ScotchReorder() {
    SCOTCH_graphExit(&graph);
    SCOTCH_stratExit(&strategy);
  }
  ScotchReorder(const ScotchReorder &other) = delete;
  ScotchReorder &operator=(const ScotchReorder &rhs) = delete;

  std::vector<SCOTCH_Num> reorder();

public:
  const char *strategy_string = "g";
};

#endif /* end of include guard: REORDER_HPP_IGDYRZTN */

/* vim:set et sw=2 ts=2 fdm=marker: */
