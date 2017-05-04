#ifndef REORDER_HPP_IGDYRZTN
#define REORDER_HPP_IGDYRZTN

#include <algorithm>
#include <cassert>
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
    std::vector<std::pair<UnsignedType, UnsignedType>> incidence;
    for (UnsignedType i = 0; i < 2 * num_edges; ++i) {
      incidence.push_back(std::make_pair(edge_list[i], i / 2));
    }
    std::sort(incidence.begin(), incidence.end());
    UnsignedType point_ind = 0;
    point_indices[0] = 0;
    for (UnsignedType i = 0; i < 2 * num_edges; ++i) {
      UnsignedType current_point = incidence[i].first;
      UnsignedType current_edge = incidence[i].second;
      while (current_point != point_ind) {
        point_indices[++point_ind] = edge_endpoints.size();
      }
      assert(edge_list[2 * current_edge] != edge_list[2 * current_edge + 1]);
      if (edge_list[2 * current_edge] != current_point) {
        assert(edge_list[2 * current_edge + 1] == current_point);
        UnsignedType other_point = edge_list[2 * current_edge];
        if (point_indices[current_point] == edge_endpoints.size() ||
            std::find(edge_endpoints.begin() + point_indices[current_point],
                      edge_endpoints.end(),
                      other_point) == edge_endpoints.end()) {
          edge_endpoints.push_back(other_point);
        }
      } else {
        UnsignedType other_point = edge_list[2 * current_edge + 1];
        if (point_indices[current_point] == edge_endpoints.size() ||
            std::find(edge_endpoints.begin() + point_indices[current_point],
                      edge_endpoints.end(),
                      other_point) == edge_endpoints.end()) {
          edge_endpoints.push_back(other_point);
        }
      }
    }
    while (point_ind != num_points) {
      point_indices[++point_ind] = edge_endpoints.size();
    }
  }

  ~GraphCSR() { delete[] point_indices; }

  GraphCSR(const GraphCSR &other) = delete;
  GraphCSR &operator=(const GraphCSR &rhs) = delete;

  const UnsignedType *pointIndices() const { return point_indices; }

  const UnsignedType *edgeEndpoints() const { return edge_endpoints.data(); }

  UnsignedType numArcs () const { return edge_endpoints.size(); }

private:
  UnsignedType *point_indices;
  std::vector<UnsignedType> edge_endpoints;
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
