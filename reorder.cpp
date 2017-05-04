#include "reorder.hpp"
#include "graph.hpp"

/*
 * Throw if the Scotch command `cmd` returns an error.
 */
#define SCOTCH_THROW(cmd)                                                      \
  do {                                                                         \
    int errorCode = cmd;                                                       \
    if (errorCode) {                                                           \
      /* Exterminate! Exterminate! */                                          \
      throw ScotchError{errorCode};                                            \
    }                                                                          \
  } while (0)

ScotchReorder::ScotchReorder(const Graph &_graph)
    : num_points(_graph.numPoints()), num_edges(_graph.numEdges()),
      csr(_graph.numPoints(), _graph.numEdges(), _graph.edge_list) {
  SCOTCH_THROW(SCOTCH_graphInit(&graph));
  SCOTCH_THROW(SCOTCH_graphBuild(&graph, 0, csr.num_points, csr.pointIndices(),
                                 NULL, NULL, NULL, csr.numArcs(),
                                 csr.edgeEndpoints(), NULL));
  SCOTCH_THROW(SCOTCH_graphCheck(&graph));
  try {
	  SCOTCH_THROW(SCOTCH_stratInit(&strategy));
	  SCOTCH_THROW(SCOTCH_stratGraphOrder(&strategy,strategy_string));
  } catch (ScotchError &) {
    SCOTCH_graphExit(&graph);
    throw;
  }
}

std::vector<SCOTCH_Num> ScotchReorder::reorder() {
  std::vector<SCOTCH_Num> permutation(num_points);
  SCOTCH_THROW(
      SCOTCH_graphOrder(&graph, &strategy, permutation.data(), NULL, NULL, NULL, NULL));
  return permutation;
}
