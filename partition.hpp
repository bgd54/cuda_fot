#ifndef PARTITION_HPP_QTFZLMGZ
#define PARTITION_HPP_QTFZLMGZ

#include <metis.h>
#include <limits>
#include <cmath>
#include "graph.hpp"

struct MetisError {
  int code;
};

std::vector<idx_t> partitionMetis (const Graph &graph, 
    MY_SIZE block_size) {
  assert(graph.numEdges() < std::numeric_limits<idx_t>::max());
  assert(graph.numPoints() < std::numeric_limits<idx_t>::max());
  GraphCSR<idx_t> csr (graph.numPoints(), graph.numEdges(), graph.edge_to_node);
  idx_t nvtxs = graph.numPoints(), ncon = 1 /* TODO */;
  idx_t nparts = std::ceil(static_cast<float>(graph.numPoints())/block_size);
  std::vector<idx_t> partition (graph.numPoints());
  idx_t objval;
  int r = METIS_PartGraphKway(&nvtxs, &ncon, csr.pointIndices(),
      csr.edgeEndpoints(),
      NULL, NULL, NULL, &nparts, NULL, NULL /* TODO ubvec */, NULL,&objval,
      partition.data());
  if (r != METIS_OK) {
    throw MetisError{r};
  }
  return partition;
}

#endif /* end of include guard: PARTITION_HPP_QTFZLMGZ */

/* vim:set et sw=2 ts=2 fdm=marker: */
