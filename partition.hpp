#ifndef PARTITION_HPP_QTFZLMGZ
#define PARTITION_HPP_QTFZLMGZ

#include "mesh.hpp"
#include <cmath>
#include <limits>
#include <metis.h>
#include <vector>

struct MetisError {
  int code;
};

std::vector<idx_t> partitionMetis(const Mesh<2> &graph, MY_SIZE block_size,
                                  real_t tolerance,
                                  idx_t options[METIS_NOPTIONS] = NULL) {
  assert(graph.numCells() < std::numeric_limits<idx_t>::max());
  assert(graph.numPoints() < std::numeric_limits<idx_t>::max());
  GraphCSR<idx_t> csr(graph.numPoints(), graph.numCells(), graph.cell_to_node);
  idx_t nvtxs = graph.numPoints(), ncon = 1;
  idx_t nparts = std::ceil(static_cast<float>(graph.numPoints()) / block_size);
  std::vector<idx_t> partition(graph.numPoints());
  idx_t objval;
  real_t ubvec[1] = {tolerance};
  int r = METIS_PartGraphKway(&nvtxs, &ncon, csr.pointIndices(),
                              csr.cellEndpoints(), NULL, NULL, NULL, &nparts,
                              NULL, ubvec, options, &objval, partition.data());
  if (r != METIS_OK) {
    throw MetisError{r};
  }
  return partition;
}

std::vector<idx_t> partitionMetisEnh(const Mesh<2> &graph, MY_SIZE block_size,
                                     real_t tolerance,
                                     idx_t options[METIS_NOPTIONS] = NULL) {
  MY_SIZE block_size2 = std::floor(static_cast<double>(block_size) / tolerance);
  real_t tolerance2 = (block_size + 0.1) / static_cast<double>(block_size2);
  return partitionMetis(graph, block_size2, tolerance2, options);
}

#endif /* end of include guard: PARTITION_HPP_QTFZLMGZ */

/* vim:set et sw=2 ts=2 fdm=marker: */
