#include "partition.hpp"

std::vector<idx_t> partitionMetis(const Mesh &graph, MY_SIZE block_size,
                                  real_t tolerance,
                                  idx_t options[METIS_NOPTIONS]) {
  assert(graph.numCells() < std::numeric_limits<idx_t>::max());
  assert(graph.numPoints(0) < std::numeric_limits<idx_t>::max());
  assert(graph.cell_to_node[0].getDim() == 2);
  GraphCSR<idx_t> csr(graph.numPoints(0), graph.numCells(),
                      graph.cell_to_node[0]);
  idx_t nvtxs = graph.numPoints(0), ncon = 1;
  idx_t nparts = std::ceil(static_cast<float>(graph.numPoints(0)) / block_size);
  std::vector<idx_t> partition(graph.numPoints(0));
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

/* vim:set et sts=2 sw=2 ts=2: */
