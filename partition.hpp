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

std::vector<idx_t> partitionMetis(const Mesh &graph, MY_SIZE block_size,
                                  real_t tolerance,
                                  idx_t options[METIS_NOPTIONS] = NULL);

inline std::vector<idx_t>
partitionMetisEnh(const Mesh &graph, MY_SIZE block_size, real_t tolerance,
                  idx_t options[METIS_NOPTIONS] = NULL) {
  MY_SIZE block_size2 = std::floor(static_cast<double>(block_size) / tolerance);
  real_t tolerance2 = (block_size + 0.1) / static_cast<double>(block_size2);
  return partitionMetis(graph, block_size2, tolerance2, options);
}

#endif /* end of include guard: PARTITION_HPP_QTFZLMGZ */

/* vim:set et sw=2 ts=2 fdm=marker: */
