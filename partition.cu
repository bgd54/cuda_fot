#include "details/heuristical_partition.hpp"
#include "reorder.hpp"
#include <cassert>
#include <cmath>
#include <limits>

// -----------------------------------------------------------------------
// -                   Simple METIS k-way partitioning                   -
// -----------------------------------------------------------------------

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

// -----------------------------------------------------------------------
// -                       Our partitioning method                       -
// -----------------------------------------------------------------------

using namespace details::partitioning;
HeuristicalPartition::colourset_t
    HeuristicalPartition::priority_t::used_colours{};
constexpr MY_SIZE HeuristicalPartition::UNDEFINED;

std::vector<MY_SIZE> partitionOurs(const Mesh &mesh, MY_SIZE block_size,
                                   MY_SIZE coarse_block_size, real_t tolerance,
                                   idx_t metis_options[METIS_NOPTIONS]) {
  Mesh cell_to_cell = mesh.getCellToCellGraph();
  GraphCSR<idx_t> csr(cell_to_cell.numPoints(0), cell_to_cell.numCells(),
                      cell_to_cell.cell_to_node[0]);
  assert(cell_to_cell.numCells() < std::numeric_limits<idx_t>::max());
  assert(cell_to_cell.numPoints(0) < std::numeric_limits<idx_t>::max());
  assert(cell_to_cell.cell_to_node[0].getDim() == 2);

  idx_t nvtxs = cell_to_cell.numPoints(0), ncon = 1;
  idx_t nparts = std::ceil(static_cast<float>(cell_to_cell.numPoints(0)) /
                           coarse_block_size);
  std::vector<idx_t> coarse_partition(cell_to_cell.numPoints(0));
  idx_t objval;
  real_t ubvec[1] = {tolerance};
  int r = METIS_PartGraphKway(
      &nvtxs, &ncon, csr.pointIndices(), csr.cellEndpoints(), NULL, NULL, NULL,
      &nparts, NULL, ubvec, metis_options, &objval, coarse_partition.data());
  if (r != METIS_OK) {
    throw MetisError{r};
  }

  return HeuristicalPartition(csr, coarse_partition, block_size).partition();
}

std::vector<MY_SIZE> HeuristicalPartition::partition() {
  reorderCSRToPartition();
  result.resize(cell_to_cell.num_points, UNDEFINED);
  MY_SIZE coarse_block_from = 0;
  MY_SIZE block_id = 0;
  for (MY_SIZE coarse_block_to = 1; coarse_block_to < coarse_partition.size();
       ++coarse_block_to) {
    if (coarse_partition[coarse_block_to] !=
        coarse_partition[coarse_block_to - 1]) {
      partitionWithin(coarse_block_from, coarse_block_to, block_id);
      coarse_block_from = coarse_block_to;
    }
  }
  partitionWithin(coarse_block_from, coarse_partition.size(), block_id);
  reorderData<false>(result, 1, inverse_permutation);
  return std::move(result);
}

void HeuristicalPartition::partitionWithin(MY_SIZE from, MY_SIZE to,
                                           MY_SIZE &block_id) {
  const MY_SIZE coarse_block_size = to - from;
  // This is the maximum number of blocks we can make out of this coarse block,
  // since we can't go above `block_size`. This number also maximises the block
  // size.
  const MY_SIZE num_blocks_in_coarse_block =
      std::ceil(static_cast<double>(coarse_block_size) / block_size);
  // The actual sizes of the blocks, chosen to minimise the difference between
  // block sizes.
  // TODO: do we want minimal variance (in the block sizes) or as many and as
  // large blocks as possible?
  const MY_SIZE block_size_small = std::floor(
      static_cast<double>(coarse_block_size) / num_blocks_in_coarse_block);
  const MY_SIZE block_size_large = std::ceil(
      static_cast<double>(coarse_block_size) / num_blocks_in_coarse_block);
  const MY_SIZE num_large_blocks =
      coarse_block_size % num_blocks_in_coarse_block;
  MY_SIZE starting_point = from;
  for (MY_SIZE block = 0; block < num_blocks_in_coarse_block; ++block) {
    while (result[starting_point] != UNDEFINED) {
      ++starting_point;
    }
    assert(starting_point < to);
    growBlock(starting_point, to,
              block < num_large_blocks ? block_size_large : block_size_small,
              block_id++);
  }
}

void HeuristicalPartition::growBlock(MY_SIZE starting_point, MY_SIZE to,
                                     MY_SIZE size, MY_SIZE block_id) {
  priority_t::used_colours.reset();
  PriorityQueue<priority_t> queue = initPriorityQueue(starting_point, to);
#ifndef NDEBUG
  std::map<MY_SIZE, MY_SIZE> assigned_colours;
#endif // NDEBUG
  for (MY_SIZE i = 0; i < size; ++i) {
    auto max_pair = queue.popMax();
    MY_SIZE cur = max_pair.first;
    assert(result[cur] == UNDEFINED);
    const priority_t &cur_priority = max_pair.second;
    result[cur] = block_id;

    // Assign colour
    colourset_t available_colours =
        ~cur_priority.occupied_colours & priority_t::used_colours;
    if (available_colours.none()) {
      priority_t::used_colours <<= 1;
      priority_t::used_colours.set(0);
      available_colours =
          ~cur_priority.occupied_colours & priority_t::used_colours;
      assert(available_colours.any());
    }
    MY_SIZE colour = Mesh::getAvailableColour<false>(available_colours, {});
    colourset_t assigned_colourset{};
    assigned_colourset.set(colour);

#ifndef NDEBUG
    assigned_colours[cur] = colour;
#endif // NDEBUG

    // Update neighbours
    const MY_SIZE neighbours_start = cell_to_cell.pointIndices()[cur];
    const MY_SIZE neighbours_end = cell_to_cell.pointIndices()[cur + 1];
    for (MY_SIZE j = neighbours_start; j < neighbours_end; ++j) {
      const MY_SIZE neighbour = cell_to_cell.cellEndpoints()[j];
      if (result[neighbour] != UNDEFINED) {
        // Already assigned to a block
        continue;
      }
      auto old_priority_ = queue.getPriority(neighbour);
      if (!old_priority_.first) {
        // In another coarse block not yet visited
        continue;
      }
      priority_t &old_priority = old_priority_.second;
      old_priority.added = i + 1;
      ++old_priority.reuse;
      old_priority.occupied_colours |= assigned_colourset;
      queue.modify(neighbour, old_priority);
    }
  }
  assert(checkColouringWithinBlock(assigned_colours, block_id));
}

void HeuristicalPartition::reorderCSRToPartition() {
  assert(coarse_partition.size() == cell_to_cell.num_points);
  inverse_permutation.resize(coarse_partition.size());
  std::iota(inverse_permutation.begin(), inverse_permutation.end(), 0);
  std::stable_sort(inverse_permutation.begin(), inverse_permutation.end(),
                   [&](MY_SIZE a, MY_SIZE b) {
                     return coarse_partition[a] < coarse_partition[b];
                   });
  reorderDataInverseAOS<idx_t, MY_SIZE>(coarse_partition.begin(),
                                        inverse_permutation, 1);
  cell_to_cell.reorderInverse(inverse_permutation);
}

PriorityQueue<HeuristicalPartition::priority_t>
HeuristicalPartition::initPriorityQueue(MY_SIZE from, MY_SIZE to) const {
  PriorityQueue<priority_t> pq{};
  for (MY_SIZE i = from; i < to; ++i) {
    if (result[i] == UNDEFINED) {
      pq.push(i, priority_t{});
    }
  }
  return pq;
}

bool HeuristicalPartition::checkColouringWithinBlock(
    const std::map<MY_SIZE, MY_SIZE> &assigned_colours,
    MY_SIZE block_id) const {
  for (auto p : assigned_colours) {
    MY_SIZE cur = p.first;
    MY_SIZE colour = p.second;
    const MY_SIZE start = cell_to_cell.pointIndices()[cur];
    const MY_SIZE end = cell_to_cell.pointIndices()[cur + 1];
    for (MY_SIZE i = start; i < end; ++i) {
      const MY_SIZE n = cell_to_cell.cellEndpoints()[i];
      if (result[n] != block_id) {
        continue;
      }
      if (colour == assigned_colours.at(n)) {
        return false;
      }
    }
  }
  return true;
}

// -----------------------------------------------------------------------
// -                          Utility functions                          -
// -----------------------------------------------------------------------

adj_list_t getCellToCell(const std::vector<MY_SIZE> &mapping,
                         MY_SIZE mapping_dim) {
  const std::multimap<MY_SIZE, MY_SIZE> point_to_cell =
      GraphCSR<MY_SIZE>::getPointToCell(mapping.begin(), mapping.end(),
                                        mapping_dim);
  const MY_SIZE num_cells = mapping.size() / mapping_dim;
  adj_list_t cell_to_cell(num_cells);
  for (MY_SIZE i = 0; i < num_cells; ++i) {
    for (MY_SIZE offset = 0; offset < mapping_dim; ++offset) {
      const MY_SIZE point = mapping[mapping_dim * i + offset];
      const auto cell_range = point_to_cell.equal_range(point);
      for (auto it = cell_range.first; it != cell_range.second; ++it) {
        const MY_SIZE other_cell = it->second;
        if (other_cell != i) {
          cell_to_cell[i].push_back(other_cell);
        }
      }
    }
  }
  return cell_to_cell;
}

/* vim:set et sts=2 sw=2 ts=2: */
