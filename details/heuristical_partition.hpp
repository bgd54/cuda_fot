#ifndef HEURISTICAL_PARTITION_HPP_ZZJ3VATR
#define HEURISTICAL_PARTITION_HPP_ZZJ3VATR

#include "partition.hpp"
#include "priority_queue.hpp"

namespace details {
namespace partitioning {

class HeuristicalPartition {
  using colourset_t = Mesh::colourset_t;
  GraphCSR<idx_t> &cell_to_cell;
  std::vector<idx_t> &coarse_partition;
  const MY_SIZE block_size;

public:
  HeuristicalPartition(GraphCSR<idx_t> &cell_to_cell_,
                       std::vector<idx_t> &coarse_partition_,
                       const MY_SIZE block_size_)
      : cell_to_cell{cell_to_cell_}, coarse_partition{coarse_partition_},
        block_size{block_size_} {}

  std::vector<MY_SIZE> partition();

private:
  struct priority_t {
    colourset_t occupied_colours;
    MY_SIZE reuse;
    MY_SIZE added;

    static colourset_t used_colours;
    priority_t() : occupied_colours{}, reuse{0}, added{0} {}

    bool operator<(const priority_t &rhs) const {
      int thread_col_delta = (~occupied_colours & used_colours).any();
      int rhs_thread_col_delta = (~rhs.occupied_colours & used_colours).any();
      if (reuse == 0 || rhs.reuse == 0) {
        return reuse < rhs.reuse;
      }
      if (thread_col_delta != rhs_thread_col_delta) {
        return thread_col_delta < rhs_thread_col_delta;
      }
      if (reuse != rhs.reuse) {
        return reuse < rhs.reuse;
      }
      return added < rhs.added;
    }
  };

  static constexpr MY_SIZE UNDEFINED = std::numeric_limits<MY_SIZE>::max();

  std::vector<MY_SIZE> result;
  // Holds the permutation performed during partitioning
  std::vector<MY_SIZE> inverse_permutation;

  void reorderCSRToPartition();
  void partitionWithin(MY_SIZE from, MY_SIZE to, MY_SIZE &block_id);
  PriorityQueue<priority_t> initPriorityQueue(MY_SIZE from, MY_SIZE to) const;
  void growBlock(MY_SIZE starting_point, MY_SIZE to, MY_SIZE block_size,
                 MY_SIZE block_id);
  bool
  checkColouringWithinBlock(const std::map<MY_SIZE, MY_SIZE> &assigned_colours,
                            MY_SIZE block_id) const;
};

} // namespace partitioning
} // namespace details

#endif /* end of include guard: HEURISTICAL_PARTITION_HPP_ZZJ3VATR */
