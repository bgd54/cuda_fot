#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <set>
#include <vector>

#include "colouring.hpp"
#include "helper_cuda.h"
#include "problem.hpp"
#include "tests.hpp"

/* copyKernels {{{1 */
__global__ void copyKernel(const float *__restrict__ a, float *__restrict__ b,
                           MY_SIZE size) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  const float4 *__restrict__ a_ = reinterpret_cast<const float4 *>(a);
  float4 *__restrict__ b_ = reinterpret_cast<float4 *>(b);
  if (tid * 4 < size) {
    b_[tid] = a_[tid];
  }
}

__global__ void copyKernel(const double *__restrict__ a, double *__restrict__ b,
                           MY_SIZE size) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  const double2 *__restrict__ a_ = reinterpret_cast<const double2 *>(a);
  double2 *__restrict__ b_ = reinterpret_cast<double2 *>(b);
  if (tid * 2 < size) {
    b_[tid] = a_[tid];
  }
}
/* 1}}} */

/* problem_stepGPU {{{1 */
template <unsigned PointDim = 1, unsigned EdgeDim = 1, bool SOA = false,
          typename DataType>
__global__ void
problem_stepGPU(const DataType *__restrict__ point_weights,
                const DataType *__restrict__ edge_weights,
                const MY_SIZE *__restrict__ edge_list,
                DataType *__restrict__ out, const MY_SIZE edge_num_in_partition,
                const MY_SIZE num_points, const MY_SIZE num_edges) {
  static_assert(
      EdgeDim == PointDim || EdgeDim == 1,
      "I know of no reason why EdgeDim should be anything but 1 or PointDim");

  MY_SIZE id = blockIdx.x * blockDim.x + threadIdx.x;
  DataType inc[2 * PointDim];
  if (id < edge_num_in_partition) {
    MY_SIZE edge_list_left = edge_list[2 * id];
    MY_SIZE edge_list_right = edge_list[2 * id + 1];
    #pragma unroll
    for (MY_SIZE d = 0; d < PointDim; ++d) {
      MY_SIZE ind_left = index<PointDim, SOA>(num_points, edge_list_left, d);
      MY_SIZE ind_right = index<PointDim, SOA>(num_points, edge_list_right, d);
      MY_SIZE edge_d = EdgeDim == 1 ? 0 : d;
      MY_SIZE edge_ind =
          index<EdgeDim, true>(edge_num_in_partition, id, edge_d);
      inc[d] =
          out[ind_right] + edge_weights[edge_ind] * point_weights[ind_left];
      inc[d + PointDim] =
          out[ind_left] + edge_weights[edge_ind] * point_weights[ind_right];
    }
    #pragma unroll
    for (MY_SIZE d = 0; d < PointDim; ++d) {
      MY_SIZE ind_left = index<PointDim, SOA>(num_points, edge_list_left, d);
      MY_SIZE ind_right = index<PointDim, SOA>(num_points, edge_list_right, d);

      out[ind_right] = inc[d];
      out[ind_left] = inc[d + PointDim];
    }
  }
}
/* 1}}} */

/* problem_stepGPUHierarchical {{{1 */
template <unsigned PointDim = 1, unsigned EdgeDim = 1, bool SOA = false,
          bool PerDataCache = false, bool SOAInShared = true,
          typename DataType = float>
__global__ void problem_stepGPUHierarchical(
    const MY_SIZE *__restrict__ edge_list,
    const DataType *__restrict__ point_weights,
    DataType *__restrict__ point_weights_out,
    const DataType *__restrict__ edge_weights,
    const MY_SIZE *__restrict__ points_to_be_cached,
    const MY_SIZE *__restrict__ points_to_be_cached_offsets,
    const std::uint8_t *__restrict__ edge_colours,
    const std::uint8_t *__restrict__ num_edge_colours,
    const MY_SIZE *__restrict__ block_offsets, const MY_SIZE num_threads,
    const MY_SIZE num_points) {
  static_assert(
      EdgeDim == PointDim || EdgeDim == 1,
      "I know of no reason why EdgeDim should be anything but 1 or PointDim");

  const float4 *__restrict__ point_weights2 =
      reinterpret_cast<const float4 *>(point_weights);
  float4 *__restrict__ point_weights_out2 =
      reinterpret_cast<float4 *>(point_weights_out);
  const double2 *__restrict__ point_weights3 =
      reinterpret_cast<const double2 *>(point_weights);
  double2 *__restrict__ point_weights_out3 =
      reinterpret_cast<double2 *>(point_weights_out);

  static_assert(SOA || PerDataCache,
                "AOS and not per data cache is currently not supported");
  MY_SIZE bid = blockIdx.x;
  MY_SIZE thread_ind = block_offsets[bid] + threadIdx.x;
  MY_SIZE tid = threadIdx.x;

  MY_SIZE cache_points_offset = points_to_be_cached_offsets[bid];
  MY_SIZE num_cached_points =
      points_to_be_cached_offsets[bid + 1] - cache_points_offset;
  MY_SIZE shared_num_cached_points;
  if (SOAInShared) {
    static_assert(32 % PointDim == 0, "Currently, shared memory bank conflict "
                                      "avoidance works only if 32 is divisible "
                                      "by PointDim");
    MY_SIZE needed_offset = 32 / PointDim;
    if (num_cached_points % 32 <= needed_offset) {
      shared_num_cached_points =
          num_cached_points - (num_cached_points % 32) + needed_offset;
    } else {
      shared_num_cached_points =
          num_cached_points - (num_cached_points % 32) + 32 + needed_offset;
    }
    assert(shared_num_cached_points >= num_cached_points);
  } else {
    shared_num_cached_points = num_cached_points;
  }

  extern __shared__ __align__(alignof(DataType)) unsigned char shared[];
  DataType *point_cache = reinterpret_cast<DataType *>(shared);

  MY_SIZE block_size = block_offsets[bid + 1] - block_offsets[bid];

  std::uint8_t our_colour;
  if (tid >= block_size) {
    our_colour = num_edge_colours[bid];
  } else {
    our_colour = edge_colours[thread_ind];
  }

  // Cache in
  if (PerDataCache) {
    if (!SOA && PointDim % 4 == 0 && std::is_same<DataType, float>::value) {
      for (MY_SIZE i = tid; i < PointDim * num_cached_points / 4;
           i += blockDim.x) {
        MY_SIZE point_ind = i * 4 / PointDim;
        MY_SIZE d = (i * 4) % PointDim;
        MY_SIZE g_ind = index<PointDim, SOA>(
            num_points, points_to_be_cached[cache_points_offset + point_ind],
            d);
        MY_SIZE c_ind0 = index<PointDim, SOAInShared>(shared_num_cached_points,
                                                      point_ind, d + 0);
        MY_SIZE c_ind1 = index<PointDim, SOAInShared>(shared_num_cached_points,
                                                      point_ind, d + 1);
        MY_SIZE c_ind2 = index<PointDim, SOAInShared>(shared_num_cached_points,
                                                      point_ind, d + 2);
        MY_SIZE c_ind3 = index<PointDim, SOAInShared>(shared_num_cached_points,
                                                      point_ind, d + 3);
        float4 tmp = point_weights2[g_ind / 4];
        point_cache[c_ind0] = tmp.x;
        point_cache[c_ind1] = tmp.y;
        point_cache[c_ind2] = tmp.z;
        point_cache[c_ind3] = tmp.w;
      }
    } else if (!SOA && PointDim % 2 == 0 &&
               std::is_same<DataType, double>::value) {
      for (MY_SIZE i = tid; i < PointDim * num_cached_points / 2;
           i += blockDim.x) {
        MY_SIZE point_ind = i * 2 / PointDim;
        MY_SIZE d = (i * 2) % PointDim;
        MY_SIZE g_ind = index<PointDim, SOA>(
            num_points, points_to_be_cached[cache_points_offset + point_ind],
            d);
        MY_SIZE c_ind0 = index<PointDim, SOAInShared>(shared_num_cached_points,
                                                      point_ind, d + 0);
        MY_SIZE c_ind1 = index<PointDim, SOAInShared>(shared_num_cached_points,
                                                      point_ind, d + 1);
        double2 tmp = point_weights3[g_ind / 2];
        point_cache[c_ind0] = tmp.x;
        point_cache[c_ind1] = tmp.y;
      }
    } else {
      for (MY_SIZE i = tid; i < PointDim * num_cached_points; i += blockDim.x) {
        MY_SIZE point_ind = i / PointDim;
        MY_SIZE d = i % PointDim;
        MY_SIZE g_ind = index<PointDim, SOA>(
            num_points, points_to_be_cached[cache_points_offset + point_ind],
            d);
        MY_SIZE c_ind = index<PointDim, SOAInShared>(shared_num_cached_points,
                                                     point_ind, d);
        point_cache[c_ind] = point_weights[g_ind];
      }
    }
  } else {
    for (MY_SIZE i = tid; i < num_cached_points; i += blockDim.x) {
      MY_SIZE g_point_to_be_cached =
          points_to_be_cached[cache_points_offset + i];
      for (MY_SIZE d = 0; d < PointDim; ++d) {
        MY_SIZE c_ind, g_ind;
        g_ind = index<PointDim, SOA>(num_points, g_point_to_be_cached, d);
        c_ind = index<PointDim, SOAInShared>(shared_num_cached_points, i, d);

        point_cache[c_ind] = point_weights[g_ind];
      }
    }
  }

  __syncthreads();

  // Computation
  DataType increment[PointDim * 2];
  MY_SIZE edge_list_left;
  MY_SIZE edge_list_right;
  if (tid < block_size) {
    edge_list_left = edge_list[index<2, true>(num_threads, thread_ind, 0)];
    edge_list_right = edge_list[index<2, true>(num_threads, thread_ind, 1)];
    for (MY_SIZE d = 0; d < PointDim; ++d) {
      MY_SIZE left_ind = index<PointDim, SOAInShared>(shared_num_cached_points,
                                                      edge_list_left, d);
      MY_SIZE right_ind = index<PointDim, SOAInShared>(shared_num_cached_points,
                                                       edge_list_right, d);
      MY_SIZE edge_d = EdgeDim == 1 ? 0 : d;
      MY_SIZE edge_ind = index<EdgeDim, true>(num_threads, thread_ind, edge_d);

      increment[d] = point_cache[left_ind] * edge_weights[edge_ind];
      increment[d + PointDim] = point_cache[right_ind] * edge_weights[edge_ind];
    }
  }

  __syncthreads();

  // Clear cache
  for (MY_SIZE i = tid; i < shared_num_cached_points * PointDim;
       i += blockDim.x) {
    point_cache[i] = 0;
  }

  __syncthreads();

  // Accumulate increment
  for (MY_SIZE i = 0; i < num_edge_colours[bid]; ++i) {
    if (our_colour == i) {
      for (MY_SIZE d = 0; d < PointDim; ++d) {
        MY_SIZE left_ind = index<PointDim, SOAInShared>(
            shared_num_cached_points, edge_list_left, d);
        MY_SIZE right_ind = index<PointDim, SOAInShared>(
            shared_num_cached_points, edge_list_right, d);

        point_cache[right_ind] += increment[d];
        point_cache[left_ind] += increment[d + PointDim];
      }
    }
    __syncthreads();
  }

  // Cache out
  if (PerDataCache) {
    if (!SOA && PointDim % 4 == 0 && std::is_same<DataType, float>::value) {
      for (MY_SIZE i = tid; i < PointDim * num_cached_points / 4;
           i += blockDim.x) {
        MY_SIZE point_ind = i * 4 / PointDim;
        MY_SIZE d = (i * 4) % PointDim;
        MY_SIZE g_ind = index<PointDim, SOA>(
            num_points, points_to_be_cached[cache_points_offset + point_ind],
            d);
        MY_SIZE c_ind0 = index<PointDim, SOAInShared>(shared_num_cached_points,
                                                      point_ind, d + 0);
        MY_SIZE c_ind1 = index<PointDim, SOAInShared>(shared_num_cached_points,
                                                      point_ind, d + 1);
        MY_SIZE c_ind2 = index<PointDim, SOAInShared>(shared_num_cached_points,
                                                      point_ind, d + 2);
        MY_SIZE c_ind3 = index<PointDim, SOAInShared>(shared_num_cached_points,
                                                      point_ind, d + 3);
        float4 result = point_weights_out2[g_ind / 4];
        result.x += point_cache[c_ind0];
        result.y += point_cache[c_ind1];
        result.z += point_cache[c_ind2];
        result.w += point_cache[c_ind3];
        point_weights_out2[g_ind / 4] = result;
      }
    } else if (!SOA && PointDim % 2 == 0 &&
               std::is_same<DataType, double>::value) {
      for (MY_SIZE i = tid; i < PointDim * num_cached_points / 2;
           i += blockDim.x) {
        MY_SIZE point_ind = i * 2 / PointDim;
        MY_SIZE d = (i * 2) % PointDim;
        MY_SIZE g_ind = index<PointDim, SOA>(
            num_points, points_to_be_cached[cache_points_offset + point_ind],
            d);
        MY_SIZE c_ind0 = index<PointDim, SOAInShared>(shared_num_cached_points,
                                                      point_ind, d + 0);
        MY_SIZE c_ind1 = index<PointDim, SOAInShared>(shared_num_cached_points,
                                                      point_ind, d + 1);
        double2 result = point_weights_out3[g_ind / 2];
        result.x += point_cache[c_ind0];
        result.y += point_cache[c_ind1];
        point_weights_out3[g_ind / 2] = result;
      }
    } else {
      for (MY_SIZE i = tid; i < num_cached_points * PointDim; i += blockDim.x) {
        MY_SIZE point_ind = i / PointDim;
        MY_SIZE d = i % PointDim;
        MY_SIZE g_ind = index<PointDim, SOA>(
            num_points, points_to_be_cached[cache_points_offset + point_ind],
            d);
        MY_SIZE c_ind = index<PointDim, SOAInShared>(shared_num_cached_points,
                                                     point_ind, d);
        DataType result = point_weights_out[g_ind] + point_cache[c_ind];
        point_weights_out[g_ind] = result;
      }
    }
  } else {
    for (MY_SIZE i = tid; i < num_cached_points; i += blockDim.x) {
      MY_SIZE g_point_to_be_cached =
          points_to_be_cached[cache_points_offset + i];
      DataType result[PointDim];
      #pragma unroll
      for (MY_SIZE d = 0; d < PointDim; ++d) {
        MY_SIZE write_g_ind =
            index<PointDim, SOA>(num_points, g_point_to_be_cached, d);
        MY_SIZE write_c_ind =
            index<PointDim, SOAInShared>(shared_num_cached_points, i, d);

        result[d] = point_weights_out[write_g_ind] + point_cache[write_c_ind];
      }
      #pragma unroll
      for (MY_SIZE d = 0; d < PointDim; ++d) {
        MY_SIZE write_g_ind =
            index<PointDim, SOA>(num_points, g_point_to_be_cached, d);
        point_weights_out[write_g_ind] = result[d];
      }
    }
  }
}
/* 1}}} */

template <unsigned Dim = 1, bool SOA = false, typename DataType = float,
          class ForwardIterator>
size_t countCacheLinesForBlock(ForwardIterator block_begin,
                               ForwardIterator block_end) {
  std::set<MY_SIZE> cache_lines;
  MY_SIZE data_per_cacheline = 32 / sizeof(DataType);

  for (; block_begin != block_end; ++block_begin) {
    MY_SIZE point_id = *block_begin;
    MY_SIZE cache_line_id = SOA ? point_id / data_per_cacheline
                                : point_id * Dim / data_per_cacheline;
    if (!SOA) {
      if (data_per_cacheline / Dim > 0) {
        assert(data_per_cacheline % Dim == 0);
        cache_lines.insert(cache_line_id);
      } else {
        assert(Dim % data_per_cacheline == 0);
        MY_SIZE cache_line_per_data =
            Dim / data_per_cacheline; // Assume that Dim is multiple of
                                      // data_per_cacheline
        for (MY_SIZE i = 0; i < cache_line_per_data; ++i) {
          cache_lines.insert(cache_line_id++);
        }
      }
    } else {
      cache_lines.insert(cache_line_id);
    }
  }
  return (SOA ? Dim : 1) * cache_lines.size();
}

/* loopGPUEdgeCentred {{{1 */
template <unsigned PointDim, unsigned EdgeDim, bool SOA, typename DataType>
void Problem<PointDim, EdgeDim, SOA, DataType>::loopGPUEdgeCentred(
    MY_SIZE num, MY_SIZE reset_every) {
  std::vector<std::vector<MY_SIZE>> partition = graph.colourEdges();
  MY_SIZE num_of_colours = partition.size();
  assert(num_of_colours > 0);
  data_t<DataType, PointDim> point_weights2(point_weights.getSize());
  std::copy(point_weights.begin(), point_weights.end(), point_weights2.begin());
  std::vector<data_t<MY_SIZE, 2>> d_edge_lists;
  std::vector<data_t<DataType, EdgeDim>> d_edge_weights;
  MY_SIZE total_num_cache_lines = 0;
  MY_SIZE total_num_blocks = 0;
  for (const std::vector<MY_SIZE> &colour : partition) {
    d_edge_lists.emplace_back(colour.size());
    d_edge_weights.emplace_back(colour.size());
    for (std::size_t i = 0; i < colour.size(); ++i) {
      d_edge_lists.back()[2 * i] = graph.edge_to_node[2 * colour[i]];
      d_edge_lists.back()[2 * i + 1] = graph.edge_to_node[2 * colour[i] + 1];
      for (unsigned d = 0; d < EdgeDim; ++d) {
        d_edge_weights.back()[index<EdgeDim, true>(colour.size(), i, d)] =
            edge_weights[index<EdgeDim, true>(graph.numEdges(), colour[i], d)];
      }
    }
    d_edge_lists.back().initDeviceMemory();
    d_edge_weights.back().initDeviceMemory();
    MY_SIZE num_blocks = std::ceil(static_cast<double>(colour.size()) /
                                   static_cast<double>(block_size));
    total_num_blocks += num_blocks;
    for (MY_SIZE i = 0; i < num_blocks; ++i) {
      total_num_cache_lines += countCacheLinesForBlock<PointDim, SOA, DataType>(
          d_edge_lists.back().begin() + 2 * block_size * i,
          d_edge_lists.back().begin() +
              2 * std::min<MY_SIZE>(colour.size(), block_size * (i + 1)));
    }
  }
  point_weights.initDeviceMemory();
  point_weights2.initDeviceMemory();
  CUDA_TIMER_START(t);
  for (MY_SIZE i = 0; i < num; ++i) {
    for (MY_SIZE c = 0; c < num_of_colours; ++c) {
      MY_SIZE num_blocks = std::ceil(static_cast<double>(partition[c].size()) /
                                     static_cast<double>(block_size));
      problem_stepGPU<PointDim, EdgeDim, SOA><<<num_blocks, block_size>>>(
          point_weights.getDeviceData(), d_edge_weights[c].getDeviceData(),
          d_edge_lists[c].getDeviceData(), point_weights2.getDeviceData(),
          partition[c].size(), graph.numPoints(), graph.numEdges());
      checkCudaErrors(cudaDeviceSynchronize());
    }
    TIMER_TOGGLE(t);
    if (reset_every && i % reset_every == reset_every - 1) {
      reset();
      // Copy to point_weights2 that is currently holding the result, the next
      // copy will put it into point_weights also.
      std::copy(point_weights.begin(), point_weights.end(),
                point_weights2.begin());
      point_weights2.flushToDevice();
    }
    checkCudaErrors(cudaMemcpy(point_weights.getDeviceData(),
                               point_weights2.getDeviceData(),
                               sizeof(DataType) * graph.numPoints() * PointDim,
                               cudaMemcpyDeviceToDevice));
    TIMER_TOGGLE(t);
  }
  PRINT_BANDWIDTH(
      t, "loopGPUEdgeCentred",
      (sizeof(DataType) *
           (2.0 * PointDim * graph.numPoints() + EdgeDim * graph.numEdges()) +
       2.0 * sizeof(MY_SIZE) * graph.numEdges()) *
          num,
      (sizeof(DataType) * graph.numPoints() * PointDim * 2.0 + // point_weights
       sizeof(DataType) * graph.numEdges() * EdgeDim * 1.0 +   // d_edge_weights
       sizeof(MY_SIZE) * graph.numEdges() * 2.0 +              // d_edge_list
       sizeof(MY_SIZE) * graph.numEdges() * 2.0                // d_partition
       ) * num);
  std::cout << " Needed " << num_of_colours << " colours" << std::endl;
  std::cout << "  average cache_line / block: "
            << static_cast<double>(total_num_cache_lines) / total_num_blocks
            << std::endl;
  PRINT_BANDWIDTH(t, "-cache line", total_num_cache_lines * 32.0,
                  total_num_cache_lines * 32.0);
  point_weights.flushToHost();
}
/* 1}}} */

/* loopGPUHierarchical {{{1 */
template <unsigned PointDim, unsigned EdgeDim, bool SOA, typename DataType>
void Problem<PointDim, EdgeDim, SOA, DataType>::loopGPUHierarchical(
    MY_SIZE num, MY_SIZE reset_every) {
  TIMER_START(t_colouring);
  HierarchicalColourMemory<PointDim, EdgeDim, SOA, DataType> memory(
      *this, partition_vector);
  TIMER_PRINT(t_colouring, "Hierarchical colouring: colouring");
  const auto d_memory = memory.getDeviceMemoryOfOneColour();
  data_t<DataType, PointDim> point_weights_out(point_weights.getSize());
  std::copy(point_weights.begin(), point_weights.end(),
            point_weights_out.begin());
  point_weights.initDeviceMemory();
  point_weights_out.initDeviceMemory();
  MY_SIZE total_cache_size = 0; // for bandwidth calculations
  DataType avg_num_edge_colours = 0;
  MY_SIZE total_num_blocks = 0;
  MY_SIZE total_shared_size = 0;
  size_t total_num_cache_lines = 0;
  for (MY_SIZE i = 0; i < memory.colours.size(); ++i) {
    const typename HierarchicalColourMemory<PointDim, EdgeDim, SOA,
                                            DataType>::MemoryOfOneColour
        &memory_of_one_colour = memory.colours[i];
    MY_SIZE num_threads = memory_of_one_colour.edge_list.size() / 2;
    MY_SIZE num_blocks = static_cast<MY_SIZE>(
        std::ceil(static_cast<double>(num_threads) / block_size));
    total_cache_size += memory_of_one_colour.points_to_be_cached.size();
    avg_num_edge_colours +=
        std::accumulate(memory_of_one_colour.num_edge_colours.begin(),
                        memory_of_one_colour.num_edge_colours.end(), 0.0f);
    total_num_blocks += num_blocks;
    total_shared_size += num_blocks * d_memory[i].shared_size;
    for (MY_SIZE j = 0;
         j < memory_of_one_colour.points_to_be_cached_offsets.size() - 1; ++j) {
      total_num_cache_lines +=
          countCacheLinesForBlock<PointDim, SOA, DataType,
                                  std::vector<MY_SIZE>::const_iterator>(
              memory_of_one_colour.points_to_be_cached.begin() +
                  memory_of_one_colour.points_to_be_cached_offsets[j],
              memory_of_one_colour.points_to_be_cached.begin() +
                  memory_of_one_colour.points_to_be_cached_offsets[j + 1]);
    }
  }
  // -----------------------
  // -  Start computation  -
  // -----------------------
  CUDA_TIMER_START(timer_calc);
  TIMER_TOGGLE(timer_calc);
  CUDA_TIMER_START(timer_copy);
  TIMER_TOGGLE(timer_copy);
  for (MY_SIZE iteration = 0; iteration < num; ++iteration) {
    for (MY_SIZE colour_ind = 0; colour_ind < memory.colours.size();
         ++colour_ind) {
      assert(memory.colours[colour_ind].edge_list.size() % 2 == 0);
      MY_SIZE num_threads = memory.colours[colour_ind].edge_list.size() / 2;
      MY_SIZE num_blocks = memory.colours[colour_ind].num_edge_colours.size();
      assert(num_blocks == memory.colours[colour_ind].block_offsets.size() - 1);
      // + 32 in case it needs to avoid shared mem bank collisions
      MY_SIZE cache_size =
          sizeof(DataType) * (d_memory[colour_ind].shared_size + 32) * PointDim;
      TIMER_TOGGLE(timer_calc);
      problem_stepGPUHierarchical<PointDim, EdgeDim, SOA, !SOA, true, DataType>
          <<<num_blocks, block_size, cache_size>>>(
              static_cast<MY_SIZE *>(d_memory[colour_ind].edge_list),
              point_weights.getDeviceData(), point_weights_out.getDeviceData(),
              static_cast<DataType *>(d_memory[colour_ind].edge_weights),
              static_cast<MY_SIZE *>(d_memory[colour_ind].points_to_be_cached),
              static_cast<MY_SIZE *>(
                  d_memory[colour_ind].points_to_be_cached_offsets),
              static_cast<std::uint8_t *>(d_memory[colour_ind].edge_colours),
              static_cast<std::uint8_t *>(
                  d_memory[colour_ind].num_edge_colours),
              static_cast<MY_SIZE *>(d_memory[colour_ind].block_offsets),
              num_threads, graph.numPoints());
      TIMER_TOGGLE(timer_calc);
      checkCudaErrors(cudaDeviceSynchronize());
    }
    MY_SIZE copy_size = graph.numPoints() * PointDim;
    TIMER_TOGGLE(timer_copy);
    MY_SIZE num_copy_blocks = std::ceil(static_cast<float>(copy_size) / 512.0);
    copyKernel<<<num_copy_blocks, 512>>>(point_weights_out.getDeviceData(),
                                         point_weights.getDeviceData(),
                                         copy_size);
    TIMER_TOGGLE(timer_copy);
    if (reset_every && iteration % reset_every == reset_every - 1) {
      reset();
      point_weights.flushToDevice();
      std::copy(point_weights.begin(), point_weights.end(),
                point_weights_out.begin());
      point_weights_out.flushToDevice();
    }
  }
  PRINT_BANDWIDTH(
      timer_calc, "GPU HierarchicalColouring",
      num * ((2.0 * PointDim * graph.numPoints() + EdgeDim * graph.numEdges()) *
                 sizeof(DataType) +
             2.0 * graph.numEdges() * sizeof(MY_SIZE)),
      num *
          (sizeof(DataType) * graph.numPoints() * PointDim *
               2.0 + // point_weights
           sizeof(DataType) * graph.numEdges() * EdgeDim * 1.0 + // edge_weights
           sizeof(MY_SIZE) * graph.numEdges() * 2.0 +            // edge_list
           sizeof(MY_SIZE) * total_cache_size * 1.0 +
           sizeof(MY_SIZE) *
               (total_num_blocks * 1.0 +
                memory.colours.size()) + // points_to_be_cached_offsets
           sizeof(MY_SIZE) * (total_num_blocks * 1.0) + // block_offsets
           sizeof(std::uint8_t) * graph.numEdges()      // edge_colours
           ));
  PRINT_BANDWIDTH(timer_copy, " -copy",
                  2.0 * num * sizeof(DataType) * PointDim * graph.numPoints(),
                  2.0 * num * sizeof(DataType) * PointDim * graph.numPoints());
  std::cout << "  reuse factor: "
            << static_cast<double>(total_cache_size) / (2 * graph.numEdges())
            << std::endl;
  std::cout << "  cache/shared mem: "
            << static_cast<double>(total_cache_size) / total_shared_size
            << "\n  shared mem reuse factor (total shared / (2 * #edges)): "
            << static_cast<double>(total_shared_size) / (2 * graph.numEdges())
            << std::endl;
  std::cout << "  average cache_line / block: "
            << static_cast<double>(total_num_cache_lines) / total_num_blocks
            << std::endl;
  PRINT_BANDWIDTH(timer_calc, "-cache line", total_num_cache_lines * 32.0,
                  total_num_cache_lines * 32.0);
  avg_num_edge_colours /= total_num_blocks;
  std::cout << "  average number of colours used: " << avg_num_edge_colours
            << std::endl;
  // ---------------
  // -  Finish up  -
  // ---------------
  point_weights.flushToHost();
}
/* 1}}} */

template <unsigned PointDim = 1, unsigned EdgeDim = 1, bool SOA = false,
          bool RunCPU = true, typename DataType = float>
void generateTimes(std::string in_file) {
  constexpr MY_SIZE num = 500;
  std::cout << ":::: Generating problems from file: " << in_file
            << "::::" << std::endl
            << "     Point dimension: " << PointDim
            << " Edge dimension: " << EdgeDim << " SOA: " << std::boolalpha
            << SOA << "\n     Data type: "
            << (sizeof(DataType) == sizeof(float) ? "float" : "double")
            << std::endl;
  std::function<void(
      implementation_algorithm_t<PointDim, EdgeDim, SOA, DataType>, MY_SIZE)>
      run = [&in_file](
          implementation_algorithm_t<PointDim, EdgeDim, SOA, DataType> algo,
          MY_SIZE num) {
        std::ifstream f(in_file);
        Problem<PointDim, EdgeDim, SOA, DataType> problem(f);
        std::cout << "--Problem created" << std::endl;
        (problem.*algo)(num, 0);
        std::cout << "--Problem finished." << std::endl;
      };
  run(&Problem<PointDim, EdgeDim, SOA, DataType>::loopCPUEdgeCentred,
      RunCPU ? num : 1);
  run(&Problem<PointDim, EdgeDim, SOA, DataType>::loopCPUEdgeCentredOMP,
      RunCPU ? num : 1);
  run(&Problem<PointDim, EdgeDim, SOA, DataType>::loopGPUEdgeCentred, num);
  run(&Problem<PointDim, EdgeDim, SOA, DataType>::loopGPUHierarchical, num);
  std::cout << "Finished." << std::endl;
}

template <unsigned PointDim = 1, unsigned EdgeDim = 1, bool SOA = false,
          typename DataType = float>
void generateTimesWithBlockDims(MY_SIZE N, MY_SIZE M,
                                std::pair<MY_SIZE, MY_SIZE> block_dims) {
  constexpr MY_SIZE num = 500;
  MY_SIZE block_size = block_dims.first == 0
                           ? block_dims.second
                           : block_dims.first * block_dims.second * 2;
  std::cout << ":::: Generating problems with block size: " << block_dims.first
            << "x" << block_dims.second << " (= " << block_size << ")"
            << "::::" << std::endl
            << "     Point dimension: " << PointDim
            << " Edge dimension: " << EdgeDim << " SOA: " << std::boolalpha
            << SOA << "\n     Data type: "
            << (sizeof(DataType) == sizeof(float) ? "float" : "double")
            << std::endl;
  std::function<void(
      implementation_algorithm_t<PointDim, EdgeDim, SOA, DataType>)>
      run = [&](
          implementation_algorithm_t<PointDim, EdgeDim, SOA, DataType> algo) {
        Problem<PointDim, EdgeDim, SOA, DataType> problem(N, M, block_dims);
        std::cout << "--Problem created" << std::endl;
        (problem.*algo)(num, 0);
        std::cout << "--Problem finished." << std::endl;
      };
  run(&Problem<PointDim, EdgeDim, SOA, DataType>::loopGPUEdgeCentred);
  run(&Problem<PointDim, EdgeDim, SOA, DataType>::loopGPUHierarchical);
  std::cout << "Finished." << std::endl;
}

template <unsigned PointDim = 1, unsigned EdgeDim = 1, bool SOA = false,
          typename DataType = float>
void generateTimesDifferentBlockDims(MY_SIZE N, MY_SIZE M) {
  generateTimesWithBlockDims<PointDim, EdgeDim, SOA, DataType>(N, M, {0, 32});
  generateTimesWithBlockDims<PointDim, EdgeDim, SOA, DataType>(N, M, {2, 8});
  generateTimesWithBlockDims<PointDim, EdgeDim, SOA, DataType>(N, M, {4, 4});
  generateTimesWithBlockDims<PointDim, EdgeDim, SOA, DataType>(N, M, {0, 128});
  generateTimesWithBlockDims<PointDim, EdgeDim, SOA, DataType>(N, M, {2, 32});
  generateTimesWithBlockDims<PointDim, EdgeDim, SOA, DataType>(N, M, {4, 16});
  generateTimesWithBlockDims<PointDim, EdgeDim, SOA, DataType>(N, M, {8, 8});
  generateTimesWithBlockDims<PointDim, EdgeDim, SOA, DataType>(N, M, {0, 288});
  generateTimesWithBlockDims<PointDim, EdgeDim, SOA, DataType>(N, M, {2, 72});
  generateTimesWithBlockDims<PointDim, EdgeDim, SOA, DataType>(N, M, {4, 36});
  generateTimesWithBlockDims<PointDim, EdgeDim, SOA, DataType>(N, M, {12, 12});
  generateTimesWithBlockDims<PointDim, EdgeDim, SOA, DataType>(N, M, {9, 8});
  generateTimesWithBlockDims<PointDim, EdgeDim, SOA, DataType>(N, M, {0, 512});
  generateTimesWithBlockDims<PointDim, EdgeDim, SOA, DataType>(N, M, {2, 128});
  generateTimesWithBlockDims<PointDim, EdgeDim, SOA, DataType>(N, M, {4, 64});
  generateTimesWithBlockDims<PointDim, EdgeDim, SOA, DataType>(N, M, {8, 32});
  generateTimesWithBlockDims<PointDim, EdgeDim, SOA, DataType>(N, M, {16, 16});
}

void generateTimesFromFile(int argc, const char **argv) {
  if (argc <= 1) {
    std::cerr << "Usage: " << argv[0] << " <input graph>" << std::endl;
    std::exit(1);
  }
  // AOS
  generateTimes<1, 1, false, false>(argv[1]);
  generateTimes<4, 1, false, false>(argv[1]);
  generateTimes<8, 1, false, false>(argv[1]);
  generateTimes<16, 1, false, false>(argv[1]);
  generateTimes<1, 1, false, false>(argv[1]);
  generateTimes<4, 4, false, false>(argv[1]);
  generateTimes<8, 8, false, false>(argv[1]);
  generateTimes<16, 16, false, false>(argv[1]);
  generateTimes<1, 1, false, false, double>(argv[1]);
  generateTimes<4, 1, false, false, double>(argv[1]);
  generateTimes<8, 1, false, false, double>(argv[1]);
  generateTimes<16, 1, false, false, double>(argv[1]);
  generateTimes<1, 1, false, false, double>(argv[1]);
  generateTimes<4, 4, false, false, double>(argv[1]);
  generateTimes<8, 8, false, false, double>(argv[1]);
  generateTimes<16, 16, false, false, double>(argv[1]);
  // SOA
  generateTimes<1, 1, true, false>(argv[1]);
  generateTimes<4, 1, true, false>(argv[1]);
  generateTimes<8, 1, true, false>(argv[1]);
  generateTimes<16, 1, true, false>(argv[1]);
  generateTimes<1, 1, true, false>(argv[1]);
  generateTimes<4, 4, true, false>(argv[1]);
  generateTimes<8, 8, true, false>(argv[1]);
  generateTimes<16, 16, true, false>(argv[1]);
  generateTimes<1, 1, true, false, double>(argv[1]);
  generateTimes<4, 1, true, false, double>(argv[1]);
  generateTimes<8, 1, true, false, double>(argv[1]);
  generateTimes<16, 1, true, false, double>(argv[1]);
  generateTimes<1, 1, true, false, double>(argv[1]);
  generateTimes<4, 4, true, false, double>(argv[1]);
  generateTimes<8, 8, true, false, double>(argv[1]);
  generateTimes<16, 16, true, false, double>(argv[1]);
}

void test() {
  MY_SIZE num = 500;
  MY_SIZE N = 100, M = 200;
  MY_SIZE reset_every = 0;
  constexpr unsigned TEST_DIM = 4;
  constexpr unsigned TEST_EDGE_DIM = 4;
  testTwoImplementations<TEST_DIM, TEST_EDGE_DIM, false, float>(
      num, N, M, reset_every,
      &Problem<TEST_DIM, TEST_EDGE_DIM, false, float>::loopCPUEdgeCentredOMP,
      &Problem<TEST_DIM, TEST_EDGE_DIM, false, float>::loopGPUHierarchical);
  testTwoImplementations<TEST_DIM, TEST_EDGE_DIM, true, float>(
      num, N, M, reset_every,
      &Problem<TEST_DIM, TEST_EDGE_DIM, true, float>::loopCPUEdgeCentredOMP,
      &Problem<TEST_DIM, TEST_EDGE_DIM, true, float>::loopGPUHierarchical);
}

void testPartitioning() {
  MY_SIZE num = 500;
  MY_SIZE N = 100, M = 200;
  MY_SIZE reset_every = 0;
  constexpr unsigned TEST_DIM = 4;
  constexpr unsigned TEST_EDGE_DIM = 4;
  testPartitioning<TEST_DIM, TEST_EDGE_DIM, false, float>(num, N, M,
                                                          reset_every);
  testPartitioning<TEST_DIM, TEST_EDGE_DIM, true, float>(num, N, M,
                                                         reset_every);
}

void generateTimesDifferentBlockDims() {
  // SOA
  generateTimesDifferentBlockDims<1, 1, true, float>(1153, 1153);
  generateTimesDifferentBlockDims<2, 1, true, float>(1153, 1153);
  generateTimesDifferentBlockDims<4, 1, true, float>(1153, 1153);
  generateTimesDifferentBlockDims<8, 1, true, float>(1153, 1153);
  generateTimesDifferentBlockDims<1, 1, true, float>(1153, 1153);
  generateTimesDifferentBlockDims<2, 2, true, float>(1153, 1153);
  generateTimesDifferentBlockDims<4, 4, true, float>(1153, 1153);
  generateTimesDifferentBlockDims<8, 8, true, float>(1153, 1153);
  // AOS
  generateTimesDifferentBlockDims<1, 1, false, float>(1153, 1153);
  generateTimesDifferentBlockDims<2, 1, false, float>(1153, 1153);
  generateTimesDifferentBlockDims<4, 1, false, float>(1153, 1153);
  generateTimesDifferentBlockDims<8, 1, false, float>(1153, 1153);
  generateTimesDifferentBlockDims<1, 1, false, float>(1153, 1153);
  generateTimesDifferentBlockDims<2, 2, false, float>(1153, 1153);
  generateTimesDifferentBlockDims<4, 4, false, float>(1153, 1153);
  generateTimesDifferentBlockDims<8, 8, false, float>(1153, 1153);
  // SOA
  generateTimesDifferentBlockDims<1, 1, true, double>(1153, 1153);
  generateTimesDifferentBlockDims<2, 1, true, double>(1153, 1153);
  generateTimesDifferentBlockDims<4, 1, true, double>(1153, 1153);
  generateTimesDifferentBlockDims<8, 1, true, double>(1153, 1153);
  generateTimesDifferentBlockDims<1, 1, true, double>(1153, 1153);
  generateTimesDifferentBlockDims<2, 2, true, double>(1153, 1153);
  generateTimesDifferentBlockDims<4, 4, true, double>(1153, 1153);
  generateTimesDifferentBlockDims<8, 8, true, double>(1153, 1153);
  // AOS
  generateTimesDifferentBlockDims<1, 1, false, double>(1153, 1153);
  generateTimesDifferentBlockDims<2, 1, false, double>(1153, 1153);
  generateTimesDifferentBlockDims<4, 1, false, double>(1153, 1153);
  generateTimesDifferentBlockDims<8, 1, false, double>(1153, 1153);
  generateTimesDifferentBlockDims<1, 1, false, double>(1153, 1153);
  generateTimesDifferentBlockDims<2, 2, false, double>(1153, 1153);
  generateTimesDifferentBlockDims<4, 4, false, double>(1153, 1153);
  generateTimesDifferentBlockDims<8, 8, false, double>(1153, 1153);
}

int main(int argc, const char **argv) {
  /*generateTimesFromFile(argc, argv);*/
  /*test();*/
  /*generateTimesDifferentBlockDims();*/
  testPartitioning();
  return 0;
}

// vim:set et sw=2 ts=2 fdm=marker:
