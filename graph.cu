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
#include "kernels/mine.hpp"
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
template <unsigned PointDim = 1, unsigned CellDim = 1, bool SOA = false,
          typename DataType, unsigned MeshDim>
__global__ void
problem_stepGPU(const DataType *__restrict__ point_weights,
                const DataType *__restrict__ cell_weights,
                const MY_SIZE *__restrict__ cell_list,
                DataType *__restrict__ out, const MY_SIZE cell_num_in_partition,
                const MY_SIZE num_points, const MY_SIZE num_cells) {
  static_assert(
      CellDim == PointDim || CellDim == 1,
      "I know of no reason why CellDim should be anything but 1 or PointDim");

  MY_SIZE id = blockIdx.x * blockDim.x + threadIdx.x;
  DataType inc[2 * PointDim];
  if (id < cell_num_in_partition) {
    #pragma unroll
    for (MY_SIZE offset = 0; offset < MeshDim; ++offset) {
      MY_SIZE edge_left = cell_list[MeshDim * id + offset];
      MY_SIZE edge_right =
          cell_list[MeshDim * id + (offset == MeshDim - 1 ? 0 : offset + 1)];
      #pragma unroll
      for (MY_SIZE d = 0; d < PointDim; ++d) {
        MY_SIZE ind_left = index<SOA>(num_points, edge_left, PointDim, d);
        MY_SIZE ind_right = index<SOA>(num_points, edge_right, PointDim, d);
        MY_SIZE cell_d = CellDim == 1 ? 0 : d;
        MY_SIZE cell_ind =
            index<true>(cell_num_in_partition, id, CellDim, cell_d);
        inc[d] =
            out[ind_right] + cell_weights[cell_ind] * point_weights[ind_left];
        inc[d + PointDim] =
            out[ind_left] + cell_weights[cell_ind] * point_weights[ind_right];
      }
      #pragma unroll
      for (MY_SIZE d = 0; d < PointDim; ++d) {
        MY_SIZE ind_left = index<SOA>(num_points, edge_left, PointDim, d);
        MY_SIZE ind_right = index<SOA>(num_points, edge_right, PointDim, d);

        out[ind_right] = inc[d];
        out[ind_left] = inc[d + PointDim];
      }
    }
  }
}
/* 1}}} */

/* problem_stepGPUHierarchical {{{1 */
template <unsigned PointDim = 1, unsigned CellDim = 1, bool SOA = false,
          bool PerDataCache = false, bool SOAInShared = true,
          typename DataType = float, unsigned MeshDim>
__global__ void problem_stepGPUHierarchical(
    const MY_SIZE *__restrict__ cell_list,
    const DataType *__restrict__ point_weights,
    DataType *__restrict__ point_weights_out,
    const DataType *__restrict__ cell_weights,
    const MY_SIZE *__restrict__ points_to_be_cached,
    const MY_SIZE *__restrict__ points_to_be_cached_offsets,
    const std::uint8_t *__restrict__ cell_colours,
    const std::uint8_t *__restrict__ num_cell_colours,
    const MY_SIZE *__restrict__ block_offsets, const MY_SIZE num_threads,
    const MY_SIZE num_points) {
  static_assert(
      CellDim == PointDim || CellDim == 1,
      "I know of no reason why CellDim should be anything but 1 or PointDim");

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
    our_colour = num_cell_colours[bid];
  } else {
    our_colour = cell_colours[thread_ind];
  }

  // Cache in
  if (PerDataCache) {
    if (!SOA && PointDim % 4 == 0 && std::is_same<DataType, float>::value) {
      for (MY_SIZE i = tid; i < PointDim * num_cached_points / 4;
           i += blockDim.x) {
        MY_SIZE point_ind = i * 4 / PointDim;
        MY_SIZE d = (i * 4) % PointDim;
        MY_SIZE g_ind = index<SOA>(
            num_points, points_to_be_cached[cache_points_offset + point_ind],
            PointDim, d);
        MY_SIZE c_ind0 = index<SOAInShared>(shared_num_cached_points, point_ind,
                                            PointDim, d + 0);
        MY_SIZE c_ind1 = index<SOAInShared>(shared_num_cached_points, point_ind,
                                            PointDim, d + 1);
        MY_SIZE c_ind2 = index<SOAInShared>(shared_num_cached_points, point_ind,
                                            PointDim, d + 2);
        MY_SIZE c_ind3 = index<SOAInShared>(shared_num_cached_points, point_ind,
                                            PointDim, d + 3);
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
        MY_SIZE g_ind = index<SOA>(
            num_points, points_to_be_cached[cache_points_offset + point_ind],
            PointDim, d);
        MY_SIZE c_ind0 = index<SOAInShared>(shared_num_cached_points, point_ind,
                                            PointDim, d + 0);
        MY_SIZE c_ind1 = index<SOAInShared>(shared_num_cached_points, point_ind,
                                            PointDim, d + 1);
        double2 tmp = point_weights3[g_ind / 2];
        point_cache[c_ind0] = tmp.x;
        point_cache[c_ind1] = tmp.y;
      }
    } else {
      for (MY_SIZE i = tid; i < PointDim * num_cached_points; i += blockDim.x) {
        MY_SIZE point_ind = i / PointDim;
        MY_SIZE d = i % PointDim;
        MY_SIZE g_ind = index<SOA>(
            num_points, points_to_be_cached[cache_points_offset + point_ind],
            PointDim, d);
        MY_SIZE c_ind = index<SOAInShared>(shared_num_cached_points, point_ind,
                                           PointDim, d);
        point_cache[c_ind] = point_weights[g_ind];
      }
    }
  } else {
    for (MY_SIZE i = tid; i < num_cached_points; i += blockDim.x) {
      MY_SIZE g_point_to_be_cached =
          points_to_be_cached[cache_points_offset + i];
      for (MY_SIZE d = 0; d < PointDim; ++d) {
        MY_SIZE c_ind, g_ind;
        g_ind = index<SOA>(num_points, g_point_to_be_cached, PointDim, d);
        c_ind = index<SOAInShared>(shared_num_cached_points, i, PointDim, d);

        point_cache[c_ind] = point_weights[g_ind];
      }
    }
  }

  __syncthreads();

  // Computation
  DataType increment[PointDim * MeshDim];

  #pragma unroll
  for (MY_SIZE i = 0; i < PointDim * MeshDim; ++i) {
    increment[i] = 0;
  }
  if (tid < block_size) {
    for (MY_SIZE offset = 0; offset < MeshDim; ++offset) {
      MY_SIZE offset_left = offset;
      MY_SIZE offset_right = offset == MeshDim - 1 ? 0 : offset + 1;
      MY_SIZE edge_left =
          cell_list[index<true>(num_threads, thread_ind, MeshDim, offset_left)];
      MY_SIZE edge_right = cell_list[index<true>(num_threads, thread_ind,
                                                 MeshDim, offset_right)];
      for (MY_SIZE d = 0; d < PointDim; ++d) {
        MY_SIZE left_ind = index<SOAInShared>(shared_num_cached_points,
                                              edge_left, PointDim, d);
        MY_SIZE right_ind = index<SOAInShared>(shared_num_cached_points,
                                               edge_right, PointDim, d);
        MY_SIZE cell_d = CellDim == 1 ? 0 : d;
        MY_SIZE cell_ind =
            index<true>(num_threads, thread_ind, PointDim, cell_d);

        increment[offset_right * PointDim + d] +=
            point_cache[left_ind] * cell_weights[cell_ind];
        increment[offset_left * PointDim + d] +=
            point_cache[right_ind] * cell_weights[cell_ind];
      }
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
  for (MY_SIZE i = 0; i < num_cell_colours[bid]; ++i) {
    if (our_colour == i) {
      for (MY_SIZE offset = 0; offset < MeshDim; ++offset) {
        MY_SIZE point =
            cell_list[index<true>(num_threads, thread_ind, MeshDim, offset)];
        for (MY_SIZE d = 0; d < PointDim; ++d) {
          MY_SIZE point_data_ind =
              index<SOAInShared>(shared_num_cached_points, point, PointDim, d);

          point_cache[point_data_ind] += increment[offset * PointDim + d];
        }
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
        MY_SIZE g_ind = index<SOA>(
            num_points, points_to_be_cached[cache_points_offset + point_ind],
            PointDim, d);
        MY_SIZE c_ind0 = index<SOAInShared>(shared_num_cached_points, point_ind,
                                            PointDim, d + 0);
        MY_SIZE c_ind1 = index<SOAInShared>(shared_num_cached_points, point_ind,
                                            PointDim, d + 1);
        MY_SIZE c_ind2 = index<SOAInShared>(shared_num_cached_points, point_ind,
                                            PointDim, d + 2);
        MY_SIZE c_ind3 = index<SOAInShared>(shared_num_cached_points, point_ind,
                                            PointDim, d + 3);
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
        MY_SIZE g_ind = index<SOA>(
            num_points, points_to_be_cached[cache_points_offset + point_ind],
            PointDim, d);
        MY_SIZE c_ind0 = index<SOAInShared>(shared_num_cached_points, point_ind,
                                            PointDim, d + 0);
        MY_SIZE c_ind1 = index<SOAInShared>(shared_num_cached_points, point_ind,
                                            PointDim, d + 1);
        double2 result = point_weights_out3[g_ind / 2];
        result.x += point_cache[c_ind0];
        result.y += point_cache[c_ind1];
        point_weights_out3[g_ind / 2] = result;
      }
    } else {
      for (MY_SIZE i = tid; i < num_cached_points * PointDim; i += blockDim.x) {
        MY_SIZE point_ind = i / PointDim;
        MY_SIZE d = i % PointDim;
        MY_SIZE g_ind = index<SOA>(
            num_points, points_to_be_cached[cache_points_offset + point_ind],
            PointDim, d);
        MY_SIZE c_ind = index<SOAInShared>(shared_num_cached_points, point_ind,
                                           PointDim, d);
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
            index<SOA>(num_points, g_point_to_be_cached, PointDim, d);
        MY_SIZE write_c_ind =
            index<SOAInShared>(shared_num_cached_points, i, PointDim, d);

        result[d] = point_weights_out[write_g_ind] + point_cache[write_c_ind];
      }
      #pragma unroll
      for (MY_SIZE d = 0; d < PointDim; ++d) {
        MY_SIZE write_g_ind =
            index<SOA>(num_points, g_point_to_be_cached, PointDim, d);
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

/* loopGPUCellCentred {{{1 */
template <unsigned PointDim, unsigned CellDim, bool SOA, typename DataType>
void Problem<PointDim, CellDim, SOA, DataType>::loopGPUCellCentred(
    MY_SIZE num) {
  std::vector<std::vector<MY_SIZE>> partition = mesh.colourCells();
  MY_SIZE num_of_colours = partition.size();
  assert(num_of_colours > 0);
  data_t point_weights2(
      data_t::create<DataType>(point_weights.getSize(), PointDim));
  std::copy(point_weights.begin(), point_weights.end(), point_weights2.begin());
  std::vector<data_t> d_cell_lists;
  std::vector<data_t> d_cell_weights;
  MY_SIZE total_num_cache_lines = 0;
  MY_SIZE total_num_blocks = 0;
  for (const std::vector<MY_SIZE> &colour : partition) {
    d_cell_lists.emplace_back(data_t::create<MY_SIZE>(colour.size(), MESH_DIM));
    d_cell_weights.emplace_back(
        data_t::create<DataType>(colour.size(), CellDim));
    for (std::size_t i = 0; i < colour.size(); ++i) {
      std::copy_n(mesh.cell_to_node.begin<MY_SIZE>() + MESH_DIM * colour[i],
                  MESH_DIM,
                  d_cell_lists.back().begin<MY_SIZE>() + MESH_DIM * i);
      for (unsigned d = 0; d < CellDim; ++d) {
        d_cell_weights.back().operator[]<DataType>(
            index<true>(colour.size(), i, CellDim, d)) =
            cell_weights.operator[]<DataType>(
                index<true>(mesh.numCells(), colour[i], CellDim, d));
      }
    }
    d_cell_lists.back().initDeviceMemory();
    d_cell_weights.back().initDeviceMemory();
    MY_SIZE num_blocks = std::ceil(static_cast<double>(colour.size()) /
                                   static_cast<double>(block_size));
    total_num_blocks += num_blocks;
    for (MY_SIZE i = 0; i < num_blocks; ++i) {
      total_num_cache_lines += countCacheLinesForBlock<PointDim, SOA, DataType>(
          d_cell_lists.back().begin<MY_SIZE>() + MESH_DIM * block_size * i,
          d_cell_lists.back().begin<MY_SIZE>() +
              MESH_DIM *
                  std::min<MY_SIZE>(colour.size(), block_size * (i + 1)));
    }
  }
  point_weights.initDeviceMemory();
  point_weights2.initDeviceMemory();
  CUDA_TIMER_START(t);
  for (MY_SIZE i = 0; i < num; ++i) {
    for (MY_SIZE c = 0; c < num_of_colours; ++c) {
      MY_SIZE num_blocks = std::ceil(static_cast<double>(partition[c].size()) /
                                     static_cast<double>(block_size));
      problem_stepGPU<PointDim, CellDim, SOA, DataType,
                      MESH_DIM><<<num_blocks, block_size>>>(
          point_weights.getDeviceData<DataType>(),
          d_cell_weights[c].getDeviceData<DataType>(),
          d_cell_lists[c].getDeviceData<MY_SIZE>(),
          point_weights2.getDeviceData<DataType>(), partition[c].size(),
          mesh.numPoints(), mesh.numCells());
      checkCudaErrors(cudaDeviceSynchronize());
    }
    TIMER_TOGGLE(t);
    checkCudaErrors(cudaMemcpy(point_weights.getDeviceData<DataType>(),
                               point_weights2.getDeviceData<DataType>(),
                               sizeof(DataType) * mesh.numPoints() * PointDim,
                               cudaMemcpyDeviceToDevice));
    TIMER_TOGGLE(t);
  }
  PRINT_BANDWIDTH(
      t, "loopGPUCellCentred",
      (sizeof(DataType) *
           (2.0 * PointDim * mesh.numPoints() + CellDim * mesh.numCells()) +
       1.0 * MESH_DIM * sizeof(MY_SIZE) * mesh.numCells()) *
          num,
      (sizeof(DataType) * mesh.numPoints() * PointDim * 2.0 + // point_weights
       sizeof(DataType) * mesh.numCells() * CellDim * 1.0 +   // d_cell_weights
       1.0 * sizeof(MY_SIZE) * mesh.numCells() * MESH_DIM     // d_cell_list
       ) * num);
  std::cout << " Needed " << num_of_colours << " colours" << std::endl;
  std::cout << "  average cache_line / block: "
            << static_cast<double>(total_num_cache_lines) / total_num_blocks
            << std::endl;
  PRINT_BANDWIDTH(
      t, " -cache line",
      num * (total_num_cache_lines * 32.0 * 2 +
             1.0 * CellDim * mesh.numCells() * sizeof(DataType) +
             1.0 * MESH_DIM * mesh.numCells() * sizeof(MY_SIZE)),
      num *
          (2 * 32.0 * total_num_cache_lines + // indirect accessed cache lines
           sizeof(DataType) * mesh.numCells() * CellDim * 1.0 + // cell_weights
           1.0 * sizeof(MY_SIZE) * mesh.numCells() * MESH_DIM   // cell_list
           ));
  point_weights.flushToHost();
}
/* 1}}} */

/* loopGPUHierarchical {{{1 */
template <unsigned PointDim, unsigned CellDim, bool SOA, typename DataType>
void Problem<PointDim, CellDim, SOA, DataType>::loopGPUHierarchical(
    MY_SIZE num) {
  TIMER_START(t_colouring);
  HierarchicalColourMemory<MESH_DIM, PointDim, CellDim, SOA, DataType> memory(
      *this, partition_vector);
  TIMER_PRINT(t_colouring, "Hierarchical colouring: colouring");
  const auto d_memory = memory.getDeviceMemoryOfOneColour();
  data_t point_weights_out(
      data_t::create<DataType>(point_weights.getSize(), PointDim));
  std::copy(point_weights.begin(), point_weights.end(),
            point_weights_out.begin());
  point_weights.initDeviceMemory();
  point_weights_out.initDeviceMemory();
  MY_SIZE total_cache_size = 0; // for bandwidth calculations
  DataType avg_num_cell_colours = 0;
  MY_SIZE total_num_blocks = 0;
  MY_SIZE total_shared_size = 0;
  size_t total_num_cache_lines = 0;
  for (MY_SIZE i = 0; i < memory.colours.size(); ++i) {
    const typename HierarchicalColourMemory<MESH_DIM, PointDim, CellDim, SOA,
                                            DataType>::MemoryOfOneColour
        &memory_of_one_colour = memory.colours[i];
    MY_SIZE num_threads = memory_of_one_colour.cell_list.size() / MESH_DIM;
    MY_SIZE num_blocks = static_cast<MY_SIZE>(
        std::ceil(static_cast<double>(num_threads) / block_size));
    total_cache_size += memory_of_one_colour.points_to_be_cached.size();
    avg_num_cell_colours +=
        std::accumulate(memory_of_one_colour.num_cell_colours.begin(),
                        memory_of_one_colour.num_cell_colours.end(), 0.0f);
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
      assert(memory.colours[colour_ind].cell_list.size() % MESH_DIM == 0);
      MY_SIZE num_threads =
          memory.colours[colour_ind].cell_list.size() / MESH_DIM;
      MY_SIZE num_blocks = memory.colours[colour_ind].num_cell_colours.size();
      assert(num_blocks == memory.colours[colour_ind].block_offsets.size() - 1);
      // + 32 in case it needs to avoid shared mem bank collisions
      MY_SIZE cache_size =
          sizeof(DataType) * (d_memory[colour_ind].shared_size + 32) * PointDim;
      TIMER_TOGGLE(timer_calc);
      problem_stepGPUHierarchical<
          PointDim, CellDim, SOA, !SOA, true, DataType,
          MESH_DIM><<<num_blocks, block_size, cache_size>>>(
          static_cast<MY_SIZE *>(d_memory[colour_ind].cell_list),
          point_weights.getDeviceData<DataType>(),
          point_weights_out.getDeviceData<DataType>(),
          static_cast<DataType *>(d_memory[colour_ind].cell_weights),
          static_cast<MY_SIZE *>(d_memory[colour_ind].points_to_be_cached),
          static_cast<MY_SIZE *>(
              d_memory[colour_ind].points_to_be_cached_offsets),
          static_cast<std::uint8_t *>(d_memory[colour_ind].cell_colours),
          static_cast<std::uint8_t *>(d_memory[colour_ind].num_cell_colours),
          static_cast<MY_SIZE *>(d_memory[colour_ind].block_offsets),
          num_threads, mesh.numPoints());
      TIMER_TOGGLE(timer_calc);
      checkCudaErrors(cudaDeviceSynchronize());
    }
    MY_SIZE copy_size = mesh.numPoints() * PointDim;
    TIMER_TOGGLE(timer_copy);
    MY_SIZE num_copy_blocks = std::ceil(static_cast<float>(copy_size) / 512.0);
    copyKernel<<<num_copy_blocks, 512>>>(
        point_weights_out.getDeviceData<DataType>(),
        point_weights.getDeviceData<DataType>(), copy_size);
    TIMER_TOGGLE(timer_copy);
  }
  PRINT_BANDWIDTH(
      timer_calc, "GPU HierarchicalColouring",
      num * ((2.0 * PointDim * mesh.numPoints() + CellDim * mesh.numCells()) *
                 sizeof(DataType) +
             1.0 * MESH_DIM * mesh.numCells() * sizeof(MY_SIZE)),
      num *
          (sizeof(DataType) * mesh.numPoints() * PointDim *
               2.0 +                                            // point_weights
           sizeof(DataType) * mesh.numCells() * CellDim * 1.0 + // cell_weights
           sizeof(MY_SIZE) * mesh.numCells() * 1.0 * MESH_DIM + // cell_list
           sizeof(MY_SIZE) * total_cache_size * 1.0 +
           sizeof(MY_SIZE) *
               (total_num_blocks * 1.0 +
                memory.colours.size()) + // points_to_be_cached_offsets
           sizeof(MY_SIZE) * (total_num_blocks * 1.0) + // block_offsets
           sizeof(std::uint8_t) * mesh.numCells()       // cell_colours
           ));
  PRINT_BANDWIDTH(timer_copy, " -copy",
                  2.0 * num * sizeof(DataType) * PointDim * mesh.numPoints(),
                  2.0 * num * sizeof(DataType) * PointDim * mesh.numPoints());
  std::cout << "  reuse factor: "
            << static_cast<double>(total_cache_size) /
                   (MESH_DIM * mesh.numCells())
            << std::endl;
  std::cout
      << "  cache/shared mem: "
      << static_cast<double>(total_cache_size) / total_shared_size
      << "\n  shared mem reuse factor (total shared / (MeshDim * #cells)): "
      << static_cast<double>(total_shared_size) / (MESH_DIM * mesh.numCells())
      << std::endl;
  std::cout << "  average cache_line / block: "
            << static_cast<double>(total_num_cache_lines) / total_num_blocks
            << std::endl;
  PRINT_BANDWIDTH(
      timer_calc, " -cache line",
      num * (total_num_cache_lines * 32.0 * 2 +
             1.0 * CellDim * mesh.numCells() * sizeof(DataType) +
             1.0 * MESH_DIM * mesh.numCells() * sizeof(MY_SIZE)),
      num *
          (2 * 32.0 * total_num_cache_lines + // indirect accessed cache lines
           sizeof(DataType) * mesh.numCells() * CellDim * 1.0 + // cell_weights
           sizeof(MY_SIZE) * mesh.numCells() * 1.0 * MESH_DIM + // cell_list
           sizeof(MY_SIZE) * total_cache_size * 1.0 +
           sizeof(MY_SIZE) *
               (total_num_blocks * 1.0 +
                memory.colours.size()) + // points_to_be_cached_offsets
           sizeof(MY_SIZE) * (total_num_blocks * 1.0) + // block_offsets
           sizeof(std::uint8_t) * mesh.numCells()       // cell_colours
           ));
  avg_num_cell_colours /= total_num_blocks;
  std::cout << "  average number of colours used: " << avg_num_cell_colours
            << std::endl;
  // ---------------
  // -  Finish up  -
  // ---------------
  point_weights.flushToHost();
}
/* 1}}} */

template <unsigned PointDim = 1, unsigned CellDim = 1, bool SOA = false,
          bool RunCPU = true, typename DataType = float>
void generateTimes(std::string in_file) {
  constexpr MY_SIZE num = 500;
  std::cout << ":::: Generating problems from file: " << in_file
            << "::::" << std::endl
            << "     Point dimension: " << PointDim
            << " Cell dimension: " << CellDim << " SOA: " << std::boolalpha
            << SOA << "\n     Data type: "
            << (sizeof(DataType) == sizeof(float) ? "float" : "double")
            << std::endl;
  std::function<void(
      implementation_algorithm_t<PointDim, CellDim, SOA, DataType>, MY_SIZE)>
      run = [&in_file](
          implementation_algorithm_t<PointDim, CellDim, SOA, DataType> algo,
          MY_SIZE num) {
        std::ifstream f(in_file);
        Problem<PointDim, CellDim, SOA, DataType> problem(f, 288);
        if (in_file.find("metis") < in_file.size()) {
          std::ifstream f_part(in_file + "_part");
          problem.readPartition(f_part);
          problem.reorderToPartition();
          problem.renumberPoints();
        }
        std::cout << "--Problem created" << std::endl;
        (problem.*algo)(num);
        std::cout << "--Problem finished." << std::endl;
      };
  run(&Problem<PointDim, CellDim, SOA,
               DataType>::template loopCPUCellCentred<mine::StepSeq>,
      RunCPU ? num : 1);
  run(&Problem<PointDim, CellDim, SOA,
               DataType>::template loopCPUCellCentredOMP<mine::StepOMP>,
      RunCPU ? num : 1);
  run(&Problem<PointDim, CellDim, SOA, DataType>::loopGPUCellCentred, num);
  run(&Problem<PointDim, CellDim, SOA, DataType>::loopGPUHierarchical, num);
  std::cout << "Finished." << std::endl;
}

template <unsigned PointDim = 1, unsigned CellDim = 1, bool SOA = false,
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
            << " Cell dimension: " << CellDim << " SOA: " << std::boolalpha
            << SOA << "\n     Data type: "
            << (sizeof(DataType) == sizeof(float) ? "float" : "double")
            << std::endl;
  std::function<void(
      implementation_algorithm_t<PointDim, CellDim, SOA, DataType>)>
      run = [&](
          implementation_algorithm_t<PointDim, CellDim, SOA, DataType> algo) {
        Problem<PointDim, CellDim, SOA, DataType> problem(
            std::move(StructuredProblem<PointDim, CellDim, SOA, DataType>(
                N, M, block_dims)));
        std::cout << "--Problem created" << std::endl;
        (problem.*algo)(num);
        std::cout << "--Problem finished." << std::endl;
      };
  run(&Problem<PointDim, CellDim, SOA, DataType>::loopGPUCellCentred);
  run(&Problem<PointDim, CellDim, SOA, DataType>::loopGPUHierarchical);
  std::cout << "Finished." << std::endl;
}

template <unsigned PointDim = 1, unsigned CellDim = 1, bool SOA = false,
          typename DataType = float>
void generateTimesDifferentBlockDims(MY_SIZE N, MY_SIZE M) {
  generateTimesWithBlockDims<PointDim, CellDim, SOA, DataType>(N, M, {0, 32});
  generateTimesWithBlockDims<PointDim, CellDim, SOA, DataType>(N, M, {2, 8});
  generateTimesWithBlockDims<PointDim, CellDim, SOA, DataType>(N, M, {4, 4});
  generateTimesWithBlockDims<PointDim, CellDim, SOA, DataType>(N, M, {0, 128});
  generateTimesWithBlockDims<PointDim, CellDim, SOA, DataType>(N, M, {2, 32});
  generateTimesWithBlockDims<PointDim, CellDim, SOA, DataType>(N, M, {4, 16});
  generateTimesWithBlockDims<PointDim, CellDim, SOA, DataType>(N, M, {8, 8});
  generateTimesWithBlockDims<PointDim, CellDim, SOA, DataType>(N, M, {0, 288});
  generateTimesWithBlockDims<PointDim, CellDim, SOA, DataType>(N, M, {2, 72});
  generateTimesWithBlockDims<PointDim, CellDim, SOA, DataType>(N, M, {4, 36});
  generateTimesWithBlockDims<PointDim, CellDim, SOA, DataType>(N, M, {12, 12});
  generateTimesWithBlockDims<PointDim, CellDim, SOA, DataType>(N, M, {9, 8});
  generateTimesWithBlockDims<PointDim, CellDim, SOA, DataType>(N, M, {0, 512});
  generateTimesWithBlockDims<PointDim, CellDim, SOA, DataType>(N, M, {2, 128});
  generateTimesWithBlockDims<PointDim, CellDim, SOA, DataType>(N, M, {4, 64});
  generateTimesWithBlockDims<PointDim, CellDim, SOA, DataType>(N, M, {8, 32});
  generateTimesWithBlockDims<PointDim, CellDim, SOA, DataType>(N, M, {16, 16});
}

void generateTimesFromFile(int argc, const char **argv) {
  if (argc <= 1) {
    std::cerr << "Usage: " << argv[0] << " <input mesh>" << std::endl;
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
  constexpr unsigned TEST_DIM = 4;
  constexpr unsigned TEST_CELL_DIM = 4;
  using TEST_DATA_TYPE = float;
  testTwoImplementations<TEST_DIM, TEST_CELL_DIM, false, TEST_DATA_TYPE>(
      num, N, M,
      &Problem<TEST_DIM, TEST_CELL_DIM, false, TEST_DATA_TYPE>::loopGPUCellCentred,
      &Problem<TEST_DIM, TEST_CELL_DIM, false, TEST_DATA_TYPE>::
          loopCPUCellCentred<mine::StepOMP4<
                 TEST_DIM,TEST_CELL_DIM, TEST_DATA_TYPE>>);
  testTwoImplementations<TEST_DIM, TEST_CELL_DIM, true, TEST_DATA_TYPE>(
      num, N, M,
      &Problem<TEST_DIM, TEST_CELL_DIM, true, TEST_DATA_TYPE>::loopGPUCellCentred,
      &Problem<TEST_DIM, TEST_CELL_DIM, true,
  TEST_DATA_TYPE>::loopCPUCellCentredOMP<mine::StepOMP4<TEST_DIM, TEST_CELL_DIM,
      TEST_DATA_TYPE>>);
}

void testReordering() {
  MY_SIZE num = 500;
  MY_SIZE N = 100, M = 200;
  constexpr unsigned TEST_DIM = 4;
  constexpr unsigned TEST_CELL_DIM = 4;
  testReordering<TEST_DIM, TEST_CELL_DIM, false, float>(
      num, N, M, &Problem<TEST_DIM, TEST_CELL_DIM, false,
                          float>::loopCPUCellCentredOMP<mine::StepOMP4<
                            TEST_DIM,TEST_CELL_DIM, float>>,
      &Problem<TEST_DIM, TEST_CELL_DIM, false,
               float>::loopCPUCellCentredOMP<mine::StepOMP4<
                 TEST_DIM, TEST_CELL_DIM, float>>);
  testReordering<TEST_DIM, TEST_CELL_DIM, true, float>(
      num, N, M,
      &Problem<TEST_DIM, TEST_CELL_DIM, true, float>::loopCPUCellCentredOMP<
      mine::StepOMP4<TEST_DIM,TEST_CELL_DIM, float>>,
      &Problem<TEST_DIM, TEST_CELL_DIM, true,
   float>::loopCPUCellCentredOMP<mine::StepOMP4<TEST_DIM, TEST_CELL_DIM, float>>);
}

/*void testPartitioning() {*/
/*  MY_SIZE num = 500;*/
/*  MY_SIZE N = 100, M = 200;*/
/*  constexpr unsigned TEST_DIM = 4;*/
/*  constexpr unsigned TEST_CELL_DIM = 4;*/
/*  testPartitioning<TEST_DIM, TEST_CELL_DIM, false, float>(num, N, M);*/
/*  testPartitioning<TEST_DIM, TEST_CELL_DIM, true, float>(num, N, M);*/
/*}*/

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
  test();
  testReordering();
  /*testPartitioning();*/
  /*generateTimesDifferentBlockDims();*/
  /*measurePartitioning();*/
  return 0;
}

// vim:set et sw=2 ts=2 fdm=marker:
