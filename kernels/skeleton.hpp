#ifndef SKELETON_HPP_BTXZV4YZ
#define SKELETON_HPP_BTXZV4YZ
#include <algorithm>

namespace skeleton {
// Sequential user function
#define USER_FUNCTION_SIGNATURE inline void user_func_host
#define RESTRICT
#include "skeleton_func.hpp"

// GPU user function
#define USER_FUNCTION_SIGNATURE __device__ void user_func_gpu
#define RESTRICT __restrict__
#include "skeleton_func.hpp"

static constexpr unsigned MESH_DIM = 1;
static constexpr unsigned POINT_DIM = 1;
static constexpr unsigned CELL_DIM = 1;
/**
 * SOA-AOS layouts:
 * - the layout of `point_data` is controlled by the SOA template parameter
 *   - if it's SOA, `point_stride` is the stride, otherwise that doesn't matter
 * - the layout of `cell_data` is always SOA with stride `point_stride`
 * - the layout of `cell_to_node` is AOS except for the hierarchical case, where
 *   it is SOA
 */

// Sequential kernel
struct StepSeq {
  template <bool SOA>
  static void call(const void **_point_data, void *_point_data_out,
                   const void **_cell_data, const MY_SIZE **cell_to_node,
                   MY_SIZE ind, const unsigned *point_stride,
                   unsigned cell_stride) {
    const float *point_data0 = reinterpret_cast<const float *>(_point_data[0]);
    float *point_data_out = reinterpret_cast<float *>(_point_data_out);
    const float *cell_data0 = reinterpret_cast<const float *>(_cell_data[0]);

    unsigned used_point_dim = !SOA ? POINT_DIM : 1;
    const float *point_data0_cur =
        point_data0 + used_point_dim * cell_to_node[0][MESH_DIM * ind + 0];
    float *point_data_out_cur =
        point_data_out + used_point_dim * cell_to_node[0][MESH_DIM * ind + 0];
    const float *cell_data0_cur = cell_data0 + ind;
    float inc[MESH_DIM * POINT_DIM];

    MY_SIZE _point_stride0 = SOA ? point_stride[0] : 1;
    // Calling user function
    user_func_host(point_data0_cur, inc, cell_data0_cur, _point_stride0,
                   cell_stride);

    // Adding increment back
    for (unsigned i = 0; i < POINT_DIM; ++i) {
      point_data_out_cur[i * _point_stride0] += inc[i];
    }
  }
};

// OMP kernel
// Should be the same as the sequential
using StepOMP = StepSeq;

// GPU global kernel
template <bool SOA>
__global__ void
stepGPUGlobal(const void **__restrict__ _point_data,
              void *__restrict__ _point_data_out,
              const void **__restrict__ _cell_data,
              const MY_SIZE **__restrict__ cell_to_node, MY_SIZE num_cells,
              MY_SIZE *__restrict__ point_stride, MY_SIZE cell_stride);

struct StepGPUGlobal {
  template <bool SOA>
  static void call(const void **__restrict__ point_data,
                   void *__restrict__ point_data_out,
                   const void **__restrict__ cell_data,
                   const MY_SIZE **__restrict__ cell_to_node, MY_SIZE num_cells,
                   MY_SIZE *__restrict__ point_stride, MY_SIZE cell_stride,
                   MY_SIZE num_blocks, MY_SIZE block_size) {
    // nvcc doesn't support a static method as a kernel
    stepGPUGlobal<SOA><<<num_blocks, block_size>>>(
        point_data, point_data_out, cell_data, cell_to_node, num_cells,
        point_stride, cell_stride);
  }
};

template <bool SOA>
__global__ void
stepGPUGlobal(const void **__restrict__ _point_data,
              void *__restrict__ _point_data_out,
              const void **__restrict__ _cell_data,
              const MY_SIZE **__restrict__ cell_to_node, MY_SIZE num_cells,
              MY_SIZE *__restrict__ point_stride, MY_SIZE cell_stride) {
  MY_SIZE ind = blockIdx.x * blockDim.x + threadIdx.x;
  float inc[MESH_DIM * POINT_DIM];
  if (ind < num_cells) {
    const float *__restrict__ point_data0 =
        reinterpret_cast<const float *>(_point_data[0]);
    float *__restrict__ point_data_out =
        reinterpret_cast<float *>(_point_data_out);
    const float *__restrict__ cell_data0 =
        reinterpret_cast<const float *>(_cell_data[0]);

    unsigned used_point_dim = !SOA ? POINT_DIM : 1;
    const float *point_data0_cur =
        point_data0 + used_point_dim * cell_to_node[0][MESH_DIM * ind + 0];
    float *point_data_out_cur =
        point_data_out + used_point_dim * cell_to_node[0][MESH_DIM * ind + 0];
    const float *cell_data0_cur = cell_data[0] + ind;

    MY_SIZE _point_stride0 = SOA ? point_stride0 : 1;

    // Calling user function
    user_func_gpu(point_data_cur, inc, cell_data_cur, _point_stride0,
                  cell_stride);

// Adding back the increment
#pragma unroll
    for (unsigned i = 0; i < POINT_DIM; ++i) {
      point_data_out_cur[i * _point_stride0] += inc[i];
    }
  }
}

// GPU hierarchical kernel
template <bool SOA>
__global__ void stepGPUHierarchical(
    const void **__restrict__ _point_data, void *__restrict__ _point_data_out,
    const MY_SIZE *__restrict__ points_to_be_cached,
    const MY_SIZE *__restrict__ points_to_be_cached_offsets,
    const void **__restrict__ _cell_data,
    const MY_SIZE **__restrict__ cell_to_node,
    const std::uint8_t *__restrict__ num_cell_colours,
    const std::uint8_t *__restrict__ cell_colours,
    const MY_SIZE *__restrict__ block_offsets, MY_SIZE num_cells,
    const MY_SIZE *__restrict__ point_stride, MY_SIZE cell_stride);

struct StepGPUHierarchical {
  template <bool SOA>
  static void
  call(const void **__restrict__ point_data, void *__restrict__ point_data_out,
       const MY_SIZE *__restrict__ points_to_be_cached,
       const MY_SIZE *__restrict__ points_to_be_cached_offsets,
       const void **__restrict__ cell_data,
       const MY_SIZE **__restrict__ cell_to_node,
       const std::uint8_t *__restrict__ num_cell_colours,
       const std::uint8_t *__restrict__ cell_colours,
       const MY_SIZE *__restrict__ block_offsets, MY_SIZE num_cells,
       const MY_SIZE *__restrict__ point_stride, MY_SIZE cell_stride,
       MY_SIZE num_blocks, unsigned block_size, unsigned cache_size) {
    // nvcc doesn't support a static method as a kernel
    stepGPUHierarchical<SOA><<<num_blocks, block_size, cache_size>>>(
        point_data, point_data_out, points_to_be_cached,
        points_to_be_cached_offsets, cell_data, cell_to_node, num_cell_colours,
        cell_colours, block_offsets, num_cells, point_stride, cell_stride);
  }
};

template <bool SOA>
__global__ void stepGPUHierarchical(
    const void **__restrict__ _point_data, void *__restrict__ _point_data_out,
    const MY_SIZE *__restrict__ points_to_be_cached,
    const MY_SIZE *__restrict__ points_to_be_cached_offsets,
    const void **__restrict__ _cell_data,
    const MY_SIZE **__restrict__ cell_to_node,
    const std::uint8_t *__restrict__ num_cell_colours,
    const std::uint8_t *__restrict__ cell_colours,
    const MY_SIZE *__restrict__ block_offsets, MY_SIZE num_cells,
    const MY_SIZE *__restrict__ point_stride, MY_SIZE cell_stride) {
  using DataType = float; // there are different algorithms based on the type of
                          // the cached points
  const DataType *__restrict__ point_data0 =
      reinterpret_cast<const DataType *>(_point_data[0]);
  DataType *__restrict__ point_data_out =
      reinterpret_cast<DataType *>(_point_data_out);
  const DataType *__restrict__ cell_data0 =
      reinterpret_cast<const DataType *>(_cell_data[0]);

  const float4 *__restrict__ point_data0_float4 =
      reinterpret_cast<const float4 *>(point_data0);
  float4 *__restrict__ point_data_out_float4 =
      reinterpret_cast<float4 *>(point_data_out);
  const double2 *__restrict__ point_data0_double2 =
      reinterpret_cast<const double2 *>(point_data0);
  double2 *__restrict__ point_data_out_double2 =
      reinterpret_cast<double2 *>(point_data_out);

  constexpr unsigned MESH_DIM = MESH_DIM;

  const MY_SIZE bid = blockIdx.x;
  const MY_SIZE thread_ind = block_offsets[bid] + threadIdx.x;
  const MY_SIZE tid = threadIdx.x;

  const MY_SIZE cache_points_offset = points_to_be_cached_offsets[bid];
  const MY_SIZE num_cached_points =
      points_to_be_cached_offsets[bid + 1] - cache_points_offset;
  MY_SIZE shared_num_cached_points;
  if (32 % PointDim == 0) {
    // Currently, shared memory bank conflict avoidance works only if 32 is
    // divisible by PointDim
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

  extern __shared__ DataType point_cache[];

  MY_SIZE block_size = block_offsets[bid + 1] - block_offsets[bid];

  std::uint8_t our_colour;
  if (tid >= block_size) {
    our_colour = num_cell_colours[bid];
  } else {
    our_colour = cell_colours[thread_ind];
  }

  // Cache in
  if (!SOA) {
    if (!SOA && PointDim % 4 == 0 && std::is_same<DataType, float>::value) {
      for (MY_SIZE i = tid; i < PointDim * num_cached_points / 4;
           i += blockDim.x) {
        MY_SIZE point_ind = i * 4 / PointDim;
        MY_SIZE d = (i * 4) % PointDim;
        MY_SIZE g_ind = index<SOA>(
            point_stride[0],
            points_to_be_cached[cache_points_offset + point_ind], PointDim, d);
        MY_SIZE c_ind0 =
            index<true>(shared_num_cached_points, point_ind, PointDim, d + 0);
        MY_SIZE c_ind1 =
            index<true>(shared_num_cached_points, point_ind, PointDim, d + 1);
        MY_SIZE c_ind2 =
            index<true>(shared_num_cached_points, point_ind, PointDim, d + 2);
        MY_SIZE c_ind3 =
            index<true>(shared_num_cached_points, point_ind, PointDim, d + 3);
        float4 tmp = point_data0_float4[g_ind / 4];
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
            point_stride[0],
            points_to_be_cached[cache_points_offset + point_ind], PointDim, d);
        MY_SIZE c_ind0 =
            index<true>(shared_num_cached_points, point_ind, PointDim, d + 0);
        MY_SIZE c_ind1 =
            index<true>(shared_num_cached_points, point_ind, PointDim, d + 1);
        double2 tmp = point_data0_double2[g_ind / 2];
        point_cache[c_ind0] = tmp.x;
        point_cache[c_ind1] = tmp.y;
      }
    } else {
      for (MY_SIZE i = tid; i < PointDim * num_cached_points; i += blockDim.x) {
        MY_SIZE point_ind = i / PointDim;
        MY_SIZE d = i % PointDim;
        MY_SIZE g_ind = index<SOA>(
            point_stride[0],
            points_to_be_cached[cache_points_offset + point_ind], PointDim, d);
        MY_SIZE c_ind =
            index<true>(shared_num_cached_points, point_ind, PointDim, d);
        point_cache[c_ind] = point_data0[g_ind];
      }
    }
  } else {
    for (MY_SIZE i = tid; i < num_cached_points; i += blockDim.x) {
      MY_SIZE g_point_to_be_cached =
          points_to_be_cached[cache_points_offset + i];
      for (MY_SIZE d = 0; d < PointDim; ++d) {
        MY_SIZE c_ind, g_ind;
        g_ind = index<SOA>(point_stride[0], g_point_to_be_cached, PointDim, d);
        c_ind = index<true>(shared_num_cached_points, i, PointDim, d);

        point_cache[c_ind] = point_data0[g_ind];
      }
    }
  }

  __syncthreads();

  // Computation
  DataType increment[PointDim * MESH_DIM];
  DataType *point_data0_cur;
  if (tid < block_size) {
    point_data0_cur = point_cache + cell_to_node[0][thread_ind + 0 * num_cells];
    const DataType *cell_data0_cur = cell_data0 + thread_ind;
    // If more than one mapping:
    // MY_SIZE _point_stride1 = SOA ? point_stride[1] : 1;
    user_func_gpu(point_data0_cur, increment + 0 * PointDim, cell_data0_cur,
                  shared_num_cached_points, cell_stride);
  }

  __syncthreads();

  // Clear cache
  for (MY_SIZE i = tid; i < shared_num_cached_points * PointDim;
       i += blockDim.x) {
    point_cache[i] = 0;
  }

  __syncthreads();

  // Accumulate increment to shared memory
  for (std::uint8_t cur_colour = 0; cur_colour < num_cell_colours[bid];
       ++cur_colour) {
    if (our_colour == cur_colour) {
      for (unsigned i = 0; i < PointDim; ++i) {
        point_data0_cur[i * shared_num_cached_points] +=
            increment[i + 0 * PointDim];
      }
    }
    __syncthreads();
  }

  // Cache out
  if (!SOA) {
    if (!SOA && PointDim % 4 == 0 && std::is_same<DataType, float>::value) {
      for (MY_SIZE i = tid; i < PointDim * num_cached_points / 4;
           i += blockDim.x) {
        MY_SIZE point_ind = i * 4 / PointDim;
        MY_SIZE d = (i * 4) % PointDim;
        MY_SIZE g_ind = index<SOA>(
            point_stride[0],
            points_to_be_cached[cache_points_offset + point_ind], PointDim, d);
        MY_SIZE c_ind0 =
            index<true>(shared_num_cached_points, point_ind, PointDim, d + 0);
        MY_SIZE c_ind1 =
            index<true>(shared_num_cached_points, point_ind, PointDim, d + 1);
        MY_SIZE c_ind2 =
            index<true>(shared_num_cached_points, point_ind, PointDim, d + 2);
        MY_SIZE c_ind3 =
            index<true>(shared_num_cached_points, point_ind, PointDim, d + 3);
        float4 result = point_data_out_float4[g_ind / 4];
        result.x += point_cache[c_ind0];
        result.y += point_cache[c_ind1];
        result.z += point_cache[c_ind2];
        result.w += point_cache[c_ind3];
        point_data_out_float4[g_ind / 4] = result;
      }
    } else if (!SOA && PointDim % 2 == 0 &&
               std::is_same<DataType, double>::value) {
      for (MY_SIZE i = tid; i < PointDim * num_cached_points / 2;
           i += blockDim.x) {
        MY_SIZE point_ind = i * 2 / PointDim;
        MY_SIZE d = (i * 2) % PointDim;
        MY_SIZE g_ind = index<SOA>(
            point_stride[0],
            points_to_be_cached[cache_points_offset + point_ind], PointDim, d);
        MY_SIZE c_ind0 =
            index<true>(shared_num_cached_points, point_ind, PointDim, d + 0);
        MY_SIZE c_ind1 =
            index<true>(shared_num_cached_points, point_ind, PointDim, d + 1);
        double2 result = point_data_out_double2[g_ind / 2];
        result.x += point_cache[c_ind0];
        result.y += point_cache[c_ind1];
        point_data_out_double2[g_ind / 2] = result;
      }
    } else {
      for (MY_SIZE i = tid; i < num_cached_points * PointDim; i += blockDim.x) {
        MY_SIZE point_ind = i / PointDim;
        MY_SIZE d = i % PointDim;
        MY_SIZE g_ind = index<SOA>(
            point_stride[0],
            points_to_be_cached[cache_points_offset + point_ind], PointDim, d);
        MY_SIZE c_ind =
            index<true>(shared_num_cached_points, point_ind, PointDim, d);
        DataType result = point_data_out[g_ind] + point_cache[c_ind];
        point_data_out[g_ind] = result;
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
            index<SOA>(point_stride[0], g_point_to_be_cached, PointDim, d);
        MY_SIZE write_c_ind =
            index<true>(shared_num_cached_points, i, PointDim, d);

        result[d] = point_data_out[write_g_ind] + point_cache[write_c_ind];
      }
#pragma unroll
      for (MY_SIZE d = 0; d < PointDim; ++d) {
        MY_SIZE write_g_ind =
            index<SOA>(point_stride[0], g_point_to_be_cached, PointDim, d);
        point_data_out[write_g_ind] = result[d];
      }
    }
  }
}
}
#endif /* end of include guard: SKELETON_HPP_BTXZV4YZ */
// vim:set et sts=2 sw=2 ts=2:
