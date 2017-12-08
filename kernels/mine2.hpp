#ifndef MINE2_HPP_ZZO46FD8
#define MINE2_HPP_ZZO46FD8

#include <algorithm>

namespace mine2 {
// Sequential user function
#define USER_FUNCTION_SIGNATURE inline void mine2_func_host
#define RESTRICT
#include "mine2_func.hpp"

// GPU user function
#define USER_FUNCTION_SIGNATURE __device__ void mine2_func_gpu
#define RESTRICT __restrict__
#include "mine2_func.hpp"

constexpr unsigned MESH_DIM0 = 1, MESH_DIM1 = 2;
constexpr unsigned POINT_DIM0 = 2, CELL_DIM0 = 2, POINT_DIM1 = 4, CELL_DIM1 = 1,
                   CELL_DIM2 = 3;

// Sequential kernel
struct StepSeq {
  template <bool SOA>
  static void call(const void **_point_data, void *_point_data_out,
                   void **_cell_data, const MY_SIZE **cell_to_node,
                   MY_SIZE ind, const unsigned *point_stride,
                   unsigned cell_stride) {
    const float *point_data0 = reinterpret_cast<const float *>(_point_data[0]);
    const float *point_data1 = reinterpret_cast<const float *>(_point_data[1]);
    float *point_data_out = reinterpret_cast<float *>(_point_data_out);
    const float *cell_data0 = reinterpret_cast<const float *>(_cell_data[0]);
    const float *cell_data1 = reinterpret_cast<const float *>(_cell_data[1]);
    const double *cell_data2 = reinterpret_cast<const double *>(_cell_data[2]);

    unsigned used_point_dim0 = !SOA ? POINT_DIM0 : 1;
    unsigned used_point_dim1 = !SOA ? POINT_DIM1 : 1;
    const float *point_data0_cur =
        point_data0 + used_point_dim0 * cell_to_node[0][MESH_DIM0 * ind + 0];
    float *point_data_out_cur =
        point_data_out + used_point_dim0 * cell_to_node[0][MESH_DIM0 * ind + 0];
    const float *point_data1_left =
        point_data1 + used_point_dim1 * cell_to_node[1][MESH_DIM1 * ind + 0];
    const float *point_data1_right =
        point_data1 + used_point_dim1 * cell_to_node[1][MESH_DIM1 * ind + 1];
    const float *cell_data0_cur = cell_data0 + ind;
    const float *cell_data1_cur = cell_data1 + ind;
    const double *cell_data2_cur = cell_data2 + ind;
    float inc[MESH_DIM0 * POINT_DIM0];

    MY_SIZE _point_stride0 = SOA ? point_stride[0] : 1;
    MY_SIZE _point_stride1 = SOA ? point_stride[1] : 1;

    // Calling user function
    mine2_func_host(point_data0_cur, inc, point_data1_left, point_data1_right,
                    cell_data0_cur, cell_data1_cur, cell_data2_cur,
                    _point_stride0, _point_stride1, cell_stride);

    // Adding increment back
    for (unsigned i = 0; i < POINT_DIM0; ++i) {
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
              void **__restrict__ _cell_data,
              const MY_SIZE **__restrict__ cell_to_node, MY_SIZE num_cells,
              MY_SIZE *__restrict__ point_stride, MY_SIZE cell_stride);

struct StepGPUGlobal {
  template <bool SOA>
  static void call(const void **__restrict__ point_data,
                   void *__restrict__ point_data_out,
                   void **__restrict__ cell_data,
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
              void **__restrict__ _cell_data,
              const MY_SIZE **__restrict__ cell_to_node, MY_SIZE num_cells,
              MY_SIZE *__restrict__ point_stride, MY_SIZE cell_stride) {
  MY_SIZE ind = blockIdx.x * blockDim.x + threadIdx.x;
  float inc[MESH_DIM0 * POINT_DIM0];
  if (ind < num_cells) {
    const float *__restrict__ point_data0 =
        reinterpret_cast<const float *>(_point_data[0]);
    float *__restrict__ point_data_out =
        reinterpret_cast<float *>(_point_data_out);
    const float *__restrict__ point_data1 =
        reinterpret_cast<const float *>(_point_data[1]);
    const float *__restrict__ cell_data0 =
        reinterpret_cast<const float *>(_cell_data[0]);
    const float *__restrict__ cell_data1 =
        reinterpret_cast<const float *>(_cell_data[1]);
    const double *__restrict__ cell_data2 =
        reinterpret_cast<const double *>(_cell_data[2]);

    unsigned used_point_dim0 = !SOA ? POINT_DIM0 : 1;
    unsigned used_point_dim1 = !SOA ? POINT_DIM1 : 1;
    const float *point_data0_cur =
        point_data0 + used_point_dim0 * cell_to_node[0][MESH_DIM0 * ind + 0];
    const float *point_data1_left =
        point_data1 + used_point_dim1 * cell_to_node[1][MESH_DIM1 * ind + 0];
    const float *point_data1_right =
        point_data1 + used_point_dim1 * cell_to_node[1][MESH_DIM1 * ind + 1];
    float *point_data_out_cur =
        point_data_out + used_point_dim0 * cell_to_node[0][MESH_DIM0 * ind + 0];
    const float *cell_data0_cur = cell_data0 + ind;
    const float *cell_data1_cur = cell_data1 + ind;
    const double *cell_data2_cur = cell_data2 + ind;

    MY_SIZE _point_stride0 = SOA ? point_stride[0] : 1;
    MY_SIZE _point_stride1 = SOA ? point_stride[1] : 1;

    // Calling user function
    mine2_func_gpu(point_data0_cur, inc, point_data1_left, point_data1_right,
                   cell_data0_cur, cell_data1_cur, cell_data2_cur,
                   _point_stride0, _point_stride1, cell_stride);

// Adding back the increment
#pragma unroll
    for (unsigned i = 0; i < POINT_DIM0; ++i) {
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
    void **__restrict__ _cell_data,
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
       void **__restrict__ cell_data,
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
    void **__restrict__ _cell_data,
    const MY_SIZE **__restrict__ cell_to_node,
    const std::uint8_t *__restrict__ num_cell_colours,
    const std::uint8_t *__restrict__ cell_colours,
    const MY_SIZE *__restrict__ block_offsets, MY_SIZE num_cells,
    const MY_SIZE *__restrict__ point_stride, MY_SIZE cell_stride) {
  const float *__restrict__ point_data0 =
      reinterpret_cast<const float *>(_point_data[0]);
  float *__restrict__ point_data_out =
      reinterpret_cast<float *>(_point_data_out);
  const float *__restrict__ point_data1 =
      reinterpret_cast<const float *>(_point_data[1]);
  const float *__restrict__ cell_data0 =
      reinterpret_cast<const float *>(_cell_data[0]);
  const float *__restrict__ cell_data1 =
      reinterpret_cast<const float *>(_cell_data[1]);
  const double *__restrict__ cell_data2 =
      reinterpret_cast<const double *>(_cell_data[2]);

  const float4 *__restrict__ point_data0_float4 =
      reinterpret_cast<const float4 *>(point_data0);
  float4 *__restrict__ point_data_out_float4 =
      reinterpret_cast<float4 *>(point_data_out);
  const double2 *__restrict__ point_data0_double2 =
      reinterpret_cast<const double2 *>(point_data0);
  double2 *__restrict__ point_data_out_double2 =
      reinterpret_cast<double2 *>(point_data_out);

  const MY_SIZE bid = blockIdx.x;
  const MY_SIZE thread_ind = block_offsets[bid] + threadIdx.x;
  const MY_SIZE tid = threadIdx.x;

  const MY_SIZE cache_points_offset = points_to_be_cached_offsets[bid];
  const MY_SIZE num_cached_points =
      points_to_be_cached_offsets[bid + 1] - cache_points_offset;
  MY_SIZE shared_num_cached_points;
  if (32 % POINT_DIM0 == 0) {
    // Currently, shared memory bank conflict avoidance works only if 32 is
    // divisible by PointDim
    MY_SIZE needed_offset = 32 / POINT_DIM0;
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

  extern __shared__ float point_cache[];

  MY_SIZE block_size = block_offsets[bid + 1] - block_offsets[bid];

  std::uint8_t our_colour;
  if (tid >= block_size) {
    our_colour = num_cell_colours[bid];
  } else {
    our_colour = cell_colours[thread_ind];
  }

  // Cache in
  if (!SOA) {
    if (!SOA && POINT_DIM0 % 4 == 0 && std::is_same<float, float>::value) {
      for (MY_SIZE i = tid; i < POINT_DIM0 * num_cached_points / 4;
           i += blockDim.x) {
        MY_SIZE point_ind = i * 4 / POINT_DIM0;
        MY_SIZE d = (i * 4) % POINT_DIM0;
        MY_SIZE g_ind =
            index<SOA>(point_stride[0],
                       points_to_be_cached[cache_points_offset + point_ind],
                       POINT_DIM0, d);
        MY_SIZE c_ind0 =
            index<true>(shared_num_cached_points, point_ind, POINT_DIM0, d + 0);
        MY_SIZE c_ind1 =
            index<true>(shared_num_cached_points, point_ind, POINT_DIM0, d + 1);
        MY_SIZE c_ind2 =
            index<true>(shared_num_cached_points, point_ind, POINT_DIM0, d + 2);
        MY_SIZE c_ind3 =
            index<true>(shared_num_cached_points, point_ind, POINT_DIM0, d + 3);
        float4 tmp = point_data0_float4[g_ind / 4];
        point_cache[c_ind0] = tmp.x;
        point_cache[c_ind1] = tmp.y;
        point_cache[c_ind2] = tmp.z;
        point_cache[c_ind3] = tmp.w;
      }
    } else if (!SOA && POINT_DIM0 % 2 == 0 &&
               std::is_same<float, double>::value) {
      for (MY_SIZE i = tid; i < POINT_DIM0 * num_cached_points / 2;
           i += blockDim.x) {
        MY_SIZE point_ind = i * 2 / POINT_DIM0;
        MY_SIZE d = (i * 2) % POINT_DIM0;
        MY_SIZE g_ind =
            index<SOA>(point_stride[0],
                       points_to_be_cached[cache_points_offset + point_ind],
                       POINT_DIM0, d);
        MY_SIZE c_ind0 =
            index<true>(shared_num_cached_points, point_ind, POINT_DIM0, d + 0);
        MY_SIZE c_ind1 =
            index<true>(shared_num_cached_points, point_ind, POINT_DIM0, d + 1);
        double2 tmp = point_data0_double2[g_ind / 2];
        point_cache[c_ind0] = tmp.x;
        point_cache[c_ind1] = tmp.y;
      }
    } else {
      for (MY_SIZE i = tid; i < POINT_DIM0 * num_cached_points;
           i += blockDim.x) {
        MY_SIZE point_ind = i / POINT_DIM0;
        MY_SIZE d = i % POINT_DIM0;
        MY_SIZE g_ind =
            index<SOA>(point_stride[0],
                       points_to_be_cached[cache_points_offset + point_ind],
                       POINT_DIM0, d);
        MY_SIZE c_ind =
            index<true>(shared_num_cached_points, point_ind, POINT_DIM0, d);
        point_cache[c_ind] = point_data0[g_ind];
      }
    }
  } else {
    for (MY_SIZE i = tid; i < num_cached_points; i += blockDim.x) {
      MY_SIZE g_point_to_be_cached =
          points_to_be_cached[cache_points_offset + i];
      for (MY_SIZE d = 0; d < POINT_DIM0; ++d) {
        MY_SIZE c_ind, g_ind;
        g_ind =
            index<SOA>(point_stride[0], g_point_to_be_cached, POINT_DIM0, d);
        c_ind = index<true>(shared_num_cached_points, i, POINT_DIM0, d);

        point_cache[c_ind] = point_data0[g_ind];
      }
    }
  }

  __syncthreads();

  // Computation
  float increment[POINT_DIM0 * MESH_DIM0];
  float *point_data0_cur;
  if (tid < block_size) {
    point_data0_cur = point_cache + cell_to_node[0][thread_ind + 0 * num_cells];
    const float *point_data1_left =
        point_data1 + index<SOA>(point_stride[1],
                                 cell_to_node[1][thread_ind + 0 * num_cells],
                                 POINT_DIM1, 0);
    const float *point_data1_right =
        point_data1 + index<SOA>(point_stride[1],
                                 cell_to_node[1][thread_ind + 1 * num_cells],
                                 POINT_DIM1, 0);
    const float *cell_data0_cur = cell_data0 + thread_ind;
    const float *cell_data1_cur = cell_data1 + thread_ind;
    const double *cell_data2_cur = cell_data2 + thread_ind;
    const MY_SIZE _point_stride1 = SOA ? point_stride[1] : 1;
    mine2_func_gpu(point_data0_cur, increment + 0 * POINT_DIM0,
                   point_data1_left, point_data1_right, cell_data0_cur,
                   cell_data1_cur, cell_data2_cur, shared_num_cached_points,
                   _point_stride1, cell_stride);
  }

  __syncthreads();

  // Clear cache
  for (MY_SIZE i = tid; i < shared_num_cached_points * POINT_DIM0;
       i += blockDim.x) {
    point_cache[i] = 0;
  }

  __syncthreads();

  // Accumulate increment to shared memory
  for (std::uint8_t cur_colour = 0; cur_colour < num_cell_colours[bid];
       ++cur_colour) {
    if (our_colour == cur_colour) {
      for (unsigned i = 0; i < POINT_DIM0; ++i) {
        point_data0_cur[i * shared_num_cached_points] += increment[i];
      }
    }
    __syncthreads();
  }

  // Cache out
  if (!SOA) {
    if (!SOA && POINT_DIM0 % 4 == 0 && std::is_same<float, float>::value) {
      for (MY_SIZE i = tid; i < POINT_DIM0 * num_cached_points / 4;
           i += blockDim.x) {
        MY_SIZE point_ind = i * 4 / POINT_DIM0;
        MY_SIZE d = (i * 4) % POINT_DIM0;
        MY_SIZE g_ind =
            index<SOA>(point_stride[0],
                       points_to_be_cached[cache_points_offset + point_ind],
                       POINT_DIM0, d);
        MY_SIZE c_ind0 =
            index<true>(shared_num_cached_points, point_ind, POINT_DIM0, d + 0);
        MY_SIZE c_ind1 =
            index<true>(shared_num_cached_points, point_ind, POINT_DIM0, d + 1);
        MY_SIZE c_ind2 =
            index<true>(shared_num_cached_points, point_ind, POINT_DIM0, d + 2);
        MY_SIZE c_ind3 =
            index<true>(shared_num_cached_points, point_ind, POINT_DIM0, d + 3);
        float4 result = point_data_out_float4[g_ind / 4];
        result.x += point_cache[c_ind0];
        result.y += point_cache[c_ind1];
        result.z += point_cache[c_ind2];
        result.w += point_cache[c_ind3];
        point_data_out_float4[g_ind / 4] = result;
      }
    } else if (!SOA && POINT_DIM0 % 2 == 0 &&
               std::is_same<float, double>::value) {
      for (MY_SIZE i = tid; i < POINT_DIM0 * num_cached_points / 2;
           i += blockDim.x) {
        MY_SIZE point_ind = i * 2 / POINT_DIM0;
        MY_SIZE d = (i * 2) % POINT_DIM0;
        MY_SIZE g_ind =
            index<SOA>(point_stride[0],
                       points_to_be_cached[cache_points_offset + point_ind],
                       POINT_DIM0, d);
        MY_SIZE c_ind0 =
            index<true>(shared_num_cached_points, point_ind, POINT_DIM0, d + 0);
        MY_SIZE c_ind1 =
            index<true>(shared_num_cached_points, point_ind, POINT_DIM0, d + 1);
        double2 result = point_data_out_double2[g_ind / 2];
        result.x += point_cache[c_ind0];
        result.y += point_cache[c_ind1];
        point_data_out_double2[g_ind / 2] = result;
      }
    } else {
      for (MY_SIZE i = tid; i < num_cached_points * POINT_DIM0;
           i += blockDim.x) {
        MY_SIZE point_ind = i / POINT_DIM0;
        MY_SIZE d = i % POINT_DIM0;
        MY_SIZE g_ind =
            index<SOA>(point_stride[0],
                       points_to_be_cached[cache_points_offset + point_ind],
                       POINT_DIM0, d);
        MY_SIZE c_ind =
            index<true>(shared_num_cached_points, point_ind, POINT_DIM0, d);
        float result = point_data_out[g_ind] + point_cache[c_ind];
        point_data_out[g_ind] = result;
      }
    }
  } else {
    for (MY_SIZE i = tid; i < num_cached_points; i += blockDim.x) {
      MY_SIZE g_point_to_be_cached =
          points_to_be_cached[cache_points_offset + i];
      float result[POINT_DIM0];
#pragma unroll
      for (MY_SIZE d = 0; d < POINT_DIM0; ++d) {
        MY_SIZE write_g_ind =
            index<SOA>(point_stride[0], g_point_to_be_cached, POINT_DIM0, d);
        MY_SIZE write_c_ind =
            index<true>(shared_num_cached_points, i, POINT_DIM0, d);

        result[d] = point_data_out[write_g_ind] + point_cache[write_c_ind];
      }
#pragma unroll
      for (MY_SIZE d = 0; d < POINT_DIM0; ++d) {
        MY_SIZE write_g_ind =
            index<SOA>(point_stride[0], g_point_to_be_cached, POINT_DIM0, d);
        point_data_out[write_g_ind] = result[d];
      }
    }
  }
}
}

#endif /* end of include guard: MINE2_HPP_ZZO46FD8 */
