#ifndef RES_CALC_HPP_BTXZV4YZ
#define RES_CALC_HPP_BTXZV4YZ
#include <algorithm>

namespace res_calc {
// Sequential user function
#define USER_FUNCTION_SIGNATURE inline void user_func_host
#include "res_calc_func.hpp"

// GPU user function
#define USER_FUNCTION_SIGNATURE __device__ void user_func_gpu
#include "res_calc_func.hpp"

/**
 * SOA-AOS layouts:
 * - the layout of `point_data` is controlled by the SOA template parameter
 *   - if it's SOA, `point_stride` is the stride, otherwise that doesn't matter
 * - the layout of `cell_data` is always SOA with stride `point_stride`
 * - the layout of `cell_to_node` is AOS except for the hierarchical case, where
 *   it is SOA
 */

constexpr unsigned MAPPING_DIM = 2;
constexpr unsigned Q_DIM = 4, X_DIM = 2, RES_DIM = 4, ADT_DIM = 1;

// Sequential kernel
struct StepSeq {

  template <bool SOA>
  static void call(const void **_point_data, void *_point_data_out,
                   const void **, const MY_SIZE **cell_to_node, MY_SIZE ind,
                   const unsigned *point_stride, unsigned) {
    const double *point_data_x =
        reinterpret_cast<const double *>(_point_data[1]);
    const double *point_data_q =
        reinterpret_cast<const double *>(_point_data[2]);
    const double *point_data_adt =
        reinterpret_cast<const double *>(_point_data[3]);
    double *point_data_out_res = reinterpret_cast<double *>(_point_data_out);

    unsigned used_point_dim_x = !SOA ? X_DIM : 1;
    unsigned used_point_dim_q = !SOA ? Q_DIM : 1;
    unsigned used_point_dim_RES = !SOA ? RES_DIM : 1;
    const double *point_data_x1 =
        point_data_x +
        used_point_dim_x * cell_to_node[1][MAPPING_DIM * ind + 0];
    const double *point_data_x2 =
        point_data_x +
        used_point_dim_x * cell_to_node[1][MAPPING_DIM * ind + 1];
    const double *point_data_q1 =
        point_data_q +
        used_point_dim_q * cell_to_node[2][MAPPING_DIM * ind + 0];
    const double *point_data_q2 =
        point_data_q +
        used_point_dim_q * cell_to_node[2][MAPPING_DIM * ind + 1];
    const double *point_data_adt1 =
        point_data_adt + cell_to_node[3][MAPPING_DIM * ind + 0];
    const double *point_data_adt2 =
        point_data_adt + cell_to_node[3][MAPPING_DIM * ind + 1];

    double *point_data_out_cur_res1 =
        point_data_out_res +
        used_point_dim_RES * cell_to_node[0][MAPPING_DIM * ind + 0];
    double *point_data_out_cur_res2 =
        point_data_out_res +
        used_point_dim_RES * cell_to_node[0][MAPPING_DIM * ind + 1];

    double inc[MAPPING_DIM * RES_DIM];

    // x
    MY_SIZE _point_stride0 = SOA ? point_stride[1] : 1;
    // q
    MY_SIZE _point_stride1 = SOA ? point_stride[2] : 1;
    // res
    MY_SIZE _point_stride3 = SOA ? point_stride[0] : 1;
    // Calling user function
    user_func_host(point_data_x1, point_data_x2, point_data_q1, point_data_q2,
                   point_data_adt1, point_data_adt2, inc, &inc[RES_DIM],
                   _point_stride0, _point_stride1);

    // Adding increment back
    for (unsigned i = 0; i < RES_DIM; ++i) {
      point_data_out_cur_res1[i * _point_stride3] += inc[i];
      point_data_out_cur_res2[i * _point_stride3] += inc[i + RES_DIM];
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
              void *__restrict__ _point_data_out, const void **__restrict__,
              const MY_SIZE **__restrict__ cell_to_node, MY_SIZE num_cells,
              MY_SIZE *__restrict__ point_stride, MY_SIZE) {
  MY_SIZE ind = blockIdx.x * blockDim.x + threadIdx.x;
  if (ind < num_cells) {
    const double *point_data_x =
        reinterpret_cast<const double *>(_point_data[1]);
    const double *point_data_q =
        reinterpret_cast<const double *>(_point_data[2]);
    const double *point_data_adt =
        reinterpret_cast<const double *>(_point_data[3]);
    double *point_data_out_res = reinterpret_cast<double *>(_point_data_out);

    unsigned used_point_dim_x = !SOA ? X_DIM : 1;
    unsigned used_point_dim_q = !SOA ? Q_DIM : 1;
    unsigned used_point_dim_RES = !SOA ? RES_DIM : 1;
    const double *point_data_x1 =
        point_data_x +
        used_point_dim_x * cell_to_node[1][MAPPING_DIM * ind + 0];
    const double *point_data_x2 =
        point_data_x +
        used_point_dim_x * cell_to_node[1][MAPPING_DIM * ind + 1];
    const double *point_data_q1 =
        point_data_q +
        used_point_dim_q * cell_to_node[2][MAPPING_DIM * ind + 0];
    const double *point_data_q2 =
        point_data_q +
        used_point_dim_q * cell_to_node[2][MAPPING_DIM * ind + 1];
    const double *point_data_adt1 =
        point_data_adt + cell_to_node[3][MAPPING_DIM * ind + 0];
    const double *point_data_adt2 =
        point_data_adt + cell_to_node[3][MAPPING_DIM * ind + 1];

    double *point_data_out_cur_res1 =
        point_data_out_res +
        used_point_dim_RES * cell_to_node[0][MAPPING_DIM * ind + 0];
    double *point_data_out_cur_res2 =
        point_data_out_res +
        used_point_dim_RES * cell_to_node[0][MAPPING_DIM * ind + 1];

    double inc[MAPPING_DIM * RES_DIM];

    // x
    MY_SIZE _point_stride0 = SOA ? point_stride[1] : 1;
    // q
    MY_SIZE _point_stride1 = SOA ? point_stride[2] : 1;
    // res
    MY_SIZE _point_stride3 = SOA ? point_stride[0] : 1;
    // Calling user function
    user_func_gpu(point_data_x1, point_data_x2, point_data_q1, point_data_q2,
                  point_data_adt1, point_data_adt2, inc, &inc[RES_DIM],
                  _point_stride0, _point_stride1);

// Adding back the increment
#pragma unroll
    for (unsigned i = 0; i < RES_DIM; ++i) {
      point_data_out_cur_res1[i * _point_stride3] += inc[i];
      point_data_out_cur_res2[i * _point_stride3] += inc[i + RES_DIM];
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
__global__ void StepGPUHierarchical(
    const void **__restrict__ _point_data, void *__restrict__ _point_data_out,
    const MY_SIZE *__restrict__ points_to_be_cached,
    const MY_SIZE *__restrict__ points_to_be_cached_offsets,
    const void **__restrict__, const MY_SIZE **__restrict__ cell_to_node,
    const std::uint8_t *__restrict__ num_cell_colours,
    const std::uint8_t *__restrict__ cell_colours,
    const MY_SIZE *__restrict__ block_offsets, MY_SIZE num_cells,
    const MY_SIZE *__restrict__ point_stride, MY_SIZE) {
  using DataType =
      double; // there are different algorithms based on the type of
              // the cached points
  const DataType *__restrict__ point_data_x =
      reinterpret_cast<const DataType *>(_point_data[1]);
  const DataType *__restrict__ point_data_q =
      reinterpret_cast<const DataType *>(_point_data[2]);
  const DataType *__restrict__ point_data_adt =
      reinterpret_cast<const DataType *>(_point_data[3]);
  DataType *__restrict__ point_data_out_res =
      reinterpret_cast<DataType *>(_point_data_out);

  double2 *__restrict__ point_data_out_double2 =
      reinterpret_cast<double2 *>(point_data_out_res);

  const MY_SIZE bid = blockIdx.x;
  const MY_SIZE thread_ind = block_offsets[bid] + threadIdx.x;
  const MY_SIZE tid = threadIdx.x;

  const MY_SIZE cache_points_offset = points_to_be_cached_offsets[bid];
  const MY_SIZE num_cached_points =
      points_to_be_cached_offsets[bid + 1] - cache_points_offset;
  MY_SIZE shared_num_cached_points;
  if (32 % RES_DIM == 0) {
    // Currently, shared memory bank conflict avoidance works only if 32 is
    // divisible by RES_DIM
    MY_SIZE needed_offset = 32 / RES_DIM;
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
  // ELM NEM KELL
  // Computation
  DataType increment[RES_DIM * MAPPING_DIM];
  DataType *point_data0_res1, *point_data0_res2;
  if (tid < block_size) {
    point_data0_res1 =
        point_cache + cell_to_node[0][thread_ind + 0 * num_cells];
    point_data0_res2 =
        point_cache + cell_to_node[0][thread_ind + 1 * num_cells];
    unsigned used_point_dim_x = !SOA ? X_DIM : 1;
    unsigned used_point_dim_q = !SOA ? Q_DIM : 1;
    const double *point_data_x1 =
        point_data_x +
        used_point_dim_x * cell_to_node[1][thread_ind + 0 * num_cells];
    const double *point_data_x2 =
        point_data_x +
        used_point_dim_x * cell_to_node[1][thread_ind + 1 * num_cells];
    const double *point_data_q1 =
        point_data_q +
        used_point_dim_q * cell_to_node[2][thread_ind + 0 * num_cells];
    const double *point_data_q2 =
        point_data_q +
        used_point_dim_q * cell_to_node[2][thread_ind + 1 * num_cells];
    const double *point_data_adt1 =
        point_data_adt + cell_to_node[3][thread_ind + 0 * num_cells];
    const double *point_data_adt2 =
        point_data_adt + cell_to_node[3][thread_ind + 1 * num_cells];

    // x
    MY_SIZE _point_stride0 = SOA ? point_stride[1] : 1;
    // q
    MY_SIZE _point_stride1 = SOA ? point_stride[2] : 1;

    user_func_gpu(point_data_x1, point_data_x2, point_data_q1, point_data_q2,
                  point_data_adt1, point_data_adt2, increment,
                  increment + RES_DIM, _point_stride0, _point_stride1);
  }

  __syncthreads();

  // Clear cache
  for (MY_SIZE i = tid; i < shared_num_cached_points * RES_DIM;
       i += blockDim.x) {
    point_cache[i] = 0;
  }

  __syncthreads();

  // Accumulate increment to shared memory
  for (std::uint8_t cur_colour = 0; cur_colour < num_cell_colours[bid];
       ++cur_colour) {
    if (our_colour == cur_colour) {
      for (unsigned i = 0; i < RES_DIM; ++i) {
        point_data0_res1[i * shared_num_cached_points] +=
            increment[i + 0 * RES_DIM];
        point_data0_res2[i * shared_num_cached_points] +=
            increment[i + 1 * RES_DIM];
      }
    }
    __syncthreads();
  }

  // Cache out
  if (!SOA) {
    if (!SOA && RES_DIM % 2 == 0 && std::is_same<DataType, double>::value) {
      for (MY_SIZE i = tid; i < RES_DIM * num_cached_points / 2;
           i += blockDim.x) {
        MY_SIZE point_ind = i * 2 / RES_DIM;
        MY_SIZE d = (i * 2) % RES_DIM;
        MY_SIZE g_ind = index<SOA>(
            point_stride[0],
            points_to_be_cached[cache_points_offset + point_ind], RES_DIM, d);
        MY_SIZE c_ind0 =
            index<true>(shared_num_cached_points, point_ind, RES_DIM, d + 0);
        MY_SIZE c_ind1 =
            index<true>(shared_num_cached_points, point_ind, RES_DIM, d + 1);
        double2 result = point_data_out_double2[g_ind / 2];
        result.x += point_cache[c_ind0];
        result.y += point_cache[c_ind1];
        point_data_out_double2[g_ind / 2] = result;
      }
    }
  } else {
    for (MY_SIZE i = tid; i < num_cached_points; i += blockDim.x) {
      MY_SIZE g_point_to_be_cached =
          points_to_be_cached[cache_points_offset + i];
      DataType result[RES_DIM];
#pragma unroll
      for (MY_SIZE d = 0; d < RES_DIM; ++d) {
        MY_SIZE write_g_ind =
            index<SOA>(point_stride[0], g_point_to_be_cached, RES_DIM, d);
        MY_SIZE write_c_ind =
            index<true>(shared_num_cached_points, i, RES_DIM, d);

        result[d] = point_data_out_res[write_g_ind] + point_cache[write_c_ind];
      }
#pragma unroll
      for (MY_SIZE d = 0; d < RES_DIM; ++d) {
        MY_SIZE write_g_ind =
            index<SOA>(point_stride[0], g_point_to_be_cached, RES_DIM, d);
        point_data_out_res[write_g_ind] = result[d];
      }
    }
  }
}
}
#endif /* end of include guard: RES_CALC_HPP_BTXZV4YZ */
// vim:set et sts=2 sw=2 ts=2:
