#ifndef MINI_AERO_HPP_SZYG7UXQ
#define MINI_AERO_HPP_SZYG7UXQ
#include <algorithm>

namespace mini_aero {
// Sequential user function
#define USER_FUNCTION_SIGNATURE inline void user_func_host
#define RESTRICT
#include "mini_aero_func.hpp"

// GPU user function
#define USER_FUNCTION_SIGNATURE __device__ void user_func_gpu
#define RESTRICT __restrict__
#include "mini_aero_func.hpp"

static constexpr unsigned MESH_DIM = 2;
static constexpr unsigned POINT_DIM0 = 5;
static constexpr unsigned POINT_DIM1 = 28;
static constexpr unsigned CELL_DIM = 12;

/**
 * SOA-AOS layouts:
 * - the layout of `point_data` is controlled by the SOA template parameter
 *   - if it's SOA, `point_stride` is the stride, otherwise that doesn't matter
 * - the layout of `cell_data` is always SOA with stride `cell_stride`
 * - the layout of `cell_to_node` is AOS except for the hierarchical case, where
 *   it is SOA
 */

/**
 * Iterates over faces
 * - mapping: face_cell_conn
 *     dim  :              2
 * - direct: face_coordinates, face_normal, face_tangent, face_binormal
 *     dim :                3,           3,            3,             3
 * - indirect: cell_values, cell_gradients, cell_limiters, cell_coordinates,
 *     cell_flux  <- result
 *     dim   :           5,             15,             5,                3,
 *             5
 */

// Sequential kernel
struct StepSeq {
  template <bool SOA>
  static void call(const void **_point_data, void *_point_data_out,
                   void **_cell_data, const MY_SIZE **cell_to_node, MY_SIZE ind,
                   const unsigned *point_stride, unsigned cell_stride) {
    const double *cell_other_data =
        reinterpret_cast<const double *>(_point_data[1]);
    double *cell_flux_out = reinterpret_cast<double *>(_point_data_out);
    const double *face_data = reinterpret_cast<const double *>(_cell_data[0]);

    unsigned used_point_dim0 = !SOA ? POINT_DIM0 : 1;
    unsigned used_point_dim1 = !SOA ? POINT_DIM1 : 1;
    MY_SIZE _point_stride0 = SOA ? point_stride[0] : 1;
    MY_SIZE _point_stride1 = SOA ? point_stride[1] : 1;
    const double *cell_values_left =
        cell_other_data +
        used_point_dim1 * cell_to_node[0][MESH_DIM * ind + 0] +
        0 * _point_stride1;
    const double *cell_values_right =
        cell_other_data +
        used_point_dim1 * cell_to_node[0][MESH_DIM * ind + 1] +
        0 * _point_stride1;
    const double *cell_coordinates_left =
        cell_other_data +
        used_point_dim1 * cell_to_node[0][MESH_DIM * ind + 0] +
        5 * _point_stride1;
    const double *cell_coordinates_right =
        cell_other_data +
        used_point_dim1 * cell_to_node[0][MESH_DIM * ind + 1] +
        5 * _point_stride1;
    const double *cell_gradients_left =
        cell_other_data +
        used_point_dim1 * cell_to_node[0][MESH_DIM * ind + 0] +
        8 * _point_stride1;
    const double *cell_gradients_right =
        cell_other_data +
        used_point_dim1 * cell_to_node[0][MESH_DIM * ind + 1] +
        8 * _point_stride1;
    const double *cell_limiters_left =
        cell_other_data +
        used_point_dim1 * cell_to_node[0][MESH_DIM * ind + 0] +
        23 * _point_stride1;
    const double *cell_limiters_right =
        cell_other_data +
        used_point_dim1 * cell_to_node[0][MESH_DIM * ind + 1] +
        23 * _point_stride1;
    double *cell_flux_out_left_cur =
        cell_flux_out + used_point_dim0 * cell_to_node[0][MESH_DIM * ind + 0];
    double *cell_flux_out_right_cur =
        cell_flux_out + used_point_dim0 * cell_to_node[0][MESH_DIM * ind + 1];
    const double *face_data_cur = face_data + ind;
    double inc[POINT_DIM0];

    // Calling user function
    user_func_host(
        inc, cell_values_left, cell_values_right, cell_coordinates_left,
        cell_coordinates_right, cell_gradients_left, cell_gradients_right,
        cell_limiters_left, cell_limiters_right,
        face_data_cur + 0 * cell_stride, face_data_cur + 3 * cell_stride,
        face_data_cur + 6 * cell_stride, face_data_cur + 9 * cell_stride,
        _point_stride1, cell_stride);

    // Adding increment back
    for (unsigned i = 0; i < POINT_DIM0; ++i) {
      cell_flux_out_left_cur[i * _point_stride0] -= inc[i];
      cell_flux_out_right_cur[i * _point_stride0] += inc[i];
    }
  }
};

// OMP kernel
// Should be the same as the sequential
using StepOMP = StepSeq;

// GPU global kernel
template <bool SOA>
__global__ void stepGPUGlobal(
    const void **__restrict__ _point_data, void *__restrict__ _point_data_out,
    void **__restrict__ _cell_data, const MY_SIZE **__restrict__ cell_to_node,
    MY_SIZE num_cells, MY_SIZE *__restrict__ point_stride, MY_SIZE cell_stride);

struct StepGPUGlobal {
  template <bool SOA>
  static void
  call(const void **__restrict__ point_data, void *__restrict__ point_data_out,
       void **__restrict__ cell_data, const MY_SIZE **__restrict__ cell_to_node,
       MY_SIZE num_cells, MY_SIZE *__restrict__ point_stride,
       MY_SIZE cell_stride, MY_SIZE num_blocks, MY_SIZE block_size) {
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
  if (ind < num_cells) {
    const double *cell_other_data =
        reinterpret_cast<const double *>(_point_data[1]);
    double *cell_flux_out = reinterpret_cast<double *>(_point_data_out);
    const double *face_data = reinterpret_cast<const double *>(_cell_data[0]);

    unsigned used_point_dim0 = !SOA ? POINT_DIM0 : 1;
    unsigned used_point_dim1 = !SOA ? POINT_DIM1 : 1;
    MY_SIZE _point_stride0 = SOA ? point_stride[0] : 1;
    MY_SIZE _point_stride1 = SOA ? point_stride[1] : 1;
    const double *cell_values_left =
        cell_other_data +
        used_point_dim1 * cell_to_node[0][MESH_DIM * ind + 0] +
        0 * _point_stride1;
    const double *cell_values_right =
        cell_other_data +
        used_point_dim1 * cell_to_node[0][MESH_DIM * ind + 1] +
        0 * _point_stride1;
    const double *cell_coordinates_left =
        cell_other_data +
        used_point_dim1 * cell_to_node[0][MESH_DIM * ind + 0] +
        5 * _point_stride1;
    const double *cell_coordinates_right =
        cell_other_data +
        used_point_dim1 * cell_to_node[0][MESH_DIM * ind + 1] +
        5 * _point_stride1;
    const double *cell_gradients_left =
        cell_other_data +
        used_point_dim1 * cell_to_node[0][MESH_DIM * ind + 0] +
        8 * _point_stride1;
    const double *cell_gradients_right =
        cell_other_data +
        used_point_dim1 * cell_to_node[0][MESH_DIM * ind + 1] +
        8 * _point_stride1;
    const double *cell_limiters_left =
        cell_other_data +
        used_point_dim1 * cell_to_node[0][MESH_DIM * ind + 0] +
        23 * _point_stride1;
    const double *cell_limiters_right =
        cell_other_data +
        used_point_dim1 * cell_to_node[0][MESH_DIM * ind + 1] +
        23 * _point_stride1;
    double *cell_flux_out_left_cur =
        cell_flux_out + used_point_dim0 * cell_to_node[0][MESH_DIM * ind + 0];
    double *cell_flux_out_right_cur =
        cell_flux_out + used_point_dim0 * cell_to_node[0][MESH_DIM * ind + 1];
    const double *face_data_cur = face_data + ind;
    double inc[POINT_DIM0];

    // Calling user function
    user_func_gpu(inc, cell_values_left, cell_values_right,
                  cell_coordinates_left, cell_coordinates_right,
                  cell_gradients_left, cell_gradients_right, cell_limiters_left,
                  cell_limiters_right, face_data_cur + 0 * cell_stride,
                  face_data_cur + 3 * cell_stride,
                  face_data_cur + 6 * cell_stride,
                  face_data_cur + 9 * cell_stride, _point_stride1, cell_stride);

// Adding back the increment
#pragma unroll
    for (unsigned i = 0; i < POINT_DIM0; ++i) {
      cell_flux_out_left_cur[i * _point_stride0] -= inc[i];
      cell_flux_out_right_cur[i * _point_stride0] += inc[i];
    }
  }
}

// GPU hierarchical kernel
template <bool SOA>
__global__ void stepGPUHierarchical(
    const void **__restrict__ _point_data, void *__restrict__ _point_data_out,
    const MY_SIZE *__restrict__ points_to_be_cached,
    const MY_SIZE *__restrict__ points_to_be_cached_offsets,
    void **__restrict__ _cell_data, const MY_SIZE **__restrict__ cell_to_node,
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
       void **__restrict__ cell_data, const MY_SIZE **__restrict__ cell_to_node,
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
    void **__restrict__ _cell_data, const MY_SIZE **__restrict__ cell_to_node,
    const std::uint8_t *__restrict__ num_cell_colours,
    const std::uint8_t *__restrict__ cell_colours,
    const MY_SIZE *__restrict__ block_offsets, MY_SIZE num_cells,
    const MY_SIZE *__restrict__ point_stride, MY_SIZE cell_stride) {
  const double *cell_other_data =
      reinterpret_cast<const double *>(_point_data[1]);
  double *cell_flux_out = reinterpret_cast<double *>(_point_data_out);
  const double *face_data = reinterpret_cast<const double *>(_cell_data[0]);

  const MY_SIZE bid = blockIdx.x;
  const MY_SIZE thread_ind = block_offsets[bid] + threadIdx.x;
  const MY_SIZE tid = threadIdx.x;

  const MY_SIZE cache_points_offset = points_to_be_cached_offsets[bid];
  const MY_SIZE num_cached_points =
      points_to_be_cached_offsets[bid + 1] - cache_points_offset;
  MY_SIZE shared_num_cached_points = num_cached_points;

  extern __shared__ double point_cache[];

  MY_SIZE block_size = block_offsets[bid + 1] - block_offsets[bid];

  std::uint8_t our_colour;
  if (tid >= block_size) {
    our_colour = num_cell_colours[bid];
  } else {
    our_colour = cell_colours[thread_ind];
  }

  // Computation
  double increment[POINT_DIM0];
  double *cell_flux_out_left_cur, *cell_flux_out_right_cur;
  if (tid < block_size) {
    unsigned used_point_dim1 = !SOA ? POINT_DIM1 : 1;
    MY_SIZE _point_stride1 = SOA ? point_stride[1] : 1;
    const double *cell_values_left =
        cell_other_data +
        used_point_dim1 * cell_to_node[1][thread_ind + num_cells *0] +
        0 * _point_stride1;
    const double *cell_values_right =
        cell_other_data +
        used_point_dim1 * cell_to_node[1][thread_ind + num_cells *1] +
        0 * _point_stride1;
    const double *cell_coordinates_left =
        cell_other_data +
        used_point_dim1 * cell_to_node[1][thread_ind + num_cells *0] +
        5 * _point_stride1;
    const double *cell_coordinates_right =
        cell_other_data +
        used_point_dim1 * cell_to_node[1][thread_ind + num_cells *1] +
        5 * _point_stride1;
    const double *cell_gradients_left =
        cell_other_data +
        used_point_dim1 * cell_to_node[1][thread_ind + num_cells *0] +
        8 * _point_stride1;
    const double *cell_gradients_right =
        cell_other_data +
        used_point_dim1 * cell_to_node[1][thread_ind + num_cells *1] +
        8 * _point_stride1;
    const double *cell_limiters_left =
        cell_other_data +
        used_point_dim1 * cell_to_node[1][thread_ind + num_cells *0] +
        23 * _point_stride1;
    const double *cell_limiters_right =
        cell_other_data +
        used_point_dim1 * cell_to_node[1][thread_ind + num_cells *1] +
        23 * _point_stride1;
    cell_flux_out_left_cur =
        point_cache + cell_to_node[0][thread_ind];
    cell_flux_out_right_cur =
        point_cache + cell_to_node[0][thread_ind + num_cells];
    const double *face_data_cur = face_data + thread_ind;
    user_func_gpu(increment, cell_values_left, cell_values_right,
                  cell_coordinates_left, cell_coordinates_right,
                  cell_gradients_left, cell_gradients_right, cell_limiters_left,
                  cell_limiters_right, face_data_cur + 0 * cell_stride,
                  face_data_cur + 3 * cell_stride,
                  face_data_cur + 6 * cell_stride,
                  face_data_cur + 9 * cell_stride, _point_stride1, cell_stride);
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
        cell_flux_out_left_cur[i * shared_num_cached_points] -= increment[i];
        cell_flux_out_right_cur[i * shared_num_cached_points] += increment[i];
      }
    }
    __syncthreads();
  }

  // Cache out
  if (!SOA) {
    for (MY_SIZE i = tid; i < num_cached_points * POINT_DIM0; i += blockDim.x) {
      MY_SIZE point_ind = i / POINT_DIM0;
      MY_SIZE d = i % POINT_DIM0;
      MY_SIZE g_ind = index<SOA>(
          point_stride[0], points_to_be_cached[cache_points_offset + point_ind],
          POINT_DIM0, d);
      MY_SIZE c_ind =
          index<true>(shared_num_cached_points, point_ind, POINT_DIM0, d);
      double result = cell_flux_out[g_ind] + point_cache[c_ind];
      cell_flux_out[g_ind] = result;
    }
  } else {
    for (MY_SIZE i = tid; i < num_cached_points; i += blockDim.x) {
      MY_SIZE g_point_to_be_cached =
          points_to_be_cached[cache_points_offset + i];
      double result[POINT_DIM0];
#pragma unroll
      for (MY_SIZE d = 0; d < POINT_DIM0; ++d) {
        MY_SIZE write_g_ind =
            index<SOA>(point_stride[0], g_point_to_be_cached, POINT_DIM0, d);
        MY_SIZE write_c_ind =
            index<true>(shared_num_cached_points, i, POINT_DIM0, d);

        result[d] = cell_flux_out[write_g_ind] + point_cache[write_c_ind];
      }
#pragma unroll
      for (MY_SIZE d = 0; d < POINT_DIM0; ++d) {
        MY_SIZE write_g_ind =
            index<SOA>(point_stride[0], g_point_to_be_cached, POINT_DIM0, d);
        cell_flux_out[write_g_ind] = result[d];
      }
    }
  }
}
}

#endif /* end of include guard: MINI_AERO_HPP_SZYG7UXQ */
