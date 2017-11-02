#ifndef LULESH_HPP_RVFK7I6X
#define LULESH_HPP_RVFK7I6X

#include <algorithm>

namespace lulesh {
static constexpr unsigned MESH_DIM = 8;
static constexpr unsigned POINT_DIM = 3;
static constexpr unsigned CELL_DIM0 = 3;
static constexpr unsigned CELL_DIM1 = 1;

// Sequential user function
#define USER_FUNCTION_SIGNATURE inline void user_func_host
#define RESTRICT
#include "lulesh_func.hpp"

// GPU user function
#define USER_FUNCTION_SIGNATURE __device__ void user_func_gpu
#define RESTRICT __restrict__
#include "lulesh_func.hpp"

// Sequential kernel
struct StepSeq {
  template <bool SOA>
  static void call(const void **_point_data, void *_point_data_out,
                   void **_cell_data, const MY_SIZE **cell_to_node, MY_SIZE ind,
                   const unsigned *point_stride, unsigned cell_stride) {
    const double *xyz_data = reinterpret_cast<const double *>(_point_data[1]);
    double *f_data_out = reinterpret_cast<double *>(_point_data_out);
    const double *sig_data = reinterpret_cast<const double *>(_cell_data[0]);
    double *determ_data = reinterpret_cast<double *>(_cell_data[1]);

    unsigned used_point_dim = !SOA ? POINT_DIM : 1;
    double *f_data_out_cur[MESH_DIM];
    const double *xyz_data_cur[MESH_DIM];
    for (unsigned i = 0; i < MESH_DIM; ++i) {
      f_data_out_cur[i] =
          f_data_out + used_point_dim * cell_to_node[0][MESH_DIM * ind + i];
      xyz_data_cur[i] =
          xyz_data + used_point_dim * cell_to_node[0][MESH_DIM * ind + i];
    }
    const double *sig_cur = sig_data + ind;
    double &determ = determ_data[ind];
    double inc[MESH_DIM * POINT_DIM];

    MY_SIZE f_stride = SOA ? point_stride[0] : 1;
    MY_SIZE xyz_stride = SOA ? point_stride[1] : 1;
    // Calling user function
    user_func_host(inc, xyz_data_cur, sig_cur, determ, xyz_stride, cell_stride);

    // Adding increment back
    for (unsigned j = 0; j < MESH_DIM; ++j) {
      for (unsigned i = 0; i < POINT_DIM; ++i) {
        f_data_out_cur[j][i * f_stride] += inc[i * MESH_DIM + j];
      }
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
    const double *xyz_data = reinterpret_cast<const double *>(_point_data[1]);
    double *f_data_out = reinterpret_cast<double *>(_point_data_out);
    const double *sig_data = reinterpret_cast<const double *>(_cell_data[0]);
    double *determ_data = reinterpret_cast<double *>(_cell_data[1]);

    unsigned used_point_dim = !SOA ? POINT_DIM : 1;
    double *f_data_out_cur[MESH_DIM];
    const double *xyz_data_cur[MESH_DIM];
    for (unsigned i = 0; i < MESH_DIM; ++i) {
      f_data_out_cur[i] =
          f_data_out + used_point_dim * cell_to_node[0][MESH_DIM * ind + i];
      xyz_data_cur[i] =
          xyz_data + used_point_dim * cell_to_node[0][MESH_DIM * ind + i];
    }
    const double *sig_cur = sig_data + ind;
    double &determ = determ_data[ind];
    double inc[MESH_DIM * POINT_DIM];

    MY_SIZE f_stride = SOA ? point_stride[0] : 1;
    MY_SIZE xyz_stride = SOA ? point_stride[1] : 1;
    // Calling user function
    user_func_gpu(inc, xyz_data_cur, sig_cur, determ, xyz_stride, cell_stride);

// Adding increment back
#pragma unroll
    for (unsigned j = 0; j < MESH_DIM; ++j) {
#pragma unroll
      for (unsigned i = 0; i < POINT_DIM; ++i) {
        f_data_out_cur[j][i * f_stride] += inc[i * MESH_DIM + j];
      }
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
  const double *xyz_data = reinterpret_cast<const double *>(_point_data[1]);
  double *f_data_out = reinterpret_cast<double *>(_point_data_out);
  const double *sig_data = reinterpret_cast<const double *>(_cell_data[0]);
  double *determ_data = reinterpret_cast<double *>(_cell_data[1]);

  const MY_SIZE bid = blockIdx.x;
  const MY_SIZE thread_ind = block_offsets[bid] + threadIdx.x;
  const MY_SIZE tid = threadIdx.x;

  const MY_SIZE cache_points_offset = points_to_be_cached_offsets[bid];
  const MY_SIZE num_cached_points =
      points_to_be_cached_offsets[bid + 1] - cache_points_offset;
  MY_SIZE shared_num_cached_points;
  if (32 % POINT_DIM == 0) {
    // Currently, shared memory bank conflict avoidance works only if 32 is
    // divisible by PointDim
    MY_SIZE needed_offset = 32 / POINT_DIM;
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

  extern __shared__ double point_cache[];

  MY_SIZE block_size = block_offsets[bid + 1] - block_offsets[bid];

  std::uint8_t our_colour;
  if (tid >= block_size) {
    our_colour = num_cell_colours[bid];
  } else {
    our_colour = cell_colours[thread_ind];
  }

  // Computation
  double increment[POINT_DIM * MESH_DIM];
  double *f_data_out_cur[MESH_DIM];
  if (tid < block_size) {
    unsigned used_point_dim = !SOA ? POINT_DIM : 1;
    const double *xyz_data_cur[MESH_DIM];
#pragma unroll
    for (unsigned i = 0; i < MESH_DIM; ++i) {
      f_data_out_cur[i] =
          point_cache + cell_to_node[0][thread_ind + i * num_cells];
      xyz_data_cur[i] =
          xyz_data +
          used_point_dim * cell_to_node[1][thread_ind + i * num_cells];
    }
    const double *sig_cur = sig_data + thread_ind;
    double &determ = determ_data[thread_ind];
    MY_SIZE xyz_stride = SOA ? point_stride[1] : 1;
    user_func_gpu(increment, xyz_data_cur, sig_cur, determ, xyz_stride,
                  cell_stride);
  }

  // Clear cache
  for (MY_SIZE i = tid; i < shared_num_cached_points * POINT_DIM;
       i += blockDim.x) {
    point_cache[i] = 0;
  }

  __syncthreads();

  // Accumulate increment to shared memory
  for (std::uint8_t cur_colour = 0; cur_colour < num_cell_colours[bid];
       ++cur_colour) {
    if (our_colour == cur_colour) {
      for (unsigned i = 0; i < POINT_DIM; ++i) {
        for (unsigned j = 0; j < MESH_DIM; ++j) {
          f_data_out_cur[j][i * shared_num_cached_points] +=
              increment[i * MESH_DIM + j];
        }
      }
    }
    __syncthreads();
  }

  // Cache out
  if (!SOA) {
    for (MY_SIZE i = tid; i < num_cached_points * POINT_DIM; i += blockDim.x) {
      MY_SIZE point_ind = i / POINT_DIM;
      MY_SIZE d = i % POINT_DIM;
      MY_SIZE g_ind = index<SOA>(
          point_stride[0], points_to_be_cached[cache_points_offset + point_ind],
          POINT_DIM, d);
      MY_SIZE c_ind =
          index<true>(shared_num_cached_points, point_ind, POINT_DIM, d);
      double result = f_data_out[g_ind] + point_cache[c_ind];
      f_data_out[g_ind] = result;
    }
  } else {
    for (MY_SIZE i = tid; i < num_cached_points; i += blockDim.x) {
      MY_SIZE g_point_to_be_cached =
          points_to_be_cached[cache_points_offset + i];
      double result[POINT_DIM];
#pragma unroll
      for (MY_SIZE d = 0; d < POINT_DIM; ++d) {
        MY_SIZE write_g_ind =
            index<SOA>(point_stride[0], g_point_to_be_cached, POINT_DIM, d);
        MY_SIZE write_c_ind =
            index<true>(shared_num_cached_points, i, POINT_DIM, d);

        result[d] = f_data_out[write_g_ind] + point_cache[write_c_ind];
      }
#pragma unroll
      for (MY_SIZE d = 0; d < POINT_DIM; ++d) {
        MY_SIZE write_g_ind =
            index<SOA>(point_stride[0], g_point_to_be_cached, POINT_DIM, d);
        f_data_out[write_g_ind] = result[d];
      }
    }
  }
}
}

#endif /* end of include guard: LULESH_HPP_RVFK7I6X */
// vim:set et sts=2 sw=2 ts=2:
