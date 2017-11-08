#ifndef SKELETON_HPP_BTXZV4YZ
#define SKELETON_HPP_BTXZV4YZ
#include <algorithm>

namespace volna {
// Sequential user function
#define USER_FUNCTION_SIGNATURE inline void user_func_host
#define RESTRICT
#include "volna_func.hpp"

// GPU user function
#define USER_FUNCTION_SIGNATURE __device__ void user_func_gpu
#define RESTRICT __restrict__
#include "volna_func.hpp"

static constexpr unsigned MESH_DIM = 2;
static constexpr unsigned INC_DIM = 4;
static constexpr unsigned FLUX_DIM = 3;
static constexpr unsigned CELL_DIM = 2;
/**
 * SOA-AOS layouts:
 * - the layout of `point_data` is controlled by the SOA template parameter
 *   - if it's SOA, `point_stride` is the stride, otherwise that doesn't matter
 * - the layout of `cell_data` is always SOA with stride `cell_stride`
 * - the layout of `cell_to_node` is AOS except for the hierarchical case, where
 *   it is SOA
 */

// Sequential kernel
struct StepSeq {
  template <bool SOA>
  static void call(const void **_point_data, void *_point_data_out,
                   void **_cell_data, const MY_SIZE **cell_to_node,
                   MY_SIZE ind, const unsigned *point_stride,
                   unsigned cell_stride) {
    const float *point_cellVolumes = reinterpret_cast<const float *>(_point_data[1]);
    float *point_data_out = reinterpret_cast<float *>(_point_data_out);
    const float *cell_fluxes = reinterpret_cast<const float *>(_cell_data[0]);
    const float *cell_bathySource = reinterpret_cast<const float *>(_cell_data[1]);
    const float *cell_edgeNormals = reinterpret_cast<const float *>(_cell_data[2]);
    const int *cell_isBoundary = reinterpret_cast<const int *>(_cell_data[3]);

    const float *cellVolumes0 =
        point_cellVolumes + cell_to_node[0][MESH_DIM * ind + 0];
    const float *cellVolumes1 =
        point_cellVolumes + cell_to_node[0][MESH_DIM * ind + 1];
    const unsigned used_point_dim = !SOA ? INC_DIM : 1;
    float *point_data_out_left =
        point_data_out + used_point_dim * cell_to_node[0][MESH_DIM * ind + 0];
    float *point_data_out_right =
        point_data_out + used_point_dim * cell_to_node[0][MESH_DIM * ind + 1];
    const float *cell_data_fluxes = cell_fluxes + ind;
    const float *cell_data_bathySource = cell_bathySource + ind;
    const float *cell_data_edgeNormals = cell_edgeNormals + ind;
    const int *cell_data_isBoundary = cell_isBoundary + ind;
    float inc[MESH_DIM * INC_DIM];

    for(unsigned i = 0; i < MESH_DIM * INC_DIM; ++i) {
      inc[i] = 0;
    }

    MY_SIZE _point_stride0 = SOA ? point_stride[0] : 1;
    // Calling user function
    user_func_host(cellVolumes0, cellVolumes1, inc, inc + INC_DIM,
		   cell_data_fluxes, cell_data_bathySource,
		   cell_data_edgeNormals, cell_data_isBoundary,
                   cell_stride);

    // Adding increment back
    for (unsigned i = 0; i < INC_DIM; ++i) {
      point_data_out_left[i * _point_stride0] += inc[i];
      point_data_out_right[i * _point_stride0] += inc[i + INC_DIM];
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
  const float *point_cellVolumes = reinterpret_cast<const float *>(_point_data[1]);
  float *point_data_out = reinterpret_cast<float *>(_point_data_out);
  const float *cell_fluxes = reinterpret_cast<const float *>(_cell_data[0]);
  const float *cell_bathySource = reinterpret_cast<const float *>(_cell_data[1]);
  const float *cell_edgeNormals = reinterpret_cast<const float *>(_cell_data[2]);
  const int *cell_isBoundary = reinterpret_cast<const int *>(_cell_data[3]);
  float inc[MESH_DIM * INC_DIM];
#pragma unroll
  for(unsigned i = 0; i < MESH_DIM * INC_DIM; ++i) inc[i] = 0;
  if (ind < num_cells) {

    const float *__restrict__ cellVolumes0 =
        point_cellVolumes + cell_to_node[0][MESH_DIM * ind + 0];
    const float *__restrict__ cellVolumes1 =
        point_cellVolumes + cell_to_node[0][MESH_DIM * ind + 1];
    const unsigned used_point_dim = !SOA ? INC_DIM : 1;
    float *__restrict__ point_data_out_left =
        point_data_out + used_point_dim * cell_to_node[0][MESH_DIM * ind + 0];
    float *__restrict__ point_data_out_right =
        point_data_out + used_point_dim * cell_to_node[0][MESH_DIM * ind + 1];
    const float *__restrict__ cell_data_fluxes = cell_fluxes + ind;
    const float *__restrict__ cell_data_bathySource = cell_bathySource + ind;
    const float *__restrict__ cell_data_edgeNormals = cell_edgeNormals + ind;
    const int *__restrict__ cell_data_isBoundary = cell_isBoundary + ind;

    MY_SIZE _point_stride0 = SOA ? point_stride[0] : 1;
    // Calling user function
    user_func_gpu(cellVolumes0, cellVolumes1, inc, inc + INC_DIM,
		   cell_data_fluxes, cell_data_bathySource,
		   cell_data_edgeNormals, cell_data_isBoundary,
                   cell_stride);

    // Adding increment back
#pragma unroll
    for (unsigned i = 0; i < INC_DIM; ++i) {
      point_data_out_left[i * _point_stride0] += inc[i];
      point_data_out_right[i * _point_stride0] += inc[i + INC_DIM];
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
  using DataType = float; // there are different algorithms based on the type of
                          // the cached points
  const DataType *__restrict__ point_cellVolumes =
      reinterpret_cast<const DataType *>(_point_data[1]);
  DataType *__restrict__ point_data_out =
      reinterpret_cast<DataType *>(_point_data_out);
  const DataType *__restrict__ cell_fluxes =
      reinterpret_cast<const DataType *>(_cell_data[0]);
  const DataType *__restrict__ cell_bathy =
      reinterpret_cast<const DataType *>(_cell_data[1]);
  const DataType *__restrict__ cell_edgeNormals =
      reinterpret_cast<const DataType *>(_cell_data[2]);
  const int *__restrict__ cell_isB =
      reinterpret_cast<const int *>(_cell_data[3]);

  float4 *__restrict__ point_data_out_float4 =
      reinterpret_cast<float4 *>(point_data_out);

  const MY_SIZE bid = blockIdx.x;
  const MY_SIZE thread_ind = block_offsets[bid] + threadIdx.x;
  const MY_SIZE tid = threadIdx.x;

  const MY_SIZE cache_points_offset = points_to_be_cached_offsets[bid];
  const MY_SIZE num_cached_points =
      points_to_be_cached_offsets[bid + 1] - cache_points_offset;
  MY_SIZE shared_num_cached_points;
  if (32 % INC_DIM == 0) {
    // Currently, shared memory bank conflict avoidance works only if 32 is
    // divisible by PointDim
    MY_SIZE needed_offset = 32 / INC_DIM;
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

  // Computation
  DataType increment[INC_DIM * MESH_DIM];
#pragma unroll
  for(unsigned i = 0; i < MESH_DIM * INC_DIM; ++i) increment[i] = 0;
  DataType *point_data_left, *point_data_right;
  if (tid < block_size) {
    point_data_left =
        point_cache + cell_to_node[0][thread_ind + 0 * num_cells];
    point_data_right =
        point_cache + cell_to_node[0][thread_ind + 1 * num_cells];
    const DataType * edgeFluxes = cell_fluxes + thread_ind;
    const DataType * bathySource = cell_bathy + thread_ind;
    const DataType * edgeNormals = cell_edgeNormals + thread_ind;
    const int * isBoundary = cell_isB + thread_ind;
    const DataType *__restrict__ cellVolumes0 = 
	point_cellVolumes + cell_to_node[1][thread_ind + 0 * num_cells];
    const DataType *__restrict__ cellVolumes1 = 
	point_cellVolumes + cell_to_node[1][thread_ind + 1 * num_cells];
    // If more than one mapping:
    // MY_SIZE _point_stride1 = SOA ? point_stride[1] : 1;
    user_func_gpu(cellVolumes0, cellVolumes1, increment, increment + INC_DIM,
                  edgeFluxes, bathySource, edgeNormals, isBoundary,
                  cell_stride);
  }

  __syncthreads();

  // Clear cache
  for (MY_SIZE i = tid; i < shared_num_cached_points * INC_DIM;
       i += blockDim.x) {
    point_cache[i] = 0;
  }

  __syncthreads();

  // Accumulate increment to shared memory
  for (std::uint8_t cur_colour = 0; cur_colour < num_cell_colours[bid];
       ++cur_colour) {
    if (our_colour == cur_colour) {
      for (unsigned i = 0; i < INC_DIM; ++i) {
        point_data_left[i * shared_num_cached_points] +=
            increment[i + 0 * INC_DIM];
        point_data_right[i * shared_num_cached_points] +=
            increment[i + 1 * INC_DIM];
      }
    }
    __syncthreads();
  }

  // Cache out
  if (!SOA) {
    if (!SOA && INC_DIM % 4 == 0 && std::is_same<DataType, float>::value) {
      for (MY_SIZE i = tid; i < INC_DIM * num_cached_points / 4;
           i += blockDim.x) {
        MY_SIZE point_ind = i * 4 / INC_DIM;
        MY_SIZE d = (i * 4) % INC_DIM;
        MY_SIZE g_ind = index<SOA>(
            point_stride[0],
            points_to_be_cached[cache_points_offset + point_ind], INC_DIM, d);
        MY_SIZE c_ind0 =
            index<true>(shared_num_cached_points, point_ind, INC_DIM, d + 0);
        MY_SIZE c_ind1 =
            index<true>(shared_num_cached_points, point_ind, INC_DIM, d + 1);
        MY_SIZE c_ind2 =
            index<true>(shared_num_cached_points, point_ind, INC_DIM, d + 2);
        MY_SIZE c_ind3 =
            index<true>(shared_num_cached_points, point_ind, INC_DIM, d + 3);
        float4 result = point_data_out_float4[g_ind / 4];
        result.x += point_cache[c_ind0];
        result.y += point_cache[c_ind1];
        result.z += point_cache[c_ind2];
        result.w += point_cache[c_ind3];
        point_data_out_float4[g_ind / 4] = result;
      }
    }
  } else {
    for (MY_SIZE i = tid; i < num_cached_points; i += blockDim.x) {
      MY_SIZE g_point_to_be_cached =
          points_to_be_cached[cache_points_offset + i];
      DataType result[INC_DIM];
#pragma unroll
      for (MY_SIZE d = 0; d < INC_DIM; ++d) {
        MY_SIZE write_g_ind =
            index<SOA>(point_stride[0], g_point_to_be_cached, INC_DIM, d);
        MY_SIZE write_c_ind =
            index<true>(shared_num_cached_points, i, INC_DIM, d);

        result[d] = point_data_out[write_g_ind] + point_cache[write_c_ind];
      }
#pragma unroll
      for (MY_SIZE d = 0; d < INC_DIM; ++d) {
        MY_SIZE write_g_ind =
            index<SOA>(point_stride[0], g_point_to_be_cached, INC_DIM, d);
        point_data_out[write_g_ind] = result[d];
      }
    }
  }
}
}
#endif /* end of include guard: SKELETON_HPP_BTXZV4YZ */
// vim:set et sts=2 sw=2 ts=2:
