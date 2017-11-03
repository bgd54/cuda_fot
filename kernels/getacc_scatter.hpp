#ifndef GETACC_SCATTER_HPP_BTXZV4YZ
#define GETACC_SCATTER_HPP_BTXZV4YZ
#include <algorithm>

namespace getacc {
// Sequential user function
#define USER_FUNCTION_SIGNATURE inline void user_func_host
#include "getacc_scatter_func1.hpp"

// GPU user function
#define USER_FUNCTION_SIGNATURE __device__ void user_func_gpu
#define RESTRICT __restrict__
#include "getacc_scatter_func.hpp"

/**
 * SOA-AOS layouts:
 * - the layout of `point_data` is controlled by the SOA template parameter
 *   - if it's SOA, `point_stride` is the stride, otherwise that doesn't matter
 * - the layout of `cell_data` is always SOA with stride `cell_stride`
 * - the layout of `cell_to_node` is AOS except for the hierarchical case, where
 *   it is SOA
 */

constexpr unsigned MAPPING_DIM = 4;
constexpr unsigned READ_DIM = 4, INC_DIM = 4, RHO_DIM = 1;
//ndmass ndarea ndub ndvb

// Sequential kernel
struct StepSeq {

  template <bool SOA>
  static void call(const void **, void *_point_data_out,
                   void ** _cell_data, const MY_SIZE **cell_to_node, MY_SIZE ind,
                   const unsigned *point_stride, unsigned cell_stride) {
    double *point_data_out = reinterpret_cast<double *>(_point_data_out);
    const double *cell_data_cnmass = reinterpret_cast<const double *>(_cell_data[0]);
    const double *cell_data_rho = reinterpret_cast<const double *>(_cell_data[1]);
    const double *cell_data_cnwt = reinterpret_cast<const double *>(_cell_data[2]);
    const double *cell_data_cnfx = reinterpret_cast<const double *>(_cell_data[3]);
    const double *cell_data_cnfy = reinterpret_cast<const double *>(_cell_data[4]);

    const double *cur_cnmass = cell_data_cnmass + ind; 
    const double *cur_rho = cell_data_rho + ind; 
    const double *cur_cnwt = cell_data_cnwt + ind; 
    const double *cur_cnfx = cell_data_cnfx + ind; 
    const double *cur_cnfy = cell_data_cnfy + ind; 
   
    unsigned used_point_dim_out = !SOA ? INC_DIM : 1;

    int map0idx = used_point_dim_out * cell_to_node[0][MAPPING_DIM * ind + 0];
    int map1idx = used_point_dim_out * cell_to_node[0][MAPPING_DIM * ind + 1];
    int map2idx = used_point_dim_out * cell_to_node[0][MAPPING_DIM * ind + 2];
    int map3idx = used_point_dim_out * cell_to_node[0][MAPPING_DIM * ind + 3];

    double inc[MAPPING_DIM * INC_DIM];
    for(unsigned i = 0; i < MAPPING_DIM*INC_DIM; ++i)
      inc[i] = 0;

    // inc
    MY_SIZE _point_stride = SOA ? point_stride[0] : 1;

    // Calling user function
    user_func_host(cur_cnmass, cur_rho, cur_cnwt, cur_cnfx, cur_cnfy,
		   inc, inc + INC_DIM, inc + 2 * INC_DIM, inc + 3 * INC_DIM,
                   cell_stride);

    // Adding increment back
    for(unsigned i = 0; i < INC_DIM; ++i) {
       point_data_out[map0idx+_point_stride * i] += inc[i];
       point_data_out[map1idx+_point_stride * i] += inc[INC_DIM + i];
       point_data_out[map2idx+_point_stride * i] += inc[2 * INC_DIM + i];
       point_data_out[map3idx+_point_stride * i] += inc[3 * INC_DIM + i];
    }
  }
};

// OMP kernel
// Should be the same as the sequential
using StepOMP = StepSeq;

// GPU global kernel
template <bool SOA>
__global__ void
stepGPUGlobal(const void **__restrict__ ,
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
stepGPUGlobal(const void **__restrict__,
              void *__restrict__ _point_data_out, void **__restrict__ _cell_data,
              const MY_SIZE **__restrict__ cell_to_node, MY_SIZE num_cells,
              MY_SIZE *__restrict__ point_stride, MY_SIZE cell_stride) {
  MY_SIZE ind = blockIdx.x * blockDim.x + threadIdx.x;
  if (ind < num_cells) {
    double *point_data_out = reinterpret_cast<double *>(_point_data_out);
    const double *cell_data_cnmass = reinterpret_cast<const double *>(_cell_data[0]);
    const double *cell_data_rho = reinterpret_cast<const double *>(_cell_data[1]);
    const double *cell_data_cnwt = reinterpret_cast<const double *>(_cell_data[2]);
    const double *cell_data_cnfx = reinterpret_cast<const double *>(_cell_data[3]);
    const double *cell_data_cnfy = reinterpret_cast<const double *>(_cell_data[4]);

    const double *cur_cnmass = cell_data_cnmass + ind; 
    const double *cur_rho = cell_data_rho + ind; 
    const double *cur_cnwt = cell_data_cnwt + ind; 
    const double *cur_cnfx = cell_data_cnfx + ind; 
    const double *cur_cnfy = cell_data_cnfy + ind; 
 
    unsigned used_point_dim_out = !SOA ? INC_DIM : 1;

    int map0idx = used_point_dim_out * cell_to_node[0][MAPPING_DIM * ind + 0];
    int map1idx = used_point_dim_out * cell_to_node[0][MAPPING_DIM * ind + 1];
    int map2idx = used_point_dim_out * cell_to_node[0][MAPPING_DIM * ind + 2];
    int map3idx = used_point_dim_out * cell_to_node[0][MAPPING_DIM * ind + 3];

    double inc[MAPPING_DIM * INC_DIM];
    for(unsigned i = 0; i < MAPPING_DIM*INC_DIM; ++i)
      inc[i] = 0;

    // inc
    MY_SIZE _point_stride = SOA ? point_stride[0] : 1;

    // Calling user function
    user_func_gpu(cur_cnmass, cur_rho, cur_cnwt, cur_cnfx, cur_cnfy,
		   inc, inc + INC_DIM, inc + 2 * INC_DIM, inc + 3 * INC_DIM,
                   cell_stride);

    // Adding increment back
    for(unsigned i = 0; i < INC_DIM; ++i) {
       point_data_out[map0idx+_point_stride * i] += inc[i];
       point_data_out[map1idx+_point_stride * i] += inc[INC_DIM + i];
       point_data_out[map2idx+_point_stride * i] += inc[2 * INC_DIM + i];
       point_data_out[map3idx+_point_stride * i] += inc[3 * INC_DIM + i];
    }
  }
}

// GPU hierarchical kernel
template <bool SOA>
__global__ void stepGPUHierarchical(
    const void **__restrict__ , void *__restrict__ _point_data_out,
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
    const void **__restrict__ , void *__restrict__ _point_data_out,
    const MY_SIZE *__restrict__ points_to_be_cached,
    const MY_SIZE *__restrict__ points_to_be_cached_offsets,
    void **__restrict__ _cell_data,
    const MY_SIZE **__restrict__ cell_to_node,
    const std::uint8_t *__restrict__ num_cell_colours,
    const std::uint8_t *__restrict__ cell_colours,
    const MY_SIZE *__restrict__ block_offsets, MY_SIZE num_cells,
    const MY_SIZE *__restrict__ point_stride, MY_SIZE cell_stride) {

  using DataType =
      double; // there are different algorithms based on the type of
              // the cached points
  const DataType *__restrict__ cnmass =
      reinterpret_cast<const DataType *>(_cell_data[0]);
  const DataType *__restrict__ rho =
      reinterpret_cast<const DataType *>(_cell_data[1]);
  const DataType *__restrict__ cnwt =
      reinterpret_cast<const DataType *>(_cell_data[2]);
  const DataType *__restrict__ cnfx =
      reinterpret_cast<const DataType *>(_cell_data[3]);
  const DataType *__restrict__ cnfy =
      reinterpret_cast<const DataType *>(_cell_data[4]);

  DataType *__restrict__ point_data_out =
      reinterpret_cast<DataType *>(_point_data_out);
  double2 *__restrict__ point_data_out_double2 =
      reinterpret_cast<double2 *>(_point_data_out);

  const MY_SIZE bid = blockIdx.x;
  const MY_SIZE thread_ind = block_offsets[bid] + threadIdx.x;
  const MY_SIZE tid = threadIdx.x;
  const MY_SIZE cache_points_offset = points_to_be_cached_offsets[bid];
  const MY_SIZE num_cached_points =
      points_to_be_cached_offsets[bid + 1] - cache_points_offset;
  MY_SIZE shared_num_cached_points = num_cached_points;
  if (32 % INC_DIM == 0) {
    // Currently, shared memory bank conflict avoidance works only if 32 is
    // divisible by RES_DIM
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
  // ELM NEM KELL
  // Computation
  DataType inc[MAPPING_DIM*INC_DIM];
  #pragma unroll
  for(unsigned i =0; i< MAPPING_DIM*INC_DIM; ++i)
    inc[i]=0;
  int map0idx, map1idx, map2idx, map3idx;
  if (tid < block_size) {

    map0idx = cell_to_node[0][thread_ind + num_cells * 0];
    map1idx = cell_to_node[0][thread_ind + num_cells * 1];
    map2idx = cell_to_node[0][thread_ind + num_cells * 2];
    map3idx = cell_to_node[0][thread_ind + num_cells * 3];
    const double *cur_cnmass = cnmass + thread_ind; 
    const double *cur_rho = rho + thread_ind; 
    const double *cur_cnwt = cnwt + thread_ind; 
    const double *cur_cnfx = cnfx + thread_ind; 
    const double *cur_cnfy = cnfy + thread_ind; 


    user_func_gpu(cur_cnmass, cur_rho, cur_cnwt, cur_cnfx, cur_cnfy,
		   inc, inc + INC_DIM, inc + 2 * INC_DIM, inc + 3 * INC_DIM,
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

#pragma unroll
      for (unsigned i = 0; i < INC_DIM; ++i) {
       point_cache[map0idx+shared_num_cached_points * i] += inc[i];
       point_cache[map1idx+shared_num_cached_points * i] += inc[INC_DIM + i];
       point_cache[map2idx+shared_num_cached_points * i] += inc[2 * INC_DIM + i];
       point_cache[map3idx+shared_num_cached_points * i] += inc[3 * INC_DIM + i];
         
      }
    }
    __syncthreads();
  }

  // Cache out
  if (!SOA) {
    if (!SOA && INC_DIM % 2 == 0 && std::is_same<DataType, double>::value) {
      for (MY_SIZE i = tid; i < INC_DIM * num_cached_points / 2;
           i += blockDim.x) {
        MY_SIZE point_ind = i * 2 / INC_DIM;
        MY_SIZE d = (i * 2) % INC_DIM;
        MY_SIZE g_ind = index<SOA>(
            point_stride[0],
            points_to_be_cached[cache_points_offset + point_ind], INC_DIM, d);
        MY_SIZE c_ind0 =
            index<true>(shared_num_cached_points, point_ind, INC_DIM, d + 0);
        MY_SIZE c_ind1 =
            index<true>(shared_num_cached_points, point_ind, INC_DIM, d + 1);
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
#endif /* end of include guard: GETACC_SCATTER_HPP_BTXZV4YZ */
// vim:set et sts=2 sw=2 ts=2:
