#ifndef SKELETON_HPP_BTXZV4YZ
#define SKELETON_HPP_BTXZV4YZ
#include <algorithm>

namespace skeleton {
// Sequential user function
#define USER_FUNCTION_SIGNATURE inline void user_func_host
#include "skeleton_func.hpp"

// GPU user function
#define USER_FUNCTION_SIGNATURE __device__ void user_func_gpu
#include "skeleton_func.hpp"

// Sequential kernel
struct StepSeq {
  static constexpr unsigned MESH_DIM = 1;
  static constexpr unsigned POINT_DIM = 1;
  static constexpr unsigned CELL_DIM = 1;
  template <bool SOA>
  static void call(const void *_point_data, void *_point_data_out,
                   const void *_cell_data, const MY_SIZE *cell_to_node,
                   MY_SIZE ind, unsigned point_stride, unsigned cell_stride) {
    const float *point_data = reinterpret_cast<const float *>(_point_data);
    float *point_data_out = reinterpret_cast<float *>(_point_data_out);
    const float *cell_data = reinterpret_cast<const float *>(_cell_data);

    unsigned used_point_dim = !SOA ? POINT_DIM : 1;
    const float *point_data_cur =
        point_data + used_point_dim * cell_to_node[MESH_DIM * ind + 0];
    float *point_data_out_cur =
        point_data_out + used_point_dim * cell_to_node[MESH_DIM * ind + 0];
    const float *cell_data_cur = cell_data + ind;
    float inc[MESH_DIM * POINT_DIM];

    // Calling user function
    user_func_host(point_data_cur, inc, cell_data_cur, point_stride,
                   cell_stride);

    // Adding increment back
    point_stride = SOA ? point_stride : 1;
    for (unsigned i = 0; i < PointDim; ++i) {
      point_data_out_left[i * point_stride] += inc[i];
    }
  }
};

// OMP kernel
// Should be the same as the sequential
using StepOMP = StepSeq;

// GPU global kernel
template <bool SOA>
__global__ void stepGPUGlobal(const void *_point_data, void *_point_data_out,
                              const void *_cell_data,
                              const MY_SIZE *cell_to_node, MY_SIZE num_cells,
                              MY_SIZE point_stride, MY_SIZE cell_stride);

struct StepGPUGlobal {
  static constexpr unsigned MESH_DIM = 1;
  static constexpr unsigned POINT_DIM = 1;
  static constexpr unsigned CELL_DIM = 1;
  template <bool SOA>
  static void call(const void *point_data, void *point_data_out,
                   const void *cell_data, const MY_SIZE *cell_to_node,
                   MY_SIZE num_cells, MY_SIZE point_stride, MY_SIZE cell_stride,
                   MY_SIZE num_blocks, MY_SIZE block_size) {
    // nvcc doesn't support a static method as a kernel
    stepGPUGlobal<SOA><<<num_blocks, block_size>>>(
        point_data, point_data_out, cell_data, cell_to_node, num_cells,
        point_stride, cell_stride);
  }
};

template <bool SOA>
__global__ void stepGPUGlobal(const void *_point_data, void *_point_data_out,
                              const void *_cell_data,
                              const MY_SIZE *cell_to_node, MY_SIZE num_cells,
                              MY_SIZE point_stride, MY_SIZE cell_stride) {
  MY_SIZE ind = blockIdx.x * blockDim.x + threadIdx.x;
  DataType inc[StepGPUGlobal::MESH_DIM * StepGPUGlobal::POINT_DIM];
  if (ind < num_cells) {
    const float *point_data = reinterpret_cast<const float *>(_point_data);
    float *point_data_out = reinterpret_cast<float *>(_point_data_out);
    const float *cell_data = reinterpret_cast<const float *>(_cell_data);

    unsigned used_point_dim = !SOA ? StepGPUGlobal::POINT_DIM : 1;
    const float *point_data_cur =
        point_data + used_point_dim * cell_to_node[MESH_DIM * ind + 0];
    float *point_data_out_cur =
        point_data_out + used_point_dim * cell_to_node[MESH_DIM * ind + 0];
    const float *cell_data_cur = cell_data + ind;

    // Calling user function
    user_func_gpu(point_data_cur, point_data_out_cur, cell_data_cur,
                  point_stride, cell_stride);

    // Adding back the increment
    point_stride = SOA ? point_stride : 1;
#pragma unroll
    for (unsigned i = 0; i < PointDim; ++i) {
      point_data_out_cur[i * point_stride] = inc[i];
    }
  }
}

// GPU hierarchical kernel
// TODO
}
#endif /* end of include guard: SKELETON_HPP_BTXZV4YZ */
// vim:set et sts=2 sw=2 ts=2:
