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
                   void **_cell_data, const MY_SIZE **cell_to_node,
                   MY_SIZE ind, const unsigned *point_stride,
                   unsigned cell_stride) {
    const double *xyz_data = reinterpret_cast<const double *>(_point_data[1]);
    double *f_data_out = reinterpret_cast<double *>(_point_data_out);
    const double *sig_data = reinterpret_cast<const double *>(_cell_data[0]);
    double *determ_data = reinterpret_cast<double *>(_cell_data[1]);

    unsigned used_point_dim = !SOA ? POINT_DIM : 1;
    double *f_data_out_cur[MESH_DIM];
    const double *xyz_data_cur[MESH_DIM];
    for (unsigned i = 0; i < MESH_DIM; ++i) {
      f_data_out_cur[i] = f_data_out +
        used_point_dim * cell_to_node[0][MESH_DIM * ind + i];
      xyz_data_cur[i] = xyz_data +
        used_point_dim * cell_to_node[0][MESH_DIM * ind + i];
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
  if (ind < num_cells) {
    const double *xyz_data = reinterpret_cast<const double *>(_point_data[1]);
    double *f_data_out = reinterpret_cast<double *>(_point_data_out);
    const double *sig_data = reinterpret_cast<const double *>(_cell_data[0]);
    double *determ_data = reinterpret_cast<double *>(_cell_data[1]);

    unsigned used_point_dim = !SOA ? POINT_DIM : 1;
    double *f_data_out_cur[MESH_DIM];
    const double *xyz_data_cur[MESH_DIM];
    for (unsigned i = 0; i < MESH_DIM; ++i) {
      f_data_out_cur[i] = f_data_out +
        used_point_dim * cell_to_node[0][MESH_DIM * ind + i];
      xyz_data_cur[i] = xyz_data +
        used_point_dim * cell_to_node[0][MESH_DIM * ind + i];
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

}

#endif /* end of include guard: LULESH_HPP_RVFK7I6X */
// vim:set et sts=2 sw=2 ts=2:
