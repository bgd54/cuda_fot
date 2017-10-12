#ifndef MINE_HPP_IWZYHXFG
#define MINE_HPP_IWZYHXFG

#include <algorithm>
#include <cassert>

namespace mine {

// Sequential user function
#define USER_FUNCTION_SIGNATURE(fname) inline void fname##_host
#include "mine_func.hpp"

// GPU user function
#define USER_FUNCTION_SIGNATURE(fname) __device__ void fname##_gpu
#include "mine_func.hpp"

// Sequential kernel
template <unsigned PointDim, unsigned CellDim, class DataType> struct StepSeq2 {
  static constexpr unsigned MESH_DIM = 2;
  template <bool SOA>
  static void call(const void *_point_data, void *_point_data_out,
                   const void *_cell_data, const MY_SIZE *cell_to_node,
                   MY_SIZE ind, unsigned point_stride, unsigned cell_stride) {
    assert(MESH_DIM_MACRO == 2);
    const DataType *point_data =
        reinterpret_cast<const DataType *>(_point_data);
    DataType *point_data_out = reinterpret_cast<DataType *>(_point_data_out);
    const DataType *cell_data = reinterpret_cast<const DataType *>(_cell_data);

    unsigned used_point_dim = !SOA ? PointDim : 1;
    const DataType *point_data_left =
        point_data + used_point_dim * cell_to_node[MESH_DIM * ind + 0];
    const DataType *point_data_right =
        point_data + used_point_dim * cell_to_node[MESH_DIM * ind + 1];
    DataType *point_data_out_left =
        point_data_out + used_point_dim * cell_to_node[MESH_DIM * ind + 0];
    DataType *point_data_out_right =
        point_data_out + used_point_dim * cell_to_node[MESH_DIM * ind + 1];
    const DataType *cell_data_cur = cell_data + ind;
    DataType inc[MESH_DIM * PointDim];

    // Calling user function
    mine_func_host<PointDim, CellDim, DataType, SOA>(
        point_data_left, point_data_right, inc, inc + PointDim, cell_data_cur,
        point_stride, cell_stride);

    // Adding increment back
    point_stride = SOA ? point_stride : 1;
    for (unsigned i = 0; i < PointDim; ++i) {
      point_data_out_left[i * point_stride] += inc[i];
      point_data_out_right[i * point_stride] += inc[i + PointDim];
    }
  }
};

template <unsigned PointDim, unsigned CellDim, class DataType> struct StepSeq4 {
  static constexpr unsigned MESH_DIM = 4;
  template <bool SOA>
  static void call(const void *_point_data, void *_point_data_out,
                   const void *_cell_data, const MY_SIZE *cell_to_node,
                   MY_SIZE ind, unsigned point_stride, unsigned cell_stride) {
    assert(MESH_DIM_MACRO == 4);
    const DataType *point_data =
        reinterpret_cast<const DataType *>(_point_data);
    DataType *point_data_out = reinterpret_cast<DataType *>(_point_data_out);
    const DataType *cell_data = reinterpret_cast<const DataType *>(_cell_data);

    unsigned used_point_dim = !SOA ? PointDim : 1;
    const DataType *point_data0 =
        point_data + used_point_dim * cell_to_node[MESH_DIM * ind + 0];
    const DataType *point_data1 =
        point_data + used_point_dim * cell_to_node[MESH_DIM * ind + 1];
    const DataType *point_data2 =
        point_data + used_point_dim * cell_to_node[MESH_DIM * ind + 2];
    const DataType *point_data3 =
        point_data + used_point_dim * cell_to_node[MESH_DIM * ind + 3];
    DataType *point_data_out0 =
        point_data_out + used_point_dim * cell_to_node[MESH_DIM * ind + 0];
    DataType *point_data_out1 =
        point_data_out + used_point_dim * cell_to_node[MESH_DIM * ind + 1];
    DataType *point_data_out2 =
        point_data_out + used_point_dim * cell_to_node[MESH_DIM * ind + 2];
    DataType *point_data_out3 =
        point_data_out + used_point_dim * cell_to_node[MESH_DIM * ind + 3];
    const DataType *cell_data_cur = cell_data + ind;
    DataType inc[PointDim * MESH_DIM];

    // Calling user function
    mine_func_host<PointDim, CellDim, DataType, SOA>(
        point_data0, point_data1, point_data2, point_data3, inc, inc + PointDim,
        inc + 2 * PointDim, inc + 3 * PointDim, cell_data_cur, point_stride,
        cell_stride);

    // Adding increment back
    point_stride = SOA ? point_stride : 1;
    for (unsigned i = 0; i < PointDim; ++i) {
      point_data_out0[i * point_stride] += inc[i + 0 * PointDim];
      point_data_out1[i * point_stride] += inc[i + 1 * PointDim];
      point_data_out2[i * point_stride] += inc[i + 2 * PointDim];
      point_data_out3[i * point_stride] += inc[i + 3 * PointDim];
    }
  }
};

// OMP kernel
// Should be the same as the sequential
template <unsigned PointDim, unsigned CellDim, class DataType>
using StepOMP2 = StepSeq2<PointDim, CellDim, DataType>;
template <unsigned PointDim, unsigned CellDim, class DataType>
using StepOMP4 = StepSeq4<PointDim, CellDim, DataType>;

// GPU global kernel
template <unsigned PointDim, unsigned CellDim, class DataType, bool SOA>
__global__ void stepGPUGlobal(const void *_point_data, void *_point_data_out,
                              const void *_cell_data,
                              const MY_SIZE *cell_to_node, MY_SIZE num_cells,
                              unsigned point_stride, unsigned cell_stride);

template <unsigned PointDim, unsigned CellDim, class DataType>
struct StepGPUGlobal2 {
  static constexpr unsigned MESH_DIM = 2;
  template <bool SOA>
  static void
  call(const void *point_data, void *point_data_out, const void *cell_data,
       const MY_SIZE *cell_to_node, MY_SIZE num_cells, unsigned point_stride,
       unsigned cell_stride, unsigned num_blocks, unsigned block_size) {
    // nvcc doesn't support a static method as a kernel
    assert(MESH_DIM_MACRO == 2);
    stepGPUGlobal<PointDim, CellDim, DataType, SOA><<<num_blocks, block_size>>>(
        point_data, point_data_out, cell_data, cell_to_node, num_cells,
        point_stride, cell_stride);
  }
};

template <unsigned PointDim, unsigned CellDim, class DataType, bool SOA>
__global__ void stepGPUGlobal(const void *_point_data, void *_point_data_out,
                              const void *_cell_data,
                              const MY_SIZE *cell_to_node, MY_SIZE num_cells,
                              unsigned point_stride, unsigned cell_stride) {
  MY_SIZE ind = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr unsigned MESH_DIM =
      StepGPUGlobal2<PointDim, CellDim, DataType>::MESH_DIM;
  DataType inc[MESH_DIM * PointDim];
  if (ind < num_cells) {
    const DataType *point_data =
        reinterpret_cast<const DataType *>(_point_data);
    DataType *point_data_out = reinterpret_cast<DataType *>(_point_data_out);
    const DataType *cell_data = reinterpret_cast<const DataType *>(_cell_data);

    unsigned used_point_dim = !SOA ? PointDim : 1;
    const DataType *point_data_left =
        point_data + used_point_dim * cell_to_node[MESH_DIM * ind + 0];
    const DataType *point_data_right =
        point_data + used_point_dim * cell_to_node[MESH_DIM * ind + 1];
    DataType *point_data_out_left =
        point_data_out + used_point_dim * cell_to_node[MESH_DIM * ind + 0];
    DataType *point_data_out_right =
        point_data_out + used_point_dim * cell_to_node[MESH_DIM * ind + 1];
    const DataType *cell_data_cur = cell_data + ind;

    // Calling user function
    mine_func_gpu<PointDim, CellDim, DataType, SOA>(
        point_data_left, point_data_right, inc, inc + PointDim, cell_data_cur,
        point_stride, cell_stride);

    // Add back the increment
    point_stride = SOA ? point_stride : 1;
#pragma unroll
    for (unsigned i = 0; i < PointDim; ++i) {
      point_data_out_left[i * point_stride] += inc[i];
      point_data_out_right[i * point_stride] += inc[i + PointDim];
    }
  }
}

template <unsigned PointDim, unsigned CellDim, class DataType, bool SOA>
__global__ void stepGPUGlobal4(const void *_point_data, void *_point_data_out,
                               const void *_cell_data,
                               const MY_SIZE *cell_to_node, MY_SIZE num_cells,
                               unsigned point_stride, unsigned cell_stride);

template <unsigned PointDim, unsigned CellDim, class DataType>
struct StepGPUGlobal4 {
  static constexpr unsigned MESH_DIM = 4;
  template <bool SOA>
  static void
  call(const void *point_data, void *point_data_out, const void *cell_data,
       const MY_SIZE *cell_to_node, MY_SIZE num_cells, unsigned point_stride,
       unsigned cell_stride, unsigned num_blocks, unsigned block_size) {
    // nvcc doesn't support a static method as a kernel
    assert(MESH_DIM_MACRO == 4);
    stepGPUGlobal4<PointDim, CellDim, DataType, SOA>
        <<<num_blocks, block_size>>>(point_data, point_data_out, cell_data,
                                     cell_to_node, num_cells, point_stride,
                                     cell_stride);
  }
};

template <unsigned PointDim, unsigned CellDim, class DataType, bool SOA>
__global__ void stepGPUGlobal4(const void *_point_data, void *_point_data_out,
                               const void *_cell_data,
                               const MY_SIZE *cell_to_node, MY_SIZE num_cells,
                               unsigned point_stride, unsigned cell_stride) {
  MY_SIZE ind = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr unsigned MESH_DIM =
      StepGPUGlobal4<PointDim, CellDim, DataType>::MESH_DIM;
  DataType inc[MESH_DIM * PointDim];
  if (ind < num_cells) {
    const DataType *point_data =
        reinterpret_cast<const DataType *>(_point_data);
    DataType *point_data_out = reinterpret_cast<DataType *>(_point_data_out);
    const DataType *cell_data = reinterpret_cast<const DataType *>(_cell_data);

    unsigned used_point_dim = !SOA ? PointDim : 1;
    const DataType *point_data0 =
        point_data + used_point_dim * cell_to_node[MESH_DIM * ind + 0];
    const DataType *point_data1 =
        point_data + used_point_dim * cell_to_node[MESH_DIM * ind + 1];
    const DataType *point_data2 =
        point_data + used_point_dim * cell_to_node[MESH_DIM * ind + 2];
    const DataType *point_data3 =
        point_data + used_point_dim * cell_to_node[MESH_DIM * ind + 3];
    DataType *point_data_out0 =
        point_data_out + used_point_dim * cell_to_node[MESH_DIM * ind + 0];
    DataType *point_data_out1 =
        point_data_out + used_point_dim * cell_to_node[MESH_DIM * ind + 1];
    DataType *point_data_out2 =
        point_data_out + used_point_dim * cell_to_node[MESH_DIM * ind + 2];
    DataType *point_data_out3 =
        point_data_out + used_point_dim * cell_to_node[MESH_DIM * ind + 3];
    const DataType *cell_data_cur = cell_data + ind;

    // Calling user function
    mine_func_gpu<PointDim, CellDim, DataType, SOA>(
        point_data0, point_data1, point_data2, point_data3, inc, inc + PointDim,
        inc + 2 * PointDim, inc + 3 * PointDim, cell_data_cur, point_stride,
        cell_stride);

    // Add back the increment
    point_stride = SOA ? point_stride : 1;
#pragma unroll
    for (unsigned i = 0; i < PointDim; ++i) {
      point_data_out0[i * point_stride] += inc[i];
      point_data_out1[i * point_stride] += inc[i + 1 * PointDim];
      point_data_out2[i * point_stride] += inc[i + 2 * PointDim];
      point_data_out3[i * point_stride] += inc[i + 3 * PointDim];
    }
  }
}

// GPU hierarchical kernel
// TODO
}

// vim:set et sts=2 sw=2 ts=2:
#endif /* end of include guard: MINE_HPP_IWZYHXFG */
