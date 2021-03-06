#ifndef MINE_HPP_IWZYHXFG
#define MINE_HPP_IWZYHXFG

#include <algorithm>
#include <cassert>

namespace mine {

// Sequential user function
#define RESTRICT
#define USER_FUNCTION_SIGNATURE(fname) inline void fname##_host
#include "mine_func.hpp"

// GPU user function
#define RESTRICT __restrict__
#define USER_FUNCTION_SIGNATURE(fname) __device__ void fname##_gpu
#include "mine_func.hpp"

// Sequential kernel
template <unsigned MeshDim, unsigned PointDim, unsigned CellDim, class DataType>
struct StepSeq;
template <unsigned PointDim, unsigned CellDim, class DataType>
struct StepSeq<2, PointDim, CellDim, DataType> {
  static constexpr unsigned MESH_DIM = 2;
  template <bool SOA>
  static void call(const void **_point_data, void *_point_data_out,
                   void **_cell_data, const MY_SIZE **cell_to_node, MY_SIZE ind,
                   const unsigned *point_stride, unsigned cell_stride) {
    const DataType *point_data =
        reinterpret_cast<const DataType *>(*_point_data);
    DataType *point_data_out = reinterpret_cast<DataType *>(_point_data_out);
    const DataType *cell_data = reinterpret_cast<const DataType *>(*_cell_data);

    unsigned used_point_dim = !SOA ? PointDim : 1;
    const DataType *point_data_left =
        point_data + used_point_dim * cell_to_node[0][MESH_DIM * ind + 0];
    const DataType *point_data_right =
        point_data + used_point_dim * cell_to_node[0][MESH_DIM * ind + 1];
    DataType *point_data_out_left =
        point_data_out + used_point_dim * cell_to_node[0][MESH_DIM * ind + 0];
    DataType *point_data_out_right =
        point_data_out + used_point_dim * cell_to_node[0][MESH_DIM * ind + 1];
    const DataType *cell_data_cur = cell_data + ind;
    DataType inc[MESH_DIM * PointDim];

    // Calling user function
    mine_func_host<PointDim, CellDim, DataType, SOA>(
        point_data_left, point_data_right, inc, inc + PointDim, cell_data_cur,
        point_stride[0], cell_stride);

    // Adding increment back
    unsigned _point_stride = SOA ? point_stride[0] : 1;
    for (unsigned i = 0; i < PointDim; ++i) {
      point_data_out_left[i * _point_stride] += inc[i];
      point_data_out_right[i * _point_stride] += inc[i + PointDim];
    }
  }
};

template <unsigned PointDim, unsigned CellDim, class DataType>
struct StepSeq<4, PointDim, CellDim, DataType> {
  static constexpr unsigned MESH_DIM = 4;
  template <bool SOA>
  static void call(const void **_point_data, void *_point_data_out,
                   void **_cell_data, const MY_SIZE **cell_to_node, MY_SIZE ind,
                   const unsigned *point_stride, unsigned cell_stride) {
    const DataType *point_data =
        reinterpret_cast<const DataType *>(*_point_data);
    DataType *point_data_out = reinterpret_cast<DataType *>(_point_data_out);
    const DataType *cell_data = reinterpret_cast<const DataType *>(*_cell_data);

    unsigned used_point_dim = !SOA ? PointDim : 1;
    const DataType *point_data0 =
        point_data + used_point_dim * cell_to_node[0][MESH_DIM * ind + 0];
    const DataType *point_data1 =
        point_data + used_point_dim * cell_to_node[0][MESH_DIM * ind + 1];
    const DataType *point_data2 =
        point_data + used_point_dim * cell_to_node[0][MESH_DIM * ind + 2];
    const DataType *point_data3 =
        point_data + used_point_dim * cell_to_node[0][MESH_DIM * ind + 3];
    DataType *point_data_out0 =
        point_data_out + used_point_dim * cell_to_node[0][MESH_DIM * ind + 0];
    DataType *point_data_out1 =
        point_data_out + used_point_dim * cell_to_node[0][MESH_DIM * ind + 1];
    DataType *point_data_out2 =
        point_data_out + used_point_dim * cell_to_node[0][MESH_DIM * ind + 2];
    DataType *point_data_out3 =
        point_data_out + used_point_dim * cell_to_node[0][MESH_DIM * ind + 3];
    const DataType *cell_data_cur = cell_data + ind;
    DataType inc[PointDim * MESH_DIM];

    // Calling user function
    mine_func_host<PointDim, CellDim, DataType, SOA>(
        point_data0, point_data1, point_data2, point_data3, inc, inc + PointDim,
        inc + 2 * PointDim, inc + 3 * PointDim, cell_data_cur, point_stride[0],
        cell_stride);

    // Adding increment back
    unsigned _point_stride = SOA ? point_stride[0] : 1;
    for (unsigned i = 0; i < PointDim; ++i) {
      point_data_out0[i * _point_stride] += inc[i + 0 * PointDim];
      point_data_out1[i * _point_stride] += inc[i + 1 * PointDim];
      point_data_out2[i * _point_stride] += inc[i + 2 * PointDim];
      point_data_out3[i * _point_stride] += inc[i + 3 * PointDim];
    }
  }
};

template <unsigned PointDim, unsigned CellDim, class DataType>
struct StepSeq<8, PointDim, CellDim, DataType> {
  static constexpr unsigned MESH_DIM = 8;
  template <bool SOA>
  static void call(const void **_point_data, void *_point_data_out,
                   void **_cell_data, const MY_SIZE **cell_to_node, MY_SIZE ind,
                   const unsigned *point_stride, unsigned cell_stride) {
    const DataType *point_data =
        reinterpret_cast<const DataType *>(*_point_data);
    DataType *point_data_out = reinterpret_cast<DataType *>(_point_data_out);
    const DataType *cell_data = reinterpret_cast<const DataType *>(*_cell_data);

    unsigned used_point_dim = !SOA ? PointDim : 1;
    const DataType *point_data0 =
        point_data + used_point_dim * cell_to_node[0][MESH_DIM * ind + 0];
    const DataType *point_data1 =
        point_data + used_point_dim * cell_to_node[0][MESH_DIM * ind + 1];
    const DataType *point_data2 =
        point_data + used_point_dim * cell_to_node[0][MESH_DIM * ind + 2];
    const DataType *point_data3 =
        point_data + used_point_dim * cell_to_node[0][MESH_DIM * ind + 3];
    const DataType *point_data4 =
        point_data + used_point_dim * cell_to_node[0][MESH_DIM * ind + 4];
    const DataType *point_data5 =
        point_data + used_point_dim * cell_to_node[0][MESH_DIM * ind + 5];
    const DataType *point_data6 =
        point_data + used_point_dim * cell_to_node[0][MESH_DIM * ind + 6];
    const DataType *point_data7 =
        point_data + used_point_dim * cell_to_node[0][MESH_DIM * ind + 7];
    DataType *point_data_out0 =
        point_data_out + used_point_dim * cell_to_node[0][MESH_DIM * ind + 0];
    DataType *point_data_out1 =
        point_data_out + used_point_dim * cell_to_node[0][MESH_DIM * ind + 1];
    DataType *point_data_out2 =
        point_data_out + used_point_dim * cell_to_node[0][MESH_DIM * ind + 2];
    DataType *point_data_out3 =
        point_data_out + used_point_dim * cell_to_node[0][MESH_DIM * ind + 3];
    DataType *point_data_out4 =
        point_data_out + used_point_dim * cell_to_node[0][MESH_DIM * ind + 4];
    DataType *point_data_out5 =
        point_data_out + used_point_dim * cell_to_node[0][MESH_DIM * ind + 5];
    DataType *point_data_out6 =
        point_data_out + used_point_dim * cell_to_node[0][MESH_DIM * ind + 6];
    DataType *point_data_out7 =
        point_data_out + used_point_dim * cell_to_node[0][MESH_DIM * ind + 7];
    const DataType *cell_data_cur = cell_data + ind;
    DataType inc[PointDim * MESH_DIM];

    // Calling user function
    mine_func_host<PointDim, CellDim, DataType, SOA>(
        point_data0, point_data1, point_data2, point_data3, point_data4,
        point_data5, point_data6, point_data7, inc, inc + PointDim,
        inc + 2 * PointDim, inc + 3 * PointDim, inc + 4 * PointDim,
        inc + 5 * PointDim, inc + 6 * PointDim, inc + 7 * PointDim,
        cell_data_cur, point_stride[0], cell_stride);

    // Adding increment back
    unsigned _point_stride = SOA ? point_stride[0] : 1;
    for (unsigned i = 0; i < PointDim; ++i) {
      point_data_out0[i * _point_stride] += inc[i + 0 * PointDim];
      point_data_out1[i * _point_stride] += inc[i + 1 * PointDim];
      point_data_out2[i * _point_stride] += inc[i + 2 * PointDim];
      point_data_out3[i * _point_stride] += inc[i + 3 * PointDim];
      point_data_out4[i * _point_stride] += inc[i + 4 * PointDim];
      point_data_out5[i * _point_stride] += inc[i + 5 * PointDim];
      point_data_out6[i * _point_stride] += inc[i + 6 * PointDim];
      point_data_out7[i * _point_stride] += inc[i + 7 * PointDim];
    }
  }
};

// OMP kernel
// Should be the same as the sequential
template <unsigned MeshDim, unsigned PointDim, unsigned CellDim, class DataType>
using StepOMP = StepSeq<MeshDim, PointDim, CellDim, DataType>;

// GPU global kernel
template <unsigned PointDim, unsigned CellDim, class DataType, bool SOA>
__global__ void
stepGPUGlobal2(const void **__restrict__ _point_data,
               void *__restrict__ _point_data_out,
               void **__restrict__ _cell_data,
               const MY_SIZE **__restrict__ cell_to_node, MY_SIZE num_cells,
               const unsigned *__restrict__ point_stride, unsigned cell_stride);

template <unsigned MeshDim, unsigned PointDim, unsigned CellDim, class DataType>
struct StepGPUGlobal;

template <unsigned PointDim, unsigned CellDim, class DataType>
struct StepGPUGlobal<2, PointDim, CellDim, DataType> {
  static constexpr unsigned MESH_DIM = 2;
  template <bool SOA>
  static void
  call(const void **__restrict__ point_data, void *__restrict__ point_data_out,
       void **__restrict__ cell_data, const MY_SIZE **__restrict__ cell_to_node,
       MY_SIZE num_cells, const unsigned *__restrict__ point_stride,
       unsigned cell_stride, unsigned num_blocks, unsigned block_size) {
    // nvcc doesn't support a static method as a kernel
    stepGPUGlobal2<PointDim, CellDim, DataType,
                   SOA><<<num_blocks, block_size>>>(
        point_data, point_data_out, cell_data, cell_to_node, num_cells,
        point_stride, cell_stride);
  }
};

template <unsigned PointDim, unsigned CellDim, class DataType, bool SOA>
__global__ void stepGPUGlobal2(const void **__restrict__ _point_data,
                               void *__restrict__ _point_data_out,
                               void **__restrict__ _cell_data,
                               const MY_SIZE **__restrict__ cell_to_node,
                               MY_SIZE num_cells,
                               const unsigned *__restrict__ point_stride,
                               unsigned cell_stride) {
  MY_SIZE ind = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr unsigned MESH_DIM =
      StepGPUGlobal<2, PointDim, CellDim, DataType>::MESH_DIM;
  DataType inc[MESH_DIM * PointDim];
  if (ind < num_cells) {
    const DataType *__restrict__ point_data =
        reinterpret_cast<const DataType *>(*_point_data);
    DataType *__restrict__ point_data_out =
        reinterpret_cast<DataType *>(_point_data_out);
    const DataType *__restrict__ cell_data =
        reinterpret_cast<const DataType *>(*_cell_data);

    unsigned used_point_dim = !SOA ? PointDim : 1;
    const DataType *point_data_left =
        point_data + used_point_dim * cell_to_node[0][MESH_DIM * ind + 0];
    const DataType *point_data_right =
        point_data + used_point_dim * cell_to_node[0][MESH_DIM * ind + 1];
    DataType *point_data_out_left =
        point_data_out + used_point_dim * cell_to_node[0][MESH_DIM * ind + 0];
    DataType *point_data_out_right =
        point_data_out + used_point_dim * cell_to_node[0][MESH_DIM * ind + 1];
    const DataType *cell_data_cur = cell_data + ind;

    // Calling user function
    mine_func_gpu<PointDim, CellDim, DataType, SOA>(
        point_data_left, point_data_right, inc, inc + PointDim, cell_data_cur,
        point_stride[0], cell_stride);

    // Add back the increment
    unsigned _point_stride = SOA ? point_stride[0] : 1;
#pragma unroll
    for (unsigned i = 0; i < PointDim; ++i) {
      point_data_out_left[i * _point_stride] += inc[i];
      point_data_out_right[i * _point_stride] += inc[i + PointDim];
    }
  }
}

template <unsigned PointDim, unsigned CellDim, class DataType, bool SOA>
__global__ void
stepGPUGlobal4(const void **__restrict__ _point_data,
               void *__restrict__ _point_data_out,
               void **__restrict__ _cell_data,
               const MY_SIZE **__restrict__ cell_to_node, MY_SIZE num_cells,
               const unsigned *__restrict__ point_stride, unsigned cell_stride);

template <unsigned PointDim, unsigned CellDim, class DataType>
struct StepGPUGlobal<4, PointDim, CellDim, DataType> {
  static constexpr unsigned MESH_DIM = 4;
  template <bool SOA>
  static void
  call(const void **__restrict__ point_data, void *__restrict__ point_data_out,
       void **__restrict__ cell_data, const MY_SIZE **__restrict__ cell_to_node,
       MY_SIZE num_cells, const unsigned *__restrict__ point_stride,
       unsigned cell_stride, unsigned num_blocks, unsigned block_size) {
    // nvcc doesn't support a static method as a kernel
    stepGPUGlobal4<PointDim, CellDim, DataType,
                   SOA><<<num_blocks, block_size>>>(
        point_data, point_data_out, cell_data, cell_to_node, num_cells,
        point_stride, cell_stride);
  }
};

template <unsigned PointDim, unsigned CellDim, class DataType, bool SOA>
__global__ void stepGPUGlobal4(const void **__restrict__ _point_data,
                               void *__restrict__ _point_data_out,
                               void **__restrict__ _cell_data,
                               const MY_SIZE **__restrict__ cell_to_node,
                               MY_SIZE num_cells,
                               const unsigned *__restrict__ point_stride,
                               unsigned cell_stride) {
  MY_SIZE ind = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr unsigned MESH_DIM =
      StepGPUGlobal<4, PointDim, CellDim, DataType>::MESH_DIM;
  DataType inc[MESH_DIM * PointDim];
  if (ind < num_cells) {
    const DataType *__restrict__ point_data =
        reinterpret_cast<const DataType *>(*_point_data);
    DataType *__restrict__ point_data_out =
        reinterpret_cast<DataType *>(_point_data_out);
    const DataType *__restrict__ cell_data =
        reinterpret_cast<const DataType *>(*_cell_data);

    unsigned used_point_dim = !SOA ? PointDim : 1;
    const DataType *point_data0 =
        point_data + used_point_dim * cell_to_node[0][MESH_DIM * ind + 0];
    const DataType *point_data1 =
        point_data + used_point_dim * cell_to_node[0][MESH_DIM * ind + 1];
    const DataType *point_data2 =
        point_data + used_point_dim * cell_to_node[0][MESH_DIM * ind + 2];
    const DataType *point_data3 =
        point_data + used_point_dim * cell_to_node[0][MESH_DIM * ind + 3];
    DataType *point_data_out0 =
        point_data_out + used_point_dim * cell_to_node[0][MESH_DIM * ind + 0];
    DataType *point_data_out1 =
        point_data_out + used_point_dim * cell_to_node[0][MESH_DIM * ind + 1];
    DataType *point_data_out2 =
        point_data_out + used_point_dim * cell_to_node[0][MESH_DIM * ind + 2];
    DataType *point_data_out3 =
        point_data_out + used_point_dim * cell_to_node[0][MESH_DIM * ind + 3];
    const DataType *cell_data_cur = cell_data + ind;

    // Calling user function
    mine_func_gpu<PointDim, CellDim, DataType, SOA>(
        point_data0, point_data1, point_data2, point_data3, inc, inc + PointDim,
        inc + 2 * PointDim, inc + 3 * PointDim, cell_data_cur, point_stride[0],
        cell_stride);

    // Add back the increment
    unsigned _point_stride = SOA ? point_stride[0] : 1;
#pragma unroll
    for (unsigned i = 0; i < PointDim; ++i) {
      point_data_out0[i * _point_stride] += inc[i];
      point_data_out1[i * _point_stride] += inc[i + 1 * PointDim];
      point_data_out2[i * _point_stride] += inc[i + 2 * PointDim];
      point_data_out3[i * _point_stride] += inc[i + 3 * PointDim];
    }
  }
}

template <unsigned PointDim, unsigned CellDim, class DataType, bool SOA>
__global__ void
stepGPUGlobal8(const void **__restrict__ _point_data,
               void *__restrict__ _point_data_out,
               void **__restrict__ _cell_data,
               const MY_SIZE **__restrict__ cell_to_node, MY_SIZE num_cells,
               const unsigned *__restrict__ point_stride, unsigned cell_stride);

template <unsigned PointDim, unsigned CellDim, class DataType>
struct StepGPUGlobal<8, PointDim, CellDim, DataType> {
  static constexpr unsigned MESH_DIM = 8;
  template <bool SOA>
  static void
  call(const void **__restrict__ point_data, void *__restrict__ point_data_out,
       void **__restrict__ cell_data, const MY_SIZE **__restrict__ cell_to_node,
       MY_SIZE num_cells, const unsigned *__restrict__ point_stride,
       unsigned cell_stride, unsigned num_blocks, unsigned block_size) {
    // nvcc doesn't support a static method as a kernel
    stepGPUGlobal8<PointDim, CellDim, DataType,
                   SOA><<<num_blocks, block_size>>>(
        point_data, point_data_out, cell_data, cell_to_node, num_cells,
        point_stride, cell_stride);
  }
};

template <unsigned PointDim, unsigned CellDim, class DataType, bool SOA>
__global__ void stepGPUGlobal8(const void **__restrict__ _point_data,
                               void *__restrict__ _point_data_out,
                               void **__restrict__ _cell_data,
                               const MY_SIZE **__restrict__ cell_to_node,
                               MY_SIZE num_cells,
                               const unsigned *__restrict__ point_stride,
                               unsigned cell_stride) {
  MY_SIZE ind = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr unsigned MESH_DIM =
      StepGPUGlobal<8, PointDim, CellDim, DataType>::MESH_DIM;
  DataType inc[MESH_DIM * PointDim];
  if (ind < num_cells) {
    const DataType *__restrict__ point_data =
        reinterpret_cast<const DataType *>(*_point_data);
    DataType *__restrict__ point_data_out =
        reinterpret_cast<DataType *>(_point_data_out);
    const DataType *__restrict__ cell_data =
        reinterpret_cast<const DataType *>(*_cell_data);

    unsigned used_point_dim = !SOA ? PointDim : 1;
    const DataType *point_data0 =
        point_data + used_point_dim * cell_to_node[0][MESH_DIM * ind + 0];
    const DataType *point_data1 =
        point_data + used_point_dim * cell_to_node[0][MESH_DIM * ind + 1];
    const DataType *point_data2 =
        point_data + used_point_dim * cell_to_node[0][MESH_DIM * ind + 2];
    const DataType *point_data3 =
        point_data + used_point_dim * cell_to_node[0][MESH_DIM * ind + 3];
    const DataType *point_data4 =
        point_data + used_point_dim * cell_to_node[0][MESH_DIM * ind + 4];
    const DataType *point_data5 =
        point_data + used_point_dim * cell_to_node[0][MESH_DIM * ind + 5];
    const DataType *point_data6 =
        point_data + used_point_dim * cell_to_node[0][MESH_DIM * ind + 6];
    const DataType *point_data7 =
        point_data + used_point_dim * cell_to_node[0][MESH_DIM * ind + 7];
    DataType *point_data_out0 =
        point_data_out + used_point_dim * cell_to_node[0][MESH_DIM * ind + 0];
    DataType *point_data_out1 =
        point_data_out + used_point_dim * cell_to_node[0][MESH_DIM * ind + 1];
    DataType *point_data_out2 =
        point_data_out + used_point_dim * cell_to_node[0][MESH_DIM * ind + 2];
    DataType *point_data_out3 =
        point_data_out + used_point_dim * cell_to_node[0][MESH_DIM * ind + 3];
    DataType *point_data_out4 =
        point_data_out + used_point_dim * cell_to_node[0][MESH_DIM * ind + 4];
    DataType *point_data_out5 =
        point_data_out + used_point_dim * cell_to_node[0][MESH_DIM * ind + 5];
    DataType *point_data_out6 =
        point_data_out + used_point_dim * cell_to_node[0][MESH_DIM * ind + 6];
    DataType *point_data_out7 =
        point_data_out + used_point_dim * cell_to_node[0][MESH_DIM * ind + 7];
    const DataType *cell_data_cur = cell_data + ind;

    // Calling user function
    mine_func_gpu<PointDim, CellDim, DataType, SOA>(
        point_data0, point_data1, point_data2, point_data3, point_data4,
        point_data5, point_data6, point_data7, inc, inc + PointDim,
        inc + 2 * PointDim, inc + 3 * PointDim, inc + 4 * PointDim,
        inc + 5 * PointDim, inc + 6 * PointDim, inc + 7 * PointDim,
        cell_data_cur, point_stride[0], cell_stride);

    // Add back the increment
    unsigned _point_stride = SOA ? point_stride[0] : 1;
#pragma unroll
    for (unsigned i = 0; i < PointDim; ++i) {
      point_data_out0[i * _point_stride] += inc[i];
      point_data_out1[i * _point_stride] += inc[i + 1 * PointDim];
      point_data_out2[i * _point_stride] += inc[i + 2 * PointDim];
      point_data_out3[i * _point_stride] += inc[i + 3 * PointDim];
      point_data_out4[i * _point_stride] += inc[i + 4 * PointDim];
      point_data_out5[i * _point_stride] += inc[i + 5 * PointDim];
      point_data_out6[i * _point_stride] += inc[i + 6 * PointDim];
      point_data_out7[i * _point_stride] += inc[i + 7 * PointDim];
    }
  }
}

// GPU hierarchical kernel
template <unsigned PointDim, unsigned CellDim, class DataType, bool SOA>
__global__ void stepGPUHierarchical2(
    const void **__restrict__ _point_data, void *__restrict__ _point_data_out,
    const MY_SIZE *__restrict__ points_to_be_cached,
    const MY_SIZE *__restrict__ points_to_be_cached_offsets,
    void **__restrict__ _cell_data, const MY_SIZE **__restrict__ cell_to_node,
    const std::uint8_t *__restrict__ num_cell_colours,
    const std::uint8_t *__restrict__ cell_colours,
    const MY_SIZE *__restrict__ block_offsets, MY_SIZE num_cells,
    const MY_SIZE *__restrict__ point_stride, MY_SIZE cell_stride);

template <unsigned MeshDim, unsigned PointDim, unsigned CellDim, class DataType>
struct StepGPUHierarchical;
template <unsigned PointDim, unsigned CellDim, class DataType>
struct StepGPUHierarchical<2, PointDim, CellDim, DataType> {
  static constexpr unsigned MESH_DIM = 2;
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
    stepGPUHierarchical2<PointDim, CellDim, DataType,
                         SOA><<<num_blocks, block_size, cache_size>>>(
        point_data, point_data_out, points_to_be_cached,
        points_to_be_cached_offsets, cell_data, cell_to_node, num_cell_colours,
        cell_colours, block_offsets, num_cells, point_stride, cell_stride);
  }
};

template <unsigned PointDim, unsigned CellDim, class DataType, bool SOA>
__global__ void stepGPUHierarchical2(
    const void **__restrict__ _point_data, void *__restrict__ _point_data_out,
    const MY_SIZE *__restrict__ points_to_be_cached,
    const MY_SIZE *__restrict__ points_to_be_cached_offsets,
    void **__restrict__ _cell_data, const MY_SIZE **__restrict__ cell_to_node,
    const std::uint8_t *__restrict__ num_cell_colours,
    const std::uint8_t *__restrict__ cell_colours,
    const MY_SIZE *__restrict__ block_offsets, MY_SIZE num_cells,
    const MY_SIZE *__restrict__ point_stride, MY_SIZE cell_stride) {
  /* stepGPUHierarch {{{1 */
  const DataType *__restrict__ point_data =
      reinterpret_cast<const DataType *>(*_point_data);
  DataType *__restrict__ point_data_out =
      reinterpret_cast<DataType *>(_point_data_out);
  const DataType *__restrict__ cell_data =
      reinterpret_cast<const DataType *>(*_cell_data);

  const float4 *__restrict__ point_data_float4 =
      reinterpret_cast<const float4 *>(point_data);
  float4 *__restrict__ point_data_out_float4 =
      reinterpret_cast<float4 *>(point_data_out);
  const double2 *__restrict__ point_data_double2 =
      reinterpret_cast<const double2 *>(point_data);
  double2 *__restrict__ point_data_out_double2 =
      reinterpret_cast<double2 *>(point_data_out);

  constexpr unsigned MESH_DIM =
      StepGPUHierarchical<2, PointDim, CellDim, DataType>::MESH_DIM;

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

  static_assert(alignof(DataType) <= alignof(double),
                "Too large alignment needed.");
  extern __shared__ __align__(alignof(double)) unsigned char shared[];
  DataType *point_cache = reinterpret_cast<DataType *>(shared);

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
        float4 tmp = point_data_float4[g_ind / 4];
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
        double2 tmp = point_data_double2[g_ind / 2];
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
        point_cache[c_ind] = point_data[g_ind];
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

        point_cache[c_ind] = point_data[g_ind];
      }
    }
  }

  __syncthreads();

  // Computation
  DataType increment[PointDim * MESH_DIM];
  DataType *point_data_left;
  DataType *point_data_right;
  if (tid < block_size) {
    point_data_left = point_cache + cell_to_node[0][thread_ind];
    point_data_right = point_cache + cell_to_node[0][thread_ind + num_cells];
    const DataType *cell_data_cur = cell_data + thread_ind;
    mine_func_gpu<PointDim, CellDim, DataType, true>(
        point_data_left, point_data_right, increment, increment + PointDim,
        cell_data_cur, shared_num_cached_points, cell_stride);
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
        point_data_left[i * shared_num_cached_points] += increment[i];
        point_data_right[i * shared_num_cached_points] +=
            increment[i + PointDim];
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
  /* 1}}} */
}

template <unsigned PointDim, unsigned CellDim, class DataType, bool SOA>
__global__ void stepGPUHierarchical4(
    const void **__restrict__ _point_data, void *__restrict__ _point_data_out,
    const MY_SIZE *__restrict__ points_to_be_cached,
    const MY_SIZE *__restrict__ points_to_be_cached_offsets,
    void **__restrict__ _cell_data,
    const MY_SIZE **__restrict__ cell_to_node,
    const std::uint8_t *__restrict__ num_cell_colours,
    const std::uint8_t *__restrict__ cell_colours,
    const MY_SIZE *__restrict__ block_offsets, MY_SIZE num_cells,
    const MY_SIZE *__restrict__ point_stride, MY_SIZE cell_stride);

template <unsigned PointDim, unsigned CellDim, class DataType>
struct StepGPUHierarchical<4, PointDim, CellDim, DataType> {
  static constexpr unsigned MESH_DIM = 4;
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
    stepGPUHierarchical4<PointDim, CellDim, DataType,
                         SOA><<<num_blocks, block_size, cache_size>>>(
        point_data, point_data_out, points_to_be_cached,
        points_to_be_cached_offsets, cell_data, cell_to_node, num_cell_colours,
        cell_colours, block_offsets, num_cells, point_stride, cell_stride);
  }
};

template <unsigned PointDim, unsigned CellDim, class DataType, bool SOA>
__global__ void stepGPUHierarchical4(
    const void **__restrict__ _point_data, void *__restrict__ _point_data_out,
    const MY_SIZE *__restrict__ points_to_be_cached,
    const MY_SIZE *__restrict__ points_to_be_cached_offsets,
    void **__restrict__ _cell_data,
    const MY_SIZE **__restrict__ cell_to_node,
    const std::uint8_t *__restrict__ num_cell_colours,
    const std::uint8_t *__restrict__ cell_colours,
    const MY_SIZE *__restrict__ block_offsets, MY_SIZE num_cells,
    const MY_SIZE *__restrict__ point_stride, MY_SIZE cell_stride) {
  /* stepGPUHierarch {{{1 */
  const DataType *__restrict__ point_data =
      reinterpret_cast<const DataType *>(*_point_data);
  DataType *__restrict__ point_data_out =
      reinterpret_cast<DataType *>(_point_data_out);
  const DataType *__restrict__ cell_data =
      reinterpret_cast<const DataType *>(*_cell_data);

  const float4 *__restrict__ point_data_float4 =
      reinterpret_cast<const float4 *>(point_data);
  float4 *__restrict__ point_data_out_float4 =
      reinterpret_cast<float4 *>(point_data_out);
  const double2 *__restrict__ point_data_double2 =
      reinterpret_cast<const double2 *>(point_data);
  double2 *__restrict__ point_data_out_double2 =
      reinterpret_cast<double2 *>(point_data_out);

  constexpr unsigned MESH_DIM =
      StepGPUHierarchical<4, PointDim, CellDim, DataType>::MESH_DIM;

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

  static_assert(alignof(DataType) <= alignof(double),
                "Too large alignment needed.");
  extern __shared__ __align__(alignof(double)) unsigned char shared[];
  DataType *point_cache = reinterpret_cast<DataType *>(shared);

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
        float4 tmp = point_data_float4[g_ind / 4];
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
        double2 tmp = point_data_double2[g_ind / 2];
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
        point_cache[c_ind] = point_data[g_ind];
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

        point_cache[c_ind] = point_data[g_ind];
      }
    }
  }

  __syncthreads();

  // Computation
  DataType increment[PointDim * MESH_DIM];
  DataType *point_data0;
  DataType *point_data1;
  DataType *point_data2;
  DataType *point_data3;
  if (tid < block_size) {
    point_data0 = point_cache + cell_to_node[0][thread_ind];
    point_data1 = point_cache + cell_to_node[0][thread_ind + num_cells];
    point_data2 = point_cache + cell_to_node[0][thread_ind + 2 * num_cells];
    point_data3 = point_cache + cell_to_node[0][thread_ind + 3 * num_cells];
    const DataType *cell_data_cur = cell_data + thread_ind;
    mine_func_gpu<PointDim, CellDim, DataType, true>(
        point_data0, point_data1, point_data2, point_data3, increment,
        increment + PointDim, increment + 2 * PointDim,
        increment + 3 * PointDim, cell_data_cur, shared_num_cached_points,
        cell_stride);
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
        point_data0[i * shared_num_cached_points] += increment[i];
        point_data1[i * shared_num_cached_points] +=
            increment[i + 1 * PointDim];
        point_data2[i * shared_num_cached_points] +=
            increment[i + 2 * PointDim];
        point_data3[i * shared_num_cached_points] +=
            increment[i + 3 * PointDim];
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
  /* 1}}} */
}

template <unsigned PointDim, unsigned CellDim, class DataType, bool SOA>
__global__ void stepGPUHierarchical8(
    const void **__restrict__ _point_data, void *__restrict__ _point_data_out,
    const MY_SIZE *__restrict__ points_to_be_cached,
    const MY_SIZE *__restrict__ points_to_be_cached_offsets,
    void **__restrict__ _cell_data,
    const MY_SIZE **__restrict__ cell_to_node,
    const std::uint8_t *__restrict__ num_cell_colours,
    const std::uint8_t *__restrict__ cell_colours,
    const MY_SIZE *__restrict__ block_offsets, MY_SIZE num_cells,
    const MY_SIZE *__restrict__ point_stride, MY_SIZE cell_stride);

template <unsigned PointDim, unsigned CellDim, class DataType>
struct StepGPUHierarchical<8, PointDim, CellDim, DataType> {
  static constexpr unsigned MESH_DIM = 8;
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
    stepGPUHierarchical8<PointDim, CellDim, DataType,
                         SOA><<<num_blocks, block_size, cache_size>>>(
        point_data, point_data_out, points_to_be_cached,
        points_to_be_cached_offsets, cell_data, cell_to_node, num_cell_colours,
        cell_colours, block_offsets, num_cells, point_stride, cell_stride);
  }
};

template <unsigned PointDim, unsigned CellDim, class DataType, bool SOA>
__global__ void stepGPUHierarchical8(
    const void **__restrict__ _point_data, void *__restrict__ _point_data_out,
    const MY_SIZE *__restrict__ points_to_be_cached,
    const MY_SIZE *__restrict__ points_to_be_cached_offsets,
    void **__restrict__ _cell_data,
    const MY_SIZE **__restrict__ cell_to_node,
    const std::uint8_t *__restrict__ num_cell_colours,
    const std::uint8_t *__restrict__ cell_colours,
    const MY_SIZE *__restrict__ block_offsets, MY_SIZE num_cells,
    const MY_SIZE *__restrict__ point_stride, MY_SIZE cell_stride) {
  /* stepGPUHierarch {{{1 */
  const DataType *__restrict__ point_data =
      reinterpret_cast<const DataType *>(*_point_data);
  DataType *__restrict__ point_data_out =
      reinterpret_cast<DataType *>(_point_data_out);
  const DataType *__restrict__ cell_data =
      reinterpret_cast<const DataType *>(*_cell_data);

  const float4 *__restrict__ point_data_float4 =
      reinterpret_cast<const float4 *>(point_data);
  float4 *__restrict__ point_data_out_float4 =
      reinterpret_cast<float4 *>(point_data_out);
  const double2 *__restrict__ point_data_double2 =
      reinterpret_cast<const double2 *>(point_data);
  double2 *__restrict__ point_data_out_double2 =
      reinterpret_cast<double2 *>(point_data_out);

  constexpr unsigned MESH_DIM =
      StepGPUHierarchical<8, PointDim, CellDim, DataType>::MESH_DIM;

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

  static_assert(alignof(DataType) <= alignof(double),
                "Too large alignment needed.");
  extern __shared__ __align__(alignof(double)) unsigned char shared[];
  DataType *point_cache = reinterpret_cast<DataType *>(shared);

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
        float4 tmp = point_data_float4[g_ind / 4];
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
        double2 tmp = point_data_double2[g_ind / 2];
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
        point_cache[c_ind] = point_data[g_ind];
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

        point_cache[c_ind] = point_data[g_ind];
      }
    }
  }

  __syncthreads();

  // Computation
  DataType increment[PointDim * MESH_DIM];
  DataType *point_data0;
  DataType *point_data1;
  DataType *point_data2;
  DataType *point_data3;
  DataType *point_data4;
  DataType *point_data5;
  DataType *point_data6;
  DataType *point_data7;
  if (tid < block_size) {
    point_data0 = point_cache + cell_to_node[0][thread_ind];
    point_data1 = point_cache + cell_to_node[0][thread_ind + num_cells];
    point_data2 = point_cache + cell_to_node[0][thread_ind + 2 * num_cells];
    point_data3 = point_cache + cell_to_node[0][thread_ind + 3 * num_cells];
    point_data4 = point_cache + cell_to_node[0][thread_ind + 4 * num_cells];
    point_data5 = point_cache + cell_to_node[0][thread_ind + 5 * num_cells];
    point_data6 = point_cache + cell_to_node[0][thread_ind + 6 * num_cells];
    point_data7 = point_cache + cell_to_node[0][thread_ind + 7 * num_cells];
    const DataType *cell_data_cur = cell_data + thread_ind;
    mine_func_gpu<PointDim, CellDim, DataType, true>(
        point_data0, point_data1, point_data2, point_data3, point_data4,
        point_data5, point_data6, point_data7, increment, increment + PointDim,
        increment + 2 * PointDim, increment + 3 * PointDim,
        increment + 4 * PointDim, increment + 5 * PointDim,
        increment + 6 * PointDim, increment + 7 * PointDim, cell_data_cur,
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
        point_data0[i * shared_num_cached_points] += increment[i];
        point_data1[i * shared_num_cached_points] +=
            increment[i + 1 * PointDim];
        point_data2[i * shared_num_cached_points] +=
            increment[i + 2 * PointDim];
        point_data3[i * shared_num_cached_points] +=
            increment[i + 3 * PointDim];
        point_data4[i * shared_num_cached_points] +=
            increment[i + 4 * PointDim];
        point_data5[i * shared_num_cached_points] +=
            increment[i + 5 * PointDim];
        point_data6[i * shared_num_cached_points] +=
            increment[i + 6 * PointDim];
        point_data7[i * shared_num_cached_points] +=
            increment[i + 7 * PointDim];
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
  /* 1}}} */
}
}

// vim:set et sts=2 sw=2 ts=2 fdm=marker:
#endif /* end of include guard: MINE_HPP_IWZYHXFG */
