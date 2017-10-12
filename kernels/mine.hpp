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
template <unsigned PointDim, unsigned CellDim, class DataType> struct StepSeq {
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

    // Calling user function
    mine_func_host<PointDim, CellDim, DataType, SOA>(
        point_data_left, point_data_right, point_data_out_left,
        point_data_out_right, cell_data_cur, point_stride, cell_stride);
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

    // Calling user function
    mine_func_host<PointDim, CellDim, DataType, SOA>(
        point_data0, point_data1, point_data2, point_data3, point_data_out0,
        point_data_out1, point_data_out2, point_data_out3, cell_data_cur,
        point_stride, cell_stride);
  }
};

// OMP kernel
// Should be the same as the sequential
template <unsigned PointDim, unsigned CellDim, class DataType>
using StepOMP = StepSeq<PointDim, CellDim, DataType>;
template <unsigned PointDim, unsigned CellDim, class DataType>
using StepOMP4 = StepSeq4<PointDim, CellDim, DataType>;

// GPU global kernel
// TODO

// GPU hierarchical kernel
// TODO
}

// vim:set et sts=2 sw=2 ts=2:
#endif /* end of include guard: MINE_HPP_IWZYHXFG */
