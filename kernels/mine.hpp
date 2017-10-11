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
template <unsigned PointDim, unsigned CellDim> struct StepSeq {
  static constexpr unsigned MESH_DIM = 2;
  static void call(const void *_point_data, void *_point_data_out,
                   const void *_cell_data, const MY_SIZE *cell_to_node,
                   MY_SIZE ind) {
    assert(MESH_DIM_MACRO == 2);
    static_assert(CellDim == 1, "Not supporting SOA");
    const float *point_data = reinterpret_cast<const float *>(_point_data);
    float *point_data_out = reinterpret_cast<float *>(_point_data_out);
    const float *cell_data = reinterpret_cast<const float *>(_cell_data);

    const float *point_data_left =
        point_data + PointDim * cell_to_node[MESH_DIM * ind + 0];
    const float *point_data_right =
        point_data + PointDim * cell_to_node[MESH_DIM * ind + 1];
    float *point_data_out_left =
        point_data_out + PointDim * cell_to_node[MESH_DIM * ind + 0];
    float *point_data_out_right =
        point_data_out + PointDim * cell_to_node[MESH_DIM * ind + 1];
    const float *cell_data_cur = cell_data + CellDim * ind;

    // Calling user function
    mine_func_host<PointDim, CellDim>(point_data_left, point_data_right,
                                      point_data_out_left, point_data_out_right,
                                      cell_data_cur);
  }
};

template <unsigned PointDim, unsigned CellDim> struct StepSeq4 {
  static constexpr unsigned MESH_DIM = 4;
  static void call(const void *_point_data, void *_point_data_out,
                   const void *_cell_data, const MY_SIZE *cell_to_node,
                   MY_SIZE ind) {
    assert(MESH_DIM_MACRO == 4);
    static_assert(CellDim == 1, "Not supporting SOA");
    const float *point_data = reinterpret_cast<const float *>(_point_data);
    float *point_data_out = reinterpret_cast<float *>(_point_data_out);
    const float *cell_data = reinterpret_cast<const float *>(_cell_data);

    const float *point_data0 =
        point_data + PointDim * cell_to_node[MESH_DIM * ind + 0];
    const float *point_data1 =
        point_data + PointDim * cell_to_node[MESH_DIM * ind + 1];
    const float *point_data2 =
        point_data + PointDim * cell_to_node[MESH_DIM * ind + 2];
    const float *point_data3 =
        point_data + PointDim * cell_to_node[MESH_DIM * ind + 3];
    float *point_data_out0 =
        point_data_out + PointDim * cell_to_node[MESH_DIM * ind + 0];
    float *point_data_out1 =
        point_data_out + PointDim * cell_to_node[MESH_DIM * ind + 1];
    float *point_data_out2 =
        point_data_out + PointDim * cell_to_node[MESH_DIM * ind + 2];
    float *point_data_out3 =
        point_data_out + PointDim * cell_to_node[MESH_DIM * ind + 3];
    const float *cell_data_cur = cell_data + CellDim * ind;

    // Calling user function
    mine_func_host<PointDim, CellDim>(
        point_data0, point_data1, point_data2, point_data3, point_data_out0,
        point_data_out1, point_data_out2, point_data_out3, cell_data_cur);
  }
};

// OMP kernel
// Should be the same as the sequential
template <unsigned PointDim, unsigned CellDim>
using StepOMP = StepSeq<PointDim, CellDim>;
template <unsigned PointDim, unsigned CellDim>
using StepOMP4 = StepSeq4<PointDim, CellDim>;

// GPU global kernel
// TODO

// GPU hierarchical kernel
// TODO
}

// vim:set et sts=2 sw=2 ts=2:
#endif /* end of include guard: MINE_HPP_IWZYHXFG */