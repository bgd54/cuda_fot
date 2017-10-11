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
  static void call(const void *_point_data, void *_point_data_out,
                   const void *_cell_data, const MY_SIZE *cell_to_node,
                   MY_SIZE ind) {
    const float *point_data = reinterpret_cast<const float *>(_point_data);
    float *point_data_out = reinterpret_cast<float *>(_point_data_out);
    const float *cell_data = reinterpret_cast<const float *>(_cell_data);

    const float *point_data_cur =
        point_data + POINT_DIM * cell_to_node[MESH_DIM * ind + 0];
    float *point_data_out_cur =
        point_data_out + POINT_DIM * cell_to_node[MESH_DIM * ind + 0];
    const float *cell_data_cur = cell_data + CELL_DIM * ind;

    // Calling user function
    user_func_host(point_data_cur, point_data_out_cur, cell_data_cur);
  }
};

// OMP kernel
// Should be the same as the sequential
using StepOMP = StepSeq;

// GPU global kernel
// TODO

// GPU hierarchical kernel
// TODO
}
#endif /* end of include guard: SKELETON_HPP_BTXZV4YZ */
// vim:set et sts=2 sw=2 ts=2:
