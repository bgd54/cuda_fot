

#ifndef USER_FUNCTION_SIGNATURE
#define USER_FUNCTION_SIGNATURE(fname) void fname
#endif /* ifndef USER_FUNCTION_SIGNATURE */

template <unsigned PointDim, unsigned CellDim>
USER_FUNCTION_SIGNATURE(mine_func)
(const float *left_point_data, const float *right_point_data,
 float *left_point_data_out, float *right_point_data_out,
 const float *cell_data) {
  static_assert(CellDim == PointDim || CellDim == 1, "CellDim makes no sense");
  for (unsigned point_d = 0; point_d < PointDim; ++point_d) {
    unsigned cell_d = CellDim == PointDim ? point_d : 0;
    left_point_data_out[point_d] +=
        2 * right_point_data[point_d] * cell_data[cell_d];
    right_point_data_out[point_d] +=
        2 * left_point_data[point_d] * cell_data[cell_d];
  }
}

template<unsigned PointDim, unsigned CellDim>
USER_FUNCTION_SIGNATURE(mine_func)
(const float *point_data0, const float *point_data1, const float *point_data2,
 const float *point_data3, float *point_data_out0, float *point_data_out1,
 float *point_data_out2, float *point_data_out3, const float *cell_data) {
  static_assert(CellDim == PointDim || CellDim == 1, "CellDim makes no sense");
  for (unsigned point_d = 0; point_d < PointDim; ++point_d) {
    unsigned cell_d = CellDim == PointDim ? point_d : 0;
    point_data_out0[point_d] += (point_data1[point_d] + point_data3[point_d])
      * cell_data[cell_d];
    point_data_out1[point_d] += (point_data2[point_d] + point_data0[point_d])
      * cell_data[cell_d];
    point_data_out2[point_d] += (point_data3[point_d] + point_data1[point_d])
      * cell_data[cell_d];
    point_data_out3[point_d] += (point_data0[point_d] + point_data2[point_d])
      * cell_data[cell_d];
  }
}

#undef USER_FUNCTION_SIGNATURE
