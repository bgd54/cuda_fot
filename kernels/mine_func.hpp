

#ifndef USER_FUNCTION_SIGNATURE
#define USER_FUNCTION_SIGNATURE(fname) void fname
#endif /* ifndef USER_FUNCTION_SIGNATURE */

template <unsigned PointDim, unsigned CellDim, class DataType, bool SOA>
USER_FUNCTION_SIGNATURE(mine_func)
(const DataType *left_point_data, const DataType *right_point_data,
 DataType *left_point_data_out, DataType *right_point_data_out,
 const DataType *cell_data, unsigned point_stride, unsigned cell_stride) {
  point_stride = SOA ? point_stride : 1;
  static_assert(CellDim == PointDim || CellDim == 1, "CellDim makes no sense");
  for (unsigned point_d = 0; point_d < PointDim; ++point_d) {
    unsigned cell_d = CellDim == PointDim ? point_d : 0;
    left_point_data_out[point_d] = 2 *
                                   right_point_data[point_d * point_stride] *
                                   cell_data[cell_d * cell_stride];
    right_point_data_out[point_d] = 2 *
                                    left_point_data[point_d * point_stride] *
                                    cell_data[cell_d * cell_stride];
  }
}

template <unsigned PointDim, unsigned CellDim, class DataType, bool SOA>
USER_FUNCTION_SIGNATURE(mine_func)
(const DataType *point_data0, const DataType *point_data1,
 const DataType *point_data2, const DataType *point_data3,
 DataType *point_data_out0, DataType *point_data_out1,
 DataType *point_data_out2, DataType *point_data_out3,
 const DataType *cell_data, unsigned point_stride, unsigned cell_stride) {
  point_stride = SOA ? point_stride : 1;
  static_assert(CellDim == PointDim || CellDim == 1, "CellDim makes no sense");
  for (unsigned point_d = 0; point_d < PointDim; ++point_d) {
    unsigned cell_d = CellDim == PointDim ? point_d : 0;
    point_data_out0[point_d] = (point_data1[point_d * point_stride] +
                                point_data3[point_d * point_stride]) *
                               cell_data[cell_d * cell_stride];
    point_data_out1[point_d] = (point_data2[point_d * point_stride] +
                                point_data0[point_d * point_stride]) *
                               cell_data[cell_d * cell_stride];
    point_data_out2[point_d] = (point_data3[point_d * point_stride] +
                                point_data1[point_d * point_stride]) *
                               cell_data[cell_d * cell_stride];
    point_data_out3[point_d] = (point_data0[point_d * point_stride] +
                                point_data2[point_d * point_stride]) *
                               cell_data[cell_d * cell_stride];
  }
}

#undef USER_FUNCTION_SIGNATURE
