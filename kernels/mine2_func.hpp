#include <cmath>

#ifndef USER_FUNCTION_SIGNATURE
#define USER_FUNCTION_SIGNATURE void user_func
#endif /* ifndef USER_FUNCTION_SIGNATURE */

#ifndef RESTRICT
#define RESTRICT
#endif /* ifndef RESTRICT */

USER_FUNCTION_SIGNATURE(
    const float *RESTRICT point_data0, float *RESTRICT point_data_out,
    const float *RESTRICT point_data1_left,
    const float *RESTRICT point_data1_right, const float *RESTRICT cell_data0,
    const float *RESTRICT cell_data1, const double *RESTRICT cell_data2,
    unsigned point_stride0, unsigned point_stride1, unsigned cell_stride) {
  constexpr unsigned POINT_DIM0 = 2, POINT_DIM1 = 4;
  double t = 0;
  for (unsigned i = 0; i < POINT_DIM1; ++i) {
    t += point_data1_left[i * point_stride1] *
         point_data1_right[i * point_stride1];
  }
  t *= cell_data1[0];
  for (unsigned i = 0; i < POINT_DIM0; ++i) {
    point_data_out[i] =
        point_data0[(1 - i) * point_stride0] * cell_data2[i * cell_stride] +
        std::pow(t, cell_data0[i * cell_stride]) + cell_data2[2 * cell_stride];
  }
}

#undef USER_FUNCTION_SIGNATURE
#undef RESTRICT
