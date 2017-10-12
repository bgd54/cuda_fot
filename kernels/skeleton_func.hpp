

#ifndef USER_FUNCTION_SIGNATURE
#define USER_FUNCTION_SIGNATURE void user_func
#endif /* ifndef USER_FUNCTION_SIGNATURE */

template <bool SOA>
USER_FUNCTION_SIGNATURE(const float *point_data, float *point_data_out,
                        const float *cell_data, unsigned point_stride,
                        unsigned cell_stride) {
  point_stride = SOA ? point_stride : 1;
  // code
}

#undef USER_FUNCTION_SIGNATURE
