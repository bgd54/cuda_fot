

#ifndef USER_FUNCTION_SIGNATURE
#define USER_FUNCTION_SIGNATURE void user_func
#endif /* ifndef USER_FUNCTION_SIGNATURE */

#ifndef RESTRICT
#define RESTRICT
#endif /* ifndef RESTRICT */

template <bool SOA>
USER_FUNCTION_SIGNATURE(const float *RESTRICT point_data,
                        float *RESTRICT point_data_out,
                        const float *RESTRICT cell_data, unsigned point_stride,
                        unsigned cell_stride) {
  point_stride = SOA ? point_stride : 1;
  // code
}

#undef USER_FUNCTION_SIGNATURE
#undef RESTRICT
