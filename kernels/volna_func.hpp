#ifndef USER_FUNCTION_SIGNATURE
#define USER_FUNCTION_SIGNATURE void user_func
#endif /* ifndef USER_FUNCTION_SIGNATURE */

#ifndef RESTRICT
#define RESTRICT
#endif /* ifndef RESTRICT */

/**
 * SOA-AOS layouts:
 * - the layout of `point_data` and `cell_data` can be either SOA or AOS,
 *   depending on their respective strides
 * - the layout of point_data_out is always AOS
 */

USER_FUNCTION_SIGNATURE(const float *RESTRICT cellVolumes0,
                        const float *RESTRICT cellVolumes1,
                        float *RESTRICT left,
                        float *RESTRICT right,
                        const float *RESTRICT edgeFluxes,
                        const float *RESTRICT bathySource,
                        const float *RESTRICT edgeNormals,
                        const int *RESTRICT isRightBoundary,
                        unsigned cell_stride) {
  // code
  left[0] -= (edgeFluxes[0*cell_stride])/cellVolumes0[0];
  left[1] -= (edgeFluxes[1*cell_stride] + bathySource[0*cell_stride] * edgeNormals[0*cell_stride])/cellVolumes0[0];
  left[2] -= (edgeFluxes[2*cell_stride] + bathySource[0*cell_stride] * edgeNormals[1*cell_stride])/cellVolumes0[0];

  if (!*isRightBoundary) {
    right[0] += edgeFluxes[0*cell_stride]/cellVolumes1[0];
    right[1] += (edgeFluxes[1*cell_stride] + bathySource[1*cell_stride] * edgeNormals[0*cell_stride])/cellVolumes1[0];
    right[2] += (edgeFluxes[2*cell_stride] + bathySource[1*cell_stride] * edgeNormals[1*cell_stride])/cellVolumes1[0];
  }

}

#undef USER_FUNCTION_SIGNATURE
#undef RESTRICT
