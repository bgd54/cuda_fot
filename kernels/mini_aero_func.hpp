
#ifndef USER_FUNCTION_SIGNATURE
#define USER_FUNCTION_SIGNATURE void user_func
#endif /* ifndef USER_FUNCTION_SIGNATURE */

#ifndef RESTRICT
#define RESTRICT
#endif /* ifndef RESTRICT */

#include "mini_aero_utils.hpp"

USER_FUNCTION_SIGNATURE(double flux[5], const double *RESTRICT cell_values_left,
                        const double *RESTRICT cell_values_right,
                        const double *RESTRICT cell_coordinates_left,
                        const double *RESTRICT cell_coordinates_right,
                        const double *RESTRICT cell_gradients_left,
                        const double *RESTRICT cell_gradients_right,
                        const double *RESTRICT cell_limiters_left,
                        const double *RESTRICT cell_limiters_right,
                        const double *RESTRICT face_coordinates,
                        const double *RESTRICT face_normal,
                        const double *RESTRICT face_tangent,
                        const double *RESTRICT face_binormal,
                        MY_SIZE cell_stride, MY_SIZE face_stride) {

  double conservatives_l[5];
  double conservatives_r[5];
  double primitives_l[5];
  double primitives_r[5];

  for (int icomp = 0; icomp < 5; ++icomp) {
    conservatives_l[icomp] = cell_values_left[icomp * cell_stride];
    conservatives_r[icomp] = cell_values_right[icomp * cell_stride];
  }

  ComputePrimitives(conservatives_l, primitives_l);
  ComputePrimitives(conservatives_r, primitives_r);

  // Extrapolation
  for (int icomp = 0; icomp < 5; ++icomp) {
    double gradient_primitive_l_tmp = 0;
    double gradient_primitive_r_tmp = 0;

    for (int idir = 0; idir < 3; ++idir) {
      gradient_primitive_l_tmp +=
          (face_coordinates[idir * face_stride] -
           cell_coordinates_left[idir * cell_stride]) *
          cell_gradients_left[(icomp * 3 + idir) * cell_stride];

      gradient_primitive_r_tmp +=
          (face_coordinates[idir * face_stride] -
           cell_coordinates_right[idir * cell_stride]) *
          cell_gradients_right[(icomp * 3 + idir) * cell_stride];
    }

    primitives_l[icomp] +=
        gradient_primitive_l_tmp * cell_limiters_left[icomp * cell_stride];
    primitives_r[icomp] +=
        gradient_primitive_r_tmp * cell_limiters_right[icomp * cell_stride];
  }

  inviscid_compute_flux(primitives_l, primitives_r, flux, face_normal,
                        face_tangent, face_binormal, face_stride);

  double primitives_face[5];
  double gradients_face[5][3];

  for (int icomp = 0; icomp < 5; ++icomp) {
    primitives_face[icomp] = 0.5 * (primitives_l[icomp] + primitives_r[icomp]);

    for (int idir = 0; idir < 3; ++idir) {
      gradients_face[icomp][idir] =
          0.5 * (cell_gradients_left[(icomp * 3 + idir) * cell_stride] +
                 cell_gradients_right[(icomp * 3 + idir) * cell_stride]);
    }
  }

  double vflux[5];
  viscous_compute_flux(gradients_face, primitives_face, face_normal, vflux,
                       face_stride);

  for (int icomp = 0; icomp < 5; ++icomp) {
    flux[icomp] -= vflux[icomp];
  }
}

#undef USER_FUNCTION_SIGNATURE
#undef RESTRICT
