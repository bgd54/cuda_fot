

#ifndef USER_FUNCTION_SIGNATURE
#define USER_FUNCTION_SIGNATURE void user_func
#endif /* ifndef USER_FUNCTION_SIGNATURE */

#ifndef RESTRICT
#define RESTRICT
#endif /* ifndef RESTRICT */

#include "lulesh_utils.hpp"

USER_FUNCTION_SIGNATURE(double *RESTRICT f_out,
                        const double *RESTRICT xyz[MESH_DIM],
                        const double *RESTRICT sig, double &determ,
                        unsigned xyz_stride, unsigned direct_stride) {
  const double x[MESH_DIM]{xyz[0][0], xyz[1][0], xyz[2][0], xyz[3][0],
                           xyz[4][0], xyz[5][0], xyz[6][0], xyz[7][0]};
  const double y[MESH_DIM]{xyz[0][xyz_stride], xyz[1][xyz_stride],
                           xyz[2][xyz_stride], xyz[3][xyz_stride],
                           xyz[4][xyz_stride], xyz[5][xyz_stride],
                           xyz[6][xyz_stride], xyz[7][xyz_stride]};
  const double z[MESH_DIM]{xyz[0][2 * xyz_stride], xyz[1][2 * xyz_stride],
                           xyz[2][2 * xyz_stride], xyz[3][2 * xyz_stride],
                           xyz[4][2 * xyz_stride], xyz[5][2 * xyz_stride],
                           xyz[6][2 * xyz_stride], xyz[7][2 * xyz_stride]};
  const double sigxx = sig[0];
  const double sigyy = sig[1 * direct_stride];
  const double sigzz = sig[2 * direct_stride];

  double B[3][8];
  // code

  CalcElemShapeFunctionDerivatives(x, y, z, B, determ);
  CalcElemNodeNormals(B[0], B[1], B[2], x, y, z);
  SumElemStressesToNodeForces(B, sigxx, sigyy, sigzz, f_out, f_out + MESH_DIM,
                              f_out + 2 * MESH_DIM);
}

#undef USER_FUNCTION_SIGNATURE
#undef RESTRICT

// vim:set et sw=2 ts=2 fdm=marker:
