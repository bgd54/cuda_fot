#ifndef LULESH_UTILS_HPP_JTPIKPXC
#define LULESH_UTILS_HPP_JTPIKPXC

__host__ __device__ __forceinline__ void
CalcElemShapeFunctionDerivatives(double const x[], double const y[],
                                 double const z[], double b[][8],
                                 double &volume) {
  /*  {{{1 */
  // clang-format off
  const double x0 = x[0] ;   const double x1 = x[1] ;
  const double x2 = x[2] ;   const double x3 = x[3] ;
  const double x4 = x[4] ;   const double x5 = x[5] ;
  const double x6 = x[6] ;   const double x7 = x[7] ;

  const double y0 = y[0] ;   const double y1 = y[1] ;
  const double y2 = y[2] ;   const double y3 = y[3] ;
  const double y4 = y[4] ;   const double y5 = y[5] ;
  const double y6 = y[6] ;   const double y7 = y[7] ;

  const double z0 = z[0] ;   const double z1 = z[1] ;
  const double z2 = z[2] ;   const double z3 = z[3] ;
  const double z4 = z[4] ;   const double z5 = z[5] ;
  const double z6 = z[6] ;   const double z7 = z[7] ;

  double fjxxi, fjxet, fjxze;
  double fjyxi, fjyet, fjyze;
  double fjzxi, fjzet, fjzze;
  double cjxxi, cjxet, cjxze;
  double cjyxi, cjyet, cjyze;
  double cjzxi, cjzet, cjzze;

  fjxxi = double(.125) * ( (x6-x0) + (x5-x3) - (x7-x1) - (x4-x2) );
  fjxet = double(.125) * ( (x6-x0) - (x5-x3) + (x7-x1) - (x4-x2) );
  fjxze = double(.125) * ( (x6-x0) + (x5-x3) + (x7-x1) + (x4-x2) );

  fjyxi = double(.125) * ( (y6-y0) + (y5-y3) - (y7-y1) - (y4-y2) );
  fjyet = double(.125) * ( (y6-y0) - (y5-y3) + (y7-y1) - (y4-y2) );
  fjyze = double(.125) * ( (y6-y0) + (y5-y3) + (y7-y1) + (y4-y2) );

  fjzxi = double(.125) * ( (z6-z0) + (z5-z3) - (z7-z1) - (z4-z2) );
  fjzet = double(.125) * ( (z6-z0) - (z5-z3) + (z7-z1) - (z4-z2) );
  fjzze = double(.125) * ( (z6-z0) + (z5-z3) + (z7-z1) + (z4-z2) );

  /* compute cofactors */
  cjxxi =    (fjyet * fjzze) - (fjzet * fjyze);
  cjxet =  - (fjyxi * fjzze) + (fjzxi * fjyze);
  cjxze =    (fjyxi * fjzet) - (fjzxi * fjyet);

  cjyxi =  - (fjxet * fjzze) + (fjzet * fjxze);
  cjyet =    (fjxxi * fjzze) - (fjzxi * fjxze);
  cjyze =  - (fjxxi * fjzet) + (fjzxi * fjxet);

  cjzxi =    (fjxet * fjyze) - (fjyet * fjxze);
  cjzet =  - (fjxxi * fjyze) + (fjyxi * fjxze);
  cjzze =    (fjxxi * fjyet) - (fjyxi * fjxet);

  /* calculate partials :
     this need only be done for l = 0,1,2,3   since , by symmetry ,
     (6,7,4,5) = - (0,1,2,3) .
  */
  b[0][0] =   -  cjxxi  -  cjxet  -  cjxze;
  b[0][1] =      cjxxi  -  cjxet  -  cjxze;
  b[0][2] =      cjxxi  +  cjxet  -  cjxze;
  b[0][3] =   -  cjxxi  +  cjxet  -  cjxze;
  b[0][4] = -b[0][2];
  b[0][5] = -b[0][3];
  b[0][6] = -b[0][0];
  b[0][7] = -b[0][1];

  b[1][0] =   -  cjyxi  -  cjyet  -  cjyze;
  b[1][1] =      cjyxi  -  cjyet  -  cjyze;
  b[1][2] =      cjyxi  +  cjyet  -  cjyze;
  b[1][3] =   -  cjyxi  +  cjyet  -  cjyze;
  b[1][4] = -b[1][2];
  b[1][5] = -b[1][3];
  b[1][6] = -b[1][0];
  b[1][7] = -b[1][1];

  b[2][0] =   -  cjzxi  -  cjzet  -  cjzze;
  b[2][1] =      cjzxi  -  cjzet  -  cjzze;
  b[2][2] =      cjzxi  +  cjzet  -  cjzze;
  b[2][3] =   -  cjzxi  +  cjzet  -  cjzze;
  b[2][4] = -b[2][2];
  b[2][5] = -b[2][3];
  b[2][6] = -b[2][0];
  b[2][7] = -b[2][1];

  /* calculate jacobian determinant (volume) */
  volume = double(8.) * ( fjxet * cjxet + fjyet * cjyet + fjzet * cjzet);
  // clang-format on
  /* 1}}} */
}

__host__ __device__ __forceinline__ void SumElemFaceNormal(
    double *normalX0, double *normalY0, double *normalZ0, double *normalX1,
    double *normalY1, double *normalZ1, double *normalX2, double *normalY2,
    double *normalZ2, double *normalX3, double *normalY3, double *normalZ3,
    const double x0, const double y0, const double z0, const double x1,
    const double y1, const double z1, const double x2, const double y2,
    const double z2, const double x3, const double y3, const double z3) {
  /* {{{ */
  // clang-format off
   double bisectX0 = double(0.5) * (x3 + x2 - x1 - x0);
   double bisectY0 = double(0.5) * (y3 + y2 - y1 - y0);
   double bisectZ0 = double(0.5) * (z3 + z2 - z1 - z0);
   double bisectX1 = double(0.5) * (x2 + x1 - x3 - x0);
   double bisectY1 = double(0.5) * (y2 + y1 - y3 - y0);
   double bisectZ1 = double(0.5) * (z2 + z1 - z3 - z0);
   double areaX = double(0.25) * (bisectY0 * bisectZ1 - bisectZ0 * bisectY1);
   double areaY = double(0.25) * (bisectZ0 * bisectX1 - bisectX0 * bisectZ1);
   double areaZ = double(0.25) * (bisectX0 * bisectY1 - bisectY0 * bisectX1);

   *normalX0 += areaX;
   *normalX1 += areaX;
   *normalX2 += areaX;
   *normalX3 += areaX;

   *normalY0 += areaY;
   *normalY1 += areaY;
   *normalY2 += areaY;
   *normalY3 += areaY;

   *normalZ0 += areaZ;
   *normalZ1 += areaZ;
   *normalZ2 += areaZ;
   *normalZ3 += areaZ;
  // clang-format on
  /*}}}*/
}

__host__ __device__ __forceinline__ void
CalcElemNodeNormals(double pfx[8], double pfy[8], double pfz[8],
                    const double x[8], const double y[8], const double z[8]) {
  /* {{{1 */
  // clang-format off
   for (unsigned i = 0 ; i < 8 ; ++i) {
      pfx[i] = double(0.0);
      pfy[i] = double(0.0);
      pfz[i] = double(0.0);
   }
   /* evaluate face one: nodes 0, 1, 2, 3 */
   SumElemFaceNormal(&pfx[0], &pfy[0], &pfz[0],
                  &pfx[1], &pfy[1], &pfz[1],
                  &pfx[2], &pfy[2], &pfz[2],
                  &pfx[3], &pfy[3], &pfz[3],
                  x[0], y[0], z[0], x[1], y[1], z[1],
                  x[2], y[2], z[2], x[3], y[3], z[3]);
   /* evaluate face two: nodes 0, 4, 5, 1 */
   SumElemFaceNormal(&pfx[0], &pfy[0], &pfz[0],
                  &pfx[4], &pfy[4], &pfz[4],
                  &pfx[5], &pfy[5], &pfz[5],
                  &pfx[1], &pfy[1], &pfz[1],
                  x[0], y[0], z[0], x[4], y[4], z[4],
                  x[5], y[5], z[5], x[1], y[1], z[1]);
   /* evaluate face three: nodes 1, 5, 6, 2 */
   SumElemFaceNormal(&pfx[1], &pfy[1], &pfz[1],
                  &pfx[5], &pfy[5], &pfz[5],
                  &pfx[6], &pfy[6], &pfz[6],
                  &pfx[2], &pfy[2], &pfz[2],
                  x[1], y[1], z[1], x[5], y[5], z[5],
                  x[6], y[6], z[6], x[2], y[2], z[2]);
   /* evaluate face four: nodes 2, 6, 7, 3 */
   SumElemFaceNormal(&pfx[2], &pfy[2], &pfz[2],
                  &pfx[6], &pfy[6], &pfz[6],
                  &pfx[7], &pfy[7], &pfz[7],
                  &pfx[3], &pfy[3], &pfz[3],
                  x[2], y[2], z[2], x[6], y[6], z[6],
                  x[7], y[7], z[7], x[3], y[3], z[3]);
   /* evaluate face five: nodes 3, 7, 4, 0 */
   SumElemFaceNormal(&pfx[3], &pfy[3], &pfz[3],
                  &pfx[7], &pfy[7], &pfz[7],
                  &pfx[4], &pfy[4], &pfz[4],
                  &pfx[0], &pfy[0], &pfz[0],
                  x[3], y[3], z[3], x[7], y[7], z[7],
                  x[4], y[4], z[4], x[0], y[0], z[0]);
   /* evaluate face six: nodes 4, 7, 6, 5 */
   SumElemFaceNormal(&pfx[4], &pfy[4], &pfz[4],
                  &pfx[7], &pfy[7], &pfz[7],
                  &pfx[6], &pfy[6], &pfz[6],
                  &pfx[5], &pfy[5], &pfz[5],
                  x[4], y[4], z[4], x[7], y[7], z[7],
                  x[6], y[6], z[6], x[5], y[5], z[5]);
  // clang-format on
  /* 1}}} */
}

__host__ __device__ __forceinline__ void
SumElemStressesToNodeForces(const double B[][8], const double stress_xx,
                            const double stress_yy, const double stress_zz,
                            double fx[], double fy[], double fz[]) {
  /* {{{1 */
  // clang-format off
   for(unsigned i = 0; i < 8; i++) {
      fx[i] = -( stress_xx * B[0][i] );
      fy[i] = -( stress_yy * B[1][i]  );
      fz[i] = -( stress_zz * B[2][i] );
   }
  // clang-format on
  /* 1}}} */
}

#endif /* end of include guard: LULESH_UTILS_HPP_JTPIKPXC */
// vim:set et sw=2 ts=2 fdm=marker:
