

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

USER_FUNCTION_SIGNATURE(const double *RESTRICT cnmass, const double *RESTRICT rho, const double *RESTRICT cnwt,
                    const double *RESTRICT cnfx, const double *RESTRICT cnfy, double *RESTRICT ndmass1,
                    double *RESTRICT ndmass2, double *RESTRICT ndmass3, double *RESTRICT ndmass4,
                    double *RESTRICT ndarea1, double *RESTRICT ndarea2, double *RESTRICT ndarea3,
                    double *RESTRICT ndarea4, double *RESTRICT ndub1, double *RESTRICT ndub2,
                    double *RESTRICT ndub3, double *RESTRICT ndub4, double *RESTRICT ndvb1, double *RESTRICT ndvb2,
                    double *RESTRICT ndvb3, double *RESTRICT ndvb4, unsigned cell_stride) {
  constexpr double zerocut = 1.0e-40;
  // code
  unsigned ii, jj;

  jj = 0;
  if (cnmass[jj*cell_stride] > zerocut) {
    *ndmass1 += cnmass[jj*cell_stride];
  } else {
    ii = jj - 1;
    if (ii == -1)
      ii = 3;
    if (cnmass[ii*cell_stride] > zerocut) {
      *ndmass1 += cnmass[ii*cell_stride];
    } else {
      *ndmass1 += *rho * cnwt[jj*cell_stride];
    }
  }
  *ndarea1 += cnwt[jj*cell_stride];
  *ndub1 += cnfx[jj*cell_stride];
  *ndvb1 += cnfy[jj*cell_stride];

  jj = 1;
  if (cnmass[jj*cell_stride] > zerocut) {
    *ndmass2 += cnmass[jj*cell_stride];
  } else {
    ii = jj - 1;
    if (ii == -1)
      ii = 3;
    if (cnmass[ii*cell_stride] > zerocut) {
      *ndmass2 += cnmass[ii*cell_stride];
    } else {
      *ndmass2 += *rho * cnwt[jj*cell_stride];
    }
  }
  *ndarea2 += cnwt[jj*cell_stride];
  *ndub2 += cnfx[jj*cell_stride];
  *ndvb2 += cnfy[jj*cell_stride];

  jj = 2;
  if (cnmass[jj*cell_stride] > zerocut) {
    *ndmass3 += cnmass[jj*cell_stride];
  } else {
    ii = jj - 1;
    if (ii == -1)
      ii = 3;
    if (cnmass[ii*cell_stride] > zerocut) {
      *ndmass3 += cnmass[ii*cell_stride];
    } else {
      *ndmass3 += *rho * cnwt[jj*cell_stride];
    }
  }
  *ndarea3 += cnwt[jj*cell_stride];
  *ndub3 += cnfx[jj*cell_stride];
  *ndvb3 += cnfy[jj*cell_stride];

  jj = 3;
  if (cnmass[jj*cell_stride] > zerocut) {
    *ndmass4 += cnmass[jj*cell_stride];
  } else {
    ii = jj - 1;
    if (ii == -1)
      ii = 3;
    if (cnmass[ii*cell_stride] > zerocut) {
      *ndmass4 += cnmass[ii*cell_stride];
    } else {
      *ndmass4 += *rho * cnwt[jj*cell_stride];
    }
  }
  *ndarea4 += cnwt[jj*cell_stride];
  *ndub4 += cnfx[jj*cell_stride];
  *ndvb4 += cnfy[jj*cell_stride];
}

#undef USER_FUNCTION_SIGNATURE
#undef RESTRICT
