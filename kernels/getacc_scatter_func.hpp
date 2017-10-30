

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
                    double *RESTRICT ndvb3, double *RESTRICT ndvb4) {
  constexpr double zerocut = 1.0e-40;
  // code
  unsigned ii, jj;

  jj = 1;
  if (cnmass[jj] > zerocut) {
    *ndmass1 += cnmass[jj];
  } else {
    ii = jj - 1;
    if (ii == 0)
      ii = 4;
    if (cnmass[ii] > zerocut) {
      *ndmass1 += cnmass[ii];
    } else {
      *ndmass1 += rho * cnwt[jj];
    }
  }
  *ndarea1 += cnwt[jj];
  *ndub1 += cnfx[jj];
  *ndvb1 += cnfy[jj];

  jj = 2;
  if (cnmass[jj] > zerocut) {
    *ndmass2 += cnmass[jj];
  } else {
    ii = jj - 1;
    if (ii == 0)
      ii = 4;
    if (cnmass[ii] > zerocut) {
      *ndmass2 += cnmass[ii];
    } else {
      *ndmass2 += rho * cnwt[jj];
    }
  }
  *ndarea2 += cnwt[jj];
  *ndub2 += cnfx[jj];
  *ndvb2 += cnfy[jj];

  jj = 3;
  if (cnmass[jj] > zerocut) {
    *ndmass3 += cnmass[jj];
  } else {
    ii = jj - 1;
    if (ii == 0)
      ii = 4;
    if (cnmass[ii] > zerocut) {
      *ndmass3 += cnmass[ii];
    } else {
      *ndmass3 += rho * cnwt[jj];
    }
  }
  *ndarea3 += cnwt[jj];
  *ndub3 += cnfx[jj];
  *ndvb3 += cnfy[jj];

  jj = 4;
  if (cnmass[jj] > zerocut) {
    *ndmass4 += cnmass[jj];
  } else {
    ii = jj - 1;
    if (ii == 0)
      ii = 4;
    if (cnmass[ii] > zerocut) {
      *ndmass4 += cnmass[ii];
    } else {
      *ndmass4 += rho * cnwt[jj];
    }
  }
  *ndarea4 += cnwt[jj];
  *ndub4 += cnfx[jj];
  *ndvb4 += cnfy[jj];
}

#undef USER_FUNCTION_SIGNATURE
#undef RESTRICT
