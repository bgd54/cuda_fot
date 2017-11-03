

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
#include <iostream>
#include <cassert>

USER_FUNCTION_SIGNATURE(const double *RESTRICT cnmass, const double *RESTRICT rho, const double *RESTRICT cnwt,
                    const double *RESTRICT cnfx, const double *RESTRICT cnfy,
		    double *point1,double *point2,double *point3,double *point4,
		    unsigned cell_stride) {
  constexpr double zerocut = 1.0e-40;

  // code
  int ii, jj;
//  std::cout << *ndmass1 << " " << cnmass[0] << "\n";


  jj = 0;
  if (cnmass[jj*cell_stride] > zerocut) {
    point1[0] += cnmass[jj*cell_stride];
  } else {
    ii = jj - 1;
    if (ii == -1)
      ii = 3;
    if (cnmass[ii*cell_stride] > zerocut) {
      point1[0] += cnmass[ii*cell_stride];
    } else {
      point1[0] += *rho * cnwt[jj*cell_stride];
    }
  }
  point1[1] += cnwt[jj*cell_stride];
  point1[2] += cnfx[jj*cell_stride];
  point1[3] += cnfy[jj*cell_stride];

  jj = 1;
  if (cnmass[jj*cell_stride] > zerocut) {
    point2[0] += cnmass[jj*cell_stride];
  } else {
    ii = jj - 1;
    if (ii == -1)
      ii = 3;
    if (cnmass[ii*cell_stride] > zerocut) {
      point2[0] += cnmass[ii*cell_stride];
    } else {
      point2[0] += *rho * cnwt[jj*cell_stride];
    }
  }
  point2[1] += cnwt[jj*cell_stride];
  point2[2] += cnfx[jj*cell_stride];
  point2[3] += cnfy[jj*cell_stride];

  jj = 2;
  if (cnmass[jj*cell_stride] > zerocut) {
    point3[0] += cnmass[jj*cell_stride];
  } else {
    ii = jj - 1;
    if (ii == -1)
      ii = 3;
    if (cnmass[ii*cell_stride] > zerocut) {
      point3[0] += cnmass[ii*cell_stride];
    } else {
      point3[0] += *rho * cnwt[jj*cell_stride];
    }
  }
  point3[1] += cnwt[jj*cell_stride];
  point3[2] += cnfx[jj*cell_stride];
  point3[3] += cnfy[jj*cell_stride];

  jj = 3;
  if (cnmass[jj*cell_stride] > zerocut) {
    point4[0] += cnmass[jj*cell_stride];
  } else {
    ii = jj - 1;
    if (ii == -1)
      ii = 3;
    if (cnmass[ii*cell_stride] > zerocut) {
      point4[0] += cnmass[ii*cell_stride];
    } else {
      point4[0] += *rho * cnwt[jj*cell_stride];
    }
  }
  point4[1] += cnwt[jj*cell_stride];
  point4[2] += cnfx[jj*cell_stride];
  point4[3] += cnfy[jj*cell_stride];
}

#undef USER_FUNCTION_SIGNATURE
#undef RESTRICT
