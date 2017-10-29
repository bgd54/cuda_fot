

#ifndef USER_FUNCTION_SIGNATURE
#define USER_FUNCTION_SIGNATURE void user_func
#endif /* ifndef USER_FUNCTION_SIGNATURE */

#ifndef RESTRICT
#define RESTRICT
#endif /* ifndef RESTRICT */

USER_FUNCTION_SIGNATURE(const double *RESTRICT x1, const double *RESTRICT x2,
                        const double *RESTRICT q1, const double *RESTRICT q2,
                        const double *RESTRICT adt1,
                        const double *RESTRICT adt2, double *RESTRICT res1,
                        double *RESTRICT res2, unsigned x_stride,
                        unsigned q_stride) {
  // globals
  double gm1 = 0.4, eps = 0.05;
  // code
  double dx, dy, mu, ri, p1, vol1, p2, vol2, f;

  dx = x1[0 * x_stride] - x2[0 * x_stride];
  dy = x1[1 * x_stride] - x2[1 * x_stride];

  ri = 1.0f / q1[0 * q_stride];
  p1 = gm1 * (q1[3 * q_stride] - 0.5f * ri *
                                     (q1[1 * q_stride] * q1[1 * q_stride] +
                                      q1[2 * q_stride] * q1[2 * q_stride]));
  vol1 = ri * (q1[1 * q_stride] * dy - q1[2 * q_stride] * dx);

  ri = 1.0f / q2[0 * q_stride];
  p2 = gm1 * (q2[3 * q_stride] - 0.5f * ri *
                                     (q2[1 * q_stride] * q2[1 * q_stride] +
                                      q2[2 * q_stride] * q2[2 * q_stride]));
  vol2 = ri * (q2[1 * q_stride] * dy - q2[2 * q_stride] * dx);

  mu = 0.5f * ((*adt1) + (*adt2)) * eps;

  f = 0.5f * (vol1 * q1[0 * q_stride] + vol2 * q2[0 * q_stride]) +
      mu * (q1[0 * q_stride] - q2[0 * q_stride]);
  res1[0] = f;
  res2[0] = -f;
  f = 0.5f * (vol1 * q1[1 * q_stride] + p1 * dy + vol2 * q2[1 * q_stride] +
              p2 * dy) +
      mu * (q1[1 * q_stride] - q2[1 * q_stride]);
  res1[1] = f;
  res2[1] = -f;
  f = 0.5f * (vol1 * q1[2 * q_stride] - p1 * dx + vol2 * q2[2 * q_stride] -
              p2 * dx) +
      mu * (q1[2 * q_stride] - q2[2 * q_stride]);
  res1[2] = f;
  res2[2] = -f;
  f = 0.5f * (vol1 * (q1[3 * q_stride] + p1) + vol2 * (q2[3 * q_stride] + p2)) +
      mu * (q1[3 * q_stride] - q2[3 * q_stride]);
  res1[3] = f;
  res2[3] = -f;
}

#undef USER_FUNCTION_SIGNATURE
#undef RESTRICT
