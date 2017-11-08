#ifndef MINI_AERO_UTILS_HPP_WIQLSJNK
#define MINI_AERO_UTILS_HPP_WIQLSJNK

#include <cmath>

__device__ __host__ double ComputePressure(const double *V) {
  const double Rgas = 287.05;
  const double rho = V[0];
  const double T = V[4];

  return rho * Rgas * T;
}

__device__ __host__ double ComputeEnthalpy(const double *V) {
  const double Cp = 1004.0;
  const double T = V[4];
  return Cp * T;
}

__device__ __host__ static void MatVec5(const double alpha, const double A[],
                                        const double x[], const double beta,
                                        double y[]) {
  for (int i = 0; i < 5; ++i) {
    y[i] *= beta;
    for (int j = 0; j < 5; ++j) {
      y[i] += alpha * A[5 * i + j] * x[j];
    }
  }
}

__device__ __host__ void
inviscid_compute_flux(const double *primitives_left,
                      const double *primitives_right, double *flux,
                      const double *face_normal, const double *face_tangent,
                      const double *face_binormal, MY_SIZE face_stride) {

  // Eigenvalue fix constants.
  const double efix_u = 0.1;
  const double efix_c = 0.1;

  const double gm1 = 0.4;

  // Left state
  const double rho_left = primitives_left[0];
  const double uvel_left = primitives_left[1];
  const double vvel_left = primitives_left[2];
  const double wvel_left = primitives_left[3];

  const double pressure_left = ComputePressure(primitives_left);
  const double enthalpy_left = ComputeEnthalpy(primitives_left);

  const double total_enthalpy_left =
      enthalpy_left +
      0.5 * (uvel_left * uvel_left + vvel_left * vvel_left +
             wvel_left * wvel_left);
  const double mass_flux_left =
      rho_left * (face_normal[0 * face_stride] * uvel_left +
                  face_normal[1 * face_stride] * vvel_left +
                  face_normal[2 * face_stride] * wvel_left);

  // Right state
  const double rho_right = primitives_right[0];
  const double uvel_right = primitives_right[1];
  const double vvel_right = primitives_right[2];
  const double wvel_right = primitives_right[3];

  const double pressure_right = ComputePressure(primitives_right);
  const double enthalpy_right = ComputeEnthalpy(primitives_right);

  const double total_enthalpy_right =
      enthalpy_right +
      0.5 * (uvel_right * uvel_right + vvel_right * vvel_right +
             wvel_right * wvel_right);
  const double mass_flux_right =
      rho_right * (face_normal[0 * face_stride] * uvel_right +
                   face_normal[1 * face_stride] * vvel_right +
                   face_normal[2 * face_stride] * wvel_right);

  const double pressure_sum = pressure_left + pressure_right;

  // Central flux contribution part
  flux[0] = 0.5 * (mass_flux_left + mass_flux_right);
  flux[1] = 0.5 * (mass_flux_left * uvel_left + mass_flux_right * uvel_right +
                   face_normal[0 * face_stride] * pressure_sum);
  flux[2] = 0.5 * (mass_flux_left * vvel_left + mass_flux_right * vvel_right +
                   face_normal[1 * face_stride] * pressure_sum);
  flux[3] = 0.5 * (mass_flux_left * wvel_left + mass_flux_right * wvel_right +
                   face_normal[2 * face_stride] * pressure_sum);
  flux[4] = 0.5 * (mass_flux_left * total_enthalpy_left +
                   mass_flux_right * total_enthalpy_right);

  // Upwinded part
  const double face_normal_norm =
      std::sqrt(face_normal[0 * face_stride] * face_normal[0 * face_stride] +
                face_normal[1 * face_stride] * face_normal[1 * face_stride] +
                face_normal[2 * face_stride] * face_normal[2 * face_stride]);

  // const double face_normal_norm =
  // MathTools<execution_space>::Vec3Norm(face_normal);
  const double face_tangent_norm =
      std::sqrt(face_tangent[0 * face_stride] * face_tangent[0 * face_stride] +
                face_tangent[1 * face_stride] * face_tangent[1 * face_stride] +
                face_tangent[2 * face_stride] * face_tangent[2 * face_stride]);

  // const double face_binormal_norm =
  // MathTools<execution_space>::Vec3Norm(face_binormal);
  const double face_binormal_norm = std::sqrt(
      face_binormal[0 * face_stride] * face_binormal[0 * face_stride] +
      face_binormal[1 * face_stride] * face_binormal[1 * face_stride] +
      face_binormal[2 * face_stride] * face_binormal[2 * face_stride]);

  const double face_normal_unit[] = {
      face_normal[0 * face_stride] / face_normal_norm,
      face_normal[1 * face_stride] / face_normal_norm,
      face_normal[2 * face_stride] / face_normal_norm};
  const double face_tangent_unit[] = {
      face_tangent[0 * face_stride] / face_tangent_norm,
      face_tangent[1 * face_stride] / face_tangent_norm,
      face_tangent[2 * face_stride] / face_tangent_norm};
  const double face_binormal_unit[] = {
      face_binormal[0 * face_stride] / face_binormal_norm,
      face_binormal[1 * face_stride] / face_binormal_norm,
      face_binormal[2 * face_stride] / face_binormal_norm};

  const double denom = 1.0 / (std::sqrt(rho_left) + std::sqrt(rho_right));
  const double alpha = sqrt(rho_left) * denom;
  const double beta = 1.0 - alpha;

  const double uvel_roe = alpha * uvel_left + beta * uvel_right;
  const double vvel_roe = alpha * vvel_left + beta * vvel_right;
  const double wvel_roe = alpha * wvel_left + beta * wvel_right;
  const double enthalpy_roe =
      alpha * enthalpy_left + beta * enthalpy_right +
      0.5 * alpha * beta *
          ((uvel_right - uvel_left) * (uvel_right - uvel_left) +
           (vvel_right - vvel_left) * (vvel_right - vvel_left) +
           (wvel_right - wvel_left) * (wvel_right - wvel_left));
  const double speed_sound_roe = std::sqrt(gm1 * enthalpy_roe);

  // Compute flux matrices
  double roe_mat_eigenvectors[25];
  // double roe_mat_right_eigenvectors[25];

  const double normal_velocity = uvel_roe * face_normal_unit[0] +
                                 vvel_roe * face_normal_unit[1] +
                                 wvel_roe * face_normal_unit[2];
  const double tangent_velocity = uvel_roe * face_tangent_unit[0] +
                                  vvel_roe * face_tangent_unit[1] +
                                  wvel_roe * face_tangent_unit[2];
  const double binormal_velocity = uvel_roe * face_binormal_unit[0] +
                                   vvel_roe * face_binormal_unit[1] +
                                   wvel_roe * face_binormal_unit[2];
  const double kinetic_energy_roe =
      0.5 * (uvel_roe * uvel_roe + vvel_roe * vvel_roe + wvel_roe * wvel_roe);
  const double speed_sound_squared_inverse =
      1.0 / (speed_sound_roe * speed_sound_roe);
  const double half_speed_sound_squared_inverse =
      0.5 * speed_sound_squared_inverse;

  // Conservative variable jumps
  double conserved_jump[] = {0.0, 0.0, 0.0, 0.0, 0.0};
  conserved_jump[0] = rho_right - rho_left;
  conserved_jump[1] = rho_right * uvel_right - rho_left * uvel_left;
  conserved_jump[2] = rho_right * vvel_right - rho_left * vvel_left;
  conserved_jump[3] = rho_right * wvel_right - rho_left * wvel_left;
  conserved_jump[4] = (rho_right * total_enthalpy_right - pressure_right) -
                      (rho_left * total_enthalpy_left - pressure_left);

  // Compute CFL number
  const double cbar = speed_sound_roe * face_normal_norm;
  const double ubar = uvel_roe * face_normal[0 * face_stride] +
                      vvel_roe * face_normal[1 * face_stride] +
                      wvel_roe * face_normal[2 * face_stride];
  const double cfl = std::abs(ubar) + cbar;

  // Eigenvalue fix
  const double eig1 = ubar + cbar;
  const double eig2 = ubar - cbar;
  const double eig3 = ubar;

  double abs_eig1 = std::abs(eig1);
  double abs_eig2 = std::abs(eig2);
  double abs_eig3 = std::abs(eig3);

  const double epuc = efix_u * cfl;
  const double epcc = efix_c * cfl;

  // Original Roe eigenvalue fix
  if (abs_eig1 < epcc)
    abs_eig1 = 0.5 * (eig1 * eig1 + epcc * epcc) / epcc;
  if (abs_eig2 < epcc)
    abs_eig2 = 0.5 * (eig2 * eig2 + epcc * epcc) / epcc;
  if (abs_eig3 < epuc)
    abs_eig3 = 0.5 * (eig3 * eig3 + epuc * epuc) / epuc;

  double eigp[] = {0.5 * (eig1 + abs_eig1), 0.5 * (eig2 + abs_eig2),
                   0.5 * (eig3 + abs_eig3), 0.0, 0.0};
  eigp[3] = eigp[4] = eigp[2];

  double eigm[] = {0.5 * (eig1 - abs_eig1), 0.5 * (eig2 - abs_eig2),
                   0.5 * (eig3 - abs_eig3), 0.0, 0.0};
  eigm[3] = eigm[4] = eigm[2];

  // Compute upwind flux
  double ldq[] = {0, 0, 0, 0, 0};
  // double lldq[] = { 0, 0, 0, 0, 0 };
  double rlldq[] = {0, 0, 0, 0, 0};

  // Left matrix
  roe_mat_eigenvectors[0] =
      gm1 * (kinetic_energy_roe - enthalpy_roe) +
      speed_sound_roe * (speed_sound_roe - normal_velocity);
  roe_mat_eigenvectors[1] =
      speed_sound_roe * face_normal_unit[0] - gm1 * uvel_roe;
  roe_mat_eigenvectors[2] =
      speed_sound_roe * face_normal_unit[1] - gm1 * vvel_roe;
  roe_mat_eigenvectors[3] =
      speed_sound_roe * face_normal_unit[2] - gm1 * wvel_roe;
  roe_mat_eigenvectors[4] = gm1;

  roe_mat_eigenvectors[5] =
      gm1 * (kinetic_energy_roe - enthalpy_roe) +
      speed_sound_roe * (speed_sound_roe + normal_velocity);
  roe_mat_eigenvectors[6] =
      -speed_sound_roe * face_normal_unit[0] - gm1 * uvel_roe;
  roe_mat_eigenvectors[7] =
      -speed_sound_roe * face_normal_unit[1] - gm1 * vvel_roe;
  roe_mat_eigenvectors[8] =
      -speed_sound_roe * face_normal_unit[2] - gm1 * wvel_roe;
  roe_mat_eigenvectors[9] = gm1;

  roe_mat_eigenvectors[10] = kinetic_energy_roe - enthalpy_roe;
  roe_mat_eigenvectors[11] = -uvel_roe;
  roe_mat_eigenvectors[12] = -vvel_roe;
  roe_mat_eigenvectors[13] = -wvel_roe;
  roe_mat_eigenvectors[14] = 1.0;

  roe_mat_eigenvectors[15] = -tangent_velocity;
  roe_mat_eigenvectors[16] = face_tangent_unit[0];
  roe_mat_eigenvectors[17] = face_tangent_unit[1];
  roe_mat_eigenvectors[18] = face_tangent_unit[2];
  roe_mat_eigenvectors[19] = 0.0;

  roe_mat_eigenvectors[20] = -binormal_velocity;
  roe_mat_eigenvectors[21] = face_binormal_unit[0];
  roe_mat_eigenvectors[22] = face_binormal_unit[1];
  roe_mat_eigenvectors[23] = face_binormal_unit[2];
  roe_mat_eigenvectors[24] = 0.0;

  MatVec5(1.0, roe_mat_eigenvectors, conserved_jump, 0.0, ldq);

  for (int j = 0; j < 5; ++j)
    ldq[j] = (eigp[j] - eigm[j]) * ldq[j];

  // Right matrix
  roe_mat_eigenvectors[0] = half_speed_sound_squared_inverse;
  roe_mat_eigenvectors[1] = half_speed_sound_squared_inverse;
  roe_mat_eigenvectors[2] = -gm1 * speed_sound_squared_inverse;
  roe_mat_eigenvectors[3] = 0.0;
  roe_mat_eigenvectors[4] = 0.0;

  roe_mat_eigenvectors[5] = (uvel_roe + face_normal_unit[0] * speed_sound_roe) *
                            half_speed_sound_squared_inverse;
  roe_mat_eigenvectors[6] = (uvel_roe - face_normal_unit[0] * speed_sound_roe) *
                            half_speed_sound_squared_inverse;
  roe_mat_eigenvectors[7] = -gm1 * uvel_roe * speed_sound_squared_inverse;
  roe_mat_eigenvectors[8] = face_tangent_unit[0];
  roe_mat_eigenvectors[9] = face_binormal_unit[0];

  roe_mat_eigenvectors[10] =
      (vvel_roe + face_normal_unit[1] * speed_sound_roe) *
      half_speed_sound_squared_inverse;
  roe_mat_eigenvectors[11] =
      (vvel_roe - face_normal_unit[1] * speed_sound_roe) *
      half_speed_sound_squared_inverse;
  roe_mat_eigenvectors[12] = -gm1 * vvel_roe * speed_sound_squared_inverse;
  roe_mat_eigenvectors[13] = face_tangent_unit[1];
  roe_mat_eigenvectors[14] = face_binormal_unit[1];

  roe_mat_eigenvectors[15] =
      (wvel_roe + face_normal_unit[2] * speed_sound_roe) *
      half_speed_sound_squared_inverse;
  roe_mat_eigenvectors[16] =
      (wvel_roe - face_normal_unit[2] * speed_sound_roe) *
      half_speed_sound_squared_inverse;
  roe_mat_eigenvectors[17] = -gm1 * wvel_roe * speed_sound_squared_inverse;
  roe_mat_eigenvectors[18] = face_tangent_unit[2];
  roe_mat_eigenvectors[19] = face_binormal_unit[2];

  roe_mat_eigenvectors[20] =
      (enthalpy_roe + kinetic_energy_roe + speed_sound_roe * normal_velocity) *
      half_speed_sound_squared_inverse;
  roe_mat_eigenvectors[21] =
      (enthalpy_roe + kinetic_energy_roe - speed_sound_roe * normal_velocity) *
      half_speed_sound_squared_inverse;
  roe_mat_eigenvectors[22] = (speed_sound_roe * speed_sound_roe -
                              gm1 * (enthalpy_roe + kinetic_energy_roe)) *
                             speed_sound_squared_inverse;
  roe_mat_eigenvectors[23] = tangent_velocity;
  roe_mat_eigenvectors[24] = binormal_velocity;

  MatVec5(1.0, roe_mat_eigenvectors, ldq, 0.0, rlldq);

  for (int icomp = 0; icomp < 5; ++icomp)
    flux[icomp] -= 0.5 * rlldq[icomp];
}

__device__ __host__ double ComputeViscosity(const double temperature) {
  const double sutherland_0 = 1.458e-6;
  const double sutherland_1 = 110.4;
  return sutherland_0 * temperature * std::sqrt(temperature) /
         (temperature + sutherland_1);
}

__device__ __host__ double ComputeThermalConductivity(const double viscosity) {
  const double Pr = 0.71;
  const double Cp = 1006.0;
  return viscosity * Cp / Pr;
}

__device__ __host__ void
viscous_compute_flux(const double grad_primitive[5][3], const double *primitive,
                     const double *a_vec, double *vflux, MY_SIZE a_vec_stride) {
  double viscosity = ComputeViscosity(primitive[4]);
  double thermal_conductivity = ComputeThermalConductivity(viscosity);
  double divergence_velocity = 0;

  for (int icomp = 0; icomp < 5; ++icomp) {
    vflux[icomp] = 0.0;
  }

  for (int i = 0; i < 3; i++) {
    divergence_velocity += grad_primitive[i + 1][i];
  }

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      const double delta_ij = (i == j) ? 1 : 0;
      const double S_ij =
          0.5 * (grad_primitive[i + 1][j] + grad_primitive[j + 1][i]);
      const double t_ij = S_ij - divergence_velocity * delta_ij / 3.;
      vflux[1 + i] += (2 * viscosity * t_ij) * a_vec[j * a_vec_stride];
      vflux[4] +=
          (2 * viscosity * t_ij) * primitive[i + 1] * a_vec[j * a_vec_stride];
    }

    vflux[4] +=
        thermal_conductivity * grad_primitive[4][i] * a_vec[i * a_vec_stride];
  }

  return;
}

__device__ __host__ void ComputePrimitives(const double *U, double *V) {
  double gamma = 1.4;
  double Rgas = 287.05;
  double r, u, v, w, T, ri, k, e;

  r = U[0];
  ri = 1.0 / r;
  u = U[1] * ri;
  v = U[2] * ri;
  w = U[3] * ri;
  k = 0.5 * (u * u + v * v + w * w);
  e = U[4] * ri - k;
  T = e * (gamma - 1.0) / Rgas;

  V[0] = r;
  V[1] = u;
  V[2] = v;
  V[3] = w;
  V[4] = T;
}

#endif /* end of include guard: MINI_AERO_UTILS_HPP_WIQLSJNK */
