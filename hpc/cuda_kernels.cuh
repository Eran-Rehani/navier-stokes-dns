#pragma once
/* ================================================================
 *  cuda_kernels.cuh — Host-callable wrappers for GPU kernels
 *
 *  Every wrapper launches on the default stream unless noted.
 *  All pointers are device pointers.
 * ================================================================ */

#include "common.hpp"

namespace ns3d { namespace kernels {

/* ================================================================
 *  Transpose packing / unpacking
 * ================================================================ */

/*  X-pencil → Y-pencil  (row_comm, Py peers)
 *  src  layout:  [nz_x][ny_x][Nx]       (X contiguous)
 *  dst  layout:  [nz_y][nx_y][Ny]       (Y contiguous)            */
void pack_x_to_y  (const Complex* src, Complex* sendbuf,
                    int Nx, int ny_x, int nx_y, int nz_x, int Py);
void unpack_x_to_y(const Complex* recvbuf, Complex* dst,
                    int Ny, int ny_x, int nx_y, int nz_x, int Py);

/*  Y-pencil → Z-pencil  (col_comm, Pz peers)
 *  src  layout:  [nz_y][nx_y][Ny]       (Y contiguous)
 *  dst  layout:  [ny_z][nx_z][Nz]       (Z contiguous)            */
void pack_y_to_z  (const Complex* src, Complex* sendbuf,
                    int Ny, int nx_y, int ny_z, int nz_y, int Pz);
void unpack_y_to_z(const Complex* recvbuf, Complex* dst,
                    int Nz, int nx_y, int ny_z, int nz_y, int Pz);

/*  Z-pencil → Y-pencil  (col_comm, Pz peers)                      */
void pack_z_to_y  (const Complex* src, Complex* sendbuf,
                    int Nz, int nx_z, int ny_z, int nz_y, int Pz);
void unpack_z_to_y(const Complex* recvbuf, Complex* dst,
                    int Ny, int nx_y, int ny_z, int nz_y, int Pz);

/*  Y-pencil → X-pencil  (row_comm, Py peers)                      */
void pack_y_to_x  (const Complex* src, Complex* sendbuf,
                    int Ny, int nx_y, int ny_x, int nz_y, int Py);
void unpack_y_to_x(const Complex* recvbuf, Complex* dst,
                    int Nx, int nx_y, int ny_x, int nz_x, int Py);

/* ================================================================
 *  Spectral-space operations  (all on Z-pencil layout)
 * ================================================================ */

/*  Multiply by  i·k_j  to obtain the j-th partial derivative.
 *  dir = 0 (X), 1 (Y), 2 (Z).
 *  kx, ky, kz are wavenumber look-up arrays on device.            */
void spectral_derivative(const Complex* src, Complex* dst,
                         const int* kx, const int* ky, const int* kz,
                         int dir, int ny_z, int nx_z, int Nz);

/*  Orszag 2/3 dealiasing:  zero modes with  kx²+ky²+kz² > k_cut² */
void dealias(Complex* data,
             const int* kx, const int* ky, const int* kz,
             int k_cut_sq, int ny_z, int nx_z, int Nz);

/*  Pressure Poisson solve:
 *     p̂  = −i ρ / (Δt |k|²)  (k · û*)                           */
void pressure_poisson(Complex* p_hat,
                      const Complex* u_hat,
                      const Complex* v_hat,
                      const Complex* w_hat,
                      const int* kx, const int* ky, const int* kz,
                      Real rho_val, Real dt_val,
                      int ny_z, int nx_z, int Nz);

/*  Divergence-free Leray projection:
 *     û_i  ←  û_i − k_i (k·û) / |k|²                            */
void project(Complex* u_hat, Complex* v_hat, Complex* w_hat,
             const int* kx, const int* ky, const int* kz,
             int ny_z, int nx_z, int Nz);

/*  RK3 sub-step (fused viscous + update):
 *     Q   = A·Q + Δt·(−NL − ν|k|²·u)
 *     u  += B·Q                                                   */
void rk3_substep(Complex* u_hat, Complex* Q,
                 const Complex* nl_hat,
                 const int* kx, const int* ky, const int* kz,
                 Real A, Real B, Real dt_val, Real nu_val,
                 int ny_z, int nx_z, int Nz);

/* ================================================================
 *  Physical-space helpers  (X-pencil layout)
 * ================================================================ */

/*  Extract real part of IFFT result and accumulate product:
 *     nl[i]  +=  u_j[i] · Re(work_x[i]) · norm                  */
void accum_product(Real* nl, const Real* u_j,
                   const Complex* work_x, Real norm, int count);

/*  nl[i] = 0  */
void zero_real(Real* arr, int count);

/*  Copy real array into complex (zero imaginary):
 *     dst[i] = { src[i], 0 }                                     */
void real_to_complex(const Real* src, Complex* dst, int count);

/*  dst[i] = Re(src[i]) · norm  */
void complex_to_real(const Complex* src, Real* dst, Real norm, int count);

/* ================================================================
 *  Initialisation & diagnostics
 * ================================================================ */

/*  Taylor–Green vortex  u = sin x cos y cos z,  v = −cos x sin y cos z,  w = 0 */
void init_taylor_green(Real* u, Real* v, Real* w,
                       int Nx, int ny_x, int nz_x,
                       int start_y, int start_z, Real dx_val);

/*  Double shear layer  */
void init_shear_layer(Real* u, Real* v, Real* w,
                      int Nx, int ny_x, int nz_x,
                      int start_y, int start_z, Real dx_val);

/*  Random uniform noise  */
void init_random(Real* u, Real* v, Real* w,
                 int Nx, int ny_x, int nz_x,
                 int start_y, int start_z);

/*  Local reduction: E_local = 0.5·Σ(u²+v²+w²)  returned on host  */
Real local_kinetic_energy(const Real* u, const Real* v, const Real* w,
                          int count);

/*  Zero-out a complex array  */
void zero_complex(Complex* arr, int count);

} }  /* namespace ns3d::kernels */
