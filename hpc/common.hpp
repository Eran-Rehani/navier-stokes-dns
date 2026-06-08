#pragma once
/* ================================================================
 *  common.hpp — Precision, Constants, Aliases, Error Macros
 *  3D Pseudo-Spectral Incompressible Navier-Stokes DNS
 * ================================================================ */

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cufft.h>
#include <mpi.h>

namespace ns3d {

/* ----------------------------------------------------------------
 *  Precision and complex type aliases
 * ---------------------------------------------------------------- */
using Real    = double;
using Complex = cufftDoubleComplex;          /* {double x, y;} */

static constexpr cufftType CUFFT_C2C_TYPE = CUFFT_Z2Z;   /* double-precision C2C */

/* ----------------------------------------------------------------
 *  Physical / numerical parameters
 *
 *  Modify these at compile time with  -DNS3D_N=512  etc.
 * ---------------------------------------------------------------- */
#ifndef NS3D_N
#define NS3D_N 256
#endif

static constexpr int    N           = NS3D_N;             /* grid points per axis       */
static constexpr Real   L           = 2.0 * M_PI;         /* domain side length [0,2π]  */
static constexpr Real   nu          = 1.0e-3;             /* kinematic viscosity         */
static constexpr Real   rho         = 1.0;                /* density (incompressible)    */
static constexpr Real   dt          = 1.0e-3;             /* time-step size              */
static constexpr int    num_steps   = 1000;               /* total integration steps     */
static constexpr int    diag_interval = 100;              /* diagnostic output interval  */
static constexpr Real   dx          = L / N;

/* ----------------------------------------------------------------
 *  Orszag 2/3-rule dealiasing cut-off
 *  Modes with  |k|² > k_cut²  are set to zero.
 * ---------------------------------------------------------------- */
static constexpr int  k_cut    = N / 3;
static constexpr int  k_cut_sq = k_cut * k_cut;

/* ----------------------------------------------------------------
 *  Low-Storage 3rd-Order Runge-Kutta  (Williamson 1980)
 *
 *  Q  = A[s] Q + Δt f(u)
 *  u  = u + B[s] Q
 * ---------------------------------------------------------------- */
static constexpr Real RK_A[3] = { 0.0,  -5.0 / 9.0,  -153.0 / 128.0 };
static constexpr Real RK_B[3] = { 1.0 / 3.0,  15.0 / 16.0,  8.0 / 15.0 };
static constexpr int  RK_STAGES = 3;

/* ----------------------------------------------------------------
 *  CUDA / cuFFT / MPI error-checking macros
 * ---------------------------------------------------------------- */
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err_ = (call);                                           \
        if (err_ != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n",                  \
                         __FILE__, __LINE__, cudaGetErrorString(err_));      \
            MPI_Abort(MPI_COMM_WORLD, static_cast<int>(err_));              \
        }                                                                    \
    } while (0)

#define CUFFT_CHECK(call)                                                    \
    do {                                                                     \
        cufftResult err_ = (call);                                           \
        if (err_ != CUFFT_SUCCESS) {                                         \
            std::fprintf(stderr, "cuFFT error %s:%d: code %d\n",            \
                         __FILE__, __LINE__, static_cast<int>(err_));        \
            MPI_Abort(MPI_COMM_WORLD, static_cast<int>(err_));              \
        }                                                                    \
    } while (0)

#define MPI_CHECK(call)                                                      \
    do {                                                                     \
        int err_ = (call);                                                   \
        if (err_ != MPI_SUCCESS) {                                           \
            char buf_[MPI_MAX_ERROR_STRING]; int len_;                       \
            MPI_Error_string(err_, buf_, &len_);                             \
            std::fprintf(stderr, "MPI error %s:%d: %s\n",                   \
                         __FILE__, __LINE__, buf_);                          \
            MPI_Abort(MPI_COMM_WORLD, err_);                                \
        }                                                                    \
    } while (0)

/* ----------------------------------------------------------------
 *  Inline complex arithmetic  (host + device)
 * ---------------------------------------------------------------- */
__host__ __device__ inline Complex make_cmplx(Real re, Real im) {
    Complex c; c.x = re; c.y = im; return c;
}

__host__ __device__ inline Complex cadd(Complex a, Complex b) {
    return make_cmplx(a.x + b.x, a.y + b.y);
}

__host__ __device__ inline Complex csub(Complex a, Complex b) {
    return make_cmplx(a.x - b.x, a.y - b.y);
}

__host__ __device__ inline Complex cmul(Complex a, Complex b) {
    return make_cmplx(a.x * b.x - a.y * b.y,
                      a.x * b.y + a.y * b.x);
}

__host__ __device__ inline Complex cscale(Complex a, Real s) {
    return make_cmplx(a.x * s, a.y * s);
}

/* Multiply by the imaginary unit:  i·(a+bi) = -b + ai */
__host__ __device__ inline Complex cmul_i(Complex a) {
    return make_cmplx(-a.y, a.x);
}

/* |z|² */
__host__ __device__ inline Real cnorm2(Complex a) {
    return a.x * a.x + a.y * a.y;
}

/* ----------------------------------------------------------------
 *  Wavenumber from DFT index
 *
 *  idx ∈ [0, N):
 *    0 … N/2       →  k = idx          (positive frequencies)
 *    N/2+1 … N-1   →  k = idx - N      (negative frequencies)
 * ---------------------------------------------------------------- */
__host__ __device__ inline int wavenumber(int idx, int Ndim) {
    return (idx <= Ndim / 2) ? idx : idx - Ndim;
}

/* ----------------------------------------------------------------
 *  CUDA launch helpers
 * ---------------------------------------------------------------- */
static constexpr int BLOCK_SIZE = 256;

inline int grid_size(int total_threads) {
    return (total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

}  /* namespace ns3d */
