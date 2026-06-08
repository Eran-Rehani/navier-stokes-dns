/* ================================================================
 *  cuda_kernels.cu — GPU Kernel Implementations
 *  3D Pseudo-Spectral Incompressible Navier-Stokes DNS
 *
 *  All global memory accesses are warp-coalesced by design:
 *  the fastest-varying index maps to threadIdx.x via a linear
 *  decomposition of the flat array index.
 * ================================================================ */

#include "cuda_kernels.cuh"

namespace ns3d { namespace kernels {

/* ================================================================
 *  §1  TRANSPOSE PACK / UNPACK KERNELS
 *
 *  Pointer arithmetic convention
 *  ─────────────────────────────────────────────────────────────
 *  Source / destination arrays are stored in row-major C order
 *  with the pencil-aligned dimension as the *fastest* index.
 *
 *  X-pencil  src[iz · ny_x · Nx  + iy · Nx  + ix]
 *  Y-pencil  src[iz · nx_y · Ny  + ix · Ny  + iy]
 *  Z-pencil  src[iy · nx_z · Nz  + ix · Nz  + iz]
 *
 *  The send buffer is laid out as Pcomm contiguous chunks of
 *  identical size  chunk = nslow · nmid · nchunk_fast ,  where
 *  nchunk_fast = nfast_full / Pcomm.
 * ================================================================ */

/* ----------------------------------------------------------------
 *  X → Y   pack / unpack       (row_comm, Py peers)
 *
 *  Pack:   for each dest p, extract X-slab [p·nx_y, (p+1)·nx_y)
 *          from every (iz, iy) in the X-pencil.
 *
 *  sendbuf[ p·chunk + iz·ny_x·nx_y + iy·nx_y + ix_loc ]
 *        = src[ iz·ny_x·Nx + iy·Nx + p·nx_y + ix_loc ]
 *
 *  Unpack: scatter received chunks into the Y-pencil.
 *
 *  recvbuf[ q·chunk + iz·ny_x·nx_y + iy_q·nx_y + ix ]
 *        → dst[ iz·nx_y·Ny + ix·Ny + q·ny_x + iy_q ]
 * ---------------------------------------------------------------- */
__global__ void k_pack_x_to_y(const Complex* __restrict__ src,
                               Complex*       __restrict__ buf,
                               int Nx, int ny_x, int nx_y, int nz_x, int Py)
{
    const int chunk = nz_x * ny_x * nx_y;
    const int total = Py * chunk;
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < total;
         tid += gridDim.x * blockDim.x)
    {
        int p      = tid / chunk;
        int rem    = tid % chunk;
        int iz     = rem / (ny_x * nx_y);
        int iy     = (rem / nx_y) % ny_x;
        int ix_loc = rem % nx_y;

        int src_idx = iz * ny_x * Nx + iy * Nx + p * nx_y + ix_loc;
        buf[tid] = src[src_idx];
    }
}

__global__ void k_unpack_x_to_y(const Complex* __restrict__ buf,
                                 Complex*       __restrict__ dst,
                                 int Ny, int ny_x, int nx_y, int nz_x, int Py)
{
    const int chunk = nz_x * ny_x * nx_y;
    const int total = Py * chunk;
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < total;
         tid += gridDim.x * blockDim.x)
    {
        int q      = tid / chunk;
        int rem    = tid % chunk;
        int iz     = rem / (ny_x * nx_y);
        int iy_q   = (rem / nx_y) % ny_x;
        int ix     = rem % nx_y;

        int dst_idx = iz * nx_y * Ny + ix * Ny + q * ny_x + iy_q;
        dst[dst_idx] = buf[tid];
    }
}

/* ----------------------------------------------------------------
 *  Y → Z   pack / unpack       (col_comm, Pz peers)
 *
 *  Pack:   for each dest p, extract Y-slab [p·ny_z, (p+1)·ny_z)
 *          from every (iz, ix) in the Y-pencil.
 *
 *  sendbuf[ p·chunk + iz·nx_y·ny_z + ix·ny_z + iy_loc ]
 *        = src[ iz·nx_y·Ny + ix·Ny + p·ny_z + iy_loc ]
 *
 *  Unpack: scatter received chunks into the Z-pencil.
 *
 *  recvbuf[ q·chunk + iz_q·nx_y·ny_z + ix·ny_z + iy ]
 *        → dst[ iy·nx_z·Nz + ix·Nz + q·nz_y + iz_q ]
 * ---------------------------------------------------------------- */
__global__ void k_pack_y_to_z(const Complex* __restrict__ src,
                               Complex*       __restrict__ buf,
                               int Ny, int nx_y, int ny_z, int nz_y, int Pz)
{
    const int chunk = nz_y * nx_y * ny_z;
    const int total = Pz * chunk;
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < total;
         tid += gridDim.x * blockDim.x)
    {
        int p      = tid / chunk;
        int rem    = tid % chunk;
        int iz     = rem / (nx_y * ny_z);
        int ix     = (rem / ny_z) % nx_y;
        int iy_loc = rem % ny_z;

        int src_idx = iz * nx_y * Ny + ix * Ny + p * ny_z + iy_loc;
        buf[tid] = src[src_idx];
    }
}

__global__ void k_unpack_y_to_z(const Complex* __restrict__ buf,
                                 Complex*       __restrict__ dst,
                                 int Nz, int nx_y, int ny_z, int nz_y, int Pz)
{
    const int chunk = nz_y * nx_y * ny_z;
    const int total = Pz * chunk;
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < total;
         tid += gridDim.x * blockDim.x)
    {
        int q      = tid / chunk;
        int rem    = tid % chunk;
        int iz_q   = rem / (nx_y * ny_z);
        int ix     = (rem / ny_z) % nx_y;
        int iy     = rem % ny_z;

        int dst_idx = iy * nx_y * Nz + ix * Nz + q * nz_y + iz_q;
        dst[dst_idx] = buf[tid];
    }
}

/* ----------------------------------------------------------------
 *  Z → Y   pack / unpack       (col_comm, Pz peers)
 *
 *  Pack:   for each dest p, extract Z-slab [p·nz_y, (p+1)·nz_y)
 *          from every (iy, ix) in the Z-pencil.
 *
 *  sendbuf[ p·chunk + iy·nx_z·nz_y + ix·nz_y + iz_loc ]
 *        = src[ iy·nx_z·Nz + ix·Nz + p·nz_y + iz_loc ]
 *
 *  Unpack: scatter received chunks into the Y-pencil.
 *
 *  recvbuf[ q·chunk + iy_q·nx_z·nz_y + ix·nz_y + iz ]
 *        → dst[ iz·nx_y·Ny + ix·Ny + q·ny_z + iy_q ]
 * ---------------------------------------------------------------- */
__global__ void k_pack_z_to_y(const Complex* __restrict__ src,
                               Complex*       __restrict__ buf,
                               int Nz, int nx_z, int ny_z, int nz_y, int Pz)
{
    const int chunk = ny_z * nx_z * nz_y;
    const int total = Pz * chunk;
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < total;
         tid += gridDim.x * blockDim.x)
    {
        int p      = tid / chunk;
        int rem    = tid % chunk;
        int iy     = rem / (nx_z * nz_y);
        int ix     = (rem / nz_y) % nx_z;
        int iz_loc = rem % nz_y;

        int src_idx = iy * nx_z * Nz + ix * Nz + p * nz_y + iz_loc;
        buf[tid] = src[src_idx];
    }
}

__global__ void k_unpack_z_to_y(const Complex* __restrict__ buf,
                                 Complex*       __restrict__ dst,
                                 int Ny, int nx_y, int ny_z, int nz_y, int Pz)
{
    const int chunk = ny_z * nx_y * nz_y;
    const int total = Pz * chunk;
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < total;
         tid += gridDim.x * blockDim.x)
    {
        int q      = tid / chunk;
        int rem    = tid % chunk;
        int iy_q   = rem / (nx_y * nz_y);
        int ix     = (rem / nz_y) % nx_y;
        int iz     = rem % nz_y;

        int dst_idx = iz * nx_y * Ny + ix * Ny + q * ny_z + iy_q;
        dst[dst_idx] = buf[tid];
    }
}

/* ----------------------------------------------------------------
 *  Y → X   pack / unpack       (row_comm, Py peers)
 *
 *  Pack:   for each dest p, extract Y-slab [p·ny_x, (p+1)·ny_x)
 *          from every (iz, ix) in the Y-pencil.
 *
 *  sendbuf[ p·chunk + iz·nx_y·ny_x + ix·ny_x + iy_loc ]
 *        = src[ iz·nx_y·Ny + ix·Ny + p·ny_x + iy_loc ]
 *
 *  Unpack: scatter received chunks into the X-pencil.
 *
 *  recvbuf[ q·chunk + iz·nx_y·ny_x + ix_q·ny_x + iy ]
 *        → dst[ iz·ny_x·Nx + iy·Nx + q·nx_y + ix_q ]
 * ---------------------------------------------------------------- */
__global__ void k_pack_y_to_x(const Complex* __restrict__ src,
                               Complex*       __restrict__ buf,
                               int Ny, int nx_y, int ny_x, int nz_y, int Py)
{
    const int chunk = nz_y * nx_y * ny_x;
    const int total = Py * chunk;
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < total;
         tid += gridDim.x * blockDim.x)
    {
        int p      = tid / chunk;
        int rem    = tid % chunk;
        int iz     = rem / (nx_y * ny_x);
        int ix     = (rem / ny_x) % nx_y;
        int iy_loc = rem % ny_x;

        int src_idx = iz * nx_y * Ny + ix * Ny + p * ny_x + iy_loc;
        buf[tid] = src[src_idx];
    }
}

__global__ void k_unpack_y_to_x(const Complex* __restrict__ buf,
                                 Complex*       __restrict__ dst,
                                 int Nx, int nx_y, int ny_x, int nz_x, int Py)
{
    const int chunk = nz_x * nx_y * ny_x;
    const int total = Py * chunk;
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < total;
         tid += gridDim.x * blockDim.x)
    {
        int q      = tid / chunk;
        int rem    = tid % chunk;
        int iz     = rem / (nx_y * ny_x);
        int ix_q   = (rem / ny_x) % nx_y;
        int iy     = rem % ny_x;

        int dst_idx = iz * ny_x * Nx + iy * Nx + q * nx_y + ix_q;
        dst[dst_idx] = buf[tid];
    }
}

/* ================================================================
 *  §2  SPECTRAL-SPACE KERNELS   (Z-pencil layout)
 *
 *  All spectral kernels operate on arrays of size  ny_z × nx_z × Nz
 *  with Z as the fastest-varying (contiguous) index:
 *
 *      flat_idx  =  iy · nx_z · Nz  +  ix · Nz  +  iz
 *
 *  Wavenumber arrays are pre-computed look-up tables:
 *      kx[ix]   for  ix ∈ [0, nx_z)    (local X indices)
 *      ky[iy]   for  iy ∈ [0, ny_z)    (local Y indices)
 *      kz[iz]   for  iz ∈ [0, Nz)      (full Z indices)
 * ================================================================ */

/* ----------------------------------------------------------------
 *  Spectral derivative:  dst = i·k_j · src
 *
 *  i·(a+bi) = (−b, a)  scaled by k_j.
 *  The wavenumber index is selected by  dir ∈ {0,1,2} = {X,Y,Z}.
 * ---------------------------------------------------------------- */
__global__ void k_spectral_deriv(const Complex* __restrict__ src,
                                  Complex*       __restrict__ dst,
                                  const int* __restrict__ kx,
                                  const int* __restrict__ ky,
                                  const int* __restrict__ kz,
                                  int dir, int ny_z, int nx_z, int Nz)
{
    const int total = ny_z * nx_z * Nz;
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < total;
         tid += gridDim.x * blockDim.x)
    {
        int iz = tid % Nz;
        int ix = (tid / Nz) % nx_z;
        int iy = tid / (nx_z * Nz);

        double k;
        switch (dir) {
            case 0:  k = static_cast<double>(kx[ix]); break;
            case 1:  k = static_cast<double>(ky[iy]); break;
            default: k = static_cast<double>(kz[iz]); break;
        }

        Complex s = src[tid];
        dst[tid].x = -s.y * k;          /* Re(i·k·s) = −k·Im(s) */
        dst[tid].y =  s.x * k;          /* Im(i·k·s) =  k·Re(s) */
    }
}

/* ----------------------------------------------------------------
 *  Orszag 2/3-rule dealiasing filter
 *
 *  Spherical truncation:  set  f̂(k) = 0  when  kx²+ky²+kz² > k_cut²
 * ---------------------------------------------------------------- */
__global__ void k_dealias(Complex* __restrict__ data,
                           const int* __restrict__ kx,
                           const int* __restrict__ ky,
                           const int* __restrict__ kz,
                           int k_cut_sq, int ny_z, int nx_z, int Nz)
{
    const int total = ny_z * nx_z * Nz;
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < total;
         tid += gridDim.x * blockDim.x)
    {
        int iz = tid % Nz;
        int ix = (tid / Nz) % nx_z;
        int iy = tid / (nx_z * Nz);

        int kxv = kx[ix], kyv = ky[iy], kzv = kz[iz];
        int ksq = kxv * kxv + kyv * kyv + kzv * kzv;

        if (ksq > k_cut_sq) {
            data[tid].x = 0.0;
            data[tid].y = 0.0;
        }
    }
}

/* ----------------------------------------------------------------
 *  Pressure Poisson solve  (diagnostic; not used in projection path)
 *
 *     p̂ = −i ρ / (Δt |k|²)  ·  (k · û*)
 *
 *  k · û  is a complex dot product;  multiplication by  −i  maps
 *  (a+bi) → (b − ai).
 * ---------------------------------------------------------------- */
__global__ void k_pressure_poisson(Complex*       __restrict__ p_hat,
                                    const Complex* __restrict__ u_hat,
                                    const Complex* __restrict__ v_hat,
                                    const Complex* __restrict__ w_hat,
                                    const int* __restrict__ kx,
                                    const int* __restrict__ ky,
                                    const int* __restrict__ kz,
                                    double rho_v, double dt_v,
                                    int ny_z, int nx_z, int Nz)
{
    const int total = ny_z * nx_z * Nz;
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < total;
         tid += gridDim.x * blockDim.x)
    {
        int iz = tid % Nz;
        int ix = (tid / Nz) % nx_z;
        int iy = tid / (nx_z * Nz);

        double kxv = static_cast<double>(kx[ix]);
        double kyv = static_cast<double>(ky[iy]);
        double kzv = static_cast<double>(kz[iz]);
        double ksq = kxv * kxv + kyv * kyv + kzv * kzv;

        if (ksq < 1.0e-14) {
            p_hat[tid].x = 0.0;
            p_hat[tid].y = 0.0;
            return;
        }

        Complex u = u_hat[tid], v = v_hat[tid], w = w_hat[tid];

        /* k · û  (complex scalar) */
        double kdot_re = kxv * u.x + kyv * v.x + kzv * w.x;
        double kdot_im = kxv * u.y + kyv * v.y + kzv * w.y;

        /* −i · (kdot_re + i·kdot_im) = kdot_im − i·kdot_re */
        double coeff = rho_v / (dt_v * ksq);
        p_hat[tid].x =  coeff * kdot_im;
        p_hat[tid].y = -coeff * kdot_re;
    }
}

/* ----------------------------------------------------------------
 *  Divergence-free Leray projection
 *
 *     û_i  ←  û_i  −  k_i · (k · û) / |k|²
 *
 *  The zero-wavenumber mode (|k|² < ε) is left untouched,
 *  preserving the mean flow set by the initial condition.
 * ---------------------------------------------------------------- */
__global__ void k_project(Complex* __restrict__ u_hat,
                           Complex* __restrict__ v_hat,
                           Complex* __restrict__ w_hat,
                           const int* __restrict__ kx,
                           const int* __restrict__ ky,
                           const int* __restrict__ kz,
                           int ny_z, int nx_z, int Nz)
{
    const int total = ny_z * nx_z * Nz;
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < total;
         tid += gridDim.x * blockDim.x)
    {
        int iz = tid % Nz;
        int ix = (tid / Nz) % nx_z;
        int iy = tid / (nx_z * Nz);

        double kxv = static_cast<double>(kx[ix]);
        double kyv = static_cast<double>(ky[iy]);
        double kzv = static_cast<double>(kz[iz]);
        double ksq = kxv * kxv + kyv * kyv + kzv * kzv;

        if (ksq < 1.0e-14) return;

        Complex u = u_hat[tid], v = v_hat[tid], w = w_hat[tid];

        double div_re = kxv * u.x + kyv * v.x + kzv * w.x;
        double div_im = kxv * u.y + kyv * v.y + kzv * w.y;

        double inv_ksq = 1.0 / ksq;

        u_hat[tid].x = u.x - kxv * div_re * inv_ksq;
        u_hat[tid].y = u.y - kxv * div_im * inv_ksq;
        v_hat[tid].x = v.x - kyv * div_re * inv_ksq;
        v_hat[tid].y = v.y - kyv * div_im * inv_ksq;
        w_hat[tid].x = w.x - kzv * div_re * inv_ksq;
        w_hat[tid].y = w.y - kzv * div_im * inv_ksq;
    }
}

/* ----------------------------------------------------------------
 *  RK3 sub-step   (fused viscous term + Williamson accumulator)
 *
 *     RHS_i = −NL̂_i − ν |k|² û_i
 *     Q_i   = A · Q_i  +  Δt · RHS_i
 *     û_i  += B · Q_i
 * ---------------------------------------------------------------- */
__global__ void k_rk3_substep(Complex* __restrict__ u_hat,
                               Complex* __restrict__ Q,
                               const Complex* __restrict__ nl_hat,
                               const int* __restrict__ kx,
                               const int* __restrict__ ky,
                               const int* __restrict__ kz,
                               double A, double B,
                               double dt_v, double nu_v,
                               int ny_z, int nx_z, int Nz)
{
    const int total = ny_z * nx_z * Nz;
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < total;
         tid += gridDim.x * blockDim.x)
    {
        int iz = tid % Nz;
        int ix = (tid / Nz) % nx_z;
        int iy = tid / (nx_z * Nz);

        int kxv = kx[ix], kyv = ky[iy], kzv = kz[iz];
        double ksq = static_cast<double>(kxv * kxv + kyv * kyv + kzv * kzv);

        Complex u  = u_hat[tid];
        Complex q  = Q[tid];
        Complex nl = nl_hat[tid];

        /* RHS = −NL − ν|k|² u */
        double rhs_re = -nl.x - nu_v * ksq * u.x;
        double rhs_im = -nl.y - nu_v * ksq * u.y;

        /* Low-storage update */
        q.x = A * q.x + dt_v * rhs_re;
        q.y = A * q.y + dt_v * rhs_im;

        u.x += B * q.x;
        u.y += B * q.y;

        u_hat[tid] = u;
        Q[tid]     = q;
    }
}

/* ================================================================
 *  §3  PHYSICAL-SPACE HELPER KERNELS   (X-pencil layout)
 * ================================================================ */

/* ----------------------------------------------------------------
 *  Fused extract-real-part / multiply / accumulate
 *
 *     nl[i]  +=  u_j[i] · Re(work_x[i]) · norm
 *
 *  Avoids a separate kernel launch for Re-extraction.
 * ---------------------------------------------------------------- */
__global__ void k_accum_product(Real*          __restrict__ nl,
                                 const Real*    __restrict__ u_j,
                                 const Complex* __restrict__ work_x,
                                 Real norm, int count)
{
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < count;
         tid += gridDim.x * blockDim.x)
    {
        nl[tid] += u_j[tid] * work_x[tid].x * norm;
    }
}

__global__ void k_zero_real(Real* __restrict__ arr, int count)
{
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < count;
         tid += gridDim.x * blockDim.x)
    {
        arr[tid] = 0.0;
    }
}

__global__ void k_real_to_complex(const Real* __restrict__ src,
                                   Complex*    __restrict__ dst,
                                   int count)
{
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < count;
         tid += gridDim.x * blockDim.x)
    {
        dst[tid].x = src[tid];
        dst[tid].y = 0.0;
    }
}

__global__ void k_complex_to_real(const Complex* __restrict__ src,
                                   Real*          __restrict__ dst,
                                   Real norm, int count)
{
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < count;
         tid += gridDim.x * blockDim.x)
    {
        dst[tid] = src[tid].x * norm;
    }
}

__global__ void k_zero_complex(Complex* __restrict__ arr, int count)
{
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < count;
         tid += gridDim.x * blockDim.x)
    {
        arr[tid].x = 0.0;
        arr[tid].y = 0.0;
    }
}

/* ================================================================
 *  §4  INITIALISATION KERNELS
 * ================================================================ */

/* ----------------------------------------------------------------
 *  Taylor–Green vortex:
 *     u =  sin(x) cos(y) cos(z)
 *     v = −cos(x) sin(y) cos(z)
 *     w =  0
 *
 *  ∇ · u = cos(x)cos(y)cos(z) − cos(x)cos(y)cos(z) = 0  ✓
 * ---------------------------------------------------------------- */
__global__ void k_init_taylor_green(Real* __restrict__ u,
                                     Real* __restrict__ v,
                                     Real* __restrict__ w,
                                     int Nx, int ny_x, int nz_x,
                                     int start_y, int start_z, Real dx_v)
{
    const int total = nz_x * ny_x * Nx;
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < total;
         tid += gridDim.x * blockDim.x)
    {
        int ix = tid % Nx;
        int iy = (tid / Nx) % ny_x;
        int iz = tid / (Nx * ny_x);

        Real x = ix * dx_v;
        Real y = (start_y + iy) * dx_v;
        Real z = (start_z + iz) * dx_v;

        u[tid] =  sin(x) * cos(y) * cos(z);
        v[tid] = -cos(x) * sin(y) * cos(z);
        w[tid] =  0.0;
    }
}

__global__ void k_init_shear_layer(Real* __restrict__ u,
                                   Real* __restrict__ v,
                                   Real* __restrict__ w,
                                   int Nx, int ny_x, int nz_x,
                                   int start_y, int start_z, Real dx_v)
{
    const int total = nz_x * ny_x * Nx;
    const double rho_shear = 30.0;
    const double delta = 0.05;

    for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < total;
         tid += gridDim.x * blockDim.x)
    {
        int ix = tid % Nx;
        int iy = (tid / Nx) % ny_x;
        int iz = tid / (Nx * ny_x);

        Real x = ix * dx_v;
        Real y = (start_y + iy) * dx_v;
        Real z = (start_z + iz) * dx_v;

        if (y <= M_PI) {
            u[tid] = tanh(rho_shear * (y - M_PI / 2.0));
        } else {
            u[tid] = tanh(rho_shear * (3.0 * M_PI / 2.0 - y));
        }
        v[tid] = delta * sin(x) * sin(z);
        w[tid] = 0.0;
    }
}

__device__ inline double hash_rand_dev(int ix, int iy, int iz, int c) {
    unsigned int n = (unsigned int)ix + ((unsigned int)iy << 10) + ((unsigned int)iz << 20) + ((unsigned int)c << 30);
    n = (n << 13) ^ n;
    n = (n * (n * n * 15731 + 789221) + 1376312589);
    return 2.0 * ((double)(n & 0x7fffffff) / 2147483647.0) - 1.0;
}

__global__ void k_init_random(Real* __restrict__ u,
                              Real* __restrict__ v,
                              Real* __restrict__ w,
                              int Nx, int ny_x, int nz_x,
                              int start_y, int start_z)
{
    const int total = nz_x * ny_x * Nx;
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < total;
         tid += gridDim.x * blockDim.x)
    {
        int ix = tid % Nx;
        int iy = (tid / Nx) % ny_x;
        int iz = tid / (Nx * ny_x);

        int gy = start_y + iy;
        int gz = start_z + iz;

        u[tid] = hash_rand_dev(ix, gy, gz, 0);
        v[tid] = hash_rand_dev(ix, gy, gz, 1);
        w[tid] = hash_rand_dev(ix, gy, gz, 2);
    }
}

/* ----------------------------------------------------------------
 *  Local kinetic energy:  E_loc = 0.5 Σ (u²+v²+w²)
 *
 *  Two-pass reduction:  partial block sums → host finalisation.
 * ---------------------------------------------------------------- */
__global__ void k_energy_reduce(const Real* __restrict__ u,
                                 const Real* __restrict__ v,
                                 const Real* __restrict__ w,
                                 Real* __restrict__ partial,
                                 int count)
{
    extern __shared__ Real smem[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    Real val = 0.0;
    if (gid < count) {
        Real uu = u[gid], vv = v[gid], ww = w[gid];
        val = 0.5 * (uu * uu + vv * vv + ww * ww);
    }
    smem[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    if (tid == 0) partial[blockIdx.x] = smem[0];
}

/* ================================================================
 *  §5  HOST-CALLABLE WRAPPER IMPLEMENTATIONS
 * ================================================================ */

/* ---- transpose wrappers ---- */

void pack_x_to_y(const Complex* src, Complex* buf,
                 int Nx, int ny_x, int nx_y, int nz_x, int Py) {
    int total = Py * nz_x * ny_x * nx_y;
    k_pack_x_to_y<<<grid_size(total), BLOCK_SIZE>>>(src, buf, Nx, ny_x, nx_y, nz_x, Py);
}

void unpack_x_to_y(const Complex* buf, Complex* dst,
                   int Ny, int ny_x, int nx_y, int nz_x, int Py) {
    int total = Py * nz_x * ny_x * nx_y;
    k_unpack_x_to_y<<<grid_size(total), BLOCK_SIZE>>>(buf, dst, Ny, ny_x, nx_y, nz_x, Py);
}

void pack_y_to_z(const Complex* src, Complex* buf,
                 int Ny, int nx_y, int ny_z, int nz_y, int Pz) {
    int total = Pz * nz_y * nx_y * ny_z;
    k_pack_y_to_z<<<grid_size(total), BLOCK_SIZE>>>(src, buf, Ny, nx_y, ny_z, nz_y, Pz);
}

void unpack_y_to_z(const Complex* buf, Complex* dst,
                   int Nz, int nx_y, int ny_z, int nz_y, int Pz) {
    int total = Pz * nz_y * nx_y * ny_z;
    k_unpack_y_to_z<<<grid_size(total), BLOCK_SIZE>>>(buf, dst, Nz, nx_y, ny_z, nz_y, Pz);
}

void pack_z_to_y(const Complex* src, Complex* buf,
                 int Nz, int nx_z, int ny_z, int nz_y, int Pz) {
    int total = Pz * ny_z * nx_z * nz_y;
    k_pack_z_to_y<<<grid_size(total), BLOCK_SIZE>>>(src, buf, Nz, nx_z, ny_z, nz_y, Pz);
}

void unpack_z_to_y(const Complex* buf, Complex* dst,
                   int Ny, int nx_y, int ny_z, int nz_y, int Pz) {
    int total = Pz * ny_z * nx_y * nz_y;
    k_unpack_z_to_y<<<grid_size(total), BLOCK_SIZE>>>(buf, dst, Ny, nx_y, ny_z, nz_y, Pz);
}

void pack_y_to_x(const Complex* src, Complex* buf,
                 int Ny, int nx_y, int ny_x, int nz_y, int Py) {
    int total = Py * nz_y * nx_y * ny_x;
    k_pack_y_to_x<<<grid_size(total), BLOCK_SIZE>>>(src, buf, Ny, nx_y, ny_x, nz_y, Py);
}

void unpack_y_to_x(const Complex* buf, Complex* dst,
                   int Nx, int nx_y, int ny_x, int nz_x, int Py) {
    int total = Py * nz_x * nx_y * ny_x;
    k_unpack_y_to_x<<<grid_size(total), BLOCK_SIZE>>>(buf, dst, Nx, nx_y, ny_x, nz_x, Py);
}

/* ---- spectral wrappers ---- */

void spectral_derivative(const Complex* src, Complex* dst,
                         const int* kx, const int* ky, const int* kz,
                         int dir, int ny_z, int nx_z, int Nz) {
    int total = ny_z * nx_z * Nz;
    k_spectral_deriv<<<grid_size(total), BLOCK_SIZE>>>(src, dst, kx, ky, kz, dir, ny_z, nx_z, Nz);
}

void dealias(Complex* data, const int* kx, const int* ky, const int* kz,
             int kcut_sq, int ny_z, int nx_z, int Nz) {
    int total = ny_z * nx_z * Nz;
    k_dealias<<<grid_size(total), BLOCK_SIZE>>>(data, kx, ky, kz, kcut_sq, ny_z, nx_z, Nz);
}

void pressure_poisson(Complex* p_hat,
                      const Complex* u_hat, const Complex* v_hat, const Complex* w_hat,
                      const int* kx, const int* ky, const int* kz,
                      Real rho_v, Real dt_v, int ny_z, int nx_z, int Nz) {
    int total = ny_z * nx_z * Nz;
    k_pressure_poisson<<<grid_size(total), BLOCK_SIZE>>>(
        p_hat, u_hat, v_hat, w_hat, kx, ky, kz, rho_v, dt_v, ny_z, nx_z, Nz);
}

void project(Complex* u_hat, Complex* v_hat, Complex* w_hat,
             const int* kx, const int* ky, const int* kz,
             int ny_z, int nx_z, int Nz) {
    int total = ny_z * nx_z * Nz;
    k_project<<<grid_size(total), BLOCK_SIZE>>>(u_hat, v_hat, w_hat, kx, ky, kz, ny_z, nx_z, Nz);
}

void rk3_substep(Complex* u_hat, Complex* Q, const Complex* nl_hat,
                 const int* kx, const int* ky, const int* kz,
                 Real A, Real B, Real dt_v, Real nu_v,
                 int ny_z, int nx_z, int Nz) {
    int total = ny_z * nx_z * Nz;
    k_rk3_substep<<<grid_size(total), BLOCK_SIZE>>>(
        u_hat, Q, nl_hat, kx, ky, kz, A, B, dt_v, nu_v, ny_z, nx_z, Nz);
}

/* ---- physical-space wrappers ---- */

void accum_product(Real* nl, const Real* u_j, const Complex* work_x,
                   Real norm, int count) {
    k_accum_product<<<grid_size(count), BLOCK_SIZE>>>(nl, u_j, work_x, norm, count);
}

void zero_real(Real* arr, int count) {
    k_zero_real<<<grid_size(count), BLOCK_SIZE>>>(arr, count);
}

void real_to_complex(const Real* src, Complex* dst, int count) {
    k_real_to_complex<<<grid_size(count), BLOCK_SIZE>>>(src, dst, count);
}

void complex_to_real(const Complex* src, Real* dst, Real norm, int count) {
    k_complex_to_real<<<grid_size(count), BLOCK_SIZE>>>(src, dst, norm, count);
}

void zero_complex(Complex* arr, int count) {
    k_zero_complex<<<grid_size(count), BLOCK_SIZE>>>(arr, count);
}

/* ---- initialisation / diagnostics wrappers ---- */

void init_taylor_green(Real* u, Real* v, Real* w,
                       int Nx, int ny_x, int nz_x,
                       int start_y, int start_z, Real dx_v) {
    int total = nz_x * ny_x * Nx;
    k_init_taylor_green<<<grid_size(total), BLOCK_SIZE>>>(
        u, v, w, Nx, ny_x, nz_x, start_y, start_z, dx_v);
}

void init_shear_layer(Real* u, Real* v, Real* w,
                      int Nx, int ny_x, int nz_x,
                      int start_y, int start_z, Real dx_v) {
    int total = nz_x * ny_x * Nx;
    k_init_shear_layer<<<grid_size(total), BLOCK_SIZE>>>(
        u, v, w, Nx, ny_x, nz_x, start_y, start_z, dx_v);
}

void init_random(Real* u, Real* v, Real* w,
                 int Nx, int ny_x, int nz_x,
                 int start_y, int start_z) {
    int total = nz_x * ny_x * Nx;
    k_init_random<<<grid_size(total), BLOCK_SIZE>>>(
        u, v, w, Nx, ny_x, nz_x, start_y, start_z);
}

Real local_kinetic_energy(const Real* u, const Real* v, const Real* w,
                          int count) {
    int nblocks = grid_size(count);
    Real* d_partial = nullptr;
    CUDA_CHECK(cudaMalloc(&d_partial, nblocks * sizeof(Real)));

    k_energy_reduce<<<nblocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(Real)>>>(
        u, v, w, d_partial, count);

    /* second-pass reduction on host (nblocks is small) */
    std::vector<Real> h_partial(nblocks);
    CUDA_CHECK(cudaMemcpy(h_partial.data(), d_partial,
                           nblocks * sizeof(Real), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_partial));

    Real sum = 0.0;
    for (int i = 0; i < nblocks; ++i) sum += h_partial[i];
    return sum;
}

} }  /* namespace ns3d::kernels */
