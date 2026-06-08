/* ================================================================
 *  solver.cpp — Primary Simulation Orchestrator
 *
 *  3D Pseudo-Spectral DNS of the Incompressible Navier-Stokes
 *  Equations on a tri-ply periodic domain  [0, 2π]³.
 *
 *  Spatial :  C2C FFT with Orszag 2/3 spherical dealiasing
 *  Temporal:  Williamson 3rd-order low-storage Runge-Kutta
 *  Coupling:  Exact Leray projection after each RK sub-step
 *
 *  Data layout:
 *    State lives in *spectral space* (Z-pencil orientation).
 *    Physical space is visited only during the evaluation of the
 *    nonlinear advection term  (u·∇)u.
 *
 *  Memory budget per rank (complex Z-pencil arrays):
 *    u_hat[3]   — velocity                (persistent)
 *    Q[3]       — RK3 registers           (persistent)
 *    work_z     — spectral work           (transient)
 *    work_y     — Y-pencil work           (transient)
 *    work_x     — X-pencil work           (transient)
 *    u_phys[3]  — physical velocity       (real, transient)
 *    nl_phys    — NL accumulator          (real, transient)
 * ================================================================ */

#include "common.hpp"
#include "decomposition.hpp"
#include "cuda_kernels.cuh"
#include <cstdio>
#include <cmath>
#include <vector>

using namespace ns3d;

/* ================================================================
 *  Forward / Inverse 3D FFT helpers
 *
 *  Forward:  physical (X-pencil, complex) → spectral (Z-pencil)
 *            FFT-X → transpose X→Y → FFT-Y → transpose Y→Z → FFT-Z
 *
 *  Inverse:  spectral (Z-pencil) → physical (X-pencil, complex)
 *            IFFT-Z → transpose Z→Y → IFFT-Y → transpose Y→X → IFFT-X
 * ================================================================ */

static void forward_fft_3d(Complex*             work_x,
                           Complex*             work_y,
                           Complex*             work_z,
                           cufftHandle           plan_x,
                           cufftHandle           plan_y,
                           cufftHandle           plan_z,
                           PencilDecomposition&  decomp)
{
    CUFFT_CHECK(cufftExecZ2Z(plan_x, work_x, work_x, CUFFT_FORWARD));
    decomp.transpose_x_to_y(work_x, work_y);
    CUFFT_CHECK(cufftExecZ2Z(plan_y, work_y, work_y, CUFFT_FORWARD));
    decomp.transpose_y_to_z(work_y, work_z);
    CUFFT_CHECK(cufftExecZ2Z(plan_z, work_z, work_z, CUFFT_FORWARD));
}

static void inverse_fft_3d(Complex*             work_z,
                           Complex*             work_y,
                           Complex*             work_x,
                           cufftHandle           plan_x,
                           cufftHandle           plan_y,
                           cufftHandle           plan_z,
                           PencilDecomposition&  decomp)
{
    CUFFT_CHECK(cufftExecZ2Z(plan_z, work_z, work_z, CUFFT_INVERSE));
    decomp.transpose_z_to_y(work_z, work_y);
    CUFFT_CHECK(cufftExecZ2Z(plan_y, work_y, work_y, CUFFT_INVERSE));
    decomp.transpose_y_to_x(work_y, work_x);
    CUFFT_CHECK(cufftExecZ2Z(plan_x, work_x, work_x, CUFFT_INVERSE));
}

/* ================================================================
 *  Nonlinear term evaluation
 *
 *  For velocity component  i ∈ {0,1,2}  compute:
 *
 *     NL_i  =  Σ_j  u_j · ∂u_i/∂x_j       (physical space)
 *
 *  then forward-FFT the result into  nl_hat  (Z-pencil, spectral).
 *
 *  The three physical velocities u_phys[] must already be populated.
 * ================================================================ */

static void compute_nonlinear(int                   comp,
                              Complex*              nl_hat,       /* out: Z-pencil */
                              const Complex* const* u_hat,        /* in:  Z-pencil */
                              const Real*    const* u_phys,       /* in:  X-pencil */
                              Real*                 nl_phys,      /* scratch: X-pencil */
                              Complex*              work_z,
                              Complex*              work_y,
                              Complex*              work_x,
                              const int*            d_kx,
                              const int*            d_ky,
                              const int*            d_kz,
                              cufftHandle           plan_x,
                              cufftHandle           plan_y,
                              cufftHandle           plan_z,
                              PencilDecomposition&  decomp)
{
    const int x_count = static_cast<int>(decomp.x_pencil_count);
    const Real inv_N3  = 1.0 / (static_cast<Real>(N) * N * N);

    /* Zero the physical-space NL accumulator */
    kernels::zero_real(nl_phys, x_count);

    for (int dir = 0; dir < 3; ++dir) {
        /* Compute  i·k_dir · û_comp  in spectral space (Z-pencil) */
        kernels::spectral_derivative(
            u_hat[comp], work_z, d_kx, d_ky, d_kz,
            dir, decomp.ny_z, decomp.nx_z, N);

        /* Dealias the derivative field */
        kernels::dealias(work_z, d_kx, d_ky, d_kz,
                         k_cut_sq, decomp.ny_z, decomp.nx_z, N);

        /* IFFT derivative to physical space */
        inverse_fft_3d(work_z, work_y, work_x,
                       plan_x, plan_y, plan_z, decomp);

        /* Accumulate:  nl_phys[i] += u_phys[dir][i] · Re(work_x[i]) / N³ */
        kernels::accum_product(nl_phys, u_phys[dir], work_x, inv_N3, x_count);
    }

    /* Forward FFT the accumulated NL term to spectral space */
    kernels::real_to_complex(nl_phys, work_x, x_count);
    forward_fft_3d(work_x, work_y, work_z, plan_x, plan_y, plan_z, decomp);

    /* Copy result into nl_hat output buffer */
    CUDA_CHECK(cudaMemcpy(nl_hat, work_z,
                           decomp.z_pencil_count * sizeof(Complex),
                           cudaMemcpyDeviceToDevice));
}

/* ================================================================
 *  Diagnostics:  kinetic energy, max divergence
 * ================================================================ */

static Real compute_total_energy(const Real* const* u_phys,
                                 int local_count, MPI_Comm comm)
{
    Real E_local = kernels::local_kinetic_energy(
        u_phys[0], u_phys[1], u_phys[2], local_count);

    Real E_global = 0.0;
    MPI_CHECK(MPI_Allreduce(&E_local, &E_global, 1,
                             MPI_DOUBLE, MPI_SUM, comm));

    return E_global / (static_cast<Real>(N) * N * N);
}

/* ================================================================
 *  Main entry point
 * ================================================================ */

int main(int argc, char** argv)
{
    MPI_CHECK(MPI_Init(&argc, &argv));

    /* ---- GPU binding: one GPU per rank (round-robin) ---- */
    int local_rank = 0;
    {
        MPI_Comm local_comm;
        MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
                                       0, MPI_INFO_NULL, &local_comm));
        MPI_CHECK(MPI_Comm_rank(local_comm, &local_rank));
        MPI_Comm_free(&local_comm);
    }
    int n_devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&n_devices));
    CUDA_CHECK(cudaSetDevice(local_rank % n_devices));

    /* ---- Decomposition ---- */
    PencilDecomposition decomp;
    const int rank = decomp.world_rank;

    const int x_count = static_cast<int>(decomp.x_pencil_count);
    const int y_count = static_cast<int>(decomp.y_pencil_count);
    const int z_count = static_cast<int>(decomp.z_pencil_count);

    /* ================================================================
     *  Wavenumber look-up arrays (device)
     *
     *  kx_arr[ix]  =  wavenumber( start_x_z + ix, N )   for ix ∈ [0, nx_z)
     *  ky_arr[iy]  =  wavenumber( start_y_z + iy, N )   for iy ∈ [0, ny_z)
     *  kz_arr[iz]  =  wavenumber( iz, N )                for iz ∈ [0, N)
     * ================================================================ */
    int* d_kx = nullptr;
    int* d_ky = nullptr;
    int* d_kz = nullptr;
    {
        std::vector<int> h_kx(decomp.nx_z), h_ky(decomp.ny_z), h_kz(N);

        for (int i = 0; i < decomp.nx_z; ++i)
            h_kx[i] = wavenumber(decomp.start_x_z + i, N);
        for (int i = 0; i < decomp.ny_z; ++i)
            h_ky[i] = wavenumber(decomp.start_y_z + i, N);
        for (int i = 0; i < N; ++i)
            h_kz[i] = wavenumber(i, N);

        CUDA_CHECK(cudaMalloc(&d_kx, decomp.nx_z * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_ky, decomp.ny_z * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_kz, N * sizeof(int)));

        CUDA_CHECK(cudaMemcpy(d_kx, h_kx.data(), decomp.nx_z * sizeof(int),
                               cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ky, h_ky.data(), decomp.ny_z * sizeof(int),
                               cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_kz, h_kz.data(), N * sizeof(int),
                               cudaMemcpyHostToDevice));
    }

    /* ================================================================
     *  Device memory allocation
     * ================================================================ */

    /* Spectral state (Z-pencil) */
    Complex* d_u_hat[3];
    Complex* d_Q[3];
    Complex* d_nl_hat = nullptr;
    for (int c = 0; c < 3; ++c) {
        CUDA_CHECK(cudaMalloc(&d_u_hat[c], z_count * sizeof(Complex)));
        CUDA_CHECK(cudaMalloc(&d_Q[c],     z_count * sizeof(Complex)));
        kernels::zero_complex(d_Q[c], z_count);
    }
    CUDA_CHECK(cudaMalloc(&d_nl_hat, z_count * sizeof(Complex)));

    /* Physical-space buffers (X-pencil, real) */
    Real* d_u_phys[3];
    Real* d_nl_phys = nullptr;
    for (int c = 0; c < 3; ++c)
        CUDA_CHECK(cudaMalloc(&d_u_phys[c], x_count * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_nl_phys, x_count * sizeof(Real)));

    /* FFT work buffers (complex) */
    Complex* d_work_x = nullptr;
    Complex* d_work_y = nullptr;
    Complex* d_work_z = nullptr;
    CUDA_CHECK(cudaMalloc(&d_work_x, x_count * sizeof(Complex)));
    CUDA_CHECK(cudaMalloc(&d_work_y, y_count * sizeof(Complex)));
    CUDA_CHECK(cudaMalloc(&d_work_z, z_count * sizeof(Complex)));

    /* ================================================================
     *  cuFFT Plan creation   (batched 1D C2C transforms)
     *
     *  For each pencil, the FFT dimension is the contiguous
     *  (fastest-varying) axis.  The batch count is the product of
     *  the two non-FFT local dimensions.
     *
     *  cufftPlanMany(plan, rank=1, n, inembed, istride, idist,
     *                                 onembed, ostride, odist, type, batch)
     * ================================================================ */
    cufftHandle plan_x, plan_y, plan_z;
    {
        int n_x[1] = { N };
        int batch_x = decomp.nz_x * decomp.ny_x;
        CUFFT_CHECK(cufftPlanMany(&plan_x, 1, n_x,
                                   nullptr, 1, N,
                                   nullptr, 1, N,
                                   CUFFT_C2C_TYPE, batch_x));

        int n_y[1] = { N };
        int batch_y = decomp.nz_y * decomp.nx_y;
        CUFFT_CHECK(cufftPlanMany(&plan_y, 1, n_y,
                                   nullptr, 1, N,
                                   nullptr, 1, N,
                                   CUFFT_C2C_TYPE, batch_y));

        int n_z[1] = { N };
        int batch_z = decomp.ny_z * decomp.nx_z;
        CUFFT_CHECK(cufftPlanMany(&plan_z, 1, n_z,
                                   nullptr, 1, N,
                                   nullptr, 1, N,
                                   CUFFT_C2C_TYPE, batch_z));
    }

    /* ================================================================
     *  Initial condition:  Taylor-Green vortex in physical space
     *  then forward-FFT to spectral state.
     * ================================================================ */
    {
        kernels::init_taylor_green(d_u_phys[0], d_u_phys[1], d_u_phys[2],
                                   N, decomp.ny_x, decomp.nz_x,
                                   decomp.start_y_x, decomp.start_z_x, dx);
        CUDA_CHECK(cudaDeviceSynchronize());

        for (int c = 0; c < 3; ++c) {
            kernels::real_to_complex(d_u_phys[c], d_work_x, x_count);
            forward_fft_3d(d_work_x, d_work_y, d_work_z,
                           plan_x, plan_y, plan_z, decomp);
            CUDA_CHECK(cudaMemcpy(d_u_hat[c], d_work_z,
                                   z_count * sizeof(Complex),
                                   cudaMemcpyDeviceToDevice));
        }

        /* Initial projection to machine-precision divergence-free */
        kernels::project(d_u_hat[0], d_u_hat[1], d_u_hat[2],
                         d_kx, d_ky, d_kz,
                         decomp.ny_z, decomp.nx_z, N);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    /* ================================================================
     *  Initial diagnostics
     * ================================================================ */
    {
        /* IFFT back for energy measurement */
        const Real inv_N3 = 1.0 / (static_cast<Real>(N) * N * N);
        for (int c = 0; c < 3; ++c) {
            CUDA_CHECK(cudaMemcpy(d_work_z, d_u_hat[c],
                                   z_count * sizeof(Complex),
                                   cudaMemcpyDeviceToDevice));
            inverse_fft_3d(d_work_z, d_work_y, d_work_x,
                           plan_x, plan_y, plan_z, decomp);
            kernels::complex_to_real(d_work_x, d_u_phys[c], inv_N3, x_count);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        Real E0 = compute_total_energy(
            const_cast<const Real**>(d_u_phys), x_count, MPI_COMM_WORLD);

        if (rank == 0) {
            std::printf("  t = %10.6f   E_kin = %18.14e\n", 0.0, E0);
            std::fflush(stdout);
        }
    }

    /* ================================================================
     *  TIME-STEPPING LOOP
     *
     *  For each time step n = 0 … num_steps−1:
     *    For each RK3 sub-step s = 0, 1, 2:
     *
     *      1.  IFFT  û → u  (physical space, dealiased)
     *      2.  Evaluate  NL_i = Σ_j u_j ∂u_i/∂x_j
     *      3.  FFT  NL → NL̂
     *      4.  RK3 update:  Q = A·Q + Δt·(−NL̂ − ν|k|²û)
     *                       û += B·Q
     *      5.  Leray projection:  û ← P û
     * ================================================================ */

    for (int step = 0; step < num_steps; ++step) {

        for (int s = 0; s < RK_STAGES; ++s) {

            /* ---- 1. IFFT velocities to physical space ---- */
            const Real inv_N3 = 1.0 / (static_cast<Real>(N) * N * N);
            for (int c = 0; c < 3; ++c) {
                /* Copy spectral field and dealias */
                CUDA_CHECK(cudaMemcpy(d_work_z, d_u_hat[c],
                                       z_count * sizeof(Complex),
                                       cudaMemcpyDeviceToDevice));
                kernels::dealias(d_work_z, d_kx, d_ky, d_kz,
                                 k_cut_sq, decomp.ny_z, decomp.nx_z, N);

                /* IFFT Z → Y → X */
                inverse_fft_3d(d_work_z, d_work_y, d_work_x,
                               plan_x, plan_y, plan_z, decomp);

                /* Extract real part with 1/N³ normalisation */
                kernels::complex_to_real(d_work_x, d_u_phys[c], inv_N3, x_count);
            }

            /* ---- 2–4. Nonlinear term + RK3 update per component ---- */
            for (int comp = 0; comp < 3; ++comp) {
                compute_nonlinear(
                    comp, d_nl_hat,
                    const_cast<const Complex* const*>(d_u_hat),
                    const_cast<const Real* const*>(d_u_phys),
                    d_nl_phys,
                    d_work_z, d_work_y, d_work_x,
                    d_kx, d_ky, d_kz,
                    plan_x, plan_y, plan_z, decomp);

                kernels::rk3_substep(
                    d_u_hat[comp], d_Q[comp], d_nl_hat,
                    d_kx, d_ky, d_kz,
                    RK_A[s], RK_B[s], dt, nu,
                    decomp.ny_z, decomp.nx_z, N);
            }

            /* ---- 5. Divergence-free projection ---- */
            kernels::project(d_u_hat[0], d_u_hat[1], d_u_hat[2],
                             d_kx, d_ky, d_kz,
                             decomp.ny_z, decomp.nx_z, N);

            CUDA_CHECK(cudaDeviceSynchronize());

        }  /* end RK sub-steps */

        /* ---- Diagnostics ---- */
        if ((step + 1) % diag_interval == 0 || step == 0) {
            const Real inv_N3 = 1.0 / (static_cast<Real>(N) * N * N);
            for (int c = 0; c < 3; ++c) {
                CUDA_CHECK(cudaMemcpy(d_work_z, d_u_hat[c],
                                       z_count * sizeof(Complex),
                                       cudaMemcpyDeviceToDevice));
                inverse_fft_3d(d_work_z, d_work_y, d_work_x,
                               plan_x, plan_y, plan_z, decomp);
                kernels::complex_to_real(d_work_x, d_u_phys[c], inv_N3, x_count);
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            Real Ek = compute_total_energy(
                const_cast<const Real**>(d_u_phys), x_count, MPI_COMM_WORLD);

            if (rank == 0) {
                std::printf("  t = %10.6f   E_kin = %18.14e\n",
                            (step + 1) * dt, Ek);
                std::fflush(stdout);
            }
        }

    }  /* end time steps */

    /* ================================================================
     *  Cleanup
     * ================================================================ */
    cufftDestroy(plan_x);
    cufftDestroy(plan_y);
    cufftDestroy(plan_z);

    for (int c = 0; c < 3; ++c) {
        cudaFree(d_u_hat[c]);
        cudaFree(d_Q[c]);
        cudaFree(d_u_phys[c]);
    }
    cudaFree(d_nl_hat);
    cudaFree(d_nl_phys);
    cudaFree(d_work_x);
    cudaFree(d_work_y);
    cudaFree(d_work_z);
    cudaFree(d_kx);
    cudaFree(d_ky);
    cudaFree(d_kz);

    MPI_CHECK(MPI_Finalize());
    return 0;
}
