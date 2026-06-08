/* ================================================================
 *  decomposition.cpp — 2D Pencil Decomposition Implementation
 *
 *  Manages MPI process grid, communicator splitting, local-extent
 *  computation, and global transposes between pencil orientations.
 *
 *  Transpose data path (GPU-resident throughout):
 *    1. CUDA pack kernel  →  contiguous send buffer on device
 *    2. MPI_Alltoall  (CUDA-aware / GPUDirect RDMA)
 *    3. CUDA unpack kernel  →  destination pencil layout on device
 * ================================================================ */

#include "decomposition.hpp"
#include "cuda_kernels.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>

namespace ns3d {

/* ----------------------------------------------------------------
 *  Constructor — call AFTER MPI_Init
 * ---------------------------------------------------------------- */
PencilDecomposition::PencilDecomposition()
{
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));

    factor_procs();
    build_comms();
    compute_local_sizes();
    alloc_buffers();

    if (world_rank == 0) {
        std::printf("──────────────────────────────────────────────\n");
        std::printf("  2D Pencil Decomposition  N = %d\n", N);
        std::printf("  Process grid:  Py = %d  ×  Pz = %d  = %d\n",
                    Py, Pz, world_size);
        std::printf("  X-pencil local:  %d × %d × %d\n", nz_x, ny_x, N);
        std::printf("  Y-pencil local:  %d × %d × %d\n", nz_y, nx_y, N);
        std::printf("  Z-pencil local:  %d × %d × %d\n", ny_z, nx_z, N);
        std::printf("  chunk_xy = %zu   chunk_yz = %zu\n", chunk_xy, chunk_yz);
        std::printf("──────────────────────────────────────────────\n");
        std::fflush(stdout);
    }
}

/* ----------------------------------------------------------------
 *  Destructor — free communicators and device buffers
 * ---------------------------------------------------------------- */
PencilDecomposition::~PencilDecomposition()
{
    if (d_send) cudaFree(d_send);
    if (d_recv) cudaFree(d_recv);
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
}

/* ----------------------------------------------------------------
 *  factor_procs — find Py × Pz closest to √P × √P that evenly
 *  divides N in both Y and Z dimensions.
 * ---------------------------------------------------------------- */
void PencilDecomposition::factor_procs()
{
    int best_py = 1, best_pz = world_size;

    for (int py = 1; py * py <= world_size; ++py) {
        if (world_size % py != 0) continue;
        int pz = world_size / py;
        if (N % py != 0 || N % pz != 0) continue;
        if (std::abs(py - pz) < std::abs(best_py - best_pz)) {
            best_py = py;
            best_pz = pz;
        }
    }

    Py = best_py;
    Pz = best_pz;

    if (N % Py != 0 || N % Pz != 0) {
        if (world_rank == 0)
            std::fprintf(stderr,
                "FATAL: N=%d is not divisible by Py=%d or Pz=%d\n", N, Py, Pz);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

/* ----------------------------------------------------------------
 *  build_comms — split MPI_COMM_WORLD into row and column
 *  communicators.  Row = same rz (Py members), Col = same ry.
 *
 *  Global rank layout (row-major):  rank = ry · Pz + rz
 * ---------------------------------------------------------------- */
void PencilDecomposition::build_comms()
{
    ry = world_rank / Pz;
    rz = world_rank % Pz;

    /* row_comm: all ranks with the same rz → Py members */
    MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, rz, ry, &row_comm));

    /* col_comm: all ranks with the same ry → Pz members */
    MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, ry, rz, &col_comm));
}

/* ----------------------------------------------------------------
 *  compute_local_sizes — derive per-rank extents for each pencil
 * ---------------------------------------------------------------- */
void PencilDecomposition::compute_local_sizes()
{
    /* X-pencil: full Nx, Y split by Py, Z split by Pz */
    ny_x     = N / Py;
    nz_x     = N / Pz;
    start_y_x = ry * ny_x;
    start_z_x = rz * nz_x;
    x_pencil_count = static_cast<size_t>(nz_x) * ny_x * N;

    /* Y-pencil: X split by Py, full Ny, Z split by Pz */
    nx_y     = N / Py;
    nz_y     = N / Pz;       /* same as nz_x */
    start_x_y = ry * nx_y;
    y_pencil_count = static_cast<size_t>(nz_y) * nx_y * N;

    /* Z-pencil: X split by Py, Y split by Pz, full Nz */
    nx_z     = N / Py;        /* same as nx_y */
    ny_z     = N / Pz;
    start_x_z = ry * nx_z;
    start_y_z = rz * ny_z;
    z_pencil_count = static_cast<size_t>(ny_z) * nx_z * N;

    /* Transpose chunk sizes (elements exchanged per peer) */
    chunk_xy = static_cast<size_t>(nz_x) * ny_x * nx_y;
    chunk_yz = static_cast<size_t>(nz_y) * nx_y * ny_z;
}

/* ----------------------------------------------------------------
 *  alloc_buffers — allocate send/recv on device, sized for the
 *  larger of the two transpose operations.
 * ---------------------------------------------------------------- */
void PencilDecomposition::alloc_buffers()
{
    size_t max_total = std::max(
        static_cast<size_t>(Py) * chunk_xy,
        static_cast<size_t>(Pz) * chunk_yz);

    CUDA_CHECK(cudaMalloc(&d_send, max_total * sizeof(Complex)));
    CUDA_CHECK(cudaMalloc(&d_recv, max_total * sizeof(Complex)));
}

/* ================================================================
 *  TRANSPOSE IMPLEMENTATIONS
 *
 *  Each transpose follows the same three-phase pattern:
 *    1. Pack   — CUDA kernel extracts non-contiguous slabs from
 *               the source pencil into a contiguous send buffer.
 *    2. Alltoall — CUDA-aware MPI exchanges equal-sized chunks
 *               between all peers in the sub-communicator.
 *    3. Unpack — CUDA kernel scatters received data into the
 *               destination pencil layout.
 *
 *  cudaDeviceSynchronize before MPI ensures the pack kernel has
 *  completed and the send buffer is coherent for DMA transfer.
 * ================================================================ */

/* ---- X-pencil → Y-pencil ---- */
void PencilDecomposition::transpose_x_to_y(const Complex* d_src, Complex* d_dst)
{
    kernels::pack_x_to_y(d_src, d_send, N, ny_x, nx_y, nz_x, Py);
    CUDA_CHECK(cudaDeviceSynchronize());

    MPI_CHECK(MPI_Alltoall(
        d_send, static_cast<int>(chunk_xy * 2), MPI_DOUBLE,
        d_recv, static_cast<int>(chunk_xy * 2), MPI_DOUBLE,
        row_comm));

    kernels::unpack_x_to_y(d_recv, d_dst, N, ny_x, nx_y, nz_x, Py);
    CUDA_CHECK(cudaDeviceSynchronize());
}

/* ---- Y-pencil → Z-pencil ---- */
void PencilDecomposition::transpose_y_to_z(const Complex* d_src, Complex* d_dst)
{
    kernels::pack_y_to_z(d_src, d_send, N, nx_y, ny_z, nz_y, Pz);
    CUDA_CHECK(cudaDeviceSynchronize());

    MPI_CHECK(MPI_Alltoall(
        d_send, static_cast<int>(chunk_yz * 2), MPI_DOUBLE,
        d_recv, static_cast<int>(chunk_yz * 2), MPI_DOUBLE,
        col_comm));

    kernels::unpack_y_to_z(d_recv, d_dst, N, nx_y, ny_z, nz_y, Pz);
    CUDA_CHECK(cudaDeviceSynchronize());
}

/* ---- Z-pencil → Y-pencil ---- */
void PencilDecomposition::transpose_z_to_y(const Complex* d_src, Complex* d_dst)
{
    kernels::pack_z_to_y(d_src, d_send, N, nx_z, ny_z, nz_y, Pz);
    CUDA_CHECK(cudaDeviceSynchronize());

    MPI_CHECK(MPI_Alltoall(
        d_send, static_cast<int>(chunk_yz * 2), MPI_DOUBLE,
        d_recv, static_cast<int>(chunk_yz * 2), MPI_DOUBLE,
        col_comm));

    kernels::unpack_z_to_y(d_recv, d_dst, N, nx_y, ny_z, nz_y, Pz);
    CUDA_CHECK(cudaDeviceSynchronize());
}

/* ---- Y-pencil → X-pencil ---- */
void PencilDecomposition::transpose_y_to_x(const Complex* d_src, Complex* d_dst)
{
    kernels::pack_y_to_x(d_src, d_send, N, nx_y, ny_x, nz_y, Py);
    CUDA_CHECK(cudaDeviceSynchronize());

    MPI_CHECK(MPI_Alltoall(
        d_send, static_cast<int>(chunk_xy * 2), MPI_DOUBLE,
        d_recv, static_cast<int>(chunk_xy * 2), MPI_DOUBLE,
        row_comm));

    kernels::unpack_y_to_x(d_recv, d_dst, N, nx_y, ny_x, nz_x, Py);
    CUDA_CHECK(cudaDeviceSynchronize());
}

}  /* namespace ns3d */
