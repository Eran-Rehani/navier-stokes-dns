#pragma once
/* ================================================================
 *  decomposition.hpp — 2D Pencil Domain Decomposition (MPI)
 *
 *  Process grid:  Py × Pz   (row-major:  rank = ry·Pz + rz)
 *
 *  Pencil orientations and local extents
 *  ──────────────────────────────────────────────────────────────
 *  X-pencil  [nz_x][ny_x][Nx]    X contiguous   (FFT-X)
 *  Y-pencil  [nz_y][nx_y][Ny]    Y contiguous   (FFT-Y)
 *  Z-pencil  [ny_z][nx_z][Nz]    Z contiguous   (FFT-Z)
 *
 *  nz_x = nz_y = Nz/Pz          (Z distributed by Pz)
 *  ny_x = Ny/Py                  (Y distributed by Py in X-pencil)
 *  nx_y = nx_z = Nx/Py           (X distributed by Py in Y/Z-pencils)
 *  ny_z = Ny/Pz                  (Y distributed by Pz in Z-pencil)
 *
 *  Transposes:
 *    X-pencil ↔ Y-pencil   via  MPI_Alltoall  in  row_comm  (Py procs)
 *    Y-pencil ↔ Z-pencil   via  MPI_Alltoall  in  col_comm  (Pz procs)
 * ================================================================ */

#include "common.hpp"

namespace ns3d {

class PencilDecomposition {
public:
    /* ---- MPI topology ---- */
    int world_rank, world_size;
    int Py, Pz;                        /* process-grid dimensions           */
    int ry, rz;                        /* coords in the Py × Pz grid       */
    MPI_Comm row_comm;                 /* Py procs sharing the same rz      */
    MPI_Comm col_comm;                 /* Pz procs sharing the same ry      */

    /* ---- X-pencil local sizes ---- */
    int ny_x;                          /* = Ny / Py                         */
    int nz_x;                          /* = Nz / Pz                         */
    int start_y_x;                     /* global y-offset                   */
    int start_z_x;                     /* global z-offset                   */
    size_t x_pencil_count;             /* ny_x * nz_x * Nx  (complex elts) */

    /* ---- Y-pencil local sizes ---- */
    int nx_y;                          /* = Nx / Py                         */
    int nz_y;                          /* = Nz / Pz  (same as nz_x)        */
    int start_x_y;                     /* global x-offset                   */
    size_t y_pencil_count;             /* nz_y * nx_y * Ny                  */

    /* ---- Z-pencil local sizes ---- */
    int nx_z;                          /* = Nx / Py  (same as nx_y)         */
    int ny_z;                          /* = Ny / Pz                         */
    int start_x_z;                     /* global x-offset (same as start_x_y) */
    int start_y_z;                     /* global y-offset                   */
    size_t z_pencil_count;             /* ny_z * nx_z * Nz                  */

    /* ---- Transpose chunk sizes (elements per peer) ---- */
    size_t chunk_xy;                   /* nz_x * ny_x * nx_y               */
    size_t chunk_yz;                   /* nz_y * nx_y * ny_z               */

    /* ---- Device send / recv buffers ---- */
    Complex* d_send;
    Complex* d_recv;

    /* ---- Lifecycle ---- */
    PencilDecomposition();
    ~PencilDecomposition();

    /* ---- Forward transposes (physical → spectral direction) ---- */
    void transpose_x_to_y(const Complex* d_src, Complex* d_dst);
    void transpose_y_to_z(const Complex* d_src, Complex* d_dst);

    /* ---- Backward transposes (spectral → physical direction) ---- */
    void transpose_z_to_y(const Complex* d_src, Complex* d_dst);
    void transpose_y_to_x(const Complex* d_src, Complex* d_dst);

private:
    void factor_procs();               /* find best Py × Pz factoring      */
    void build_comms();                /* create row / col communicators    */
    void compute_local_sizes();        /* derive all local extents          */
    void alloc_buffers();              /* allocate d_send / d_recv on GPU   */
};

}  /* namespace ns3d */
