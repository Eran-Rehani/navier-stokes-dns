# HPC backend — multi-GPU CUDA + MPI DNS

The distributed version of the spectral Navier–Stokes solver from the project
root, scaled to **multiple GPUs** across nodes. Same algorithm (Orszag 2/3
dealiasing, Williamson RK3, Leray projection); the difference is how the FFTs
and data are parallelised.

> This backend needs CUDA-aware MPI and GPU hardware. The root project is the
> CPU-only build that runs on an ordinary workstation — start there if you just
> want to explore the physics.

## Architecture

- **Precision / types / constants** live in `common.hpp` (`Real=double`,
  `Complex=cufftDoubleComplex`). Physical and numerical parameters (`nu`, `dt`,
  `num_steps`, grid `N`) are **compile-time constants** here, not CLI flags.
- **`cuda_kernels.cu/.cuh`** — device kernels: spectral derivatives, dealiasing,
  projection, pressure Poisson solve, the RK3 sub-step, and transpose pack/unpack.
- **`decomposition.cpp/.hpp`** — 2D *pencil* domain decomposition over an MPI
  process grid `Py × Pz`. The global 3D FFT is done one axis at a time with
  `MPI_Alltoall` transposes between X-, Y- and Z-pencil orientations.
- **`solver.cpp`** — orchestrates the run; state lives in spectral space
  (Z-pencil), physical space is visited only for the nonlinear term.

### Data flow (forward FFT)

```
FFT-X  →  transpose X→Y  →  FFT-Y  →  transpose Y→Z  →  FFT-Z
```

The inverse reverses it. Transposes pack data on the GPU, exchange via
CUDA-aware MPI, then unpack.

## Build

Requires the CUDA Toolkit (nvcc, cuFFT ≥ 11.0) and a CUDA-aware MPI
(OpenMPI ≥ 4.1 or MVAPICH2-GDR).

```bash
make -j4 CUDA_ARCH="-arch=sm_80" NS3D_N=256      # builds ./ns3d_solver
mpirun -np <nprocs> ./ns3d_solver
```

Override `CUDA_ARCH` to match your GPU (e.g. `sm_90` for Hopper) and `NS3D_N`
for the grid resolution. Edit `common.hpp` to change physics parameters.

## Relationship to the CPU build

`cpu_kernels.hpp` (in the project root) is the serial/OpenMP twin of
`cuda_kernels.cu`. When changing the numerical method, update **both** so the
two backends stay consistent.
