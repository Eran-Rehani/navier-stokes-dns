# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A pedagogical DNS of the incompressible Navier–Stokes equations on a periodic
`[0, 2π]` box. Three layers, all sharing one numerical method:

- **2D explorer** (`viz/ns2d.py` + `viz/explorer.py`): pure-Python interactive solver with matplotlib sliders. The teaching tool — drag Reynolds/scenario, watch vorticity live.
- **3D CPU DNS** (root: `solver_cpu.cpp` + `cpu_kernels.hpp`): real pseudo-spectral solver, FFTW3 + OpenMP, single workstation. The default build. **No CUDA/MPI dependencies.**
- **HPC backend** (`hpc/`): the same algorithm scaled to multi-GPU via CUDA + cuFFT + CUDA-aware MPI with 2D pencil decomposition. Built separately; needs GPU hardware not present on the dev machine.

This is **not yet pushed**; the GitHub remote is `eran-rehani/navier-stokes-dns` (public).

## Numerical method (shared by all three)

`cpu_kernels.hpp`, `viz/ns2d.py`, and `hpc/cuda_kernels.cu` implement the same scheme:

- Spatial: complex-to-complex FFT, **Orszag 2/3** dealiasing (`k_cut = N/3`).
- Temporal: **Williamson low-storage RK3** (`RK_A`, `RK_B`); persistent `Q` register per component.
- Incompressibility: exact **Leray projection** after each RK sub-step.
- Boundaries: **volume penalization** (Brinkman) — `nl += mask*(u - target)/eta` drags velocity to a target inside solids/inlets. `target=0` → no-slip wall; non-zero → inlet. This is how the periodic spectral solver fakes non-periodic BCs.
- Nonlinear `(u·∇)u` is the only physical-space evaluation; rest is spectral.
- `N` is a **compile-time constant** (`NS3D_N`) in the C++ builds; a runtime arg in the 2D explorer.

When changing numerics, update **all three** (`cpu_kernels.hpp`, `viz/ns2d.py`, `hpc/cuda_kernels.cu`) to keep them consistent.

## Build & run

### CPU 3D DNS (default)
```sh
cmake -S . -B build -DNS3D_N=64        # needs FFTW3 + fftw3_omp + OpenMP
cmake --build build -j
./build/ns3d_cpu --ic <tg|shear|hit|cylinder|channel|jet|step> --nu 1e-3 --steps 1000
```
Flags: `--ic`, `--nu`, `--dt`, `--steps`, `--penalty <eta>`, `--u0 <inflow>`, `--force <channel drive>` (`--help`). Changing resolution requires reconfiguring with a new `-DNS3D_N`.

### 2D explorer
```sh
source venv/bin/activate          # or: pip install -r requirements.txt
python viz/explorer.py
```

### HPC backend
```sh
cd hpc && make -j4 CUDA_ARCH="-arch=sm_80" NS3D_N=256
mpirun -np <procs> ./ns3d_solver
```
Physics params are compile-time constants in `hpc/common.hpp`, not CLI flags.

## Output & visualization

`ns3d_cpu` writes `output/vel_%06d.bin` every `output_interval` steps. **Format**:
`int32 N`, then `float64 u[N³], v[N³], w[N³]` (full velocity, index order `[iz][iy][ix]`).
The old format stored only magnitude; `viz/fields.py:load()` handles both.

Kinetic energy printed every `diag_interval` steps is the primary correctness check.

`viz/` (Python, deps in `requirements.txt`):
- `fields.py` — shared loader + vorticity / energy-spectrum diagnostics.
- `plot_velocity.py` — speed + vorticity mid-plane slices (PNG per frame).
- `analyze.py` — energy spectrum `E(k)` (look for `k^-5/3`) + KE decay.
- `plot_3d_animation.py` / `plot_interactive.py` — pyvista volume render (GIF / live).

## Conventions

- Wavenumber indexing: `wavenumber(idx, N) = idx <= N/2 ? idx : idx-N`.
- The HPC build compiles `.cpp` with `nvcc -x cu`; `hpc/common.hpp` defines precision, constants, and `CUDA_CHECK`/`CUFFT_CHECK`/`MPI_CHECK` macros.
- `venv/` and `build/` are gitignored; `requirements.txt` reproduces the Python env.
