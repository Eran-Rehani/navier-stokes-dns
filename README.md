# Navier–Stokes DNS — a pedagogical fluid solver

A small, readable **direct numerical simulation (DNS)** of the incompressible
Navier–Stokes equations, built to *understand* the equations by watching the
fluid move. It comes in three layers:

| Layer | What it is | Where |
|-------|------------|-------|
| 🎮 **2D explorer** | Interactive Python app — drag sliders, switch boundary conditions, watch vorticity evolve live | `viz/explorer.py` |
| 💻 **3D CPU solver** | A real pseudo-spectral DNS in C++, runs on any multi-core machine | root (`solver_cpu.cpp`) |
| 🖥️ **HPC backend** | The same method scaled to multiple GPUs with CUDA + MPI | [`hpc/`](hpc/) |

The 2D explorer and the C++ solver use the **same numerical method**, so the
intuition you build by playing transfers directly to the serious runs.

---

## The equations

For an incompressible Newtonian fluid of constant density,

$$
\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u}\cdot\nabla)\mathbf{u}
   = -\frac{1}{\rho}\nabla p + \nu\,\nabla^2 \mathbf{u},
\qquad \nabla\cdot\mathbf{u} = 0 .
$$

- $(\mathbf{u}\cdot\nabla)\mathbf{u}$ — **advection**: the fluid carries itself (the nonlinear term, source of turbulence).
- $\nu\nabla^2\mathbf{u}$ — **diffusion**: viscosity smooths the flow.
- $\nabla p$ + $\nabla\cdot\mathbf{u}=0$ — **pressure** enforces incompressibility.

The single most important knob is the **Reynolds number** $\mathrm{Re}=UL/\nu$:
small Re is smooth laminar flow, large Re is turbulence. In the explorer, the
`Reynolds` slider sets $\nu$ directly.

---

## Quick start

### 1. Interactive 2D explorer (start here)

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python viz/explorer.py
```

A window opens. Pick a scenario, drag the **Reynolds** slider, and watch the
vorticity field evolve in real time. Try `cylinder` at Re ≈ 200 to see a von
Kármán vortex street form behind the obstacle.

### 2. The 3D CPU DNS

Needs a C++17 compiler, **FFTW3** (with the OpenMP variant) and **OpenMP**.

```bash
cmake -S . -B build -DNS3D_N=64       # grid resolution is a compile-time constant
cmake --build build -j
./build/ns3d_cpu --ic cylinder --nu 1e-3 --steps 1000
```

Then visualise the output:

```bash
python viz/plot_velocity.py            # speed + vorticity slices (PNG per frame)
python viz/analyze.py                  # energy spectrum E(k) + energy decay
python viz/plot_3d_animation.py        # 3D volume-render GIF (pyvista)
python viz/plot_interactive.py         # interactive 3D viewer with a time slider
```

---

## Boundary conditions & scenarios

The solver is **pseudo-spectral**, which means the domain is a periodic box —
the natural setting for Fourier methods. To show non-periodic effects (walls,
obstacles, inlets) we use **volume penalization** (the Brinkman immersed-boundary
method): inside a "solid" or "inlet" region the velocity is dragged toward a
target value on a short time-scale $\eta$,

$$ \frac{\partial \mathbf u}{\partial t} \mathrel{+}= \frac{\text{mask}}{\eta}\,(\mathbf u^\* - \mathbf u). $$

`target = 0` makes a no-slip wall; a non-zero target makes a velocity inlet.
This keeps the fast FFT solver while letting you *see* boundary effects.

| `--ic` | Scenario | What you learn |
|--------|----------|----------------|
| `tg` | Taylor–Green vortex | classic decaying benchmark; energy cascade |
| `shear` | double shear layer | Kelvin–Helmholtz roll-up |
| `hit` | random field | homogeneous decaying turbulence, $k^{-5/3}$ spectrum |
| `cylinder` | flow past a cylinder | von Kármán vortex street |
| `channel` | plane channel, no-slip walls | wall-bounded shear, driven by a body force |
| `jet` | jet through a slot | shear-layer instability, entrainment |
| `step` | backward-facing step | separation + recirculation bubble |

CPU solver flags: `--ic`, `--nu`, `--dt`, `--steps`, `--penalty <eta>`,
`--u0 <inflow speed>`, `--force <channel drive>` (`--help` for all).

---

## How it works (numerics)

Both the 2D explorer and the 3D DNS share the same scheme:

- **Space**: complex-to-complex FFT on a periodic $[0,2\pi]^d$ box.
- **Dealiasing**: Orszag 2/3 rule (zeroes the top third of wavenumbers).
- **Time**: Williamson low-storage 3rd-order Runge–Kutta.
- **Incompressibility**: exact Leray projection after every RK sub-step.
- **Boundaries**: volume penalization (above).

The nonlinear advection term is the only thing evaluated in physical space;
everything else stays spectral.

---

## Repository layout

```
.
├── solver_cpu.cpp      # 3D pseudo-spectral DNS (CPU, FFTW3 + OpenMP)
├── cpu_kernels.hpp     # the numerical kernels (read this to learn the method)
├── CMakeLists.txt      # CPU build
├── viz/
│   ├── ns2d.py         # compact 2D solver — the Python mirror of the C++ math
│   ├── explorer.py     # interactive sliders + live vorticity
│   ├── fields.py       # snapshot loader + vorticity/spectrum diagnostics
│   ├── plot_velocity.py, analyze.py, plot_3d_*.py
├── hpc/                # multi-GPU CUDA + MPI backend (see hpc/README.md)
└── requirements.txt
```

## HPC / multi-GPU backend

The `hpc/` directory holds the distributed version of the exact same algorithm:
CUDA kernels, cuFFT, and a 2D pencil domain decomposition over MPI for
multi-GPU clusters. It requires CUDA-aware MPI and is built with its own
`Makefile`. See [`hpc/README.md`](hpc/README.md). The root project deliberately
has **no GPU/MPI dependencies** so it runs on an ordinary workstation.

## License

MIT.
