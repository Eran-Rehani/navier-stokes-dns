# Navier–Stokes DNS

A pedagogical direct numerical simulation of the incompressible Navier–Stokes
equations. The language below is the shared vocabulary across the three
solver implementations and the teaching tools.

## Language

### Solvers & tools

**DNS** (Direct Numerical Simulation):
A simulation that resolves all scales of the flow directly, with no turbulence model.
_Avoid_: "CFD run", "simulation" (too vague).

**Backend**:
One concrete implementation of the shared numerical method. There are three: the **CPU DNS**, the **HPC backend**, and the **Explorer** solvers.
_Avoid_: "version", "engine".

**CPU DNS**:
The heavy, offline 3D solver in C++ (`solver_cpu.cpp`), run from the terminal and written to disk.
_Avoid_: "the C++ version".

**HPC backend**:
The multi-GPU CUDA + MPI implementation in `hpc/`, for clusters.

**Explorer**:
An interactive, real-time teaching tool that solves and renders simultaneously. The **2D Explorer** (matplotlib, fast) and the **3D Explorer** (pyvista, live low-resolution) are distinct.
_Avoid_: "viewer" (that implies replay of precomputed data, which the Explorer is not), "GUI".

**Solver** (Explorer-facing):
The live, in-process object an Explorer drives: it can step, reset to a Scenario, expose field views, and report Diagnostics. `NS2D` and `NS3D` are the two Solvers. A Solver is the in-process half of an Explorer Backend; the CPU DNS and HPC backend are Backends but not Solvers (they run offline).
_Avoid_: "sim", "engine", "the simulation object".

### Flow setup

**Scenario**:
A named flow setup = an initial condition plus its boundary configuration (e.g. `cylinder`, `channel`, `jet`, `step`, `tg`, `shear`, `hit`).
_Avoid_: "case", "test", "example", "preset".

**Volume penalization** (Brinkman immersed boundary):
The technique that imposes solids and inlets inside the periodic box by dragging velocity toward a target. The periodic spectral method has no true walls; penalization fakes them.
_Avoid_: "boundary condition" used loosely — the box is always periodic; penalization only *emulates* non-periodic boundaries.

**Mask**:
The field marking where penalization acts (1 = solid/inlet, 0 = open fluid).

**Target**:
The velocity penalization drives the masked region toward. `0` = no-slip wall; non-zero = velocity inlet.

### Diagnostics

**Reynolds number** (Re = U·L/ν):
The single dimensionless control for laminar-vs-turbulent behaviour; in the Explorers the Reynolds slider sets ν.

**Vorticity** (ω = ∇×u):
The curl of velocity — the field that makes rotational flow structure visible.

**Enstrophy** (Ω):
The integral of ½|ω|² — a measure of total rotational intensity; it grows as the flow develops fine structure.

**Dissipation rate** (ε = 2νΩ):
The rate kinetic energy is converted to heat by viscosity.

**Divergence** (∇·u):
Should stay ≈ 0 for incompressible flow; a live divergence readout is an incompressibility sanity check.

**Term decomposition**:
Viewing the right-hand-side forces — advection (u·∇)u, diffusion ν∇²u, pressure force −∇p — as separate fields, to see each one's effect on the equation.

**Pressure field** (p):
The signed scalar recovered from ∇²p = ∇·R, i.e. exactly what the Leray projection removes. High at stagnation points, low in vortex cores (the pressure-minimum vortex criterion in 3D). Distinct from the **pressure force** |−∇p| shown in the term decomposition.

**Passive tracer** (dye):
A scalar field advected (and weakly diffused) by the flow but exerting no force back on it; used to make advection and mixing visible.
_Avoid_: "particle" (the tracer is a continuous scalar field, not Lagrangian particles).

## Relationships

- A **Backend** implements the shared numerical **method** (spectral FFT, Orszag 2/3 dealiasing, Williamson RK3, Leray projection, volume penalization).
- An **Explorer** runs a **Scenario** and renders **Diagnostics** live.
- A **Scenario** is defined by an initial condition plus a **Mask** + **Target** (for penalized scenarios).
- **Dissipation** is derived from **Enstrophy**; both are **Diagnostics**.

## Example dialogue

> **Dev:** "When the 3D Explorer runs the `cylinder` Scenario, does it reuse the CPU DNS?"
> **Author:** "No — each Explorer is its own Backend (a Python solver) so it can solve *live* as you move sliders. The CPU DNS is the offline Backend; it shares the method, not the code path."
> **Dev:** "And the cylinder is a real boundary?"
> **Author:** "No. The box is periodic. The cylinder is a Mask with Target 0 — Volume penalization, not a true wall."

## Flagged ambiguities

- "viewer" vs **Explorer** — an Explorer solves live; a viewer (e.g. `plot_interactive.py`) only scrubs precomputed output. Kept distinct.
- "boundary condition" — used informally for penalized solids/inlets, but the actual numerical domain is always periodic; penalization only emulates boundaries.
