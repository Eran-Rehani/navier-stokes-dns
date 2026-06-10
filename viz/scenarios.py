"""
scenarios.py - the Scenario module: one source of truth for every flow setup.

A Scenario = initial velocity + penalization Mask + Target + body force +
dye-source region.  Both Python Solvers (ns2d.py, ns3d.py) build their state
from here, so geometry lives in exactly one place.  The C++ / CUDA backends
keep their own init_* kernels (see docs/adr/0002); the test suite guards
against drift.

Shared run defaults (CFL, penalization eta, dye diffusivity, drive speed)
also live here so the Solvers and Explorers cannot silently diverge.
"""

from dataclasses import dataclass

import numpy as np

TWO_PI = 2.0 * np.pi

NAMES = ["cylinder", "channel", "jet", "step", "shear", "tg", "hit"]
HELP = {
    "cylinder": "flow past a cylinder -> von Karman vortex street",
    "channel": "plane channel, no-slip walls (wall turbulence)",
    "jet": "a jet injected through a slot in a solid plate",
    "step": "backward-facing step -> recirculation bubble",
    "shear": "double shear layer -> Kelvin-Helmholtz roll-up",
    "tg": "Taylor-Green vortex -> exact decaying benchmark",
    "hit": "random field -> homogeneous decaying turbulence",
}

# Run defaults shared by both Solvers (override per run via constructor args
# or the Explorers' CLI flags).
DEFAULT_U0 = 1.5
DEFAULT_CFL = 0.4
DEFAULT_ETA = 0.01
DEFAULT_DYE_DIFFUSIVITY = 2.0e-3

# Geometry constants (the box is [0, 2pi]^dim).
CYLINDER_CENTER = (np.pi, np.pi)
CYLINDER_RADIUS = 0.5
CHANNEL_WALL = 0.12 * TWO_PI          # wall thickness on each side
CHANNEL_FORCE = 0.01                  # streamwise body force (drive)
CHANNEL_PERTURBATION = 0.02           # relative cross-stream seed perturbation
JET_PLATE = 0.10 * TWO_PI             # plate thickness (x extent)
JET_SLOT_HALF_WIDTH = 0.35
STEP_INLET = 0.08 * TWO_PI            # inflow strip thickness (x extent)
STEP_X = TWO_PI / 3.0                 # step face position
SHEAR_RHO = 30.0                      # shear-layer steepness
SHEAR_DELTA = 0.05                    # perturbation amplitude
HIT_SEED = 0


@dataclass
class Setup:
    """Everything a Solver needs to start a Scenario."""
    velocity: list        # dim physical velocity components
    mask: np.ndarray      # 1 = solid/inlet (penalized), 0 = open fluid
    target: list          # dim components penalization drags toward
    force: float          # streamwise body force
    dye_source: np.ndarray  # continuous dye injection region (0/1)


def build(name, coords, U0=DEFAULT_U0):
    """Build a Scenario on the grid given by coords = (X, Y) or (X, Y, Z),
    the meshgrid('ij') coordinate arrays of a [0, 2pi]^dim box."""
    dim = len(coords)
    X, Y = coords[0], coords[1]
    shape = X.shape
    vel = [np.zeros(shape) for _ in range(dim)]
    mask = np.zeros(shape)
    target = [np.zeros(shape) for _ in range(dim)]
    force = 0.0

    # In 3D the periodic perturbations get a z dependence so structure can
    # develop along the third axis; in 2D the factor is just 1.
    z_wave = np.sin(coords[2]) if dim == 3 else 1.0

    if name == "tg":
        z_cos = np.cos(coords[2]) if dim == 3 else 1.0
        vel[0] = np.sin(X) * np.cos(Y) * z_cos
        vel[1] = -np.cos(X) * np.sin(Y) * z_cos
    elif name == "shear":
        vel[0] = np.where(Y <= np.pi,
                          np.tanh(SHEAR_RHO * (Y - np.pi / 2)),
                          np.tanh(SHEAR_RHO * (3 * np.pi / 2 - Y)))
        vel[1] = SHEAR_DELTA * np.sin(X) * z_wave
    elif name == "hit":
        rng = np.random.default_rng(HIT_SEED)
        vel = [rng.standard_normal(shape) for _ in range(dim)]
    elif name == "cylinder":
        xc, yc = CYLINDER_CENTER
        solid = (X - xc) ** 2 + (Y - yc) ** 2 <= CYLINDER_RADIUS ** 2
        mask[solid] = 1.0
        vel[0][:] = U0
        vel[0][solid] = 0.0
    elif name == "channel":
        solid = (Y < CHANNEL_WALL) | (Y > TWO_PI - CHANNEL_WALL)
        mask[solid] = 1.0
        yy = np.clip((Y - CHANNEL_WALL) / (TWO_PI - 2 * CHANNEL_WALL), 0, 1)
        vel[0] = U0 * 4.0 * yy * (1 - yy)            # parabolic profile
        vel[1] = CHANNEL_PERTURBATION * U0 * np.sin(X) * z_wave
        vel[0][solid] = 0.0
        vel[1][solid] = 0.0
        force = CHANNEL_FORCE
    elif name == "jet":
        slot = np.abs(Y - np.pi) < JET_SLOT_HALF_WIDTH
        inplate = X < JET_PLATE
        mask[inplate] = 1.0
        target[0][inplate & slot] = U0
    elif name == "step":
        block = (X < STEP_X) & (Y < np.pi)
        inflow = (X < STEP_INLET) & (Y >= np.pi)
        mask[block] = 1.0
        mask[inflow] = 1.0
        target[0][inflow] = U0
        vel[0][(Y >= np.pi) & ~block & ~inflow] = U0
    else:
        raise ValueError(f"unknown scenario {name!r}")

    # Dye source: a band just downstream of the inlet (so advection is
    # immediately visible), or a central blob for the periodic scenarios.
    if name == "jet":
        band = (X > 0.7) & (X < 1.1) & (np.abs(Y - np.pi) < 0.4)
    elif name == "step":
        band = (X > 0.6) & (X < 1.0) & (Y >= np.pi)
    elif name in ("cylinder", "channel"):
        band = (X > 0.2) & (X < 0.6)
    else:
        r = 0.3 if dim == 2 else 0.4
        band = sum((c - np.pi) ** 2 for c in coords) < r ** 2
    dye_source = np.zeros(shape)
    dye_source[band & (mask < 0.5)] = 1.0

    return Setup(vel, mask, target, force, dye_source)
