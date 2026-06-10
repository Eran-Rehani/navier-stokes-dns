"""
solver_api.py - the Solver seam between the Explorers and the Python solvers.

A Solver is the live, in-process object an Explorer drives (see CONTEXT.md).
NS2D and NS3D are the two Solvers; both Explorers program against this
Protocol only, so shared Explorer logic (field dispatch, Reynolds -> nu)
lives here and a Solver can be exercised headless (e.g. by the test suite,
or with a stub) without matplotlib / pyvista.
"""

from typing import Protocol, runtime_checkable

import numpy as np

# Displayable fields every Solver provides: the flow itself (vorticity,
# speed, dye, signed pressure p), then the term decomposition of the RHS
# forces |.| (advection -(u.grad)u, diffusion nu lap(u), pressure force
# -grad(p)).
FIELDS = ["vorticity", "speed", "dye", "pressure",
          "advection", "diffusion", "pressure force"]


@runtime_checkable
class Solver(Protocol):
    N: int
    nu: float
    scenario: str
    U0: float
    eta: float
    time: float
    mask: np.ndarray
    dye: np.ndarray

    def step(self, dt=None) -> float: ...
    def reset(self) -> None: ...
    def puff(self) -> None: ...
    def vorticity_view(self) -> np.ndarray: ...
    def speed(self) -> np.ndarray: ...
    def pressure_field(self) -> np.ndarray: ...
    def terms(self) -> dict: ...
    def kinetic_energy(self) -> float: ...
    def enstrophy(self) -> float: ...
    def dissipation(self) -> float: ...
    def max_divergence(self) -> float: ...
    def reynolds(self) -> float: ...


def field_data(sim: Solver, name: str) -> np.ndarray:
    """The scalar field a given display name maps to, for any Solver."""
    if name == "vorticity":
        return sim.vorticity_view()
    if name == "speed":
        return sim.speed()
    if name == "dye":
        return sim.dye
    if name == "pressure":
        return sim.pressure_field()
    return sim.terms()[name]            # advection / diffusion / pressure force


def set_reynolds(sim: Solver, re: float) -> None:
    """Reynolds slider semantics: Re = U0 L / nu with L = 1, so nu = U0/Re."""
    sim.nu = sim.U0 / float(re)
