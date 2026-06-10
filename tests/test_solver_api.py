"""The Solver seam: both Python Solvers satisfy the Protocol, and the shared
Explorer helpers work against it (headless, no matplotlib / pyvista)."""

import numpy as np
import pytest

import solver_api
from ns2d import NS2D
from ns3d import NS3D


@pytest.fixture(params=["2d", "3d"])
def sim(request):
    if request.param == "2d":
        return NS2D(N=32, scenario="cylinder")
    return NS3D(N=16, scenario="cylinder")


def test_solvers_satisfy_protocol(sim):
    assert isinstance(sim, solver_api.Solver)


def test_field_data_serves_every_field(sim):
    sim.step()
    for name in solver_api.FIELDS:
        data = solver_api.field_data(sim, name)
        assert data.shape == sim.mask.shape
        assert np.isfinite(data).all()


def test_set_reynolds(sim):
    solver_api.set_reynolds(sim, 750)
    assert sim.reynolds() == pytest.approx(750)
