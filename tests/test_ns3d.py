"""3D Solver tests: incompressibility, dissipation, stability (small N so the
whole module stays fast; correctness vs the C++ Backend is in
test_cpu_backend.py)."""

import numpy as np
import pytest

import scenarios
from ns3d import NS3D


def test_leray_projection_kills_divergence():
    sim = NS3D(N=24, nu=4e-3, scenario="cylinder")
    for _ in range(10):
        sim.step()
    assert sim.max_divergence() < 1e-8


def test_pressure_field_matches_taylor_green_exact():
    # 3D TG exact initial pressure: p = (cos2x + cos2y)(2 + cos2z)/16.
    N = 24
    sim = NS3D(N=N, nu=0.05, scenario="tg")
    x = np.arange(N) * 2 * np.pi / N
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    exact = (np.cos(2 * X) + np.cos(2 * Y)) * (2 + np.cos(2 * Z)) / 16.0
    assert np.allclose(sim.pressure_field(), exact, atol=1e-10)


def test_unforced_flow_dissipates():
    sim = NS3D(N=24, nu=0.05, scenario="tg")
    E0 = sim.kinetic_energy()
    for _ in range(20):
        sim.step()
    assert sim.kinetic_energy() < E0


@pytest.mark.parametrize("name", scenarios.NAMES)
def test_scenario_runs_stably(name):
    sim = NS3D(N=24, nu=5e-3, scenario=name)
    for _ in range(8):
        sim.step()                         # raises BlowUp on instability
    assert np.isfinite(sim.kinetic_energy())
    assert sim.dye.min() >= 0.0 and sim.dye.max() <= 1.0


def test_eta_and_dye_diffusivity_are_customizable():
    sim = NS3D(N=16, eta=0.005, dye_diffusivity=1e-3)
    assert sim.eta == 0.005
    assert sim.Dc == 1e-3
