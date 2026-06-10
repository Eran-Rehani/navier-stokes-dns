"""2D Solver tests: Taylor-Green exactness, incompressibility, stability."""

import numpy as np
import pytest

import scenarios
from ns2d import NS2D, validate_taylor_green

TG_TOL = 2e-2   # relative energy error vs the exact decay law


def test_taylor_green_energy_decay():
    rel, measured, exact, ok = validate_taylor_green(N=64, nu=0.05, steps=200,
                                                     tol=TG_TOL)
    assert ok, f"TG decay off by {rel:.2e} (measured {measured}, exact {exact})"


def test_pressure_field_matches_taylor_green_exact():
    # TG has an exact pressure: p = (cos 2x + cos 2y)/4 at t=0 (zero mean).
    N = 64
    sim = NS2D(N=N, nu=0.05, scenario="tg")
    x = np.arange(N) * 2 * np.pi / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    exact = 0.25 * (np.cos(2 * X) + np.cos(2 * Y))
    assert np.allclose(sim.pressure_field(), exact, atol=1e-10)


def test_term_vectors_sum_to_acceleration():
    # With no penalization/forcing, du/dt = advection + diffusion - grad(p).
    # For TG at t=0 that sum is exactly -2 nu u (self-similar decay).
    sim = NS2D(N=64, nu=0.05, scenario="tg")
    tv = sim.term_vectors()
    fx = sum(f[0] for f in tv.values())
    fy = sum(f[1] for f in tv.values())
    u, v = sim.velocity()
    assert np.allclose(fx, -2 * sim.nu * u, atol=1e-10)
    assert np.allclose(fy, -2 * sim.nu * v, atol=1e-10)


def test_leray_projection_kills_divergence():
    sim = NS2D(N=64, nu=2e-3, scenario="cylinder")
    for _ in range(20):
        sim.step()
    assert sim.max_divergence() < 1e-8


@pytest.mark.parametrize("name", scenarios.NAMES)
def test_scenario_runs_stably(name):
    sim = NS2D(N=64, nu=3e-3, scenario=name)
    for _ in range(30):
        sim.step()                         # raises BlowUp on instability
    assert np.isfinite(sim.kinetic_energy())
    assert sim.dye.min() >= 0.0 and sim.dye.max() <= 1.0


def test_unforced_flow_dissipates():
    sim = NS2D(N=64, nu=0.02, scenario="tg")
    E0 = sim.kinetic_energy()
    for _ in range(50):
        sim.step()
    assert sim.kinetic_energy() < E0


def test_eta_and_dye_diffusivity_are_customizable():
    sim = NS2D(N=32, eta=0.005, dye_diffusivity=1e-3)
    assert sim.eta == 0.005
    assert sim.Dc == 1e-3
