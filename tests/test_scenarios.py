"""Tests for the Scenario module: every Scenario must build a consistent
initial condition + Mask + Target + dye source in both 2D and 3D."""

import numpy as np
import pytest

import scenarios


def coords(dim, N=32):
    x = np.arange(N) * 2 * np.pi / N
    return np.meshgrid(*([x] * dim), indexing="ij")


@pytest.mark.parametrize("name", scenarios.NAMES)
@pytest.mark.parametrize("dim", [2, 3])
def test_setup_is_consistent(name, dim):
    c = coords(dim)
    setup = scenarios.build(name, c, U0=1.5)
    shape = c[0].shape

    assert len(setup.velocity) == dim
    assert len(setup.target) == dim
    for f in (*setup.velocity, setup.mask, *setup.target, setup.dye_source):
        assert f.shape == shape
        assert np.isfinite(f).all()

    # Mask and dye source are indicator fields.
    assert set(np.unique(setup.mask)) <= {0.0, 1.0}
    assert set(np.unique(setup.dye_source)) <= {0.0, 1.0}

    # Target only drives penalized regions; dye is never injected inside solids.
    solid = setup.mask > 0.5
    for t in setup.target:
        assert np.all(t[~solid] == 0.0)
    assert np.all(setup.dye_source[solid] == 0.0)

    # Every Scenario must seed something to watch.
    assert setup.dye_source.any()


@pytest.mark.parametrize("dim", [2, 3])
def test_no_slip_scenarios_start_at_rest_inside_solids(dim):
    for name in ("cylinder", "channel", "step"):
        setup = scenarios.build(name, coords(dim))
        solid = setup.mask > 0.5
        for vc in setup.velocity:
            assert np.all(vc[solid] == 0.0), f"{name}: flow inside solid"


def test_jet_inlet_has_nonzero_target():
    setup = scenarios.build("jet", coords(2), U0=2.0)
    assert setup.target[0].max() == pytest.approx(2.0)


def test_channel_is_driven():
    assert scenarios.build("channel", coords(2)).force > 0
    assert scenarios.build("tg", coords(2)).force == 0


def test_unknown_scenario_rejected():
    with pytest.raises(ValueError):
        scenarios.build("nope", coords(2))
