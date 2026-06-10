"""Unit tests for the shared Diagnostics module, on synthetic fields with
known analytic answers (no time-stepping involved)."""

import numpy as np
import pytest

import diagnostics


def grid_2d(N=64):
    x = np.arange(N) * 2 * np.pi / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    k = np.fft.fftfreq(N, d=1.0 / N)
    kx = k[:, None] * np.ones((1, N))
    ky = np.ones((N, 1)) * k[None, :]
    return X, Y, kx, ky


def grid_3d(N=24):
    x = np.arange(N) * 2 * np.pi / N
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    k = np.fft.fftfreq(N, d=1.0 / N)
    kx, ky, kz = np.meshgrid(k, k, k, indexing="ij")
    return (X, Y, Z), (kx, ky, kz)


def test_kinetic_energy_uniform_flow():
    u = np.full((16, 16), 2.0)
    v = np.zeros((16, 16))
    assert diagnostics.kinetic_energy(u, v) == pytest.approx(2.0)


def test_kinetic_energy_single_mode():
    # u = sin(x): <u^2> = 1/2, E = 1/4.
    X, Y, _, _ = grid_2d()
    assert diagnostics.kinetic_energy(np.sin(X), np.zeros_like(X)) == pytest.approx(0.25)


def test_enstrophy_and_dissipation():
    w = np.full((8, 8), 3.0)
    Z = diagnostics.enstrophy(w)
    assert Z == pytest.approx(4.5)
    assert diagnostics.dissipation(0.1, Z) == pytest.approx(0.9)


def test_curl_2d_known_field():
    # u = sin(Y), v = 0  ->  omega = dv/dx - du/dy = -cos(Y).
    X, Y, kx, ky = grid_2d()
    uh, vh = np.fft.fft2(np.sin(Y)), np.fft.fft2(np.zeros_like(Y))
    omega = diagnostics.curl_2d(uh, vh, kx, ky)
    assert np.allclose(omega, -np.cos(Y), atol=1e-10)


def test_curl_3d_known_field():
    # u = (0, 0, sin(X))  ->  curl u = (0, -cos(X), 0).
    (X, Y, Z), (kx, ky, kz) = grid_3d()
    zero = np.zeros_like(X)
    uh, vh, wh = (np.fft.fftn(f) for f in (zero, zero, np.sin(X)))
    wx, wy, wz = diagnostics.curl_3d(uh, vh, wh, kx, ky, kz)
    assert np.allclose(wx, 0.0, atol=1e-10)
    assert np.allclose(wy, -np.cos(X), atol=1e-10)
    assert np.allclose(wz, 0.0, atol=1e-10)


def test_max_divergence():
    X, Y, kx, ky = grid_2d()
    # Solenoidal field: u = sin(Y), v = sin(X)  ->  div = 0.
    hats = (np.fft.fft2(np.sin(Y)), np.fft.fft2(np.sin(X)))
    assert diagnostics.max_divergence(hats, (kx, ky)) < 1e-10
    # Compressible field: u = sin(X)  ->  div = cos(X), max 1.
    hats = (np.fft.fft2(np.sin(X)), np.fft.fft2(np.zeros_like(X)))
    assert diagnostics.max_divergence(hats, (kx, ky)) == pytest.approx(1.0)


def test_energy_spectrum_single_shell():
    # u = sin(3x) puts all energy in the |k| = 3 shell.
    (X, Y, Z), _ = grid_3d()
    zero = np.zeros_like(X)
    k, E = diagnostics.energy_spectrum(np.sin(3 * X), zero, zero)
    assert k[np.argmax(E)] == 3
    assert E[np.argmax(E)] == pytest.approx(0.25)   # E = <u^2>/2 = 1/4
    assert E.sum() == pytest.approx(0.25)
