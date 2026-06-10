"""
diagnostics.py - shared flow Diagnostics for every Python consumer.

One implementation of each formula, used by the Solvers (ns2d.py, ns3d.py),
the snapshot loader (fields.py) and the analysis scripts (analyze.py).
All curls / divergences are computed spectrally, so they are exact for the
periodic fields this project works with.

Conventions:
    * "hat" arguments are full (not dealiased) spectral fields, numpy fft layout.
    * wavenumber arrays are the integer grids built with np.fft.fftfreq(N, 1/N),
      broadcast to the field shape; they only need to be *named* consistently
      with the axes of the arrays they multiply, not stored in any layout.
"""

import numpy as np


# ----------------------------------------------------------------------
#  Scalar diagnostics.
# ----------------------------------------------------------------------
def kinetic_energy(*components):
    """Mean kinetic energy E = 0.5 <|u|^2> from physical velocity components."""
    return 0.5 * float(np.mean(sum(c * c for c in components)))


def enstrophy(*vorticity_components):
    """Enstrophy Omega = 0.5 <|omega|^2> from physical vorticity components
    (one scalar component in 2D, three in 3D)."""
    return 0.5 * float(np.mean(sum(w * w for w in vorticity_components)))


def dissipation(nu, enstrophy_value):
    """Dissipation rate eps = 2 nu Omega (periodic-box identity)."""
    return 2.0 * nu * enstrophy_value


def max_divergence(hats, wavenumbers):
    """max |div u| from spectral velocity components; ~0 for incompressible flow."""
    div_hat = 1j * sum(k * h for k, h in zip(wavenumbers, hats))
    return float(np.max(np.abs(np.real(np.fft.ifftn(div_hat)))))


# ----------------------------------------------------------------------
#  Vorticity (spectral curl).
# ----------------------------------------------------------------------
def curl_2d(uh, vh, kx, ky):
    """Scalar 2D vorticity omega = dv/dx - du/dy from spectral velocity."""
    return np.real(np.fft.ifftn(1j * (kx * vh - ky * uh)))


def curl_3d(uh, vh, wh, kx, ky, kz):
    """Vector 3D vorticity (wx, wy, wz) = curl u from spectral velocity."""
    wx = np.real(np.fft.ifftn(1j * (ky * wh - kz * vh)))
    wy = np.real(np.fft.ifftn(1j * (kz * uh - kx * wh)))
    wz = np.real(np.fft.ifftn(1j * (kx * vh - ky * uh)))
    return wx, wy, wz


# ----------------------------------------------------------------------
#  Energy spectrum.
# ----------------------------------------------------------------------
def energy_spectrum(u, v, w):
    """Shell-averaged kinetic-energy spectrum E(k) from physical velocity.

    The hallmark of 3D turbulence is the Kolmogorov k^-5/3 inertial range;
    plotting E(k) against k on log axes lets you see it directly.
    """
    N = u.shape[0]
    k = np.fft.fftfreq(N, d=1.0 / N)
    k1, k2, k3 = np.meshgrid(k, k, k, indexing="ij")
    kmag = np.sqrt(k1 ** 2 + k2 ** 2 + k3 ** 2)
    uh = np.fft.fftn(u) / N ** 3
    vh = np.fft.fftn(v) / N ** 3
    wh = np.fft.fftn(w) / N ** 3
    e = 0.5 * (np.abs(uh) ** 2 + np.abs(vh) ** 2 + np.abs(wh) ** 2)

    kbin = kmag.astype(int).ravel()
    E = np.bincount(kbin, weights=e.ravel(), minlength=N // 2 + 1)
    kk = np.arange(E.size)
    return kk[1:N // 2], E[1:N // 2]              # drop k=0 mean, Nyquist tail
