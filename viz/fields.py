"""
fields.py - load and analyse 3D DNS snapshots written by ns3d_cpu.

Binary snapshot format (little-endian):
    int32   N
    float64 u[N^3], v[N^3], w[N^3]    (index order [iz][iy][ix])

Provides the post-processing diagnostics used by the plotting scripts:
velocity magnitude, vorticity, and the kinetic-energy spectrum E(k).
"""

import numpy as np

TWO_PI = 2.0 * np.pi


def load(filename):
    """Return (N, u, v, w) with each component shaped (N, N, N) as [iz, iy, ix]."""
    with open(filename, "rb") as f:
        N = int(np.fromfile(f, dtype=np.int32, count=1)[0])
        raw = np.fromfile(f, dtype=np.float64)
    if raw.size == N ** 3:                       # legacy magnitude-only files
        mag = raw.reshape((N, N, N))
        return N, mag, None, None
    u, v, w = (raw[i * N ** 3:(i + 1) * N ** 3].reshape((N, N, N)) for i in range(3))
    return N, u, v, w


def speed(u, v, w):
    if v is None:                                # legacy file already holds |u|
        return u
    return np.sqrt(u * u + v * v + w * w)


def _wavenumbers(N):
    k = np.fft.fftfreq(N, d=1.0 / N)
    return np.meshgrid(k, k, k, indexing="ij")


def vorticity_magnitude(u, v, w):
    """|curl u| computed spectrally (exact for the periodic field)."""
    N = u.shape[0]
    kz, ky, kx = _wavenumbers(N)                 # arrays are [iz, iy, ix]
    uh, vh, wh = np.fft.fftn(u), np.fft.fftn(v), np.fft.fftn(w)
    wx = np.real(np.fft.ifftn(1j * (ky * wh - kz * vh)))
    wy = np.real(np.fft.ifftn(1j * (kz * uh - kx * wh)))
    wz = np.real(np.fft.ifftn(1j * (kx * vh - ky * uh)))
    return np.sqrt(wx * wx + wy * wy + wz * wz)


def energy_spectrum(u, v, w):
    """Shell-averaged kinetic-energy spectrum E(k).

    The hallmark of 3D turbulence is the Kolmogorov k^-5/3 inertial range;
    plotting E(k) against k on log axes lets you see it directly.
    """
    N = u.shape[0]
    kz, ky, kx = _wavenumbers(N)
    kmag = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)
    uh = np.fft.fftn(u) / N ** 3
    vh = np.fft.fftn(v) / N ** 3
    wh = np.fft.fftn(w) / N ** 3
    e = 0.5 * (np.abs(uh) ** 2 + np.abs(vh) ** 2 + np.abs(wh) ** 2)

    kbin = kmag.astype(int).ravel()
    E = np.bincount(kbin, weights=e.ravel(), minlength=N // 2 + 1)
    k = np.arange(E.size)
    return k[1:N // 2], E[1:N // 2]              # drop k=0 mean, Nyquist tail
