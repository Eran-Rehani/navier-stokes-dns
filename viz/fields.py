"""
fields.py - load 3D DNS snapshots written by ns3d_cpu.

Binary snapshot format (little-endian):
    int32   N
    float64 u[N^3], v[N^3], w[N^3]    (index order [iz][iy][ix])

The Diagnostics themselves (vorticity, energy spectrum, ...) live in
diagnostics.py; this module owns the snapshot format and re-exports the
post-processing helpers the plotting scripts use.
"""

import numpy as np

from diagnostics import curl_3d, energy_spectrum  # noqa: F401  (re-export)

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


def vorticity_magnitude(u, v, w):
    """|curl u| computed spectrally (exact for the periodic field)."""
    N = u.shape[0]
    k = np.fft.fftfreq(N, d=1.0 / N)
    kz, ky, kx = np.meshgrid(k, k, k, indexing="ij")   # arrays are [iz, iy, ix]
    uh, vh, wh = np.fft.fftn(u), np.fft.fftn(v), np.fft.fftn(w)
    wx, wy, wz = curl_3d(uh, vh, wh, kx, ky, kz)
    return np.sqrt(wx * wx + wy * wy + wz * wz)
