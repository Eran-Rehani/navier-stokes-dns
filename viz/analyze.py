"""
analyze.py - turbulence diagnostics across a run.

    python viz/analyze.py

Produces output/spectrum.png: the kinetic-energy spectrum E(k) of the
last snapshot (look for the Kolmogorov k^-5/3 inertial range) together
with the kinetic-energy decay across all snapshots.
"""

import glob
import sys

import numpy as np
import matplotlib.pyplot as plt

from fields import load, speed, energy_spectrum


def main():
    files = sys.argv[1:] or sorted(glob.glob("output/vel_*.bin"))
    if not files:
        print("no .bin files found in output/")
        sys.exit(1)

    # Kinetic-energy history across snapshots.
    ke = []
    for f in files:
        N, u, v, w = load(f)
        s = speed(u, v, w)
        ke.append(0.5 * np.mean(s ** 2) if v is None else
                  0.5 * np.mean(u ** 2 + v ** 2 + w ** 2))

    N, u, v, w = load(files[-1])
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5))

    axL.plot(range(len(ke)), ke, "o-")
    axL.set_xlabel("snapshot"); axL.set_ylabel("mean kinetic energy")
    axL.set_title("energy history")
    axL.grid(alpha=0.3)

    if v is not None:
        k, E = energy_spectrum(u, v, w)
        good = E > 0
        axR.loglog(k[good], E[good], label="E(k)")
        kk = k[(k > 2) & (k < N // 4)]
        if kk.size:
            ref = E[good][np.argmax(k[good] >= kk[0])] * (kk / kk[0]) ** (-5.0 / 3.0)
            axR.loglog(kk, ref, "k--", label=r"$k^{-5/3}$")
        axR.set_xlabel("wavenumber k"); axR.set_ylabel("E(k)")
        axR.set_title("energy spectrum (last frame)")
        axR.legend(); axR.grid(alpha=0.3, which="both")
    else:
        axR.text(0.5, 0.5, "spectrum needs full u,v,w\n(re-run ns3d_cpu)",
                 ha="center", va="center")

    fig.tight_layout()
    fig.savefig("output/spectrum.png", dpi=110)
    print("saved output/spectrum.png")


if __name__ == "__main__":
    main()
