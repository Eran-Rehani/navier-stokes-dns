"""
plot_velocity.py - mid-plane speed + vorticity slices for DNS snapshots.

    python viz/plot_velocity.py [output/vel_*.bin]

Writes a PNG next to each .bin showing the velocity magnitude and the
vorticity (where the rotational structure of the flow is visible).
"""

import os
import sys
import glob

import numpy as np
import matplotlib.pyplot as plt

from fields import load, speed, vorticity_magnitude


def plot_slice(filename):
    N, u, v, w = load(filename)
    z = N // 2
    spd = speed(u, v, w)[z]
    has_vort = v is not None
    vort = vorticity_magnitude(u, v, w)[z] if has_vort else None

    ncols = 2 if has_vort else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5), squeeze=False)
    ext = [0, 2 * np.pi, 0, 2 * np.pi]

    im0 = axes[0, 0].imshow(spd, origin="lower", extent=ext, cmap="inferno")
    axes[0, 0].set_title("speed |u|")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    if has_vort:
        im1 = axes[0, 1].imshow(vort, origin="lower", extent=ext, cmap="RdBu_r")
        axes[0, 1].set_title("vorticity |curl u|")
        fig.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    for ax in axes[0]:
        ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.suptitle(f"z = {z}  |  {os.path.basename(filename)}")
    fig.tight_layout()

    out = filename.replace(".bin", ".png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    files = sys.argv[1:] or sorted(glob.glob("output/vel_*.bin"))
    if not files:
        print("no .bin files found in output/")
        sys.exit(1)
    for f in files:
        plot_slice(f)
