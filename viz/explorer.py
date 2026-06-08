"""
explorer.py - interactive 2D Navier-Stokes playground.

Run it and *watch* the equations come alive:

    python viz/explorer.py

Controls (matplotlib widgets):
    * scenario radio   - pick the boundary-condition / flow setup
    * Reynolds slider  - viscosity nu = U0 / Re  (high Re -> turbulence)
    * dt slider        - time-step size
    * field radio      - show vorticity or speed
    * Pause / Reset    - freeze or restart the current run

The little 2D solver (ns2d.py) is the same numerical method as the heavy
C++ 3D DNS in the project root, so the intuition you build here transfers.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button

from ns2d import NS2D

SCENARIOS = ["cylinder", "channel", "jet", "step", "shear", "tg", "hit"]
SCENARIO_HELP = {
    "cylinder": "flow past a cylinder -> von Karman vortex street",
    "channel": "plane channel with no-slip walls (wall turbulence)",
    "jet": "a jet injected through a slot in a solid plate",
    "step": "backward-facing step -> recirculation bubble",
    "shear": "double shear layer -> Kelvin-Helmholtz roll-up",
    "tg": "Taylor-Green vortex -> classic decaying benchmark",
    "hit": "random field -> homogeneous decaying turbulence",
}

N = 128
U0 = 1.5


class Explorer:
    def __init__(self):
        self.sim = NS2D(N=N, nu=U0 / 500.0, scenario="cylinder", U0=U0)
        self.dt = 4.0e-3
        self.field = "vorticity"
        self.running = True
        self.substeps = 2          # solver steps per rendered frame

        self.fig = plt.figure(figsize=(11, 7))
        self.fig.canvas.manager.set_window_title("Navier-Stokes 2D Explorer")
        self.ax = self.fig.add_axes([0.30, 0.12, 0.66, 0.80])
        self.ax.set_xticks([]); self.ax.set_yticks([])

        data = self._field_data()
        self.im = self.ax.imshow(data.T, origin="lower", cmap="RdBu_r",
                                 extent=[0, 2 * np.pi, 0, 2 * np.pi],
                                 vmin=-self._scale(), vmax=self._scale())
        self._overlay_solid()
        self.title = self.ax.set_title("")

        self._build_widgets()
        self.timer = self.fig.canvas.new_timer(interval=30)
        self.timer.add_callback(self._tick)
        self.timer.start()

    # ------------------------------------------------------------------
    def _field_data(self):
        return self.sim.vorticity() if self.field == "vorticity" else \
            np.hypot(*self.sim.velocity())

    def _scale(self):
        if self.field == "vorticity":
            return 8.0
        return self.sim.U0 * 1.5

    def _overlay_solid(self):
        """Hatch the solid / inlet region so boundaries are visible."""
        if hasattr(self, "_solid_im"):
            self._solid_im.remove()
        m = np.ma.masked_where(self.sim.mask < 0.5, self.sim.mask)
        self._solid_im = self.ax.imshow(m.T, origin="lower", cmap="gray",
                                        extent=[0, 2 * np.pi, 0, 2 * np.pi],
                                        alpha=0.35, vmin=0, vmax=1)

    def _build_widgets(self):
        ax_scn = self.fig.add_axes([0.02, 0.55, 0.22, 0.37])
        ax_scn.set_title("scenario", fontsize=10)
        self.r_scn = RadioButtons(ax_scn, SCENARIOS, active=0)
        self.r_scn.on_clicked(self._on_scenario)

        ax_fld = self.fig.add_axes([0.02, 0.40, 0.22, 0.12])
        ax_fld.set_title("field", fontsize=10)
        self.r_fld = RadioButtons(ax_fld, ["vorticity", "speed"], active=0)
        self.r_fld.on_clicked(self._on_field)

        ax_re = self.fig.add_axes([0.05, 0.30, 0.18, 0.03])
        self.s_re = Slider(ax_re, "Reynolds", 50, 5000, valinit=500, valstep=50)
        self.s_re.on_changed(self._on_re)

        ax_dt = self.fig.add_axes([0.05, 0.24, 0.18, 0.03])
        self.s_dt = Slider(ax_dt, "dt", 1e-3, 8e-3, valinit=self.dt)
        self.s_dt.on_changed(lambda v: setattr(self, "dt", v))

        ax_pause = self.fig.add_axes([0.05, 0.14, 0.08, 0.05])
        self.b_pause = Button(ax_pause, "Pause")
        self.b_pause.on_clicked(self._on_pause)

        ax_reset = self.fig.add_axes([0.15, 0.14, 0.08, 0.05])
        self.b_reset = Button(ax_reset, "Reset")
        self.b_reset.on_clicked(lambda e: self.sim.reset())

    # ------------------------------------------------------------------
    def _on_scenario(self, label):
        self.sim.scenario = label
        self.sim.reset()
        self._overlay_solid()
        self.im.set_clim(-self._scale(), self._scale())

    def _on_field(self, label):
        self.field = label
        self.im.set_cmap("RdBu_r" if label == "vorticity" else "inferno")
        self.im.set_clim(-self._scale() if label == "vorticity" else 0, self._scale())

    def _on_re(self, val):
        self.sim.nu = self.sim.U0 / val

    def _on_pause(self, event):
        self.running = not self.running
        self.b_pause.label.set_text("Resume" if not self.running else "Pause")

    def _tick(self):
        if self.running:
            try:
                for _ in range(self.substeps):
                    self.sim.step(self.dt)
            except FloatingPointError:
                self.sim.reset()
        data = self._field_data()
        self.im.set_data(data.T)
        self.title.set_text(
            f"{SCENARIO_HELP[self.sim.scenario]}\n"
            f"Re={self.sim.reynolds():.0f}   t={self.sim.time:.2f}   "
            f"E={self.sim.kinetic_energy():.3f}")
        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    Explorer()
    plt.show()
