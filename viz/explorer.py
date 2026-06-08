"""
explorer.py - interactive 2D Navier-Stokes playground.

    python viz/explorer.py

Watch the equations come alive and build intuition for each term.

Controls:
    scenario   - boundary-condition / flow setup
    field      - what to display:
                   vorticity / speed / dye  (the flow)
                   advection / diffusion / pressure  (the RHS forces, |.|)
    Reynolds   - sets viscosity nu = U0 / Re  (high Re -> turbulence)
    compare Re - second Reynolds number for the side-by-side Compare mode
    Pause / Reset / Puff dye / Compare

The diagnostics panel tracks kinetic energy, enstrophy and the dissipation
rate live; for the Taylor-Green scenario it overlays the exact analytic
energy decay E = E0 e^{-4 nu t} as a correctness check.  The time step is
chosen adaptively (CFL) so the sim stays stable across the whole slider range.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button

from ns2d import NS2D, BlowUp

SCENARIOS = ["cylinder", "channel", "jet", "step", "shear", "tg", "hit"]
SCENARIO_HELP = {
    "cylinder": "flow past a cylinder -> von Karman vortex street",
    "channel": "plane channel, no-slip walls (wall turbulence)",
    "jet": "a jet injected through a slot in a solid plate",
    "step": "backward-facing step -> recirculation bubble",
    "shear": "double shear layer -> Kelvin-Helmholtz roll-up",
    "tg": "Taylor-Green vortex -> exact decaying benchmark",
    "hit": "random field -> homogeneous decaying turbulence",
}
FIELDS = ["vorticity", "speed", "dye", "advection", "diffusion", "pressure"]
CMAP = {"vorticity": "RdBu_r", "speed": "inferno", "dye": "viridis",
        "advection": "magma", "diffusion": "magma", "pressure": "magma"}

N = 128
U0 = 1.5


class Explorer:
    def __init__(self):
        self.sim = NS2D(N=N, nu=U0 / 500.0, scenario="cylinder", U0=U0)
        self.cmp = None                 # second sim for Compare mode
        self.field = "vorticity"
        self.running = True
        self.substeps = 2
        self.hist = {"t": [], "E": [], "Z": [], "eps": []}

        self.fig = plt.figure(figsize=(13, 7.5))
        self.fig.canvas.manager.set_window_title("Navier-Stokes 2D Explorer")
        self.ax = self.fig.add_axes([0.27, 0.40, 0.40, 0.55]); self._blank(self.ax)
        self.axc = self.fig.add_axes([0.70, 0.40, 0.28, 0.55]); self._blank(self.axc)
        self.axc.set_visible(False)
        self.axd = self.fig.add_axes([0.27, 0.07, 0.71, 0.24])

        ext = [0, 2 * np.pi, 0, 2 * np.pi]
        self.im = self.ax.imshow(self._data(self.sim).T, origin="lower", extent=ext,
                                 cmap=CMAP[self.field], vmin=-8, vmax=8)
        self.imc = self.axc.imshow(np.zeros((N, N)), origin="lower", extent=ext,
                                   cmap=CMAP[self.field], vmin=-8, vmax=8)
        self._overlay(self.ax, self.sim)
        self.title = self.ax.set_title("")
        self.titlec = self.axc.set_title("")

        self._build_widgets()
        self._reset_clim()
        self.timer = self.fig.canvas.new_timer(interval=30)
        self.timer.add_callback(self._tick)
        self.timer.start()

    # ------------------------------------------------------------------
    @staticmethod
    def _blank(ax):
        ax.set_xticks([]); ax.set_yticks([])

    def _data(self, sim):
        f = self.field
        if f == "vorticity":
            return sim.vorticity()
        if f == "speed":
            return sim.speed()
        if f == "dye":
            return sim.dye
        return sim.terms()[f]                      # advection / diffusion / pressure

    def _clim(self, sim):
        f = self.field
        if f == "vorticity":
            return -8, 8
        if f == "dye":
            return 0, 1
        if f == "speed":
            return 0, sim.U0 * 1.5
        return 0, float(np.percentile(self._data(sim), 99) + 1e-6)

    def _reset_clim(self):
        lo, hi = self._clim(self.sim)
        self.im.set_clim(lo, hi)
        self.imc.set_clim(lo, hi)

    def _overlay(self, ax, sim):
        """Shade solid / inlet regions so boundaries are visible."""
        attr = "_solid_" + ("a" if ax is self.ax else "b")
        old = getattr(self, attr, None)
        if old is not None:
            old.remove()
        m = np.ma.masked_where(sim.mask < 0.5, sim.mask)
        h = ax.imshow(m.T, origin="lower", extent=[0, 2 * np.pi, 0, 2 * np.pi],
                      cmap="gray", alpha=0.35, vmin=0, vmax=1)
        setattr(self, attr, h)

    # ------------------------------------------------------------------
    def _build_widgets(self):
        ax = self.fig.add_axes([0.01, 0.55, 0.13, 0.40]); ax.set_title("scenario", fontsize=9)
        self.r_scn = RadioButtons(ax, SCENARIOS, active=0); self.r_scn.on_clicked(self._on_scn)

        ax = self.fig.add_axes([0.14, 0.55, 0.11, 0.40]); ax.set_title("field", fontsize=9)
        self.r_fld = RadioButtons(ax, FIELDS, active=0); self.r_fld.on_clicked(self._on_fld)

        ax = self.fig.add_axes([0.04, 0.49, 0.18, 0.02])
        self.s_re = Slider(ax, "Reynolds", 50, 5000, valinit=500, valstep=50)
        self.s_re.on_changed(lambda v: setattr(self.sim, "nu", self.sim.U0 / v))

        ax = self.fig.add_axes([0.04, 0.45, 0.18, 0.02])
        self.s_rec = Slider(ax, "compare Re", 50, 5000, valinit=2000, valstep=50,
                            color="0.6")
        self.s_rec.on_changed(self._on_cmp_re)

        ax = self.fig.add_axes([0.01, 0.38, 0.06, 0.04]); self.b_pause = Button(ax, "Pause")
        self.b_pause.on_clicked(self._on_pause)
        ax = self.fig.add_axes([0.08, 0.38, 0.06, 0.04]); self.b_reset = Button(ax, "Reset")
        self.b_reset.on_clicked(lambda e: self._reset_all())
        ax = self.fig.add_axes([0.15, 0.38, 0.06, 0.04]); self.b_puff = Button(ax, "Puff")
        self.b_puff.on_clicked(lambda e: (self.sim.puff(), self.cmp and self.cmp.puff()))
        ax = self.fig.add_axes([0.01, 0.33, 0.20, 0.04]); self.b_cmp = Button(ax, "Compare: off")
        self.b_cmp.on_clicked(self._on_compare)

    # ------------------------------------------------------------------
    def _on_scn(self, label):
        self.sim.scenario = label
        self._reset_all()

    def _on_fld(self, label):
        self.field = label
        self.im.set_cmap(CMAP[label]); self.imc.set_cmap(CMAP[label])
        self._reset_clim()

    def _on_cmp_re(self, v):
        if self.cmp is not None:
            self.cmp.nu = self.cmp.U0 / v

    def _on_pause(self, e):
        self.running = not self.running
        self.b_pause.label.set_text("Resume" if not self.running else "Pause")

    def _on_compare(self, e):
        if self.cmp is None:
            self.cmp = NS2D(N=N, nu=U0 / self.s_rec.val, scenario=self.sim.scenario, U0=U0)
            self.axc.set_visible(True)
            self.b_cmp.label.set_text("Compare: on")
        else:
            self.cmp = None
            self.axc.set_visible(False)
            self.b_cmp.label.set_text("Compare: off")

    def _reset_all(self):
        self.hist = {k: [] for k in self.hist}
        self.sim.reset()
        self._overlay(self.ax, self.sim)
        if self.cmp is not None:
            self.cmp.scenario = self.sim.scenario
            self.cmp.reset()
            self._overlay(self.axc, self.cmp)
        self._reset_clim()

    # ------------------------------------------------------------------
    def _advance(self, sim):
        try:
            for _ in range(self.substeps):
                sim.step()
            return True
        except BlowUp:
            sim.reset()
            return False

    def _tick(self):
        if self.running:
            ok = self._advance(self.sim)
            if self.cmp is not None:
                self._advance(self.cmp)
            if ok:
                self.hist["t"].append(self.sim.time)
                self.hist["E"].append(self.sim.kinetic_energy())
                self.hist["Z"].append(self.sim.enstrophy())
                self.hist["eps"].append(self.sim.dissipation())
                for k in self.hist:
                    self.hist[k] = self.hist[k][-400:]

        self.im.set_data(self._data(self.sim).T)
        self.title.set_text(f"{SCENARIO_HELP[self.sim.scenario]}\n"
                            f"Re={self.sim.reynolds():.0f}  t={self.sim.time:.2f}  "
                            f"div(u)={self.sim.max_divergence():.1e}")
        if self.cmp is not None:
            self.imc.set_data(self._data(self.cmp).T)
            self.titlec.set_text(f"Re={self.cmp.reynolds():.0f}")
        self._draw_diag()
        self.fig.canvas.draw_idle()

    def _draw_diag(self):
        ax = self.axd
        ax.clear()
        t = self.hist["t"]
        if len(t) > 1:
            ax.plot(t, self.hist["E"], "C0", label="kinetic energy E")
            ax.plot(t, self.hist["eps"], "C3", label="dissipation 2nu*Z")
            if self.sim.scenario == "tg":
                E0 = self.hist["E"][0]
                exact = E0 * np.exp(-4 * self.sim.nu * (np.array(t) - t[0]))
                ax.plot(t, exact, "k--", lw=1, label="exact E (TG)")
            ax.legend(loc="upper right", fontsize=8, ncol=3)
        ax.set_xlabel("time"); ax.set_ylabel("diagnostics")
        ax.grid(alpha=0.3)


if __name__ == "__main__":
    Explorer()
    plt.show()
