"""
explorer.py - interactive 2D Navier-Stokes playground.

    python viz/explorer.py
    python viz/explorer.py --scenario jet --n 96 --eta 0.005 --dye-diff 1e-3

Watch the equations come alive and build intuition for each term.

Controls:
    scenario   - boundary-condition / flow setup
    field      - what to display:
                   vorticity / speed / dye / pressure  (the flow; pressure is
                   the signed p - high at stagnation points, low in vortex cores)
                   advection / diffusion / pressure force  (the RHS forces, |.|)
    Reynolds   - sets viscosity nu = U0 / Re  (high Re -> turbulence)
    compare Re - second Reynolds number for the side-by-side Compare mode
    Pause / Reset / Puff dye / Compare / Arrows

Arrows overlay the *direction* the colour maps hide: the velocity field on
the flow fields, and the actual force vectors on the term fields (e.g. watch
-grad(p) point away from the cylinder nose, opposing the advection arrows).

The diagnostics panel tracks kinetic energy, enstrophy and the dissipation
rate live, plus the measured -dE/dt: for unforced flow the energy budget
dE/dt = -eps means the two curves must lie on top of each other (for driven
or penalized scenarios the gap is the work done by the forcing/walls).  For
the Taylor-Green scenario the exact analytic decay E = E0 e^{-4 nu t} is
overlaid as a correctness check.  The time step is chosen adaptively (CFL)
so the sim stays stable across the whole slider range.
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button

import scenarios
from ns2d import NS2D, BlowUp
from solver_api import FIELDS, field_data, set_reynolds

SCENARIOS = scenarios.NAMES
CMAP = {"vorticity": "RdBu_r", "speed": "inferno", "dye": "viridis",
        "pressure": "RdBu_r",
        "advection": "magma", "diffusion": "magma", "pressure force": "magma"}


class Explorer:
    def __init__(self, n=128, scenario="cylinder", eta=scenarios.DEFAULT_ETA,
                 dye_diffusivity=scenarios.DEFAULT_DYE_DIFFUSIVITY):
        self.N = n
        self.U0 = scenarios.DEFAULT_U0
        # Shared by the main and Compare Solvers so they differ only in nu.
        self.sim_kwargs = dict(N=n, scenario=scenario, U0=self.U0,
                               eta=eta, dye_diffusivity=dye_diffusivity)
        self.sim = NS2D(nu=self.U0 / 500.0, **self.sim_kwargs)
        self.cmp = None                 # second sim for Compare mode
        self.field = "vorticity"
        self.running = True
        self.show_arrows = False
        self.quiv = None
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
        self.imc = self.axc.imshow(np.zeros((self.N, self.N)), origin="lower", extent=ext,
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
        return field_data(sim, self.field)

    def _clim(self, sim):
        f = self.field
        if f == "vorticity":
            return -8, 8
        if f == "dye":
            return 0, 1
        if f == "speed":
            return 0, sim.U0 * 1.5
        if f == "pressure":                # signed: keep 0 at the colormap centre
            m = float(np.percentile(np.abs(self._data(sim)), 99) + 1e-6)
            return -m, m
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
        self.r_scn = RadioButtons(ax, SCENARIOS, active=SCENARIOS.index(self.sim.scenario))
        self.r_scn.on_clicked(self._on_scn)

        ax = self.fig.add_axes([0.14, 0.55, 0.11, 0.40]); ax.set_title("field", fontsize=9)
        self.r_fld = RadioButtons(ax, FIELDS, active=0); self.r_fld.on_clicked(self._on_fld)

        ax = self.fig.add_axes([0.04, 0.49, 0.18, 0.02])
        self.s_re = Slider(ax, "Reynolds", 50, 5000, valinit=500, valstep=50)
        self.s_re.on_changed(lambda v: set_reynolds(self.sim, v))

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
        ax = self.fig.add_axes([0.01, 0.28, 0.20, 0.04]); self.b_arr = Button(ax, "Arrows: off")
        self.b_arr.on_clicked(self._on_arrows)

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
            set_reynolds(self.cmp, v)

    def _on_arrows(self, e):
        self.show_arrows = not self.show_arrows
        self.b_arr.label.set_text(f"Arrows: {'on' if self.show_arrows else 'off'}")

    def _on_pause(self, e):
        self.running = not self.running
        self.b_pause.label.set_text("Resume" if not self.running else "Pause")

    def _on_compare(self, e):
        if self.cmp is None:
            kwargs = dict(self.sim_kwargs, scenario=self.sim.scenario)
            self.cmp = NS2D(nu=self.U0 / self.s_rec.val, **kwargs)
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
        self._draw_arrows()
        self.title.set_text(f"{scenarios.HELP[self.sim.scenario]}\n"
                            f"Re={self.sim.reynolds():.0f}  t={self.sim.time:.2f}  "
                            f"div(u)={self.sim.max_divergence():.1e}")
        if self.cmp is not None:
            self.imc.set_data(self._data(self.cmp).T)
            self.titlec.set_text(f"Re={self.cmp.reynolds():.0f}")
        self._draw_diag()
        self.fig.canvas.draw_idle()

    def _draw_arrows(self):
        """Direction overlay: the colour maps show only magnitudes (or a
        scalar), so arrows restore the vector picture.  Flow fields get the
        velocity u; each term field gets its own force vectors; the signed
        pressure field gets -grad(p), so you see the force point from high
        to low pressure."""
        if self.quiv is not None:
            self.quiv.remove()
            self.quiv = None
        if not self.show_arrows:
            return
        sim = self.sim
        if self.field in ("advection", "diffusion", "pressure force"):
            fx, fy = sim.term_vectors()[self.field]
        elif self.field == "pressure":
            fx, fy = sim.term_vectors()["pressure force"]
        else:
            fx, fy = sim.velocity()
        s = max(1, sim.N // 16)
        x = np.arange(sim.N) * sim.dx
        X, Y = np.meshgrid(x, x, indexing="ij")
        self.quiv = self.ax.quiver(X[::s, ::s], Y[::s, ::s],
                                   fx[::s, ::s], fy[::s, ::s],
                                   color="k", alpha=0.7, width=0.003)

    def _draw_diag(self):
        ax = self.axd
        ax.clear()
        t = self.hist["t"]
        if len(t) > 1:
            ax.plot(t, self.hist["E"], "C0", label="kinetic energy E")
            ax.plot(t, self.hist["eps"], "C3", label="dissipation 2nu*Z")
            if len(t) > 4:
                # Energy budget: for unforced flow dE/dt = -eps, so this
                # curve must sit on the dissipation one; any gap is the work
                # done by the body force / penalized walls.
                dEdt = np.gradient(np.array(self.hist["E"]), np.array(t))
                ax.plot(t, -dEdt, "C3--", lw=1, label="-dE/dt (=eps unforced)")
            if self.sim.scenario == "tg":
                E0 = self.hist["E"][0]
                exact = E0 * np.exp(-4 * self.sim.nu * (np.array(t) - t[0]))
                ax.plot(t, exact, "k--", lw=1, label="exact E (TG)")
            ax.legend(loc="upper right", fontsize=8, ncol=3)
        ax.set_xlabel("time"); ax.set_ylabel("diagnostics")
        ax.grid(alpha=0.3)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128, help="grid resolution")
    ap.add_argument("--scenario", default="cylinder", choices=SCENARIOS)
    ap.add_argument("--eta", type=float, default=scenarios.DEFAULT_ETA,
                    help="Brinkman penalization time-scale")
    ap.add_argument("--dye-diff", type=float, default=scenarios.DEFAULT_DYE_DIFFUSIVITY,
                    help="passive-tracer (dye) diffusivity")
    args = ap.parse_args()
    Explorer(n=args.n, scenario=args.scenario, eta=args.eta,
             dye_diffusivity=args.dye_diff)
    plt.show()
