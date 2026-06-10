"""
explorer3d.py - interactive *live* 3D Navier-Stokes playground (pyvista).

    python viz/explorer3d.py                       # single view
    python viz/explorer3d.py --scenario jet --n 40
    python viz/explorer3d.py --compare             # two Reynolds numbers side by side

A small 3D solver (ns3d.py) runs live while you interact; the selected field
is shown as a 3D isosurface that updates every frame.  Same numerics and
pedagogy as the 2D explorer, in 3D.

Keyboard:
    1..7   choose field: vorticity / speed / dye / pressure (signed p; a low
           isosurface level traces vortex cores, the pressure-minimum
           criterion) / advection / diffusion / pressure force
    n      next scenario        r   reset        space  pause/resume
    p      inject a dye puff     q   quit
Sliders: Reynolds (and 'compare Re' in --compare mode), isosurface level.

Note: a real-time toggle of the side-by-side layout isn't possible with the
plain-VTK backend, so Compare mode is selected at startup with --compare.
"""

import argparse
import numpy as np
import pyvista as pv

import scenarios
from ns3d import NS3D, BlowUp
from solver_api import FIELDS, field_data, set_reynolds

SCENARIOS = scenarios.NAMES
CMAP = {"vorticity": "hot", "speed": "viridis", "dye": "cool",
        "pressure": "RdBu_r",
        "advection": "magma", "diffusion": "magma", "pressure force": "magma"}


class Explorer3D:
    def __init__(self, n=32, scenario="cylinder", compare=False,
                 eta=scenarios.DEFAULT_ETA,
                 dye_diffusivity=scenarios.DEFAULT_DYE_DIFFUSIVITY):
        self.U0 = scenarios.DEFAULT_U0
        self.field = "vorticity"
        self.running = True
        self.level_frac = 0.4          # isosurface level as a fraction of max
        self.quit = False
        self.scn_idx = SCENARIOS.index(scenario)

        kwargs = dict(N=n, scenario=scenario, U0=self.U0,
                      eta=eta, dye_diffusivity=dye_diffusivity)
        self.sims = [NS3D(nu=self.U0 / 500.0, **kwargs)]
        shape = (1, 1)
        if compare:
            self.sims.append(NS3D(nu=self.U0 / 2000.0, **kwargs))
            shape = (1, 2)
        self.compare = compare

        self.plotter = pv.Plotter(shape=shape, title="Navier-Stokes 3D Explorer")
        self.grids, self.dx = [], self.sims[0].dx
        for j, sim in enumerate(self.sims):
            self.plotter.subplot(0, j)
            g = pv.ImageData(dimensions=(sim.N,) * 3, spacing=(self.dx,) * 3)
            self.grids.append(g)
            self._add_solid(sim, j)
            self._render_iso(sim, j, reset_cam=True)
            self.plotter.add_text("", name="diag", font_size=9, position="upper_left")
        self._bind_keys()
        self._add_sliders()

    # ------------------------------------------------------------------
    def _add_solid(self, sim, j):
        if sim.mask.any():
            g = pv.ImageData(dimensions=(sim.N,) * 3, spacing=(self.dx,) * 3)
            g["m"] = sim.mask.flatten(order="F")
            solid = g.contour([0.5])
            if solid.n_points:
                self.plotter.add_mesh(solid, color="silver", opacity=0.25,
                                      name="solid")

    def _render_iso(self, sim, j, reset_cam=False):
        self.plotter.subplot(0, j)
        data = field_data(sim, self.field)
        g = self.grids[j]
        g["s"] = data.flatten(order="F")
        lo, hi = float(data.min()), float(data.max())
        if self.field == "pressure":       # signed: keep 0 at the colormap centre
            m = max(abs(lo), abs(hi))
            clim = (-m, m + 1e-9)
        else:
            clim = (lo, hi + 1e-9)
        level = lo + self.level_frac * (hi - lo) + 1e-9
        try:
            iso = g.contour([level])
        except Exception:
            iso = None
        if iso is not None and iso.n_points:
            self.plotter.add_mesh(iso, scalars="s", cmap=CMAP[self.field],
                                  clim=clim,
                                  name="iso", show_scalar_bar=False)
        else:
            self.plotter.remove_actor("iso", render=False)
        if reset_cam:
            self.plotter.camera_position = "iso"

    # ------------------------------------------------------------------
    def _bind_keys(self):
        p = self.plotter
        for i, f in enumerate(FIELDS):
            p.add_key_event(str(i + 1), lambda f=f: self._set_field(f))
        p.add_key_event("n", self._next_scenario)
        p.add_key_event("r", self._reset)
        p.add_key_event("space", self._toggle_pause)
        p.add_key_event("p", self._puff)
        p.add_key_event("q", self._set_quit)

    def _add_sliders(self):
        self.plotter.subplot(0, 0)
        self.plotter.add_slider_widget(
            self._set_re, [50, 5000], value=500, title="Reynolds",
            pointa=(0.02, 0.10), pointb=(0.32, 0.10), style="modern")
        self.plotter.add_slider_widget(
            self._set_level, [0.05, 0.9], value=self.level_frac, title="iso level",
            pointa=(0.40, 0.10), pointb=(0.70, 0.10), style="modern")
        if self.compare:
            self.plotter.subplot(0, 1)
            self.plotter.add_slider_widget(
                self._set_re_cmp, [50, 5000], value=2000, title="compare Re",
                pointa=(0.02, 0.10), pointb=(0.32, 0.10), style="modern")

    # ---- widget / key callbacks --------------------------------------
    def _set_re(self, v):
        set_reynolds(self.sims[0], v)

    def _set_re_cmp(self, v):
        if self.compare:
            set_reynolds(self.sims[1], v)

    def _set_level(self, v):
        self.level_frac = float(v)

    def _set_field(self, f):
        self.field = f

    def _set_quit(self):
        self.quit = True

    def _toggle_pause(self):
        self.running = not self.running

    def _puff(self):
        for sim in self.sims:
            sim.puff()

    def _next_scenario(self):
        self.scn_idx = (self.scn_idx + 1) % len(SCENARIOS)
        for sim in self.sims:
            sim.scenario = SCENARIOS[self.scn_idx]
        self._reset()

    def _reset(self):
        for j, sim in enumerate(self.sims):
            sim.reset()
            self.plotter.subplot(0, j)
            self.plotter.remove_actor("solid", render=False)
            self._add_solid(sim, j)

    # ------------------------------------------------------------------
    def _advance(self, sim):
        try:
            sim.step()
            return True
        except BlowUp:
            sim.reset()
            return False

    def _diag_text(self, sim):
        return (f"scenario: {sim.scenario}   field: {self.field}\n"
                f"Re = {sim.reynolds():.0f}    t = {sim.time:.2f}\n"
                f"E = {sim.kinetic_energy():.3f}   enstrophy = {sim.enstrophy():.1f}\n"
                f"dissipation = {sim.dissipation():.2e}\n"
                f"div(u) = {sim.max_divergence():.1e}  (should be ~0)")

    def frame(self):
        for j, sim in enumerate(self.sims):
            if self.running:
                self._advance(sim)
            self._render_iso(sim, j)
            self.plotter.subplot(0, j)
            self.plotter.add_text(self._diag_text(sim), name="diag",
                                  font_size=9, position="upper_left")

    def run(self):
        self.plotter.show(interactive_update=True, auto_close=False)
        while not self.quit:
            self.frame()
            self.plotter.update()
        self.plotter.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32, help="grid resolution (real-time ~32)")
    ap.add_argument("--scenario", default="cylinder", choices=SCENARIOS)
    ap.add_argument("--compare", action="store_true", help="two Reynolds numbers side by side")
    ap.add_argument("--eta", type=float, default=scenarios.DEFAULT_ETA,
                    help="Brinkman penalization time-scale")
    ap.add_argument("--dye-diff", type=float, default=scenarios.DEFAULT_DYE_DIFFUSIVITY,
                    help="passive-tracer (dye) diffusivity")
    args = ap.parse_args()
    Explorer3D(n=args.n, scenario=args.scenario, compare=args.compare,
               eta=args.eta, dye_diffusivity=args.dye_diff).run()
