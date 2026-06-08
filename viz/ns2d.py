"""
ns2d.py - 2D pseudo-spectral incompressible Navier-Stokes solver.

The *teaching* solver: a compact, readable Python mirror of the C++ 3D DNS
in the project root.  Same numerical method, one dimension lower so it runs
interactively:

    * spatial      : Fourier (FFT) on a periodic [0, 2pi]^2 box
    * dealiasing   : Orszag 2/3 rule
    * time stepping: Williamson low-storage 3rd-order Runge-Kutta
    * incompressible: exact Leray projection each sub-step
    * boundaries   : volume penalization (Brinkman) immersed solids/inlets
    * robustness   : adaptive CFL time step + blow-up detection
    * pedagogy     : term decomposition, passive-tracer dye, live diagnostics,
                     and an exact Taylor-Green analytic self-test

Read this alongside cpu_kernels.hpp - every routine has a direct counterpart.
"""

import numpy as np

TWO_PI = 2.0 * np.pi

# Williamson (1980) low-storage RK3 coefficients - identical to the C++ code.
RK_A = np.array([0.0, -5.0 / 9.0, -153.0 / 128.0])
RK_B = np.array([1.0 / 3.0, 15.0 / 16.0, 8.0 / 15.0])


class BlowUp(Exception):
    """Raised when the solution stops being finite (instability)."""


class NS2D:
    """A periodic 2D incompressible Navier-Stokes solver in spectral space."""

    def __init__(self, N=128, nu=2.0e-3, scenario="cylinder", U0=1.5,
                 cfl=0.4, dye_diffusivity=2.0e-3):
        self.N = N
        self.nu = nu
        self.scenario = scenario
        self.U0 = U0
        self.cfl = cfl
        self.Dc = dye_diffusivity
        self.dx = TWO_PI / N
        self.time = 0.0
        self.step_count = 0

        # Integer wavenumbers (L = 2pi -> k integers), broadcast to 2D grids.
        k = np.fft.fftfreq(N, d=1.0 / N)
        self.kx = k[:, None] * np.ones((1, N))
        self.ky = np.ones((N, 1)) * k[None, :]
        self.ksq = self.kx ** 2 + self.ky ** 2
        self.ksq_safe = np.where(self.ksq == 0, 1.0, self.ksq)
        self.ksq_inv = np.where(self.ksq > 0, 1.0 / self.ksq_safe, 0.0)

        kcut = N // 3                                  # Orszag 2/3 dealiasing
        self.dealias_mask = (np.abs(k)[:, None] <= kcut) & (np.abs(k)[None, :] <= kcut)

        self.mask = np.zeros((N, N))                   # penalization fields
        self.tu = np.zeros((N, N))
        self.tv = np.zeros((N, N))
        self.eta = 0.01
        self.force = 0.0

        self.dye = np.zeros((N, N))                    # passive tracer
        self.dye_source = np.zeros((N, N))             # continuous injection region

        self.reset()

    # ------------------------------------------------------------------
    #  Scenario setup: initial field + penalization mask / target + dye.
    # ------------------------------------------------------------------
    def reset(self):
        N, dx = self.N, self.dx
        x = np.arange(N) * dx
        X, Y = np.meshgrid(x, x, indexing="ij")
        u = np.zeros((N, N))
        v = np.zeros((N, N))
        self.mask[:] = 0.0
        self.tu[:] = 0.0
        self.tv[:] = 0.0
        self.dye[:] = 0.0
        self.dye_source[:] = 0.0
        self.force = 0.0
        self.time = 0.0
        self.step_count = 0
        s = self.scenario

        if s == "tg":
            u = np.sin(X) * np.cos(Y)
            v = -np.cos(X) * np.sin(Y)
        elif s == "shear":
            rho, delta = 30.0, 0.05
            u = np.where(Y <= np.pi,
                         np.tanh(rho * (Y - np.pi / 2)),
                         np.tanh(rho * (3 * np.pi / 2 - Y)))
            v = delta * np.sin(X)
        elif s == "hit":
            rng = np.random.default_rng(0)
            u = rng.standard_normal((N, N))
            v = rng.standard_normal((N, N))
        elif s == "cylinder":
            xc, yc, R = np.pi, np.pi, 0.5
            solid = (X - xc) ** 2 + (Y - yc) ** 2 <= R ** 2
            self.mask[solid] = 1.0
            u[:] = self.U0
            u[solid] = 0.0
        elif s == "channel":
            wall = 0.12 * TWO_PI
            solid = (Y < wall) | (Y > TWO_PI - wall)
            self.mask[solid] = 1.0
            yy = np.clip((Y - wall) / (TWO_PI - 2 * wall), 0, 1)
            u = self.U0 * 4.0 * yy * (1 - yy)
            v = 0.02 * self.U0 * np.sin(X)
            u[solid] = 0.0
            v[solid] = 0.0
            self.force = 0.01
        elif s == "jet":
            plate = 0.10 * TWO_PI
            slot = np.abs(Y - np.pi) < 0.35
            inplate = X < plate
            self.mask[inplate] = 1.0
            self.tu[inplate & slot] = self.U0
        elif s == "step":
            inlet = 0.08 * TWO_PI
            stepx = TWO_PI / 3.0
            block = (X < stepx) & (Y < np.pi)
            inflow = (X < inlet) & (Y >= np.pi)
            self.mask[block] = 1.0
            self.mask[inflow] = 1.0
            self.tu[inflow] = self.U0
            u[(Y >= np.pi) & ~block & ~inflow] = self.U0
        else:
            raise ValueError(f"unknown scenario {s!r}")

        # Dye source: a band just downstream of the inlet (so advection is
        # immediately visible), or a central blob for the periodic scenarios.
        if s == "jet":
            band = (X > 0.7) & (X < 1.1) & (np.abs(Y - np.pi) < 0.4)
        elif s == "step":
            band = (X > 0.6) & (X < 1.0) & (Y >= np.pi)
        elif s in ("cylinder", "channel"):
            band = (X > 0.2) & (X < 0.6)
        else:
            band = (X - np.pi) ** 2 + (Y - np.pi) ** 2 < 0.3 ** 2
        self.dye_source[band & (self.mask < 0.5)] = 1.0

        self.uh = np.fft.fft2(u)
        self.vh = np.fft.fft2(v)
        self._project()
        self.Qu = np.zeros((N, N), dtype=complex)
        self.Qv = np.zeros((N, N), dtype=complex)
        self.Qc = np.zeros((N, N), dtype=complex)

    # ------------------------------------------------------------------
    #  Core spectral operators (mirror cpu_kernels.hpp).
    # ------------------------------------------------------------------
    def _project(self):
        """Leray projection: remove the divergent part."""
        div = self.kx * self.uh + self.ky * self.vh
        self.uh -= self.kx * div * self.ksq_inv
        self.vh -= self.ky * div * self.ksq_inv

    def _advection(self, u, v):
        """Pure -(u.grad)u for both components, in physical space."""
        out = []
        for comp_h in (self.uh, self.vh):
            ddx = np.real(np.fft.ifft2(1j * self.kx * comp_h * self.dealias_mask))
            ddy = np.real(np.fft.ifft2(1j * self.ky * comp_h * self.dealias_mask))
            out.append(-(u * ddx + v * ddy))
        return out                                     # [au, av]

    def _rhs_components(self):
        """Return the physical-space RHS pieces used by step() and terms().

        Returns (u, v, au, av, du, dv, pu, pv) where
            au,av = advection,  du,dv = diffusion,  pu,pv = penalization.
        """
        u = np.real(np.fft.ifft2(self.uh * self.dealias_mask))
        v = np.real(np.fft.ifft2(self.vh * self.dealias_mask))
        au, av = self._advection(u, v)
        du = np.real(np.fft.ifft2(-self.nu * self.ksq * self.uh))
        dv = np.real(np.fft.ifft2(-self.nu * self.ksq * self.vh))
        pu = -self.mask * (u - self.tu) / self.eta if self.mask.any() else np.zeros_like(u)
        pv = -self.mask * (v - self.tv) / self.eta if self.mask.any() else np.zeros_like(v)
        if self.force:
            pu = pu + self.force                       # channel body force
        return u, v, au, av, du, dv, pu, pv

    # ------------------------------------------------------------------
    #  Time stepping with adaptive CFL.
    # ------------------------------------------------------------------
    def cfl_dt(self):
        """Stable explicit time step from the advective, viscous and
        penalization limits.  The viscous limit scales with the largest
        resolved wavenumber (kmax = N/2), which is what actually constrains
        an explicit spectral scheme."""
        u, v = self.velocity()
        umax = float(np.sqrt(np.max(u ** 2 + v ** 2))) + 1e-9
        kmax = self.N / 2.0
        dt_adv = self.cfl * self.dx / umax
        dt_visc = 1.0 / (self.nu * kmax ** 2 + 1e-12)
        dt_pen = 0.5 * self.eta if self.mask.any() else np.inf
        return float(min(dt_adv, dt_visc, dt_pen, 0.01))

    def step(self, dt=None):
        """Advance one full RK3 step.  dt=None -> adaptive CFL.  Returns dt."""
        if dt is None:
            dt = self.cfl_dt()
        for s in range(3):
            u, v, au, av, du, dv, pu, pv = self._rhs_components()
            rhs_u = np.fft.fft2(au + du + pu)
            rhs_v = np.fft.fft2(av + dv + pv)
            self.Qu = RK_A[s] * self.Qu + dt * rhs_u
            self.Qv = RK_A[s] * self.Qv + dt * rhs_v
            self.uh += RK_B[s] * self.Qu
            self.vh += RK_B[s] * self.Qv
            self._project()
            # Passive tracer: advected and weakly diffused, no back-reaction.
            ch = np.fft.fft2(self.dye)
            cx = np.real(np.fft.ifft2(1j * self.kx * ch * self.dealias_mask))
            cy = np.real(np.fft.ifft2(1j * self.ky * ch * self.dealias_mask))
            rhs_c = np.fft.fft2(-(u * cx + v * cy)) - self.Dc * self.ksq * ch
            self.Qc = RK_A[s] * self.Qc + dt * rhs_c
            self.dye = np.real(np.fft.ifft2(np.fft.fft2(self.dye) + RK_B[s] * self.Qc))
        # Inject dye and keep it bounded.
        self.dye += self.dye_source * dt * 2.0
        np.clip(self.dye, 0.0, 1.0, out=self.dye)
        self.time += dt
        self.step_count += 1
        if not np.isfinite(self.uh).all():
            raise BlowUp(f"non-finite field at t={self.time:.3f}")
        return dt

    def puff(self, xc=None, yc=None, r=0.4):
        """Inject a dye puff (default: domain centre) to watch it advect."""
        x = np.arange(self.N) * self.dx
        X, Y = np.meshgrid(x, x, indexing="ij")
        xc = np.pi if xc is None else xc
        yc = np.pi if yc is None else yc
        self.dye += np.exp(-((X - xc) ** 2 + (Y - yc) ** 2) / (2 * r ** 2))
        np.clip(self.dye, 0.0, 1.0, out=self.dye)

    # ------------------------------------------------------------------
    #  Fields for visualization.
    # ------------------------------------------------------------------
    def velocity(self):
        return np.real(np.fft.ifft2(self.uh)), np.real(np.fft.ifft2(self.vh))

    def speed(self):
        u, v = self.velocity()
        return np.hypot(u, v)

    def vorticity(self):
        """omega = dv/dx - du/dy."""
        return np.real(np.fft.ifft2(1j * self.kx * self.vh - 1j * self.ky * self.uh))

    def terms(self):
        """Magnitude of each force in du/dt: advection, diffusion, pressure.

        Pressure is recovered as exactly the part the Leray projection removes,
        so |pressure| shows where the flow has to push against itself to stay
        incompressible (e.g. stagnation points, obstacle faces).
        """
        _, _, au, av, du, dv, pu, pv = self._rhs_components()
        Ru, Rv = np.fft.fft2(au + du + pu), np.fft.fft2(av + dv + pv)
        divR = self.kx * Ru + self.ky * Rv
        pfx = np.real(np.fft.ifft2(self.kx * divR * self.ksq_inv))   # pressure force
        pfy = np.real(np.fft.ifft2(self.ky * divR * self.ksq_inv))
        return {
            "advection": np.hypot(au, av),
            "diffusion": np.hypot(du, dv),
            "pressure": np.hypot(pfx, pfy),
        }

    # ------------------------------------------------------------------
    #  Scalar diagnostics (tie the picture to the equations).
    # ------------------------------------------------------------------
    def kinetic_energy(self):
        u, v = self.velocity()
        return 0.5 * np.mean(u ** 2 + v ** 2)

    def enstrophy(self):
        w = self.vorticity()
        return 0.5 * np.mean(w ** 2)

    def dissipation(self):
        return 2.0 * self.nu * self.enstrophy()

    def max_divergence(self):
        d = np.real(np.fft.ifft2(1j * (self.kx * self.uh + self.ky * self.vh)))
        return float(np.max(np.abs(d)))

    def reynolds(self):
        return self.U0 / self.nu

    # ------------------------------------------------------------------
    #  Correctness self-test: 2D Taylor-Green has an exact solution.
    # ------------------------------------------------------------------
    def tg_energy_analytic(self):
        """Exact mean KE for the Taylor-Green vortex: E0 * exp(-4 nu t)."""
        return 0.25 * np.exp(-4.0 * self.nu * self.time)   # E0 = 1/4 for this IC


def validate_taylor_green(N=128, nu=0.05, steps=200, tol=2e-2):
    """Run the TG vortex and compare measured energy decay to the exact law."""
    sim = NS2D(N=N, nu=nu, scenario="tg")
    for _ in range(steps):
        sim.step()
    measured = sim.kinetic_energy()
    exact = sim.tg_energy_analytic()
    rel = abs(measured - exact) / exact
    return rel, measured, exact, rel < tol


if __name__ == "__main__":
    rel, meas, exact, ok = validate_taylor_green()
    print(f"Taylor-Green check: measured={meas:.5e} exact={exact:.5e} "
          f"rel_err={rel:.2e} -> {'PASS' if ok else 'FAIL'}")
