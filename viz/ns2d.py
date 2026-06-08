"""
ns2d.py - 2D pseudo-spectral incompressible Navier-Stokes solver.

This is the *teaching* solver: a compact, readable Python mirror of the
C++ 3D DNS in the project root.  Same numerical method, one dimension
lower so it runs interactively on a laptop:

    * spatial      : Fourier (FFT) on a periodic [0, 2pi]^2 box
    * dealiasing   : Orszag 2/3 rule
    * time stepping: Williamson low-storage 3rd-order Runge-Kutta
    * incompressible: exact Leray projection each sub-step
    * boundaries   : volume penalization (Brinkman) immersed solids/inlets

Read this file alongside cpu_kernels.hpp - every routine here has a
direct counterpart there.  The physics is identical; only the syntax and
the dimensionality differ.
"""

import numpy as np

TWO_PI = 2.0 * np.pi

# Williamson (1980) low-storage RK3 coefficients - identical to the C++ code.
RK_A = np.array([0.0, -5.0 / 9.0, -153.0 / 128.0])
RK_B = np.array([1.0 / 3.0, 15.0 / 16.0, 8.0 / 15.0])


class NS2D:
    """A periodic 2D incompressible Navier-Stokes solver in spectral space.

    The unknown lives in spectral space (uh, vh).  Physical space is only
    visited to evaluate the nonlinear advection term (u . grad) u, exactly
    as in the 3D DNS.
    """

    def __init__(self, N=128, nu=2.0e-3, scenario="cylinder", U0=1.5):
        self.N = N
        self.nu = nu
        self.scenario = scenario
        self.U0 = U0
        self.dx = TWO_PI / N
        self.time = 0.0
        self.step_count = 0

        # Integer wavenumbers (L = 2pi -> k are integers), broadcast to 2D grids.
        k = np.fft.fftfreq(N, d=1.0 / N)            # 0,1,..,N/2-1,-N/2,..,-1
        self.kx = k[:, None] * np.ones((1, N))
        self.ky = np.ones((N, 1)) * k[None, :]
        self.ksq = self.kx ** 2 + self.ky ** 2
        self.ksq_inv = np.where(self.ksq > 0, 1.0 / np.where(self.ksq == 0, 1, self.ksq), 0.0)

        # Orszag 2/3 dealiasing mask.
        kcut = N // 3
        self.dealias_mask = (np.abs(k)[:, None] <= kcut) & (np.abs(k)[None, :] <= kcut)

        # Penalization fields (built per scenario): solid/inlet mask + target.
        self.mask = np.zeros((N, N))
        self.tu = np.zeros((N, N))
        self.tv = np.zeros((N, N))
        self.eta = 0.01            # penalization time-scale
        self.force = 0.0           # streamwise body force (channel drive)

        self.reset()

    # ------------------------------------------------------------------
    #  Scenario setup: initial field + penalization mask / target.
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
        self.force = 0.0
        self.time = 0.0
        self.step_count = 0
        s = self.scenario

        if s == "tg":                                  # Taylor-Green vortex
            u = np.sin(X) * np.cos(Y)
            v = -np.cos(X) * np.sin(Y)
        elif s == "shear":                             # double shear layer
            rho, delta = 30.0, 0.05
            u = np.where(Y <= np.pi,
                         np.tanh(rho * (Y - np.pi / 2)),
                         np.tanh(rho * (3 * np.pi / 2 - Y)))
            v = delta * np.sin(X)
        elif s == "hit":                               # decaying turbulence
            rng = np.random.default_rng(0)
            u = rng.standard_normal((N, N))
            v = rng.standard_normal((N, N))
        elif s == "cylinder":                          # flow past a cylinder
            xc, yc, R = np.pi, np.pi, 0.5
            solid = (X - xc) ** 2 + (Y - yc) ** 2 <= R ** 2
            self.mask[solid] = 1.0
            u[:] = self.U0
            u[solid] = 0.0
        elif s == "channel":                           # plane channel, no-slip walls
            wall = 0.12 * TWO_PI
            solid = (Y < wall) | (Y > TWO_PI - wall)
            self.mask[solid] = 1.0
            yy = np.clip((Y - wall) / (TWO_PI - 2 * wall), 0, 1)
            u = self.U0 * 4.0 * yy * (1 - yy)
            v = 0.02 * self.U0 * np.sin(X)
            u[solid] = 0.0
            v[solid] = 0.0
            self.force = 0.01                          # body force drives the flow
        elif s == "jet":                               # central jet through a plate
            plate = 0.10 * TWO_PI
            slot = np.abs(Y - np.pi) < 0.35
            inplate = X < plate
            self.mask[inplate] = 1.0
            self.tu[inplate & slot] = self.U0          # velocity inlet
        elif s == "step":                              # backward-facing step
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

        self.uh = np.fft.fft2(u)
        self.vh = np.fft.fft2(v)
        self._project()
        self.Qu = np.zeros((N, N), dtype=complex)
        self.Qv = np.zeros((N, N), dtype=complex)

    # ------------------------------------------------------------------
    #  Core spectral operators (mirror cpu_kernels.hpp).
    # ------------------------------------------------------------------
    def _project(self):
        """Leray projection: remove the divergent part, u <- u - k (k.u)/|k|^2."""
        div = self.kx * self.uh + self.ky * self.vh
        self.uh -= self.kx * div * self.ksq_inv
        self.vh -= self.ky * div * self.ksq_inv

    def _nonlinear(self):
        """Return spectral (u.grad)u for both momentum components."""
        u = np.real(np.fft.ifft2(self.uh * self.dealias_mask))
        v = np.real(np.fft.ifft2(self.vh * self.dealias_mask))
        nl = []
        for comp_h in (self.uh, self.vh):
            ddx = np.real(np.fft.ifft2(1j * self.kx * comp_h * self.dealias_mask))
            ddy = np.real(np.fft.ifft2(1j * self.ky * comp_h * self.dealias_mask))
            nl.append(u * ddx + v * ddy)
        nlu, nlv = nl
        # Penalization: drag velocity to target inside solids / inlets.
        if self.mask.any():
            nlu += self.mask * (u - self.tu) / self.eta
            nlv += self.mask * (v - self.tv) / self.eta
        if self.force:
            nlu -= self.force                              # channel drive
        return np.fft.fft2(nlu), np.fft.fft2(nlv)

    def step(self, dt):
        """Advance one full RK3 step."""
        for s in range(3):
            nlu, nlv = self._nonlinear()
            rhs_u = -nlu - self.nu * self.ksq * self.uh
            rhs_v = -nlv - self.nu * self.ksq * self.vh
            self.Qu = RK_A[s] * self.Qu + dt * rhs_u
            self.Qv = RK_A[s] * self.Qv + dt * rhs_v
            self.uh += RK_B[s] * self.Qu
            self.vh += RK_B[s] * self.Qv
            self._project()
        self.time += dt
        self.step_count += 1

    # ------------------------------------------------------------------
    #  Diagnostics for plotting.
    # ------------------------------------------------------------------
    def velocity(self):
        u = np.real(np.fft.ifft2(self.uh))
        v = np.real(np.fft.ifft2(self.vh))
        return u, v

    def vorticity(self):
        """omega = dv/dx - du/dy (the field that makes flow structure visible)."""
        return np.real(np.fft.ifft2(1j * self.kx * self.vh - 1j * self.ky * self.uh))

    def kinetic_energy(self):
        u, v = self.velocity()
        return 0.5 * np.mean(u ** 2 + v ** 2)

    def reynolds(self):
        """A nominal Reynolds number Re = U L / nu (L = 1 characteristic length)."""
        return self.U0 / self.nu
