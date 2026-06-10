"""
ns3d.py - 3D pseudo-spectral incompressible Navier-Stokes solver.

The 3D twin of ns2d.py, sized for *live* low-resolution exploration
(N ~ 32-48) by the 3D Explorer.  Identical numerical method to ns2d.py and
to the C++ DNS (FFT, Orszag 2/3 dealiasing, Williamson RK3, Leray projection,
volume penalization), with the same robustness (adaptive CFL, blow-up guard)
and pedagogy (term decomposition, passive-tracer dye, live diagnostics).

Fields are indexed [ix, iy, iz]; x is the streamwise direction.
"""

import numpy as np

import diagnostics
import scenarios

TWO_PI = 2.0 * np.pi
RK_A = np.array([0.0, -5.0 / 9.0, -153.0 / 128.0])
RK_B = np.array([1.0 / 3.0, 15.0 / 16.0, 8.0 / 15.0])


class BlowUp(Exception):
    """Raised when the solution stops being finite (instability)."""


class NS3D:
    def __init__(self, N=32, nu=4.0e-3, scenario="cylinder",
                 U0=scenarios.DEFAULT_U0, cfl=scenarios.DEFAULT_CFL,
                 dye_diffusivity=scenarios.DEFAULT_DYE_DIFFUSIVITY,
                 eta=scenarios.DEFAULT_ETA):
        self.N = N
        self.nu = nu
        self.scenario = scenario
        self.U0 = U0
        self.cfl = cfl
        self.Dc = dye_diffusivity
        self.dx = TWO_PI / N
        self.time = 0.0
        self.step_count = 0

        k = np.fft.fftfreq(N, d=1.0 / N)
        self.kx, self.ky, self.kz = np.meshgrid(k, k, k, indexing="ij")
        self.ksq = self.kx ** 2 + self.ky ** 2 + self.kz ** 2
        ksq_safe = np.where(self.ksq == 0, 1.0, self.ksq)
        self.ksq_inv = np.where(self.ksq > 0, 1.0 / ksq_safe, 0.0)
        self.K = (self.kx, self.ky, self.kz)

        kcut = N // 3
        m = np.abs(k) <= kcut
        self.dealias_mask = m[:, None, None] & m[None, :, None] & m[None, None, :]

        sh = (N, N, N)
        self.mask = np.zeros(sh)
        self.target = [np.zeros(sh) for _ in range(3)]
        self.eta = eta
        self.force = 0.0
        self.dye = np.zeros(sh)
        self.dye_source = np.zeros(sh)

        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        N, dx = self.N, self.dx
        x = np.arange(N) * dx
        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
        self.dye[:] = 0.0
        self.time = 0.0
        self.step_count = 0

        setup = scenarios.build(self.scenario, (X, Y, Z), self.U0)
        self.mask[:] = setup.mask
        for i in range(3):
            self.target[i][:] = setup.target[i]
        self.force = setup.force
        self.dye_source[:] = setup.dye_source

        self.uh = [np.fft.fftn(f) for f in setup.velocity]
        self._project()
        self.Q = [np.zeros((N, N, N), dtype=complex) for _ in range(3)]
        self.Qc = np.zeros((N, N, N), dtype=complex)

    # ------------------------------------------------------------------
    def _project(self):
        div = sum(self.K[i] * self.uh[i] for i in range(3))
        for i in range(3):
            self.uh[i] -= self.K[i] * div * self.ksq_inv

    def _rhs_components(self):
        u = [np.real(np.fft.ifftn(self.uh[i] * self.dealias_mask)) for i in range(3)]
        adv, diff, pen = [], [], []
        for i in range(3):
            a = np.zeros((self.N,) * 3)
            for j in range(3):
                dj = np.real(np.fft.ifftn(1j * self.K[j] * self.uh[i] * self.dealias_mask))
                a -= u[j] * dj
            adv.append(a)
            diff.append(np.real(np.fft.ifftn(-self.nu * self.ksq * self.uh[i])))
            p = -self.mask * (u[i] - self.target[i]) / self.eta if self.mask.any() \
                else np.zeros((self.N,) * 3)
            if self.force and i == 0:
                p = p + self.force
            pen.append(p)
        return u, adv, diff, pen

    # ------------------------------------------------------------------
    def cfl_dt(self):
        u = [np.real(np.fft.ifftn(self.uh[i])) for i in range(3)]
        umax = float(np.sqrt(np.max(u[0] ** 2 + u[1] ** 2 + u[2] ** 2))) + 1e-9
        kmax = self.N / 2.0
        dt_adv = self.cfl * self.dx / umax
        dt_visc = 1.0 / (self.nu * kmax ** 2 + 1e-12)
        dt_pen = 0.5 * self.eta if self.mask.any() else np.inf
        return float(min(dt_adv, dt_visc, dt_pen, 0.01))

    def step(self, dt=None):
        if dt is None:
            dt = self.cfl_dt()
        for s in range(3):
            u, adv, diff, pen = self._rhs_components()
            for i in range(3):
                rhs = np.fft.fftn(adv[i] + diff[i] + pen[i])
                self.Q[i] = RK_A[s] * self.Q[i] + dt * rhs
                self.uh[i] += RK_B[s] * self.Q[i]
            self._project()
            ch = np.fft.fftn(self.dye)
            cgrad = sum(u[j] * np.real(np.fft.ifftn(1j * self.K[j] * ch * self.dealias_mask))
                        for j in range(3))
            rhs_c = np.fft.fftn(-cgrad) - self.Dc * self.ksq * ch
            self.Qc = RK_A[s] * self.Qc + dt * rhs_c
            self.dye = np.real(np.fft.ifftn(np.fft.fftn(self.dye) + RK_B[s] * self.Qc))
        self.dye += self.dye_source * dt * 2.0
        np.clip(self.dye, 0.0, 1.0, out=self.dye)
        self.time += dt
        self.step_count += 1
        if not np.isfinite(self.uh[0]).all():
            raise BlowUp(f"non-finite field at t={self.time:.3f}")
        return dt

    def puff(self, center=None, r=0.5):
        x = np.arange(self.N) * self.dx
        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
        c = (np.pi, np.pi, np.pi) if center is None else center
        self.dye += np.exp(-((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2) / (2 * r ** 2))
        np.clip(self.dye, 0.0, 1.0, out=self.dye)

    # ------------------------------------------------------------------
    def velocity(self):
        return [np.real(np.fft.ifftn(self.uh[i])) for i in range(3)]

    def speed(self):
        u = self.velocity()
        return np.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)

    def vorticity(self):
        """Vector vorticity components (wx, wy, wz)."""
        return diagnostics.curl_3d(*self.uh, self.kx, self.ky, self.kz)

    def vorticity_magnitude(self):
        wx, wy, wz = self.vorticity()
        return np.sqrt(wx ** 2 + wy ** 2 + wz ** 2)

    def vorticity_view(self):
        """Displayable vorticity field (|curl u| in 3D)."""
        return self.vorticity_magnitude()

    def _divR_hat(self):
        """Spectral (k . R_hat) of the unprojected RHS; see ns2d._divR_hat."""
        _, adv, diff, pen = self._rhs_components()
        R = [np.fft.fftn(adv[i] + diff[i] + pen[i]) for i in range(3)]
        return adv, diff, sum(self.K[i] * R[i] for i in range(3))

    def pressure_field(self):
        """The pressure p itself (zero-mean), from lap(p) = div(R).
        Its low-p isosurfaces trace vortex cores - the classic pressure-minimum
        vortex criterion."""
        _, _, divR = self._divR_hat()
        return np.real(np.fft.ifftn(-1j * divR * self.ksq_inv))

    def terms(self):
        adv, diff, divR = self._divR_hat()
        gp = [np.real(np.fft.ifftn(self.K[i] * divR * self.ksq_inv)) for i in range(3)]
        mag = lambda f: np.sqrt(sum(c ** 2 for c in f))
        return {
            "advection": mag(adv),
            "diffusion": mag(diff),
            "pressure force": mag(gp),
        }

    # ------------------------------------------------------------------
    def kinetic_energy(self):
        return diagnostics.kinetic_energy(*self.velocity())

    def enstrophy(self):
        return diagnostics.enstrophy(*self.vorticity())

    def dissipation(self):
        return diagnostics.dissipation(self.nu, self.enstrophy())

    def max_divergence(self):
        return diagnostics.max_divergence(self.uh, self.K)

    def reynolds(self):
        return self.U0 / self.nu


if __name__ == "__main__":
    import time as _t
    sim = NS3D(N=32, nu=4e-3, scenario="cylinder")
    t0 = _t.time()
    for _ in range(20):
        sim.step()
    dt_wall = (_t.time() - t0) / 20
    print(f"3D N=32: {dt_wall*1000:.0f} ms/step, E={sim.kinetic_energy():.3f} "
          f"|w|max={sim.vorticity_magnitude().max():.1f} div={sim.max_divergence():.1e}")
