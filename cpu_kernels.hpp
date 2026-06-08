#pragma once

#include <fftw3.h>
#include <omp.h>
#include <cmath>
#include <cstring>
#include <vector>

namespace ns3d { namespace cpu {

using Real = double;
using Complex = fftw_complex;

inline int wavenumber(int idx, int Ndim) {
    return (idx <= Ndim / 2) ? idx : idx - Ndim;
}

class FFT3D {
public:
    int N;
    fftw_plan p_forward;
    fftw_plan p_backward;

    FFT3D(int N_val) : N(N_val) {
        fftw_init_threads();
        fftw_plan_with_nthreads(omp_get_max_threads());
        
        Complex* dummy = fftw_alloc_complex(N * N * N);
        p_forward = fftw_plan_dft_3d(N, N, N, dummy, dummy, FFTW_FORWARD, FFTW_MEASURE);
        p_backward = fftw_plan_dft_3d(N, N, N, dummy, dummy, FFTW_BACKWARD, FFTW_MEASURE);
        fftw_free(dummy);
    }

    ~FFT3D() {
        fftw_destroy_plan(p_forward);
        fftw_destroy_plan(p_backward);
        fftw_cleanup_threads();
    }

    void forward(Complex* data) {
        fftw_execute_dft(p_forward, data, data);
    }

    void backward(Complex* data) {
        fftw_execute_dft(p_backward, data, data);
    }
};

inline void spectral_derivative(const Complex* src, Complex* dst, int dir, int N) {
    const int total = N * N * N;
    #pragma omp parallel for
    for (int tid = 0; tid < total; ++tid) {
        int iz = tid / (N * N);
        int iy = (tid / N) % N;
        int ix = tid % N;
        
        int k;
        switch(dir) {
            case 0: k = wavenumber(ix, N); break;
            case 1: k = wavenumber(iy, N); break;
            case 2: k = wavenumber(iz, N); break;
        }
        
        // i * (re + i*im) * k = (-im + i*re) * k
        double re = src[tid][0];
        double im = src[tid][1];
        dst[tid][0] = -im * k;
        dst[tid][1] = re * k;
    }
}

inline void dealias(Complex* data, int k_cut_sq, int N) {
    const int total = N * N * N;
    #pragma omp parallel for
    for (int tid = 0; tid < total; ++tid) {
        int iz = tid / (N * N);
        int iy = (tid / N) % N;
        int ix = tid % N;
        
        int kxv = wavenumber(ix, N);
        int kyv = wavenumber(iy, N);
        int kzv = wavenumber(iz, N);
        
        if (kxv*kxv + kyv*kyv + kzv*kzv > k_cut_sq) {
            data[tid][0] = 0.0;
            data[tid][1] = 0.0;
        }
    }
}

inline void project(Complex* u_hat, Complex* v_hat, Complex* w_hat, int N) {
    const int total = N * N * N;
    #pragma omp parallel for
    for (int tid = 0; tid < total; ++tid) {
        int iz = tid / (N * N);
        int iy = (tid / N) % N;
        int ix = tid % N;
        
        double kxv = wavenumber(ix, N);
        double kyv = wavenumber(iy, N);
        double kzv = wavenumber(iz, N);
        double ksq = kxv*kxv + kyv*kyv + kzv*kzv;
        
        if (ksq < 1e-14) continue;
        
        double div_re = kxv * u_hat[tid][0] + kyv * v_hat[tid][0] + kzv * w_hat[tid][0];
        double div_im = kxv * u_hat[tid][1] + kyv * v_hat[tid][1] + kzv * w_hat[tid][1];
        
        double inv_ksq = 1.0 / ksq;
        
        u_hat[tid][0] -= kxv * div_re * inv_ksq;
        u_hat[tid][1] -= kxv * div_im * inv_ksq;
        v_hat[tid][0] -= kyv * div_re * inv_ksq;
        v_hat[tid][1] -= kyv * div_im * inv_ksq;
        w_hat[tid][0] -= kzv * div_re * inv_ksq;
        w_hat[tid][1] -= kzv * div_im * inv_ksq;
    }
}

inline void rk3_substep(Complex* u_hat, Complex* Q, const Complex* nl_hat, 
                        double A, double B, double dt, double nu, int N) {
    const int total = N * N * N;
    #pragma omp parallel for
    for (int tid = 0; tid < total; ++tid) {
        int iz = tid / (N * N);
        int iy = (tid / N) % N;
        int ix = tid % N;
        
        double kxv = wavenumber(ix, N);
        double kyv = wavenumber(iy, N);
        double kzv = wavenumber(iz, N);
        double ksq = kxv*kxv + kyv*kyv + kzv*kzv;
        
        double rhs_re = -nl_hat[tid][0] - nu * ksq * u_hat[tid][0];
        double rhs_im = -nl_hat[tid][1] - nu * ksq * u_hat[tid][1];
        
        Q[tid][0] = A * Q[tid][0] + dt * rhs_re;
        Q[tid][1] = A * Q[tid][1] + dt * rhs_im;
        
        u_hat[tid][0] += B * Q[tid][0];
        u_hat[tid][1] += B * Q[tid][1];
    }
}

inline void pressure_poisson(Complex* p_hat, const Complex* u_hat, const Complex* v_hat, const Complex* w_hat, double rho_v, double dt_v, int N) {
    const int total = N * N * N;
    #pragma omp parallel for
    for (int tid = 0; tid < total; ++tid) {
        int iz = tid / (N * N);
        int iy = (tid / N) % N;
        int ix = tid % N;
        
        double kxv = wavenumber(ix, N);
        double kyv = wavenumber(iy, N);
        double kzv = wavenumber(iz, N);
        double ksq = kxv*kxv + kyv*kyv + kzv*kzv;
        
        if (ksq < 1e-14) {
            p_hat[tid][0] = 0.0;
            p_hat[tid][1] = 0.0;
            continue;
        }
        
        double div_re = kxv * u_hat[tid][0] + kyv * v_hat[tid][0] + kzv * w_hat[tid][0];
        double div_im = kxv * u_hat[tid][1] + kyv * v_hat[tid][1] + kzv * w_hat[tid][1];
        
        double coeff = rho_v / (dt_v * ksq);
        p_hat[tid][0] = coeff * div_im;
        p_hat[tid][1] = -coeff * div_re;
    }
}

inline void accum_product(Real* nl, const Real* u_j, const Complex* work, double norm, int count) {
    #pragma omp parallel for
    for (int tid = 0; tid < count; ++tid) {
        nl[tid] += u_j[tid] * work[tid][0] * norm;
    }
}

inline void zero_real(Real* arr, int count) {
    #pragma omp parallel for
    for (int tid = 0; tid < count; ++tid) {
        arr[tid] = 0.0;
    }
}

inline void zero_complex(Complex* arr, int count) {
    #pragma omp parallel for
    for (int tid = 0; tid < count; ++tid) {
        arr[tid][0] = 0.0;
        arr[tid][1] = 0.0;
    }
}

inline void real_to_complex(const Real* src, Complex* dst, int count) {
    #pragma omp parallel for
    for (int tid = 0; tid < count; ++tid) {
        dst[tid][0] = src[tid];
        dst[tid][1] = 0.0;
    }
}

inline void complex_to_real(const Complex* src, Real* dst, double norm, int count) {
    #pragma omp parallel for
    for (int tid = 0; tid < count; ++tid) {
        dst[tid] = src[tid][0] * norm;
    }
}

inline void init_taylor_green(Real* u, Real* v, Real* w, int N, double dx) {
    const int total = N * N * N;
    #pragma omp parallel for
    for (int tid = 0; tid < total; ++tid) {
        int iz = tid / (N * N);
        int iy = (tid / N) % N;
        int ix = tid % N;
        
        double x = ix * dx;
        double y = iy * dx;
        double z = iz * dx;
        
        u[tid] = std::sin(x) * std::cos(y) * std::cos(z);
        v[tid] = -std::cos(x) * std::sin(y) * std::cos(z);
        w[tid] = 0.0;
    }
}

inline void init_shear_layer(Real* u, Real* v, Real* w, int N, double dx) {
    const int total = N * N * N;
    const double rho_shear = 30.0;
    const double delta = 0.05;
    
    #pragma omp parallel for
    for (int tid = 0; tid < total; ++tid) {
        int iz = tid / (N * N);
        int iy = (tid / N) % N;
        int ix = tid % N;
        
        double x = ix * dx;
        double y = iy * dx;
        double z = iz * dx;
        
        if (y <= M_PI) {
            u[tid] = std::tanh(rho_shear * (y - M_PI / 2.0));
        } else {
            u[tid] = std::tanh(rho_shear * (3.0 * M_PI / 2.0 - y));
        }
        
        v[tid] = delta * std::sin(x) * std::sin(z);
        w[tid] = 0.0;
    }
}

inline double hash_rand(int ix, int iy, int iz, int c) {
    unsigned int n = (unsigned int)ix + ((unsigned int)iy << 10) + ((unsigned int)iz << 20) + ((unsigned int)c << 30);
    n = (n << 13) ^ n;
    n = (n * (n * n * 15731 + 789221) + 1376312589);
    return 2.0 * ((double)(n & 0x7fffffff) / 2147483647.0) - 1.0;
}

inline void init_random(Real* u, Real* v, Real* w, int N) {
    const int total = N * N * N;
    #pragma omp parallel for
    for (int tid = 0; tid < total; ++tid) {
        int iz = tid / (N * N);
        int iy = (tid / N) % N;
        int ix = tid % N;
        
        u[tid] = hash_rand(ix, iy, iz, 0);
        v[tid] = hash_rand(ix, iy, iz, 1);
        w[tid] = hash_rand(ix, iy, iz, 2);
    }
}

inline void init_cylinder_flow(Real* u, Real* v, Real* w, int N, double dx) {
    const int total = N * N * N;
    const double xc = M_PI;
    const double yc = M_PI;
    const double R2 = 0.5 * 0.5;
    
    #pragma omp parallel for
    for (int tid = 0; tid < total; ++tid) {
        int iz = tid / (N * N);
        int iy = (tid / N) % N;
        int ix = tid % N;
        
        double x = ix * dx;
        double y = iy * dx;
        
        double dx_c = x - xc;
        double dy_c = y - yc;
        
        if (dx_c * dx_c + dy_c * dy_c <= R2) {
            u[tid] = 0.0;
            v[tid] = 0.0;
            w[tid] = 0.0;
        } else {
            u[tid] = 2.0 + 0.1 * hash_rand(ix, iy, iz, 0);
            v[tid] = 0.1 * hash_rand(ix, iy, iz, 1);
            w[tid] = 0.1 * hash_rand(ix, iy, iz, 2);
        }
    }
}

/* ----------------------------------------------------------------
 *  Scenario builders for penalization-based boundary conditions.
 *
 *  Each fills:  u,v,w   initial velocity (fluid region)
 *               mask    1 inside solid/forced region, else 0
 *               tu,tv,tw target velocity inside that region
 *  All geometries are 2D in (x,y) and periodic (extruded) in z.
 *  Domain is [0, 2π]^3 with x the streamwise direction.
 * ---------------------------------------------------------------- */

/*  Plane channel: no-slip walls at the bottom/top y-strips, flow
 *  driven downstream by a constant body force (see solver).        */
inline void init_channel(Real* u, Real* v, Real* w,
                         Real* mask, Real* tu, Real* tv, Real* tw,
                         int N, double dx, double U0) {
    const int total = N * N * N;
    const double wall = 0.12 * (2.0 * M_PI);          /* wall thickness */
    const double H    = 2.0 * M_PI;
    #pragma omp parallel for
    for (int tid = 0; tid < total; ++tid) {
        int iz = tid / (N * N);
        int iy = (tid / N) % N;
        int ix = tid % N;
        double x = ix * dx, y = iy * dx, z = iz * dx;

        bool solid = (y < wall) || (y > H - wall);
        mask[tid] = solid ? 1.0 : 0.0;
        tu[tid] = tv[tid] = tw[tid] = 0.0;

        if (solid) { u[tid] = v[tid] = w[tid] = 0.0; }
        else {
            /* parabolic-ish seed + small 3D perturbation to trip turbulence */
            double yy = (y - wall) / (H - 2.0 * wall);     /* 0..1 across channel */
            u[tid] = U0 * 4.0 * yy * (1.0 - yy);
            v[tid] = 0.02 * U0 * std::sin(x) * std::sin(z);
            w[tid] = 0.02 * U0 * std::cos(x) * std::cos(z);
        }
    }
}

/*  Free jet: a solid plate at x=0 with a central slot that injects
 *  fluid at speed U0 in +x.  Everything else starts at rest.       */
inline void init_jet(Real* u, Real* v, Real* w,
                     Real* mask, Real* tu, Real* tv, Real* tw,
                     int N, double dx, double U0) {
    const int total = N * N * N;
    const double plate = 0.10 * (2.0 * M_PI);   /* inlet plate thickness in x */
    const double yc    = M_PI;
    const double half  = 0.35;                   /* slot half-height */
    #pragma omp parallel for
    for (int tid = 0; tid < total; ++tid) {
        int iz = tid / (N * N);
        int iy = (tid / N) % N;
        int ix = tid % N;
        double x = ix * dx, y = iy * dx, z = iz * dx;

        bool in_plate = (x < plate);
        bool in_slot  = std::fabs(y - yc) < half;

        u[tid] = v[tid] = w[tid] = 0.0;
        if (in_plate) {
            mask[tid] = 1.0;
            if (in_slot) {                       /* velocity inlet */
                tu[tid] = U0 * (1.0 + 0.05 * std::sin(3.0 * z));
                tv[tid] = 0.0; tw[tid] = 0.0;
            } else {                             /* solid plate */
                tu[tid] = tv[tid] = tw[tid] = 0.0;
            }
        } else {
            mask[tid] = 0.0;
            tu[tid] = tv[tid] = tw[tid] = 0.0;
        }
    }
}

/*  Backward-facing step: uniform inflow over the upper half at x=0,
 *  a solid block (the step) filling the lower-left of the domain.   */
inline void init_step(Real* u, Real* v, Real* w,
                      Real* mask, Real* tu, Real* tv, Real* tw,
                      int N, double dx, double U0) {
    const int total = N * N * N;
    const double inlet = 0.08 * (2.0 * M_PI);
    const double yc    = M_PI;
    const double step_x = (2.0 * M_PI) / 3.0;    /* step extends to L/3 */
    #pragma omp parallel for
    for (int tid = 0; tid < total; ++tid) {
        int iz = tid / (N * N);
        int iy = (tid / N) % N;
        int ix = tid % N;
        double x = ix * dx, y = iy * dx, z = iz * dx;

        bool block = (x < step_x) && (y < yc);                  /* the step */
        bool inlet_plane = (x < inlet) && (y >= yc);            /* upper inflow */

        u[tid] = v[tid] = w[tid] = 0.0;
        mask[tid] = 0.0; tu[tid] = tv[tid] = tw[tid] = 0.0;

        if (block) { mask[tid] = 1.0; }                         /* target 0 */
        else if (inlet_plane) {
            mask[tid] = 1.0;
            tu[tid] = U0 * (1.0 + 0.03 * std::sin(2.0 * z));
        } else if (y >= yc) {
            u[tid] = U0;                                        /* seed upstream */
        }
    }
}

inline void add_penalty(Real* nl, const Real* u, int N, double dx, double eta) {
    const int total = N * N * N;
    const double xc = M_PI;
    const double yc = M_PI;
    const double R2 = 0.5 * 0.5;

    #pragma omp parallel for
    for (int tid = 0; tid < total; ++tid) {
        int iy = (tid / N) % N;
        int ix = tid % N;

        double x = ix * dx;
        double y = iy * dx;

        double dx_c = x - xc;
        double dy_c = y - yc;

        if (dx_c * dx_c + dy_c * dy_c <= R2) {
            nl[tid] += u[tid] / eta;
        }
    }
}

/* ----------------------------------------------------------------
 *  Generalised volume penalization (Brinkman immersed boundary).
 *
 *  Inside a solid/forced region (mask=1) the velocity is dragged
 *  towards a target field u* on the time-scale eta:
 *      du/dt  +=  mask * (u* - u) / eta
 *  In the RHS convention  du/dt = -nl - nu k^2 u, this means
 *      nl  +=  mask * (u - u*) / eta.
 *
 *  target = 0   -> no-slip wall / solid obstacle
 *  target = U0  -> velocity inlet (jet, channel inflow, ...)
 *  This is how the periodic spectral solver fakes non-periodic
 *  boundary conditions without abandoning the FFT.
 * ---------------------------------------------------------------- */
inline void add_penalty_masked(Real* nl, const Real* u, const Real* mask,
                               const Real* target, double eta, int count) {
    #pragma omp parallel for
    for (int tid = 0; tid < count; ++tid) {
        nl[tid] += mask[tid] * (u[tid] - target[tid]) / eta;
    }
}

/*  Constant body force on one momentum component (e.g. the mean
 *  pressure gradient that drives channel flow).  du/dt += f. */
inline void add_body_force(Real* nl, double f, int count) {
    #pragma omp parallel for
    for (int tid = 0; tid < count; ++tid) {
        nl[tid] -= f;
    }
}

inline Real local_kinetic_energy(const Real* u, const Real* v, const Real* w, int count) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int tid = 0; tid < count; ++tid) {
        sum += 0.5 * (u[tid]*u[tid] + v[tid]*v[tid] + w[tid]*w[tid]);
    }
    return sum;
}

/* ----------------------------------------------------------------
 *  Robustness helpers: adaptive CFL, divergence check, NaN guard.
 * ---------------------------------------------------------------- */

/*  Max velocity magnitude (for the advective CFL limit). */
inline Real max_speed(const Real* u, const Real* v, const Real* w, int count) {
    double m = 0.0;
    #pragma omp parallel for reduction(max:m)
    for (int tid = 0; tid < count; ++tid) {
        double s = u[tid]*u[tid] + v[tid]*v[tid] + w[tid]*w[tid];
        if (s > m) m = s;
    }
    return std::sqrt(m);
}

/*  True if every entry is finite (no NaN/Inf). */
inline bool field_finite(const Real* a, int count) {
    int bad = 0;
    #pragma omp parallel for reduction(+:bad)
    for (int tid = 0; tid < count; ++tid)
        if (!std::isfinite(a[tid])) bad += 1;
    return bad == 0;
}

/*  dst[i] += Re(src[i]) * norm   (used to assemble the divergence field). */
inline void accum_real(Real* dst, const Complex* src, double norm, int count) {
    #pragma omp parallel for
    for (int tid = 0; tid < count; ++tid)
        dst[tid] += src[tid][0] * norm;
}

inline Real max_abs(const Real* a, int count) {
    double m = 0.0;
    #pragma omp parallel for reduction(max:m)
    for (int tid = 0; tid < count; ++tid) {
        double v = std::fabs(a[tid]);
        if (v > m) m = v;
    }
    return m;
}

}} // namespace ns3d::cpu
