#include "cpu_kernels.hpp"
#include <cstdio>
#include <cmath>
#include <vector>
#include <sys/stat.h>
#include <string>

using namespace ns3d::cpu;

#ifndef NS3D_N
#define NS3D_N 64
#endif

static constexpr int N = NS3D_N;
static constexpr double L = 2.0 * M_PI;
static constexpr int diag_interval = 100;
static constexpr int output_interval = 200;
static constexpr int k_cut = N / 3;
static constexpr int k_cut_sq = k_cut * k_cut;
static constexpr double RK_A[3] = {0.0, -5.0/9.0, -153.0/128.0};
static constexpr double RK_B[3] = {1.0/3.0, 15.0/16.0, 8.0/15.0};
static constexpr double dx = L / N;

/*  Binary snapshot format (little-endian):
 *      int32   N
 *      float64 u[N^3], v[N^3], w[N^3]   (index order [iz][iy][ix])
 *  The full velocity field is stored so post-processing can derive
 *  magnitude, vorticity and the energy spectrum. */
static void write_binary_output(const Real* const* u_phys, int step, int total) {
    mkdir("output", 0777);

    char filename[256];
    std::snprintf(filename, sizeof(filename), "output/vel_%06d.bin", step);

    FILE* fp = std::fopen(filename, "wb");
    if (!fp) {
        std::fprintf(stderr, "Failed to open %s for writing\n", filename);
        return;
    }

    int n_val = N;
    std::fwrite(&n_val, sizeof(int), 1, fp);
    for (int c = 0; c < 3; ++c)
        std::fwrite(u_phys[c], sizeof(double), total, fp);
    std::fclose(fp);

    std::printf("  Wrote output to %s\n", filename);
}

int main(int argc, char** argv) {
    double param_nu = 1.0e-3;
    double param_dt = 1.0e-3;
    int param_steps = 1000;
    std::string param_ic = "tg";
    double param_penalty = 0.0;
    double param_U0 = 1.5;      /* inflow / drive speed for BC scenarios */
    double param_force = 0.0;   /* streamwise body force (channel drive) */
    bool   param_adaptive = false;  /* adaptive CFL time stepping */
    double param_cfl = 0.4;     /* CFL number when adaptive */

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--nu" && i + 1 < argc) param_nu = std::stod(argv[++i]);
        else if (arg == "--dt" && i + 1 < argc) param_dt = std::stod(argv[++i]);
        else if (arg == "--steps" && i + 1 < argc) param_steps = std::stoi(argv[++i]);
        else if (arg == "--ic" && i + 1 < argc) param_ic = argv[++i];
        else if (arg == "--penalty" && i + 1 < argc) param_penalty = std::stod(argv[++i]);
        else if (arg == "--u0" && i + 1 < argc) param_U0 = std::stod(argv[++i]);
        else if (arg == "--force" && i + 1 < argc) param_force = std::stod(argv[++i]);
        else if (arg == "--adaptive-cfl") param_adaptive = true;
        else if (arg == "--cfl" && i + 1 < argc) param_cfl = std::stod(argv[++i]);
        else if (arg == "--help") {
            std::printf("Usage: %s [options]\n"
                        "  --ic <tg|shear|hit|cylinder|channel|jet|step>  initial condition / scenario\n"
                        "  --nu <val>        kinematic viscosity (lower = higher Reynolds number)\n"
                        "  --dt <val>        time step (fixed, unless --adaptive-cfl)\n"
                        "  --steps <n>       number of steps\n"
                        "  --penalty <eta>   Brinkman penalization time-scale for solid/inflow BCs\n"
                        "  --u0 <val>        inflow/drive speed (jet, step, channel)\n"
                        "  --force <f>       streamwise body force (channel drive)\n"
                        "  --adaptive-cfl    choose dt each step from a CFL stability limit\n"
                        "  --cfl <C>         CFL number for --adaptive-cfl (default 0.4)\n", argv[0]);
            return 0;
        }
    }

    /* Scenarios that impose boundary conditions via volume penalization. */
    const bool use_mask = (param_ic == "cylinder" || param_ic == "channel" ||
                           param_ic == "jet"      || param_ic == "step");

    if (use_mask && param_penalty == 0.0)
        param_penalty = (param_ic == "cylinder") ? 0.005 : 0.01;  /* safe defaults */
    if (param_ic == "channel" && param_force == 0.0) param_force = 0.01;  /* default drive */

    std::printf("──────────────────────────────────────────────\n");
    std::printf("  3D Pseudo-Spectral DNS (CPU Backend)\n");
    std::printf("  N = %d, nu = %g, dt = %g, steps = %d\n", N, param_nu, param_dt, param_steps);
    std::printf("  Initial Condition: %s\n", param_ic.c_str());
    std::printf("  OpenMP Threads: %d\n", omp_get_max_threads());
    std::printf("──────────────────────────────────────────────\n");
    
    int total = N * N * N;
    
    FFT3D fft(N);
    
    Complex* u_hat[3];
    Complex* Q[3];
    Real* u_phys[3];
    
    for (int c = 0; c < 3; ++c) {
        u_hat[c] = fftw_alloc_complex(total);
        Q[c] = fftw_alloc_complex(total);
        u_phys[c] = fftw_alloc_real(total);
        zero_complex(Q[c], total);
    }
    
    Complex* nl_hat = fftw_alloc_complex(total);
    Real* nl_phys = fftw_alloc_real(total);
    Complex* work = fftw_alloc_complex(total);
    Complex* dwork = fftw_alloc_complex(total);

    /* Penalization mask + target velocity (only used by BC scenarios). */
    Real* mask = nullptr;
    Real* target[3] = {nullptr, nullptr, nullptr};
    if (use_mask) {
        mask = fftw_alloc_real(total);
        for (int c = 0; c < 3; ++c) target[c] = fftw_alloc_real(total);
    }

    if (param_ic == "shear") {
        init_shear_layer(u_phys[0], u_phys[1], u_phys[2], N, dx);
    } else if (param_ic == "hit") {
        init_random(u_phys[0], u_phys[1], u_phys[2], N);
    } else if (param_ic == "cylinder") {
        init_cylinder_flow(u_phys[0], u_phys[1], u_phys[2], N, dx);
        /* legacy cylinder mask: disk handled by add_penalty(); also fill
           the generic mask so the unified penalty path can be used. */
        zero_real(mask, total);
        for (int c = 0; c < 3; ++c) zero_real(target[c], total);
        #pragma omp parallel for
        for (int tid = 0; tid < total; ++tid) {
            int iy = (tid / N) % N, ix = tid % N;
            double xx = ix * dx - M_PI, yy = iy * dx - M_PI;
            if (xx * xx + yy * yy <= 0.25) mask[tid] = 1.0;
        }
    } else if (param_ic == "channel") {
        init_channel(u_phys[0], u_phys[1], u_phys[2], mask,
                     target[0], target[1], target[2], N, dx, param_U0);
    } else if (param_ic == "jet") {
        init_jet(u_phys[0], u_phys[1], u_phys[2], mask,
                 target[0], target[1], target[2], N, dx, param_U0);
    } else if (param_ic == "step") {
        init_step(u_phys[0], u_phys[1], u_phys[2], mask,
                  target[0], target[1], target[2], N, dx, param_U0);
    } else {
        init_taylor_green(u_phys[0], u_phys[1], u_phys[2], N, dx);
    }
    
    for (int c = 0; c < 3; ++c) {
        real_to_complex(u_phys[c], work, total);
        fft.forward(work);
        std::memcpy(u_hat[c], work, total * sizeof(Complex));
    }
    
    project(u_hat[0], u_hat[1], u_hat[2], N);
    
    const double inv_N3 = 1.0 / (1.0 * N * N * N);
    
    for (int c = 0; c < 3; ++c) {
        std::memcpy(work, u_hat[c], total * sizeof(Complex));
        fft.backward(work);
        complex_to_real(work, u_phys[c], inv_N3, total);
    }
    
    double E0 = local_kinetic_energy(u_phys[0], u_phys[1], u_phys[2], total) * inv_N3;
    std::printf("  t = %10.6f   E_kin = %18.14e\n", 0.0, E0);
    write_binary_output(u_phys, 0, total);

    /* Adaptive-CFL helper: stable dt from advective / viscous / penalty limits. */
    const double kmax = N / 2.0;
    auto cfl_dt = [&](double umax) {
        double dt_adv  = param_cfl * dx / (umax + 1e-9);
        double dt_visc = 1.0 / (param_nu * kmax * kmax + 1e-12);
        double dt_pen  = use_mask ? 0.5 * param_penalty : 1e30;
        double d = dt_adv; if (dt_visc < d) d = dt_visc;
        if (dt_pen < d) d = dt_pen; if (d > 0.01) d = 0.01;
        return d;
    };

    double t_sim = 0.0;
    for (int step = 0; step < param_steps; ++step) {
        double dt = param_dt;
        if (param_adaptive) {
            for (int c = 0; c < 3; ++c) {
                std::memcpy(work, u_hat[c], total * sizeof(Complex));
                fft.backward(work);
                complex_to_real(work, u_phys[c], inv_N3, total);
            }
            dt = cfl_dt(max_speed(u_phys[0], u_phys[1], u_phys[2], total));
        }

        for (int s = 0; s < 3; ++s) {
            for (int c = 0; c < 3; ++c) {
                std::memcpy(work, u_hat[c], total * sizeof(Complex));
                dealias(work, k_cut_sq, N);
                fft.backward(work);
                complex_to_real(work, u_phys[c], inv_N3, total);
            }

            for (int comp = 0; comp < 3; ++comp) {
                zero_real(nl_phys, total);

                for (int dir = 0; dir < 3; ++dir) {
                    spectral_derivative(u_hat[comp], dwork, dir, N);
                    dealias(dwork, k_cut_sq, N);
                    fft.backward(dwork);
                    accum_product(nl_phys, u_phys[dir], dwork, inv_N3, total);
                }

                if (use_mask) {
                    add_penalty_masked(nl_phys, u_phys[comp], mask,
                                       target[comp], param_penalty, total);
                }
                if (param_ic == "channel" && comp == 0 && param_force != 0.0) {
                    add_body_force(nl_phys, param_force, total);  /* drive flow in x */
                }

                real_to_complex(nl_phys, work, total);
                fft.forward(work);
                std::memcpy(nl_hat, work, total * sizeof(Complex));

                rk3_substep(u_hat[comp], Q[comp], nl_hat, RK_A[s], RK_B[s], dt, param_nu, N);
            }

            project(u_hat[0], u_hat[1], u_hat[2], N);
        }
        t_sim += dt;

        if ((step + 1) % diag_interval == 0) {
            for (int c = 0; c < 3; ++c) {
                std::memcpy(work, u_hat[c], total * sizeof(Complex));
                fft.backward(work);
                complex_to_real(work, u_phys[c], inv_N3, total);
            }
            /* NaN/blow-up guard: abort cleanly rather than write garbage. */
            if (!field_finite(u_phys[0], total)) {
                std::fprintf(stderr,
                    "\n  ** instability: non-finite velocity at step %d (t=%.4f).\n"
                    "     Reduce --dt or use --adaptive-cfl.\n", step + 1, t_sim);
                return 2;
            }
            /* Incompressibility check: max |div u| should stay ~0. */
            zero_real(nl_phys, total);
            for (int dir = 0; dir < 3; ++dir) {
                spectral_derivative(u_hat[dir], dwork, dir, N);
                fft.backward(dwork);
                accum_real(nl_phys, dwork, inv_N3, total);
            }
            double div_max = max_abs(nl_phys, total);
            double Ek = local_kinetic_energy(u_phys[0], u_phys[1], u_phys[2], total) * inv_N3;
            std::printf("  t = %10.6f   E_kin = %18.14e   max|div u| = %.3e   dt = %.2e\n",
                        t_sim, Ek, div_max, dt);
            std::fflush(stdout);
        }
        
        if ((step + 1) % output_interval == 0) {
            for (int c = 0; c < 3; ++c) {
                std::memcpy(work, u_hat[c], total * sizeof(Complex));
                fft.backward(work);
                complex_to_real(work, u_phys[c], inv_N3, total);
            }
            write_binary_output(u_phys, step + 1, total);
        }
    }
    
    for (int c = 0; c < 3; ++c) {
        fftw_free(u_hat[c]);
        fftw_free(Q[c]);
        fftw_free(u_phys[c]);
    }
    fftw_free(nl_hat);
    fftw_free(nl_phys);
    fftw_free(work);
    fftw_free(dwork);
    if (use_mask) {
        fftw_free(mask);
        for (int c = 0; c < 3; ++c) fftw_free(target[c]);
    }

    return 0;
}
