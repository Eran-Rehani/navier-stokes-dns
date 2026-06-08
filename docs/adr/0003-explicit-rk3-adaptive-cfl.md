# Explicit RK3 with adaptive CFL time-stepping

Time integration stays explicit (Williamson low-storage RK3) with the time step
limited adaptively by a CFL condition (dt ≈ C·dx/max|u|, also bounded by the
viscous limit). We considered treating the viscous term implicitly via an
integrating factor exp(−νk²dt) to remove the viscous stability limit, but
rejected it for now: it would change the time-stepping scheme and have to be
mirrored consistently across all four implementations, and the adaptive CFL step
already keeps the explorers stable across the full slider range. This is the
more reversible of our recorded decisions — if high-viscosity runs become a
bottleneck, revisit the integrating factor.
