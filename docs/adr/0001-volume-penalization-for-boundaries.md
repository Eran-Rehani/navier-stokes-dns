# Volume penalization to emulate boundaries

The solver is pseudo-spectral, so the numerical domain is a triply-periodic box
with no true walls. To show non-periodic effects (cylinders, channel walls,
inlets, a backward-facing step) we impose them with **volume penalization**
(the Brinkman immersed-boundary method): inside a masked region the velocity is
dragged toward a target on a short time-scale η. We chose this over building a
separate finite-difference solver with genuine Dirichlet/inflow boundaries
because it keeps the fast FFT method and a single shared algorithm across all
backends; the cost is that boundaries are approximate (smeared over a few cells)
and the far field is always periodic.
