# Atom population core

One Rust implementation of the fixed-manifold density-matrix evolution used
by Atom Workbench. The existing TypeScript builder supplies all states,
Hamiltonians, collapse channels and exact frequency coordinates. This crate
does not rebuild atomic constants or remove physical couplings.

`owned_prepared` owns request-local coefficients. `runtime_core::workspace`
extends trial spaces without repeating old operator products; `domain_transfer`
computes only newly required output rows. `finite_driver` combines this with
`population_control`. The default absolute population target is `0.005`;
the control supports `0 < epsilon <= 0.02` and reports empirical convergence,
not a certified universal error bound. Probability-output adjustments are
explicitly charged to the same target and do not repair the raw density.

The production worker and embedded standalone application depend on this
source directory. `static-numerics` must be forwarded by the parent build;
the existing OpenBLAS archive is linked into the single EXE. Python
and the former research driver are not runtime dependencies.

Unit tests are self-contained. Full32 fixture/performance tests are ignored
by default and require an explicit `ATOM_PREPARATION_FIXTURE` and report path.
No test or solver loads a saved trajectory as an initial guess or answer.
