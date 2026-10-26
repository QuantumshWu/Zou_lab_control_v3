# Atom population core

The one Rust implementation of the Rb87 full-density Lindblad solver used by
Atom Workbench. The TypeScript builder supplies all states, Hamiltonians,
collapse channels and exact frequency coordinates; this crate does not rebuild
atomic constants or remove physical couplings.

- `preparation*`: exact changes of variables (gauge phases, block Hilbert
  basis, optical shear S, recycling shear K) giving the pencil `E x' = F x` on
  Fourier envelope modes. Choices among exact alternatives are made by coupling
  magnitude, never by exact zeros.
- `owned_prepared` / `pencil`: the request-local operator library and the
  blocked E/F kernels; `b0_*`: the mode-diagonal P0 preconditioner.
- `runtime_core`: one prepared request (library, initial state, population
  readout duals, P0 factory, Fourier steps).
- `trial`, `directions`, `reduced`, `readout`, `engine`: the finite-time
  engine (rational Krylov trial space, inner GMRES, Galerkin model, readout,
  error-controlled growth of columns and Fourier modes).
- `slow_dual`, `error_estimate`: the dual-weighted residual estimates of the
  basis and Fourier-truncation (halo) errors.
- `asymptotic`: the long-time mean on the same pencil.

The parent build forwards `static-numerics` to link the OpenBLAS archive into
the single EXE; otherwise `ATOM_OPENBLAS_DLL` names a SciPy OpenBLAS DLL. BLAS
runs single-threaded; `ATOM_PENCIL_THREADS` sizes the worker pool. No test or
solve loads a saved trajectory as an initial guess or answer.
