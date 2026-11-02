# Atom population core

The one Rust implementation of the Rb87 full-density Lindblad solver used by
Atom Workbench. The TypeScript builder supplies all states, Hamiltonians,
collapse channels and exact frequency coordinates; this crate does not rebuild
atomic constants or remove physical couplings.

- `preparation`: the supplied model in the request's own frame, as Fourier
  components of H and the collapse channels, with frequencies kept as
  compensated (hi, lo) parts.
- `lattice`: the folded Fourier lattice (its module documentation lists the
  pipeline). Near charge steps are explicit lattice couplings; far ones (the
  tweezer, every counter-rotating harmonic) are folded statically at second
  order, and a run leaves out only the pieces of the fold whose action over
  its time is negligible whatever the initial state. Every output of a run
  (the trajectory on its grid, the means at the standard horizons, the
  slowest relaxation) is read from one Galerkin model of one rational Krylov
  space, accepted when its change between consecutive checkpoints (every 4
  solves) is within the fixed tolerance 0.01 and the space grew. Every
  checkpoint carries the run's estimate of its remaining wall time from the
  run itself (`pipeline::Progress`), reported only.
- `lapack`, `parallel`, `memory`: the LAPACK/BLAS wrappers, the worker pool
  and the installed memory the memory guard reads.

The parent build forwards `static-numerics` to link the OpenBLAS archive into
the single EXE; otherwise `ATOM_OPENBLAS_DLL` names a SciPy OpenBLAS DLL. BLAS
runs single-threaded; `ATOM_SOLVER_THREADS` sizes the worker pool. No test or
solve loads a saved trajectory as an initial guess or answer.
