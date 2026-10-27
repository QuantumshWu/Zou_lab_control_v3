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
  order and every remainder is charged. On it: the long-time mean (`steady`),
  the slow eigenpairs (`slow`) and the finite-time trajectory (`finite`),
  each accepted by its a-posteriori error ledger (`ledger`) at the fixed
  tolerance 0.005.
- `lapack`, `dense_expm`, `parallel`: the LAPACK/BLAS wrappers, the dense
  exponential of the reduced model and the worker pool.

The parent build forwards `static-numerics` to link the OpenBLAS archive into
the single EXE; otherwise `ATOM_OPENBLAS_DLL` names a SciPy OpenBLAS DLL. BLAS
runs single-threaded; `ATOM_SOLVER_THREADS` sizes the worker pool. The
`validation` feature adds the exact dense dressed preconditioner that the
structured one is measured against. No test or solve loads a saved trajectory
as an initial guess or answer.
