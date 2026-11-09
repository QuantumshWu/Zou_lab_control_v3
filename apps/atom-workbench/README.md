# Atom Workbench

Rb87 energy levels, laser couplings and density-matrix dynamics, developed with
Zou Lab Control v3 and distributed as an independent Windows application.

## Run and share

Run `atom_workbench.exe` at the repository root. The same EXE can be copied to
another computer: the interface, Rust solver and numerical libraries are inside
it. It uses the system WebView2 Runtime. Application resources are read in
memory; WebView2 keeps its normal user profile and saved scenes in the user's
application-data directory.

The executable is generated and ignored by Git. No v3 experiment window, Node.js
or Python installation is needed to run it.

## Build

With the v3 development tools installed, run:

```bat
bin\scripts\build_atom_workbench.bat
```

The build reads the current shared v3 UI sources, compiles the interface and
native worker, embeds the resources, and replaces the root EXE only after the
new build succeeds. `-Run` starts it after building. `-RebuildNumerics` also
rebuilds the pinned native numerical libraries.

The first build downloads checksum-pinned build dependencies into `.cache/`.
These inputs and build products are not application runtime dependencies.

## Source layout

| Path | Responsibility |
|---|---|
| `src/`, `index.html`, `style.css` | Workbench interface and atomic model construction |
| `rust-solver/` | Density-matrix numerical kernel and its scoped tests |
| `desktop-native/` | Standalone WebView2 host, embedded worker and in-memory resources |
| `native/` | Native dependency lock, small bridge sources and redistribution notices |
| `scripts/` | Reproducible build, package and validation entry points |
| `tests/` | Interface, model and protocol regression tests |
| `../desktop/ui/` | Shared v3 widgets, typography and style tokens; no copied widget fork |

The workbench remains a separate Cargo/npm project. Building the experiment
control application does not require building this tool. The repository-level
entry is `tools/atom_workbench.ps1`; the independent package entry is
`npm run package:single-exe`.

## Numerical scope

Every request (schema `rb87-full/2`) is solved by the population core in
`rust-solver/population-core`; the worker in `rust-solver/src` speaks the
JSONL protocol and defines the one result contract
(`lattice_core_adapter.rs`). The complete 32-state Hamiltonian, drives and
collapse operators of the TypeScript model are kept: the core works on the
Fourier envelopes of the request's own frame and never drops a coupling
because it is small or labelled forbidden.

- The folded Fourier lattice: near-resonant charge steps (the beats between
  lasers) are explicit couplings between lattice modes; far steps (the
  tweezer and every counter-rotating harmonic) are folded statically at
  second order into the near couplings, and every remainder of the fold is
  charged as a bound.
- Long-time means p̄(T) at T = 100 ms and 1 s (there is no t -> infinity
  mean): the state at T of the initial density with only its fast
  oscillations averaged out, from one run for both horizons, each with its
  own estimated error. The slowest eigenpairs give the slowest relaxation
  time: at a T shorter than it p̄(T) is still relaxing.
- Models without lasers are static: the same pipeline on the lattice's
  centre block alone.
- Finite-time trajectories, up to 10 s with at most 2000 samples (a longer
  or denser grid is refused with the reason): rational Krylov with lattice-translation
  augmentation and a Galerkin reduced model, its basis, near-shell and fold
  errors propagated through the dressed blocks; beyond 1 ms the trajectory
  continues as the expansion in the slow eigenmodes.

Motion (运动, on by default; docs/ui-design.md §6.20–§6.21): a single atom moving in 3D
in the optical tweezer the operator designates (设为光镊; the motion block's
`trapBeamId`), semiclassically, served
by `rust-solver/motion` (crate `atom-motion`, specified in `motion/SPEC.md`,
through `src/motion_adapter.rs`; its `model` eliminates the excited states). A request whose model carries the `motion`
block is a trajectory alone; only 3D motion is served. Each trajectory is a
classical point carrying the internal state of the eliminated ground manifold
on one cluster of states, evolved by quantum jumps: dipole-pattern emission
recoil, beam-resolved absorption recoil with the exact mean force, the
Ehrenfest force between jumps. Every coupling between ground states is folded
at second order (van Vleck) or kept by one rule; the states kept couplings join
form the clusters (whole hyperfine levels near B = 0, single sublevels at large
fields, F = 1 and F = 2 together through a Raman-resonant pair). The trap's
harmonic part is moved exactly, the time step comes from step doubling
(populations, impulse and friction), and trajectories run in batches until each
output's total estimated error -- the approximation ledger
(`diagnostics.approximations`), the step's bias and two standard errors --
meets 5% in the populations and 10% in T; an output that cannot is flagged. An
atom is lost when its mechanical energy in the trap is positive. A point
particle has no zero-point motion: results are validated for k_B T >= 4 hbar
omega on every axis and marked reference-only below. The Ehrenfest force averages
the dipole force's fluctuations in a superposition of ground states, so the
momentum diffusion at a fixed point of a polarization gradient is low (0.23-0.27
of the second-order expansion's at a lin-perp-lin focus); in the tweezer the
trapped T stays within a few percent of the 1D GMW references in the validated
regime. The example 光镊 · 3D PGC
cools the atom with sigma+sigma- pairs along x, y and z; with motion on, the
Raman-resonant Lambda of 光镊 + 探测光 + 回泵光 runs on one cluster spanning
F = 1 and F = 2.

The trajectories run on the device the request names (`options.device`, 计算设备):
an NVIDIA GPU by default, or the CPU. The GPU engine (`motion-3d/src/gpu`,
`kernels/motion3d.cu`) runs the CPU engine's steps, jumps, losses and readouts
with the same initial states, random streams, ledger, time step and statistics,
one warp per trajectory, single precision where a step is local and double where
it accumulates. It needs compute capability 7.5 (Turing) or newer and a driver
of CUDA 12.0 (R525, Windows 527.41) or newer: the EXE embeds PTX for compute_75,
compiled by a pinned NVRTC 12.0 (`motion-3d/scripts/ptx.ps1`; a test checks it
against the source), and loads `nvcuda.dll` at run time, the driver API alone.
Without such a GPU, with an older driver or with a model beyond the kernel the
solve runs on the CPU and `diagnostics.deviceNote` says why; `diagnostics.device`
is the device used. The worker keeps its CUDA context for every later solve. The
driver compiles the PTX at its first load, and after a driver update (a minute or
two): the worker's `gpu` operation answers when it has, and the window waits
frozen meanwhile, a moving bar and the elapsed time on it (docs/ui-design.md §6.24).

Every request asks for the same absolute population error,
`populationTolerance: 0.01` on the probability scale (1%); it is not a user
setting. The estimates are calibrated a-posteriori estimates (checked against
independent references), not certified bounds; a safety factor of 1.5 is
applied. A result that does not reach the target is reported as a failure,
never as a relaxed result.

Every enabled laser is an explicit drive of the master equation, with its co-
and counter-rotating terms on every allowed D1/D2 dipole, whatever its
detuning. Spontaneous emission has one collapse operator per photon
polarization across both D lines and both ground F (no secular
approximation). The static H0 holds the magnetic field and, for every laser,
the second-order shift through the states outside the 32-state model and the
Rb⁺ core polarizability. A laser's "显示光移" switch only adds its all-order
static light shift to the levels chart's background; it never changes the
dynamics. Reference calculations using QuTiP/Python are development evidence,
not shipped runtime components.
