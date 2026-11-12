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
`trapBeamId`), semiclassically, served by `rust-solver/motion` (crate
`atom-motion`, specified in `motion/SPEC.md`) through `src/motion_adapter.rs`.
A request whose model carries the `motion` block is a trajectory alone; only 3D
motion is served. Each trajectory is a quantum-jump trajectory of the internal
state, carried by a classical centre of mass: between jumps the atom moves under
the mean force on its internal state; a jump gives the emitted photon's recoil
and the mean momentum of the light it absorbed (one photon's at most); the rest
of the scattering force and the momentum diffusion act within each step. Each
optical drive is treated by its own detuning: its excited state is adiabatically
eliminated (a light shift and a scattering rate) or tracked explicitly, and the
states the tracked couplings join form the clusters the internal state lives on.

One constant time step serves the whole run. It is chosen on a few pilot
trajectories by a step-convergence check: they are run at a candidate step and
at twice it, and the step is halved while the difference exceeds the step's part
of the error budget. This is the check one makes by halving a step by hand, not
an error bound; the method line of a result says what it could not resolve.
Trajectories then run in batches until each output's total estimated error --
the model's approximations (the ledger, `diagnostics.approximations`), the step
and two standard errors -- meets the targets: 5% absolute in the populations and
the loss, 10% relative in the total temperature (T_x + T_y + T_z)/3. An output
that cannot meet them is returned and flagged. An atom is lost when it first
leaves the focused beams' region (every focused beam below 1/20 of its focal
intensity at the atom) with an energy above the value far away (SPEC §1.7).
The example 光镊 · 3D PGC cools the atom with sigma+sigma- pairs along x, y and z.

Known limits (SPEC §8). The centre of mass is a point that moves under the mean
force: it cannot carry the correlation between the momentum and the components
of the internal state that sub-Doppler cooling builds. The friction is exact;
the temperature of polarization-gradient cooling comes out low. Against the same
model's quantum answer in 1D, lin-perp-lin is 2% to 27% low and sigma+sigma- as
low as a quarter to a third of the true value; for six beams in 3D there is no
quantum reference and the bias is unknown. This is the model's own limit, not a
numerical error, and it is not in the error budget. Doppler cooling, recoil
heating and a closed transition in a standing wave have no such bias. A point
has no zero-point motion either: k_B T >= 4 hbar omega on every axis is the
validated regime, and outputs below it are marked reference-only.

This version computes on the CPU only. `options.device` (计算设备) is accepted:
a request for the GPU is computed on the CPU, `diagnostics.device` is "cpu" and
`diagnostics.deviceNote` says so. The worker's `gpu` operation answers at once
with the same sentence, so the window's wait for a kernel compile
(docs/ui-design.md §6.24) ends as soon as the worker has started. A GPU engine is
to follow as separate code.

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
