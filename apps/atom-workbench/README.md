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
`rust-solver/population-core`; the worker in `rust-solver/src` only speaks the
JSONL protocol. The complete 32-state Hamiltonian, drives and collapse
operators of the TypeScript model are kept: the core changes variables
exactly (gauge, Fourier envelope, optical and recycling shears) and never
drops a coupling because it is small or labelled forbidden.

- Finite-time trajectories: a rational Krylov trial space on the gauge pencil
  with a Galerkin reduced model, grown until a dual-weighted residual estimate
  of the population error (trial-space basis plus Fourier truncation) meets the
  requested tolerance.
- Long-time means: the null vector of the same pencil (or, when the null space
  is degenerate, the projection of the initial state), with its own basis and
  Fourier-truncation estimates. For several lasers with incommensurate
  frequencies this is the time average of the quasi-periodic long-time state.

Every request asks for the same absolute population error,
`populationTolerance: 0.005` on the probability scale (0.5%); it is not a user
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
