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

## Stable code and research

The stable numerical source is recorded against its validated baseline and
explicit overlays. New numerical experiments stay in the research worktree
until they pass independent accuracy and performance comparisons. Reference
calculations using QuTiP/Python are development evidence, not shipped runtime
components.

The accuracy target is an absolute population error of `1e-3`. The configured
32-state Hamiltonian and collapse operators are retained. Near-resonant lasers
participate in the master equation; user-designated far-detuned lasers contribute
the existing Stark model. Long-time results may be a phase average and may
depend on the initial state. A failed convergence check must remain a visible
failure, not a fabricated population result.

Performance work remains active: some fully coupled multi-frequency cases are
still slow. Short-window or closed-transition checks do not establish accuracy
or speed for every long trajectory.
