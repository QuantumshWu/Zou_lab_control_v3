# The programs, the engine and the scripts

A user double-clicks one of the four programs at the checkout's root:

| Program | Opens |
|---|---|
| `installation.exe` | Installation@Zou lab: what this PC has (the program -- this checkout's release of the engine --, the plugins' Python environment and instrument links, Basler pylon, the SLM GPU, Vivado and the FPGA board, under Advanced a developer's toolchain), each with its status; *Install selected* or *Fill in missing*, the SLM GPU's *Re-warm*, and the FPGA's *Estimate resources*, *Build*, *Program*, *Flash* and *Diagnose*. The output of each step streams into the window. Run it first. |
| `experiment.exe` | Devices@Zou lab; Init opens the Task Console, as in v2. |
| `figure_viewer.exe` | FigureViewer@Zou lab alone; a Figure archive dropped on it opens. |
| `pulse_editor.exe` | PulseGUI@Zou lab alone; a pulse document dropped on it opens. |

The four are one small launcher under four names: it starts the engine in
`bin\engine` in the mode its name says and ends. None has a console; what the
engine reports goes to `.local\logs` (below), and an error that stops it is
also shown in a message box. The engine is not in git: `bin\release.json` names
the release this checkout runs (its version, tag, the source stamp and commit
it was built from, and its archive's name, size and sha256). When `bin\engine`
is of another release, `experiment.exe`, `figure_viewer.exe` and
`pulse_editor.exe` do not start it but open `installation.exe`, which opens as
it is: its program row reads *Update*, and nothing is downloaded until the
operator installs it there. When there is no engine at all (or one from before
releases were downloaded), `installation.exe` asks first, then downloads this
checkout's release in a console window of its own. Either way the download is
`tools\fetch_engine.ps1`: the archive from the GitHub release, its size and
sha256 checked, unpacked beside the live engine as `bin\engine.next` and renamed
in. An engine still running is never replaced: the download waits in
`bin\engine.next` and goes in when the programs next start.

`bin` holds the rest:

- `bin\release.json` (in git): the release, written by `build_release.bat`.
- `bin\engine\` (downloaded, not in git): the engine -- `zlc-desktop.exe` (every window, and
  installation.exe's as `zlc-desktop --installer`), `zlc-pulse-server.exe` (the
  board on this computer, also the FPGA build's `fpga` command) and the UI page
  they serve (`ui\`).
- `bin\ai\zlc-mcp.exe` (downloaded with the engine): for AI clients only (MCP over stdio,
  `zlc-mcp --client-config claude`).
- `bin\scripts\*.bat`: every step for scripting and automation, the same
  `tools\*.ps1` installation.exe runs. Each runs from any folder and pauses on
  an error (set `ZLC_NO_PAUSE=1` for automation).

| Script | What it does |
|---|---|
| `install_requirements.bat [pylon \| slm-gpu] [--dry-run]` | Sets up `.tools\science-env`, the Python the device and logic-node plugins run in (below). |
| `install_slm_gpu.bat [--dry-run]` | Installs CuPy and the CUDA runtime for the SLM solver, then warms them on this GPU. |
| `warm_slm_gpu.bat` | The warm step alone, after a GPU, driver or solver change. |
| `build_and_program.bat [--build-only \| --program-only \| --flash \| --diagnose \| --force-build]` | Builds the pulse board's FPGA from `crates\zlc-pulse\fpga` and programs it (Vivado). |
| `estimate_resources.bat` | Whether the configured FPGA part has the LUT/FF/DSP/BRAM for the configured geometry. |
| `experiment.bat`, `figure_viewer.bat`, `pulse_editor.bat` | On a development machine: the three apps through `tools\desktop.ps1`, from a build current for the sources (below). |
| `build_release.bat [--no-commit \| --publish [--dry-run]]` | On a development machine: builds the release into `bin\engine` and `bin\ai` and its archive (`target\engine-releases\zlc-engine-<version>.zip`), and commits `bin\release.json` with the four programs; `--publish`, once that commit is pushed, makes its GitHub release with the archive (`gh`). |
| `dev_setup.bat [--dry-run]` | On a development machine: installs the toolchain into `.tools`. |

## Where things live

Everything stays inside the checkout:

- `.tools\` (not in git): `science-env` on every PC; on a development machine
  also the portable toolchain. `ZLC_V3_TOOLS` names another tools folder.
- `target\` (not in git): development builds.
- The four programs and `bin\release.json` (in git, written only by
  `build_release.bat`) and `bin\engine\` and `bin\ai\` (not in git: downloaded
  by `installation.exe`, or built by `build_release.bat`): the release.
- `.local\` (not in git): what running programs write -- `webview2` (the windows'
  browser profile), `mcp` (the MCP token and this checkout's running-instrument
  records), `logs` (zlc-desktop's errors and plugin output and installation.exe's
  steps, one file per run, the last 20 of each kept), `vivado` (the pulse server's
  JTAG session), and the plugins' `simulation`, `slm_gpu_cache` and `cupy` caches.
  `ZLC_V3_LOCAL` names another.

An executable finds the UI, `science\src`, the FPGA tree, fonts and recipes from
where it sits in the checkout, so a release built on one machine runs from any
clone. The four programs always run the release in `bin\engine`. On a development
machine `tools\desktop.ps1` (and `bin\scripts\experiment.bat`) starts the
development build when it was built from the current Rust/TS sources, else the
release when that was, else builds when this machine has the toolchain, else
runs the release as it is; installation.exe then reads "Built here". The sources are `crates\`
(without tests and the FPGA tree), the UI page and `Cargo.toml`/`Cargo.lock`,
hashed by git's blob ids; a change to the Python plugins never rebuilds anything.

## A lab PC

0. `installation.exe` first puts the program in place: this checkout's release,
   downloaded from GitHub (needs the network once per release).
1. The Python v2 runs with is the one v3 uses (3.13 or newer, found as v2's
   launchers find it: `ZLC_PY_CMD`, PATH, conda's base, the Anaconda, Miniconda,
   Miniforge and python.org folders). A machine with none gets a 64-bit CPython
   3.13 from python.org (3.13.12 is the tested release).
2. `installation.exe` (`install_requirements.bat`): a venv of that Python in `.tools\science-env` with
   numpy, scipy and Pillow and the instrument links (pyserial for the N100;
   PyVISA, PyVISA-py, pyusb and libusb-package for the Rigol and Tek) at
   `science\constraints.txt`'s versions; when that Python already has them all (a
   v2 machine does) the venv uses them and nothing is downloaded. The base Python
   is never changed. Basler pylon (`install_requirements.bat pylon`) adds
   Basler's pypylon, about 270 MB installed, on the PC with a Basler camera. v2
   neither installs nor pins it (a vendor runtime the machine has): any version
   the venv sees (its own or that Python's) counts, and only a PC without one
   downloads it.
   The checkout's own packages are not installed; plugins import them from
   `science\src`.
3. The SLM GPU (`install_slm_gpu.bat`) on the PC that drives the SLM.
4. `experiment.exe`.

The Hamamatsu DCAM camera and the X15213 SLM are driven through their vendors'
DLLs (ctypes) and tkinter, which ships with Python: install them with the
vendors' own installers, as for v2; no pip package supplies them. Vivado (for
the FPGA steps), a USB-TMC instrument's VISA driver and the NVIDIA display
driver are likewise the machine's own installations.

## Once: a checkout from before the engine left git

When the engine left git the repository's history was rewritten into a new
first commit, so a checkout cloned before cannot simply pull. Close every Zou
lab program first. `workspace\`, `.tools\` and `.local\` are not in git and stay
as they are.

With git, in the checkout's folder:

```
git status                 (commit or copy away a change of your own: the reset discards it)
git fetch origin
git reset --hard origin/main
```

then start `installation.exe` and answer Yes to download the program. The old history's
objects can be dropped with `git reflog expire --expire=now --all` and
`git gc --prune=now`.

With GitHub Desktop, which cannot reset a branch to a rewritten one, clone anew
in the same place:

1. Repository > Remove..., without moving the folder to the Recycle Bin.
2. In Explorer, rename the folder (`Zou_lab_control_v3` to `Zou_lab_control_v3_old`).
3. File > Clone repository..., URL `QuantumshWu/Zou_lab_control_v3`, Local path:
   the folder's original path.
4. Move `workspace`, `.tools`, `.local` and `.zlc_python_path` from the old
   folder into the new one (the science environment keeps working: its path is
   the same).
5. Start `installation.exe` and answer Yes to download the program. Once the experiment runs,
   delete the old folder.

## A development machine

installation.exe's Advanced card (`dev_setup.bat`) installs, into `.tools`, Rust 1.99.1 (GNU host, MSVC target),
lld-link, llvm-dlltool, the MSVC SDK/CRT libraries (xwin 0.10.0), FreeType 2.6.1
built as Matplotlib 3.10.8 builds it (zig 0.17.0), Node.js 24.19.0 unless npm is
already on PATH, TypeScript 5.9.3 from `apps\desktop\ui\package-lock.json`, and the
crates of `Cargo.lock`, and sets up the science environment with the plotting
oracle's numba, llvmlite, matplotlib and cycler (the `oracle` extra: only the
Python `zlc_plot` the native engine is compared against imports them, and it is
not in the repository). Every archive is an official release checked against its
publisher's SHA-256; what is present is skipped.

## v2's entries

| v2 entry | v3 | Why |
|---|---|---|
| `experiment.bat`, `figure_viewer.bat`, `pulse_editor.bat` | ported | The windows are v3's; same entries, now `experiment.exe`, `figure_viewer.exe` and `pulse_editor.exe` with no console (the .bat files stay in `bin\scripts` for a development build). |
| `build_and_program.bat` | ported | Same options and receipt rule; v2's FPGA tree is `crates\zlc-pulse\fpga`. The config check, geometry header, BRAM sizing Tcl and capacity estimate are `zlc-pulse-server fpga` (v2's `zlc fpga` arguments; the pulse crate owns that config) instead of v2's Python; the Vivado rule is the pulse server's. |
| `estimate_resources.bat` | ported | The same report, line for line; an error now exits 2 instead of reading as INSUFFICIENT. |
| `install_requirements.bat [slm-gpu]` | ported | Into `.tools\science-env` instead of the machine's global Python, with only what the plugins import; pypylon (v2: the machine's own, unpinned) is the `pylon` option, at any version; the plotting oracle is the development machine's (`dev_setup.bat`). |
| `install_slm_gpu.bat` | ported | Now also warms, once installed. |
| `warm_slm_gpu.bat` | ported | Same measurement, same cache the plugin reads; the plugin re-measures by itself when the GPU, driver, cuBLAS or CUDA source changed. |
| `warm_numba_cache.bat` | not needed | No plugin imports numba: only the Python plotting oracle (`zlc_plot`, development machines) did, and the native renderer compiles nothing at run time. |
| `migrate_pulses.bat` | not needed | A one-shot rewrite of v2's 63-lane pulses for the 69-lane board; current pulses are already in that form, and v3 keeps no back-compatibility. |
| `migrate_units.bat` | not needed | A one-shot respelling of v2's units; files are already in the one spelling, and `us` is still read. |
| `slm_t400_bench.bat` | not needed | A one-off benchmark of v2's solver against v2 commit de73d721 on one recorded run (`bench\slm_t400`, not in v3); its per-GPU choice is what `warm_slm_gpu.bat` measures. |
