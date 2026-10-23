# Zou Lab Control v3

Experiment control for the Zou lab's neutral-atom apparatus: devices, pulse
sequences, live analysis and saved figures. v3 is the successor of
[Zou_lab_control_v2](https://github.com/QuantumshWu/Zou_lab_control_v2), with the same windows,
behaviour and file formats. It is built on a native core: Rust for data, analysis,
plotting and the pulse board, a TypeScript interface in WebView2, and Python plugins
for devices and logic nodes.

> Status: verified end to end on the virtual bench (simulated cameras, sequencer,
> SLM and RF). Real hardware has not been accepted on v3 yet; keep v2 for lab
> runs until it is.

## Start

On a Windows PC with this repository checked out and the Python v2 runs with
(3.13 or newer: Anaconda or python.org's, found as v2's launchers find it):

1. Run `installation.exe` once. The first time it asks to download the program:
   this checkout's release of the engine (about 10 MB, from this repository's
   GitHub releases; `bin\release.json` names it). Then it shows what this PC has
   and installs what is selected: the program (its card says which release is
   installed and how it relates to this checkout -- *Update* after a pull that
   brings a new release -- and, when GitHub answers, whether a newer release is
   published: pull the repository to get it), the plugins' Python environment (the
   Python v2 runs with, as it is when it has everything v2 needs, else a venv of it
   inside this folder, `.tools\science-env`; nothing outside the checkout is
   changed) with the instrument links, Basler pylon on the PC with a Basler camera, and on the PC
   that runs SLM rearrangement the GPU packages, warmed for its GPU. What is
   missing or broken starts checked, and *Fill in missing* installs just that.
   Its FPGA card builds and programs the pulse board with Vivado.
2. Double-click a program:

| Program | Opens |
|---|---|
| `experiment.exe` | Devices@Zou lab; Init opens the Task Console (the experiment) |
| `figure_viewer.exe` | FigureViewer@Zou lab alone; drop a figure archive on it to open it |
| `pulse_editor.exe` | PulseGUI@Zou lab alone (Offline, Virtual or a remote board) |

They run the release in `bin\engine`, so a lab PC needs no compiler. After a pull
that brings a new release, each of them opens `installation.exe` instead, where
the program reads *Update*.
Experiment data lives in `workspace\` inside the checkout (apparatus, layouts,
pulses, runs; not tracked by git); `ZLC_WORKSPACE` points elsewhere.

Every step of `installation.exe` is also a `.bat` in `bin\scripts` for scripting
(the FPGA's `build_and_program.bat` with v2's options, `estimate_resources.bat`,
`warm_slm_gpu.bat` after a GPU, driver or code update, ...); the layout and each
entry are described in [bin/README.md](bin/README.md). A checkout cloned before the
engine left git is re-synced once, with git or GitHub Desktop, as
[that page says](bin/README.md#once-a-checkout-from-before-the-engine-left-git).

## Working with it

- **Plugins.** A device is one folder under `science/src/zlc_atom/devices/`, a logic
  node one folder under `nodes/` (see [the plugin protocol](docs/PLUGIN_PROTOCOL.md)).
  Changing a plugin needs no build: it takes effect when its process restarts
  (re-open the device, restart the node); a new type appears when the app starts.
- **Interface.** After a TypeScript edit, run `bin\scripts\build_ui.bat` and commit
  `apps\desktop\ui\dist` with the sources. HTML/CSS need no build. Root programs
  read this interface from the checkout; pull and restart to update it without
  rebuilding or downloading an exe.
- **AI clients.** Every window action is a command on one bus, also served to MCP
  clients on loopback with a token. `--mcp-read-only` limits them to reading. The
  `bin\ai\zlc-mcp.exe` entry point lists the experiments running on this machine and connects
  to one. See [the MCP tools](docs/MCP_V2_PARITY.md) and
  [the workbench commands](docs/WORKBENCH_COMMANDS.md).
- **Other computers.** With Remote on, a device can be published to another
  computer's experiment, and the window can be shared read-only to browsers on the
  lab network (Devices@Zou lab).

## Development

`installation.exe`'s Advanced card (or `bin\scripts\dev_setup.bat`) installs the
build tools into `.tools\` (Rust toolchain, MSVC libraries, FreeType,
Node/TypeScript). `tools\desktop.ps1` and `bin\scripts\experiment.bat`,
`figure_viewer.bat` and `pulse_editor.bat` compile the interface when Node is
available and build `target\` when Rust sources or build dependencies changed.
The four root programs run the engine release in `bin\engine`, with the interface
in `apps\desktop\ui`. Python and interface edits need no engine release.
`bin\scripts\build_release.bat` builds the release into `bin\engine` and `bin\ai`
and its archive, and commits `bin\release.json` (which names it) with the four
programs; `build_release.bat --publish`, once that commit is pushed, makes the
GitHub release lab PCs download. The engine itself is never in git.

| Folder | Contents |
|---|---|
| `crates/` | the Rust core: runtime, command bus, workbench, plugin host, pulse stack and FPGA project, plot engine, desktop |
| `apps/desktop/ui/` | the TypeScript interface |
| `science/src/` | the Python plugins (`zlc_atom`) and their SDK and formats (`zlc_plugin`, `zlc_data`, `zlc_durable`) |
| (root) | the four programs: `installation.exe`, `experiment.exe`, `figure_viewer.exe`, `pulse_editor.exe` |
| `bin/` | the release this checkout names (`release.json`), its engine (`engine/`) and AI entry point (`ai/`), both downloaded, and the scripts (`scripts/`) |
| `tools/` | build and setup scripts |
| `docs/` | interface specifications |

How the pieces fit together is in [ARCHITECTURE.md](ARCHITECTURE.md).
