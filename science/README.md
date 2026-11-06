# Plugins (Python)

Devices and logic nodes are Python plugins. The desktop runs them all in one
process of the science environment's `python.exe` (`python -m zlc_plugin host`,
see the [plugin protocol](../docs/PLUGIN_PROTOCOL.md)); the desktop itself embeds no interpreter.

- `zlc_plugin`: the SDK, protocol and `services` client.
- `zlc_atom`: the devices (`devices/`, one folder per device), logic nodes
  (`nodes/`) and scientific configuration declarations.
- `zlc_data`, `zlc_durable`: data and file formats, with adapters to the shared
  native rules and publication primitives.

`zlc_plugin.services.call(...)` uses the existing plugin channel in hosted code;
`RunContext.service(...)` binds a call to its run. Offline scripts reuse the
existing `zlc-desktop.exe --plugin-services` headless entry point.
`ZLC_NATIVE_EXECUTABLE` can select that desktop executable explicitly. The client
also searches the packaged `bin/engine` and this checkout's build outputs.
Importing the service client does not start a plugin server or desktop process.
No separate native-service executable or capture DLL is required.

The workspace retains 17 crates. Typed scan, scientific, parameter, unit and
capture rules live in `zlc-core`; Workbench adapts application requests;
`zlc-archive` supplies portable array/file representations and durability;
`zlc-plugin-host` supplies generic process/resource ownership. The desktop
composes and injects these services. See [architecture](../ARCHITECTURE.md) and
[plugin protocol](../docs/PLUGIN_PROTOCOL.md) for the boundaries.

Driver `RecordQueue` FIFOs, conditions, SDK workers and buffer lifetimes stay in
Python. Queue operations do not invoke native services. Camera and waveform
nodes validate evidence at batch/end boundaries through the existing service
path. Scientific payload encoding and device-specific behavior also remain
Python responsibilities.

Build the desktop engine when native interfaces or rules change. Python-only
plugin edits need no compilation; there is no Python fallback implementation
for a requested native service.

A Python-only plugin change takes effect the next time the plugin process
starts (once every device is shut down and every node removed, or when the app starts
again); a new device or node type appears when the app starts. The science environment is set up by `installation.exe` (or
`bin\scripts\install_requirements.bat`): the Python v2 runs with as it is when it
already has what is below, else `.tools\science-env`, a venv of it holding the third-party
packages `pyproject.toml` names (numpy, scipy, Pillow and the `devices` links;
`pylon` for Basler's pypylon, `slm-gpu` for CuPy, `oracle` for a development
machine's plotting oracle) at v2's versions (`constraints.txt` holds v2's pins, and only those). These packages themselves are not installed: the
plugins import them from `science\src`.
