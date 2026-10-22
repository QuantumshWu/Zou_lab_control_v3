# Plugins (Python)

Devices and logic nodes are Python plugins. The desktop runs them all in one
process of the science environment's `python.exe` (`python -m zlc_plugin host`,
see `docs/PLUGIN_PROTOCOL.md`); the desktop itself embeds no interpreter.

- `zlc_plugin`: the plugin SDK and protocol.
- `zlc_atom`: the devices (`devices/`, one folder per device) and logic nodes
  (`nodes/`).
- `zlc_data`, `zlc_durable`: the data and file formats the plugins share.

Changing a plugin needs no build: it takes effect the next time the plugin process
starts (once every device is shut down and every node removed, or when the app starts
again); a new device or node type appears when the app starts. The science environment is set up by `installation.exe` (or
`bin\scripts\install_requirements.bat`): the Python v2 runs with as it is when it
already has what is below, else `.tools\science-env`, a venv of it holding the third-party
packages `pyproject.toml` names (numpy, scipy, Pillow and the `devices` links;
`pylon` for Basler's pypylon, `slm-gpu` for CuPy, `oracle` for a development
machine's plotting oracle) at v2's versions (`constraints.txt` holds v2's pins, and only those). These packages themselves are not installed: the
plugins import them from `science\src`.