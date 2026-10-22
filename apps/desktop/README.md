# Zou lab control v3 desktop

The experiment desktop: v2's windows (Devices@, TaskConsole@, FigureViewer@,
PulseGUI@Zou lab and the device, review and SLM Editor windows) on the native
Rust core. Devices and logic nodes run as Python plugin processes; plots are drawn
natively. There is no Qt.

## Start

Double-click `experiment.exe` at the checkout's root, as v2's experiment.bat:
Devices@Zou lab opens, and Init opens the Task Console; it runs the committed
release in `bin\engine`. On a development machine `tools\desktop.ps1 [-Open
figure.npz] [-Build]` (or `bin\scripts\experiment.bat`) does the same with a build
in `target\` current for the Rust/TS sources (made when they change, a few
minutes); see [bin/README.md](../../bin/README.md).

It needs the checkout's science environment in `.tools` (`installation.exe`,
[science/README.md](../../science/README.md)); a development machine also keeps
its toolchain there (`installation.exe`'s Advanced card). No system PATH or
Python installation is modified. `installation.exe` is this executable's
installer mode (`zlc-desktop --installer`, src/installer.rs): its checks and
steps are the checkout's `tools\*.ps1`.

Without `-Open`, use the native Open dialog or Start camera. The camera defaults
to the real SimulationWorld MOT source: 1200×1920 uint8, 100 ms exposure. Start
camera is shorthand for the workbench (docs/WORKBENCH_COMMANDS.md): it puts
the `camera.virtual_mot` device `mot_camera` in the apparatus, applies it, and
starts a `camera_measurement` node, both plugins in their host processes; the
workspace (`--workspace` or `ZLC_WORKSPACE`, else the checkout's untracked
`workspace` folder, as v2) holds apparatus, layouts and runs. Stop
retains its last accepted data for ROI, fit and saving. Save selects a PNG image
filename; the same writer creates the paired Figure NPZ. PDF/SVG export is refused
until the native vector backends land.
Add panel selects a declared live signal and plot kind. Its Source/Window fields
belong to the native binding: Window=1 retains no past data, larger values acquire
the actual Runtime lease, and closing/rebinding releases it. The same Window
drives scientific vocabularies that consume a window; there is no second editable
display Window. Figure files retain their own ordinary scientific controls.
`-DebugPort 9234` explicitly enables
local CDP diagnostics for development; normal launch has no debugging port.
The executable reads its page, `science\src`, fonts (`crates/zlc-plot/fonts`,
licences in its NOTICE) and the FPGA tree from the checkout it sits in (a release
carries its own page in `bin\engine\ui`), so it runs from any clone.

## Ownership and execution

- Rust serializes bounded user commands and owns accepted panel/front identities.
  Only consecutive unsent pointer moves may coalesce; press/release, authoring,
  file operations and their errors keep their ordering.
- One scientific worker runs the native Figure document (`zlc-plot`):
  configuration, projection, fit, session, gestures and rendering. Its panels
  prepare and present concurrently on the process's one bounded worker pool,
  which projection, fit and raster bands share; a live fit is published to the
  Runtime before its follower panel starts. There is no interpreter on this path.
- The document reads Runtime blocks in place: a read is a view of the Runtime's
  immutable planes, nothing is converted or copied per shot.
- A native acquisition worker drains the camera's existing bounded FIFO into the
  single Runtime; the camera plugin lends each frame as a shared-memory block,
  which becomes the Runtime payload without a copy. A dirty bit coalesces only display notifications, never camera
  records. The existing scientific-command dispatcher accepts the newest owned
  publication when idle; user commands retain their order. Stop is an immediate
  native cancellation request, followed by producer join, accepted-tail drain,
  count verification, device close and Runtime seal. Restart has a fresh generation.
- A completed CPU front is one immutable buffer, shared by the accepted state
  and MCP, and copied once into a free persistent WebView shared slot. Three
  slots are reused, and resizing/document replacement waits for old mapping
  release before allocating the new set.
- Pixels and their control/axis descriptions are promoted together in the UI.
  Pointer messages carry the actual displayed immutable transform; scientific
  coordinate conversion remains in the existing Session gesture owner.
- The page only calls `putImageData` for finished pixels. It never recomputes data,
  SEM, fit, scales, selectors or a second chart.
- Native panel bindings replace the single fixed camera-panel path. Input reads
  are acquired together and shared by signal/window; one failed panel retains
  its last picture and controls without canceling successful siblings. The two
  title lines describe the accepted read's exact three-domain schema.

Errors preserve the previous accepted picture and editable controls. A transport
failure stops native writes rather than reusing a slot whose consumption is
uncertain. Closing is idempotent.


## AI clients (MCP)

The window and AI clients share one command catalogue (`crates/zlc-bus`): the
same names, the same JSON-schema-checked parameters and the same queue. Whatever an AI does appears in the window as an ordinary state
update; whatever the operator does appears in the AI's `get_state`.

At start the app serves the Model Context Protocol (Streamable HTTP) on the
loopback interface only, at `http://127.0.0.1:8767/mcp` (`ZLC_V3_MCP_PORT`
overrides; a busy port falls back to a free one). It writes the address and a
bearer token to the checkout's `.local\mcp\mcp.json`, including the exact
registration command, for example:

```powershell
claude mcp add --transport http zlc http://127.0.0.1:8767/mcp --header "Authorization: Bearer <token from mcp.json>"
```

Deleting `mcp.json` revokes every registered client (a new token is made at the
next start). Browsers are refused (Origin check), and requests without the
token are refused. One entry point for every experiment running on the computer
is `bin\ai\zlc-mcp.exe` (`zlc-mcp --client-config claude`): each
instrument leaves a connection record in its checkout's `.local\mcp`, and
zlc-mcp reads those of every checkout a zlc-desktop.exe runs from.

What a client gets: `initialize` instructions (a short capability map), every
catalogue command as a tool (`get_state`, `list_signals`, `get_panel_image`
(PNG of exactly the shown pixels), `get_fit` (parameter table with standard
errors), `wait`, `start_camera`, `stop_camera`, `stop_all`, `add_panel`,
`configure` (semantic roles, display parameters, fit, size, binding,
selectors, viewport, facet focus, thresholds), `focus`, `reset_view`, `fit`,
`scope_step`, `pointer`, `close_panel`, `open`, `save`, and
the workbench's apparatus/logic/board commands), each open device's own commands
(`<instance>__<command>`, announced on the `GET /mcp` event stream), and recipes
as prompts (`camera_roi_profile_fit`, `shot_history_facets`).

Control (v2 `allow_control`): a client may drive the instrument as the operator
does, with no arming step. Started with `--mcp-read-only`, the app offers only the
tools that read (`readOnlyHint`) and refuses the rest; `mcp.json` says
`control_enabled`. `stop_all` stops every run and drives every registered output
(the pulse board, device outputs) safe; it is never refused.

## Other computers of the lab (v2's Remote and Browser sharing)

Both are off until they are turned on in the Device Manager (or by an MCP
client with control, as v2's `lab_api` could).

- **Remote** (a loaded device's switch, `publish_device`): the device is handed
  to the other bench computers through the bench fabric (`crates/zlc-fabric`,
  PLUGIN_PROTOCOL §10): TCP/UDP port 18859, listening only while something is
  published. The other computer's Scan hardware finds it; it opens as a
  `remote.device` row and works there as its own device, frames included.
  As v2's fabric there is no key: a computer is reachable only while one of
  its devices is published, on a trusted lab network. `ZLC_FABRIC_PEERS` names
  computers a broadcast cannot reach, `ZLC_FABRIC_PORT`/`ZLC_FABRIC_BIND` move
  the listener.
- **Browser sharing** (`share_start`, `share_decide`, `share_stop`): this
  window, read-only, at `http://<this computer>/zlc/` (port 80, else 18865;
  `ZLC_SHARE_PORT`, `ZLC_SHARE_BIND`). Every browser waits for Allow; it gets
  the same page, fed by an event stream of the state sections that changed and
  a PNG of each new plot picture; nothing it presses reaches the instrument.
  Only this computer's names are served (Host checked), nothing outside the
  UI, pictures only to an allowed browser.

## Verification boundary

`research/desktop-acceptance/cdp.mjs` inspects the real native window through its
explicit development port and exercises DOM/control/pointer events. It is not
imported by the application. Automated runs park the shown window off every
screen with `ZLC_V3_WINDOW_AT=x,y`. Acceptance images and logs are local artifacts,
not an additional production UI implementation. Physical screen scanout and
four-panel live acquisition are not established by this initial live slice.
The [Virtual MOT gate](../../research/VIRTUAL_MOT_GATE.md) records the actual
Start/ROI/fit/Stop/restart/save/reopen verification and its remaining limits.
