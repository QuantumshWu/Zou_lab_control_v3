# Desktop UI

TypeScript and HTML hosted by the native desktop windows: v2's windows
(Devices@, TaskConsole@, one FigureViewer@ per Figure, PulseGUI@Zou lab and each
device's control and log), one native window and one page each
(`index.html?window=devices|task-console|figure-viewer&document=<n>|pulse-editor|
control&device=<id>|log&device=<id>`), all sharing the one core. The UI draws the accepted state and sends commands; every action
is a catalogue command on the bus that MCP clients send too. There is no
charting library, browser fit, projection or JavaScript selector rasterization.
The canvas only copies a finished, opaque backend RGBA image with `putImageData`.

## Build

```powershell
npm --prefix apps/desktop/ui ci --ignore-scripts
npm --prefix apps/desktop/ui run build
```

The native `zlc://app` protocol serves this directory. `index.html` loads
`dist/main.js`; no web development server or remote asset is required. TypeScript
is a build-time dependency only. `dist/` and `node_modules/` are not source.

## Layout of the source

- `src/fluent/` is the control library, the TypeScript image of `zlc_ui.fluent`
  (`fluent.css` styles it, `tokens.css` holds every colour and length). A
  control shows what it is told and hands each operator intent to its owner.
- `src/shell.ts` hosts the page's one view (`views/index.ts` maps each window
  to its view); a view brings another window forward through the host
  (`show_window`). A view implements `View`. `#gallery` (development only)
  shows every control with fake data.
- `src/transport.ts` is the page's connection: `send` (feedback words while
  pending), `request` (a query; resolves with the state's `result`), errors.
  Each window numbers its commands from its own base and leaves another
  window's answers alone; the host sends each page only the sections its view
  reads, and its own plot fronts.
- `src/protocol.ts` and `src/workbench.ts` are the wire types (panel state;
  the workbench sections of docs/WORKBENCH_COMMANDS.md §3).
- `src/views/`: `task_console`, `logic` (rows, Edit pages, panel publishers),
  `device_manager`, `device_control` (Control and Log views), `device_windows`
  (a device's control and log windows), `slm_editor` (an SLM's Control: its
  SLM Editor window over the device's editor_* commands), `figure_viewer`
  (its own document, `state.viewer`), `board` (panel cards; one per plot
  window), `pulse/` (the Pulse Editor), `gallery`.

## Boundary

- Fields keep their mount and an operator's newer unsent draft; an accepted
  projection completes a field only for the request it answers. Scope has one
  menu choice plus a coordinate wheel; no large coordinate dropdown.
- Which page is shown, open Edit/Control tabs, drafts and focus are the
  window's; everything else is state. Admission (can a node start, what Apply
  would do, who may tune a field) is shown as the owner says it.
- New panel state and its frame sequence are promoted together. Errors retain
  the last accepted image and all editable controls. Superseded frames return
  their borrow without becoming visible.
- Shared mappings are posted once and retained across frames. `frame_consumed`
  follows the synchronous pixel copy; `mappings_released` follows WebView2's
  explicit `releaseBuffer` for every slot in that mapping. No PNG, base64 or
  per-frame mapping is used for local presentation.

Pointer coordinates are normalized `[0,1]` from the actual canvas rectangle,
origin at its top left. Positions outside that range remain available during a
captured drag. Axis transforms are promoted with their corresponding frame.
The UI only reverse-order hit-tests their normalized rectangles, then forwards
the complete opaque transform. Press captures that transform for the full drag;
scroll and double-click use the transform under the current pointer. It does not
calculate scales or data coordinates. Messages include the displayed frame
sequence. Hover has no button; dragging keeps the button that began the
gesture. Moves coalesce once per animation frame, but pending motion is flushed
before release, wheel or double-click. Blur and lost capture cancel through the
same native gesture entrypoint. Only the backend decides whether movement
activates a selector. A card is carried by its title band to reorder the board
(`order_panels`).

## Looking at it without the host

`research/ui-mock` serves this directory to headless Edge at the real window's
device scale factor, replaces the native bridge with a mock and drives views
with scripted states (`node research/ui-mock/run.mjs <scenario> <out> [#view]`).
`research/desktop-acceptance/views.mjs` walks the real window.
