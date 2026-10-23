# Architecture

## Processes

- **The desktop** (`crates/zlc-desktop`, one process). It holds the runtime, the
  workbench, the command bus, the plot engine and the pulse stack. Each v2 window
  (Devices@, TaskConsole@, FigureViewer@ per figure, PulseGUI@, device Control and
  Log, point review, SLM Editor) is a WebView2 window of this process, drawn by the
  TypeScript interface in `apps/desktop/ui`.
- **Plugins** (Python, every device and logic-node instance in one process, as
  v2; a type that declares why it cannot share it gets one of its own). The desktop
  starts it as `python -m zlc_plugin host` from the science environment, ends it when
  nothing uses it, and talks to it over a
  framed channel; frames and arrays travel as shared-memory blocks lent to the
  runtime without copying ([plugin protocol](docs/PLUGIN_PROTOCOL.md)).
- **A pulse server** only where a pulse board is attached to another computer
  (`zlc-pulse-server`, v2's board protocol over the network or JTAG-to-AXI through
  Vivado's hardware server).

## Data and display

A source commits complete events to the runtime (`crates/zlc-core`), which owns
identity, values, validity, geometry, lineage and history. Logic nodes and
processors read events in order; nothing a consumer needs is lost when it starts
late.

Panels show data at their own update interval, as in v2: a panel takes the newest
published data when it is due, and its fit, readouts and the values it publishes are
computed on what it shows, once per shown frame. ROI outputs follow the board's
100 ms beat. Panels of the same shot are presented together. The plot engine
(`crates/zlc-plot` with `zlc-figure`, `zlc-agg`, `zlc-text`, `zlc-mathtext`,
`zlc-render`) reproduces v2's Matplotlib figures on the CPU and writes PNG, PDF and
SVG from the same drawing; the window only receives finished pixels.

The finite store freezes conditional Repeat coordinate counts with each snapshot,
using its acquired coverage rather than scientific validity. Plot titles consume
those exact-read counts. Plot configurations keep authored choices separately
from resolved defaults, so Auto follows schema changes and survives layout saves.

Workbench Start builds a read-only candidate before replacing a run. Its frozen
configuration owns values, resources, output names, previews and device claims;
all cancellation paths discard pending candidates and answer acquisition callers.
Async editor and device replies apply only to the request that is still current.


## Commands

Every window action is a schema-described command on one bus (`crates/zlc-bus`),
executed by the workbench (`crates/zlc-workbench`) or the pulse stack
(`crates/zlc-pulse`, `crates/zlc-pulse-editor`). The windows and MCP clients are
equal clients of the same catalogue ([workbench commands](docs/WORKBENCH_COMMANDS.md),
[pulse](docs/PULSE_V3.md), [MCP](docs/MCP_V2_PARITY.md)). A run claims the devices
it uses; claimed fields are protected, and Stop drives outputs to their safe state.
MCP clients connect on loopback with a token; a read-only start offers only reading
commands.

## Devices across computers

`crates/zlc-fabric` lets one experiment use a device another computer publishes
while Remote is on; a lost link puts the device into its error state as a crashed
plugin does. Browser sharing serves the same interface read-only.
