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

## Scan execution

`crates/zlc-scan` owns immutable execution geometry: independent and linked
axes, host/board grouping, bounded row pages, effective board coordinates and
the scan's Dataset mappings. It stores axis vectors and mixed-radix strides,
not a Cartesian table. Repeat coordinates and code mappings stay factored.
It has no device calls, scheduler, plotting or Python dependency.

The plugin host's run-scoped scan service consumes an exact Runtime input
directly. It places each original immutable DataBlock into the native finite
store with its real parent publication, source ordinal and event metadata.
No source pixels, per-shot schema or placement messages travel through the
Python scan. Runtime still owns storage, overlap checks, subscriptions,
lineage and sealing; the scan service does not create a second data store.

Rust also compiles the authored Range/Values/formula banks and supplies the
editor's preview, validation, binding and ordering. Authored documents retain
their inactive banks; execution plans contain only selected values. TypeScript
edits those documents and only applies a preview matching its current draft.
Python supplies device-specific facts and conversions, manual prompts, device
writes/readbacks/restoration and Pulse completion policy. It prepares one native
plan, applies settings at segment boundaries and collects bounded batches of
progress. A run-bound plan handle reaches Pulse directly: slot/count/memory
admission, table construction and quantization happen natively. Full applied
tables remain native for the run record and staged LOAD/FIRE verification;
Python receives a compact receipt. The first segment keeps
a bounded quiet drain for sources without an arm/drain acknowledgement. Later
segments omit that silence timer only after the prior expected records and
board DONE have both been observed. This is not a hardware trigger-ID protocol:
native collection rejects source ordinal gaps and generation/schema changes,
but cannot identify a missed hardware trigger whose driver supplied no evidence.

TypeScript owns interaction and display, including focus, mounted controls,
wheel gestures and authored-bank memory. It contains no second scan compiler.

## Shared rules and plugin services

`zlc-authoring` projects generic field values for both the workbench and Python
adapters. Scientific cross-field validators and device-dependent conversions
remain plugin code. Package `configuration.APPARATUS_SCHEMAS` declarations own
scientific apparatus settings, including the simulation world; the workbench
does not hardcode those fields or defaults and preserves saved settings when
their declaring package is unavailable.

`zlc-core::units` is the common registry, conversion, quantity formatting and
parsing implementation. Python exposes immutable unit values and bounded
caches. `zlc-scientific` owns named-axis geometry, typed array operations,
statistical reductions and grouped accumulators, shared by Derive and Plot.
Python retains expression authoring/dependencies and research-specific policy;
bulk arrays use the existing shared-memory channel. Mutable native kernel
handles belong to one run and cannot alias a later run's state.

The small Python `zlc_native` client and codec have no dependency on the plugin
SDK. Hosted calls use the existing Rust host; offline tools call the same rules
through a persistent `zlc-native` helper. There is no second Python numerical or
schema fallback. `zlc-capture` runs inside the plugin process as a small native
library: it owns queue admission, source ordinals, capture/group counts and
terminal evidence; Python keeps SDK threads, record objects, waits and vendor
buffer lifetimes. Capture policy calls do not cross the process channel.

The core arbitrates Stop against protected terminal work. The SDK performs
cleanup before the core accepts its finish token, so immediate restart cannot
see a still-registered old Python run. Device lanes, concurrent work and wire
admission are bounded, emergency actions have a separate reserve, and unsent
or queued expired requests cannot later act. A started call's timeout does not
release its device or argument-buffer ownership prematurely.

Run artifacts use a core-owned staging/publication service over `zlc-archive`
durability primitives. Python formats scientific content; the native owner
reserves names, checks containment, syncs, publishes and registers results.
Offline durability uses those same primitives. A complete visible file whose
directory acknowledgement fails remains named as published-but-not-durable.
The engine bundle includes `zlc-native.exe` and `zlc_capture.dll` alongside the
application; changing native rules requires rebuilding these products.


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
