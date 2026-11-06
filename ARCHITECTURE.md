# Architecture

The Cargo workspace has 17 crates. Scan, scientific, parameter and capture rules
extend the existing core and workbench modules; they are not separate workspace
products.

## Ownership

| Layer | Responsibility |
|---|---|
| `zlc-core` | Typed storage, schemas, Runtime identity/history/lineage, and domain rules in `scan`, `scientific`, `parameters`, `capture` and `units`. Scan geometry and numerical kernels are independent of service dispatch and device calls. |
| `zlc-workbench` | Application state and authoring; `scan` and `services` adapt requests, schemas, diagnostics and service handles to the typed core. |
| `zlc-archive` | Artifact formats, durable publication primitives and generic inline array codecs in `array_wire`. |
| `zlc-plugin-host` | Processes, framing, deadlines, authoritative run lifecycle, device claims, exact inputs, shared-memory holds and opaque run/process resources. The application installs its service handler. |
| `zlc-desktop` | Composes and injects Workbench services, connects scan execution to Runtime/Pulse, applies service cancellation policy, and owns UI/run-artifact integration. The same executable provides the offline service entry point. |
| Python / TypeScript | Python owns device SDK glue, driver queues and scientific workflow/payload authoring. TypeScript owns controls, interaction and draft edits. |

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
processors use their declared delivery policy. Exact inputs retain events within
their configured bounds and fail explicitly on overflow; Latest inputs coalesce
updates. A finite source can replay its retained committed events.

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

`zlc-core::scan` owns immutable execution geometry: independent and linked
axes, host/board grouping, bounded row pages, effective board coordinates and
the scan's Dataset mappings. It stores axis vectors and mixed-radix strides,
not a Cartesian table. Repeat coordinates and code mappings stay factored.
It has no device calls, scheduler, plotting or Python dependency.

`zlc-workbench::scan::execution` consumes an exact Runtime input through a
run-scoped context supplied by the desktop. The plugin host owns that input
and the run resources without interpreting scan operations. The scan service
places each original immutable DataBlock into the native finite
store with its real parent publication, source ordinal and event metadata.
No source pixels, per-shot schema or placement messages travel through the
Python scan. Runtime still owns storage, overlap checks, subscriptions,
lineage and sealing; the scan service does not create a second data store.

`zlc-workbench::scan::authoring` compiles the authored Range/Values/formula
banks and supplies the editor's preview, validation, binding and ordering, using
core parameter and scan rules. Authored documents retain
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

`zlc-core::parameters` owns generic value contracts and projection rules.
Workbench keeps presentation fields and adapts Python-facing values and errors
in its service boundary. Scientific cross-field validators and device-dependent
conversions remain plugin code. Package `configuration.APPARATUS_SCHEMAS`
declarations own scientific apparatus settings, including the simulation world;
the workbench preserves saved settings when their declaring package is unavailable.

`zlc-core::units` provides the common registry, conversion and quantity
parsing/formatting rules. `zlc-core::scientific` provides typed named-axis geometry,
array operations, statistical reductions and accumulator state used by Derive
and Plot. `zlc-workbench::services` owns the operation names, JSON adaptation and
opaque service handles. `zlc-archive::array_wire` owns the common inline-array
representation; the plugin host supplies live shared-memory array ownership.

The SDK entry point is `zlc_plugin.services.call(...)`, or `ctx.service(...)`
inside a run. The SDK installs its existing plugin-channel transport. The host
checks process/run ownership and invokes an application-installed handler;
`zlc-desktop::plugin_services` chooses Workbench implementations and supplies their
runtime context. The host does not dispatch scientific or scan families itself.

Outside a host, the SDK starts the existing `zlc-desktop.exe --plugin-services`
entry point. This headless mode serves the same Workbench adapters without opening
a window, device or plugin interpreter. There is no additional service executable
or capture library to deploy. Python changes alone need no Rust build; changes
to these native rules require rebuilding the desktop engine.

Capture has two distinct owners. A Python driver's `RecordQueue` owns its FIFO,
condition variable, capacity, producer worker and SDK-buffer lifetime locally.
Queue push/read operations make no service calls. Camera and waveform nodes use
`zlc-core::capture` through the Workbench adapter to validate batch ordinals,
group boundaries and stopped/joined terminal evidence. The returned batch facts
support publication; they do not duplicate the driver's queue state. Capture
validation remains available for Stop cleanup while its run is active.

The core arbitrates Stop against protected terminal work. The SDK performs
cleanup before the core accepts its finish token, so immediate restart cannot
see a still-registered old Python run. Device lanes, concurrent work and wire
admission are bounded and emergency actions have a separate reserve. Unsent
frames and queued requests are checked for expiry before transmission or SDK
dispatch. Absolute receive-time expiry prevents a late-completed frame from
acquiring a fresh relative budget. A started call's
timeout does not release its device or argument-buffer ownership prematurely.

Run artifacts use the desktop's run-contained staging/publication service over
`zlc-archive` durability primitives. Python formats scientific content; native
publication checks containment, syncs, publishes and registers results. Offline
durability uses the same primitives through the headless desktop entry point.
Per-run locks serialize file work; retiring a run marks its slot immediately and
schedules temporary cleanup on workers. A complete visible file whose directory
acknowledgement fails remains named as published-but-not-durable.

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
