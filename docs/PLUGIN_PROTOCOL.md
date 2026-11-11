# Plugin protocol (v3)

Python plugins (device drivers, logic nodes, the WGS/SLM glue they contain) run
outside the Rust core. The core has no interpreter and no GIL; a plugin crash or
hang loses plugins, never the application or its data. This document is the
contract between `crates/zlc-plugin-host` (Rust) and `science/src/zlc_plugin`
(the Python SDK). Changes after announcement are small commits titled `API: ...`.

Version 3 requires core-issued terminal tokens and matching SDK/application
service interfaces. Older protocol major versions are rejected during the handshake, before
devices or runs are admitted; they cannot silently fall back to plugin-decided
terminal outcomes. The scientific artifact formats are unchanged.

The workspace has 17 crates. Typed domain rules live in `zlc-core`; Workbench
adapts application operations; Archive supplies formats and generic inline array
codecs. The plugin host owns processes, runs, inputs and array/resource lifetimes,
and accepts an injected service handler. The desktop composes that handler from
Workbench implementations. Domain operation dispatch is not built into the host.

Terms: a **plugin type** is one discovered declaration (`camera.virtual_mot`, the
`camera_measurement` node). A **plugin instance** is one configured use of a type:
one Device Manager row (`mot_camera`) or one Logic row (`mot_monitor`). A **run**
is one Start..terminal of a node instance.

## 1. Process model

**One plugin process, as v2.** The core runs every opened device instance and
every node instance in one `python -m zlc_plugin host` process, each instance on
its own lane (section 4), so devices still open in parallel and a node's run, a
device's call and a query never wait on one another beyond the GIL. The process
starts with the first instance (or the catalogue, or a hardware scan) and ends
when nothing uses it any more: no process waits warm for the next open, none is
started ahead. A node instance keeps its warm imports, Numba caches and GPU context
across runs, so Start does not pay interpreter startup. State instances share --
the simulation `SimulationWorld` every virtual device stands in, a vendor runtime
that is a process singleton (libdaq2, DCAM) -- is simply in that one process.
Plugins never render plots and never own Runtime history; the GIL that once made
one process slow belonged to the plots, which are Rust now.

**Isolation by declaration only.** A type that truly cannot share the process
declares why (`DeviceType(isolated="...")`, `NodeType(isolated="...")`, descriptor
`isolated`): a vendor SDK that crashes or hangs its process, a DLL that conflicts
with another's, a GPU context that measurably slows the others. Its instances
then share one process of their own; the reason stands beside the declaration.
No type declares it today.

Every child is placed in one Windows Job object with KILL_ON_JOB_CLOSE, so no
plugin survives the core (no orphan holding a camera). The core spawns children
with `PYTHONHOME` removed, `PYTHONPATH` = the configured science/plugin roots,
the interpreter's base installation (a venv's `pyvenv.cfg` `home`; for conda its
`Library\bin` and the other folders activation adds) first on `PATH`, and v2's
thread policy (`OPENBLAS_NUM_THREADS` min(4, CPUs), `NUMBA_THREADING_LAYER=omp`,
`OMP_WAIT_POLICY=PASSIVE`, each unless set). The interpreter is the one
`zlc_plugin_host::plugin_python` resolves (`ZLC_PLUGIN_PYTHON` when set, else the
answer of `tools\install_science.ps1 -PluginPython`), for the plugin process, the
catalogue and hardware scans alike: the executable needs no launcher.

## 2. Discovery and descriptors

Plugin folders follow v2's layout rule, one folder = one device:
`zlc_atom/devices/<family>/<device>/device_types.py` exporting `DEVICE_TYPES`
(a tuple of `zlc_plugin.DeviceType`) and `zlc_atom/nodes/<node>/logic_node.py`
exporting `LOGIC_NODE` (a `zlc_plugin.NodeType`). Family folders hold only shared
contracts. The descriptor is written once, in Python, beside the code it
describes. The core learns it by asking the plugin process `describe {packages}`,
which imports each leaf module and answers one JSON document `{"protocol": [3, 0],
"types": [...], "errors": [...]}` (kept in `.local\plugins\catalog.cache` until a
source or the environment changes).
A module that fails to import is reported in `errors` with its traceback; the
other types remain available. The core never imports plugin code.

A type descriptor (JSON, the only form the core sees):

```json
{
  "type_id": "camera.virtual_mot",
  "kind": "device",                      // "device" | "node"
  "family": "camera",                    // device family / node category
  "label": "Virtual MOT camera",
  "description": "What it is for, for operators and AI clients.",
  "module": "zlc_atom.devices.simulation.camera.device_types",
  "simulation": true,                    // device only: stands in the simulated world (section 3)
  "isolated": "",                        // why it cannot share the plugin process ("" = it shares; section 1)
  "capabilities": ["camera.adapter"],    // device: SPI contracts it implements
  "parameters": [Field...],              // instance configuration (authoring schema)
  "physical_outputs": [{"name": "mot_coils"}],   // device only: what Stop all drives safe
  "commands": [Command...],              // operator/AI commands (section 8)
  "control_window": "slm-editor",        // device only: the window its Control opens ("" = the field form)
  "announces": "",                       // device only: the client type its own server is opened as (section 10)
  "identity": "visa:{resource}",         // device only: the instrument a row names, over its parameters ("" = every row its own)
  "resources": ["serial:{port}"],        // device only: the physical resources a row binds, over its parameters (below)
  "node": {                              // node only
    "node_kind": "measurement",          // measurement | task | processor
    "inputs": [{"name": "source", "contract": "zlc.camera.frames", "delivery": "exact", "select_bundle": false}],
    "artifact_inputs": [{"name": "calibrations", "label": "Calibrations", "contract_id": "...",
                         "file_filter": "Calibration artifacts (*.json)", "required": true, "frame_claims": true}],
    "outputs": [{"name": "frames", "contract": "zlc.camera.frames", "index_by_source": false}],
    "devices": [{"capability": "camera.adapter", "argument": "camera",
                 "protected_fields": ["exposure", "roi_x"], "enabled_when": null}],
    "previews": [...], "reports_ready": true, "acquisition_input": "", "acquisition_completion": "ready",
    "artifact_outputs": [...], "selection_mappings": [...], "workspace_resources": [...],
    "replaces": {"post_selected_occupancy": {"post_selection": true}},  // types it took over: a saved row naming one loads as this type, these values over its own
    "answers": {"editor_facts": {...}, "resolve_defaults": {...}}  // its draft queries' answers for its default draft
  },
  "queries": ["declare_outputs", "resolve_field_availability"]
}
```

`Field` is v2's `AuthoringField` as data: `{name, type, label, default, required,
minimum, maximum, step, unit, display_unit, choices: [{value, label}], segmented, blank_label,
description, enabled_when: [field, [values]] | null, derived, group, columns, file_filter}` with
`type` in `int | float | bool | str | text | pair | choice | rows | resource`. Choices
are `(value, label)` pairs stated by the declaring plugin; no layer transforms a value
into a label. Units are the field's own unit string (zlc_data unit grammar).
`display_unit` (optional, null for none) names the unit family a view shows the
number in by default when its owner prefers one to `unit`'s own -- a function
generator's amplitude, its `unit` dBm, shown in Vpp; the view picks the prefix in
that family that reads the number between 1 and 1000. It is display only: commands,
bounds and records stay in `unit`, and the view reads the number in that family
through the owner's own conversion (`fields {units}`). `group`
names the form section the field stands in ("Imaging"); the form titles each change
of it. A nonempty `file_filter` (for example `Pulse Config (*.json)`) presents a
string field as a file path with Browse; its stored value and validation stay text.

Draft-dependent declarations stay functions of the plugin, answered by the node
instance's own process through `query` (section 5): `declare_outputs`,
`declare_previews`, `field_availability`, `resolve_defaults`, `admit_source`,
`admit_resources`, `estimate_run`, `selection_patch`, `describe_output_frames`.
`queries` lists which ones a type implements. They are pure functions of the
draft values, resolved devices (as proxies) and resources; they never start work.
Every node also answers `admit(values, resources, artifacts)` (the SDK, v2
`finalize_logic_draft`): the draft projected by its schema with the cross-field
validator, the type's `admit_resources`, and each artifact file decoded; its answers,
like any draft query that fails, are the draft's issues and Start waits for it.
`editor_facts(values, resources, devices, source)` answers the model of a node's own
editor, `{editor: kind, manages: [field, ...], ...}`: the UI's editor of that kind
draws it in place of the fields it manages and writes them back with `edit_node`; it
holds no rule of the node's (the model says what is offered, derived, kept or wrong).
`source` is the node's source frames `[axis_name | null, [labels...]]` or null.
Kinds: `scan_plan` (seamless scan, `scan/editor.py`: `ports [{port, label, unit, lo,
hi, current, integer, display_unit, group, leaf, units, families, limits, currents}]` --
`current` where the knob stands in its unit (a slot's value in the pulse, this run's
API value, the device's reading; null when nothing says), the editor's first sweep
around it; `limits`/`currents` its limits and reading in each spelling no decimal
shift reaches, as the device converts them --, `withheld {port: reason}`, `families`,
`host_advanced` prefixes, `summary`/`problem`, `api {fields, overrides, parked, note}`
or null) and `pair_matrix` (frame survival, `frame_survival/editor.py`: `grid {rows,
columns, cells}`, `selected`, `states`, `tips`, `identity`, `override_rows/_cells`,
`list_rows/_cells`, `keeps_list`, `summary`, `note`, `parked`, `error`).
`resolve_device_claims(values)` answers `{argument: {instance, fields}}`: devices
the draft itself names (a scan's `device:<key>:<field>` axes), beside the declared
requirements; the core claims those fields for the run (Device Control cannot
move them meanwhile) and passes each device as `argument` (a tunable proxy). Such a
claim shares the device (v2's runtime claim, `RunRequest.shared`): it conflicts with
no other run's claim, while two exclusive claims never hold one device.
`resolve_field_units(values, devices)` answers `{field: unit}`, the unit a field is typed
in when the draft decides it (a trigger level, its channel's). `declare_previews(values,
devices, node)` is told the node's id, the producer its axes are named after, so a preview's
`semantic` can hold one coordinate of one (`{"fate:cell_data:<axis>": ["scope-value", position]}`).
`resolve_field_choices(values, resources, devices)` answers `{field: [[value, label], ...]}`
for fields whose options come from the draft (an SLM rearrangement's imaging
periods are the chosen pulse's periods, a waveform trigger's channels the bound
source's); the form shows that field as a choice,
and a value it holds that is not among them as unavailable, never replaced.

A node's page draws the newest answers it has, never an empty draft: a node just
added draws `node.answers` (the SDK answers each draft query for the type's default
draft, no devices, files or source, while it describes the type; one that cannot is
left out), and a draft just edited draws the answers of the one before it until its
own land. Only a Start waits for the draft's own answers (`Checking the draft…`).

**Physical resources.** A device type's `resources` are templates over a row's
parameters, as `identity` is (`serial:{port}`, `visa:{resource}`; SDK
`DeviceType(resources=...)`). The core names each `<kind>:<value>`: the kind
lower-case; a `serial` value upper-case without `\\.\` (`serial:COM3`), any other
value trimmed and lower-case. A template whose value is blank binds nothing yet. The
bound set is every apparatus row's resources and every live device's that holds them
(opening, open, closing, or a close that failed), each with the row that binds it. A
row binding one an earlier row or another live device binds is refused before it
opens (`RuntimeError: serial:COM3 is already bound (to n100)`). Every scan is handed
the bound set and every resolve the bound set less its row's own (`excluded`, section
3); neither opens a resource in it. A device type's `queries` say what it answers
without an instance: `discover` (its hardware scan; an SDK `discover` that takes
`excluded` is handed it, one that takes nothing scans as before) and `resolve` (SDK
`resolve(parameters, excluded)`): a row of such a type with a resource still blank is
resolved before it opens, one row at a time in the operator's order while the other
rows open, and what it finds is pinned into the apparatus (WORKBENCH_COMMANDS §4.2).

Packages may export `configuration.APPARATUS_SCHEMAS`: a mapping from apparatus
namespace to an authoring schema. Discovery serializes these under
`configuration_schemas`; conflicting declarations are catalogue errors. The
workbench projects settings using these declarations, and preserves saved values
when their owner is unavailable. The simulation schema is declared by `zlc_atom`,
including its defaults and bounds, rather than copied into the native workbench.

Capabilities name a device SPI by token (`camera.adapter`, `motion.axis`, `rf.source`,
`slm.phase`, `waveform.source`; the pulse board is native and no plugin claims it:
a node plays a pulse through the core's `pulse_*` commands). `zlc_plugin.capabilities`
maps each token to its Protocol class by *dotted path*, so importing the table
does not import every family (v2's `CAPABILITY_TYPES` imported all of them).

## 3. Lifecycle

```
spawned -> ready -> open -> (idle <-> running) -> closing -> closed
                 \-> failed (any state: process exit, unanswered deadline, protocol error)
```

| Step | Core sends | Plugin does | Reply |
|---|---|---|---|
| handshake (process) | `hello {protocol, session}` | creates its block arena | `{protocol, pid, python, sdk}` |
| open | `open {instance, type_id, module, parameters, simulation}` | imports the type's module; device: `factory(instance, projected parameters)` builds and opens the adapter; node: keeps the draft | `{state, tunable, reads, narrates, snapshot}` |
| configure | `configure {instance, parameters}` | device: `adapter.reconfigure(values)` or refuses | `{state}` |
| prepare (node) | `query {instance, name: "prepare_run", values, devices, resources, source_signal}` | projects and builds a candidate with read-only device proxies; the active run is untouched | `{prepared: token}` |
| discard (node) | `run.discard {instance, prepared}` | releases an unstarted candidate; idempotent | `{}` |
| start (node) | `run.start {instance, run, prepared?, values, devices, resources, source_signal, run_directory, origin}` | consumes the prepared candidate and binds its proxies to the acquired run; without a token, builds on the run thread for direct SDK clients | `{run}` immediately; terminal via tokenized `run.begin_terminal` / `run.finish` requests |
| stop (node) | event `run.stop {instance, run}` | sets the run's cancel flag | none |
| health | `ping` | answered by the reader thread (proves the interpreter and GIL are alive) | `{}` |
| catalogue (process) | `describe {packages}` | imports every leaf module on a thread of its own | the descriptor document (section 2) |
| hardware scan (process) | `discover {type_id, module, excluded}` (no instance; in the process its instances open in, so a vendor runtime that is a process singleton -- DCAM -- is the one its devices use; `excluded`: the bound resources, section 2) | the type's `discover(excluded=...)` (or `discover()`) on a thread of its own | `{found: [{instance_id, role, parameters}], notes}`; the core stops waiting at its deadline (20 s, v2) and drops a late answer |
| resolve (process) | `resolve {type_id, module, parameters, excluded}` (no instance; Init, before the row opens; `excluded`: the bound resources but the row's own) | the type's `resolve(projected parameters, excluded)` on a thread of its own | `{parameters: {name: value}, note}`: the values found (none: open as authored), pinned into the row; an error fails the row and says what it tried and passed over. Deadline 20 s |
| close | `close {instance}` | device: its closer (stop capture, release SDK); node: stop run | `{state}`; a process left with nothing to serve ends |
| abandon | event `abandon {instance}` | stop accepting work for the instance; keep its name reserved until the overrunning call returns and cleanup succeeds | none |
| exit (process) | `exit` | closes what is left | `{}`, then exits 0 |

`simulation` is the installation's simulation mapping, given to a device type
that declares `simulation=True` (v2 world_config; the workbench opens those again
when it changes); `zlc_plugin.simulation_parameters()` returns the ones the device
being opened was given, and `zlc_plugin.share(name, build)` gives the process's
instances one shared object (the world), built by the first and dropped by the
last to close. Device open is per instance and
parallel; the core admits opened instances in the operator's order (v2's
`_build_together`). A failed open reports its error and the instance is gone;
nothing half-open is admitted. The core ends a process with `exit` (or by
closing stdin); it waits at most 5 s, then terminates it.

A successful Close removes the Python instance from the dispatch registry **before**
its acknowledgement is sent. Immediate re-Init therefore reaches a fresh lane.
An Open error also completes cleanup and retirement before it is reported. If cleanup
or Close fails, its adapter and closer remain owned and retryable; the Rust host and
Device Manager retain the failed-close state. Apply retries Close before opening a
replacement. A failed Close must never be shown as a successfully released device.

Device metadata in draft queries and `run.start` includes `instance` (stable ID),
`role` (the accepted operator name) and `units` (field -> the unit the operator
keeps for it, picked in Device Control and saved with the apparatus). Proxies
expose `role` for display and `units` for the unit a scan port is offered in and
a device record is written in; commands, scan port keys and claims continue to
use `instance`, and a recorded device stays keyed by it with `role` beside it.
Draft and run proxies receive the same declared `reads`. Every tunable proxy
also offers `tune_in_unit`; the device-side SDK applies its native conversion
when available, with the same rules as Device Control. Draft queries cannot tune.
At `run.begin` the host stamps `node` with the instance ID and `node_name` with
the current operator name, overriding plugin-supplied identity. Later renames
do not change this historical record. It also writes the core's configured
acquisition `apparatus` (section 5); a plugin's own key of that name is dropped.

The core owns the Stop/terminal race. `run.begin_terminal {accept_stop, partial}`
atomically enters protected terminal work and returns an opaque token. A Stop
accepted first refuses this transition unless `accept_stop`; a later Stop is
ignored. At executor exit the SDK calls `run.begin_terminal` with
`finalizing: true`, `intent: success|error|invalid` and `accept_stop: true`.
The core returns the frozen `outcome` and whether to invoke the partial-exit writer.
After cleanup, `run.finish {token, result, error}` validates that token, seals the
Runtime generation and emits the application `RunFinished` event. The plugin
retires its local runner and unread deliveries **before** this last request, so
an immediate restart after completion reaches a reusable lane. An old
plugin-authored `run.finished` event is a protocol failure, not an authoritative
successful outcome. Stop is not deletion: a stopped run's
committed data stays readable. A restart is a new `run` id and a new generation;
commits carrying an old run id are refused (`stale_run`).

## 4. Control channel

Transport: the child's stdin (core -> plugin) and stdout (plugin -> core). The
SDK takes file descriptor 1 for the channel at startup and points `sys.stdout`
at stderr, so a stray `print` cannot corrupt framing. stderr is the plugin's log:
the core reads it line by line, tags it with the instance and keeps a bounded
ring for the UI and `get_state`.

Framing: `u32` little-endian byte length, then that many bytes of UTF-8 JSON
(one object). Maximum frame 16 MiB; bulk data never travels in frames (section 6).

```json
{"t": "req", "id": 7, "op": "spi", "p": {...}, "deadline_ms": 2000, "blocks": [...]}
{"t": "res", "id": 7, "ok": {...}, "blocks": [...]}
{"t": "res", "id": 7, "error": {"kind": "refused", "type": "TuneRefused", "message": "...", "traceback": "..."}}
{"t": "evt", "op": "run.progress", "p": {...}, "blocks": [...]}
```

Both sides issue requests; each side numbers its own (independent id spaces).
`blocks` lists every shared-memory block the message lends (section 6), with
multiplicity. Unknown fields are ignored; an unknown `op` is answered with
`error.kind = "unknown_op"`. The protocol version is `[major, minor]`; a major
mismatch at `hello` fails the instance, a newer minor only adds fields/ops.

Execution inside a plugin process: every request for one instance runs on that
instance's **lane**, one thread, in arrival order (vendor SDK thread affinity,
no reentrancy into a driver). `ping` and the events (`release`, `run.stop`) are
handled by the reader thread and never wait for the lane. A node's `execute` runs on its own
run thread, so `query`, `run.stop` and device-call replies stay responsive while
a run is in progress. Long operations take bounded timeouts and check
`ctx.cancel_requested()` (v2's rule: a blocking read is interrupted between reads).
A device type may name adapter calls its adapter answers concurrently
(`DeviceType(concurrent=(...))`, descriptor `concurrent`): an `spi` request for a
named method, a `command` of a named command, or `make_safe` when it is named, runs
on its own thread, beside the lane. An SLM's `play_phase_sequence`
holds the lane for the whole movie while the caller streams `submit_phase_frame`
and may `cancel_phase_sequence`; a callable (v2's `stop_requested`) cannot cross
processes, so the caller cancels instead. A motion axis's `tune` of its position
holds the lane until the stage stops; its `stop` command and Stop all's `make_safe`
are concurrent, so they reach the controller during the move, which then reports
where it stopped (`zlc_atom/devices/motion/contract.py`).

## 5. Operations

Core -> plugin requests of a type, with no instance (section 3): `describe {packages}`,
`discover {type_id, module, excluded}` and `resolve {type_id, module, parameters,
excluded}`, for example `{"op": "resolve", "p": {"type_id": "waveform.wheeltec_n100",
"module": "...", "parameters": {"port": "", "baud": 921600, "timeout_seconds": 2.0},
"excluded": ["serial:COM3"]}}` answered `{"parameters": {"port": "COM5"}, "note": "N100
streaming on COM5 (passed over COM3: bound)"}`.

Core -> plugin requests (each names its `instance`): `hello`, `open`, `configure`,
`close`, `exit`, `ping`, `spi {method, args, kwargs}` (call the open device's
adapter; `get:<name>` reads a property), `command {name, params}` (section 8),
`fields {units?}` / `tune {field, value, unit?}` / `values` (the tunable quartet), `snapshot {units?}`
(the device's configured state for an acquisition, below), `make_safe
{output}`, `publish {published}` (Remote of a type that `announces`: section 10), `query {name, values?, devices?, resources?, source?, output_name?,
installation?}` (the facts by name; the SDK calls the v2 callable with them in v2's
positional order, `zlc_plugin.declare.QUERY_PARAMETERS`; `devices` become proxies), `run.start`. Events: `release
{blocks}`, `run.stop {run}`. A request with `"plain": true` gets its result as
plain JSON (dataclasses as objects, tuples as lists, dtypes and enums as their
values, arrays still as references); the core always asks for that, while a
plugin-to-plugin call keeps the tagged types of section 6.

**Device state at acquisition.** Workbench Start prepares and launches a node without
reading the whole bench. A measurement/task run opts into the application's capture
through `RunRequest.record_devices`; the SDK's `context.record_devices()` sends
`run.snapshot {wait: false}` at the configured acquisition boundary. Cameras and
waveform sources call it after arm and accepted working-point readback, before their
read/publish loop. A scan sets its first device point and loads its pulse before
starting its acquisition source, then reuses that capture before FIRE. Restoration
backups stay inside the device-knob owner; saved scan state is the applied first point.
Each point additionally keeps actual readbacks, units and provenance with its data.

A task and a child acquired with `completion: ready` share one capture context.
The first explicit caller records it; subsequent callers receive the same record.
A `terminal` acquisition has its own context. Before a run's first publication the
SDK requests `run.snapshot {wait: true}` if it has no record yet. This joins an
in-flight capture, or returns no additional apparatus when there was no explicit capture (a task
that only consumes another acquisition keeps its source records). It never reads
later hardware state and calls it the earlier acquisition state.
Stop, failure and context completion release waiting requests. Processor runs
inherit their sources' records and do not collect the bench themselves.

The workbench asks every open device that advertises `snapshot: true` concurrently,
including static devices that a node does not directly call. The SDK uses the
adapter's own `snapshot()` when provided, otherwise the tunable quartet returns
`{values, units, provenance}`; a field the apparatus keeps a unit for (`units`) is
recorded in that unit only, converted by the device's own conversion when it has
one (an RF level into its load). The native sequencer includes its loaded program
and pulse. Missing support, busy devices, read errors and timeouts remain explicit
`unavailable` values. The 2 s budget includes the local pipe writer; at most one
snapshot per device is in flight. Expired unsent requests are cancelled; a partially
written local frame completes and its late answer is dropped. Snapshot timeouts
never mark an otherwise responsive device failed or take its owner. Explicit host
shutdown retains its own close/exit deadline.

The core installs the observation before acknowledging `run.snapshot`; `run.begin`
writes it as `apparatus`. Plugin-supplied `apparatus` cannot replace this record.
The envelope contains `phase: "acquisition"`, `context`, `taken_at`, `completed_at`,
`devices`, `sequencer`, and optional `device_transports`. The timestamps bound the
read interval, not an assertion that every device was sampled simultaneously.
Each device includes its identity and complete original payload. Identity conflicts
are retained in `snapshot_fields`; remote endpoints live in `device_transports`
and never overwrite device-owned provenance. Remote snapshots forward the owner's
same complete payload. An old publisher without the capability is unavailable.

A figure archive keeps its lineage out of `info.npy`, in its own zip member
`lineage.json`: UTF-8 JSON `{"schema": "zlc.figure.lineage/1", "lineage": ...,
"apparatus_table"?: ...}`. Drawing a figure and working with its panels never read it;
it is read for the Figure Viewer's record tabs, for saving the figure again and by
`read_file`. A file written before this member existed keeps `sections.lineage` inside
`info.npy` and reads the same.

Figure archives store identical shared apparatus records once when they occur in
multiple records of one document (`info.npy` or `lineage.json`). The optional root
`apparatus_table` has schema `zlc.figure.apparatus-table/1` and a `records` array; only
designated record.apparatus positions use exact `{"$apparatus": index}` references.
Rust and Python readers expand these references and remove the table before exposing
the document. The public metadata structure is unchanged; records with different
context, timestamps or payloads are never conflated. Files without a table retain their
existing format.
Finite dataset provenance includes all acquired ranges, not only the last event.
Its `acquisition` projection holds `runs`, every event once and the placed `ranges`;
event parents retain the source's settings epochs. Publication prefixes hold immutable
backward metadata chains, so later commits cannot change a saved prefix or make an old
read retain future facts. Viewer groups these records under their actual Logic runs and
device states rather than adding a Flow node per point.

`acquisition` is stored in columns (`zlc_core::acquisition`, lossless; format
`zlc.acquisition/2`; without a `format` it is the older list of event objects, read the
same): `signals` and `records` tables and `events`/`ranges` rows under rules. A rule is
`{"rows": [...], "repeat": [n, ...], "step": [level, ...]}`: its rows repeated over the
nested counts, the last fastest, each row at indices `i` being `rows[r] + sum(i[d] *
step[d][r])`; a chunk without `repeat` is literal rows, so a dropped or retried shot
just ends one rule. An event row is `[run, signal, shape, position, revision, ordinal,
parent...]`, a range row `[extent0, extent1, event, origin0, origin1]`. `records` is a list
of shapes `{"template": ..., "columns"?: [{"path": [...], "values": [...]}]}`: an event's
record is the template with each column's `position`-th value at its path. A `window`
(format `zlc.window/2`) is `{"width", "members": [event index, ...], "acquisition"}`, its
runs and each member's ancestors held once. The native `read_archive` and `read_file` give
both in their older structure; Python's `read_archive` gives them as stored.

Plugin -> core events: `release {blocks, leases}`, `run.ready`, `run.progress`,
`device.log {instance, line}` (one line of
an open device's own log, v2's: each INFO record of the loggers under `zlc_pulse.remote`,
`zlc_atom.devices` and the type's `log_channels`, as `[HH:MM:SS] message`, for the devices
it is about -- a logger under the type's `log_channels`, or a line ending in
`device=<identity>`, the adapter's `identity`). A device's `open` answer says
`narrates` (it has `log_channels` or an `identity`). And `device.notice {instance, name, value}`:
an open device tells the core a fact of its own outside any request, its arrays lent
(`HostEvent::Notice`; an SLM Editor's draft, sent whenever it changes -- a solve ends
long after the command that started it). SDK: `zlc_plugin.notifier()` inside a device
command gives the instance's `Notifier`, callable from any thread while that device is open.
And `journal {lines: [{us, src, ev, facts}]}`: what a plugin says for the application's flight
journal (`zlc_core::journal`, the last hour of commands, runs, board commands, publications and
drawn reads, exported by `export_diagnostics`) -- a camera frame arriving, a capture armed or
ended, a scan segment fired. `us` is microseconds since the Unix epoch when this process observed
the fact; the core files the line among its own in arrival order and adds the process id. Plain
JSON, identities and times only, never data. SDK: `zlc_plugin.journal.note(source, event, **facts)`,
which is silently not kept outside a plugin process and never fails its caller.

Plugin -> core (only during a run, bound to its run id):

| op | Meaning |
|---|---|
| `run.begin {outputs: [{name, schema, storage, index_by_source}], record}` | Declare the run's outputs and run record. `Runtime::begin` under owner `<instance>` with the record (plus the host's `node`, `node_name` and acquisition `apparatus`, section 3) as the run's metadata (who started it goes in the seal); a run consuming a source (a processor, or a run following its source signal) begins with `Runtime::begin_consuming`, a same-shot route of that signal, so a result is shown with the source event its `parents` name (v2 `source_name`); signals are the names the run's start gave (`RunRequest.outputs`, e.g. `@logic/<node>/<output>`), else `<instance>/<output>`. Storage `latest | finite`. |
| `run.commit {inputs: [{output, ordinal, origin, data, schema?}], metadata, parents}` | One atomic bundle (`Runtime::commit`): siblings commit together or not at all. `origin` is the finite `(repeat, point)` placement; a finite event carries its chunk `schema` when it differs from the last one sent. `parents` are read leases the node consumed. Metadata is `{records: {output: event_record}, shot_time_seconds: {output: t}}`; `t` stamps that output's shot (every shot of a signal or none, later shots later), and a history window over it carries the shot time as an alternative coordinate of its source index. Returns each output's publication key. |
| `run.commit_events {count, ordinals, inputs: [{output, data, schema?}], shot_times?, origins?, records?}` | `count` consecutive event bundles in one round trip (`Runtime::commit_events`), each the event `run.commit` would make of it: `ordinals` (u64), `shot_times` (f64) and `origins` (`(count, 2)` u64) have one row per event, and each output's `data` planes hold the events one after another, first axis the event, each event borrowing its part. `records` is one JSON record per event, kept under every output of the bundle. Every Exact and Follow subscriber takes each event; a Latest subscriber and the change watch only the last. Answers nothing. |
| `run.operator_input {kind, title, message, payload}` | Blocks the run thread until the operator answers or the run stops. |
| `run.begin_terminal {accept_stop, partial, finalizing?, intent?}` | Core-owned Stop/terminal arbitration; returns `{token, stop_accepted, partial, outcome, run_exit_writer}`. `outcome` is fixed when finalizing. |
| `run.finish {token, result, error}` | Completes an already-finalizing run after plugin cleanup, using the core's frozen outcome. |
| `run.artifacts {action, ...}` | Run-contained native publication and registration; see section 5.3. |
| `run.artifact {name, path, role, contract_id}` | Legacy registration spelling, routed through the same confined artifact store. |
| `run.figure {path, data, overlay, recipe, source, lineage, image}` | Save one figure (`RunContext.save_figure`, `zlc_plugin.figures`): the core writes the `zlc.figure` archive `<path stem>.npz` and, when `image`, renders `path` (`.png`/`.pdf`/`.svg`; no suffix = `.png`). `data` is one complete dataset `{schema, data: {values, validity, sigma}, block_id, revision, generation}` (`validity` true/false or its storage mask). `recipe` is `{spec, parameters, size}` plus any of `viewport, classifier_thresholds, facet_focus, fit, selectors, interaction, presentation`: `spec` is the archive recipe's spec tree; the core completes `parameters` with the kind's defaults, validates `size` and normalizes `classifier_thresholds` as a plot session keeps them (v2 `encode_plot_recipe`). `overlay` is null or `{revision, coordinates (N,2), point_ids, labels, static_statuses, status (a dataset), paths_xy (N,K,2)}`. `source`/`lineage` go into the archive's sections as v2's `save_figure_artifact` put them. Answers `{archive, image}` (`image` null when not drawn). Registering them as artifacts stays the plugin's. |
| `run.animation {path, images, frame_ms}` | A looping GIF of the run's images (its figure previews, inside the run directory), `frame_ms` each, each frame on its own palette (v2's Pillow save of the recording previews); answers `{path}`. `RunContext.save_animation`. |
| `run.fit {model, x, y, sigma, sigma_dof, warm}` | One independent fit per row of `y` `(cells, n)` over `x` `(n)` by the core's fit engine with its default options (`RunContext.fit`, `zlc_plugin.fit`); `sigma` as `y` or null, `sigma_dof` (each sigma's degrees of freedom, as a panel's fit weighs them) as `y` or null, `warm` `(cells, parameters)` or null, a row with a non-finite value being none. `model` is an id of the read-only catalogue command `fit_models` (`{models: [{id, label, parameters: [{name, unit_relation}]}]}`, series models only; `zlc_plugin.fit_models`, also from a draft query). Answers `{parameters, values, errors, success, reduced_chi_square, messages}`, a row per cell, null for NaN. A cell that cannot be fitted (too few finite points, a solver failure) is `success: false` with its reason, not a failed call; non-finite points are dropped by the engine. Solved on a standing worker, off the window's turn and the channel's thread; the run's Stop ends the batch with `cancelled`. |
| `run.acquire {node, completion}` | Start the acquisition node this run drives (v2's `restart_logic`); answered at `ready` or `terminal`. |
| `run.snapshot {wait}` | Record configured acquisition state (`wait: false`) or join an existing observation before publication (`wait: true`). A ready parent/child context shares one observation; without an explicit capture, publication adds no empty or retrospective record. |
| `core {command, params}` | Run one catalogue command (e.g. a Pulse operation) under the run's origin; the run's claims apply (a `pulse_*` command needs the sequencer claimed). From a draft query (`zlc_plugin.core`, `run: null`) only a `read` command is answered: an estimate's `pulse_estimate`. |
| `device {argument, method, args, kwargs}` | Call a claimed device's SPI (section 7 checks apply). `tunable_fields` answers the same rows as the `fields` op, as data. |
| `runtime.read {signal, mode, width}` | Read a Runtime signal: `event | dataset | window` (the event of a dataset publication, e.g. a Fit over a finite run, is its dataset). Returns `{lease, key, schema, window, segments: [{origin, extent, data}], metadata, run_record, placement, lineage}`, `lineage` its ancestry nearest first, each publication once: `[{key, run_record, event_record}]`; the lease is released (event `release {leases}`) when the SDK's `Read` is garbage. |

Operator input, artifacts, figure/animation saves, fits, acquisition, snapshots and
`core` requests are answered by the application. The host passes those on (`HostEvent::Request {ticket, op, params, origin, lent}`, `lent`
holding the blocks its arrays are on) and the
run thread waits for `PluginHost::respond(ticket, result)`.

**Processors** have no `execute`. At `run.start` the core subscribes the node's
dataset input (the first whose `enabled_when` the run's values meet;
`RunRequest.source_signal`; with `select_bundle`, every output of that signal's
producer, delivered together) on the source's generation: exact under v2's follow
bounds (1024 events pending, pinning at most max(128 MiB, 32 × the first event's
own payload); overflow fails the run), a live finite source first replaying the
events it has committed, or latest as declared. A following processor (`PluginHost::follow`, from Start until Stop) is
subscribed instead at each source generation's `run.begin`, before its first
commit, and the run the application starts on that generation takes it: however
late that run starts, it reads the generation from its first event. Each delivery is the request `run.deliver {run, kind: "data", inputs:
{member: read}, anchor, schemas?, run_records?, refs_reset?}`. A delivery's read is a `runtime.read` answer
without `lineage`, and a run is sent each schema and each source run record whole once: its reads name their
schema, canonical schema and run record by number (`schema_ref`, `placement.canonical_schema_ref`,
`run_record_ref`), and the delivery that first names a number carries it in `schemas: {number: tree}` or
`run_records: {number: record}`; the run keeps that table while it lives, and a following run's first delivery
from a new source says `refs_reset: true` (the table starts again). Deliveries leave one at a time and the run
takes them in order, so a number is defined before it is named.
The SDK builds `SignalValue`s (`name`, `snapshot`,
`canonical_schema`, `cell_origin`, `run_record`, `event_record`, `lease`, `key`),
calls `evaluate(value)` (or `evaluate_inputs({member: value})`, each member made a
`SignalValue` when the node first reads it; the leases of members it never reads
are released after the commit), on the first
delivery `describe_run(inputs)` as the run record, commits the returned outputs
with the anchor as parent and its ordinal, and only then answers, so the next
delivery waits (v2's follow processor). `kind: "sealed"` completes the run (it
fails if nothing was delivered), `"retired"` stops it (the source moved on),
`"failed"` fails it with the core's reason. A finite source that has sealed, or a
sealed Snapshot (a saved Figure's dataset in a Figure Viewer), is
delivered whole, once, then `sealed` (v2's frozen processor): each member is its
run's complete dataset (`segments` as placed), its own `canonical_schema` with
`cell_origin` (0, 0); a sealed live source delivers its current event.

A node of another kind started with a source signal over an exact dataset input
with `consumer: "plugin"` is subscribed the same way, from the source's current
event, and gets the same `run.deliver` requests;
its `execute` takes them in order with `ctx.next_input(timeout)` (acknowledged as
taken, so the next is sent) and drops what came before a sampling boundary with
`ctx.discard_inputs()`. A `sealed`/`retired`/`failed` delivery ends the wait with an
error. Over a `latest` input it reads its source by name (`ctx.read`): nothing is
delivered and the run is no route of that signal (v2: `source_name` only for exact).

A delivery waits as long as its run: answered when the run takes it, refused
(`stale_run`, its leases let go) when the run ends first, lost with the process; how
long a run leaves its input waiting is not a health fact (an untracked request). A
processor that ends before its source (stopped, failed, its source run replaced) is
withdrawn with its signals (v2); one completed with its source is sealed. Closing a
node instance withdraws its last run.

A processor's subscription declares it a member of its source signal's same-shot group
(`Runtime::begin_consuming` with the source run): from its run's begin until it is
withdrawn, or released when the node no longer follows its source (Stop:
`Runtime::release_route`; its completed run keeps its signals), whether it has published,
has ended with its source, or is still of a replaced source run. While it has no result
of the source's newest shots, a board showing both keeps the last shot both have; they
flip together. A run following its source signal across runs (no source run) is a member
only while it runs.

`ctx` (the SDK's `RunContext`) keeps v2's `NodeExecutionContext` surface:
`instance_id`, `generation`, `cancel_requested()`, `report_ready()`,
`report_progress()`, `set_run_record()`, `commit_live(outputs, source_publication=)`,
`declare_now()` (what was declared stands before a first commit), `commit_events(outputs, ordinals=, shot_times=, origins=, records=)` (new, with `LiveEvents`),
`request_operator_input()`, `run_directory`, `register_artifact()`,
`seal_terminal(accept_stop=, partial=)` and `register_partial_exit_writer(writer)`.
Sealing is decided by the core; the SDK only caches its acknowledgement. The
partial writer remains plugin code, runs once with the core-decided
`"stopped"`/`"failed"` status before `run.finish`, and its failure is retained in
the final error. Protected terminal/partial cleanup may use pure native services
and artifact publication after Stop; it grants no additional device claims. A v2 `execute(context)` therefore ports unchanged. `set_run_record` plus the first
`commit_live` produce `run.begin` (outputs not present in the first commit must
be declared with `ctx.declare_output(name, schema, storage)` before it, because
Runtime declares every output of a generation at once).

### 5.1 Core-consumed scan input

An exact, single-signal input of a non-processor node may declare
`consumer: "core"` (`DatasetInput(..., consumer="core")`). The default is
`"plugin"`. Core consumption keeps the existing bounded Runtime subscription
in the host; it sends no `run.deliver` messages or arrays to Python. A run may
prepare one scan service over that input. Other input and processor semantics
are unchanged.

`ctx.scan(spec)` creates a run-scoped `NativeScan`. The operation names below
travel as `service` requests with `{operation, params, instance, run}`; the SDK
wraps the scan calls in that envelope. Workbench's `scan::execution` implements
the operations using a desktop-supplied context over the host's generic exact
input and run resources. Every operation after prepare carries the returned
opaque handle:

| Operation | Parameters / result |
|---|---|
| `run.scan.prepare` | `spec: {axes, scan_repeats, run_repeats}`; returns handle, ordered ports, outer/inner/point/shot counts, outer/inner widths and maximum row-page values |
| `run.scan.rows` | `scope: outer/inner/all`, `start`, `count`; returns a bounded page `{rows}` |
| `run.scan.arm` | `row_offset`, `scan_repeat_base`, `sweeps`, `actual_devices`, `quiet_seconds`, optional `new_source`; checks segment order and drains pre-segment input |
| `run.scan.configure` | `output`, `record`, optional `effective_inner` receipt in plan units/order; freezes effective geometry and run record once, returning effective axes |
| `run.scan.collect` | `timeout_seconds` (0..1), `max_events` (1..4096); returns collected count, segment written/total, run written/total and `complete` |

Each axis states `{port, identity, label, unit, values, coordinate_of,
host_advanced}`. `identity` is the stable owner-supplied name from which Dataset
axis IDs derive; `label` is display text. `coordinate_of` names an independent
parent port, or is null. A group with a host-advanced member is wholly outside
the board-advanced groups. Values must be finite and nonempty, linked lengths
must match, and counts must fit the native layout. Port owners supply their
units and device facts; device-specific conversions remain with the adapter.

Prepare and configure keep axis vectors, not expanded host × board rows. Pulse
resolves the run's plan handle natively and admits slots, counter limits and a
64 MiB conservative materialization budget before constructing its inner table.
Full tables and applied snapshots stay in Rust; only compact receipts cross to
Python. Effective inner coordinates must map one authored value consistently; they do
not change group lengths or the scan's traversal. The source's singleton
Repeat carrier is consumed, its Point/Cell axes preserved, and `scan.repeat`
and `pulse.run` describe the scan's two repeat counts. Implicit integer repeat
coordinates are semantically equivalent to the older explicit vectors.
Outer coordinates retain their nominal setpoint semantics, including a pulse
column linked to a host axis; actual carried values remain in the applied Pulse
snapshot. Device readbacks are recorded separately from those setpoints.

Workbench scan configuration freezes `record.plan` and `scan_shape` from the
effective factored plan. The first collect derives the output schema from its
source event and uses the host's same declaration path as `run.begin`; the host
adds node identity and apparatus facts. Each commit reuses the source DataBlock and real parent. It
preserves the source ordinal and source event record, merging the segment's
device readbacks into `actual_devices`. The output keeps the prior scan's
unstamped-shot-time contract; the parent's timing remains in its provenance.

Within a segment, source ordinals must be consecutive. Source generation or
schema changes, source termination, queue overflow and malformed placement
fail collection; collection failures remain latched. Cancellation is checked
between events and during sliced native waits. Ending the run drops its input
and plan, and old run IDs or handles cannot be reused. Completion/Stop retains
committed finite data under the existing Runtime seal rules.

The caller arms before firing and configures before collecting. Python still
owns board DONE/fault checks and missing-source grace. It may use zero quiet
time at a later arm only after both the previous expected records and board
DONE; the first segment uses the legacy quiet drain, now bounded to two
seconds and cancellable. A publication boundary alone does not prove that a
physical device has drained all delayed triggers.
When the first segment will explicitly start an acquisition, `new_source=true`
forgets the old subscription before that start. The follower remains registered
and adopts the new generation before its first publication, so the old source's
Stop cannot fail the new collection. A source cannot be replaced this way after
the scan has published data.

### 5.2 Capture ownership and batch evidence

Each Python driver owns its `RecordQueue`: the record FIFO, condition variable,
capacity, producer ordinal, readiness, worker, and SDK buffer lifetime change
under one local owner. Queue push/read operations make no native calls. A failed
join keeps that worker and capture retryable; overflow retains accepted records
and reports the failure before data is handed out.

`zlc-core::capture` supplies pure typed validation of acquisition evidence at
batch and end boundaries. The Workbench service adapter exposes `capture.batch`,
`capture.group` and `capture.terminal` through the existing native request path.
A batch supplies its expected start, observed ordinals and group size; the core
returns the next ordinal and completed group boundaries. Terminal validation
checks stopped/joined evidence, expected counts and any partial cycle allowed by
Stop. These read-only validations remain available during cleanup while the run
is active.

Camera and waveform nodes validate each nonempty intake batch once. Camera
buffer assembly uses the returned boundaries, with no per-frame group RPC;
monitor reads and drains use bounded batches. First-frame action callbacks keep
their existing read granularity. The driver queue is a local implementation,
not a second copy of a native queue state machine. No capture DLL, C ABI handle,
or ctypes/JSON exchange occurs for a queue operation.

### 5.3 Run artifact publication

`RunContext.artifacts` provides `write_json`, `write_text`, `write_bytes`,
`write_file`, `stage`, and `register`. Python retains scientific JSON/NPZ formats;
the core owns publication. `write_file(path, writer)` stages a same-directory
file, lets the scientific callback fill its stream, then asks the core to sync,
atomically publish and acknowledge the parent directory. An exception in the
writer aborts only the temporary. Stages are also retired with their owning run.

`run.artifacts` actions are `stage`, `finalize`, `abort`, `write_text`,
`write_bytes`, `write_json`, `register`, `flush`, `mkdir`, `makedirs` and
`claim_directory`. A stage returns `{token, path, target}`. Finalize/abort requires
that run's token. Write/stage options include `name`, `role`, `contract_id`,
`lineage` and `unique`; a supplied name registers the result in the same native
operation. Unique publication never overwrites an occupied name. A name may be
updated at its original path but cannot silently change to another artifact.

The store resolves paths before checking containment, including existing links;
`start.json` and `run.json` remain reserved. Registration checks a complete regular
file and records size, run identity, role, contract and supplied lineage.
Figure and animation paths use the same confinement rule. Unfinalized stages
are bounded to 128 per run, and operations on a retired run are refused.

Replies use `{ok: ...}` or `{error: {type, message, published}}` inside the normal
RPC result. If replacement succeeded but directory acknowledgement failed,
`published` names the complete visible file. The SDK raises
`DirectoryDurabilityError(published=...)` and leaves that file in place. A named
visible result is registered with `durable: false` and its publication error;
it is never silently presented as durable or removed as a failed temporary.

Offline `zlc_durable` calls use the same `zlc-archive::durable::StagedWrite`
implementation through the headless desktop service entry point. During a hosted
run they route through
the run-confined artifact facade. Directory fsync, unique-name collision handling
and atomic replacement have one implementation; payload encoding remains Python.

### 5.4 Application services and offline tools

The SDK entry point is `zlc_plugin.services.call(operation, **params)` or
`ctx.service(operation, **params)`. Hosted calls use the `service` request
`{operation, params, instance?, run?}`. The SDK client and wire codec live in
`zlc_plugin`; importing them does not import/start the plugin server.

The desktop installs the handler through `PluginHost::install_services` before
starting plugins. The host checks the calling process/run and supplies generic
array, input and resource ownership. `zlc-desktop::plugin_services` connects that
context to `zlc-workbench::services` and `zlc-workbench::scan::execution`.
Workbench owns JSON adaptation, operation names, diagnostics and opaque service
handles. Typed plans, parameter contracts, capture validation and numerical
kernels live in the corresponding `zlc-core` modules. Common inline-array
encoding is in `zlc-archive::array_wire`; live arrays use the host's shared-memory
ownership rules.

Outside a host, the SDK starts `zlc-desktop.exe --plugin-services` and reuses its
framed stdin/stdout connection. The mode dispatches the same Workbench adapters
before window, device or plugin-host startup. `ZLC_NATIVE_EXECUTABLE` selects an
explicit desktop executable. There is no separate service executable, capture
DLL or additional Python client package. Offline arrays use the inline codec;
hosted arrays keep the shared-memory path.

`scan.compile` normalizes, compiles or previews the authored document. Compact
Range descriptors `{start,stop,count}`, explicit Values, linked formulas and
inactive banks are preserved as authored data. `scan.order` and `scan.bind`
provide execution order and unit/range/port admission for draft and runtime
callers, including binding views that select only some axes of a linked group.
TypeScript sends authored patches and accepts canonical preview rows only when
their echoed source matches the current draft. It does not expand numeric grids
or interpret formulas. The run-scoped `scan_plan {handle,outer_index}` Pulse
parameter resolves to the native `zlc-core::scan::Plan`; full applied tables stay
with the native execution state for run records and LOAD/FIRE checks.

`fields.project` and `fields.value` adapt `zlc-core::parameters` for the Workbench
and Python callers, including typed projection failures and inactive draft values.
Workbench retains its presentation fields. `units.*` adapts the core registry,
array conversion and quantity parsing/formatting rules. Custom unit definitions
remain explicit data, including device-specific loads; Python caches immutable
results with bounded capacity and invalidates them when its registry changes.

`scientific.*` adapts `zlc-core::scientific` axis/selection/reduction plans, bulk
typed operations and grouped accumulators. Plot and Derive use the same numerical
kernels and SEM rules. Hosted service handles live in generic run resources;
ended runs cannot access their former state. The desktop chooses cancellation
policy: scientific computation checks Stop unless protected terminal work is
active, scan execution requires a running run, and pure capture evidence remains
available during cleanup while its run still exists. The host performs the
common ownership/liveness checks without choosing these domain policies.

The offline mode accepts `durable.*` for local file ownership. Hosted run writes
use `ctx.artifacts` and its run-directory authority; application service dispatch
refuses a durability request that would bypass that path. Non-run plugin caches
use the headless desktop mode for their local files. Offline service state is
keyed by scope and can be released with `scope.release`; it also ends with the
process. Native engine code and the Python SDK are shipped together.

## 6. Data plane

**Blocks.** Bulk values live in shared-memory blocks: named, paging-file-backed
file mappings (`Local\zlc.<session>.<pid>.<n>`), sizes in 64 KiB multiples, made
by the producing process's SDK arena and recycled by it. A lent array is

```json
{"$array": {"block": "Local\\zlc.4f2a.9120.17", "size": 2359296, "offset": 0, "dtype": "|u1", "shape": [1200, 1920]}}
```

C-contiguous, little-endian, `offset` aligned to the dtype, `size` the whole
block (receivers map a block once). The pid in the name is the owner. Arrays under 16 KiB are
sent inline as `{"$inline": {"b64": "...", "dtype": "<f8", "shape": [3]}}`.

**Ownership.** The core is the block registry. A block named in a message's
`blocks` is lent: every receiver (the core's Runtime or another plugin) holds it
until it releases it, and the producer may not write the block again until the
core sends it `release`. The core maps each block read-only once and wraps the
view as an `OwnedBuffer` (`zlc_core::OwnedBuffer::from_foreign`) with no copy;
dropping the last Runtime reader releases it. A plugin receiving an array holds
it while any numpy view of it is alive; the SDK sends one release per lend when
the view is garbage. If the producer dies, everything the core already mapped
stays valid (the section lives while any view exists).

**Copies.** The SDK passes an array by reference when it is a view of an SDK
block (`ctx.allocate(shape, dtype)`, or an array received from another plugin);
any other numpy array is copied once into a fresh block when sent. A camera frame
therefore costs exactly the copies its adapter already makes. Core-owned (Rust
heap) values a plugin reads are exported into a core block once and shared by
every reader of that value.

**Datasets.** A Runtime value on the wire is a DataBlock:

```json
{"values": {"$array": ...}, "validity": true, "sigma": null}
```

`validity` is `true | false | $array(bool)` in the compact component layout;
`sigma` is `null | $array(<f8)`. Schemas are the `zlc_data.DatasetSchema` tree
exactly as `zlc_data.codec.dataset_schema_to_tree` writes it and
`zlc_archive::tree::schema_from_tree` reads it (one grammar for archives and
plugins). An output's schema is declared once at `run.begin`; a commit's block
uses it unless it carries `"schema": tree`, which must then equal the declaration
(Runtime refuses `SchemaChanged`). Validity, sigma, typed axes, units and
coordinate frames survive as they are; nothing is inferred from names.

**Reads.** `runtime.read` returns `{lease, key: {generation, signal, revision,
ordinal}, schema, data | segments}` with its schemas whole (a delivery's reads name
them by reference, section 5); a window read is its entries with relative
coordinates (newest 0). The lease pins exactly that publication, like a Frozen
reader, independently of live eviction. Exact subscriptions follow Runtime's
`DeliveryPolicy::Exact` bounds: at most `max_events` unreleased deliveries are in
flight; overflow ends that subscription with an error, never a silent drop.

## 7. Errors, cancellation, timeouts, Stop all

Every error is `{kind, type, message, traceback}`. `kind` keeps v2's command
outcome distinctions:

| kind | Meaning | Device state |
|---|---|---|
| `refused` | Rejected before anything was written (`TuneRefused`, validation) | known old |
| `invalid` | Arguments do not match the schema | known old |
| `failed` | The operation raised after it may have acted | unknown |
| `timeout` | No reply by `deadline_ms` | unknown |
| `lost` | The process exited or stopped answering | unknown |
| `cancelled` | Stopped by `run.stop` | as reported |
| `stale_run`, `unknown_op`, `not_claimed` | Protocol/authority refusals | known old |

Python exceptions round-trip: a plugin calling another plugin through `device`
receives the original type when it is a builtin (`TimeoutError`, `ValueError`,
`RuntimeError`, `TypeError`, ...) or `zlc_plugin.PluginError(kind, ...)` otherwise,
with the remote traceback attached as a note.

Deadlines cover writer admission, pipe contention and reply waiting as one
monotonic budget. The single native wire writer has separate bounded normal and
control queues (64 frames each); it discards an expired unsent frame and sends
the remaining budget to the SDK. It also stamps `deadline_monotonic_ns` in the
shared Windows QPC clock (and `deadline_unix_ns` as a fallback); the SDK takes
the earlier of that absolute deadline and its receipt-relative budget. Thus a
frame delayed inside a blocked write cannot acquire a fresh execution budget
when it finally arrives. A frame already being written must finish
without interleaving. Pending requests remain owned until the actual response or
process exit, including a caller that has stopped waiting. Normal pending-call
admission is capped at 256, with 64 additional control slots.

The Python SDK admits at most 64 ordinary requests per instance lane, with a
reserved control wake slot. It checks a queued request's expiry before executing
its handler. Declared concurrent calls have eight ordinary slots and two
independent `stop`/`make_safe` slots. Stop can interrupt admitted calls during
shutdown, but cannot race the final SDK Close after the last call relinquishes
the handle. Catalogue/discovery/resolve together admit four calls per process.
Overload is an explicit `refused` reply, never an unbounded thread or queue.

On expiry the caller gets `timeout`. A callee already executing is not interrupted (a running operation stops cooperatively, through
its bounded timeouts or the run's `run.stop`). An instance that leaves a request
unanswered for deadline + 5 s is **unresponsive**: the core marks it failed, seals
its running generation as failed with that reason and tells the process `abandon`;
the process and its other instances carry on. A new Open is explicitly refused while
the previous operation still owns its device; it may retry after cleanup completes.
A process whose reader thread misses 60 consecutive `ping`s (1 s period, sent
whatever its lanes are doing; a Numba compilation holds the GIL for seconds) is
stuck as a whole: the core terminates it. Process exit (stdout EOF) is detected
immediately; every instance in the process then fails with the exit code and the
last stderr lines as the reason, and each can be opened again.

Stop all (core, `zlc_core::control::SafeActions`): on open the core registers every
declared physical output; the safe action is `make_safe {output}` to the plugin
(deadline 2 s), which calls the adapter's `make_safe(output)`. `stop_all` runs every
one and is never refused. A device that fails or hangs is that device's error state
(v2): the runs that claim it stop with the error, nothing else is touched, and
`apply_devices` reopens it. A plugin process that exits takes every instance in it
into that state.

Claims (core): `run.start` names the device instances bound to each requirement;
the core claims them for the run. `device` calls are routed only to devices the
calling run has claimed (`not_claimed` otherwise); while claimed, operator/AI
commands that write a requirement's `protected_fields` are refused. Claims are
released when the run is terminal. A draft query has no run: its proxies send
`device` with `run: null` and the `device` instance they stand for, and the core
answers only reads of an open device, claiming nothing (v2's `declare_outputs`
read `source.outputs`): `get:<name>` properties, `tunable_fields`, and the reads
the device's `open` reply lists (`reads`: of `tunable_values`, `settings_provenance`,
`convert_tunable_value`, `read_tunable_in_unit` the ones it implements -- a scan
editor converts a knob's limits through the device's own law, an RF amplitude
into its channel's load). Anything that changes the device is refused.

## 8. Commands in the catalogue

A type's `commands` are operator/AI actions, declared with their schema:

```json
{"name": "set_exposure", "summary": "Set the exposure time; returns the actual working point.",
 "read_only": false, "params": [Field...], "input_schema": {"type": "object", ...}}
```

`drives` (SDK `Command(..., drives=True)`) marks a command that commands the device
itself (an SLM Editor's Send, Adopt, Load science context): the core refuses it while a run
claims the device ("device is in use by <node>", v2's device-command lease); a command
that only edits the device's own draft is not refused.
`file` (SDK `Command(..., file=FileParam("path", "read" | "write"))`, descriptor
`{"param", "direction"}`) names the parameter that is a file on the operator's computer
(an SLM Editor's Load/Save target and science context). The core resolves it (relative:
the workspace's) and does the file's I/O itself, so a device another core serves (§10)
reads and writes the operator's files as a local one does: for a read it sends
`{name, bytes}` (the file's name and its bytes as an array of `|u1`), for a write
`{name}`, and the reply is `{file: <bytes>, value: <the command's answer>}`, whose bytes the
core writes in place of the file. The SDK hands the handler the parameter as the path of a
private copy with that name: the file to read, or the file to write.
`read_only` (SDK `Command(..., read_only=True)`) marks a command that only reads: a
read-only MCP client (`--mcp-read-only`) is offered only those. When an instance opens, each command becomes one catalogue entry named
`<instance>__<command>` with the `input_schema` the SDK generated
from its `params` (section 2's field grammar). The core validates parameters
against that schema and sends
`command {name, params}` to the instance's lane; the reply is the tool result.
The catalogue changes when instances open and close (MCP `tools/list_changed`).
UI and MCP use the same entries; there is no second plugin registry.
Ids are not restricted (an instance is named as its apparatus names it, `n100_COM3`);
the MCP tool name spells each part as an MCP name allows (`zlc_core::control::tool_name`):
a part of letters, digits and single `_` of at most 24 characters stands as it is, any
other becomes its letters and digits, `-` and an 8-hex hash of its exact text
(`pump_laser_2-1f0c93ab__set_power`). The windows may also send `<id>__<command>` with
the id as written.

Tunable devices (v2's quartet `tunable_fields`/`tune`/`tunable_values`/
`settings_provenance`) need no per-device commands: the SDK serves them as the
standard ops `fields`, `tune {field, value, unit}`, `values` (and `snapshot`, section 5), and the Workbench
catalogue exposes them once for every instance. `fields {units: {field: unit}}` reads
those fields in the operator's spelling, and `tune {unit}` takes the value and answers
the readback in it (v2 `read_tunable_in_unit`/`tune_in_unit`: the adapter's own
conversion when it has one, an RF level into its load, else the unit registry's;
`zlc_plugin.tunable`).

## 9. Python SDK surface (`zlc_plugin`)

Declarations: `DeviceType`, `Opened`, `NodeType`, `NodeKind`, `Command`,
`DatasetOutputDeclaration`, `DatasetInput`, `DeviceRequirement`, `PhysicalOutput`.
Fields are the plugin package's own `AuthoringField`/`AuthoringSchema` (read by
attribute; the SDK has no second field grammar). Runtime side: `RunContext`
(section 5), `LiveDatasetOutput`, `Read`, `allocate(shape, dtype)`, `share`,
`group`, `PluginError`, `Refused`. Porting a v2 device: replace
`DeviceTypeDescriptor(type_id, domain, schema, capabilities, factory=...)` with
`DeviceType(type_id, family, schema, capabilities, factory=...)`; the factory
takes `(key, projected values)` and returns the adapter, or `Opened(adapter,
close)` when closing needs more than `adapter.close()` (no `InstalledLeaf`, broker
or world context). Porting a v2 node: `LogicNodeDescriptor(...)` -> `NodeType(...)`
with the same fields; `build` receives device proxies that implement the
capability Protocol by forwarding to the claimed device (no `signal_plane`
argument); `execute(context)` is unchanged; `LiveDatasetOutput` drops `coverage`
(Runtime's fact) and gains an optional `ordinal`.

Plugins import only `zlc_plugin`, `zlc_data` and their own family contracts --
never Workbench, UI, Runtime or plotting packages (v2's nodes imported
`zlc_runtime` declarations; those types now come from `zlc_plugin`). Plotting
and figure archives are native: a figure is described with
`zlc_plugin.figures` (spec trees, point overlays) and saved through
`run.figure`; the image point overlay vocabulary is data (`zlc_data.point_overlay`).

## 10. Between cores (`crates/zlc-fabric`, v2's device fabric)

Remote on is one flow whatever the device, as v2's: the other computer's Scan
hardware lists it as `remote_<instance>`, the operator adds it from that list,
and from then on it is an apparatus row like any other (Init opens it, Remove
takes it out). A device with a server of its own -- the pulse board
(`sequencer.local`), the SLM head (`slm.hamamatsu_x15213_local`, SDK
`DeviceType(announce=...)` and `Opened(admit_peers=...)`) -- is announced as the
client type another computer opens at that server (`sequencer.hardware`,
`slm.hamamatsu_x15213`, the host being the address the scan reached): Remote on
asks it `publish {published: true}`, its server admits other machines, and the
answer `{type_id, parameters}` is announced; off asks `publish {published:
false}` and the server drops them. Any other device is served by the fabric
itself and offered as `remote.device` (v2's `remote.tunable`):

These are two transports behind the same Device Manager workflow, not two
RF/SLM policies. Keep RF on the generic fabric: its USB/VISA driver already
implements the plugin contract, so a separate RF server/client would repeat
discovery, ownership, timeout and shutdown code. Keep the SLM and sequencer's
existing device protocols: SLM commands carry revision checks, phase sequence
ownership and presentation receipts; the pulse server owns board-specific
SAFE, cancellation and single-owner operations and also serves headless clients.
The fabric discovers and withdraws those endpoints without wrapping every
hardware command in a second network protocol. Camera and other plugin devices
use the same generic transport as RF, with array blocks as needed.

A device can be served by another core: the publishing core hands an open
device over (`PluginHost::lend_out`; its own runs, commands, tunes and calls are
refused until `take_back`) and answers the using core's instance ops through
`PluginHost::serve`. The using core admits it with `PluginHost::attach(instance,
descriptor, endpoint, tunable)`: the instance has the publisher's descriptor,
its outputs join the using core's Stop all (made safe through the link), its
commands the catalogue, and runs claim it like a local device. Only the
instance ops travel (`spi`, `fields`, `tune`, `values`, `snapshot`, `command`, `make_safe`,
plus `attach`/`detach`/`list`), with the same params and replies as here, and
what the device says for itself: its `device.notice` and `device.log` go to the
core using it (`lend_out`'s forward, an `evt` frame with its blocks), which hears
them as its own device's (`PluginHost::remote_event`) -- an SLM Editor's draft and
pictures reach the window of the core that opened it.

Frames: u32 little-endian JSON length, the JSON, then the bytes of every block
the message `attach`es (`[{name, size}]`); `lent` lists them with the holds'
multiplicity. The receiver reads each block straight into a block of its own
core (`Registry::export_with`) and renames the `$array` references, so a frame
lands where its node and the Runtime read it without another copy. A
connection opens with `hello` (version 5 and the machine; another version is
refused by name); both sides beat every 2 s and a peer silent for 6 s is gone.
The device attach reply carries its declared `reads` alongside `tunable`;
the receiving core preserves both for draft and run proxies. Its optional
`snapshot` boolean describes the owner's actual open adapter, rather than
guessing support from the device type or tunable status. Adding it and the
snapshot operation is backward compatible with existing fabric version 5
clients; an absent capability is not mistaken for an unsupported device.
Snapshot access requires the attached owner, like other instance operations.
Its total deadline includes waiting for the link, writer and reply, without
the ordinary response grace. A reply timeout drops only that metadata request;
so does a write timeout before any request bytes were sent. A failure after a
partial frame is a broken transport and follows the normal link recovery path.
The publishing plugin uses an untracked request so it neither fails the device
nor interrupts the owning run when the device is busy.
As v2's fabric there is no key: a core accepts other machines only while
something is published (Remote on). The
publisher drives an abandoned device's outputs safe (lost user, a user's
reconnect, `detach`, withdrawal). The user fails requests in flight on a break
(never resent), redials for 10 s, and otherwise loses the instance as a
crashed plugin's is (`endpoint_lost`: `HostEvent::Failed`, v2's device error state;
Apply opens it again).

Generic-device ownership changes are barriers: first refuse new operations,
then wait for all admitted requests to actually answer, then confirm SAFE,
and only then admit another owner or return local control. A caller timeout
does not count as a plugin reply; concurrent operations are still pending until
the plugin answers or ends. Failed draining or SAFE keeps the device reserved
and reports the failure; withdrawing Remote retries the barrier. Closing a
remote client releases its claim, never the publishing machine's device handle.
Duplicate publication cannot overwrite an owner, and queued requests from a
socket that has already closed cannot claim or drive a device. Servers with
their own protocol retain their announcement if refusing peers fails, so Remote
does not falsely report a successful withdrawal.
Initial publication also waits for local operations admitted before Remote was
enabled, including calls that have not yet sent their plugin request. It offers
the device only after those calls settle, preserving their resulting hardware
state; if that initial barrier fails, local control is restored and no remote
owner has been admitted.
