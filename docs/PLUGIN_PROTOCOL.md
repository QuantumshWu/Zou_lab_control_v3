# Plugin protocol (v2)

Python plugins (device drivers, logic nodes, the WGS/SLM glue they contain) run
outside the Rust core. The core has no interpreter and no GIL; a plugin crash or
hang loses that plugin, never the application or its data. This document is the
contract between `crates/zlc-plugin-host` (Rust) and `science/src/zlc_plugin`
(the Python SDK). Changes after announcement are small commits titled `API: ...`.

Terms: a **plugin type** is one discovered declaration (`camera.virtual_mot`, the
`camera_measurement` node). A **plugin instance** is one configured use of a type:
one Device Manager row (`mot_camera`) or one Logic row (`mot_monitor`). A **run**
is one Start..terminal of a node instance.

## 1. Process model

**One host process per plugin instance.** The core starts
`python -m zlc_plugin host` for every opened device instance and for every node
instance; it lives until that instance is closed. A node instance keeps its
process (and its warm imports, Numba caches, GPU context) across runs, so Start
does not pay interpreter startup.

**Process groups** are the one exception: a type may declare
`process_group = "<key>"`; all open instances whose types name the same key share
one host process, each still with its own lane (section 4). This exists only for
real in-process shared state: the simulation `SimulationWorld` that the virtual
camera, sequencer and SLM all observe, or a vendor library that is a process
singleton (libdaq2 addressed by serial, DCAM). It is not a performance knob.

Why per instance (rather than one shared plugin process): the v3 measurements
put the four-panel slowness on one shared GIL (`FOUR_PANEL_GIL_ROOT_CAUSE.md`);
a camera intake thread must not wait on another device's numpy or a node's fit.
A vendor SDK that hangs or segfaults takes only its own instance. SDK thread
affinity is per process. Devices open in parallel because they are separate
processes ("devices do not compose": no device sees a sibling). Cost: one Python
interpreter per instance (~40-150 MB with numpy/zlc_data), paid at open, not per
run or per shot. Plugins never render plots and never own Runtime history.

Every child is placed in one Windows Job object with KILL_ON_JOB_CLOSE, so no
plugin survives the core (no orphan holding a camera). The core spawns children
with `PYTHONHOME` removed, `PYTHONPATH` = the configured science/plugin roots,
the interpreter's base installation (a venv's `pyvenv.cfg` `home`; for conda its
`Library\bin` and the other folders activation adds) first on `PATH`, and v2's
thread policy (`OPENBLAS_NUM_THREADS` min(4, CPUs), `NUMBA_THREADING_LAYER=omp`,
`OMP_WAIT_POLICY=PASSIVE`, each unless set). The interpreter is
`ZLC_PLUGIN_PYTHON` when set, else the checkout's `.tools\science-env`
(`ZLC_V3_TOOLS` moves `.tools`): the executable needs no launcher.

## 2. Discovery and descriptors

Plugin folders follow v2's layout rule, one folder = one device:
`zlc_atom/devices/<family>/<device>/device_types.py` exporting `DEVICE_TYPES`
(a tuple of `zlc_plugin.DeviceType`) and `zlc_atom/nodes/<node>/logic_node.py`
exporting `LOGIC_NODE` (a `zlc_plugin.NodeType`). Family folders hold only shared
contracts. The descriptor is written once, in Python, beside the code it
describes. The core learns it by running one short-lived
`python -m zlc_plugin describe <package...>`, which imports each leaf module and
prints one JSON document `{"protocol": [2, 0], "types": [...], "errors": [...]}`.
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
  "process_group": null,
  "capabilities": ["camera.adapter"],    // device: SPI contracts it implements
  "parameters": [Field...],              // instance configuration (authoring schema)
  "physical_outputs": [{"name": "mot_coils"}],   // device only: what Stop all drives safe
  "commands": [Command...],              // operator/AI commands (section 8)
  "control_window": "slm-editor",        // device only: the window its Control opens ("" = the field form)
  "announces": false,                    // device only: it has a server of its own (section 10)
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
minimum, maximum, step, unit, choices: [{value, label}], segmented, blank_label,
description, enabled_when: [field, [values]] | null, derived, group, columns}` with
`type` in `int | float | bool | str | text | pair | choice | rows | resource`. Choices
are `(value, label)` pairs stated by the declaring plugin; no layer transforms a value
into a label. Units are the field's own unit string (zlc_data unit grammar). `group`
names the form section the field stands in ("Imaging"); the form titles each change
of it.

Draft-dependent declarations stay functions of the plugin, answered by the node
instance's own process through `query` (section 5): `declare_outputs`,
`declare_previews`, `field_availability`, `resolve_defaults`, `admit_source`,
`admit_resources`, `estimate_run`, `selection_patch`, `describe_output_frames`.
`queries` lists which ones a type implements. They are pure functions of the
draft values, resolved devices (as proxies) and resources; they never start work.
`editor_facts(values, resources, devices, source)` answers the model of a node's own
editor, `{editor: kind, manages: [field, ...], ...}`: the UI's editor of that kind
draws it in place of the fields it manages and writes them back with `edit_node`; it
holds no rule of the node's (the model says what is offered, derived, kept or wrong).
`source` is the node's source frames `[axis_name | null, [labels...]]` or null.
Kinds: `scan_plan` (seamless scan, `scan/editor.py`: `ports [{port, label, unit, lo,
hi, seed_lo, seed_hi, group, leaf}]`, `withheld {port: reason}`, `families`,
`host_advanced` prefixes, `summary`/`problem`, `api {fields, overrides, parked, note}`
or null) and `pair_matrix` (frame survival, `frame_survival/editor.py`: `grid {rows,
columns, cells}`, `selected`, `states`, `tips`, `identity`, `override_rows/_cells`,
`list_rows/_cells`, `keeps_list`, `summary`, `note`, `parked`, `error`).
`resolve_device_claims(values)` answers `{argument: {instance, fields}}`: devices
the draft itself names (a scan's `device:<key>:<field>` axes), beside the declared
requirements; the core claims those fields for the run (Device Control cannot
move them meanwhile) and passes each device as `argument` (a tunable proxy).
`resolve_field_choices(values, resources)` answers `{field: [[value, label], ...]}`
for fields whose options come from the draft (an SLM rearrangement's imaging
periods are the chosen pulse's periods); the form shows that field as a choice,
and a value it holds that is not among them as unavailable, never replaced.

A node's page draws the newest answers it has, never an empty draft: a node just
added draws `node.answers` (the SDK answers each draft query for the type's default
draft, no devices, files or source, while it describes the type; one that cannot is
left out), and a draft just edited draws the answers of the one before it until its
own land. Only a Start waits for the draft's own answers (`Checking the draft…`).

Capabilities name a device SPI by token (`camera.adapter`, `rf.source`,
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
| open | `open {instance, type_id, module, parameters, group}` | imports the type's module; device: `factory(instance, projected parameters)` builds and opens the adapter; node: keeps the draft | `{state, tunable, reads}` |
| configure | `configure {instance, parameters}` | device: `adapter.reconfigure(values)` or refuses | `{state}` |
| start (node) | `run.start {instance, run, values, devices, resources, source_signal, run_directory, origin}` | builds the node (`build(**values, **devices, **{arg_key: instance})`), runs `execute(ctx)` on its run thread | `{run}` immediately; terminal via `run.finished` event |
| stop (node) | event `run.stop {instance, run}` | sets the run's cancel flag | none |
| health | `ping` | answered by the reader thread (proves the interpreter and GIL are alive) | `{}` |
| hardware scan (process) | `discover {type_id, module}` (no instance; in the type's process, so a group's shared runtime -- DCAM -- is the one asked) | the type's `discover()` on a thread of its own | `{found: [{instance_id, role, parameters}], notes}`; the core stops waiting at its deadline (20 s, v2) and drops a late answer |
| close | `close {instance}` | device: its closer (stop capture, release SDK); node: stop run | `{state}`; the process stays |
| exit (process) | `exit` | closes what is left | `{}`, then exits 0 |

`group` is the process group's parameters as the installation states them
(the simulation mapping); `zlc_plugin.group()` returns them and
`zlc_plugin.share(name, build)` gives a group's devices one shared object, built
by the first and dropped by the last to close. Device open is per instance and
parallel; the core admits opened instances in the operator's order (v2's
`_build_together`). A failed open reports its error and the instance is gone;
nothing half-open is admitted. The core ends a process with `exit` (or by
closing stdin); it waits at most 5 s, then terminates it.

A run ends with event `run.finished {run, outcome: completed|stopped|failed,
result, error}`. The core seals the run's Runtime generation with that outcome
(`Runtime::seal`), never the plugin. Stop is not deletion: a stopped run's
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
{"t": "evt", "op": "run.finished", "p": {...}, "blocks": [...]}
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
A device type may name SPI methods its adapter answers concurrently
(`DeviceType(concurrent=(...))`, descriptor `concurrent`): an `spi` request for one
of them runs on its own thread, beside the lane. An SLM's `play_phase_sequence`
holds the lane for the whole movie while the caller streams `submit_phase_frame`
and may `cancel_phase_sequence`; a callable (v2's `stop_requested`) cannot cross
processes, so the caller cancels instead.

## 5. Operations

Core -> plugin requests (each names its `instance`): `hello`, `open`, `configure`,
`close`, `exit`, `ping`, `spi {method, args, kwargs}` (call the open device's
adapter; `get:<name>` reads a property), `command {name, params}` (section 8),
`fields {units?}` / `tune {field, value, unit?}` / `values` (the tunable quartet), `make_safe
{output}`, `publish {published}` (Remote of a type that `announces`: section 10), `query {name, values?, devices?, resources?, source?, output_name?,
installation?}` (the facts by name; the SDK calls the v2 callable with them in v2's
positional order, `zlc_plugin.declare.QUERY_PARAMETERS`; `devices` become proxies), `run.start`. Events: `release
{blocks}`, `run.stop {run}`. A request with `"plain": true` gets its result as
plain JSON (dataclasses as objects, tuples as lists, dtypes and enums as their
values, arrays still as references); the core always asks for that, while a
plugin-to-plugin call keeps the tagged types of section 6.

Plugin -> core events: `release {blocks, leases}`, `run.ready`, `run.progress`,
`run.finished {run, outcome, result, error}`, `device.log {instance, line}` (one line of
an open device's own log, v2's: each INFO record of the loggers under `zlc_pulse.remote`,
`zlc_atom.devices` and the type's `log_channels`, as `[HH:MM:SS] message`, for the devices
it is about -- a logger under the type's `log_channels`, or a line ending in
`device=<identity>`, the adapter's `identity`). A device's `open` answer says
`narrates` (it has `log_channels` or an `identity`). And `device.notice {instance, name, value}`:
an open device tells the core a fact of its own outside any request, its arrays lent
(`HostEvent::Notice`; an SLM Editor's draft, sent whenever it changes -- a solve ends
long after the command that started it). SDK: `zlc_plugin.notifier()` inside a device
command gives the instance's `Notifier`, callable from any thread while that device is open.

Plugin -> core (only during a run, bound to its run id):

| op | Meaning |
|---|---|
| `run.begin {outputs: [{name, schema, storage, index_by_source}], record}` | Declare the run's outputs and run record. `Runtime::begin` under owner `<instance>` with the record as the run's metadata (who started it goes in the seal); signals are the names the run's start gave (`RunRequest.outputs`, e.g. `@logic/<node>/<output>`), else `<instance>/<output>`. Storage `latest | finite`. |
| `run.commit {inputs: [{output, ordinal, origin, data, schema?}], metadata, parents}` | One atomic bundle (`Runtime::commit`): siblings commit together or not at all. `origin` is the finite `(repeat, point)` placement; a finite event carries its chunk `schema` when it differs from the last one sent. `parents` are read leases the node consumed. Metadata is `{records: {output: event_record}, shot_time_seconds: {output: t}}`. Returns each output's publication key. |
| `run.operator_input {kind, title, message, payload}` | Blocks the run thread until the operator answers or the run stops. |
| `run.artifact {name, path, role, contract_id}` | Register a completed run-contained artifact. |
| `run.figure {path, data, overlay, recipe, source, lineage, image}` | Save one figure (`RunContext.save_figure`, `zlc_plugin.figures`): the core writes the `zlc.figure` archive `<path stem>.npz` and, when `image`, renders `path` (`.png`/`.pdf`/`.svg`; no suffix = `.png`). `data` is one complete dataset `{schema, data: {values, validity, sigma}, block_id, revision, generation}` (`validity` true/false or its storage mask). `recipe` is `{spec, parameters, size}` plus any of `viewport, classifier_thresholds, facet_focus, fit, selectors, interaction, presentation`: `spec` is the archive recipe's spec tree; the core completes `parameters` with the kind's defaults, validates `size` and normalizes `classifier_thresholds` as a plot session keeps them (v2 `encode_plot_recipe`). `overlay` is null or `{revision, coordinates (N,2), point_ids, labels, static_statuses, status (a dataset), paths_xy (N,K,2)}`. `source`/`lineage` go into the archive's sections as v2's `save_figure_artifact` put them. Answers `{archive, image}` (`image` null when not drawn). Registering them as artifacts stays the plugin's. |
| `run.animation {path, images, frame_ms}` | A looping GIF of the run's images (its figure previews, inside the run directory), `frame_ms` each, each frame on its own palette (v2's Pillow save of the recording previews); answers `{path}`. `RunContext.save_animation`. |
| `run.acquire {node, completion}` | Start the acquisition node this run drives (v2's `restart_logic`); answered at `ready` or `terminal`. |
| `core {command, params}` | Run one catalogue command (e.g. a Pulse operation) under the run's origin; the run's claims apply (a `pulse_*` command needs the sequencer claimed). From a draft query (`zlc_plugin.core`, `run: null`) only a `read` command is answered: an estimate's `pulse_estimate`. |
| `device {argument, method, args, kwargs}` | Call a claimed device's SPI (section 7 checks apply). `tunable_fields` answers the same rows as the `fields` op, as data. |
| `runtime.read {signal, mode, width}` | Read a Runtime signal: `event | dataset | window` (the event of a dataset publication, e.g. a Fit over a finite run, is its dataset). Returns `{lease, key, schema, window, segments: [{origin, extent, data}], metadata, run_record, placement, lineage}`, `lineage` its ancestry nearest first, each publication once: `[{key, run_record, event_record}]`; the lease is released (event `release {leases}`) when the SDK's `Read` is garbage. |

The six requests above `device` are answered by the application, not the host:
the host passes them on (`HostEvent::Request {ticket, op, params, origin, lent}`, `lent`
holding the blocks its arrays are on) and the
run thread waits for `PluginHost::respond(ticket, result)`.

**Processors** have no `execute`. At `run.start` the core subscribes the node's
one dataset input (`RunRequest.source_signal`; with `select_bundle`, every output
of that signal's producer, delivered together) on the source's generation,
`Start::Current`, exact (64 events / 2 GiB, overflow fails the run) or latest as
declared. A following processor (`PluginHost::follow`, from Start until Stop) is
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
`"failed"` fails it with the core's reason. A finite source that has sealed is
delivered whole, once, then `sealed` (v2's frozen processor): each member is its
run's complete dataset (`segments` as placed), its own `canonical_schema` with
`cell_origin` (0, 0); a sealed live source delivers its current event.

A node of another kind started with a source signal over its dataset input (the
seamless scan: it fires the pulse that drives its own source) is subscribed the
same way and gets the same `run.deliver` requests; its `execute` takes them in
order with `ctx.next_input(timeout)` (acknowledged as taken, so the next is sent)
and drops what came before a sampling boundary with `ctx.discard_inputs()`. A
`sealed`/`retired`/`failed` delivery ends the wait with an error.

`ctx` (the SDK's `RunContext`) keeps v2's `NodeExecutionContext` surface:
`instance_id`, `generation`, `cancel_requested()`, `report_ready()`,
`report_progress()`, `set_run_record()`, `commit_live(outputs, source_publication=)`,
`request_operator_input()`, `run_directory`, `register_artifact()`,
`seal_terminal(accept_stop=, partial=)` and `register_partial_exit_writer(writer)`,
both answered inside the plugin: a sealed run ignores a later `run.stop`; an
error out of a stopped (unsealed) run ends it `stopped`, as v2's NodeHost did; the
writer runs once with `"stopped"`/`"failed"` before `run.finished`, and its own
failure is that run's reported error. A v2 `execute(context)` therefore ports unchanged. `set_run_record` plus the first
`commit_live` produce `run.begin` (outputs not present in the first commit must
be declared with `ctx.declare_output(name, schema, storage)` before it, because
Runtime declares every output of a generation at once).

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

Deadlines: the caller states `deadline_ms`; on expiry the caller gets `timeout`.
The callee is not interrupted (a running operation stops cooperatively, through
its bounded timeouts or the run's `run.stop`). An instance that leaves a request unanswered for deadline + 5 s,
or, with no request in flight, misses 30 consecutive `ping`s (1 s period; a Numba
compilation holds the GIL for seconds), is **unresponsive**: the core marks it
failed, seals its running generation as failed with that reason, and terminates
the process. Process exit (stdout EOF) is detected immediately and handled the
same way, with the exit code and the last stderr lines as the reason.

Stop all (core, `zlc_core::control::SafeActions`): on open the core registers every
declared physical output; the safe action is `make_safe {output}` to the plugin
(deadline 2 s), which calls the adapter's `make_safe(output)`. `stop_all` runs every
one and is never refused. A device that fails, hangs or exits is that device's error
state (v2): the runs that claim it stop with the error, nothing else is touched, and
`apply_devices` reopens it.

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
editor converts a knob's values through the device's own law, an RF amplitude
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
standard ops `fields`, `tune {field, value, unit}`, `values`, and the Workbench
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

A device can be served by another core: the publishing core hands an open
device over (`PluginHost::lend_out`; its own runs, commands, tunes and calls are
refused until `take_back`) and answers the using core's instance ops through
`PluginHost::serve`. The using core admits it with `PluginHost::attach(instance,
descriptor, endpoint, tunable)`: the instance has the publisher's descriptor,
its outputs join the using core's Stop all (made safe through the link), its
commands the catalogue, and runs claim it like a local device. Only the
instance ops travel (`spi`, `fields`, `tune`, `values`, `command`, `make_safe`,
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
connection opens with `hello` (version 4 and the machine; another version is
refused by name); both sides beat every 2 s and a peer silent for 6 s is gone.
As v2's fabric there is no key: a core accepts other machines only while
something is published (Remote on). The
publisher drives an abandoned device's outputs safe (lost user, a user's
reconnect, `detach`, withdrawal). The user fails requests in flight on a break
(never resent), redials for 10 s, and otherwise loses the instance as a
crashed plugin's is (`endpoint_lost`: `HostEvent::Failed`, v2's device error state;
Apply opens it again).
