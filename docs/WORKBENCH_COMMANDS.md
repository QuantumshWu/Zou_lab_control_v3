# Workbench: domain model and commands (v3, line B)

The contract the UI (line C) builds its views on and the plugin host (line A,
`docs/PLUGIN_PROTOCOL.md`) serves. After announcement it changes only in commits
whose subject starts with `API:`. v2 reference: `zlc_workbench` (session,
device_manager, device_use, logic, console, console_layout, panel_state,
apps/task_console, apps/mcp), `zlc_runtime` (host, task_run), `zlc_atom/install`.

## 1. Ownership

| Owner | Holds | Never holds |
|---|---|---|
| `zlc-workbench` (Rust crate, no GUI, no Python) | workspace; apparatus draft and live device instances; claims; logic rows (drafts, running configurations, follow intent); runs (lifecycle, run directory, operator requests); board (panel order, titles, layout files); the signal-name grammar | scientific data, plot semantics, plugin science |
| `zlc-core` Runtime / SafeActions | publications, generations, history leases; Stop all's safe actions | names, drafts, panels |
| `zlc-plot` document (desktop science worker) | each panel's authored target, controls, fronts, fit | panel order, layout files |
| plugin host (line A) | plugin processes, their lanes, block leases, safe-action registration for physical outputs | application state: it reports facts as events |
| TS UI (line C) | views and unsent edits | everything else: it renders `state` and sends commands |

One owner turn (the desktop event loop) applies every command and every plugin event.
A command is checked against its catalogue schema, applied, and answered with the
settled `state`. Nothing
blocks the turn: plugin work is a request whose completion is an event.

## 2. Identities and names

- **Device instance** `instance_id`: stable identity and claim key, unique, v2's rule
  only: non-empty text with no surrounding space and no `/` (`n100_COM3`, as v2's scan
  names it). Its commands' MCP tool names are spelled from it (PLUGIN_PROTOCOL §8).
  Minted from the family: `camera`, `camera2`, ... **Role**: display label, unique,
  editable (v2 `role`).
- **Logic node** `node_id`: minted once from the type, never changed: `occupancy`,
  `occupancy2`, ... (v2 `_free_logic_id`). **Name**: display name, defaults to the id,
  v2 `logic_name_problem` rules (one line, no `/`, not `panel-N`, unique by NFKC
  casefold, not another node's id).
- **Panel** `panel-<n>`, monotonic per session.
- **Signals**, one grammar (v2 `stable_signal_key`): `@logic/<producer>/<output>`;
  the producer is a node id or a panel id; it is split at the last `/`. A node's
  outputs are `@logic/<node_id>/<output>` (the RunSpec names them; PLUGIN_PROTOCOL's
  `<instance>/<output>` is this). Panel outputs: `@logic/<panel>/roi_frame` (area
  selector) and `@logic/<panel>/<fit parameter>` (published fit parameters). v3's
  `panel-1/roi/frame`, `panel-1/fit/<p>` and `camera/frames` are renamed accordingly.
- **Run** `run`: `<node_id>#<n>`, n counting that node's starts this session. A task's
  run directory is reported beside it.
- **Operator request** `request`: `<node_id>#<n>:<seq>`, from the run.
- **Draft revision** `draft_revision`: increments on every accepted draft change of a
  node; commands may carry `expected_draft_revision` (stale intent is refused).

## 3. State (what `get_state` and every window `state` message report)

Existing keys stay (`document`, `panels`, `signals`, `plot_kinds`, `acquisition`,
`error`, `request`). Added: `workspace`, `apparatus`, `devices`,
`logic`, `board`, `runs`. `get_state {sections: [...]}` returns only the named keys;
`get_state {panel}` still filters `panels`.

Documents, as v2 had windows: `document`/`panels`/`signals` are the live board's
(panel ids `panel-<n>`; signals of the apparatus only), and each opened Figure is a
document of its own in its own Figure Viewer window (v2 opened one per Figure):
`viewers: [{window, document {path, title, info, flow}, panels, signals, plot_kinds}]`,
window `figure-viewer-<n>`, panel ids `figure-<n>-<k>`, and as `signals` the opened
archive's datasets (`@figure/<k>/<dataset name>`) and its panels' outputs. `open {path}`
opens the Figure in a new window; `open {path, window}` replaces that window's Figure
(as its File field does); the live board, its camera and its runs go on. A command
naming a `figure-<n>-<k>` panel, or `add_panel` of a viewer's signal, acts on that
viewer; a panel cannot be bound to another document's signal. The host sends each
viewer to its window as a `state` message `{request, error, viewer}`.

Windows, as v2 had them: Devices@Zou lab (`devices`), TaskConsole@Zou lab
(`task-console`), one FigureViewer@Zou lab per opened Figure (`figure-viewer-<n>`) and
PulseGUI@Zou lab (`pulse-editor`), and each open device's `<role> control` (`control:<device>`) and
`<role> log@Zou lab` (`log:<device>`), and a run's point review (`review:<node>`), one native window and page each; `get_state` lists
them as `windows: [{window, title, visible, size}]`. Devices opens first, and each
other window is made when it is first shown, as v2 made its windows; Init (the apparatus
initialized) shows TaskConsole in front, Shut down hides it and shows Devices; `open`
shows a FigureViewer with the Figure (closing it drops its Figure); a sequencer's Control shows PulseGUI
(closing hides it); another device's Control reads it and opens its control window
once its fields are in, at least half the standard width and as tall as its content
(v2's snug fit; closing hides it), Log opens its log window at 0.45 of the screen fit
(closing ends it), and a device that closes takes both; closing Devices or TaskConsole
closes the experiment. Each page is sent only the sections its view reads (TaskConsole:
the live board and the workbench; Devices: the workbench; a FigureViewer: its `viewer`;
PulseGUI: `pulse`; a device's windows: `devices`) and its own plot fronts; each numbers
its commands from its own base (`?base=`, a multiple of 1e12 per window made).
`show_window {window}` brings one forward; `close_window {window}` closes one as its ×
does (a Figure Viewer drops its Figure; a control or PulseGUI hides; a log ends); `save_image {path?, window?}` captures the
named window (the asking window, a remote client's default TaskConsole).

### 3.1 `workspace`
```json
{"root": "C:/lab/workspace", "data": "C:/lab/workspace/data", "today": "C:/lab/workspace/data/2026_10_20",
 "pulses": "C:/lab/workspace/pulses", "layouts": "C:/lab/workspace/layouts", "layout": {"path": "C:/lab/workspace/layouts/layout.json", "dirty": true},
 "notes": ["logic Occupancy names 'camera' as its camera, which does not provide camera.adapter; the row is kept and cannot start until another device is chosen"]}
```
The workspace is the v2 directory (`apparatus.json`, `pulses/`, `data/<YYYY_MM_DD>/`,
`config_values/`) and `layouts/`, chosen by `--workspace` or `ZLC_WORKSPACE`. Each kind of file
a window opens or saves starts at its open document, else in its own folder (`zlc_workbench::workspace`). `notes` are the last
load's warnings (v2 status strip), cleared by the next load/clear.

### 3.2 `apparatus` (Device Manager draft and catalogue)
```json
{"path": "C:/lab/workspace/apparatus.json", "dirty": false,
 "simulation": {"image_shape_yx": [96, 128], "grid_shape_yx": [5, 7], "seed": 0, "world_profile": ""},
 "types": [{"type_id": "camera.virtual_mot", "family": "camera", "label": "Virtual MOT camera",
            "description": "...", "capabilities": ["camera.adapter"], "physical_outputs": [], "addable": true}],
 "unavailable": [{"module": "zlc_atom.devices.camera.pylon.device_types", "reason": "No module named pypylon"}],
 "templates": ["virtual", "hardware"],
 "discovered": [{"instance_id": "dcam_0", "type_id": "camera.dcam", "role": "dcam_0", "parameters": {...}}],
 "action": "init" | "apply" | "shut_down" | null,   // what apply_devices would do now (the v2 lifecycle button)
 "busy": null | "Opening 3 devices"}
```
Each `types` row also has `family_label` (the family in words, for card titles).

`node_types` (top level, the Add node chooser): `[{"type_id", "label", "node_kind",
"publishes" (output names, "what its draft declares" or "nothing"), "description"}]`.

### 3.3 `devices` (one row per draft or live instance, in operator order)
```json
{"instance_id": "mot_camera", "role": "mot_camera", "type_id": "camera.virtual_mot", "family": "camera",
 "parameters": {...}, "controls": [Control...],   // the draft's Init settings
 "state": "draft" | "opening" | "open" | "closing" | "failed",  // live state of this instance
 "pending": "add" | "change" | "remove" | null,     // what apply_devices would do to it
 "error": null | "text",
 "claimed_by": [{"node_id": "camera_measurement", "run": "camera_measurement#2", "protected_fields": ["exposure"]}],
 "physical_outputs": [{"name": "mot_coils", "class": "coil"}],
 "commands": ["mot_camera__snap"],                 // its catalogue entries (PLUGIN_PROTOCOL §8)
 "fields": [TunableField...] | null}              // live Device Control, once read (§3.7)
```
`TunableField` = `Control` + `current` (readback), `desired`, `live_write`,
`device_limits`, `editable` (bool) and `reason` (v2: "Protected by Camera
measurement", "Stop the active Logic; this field is not live-writable", "Accept risk
to edit this unclaimed live-safe field"), `current_text`/`limits_text` (v2
format_quantity: the readable prefix of its unit, or exactly the spelling the operator
picked), `units` (the spellings a number may be read in, `[{value, label, group}]`
grouped by unit family; `read_device {units}` re-reads the row in one) and
`status`/`severity` (v2's words at rest: Ready, Protected, Stands outside its bounds).
Once read, the row also has `control: {owners (node names), reason, risk_enabled}`.

Bench fabric (PLUGIN_PROTOCOL §10): `published` (Remote on: handed to another
computer; Control, tunes, commands, close and rebuild of it are refused here, and a
node draft that uses it has the issue "… is published on the bench fabric"),
`publishable` (an open device of this computer), `remote` (`{host, port,
instance_id, publisher, role, type_id, label}` for a `remote.device` row: the
device another computer publishes; open, the row's `family`, `label`, outputs,
commands and capabilities are that type's).

### 3.4 `logic` (Task Console rows, board order)
```json
{"node_id": "occupancy", "name": "Occupancy", "type_id": "occupancy", "node_kind": "processor",
 "description": "...", "draft_revision": 12,
 "values": {...}, "controls": [Control...],          // authoring fields, problems inline (§3.7)
 "source_signal": "@logic/camera_measurement/frames",
 "source_choices": [["@logic/camera_measurement/frames", "Camera measurement / frames"]],
 "device_keys": {"camera": "mot_camera"},
 "device_choices": {"camera": [["mot_camera", "mot_camera"]]},   // value = instance_id, label = role
 "editor_facts": {...} | null, "editor_facts_error": "..." | null,  // the node's editor_facts answer for this draft
                                                      // (the bench's tunable devices offered as tunable:<key>)
 "artifact_inputs": {"calibrations": [{"path": "...", "frames": ["rest"]}]},
 "auto_preview": true,
 "device_labels": {"camera": "Camera"},             // each device argument in words
 "outputs": [{"signal": "@logic/occupancy/occupied", "contract": "occupancy.occupied", "label": "Occupancy/occupied",
              "shape": "(1) × (2) × (35)" | null}],   // shape: axis sizes per domain once published
 "can_stop": true,                                    // something to stop: a run, a follow, a queued start
 "preview_offered": false,                            // the type declares previews (auto_preview has an effect)
 "issues": ["Calibrations must be selected"],        // admission; Start is refused while any
 "can_start": false, "edited": false,                 // edited: draft differs from the running configuration
 "state": "idle" | "draft" | "running" | "warning" | "error",   // the v2 row state
 "status": "Draft: Calibrations must be selected",    // the v2 status text
 "phase": "idle" | "following" | "waiting" | "starting" | "running" | "stopping" | "removing",
 "run": RunRow | null}                                // current, else last run
```
The acquisition field is a `choice` in `controls`: labels are node names, values
are node IDs, `None` is the empty string, and unavailable held references remain visible.
Candidates match the consumer's ready/terminal requirement.
Clearing a node's Name restores its ID. Names cannot take the `signals` heading
or an existing external signal producer's name. While a Task owns the console,
all logic draft controls (including that Task's) are locked; Name stays editable
and the owning Task keeps Stop. A refused Restart leaves the current run intact
and projects a warning with `Restart refused` until the draft changes, is reverted,
or a Restart is accepted.

Start freezes the accepted values, resources, outputs, previews and claims, then
builds a candidate before stopping the previous run. Its command reply can be
`{node, phase: "preparing"}`; construction errors appear as a refused Restart.
A successful processor Restart stops its old run even if its new source has not
published yet, and waits for that source. Stop cancels checking, construction,
and queued starts. Acquisition ownership names a concrete child run; it cannot
stop a later replacement run. Run sequence numbers survive node removal and
board replacement, so late events cannot address a new run with an old ID. `ready` requires a ready event, whereas `terminal`
requires successful completion and a source derived from the selected child.

Plot Auto choices are stored as authored intent, separate from the roles resolved
for one schema. A new scan schema recomputes defaults; valid manual roles and
Scope coordinates survive, while a missing Scope coordinate returns to its new
default. Saved targets omit automatically resolved fates so reopening a layout
does not turn Auto into a fixed choice. Older explicit targets retain their
recorded roles.

An empty panel loaded from a layout already carries its saved size, available size
presets and logical display dimensions. These come from the same layout geometry
as rendered frames. Resizing and saving before Start preserve that authored size;
the first frame fills the reserved area without changing the card's dimensions.

The Plot header's Repeat counts come from the accepted immutable dataset prefix.
For each Repeat axis, the other Repeat axes and the last written Point coordinate
are held fixed. Counts measure acquired positions, even when the samples are
scientifically invalid. Point and Cell dimensions remain their declared sizes;
Scope, Facet and reductions do not change this summary. Stopped and Frozen
panels keep the counts of their own snapshots.

The parity regression coverage for these boundaries is kept with their owners:

| Boundary | Root correction | Regression coverage |
| --- | --- | --- |
| Restart and follow | Prepare before replacing; stop the previous run even when the next source is unpublished | Workbench candidate failure, unpublished source and frozen output tests; real plugin construction failure |
| Acquisition | Own concrete run IDs; cancel every pending phase; validate declared ready/terminal and source lineage | Parent/child Stop, parent Restart, replacement child, premature completion and late Ready tests |
| Draft and device replies | Apply only current tickets, including remove/recreate and selection changes away and back | Refresh, selection, lifecycle replies, device read and role Apply tests |
| File controls | Reconcile each normal Field independently | Browser check keeps another dirty path and its focus |
| Names and previews | Unicode NFKC/casefold uniqueness; first preview may come from the latest completed run | Unicode names and short-run preview tests |
| Plot configuration | Carry authored intent; resolve automatic roles against the new schema; reject new invalid edits | `restart_configuration.rs` and the v2 configuration corpus |
| Repeat summary | Store acquired coordinate counts with immutable prefixes, independently of validity | Core mapped-coordinate coverage and Plot document header tests |
| Engine release | Recheck the Rust source fingerprint after build and before staging/recording | Release script source-stamp guard; Python/UI stay outside that fingerprint |


v2 status texts are kept: `not started`, `Draft: …`, `following X (waiting for the
signal)`, `following X (source stopped)`, `failed; restarts with the source's next
run: …`, `waiting for A, B`, `restart queued: …`, suffix ` · edited`.

`RunRow`:
```json
{"run": "occupancy#2", "node_id": "occupancy", "state": "starting" | "running" | "stopping" | "completed" | "stopped" | "failed",
 "ready": true, "started_at": "2026-10-20T13:02:11.512+02:00", "ended_at": null,
 "progress": {"message": "point", "current": 3, "total": 20} | null,
 "request": {"request": "occupancy#2:1", "kind": "point-selection", "title": "...", "message": "...", "payload": {...}} | null,
 "directory": "C:/lab/workspace/data/2026_10_20/calibration-2" | null,   // tasks only
 "artifacts": [{"name": "calibration", "path": "final/calibration.npz", "role": "final", "contract_id": "...", "size_bytes": 1234}],
 "generation": "uuid" | null, "stop_reason": null, "error": null | {"type": "...", "message": "..."}}
```

### 3.5 `board`
```json
{"panels": [{"panel": "panel-1", "title": null, "name": "Camera measurement/frames", "interval_ms": 100,
             "overlay_signal": "@logic/occupancy/occupied", "overlay_choices": [["@logic/occupancy/occupied", "Occupancy/occupied"]],
             "published_outputs": {"amplitude": true, "center": false},
             "outputs": [{"name": "amplitude", "label": "amplitude", "signal": "@logic/panel-1/amplitude", "shape": "(1) × (1) × (1)", "published": true},
                         {"name": "center", "label": "center", "signal": "@logic/panel-1/center", "shape": null, "published": false}]}],
 "paused": false, "selectors": false,
 "active_task": "seamless_scan" | null,
 "frozen": {"panel-3": "Seamless scan is running: this panel's source and fates are fixed until it stops"}}
```
`board.panels` is the panel order (the order of `panels` follows it; it is the order
a layout and a saved figure use). `title` is the operator's panel name or `null` for
automatic; `name` is what the card shows (automatic = the signal spelled with node
names, v2 `<name>/<output>`). Card grid positions stay view state (v2 kept only order
and size; size is the panel target's `size`, presets `1x2 2x2 4x2 1x4 2x4 4x4 4x8 8x4
8x8`). `interval_ms` is one of 100/200/400/800. `published_outputs`: which ROI/fit
outputs the panel publishes as `@logic/<panel>/<output>` (absent = published).
`overlay_signal` (v2 Overlay, `""` = none): the judgement whose site rings the panel's
image is drawn with; `overlay_choices` are the outputs of the overlay contract
(`zlc_plot.image-point-overlay-status`) of nodes reading the same measurement as the
panel's signal (v2 `overlay_signal_groups`). The desktop reads the judgement in the same
coherent front as the image (the picture waits for its judgement while its producer runs
or once both runs are sealed; a judgement of another shot is not drawn), builds the rings from the judgement run's
`image_point_overlay_geometry` record and their colours from its data; without an
Overlay a geometry recorded with the image's own run draws passive rings. The row shows
for a panel whose surfaces are images (v2 paints_images): an Image, or a FacetGrid of
image cells, each cell with its own facet coordinate's judgements.
`frozen`: while a Task runs, panels showing its protected signals refuse
semantic/source/overlay edits; display parameters stay editable (v2 §4.5).

### 3.6 `runs`
Up to 50 runs of this session, newest first (`RunRow`).

### 3.7 Controls
Device parameters, node fields, simulation settings and request fields use the panel
`Control` shape (`apps/desktop/ui/src/protocol.ts`), produced from the descriptor's
`Field` (PLUGIN_PROTOCOL §2) and extended by: `"kind": "rows"` with `"columns":
[Control]` and `"value": [{column: value}]`; `"kind": "path"` (a resource's file, or with
`"mode": "dir"` a folder: any path, a relative one naming a path in `directory`; a
resource's offers `filter` and `refreshable`); `"kind": "pair"`; `"enabled": false` (its `enabled_when`
is off: kept, not validated); `"problem"` (this field's own admission note);
`"unit"`, `"description"`, `"required"`, `"blank_label"`.

## 4. Commands (catalogue additions)

Every command is a `zlc_bus::catalog` entry: same schema for the window and MCP.
`Access` is `read` (offered to a read-only MCP client too, §5) or `control`. The result is the settled state's
affected rows plus the named fields; `accepted` means the command was applied, a later
plugin event changes `state` again (use `wait`).

### 4.1 Queries (`read`, answered without the work queue)
| Command | Parameters | Result |
|---|---|---|
| `get_state` (extended) | `panel?`, `sections?` | state |
| `list_plugins` | `type_id?` | full type descriptors (PLUGIN_PROTOCOL §2) |
| `get_device_log` | `device`, `lines?` (≤2000) | the instance's stderr ring |
| `estimate_node` | `node` | the type's `estimate_run` answer, or `{status: "unknown", reason}` |
| `list_runs` | `node?`, `limit?` | `RunRow[]` |
| `get_run` | `run` | `RunRow` + `start` (start.json) + `record` (run.json) |
| `read_layout` | `path` | what `load_layout` would produce (nodes, panels, notes), changing nothing |
| `list_recipes` | `name?` | v2 `zlc.experiment-recipe` list or one recipe with defaults: the workbench's bundled recipes (`crates/zlc-workbench/recipes`, v2's trap_lifetime), a workspace `recipes/` one of the same name replacing it |

### 4.2 Apparatus and devices (v2 Device Manager + Device Control)
| Command | Parameters | Access | v2 action → effect |
|---|---|---|---|
| `add_device` | `type_id`, `instance_id?`, `role?`, `parameters?` | control | Add: a draft row at the type's defaults. |
| `remove_device` | `device` | control | Remove: draft only; the live instance closes at `apply_devices`. |
| `edit_device` | `device`, `role?`, `type_id?`, `parameters?` (partial) | control | Rename role / change type (parameters reset to the new type's defaults) / edit settings: draft only. |
| `edit_simulation` | `values` (partial) | control | Simulation world settings of the draft. |
| `use_template` | `template` (`virtual`/`hardware`) | control | New from template: replaces the draft. |
| `discover_devices` | — | control | Scan hardware: every type with a scan (`types[].discoverable`) at once, in its own plugin process (PLUGIN_PROTOCOL `discover`), one 20 s deadline from the scan's start; fills `apparatus.discovered` (with each find's `type_id`) and answers `{found, message, severity}` once every family answered or the deadline passed (`discovered N devices; <families>: <reason>`, `no answer within 20s`, v2's words); a family's answer after the deadline is dropped. `apparatus.busy` says Scanning hardware; while the scan runs, or a family past the deadline still answers, `apply_devices` and another scan are refused (`hardware scan still running: <families>`). |
| `add_discovered` | `instance_id` | control | Copies a discovered entry into the draft. |
| `load_apparatus` | `path?` | control | Replaces the draft from a `zlc.installation` file, entry by entry: an entry v2 would refuse is left out (`skipped`, with why) and the rest load; dropped settings are `notes`. The Devices@ line says both. |
| `save_apparatus` | `path?`, `replace_unread?` | control | Writes the draft (atomic, strict). A file the draft was not read from in full (it could not be read, or entries or settings were left out: `apparatus.unread` says why) is replaced only with `replace_unread` (the window's Save goes through Save as). |
| `apply_devices` | — | control | Init / Apply: opens what the draft adds or changes (in parallel, admitted in operator order), closes what it removes; runs that claim a closing device are stopped first. Partial failures stay per instance. |
| `close_device` | `device` | control | Close one live instance (its claiming runs are stopped first); the draft keeps it, so `apply_devices` reopens it. |
| `shut_down_devices` | — | control | Shut down: stops every run, closes every instance. Never refused. |
| `read_device` | `device`, `units?` | read | Device Control open/Refresh: reads `fields` (PLUGIN_PROTOCOL ops `fields`, `values`); `units` {field: spelling} reads those fields in the operator's unit from now on (the device converts: an RF level depends on its load). |
| `tune_device` | `device`, `field`, `value`, `unit?` | control | Device Control Apply of one field (`value` in `unit`, else the field's own); the reply holds the readback in that unit. Refused when protected by a claim, or claimed and not `live_write`, or claimed without the operator's risk acceptance. |
| `accept_device_risk` | `device`, `accepted` | control | Device Control's Accept risk: allow tuning unprotected live-safe fields of a device a run claims (bound to that claim). |
| `publish_device` | `device`, `published` | control | Remote on (v2's switch): hand an open device to the other bench computers, or take it back. Refused for a device a run holds and for a `remote.device` row (a relay loop). Shut down withdraws first. |
| `<instance>__<command>` | the command's fields | the declared `read_only`, else control | Device commands (PLUGIN_PROTOCOL §8); entries appear while the instance is open. |

### 4.3 Logic (v2 Task Console rows and editors)
| Command | Parameters | Access | v2 action → effect |
|---|---|---|---|
| `add_node` | `type_id`, `node_id?`, `name?`, `values?`, `source_signal?`, `device_keys?`, `artifact_inputs?` | control | Add logic: a stopped draft (devices default to the first compatible instance). Its plugin instance opens (PLUGIN_PROTOCOL §3) to answer draft queries. |
| `rename_node` | `node`, `name` | control | Rename. |
| `edit_node` | `node`, `values?` (partial), `source_signal?`, `device_keys?` (partial), `artifact_inputs?` (partial), `auto_preview?`, `expected_draft_revision?` | control | Edit fields / connect a source / pick devices / choose files / auto-preview. Each value is checked against its field first (unknown field or invalid value: refused); completeness and cross-field admission become `issues`. Allowed while running: the run keeps its configuration, the row shows `edited`. |
| `check_node` | `node` | control | Validate / Refresh files: re-reads the chosen files and re-runs admission. |
| `revert_node` | `node` | control | Revert: the draft becomes the running configuration again. |
| `start_node` | `node`, `expected_draft_revision?` | control | Start, or Restart when running: admission (`issues` empty, devices open and unclaimed by other kinds of lease); claims; a Task gets its run directory and `start.json`; `run.start` to the instance. A processor whose source is not published follows it (starts when it publishes, restarts on its next run). With `auto_preview`, declared previews get panels when their signals first publish. |
| `stop_node` | `node` | control | Stop: cancels a following processor, a waiting start and an operator request; the run drains, seals, writes `run.json`; claims release at terminal. Never refused. |
| `remove_node` | `node` | control | Remove: a running node is stopped and removed at its terminal (`phase: removing`). |
| `answer_request` | `node`, `request`, `response` (object, or null to cancel = Stop) | control | Operator input (`point-selection`, `manual-axis`, any kind a type declares): refused unless `request` is current; a `manual-axis` (a hand moves the knob) is answered only at the window, a client may only cancel it (v2). A request made after Stop is answered as the Stop and never shown. A `point-selection` payload names `output_name` (the image), `points` (`[id, label, x, y]`, x/y on the image's axes) and `confirm_label`; its answer is `{excluded_point_ids}`. The window reviews it in its own window (`review:<node>`, titled with the request, the other windows waiting) on `review-<node>`, an Image panel held on the image's read with the candidates drawn as the review marks them (window commands `open_point_review {node, request}`, `point_review_state {panel, excluded, selected}`, `close_point_review {node}`; closing the window answers null). |

### 4.4 Board and layout (v2 Task Console board)
| Command | Parameters | Access | v2 action → effect |
|---|---|---|---|
| `order_panels` | `order` (panel ids) | control | Drag cards: missing ids keep their relative order after the given ones. |
| `rename_panel` | `panel`, `title` (null = automatic) | control | Panel name. |
| `set_panel_interval` | `panel`, `interval_ms` | control | Update interval (100/200/400/800). |
| `set_panel_overlay` | `panel`, `signal` (one of `overlay_choices`; null/"" = none) | control | v2 Overlay; refused for a Task-frozen panel. |
| `publish_panel_outputs` | `panel`, `outputs` ({name: bool}) | control | Panel publisher row: which ROI/fit outputs it publishes; refused for a frozen panel. |
| `set_board` | `paused?`, `selectors?` | control | Pause live display; selectors (plot gestures) on/off. |
| `save_layout` | `path?` | control | Save layout (`zlc.console-board`, §7.1). Only stopped drafts are written. |
| `load_layout` | `path?`, `stop_running?` | control | Load: whole-board replace; refused while anything runs unless `stop_running`, which stops it and replaces the board once it has stopped (`replaced: false`, `waiting_for`; v2 _replace_board). |
| `clear_board` | `stop_running?` | control | Clear: same rule, empty board. |
| `run_recipe` | `name`, `parameters?`, `start?` (default true) | control | v2 `lab_run_recipe`: loads the recipe's layout with its parameters (refused while anything runs), starts followers then the entry node, their runs acting for the client that ran it; the reply names the new panel ids. |

`path?` may be omitted only by the window (file dialog). Existing panel commands keep
their schema except that `add_panel`'s `signal` is optional (v2 adds a blank panel whose
signal is chosen in its Setting); `configure` is refused for the semantic part of a
`board.frozen` panel. `start_camera`/`stop_camera` are aliases: they put the Virtual MOT
camera in the apparatus (device `mot_camera`), apply it, and start/stop a
`camera_measurement` node; the node commands do the same directly.

Panel Edit (v2 PanelEditorView: an exact revision held still to edit, fit and save):
`edit_panel {panel}` opens the board panel's Edit, `edit-<panel>`, a panel of the live
document on the read the panel had then (and its Overlay's judgement); a second
`edit_panel` only names it. `refresh_panel_edit {panel}` (Refresh snapshot) holds the
panel's newest read instead; `close_panel {panel: "edit-<panel>"}` closes it, and
closing the panel closes its Edit. The two share one configuration: what an operator
changes on either (semantic, parameters, fit, size, selectors, viewport, focus,
thresholds) is given to the other; the Edit's binding is fixed, the panel's Task
protection holds for it, and it publishes no ROI or fit outputs. The Edit's state row
carries `edit: {panel, at_ms, revision, advanced}` (`advanced`: the panel's signal has
newer data); it is not on `board.panels`. Its page's Save figure (v2's group) is
`save {panel: "edit-<panel>", path}`: the held picture and its archive, the state's
`saved` naming both.

### 4.5 `wait` (extended)
`wait {node, until: "ready" | "terminal" | "request", timeout_seconds}` answers when
that node's run reaches it (immediately if it already has) with its `RunRow`; a run
that ends before `ready` answers too. `produced`/`panel` waits are unchanged.

## 5. Who may change things (v2's model)

- The operator at the window: everything, with no arming step (Init/Apply, tune,
  On Pulse, pulse play work as in v2).
- MCP/AI clients: the loopback endpoint with its token and Origin check, plus v2's one
  switch `allow_control`: on by default, off when the instrument starts with
  `--mcp-read-only` (recorded as `control_enabled` in `mcp.json`). With it off a client
  is offered, and may call, only the `read` commands and the device commands declared
  `read_only`; with it on it drives the instrument as the operator does.
- What protects a running experiment is the run's claims: owners and protected fields
  (`tune_device`, Device Control's Accept risk), and the claims check on a run's
  `pulse_*` commands.
- A plugin crash is that device's error state (v2): the runs that claim it stop with the
  error, and `apply_devices` reopens it.
- `stop_all` stops every run and drives every registered output safe: the pulse board's
  SAFE and each open device's `physical_outputs` (PLUGIN_PROTOCOL §7). Never refused.

## 6. Plugin host interface (B ↔ A)

The workbench consumes descriptors and never branches on a concrete type. It drives
the host through one Rust trait (`zlc_workbench::Plugins`), implemented in the
desktop over line A's `zlc-plugin-host` (`crates/zlc-desktop/src/workbench_host.rs`:
each call runs on a worker thread; answers and host events come back as workbench
events on the window's turn):

```rust
pub trait Plugins {
    fn catalog(&mut self) -> TypeCatalog;                                           // describe result: types + errors
    fn open(&mut self, instance: &str, type_id: &str, parameters: &Value, context: &Value); // device or node instance;
                                                                                    // context.simulation = group parameters
    fn close(&mut self, instance: &str);
    fn query(&mut self, instance: &str, ticket: u64, name: &str, args: &Value);    // declare_outputs, resolve_field_availability
    fn call(&mut self, instance: &str, ticket: u64, op: &str, params: &Value);     // fields, values, tune, command
    fn start(&mut self, run: &RunSpec);
    fn stop(&mut self, run: &str);
    fn answer(&mut self, run: &str, request: &str, response: Option<&Value>);     // operator input (None cancels)
    fn acquired(&mut self, run: &str, ticket: u64, result: Result<Value, String>); // reply to RunAcquire
}
```
The host's application requests map as: `run.operator_input` → `RunRequest`
(answered by `answer_request`), `run.artifact` → registered only when the file exists
inside the run directory, `run.acquire` → `RunAcquire`, `core {command, params}` → that
catalogue command, sent exactly as a remote client's call under the run's origin, its
result answering the run; the command arrives with
`_run` (the asking run id) beside its parameters, and `Workbench::run_claims(run)`
names the devices that run holds. Device instances open when the
apparatus is applied and stay open across runs; a run claims them by `RunSpec.devices`.
`RunSpec`: `run`, `node_id`, `type_id`, `values` (enabled fields), `source_signal`,
`devices` (argument → instance_id), `artifacts` (name → path or frame claims),
`resources`, `directory` (tasks), `outputs` (output name → signal name), `origin`.
Events into the workbench: `Opened{instance, working_point}`, `Failed{instance,
error}`, `Closed{instance, error}`, `Answered{ticket, result}`, `RunReady{run}`,
`RunProgress{run, message, current, total}`, `RunRequest{run, request}`,
`RunArtifact{run, artifact}`, `RunAcquire{run, ticket, node, completion}`,
`RunFinished{run, outcome, error, generation}`, `Log{instance, line}`.
`RunAcquire` is v2's `restart_logic` (Seamless/feedback starting their acquisition
node): the workbench restarts that node with its running configuration and answers
`ticket` at `ready` or `terminal`. Data never crosses this trait: plugins publish into
the Runtime through A's data plane under the signal names the RunSpec gives.

## 7. Persistence (v2 formats, read and written as v2)

### 7.1 Layout `zlc.console-board`
```json
{"format": "zlc.console-board",
 "panels": [{"panel_id": "panel-1", "signal": "@logic/camera_measurement/frames", "title": "...", "kind": "facet_grid",
             "cell_kind": "curve", "size": "2x2", "interval_ms": 100, "semantic": {...}, "display": {...},
             "interaction": {...}, "fit": {...}, "overlay_signal": "", "published_outputs": {...},
             "selector": {...}, "crosshair": {...}, "classifier_thresholds": [], "focused_cell": null}],
 "logic": [{"node_id": "camera_measurement", "name": "camera_measurement", "api_name": "camera_measurement",
            "values": {...}, "source_signal": "", "device_keys": {"camera": "mot_camera"},
            "artifact_inputs": {}, "auto_preview": true}]}
```
Load = v2 `resolve_layout`: wrong `format` or unknown `api_name` refuses the file;
values are intersected with today's fields (unknown ignored, missing at default, a
refused value reset with a note); a device that no longer provides the capability or a
source of another contract keeps the row with a note (Start refuses it); node ids are
kept; panel ids are minted fresh and every `@logic/<saved panel>/…` respelled; a
reference to a panel the file lacks is dropped with a note. Panel fields map to the v3
panel target: `semantic`→`semantic`, `display`→`parameters` (`window` → the binding
window), `fit`, `selector`+`crosshair`→`selectors`, `focused_cell`→`facet_focus`,
`classifier_thresholds`, `size`; `title`, `interval_ms`, `interaction`,
`overlay_signal` (respelled like `signal`), `published_outputs` are kept and written back.

### 7.2 Apparatus `zlc.installation` (`<workspace>/apparatus.json`)
`{"format": "zlc.installation", "simulation": {...}, "devices": [{"instance_id",
"role", "type_id", "parameters"}]}`, exactly these keys; strict JSON; written only by
`save_apparatus`. An unknown `type_id` is kept and shown failed with the import
reason; an unknown parameter is ignored with a note (v2 refused the whole file). Such a
file is `unread`: the draft shows unsaved and Save does not write back fewer devices or
settings than it holds. A device type may declare `identity` (`visa:{resource}`): of
two rows naming one instrument the later is refused before it opens (v2's broker). A
device and a logic node may share an id: a node's plugin instance is `node/<id>`.

### 7.3 Task runs (`zlc.task-start` / `zlc.task-run`)
`<data>/<YYYY_MM_DD>/<api_name>[-n]/`, allocated only at a real Start. `start.json`
once, before any irreversible work: `{"schema": "zlc.task-start", "run_id", "task":
{"api_name", "instance_id"}, "input": {"authored", "source_signal", "devices": {arg:
{"instance_id", "capability"}}, "artifacts", "resources"}, "started_at"}`. `run.json`
once at terminal: `{"schema": "zlc.task-run", "run_id", "task", "input", "status":
{"state", "started_at", "ended_at", "progress", "stop_reason"}, "artifacts": [{"name",
"path", "role", "contract_id", "size_bytes"}], "error"}`. Nothing in between; a
crash leaves only `start.json`. Measurements and processors have no run directory.

## 8. v2 design mistakes fixed here

1. **Composition lived in Qt presenters** (`console.py` 10.5k lines, `DeviceManagerPresenter`,
   `ExperimentGuiFlow`): run orchestration (restart queueing, follow intent,
   acquisition parent/child, task takeover, device preemption), apparatus reconcile
   and Device Control state ran on the GUI thread, spread over three owners. Now one
   Rust model; views only render state.
2. **MCP was a second path** (`control_action`, `control_device`, string dispatch,
   developer reflection, 34 tools with gaps: no device add/remove, no
   `published_outputs`, no request answers). Now the window and MCP send the same
   schema'd commands; nothing is UI-only except file dialogs.
3. **One button inferred its command** (Init/Apply/Shut down/Refresh from derived
   state): explicit `apply_devices`/`shut_down_devices`, with `apparatus.action`
   telling the view what Apply would do.
4. **Hard-coded device keys** (`"sequencer"`, first `camera.adapter`): claims are by
   instance id through device fields only.
5. **Five overlapping lifecycles** (NodeHost phase, TaskRun state, plane generation,
   console row pseudo-states, following): one run state machine (`RunRow.state`) plus
   the row's `phase`; the row `state`/`status` are projections of them.
6. **Undeclared bench extras** (`restart_logic`, `tunable_devices`, ...) injected by
   parameter name: acquisition is a declared host event (`RunAcquire`); device access
   goes through claimed proxies.
7. **Request kinds hard-coded in the console**: any kind is shown from its declared
   payload; the response is the request's fields.
8. **Stale outputs after Stop** (`_logic_outputs` returned the last host's
   declarations): outputs always come from the current draft's `declare_outputs`.
9. **Inconsistent load tolerance** (unknown device type kept, unknown parameter refused
   the file): both are kept/ignored with a note.
10. **Two spellings for derived signals** in v3 (`panel-1/roi/frame`) against v2's
    `@logic/panel-1/roi_frame`: one grammar, owned here.

## 9. v2 action → command

| v2 (Qt signal / method / MCP) | v3 command |
|---|---|
| Device Manager add / remove / role / type / parameters / template / scan / add discovered | `add_device` / `remove_device` / `edit_device` / `use_template` / `discover_devices` / `add_discovered` |
| Device Manager load / save (`lab_devices`) | `load_apparatus` / `save_apparatus` |
| `toggle_lifecycle` Init / Apply (`lab_initialize`) / Shut down (`lab_close session`) | `apply_devices` / `shut_down_devices` |
| `close_device` | `close_device` |
| Device Control open / refresh / apply / risk / unit (`lab_device_action`) | `read_device` / `tune_device` / `accept_device_risk` (unit: `read_device {units}`) |
| Device log | `get_device_log` |
| Remote publication toggle / the other machine's Scan hardware | `publish_device` / `discover_devices` (`remote.device` rows, crates/zlc-fabric) |
| `add_logic` / `rename_logic` / `update_logic_draft` (`logic.add/rename/update`) | `add_node` / `rename_node` / `edit_node` |
| `refresh_logic_files` (`logic.validate`) / `revert_logic_draft` (`logic.revert`) | `check_node` / `revert_node` |
| `start_logic` / `stop_logic` / `stop_active_task` / `remove_logic` | `start_node` / `stop_node` / `stop_node` / `remove_node` |
| `set_logic_auto_preview` | `edit_node {auto_preview}` |
| point review / manual axis (`lab_review_points`) | `answer_request` |
| `control_estimate` (`lab_estimate`) | `estimate_node` |
| `reorder_panels` / title edit / interval / `set_paused` / `set_deriving` | `order_panels` / `rename_panel` / `set_panel_interval` / `set_board` |
| Overlay (`overlay_signal` of the panel record) | `set_panel_overlay` |
| `update_panel_published_outputs` (panel publisher row) | `publish_panel_outputs` |
| `save_screenshot` (window image, `lab_screenshot`) | `save_image {path?, window?}` (that window's page as composed on screen; the window asks for the file, a remote client's goes to today's folder and its reply carries the PNG); `get_panel_image` for one panel |
| `save_layout` / `load_layout` / `clear_board` / `lab_run_recipe` / `lab_recipes` | same names / `run_recipe` / `list_recipes` |
| `lab_state` / `lab_catalog` / `lab_wait` | `get_state` / `list_plugins` / `wait {node, until}` |
| `lab_results` / `lab_screenshot` | `get_fit` / `get_panel_image` (existing) |
| `edit_panel` / `refresh_panel_snapshot` / `save_panel_figure` | later: `edit_panel` / `refresh_panel`; `save` (existing) |

## 10. Alignment with PLUGIN_PROTOCOL v1 (all settled in PLUGIN_PROTOCOL a21083b)

1. Output signal names come from `RunSpec.outputs` (`@logic/<node_id>/<output>`), so v2
   layouts and panel references load unchanged; `run.begin` uses those names.
2. Node ids are v2's `<api_name><n>` under v2's name rule; no other grammar applies
   to node or device ids. A layout node whose id a device also has gets a fresh id on
   load, with a note (devices and nodes share the plugin host's instance names).
3. Node descriptors also need v2's `artifact_inputs` (`name`, `label`, `contract_id`,
   `suffixes`, `required`, `frame_claims`), `previews` (or the `declare_previews`
   query), `acquisition_input` + `acquisition_completion`, `artifact_outputs`,
   `description`, and per-requirement `enabled_when`.
4. A run asks for its acquisition node with an event (`run.acquire {node,
   completion}` → workbench `RunAcquire`), replacing v2's injected `restart_logic`.
5. Claims are decided by the workbench (a requirement takes its device exclusively, a
   device the draft names is shared, v2's rule; Device Control tunes are refused on
   protected fields and the fields their dependency groups couple to); the host routes `device` calls to
   the instances `RunSpec.devices` names. Besides the requirements, a draft may name
   devices itself (`resolve_device_claims`: a scan's device axes): each is claimed with
   its named fields protected and passed as its own argument (`tunable:<key>`).

## 11. Where it lives

`crates/zlc-workbench`: the model, admission, layout/apparatus/run-directory codecs,
the `Plugins` trait; pure Rust, tested without a window. `crates/zlc-bus`: the
catalogue entries above, and the MCP endpoint, which also lists each open device's
`<instance>__<command>` tools and announces changes on its `GET /mcp` event stream.
`crates/zlc-desktop`: routes these commands to the workbench on its event loop
(`main.rs`), drives the plugin host and forwards its facts (`workbench_host.rs`),
and adds the state sections to every `state` message.

Browser sharing is the application's (apps/desktop/README.md).
