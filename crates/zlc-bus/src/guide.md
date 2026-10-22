# ZLC operator guide (the `guide` command; one `## topic` per section)

## overview

ZLC already holds the whole chain: devices → acquisition → processing nodes → plot/fit panels → saved
Figures. You drive it with the same commands the window sends; `tools/list` gives each one's schema.

Read first, without devices: `list_plugins` (every device and node type: purpose, fields with choices and
units, inputs, outputs, device needs; `{type_id}` for one), `plot_catalog` (plot kinds, axis roles,
reductions, fit models), `list_recipes`, and the workspace's `EXPERIMENT_CONTEXT.md` (`read_file`) when it
exists. `get_state` is the whole instrument; `{sections: [...]}` or `{node}`/`{panel}` keeps it small.

Names. A device has an `instance_id` (its key everywhere) and a `role` (its label). A logic node has a
`node_id` that never changes and a `name` the operator may change at any time: address nodes by `node_id`;
when the operator names a node, find it in `get_state` `logic[].name`. Signals are
`@logic/<node_id>/<output>`; a panel's ROI and fit outputs are `@logic/<panel>/<output>`. Runs are
`<node_id>#<n>`.

Replies. A command's reply means it was applied. A run, a device opening or a Fire finishes later: read it
with `wait {node, until}`, `wait_for` (a field of a read command equals a value) or `get_state`. When a reply
is lost or uncertain, read the state before doing anything again; never repeat a Start, Fire or Apply on a
guess. `batch` runs steps you have already decided, in order, and stops at the first refusal.

Topics: setup, devices, acquisition, pulse, calibration, processing, scan, plot, fit, files, recipes,
lifetime, slm, errors; `type:<type_id>` is one plugin type's descriptor.

## setup

The workspace is one folder: `apparatus.json` (the device draft), `pulses/`, `layouts/` (Task Console
layouts), `data/<YYYY_MM_DD>/` (runs, saved figures, screenshots), `recipes/`. `get_state {sections:
["workspace"]}` names them.

1. Devices: `load_apparatus {path}` or `add_device`, then `apply_devices` (Init). See devices.
2. Nodes: `add_node {type_id, values, source_signal, device_keys, artifact_inputs}` adds a stopped draft.
   Its row in `get_state logic` has `controls` (each field with its choices, unit and problem),
   `source_choices`, `device_choices`, `issues` and `can_start`. `edit_node` changes the draft;
   `start_node` starts it once `issues` is empty.
3. Panels: `add_panel {signal, kind}` and `configure`. See plot and fit.

`save_layout {path: "layouts/<name>.json"}` keeps the board (node drafts and panels); `load_layout {path}` brings it back
(`read_layout` shows what it would load). A row whose device or source no longer fits loads with a note and
an issue that says what to choose.

`shut_down_devices` stops every run and closes every device. `close_experiment` closes the application;
it is refused while the Pulse Editor holds unsaved changes unless `discard_changes` (or `pulse_save` first).

## devices

`add_device {type_id, instance_id?, role?, parameters?}` adds a draft row (its Init settings in
`parameters`; the row's `controls` list them). `apply_devices` opens what the draft adds or changes and
closes what it removes; `apparatus.action` says whether it would Init, Apply or Shut down. Opening is
asynchronous: wait for `devices[i].state` to be `open` (or `failed`, with `error`), e.g.
`wait_for {tool: "get_state", arguments: {sections: ["devices"]}, path: ["devices", 0, "state"], equals: "open"}`.
A failed device does not stop the others; the operator may keep rows that cannot open here on purpose.
`use_template {template: "virtual"}` fills the draft with the simulated bench.

Device Control: `read_device {device}` reads `devices[i].fields` (readback, limits, `editable` and the
`reason` when not). `tune_device {device, field, value, unit?}` applies one field; its reply holds the
readback. A field a running node protects is refused; an unprotected live-safe field of a held device needs
the operator's `accept_device_risk`. `read_device {units: {field: "MHz"}}` reads a field in another unit.
Commands a device type declares appear as `<instance>__<command>` tools while it is open
(an id outside letters, digits and `_` is spelled `<letters>-<hash>`; `get_state` devices list each row's tool names).
`get_device_log {device}` is its plugin's log.

## acquisition

`camera_measurement` publishes `@logic/<node>/frames`: whole cycles of `frames_per_cycle` frames, `repeat`
cycles (0 = until stopped). A blank exposure or ROI field keeps the camera's current value for that field.
It arms the camera and reports `ready`; the triggers come from the Pulse. For a finite manual capture, open
the pulse in the Pulse Editor first, then in one batch: `start_node {node}`, `wait {node, until: "ready"}`,
`pulse_fire`. Then `wait {node, until: "terminal"}` and `pulse_stop` if the pulse repeats forever.

Calibration and Seamless Scan run their own acquisition and Pulse: start them once, not the camera by hand.
Processors (Occupancy, Frame Survival, Derive, ...) can start before their source publishes: they follow
it (`phase: following`) and run on each new acquisition. `waveform_measurement` does the same for
waveform devices; read the source's actual axes in `list_signals`.

## pulse

Two ways to play a pulse document (`zlc.pulse` JSON in `pulses/`):

- Directly on the board: `pulse_describe {document}` lists its API fields in play order (field_id, label,
  unit, value), scan fields and periods; `pulse_run {document, api_values: {field_id: [value, unit]},
  run_repeats}` compiles, loads and fires, answering `command_id`; `pulse_wait_done {command_id,
  timeout_seconds}` reports it once (done, status); `pulse_stop` makes the board safe. `pulse_estimate`
  times a scan plan. While a run holds the sequencer, only that run plays on it.
- The Pulse Editor (what the operator sees): `pulse_open {path}`, `pulse_state` (periods, ports, bindings,
  scan table, Config, board), `pulse_edit`, `pulse_scan`, `pulse_config`, `pulse_subpulse`, `pulse_target`,
  `pulse_validate`, `pulse_save`; `pulse_fire` plays what it shows with its authored repeats; `pulse_hold` /
  `pulse_step` hold one scan point.

`run_repeats` 0 plays until `pulse_stop`. Validating or previewing loads nothing. `pulse_stop` and
`stop_all` are never refused.

## calibration

`calibration` captures (or rereads) frames, finds the trap sites and writes a calibration file with its
report figures in the run directory (`logic[].run.directory`, `.artifacts`). Fill its camera, sequencer,
pulse and readout fields from its controls, then start it once; it plays its own pulse. To calibrate again
from saved frames, set `frame_source` to `saved` and `saved_frames_path` to the `figures` folder of a run
that saved frames: it then uses no camera or sequencer.

During site review the run carries `request` (`kind: "point-selection"`, payload `points` as `[id, label,
x, y]`, `wait {node, until: "request"}` returns then). Look at the review picture (`get_window_image` of
its window, named in `get_state windows`, or the run's figures) and answer
`answer_request {node, request, response: {excluded_point_ids: [...]}}` ([] keeps every site);
`response: null` stops the run. A `manual-axis` request asks the operator to move something by hand: answer
it once the operator says it is done.

## processing

Occupancy (`occupancy`) reads camera frames with calibrations and publishes `counts`, `occupied` and
`frame_judged`. Its `calibrations` artifact input is a list of rows `{path, frames}`: `["rest"]` reads every
frame; frames taken at another exposure get their own row, e.g. `[{"path": LOAD, "frames": ["first"]},
{"path": READOUT, "frames": ["rest"]}]`. Frame words: numbers from 0, `first`, `last`, `rest`. A row with
`frames: []` is kept and not read. The row's `editor_facts` say which frames each file reads. With
`post_selection: true` (off by default) counts and occupied are valid only where a condition on two frames
holds (`mode: relation` with `frame_a`, `frame_b`, `relation`; or `mode: values` with `value_a`/`value_b`
true, false or any).

Frame Survival (`frame_survival`) reads `occupied` and publishes the later verdict where the earlier frame
was occupied and both were judged: its mean over trials is conditional survival, per site. `selection`:
`all`, `consecutive`, `from_first`, `to_last` (resolved against the frames the source carries), `none`, or
`custom` with `custom_pairs: [{start, end, selected}]`; ends are `0`/`first`, `last` or a middle number.

Derive (`derive`) is algebra on one producer's bundle. `values.variables` is a list of rows `{name, terms,
statistic?, over?, publish?}`; a term is `{of}` (a member or another variable), `{number}` or `{boolean}`,
optionally with a pick `at` (an axis or a domain) and `labels` (a list; `first`/`last` follow the axis);
every term after the first has `op` (`+ - * /`, comparisons, `and`, `or`, `and not`, `where`, `where not`).
Example: `{"name": "survived", "terms": [{"of": "occupied", "at": "frame", "labels": ["last"]},
{"op": "where", "of": "occupied", "at": "frame", "labels": ["0"]}], "statistic": "mean", "over": ["Site"]}`.
The row's `editor_facts` list the members, axes and labels; issues read `<name> · term k: <reason>`.

## scan

Seamless Scan (`seamless_scan`) plays a pulse over a scan plan, restarting its acquisition node itself, and
publishes `@logic/<node>/scan` from its source signal. `plan` is JSON text `{"axes": [{"port": ...,
"values": [...], "unit": ...}]}`; the ports (`pulse:param:<field_id>`, `api:<field_id>`, device fields) are
in the row's `editor_facts`. Every listed axis is scanned. `repeats` are complete sweeps;
`shots_per_point` repeat one point before the next. `estimate_node {node}` gives the plan's Pulse time
before Start. Configure the result panel and its fit before Start: the first sweep already gives a curve.

## plot

`add_panel {signal, kind}` (`image`, `curve`, `histogram`, `rolling`, `facet_grid`; a blank panel without
`signal`, bound later with `configure {binding: {signal}}`); its reply's `added_panels` names it. Its row in
`get_state panels` has `controls`:
section `semantic` (`kind`, `cell_kind`, `reduction`, and one `fate:<domain>:<axis>` per data axis),
`display` (colormap, limits, bins, uncertainty, ...). `configure {panel, semantic: {...}, parameters: {...},
size, binding: {window}}` sets them with exactly the values the controls list.

Axis roles: `x` for the scanned coordinate, `facet` for independent cells such as sites, `group` for
overlaid series, `reduce` (with `reduction: "mean"`) for repeated trials, a scope pin for one coordinate
such as a frame pair. Per-site scan curves: `facet_grid` with `cell_kind: "curve"`, the scan axis `x`, site
`facet`, repeat axes `reduce`, the pair scoped.

Selectors (ROI, range, threshold, crosshair): `configure {panel, selectors: [{kind: "area", value: {x: [x0,
x1], y: [y0, y1]}, facet_index: null}]}`; an image's area publishes `@logic/<panel>/roi_frame`.
`set_panel_overlay` draws Occupancy's site rings on an image. `rename_panel`, `set_panel_interval`,
`order_panels`, `close_panel`. `get_panel_image {panel}` is the picture; `get_window_image` the whole window.

## fit

`configure {panel, fit: {model, fixed?, initial?, bounds?, fit_all_facets?}}` with a model from the
panel's `fit.models` (`plot_catalog {details: true}` gives formulas and parameters). A fixed value is not
optimised. In a facet grid, `fit_all_facets: true` fits every cell. Live fits follow new data.

`get_fit {panel}`: `status` (`current` matches the display; `lagging` rows are earlier but usable; `pending`,
`not_enabled`), `parameter_names`/`parameter_units`, `sample_axes`, and per cell `coordinates`, `values`,
`standard_errors`, `valid`, `error_valid`, `fixed_parameters`, `message`; paged by `offset`/`limit`. The
fit's parameters are published as signals (`board.panels[i].outputs`, `publish_panel_outputs`), each with
its uncertainty, so another panel can plot e.g. `decay_time` per site.

## files

`list_files {directory}` and `read_file {path}` read the workspace (relative paths are from its root): text
a page of bytes at a time (`offset`, `next_offset`), PNG/JPEG as an image, a Figure archive (`.npz`) as its
datasets and sections (`dataset`, `plot` with the recipe and fit, `source`, `lineage`; `section` picks one).
Relative paths in other commands are from
the workspace root too, except resource fields (a pulse), which name a file in their folder (`pulses/`).

`save {panel, path: "data/.../result.png"}` writes the panel as shown (a PNG, or a PDF/SVG by the suffix) and its Figure archive
(`.npz`); the reply's `saved` lists both. `save` on `edit-<panel>` (`edit_panel`, `refresh_panel_edit`) writes its held
snapshot. `open {path}` shows an archive in the Figure Viewer (`get_state viewer`). `read_signal {signal}`
reads a published signal's numbers with validity and axes, a page at a time; `dataset: true` reads a finite
run's dataset or a monitor's retained history.

## recipes

A recipe is an ordinary layout plus named parameters. `list_recipes` lists them (workspace `recipes/`
replaces a bundled one of the same name); `list_recipes {name}` gives its parameters, defaults,
requirements, completion and save notes. `run_recipe {name, parameters, start}` loads it (refused while
nodes run), starts its followers and then its entry node; `start: false` only prepares. The reply names the
result panels. Nothing is initialized, sent or calibrated for you.

## lifetime

Recipe `trap_lifetime`: Camera (two-frame cycles) → Occupancy → Frame Survival (0 → last) → Seamless Scan
→ a facet grid of per-site curves with an Exponential fit (`decay_time`). Its parameters: `camera`,
`sequencer`, `pulse_path`, `calibrations`, `scan_plan` (one hold-time axis), `panel_semantic` (copy
`defaults.panel_semantic` and add `"fate:point:scan.<FIELD_ID>": "x"` for the plan's API field
`<FIELD_ID>`), `repeats`, `shots_per_point`, `frames_per_cycle`, `start_frame`, `end_frame`, `fit_fixed`.
Read `pulse_describe` for the real field id. While it runs, `get_fit {panel}` already has partial rows;
`save` the panels and `save_layout` without stopping the scan.

## slm

SLM devices (`slm.virtual`, the Hamamatsu X15213) are driven by the SLM nodes: `slm_feedback` (camera
feedback onto a Science Context), `slm_rearrangement` (GPU solver, motion plan, movie playback, report),
`rearrangement_measurement`. Their calibration and Science Context are artifact inputs. The SLM nodes run
on the plugin interpreter given at launch; the GPU solver needs CuPy there. A virtual run's success is not
real atom survival.

## errors

A refused command answers an error saying why; nothing changed. Unknown fields or values: read that node's
`controls` or that panel's controls. Issues and run errors name fields by their labels on screen.

A node's row: `phase` (idle, following, waiting, starting, running, stopping, removing), `state`/`status`
(the window's words), `issues` (what blocks Start), `run` (state, ready, progress, request, error,
directory, artifacts). `start_node` on a running node restarts it; `edit_node` while running edits only the
draft (`edited` is true), `revert_node` puts the running configuration back. `stop_node` and `stop_all` are
never refused.

`wait {node, until, run?}` answers with `reached` and the run row; a request answers a ready or terminal wait
early (`reached: false`). `timed_out` only ends the watching, never the run. `batch` stops at an error, a
refusal, an unmet `expect`, a timed-out or unmatched wait, or a run that stopped or failed; what ran stays
done. `health` answers even while the instrument is busy.
