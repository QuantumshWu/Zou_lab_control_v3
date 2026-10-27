# MCP: v2 tools and their v3 equivalents (line K)

v2 reference: `packages/zlc_workbench/src/zlc_workbench/apps/mcp.py` (the endpoint and the stdio proxy) and
`guides/mcp.md` (the `lab_guide` topics). v3: every tool is a catalogue command (`crates/zlc-bus/src/catalog.rs`)
with a JSON Schema, a read/control class (v2's `allow_control`: `--mcp-read-only` offers only the read class) and a
one-line purpose. The window and MCP send the same commands; a few commands are composed by the endpoint itself
from other commands (`guide`, `health`, `wait_for`, `batch`) or answered off the window's turn (`list_files`,
`read_file`, `read_signal`).

Status: **present** (v3 had it), **added** (this line), **different** (deliberately not as in v2, with the reason).

## Tools

| v2 tool | What it let an AI do (result, limits) | v3 | Status |
|---|---|---|---|
| `lab_health` | Protocol health even when the GUI owner hangs: service running/closing, pid, workspace, seconds since the owner last answered, queued owner requests, control_enabled. | `health`: answered by the endpoint without the window's turn: service, pid, workspace, seconds since the instrument last answered, calls in flight, control. | added |
| `lab_state(after_sequence, node_id, panel_id, details)` | Compact overview: workspace, devices, session, logic rows (status, generation, ready, error, issues, request, artifacts, published signals), panels, signals, console reports since `after_sequence`. | `get_state {sections?, panel?, node?}`: the same facts as sections (`workspace`, `apparatus`, `devices`, `logic`, `board`, `runs`, `panels`, `pulse`, `windows`, `viewer`); `node` keeps one logic row. | present (`node` added) |
| ↳ `reports` since `after_sequence` | Console messages (saved, refused, warnings) as a log. | Every command is answered with its own result or error; lasting facts stay on their rows (`logic[].run.error`, `devices[].error`, `panels[].error`, `workspace.notes`). | different: no second log beside the replies |
| `lab_catalog()` / `(api_name)` | Installed node types before Init (purpose, inputs, outputs, fields, form, artifact inputs, devices, resources); plot kinds, facet cell kinds, axis roles, reductions, fit models (formulas with `details`). | `list_plugins {type_id?}` (device and node types, full descriptors); `plot_catalog {details?}` (plot kinds, cell kinds, roles, reductions, fit models with formulas and parameters). | present + `plot_catalog` added |
| `lab_catalog(node_id / panel_id)` | Bound choices, units, source options, editor parameters of one node or panel. | `get_state {node}` (controls with choices and problems, `source_choices`, `device_choices`, `editor_facts`, `issues`); `get_state {panel}` (controls with choices, `fit.models`, axes). | present |
| `lab_guide(topic)` | Workflows by topic (overview, setup, acquisition, pulse, slm, calibration, processing, scan, plot, fit, files, recipes, lifetime, errors); `node:<api_name>`. | `guide {topic?}`: the same topics written for v3 (`crates/zlc-bus/src/guide.md`); `type:<type_id>` = that type's descriptor. | added |
| server instructions | What ZLC is, where to start, acceptance ≠ completion, never replay an uncertain write, ids vs names, use native fits. | `initialize.instructions`, rewritten for v3 (short; points at `guide`). | present, rewritten |
| `lab_screenshot(window, device_id)` | PNG of the Console/Devices/Device Control window with its open dialog; read class, no file. | `get_window_image {window?}` (read, no file); `save_image {window?, path?}` (control: also writes the PNG); `get_panel_image {panel}`. Dialogs are part of the page. | `get_window_image` added |
| `lab_wait(node_id, generation, until, timeout ≤1800 s)` | Observe one exact run until ready/terminal; returns early (`reached=false`) when the run asks the operator. | `wait {node, until, run?, timeout_seconds ≤1800}`: `run` refuses another run than the named one; a request returns a ready/terminal wait early with `reached: false`. | present (`run`, early return, 1800 s added) |
| `lab_signal(signal, generation, offset, limit ≤4096, dataset)` | Paged numeric values + validity + sigma of a published signal, with every axis's sample codes, coordinates and labels per value; latest chunk or the finite dataset / retained window. | `read_signal {signal, generation?, offset?, limit?, dataset?}`: the same page shape (flat and physical indices, values with null for non-finite, valid, sigma, per-axis codes/coordinates/labels, holes of a partial finite dataset). | added |
| `lab_results(panel_id, offset, limit)` | Native fit rows: coordinates, units, sigma, fixed/valid, status, source refs, published signal names. | `get_fit {panel, offset?, limit?}` answers the table (status, rows, parameter names/units, sample axes, revisions); signal names in `get_state board.panels[].outputs`. | present |
| `lab_estimate(node_id)` | The node's run projection (Scan: sweeps, shots, Pulse time). | `estimate_node {node}`. | present |
| `lab_recipes(name)` | Recipes and one recipe's parameters/defaults. | `list_recipes {name?}`. | present |
| `lab_files(directory, offset, limit ≤200)` | Workspace listing, workspace-relative paths; never outside the workspace or its `.mcp`. | `list_files {directory?, offset?, limit?}`: same, confined to the workspace. | added |
| `lab_artifact(path, offset, limit ≤64000, section)` | Read a workspace file: text/json/md/log paged by bytes; PNG/JPEG as an image; a zlc.figure NPZ as a summary (datasets, sections, members), paged, one `section`. | `read_file {path, offset?, limit?, section?}`: the same, through the native archive reader; other NPZ files list their members. | added |
| `lab_device_state(device_id)` | A Device Control's cached reading, permission, pending. | `get_state {sections: ["devices"]}`: `devices[].fields`, `.control`, `.state`, `.error` (after `read_device`). | present |
| `lab_pulse_state(device_id)` | The Pulse Control's cached board state, pending, warning. | `pulse_state`. | present |
| `lab_editor_catalog(device_id)` (Pulse) | The Pulse editor's actions, ids, units, choices. | The `pulse_*` commands' schemas in `tools/list` + `pulse_state` (periods, ports, bindings, scan, Config, board). | present |
| `lab_editor_state(device_id, details)` (Pulse) | Draft, pending/error, accepted-device facts; document/Config with details. | `pulse_state`. | present |
| `lab_editor_action(device_id, action, arguments, expected_revision)` (Pulse) | Any Pulse editor action through its own worker. | `pulse_edit`, `pulse_scan`, `pulse_config`, `pulse_subpulse`, `pulse_target`, `pulse_preview`, `pulse_page`, `pulse_new`, `pulse_clear`, `pulse_open`, `pulse_save`, `pulse_connect`, `pulse_validate`, `pulse_fire`, `pulse_hold`, `pulse_step`, `pulse_sync`, `pulse_stop`. | present |
| `lab_editor_*` (SLM Control) | Target preset/spots, pupil, operator, solve, Send, adopt, Target/Science Context load/save. | — | missing in v3 as a feature (see open items) |
| `lab_wait_for(tool, arguments, path, equals, timeout ≤1800)` | Wait until a JSON field of a read tool's reply equals a value. | `wait_for {tool, arguments?, path, equals, timeout_seconds?}`: the same over any read command (not wait/wait_for/batch). | added |
| `lab_review_points(node_id, generation, request_id, excluded_ids)` | Read the site review's points and checks; confirm with the excluded ids. | The request in `logic[].run.request` (payload points `[id, label, x, y]`); `answer_request {node, request, response: {excluded_point_ids}}`. | present |
| `lab_save(panel_id, path, refresh, layout_path)` | Save a panel's Figure (image + NPZ), optionally freezing current data first and saving the Layout. | `save {panel, path, overwrite?}` writes the panel as shown (an `edit-<panel>` saves its held snapshot; `refresh_panel_edit` refreshes it); the reply lists the written files; a file already there is refused, named, unless `overwrite`; `save_layout {path}` for the Layout. | present (reply paths added) |
| `lab_devices(load / save, path)` | Load or save the apparatus draft. | `load_apparatus` / `save_apparatus`. | present |
| `lab_initialize()` | Init the authored apparatus. | `apply_devices`. | present |
| `lab_close(scope=session)` | Shut down devices. | `shut_down_devices`. | present |
| `lab_close(scope=experiment, decision)` | Close the application; unsaved drafts first answer `needs_confirmation` with discard/cancel. | `close_experiment {discard_changes?}`: refused while the Pulse Editor holds unsaved changes unless `discard_changes`. | added |
| `lab_close(scope=editor, device_id)` | Close one Control window. | — | different: a v3 editor window is a view of shared state (`show_window` brings it back); closing it changes nothing a client needs |
| `lab_run_recipe(name, parameters, start)` | Apply a recipe's layout, start its followers and entry node. | `run_recipe`. | present |
| `lab_action(logic.add / update / rename / validate / start / stop / revert / remove)` | Logic rows. | `add_node` / `edit_node` / `rename_node` / `check_node` / `start_node` / `stop_node` / `revert_node` / `remove_node`. | present |
| `lab_action(panel.add / update / remove / edit / refresh / save)` | Panels. | `add_panel` / `configure` (+ `rename_panel`, `set_panel_interval`, `set_panel_overlay`, `publish_panel_outputs`) / `close_panel` / `edit_panel` / `refresh_panel_edit` / `save`. | present |
| `lab_action(layout.save / load)` | Layouts. | `save_layout` / `load_layout` (+ `read_layout`, `clear_board`). | present |
| `lab_action(view.monitor / view.logic)` | Show the Console's Monitor or Logic page. | `show_window {window: "task-console", page: "monitor" \| "logic"}`. | added (`page`) |
| `lab_device_action(open / refresh / unit / apply)` | Device Control. | `read_device {device, units?}` / `tune_device` / `accept_device_risk`. | present |
| `lab_pulse(open / inspect / fire / stop, device_id, path)` | The bound Pulse Editor. | `pulse_open` / `pulse_state` / `pulse_fire` / `pulse_stop`. | present |
| `lab_batch(calls ≤16, 30 s)` | Already-decided calls in order with `expect`; `{"$result": [i, ...path]}` reads an earlier reply; stops at the first error/refusal/unmet wait; no rollback or replay. | `batch {calls}`: the same rules over the catalogue (control class). | added |
| `lab_handles`, `lab_api`, `lab_call`, `lab_intent` (developer profile) | Reflection over the live Python owners' public methods and Qt signals. | — | different: v3 has no second path (WORKBENCH_COMMANDS §8.2); every owner action is a schema'd command, device methods a plugin declares are `<instance>__<command>` tools |
| proxy `lab_connections` | List this workspace's running experiments and which are live, without tokens. | `zlc-mcp` (stdio entry point) `list_connections`: every running instrument on this computer (pid, url, workspace, control, live), never a token. | added |
| proxy `lab_connect(connection)` | Attach the AI connection to one running experiment; tool list changes. | `zlc-mcp` `connect {connection}`; `notifications/tools/list_changed` follows. | added |
| `--client-config codex/claude` | Print the client registration. | `zlc-mcp --client-config claude/codex`. | added |
| `--tools operator/developer` | Two tool profiles. | One profile. | different: no reflection tools to hide |

## Endpoint rules

| v2 | v3 | Status |
|---|---|---|
| Loopback only, bearer token, any `Origin` refused. | Loopback only, bearer token, non-loopback `Origin` refused. | present |
| `allow_control` (default on): off offers only read tools. | `--mcp-read-only`; `control_enabled` in the connection record. | present |
| Remote file access confined to the workspace, `.mcp` excluded. | `list_files`/`read_file` confined to the workspace; relative paths of every remote command resolve against the workspace root; absolute command paths are accepted as the window's file dialogs accept them. | different for command paths (typed readers only; nothing returns arbitrary file content) |
| GUI owner 5 s answer limit. | Commands wait for their settled reply; `health` answers even while the window's turn is busy. | different: v3's turn does not block on device work |

Replies made usable on the way (found by the end-to-end check): `get_fit` answers with the fit table itself
(it was a panel summary with the table under `result`); panel commands name the panels they added
(`added_panels`, v2 `panel.add` returned its id); `save` lists the files it wrote; a `wait` begun while the
node's next run has not started (draft check, queued restart, a following processor) does not answer with the
ended run the row still shows.

## Counts

Tools table, 42 rows (one per v2 tool, `lab_action`/`lab_close`/`lab_editor_*` split by what they act on):

| | rows | |
|---|---|---|
| present | 23 | v3 already had the capability (some extended: `get_state {node}`, `wait {run}`, `save` reply) |
| added | 14 | `health`, `plot_catalog`, `guide`, `get_window_image`, `read_signal`, `list_files`, `read_file`, `wait_for`, `close_experiment`, `show_window {page}`, `batch`, `list_connections`, `connect`, `--client-config` |
| different | 4 | console reports log, closing an editor window, developer reflection (4 tools), tool profiles: each for the reason in its row |
| missing | 1 | the SLM Control editor (below) |

Endpoint rules, 4 rows: 2 present, 2 different (command paths outside the workspace, the 5 s owner limit).

## Checks

- `cargo test -p zlc-bus`: catalogue schemas, endpoint composition (guide topics, health, wait_for, batch
  stops at the first refusal / unmet expect, `$result` binding, read-only endpoint offers no batch), the
  proxy over two in-process endpoints (list without tokens, choose, attach, forward, switch, a gone record
  removed).
- End-to-end client (`v2port\K_tools\e2e.mjs` via `K_tools\run_app.ps1`, loopback + token, only the tools
  and the guide): guide topics → list_files/read_file (paged, confined) → batch add_device ×2 + apply_devices
  → wait_for each device open → camera_measurement + occupancy (workspace-relative calibration) admitted →
  batch start → wait ready → pulse_fire → wait {run} terminal → read_signal counts/occupied pages with axis
  codes → histogram panel + bimodal_gaussian fit → get_fit (status current) → save → read_file of the
  archive (summary, one section) and the PNG → get_window_image before/after show_window {page: logic} →
  batch stopped at a refusal → shut_down_devices → close_experiment, record removed. Passes.
- Two instruments on loopback (`K_tools\connect.mjs`, `run_app.ps1 -Apps 2`, one LOCALAPPDATA): zlc-mcp
  over stdio lists both live without tokens, refuses a call before a choice, attaches to each (tools/list_changed,
  104 tools, get_state/health are that instrument's), answers health in 2 ms during a 4 s wait_for, refuses
  connect while a call runs, closes both, the list empties; no token in anything the client received. Passes.

## Open items

1. **SLM Control editor.** v2's SLM Device Control (Target preset/spots, Pupil, operator layers, solve,
   Send, adopt the device phase, Target and Science Context load/save) has no v3 equivalent: line D removed the
   1811-line Qt editor as a device↔application coupling and no line ported its functions, so there is neither a
   window nor a command for them (a Science Context can only come from a previous `slm_feedback` run). The
   v3 shape is SLM device commands (`<instance>__<command>`, PLUGIN_PROTOCOL §8; the solver is
   `zlc_atom/devices/slm/solver.py`) plus a TS view; an AI then reaches them through the catalogue unchanged.
2. Window names for `get_window_image`/`save_image`/`show_window` are what `windows::Kind::parse` knows on
   this branch (the four v2 windows); the per-device Control/Log, per-Figure and review windows on other
   lines' branches become reachable the same way once merged.
3. A crashed instrument's connection record stays until the next `list_connections` finds nothing listening
   there and removes it.
4. Remote command paths outside the workspace are accepted (other lines' checks open archives elsewhere);
   only the readers that return file content are confined.
5. For F: `wait {until: "terminal" | "ready"}` now answers early with `reached: false` when the run asks the
   operator (v2 rule), so a flow that waits over a review answers the request and waits again.
