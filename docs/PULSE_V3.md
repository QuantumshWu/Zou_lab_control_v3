# Pulse in v3: placement, commands and SAFE

Line E of the v2 → v3 port (2026-10-20). v2 reference: `Zou_lab_control_v2`
master `33ce4ae0` (`packages/zlc_pulse`, `zlc_workbench/pulse_editor.py`,
`zlc_ui/pulse`). The RTL, the XDC, the board manifest and the deployed bitstream
are not touched: v3 drives the same frozen period-table board
(`LAYOUT_STRUCT_VERSION` 8, word-63 fingerprint).

## Decision: the pulse stack is native Rust, in the core process

| Layer (v2 module) | v3 owner | Why |
|---|---|---|
| model, loops, binding, scan, schedule (`model.py` … `schedule.py`) | `crates/zlc-pulse` | Every editor keystroke builds a candidate sequence and lets the model refuse it. In-process validation keeps that a function call, not an IPC round trip to a plugin. |
| compile, canonical digest, wire geometry/fingerprint/pack (`compile.py`, `canonical.py`, `wire.py`) | `crates/zlc-pulse` | They define what the FPGA plays. Small, deterministic, no vendor SDK; one implementation in the core instead of a Python one beside a Rust host. Verified bit-exact against v2 (below). |
| codec: `zlc.pulse`, `zlc.subpulse`, `zlc.pulse.config_values` (`codec.py`, `pulse_state.py`) | `crates/zlc-pulse` | Same grammar, same readable layout: a file saved by v3 reads in v2 and diffs cleanly. |
| board manifest → target, XDC/RTL cross-check (`manifest.py`) | `crates/zlc-pulse` | Reads the deployed board's `streamer_config.json`, `board.xdc` and RTL top, which ship in `crates/zlc-pulse/fpga` (copied from the v2 build tree that made the bitstream; `ZLC_PULSE_FPGA` points elsewhere). |
| transports: memory twin, UART CRC framing + COM link (`transport/memory.py`, `uart*.py`) | `crates/zlc-pulse` | UART is bytes and a CRC-16; the Windows COM port is opened with the same DTR/RTS-low rule through `windows-sys`. |
| device: open/load/fire/observer/bank refill/wait/SAFE (`device.py`) | `crates/zlc-pulse` | Safety-relevant: Stop and Stop all call `PulseStreamer::safe` directly, with no process hop before `CMD_SAFE`. |
| remote client and server (`remote.py`) | `crates/zlc-pulse` | Same length-prefixed JSON protocol and the same `__type__` trees, so a v3 client drives an existing v2 `pulse_server` (and a v2 client a v3 server) while the FPGA machine migrates. |
| JTAG-to-AXI transport (`transport/axi.py`) | `crates/zlc-pulse` (`axi.rs`) | One resident `vivado -mode tcl` per board, driven as v2 drives it: marker-wrapped Tcl blocks, INCR bursts (≤256 words, never across 4 KiB), the command acknowledgement polled inside its block, a Stop abandons a read without ending Vivado (the next block drains it). `zlc-pulse-server --backend jtag-axi` (and `auto` when no UART answers) serves it; the server tells clients each request's budget. Tested against `axi::FakeVivado` (a Tcl session in front of the memory twin), never against Vivado or a board. |
| Vivado build/capacity tooling (`fpga.py`) | stays in v2 | Build-time only. |
| editor presenter (`pulse_editor.py`, `pulse_state.py`) | `crates/zlc-pulse-editor` | Authoring state, edits and every projection (schedule, scan, config, subpulse, target pages; preview timeline) are pure functions of the state; the TS view only renders them. |
| editor view (`zlc_ui/pulse/*`) | `apps/desktop/ui/src/pulse/` (TS) | Draws the view model, sends intents. No pulse rules in TypeScript. |
| preview (`timeline_of` → `zlc_plot` pulse timeline) | `zlc-pulse-editor` → `zlc_plot::PlotSession::pulse_timeline` | The native PulseTimeline kind already reproduces v2's fronts; the editor only supplies `PulseTimelineData`. |

Python plugins (atom logic nodes, the simulation world) do not import a pulse
compiler to execute anything: they send a document plus API values / scan table
to the core through the bus commands below, and the core resolves, compiles,
loads and fires — one compiler on every execution path. The virtual sequencer is
the native memory twin plus a playback thread that publishes STATUS/CURSOR at the
real wall duration of each Pulse run and calls a world hook per run (the hook the
simulation plugin connects to through the plugin host).

Two v2 design points are changed on the way, each for a stated reason:

- `ConfigValueHolder` was a mixin shared by the device and the remote client, and
  its `_prepare_config_program` recompiled inside `fire` using `self.describe()`.
  In v3 Config resolution is one free function of (authored source, Config
  values, geometry, clock); both device kinds call it. No mixin state is copied.
- v2 Python `Fraction` arithmetic on `str(float)` is reproduced with exact
  decimal-rational arithmetic (`zlc_pulse::exact`): same decisions on the clock
  grid and the same correctly rounded floats. A value so large that its tick
  count does not fit 128 bits is refused when the period is built, where v2
  accepted it into the model and refused it at compile.

## Who owns the board

One board per application, held by the desktop's pulse host (`zlc_pulse::board::Board`:
the virtual board or a remote pulse server). It is attached one of two ways, as in v2:

- as an apparatus device: native device types `sequencer.virtual` (the memory twin,
  played into the simulated world) and `sequencer.hardware` (a pulse server at
  host:port), capability `sequencer.streamer`, physical output `sequence`.
  Apply opens them like any device; a node that declares
  `DeviceRequirement("sequencer.streamer", ...)` claims that instance for its run.
  The Pulse Editor then shows the connection locked (v2's embedded editor:
  "Experiment session").
- from the Pulse Editor's own connection control (Virtual (sim) / Remote server /
  Offline), when no sequencer device is open (v2's standalone PulseGUI).

The virtual board's world is line D's `simulation.board` device (simulation group):
every played run is handed to its `play_run(timeline)` (`board::run_timeline`, the
grammar of D's `SimulationWorld.fire`), and a SAFE calls its `safe()`. Without one
the virtual board still plays at wall time.

Run claims: a `core` command from a plugin run arrives with `_run`. `pulse_run`,
`pulse_wait_done` and `pulse_stop` carrying `_run` are refused unless that run
claims the open sequencer device instance (`Workbench::run_claims`), exactly as a
plugin device call is refused for an unclaimed device; a board attached only by
the editor cannot be driven by a run. While a run claims the sequencer the editor's
On Pulse / Hold / Step / Stop and any other client's `pulse_run` / `pulse_wait_done` /
`pulse_stop` are refused, "device is in use by <node>" (v2's device-use lease): the
operator stops the node, whose own Stop makes the board safe. A run's `pulse_stop`
answers with the SAFE's outcome, before the run goes on.

A run's documents are pinned: the first `pulse_run` of a document reads it, and a
later one of the same run that finds the file changed is refused ("changed during
this run"), so a run's record names the one pulse it played. A run that ends -- its
node returned, failed or was stopped, or its plugin process died -- with a FIRE of
its still unreported has the board made SAFE (v2's per-node `finally`).

## Commands (bus catalogue, `crates/zlc-bus`)

Editor commands edit the open document; device commands act on the board. UI and
MCP send the same names with the same schemas. `read` commands are offered to a
read-only MCP client too; every other one is `control`.

| Command | Access | Meaning |
|---|---|---|
| `pulse_state` | read | The `pulse` section of the app state: document, every page's projection, connection, board snapshot, last message. |
| `pulse_open`, `pulse_save`, `pulse_new`, `pulse_clear` | control | File and document lifecycle (explicit paths for remote clients). |
| `pulse_edit` | control | One edit: `period_name`, `duration`, `digital`, `analog`, `delay`, `binding`, `config_binding`, `bracket`, `bracket_add`, `bracket_remove`, `period_add`, `spacer_add`, `period_remove`, `remove_items`, `reorder`, `run_repeats`, `visible_ports`, `fill_port`, `clear_port`, `document_name`, `port_label`. Same arguments as v2's `control_action`. |
| `pulse_scan` | control | Scan page, by `op`: `template`, `edit_source`, `load_program`, `run` (the Python program in a child process, 2 GB / 120 s cap, returning the table), `load_array` (.npy/.csv/.txt), `save_array`, `clear`, `repeats`. |
| `pulse_config` | control | Value Config page, by `op`: `edit`, `new`, `load`, `refresh`, `unload`, `save`, `mode`, `binding`. |
| `pulse_subpulse` | control | Subpulse page, by `op`: `select`, `source`, `open_source`, `new`, `open`, `save`, `save_as`, `export`, `insert`, `insert_file`, `group`, `ungroup`, `remove`, `rename`, `edit` (a `pulse_edit` on the selected Subpulse). |
| `pulse_target` | control | Target page: apply port records. |
| `pulse_preview` | control | Preview page: `include_off`, `selectors`, `size`; `save` writes the drawn figure (PNG). The drawing is the native `pulse_timeline` panel `pulse-preview`. |
| `pulse_page` | control | The page shown (Edit, Preview, Scan, Value Config, Subpulse Config, Target); Preview redraws on show. |
| `pulse_sync` | control | Bring the board's pulse into the editor; a board playing this draft, or one row of its scan (Hold, Step), keeps the draft and its table; unsaved changes are replaced only with `discard_changes` (the window asks, v2 confirm_pulse_discard). |
| `pulse_validate` | read | Compile with the loaded Config, without touching the board. |
| `pulse_connect` | control | Attach the editor to `virtual`, `remote` (`endpoint` host:port), `local` (a pulse server this app starts on the UART; never run in tests) or `offline`. Refused while an apparatus sequencer device holds the board. Opening a board drives it SAFE first. |
| `pulse_fire` | control | On Pulse: load what the editor shows and fire it. Multi-point tables normalize infinite Run repeats to 1 (also when generated/loaded), reflected in the editable draft; infinite Scan repeats cycles the whole table. Finite Run repeats are retained. Without a multi-point table, infinite Run repeats retains ordinary continuous playback. |
| `pulse_hold`, `pulse_step` | control | Play one scan point as an ordinary pulse. |
| `pulse_describe` | read | `{document}` → `{path, name, api_fields, scan_fields, unscannable, periods}` (what a node's pulse resource resolves to). `api_fields` in played order: `{field_id, label, kind, unit, value}`; `scan_fields` the template's scan slots in slot order: `{field_id, label, kind, unit, value}` (a DAC's unit is the signed code; `value` what the slot holds in the pulse, where a plan's first sweep is centred); both with the column facts `is_dac, limit_lo, limit_hi` (v2 `api_parameter_columns_for` / `scan_columns_for`; v2's seeded `lo, hi` are not stated); `unscannable` maps an API count of an empty bracket to the sentence why; `periods` `{period_id, name, label}`. |
| `pulse_run` | control | `{document, api_values:{field_id:[number, unit]}, run_repeats, scan_repeats, scan?:{fields, rows}}`: a time converted exactly into its field's unit (any other field takes its own unit only); with `scan` exactly those fields are the board's scan slots (rows in the fields' units and order, quantized to what plays; two rows that play the same, a field the pulse lacks or more fields than the board's slots are refused), without it no slot streams; ABI check, compile, load, fire (a run's `stage: load` loads only and `stage: fire` fires what that load left on the board, v2's load-then-arm-then-fire) → `{command_id, api_values, applied (v2 sequencer archive snapshot), timing, scan?:{fields, rows as played}}`. `timing` is the program the FIRE plays (Config filled), one run of the table's first row, every loop body replayed at most twice: `{clock_hz, digest, duration_ticks, earliest_lane_delay_ticks, periods:[{period_id, name, label, starts, duration_ticks}], rising_edges:{lane: ticks}}` (non-clock lanes; v2 `frame_visits` / `trigger_edge_ticks`, `bracket_bodies=2`). |
| `pulse_wait_done` | control | `{command_id, timeout_seconds}` → `{done, status: done\|error\|underflow, cursor, elapsed_seconds, command_id, fault}`, or `{done: false}` while that FIRE plays; waits off the command lane. See "One report per FIRE". |
| `pulse_estimate` | read | `{document, api_points?:[{field_id:[number, unit]}] (≤ 256), scan?:{fields, columns}}` → `{clock_hz, table_ticks, delay_ticks, points:[{api_values, board_rows, first_row_ticks, table_ticks, delay_ticks}]}`: v2 `estimate_scan_time`'s compile loop, on the attached board's geometry with the Config it holds; the node multiplies by its host points, shots per point and sweeps. |
| `pulse_stop` | control | SAFE now, never queued behind a pending LOAD/FIRE (a superseded On Pulse / Hold goes no further and leaves the board SAFE); refused while a run holds the board. |

## One report per FIRE

The board keeps what became of its last FIRE (`Board::last_fire`: pending,
reported or made safe). Its report is taken exactly once, by whoever waits for it
(`pulse_wait_done`, or the editor's completion lane); until then a LOAD or FIRE
is refused with a sentence naming that FIRE (the device's own rule, now said
plainly). `pulse_wait_done` refuses, instead of answering `{done: false}`
forever, a FIRE that can no longer report: its report already taken, the board
made safe (Stop, stop_all, a SAFE elsewhere on a remote board), another FIRE sent
since, or one this board never sent. A waiter blocked on a FIRE that Stop ends is
answered at once.

## SAFE

- The operator plays a pulse with no arming step, as in v2 (On Pulse, pulse play);
  an MCP client does when the endpoint allows control (v2 `allow_control`). A run's
  `pulse_run`, `pulse_wait_done` and `pulse_stop` need the sequencer claimed by
  that run; the operator's need it free of other runs.
- On connect the host registers the board's SAFE as Stop all's safe action for
  output `pulse-streamer`; disconnect unregisters it. `pulse_stop` and `stop_all`
  run `PulseStreamer::safe` (stop observer, `CMD_SAFE`, verified readback) before the
  result is reported; `stop_all` is never refused (it also stops the runs).
- Stop is sticky exactly as in v2: a FIRE still being prepared (Config reread,
  compile, reload) when a Stop arrives is refused instead of being sent after it.
- Editing, validating, saving and previewing are never blocked: they cannot move
  atoms. The board digest comparison ("is the board playing what I see?") is a
  read.

## Verification

- Golden (`research/pulse-golden/`): v2 run on its own example pulses and on the
  sim/diff variants (`fpga/pulse_streamer/sim/diff/make_diff_pulses.py`, packed by
  `pack_diff.py`'s rules), writing compiled programs, digests, packed images, scan
  wire rows, schedule windows, analog levels, timeline data and codec round trips.
  The Rust tests compare every field and word. xsim/Vivado are not run: the
  2026-09-23 differential simulation already proved those images play identically
  on the RTL, so an identical image is an identical pin trace.
- Device and transport against the memory twin (the same twin v2 tests with);
  remote client/server cross-checked against a v2 `pulse_server --backend memory`.
- Run surface (`research/pulse-golden/make_run_golden.py`): v2's functions on the
  same 595 pulses and line D's four documents — the description's scan facts, a
  FIRE's API values, scan table as played (also with its fields reversed) and
  receipt timing, and the estimate's ticks — compared by
  `zlc-pulse/tests/run_golden.rs`; equal to D's `out_pulse_timing.json`. The
  report semantics are `zlc-pulse/tests/board.rs`.
- Editor layer (`research/pulse-golden/make_editor_golden.py`): v2's presenter on the
  same 595 pulses — item order, schedule projections (plain, on a board, with Config,
  visible ports, scan points), target records, preview data and a seeded 14-step
  edit script — compared field by field by `zlc-pulse-editor/tests/golden.rs`.
- The window: an off-screen walk (open, every page, preview front, virtual connect,
  On Pulse, Stop, `pulse_describe`) with
  screenshots beside v2's.
- No real hardware is touched by any of this.


### Staged LOAD / FIRE receipts

The existing per-run board owner retains the accepted LOAD receipt, including named
quantized scan rows and API inputs. Before FIRE it checks both the program digest and
wire table against the board. The reply combines that LOAD's `scan` / `api_values`
with the current FIRE's command ID, applied snapshot and timing; it does not reread
or recompile the pulse file. A successful new LOAD replaces this record, and a combined
run supersedes it. Table columns retain their declared names and order.

Seamless Scan resolves returned columns by field identity and reports a descriptive
protocol error if its required table is absent or malformed, then follows normal Stop
cleanup. The regression uses real PulseHost/virtual-board LOAD and FIRE responses,
including reversed column order and quantization, as input to Python Scan execution.
