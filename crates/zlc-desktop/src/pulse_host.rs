//! The pulse board and the Pulse Editor on the window's event loop
//! (docs/PULSE_V3.md): the editor's jobs run on lanes off the window's
//! turn (device: one at a time in order; SAFE beside it; a finite run's
//! completion wait; the operator's scan program in the science interpreter;
//! the preview's drawing), the board's SAFE is one of Stop all's safe actions,
//! a plugin run's pulse commands are held to the run's claims, and the
//! sequencer apparatus devices (`sequencer.virtual|hardware|local`) are
//! native: opening one attaches the board the editor and the runs share.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, mpsc};
use std::time::Duration;

use serde_json::{Map, Value, json};
use zlc_core::control::SafeActions;
use zlc_plot::configuration::PlotLabels;
use zlc_plot::configuration::resolve::Defaults;
use zlc_plot::fit::FitEngine;
use zlc_plot::interaction::gesture::InteractionDefaults;
use zlc_plot::render::panel::NativeRendererFactory;
use zlc_plot::session::{EngineFitLayer, PlotSession, SessionServices};
use zlc_pulse::board::{Board, Fate, World};
use zlc_pulse::run::{ScanAxes, ScanRows, describe_document, done_json, estimate_document, fire_loaded, load_sequence, read_document, run_sequence};
use zlc_pulse::json::Json;
use zlc_pulse::model::Sequence;
use zlc_pulse::scan::ScanTable;
use zlc_pulse::transport::Event;
use zlc_pulse_editor::editor::{Deployed, Editor};
use zlc_pulse_editor::session::{Job, Outcome};
use zlc_pulse_editor::{tables, to_value};
use zlc_plugin_host::PluginHost;

use crate::science::Front;

/// The Pixels panel id the Preview page's timeline is delivered as.
pub const PREVIEW_PANEL: &str = "pulse-preview";
/// Stop all's name for the board's SAFE.
pub const SAFE_OUTPUT: &str = "pulse-streamer";

/// What a lane or a worker answers on the window's turn.
pub enum PulseEvent {
    Outcome(Outcome),
    /// A command answered later (a run, a completion wait); `run`: what a run's pulse_run leaves it holding.
    Reply { request: u64, result: Result<Value, String>, touched: bool, run: Option<RunReply> },
    /// Someone else drove the board: the editor asks it again.
    Touched,
    /// A native sequencer device opened (or did not).
    Opened { instance: String, result: Result<(Arc<Board>, Option<Served>), String> },
    Closed { instance: String, error: Option<String> },
    /// A preview front, the words above it and its axes (what the page hit-tests).
    Preview { result: Result<(Front, String, Vec<Value>), String>, key: String },
    /// A pointer message the preview handled: its new front and axes when the picture changed.
    Gesture(Result<Option<(Front, Vec<Value>)>, String>),
    PreviewSaved(Result<PathBuf, String>),
    /// A line of a native sequencer's device log (the local pulse server's narration).
    Log { instance: String, line: String },
}

pub type Post = Arc<dyn Fn(PulseEvent) + Send + Sync>;

/// One worker thread taking work in order.
pub(crate) struct Lane(mpsc::Sender<Box<dyn FnOnce() + Send>>);

impl Lane {
    pub(crate) fn new(name: &str) -> Lane {
        let (sender, receiver) = mpsc::channel::<Box<dyn FnOnce() + Send>>();
        let _ = std::thread::Builder::new().name(format!("zlc-{name}")).spawn(move || { for work in receiver { work(); } });
        Lane(sender)
    }
    pub(crate) fn run(&self, work: impl FnOnce() + Send + 'static) { let _ = self.0.send(Box::new(work)); }
}

/// The world a virtual board plays into: line D's `simulation.board`.
struct PluginWorld { host: PluginHost, instance: String }

impl World for PluginWorld {
    fn play_run(&self, timeline: &Json, _stop: &Arc<Event>) -> zlc_pulse::Result<()> {
        let ticks = timeline.get("duration_ticks").and_then(Json::as_int).unwrap_or(0) as f64;
        let clock = timeline.get("clock_hz").and_then(Json::as_f64).unwrap_or(1.0);
        let deadline = Duration::from_secs_f64(ticks / clock + 30.0);
        self.host.spi(&self.instance, "play_run", json!([to_value(timeline)]), json!({}), deadline).map(|_| ())
            .map_err(|error| zlc_pulse::Error::new(zlc_pulse::Kind::Runtime, format!("{}: {error}", self.instance)))
    }
    fn safe(&self) { let _ = self.host.spi(&self.instance, "safe", json!([]), json!({}), Duration::from_secs(5)); }
}

/// How a command was answered.
pub enum Handled { Now(Result<Value, String>), Later }

/// What one of a run's `pulse_run`s leaves the run holding: the document it
/// read (pinned at the first), the FIRE it sent, the program a `load` stage
/// put on the board (its digest).
pub struct RunReply { run: String, pinned: Option<(PathBuf, Arc<Sequence>)>, fired: Option<u32>, loaded: Option<String> }

/// A plugin run's use of the board: each pulse document as its first fire read
/// it (a later fire of a changed file is refused: a run's record names one
/// pulse), the FIRE it sent last, the program its `load` stage put on the board,
/// and its `pulse_run`s still on the lane.
#[derive(Default)]
struct RunUse { pinned: BTreeMap<PathBuf, Arc<Sequence>>, fire: Option<u32>, loaded: Option<String>, in_flight: usize, ended: bool }

/// The words a refused fire of a changed pulse file says (a node ends its batch on them).
pub const PULSE_CHANGED: &str = "changed during this run";

/// One `pulse_run` stage of a run (or a client): `load` (LOAD only, v2
/// `arm_sequencer`), `fire` (FIRE what that LOAD put on the board) or both.
fn play(board: &Board, stage: &str, document: &Path, pinned: Option<&Sequence>, loaded: Option<&str>, values: &[(String, (f64, String))],
        scan: Option<&ScanRows>, runs: u64, sweeps: u64) -> (Result<Value, String>, Option<(PathBuf, Arc<Sequence>)>, Option<u32>, Option<String>) {
    let fired = |receipt: &Json| receipt.get("command_id").and_then(Json::as_int).map(|id| id as u32);
    if stage == "fire" {
        let Some(loaded) = loaded else { return (Err("stage fire plays what this run's stage load put on the board; load it first".into()), None, None, None) };
        return match fire_loaded(board, loaded, runs, sweeps) {
            Ok(receipt) => { let id = fired(&receipt); (Ok(to_value(&receipt)), None, id, None) }
            Err(error) => (Err(error.message), None, None, None),
        };
    }
    let (path, sequence) = match read_document(document) { Ok(read) => read, Err(error) => return (Err(error.message), None, None, None) };
    if pinned.is_some_and(|pinned| *pinned != sequence) {
        let name = path.file_name().map_or_else(|| path.display().to_string(), |name| name.to_string_lossy().into_owned());
        return (Err(format!("the pulse {name} {PULSE_CHANGED}: the run's record names the pulse it began with, so the changed file is not played; \
                             start the run again to play it")), None, None, None);
    }
    let played = if stage == "load" { load_sequence(board, &sequence, values, scan) } else { run_sequence(board, &sequence, values, scan, runs, sweeps) };
    match played {
        Err(error) => (Err(error.message), None, None, None),
        Ok(receipt) => {
            let digest = (stage == "load").then(|| receipt.get("applied").and_then(|applied| applied.get("program"))
                .and_then(|program| program.get("digest")).and_then(Json::as_str).map(str::to_owned)).flatten();
            let id = fired(&receipt);
            (Ok(to_value(&receipt)), Some((path, Arc::new(sequence))), id, digest)
        }
    }
}

/// The apparatus device the board is: its instance, a local server child, and for the virtual
/// board the plugin instance of its world side (line D's simulation.board, simulation group).
/// `published`: Remote on -- the board's own pulse server admits other computers,
/// one of which now drives it; this application leaves it alone (Stop excepted).
struct Given { instance: String, child: Option<Served>, world: Option<String>, published: bool }

/// The pulse server this application started for `sequencer.local`: its
/// process, its port, and the line it hears admission on (`admit` / `refuse`;
/// the line closing ends the server, which drives the board SAFE first).
pub struct Served { child: std::process::Child, port: u16, control: Option<std::process::ChildStdin> }

impl Served {
    fn end(mut self) {
        drop(self.control.take());
        let until = std::time::Instant::now() + Duration::from_secs(5);
        while std::time::Instant::now() < until {
            if let Ok(Some(_)) = self.child.try_wait() { return; }
            std::thread::sleep(Duration::from_millis(50));
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

const PUBLISHED_BOARD: &str = "the pulse board is published on the bench fabric: another computer drives it -- withdraw Remote to use it here";

/// The virtual sequencer's world side: one plugin instance it opens and closes with itself, so
/// the apparatus has one virtual sequencer row as in v2 (whose sequencer.virtual drove the world).
fn world_instance(instance: &str) -> String { format!("{instance}_world") }

pub struct PulseHost {
    pub editor: Editor,
    device: Lane,
    safe: Lane,
    program: Lane,
    preview: Lane,
    post: Post,
    safe_actions: SafeActions,
    plugins: Option<PluginHost>,
    given: Option<Given>,
    python: Option<PathBuf>,
    session: Arc<Mutex<Option<PlotSession>>>,
    pub dpr: f64,
    preview_busy: bool,
    preview_key: String,
    preview_wanted: bool,
    pub frame: Option<(u32, u32, u64)>,
    frames: u64,
    /// The drawn preview's axes, as panels report theirs.
    axes: Vec<Value>,
    /// Pointer messages for the preview, in order (a newer move replaces an unsent one).
    pointers: std::collections::VecDeque<Value>,
    pointer_busy: bool,
    /// The workbench's record of outside drivers: the session's sequencer is "PulseGUI"'s while
    /// the editor's own pulse plays (v2 `_drive_lease`), so no run may claim it meanwhile.
    pub users: zlc_workbench::DeviceUsers,
    /// Each plugin run's use of the board, until it has ended and its replies are in.
    runs: BTreeMap<String, RunUse>,
}

/// The preview's front and its axes, as panels report theirs.
fn drawn(session: &mut PlotSession) -> zlc_plot::configuration::error::Result<(Front, Vec<Value>)> {
    let front = session.rgba()?;
    let axes = session.raster_axes_snapshot()?.iter().map(zlc_plot::document::axis_wire).collect();
    Ok((front, axes))
}

pub(crate) fn services() -> SessionServices {
    SessionServices { renderers: Arc::new(NativeRendererFactory), fit: Arc::new(EngineFitLayer::new(Arc::new(FitEngine::default()))),
                      defaults: Arc::new(Defaults::default()), registry: None, interaction: InteractionDefaults::default() }
}

/// `{field_id: [number, unit]}` as the catalogue checked it.
fn api_values(values: &Value) -> Vec<(String, (f64, String))> {
    values.as_object().map(|values| values.iter().filter_map(|(key, pair)| {
        Some((key.clone(), (pair.get(0)?.as_f64()?, pair.get(1)?.as_str()?.to_owned())))
    }).collect()).unwrap_or_default()
}

fn strings(items: &Value) -> Vec<String> { items.as_array().map(|items| items.iter().filter_map(|item| item.as_str().map(str::to_owned)).collect()).unwrap_or_default() }

fn rows(rows: &Value) -> Vec<Vec<f64>> {
    rows.as_array().map(|rows| rows.iter().map(|row| row.as_array().map(|row| row.iter().filter_map(Value::as_f64).collect()).unwrap_or_default()).collect())
        .unwrap_or_default()
}

/// The operator's scan program, run by the science interpreter with a memory
/// cap and a deadline: whatever it leaves in `scan_table`, as JSON.
const SCAN_RUNNER: &str = r#"
import json, sys
source = sys.stdin.read()
namespace = {}
exec(compile(source, "<scan program>", "exec"), namespace)
table = namespace.get("scan_table")
if table is None:
    print(json.dumps({"none": True}))
elif isinstance(table, dict):
    import numpy as np
    print(json.dumps({"columns": [[str(k), np.asarray(v, dtype=float).reshape(-1).tolist()] for k, v in table.items()]}))
else:
    import numpy as np
    rows = np.asarray(table, dtype=float)
    if rows.ndim == 1:
        rows = rows.reshape(-1, 1)
    if rows.ndim != 2:
        raise ValueError("scan_table must be two-dimensional: one row per point, one column per Scan field")
    print(json.dumps({"rows": rows.tolist()}))
"#;

fn run_scan_program(python: Option<&Path>, source: &str) -> Result<Option<ScanTable>, String> {
    use std::io::{Read, Write};
    use std::os::windows::process::CommandExt;
    let python = python.ok_or_else(|| format!("{} (scan programs run in it)", zlc_plugin_host::NOT_INSTALLED))?;
    // Run as the plugin processes are (no -I: the same packages, user site and PYTHONPATH; no
    // PYTHONHOME; the base installation's DLLs and the thread policy); a console program, given no
    // console window (this application has none).
    let mut child = std::process::Command::new(python).env_remove("PYTHONHOME").envs(zlc_plugin_host::plugin_environment(python)).args(["-c", SCAN_RUNNER]).stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::piped()).creation_flags(windows_sys::Win32::System::Threading::CREATE_NO_WINDOW)
        .spawn().map_err(|error| error.to_string())?;
    let job = crate::job::limit(&child, Some(2048));
    if let Some(mut stdin) = child.stdin.take() { let _ = stdin.write_all(source.as_bytes()); }
    let deadline = std::time::Instant::now() + Duration::from_secs(120);
    let status = loop {
        if let Some(status) = child.try_wait().map_err(|error| error.to_string())? { break status; }
        if std::time::Instant::now() > deadline { let _ = child.kill(); return Err("the scan program ran longer than 120 s and was stopped".into()); }
        std::thread::sleep(Duration::from_millis(20));
    };
    drop(job);
    let (mut out, mut err) = (String::new(), String::new());
    if let Some(mut stdout) = child.stdout.take() { let _ = stdout.read_to_string(&mut out); }
    if let Some(mut stderr) = child.stderr.take() { let _ = stderr.read_to_string(&mut err); }
    if !status.success() {
        let last = err.lines().rev().find(|line| !line.trim().is_empty()).unwrap_or("the interpreter failed").trim().to_owned();
        return Err(last);
    }
    let answer: Value = serde_json::from_str(out.lines().last().unwrap_or("")).map_err(|error| format!("the scan program's table: {error}"))?;
    if answer.get("none").is_some() { return Ok(None); }
    if let Some(rows) = answer.get("rows").and_then(Value::as_array) {
        return Ok(Some(ScanTable::Rows(rows.iter().map(|row| row.as_array().into_iter().flatten().filter_map(Value::as_f64).collect()).collect())));
    }
    let columns = answer.get("columns").and_then(Value::as_array).ok_or("the scan program's table is neither rows nor columns")?;
    Ok(Some(ScanTable::Columns(columns.iter().filter_map(|pair| Some((pair[0].as_str()?.to_owned(),
        pair[1].as_array()?.iter().filter_map(Value::as_f64).collect()))).collect())))
}

/// `sequencer.local`: the plugged-in board served by `zlc-pulse-server` on
/// this machine (the server owns the hardware, as v2's local service did);
/// the board is its loopback client. Other computers are admitted only while
/// the operator has Remote on for it (v2 `admit_peers`), told on its stdin.
/// `narrate` takes the server's lifecycle lines as v2's device log wrote them
/// (its `zlc_pulse.remote` logger: `[HH:MM:SS] ZLC <event> ...`).
fn serve_locally(parameters: &Value, narrate: Box<dyn Fn(String) + Send>) -> Result<(Board, Served), String> {
    use std::io::BufRead;
    use std::os::windows::process::CommandExt;
    let server = std::env::current_exe().map_err(|error| error.to_string())?.with_file_name("zlc-pulse-server.exe");
    if !server.is_file() { return Err(format!("{} is not built", server.display())); }
    let mut arguments = vec!["--host".to_owned(), zlc_fabric::bind_address(), "--admission".to_owned(), "stdin".to_owned(),
                             "--backend".to_owned(), parameters["backend"].as_str().unwrap_or("auto").to_owned(),
                             "--port".to_owned(), parameters["port"].as_u64().unwrap_or(0).to_string()];
    if let Some(port) = parameters["uart_port"].as_str().filter(|port| !port.trim().is_empty()) { arguments.extend(["--uart-port".to_owned(), port.trim().to_owned()]); }
    // A console program, given no console window; its errors go where this application's go.
    let mut child = std::process::Command::new(&server).args(&arguments).stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::inherit())
        .creation_flags(windows_sys::Win32::System::Threading::CREATE_NO_WINDOW)
        .spawn().map_err(|error| error.to_string())?;
    let control = child.stdin.take();
    let stdout = child.stdout.take().ok_or("no server output")?;
    let (found, port) = mpsc::channel();
    std::thread::spawn(move || {
        let mut lines = std::io::BufReader::new(stdout).lines();
        for line in lines.by_ref().map_while(Result::ok) {
            if let Some(rest) = line.strip_prefix("{\"type\":\"pulse-server\",\"port\":") {
                let _ = found.send(rest.trim_end_matches('}').parse::<u16>().ok());
                break;
            }
            if let Some(line) = narration(&line) { narrate(line); }
        }
        for line in lines.map_while(Result::ok) { if let Some(line) = narration(&line) { narrate(line); } }
    });
    // A JTAG board waits for Vivado to come up and open its target first.
    let startup = if matches!(parameters["backend"].as_str(), Some("uart" | "memory")) { Duration::ZERO } else { zlc_pulse::axi::STARTUP_TIMEOUT };
    let port = match port.recv_timeout(Duration::from_secs(60) + startup) {
        Ok(Some(port)) => port,
        _ => { let _ = child.kill(); return Err("the local pulse server did not start (see its log)".into()); }
    };
    match Board::remote("127.0.0.1", port) {
        Ok(board) => Ok((board, Served { child, port, control })),
        Err(error) => { let _ = child.kill(); Err(error.message) }
    }
}

/// A server stdout line as a device log line: its `ZLC ...` narration, at this
/// computer's local time; anything else is not narration.
fn narration(line: &str) -> Option<String> {
    let message = line.split_once("] ").map_or(line, |(_, rest)| rest);
    if !message.starts_with("ZLC ") { return None; }
    let mut time = unsafe { std::mem::zeroed::<windows_sys::Win32::Foundation::SYSTEMTIME>() };
    unsafe { windows_sys::Win32::System::SystemInformation::GetLocalTime(&mut time) };
    Some(format!("[{:02}:{:02}:{:02}] {message}", time.wHour, time.wMinute, time.wSecond))
}

fn open_device(type_id: &str, parameters: &Value, narrate: Box<dyn Fn(String) + Send>) -> Result<(Board, Option<Served>), String> {
    let (board, child) = match type_id {
        "sequencer.virtual" => (Board::virtual_board(None).map_err(|error| error.message)?, None),
        "sequencer.hardware" => (Board::remote(parameters["host"].as_str().unwrap_or("127.0.0.1"),
                                               parameters["port"].as_u64().unwrap_or(zlc_pulse::remote::DEFAULT_PORT.into()) as u16)
                                     .map_err(|error| error.message)?, None),
        "sequencer.local" => { let (board, child) = serve_locally(parameters, narrate)?; (board, Some(child)) }
        other => return Err(format!("{other} is not a native sequencer type")),
    };
    if let Some(path) = parameters["config_file"].as_str().filter(|path| !path.trim().is_empty()) {
        if let Err(error) = board.config().load_file(Some(Path::new(path.trim()))) { let _ = board.close(); return Err(error.message); }
    }
    Ok((board, child))
}

/// What the board on this computer is announced as (v2 _announce_local): other computers open it there.
pub const ANNOUNCED_AS: &str = "sequencer.hardware";

/// The native device types as the workbench reads descriptors.
pub fn device_descriptors() -> Vec<Value> {
    let config_file = json!({"name": "config_file", "type": "str", "label": "Config file (optional)", "default": "",
                             "file_filter": "Pulse Config (*.json)",
                             "description": "a zlc.pulse.config_values file the board compiles Config fields against"});
    let output = json!([{"name": "sequence"}]);
    // One sequencer family, as v2's Device Manager offered it; the virtual board shares the
    // simulated world (the simulation settings reach it like every simulated device).
    let common = |type_id: &str, label: &str, description: &str, simulation: bool, parameters: Value| json!({
        "type_id": type_id, "kind": "device", "family": "sequencer", "label": label, "description": description, "module": "zlc_pulse (native)",
        "simulation": simulation, "capabilities": ["sequencer.streamer"], "parameters": parameters, "physical_outputs": output, "commands": [],
        "addable": true});
    let mut types = vec![
        common("sequencer.virtual", "Virtual sequencer", "The pulse board's memory twin, played at wall time into the simulated world \
                (simulation.board): the virtual apparatus' sequencer.", true, json!([config_file])),
        common("sequencer.hardware", "Pulse server", "The real pulse board, reached over the network through the pulse server that owns it.",
               false, json!([{"name": "host", "type": "str", "label": "Pulse server host", "default": "127.0.0.1"},
                                   {"name": "port", "type": "int", "label": "Pulse server port", "default": zlc_pulse::remote::DEFAULT_PORT, "minimum": 1, "maximum": 65535},
                                   config_file])),
        common("sequencer.local", "Pulse board on this computer", "The board plugged into this computer, served here by zlc-pulse-server \
                (its UART, or its JTAG cable through a resident Vivado); this bench is its loopback client.", false,
               json!([{"name": "backend", "type": "choice", "label": "Board transport", "default": "auto",
                       "choices": [{"value": "auto", "label": "Auto (UART, else JTAG)"}, {"value": "uart", "label": "UART"},
                                   {"value": "jtag-axi", "label": "JTAG-to-AXI (Vivado)"}, {"value": "memory", "label": "Memory mock (no hardware)"}]},
                      {"name": "uart_port", "type": "str", "label": "UART port (blank = probe)", "default": ""},
                      {"name": "port", "type": "int", "label": "Serve on port", "default": zlc_pulse::remote::DEFAULT_PORT, "minimum": 1, "maximum": 65535},
                      config_file])),
    ];
    for item in &mut types { if item["type_id"] == "sequencer.local" { item["announces"] = json!(ANNOUNCED_AS); } }
    // The board a row names (v2 pulse_board_identity): one pulse server serves one board, here or reached over the network.
    for (type_id, identity) in [("sequencer.hardware", "pulse-server:{host}:{port}"), ("sequencer.local", "pulse-server:127.0.0.1:{port}")] {
        if let Some(item) = types.iter_mut().find(|item| item["type_id"] == type_id) { item["identity"] = json!(identity); }
    }
    types
}

impl PulseHost {
    /// The editor's pulses folder is the workspace's; its value files live beside it (v2 Workspace.prepare made both).
    pub fn new(post: Post, safe_actions: SafeActions, plugins: Option<PluginHost>, pulses: &Path) -> PulseHost {
        let deployed = Deployed::load().map_err(|error| eprintln!("{}", json!({"type": "pulse-error", "error": error.message}))).ok();
        if let Some(workspace) = pulses.parent() { let _ = std::fs::create_dir_all(workspace.join(zlc_pulse::codec::CONFIG_VALUES_DIRECTORY)); }
        let python = zlc_plugin_host::plugin_python().ok();
        PulseHost { editor: Editor::new(deployed, pulses.to_path_buf()), device: Lane::new("pulse-command"), safe: Lane::new("pulse-safe"), program: Lane::new("pulse-program"),
                    preview: Lane::new("pulse-preview"), post, safe_actions, plugins, given: None, python, session: Arc::new(Mutex::new(None)),
                    dpr: 1.0, preview_busy: false, preview_key: String::new(), preview_wanted: false, frame: None, frames: 0,
                    axes: Vec::new(), pointers: Default::default(), pointer_busy: false, users: Default::default(), runs: BTreeMap::new() }
    }

    /// The pulse section of a state message: the editor's view plus the preview's frame.
    pub fn state(&mut self) -> Value {
        let _span = crate::trace::span("pulse.state", || json!({}));
        let mut state = self.editor.state();
        state["preview"]["panel"] = json!(PREVIEW_PANEL);
        state["preview"]["frame"] = self.frame.map_or(Value::Null, |(width, height, sequence)| json!({"width": width, "height": height, "sequence": sequence}));
        state["preview"]["axes"] = Value::Array(self.axes.clone());
        state["apparatus_device"] = self.given.as_ref().map_or(Value::Null, |given| json!(given.instance));
        state
    }

    /// A pulse section once it is sent: the editor takes its unchanged schedule cards back for the next one.
    pub fn recycle(&mut self, pulse: &mut Value) { self.editor.recycle(pulse); }

    /// The apparatus' sequencer device instance, when the board is one.
    pub fn apparatus_device(&self) -> Option<String> { self.given.as_ref().map(|given| given.instance.clone()) }

    fn world(&self) -> Option<Arc<dyn World>> {
        let host = self.plugins.clone()?;
        let instance = self.given.as_ref()?.world.clone()?;
        Some(Arc::new(PluginWorld { host, instance }))
    }

    /// Run the editor's jobs on their lanes; move Stop all's SAFE with the board.
    pub fn dispatch(&mut self) {
        // Every editor change ends here: the record is current before any later start is admitted.
        let users = self.given.as_ref().filter(|_| self.editor.drives()).map(|given| (given.instance.clone(), "PulseGUI".to_owned()));
        if let Ok(mut record) = self.users.lock() { *record = users.into_iter().collect(); }
        if let Some(board) = self.editor.take_board_change() {
            self.safe_actions.unregister(SAFE_OUTPUT);
            if let Some(board) = board {
                let held = board.clone();
                self.safe_actions.register(SAFE_OUTPUT, Arc::new(move || held.safe().map(|_| ()).map_err(|error| error.message)));
            }
        }
        if let Some(board) = self.editor.board() { board.set_world(self.world()); }
        for job in self.editor.take_jobs() {
            let post = self.post.clone();
            match job {
                Job::Device { work, done } => self.device.run(move || { let outcome = work(); done.set(); post(PulseEvent::Outcome(outcome)); }),
                Job::Safe { work } => self.safe.run(move || post(PulseEvent::Outcome(work()))),
                Job::Completion { work } => { let _ = std::thread::Builder::new().name("zlc-pulse-completion".into()).spawn(move || post(PulseEvent::Outcome(work()))); }
                Job::Program { source, fields } => {
                    let python = self.python.clone();
                    self.program.run(move || {
                        let table = run_scan_program(python.as_deref(), &source);
                        post(PulseEvent::Outcome(Outcome::Program { source, fields, table }));
                    });
                }
            }
        }
        self.refresh_preview();
    }

    /// Draw the preview again when its page is shown and what it draws changed.
    pub fn refresh_preview(&mut self) {
        if self.editor.page() != zlc_pulse_editor::editor::PREVIEW_PAGE && self.session.lock().map(|s| s.is_none()).unwrap_or(true) { return; }
        let Some(request) = self.editor.preview_request() else { return };
        let (sequence, include_off, pinned) = match request {
            Ok(request) => request,
            Err(message) => { self.editor.set_preview_status(message, "error"); return; }
        };
        let key = format!("{:?}|{include_off}|{pinned:?}|{}", zlc_pulse::codec::sequence_to_tree(&sequence).canonical(), self.dpr);
        if key == self.preview_key { return; }
        if self.preview_busy { self.preview_wanted = true; return; }
        self.preview_busy = true;
        self.editor.set_preview_status("drawing preview…", "task");
        let (session, post, dpr) = (self.session.clone(), self.post.clone(), self.dpr);
        self.preview.run(move || {
            let result = (|| -> Result<(Front, String, Vec<Value>), String> {
                let preview = zlc_pulse_editor::timeline::preview_candidate(&sequence, include_off, pinned.as_deref()).map_err(|e| e.message)?;
                let mut held = session.lock().map_err(|_| "preview lost")?;
                let failure = |error: zlc_plot::configuration::error::PlotError| error.to_string();
                match held.as_mut() {
                    None => *held = Some(PlotSession::pulse_timeline(preview.data.clone(), PlotLabels::default(), Some(&preview.size), None, dpr, None,
                                                                    services()).map_err(failure)?),
                    Some(open) => {
                        open.set_device_pixel_ratio(dpr).map_err(failure)?;
                        open.set_size(&preview.size).map_err(failure)?;
                        open.update_pulse(preview.data.clone(), None).map_err(failure)?;
                    }
                }
                let (front, axes) = drawn(held.as_mut().expect("drawn")).map_err(failure)?;
                let words = [zlc_pulse_editor::text::counted(preview.periods, "period"), zlc_pulse_editor::text::counted(preview.rows, "channel"),
                             zlc_pulse_editor::text::readable(preview.total_ns), preview.size.clone()].join(zlc_pulse_editor::text::SEPARATOR);
                Ok((front, words, axes))
            })();
            post(PulseEvent::Preview { result, key });
        });
    }

    /// A preview the lane drew: the front for Pixels, the words for the page.
    pub fn preview_drawn(&mut self, result: Result<(Front, String, Vec<Value>), String>, key: String) -> Option<(u64, Front)> {
        self.preview_busy = false;
        let drawn = match result {
            Ok((front, words, axes)) => {
                self.preview_key = key;
                self.editor.set_preview_status(words, "task");
                Some(self.presented(front, axes))
            }
            Err(error) => { self.preview_key = key; self.editor.set_preview_status(format!("cannot draw this pulse: {error}"), "error"); None }
        };
        if std::mem::take(&mut self.preview_wanted) { self.refresh_preview(); }
        drawn
    }

    fn presented(&mut self, front: Front, axes: Vec<Value>) -> (u64, Front) {
        self.frames += 1;
        self.frame = Some((front.width, front.height, self.frames));
        self.axes = axes;
        (self.frames, front)
    }

    /// A pointer message on the preview (the page sends it as a panel's,
    /// `panel: "pulse-preview"`): zoom, pan and the selectors of the drawn
    /// timeline, in order on the preview's lane, while the page's Selectors
    /// switch is on (v2 `set_interaction_enabled`; off, the picture takes no
    /// input). A newer move replaces one not yet handled.
    pub fn pointer(&mut self, command: Value) {
        let gesture_end = matches!(command["event"].as_str(), Some("release" | "cancel" | "leave"));
        if !self.editor.preview_selectors() && !gesture_end { return; }
        if command["event"] == "move" && self.pointers.back().is_some_and(|last| last["event"] == "move") { self.pointers.pop_back(); }
        self.pointers.push_back(command);
        self.next_pointer();
    }

    fn next_pointer(&mut self) {
        if self.pointer_busy { return; }
        let Some(command) = self.pointers.pop_front() else { return };
        self.pointer_busy = true;
        let (session, post) = (self.session.clone(), self.post.clone());
        self.preview.run(move || {
            let result = session.lock().map_err(|_| "preview lost".to_owned()).and_then(|mut held| {
                let Some(open) = held.as_mut() else { return Ok(None) };
                let failure = |error: zlc_plot::configuration::error::PlotError| error.to_string();
                let event = zlc_plot::document::pointer_event(&command, None).map_err(failure)?;
                if !open.raster_pointer_event(&event).map_err(failure)? { return Ok(None); }
                drawn(open).map(Some).map_err(failure)
            });
            post(PulseEvent::Gesture(result));
        });
    }

    /// The lane handled a pointer message: a changed picture goes to Pixels.
    pub fn gestured(&mut self, result: Result<Option<(Front, Vec<Value>)>, String>) -> Option<(u64, Front)> {
        self.pointer_busy = false;
        let presented = match result {
            Ok(Some((front, axes))) => Some(self.presented(front, axes)),
            Ok(None) => None,
            Err(error) => { self.editor.set_preview_status(format!("preview gesture: {error}"), "error"); None }
        };
        self.next_pointer();
        presented
    }

    /// `save_preview_image`: the drawn preview as PNG, beside the pulse.
    fn save_preview(&mut self, path: Option<String>) -> Result<Value, String> {
        if self.session.lock().map(|s| s.is_none()).unwrap_or(true) { self.editor.warn("There is no preview to save"); return Ok(Value::Null); }
        if self.preview_busy {
            self.editor.set_preview_status("preview is busy; save again when the drawing is ready", "warning");
            return Ok(Value::Null);
        }
        let name = self.editor.sequence().map_or_else(|| "pulse".to_owned(), |s| if s.name.is_empty() { "pulse".into() } else { s.name.clone() });
        let target = path.map(PathBuf::from).unwrap_or_else(|| tables::unique_path(&self.editor_save_folder(), &name, ".png"));
        let (session, post) = (self.session.clone(), self.post.clone());
        self.editor.set_preview_status("saving preview…", "task");
        self.preview.run(move || {
            let result = session.lock().map_err(|_| "preview lost".to_owned()).and_then(|mut held| {
                let bytes = held.as_mut().ok_or("There is no preview to save")?.save_image("png").map_err(|error| error.to_string())?;
                zlc_pulse::codec::atomic_write(&target, &bytes).map_err(|error| error.message)?;
                Ok(target)
            });
            post(PulseEvent::PreviewSaved(result));
        });
        Ok(json!({"accepted": true}))
    }

    fn editor_save_folder(&self) -> PathBuf { PathBuf::from(self.editor.state_folder()) }

    pub fn preview_saved(&mut self, result: Result<PathBuf, String>) {
        match result {
            Ok(path) => self.editor.set_preview_status(format!("saved {}", path.display()), "task"),
            Err(error) => self.editor.warn(format!("Cannot save the preview: {error}")),
        }
    }

    /// A pulse command a plugin run sent must be one the run may give.
    /// A run that names a run (`_run`) may drive only the sequencer it claims.
    fn check_claims(&self, run: Option<&str>, claims: impl Fn(&str) -> Vec<String>) -> Result<(), String> {
        let Some(run) = run else { return Ok(()) };
        let held = self.given.as_ref().is_some_and(|given| claims(run).contains(&given.instance));
        if held { return Ok(()); }
        Err(match &self.given {
            None => format!("run {run} cannot drive the pulse board: no sequencer device is open in the apparatus"),
            Some(given) => format!("run {run} does not claim the sequencer {}; its node must require sequencer.streamer", given.instance),
        })
    }

    /// One `pulse_*` command, whoever sent it; `dialog` asks the operator for a path.
    pub fn command(&mut self, action: &str, command: &Value, operator: bool, request: u64, claims: impl Fn(&str) -> Vec<String>,
                   claimant: Option<String>, dialog: impl Fn(&str, &str, bool, &Path, &Path) -> Result<Option<String>, String>) -> Handled {
        self.editor.set_claimed_by(claimant.clone());
        // Published, the board is the other computer's: nothing here plays it (Stop still makes it safe).
        let drives = matches!(action, "pulse_fire" | "pulse_hold" | "pulse_step" | "pulse_run" | "pulse_wait_done" | "pulse_sync" | "pulse_estimate");
        if drives && self.given.as_ref().is_some_and(|given| given.published) { return Handled::Now(Err(PUBLISHED_BOARD.into())); }
        let run = command["_run"].as_str();
        let mut args: Map<String, Value> = crate::remote::parameters(command).as_object().cloned().unwrap_or_default();
        match action {
            "pulse_describe" => return Handled::Now(describe_document(Path::new(command["document"].as_str().unwrap_or("")))
                .map(|description| to_value(&description)).map_err(|error| error.message)),
            "pulse_run" | "pulse_wait_done" | "pulse_stop" if run.is_some() => {
                if let Err(error) = self.check_claims(run, &claims) { return Handled::Now(Err(error)); }
            }
            // A run holding the board is the only client that may play on it or take its reports (v2 device use).
            "pulse_run" | "pulse_wait_done" => if let Some(holder) = &claimant { return Handled::Now(Err(format!("device is in use by {holder}"))); },
            _ => {}
        }
        match action {
            "pulse_run" => {
                let Some(board) = self.editor.board().cloned() else { return Handled::Now(Err("no pulse board is attached".into())) };
                let stage = command["stage"].as_str().unwrap_or("run").to_owned();
                if stage != "run" && run.is_none() { return Handled::Now(Err("stage load and fire are a run's two-step play".into())); }
                board.set_world(self.world());
                let document = PathBuf::from(command["document"].as_str().unwrap_or(""));
                let values = api_values(&command["api_values"]);
                let scan = command["scan"].is_object().then(|| ScanRows { fields: strings(&command["scan"]["fields"]), rows: rows(&command["scan"]["rows"]) });
                let (runs, sweeps) = (command["run_repeats"].as_u64().unwrap_or(1), command["scan_repeats"].as_u64().unwrap_or(1));
                let owner = run.map(str::to_owned);
                let used = owner.as_ref().map(|run| self.runs.entry(run.clone()).or_default());
                let (pinned, loaded) = used.map_or((None, None), |used| {
                    used.in_flight += 1;
                    (used.pinned.get(&zlc_pulse::codec::absolute(&document)).cloned(), used.loaded.clone())
                });
                let post = self.post.clone();
                self.device.run(move || {
                    let (result, pin, fired, loaded) = play(&board, &stage, &document, pinned.as_deref(), loaded.as_deref(), &values, scan.as_ref(), runs, sweeps);
                    let run = owner.map(|run| RunReply { run, pinned: pin, fired, loaded });
                    post(PulseEvent::Reply { request, result, touched: true, run });
                });
                return Handled::Later;
            }
            // A run's Stop answers with what the board said (v2 nodes' sequencer.safe()), before the
            // run goes on; the editor's and a client's Stop is the editor's.
            "pulse_stop" if run.is_some() => {
                let Some(board) = self.editor.board().cloned() else { return Handled::Now(Err("no pulse board is attached".into())) };
                let post = self.post.clone();
                self.safe.run(move || {
                    let result = board.safe().map(|readback| json!({"status": readback.status, "command_id": readback.command_id}))
                        .map_err(|error| format!("the board did not go SAFE: {}", error.message));
                    post(PulseEvent::Reply { request, result, touched: true, run: None });
                });
                return Handled::Later;
            }
            "pulse_wait_done" => {
                let Some(board) = self.editor.board().cloned() else { return Handled::Now(Err("no pulse board is attached".into())) };
                let id = command["command_id"].as_u64().unwrap_or(0) as u32;
                let timeout = Duration::from_secs_f64(command["timeout_seconds"].as_f64().unwrap_or(0.0).clamp(0.0, 600.0));
                let post = self.post.clone();
                let _ = std::thread::Builder::new().name("zlc-pulse-wait".into()).spawn(move || {
                    let result = board.wait_report(id, Some(timeout)).map(|report| to_value(&done_json(report.as_ref()))).map_err(|error| error.message);
                    post(PulseEvent::Reply { request, result, touched: true, run: None });
                });
                return Handled::Later;
            }
            "pulse_estimate" => {
                let Some(board) = self.editor.board().cloned() else {
                    return Handled::Now(Err("no pulse board is attached: an estimate compiles for the board's geometry and Config".into()));
                };
                let document = PathBuf::from(command["document"].as_str().unwrap_or(""));
                let points: Vec<_> = command["api_points"].as_array().map(|points| points.iter().map(api_values).collect()).unwrap_or_default();
                let axes = command["scan"].is_object().then(|| ScanAxes { fields: strings(&command["scan"]["fields"]), columns: rows(&command["scan"]["columns"]) });
                let post = self.post.clone();
                let _ = std::thread::Builder::new().name("zlc-pulse-estimate".into()).spawn(move || {
                    let result = estimate_document(&board, &document, &points, axes.as_ref()).map(|estimate| to_value(&estimate)).map_err(|error| error.message);
                    post(PulseEvent::Reply { request, result, touched: false, run: None });
                });
                return Handled::Later;
            }
            // v2 save_preview_image: the drawing as it is, beside the pulse under a fresh name (no dialog).
            "pulse_preview" if command["save"].as_bool() == Some(true) => return Handled::Now(self.save_preview(command["path"].as_str().map(str::to_owned))),
            "pulse_ready" => { self.dpr = command["dpr"].as_f64().filter(|dpr| *dpr > 0.0).unwrap_or(1.0); self.dispatch(); return Handled::Now(Ok(Value::Null)); }
            _ => {}
        }
        // The window asks where a file goes when its command does not say.
        if operator {
            // v2: Save Pulse always asks, at the file it has (a Save as… of Value Config, or a first save, likewise).
            if let Some((title, filter, save, start, home)) = self.editor.file_dialog(action, &args) {
                match dialog(title, filter, save, &start, &home) {
                    Ok(Some(path)) => { args.insert("path".into(), json!(path)); }
                    Ok(None) => return Handled::Now(Ok(Value::Null)),
                    Err(error) => return Handled::Now(Err(error)),
                }
            }
        }
        let answer = self.editor.command(action, &args, operator);
        // Selectors off ends a drag in progress (v2 cancelled the active interaction).
        if action == "pulse_preview" && args.get("selectors") == Some(&Value::Bool(false)) { self.pointer(json!({"event": "cancel", "x": 0.0, "y": 0.0})); }
        self.dispatch();
        Handled::Now(answer)
    }

    /// A reply arrived: what a run's pulse_run leaves it holding is kept, and
    /// anything that drove the board makes the editor ask it again.
    pub fn replied(&mut self, touched: bool, run: Option<RunReply>) {
        if let Some(RunReply { run, pinned, fired, loaded }) = run {
            let used = self.runs.entry(run.clone()).or_default();
            used.in_flight = used.in_flight.saturating_sub(1);
            if let Some((path, sequence)) = pinned { used.pinned.entry(path).or_insert(sequence); }
            if fired.is_some() { used.fire = fired; }
            if loaded.is_some() { used.loaded = loaded; }
            if used.ended && used.in_flight == 0 { self.settle_run(&run); }
        }
        if touched { self.editor.board_touched(); self.dispatch(); }
    }

    /// The runs that have ended (`active`: whether the workbench still runs one) are
    /// let go once their replies are in: see [`Self::settle_run`].
    pub fn settle_runs(&mut self, active: impl Fn(&str) -> bool) {
        let ended: Vec<String> = self.runs.iter_mut().filter(|(run, used)| !used.ended && !active(run))
            .map(|(run, used)| { used.ended = true; run.clone() }).collect();
        for run in ended { if self.runs.get(&run).is_some_and(|used| used.in_flight == 0) { self.settle_run(&run); } }
    }

    /// A run is over -- its node returned, failed or was stopped, or its process
    /// died: its pins go, and a FIRE of its that is still unreported (a run that
    /// ended without its own Stop) is made SAFE and its report settled, as each
    /// v2 node's finally did.
    fn settle_run(&mut self, run: &str) {
        let Some(used) = self.runs.remove(run) else { return };
        let (Some(fire), Some(board)) = (used.fire, self.editor.board().cloned()) else { return };
        if board.last_fire() != Some((fire, Fate::Pending)) { return; }
        let (post, instance, run) = (self.post.clone(), self.given.as_ref().map(|given| given.instance.clone()), run.to_owned());
        self.safe.run(move || {
            let outcome = match board.safe() {
                Ok(_) => format!("ZLC SAFE after {run}: it ended with FIRE {fire} unreported"),
                Err(error) => format!("ZLC SAFE FAILED after {run} ended with FIRE {fire} unreported: {}", error.message),
            };
            if let Some((instance, line)) = instance.zip(narration(&outcome)) { post(PulseEvent::Log { instance, line }); }
            post(PulseEvent::Touched);
        });
    }

    // ------------------------------------------------------------- native devices
    /// The apparatus opens a sequencer device: on the device lane.
    pub fn open_device(&mut self, instance: &str, type_id: &str, parameters: &Value, context: &Value) {
        let post = self.post.clone();
        if let Some(given) = &self.given {
            post(PulseEvent::Opened { instance: instance.to_owned(), result: Err(format!("the pulse board is already attached as {}", given.instance)) });
            return;
        }
        let (instance, type_id, parameters) = (instance.to_owned(), type_id.to_owned(), parameters.clone());
        let world = (type_id == "sequencer.virtual").then(|| (self.plugins.clone(), world_instance(&instance), context["simulation"].clone()));
        self.device.run(move || {
            // The world opens first: a virtual board without one would play into nothing.
            if let Some((plugins, name, group)) = &world {
                let opened = plugins.as_ref().ok_or_else(|| "the virtual sequencer needs the plugin host for its world".to_owned())
                    .and_then(|host| host.open(name, "simulation.board", json!({}), if group.is_object() { group.clone() } else { json!({}) })
                        .map_err(|error| format!("its world side: {error}")));
                if let Err(error) = opened { post(PulseEvent::Opened { instance, result: Err(error) }); return; }
            }
            let (logs, name) = (post.clone(), instance.clone());
            let narrate = Box::new(move |line| logs(PulseEvent::Log { instance: name.clone(), line }));
            let result = open_device(&type_id, &parameters, narrate).map(|(board, child)| (Arc::new(board), child));
            if result.is_err() { if let Some((Some(host), name, _)) = &world { let _ = host.close(name); } }
            post(PulseEvent::Opened { instance, result });
        });
    }

    /// The open finished: the editor shows the session's board, locked, under the device's
    /// `label` (v2 set_device_label: its role).
    pub fn device_opened(&mut self, instance: &str, label: &str, result: Result<(Arc<Board>, Option<Served>), String>) -> Result<Value, String> {
        let (board, child) = result?;
        if self.given.is_some() { let _ = board.close(); return Err("the pulse board is already attached".into()); }
        let description = board.description().clone();
        let world = (board.is_virtual() && self.plugins.is_some()).then(|| world_instance(instance));
        self.given = Some(Given { instance: instance.to_owned(), child, world, published: false });
        self.editor.attach_given(board, label);
        self.dispatch();
        Ok(json!({"board": self.editor.board().map(|b| b.name().to_owned()), "ports": description.target.ports.len(),
                  "clock_hz": description.clock_hz}))
    }

    pub fn close_device(&mut self, instance: &str) {
        let post = self.post.clone();
        if self.given.as_ref().is_none_or(|given| given.instance != instance) {
            post(PulseEvent::Closed { instance: instance.to_owned(), error: None });
            return;
        }
        self.let_go(instance, true);
    }

    /// The board's computer is gone (the workbench holds the device failed): the editor lets it go, nobody is told it closed.
    pub fn lose_device(&mut self, instance: &str) {
        if self.given.as_ref().is_some_and(|given| given.instance == instance) { self.let_go(instance, false); }
    }

    fn let_go(&mut self, instance: &str, closed: bool) {
        let post = self.post.clone();
        let given = self.given.take().expect("checked");
        let board = self.editor.board().cloned();
        self.editor.detach_given();
        self.dispatch();
        let instance = instance.to_owned();
        let plugins = self.plugins.clone();
        self.device.run(move || {
            let error = board.and_then(|board| board.close().err()).map(|error| error.message);
            if let Some(served) = given.child { served.end(); }
            if let (Some(host), Some(world)) = (plugins, &given.world) { let _ = host.close(world); }
            if closed { post(PulseEvent::Closed { instance, error }); }
        });
    }

    /// Closing the window: SAFE, then let every connection go.
    pub fn shutdown(&mut self) {
        if let Some(board) = self.editor.board().cloned() { let _ = board.close(); }
        if let Some(mut given) = self.given.take() { if let Some(served) = given.child.take() { served.end(); } }
        self.safe_actions.unregister(SAFE_OUTPUT);
    }

    /// Remote on/off for the board on this computer (v2: a local pulse board's
    /// own server admits peers while published). Off drops the other computer
    /// (its server drives the board SAFE) and this application takes it back.
    /// Answers what the fabric announces: the board's server port.
    pub fn admit_peers(&mut self, instance: &str, published: bool) -> Result<u16, String> {
        use std::io::Write;
        let given = self.given.as_mut().filter(|given| given.instance == instance).ok_or_else(|| format!("{instance} is not the open pulse board"))?;
        let served = given.child.as_mut().ok_or_else(|| format!("{instance} is not a board plugged into this computer: other computers reach it at its own pulse server"))?;
        let control = served.control.as_mut().ok_or("the local pulse server no longer listens to this application")?;
        control.write_all(if published { b"admit\n" } else { b"refuse\n" }).and_then(|()| control.flush())
            .map_err(|error| format!("the local pulse server did not take the change: {error}"))?;
        given.published = published;
        let port = served.port;
        if !published {
            if let Some(board) = self.editor.board().cloned() {
                let post = self.post.clone();
                self.device.run(move || {
                    let result = board.rejoin().map(|()| json!({})).map_err(|error| error.message);
                    post(PulseEvent::Reply { request: 0, result, touched: true, run: None });
                });
            }
        }
        Ok(port)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::{Receiver, channel};

    /// A pulse host with the virtual board as the apparatus' sequencer, its lanes' events in a
    /// channel, and a pulse document of its own (the run claims "sequencer").
    fn host(name: &str) -> (PulseHost, Receiver<PulseEvent>, PathBuf) {
        let folder = std::env::temp_dir().join(format!("zlc_pulse_host_{}_{name}", std::process::id()));
        std::fs::create_dir_all(folder.join("pulses")).unwrap();
        let (sender, events) = channel();
        let sender = Mutex::new(sender);
        let post: Post = Arc::new(move |event| { let _ = sender.lock().unwrap().send(event); });
        let mut host = PulseHost::new(post, SafeActions::new(), None, &folder.join("pulses"));
        let board = Arc::new(Board::virtual_board(None).unwrap());
        host.device_opened("sequencer", "sequencer", Ok((board, None))).unwrap();
        let document = folder.join("pulses").join("shot.json");
        std::fs::copy(Path::new(env!("CARGO_MANIFEST_DIR")).join("../zlc-pulse/tests/pulses/imaging_template.json"), &document).unwrap();
        (host, events, document)
    }

    /// One command, answered now or by a lane (the editor's outcomes delivered meanwhile).
    fn ask(host: &mut PulseHost, events: &Receiver<PulseEvent>, action: &str, command: Value, claimant: Option<&str>) -> Result<Value, String> {
        let handled = host.command(action, &command, false, 1, |_| vec!["sequencer".to_owned()], claimant.map(str::to_owned), |_, _, _, _, _| Ok(None));
        match handled {
            Handled::Now(answer) => answer,
            Handled::Later => loop {
                match events.recv_timeout(Duration::from_secs(30)).expect("an answer") {
                    PulseEvent::Reply { result, touched, run, .. } => { host.replied(touched, run); return result; }
                    PulseEvent::Outcome(outcome) => { host.editor.deliver(outcome); host.dispatch(); }
                    _ => {}
                }
            },
        }
    }

    fn played(document: &Path, run: &str, repeats: u64) -> Value {
        json!({"document": document.display().to_string(), "run_repeats": repeats, "_run": run})
    }

    fn board(host: &PulseHost) -> Arc<Board> { host.editor.board().unwrap().clone() }

    /// A run's record names one pulse: a later fire of its file changed meanwhile is refused;
    /// another run plays the new one.
    #[test]
    fn a_run_plays_the_pulse_it_began_with() {
        let (mut host, events, document) = host("pinned");
        let receipt = ask(&mut host, &events, "pulse_run", played(&document, "scan#1", 1), None).unwrap();
        let id = receipt["command_id"].as_u64().unwrap();
        let done = ask(&mut host, &events, "pulse_wait_done", json!({"command_id": id, "timeout_seconds": 30, "_run": "scan#1"}), None).unwrap();
        assert_eq!(done["status"], "done");
        let mut tree: Value = serde_json::from_str(&std::fs::read_to_string(&document).unwrap()).unwrap();
        tree["periods"][1]["duration"] = json!(0.03);
        std::fs::write(&document, tree.to_string()).unwrap();
        let refused = ask(&mut host, &events, "pulse_run", played(&document, "scan#1", 1), None).unwrap_err();
        assert!(refused.contains(PULSE_CHANGED), "{refused}");
        let other = ask(&mut host, &events, "pulse_run", played(&document, "scan#2", 0), None).unwrap();
        assert!(other["command_id"].is_u64());
        ask(&mut host, &events, "pulse_stop", json!({"_run": "scan#2"}), None).unwrap();
    }

    /// v2 device use: while a run holds the board, no other client plays on it, takes its
    /// reports or stops it.
    #[test]
    fn only_the_holding_run_drives_the_board() {
        let (mut host, events, document) = host("claimed");
        let claimant = Some("Calibration");
        for (action, command) in [("pulse_wait_done", json!({"command_id": 1, "timeout_seconds": 0})),
                                  ("pulse_run", json!({"document": document.display().to_string()})), ("pulse_stop", json!({}))] {
            assert_eq!(ask(&mut host, &events, action, command, claimant).unwrap_err(), "device is in use by Calibration", "{action}");
        }
    }

    /// v2 nodes' sequencer.safe(): a run's Stop answers once the board is SAFE.
    #[test]
    fn a_runs_stop_answers_with_the_safe() {
        let (mut host, events, document) = host("stop");
        let id = ask(&mut host, &events, "pulse_run", played(&document, "cal#1", 0), None).unwrap()["command_id"].as_u64().unwrap() as u32;
        let answer = ask(&mut host, &events, "pulse_stop", json!({"_run": "cal#1"}), None).unwrap();
        assert!(answer["status"].is_u64(), "{answer}");
        assert_eq!(board(&host).last_fire(), Some((id, Fate::Safed)));
    }

    /// A run that ends with its FIRE unreported -- its process died, say -- leaves the board SAFE.
    #[test]
    fn a_run_ended_without_its_stop_leaves_the_board_safe() {
        let (mut host, events, document) = host("ended");
        let id = ask(&mut host, &events, "pulse_run", played(&document, "scan#1", 0), None).unwrap()["command_id"].as_u64().unwrap() as u32;
        host.settle_runs(|_| true);
        assert_eq!(board(&host).last_fire(), Some((id, Fate::Pending)), "still running");
        host.settle_runs(|_| false);
        loop {
            if let PulseEvent::Touched = events.recv_timeout(Duration::from_secs(30)).expect("the SAFE") { break; }
        }
        assert_eq!(board(&host).last_fire(), Some((id, Fate::Safed)));
    }

    /// v2 slm_rearrangement: the Pulse is loaded first and the FIRE is then the FIRE alone.
    #[test]
    fn a_run_loads_then_fires() {
        let (mut host, events, document) = host("stages");
        let mut load = played(&document, "rearrangement#1", 1);
        load["stage"] = json!("load");
        let loaded = ask(&mut host, &events, "pulse_run", load, None).unwrap();
        assert!(loaded.get("command_id").is_none() && loaded["timing"]["periods"].is_array(), "{loaded}");
        assert_eq!(board(&host).last_fire(), None, "loaded, not fired");
        let mut fire = played(&document, "rearrangement#1", 1);
        fire["stage"] = json!("fire");
        let fired = ask(&mut host, &events, "pulse_run", fire, None).unwrap();
        let id = fired["command_id"].as_u64().unwrap();
        let done = ask(&mut host, &events, "pulse_wait_done", json!({"command_id": id, "timeout_seconds": 30, "_run": "rearrangement#1"}), None).unwrap();
        assert_eq!(done["status"], "done");
    }
}
