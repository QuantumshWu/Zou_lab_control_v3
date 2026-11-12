//! The pulse board and the Pulse Editor (docs/PULSE_V3.md). The board's own commands -- a plugin
//! run's play, report wait, Stop, estimate, a document's description and a lane's schedule -- are
//! answered where they arrive by the [`PulseDesk`], off the window's turn: a run's commands are held
//! to the claims the window last stated, and the window hears only a coalesced notice that what it
//! shows of the board may have changed. The editor lives on the window's turn ([`PulseHost`]); its
//! jobs run on lanes off it (device: one at a time in order, the board's and the pulse files'; SAFE
//! beside it; a finite run's completion wait; the operator's scan program in the science
//! interpreter; the preview's drawing), its files are read ahead and written off it, and its view is
//! made only when something it shows changed. The board's SAFE is one of Stop all's safe actions,
//! and the sequencer apparatus devices (`sequencer.virtual|hardware|local`) are native: opening one
//! attaches the board the editor and the runs share.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, mpsc};
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
use zlc_pulse::run::{ScanAxes, ScanRows, apply_run_plan, describe_document, done_json, estimate_document, fire_loaded, load_sequence, read_document, run_sequence};
use zlc_pulse::json::Json;
use zlc_pulse::model::Sequence;
use zlc_pulse::scan::ScanTable;
use zlc_pulse::scan_plan::{ScanPlanRequest, effective_inner, prepare_application};
use zlc_pulse::transport::Event;
use zlc_pulse_editor::editor::{Answer, Deployed, Editor, Writes};
use zlc_pulse_editor::session::{Job, Outcome};
use zlc_pulse_editor::{tables, to_value};
use zlc_plugin_host::{HostEvent, PluginHost};

use crate::science::Front;

/// The Pixels panel id the Preview page's timeline is delivered as.
pub const PREVIEW_PANEL: &str = "pulse-preview";
/// Stop all's name for the board's SAFE.
pub const SAFE_OUTPUT: &str = "pulse-streamer";

/// What a lane, the desk or a worker tells the window's turn.
pub enum PulseEvent {
    Outcome(Outcome),
    /// The answer to a board command the window or a client of the bus gave.
    Reply { request: u64, result: Result<Value, String> },
    /// What the window shows of the board may have changed (coalesced: one notice until it is taken).
    Touched,
    /// The file an editor command reads was read ahead: the command runs now.
    Read(Box<ReadAhead>),
    /// The files an editor command wrote are on the disk (or are not).
    Written(Box<Written>),
    /// A Subpulse file the open pulse resolves changed on the disk.
    FilesChanged,
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
    /// Closing the window: the board is SAFE and every connection let go (`PulseHost::shutdown`).
    ShutDown,
}

/// An editor command waiting for the file it reads.
pub struct ReadAhead { action: String, args: Map<String, Value>, operator: bool, request: u64, path: PathBuf }

/// An editor command's files, written: its answer once they are.
pub struct Written { request: u64, writes: Writes, result: Result<(), String>, answer: Answer }

pub type Post = Arc<dyn Fn(PulseEvent) + Send + Sync>;

fn lock<T>(held: &Mutex<T>) -> MutexGuard<'_, T> { held.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) }

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


/// One `pulse_run` stage of a run (or a client): `load` (LOAD only, v2
/// `arm_sequencer`), `fire` (FIRE what that LOAD put on the board) or both.
fn play(board: &Board, stage: &str, document: &Path, pinned: Option<&Sequence>, loaded: Option<&Value>, values: &[(String, (f64, String))],
        scan: Option<&ScanRows>, native: Option<&ScanPlanRequest>, runs: u64, sweeps: u64) -> (Result<Value, String>, Option<(PathBuf, Arc<Sequence>)>, Option<u32>, Option<Arc<Value>>) {
    let fired = |receipt: &Json| receipt.get("command_id").and_then(Json::as_int).map(|id| id as u32);
    if stage == "fire" {
        let Some(loaded) = loaded else { return (Err("stage fire plays what this run's stage load put on the board; load it first".into()), None, None, None) };
        let expected = &loaded["applied"]["program"];
        let digest = expected["digest"].as_str().expect("native LOAD receipt has a digest");
        let held = match board.applied() { Ok(Some(held)) => held, Ok(None) => return (Err("the board no longer holds the loaded pulse".into()), None, None, None),
            Err(error) => return (Err(error.message), None, None, None) };
        if held.program.digest() != digest || json!(&*held.rows) != expected["rows"] {
            return (Err("the board no longer holds the program and scan table this run loaded; load it again".into()), None, None, None);
        }
        return match fire_loaded(board, digest, runs, sweeps) {
            Ok(receipt) => {
                let id = fired(&receipt);
                let mut receipt = to_value(&receipt);
                // Named columns and API inputs belong to the accepted LOAD; FIRE
                // supplies its own execution identity, applied state and timing.
                for field in ["scan", "api_values"] { if let Some(value) = loaded.get(field) { receipt[field] = value.clone(); } }
                (Ok(receipt), None, id, None)
            }
            Err(error) => (Err(error.message), None, None, None),
        };
    }
    let (path, sequence) = match read_document(document) { Ok(read) => read, Err(error) => return (Err(error.message), None, None, None) };
    if pinned.is_some_and(|pinned| *pinned != sequence) {
        let name = path.file_name().map_or_else(|| path.display().to_string(), |name| name.to_string_lossy().into_owned());
        return (Err(format!("the pulse {name} {PULSE_CHANGED}: the run's record names the pulse it began with, so the changed file is not played; \
                             start the run again to play it")), None, None, None);
    }
    let played = if let Some(native)=native {
        prepare_application(&sequence,native,values,&board.description().geometry)
            .and_then(|plan|apply_run_plan(board,&plan,(stage!="load").then_some((runs,sweeps))))
    } else if stage == "load" { load_sequence(board, &sequence, values, scan) } else { run_sequence(board, &sequence, values, scan, runs, sweeps) };
    match played {
        Err(error) => (Err(error.message), None, None, None),
        Ok(receipt) => {
            let loaded = (stage == "load").then(|| Arc::new(to_value(&receipt)));
            let id = fired(&receipt);
            (Ok(to_value(&receipt)), Some((path, Arc::new(sequence))), id, loaded)
        }
    }
}

/// Full applied facts and quantized rows remain native. The plugin gets only
/// the receipt identity, Config witness and table dimensions it needs to proceed.
fn accept_native_receipt(plugins: &PluginHost, run: &str, handle: &str, request: &ScanPlanRequest,
                         board: &Board, receipt: &mut Value) -> Result<(), String> {
    let held = board.applied().map_err(|e|e.message)?.ok_or("the board has no applied native scan")?;
    let sequence = held.source.as_ref().ok_or("the board has no applied pulse source")?;
    let played = receipt["scan"].is_object().then(|| ScanRows { fields: strings(&receipt["scan"]["fields"]), rows: rows(&receipt["scan"]["rows"]) });
    let effective = effective_inner(sequence, &request.plan, played.as_ref()).map_err(|e|e.message)?;
    crate::plugin_services::accept_scan_pulse(plugins, run, handle, effective, receipt["applied"].clone()).map_err(|e|e.to_string())?;
    let mut compact = json!({"applied":{"config":receipt["applied"]["config"],
        "program":{"digest":receipt["applied"]["program"]["digest"]}},
        "scan":{"handle":handle,"outer_index":request.outer_index,"fields":played.as_ref().map(|p|p.fields.clone()).unwrap_or_default(),
                "row_count":request.plan.inner_count()}});
    for key in ["command_id","api_values"] { if let Some(value) = receipt.get(key) { compact[key] = value.clone(); } }
    *receipt = compact;
    Ok(())
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

/// The virtual sequencer's world side: one plugin instance it opens and closes with itself, so
/// the apparatus has one virtual sequencer row as in v2 (whose sequencer.virtual drove the world).
fn world_instance(instance: &str) -> String { format!("{instance}_world") }


// ---------------------------------------------------------------- the desk
/// A plugin run's use of the board: its name, each pulse document as its first fire read it (a
/// later fire of a changed file is refused: a run's record names one pulse), the FIRE it sent
/// last, the program its `load` stage put on the board, its `pulse_run`s still on the lane, and
/// whether it has ended.
#[derive(Default)]
struct RunUse { name: String, pinned: BTreeMap<PathBuf, Arc<Sequence>>, fire: Option<u32>, loaded: Option<Arc<Value>>, in_flight: usize, ended: bool }

/// The words a refused fire of a changed pulse file says (a node ends its batch on them).
pub const PULSE_CHANGED: &str = "changed during this run";

/// What the window last stated of the workbench's runs, by their node's plugin instance: each
/// active run's name and the devices it claims (a run's board commands are held to them).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Claims(pub BTreeMap<String, (String, Vec<String>)>);

/// The apparatus device the board is, as the desk needs it: its instance, its world's plugin
/// instance (the virtual board's), and whether another computer drives it now (Remote on).
#[derive(Clone)]
struct Device { instance: String, world: Option<String>, published: bool }

/// Who asked a board command.
pub enum Who {
    /// A plugin run: the plugin host's id of it and its node's plugin instance.
    Run { hosted: String, instance: String },
    /// The window or a client of the bus; `claimant`: the run that holds the board, if one does.
    Client { claimant: Option<String> },
}

/// Where an answer goes: the asking run's request, or the window's turn.
pub type Reply = Box<dyn FnOnce(Result<Value, String>) + Send>;

/// One board command, as asked.
struct Asked { who: Who, action: String, command: Value, reply: Reply }

/// What a watch thread hears: the board changed, or its run (or its board) is gone.
enum Heard { Changed, Ended }

/// One run's watch of one lane's schedule: the plugin host's id of the run, the watch's name.
#[derive(Clone)]
struct Watch { lane: String, run: String, id: String }

/// A board command that drives the board: refused while another computer drives it.
fn drives(action: &str) -> bool {
    matches!(action, "pulse_fire" | "pulse_hold" | "pulse_step" | "pulse_run" | "pulse_wait_done" | "pulse_sync" | "pulse_estimate")
}

const PUBLISHED_BOARD: &str = "the pulse board is published on the bench fabric: another computer drives it -- withdraw Remote to use it here";

/// The board commands a plugin run gives that the desk answers.
const RUN_COMMANDS: [&str; 6] = ["pulse_run", "pulse_wait_done", "pulse_stop", "pulse_estimate", "pulse_describe", "pulse_schedule"];
/// Those the window's and the bus's clients give it (their Stop is the editor's).
const CLIENT_COMMANDS: [&str; 5] = ["pulse_run", "pulse_wait_done", "pulse_estimate", "pulse_describe", "pulse_schedule"];

#[derive(Default)]
struct Held {
    board: Option<Arc<Board>>,
    device: Option<Device>,
    /// None until the window first states them.
    claims: Option<Claims>,
    /// A run's commands asked before the window stated its claims: judged at the next statement.
    parked: Vec<Asked>,
}

struct Desk {
    /// The board's lane: one piece of board or pulse-file work at a time, in order (the editor's too).
    device: Lane,
    safe: Lane,
    post: Post,
    /// The workspace a relative document path is in.
    root: Mutex<PathBuf>,
    plugins: Mutex<Option<PluginHost>>,
    held: Mutex<Held>,
    runs: Mutex<BTreeMap<String, RunUse>>,
    /// Each watch of a lane's schedule, and how its thread is told to end.
    watches: Mutex<Vec<(Watch, mpsc::Sender<Heard>)>>,
    /// Watch threads still running.
    watching: AtomicUsize,
    /// A notice for the window is on its way.
    noticed: AtomicBool,
}

/// The board's commands, answered off the window's turn by whoever owns each fact: the board its
/// play and its reports, the desk the runs' use of it, the window's last statement their claims.
#[derive(Clone)]
pub struct PulseDesk(Arc<Desk>);

/// A thread that counts itself among the desk's watch threads while it runs.
struct Watching(Arc<Desk>);
impl Drop for Watching { fn drop(&mut self) { self.0.watching.fetch_sub(1, Ordering::SeqCst); } }

fn refused(message: String) -> zlc_plugin_host::CallError { zlc_plugin_host::CallError::new(zlc_plugin_host::ErrorKind::Refused, message) }

impl PulseDesk {
    pub fn new(post: Post) -> PulseDesk {
        PulseDesk(Arc::new(Desk { device: Lane::new("pulse-command"), safe: Lane::new("pulse-safe"), post, root: Mutex::default(),
            plugins: Mutex::default(), held: Mutex::default(), runs: Mutex::default(), watches: Mutex::default(),
            watching: AtomicUsize::new(0), noticed: AtomicBool::new(false) }))
    }

    pub fn set_plugins(&self, plugins: Option<PluginHost>) { *lock(&self.0.plugins) = plugins; }
    fn plugins(&self) -> Option<PluginHost> { lock(&self.0.plugins).clone() }
    pub(crate) fn device_lane(&self) -> &Lane { &self.0.device }
    pub(crate) fn safe_lane(&self) -> &Lane { &self.0.safe }
    /// How many watch threads are running.
    #[cfg(test)]
    pub fn watching(&self) -> usize { self.0.watching.load(Ordering::SeqCst) }

    /// A plugin host event, before the window hears it: a run's board command is answered here
    /// (None: taken), and a run's end lets its use of the board go.
    pub fn hears(&self, event: HostEvent) -> Option<HostEvent> {
        match event {
            HostEvent::Request { instance, run, ticket, op, params, .. }
                if op == "core" && !run.is_empty() && params["command"].as_str().is_some_and(|name| RUN_COMMANDS.contains(&name)) => {
                let action = params["command"].as_str().unwrap_or("").to_owned();
                let mut command = params.get("params").cloned().filter(Value::is_object).unwrap_or_else(|| json!({}));
                let host = self.plugins();
                let reply: Reply = Box::new(move |result| if let Some(host) = host { host.respond(ticket, result.map_err(refused)); });
                // As every catalogue command is: its parameters checked, a control one written in the journal.
                let spec = zlc_bus::catalog::find(&action).expect("the run commands are catalogue commands");
                if let Err(error) = spec.validate(&crate::remote::parameters(&command)) { reply(Err(format!("Invalid {action}: {error}"))); return None; }
                if !spec.read_only { zlc_core::journal::note_with("app", "action", || json!({"action": action, "run": run, "command": command})); }
                crate::remote::workspace_paths(&mut command, &lock(&self.0.root));
                self.ask(Who::Run { hosted: run, instance }, &action, command, reply);
                None
            }
            HostEvent::RunFinished { ref run, .. } => { self.ended(run); Some(event) }
            event => Some(event),
        }
    }

    /// A board command the window or a client gave: answered off the window's turn, as a
    /// [`PulseEvent::Reply`] (false: not one of the desk's).
    pub fn client(&self, action: &str, command: &Value, request: u64, claimant: Option<String>) -> bool {
        if !CLIENT_COMMANDS.contains(&action) { return false; }
        let post = self.0.post.clone();
        self.ask(Who::Client { claimant }, action, crate::remote::parameters(command), Box::new(move |result| post(PulseEvent::Reply { request, result })));
        true
    }

    /// The window's record of the runs and their claims: the commands held for it are judged now.
    pub fn stated(&self, claims: Claims) {
        let parked = {
            let mut held = lock(&self.0.held);
            if held.claims.as_ref() == Some(&claims) && held.parked.is_empty() { return; }
            held.claims = Some(claims);
            std::mem::take(&mut held.parked)
        };
        for asked in parked { self.judge(asked, true); }
    }

    /// The board the editor shows (the apparatus' or its own): each watch is told it, then its changes.
    pub(crate) fn set_board(&self, board: Option<Arc<Board>>) {
        lock(&self.0.held).board = board;
        let watches = std::mem::take(&mut *lock(&self.0.watches));
        for (watch, ending) in watches {
            let _ = ending.send(Heard::Ended);
            self.tell(watch, true);
        }
    }

    fn set_device(&self, device: Option<Device>) { lock(&self.0.held).device = device; }
    fn set_published(&self, published: bool) { if let Some(device) = lock(&self.0.held).device.as_mut() { device.published = published; } }

    /// The window took the notice: the next change is told again.
    pub fn noticed(&self) { self.0.noticed.store(false, Ordering::SeqCst); }
    fn notice(&self) { if !self.0.noticed.swap(true, Ordering::SeqCst) { (self.0.post)(PulseEvent::Touched); } }

    pub(crate) fn world(&self) -> Option<Arc<dyn World>> {
        let host = self.plugins()?;
        let instance = lock(&self.0.held).device.as_ref()?.world.clone()?;
        Some(Arc::new(PluginWorld { host, instance }))
    }

    pub(crate) fn ask(&self, who: Who, action: &str, command: Value, reply: Reply) {
        self.judge(Asked { who, action: action.to_owned(), command, reply }, false);
    }

    /// One command, held to its asker's rights (`decided`: the window has stated the claims since it
    /// was asked, so a run it does not name holds nothing), then answered.
    fn judge(&self, asked: Asked, decided: bool) {
        let mut held = lock(&self.0.held);
        if drives(&asked.action) && held.device.as_ref().is_some_and(|device| device.published) {
            drop(held);
            return (asked.reply)(Err(PUBLISHED_BOARD.into()));
        }
        // A run may drive only the sequencer it claims; a client may not while a run holds it (v2 device use).
        let verdict = match &asked.who {
            Who::Run { instance, .. } if matches!(asked.action.as_str(), "pulse_run" | "pulse_wait_done" | "pulse_stop") => {
                match (held.claims.as_ref().and_then(|claims| claims.0.get(instance)), &held.device) {
                    (None, _) if !decided => None,
                    (None, _) => Some(Err(format!("the run of {instance} does not claim the pulse board; its node must require sequencer.streamer"))),
                    (Some((run, _)), None) => Some(Err(format!("run {run} cannot drive the pulse board: no sequencer device is open in the apparatus"))),
                    (Some((run, devices)), Some(device)) if !devices.contains(&device.instance) =>
                        Some(Err(format!("run {run} does not claim the sequencer {}; its node must require sequencer.streamer", device.instance))),
                    (Some((run, _)), Some(_)) => Some(Ok(Some(run.clone()))),
                }
            }
            Who::Client { claimant: Some(holder) } if matches!(asked.action.as_str(), "pulse_run" | "pulse_wait_done") => {
                Some(Err(format!("device is in use by {holder}")))
            }
            _ => Some(Ok(None)),
        };
        let name = match verdict {
            // Asked before the window stated its run: judged at its next statement.
            None => { held.parked.push(asked); return; }
            Some(Err(message)) => { drop(held); return (asked.reply)(Err(message)); }
            Some(Ok(name)) => name,
        };
        let board = held.board.clone();
        drop(held);
        let Asked { who, action, command, reply } = asked;
        match action.as_str() {
            "pulse_describe" => {
                let document = PathBuf::from(command["document"].as_str().unwrap_or(""));
                let _ = std::thread::Builder::new().name("zlc-pulse-describe".into())
                    .spawn(move || reply(describe_document(&document).map(|description| to_value(&description)).map_err(|error| error.message)));
            }
            // A read whoever holds the board: a camera aligning its frames to the lane that triggers it claims no sequencer.
            // A run that watches it (`_watch`) is answered now and then told each change by the board itself.
            "pulse_schedule" => {
                let lane = command["lane"].as_str().unwrap_or("").to_owned();
                // Watched from before the answer is read: no change falls between the two.
                let watch = match (&who, command["_watch"].as_str()) {
                    (Who::Run { hosted, .. }, Some(id)) => Some(Watch { lane: lane.clone(), run: hosted.clone(), id: id.to_owned() }),
                    _ => None,
                };
                if let Some(watch) = &watch { self.tell(watch.clone(), false); }
                let answer = schedule(board.as_deref(), &lane);
                if let (Some(watch), Err(_)) = (&watch, &answer) { self.unwatch(|held| held.run == watch.run && held.id == watch.id); }
                reply(answer);
            }
            _ => {
                let Some(board) = board else {
                    let missing = if action == "pulse_estimate" { "no pulse board is attached: an estimate compiles for the board's geometry and Config" } else { "no pulse board is attached" };
                    return reply(Err(missing.into()));
                };
                match action.as_str() {
                    "pulse_run" => self.play(who, name, board, command, reply),
                    // A run's Stop answers with what the board said (v2 nodes' sequencer.safe()), before the run goes on.
                    "pulse_stop" => {
                        let desk = self.clone();
                        self.0.safe.run(move || {
                            let result = board.safe().map(|readback| json!({"status": readback.status, "command_id": readback.command_id}))
                                .map_err(|error| format!("the board did not go SAFE: {}", error.message));
                            reply(result);
                            desk.notice();
                        });
                    }
                    "pulse_wait_done" => self.wait(who, board, command, reply),
                    "pulse_estimate" => {
                        let document = PathBuf::from(command["document"].as_str().unwrap_or(""));
                        let points: Vec<_> = command["api_points"].as_array().map(|points| points.iter().map(api_values).collect()).unwrap_or_default();
                        let axes = command["scan"].is_object().then(|| ScanAxes { fields: strings(&command["scan"]["fields"]), columns: rows(&command["scan"]["columns"]) });
                        let _ = std::thread::Builder::new().name("zlc-pulse-estimate".into()).spawn(move || {
                            reply(estimate_document(&board, &document, &points, axes.as_ref()).map(|estimate| to_value(&estimate)).map_err(|error| error.message));
                        });
                    }
                    other => reply(Err(format!("{other} is not a board command"))),
                }
            }
        }
    }

    /// `pulse_run`: one stage of a run's play (or a client's whole play), on the board's lane; what it
    /// leaves the run holding is kept before the run hears its answer.
    fn play(&self, who: Who, name: Option<String>, board: Arc<Board>, command: Value, reply: Reply) {
        let hosted = match &who { Who::Run { hosted, .. } => Some(hosted.clone()), Who::Client { .. } => None };
        let stage = command["stage"].as_str().unwrap_or("run").to_owned();
        if stage != "run" && hosted.is_none() { return reply(Err("stage load and fire are a run's two-step play".into())); }
        board.set_world(self.world());
        let document = PathBuf::from(command["document"].as_str().unwrap_or(""));
        let values = api_values(&command["api_values"]);
        let scan = command["scan"].is_object().then(|| ScanRows { fields: strings(&command["scan"]["fields"]), rows: rows(&command["scan"]["rows"]) });
        let plugins = self.plugins();
        let native = if command["scan_plan"].is_object() {
            if scan.is_some() { return reply(Err("scan and scan_plan are mutually exclusive".into())); }
            let (Some(hosted), Some(plugins)) = (hosted.as_deref(), plugins.as_ref()) else { return reply(Err("scan_plan belongs to a live plugin run".into())) };
            let Some(handle) = command["scan_plan"]["handle"].as_str().filter(|s| !s.is_empty()) else { return reply(Err("scan_plan needs its native handle".into())) };
            let Some(outer_index) = command["scan_plan"]["outer_index"].as_u64().and_then(|v| usize::try_from(v).ok()) else {
                return reply(Err("scan_plan needs an outer point index".into()));
            };
            match crate::plugin_services::scan_plan(plugins, hosted, handle) {
                Ok(plan) => Some((ScanPlanRequest { plan, outer_index }, handle.to_owned())),
                Err(error) => return reply(Err(error.to_string())),
            }
        } else { None };
        let (runs, sweeps) = (command["run_repeats"].as_u64().unwrap_or(1), command["scan_repeats"].as_u64().unwrap_or(1));
        if let Some((native, _)) = &native {
            let expected_sweeps = if native.plan.outer_axis_count() == 0 { native.plan.spec().scan_repeats } else { 1 };
            if stage != "load" && (runs as u128 != native.plan.spec().run_repeats as u128 || sweeps as u128 != expected_sweeps as u128) {
                return reply(Err("pulse repeats differ from the armed native scan plan".into()));
            }
        }
        let (pinned, loaded) = hosted.as_ref().map_or((None, None), |hosted| {
            let mut runs = lock(&self.0.runs);
            let used = runs.entry(hosted.clone()).or_default();
            used.name = name.clone().unwrap_or_default();
            used.in_flight += 1;
            (used.pinned.get(&zlc_pulse::codec::absolute(&document)).cloned(), used.loaded.clone())
        });
        let desk = self.clone();
        self.0.device.run(move || {
            let descriptor = native.as_ref().map(|(request, handle)| json!({"handle": handle, "outer_index": request.outer_index}));
            let mismatch = stage == "fire" && loaded.as_ref().and_then(|value| value.get("native_scan")) != descriptor.as_ref();
            let (mut result, pin, fired, mut loaded) = if mismatch {
                (Err("stage fire must use the native scan handle and outer point accepted by stage load".into()), None, None, None)
            } else {
                play(&board, &stage, &document, pinned.as_deref(), loaded.as_deref(), &values, scan.as_ref(), native.as_ref().map(|(request, _)| request), runs, sweeps)
            };
            if let (Some((native, handle)), Some(hosted)) = (&native, &hosted) {
                if let Some(value) = &mut loaded { Arc::make_mut(value)["native_scan"] = descriptor.clone().unwrap_or_default(); }
                if let (Ok(receipt), Some(plugins)) = (&mut result, &plugins) {
                    if let Err(error) = accept_native_receipt(plugins, hosted, handle, native, &board, receipt) { result = Err(error); }
                }
            }
            if let Some(hosted) = &hosted { desk.played(hosted, pin, fired, loaded); }
            reply(result);
            desk.notice();
        });
    }

    /// What one of a run's `pulse_run`s left it holding; a run that ended meanwhile is let go once its last is in.
    fn played(&self, hosted: &str, pinned: Option<(PathBuf, Arc<Sequence>)>, fired: Option<u32>, loaded: Option<Arc<Value>>) {
        let settle = {
            let mut runs = lock(&self.0.runs);
            let used = runs.entry(hosted.to_owned()).or_default();
            used.in_flight = used.in_flight.saturating_sub(1);
            if let Some((path, sequence)) = pinned { used.pinned.entry(path).or_insert(sequence); used.loaded = loaded; }
            if fired.is_some() { used.fire = fired; }
            used.ended && used.in_flight == 0
        };
        if settle { self.settle(hosted); }
    }

    /// `pulse_wait_done`: FIRE `command_id`'s report, taken by whoever waits for it (taking it frees the
    /// board for the next FIRE). A run that watches it (`_watch`) is answered at once -- the report if it
    /// is in, else not done -- and told once when the board reports (or can no longer report); any other
    /// wait is answered when the report is in or its `timeout_seconds` ran out.
    fn wait(&self, who: Who, board: Arc<Board>, command: Value, reply: Reply) {
        let id = command["command_id"].as_u64().unwrap_or(0) as u32;
        let report = |report: zlc_pulse::Result<Option<zlc_pulse::device::DoneReport>>| report.map(|report| to_value(&done_json(report.as_ref()))).map_err(|error| error.message);
        let desk = self.clone();
        if let (Who::Run { hosted, .. }, Some(watch)) = (&who, command["_watch"].as_str()) {
            let now = board.wait_report(id, Some(Duration::ZERO));
            let taken = matches!(now, Ok(Some(_)));
            let pending = matches!(now, Ok(None));
            reply(report(now));
            if taken { self.notice(); }
            let Some(host) = self.plugins().filter(|_| pending) else { return };
            let (run, watch) = (hosted.clone(), watch.to_owned());
            self.0.watching.fetch_add(1, Ordering::SeqCst);
            let watching = Watching(self.0.clone());
            let _ = std::thread::Builder::new().name("zlc-pulse-done".into()).spawn(move || {
                let _watching = watching;
                // Ends with its FIRE: a report, or the SAFE that a Stop or the run's end sends.
                let done = board.wait_report(id, None);
                let taken = matches!(done, Ok(Some(_)));
                let _ = host.tell_watch(&run, &watch, report(done));
                if taken { desk.notice(); }
            });
            return;
        }
        let timeout = Duration::from_secs_f64(command["timeout_seconds"].as_f64().unwrap_or(0.0).clamp(0.0, 600.0));
        let _ = std::thread::Builder::new().name("zlc-pulse-wait".into()).spawn(move || {
            let done = board.wait_report(id, Some(timeout));
            // Only a report taken changes the board (its FIRE is over); a wait that ran out touched nothing.
            let taken = matches!(done, Ok(Some(_)));
            reply(report(done));
            if taken { desk.notice(); }
        });
    }

    /// Tell `watch` the attached board's schedule (`now`: at once) and then each change of it, on a
    /// thread of its own: the board says when it changed, nothing asks, and the window's turn is not
    /// in the way. It ends when its run ends, when the board is replaced (the watch is then told the
    /// new one) or when the run no longer hears it.
    fn tell(&self, watch: Watch, now: bool) {
        let Some(host) = self.plugins() else { return };
        let board = lock(&self.0.held).board.clone();
        let (ending, heard) = mpsc::channel();
        if let Some(board) = &board {
            let changed = ending.clone();
            board.on_change(move |_| changed.send(Heard::Changed).is_ok());
        }
        lock(&self.0.watches).push((watch.clone(), ending));
        let board = board.as_ref().map(Arc::downgrade);
        self.0.watching.fetch_add(1, Ordering::SeqCst);
        let watching = Watching(self.0.clone());
        let _ = std::thread::Builder::new().name("zlc-pulse-watch".into()).spawn(move || {
            let _watching = watching;
            let told = |board: Option<&Board>| host.tell_watch(&watch.run, &watch.id, schedule(board, &watch.lane)).is_ok();
            if now && !told(board.as_ref().and_then(std::sync::Weak::upgrade).as_deref()) { return; }
            let Some(board) = board else { return };
            while let Ok(Heard::Changed) = heard.recv() {
                // Changes that came together are told once, as the board now is.
                loop {
                    match heard.try_recv() {
                        Ok(Heard::Changed) => {}
                        Ok(Heard::Ended) => return,
                        Err(_) => break,
                    }
                }
                let Some(board) = board.upgrade() else { return };
                if !told(Some(&*board)) { return; }
            }
        });
    }

    /// End the watches `which` names.
    fn unwatch(&self, which: impl Fn(&Watch) -> bool) {
        lock(&self.0.watches).retain(|(watch, ending)| if which(watch) { let _ = ending.send(Heard::Ended); false } else { true });
    }

    /// A run is over -- its node returned, failed or was stopped, or its process died: its watches
    /// end, and once its last `pulse_run` is in it is let go ([`Self::settle`]).
    fn ended(&self, hosted: &str) {
        self.unwatch(|watch| watch.run == hosted);
        let settle = lock(&self.0.runs).get_mut(hosted).is_some_and(|used| { used.ended = true; used.in_flight == 0 });
        if settle { self.settle(hosted); }
    }

    /// A run let go: its pins go, and a FIRE of its that is still unreported (a run that ended
    /// without its own Stop) is made SAFE, as each v2 node's finally did.
    fn settle(&self, hosted: &str) {
        let Some(used) = lock(&self.0.runs).remove(hosted) else { return };
        let (board, instance) = { let held = lock(&self.0.held); (held.board.clone(), held.device.as_ref().map(|device| device.instance.clone())) };
        let (Some(fire), Some(board)) = (used.fire, board) else { return };
        if board.last_fire() != Some((fire, Fate::Pending)) { return; }
        let (desk, run) = (self.clone(), used.name);
        self.0.safe.run(move || {
            let outcome = match board.safe() {
                Ok(_) => format!("ZLC SAFE after {run}: it ended with FIRE {fire} unreported"),
                Err(error) => format!("ZLC SAFE FAILED after {run} ended with FIRE {fire} unreported: {}", error.message),
            };
            if let Some((instance, line)) = instance.zip(narration(&outcome)) { (desk.0.post)(PulseEvent::Log { instance, line }); }
            desk.notice();
        });
    }
}

pub struct PulseHost {
    pub editor: Editor,
    /// The board's commands, answered off the window's turn; its lanes are the editor's board lanes.
    pub desk: PulseDesk,
    program: Lane,
    preview: Lane,
    post: Post,
    safe_actions: SafeActions,
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
    /// What the sequencer device last stated of itself, and the board and revision that was of.
    facts: Value,
    facts_of: Option<(usize, u64)>,
    /// Something the window shows changed since its view was last made.
    dirty: bool,
    /// Editor commands whose files are being written: the view waits for them.
    writing: usize,
    /// Closing the window: told once the board's close, begun by `shutdown`, has ended.
    closing: Option<mpsc::Receiver<()>>,
    /// The board's close at the window's close has ended.
    closed: bool,
}

/// How a command was answered.
pub enum Handled { Now(Result<Value, String>), Later }

impl PulseHost {
    /// The editor's pulses folder is the workspace's; its value files live beside it (v2 Workspace.prepare made both).
    pub fn new(desk: PulseDesk, post: Post, safe_actions: SafeActions, plugins: Option<PluginHost>, pulses: &Path) -> PulseHost {
        let deployed = Deployed::load().map_err(|error| eprintln!("{}", json!({"type": "pulse-error", "error": error.message}))).ok();
        if let Some(workspace) = pulses.parent() {
            let _ = std::fs::create_dir_all(workspace.join(zlc_pulse::codec::CONFIG_VALUES_DIRECTORY));
            *lock(&desk.0.root) = workspace.to_path_buf();
        }
        desk.set_plugins(plugins);
        let python = zlc_plugin_host::plugin_python().ok();
        PulseHost { editor: Editor::new(deployed, pulses.to_path_buf()), desk, program: Lane::new("pulse-program"),
                    preview: Lane::new("pulse-preview"), post, safe_actions, given: None, python, session: Arc::new(Mutex::new(None)),
                    dpr: 1.0, preview_busy: false, preview_key: String::new(), preview_wanted: false, frame: None, frames: 0,
                    axes: Vec::new(), pointers: Default::default(), pointer_busy: false, users: Default::default(),
                    facts: Value::Null, facts_of: None, dirty: true, writing: 0, closing: None, closed: false }
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

    /// The pulse section when something it shows changed since it was last made (and no command's
    /// files are still being written): made at most once per change, never for a reply alone.
    pub fn view(&mut self) -> Option<Value> {
        if !self.dirty || self.writing > 0 { return None; }
        self.dirty = false;
        Some(self.state())
    }

    /// A pulse section once it is sent: the editor takes its unchanged schedule cards back for the next one.
    pub fn recycle(&mut self, pulse: &mut Value) { self.editor.recycle(pulse); }

    /// The apparatus' sequencer device instance, when the board is one.
    pub fn apparatus_device(&self) -> Option<String> { self.given.as_ref().map(|given| given.instance.clone()) }

    /// Run the editor's jobs on their lanes; move Stop all's SAFE and the desk with the board.
    pub fn dispatch(&mut self) {
        // Every editor change ends here: the record is current before any later start is admitted.
        let users = self.given.as_ref().filter(|_| self.editor.drives()).map(|given| (given.instance.clone(), "PulseGUI".to_owned()));
        if let Ok(mut record) = self.users.lock() { *record = users.into_iter().collect(); }
        if let Some(board) = self.editor.take_board_change() {
            self.safe_actions.unregister(SAFE_OUTPUT);
            if let Some(board) = &board {
                let held = board.clone();
                self.safe_actions.register(SAFE_OUTPUT, Arc::new(move || held.safe().map(|_| ()).map_err(|error| error.message)));
            }
            self.desk.set_board(board);
        }
        if let Some(board) = self.editor.board() { board.set_world(self.desk.world()); }
        for job in self.editor.take_jobs() {
            let post = self.post.clone();
            match job {
                Job::Device { work, done } => self.desk.device_lane().run(move || { let outcome = work(); done.set(); post(PulseEvent::Outcome(outcome)); }),
                Job::Safe { work } => self.desk.safe_lane().run(move || post(PulseEvent::Outcome(work()))),
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

    /// The status poll: the board asked how it is, and the Subpulse files the open pulse resolves
    /// looked at off the window's turn (a changed one is told as [`PulseEvent::FilesChanged`]).
    pub fn poll(&mut self) {
        self.editor.poll();
        self.dispatch();
        let post = self.post.clone();
        self.desk.device_lane().run(move || if !zlc_pulse::codec::changed_subpulses().is_empty() { post(PulseEvent::FilesChanged) });
    }

    /// A Subpulse file changed on the disk: what the pages show of it is worked out again.
    pub fn files_changed(&mut self) { self.editor.subpulse_files_changed(); self.dirty = true; }

    /// Someone else drove the board: the editor asks it again.
    pub fn touched(&mut self) {
        self.desk.noticed();
        self.editor.board_touched();
        self.dispatch();
        self.dirty = true;
    }

    /// A lane's answer to the editor: whether what the window shows may have changed.
    pub fn deliver(&mut self, outcome: Outcome) {
        if self.editor.deliver(outcome) { self.dirty = true; }
        self.dispatch();
    }

    /// The run holding the session's sequencer changed: the editor may (not) drive it now.
    pub fn claimed_by(&mut self, claimant: Option<String>) {
        if self.editor.set_claimed_by(claimant) { self.editor.poll(); self.dispatch(); self.dirty = true; }
    }

    /// Draw the preview again when its page is shown and what it draws changed.
    pub fn refresh_preview(&mut self) {
        if self.editor.page() != zlc_pulse_editor::editor::PREVIEW_PAGE && self.session.lock().map(|s| s.is_none()).unwrap_or(true) { return; }
        let Some(request) = self.editor.preview_request() else { return };
        let (sequence, include_off, pinned) = match request {
            Ok(request) => request,
            Err(message) => { self.editor.set_preview_status(message, "error"); self.dirty = true; return; }
        };
        let key = format!("{:?}|{include_off}|{pinned:?}|{}", zlc_pulse::codec::sequence_to_tree(&sequence).canonical(), self.dpr);
        if key == self.preview_key { return; }
        if self.preview_busy { self.preview_wanted = true; return; }
        self.preview_busy = true;
        self.editor.set_preview_status("drawing preview…", "task");
        self.dirty = true;
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
        self.dirty = true;
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
        self.dirty = true;
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
        if self.pointers.back_mut().is_some_and(|last| crate::pointer_queue::merge_scroll(last, &command)) {
            self.next_pointer();
            return;
        }
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
            Err(error) => { self.editor.set_preview_status(format!("preview gesture: {error}"), "error"); self.dirty = true; None }
        };
        self.next_pointer();
        presented
    }

    /// `save_preview_image`: the drawn preview as PNG, beside the pulse.
    fn save_preview(&mut self, path: Option<String>) -> Result<Value, String> {
        self.dirty = true;
        if self.session.lock().map(|s| s.is_none()).unwrap_or(true) { self.editor.warn("There is no preview to save"); return Ok(Value::Null); }
        if self.preview_busy {
            self.editor.set_preview_status("preview is busy; save again when the drawing is ready", "warning");
            return Ok(Value::Null);
        }
        let name = self.editor.sequence().map_or_else(|| "pulse".to_owned(), |s| if s.name.is_empty() { "pulse".into() } else { s.name.clone() });
        let target = path.map(PathBuf::from).unwrap_or_else(|| tables::unique_path(&PathBuf::from(self.editor.state_folder()), &name, ".png"));
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

    pub fn preview_saved(&mut self, result: Result<PathBuf, String>) {
        self.dirty = true;
        match result {
            Ok(path) => self.editor.set_preview_status(format!("saved {}", path.display()), "task"),
            Err(error) => self.editor.warn(format!("Cannot save the preview: {error}")),
        }
    }

    /// One `pulse_*` command the window or a client of the bus gave (the operator's path already
    /// asked where a file goes: main.rs, dialog.rs). The board's own commands are the desk's; the
    /// editor's run here, the file they read read ahead and the files they write written off the
    /// window's turn. `claimant`: the run that holds the session's sequencer, if one does.
    pub fn command(&mut self, action: &str, command: &Value, operator: bool, request: u64, claimant: Option<String>) -> Handled {
        self.claimed_by(claimant.clone());
        if self.desk.client(action, command, request, claimant) { return Handled::Later; }
        // Published, the board is the other computer's: nothing here plays it (Stop still makes it safe).
        if drives(action) && self.given.as_ref().is_some_and(|given| given.published) { return Handled::Now(Err(PUBLISHED_BOARD.into())); }
        match action {
            // v2 save_preview_image: the drawing as it is, beside the pulse under a fresh name (no dialog).
            "pulse_preview" if command["save"].as_bool() == Some(true) => return Handled::Now(self.save_preview(command["path"].as_str().map(str::to_owned))),
            "pulse_ready" => { self.dpr = command["dpr"].as_f64().filter(|dpr| *dpr > 0.0).unwrap_or(1.0); self.dispatch(); return Handled::Now(Ok(Value::Null)); }
            _ => {}
        }
        let args: Map<String, Value> = crate::remote::parameters(command).as_object().cloned().unwrap_or_default();
        if let Some(path) = self.editor.reads(action, &args) {
            let (post, action) = (self.post.clone(), action.to_owned());
            self.desk.device_lane().run(move || {
                zlc_pulse::codec::read_ahead(&path);
                post(PulseEvent::Read(Box::new(ReadAhead { action, args, operator, request, path })));
            });
            return Handled::Later;
        }
        self.edit(action, &args, operator, request)
    }

    /// The file an editor command reads is in: the command runs. Its answer, and the request it answers.
    pub fn read(&mut self, ahead: ReadAhead) -> (u64, Handled) {
        let ReadAhead { action, args, operator, request, path } = ahead;
        let handled = self.edit(&action, &args, operator, request);
        zlc_pulse::codec::forget_read_ahead(&path);
        (request, handled)
    }

    fn edit(&mut self, action: &str, args: &Map<String, Value>, operator: bool, request: u64) -> Handled {
        let answer = self.editor.command(action, args, operator);
        self.dirty = true;
        // Selectors off ends a drag in progress (v2 cancelled the active interaction).
        if action == "pulse_preview" && args.get("selectors") == Some(&Value::Bool(false)) { self.pointer(json!({"event": "cancel", "x": 0.0, "y": 0.0})); }
        let writes = self.editor.take_writes();
        self.dispatch();
        let Some(writes) = writes else { return Handled::Now(answer) };
        // On the board's lane: a FIRE after a save plays what was saved.
        self.writing += 1;
        let post = self.post.clone();
        self.desk.device_lane().run(move || {
            let result = writes.files.iter().try_for_each(|(path, bytes)| zlc_pulse::codec::write_file(path, bytes)
                .map_err(|error| format!("{}: {}", path.display(), error.message)));
            post(PulseEvent::Written(Box::new(Written { request, writes, result, answer })));
        });
        Handled::Later
    }

    /// An editor command's files were written (or not): its answer, and the request it answers.
    pub fn written(&mut self, written: Written) -> (u64, Result<Value, String>) {
        let Written { request, writes, result, answer } = written;
        self.writing = self.writing.saturating_sub(1);
        self.dirty = true;
        let answer = self.editor.written(writes, result, answer);
        self.dispatch();
        (request, answer)
    }

    /// A pulse a Figure recorded, opened as an unsaved draft.
    pub fn open_recorded(&mut self, document: &str, discard_changes: bool) -> Result<Value, String> {
        let answer = self.editor.open_recorded(document, discard_changes);
        self.dirty = true;
        self.dispatch();
        answer
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
        let world = (type_id == "sequencer.virtual").then(|| (self.desk.plugins(), world_instance(&instance), context["simulation"].clone()));
        self.desk.device_lane().run(move || {
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
        let world = (board.is_virtual() && self.desk.plugins().is_some()).then(|| world_instance(instance));
        self.desk.set_device(Some(Device { instance: instance.to_owned(), world: world.clone(), published: false }));
        self.given = Some(Given { instance: instance.to_owned(), child, published: false, world });
        self.editor.attach_given(board, label);
        // The desk takes the board here; each watch is then told this one's schedule, then its changes.
        self.dispatch();
        self.dirty = true;
        // `snapshot`: it reports its state for the bench a Start records (`bench_snapshot`).
        let facts = self.device_facts().map_or(Value::Null, |(_, facts)| facts);
        Ok(json!({"board": self.editor.board().map(|b| b.name().to_owned()), "ports": description.target.ports.len(),
                  "clock_hz": description.clock_hz, "snapshot": true, "facts": facts}))
    }

    /// What the apparatus' sequencer states of itself, when it differs from what was last said: its
    /// trigger lanes -- `{trigger_lanes: [{lane, port, label, frames}]}`, `frames` how often the lane rises
    /// in one shot of what the board holds, null while it holds nothing. Worked out when the board changes
    /// what it holds, not at every turn; the workbench is told (`Event::Facts`) and asks its drafts again.
    pub fn device_facts(&mut self) -> Option<(String, Value)> {
        let instance = self.given.as_ref()?.instance.clone();
        let board = self.editor.board()?.clone();
        let key = (std::sync::Arc::as_ptr(&board) as usize, board.revision());
        if self.facts_of == Some(key) { return None; }
        self.facts_of = Some(key);
        let counts = board.trigger_counts().ok().and_then(|(_, counts)| counts);
        let lanes: Vec<Value> = board.trigger_lanes().into_iter().enumerate().map(|(at, (lane, port, label))| json!({
            "lane": lane, "port": port, "label": label, "frames": counts.as_ref().and_then(|counts| counts.get(at).copied())})).collect();
        let facts = json!({"trigger_lanes": lanes});
        if self.facts == facts { return None; }
        self.facts = facts.clone();
        Some((instance, facts))
    }

    /// The apparatus board's `snapshot`, read off the window's turn (a pulse server's client waits
    /// behind a LOAD) and given up after `wait`: the work to run on a thread of its own.
    pub fn bench_snapshot(&self, wait: Duration) -> impl FnOnce() -> Result<Value, String> + Send + 'static {
        let board = self.given.as_ref().and(self.editor.board().cloned());
        move || {
            let board = board.ok_or("no pulse board is attached")?;
            let (sender, answer) = mpsc::channel();
            std::thread::Builder::new().name("zlc-pulse-bench".into()).spawn(move || { let _ = sender.send(bench_entry(&board)); })
                .map_err(|error| error.to_string())?;
            answer.recv_timeout(wait).unwrap_or_else(|_| Err(format!("the pulse board did not answer within {} s", wait.as_secs())))
        }
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
        self.desk.set_device(None);
        self.editor.detach_given();
        // No board: each watch is told nothing is held any more.
        self.dispatch();
        self.dirty = true;
        let instance = instance.to_owned();
        let plugins = self.desk.plugins();
        self.desk.device_lane().run(move || {
            let error = board.and_then(|board| board.close().err()).map(|error| error.message);
            if let Some(served) = given.child { served.end(); }
            if let (Some(host), Some(world)) = (plugins, &given.world) { let _ = host.close(world); }
            if closed { post(PulseEvent::Closed { instance, error }); }
        });
    }

    /// Closing the window: SAFE, then let every connection go -- on the board's SAFE lane, as Stop
    /// all's SAFE is, never on the window's turn, which goes on meanwhile (the runs it stops end and
    /// are answered). Each step is bounded as it always was (the board's own call, the local
    /// server's end); [`PulseEvent::ShutDown`] says it is done, [`PulseHost::shut_down`] reads it.
    pub fn shutdown(&mut self) {
        if self.closing.is_some() || self.closed { return; }
        let board = self.editor.board().cloned();
        let served = self.given.take().and_then(|mut given| given.child.take());
        self.safe_actions.unregister(SAFE_OUTPUT);
        let (done, closing) = mpsc::channel();
        self.closing = Some(closing);
        let post = self.post.clone();
        self.desk.safe_lane().run(move || {
            if let Some(board) = board { let _ = board.close(); }
            if let Some(served) = served { served.end(); }
            let _ = done.send(());
            post(PulseEvent::ShutDown);
        });
    }

    /// Whether the board's close begun by [`PulseHost::shutdown`] has ended.
    pub fn shut_down(&mut self) -> bool {
        if self.closing.as_ref().is_some_and(|closing| !matches!(closing.try_recv(), Err(mpsc::TryRecvError::Empty))) {
            self.closing = None;
            self.closed = true;
        }
        self.closed
    }

    /// The application ends: a board close begun by [`PulseHost::shutdown`] is waited for, so
    /// the board is SAFE before the process goes, whichever way it ends.
    pub fn await_shutdown(&mut self) {
        if let Some(closing) = self.closing.take() { let _ = closing.recv(); self.closed = true; }
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
        self.desk.set_published(published);
        if !published {
            if let Some(board) = self.editor.board().cloned() {
                let (desk, post, instance) = (self.desk.clone(), self.post.clone(), instance.to_owned());
                self.desk.device_lane().run(move || {
                    if let Err(error) = board.rejoin() {
                        if let Some(line) = narration(&format!("ZLC REJOIN FAILED: {}", error.message)) { post(PulseEvent::Log { instance, line }); }
                    }
                    desk.notice();
                });
            }
        }
        Ok(port)
    }
}

/// `pulse_schedule` of `board` (none attached: nothing held, nothing played).
fn schedule(board: Option<&Board>, lane: &str) -> Result<Value, String> {
    match board {
        None => Ok(json!({"revision": 0, "state": "none", "lane": lane, "points": []})),
        Some(board) => board.trigger_schedule(lane).map(|schedule| to_value(&schedule)).map_err(|error| error.message),
    }
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
    let (backend, arguments) = server_arguments(parameters)?;
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
    let startup = if matches!(backend, "uart" | "memory") { Duration::ZERO } else { zlc_pulse::axi::STARTUP_TIMEOUT };
    let port = match port.recv_timeout(Duration::from_secs(60) + startup) {
        Ok(Some(port)) => port,
        _ => { let _ = child.kill(); return Err("the local pulse server did not start (see its log)".into()); }
    };
    match Board::remote("127.0.0.1", port) {
        Ok(board) => Ok((board, Served { child, port, control })),
        Err(error) => { let _ = child.kill(); Err(error.message) }
    }
}

/// The local server's backend and command line for a row. It never probes a port: Init found
/// the board's UART first (`resolve`), so `auto` with none found is its JTAG cable, `uart`
/// without a port is refused, and a port the row names is the only one it opens.
fn server_arguments(parameters: &Value) -> Result<(&str, Vec<String>), String> {
    let port = parameters["uart_port"].as_str().map(str::trim).filter(|port| !port.is_empty());
    let backend = match (parameters["backend"].as_str().unwrap_or("auto"), port) {
        ("auto", None) => "jtag-axi",
        ("uart", None) => return Err("no UART port: Init finds it, or set one".into()),
        (backend, _) => backend,
    };
    let mut arguments = vec!["--host".to_owned(), zlc_fabric::bind_address(), "--admission".to_owned(), "stdin".to_owned(),
                             "--backend".to_owned(), backend.to_owned(), "--port".to_owned(), parameters["port"].as_u64().unwrap_or(0).to_string()];
    if let Some(port) = port { arguments.extend(["--uart-port".to_owned(), port.to_owned()]); }
    Ok((backend, arguments))
}

/// How long the local board's UART is looked for on Init: what its server's start was given for the same probe.
pub const RESOLVE_WAIT: Duration = Duration::from_secs(60);

/// `resolve` of the local board (`sequencer.local`) before it opens: its blank UART port, looked
/// for on the free ports only (a bound one -- an IMU's COM3 -- is never opened) with the server's
/// own handshake, against the deployed config its server loads. A JTAG or memory board has nothing
/// to find; `auto` with no UART answering opens over JTAG, `uart` fails with what it tried.
pub fn resolve(type_id: &str, parameters: &Value, excluded: &[String], wait: Duration) -> Result<Value, String> {
    if type_id != "sequencer.local" { return Err(format!("{type_id} finds nothing before it opens")); }
    let backend = parameters["backend"].as_str().unwrap_or("auto").to_owned();
    if matches!(backend.as_str(), "jtag-axi" | "memory") { return Ok(json!({"parameters": {}, "note": ""})); }
    let bound: Vec<String> = excluded.iter().filter_map(|resource| resource.strip_prefix("serial:")).map(str::to_owned).collect();
    let (sender, answer) = mpsc::channel();
    std::thread::Builder::new().name("zlc-pulse-probe".into()).spawn(move || {
        let (free, passed) = zlc_pulse::serial::free_ports(&zlc_pulse::serial::list_com_ports(), &bound);
        let probed = (|| -> zlc_pulse::Result<_> {
            let found = zlc_pulse::wire::load_streamer_config(None)?;
            let source = found.source.clone().ok_or_else(|| zlc_pulse::Error::new(zlc_pulse::Kind::Value, format!("no deployed streamer config: {}", found.warnings.join("; "))))?;
            let config = zlc_pulse::wire::require_streamer_config(&source)?;
            let target = zlc_pulse::manifest::target_from_config(&config, None, None)?;
            Ok(zlc_pulse::serial::find_board_uart(&free, config.uart_baud, config.params, config.clock_hz, &target))
        })();
        let _ = sender.send(probed.map(|(port, attempts)| (port, attempts, passed)).map_err(|error| error.message));
    }).map_err(|error| error.to_string())?;
    let (port, attempts, passed) = answer.recv_timeout(wait).unwrap_or_else(|_| Err(format!("its UART was not found within {} s", wait.as_secs())))?;
    // What it tried (the one that answered aside) and what it passed over, as the line says them.
    let mut said: Vec<String> = Vec::new();
    let tried = &attempts[..attempts.len() - usize::from(port.is_some())];
    if !tried.is_empty() { said.push(format!("tried {}", tried.join("; "))); }
    if !passed.is_empty() { said.push(format!("passed over {}: bound", passed.join(", "))); }
    let context = if said.is_empty() { String::new() } else { format!(" ({})", said.join("; ")) };
    match port {
        Some(port) => Ok(json!({"parameters": {"uart_port": port}, "note": format!("board UART found on {port}{context}")})),
        None if backend == "uart" => Err(format!("no board UART on the free ports{}", if context.is_empty() { " (no COM port is free)".to_owned() } else { context })),
        None => Ok(json!({"parameters": {}, "note": format!("no board UART on the free ports{context}; opening over JTAG")})),
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

/// The board's entry in the bench a Start records (PLUGIN_PROTOCOL §5 `snapshot`): whether it is
/// loaded and firing, the loaded program's digest, and what an archive keeps of it (v2
/// `sequencer_archive_snapshot`: the board's facts, its state, the program and its pulse).
fn bench_entry(board: &Board) -> Result<Value, String> {
    let state = board.snapshot().map_err(|error| error.message)?;
    let applied = board.applied().map_err(|error| error.message)?;
    let kept = applied.as_ref().map(|applied| zlc_pulse::board::archive_snapshot(board, applied)).transpose().map_err(|error| error.message)?;
    let flag = |key: &str| state.get(key).and_then(Json::as_bool).unwrap_or(false);
    Ok(json!({"loaded": flag("loaded"), "firing": flag("firing"), "digest": applied.as_ref().map(|applied| applied.program.digest()),
              "snapshot": kept.as_ref().map(to_value)}))
}

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
        "simulation": simulation, "capabilities": ["sequencer.streamer", "trigger.source"], "parameters": parameters, "physical_outputs": output, "commands": [],
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
                      {"name": "uart_port", "type": "str", "label": "UART port", "default": "", "blank_label": "found on Init"},
                      {"name": "port", "type": "int", "label": "Serve on port", "default": zlc_pulse::remote::DEFAULT_PORT, "minimum": 1, "maximum": 65535},
                      config_file])),
    ];
    // The local board binds its UART, which Init finds when the row leaves it blank (`resolve`):
    // no other row, scan or resolve touches it, and its server never probes for it.
    for item in &mut types {
        if item["type_id"] == "sequencer.local" {
            (item["announces"], item["resources"], item["queries"]) = (json!(ANNOUNCED_AS), json!(["serial:{uart_port}"]), json!(["resolve"]));
        }
    }
    // The board a row names (v2 pulse_board_identity): one pulse server serves one board, here or reached over the network.
    for (type_id, identity) in [("sequencer.hardware", "pulse-server:{host}:{port}"), ("sequencer.local", "pulse-server:127.0.0.1:{port}")] {
        if let Some(item) = types.iter_mut().find(|item| item["type_id"] == type_id) { item["identity"] = json!(identity); }
    }
    types
}

#[cfg(test)]
mod tests {
    #[test]
    fn scan_program_keeps_dependent_numeric_columns_paired() {
        let Some(python) = std::env::var_os("ZLC_PLUGIN_PYTHON").map(std::path::PathBuf::from) else { return };
        let table = super::run_scan_program(Some(&python), "import numpy as np\na=np.array([1.,2.,3.])\nb=5-a\nscan_table={'a':a,'b':b}").unwrap().unwrap();
        match table {
            zlc_pulse::scan::ScanTable::Columns(columns) => assert_eq!(columns, vec![("a".into(), vec![1.,2.,3.]), ("b".into(),vec![4.,3.,2.])]),
            _ => panic!("expected named columns"),
        }
    }

    use super::*;
    use std::sync::mpsc::{Receiver, channel};

    /// A pulse host with the virtual board as the apparatus' sequencer, its lanes' events in a
    /// channel (nobody turns the window's loop unless a test does), and a pulse document of its own.
    fn host(name: &str) -> (PulseHost, Receiver<PulseEvent>, PathBuf) {
        let folder = std::env::temp_dir().join(format!("zlc_pulse_host_{}_{name}", std::process::id()));
        std::fs::create_dir_all(folder.join("pulses")).unwrap();
        let (sender, events) = channel();
        let sender = Mutex::new(sender);
        let post: Post = Arc::new(move |event| { let _ = sender.lock().unwrap().send(event); });
        let mut host = PulseHost::new(PulseDesk::new(post.clone()), post, SafeActions::new(), None, &folder.join("pulses"));
        let board = Arc::new(Board::virtual_board(None).unwrap());
        host.device_opened("sequencer", "sequencer", Ok((board, None))).unwrap();
        let document = folder.join("pulses").join("shot.json");
        std::fs::copy(Path::new(env!("CARGO_MANIFEST_DIR")).join("../zlc-pulse/tests/pulses/imaging_template.json"), &document).unwrap();
        (host, events, document)
    }

    /// As the window states the workbench's runs: each named run (its node's instance is its own
    /// name here) claims the sequencer.
    fn claim(host: &PulseHost, runs: &[&str]) {
        host.desk.stated(Claims(runs.iter().map(|run| (run.to_string(), (run.to_string(), vec!["sequencer".to_owned()]))).collect()));
    }

    /// A run's board command, answered by the desk where it arrives: the window's loop does not turn.
    fn run_asks(host: &PulseHost, run: &str, action: &str, command: Value) -> Result<Value, String> {
        let (sender, answer) = channel();
        host.desk.ask(Who::Run { hosted: run.to_owned(), instance: run.to_owned() }, action, command, Box::new(move |result| { let _ = sender.send(result); }));
        answer.recv_timeout(Duration::from_secs(30)).expect("an answer")
    }

    /// A client's command, answered now or on a later turn of the window's loop (which this turns).
    fn ask(host: &mut PulseHost, events: &Receiver<PulseEvent>, action: &str, command: Value, claimant: Option<&str>) -> Result<Value, String> {
        let mut handled = host.command(action, &command, false, 1, claimant.map(str::to_owned));
        loop {
            if let Handled::Now(answer) = handled { return answer; }
            handled = match events.recv_timeout(Duration::from_secs(30)).expect("an answer") {
                PulseEvent::Reply { result, .. } => Handled::Now(result),
                PulseEvent::Read(ahead) => host.read(*ahead).1,
                PulseEvent::Written(written) => Handled::Now(host.written(*written).1),
                PulseEvent::Outcome(outcome) => { host.deliver(outcome); Handled::Later }
                PulseEvent::Touched => { host.touched(); Handled::Later }
                _ => Handled::Later,
            };
        }
    }

    fn played(document: &Path, repeats: u64) -> Value {
        json!({"document": document.display().to_string(), "run_repeats": repeats})
    }

    fn board(host: &PulseHost) -> Arc<Board> { host.editor.board().unwrap().clone() }

    /// A run's play, its report wait and its Stop are answered off the window's turn: nothing here
    /// turns the loop, and what the board did reaches the window as one coalesced notice.
    #[test]
    fn a_runs_board_commands_are_answered_while_the_window_does_not_turn() {
        let (host, events, document) = host("offloop");
        claim(&host, &["scan#1"]);
        let id = run_asks(&host, "scan#1", "pulse_run", played(&document, 1)).unwrap()["command_id"].as_u64().unwrap();
        let done = run_asks(&host, "scan#1", "pulse_wait_done", json!({"command_id": id, "timeout_seconds": 30})).unwrap();
        assert_eq!(done["status"], "done");
        let id = run_asks(&host, "scan#1", "pulse_run", played(&document, 0)).unwrap()["command_id"].as_u64().unwrap() as u32;
        assert!(run_asks(&host, "scan#1", "pulse_stop", json!({})).unwrap()["status"].is_u64());
        assert_eq!(board(&host).last_fire(), Some((id, Fate::Safed)));
        let notices: Vec<PulseEvent> = events.try_iter().collect();
        assert_eq!(notices.iter().filter(|event| matches!(event, PulseEvent::Touched)).count(), 1, "one notice until the window takes it");
        assert!(!notices.iter().any(|event| matches!(event, PulseEvent::Reply { .. })), "a run's answer never passes the window");
    }

    /// A run that asks before the window stated it (its start reply still on the way) is answered
    /// at the window's next statement: held to the claims then, refused when it holds none.
    #[test]
    fn a_run_asking_before_its_claims_are_stated_is_judged_at_the_next_statement() {
        let (host, _events, document) = host("parked");
        claim(&host, &[]);
        let (sender, answer) = channel();
        for run in ["early#1", "stranger#1"] {
            let sender = sender.clone();
            host.desk.ask(Who::Run { hosted: run.to_owned(), instance: run.to_owned() }, "pulse_run", played(&document, 1),
                          Box::new(move |result| { let _ = sender.send((run, result)); }));
        }
        assert!(answer.recv_timeout(Duration::from_millis(200)).is_err(), "held until the window states its runs");
        claim(&host, &["early#1"]);
        let mut answers: Vec<(&str, Result<Value, String>)> = (0..2).map(|_| answer.recv_timeout(Duration::from_secs(30)).unwrap()).collect();
        answers.sort_by_key(|(run, _)| *run);
        assert!(answers[0].1.as_ref().unwrap()["command_id"].is_u64(), "{:?}", answers[0]);
        assert!(answers[1].1.as_ref().unwrap_err().contains("does not claim"), "{:?}", answers[1]);
        let id = answers[0].1.as_ref().unwrap()["command_id"].as_u64().unwrap();
        run_asks(&host, "early#1", "pulse_wait_done", json!({"command_id": id, "timeout_seconds": 30})).unwrap();
    }

    /// A run's record names one pulse: a later fire of its file changed meanwhile is refused;
    /// another run plays the new one.
    #[test]
    fn a_run_plays_the_pulse_it_began_with() {
        let (host, _events, document) = host("pinned");
        claim(&host, &["scan#1", "scan#2"]);
        let receipt = run_asks(&host, "scan#1", "pulse_run", played(&document, 1)).unwrap();
        let id = receipt["command_id"].as_u64().unwrap();
        let done = run_asks(&host, "scan#1", "pulse_wait_done", json!({"command_id": id, "timeout_seconds": 30})).unwrap();
        assert_eq!(done["status"], "done");
        let mut tree: Value = serde_json::from_str(&std::fs::read_to_string(&document).unwrap()).unwrap();
        tree["periods"][1]["duration"] = json!(0.03);
        std::fs::write(&document, tree.to_string()).unwrap();
        let refused = run_asks(&host, "scan#1", "pulse_run", played(&document, 1)).unwrap_err();
        assert!(refused.contains(PULSE_CHANGED), "{refused}");
        let other = run_asks(&host, "scan#2", "pulse_run", played(&document, 0)).unwrap();
        assert!(other["command_id"].is_u64());
        run_asks(&host, "scan#2", "pulse_stop", json!({})).unwrap();
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

    /// A run that ends with its FIRE unreported -- its process died, say -- leaves the board SAFE.
    #[test]
    fn a_run_ended_without_its_stop_leaves_the_board_safe() {
        let (host, events, document) = host("ended");
        claim(&host, &["scan#1"]);
        let id = run_asks(&host, "scan#1", "pulse_run", played(&document, 0)).unwrap()["command_id"].as_u64().unwrap() as u32;
        assert_eq!(board(&host).last_fire(), Some((id, Fate::Pending)), "still running");
        let finished = HostEvent::RunFinished { instance: "scan#1".into(), run: "scan#1".into(), outcome: "failed".into(), result: Value::Null,
                                                error: None, generation: None };
        assert!(host.desk.hears(finished).is_some(), "the window still hears a run's end");
        loop {
            if let PulseEvent::Log { line, .. } = events.recv_timeout(Duration::from_secs(30)).expect("the SAFE") { assert!(line.contains("ZLC SAFE after scan#1"), "{line}"); break; }
        }
        assert_eq!(board(&host).last_fire(), Some((id, Fate::Safed)));
    }

    /// Closing the window does not wait for the board: its SAFE close runs on the SAFE lane, however
    /// slow, and is told when done; the application's end waits for it, so the board is SAFE first.
    #[test]
    fn the_windows_close_makes_the_board_safe_off_the_turn_and_its_end_waits_for_it() {
        struct SlowSafe(Mutex<Option<std::time::Instant>>);
        impl World for SlowSafe {
            fn play_run(&self, _timeline: &Json, _stop: &Arc<Event>) -> zlc_pulse::Result<()> { Ok(()) }
            fn safe(&self) { std::thread::sleep(Duration::from_millis(600)); *lock(&self.0) = Some(std::time::Instant::now()); }
        }
        for awaited in [false, true] {
            let (mut host, events, _document) = host(if awaited { "shutdown_awaited" } else { "shutdown" });
            let world = Arc::new(SlowSafe(Mutex::new(None)));
            board(&host).set_world(Some(world.clone()));
            let began = std::time::Instant::now();
            host.shutdown();
            host.shutdown();
            assert!(began.elapsed() < Duration::from_millis(300), "the turn goes on: {:?}", began.elapsed());
            assert!(!host.shut_down() && lock(&world.0).is_none(), "the board is still closing");
            if awaited {
                // The application ends while it closes: its end waits for the SAFE.
                host.await_shutdown();
            } else {
                loop { if let PulseEvent::ShutDown = events.recv_timeout(Duration::from_secs(30)).expect("the close is told") { break; } }
            }
            assert!(host.shut_down());
            let safed = lock(&world.0).expect("SAFE at the window's close");
            assert!(safed - began >= Duration::from_millis(600));
        }
    }

    /// The local board's UART is found on Init (it binds that port): a JTAG or memory board has
    /// nothing to find, and its server never probes -- `auto` with none found is the JTAG cable,
    /// `uart` without a port is refused, and a named port is the only one it opens.
    #[test]
    fn the_local_board_is_found_on_init_and_its_server_never_probes() {
        let local = device_descriptors().into_iter().find(|item| item["type_id"] == "sequencer.local").unwrap();
        assert_eq!((&local["resources"], &local["queries"]), (&json!(["serial:{uart_port}"]), &json!(["resolve"])));
        for backend in ["jtag-axi", "memory"] {
            let answer = resolve("sequencer.local", &json!({"backend": backend, "uart_port": ""}), &["serial:COM3".to_owned()], Duration::from_secs(1));
            assert_eq!(answer.unwrap(), json!({"parameters": {}, "note": ""}), "{backend}");
        }
        let line = |parameters: Value| server_arguments(&parameters).map(|(backend, arguments)| (backend.to_owned(), arguments.join(" ")));
        let (backend, arguments) = line(json!({"backend": "auto", "uart_port": " ", "port": 18861})).unwrap();
        assert!(backend == "jtag-axi" && arguments.ends_with("--backend jtag-axi --port 18861"), "{arguments}");
        let (backend, arguments) = line(json!({"backend": "auto", "uart_port": "COM5", "port": 18861})).unwrap();
        assert!(backend == "auto" && arguments.ends_with("--backend auto --port 18861 --uart-port COM5"), "{arguments}");
        assert_eq!(line(json!({"backend": "uart", "uart_port": ""})).unwrap_err(), "no UART port: Init finds it, or set one");
        assert!(line(json!({"backend": "uart", "uart_port": "COM5"})).unwrap().1.ends_with("--backend uart --port 0 --uart-port COM5"));
    }

    /// v2 slm_rearrangement: the Pulse is loaded first and the FIRE is then the FIRE alone.
    #[test]
    fn a_run_loads_then_fires() {
        let (host, _events, document) = host("stages");
        claim(&host, &["rearrangement#1"]);
        let mut load = played(&document, 1);
        load["stage"] = json!("load");
        let loaded = run_asks(&host, "rearrangement#1", "pulse_run", load).unwrap();
        assert!(loaded.get("command_id").is_none() && loaded["timing"]["periods"].is_array(), "{loaded}");
        assert_eq!(board(&host).last_fire(), None, "loaded, not fired");
        let mut fire = played(&document, 1);
        fire["stage"] = json!("fire");
        let fired = run_asks(&host, "rearrangement#1", "pulse_run", fire).unwrap();
        let id = fired["command_id"].as_u64().unwrap();
        let done = run_asks(&host, "rearrangement#1", "pulse_wait_done", json!({"command_id": id, "timeout_seconds": 30})).unwrap();
        assert_eq!(done["status"], "done");
    }

    #[test]
    fn split_scan_receipt_preserves_loaded_columns_and_api_values() {
        let (host, _events, document) = host("scan_receipt");
        claim(&host, &["scan#1"]);
        let mut load=played(&document,1);
        load["stage"]=json!("load");
        load["api_values"]=json!({"duration:short":[0.000100003,"s"]});
        load["scan"]=json!({"fields":["duration:long_after","duration:long_before"],
            "rows":[[0.000400003,0.000200007],[0.000500003,0.000300007]]});
        let loaded=run_asks(&host,"scan#1","pulse_run",load).unwrap();
        let mut fire=played(&document,1);fire["stage"]=json!("fire");
        let fired=run_asks(&host,"scan#1","pulse_run",fire.clone()).unwrap();
        let done=run_asks(&host,"scan#1","pulse_wait_done",json!({"command_id":fired["command_id"],"timeout_seconds":30})).unwrap();
        assert_eq!(done["status"],"done");
        assert_eq!(fired["scan"],loaded["scan"],"FIRE must retain LOAD's named, quantized table");
        assert_eq!(fired["api_values"],loaded["api_values"],"calibration consumes these too");
        assert!(fired["scan"]["rows"].is_array());
        assert_ne!(fired["scan"]["rows"][0][0],json!(0.000400003),"test must exercise hardware quantization");
        if let Some(path)=std::env::var_os("ZLC_SCAN_RECEIPT_FIXTURE") {
            std::fs::write(path,json!({"loaded":loaded,"fired":fired,"done":done}).to_string()).unwrap();
        }
        // Replacing only the table also invalidates the staged load.
        let board=board(&host);let held=board.applied().unwrap().unwrap();
        let mut rows=(*held.rows).clone();rows.reverse();
        board.load(held.program.clone(),held.source.clone(),rows).unwrap();
        let refused=run_asks(&host,"scan#1","pulse_run",fire).unwrap_err();
        assert!(refused.contains("loaded"),"{refused}");
        let mut load=played(&document,1);load["stage"]=json!("load");
        load["api_values"]=json!({"duration:short":[0.0002,"s"]});
        let loaded=run_asks(&host,"scan#1","pulse_run",load).unwrap();
        let mut fire=played(&document,1);fire["stage"]=json!("fire");
        let again=run_asks(&host,"scan#1","pulse_run",fire.clone()).unwrap();
        assert!(again.get("scan").is_none(),"an API-only LOAD must not retain an earlier scan table");
        assert_eq!(again["api_values"],loaded["api_values"]);
        run_asks(&host,"scan#1","pulse_wait_done",json!({"command_id":again["command_id"],"timeout_seconds":30})).unwrap();
        let direct=run_asks(&host,"scan#1","pulse_run",played(&document,1)).unwrap();
        run_asks(&host,"scan#1","pulse_wait_done",json!({"command_id":direct["command_id"],"timeout_seconds":30})).unwrap();
        assert!(run_asks(&host,"scan#1","pulse_run",fire).unwrap_err().contains("load it first"),"a combined run supersedes any staged LOAD");
    }

    /// The Pulse Editor's open and save read and write their file off the window's turn: the
    /// command waits for its file, and its answer for the file it wrote.
    #[test]
    fn open_and_save_read_and_write_their_files_off_the_window() {
        let (mut host, events, document) = host("files");
        let path = document.display().to_string();
        assert!(matches!(host.command("pulse_open", &json!({"path": path, "discard_changes": true}), false, 7, None), Handled::Later), "the file is read first");
        let PulseEvent::Read(ahead) = events.recv_timeout(Duration::from_secs(30)).unwrap() else { panic!("the read-ahead") };
        let (request, opened) = host.read(*ahead);
        assert_eq!(request, 7);
        let Handled::Now(Ok(opened)) = opened else { panic!("opened") };
        assert_eq!(opened["path"], json!(path));
        let copy = document.with_file_name("copy.json");
        assert!(matches!(host.command("pulse_save", &json!({"path": copy.display().to_string()}), false, 8, None), Handled::Later), "written first");
        assert!(host.view().is_none(), "no view while its file is being written");
        let written = loop {
            match events.recv_timeout(Duration::from_secs(30)).unwrap() {
                PulseEvent::Written(written) => break written,
                PulseEvent::Outcome(outcome) => host.deliver(outcome),
                _ => {}
            }
        };
        let (request, saved) = host.written(*written);
        assert_eq!((request, saved.unwrap()["path"].as_str().map(str::to_owned)), (8, Some(copy.display().to_string())));
        assert!(copy.is_file());
        assert!(host.view().is_some_and(|view| view["dirty"] == false), "saved");
        // A file that cannot be written leaves the draft unsaved, and says why.
        let blocked = document.with_file_name("missing").join("deeper");
        std::fs::write(document.with_file_name("missing"), b"a file, not a folder").unwrap();
        assert!(ask(&mut host, &events, "pulse_save", json!({"path": blocked.display().to_string()}), None).unwrap_err().contains("Cannot save"));
        assert_eq!(host.editor.path, copy.display().to_string(), "still the file it was saved to");
    }

    #[test]
    fn native_scan_plugin_and_real_board_keep_tables_out_of_python_receipts() {
        use zlc_core::{DType,OwnedBuffer};
        use zlc_core::control::Origin;
        use zlc_core::dataset::{DataBlock,Validity};
        use zlc_core::runtime::{Bundle,Declaration,Input,Runtime,Storage};
        use zlc_core::schema::{Axis,Coordinates,DatasetSchema,Domain,ValueSchema,ValidityContract};
        use zlc_plugin_host::{HostConfig,RunRequest};
        let Some(python)=std::env::var_os("ZLC_PLUGIN_PYTHON") else { return };
        let (pulse,_events,document)=host("native_scan_bridge");
        claim(&pulse,&["scan"]);
        let root=PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let runtime=Runtime::new();
        let (sender,plugin_events)=channel(); let sender=Mutex::new(sender);
        // As the window's loop is reached: through the desk, which answers the board's commands itself.
        let desk=pulse.desk.clone();
        let plugins=PluginHost::new(HostConfig {python:python.into(),python_path:vec![root.join("../../science/src"),root.join("../zlc-plugin-host/tests/plugins")],
            packages:vec!["zlc_test_plugins".into()],memory_limit_mib:Some(3072)},runtime.clone(),SafeActions::new(),
            move|event|{if let Some(event)=desk.hears(event){let _=sender.lock().unwrap().send(event);}}).unwrap();
        crate::plugin_services::install(&plugins).unwrap();
        pulse.desk.set_plugins(Some(plugins.clone()));
        let singleton=Domain::new(vec![1],vec![],vec![]).unwrap();
        let site=Axis::new("site".into(),"site".into(),"coordinate".into(),Coordinates::Index {origin:0,len:1},None,None,None,None).unwrap();
        let schema=DatasetSchema::new(singleton.clone(),singleton,Domain::dense(vec![1],vec![site]).unwrap(),
            ValueSchema::new(DType::U16,None,None,ValidityContract::Value).unwrap()).unwrap();
        let source=runtime.begin("camera",vec![Declaration {signal:"source".into(),schema:schema.clone(),storage:Storage::Latest}],json!({})).unwrap();
        plugins.open("scan","native_scan",json!({}),json!({})).unwrap();
        let spec=json!({"axes":[{"port":"pulse:param:duration:long_before","identity":"duration:long_before","label":"Before","unit":"s",
            "values":[0.000200003,0.000300007],"coordinate_of":null,"host_advanced":false}],"scan_repeats":2,"run_repeats":2,"pulse_path":document});
        let run=plugins.start_run("scan",RunRequest {prepared:None,device_labels:Default::default(),device_units:Default::default(),devices:Default::default(),shared: Default::default(), reads: Default::default(),
            values:json!({"spec":spec.to_string(),"mode":"seamless","configure_first":true,"effective_inner":""}),origin:Origin::Operator,
            resources:json!({}),source_signal:Some("source".into()),run_directory:None,outputs:[("scan".into(),"output".into())].into(),record_devices:false}).unwrap();
        // The FIRE is followed by its shots, as a camera it triggers would deliver them.
        let mut fired=false;
        let (outcome,result,error)=loop {
            if !fired && board(&pulse).last_fire().is_some() {
                fired=true;
                for ordinal in 0..8 {
                    let data=DataBlock::new(schema.clone(),OwnedBuffer::from_vec(vec![ordinal as u16]).into(),Validity::Uniform(true),None).unwrap();
                    runtime.commit(&source,&Bundle {inputs:vec![Input::Event {signal:"source".into(),ordinal,data,origin:None,shot_time:None}],
                        metadata:json!({}),parents:vec![]}).unwrap();
                }
            }
            match plugin_events.recv_timeout(Duration::from_millis(20)) {
                Ok(HostEvent::RunFinished {run:id,outcome,result,error,..}) if id==run=>break(outcome,result,error),
                Ok(HostEvent::Request {op,params,..}) => panic!("the window was asked {op} {params}: the desk answers the board's commands"),
                Ok(HostEvent::Failed {reason,..})=>panic!("plugin failed: {reason}"),
                _=>{},
            }
        };
        assert_eq!(outcome,"completed","{error:?}"); assert!(fired);
        assert_eq!(result,json!({"points":2,"shots":8}));
        let output=runtime.signals().into_iter().find(|signal|&*signal.signal=="output").unwrap();
        let latest=runtime.read_latest(&output.generation,"output").unwrap();
        assert_eq!(latest.canonical().unwrap().written_cells(),8);
        assert_eq!(latest.run().metadata["device_snapshots"]["sequencer"]["program"]["rows"].as_array().unwrap().len(),2);
        assert_ne!(latest.run().metadata["plan"]["axes"][0]["values"][0],json!(0.000200003));
        plugins.shutdown(); let mut pulse=pulse; pulse.shutdown();
    }

    /// A plugin host for a test of runs that watch the board, its events (as the window would hear
    /// them, past the desk) in a channel.
    fn watching_host(pulse: &PulseHost, python: std::ffi::OsString) -> (PluginHost, Receiver<HostEvent>) {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let (sender, plugin_events) = channel(); let sender = Mutex::new(sender);
        let desk = pulse.desk.clone();
        let plugins = PluginHost::new(zlc_plugin_host::HostConfig { python: python.into(), python_path: vec![root.join("../../science/src"), root.join("../zlc-plugin-host/tests/plugins")],
            packages: vec!["zlc_test_plugins".into()], memory_limit_mib: Some(3072) }, zlc_core::runtime::Runtime::new(), SafeActions::new(),
            move |event| { if let Some(event) = desk.hears(event) { let _ = sender.lock().unwrap().send(event); } }).unwrap();
        crate::plugin_services::install(&plugins).unwrap();
        pulse.desk.set_plugins(Some(plugins.clone()));
        (plugins, plugin_events)
    }

    fn start(plugins: &PluginHost, instance: &str, values: Value) -> String {
        plugins.start_run(instance, zlc_plugin_host::RunRequest { prepared: None, device_labels: Default::default(), device_units: Default::default(), devices: Default::default(),
            shared: Default::default(), reads: Default::default(), values, origin: zlc_core::control::Origin::Operator, resources: json!({}), source_signal: None,
            run_directory: None, outputs: Default::default(), record_devices: false }).unwrap()
    }

    /// The desk's watch threads end once their runs have (none is left waiting for a board change).
    fn no_watch_left(pulse: &PulseHost) {
        let until = std::time::Instant::now() + Duration::from_secs(10);
        while pulse.desk.watching() > 0 { assert!(std::time::Instant::now() < until, "a watch outlived its run"); std::thread::sleep(Duration::from_millis(10)); }
    }

    /// A run that watches a lane's schedule (ctx.watch) asks once; each later change reaches it as
    /// the board makes it -- a FIRE played meanwhile is told without the run asking again -- and its
    /// watch ends with it, not at the board's next change.
    #[test]
    fn a_watching_run_is_told_each_play_without_asking() {
        let Some(python) = std::env::var_os("ZLC_PLUGIN_PYTHON") else { return };
        let (pulse, _events, document) = host("watch");
        claim(&pulse, &["scan#1"]);
        // A first play, so the lane that rises in this pulse is known.
        let first = run_asks(&pulse, "scan#1", "pulse_run", played(&document, 1)).unwrap();
        run_asks(&pulse, "scan#1", "pulse_wait_done", json!({"command_id": first["command_id"], "timeout_seconds": 30})).unwrap();
        let counts = board(&pulse).trigger_counts().unwrap().1.unwrap();
        let lane = board(&pulse).trigger_lanes()[counts.iter().position(|count| *count > 0).unwrap()].0.clone();
        let (plugins, plugin_events) = watching_host(&pulse, python);
        plugins.open("camera", "watcher", json!({}), json!({})).unwrap();
        let run = start(&plugins, "camera", json!({"lane": lane}));
        let mut fired = None;
        let (outcome, result, error) = loop {
            match plugin_events.recv_timeout(Duration::from_secs(30)).expect("a watcher event") {
                HostEvent::Request { op, params, .. } => panic!("the window was asked {op} {params}: the desk answers the schedule"),
                // Watching: the board plays again, and nobody asks it what.
                HostEvent::RunReady { run: id, .. } if id == run => {
                    fired = run_asks(&pulse, "scan#1", "pulse_run", played(&document, 1)).unwrap()["command_id"].as_u64();
                }
                HostEvent::RunFinished { run: id, outcome, result, error, .. } if id == run => break (outcome, result, error),
                HostEvent::Failed { reason, .. } => panic!("plugin failed: {reason}"),
                _ => {}
            }
        };
        assert_eq!(outcome, "completed", "{error:?}");
        assert_eq!(result["states"][0], "done", "answered with what the board held when the watch began");
        assert_eq!((result["states"].as_array().unwrap().last().unwrap(), result["command_id"].as_u64()), (&json!("playing"), fired), "{result}");
        assert_eq!(result["points"], 1);
        no_watch_left(&pulse);
        let id = fired.unwrap();
        run_asks(&pulse, "scan#1", "pulse_wait_done", json!({"command_id": id, "timeout_seconds": 30})).unwrap();
        plugins.shutdown(); let mut pulse = pulse; pulse.shutdown();
    }

    /// A run that watches its FIRE's report is told it when the board has it (nobody polls); one
    /// stopped meanwhile is told its FIRE can no longer report, and no wait outlives it.
    #[test]
    fn a_run_is_told_its_report_and_its_wait_ends_with_it() {
        let Some(python) = std::env::var_os("ZLC_PLUGIN_PYTHON") else { return };
        let (pulse, _events, document) = host("done_watch");
        claim(&pulse, &["fire"]);
        let (plugins, plugin_events) = watching_host(&pulse, python);
        plugins.open("fire", "fire_watcher", json!({}), json!({})).unwrap();
        let finished = |run: &str, stop: bool| loop {
            match plugin_events.recv_timeout(Duration::from_secs(30)).expect("a run event") {
                HostEvent::Request { op, params, .. } => panic!("the window was asked {op} {params}: the desk answers the board's commands"),
                HostEvent::RunReady { run: id, .. } if id == run && stop => { plugins.stop_run("fire", &id).unwrap(); }
                HostEvent::RunFinished { run: id, outcome, result, error, .. } if id == run => break (outcome, result, error),
                HostEvent::Failed { reason, .. } => panic!("plugin failed: {reason}"),
                _ => {}
            }
        };
        let run = start(&plugins, "fire", json!({"document": document.display().to_string(), "repeats": 1}));
        let (outcome, result, error) = finished(&run, false);
        assert_eq!((outcome.as_str(), &result["status"]), ("completed", &json!("done")), "{error:?} {result}");
        assert_eq!(board(&pulse).last_fire(), Some((result["command_id"].as_u64().unwrap() as u32, Fate::Reported)), "the watch took the report");
        no_watch_left(&pulse);
        let run = start(&plugins, "fire", json!({"document": document.display().to_string(), "repeats": 0}));
        let (_, result, _) = finished(&run, true);
        assert_eq!(result["first_done"], false);
        assert!(result["error"].as_str().is_some_and(|error| error.contains("cancelled") || error.contains("can no longer report")), "{result}");
        no_watch_left(&pulse);
        plugins.shutdown(); let mut pulse = pulse; pulse.shutdown();
    }
}
