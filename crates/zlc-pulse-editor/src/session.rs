//! The editor's board: connecting, On Pulse, Hold/Step, Stop, Sync and the
//! board's own answers (v2 `PulseEditorPresenter`'s hardware half). Nothing
//! here blocks the window: device work leaves as a [`Job`] the host runs on
//! its device lane (one at a time, in order), SAFE on a lane of its own, a
//! finite run's completion wait on a third; each comes back as an
//! [`Outcome`] for [`Editor::deliver`], applied only if no later command
//! superseded it.
use std::cell::RefCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use serde_json::{Value, json};
use zlc_pulse::binding::{apply_config_values, resolve_subpulse_sources};
use zlc_pulse::board::Board;
use zlc_pulse::compile::{CompiledProgram, compile_sequence};
use zlc_pulse::device::{AppliedState, BoardDescription, DoneReport};
use zlc_pulse::json::Json;
use zlc_pulse::model::{Sequence, Target};
use zlc_pulse::remote::DEFAULT_PORT;
use zlc_pulse::scan::{ScanTable, prepare_scan_application, resolve_scan_point, scan_columns_for, scan_rows_from_wire, wire_rows};
use zlc_pulse::transport::Event;
use zlc_pulse::{Error, Kind};

use crate::editor::{CONNECTION_GIVEN, CONNECTION_OFFLINE, CONNECTION_REMOTE, CONNECTION_VIRTUAL, Editor, EditorState};
use crate::project::period_counts;
use crate::text::{SEPARATOR, counted, number};

/// What the board says it is doing, asked rather than remembered (`BoardState`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BoardState {
    pub attached: bool,
    /// False when attached but not answering: "I cannot tell" is an answer.
    pub answering: bool,
    pub firing: bool,
    pub loaded: bool,
    pub cursor: Option<u32>,
    pub applied_digest: String,
    pub run_repeats: u64,
    pub scan_repeats: u64,
    pub fault: String,
}

impl BoardState {
    /// `_board_state_for(sequencer)`: one snapshot, one round trip.
    pub fn of(board: &Board) -> BoardState {
        match board.snapshot() {
            Err(error) => BoardState { attached: true, answering: false, fault: error.message, ..Default::default() },
            Ok(snapshot) => BoardState {
                attached: true, answering: true, firing: snapshot.get("firing").and_then(Json::as_bool).unwrap_or(false),
                loaded: snapshot.get("loaded").and_then(Json::as_bool).unwrap_or(false),
                cursor: snapshot.get("cursor").and_then(Json::as_int).map(|cursor| cursor as u32),
                applied_digest: snapshot.get("applied_digest").and_then(Json::as_str).unwrap_or("").to_owned(),
                run_repeats: snapshot.get("run_repeats").and_then(Json::as_int).unwrap_or(1) as u64,
                scan_repeats: snapshot.get("scan_repeats").and_then(Json::as_int).unwrap_or(1) as u64,
                fault: String::new(),
            },
        }
    }

    pub fn to_json(&self) -> Value {
        json!({"attached": self.attached, "answering": self.answering, "firing": self.firing, "loaded": self.loaded, "cursor": self.cursor,
               "applied_digest": self.applied_digest, "run_repeats": self.run_repeats, "scan_repeats": self.scan_repeats, "fault": self.fault})
    }
}

/// What an attached board said about itself, and what was dialled.
#[derive(Clone)]
pub struct Facts { pub description: BoardDescription, pub name: String }

type Work = Box<dyn FnOnce() -> Outcome + Send>;

/// Work for one of the host's lanes.
pub enum Job {
    /// The device lane: one at a time, in order; `done` is set when it ends.
    Device { work: Work, done: Arc<Event> },
    /// SAFE, beside whatever the device lane is doing.
    Safe { work: Work },
    /// A finite run's completion wait.
    Completion { work: Work },
    /// The operator's scan program (Python), run by the host's interpreter.
    Program { source: String, fields: Vec<(String, String)> },
}

impl Job {
    pub(crate) fn program(source: String, fields: Vec<(String, String)>) -> Job { Job::Program { source, fields } }
}

/// What a play gesture was.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Drive { Fire, Hold(usize) }

/// One lane's answer.
pub enum Outcome {
    Connected { operation: u64, mode: String, endpoint: String, released: Option<String>, result: Result<Option<(Arc<Board>, BoardState)>, String> },
    /// `result`: the FIRE's command id, or None when a later command superseded the gesture before it fired.
    Drove { operation: u64, drive: Drive, revision: u64, result: Result<Option<u32>, String>, state: BoardState,
            source: Arc<Sequence>, rows: Vec<Vec<u64>>, finite: bool },
    /// `board`: the board this Stop made safe (a Connect may have replaced it since).
    Stopped { result: Result<(), String>, state: BoardState, board: Arc<Board> },
    Synced { operation: u64, discard: bool, operator: bool, result: Result<(Option<AppliedState>, BoardState), String> },
    Polled { operation: u64, epoch: u64, state: BoardState },
    Completed { run: u64, result: Result<Option<DoneReport>, String>, state: BoardState },
    Program { source: String, fields: Vec<(String, String)>, table: Result<Option<ScanTable>, String> },
}

/// The editor's side of its board.
#[derive(Default)]
pub struct BoardLink {
    pub board: Option<Arc<Board>>,
    pub facts: Option<Facts>,
    /// The bare board's target and clock, shown while no pulse is open.
    pub board_target: Option<Arc<Target>>,
    pub board_step_ns: Option<f64>,
    pub state: BoardState,
    pub mode: String,
    pub endpoint: String,
    pub status: String,
    pub word: &'static str,
    /// The apparatus device the board belongs to, when it is one: the
    /// connection is the session's, not this editor's.
    pub locked: Option<String>,
    pub busy: Option<String>,
    pub stopping: bool,
    /// The latest board command (a later one supersedes it); `current` is it, for the lanes.
    pub operation: u64,
    pub(crate) current: Arc<AtomicU64>,
    /// The command that set `busy`: its delivery ends it, whatever superseded it (v2 `_run_device_command`).
    pub(crate) busy_operation: u64,
    /// A Stop's answer held until the command it superseded has delivered (v2 `_status_followups`).
    pub(crate) deferred_stop: Option<Outcome>,
    /// Bumped when someone else drove the board: a status answer asked before it is stale.
    pub(crate) poll_epoch: u64,
    pub finite_run: Option<u64>,
    /// This editor's own FIRE plays (v2 `_drive_lease`): from its drive until the board stops.
    pub driving: bool,
    /// The run that claims the board, when one does.
    pub claimed_by: Option<String>,
    pub held_point: Option<usize>,
    pub held_program: Option<(u64, String)>,
    /// The table text of what was loaded: digest, column names, each row's words.
    pub applied_scan: Option<(String, Vec<String>, Vec<Vec<String>>)>,
    pub(crate) digest: RefCell<(u64, String)>,
    pub(crate) device_done: Option<Arc<Event>>,
    pub(crate) poll_in_flight: bool,
    pub(crate) changed: bool,
}

impl BoardLink {
    pub fn connection_json(&self) -> Value {
        let choices = match &self.locked {
            Some(label) => json!([{"label": label, "value": CONNECTION_GIVEN, "endpoint_editable": false}]),
            None => json!([{"label": "Virtual (sim)", "value": CONNECTION_VIRTUAL, "endpoint_editable": false},
                           {"label": "Remote server", "value": CONNECTION_REMOTE, "endpoint_editable": true},
                           {"label": "Offline (edit only)", "value": CONNECTION_OFFLINE, "endpoint_editable": false}]),
        };
        json!({"choices": choices, "selected": if self.locked.is_some() { CONNECTION_GIVEN } else if self.mode.is_empty() { CONNECTION_OFFLINE } else { &self.mode },
               "endpoint": self.endpoint, "status": if self.status.is_empty() { "not connected" } else { &self.status },
               "status_word": if self.status.is_empty() { "idle" } else { self.word }, "locked": self.locked.is_some()})
    }

    /// `_status_token`: the dot's answer.
    pub fn status_token(&self, editor: &Editor) -> (&'static str, &'static str) {
        let state = &self.state;
        let token = if !state.attached { "idle" } else if !state.answering { "unreachable" } else if state.firing {
            if self.busy.is_none() && (editor.synchronized() || self.holding_this_pulse(editor)) { "running-synced" } else { "running-stale" }
        } else if editor.state.sequence.is_some() { "dirty-ready" } else { "idle" };
        let word = match token { "running-synced" => "running", "running-stale" => "warning", "unreachable" => "error", _ => "idle" };
        (token, word)
    }

    fn holding_this_pulse(&self, editor: &Editor) -> bool {
        self.state.firing && self.held_program.as_ref() == Some(&(editor.revision, self.state.applied_digest.clone()))
            && (self.state.run_repeats, self.state.scan_repeats) == (0, 1)
    }

    /// `_render_run_state`'s control facts.
    pub fn control(&self, editor: &Editor) -> Value {
        let attached = self.board.is_some();
        let live = attached && editor.state.sequence.is_some();
        let idle = self.busy.is_none() && !self.stopping;
        let ok = editor.config_error.is_empty();
        json!({"running": self.state.firing, "synchronized": self.busy.is_none() && editor.synchronized(), "file_dirty": editor.dirty(),
               "can_run": live && idle && ok && self.claimed_by.is_none(), "can_stop": attached && !self.stopping,
               "can_sync": attached && idle, "can_hold": live && idle && ok && self.claimed_by.is_none(),
               "can_step": live && !editor.state.section.scan_rows.is_empty() && idle && ok && self.claimed_by.is_none(),
               "claimed_by": self.claimed_by, "held_point": self.held_point})
    }
}

fn parse_endpoint(endpoint: &str) -> Result<(String, u16), String> {
    let text = endpoint.trim();
    if text.is_empty() { return Err("give the pulse server's address as host:port".into()); }
    match text.rsplit_once(':') {
        Some((host, port)) => Ok((host.trim().to_owned(), port.trim().parse().map_err(|_| format!("{port:?} is not a TCP port"))?)),
        None => Ok((text.to_owned(), DEFAULT_PORT)),
    }
}

/// `_drive_program`: stop what plays (when it plays, or always for Hold),
/// load, fire; a board touched and not handed the whole program goes SAFE.
/// `current` says whether the gesture is still the latest command: a
/// superseded one (a Stop, another command) touches the board no further and
/// leaves it SAFE. The FIRE's command id, or None when it was not fired.
fn drive(board: &Board, program: Arc<CompiledProgram>, source: Arc<Sequence>, rows: Vec<Vec<u64>>, run_repeats: u64, scan_repeats: u64,
         halt_first: bool, current: &dyn Fn() -> bool) -> Result<Option<u32>, String> {
    let mut touched = false;
    let result = (|| -> zlc_pulse::Result<Option<u32>> {
        if current() && (halt_first || board.snapshot().ok().and_then(|s| s.get("firing").and_then(Json::as_bool)).unwrap_or(false)) {
            touched = true;
            board.safe()?;
        }
        if current() { touched = true; board.load(program, Some(source), rows)?; }
        let fired = if current() { Some(board.fire(run_repeats, scan_repeats)?.1) } else { None };
        if !current() && touched { board.safe()?; }
        Ok(fired)
    })();
    result.map_err(|error| {
        if touched {
            if let Err(safe) = board.safe() { return format!("{}; SAFE also failed: {}", error.message, safe.message); }
        }
        error.message
    })
}

/// The table words of what a scan loaded: each axis in its own decimals,
/// a digit-wide minus, padded with figure spaces to its widest.
fn applied_scan_text(source: &Sequence, rows: &[Vec<u64>], digest: &str, geometry: &zlc_pulse::wire::StreamerParams)
    -> Option<(String, Vec<String>, Vec<Vec<String>>)> {
    if rows.is_empty() || source.scan_bindings().is_empty() { return None; }
    let columns = scan_columns_for(source, geometry).ok()?;
    let wire: Vec<Vec<i128>> = rows.iter().map(|row| row.iter().map(|v| *v as i128).collect()).collect();
    let values = scan_rows_from_wire(&wire, &columns);
    let mut texts: Vec<Vec<String>> = Vec::new();
    for column in 0..columns.len() {
        let axis: Vec<f64> = values.iter().map(|row| row[column]).collect();
        let mut shown: Vec<String> = axis.iter().map(|v| zlc_core::pyfmt::format_g(*v, 12)).collect();
        if shown.iter().any(|text| text.contains('e')) {
            let digits = shown.iter().map(|text| text.split('e').next().unwrap_or("").chars().filter(char::is_ascii_digit).count()).max().unwrap_or(1);
            shown = axis.iter().map(|v| zlc_core::pyfmt::format_e(*v, digits.saturating_sub(1))).collect();
        } else {
            let decimals = shown.iter().map(|text| text.split_once('.').map_or(0, |(_, fraction)| fraction.len())).max().unwrap_or(0);
            shown = axis.iter().map(|v| format!("{v:.decimals$}")).collect();
        }
        let shown: Vec<String> = shown.into_iter().map(|text| text.replace('-', "\u{2012}")).collect();
        let widest = shown.iter().map(|text| text.chars().count()).max().unwrap_or(0);
        texts.push(shown.into_iter().map(|text| format!("{}{text}", "\u{2007}".repeat(widest - text.chars().count()))).collect());
    }
    let rows_text = (0..values.len()).map(|row| texts.iter().map(|column| column[row].clone()).collect()).collect();
    Some((digest.to_owned(), columns.iter().map(|c| c.name.clone()).collect(), rows_text))
}

impl Editor {
    fn device_available(&mut self) -> Result<(), String> {
        if self.link.busy.is_some() || self.link.stopping { return Err("A pulse command is already in progress".into()); }
        Ok(())
    }

    /// A new latest command: the one before it is superseded.
    fn supersede(&mut self) {
        self.link.operation += 1;
        self.link.current.store(self.link.operation, Ordering::SeqCst);
    }

    /// Whether `operation` is still the latest command, asked on a lane.
    fn current(&self, operation: u64) -> impl Fn() -> bool + Send + 'static {
        let current = self.link.current.clone();
        move || current.load(Ordering::SeqCst) == operation
    }

    fn submit_device(&mut self, summary: &str, work: Work) {
        self.supersede();
        self.link.busy_operation = self.link.operation;
        self.link.busy = Some(summary.to_owned());
        self.summary_note = None;
        let done = Arc::new(Event::default());
        self.link.device_done = Some(done.clone());
        self.jobs.push(Job::Device { work, done });
    }

    /// The jobs the last commands produced, for the host's lanes.
    pub fn take_jobs(&mut self) -> Vec<Job> { std::mem::take(&mut self.jobs) }

    /// Whether the attached board changed since the last call (the host
    /// moves Stop all's SAFE action with it), and the board now.
    pub fn take_board_change(&mut self) -> Option<Option<Arc<Board>>> {
        if !std::mem::take(&mut self.link.changed) { return None; }
        Some(self.link.board.clone())
    }

    pub fn board(&self) -> Option<&Arc<Board>> { self.link.board.as_ref() }
    pub fn connection_locked(&self) -> bool { self.link.locked.is_some() }

    /// The run that claims the sequencer, which the editor then may not drive.
    /// The run that claims the board now; true when that changed.
    pub fn set_claimed_by(&mut self, run: Option<String>) -> bool { std::mem::replace(&mut self.link.claimed_by, run) != self.link.claimed_by }

    // ------------------------------------------------------------- the attached board
    fn adopt(&mut self, board: Arc<Board>, state: BoardState, mode: &str, endpoint: &str) {
        let description = board.description().clone();
        let name = board.name().to_owned();
        self.link.board = Some(board);
        self.link.changed = true;
        self.link.facts = Some(Facts { description: description.clone(), name: name.clone() });
        self.link.mode = mode.to_owned();
        self.link.endpoint = endpoint.to_owned();
        self.revision += 1;
        if self.state.sequence.is_none() {
            self.link.board_target = Some(description.target.clone());
            self.link.board_step_ns = Some(description.time_step_ns());
        } else {
            self.align_sequence_to(&description);
        }
        self.link.state = state;
        // v2 _connection_name: what was dialled -- "given", "virtual", "remote <endpoint>".
        let label = if mode == CONNECTION_GIVEN { mode.to_owned() } else { name };
        self.link.status = [label, counted(description.target.ports.len(), "port"), counted(description.target.raw_lanes.len(), "lane"),
                            format!("{} MHz", number(description.clock_hz / 1e6))].join(SEPARATOR);
        self.link.word = "running";
    }

    fn forget(&mut self) {
        if self.link.board.is_some() { self.link.changed = true; }
        self.link.board = None;
        self.link.facts = None;
        self.link.board_target = None;
        self.link.board_step_ns = None;
        self.link.state = BoardState::default();
        self.link.finite_run = None;
        self.link.driving = false;
        self.revision += 1;
    }

    /// The apparatus opened its sequencer device: the session's board.
    pub fn attach_given(&mut self, board: Arc<Board>, label: &str) {
        let had_sequence = self.state.sequence.is_some();
        if let Some(old) = self.link.board.take() { if self.link.locked.is_none() { let _ = old.close(); } }
        self.link.locked = Some(label.to_owned());
        // v2 PE:1306: the session's board brings its own Config (the device row's file), which this editor shows as loaded.
        self.config_loaded_path = board.config().source();
        self.config_loaded_values = board.config().values();
        if !self.config_dirty() {
            self.config_path = self.config_loaded_path.clone();
            self.config_rows = Self::rows_of(&self.config_loaded_values);
            self.config_saved_rows = self.config_rows.clone();
        }
        let state = BoardState::of(&board);
        self.adopt(board, state, CONNECTION_GIVEN, "");
        if !had_sequence { self.start_new_pulse(); }
    }

    /// The apparatus closed it: the editor is offline again.
    pub fn detach_given(&mut self) {
        if self.link.locked.take().is_none() { return; }
        self.forget();
        self.link.mode = CONNECTION_OFFLINE.into();
        self.link.status = "edit only".into();
        self.link.word = "idle";
    }

    /// `connect_to(mode, endpoint)`, on the device lane.
    pub fn connect(&mut self, mode: &str, endpoint: &str) -> Result<Value, String> {
        if self.link.locked.is_some() { return Err("the experiment session owns this connection".into()); }
        if ![CONNECTION_VIRTUAL, CONNECTION_REMOTE, CONNECTION_OFFLINE].contains(&mode) { return Err(format!("unknown connection mode {mode:?}")); }
        self.device_available()?;
        let target = if mode == CONNECTION_REMOTE { Some(parse_endpoint(endpoint)?) } else { None };
        let previous = self.link.board.clone();
        let drove = self.link.state.firing || self.link.finite_run.is_some();
        let (mode_owned, endpoint_owned) = (mode.to_owned(), endpoint.to_owned());
        let operation = self.link.operation + 1;
        self.submit_device("Connecting...", Box::new(move || {
            let mut released = None;
            if let Some(previous) = previous {
                if drove { if let Err(error) = previous.safe() { if error.kind == Kind::Connection { released = Some(error.message); } } }
                let _ = previous.close();
            }
            let result = match (mode_owned.as_str(), target) {
                (CONNECTION_OFFLINE, _) => Ok(None),
                (CONNECTION_REMOTE, Some((host, port))) => Board::remote(&host, port).map(Some),
                _ => Board::virtual_board(None).map(Some),
            }.map(|board| board.map(|board| { let board = Arc::new(board); let state = BoardState::of(&board); (board, state) }))
             .map_err(|error| error.message);
            Outcome::Connected { operation, mode: mode_owned, endpoint: endpoint_owned, released, result }
        }));
        Ok(json!({"accepted": true}))
    }

    /// `_execution_request`: the executable document, its wire rows and sweeps.
    fn execution_request(&self) -> zlc_pulse::Result<(Arc<Sequence>, Vec<Vec<u64>>, u64)> {
        let source = self.execution_sequence()?;
        let schema: Vec<(String, String)> = source.scan_bindings().iter().map(|b| (b.field_id(), b.unit.clone())).collect();
        let section = &self.state.section;
        if !section.scan_rows.is_empty() && section.scan_fields != schema {
            return Err(Error::new(Kind::Value, "Scan fields or units changed. Generate or open the scan table again."));
        }
        let armed = !section.scan_rows.is_empty() && !source.scan_bindings().is_empty();
        let (geometry, _) = self.compiler_target()?;
        let rows = if armed { wire_rows(&prepare_scan_application(&source, &section.scan_rows, &geometry)?.1)? } else { Vec::new() };
        Ok((Arc::new(source), rows, if armed { section.scan_repeats } else { 1 }))
    }

    fn require_board(&self) -> Result<Arc<Board>, String> {
        if let Some(holder) = &self.link.claimed_by { return Err(format!("device is in use by {holder}")); }
        self.link.board.clone().ok_or_else(|| "This editor is not connected to a sequencer".to_owned())
    }

    /// `request_fire`: On Pulse.
    pub fn fire(&mut self) -> Result<Value, String> {
        self.last_warning.clear();
        self.device_available()?;
        if self.state.sequence.is_none() { return Err("No pulse is open".into()); }
        let board = self.require_board()?;
        let (source, rows, sweeps) = self.execution_request().map_err(|error| format!("Cannot load this pulse: {}", error.message))?;
        let (geometry, clock) = self.compiler_target().map_err(|e| e.message)?;
        let run_repeats = source.run_repeats;
        let finite = run_repeats > 0 && sweeps > 0;
        let (operation, revision) = (self.link.operation + 1, self.revision);
        let current = self.current(operation);
        self.link.finite_run = None;
        // v2 acquires its drive at the command, before the board is touched.
        self.link.driving = true;
        self.submit_device("Starting...", Box::new(move || {
            let compiled = compile_sequence(&source, &geometry, clock).map(Arc::new).map_err(|error| error.message);
            let result = compiled.and_then(|program| drive(&board, program, source.clone(), rows.clone(), run_repeats, sweeps, false, &current));
            Outcome::Drove { operation, drive: Drive::Fire, revision, result, state: BoardState::of(&board), source, rows, finite }
        }));
        Ok(json!({"accepted": true}))
    }

    /// `hold_scan_point` (None: the row the board is on) and `step_scan_point`.
    pub fn hold(&mut self, delta: Option<i64>) -> Result<Value, String> {
        self.device_available()?;
        if self.state.sequence.is_none() { return Err("No pulse is open".into()); }
        let board = self.require_board()?;
        if !self.scan_armed() { return Err(if delta.is_some() { "Nothing is held to step" } else { "There is no scan to hold a point of" }.into()); }
        let count = self.state.section.scan_rows.len();
        let point = delta.map(|delta| (self.link.held_point.unwrap_or(0) as i64 + delta).clamp(0, count as i64 - 1) as usize);
        let (source, effective) = match self.scan_points() {
            Ok(prepared) => prepared,
            Err(error) => {
                let held = point.unwrap_or(0);
                self.link.held_point = Some(held);
                self.say_scan(format!("Cannot hold that point: {}", error.message), "error");
                return Err(format!("Cannot hold scan point {}: {}", held + 1, error.message));
            }
        };
        let (geometry, clock) = self.compiler_target().map_err(|e| e.message)?;
        let (operation, revision) = (self.link.operation + 1, self.revision);
        let current = self.current(operation);
        self.link.finite_run = None;
        self.link.driving = true;
        self.submit_device("Holding...", Box::new(move || {
            let held = point.unwrap_or_else(|| board.cursor().ok().flatten().map_or(0, |cursor| cursor as usize % count.max(1)));
            let resolved = resolve_scan_point(&source, Some(&effective[held])).map(Arc::new);
            let result = resolved.and_then(|resolved| compile_sequence(&resolved, &geometry, clock).map(|program| (resolved, Arc::new(program))))
                .map_err(|error| error.message);
            let (result, source) = match result {
                Ok((resolved, program)) => (drive(&board, program, resolved.clone(), Vec::new(), 0, 1, true, &current), resolved),
                Err(error) => (Err(error), Arc::new(source)),
            };
            Outcome::Drove { operation, drive: Drive::Hold(held), revision, result, state: BoardState::of(&board), source, rows: Vec::new(), finite: false }
        }));
        Ok(json!({"accepted": true}))
    }

    /// `stop`: SAFE now, whatever the board is doing; never queued behind a
    /// command. A run holding the board refuses it, as v2's device use did:
    /// its node is stopped instead, and its own Stop makes the board safe.
    pub fn stop(&mut self) -> Result<Value, String> {
        self.last_warning.clear();
        if self.link.stopping { return Ok(json!({"accepted": true})); }
        if let Some(holder) = &self.link.claimed_by { return Err(format!("device is in use by {holder}")); }
        let Some(board) = self.link.board.clone() else { return Ok(json!({"accepted": false})) };
        self.supersede();
        self.link.finite_run = None;
        self.link.stopping = true;
        self.summary_note = Some("Stopping...".into());
        let pending = self.link.device_done.clone().filter(|done| !done.is_set());
        self.jobs.push(Job::Safe { work: Box::new(move || {
            let mut result = board.safe().map(|_| ()).map_err(|error| error.message);
            if let Some(pending) = pending {
                // A command already on the lane may still load and fire: wait
                // for it, then SAFE again.
                pending.wait(Some(Duration::from_secs(5)));
                result = board.safe().map(|_| ()).map_err(|error| error.message);
            }
            Outcome::Stopped { result, state: BoardState::of(&board), board }
        }) });
        Ok(json!({"accepted": true}))
    }

    /// `sync_from_sequencer`: what the board holds back into the editor;
    /// `discard` lets it replace unsaved work (v2 asked first).
    pub fn sync(&mut self, discard: bool) -> Result<Value, String> {
        self.device_available()?;
        let board = self.link.board.clone().ok_or("This editor is not connected to a sequencer")?;
        let (operation, operator) = (self.link.operation + 1, self.operator);
        self.submit_device("Syncing...", Box::new(move || {
            let result = board.applied().map(|applied| (applied, BoardState::of(&board))).map_err(|error| error.message);
            Outcome::Synced { operation, discard, operator, result }
        }));
        Ok(json!({"accepted": true}))
    }

    /// `ask_run_state`: one status question when nothing else is in flight.
    pub fn poll(&mut self) {
        if self.link.poll_in_flight || self.link.busy.is_some() || self.link.stopping { return; }
        let Some(board) = self.link.board.clone() else { return };
        self.link.poll_in_flight = true;
        let (operation, epoch) = (self.link.operation, self.link.poll_epoch);
        self.jobs.push(Job::Device { work: Box::new(move || Outcome::Polled { operation, epoch, state: BoardState::of(&board) }), done: Arc::new(Event::default()) });
    }

    /// Something else (a node's run) drove the board: a status answer asked
    /// before it is stale, so ask again. The editor's own command in flight
    /// is not superseded by it.
    pub fn board_touched(&mut self) { self.link.poll_epoch += 1; self.link.finite_run = None; self.poll(); }

    /// The draft as Hold plays a row of it: the executable pulse with the
    /// armed table's fields, and the table quantized as the board plays it.
    fn scan_points(&self) -> zlc_pulse::Result<(Sequence, Vec<Vec<f64>>)> {
        let sequence = self.state.sequence.as_ref().ok_or_else(|| Error::new(Kind::Runtime, "no pulse is open"))?;
        let resolved = resolve_subpulse_sources(sequence, self.base_directory().as_deref())?;
        let source = self.played_scan(&zlc_pulse::binding::resolve_api_parameters(&resolved, None)?)?;
        let (geometry, _) = self.compiler_target()?;
        let effective = prepare_scan_application(&source, &self.state.section.scan_rows, &geometry)?.0;
        Ok((source, effective))
    }

    /// `_validate_control`: compile what On Pulse would play, with the loaded
    /// Config, without touching the board.
    pub fn validate(&self) -> Result<Value, String> {
        let (source, rows, sweeps) = self.execution_request().map_err(|e| e.message)?;
        let filled = apply_config_values(&source, &self.active_config_values(), None).map_err(|e| e.message)?.0;
        let program = self.compile(&filled).map_err(|e| e.message)?;
        Ok(json!({"valid": true, "revision": self.revision, "digest": program.digest(), "duration_seconds": program.duration_seconds,
                  "scan_rows": rows, "scan_repeats": sweeps,
                  "scope": "compilation using loaded Config values; not applied board evidence"}))
    }

    /// `synchronized`: is the board holding what the editor shows?
    pub fn synchronized(&self) -> bool {
        let state = &self.link.state;
        if state.applied_digest.is_empty() || self.state.sequence.is_none() { return false; }
        let shown = {
            let mut cache = self.link.digest.borrow_mut();
            if cache.0 != self.revision + 1 {
                let digest = self.execution_sequence().and_then(|source| self.effective_sequence(&source))
                    .and_then(|effective| self.compile(&effective)).map(|program| program.digest()).unwrap_or_default();
                *cache = (self.revision + 1, digest);
            }
            cache.1.clone()
        };
        if shown.is_empty() || shown != state.applied_digest { return false; }
        if !state.firing { return true; }
        let sweeps = if self.scan_armed() { self.state.section.scan_repeats } else { 1 };
        self.execution_sequence().is_ok_and(|source| state.run_repeats == source.run_repeats && state.scan_repeats == sweeps)
    }

    fn adopt_board_state(&mut self, state: BoardState) {
        if !state.firing { self.link.driving = false; }
        self.link.state = state;
    }

    /// v2 `_drive_lease`: the board plays this editor's own pulse, so no run may take it
    /// ("device is in use by PulseGUI") until Stop or its finite run ends.
    pub fn drives(&self) -> bool { self.link.driving && self.link.claimed_by.is_none() }

    /// One lane's answer, applied if still current.
    pub fn deliver(&mut self, outcome: Outcome) {
        match outcome {
            Outcome::Connected { operation, mode, endpoint, released, result } => {
                // Delivered whatever superseded it (v2 always_deliver): the old board is
                // hung up and the new one dialled, and only this takes the one and drops the other.
                self.command_delivered(operation);
                if let Some(error) = released {
                    self.warn(format!("The pulse server connection ended before the previous board could be told to stop ({error}). If that server \
                                       is still running it drives the board safe when a client disconnects; if it is not, THAT SEQUENCE IS STILL \
                                       PLAYING and the board must be stopped another way."));
                }
                let had_sequence = self.state.sequence.is_some();
                self.forget();
                match result {
                    Ok(None) => {
                        self.link.mode = CONNECTION_OFFLINE.into();
                        self.link.endpoint = endpoint;
                        self.link.status = "edit only".into();
                        self.link.word = "idle";
                        self.done("Disconnected; this editor is now edit only");
                    }
                    Ok(Some((board, state))) => {
                        self.adopt(board, state, &mode, &endpoint);
                        if !had_sequence { self.start_new_pulse(); }
                    }
                    Err(error) => {
                        self.link.mode = CONNECTION_OFFLINE.into();
                        self.link.endpoint = endpoint.clone();
                        self.link.status = format!("failed: {error}");
                        self.link.word = "error";
                        let name = if mode == CONNECTION_REMOTE { format!("remote {endpoint}") } else { mode };
                        self.warn(format!("Cannot connect to {name}: {error}"));
                    }
                }
                self.deliver_deferred_stop();
            }
            Outcome::Drove { operation, drive, revision, result, state, source, rows, finite } => {
                self.command_delivered(operation);
                // A superseded gesture left the board SAFE; what superseded it reports the board.
                if operation != self.link.operation { self.deliver_deferred_stop(); return; }
                self.adopt_board_state(state.clone());
                // v2: what the board reports holding IS what this revision plays -- its Config
                // filled in -- not the digest of the editor's own compile.
                let digest = state.applied_digest.clone();
                match (result, drive) {
                    (Ok(fired), Drive::Fire) => {
                        self.link.held_point = None;
                        let geometry = self.compiler_target().map(|(g, _)| g).ok();
                        self.link.applied_scan = geometry.and_then(|geometry| applied_scan_text(&source, &rows, &digest, &geometry));
                        if revision == self.revision { *self.link.digest.borrow_mut() = (revision + 1, digest); }
                        self.summary_note = Some("Started".into());
                        if let (true, Some(command_id)) = (finite, fired) {
                            self.link.finite_run = Some(operation);
                            if let Some(board) = self.link.board.clone() {
                                self.jobs.push(Job::Completion { work: Box::new(move || {
                                    let result = board.wait_done(None, Some(command_id)).map_err(|error| error.message);
                                    Outcome::Completed { run: operation, result, state: BoardState::of(&board) }
                                }) });
                            }
                        }
                    }
                    (Ok(_), Drive::Hold(held)) => {
                        let count = self.state.section.scan_rows.len();
                        self.link.held_point = Some(held);
                        self.link.held_program = Some((revision, digest));
                        self.say_scan(format!("Held at scan point {} of {count}", crate::text::figure_padded(&(held + 1).to_string(), &count.to_string())), "task");
                    }
                    (Err(error), Drive::Fire) => {
                        let message = format!("Firing stopped: {error}");
                        self.summary_note = Some(message.clone());
                        self.warn(message);
                    }
                    (Err(error), Drive::Hold(held)) => {
                        self.link.held_point = Some(held);
                        self.say_scan(format!("Cannot hold that point: {error}"), "error");
                        self.warn(format!("Cannot hold scan point {}: {error}", held + 1));
                    }
                }
            }
            Outcome::Stopped { result, state, board } => {
                // Heard after the command it superseded, which may yet change the board (v2).
                if self.link.busy.is_some() {
                    self.link.deferred_stop = Some(Outcome::Stopped { result, state, board });
                    return;
                }
                self.link.stopping = false;
                // A Connect delivered meanwhile: the board this Stop made safe is no longer
                // this editor's, and the Connect said what became of it.
                if !self.link.board.as_ref().is_some_and(|current| Arc::ptr_eq(current, &board)) { return; }
                let worked = result.is_ok() && state.answering && !state.firing;
                self.adopt_board_state(state);
                if worked { self.summary_note = Some("Stopped".into()); }
                else {
                    let message = match result { Err(error) => format!("Stop failed: {error}"), Ok(()) => "The board did not confirm SAFE".into() };
                    self.summary_note = Some(message.clone());
                    self.warn(message);
                }
            }
            Outcome::Synced { operation, discard, operator, result } => {
                self.command_delivered(operation);
                // v2: a superseded Sync is not delivered; what superseded it reports the board.
                if operation != self.link.operation { self.deliver_deferred_stop(); return; }
                match result {
                    Err(error) => self.warn(format!("Cannot sync from the board: {error}")),
                    Ok((applied, state)) => { self.adopt_board_state(state); self.adopt_applied(applied, discard, operator); }
                }
            }
            Outcome::Polled { operation, epoch, state } => {
                self.link.poll_in_flight = false;
                if operation == self.link.operation && epoch == self.link.poll_epoch && self.link.busy.is_none() { self.adopt_board_state(state); }
            }
            Outcome::Completed { run, result, state } => {
                if self.link.finite_run != Some(run) { return; }
                self.link.finite_run = None;
                match result {
                    Err(error) => self.warn(format!("Finite pulse completion failed: {error}")),
                    Ok(Some(report)) if !report.fault().is_empty() => {
                        self.warn(format!("Finite pulse stopped: {}", report.fault()));
                        self.adopt_board_state(state);
                        let _ = self.stop();
                        return;
                    }
                    Ok(_) => {}
                }
                self.adopt_board_state(state);
            }
            Outcome::Program { source, fields, table } => self.scan_program_finished(&source, &fields, table),
        }
    }

    /// A device command's delivery: the busy it set is over, whatever
    /// superseded it (v2 `_run_device_command.finished`).
    fn command_delivered(&mut self, operation: u64) {
        if self.link.busy_operation == operation { self.link.busy = None; }
    }

    /// A Stop held back for the command it superseded, now that it delivered.
    fn deliver_deferred_stop(&mut self) {
        if self.link.busy.is_none() { if let Some(stop) = self.link.deferred_stop.take() { self.deliver(stop); } }
    }

    /// `_adopt_applied`: the board's pulse as the draft (asked first when it
    /// would replace unsaved work; a board playing this draft, or one row of
    /// its scan as Hold and Step resolved it, keeps the draft and its table).
    fn adopt_applied(&mut self, applied: Option<AppliedState>, discard: bool, operator: bool) {
        let Some(applied) = applied else { self.warn("The board has no pulse applied yet; there is nothing to sync"); return };
        let Some(source) = applied.source.clone() else {
            self.warn("The board is holding a compiled program it was given without its pulse, so there is nothing an editor can show");
            return;
        };
        let adopted = applied.authored_source.clone().unwrap_or_else(|| source.clone());
        let playing_this = self.state.sequence.as_ref().is_some_and(|current| {
            **current == *adopted || self.execution_sequence().is_ok_and(|execution| execution == *adopted)
        });
        if playing_this {
            self.done(format!("Synced from the board{SEPARATOR}{}", period_counts(&source.periods)));
            return;
        }
        if let Some(held) = self.link.held_point.filter(|_| self.scan_armed()) {
            let point = self.scan_points().ok().and_then(|(source, effective)| {
                effective.get(held).and_then(|row| resolve_scan_point(&source, Some(row)).ok())
            });
            if point.is_some_and(|point| point == *adopted) {
                self.done(format!("The board is holding scan point {} of this pulse; the draft and its scan table are kept", held + 1));
                return;
            }
        }
        if self.dirty() && !discard {
            // v2 asked whether to discard (confirm_pulse_discard): the window asks, then syncs again.
            if operator { self.ask_sync_discard(); }
            else { self.warn("This editor has unsaved changes; save them, or pass discard_changes=true to replace them with the board's pulse"); }
            return;
        }
        let geometry = self.compiler_target().map(|(g, _)| g).ok();
        let rows: Vec<Vec<f64>> = match (&geometry, applied.rows.is_empty()) {
            (Some(geometry), false) => scan_columns_for(&source, geometry).map(|columns| {
                let wire: Vec<Vec<i128>> = applied.rows.iter().map(|row| row.iter().map(|v| *v as i128).collect()).collect();
                scan_rows_from_wire(&wire, &columns)
            }).unwrap_or_default(),
            _ => Vec::new(),
        };
        let fields: Vec<(String, String)> = if rows.is_empty() { Vec::new() } else {
            resolve_subpulse_sources(&source, None).map(|s| s.scan_bindings().iter().map(|b| (b.field_id(), b.unit.clone())).collect()).unwrap_or_default()
        };
        let scan_dirty = !self.state.section.scan_source.is_empty();
        let sweeps = applied.scan_repeats;
        let points = applied.rows.len();
        let mut candidate = EditorState { sequence: Some(adopted), section: self.state.section.clone() };
        candidate.section.scan_rows = rows;
        candidate.section.scan_fields = fields;
        candidate.section.scan_source_dirty = scan_dirty;
        candidate.section.scan_repeats = sweeps;
        self.accept_state(candidate);
        self.done(format!("Synced from the board{SEPARATOR}{}{}", period_counts(&source.periods),
                          if points > 0 { format!("{SEPARATOR}{}", counted(points, "scan point")) } else { String::new() }));
    }
}
