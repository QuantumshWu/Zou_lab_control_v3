//! The pulse board an application drives: the virtual board (the device on
//! its memory twin, played at wall time into a simulated world) or a remote
//! pulse server. One surface for both, the one the editor's On Pulse, a
//! node's `pulse_run` and Stop all use.
//!
//! The virtual board is v2's `VirtualPulseStreamer`: the real `PulseStreamer`
//! on a memory transport that does not finish by itself, and a playback thread
//! that advances STATUS and CURSOR the way the RTL does, one run at a time,
//! handing each run to the [`World`] (the simulation's `simulation.board`).
use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use zlc_core::journal::{self, json, Value};

use crate::codec::sequence_to_tree;
use crate::compile::CompiledProgram;
use crate::device::{AppliedState, BoardDescription, ConfigHolder, DoneReport, PulseStreamer, SafeReadback};
use crate::error::{Error, Kind, Result};
use crate::json::Json;
use crate::manifest::target_from_config;
use crate::model::Sequence;
use crate::remote::RemotePulseStreamer;
use crate::schedule::{run_duration_seconds, trigger_edge_ticks, trigger_windows_by_channel};
use crate::transport::{Event, MemoryRegisterTransport, RegisterTransport};
use crate::wire::{STATUS_DONE, STATUS_ERROR, STATUS_RUNNING, load_streamer_config, require_streamer_config};

/// The world side of the virtual board: what one played run does.
pub trait World: Send + Sync {
    /// Play one run from now (its [`run_timeline`]); answer when its physical
    /// duration has passed or `stop` was set.
    fn play_run(&self, timeline: &Json, stop: &Arc<Event>) -> Result<()>;
    /// The board went SAFE.
    fn safe(&self) {}
}

fn int(value: impl Into<i128>) -> Json { Json::Int(value.into()) }

/// One run's timeline for the world (`SimulationWorld.fire`'s input): every
/// digital output's high windows and every DAC bus's actions, in board ticks.
pub fn run_timeline(program: &CompiledProgram, point: Option<&[u64]>) -> Result<Json> {
    let table: Option<Vec<Vec<u64>>> = point.map(|point| vec![point.to_vec()]);
    let channels: Vec<&str> = program.logical_digital_outputs.iter().map(|(key, _)| key.as_str()).collect();
    let windows = trigger_windows_by_channel(program, &channels, table.as_deref(), 1, 1)?;
    let point = point.unwrap_or(&[]);
    let durations = program.resolved_durations(point)?;
    let visits = program.frame_visits(point, None)?;
    let mut buses = Vec::new();
    for (bus_index, name) in program.bus_names.iter().enumerate() {
        let safe = program.bus_safe_values[bus_index];
        let delay = program.bus_delays.iter().find(|item| item.bus_index as usize == bus_index).map_or(0, |item| item.delay_ticks);
        let actions: Vec<_> = program.bus_actions.iter().filter(|action| action.bus_index as usize == bus_index).collect();
        let steps = visits.iter().filter_map(|(row, start)| actions.iter().find(|action| action.row as usize == *row).map(|action| {
            Json::Array(vec![int(*start as i128), int(durations[*row]), Json::str(action.mode.name()), int(program.resolved_bus_value(action, point))])
        })).collect();
        let final_value = actions.iter().max_by_key(|action| action.row).map_or(safe, |action| program.resolved_bus_value(action, point));
        buses.push(Json::object(vec![("name", Json::str(name)), ("safe", int(safe)), ("delay_ticks", int(delay)), ("final", int(final_value)),
                                     ("steps", Json::Array(steps))]));
    }
    let seconds = run_duration_seconds(program, table.as_deref(), 1, 1)?;
    Ok(Json::object(vec![
        ("clock_hz", Json::Float(program.clock_hz)),
        ("duration_ticks", int((seconds * program.clock_hz).round_ties_even() as i128)),
        ("windows", Json::Object(windows.into_iter().map(|(name, spans)| {
            (name, Json::Array(spans.into_iter().map(|(start, end)| Json::Array(vec![int(start as i128), int(end as i128)])).collect()))
        }).collect())),
        ("buses", Json::Array(buses)),
    ]))
}

fn guard<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> { mutex.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) }

struct Playback { finished: Arc<Event>, error: Arc<Mutex<Option<String>>> }

/// The device on its memory twin, played at wall time.
pub struct VirtualBoard {
    streamer: PulseStreamer,
    memory: Arc<MemoryRegisterTransport>,
    world: Mutex<Option<Arc<dyn World>>>,
    playback: Mutex<Option<Playback>>,
    stop: Arc<Event>,
}

impl VirtualBoard {
    /// The deployed board's geometry and wiring (its streamer config and
    /// manifest), on a memory twin; opened.
    pub fn open(config_path: Option<&Path>) -> Result<VirtualBoard> {
        let found = load_streamer_config(config_path)?;
        let Some(source) = found.source.clone() else {
            return Err(Error::new(Kind::Runtime, "no streamer config was found, so the virtual board geometry is unknown"));
        };
        let config = require_streamer_config(&source)?;
        let target = target_from_config(&config, None, None)?;
        let memory = Arc::new(MemoryRegisterTransport::new(&config.params, false, false)?);
        let streamer = PulseStreamer::new(memory.clone() as Arc<dyn RegisterTransport>, config.params, config.clock_hz, target)?;
        streamer.open()?;
        Ok(VirtualBoard { streamer, memory, world: Mutex::new(None), playback: Mutex::new(None), stop: Event::new() })
    }

    pub fn set_world(&self, world: Option<Arc<dyn World>>) { *guard(&self.world) = world; }

    fn playing(&self) -> bool { guard(&self.playback).as_ref().is_some_and(|playback| !playback.finished.is_set()) }

    fn fire(&self, run_repeats: u64, scan_repeats: u64) -> Result<AppliedState> {
        if self.playing() { return Err(Error::new(Kind::Runtime, "the previous virtual pulse is still playing")); }
        let applied = self.streamer.fire(run_repeats, scan_repeats)?;
        self.stop.clear();
        let playback = Playback { finished: Event::new(), error: Arc::new(Mutex::new(None)) };
        let (finished, error, stop, memory) = (playback.finished.clone(), playback.error.clone(), self.stop.clone(), self.memory.clone());
        let world = guard(&self.world).clone();
        let program = applied.program.clone();
        let rows = applied.rows.clone();
        let spawned = std::thread::Builder::new().name("zlc-virtual-world".into()).spawn(move || {
            let outcome = play(&program, &rows, run_repeats, scan_repeats, world.as_deref(), &stop, &memory);
            if let Err((failure, cursor)) = outcome {
                *guard(&error) = Some(failure.message);
                let _ = memory.publish_execution_readback(STATUS_ERROR, cursor);
            }
            finished.set();
        });
        if let Err(failure) = spawned {
            let _ = self.streamer.safe();
            return Err(Error::new(Kind::Runtime, format!("cannot start the virtual world playback: {failure}")));
        }
        *guard(&self.playback) = Some(playback);
        Ok(applied)
    }

    fn wait_done(&self, timeout: Option<Duration>, command_id: Option<u32>) -> Result<Option<DoneReport>> {
        let started = Instant::now();
        let playback = guard(&self.playback).as_ref().map(|p| (p.finished.clone(), p.error.clone()));
        if let Some((finished, error)) = &playback {
            if !finished.wait(timeout) { return Ok(None); }
            if let Some(failure) = guard(error).take() {
                let _ = self.streamer.wait_done(Some(Duration::from_secs(1)), command_id);
                return Err(Error::new(Kind::Runtime, format!("virtual world playback failed: {failure}")));
            }
        }
        let remaining = timeout.map(|timeout| timeout.saturating_sub(started.elapsed()));
        self.streamer.wait_done(remaining, command_id)
    }

    fn safe(&self) -> Result<SafeReadback> {
        let readback = self.streamer.safe();
        self.stop.set();
        let finished = guard(&self.playback).as_ref().map(|p| p.finished.clone());
        let stopped = finished.is_none_or(|finished| finished.wait(Some(Duration::from_secs(2))));
        if let Some(world) = guard(&self.world).clone() { world.safe(); }
        let readback = readback?;
        if !stopped { return Err(Error::new(Kind::Runtime, "virtual world playback did not stop")); }
        Ok(readback)
    }
}

/// Wait out one run's physical duration, or until stopped.
fn play_one(program: &CompiledProgram, point: Option<&[u64]>, world: Option<&dyn World>, stop: &Arc<Event>,
            timelines: &mut Vec<Option<Json>>, row: usize) -> Result<bool> {
    if stop.is_set() { return Ok(false); }
    let started = Instant::now();
    if let Some(world) = world {
        if timelines[row].is_none() { timelines[row] = Some(run_timeline(program, point)?); }
        world.play_run(timelines[row].as_ref().expect("made"), stop)?;
    }
    let table: Option<Vec<Vec<u64>>> = point.map(|point| vec![point.to_vec()]);
    let duration = Duration::from_secs_f64(run_duration_seconds(program, table.as_deref(), 1, 1)?.max(0.0));
    Ok(!stop.wait(Some(duration.saturating_sub(started.elapsed()))))
}

/// The RTL's walk: each row for `run_repeats` runs (0: the first row until
/// Stop), the table `scan_repeats` times (0: until Stop); CURSOR counts row
/// visits, DONE after the last.
fn play(program: &CompiledProgram, rows: &[Vec<u64>], run_repeats: u64, scan_repeats: u64, world: Option<&dyn World>, stop: &Arc<Event>,
        memory: &MemoryRegisterTransport) -> std::result::Result<(), (Error, u32)> {
    let points: Vec<Option<&[u64]>> = if rows.is_empty() { vec![None] } else { rows.iter().map(|row| Some(row.as_slice())).collect() };
    let mut timelines = vec![None; points.len()];
    let mut cursor: u32 = 0;
    let fail = |error: Error, cursor: u32| (error, cursor);
    if run_repeats == 0 {
        while play_one(program, points[0], world, stop, &mut timelines, 0).map_err(|e| fail(e, cursor))? {}
        return Ok(());
    }
    let mut sweep = 0;
    while scan_repeats == 0 || sweep < scan_repeats {
        for (row, point) in points.iter().enumerate() {
            for _ in 0..run_repeats {
                if !play_one(program, *point, world, stop, &mut timelines, row).map_err(|e| fail(e, cursor))? { return Ok(()); }
            }
            if stop.is_set() { return Ok(()); }
            if scan_repeats != 0 && sweep + 1 == scan_repeats && row + 1 == points.len() {
                let _ = memory.publish_execution_readback(STATUS_DONE, cursor);
                return Ok(());
            }
            cursor = cursor.wrapping_add(1);
            if !memory.publish_execution_readback(STATUS_RUNNING, cursor).map_err(|e| fail(e, cursor))? { return Ok(()); }
        }
        sweep += 1;
    }
    Ok(())
}

enum Link { Virtual(VirtualBoard), Remote(RemotePulseStreamer) }

/// What became of the board's last FIRE. Its report is taken exactly once,
/// by whoever waits for it; until then the next FIRE is refused (the device's
/// rule), and a FIRE that can no longer report is said so, never waited on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fate {
    /// Fired; its report has not been taken.
    Pending,
    /// Its report was taken.
    Reported,
    /// The board was made safe (or stopped playing it) before it reported.
    Safed,
}

/// What the board is playing, as whoever aligns to its triggers needs it: which LOAD or FIRE
/// (`revision` counts every change), and when that FIRE was sent and answered (this computer's
/// clock, microseconds since the Unix epoch).
#[derive(Clone, Copy, Debug)]
struct Play { revision: u64, state: &'static str, command_id: Option<u32>, run_repeats: u64, scan_repeats: u64, sent_us: u64, answered_us: u64 }

/// One scan point as a trigger lane plays it: when the lane rises after the point begins, and
/// how long the point lasts (seconds, exact multiples of the board's tick).
pub struct TriggerPoint { pub rising: Vec<f64>, pub duration: f64 }

/// The rising edges of `lane` (a raw lane, `ch11`, or a port key) at every point of `rows`, in
/// the order the table is played. An edge a lane delay carries past its point's end stays with
/// the point that made it.
pub fn trigger_points(program: &CompiledProgram, rows: &[Vec<u64>], lane: &str) -> Result<Vec<TriggerPoint>> {
    let table = (!rows.is_empty()).then_some(rows);
    let edges = trigger_edge_ticks(program, &[lane], table, 1, 1, None)?.remove(0).1;
    let mut rising = edges.into_iter().step_by(2).peekable();
    let single = [Vec::new()];
    let played: &[Vec<u64>] = if rows.is_empty() { &single } else { rows };
    let (mut points, mut start) = (Vec::with_capacity(played.len()), 0u128);
    for (index, row) in played.iter().enumerate() {
        let ticks = program.frame_ticks(row)?;
        let (end, last) = (start + ticks, index + 1 == played.len());
        let mut offsets = Vec::new();
        while let Some(tick) = rising.next_if(|tick| last || *tick < end) { offsets.push((tick - start) as f64 / program.clock_hz); }
        points.push(TriggerPoint { rising: offsets, duration: ticks as f64 / program.clock_hz });
        start = end;
    }
    Ok(points)
}

/// One attached board, as the application holds it.
pub struct Board { link: Link, description: BoardDescription, name: String, last: Mutex<Option<(u32, Fate)>>, play: Mutex<Play>,
                   /// What the board confirmed it holds at its last LOAD or FIRE, and the schedules worked out from it
                   /// (revision, lane, points): asked for at every frame's pace, they are neither fetched nor walked twice.
                   held: Mutex<Option<AppliedState>>, schedules: Mutex<Vec<(u64, String, Json)>> }

const NOTHING: Play = Play { revision: 0, state: "none", command_id: None, run_repeats: 0, scan_repeats: 0, sent_us: 0, answered_us: 0 };

/// How long a remote server may take to answer a connect.
pub const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

impl Board {
    pub fn virtual_board(config_path: Option<&Path>) -> Result<Board> {
        let board = VirtualBoard::open(config_path)?;
        let description = board.streamer.describe()?;
        Ok(Board { link: Link::Virtual(board), description, name: "virtual".into(), last: Mutex::new(None), play: Mutex::new(NOTHING), held: Mutex::default(), schedules: Mutex::default() })
    }

    /// A pulse server at `host:port` (v2's `pulse_server` or `zlc-pulse-server`).
    pub fn remote(host: &str, port: u16) -> Result<Board> {
        let client = RemotePulseStreamer::new(host, port, None, CONNECT_TIMEOUT)?;
        client.open()?;
        let description = match client.describe() { Ok(description) => description, Err(error) => { let _ = client.close(); return Err(error); } };
        Ok(Board { link: Link::Remote(client), description, name: format!("remote {host}:{port}"), last: Mutex::new(None), play: Mutex::new(NOTHING), held: Mutex::default(), schedules: Mutex::default() })
    }

    /// Take a remote board back after another client held it (the server SAFEs
    /// the one it drops first); a virtual board is always this application's.
    pub fn rejoin(&self) -> Result<()> {
        match &self.link { Link::Virtual(_) => Ok(()), Link::Remote(client) => { client.disconnect(); client.open() } }
    }

    /// What was dialled: "virtual" or "remote host:port".
    pub fn name(&self) -> &str { &self.name }
    pub fn description(&self) -> &BoardDescription { &self.description }
    pub fn is_virtual(&self) -> bool { matches!(self.link, Link::Virtual(_)) }

    /// The world a virtual board plays into (ignored by a remote one).
    pub fn set_world(&self, world: Option<Arc<dyn World>>) { if let Link::Virtual(board) = &self.link { board.set_world(world); } }

    pub fn config(&self) -> &ConfigHolder {
        match &self.link { Link::Virtual(board) => board.streamer.config(), Link::Remote(client) => client.config() }
    }

    /// LOAD a program (and its table); refused while the last FIRE has not
    /// reported, since nothing new reaches a board that is still playing.
    pub fn load(&self, program: Arc<CompiledProgram>, source: Option<Arc<Sequence>>, rows: Vec<Vec<u64>>) -> Result<()> {
        self.free()?;
        let (began, row_count) = (Instant::now(), rows.len());
        let loaded = match &self.link { Link::Virtual(board) => board.streamer.load(program, source, rows), Link::Remote(client) => client.load(program, source, rows) };
        self.say("load", json!({"rows": row_count, "ms": began.elapsed().as_secs_f64() * 1e3, "error": loaded.as_ref().err().map(|error| error.message.clone())}));
        // A LOAD makes the board safe first: whatever played has stopped, whether or not this one took.
        *guard(&self.held) = if loaded.is_ok() { self.applied().ok().flatten() } else { None };
        self.playing(if loaded.is_ok() { "loaded" } else { "stopped" }, None, (0, 0), journal::now_us());
        loaded
    }

    /// The board now holds or plays something else: every reader of its schedule sees a new revision.
    fn playing(&self, state: &'static str, command_id: Option<u32>, repeats: (u64, u64), sent_us: u64) {
        let mut play = guard(&self.play);
        *play = Play { revision: play.revision + 1, state, command_id, run_repeats: repeats.0, scan_repeats: repeats.1, sent_us, answered_us: journal::now_us() };
    }

    /// Counts every LOAD, FIRE, stop and report: what a reader of the schedule compares.
    pub fn revision(&self) -> u64 { guard(&self.play).revision }

    /// The digital lanes a camera can be triggered from: (raw lane, port key, label), in the target's order.
    pub fn trigger_lanes(&self) -> Vec<(String, String, String)> {
        self.description.target.ports.iter().filter(|port| port.kind == crate::model::PortKind::Digital && port.lanes.len() == 1)
            .map(|port| (port.lanes[0].clone(), port.key.clone(), port.label.clone())).collect()
    }

    /// How often each trigger lane rises in one shot of what the board holds (its table's first
    /// point); None while it holds nothing. With the revision that count is of.
    pub fn trigger_counts(&self) -> Result<(u64, Option<Vec<usize>>)> {
        let play = *guard(&self.play);
        let Some(applied) = guard(&self.held).clone() else { return Ok((play.revision, None)) };
        let lanes = self.trigger_lanes();
        let names: Vec<&str> = lanes.iter().map(|(lane, ..)| lane.as_str()).collect();
        let first: Vec<Vec<u64>> = applied.rows.first().cloned().into_iter().collect();
        let edges = trigger_edge_ticks(&applied.program, &names, (!first.is_empty()).then_some(first.as_slice()), 1, 1, None)?;
        Ok((play.revision, Some(edges.into_iter().map(|(_, ticks)| ticks.len().div_ceil(2)).collect())))
    }

    /// `pulse_schedule`: what the board holds or plays, for whoever is triggered by `lane`:
    /// `{revision, state: none|loaded|playing|stopped|done, command_id, run_repeats, scan_repeats,
    /// sent_us, answered_us, lane, points: [{rising: [s], duration: s}]}`. A shot is one play of one
    /// point; each point is played `run_repeats` times before the next, the table `scan_repeats`
    /// times (0: without end). `known`: the revision the asker already has, answered `unchanged`.
    pub fn trigger_schedule(&self, lane: &str, known: Option<u64>) -> Result<Json> {
        let play = *guard(&self.play);
        let mut facts = vec![("revision", int(play.revision)), ("state", Json::str(play.state)),
            ("command_id", play.command_id.map_or(Json::Null, int)), ("run_repeats", int(play.run_repeats)), ("scan_repeats", int(play.scan_repeats)),
            ("sent_us", int(play.sent_us)), ("answered_us", int(play.answered_us)), ("lane", Json::str(lane))];
        if known == Some(play.revision) { facts.push(("unchanged", Json::Bool(true))); return Ok(Json::object(facts)); }
        let known = guard(&self.schedules).iter().find(|(revision, of, _)| *revision == play.revision && of == lane).map(|(.., points)| points.clone());
        let points = match known {
            Some(points) => points,
            None => {
                let points = match guard(&self.held).clone() {
                    Some(applied) => trigger_points(&applied.program, &applied.rows, lane)?,
                    None => Vec::new(),
                };
                let points = Json::Array(points.into_iter().map(|point| Json::object(vec![
                    ("rising", Json::Array(point.rising.into_iter().map(Json::Float).collect())), ("duration", Json::Float(point.duration))])).collect());
                let mut schedules = guard(&self.schedules);
                schedules.retain(|(revision, ..)| *revision == play.revision);
                schedules.push((play.revision, lane.to_owned(), points.clone()));
                points
            }
        };
        facts.push(("points", points));
        Ok(Json::object(facts))
    }

    /// One line of the journal for this board: what was asked of it and what it answered.
    fn say(&self, event: &str, mut facts: Value) {
        facts["board"] = json!(self.name);
        journal::note("board", event, facts);
    }

    /// FIRE what is loaded; the confirmed application and its command id.
    /// Refused while the last FIRE's report has not been taken.
    pub fn fire(&self, run_repeats: u64, scan_repeats: u64) -> Result<(AppliedState, u32)> {
        self.free()?;
        let began = Instant::now();
        let fired = match &self.link {
            Link::Virtual(board) => board.fire(run_repeats, scan_repeats).map(|applied| (applied, board.streamer.fire_command_id())),
            Link::Remote(client) => client.fire(run_repeats, scan_repeats).map(|applied| (applied, client.fire_command_id().unwrap_or(0))),
        };
        // Said once the board has answered: `ms` earlier is when the FIRE was sent.
        self.say("fire", json!({"run_repeats": run_repeats, "scan_repeats": scan_repeats, "ms": began.elapsed().as_secs_f64() * 1e3,
            "command_id": fired.as_ref().ok().map(|(_, id)| *id), "error": fired.as_ref().err().map(|error| error.message.clone())}));
        let sent_us = journal::now_us().saturating_sub(began.elapsed().as_micros() as u64);
        let (applied, command_id) = fired?;
        *guard(&self.held) = Some(applied.clone());
        self.playing("playing", Some(command_id), (run_repeats, scan_repeats), sent_us);
        *guard(&self.last) = Some((command_id, Fate::Pending));
        Ok((applied, command_id))
    }

    /// Nothing unreported holds the board: the last FIRE reported or was made
    /// safe -- here, or elsewhere (a board that no longer holds it).
    fn free(&self) -> Result<()> {
        let Some((command_id, Fate::Pending)) = self.last_fire() else { return Ok(()) };
        if self.holds(command_id)? {
            return Err(Error::new(Kind::Runtime, format!("FIRE {command_id} has not reported yet, so nothing new is sent to the board; \
                                                          waiting for it (pulse_wait_done) takes its report, Stop makes the board safe")));
        }
        let mut last = guard(&self.last);
        if *last == Some((command_id, Fate::Pending)) { *last = Some((command_id, Fate::Safed)); }
        Ok(())
    }

    /// The last FIRE and what became of it.
    pub fn last_fire(&self) -> Option<(u32, Fate)> { *guard(&self.last) }

    /// Wait for the current FIRE's report (`command_id`: that FIRE only);
    /// `None` when it has not finished in time. Taking a report frees the board.
    pub fn wait_done(&self, timeout: Option<Duration>, command_id: Option<u32>) -> Result<Option<DoneReport>> {
        let report = match &self.link { Link::Virtual(board) => board.wait_done(timeout, command_id), Link::Remote(client) => client.wait_done(timeout, command_id) }?;
        if let Some(report) = &report {
            // `elapsed_seconds` after its FIRE is when the board itself was done; this line is when that was read.
            self.say("report", json!({"command_id": report.command_id, "status": report.status, "elapsed_seconds": report.elapsed_seconds,
                "cursor": report.cursor, "underflow": report.underflow, "observer_error": report.observer_error}));
            let mut last = guard(&self.last);
            if last.is_some_and(|(id, _)| id == report.command_id) { *last = Some((report.command_id, Fate::Reported)); }
            drop(last);
            // The FIRE that reported has played all it had: nothing more comes of this schedule.
            let play = *guard(&self.play);
            if play.state == "playing" && play.command_id == Some(report.command_id) {
                self.playing("done", play.command_id, (play.run_repeats, play.scan_repeats), play.sent_us);
            }
        }
        Ok(report)
    }

    /// Is the board still playing (or holding the unreported end of) this FIRE?
    fn holds(&self, command_id: u32) -> Result<bool> {
        match &self.link {
            Link::Virtual(board) => Ok(board.streamer.is_firing() && board.streamer.fire_command_id() == command_id),
            Link::Remote(client) => Ok(client.snapshot()?.get("firing").and_then(Json::as_bool) == Some(true)
                                       && client.fire_command_id() == Some(command_id)),
        }
    }

    /// `pulse_wait_done`: FIRE `command_id`'s report, taken once; `None` while
    /// it plays. A FIRE that can no longer report -- its report already taken,
    /// the board made safe, another FIRE sent since, or never fired here -- is
    /// refused rather than waited on.
    pub fn wait_report(&self, command_id: u32, timeout: Option<Duration>) -> Result<Option<DoneReport>> {
        let refuse = |why: String| Err(Error::new(Kind::Runtime, format!("FIRE {command_id} can no longer report: {why}")));
        match self.last_fire() {
            None => return refuse("this board has not fired it".into()),
            Some((id, _)) if id != command_id => return refuse(format!("FIRE {id} was sent since")),
            Some((_, Fate::Reported)) => return refuse("its report was already taken".into()),
            Some((_, Fate::Safed)) => return refuse("the board was made safe before it finished".into()),
            Some((_, Fate::Pending)) => {}
        }
        if let Some(report) = self.wait_done(timeout, Some(command_id))? { return Ok(Some(report)); }
        match self.last_fire() {
            Some((id, Fate::Pending)) if id == command_id => {
                if self.holds(command_id)? { return Ok(None); }
                let mut last = guard(&self.last);
                if *last == Some((command_id, Fate::Pending)) { *last = Some((command_id, Fate::Safed)); }
                refuse("the board is no longer playing it".into())
            }
            Some((id, _)) if id != command_id => refuse(format!("FIRE {id} was sent since")),
            Some((_, Fate::Reported)) => refuse("its report was already taken".into()),
            _ => refuse("the board was made safe before it finished".into()),
        }
    }

    pub fn cursor(&self) -> Result<Option<u32>> {
        match &self.link { Link::Virtual(board) => board.streamer.cursor(), Link::Remote(client) => client.cursor() }
    }

    /// SAFE now; an unreported FIRE will not report.
    pub fn safe(&self) -> Result<SafeReadback> {
        let began = Instant::now();
        let readback = match &self.link { Link::Virtual(board) => board.safe(), Link::Remote(client) => client.safe() };
        self.say("safe", json!({"ms": began.elapsed().as_secs_f64() * 1e3, "error": readback.as_ref().err().map(|error| error.message.clone())}));
        let play = *guard(&self.play);
        if matches!(play.state, "playing" | "loaded") { self.playing("stopped", play.command_id, (play.run_repeats, play.scan_repeats), play.sent_us); }
        let mut last = guard(&self.last);
        if let Some((command_id, Fate::Pending)) = *last { *last = Some((command_id, Fate::Safed)); }
        readback
    }

    /// What the board says it holds and does (`snapshot()`).
    pub fn snapshot(&self) -> Result<Json> {
        match &self.link { Link::Virtual(board) => Ok(board.streamer.snapshot().to_json()), Link::Remote(client) => client.snapshot() }
    }

    pub fn applied(&self) -> Result<Option<AppliedState>> {
        match &self.link { Link::Virtual(board) => Ok(board.streamer.applied()), Link::Remote(client) => client.applied() }
    }

    /// SAFE, then let the connection go.
    pub fn close(&self) -> Result<()> {
        let closed = match &self.link {
            Link::Virtual(board) => { let _ = board.safe(); board.streamer.close() }
            Link::Remote(client) => client.close(),
        };
        let mut last = guard(&self.last);
        if let Some((command_id, Fate::Pending)) = *last { *last = Some((command_id, Fate::Safed)); }
        closed
    }
}

// ------------------------------------------------------------ for a run's caller
/// A board's proven facts as an archive keeps them (v2 `_description_snapshot`).
pub fn description_snapshot(description: &BoardDescription) -> Json {
    Json::object(vec![
        ("clock_hz", Json::Float(description.clock_hz)), ("time_step_ns", Json::Float(description.time_step_ns())),
        ("layout_fingerprint", int(description.layout_fingerprint())),
        ("target_abi_fingerprint", Json::str(description.target.abi_fingerprint())),
        ("geometry", Json::Object(crate::wire::PARAM_FIELDS.iter()
            .map(|name| ((*name).to_owned(), int(description.geometry.get(name).unwrap_or(0)))).collect())),
        ("target", crate::codec::target_to_tree(&description.target)),
    ])
}

/// The board facts, state and played program an archive keeps of one run
/// (v2 `sequencer_archive_snapshot`).
pub fn archive_snapshot(board: &Board, applied: &AppliedState) -> Result<Json> {
    let description = board.description();
    let config = board.config().values();
    let state = board.snapshot()?;
    let keep = ["opened", "config_source", "loaded", "firing", "run_repeats", "scan_repeats", "cursor", "scan_count", "scan_next_chunk",
                "underflow", "status", "applied_digest"];
    let selected = state.as_object().map(|items| items.iter().filter(|(key, _)| keep.contains(&key.as_str())).cloned().collect()).unwrap_or_default();
    let program = &applied.program;
    let mut items = vec![
        ("description", description_snapshot(description)),
        ("config", Json::Object(config.iter().map(|(key, (value, unit))| (key.clone(), Json::Array(vec![Json::Float(*value), Json::str(unit)]))).collect())),
        ("state", Json::Object(selected)),
        ("program", Json::object(vec![
            ("digest", Json::str(program.digest())), ("clock_hz", Json::Float(program.clock_hz)),
            ("duration_seconds", Json::Float(program.duration_seconds)),
            ("loops", Json::Array(program.loops.iter().map(|lp| Json::Array(vec![int(lp.0 as i128), int(lp.1 as i128), int(lp.2 as i128)])).collect())),
            ("rows", Json::Array(applied.rows.iter().map(|row| Json::Array(row.iter().map(|value| int(*value)).collect())).collect())),
            ("run_repeats", int(applied.run_repeats)), ("scan_repeats", int(applied.scan_repeats)),
        ])),
    ];
    if let Some(source) = &applied.source { items.push(("pulse", sequence_to_tree(source))); }
    Ok(Json::object(items))
}
