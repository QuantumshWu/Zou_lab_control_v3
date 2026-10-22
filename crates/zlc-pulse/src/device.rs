//! The physical pulse-streamer session: open (layout handshake), load (one
//! complete application), fire, the observer that alone reads STATUS/CURSOR and
//! refills scan banks while a FIRE plays, completion reports and SAFE.
//!
//! A Stop is sticky: a FIRE still being prepared under the device lock (Config
//! reread, recompile, reload) when another thread asks for SAFE is refused
//! instead of being sent after it.
use std::collections::{BTreeSet, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::{JoinHandle, ThreadId};
use std::time::{Duration, Instant, SystemTime};

use crate::binding::{apply_config_values, resolve_subpulse_sources};
use crate::codec::{ConfigValues, config_value_entry, read_config_values};
use crate::compile::{CompiledProgram, compile_sequence};
use crate::error::{Error, Kind, Result, runtime_error, value_error};
use crate::json::Json;
use crate::model::{FieldKind, MAXIMUM_REPEAT_COUNT, PortKind, Sequence, Target};
use crate::schedule::validate_delay_capacity;
use crate::transport::{DEFAULT_OBSERVER_INTERVAL, Event, RegisterTransport};
use crate::wire::{CMD_FIRE, CMD_LOAD, CMD_SAFE, STATUS_DONE, STATUS_ERROR, STATUS_LINK_ERROR, STATUS_LOADED, STATUS_RUNNING,
                  STATUS_UNDERFLOW, StreamerParams, build_fingerprint, ctrl, pack_program, pack_scan_rows};

pub const LOAD_TIMEOUT: Duration = Duration::from_secs(5);
pub const SAFE_TIMEOUT: Duration = Duration::from_secs(5);
pub const OBSERVER_STOP_TIMEOUT: Duration = Duration::from_secs(2);

fn repeat_count(value: u64, name: &str) -> Result<u64> {
    if value > MAXIMUM_REPEAT_COUNT { return value_error(format!("{name} must be in the hardware range [0, 2^32-1]")); }
    Ok(value)
}

/// One FIRE's completion as the observer saw it.
#[derive(Clone, Debug, PartialEq)]
pub struct DoneReport {
    pub status: u32,
    /// Cumulative row-visit ordinal; table row = cursor % rows.
    pub cursor: Option<u32>,
    pub underflow: bool,
    /// FIRE command start to observer-confirmed terminal status.
    pub elapsed_seconds: f64,
    pub command_id: u32,
    pub observer_error: String,
    /// Status polls that failed during this shot, and frames the line resent.
    pub poll_failures: u64,
    pub resent_frames: u64,
    pub command_seconds: f64,
    pub report_delay_seconds: f64,
}

impl DoneReport {
    /// Why this shot was not a good shot, or "" if it was.
    pub fn fault(&self) -> String {
        let mut reasons = Vec::new();
        if !self.observer_error.is_empty() {
            reasons.push(format!("pulse observer failed: {}", self.observer_error));
            if self.poll_failures > 0 || self.resent_frames > 0 {
                reasons.push(format!("the line failed {} poll(s) and resent {} frame(s) during this shot", self.poll_failures, self.resent_frames));
            }
        }
        if self.status & STATUS_ERROR != 0 { reasons.push("the board reported an error".into()); }
        if self.status & STATUS_UNDERFLOW != 0 || self.underflow { reasons.push("the scan bank underran".into()); }
        reasons.join("; ")
    }

    pub fn link_error(&self) -> bool { self.status & STATUS_LINK_ERROR != 0 }
}

/// A completed command or terminal run proving that pins are safe.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SafeReadback { pub status: u32, pub command_id: u32 }

impl SafeReadback {
    pub fn stable(&self) -> bool {
        self.status == 0 || self.status == STATUS_LOADED
            || (self.status & STATUS_DONE != 0 && self.status & (STATUS_RUNNING | STATUS_ERROR | STATUS_UNDERFLOW) == 0)
    }
}

/// Immutable echo of the executable application the device holds.
#[derive(Clone, Debug)]
pub struct AppliedState {
    pub program: Arc<CompiledProgram>,
    pub source: Option<Arc<Sequence>>,
    pub rows: Arc<Vec<Vec<u64>>>,
    pub run_repeats: u64,
    pub scan_repeats: u64,
    /// Unix seconds of the LOAD.
    pub loaded_at: f64,
    pub authored_source: Option<Arc<Sequence>>,
}

impl AppliedState {
    pub fn new(program: Arc<CompiledProgram>, source: Option<Arc<Sequence>>, authored_source: Option<Arc<Sequence>>, rows: Arc<Vec<Vec<u64>>>,
               run_repeats: u64, scan_repeats: u64, loaded_at: f64) -> Result<AppliedState> {
        if rows.iter().any(|row| row.len() != program.slot_count()) { return value_error("applied row width differs from the compiled program"); }
        if program.slot_count() > 0 && rows.is_empty() { return value_error("a slotted program requires applied rows"); }
        if program.slot_count() == 0 && !rows.is_empty() { return value_error("an unslotted program has no value rows"); }
        repeat_count(run_repeats, "run_repeats")?;
        repeat_count(scan_repeats, "scan_repeats")?;
        if rows.is_empty() && scan_repeats != 1 { return value_error("scan_repeats must be 1 when no scan table is loaded"); }
        if !loaded_at.is_finite() || loaded_at < 0.0 { return value_error("applied loaded_at must be a finite non-negative timestamp"); }
        Ok(AppliedState { program, source, rows, run_repeats, scan_repeats, loaded_at, authored_source })
    }

    pub fn with_repeats(&self, run_repeats: u64, scan_repeats: u64) -> Result<AppliedState> {
        repeat_count(run_repeats, "run_repeats")?;
        repeat_count(scan_repeats, "scan_repeats")?;
        if self.rows.is_empty() && scan_repeats != 1 { return value_error("scan_repeats must be 1 when no scan table is loaded"); }
        Ok(AppliedState { run_repeats, scan_repeats, ..self.clone() })
    }
}

/// What a board is, in its own words, proven by the open handshake.
#[derive(Clone, Debug)]
pub struct BoardDescription { pub target: Arc<Target>, pub geometry: StreamerParams, pub clock_hz: f64 }

impl BoardDescription {
    pub fn time_step_ns(&self) -> f64 { 1e9 / self.clock_hz }
    pub fn layout_fingerprint(&self) -> u32 { build_fingerprint(&self.geometry).unwrap_or(0) }
}

/// The calibrated Config set a board compiles against: a fact about the
/// apparatus, kept across connections. `source` is a label; only an explicitly
/// bound file is reread.
#[derive(Default)]
pub struct ConfigHolder { state: Mutex<(ConfigValues, String, Option<PathBuf>)> }

impl ConfigHolder {
    fn guard(&self) -> MutexGuard<'_, (ConfigValues, String, Option<PathBuf>)> { self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) }

    /// Hold in-memory values and stop following any bound file.
    pub fn load_values(&self, entries: &[(String, (f64, String))], source: &str) -> Result<()> {
        let mut held = Vec::new();
        for (name, (value, unit)) in entries { held.push((name.clone(), config_value_entry(name, &Json::Float(*value), &Json::str(unit))?)); }
        *self.guard() = (held, source.to_owned(), None);
        Ok(())
    }

    /// Bind and read a file, or clear the values when the path is empty.
    pub fn load_file(&self, path: Option<&Path>) -> Result<()> {
        let Some(path) = path.filter(|path| !path.as_os_str().is_empty()) else { return self.load_values(&[], ""); };
        let path = crate::codec::absolute(path);
        let entries = read_config_values(&path)?;
        *self.guard() = (entries, path.display().to_string(), Some(path));
        Ok(())
    }

    fn refresh(&self) -> Result<()> {
        let mut state = self.guard();
        if let Some(path) = state.2.clone() { state.0 = read_config_values(&path)?; }
        Ok(())
    }

    pub fn values(&self) -> ConfigValues { self.guard().0.clone() }
    pub fn source(&self) -> String { self.guard().1.clone() }

    /// The next execution's saved values, without changing what is held.
    pub fn preview(&self) -> Result<ConfigValues> {
        let state = self.guard();
        match &state.2 { Some(path) => read_config_values(path), None => Ok(state.0.clone()) }
    }

    /// What a FIRE plays from an authored pulse: its Subpulse files and the held
    /// Config values resolved, recompiled only when that changes the program.
    /// `compiled` is what the held program was compiled from.
    pub fn prepare(&self, program: &Arc<CompiledProgram>, authored: Option<&Arc<Sequence>>, compiled: Option<&Arc<Sequence>>,
                   geometry: &StreamerParams) -> Result<(Arc<CompiledProgram>, Option<Arc<Sequence>>)> {
        let Some(authored) = authored else { return Ok((program.clone(), compiled.cloned())) };
        if authored.config_enabled { self.refresh()?; }
        let resolved = resolve_subpulse_sources(authored, None)?;
        if let Some(compiled) = compiled.filter(|compiled| !Arc::ptr_eq(compiled, authored)) {
            let fields = |sequence: &Sequence| sequence.scan_bindings().iter().map(|b| (b.field_id(), b.unit.clone())).collect::<Vec<_>>();
            if fields(compiled) != fields(&resolved) {
                return value_error("Subpulse scan fields changed; prepare and load a new scan table before Fire");
            }
        }
        let (filled, _, _) = apply_config_values(&resolved, &self.values(), None)?;
        if compiled.is_some_and(|compiled| **compiled == filled) { return Ok((program.clone(), compiled.cloned())); }
        let program = compile_sequence(&filled, geometry, program.clock_hz)?;
        Ok((Arc::new(program), Some(Arc::new(filled))))
    }
}

struct Inner {
    opened: bool,
    loaded: bool,
    program: Option<Arc<CompiledProgram>>,
    applied: Option<AppliedState>,
    applied_digest: String,
    validated_execution: Option<(u64, u64)>,
    firing: bool,
    run_repeats: u64,
    scan_repeats: u64,
    scan_rows: Arc<Vec<Vec<u64>>>,
    scan_next_chunk: u64,
    scan_armed: bool,
    scan_count: u64,
    scan_last_cursor: u64,
    scan_cursor_total: u64,
    cursor_value: Option<u32>,
    underflow: bool,
    observer_error: String,
    poll_failures: u64,
    resends_at_fire: u64,
    done: Arc<Event>,
    terminal_status: u32,
    fire_started: Instant,
    fire_acknowledged: Instant,
    fire_finished: Instant,
    safe_readback: Option<SafeReadback>,
    command_id: Option<u32>,
    fire_command_id: u32,
}

struct Shared {
    transport: Arc<dyn RegisterTransport>,
    geom: StreamerParams,
    clock_hz: f64,
    target: Arc<Target>,
    observer_interval: Duration,
    inner: Mutex<Inner>,
    stop: Arc<Event>,
    stopping: Mutex<HashSet<ThreadId>>,
    worker: Mutex<Option<JoinHandle<()>>>,
    config: ConfigHolder,
}

/// Host control of the frozen streamer. Cheap to clone; clones share one board.
#[derive(Clone)]
pub struct PulseStreamer(Arc<Shared>);

/// A passive snapshot of what the device holds and does.
#[derive(Clone, Debug, PartialEq)]
pub struct Snapshot {
    pub opened: bool, pub loaded: bool, pub firing: bool, pub run_repeats: u64, pub scan_repeats: u64, pub cursor: Option<u32>,
    pub scan_count: u64, pub scan_next_chunk: u64, pub underflow: bool, pub status: u32, pub applied_digest: String, pub config_source: String,
}

impl Snapshot {
    pub fn to_json(&self) -> Json {
        Json::object(vec![
            ("opened", Json::Bool(self.opened)), ("loaded", Json::Bool(self.loaded)), ("firing", Json::Bool(self.firing)),
            ("run_repeats", Json::Int(self.run_repeats.into())), ("scan_repeats", Json::Int(self.scan_repeats.into())),
            ("cursor", self.cursor.map_or(Json::Null, |cursor| Json::Int(cursor.into()))), ("scan_count", Json::Int(self.scan_count.into())),
            ("scan_next_chunk", Json::Int(self.scan_next_chunk.into())), ("underflow", Json::Bool(self.underflow)),
            ("status", Json::Int(self.status.into())), ("applied_digest", Json::str(&self.applied_digest)),
            ("config_source", Json::str(&self.config_source)),
        ])
    }
}

impl PulseStreamer {
    pub fn new(transport: Arc<dyn RegisterTransport>, geom: StreamerParams, clock_hz: f64, target: Arc<Target>) -> Result<PulseStreamer> {
        if !clock_hz.is_finite() || clock_hz <= 0.0 { return value_error("clock_hz must be positive and finite"); }
        let buses: Vec<_> = target.ports.iter().filter(|port| port.kind == PortKind::Dac).collect();
        let widths: BTreeSet<u64> = buses.iter().map(|port| port.width).collect();
        let mut mismatches = Vec::new();
        if target.raw_lanes.len() as u64 != geom.channel_count {
            mismatches.push(format!("target lanes={} but geometry has {}", target.raw_lanes.len(), geom.channel_count));
        }
        if buses.len() as u64 != geom.bus_count { mismatches.push(format!("target DAC buses={} but geometry has {}", buses.len(), geom.bus_count)); }
        if widths != BTreeSet::from([geom.bus_width]) {
            mismatches.push(format!("target DAC widths={:?} but geometry has {}", widths.iter().collect::<Vec<_>>(), geom.bus_width));
        }
        if !mismatches.is_empty() { return value_error(format!("target/geometry mismatch: {}", mismatches.join("; "))); }
        let now = Instant::now();
        let observer_interval = transport.observer_interval();
        Ok(PulseStreamer(Arc::new(Shared {
            transport, geom, clock_hz, target, observer_interval, stop: Event::new(), stopping: Mutex::new(HashSet::new()),
            worker: Mutex::new(None), config: ConfigHolder::default(),
            inner: Mutex::new(Inner {
                opened: false, loaded: false, program: None, applied: None, applied_digest: String::new(), validated_execution: None,
                firing: false, run_repeats: 1, scan_repeats: 1, scan_rows: Arc::new(Vec::new()), scan_next_chunk: 2, scan_armed: false,
                scan_count: 0, scan_last_cursor: 0, scan_cursor_total: 0, cursor_value: None, underflow: false, observer_error: String::new(),
                poll_failures: 0, resends_at_fire: 0, done: Event::new(), terminal_status: 0, fire_started: now, fire_acknowledged: now,
                fire_finished: now, safe_readback: None, command_id: None, fire_command_id: 0,
            }),
        })))
    }

    fn inner(&self) -> MutexGuard<'_, Inner> { self.0.inner.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) }
    pub fn transport(&self) -> &Arc<dyn RegisterTransport> { &self.0.transport }
    pub fn config(&self) -> &ConfigHolder { &self.0.config }
    pub fn geometry(&self) -> StreamerParams { self.0.geom }
    pub fn clock_hz(&self) -> f64 { self.0.clock_hz }

    pub fn open(&self) -> Result<()> {
        let mut inner = self.inner();
        if inner.opened { return Ok(()); }
        self.0.transport.start()?;
        let layout = self.0.transport.read_word(ctrl::LAYOUT_ID, None, None);
        let expected = build_fingerprint(&self.0.geom)?;
        let checked = layout.and_then(|layout| if layout == expected { Ok(()) } else {
            runtime_error(format!("geometry/layout mismatch: device=0x{layout:08X}, host=0x{expected:08X}"))
        });
        if let Err(error) = checked { let _ = self.0.transport.close(); return Err(error); }
        inner.opened = true;
        inner.command_id = None;
        Ok(())
    }

    pub fn close(&self) -> Result<()> {
        if !self.inner().opened { return Ok(()); }
        self.stop_worker()?;
        let mut inner = self.inner();
        self.safe_locked(&mut inner)?;
        self.0.transport.close()?;
        inner.opened = false;
        inner.loaded = false;
        inner.program = None;
        inner.applied = None;
        inner.applied_digest.clear();
        inner.safe_readback = None;
        Ok(())
    }

    pub fn describe(&self) -> Result<BoardDescription> {
        if !self.inner().opened { return runtime_error("PulseStreamer is not open"); }
        Ok(BoardDescription { target: self.0.target.clone(), geometry: self.0.geom, clock_hz: self.0.clock_hz })
    }

    /// Load one complete application; Config and Subpulse sources resolved first.
    pub fn load(&self, program: Arc<CompiledProgram>, source: Option<Arc<Sequence>>, rows: Vec<Vec<u64>>) -> Result<()> {
        let mut inner = self.inner();
        let (program, filled) = self.0.config.prepare(&program, source.as_ref(), source.as_ref(), &self.0.geom)?;
        self.load_program(&mut inner, program, filled, source, Arc::new(rows))
    }

    /// Load exactly this program (already resolved by its caller).
    pub fn load_program_exact(&self, program: Arc<CompiledProgram>, source: Option<Arc<Sequence>>, authored: Option<Arc<Sequence>>,
                              rows: Arc<Vec<Vec<u64>>>) -> Result<()> {
        let mut inner = self.inner();
        self.load_program(&mut inner, program, source, authored, rows)
    }

    fn load_program(&self, inner: &mut Inner, program: Arc<CompiledProgram>, source: Option<Arc<Sequence>>, authored: Option<Arc<Sequence>>,
                    rows: Arc<Vec<Vec<u64>>>) -> Result<()> {
        program.validate()?;
        if source.as_ref().is_some_and(|source| source.target.abi_fingerprint() != program.target_abi_fingerprint) {
            return value_error("source target ABI differs from the compiled program");
        }
        if rows.len() as u64 > MAXIMUM_REPEAT_COUNT { return value_error("scan table row count does not fit the 32-bit SCAN_COUNT"); }
        if rows.iter().any(|row| row.len() != program.slot_count()) { return value_error("application row width differs from the compiled program"); }
        if program.slot_count() > 0 && rows.is_empty() { return value_error("a slotted program requires a non-empty value table"); }
        if program.slot_count() == 0 && !rows.is_empty() { return value_error("an unslotted program does not accept value rows"); }
        if !inner.opened { return runtime_error("PulseStreamer is not open"); }
        if inner.firing { return runtime_error("the streamer is already firing"); }
        self.0.stop.clear();
        if inner.program.as_ref().is_some_and(|held| **held == *program) && *inner.scan_rows == *rows {
            self.safe_locked(inner)?;
            self.0.stop.clear();
            if let Some(applied) = inner.applied.as_mut() { applied.source = source; applied.authored_source = authored; }
            return Ok(());
        }
        self.validate_application(&program, &rows)?;
        let words = pack_program(&program, &self.0.geom, &self.0.target)?;
        self.safe_locked(inner)?;
        self.0.stop.clear();
        inner.loaded = false;
        inner.program = None;
        inner.applied = None;
        inner.validated_execution = None;
        inner.applied_digest.clear();
        inner.scan_count = rows.len() as u64;
        inner.scan_rows = rows.clone();
        inner.run_repeats = 1;
        inner.scan_repeats = 1;
        inner.scan_armed = false;
        let mut batch = words;
        batch.push((ctrl::BANK_READY, 0b11));
        self.write(&batch, Some(&self.0.stop), None)?;
        let (status, _cursor) = self.command(inner, CMD_LOAD, 1, 1, Some(&self.0.stop))?;
        if status != STATUS_LOADED { return runtime_error(format!("LOAD did not complete successfully (STATUS=0x{status:08X})")); }
        inner.safe_readback = Some(SafeReadback { status, command_id: inner.command_id.unwrap_or(0) });
        inner.program = Some(program.clone());
        inner.loaded = true;
        inner.scan_next_chunk = 2;
        inner.scan_armed = false;
        inner.scan_last_cursor = 0;
        inner.scan_cursor_total = 0;
        inner.cursor_value = Some(0);
        inner.underflow = false;
        inner.observer_error.clear();
        let loaded_at = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map_or(0.0, |elapsed| elapsed.as_secs_f64());
        inner.applied_digest = program.digest();
        inner.applied = Some(AppliedState::new(program, source, authored, rows, 1, 1, loaded_at)?);
        Ok(())
    }

    /// FIRE the loaded application, rereading Config and reloading when the
    /// effective values changed; the confirmed application comes back.
    pub fn fire(&self, run_repeats: u64, scan_repeats: u64) -> Result<AppliedState> {
        let mut inner = self.inner();
        if let Some(applied) = inner.applied.clone() {
            let (program, source) = self.0.config.prepare(&applied.program, applied.authored_source.as_ref(), applied.source.as_ref(), &self.0.geom)?;
            let same = match (&source, &applied.source) { (Some(a), Some(b)) => Arc::ptr_eq(a, b), (None, None) => true, _ => false };
            if !same { self.load_program(&mut inner, program, source, applied.authored_source.clone(), applied.rows.clone())?; }
        }
        self.fire_program(&mut inner, run_repeats, scan_repeats)
    }

    /// FIRE the loaded application exactly as it is: its Config was resolved by
    /// whoever loaded it (a remote client, for a served board).
    pub fn fire_exact(&self, run_repeats: u64, scan_repeats: u64) -> Result<AppliedState> {
        let mut inner = self.inner();
        self.fire_program(&mut inner, run_repeats, scan_repeats)
    }

    fn fire_program(&self, inner: &mut Inner, run_repeats: u64, scan_repeats: u64) -> Result<AppliedState> {
        let run_repeats = repeat_count(run_repeats, "run_repeats")?;
        let scan_repeats = repeat_count(scan_repeats, "scan_repeats")?;
        if !inner.opened { return runtime_error("PulseStreamer is not open"); }
        if !inner.loaded || inner.program.is_none() { return runtime_error("no compiled program is loaded"); }
        if inner.firing { return runtime_error("the streamer is already firing"); }
        self.0.stop.clear();
        if !self.0.stopping.lock().unwrap_or_else(|p| p.into_inner()).is_empty() {
            return runtime_error("a Stop arrived while this FIRE was being prepared; it was not sent");
        }
        let program = inner.program.clone().expect("loaded");
        if inner.scan_rows.is_empty() && scan_repeats != 1 { return value_error("scan_repeats must be 1 when no scan table is loaded"); }
        if !inner.scan_rows.is_empty() && run_repeats != 0 && scan_repeats != 0 && inner.scan_rows.len() as u128 * u128::from(scan_repeats) > 1 << 32 {
            return value_error("finite scan row visits exceed the 32-bit CURSOR range");
        }
        if inner.validated_execution != Some((run_repeats, scan_repeats)) {
            validate_delay_capacity(&program, &inner.scan_rows, &self.0.geom, run_repeats, scan_repeats)?;
        }
        inner.validated_execution = Some((run_repeats, scan_repeats));
        inner.run_repeats = run_repeats;
        inner.scan_repeats = scan_repeats;
        let arming = self.scan_bank_arming(inner)?;
        if !arming.is_empty() {
            let mut batch = vec![(ctrl::SCAN_COUNT, inner.scan_count as u32), (ctrl::SCAN_ENABLE, u32::from(!inner.scan_rows.is_empty()))];
            batch.extend(arming);
            self.write(&batch, Some(&self.0.stop), None)?;
            inner.scan_armed = true;
        }
        inner.safe_readback = None;
        inner.done = Event::new();
        inner.underflow = false;
        inner.cursor_value = Some(0);
        inner.scan_last_cursor = 0;
        inner.scan_cursor_total = 0;
        inner.observer_error.clear();
        inner.poll_failures = 0;
        inner.resends_at_fire = self.0.transport.resends();
        inner.fire_started = Instant::now();
        inner.fire_acknowledged = inner.fire_started;
        inner.fire_finished = inner.fire_started;
        let (status, _cursor) = self.command(inner, CMD_FIRE, run_repeats as u32, scan_repeats as u32, Some(&self.0.stop))?;
        inner.fire_acknowledged = Instant::now();
        if status & STATUS_RUNNING == 0 || status & STATUS_ERROR != 0 {
            // What this host believes the board holds is no longer evidence.
            inner.loaded = false;
            inner.program = None;
            inner.applied = None;
            inner.applied_digest.clear();
            return runtime_error(format!("FIRE was not accepted (STATUS=0x{status:08X})"));
        }
        let applied = inner.applied.as_ref().expect("loaded").with_repeats(run_repeats, scan_repeats)?;
        inner.applied = Some(applied.clone());
        inner.fire_command_id = inner.command_id.unwrap_or(0);
        inner.firing = true;
        inner.terminal_status = STATUS_RUNNING;
        let shared = self.clone();
        let handle = std::thread::Builder::new().name("zlc-pulse-observer".into()).spawn(move || shared.observe())
            .map_err(|error| Error::new(Kind::Runtime, format!("cannot start the pulse observer: {error}")))?;
        *self.0.worker.lock().unwrap_or_else(|p| p.into_inner()) = Some(handle);
        Ok(applied)
    }

    /// Wait for the current FIRE (or the named one) to report; None on timeout
    /// or when that FIRE was stopped or replaced.
    pub fn wait_done(&self, timeout: Option<Duration>, command_id: Option<u32>) -> Result<Option<DoneReport>> {
        let (done, command_id) = {
            let inner = self.inner();
            if !inner.opened { return runtime_error("PulseStreamer is not open"); }
            if !inner.firing || command_id.is_some_and(|id| id != inner.fire_command_id) { return Ok(None); }
            (inner.done.clone(), inner.fire_command_id)
        };
        if !done.wait(timeout) { return Ok(None); }
        self.join_worker(Duration::from_secs(1), "pulse observer did not exit after terminal readback")?;
        let mut inner = self.inner();
        if !inner.firing || self.0.stop.is_set() || command_id != inner.fire_command_id || !Arc::ptr_eq(&done, &inner.done) { return Ok(None); }
        let report = DoneReport {
            status: inner.terminal_status, cursor: inner.cursor_value, underflow: inner.underflow,
            elapsed_seconds: inner.fire_finished.saturating_duration_since(inner.fire_started).as_secs_f64(),
            command_id: inner.fire_command_id, observer_error: inner.observer_error.clone(), poll_failures: inner.poll_failures,
            resent_frames: self.0.transport.resends().saturating_sub(inner.resends_at_fire),
            command_seconds: inner.fire_acknowledged.saturating_duration_since(inner.fire_started).as_secs_f64(),
            report_delay_seconds: inner.fire_finished.elapsed().as_secs_f64(),
        };
        inner.firing = false;
        if report.status & STATUS_DONE != 0 && report.fault().is_empty() {
            inner.safe_readback = Some(SafeReadback { status: report.status, command_id: report.command_id });
        }
        Ok(Some(report))
    }

    fn join_worker(&self, limit: Duration, message: &str) -> Result<()> {
        let handle = self.0.worker.lock().unwrap_or_else(|p| p.into_inner()).take();
        let Some(handle) = handle else { return Ok(()) };
        if handle.thread().id() == std::thread::current().id() { return Ok(()); }
        let until = Instant::now() + limit;
        while !handle.is_finished() {
            if Instant::now() >= until {
                *self.0.worker.lock().unwrap_or_else(|p| p.into_inner()) = Some(handle);
                return runtime_error(message.to_owned());
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        let _ = handle.join();
        Ok(())
    }

    pub fn cursor(&self) -> Result<Option<u32>> {
        let inner = self.inner();
        if inner.firing { return Ok(inner.cursor_value); }
        if !inner.opened { return Ok(None); }
        Ok(Some(self.0.transport.read_word(ctrl::CURSOR, None, None)?))
    }

    /// Stop now: the observer stopped, `CMD_SAFE`, the readback verified.
    pub fn safe(&self) -> Result<SafeReadback> {
        let caller = std::thread::current().id();
        self.0.stopping.lock().unwrap_or_else(|p| p.into_inner()).insert(caller);
        let result = (|| {
            self.stop_worker()?;
            let mut inner = self.inner();
            self.0.stopping.lock().unwrap_or_else(|p| p.into_inner()).remove(&caller);
            self.safe_locked(&mut inner)
        })();
        self.0.stopping.lock().unwrap_or_else(|p| p.into_inner()).remove(&caller);
        result
    }

    fn safe_locked(&self, inner: &mut Inner) -> Result<SafeReadback> {
        self.0.stop.set();
        inner.done.set();
        if !inner.opened { return runtime_error("PulseStreamer is not open"); }
        if !inner.safe_readback.is_some_and(|readback| readback.stable()) {
            let (status, _cursor) = self.command(inner, CMD_SAFE, 1, 1, None)?;
            if status != 0 { return runtime_error(format!("SAFE did not complete (STATUS=0x{status:08X})")); }
            inner.safe_readback = Some(SafeReadback { status, command_id: inner.command_id.unwrap_or(0) });
        }
        inner.firing = false;
        let readback = inner.safe_readback.expect("set");
        inner.terminal_status = readback.status;
        Ok(readback)
    }

    /// Stop the observer, and refuse to go on if it will not stop: it holds the
    /// I/O lock while it reads, so a transport that ignores the stop event would
    /// make `CMD_SAFE` wait for ever.
    fn stop_worker(&self) -> Result<()> {
        self.0.stop.set();
        self.inner().done.set();
        self.join_worker(OBSERVER_STOP_TIMEOUT, "pulse observer did not stop")
    }

    pub fn snapshot(&self) -> Snapshot {
        let inner = self.inner();
        Snapshot { opened: inner.opened, loaded: inner.loaded, firing: inner.firing, run_repeats: inner.run_repeats, scan_repeats: inner.scan_repeats,
                   cursor: inner.cursor_value, scan_count: inner.scan_count, scan_next_chunk: inner.scan_next_chunk, underflow: inner.underflow,
                   status: inner.terminal_status, applied_digest: inner.applied_digest.clone(), config_source: self.0.config.source() }
    }

    pub fn applied(&self) -> Option<AppliedState> { self.inner().applied.clone() }

    /// The FIRE command id of the current run (0 before any).
    pub fn fire_command_id(&self) -> u32 { self.inner().fire_command_id }
    pub fn is_firing(&self) -> bool { self.inner().firing }
    pub fn stop_requested(&self) -> bool { self.0.stop.is_set() }
    pub fn command_id(&self) -> Option<u32> { self.inner().command_id }

    // ------------------------------------------------------------ observer
    fn observe(&self) {
        let shared = &self.0;
        let mut consecutive = 0;
        loop {
            if shared.stop.is_set() { return; }
            let words = shared.transport.read_words(ctrl::STATUS, ctrl::CURSOR - ctrl::STATUS + 1, Some(&shared.stop), None);
            let (status, cursor) = match words {
                Ok(words) => (words[0], *words.last().expect("words")),
                Err(error) => {
                    if shared.stop.is_set() { return; }
                    consecutive += 1;
                    let mut inner = self.inner();
                    inner.poll_failures += 1;
                    if consecutive >= 2 {
                        inner.observer_error = format!("{}: {}", error.kind.python_name(), error.message);
                        inner.fire_finished = Instant::now();
                        inner.done.set();
                        return;
                    }
                    continue;
                }
            };
            consecutive = 0;
            let mut inner = self.inner();
            inner.cursor_value = Some(cursor);
            inner.underflow |= status & STATUS_UNDERFLOW != 0;
            if status & (STATUS_UNDERFLOW | STATUS_DONE | STATUS_ERROR) != 0 {
                inner.terminal_status = status;
                inner.fire_finished = Instant::now();
                inner.done.set();
                return;
            }
            if !inner.scan_rows.is_empty() {
                if let Err(error) = self.refill(&mut inner, cursor) {
                    if !shared.stop.is_set() {
                        inner.observer_error = format!("{}: {}", error.kind.python_name(), error.message);
                        inner.fire_finished = Instant::now();
                        inner.done.set();
                    }
                    return;
                }
            }
            drop(inner);
            // Slept in slices of the default period, so a Stop joining this
            // thread is not held behind a slow link's period.
            let resume = Instant::now() + shared.observer_interval;
            while !shared.stop.is_set() {
                let left = resume.saturating_duration_since(Instant::now());
                if left.is_zero() { break; }
                std::thread::sleep(left.min(DEFAULT_OBSERVER_INTERVAL));
            }
        }
    }

    fn initial_ready(&self, count: u64) -> u32 { u32::from(count > 0) | if count > self.0.geom.bank_size { 2 } else { 0 } }

    /// Rows that put the scan banks back at chunks 0/1, ready for point 0.
    fn scan_bank_arming(&self, inner: &mut Inner) -> Result<Vec<(u32, u32)>> {
        if inner.scan_count == 0 || inner.scan_armed { return Ok(Vec::new()); }
        let mut rows = Vec::new();
        for chunk in 0..2u64 {
            if chunk * self.0.geom.bank_size < inner.scan_count { rows.extend(pack_scan_rows(&inner.scan_rows, &self.0.geom, chunk & 1, chunk)?); }
        }
        rows.push((ctrl::BANK0_CHUNK, 0));
        if self.0.geom.bank_size < inner.scan_count { rows.push((ctrl::BANK1_CHUNK, 1)); }
        rows.push((ctrl::BANK_READY, self.initial_ready(inner.scan_count)));
        inner.scan_next_chunk = 2;
        inner.scan_last_cursor = 0;
        inner.scan_cursor_total = 0;
        Ok(rows)
    }

    /// Keep the far bank one chunk ahead of where the engine plays. CURSOR is a
    /// cumulative row-visit ordinal, so the sweep needs no witnessed wrap.
    fn refill(&self, inner: &mut Inner, cursor: u32) -> Result<()> {
        let geom = &self.0.geom;
        if inner.scan_count <= 2 * geom.bank_size { return Ok(()); }
        inner.scan_cursor_total += u64::from(cursor.wrapping_sub(inner.scan_last_cursor as u32));
        inner.scan_last_cursor = u64::from(cursor);
        let chunks_per_sweep = inner.scan_count.div_ceil(geom.bank_size);
        let (sweep, row) = (inner.scan_cursor_total / inner.scan_count, inner.scan_cursor_total % inner.scan_count);
        let current = sweep * chunks_per_sweep + row / geom.bank_size;
        while inner.scan_next_chunk <= current + 1 {
            if inner.scan_repeats != 0 && inner.scan_next_chunk >= chunks_per_sweep * inner.scan_repeats { return Ok(()); }
            let stream_chunk = inner.scan_next_chunk;
            let table_chunk = stream_chunk % chunks_per_sweep;
            let bank = stream_chunk & 1;
            let words = pack_scan_rows(&inner.scan_rows, geom, bank, table_chunk)?;
            let register = if bank == 0 { ctrl::BANK0_CHUNK } else { ctrl::BANK1_CHUNK };
            inner.scan_armed = false;
            // The chunk register is what makes the bank resident: written on its
            // own, once every scan word has been acknowledged.
            self.write(&words, Some(&self.0.stop), None)?;
            self.write(&[(register, table_chunk as u32)], Some(&self.0.stop), None)?;
            inner.scan_next_chunk += 1;
        }
        Ok(())
    }

    // ------------------------------------------------------------ I/O
    fn command(&self, inner: &mut Inner, code: u32, run_repeats: u32, scan_repeats: u32, stop: Option<&Event>) -> Result<(u32, u32)> {
        inner.safe_readback = None;
        let current = match inner.command_id { Some(id) => id, None => self.0.transport.read_word(ctrl::ACK_ID, stop, None)? };
        let next = current.wrapping_add(1);
        let next = if next == 0 { 1 } else { next };
        inner.command_id = Some(next);
        let deadline = Instant::now() + if code == CMD_SAFE { SAFE_TIMEOUT } else { LOAD_TIMEOUT };
        self.0.transport.command(code, next, run_repeats, scan_repeats, stop, Some(deadline))
    }

    /// Data words only: commands use their completion protocol and identity.
    fn write(&self, rows: &[(u32, u32)], stop: Option<&Event>, deadline: Option<Instant>) -> Result<()> {
        debug_assert!(rows.iter().all(|(address, _)| *address != ctrl::COMMAND), "commands must use the completion protocol");
        self.0.transport.write_words(rows, stop, deadline)
    }

    fn validate_application(&self, program: &CompiledProgram, rows: &[Vec<u64>]) -> Result<()> {
        if program.target_abi_fingerprint != self.0.target.abi_fingerprint() { return value_error("compiled target ABI does not match the connected sequencer"); }
        if program.clock_hz != self.0.clock_hz { return value_error("compiled clock does not match the connected sequencer"); }
        if program.geometry_fingerprint != build_fingerprint(&self.0.geom)? { return value_error("compiled geometry does not match the connected sequencer"); }
        let tick_limit = 1u128 << self.0.geom.tick_width;
        let code_limit = 1u128 << self.0.geom.bus_width;
        let empty = [Vec::new()];
        for row in if rows.is_empty() { &empty[..] } else { rows } {
            for (index, (kind, value)) in program.slot_kinds.iter().zip(row).enumerate() {
                let value = u128::from(*value);
                if *kind == FieldKind::Duration {
                    if !(1..tick_limit).contains(&value) {
                        return value_error(format!("scan slot {index} duration {value} is outside the board's [1, {}] tick range", tick_limit - 1));
                    }
                } else if value >= code_limit {
                    return value_error(format!("scan slot {index} DAC code {value} is outside the board's [0, {}] range", code_limit - 1));
                }
            }
        }
        Ok(())
    }
}
