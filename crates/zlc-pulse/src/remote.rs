//! The thin JSON RPC between one pulse server (the process that owns a board)
//! and its current client, wire-compatible with v2's `zlc_pulse.remote`
//! (board_protocol 8): frames are a 4-byte big-endian length and compact JSON;
//! model values travel as `{"__type__": name, field: value, ...}` trees.
//!
//! One owner at a time: the first valid control request claims the board; a
//! newcomer takes over only after the old owner's outputs are verified SAFE, and
//! a lost connection or a server shutdown drives SAFE too. `wait_done` and the
//! owner's Stop travel on connections of their own, beside the command lane.
use std::collections::{BTreeSet, HashSet};
use std::io::{Read, Write};
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::ThreadId;
use std::time::{Duration, Instant, SystemTime};

use crate::codec::{sequence_from_tree, sequence_to_tree, target_from_tree, target_to_tree};
use crate::compile::{CompiledProgram, TargetBusAction, TargetBusDelay, bus_action, bus_delay};
use crate::device::{AppliedState, BoardDescription, ConfigHolder, DoneReport, PulseStreamer, SafeReadback};
use crate::error::{Error, Kind, Result, value_error};
use crate::json::{self, Json};
use crate::model::{FieldKind, Sequence};
use crate::wire::{LAYOUT_STRUCT_VERSION, PARAM_FIELDS, StreamerParams};

pub const DEFAULT_PORT: u16 = 18861;
pub const DEFAULT_HOST: &str = "127.0.0.1";
pub const DEFAULT_BIND_HOST: &str = "0.0.0.0";
pub const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
pub const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
pub const MAX_FRAME_BYTES: usize = 8 * 1024 * 1024;
const WAIT_SLICE: Duration = Duration::from_secs(1);
const CONNECTION_ENDED: &str = "the connection to the pulse server ended; if another editor has connected since, that one now holds the board";
pub const REMOTE_METHODS: [&str; 10] = ["open", "describe", "close", "load", "fire", "wait_done", "cursor", "safe", "snapshot", "applied"];

// ------------------------------------------------------------------ trees
fn typed(name: &str, mut items: Vec<(&str, Json)>) -> Json {
    items.insert(0, ("__type__", Json::str(name)));
    Json::object(items)
}

/// Strip `__type__` markers, leaving the plain dataclass trees.
fn untyped(value: &Json) -> Json {
    match value {
        Json::Object(items) => Json::Object(items.iter().filter(|(key, _)| key != "__type__").map(|(k, v)| (k.clone(), untyped(v))).collect()),
        Json::Array(items) => Json::Array(items.iter().map(untyped).collect()),
        other => other.clone(),
    }
}

fn type_of(value: &Json) -> Option<&str> { value.get("__type__").and_then(Json::as_str) }

fn expect_type<'a>(value: &'a Json, name: &str) -> Result<&'a [(String, Json)]> {
    match (type_of(value), value) {
        (Some(found), Json::Object(items)) if found == name => Ok(items),
        (Some(found), _) => value_error(format!("expected a {name} remote tree, got {found}")),
        _ => value_error(format!("expected a {name} remote tree")),
    }
}

fn field<'a>(items: &'a [(String, Json)], key: &str, owner: &str) -> Result<&'a Json> {
    items.iter().find(|(name, _)| name == key).map(|(_, value)| value)
        .ok_or_else(|| Error::new(Kind::Value, format!("missing {owner} remote field(s): {key}")))
}

fn int(value: &Json, name: &str) -> Result<i128> { value.as_int().ok_or_else(|| Error::new(Kind::Type, format!("{name} must be an integer"))) }
fn uint(value: &Json, name: &str) -> Result<u64> { u64::try_from(int(value, name)?).map_err(|_| Error::new(Kind::Value, format!("{name} must be non-negative"))) }
fn float(value: &Json, name: &str) -> Result<f64> { value.as_f64().ok_or_else(|| Error::new(Kind::Type, format!("{name} must be a number"))) }
fn text(value: &Json, name: &str) -> Result<String> { value.as_str().map(str::to_owned).ok_or_else(|| Error::new(Kind::Type, format!("{name} must be text"))) }
fn list<'a>(value: &'a Json, name: &str) -> Result<&'a [Json]> { value.as_array().ok_or_else(|| Error::new(Kind::Type, format!("{name} must be a list"))) }

pub fn sequence_tree(sequence: &Sequence) -> Json {
    let Json::Object(items) = sequence_to_tree(sequence) else { unreachable!() };
    let get = |key: &str| items.iter().find(|(name, _)| name == key).map(|(_, value)| value.clone()).unwrap_or(Json::Null);
    let wrap = |name: &str, value: Json| match value {
        Json::Array(entries) => Json::Array(entries.into_iter().map(|entry| match entry {
            Json::Object(fields) => { let mut fields = fields; fields.insert(0, ("__type__".into(), Json::str(name))); Json::Object(fields) }
            other => other,
        }).collect()),
        other => other,
    };
    let periods = match get("periods") {
        Json::Array(periods) => Json::Array(periods.into_iter().map(|period| {
            let Json::Object(fields) = period else { unreachable!() };
            let get = |key: &str| fields.iter().find(|(name, _)| name == key).map(|(_, value)| value.clone()).unwrap_or(Json::Null);
            typed("PulsePeriod", vec![("period_id", get("period_id")), ("duration", get("duration")), ("unit", get("unit")),
                                      ("states", get("states")), ("analog_steps", wrap("AnalogStep", get("analog_steps"))),
                                      ("name", get("name")), ("kind", get("kind"))])
        }).collect()),
        other => other,
    };
    let bindings = Json::Array(sequence.bindings.iter().map(|binding| typed("PulseBinding", vec![
        ("field_ref", typed("PulseFieldRef", vec![("kind", Json::str(binding.field_ref.kind.name())),
            ("period_id", Json::opt_str(binding.field_ref.period_id.as_deref())), ("port", Json::opt_str(binding.field_ref.port.as_deref())),
            ("bracket_id", Json::opt_str(binding.field_ref.bracket_id.as_deref()))])),
        ("unit", Json::str(&binding.unit)), ("scan", Json::Bool(binding.scan)), ("source", Json::str(binding.source.name())),
        ("config_key", Json::str(&binding.config_key)),
    ])).collect());
    typed("PulseSequence", vec![
        ("name", get("name")), ("target", target_tree(&sequence.target)), ("time_step_ns", get("time_step_ns")), ("periods", periods),
        ("bindings", bindings), ("delays", wrap("OutputDelay", get("delays"))), ("brackets", wrap("PulseBracket", get("brackets"))),
        ("run_repeats", get("run_repeats")),
        ("components", match items.iter().find(|(name, _)| name == "components") { Some((_, value)) => wrap("PulseComponent", value.clone()), None => Json::Array(vec![]) }),
        ("config_enabled", get("config_enabled")),
    ])
}

pub fn target_tree(target: &crate::model::Target) -> Json {
    let Json::Object(items) = target_to_tree(target) else { unreachable!() };
    let get = |key: &str| items.iter().find(|(name, _)| name == key).map(|(_, value)| value.clone()).unwrap_or(Json::Null);
    let ports = match get("ports") {
        Json::Array(ports) => Json::Array(ports.into_iter().map(|port| match port {
            Json::Object(mut fields) => { fields.insert(0, ("__type__".into(), Json::str("PulsePortSpec"))); Json::Object(fields) }
            other => other,
        }).collect()),
        other => other,
    };
    typed("PulseTarget", vec![("raw_lanes", get("raw_lanes")), ("ports", ports), ("package_pins", get("package_pins"))])
}

pub fn sequence_from_remote(value: &Json) -> Result<Sequence> {
    expect_type(value, "PulseSequence")?;
    let Json::Object(mut items) = untyped(value) else { unreachable!() };
    items.insert(0, ("format".into(), Json::str(crate::codec::PULSE_TREE_FORMAT)));
    sequence_from_tree(&Json::Object(items))
}

pub fn program_tree(program: &CompiledProgram) -> Json {
    typed("CompiledProgram", vec![
        ("clock_hz", Json::Float(program.clock_hz)), ("target_abi_fingerprint", Json::str(&program.target_abi_fingerprint)),
        ("geometry_fingerprint", Json::Int(program.geometry_fingerprint.into())), ("channels", Json::strings(&program.channels)),
        ("durations", Json::ints(&program.durations)), ("duration_slots", Json::ints(&program.duration_slots)),
        ("masks", Json::Array(program.masks.iter().map(|mask| Json::Int(*mask as i128)).collect())),
        ("loops", Json::Array(program.loops.iter().map(|&(a, b, c)| Json::Array(vec![Json::Int(a as i128), Json::Int(b as i128), Json::Int(c.into())])).collect())),
        ("duration_seconds", Json::Float(program.duration_seconds)),
        ("slot_kinds", Json::Array(program.slot_kinds.iter().map(|kind| Json::str(kind.name())).collect())),
        ("bus_names", Json::strings(&program.bus_names)),
        ("bus_actions", Json::Array(program.bus_actions.iter().map(|action| typed("TargetBusAction", vec![
            ("row", Json::Int(action.row.into())), ("bus_index", Json::Int(action.bus_index.into())), ("bus_name", Json::str(&action.bus_name)),
            ("mode", Json::str(action.mode.name())), ("value", Json::Int(action.value.into())), ("value_select", Json::Int(action.value_select.into())),
        ])).collect())),
        ("bus_delays", Json::Array(program.bus_delays.iter().map(|delay| typed("TargetBusDelay", vec![
            ("bus_index", Json::Int(delay.bus_index.into())), ("delay_ticks", Json::Int(delay.delay_ticks.into())),
        ])).collect())),
        ("channel_delays", Json::ints(&program.channel_delays)), ("clk_enable", Json::Int(program.clk_enable as i128)),
        ("logical_digital_outputs", Json::Array(program.logical_digital_outputs.iter().map(|(a, b)| Json::Array(vec![Json::str(a), Json::str(b)])).collect())),
        ("bus_safe_values", Json::ints(&program.bus_safe_values)),
    ])
}

pub fn program_from_remote(value: &Json) -> Result<CompiledProgram> {
    let items = expect_type(value, "CompiledProgram")?;
    let f = |key: &str| field(items, key, "CompiledProgram");
    let ints = |key: &str| -> Result<Vec<u64>> { list(f(key)?, key)?.iter().map(|v| uint(v, key)).collect() };
    let actions: Vec<TargetBusAction> = list(f("bus_actions")?, "bus_actions")?.iter().map(|action| {
        let items = expect_type(action, "TargetBusAction")?;
        let g = |key: &str| field(items, key, "TargetBusAction");
        bus_action(int(g("row")?, "row")?, int(g("bus_index")?, "bus_index")?, &text(g("bus_name")?, "bus_name")?, &text(g("mode")?, "mode")?,
                   int(g("value")?, "value")?, int(g("value_select")?, "value_select")?)
    }).collect::<Result<_>>()?;
    let delays: Vec<TargetBusDelay> = list(f("bus_delays")?, "bus_delays")?.iter().map(|delay| {
        let items = expect_type(delay, "TargetBusDelay")?;
        bus_delay(int(field(items, "bus_index", "TargetBusDelay")?, "bus_index")?, int(field(items, "delay_ticks", "TargetBusDelay")?, "delay_ticks")?)
    }).collect::<Result<_>>()?;
    let loops = list(f("loops")?, "loops")?.iter().map(|entry| {
        let entry = list(entry, "loop")?;
        if entry.len() != 3 { return value_error("a loop is (first, last, count)"); }
        Ok((uint(&entry[0], "loop")? as usize, uint(&entry[1], "loop")? as usize, uint(&entry[2], "loop")?))
    }).collect::<Result<_>>()?;
    let kinds = list(f("slot_kinds")?, "slot_kinds")?.iter().map(|kind| {
        FieldKind::parse(kind.as_str().unwrap_or("")).filter(|kind| kind.scannable()).ok_or_else(|| Error::new(Kind::Value, "program contains an unsupported slot kind"))
    }).collect::<Result<_>>()?;
    let program = CompiledProgram {
        clock_hz: float(f("clock_hz")?, "clock_hz")?,
        target_abi_fingerprint: text(f("target_abi_fingerprint")?, "target_abi_fingerprint")?,
        geometry_fingerprint: u32::try_from(uint(f("geometry_fingerprint")?, "geometry_fingerprint")?)
            .map_err(|_| Error::new(Kind::Value, "geometry_fingerprint must be an unsigned 32-bit integer"))?,
        channels: list(f("channels")?, "channels")?.iter().map(|lane| text(lane, "channel")).collect::<Result<_>>()?,
        durations: ints("durations")?, duration_slots: ints("duration_slots")?,
        masks: list(f("masks")?, "masks")?.iter().map(|mask| int(mask, "mask").and_then(|mask| u128::try_from(mask)
            .map_err(|_| Error::new(Kind::Value, "row masks must be non-negative")))).collect::<Result<_>>()?,
        loops, duration_seconds: float(f("duration_seconds")?, "duration_seconds")?, slot_kinds: kinds,
        bus_names: list(f("bus_names")?, "bus_names")?.iter().map(|name| text(name, "bus name")).collect::<Result<_>>()?,
        bus_actions: actions, bus_delays: delays, channel_delays: ints("channel_delays")?,
        clk_enable: u128::try_from(int(f("clk_enable")?, "clk_enable")?).map_err(|_| Error::new(Kind::Value, "clk_enable must be non-negative"))?,
        logical_digital_outputs: list(f("logical_digital_outputs")?, "logical_digital_outputs")?.iter().map(|pair| {
            match pair.as_array() { Some([a, b]) => Ok((text(a, "output")?, text(b, "lane")?)), _ => value_error("a logical output is (key, lane)") }
        }).collect::<Result<_>>()?,
        bus_safe_values: ints("bus_safe_values")?,
    };
    program.validate()?;
    Ok(program)
}

fn params_tree(params: &StreamerParams) -> Json {
    let mut items: Vec<(&str, Json)> = PARAM_FIELDS.iter().map(|name| (*name, Json::Int(params.get(name).expect("field").into()))).collect();
    items.insert(0, ("__type__", Json::str("StreamerParams")));
    Json::object(items)
}

fn params_from_remote(value: &Json) -> Result<StreamerParams> {
    let items = expect_type(value, "StreamerParams")?;
    let mut params = StreamerParams::default();
    for name in PARAM_FIELDS { params.set(name, uint(field(items, name, "StreamerParams")?, name)?); }
    Ok(params)
}

pub fn description_tree(description: &BoardDescription) -> Json {
    typed("BoardDescription", vec![("target", target_tree(&description.target)), ("geometry", params_tree(&description.geometry)),
                                   ("clock_hz", Json::Float(description.clock_hz))])
}

pub fn description_from_remote(value: &Json) -> Result<BoardDescription> {
    let items = expect_type(value, "BoardDescription")?;
    let target = untyped(field(items, "target", "BoardDescription")?);
    Ok(BoardDescription { target: target_from_tree(&target)?, geometry: params_from_remote(field(items, "geometry", "BoardDescription")?)?,
                          clock_hz: float(field(items, "clock_hz", "BoardDescription")?, "clock_hz")? })
}

pub fn safe_tree(readback: &SafeReadback) -> Json {
    typed("SafeReadback", vec![("status", Json::Int(readback.status.into())), ("command_id", Json::Int(readback.command_id.into()))])
}

pub fn safe_from_remote(value: &Json) -> Result<SafeReadback> {
    let items = expect_type(value, "SafeReadback")?;
    Ok(SafeReadback { status: uint(field(items, "status", "SafeReadback")?, "status")? as u32,
                      command_id: uint(field(items, "command_id", "SafeReadback")?, "command_id")? as u32 })
}

pub fn report_tree(report: &DoneReport) -> Json {
    typed("DoneReport", vec![
        ("status", Json::Int(report.status.into())), ("cursor", report.cursor.map_or(Json::Null, |c| Json::Int(c.into()))),
        ("underflow", Json::Bool(report.underflow)), ("elapsed_seconds", Json::Float(report.elapsed_seconds)),
        ("command_id", Json::Int(report.command_id.into())), ("observer_error", Json::str(&report.observer_error)),
        ("poll_failures", Json::Int(report.poll_failures.into())), ("resent_frames", Json::Int(report.resent_frames.into())),
        ("command_seconds", Json::Float(report.command_seconds)), ("report_delay_seconds", Json::Float(report.report_delay_seconds)),
    ])
}

pub fn report_from_remote(value: &Json) -> Result<DoneReport> {
    let items = expect_type(value, "DoneReport")?;
    let f = |key: &str| field(items, key, "DoneReport");
    Ok(DoneReport {
        status: uint(f("status")?, "status")? as u32,
        cursor: match f("cursor")? { Json::Null => None, value => Some(uint(value, "cursor")? as u32) },
        underflow: f("underflow")?.as_bool().unwrap_or(false), elapsed_seconds: float(f("elapsed_seconds")?, "elapsed_seconds")?,
        command_id: uint(f("command_id")?, "command_id")? as u32, observer_error: text(f("observer_error")?, "observer_error")?,
        poll_failures: uint(f("poll_failures")?, "poll_failures")?, resent_frames: uint(f("resent_frames")?, "resent_frames")?,
        command_seconds: float(f("command_seconds")?, "command_seconds")?, report_delay_seconds: float(f("report_delay_seconds")?, "report_delay_seconds")?,
    })
}

fn rows_tree(rows: &[Vec<u64>]) -> Json { Json::Array(rows.iter().map(|row| Json::ints(&row.iter().map(|v| *v as i64).collect::<Vec<_>>())).collect()) }

fn rows_from(value: &Json) -> Result<Vec<Vec<u64>>> {
    list(value, "rows")?.iter().map(|row| list(row, "row")?.iter().map(|v| uint(v, "row value")).collect()).collect()
}

pub fn applied_tree(applied: &AppliedState) -> Json {
    typed("AppliedState", vec![
        ("program", program_tree(&applied.program)), ("source", applied.source.as_deref().map_or(Json::Null, sequence_tree)),
        ("rows", rows_tree(&applied.rows)), ("run_repeats", Json::Int(applied.run_repeats.into())),
        ("scan_repeats", Json::Int(applied.scan_repeats.into())), ("loaded_at", Json::Float(applied.loaded_at)),
        ("authored_source", applied.authored_source.as_deref().map_or(Json::Null, sequence_tree)),
    ])
}

pub fn applied_from_remote(value: &Json) -> Result<AppliedState> {
    let items = expect_type(value, "AppliedState")?;
    let f = |key: &str| field(items, key, "AppliedState");
    let sequence = |value: &Json| -> Result<Option<Arc<Sequence>>> { if value.is_null() { Ok(None) } else { sequence_from_remote(value).map(|s| Some(Arc::new(s))) } };
    AppliedState::new(Arc::new(program_from_remote(f("program")?)?), sequence(f("source")?)?, sequence(f("authored_source")?)?,
                      Arc::new(rows_from(f("rows")?)?), uint(f("run_repeats")?, "run_repeats")?, uint(f("scan_repeats")?, "scan_repeats")?,
                      float(f("loaded_at")?, "loaded_at")?)
}

// ------------------------------------------------------------------ frames
fn send_frame(stream: &mut TcpStream, value: &Json, deadline: Option<Instant>) -> Result<()> {
    let payload = value.compact_text().into_bytes();
    if payload.len() > MAX_FRAME_BYTES { return value_error("remote frame exceeds the maximum size"); }
    if let Some(deadline) = deadline {
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() { return Err(Error::new(Kind::Timeout, "remote request deadline expired before send")); }
        stream.set_write_timeout(Some(left))?;
    }
    let mut frame = (payload.len() as u32).to_be_bytes().to_vec();
    frame.extend_from_slice(&payload);
    stream.write_all(&frame)?;
    Ok(())
}

/// Read exactly `size` bytes; `None` for a clean end before the first byte.
fn recv_exact(stream: &mut TcpStream, size: usize, deadline: Option<Instant>) -> Result<Option<Vec<u8>>> {
    let mut buffer = vec![0u8; size];
    let mut filled = 0;
    while filled < size {
        if let Some(deadline) = deadline {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() { return Err(Error::new(Kind::Timeout, "remote response deadline expired")); }
            stream.set_read_timeout(Some(left))?;
        }
        match stream.read(&mut buffer[filled..]) {
            Ok(0) if filled == 0 => return Ok(None),
            Ok(0) => return Err(Error::new(Kind::Connection, "remote connection closed mid-frame")),
            Ok(count) => filled += count,
            Err(error) if matches!(error.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut) =>
                return Err(Error::new(Kind::Timeout, "remote response deadline expired")),
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error.into()),
        }
    }
    Ok(Some(buffer))
}

/// The tree of one frame, not the objects it describes: a payload this side
/// cannot interpret is an error on the request, never a dropped connection.
fn recv_frame(stream: &mut TcpStream, deadline: Option<Instant>) -> Result<Option<Json>> {
    let Some(header) = recv_exact(stream, 4, deadline)? else { return Ok(None) };
    let size = u32::from_be_bytes(header.try_into().unwrap()) as usize;
    if size > MAX_FRAME_BYTES { return value_error("remote frame exceeds the maximum size"); }
    let payload = recv_exact(stream, size, deadline)?.ok_or_else(|| Error::new(Kind::Connection, "remote connection closed before frame payload"))?;
    let text = String::from_utf8(payload).map_err(|_| Error::new(Kind::Value, "remote JSON is not UTF-8"))?;
    Ok(Some(json::parse(&text, "remote JSON")?))
}

/// An exception the server raised while handling one request.
#[derive(Clone, Debug)]
pub struct RemoteError { pub remote_type: String, pub message: String }

fn remote_error(remote_type: &str, message: &str) -> Error {
    let kind = match remote_type {
        "ValueError" => Kind::Value, "TypeError" => Kind::Type, "KeyError" => Kind::Key, "TimeoutError" => Kind::Timeout,
        "ConnectionError" | "ConnectionResetError" => Kind::Connection, _ => Kind::Runtime,
    };
    Error::new(kind, format!("{remote_type}: {message}"))
}

// ------------------------------------------------------------------ server log
/// One lifecycle line on stdout (and to whoever embeds the server).
pub fn server_log(event: &str, client: Option<&str>, detail: &str) {
    let now = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap_or_default();
    let mut line = format!("ZLC {event}");
    if let Some(client) = client { line.push_str(&format!(" client={client}")); }
    if !detail.is_empty() { line.push(' '); line.push_str(detail); }
    println!("[{:.3}] {line}", now.as_secs_f64());
    if let Some(sink) = LOG_SINK.lock().unwrap_or_else(|p| p.into_inner()).as_ref() { sink(&line); }
}

type Sink = Box<dyn Fn(&str) + Send>;
static LOG_SINK: Mutex<Option<Sink>> = Mutex::new(None);

/// Send every server line to `sink` as well (a bench window).
pub fn set_log_sink(sink: Option<Sink>) { *LOG_SINK.lock().unwrap_or_else(|p| p.into_inner()) = sink; }

fn is_loopback(address: &SocketAddr) -> bool {
    match address {
        SocketAddr::V4(v4) => v4.ip().is_loopback(),
        SocketAddr::V6(v6) => v6.ip().is_loopback() || v6.ip().to_ipv4_mapped().is_some_and(|ip| ip.is_loopback()),
    }
}

// ------------------------------------------------------------------ server
struct Ownership { epoch: u64, client: Option<String>, connection: Option<u64>, token: Option<String>, fault: Option<String> }

struct ServerShared {
    streamer: PulseStreamer,
    ownership: Mutex<Ownership>,
    command_lane: Mutex<()>,
    peers: AtomicBool,
    connections: Mutex<Vec<(u64, TcpStream)>>,
    next_connection: AtomicU64,
    closing: AtomicBool,
    served: AtomicBool,
}

/// One board owner served over TCP.
pub struct PulseRemoteServer { shared: Arc<ServerShared>, listener: TcpListener, address: SocketAddr, thread: Option<std::thread::JoinHandle<()>> }

fn token() -> String {
    let mut bytes = [0u8; 16];
    for chunk in bytes.chunks_mut(8) {
        let seed = std::collections::hash_map::RandomState::new();
        use std::hash::{BuildHasher, Hasher};
        let mut hasher = seed.build_hasher();
        hasher.write_u128(SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap_or_default().as_nanos());
        chunk.copy_from_slice(&hasher.finish().to_le_bytes());
    }
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn drop_connection(stream: &TcpStream) { let _ = stream.shutdown(Shutdown::Both); }

impl ServerShared {
    fn ownership(&self) -> MutexGuard<'_, Ownership> { self.ownership.lock().unwrap_or_else(|p| p.into_inner()) }

    fn drop_registered(&self, connection: u64) {
        if let Some((_, stream)) = self.connections.lock().unwrap_or_else(|p| p.into_inner()).iter().find(|(id, _)| *id == connection) {
            drop_connection(stream);
        }
    }

    /// SAFE that decides a takeover or a disconnect, on the command lane. A
    /// board no longer open is not a failure: nothing it drives can be unsafe.
    fn safe_on_lane(&self) -> std::result::Result<Option<SafeReadback>, Error> {
        match self.streamer.safe() {
            Ok(readback) if readback.stable() => Ok(Some(readback)),
            Ok(_) => Err(Error::new(Kind::Runtime, "SAFE readback was not stable")),
            Err(error) => if self.streamer.snapshot().opened { Err(error) } else { Ok(None) },
        }
    }

    fn claim(&self, client: &str, connection: u64) -> Result<()> {
        let (previous, to_drop, transition) = {
            let mut owner = self.ownership();
            if owner.client.is_none() && owner.fault.is_none() {
                owner.epoch += 1;
                owner.client = Some(client.to_owned());
                owner.connection = Some(connection);
                owner.token = Some(token());
                return Ok(());
            }
            let previous = owner.client.take();
            let to_drop = owner.connection.take();
            owner.token = None;
            owner.epoch += 1;
            owner.fault = Some("takeover SAFE has not completed".into());
            (previous, to_drop, owner.epoch)
        };
        if let Some(previous) = &previous { server_log("CLIENT REPLACED", Some(previous), &format!("by={client}")); }
        if let Some(old) = to_drop { self.drop_registered(old); }
        let _ = self.streamer.safe();
        let _lane = self.command_lane.lock().unwrap_or_else(|p| p.into_inner());
        let outcome = self.safe_on_lane();
        let mut owner = self.ownership();
        if owner.epoch != transition { return Err(Error::new(Kind::Runtime, "client takeover was superseded")); }
        match outcome {
            Ok(_) => {
                owner.fault = None;
                owner.client = Some(client.to_owned());
                owner.connection = Some(connection);
                owner.token = Some(token());
                Ok(())
            }
            Err(error) => {
                let message = format!("takeover SAFE failed: {}: {}", error.kind.python_name(), error.message.replace('\n', " "));
                owner.fault = Some(message.clone());
                drop(owner);
                server_log("TAKEOVER FAILED", Some(client), &format!("error={message:?}"));
                Err(Error::new(Kind::Runtime, message))
            }
        }
    }

    fn owns(&self, client: &str, connection: u64, epoch: Option<u64>) -> bool {
        let owner = self.ownership();
        owner.fault.is_none() && owner.client.as_deref() == Some(client) && owner.connection == Some(connection) && epoch.is_none_or(|e| e == owner.epoch)
    }

    fn dispatch(&self, method: &str, params: &Json, client: &str, connection: u64) -> Result<Json> {
        let lost = || Error::new(Kind::Runtime, "this connection no longer owns the pulse server");
        let epoch = { let owner = self.ownership(); if !self.owns_locked(&owner, client, connection) { return Err(lost()); } owner.epoch };
        let _lane = self.command_lane.lock().unwrap_or_else(|p| p.into_inner());
        if !self.owns(client, connection, Some(epoch)) { return Err(lost()); }
        let Json::Object(items) = params else { return value_error("request params must be an object") };
        let keys: BTreeSet<&str> = items.iter().map(|(key, _)| key.as_str()).collect();
        let expected: BTreeSet<&str> = match method {
            "load" => ["program", "source", "authored_source", "reuse_authored_source", "rows", "reuse_rows"].into(),
            "fire" => ["run_repeats", "scan_repeats"].into(),
            "open" => ["board_protocol", "operation_budget"].into(),
            _ => BTreeSet::new(),
        };
        if !REMOTE_METHODS.contains(&method) { return value_error(format!("unknown remote method: {method}")); }
        if expected.is_empty() && !keys.is_empty() { return value_error(format!("{method} does not accept parameters")); }
        if !expected.is_empty() && keys != expected {
            return value_error(format!("{method} parameters must be exactly {}", expected.iter().copied().collect::<Vec<_>>().join(", ")));
        }
        let get = |key: &str| params.get(key).unwrap_or(&Json::Null);
        let result = (|| -> Result<Json> {
            Ok(match method {
                "open" => {
                    self.streamer.open()?;
                    server_log("OPEN", Some(client), "device_session=ready");
                    let token = self.ownership().token.clone().unwrap_or_default();
                    Json::object(vec![("cancel_token", Json::str(token)), ("board_protocol", Json::Int(LAYOUT_STRUCT_VERSION.into())),
                                      ("operation_budget", Json::Bool(true)), ("completion_notifications", Json::Bool(true))])
                }
                "describe" => {
                    let description = self.streamer.describe()?;
                    server_log("DESCRIBE", Some(client), &format!("ports={} lanes={} clock_hz={} layout=0x{:08X}", description.target.ports.len(),
                        description.target.raw_lanes.len(), description.clock_hz, description.layout_fingerprint()));
                    description_tree(&description)
                }
                "close" => {
                    let firing = self.streamer.snapshot().firing;
                    self.streamer.close()?;
                    server_log("CLOSE", Some(client), &format!("firing={} device_session=closed", if firing { "True" } else { "False" }));
                    Json::Null
                }
                "load" => {
                    let (Some(reuse_authored), Some(reuse_rows)) = (get("reuse_authored_source").as_bool(), get("reuse_rows").as_bool()) else {
                        return Err(Error::new(Kind::Type, "reuse_authored_source and reuse_rows must be booleans"));
                    };
                    let previous = if reuse_authored || reuse_rows {
                        Some(self.streamer.applied().ok_or_else(|| Error::new(Kind::Runtime, "there is no loaded application to reuse"))?)
                    } else { None };
                    let rows = if reuse_rows {
                        if !get("rows").is_null() { return value_error("reused scan rows must not also be supplied"); }
                        previous.as_ref().expect("previous").rows.clone()
                    } else { Arc::new(rows_from(get("rows"))?) };
                    let authored = if reuse_authored {
                        if !get("authored_source").is_null() { return value_error("a reused authored source must not also be supplied"); }
                        previous.as_ref().expect("previous").authored_source.clone()
                    } else if get("authored_source").is_null() { None } else { Some(Arc::new(sequence_from_remote(get("authored_source"))?)) };
                    let program = Arc::new(program_from_remote(get("program"))?);
                    let source = if get("source").is_null() { None } else { Some(Arc::new(sequence_from_remote(get("source"))?)) };
                    let periods = program.row_count();
                    self.streamer.load_program_exact(program.clone(), source, authored, rows.clone())?;
                    let resends = self.streamer.transport().resends();
                    server_log("LOAD", Some(client), &format!("periods={periods} loops={} slots={} duration_us={:.3} rows={}{}", program.loops.len(),
                        program.slot_count(), program.duration_seconds * 1e6, rows.len(), if resends > 0 { format!(" resent_frames={resends}") } else { String::new() }));
                    let applied = self.streamer.applied().ok_or_else(|| Error::new(Kind::Runtime, "LOAD left no application"))?;
                    Json::object(vec![("run_repeats", Json::Int(applied.run_repeats.into())), ("scan_repeats", Json::Int(applied.scan_repeats.into())),
                                      ("loaded_at", Json::Float(applied.loaded_at))])
                }
                "fire" => {
                    let (Some(run), Some(scan)) = (get("run_repeats").as_int(), get("scan_repeats").as_int()) else {
                        return Err(Error::new(Kind::Type, "fire run_repeats must be an integer"));
                    };
                    let run = u64::try_from(run).map_err(|_| Error::new(Kind::Value, "run_repeats must be in the hardware range [0, 2^32-1]"))?;
                    let scan = u64::try_from(scan).map_err(|_| Error::new(Kind::Value, "scan_repeats must be in the hardware range [0, 2^32-1]"))?;
                    let applied = self.streamer.fire_exact(run, scan)?;
                    let infinite = |count: u64| if count == 0 { "INFINITE".to_owned() } else { count.to_string() };
                    server_log("FIRE", Some(client), &format!("run_repeats={} scan_repeats={} periods={}", infinite(run), infinite(scan), applied.program.row_count()));
                    Json::object(vec![("command_id", Json::Int(self.streamer.fire_command_id().into()))])
                }
                "cursor" => self.streamer.cursor()?.map_or(Json::Null, |cursor| Json::Int(cursor.into())),
                "safe" => {
                    let readback = self.streamer.safe()?;
                    server_log("STOP/SAFE", Some(client), &format!("stable={} status={} command_id={}", readback.stable(), readback.status, readback.command_id));
                    safe_tree(&readback)
                }
                "snapshot" => self.streamer.snapshot().to_json(),
                _ => self.streamer.applied().as_ref().map_or(Json::Null, applied_tree),
            })
        })();
        if !self.owns(client, connection, Some(epoch)) { return Err(lost()); }
        result
    }

    fn owns_locked(&self, owner: &Ownership, client: &str, connection: u64) -> bool {
        owner.fault.is_none() && owner.client.as_deref() == Some(client) && owner.connection == Some(connection)
    }

    fn wait_owner_done(&self, params: &Json) -> Result<Json> {
        let (Some(token), Some(command_id), Some(timeout)) = (params.get("token").and_then(Json::as_str), params.get("command_id").and_then(Json::as_int),
                                                              params.get("timeout").and_then(Json::as_f64)) else {
            return value_error("wait_done requires token, positive command_id and timeout in [0, 1]");
        };
        if params.as_object().is_some_and(|items| items.len() != 3) || command_id <= 0 || !(0.0..=WAIT_SLICE.as_secs_f64()).contains(&timeout) {
            return value_error("wait_done requires token, positive command_id and timeout in [0, 1]");
        }
        let owner_name = {
            let owner = self.ownership();
            if owner.client.is_none() || owner.fault.is_some() || owner.token.as_deref() != Some(token) {
                return Err(Error::new(Kind::Runtime, "the wait token does not name the current owner"));
            }
            owner.client.clone()
        };
        let command_id = command_id as u32;
        let mut report = self.streamer.wait_done(Some(Duration::from_secs_f64(timeout)), Some(command_id))?;
        let current = !self.streamer.stop_requested() && command_id == self.streamer.fire_command_id() && Some(command_id) == self.streamer.command_id();
        let pending = self.streamer.is_firing() && current;
        if !current { report = None; }
        {
            let owner = self.ownership();
            if owner.fault.is_some() || owner.token.as_deref() != Some(token) {
                return Err(Error::new(Kind::Runtime, "the wait token no longer names the current owner"));
            }
        }
        if let Some(report) = &report {
            let event = if report.fault().is_empty() { "DONE" } else { "ERROR" };
            server_log(event, owner_name.as_deref(), &format!("status=0x{:02X} cursor={:?} underflow={} elapsed_ms={:.3} command_id={}",
                report.status, report.cursor, report.underflow, report.elapsed_seconds * 1e3, report.command_id));
        }
        Ok(Json::object(vec![("report", report.as_ref().map_or(Json::Null, report_tree)), ("pending", Json::Bool(pending))]))
    }

    fn cancel_owner_command(&self, params: &Json, client: &str) -> Result<Json> {
        let Some(token) = params.get("token").and_then(Json::as_str).filter(|_| params.as_object().is_some_and(|items| items.len() == 1)) else {
            return value_error("cancel parameters must be exactly token");
        };
        let owner_name = {
            let owner = self.ownership();
            if owner.client.is_none() || owner.fault.is_some() || owner.token.as_deref() != Some(token) {
                return Err(Error::new(Kind::Runtime, "the cancel token does not name the current owner"));
            }
            owner.client.clone()
        };
        server_log("CANCEL", owner_name.as_deref(), &format!("lane={client} action=\"SAFE beside the command lane\""));
        let readback = self.streamer.safe()?;
        server_log("CANCEL SAFE", owner_name.as_deref(), &format!("stable={} status={} command_id={}", readback.stable(), readback.status, readback.command_id));
        Ok(Json::Null)
    }

    /// Release the owner's outputs: SAFE beside the lane, then on it. `None`
    /// when this connection no longer owned the board.
    fn client_disconnected(&self, client: Option<(&str, u64)>, reason: &str) -> Option<bool> {
        let transition = {
            let mut owner = self.ownership();
            if let Some((name, connection)) = client {
                if owner.client.as_deref() != Some(name) || owner.connection != Some(connection) { return None; }
            }
            let to_drop = owner.connection.take();
            owner.client = None;
            owner.token = None;
            owner.epoch += 1;
            owner.fault = Some("disconnect SAFE has not completed".into());
            if let Some(old) = to_drop { self.drop_registered(old); }
            owner.epoch
        };
        server_log("AUTO-SAFE", client.map(|(name, _)| name), &format!("reason={reason:?}"));
        let _ = self.streamer.safe();
        let _lane = self.command_lane.lock().unwrap_or_else(|p| p.into_inner());
        let outcome = self.safe_on_lane();
        let mut owner = self.ownership();
        if owner.epoch != transition { return None; }
        match outcome {
            Ok(readback) => {
                owner.fault = None;
                drop(owner);
                if let Some(readback) = readback {
                    server_log("AUTO-SAFE DONE", client.map(|(name, _)| name), &format!("stable={} status={} command_id={}", readback.stable(),
                               readback.status, readback.command_id));
                }
                Some(true)
            }
            Err(error) => {
                let message = format!("disconnect SAFE failed: {}: {}", error.kind.python_name(), error.message.replace('\n', " "));
                owner.fault = Some(message.clone());
                drop(owner);
                server_log("AUTO-SAFE FAILED", client.map(|(name, _)| name), &format!("error={message:?}"));
                Some(false)
            }
        }
    }

    /// How long `method` may take on a slow link (JTAG-to-AXI): its own I/O plus,
    /// when this request takes the board from an owner or a fault, two SAFEs
    /// (v2 `request_budget`). An ordinary link gets the default timeout.
    fn request_budget(&self, method: &str, taking: bool) -> Duration {
        use crate::device::{LOAD_TIMEOUT, OBSERVER_STOP_TIMEOUT, SAFE_TIMEOUT};
        let Some(bounds) = self.streamer.transport().io_bounds() else { return DEFAULT_REQUEST_TIMEOUT };
        let safe = OBSERVER_STOP_TIMEOUT + bounds.action + SAFE_TIMEOUT;
        let mut budget = match method {
            "open" => bounds.startup + bounds.action,
            "load" => safe + bounds.action + LOAD_TIMEOUT,
            "fire" => 2 * bounds.action + LOAD_TIMEOUT,
            "safe" => safe,
            "cursor" => bounds.action,
            "close" => safe + bounds.close,
            _ => Duration::ZERO,
        };
        if taking { let owner = self.ownership(); if owner.client.is_some() || owner.fault.is_some() { budget += 2 * safe; } }
        budget.max(DEFAULT_REQUEST_TIMEOUT)
    }

    fn handle(self: Arc<Self>, mut stream: TcpStream, peer: SocketAddr, connection: u64) {
        let client = peer.to_string();
        let _ = stream.set_nodelay(true);
        let mut claimed = false;
        let mut reason = "client EOF".to_owned();
        loop {
            let request = match recv_frame(&mut stream, None) {
                Ok(Some(request)) => request,
                Ok(None) => break,
                Err(error) => { reason = if error.kind == Kind::Value { "unreadable request frame".into() } else { "client connection dropped".into() }; break; }
            };
            let id = request.get("id").cloned().unwrap_or(Json::Null);
            let method = request.get("method").and_then(Json::as_str).unwrap_or("<invalid>").to_owned();
            let outcome = (|| -> Result<Json> {
                let Json::Object(items) = &request else { return value_error("request must be an object") };
                let unknown: Vec<&str> = items.iter().map(|(k, _)| k.as_str()).filter(|k| !["id", "method", "params"].contains(k)).collect();
                if !unknown.is_empty() { return value_error(format!("unknown request field(s): {}", unknown.join(", "))); }
                let missing: Vec<&str> = ["id", "method", "params"].into_iter().filter(|k| request.get(k).is_none()).collect();
                if !missing.is_empty() { return value_error(format!("missing request field(s): {}", missing.join(", "))); }
                if !id.as_int().is_some_and(|id| id > 0) { return Err(Error::new(Kind::Type, "request id must be a positive integer")); }
                let Some(method) = request.get("method").and_then(Json::as_str) else { return value_error("request method must be text") };
                if !REMOTE_METHODS.contains(&method) && method != "cancel" { return value_error(format!("unknown remote method: {method}")); }
                let params = request.get("params").expect("checked");
                if !matches!(params, Json::Object(_)) { return value_error("request params must be an object"); }
                if method == "cancel" || method == "wait_done" {
                    if claimed { return value_error(format!("{method} travels on a connection of its own, beside the command lane")); }
                    return if method == "cancel" { self.cancel_owner_command(params, &client) } else { self.wait_owner_done(params) };
                }
                if method == "open" && (params.as_object().map_or(0, <[_]>::len) != 2 || params.get("board_protocol").and_then(Json::as_int) != Some(LAYOUT_STRUCT_VERSION.into())
                                        || params.get("operation_budget").and_then(Json::as_bool) != Some(true)) {
                    return value_error(format!("Pulse RPC open requires board_protocol={LAYOUT_STRUCT_VERSION} and operation_budget=true; update both Pulse \
                                                client and server software"));
                }
                // A slow link says how long this request may take before it starts.
                let budget = self.request_budget(method, !claimed);
                if budget > DEFAULT_REQUEST_TIMEOUT {
                    send_frame(&mut stream, &Json::object(vec![("id", id.clone()), ("budget_seconds", Json::Float(budget.as_secs_f64()))]), None)?;
                    server_log("RPC BUDGET", Some(&client), &format!("request_id={} method={method} seconds={}", id.compact_text(), budget.as_secs_f64()));
                }
                if !claimed {
                    self.claim(&client, connection)?;
                    claimed = true;
                    server_log("CLIENT CONNECTED", Some(&client), "status=OWNER");
                }
                self.dispatch(method, params, &client, connection)
            })();
            let response = match outcome {
                Ok(result) => Json::object(vec![("id", id.clone()), ("ok", Json::Bool(true)), ("result", result)]),
                Err(error) => {
                    server_log("RPC ERROR", Some(&client), &format!("request_id={} method={method} error_type={} error={:?}", id.compact_text(),
                                                                   error.kind.python_name(), error.message));
                    let (kind, message) = (error.kind.python_name(), error.message.clone());
                    Json::object(vec![("id", id.clone()), ("ok", Json::Bool(false)),
                                      ("error", Json::object(vec![("type", Json::str(kind)), ("message", Json::str(message))]))])
                }
            };
            if send_frame(&mut stream, &response, None).is_err() { reason = "client connection dropped".into(); break; }
            if !claimed { break; }
            if method == "close" { reason = "client EOF after close".into(); }
        }
        self.connections.lock().unwrap_or_else(|p| p.into_inner()).retain(|(id, _)| *id != connection);
        if claimed {
            let outputs = self.client_disconnected(Some((&client, connection)), &reason);
            let status = match outputs { Some(true) => "SAFE", Some(false) => "NOT_VERIFIED", None => "NO_ACTION" };
            server_log("CLIENT DISCONNECTED", Some(&client), &format!("outputs={status} reason={reason:?}"));
        }
    }
}

impl PulseRemoteServer {
    /// Bind the listening port first: a second server for this port is refused
    /// here, before anything could stop the running server's shot.
    pub fn bind(address: &str, streamer: PulseStreamer, peers: bool) -> Result<PulseRemoteServer> {
        let listener = TcpListener::bind(address).map_err(|error| Error::new(Kind::Io,
            format!("port {} is taken: another server on this machine already listens on it, or the system reserves it ({error})",
                    address.rsplit(':').next().unwrap_or(address))))?;
        let address = listener.local_addr()?;
        Ok(PulseRemoteServer { listener, address, thread: None, shared: Arc::new(ServerShared {
            streamer, ownership: Mutex::new(Ownership { epoch: 0, client: None, connection: None, token: None, fault: None }),
            command_lane: Mutex::new(()), peers: AtomicBool::new(peers), connections: Mutex::new(Vec::new()),
            next_connection: AtomicU64::new(1), closing: AtomicBool::new(false), served: AtomicBool::new(false),
        }) })
    }

    pub fn address(&self) -> SocketAddr { self.address }
    pub fn streamer(&self) -> &PulseStreamer { &self.shared.streamer }

    /// Whether clients on other machines are admitted; withdrawing drops theirs.
    pub fn admit_peers(&self, admitted: bool) {
        self.shared.peers.store(admitted, Ordering::SeqCst);
        if !admitted {
            for (_, stream) in self.shared.connections.lock().unwrap_or_else(|p| p.into_inner()).iter() {
                if stream.peer_addr().is_ok_and(|peer| !is_loopback(&peer)) { drop_connection(stream); }
            }
        }
        server_log(if admitted { "PEERS ADMITTED" } else { "PEERS REFUSED" }, None, &format!("endpoint={}", self.address));
    }

    /// Serve on a background thread until [`PulseRemoteServer::close`].
    pub fn start(&mut self) -> Result<()> {
        let listener = self.listener.try_clone()?;
        let shared = self.shared.clone();
        shared.served.store(true, Ordering::SeqCst);
        self.thread = Some(std::thread::Builder::new().name(format!("pulse-server:{}", self.address.port())).spawn(move || {
            for stream in listener.incoming() {
                if shared.closing.load(Ordering::SeqCst) { break; }
                let Ok(stream) = stream else { continue };
                let Ok(peer) = stream.peer_addr() else { continue };
                if !shared.peers.load(Ordering::SeqCst) && !is_loopback(&peer) {
                    server_log("PEER REFUSED", Some(&peer.to_string()), "reason=\"not published; a peer is admitted once the device is on the bench fabric\"");
                    drop_connection(&stream);
                    continue;
                }
                let connection = shared.next_connection.fetch_add(1, Ordering::SeqCst);
                if let Ok(copy) = stream.try_clone() { shared.connections.lock().unwrap_or_else(|p| p.into_inner()).push((connection, copy)); }
                let worker = shared.clone();
                let _ = std::thread::Builder::new().name(format!("pulse-client:{peer}")).spawn(move || worker.handle(stream, peer, connection));
            }
        }).map_err(|error| Error::new(Kind::Runtime, error.to_string()))?);
        server_log("RPC LISTENING", None, &format!("endpoint={}", self.address));
        Ok(())
    }

    /// Stop listening, drop every client and drive the board SAFE.
    pub fn close(&mut self) {
        self.shared.closing.store(true, Ordering::SeqCst);
        if self.shared.served.load(Ordering::SeqCst) { self.shared.client_disconnected(None, "server shutdown"); }
        for (_, stream) in self.shared.connections.lock().unwrap_or_else(|p| p.into_inner()).iter() { drop_connection(stream); }
        let _ = TcpStream::connect_timeout(&SocketAddr::from(([127, 0, 0, 1], self.address.port())), Duration::from_millis(200));
        if let Some(thread) = self.thread.take() { let _ = thread.join(); }
    }
}

impl Drop for PulseRemoteServer {
    fn drop(&mut self) { if self.thread.is_some() { self.close(); } }
}

// ------------------------------------------------------------------ client
struct ClientLane { stream: Option<TcpStream>, request_id: u64, cancel_token: Option<String>, fire_command_id: Option<u32>,
                    description: Option<BoardDescription>, loaded: Option<AppliedState> }

/// The device surface of a board served by another process, plus its TCP
/// connection lifecycle. The command lane is one connection; Stop never queues
/// behind a pending LOAD/FIRE (it goes to the server's cancel lane first).
pub struct RemotePulseStreamer {
    pub host: String,
    pub port: u16,
    automatic_timeout: bool,
    request_timeout: Duration,
    connect_timeout: Duration,
    lane: Mutex<ClientLane>,
    stopping: Mutex<HashSet<ThreadId>>,
    cancel_token: Mutex<Option<String>>,
    config: ConfigHolder,
}

impl RemotePulseStreamer {
    pub fn new(host: &str, port: u16, request_timeout: Option<Duration>, connect_timeout: Duration) -> Result<RemotePulseStreamer> {
        if host.is_empty() { return value_error("remote host is required"); }
        if port == 0 { return value_error("remote port is outside the TCP range"); }
        Ok(RemotePulseStreamer { host: host.to_owned(), port, automatic_timeout: request_timeout.is_none(),
            request_timeout: request_timeout.unwrap_or(DEFAULT_REQUEST_TIMEOUT), connect_timeout,
            lane: Mutex::new(ClientLane { stream: None, request_id: 0, cancel_token: None, fire_command_id: None, description: None, loaded: None }),
            stopping: Mutex::new(HashSet::new()), cancel_token: Mutex::new(None), config: ConfigHolder::default() })
    }

    fn lane(&self) -> MutexGuard<'_, ClientLane> { self.lane.lock().unwrap_or_else(|p| p.into_inner()) }
    pub fn config(&self) -> &ConfigHolder { &self.config }

    fn connect(&self) -> Result<TcpStream> {
        let addresses: Vec<SocketAddr> = (self.host.as_str(), self.port).to_socket_addrs()
            .map_err(|error| Error::new(Kind::Connection, format!("could not reach a pulse server at {}:{}: {error}", self.host, self.port)))?.collect();
        let mut last = None;
        for address in addresses {
            match TcpStream::connect_timeout(&address, self.connect_timeout) {
                Ok(stream) => { let _ = stream.set_nodelay(true); return Ok(stream); }
                Err(error) => last = Some(error),
            }
        }
        let reason = match last.as_ref().map(std::io::Error::kind) {
            Some(std::io::ErrorKind::ConnectionRefused) => "the machine is up but nothing is listening on that port",
            _ => "the connection itself was never answered, which is what a firewall does to an inbound port -- on the server machine, \
                  allow the pulse server on the private network",
        };
        Err(Error::new(Kind::Connection, format!("could not reach a pulse server at {}:{} within {}s: {reason}", self.host, self.port,
                                                 self.connect_timeout.as_secs_f64())))
    }

    fn disconnect_locked(&self, lane: &mut ClientLane) {
        if let Some(stream) = lane.stream.take() { drop_connection(&stream); }
        lane.cancel_token = None;
        *self.cancel_token.lock().unwrap_or_else(|p| p.into_inner()) = None;
        lane.fire_command_id = None;
        lane.description = None;
        lane.loaded = None;
    }

    fn call_locked(&self, lane: &mut ClientLane, method: &str, params: Json) -> Result<Json> {
        let Some(stream) = lane.stream.as_mut() else { return Err(Error::new(Kind::Connection, "remote PulseStreamer is not open")) };
        lane.request_id += 1;
        let id = lane.request_id;
        let request = Json::object(vec![("id", Json::Int(id.into())), ("method", Json::str(method)), ("params", params)]);
        let started = Instant::now();
        let mut deadline = started + self.request_timeout;
        let exchanged = (|| -> Result<Option<Json>> {
            send_frame(stream, &request, Some(deadline))?;
            let mut response = recv_frame(stream, Some(deadline))?;
            if let Some(budget) = response.as_ref().and_then(|value| value.get("budget_seconds")).and_then(Json::as_f64) {
                if self.automatic_timeout { deadline = deadline.max(Instant::now() + Duration::from_secs_f64(budget) + DEFAULT_REQUEST_TIMEOUT); }
                response = recv_frame(stream, Some(deadline))?;
            }
            Ok(response)
        })();
        let response = match exchanged {
            Ok(Some(response)) => response,
            Ok(None) => { self.disconnect_locked(lane); return Err(Error::new(Kind::Connection, CONNECTION_ENDED)); }
            Err(error) => {
                self.disconnect_locked(lane);
                return Err(match error.kind {
                    Kind::Timeout => Error::new(Kind::Connection, format!("the pulse server did not answer within {:.3}s", started.elapsed().as_secs_f64())),
                    Kind::Value => Error::new(Kind::Connection, format!("the pulse server's reply could not be read: {}", error.message)),
                    _ => Error::new(Kind::Connection, format!("{CONNECTION_ENDED} ({})", error.message)),
                });
            }
        };
        let protocol = |message: &str| Error::new(Kind::Connection, message.to_owned());
        if response.get("id").and_then(Json::as_int) != Some(id.into()) { self.disconnect_locked(lane); return Err(protocol("remote response id differs from the request")); }
        match response.get("ok").and_then(Json::as_bool) {
            Some(true) => Ok(response.get("result").cloned().unwrap_or(Json::Null)),
            Some(false) => {
                let error = response.get("error").cloned().unwrap_or(Json::Null);
                match (error.get("type").and_then(Json::as_str), error.get("message").and_then(Json::as_str)) {
                    (Some(kind), Some(message)) => Err(remote_error(kind, message)),
                    _ => { self.disconnect_locked(lane); Err(protocol("remote error response differs from the protocol")) }
                }
            }
            None => { self.disconnect_locked(lane); Err(protocol("remote response ok field is not boolean")) }
        }
    }

    pub fn open(&self) -> Result<()> {
        let mut lane = self.lane();
        if lane.stream.is_none() { lane.stream = Some(self.connect()?); }
        let answer = self.call_locked(&mut lane, "open", Json::object(vec![("board_protocol", Json::Int(LAYOUT_STRUCT_VERSION.into())),
                                                                          ("operation_budget", Json::Bool(true))]));
        let checked = answer.and_then(|answer| {
            let token = answer.get("cancel_token").and_then(Json::as_str).filter(|token| !token.is_empty()).map(str::to_owned);
            let exact = answer.as_object().is_some_and(|items| items.len() == 4)
                && answer.get("board_protocol").and_then(Json::as_int) == Some(LAYOUT_STRUCT_VERSION.into())
                && answer.get("operation_budget").and_then(Json::as_bool) == Some(true)
                && answer.get("completion_notifications").and_then(Json::as_bool) == Some(true);
            match token.filter(|_| exact) {
                Some(token) => Ok(token),
                None => Err(Error::new(Kind::Connection, "Pulse server RPC/board contract differs; update both Pulse client and server software and \
                                                         restart the server before connecting")),
            }
        });
        match checked {
            Ok(token) => { lane.cancel_token = Some(token.clone()); *self.cancel_token.lock().unwrap_or_else(|p| p.into_inner()) = Some(token); Ok(()) }
            Err(error) => { self.disconnect_locked(&mut lane); Err(error) }
        }
    }

    pub fn describe(&self) -> Result<BoardDescription> {
        let mut lane = self.lane();
        if let Some(description) = &lane.description { return Ok(description.clone()); }
        let description = description_from_remote(&self.call_locked(&mut lane, "describe", Json::Object(vec![]))?)?;
        lane.description = Some(description.clone());
        Ok(description)
    }

    pub fn close(&self) -> Result<()> {
        let mut lane = self.lane();
        if lane.stream.is_none() { return Ok(()); }
        let result = self.call_locked(&mut lane, "close", Json::Object(vec![]));
        self.disconnect_locked(&mut lane);
        match result { Err(error) if error.message.starts_with(CONNECTION_ENDED) => Ok(()), other => other.map(|_| ()) }
    }

    /// Drop the TCP connection without asking the server to close the board.
    pub fn disconnect(&self) { let mut lane = self.lane(); self.disconnect_locked(&mut lane); }

    pub fn is_connected(&self) -> bool { self.lane().stream.is_some() }

    pub fn load(&self, program: Arc<CompiledProgram>, source: Option<Arc<Sequence>>, rows: Vec<Vec<u64>>) -> Result<()> {
        let mut lane = self.lane();
        let geometry = self.describe_locked(&mut lane)?.geometry;
        let (program, filled) = self.config.prepare(&program, source.as_ref(), source.as_ref(), &geometry)?;
        self.load_locked(&mut lane, program, filled, source, Arc::new(rows))
    }

    fn describe_locked(&self, lane: &mut ClientLane) -> Result<BoardDescription> {
        if let Some(description) = &lane.description { return Ok(description.clone()); }
        let description = description_from_remote(&self.call_locked(lane, "describe", Json::Object(vec![]))?)?;
        lane.description = Some(description.clone());
        Ok(description)
    }

    fn load_locked(&self, lane: &mut ClientLane, program: Arc<CompiledProgram>, source: Option<Arc<Sequence>>, authored: Option<Arc<Sequence>>,
                   rows: Arc<Vec<Vec<u64>>>) -> Result<()> {
        let previous = lane.loaded.take();
        let reuse_authored = authored.is_some() && previous.as_ref().is_some_and(|previous| previous.authored_source.as_deref() == authored.as_deref());
        let reuse_rows = previous.as_ref().is_some_and(|previous| Arc::ptr_eq(&previous.rows, &rows));
        let receipt = self.call_locked(lane, "load", Json::object(vec![
            ("program", program_tree(&program)), ("source", source.as_deref().map_or(Json::Null, sequence_tree)),
            ("authored_source", if reuse_authored { Json::Null } else { authored.as_deref().map_or(Json::Null, sequence_tree) }),
            ("reuse_authored_source", Json::Bool(reuse_authored)), ("rows", if reuse_rows { Json::Null } else { rows_tree(&rows) }),
            ("reuse_rows", Json::Bool(reuse_rows)),
        ]))?;
        let (Some(run), Some(scan), Some(loaded_at)) = (receipt.get("run_repeats").and_then(Json::as_int), receipt.get("scan_repeats").and_then(Json::as_int),
                                                        receipt.get("loaded_at").and_then(Json::as_f64)) else {
            return Err(Error::new(Kind::Runtime, "Pulse server did not return the load receipt; update the machine that serves the board and restart its pulse server"));
        };
        lane.loaded = Some(AppliedState::new(program, source, authored, rows, run as u64, scan as u64, loaded_at)?);
        lane.fire_command_id = None;
        Ok(())
    }

    pub fn fire(&self, run_repeats: u64, scan_repeats: u64) -> Result<AppliedState> {
        let mut lane = self.lane();
        if lane.loaded.is_none() {
            let applied = self.call_locked(&mut lane, "applied", Json::Object(vec![]))?;
            lane.loaded = if applied.is_null() { None } else { Some(applied_from_remote(&applied)?) };
        }
        if let Some(applied) = lane.loaded.clone() {
            let geometry = self.describe_locked(&mut lane)?.geometry;
            let (program, source) = self.config.prepare(&applied.program, applied.authored_source.as_ref(), applied.source.as_ref(), &geometry)?;
            let same = match (&source, &applied.source) { (Some(a), Some(b)) => Arc::ptr_eq(a, b), (None, None) => true, _ => false };
            if !same { self.load_locked(&mut lane, program, source, applied.authored_source.clone(), applied.rows.clone())?; }
        }
        lane.fire_command_id = None;
        if !self.stopping.lock().unwrap_or_else(|p| p.into_inner()).is_empty() {
            return Err(Error::new(Kind::Runtime, "a Stop arrived while this FIRE was being prepared; it was not sent"));
        }
        let receipt = match self.call_locked(&mut lane, "fire", Json::object(vec![("run_repeats", Json::Int(run_repeats.into())),
                                                                               ("scan_repeats", Json::Int(scan_repeats.into()))])) {
            Ok(receipt) => receipt,
            Err(error) => { lane.loaded = None; return Err(error); }
        };
        let Some(command_id) = receipt.get("command_id").and_then(Json::as_int).filter(|id| *id > 0)
            .filter(|_| receipt.as_object().is_some_and(|items| items.len() == 1)) else {
            return Err(Error::new(Kind::Connection, "Pulse server did not return the FIRE command identity"));
        };
        lane.fire_command_id = Some(command_id as u32);
        let applied = lane.loaded.as_ref().ok_or_else(|| Error::new(Kind::Runtime, "no application is loaded"))?.with_repeats(run_repeats, scan_repeats)?;
        lane.loaded = Some(applied.clone());
        Ok(applied)
    }

    /// Await this FIRE's completion on connections of its own, in slices.
    pub fn wait_done(&self, timeout: Option<Duration>, command_id: Option<u32>) -> Result<Option<DoneReport>> {
        let deadline = timeout.map(|timeout| Instant::now() + timeout);
        let (token, command_id) = {
            let lane = self.lane();
            if lane.stream.is_none() { return Err(Error::new(Kind::Connection, "remote PulseStreamer is not open")); }
            if command_id.is_some() && command_id != lane.fire_command_id { return Ok(None); }
            (lane.cancel_token.clone(), lane.fire_command_id)
        };
        let Some(command_id) = command_id else { return Ok(None) };
        loop {
            let mut slice = WAIT_SLICE.min(self.request_timeout / 2);
            if let Some(deadline) = deadline { slice = slice.min(deadline.saturating_duration_since(Instant::now())); }
            let mut lane = self.connect().map_err(|error| Error::new(Kind::Connection, format!("{CONNECTION_ENDED} ({})", error.message)))?;
            let request = Json::object(vec![("id", Json::Int(1)), ("method", Json::str("wait_done")), ("params", Json::object(vec![
                ("token", Json::opt_str(token.as_deref())), ("command_id", Json::Int(command_id.into())), ("timeout", Json::Float(slice.as_secs_f64())),
            ]))]);
            let answer = send_frame(&mut lane, &request, Some(Instant::now() + self.request_timeout))
                .and_then(|()| recv_frame(&mut lane, Some(Instant::now() + self.request_timeout)));
            drop_connection(&lane);
            let answer = match answer {
                Ok(Some(answer)) => answer,
                Ok(None) => return Err(Error::new(Kind::Connection, CONNECTION_ENDED)),
                Err(error) if error.kind == Kind::Timeout => return Err(Error::new(Kind::Connection, "the pulse server did not answer the completion wait in time")),
                Err(error) => return Err(Error::new(Kind::Connection, format!("{CONNECTION_ENDED} ({})", error.message))),
            };
            match answer.get("ok").and_then(Json::as_bool) {
                Some(true) => {}
                Some(false) => {
                    let error = answer.get("error").cloned().unwrap_or(Json::Null);
                    return Err(remote_error(error.get("type").and_then(Json::as_str).unwrap_or("RuntimeError"),
                                            error.get("message").and_then(Json::as_str).unwrap_or("")));
                }
                None => return Err(Error::new(Kind::Connection, "invalid Pulse completion response")),
            }
            let result = answer.get("result").cloned().unwrap_or(Json::Null);
            let report = match result.get("report") { None | Some(Json::Null) => None, Some(report) => Some(report_from_remote(report)?) };
            let Some(pending) = result.get("pending").and_then(Json::as_bool) else { return Err(Error::new(Kind::Connection, "invalid Pulse completion result")) };
            if report.as_ref().is_some_and(|report| report.command_id != command_id) {
                return Err(Error::new(Kind::Connection, "Pulse completion belongs to another FIRE"));
            }
            {
                let lane = self.lane();
                if lane.cancel_token != token || lane.fire_command_id != Some(command_id) { return Ok(None); }
            }
            if report.is_some() || !pending { return Ok(report); }
            if deadline.is_some_and(|deadline| Instant::now() >= deadline) { return Ok(None); }
        }
    }

    /// The command id of the FIRE this client sent last, while it can still be awaited.
    pub fn fire_command_id(&self) -> Option<u32> { self.lane().fire_command_id }

    pub fn cursor(&self) -> Result<Option<u32>> {
        let mut lane = self.lane();
        let value = self.call_locked(&mut lane, "cursor", Json::Object(vec![]))?;
        Ok(value.as_int().map(|cursor| cursor as u32))
    }

    /// Stop now. When another thread holds the command lane, the intent goes to
    /// the server's cancel lane first; the readback comes from the lane after.
    pub fn safe(&self) -> Result<SafeReadback> {
        if let Ok(mut lane) = self.lane.try_lock() {
            lane.fire_command_id = None;
            return safe_from_remote(&self.call_locked(&mut lane, "safe", Json::Object(vec![]))?);
        }
        let caller = std::thread::current().id();
        self.stopping.lock().unwrap_or_else(|p| p.into_inner()).insert(caller);
        self.cancel_pending();
        let result = (|| {
            let mut lane = self.lane();
            self.stopping.lock().unwrap_or_else(|p| p.into_inner()).remove(&caller);
            lane.fire_command_id = None;
            safe_from_remote(&self.call_locked(&mut lane, "safe", Json::Object(vec![]))?)
        })();
        self.stopping.lock().unwrap_or_else(|p| p.into_inner()).remove(&caller);
        result
    }

    fn cancel_pending(&self) {
        let Some(token) = self.cancel_token.lock().unwrap_or_else(|p| p.into_inner()).clone() else { return };
        let Ok(mut lane) = self.connect() else { return };
        let request = Json::object(vec![("id", Json::Int(1)), ("method", Json::str("cancel")), ("params", Json::object(vec![("token", Json::str(token))]))]);
        let _ = send_frame(&mut lane, &request, Some(Instant::now() + self.request_timeout))
            .and_then(|()| recv_frame(&mut lane, Some(Instant::now() + self.request_timeout)));
        drop_connection(&lane);
    }

    pub fn snapshot(&self) -> Result<Json> {
        let mut lane = self.lane();
        let mut state = self.call_locked(&mut lane, "snapshot", Json::Object(vec![]))?;
        if let Json::Object(items) = &mut state {
            items.retain(|(key, _)| key != "config_source");
            items.push(("config_source".into(), Json::str(self.config.source())));
        }
        Ok(state)
    }

    pub fn applied(&self) -> Result<Option<AppliedState>> {
        let mut lane = self.lane();
        let applied = self.call_locked(&mut lane, "applied", Json::Object(vec![]))?;
        if applied.is_null() { Ok(None) } else { applied_from_remote(&applied).map(Some) }
    }
}

impl Drop for RemotePulseStreamer {
    fn drop(&mut self) { let mut lane = self.lane(); self.disconnect_locked(&mut lane); }
}
