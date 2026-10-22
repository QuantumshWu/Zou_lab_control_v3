//! The JTAG-to-AXI register transport: the board's register map driven through
//! one resident Vivado Tcl interpreter (`vivado -mode tcl`), as v2's
//! `transport/axi.py` drives it.
//!
//! Vivado is started once (`start`) and opens the one hardware target, device
//! and JTAG-to-AXI core; it never selects or programs a bitstream (the
//! deployment is frozen). Every action is one Tcl block wrapped in markers
//! (`ZLC_AXI_<n>_BEGIN` .. `_OK|_ERROR` .. `_END`), so its reply is read up to
//! its own end marker. Contiguous writes are coalesced into INCR bursts (at
//! most 256 words, never across a 4 KiB boundary, never reordered); a command
//! writes its identity and code and polls the acknowledgement inside the same
//! block. A Stop abandons a read without closing the session: the next action
//! first drains the abandoned reply, so the SAFE behind a Stop still travels.
//! A deadline that passes, or an interpreter that exits, ends the session.
//!
//! [`TclLauncher`] starts the interpreter; [`VivadoLauncher`] is the real one
//! and [`FakeVivado`] the twin the tests use (a memory board behind a Tcl
//! session that understands exactly the blocks this transport writes).
use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, mpsc};
use std::time::{Duration, Instant};

use crate::error::{Error, Kind, Result};
use crate::transport::{Event, IoBounds, MemoryRegisterTransport, RegisterTransport, aborted};
use crate::wire::ctrl;

/// An INCR burst never crosses this byte boundary (AXI's 4 KiB rule).
pub const AXI_BURST_BOUNDARY_BYTES: u32 = 4096;
/// The longest burst the JTAG-to-AXI master does: the hardware's own limit.
pub const BURST_MAX: usize = 256;
/// Bursts per Tcl block: one round trip, traded against what is in flight when cancelled.
pub const WRITE_BATCH: usize = 200;
/// The observer's period on this link (a JTAG read takes tens of milliseconds).
pub const JTAG_AXI_OBSERVER_INTERVAL: Duration = Duration::from_millis(50);
/// How long Vivado may take to come up and open the hardware target.
pub const STARTUP_TIMEOUT: Duration = Duration::from_secs(180);
/// How long any one action may take once it is up.
pub const ACTION_TIMEOUT: Duration = Duration::from_secs(120);
pub const PROCESS_EXIT_TIMEOUT: Duration = Duration::from_secs(5);
pub const PROCESS_KILL_TIMEOUT: Duration = Duration::from_secs(3);
pub const READER_EXIT_TIMEOUT: Duration = Duration::from_secs(5);
/// One wait for an output line; Stop and the deadline are looked at between them.
const READ_SLICE: Duration = Duration::from_millis(200);

// ------------------------------------------------------------------ the interpreter
/// One resident Tcl interpreter: text in, output lines out.
pub trait TclSession: Send {
    fn write(&mut self, text: &str) -> Result<()>;
    /// The next output line, or `None` when none came within `timeout`; an
    /// error once the interpreter has ended.
    fn read_line(&mut self, timeout: Duration) -> Result<Option<String>>;
    /// End it: politely (`exit`, then a wait) or at once.
    fn end(&mut self, graceful: bool);
}

pub trait TclLauncher: Send + Sync {
    fn launch(&self, state_dir: &Path) -> Result<Box<dyn TclSession>>;
    /// What the interpreter is, for logs (`C:/AMD/Vivado/2024.2/bin/vivado.bat`).
    fn describe(&self) -> String;
}

/// The Vivado this machine builds with: `ZLC_PS_VIVADO_BIN`, else the newest
/// release under `{C,D}:/{Xilinx,AMD}/Vivado/*/bin/vivado.bat` (release folders
/// are named YYYY.N, so names order as versions), else `vivado.bat` on PATH.
pub fn default_vivado() -> PathBuf {
    if let Some(configured) = std::env::var_os("ZLC_PS_VIVADO_BIN").filter(|value| !value.is_empty()) { return configured.into(); }
    let mut newest: Option<(String, PathBuf)> = None;
    for drive in ["C", "D"] {
        for vendor in ["Xilinx", "AMD"] {
            let Ok(releases) = std::fs::read_dir(format!("{drive}:/{vendor}/Vivado")) else { continue };
            for release in releases.flatten() {
                let path = release.path().join("bin").join("vivado.bat");
                let name = release.file_name().to_string_lossy().into_owned();
                if path.is_file() && newest.as_ref().is_none_or(|(best, _)| name > *best) { newest = Some((name, path)); }
            }
        }
    }
    newest.map(|(_, path)| path).unwrap_or_else(|| "vivado.bat".into())
}

/// The debug-probe file a deployment names (`ZLC_PS_VIVADO_LTX`, `ZLC_PS_LTX`,
/// or the implementation run's `.ltx` under `ZLC_PS_PROJECT_DIR`).
pub fn default_probes() -> Option<PathBuf> {
    for name in ["ZLC_PS_VIVADO_LTX", "ZLC_PS_LTX"] {
        if let Some(value) = std::env::var_os(name).filter(|value| !value.is_empty()) { return Some(value.into()); }
    }
    let candidate = PathBuf::from(std::env::var_os("ZLC_PS_PROJECT_DIR")?).join("ps.runs/impl_1/zlc_pulse_streamer_top.ltx");
    candidate.is_file().then_some(candidate)
}

/// `vivado -mode tcl -nolog -nojournal`, started in the state directory, its
/// stdout and stderr read line by line into one queue. It is placed in a job
/// that ends it with this process, so a crash never leaves a Vivado holding
/// the cable.
pub struct VivadoLauncher { pub vivado: PathBuf }

impl VivadoLauncher {
    pub fn from_env() -> VivadoLauncher { VivadoLauncher { vivado: default_vivado() } }
}

struct VivadoProcess { child: std::process::Child, stdin: Option<std::process::ChildStdin>, lines: mpsc::Receiver<Option<String>>,
                       _job: Option<KillJob>, ended: bool }

impl TclLauncher for VivadoLauncher {
    fn describe(&self) -> String { self.vivado.display().to_string() }

    fn launch(&self, state_dir: &Path) -> Result<Box<dyn TclSession>> {
        use std::os::windows::process::CommandExt;
        let mut child = std::process::Command::new(&self.vivado).args(["-mode", "tcl", "-nolog", "-nojournal"]).current_dir(state_dir)
            .stdin(std::process::Stdio::piped()).stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::piped())
            .creation_flags(0x0800_0000) // CREATE_NO_WINDOW
            .spawn().map_err(|error| {
                let note = if error.kind() == std::io::ErrorKind::NotFound { format!("Vivado executable was not found: {:?}", self.vivado.display().to_string()) }
                           else { format!("Vivado could not start ({}): {error}", self.vivado.display()) };
                record_diagnostic(state_dir, "start_failure", &note);
                Error::new(Kind::Runtime, format!("persistent Vivado transport could not start: {note}"))
            })?;
        let job = KillJob::new().and_then(|job| job.assign(&child).map(|()| job)).ok();
        let (sender, lines) = mpsc::channel();
        for stream in [child.stdout.take().map(|out| Box::new(out) as Box<dyn std::io::Read + Send>),
                       child.stderr.take().map(|err| Box::new(err) as Box<dyn std::io::Read + Send>)].into_iter().flatten() {
            let sender = sender.clone();
            let _ = std::thread::Builder::new().name("zlc-vivado-axi-reader".into()).spawn(move || {
                for line in BufReader::new(stream).lines() {
                    let Ok(line) = line else { break };
                    if sender.send(Some(line)).is_err() { return; }
                }
                let _ = sender.send(None);
            });
        }
        let stdin = child.stdin.take();
        Ok(Box::new(VivadoProcess { child, stdin, lines, _job: job, ended: false }))
    }
}

impl TclSession for VivadoProcess {
    fn write(&mut self, text: &str) -> Result<()> {
        let stdin = self.stdin.as_mut().ok_or_else(|| Error::new(Kind::Connection, "Vivado's input is closed"))?;
        stdin.write_all(text.as_bytes()).and_then(|()| stdin.flush()).map_err(Error::from)
    }

    fn read_line(&mut self, timeout: Duration) -> Result<Option<String>> {
        if self.ended { return Err(Error::new(Kind::Connection, "Vivado has exited")); }
        match self.lines.recv_timeout(timeout) {
            Ok(Some(line)) => Ok(Some(line)),
            Err(mpsc::RecvTimeoutError::Timeout) => Ok(None),
            // stdout and stderr each end with None; either one means the process is going.
            Ok(None) | Err(mpsc::RecvTimeoutError::Disconnected) => { self.ended = true; Err(Error::new(Kind::Connection, "Vivado has exited")) }
        }
    }

    fn end(&mut self, graceful: bool) {
        if graceful {
            if let Some(stdin) = self.stdin.as_mut() {
                let _ = stdin.write_all(b"catch {close_hw_target}\ncatch {disconnect_hw_server}\nexit\n").and_then(|()| stdin.flush());
            }
        } else { let _ = self.child.kill(); }
        self.stdin = None;
        let deadline = Instant::now() + if graceful { PROCESS_EXIT_TIMEOUT } else { PROCESS_KILL_TIMEOUT };
        loop {
            if let Ok(Some(_)) = self.child.try_wait() { break; }
            if Instant::now() >= deadline { let _ = self.child.kill(); let _ = self.child.wait(); break; }
            std::thread::sleep(Duration::from_millis(20));
        }
        self.ended = true;
    }
}

impl Drop for VivadoProcess {
    fn drop(&mut self) { if !self.ended { self.end(false); } }
}

/// A job object that ends its processes when this process ends.
struct KillJob(usize);

impl KillJob {
    fn new() -> Result<KillJob> {
        use windows_sys::Win32::System::JobObjects::{CreateJobObjectW, JobObjectExtendedLimitInformation, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, SetInformationJobObject};
        let handle = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
        if handle.is_null() { return Err(Error::new(Kind::Io, std::io::Error::last_os_error().to_string())); }
        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { std::mem::zeroed() };
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        unsafe { SetInformationJobObject(handle, JobObjectExtendedLimitInformation, (&info as *const JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast(),
                                         size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32) };
        Ok(KillJob(handle as usize))
    }
    fn assign(&self, child: &std::process::Child) -> Result<()> {
        use std::os::windows::io::AsRawHandle;
        if unsafe { windows_sys::Win32::System::JobObjects::AssignProcessToJobObject(self.0 as _, child.as_raw_handle() as _) } == 0 {
            return Err(Error::new(Kind::Io, std::io::Error::last_os_error().to_string()));
        }
        Ok(())
    }
}

impl Drop for KillJob {
    fn drop(&mut self) { unsafe { windows_sys::Win32::Foundation::CloseHandle(self.0 as _) }; }
}

fn record_diagnostic(state_dir: &Path, name: &str, text: &str) { let _ = std::fs::write(state_dir.join(format!("{name}.log")), text); }

// ------------------------------------------------------------------ the Tcl it writes
fn byte_address(word: u32) -> Result<u32> {
    word.checked_mul(4).ok_or_else(|| Error::new(Kind::Value, "AXI word address is outside the 32-bit byte-address range"))
}

/// Insertion-order contiguous writes as bursts; nothing is reordered.
pub fn burst_runs(pending: &[(u32, u32)]) -> Vec<(u32, Vec<u32>)> {
    let mut runs = Vec::new();
    let mut index = 0;
    while index < pending.len() {
        let (base, value) = pending[index];
        let mut values = vec![value];
        let mut cursor = index + 1;
        while cursor < pending.len() && values.len() < BURST_MAX {
            let next = u64::from(base) + 4 * values.len() as u64;
            if u64::from(pending[cursor].0) != next || next / u64::from(AXI_BURST_BOUNDARY_BYTES) != u64::from(base / AXI_BURST_BOUNDARY_BYTES) { break; }
            values.push(pending[cursor].1);
            cursor += 1;
        }
        runs.push((base, values));
        index = cursor;
    }
    runs
}

/// One INCR write burst; DATA is the words most significant (last) first.
pub fn write_burst_tcl(byte_address: u32, values: &[u32]) -> Vec<String> {
    let data: String = values.iter().rev().map(|value| format!("{value:08X}")).collect();
    vec![format!("create_hw_axi_txn zlc_w [get_hw_axis] -address {byte_address:08X} -data {data} -len {} -type write -burst INCR -force", values.len()),
         "run_hw_axi zlc_w".into(), "delete_hw_axi_txn zlc_w".into()]
}

pub fn read_txn_tcl(byte_address: u32, marker: &str, count: u32) -> Vec<String> {
    vec![format!("create_hw_axi_txn zlc_r [get_hw_axis] -address {byte_address:08X} -len {count} -type read -force"), "run_hw_axi zlc_r".into(),
         format!("puts \"{marker} [get_property DATA [get_hw_axi_txns zlc_r]]\""), "delete_hw_axi_txn zlc_r".into()]
}

/// The words of the last `marker` line: DATA is one hex number, word 0 in its low 32 bits.
pub fn parse_read(output: &str, marker: &str, count: u32) -> Result<Vec<u32>> {
    let token = output.lines().filter_map(|line| line.split_once(marker)).last()
        .and_then(|(_, rest)| rest.split_whitespace().last().map(str::to_owned)).unwrap_or_default();
    if token.is_empty() { return Err(Error::new(Kind::Runtime, "hw_axi read returned no DATA")); }
    let digits = token.strip_prefix("0x").or_else(|| token.strip_prefix("0X")).unwrap_or(&token);
    if digits.is_empty() || digits.len() > 8 * count as usize || !digits.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(Error::new(Kind::Runtime, format!("hw_axi read returned non-binary DATA {token:?}")));
    }
    let padded = format!("{digits:0>width$}", width = 8 * count as usize);
    Ok((0..count as usize).map(|index| {
        let end = padded.len() - 8 * index;
        u32::from_str_radix(&padded[end - 8..end], 16).expect("hex digits")
    }).collect())
}

/// The block around an action: private transaction names cleared before and
/// after (a failed run_hw_axi skips its own delete, and one Xicom fault must not
/// poison the SAFE after it), the body's error caught and printed.
pub fn wrap_tcl(lines: &[String], marker: &str) -> String {
    format!("puts \"{marker}_BEGIN\"\n\
             catch {{delete_hw_axi_txn -quiet [get_hw_axi_txns -quiet zlc_w]}}\n\
             catch {{delete_hw_axi_txn -quiet [get_hw_axi_txns -quiet zlc_r]}}\n\
             set zlc_axi_code [catch {{\n{}\n}} zlc_axi_result zlc_axi_options]\n\
             catch {{delete_hw_axi_txn -quiet [get_hw_axi_txns -quiet zlc_w]}}\n\
             catch {{delete_hw_axi_txn -quiet [get_hw_axi_txns -quiet zlc_r]}}\n\
             if {{$zlc_axi_code}} {{\n    puts \"{marker}_ERROR $zlc_axi_result\"\n    \
             if {{[dict exists $zlc_axi_options -errorinfo]}} {{ puts [dict get $zlc_axi_options -errorinfo] }}\n\
             }} else {{\n    puts \"{marker}_OK\"\n}}\n\
             puts \"{marker}_END\"\nflush stdout\n", lines.join("\n"))
}

fn init_tcl(hw_server_url: &str, probes: Option<&Path>) -> Vec<String> {
    let mut lines = vec![
        "if {[llength [info commands load_features]]} { catch {load_features labtools} }".to_owned(),
        "if {[llength [info commands open_hw_manager]]} { open_hw_manager } elseif {[llength [info commands open_hw]]} { open_hw }".to_owned(),
        if hw_server_url.is_empty() { "if {[catch {connect_hw_server}]} { connect_hw_server }".to_owned() } else { format!("connect_hw_server -url {hw_server_url}") },
        "catch {refresh_hw_server}".into(),
        "set zlc_targets [get_hw_targets]".into(),
        "if {[llength $zlc_targets] != 1} { error \"Expected exactly one Vivado hardware target\" }".into(),
        "current_hw_target $zlc_targets".into(),
        "if {[catch {open_hw_target}]} { catch {close_hw_target}; after 2000; open_hw_target }".into(),
        "set zlc_devices [get_hw_devices]".into(),
        "if {[llength $zlc_devices] != 1} { error \"Expected exactly one Vivado hardware device\" }".into(),
        "current_hw_device $zlc_devices".into(),
    ];
    if let Some(probes) = probes {
        let path = probes.display().to_string().replace('\\', "/");
        lines.push(format!("set_property PROBES.FILE {{{path}}} [current_hw_device]"));
        lines.push(format!("set_property FULL_PROBES.FILE {{{path}}} [current_hw_device]"));
    }
    // Frozen deployment: this owner never selects or programs a bitstream.
    lines.extend(["refresh_hw_device [current_hw_device]".to_owned(), "set zlc_axi [get_hw_axis]".to_owned(),
                  "if {[llength $zlc_axi] != 1} { error \"Expected exactly one JTAG-to-AXI core\" }".to_owned()]);
    lines
}

// ------------------------------------------------------------------ the transport
struct AxiState {
    session: Option<Box<dyn TclSession>>,
    counter: u64,
    /// A reply a cancelled read walked away from: its lines are still coming,
    /// and the next action reads past them before it issues.
    abandoned: Option<String>,
}

/// The I/O owner: one action at a time, waited for no longer than its deadline.
struct Gate { busy: Mutex<bool>, freed: Condvar }

struct Held<'a>(&'a Gate);
impl Drop for Held<'_> {
    fn drop(&mut self) { *self.0.busy.lock().unwrap_or_else(|p| p.into_inner()) = false; self.0.freed.notify_one(); }
}

impl Gate {
    fn acquire(&self, deadline: Instant, action: &str) -> Result<Held<'_>> {
        let mut busy = self.busy.lock().unwrap_or_else(|p| p.into_inner());
        while *busy {
            let now = Instant::now();
            if now >= deadline { return Err(Error::new(Kind::Timeout, format!("{action} timed out waiting for the AXI I/O owner"))); }
            busy = self.freed.wait_timeout(busy, deadline - now).unwrap_or_else(|p| p.into_inner()).0;
        }
        *busy = true;
        Ok(Held(self))
    }
}

/// Ordered 32-bit register and BRAM access through one persistent Vivado Tcl owner.
pub struct JtagAxiRegisterTransport {
    launcher: Arc<dyn TclLauncher>,
    state_dir: PathBuf,
    probes: Option<PathBuf>,
    hw_server_url: String,
    gate: Gate,
    /// Outside the state lock: `close` sets it while a read holds the session.
    closed: AtomicBool,
    state: Mutex<AxiState>,
}

impl JtagAxiRegisterTransport {
    /// `hw_server_url` empty = Vivado's own local hw_server.
    pub fn new(launcher: Arc<dyn TclLauncher>, state_dir: &Path, probes: Option<PathBuf>, hw_server_url: &str) -> Result<JtagAxiRegisterTransport> {
        std::fs::create_dir_all(state_dir).map_err(|error| Error::new(Kind::Io, format!("Vivado state directory {}: {error}", state_dir.display())))?;
        Ok(JtagAxiRegisterTransport { launcher, state_dir: state_dir.to_path_buf(), probes, hw_server_url: hw_server_url.trim().to_owned(),
            gate: Gate { busy: Mutex::new(false), freed: Condvar::new() }, closed: AtomicBool::new(true),
            state: Mutex::new(AxiState { session: None, counter: 0, abandoned: None }) })
    }

    /// The deployment's Vivado (`default_vivado`), probes and hw_server
    /// (`ZLC_PS_HW_SERVER_URL` / `ZLC_HW_SERVER_URL`), with its working files in `state_dir`.
    pub fn from_env(state_dir: &Path) -> Result<JtagAxiRegisterTransport> {
        let url = std::env::var("ZLC_PS_HW_SERVER_URL").or_else(|_| std::env::var("ZLC_HW_SERVER_URL")).unwrap_or_default();
        Self::new(Arc::new(VivadoLauncher::from_env()), state_dir, default_probes(), &url)
    }

    fn state(&self) -> MutexGuard<'_, AxiState> { self.state.lock().unwrap_or_else(|p| p.into_inner()) }

    fn effective_deadline(given: Option<Instant>) -> Result<Instant> {
        let local = Instant::now() + ACTION_TIMEOUT;
        let deadline = given.map_or(local, |given| given.min(local));
        if deadline <= Instant::now() { return Err(Error::new(Kind::Timeout, "AXI transaction deadline expired before issue")); }
        Ok(deadline)
    }

    fn stop_session(state: &mut AxiState, graceful: bool) {
        if let Some(mut session) = state.session.take() { session.end(graceful); }
        // The reply nobody read went with the interpreter that owed it.
        state.abandoned = None;
    }

    fn is_closed(&self) -> bool { self.closed.load(Ordering::Acquire) }

    /// The session ends here, for good: a timeout or an exit leaves nothing to trust.
    fn fail(&self, state: &mut AxiState, graceful: bool) {
        self.closed.store(true, Ordering::Release);
        Self::stop_session(state, graceful);
    }

    fn run_tcl(&self, lines: &[String], action: &str, deadline: Instant, stop: Option<&Event>) -> Result<String> {
        let _held = self.gate.acquire(deadline, action)?;
        let mut state = self.state();
        if self.is_closed() { return Err(Error::new(Kind::Runtime, "Vivado AXI transport is closed; call start() before another transaction")); }
        if stop.is_some_and(Event::is_set) { return Err(aborted(&format!("{action} before issue"))); }
        self.discard_abandoned(&mut state, deadline)?;
        self.execute(&mut state, lines, action, deadline, stop)
    }

    fn execute(&self, state: &mut AxiState, lines: &[String], action: &str, deadline: Instant, stop: Option<&Event>) -> Result<String> {
        if Instant::now() >= deadline { return Err(Error::new(Kind::Timeout, format!("{action} exceeded its absolute deadline"))); }
        state.counter += 1;
        let marker = format!("ZLC_AXI_{:06}", state.counter);
        let text = wrap_tcl(lines, &marker);
        let written = state.session.as_mut().ok_or_else(|| Error::new(Kind::Runtime, "persistent Vivado transport is not running; call start() explicitly"))?
            .write(&text);
        if written.is_err() {
            self.fail(state, true);
            return Err(Error::new(Kind::Runtime, format!("Vivado stopped before {action}")));
        }
        let output = self.read_until(state, &marker, deadline, stop)?;
        record_diagnostic(&self.state_dir, action, &output);
        if output.contains(&format!("{marker}_ERROR")) {
            let detail = output.lines().map(str::trim).find(|line| line.starts_with("ERROR:")).unwrap_or("Vivado Tcl action returned an error");
            return Err(Error::new(Kind::Runtime, format!("persistent Vivado {action} failed: {detail}")));
        }
        Ok(output)
    }

    fn read_until(&self, state: &mut AxiState, marker: &str, deadline: Instant, stop: Option<&Event>) -> Result<String> {
        let end = format!("{marker}_END");
        let mut lines = Vec::new();
        loop {
            if self.is_closed() {
                Self::stop_session(state, false);
                return Err(aborted(&format!("transport closed waiting for {marker}")));
            }
            if stop.is_some_and(Event::is_set) {
                // A cancelled read is not a broken transport: the SAFE behind a
                // Stop travels next, on this same session (v2's STOP defect:
                // killing Vivado here left the board playing). The reply is
                // remembered, to be drained by the next action.
                state.abandoned = Some(marker.to_owned());
                return Err(aborted(&format!("read waiting for {marker}")));
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                self.fail(state, false);
                return Err(Error::new(Kind::Timeout, format!("Vivado action timed out waiting for {marker}")));
            }
            let session = state.session.as_mut().ok_or_else(|| Error::new(Kind::Runtime, "persistent Vivado process exited unexpectedly"))?;
            match session.read_line(remaining.min(READ_SLICE)) {
                Ok(None) => continue,
                Ok(Some(line)) => {
                    let done = line.contains(&end);
                    lines.push(line);
                    if done { return Ok(lines.join("\n")); }
                }
                Err(_) => {
                    self.fail(state, false);
                    return Err(Error::new(Kind::Runtime, "persistent Vivado process exited unexpectedly"));
                }
            }
        }
    }

    /// Read past the reply a cancelled read walked away from (Vivado answers one
    /// block at a time, so its lines would be taken for the next reply's). Only
    /// a block that never ends is a broken transport.
    fn discard_abandoned(&self, state: &mut AxiState, deadline: Instant) -> Result<()> {
        let Some(marker) = state.abandoned.clone() else { return Ok(()) };
        let end = format!("{marker}_END");
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                self.fail(state, false);
                return Err(Error::new(Kind::Timeout, format!("the abandoned Vivado reply {marker} never finished")));
            }
            let session = state.session.as_mut().ok_or_else(|| Error::new(Kind::Runtime, "persistent Vivado process exited unexpectedly"))?;
            match session.read_line(remaining.min(READ_SLICE)) {
                Ok(None) => continue,
                Ok(Some(line)) => if line.contains(&end) { state.abandoned = None; return Ok(()); },
                Err(_) => {
                    self.fail(state, false);
                    return Err(Error::new(Kind::Runtime, "persistent Vivado process exited unexpectedly"));
                }
            }
        }
    }
}

impl RegisterTransport for JtagAxiRegisterTransport {
    fn observer_interval(&self) -> Duration { JTAG_AXI_OBSERVER_INTERVAL }

    fn io_bounds(&self) -> Option<IoBounds> {
        Some(IoBounds { startup: STARTUP_TIMEOUT, action: ACTION_TIMEOUT, close: PROCESS_EXIT_TIMEOUT + PROCESS_KILL_TIMEOUT + READER_EXIT_TIMEOUT })
    }

    fn start(&self) -> Result<()> {
        let deadline = Instant::now() + STARTUP_TIMEOUT;
        let _held = self.gate.acquire(deadline, "transport_start")?;
        let mut state = self.state();
        if !self.is_closed() { return Ok(()); }
        state.session = Some(self.launcher.launch(&self.state_dir)?);
        self.closed.store(false, Ordering::Release);
        let lines = init_tcl(&self.hw_server_url, self.probes.as_deref());
        if let Err(error) = self.execute(&mut state, &lines, "transport_start", deadline, None) {
            self.fail(&mut state, false);
            return Err(error);
        }
        Ok(())
    }

    fn close(&self) -> Result<()> {
        // Closed first: a read in progress sees it and ends its own wait.
        self.closed.store(true, Ordering::Release);
        let _held = self.gate.acquire(Instant::now() + ACTION_TIMEOUT, "transport_close")?;
        Self::stop_session(&mut self.state(), true);
        Ok(())
    }

    fn write_words(&self, rows: &[(u32, u32)], stop: Option<&Event>, deadline: Option<Instant>) -> Result<()> {
        let absolute = Self::effective_deadline(deadline)?;
        let pending = rows.iter().map(|&(word, value)| Ok((byte_address(word)?, value))).collect::<Result<Vec<_>>>()?;
        let mut lines = Vec::new();
        let mut bursts = 0;
        for (base, values) in burst_runs(&pending) {
            lines.extend(write_burst_tcl(base, &values));
            bursts += 1;
            if bursts >= WRITE_BATCH {
                self.run_tcl(&std::mem::take(&mut lines), "axi_write", absolute, stop)?;
                bursts = 0;
            }
        }
        if !lines.is_empty() { self.run_tcl(&lines, "axi_write", absolute, stop)?; }
        Ok(())
    }

    fn read_words(&self, offset: u32, count: u32, stop: Option<&Event>, deadline: Option<Instant>) -> Result<Vec<u32>> {
        if !(1..=256).contains(&count) { return Err(Error::new(Kind::Value, "AXI read count must be in [1, 256]")); }
        let absolute = Self::effective_deadline(deadline)?;
        let output = self.run_tcl(&read_txn_tcl(byte_address(offset)?, "ZLCDATA", count), "axi_read", absolute, stop)?;
        parse_read(&output, "ZLCDATA", count)
    }

    fn command(&self, code: u32, command_id: u32, run_repeats: u32, scan_repeats: u32, stop: Option<&Event>,
               deadline: Option<Instant>) -> Result<(u32, u32)> {
        let absolute = Self::effective_deadline(deadline)?;
        let mut lines = Vec::new();
        for (address, value) in [(ctrl::RUN_REPEAT_COUNT, run_repeats), (ctrl::SCAN_REPEAT_COUNT, scan_repeats),
                                 (ctrl::COMMAND_ID, command_id), (ctrl::COMMAND, 0), (ctrl::COMMAND, code)] {
            lines.extend(write_burst_tcl(byte_address(address)?, &[value]));
        }
        // The board answers on its own ACK words; polled inside this block.
        let milliseconds = absolute.saturating_duration_since(Instant::now()).as_millis().max(1);
        lines.extend([
            format!("set zlc_cmd_deadline [expr {{[clock milliseconds] + {milliseconds}}}]"),
            "while {1} {".to_owned(),
            format!("create_hw_axi_txn zlc_r [get_hw_axis] -address {:08X} -len 3 -type read -force", byte_address(ctrl::ACK_ID)?),
            "run_hw_axi zlc_r".into(),
            "set zlc_cmd_answer [get_property DATA [get_hw_axi_txns zlc_r]]".into(),
            "delete_hw_axi_txn zlc_r".into(),
            format!("if {{([scan [string range $zlc_cmd_answer end-7 end] %x] & 0xffffffff) == {command_id}}} {{puts \"ZLCCOMMAND $zlc_cmd_answer\"; break}}"),
            "if {[clock milliseconds] >= $zlc_cmd_deadline} {error {pulse command completion timed out}}".into(),
            "}".into(),
        ]);
        let output = self.run_tcl(&lines, "pulse_command", absolute, stop)?;
        let reply = parse_read(&output, "ZLCCOMMAND", 3)?;
        if reply[0] != command_id { return Err(Error::new(Kind::Runtime, "AXI command completion belongs to another command")); }
        Ok((reply[1], reply[2]))
    }

    fn describe_link(&self) -> String { format!("JTAG-to-AXI through {}", self.launcher.describe()) }
}

// ------------------------------------------------------------------ the twin
/// What a [`FakeVivado`] does besides answering: latency, faults, a record.
#[derive(Default)]
pub struct FakeConduct {
    /// Each block's output becomes readable this long after it was written.
    pub latency: Duration,
    /// Hardware targets the fake hw_server reports (the real one must report one).
    pub targets: usize,
    /// The next block never finishes (Vivado wedged on the cable).
    pub hang_next: bool,
    /// The interpreter exits on the next block.
    pub exit_next: bool,
    /// The next `run_hw_axi` raises this (a Xicom fault).
    pub fail_run: Option<String>,
    /// Every block written, as text.
    pub blocks: Vec<String>,
    /// Interpreters started.
    pub launches: usize,
}

/// A Vivado Tcl session in front of a memory board: it understands the blocks
/// [`JtagAxiRegisterTransport`] writes (init, write and read transactions, the
/// command poll) and nothing it does not.
pub struct FakeVivado { pub board: Arc<MemoryRegisterTransport>, pub conduct: Arc<Mutex<FakeConduct>> }

impl FakeVivado {
    pub fn new(board: Arc<MemoryRegisterTransport>) -> FakeVivado {
        FakeVivado { board, conduct: Arc::new(Mutex::new(FakeConduct { targets: 1, ..FakeConduct::default() })) }
    }
    pub fn conduct(&self) -> MutexGuard<'_, FakeConduct> { self.conduct.lock().unwrap_or_else(|p| p.into_inner()) }
}

impl TclLauncher for FakeVivado {
    fn describe(&self) -> String { "a fake Vivado".into() }
    fn launch(&self, _state_dir: &Path) -> Result<Box<dyn TclSession>> {
        self.conduct().launches += 1;
        Ok(Box::new(FakeSession { board: self.board.clone(), conduct: self.conduct.clone(), output: VecDeque::new(), exited: false,
                                  txns: Default::default() }))
    }
}

struct Txn { address: u32, len: u32, write: Option<Vec<u32>>, data: String }

struct FakeSession {
    board: Arc<MemoryRegisterTransport>,
    conduct: Arc<Mutex<FakeConduct>>,
    output: VecDeque<(Instant, String)>,
    exited: bool,
    txns: std::collections::HashMap<String, Txn>,
}

fn option<'a>(words: &[&'a str], flag: &str) -> Option<&'a str> { words.iter().position(|word| *word == flag).and_then(|at| words.get(at + 1).copied()) }

impl FakeSession {
    fn conduct(&self) -> MutexGuard<'_, FakeConduct> { self.conduct.lock().unwrap_or_else(|p| p.into_inner()) }

    /// One body line; `Err` is a Tcl error.
    fn run_line(&mut self, line: &str, out: &mut Vec<String>) -> std::result::Result<(), String> {
        let words: Vec<&str> = line.split_whitespace().collect();
        match words.first().copied() {
            Some("create_hw_axi_txn") => {
                let name = words.get(1).copied().unwrap_or("").to_owned();
                let address = u32::from_str_radix(option(&words, "-address").ok_or("no -address")?, 16).map_err(|e| e.to_string())?;
                let len: u32 = option(&words, "-len").ok_or("no -len")?.parse().map_err(|_| "bad -len")?;
                let write = match option(&words, "-type") {
                    Some("write") => {
                        let data = option(&words, "-data").ok_or("no -data")?;
                        if data.len() != 8 * len as usize { return Err(format!("-data has {} digits for -len {len}", data.len())); }
                        Some((0..len as usize).map(|index| { let end = data.len() - 8 * index; u32::from_str_radix(&data[end - 8..end], 16).unwrap_or(0) }).collect())
                    }
                    Some("read") => None,
                    other => return Err(format!("unknown -type {other:?}")),
                };
                self.txns.insert(name, Txn { address, len, write, data: String::new() });
            }
            Some("run_hw_axi") => {
                if let Some(error) = self.conduct().fail_run.take() { return Err(error); }
                let txn = self.txns.get_mut(words.get(1).copied().unwrap_or("")).ok_or("no such transaction")?;
                if txn.address % 4 != 0 { return Err("unaligned AXI address".into()); }
                let word = txn.address / 4;
                match &txn.write {
                    Some(values) => {
                        let rows: Vec<(u32, u32)> = values.iter().enumerate().map(|(index, value)| (word + index as u32, *value)).collect();
                        self.board.write_words(&rows, None, None).map_err(|error| error.message)?;
                    }
                    None => {
                        let values = self.board.read_words(word, txn.len, None, None).map_err(|error| error.message)?;
                        txn.data = values.iter().rev().map(|value| format!("{value:08x}")).collect();
                    }
                }
            }
            Some("puts") => {
                // puts "MARKER [get_property DATA [get_hw_axi_txns zlc_r]]"
                let text = line.trim_start_matches("puts").trim().trim_matches('"');
                let printed = match text.split_once(" [get_property DATA [get_hw_axi_txns ") {
                    Some((marker, rest)) => {
                        let name = rest.trim_end_matches("]]");
                        format!("{marker} {}", self.txns.get(name).ok_or("no such transaction")?.data)
                    }
                    None => text.to_owned(),
                };
                out.push(printed);
            }
            Some("delete_hw_axi_txn") => { if let Some(name) = words.get(1) { self.txns.remove(*name); } }
            Some("if") if line.contains("Expected exactly one Vivado hardware target") && self.conduct().targets != 1 => {
                return Err("Expected exactly one Vivado hardware target".into());
            }
            _ => {}
        }
        Ok(())
    }

    /// The command poll: ACK_ID..ACK_CURSOR until the id answers, or the deadline.
    fn run_poll(&mut self, body: &[&str], milliseconds: u64, out: &mut Vec<String>) -> std::result::Result<(), String> {
        let wanted: u32 = body.iter().find_map(|line| line.split_once(") == ").and_then(|(_, rest)| rest.split('}').next()).and_then(|id| id.trim().parse().ok()))
            .ok_or("the poll names no command id")?;
        let until = Instant::now() + Duration::from_millis(milliseconds);
        loop {
            let words = self.board.read_words(ctrl::ACK_ID, 3, None, None).map_err(|error| error.message)?;
            if words[0] == wanted {
                out.push(format!("ZLCCOMMAND {}", words.iter().rev().map(|value| format!("{value:08x}")).collect::<String>()));
                return Ok(());
            }
            if Instant::now() >= until { return Err("pulse command completion timed out".into()); }
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    fn run_block(&mut self, text: &str) {
        let lines: Vec<&str> = text.lines().collect();
        let marker = lines.first().and_then(|line| line.strip_prefix("puts \"")).and_then(|line| line.strip_suffix("_BEGIN\"")).map(str::to_owned);
        let Some(marker) = marker else {
            // Not an action block: the polite close.
            if lines.iter().any(|line| line.trim() == "exit") { self.exited = true; }
            return;
        };
        let (latency, hang) = { let mut conduct = self.conduct(); conduct.blocks.push(text.to_owned()); (conduct.latency, std::mem::take(&mut conduct.hang_next)) };
        if std::mem::take(&mut self.conduct().exit_next) { self.exited = true; return; }
        let mut out = vec![format!("{marker}_BEGIN")];
        let start = lines.iter().position(|line| line.starts_with("set zlc_axi_code [catch {")).map_or(lines.len(), |at| at + 1);
        let stop = lines.iter().position(|line| line.starts_with("} zlc_axi_result")).unwrap_or(lines.len());
        let body = &lines[start.min(stop)..stop];
        let mut index = 0;
        let mut failure = None;
        while index < body.len() && failure.is_none() {
            let line = body[index].trim();
            if let Some(rest) = line.strip_prefix("set zlc_cmd_deadline [expr {[clock milliseconds] + ") {
                let milliseconds = rest.split('}').next().and_then(|value| value.trim().parse().ok()).unwrap_or(0);
                let close = body[index..].iter().position(|line| line.trim() == "}").map_or(body.len(), |at| index + at);
                failure = self.run_poll(&body[index..close], milliseconds, &mut out).err();
                index = close + 1;
                continue;
            }
            failure = self.run_line(line, &mut out).err();
            index += 1;
        }
        if hang { out.truncate(1); }
        else {
            match failure {
                Some(error) => { out.push(format!("ERROR: [Labtools 27-3176] {error}")); out.push(format!("{marker}_ERROR {error}")); out.push(error); }
                None => out.push(format!("{marker}_OK")),
            }
            out.push(format!("{marker}_END"));
        }
        let ready = Instant::now() + latency;
        self.output.extend(out.into_iter().map(|line| (ready, line)));
    }
}

impl TclSession for FakeSession {
    fn write(&mut self, text: &str) -> Result<()> {
        if self.exited { return Err(Error::new(Kind::Connection, "the fake Vivado has exited")); }
        self.run_block(text);
        Ok(())
    }

    fn read_line(&mut self, timeout: Duration) -> Result<Option<String>> {
        let until = Instant::now() + timeout;
        loop {
            match self.output.front() {
                Some((ready, _)) if *ready <= Instant::now() => return Ok(self.output.pop_front().map(|(_, line)| line)),
                Some((ready, _)) if *ready <= until => std::thread::sleep(ready.saturating_duration_since(Instant::now())),
                _ if self.exited => return Err(Error::new(Kind::Connection, "the fake Vivado has exited")),
                _ => { std::thread::sleep(until.saturating_duration_since(Instant::now())); return Ok(None); }
            }
        }
    }

    fn end(&mut self, _graceful: bool) { self.exited = true; self.output.clear(); }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_coalesce_in_order_within_bursts_and_4k_pages() {
        let rows: Vec<(u32, u32)> = (0..300).map(|word| (word * 4, word)).collect();
        let runs = burst_runs(&rows);
        assert_eq!(runs.iter().map(|(base, values)| (*base, values.len())).collect::<Vec<_>>(), [(0, 256), (1024, 44)]);
        // A 4 KiB page ends a burst; a gap or a backwards step starts a new one.
        let runs = burst_runs(&[(4088, 1), (4092, 2), (4096, 3), (8, 4), (4, 5), (12, 6)]);
        assert_eq!(runs, [(4088, vec![1, 2]), (4096, vec![3]), (8, vec![4]), (4, vec![5]), (12, vec![6])]);
        assert_eq!(write_burst_tcl(0x40, &[1, 0xDEADBEEF])[0],
                   "create_hw_axi_txn zlc_w [get_hw_axis] -address 00000040 -data DEADBEEF00000001 -len 2 -type write -burst INCR -force");
    }

    #[test]
    fn reads_take_the_last_marker_line_word_zero_lowest() {
        assert_eq!(parse_read("noise\nZLCDATA 0000000200000001\n", "ZLCDATA", 2).unwrap(), [1, 2]);
        assert_eq!(parse_read("ZLCDATA 1f\n", "ZLCDATA", 1).unwrap(), [0x1f]);
        assert_eq!(parse_read("ZLCDATA 0x00000003\nZLCDATA 00000004", "ZLCDATA", 1).unwrap(), [4]);
        assert!(parse_read("nothing", "ZLCDATA", 1).unwrap_err().message.contains("no DATA"));
        assert!(parse_read("ZLCDATA xyz", "ZLCDATA", 1).unwrap_err().message.contains("non-binary"));
        assert!(parse_read("ZLCDATA 123456789", "ZLCDATA", 1).unwrap_err().message.contains("non-binary"));
    }
}
