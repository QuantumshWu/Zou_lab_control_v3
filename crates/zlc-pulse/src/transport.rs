//! Register transports of the pulse device, and the in-memory board twin.
//!
//! A transport moves 32-bit words to and from the board's word-addressed
//! register map and runs one identified command (LOAD, FIRE, SAFE, RESET) with
//! its completion acknowledgement. Every call may be cancelled by the owner's
//! stop event and is bounded by a deadline.
use std::collections::BTreeMap;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use crate::error::{Error, Kind, Result};
use crate::wire::{CMD_FIRE, CMD_LOAD, CMD_RESET, CMD_SAFE, STATUS_DONE, STATUS_ERROR, STATUS_LOADED, STATUS_RUNNING, StreamerParams,
                  build_fingerprint, ctrl};

/// How long the observer sleeps between two STATUS/CURSOR reads while a FIRE
/// plays: DONE is seen at most 5 ms (plus one read) after the board reaches it.
pub const DEFAULT_OBSERVER_INTERVAL: Duration = Duration::from_millis(5);

/// `threading.Event`: a flag threads can set, clear and wait for.
#[derive(Default)]
pub struct Event { flag: Mutex<bool>, changed: Condvar }

impl Event {
    pub fn new() -> Arc<Event> { Arc::new(Event::default()) }
    fn guard(&self) -> MutexGuard<'_, bool> { self.flag.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) }
    pub fn set(&self) { *self.guard() = true; self.changed.notify_all(); }
    pub fn clear(&self) { *self.guard() = false; }
    pub fn is_set(&self) -> bool { *self.guard() }
    /// Wait until set or until the timeout; whether it is set.
    pub fn wait(&self, timeout: Option<Duration>) -> bool {
        let mut flag = self.guard();
        let deadline = timeout.map(|timeout| Instant::now() + timeout);
        while !*flag {
            match deadline {
                None => flag = self.changed.wait(flag).unwrap_or_else(|poisoned| poisoned.into_inner()),
                Some(deadline) => {
                    let now = Instant::now();
                    if now >= deadline { return false; }
                    flag = self.changed.wait_timeout(flag, deadline - now).unwrap_or_else(|poisoned| poisoned.into_inner()).0;
                }
            }
        }
        true
    }
}

/// A pending register action cancelled by the owning session.
pub fn aborted(what: &str) -> Error { Error::new(Kind::Runtime, format!("{what} cancelled")) }

/// How long a slow link may take: to start, per action, to close. A server
/// tells its client the budget of a request from these (v2 `request_budget`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IoBounds { pub startup: Duration, pub action: Duration, pub close: Duration }

pub trait RegisterTransport: Send + Sync {
    /// The observer's polling period on this link.
    fn observer_interval(&self) -> Duration { DEFAULT_OBSERVER_INTERVAL }
    /// The bounds of a link whose actions can outlast a request's ordinary timeout.
    fn io_bounds(&self) -> Option<IoBounds> { None }
    fn start(&self) -> Result<()>;
    fn close(&self) -> Result<()>;
    fn write_words(&self, rows: &[(u32, u32)], stop: Option<&Event>, deadline: Option<Instant>) -> Result<()>;
    fn read_words(&self, offset: u32, count: u32, stop: Option<&Event>, deadline: Option<Instant>) -> Result<Vec<u32>>;
    fn read_word(&self, offset: u32, stop: Option<&Event>, deadline: Option<Instant>) -> Result<u32> {
        Ok(self.read_words(offset, 1, stop, deadline)?[0])
    }
    /// One identified command; `(status, cursor)` of its acknowledgement.
    #[allow(clippy::too_many_arguments)]
    fn command(&self, code: u32, command_id: u32, run_repeats: u32, scan_repeats: u32, stop: Option<&Event>,
               deadline: Option<Instant>) -> Result<(u32, u32)>;
    /// Frames sent again since the transport was made (a line quietly degrading).
    fn resends(&self) -> u64 { 0 }
    fn last_retry_reason(&self) -> String { String::new() }
    fn describe_link(&self) -> String { String::new() }
}

// ------------------------------------------------------------------ memory twin
struct MemoryState {
    words: BTreeMap<u32, u32>,
    write_batches: Vec<Vec<(u32, u32)>>,
    read_log: Vec<u32>,
    status: u32,
    cursor: u32,
    started: bool,
    closed: bool,
    command_seen: u32,
    resident: bool,
    last_command_id: u32,
    last_reply: (u32, u32),
}

/// The register file of a board in memory: the sole hardware twin. The host
/// writes and reads it as it would the board; a virtual world plays the board
/// side through [`MemoryRegisterTransport::publish_execution_readback`].
pub struct MemoryRegisterTransport { state: Mutex<MemoryState>, auto_done: bool, record_history: bool }

impl MemoryRegisterTransport {
    pub fn new(geometry: &StreamerParams, auto_done: bool, record_history: bool) -> Result<MemoryRegisterTransport> {
        Ok(Self::with_layout(build_fingerprint(geometry)?, auto_done, record_history))
    }

    pub fn with_layout(layout_id: u32, auto_done: bool, record_history: bool) -> MemoryRegisterTransport {
        let words = BTreeMap::from([(ctrl::LAYOUT_ID, layout_id)]);
        MemoryRegisterTransport {
            state: Mutex::new(MemoryState { words, write_batches: Vec::new(), read_log: Vec::new(), status: 0, cursor: 0, started: false,
                                            closed: false, command_seen: 0, resident: false, last_command_id: 0, last_reply: (0, 0) }),
            auto_done, record_history,
        }
    }

    fn state(&self) -> MutexGuard<'_, MemoryState> { self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) }

    pub fn word(&self, address: u32) -> u32 { self.state().words.get(&address).copied().unwrap_or(0) }
    pub fn status(&self) -> u32 { self.state().status }
    pub fn write_batches(&self) -> Vec<Vec<(u32, u32)>> { self.state().write_batches.clone() }
    pub fn read_log(&self) -> Vec<u32> { self.state().read_log.clone() }
    pub fn is_started(&self) -> bool { let state = self.state(); state.started && !state.closed }

    fn complete_command(&self, state: &mut MemoryState, code: u32, command_id: u32, run_repeats: u32, scan_repeats: u32) -> (u32, u32) {
        if command_id != 0 && command_id == state.last_command_id { return state.last_reply; }
        state.words.insert(ctrl::COMMAND_ID, command_id);
        match code {
            CMD_RESET => { state.resident = false; state.status = 0; state.cursor = 0; }
            CMD_SAFE => { state.status = 0; state.cursor = 0; }
            CMD_LOAD => { state.resident = true; state.status = STATUS_LOADED; }
            CMD_FIRE if state.resident => {
                state.words.insert(ctrl::RUN_REPEAT_COUNT, run_repeats);
                state.words.insert(ctrl::SCAN_REPEAT_COUNT, scan_repeats);
                state.cursor = 0;
                state.status = STATUS_RUNNING;
            }
            _ => state.status = STATUS_ERROR,
        }
        let reply = (state.status, state.cursor);
        state.last_command_id = command_id;
        state.last_reply = reply;
        state.words.insert(ctrl::ACK_ID, command_id);
        state.words.insert(ctrl::ACK_STATUS, reply.0);
        state.words.insert(ctrl::ACK_CURSOR, reply.1);
        if code == CMD_FIRE && state.status == STATUS_RUNNING && self.auto_done && run_repeats != 0 && scan_repeats != 0 {
            let count = state.words.get(&ctrl::SCAN_COUNT).copied().unwrap_or(0);
            let enabled = state.words.get(&ctrl::SCAN_ENABLE).copied().unwrap_or(0) != 0;
            state.cursor = if count != 0 && enabled { count.wrapping_mul(scan_repeats).wrapping_sub(1) } else { 0 };
            state.status = STATUS_DONE;
        }
        reply
    }

    /// One board-owned runtime transition (STATUS and CURSOR) at the seams the
    /// RTL updates them. Ignored once the host has driven the twin safe, so a
    /// late virtual-world callback cannot resurrect RUNNING.
    pub fn publish_execution_readback(&self, status: u32, cursor: u32) -> Result<bool> {
        if ![STATUS_RUNNING, STATUS_DONE, STATUS_ERROR].contains(&status) {
            return Err(Error::new(Kind::Value, "memory execution readback must be RUNNING, DONE, or ERROR"));
        }
        let mut state = self.state();
        if state.status & STATUS_RUNNING == 0 { return Ok(false); }
        state.cursor = cursor;
        state.status = status;
        Ok(true)
    }
}

impl RegisterTransport for MemoryRegisterTransport {
    fn start(&self) -> Result<()> { let mut state = self.state(); state.started = true; state.closed = false; Ok(()) }
    fn close(&self) -> Result<()> { let mut state = self.state(); state.closed = true; state.started = false; Ok(()) }

    fn write_words(&self, rows: &[(u32, u32)], stop: Option<&Event>, _deadline: Option<Instant>) -> Result<()> {
        if stop.is_some_and(Event::is_set) { return Err(aborted("memory transport write")); }
        let mut state = self.state();
        if self.record_history { state.write_batches.push(rows.to_vec()); }
        for &(address, value) in rows {
            match address {
                ctrl::STATUS => state.status = value,
                ctrl::CURSOR => state.cursor = value,
                ctrl::COMMAND => {
                    // The board runs a command on a rising edge and never clears the register itself.
                    let risen = value & !state.command_seen;
                    state.command_seen = value;
                    if risen != 0 {
                        let id = state.words.get(&ctrl::COMMAND_ID).copied().unwrap_or(0);
                        let run = state.words.get(&ctrl::RUN_REPEAT_COUNT).copied().unwrap_or(1);
                        let scan = state.words.get(&ctrl::SCAN_REPEAT_COUNT).copied().unwrap_or(1);
                        self.complete_command(&mut state, risen, id, run, scan);
                    }
                }
                _ => {}
            }
            state.words.insert(address, value);
        }
        Ok(())
    }

    fn read_words(&self, offset: u32, count: u32, stop: Option<&Event>, _deadline: Option<Instant>) -> Result<Vec<u32>> {
        if stop.is_some_and(Event::is_set) { return Err(aborted("memory transport read")); }
        let mut state = self.state();
        let mut values = Vec::with_capacity(count as usize);
        for address in offset..offset + count {
            if self.record_history { state.read_log.push(address); }
            values.push(match address {
                ctrl::STATUS => state.status,
                ctrl::CURSOR => state.cursor,
                _ => state.words.get(&address).copied().unwrap_or(0),
            });
        }
        Ok(values)
    }

    fn command(&self, code: u32, command_id: u32, run_repeats: u32, scan_repeats: u32, stop: Option<&Event>,
               _deadline: Option<Instant>) -> Result<(u32, u32)> {
        if stop.is_some_and(Event::is_set) { return Err(aborted("memory command")); }
        let mut state = self.state();
        if self.record_history { state.write_batches.push(vec![(ctrl::COMMAND_ID, command_id), (ctrl::COMMAND, code)]); }
        Ok(self.complete_command(&mut state, code, command_id, run_repeats, scan_repeats))
    }

    fn describe_link(&self) -> String { "memory".into() }
}
