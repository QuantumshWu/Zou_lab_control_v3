//! The CRC-framed UART register transport at the deployed board's line rate.
//!
//! A request frame is `5A A5 | op seq | address u32 | count u16 | words... | crc16`
//! (CRC-16/CCITT-FALSE of the body, little-endian); a reply is
//! `5A A5 | 81 seq status | count u16 | words... | crc16`. Writes and reads are
//! idempotent and are sent again until answered within their deadline; a
//! command carries its own board-deduplicated identity, so resending a lost
//! command reply cannot start a second shot.
use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::error::{Error, Kind, Result};
use crate::transport::{DEFAULT_OBSERVER_INTERVAL, Event, RegisterTransport, aborted};

pub const SYNC0: u8 = 0x5A;
pub const SYNC1: u8 = 0xA5;
pub const OP_WRITE: u8 = 0x01;
pub const OP_READ: u8 = 0x02;
pub const OP_COMMAND: u8 = 0x03;
pub const RESP: u8 = 0x81;
pub const ST_OK: u8 = 0x00;
pub const ST_CRC_FAIL: u8 = 0x01;
pub const MAX_FRAME_WORDS: usize = 256;
/// Bytes one register word costs on the wire once framed, for deadline budgets.
pub const BYTES_PER_WORD_ESTIMATE: usize = 6;

/// `binascii.crc_hqx(data, 0xFFFF)`.
pub fn crc16_ccitt(data: &[u8]) -> u16 {
    let mut crc: u16 = 0xFFFF;
    for &byte in data {
        crc ^= u16::from(byte) << 8;
        for _ in 0..8 { crc = if crc & 0x8000 != 0 { (crc << 1) ^ 0x1021 } else { crc << 1 }; }
    }
    crc
}

fn frame(body: &[u8]) -> Vec<u8> {
    let mut out = vec![SYNC0, SYNC1];
    out.extend_from_slice(body);
    out.extend_from_slice(&crc16_ccitt(body).to_le_bytes());
    out
}

fn span(address: u32, count: usize) -> Result<()> {
    if count == 0 || count > MAX_FRAME_WORDS || u64::from(address) + count as u64 - 1 > u64::from(u32::MAX) {
        return Err(Error::new(Kind::Value, "invalid UART frame span"));
    }
    Ok(())
}

pub fn encode_write(address: u32, values: &[u32], seq: u8) -> Result<Vec<u8>> {
    span(address, values.len())?;
    let mut body = vec![OP_WRITE, seq];
    body.extend_from_slice(&address.to_le_bytes());
    body.extend_from_slice(&(values.len() as u16).to_le_bytes());
    for value in values { body.extend_from_slice(&value.to_le_bytes()); }
    Ok(frame(&body))
}

pub fn encode_read(address: u32, count: usize, seq: u8) -> Result<Vec<u8>> {
    span(address, count)?;
    let mut body = vec![OP_READ, seq];
    body.extend_from_slice(&address.to_le_bytes());
    body.extend_from_slice(&(count as u16).to_le_bytes());
    Ok(frame(&body))
}

pub fn encode_command(code: u32, command_id: u32, run_repeats: u32, scan_repeats: u32, seq: u8) -> Result<Vec<u8>> {
    if ![1, 2, 4, 8].contains(&code) { return Err(Error::new(Kind::Value, "unknown pulse command")); }
    if command_id == 0 { return Err(Error::new(Kind::Value, "command_id must be nonzero")); }
    let mut body = vec![OP_COMMAND, seq];
    body.extend_from_slice(&code.to_le_bytes());
    body.extend_from_slice(&3u16.to_le_bytes());
    for value in [command_id, run_repeats, scan_repeats] { body.extend_from_slice(&value.to_le_bytes()); }
    Ok(frame(&body))
}

pub fn encode_reply(seq: u8, status: u8, words: &[u32]) -> Vec<u8> {
    let mut body = vec![RESP, seq, status];
    body.extend_from_slice(&(words.len() as u16).to_le_bytes());
    for word in words { body.extend_from_slice(&word.to_le_bytes()); }
    frame(&body)
}

pub fn reply_frame_len(count: usize) -> usize { 2 + 3 + 2 + 4 * count + 2 }

/// `(seq, status, words)` of one verified reply.
pub fn decode_reply(frame: &[u8]) -> Result<(u8, u8, Vec<u32>)> {
    let bad = |message: &str| Error::new(Kind::Value, message.to_owned());
    if frame.len() < 9 || frame[0] != SYNC0 || frame[1] != SYNC1 || frame[2] != RESP { return Err(bad("invalid UART reply header")); }
    let count = u16::from_le_bytes([frame[5], frame[6]]) as usize;
    if count > MAX_FRAME_WORDS { return Err(bad("UART reply count exceeds the wire limit")); }
    if frame.len() != reply_frame_len(count) { return Err(bad("invalid UART reply length")); }
    let body = &frame[2..frame.len() - 2];
    if crc16_ccitt(body) != u16::from_le_bytes([frame[frame.len() - 2], frame[frame.len() - 1]]) { return Err(bad("UART reply CRC mismatch")); }
    let words = (0..count).map(|index| u32::from_le_bytes(frame[7 + 4 * index..11 + 4 * index].try_into().unwrap())).collect();
    Ok((frame[3], frame[4], words))
}

/// Consecutive addresses grouped into frames of at most `max_words` words.
pub fn coalesce_runs(pairs: &[(u32, u32)], max_words: usize) -> Vec<(u32, Vec<u32>)> {
    let mut result: Vec<(u32, Vec<u32>)> = Vec::new();
    let mut index = 0;
    while index < pairs.len() {
        let (base, value) = pairs[index];
        let mut values = vec![value];
        let mut cursor = index + 1;
        while cursor < pairs.len() && values.len() < max_words && u64::from(pairs[cursor].0) == u64::from(base) + values.len() as u64 {
            values.push(pairs[cursor].1);
            cursor += 1;
        }
        result.push((base, values));
        index = cursor;
    }
    result
}

/// The next verified reply in `buffer`, resynchronising past damage one byte at a time.
pub fn extract_reply(buffer: &mut Vec<u8>) -> Option<Vec<u8>> {
    loop {
        while buffer.len() >= 2 && !(buffer[0] == SYNC0 && buffer[1] == SYNC1) { buffer.remove(0); }
        if buffer.len() < 7 { return None; }
        let count = u16::from_le_bytes([buffer[5], buffer[6]]) as usize;
        if count > MAX_FRAME_WORDS { buffer.remove(0); continue; }
        let length = reply_frame_len(count);
        if buffer.len() < length { return None; }
        let candidate: Vec<u8> = buffer[..length].to_vec();
        if decode_reply(&candidate).is_err() { buffer.remove(0); continue; }
        buffer.drain(..length);
        return Some(candidate);
    }
}

/// One serial line: send bytes, collect one verified reply per pending SEQ.
pub trait UartLink: Send {
    fn open(&mut self) -> Result<()>;
    fn close(&mut self) -> Result<()>;
    fn port(&self) -> String;
    fn baud(&self) -> u64;
    /// Write `requests` (joined by eight 0xFF bytes) and collect the replies
    /// to their sequence numbers until `deadline`; fewer replies than requests
    /// is reported in `shortfall`, not raised.
    fn write_batch(&mut self, requests: &[Vec<u8>], deadline: Instant, stop: Option<&Event>) -> Result<Vec<Vec<u8>>>;
    /// What the last short read looked like, for whoever decides it is fatal.
    fn last_shortfall(&self) -> String { String::new() }
}

struct UartState { link: Box<dyn UartLink>, sequence: u8, closed: bool, resends: u64, last_retry_reason: String }

/// The UART register transport: frames, retries and deadlines over one link.
pub struct UartRegisterTransport { state: Mutex<UartState>, action_timeout: Duration, round_trip_allowance: f64, retry_slack: f64, max_frame_words: usize }

impl UartRegisterTransport {
    pub fn new(link: Box<dyn UartLink>, action_timeout: Duration) -> UartRegisterTransport {
        UartRegisterTransport { state: Mutex::new(UartState { link, sequence: 0, closed: true, resends: 0, last_retry_reason: String::new() }),
                                action_timeout, round_trip_allowance: 0.05, retry_slack: 0.08, max_frame_words: MAX_FRAME_WORDS }
    }

    fn state(&self) -> std::sync::MutexGuard<'_, UartState> { self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) }

    /// The transaction deadline: the bytes' wire time, one round trip per frame,
    /// plus `action_timeout` of slack, unless the caller gave one.
    fn deadline(&self, given: Option<Instant>, baud: u64, frames: usize, words: usize) -> Result<Instant> {
        let result = given.unwrap_or_else(|| {
            let wire = (words.max(1) * BYTES_PER_WORD_ESTIMATE) as f64 * 10.0 / (baud.max(1) as f64);
            Instant::now() + self.action_timeout + Duration::from_secs_f64(wire + frames.max(1) as f64 * self.round_trip_allowance)
        });
        if result <= Instant::now() { return Err(Error::new(Kind::Timeout, "UART transaction deadline expired")); }
        Ok(result)
    }

    /// How long one attempt may take for these frames: wire time with a host
    /// factor, the scheduling slack, and a little per frame for batched replies.
    fn attempt_budget(&self, frames: &[Vec<u8>], baud: u64) -> Duration {
        let mut payload: usize = frames.iter().map(|frame| frame.len() + 8).sum();
        payload += frames.iter().map(|frame| reply_frame_len(match frame[2] {
            OP_READ => u16::from_le_bytes([frame[8], frame[9]]) as usize,
            OP_COMMAND => 3,
            _ => 0,
        })).sum::<usize>();
        let wire = payload as f64 * 10.0 / (baud.max(1) as f64);
        Duration::from_secs_f64(wire * 1.5 + self.retry_slack + 0.002 * frames.len() as f64)
    }

    fn next_sequence(state: &mut UartState) -> u8 { state.sequence = state.sequence.wrapping_add(1); state.sequence }

    /// Send `frames` until every one is answered (`classify` sorts each reply),
    /// or the deadline passes. Failed attempts keep their real error.
    fn until_answered(&self, state: &mut UartState, frames: Vec<Vec<u8>>, absolute: Instant, stop: Option<&Event>, what: &str,
                      mut accept: impl FnMut(&[u8], &[u8]) -> Result<Verdict>) -> Result<()> {
        let started = Instant::now();
        let mut outstanding = frames;
        let total = outstanding.len();
        let mut failures: Vec<(usize, usize, String)> = Vec::new();
        let mut attempts = 0usize;
        let mut next_at = started;
        let mut refused_all = false;
        let baud = state.link.baud();
        loop {
            let now = Instant::now();
            if now >= absolute { break; }
            if now < next_at {
                let pause = next_at.min(absolute) - now;
                match stop { Some(stop) => if stop.wait(Some(pause)) { return Err(aborted("UART request")); }, None => std::thread::sleep(pause) }
                continue;
            }
            if let Some(last) = failures.last() {
                state.resends += outstanding.len() as u64;
                state.last_retry_reason = format!("{what}: {}", last.2);
            }
            attempts += 1;
            next_at = now + self.attempt_budget(&outstanding, baud);
            refused_all = false;
            let failure = match state.link.write_batch(&outstanding, next_at.min(absolute), stop) {
                Err(error) if error.kind == Kind::Timeout => Some(error.message),
                Err(error) => return Err(error),
                Ok(replies) => {
                    let mut answered = BTreeSet::new();
                    let mut rejected = BTreeSet::new();
                    for reply in &replies {
                        let Some(request) = outstanding.iter().find(|frame| frame[3] == reply.get(3).copied().unwrap_or(0)) else { continue };
                        match accept(request, reply)? {
                            Verdict::Answered => { answered.insert(request[3]); }
                            Verdict::Rejected => { rejected.insert(request[3]); }
                            Verdict::Noise => {}
                        }
                    }
                    outstanding.retain(|frame| !answered.contains(&frame[3]));
                    if outstanding.is_empty() { return Ok(()); }
                    refused_all = outstanding.iter().all(|frame| rejected.contains(&frame[3]));
                    Some(if refused_all { format!("{} of {total} frame(s) rejected as damaged (request CRC)", outstanding.len()) }
                         else { format!("{} of {total} frame(s) unanswered: {}", outstanding.len(),
                                        Some(state.link.last_shortfall()).filter(|text| !text.is_empty()).unwrap_or_else(|| "no detail".into())) })
                }
            };
            if let Some(failure) = failure {
                match failures.last_mut() { Some(last) if last.2 == failure => last.1 = attempts, _ => failures.push((attempts, attempts, failure)) }
            }
        }
        let record = format!("on {} at {} baud after {attempts} attempt(s) in {:.2}s ({what}): {}", state.link.port(), baud,
                             started.elapsed().as_secs_f64(), attempt_record(&failures));
        if !failures.is_empty() && refused_all { return Err(Error::new(Kind::Runtime, format!("UART request rejected as damaged (request CRC) {record}"))); }
        Err(Error::new(Kind::Timeout, format!("UART reply timed out {record}")))
    }

    fn exchange(&self, state: &mut UartState, request: Vec<u8>, absolute: Instant, stop: Option<&Event>, count: usize) -> Result<Vec<u32>> {
        let what = if request[2] == OP_COMMAND { format!("command {}", u32::from_le_bytes(request[4..8].try_into().unwrap())) }
                   else { format!("read of {count} words at {}", u32::from_le_bytes(request[4..8].try_into().unwrap())) };
        let mut value = None;
        self.until_answered(state, vec![request], absolute, stop, &what, |request, reply| {
            let Ok((sequence, status, words)) = decode_reply(reply) else { return Ok(Verdict::Noise) };
            if sequence != request[3] { return Ok(Verdict::Noise); }
            if status == ST_CRC_FAIL { return Ok(Verdict::Rejected); }
            if status != ST_OK || words.len() != count {
                return Err(Error::new(Kind::Runtime, format!("UART read reply was invalid (status=0x{status:02X})")));
            }
            value = Some(words);
            Ok(Verdict::Answered)
        })?;
        Ok(value.expect("answered"))
    }
}

enum Verdict { Answered, Rejected, Noise }

fn attempt_record(failures: &[(usize, usize, String)]) -> String {
    if failures.is_empty() { return "no attempt ran".into(); }
    failures.iter().map(|(first, last, failure)| if first == last { format!("#{first}: {failure}") } else { format!("#{first}-{last}: {failure}") })
        .collect::<Vec<_>>().join("; ")
}

impl RegisterTransport for UartRegisterTransport {
    fn observer_interval(&self) -> Duration { DEFAULT_OBSERVER_INTERVAL }

    fn start(&self) -> Result<()> {
        let mut state = self.state();
        if state.closed { state.link.open()?; state.closed = false; }
        Ok(())
    }

    /// Closed means released: a link close that fails leaves the transport open.
    fn close(&self) -> Result<()> {
        let mut state = self.state();
        if state.closed { return Ok(()); }
        state.link.close()?;
        state.closed = true;
        Ok(())
    }

    fn write_words(&self, rows: &[(u32, u32)], stop: Option<&Event>, deadline: Option<Instant>) -> Result<()> {
        let mut state = self.state();
        if state.closed { return Err(Error::new(Kind::Runtime, "UART transport is closed")); }
        let runs = coalesce_runs(rows, self.max_frame_words);
        let mut frames = Vec::with_capacity(runs.len());
        for (base, values) in runs { let seq = Self::next_sequence(&mut state); frames.push(encode_write(base, &values, seq)?); }
        let baud = state.link.baud();
        let absolute = self.deadline(deadline, baud, frames.len(), rows.len())?;
        // Groups that cannot exhaust the 8-bit SEQ space: replies are matched by SEQ.
        for group in frames.chunks(255) {
            let what = format!("write of {} frame(s)", group.len());
            self.until_answered(&mut state, group.to_vec(), absolute, stop, &what, |_request, reply| {
                let Ok((_sequence, status, words)) = decode_reply(reply) else { return Ok(Verdict::Noise) };
                if status == ST_CRC_FAIL { return Ok(Verdict::Rejected); }
                if status != ST_OK || !words.is_empty() {
                    return Err(Error::new(Kind::Runtime, format!("UART write acknowledgement was invalid (status=0x{status:02X})")));
                }
                Ok(Verdict::Answered)
            })?;
        }
        Ok(())
    }

    fn read_words(&self, offset: u32, count: u32, stop: Option<&Event>, deadline: Option<Instant>) -> Result<Vec<u32>> {
        let mut state = self.state();
        if state.closed { return Err(Error::new(Kind::Runtime, "UART transport is closed")); }
        let absolute = self.deadline(deadline, state.link.baud(), 1, 1)?;
        let seq = Self::next_sequence(&mut state);
        let request = encode_read(offset, count as usize, seq)?;
        self.exchange(&mut state, request, absolute, stop, count as usize)
    }

    fn command(&self, code: u32, command_id: u32, run_repeats: u32, scan_repeats: u32, stop: Option<&Event>, deadline: Option<Instant>) -> Result<(u32, u32)> {
        let mut state = self.state();
        if state.closed { return Err(Error::new(Kind::Runtime, "UART transport is closed")); }
        let absolute = self.deadline(deadline, state.link.baud(), 1, 1)?;
        let seq = Self::next_sequence(&mut state);
        let request = encode_command(code, command_id, run_repeats, scan_repeats, seq)?;
        let reply = self.exchange(&mut state, request, absolute, stop, 3)?;
        if reply[0] != command_id { return Err(Error::new(Kind::Runtime, format!("command reply belongs to {}, expected {command_id}", reply[0]))); }
        Ok((reply[1], reply[2]))
    }

    fn resends(&self) -> u64 { self.state().resends }
    fn last_retry_reason(&self) -> String { self.state().last_retry_reason.clone() }
    fn describe_link(&self) -> String { let state = self.state(); format!("UART {} at {} baud", state.link.port(), state.link.baud()) }
}

// ------------------------------------------------------------------ a board behind a fake line
/// A UART board twin for tests: frames go into the memory register file and
/// replies come back, optionally losing or damaging some on the way.
pub struct LoopbackLink {
    pub board: Arc<crate::transport::MemoryRegisterTransport>,
    /// Replies to drop, by the index of the frame they answer (counted over the link's life).
    pub drop_replies: BTreeSet<usize>,
    /// Requests the board should refuse as damaged.
    pub damage_requests: BTreeSet<usize>,
    frames_seen: usize,
    open: bool,
}

impl LoopbackLink {
    pub fn new(board: Arc<crate::transport::MemoryRegisterTransport>) -> Self {
        LoopbackLink { board, drop_replies: BTreeSet::new(), damage_requests: BTreeSet::new(), frames_seen: 0, open: false }
    }

    fn answer(&self, request: &[u8]) -> Vec<u8> {
        let seq = request[3];
        let address = u32::from_le_bytes(request[4..8].try_into().unwrap());
        let count = u16::from_le_bytes([request[8], request[9]]) as usize;
        let body = &request[2..request.len() - 2];
        if crc16_ccitt(body) != u16::from_le_bytes([request[request.len() - 2], request[request.len() - 1]]) { return encode_reply(seq, ST_CRC_FAIL, &[]); }
        let data = &request[10..request.len() - 2];
        let words: Vec<u32> = data.chunks_exact(4).map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap())).collect();
        match request[2] {
            OP_WRITE => { let rows: Vec<(u32, u32)> = words.iter().enumerate().map(|(i, v)| (address + i as u32, *v)).collect();
                          let _ = self.board.write_words(&rows, None, None); encode_reply(seq, ST_OK, &[]) }
            OP_READ => encode_reply(seq, ST_OK, &self.board.read_words(address, count as u32, None, None).unwrap_or_default()),
            OP_COMMAND => {
                let (status, cursor) = self.board.command(address, words[0], words[1], words[2], None, None).unwrap_or((0, 0));
                encode_reply(seq, ST_OK, &[words[0], status, cursor])
            }
            _ => encode_reply(seq, 0x02, &[]),
        }
    }
}

impl UartLink for LoopbackLink {
    fn open(&mut self) -> Result<()> { self.open = true; Ok(()) }
    fn close(&mut self) -> Result<()> { self.open = false; Ok(()) }
    fn port(&self) -> String { "LOOPBACK".into() }
    fn baud(&self) -> u64 { crate::wire::DEFAULT_UART_BAUD }
    fn write_batch(&mut self, requests: &[Vec<u8>], deadline: Instant, stop: Option<&Event>) -> Result<Vec<Vec<u8>>> {
        if stop.is_some_and(Event::is_set) { return Err(aborted("UART request")); }
        let mut replies = Vec::new();
        for request in requests {
            let index = self.frames_seen;
            self.frames_seen += 1;
            let mut request = request.clone();
            if self.damage_requests.contains(&index) { let at = request.len() - 3; request[at] ^= 0xFF; }
            let reply = self.answer(&request);
            if !self.drop_replies.contains(&index) { replies.push(reply); }
        }
        if replies.len() < requests.len() {
            let now = Instant::now();
            if deadline > now { std::thread::sleep((deadline - now).min(Duration::from_millis(20))); }
        }
        Ok(replies)
    }
}
