//! Where the desktop's diagnostics go. zlc-desktop is a Windows GUI program with
//! no console. Its JSON lines and errors (stdout, stderr) stay where whoever
//! started it redirected them (a harness's file or pipe); a stream nobody
//! redirected goes to this run's log, `<checkout>\.local\logs\<name>-<date>-
//! <time>-<pid>.log` (zlc-desktop; installation for installation.exe and its
//! steps), which the children that inherit stderr (the local pulse server)
//! write to as well.
//!
//! The streams reach the log through a pipe this process reads, so the log keeps
//! within [`BUDGET`] however much a run writes: past it, the oldest half goes and a
//! line says how much went. [`drain`] waits until what was written so far is in the
//! file (at a panic, a fatal error and the application's end).
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::windows::io::IntoRawHandle;
use std::path::PathBuf;
use std::sync::{Condvar, Mutex, OnceLock};
use std::time::Duration;

use windows_sys::Win32::Foundation::{HANDLE, INVALID_HANDLE_VALUE};
use windows_sys::Win32::Storage::FileSystem::{FILE_TYPE_DISK, FILE_TYPE_PIPE, GetFileType};
use windows_sys::Win32::System::Console::{GetStdHandle, STD_ERROR_HANDLE, STD_HANDLE, STD_OUTPUT_HANDLE, SetStdHandle};

/// The runs whose logs are kept; older ones are removed when a run starts.
const KEPT: usize = 20;
/// The most of a run's log kept on disk. A run's log is read through the diagnostics
/// archive, which carries each program log's last `LOG_LIMIT` bytes: past that, nothing
/// of it would ever be read. Trimmed to half of it, the whole log still fits the archive.
const BUDGET: u64 = zlc_workbench::diagnostics::LOG_LIMIT;
/// How long [`drain`] waits for the log to have everything written before it.
const DRAIN_WAIT: Duration = Duration::from_secs(2);
/// What [`drain`] writes into the pipe and the reader takes out: everything before it
/// is in the file once the reader has met it. Escape-framed, never a log's own text.
const MARKER: &[u8] = b"\x1b]zlc-run-log-drained\x07";

pub struct Routed {
    /// This run's log, when a stream goes to it.
    pub log: Option<PathBuf>,
    /// Whether a caller reads stderr (it was redirected to a file or pipe).
    errors_read: bool,
}

/// Whether `stream` was handed to this process as a file or a pipe.
fn redirected(stream: STD_HANDLE) -> bool {
    let handle: HANDLE = unsafe { GetStdHandle(stream) };
    !handle.is_null() && handle != INVALID_HANDLE_VALUE && matches!(unsafe { GetFileType(handle) }, FILE_TYPE_DISK | FILE_TYPE_PIPE)
}

/// Send the streams nobody redirected to this run's log, `<name>-…` (Rust's stdout
/// and stderr look the standard handles up on every write).
pub fn route(name: &str) -> Routed {
    let streams: Vec<STD_HANDLE> = [STD_OUTPUT_HANDLE, STD_ERROR_HANDLE].into_iter().filter(|stream| !redirected(*stream)).collect();
    let errors_read = !streams.contains(&STD_ERROR_HANDLE);
    if streams.is_empty() { return Routed { log: None, errors_read }; }
    let folder = zlc_core::checkout::local("logs");
    if std::fs::create_dir_all(&folder).is_err() { return Routed { log: None, errors_read }; }
    // The names sort by start time: all but the newest KEPT - 1 of this name go.
    let prefix = format!("{name}-");
    let mut old: Vec<PathBuf> = std::fs::read_dir(&folder).into_iter().flatten().filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.file_name().and_then(|name| name.to_str()).is_some_and(|name| name.starts_with(&prefix) && name.ends_with(".log")))
        .collect();
    old.sort();
    for path in old.iter().rev().skip(KEPT - 1) { let _ = std::fs::remove_file(path); }
    let mut time = unsafe { std::mem::zeroed::<windows_sys::Win32::Foundation::SYSTEMTIME>() };
    unsafe { windows_sys::Win32::System::SystemInformation::GetLocalTime(&mut time) };
    let path = folder.join(format!("{prefix}{:04}{:02}{:02}-{:02}{:02}{:02}-{}.log", time.wYear, time.wMonth, time.wDay,
                                   time.wHour, time.wMinute, time.wSecond, std::process::id()));
    // Written by this process alone (its reader of the pipe), so trimming may rewrite it in place.
    let Ok(file) = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(true).open(&path) else { return Routed { log: None, errors_read } };
    let Ok((reader, writer)) = std::io::pipe() else { return Routed { log: None, errors_read } };
    let Ok(marker) = writer.try_clone() else { return Routed { log: None, errors_read } };
    let mut log = Budgeted::new(file, BUDGET);
    let started = std::thread::Builder::new().name("zlc-run-log".into()).spawn(move || copy(reader, &mut log));
    if started.is_err() { return Routed { log: None, errors_read }; }
    let _ = DRAIN.set(Drain { marker: Mutex::new(marker), asked: Mutex::new((0, 0)), met: Condvar::new() });
    // The write end lives as long as the process (children inheriting stderr get duplicates).
    let handle = writer.into_raw_handle() as HANDLE;
    for stream in streams { unsafe { SetStdHandle(stream, handle) }; }
    // A panic's message is in the file before the process can end.
    let reported = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| { reported(info); drain(); }));
    Routed { log: Some(path), errors_read }
}

/// Read the pipe into the log until every write end has closed.
fn copy(mut reader: std::io::PipeReader, log: &mut Budgeted) {
    let mut markers = Markers::default();
    let mut buffer = vec![0u8; 64 << 10];
    loop {
        let read = match reader.read(&mut buffer) { Ok(0) | Err(_) => return, Ok(read) => read };
        let (bytes, met) = markers.take(&buffer[..read]);
        // A log that cannot be written (a full disk) is not this process's failure: the pipe is still emptied.
        let _ = log.write(&bytes);
        if met > 0 {
            let Some(drain) = DRAIN.get() else { continue };
            drain.asked.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).1 += met;
            drain.met.notify_all();
        }
    }
}

struct Drain { marker: Mutex<std::io::PipeWriter>, asked: Mutex<(u64, u64)>, met: Condvar }

static DRAIN: OnceLock<Drain> = OnceLock::new();

/// Wait (at most [`DRAIN_WAIT`]) until everything written to the run's log so far is in
/// its file. Nothing to wait for when the streams go elsewhere.
pub fn drain() {
    let Some(drain) = DRAIN.get() else { return };
    let ticket = {
        let mut asked = drain.asked.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        asked.0 += 1;
        asked.0
    };
    // One small write: a pipe never interleaves it with another writer's.
    if drain.marker.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).write_all(MARKER).is_err() { return; }
    let asked = drain.asked.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let _ = drain.met.wait_timeout_while(asked, DRAIN_WAIT, |(_, met)| *met < ticket);
}

/// Takes [`MARKER`]s out of the stream, across reads that split one.
#[derive(Default)]
struct Markers { carried: Vec<u8> }

impl Markers {
    /// The bytes of `chunk` (after what an earlier one left) that are the log's, and how many markers were met.
    fn take(&mut self, chunk: &[u8]) -> (Vec<u8>, u64) {
        let mut data = std::mem::take(&mut self.carried);
        data.extend_from_slice(chunk);
        let (mut kept, mut met, mut at) = (Vec::with_capacity(data.len()), 0, 0);
        while let Some(found) = data[at..].windows(MARKER.len()).position(|window| window == MARKER) {
            kept.extend_from_slice(&data[at..at + found]);
            at += found + MARKER.len();
            met += 1;
        }
        // A tail that may be a marker's beginning waits for the next read.
        let rest = &data[at..];
        let carried = (1..MARKER.len().min(rest.len() + 1)).rev().find(|&length| rest.ends_with(&MARKER[..length])).unwrap_or(0);
        kept.extend_from_slice(&rest[..rest.len() - carried]);
        self.carried = rest[rest.len() - carried..].to_vec();
        (kept, met)
    }
}

/// The log file, kept within `budget` bytes: past it, the oldest content goes, down to
/// the newest half (from a line's start), behind a line saying how much went. Each
/// write is one pipe read, far smaller than the budget.
struct Budgeted { file: File, size: u64, budget: u64, dropped: u64, notice: u64 }

impl Budgeted {
    fn new(file: File, budget: u64) -> Self { Self { file, size: 0, budget, dropped: 0, notice: 0 } }

    fn write(&mut self, bytes: &[u8]) -> std::io::Result<()> {
        if self.size + bytes.len() as u64 > self.budget { self.trim()?; }
        self.file.write_all(bytes)?;
        self.size += bytes.len() as u64;
        Ok(())
    }

    fn trim(&mut self) -> std::io::Result<()> {
        // The newest half from a line's start: read from the byte before it, to see whether one starts there.
        let from = self.size.saturating_sub(self.budget / 2).max(self.notice);
        let look = if from > self.notice { from - 1 } else { from };
        self.file.seek(SeekFrom::Start(look))?;
        let mut kept = Vec::with_capacity((self.size - look) as usize);
        (&self.file).take(self.size - look).read_to_end(&mut kept)?;
        if look < from { kept.drain(..kept.iter().position(|&byte| byte == b'\n').map_or(kept.len(), |line| line + 1)); }
        // The notice of an earlier trim is not the log's own: it is replaced, not counted.
        let gone = self.size - self.notice - kept.len() as u64;
        self.dropped += gone;
        let said = format!("[{}] [run log] {gone} bytes before this line were dropped ({} in this run) to keep this log within {} MiB, the most of a log the diagnostics archive carries; the newest follow.\n",
            zlc_core::process::local_time(zlc_core::journal::now_us()), self.dropped, self.budget >> 20);
        self.file.set_len(0)?;
        self.file.seek(SeekFrom::Start(0))?;
        self.file.write_all(said.as_bytes())?;
        self.file.write_all(&kept)?;
        self.notice = said.len() as u64;
        self.size = self.notice + kept.len() as u64;
        Ok(())
    }
}

impl Routed {
    /// A failure that ends the program, said where the operator sees it: a message
    /// box when no caller reads stderr and the windows are on screen.
    pub fn fatal(&self, error: &str) {
        eprintln!("{}", serde_json::json!({"type": "desktop-error", "error": error}));
        drain();
        if self.errors_read || std::env::var_os("ZLC_V3_WINDOW_AT").is_some() { return; }
        let log = self.log.as_ref().map(|path| format!("\n\nLog: {}", path.display())).unwrap_or_default();
        let text: Vec<u16> = format!("Zou lab control stopped: {error}{log}").encode_utf16().chain([0]).collect();
        let title: Vec<u16> = "Zou lab control".encode_utf16().chain([0]).collect();
        use windows_sys::Win32::UI::WindowsAndMessaging::{MB_ICONERROR, MB_OK, MB_SETFOREGROUND, MessageBoxW};
        unsafe { MessageBoxW(std::ptr::null_mut(), text.as_ptr(), title.as_ptr(), MB_OK | MB_ICONERROR | MB_SETFOREGROUND) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_run_log_keeps_its_newest_half_past_its_budget_and_says_what_went() {
        let path = std::env::temp_dir().join(format!("zlc-run-log-budget-{}.log", std::process::id()));
        let file = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(true).open(&path).unwrap();
        let mut log = Budgeted::new(file, 1000);
        for line in 0..100 { log.write(format!("line {line:03} ........\n").as_bytes()).unwrap(); }
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.len() as u64 <= 1000 && log.size == text.len() as u64, "{} bytes", text.len());
        let (said, rest) = text.split_once('\n').unwrap();
        assert!(said.contains("[run log]") && said.contains("bytes before this line were dropped") && said.contains("in this run"), "{said}");
        assert!(rest.starts_with("line ") && rest.ends_with("line 099 ........\n"), "whole lines, the newest last: {rest}");
        let dropped: u64 = said.split(" in this run").next().unwrap().rsplit('(').next().unwrap().trim().parse().unwrap();
        assert_eq!(dropped as usize + rest.len(), 100 * "line 000 ........\n".len(), "every byte is kept or counted as dropped");
        assert!(rest.len() >= 500 - "line 000 ........\n".len(), "the newest half is kept: {} bytes", rest.len());
        drop(log);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn drain_returns_once_everything_written_before_it_is_in_the_file() {
        let path = std::env::temp_dir().join(format!("zlc-run-log-drain-{}.log", std::process::id()));
        let file = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(true).open(&path).unwrap();
        let (reader, mut writer) = std::io::pipe().unwrap();
        assert!(DRAIN.set(Drain { marker: Mutex::new(writer.try_clone().unwrap()), asked: Mutex::new((0, 0)), met: Condvar::new() }).is_ok());
        let mut log = Budgeted::new(file, BUDGET);
        // The reader lives as long as the test process (the drain's write end stays open).
        std::thread::spawn(move || copy(reader, &mut log));
        for line in 0..1000 { writeln!(writer, "line {line}").unwrap(); }
        drain();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.ends_with("line 999\n") && !text.contains('\x1b'), "everything before the drain, and no marker");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn drain_markers_are_taken_out_even_split_across_reads() {
        let mut markers = Markers::default();
        let (half, rest) = MARKER.split_at(5);
        let (kept, met) = markers.take(&[b"before ".as_slice(), half].concat());
        assert_eq!((kept.as_slice(), met), (b"before ".as_slice(), 0));
        let (kept, met) = markers.take(&[rest, b"after\x1b".as_slice()].concat());
        assert_eq!((kept.as_slice(), met), (b"after".as_slice(), 1), "a lone escape waits: it may begin a marker");
        let (kept, met) = markers.take(b"x");
        assert_eq!((kept.as_slice(), met), (b"\x1bx".as_slice(), 0));
        let (kept, met) = markers.take(&[MARKER, MARKER].concat());
        assert_eq!((kept.len(), met), (0, 2));
    }
}
