//! The flight journal: what the application did, in the order it did it, kept for the last hour.
//!
//! A fault that shows once (a frame paired with the wrong shot, a panel drawing an older run) leaves
//! nothing to read afterwards. The journal is that reading: every part of the application says the
//! facts it acts on -- a command taken, a run begun or ended, a board command and its status, a
//! publication's ordinals, a camera frame's arrival, the reads a panel was drawn from -- as one line
//! each, stamped by one clock and one counter. It records identities and times, never scientific
//! payloads: a line names a frame, it does not hold one.
//!
//! One line is one JSON object: `us` (microseconds since the Unix epoch), `n` (the order notes were
//! taken in, across every thread), `src` (who says it), `ev` (what happened), and that event's own
//! facts. A fact written elsewhere and brought here later (a plugin process's) keeps the time it was
//! observed at in `us` and is ordered by `n` as it arrived.
//!
//! Two kinds of line are kept: what happened (a note), and what the application's state then was. State is
//! said as its changes, and whole at the start of every segment (`at_segment_start`), so each segment
//! file can be read without the one before it.
//!
//! Lines are written to segment files of [`SEGMENT`] each (`journal-<start us>.jsonl`), a second or
//! so after they are said, so a crash or a freeze leaves the journal on disk. Segments older than
//! [`KEPT`] are removed, and the oldest first when the folder passes [`FOLDER_LIMIT`]. Nothing is
//! recorded before [`open`]: a program that never opens it (a test, a helper) pays one atomic read.
//!
//! A few facts must outlive the hour: how the process began and how each device came to be as it is
//! (opened with what, how long it took, what it said of itself, why it failed or closed), and each
//! plugin process's start and end. Said with [`lifecycle`], such a line goes into the rolling journal
//! in its order like any other, and also into this process's session file (`session-<start us>.jsonl`),
//! which nothing rotates: it holds the whole life of the process ([`SESSION_LIMIT`]).
use std::collections::VecDeque;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::Map;
/// What a note's facts are written with, for a crate that has no other use of JSON.
pub use serde_json::{json, Value};

/// How long a note is kept.
pub const KEPT: Duration = Duration::from_secs(60 * 60);
/// How much time one segment file holds.
pub const SEGMENT: Duration = Duration::from_secs(5 * 60);
/// How often said lines reach their file.
const WRITE_EVERY: Duration = Duration::from_millis(500);
/// The most the folder holds; past it the oldest segments go, whatever their age.
const FOLDER_LIMIT: u64 = 512 << 20;
/// The most that waits for the writer; past it notes are counted, not kept (the writer says how many).
const WAITING_LIMIT: usize = 16 << 20;
/// A text fact longer than this is cut and says how long it was.
const TEXT_LIMIT: usize = 2048;
/// An array fact longer than this keeps its first items and says how many it had.
const ARRAY_LIMIT: usize = 64;
/// A text fact of a lifecycle line is cut only past this: a plugin's Python traceback through a deep
/// import chain is a few kilobytes, and it is the one record of why a device did not open.
const LIFECYCLE_TEXT_LIMIT: usize = 64 << 10;
/// The most one process says to its session file; past it, lifecycle lines go to the rolling journal
/// alone and the session says so once. A device's open, close or failure is one line, a few per device
/// per Init or Apply: a day's work is some hundred kilobytes with its tracebacks. A process that says
/// this much is in a loop (a device failing and opened again), whose beginning is what explains it.
pub const SESSION_LIMIT: usize = 8 << 20;
/// How many session files the folder keeps (this process's among them): a process that ended badly
/// is still there when the next one exports. Eight: a working day of restarts.
const SESSIONS_KEPT: usize = 8;

struct Journal { folder: PathBuf, session: PathBuf, waiting: Mutex<Waiting> }
#[derive(Default)]
struct Waiting { lines: Vec<u8>, dropped: u64, session: Vec<u8>, session_said: usize, session_full: bool }

static OPEN: AtomicBool = AtomicBool::new(false);
static ORDER: AtomicU64 = AtomicU64::new(0);
static JOURNAL: OnceLock<Journal> = OnceLock::new();

/// Microseconds since the Unix epoch, now.
pub fn now_us() -> u64 { SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |elapsed| elapsed.as_micros() as u64) }

/// Whether notes are being kept.
pub fn is_open() -> bool { OPEN.load(Ordering::Relaxed) }

/// Begin keeping notes in `folder` (once per process; a second call keeps the first folder).
pub fn open(folder: &Path) -> std::io::Result<()> {
    if JOURNAL.get().is_some() { return Ok(()); }
    std::fs::create_dir_all(folder)?;
    let start = now_us();
    let session = folder.join(format!("session-{start}.jsonl"));
    if JOURNAL.set(Journal { folder: folder.to_owned(), session, waiting: Mutex::default() }).is_err() { return Ok(()); }
    prune_sessions(folder, start);
    OPEN.store(true, Ordering::Release);
    std::thread::Builder::new().name("zlc-journal".into()).spawn(write_forever)?;
    Ok(())
}

/// Say that `event` happened to `source`, with its facts (an object's fields; any other value is `value`).
pub fn note(source: &str, event: &str, facts: Value) { if is_open() { keep(now_us(), source, event, facts); } }

/// As [`note`], with facts that are only gathered while the journal is open.
pub fn note_with(source: &str, event: &str, facts: impl FnOnce() -> Value) { if is_open() { keep(now_us(), source, event, facts()); } }

/// As [`note`], with `fields` -- the text of a JSON object -- kept whole: a state's changes are what they
/// are, and a change cut short could not be applied by whoever reads the journal.
pub fn note_fields(source: &str, event: &str, fields: &str) {
    if !is_open() { return; }
    let Some(journal) = JOURNAL.get() else { return };
    let line = whole_line(source, event, fields);
    let mut waiting = journal.waiting.lock().unwrap_or_else(|error| error.into_inner());
    if waiting.lines.len() > WAITING_LIMIT { waiting.dropped += 1; return; }
    waiting.lines.extend_from_slice(&line);
}

/// One line with `fields` (a JSON object's text) after its stamp.
fn whole_line(source: &str, event: &str, fields: &str) -> Vec<u8> {
    let stamp = serde_json::json!({"us": now_us(), "n": ORDER.fetch_add(1, Ordering::Relaxed), "src": source, "ev": event}).to_string();
    let inner = fields.trim().strip_prefix('{').and_then(|rest| rest.strip_suffix('}')).unwrap_or("").trim();
    let mut line = stamp.into_bytes();
    if !inner.is_empty() {
        line.pop();
        line.push(b',');
        line.extend_from_slice(inner.as_bytes());
        line.push(b'}');
    }
    line.push(b'\n');
    line
}

/// What opens every segment: `opening` is called when one begins and says, through the function it is
/// handed (source, event, the text of a JSON object), the lines that make the segment readable alone --
/// the application's whole state. They are stamped as they are said and written before anything else.
pub fn at_segment_start(opening: impl Fn(&mut dyn FnMut(&str, &str, &str)) + Send + Sync + 'static) {
    *OPENING.lock().unwrap_or_else(|error| error.into_inner()) = Some(Box::new(opening));
}

type Opening = Box<dyn Fn(&mut dyn FnMut(&str, &str, &str)) + Send + Sync>;
static OPENING: Mutex<Option<Opening>> = Mutex::new(None);

/// A fact observed at `us` by another process, brought here now.
pub fn note_at(us: u64, source: &str, event: &str, facts: Value) { if is_open() { keep(us, source, event, facts); } }

/// As [`note`], for a fact of how the process or a device came to be as it is (see the module's
/// words): kept in the rolling journal in its order and in this process's session file, for the
/// process's whole life. Its texts are cut only past [`LIFECYCLE_TEXT_LIMIT`] (a traceback stays whole).
pub fn lifecycle(source: &str, event: &str, facts: Value) {
    if !is_open() { return; }
    let Some(journal) = JOURNAL.get() else { return };
    let line = line(now_us(), source, event, facts, LIFECYCLE_TEXT_LIMIT);
    let mut waiting = journal.waiting.lock().unwrap_or_else(|error| error.into_inner());
    if waiting.session_said + line.len() <= SESSION_LIMIT {
        waiting.session_said += line.len();
        waiting.session.extend_from_slice(&line);
    } else if !waiting.session_full {
        waiting.session_full = true;
        let full = self::line(now_us(), "journal", "session_full", serde_json::json!({"limit_bytes": SESSION_LIMIT}), TEXT_LIMIT);
        waiting.session.extend_from_slice(&full);
    }
    if waiting.lines.len() > WAITING_LIMIT { waiting.dropped += 1; return; }
    waiting.lines.extend_from_slice(&line);
}

/// One line: its stamp, then the event's facts bounded at `text_limit`.
fn line(us: u64, source: &str, event: &str, facts: Value, text_limit: usize) -> Vec<u8> {
    let mut line = Map::new();
    line.insert("us".into(), us.into());
    line.insert("n".into(), ORDER.fetch_add(1, Ordering::Relaxed).into());
    line.insert("src".into(), source.into());
    line.insert("ev".into(), event.into());
    match bounded(facts, text_limit) {
        Value::Object(fields) => for (name, value) in fields { line.entry(name).or_insert(value); },
        Value::Null => {}
        other => { line.insert("value".into(), other); }
    }
    let mut bytes = serde_json::to_vec(&Value::Object(line)).unwrap_or_default();
    bytes.push(b'\n');
    bytes
}

fn keep(us: u64, source: &str, event: &str, facts: Value) {
    let Some(journal) = JOURNAL.get() else { return };
    let line = line(us, source, event, facts, TEXT_LIMIT);
    let mut waiting = journal.waiting.lock().unwrap_or_else(|error| error.into_inner());
    if waiting.lines.len() > WAITING_LIMIT { waiting.dropped += 1; return; }
    waiting.lines.extend_from_slice(&line);
}

/// `value` with nothing in it large: the journal names things, it does not hold them.
fn bounded(value: Value, text_limit: usize) -> Value {
    match value {
        Value::String(text) if text.len() > text_limit => {
            let cut = (0..=text_limit).rev().find(|at| text.is_char_boundary(*at)).unwrap_or(0);
            Value::String(format!("{}… ({} bytes)", &text[..cut], text.len()))
        }
        Value::Array(items) if items.len() > ARRAY_LIMIT => {
            let count = items.len();
            let mut kept: Vec<Value> = items.into_iter().take(ARRAY_LIMIT).map(|item| bounded(item, text_limit)).collect();
            kept.push(Value::String(format!("… ({count} items)")));
            Value::Array(kept)
        }
        Value::Array(items) => Value::Array(items.into_iter().map(|item| bounded(item, text_limit)).collect()),
        Value::Object(fields) => Value::Object(fields.into_iter().map(|(name, value)| (name, bounded(value, text_limit))).collect()),
        other => other,
    }
}

/// The words that mark a setting as a secret. A field's name is split into words at `_`, `-`, `.`,
/// spaces and a lower-to-upper case change (`apiKey` is `api`, `key`); a value whose name has one of
/// these words is said as [`REDACTED`]. The parameter schema has no kind of its own for a secret, so
/// its name is what says it; `key` counts only as a whole word (`keyboard`, `keys` are not secrets).
pub const SECRET_WORDS: &[&str] = &["password", "passwd", "passphrase", "pwd", "secret", "token", "key", "apikey", "credential", "credentials", "auth"];
/// What a secret's value is said as: that there was one, never what it was.
pub const REDACTED: &str = "<redacted>";

/// Whether a field's name marks it as a secret ([`SECRET_WORDS`]).
pub fn secret_name(name: &str) -> bool {
    let mut words: Vec<String> = Vec::new();
    let mut word = String::new();
    let mut lower_before = false;
    for character in name.chars() {
        if !character.is_alphanumeric() { if !word.is_empty() { words.push(std::mem::take(&mut word)); } lower_before = false; continue; }
        if character.is_uppercase() && lower_before && !word.is_empty() { words.push(std::mem::take(&mut word)); }
        lower_before = character.is_lowercase() || character.is_ascii_digit();
        word.extend(character.to_lowercase());
    }
    if !word.is_empty() { words.push(word); }
    words.iter().any(|word| SECRET_WORDS.contains(&word.as_str()))
}

/// `value` with every field a secret's name marks ([`secret_name`]) said as [`REDACTED`], at any depth
/// (a table's rows too): settings as a diagnostics archive may carry them.
pub fn redacted(value: &Value) -> Value {
    match value {
        Value::Object(fields) => Value::Object(fields.iter().map(|(name, value)| {
            (name.clone(), if secret_name(name) && !value.is_null() { Value::String(REDACTED.into()) } else { redacted(value) })
        }).collect()),
        Value::Array(items) => Value::Array(items.iter().map(redacted).collect()),
        other => other.clone(),
    }
}

fn session_start(path: &Path) -> Option<u64> {
    path.file_name()?.to_str()?.strip_prefix("session-")?.strip_suffix(".jsonl")?.parse().ok()
}

/// The session files of `folder`, oldest first.
fn sessions_in(folder: &Path) -> Vec<(u64, PathBuf)> {
    let mut found: Vec<(u64, PathBuf)> = std::fs::read_dir(folder).into_iter().flatten().flatten()
        .filter_map(|entry| { let path = entry.path(); Some((session_start(&path)?, path)) }).collect();
    found.sort();
    found
}

/// Keep the newest [`SESSIONS_KEPT`] session files, the one begun at `current` among them.
fn prune_sessions(folder: &Path, current: u64) {
    let earlier: Vec<PathBuf> = sessions_in(folder).into_iter().filter(|(start, _)| *start != current).map(|(_, path)| path).collect();
    let over = (earlier.len() + 1).saturating_sub(SESSIONS_KEPT);
    for path in earlier.into_iter().take(over) { let _ = std::fs::remove_file(path); }
}

fn segment_start(path: &Path) -> Option<u64> {
    path.file_name()?.to_str()?.strip_prefix("journal-")?.strip_suffix(".jsonl")?.parse().ok()
}

/// The segment files of `folder`, oldest first.
fn segments_in(folder: &Path) -> Vec<(u64, PathBuf)> {
    let mut found: Vec<(u64, PathBuf)> = std::fs::read_dir(folder).into_iter().flatten().flatten()
        .filter_map(|entry| { let path = entry.path(); Some((segment_start(&path)?, path)) }).collect();
    found.sort();
    found
}

/// Remove what is older than [`KEPT`], then the oldest while the folder is over its limit (never `current`).
fn prune(folder: &Path, current: u64) {
    let mut held: VecDeque<(u64, PathBuf, u64)> = segments_in(folder).into_iter()
        .map(|(start, path)| { let size = std::fs::metadata(&path).map_or(0, |file| file.len()); (start, path, size) }).collect();
    let oldest_kept = now_us().saturating_sub((KEPT + SEGMENT).as_micros() as u64);
    let mut total: u64 = held.iter().map(|(_, _, size)| size).sum();
    while let Some((start, path, size)) = held.front() {
        if *start == current || (*start >= oldest_kept && total <= FOLDER_LIMIT) { break; }
        let _ = std::fs::remove_file(path);
        total -= size;
        held.pop_front();
    }
}

struct Writer { start: u64, file: std::fs::File }

impl Writer {
    fn begin(folder: &Path) -> std::io::Result<Self> {
        let start = now_us();
        let file = std::fs::OpenOptions::new().create(true).append(true).open(folder.join(format!("journal-{start}.jsonl")))?;
        prune(folder, start);
        Ok(Self { start, file })
    }
}

/// Write what waits; a new segment once the current one has held its time.
fn write_waiting(journal: &Journal, writer: &mut Option<Writer>) {
    let (lines, dropped, session) = {
        let mut waiting = journal.waiting.lock().unwrap_or_else(|error| error.into_inner());
        (std::mem::take(&mut waiting.lines), std::mem::take(&mut waiting.dropped), std::mem::take(&mut waiting.session))
    };
    if !session.is_empty() {
        // Appended as it is said; a disk that refuses loses these lines and says so, as for a segment.
        let written = std::fs::OpenOptions::new().create(true).append(true).open(&journal.session)
            .and_then(|mut file| file.write_all(&session).and_then(|()| file.flush()));
        if written.is_err() { journal.waiting.lock().unwrap_or_else(|error| error.into_inner()).dropped += 1; }
    }
    if dropped > 0 { keep(now_us(), "journal", "dropped", serde_json::json!({"notes": dropped})); }
    if lines.is_empty() { return; }
    let mut opening = Vec::new();
    if writer.as_ref().is_none_or(|writer| now_us().saturating_sub(writer.start) >= SEGMENT.as_micros() as u64) {
        *writer = Writer::begin(&journal.folder).ok();
        if let Some(say) = OPENING.lock().unwrap_or_else(|error| error.into_inner()).as_ref() {
            say(&mut |source, event, fields| opening.extend_from_slice(&whole_line(source, event, fields)));
        }
    }
    // A disk that refuses is not the application's fault to stop for: the lines are lost and said so.
    let written = writer.as_mut().is_some_and(|writer| writer.file.write_all(&opening).and_then(|()| writer.file.write_all(&lines))
        .and_then(|()| writer.file.flush()).is_ok());
    if !written { *writer = None; journal.waiting.lock().unwrap_or_else(|error| error.into_inner()).dropped += 1; }
}

static WRITER: Mutex<Option<Writer>> = Mutex::new(None);

fn write_forever() {
    let journal = JOURNAL.get().expect("opened");
    loop {
        std::thread::sleep(WRITE_EVERY);
        write_waiting(journal, &mut WRITER.lock().unwrap_or_else(|error| error.into_inner()));
    }
}

/// Everything said so far is on disk; the segment files, oldest first. Empty while the journal is not open.
pub fn segments() -> Vec<PathBuf> {
    let Some(journal) = JOURNAL.get() else { return Vec::new() };
    write_waiting(journal, &mut WRITER.lock().unwrap_or_else(|error| error.into_inner()));
    segments_in(&journal.folder).into_iter().map(|(_, path)| path).collect()
}

/// Everything said so far is on disk; this process's session file (absent while it has said none),
/// and the earlier processes' still kept, oldest first. None while the journal is not open.
pub fn sessions() -> Option<(PathBuf, Vec<PathBuf>)> {
    let journal = JOURNAL.get()?;
    write_waiting(journal, &mut WRITER.lock().unwrap_or_else(|error| error.into_inner()));
    let earlier = sessions_in(&journal.folder).into_iter().map(|(_, path)| path).filter(|path| *path != journal.session).collect();
    Some((journal.session.clone(), earlier))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_fact_is_named_never_held() {
        let long = "é".repeat(TEXT_LIMIT);
        let Value::Object(kept) = bounded(json!({"plan": long, "rows": (0..200).collect::<Vec<_>>(), "inner": {"text": "x".repeat(10_000)}, "n": 3}), TEXT_LIMIT) else { panic!() };
        let plan = kept["plan"].as_str().unwrap();
        assert!(plan.len() < TEXT_LIMIT + 40 && plan.ends_with(&format!("({} bytes)", long.len())), "{}", plan.len());
        assert_eq!((kept["rows"].as_array().unwrap().len(), kept["rows"][ARRAY_LIMIT].as_str()), (ARRAY_LIMIT + 1, Some("… (200 items)")));
        assert!(kept["inner"]["text"].as_str().unwrap().ends_with("(10000 bytes)"));
        assert_eq!(kept["n"], 3);
    }

    #[test]
    fn whole_fields_follow_the_stamp_uncut() {
        let long = "x".repeat(TEXT_LIMIT * 3);
        let line = whole_line("app", "state", &serde_json::json!({"changes": [[["logic", 0, "name"], long]], "request": 7}).to_string());
        let read: Value = serde_json::from_slice(&line).unwrap();
        assert_eq!((&read["src"], &read["ev"], &read["request"]), (&json!("app"), &json!("state"), &json!(7)));
        assert_eq!(read["changes"][0][1].as_str().unwrap().len(), TEXT_LIMIT * 3, "a change is kept whole");
        assert!(read["us"].is_u64() && read["n"].is_u64());
        let bare: Value = serde_json::from_slice(&whole_line("app", "state", "{}")).unwrap();
        assert_eq!(bare.as_object().unwrap().len(), 4, "no fields: the stamp alone");
    }

    #[test]
    fn old_segments_go_and_the_current_one_stays() {
        let folder = std::env::temp_dir().join(format!("zlc-journal-test-{}", now_us()));
        std::fs::create_dir_all(&folder).unwrap();
        let now = now_us();
        let hour = KEPT.as_micros() as u64;
        let (stale, recent) = (now - 3 * hour, now - hour / 2);
        for start in [stale, recent, now] { std::fs::write(folder.join(format!("journal-{start}.jsonl")), b"{}\n").unwrap(); }
        std::fs::write(folder.join("notes.txt"), b"not a segment").unwrap();
        prune(&folder, now);
        let left: Vec<u64> = segments_in(&folder).into_iter().map(|(start, _)| start).collect();
        assert_eq!(left, [recent, now]);
        assert!(folder.join("notes.txt").is_file(), "only its own files are touched");
        std::fs::remove_dir_all(&folder).unwrap();
    }

    #[test]
    fn sessions_outlive_the_hour_and_the_newest_are_kept() {
        let folder = std::env::temp_dir().join(format!("zlc-journal-sessions-{}", now_us()));
        std::fs::create_dir_all(&folder).unwrap();
        let now = now_us();
        let hour = KEPT.as_micros() as u64;
        // Ten processes, the oldest started a day ago; each left its session and a segment.
        let starts: Vec<u64> = (0..10).map(|index| now - 24 * hour + index * hour).collect();
        for start in &starts {
            std::fs::write(folder.join(format!("session-{start}.jsonl")), b"{}\n").unwrap();
            std::fs::write(folder.join(format!("journal-{start}.jsonl")), b"{}\n").unwrap();
        }
        prune(&folder, now);
        assert!(segments_in(&folder).is_empty(), "every segment is past the hour");
        assert_eq!(sessions_in(&folder).len(), 10, "rotation never touches a session");
        prune_sessions(&folder, starts[9]);
        let left: Vec<u64> = sessions_in(&folder).into_iter().map(|(start, _)| start).collect();
        assert_eq!(left, starts[10 - SESSIONS_KEPT..], "the newest {SESSIONS_KEPT}, this process's among them");
        std::fs::remove_dir_all(&folder).unwrap();
    }

    #[test]
    fn a_secret_is_said_to_be_there_never_what_it_is() {
        for name in ["password", "api_key", "apiKey", "AccessToken", "client-secret", "auth", "SMTP.Passwd", "key"] { assert!(secret_name(name), "{name}"); }
        for name in ["port", "keyboard", "device_keys", "monkey", "author", "serial", "exposure_seconds", "tokens_per_s"] { assert!(!secret_name(name), "{name}"); }
        let said = redacted(&json!({"host": "10.0.0.2", "password": "hunter2", "empty_token": null,
            "rows": [{"name": "a", "apiKey": "k1"}], "inner": {"secret": 7}}));
        assert_eq!(said, json!({"host": "10.0.0.2", "password": REDACTED, "empty_token": null,
            "rows": [{"name": "a", "apiKey": REDACTED}], "inner": {"secret": REDACTED}}));
    }

    #[test]
    fn notes_reach_their_segment_in_the_order_they_were_said() {
        // The one journal of this test process.
        let folder = std::env::temp_dir().join(format!("zlc-journal-open-{}", now_us()));
        note("early", "before open", Value::Null);
        open(&folder).unwrap();
        note("board", "fire", json!({"command_id": 7, "status": "running"}));
        note_at(5, "plugin", "frame", json!({"ordinal": 0, "us": 99}));
        note("panel", "drawn", json!("p1"));
        let traceback = format!("Traceback (most recent call last):\n{}RuntimeError: no camera", "  File \"adapter.py\", line 1\n".repeat(200));
        lifecycle("plugins", "device_open", json!({"instance": "camera", "error": {"traceback": traceback}}));
        let text: String = segments().iter().map(|path| std::fs::read_to_string(path).unwrap()).collect();
        let lines: Vec<Value> = text.lines().map(|line| serde_json::from_str(line).unwrap()).collect();
        let said: Vec<(&str, &str)> = lines.iter().map(|line| (line["src"].as_str().unwrap(), line["ev"].as_str().unwrap())).collect();
        assert_eq!(said, [("board", "fire"), ("plugin", "frame"), ("panel", "drawn"), ("plugins", "device_open")], "nothing from before it opened");
        assert!(lines.windows(2).all(|pair| pair[0]["n"].as_u64() < pair[1]["n"].as_u64()));
        assert_eq!((&lines[0]["command_id"], &lines[1]["us"], &lines[1]["ordinal"], &lines[2]["value"]), (&json!(7), &json!(5), &json!(0), &json!("p1")),
                   "an event's own facts beside the line's, whose time and order no fact replaces");
        // The lifecycle line is in the rolling journal in its order, and in the session file, whole.
        let (session, _) = sessions().unwrap();
        let kept: Vec<Value> = std::fs::read_to_string(&session).unwrap().lines().map(|line| serde_json::from_str(line).unwrap()).collect();
        assert_eq!(kept.len(), 1, "only lifecycle facts are the session's");
        assert_eq!(kept[0], lines[3], "the same line, the same order");
        assert_eq!(kept[0]["error"]["traceback"], json!(traceback), "a traceback is kept whole");
        std::fs::remove_dir_all(&folder).ok();
    }
}
