//! The diagnostics archive: the last hour as the journal kept it, with what the board holds now.
//!
//! One file an operator hands to whoever looks into a fault: the journal's segments (what happened,
//! in order), the session's lifecycle (how the process began and each device was opened, failed or
//! closed, however long ago), each device's log as its window shows it, the board as it stands (its layout, the apparatus, the workbench's own state), the
//! release that was running, the newest program logs, and the text files the board names -- a
//! node's pulse, the Subpulses and the Config that pulse names in turn -- as they are now. It
//! names a calibration or a dataset the board uses without carrying it; the one data it does carry is
//! what the application keeps of recent publications within its budget (`data/`), when asked to.
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};
use zlc_archive::zip::ZipWriter;
use zlc_core::journal;

use crate::{Host, Workbench, workspace};

/// The workspace folder the archives are written to.
pub const FOLDER: &str = "diagnostics";
/// The newest program logs it carries, and the most of each.
const LOGS: usize = 3;
pub const LOG_LIMIT: u64 = 8 << 20;
/// The files the board names that it carries: JSON documents, each at most this large, at most this many.
const NAMED_LIMIT: u64 = 4 << 20;
const NAMED_FILES: usize = 128;

const README: &str = "\
Zou lab control v3 -- diagnostics archive

journal/journal-<start us>.jsonl
    What the application did, oldest file first, one JSON object per line:
      us   microseconds since the Unix epoch (UTC) when the fact was observed
      n    the order facts were taken in, across every thread and process
      src  who says it: app, ui, workbench, runtime, board, plugins, scan, camera, journal, ...
      ev   what happened
    and that event's own fields. Sort by `n` for the order the application saw; by `us` for
    wall time (a plugin's fact keeps the time its own process observed it).
    Identities and times only: a line names a frame or a publication, it never holds one.

    runtime begin/commit/seal/retire   a run's publications: `out` is signal -> ordinal, `of` its parents
    workbench launch/event/refused     node runs: the configuration launched, each plugin fact, each refusal
    app action/drawn/report            commands a window sent, the reads each panel was drawn from,
                                       the console's line whenever it changed
    app state_snapshot / state         the application's whole state at the start of each file, then every
                                       change: `changes` is a list of [path, value] (set), [path] (delete a
                                       key) and [path, items, start, removed] (splice an array); a line with
                                       `request` is that command's reply (`result`, `error`). A read-only
                                       command that succeeds is said by its reply alone
    plugins device_repeated            the same device call made again `times` with nothing going wrong
                                       (a capture polling its camera), said when something else follows
    board load/fire/report/safe        the pulse board; a report's `elapsed_seconds` after its fire is
                                       when the board itself was done
    scan segment_armed/segment_filled  a scan's segments; `dropped` is what arrived before one was armed
    plugins call/device                what the core asked of a plugin, and a run of a device
    camera arm/finish/frame            a camera's captures and each frame as its measurement read it
    camera aligned_to/frame_dropped/   a measurement that follows a trigger channel: the play it aligned to, a
      shot_incomplete/shot_abandoned   frame no trigger asked for, a shot one or more of whose frames never came
    ui click/key/popup/error           what the operator did in a window
    ui long_tasks                      the tasks the browser reported as holding a page's thread (50 ms and
                                       over, its definition) since the page's last batch: count, total, longest
    system machine / resources         once at start the machine's processors and the sampling period; then per
                                       period `processes` -- the application and its descendants (plugin and
                                       page processes; `plugins process_started` says which pid a plugin is), each with
                                       `cpu` (processors used since the last line: 1 is one whole processor) and
                                       its working and private memory -- `machine` (its processors' busy share,
                                       free memory) and `loop`: the event loop's turns in the period, their
                                       time together and the longest with what it handled
    and the lifecycle lines below, each in its place.

journal/session.jsonl
    How this process began and how each device came to be as it is, for the process's whole life (the
    journal above keeps an hour): the lifecycle lines alone, the same lines (same `n`) the journal holds.
    Texts are cut only past 64 KB, so a plugin's Python traceback stays whole; past 8 MB a process says
    `journal session_full` and further lines are in the hourly journal alone. A setting whose name has
    a secret's word (password, secret, token, key, auth, credential, ...) is written <redacted>.

    app started                        pid, release, machine, the program's arguments
    system machine                     the machine's processors
    plugins process_started / ended    a plugin process (`key`: the one shared process, or `type:<id>`), its
      / process_not_started            pid and start time; at its end the reason, exit code, whether it was
                                       asked to end (`expected`; else it crashed), its instances, its last log
    workbench devices_apply            an Init, Apply, Close or shut down: rows to open (`retrying`: ones that
                                       had failed), to close, to close and open again as they now stand
    workbench device_resolving /       a row whose port is left blank, found before it opens
      device_resolved
    workbench device_open              a row opened: instance, role, type_id, the parameters it opens with
                                       (its type's defaults filled in), the simulation for a simulated type,
                                       `attempt` (its how-many-th open in this process); `outcome: failed`
                                       with `error` when refused before reaching the device
    plugins device_open                the plugin side of that open: process, pid, `ms`, outcome, and on a
                                       failure `error` {kind, type, message, traceback}
    workbench device_opened            `ms` since it was asked, and `device_info`: what the device says of
                                       itself (model, serial, firmware, driver or SDK version), when its type
                                       says it
    workbench device_failed            `phase`: resolve, open (it did not open) or lost (it was open), `error`;
                                       `looking_again` when its named serial port could not be used and its
                                       type's resolve looks for it on the ports there are
    workbench device_close / closed    a close asked, and its outcome with `ms`;
    plugins device_close /             the plugin side of a close; a device whose call overran its deadline
      device_unresponsive
    workbench devices_settled          the Device Manager's line once all of it is done, the rows open

journal/sessions/session-<start us>.jsonl
    The same, of earlier processes (the newest eight sessions are kept): how one that ended badly ended.

state.json     the workbench's state when this was exported (nodes, runs, devices, board)
layout.json     the board as Save would write it
apparatus.json  the apparatus as it is loaded now, whichever file it was read from
release.json    the release that was running
logs/           the newest program logs
logs/devices/<instance>.log
                each device seen this session (open, failed or closed): the lines its Log window shows,
                its newest 2000
data/           the data of recent publications, as far back as the application's budget for it holds --
                every signal for the same length of time: one .npy per signal and run, its events stacked
                in commit order; data/index.json names each file's signal, run, ordinals and commit times,
                by which a `runtime commit` line of the journal finds its event. Absent when the archive
                was exported without data, or none was kept
files/          the JSON files the board names (a node's pulse, the Subpulses and Config it names),
                as they were when this was exported; files.json says which path each one is
";

/// What one export wrote.
pub struct Exported { pub path: PathBuf, pub segments: usize, pub bytes: u64 }

impl Workbench {
    /// Take what the archive holds of the board now (its state, layout and apparatus, the recent
    /// data kept), and write the archive into the workspace's diagnostics folder off the owner
    /// turn; `diagnostics_exported` says where.
    pub(crate) fn export_diagnostics(&mut self, with_data: bool, request: u64, host: &dyn Host) {
        let board = Board { state: Value::Object(self.state(host)), layout: self.layout_document(host), apparatus: self.apparatus.tree(),
            data: with_data.then(|| host.diagnostic_data()), device_logs: self.device_logs() };
        let folder = self.workspace.root.join(FOLDER);
        let (root, ticket) = (self.workspace.root.clone(), self.ticket(crate::Ticket::Diagnostics { request }));
        self.plugins.work(ticket, crate::Lane::Export, Box::new(move || board.export(&folder, &root)));
    }

    /// Every device seen this session -- opened, failed, closed, or one whose lines came another way --
    /// with what its Log window shows: (instance, lines).
    fn device_logs(&self) -> Vec<(String, Vec<String>)> {
        let mut seen: BTreeSet<&String> = self.device_opens.keys().collect();
        seen.extend(self.logs.keys().filter(|instance| !instance.starts_with(zlc_core::control::NODE_INSTANCE)));
        seen.into_iter().map(|instance| (instance.clone(), self.device_log(instance, crate::LOG_LINES))).collect()
    }

    /// The archive was written (or why not): said on the console's line.
    pub(crate) fn diagnostics_exported(&mut self, exported: Result<Value, String>) -> Result<Value, String> {
        let exported = exported?;
        self.console_report = (format!("diagnostics exported to {}", exported["path"].as_str().unwrap_or_default()), "task");
        Ok(exported)
    }
}

/// What the owner turn takes of the board for an archive; the rest is read when it is written.
struct Board { state: Value, layout: Value, apparatus: Value, data: Option<crate::DiagnosticData>, device_logs: Vec<(String, Vec<String>)> }

impl Board {
    /// Write the archive into `folder`: its path and what it holds.
    fn export(self, folder: &Path, workspace: &Path) -> Result<Value, String> {
        let stamp: String = workspace::utc_now().chars().filter(char::is_ascii_digit).take(14).collect();
        std::fs::create_dir_all(folder).map_err(|error| format!("{}: {error}", folder.display()))?;
        let path = folder.join(format!("zlc-diagnostics-{}-{}.zip", &stamp[..8.min(stamp.len())], &stamp[8.min(stamp.len())..]));
        let mut members: Vec<(String, Vec<u8>)> = vec![("README.txt".into(), README.as_bytes().to_vec())];
        let pretty = |value: &Value| serde_json::to_vec_pretty(value).unwrap_or_default();
        members.push(("state.json".into(), pretty(&self.state)));
        members.push(("layout.json".into(), pretty(&self.layout)));
        // The apparatus as it is loaded and edited now, whichever file it came from.
        members.push(("apparatus.json".into(), pretty(&self.apparatus)));
        if let Ok(bytes) = std::fs::read(zlc_core::checkout::path("bin/release.json")) { members.push(("release.json".into(), bytes)); }
        members.extend(newest_logs(&zlc_core::checkout::local("logs")));
        members.extend(device_log_members(self.device_logs));
        let named = named_files(&[&members[1].1, &members[2].1], workspace);
        let files = named.len();
        members.push(("files.json".into(), pretty(&Value::Object(named.iter().map(|(member, path, _)| (member.clone(), json!(path))).collect()))));
        members.extend(named.into_iter().map(|(member, _, bytes)| (member, bytes)));
        // The data of the publications the journal names, as far back as the application keeps it.
        let data = self.data.map(|data| data()).unwrap_or_default();
        let data_files = data.len();
        members.extend(data);
        journal::note("workbench", "diagnostics_exported", json!({"path": path}));
        members.extend(session_members(journal::sessions()));
        let exported = write(&path, &journal::segments(), &members).map_err(|error| format!("{}: {error}", path.display()))?;
        Ok(json!({"path": exported.path, "segments": exported.segments, "files": files, "data_files": data_files, "bytes": exported.bytes}))
    }
}

/// The newest [`LOGS`] program logs, each one's last [`LOG_LIMIT`] bytes.
fn newest_logs(folder: &Path) -> Vec<(String, Vec<u8>)> {
    let mut logs: Vec<PathBuf> = std::fs::read_dir(folder).into_iter().flatten().flatten().map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "log")).collect();
    logs.sort_by_key(|path| std::fs::metadata(path).and_then(|file| file.modified()).ok());
    logs.into_iter().rev().take(LOGS).filter_map(|path| {
        let bytes = std::fs::read(&path).ok()?;
        let from = bytes.len().saturating_sub(LOG_LIMIT as usize);
        Some((format!("logs/{}", path.file_name()?.to_string_lossy()), bytes[from..].to_vec()))
    }).collect()
}

/// Each device's log as `logs/devices/<instance>.log`, one line each; a character a file name cannot
/// hold is written `_` (the instance itself is the file's first line when its name was changed so).
fn device_log_members(logs: Vec<(String, Vec<String>)>) -> Vec<(String, Vec<u8>)> {
    logs.into_iter().map(|(instance, lines)| {
        let file: String = instance.chars().map(|character| if character.is_alphanumeric() || "-_.".contains(character) { character } else { '_' }).collect();
        let mut text = if file == instance { String::new() } else { format!("(device {instance})\n") };
        for line in lines { text.push_str(&line); text.push('\n'); }
        (format!("logs/devices/{file}.log"), text.into_bytes())
    }).collect()
}

/// The session files ([`journal::sessions`]): this process's as `journal/session.jsonl`, the earlier
/// processes' still kept under `journal/sessions/` by their own names. A file not there is left out.
fn session_members(sessions: Option<(PathBuf, Vec<PathBuf>)>) -> Vec<(String, Vec<u8>)> {
    let Some((current, earlier)) = sessions else { return Vec::new() };
    let read = |path: &Path, member: String| std::fs::read(path).ok().map(|bytes| (member, bytes));
    read(&current, "journal/session.jsonl".into()).into_iter()
        .chain(earlier.iter().filter_map(|path| read(path, format!("journal/sessions/{}", path.file_name()?.to_string_lossy()))))
        .collect()
}

/// Every text value of `document`, wherever it stands.
fn texts<'a>(document: &'a Value, found: &mut Vec<&'a str>) {
    match document {
        Value::String(text) => found.push(text),
        Value::Array(items) => for item in items { texts(item, found); },
        Value::Object(fields) => for value in fields.values() { texts(value, found); },
        _ => {}
    }
}

/// The JSON files `documents` name (a path as written, or one inside `workspace`), and the ones
/// those name in turn: (archive member, the file's path, its bytes). Each file once, the first
/// [`NAMED_FILES`] found, none over [`NAMED_LIMIT`].
fn named_files(documents: &[&Vec<u8>], workspace: &Path) -> Vec<(String, PathBuf, Vec<u8>)> {
    let mut waiting: Vec<Vec<u8>> = documents.iter().map(|document| (*document).clone()).collect();
    let (mut seen, mut found) = (BTreeSet::new(), Vec::new());
    while let Some(bytes) = waiting.pop() {
        let Ok(document) = serde_json::from_slice::<Value>(&bytes) else { continue };
        let mut named = Vec::new();
        texts(&document, &mut named);
        for text in named {
            if found.len() >= NAMED_FILES { return found; }
            // A name of a file, not a sentence that happens to end in one.
            if text.len() > 512 || !text.to_ascii_lowercase().ends_with(".json") { continue; }
            let path = if Path::new(text).is_absolute() { PathBuf::from(text) } else { workspace.join(text) };
            let Ok(file) = std::fs::metadata(&path) else { continue };
            if !file.is_file() || file.len() > NAMED_LIMIT || !seen.insert(path.clone()) { continue; }
            let Ok(bytes) = std::fs::read(&path) else { continue };
            let name = path.file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default();
            found.push((format!("files/{:03}-{name}", found.len() + 1), path, bytes.clone()));
            waiting.push(bytes);
        }
    }
    found
}

/// The archive: every journal segment under `journal/`, then `members`.
pub fn write(path: &Path, segments: &[PathBuf], members: &[(String, Vec<u8>)]) -> std::io::Result<Exported> {
    let mut archive = ZipWriter::new(std::io::BufWriter::new(std::fs::File::create(path)?))?;
    let mut written = 0;
    for segment in segments {
        // A segment pruned between the listing and now is simply no longer part of the hour.
        let (Some(name), Ok(bytes)) = (segment.file_name(), std::fs::read(segment)) else { continue };
        archive.write_member(&format!("journal/{}", name.to_string_lossy()), true, &[&bytes])?;
        written += 1;
    }
    for (name, bytes) in members { archive.write_member(name, true, &[bytes])?; }
    use std::io::Write;
    archive.finish()?.flush()?;
    Ok(Exported { path: path.to_owned(), segments: written, bytes: std::fs::metadata(path)?.len() })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_archive_holds_the_journal_and_the_board() {
        let folder = std::env::temp_dir().join(format!("zlc-diagnostics-test-{}", journal::now_us()));
        std::fs::create_dir_all(&folder).unwrap();
        let segment = folder.join("journal-1.jsonl");
        std::fs::write(&segment, b"{\"us\":1,\"n\":0,\"src\":\"board\",\"ev\":\"fire\"}\n").unwrap();
        let path = folder.join("out.zip");
        let exported = write(&path, &[segment, folder.join("journal-gone.jsonl")], &[("state.json".into(), b"{}".to_vec())]).unwrap();
        assert_eq!(exported.segments, 1, "a segment that has gone is left out");
        let mut archive = zlc_archive::zip::ZipReader::open(&path).unwrap();
        assert_eq!(archive.names(), ["journal/journal-1.jsonl", "state.json"]);
        assert!(String::from_utf8(archive.read("journal/journal-1.jsonl").unwrap()).unwrap().contains("\"ev\":\"fire\""));
        std::fs::remove_dir_all(&folder).unwrap();
    }

    /// Every device's life is the session's, whatever the hour: what it opened with (a secret said
    /// only to be there), how long, what it said of itself, its failure; and its log goes with it.
    #[test]
    fn a_devices_life_and_log_go_into_the_archive() {
        use crate::testing::{FakeHost, FakePlugins, catalog, handed};
        use crate::Event;
        let folder = std::env::temp_dir().join(format!("zlc-diagnostics-lifecycle-{}", journal::now_us()));
        journal::open(&folder.join("journal")).unwrap();
        let mut types = catalog();
        types["types"][0]["parameters"].as_array_mut().unwrap().push(json!({"name": "api_token", "type": "str", "label": "Token", "default": ""}));
        std::fs::write(folder.join("apparatus.json"), serde_json::to_vec(&json!({"format": "zlc.installation", "simulation": {"seed": 3},
            "devices": [{"instance_id": "life_camera", "role": "Imaging", "type_id": "camera.virtual_mot", "parameters": {"api_token": "s3cret"}},
                        {"instance_id": "life:basler", "role": "Basler", "type_id": "camera.pylon", "parameters": {"serial": "x"}}]})).unwrap()).unwrap();
        let mut workbench = Workbench::new(&folder, Box::new(FakePlugins { calls: Default::default(), catalog: types })).unwrap();
        let mut host = FakeHost::default();
        workbench.apply("apply_devices", &json!({}), &zlc_core::control::Origin::Operator, 1, &mut host);
        workbench.on_event(Event::Opened { instance: "life_camera".into(), working_point: json!({"narrates": true,
            "device_info": {"model": "MOT-1", "serial": "42"}}) }, &mut host);
        workbench.on_event(Event::Log { instance: "life_camera".into(), line: "exposure 0.1 s".into() }, &mut host);
        workbench.apply("close_device", &json!({"device": "life_camera"}), &zlc_core::control::Origin::Operator, 2, &mut host);
        workbench.on_event(Event::Closed { instance: "life_camera".into(), error: None }, &mut host);

        let (session, _) = journal::sessions().unwrap();
        let lines: Vec<Value> = std::fs::read_to_string(&session).unwrap().lines().map(|line| serde_json::from_str(line).unwrap()).collect();
        let of = |instance: &str| lines.iter().filter(|line| line["instance"] == instance).cloned().collect::<Vec<_>>();
        let camera = of("life_camera");
        let events: Vec<&str> = camera.iter().map(|line| line["ev"].as_str().unwrap()).collect();
        assert_eq!(events, ["device_open", "device_opened", "device_close", "device_closed"]);
        let open = &camera[0];
        assert_eq!((&open["role"], &open["type_id"], &open["attempt"]), (&json!("Imaging"), &json!("camera.virtual_mot"), &json!(1)));
        assert_eq!(open["parameters"], json!({"exposure_seconds": 0.1, "api_token": journal::REDACTED}), "its type's defaults filled in, its secret hidden");
        assert_eq!(open["simulation"]["seed"], 3);
        assert_eq!(camera[1]["device_info"], json!({"model": "MOT-1", "serial": "42"}));
        assert!(camera[1]["ms"].is_f64() && camera[3]["ms"].is_f64());
        assert_eq!(camera[3]["outcome"], "closed");
        let basler = of("life:basler");
        assert_eq!((basler.len(), &basler[0]["outcome"]), (1, &json!("failed")), "refused before it reached any device: {basler:?}");
        assert!(basler[0]["error"].as_str().unwrap().contains("not installed"));
        assert!(lines.iter().any(|line| line["ev"] == "devices_apply" && line["opening"] == json!(["life_camera", "life:basler"])));

        // The archive carries the session and each device's log, the failed one's too.
        workbench.apply("export_diagnostics", &json!({"data": false}), &zlc_core::control::Origin::Operator, 3, &mut host);
        let (works, _) = handed();
        let exported = works.into_iter().map(|(_, _, work)| work()).find_map(Result::ok).expect("exported");
        let mut archive = zlc_archive::zip::ZipReader::open(std::path::Path::new(exported["path"].as_str().unwrap())).unwrap();
        let session = String::from_utf8(archive.read("journal/session.jsonl").unwrap()).unwrap();
        assert!(session.contains("\"ev\":\"device_opened\"") && !session.contains("s3cret"));
        assert_eq!(String::from_utf8(archive.read("logs/devices/life_camera.log").unwrap()).unwrap(), "exposure 0.1 s\n");
        // The failed one's log says why it did not open, as its Log window does.
        let basler = String::from_utf8(archive.read("logs/devices/life_basler.log").unwrap()).unwrap();
        let lines: Vec<&str> = basler.lines().collect();
        assert_eq!((lines.len(), lines[0]), (2, "(device life:basler)"), "{basler}");
        assert!(lines[1].starts_with('[') && lines[1].contains("] not opened: camera.pylon is not installed"), "{basler}");
        assert!(String::from_utf8(archive.read("README.txt").unwrap()).unwrap().contains("journal/session.jsonl"));
        std::fs::remove_dir_all(&folder).ok();
    }

    #[test]
    fn the_files_a_board_names_are_carried_with_the_ones_they_name() {
        let workspace = std::env::temp_dir().join(format!("zlc-diagnostics-named-{}", journal::now_us()));
        std::fs::create_dir_all(workspace.join("pulses")).unwrap();
        let subpulse = workspace.join("pulses").join("detect.subpulse.json");
        std::fs::write(&subpulse, br#"{"format": "zlc.pulse"}"#).unwrap();
        // A pulse named relative to the workspace names its Subpulse by a whole path, and itself.
        std::fs::write(workspace.join("pulses").join("scan.json"), serde_json::to_vec(&json!({"components": [{"config_path": subpulse}],
            "self": "pulses/scan.json", "note": "see the file notes.json"})).unwrap()).unwrap();
        std::fs::write(workspace.join("calibration.npz"), b"data").unwrap();
        let layout = serde_json::to_vec(&json!({"nodes": [{"values": {"pulse": "pulses/scan.json", "calibration": "calibration.npz", "missing": "gone.json"}}]})).unwrap();
        let found = named_files(&[&layout], &workspace);
        let members: Vec<&str> = found.iter().map(|(member, _, _)| member.as_str()).collect();
        assert_eq!(members, ["files/001-scan.json", "files/002-detect.subpulse.json"], "each once; data and missing files are only named");
        assert_eq!(found[1].1, subpulse);
        std::fs::remove_dir_all(&workspace).unwrap();
    }
}
