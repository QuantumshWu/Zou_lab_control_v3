//! The diagnostics archive: the last hour as the journal kept it, with what the board holds now.
//!
//! One file an operator hands to whoever looks into a fault: the journal's segments (what happened,
//! in order), the board as it stands (its layout, the apparatus, the workbench's own state), the
//! release that was running, the newest program logs, and the text files the board names -- a
//! node's pulse, the Subpulses and the Config that pulse names in turn -- as they are now. It
//! holds no acquired data: a calibration or a dataset the board names is named, not carried.
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
const LOG_LIMIT: u64 = 8 << 20;
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
    board load/fire/report/safe        the pulse board; a report's `elapsed_seconds` after its fire is
                                       when the board itself was done
    scan segment_armed/segment_filled  a scan's segments; `dropped` is what arrived before one was armed
    plugins call/device                what the core asked of a plugin, and a run of a device
    camera arm/finish/frame            a camera's captures and each frame as its measurement read it
    ui click/key/popup/error           what the operator did in a window

state.json      the workbench's state when this was exported (nodes, runs, devices, board)
layout.json     the board as Save would write it
apparatus.json  the apparatus file
release.json    the release that was running
logs/           the newest program logs
files/          the JSON files the board names (a node's pulse, the Subpulses and Config it names),
                as they were when this was exported; files.json says which path each one is
";

/// What one export wrote.
pub struct Exported { pub path: PathBuf, pub segments: usize, pub bytes: u64 }

impl Workbench {
    /// Write the archive into the workspace's diagnostics folder and say where.
    pub(crate) fn export_diagnostics(&mut self, host: &dyn Host) -> Result<Value, String> {
        let stamp: String = workspace::utc_now().chars().filter(char::is_ascii_digit).take(14).collect();
        let folder = self.workspace.root.join(FOLDER);
        std::fs::create_dir_all(&folder).map_err(|error| format!("{}: {error}", folder.display()))?;
        let path = folder.join(format!("zlc-diagnostics-{}-{}.zip", &stamp[..8.min(stamp.len())], &stamp[8.min(stamp.len())..]));
        let mut members: Vec<(String, Vec<u8>)> = vec![("README.txt".into(), README.as_bytes().to_vec())];
        let pretty = |value: &Value| serde_json::to_vec_pretty(value).unwrap_or_default();
        members.push(("state.json".into(), pretty(&Value::Object(self.state(host)))));
        members.push(("layout.json".into(), pretty(&self.layout_document(host))));
        for (name, file) in [("apparatus.json", self.workspace.apparatus()), ("release.json", zlc_core::checkout::path("bin/release.json"))] {
            if let Ok(bytes) = std::fs::read(&file) { members.push((name.into(), bytes)); }
        }
        members.extend(newest_logs(&zlc_core::checkout::local("logs")));
        let named = named_files(&[&members[1].1, &members[2].1], &self.workspace.root);
        let files = named.len();
        members.push(("files.json".into(), pretty(&Value::Object(named.iter().map(|(member, path, _)| (member.clone(), json!(path))).collect()))));
        members.extend(named.into_iter().map(|(member, _, bytes)| (member, bytes)));
        journal::note("workbench", "diagnostics_exported", json!({"path": path}));
        let exported = write(&path, &journal::segments(), &members).map_err(|error| format!("{}: {error}", path.display()))?;
        self.console_report = (format!("diagnostics exported to {}", exported.path.display()), "task");
        Ok(json!({"path": exported.path, "segments": exported.segments, "files": files, "bytes": exported.bytes}))
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
