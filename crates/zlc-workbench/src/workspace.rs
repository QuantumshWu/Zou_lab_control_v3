//! Where workbench files live (v2's workspace: `apparatus.json`, `pulses/`,
//! `data/<YYYY_MM_DD>/`, `recipes/`, and `layouts/`), and the one strict reader
//! and atomic writer of its JSON documents.
//!
//! Each kind of file the operator opens or saves has its own place here, one
//! name per kind (the Pulse Editor's value files: `zlc_pulse::codec::
//! CONFIG_VALUES_DIRECTORY`). A kind's file dialog starts at its open document,
//! else at its place ([`dialog_start`]), so no kind moves another's.
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Value};

/// Reads one workspace file of a contract family into what a build receives
/// (a pulse document resolves, in the core, to its description).
pub type Resolver = Arc<dyn Fn(&Path) -> Result<Value, String> + Send + Sync>;

pub struct Workspace { pub root: PathBuf, resolvers: Vec<(String, Resolver)> }

/// The apparatus Devices@ edits.
pub const APPARATUS: &str = "apparatus.json";
/// Task Console layouts.
pub const LAYOUTS: &str = "layouts";
pub const PULSES: &str = "pulses";
pub const RECIPES: &str = "recipes";
/// Runs, saved figures, screenshots (each day's [`Workspace::today`]), calibrations, SLM targets.
pub const DATA: &str = "data";

impl Workspace {
    /// `root` with its data, pulse and layout folders, created when missing (a fresh installation starts empty).
    pub fn new(root: &Path) -> Result<Self, String> {
        let made = |folder: &Path| zlc_archive::durable::durable_makedirs(folder).map_err(|error| format!("Workspace {}: {error}", root.display()));
        let data = made(&root.join(DATA))?;
        let root = data.parent().unwrap_or(root).to_path_buf();
        for folder in [PULSES, LAYOUTS] { made(&root.join(folder))?; }
        Ok(Self { root, resolvers: Vec::new() })
    }
    pub fn data(&self) -> PathBuf { self.root.join(DATA) }
    pub fn pulses(&self) -> PathBuf { self.root.join(PULSES) }
    pub fn recipes(&self) -> PathBuf { self.root.join(RECIPES) }
    pub fn layouts(&self) -> PathBuf { self.root.join(LAYOUTS) }
    pub fn apparatus(&self) -> PathBuf { self.root.join(APPARATUS) }
    /// `data/<YYYY_MM_DD>` for today's local date, whether or not it exists.
    pub fn today(&self) -> PathBuf {
        let (year, month, day) = local_date();
        self.data().join(format!("{year:04}_{month:02}_{day:02}"))
    }
    /// Today's folder, created durably (a run root or a default save place).
    pub fn day_folder(&self) -> Result<PathBuf, String> {
        zlc_archive::durable::durable_makedirs(&self.today()).map_err(|error| error.to_string())
    }
    /// A relative path is read against the workspace root.
    pub fn resolve(&self, path: &str) -> PathBuf {
        let path = Path::new(path);
        if path.is_absolute() { path.to_path_buf() } else { self.root.join(path) }
    }
    /// Files of `contract_prefix...` resolve through `resolver`; any other contract resolves to its absolute path.
    pub fn add_resolver(&mut self, contract_prefix: &str, resolver: Resolver) { self.resolvers.push((contract_prefix.to_owned(), resolver)); }
    /// v2 `_resolve_workspace_resources` for one field: the file `selected` names in the
    /// collection folder, read by its contract's resolver; or the issue a Start reports.
    pub fn resource(&self, spec: &crate::types::WorkspaceResource, label: &str, selected: &str) -> Result<Value, String> {
        let directory = self.root.join(&spec.directory);
        let chosen = Path::new(selected.trim());
        let path = if chosen.is_absolute() { chosen.to_path_buf() } else { directory.join(chosen) };
        let suffixed = spec.suffixes.iter().any(|suffix| path.to_string_lossy().to_lowercase().ends_with(&suffix.to_lowercase()));
        if selected.trim().is_empty() || path.parent() != Some(directory.as_path()) || !suffixed || !path.is_file() {
            return Err(format!("{label} must be a file in {}/", spec.directory));
        }
        match self.resolvers.iter().find(|(prefix, _)| spec.contract.starts_with(prefix.as_str())) {
            Some((_, resolver)) => resolver(&path).map_err(|error| format!("{} resource “{}” is invalid: {error}", spec.contract, path.display())),
            None => Ok(Value::String(path.display().to_string())),
        }
    }
}

/// Where a file dialog opens and the name it offers (v2 FluentPathEdit._dialog_start): `start` is
/// a folder, or a file opened in its folder with its name in the box. A start whose folder is
/// not there (another computer's drive, a day not made yet) opens in `home`, its kind's own
/// folder (or the nearest folder above it that is there), keeping the name only for a save.
/// Paths come back in native spelling (Qt's toNativeSeparators): the shell refuses `C:/a/b`.
pub fn dialog_start(start: &Path, home: &Path, save: bool) -> (PathBuf, String) {
    // GetFullPathNameW: separators made native, nothing on disk read (no empty path: it stays empty).
    let native = |path: &Path| std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
    let start = native(start);
    if start.is_dir() { return (start, String::new()); }
    let name = start.file_name().map_or_else(String::new, |name| name.to_string_lossy().into_owned());
    if let Some(folder) = start.parent().filter(|folder| folder.is_dir()) { return (folder.to_path_buf(), name); }
    let home = native(home);
    (home.ancestors().find(|folder| folder.is_dir()).map_or_else(|| home.clone(), Path::to_path_buf), if save { name } else { String::new() })
}

/// `<parent>/<stem>`, or the first free `<stem>-n`: created, so two runs can
/// never take the same directory (v2 `unique_path` with an empty suffix).
pub fn unique_directory(parent: &Path, stem: &str) -> Result<PathBuf, String> {
    let safe: String = stem.chars().map(|c| if c.is_control() || "<>:\"/\\|?* ".contains(c) { '-' } else { c }).collect();
    let safe = safe.trim_matches(|c| c == ' ' || c == '.' || c == '-');
    let safe = if safe.is_empty() { "untitled" } else { safe };
    for ordinal in 1u32.. {
        let candidate = if ordinal == 1 { parent.join(safe) } else { parent.join(format!("{safe}-{ordinal}")) };
        match std::fs::create_dir(&candidate) {
            Ok(()) => {
                zlc_archive::durable::flush_directory(parent).map_err(|error| error.to_string())?;
                return Ok(candidate);
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(format!("Cannot create {}: {error}", candidate.display())),
        }
    }
    unreachable!("the ordinals do not run out")
}

/// UTC now as v2 writes it: `2026-10-20T12:02:11.512+00:00`.
pub fn utc_now() -> String {
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
    let seconds = now.as_secs() as i64;
    let (year, month, day) = civil(seconds.div_euclid(86_400));
    let second = seconds.rem_euclid(86_400);
    format!("{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{:03}+00:00", second / 3600, second / 60 % 60, second % 60, now.subsec_millis())
}

/// Days since 1970-01-01 to (year, month, day), proleptic Gregorian.
fn civil(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (yoe + era * 400 + i64::from(month <= 2), month, day)
}

/// Today's local calendar date (the day folder an operator looks in).
#[cfg(windows)]
pub fn local_date() -> (i64, u32, u32) {
    let mut time = unsafe { std::mem::zeroed::<windows_sys::Win32::Foundation::SYSTEMTIME>() };
    unsafe { windows_sys::Win32::System::SystemInformation::GetLocalTime(&mut time) };
    (i64::from(time.wYear), u32::from(time.wMonth), u32::from(time.wDay))
}

#[cfg(not(windows))]
pub fn local_date() -> (i64, u32, u32) {
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
    civil(now.as_secs() as i64 / 86_400)
}

/// JSON that means exactly what it says: a duplicate key or a non-finite
/// number refuses the document, naming `what` (v2 `strict_json_loads`).
pub fn parse_strict(text: &str, what: &str) -> Result<Value, String> {
    let mut deserializer = serde_json::Deserializer::from_str(text);
    let value = Strict::deserialize(&mut deserializer).and_then(|value| deserializer.end().map(|()| value.0))
        .map_err(|error| format!("{what}: {error}"))?;
    Ok(value)
}

pub fn read_strict(path: &Path, what: &str) -> Result<Value, String> {
    let text = std::fs::read_to_string(path).map_err(|error| format!("Cannot read {}: {error}", path.display()))?;
    parse_strict(&text, what)
}

/// Readable JSON (two-space indent, trailing newline), written atomically.
pub fn write_readable(path: &Path, value: &Value) -> Result<PathBuf, String> {
    let mut text = serde_json::to_string_pretty(value).map_err(|error| error.to_string())?;
    text.push('\n');
    zlc_archive::durable::atomic_write_file(path, |mut file| {
        use std::io::Write;
        file.write_all(text.as_bytes()).map_err(|error| zlc_archive::ArchiveError::os(&error, None))?;
        Ok(file)
    }).map_err(|error| format!("Cannot write {}: {error}", path.display()))
}

struct Strict(Value);

impl<'de> Deserialize<'de> for Strict {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> { deserializer.deserialize_any(StrictVisitor) }
}

struct StrictVisitor;

impl<'de> Visitor<'de> for StrictVisitor {
    type Value = Strict;
    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result { formatter.write_str("JSON") }
    fn visit_bool<E>(self, value: bool) -> Result<Strict, E> { Ok(Strict(Value::Bool(value))) }
    fn visit_i64<E>(self, value: i64) -> Result<Strict, E> { Ok(Strict(Value::from(value))) }
    fn visit_u64<E>(self, value: u64) -> Result<Strict, E> { Ok(Strict(Value::from(value))) }
    fn visit_f64<E: de::Error>(self, value: f64) -> Result<Strict, E> {
        serde_json::Number::from_f64(value).map(|number| Strict(Value::Number(number))).ok_or_else(|| E::custom("non-finite number"))
    }
    fn visit_str<E>(self, value: &str) -> Result<Strict, E> { Ok(Strict(Value::String(value.to_owned()))) }
    fn visit_string<E>(self, value: String) -> Result<Strict, E> { Ok(Strict(Value::String(value))) }
    fn visit_unit<E>(self) -> Result<Strict, E> { Ok(Strict(Value::Null)) }
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Strict, A::Error> {
        let mut items = Vec::new();
        while let Some(Strict(item)) = seq.next_element()? { items.push(item); }
        Ok(Strict(Value::Array(items)))
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Strict, A::Error> {
        let mut object = Map::new();
        while let Some(key) = map.next_key::<String>()? {
            if object.contains_key(&key) { return Err(de::Error::custom(format!("duplicate key {key:?}"))); }
            let Strict(value) = map.next_value()?;
            object.insert(key, value);
        }
        Ok(Strict(Value::Object(object)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_json_refuses_duplicates_and_dates_are_civil() {
        assert!(parse_strict(r#"{"a": 1, "a": 2}"#, "doc").unwrap_err().contains("duplicate key"));
        assert!(parse_strict("1e999", "doc").is_err());
        assert_eq!(parse_strict(r#"{"b": [1, 2.5, "x"], "a": null}"#, "doc").unwrap().to_string(), r#"{"b":[1,2.5,"x"],"a":null}"#);
        assert_eq!(civil(0), (1970, 1, 1));
        assert_eq!(civil(20_746), (2026, 10, 20));
        assert_eq!(utc_now().len(), "2026-10-20T12:02:11.512+00:00".len());
    }
}
