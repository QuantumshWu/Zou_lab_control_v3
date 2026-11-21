//! What an event's record may carry. A committed event keeps its metadata with every
//! publication, and every saved Figure keeps it in its lineage, so a field written on each
//! event costs every event, every save and every reader. Each field a producer writes is
//! registered in `record_fields.json` with its owner, its purpose and the code that reads it:
//! - `metadata`: the event metadata's own keys (`records`, `shot_time_seconds`, ...);
//! - `event_record`: the keys of one output's record, `metadata.records.<output>`.
//!
//! A field says something: no null, "", [] or {} value, and an output with nothing to say
//! has no record. A fact that is the same for every event of a run belongs in the run record;
//! an event records how it differs (docs/PLUGIN_PROTOCOL.md, "Event records").
//!
//! Producers are checked where they first write (the plugin host's commits, the native
//! writers), never where a saved record is committed again. A debug build refuses a field
//! that is not registered or says nothing; a release build keeps the event and notes the field
//! once in the journal, so a missed registration never stops a run in the lab.
use std::collections::{BTreeMap, HashSet};
use std::sync::{Mutex, OnceLock};

use serde_json::Value;

const REGISTRY: &str = include_str!("record_fields.json");

/// One registered field: who writes it, what it says, which code reads it (repository paths).
pub struct Field { pub owner: String, pub purpose: String, pub readers: Vec<String> }

pub struct Registry { pub metadata: BTreeMap<String, Field>, pub event_record: BTreeMap<String, Field> }

/// The registry `record_fields.json` holds.
pub fn registry() -> &'static Registry {
    static PARSED: OnceLock<Registry> = OnceLock::new();
    PARSED.get_or_init(|| parse(REGISTRY).unwrap_or_else(|error| panic!("record_fields.json: {error}")))
}

fn parse(text: &str) -> Result<Registry, String> {
    let root: Value = serde_json::from_str(text).map_err(|error| error.to_string())?;
    let section = |name: &str| -> Result<BTreeMap<String, Field>, String> {
        let fields = root.get(name).and_then(Value::as_object).ok_or(format!("no {name} section"))?;
        fields.iter().map(|(key, entry)| {
            let text = |part: &str| entry.get(part).and_then(Value::as_str).filter(|text| !text.trim().is_empty())
                .map(str::to_owned).ok_or(format!("{name}.{key} has no {part}"));
            let readers: Vec<String> = entry.get("readers").and_then(Value::as_array).into_iter().flatten()
                .filter_map(Value::as_str).map(str::to_owned).collect();
            if readers.is_empty() { return Err(format!("{name}.{key} names no reader")); }
            Ok((key.clone(), Field { owner: text("owner")?, purpose: text("purpose")?, readers }))
        }).collect()
    };
    Ok(Registry { metadata: section("metadata")?, event_record: section("event_record")? })
}

fn says_nothing(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::String(text) => text.is_empty(),
        Value::Array(items) => items.is_empty(),
        Value::Object(fields) => fields.is_empty(),
        _ => false,
    }
}

/// The first field of an event's metadata that is not registered or says nothing, as its path and why.
pub fn fault(metadata: &Value) -> Option<String> {
    let registry = registry();
    let fields = metadata.as_object()?;
    for (key, value) in fields {
        if !registry.metadata.contains_key(key) { return Some(format!("metadata.{key} is not registered")); }
        if says_nothing(value) { return Some(format!("metadata.{key} says nothing ({value})")); }
    }
    for (output, record) in fields.get("records").and_then(Value::as_object).into_iter().flatten() {
        let Some(record) = record.as_object() else { return Some(format!("records.{output} is not a record")) };
        if record.is_empty() { return Some(format!("records.{output} says nothing ({{}})")); }
        for (key, value) in record {
            if !registry.event_record.contains_key(key) { return Some(format!("records.{output}.{key} is not registered")); }
            if says_nothing(value) { return Some(format!("records.{output}.{key} says nothing ({value})")); }
        }
    }
    None
}

/// A producer's metadata as it is first written: an error in a debug build when a field is not
/// registered or says nothing; in a release build the event is kept and the field noted once.
pub fn admit(producer: &str, metadata: &Value) -> Result<(), String> {
    let Some(fault) = fault(metadata) else { return Ok(()) };
    let message = format!("{producer}: {fault}; an event records only registered fields that say something \
                           (crates/zlc-core/src/record_fields.json)");
    if cfg!(debug_assertions) { return Err(message); }
    static NOTED: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    if NOTED.get_or_init(Default::default).lock().unwrap_or_else(|error| error.into_inner()).insert(fault) {
        crate::journal::note("runtime", "unregistered_record_field", serde_json::json!({"producer": producer, "fault": message}));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::path::Path;

    /// The registry parses, and each field's readers are code outside the tests that names it.
    #[test]
    fn every_registered_field_is_read_by_name_outside_the_tests() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        for (section, fields) in [("metadata", &registry().metadata), ("event_record", &registry().event_record)] {
            for (key, field) in fields {
                for reader in &field.readers {
                    let path = reader.replace('\\', "/");
                    assert!(!path.split('/').any(|part| part == "tests" || part.starts_with("test_")), "{section}.{key}: {reader} is a test");
                    let text = std::fs::read_to_string(root.join(&path)).unwrap_or_else(|error| panic!("{section}.{key}: {reader}: {error}"));
                    // A Rust file's test module follows its code.
                    let tests = text.match_indices("#[cfg(test)]").map(|(at, _)| at)
                        .find(|at| text[at + "#[cfg(test)]".len()..].trim_start().trim_start_matches("pub ").starts_with("mod "));
                    let code = if path.ends_with(".rs") { &text[..tests.unwrap_or(text.len())] } else { &text };
                    let named = [format!("\"{key}\""), format!("'{key}'")].iter().any(|quoted| code.contains(quoted.as_str()))
                        || code.match_indices(&format!(".{key}")).any(|(at, found)| !code[at + found.len()..].starts_with(|c: char| c.is_alphanumeric() || c == '_'));
                    assert!(named, "{section}.{key}: {reader} does not read it by name");
                }
            }
        }
    }

    #[test]
    fn a_field_is_registered_and_says_something() {
        assert_eq!(fault(&json!({})), None);
        assert_eq!(fault(&json!({"records": {"frames": {"frames_missing": [1]}}, "shot_time_seconds": {"frames": 0.5}})), None);
        assert_eq!(fault(&json!({"shot": 1})).unwrap(), "metadata.shot is not registered");
        assert_eq!(fault(&json!({"records": {"frames": {"trigger_alignment": {"missing": []}}}})).unwrap(), "records.frames.trigger_alignment is not registered");
        assert_eq!(fault(&json!({"records": {"frames": {"frames_missing": []}}})).unwrap(), "records.frames.frames_missing says nothing ([])");
        assert_eq!(fault(&json!({"records": {"scan": {}}})).unwrap(), "records.scan says nothing ({})");
        assert!(admit("test", &json!({"records": {"frames": {"x": 1}}})).unwrap_err().contains("records.frames.x is not registered"));
    }
}
