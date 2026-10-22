//! The pulse editor's authoring layer (v2 `zlc_workbench.pulse_editor`): the
//! edits a window or a command makes to a pulse, the projections its pages
//! show, and the timeline its preview draws. Every edit builds a new
//! `Sequence` and lets the model refuse it, so the editor can never hold a
//! pulse the compiler would refuse; nothing here talks to a board.
pub mod args;
pub mod items;
pub mod text;
pub mod edits;
pub mod target;
pub mod project;
pub mod timeline;
pub mod tables;
pub mod editor;
pub mod session;

pub use zlc_pulse::{Error, Kind, Result};

/// A pulse-codec JSON tree as a `serde_json` value (ints stay ints).
pub fn to_value(json: &zlc_pulse::json::Json) -> serde_json::Value {
    use serde_json::Value;
    use zlc_pulse::json::Json;
    match json {
        Json::Null => Value::Null,
        Json::Bool(flag) => Value::Bool(*flag),
        Json::Int(int) => i64::try_from(*int).map(Value::from).or_else(|_| u64::try_from(*int).map(Value::from))
            .unwrap_or_else(|_| Value::from(*int as f64)),
        Json::Float(float) => serde_json::Number::from_f64(*float).map_or(Value::Null, Value::Number),
        Json::Str(text) => Value::String(text.clone()),
        Json::Array(items) => Value::Array(items.iter().map(to_value).collect()),
        Json::Object(items) => Value::Object(items.iter().map(|(key, value)| (key.clone(), to_value(value))).collect()),
    }
}

/// A `serde_json` value as a pulse-codec tree, through the codec's strict reader.
pub fn from_value(value: &serde_json::Value) -> Result<zlc_pulse::json::Json> {
    zlc_pulse::json::parse(&value.to_string(), "pulse document")
}
