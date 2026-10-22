//! One pulse as a tree a file can hold: `zlc.pulse` documents, `zlc.subpulse`
//! fragments and `zlc.pulse.config_values` tables, in the v2 grammar exactly.
//! Every value goes back through the model's constructors, so a tree that
//! describes an illegal pulse is refused here rather than becoming one.
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::error::{Error, Kind, Result, repr, type_error, value_error};
use crate::json::{self, Json};
use crate::model::{AnalogStep, Binding, Bracket, Component, FieldRef, OutputDelay, Period, PortSpec, Sequence, SequenceParts,
                   Source, Subpulse, Target, config_parameter_key};
use crate::num::Num;

pub const PULSE_TREE_FORMAT: &str = "zlc.pulse";
pub const SUBPULSE_TREE_FORMAT: &str = "zlc.subpulse";
pub const CONFIG_VALUES_FORMAT: &str = "zlc.pulse.config_values";
pub const CONFIG_VALUES_DIRECTORY: &str = "config_values";
pub const CURRENT_CONFIG_VALUES: &str = "current.json";

// ------------------------------------------------------------------ tree access
/// `_object(value, expected, name, optional=)`: exactly these members.
pub fn object<'a>(value: &'a Json, expected: &[&str], name: &str, optional: &[&str]) -> Result<&'a [(String, Json)]> {
    let Json::Object(items) = value else { return type_error(format!("{name} must be an object")); };
    let unknown: Vec<&str> = items.iter().map(|(key, _)| key.as_str()).filter(|key| !expected.contains(key) && !optional.contains(key)).collect();
    if !unknown.is_empty() { return value_error(format!("unknown {name} field(s): {}", unknown.join(", "))); }
    let missing: Vec<&str> = expected.iter().copied().filter(|key| !items.iter().any(|(item, _)| item == key)).collect();
    if !missing.is_empty() { return value_error(format!("missing {name} field(s): {}", missing.join(", "))); }
    Ok(items)
}

pub fn member<'a>(items: &'a [(String, Json)], key: &str) -> &'a Json {
    items.iter().find(|(name, _)| name == key).map(|(_, value)| value).unwrap_or(&Json::Null)
}

fn optional<'a>(items: &'a [(String, Json)], key: &str) -> Option<&'a Json> { items.iter().find(|(name, _)| name == key).map(|(_, value)| value) }

/// `_array(value, name)`.
pub fn array<'a>(value: &'a Json, name: &str) -> Result<&'a [Json]> {
    value.as_array().ok_or_else(|| Error::new(Kind::Type, format!("{name} must be a JSON array")))
}

/// A member the model reads as text: a non-string is refused as `_text` refuses it.
fn text(value: &Json, field_name: &str) -> Result<String> {
    value.as_str().map(str::to_owned).ok_or_else(|| Error::new(Kind::Type, format!("{field_name} must be non-empty text")))
}

fn optional_text(value: &Json, field_name: &str) -> Result<Option<String>> {
    if value.is_null() { Ok(None) } else { text(value, field_name).map(Some) }
}

/// `_nonnegative_int`'s type rule: an int, not a bool.
fn integer(value: &Json, field_name: &str) -> Result<i128> {
    value.as_int().ok_or_else(|| Error::new(Kind::Type, format!("{field_name} must be an integer")))
}

fn optional_integer(value: &Json, field_name: &str) -> Result<Option<i128>> {
    if value.is_null() { Ok(None) } else { integer(value, field_name).map(Some) }
}

fn strings(value: &Json, name: &str, item_name: &str) -> Result<Vec<String>> {
    array(value, name)?.iter().map(|item| text(item, item_name)).collect()
}

// ------------------------------------------------------------------ target
pub fn port_from_tree(item: &Json) -> Result<PortSpec> {
    let port = object(item, &["key", "kind", "lanes", "label", "bus_index", "width", "encoding", "safe_value", "latch_clock"], "pulse port", &[])?;
    let key = text(member(port, "key"), "port key")?;
    let kind = text(member(port, "kind"), "port kind")?;
    let lanes = strings(member(port, "lanes"), "pulse port lanes", "raw lane")?;
    let label = text(member(port, "label"), "port label")?;
    let encoding = match member(port, "encoding") {
        Json::Null => None,
        Json::Str(text) => Some(text.clone()),
        other => return value_error(format!("only offset-binary DAC encoding is supported, not {}", other.type_name())),
    };
    PortSpec::new(&key, &kind, &lanes, &label,
                  optional_integer(member(port, "bus_index"), if kind == "dac" { "DAC bus_index" } else { "bus_index" })?,
                  optional_integer(member(port, "width"), "port width")?, encoding.as_deref(),
                  optional_integer(member(port, "safe_value"), if kind == "dac" { "DAC safe_value" } else { "safe_value" })?,
                  optional_text(member(port, "latch_clock"), "latch clock")?.as_deref())
}

pub fn target_from_tree(value: &Json) -> Result<Arc<Target>> {
    let target = object(value, &["raw_lanes", "package_pins", "ports"], "pulse target", &[])?;
    let Json::Object(pins) = member(target, "package_pins") else { return type_error("pulse target package_pins must be an object"); };
    let pins = pins.iter().map(|(lane, pin)| pin.as_str().map(|pin| (lane.clone(), pin.to_owned()))
        .ok_or_else(|| Error::new(Kind::Type, "pulse target package_pins must map text to text"))).collect::<Result<Vec<_>>>()?;
    let lanes = strings(member(target, "raw_lanes"), "pulse target raw_lanes", "target lane")?;
    let ports = array(member(target, "ports"), "pulse target ports")?.iter().map(port_from_tree).collect::<Result<Vec<_>>>()?;
    Target::new(&lanes, ports, pins)
}

pub fn port_to_tree(port: &PortSpec) -> Json {
    Json::object(vec![
        ("key", Json::str(&port.key)), ("kind", Json::str(port.kind.name())), ("lanes", Json::strings(&port.lanes)),
        ("label", Json::str(&port.label)), ("bus_index", port.bus_index.map_or(Json::Null, |bus| Json::Int(bus.into()))),
        ("width", Json::Int(port.width.into())), ("encoding", Json::str(&port.encoding)),
        ("safe_value", Json::Int(port.safe_value.into())), ("latch_clock", Json::opt_str(port.latch_clock.as_deref())),
    ])
}

pub fn target_to_tree(target: &Target) -> Json {
    Json::object(vec![
        ("raw_lanes", Json::strings(&target.raw_lanes)),
        ("package_pins", Json::Object(target.package_pins.iter().map(|(lane, pin)| (lane.clone(), Json::str(pin))).collect())),
        ("ports", Json::Array(target.ports.iter().map(port_to_tree).collect())),
    ])
}

// ------------------------------------------------------------------ pieces
pub fn period_from_tree(item: &Json) -> Result<Period> {
    let period = object(item, &["period_id", "duration", "unit", "states", "name", "analog_steps"], "pulse period", &["kind"])?;
    let states = array(member(period, "states"), "pulse period states")?;
    let steps = array(member(period, "analog_steps"), "pulse period analog_steps")?.iter().map(|item| {
        let step = object(item, &["port", "mode", "value"], "pulse analog step", &[])?;
        AnalogStep::new(&text(member(step, "port"), "analog port")?, &text(member(step, "mode"), "analog mode")?,
                        Num::from_json(member(step, "value"), "analog value")?)
    }).collect::<Result<Vec<_>>>()?;
    let period_id = text(member(period, "period_id"), "period_id")?;
    let kind = match optional(period, "kind") { None => "period".to_owned(), Some(kind) => text(kind, "period kind")? };
    let duration = Num::from_json(member(period, "duration"), "period duration")?;
    let unit = text(member(period, "unit"), "period unit")?;
    let states = states.iter().map(|state| match state {
        Json::Int(value @ (0 | 1)) => Ok(*value as u8),
        Json::Bool(flag) => Ok(u8::from(*flag)),
        _ => value_error("period states must contain only 0/1 values"),
    }).collect::<Result<Vec<_>>>();
    let name = member(period, "name");
    let period = Period::new(&period_id, duration, &unit, states?, steps, name.as_str().unwrap_or(""), &kind)?;
    if !matches!(name, Json::Str(_)) { return type_error("period name must be non-empty text"); }
    Ok(period)
}

pub fn period_to_tree(period: &Period) -> Json {
    Json::object(vec![
        ("period_id", Json::str(&period.period_id)), ("kind", Json::str(period.kind.name())), ("duration", period.duration.to_json()),
        ("unit", Json::str(&period.unit)), ("states", Json::ints(&period.states)), ("name", Json::str(&period.name)),
        ("analog_steps", Json::Array(period.analog_steps.iter().map(|step| Json::object(vec![
            ("port", Json::str(&step.port)), ("mode", Json::str(step.mode.name())), ("value", Json::Int(step.value.into())),
        ])).collect())),
    ])
}

pub fn field_ref_from_tree(value: &Json) -> Result<FieldRef> {
    let field = object(value, &["kind", "period_id", "port"], "pulse field reference", &["bracket_id"])?;
    FieldRef::new(&text(member(field, "kind"), "field kind")?, optional_text(member(field, "period_id"), "period_id")?.as_deref(),
                  optional_text(member(field, "port"), "port")?.as_deref(),
                  optional_text(optional(field, "bracket_id").unwrap_or(&Json::Null), "count bracket_id")?.as_deref())
}

pub fn field_ref_to_tree(reference: &FieldRef) -> Json {
    Json::object(vec![
        ("kind", Json::str(reference.kind.name())), ("period_id", Json::opt_str(reference.period_id.as_deref())),
        ("port", Json::opt_str(reference.port.as_deref())), ("bracket_id", Json::opt_str(reference.bracket_id.as_deref())),
    ])
}

pub fn binding_from_tree(item: &Json) -> Result<Binding> {
    let binding = object(item, &["field_ref", "unit", "scan", "source", "config_key"], "pulse binding", &[])?;
    let field_ref = field_ref_from_tree(member(binding, "field_ref"))?;
    let Some(scan) = member(binding, "scan").as_bool() else { return type_error("binding scan must be boolean"); };
    let source = match member(binding, "source") { Json::Str(text) => text.clone(), _ => return value_error("binding source must be default, api or config") };
    let unit = text(member(binding, "unit"), "binding unit")?;
    let config_key = text(member(binding, "config_key"), "Config key")?;
    Binding::new(field_ref, &unit, scan, &source, &config_key)
}

pub fn binding_to_tree(binding: &Binding) -> Json {
    Json::object(vec![
        ("field_ref", field_ref_to_tree(&binding.field_ref)), ("unit", Json::str(&binding.unit)), ("scan", Json::Bool(binding.scan)),
        ("source", Json::str(binding.source.name())), ("config_key", Json::str(&binding.config_key)),
    ])
}

pub fn delay_from_tree(item: &Json) -> Result<OutputDelay> {
    let delay = object(item, &["port", "value", "unit"], "pulse delay", &[])?;
    OutputDelay::new(&text(member(delay, "port"), "delay port")?, Num::from_json(member(delay, "value"), "delay value")?,
                     &text(member(delay, "unit"), "delay unit")?)
}

pub fn delay_to_tree(delay: &OutputDelay) -> Json {
    Json::object(vec![("port", Json::str(&delay.port)), ("value", delay.value.to_json()), ("unit", Json::str(&delay.unit))])
}

pub fn bracket_from_tree(item: &Json) -> Result<Bracket> {
    let bracket = object(item, &["bracket_id", "start_period_id", "end_period_id", "count"], "pulse bracket", &[])?;
    Bracket::new(&text(member(bracket, "bracket_id"), "bracket_id")?, optional_text(member(bracket, "start_period_id"), "bracket start")?.as_deref(),
                 optional_text(member(bracket, "end_period_id"), "bracket end")?.as_deref(), integer(member(bracket, "count"), "bracket count")?)
}

pub fn bracket_to_tree(bracket: &Bracket) -> Json {
    Json::object(vec![
        ("bracket_id", Json::str(&bracket.bracket_id)), ("start_period_id", Json::opt_str(bracket.start_period_id.as_deref())),
        ("end_period_id", Json::opt_str(bracket.end_period_id.as_deref())), ("count", Json::Int(bracket.count.into())),
    ])
}

fn component_from_tree(item: &Json) -> Result<Component> {
    let component = object(item, &["component_id", "name", "period_ids", "bracket_ids"], "pulse component",
                           &["source", "config_path", "period_id_map", "bracket_id_map"])?;
    let pairs = |key: &str, name: &str, pair_name: &str| -> Result<Vec<Vec<String>>> {
        match optional(component, key) {
            None => Ok(Vec::new()),
            Some(value) => array(value, name)?.iter().map(|pair| strings(pair, pair_name, "Subpulse source id")).collect(),
        }
    };
    Component::new(&text(member(component, "component_id"), "component id")?, &text(member(component, "name"), "component name")?,
                   &strings(member(component, "period_ids"), "component period_ids", "component period id")?,
                   &strings(member(component, "bracket_ids"), "component bracket_ids", "component bracket id")?,
                   &optional(component, "source").map_or(Ok("default".to_owned()), |value| text(value, "Subpulse source"))?,
                   &optional(component, "config_path").map_or(Ok(String::new()), |value| text(value, "Subpulse Config path"))?,
                   &pairs("period_id_map", "Subpulse period id map", "Subpulse period id pair")?,
                   &pairs("bracket_id_map", "Subpulse bracket id map", "Subpulse bracket id pair")?)
}

fn component_to_tree(component: &Component) -> Json {
    let pairs = |items: &[(String, String)]| Json::Array(items.iter().map(|(a, b)| Json::Array(vec![Json::str(a), Json::str(b)])).collect());
    Json::object(vec![
        ("component_id", Json::str(&component.component_id)), ("name", Json::str(&component.name)),
        ("period_ids", Json::strings(&component.period_ids)), ("bracket_ids", Json::strings(&component.bracket_ids)),
        ("source", Json::str(component.source.name())), ("config_path", Json::str(&component.config_path)),
        ("period_id_map", pairs(&component.period_id_map)), ("bracket_id_map", pairs(&component.bracket_id_map)),
    ])
}

// ------------------------------------------------------------------ sequences
/// `sequence_to_tree(sequence)`: everything needed to rebuild it exactly.
pub fn sequence_to_tree(sequence: &Sequence) -> Json {
    let mut tree = vec![
        ("format", Json::str(PULSE_TREE_FORMAT)),
        ("name", Json::str(&sequence.name)),
        ("time_step_ns", Json::Float(sequence.time_step_ns)),
        ("target", target_to_tree(&sequence.target)),
        ("periods", Json::Array(sequence.periods.iter().map(period_to_tree).collect())),
        ("bindings", Json::Array(sequence.bindings.iter().map(binding_to_tree).collect())),
        ("delays", Json::Array(sequence.delays.iter().map(delay_to_tree).collect())),
        ("brackets", Json::Array(sequence.brackets.iter().map(bracket_to_tree).collect())),
        ("run_repeats", Json::Int(sequence.run_repeats.into())),
        ("config_enabled", Json::Bool(sequence.config_enabled)),
    ];
    if !sequence.components.is_empty() {
        tree.push(("components", Json::Array(sequence.components.iter().map(component_to_tree).collect())));
    }
    Json::object(tree)
}

/// `sequence_from_tree(tree)`.
pub fn sequence_from_tree(tree: &Json) -> Result<Sequence> {
    let items = object(tree, &["format", "name", "time_step_ns", "target", "periods", "bindings", "delays", "brackets", "run_repeats"],
                       "pulse", &["components", "config_enabled"])?;
    let Json::Str(declared) = member(items, "format") else { return type_error("pulse format must be text"); };
    if declared != PULSE_TREE_FORMAT {
        return value_error(format!("not a {PULSE_TREE_FORMAT} pulse: {}", if declared.is_empty() { "no format" } else { declared }));
    }
    let target = target_from_tree(member(items, "target"))?;
    let periods = array(member(items, "periods"), "pulse periods")?.iter().map(period_from_tree).collect::<Result<Vec<_>>>()?;
    let bindings = array(member(items, "bindings"), "pulse bindings")?.iter().map(binding_from_tree).collect::<Result<Vec<_>>>()?;
    let delays = array(member(items, "delays"), "pulse delays")?.iter().map(delay_from_tree).collect::<Result<Vec<_>>>()?;
    let brackets = array(member(items, "brackets"), "pulse brackets")?.iter().map(bracket_from_tree).collect::<Result<Vec<_>>>()?;
    let config_enabled = match optional(items, "config_enabled") {
        None => bindings.iter().any(|binding| binding.source == Source::Config),
        Some(Json::Bool(flag)) => *flag,
        Some(_) => return type_error("config_enabled must be boolean"),
    };
    let components = match optional(items, "components") {
        None => Vec::new(),
        Some(value) => array(value, "pulse components")?.iter().map(component_from_tree).collect::<Result<Vec<_>>>()?,
    };
    let time_step_ns = match member(items, "time_step_ns") {
        Json::Int(value) => *value as f64,
        Json::Float(value) => *value,
        _ => return value_error("time_step_ns must be positive and finite"),
    };
    let run_repeats = crate::model::nonnegative(integer(member(items, "run_repeats"), "run_repeats")?, "run_repeats")
        .or_else(|error| if member(items, "run_repeats").as_int().is_some_and(|value| value > 0) {
            value_error("run_repeats does not fit the hardware 32-bit count") } else { Err(error) })?;
    let name = text(member(items, "name"), "sequence name")?;
    Sequence::new(SequenceParts { name, target, time_step_ns, periods, bindings, delays, brackets, run_repeats, components, config_enabled })
}

/// `subpulse_to_tree(subpulse)`: its own non-executable grammar.
pub fn subpulse_to_tree(subpulse: &Subpulse) -> Json {
    let Json::Object(items) = sequence_to_tree(subpulse.to_sequence()) else { unreachable!() };
    Json::Object(items.into_iter().filter(|(key, _)| !matches!(key.as_str(), "delays" | "run_repeats" | "config_enabled"))
        .map(|(key, value)| if key == "format" { (key, Json::str(SUBPULSE_TREE_FORMAT)) } else { (key, value) }).collect())
}

pub fn subpulse_from_tree(tree: &Json) -> Result<Subpulse> {
    let items = object(tree, &["format", "name", "time_step_ns", "target", "periods", "bindings", "brackets"], "subpulse", &[])?;
    if member(items, "format").as_str() != Some(SUBPULSE_TREE_FORMAT) { return value_error(format!("not a {SUBPULSE_TREE_FORMAT} subpulse")); }
    let mut full: Vec<(String, Json)> = items.to_vec();
    for (key, value) in &mut full { if key == "format" { *value = Json::str(PULSE_TREE_FORMAT); } }
    full.push(("delays".into(), Json::Array(Vec::new())));
    full.push(("run_repeats".into(), Json::Int(0)));
    let sequence = sequence_from_tree(&Json::Object(full))?;
    Subpulse::new(&sequence.name, sequence.target.clone(), sequence.time_step_ns, sequence.periods.clone(), sequence.bindings.clone(),
                  sequence.brackets.clone())
}

/// `parse_pulse_tree_json(text)`: one JSON object.
pub fn parse_pulse_json(text: &str) -> Result<Json> {
    let value = json::parse(text, "pulse JSON")?;
    if !matches!(value, Json::Object(_)) { return type_error("pulse JSON must contain one object"); }
    Ok(value)
}

pub fn read_subpulse(path: &Path) -> Result<Subpulse> {
    let text = std::fs::read_to_string(path).map_err(|error| Error::new(Kind::Io, format!("{}: {error}", path.display())))?;
    subpulse_from_tree(&parse_pulse_json(&text)?)
}

pub fn write_subpulse(path: &Path, subpulse: &Subpulse) -> Result<()> { atomic_write(path, subpulse_to_tree(subpulse).readable().as_bytes()) }

/// Replace one file atomically: a sibling temporary file renamed over the target.
pub fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) { std::fs::create_dir_all(parent)?; }
    let name = path.file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default();
    let temporary = path.with_file_name(format!(".{name}.{}.tmp", std::process::id()));
    std::fs::write(&temporary, bytes)?;
    if let Err(error) = std::fs::rename(&temporary, path) {
        let _ = std::fs::remove_file(&temporary);
        return Err(error.into());
    }
    Ok(())
}

// ------------------------------------------------------------------ config values
/// One named Config value: its number and unit.
pub type ConfigValues = Vec<(String, (f64, String))>;

/// `_config_value_entry(key, number, unit)`.
pub fn config_value_entry(key: &str, number: &Json, unit: &Json) -> Result<(f64, String)> {
    config_parameter_key(key)?;
    let Some(value) = number.as_f64() else { return type_error(format!("Config value {} must be a number", repr(key))); };
    if !value.is_finite() { return value_error(format!("Config value {} must be finite", repr(key))); }
    match unit.as_str() {
        Some(unit) if !unit.trim().is_empty() => Ok((value, unit.to_owned())),
        _ => value_error(format!("Config value {} must carry a unit", repr(key))),
    }
}

pub fn config_values_to_tree(values: &ConfigValues) -> Result<Json> {
    let mut entries = Vec::new();
    for (key, (number, unit)) in values {
        let (number, unit) = config_value_entry(key, &Json::Float(*number), &Json::str(unit))?;
        entries.push((key.clone(), Json::object(vec![("value", crate::num::number_for(number).to_json()), ("unit", Json::str(unit))])));
    }
    Ok(Json::object(vec![("format", Json::str(CONFIG_VALUES_FORMAT)), ("values", Json::Object(entries))]))
}

pub fn config_values_from_tree(tree: &Json) -> Result<ConfigValues> {
    let items = object(tree, &["format", "values"], "Config values", &[])?;
    if member(items, "format").as_str() != Some(CONFIG_VALUES_FORMAT) {
        return value_error(format!("Config values must declare format {}", repr(CONFIG_VALUES_FORMAT)));
    }
    let Json::Object(values) = member(items, "values") else { return type_error("Config values must be an object"); };
    let mut entries = Vec::new();
    for (key, item) in values {
        let entry = object(item, &["value", "unit"], &format!("Config value {}", repr(key)), &[])?;
        entries.push((key.clone(), config_value_entry(key, member(entry, "value"), member(entry, "unit"))?));
    }
    Ok(entries)
}

pub fn read_config_values(path: &Path) -> Result<ConfigValues> {
    if !path.extension().is_some_and(|extension| extension.eq_ignore_ascii_case("json")) {
        return value_error(format!("config values must be JSON: {}", path.display()));
    }
    let text = std::fs::read_to_string(path).map_err(|error| Error::new(Kind::Io, format!("{}: {error}", path.display())))?;
    config_values_from_tree(&parse_pulse_json(&text)?)
}

pub fn write_config_values(path: &Path, values: &ConfigValues) -> Result<()> {
    atomic_write(path, config_values_to_tree(values)?.readable().as_bytes())
}

// ------------------------------------------------------------------ documents
/// What an editor section means by a field it leaves out.
#[derive(Clone, Debug, PartialEq)]
pub struct EditorSection {
    pub visible_ports: Option<Vec<String>>,
    pub scan_source: String,
    pub scan_rows: Vec<Vec<f64>>,
    pub scan_fields: Vec<(String, String)>,
    pub scan_source_dirty: bool,
    /// Complete scan-table sweeps; 0 runs until Stop.
    pub scan_repeats: u64,
}

impl Default for EditorSection {
    fn default() -> Self {
        EditorSection { visible_ports: None, scan_source: String::new(), scan_rows: Vec::new(), scan_fields: Vec::new(),
                        scan_source_dirty: false, scan_repeats: 0 }
    }
}

const EDITOR_FIELDS: [&str; 6] = ["visible_ports", "scan_source", "scan_rows", "scan_fields", "scan_source_dirty", "scan_repeats"];

/// `split_pulse_document_tree` + `check_pulse_editor_fields`: the sequence
/// tree and its editor half.
pub fn split_document(tree: &Json) -> Result<(Json, EditorSection)> {
    let Json::Object(items) = tree else { return type_error("pulse document must be an object"); };
    let sequence = Json::Object(items.iter().filter(|(key, _)| key != "editor").cloned().collect());
    let Some(editor) = items.iter().find(|(key, _)| key == "editor").map(|(_, value)| value) else { return Ok((sequence, EditorSection::default())); };
    let Json::Object(fields) = editor else { return type_error("pulse editor state must be an object"); };
    let unknown: Vec<&str> = fields.iter().map(|(key, _)| key.as_str()).filter(|key| !EDITOR_FIELDS.contains(key)).collect();
    if !unknown.is_empty() { return value_error(format!("unknown pulse editor field(s): {}", unknown.join(", "))); }
    let get = |key: &str| fields.iter().find(|(name, _)| name == key).map(|(_, value)| value);
    let mut section = EditorSection::default();
    if let Some(value) = get("visible_ports") {
        section.visible_ports = match value {
            Json::Null => None,
            Json::Array(items) => Some(items.iter().map(|item| item.as_str().map(str::to_owned)
                .ok_or_else(|| Error::new(Kind::Type, "visible_ports must be null or a collection of port names"))).collect::<Result<_>>()?),
            _ => return type_error("visible_ports must be null or a collection of port names"),
        };
    }
    if let Some(value) = get("scan_source") {
        section.scan_source = value.as_str().ok_or_else(|| Error::new(Kind::Type, "scan_source must be a string"))?.to_owned();
    }
    if let Some(value) = get("scan_rows") {
        let rows = value.as_array().ok_or_else(|| Error::new(Kind::Type, "scan_rows must be a table"))?;
        section.scan_rows = rows.iter().map(|row| {
            let row = row.as_array().ok_or_else(|| Error::new(Kind::Type, "each scan_rows row must be a sequence of numbers"))?;
            row.iter().map(|value| value.as_f64().ok_or_else(|| Error::new(Kind::Type, "each scan_rows row must be a sequence of numbers"))).collect()
        }).collect::<Result<_>>()?;
    }
    if let Some(value) = get("scan_fields") {
        let pairs = value.as_array().ok_or_else(|| Error::new(Kind::Type, "scan_fields must contain field id/unit pairs"))?;
        section.scan_fields = pairs.iter().map(|pair| match pair.as_array() {
            Some([Json::Str(field), Json::Str(unit)]) if !field.is_empty() && !unit.is_empty() => Ok((field.clone(), unit.clone())),
            _ => type_error("scan_fields must contain field id/unit pairs"),
        }).collect::<Result<_>>()?;
    }
    if let Some(value) = get("scan_source_dirty") {
        section.scan_source_dirty = value.as_bool().ok_or_else(|| Error::new(Kind::Type, "scan_source_dirty must be a boolean"))?;
    }
    if let Some(value) = get("scan_repeats") {
        let repeats = value.as_int().ok_or_else(|| Error::new(Kind::Type, "scan_repeats must be an integer"))?;
        if repeats < 0 { return value_error("scan_repeats must be non-negative"); }
        if repeats > i128::from(crate::model::MAXIMUM_REPEAT_COUNT) { return value_error("scan_repeats does not fit the hardware 32-bit count"); }
        section.scan_repeats = repeats as u64;
    }
    Ok((sequence, section))
}

pub fn editor_section_to_tree(section: &EditorSection, sequence: &Sequence) -> Json {
    Json::object(vec![
        ("visible_ports", match &section.visible_ports {
            None => Json::Null,
            Some(shown) => Json::Array(sequence.target.ports.iter().filter(|port| shown.contains(&port.key)).map(|port| Json::str(&port.key)).collect()),
        }),
        ("scan_source", Json::str(&section.scan_source)),
        ("scan_rows", Json::Array(section.scan_rows.iter().map(|row| Json::Array(row.iter().map(|value| Json::Float(*value)).collect())).collect())),
        ("scan_fields", Json::Array(section.scan_fields.iter().map(|(field, unit)| Json::Array(vec![Json::str(field), Json::str(unit)])).collect())),
        ("scan_source_dirty", Json::Bool(section.scan_source_dirty)),
        ("scan_repeats", Json::Int(section.scan_repeats.into())),
    ])
}

/// `read_pulse_document(path)`: one pulse from disk and its editor half; a
/// relative Subpulse source is resolved beside the file.
pub fn read_pulse_document(path: &Path) -> Result<(Sequence, EditorSection)> {
    let source = absolute(path);
    let text = std::fs::read_to_string(&source).map_err(|error| Error::new(Kind::Io, format!("{}: {error}", source.display())))?;
    let (tree, section) = split_document(&parse_pulse_json(&text)?)?;
    let sequence = sequence_from_tree(&tree)?;
    let parent = source.parent().map(Path::to_path_buf).unwrap_or_default();
    let sequence = sequence.with(|parts| for component in &mut parts.components {
        if !component.config_path.is_empty() && !Path::new(&component.config_path).is_absolute() {
            component.config_path = absolute(&parent.join(&component.config_path)).display().to_string();
        }
    })?;
    Ok((sequence, section))
}

/// A path made absolute without requiring it to exist (`Path.resolve()`):
/// `.` and `..` folded, written the way a person reads it (no verbatim `\\?\` prefix).
pub fn absolute(path: &Path) -> PathBuf {
    let full = std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
    match full.to_str().and_then(|text| text.strip_prefix(r"\\?\")) {
        Some(plain) if !plain.starts_with("UNC") => PathBuf::from(plain),
        _ => full,
    }
}

/// The document a save writes: the sequence tree plus its editor section.
pub fn document_to_tree(sequence: &Sequence, section: &EditorSection) -> Json {
    let Json::Object(mut items) = sequence_to_tree(sequence) else { unreachable!() };
    items.push(("editor".into(), editor_section_to_tree(section, sequence)));
    Json::Object(items)
}
