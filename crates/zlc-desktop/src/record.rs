//! A saved Figure's record as the Figure Viewer reads it (v2
//! `zlc_workbench.viewer.describe_archive`): the info pane's tabs -- what the
//! file holds (Plot), the Logic that made it (Logic), the devices it used
//! (Devices) -- and the Flow graph of its exact
//! saved lineage. Device truth has one owner, the Devices tab; Logic rows
//! carry the non-device run and event facts. An `apparatus` record adds each
//! device at its recorded phase, preserving the capture's source and time.
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use serde_json::{Value, json};
use zlc_core::units::{DEFAULT_UNITS, Number, format_reading};

const PREFIX: &str = "@logic/";
const DEVICE_RECORD_FIELDS: [&str; 5] = ["named_devices", "device_snapshots", "actual_devices", "device_settings", "apparatus"];
/// The scope a bench entry is listed under on the Devices tab.
const BENCH: &str = "Before acquisition";
const SEPARATOR: &str = " · ";

/// One recorded pulse a run of the archive played, offered on the Devices tab.
pub struct PlayedPulse { pub key: String, pub name: String, pub tree: Value }

/// What the info pane shows of one archive.
pub struct Record { pub tabs: Value, pub flow: Value, pub pulses: Vec<PlayedPulse> }

type Rows = Vec<(String, Value)>;

/// `record` is the run's record, shared by every capture of the run (a scan's is tens of kB, its captures thousands).
#[derive(Clone)]
struct Node { id: String, event: Value, parents: Vec<String>, signals: Vec<String>, record: Arc<Value>, event_record: Value }

fn text<'a>(value: &'a Value, key: &str) -> &'a str { value.get(key).and_then(Value::as_str).map(str::trim).unwrap_or("") }

/// `split_signal_key`: (producer, output) of `@logic/<producer>/<output>`.
fn split_signal(signal: &str) -> Option<(&str, &str)> {
    let rest = signal.strip_prefix(PREFIX)?;
    let (owner, output) = rest.rsplit_once('/')?;
    (!owner.is_empty() && !output.is_empty()).then_some((owner, output))
}

fn signal_name(signal: &str) -> String { split_signal(signal).map_or(signal, |(_, output)| output).to_owned() }

/// The root, the nodes (their event records without their acquisitions) and each node's
/// acquisition where it is: a scan's records every shot, and no node carries a copy of that.
fn lineage_nodes(lineage: &Value) -> Result<(Option<String>, Vec<Node>, Vec<(String, &Value)>), String> {
    let root = lineage.get("root").and_then(Value::as_str).map(str::to_owned);
    let mut nodes = Vec::new();
    let mut acquisitions = Vec::new();
    for entry in lineage.get("nodes").and_then(Value::as_array).ok_or("figure lineage nodes must be an array")? {
        let id = entry.get("id").and_then(Value::as_str).filter(|id| !id.is_empty()).ok_or("figure lineage node IDs must be text")?.to_owned();
        if nodes.iter().any(|node: &Node| node.id == id) { return Err("figure lineage node IDs must be unique".into()); }
        let strings = |key: &str| -> Vec<String> { entry.get(key).and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_str).map(str::to_owned).collect() };
        let event_record = match entry.get("event_record") {
            Some(Value::Object(fields)) => {
                if let Some(acquisition) = fields.get("acquisition") { acquisitions.push((id.clone(), acquisition)); }
                let mut kept: serde_json::Map<String, Value> = fields.iter().filter(|(name, _)| *name != "acquisition").map(|(name, value)| (name.clone(), value.clone())).collect();
                if let Some(window) = kept.get_mut("window") {
                    if let Cow::Owned(members) = zlc_core::acquisition::expand_window(window)? { *window = members; }
                }
                Value::Object(kept)
            }
            Some(other) => other.clone(),
            None => json!({}),
        };
        nodes.push(Node { id, event: entry.get("event").cloned().unwrap_or(Value::Null), parents: strings("parents"), signals: strings("signals"),
                          record: Arc::new(entry.get("record").cloned().unwrap_or_else(|| json!({}))), event_record });
    }
    if root.is_none() && !nodes.is_empty() { return Err("empty figure lineage cannot contain nodes".into()); }
    if let Some(root) = &root {
        if !nodes.iter().any(|node| &node.id == root) || nodes.iter().any(|node| node.parents.iter().any(|parent| !nodes.iter().any(|other| &other.id == parent))) {
            return Err("figure lineage refers to an unknown node".into());
        }
    }
    Ok((root, nodes, acquisitions))
}

/// A value's text such that two values are `==` exactly when their texts are equal: object
/// fields in name order (the maps keep their written order but compare as sets), and zero
/// unsigned (`-0.0 == 0.0`). It keys values in hashed sets where `==` would need a search.
fn equality_key(value: &Value) -> String {
    fn write(value: &Value, out: &mut String) {
        match value {
            Value::Object(map) => {
                let mut fields: Vec<_> = map.iter().collect();
                fields.sort_unstable_by(|left, right| left.0.cmp(right.0));
                out.push('{');
                for (index, (name, item)) in fields.into_iter().enumerate() {
                    if index > 0 { out.push(','); }
                    out.push_str(&Value::String(name.clone()).to_string());
                    out.push(':');
                    write(item, out);
                }
                out.push('}');
            }
            Value::Array(items) => {
                out.push('[');
                for (index, item) in items.iter().enumerate() { if index > 0 { out.push(','); } write(item, out); }
                out.push(']');
            }
            Value::Number(number) if number.as_f64() == Some(0.0) && number.is_f64() => out.push_str("0.0"),
            other => out.push_str(&other.to_string()),
        }
    }
    let mut out = String::new();
    write(value, &mut out);
    out
}

/// Finite acquisitions retain each point's facts without adding nodes to the Flow graph.
/// Linear in the captures: ranges are grouped by event and records found by their identity.
fn acquisition_records(nodes: &[Node], acquisitions: &[(String, &Value)]) -> Result<Vec<Node>, String> {
    // (signal, stream, generation, sequence) of a record, by `equality_key`.
    fn identity(signal: &str, event: &Value) -> (String, String, String, String) {
        (signal.to_owned(), equality_key(&event["stream"]), equality_key(&event["generation"]), equality_key(&event["sequence"]))
    }
    let mut records = nodes.to_vec();
    // The first record that answers to each identity (as a search in order would find it).
    let mut found: HashMap<(String, String, String, String), usize> = HashMap::new();
    for (index, record) in records.iter().enumerate() {
        for signal in &record.signals { found.entry(identity(signal, &record.event)).or_insert(index); }
    }
    for (owner, acquisition) in acquisitions {
        let acquisition = zlc_core::acquisition::expand_acquisition(acquisition)?;
        let mut ranges: HashMap<u64, Vec<Value>> = HashMap::new();
        for range in acquisition["ranges"].as_array().into_iter().flatten() {
            if let Some(event) = range["event"].as_u64() { ranges.entry(event).or_default().push(json!({"origin": range["origin"], "extent": range["extent"]})); }
        }
        let runs: Vec<Arc<Value>> = acquisition["runs"].as_array().into_iter().flatten().map(|run| Arc::new(run["record"].clone())).collect();
        for (index, capture) in acquisition["events"].as_array().into_iter().flatten().enumerate() {
            let Some(run_index) = capture["run"].as_u64().map(|index| index as usize).filter(|index| *index < runs.len()) else { continue };
            let run = &acquisition["runs"][run_index];
            let signal = text(capture, "signal");
            let mut event = json!({"stream": run["stream"], "generation": run["generation"], "sequence": capture["revision"], "ordinal": capture["ordinal"]});
            if let Some(ranges) = ranges.remove(&(index as u64)) { event["ranges"] = json!(ranges); }
            let key = identity(signal, &event);
            if let Some(existing) = found.get(&key) {
                records[*existing].event = event;
                continue;
            }
            found.insert(key, records.len());
            records.push(Node { id: owner.clone(), event, parents: Vec::new(), signals: vec![signal.to_owned()], record: runs[run_index].clone(), event_record: capture["record"].clone() });
        }
    }
    let mut order: Vec<(Value, Value)> = Vec::new();
    for record in &records {
        let run = (record.event["stream"].clone(), record.event["generation"].clone());
        if !order.contains(&run) { order.push(run); }
    }
    records.sort_by_cached_key(|record| (order.iter().position(|run| run.0 == record.event["stream"] && run.1 == record.event["generation"]).unwrap(), record.event["sequence"].as_u64().unwrap_or(0)));
    for record in &mut records { if let Some(map) = record.event_record.as_object_mut() { map.remove("acquisition"); } }
    Ok(records)
}

/// The operator-facing Logic identity: its name when the run was built, else its id, else the stream's producer.
fn logic_name(node: &Node) -> String {
    let explicit = [text(&node.record, "node_name"), text(&node.record, "node")].into_iter().find(|name| !name.is_empty());
    if let Some(name) = explicit { return name.to_owned(); }
    let stream = text(&node.event, "stream");
    split_signal(stream).map_or(stream, |(owner, _)| owner).to_owned()
}

/// One label per node, told apart by its event only when a name repeats.
fn logic_labels(nodes: &[Node]) -> Vec<String> {
    let names: Vec<String> = nodes.iter().map(logic_name).collect();
    nodes.iter().zip(&names).map(|(node, name)| {
        if names.iter().filter(|other| *other == name).count() > 1 {
            let generation: String = text(&node.event, "generation").chars().take(8).collect();
            format!("{name}{SEPARATOR}sequence {}{SEPARATOR}{generation}", node.event.get("sequence").cloned().unwrap_or(Value::Null))
        } else { name.clone() }
    }).collect()
}

/// A Task's saved source record not already a lineage node: (task, output, record).
fn source_run_record(source: &Value, nodes: &[Node]) -> Option<(String, String, Value)> {
    let record = source.get("run_record").filter(|record| record.is_object())?;
    let task = [text(record, "node_name"), text(source, "task"), text(record, "node")].into_iter().find(|name| !name.is_empty())?.to_owned();
    let output = ["report", "artifact", "signal"].into_iter().map(|key| text(source, key)).find(|output| !output.is_empty())?.to_owned();
    if nodes.iter().any(|node| *node.record == *record) { return None; }
    Some((task, output, record.clone()))
}

/// The record minus device truth, recursively (the Devices tab owns it).
fn logic_record(value: &Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(map.iter().filter(|(key, _)| !DEVICE_RECORD_FIELDS.contains(&key.as_str()))
            .map(|(key, item)| (key.clone(), logic_record(item))).collect()),
        Value::Array(items) => Value::Array(items.iter().map(logic_record).collect()),
        other => other.clone(),
    }
}

fn event_facts(record: &Value) -> Value {
    let mut projected = logic_record(record);
    if let Some(outputs) = projected.get_mut("records").and_then(Value::as_object_mut) {
        outputs.retain(|_, value| !value.as_object().is_some_and(|fields| fields.is_empty()));
        if outputs.is_empty() { projected.as_object_mut().unwrap().remove("records"); }
    }
    projected
}

fn event_label(event: &Value) -> String {
    if let Some(ranges) = event["ranges"].as_array().filter(|ranges| !ranges.is_empty()) {
        if ranges.len() == 1 { return format!("repeat {}{SEPARATOR}point {}{SEPARATOR}extent {} × {}", ranges[0]["origin"][0], ranges[0]["origin"][1], ranges[0]["extent"][0], ranges[0]["extent"][1]); }
        return format!("{} acquired ranges", ranges.len());
    }
    event.get("ordinal").filter(|value| !value.is_null()).map_or_else(|| format!("event {}", event["sequence"]), |ordinal| format!("sample {ordinal}"))
}

/// A record whose fields read in the order v2 wrote them, not sorted by name:
/// `{"$fields": [[name, value], ...]}` (the info tree's ordered record).  The
/// saved file's own mappings are already in their written (sorted) order.
fn fields(pairs: Vec<(String, Value)>) -> Value {
    json!({"$fields": pairs.into_iter().map(|(name, value)| json!([name, value])).collect::<Vec<_>>()})
}

/// The run's settings and this event's facts have distinct scopes.
fn logic_row(outputs: Vec<String>, record: &Value, event_record: &Value) -> Value {
    let mut row = vec![("outputs".to_owned(), json!(outputs))];
    for (scope, record) in [("run", record), ("event", event_record)] {
        if let Value::Object(mut projected) = if scope == "event" { event_facts(record) } else { logic_record(record) } {
            projected.remove("node"); projected.remove("node_name");
            if !projected.is_empty() { row.push((scope.to_owned(), Value::Object(projected))); }
        }
    }
    fields(row)
}

fn saved_label(name: &str, beside_nodes: bool) -> String { if beside_nodes { format!("{name}{SEPARATOR}saved source") } else { name.to_owned() } }

fn logic_rows(nodes: &[Node], source: &Value) -> Rows {
    let mut rows = Vec::new();
    if let Some((name, output, record)) = source_run_record(source, nodes) {
        rows.push((saved_label(&name, !nodes.is_empty()), logic_row(vec![output], &record, &Value::Null)));
    }
    let mut groups: Vec<Vec<&Node>> = Vec::new();
    for node in nodes {
        if let Some(group) = groups.iter_mut().find(|group| group[0].event["stream"] == node.event["stream"] && group[0].event["generation"] == node.event["generation"]) { group.push(node); }
        else { groups.push(vec![node]); }
    }
    let representatives: Vec<Node> = groups.iter().map(|group| (*group[0]).clone()).collect();
    for (mut group, label) in groups.into_iter().zip(logic_labels(&representatives)) {
        group.sort_by_key(|node| node.event["sequence"].as_u64().unwrap_or(0));
        let first = group[0];
        let mut outputs = Vec::new();
        for node in &group { for signal in &node.signals { let name = signal_name(signal); if !outputs.contains(&name) { outputs.push(name); } } }
        if group.len() == 1 { rows.push((label, logic_row(outputs, &first.record, &first.event_record))); continue; }
        let mut row = logic_row(outputs, &first.record, &Value::Null);
        let events: Rows = group.into_iter().filter_map(|node| {
            let record = event_facts(&node.event_record);
            if record.as_object().is_none_or(|map| map.is_empty()) { return None; }
            let point = event_label(&node.event);
            let label = format!("{}{SEPARATOR}{point}", node.signals.iter().map(|signal| signal_name(signal)).collect::<Vec<_>>().join(", "));
            Some((label, record))
        }).collect();
        if !events.is_empty() { row["$fields"].as_array_mut().unwrap().push(json!(["events", fields(events)])); }
        rows.push((label, row));
    }
    rows
}

fn named_devices(record: &Value) -> Vec<(String, String)> {
    record.get("named_devices").and_then(Value::as_object).into_iter().flatten()
        .filter_map(|(role, key)| Some((role.clone(), key.as_str()?.to_owned()))).collect()
}

/// (role, device) of a record: its named devices, then each actual device not already named.
fn device_associations(record: &Value) -> Vec<(String, String)> {
    let mut result = named_devices(record);
    for key in record.get("actual_devices").and_then(Value::as_object).into_iter().flatten().map(|(key, _)| key) {
        if !result.iter().any(|(_, existing)| existing == key) { result.push((key.clone(), key.clone())); }
    }
    result
}

/// (role, device, snapshot) of a record's device snapshots.
fn record_devices(record: &Value, named: &[(String, String)]) -> Vec<(String, String, Value)> {
    let mut named = named.to_vec();
    for pair in named_devices(record) { if !named.iter().any(|(role, _)| *role == pair.0) { named.push(pair); } }
    let mut result: Vec<(String, String, Value)> = Vec::new();
    for (role, snapshot) in record.get("device_snapshots").and_then(Value::as_object).into_iter().flatten() {
        if let Some((_, key)) = named.iter().find(|(named_role, _)| named_role == role) { result.push((role.clone(), key.clone(), snapshot.clone())); }
    }
    for (key, snapshot) in record.get("actual_devices").and_then(Value::as_object).into_iter().flatten() {
        if !result.iter().any(|(_, existing, _)| existing == key) { result.push((key.clone(), key.clone(), snapshot.clone())); }
    }
    result
}

/// The snapshot as the Devices tab reads it: a played pulse named and counted, a scan table counted.
fn device_tab_snapshot(snapshot: &Value) -> Value {
    let mut shown = snapshot.clone();
    readable_settings(&mut shown);
    if snapshot["snapshot"].is_object() { shown["snapshot"] = device_tab_snapshot(&snapshot["snapshot"]); }
    if let Some(document) = snapshot.get("pulse").filter(|document| document.is_object()) {
        shown["pulse"] = json!({"name": document.get("name").cloned().unwrap_or(Value::Null),
                                "periods": document.get("periods").and_then(Value::as_array).map_or(0, Vec::len)});
    }
    if let Some(rows) = snapshot.get("program").and_then(|program| program.get("rows")).and_then(Value::as_array) {
        shown["program"]["rows"] = json!(rows.len());
    }
    shown
}

/// A bench entry's settings as Device Control reads them: each number with a `units` entry in words,
/// in its readable prefix by the ≈ rule ("1.2345 GHz", "≈ -23.98 dBm"); the archive keeps the exact values.
fn readable_settings(entry: &mut Value) {
    let Some(mut units) = entry.get("units").and_then(Value::as_object).cloned() else { return };
    let Some(values) = entry.get_mut("values").and_then(Value::as_object_mut) else { return };
    for (name, value) in values.iter_mut() {
        let (Some(unit), Some(number)) = (units.get(name).and_then(Value::as_str), value.as_f64()) else { continue };
        let number = match value.as_i64() { Some(integer) => Number::Int(integer.into()), None => Number::Float(number) };
        if let Ok(text) = DEFAULT_UNITS.resolve(unit).and_then(|unit| format_reading(number, &unit, None, None)) { *value = json!(text); units.remove(name); }
    }
    if units.is_empty() { if let Some(fields) = entry.as_object_mut() { fields.remove("units"); } }
    else { entry["units"] = Value::Object(units); }
}

/// The bench a run record says its run started on (`apparatus`): (device, entry), its devices then its pulse board.
fn bench_devices(record: &Value) -> Vec<(String, Value)> {
    ["devices", "sequencer"].into_iter().flat_map(|part| record.get("apparatus").and_then(|bench| bench.get(part)).and_then(Value::as_object).into_iter().flatten())
        .map(|(device, entry)| (device.clone(), entry.clone())).collect()
}

/// Whether the record's own device snapshots keep the program a board entry names (a scan's
/// `device_snapshots.sequencer`): the bench then says only how the board stood.
fn recorded_program(record: &Value, device: &str, entry: &Value) -> bool {
    !entry["digest"].is_null() && record_devices(record, &named_devices(record)).iter()
        .any(|(_, key, snapshot)| key == device && snapshot["program"]["digest"] == entry["digest"])
}

/// (pulse, device, sequence, recorded phase).
fn played_pulses(nodes: &[Node], source: &Value) -> Vec<(PlayedPulse, String, Option<Value>, &'static str)> {
    let mut records: Vec<(String, Option<Value>, String, Value)> = nodes.iter()
        .map(|node| (logic_name(node), node.event.get("sequence").cloned(), text(&node.event, "generation").to_owned(), (*node.record).clone())).collect();
    if let Some((task, _, record)) = source_run_record(source, nodes) { records.push((task, None, String::new(), record)); }
    let mut found = Vec::new();
    for (logic, sequence, generation, record) in records {
        let named = named_devices(&record);
        for (role, device, snapshot) in record_devices(&record, &named) {
            let Some(document) = snapshot.get("pulse").filter(|document| document.is_object()) else { continue };
            let name = record.get("pulse").and_then(|file| file.get("name")).and_then(Value::as_str).filter(|name| !name.is_empty()).map(str::to_owned)
                .or_else(|| document.get("name").and_then(Value::as_str).filter(|name| !name.is_empty()).map(str::to_owned)).unwrap_or(role);
            let key = match &sequence { None => format!("saved-source:{logic}:{device}"), Some(sequence) => format!("{logic}:{sequence}:{generation}:{device}") };
            found.push((PlayedPulse { key, name, tree: document.clone() }, device, sequence.clone(), "pulse"));
        }
        for (device, entry) in bench_devices(&record) {
            let Some(document) = entry["snapshot"].get("pulse").filter(|document| document.is_object() && !recorded_program(&record, &device, &entry)) else { continue };
            let name = document.get("name").and_then(Value::as_str).filter(|name| !name.is_empty()).unwrap_or(&device).to_owned();
            let key = match &sequence { None => format!("saved-source:{logic}:bench:{device}"), Some(sequence) => format!("{logic}:{sequence}:{generation}:bench:{device}") };
            let phase = if record["apparatus"]["phase"] == "acquisition" { "pulse at acquisition" } else { "pulse at Start" };
            found.push((PlayedPulse { key, name, tree: document.clone() }, device, sequence.clone(), phase));
        }
    }
    found
}

fn device_rows(lineage: &Value, nodes: &[Node], source: &Value, played: &[(PlayedPulse, String, Option<Value>, &'static str)]) -> Rows {
    let mut devices: Vec<(String, Value)> = Vec::new();
    fn entry<'a>(devices: &'a mut Vec<(String, Value)>, key: &str) -> &'a mut Value {
        if let Some(index) = devices.iter().position(|(existing, _)| existing == key) { return &mut devices[index].1; }
        devices.push((key.to_owned(), json!({"roles": [], "used_by": [], "snapshots": [], "settings_epochs": []})));
        &mut devices.last_mut().unwrap().1
    }
    fn push_unique(list: &mut Value, item: Value) { if let Some(items) = list.as_array_mut() { if !items.contains(&item) { items.push(item); } } }
    /// Each device section's states and their sources by `equality_key`, so a scan's thousands of
    /// observations are grouped in linear time.
    #[derive(Default)]
    struct Seen { states: HashMap<(String, &'static str, String), usize>, sources: HashSet<(String, &'static str, usize, String)> }
    // Compare exact device state before formatting units or summarizing pulse data.
    fn add_snapshot(devices: &mut Vec<(String, Value)>, seen: &mut Seen, key: &str, section: &'static str, state: Value, origin: Value) {
        let snapshots = entry(devices, key)[section].as_array_mut().unwrap();
        let at = *seen.states.entry((key.to_owned(), section, equality_key(&state))).or_insert_with(|| {
            snapshots.push(json!({"state": state, "sources": []}));
            snapshots.len() - 1
        });
        if seen.sources.insert((key.to_owned(), section, at, equality_key(&origin))) {
            snapshots[at]["sources"].as_array_mut().unwrap().push(origin);
        }
    }
    let mut seen = Seen::default();
    let mut records: Vec<(Value, Value, Vec<(String, String)>)> = Vec::new();
    let mut runs = Vec::new();
    for node in nodes {
        let mut named = named_devices(&node.record);
        for pair in named_devices(&node.event_record) { if !named.iter().any(|(role, _)| *role == pair.0) { named.push(pair); } }
        for (scope, record) in [("run", &*node.record), ("event", &node.event_record)] {
            let run = (node.event["stream"].clone(), node.event["generation"].clone());
            if scope == "run" {
                if runs.contains(&run) { continue; }
                runs.push(run);
            }
            let origin = json!({"logic": logic_name(node), "scope": scope, "node_id": node.id, "event": node.event});
            records.push((origin.clone(), record.clone(), named.clone()));
            if scope == "event" {
                for signal in &node.signals {
                    if let Some(output) = record["records"].get(signal_name(signal)) {
                        let mut origin = origin.clone(); origin["signal"] = json!(signal);
                        records.push((origin, output.clone(), named.clone()));
                    }
                }
            }
        }
    }
    if let Some((task, _, record)) = source_run_record(source, nodes) {
        let named = named_devices(&record);
        let mut identity = source.clone();
        if let Some(map) = identity.as_object_mut() { map.remove("run_record"); }
        records.push((json!({"logic": task, "scope": "task run", "source": identity}), record, named));
    }
    for (origin, record, named) in &records {
        for (role, key) in device_associations(record) {
            let facts = entry(&mut devices, &key);
            push_unique(&mut facts["roles"], json!(role));
            push_unique(&mut facts["used_by"], origin["logic"].clone());
        }
        for (role, key, snapshot) in record_devices(record, named) {
            let facts = entry(&mut devices, &key);
            push_unique(&mut facts["roles"], json!(role));
            push_unique(&mut facts["used_by"], origin["logic"].clone());
            add_snapshot(&mut devices, &mut seen, &key, "snapshots", snapshot, origin.clone());
        }
        for (role, reference) in record["device_settings"].as_object().into_iter().flatten() {
            if let Some((_, key)) = named.iter().find(|(name, _)| name == role) {
                add_snapshot(&mut devices, &mut seen, key, "settings_epochs", reference.clone(), origin.clone());
            }
        }
    }
    // Capture metadata identifies the observation, not the device's effective state.
    for (origin, record, _) in &records {
        for (key, stood) in bench_devices(record) {
            let mut source = origin.clone();
            source["scope"] = json!(if record["apparatus"]["phase"] == "acquisition" { "Acquisition state" } else { BENCH });
            for field in ["phase", "context", "taken_at", "completed_at"] {
                if let Some(value) = record["apparatus"].get(field) { source[field] = value.clone(); }
            }
            if let Some(transport) = record["apparatus"]["device_transports"].get(&key) { source["transport"] = transport.clone(); }
            let mut state = stood;
            if let Some(map) = state.as_object_mut() {
                for field in ["type_id", "role", "driver"] {
                    if let Some(value) = map.remove(field) { source[field] = value; }
                }
            }
            add_snapshot(&mut devices, &mut seen, &key, "snapshots", state, source);
        }
    }
    for item in lineage.get("device_settings").and_then(Value::as_array).into_iter().flatten() {
        let key = text(item, "device_key");
        if key.is_empty() { continue; }
        add_snapshot(&mut devices, &mut seen, key, "settings_epochs", item.clone(), json!({"logic": key, "scope": "recorded settings"}));
    }
    let mut rows: Rows = devices.into_iter().map(|(key, mut facts)| {
        for section in ["snapshots", "settings_epochs"] {
          let mut snapshots = Vec::new();
          let mut snapshot_labels = HashSet::new();
          for group in facts[section].as_array().unwrap() {
            let mut sources = Vec::new();
            let mut source_labels = HashSet::new();
            let mut names = Vec::new();
            let mut named = HashSet::new();
            for origin in group["sources"].as_array().unwrap() {
                let mut name = format!("{}{SEPARATOR}{}", text(origin, "logic"), text(origin, "scope"));
                let common = name.clone();
                if origin["scope"] == "event" {
                    let point = event_label(&origin["event"]);
                    name.push_str(&format!("{SEPARATOR}{point}"));
                }
                let title = if group["sources"].as_array().unwrap().len() == 1 { &name } else { &common };
                if named.insert(title.clone()) { names.push(title.clone()); }
                let mut label = name;
                if let Some(time) = origin.get("taken_at").and_then(Value::as_str) { label.push_str(&format!("{SEPARATOR}{time}")); }
                if source_labels.contains(&label) { label.push_str(&format!("{SEPARATOR}record {}", sources.len() + 1)); }
                source_labels.insert(label.clone());
                sources.push((label, origin.clone()));
            }
            let mut label = names.join(" + ");
            if names.len() < sources.len() { label.push_str(&format!("{SEPARATOR}{} observations", sources.len())); }
            if snapshot_labels.contains(&label) { label.push_str(&format!("{SEPARATOR}state {}", snapshots.len() + 1)); }
            snapshot_labels.insert(label.clone());
            snapshots.push((label, fields(vec![("state".into(), device_tab_snapshot(&group["state"])), ("sources".into(), fields(sources))])));
          }
          facts[section] = if snapshots.is_empty() { json!([]) } else { fields(snapshots) };
        }
        (key, fields(["roles", "used_by", "snapshots", "settings_epochs"].into_iter()
            .filter(|name| !facts[*name].as_array().is_some_and(Vec::is_empty))
            .map(|name| (name.to_owned(), facts[name].clone())).collect()))
    }).collect();
    // The pulse a run played is offered, not printed: it opens in the Pulse Editor.
    for (pulse, device, sequence, what) in played {
        let label = match sequence { None => format!("{device} {what}"), Some(sequence) => format!("{device} {what} {sequence}") };
        rows.push((label, json!({"text": pulse.name, "action": format!("pulse:{}", pulse.key)})));
    }
    rows
}

/// `_lineage_graph`: the exact saved DAG as unique Logic and Device nodes and their edges.
fn lineage_graph(root: Option<&str>, nodes: &[Node], source: &Value) -> Result<Value, String> {
    let mut graph_nodes: Vec<Value> = Vec::new();
    let mut edges: Vec<Value> = Vec::new();
    let device_node = |key: &str, roles: &[String], users: &[String]| json!({"id": format!("device:{key}"), "kind": "device", "title": key,
        "subtitle": roles.join(", "), "root": false, "tooltip": format!("Device: {key}\nRoles: {}\nUsed by: {}", roles.join(", "), users.join(", ")),
        "row": ["Devices", key]});
    let saved = source_run_record(source, nodes);
    let Some(root) = root else {
        let Some((name, output, record)) = saved else { return Ok(json!({"nodes": [], "edges": []})) };
        graph_nodes.push(json!({"id": "logic:source", "kind": "logic", "title": name, "subtitle": output, "root": true,
                                "tooltip": format!("Logic: {name}\nSaved result: {output}"), "row": ["Logic", saved_label(&name, false)]}));
        let mut grouped: Vec<(String, Vec<String>)> = Vec::new();
        for (role, key) in device_associations(&record) {
            match grouped.iter_mut().find(|(existing, _)| *existing == key) { Some((_, roles)) => roles.push(role), None => grouped.push((key, vec![role])) }
        }
        for (key, roles) in grouped {
            graph_nodes.push(device_node(&key, &roles, std::slice::from_ref(&name)));
            edges.push(json!({"source": format!("device:{key}"), "target": "logic:source", "kind": "device", "label": roles.join(", ")}));
        }
        return Ok(json!({"nodes": graph_nodes, "edges": edges}));
    };
    // Parents before children, from the root (a cycle or a stray node is malformed).
    let index_of = |id: &str| nodes.iter().position(|node| node.id == id);
    let mut order: Vec<usize> = Vec::new();
    let mut visiting: Vec<usize> = Vec::new();
    fn visit(at: usize, nodes: &[Node], order: &mut Vec<usize>, visiting: &mut Vec<usize>, index_of: &dyn Fn(&str) -> Option<usize>) -> Result<(), String> {
        if visiting.contains(&at) { return Err("figure lineage contains a cycle".into()); }
        if order.contains(&at) { return Ok(()); }
        visiting.push(at);
        for parent in &nodes[at].parents { visit(index_of(parent).ok_or("figure lineage refers to an unknown node")?, nodes, order, visiting, index_of)?; }
        visiting.retain(|item| *item != at);
        order.push(at);
        Ok(())
    }
    visit(index_of(root).ok_or("figure lineage refers to an unknown node")?, nodes, &mut order, &mut visiting, &index_of)?;
    if order.len() != nodes.len() { return Err("figure lineage contains nodes outside the root graph".into()); }
    let labels = logic_labels(nodes);
    let mut devices: Vec<(String, Vec<String>, Vec<String>)> = Vec::new();
    let mut device_edges: Vec<(String, String, String)> = Vec::new();
    for at in order {
        let node = &nodes[at];
        let name = logic_name(node);
        let signals: Vec<String> = node.signals.iter().map(|signal| signal_name(signal)).collect();
        let stream = text(&node.event, "stream");
        let outputs = if signals.is_empty() { signal_name(stream) } else { signals.join(", ") };
        graph_nodes.push(json!({"id": format!("logic:{}", node.id), "kind": "logic", "title": name,
            "subtitle": format!("{outputs}{}", &labels[at][name.len()..]), "root": node.id == root,
            "tooltip": format!("Logic: {name}\nOutputs: {}", if node.signals.is_empty() { stream.to_owned() } else { node.signals.join(", ") }),
            "row": ["Logic", labels[at]]}));
        for parent in &node.parents {
            edges.push(json!({"source": format!("logic:{parent}"), "target": format!("logic:{}", node.id), "kind": "causal", "label": ""}));
        }
        let mut associations = named_devices(&node.record);
        for pair in named_devices(&node.event_record) {
            if let Some((_, existing)) = associations.iter().find(|(role, _)| *role == pair.0) {
                if *existing != pair.1 { return Err(format!("device role {:?} changes inside one Logic event", pair.0)); }
            } else { associations.push(pair); }
        }
        for pair in device_associations(&node.record).into_iter().chain(device_associations(&node.event_record)) {
            if !associations.contains(&pair) { associations.push(pair); }
        }
        for (role, key) in associations {
            match devices.iter_mut().find(|(existing, ..)| *existing == key) {
                Some((_, roles, users)) => { if !roles.contains(&role) { roles.push(role.clone()); } if !users.contains(&name) { users.push(name.clone()); } }
                None => devices.push((key.clone(), vec![role.clone()], vec![name.clone()])),
            }
            let edge = (key.clone(), node.id.clone(), role.clone());
            if !device_edges.contains(&edge) {
                edges.push(json!({"source": format!("device:{key}"), "target": format!("logic:{}", node.id), "kind": "device", "label": role}));
                device_edges.push(edge);
            }
        }
    }
    for (key, roles, users) in &devices { graph_nodes.push(device_node(key, roles, users)); }
    if let Some((name, output, record)) = saved {
        graph_nodes.push(json!({"id": "logic:saved-source", "kind": "logic", "title": name, "subtitle": format!("{output}{SEPARATOR}saved source"),
            "root": false, "tooltip": format!("Logic: {name}\nSaved result: {output}\nThis Task record has no exact Runtime event edge in the archive."),
            "row": ["Logic", saved_label(&name, true)]}));
        let mut grouped: Vec<(String, Vec<String>)> = Vec::new();
        for (role, key) in device_associations(&record) {
            match grouped.iter_mut().find(|(existing, _)| *existing == key) { Some((_, roles)) => roles.push(role), None => grouped.push((key, vec![role])) }
        }
        for (key, roles) in grouped {
            if !devices.iter().any(|(existing, ..)| *existing == key) { graph_nodes.push(device_node(&key, &roles, std::slice::from_ref(&name))); devices.push((key.clone(), roles.clone(), vec![name.clone()])); }
            edges.push(json!({"source": format!("device:{key}"), "target": "logic:saved-source", "kind": "device", "label": roles.join(", ")}));
        }
    }
    Ok(json!({"nodes": graph_nodes, "edges": edges}))
}

/// numpy's dtype name for an array-protocol descriptor (`<f4` → float32).
fn dtype_name(descriptor: &str) -> String {
    let body = descriptor.trim_start_matches(['<', '>', '|', '=']);
    let (kind, size) = body.split_at(body.find(|c: char| c.is_ascii_digit()).unwrap_or(body.len()));
    let bytes: usize = size.parse().unwrap_or(0);
    match kind {
        "b" => "bool".into(), "i" => format!("int{}", bytes * 8), "u" => format!("uint{}", bytes * 8), "f" => format!("float{}", bytes * 8),
        "c" => format!("complex{}", bytes * 8), "U" => format!("str{}", bytes * 32), "S" => format!("bytes{}", bytes * 8),
        _ => descriptor.to_owned(),
    }
}

/// `_plot_rows`: every array in the file (shape, dtype, whether a panel reopens it), then each panel's plot.
fn plot_rows(info: &Value, plot: &Value) -> Rows {
    let members: BTreeMap<&String, &Value> = info.get("members").and_then(Value::as_object).into_iter().flatten().collect();
    let mut rows: Rows = members.into_iter().map(|(name, member)| {
        let shape = member.get("shape").and_then(Value::as_array).into_iter().flatten().map(|size| size.to_string()).collect::<Vec<_>>().join("x");
        let dtype = dtype_name(text(member, "dtype"));
        let reopenable = if plot.get(name.as_str()).is_some() { "" } else { " · array only" };
        (name.clone(), json!(format!("{shape} {dtype}{reopenable}").trim().to_owned()))
    }).collect();
    for (dataset, recipe) in plot.as_object().into_iter().flatten() {
        rows.push((format!("plot {dataset}"), recipe.clone()));
    }
    rows
}

fn rows_json(rows: Rows) -> Value { Value::Array(rows.into_iter().map(|(label, value)| json!([label, value])).collect()) }

/// The info pane's tabs before the record is read: what the file holds, which no lineage is needed for.
pub fn unread_tabs(info: &Value) -> Value {
    json!([["Plot", rows_json(plot_rows(info, &info["sections"]["plot"]))], ["Logic", [["record", "reading this figure's record\u{2026}"]]], ["Devices", []], ["Flow", []]])
}

/// How an archive's datasets read in the Data chooser (v2: `title — signal`, or either).
pub fn label(source: &Value) -> String {
    let (title, signal) = (text(source, "title"), text(source, "signal"));
    if !title.is_empty() && !signal.is_empty() && title != signal { format!("{title} — {signal}") }
    else if !title.is_empty() { title.to_owned() } else { signal.to_owned() }
}

/// The archive's record for the info pane. A lineage the viewer cannot read
/// leaves its Logic, Devices and Flow empty and says why on the Logic tab;
/// the file's arrays and plot configuration are still shown.
pub fn describe(info: &Value) -> Record {
    let sections = &info["sections"];
    let (lineage, source) = (&sections["lineage"], &sections["source"]);
    present(info, lineage_nodes(lineage).and_then(|(root, nodes, acquisitions)| {
        let flow = lineage_graph(root.as_deref(), &nodes, source)?;
        let played = played_pulses(&nodes, source);
        let records = acquisition_records(&nodes, &acquisitions)?;
        let devices = device_rows(lineage, &records, source, &played);
        Ok((logic_rows(&records, source), devices, flow, played.into_iter().map(|(pulse, ..)| pulse).collect()))
    }))
}

/// The record of a figure whose lineage cannot be read at all.
pub fn unreadable(info: &Value, error: &str) -> Record { present(info, Err(error.to_owned())) }

fn present(info: &Value, described: Result<(Rows, Rows, Value, Vec<PlayedPulse>), String>) -> Record {
    let (logic, devices, flow, pulses) = described.unwrap_or_else(|error| (vec![("record".to_owned(), json!(format!("cannot read this figure's lineage: {error}")))],
        Vec::new(), json!({"nodes": [], "edges": []}), Vec::new()));
    let tabs = json!([["Plot", rows_json(plot_rows(info, &info["sections"]["plot"]))], ["Logic", rows_json(logic)], ["Devices", rows_json(devices)],
                      ["Flow", []]]);
    Record { tabs, flow, pulses }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field<'a>(value: &'a Value, name: &str) -> &'a Value {
        &value["$fields"].as_array().unwrap().iter().find(|pair| pair[0] == name).unwrap()[1]
    }

    #[test]
    fn record_tabs_keep_event_facts_and_plot_configuration_without_raw() {
        let info = json!({"members": {"data": {"dtype": "<f4", "shape": [1, 6, 96, 128]}}, "sections": {
            "dataset": {"data": {}}, "plot": {"data": {"size": "2x2", "spec": {"kind": "facet_grid"}}},
            "lineage": {"root": "event-1", "device_settings": [], "nodes": [{"id": "event-1", "parents": [],
                "event": {"generation": "2c3491d0f54c", "sequence": 480, "stream": "camera_measurement"},
                "event_record": {"frame": {"index": 4}, "device_settings": {"camera": {"mixed": false}}},
                "record": {"node": "camera_measurement", "named_devices": {"camera": "camera"}, "parameters": {"repeat": 0},
                           "device_snapshots": {"camera": {"exposure_seconds": 0.02}}},
                "signals": ["@logic/camera_measurement/frames"]}]},
            "source": {"signal": "@logic/camera_measurement/frames", "title": "@logic/camera_measurement/frames"}}});
        let record = describe(&info);
        assert_eq!(record.tabs[0][1], json!([["data", "1x6x96x128 float32"], ["plot data", {"size": "2x2", "spec": {"kind": "facet_grid"}}]]));
        assert_eq!(record.tabs[1][1], json!([["camera_measurement", {"$fields": [["outputs", ["frames"]], ["run", {"parameters": {"repeat": 0}}], ["event", {"frame": {"index": 4}}]]}]]));
        assert_eq!(record.tabs[2][1][0][0], "camera");
        let facts = &record.tabs[2][1][0][1]["$fields"];
        assert_eq!(facts.as_array().unwrap().iter().map(|pair| pair[0].as_str().unwrap()).collect::<Vec<_>>(), ["roles", "used_by", "snapshots", "settings_epochs"]);
        let snapshot = field(&facts[2][1], "camera_measurement · run");
        assert_eq!(field(snapshot, "state"), &json!({"exposure_seconds": 0.02}));
        assert_eq!(field(field(snapshot, "sources"), "camera_measurement · run"), &json!({"logic": "camera_measurement", "scope": "run", "node_id": "event-1",
            "event": {"generation": "2c3491d0f54c", "sequence": 480, "stream": "camera_measurement"}}));
        assert_eq!(record.flow["nodes"].as_array().unwrap().len(), 2);
        assert_eq!(record.flow["nodes"][0]["subtitle"], "frames");
        assert_eq!(record.flow["edges"], json!([{"source": "device:camera", "target": "logic:event-1", "kind": "device", "label": "camera"}]));
        assert_eq!(record.tabs.as_array().unwrap().iter().map(|tab| tab[0].as_str().unwrap()).collect::<Vec<_>>(), ["Plot", "Logic", "Devices", "Flow"]);
    }

    /// A run record's `apparatus`: every device as it stood at Start on the Devices tab (not on
    /// the Logic tab), the board's pulse offered once -- not again when the run's own record
    /// keeps the same program.
    #[test]
    fn the_bench_at_start_reads_on_the_devices_tab() {
        let pulse = json!({"name": "imaging", "periods": [{}, {}]});
        let board = json!({"type_id": "sequencer.virtual", "role": "sequencer", "driver": {"outside": "PulseGUI"}, "loaded": true, "firing": true,
            "digest": "5f0c", "snapshot": {"program": {"digest": "5f0c", "rows": [[1], [2], [3]]}, "pulse": pulse}});
        let bench = json!({"taken_at": "2026-10-25T12:00:00.000+00:00", "sequencer": {"sequencer": board},
            "devices": {"rf": {"type_id": "rf.virtual", "role": "Trap RF", "values": {"ch1_frequency": 8.0e7, "ch1_power": -23.97940008672037, "ch1_output": true},
                "units": {"ch1_frequency": "Hz", "ch1_power": "dBm"}, "provenance": {"settings_epoch": 2}}}});
        let info = |record: Value| json!({"members": {}, "sections": {"dataset": {}, "plot": {}, "source": {"signal": "@logic/camera_measurement/frames"},
            "lineage": {"root": "event-1", "device_settings": [], "nodes": [{"id": "event-1", "parents": [], "event_record": {},
                "event": {"generation": "2c3491d0f54c", "sequence": 7, "stream": "camera_measurement"}, "record": record,
                "signals": ["@logic/camera_measurement/frames"]}]}}});
        let record = describe(&info(json!({"node": "camera_measurement", "named_devices": {"camera": "mot_camera"}, "apparatus": bench})));
        assert_eq!(record.tabs[1][1], json!([["camera_measurement", {"$fields": [["outputs", ["frames"]]]}]]), "device truth is the Devices tab's");
        let rows = record.tabs[2][1].as_array().unwrap();
        let labels: Vec<&str> = rows.iter().map(|row| row[0].as_str().unwrap()).collect();
        assert_eq!(labels, ["mot_camera", "rf", "sequencer", "sequencer pulse at Start 7"]);
        let snapshot = field(field(&rows[1][1], "snapshots"), "camera_measurement · Before acquisition");
        assert_eq!(field(snapshot, "state"), &json!({"values": {"ch1_frequency": "80 MHz", "ch1_power": "\u{2248} -23.98 dBm", "ch1_output": true},
            "provenance": {"settings_epoch": 2}}), "each number in words, with its unit");
        let origin = field(snapshot, "sources")["$fields"][0][1].clone();
        assert_eq!((origin["taken_at"].as_str(), origin["type_id"].as_str(), origin["role"].as_str()), (Some("2026-10-25T12:00:00.000+00:00"), Some("rf.virtual"), Some("Trap RF")));
        assert_eq!(field(&field(&rows[2][1], "snapshots")["$fields"][0][1], "state")["snapshot"], json!({"program": {"digest": "5f0c", "rows": 3}, "pulse": {"name": "imaging", "periods": 2}}));
        assert_eq!(rows[3][1], json!({"text": "imaging", "action": "pulse:camera_measurement:7:2c3491d0f54c:bench:sequencer"}));
        assert_eq!(record.pulses[0].tree, pulse);
        // A scan's record keeps the program it played: the same one on the board is not offered twice.
        let record = describe(&info(json!({"node": "seamless_scan", "named_devices": {"sequencer": "sequencer"}, "apparatus": bench,
            "device_snapshots": {"sequencer": {"program": {"digest": "5f0c"}, "pulse": pulse}}})));
        let rows = record.tabs[2][1].as_array().unwrap();
        assert_eq!(rows.iter().map(|row| row[0].as_str().unwrap()).collect::<Vec<_>>(), ["sequencer", "rf", "sequencer pulse 7"]);
        let shown = field(field(&rows[0][1], "snapshots"), "seamless_scan · Before acquisition");
        assert_eq!(field(shown, "state")["digest"], json!("5f0c"));
        assert!(field(shown, "state")["snapshot"].is_object(), "the board state remains separate from the run's played program");
        assert_eq!(record.pulses.len(), 1);
    }

    #[test]
    fn equal_device_states_keep_named_origins_and_distinct_readbacks_stay_separate() {
        let node = |id: &str, name: &str, generation: &str, frequency: f64, parent: Option<&str>| json!({
            "id": id, "parents": parent.into_iter().collect::<Vec<_>>(), "signals": [format!("@logic/{name}/data")],
            "event": {"generation": generation, "sequence": 403, "stream": name}, "event_record": {},
            "record": {"node": name, "apparatus": {"phase": "acquisition", "context": "scan-run", "taken_at": format!("{id}-begin"), "completed_at": format!("{id}-end"),
                "device_transports": {"rf": {"kind": "fabric", "endpoint": {"host": "lab-pc", "device": "rf"}}},
                "devices": {"rf": {"type_id": "rf.virtual", "role": "Trap RF", "values": {"frequency": frequency}, "units": {"frequency": "Hz"}}}}}
        });
        let info = json!({"members": {}, "sections": {"plot": {}, "source": {}, "lineage": {"root": "third", "nodes": [
            node("first", "camera", "camera-generation", 80_000_000.0, None),
            node("second", "scan", "scan-generation", 80_000_000.0, Some("first")),
            node("third", "scan", "different-generation", 80_000_000.0001, Some("second"))
        ]}}});
        let original = info.clone();
        let record = describe(&info);
        assert_eq!(info, original, "display projection never rewrites saved metadata");
        let snapshots = &field(&record.tabs[2][1][0][1], "snapshots")["$fields"];
        assert_eq!(snapshots.as_array().unwrap().len(), 2, "compare exact readings, never their formatted labels");
        assert_eq!(snapshots[0][0], "camera · Acquisition state + scan · Acquisition state");
        let sources = &field(&snapshots[0][1], "sources")["$fields"];
        assert_eq!(sources.as_array().unwrap().len(), 2);
        for (index, id, generation) in [(0, "first", "camera-generation"), (1, "second", "scan-generation")] {
            let source = &sources[index][1];
            assert_eq!(source["node_id"], id);
            assert_eq!(source["event"]["generation"], generation);
            assert_eq!(source["context"], "scan-run");
            assert_eq!(source["taken_at"], format!("{id}-begin"));
            assert_eq!(source["completed_at"], format!("{id}-end"));
            assert_eq!(source["transport"]["endpoint"]["host"], "lab-pc");
        }
        assert_eq!(field(&snapshots[0][1], "state")["values"]["frequency"], "80 MHz");
        assert!(!snapshots.to_string().contains("Before acquisition"));
    }

    /// Before the record is read the pane shows what the file holds, and the chooser its label,
    /// whatever the lineage: an unreadable one is not looked at.
    #[test]
    fn what_the_file_holds_reads_without_the_lineage() {
        let info = json!({"members": {"data": {"dtype": "<f4", "shape": [2, 3]}}, "sections": {"plot": {"data": {"size": "2x2"}}, "lineage": "not a lineage",
            "source": {"signal": "@logic/camera/frames", "title": "a scan"}}});
        let tabs = unread_tabs(&info);
        assert_eq!(tabs[0][1], describe(&info).tabs[0][1], "the Plot tab is the record's own");
        assert_eq!(tabs.as_array().unwrap().iter().map(|tab| tab[0].as_str().unwrap()).collect::<Vec<_>>(), ["Plot", "Logic", "Devices", "Flow"]);
        assert_eq!(label(&info["sections"]["source"]), "a scan — @logic/camera/frames");
    }

    #[test]
    fn acquisition_pulse_and_legacy_start_pulse_keep_their_recorded_phase() {
        let document = json!({"name": "imaging", "periods": []});
        let node = |id: &str, acquisition: bool| {
            let mut apparatus = json!({"sequencer": {"seq": {"snapshot": {"pulse": document}}}});
            if acquisition { apparatus["phase"] = json!("acquisition"); }
            json!({"id": id, "parents": if acquisition { vec!["before"] } else { vec![] }, "signals": [],
                "event": {"generation": id, "sequence": 1, "stream": id}, "record": {"node": id, "apparatus": apparatus}, "event_record": {}})
        };
        let record = describe(&json!({"members": {}, "sections": {"plot": {}, "source": {}, "lineage": {"root": "during", "nodes": [node("before", false), node("during", true)]}}}));
        let labels: Vec<_> = record.tabs[2][1].as_array().unwrap().iter().map(|row| row[0].as_str().unwrap()).collect();
        assert_eq!(labels, ["seq", "seq pulse at Start 1", "seq pulse at acquisition 1"]);
        assert_eq!(record.pulses.len(), 2);
        assert!(record.pulses.iter().all(|pulse| pulse.tree == document));
    }

    #[test]
    fn finite_acquisition_keeps_all_working_points_and_camera_epochs_without_duplicate_tail() {
        let scan_run = json!({"node": "scan", "named_devices": {"rf": "rf"}});
        let camera_run = json!({"node": "camera", "named_devices": {"camera": "cam"}});
        let scan_event = |point: usize| json!({"records": {"scan": {"point": point, "actual_devices": {"rf": {"values": {"frequency": 80.0 + point as f64}, "units": {"frequency": "MHz"}}}},
            "other": {"actual_devices": {"rf": {"values": {"frequency": 999}, "units": {"frequency": "MHz"}}}}}});
        let camera_event = |epoch: usize| json!({"records": {"frames": {"device_settings": {"camera": {"device_session_id": "camera-session", "epoch_ranges": [[epoch, epoch]], "mixed": false}}}}});
        let acquisition = json!({"runs": [{"stream": "scan", "generation": "scan-gen", "record": scan_run}, {"stream": "camera", "generation": "camera-gen", "record": camera_run}],
            "events": [
                {"run": 1, "signal": "@logic/camera/frames", "revision": 11, "ordinal": 0, "record": camera_event(1), "parents": []},
                {"run": 0, "signal": "@logic/scan/scan", "revision": 1, "ordinal": 0, "record": scan_event(0), "parents": [0]},
                {"run": 1, "signal": "@logic/camera/frames", "revision": 12, "ordinal": 1, "record": camera_event(2), "parents": []},
                {"run": 0, "signal": "@logic/scan/scan", "revision": 2, "ordinal": 1, "record": scan_event(1), "parents": [2]}],
            "ranges": [{"event": 1, "origin": [0, 0], "extent": [1, 1]}, {"event": 3, "origin": [0, 1], "extent": [1, 1]}]});
        let mut latest = scan_event(1); latest["acquisition"] = acquisition;
        let info = json!({"members": {}, "sections": {"plot": {}, "source": {}, "lineage": {"root": "scan-node", "nodes": [
            {"id": "camera-node", "parents": [], "signals": ["@logic/camera/frames"], "event": {"stream": "camera", "generation": "camera-gen", "sequence": 12}, "record": camera_run, "event_record": camera_event(2)},
            {"id": "scan-node", "parents": ["camera-node"], "signals": ["@logic/scan/scan"], "event": {"stream": "scan", "generation": "scan-gen", "sequence": 2}, "record": scan_run, "event_record": latest}
        ]}}});
        let original = info.clone();
        let record = describe(&info);
        assert_eq!(info, original);
        let logic = record.tabs[1][1].as_array().unwrap();
        assert_eq!(logic.len(), 2, "one Logic row per actual run, not per historical point");
        let scan = &logic.iter().find(|row| row[0] == "scan").unwrap()[1];
        let events = &field(scan, "events")["$fields"];
        assert_eq!(events.as_array().unwrap().len(), 2);
        assert!(events[0][0].as_str().unwrap().contains("point 0"));
        assert_eq!(events[0][1]["records"]["scan"]["point"], 0);
        assert!(!scan.to_string().contains("actual_devices"));
        let devices = record.tabs[2][1].as_array().unwrap();
        let rf = &devices.iter().find(|row| row[0] == "rf").unwrap()[1];
        let states = &field(rf, "snapshots")["$fields"];
        assert_eq!(states.as_array().unwrap().len(), 2, "both point readbacks, with neither duplicated tail nor another output's state");
        for (index, frequency) in [(0, "80 MHz"), (1, "81 MHz")] {
            assert_eq!(field(&states[index][1], "state")["values"]["frequency"], frequency);
            let source = &field(&states[index][1], "sources")["$fields"];
            assert_eq!(source.as_array().unwrap().len(), 1);
            assert_eq!(source[0][1]["event"]["ranges"][0]["origin"], json!([0, index]));
        }
        let camera = &devices.iter().find(|row| row[0] == "cam").unwrap()[1];
        let epochs = &field(camera, "settings_epochs")["$fields"];
        assert_eq!(epochs.as_array().unwrap().len(), 2, "camera settings references from both parent frames are retained");
        assert_eq!(field(&epochs[0][1], "state")["epoch_ranges"], json!([[1, 1]]));
        assert_eq!(field(&epochs[1][1], "state")["epoch_ranges"], json!([[2, 2]]));
        assert_eq!(record.flow["nodes"].as_array().unwrap().iter().filter(|node| node["kind"] == "logic").count(), 2);
    }

    /// A window's event facts read on the Logic tab, each member and its ancestors whole, as the
    /// file held them before its runs were shared.
    #[test]
    fn a_stored_window_reads_as_its_members() {
        let run = json!({"node": "camera", "device_snapshots": {"camera": {"exposure_seconds": 0.02}}});
        let member = |shot: i64| json!({"publication": {"stream": "camera", "generation": "camera-gen", "signal": "@logic/camera/frames", "revision": shot + 1, "ordinal": shot},
            "run": run, "event": {"records": {"frames": {"index": shot}}}, "parents": []});
        let trees = json!({"width": 4, "members": (0..4).map(member).collect::<Vec<_>>()});
        let columns = zlc_core::acquisition::encode_window(&trees).unwrap();
        assert!(columns.to_string().len() < trees.to_string().len());
        let info = |window: &Value| json!({"members": {}, "sections": {"plot": {}, "source": {}, "lineage": {"root": "n", "device_settings": [], "nodes": [
            {"id": "n", "parents": [], "signals": ["@logic/camera/frames"], "event": {"stream": "camera", "generation": "camera-gen", "sequence": 4}, "record": run,
             "event_record": {"window": window}}]}}});
        let (from_trees, from_columns) = (describe(&info(&trees)), describe(&info(&columns)));
        assert_eq!(from_trees.tabs, from_columns.tabs);
        assert!(from_columns.tabs.to_string().contains("\"index\":3"), "{}", from_columns.tabs);
    }

    /// The record tabs read an acquisition in its columns as they read it event by event, whatever
    /// its shots and records.
    #[test]
    fn stored_columns_describe_as_the_events_they_hold() {
        let run = json!({"node": "scan", "named_devices": {"rf": "rf"}});
        let reading = |shot: usize| json!({"records": {"scan": {"point": shot, "actual_devices": {"rf": {"values": {"frequency": 80.0 + (shot % 3) as f64}, "units": {"frequency": "MHz"}}}}}});
        let mut events = Vec::new();
        for shot in 0..12usize {
            if shot == 7 { continue; }
            let parents = if events.is_empty() { json!([]) } else { json!([events.len() - 1]) };
            events.push(json!({"run": 0, "signal": "@logic/scan/scan", "revision": shot + 1, "ordinal": shot, "record": reading(shot), "parents": parents}));
        }
        let ranges: Vec<Value> = (0..events.len()).map(|event| json!({"event": event, "origin": [event / 4, event % 4], "extent": [1, 1]})).collect();
        let list = json!({"runs": [{"stream": "scan", "generation": "scan-gen", "record": run}], "events": events, "ranges": ranges});
        let columns = zlc_core::acquisition::encode_acquisition(&list).unwrap();
        assert!(columns.get("format").is_some());
        let info = |acquisition: &Value| json!({"members": {}, "sections": {"plot": {}, "source": {}, "lineage": {"root": "scan-node", "device_settings": [], "nodes": [
            {"id": "scan-node", "parents": [], "signals": ["@logic/scan/scan"], "event": {"stream": "scan", "generation": "scan-gen", "sequence": 12}, "record": run,
             "event_record": {"records": {"scan": {"point": 11}}, "acquisition": acquisition}}]}}});
        let (from_list, from_columns) = (describe(&info(&list)), describe(&info(&columns)));
        assert_eq!(from_list.tabs, from_columns.tabs);
        assert_eq!(from_list.flow, from_columns.flow);
        assert!(!from_columns.tabs.to_string().contains("cannot read"));
    }

    /// A scan's captures are found by identity and grouped by exact state in hashed sets: the result
    /// is the one the ordered searches gave, maps equal in another field order still one state.
    #[test]
    fn scan_captures_group_by_equal_state_and_keep_each_point() {
        let run = json!({"node": "scan", "named_devices": {"rf": "rf"}});
        let reading = |reversed: bool| if reversed { json!({"units": {"frequency": "MHz"}, "values": {"frequency": 80.0}}) }
            else { json!({"values": {"frequency": 80.0}, "units": {"frequency": "MHz"}}) };
        let event = |point: usize| json!({"records": {"scan": {"point": point, "actual_devices": {"rf": reading(point == 2)}}}});
        let capture = |revision: usize| json!({"run": 0, "signal": "@logic/scan/scan", "revision": revision, "ordinal": revision - 1, "record": event(revision - 1), "parents": []});
        let mut latest = event(2);
        latest["acquisition"] = json!({"runs": [{"stream": "scan", "generation": "scan-gen", "record": run}],
            "events": [capture(1), capture(2), capture(3)],
            "ranges": [{"event": 0, "origin": [0, 0], "extent": [1, 1]}, {"event": 1, "origin": [0, 1], "extent": [1, 1]}, {"event": 0, "origin": [1, 0], "extent": [1, 1]}]});
        let info = json!({"members": {}, "sections": {"plot": {}, "source": {}, "lineage": {"root": "scan-node", "nodes": [
            {"id": "scan-node", "parents": [], "signals": ["@logic/scan/scan"], "event": {"stream": "scan", "generation": "scan-gen", "sequence": 3}, "record": run, "event_record": latest}
        ]}}});
        let record = describe(&info);
        let rf = &record.tabs[2][1].as_array().unwrap().iter().find(|row| row[0] == "rf").unwrap()[1];
        let states = &field(rf, "snapshots")["$fields"];
        assert_eq!(states.as_array().unwrap().len(), 1, "one state, whatever order its fields were written in");
        assert_eq!(states[0][0], "scan · event · 3 observations");
        assert_eq!(field(&states[0][1], "state")["values"]["frequency"], "80 MHz");
        let sources: Vec<_> = field(&states[0][1], "sources")["$fields"].as_array().unwrap().iter().map(|source| source[0].as_str().unwrap().to_owned()).collect();
        assert_eq!(sources, ["scan · event · 2 acquired ranges", "scan · event · repeat 0 · point 1 · extent 1 × 1", "scan · event · sample 2"]);
        let events = &field(&record.tabs[1][1][0][1], "events")["$fields"];
        assert_eq!(events.as_array().unwrap().len(), 3, "each capture once, the lineage node's own event not again");
    }
}
