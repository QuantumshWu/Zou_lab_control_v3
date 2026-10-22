//! A saved Figure's record as the Figure Viewer reads it (v2
//! `zlc_workbench.viewer.describe_archive`): the info pane's tabs -- what the
//! file holds (Plot), the Logic that made it (Logic), the devices it used
//! (Devices), the document itself (Raw) -- and the Flow graph of its exact
//! saved lineage. Device truth has one owner, the Devices tab; Logic rows
//! carry everything else of their run records.
use std::collections::BTreeMap;

use serde_json::{Value, json};

const PREFIX: &str = "@logic/";
const DEVICE_RECORD_FIELDS: [&str; 4] = ["named_devices", "device_snapshots", "actual_devices", "device_settings"];
const SEPARATOR: &str = " · ";

/// One recorded pulse a run of the archive played, offered on the Devices tab.
pub struct PlayedPulse { pub key: String, pub name: String, pub tree: Value }

/// What the info pane shows of one archive.
/// `label` is how its datasets read in the Data chooser (v2: `title — signal`, or either).
pub struct Record { pub tabs: Value, pub flow: Value, pub pulses: Vec<PlayedPulse>, pub label: String }

type Rows = Vec<(String, Value)>;

struct Node { id: String, event: Value, parents: Vec<String>, signals: Vec<String>, record: Value, event_record: Value }

fn text<'a>(value: &'a Value, key: &str) -> &'a str { value.get(key).and_then(Value::as_str).map(str::trim).unwrap_or("") }

/// `split_signal_key`: (producer, output) of `@logic/<producer>/<output>`.
fn split_signal(signal: &str) -> Option<(&str, &str)> {
    let rest = signal.strip_prefix(PREFIX)?;
    let (owner, output) = rest.rsplit_once('/')?;
    (!owner.is_empty() && !output.is_empty()).then_some((owner, output))
}

fn signal_name(signal: &str) -> String { split_signal(signal).map_or(signal, |(_, output)| output).to_owned() }

fn lineage_nodes(lineage: &Value) -> Result<(Option<String>, Vec<Node>), String> {
    let root = lineage.get("root").and_then(Value::as_str).map(str::to_owned);
    let mut nodes = Vec::new();
    for entry in lineage.get("nodes").and_then(Value::as_array).ok_or("figure lineage nodes must be an array")? {
        let id = entry.get("id").and_then(Value::as_str).filter(|id| !id.is_empty()).ok_or("figure lineage node IDs must be text")?.to_owned();
        if nodes.iter().any(|node: &Node| node.id == id) { return Err("figure lineage node IDs must be unique".into()); }
        let strings = |key: &str| -> Vec<String> { entry.get(key).and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_str).map(str::to_owned).collect() };
        nodes.push(Node { id, event: entry.get("event").cloned().unwrap_or(Value::Null), parents: strings("parents"), signals: strings("signals"),
                          record: entry.get("record").cloned().unwrap_or_else(|| json!({})), event_record: entry.get("event_record").cloned().unwrap_or_else(|| json!({})) });
    }
    if root.is_none() && !nodes.is_empty() { return Err("empty figure lineage cannot contain nodes".into()); }
    if let Some(root) = &root {
        if !nodes.iter().any(|node| &node.id == root) || nodes.iter().any(|node| node.parents.iter().any(|parent| !nodes.iter().any(|other| &other.id == parent))) {
            return Err("figure lineage refers to an unknown node".into());
        }
    }
    Ok((root, nodes))
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
    if nodes.iter().any(|node| &node.record == record) { return None; }
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

/// A record whose fields read in the order v2 wrote them, not sorted by name:
/// `{"$fields": [[name, value], ...]}` (the info tree's ordered record).  The
/// saved file's own mappings are already in their written (sorted) order.
fn fields(pairs: Vec<(String, Value)>) -> Value {
    json!({"$fields": pairs.into_iter().map(|(name, value)| json!([name, value])).collect::<Vec<_>>()})
}

/// v2 `_logic_rows`: the outputs first, then the run's own record.
fn logic_row(outputs: Vec<String>, record: &Value) -> Value {
    let mut row = vec![("outputs".to_owned(), json!(outputs))];
    if let Value::Object(projected) = logic_record(record) {
        row.extend(projected.into_iter().filter(|(key, _)| key != "node" && key != "node_name"));
    }
    fields(row)
}

fn saved_label(name: &str, beside_nodes: bool) -> String { if beside_nodes { format!("{name}{SEPARATOR}saved source") } else { name.to_owned() } }

fn logic_rows(nodes: &[Node], source: &Value) -> Rows {
    let mut rows = Vec::new();
    if let Some((name, output, record)) = source_run_record(source, nodes) {
        rows.push((saved_label(&name, !nodes.is_empty()), logic_row(vec![output], &record)));
    }
    for (node, label) in nodes.iter().zip(logic_labels(nodes)) {
        rows.push((label, logic_row(node.signals.iter().map(|signal| signal_name(signal)).collect(), &node.record)));
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
    if let Some(document) = snapshot.get("pulse").filter(|document| document.is_object()) {
        shown["pulse"] = json!({"name": document.get("name").cloned().unwrap_or(Value::Null),
                                "periods": document.get("periods").and_then(Value::as_array).map_or(0, Vec::len)});
    }
    if let Some(rows) = snapshot.get("program").and_then(|program| program.get("rows")).and_then(Value::as_array) {
        shown["program"]["rows"] = json!(rows.len());
    }
    shown
}

fn played_pulses(nodes: &[Node], source: &Value) -> Vec<(PlayedPulse, String, Option<Value>)> {
    let mut records: Vec<(String, Option<Value>, String, Value)> = nodes.iter()
        .map(|node| (logic_name(node), node.event.get("sequence").cloned(), text(&node.event, "generation").to_owned(), node.record.clone())).collect();
    if let Some((task, _, record)) = source_run_record(source, nodes) { records.push((task, None, String::new(), record)); }
    let mut found = Vec::new();
    for (logic, sequence, generation, record) in records {
        let named = named_devices(&record);
        for (role, device, snapshot) in record_devices(&record, &named) {
            let Some(document) = snapshot.get("pulse").filter(|document| document.is_object()) else { continue };
            let name = record.get("pulse").and_then(|file| file.get("name")).and_then(Value::as_str).filter(|name| !name.is_empty()).map(str::to_owned)
                .or_else(|| document.get("name").and_then(Value::as_str).filter(|name| !name.is_empty()).map(str::to_owned)).unwrap_or(role);
            let key = match &sequence { None => format!("saved-source:{logic}:{device}"), Some(sequence) => format!("{logic}:{sequence}:{generation}:{device}") };
            found.push((PlayedPulse { key, name, tree: document.clone() }, device, sequence.clone()));
        }
    }
    found
}

fn device_rows(lineage: &Value, nodes: &[Node], source: &Value, played: &[(PlayedPulse, String, Option<Value>)]) -> Rows {
    let mut devices: Vec<(String, Value)> = Vec::new();
    fn entry<'a>(devices: &'a mut Vec<(String, Value)>, key: &str) -> &'a mut Value {
        if let Some(index) = devices.iter().position(|(existing, _)| existing == key) { return &mut devices[index].1; }
        devices.push((key.to_owned(), json!({"roles": [], "used_by": [], "snapshots": [], "settings_epochs": []})));
        &mut devices.last_mut().unwrap().1
    }
    fn push_unique(list: &mut Value, item: Value) { if let Some(items) = list.as_array_mut() { if !items.contains(&item) { items.push(item); } } }
    let mut records: Vec<(String, Option<Value>, &str, Value, Vec<(String, String)>)> = Vec::new();
    for node in nodes {
        let mut named = named_devices(&node.record);
        for pair in named_devices(&node.event_record) { if !named.iter().any(|(role, _)| *role == pair.0) { named.push(pair); } }
        let sequence = node.event.get("sequence").cloned();
        records.push((logic_name(node), sequence.clone(), "run", node.record.clone(), named.clone()));
        records.push((logic_name(node), sequence, "event", node.event_record.clone(), named));
    }
    if let Some((task, _, record)) = source_run_record(source, nodes) { let named = named_devices(&record); records.push((task, None, "task run", record, named)); }
    for (logic, sequence, scope, record, named) in &records {
        for (role, key) in device_associations(record) {
            let facts = entry(&mut devices, &key);
            push_unique(&mut facts["roles"], json!(role));
            push_unique(&mut facts["used_by"], json!(logic));
        }
        for (role, key, snapshot) in record_devices(record, named) {
            let facts = entry(&mut devices, &key);
            push_unique(&mut facts["roles"], json!(role));
            push_unique(&mut facts["used_by"], json!(logic));
            let mut candidate = json!({"logic": logic, "scope": scope});
            if let Some(sequence) = sequence { candidate["sequence"] = sequence.clone(); }
            candidate["snapshot"] = device_tab_snapshot(&snapshot);
            push_unique(&mut facts["snapshots"], candidate);
        }
    }
    for item in lineage.get("device_settings").and_then(Value::as_array).into_iter().flatten() {
        let key = text(item, "device_key");
        if key.is_empty() { continue; }
        if let Some(settings) = entry(&mut devices, key)["settings_epochs"].as_array_mut() { settings.push(item.clone()); }
    }
    // v2 `_device_rows`: roles, used_by, snapshots, settings_epochs, in that order.
    let mut rows: Rows = devices.into_iter().map(|(key, facts)| (key, fields(["roles", "used_by", "snapshots", "settings_epochs"].into_iter()
        .map(|name| (name.to_owned(), facts[name].clone())).collect()))).collect();
    // The pulse a run played is offered, not printed: it opens in the Pulse Editor.
    for (pulse, device, sequence) in played {
        let label = match sequence { None => format!("{device} pulse"), Some(sequence) => format!("{device} pulse {sequence}") };
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
        rows.push((format!("plot {dataset}"), json!(format!("{}, {}", text(&recipe["spec"], "kind"), text(recipe, "size")))));
    }
    rows
}

fn rows_json(rows: Rows) -> Value { Value::Array(rows.into_iter().map(|(label, value)| json!([label, value])).collect()) }

/// The archive's record for the info pane. A lineage the viewer cannot read
/// leaves its Logic, Devices and Flow empty and says why on the Logic tab;
/// what the file holds (Plot, Raw) is still shown.
pub fn describe(info: &Value) -> Record {
    let sections = &info["sections"];
    let (plot, lineage, source) = (&sections["plot"], &sections["lineage"], &sections["source"]);
    let raw = ["dataset", "plot", "lineage", "source"].into_iter().map(|name| (name.to_owned(), sections.get(name).cloned().unwrap_or(Value::Null))).collect();
    let described = lineage_nodes(lineage).and_then(|(root, nodes)| {
        let flow = lineage_graph(root.as_deref(), &nodes, source)?;
        let played = played_pulses(&nodes, source);
        let devices = device_rows(lineage, &nodes, source, &played);
        Ok((logic_rows(&nodes, source), devices, flow, played.into_iter().map(|(pulse, ..)| pulse).collect()))
    });
    let (logic, devices, flow, pulses) = described.unwrap_or_else(|error| (vec![("record".to_owned(), json!(format!("cannot read this figure's lineage: {error}")))],
        Vec::new(), json!({"nodes": [], "edges": []}), Vec::new()));
    let tabs = json!([["Plot", rows_json(plot_rows(info, plot))], ["Logic", rows_json(logic)], ["Devices", rows_json(devices)],
                      ["Flow", []], ["Raw", rows_json(raw)]]);
    let (title, signal) = (text(source, "title"), text(source, "signal"));
    let label = if !title.is_empty() && !signal.is_empty() && title != signal { format!("{title} — {signal}") }
        else if !title.is_empty() { title.to_owned() } else { signal.to_owned() };
    Record { tabs, flow, pulses, label }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_camera_archive_reads_as_v2_describes_it() {
        let info = json!({"members": {"data": {"dtype": "<f4", "shape": [1, 6, 96, 128]}}, "sections": {
            "dataset": {"data": {}}, "plot": {"data": {"size": "2x2", "spec": {"kind": "facet_grid"}}},
            "lineage": {"root": "event-1", "device_settings": [], "nodes": [{"id": "event-1", "parents": [],
                "event": {"generation": "2c3491d0f54c", "sequence": 480, "stream": "camera_measurement"},
                "event_record": {"device_settings": {"camera": {"mixed": false}}},
                "record": {"node": "camera_measurement", "named_devices": {"camera": "camera"}, "parameters": {"repeat": 0},
                           "device_snapshots": {"camera": {"exposure_seconds": 0.02}}},
                "signals": ["@logic/camera_measurement/frames"]}]},
            "source": {"signal": "@logic/camera_measurement/frames", "title": "@logic/camera_measurement/frames"}}});
        let record = describe(&info);
        assert_eq!(record.tabs[0][1], json!([["data", "1x6x96x128 float32"], ["plot data", "facet_grid, 2x2"]]));
        assert_eq!(record.tabs[1][1], json!([["camera_measurement", {"$fields": [["outputs", ["frames"]], ["parameters", {"repeat": 0}]]}]]));
        assert_eq!(record.tabs[2][1][0][0], "camera");
        let facts = &record.tabs[2][1][0][1]["$fields"];
        assert_eq!(facts.as_array().unwrap().iter().map(|pair| pair[0].as_str().unwrap()).collect::<Vec<_>>(), ["roles", "used_by", "snapshots", "settings_epochs"]);
        assert_eq!(facts[2][1][0], json!({"logic": "camera_measurement", "scope": "run", "sequence": 480, "snapshot": {"exposure_seconds": 0.02}}));
        assert_eq!(record.flow["nodes"].as_array().unwrap().len(), 2);
        assert_eq!(record.flow["nodes"][0]["subtitle"], "frames");
        assert_eq!(record.flow["edges"], json!([{"source": "device:camera", "target": "logic:event-1", "kind": "device", "label": "camera"}]));
        assert_eq!(record.tabs[4][1][3][0], "source");
    }
}
