//! A figure's causal history (`zlc_science.figure.capture_figure_provenance`):
//! the Runtime publications behind a read (`Read::provenance`) as v2's Figure
//! DAG, with an imported figure's saved DAG and source kept whole as the
//! boundary they are.

use std::collections::{HashMap, HashSet};

use crate::error::{ArchiveError, Result};
use crate::json::Json;

/// Where a mounted figure's saved lineage and source travel in its run record.
pub const IMPORTED_LINEAGE: &str = "zlc.figure.imported-lineage";

/// The deepest causal chain `visit` (two interpreter frames a level) walks
/// from a script's top level before Python raises `RecursionError`.
const MAX_DEPTH: usize = 498;

/// `(stream, generation, sequence)`; the sequence is any Python int, keyed by its decimal text.
type Identity = (String, String, String);

#[derive(Default)]
struct Capture {
    /// Python's dict: a later assignment replaces the earlier one.
    identities: HashMap<Identity, Option<String>>,
    nodes: Vec<Json>,
    node_index: HashMap<String, usize>,
    run_records: HashMap<(String, String), Json>,
    settings: Vec<Json>,
    source: Option<Json>,
    visiting: HashSet<Identity>,
    imported: HashSet<Identity>,
}

fn field<'a>(value: &'a Json, key: &str) -> Result<&'a Json> {
    value.get(key).ok_or_else(|| ArchiveError::new(crate::ErrorKind::Key, crate::json::py_repr_str(key)))
}

impl Capture {
    fn add_node(&mut self, id: String, node: Json) {
        self.node_index.insert(id, self.nodes.len());
        self.nodes.push(node);
    }

    fn visit(&mut self, current: &Json, depth: usize) -> Result<Option<String>> {
        if depth > MAX_DEPTH { return Err(ArchiveError::new(crate::ErrorKind::Recursion, "maximum recursion depth exceeded")); }
        let publication = field(current, "publication")?;
        let (stream, generation, sequence) = match (field(publication, "stream")?, field(publication, "generation")?, field(publication, "revision")?) {
            (Json::Str(stream), Json::Str(generation), sequence @ (Json::Int(_) | Json::BigInt(_))) => (stream.clone(), generation.clone(), sequence.clone()),
            _ => return Err(ArchiveError::type_error("native lineage event identity is malformed")),
        };
        let key = match &sequence { Json::Int(value) => value.to_string(), Json::BigInt(text) => text.clone(), _ => unreachable!("matched an int") };
        let event = Json::object([("stream", Json::Str(stream.clone())), ("generation", Json::Str(generation.clone())), ("sequence", sequence)]);
        let identity: Identity = (stream.clone(), generation.clone(), key);
        let signal = field(publication, "signal")?.clone();
        if let Some(node_id) = self.identities.get(&identity).cloned() {
            if let (false, Some(node_id)) = (self.imported.contains(&identity), &node_id) {
                let index = self.node_index[node_id];
                if let Json::Object(pairs) = &mut self.nodes[index] {
                    if let Some((_, Json::Array(signals))) = pairs.iter_mut().find(|(key, _)| key == "signals") {
                        if !signals.contains(&signal) { signals.push(signal); }
                    }
                }
            }
            return Ok(node_id);
        }
        if let Some(boundary) = imported(current)? {
            self.imported.insert(identity.clone());
            let source = boundary.get("source").unwrap().clone();
            if self.source.as_ref().is_some_and(|inherited| *inherited != source) { return Err(ArchiveError::value("imported Figure source documents differ")); }
            self.source = Some(source);
            let lineage = boundary.get("lineage").unwrap();
            self.settings.extend(lineage.get("device_settings").unwrap().as_array().unwrap().iter().cloned());
            let nodes = lineage.get("nodes").unwrap().as_array().unwrap();
            let root = match lineage.get("root").unwrap() {
                Json::Null => {
                    if !nodes.is_empty() { return Err(ArchiveError::value("empty imported Figure lineage contains nodes")); }
                    self.identities.insert(identity, None);
                    return Ok(None);
                }
                Json::Str(root) => root.clone(),
                _ => return Err(ArchiveError::value("imported Figure lineage needs one root")),
            };
            for raw in nodes {
                let Some(Json::Str(node_id)) = raw.as_object().and(raw.get("id")) else {
                    return Err(ArchiveError::type_error("imported Figure lineage node is malformed"));
                };
                match self.node_index.get(node_id) {
                    Some(index) if self.nodes[*index] != *raw => return Err(ArchiveError::value("imported Figure lineage node ids collide")),
                    Some(_) => {}
                    None => self.add_node(node_id.clone(), raw.clone()),
                }
            }
            if !self.node_index.contains_key(&root) { return Err(ArchiveError::value("imported Figure lineage root is missing")); }
            self.identities.insert(identity, Some(root.clone()));
            return Ok(Some(root));
        }
        if self.visiting.contains(&identity) { return Err(ArchiveError::value("Runtime causal publications contain a cycle")); }
        self.visiting.insert(identity.clone());
        let mut parents: Vec<String> = Vec::new();
        let mut seen = HashSet::new();
        let Json::Array(parent_trees) = field(current, "parents")? else { return Err(ArchiveError::type_error("native lineage parents are malformed")) };
        for parent in parent_trees {
            if let Some(node_id) = self.visit(parent, depth + 1)? { if seen.insert(node_id.clone()) { parents.push(node_id); } }
        }
        self.visiting.remove(&identity);
        let mut serial = self.nodes.len() + 1;
        let mut node_id = format!("event-{serial}");
        while self.node_index.contains_key(&node_id) { serial += 1; node_id = format!("event-{serial}"); }
        self.identities.insert(identity, Some(node_id.clone()));
        let run_key = (stream, generation);
        if !self.run_records.contains_key(&run_key) {
            let mut run = field(current, "run")?.clone();
            run.remove(IMPORTED_LINEAGE);
            self.run_records.insert(run_key.clone(), run);
        }
        let run = self.run_records[&run_key].clone();
        let mut event_record = field(current, "event")?.clone();
        if let Some(window) = current.get("window") { event_record.insert("window", window.clone()); }
        self.add_node(node_id.clone(), Json::object([
            ("id", Json::Str(node_id.clone())), ("event", event), ("parents", Json::Array(parents.into_iter().map(Json::Str).collect())),
            ("signals", Json::Array(vec![signal])), ("record", run), ("event_record", event_record),
        ]));
        Ok(Some(node_id))
    }
}

/// The imported-figure boundary a run record carries, validated.
fn imported(current: &Json) -> Result<Option<&Json>> {
    let Some(candidate) = field(current, "run")?.get(IMPORTED_LINEAGE) else { return Ok(None) };
    if candidate.is_null() { return Ok(None); }
    let keys = |value: &Json, expected: &[&str]| value.as_object().is_some_and(|pairs| pairs.len() == expected.len() && expected.iter().all(|key| value.get(key).is_some()));
    if !keys(candidate, &["lineage", "source"]) { return Err(ArchiveError::value("imported Figure payload requires lineage and source")); }
    if candidate.get("source").unwrap().as_object().is_none() { return Err(ArchiveError::type_error("imported Figure source must be a mapping")); }
    let lineage = candidate.get("lineage").unwrap();
    if !keys(lineage, &["root", "nodes", "device_settings"]) { return Err(ArchiveError::value("imported Figure lineage fields differ")); }
    if lineage.get("nodes").unwrap().as_array().is_none() || lineage.get("device_settings").unwrap().as_array().is_none() {
        return Err(ArchiveError::type_error("imported Figure lineage arrays are malformed"));
    }
    Ok(Some(candidate))
}

/// `capture_figure_provenance(tree)`: the lineage DAG and the inherited source document.
pub fn capture_figure_provenance(tree: Option<&Json>) -> Result<(Json, Json)> {
    let empty = Json::object([("root", Json::Null), ("nodes", Json::Array(Vec::new())), ("device_settings", Json::Array(Vec::new()))]);
    let Some(tree) = tree else { return Ok((empty, Json::Object(Vec::new()))) };
    let mut capture = Capture::default();
    let root = capture.visit(tree, 1)?;
    let lineage = Json::object([
        ("root", root.map_or(Json::Null, Json::Str)),
        ("nodes", Json::Array(capture.nodes)),
        ("device_settings", Json::Array(capture.settings)),
    ]);
    Ok((lineage, capture.source.unwrap_or(Json::Object(Vec::new()))))
}
