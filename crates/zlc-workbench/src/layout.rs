//! The board (panel order and the board-only facts of each panel) and its
//! one persisted document, `zlc.console-board`, read and written as v2 does:
//! values are intersected with today's declarations, rows that no longer fit
//! are kept with a note, panel ids are minted fresh and every reference to a
//! saved panel is respelled.
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde_json::{Map, Value, json};

use zlc_core::control::node_instance;

use crate::logic::{Draft, Instance, Node, name_problem, name_text, signal_key, split_signal};
use crate::{Host, Workbench, fields, workspace};

pub const FORMAT: &str = "zlc.console-board";
const KINDS: [&str; 5] = ["image", "curve", "rolling", "histogram", "facet_grid"];
/// v2's panel update intervals (live_policy.refresh_intervals_ms); the first is the default and the board's beat.
pub const INTERVALS: [u64; 4] = [100, 200, 400, 800];
/// v2 panel fields the board keeps as written (v3 reads them as selectors).
const KEPT: [&str; 2] = ["selector", "crosshair"];
/// The contract of a per-point status signal paired with an image (zlc_data.point_overlay).
const OVERLAY_CONTRACT: &str = "zlc_plot.image-point-overlay-status";

pub struct BoardPanel {
    pub panel: String,
    /// The operator's name; `None` is automatic (the signal, spelled with node names).
    pub title: Option<String>,
    pub interval_ms: u64,
    /// The judgement signal whose site rings the panel's image is drawn with; empty: none.
    pub overlay_signal: String,
    pub published_outputs: Map<String, Value>,
    pub kept: Map<String, Value>,
    /// The v3 selectors the kept v2 `selector`/`crosshair` were read as.
    pub loaded_selectors: Option<Value>,
}

#[derive(Default)]
pub struct Board {
    pub panels: Vec<BoardPanel>,
    pub paused: bool,
    pub selectors: bool,
    pub path: Option<PathBuf>,
    pub dirty: bool,
    /// A replacement waiting for the runs it stopped, and what it says when it is done.
    pub pending: Option<(Loaded, &'static str)>,
    /// A recipe's nodes waiting to start, and who ran it (their runs act for that client).
    pub recipe_start: Option<(Vec<String>, String)>,
}

impl Board {
    pub fn add(&mut self, panel: &str) {
        if self.get(panel).is_none() {
            self.panels.push(BoardPanel { panel: panel.to_owned(), title: None, interval_ms: 100, overlay_signal: String::new(), published_outputs: Map::new(),
                kept: Map::new(), loaded_selectors: None });
            self.dirty = true;
        }
    }
    pub fn remove(&mut self, panel: &str) {
        let before = self.panels.len();
        self.panels.retain(|entry| entry.panel != panel);
        if self.panels.len() != before { self.dirty = true; }
    }
    pub fn position(&self, panel: &str) -> Option<usize> { self.panels.iter().position(|entry| entry.panel == panel) }
    pub fn get(&self, panel: &str) -> Option<&BoardPanel> { self.panels.iter().find(|entry| entry.panel == panel) }
    fn get_mut(&mut self, panel: &str) -> Result<&mut BoardPanel, String> {
        self.panels.iter_mut().find(|entry| entry.panel == panel).ok_or_else(|| format!("No panel {panel:?} on the board"))
    }
}

/// A resolved document on fresh panel ids, waiting to replace the board.
/// `renamed`: the saved id of each panel to its id on this board.
pub struct Loaded { pub nodes: Vec<LoadedNode>, pub panels: Vec<LoadedPanel>, pub notes: Vec<String>, pub path: Option<PathBuf>, pub renamed: BTreeMap<String, String> }
pub struct LoadedNode { pub id: String, pub name: String, pub type_id: String, pub draft: Draft, pub auto_preview: bool }
pub struct LoadedPanel { pub panel: String, pub signal: String, pub kind: String, pub window: u64, pub target: Value, pub board: BoardPanel }

fn object<'a>(value: &'a Value, where_: &str) -> Result<&'a Map<String, Value>, String> {
    value.as_object().ok_or_else(|| format!("{where_} must be an object"))
}

fn string(value: Option<&Value>, default: &str, where_: &str) -> Result<String, String> {
    match value { None => Ok(default.to_owned()), Some(Value::String(text)) => Ok(text.clone()), Some(_) => Err(format!("{where_} must be a string")) }
}

/// A saved panel's id and entry, when it can be a panel of this board (v2 `_panel_from_tree`); Err says why not.
fn panel_entry<'a>(entry: &'a Value, kept: &[(String, &Map<String, Value>)]) -> Result<(String, &'a Map<String, Value>), String> {
    let entry = object(entry, "it")?;
    let id = string(entry.get("panel_id"), "", "its panel_id")?.trim().to_owned();
    if id.is_empty() { return Err("it requires panel_id".into()); }
    if kept.iter().any(|(other, _)| *other == id) { return Err(format!("another panel entry is also {id}")); }
    let kind = entry.get("kind").and_then(Value::as_str).ok_or("panel state requires a text kind")?;
    if !KINDS.contains(&kind) { return Err(format!("{kind:?} is not a Task Console plot kind")); }
    for key in ["title", "signal", "cell_kind", "overlay_signal"] { string(entry.get(key), "", &format!("its {key}"))?; }
    Ok((id, entry))
}

/// `@logic/<saved panel>/<output>` respelled to the fresh id; a panel the file
/// does not carry drops the connection (with a note).
fn respell(reference: &str, renamed: &BTreeMap<String, String>, who: &str, notes: &mut Vec<String>) -> String {
    let Some((producer, output)) = split_signal(reference) else { return reference.to_owned() };
    let (head, rest) = producer.split_once('/').map_or((producer, None), |(head, rest)| (head, Some(rest)));
    if let Some(fresh) = renamed.get(head) {
        return signal_key(&rest.map_or_else(|| fresh.clone(), |rest| format!("{fresh}/{rest}")), output);
    }
    if head.strip_prefix("panel-").is_some_and(|digits| !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit())) {
        notes.push(format!("{who} read {reference}, a panel this board does not carry; the connection was dropped"));
        return String::new();
    }
    reference.to_owned()
}

/// v2's `selector`/`crosshair` as v3 selectors. An area names its two ranges
/// by axis id: the panel's own `x`/`y` fates say which is which, else the
/// image axes' spatial roles do.
fn selectors_from_v2(selector: &Value, crosshair: &Value, semantic: &Map<String, Value>) -> (Vec<Value>, Option<String>) {
    let mut selectors = Vec::new();
    let mut problem = None;
    let role_of = |axis: &str, domain: &str| -> Option<&'static str> {
        let fate = semantic.get(&format!("fate:{domain}:{axis}")).and_then(Value::as_str);
        match fate { Some("x") => Some("x"), Some("y") => Some("y"), _ => if axis.ends_with("spatial-x") { Some("x") } else if axis.ends_with("spatial-y") { Some("y") } else { None } }
    };
    let facet_index = selector["facets"].as_array().and_then(|facets| facets.first()).and_then(|facet| facet["value"].as_f64())
        .filter(|value| value.fract() == 0.0 && *value >= 0.0).map(|value| json!(value as u64)).unwrap_or(Value::Null);
    let ranges = selector["ranges"].as_array().cloned().unwrap_or_default();
    match selector["selector_kind"].as_str() {
        Some("area") => {
            let mut area = Map::new();
            for range in &ranges {
                let axis = range["axis"].as_str().unwrap_or("");
                match role_of(axis, range["domain"].as_str().unwrap_or("")) {
                    Some(role) => { area.insert(role.into(), json!([range["lower"], range["upper"]])); }
                    None => problem = Some(format!("its area selector names axis {axis}, which is neither x nor y")),
                }
            }
            if area.contains_key("x") && area.contains_key("y") {
                selectors.push(json!({"kind": "area", "value": area, "facet_index": facet_index, "revision": 1}));
            } else if problem.is_none() { problem = Some("its area selector does not name both axes".into()); }
        }
        Some("x_range") => match ranges.first() {
            Some(range) => selectors.push(json!({"kind": "x_range", "value": [range["lower"], range["upper"]], "facet_index": facet_index, "revision": 1})),
            None => problem = Some("its x-range selector has no range".into()),
        },
        Some(other) => problem = Some(format!("its {other} selector is not restored")),
        None => {}
    }
    if let (Some(x), Some(y)) = (crosshair["x"].as_f64(), crosshair["y"].as_f64()) {
        selectors.push(json!({"kind": "crosshair", "value": [x, y], "facet_index": Value::Null, "revision": 1}));
    }
    (selectors, problem)
}

impl Workbench {
    /// Parse and resolve a layout document, touching nothing, entry by entry (v2
    /// `resolve_layout`): a panel or logic entry this board cannot carry is left out
    /// with its reason and every other one loads; a saved value that cannot be read,
    /// or that its field refuses, is left at its default. Each is said in `notes`.
    /// Only a document that is not a board is refused.
    fn resolve_layout(&self, document: &Value, host: &mut dyn Host) -> Result<Loaded, String> {
        let document = object(document, "a layout")?;
        if document.get("format").and_then(Value::as_str) != Some(FORMAT) { return Err("that file is not a saved board".into()); }
        let empty = Vec::new();
        let panels = match document.get("panels") { None => &empty, Some(Value::Array(items)) => items, Some(_) => return Err("panels must be a list".into()) };
        let logic = match document.get("logic") { None => &empty, Some(Value::Array(items)) => items, Some(_) => return Err("logic must be a list".into()) };
        let mut notes = Vec::new();
        let mut kept: Vec<(String, &Map<String, Value>)> = Vec::new();
        for (index, entry) in panels.iter().enumerate() {
            match panel_entry(entry, &kept) {
                Ok(panel) => kept.push(panel),
                Err(reason) => notes.push(format!("panel entry {}: {reason}; it is left out", index + 1)),
            }
        }
        let fresh = host.reserve_panel_ids(kept.len());
        let renamed: BTreeMap<String, String> = kept.iter().map(|(saved, _)| saved.clone()).zip(fresh.iter().cloned()).collect();
        let mut nodes: Vec<LoadedNode> = Vec::new();
        let reserved: Vec<String> = host.signals().iter().map(|signal| split_signal(&signal.name)
            .map_or(signal.owner.as_str(), |(owner, _)| owner)).filter(|owner| self.node(owner).is_none()).map(str::to_owned).collect();
        for (index, entry) in logic.iter().enumerate() {
            match self.logic_entry(entry, &nodes, &renamed, &reserved, &mut notes) {
                Ok(node) => nodes.push(node),
                Err(reason) => notes.push(format!("logic entry {}: {reason}; it is left out", index + 1)),
            }
        }
        // Sources are respelled once every fresh id is known.
        for node in &mut nodes {
            let source = std::mem::take(&mut node.draft.source_signal);
            node.draft.source_signal = respell(&source, &renamed, &format!("logic {}", node.name), &mut notes);
        }
        // A source of another contract keeps its row, which cannot start until rewired.
        let contracts: BTreeMap<String, Option<String>> = nodes.iter().flat_map(|node| self.catalog.get(&node.type_id).into_iter()
            .flat_map(|item| item.node().outputs.iter().map(|output| (signal_key(&node.id, &output.name), output.contract.clone())))).collect();
        for node in &nodes {
            let wanted = self.catalog.get(&node.type_id).and_then(|item| item.node().inputs.first()).and_then(|input| input.contract.clone());
            if let (Some(wanted), Some(Some(actual))) = (wanted, contracts.get(&node.draft.source_signal)) {
                if *actual != wanted {
                    notes.push(format!("logic {} reads {}, which publishes {actual}, not {wanted}; the row is kept and cannot start until another source is chosen",
                        node.name, node.draft.source_signal));
                }
            }
        }
        let mut loaded_panels = Vec::new();
        for ((_, entry), fresh) in kept.into_iter().zip(fresh) {
            let title = string(entry.get("title"), "", "panel title")?;
            let signal = string(entry.get("signal"), "", "panel signal")?;
            let who = format!("panel {:?}", if title.is_empty() { &signal } else { &title });
            let signal = respell(&signal, &renamed, &who, &mut notes);
            let kind = entry["kind"].as_str().unwrap().to_owned();
            let mut semantic = entry.get("semantic").and_then(Value::as_object).cloned().unwrap_or_default();
            semantic.remove("kind");
            let cell_kind = string(entry.get("cell_kind"), "", "panel cell_kind")?;
            if !cell_kind.is_empty() { semantic.insert("cell_kind".into(), json!(cell_kind)); }
            let mut parameters = entry.get("display").and_then(Value::as_object).cloned().unwrap_or_default();
            let window = parameters.remove("window").and_then(|window| window.as_u64()).filter(|window| *window > 0)
                .unwrap_or(if kind == "rolling" { 100 } else { 1 });
            let interval = entry.get("interval_ms").and_then(Value::as_u64).unwrap_or(100);
            if !INTERVALS.contains(&interval) { notes.push(format!("{who}: update interval {interval} ms is not one of 100/200/400/800; it is 100")); }
            let selectors = match entry.get("selectors") {
                Some(Value::Array(selectors)) => (selectors.clone(), None),
                _ => selectors_from_v2(entry.get("selector").unwrap_or(&Value::Null), entry.get("crosshair").unwrap_or(&Value::Null), &semantic),
            };
            if let (_, Some(problem)) = &selectors { notes.push(format!("{who}: {problem}")); }
            let mut target = json!({"semantic": semantic, "parameters": parameters, "fit": entry.get("fit").cloned().unwrap_or(json!({})),
                "size": entry.get("size").cloned().unwrap_or(json!("2x2")), "selectors": selectors.0,
                "classifier_thresholds": entry.get("classifier_thresholds").cloned().unwrap_or(json!([])),
                "facet_focus": entry.get("focused_cell").cloned().unwrap_or(Value::Null)});
            if let Some(interaction) = entry.get("interaction").filter(|value| value.is_object()) { target["interaction"] = interaction.clone(); }
            let kept: Map<String, Value> = KEPT.iter().filter_map(|key| entry.get(*key).map(|value| ((*key).to_owned(), value.clone()))).collect();
            let overlay = string(entry.get("overlay_signal"), "", "panel overlay_signal")?;
            let overlay = if overlay.is_empty() { overlay } else { respell(&overlay, &renamed, &format!("{who} overlay"), &mut notes) };
            let automatic = title.is_empty() || title == signal || title == entry["signal"].as_str().unwrap_or("");
            loaded_panels.push(LoadedPanel { panel: fresh.clone(), signal, kind, window, target: target.clone(),
                board: BoardPanel { panel: fresh, title: (!automatic).then_some(title), interval_ms: if INTERVALS.contains(&interval) { interval } else { 100 },
                    overlay_signal: overlay,
                    published_outputs: entry.get("published_outputs").and_then(Value::as_object).cloned().unwrap_or_default(),
                    kept, loaded_selectors: Some(target["selectors"].clone()) } });
        }
        Ok(Loaded { nodes, panels: loaded_panels, notes, path: None, renamed })
    }

    /// One saved logic row as v2 `_logic_from_tree` + `resolve_layout` read it, onto the
    /// rows `loaded` before it; Err says why it is left out.
    fn logic_entry(&self, entry: &Value, loaded: &[LoadedNode], renamed: &BTreeMap<String, String>,
                   reserved: &[String], notes: &mut Vec<String>) -> Result<LoadedNode, String> {
        let entry = object(entry, "it")?;
        let saved_id = string(entry.get("node_id"), "", "its node_id")?.trim().to_owned();
        let type_id = string(entry.get("api_name"), "", "its api_name")?.trim().to_owned();
        if saved_id.is_empty() || type_id.is_empty() { return Err("logic entries require node_id and api_name".into()); }
        if reserved.iter().any(|owner| crate::logic::name_key(owner) == crate::logic::name_key(&saved_id)) {
            return Err(format!("{saved_id} is already a signal producer"));
        }
        if loaded.iter().any(|node| node.id == saved_id || renamed.get(&saved_id) == Some(&node.id)) { return Err(format!("another logic entry is also {saved_id}")); }
        let board = || loaded.iter().map(|node| (node.id.as_str(), node.name.as_str()));
        if let Some(problem) = name_problem(&saved_id, &saved_id, board()) { return Err(problem); }
        let (descriptor, implied) = match self.catalog.get(&type_id).filter(|item| !item.is_device()) {
            Some(item) => (item, None),
            None => self.catalog.nodes().find_map(|item| item.node().replaces.get(&type_id).map(|values| (item, Some(values))))
                .ok_or_else(|| format!("no logic node named {type_id:?}"))?,
        };
        let type_id = descriptor.type_id.clone();
        let mut name = name_text(&string(entry.get("name"), "", "its name")?);
        if name.is_empty() { name = saved_id.clone(); }
        if name != saved_id && reserved.iter().any(|owner| crate::logic::name_key(owner) == crate::logic::name_key(&name)) {
            notes.push(format!("logic {saved_id} is called by its internal ID: {name} is a signal producer"));
            name = saved_id.clone();
        }
        if let Some(clash) = (name != saved_id).then(|| name_problem(&name, &saved_id, board())).flatten() {
            notes.push(format!("logic {saved_id} is called by its internal ID: {clash}"));
            name = saved_id.clone();
        }
        let id = saved_id;
        // A saved part that cannot be read is left at its default, and said.
        let mut part = |key: &str, read: &dyn Fn(&Value) -> bool| match entry.get(key) {
            Some(value) if !read(value) => { notes.push(format!("logic {name}: its {key} {value} cannot be read; it is at its default")); None }
            other => other.cloned(),
        };
        let mut values = part("values", &Value::is_object).and_then(|value| value.as_object().cloned()).unwrap_or_default();
        values.extend(implied.into_iter().flatten().map(|(key, value)| (key.clone(), value.clone())));
        let keys = part("device_keys", &Value::is_object).and_then(|value| value.as_object().cloned()).unwrap_or_default();
        let artifacts = part("artifact_inputs", &Value::is_object).and_then(|value| value.as_object().cloned()).unwrap_or_default();
        let mut source = part("source_signal", &Value::is_string).and_then(|value| value.as_str().map(|text| text.trim().to_owned())).unwrap_or_default();
        let auto_preview = part("auto_preview", &Value::is_boolean).and_then(|value| value.as_bool()).unwrap_or(true);
        let (values, refusals) = fields::restore(&descriptor.parameters, &values).map_err(|error| format!("{name}: {error}"))?;
        notes.extend(refusals.into_iter().map(|refusal| format!("logic {name}: {refusal}; the row is kept with that field at its default")));
        let mut device_keys = BTreeMap::new();
        for requirement in &descriptor.node().devices {
            let available = self.device_options(&requirement.capability);
            let key = match keys.get(&requirement.argument) {
                Some(Value::String(key)) if !key.is_empty() => key.clone(),
                saved => {
                    if saved.is_some_and(|key| !key.is_string()) {
                        notes.push(format!("logic {name}: its device input {:?} must be text; it is the first device that fits", requirement.argument));
                    }
                    available.first().map(|(key, _)| key.clone()).unwrap_or_default()
                }
            };
            if self.devices.contains_key(&key) && !self.capability_of(&key, &requirement.capability) {
                notes.push(format!("logic {name} names {key:?} as its {}, which does not provide {}; the row is kept and cannot start until another device is chosen",
                    requirement.argument, requirement.capability));
            }
            device_keys.insert(requirement.argument.clone(), key);
        }
        let mut artifact_inputs = Map::new();
        for spec in &descriptor.node().artifact_inputs {
            let blank = if spec.frame_claims { json!([]) } else { json!("") };
            let value = artifacts.get(&spec.name).cloned().unwrap_or_else(|| blank.clone());
            let valid = if spec.frame_claims { value.as_array().is_some_and(|claims| claims.iter().all(|claim| claim["path"].is_string() && claim["frames"].is_array())) }
                else { value.is_string() };
            if !valid { notes.push(format!("logic {name}: artifact input {:?} must be {}; it is empty", spec.name, if spec.frame_claims { "a list of {path, frames}" } else { "a path" })); }
            artifact_inputs.insert(spec.name.clone(), if valid { value } else { blank });
        }
        if !source.is_empty() && descriptor.node().inputs.is_empty() {
            notes.push(format!("logic {name}: {type_id} has no dataset input; its source {source} was dropped"));
            source.clear();
        }
        Ok(LoadedNode { id, name, type_id, draft: Draft { values, source_signal: source, device_keys, artifact_inputs }, auto_preview })
    }

    pub(crate) fn read_layout(&self, path: &str, host: &mut dyn Host) -> Result<Value, String> {
        let path = self.workspace.resolve(path);
        let document = workspace::read_strict(&path, &format!("console layout {}", path.display()))?;
        let loaded = self.resolve_layout(&document, host)?;
        Ok(json!({"path": path, "nodes": loaded.nodes.iter().map(|node| json!({"node_id": node.id, "name": node.name, "type_id": node.type_id,
                "values": node.draft.values, "source_signal": node.draft.source_signal, "device_keys": node.draft.device_keys})).collect::<Vec<_>>(),
            "panels": loaded.panels.iter().map(|panel| json!({"panel": panel.panel, "signal": panel.signal, "kind": panel.kind, "window": panel.window,
                "target": panel.target})).collect::<Vec<_>>(),
            "notes": loaded.notes}))
    }

    /// v2 _running_logic: the nodes still at work, started or waiting to start (a follower
    /// waiting for its source is not).
    fn anything_runs(&self) -> Vec<String> {
        self.nodes.iter().filter(|node| self.active_run(node).is_some() || node.pending.is_some() || node.start_after_check.is_some())
            .map(|node| node.name.clone()).collect()
    }

    /// v2 _replace_board: replace the board now, or stop what runs and replace it once the
    /// last of those has stopped, saying so; a newer replacement takes a waiting one's place.
    /// `done` is said when it is replaced; `refusal` when something runs and may not be stopped.
    fn replace_board(&mut self, loaded: Loaded, done: &'static str, stop_running: bool, refusal: &str, host: &mut dyn Host) -> Result<Value, String> {
        let running = self.anything_runs();
        if !running.is_empty() {
            if !stop_running { self.console_report = (refusal.to_owned(), "warning"); return Err(refusal.to_owned()); }
            self.stop_all(host);
        }
        let notes = loaded.notes.clone();
        self.board.pending = Some((loaded, done));
        if self.commit_board(host, false) { return Ok(json!({"replaced": true, "waiting_for": []})); }
        self.console_report = (format!("stopping {}; the board is replaced when they have stopped", running.join(", ")), "task");
        // v2 says what the layout could not keep at once; the replacement then says only `done`.
        if !notes.is_empty() { self.console_report = (notes.join("\n"), "warning"); }
        Ok(json!({"replaced": false, "waiting_for": self.anything_runs()}))
    }

    /// A run ended: the waiting board replaces the current one once nothing runs.
    pub(crate) fn commit_pending_board(&mut self, host: &mut dyn Host) { self.commit_board(host, true); }

    /// Whether the waiting board replaced the current one.
    fn commit_board(&mut self, host: &mut dyn Host, waited: bool) -> bool {
        if self.board.pending.is_none() || !self.anything_runs().is_empty() { return false; }
        let (loaded, done) = self.board.pending.take().unwrap();
        let said = loaded.notes.clone();
        for panel in std::mem::take(&mut self.board.panels) { host.close_panel(&panel.panel); }
        for node in std::mem::take(&mut self.nodes) { self.plugins.close(&node_instance(&node.id)); }
        self.tickets.retain(|_, ticket| !matches!(ticket, crate::Ticket::Query { .. } | crate::Ticket::Selection { .. }));
        // What the old board's regions set and read goes with it.
        self.selection_restore.clear();
        self.routed_selections.clear();
        self.bundles.clear();
        for node in loaded.nodes {
            self.plugins.open(&node_instance(&node.id), &node.type_id, &json!({}), &json!({}));
            let descriptor = self.catalog.get(&node.type_id);
            self.nodes.push(Node::new(node.id, node.name, node.type_id, node.draft, node.auto_preview, descriptor));
        }
        let mut notes = loaded.notes;
        for panel in loaded.panels {
            match host.add_panel(&panel.panel, &panel.signal, &panel.kind, panel.window, panel.target) {
                Ok(id) => { let mut board = panel.board; board.panel = id; self.board.panels.push(board); }
                Err(error) => notes.push(format!("panel {}: {error}", panel.panel)),
            }
        }
        self.notes = notes;
        // Notes said while it waited are not said again (only a panel that failed now is new).
        if waited { self.notes_said = said; }
        self.board.path = loaded.path;
        self.board.dirty = false;
        self.console_report = (done.to_owned(), "task");
        true
    }

    pub(crate) fn load_layout(&mut self, path: &Path, stop_running: bool, host: &mut dyn Host) -> Result<Value, String> {
        let document = workspace::read_strict(path, &format!("console layout {}", path.display()))?;
        let mut loaded = self.resolve_layout(&document, host)?;
        loaded.path = Some(path.to_path_buf());
        let notes = loaded.notes.clone();
        let mut reply = self.replace_board(loaded, "layout loaded", stop_running, "Loading this layout needs stop_running=true while Logic is active", host)?;
        reply["notes"] = json!(notes);
        Ok(reply)
    }

    pub(crate) fn clear_board(&mut self, stop_running: bool, host: &mut dyn Host) -> Result<Value, String> {
        // v2: an empty board is left as it is ("the board is already empty").
        if self.board.panels.is_empty() && self.nodes.is_empty() {
            self.console_report = ("the board is already empty".into(), "idle");
            return Ok(json!({"replaced": false, "waiting_for": []}));
        }
        self.replace_board(Loaded { nodes: Vec::new(), panels: Vec::new(), notes: Vec::new(), path: None, renamed: BTreeMap::new() }, "board cleared",
            stop_running, "Clearing the board needs stop_running=true while Logic is active", host)
    }

    /// The current board as `zlc.console-board`: stopped drafts and every panel, in board order.
    pub(crate) fn layout_document(&self, host: &dyn Host) -> Value {
        let contents = host.panels();
        let panels: Vec<Value> = self.board.panels.iter().filter_map(|entry| contents.iter().find(|content| content.panel == entry.panel).map(|content| (entry, content)))
            .map(|(entry, content)| {
                let target = &content.target;
                let mut semantic = target["semantic"].as_object().cloned().unwrap_or_default();
                semantic.remove("kind");
                let cell_kind = semantic.remove("cell_kind").and_then(|value| value.as_str().map(str::to_owned)).unwrap_or_default();
                let mut display = target["parameters"].as_object().cloned().unwrap_or_default();
                if content.kind != "image" { display.insert("window".into(), json!(content.window)); }
                let selectors = target["selectors"].clone();
                let unchanged = entry.loaded_selectors.as_ref() == Some(&selectors);
                let mut row = json!({"panel_id": content.panel, "signal": content.signal, "title": entry.title.clone().unwrap_or_else(|| content.signal.clone()),
                    "kind": content.kind, "cell_kind": cell_kind, "size": target.get("size").cloned().unwrap_or(json!("2x2")), "interval_ms": entry.interval_ms,
                    "semantic": semantic, "display": display, "interaction": target.get("interaction").cloned().unwrap_or(json!({"series_lock": null})),
                    "fit": target.get("fit").cloned().unwrap_or(json!({})),
                    "overlay_signal": entry.overlay_signal,
                    "published_outputs": entry.published_outputs,
                    "selector": if unchanged { entry.kept.get("selector").cloned().unwrap_or(json!({})) } else { json!({}) },
                    "crosshair": if unchanged { entry.kept.get("crosshair").cloned().unwrap_or(json!({})) } else { json!({}) },
                    "classifier_thresholds": target.get("classifier_thresholds").cloned().unwrap_or(json!([])),
                    "focused_cell": target.get("facet_focus").cloned().unwrap_or(Value::Null)});
                // v3 selectors travel beside v2's grammar; a v2 reader ignores the key.
                if selectors.as_array().is_some_and(|items| !items.is_empty()) { row["selectors"] = selectors; }
                row
            }).collect();
        let logic: Vec<Value> = self.nodes.iter().map(|node| json!({"node_id": node.id, "name": node.name, "api_name": node.type_id,
            "values": node.draft.values, "source_signal": node.draft.source_signal, "device_keys": node.draft.device_keys,
            "artifact_inputs": node.draft.artifact_inputs, "auto_preview": node.auto_preview})).collect();
        json!({"format": FORMAT, "panels": panels, "logic": logic})
    }

    pub(crate) fn save_layout(&mut self, path: &Path, host: &mut dyn Host) -> Result<Value, String> {
        let document = self.layout_document(host);
        let written = workspace::write_readable(path, &document)?;
        self.board.path = Some(written.clone());
        self.board.dirty = false;
        Ok(json!({"path": written}))
    }

    pub(crate) fn order_panels(&mut self, order: &[Value], _host: &mut dyn Host) -> Result<Value, String> {
        let wanted: Vec<String> = order.iter().filter_map(Value::as_str).filter(|id| self.board.get(id).is_some()).map(str::to_owned).collect();
        let mut panels = std::mem::take(&mut self.board.panels);
        let mut ordered = Vec::with_capacity(panels.len());
        for id in &wanted { if let Some(index) = panels.iter().position(|entry| &entry.panel == id) { ordered.push(panels.remove(index)); } }
        ordered.extend(panels);
        self.board.panels = ordered;
        self.board.dirty = true;
        Ok(json!({"order": self.board.panels.iter().map(|entry| &entry.panel).collect::<Vec<_>>()}))
    }

    pub(crate) fn rename_panel(&mut self, panel: &str, title: Option<&str>) -> Result<Value, String> {
        let title = title.map(|title| title.split_whitespace().collect::<Vec<_>>().join(" ")).filter(|title| !title.is_empty());
        self.board.get_mut(panel)?.title = title.clone();
        self.board.dirty = true;
        Ok(json!({"panel": panel, "title": title}))
    }

    pub(crate) fn set_panel_interval(&mut self, panel: &str, interval: u64) -> Result<Value, String> {
        if !INTERVALS.contains(&interval) { return Err("The update interval is 100, 200, 400 or 800 ms".into()); }
        self.board.get_mut(panel)?.interval_ms = interval;
        self.board.dirty = true;
        Ok(json!({"panel": panel, "interval_ms": interval}))
    }

    /// v2 Overlay: one of the panel's overlay choices, or "" for none.
    pub(crate) fn set_panel_overlay(&mut self, panel: &str, signal: &str, host: &mut dyn Host) -> Result<Value, String> {
        let signal = signal.trim();
        let contents = host.panels();
        let content = contents.iter().find(|content| content.panel == panel).ok_or_else(|| format!("No panel {panel:?} on the board"))?;
        // A Task-protected panel keeps its source, the Overlay with it (v2 science_locked).
        if let Some(reason) = self.panel_frozen(panel, &content.signal) { return Err(reason); }
        if !signal.is_empty() && !self.overlay_choices(&content.signal, &contents).iter().any(|(choice, _)| choice == signal) {
            return Err(format!("{} judges no sites of this panel's image", self.signal_label(signal)));
        }
        self.board.get_mut(panel)?.overlay_signal = signal.to_owned();
        self.board.dirty = true;
        Ok(json!({"panel": panel, "overlay_signal": signal}))
    }

    /// v2 `overlay_signal_groups`: the site-status outputs (the overlay contract)
    /// of the nodes that read the same measurement as this image.
    fn overlay_choices(&self, signal: &str, contents: &[crate::PanelContent]) -> Vec<(String, String)> {
        let Some(root) = self.signal_root(signal, contents) else { return Vec::new() };
        self.nodes.iter().flat_map(|node| self.node_outputs(node))
            .filter(|(output, contract)| contract.as_deref() == Some(OVERLAY_CONTRACT) && self.signal_root(output, contents).as_deref() == Some(root.as_str()))
            .map(|(output, _)| { let label = self.signal_label(&output); (output, label) }).collect()
    }

    /// The measurement node a signal comes from, through node sources and panel outputs.
    fn signal_root(&self, signal: &str, contents: &[crate::PanelContent]) -> Option<String> {
        let mut signal = signal.to_owned();
        for _ in 0..32 {
            let producer = split_signal(&signal)?.0.split('/').next()?.to_owned();
            if let Some(node) = self.node(&producer) {
                if node.draft.source_signal.is_empty() { return Some(producer); }
                signal = node.draft.source_signal.clone();
            } else {
                signal = contents.iter().find(|content| content.panel == producer)?.signal.clone();
            }
        }
        None
    }

    pub(crate) fn publish_panel_outputs(&mut self, panel: &str, outputs: &Value) -> Result<Value, String> {
        let outputs = outputs.as_object().ok_or("outputs is an object of output name -> true/false")?;
        if outputs.values().any(|value| !value.is_boolean()) { return Err("outputs is an object of output name -> true/false".into()); }
        if let Some(task) = self.active_task() {
            let protected = self.task_protected_signals(task);
            if protected.iter().any(|signal| split_signal(signal).is_some_and(|(producer, _)| producer == panel)) {
                return Err(format!("{} is running; its preview's outputs are fixed until it stops", task.name));
            }
        }
        let entry = self.board.get_mut(panel)?;
        entry.published_outputs.extend(outputs.clone());
        let published = entry.published_outputs.clone();
        self.board.dirty = true;
        Ok(json!({"panel": panel, "published_outputs": published}))
    }

    /// A panel's ROI/fit outputs (v2 `_panel_publisher_fields`): what its surface and fit can
    /// publish, each with the board's switch (absent = on) and, while it is published, its signal's shape.
    fn panel_outputs(&self, entry: &BoardPanel, content: Option<&crate::PanelContent>, signals: &[crate::SignalInfo]) -> Vec<Value> {
        content.map(|content| content.outputs.as_slice()).unwrap_or_default().iter().map(|(name, label)| {
            let signal = signal_key(&entry.panel, name);
            let live = signals.iter().find(|info| info.name == signal);
            json!({"name": name, "label": label, "signal": signal, "shape": live.map(|info| info.shape.clone()), "live": live.is_some(),
                "published": entry.published_outputs.get(name) != Some(&Value::Bool(false))})
        }).collect()
    }

    pub(crate) fn board_state(&self, contents: &[crate::PanelContent], signals: &[crate::SignalInfo]) -> Value {
        let frozen: Map<String, Value> = contents.iter()
            .filter_map(|content| self.panel_frozen(&content.panel, &content.signal).map(|reason| (content.panel.clone(), json!(reason)))).collect();
        json!({"panels": self.board.panels.iter().map(|entry| {
                let content = contents.iter().find(|content| content.panel == entry.panel);
                let overlays = content.map(|content| self.overlay_choices(&content.signal, contents)).unwrap_or_default();
                json!({"panel": entry.panel, "title": entry.title,
                "name": entry.title.clone().or_else(|| content.map(|content| self.signal_label(&content.signal))),
                "interval_ms": entry.interval_ms, "overlay_signal": entry.overlay_signal,
                // Setting's Signal and Overlay pickers (v2 keyed_choice): the trees they open.
                "signal_tree": self.signal_tree(&entry.panel, content.map_or("", |content| content.signal.as_str()), signals, contents),
                "overlay_tree": self.overlay_tree(&overlays, &entry.overlay_signal, signals),
                "overlay_choices": overlays.into_iter().map(|(signal, label)| json!([signal, label])).collect::<Vec<_>>(),
                "published_outputs": entry.published_outputs,
                "outputs": self.panel_outputs(entry, content, signals)})}).collect::<Vec<_>>(),
            "paused": self.board.paused, "selectors": self.board.selectors,
            "report": {"text": self.console_report.0, "severity": self.console_report.1},
            "active_task": self.active_task().map(|task| task.id.clone()), "frozen": frozen,
            "pending_load": self.board.pending.as_ref().map(|(loaded, _)| loaded.path.clone())})
    }

    /// v2 experiment_recipes: the recipes shipped with the workbench (`recipes/` beside
    /// this crate, v2's zlc_workbench/recipes), a workspace recipe of the same name
    /// replacing one.
    fn recipe_paths(&self) -> BTreeMap<String, PathBuf> {
        let found = |folder: PathBuf| std::fs::read_dir(folder).into_iter().flatten().flatten().map(|entry| entry.path())
            .filter_map(|path| path.file_name().and_then(|name| name.to_str()).and_then(|name| name.strip_suffix(".recipe.json")).map(|name| (name.to_owned(), path.clone())))
            .collect::<Vec<_>>();
        let mut paths: BTreeMap<String, PathBuf> = found(zlc_core::checkout::path("crates/zlc-workbench/recipes")).into_iter().collect();
        paths.extend(found(self.workspace.recipes()));
        paths
    }

    fn read_recipe(&self, name: &str, parameters: Option<&Map<String, Value>>) -> Result<(Value, Value, Map<String, Value>), String> {
        let paths = self.recipe_paths();
        let path = paths.get(name).ok_or_else(|| format!("Unknown recipe {name:?}; available: {}", paths.keys().cloned().collect::<Vec<_>>().join(", ")))?;
        let recipe = workspace::read_strict(path, &format!("recipe {name}"))?;
        let required = ["completion", "description", "entry_node", "followers", "format", "layout", "name", "parameters", "requirements", "result_panels", "save", "title"];
        let mut keys: Vec<&str> = recipe.as_object().map(|object| object.keys().map(String::as_str).collect()).unwrap_or_default();
        keys.sort_unstable();
        if keys != required || recipe["format"] != "zlc.experiment-recipe" || recipe["name"] != name {
            return Err("Recipe must use the current zlc.experiment-recipe grammar and matching name".into());
        }
        let folder = path.parent().unwrap();
        let layout_path = folder.join(recipe["layout"].as_str().unwrap_or(""));
        if !layout_path.canonicalize().is_ok_and(|layout| folder.canonicalize().is_ok_and(|folder| layout.starts_with(folder))) {
            return Err("Recipe Layout must be beside or beneath its recipe file".into());
        }
        let mut layout = workspace::read_strict(&layout_path, &format!("recipe layout {}", layout_path.display()))?;
        let supplied = parameters.cloned().unwrap_or_default();
        let declared = recipe["parameters"].as_object().cloned().unwrap_or_default();
        if let Some(unknown) = supplied.keys().find(|key| !declared.contains_key(*key)) { return Err(format!("Unknown recipe parameter {unknown:?}")); }
        let mut defaults = Map::new();
        let mut missing = Vec::new();
        for (key, binding) in &declared {
            let parts = binding["path"].as_array().cloned().unwrap_or_default();
            if parts.len() < 3 || !matches!(parts[0].as_str(), Some("logic" | "panels")) { return Err(format!("Recipe parameter {key:?} needs a native Layout field path")); }
            let mut target = &mut layout;
            for part in &parts[..parts.len() - 1] {
                target = match part { Value::String(name) => &mut target[name.as_str()], Value::Number(index) => &mut target[index.as_u64().unwrap_or(0) as usize], _ => return Err(format!("Recipe parameter {key:?} path is malformed")) };
            }
            let last = &parts[parts.len() - 1];
            let slot = match last { Value::String(name) => &mut target[name.as_str()], Value::Number(index) => &mut target[index.as_u64().unwrap_or(0) as usize], _ => return Err(format!("Recipe parameter {key:?} path is malformed")) };
            defaults.insert(key.clone(), slot.clone());
            if let Some(value) = supplied.get(key) { *slot = value.clone(); }
            else if parameters.is_some() && binding["required"] == true { missing.push(key.clone()); }
        }
        if !missing.is_empty() { return Err(format!("Recipe requires explicit parameters: {}", missing.join(", "))); }
        Ok((recipe, layout, defaults))
    }

    pub(crate) fn list_recipes(&self, name: Option<&str>) -> Result<Value, String> {
        match name {
            Some(name) => {
                let (recipe, layout, defaults) = self.read_recipe(name, None)?;
                Ok(json!({"recipe": recipe, "defaults": defaults, "layout": layout}))
            }
            // Recipe by recipe: one that cannot be read is listed with why, beside the others.
            None => Ok(Value::Array(self.recipe_paths().into_iter().map(|(name, _)| match self.read_recipe(&name, None) {
                Ok((recipe, _, _)) => json!({"name": recipe["name"], "title": recipe["title"], "description": recipe["description"], "requirements": recipe["requirements"]}),
                Err(problem) => json!({"name": name, "problem": problem}),
            }).collect())),
        }
    }

    /// Load a recipe's layout with its parameters; start followers then the
    /// entry node once their plugins have answered.
    pub(crate) fn run_recipe(&mut self, name: &str, parameters: &Value, start: bool, origin: &str, host: &mut dyn Host) -> Result<Value, String> {
        let (recipe, layout, _) = self.read_recipe(name, Some(&parameters.as_object().cloned().unwrap_or_default()))?;
        let followers: Vec<&str> = recipe["followers"].as_array().into_iter().flatten().filter_map(Value::as_str).collect();
        let entry = recipe["entry_node"].as_str().unwrap_or("");
        // v2 lab_run_recipe: each follower once, apart from the entry node, and a standing processor.
        let unique: BTreeSet<&str> = followers.iter().copied().collect();
        if unique.len() != followers.len() || unique.contains(entry) { return Err("A recipe starts each follower once, separately from its entry node".into()); }
        let declared = |id: &str| layout["logic"].as_array().into_iter().flatten().find(|item| item["node_id"] == id).and_then(|item| item["api_name"].as_str())
            .and_then(|type_id| self.catalog.get(type_id)).map(|item| item.node().node_kind.clone());
        if followers.iter().any(|id| declared(id).as_deref() != Some("processor")) {
            return Err("Recipe followers must be standing processors, not device executions".into());
        }
        let mut loaded = self.resolve_layout(&layout, host)?;
        let renamed = loaded.renamed.clone();
        loaded.path = None;
        let notes = loaded.notes.clone();
        let reply = self.replace_board(loaded, "layout loaded", false, "stop running logic before loading a board", host)?;
        let order: Vec<String> = followers.iter().copied().chain([entry]).map(str::to_owned).collect();
        if start { self.board.recipe_start = Some((order.clone(), origin.to_owned())); }
        self.poll_recipe(host);
        Ok(json!({"recipe": name, "replaced": reply["replaced"], "nodes": order, "notes": notes,
            "result_panels": recipe["result_panels"].as_array().into_iter().flatten().filter_map(Value::as_str).filter_map(|saved| renamed.get(saved)).collect::<Vec<_>>()}))
    }

    /// Start a loaded recipe's nodes once each has opened and answered its draft queries,
    /// and only when every one of them may start (v2: validated first, then started in order).
    pub(crate) fn poll_recipe(&mut self, host: &mut dyn Host) {
        let Some((order, origin)) = self.board.recipe_start.clone() else { return };
        let signals = host.signals();
        let settled = order.iter().all(|id| self.node(id).is_some_and(|node| node.instance != Instance::Opening
            && !self.issues(node, &signals).iter().any(|issue| issue == crate::logic::CHECKING || issue.ends_with(crate::logic::STARTING_PLUGIN))));
        if !settled { return; }
        self.board.recipe_start = None;
        let invalid: Vec<String> = order.iter().filter_map(|id| self.node(id)).filter_map(|node| self.issues(node, &signals).first()
            .map(|issue| format!("{}: {issue}", node.name))).collect();
        if !invalid.is_empty() {
            self.notes.push(format!("recipe: inputs are not valid, nothing started -- {}", invalid.join("; ")));
            return;
        }
        for id in order {
            if let Err(error) = self.start_node(&id, None, &origin, None, host) { self.notes.push(format!("recipe: {id}: {error}")); break; }
        }
    }
}
