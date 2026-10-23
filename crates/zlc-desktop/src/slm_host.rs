//! The SLM Editor's pictures (v2's "<role> SLM Editor" plots): an open
//! editor's target, science phase and wavefront, each an Image panel
//! `slm-<device>-<plot>` of a Figure document of its own, drawn on one lane
//! off the window's turn from the planes the SLM plugin tells
//! (`device.notice` slm_editor: its state and the planes that changed). The
//! window `slm-editor:<device>` shows them; its gestures (zoom, pan, the
//! selectors) are the panels', and a paint stroke on the target goes to the
//! plugin as canonical points of the drawn image (v2 canonical_from_normalized).
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::{Arc, Mutex};

use serde_json::{Value, json};
use zlc_plot::document::{DocumentRead, FigureDocument, axis_from_wire, error_text};
use zlc_plot::projection::{Source, SourceData};
use zlc_plugin_host::Lent;

use crate::pulse_host::{Lane, services};
use crate::science::Front;

pub const PANEL_PREFIX: &str = "slm-";

pub fn panel_of(device: &str, plot: &str) -> String { format!("{PANEL_PREFIX}{device}-{plot}") }
/// The device an `slm-<device>-<plot>` panel is of (plot names carry no `-`; a device id may).
pub fn device_of(panel: &str) -> Option<&str> { panel.strip_prefix(PANEL_PREFIX)?.rsplit_once('-').map(|(device, _)| device) }

/// A plane the plugin lent, as a panel reads it.
#[derive(Clone)]
pub struct EditorRead { source: Arc<Source> }

impl DocumentRead for EditorRead {
    fn source(&self) -> &Arc<Source> { &self.source }
    fn provenance_json(&self) -> String { "{}".into() }
}

/// What the lane answers on the window's turn.
pub struct Drawn { device: String, request: u64, result: Result<(Value, BTreeMap<String, Front>), String> }

pub type Post = Arc<dyn Fn(Drawn) + Send + Sync>;

/// One job of an editor's document: its pictures again, one panel's command (a pointer, a resize), its ratio.
enum Work { Sync, Panel(Value), Ready(f64) }

struct Editor {
    state: Value,
    catalog: Value,
    reads: BTreeMap<String, EditorRead>,
    /// Each plot's words: its title and axis labels.
    words: BTreeMap<String, (String, String, String)>,
    document: Arc<Mutex<FigureDocument<EditorRead>>>,
    /// The panels as last drawn (targets, axes, frames).
    panels: Vec<Value>,
    size: String,
    fronts: BTreeMap<String, (u64, Front)>,
    queue: VecDeque<Work>,
    busy: bool,
    sequence: u64,
}

/// `shown`: the device ratio of each editor window that is up; an editor is drawn only while its window is
/// (a draft an MCP client edits with no window is drawn when one opens, at its ratio).
pub struct SlmHost { editors: BTreeMap<String, Editor>, lane: Lane, post: Post, shown: BTreeMap<String, f64> }

impl SlmHost {
    pub fn new(post: Post) -> Self { Self { editors: BTreeMap::new(), lane: Lane::new("slm-editor"), post, shown: BTreeMap::new() } }

    /// Every panel an open editor draws (their mappings are kept).
    pub fn panels(&self) -> Vec<String> {
        self.editors.iter().flat_map(|(device, editor)| editor.reads.keys().map(move |plot| panel_of(device, plot))).collect()
    }

    /// The plugin's notice: the editor's state, and the planes that changed.
    /// Answers the message for its window (the panels come with their pictures).
    pub fn notice(&mut self, device: &str, value: &Value, lent: &Lent) -> Result<Value, String> {
        let dpr = self.shown.get(device).copied();
        let shown = dpr.is_some();
        // A new editor's document draws at its window's ratio from the first picture on.
        let editor = self.editors.entry(device.to_owned()).or_insert_with(|| Editor {
            state: Value::Null, catalog: Value::Null, reads: BTreeMap::new(), words: BTreeMap::new(),
            document: Arc::new(Mutex::new(FigureDocument::new(services(), None))), panels: Vec::new(),
            size: String::new(), fronts: BTreeMap::new(), queue: dpr.map(Work::Ready).into_iter().collect(), busy: false, sequence: 0,
        });
        if value["catalog"].is_object() { editor.catalog = value["catalog"].clone(); }
        editor.state = value["state"].clone();
        // Its window is told the state (the page gets only what changed, delta.rs); the pictures follow when drawn.
        let message = json!({"type": "state", "slm": {"device": device, "state": editor.state, "catalog": editor.catalog}});
        let mut changed = false;
        for (plot, given) in value["plots"].as_object().into_iter().flatten() {
            let dataset = lent.dataset(&given["dataset"])?;
            let source = Source { repeat_counts: None, schema: dataset.block.schema().clone(), data: SourceData::Dense(dataset.block),
                                  block_id: dataset.block_id.into(), generation: dataset.generation.into(), revision: dataset.revision, window: None };
            editor.reads.insert(plot.clone(), EditorRead { source: Arc::new(source) });
            let text = |name: &str| given[name].as_str().unwrap_or("").to_owned();
            editor.words.insert(plot.clone(), (text("title"), text("x"), text("y")));
            changed = true;
        }
        let size = editor.state["plot_size"].as_str().unwrap_or("").to_owned();
        // Every plot at the editor's size (v2 set_size on the three hosts).
        if !size.is_empty() && size != editor.size {
            for row in &editor.panels { editor.queue.push_back(Work::Panel(json!({"id": 0, "action": "configure", "panel": row["id"], "size": size}))); }
            editor.size = size;
        }
        if changed && shown && !editor.queue.iter().any(|work| matches!(work, Work::Sync)) { editor.queue.push_back(Work::Sync); }
        self.next(device);
        Ok(message)
    }

    /// Everything its window shows (a page that is up): the state, the catalogue, the panels.
    pub fn message(&self, device: &str) -> Value {
        match self.editors.get(device) {
            Some(editor) => json!({"type": "state", "slm": {"device": device, "state": editor.state, "catalog": editor.catalog}, "panels": editor.panels}),
            None => json!({"type": "state", "slm": {"device": device, "state": {"open": false}, "catalog": null}, "panels": []}),
        }
    }

    /// The fronts its window shows, to give a page that is (again) up.
    pub fn fronts(&self, device: &str) -> Vec<(String, u64, Front)> {
        self.editors.get(device).map(|editor| editor.fronts.iter().map(|(panel, (sequence, front))| (panel.clone(), *sequence, front.clone())).collect())
            .unwrap_or_default()
    }

    /// The window is up at `dpr`: its panels are drawn at that ratio.
    pub fn ready(&mut self, device: &str, dpr: f64) {
        self.shown.insert(device.to_owned(), dpr);
        if let Some(editor) = self.editors.get_mut(device) { editor.queue.extend([Work::Ready(dpr), Work::Sync]); }
        self.next(device);
    }

    /// The editor closed (its window, its device): its document goes.
    pub fn forget(&mut self, device: &str) { self.editors.remove(device); self.shown.remove(device); }

    /// A pointer on one of its panels: zoom, pan and the selectors, in order (a newer move replaces an unsent one).
    pub fn pointer(&mut self, command: Value) {
        let Some(device) = command["panel"].as_str().and_then(device_of).map(str::to_owned) else { return };
        let Some(editor) = self.editors.get_mut(&device) else { return };
        if command["event"] == "move" {
            if let Some(Work::Panel(last)) = editor.queue.back() { if last["event"] == "move" && last["panel"] == command["panel"] { editor.queue.pop_back(); } }
        }
        editor.queue.push_back(Work::Panel(command));
        self.next(&device);
    }

    /// `slm_paint {device, mode, stroke, intensity, points: [[fx, fy], ...]}` from the window
    /// (canvas fractions of the target): the device's `editor_paint` command at the
    /// canonical points of the image as drawn.
    pub fn paint(&self, command: &Value) -> Result<Value, String> {
        let device = command["device"].as_str().ok_or("Name the SLM")?;
        let panel = panel_of(device, "target");
        let editor = self.editors.get(device).ok_or_else(|| format!("{device} has no open SLM Editor"))?;
        let axis = editor.panels.iter().find(|row| row["id"] == panel.as_str())
            .and_then(|row| row["axes"].as_array()?.iter().find(|axis| axis["role"] == "image").cloned())
            .ok_or("The target is not drawn yet")?;
        let axis = axis_from_wire(&axis).map_err(|error| error_text(&error))?;
        let mut points = Vec::new();
        for point in command["points"].as_array().into_iter().flatten() {
            let (Some(x), Some(y)) = (point[0].as_f64(), point[1].as_f64()) else { continue };
            if let Ok(at) = axis.canonical_from_normalized(x, y) { points.push(json!({"x": at.x, "y": at.y})); }
        }
        Ok(json!({"id": command["id"], "action": format!("{device}__editor_paint"), "mode": command["mode"], "stroke": command["stroke"],
                  "intensity": command["intensity"], "points": points}))
    }

    /// What the window's file action asks (v2 _choose): save or open, the offered name, the filter.
    pub fn file(&self, device: &str, action: &str) -> Option<Value> {
        self.editors.get(device).map(|editor| editor.catalog["files"][action].clone()).filter(Value::is_object)
    }

    fn next(&mut self, device: &str) {
        let Some(editor) = self.editors.get_mut(device) else { return };
        if editor.busy { return; }
        let Some(work) = editor.queue.pop_front() else { return };
        editor.busy = true;
        // Every drawn panel's binding and target, as the document keeps them.
        let mut targets = serde_json::Map::new();
        let mut inputs = Vec::new();
        for (plot, (title, x, y)) in &editor.words {
            let panel = panel_of(device, plot);
            let mut target = editor.panels.iter().find(|row| row["id"] == panel.as_str()).map(|row| row["target"].clone())
                .filter(Value::is_object).unwrap_or_else(|| json!({"semantic": {"kind": "image"},
                    "parameters": {"title": title, "x_label": x, "y_label": y, "side_distribution": false}}));
            target["size"] = json!(editor.size);
            targets.insert(panel.clone(), target);
            inputs.push(json!({"panel": panel, "kind": "image", "window": 1, "signal": format!("@slm/{device}/{plot}"), "epoch": 1,
                               "reason": null, "waiting": false}));
        }
        let reads: BTreeMap<String, EditorRead> = editor.reads.iter().map(|(plot, read)| (panel_of(device, plot), read.clone())).collect();
        let command = match work {
            Work::Sync => json!({"id": 0, "action": "sync_panels"}),
            Work::Ready(dpr) => json!({"id": 0, "action": "ready", "dpr": dpr}),
            Work::Panel(pointer) => pointer,
        };
        let request = command["id"].as_u64().unwrap_or(0);
        let panel = command["panel"].as_str().map(str::to_owned);
        let mut command = command;
        command["targets"] = Value::Object(targets);
        command["inputs"] = Value::Array(match &panel { Some(panel) => inputs.into_iter().filter(|input| input["panel"] == panel.as_str()).collect(), None => inputs });
        let reads = match &panel { Some(panel) => reads.into_iter().filter(|(id, _)| id == panel).collect(), None => reads };
        let (document, post, device) = (editor.document.clone(), self.post.clone(), device.to_owned());
        self.lane.run(move || {
            let _span = crate::trace::span("slm.document", || json!({"device": device}));
            let result = (|| -> Result<(Value, BTreeMap<String, Front>), String> {
                let mut document = document.lock().map_err(|_| "the SLM Editor's document failed")?;
                let execution = document.execute(&command.to_string(), reads, None, None).map_err(|error| error_text(&error))?;
                // The editor's pictures have no cohort and no other window: what is drawn is shown.
                let targets: BTreeSet<String> = execution.state["panels"].as_array().into_iter().flatten()
                    .filter_map(|row| row["id"].as_str().map(str::to_owned)).collect();
                let fronts: BTreeSet<String> = execution.fronts.keys().cloned().collect();
                document.finish_presentation(&targets, &fronts);
                Ok((execution.state, execution.fronts))
            })();
            post(Drawn { device, request, result });
        });
    }

    /// The lane answered: the panels' rows for the window, and the fronts to give it.
    pub fn drawn(&mut self, drawn: Drawn) -> Option<(String, Value, Vec<(String, u64, Front)>)> {
        let Drawn { device, request, result } = drawn;
        crate::trace::point("slm.drawn", || json!({"device": device}));
        let editor = self.editors.get_mut(&device)?;
        editor.busy = false;
        let (message, fronts) = match result {
            Ok((state, fronts)) => {
                editor.sequence += 1;
                let sequence = editor.sequence;
                let mut rows: Vec<Value> = state["panels"].as_array().cloned().unwrap_or_default();
                for row in &mut rows {
                    let id = row["id"].as_str().unwrap_or("").to_owned();
                    row["frame"] = match fronts.get(&id) {
                        Some(front) => json!({"width": front.width, "height": front.height, "sequence": sequence}),
                        None => editor.panels.iter().find(|old| old["id"] == id.as_str()).map(|old| old["frame"].clone()).unwrap_or(Value::Null),
                    };
                }
                editor.panels = rows;
                let given: Vec<(String, u64, Front)> = fronts.into_iter().map(|(panel, front)| (panel, sequence, front)).collect();
                for (panel, sequence, front) in &given { editor.fronts.insert(panel.clone(), (*sequence, front.clone())); }
                let mut message = json!({"type": "state", "panels": editor.panels, "plot_kinds": state["plot_kinds"]});
                if request != 0 { message["request"] = json!(request); message["error"] = Value::Null; }
                (message, given)
            }
            Err(error) => {
                eprintln!("{}", json!({"type": "slm-editor-error", "device": device, "error": error}));
                (json!({"type": "state", "request": request, "error": error}), Vec::new())
            }
        };
        self.next(&device);
        Some((device, message, fronts))
    }
}
