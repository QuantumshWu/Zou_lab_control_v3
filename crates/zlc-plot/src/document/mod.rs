//! The Figure document (`zlc_science.desktop.FigureDocument`): every panel's
//! configuration, session and status, the commands a frontend or an AI
//! client sends, the derivative effects (ROI plans, fit outputs) a host
//! applies, and the transactional presentation the host accepts per
//! same-shot cohort.
//!
//! [`FigureDocument::execute`] answers one command with the state, the
//! fronts, the effects and the presented/admitted reads; every execute
//! holds the previous scene of every panel until
//! [`FigureDocument::finish_presentation`] keeps the accepted panels and
//! restores the rest.  A panel's session sits behind a mutex so panels
//! prepare and present on the shared pool; a transaction's snapshot clones
//! the maps of handles, never data.

mod edit;
mod effects;
mod panel;
mod pipeline;

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

use rayon::prelude::*;

use serde_json::{json, Map, Value as Json};

use crate::configuration::controls::{describe_panel, plot_kinds};
use crate::configuration::data_contract::{schema_structure, PlotSchema};
use crate::configuration::error::{PlotError, Result};
use crate::configuration::ordered::OrderedMap;
use crate::configuration::resolve::{InteractionUpdates, ResolvedConfiguration};
use crate::configuration::value::{Value, ValueMap};
use crate::interaction::transform::AxisTransform;
use crate::projection::Source;
use crate::render::scale::AxisScale;
use crate::session::{ImagePointOverlay, PlotSession, PointerAction, PointerEvent, PreparedPlot, RgbaFront, SessionServices};

use edit::{Edit, Queue};
use panel::{Panel, Shared};

pub use effects::{fit_parameter_data, selection_document, selector_json, subject_json, Effect, FigureSaver, FigureWrite, SaveRequest};
pub(crate) use effects::scalar_domain;

/// One acquired read a panel is given.
pub trait DocumentRead: Clone + Send + Sync + 'static {
    /// The read's immutable scientific data.
    fn source(&self) -> &Arc<Source>;
    /// The causal record of this read, for a saved figure.
    fn provenance_json(&self) -> String;
    /// The point layer this read's image carries (`ImageFrame.overlay`):
    /// `None` says nothing, `Some(None)` is an image without one.
    fn image_overlay(&self) -> Option<Option<Arc<ImagePointOverlay>>> { None }
}

/// The host's fit publication: a live fit is published while its panel is
/// prepared, so a follower bound to `@logic/<panel>/<parameter>` reads it in the same
/// command. A failure is the panel's error.
pub trait FitPublisher<R>: Sync {
    fn publish(&self, panel: &str, read: &R, values: BTreeMap<String, zlc_core::store::Snapshot>, records: bool, metadata: Json)
               -> std::result::Result<(), String>;
    fn read(&self, signal: &str, window: usize, records: bool) -> std::result::Result<Option<R>, String>;
}

/// What one command answers.
pub struct Execution<R> {
    pub state: Json,
    pub fronts: BTreeMap<String, RgbaFront>,
    pub effects: Vec<Effect<R>>,
    /// The read each panel's accepted scene shows.
    pub presented: BTreeMap<String, R>,
    /// The reads this command's new scenes were prepared from.
    pub admitted: BTreeMap<String, R>,
}

pub type SessionHandle = Arc<Mutex<PlotSession>>;

pub(crate) fn lock(handle: &SessionHandle) -> MutexGuard<'_, PlotSession> { handle.lock().unwrap_or_else(|error| error.into_inner()) }

/// One panel binding (`inputs[i]`).
#[derive(Clone, Debug)]
pub(crate) struct Binding { pub raw: Json, pub panel: String, pub signal: String, pub window: i64, pub epoch: u64 }

impl Binding {
    pub fn kind(&self) -> Option<&str> { self.raw.get("kind").and_then(Json::as_str) }
    pub fn reason(&self) -> Option<&str> { self.raw.get("reason").and_then(Json::as_str).filter(|reason| !reason.is_empty()) }
    pub fn waiting(&self) -> bool { self.raw.get("waiting") == Some(&Json::Bool(true)) }
}

/// A panel's status (`panel_status[key]`): absent keys read as `None`.
#[derive(Clone, Default)]
pub(crate) struct PanelStatus {
    pub binding: Option<Binding>,
    pub target_request: Option<Json>,
    pub authoring_pending: Option<bool>,
    pub error: Option<String>,
    pub message: Option<String>,
    pub derivation_error: Option<String>,
    /// Queued edits the arriving data refused (and a command's own edit
    /// refused after them).
    pub edit_error: Option<String>,
    /// The edits made while the panel had no read.
    pub queue: Option<Queue>,
}

/// What a panel's front shows (`panel_sources[key]`).
#[derive(Clone, Debug)]
pub(crate) struct PanelSource { pub signal: String, pub window: i64, pub epoch: u64 }

/// A panel's resolved configuration with the binding it was resolved for.
#[derive(Clone)]
pub(crate) struct Configured { pub signal: String, pub window: i64, pub configuration: Arc<ResolvedConfiguration> }

/// What a staged panel presents: a display-only change of the accepted
/// answer, a prepared scene, or the scene its session already presents.
#[derive(Clone)]
pub(crate) enum StagedScene { Display, Prepared(PreparedPlot), Shown }

/// One staged presentation (`_staged[key]`).
#[derive(Clone)]
pub(crate) struct Staged<R> { pub read: R, pub scene: StagedScene, pub configuration: Arc<ResolvedConfiguration>, pub live_data: bool }

/// The ROI a panel's selection planned, with its identity.
#[derive(Clone, PartialEq)]
pub(crate) struct RoiKey {
    pub signal: Option<String>, pub window: i64, pub schema: Arc<zlc_core::schema::DatasetSchema>, pub subject: Json, pub kind: String, pub value: Json,
}

/// The document as it stood when a command began (`_begin_presentation`).
struct Transaction<R> {
    sessions: OrderedMap<String, SessionHandle>,
    inputs: OrderedMap<String, R>,
    panel_sources: OrderedMap<String, PanelSource>,
    configurations: OrderedMap<String, Configured>,
    roi_keys: OrderedMap<String, RoiKey>,
    panel_status: OrderedMap<String, PanelStatus>,
    fit_outputs: BTreeSet<String>,
    document: (String, Json),
    action: String,
}

/// The Figure document.
pub struct FigureDocument<R: DocumentRead> {
    pub(crate) services: SessionServices,
    pub path: String,
    pub info: Json,
    pub(crate) sessions: OrderedMap<String, SessionHandle>,
    pub(crate) inputs: OrderedMap<String, R>,
    pub(crate) panel_status: OrderedMap<String, PanelStatus>,
    pub(crate) panel_sources: OrderedMap<String, PanelSource>,
    pub dpr: f64,
    pub(crate) roi_keys: OrderedMap<String, RoiKey>,
    pub(crate) fit_outputs: BTreeSet<String>,
    pub(crate) staged: OrderedMap<String, Staged<R>>,
    pub(crate) manual_fits: Vec<Effect<R>>,
    pub(crate) offered_fronts: BTreeSet<String>,
    pub(crate) initial_overlays: OrderedMap<String, Option<Arc<ImagePointOverlay>>>,
    pub(crate) configurations: OrderedMap<String, Configured>,
    pub(crate) removed_outputs: BTreeSet<(String, String)>,
    presentation: Option<Transaction<R>>,
    pub(crate) saver: Option<Arc<dyn FigureSaver<R>>>,
}

fn status_text(error: &PlotError) -> String { error.message.clone() }

/// `str(error)` of a command-level failure, as `Type: message`.
pub fn error_text(error: &PlotError) -> String { format!("{}: {}", error.python_name(), error.message) }

fn json_key_error(name: &str) -> PlotError { PlotError::key(format!("'{name}'")) }

/// The `_axis_wire` form of one axes transform.
pub fn axis_wire(transform: &AxisTransform) -> Json {
    fn number(value: f64) -> Json { if value.is_finite() { json!(value) } else { json!({"nonfinite": crate::pyfmt::repr(value)}) } }
    fn pair(value: (f64, f64)) -> Json { json!([number(value.0), number(value.1)]) }
    fn scale(value: &AxisScale) -> Json {
        match value {
            AxisScale::Linear => json!("linear"),
            AxisScale::Log => json!("log"),
            AxisScale::Samples(values) => Json::Array(values.iter().map(|v| number(*v)).collect()),
            AxisScale::Converted { mapping, canonical, display } =>
                json!([scale(mapping), {"unit": canonical.symbol}, {"unit": display.symbol}]),
        }
    }
    json!({
        "role": transform.role, "cell_index": transform.cell_index,
        "bounds": [number(transform.bounds.0), number(transform.bounds.1), number(transform.bounds.2), number(transform.bounds.3)],
        "x_limits": pair(transform.x_limits), "y_limits": pair(transform.y_limits),
        "canonical_x_limits": pair(transform.canonical_x_limits), "canonical_y_limits": pair(transform.canonical_y_limits),
        "x_scale": scale(&transform.x_scale), "y_scale": scale(&transform.y_scale),
        "canonical_x_scale": transform.canonical_x_scale.as_ref().map_or(Json::Null, scale),
        "canonical_y_scale": transform.canonical_y_scale.as_ref().map_or(Json::Null, scale),
    })
}

/// `AxisTransform(**_axis_value(...))` of an axes wire a frontend sent back.
pub fn axis_from_wire(wire: &Json) -> Result<AxisTransform> {
    fn number(value: &Json) -> Result<f64> {
        match value {
            Json::Number(n) => Ok(n.as_f64().unwrap_or(f64::NAN)),
            Json::Object(map) if map.len() == 1 && map.contains_key("nonfinite") =>
                map["nonfinite"].as_str().and_then(crate::configuration::value::parse_python_float)
                    .ok_or_else(|| PlotError::value("could not convert string to float")),
            _ => Err(PlotError::type_error("axes limits must be numbers")),
        }
    }
    fn pair(value: &Json) -> Result<(f64, f64)> {
        let items = value.as_array().filter(|items| items.len() == 2).ok_or_else(|| PlotError::type_error("axes limits must be pairs"))?;
        Ok((number(&items[0])?, number(&items[1])?))
    }
    fn unit(value: &Json) -> Result<Arc<crate::units::Unit>> {
        let symbol = value.get("unit").and_then(Json::as_str).ok_or_else(|| PlotError::type_error("axes unit must be text"))?;
        Ok(crate::units::resolve_unit(symbol, None)?)
    }
    fn scale(value: &Json) -> Result<AxisScale> {
        Ok(match value {
            Json::String(text) if text == "linear" => AxisScale::Linear,
            Json::String(text) if text == "log" => AxisScale::Log,
            Json::Array(items) if items.len() == 3 && items[1].is_object() =>
                AxisScale::Converted { mapping: Box::new(scale(&items[0])?), canonical: unit(&items[1])?, display: unit(&items[2])? },
            Json::Array(items) => AxisScale::Samples(items.iter().map(number).collect::<Result<Vec<_>>>()?.into()),
            _ => return Err(PlotError::value("unknown axis scale")),
        })
    }
    let field = |name: &str| wire.get(name).ok_or_else(|| PlotError::type_error(format!("AxisTransform.__init__() missing required argument: '{name}'")));
    let bounds = field("bounds")?.as_array().filter(|items| items.len() == 4).ok_or_else(|| PlotError::type_error("axes bounds must have four numbers"))?;
    let optional_scale = |name: &str| -> Result<Option<AxisScale>> { match wire.get(name) { None | Some(Json::Null) => Ok(None), Some(value) => scale(value).map(Some) } };
    Ok(AxisTransform {
        role: field("role")?.as_str().unwrap_or("main").to_owned(), cell_index: field("cell_index")?.as_u64(),
        bounds: (number(&bounds[0])?, number(&bounds[1])?, number(&bounds[2])?, number(&bounds[3])?),
        x_limits: pair(field("x_limits")?)?, y_limits: pair(field("y_limits")?)?,
        canonical_x_limits: pair(field("canonical_x_limits")?)?, canonical_y_limits: pair(field("canonical_y_limits")?)?,
        x_scale: wire.get("x_scale").map(scale).transpose()?.unwrap_or(AxisScale::Linear),
        y_scale: wire.get("y_scale").map(scale).transpose()?.unwrap_or(AxisScale::Linear),
        canonical_x_scale: optional_scale("canonical_x_scale")?, canonical_y_scale: optional_scale("canonical_y_scale")?,
    })
}

/// Python's `json.dumps(target, sort_keys=True)` equality of two targets.
fn same_target(a: &ValueMap, b: &ValueMap) -> bool { Value::Map(a.clone()).to_json() == Value::Map(b.clone()).to_json() }

fn file_name(path: &str) -> String { path.rsplit(['/', '\\']).find(|part| !part.is_empty()).unwrap_or("").to_owned() }

impl<R: DocumentRead> FigureDocument<R> {
    pub fn new(services: SessionServices, saver: Option<Arc<dyn FigureSaver<R>>>) -> Self {
        FigureDocument {
            services, path: String::new(), info: json!({}), sessions: OrderedMap::new(), inputs: OrderedMap::new(), panel_status: OrderedMap::new(),
            panel_sources: OrderedMap::new(), dpr: 1.0, roi_keys: OrderedMap::new(), fit_outputs: BTreeSet::new(), staged: OrderedMap::new(),
            manual_fits: Vec::new(), offered_fronts: BTreeSet::new(), initial_overlays: OrderedMap::new(), configurations: OrderedMap::new(),
            removed_outputs: BTreeSet::new(), presentation: None, saver,
        }
    }

    // ------------------------------------------------------------- transactions
    fn begin_presentation(&mut self, action: &str) -> Result<()> {
        if self.presentation.is_some() { return Err(PlotError::runtime("The previous native presentation has not been acknowledged")); }
        self.presentation = Some(Transaction {
            sessions: self.sessions.clone(), inputs: self.inputs.clone(), panel_sources: self.panel_sources.clone(),
            configurations: self.configurations.clone(), roi_keys: self.roi_keys.clone(), panel_status: self.panel_status.clone(),
            fit_outputs: self.fit_outputs.clone(), document: (self.path.clone(), self.info.clone()), action: action.to_owned(),
        });
        for handle in self.sessions.values() { lock(handle).begin_presentation()?; }
        Ok(())
    }

    /// `finish_presentation(accepted_targets, accepted_fronts)`: keep the
    /// accepted panels' new authoring and visible data, restore the rest.
    pub fn finish_presentation(&mut self, accepted_targets: &BTreeSet<String>, accepted_fronts: &BTreeSet<String>) {
        let Some(saved) = self.presentation.take() else { return };
        let mut keys: Vec<String> = Vec::new();
        for key in saved.sessions.keys().chain(self.sessions.keys()).chain(saved.configurations.keys()).chain(self.configurations.keys())
            .chain(saved.panel_status.keys()).chain(self.panel_status.keys()) {
            if !keys.contains(key) { keys.push(key.clone()); }
        }
        for key in &keys {
            let old = saved.sessions.get(key).cloned();
            let current = self.sessions.get(key).cloned();
            let removed = current.is_none() && !self.configurations.contains_key(key);
            if !accepted_targets.contains(key) {
                restore(&mut self.configurations, &saved.configurations, key);
                restore(&mut self.roi_keys, &saved.roi_keys, key);
                restore(&mut self.panel_status, &saved.panel_status, key);
                if saved.fit_outputs.contains(key) { self.fit_outputs.insert(key.clone()); } else { self.fit_outputs.remove(key); }
            }
            let candidate = self.configurations.get(key).map(|configured| configured.configuration.clone());
            let promoted = accepted_fronts.contains(key);
            let accepted_unchanged = accepted_targets.contains(key) && !self.offered_fronts.contains(key) && !self.staged.contains_key(key);
            let same = |a: &Option<SessionHandle>, b: &Option<SessionHandle>| matches!((a, b), (Some(a), Some(b)) if Arc::ptr_eq(a, b));
            if let Some(current) = &current { lock(current).finish_presentation(promoted || accepted_unchanged, candidate.clone()); }
            if let Some(old) = old.as_ref().filter(|_| !same(&current, &old)) { lock(old).finish_presentation(false, candidate.clone()); }
            if promoted || accepted_unchanged || (removed && accepted_targets.contains(key)) {
                if let Some(old) = old.as_ref().filter(|_| !same(&current, &old)) { lock(old).close(); }
                continue;
            }
            if let Some(current) = current.as_ref().filter(|_| !same(&current, &old)) { lock(current).close(); }
            restore(&mut self.sessions, &saved.sessions, key);
            restore(&mut self.inputs, &saved.inputs, key);
            restore(&mut self.panel_sources, &saved.panel_sources, key);
        }
        if (saved.action == "replace_document" || saved.action == "close") && !keys.iter().all(|key| accepted_targets.contains(key)) {
            (self.path, self.info) = saved.document;
        }
        self.staged = OrderedMap::new();
        self.manual_fits.clear();
        self.initial_overlays = OrderedMap::new();
    }

    /// `close`: every panel's outputs are removed.
    fn close(&mut self) {
        for key in self.sessions.keys() {
            self.removed_outputs.insert((key.clone(), "roi".into()));
            self.removed_outputs.insert((key.clone(), "fit".into()));
        }
        if self.presentation.is_none() { for handle in self.sessions.values() { lock(handle).close(); } }
        self.sessions = OrderedMap::new();
        self.inputs = OrderedMap::new();
        self.panel_status = OrderedMap::new();
        self.panel_sources = OrderedMap::new();
        self.path.clear();
        self.info = json!({});
        self.roi_keys = OrderedMap::new();
        self.fit_outputs.clear();
        self.configurations = OrderedMap::new();
    }

    // ------------------------------------------------------------- describe
    fn describe(&self, key: &str, target: Option<&Json>) -> Result<Json> {
        let session = self.sessions.get(&key.to_owned()).cloned();
        let Some(configured) = self.configurations.get(&key.to_owned()) else { return self.placeholder(key, target) };
        let configuration = &configured.configuration;
        let status = self.panel_status.get(&key.to_owned()).cloned().unwrap_or_default();
        let mut row = describe_panel(configuration, self.panel_status.contains_key(&key.to_owned()))?;
        let session = session.as_ref().map(lock);
        if status.authoring_pending == Some(true) { if let Some(target) = target { row["target"] = target.clone(); } }
        let structure = self.inputs.get(&key.to_owned()).map(|read| {
            let schema = PlotSchema::new(read.source().schema.clone());
            let mut structure = schema_structure(&schema);
            if let Some(counts) = &read.source().repeat_counts {
                structure[0] = read.source().schema.domain(zlc_core::schema::DomainRole::Repeat).axes().iter().zip(counts)
                    .filter(|(axis, _)| axis.coordinate_of().is_none()).map(|(axis, count)| (axis.name().to_owned(), *count)).collect();
            }
            Json::Array(structure.iter().map(|group| Json::Array(group.iter().map(|(name, size)| json!([name, size])).collect())).collect())
        });
        let source = self.panel_sources.get(&key.to_owned());
        let object = row.as_object_mut().expect("panel row");
        object.insert("id".into(), json!(key));
        object.insert("title".into(), json!(key.rsplit('/').next().unwrap_or(key)));
        if let Some(session) = &session { object.insert("kind".into(), json!(session.kind().value())); }
        object.insert("focus".into(), session.as_ref().and_then(|session| session.facet_focus()).map_or(Json::Null, |focus| json!(focus)));
        object.insert("structure".into(), structure.unwrap_or(Json::Null));
        object.insert("source_signal".into(), source.map_or(Json::Null, |source| json!(source.signal)));
        object.insert("data_binding_epoch".into(), source.map_or(Json::Null, |source| json!(source.epoch)));
        object.insert("error".into(), status.error.clone().or(status.edit_error.clone()).or(status.derivation_error.clone())
            .map_or(Json::Null, Json::String));
        object.insert("message".into(), status.message.clone().map_or(Json::Null, Json::String));
        object.insert("authoring_pending".into(), json!(status.authoring_pending.unwrap_or(false)));
        object.insert("target_request".into(), status.target_request.clone().unwrap_or(Json::Null));
        let axes = match &session { None => Vec::new(), Some(session) => session.raster_axes_snapshot()?.iter().map(axis_wire).collect() };
        object.insert("axes".into(), Json::Array(axes));
        let computed = session.as_ref().map_or_else(ValueMap::new, |session| session.fit_configuration());
        object["fit"]["computed"] = Value::Map(computed).to_json();
        Ok(row)
    }

    fn placeholder(&self, key: &str, target: Option<&Json>) -> Result<Json> {
        let status = self.panel_status.get(&key.to_owned()).cloned().unwrap_or_default();
        let binding = status.binding.as_ref().ok_or_else(|| json_key_error("binding"))?;
        let layout = &self.services.defaults.layout;
        let mut size = target.and_then(|target| target["size"].as_str()).unwrap_or(layout.default_preset).to_owned();
        if let Some(queue) = &status.queue {
            for edit in queue.start.iter().chain(&queue.edits) {
                if let Some(value) = edit.target.get("size").and_then(Value::as_str) { size = value.to_owned(); }
            }
        }
        let size = layout.validate_preset(&size).map_err(PlotError::value)?;
        let kind = crate::layout::Kind::parse(binding.kind().unwrap_or("curve")).unwrap_or(crate::layout::Kind::Curve);
        let (width, height) = layout.logical_size(size, kind).map_err(PlotError::value)?;
        let mut authored = target.cloned().unwrap_or_else(|| json!({}));
        if !authored.is_object() { authored = json!({}); }
        authored["size"] = json!(size);
        Ok(json!({
            "id": key, "title": key.rsplit('/').next().unwrap_or(key), "kind": binding.raw.get("kind").cloned().unwrap_or(Json::Null),
            "controls": [], "axes": [], "target": authored, "focus": null,
            "authoring_pending": status.authoring_pending.unwrap_or(false), "target_request": status.target_request.clone().unwrap_or(Json::Null),
            "fit": {"models": [], "selected": "", "configuration": {}, "computed": {}, "expression": "", "error": ""},
            "size": size, "size_choices": crate::layout::PANEL_PRESETS.iter().map(|preset| preset.0).collect::<Vec<_>>(),
            "display_size": {"width": width, "height": height}, "structure": null, "source_signal": null,
            "data_binding_epoch": self.panel_sources.get(&key.to_owned()).map_or(Json::Null, |source| json!(source.epoch)),
            "error": status.error.clone().map_or(Json::Null, Json::String), "message": status.message.clone().map_or(Json::Null, Json::String),
        }))
    }

    pub(crate) fn status_mut(&mut self, key: &str) -> &mut PanelStatus {
        if !self.panel_status.contains_key(&key.to_owned()) { self.panel_status.insert(key.to_owned(), PanelStatus::default()); }
        self.panel_status.get_mut(&key.to_owned()).expect("status")
    }

    // ------------------------------------------------------------- commands
    fn bindings(command: &Json) -> Result<Vec<Binding>> {
        let Some(inputs) = command.get("inputs") else { return Ok(Vec::new()) };
        let inputs = inputs.as_array().ok_or_else(|| PlotError::type_error("inputs must be an array of panel bindings"))?;
        let mut identities = BTreeSet::new();
        let mut bindings = Vec::with_capacity(inputs.len());
        for raw in inputs {
            let key = raw.get("panel").and_then(Json::as_str);
            let canonical = |text: &str| !crate::configuration::value::py_strip(text).is_empty() && text == crate::configuration::value::py_strip(text);
            let Some(key) = key.filter(|key| canonical(key) && !identities.contains(*key)) else {
                return Err(PlotError::value("panel bindings require unique nonempty canonical identities"));
            };
            identities.insert(key.to_owned());
            // A panel added before its signal is chosen (v2's blank panel) binds none
            // yet: it reads nothing and says why (its binding's reason).
            let signal = raw.get("signal").and_then(Json::as_str).filter(|signal| signal.is_empty() || canonical(signal))
                .ok_or_else(|| PlotError::value("signal identity must be nonempty canonical text"))?;
            let window = raw.get("window").and_then(|w| if w.is_i64() || w.is_u64() { w.as_i64() } else { None }).filter(|w| *w >= 1)
                .ok_or_else(|| PlotError::value("binding window must be a positive integer"))?;
            let epoch = raw.get("epoch").and_then(Json::as_u64).filter(|_| raw["epoch"].is_u64())
                .ok_or_else(|| PlotError::value("binding epoch must be an unsigned 64-bit integer"))?;
            bindings.push(Binding { raw: raw.clone(), panel: key.to_owned(), signal: signal.to_owned(), window, epoch });
        }
        Ok(bindings)
    }

    /// `execute(command, reads, publisher, initial_overlays)`.
    pub fn execute(&mut self, command_json: &str, mut reads: BTreeMap<String, R>, publisher: Option<&dyn FitPublisher<R>>,
                   initial_overlays: Option<BTreeMap<String, Option<Arc<ImagePointOverlay>>>>) -> Result<Execution<R>> {
        let started = Instant::now();
        let command: Json = serde_json::from_str(command_json).map_err(|error| PlotError::value(error.to_string()))?;
        let action = command.get("action").and_then(Json::as_str).ok_or_else(|| json_key_error("action"))?.to_owned();
        let replacing = command.get("initial_document").is_some_and(|value| truthy_json(value));
        self.begin_presentation(if replacing { "replace_document" } else { &action })?;
        self.staged = OrderedMap::new();
        self.manual_fits.clear();
        self.initial_overlays = OrderedMap::from_pairs(initial_overlays.unwrap_or_default());
        self.offered_fronts.clear();
        let mut targets: Map<String, Json> = command.get("targets").and_then(Json::as_object).cloned().unwrap_or_default();
        if action != "sync_panels" {
            for binding in Self::bindings(&command)? {
                let panel = binding.panel.clone();
                self.status_mut(&panel).binding = Some(binding);
            }
            if let Some(requests) = command.get("target_requests").and_then(Json::as_object) {
                for (key, request) in requests {
                    if let Some(status) = self.panel_status.get_mut(key) { status.target_request = Some(request.clone()); }
                }
            }
        }
        let mut fronts: BTreeMap<String, RgbaFront> = BTreeMap::new();
        let mut saved: Json = Json::Null;
        match action.as_str() {
            "ready" => {
                let dpr = command.get("dpr").map_or(Some(1.0), Json::as_f64).unwrap_or(f64::NAN);
                if !dpr.is_finite() || dpr <= 0.0 { return Err(PlotError::value("Device pixel ratio must be finite and positive")); }
                self.dpr = dpr;
            }
            "sync_panels" => fronts = self.sync_panels(&command, &mut reads, &mut targets, publisher)?,
            "close" => self.close(),
            "describe" => {}
            "get_fit" => {
                let key = command.get("panel").and_then(Json::as_str).ok_or_else(|| json_key_error("panel"))?;
                let handle = self.sessions.get(&key.to_owned()).cloned()
                    .ok_or_else(|| PlotError::value("This panel has no presented data, so it has no fit yet"))?;
                let integer = |name: &str, default: i64| -> Result<i64> {
                    match command.get(name) {
                        None | Some(Json::Null) => Ok(default),
                        Some(value) => value.as_i64().filter(|_| value.is_i64() || value.is_u64()).or_else(|| value.as_f64().map(|v| v as i64))
                            .ok_or_else(|| PlotError::value(format!("invalid literal for int() with base 10: {value}"))),
                    }
                };
                let (offset, limit) = (integer("offset", 0)?, integer("limit", 256)?);
                let (offset, limit) = (if offset == 0 { 0 } else { offset }, if limit == 0 { 256 } else { limit });
                saved = lock(&handle).fit_summary(offset, limit)?;
            }
            "close_panel" => {
                let key = command.get("panel").and_then(Json::as_str).ok_or_else(|| json_key_error("panel"))?.to_owned();
                self.removed_outputs.insert((key.clone(), "roi".into()));
                self.removed_outputs.insert((key.clone(), "fit".into()));
                self.roi_keys.remove(&key);
                self.fit_outputs.remove(&key);
                self.configurations.remove(&key);
                self.sessions.remove(&key);
                self.inputs.remove(&key);
                self.panel_status.remove(&key);
                self.panel_sources.remove(&key);
            }
            _ => {
                let key = command.get("panel").and_then(Json::as_str).ok_or_else(|| json_key_error("panel"))?.to_owned();
                if action == "save" {
                    saved = self.save(&key, &command)?;
                } else {
                    let _span = crate::render::profile::span("document.live");
                    fronts.extend(self.live_command(&command, &reads, &mut targets)?);
                }
            }
        }
        if action != "sync_panels" { let _span = crate::render::profile::span("document.present_staged"); fronts.extend(self.present_staged()); }
        let effects = self.derivative_effects()?;
        let mut keys: Vec<String> = Vec::new();
        for key in self.panel_status.keys().chain(self.sessions.keys()).chain(self.configurations.keys()) {
            if !keys.contains(key) { keys.push(key.clone()); }
        }
        let span = crate::render::profile::span("document.describe");
        let mut panels = keys.iter().map(|key| self.describe(key, targets.get(key))).collect::<Result<Vec<_>>>()?;
        drop(span);
        if action == "save" {
            let key = command["panel"].as_str().unwrap_or_default();
            if self.panel_status.get(&key.to_owned()).and_then(|status| status.queue.as_ref()).is_some_and(|queue| !queue.edits.is_empty()) {
                // A save writes the figure as shown, as v2's did; it says what that leaves out.
                if let Some(row) = panels.iter_mut().find(|row| row["id"] == key) {
                    row["message"] = json!("Saved as shown: the edits waiting for data are not in the file");
                }
            }
        }
        let title = if !self.path.is_empty() { file_name(&self.path) } else if !self.panel_status.is_empty() { "Live sources".into() } else { "Figure Viewer".into() };
        let state = json!({
            "type": "state", "request": command.get("id").cloned().unwrap_or(json!(0)),
            "document": {"path": self.path, "title": title}, "panels": panels, "plot_kinds": plot_kinds(), "error": null,
            "saved": saved, "result": if action == "get_fit" { saved.clone() } else { Json::Null },
            "scientific_ms": started.elapsed().as_secs_f64() * 1000.0,
        });
        let admitted: BTreeMap<String, R> = self.staged.iter().filter(|(_, staged)| !matches!(staged.scene, StagedScene::Display))
            .map(|(key, staged)| (key.clone(), staged.read.clone())).collect();
        self.offered_fronts = fronts.keys().cloned().collect();
        let presented = self.inputs.iter().map(|(key, read)| (key.clone(), read.clone())).collect();
        Ok(Execution { state, fronts, effects, presented, admitted })
    }

    /// `_live_command`: one panel's authoring command.
    fn live_command(&mut self, command: &Json, reads: &BTreeMap<String, R>, targets: &mut Map<String, Json>) -> Result<BTreeMap<String, RgbaFront>> {
        let key = command["panel"].as_str().expect("panel").to_owned();
        let action = command["action"].as_str().unwrap_or("").to_owned();
        let status = self.panel_status.get(&key).cloned().ok_or_else(|| json_key_error(&key))?;
        let read = reads.get(&key).cloned();
        let at = crate::session::gesture::monotonic();
        let mut frames = BTreeMap::new();
        let mut interaction_updates: Option<InteractionUpdates> = None;
        if action == "pointer" {
            let handle = self.sessions.get(&key).cloned().ok_or_else(|| PlotError::value("There is no presented scene to interact with"))?;
            if read.is_none() || status.queue.is_none() {
                // Drawn now; a waiting panel also replays it at its turn.
                let mut session = lock(&handle);
                let before = session.figure_recipe()?;
                let event = pointer_event(command, Some(at))?;
                let publish = guarded(|| session.raster_pointer_event(&event))?;
                let recipe = gesture_updates(&before, &session.figure_recipe()?);
                if publish { frames.insert(key.clone(), guarded(|| session.rgba())?); }
                if read.is_some() && recipe.is_empty() { return Ok(frames); }
                interaction_updates = Some(InteractionUpdates { recipe, ..Default::default() });
            }
        } else if !["configure", "scope_step", "focus", "reset_view", "fit"].contains(&action.as_str()) {
            return Err(PlotError::value(format!("Unknown Figure action: {action}")));
        }
        let binding = status.binding.clone().ok_or_else(|| json_key_error("binding"))?;
        let Some(read) = read else {
            let previous = self.configurations.get(&key).cloned();
            let pointer = if action == "pointer" { vec![(command.clone(), at)] } else { Vec::new() };
            let edit = Edit::of(&key, &binding, command, None, pointer)?;
            let configured = self.configurations.contains_key(&key);
            edit::defer(&key, self.status_mut(&key), configured, &binding, targets.get(&key), Some(edit.clone()))?;
            // The pending target shows the queued intent; it is never resolved.
            let mut target: ValueMap = match targets.get(&key).filter(|target| truthy_json(target)) {
                Some(target) => Value::from_json(target).as_map().cloned().unwrap_or_default(),
                None => match &previous { Some(previous) => previous.configuration.target()?, None => ValueMap::new() },
            };
            target.update(&edit.target);
            for (name, value) in interaction_updates.iter().flat_map(|updates| updates.recipe.iter()) {
                let value = match (name, value) {
                    ("parameters", Value::Map(stated)) => {
                        let mut merged = target.get(name).and_then(Value::as_map).cloned().unwrap_or_default();
                        merged.update(stated);
                        Value::Map(merged)
                    }
                    _ => value.clone(),
                };
                target.insert(name, value);
            }
            targets.insert(key.clone(), Value::Map(target).to_json());
            self.status_mut(&key).message = Some(binding.reason().unwrap_or("Waiting for source data").to_owned());
            return Ok(frames);
        };
        let mut panel = self.panel(&key);
        let shared = Shared { services: &self.services, dpr: self.dpr };
        let outcome = panel.resolve(&shared, &binding, &read, targets.get(&key), command, interaction_updates, at).and_then(|resolution| {
            if let Some(failure) = resolution.failure { return Err(failure); }
            targets.insert(key.clone(), Value::Map(resolution.configuration.target()?).to_json());
            if action == "pointer" && resolution.shown.is_none() { return Ok(()); }
            panel.apply(&shared, &read, resolution.configuration, resolution.changed, false, None, resolution.shown)
        });
        self.store_prepared(&mut panel);
        self.store_presented(panel);
        outcome.map(|()| frames)
    }

    /// `_present_staged`: present every staged panel, each on its own worker.
    fn present_staged(&mut self) -> BTreeMap<String, RgbaFront> {
        let mut panels: Vec<Panel<R>> = self.staged.keys().map(|key| self.panel(key)).collect();
        let shared = Shared { services: &self.services, dpr: self.dpr };
        let presented: Vec<(Option<RgbaFront>, bool)> = panels.par_iter_mut().map(|panel| panel.present(&shared)).collect();
        let mut results = Vec::with_capacity(panels.len());
        for (panel, (front, unchanged)) in panels.into_iter().zip(presented) {
            results.push((panel.key.clone(), front, unchanged));
            self.store_presented(panel);
        }
        self.collect_fronts(results)
    }

    pub(crate) fn collect_fronts(&mut self, results: Vec<(String, Option<RgbaFront>, bool)>) -> BTreeMap<String, RgbaFront> {
        let mut frames = BTreeMap::new();
        for (key, front, unchanged) in results {
            if let Some(front) = front { frames.insert(key.clone(), front); }
            if unchanged { self.staged.remove(&key); }
        }
        frames
    }

    /// The session of one panel.
    pub fn session(&self, panel: &str) -> Option<SessionHandle> { self.sessions.get(&panel.to_owned()).cloned() }
}

/// A defect while one panel draws is that panel's error: the worker, the
/// document and the other panels go on.
pub(crate) fn guarded<T>(work: impl FnOnce() -> Result<T>) -> Result<T> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(work)).unwrap_or_else(|panic| {
        let reason = panic.downcast_ref::<String>().cloned().or_else(|| panic.downcast_ref::<&str>().map(|text| text.to_string()));
        Err(PlotError::runtime(format!("Drawing failed: {}", reason.unwrap_or_default())))
    })
}

pub(crate) fn truthy_json(value: &Json) -> bool {
    match value {
        Json::Null => false, Json::Bool(flag) => *flag, Json::Number(n) => n.as_f64().is_some_and(|v| v != 0.0),
        Json::String(text) => !text.is_empty(), Json::Array(items) => !items.is_empty(), Json::Object(map) => !map.is_empty(),
    }
}

fn restore<V: Clone>(map: &mut OrderedMap<String, V>, saved: &OrderedMap<String, V>, key: &String) {
    match saved.get(key) { Some(value) => map.insert(key.clone(), value.clone()), None => { map.remove(key); } }
}

/// `_pointer(session, command, at)`: the pointer message of a command.
pub fn pointer_event(command: &Json, time: Option<f64>) -> Result<PointerEvent> {
    let event = command.get("event").and_then(Json::as_str).ok_or_else(|| json_key_error("event"))?;
    let axes = match command.get("axes") { None | Some(Json::Null) => None, Some(wire) => Some(axis_from_wire(wire)?) };
    let number = |name: &str| command.get(name).map_or(Some(0.0), Json::as_f64).unwrap_or(f64::NAN);
    let button = match command.get("button").and_then(Json::as_str) { Some("left") => Some(1), Some("middle") => Some(2), Some("right") => Some(3), _ => None };
    Ok(PointerEvent {
        action: PointerAction::parse(if event == "double_click" { "press" } else { event })?, x: number("x"), y: number("y"), button,
        double: event == "double_click", step: number("steps"), key: command.get("key").and_then(Json::as_str).map(str::to_owned), axes, time,
    })
}

/// `_gesture_updates(before, after)`: what a gesture changed in a session's
/// recipe, as interaction updates.
pub(crate) fn gesture_updates(before: &ValueMap, after: &ValueMap) -> ValueMap {
    let mut recipe = ValueMap::new();
    for name in ["selectors", "viewport", "facet_focus", "classifier_thresholds", "interaction", "presentation"] {
        if before.get(name) != after.get(name) { recipe.insert(name, after.get(name).cloned().unwrap_or(Value::None)); }
    }
    let before_parameters = before.get("parameters").and_then(Value::as_map).cloned().unwrap_or_default();
    let mut parameters = ValueMap::new();
    for (name, value) in after.get("parameters").and_then(Value::as_map).cloned().unwrap_or_default().iter() {
        if before_parameters.get(name) != Some(value) { parameters.insert(name, value.clone()); }
    }
    if !parameters.is_empty() { recipe.insert("parameters", Value::Map(parameters)); }
    recipe
}
