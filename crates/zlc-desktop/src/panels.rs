//! Panel source bindings own leases, not scientific data or accepted fronts.
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use serde_json::{Value, json};
use zlc_core::runtime::{Front, Generation, HistoryLease, Read, Runtime, RuntimeError, SignalDescription, PublicationKey};
use zlc_plot::session::{ImagePointOverlay, OverlayArray, PointStatus};

/// The worker must hold this same lock across the permission check and commit.
pub type BindingAuthority = Arc<Mutex<bool>>;

struct Binding {
    signal: Arc<str>,
    window: usize,
    generation: Option<Generation>,
    lease: Option<HistoryLease>,
    epoch: u64,
    initial_kind: String,
    reason: Option<String>,
    authority: BindingAuthority,
    intent_authority: BindingAuthority,
    /// An Edit's: the one read it shows until Refresh snapshot.
    frozen: Option<Frozen>,
    /// A data copy's preview: the archive's point layer it keeps while its coordinates still match.
    layer: Option<Arc<ImagePointOverlay>>,
}

/// What a panel's Edit holds still (v2 frozen_data): the read the panel had
/// when the Edit opened or was refreshed, its Overlay's judgement, and when.
/// A point review holds its image the same way, with its candidates.
struct Frozen { of: String, read: Read, overlay: Option<Read>, at_ms: u128, review: Option<Review> }

/// A run's point review (v2 PointReviewView's plot): `review-<node>`, the
/// image a point-selection request names, on the read it had when the review
/// opened, every candidate drawn as the review marks it -- kept, excluded or
/// selected (v2 _ImagePointReviewInteraction.set_state).  Never a board panel.
pub const REVIEW_PREFIX: &str = "review-";
pub fn review_of(node: &str) -> String { format!("{REVIEW_PREFIX}{node}") }
pub fn is_review(panel: &str) -> bool { panel.starts_with(REVIEW_PREFIX) }

/// One candidate: its id, its label and where it is (canonical x, y).
pub type ReviewPoint = (String, String, f64, f64);

struct Review { points: Vec<ReviewPoint>, layer: Arc<ImagePointOverlay> }

/// The candidates as one point layer: selected ones OCCUPIED, excluded INVALID, the rest UNKNOWN.
fn review_layer(points: &[ReviewPoint], excluded: &BTreeSet<String>, selected: &BTreeSet<String>, revision: u64) -> Result<Arc<ImagePointOverlay>, String> {
    let coordinates = points.iter().flat_map(|(_, _, x, y)| [*x, *y]).collect();
    let statuses = points.iter().map(|(id, ..)| if selected.contains(id) { PointStatus::Occupied }
        else if excluded.contains(id) { PointStatus::Invalid } else { PointStatus::Unknown }).collect();
    ImagePointOverlay::new(i128::from(revision), OverlayArray { values: Some(coordinates), shape: vec![points.len(), 2] },
        Some(points.iter().map(|(id, ..)| id.clone()).collect()), Some(points.iter().map(|(_, label, ..)| Some(label.clone())).collect()),
        Some(statuses), None, None).map(Arc::new).map_err(|error| error.message)
}

/// A panel's Edit (v2 Panel Edit) is `edit-<panel>`: that panel's
/// configuration on one frozen read, never a board panel.
pub const EDIT_PREFIX: &str = "edit-";
pub fn edit_of(panel: &str) -> String { format!("{EDIT_PREFIX}{panel}") }
pub fn is_edit(panel: &str) -> bool { panel.starts_with(EDIT_PREFIX) }

impl Binding {
    fn revoke(&self) {
        // Revocation must still happen if a previous publisher unwound while
        // holding the lock; do not panic during Binding's destruction.
        *self.authority.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = false;
    }
}

impl Drop for Binding {
    fn drop(&mut self) {
        self.revoke();
        *self.intent_authority.lock().unwrap_or_else(|poisoned|poisoned.into_inner())=false;
    }
}

pub struct PanelInput {
    pub panel: String,
    pub kind: String,
    pub window: usize,
    pub signal: Arc<str>,
    pub epoch: u64,
    pub read: Option<Read>,
    /// The panel's Overlay judgement, read with `read` (its exact parent picture).
    pub overlay: Option<Read>,
    /// A point layer of its own (a point review's candidates), drawn instead of any Overlay.
    pub layer: Option<Arc<ImagePointOverlay>>,
    pub reason: Option<String>,
    pub waiting: bool,
}

/// A Figure Viewer's panels are their own set: `figure-<viewer>-<k>`, never a board's `panel-<k>`.
pub const VIEWER_PREFIX: &str = "figure";
pub fn viewer_prefix(viewer: u32) -> String { format!("{VIEWER_PREFIX}-{viewer}") }

pub struct Panels {
    bindings: BTreeMap<String, Binding>,
    next_id: u64,
    prefix: String,
    /// The point reviews' layer revisions, one counter for all.
    review_revision: u64,
    /// This board's last same-shot selection, its own whatever other boards read.
    front: Mutex<Front>,
}

impl Default for Panels { fn default() -> Self { Self::new() } }

impl Panels {
    pub fn new() -> Self { Self::with_prefix("panel") }
    pub fn with_prefix(prefix: impl Into<String>) -> Self { Self { bindings: BTreeMap::new(), next_id: 1, prefix: prefix.into(), review_revision: 0, front: Mutex::default() } }

    pub fn has(&self, panel: &str) -> bool { self.bindings.contains_key(panel) }

    pub fn reserve_ids(&mut self,count:usize)->Result<Vec<String>,String> {
        let end=self.next_id.checked_add(count as u64).ok_or("Panel identity counter exhausted")?;
        let ids=(self.next_id..end).map(|id|format!("{}-{id}",self.prefix)).collect();self.next_id=end;Ok(ids)
    }
    /// An opened archive's panels, a candidate for its Figure Viewer's set.
    pub fn from_inputs(runtime:&Runtime,inputs:&[PanelInput],prefix:&str)->Result<Self,String> {
        let mut panels=Self::with_prefix(prefix);
        let sources=runtime.signals();
        for input in inputs {
            let source=sources.iter().find(|source|source.signal==input.signal).ok_or("Imported source is not published")?;
            panels.bindings.insert(input.panel.clone(),Binding {signal:input.signal.clone(),window:input.window,
                generation:Some(source.generation.clone()),lease:lease_for(runtime,Some(source),input.window)?,epoch:1,
                initial_kind:input.kind.clone(),reason:None,authority:Arc::new(Mutex::new(true)),intent_authority:Arc::new(Mutex::new(true)),frozen:None,layer:None});
        }
        Ok(panels)
    }
    pub fn continue_ids_from(&mut self,previous:&Self) {self.next_id=self.next_id.max(previous.next_id);}

    pub fn add(&mut self, runtime: &Runtime, signal: &str, kind: &str) -> Result<String, String> {
        let id = format!("{}-{}", self.prefix, self.next_id);
        let next = self.next_id.checked_add(1).ok_or("Panel identity counter exhausted")?;
        self.add_as(runtime, &id, signal, kind)?;
        self.next_id = next;
        Ok(id)
    }

    /// Bind a panel under an id `reserve_ids` gave (a loaded layout's).
    pub fn add_as(&mut self, runtime: &Runtime, id: &str, signal: &str, kind: &str) -> Result<(), String> {
        if signal.trim() != signal || kind.trim() != kind || kind.is_empty() {
            return Err("Signal and plot kind must be canonical text; plot kind cannot be empty".into());
        }
        if self.bindings.contains_key(id) { return Err(format!("{id} is already on the board")); }
        let source = runtime.signals().into_iter().find(|source| source.signal.as_ref() == signal);
        // A new panel reads what its kind shows: a Rolling trace its recent history.
        let window = zlc_plot::kinds::PlotKind::parse(kind).map_or(1, zlc_plot::kinds::PlotKind::default_window);
        let lease = lease_for(runtime, source.as_ref(), window)?;
        self.bindings.insert(id.to_owned(), Binding { signal: signal.into(), window,
            generation: source.map(|source| source.generation), lease, epoch: 1,
            initial_kind: kind.into(), reason: None, authority: Arc::new(Mutex::new(true)),intent_authority:Arc::new(Mutex::new(true)), frozen: None, layer: None });
        Ok(())
    }

    /// Hold `panel`'s newest read still as its Edit: opening it, or Refresh snapshot.
    pub fn freeze(&mut self, runtime: &Runtime, panel: &str, accepted: &Value, overlays: &BTreeMap<String, String>) -> Result<String, String> {
        let input = self.capture(runtime, &Value::Null, accepted, overlays).into_iter().find(|input| input.panel == panel)
            .ok_or_else(|| format!("No panel {panel} on the board"))?;
        let read = input.read.ok_or_else(|| input.reason.unwrap_or_else(|| "The panel has no data to hold".into()))?;
        let at_ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis();
        let frozen = Frozen { of: panel.to_owned(), read, overlay: input.overlay, at_ms, review: None };
        let edit = edit_of(panel);
        match self.bindings.get_mut(&edit) {
            Some(binding) => {
                binding.epoch = binding.epoch.checked_add(1).ok_or("Panel binding epoch exhausted")?;
                (binding.signal, binding.window, binding.frozen) = (input.signal, input.window, Some(frozen));
            }
            None => { self.bindings.insert(edit.clone(), Binding { signal: input.signal, window: input.window, generation: None, lease: None, epoch: 1,
                initial_kind: input.kind, reason: None, authority: Arc::new(Mutex::new(true)), intent_authority: Arc::new(Mutex::new(true)), frozen: Some(frozen), layer: None }); }
        }
        Ok(edit)
    }

    /// An Edit follows its panel to a new run of its producer, or to the signal the panel now
    /// shows: once that has published, it takes the newest read as Refresh snapshot does.
    /// Within one run it keeps the read it holds, however far the live data goes.
    pub fn follow_runs(&mut self, runtime: &Runtime, accepted: &Value, overlays: &BTreeMap<String, String>) {
        let sources = runtime.signals();
        let due: Vec<String> = self.bindings.values().filter_map(|edit| {
            let frozen = edit.frozen.as_ref().filter(|frozen| frozen.review.is_none())?;
            let live = self.bindings.get(&frozen.of)?;
            let newest = sources.iter().find(|source| source.signal == live.signal && source.published)?;
            (live.signal != edit.signal || newest.generation.id() != frozen.read.key().generation).then(|| frozen.of.clone())
        }).collect();
        // A panel whose new data cannot be read yet keeps its picture until a later turn.
        for panel in due { let _ = self.freeze(runtime, &panel, accepted, overlays); }
    }

    /// `node`'s point review on `signal`'s newest read, every candidate kept.
    pub fn review(&mut self, runtime: &Runtime, node: &str, signal: &str, points: Vec<ReviewPoint>) -> Result<String, String> {
        if points.is_empty() { return Err("The review names no points".into()); }
        let source = runtime.signals().into_iter().find(|source| &*source.signal == signal && source.published)
            .ok_or_else(|| format!("{signal} has published no image to review"))?;
        let read = runtime.read_views(&[(source.generation.clone(), source.signal.clone(), None)]).remove(0).map_err(|error| error.to_string())?;
        self.review_revision += 1;
        let layer = review_layer(&points, &BTreeSet::new(), &BTreeSet::new(), self.review_revision)?;
        let at_ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis();
        let panel = review_of(node);
        let epoch = self.bindings.get(&panel).map_or(1, |binding| binding.epoch + 1);
        self.bindings.insert(panel.clone(), Binding { signal: signal.into(), window: 1, generation: None, lease: None, epoch,
            initial_kind: "image".into(), reason: None, authority: Arc::new(Mutex::new(true)), intent_authority: Arc::new(Mutex::new(true)),
            frozen: Some(Frozen { of: node.to_owned(), read, overlay: None, at_ms, review: Some(Review { points, layer }) }), layer: None });
        Ok(panel)
    }

    /// The review's marks drawn: `excluded` and `selected` of its candidates.
    pub fn mark_review(&mut self, panel: &str, excluded: &BTreeSet<String>, selected: &BTreeSet<String>) -> Result<(), String> {
        self.review_revision += 1;
        let revision = self.review_revision;
        let binding = self.bindings.get_mut(panel).ok_or_else(|| format!("No point review {panel}"))?;
        let review = binding.frozen.as_mut().and_then(|frozen| frozen.review.as_mut()).ok_or_else(|| format!("{panel} is not a point review"))?;
        if let Some(unknown) = excluded.iter().chain(selected).find(|id| !review.points.iter().any(|(point, ..)| point == *id)) {
            return Err(format!("The review has no point {unknown}"));
        }
        review.layer = review_layer(&review.points, excluded, selected, revision)?;
        binding.epoch = binding.epoch.checked_add(1).ok_or("Panel binding epoch exhausted")?;
        Ok(())
    }

    /// (panel, signal, window, kind as added) of every board panel (not the Edits).
    pub fn describe(&self) -> Vec<(String, String, usize, String)> {
        self.bindings.iter().filter(|(_, binding)| binding.frozen.is_none()).map(|(panel, binding)| (panel.clone(), binding.signal.to_string(), binding.window, binding.initial_kind.clone())).collect()
    }

    /// A rejected edit keeps both the previous binding and its retained data demand.
    pub fn configure(&mut self, runtime: &Runtime, panel: &str, patch: &Value) -> Result<bool, String> {
        let old = self.bindings.get(panel).ok_or_else(|| format!("Unknown panel {panel}"))?;
        let patch = patch.as_object().ok_or("Panel binding must be an object")?;
        if patch.keys().any(|key| !matches!(key.as_str(), "signal" | "window")) {
            return Err("Panel binding contains an unknown field".into());
        }
        if old.frozen.is_some() { return Err("An Edit shows its frozen snapshot; Refresh snapshot takes the panel's newest data".into()); }
        let signal = match patch.get("signal") {
            Some(value) => value.as_str().ok_or("Panel signal must be text")?,
            None => old.signal.as_ref(),
        };
        if signal.trim() != signal { return Err("Panel signal must be canonical text".into()); }
        let window = match patch.get("window") {
            Some(value) => value.as_u64().filter(|width| *width > 0 && *width <= i64::MAX as u64)
                .and_then(|width| usize::try_from(width).ok()).ok_or("Panel window must be a positive int64 integer")?,
            None => old.window,
        };
        if signal == old.signal.as_ref() && window == old.window { return Ok(false); }
        let epoch = old.epoch.checked_add(1).ok_or("Panel binding epoch exhausted")?;
        let source = runtime.signals().into_iter().find(|source| source.signal.as_ref() == signal);
        let lease = lease_for(runtime, source.as_ref(), window)?;
        let replacement = Binding { signal: signal.into(), window,
            generation: source.map(|source| source.generation), lease, epoch,
            initial_kind: old.initial_kind.clone(), reason: None, authority: Arc::new(Mutex::new(true)),intent_authority:Arc::new(Mutex::new(true)), frozen: None, layer: None };
        self.bindings.insert(panel.into(), replacement);
        Ok(true)
    }

    pub fn remove(&mut self, panel: &str) -> bool { self.bindings.remove(panel).is_some() }
    pub fn is_empty(&self) -> bool { self.bindings.is_empty() }
    pub fn epoch(&self, panel: &str) -> Option<u64> { self.bindings.get(panel).map(|binding| binding.epoch) }
    /// The point layer a data copy's preview draws (v2 `_manual_plot_input`'s ImageFrame overlay).
    pub fn set_layer(&mut self, panel: &str, layer: Option<Arc<ImagePointOverlay>>) {
        if let Some(binding) = self.bindings.get_mut(panel) { binding.layer = layer; }
    }

    pub fn accepts_source(&self,panel:&str,key:&PublicationKey)->bool {
        self.bindings.get(panel).is_some_and(|binding|binding.signal==key.signal
            && binding.generation.as_ref().is_some_and(|generation|generation.id()==key.generation))
    }

    /// Dispatch snapshots share each binding's one revocable owner, not epochs.
    pub fn authorities(&self) -> BTreeMap<String, BindingAuthority> {
        self.bindings.iter().map(|(panel, binding)| (panel.clone(), binding.authority.clone())).collect()
    }

    /// Scientific intent survives source restart. Rebinding/removing revokes
    /// it; a source generation change only invalidates previously shown fronts.
    pub fn intent_authorities(&self) -> BTreeMap<String, BindingAuthority> {
        self.bindings.iter().map(|(panel,binding)|(panel.clone(),binding.intent_authority.clone())).collect()
    }

    pub fn sync_generations(&mut self, runtime: &Runtime) -> bool {
        let sources = runtime.signals();
        let mut changed = false;
        for binding in self.bindings.values_mut().filter(|binding| binding.frozen.is_none()) {
            let source = sources.iter().find(|source| source.signal == binding.signal);
            if binding.generation.as_ref().map(Generation::id) == source.map(|source| source.generation.id()) { continue; }
            binding.revoke();
            binding.authority = Arc::new(Mutex::new(true));
            let lease = lease_for(runtime, source, binding.window);
            binding.generation = source.map(|source| source.generation.clone());
            binding.epoch += 1;
            (binding.lease, binding.reason) = match lease {
                Ok(lease) => (lease, None),
                Err(error) => (None, Some(error)),
            };
            changed = true;
        }
        changed
    }

    /// Authoring may already name a new kind while the last complete front
    /// still shows the old kind: input intent follows the authored `targets`.
    /// A panel's Overlay (`overlays`, the board's) is read in the same front,
    /// so the picture is the one its judgement is of (v2 `_overlay_companions`):
    /// only while that judgement is being made, or was made of this run.
    pub fn capture(&self, runtime: &Runtime, targets: &Value, accepted: &Value, overlays: &BTreeMap<String, String>) -> Vec<PanelInput> {
        let sources = runtime.signals();
        let mut inputs = Vec::with_capacity(self.bindings.len());
        let mut requests = Vec::with_capacity(self.bindings.len());
        let mut positions = Vec::with_capacity(self.bindings.len());
        for (panel, binding) in &self.bindings {
            let row = accepted["panels"].as_array().and_then(|panels| panels.iter()
                .find(|value| value["id"].as_str() == Some(panel.as_str())));
            let kind = targets.get(panel).or_else(||row.map(|row|&row["target"]))
                .and_then(|target|target["semantic"]["kind"].as_str())
                .or_else(||row.and_then(|row|row["kind"].as_str())).unwrap_or(&binding.initial_kind);
            if let Some(frozen) = &binding.frozen {
                inputs.push(PanelInput { panel: panel.clone(), kind: kind.into(), window: binding.window, signal: binding.signal.clone(),
                    epoch: binding.epoch, read: Some(frozen.read.clone()), overlay: frozen.overlay.clone(),
                    layer: frozen.review.as_ref().map(|review| review.layer.clone()), reason: None, waiting: false });
                continue;
            }
            // A blank panel (v2: added before its signal is chosen) waits, saying what to do next.
            let blank = binding.signal.is_empty() && binding.reason.is_none();
            let reason = binding.reason.clone().or_else(|| binding.generation.is_none().then(||
                if blank { "Choose a signal in Setting".into() }
                else { format!("Signal is not available: {}", binding.signal) }));
            if reason.is_none() {
                let records = accepted["plot_kinds"].as_array().is_some_and(|kinds|kinds.iter()
                    .any(|item|item["id"]==kind && item["record_input"]==true));
                // A complete-scope snapshot already contains its records. The
                // binding window remains scientific intent, not another outer
                // history. Missing/stale sources retain the existing read path
                // and Runtime supplies its authoritative availability error.
                let history_capable = sources.iter().find(|source|
                    source.signal == binding.signal && binding.generation.as_ref()
                        .is_some_and(|generation| generation.id() == source.generation.id()))
                    .is_none_or(|source| source.history_capable);
                let width = (history_capable && (binding.window > 1 || records)).then_some(binding.window);
                positions.push((inputs.len(), false));
                requests.push((binding.generation.as_ref().unwrap().clone(), binding.signal.clone(), width));
                let image = sources.iter().find(|source| source.signal == binding.signal);
                let judgement = overlays.get(panel).filter(|_| width.is_none()).and_then(|overlay| sources.iter()
                    .find(|source| *source.signal == **overlay && source.published && (!source.sealed || image.is_some_and(|image| image.sealed))));
                if let Some(judgement) = judgement {
                    positions.push((inputs.len(), true));
                    requests.push((judgement.generation.clone(), judgement.signal.clone(), None));
                }
            }
            inputs.push(PanelInput { panel: panel.clone(), kind: kind.into(), window: binding.window, signal: binding.signal.clone(),
                epoch: binding.epoch, read: None, overlay: None, layer: binding.layer.clone(), reason, waiting: blank });
        }
        // One Runtime lock captures all panels, including different windows on one source.
        let answers = runtime.read_coherent_views(&requests, &mut self.front.lock().unwrap_or_else(|error| error.into_inner()));
        for ((index, overlay), answer) in positions.into_iter().zip(answers) {
            match answer {
                Ok(read) if overlay => inputs[index].overlay = Some(read),
                Ok(read) => inputs[index].read = Some(read),
                // A judgement not of this picture is not drawn on it.
                Err(_) if overlay => {}
                Err(RuntimeError::NoData) => { inputs[index].reason = Some("Waiting for data".into()); inputs[index].waiting = true; }
                Err(error) => inputs[index].reason = Some(error.to_string()),
            }
        }
        // A judgement is drawn only on the shot it judged: one that is not of
        // the picture (routed from elsewhere, or left from an older shot) is not drawn.
        for input in &mut inputs {
            if input.overlay.as_ref().is_some_and(|overlay| input.read.as_ref().is_none_or(|read| overlay.shot_roots() != read.shot_roots())) {
                input.overlay = None;
            }
        }
        inputs
    }

    /// The bindings of the shown panels, and the signals `offered` keeps.
    pub fn decorate(&self, state: &mut Value, runtime: &Runtime, offered: impl Fn(&str) -> bool) {
        let sources = runtime.signals();
        if let Some(panels) = state["panels"].as_array_mut() {
            for panel in panels {
                if let Some(binding) = panel["id"].as_str().and_then(|id| self.bindings.get(id)) {
                    panel["binding"] = json!({"signal": binding.signal.as_ref(), "window": binding.window});
                    // A review says where its candidates stand on the picture: canvas
                    // fractions through the drawn image axes, for its picking.
                    if let Some(review) = binding.frozen.as_ref().and_then(|frozen| frozen.review.as_ref()) {
                        let axis = panel["axes"].as_array().and_then(|axes| axes.iter().find(|axis| axis["role"] == "image"))
                            .and_then(|axis| zlc_plot::document::axis_from_wire(axis).ok())
                            .filter(|axis| axis.canonical_x_scale.is_none() && axis.canonical_y_scale.is_none());
                        let points: Vec<Value> = review.points.iter().map(|(id, label, x, y)| {
                            let at = axis.as_ref().map(|axis| axis.display_to_normalized(*x, *y));
                            json!([id, label, x, y, at.map(|at| at.0), at.map(|at| at.1)])
                        }).collect();
                        panel["review"] = json!({"node": binding.frozen.as_ref().map(|frozen| frozen.of.clone()), "points": points});
                        continue;
                    }
                    // An Edit says what it holds and whether its panel has moved on since.
                    if let Some(frozen) = &binding.frozen {
                        let key = frozen.read.key();
                        let newest = sources.iter().find(|source| source.signal == binding.signal)
                            .and_then(|source| runtime.read_views(&[(source.generation.clone(), source.signal.clone(), None)]).remove(0).ok());
                        // v2 _snapshot_text: what the held picture is to Save, in v2's words.
                        let moved = self.bindings.get(&frozen.of).is_some_and(|live| live.signal != binding.signal);
                        let status = if moved { "Stale: this panel now shows another signal; refresh before saving" }
                            else if newest.as_ref().is_some_and(|read| read.key().generation != key.generation) { "Frozen from an earlier run; Refresh to update; Save keeps this snapshot" }
                            else if newest.as_ref().is_some_and(|read| read.key() != key) { "Live data advanced; Refresh to update; Save keeps this snapshot" }
                            else { "frozen and ready to save" };
                        panel["edit"] = json!({"panel": frozen.of, "at_ms": frozen.at_ms as u64, "revision": key.revision,
                            "advanced": newest.is_some_and(|read| read.key() != key), "shape": frozen.read.schema().physical_shape(), "status": status});
                    }
                }
            }
        }
        state["signals"] = Value::Array(runtime.signals().into_iter().filter(|source| offered(&source.signal)).map(|source| json!({
            "id":source.signal.as_ref(), "label":source.signal.as_ref(), "available":source.published,
            // A viewer's Signal tree files it under its owner and reads its shape (v2 signal_groups).
            "owner":source.generation.owner(), "shape":crate::workbench_host::shape_text(&source.schema),
            "reason":if source.published { "" } else { "Waiting for data" },
            "finite":source.finite, "sealed":source.sealed,
            "history_capable":source.history_capable,
            "history_capacity":source.history_capacity,"retained_events":source.retained_events,
        })).collect());
    }
}

fn lease_for(runtime: &Runtime, source: Option<&SignalDescription>, window: usize) -> Result<Option<HistoryLease>, String> {
    match source {
        Some(source) if source.history_capable && window > 1 => runtime.retain_history(&source.generation, &source.signal, window)
            .map(Some).map_err(|error| error.to_string()),
        _ => Ok(None),
    }
}

/// The Figure Viewer whose document a panel is of (`figure-<viewer>-<k>`, or its Edit), if any.
pub fn viewer_of(panel: &str) -> Option<u32> {
    let panel = panel.strip_prefix(EDIT_PREFIX).unwrap_or(panel);
    let (viewer, rest) = panel.strip_prefix(VIEWER_PREFIX)?.strip_prefix('-')?.split_once('-')?;
    (!rest.is_empty()).then(|| viewer.parse().ok()).flatten()
}

/// What a viewer's card calls its panel: each v2 FigureViewer numbers its own
/// `panel-<k>`, whichever Figure it shows.
pub fn viewer_caption(panel: &str) -> Option<String> {
    if panel.starts_with(EDIT_PREFIX) { return None; }
    viewer_of(panel)?;
    let (_, rest) = panel.strip_prefix(VIEWER_PREFIX)?.strip_prefix('-')?.split_once('-')?;
    Some(format!("panel-{rest}"))
}
