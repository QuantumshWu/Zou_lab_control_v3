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
    /// whenever its signal is in the Runtime, a member of the picture's group.
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
                // The Overlay is a member of the picture's group whatever its run's state: the
                // Runtime keeps the last shot both have until its judgement of a newer one comes.
                let judgement = overlays.get(panel).filter(|_| width.is_none() && kind != "point_map")
                    .and_then(|overlay| sources.iter().find(|source| *source.signal == **overlay));
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
                            .and_then(|source| runtime.read_coherent_views(&[(source.generation.clone(), source.signal.clone(), None)], &mut Default::default()).remove(0).ok());
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

#[cfg(test)]
pub(crate) mod tests {
    //! The same-shot group on a board, driven as the desktop drives it: Runtime runs
    //! begin, commit, seal and leave; each turn syncs the bindings, captures the board
    //! (`Panels::capture`) and builds the image's point layer (`LiveOverlays::layer`).
    use super::*;
    use crate::overlay::LiveOverlays;
    use zlc_core::dataset::{DataBlock, Validity};
    use zlc_core::runtime::{Bundle, Declaration, GenerationId, Input, Publication, Storage};
    use zlc_core::schema::{Axis, Coordinates, DatasetSchema, Domain, DomainRole, ValidityContract, ValueSchema};
    use zlc_core::{DType, OwnedBuffer};

    const SITES: usize = 2;
    fn axis(id: &str, role: &str, len: usize, frame: Option<&str>) -> Arc<Axis> {
        Axis::new(id.into(), id.into(), role.into(), Coordinates::Index { origin: 0, len }, None, frame.map(Into::into), None, None).unwrap()
    }
    pub(crate) fn image_schema() -> Arc<DatasetSchema> {
        DatasetSchema::new(Domain::dense(vec![1], vec![axis("repeat", "repeat", 1, None)]).unwrap(),
            Domain::dense(vec![3], vec![axis("frame", "readout-event", 3, None)]).unwrap(),
            Domain::dense(vec![2, 3], vec![axis("y", "spatial-y", 2, Some("camera")), axis("x", "spatial-x", 3, Some("camera"))]).unwrap(),
            ValueSchema::new(DType::F64, None, None, ValidityContract::Value).unwrap()).unwrap()
    }
    pub(crate) fn status_schema() -> Arc<DatasetSchema> {
        let image = image_schema();
        DatasetSchema::new(image.domain(DomainRole::Repeat).clone(), image.domain(DomainRole::Point).clone(),
            Domain::dense(vec![SITES], vec![axis("site", "site", SITES, None)]).unwrap(),
            ValueSchema::new(DType::Bool, None, None, ValidityContract::Value).unwrap()).unwrap()
    }
    /// The run record of an Occupancy run: where its sites are on the camera's image.
    pub(crate) fn geometry() -> Value {
        json!({"image_point_overlay_geometry": {"coordinate_frame": "camera", "x_axis_id": "x", "y_axis_id": "y", "point_ids": ["a", "b"],
            "coordinates_xy": [[0.0, 0.0], [1.0, 1.0]], "labels": ["a", "b"], "status_axis_id": "site", "status_coordinates": [0, 1]}})
    }
    pub(crate) fn event(schema: &Arc<DatasetSchema>, signal: &str, ordinal: u64, parents: Vec<Arc<Publication>>) -> Bundle {
        let data: zlc_core::AnyBuffer = if schema.value().dtype() == DType::Bool { OwnedBuffer::from_vec(vec![true; schema.value_len()]).into() }
            else { OwnedBuffer::from_vec(vec![ordinal as f64; schema.value_len()]).into() };
        Bundle { inputs: vec![Input::Event { signal: signal.into(), ordinal, origin: None, shot_time: None,
            data: DataBlock::new(schema.clone(), data, Validity::Uniform(true), None).unwrap() }], metadata: json!({}), parents }
    }
    pub(crate) fn latest(signal: &str, schema: &Arc<DatasetSchema>) -> Vec<Declaration> { vec![Declaration { signal: signal.into(), schema: schema.clone(), storage: Storage::Latest }] }

    /// What the board's turns showed of `panel-1`, the image with its Overlay.
    #[derive(Debug, Default, Clone, Copy, PartialEq)]
    pub(crate) struct Counts {
        /// Turns that drew the image.
        pub shown: usize,
        /// Turns whose image carried the rings of its judgement of that very shot.
        pub paired: usize,
        /// A new shot of the image drawn without its judgement while the Overlay is a member.
        pub unpaired: usize,
        /// The rings left a shot they had been drawn on (rings vanish, picture unchanged).
        pub dropped: usize,
        /// The image went back to an older shot (or an older run) than it showed.
        pub regressions: usize,
        /// The image went blank (nothing to draw) after it had shown a shot.
        pub blanked: usize,
        /// Turns after both runs ended that showed something else than the turn before.
        pub changed_after_stop: usize,
        /// Turns after the Overlay left (removed, its producer gone) that did not show the
        /// camera's newest shot: a picture held by a member that left.
        pub held_without_member: usize,
        /// `panel-2` (a member downstream) at another shot than the image.
        pub apart: usize,
    }

    pub(crate) struct Board {
        pub runtime: Runtime,
        panels: Panels,
        pub overlays: BTreeMap<String, String>,
        layers: LiveOverlays,
        targets: Value,
        runs: Vec<GenerationId>,
        last: Option<((usize, u64), bool)>,
        pub stopped: bool,
        /// The Overlay has left: the picture must be the camera's newest.
        pub alone: bool,
        pub counts: Counts,
        pub log: Vec<String>,
    }

    impl Board {
        /// `panel-1` an image of `frames` with Overlay `occupied`; `panel-2`, if given, (signal, kind).
        pub(crate) fn new(second: Option<(&str, &str)>) -> Self {
            let runtime = Runtime::new();
            let mut panels = Panels::new();
            panels.add(&runtime, "frames", "image").unwrap();
            let mut targets = json!({"panel-1": {"semantic": {"kind": "image"}}});
            if let Some((signal, kind)) = second {
                panels.add(&runtime, signal, kind).unwrap();
                targets["panel-2"] = json!({"semantic": {"kind": kind}});
            }
            Self { runtime, panels, overlays: [("panel-1".to_owned(), "occupied".to_owned())].into(), layers: LiveOverlays::default(),
                targets, runs: Vec::new(), last: None, stopped: false, alone: false, counts: Counts::default(), log: Vec::new() }
        }

        fn run_index(&mut self, generation: GenerationId) -> usize {
            self.runs.iter().position(|run| *run == generation).unwrap_or_else(|| { self.runs.push(generation); self.runs.len() - 1 })
        }

        /// One board turn, counted.
        pub(crate) fn turn(&mut self, label: &str) {
            self.panels.sync_generations(&self.runtime);
            let inputs = self.panels.capture(&self.runtime, &self.targets, &json!({"panels": [], "plot_kinds": []}), &self.overlays);
            let sources = self.runtime.signals();
            let member = self.overlays.get("panel-1").is_some_and(|overlay| sources.iter().any(|source| *source.signal == **overlay));
            let image = inputs.iter().find(|input| input.panel == "panel-1").unwrap();
            let second = inputs.iter().find(|input| input.panel == "panel-2");
            let Some(read) = image.read.clone() else {
                self.log.push(format!("{label}: none ({:?})", image.reason));
                if self.stopped && self.last.is_some() { self.counts.changed_after_stop += 1; }
                if self.last.is_some() { self.counts.blanked += 1; }
                self.last = None;
                return;
            };
            let overlay = image.overlay.clone();
            let layer = self.layers.layer("panel-1", &read, overlay.as_ref()).unwrap();
            let paired = overlay.as_ref().is_some_and(|status| status.shot_roots() == read.shot_roots()) && layer.is_some();
            let shot = (self.run_index(read.key().generation), read.key().ordinal.unwrap());
            self.counts.shown += 1;
            if paired { self.counts.paired += 1; }
            match self.last {
                Some((before, rings)) => {
                    if shot < before { self.counts.regressions += 1; }
                    if shot != before && member && !paired { self.counts.unpaired += 1; }
                    if shot == before && rings && !paired { self.counts.dropped += 1; }
                    if self.stopped && (shot != before || rings != paired) { self.counts.changed_after_stop += 1; }
                }
                None => if member && !paired { self.counts.unpaired += 1; },
            }
            if self.alone {
                let newest = sources.iter().find(|source| &*source.signal == "frames")
                    .and_then(|source| self.runtime.read_latest(&source.generation, "frames").ok());
                if newest.is_some_and(|newest| *newest.key() != *read.key()) { self.counts.held_without_member += 1; }
            }
            if let Some(read2) = second.and_then(|input| input.read.as_ref()) {
                if read2.shot_roots() != read.shot_roots() { self.counts.apart += 1; }
            }
            self.log.push(format!("{label}: run{} #{} {} panel-2={:?}", shot.0, shot.1, if paired { "rings" } else { "no rings" },
                second.map(|input| input.read.as_ref().map(|read| read.key().ordinal))));
            self.last = Some((shot, paired));
        }
    }

    fn frame(board: &mut Board, camera: &Generation, ordinal: u64) -> Arc<Publication> {
        let shot = board.runtime.commit(camera, &event(&image_schema(), "frames", ordinal, vec![])).unwrap().publications[0].clone();
        board.turn(&format!("cam{ordinal}"));
        shot
    }
    fn judge(board: &mut Board, occupancy: &Generation, shot: &Arc<Publication>) -> Arc<Publication> {
        let ordinal = shot.key().ordinal.unwrap();
        let judged = board.runtime.commit(occupancy, &event(&status_schema(), "occupied", ordinal, vec![shot.clone()])).unwrap().publications[0].clone();
        board.turn(&format!("occ{ordinal}"));
        judged
    }
    fn camera(board: &Board) -> Generation { board.runtime.begin("camera", latest("frames", &image_schema()), json!({})).unwrap() }
    fn occupancy(board: &Board, camera: &Generation) -> Generation {
        board.runtime.begin_consuming("occupancy", latest("occupied", &status_schema()), geometry(), "frames".into(), Some(camera.id())).unwrap()
    }

    /// `shots` camera shots from `from`, each judged by Occupancy `lag` shots later.
    fn acquire(board: &mut Board, camera: &Generation, occupancy: &Generation, from: u64, shots: u64, lag: usize) -> Vec<Arc<Publication>> {
        let mut taken = Vec::new();
        for ordinal in from..from + shots {
            taken.push(frame(board, camera, ordinal));
            if taken.len() > lag { let shot = taken[taken.len() - 1 - lag].clone(); judge(board, occupancy, &shot); }
        }
        taken
    }
    /// Both runs end, the camera's or the processor's first; the judgements still owed come between.
    fn stop(board: &mut Board, camera: &Generation, occupancy: &Generation, owed: &[Arc<Publication>], camera_first: bool) {
        if camera_first {
            board.runtime.seal(camera, json!({})).unwrap();
            board.turn("camera sealed");
            for shot in owed { judge(board, occupancy, shot); }
            board.runtime.seal(occupancy, json!({"status": "completed"})).unwrap();
        } else {
            for shot in owed { judge(board, occupancy, shot); }
            board.runtime.seal(occupancy, json!({"status": "completed"})).unwrap();
            board.turn("occupancy sealed");
            board.runtime.seal(camera, json!({})).unwrap();
        }
        board.stopped = true;
        for turn in 0..4 { board.turn(&format!("stopped {turn}")); }
    }

    /// Every sequence the same-shot group must come through, with what its board showed.
    pub(crate) fn sequences() -> Vec<(&'static str, Counts, Vec<String>)> {
        let mut results = Vec::new();
        let mut done = |name: &'static str, board: Board| results.push((name, board.counts, board.log));

        // Start: the camera runs, Occupancy begins late and its first judgement is slow.
        let mut board = Board::new(None);
        let cam = camera(&board);
        let mut shots: Vec<_> = (0..3).map(|ordinal| frame(&mut board, &cam, ordinal)).collect();
        let occ = occupancy(&board, &cam);
        board.turn("occupancy began");
        for ordinal in 3..5 { shots.push(frame(&mut board, &cam, ordinal)); }
        for shot in &shots { judge(&mut board, &occ, shot); }
        acquire(&mut board, &cam, &occ, 5, 4, 1);
        done("start", board);

        // Steady running, the judgement one shot behind.
        let mut board = Board::new(None);
        let cam = camera(&board);
        let occ = occupancy(&board, &cam);
        acquire(&mut board, &cam, &occ, 0, 12, 1);
        done("steady", board);

        // Stop, the camera's run ending first, or the processor's.
        for (name, camera_first) in [("stop: camera first", true), ("stop: processor first", false)] {
            let mut board = Board::new(None);
            let cam = camera(&board);
            let occ = occupancy(&board, &cam);
            let shots = acquire(&mut board, &cam, &occ, 0, 6, 1);
            stop(&mut board, &cam, &occ, &shots[5..], camera_first);
            done(name, board);
        }

        // Restart: a stopped camera starts again; Occupancy's new run begins late, slow at first.
        let mut board = Board::new(None);
        let cam = camera(&board);
        let occ = occupancy(&board, &cam);
        let shots = acquire(&mut board, &cam, &occ, 0, 4, 1);
        stop(&mut board, &cam, &occ, &shots[3..], true);
        board.stopped = false;
        let cam = camera(&board);
        board.turn("camera began");
        let mut shots: Vec<_> = (0..2).map(|ordinal| frame(&mut board, &cam, ordinal)).collect();
        let occ = occupancy(&board, &cam);
        board.turn("occupancy began");
        shots.push(frame(&mut board, &cam, 2));
        for shot in &shots { judge(&mut board, &occ, shot); }
        acquire(&mut board, &cam, &occ, 3, 4, 1);
        done("restart", board);

        // An ROI of the camera (panel-2 shows it) across the camera's restart: the region
        // answers at display cadence, every other shot, and begins again on the new run.
        let roi_signal = "@logic/panel-1/roi_frame";
        let mut board = Board::new(Some((roi_signal, "image")));
        board.overlays.clear();
        let roi_of = |board: &Board, shot: &Arc<Publication>| board.runtime.begin_derived("panel-1/roi",
            latest(roi_signal, &image_schema()), json!({}), shot.key(), true).unwrap();
        let answer = |board: &mut Board, roi: &Generation, shot: &Arc<Publication>| {
            board.runtime.commit(roi, &event(&image_schema(), roi_signal, shot.key().ordinal.unwrap(), vec![shot.clone()])).unwrap();
            board.turn(&format!("roi{}", shot.key().ordinal.unwrap()));
        };
        let cam = camera(&board);
        let first = frame(&mut board, &cam, 0);
        let roi = roi_of(&board, &first);
        answer(&mut board, &roi, &first);
        for ordinal in 1..7 { let shot = frame(&mut board, &cam, ordinal); if ordinal % 2 == 0 { answer(&mut board, &roi, &shot); } }
        board.runtime.seal(&cam, json!({})).unwrap();
        board.runtime.seal(&roi, json!({})).unwrap();
        board.turn("stopped");
        let cam = camera(&board);
        board.turn("camera began");
        let first = frame(&mut board, &cam, 0);
        frame(&mut board, &cam, 1);
        let roi = roi_of(&board, &first);
        board.turn("roi began");
        answer(&mut board, &roi, &first);
        for ordinal in 2..7 { let shot = frame(&mut board, &cam, ordinal); if ordinal % 2 == 0 { answer(&mut board, &roi, &shot); } }
        done("roi across a restart", board);

        // A chain: Frame Survival of Occupancy's judgement (panel-2, its rolling history),
        // through two rounds of all three runs.
        let mut board = Board::new(Some(("survival", "rolling")));
        for round in 0..2 {
            let cam = camera(&board);
            board.turn("camera began");
            let first = frame(&mut board, &cam, 0);
            let occ = occupancy(&board, &cam);
            board.turn("occupancy began");
            let mut judged = vec![judge(&mut board, &occ, &first)];
            let survival = board.runtime.begin_consuming("survival", latest("survival", &status_schema()), json!({}), "occupied".into(), Some(occ.id())).unwrap();
            board.turn("survival began");
            board.runtime.commit(&survival, &event(&status_schema(), "survival", 0, vec![judged[0].clone()])).unwrap();
            board.turn("surv0");
            for ordinal in 1..6u64 {
                let shot = frame(&mut board, &cam, ordinal);
                judged.push(judge(&mut board, &occ, &shot));
                let at = ordinal as usize - 1;
                board.runtime.commit(&survival, &event(&status_schema(), "survival", at as u64, vec![judged[at].clone()])).unwrap();
                board.turn(&format!("surv{at}"));
            }
            board.runtime.seal(&cam, json!({})).unwrap();
            board.turn("camera sealed");
            board.runtime.seal(&occ, json!({"status": "completed"})).unwrap();
            board.turn("occupancy sealed");
            board.runtime.commit(&survival, &event(&status_schema(), "survival", 5, vec![judged[5].clone()])).unwrap();
            board.turn("surv5");
            board.runtime.seal(&survival, json!({"status": "completed"})).unwrap();
            board.stopped = true;
            for turn in 0..3 { board.turn(&format!("round {round} stopped {turn}")); }
            board.stopped = false;
        }
        done("chain occupancy -> frame survival", board);

        // The Overlay removed while both run: the picture goes on at once.
        let mut board = Board::new(None);
        let cam = camera(&board);
        let occ = occupancy(&board, &cam);
        acquire(&mut board, &cam, &occ, 0, 4, 2);
        board.overlays.clear();
        board.alone = true;
        board.turn("overlay removed");
        acquire(&mut board, &cam, &occ, 4, 4, 2);
        done("overlay removed", board);

        // The processor fails while the camera runs: its host withdraws it (v2), and the
        // picture goes on without it.
        let mut board = Board::new(None);
        let cam = camera(&board);
        let occ = occupancy(&board, &cam);
        acquire(&mut board, &cam, &occ, 0, 4, 2);
        board.runtime.retire(&occ).unwrap();
        board.alone = true;
        board.turn("occupancy failed");
        for ordinal in 4..8 { frame(&mut board, &cam, ordinal); }
        done("processor fails", board);
        results
    }

    #[test]
    fn a_picture_and_its_judgement_are_one_shot_through_every_run_transition() {
        let results = sequences();
        let mut failures = Vec::new();
        for (name, counts, log) in &results {
            eprintln!("{name}: {counts:?}");
            let bad = counts.unpaired + counts.dropped + counts.regressions + counts.blanked + counts.changed_after_stop
                + counts.held_without_member + counts.apart;
            if bad != 0 { failures.push(format!("{name}: {counts:?}\n  {}", log.join("\n  "))); }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
        for (name, counts, _) in &results {
            if *name != "roi across a restart" { assert!(counts.paired > 0, "{name} never showed its rings"); }
        }
    }
}
