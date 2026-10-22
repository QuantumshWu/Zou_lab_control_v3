//! One panel's part of the document: what a worker owns while it resolves,
//! prepares and presents that panel, apart from every other panel.

use std::sync::{Arc, Mutex};

use serde_json::{json, Value as Json};

use crate::configuration::data_contract::PlotSchema;
use crate::configuration::error::{PlotError, Result};
use crate::configuration::kinds::PlotKind;
use crate::configuration::resolve::{resolve_configuration, ConfigurationUpdates, InteractionUpdates, OverlayHandle, ResolveRequest,
                                    ResolvedConfiguration};
use crate::configuration::value::{Value, ValueMap};
use crate::interaction::gesture::ClickHistory;
use crate::projection::Source;
use crate::session::core::same_overlay;
use crate::session::{DataPreparation, ImagePointOverlay, PlotInput, PlotSession, PreparedPlot, RgbaFront, SessionServices, SnapshotRef};

use super::edit::{declared_start, Edit, Queue};
use super::{axis_wire, fit_parameter_data, gesture_updates, json_key_error, lock, pointer_event, status_text, Binding, Configured,
            DocumentRead, Effect, FigureDocument, FitPublisher, PanelSource, PanelStatus, SessionHandle, Staged, StagedScene};

/// What every panel reads and none owns.
pub(crate) struct Shared<'a> { pub services: &'a SessionServices, pub dpr: f64 }

/// One panel's entries of the document's maps.
pub(crate) struct Panel<R> {
    pub key: String,
    pub session: Option<SessionHandle>,
    pub input: Option<R>,
    pub status: Option<PanelStatus>,
    pub source: Option<PanelSource>,
    pub configured: Option<Configured>,
    pub staged: Option<Staged<R>>,
    /// `key in initial_overlays`, and its overlay.
    pub initial_overlay: Option<Option<Arc<ImagePointOverlay>>>,
    /// The point layer the read being resolved carries (a live image's).
    pub read_overlay: Option<Option<Arc<ImagePointOverlay>>>,
    pub fit_output: bool,
    pub manual_fits: Vec<Effect<R>>,
}

/// What resolving a panel's edits on a read gives: the candidate, whether
/// it changed, the packet its session already presents it with (`None`:
/// present it later) and the command's own edit's failure when queued
/// edits were kept.
pub(crate) struct Resolution {
    pub configuration: Arc<ResolvedConfiguration>,
    pub changed: bool,
    pub shown: Option<PreparedPlot>,
    pub failure: Option<PlotError>,
}

/// A configuration a session presents on a source, with its packet.
type Showing = (Arc<ResolvedConfiguration>, Arc<Source>, PreparedPlot);

fn presents(showing: &Option<Showing>, configuration: &Arc<ResolvedConfiguration>, source: &Arc<Source>) -> bool {
    showing.as_ref().is_some_and(|(shown, on, _)| Arc::ptr_eq(shown, configuration) && Arc::ptr_eq(on, source))
}

/// A session that failed to present restores its presentation to the
/// command's start; reopen it for the edits after.
fn reopen(session: &mut PlotSession) {
    session.finish_presentation(false, None);
    let _ = session.begin_presentation();
}

impl<R: DocumentRead> FigureDocument<R> {
    /// The panel's entries, cloned out of the maps.
    pub(crate) fn panel(&self, key: &str) -> Panel<R> {
        let key = key.to_owned();
        Panel {
            session: self.sessions.get(&key).cloned(), input: self.inputs.get(&key).cloned(), status: self.panel_status.get(&key).cloned(),
            source: self.panel_sources.get(&key).cloned(), configured: self.configurations.get(&key).cloned(), staged: self.staged.get(&key).cloned(),
            initial_overlay: self.initial_overlays.get(&key).cloned(), read_overlay: None, fit_output: false, manual_fits: Vec::new(), key,
        }
    }

    /// Write back what preparing the panel set.
    pub(crate) fn store_prepared(&mut self, panel: &mut Panel<R>) {
        if let Some(configured) = panel.configured.clone() { self.configurations.insert(panel.key.clone(), configured); }
        if let Some(staged) = panel.staged.clone() { self.staged.insert(panel.key.clone(), staged); }
        if panel.fit_output { self.fit_outputs.insert(panel.key.clone()); }
        self.manual_fits.append(&mut panel.manual_fits);
    }

    /// Write back what presenting the panel set (a session that clamped its
    /// grid also changed the configuration).
    pub(crate) fn store_presented(&mut self, panel: Panel<R>) {
        if let Some(configured) = panel.configured { self.configurations.insert(panel.key.clone(), configured); }
        if let Some(session) = panel.session { self.sessions.insert(panel.key.clone(), session); }
        if let Some(input) = panel.input { self.inputs.insert(panel.key.clone(), input); }
        if let Some(source) = panel.source { self.panel_sources.insert(panel.key.clone(), source); }
        if let Some(status) = panel.status { self.panel_status.insert(panel.key, status); }
    }
}

impl<R: DocumentRead> Panel<R> {
    pub(crate) fn status_mut(&mut self) -> &mut PanelStatus { self.status.get_or_insert_with(PanelStatus::default) }

    /// `_defer`: queue `edit` until the panel has a read to resolve it on.
    pub(crate) fn defer(&mut self, binding: &Binding, target: Option<&Json>, edit: Option<Edit>) -> Result<()> {
        let (configured, key) = (self.configured.is_some(), self.key.clone());
        super::edit::defer(&key, self.status_mut(), configured, binding, target, edit)
    }

    /// `_resolve_configuration`: the command's own edit on `read`, after the
    /// edits the panel queued; without queued edits it is resolved from the
    /// configuration, or from the start of a panel that has none, and
    /// presented later.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn resolve(&mut self, shared: &Shared<'_>, binding: &Binding, read: &R, target: Option<&Json>, command: &Json,
                          interaction_updates: Option<InteractionUpdates>, at: f64) -> Result<Resolution> {
        self.read_overlay = if self.initial_overlay.is_some() { None } else { read.image_overlay() };
        let queue = self.status_mut().queue.take();
        let pointer =if queue.is_some() && command.get("action").and_then(Json::as_str) == Some("pointer") { vec![(command.clone(), at)] } else { Vec::new() };
        let own = Edit::of(&self.key, binding, command, interaction_updates, pointer)?;
        if let Some(queue) = queue { return self.apply_queue(shared, binding, read, target, queue, own); }
        let schema = PlotSchema::new(read.source().schema.clone());
        let same_source = self.initial_overlay.is_some() || self.source.as_ref().is_some_and(|source| source.signal == binding.signal);
        let candidate = match self.configured.clone() {
            None => {
                let (start, mut kept) = declared_start(Edit::start(&self.key, binding, target)?, vec![own]);
                let own = kept.pop().expect("own edit");
                let candidate = self.resolve_edit(shared, Some(schema.clone()), None, &start, same_source)?;
                if own.changes() { self.resolve_edit(shared, Some(schema), Some(&candidate), &own, true)? } else { candidate }
            }
            Some(previous) if !own.changes() && previous.signal == binding.signal && previous.window == binding.window
                && previous.configuration.same_source == same_source
                && self.read_overlay.as_ref().is_none_or(|overlay| same_overlay(&previous.configuration.image_overlay,
                    &overlay.clone().map(|overlay| overlay as OverlayHandle)))
                && previous.configuration.schema.as_deref().is_some_and(|held| *held.dataset() == *schema.dataset()) => {
                self.status_mut().authoring_pending = Some(false);
                return Ok(Resolution { configuration: previous.configuration, changed: false, shown: None, failure: None });
            }
            Some(previous) => self.resolve_edit(shared, Some(schema), Some(&previous.configuration), &own, same_source)?,
        };
        self.configured = Some(Configured { signal: binding.signal.clone(), window: binding.window, configuration: candidate.clone() });
        let status = self.status_mut();
        status.error = None;
        status.edit_error = None;
        status.authoring_pending = Some(false);
        status.message = (candidate.has_numerical_work() && candidate.science_changed).then(|| "Preparing numerical results".to_owned());
        Ok(Resolution { configuration: candidate, changed: true, shown: None, failure: None })
    }

    /// `_apply_queue`: the edits the panel queued while it had no read, then
    /// `own`, applied to its session at their turn as typed edits are once
    /// data is present: each resolved from the configuration presented then
    /// (whose picture gives fixed limits and a clamped facet focus), prepared
    /// and presented on the read of the source it was made for; a gesture
    /// replays its pointer commands on that picture. An edit for a source
    /// that gave no read is only resolved. A queued edit that fails is undone
    /// and reported; a start that fails fails the command, and `own`'s
    /// failure is returned once the queued edits are presented.
    fn apply_queue(&mut self, shared: &Shared<'_>, binding: &Binding, read: &R, target: Option<&Json>, queue: Queue, own: Edit) -> Result<Resolution> {
        enum Role { Start, Queued, Own }
        let previous = self.configured.clone();
        let mut candidate = previous.as_ref().map(|previous| previous.configuration.clone());
        let schema = PlotSchema::new(read.source().schema.clone());
        let shown_source = self.source.as_ref().map(|source| (source.signal.clone(), source.window));
        let shown_signal = self.source.as_ref().map(|source| source.signal.clone());
        let snapshot = read.source().clone();
        let count = queue.edits.len();
        let mut edits: Vec<Edit> = queue.edits.into_iter().chain([own]).collect();
        let mut start = None;
        if candidate.is_none() {
            let declared = declared_start(match queue.start { Some(start) => start, None => Edit::start(&self.key, binding, target)? }, edits);
            start = Some(declared.0);
            edits = declared.1;
        }
        let edits: Vec<(Edit, Role)> = start.map(|start| (start, Role::Start)).into_iter()
            .chain(edits.into_iter().enumerate().map(|(index, edit)| (edit, if index < count { Role::Queued } else { Role::Own }))).collect();
        let (mut showing, mut running, mut rejected, mut failure): (Option<Showing>, Option<(String, i64)>, Vec<String>, Option<PlotError>) =
            (None, shown_source.clone(), Vec::new(), None);
        if let Some(handle) = self.session.as_ref().filter(|_| edits.iter().any(|(edit, _)| !edit.pointer.is_empty())) {
            // Replayed presses carry their own times; the same presses drawn
            // while waiting must not make them double clicks.
            lock(handle).click_history = ClickHistory::new();
        }
        for (edit, role) in edits {
            let source = edit.source();
            if candidate.is_some() && Some(&source) == running.as_ref() && !edit.changes() { continue; }
            // An edit made before the panel was rebound is on the source it showed.
            let on_shown = previous.is_some() && Some(&edit.signal) == shown_signal.as_ref() && shown_signal.as_deref() != Some(binding.signal.as_str());
            let picture = if on_shown && Some(&source) == shown_source.as_ref() { self.input.as_ref().map(|input| input.source().clone()) }
                          else if source == (binding.signal.clone(), binding.window) { Some(snapshot.clone()) } else { None };
            if let (Some(picture), Some(base)) = (&picture, &candidate) {
                if !presents(&showing, base, picture) {
                    showing = self.prepare_on(shared, picture, base.clone()).and_then(|packet| self.present_in(shared, base.clone(), picture, packet)).ok();
                    if let Some((presented, _, _)) = &showing { candidate = Some(presented.clone()); }
                }
            }
            let outcome = (|| -> Result<()> {
                if !edit.pointer.is_empty() {
                    let current = showing.clone().filter(|(_, on, _)| picture.as_ref().is_some_and(|picture| Arc::ptr_eq(on, picture)))
                        .ok_or_else(|| PlotError::value("There is no presented scene to interact with"))?;
                    let handle = self.session.clone().ok_or_else(|| json_key_error(&self.key))?;
                    let recipe = {
                        let mut session = lock(&handle);
                        let before = session.figure_recipe()?;
                        if let Err(error) = replay_events(&mut session, &edit.pointer) {
                            showing = None;
                            reopen(&mut session);
                            return Err(error);
                        }
                        gesture_updates(&before, &session.figure_recipe()?)
                    };
                    if !recipe.is_empty() {
                        let gesture = Edit { interaction: Some(InteractionUpdates { recipe, ..Default::default() }), ..edit.clone() };
                        candidate = Some(self.resolve_edit(shared, Some(schema.clone()), candidate.as_ref(), &gesture, true)?);
                    }
                    showing = Some((candidate.clone().expect("candidate"), current.1, current.2));
                } else {
                    let same_source = self.initial_overlay.is_some() || running.as_ref().is_some_and(|running| running.0 == edit.signal);
                    let edit_schema = if on_shown { previous.as_ref().and_then(|previous| previous.configuration.schema.clone()) } else { Some(schema.clone()) };
                    let resolved = self.resolve_edit(shared, edit_schema, candidate.as_ref(), &edit, same_source)?;
                    match &picture {
                        None => candidate = Some(resolved),
                        Some(picture) => {
                            let packet = self.prepare_on(shared, picture, resolved.clone())?;
                            showing = None;
                            let presented = self.present_in(shared, resolved, picture, packet)?;
                            candidate = Some(presented.0.clone());
                            showing = Some(presented);
                        }
                    }
                }
                Ok(())
            })();
            match outcome {
                Ok(()) => running = Some(source),
                Err(error) => match role {
                    Role::Start => return Err(error),
                    Role::Own => { failure = Some(error); break; }
                    Role::Queued => rejected.push(format!("Not applied when data arrived: {}", status_text(&error))),
                },
            }
        }
        let mut candidate = candidate.expect("a started panel has a candidate");
        if !presents(&showing, &candidate, &snapshot) {
            let packet = self.prepare_on(shared, &snapshot, candidate.clone())?;
            let presented = self.present_in(shared, candidate, &snapshot, packet)?;
            candidate = presented.0.clone();
            showing = Some(presented);
        }
        self.configured = Some(Configured { signal: binding.signal.clone(), window: binding.window, configuration: candidate.clone() });
        let errors: Vec<String> = rejected.into_iter().chain(failure.as_ref().map(status_text)).collect();
        let status = self.status_mut();
        status.error = None;
        status.edit_error = (!errors.is_empty()).then(|| errors.join("; "));
        status.authoring_pending = Some(false);
        status.message = None;
        Ok(Resolution { configuration: candidate, changed: true, shown: showing.map(|(_, _, packet)| packet), failure })
    }

    /// `_resolve_edit`: one edit from `base`, as the panel's session (or,
    /// without one, the configuration resolver) resolves a typed edit.
    fn resolve_edit(&self, shared: &Shared<'_>, schema: Option<Arc<PlotSchema>>, base: Option<&Arc<ResolvedConfiguration>>, edit: &Edit,
                    same_source: bool) -> Result<Arc<ResolvedConfiguration>> {
        let mut semantic = edit.semantic_map()?;
        if let Some((name, steps)) = &edit.scope {
            let field = base.and_then(|base| base.semantics.fields.iter().find(|field| field.name == *name));
            let cycle = field.and_then(|field| field.cycle_choices.as_ref()).filter(|cycle| !cycle.is_empty())
                .ok_or_else(|| PlotError::value("This field has no coordinate scope"))?;
            let index = (cycle.position_of(&field.expect("scoped field").value).unwrap_or(0) as i64 + steps).clamp(0, cycle.len() as i64 - 1) as usize;
            semantic = Some(ValueMap::from_pairs([(name.as_str(), cycle.get(index).expect("cycle position").0.clone())]));
        }
        // Parameters are typed updates: one restated unchanged authors nothing.
        let mut updates = ConfigurationUpdates { fit_live: Some(true), ..Default::default() };
        if let Some(parameters) = &edit.parameters { updates.parameters = Value::from_json(parameters).as_map().cloned(); }
        if let Some(overlay) = &self.initial_overlay { updates.image_overlay = Some(overlay.clone().map(|overlay| overlay as OverlayHandle)); }
        else if let Some(overlay) = &self.read_overlay {
            // A live image's layer follows the kind: a panel turned into a curve drops it.
            updates.image_overlay = Some(overlay.clone().map(|overlay| overlay as OverlayHandle));
            updates.overlay_follows_kind = true;
        }
        let services = shared.services;
        match &self.session {
            None => {
                let kind = edit.kind.as_deref().and_then(PlotKind::parse)
                    .ok_or_else(|| PlotError::value(format!("{} is not a valid PlotKind", crate::configuration::value::repr_str(edit.kind.as_deref().unwrap_or("None")))))?;
                resolve_configuration(ResolveRequest {
                    schema, kind: Some(kind), target: Some(&edit.target), base, semantic_updates: semantic, different_source: !same_source,
                    window: Some(edit.window as i128), defaults: Some(services.defaults.clone()), fit: Some(services.fit.clone()),
                    unit_registry: services.registry.clone(), configuration_updates: Some(updates), ..Default::default()
                })
            }
            Some(handle) => lock(handle).resolve_configuration(schema, Some(&edit.target), base, semantic, None, same_source, Some(edit.window as i128),
                                                               edit.interaction.clone(), Some(updates), None),
        }
    }

    /// `_prepare(key, snapshot, configuration)`.
    fn prepare_on(&self, shared: &Shared<'_>, source: &Arc<Source>, configuration: Arc<ResolvedConfiguration>) -> Result<PreparedPlot> {
        let services = shared.services;
        match &self.session {
            None => {
                let mut schemas = crate::projection::SchemaCache::default();
                DataPreparation::new(configuration, None).prepare(&PlotInput::Dataset(source.clone()), None, &mut schemas, &services.settings(), &services.fit.engine)
            }
            Some(handle) => lock(handle).prepare_data(&PlotInput::Dataset(source.clone()), None, Some(configuration)),
        }
    }

    /// `_present`: `packet` in the panel's session, creating it.
    fn present_in(&mut self, shared: &Shared<'_>, configuration: Arc<ResolvedConfiguration>, source: &Arc<Source>, packet: PreparedPlot) -> Result<Showing> {
        let Some(handle) = self.session.clone() else {
            let session = PlotSession::from_prepared(packet.clone(), configuration, shared.dpr, &self.key, shared.services.clone())?;
            let presented = session.presented_configuration().cloned().expect("a presented session");
            self.session = Some(Arc::new(Mutex::new(session)));
            return Ok((presented, source.clone(), packet));
        };
        let mut session = lock(&handle);
        match session.present_prepared(packet.clone(), Some(configuration), false) {
            Ok(presented) => Ok((presented, source.clone(), packet)),
            Err(error) => { reopen(&mut session); Err(error) }
        }
    }

    /// `_apply_configuration`: stage the panel's new scene (prepared, a
    /// display-only change of the accepted one, or `shown`: the packet its
    /// session already presents it with).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apply(&mut self, shared: &Shared<'_>, read: &R, configuration: Arc<ResolvedConfiguration>, changed: bool, live_data: bool,
                        publisher: Option<&dyn FitPublisher<R>>, shown: Option<PreparedPlot>) -> Result<()> {
        let source = read.source().clone();
        let shown_now = shown.is_some();
        let prepared = match shown {
            Some(packet) => packet,
            None => {
                let same_shown = self.input.as_ref().is_some_and(|input| SnapshotRef::of(input.source()) == SnapshotRef::of(&source));
                let presented = self.session.as_ref().and_then(|handle| lock(handle).presented_configuration().cloned());
                let already = presented.as_ref().is_some_and(|presented| presented.signature == configuration.signature
                    && super::same_target(&presented.target().unwrap_or_default(), &configuration.target().unwrap_or_default()));
                if !changed && already && same_shown { return Ok(()); }
                if self.session.is_some() && same_shown && presented.as_ref().is_some_and(|presented| presented.signature == configuration.signature) {
                    self.staged = Some(Staged { read: read.clone(), scene: StagedScene::Display, configuration, live_data: false });
                    return Ok(());
                }
                self.prepare_on(shared, &source, configuration.clone())?
            }
        };
        if let Some(evaluation) = &prepared.evaluation {
            let (values, records) = fit_parameter_data(evaluation, &source)?;
            let metadata = json!({"operation": "plot-fit", "fit": Value::Map(configuration.fit_target.clone()).to_json()});
            match publisher.filter(|_| live_data) {
                Some(publisher) => publisher.publish(&self.key, read, values, records, metadata).map_err(PlotError::runtime)?,
                None => self.manual_fits.push(Effect::Fit { panel: self.key.clone(), source: read.clone(), values, records, metadata }),
            }
            self.fit_output = true;
        }
        let scene = if shown_now { StagedScene::Shown } else { StagedScene::Prepared(prepared) };
        self.staged = Some(Staged { read: read.clone(), scene, configuration, live_data });
        Ok(())
    }

    /// `_present_one(key)`: the staged scene's front, and whether nothing changed.
    pub(crate) fn present(&mut self, shared: &Shared<'_>) -> (Option<RgbaFront>, bool) {
        let Some(staged) = self.staged.clone() else { return (None, false) };
        let outcome = super::guarded(|| -> Result<(Option<RgbaFront>, bool)> {
            match staged.scene {
                StagedScene::Shown => Ok((Some(self.accept_read(&staged.read)?), false)),
                StagedScene::Display => {
                    let handle = self.session.clone().ok_or_else(|| json_key_error(&self.key))?;
                    let mut session = lock(&handle);
                    let epoch = session.presentation_epoch();
                    session.present_configuration(staged.configuration.clone())?;
                    if epoch != session.presentation_epoch() { return Ok((Some(session.rgba()?), false)); }
                    Ok((None, true))
                }
                StagedScene::Prepared(prepared) => Ok((Some(self.accept(shared, &staged.read, prepared, staged.configuration.clone(), staged.live_data)?), false)),
            }
        });
        match outcome {
            Ok(outcome) => outcome,
            Err(error) => { self.status_mut().error = Some(status_text(&error)); (None, false) }
        }
    }

    /// `_accept_prepared`: the new scene goes on screen.
    fn accept(&mut self, shared: &Shared<'_>, read: &R, prepared: PreparedPlot, configuration: Arc<ResolvedConfiguration>, live_data: bool) -> Result<RgbaFront> {
        if prepared.source_ref != Some(SnapshotRef::of(read.source())) {
            return Err(PlotError::value("prepared result and NativeRead identify different scientific data"));
        }
        let presented = match self.session.clone() {
            None => {
                let session = PlotSession::from_prepared(prepared, configuration.clone(), shared.dpr, &self.key, shared.services.clone())?;
                let presented = session.presented_configuration().cloned().expect("a presented session");
                self.session = Some(Arc::new(Mutex::new(session)));
                presented
            }
            Some(handle) => lock(&handle).present_prepared(prepared, Some(configuration.clone()), live_data)?,
        };
        if !Arc::ptr_eq(&presented, &configuration) {
            // The session clamped what the grid has no cell for: record what it shows.
            if let Some(configured) = &mut self.configured { configured.configuration = presented; }
        }
        self.accept_read(read)
    }

    /// The panel's session presents its configuration on `read`: its front.
    fn accept_read(&mut self, read: &R) -> Result<RgbaFront> {
        self.input = Some(read.clone());
        let configured = self.configured.clone().ok_or_else(|| json_key_error(&self.key))?;
        let binding = self.status.as_ref().and_then(|status| status.binding.clone()).ok_or_else(|| json_key_error("binding"))?;
        self.source = Some(PanelSource { signal: configured.signal, window: configured.window, epoch: binding.epoch });
        let status = self.status_mut();
        status.error = None;
        status.message = None;
        let handle = self.session.clone().ok_or_else(|| json_key_error(&self.key))?;
        let front = lock(&handle).rgba()?;
        Ok(front)
    }
}

/// `_replay_events`: queued pointer commands in order, on the picture the
/// session presents. A wheel notch zooms the committed viewport (the drawn
/// limits only before there is one) and reads nothing else drawing changes,
/// so notches in a row are drawn once, after the last of them.
fn replay_events(session: &mut PlotSession, events: &[(Json, f64)]) -> Result<()> {
    let outcome = (|| -> Result<()> {
        let mut held = false;
        for (command, at) in events {
            let wheel = command.get("event").and_then(Json::as_str) == Some("scroll") && !session.renderer().height_bars_scene();
            if wheel && !held {
                session.hold_drawing();
                held = true;
            } else if held && !wheel {
                session.draw_held(false)?;
                held = false;
            }
            replay(session, command, *at)?;
        }
        if held { session.draw_held(false)?; }
        Ok(())
    })();
    if outcome.is_err() { let _ = session.draw_held(true); }
    outcome
}

/// `_replay`: one queued pointer command on the picture the session
/// presents at its turn. It keeps its place on its axes: the axes of the
/// same role and cell (else the one axes of that role), at the same
/// fraction of their box.
fn replay(session: &mut PlotSession, command: &Json, at: f64) -> Result<()> {
    let mut command = command.clone();
    if let Some(axes) = command.get("axes").filter(|axes| !axes.is_null()).cloned() {
        let drawn: Vec<Json> = session.raster_axes_snapshot()?.iter().map(axis_wire).collect();
        let same: Vec<&Json> = {
            let exact: Vec<&Json> = drawn.iter().filter(|item| item["role"] == axes["role"] && item["cell_index"] == axes["cell_index"]).collect();
            if exact.is_empty() { drawn.iter().filter(|item| item["role"] == axes["role"]).collect() } else { exact }
        };
        let target = if same.len() == 1 { Some(same[0].clone()) } else { None };
        let bounds = |wire: &Json| -> Option<[f64; 4]> {
            let items = wire["bounds"].as_array()?;
            Some([items.first()?.as_f64()?, items.get(1)?.as_f64()?, items.get(2)?.as_f64()?, items.get(3)?.as_f64()?])
        };
        if let (Some([left, top, right, bottom]), Some([to_left, to_top, to_right, to_bottom])) = (bounds(&axes), target.as_ref().and_then(bounds)) {
            if [left, top, right, bottom] != [to_left, to_top, to_right, to_bottom] {
                let number = |name: &str| command.get(name).map_or(Some(0.0), Json::as_f64).unwrap_or(f64::NAN);
                let (x, y) = (number("x"), number("y"));
                command["x"] = json!(to_left + (x - left) / (right - left) * (to_right - to_left));
                command["y"] = json!(to_bottom - (bottom - y) / (bottom - top) * (to_bottom - to_top));
            }
        }
        command["axes"] = target.unwrap_or(Json::Null);
    }
    let event = pointer_event(&command, Some(at))?;
    super::guarded(|| session.raster_pointer_event(&event))?;
    Ok(())
}
