//! `PlotSession`: one accepted scientific scene, its configuration,
//! selectors, fit and classifier, and the transactional presentation of
//! each new scene through the panel's renderer.

use std::any::Any;
use std::collections::BTreeMap;
use std::sync::Arc;

use serde_json::{json, Value as Json};

use crate::configuration::data_contract::PlotSchema;
use crate::configuration::display::accepts_classifier_thresholds;
use crate::configuration::error::{ErrorKind, PlotError, Result};
use crate::configuration::ordered::OrderedMap;
use crate::configuration::parameters::{ParameterSchema, RenderEffect, LIMIT_PAIRS};
use crate::configuration::recipe::{encode_plot_recipe, RecipeParts};
use crate::configuration::resolve::{resolve_configuration, ConfigurationUpdates, Defaults, InteractionUpdates, OverlayHandle,
                                    ResolveRequest, ResolvedConfiguration};
use crate::configuration::selectors::{GaussianComponents, SelectorKind, SelectorState, Viewport};
use crate::configuration::spec::{PlotLabels, PlotSpec, PulseTimelinePlot};
use crate::configuration::state::{DisplayState, DisplayStateStore, Interaction, InteractionUpdate, PresentationUpdate};
use crate::configuration::value::{repr_str, Value, ValueMap};
use crate::fit::target::{fit_request_target, LiveFitRequest};
use crate::fit::FitResult;
use crate::interaction::controller::{SelectorController, SelectorSnapshot};
use crate::interaction::engine::{AxesId, PointerGesture};
use crate::interaction::gesture::{ClickHistory, InteractionDefaults};
use crate::interaction::transform::{AxisTransform, CanvasSize};
use crate::kinds::PlotKind;
use crate::layout::{resolve_surface, FacetTopology, Kind as LayoutKind, SurfaceOptions, SurfacePlan};
use crate::projection::{Payload, SchemaCache};
use crate::pulse::PulseTimelineData;
use crate::render::scale::AxisScale;
use crate::units::UnitRegistry;
use zlc_core::schema::{CodeBase, Coordinates, DatasetSchema, DomainRole, ValidityContract};
use crate::projection::model::PRIMARY_INDEX;

use super::evaluation::{ClassifierEvaluation, FitEvaluation, ThresholdKey};
use super::fit_layer::EngineFitLayer;
use super::overlay::{FitOverlay, ImagePointOverlay};
use super::preparation::{DataPreparation, PreparedPlot};
use super::projected::{FitProjection, FitSelection, PlotInput, ProjectionContext, ProjectionSettings, SampleAxis};
use super::renderer::{PanelRenderer, RenderFrame, RendererFactory};

/// What every session of a document shares.
#[derive(Clone)]
pub struct SessionServices {
    pub renderers: Arc<dyn RendererFactory>,
    pub fit: Arc<EngineFitLayer>,
    pub defaults: Arc<Defaults>,
    pub registry: Option<Arc<UnitRegistry>>,
    pub interaction: InteractionDefaults,
}

impl SessionServices {
    pub fn settings(&self) -> ProjectionSettings { ProjectionSettings { registry: self.registry.clone(), defaults: self.defaults.clone() } }
}

/// One accepted fit and its painted presentation (`_AcceptedFit`).
#[derive(Clone, Debug)]
pub struct AcceptedFit {
    pub answer: FitAnswer,
    pub selection: Option<Arc<FitSelection>>,
    pub context_generation: u64,
    pub source_generation: Arc<str>,
    pub request: LiveFitRequest,
    pub evaluation: Option<Arc<FitEvaluation>>,
    pub overlays: Arc<[FitOverlay]>,
    pub selections: Vec<Option<Arc<FitSelection>>>,
}

/// A single fit result, or a facet batch (`FacetFitBatchResult`).
#[derive(Clone, Debug)]
pub enum FitAnswer {
    Single(FitResult),
    Batch { results: Vec<Option<FitResult>>, failure_messages: Vec<Option<String>>, source_revision: u64,
            parameter_units: Vec<(String, String)>, sample_axes: Vec<SampleAxis>, batch_revision: u64 },
}

impl FitAnswer {
    pub fn source_revision(&self) -> u64 { match self { FitAnswer::Single(result) => result.source_revision, FitAnswer::Batch { source_revision, .. } => *source_revision } }
    pub fn model_id(&self, request: &LiveFitRequest) -> String { match self { FitAnswer::Single(result) => result.model.model_id.clone(), FitAnswer::Batch { .. } => request.model.model_id.clone() } }
}

/// Everything a refused presentation restores (`_CONFIGURATION_STATE_NAMES`
/// plus the display store, the selector controller and the warm seeds).
#[derive(Clone)]
pub(crate) struct Scene {
    pub spec: PlotSpec,
    pub parameter_schema: Arc<ParameterSchema>,
    pub display: DisplayStoreState,
    pub projection: FitProjection,
    pub image_overlay: Option<OverlayHandle>,
    pub viewport: Option<Viewport>,
    pub focused_facet_index: Option<usize>,
    pub facet_focus_index: Option<usize>,
    pub accepted_fit: Option<Arc<AcceptedFit>>,
    pub classifier_results: Arc<[Option<FitResult>]>,
    pub classifier_overlays: Arc<[FitOverlay]>,
    pub classifier_thresholds: OrderedMap<ThresholdKey, Option<f64>>,
    pub classifier_components: OrderedMap<ThresholdKey, Option<Option<GaussianComponents>>>,
    pub fit_expression_failure: Option<(String, String)>,
    pub layout_revision: u64,
    pub size: &'static str,
    pub fit_context_generation: u64,
    pub fit_request_generation: i64,
    pub fit_batch_revision: u64,
    pub live_fit_request: Option<LiveFitRequest>,
    pub presentation_epoch: u64,
    pub accepted_configuration: Option<Arc<ResolvedConfiguration>>,
    pub accepted_prepared: Option<PreparedPlot>,
    pub classifier_evaluation: Option<Arc<ClassifierEvaluation>>,
    pub gesture: Option<PointerGesture>,
    pub pending_configuration: Option<Arc<ResolvedConfiguration>>,
    pub display_preparation: Option<DataPreparation>,
    pub selectors: SelectorController,
    pub fit_warm_starts: BTreeMap<(i64, String, Option<usize>), Vec<f64>>,
}

/// The display store as a value: its schema and current state.
#[derive(Clone)]
pub(crate) struct DisplayStoreState { pub state: Arc<DisplayState> }

impl DisplayStoreState {
    fn new(schema: Arc<ParameterSchema>, values: &ValueMap, interaction: Interaction) -> Result<Self> {
        let store = DisplayStateStore::new(schema, Some(values), 0, RenderEffect::LAYOUT, interaction)?;
        Ok(Self { state: store.state().clone() })
    }
    /// `_commit_interaction`: an unchanged lock is no revision.
    pub(crate) fn commit_interaction(&mut self, interaction: &Interaction) -> bool {
        let current = &self.state;
        if *interaction == current.interaction { return false; }
        self.state = Arc::new(DisplayState { revision: current.revision + 1, values: current.values.clone(), changed_names: Default::default(),
                                             effects: RenderEffect::OVERLAY, interaction: interaction.clone() });
        true
    }
}

struct Pending {
    snapshot: Scene,
    renderer_plan: SurfacePlan,
    presentation: crate::configuration::state::Presentation,
    changed: bool,
    draw_attempted: bool,
    live_fit_batch: Option<u64>,
}

/// The session of one panel.
pub struct PlotSession {
    pub(crate) services: SessionServices,
    pub(crate) closed: bool,
    pub(crate) for_export: bool,
    pub(crate) device_pixel_ratio: f64,
    pub(crate) scene: Scene,
    pub(crate) renderer: Box<dyn PanelRenderer>,
    pub(crate) click_history: ClickHistory,
    pub(crate) schemas: SchemaCache,
    pending: Option<Pending>,
    /// Effects held instead of drawn (`_configuration_effects`), drawn at
    /// once by [`PlotSession::draw_held`].
    pub(crate) held: Option<RenderEffect>,
}

/// A configuration's image overlay as the frame's point layer.
pub fn overlay_of(handle: &Option<OverlayHandle>) -> Option<Arc<ImagePointOverlay>> {
    handle.clone().and_then(|handle| (handle as Arc<dyn Any + Send + Sync>).downcast::<ImagePointOverlay>().ok())
}

/// `a is b` for overlay handles.
pub fn same_overlay(a: &Option<OverlayHandle>, b: &Option<OverlayHandle>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => std::ptr::eq(Arc::as_ptr(a) as *const u8, Arc::as_ptr(b) as *const u8),
        _ => false,
    }
}

fn validated_ratio(value: f64) -> Result<f64> {
    if !value.is_finite() || value <= 0.0 { return Err(PlotError::value("device pixel ratio must be a positive finite number")); }
    Ok(value)
}

fn layout_kind(kind: PlotKind) -> LayoutKind {
    match kind {
        PlotKind::Curve => LayoutKind::Curve, PlotKind::Image => LayoutKind::Image, PlotKind::Histogram => LayoutKind::Histogram,
        PlotKind::Rolling => LayoutKind::Rolling, PlotKind::FacetGrid => LayoutKind::FacetGrid, PlotKind::PulseTimeline => LayoutKind::PulseTimeline,
    }
}

/// The canvas pointer events are measured in (`canvas_physical_size`).
pub fn canvas(renderer: &dyn PanelRenderer) -> CanvasSize { renderer.canvas_size() }

impl PlotSession {
    // ------------------------------------------------------------- construction
    /// `PlotSession.from_prepared(prepared, configuration=, device_pixel_ratio=)`:
    /// the new session holds the presentation open for the frontend's receipt.
    pub fn from_prepared(prepared: PreparedPlot, configuration: Arc<ResolvedConfiguration>, device_pixel_ratio: f64, panel: &str,
                         services: SessionServices) -> Result<PlotSession> {
        Self::build(prepared, configuration, device_pixel_ratio, Some(panel), services, false)
    }

    /// `pulse_timeline(data, labels=, size=, parameters=, device_pixel_ratio=)`
    /// (`PlotSession.__init__` of a PulseTimelineData): one authored pulse
    /// timeline's session, its first front accepted.
    pub fn pulse_timeline(data: Arc<PulseTimelineData>, labels: PlotLabels, size: Option<&str>, parameters: Option<&ValueMap>,
                          device_pixel_ratio: f64, panel: Option<&str>, services: SessionServices) -> Result<PlotSession> {
        let spec = PlotSpec::PulseTimeline(PulseTimelinePlot { labels });
        let values = crate::configuration::display::parameter_schema_for(&spec, &services.defaults.style).initial_values(parameters)?;
        let layout = &services.defaults.layout;
        let size = match size { None => layout.default_preset, Some(size) => layout.validate_preset(size).map_err(PlotError::value)? };
        validated_ratio(device_pixel_ratio)?;
        let updates = ConfigurationUpdates { parameters: Some(values), size: Some(size.to_owned()), image_overlay: Some(None), ..Default::default() };
        let candidate = resolve_configuration(ResolveRequest {
            spec: Some(spec), defaults: Some(services.defaults.clone()), fit: Some(services.fit.clone()), unit_registry: services.registry.clone(),
            configuration_updates: Some(updates), ..Default::default()
        })?;
        let prepared = DataPreparation::new(candidate.clone(), None)
            .prepare(&PlotInput::Pulse(data), Some(0), &mut SchemaCache::default(), &services.settings(), &services.fit.engine)?;
        let mut session = Self::build(prepared, candidate, device_pixel_ratio, panel, services, false)?;
        session.finish_presentation(true, None);
        Ok(session)
    }

    pub(crate) fn build(prepared: PreparedPlot, configuration: Arc<ResolvedConfiguration>, device_pixel_ratio: f64, panel: Option<&str>,
                        services: SessionServices, for_export: bool) -> Result<PlotSession> {
        let mut candidate = configuration;
        let size = services.defaults.layout.validate_preset(candidate.size).map_err(PlotError::value)?;
        let display = DisplayStoreState::new(candidate.parameter_schema.clone(), &candidate.display_state.values, candidate.interaction.clone())?;
        let spec = candidate.spec.clone();
        let focused = if spec.is_facet_grid() { Some(0) } else { None };
        let scene = Scene {
            parameter_schema: candidate.parameter_schema.clone(), display, projection: prepared.projection.clone(), image_overlay: None,
            viewport: candidate.viewport,
            focused_facet_index: candidate.focused_facet_index.map(|i| i as usize).or(if spec.is_facet_grid() { focused } else { None }),
            facet_focus_index: candidate.facet_focus.map(|i| i as usize), accepted_fit: None, classifier_results: Arc::from(Vec::new()),
            classifier_overlays: Arc::from(Vec::new()), classifier_thresholds: OrderedMap::new(), classifier_components: OrderedMap::new(),
            fit_expression_failure: None, layout_revision: 0, size, fit_context_generation: 0, fit_request_generation: 0,
            fit_batch_revision: 0, live_fit_request: None, presentation_epoch: 0, accepted_configuration: None, accepted_prepared: None,
            classifier_evaluation: None, gesture: None, pending_configuration: None, display_preparation: None,
            selectors: SelectorController::new(), fit_warm_starts: BTreeMap::new(), spec,
        };
        let device_pixel_ratio = validated_ratio(device_pixel_ratio)?;
        let plan = surface_plan_for(&scene, &services, device_pixel_ratio)?;
        let mut scene = scene;
        scene.size = plan.preset;
        if candidate.size != plan.preset {
            let updates = ConfigurationUpdates { size: Some(plan.preset.to_owned()), ..Default::default() };
            candidate = resolve_configuration(ResolveRequest {
                schema: candidate.schema.clone(), base: Some(&candidate), presented: Some(&candidate), defaults: Some(services.defaults.clone()),
                fit: Some(services.fit.clone()), unit_registry: services.registry.clone(), configuration_updates: Some(updates), ..Default::default()
            })?;
        }
        let renderer = services.renderers.create(panel, &candidate.spec, &plan, &services.defaults.style)?;
        scene.pending_configuration = Some(candidate.clone());
        let mut session = PlotSession { services, closed: false, for_export, device_pixel_ratio, scene, renderer,
                                        click_history: ClickHistory::new(), schemas: SchemaCache::default(), pending: None, held: None };
        session.present_prepared(prepared, Some(candidate), false)?;
        Ok(session)
    }

    fn assert_open(&self) -> Result<()> { if self.closed { Err(PlotError::runtime("plot session is closed")) } else { Ok(()) } }

    // ------------------------------------------------------------- accessors
    pub fn spec(&self) -> &PlotSpec { &self.scene.spec }
    pub fn kind(&self) -> PlotKind { self.scene.spec.kind() }
    pub fn display_state(&self) -> &Arc<DisplayState> { &self.scene.display.state }
    pub fn parameter_schema(&self) -> &Arc<ParameterSchema> { &self.scene.parameter_schema }
    pub fn presented_configuration(&self) -> Option<&Arc<ResolvedConfiguration>> { self.scene.accepted_configuration.as_ref() }
    pub fn image_overlay(&self) -> Option<Arc<ImagePointOverlay>> { overlay_of(&self.scene.image_overlay) }
    pub fn projection(&self) -> &FitProjection { &self.scene.projection }
    pub fn data(&self) -> PlotInput { self.scene.projection.data() }
    pub fn data_revision(&self) -> u64 { self.scene.projection.data_revision() }
    /// Which run the data came out of; a pulse timeline has none.
    pub fn data_generation(&self) -> Option<Arc<str>> { self.scene.projection.data_generation().cloned() }
    pub fn presentation_epoch(&self) -> u64 { self.scene.presentation_epoch }
    pub fn size(&self) -> &'static str { self.renderer.plan().preset }
    pub fn surface_plan(&self) -> &SurfacePlan { self.renderer.plan() }
    pub fn viewport(&self) -> Option<Viewport> { self.scene.viewport }
    pub fn renderer(&self) -> &dyn PanelRenderer { &*self.renderer }
    /// The open FacetGrid cell, `None` while showing the overview.
    pub fn facet_focus(&self) -> Option<usize> { if self.scene.spec.is_facet_grid() { self.scene.facet_focus_index } else { None } }
    pub(crate) fn semantic_kind(&self) -> PlotKind { self.scene.spec.semantic().kind() }

    /// `_projected`: the projection under the session's current context.
    pub(crate) fn projected(&self) -> FitProjection { self.scene.projection.with_context(Arc::new(self.context())) }

    pub(crate) fn context(&self) -> ProjectionContext {
        ProjectionContext {
            display_state: self.scene.display.state.clone(),
            selectors: self.scene.selectors.snapshot(true).expect("controller snapshot"),
            viewport: self.scene.viewport, focused_facet_index: self.scene.focused_facet_index,
        }
    }

    // ------------------------------------------------------------- configuration
    /// `_fit_configuration`: the normalized fit target of the accepted fit.
    pub fn fit_configuration(&self) -> ValueMap {
        let request = self.scene.accepted_fit.as_ref().map(|accepted| &accepted.request);
        Value::from_json(&fit_request_target(request)).as_map().cloned().unwrap_or_default()
    }

    pub(crate) fn current_limits_for(&self, low: &str) -> Result<(f64, f64)> {
        let sorted = |(a, b): (f64, f64)| if b < a { (b, a) } else { (a, b) };
        if low == "color_min" {
            return Ok(sorted(self.renderer.resolved_color_limits().ok_or_else(|| PlotError::type_error("color limits require an active Image artist"))?));
        }
        let geometry = self.renderer.geometry(&self.renderer.primary_axes()).ok_or_else(|| PlotError::runtime("primary axes have no geometry"))?;
        match low {
            "y_min" => Ok(sorted(geometry.y_limits)),
            "x_min" => Ok(sorted(geometry.x_limits)),
            other => Err(PlotError::key(repr_str(&format!("{} is an authored limit pair with nowhere to take its current value from; say where it materialises from here", repr_str(other))))),
        }
    }

    fn configuration_limits(&self) -> Result<OrderedMap<String, (f64, f64)>> {
        let mut limits = OrderedMap::new();
        for (_mode, low, high) in LIMIT_PAIRS {
            if self.scene.parameter_schema.contains(low) && self.scene.parameter_schema.contains(high) {
                limits.insert(low.to_owned(), self.current_limits_for(low)?);
            }
        }
        Ok(limits)
    }

    fn schema(&self) -> Option<Arc<PlotSchema>> { self.data().schema() }

    /// `_capture_configuration`: the accepted facts as resolver input.
    pub fn capture_configuration(&self) -> Result<Arc<ResolvedConfiguration>> {
        let base = self.scene.accepted_configuration.as_ref();
        let updates = ConfigurationUpdates {
            parameters: Some(self.scene.display.state.values.clone()), size: Some(self.scene.size.to_owned()),
            selectors: Some(self.scene.selectors.states()), viewport: Some(self.scene.viewport),
            facet_focus: Some(self.scene.facet_focus_index.map(|i| i as i128)), focused_facet_index: Some(self.scene.focused_facet_index.map(|i| i as i128)),
            classifier_thresholds: Some(crate::configuration::selectors::thresholds_wire(&self.classifier_threshold_targets_state(false)?)),
            fit: Some(self.fit_configuration()), fit_live: Some(self.scene.live_fit_request.is_some()),
            interaction: Some(InteractionUpdate { series_lock: Some(self.scene.display.state.interaction.series_lock.clone()) }),
            presentation: Some(PresentationUpdate { series_readout: Some(self.renderer.series_presentation().series_readout) }),
            image_overlay: Some(self.scene.image_overlay.clone()), ..Default::default()
        };
        resolve_configuration(ResolveRequest {
            schema: self.schema(), base, presented: base, spec: Some(self.scene.spec.clone()), defaults: Some(self.services.defaults.clone()),
            fit: Some(self.services.fit.clone()), unit_registry: self.services.registry.clone(), fixed_limits: self.configuration_limits()?,
            configuration_updates: Some(updates), ..Default::default()
        })
    }

    /// `resolve_configuration(...)` against the presented configuration; the
    /// candidate becomes the pending one.
    #[allow(clippy::too_many_arguments)]
    pub fn resolve_configuration(&mut self, schema: Option<Arc<PlotSchema>>, target: Option<&ValueMap>, base: Option<&Arc<ResolvedConfiguration>>,
                                 semantic_updates: Option<ValueMap>, parameter_updates: Option<ValueMap>, same_source: bool, window: Option<i128>,
                                 interaction_updates: Option<InteractionUpdates>, configuration_updates: Option<ConfigurationUpdates>,
                                 spec: Option<PlotSpec>) -> Result<Arc<ResolvedConfiguration>> {
        self.assert_open()?;
        let presented = self.capture_configuration()?;
        let selected = base.cloned().unwrap_or_else(|| presented.clone());
        let candidate = resolve_configuration(ResolveRequest {
            schema: schema.or_else(|| presented.schema.clone()), target, base: Some(&selected), presented: Some(&presented), spec,
            semantic_updates, parameter_updates, different_source: !same_source, window, defaults: Some(self.services.defaults.clone()),
            fit: Some(self.services.fit.clone()), unit_registry: self.services.registry.clone(), fixed_limits: self.configuration_limits()?,
            interaction_updates, configuration_updates, ..Default::default()
        })?;
        self.scene.pending_configuration = Some(candidate.clone());
        Ok(candidate)
    }

    // ------------------------------------------------------------- preparation
    /// `prepare_data(snapshot, configuration=, revision=)`: one template for
    /// display-owned scientific updates (`revision` numbers a pulse timeline).
    pub fn prepare_data(&mut self, input: &PlotInput, revision: Option<u64>, configuration: Option<Arc<ResolvedConfiguration>>)
                        -> Result<PreparedPlot> {
        let candidate = match configuration.or_else(|| self.scene.pending_configuration.clone()) { Some(c) => c, None => self.capture_configuration()? };
        let reusable = self.scene.display_preparation.as_ref().is_some_and(|template| {
            let held = &template.configuration;
            held.signature == candidate.signature && candidate.same_source && held.display_state.values == candidate.display_state.values
                && held.spec == candidate.spec && held.expression_failure == candidate.expression_failure
        });
        if !reusable {
            let previous = self.current_prepared()?;
            self.scene.display_preparation = Some(DataPreparation::new(candidate, previous));
        }
        let settings = self.services.settings();
        let engine = self.services.fit.engine.clone();
        let template = self.scene.display_preparation.as_mut().expect("template");
        template.prepare(input, revision, &mut self.schemas, &settings, &engine)
    }

    /// `_current_prepared`: the accepted answer under the current context.
    pub(crate) fn current_prepared(&self) -> Result<Option<PreparedPlot>> {
        let previous = self.scene.accepted_prepared.clone();
        let evaluation = self.scene.accepted_fit.as_ref().and_then(|accepted| accepted.evaluation.clone());
        if let Some(previous) = &previous {
            let same_evaluation = match (&previous.evaluation, &evaluation) { (None, None) => true, (Some(a), Some(b)) => Arc::ptr_eq(a, b), _ => false };
            if same_evaluation && previous.projection.same_built(&self.scene.projection) { return Ok(Some(previous.clone())); }
        }
        let Some(classifier) = &self.scene.classifier_evaluation else { return Ok(previous) };
        let configuration = self.capture_configuration()?;
        let projection = self.projected();
        Ok(Some(PreparedPlot {
            configuration, classifier: Arc::new(classifier.with_projection(projection.clone())), source_ref: projection.source_ref(), projection,
            evaluation, request: self.scene.accepted_fit.as_ref().map(|accepted| accepted.request.clone()),
            expression_failure: self.scene.fit_expression_failure.clone(),
        }))
    }

    // ------------------------------------------------------------- presentation
    fn scene_snapshot(&self) -> Pending {
        Pending { snapshot: self.scene.clone(), renderer_plan: self.renderer.plan().clone(), presentation: self.renderer.series_presentation(), changed: false, draw_attempted: false, live_fit_batch: None }
    }

    pub(crate) fn pending_presentation(&self) -> Option<()> { self.pending.as_ref().map(|_| ()) }

    /// Hold the current complete scene until its frontend accepts a front.
    pub fn begin_presentation(&mut self) -> Result<()> {
        self.assert_open()?;
        if self.pending.is_none() { self.pending = Some(self.scene_snapshot()); }
        Ok(())
    }

    /// Commit after the frontend lease, or restore the complete scene.
    pub fn finish_presentation(&mut self, accepted: bool, configuration: Option<Arc<ResolvedConfiguration>>) {
        let Some(pending) = self.pending.take() else {
            if configuration.is_some() { self.scene.pending_configuration = configuration; }
            return;
        };
        if configuration.is_some() { self.scene.pending_configuration = configuration.clone(); }
        if accepted { return; }
        if pending.changed {
            let epoch = pending.snapshot.presentation_epoch;
            let batch = pending.live_fit_batch.unwrap_or(0);
            let compose = pending.draw_attempted;
            self.pending = Some(Pending { changed: false, draw_attempted: false, ..self.scene_snapshot() });
            self.restore_scene(pending, compose);
            self.scene.presentation_epoch = epoch;
            self.scene.fit_batch_revision = self.scene.fit_batch_revision.max(batch);
            self.pending = None;
        }
        if configuration.is_some() { self.scene.pending_configuration = configuration; }
    }

    fn restore_scene(&mut self, pending: Pending, compose: bool) {
        let layout_changed = *self.renderer.plan() != pending.renderer_plan || *self.renderer.spec() != pending.snapshot.spec;
        self.scene = pending.snapshot;
        self.renderer.set_spec(self.scene.spec.clone());
        let result = if layout_changed {
            self.apply_layout_plan(pending.renderer_plan, compose, Some(pending.presentation))
        } else {
            self.update_renderer(RenderEffect::LAYOUT | RenderEffect::TEXT | RenderEffect::CHROME | RenderEffect::BASE_STYLE
                                 | RenderEffect::AXIS_TRANSFORM | RenderEffect::OVERLAY, compose, Some(pending.presentation))
        };
        if result.is_err() { let _ = self.renderer.draw(); }
    }

    /// `redraw_surface`.
    pub fn redraw_surface(&mut self) -> Result<()> { self.assert_open()?; self.renderer.draw() }

    pub(crate) fn mark_changed(&mut self, compose: bool) {
        if let Some(pending) = &mut self.pending { pending.changed = true; pending.draw_attempted |= compose; }
    }

    /// `_update_renderer(renderer, effects, compose=, presentation=)`.
    pub(crate) fn update_renderer(&mut self, effects: RenderEffect, compose: bool, presentation: Option<crate::configuration::state::Presentation>) -> Result<()> {
        if let Some(held) = &mut self.held {
            *held = *held | effects;
            return Ok(());
        }
        let compose = compose && !self.for_export;
        self.mark_changed(compose);
        let viewport = match &self.scene.gesture {
            Some(PointerGesture::Pan(pan)) if pan.candidate.is_some() => pan.candidate,
            _ => self.scene.viewport,
        };
        let image = self.semantic_kind() == PlotKind::Image;
        let view_limits = match viewport {
            None => None,
            Some((x, y)) => Some((x.map(|x| self.viewport_x_to_axes(x)).transpose()?.map(|x| (x.low, x.high)),
                                  y.map(|y| if image { (y.high, y.low) } else { (y.low, y.high) }))),
        };
        let (thresholds, labels) = self.classifier_frame_labels()?;
        let accepted = self.scene.accepted_fit.clone();
        let frame = RenderFrame {
            payload: self.scene.projection.payload().clone(), state: self.scene.display.state.clone(), effects,
            data_revision: self.data_revision(),
            fit_overlays: accepted.as_ref().map_or_else(|| Arc::from(Vec::new()), |fit| fit.overlays.clone()),
            fit_model_id: accepted.as_ref().map(|fit| fit.answer.model_id(&fit.request)),
            classifier_overlays: self.scene.classifier_overlays.clone(), classifier_thresholds: thresholds, classifier_labels: labels,
            image_overlay: overlay_of(&self.scene.image_overlay), selectors: self.painted_selector_snapshot()?,
            facet_index: self.scene.focused_facet_index, facet_focus_index: self.scene.facet_focus_index, view_limits, presentation,
        };
        self.renderer.present(&frame, compose)?;
        let interaction = self.renderer.series_interaction();
        self.scene.display.commit_interaction(&interaction);
        if compose { self.scene.presentation_epoch += 1; }
        Ok(())
    }

    pub(crate) fn render_current(&mut self, effects: RenderEffect) -> Result<()> { self.update_renderer(effects, true, None) }

    /// Hold what this session would draw until [`PlotSession::draw_held`].
    pub fn hold_drawing(&mut self) { self.held = Some(RenderEffect::NONE); }

    /// Draw what was held, once; `discard` drops it undrawn.
    pub fn draw_held(&mut self, discard: bool) -> Result<()> {
        match self.held.take() {
            Some(effects) if !discard && effects != RenderEffect::NONE => self.render_current(effects),
            _ => Ok(()),
        }
    }

    /// `_apply_layout_plan(plan, compose=, presentation=)`.
    pub(crate) fn apply_layout_plan(&mut self, plan: SurfacePlan, compose: bool, presentation: Option<crate::configuration::state::Presentation>) -> Result<()> {
        if let Some(pending) = &mut self.pending { pending.changed = true; }
        self.cancel_gesture()?;
        self.renderer.relayout(plan, self.scene.focused_facet_index, self.scene.facet_focus_index)?;
        self.update_renderer(RenderEffect::LAYOUT, compose, presentation)?;
        self.assert_open()
    }

    /// `_resolve_plan`.
    pub(crate) fn resolve_plan(&self) -> Result<SurfacePlan> { surface_plan_for(&self.scene, &self.services, self.device_pixel_ratio) }

    /// `present_configuration(configuration)`: a display-only change of the
    /// accepted answer.
    pub fn present_configuration(&mut self, configuration: Arc<ResolvedConfiguration>) -> Result<()> {
        let current = self.capture_configuration()?;
        if configuration.signature != current.signature { return Err(PlotError::value("configuration requires scientific preparation")); }
        let previous = self.current_prepared()?;
        let input = self.data();
        let settings = self.services.settings();
        let engine = self.services.fit.engine.clone();
        let packet = DataPreparation::new(configuration.clone(), previous)
            .prepare(&input, Some(self.data_revision()), &mut self.schemas, &settings, &engine)?;
        self.present_prepared(packet, Some(configuration), false).map(|_| ())
    }

    /// `present_prepared(prepared, configuration=, live_data=)`: stage one
    /// complete data/configuration/fit/front for the frontend's receipt.
    /// Returns the configuration now presented: the candidate, or what a
    /// grid with fewer cells than it names shows.
    pub fn present_prepared(&mut self, prepared: PreparedPlot, configuration: Option<Arc<ResolvedConfiguration>>, live_data: bool)
                            -> Result<Arc<ResolvedConfiguration>> {
        self.assert_open()?;
        let candidate = configuration.or_else(|| self.scene.pending_configuration.clone()).unwrap_or_else(|| prepared.configuration.clone());
        if candidate.cancelled() || prepared.configuration.cancelled() {
            return Err(PlotError::new(ErrorKind::FitCancelled, "configuration was cancelled"));
        }
        if candidate.signature != prepared.configuration.signature {
            return Err(PlotError::new(ErrorKind::FitCancelled, "prepared scientific intent is no longer current"));
        }
        let mut prepared = prepared;
        if prepared.configuration.display_state.values != candidate.display_state.values || prepared.configuration.spec != candidate.spec {
            let (input, revision) = (prepared.projection.data(), prepared.projection.data_revision());
            let settings = self.services.settings();
            let engine = self.services.fit.engine.clone();
            prepared = DataPreparation::new(candidate.clone(), Some(prepared)).prepare(&input, Some(revision), &mut self.schemas, &settings, &engine)?;
        }
        if self.scene.pending_configuration.as_ref().is_some_and(|pending| pending.signature != candidate.signature) {
            return Err(PlotError::new(ErrorKind::FitCancelled, "prepared configuration was superseded"));
        }
        if let (Some(prepared_ref), PlotInput::Dataset(current)) = (&prepared.source_ref, self.data()) {
            if candidate.same_source && prepared_ref.generation == current.generation && prepared_ref.block_id == current.block_id
                && prepared_ref.revision < current.revision {
                return Err(PlotError::value("prepared data revision moved backwards"));
            }
        }
        self.begin_presentation()?;
        let result = self.present_body(prepared, candidate, live_data);
        if result.is_err() { self.finish_presentation(false, None); }
        result
    }

    fn present_body(&mut self, prepared: PreparedPlot, candidate: Arc<ResolvedConfiguration>, live_data: bool) -> Result<Arc<ResolvedConfiguration>> {
        if !candidate.same_source {
            // An unfinished press or fractional wheel belongs to its source,
            // even when the replacement has identical axes and group names.
            // Ordinary live frames on the same source keep these gestures.
            let interaction = self.services.interaction;
            self.renderer.series_focus(super::renderer::SeriesAction::Clear, None, 0.0, 0.0,
                interaction.selector_handle_radius_px, interaction.double_click_radius_px)?;
        }
        let previous_focus = self.pending.as_ref().expect("presentation begun").snapshot.facet_focus_index;
        let old_state = self.scene.display.state.clone();
        let changed: Vec<&'static str> = candidate.parameter_schema.names()
            .filter(|name| old_state.values.get(name) != candidate.display_state.values.get(name)).collect();
        let effects = if changed.is_empty() { RenderEffect::NONE } else { candidate.parameter_schema.effects_for(changed.iter().copied())? };
        let spec_changed = self.scene.spec != candidate.spec;
        let old_fit = self.scene.accepted_fit.clone();
        let same_fit = match (&old_fit, &prepared.evaluation) {
            (None, None) => true,
            (Some(old), Some(new)) => old.evaluation.as_ref().is_some_and(|held| Arc::ptr_eq(held, new)),
            _ => false,
        };
        // `results is results`: Python's empty tuple is one object.
        let same_classifier = self.scene.classifier_evaluation.as_ref().is_some_and(|held| Arc::ptr_eq(&held.results, &prepared.classifier.results)
            || (held.results.is_empty() && prepared.classifier.results.is_empty()));
        let science_changed = self.scene.accepted_configuration.as_ref().is_none_or(|accepted| accepted.signature != candidate.signature);
        let data_changed = !self.data().same(&prepared.projection.data());
        let selector_key = |states: &[SelectorState]| states.iter().map(|s| (s.kind(), s.value, s.facet_index)).collect::<Vec<_>>();
        let selectors_changed = selector_key(&self.scene.selectors.states()) != selector_key(&candidate.selectors);
        // A waiting panel can preview gestures without changing its accepted
        // configuration. Re-presenting that configuration must restore its
        // actual interaction state before the queued gestures are replayed.
        if candidate.same_source && changed.is_empty() && !spec_changed && !science_changed && !data_changed && same_fit && same_classifier
            && self.scene.viewport == candidate.viewport && !selectors_changed
            && self.scene.focused_facet_index == candidate.focused_facet_index.map(|i| i as usize)
            && self.scene.size == candidate.size && old_state.interaction == candidate.interaction
            && self.renderer.series_presentation() == candidate.presentation
            && self.scene.facet_focus_index == candidate.facet_focus.map(|i| i as usize) && same_overlay(&self.scene.image_overlay, &candidate.image_overlay) {
            self.scene.pending_configuration = Some(candidate.clone());
            if self.scene.fit_expression_failure != prepared.expression_failure {
                self.scene.fit_expression_failure = prepared.expression_failure.clone();
                self.scene.accepted_configuration = Some(candidate.clone());
                self.scene.accepted_prepared = Some(prepared.with_configuration(candidate.clone()));
            }
            return Ok(candidate);
        }
        let interaction_changed = old_state.interaction != candidate.interaction;
        let state = Arc::new(DisplayState {
            revision: old_state.revision + u64::from(!changed.is_empty() || spec_changed || interaction_changed),
            values: candidate.display_state.values.clone(), changed_names: changed.iter().copied().collect(), effects,
            interaction: candidate.interaction.clone(),
        });
        let display = DisplayStoreState { state: state.clone() };
        let mut context = (**prepared.projection.context()).clone();
        context.display_state = state.clone();
        context.selectors = SelectorSnapshot { committed: candidate.selectors.clone(), candidate: None };
        context.viewport = candidate.viewport;
        context.focused_facet_index = candidate.focused_facet_index.map(|i| i as usize);
        let mut projection = prepared.projection.with_context(Arc::new(context));
        if *projection.spec() != candidate.spec { projection = projection.with_spec(candidate.spec.clone()); }
        let (old_schema, new_schema) = (self.data().source().map(|s| s.schema.clone()), projection.data().source().map(|s| s.schema.clone()));
        let geometry_changed = !candidate.same_source || spec_changed
            || matches!((&old_schema, &new_schema), (Some(old), Some(new)) if !structure_equal(old, new));
        let topology_changed = facet_cell_count(&self.scene.spec, self.scene.projection.payload()) != facet_cell_count(&candidate.spec, projection.payload());
        let mut controller = self.scene.selectors.clone();
        if geometry_changed || selectors_changed {
            controller = SelectorController::new();
            for selector in &candidate.selectors { controller.install(*selector)?; }
        }
        let mut accepted_fit = if same_fit && !science_changed { old_fit.clone() } else { None };
        if let (Some(evaluation), None) = (&prepared.evaluation, &accepted_fit) {
            let request = prepared.request.clone().ok_or_else(|| PlotError::runtime("prepared fit has no request"))?;
            let context_generation = self.scene.fit_context_generation + u64::from(science_changed);
            accepted_fit = Some(Arc::new(self.resolved_accepted_fit(evaluation.clone(), request, context_generation, &projection)?));
        }
        if live_data && !same_fit && accepted_fit.is_some() {
            if let Some(pending) = &mut self.pending { pending.live_fit_batch = Some(self.scene.fit_batch_revision); }
        }
        if let Some(pending) = &mut self.pending { pending.changed = true; }
        self.scene.spec = candidate.spec.clone();
        self.scene.parameter_schema = candidate.parameter_schema.clone();
        self.scene.display = display;
        self.scene.projection = projection.clone();
        self.scene.selectors = controller;
        self.scene.viewport = candidate.viewport;
        self.scene.focused_facet_index = candidate.focused_facet_index.map(|i| i as usize);
        self.scene.facet_focus_index = candidate.facet_focus.map(|i| i as usize);
        self.scene.image_overlay = candidate.image_overlay.clone();
        self.scene.size = candidate.size;
        if geometry_changed || topology_changed || selectors_changed { self.scene.gesture = None; }
        self.scene.fit_expression_failure = prepared.expression_failure.clone();
        if science_changed {
            self.scene.fit_context_generation += 1;
            self.scene.fit_request_generation += 1;
        }
        self.scene.accepted_fit = accepted_fit;
        self.scene.live_fit_request = if candidate.fit_live { prepared.request.clone() } else { None };
        if same_fit && self.scene.accepted_fit.is_some() && effects.intersects(RenderEffect::VIEW_PROJECTION) {
            let refreshed = self.refresh_accepted_fit_overlays(self.scene.accepted_fit.as_ref().unwrap())?;
            self.scene.accepted_fit = Some(Arc::new(refreshed));
        }
        let classifier = Arc::new(prepared.classifier.with_projection(projection.clone()));
        self.scene.classifier_thresholds = classifier.capture.thresholds.clone();
        self.scene.classifier_components = classifier.capture.components.clone();
        if same_classifier && !science_changed && !effects.intersects(RenderEffect::VIEW_PROJECTION) {
            self.scene.classifier_evaluation = Some(classifier.clone());
        } else {
            self.install_threshold_classifier(classifier.clone())?;
        }
        let clamped = candidate.spec.is_facet_grid() && self.clamp_facet_state(facet_cell_count(&candidate.spec, projection.payload()).unwrap_or(0))?;
        self.scene.accepted_configuration = Some(candidate.clone());
        let mut accepted_prepared = prepared.with_configuration(candidate.clone());
        accepted_prepared.projection = projection;
        accepted_prepared.classifier = classifier;
        let evaluation = accepted_prepared.evaluation.clone();
        self.scene.accepted_prepared = Some(accepted_prepared);
        self.scene.pending_configuration = Some(candidate.clone());
        let plan = self.resolve_plan()?;
        let relayout = spec_changed || plan != *self.renderer.plan() || previous_focus != self.scene.facet_focus_index;
        self.renderer.set_spec(candidate.spec.clone());
        if relayout {
            self.scene.layout_revision += 1;
            self.apply_layout_plan(plan, true, Some(candidate.presentation.clone()))?;
        } else {
            self.update_renderer(effects | RenderEffect::BASE_GEOMETRY, true, Some(candidate.presentation.clone()))?;
        }
        if clamped {
            // The grid has fewer cells than the candidate names: the
            // accepted configuration is the one the picture shows.
            let shown = self.capture_configuration()?;
            self.scene.accepted_configuration = Some(shown.clone());
            self.scene.pending_configuration = Some(shown);
        }
        if let Some(evaluation) = evaluation { self.remember_fit_warm_starts(&evaluation, self.scene.fit_request_generation); }
        Ok(self.scene.accepted_configuration.clone().expect("accepted configuration"))
    }

    // ------------------------------------------------------------- facets
    pub(crate) fn facet_cell_count(&self) -> Option<usize> { facet_cell_count(&self.scene.spec, self.scene.projection.payload()) }

    /// `_clamp_facet_state(cell_count)`: whether anything moved.
    pub(crate) fn clamp_facet_state(&mut self, cell_count: usize) -> Result<bool> {
        let (selected, opened) = if cell_count == 0 { (None, None) } else {
            let selected = match self.scene.focused_facet_index { None => cell_count - 1, Some(index) => index.min(cell_count - 1) };
            (Some(selected), self.scene.facet_focus_index.map(|_| selected))
        };
        if selected == self.scene.focused_facet_index && opened == self.scene.facet_focus_index { return Ok(false); }
        self.scene.focused_facet_index = selected;
        self.scene.facet_focus_index = opened;
        self.scene.viewport = None;
        if let Some(selected) = selected { self.retarget_selectors(selected)?; }
        Ok(true)
    }

    /// `_retarget_selectors(index)`: the shared selectors go with the cell.
    pub(crate) fn retarget_selectors(&mut self, index: usize) -> Result<()> {
        let kinds: Vec<SelectorKind> = SelectorKind::ALL.into_iter()
            .filter(|kind| *kind != SelectorKind::Threshold || !self.threshold_classifier_enabled()).collect();
        self.scene.selectors = self.scene.selectors.retarget_facet(index as u64, &kinds)?;
        Ok(())
    }

    pub(crate) fn threshold_classifier_enabled(&self) -> bool { accepts_classifier_thresholds(&self.scene.spec, &self.scene.display.state.values) }

    // ------------------------------------------------------------- recipe and description
    /// `_configured_selectors`: the committed non-threshold selectors.
    pub fn configured_selectors(&self) -> Result<Vec<SelectorState>> {
        Ok(self.resolved_selector_snapshot()?.committed.into_iter().filter(|state| state.kind() != SelectorKind::Threshold).collect())
    }

    /// `_figure_recipe`: the prepared scientific/configuration state as a recipe.
    pub fn figure_recipe(&self) -> Result<ValueMap> {
        self.assert_open()?;
        let thresholds = self.classifier_threshold_targets_state(true)?;
        let fit = self.fit_configuration();
        let selectors = self.configured_selectors()?;
        let presentation = self.renderer.series_presentation();
        encode_plot_recipe(&self.scene.spec, &self.scene.display.state.values, self.renderer.plan().preset, &RecipeParts {
            viewport: self.scene.viewport, classifier_thresholds: &thresholds, facet_focus: self.scene.facet_focus_index.map(|i| i as i128),
            fit: Some(&fit), selectors: &selectors, interaction: Some(&self.scene.display.state.interaction), presentation: Some(&presentation),
        })
    }

    /// The selection subject of the accepted projection.
    pub fn selection_subject(&self) -> Result<crate::projection::output::SelectionSubject> {
        let window = self.scene.display.state.values.get("window").and_then(|value| match value { Value::Int(v) => Some(*v as usize), _ => None });
        self.scene.projection.selection_subject(window)
    }

    // ------------------------------------------------------------- raster
    /// `_raster_axes_snapshot`: every visible axes as a pointer transform.
    pub fn raster_axes_snapshot(&self) -> Result<Vec<AxisTransform>> {
        self.assert_open()?;
        let projected = self.projected();
        let canvas = self.renderer.canvas_size();
        self.renderer.visible_axes().iter().map(|geometry| self.axis_transform(geometry, &projected, canvas)).collect()
    }

    fn axis_transform(&self, geometry: &super::renderer::AxesGeometry, projected: &FitProjection, canvas: CanvasSize) -> Result<AxisTransform> {
        let role = geometry.id.role.clone();
        // A pulse timeline has no view: its axes are their own canonical frame.
        let (canonical_x, canonical_y) = if projected.is_pulse() { (geometry.x_limits, geometry.y_limits) }
            else { canonical_axes_limits(&role, geometry.x_limits, geometry.y_limits, projected)? };
        let mut canonical_scales = [None, None];
        if !projected.is_pulse() && matches!(role.as_str(), "main" | "image" | "history" | "facet_cell") {
            let y_source = if projected.is_histogram_plot() { None } else { Some(projected.y_quantity()?) };
            let sources = [(Some(projected.x_selector_quantity()?), &geometry.x_scale), (y_source, &geometry.y_scale)];
            for (index, (source, scale)) in sources.into_iter().enumerate() {
                let Some(quantity) = source else { continue };
                if *quantity.canonical != *quantity.display && (matches!(scale, AxisScale::Samples(_)) || !(quantity.canonical.is_linear() && quantity.display.is_linear())) {
                    canonical_scales[index] = Some(AxisScale::Converted { mapping: Box::new(scale.clone()), canonical: quantity.canonical, display: quantity.display });
                }
            }
        }
        let (x0, y0, x1, y1) = geometry.bbox;
        let [canonical_x_scale, canonical_y_scale] = canonical_scales;
        Ok(AxisTransform {
            role, cell_index: geometry.id.cell_index,
            bounds: (x0 / canvas.width, 1.0 - y1 / canvas.height, x1 / canvas.width, 1.0 - y0 / canvas.height),
            x_limits: geometry.x_limits, y_limits: geometry.y_limits, canonical_x_limits: canonical_x, canonical_y_limits: canonical_y,
            x_scale: geometry.x_scale.clone(), y_scale: geometry.y_scale.clone(), canonical_x_scale, canonical_y_scale,
        })
    }

    /// The current front.
    pub fn rgba(&mut self) -> Result<super::renderer::RgbaFront> { self.assert_open()?; self.renderer.rgba() }

    /// `close`.
    pub fn close(&mut self) {
        if self.closed { return; }
        self.closed = true;
        let _ = self.cancel_gesture();
        self.scene.fit_context_generation += 1;
        self.scene.fit_request_generation += 1;
        self.scene.fit_warm_starts.clear();
        self.scene.live_fit_request = None;
    }

    /// The renderer's id of the primary axes.
    pub(crate) fn primary_axes(&self) -> AxesId { self.renderer.primary_axes() }
}

/// `_canonical_axes_limits(role, x_limits, y_limits, projected)`.
fn canonical_axes_limits(role: &str, x: (f64, f64), y: (f64, f64), projected: &FitProjection) -> Result<((f64, f64), (f64, f64))> {
    if role == "distribution" {
        let quantity = projected.value_quantity();
        return Ok(((quantity.to_canonical(y.0), quantity.to_canonical(y.1)), (0.0, 0.0)));
    }
    let xq = projected.x_selector_quantity()?;
    let x_values = (xq.to_canonical(x.0), xq.to_canonical(x.1));
    let y_values = if projected.is_histogram_plot() { y } else { let yq = projected.y_quantity()?; (yq.to_canonical(y.0), yq.to_canonical(y.1)) };
    Ok((x_values, y_values))
}

/// How many cells a FacetGrid payload has; `None` for any other kind.
pub(crate) fn facet_cell_count(spec: &PlotSpec, payload: &Payload) -> Option<usize> {
    if !spec.is_facet_grid() { return None; }
    Some(match payload { Payload::Facet(facet) => facet.cells.len(), _ => 0 })
}

/// What a schema IS without what its coordinates currently read
/// (`structure_fingerprint`): two schemas with equal keys have one geometry.
pub(crate) fn structure_key(schema: &DatasetSchema) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    for (role, coded) in [(DomainRole::Repeat, true), (DomainRole::Point, true), (DomainRole::Cell, false)] {
        let domain = schema.domain(role);
        let sliding: Vec<&str> = domain.axes().iter().filter(|axis| axis.role() == PRIMARY_INDEX).map(|axis| axis.id()).collect();
        for axis in domain.axes() {
            let (coordinates, origin, origins) = match axis.coordinates() {
                Coordinates::Index { origin, .. } => (None, *origin, None),
                Coordinates::Numeric(buffer) => (Some(buffer.len()), 0, None),
                Coordinates::Values(values) => (Some(values.len()), 0, None),
                Coordinates::OriginsAndOffsets { origins, offsets } => (Some(offsets.len()), 0, Some(origins.len())),
            };
            let slides = !sliding.is_empty() && (sliding.contains(&axis.id()) || axis.coordinate_of().is_some_and(|of| sliding.contains(&of))
                                                  || origins.is_some());
            let labels = axis.labels().map(<[_]>::len);
            if slides {
                let coordinates = if origins.is_none() && coordinates.is_some() { "sliding".to_owned() } else { format!("{coordinates:?}") };
                let _ = write!(out, "[{} {} {} sliding {coordinates} sliding {:?} {:?}]", axis.id(), axis.name(), axis.role(),
                               labels.map(|_| "sliding"), origins.map(|_| "sliding"));
            } else {
                let _ = write!(out, "[{} {} {} {} {coordinates:?} {origin} {labels:?} {origins:?}]", axis.id(), axis.name(), axis.role(), axis.len());
            }
            let _ = write!(out, "{:?} {:?} {:?};", axis.unit(), axis.coordinate_frame(), axis.coordinate_of());
        }
        for (index, size) in domain.physical_shape().iter().enumerate() {
            let slides = !sliding.is_empty() && (coded || domain.axes().get(index).is_some_and(|axis| sliding.contains(&axis.id())));
            if slides { out.push_str("shape sliding;") } else { let _ = write!(out, "shape {size};"); }
        }
        if !coded { out.push_str("codes none;"); }
        else if !sliding.is_empty() { out.push_str("codes sliding;"); }
        else {
            for axis in domain.axes() {
                let mapping = domain.mapping(axis.id()).expect("mapped axis");
                let base = match mapping.base() {
                    CodeBase::Range { start, step, len } => format!("range {start} {step} {len}"),
                    CodeBase::Explicit(values) => format!("{:?}", values.as_slice()),
                };
                let _ = write!(out, "code {base} {} {};", mapping.inner(), mapping.outer());
            }
        }
    }
    let value = schema.value();
    let validity = match value.validity() { ValidityContract::Value => "value".to_owned(), ValidityContract::Components(ids) => format!("{ids:?}") };
    let _ = write!(out, "value {:?} {:?} {:?} {validity}", schema.dtype(), value.unit(), value.name());
    out
}

fn structure_equal(a: &Arc<DatasetSchema>, b: &Arc<DatasetSchema>) -> bool { Arc::ptr_eq(a, b) || structure_key(a) == structure_key(b) }

/// `_surface_plan_for(spec, payload, state)`.
pub(crate) fn surface_plan_for(scene: &Scene, services: &SessionServices, device_pixel_ratio: f64) -> Result<SurfacePlan> {
    let spec = &scene.spec;
    let state = &scene.display.state.values;
    let semantic_image = spec.semantic().kind() == PlotKind::Image;
    let topology = spec.is_facet_grid().then(|| FacetTopology {
        cell_count: facet_cell_count(spec, scene.projection.payload()).unwrap_or(0).max(1) as u32,
        cell_height_over_width: semantic_image.then_some(1.0),
    });
    let flag = |name: &str| matches!(state.get(name), Some(Value::Bool(true)));
    let options = SurfaceOptions {
        facet_topology: topology, device_pixel_ratio: Some(device_pixel_ratio), export_scale: None,
        rolling_side_distribution: (spec.kind() == PlotKind::Rolling).then(|| flag("side_distribution")),
        image_height_over_width: (spec.kind() == PlotKind::Image).then_some(1.0),
        image_scene: semantic_image && state.get("presentation").is_some_and(|value| value.py_str() == "height_bars"),
        image_side_distribution: Some(!semantic_image || flag("side_distribution")),
    };
    resolve_surface(Some(scene.size), layout_kind(spec.kind()), &options, &services.defaults.layout, &services.defaults.style.fonts)
        .map_err(PlotError::value)
}

/// A wire value of the fit summary.
pub(crate) fn finite_or_null(value: f64) -> Json { if value.is_finite() { json!(value) } else { Json::Null } }
