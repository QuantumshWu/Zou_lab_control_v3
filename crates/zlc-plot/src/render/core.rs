//! The renderer core: one fixed-layout figure with persistent artists
//! (`MatplotlibRenderer`): figure and axes from the SurfacePlan, relayout,
//! `present` orchestration, pixel-snapped boxes, home and requested views,
//! labels, titles, grid, the tick-label fitting pass, and the materialized IR.
//!
//! A kind (Curve, Rolling, ...) is a [`Kind`] whose `render` updates the
//! artists of one painted surface through the core's shared helpers.
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};

use zlc_figure::ir;

use super::curve::CurveTables;
use super::facet::FacetTables;
use super::fonts::{FontSet, TextToPath};
use super::frame::{RenderFrame, ViewLimits};
use super::height_bars::HeightBars;
use super::image::ImageTables;
use super::geometry;
use super::labels::state_label;
use super::limits::{autoscaled_limits, relim_retains, scalar_close, select_display_limits};
use super::mpl::axes::{AxesModel, Child, ChildId, TitleText};
use super::mpl::figure::FigureModel;
use super::mpl::rc::Rc;
use super::rolling::RollingTables;
use crate::configuration::spec::Surface;
use crate::configuration::{DisplayState, PlotSpec, RenderEffect, Value};
use crate::kinds::RelimMode;
use crate::layout::{AxesPlan, SurfacePlan};
use crate::projection::output::Payload;
use crate::style::PlotStyle;

/// What one plot kind draws on its surface.
pub trait Kind: Send + Sync {
    /// `handler.render(renderer, payload, state, axes=..., key=...)`.
    fn render(&self, renderer: &mut Renderer, payload: &Arc<Payload>, state: &DisplayState, axes: usize, key: &str);
    /// The surface key (`handler.kind.value`).
    fn key(&self) -> &'static str;
}

/// An overlay layer (classifier, selectors, height-bar cage, fit, image point
/// overlay): updated in `present` where `MatplotlibRenderer.present` updates
/// it -- after the requested view is applied, before the boxes settle and
/// the tick typography is synced. Its retained state lives on the renderer.
pub trait Layer: Send + Sync {
    fn update(&self, renderer: &mut Renderer, frame: &RenderFrame);
}

/// The kind that draws `spec`, when it is one this renderer knows.
pub fn kind_for(spec: &PlotSpec) -> Option<Arc<dyn Kind>> {
    match spec {
        PlotSpec::Curve(_) => Some(Arc::new(super::curve::CurveKind)),
        PlotSpec::Rolling(_) => Some(Arc::new(super::rolling::RollingKind)),
        PlotSpec::PulseTimeline(_) => Some(Arc::new(super::pulse::PulseKind)),
        PlotSpec::Histogram(_) => Some(Arc::new(super::histogram::HistogramKind)),
        PlotSpec::Image(_) => Some(Arc::new(super::image::ImageKind)),
        PlotSpec::FacetGrid(grid) if super::facet::draws_cell(&grid.cell) => Some(Arc::new(super::facet::FacetKind)),
        _ => None,
    }
}

/// The automatic limits one axis retains between frames (`_resolve_data_limits`).
#[derive(Clone, Debug, PartialEq)]
pub struct LimitMemo { pub limits: Option<(f64, f64)>, pub asked: Option<(RelimMode, bool)> }

#[derive(Clone)]
pub struct Renderer {
    pub spec: PlotSpec,
    pub plan: SurfacePlan,
    pub style: Arc<PlotStyle>,
    pub rc: Arc<Rc>,
    pub measure: Arc<TextToPath>,
    /// The FT2Fonts tick-label boxes are measured with (shared by copies).
    pub fonts: Arc<Mutex<FontSet>>,
    pub fig: FigureModel,
    /// `_axes`: role -> axes indices, in plan order of first appearance.
    pub roles: Vec<(String, Vec<usize>)>,
    /// The kind that draws the spec; `None` for a kind this renderer does not draw.
    pub(crate) kind: Option<Arc<dyn Kind>>,
    pub last_payload: Option<Arc<Payload>>,
    pub last_state: Option<Arc<DisplayState>>,
    pub data_revision: Option<u64>,
    pub requested_view: Option<ViewLimits>,
    pub home_limits: HashMap<usize, ((f64, f64), (f64, f64))>,
    /// Painted fit overlays of the last frame (the fit layer's).
    pub fit_overlays: usize,
    pub(crate) quantized_bounds: HashMap<usize, [f64; 4]>,
    pub(crate) box_exact: HashMap<usize, bool>,
    pub(crate) planned_ratio: HashMap<usize, Option<f64>>,
    pub(crate) owned_box: HashMap<usize, bool>,
    pub(crate) owned_axes: HashSet<usize>,
    pub(crate) chrome_dirty: HashSet<usize>,
    pub(crate) limits: HashMap<String, LimitMemo>,
    pub(crate) ticks_layout: Option<u64>,
    pub(crate) curve: CurveTables,
    pub(crate) rolling: RollingTables,
    pub(crate) pulse: super::pulse::PulseTables,
    pub(crate) histogram: super::histogram::HistogramTables,
    pub(crate) image: ImageTables,
    pub(crate) height_bars: HeightBars,
    pub(crate) facet: FacetTables,
    pub raster_generation: u64,
    pub composed_generation: i64,
    /// Inside `present` (the style context): artists made outside it take
    /// matplotlib's rc defaults.
    pub(crate) in_style: bool,
    /// The axes a view drag is confined to (`_confined_gesture_axes`).
    pub confined: Option<usize>,
    /// The selector kind a gesture holds (`_selector_gesture_kind`).
    pub selector_gesture: Option<crate::configuration::selectors::SelectorKind>,
    /// The last composed front.
    pub front: Option<Arc<super::front::Front>>,
    /// The last drawn chrome background: its IR hash and pixels.
    pub(crate) background: Option<(u64, Arc<Vec<u8>>)>,
    /// The canvas under an image's point overlay, kept for a frame that changes only the rings.
    pub(crate) under: Option<super::front::Under>,
    /// The last present kept its payload: a re-present of the same data.
    pub(crate) payload_kept: bool,
    /// The boundary chrome strokes the compose lowered, for the next one.
    pub(crate) lowered: Arc<Mutex<super::foreground::LoweredMemo>>,
    /// The buffers fronts are published in, recycled once their readers release them.
    pub(crate) front_pool: Arc<zlc_core::BufferPool<u8>>,
    /// The canvas backend of the last compose (its scratch state; the pixels are the fronts').
    pub(crate) canvas: super::replay::Retained<zlc_figure::agg::AggBackend>,
    /// A grid chrome's recorded calls.
    pub(crate) chrome_calls: super::replay::Retained<super::replay::Recordings>,
    /// Artists the overlay layers own (selectors, fit, classifier), in their
    /// insertion order: dynamic in the compose, after the series' own.
    pub overlay_artists: Vec<(usize, ChildId)>,
    /// The fit lines among them, which the stroke kernels paint when eligible.
    pub fit_lines: Vec<(usize, ChildId)>,
    /// The overlay layers, in `present`'s order.
    pub layers: Vec<Arc<dyn Layer>>,
    /// The fit layer's topologies and source scatter.
    pub fit: super::fit::FitTables,
    /// The selector layer's artists and gesture candidate.
    pub selectors: super::selectors::SelectorTables,
    /// The classifier layer's artists and readings.
    pub classifier: super::classifier::ClassifierTables,
}

/// Python's `bool(effects & flags)`.
fn any(effects: RenderEffect, flags: RenderEffect) -> bool { effects.intersects(flags) }

fn or(a: RenderEffect, b: RenderEffect) -> RenderEffect { RenderEffect(a.0 | b.0) }

impl Renderer {
    /// `MatplotlibRenderer(spec, plan, style=style)`.
    pub fn new(spec: PlotSpec, plan: SurfacePlan, style: Arc<PlotStyle>) -> Renderer {
        let kind = kind_for(&spec);
        let rc = Arc::new(Rc::new(&style));
        let measure = Arc::new(TextToPath::new(rc.font_family.clone()));
        let fig = FigureModel::new([plan.figure_size_inches.0, plan.figure_size_inches.1], plan.dpi);
        let mut renderer = Renderer {
            spec, plan, style, rc, measure, fonts: Arc::new(Mutex::new(FontSet::new())), fig, roles: Vec::new(), kind,
            last_payload: None, last_state: None, data_revision: None, requested_view: None,
            home_limits: HashMap::new(), fit_overlays: 0,
            quantized_bounds: HashMap::new(), box_exact: HashMap::new(), planned_ratio: HashMap::new(),
            owned_box: HashMap::new(), owned_axes: HashSet::new(), chrome_dirty: HashSet::new(),
            limits: HashMap::new(), ticks_layout: None,
            curve: CurveTables::default(), rolling: RollingTables::default(), histogram: Default::default(), pulse: Default::default(),
            image: ImageTables::default(), height_bars: HeightBars::default(), facet: FacetTables::default(),
            raster_generation: 0, composed_generation: -1, in_style: false, confined: None,
            selector_gesture: None, front: None, background: None, under: None, payload_kept: false, lowered: Default::default(), front_pool: Default::default(),
            canvas: Default::default(), chrome_calls: Default::default(),
            overlay_artists: Vec::new(), fit_lines: Vec::new(),
            layers: vec![Arc::new(super::classifier::ClassifierLayer), Arc::new(super::selectors::SelectorLayer), Arc::new(super::fit::FitLayer)],
            fit: Default::default(), selectors: Default::default(), classifier: Default::default(),
        };
        renderer.create_axes(VecDeque::new(), true);
        renderer
    }

    /// `_create_axes(figure, plan, reserved)`: grid cells come back from
    /// `kept` (a relayout keeps them, policy and all), else from the reserve
    /// (`reserve`: the constructor's) or plain `add_axes`.
    fn create_axes(&mut self, mut kept: VecDeque<AxesModel>, reserve: bool) {
        self.fig.axes.clear();
        self.roles.clear();
        let plans: Vec<AxesPlan> = self.plan.axes.clone();
        for axes_plan in plans {
            let bounds = axes_plan.r#box.matplotlib_bounds();
            let mut axes = match (axes_plan.role == "facet_cell", kept.pop_front()) {
                (true, Some(mut axes)) => {
                    axes.owned_box = false;
                    axes.aspect = None;
                    axes.anchor = "C";
                    axes.set_position(bounds);
                    axes
                }
                (true, None) if reserve => self.grid_cell(bounds),
                _ => AxesModel::new(&self.rc, bounds),
            };
            axes.room = Some(axes_plan.room);
            axes.role = axes_plan.role.clone();
            axes.cell_index = axes_plan.cell_index;
            axes.gid = match axes_plan.cell_index {
                Some(index) => format!("{}:{index}", axes_plan.role),
                None => axes_plan.role.clone(),
            };
            self.fig.axes.push(axes);
            let index = self.fig.axes.len() - 1;
            if axes_plan.role == "facet_cell" {
                self.fig.axes[index].axison = false;
                self.clear_cell_title(index);
            }
            match self.roles.iter_mut().find(|(role, _)| *role == axes_plan.role) {
                Some((_, list)) => list.push(index),
                None => self.roles.push((axes_plan.role.clone(), vec![index])),
            }
        }
    }

    /// `set_view_dragging(axes)`: a hand drags this axes' view; its Axis
    /// becomes dynamic for the length of the gesture.
    pub fn set_view_dragging(&mut self, axes: Option<usize>) {
        self.composed_generation = -1;
        self.confined = axes;
    }

    /// The session replaced the specification (same layout, or before a relayout).
    pub fn set_spec(&mut self, spec: PlotSpec) {
        self.kind = kind_for(&spec);
        self.spec = spec;
    }

    /// Whether this renderer draws its spec's kind.
    pub fn draws_kind(&self) -> bool { self.kind.is_some() }

    pub fn axes_of(&self, role: &str) -> &[usize] {
        self.roles.iter().find(|(r, _)| r == role).map(|(_, list)| list.as_slice()).unwrap_or(&[])
    }

    /// `relayout(plan, facet_index=..., facet_focus_index=...)`: a new
    /// layout; grid cells are kept (their children go, their tick policy
    /// stays with its settled step forgotten).
    pub fn relayout(&mut self, plan: SurfacePlan, facet_index: Option<usize>, facet_focus_index: Option<usize>)
        -> crate::configuration::error::Result<()> {
        let cells = self.facet_cells();
        let mut old = std::mem::replace(&mut self.fig, FigureModel::new([plan.figure_size_inches.0, plan.figure_size_inches.1], plan.dpi));
        let kept: VecDeque<AxesModel> = cells.into_iter().map(|index| {
            let mut axes = std::mem::replace(&mut old.axes[index], AxesModel::new(&self.rc, [0.0, 0.0, 1.0, 1.0]));
            axes.children.clear();
            axes.strokes.clear();
            for x in [true, false] { axes.axis_mut(x).forget_settled_step(); }
            axes
        }).collect();
        self.plan = plan;
        self.create_axes(kept, false);
        self.facet.chrome = None;
        self.facet.chrome_signature = None;
        self.facet.outer = None;
        self.facet.title = None;
        self.curve = CurveTables::default();
        self.rolling = RollingTables::default();
        self.pulse = Default::default();
        self.histogram = Default::default();
        self.image = ImageTables::default();
        self.facet.images.clear();
        self.facet.image_key = None;
        self.facet.image_scene = None;
        self.facet.image_signature = None;
        self.facet.focus_chrome = None;
        self.facet.fit_native = None;
        // The scene's artists and render cache went with the axes; its last scene stays.
        self.height_bars.call = None;
        self.height_bars.chrome = None;
        self.height_bars.cage = None;
        self.height_bars.readout = None;
        self.height_bars.axes = None;
        self.height_bars.engine = Default::default();
        self.fit = Default::default();
        self.selectors = Default::default();
        self.classifier = Default::default();
        self.overlay_artists.clear();
        self.fit_lines.clear();
        self.limits.clear();
        self.ticks_layout = None;
        self.owned_box.clear();
        self.owned_axes.clear();
        self.box_exact.clear();
        self.planned_ratio.clear();
        self.quantized_bounds.clear();
        self.data_revision = None;
        self.last_payload = None;
        self.last_state = None;
        self.home_limits.clear();
        self.requested_view = None;
        self.chrome_dirty.clear();
        self.confined = None;
        self.check_facet_presentation(facet_index, facet_focus_index, None)?;
        self.consume_facet_presentation(facet_index, facet_focus_index, None);
        Ok(())
    }

    /// `_whole_surface()`: (kind key, axes).
    fn whole_surface(&self) -> (String, usize) {
        let key = self.kind.as_ref().map_or("", |kind| kind.key()).to_owned();
        for role in ["main", "history", "image", "facet_cell"] {
            if let Some(&first) = self.axes_of(role).first() { return (key, first); }
        }
        (key, self.roles.first().map(|(_, list)| list[0]).unwrap_or(0))
    }

    /// `primary_surface`: (key, axes) of the one surface gestures act on.
    pub fn primary_surface(&self) -> (String, usize) {
        let surfaces = self.painted_surfaces();
        if surfaces.is_empty() { return self.whole_surface(); }
        let index = self.facet.focused.unwrap_or(0).min(surfaces.len() - 1);
        let (key, axes, _) = surfaces.into_iter().nth(index).unwrap();
        (key, axes)
    }

    pub fn primary_axes(&self) -> usize { self.primary_surface().1 }

    /// `painted_surfaces`: (key, axes, cell) of every data surface on screen.
    pub fn painted_surfaces(&self) -> Vec<(String, usize, Option<usize>)> {
        if self.is_facet() { return self.facet_surfaces(); }
        let (key, axes) = self.whole_surface();
        vec![(key, axes, None)]
    }

    /// `interactive_axes_at(x=..., y=...)` in figure pixels (bottom-left origin).
    pub fn interactive_axes_at(&self, x: f64, y: f64) -> Option<usize> {
        let contains = |index: usize| {
            let axes = &self.fig.axes[index];
            let b = axes.bbox_exact(self.fig.size_inches, self.fig.dpi);
            axes.visible && ((b[0] <= x && x <= b[2]) || (b[0] >= x && x >= b[2])) && ((b[1] <= y && y <= b[3]) || (b[1] >= y && y >= b[3]))
        };
        let primary = self.primary_axes();
        if contains(primary) { return Some(primary); }
        (0..self.fig.axes.len()).rev().find(|&index| contains(index))
    }

    // ------------------------------------------------------------- present
    /// `present(frame)`: prepare every scene layer of one frame.
    pub fn present(&mut self, frame: &RenderFrame) {
        let _span = super::profile::span("present");
        self.in_style = true;
        self.present_in_style(frame);
        self.in_style = false;
    }

    fn present_in_style(&mut self, frame: &RenderFrame) {
        let state = frame.state.clone();
        let effects = frame.effects;
        self.composed_generation = -1;
        let fresh_axes = self.last_state.is_none();
        let state_changed = fresh_axes || Some(state.revision) != self.last_state.as_ref().map(|s| s.revision);
        let payload = Some(frame.payload.clone());
        let payload_changed = !same_payload(&payload, &self.last_payload) || Some(frame.data_revision) != self.data_revision;
        let base_effects = or(or(RenderEffect::VIEW_PROJECTION, RenderEffect::PAYLOAD_PROJECTION),
                              or(RenderEffect::BASE_GEOMETRY, RenderEffect::LAYOUT));
        let mut base_changed = payload_changed || any(effects, base_effects);
        self.payload_kept = !payload_changed;
        if state_changed && any(effects, RenderEffect::AXIS_TRANSFORM) { base_changed = true; }
        // An image front is viewport-sized: a view edit changes its crop.
        if any(effects, RenderEffect::AXIS_TRANSFORM) && matches!(self.spec.semantic(), Surface::Image(_)) { base_changed = true; }
        let style_only = !base_changed && state_changed && any(effects, RenderEffect::BASE_STYLE);
        let visible_cells = match &*frame.payload { Payload::Facet(facet) if self.is_facet() => Some(facet.cells.len()), _ => self.is_facet().then_some(0) };
        self.consume_facet_presentation(frame.facet_index, frame.facet_focus_index, visible_cells);
        self.requested_view = frame.view_limits;
        self.last_payload = payload.clone();
        self.last_state = Some(state.clone());
        self.data_revision = Some(frame.data_revision);
        self.fit_overlays = frame.fit_overlays.len();
        let painted = self.painted_surfaces();
        let positioned = super::profile::span("present.position");
        self.position_axes_for_frame();
        drop(positioned);
        let plotted = super::profile::span("present.plot");
        if base_changed {
            if let Some(payload) = payload.clone() { self.update_plot(&payload, &state); }
                for &(_, axes, _) in &painted { self.capture_home_limits(axes); }
        } else if style_only {
            if let Some(payload) = self.last_payload.clone() { self.update_plot(&payload, &state); }
            for (_, axes, _) in self.painted_surfaces() { self.capture_home_limits(axes); }
        }
        drop(plotted);
        // A kind's refusal ends the present where Python's raise would.
        if self.pulse.failure.is_some() { return; }
        let interaction = self.series_interaction();
        if frame.presentation.is_some() || fresh_axes || state.interaction != interaction {
            let presentation = frame.presentation.clone();
            self.restore_series_interaction(&state.interaction, presentation.as_ref());
        }
        if fresh_axes || (state_changed && any(effects, RenderEffect::TEXT)) {
            if let Some(payload) = payload.clone() { self.update_text_artists(&payload, &state); }
        }
        if fresh_axes || (state_changed && any(effects, RenderEffect::CHROME)) {
            self.apply_colorbar_visibility(&state);
            self.apply_grid(&state);
        }
        // A height-bar scene owns its whole data region: no 2D view applies.
        let scene_3d = self.settle_height_bars_scene(&state);
        if !scene_3d {
            for &(_, axes, _) in &painted {
                let requested = frame.view_limits;
                self.apply_requested_view(axes, requested);
            }
        }
        let layered = super::profile::span("present.layers");
        for layer in self.layers.clone() { layer.update(self, frame); }
        drop(layered);
        self.update_height_bars_cage(&if scene_3d { frame.selectors.states() } else { Vec::new() });
        self.settle_owned_boxes();
        self.update_image_point_overlay(if scene_3d { None } else { frame.image_overlay.as_ref() }, &state);
        self.sync_axes_chrome();
        let _settled = super::profile::span("present.path_labels");
        self.settle_path_labels();
    }

    fn update_plot(&mut self, payload: &Arc<Payload>, state: &DisplayState) {
        let (key, axes) = self.primary_surface();
        if let Some(kind) = self.kind.clone() { kind.render(self, payload, state, axes, &key); }
    }

    fn capture_home_limits(&mut self, axes: usize) {
        // An image registered its whole source extent before the frame's viewport.
        if matches!(self.spec.semantic(), Surface::Image(_)) && self.home_limits.contains_key(&axes) { return; }
        let a = &self.fig.axes[axes];
        self.home_limits.insert(axes, (a.xlim, a.ylim));
    }

    /// `_resolved_view_limits`.
    pub fn resolved_view_limits(&self, axes: usize, requested: Option<ViewLimits>) -> ((f64, f64), (f64, f64)) {
        let home = self.home_limits.get(&axes).copied().unwrap_or((self.fig.axes[axes].xlim, self.fig.axes[axes].ylim));
        match requested {
            None => home,
            Some((x, y)) => (x.unwrap_or(home.0), y.unwrap_or(home.1)),
        }
    }

    /// `axis_scale(axis, name)`: how the axis maps a value to its position.
    pub fn axis_scale(&self, axes: usize, x: bool) -> super::scale::AxisScale {
        let axis = self.fig.axes[axes].axis(x);
        match &axis.function {
            Some(lattice) => super::scale::AxisScale::Samples(lattice.clone()),
            None if axis.log => super::scale::AxisScale::Log,
            None => super::scale::AxisScale::Linear,
        }
    }

    fn apply_requested_view(&mut self, axes: usize, requested: Option<ViewLimits>) {
        let (x, y) = self.resolved_view_limits(axes, requested);
        self.set_xlim(axes, x);
        self.set_ylim(axes, y);
    }

    /// An `axes.plot` / `axes.scatter` on `axes`: its autoscale request.
    pub fn plotted(&mut self, axes: usize) {
        let margins = self.rc.axes_margins;
        self.fig.axes[axes].request_autoscale(margins);
    }

    /// `_set_xlim`: only a real change moves the limits (and re-envelopes
    /// the dense lines of the axes).
    pub fn set_xlim(&mut self, axes: usize, limits: (f64, f64)) {
        let current = self.fig.axes[axes].xlim;
        if !(scalar_close(current.0, limits.0) && scalar_close(current.1, limits.1)) {
            self.fig.axes[axes].set_xlim(limits.0, limits.1);
            self.chrome_dirty.insert(axes);
            self.refresh_enveloped_lines(axes);
        }
    }

    pub fn set_ylim(&mut self, axes: usize, limits: (f64, f64)) {
        let current = self.fig.axes[axes].ylim;
        if !(scalar_close(current.0, limits.0) && scalar_close(current.1, limits.1)) {
            self.fig.axes[axes].set_ylim(limits.0, limits.1);
            self.chrome_dirty.insert(axes);
        }
    }

    // ---------------------------------------------------------------- boxes
    /// `_position_axes_for_frame`: every standalone axes box on whole pixels.
    fn position_axes_for_frame(&mut self) {
        let mut plans_by_role: BTreeMap<String, Vec<AxesPlan>> = BTreeMap::new();
        for plan in &self.plan.axes { plans_by_role.entry(plan.role.clone()).or_default().push(plan.clone()); }
        let roles = self.roles.clone();
        for (role, list) in roles {
            if role == "facet_cell" { continue; }
            let plans = plans_by_role.get(&role).cloned().unwrap_or_default();
            for (axes, plan) in list.into_iter().zip(plans) {
                let ratio = self.planned_image_box_ratio(&role);
                let bounds = self.pixel_quantized_bounds(axes, plan.r#box.matplotlib_bounds(), ratio);
                let applied = self.quantized_bounds.get(&axes).copied();
                let differs = applied.is_none_or(|applied| applied.iter().zip(&bounds).any(|(now, wanted)| (now - wanted).abs() > 1e-12));
                if differs {
                    self.fig.axes[axes].set_position(bounds);
                    self.quantized_bounds.insert(axes, bounds);
                    self.chrome_dirty.insert(axes);
                }
            }
        }
    }

    /// `_pixel_quantized_bounds` (the planned ratio, else the drawn one).
    pub(crate) fn pixel_quantized_bounds(&mut self, axes: usize, bounds: [f64; 4], planned_ratio: Option<f64>) -> [f64; 4] {
        self.box_exact.insert(axes, false);
        let (width, height) = self.fig.size_px();
        if width <= 1.0 || height <= 1.0 { return bounds; }
        let ratio = planned_ratio.or_else(|| self.drawn_box_ratio(axes));
        let overview = self.facet_overview();
        let (snapped, exact) = geometry::pixel_quantized_bounds(bounds, width, height, ratio, overview, self.style.render.image_anchor);
        // pixel_quantized_bounds returns the plan's bounds untouched when the box is too small.
        let x0 = (bounds[0] * width).floor();
        let y0 = (bounds[1] * height).floor();
        let (x1, y1) = if overview {
            (x0 + (bounds[2] * width - 1e-9).ceil(), y0 + (bounds[3] * height - 1e-9).ceil())
        } else {
            (((bounds[0] + bounds[2]) * width).ceil(), ((bounds[1] + bounds[3]) * height).ceil())
        };
        if x1 - x0 < 2.0 || y1 - y0 < 2.0 { return bounds; }
        self.planned_ratio.insert(axes, ratio);
        self.box_exact.insert(axes, exact);
        snapped
    }

    /// `_planned_image_box_ratio(role)`: the picture's own axes take the plan's shape.
    pub(crate) fn planned_image_box_ratio(&self, role: &str) -> Option<f64> {
        let image = matches!(self.spec.semantic(), Surface::Image(_)) && matches!(role, "image" | "facet_cell");
        if image { self.plan.image_height_over_width } else { None }
    }

    /// `axes.apply_aspect()`, which every figure draw (and `get_position()`)
    /// runs on every axes: zlc's override keeps the layout's box; otherwise
    /// matplotlib shrinks an aspect-locked box to its aspect and anchors it.
    pub(crate) fn apply_aspect(&mut self, axes: usize) {
        let (w, h) = self.fig.size_px();
        let a = &mut self.fig.axes[axes];
        let aspect = match a.aspect { Some(aspect) if !a.owned_box && !a.colorbar_locator => aspect, _ => { a.active = a.position; return; } };
        let space = |axis: &super::mpl::axes::AxisModel, v: f64| match &axis.function { Some(l) => zlc_figure::transform::piecewise(v, l), None => v };
        let sorted = |l: (f64, f64)| if l.0 < l.1 { l } else { (l.1, l.0) };
        let ((x0, x1), (y0, y1)) = (sorted(a.xlim), sorted(a.ylim));
        let xsize = (space(&a.xaxis, x1) - space(&a.xaxis, x0)).abs().max(1e-30);
        let ysize = (space(&a.yaxis, y1) - space(&a.yaxis, y0)).abs().max(1e-30);
        let (box_aspect, fig_aspect) = (aspect * (ysize / xsize), h / w);
        if box_aspect <= 0.0 || fig_aspect <= 0.0 { return; }
        // `Bbox.shrunk_to_aspect(box_aspect, position, fig_aspect).anchored(anchor, position)`.
        let p = a.position;
        let (pw, ph) = (p[2] - p[0], p[3] - p[1]);
        let mut big_h = pw * box_aspect / fig_aspect;
        let big_w = if big_h <= ph { pw } else { big_h = ph; ph * fig_aspect / box_aspect };
        let shrunk = [p[0], p[1], p[0] + big_w, p[1] + big_h];
        let (cx, cy) = geometry::anchor_fractions(a.anchor);
        let dx = (p[0] + cx * (pw - (shrunk[2] - shrunk[0]))) - shrunk[0];
        let dy = (p[1] + cy * (ph - (shrunk[3] - shrunk[1]))) - shrunk[1];
        a.active = [shrunk[0] + dx, shrunk[1] + dy, shrunk[2] + dx, shrunk[3] + dy];
    }

    fn drawn_box_ratio(&self, axes: usize) -> Option<f64> {
        let a = &self.fig.axes[axes];
        let aspect = a.aspect?;
        let x_span = (a.xaxis.space(a.xlim.1) - a.xaxis.space(a.xlim.0)).abs();
        let y_span = (a.yaxis.space(a.ylim.1) - a.yaxis.space(a.ylim.0)).abs();
        if !(x_span > 0.0 && y_span > 0.0) { return None; }
        Some(aspect * y_span / x_span)
    }

    /// `_settle_owned_boxes`.
    pub(crate) fn settle_owned_boxes(&mut self) {
        let (width, height) = self.fig.size_px();
        for axes in 0..self.fig.axes.len() {
            let bounds = self.quantized_bounds.get(&axes).copied();
            let exact = self.box_exact.get(&axes).copied().unwrap_or(false);
            let mut owned = false;
            if let (Some(bounds), true) = (bounds, exact) {
                owned = geometry::owns_box(bounds, width, height, self.planned_ratio.get(&axes).copied().flatten());
            }
            self.owned_box.insert(axes, owned);
            let installed = self.owned_axes.contains(&axes);
            if owned != installed {
                if owned { self.owned_axes.insert(axes); } else { self.owned_axes.remove(&axes); }
                self.fig.axes[axes].owned_box = owned;
                if let Some(bounds) = bounds {
                    self.fig.axes[axes].set_position(bounds);
                    self.chrome_dirty.insert(axes);
                }
            }
        }
    }

    // ----------------------------------------------------------------- text
    /// The payload a grid's labels read: the selected cell's.
    pub(crate) fn labelled_payload<'p>(&self, payload: &'p Payload) -> &'p Payload {
        match payload {
            Payload::Facet(facet) if !facet.cells.is_empty() => {
                let index = self.facet.focused.unwrap_or(0).min(facet.cells.len() - 1);
                &facet.cells[index].payload
            }
            other => other,
        }
    }

    /// `_effective_labels`: the x and y labels.
    pub(crate) fn effective_labels(&self, payload: &Payload, state: &DisplayState) -> (String, String) {
        let payload = self.labelled_payload(payload);
        match (self.spec.semantic(), payload) {
            (Surface::Curve(_) | Surface::Rolling(_), _) => super::curve::curve_labels(&self.spec, super::curve::curve_series(payload), state),
            (Surface::Image(_), Payload::Image(image)) => { let (x, y, _) = self.image_labels(image, state); (x, y) }
            (Surface::PulseTimeline(_), Payload::Pulse(pulse)) => self.pulse_labels(&pulse.data, state),
            (Surface::Histogram(_), Payload::Histogram(histogram)) => super::histogram::histogram_labels(&self.spec, histogram, state),
            _ => (String::new(), String::new()),
        }
    }

    fn update_text_artists(&mut self, payload: &Payload, state: &DisplayState) {
        let (x_label, y_label) = self.effective_labels(payload, state);
        if self.is_facet() {
            self.update_facet_text(&x_label, &y_label);
        } else {
            let primary = self.primary_axes();
            let axes = &mut self.fig.axes[primary];
            if axes.xaxis.label != x_label { axes.xaxis.label = x_label; }
            if axes.yaxis.label != y_label { axes.yaxis.label = y_label; }
        }
        if let (Payload::Image(image), Some(colorbar)) = (self.labelled_payload(payload), &self.image.colorbar) {
            let (_, _, value_label) = self.image_labels(image, state);
            self.fig.axes[colorbar.axes].yaxis.label = value_label;
        }
        self.update_title_artist(state);
    }

    /// `_update_title_artist`.
    fn update_title_artist(&mut self, state: &DisplayState) {
        let spec_title = self.spec.labels().title.as_deref().map(str::to_owned);
        let title = state_label(text_value(state.get("title")).as_deref(), spec_title.as_deref()).unwrap_or_default();
        if self.is_facet() {
            self.update_facet_title(&title);
            return;
        }
        let mut pad = self.style.render.axes_title_pad_pt;
        if matches!(self.spec, PlotSpec::Rolling(_)) {
            pad += self.style.fonts.annotation_pt + self.style.render.compact_axes_title_pad_pt;
        }
        let primary = self.primary_axes();
        let size = self.style.fonts.figure_title_pt;
        let color = self.rc.axes_titlecolor.clone();
        let axes = &mut self.fig.axes[primary];
        axes.titles[0] = TitleText { text: title, size, color };
        axes.title_pad = pad;
        axes.autotitlepos = false;
    }

    /// `_apply_grid`.
    fn apply_grid(&mut self, state: &DisplayState) {
        let show = matches!(state.get("show_grid"), Some(Value::Bool(true)));
        if matches!(self.spec, PlotSpec::PulseTimeline(_)) {
            // Time rules only: `grid(True, axis="x", color=..., linewidth=...)`.
            let (colour, width) = (self.style.palette.pulse_grid, self.style.pulse.grid_linewidth);
            let primary = self.primary_axes();
            let axes = &mut self.fig.axes[primary];
            axes.xaxis.major.grid_on = show;
            if show {
                axes.xaxis.major.grid_color = Some(colour.into());
                axes.xaxis.major.grid_linewidth = Some(width);
            }
            return;
        }
        let data_axes = if self.is_facet() { self.facet_cells() } else { vec![self.primary_axes()] };
        for index in data_axes {
            let axes = &mut self.fig.axes[index];
            axes.xaxis.major.grid_on = show;
            axes.yaxis.major.grid_on = show;
        }
    }

    // --------------------------------------------------------------- limits
    /// `_resolve_data_limits(key, data_range, state, quantity, cache_selected=...)`.
    pub(crate) fn resolve_data_limits(&mut self, key: &str, data_range: Option<(f64, f64)>, state: &DisplayState,
                                      quantity: &str, cache_selected: bool) -> (f64, f64) {
        self.resolve_limits(key, data_range, state, quantity, cache_selected, false, false)
    }

    /// `_resolve_data_limits(..., allow_partial=..., cache_selected=..., rules_geometry=...)`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn resolve_limits(&mut self, key: &str, data_range: Option<(f64, f64)>, state: &DisplayState, quantity: &str,
                                 cache_selected: bool, allow_partial: bool, rules_geometry: bool) -> (f64, f64) {
        let limits_key = format!("{key}:automatic_{quantity}_limits");
        let mode = relim_mode(state);
        let asked = (mode, rules_geometry);
        let current = self.limits.get(&limits_key).and_then(|m| m.limits);
        let mode_asked = self.limits.get(&format!("{key}:relim_mode")).and_then(|m| m.asked);
        let automatic = if mode == RelimMode::Fixed {
            None
        } else if let Some(range) = data_range {
            let policy = &self.style.render;
            Some(autoscaled_limits(range, current, policy.image_color_padding_fraction, policy.image_color_deadband_fraction,
                                   mode == RelimMode::Normal, relim_retains(mode, rules_geometry) && mode_asked == Some(asked)))
        } else {
            Some(current.unwrap_or((0.0, 1.0)))
        };
        let authored = (real_value(state.get(&format!("{quantity}_min"))), real_value(state.get(&format!("{quantity}_max"))));
        let selected = select_display_limits(mode, automatic, authored, allow_partial)
            .unwrap_or_else(|error| panic!("{error}"));
        let cached = if cache_selected { Some(selected) } else { automatic };
        if let Some(cached) = cached {
            self.limits.entry(limits_key).or_insert(LimitMemo { limits: None, asked: None }).limits = Some(cached);
        }
        self.limits.entry(format!("{key}:relim_mode")).or_insert(LimitMemo { limits: None, asked: None }).asked = Some(asked);
        selected
    }

    // ------------------------------------------------------------------- IR
    /// The figure's IR as it stands (the live picture's artists).
    pub fn ir(&self) -> ir::Figure { self.fig.to_ir(&self.rc) }

    /// `save`'s preparation (`_settle_owned_boxes`, `_materialize_prepared_curve`,
    /// `_sync_axes_chrome`) on a copy, and its IR: what an export draws.
    pub fn materialized_ir(&self) -> ir::Figure { self.materialized_ir_drawn_at(None) }

    /// [`Renderer::materialized_ir`] for a draw at `draw_dpi` instead of the
    /// figure's: `print_svg`/`print_pdf` draw at 72 dpi what `save` prepared
    /// at the export dpi, so the draw's own work (ticks, the IR's draw-time
    /// geometry, label positions) runs at 72.
    pub fn materialized_ir_drawn_at(&self, draw_dpi: Option<f64>) -> ir::Figure {
        let mut copy = self.clone();
        copy.in_style = true;
        copy.settle_owned_boxes();
        copy.materialize_prepared_curve();
        copy.materialize_prepared_facet_images();
        copy.materialize_prepared_image();
        copy.sync_axes_chrome();
        if let Some(dpi) = draw_dpi { copy.fig.dpi = dpi; }
        copy.update_ticks_for_draw();
        let mut figure = copy.fig.to_ir_with(&copy.rc, true);
        copy.settle_draw_positions(&mut figure);
        figure
    }

    /// What a draw leaves on the artists it positions: the axis labels'
    /// (and unpinned offset texts') positions, from the tick label boxes.
    pub fn settle_draw_positions(&self, figure: &mut ir::Figure) {
        for (index, x, label, offset) in self.draw_positions(figure) {
            let axis = if x { &mut figure.axes[index].xaxis } else { &mut figure.axes[index].yaxis };
            axis.label.x = label.0;
            axis.label.y = label.1;
            if let Some(offset) = offset { axis.offset_text.x = offset.0; axis.offset_text.y = offset.1; }
        }
    }

    /// What a draw of `figure` leaves on the axes models: an Axis keeps the
    /// label (and unpinned offset text) positions of its last draw.
    pub(crate) fn record_draw_positions(&mut self, figure: &ir::Figure) {
        for (index, x, label, offset) in self.draw_positions(figure) {
            let axis = self.fig.axes[index].axis_mut(x);
            axis.drawn_label_xy = Some(label);
            if offset.is_some() { axis.drawn_offset_xy = offset; }
        }
    }

    /// (axes, x?, label position, unpinned offset position) of every drawn axis.
    #[allow(clippy::type_complexity)]
    fn draw_positions(&self, figure: &ir::Figure) -> Vec<(usize, bool, (f64, f64), Option<(f64, f64)>)> {
        use zlc_figure::query::{NullBackend, Scene, Which};
        let mut fonts = self.fonts.lock().unwrap();
        let mut null = NullBackend;
        let mut math = super::fonts::Math;
        let mut settled = Vec::new();
        {
            let mut scene = Scene::new(figure, &mut null, &mut *fonts, &mut math).expect("scene");
            for (index, axes) in figure.axes.iter().enumerate() {
                if !(axes.visible && axes.axison) { continue; }
                for which in [Which::X, Which::Y] {
                    let label = scene.axis_label_position(index, which).expect("axis label position");
                    let pinned = match which { Which::X => axes.xaxis.offset_text_pinned, Which::Y => axes.yaxis.offset_text_pinned };
                    let offset = if pinned { None } else { Some(scene.offset_text_position(index, which).expect("offset text position")) };
                    settled.push((index, matches!(which, Which::X), label, offset));
                }
            }
        }
        settled
    }

    /// A full draw of the materialized figure at `dpi` (matplotlib's
    /// `canvas.draw()` after `set_dpi`): width, height and RGBA rows.
    pub fn full_draw(&self, dpi: f64) -> std::result::Result<(u32, u32, Vec<u8>), String> {
        let mut copy = self.clone();
        copy.fig.dpi = dpi;
        let figure = copy.materialized_ir();
        let (w, h) = copy.fig.size_px();
        let mut backend = zlc_figure::agg::AggBackend::new(w, h, dpi).map_err(|e| e.to_string())?;
        let mut fonts = self.fonts.lock().unwrap();
        zlc_figure::draw(&figure, &mut backend, &mut *fonts, &mut super::fonts::Math).map_err(|e| e.to_string())?;
        Ok((backend.renderer.width, backend.renderer.height, std::mem::take(&mut backend.renderer.pixels)))
    }

    /// `save(format="png", dpi=dpi)`: the materialized figure drawn at `dpi`.
    pub fn export_png(&self, dpi: f64) -> std::result::Result<Vec<u8>, String> {
        let (width, height, pixels) = self.full_draw(dpi)?;
        let mut out = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut out, width, height);
            encoder.set_color(png::ColorType::Rgba);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().map_err(|e| e.to_string())?;
            writer.write_image_data(&pixels).map_err(|e| e.to_string())?;
        }
        Ok(out)
    }

    /// `save(format="svg"|"pdf", dpi=dpi)`: the figure materialized at `dpi`
    /// and drawn at 72 dpi, as `print_svg`/`print_pdf` draw it.
    pub fn export_vector(&self, dpi: f64, format: crate::vector::Format) -> std::result::Result<Vec<u8>, String> {
        let mut copy = self.clone();
        copy.fig.dpi = dpi;
        let figure = copy.materialized_ir_drawn_at(Some(72.0));
        let mut fonts = self.fonts.lock().unwrap();
        crate::vector::export(&figure, format, dpi, &mut *fonts, &mut super::fonts::Math)
    }

    /// The `_update_ticks` an Axis draw runs on every drawn axis.
    pub fn update_ticks_for_draw(&mut self) {
        let measure = self.measure.clone();
        let size_inches = self.fig.size_inches;
        let dpi = self.fig.dpi;
        let figure_px = self.fig.size_px();
        for axes in &mut self.fig.axes {
            if !(axes.visible && axes.axison) { continue; }
            let geometry = super::mpl::axes::TickGeometry { bbox: axes.bbox_exact(size_inches, dpi), figure: figure_px, dpi, room: axes.room };
            let (xlim, ylim) = (axes.xlim, axes.ylim);
            axes.xaxis.update_ticks(xlim, &geometry, measure.as_ref());
            axes.yaxis.update_ticks(ylim, &geometry, measure.as_ref());
            // `Axis.draw`: `offsetText.set_text(formatter.get_offset())`.
            for axis in [&mut axes.xaxis, &mut axes.yaxis] { axis.offset_text = axis.offset_string.clone(); }
        }
    }

    pub fn children_of(&self, axes: usize) -> &[(u64, Child)] { &self.fig.axes[axes].children }
}

/// `payload is previous`: a pulse timeline's payload is the timeline object itself.
pub(crate) fn same_payload(a: &Option<Arc<Payload>>, b: &Option<Arc<Payload>>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => match (&**a, &**b) {
            (Payload::Pulse(a), Payload::Pulse(b)) => Arc::ptr_eq(&a.data, &b.data),
            _ => Arc::ptr_eq(a, b),
        },
        _ => false,
    }
}

pub(crate) fn text_value(value: Option<&Value>) -> Option<String> {
    match value { None | Some(Value::None) => None, Some(Value::Text(text)) => Some(text.to_string()), Some(other) => Some(other.py_str()) }
}

/// Python `bool(value)` of a display value.
pub(crate) fn truthy(value: &Value) -> bool {
    match value {
        Value::None => false, Value::Bool(b) => *b, Value::Int(i) => *i != 0, Value::Float(f) => *f != 0.0,
        Value::Text(t) => !t.is_empty(), Value::List(l) => !l.is_empty(), Value::Map(m) => !m.is_empty(),
    }
}

pub(crate) fn real_value(value: Option<&Value>) -> Option<f64> {
    match value { None | Some(Value::None) => None, Some(v) => v.real() }
}

pub(crate) fn relim_mode(state: &DisplayState) -> RelimMode {
    let text = text_value(state.get("relim_mode")).unwrap_or_else(|| "normal".into());
    RelimMode::parse(&text).unwrap_or_else(|| panic!("relim_mode {text:?}"))
}
