//! The projection a session presents (`_fit_projection.FitProjection`): one
//! built view and payload of one input revision, read under one context --
//! the units of every plotted quantity, the data a fit reads, the fitted
//! model drawn in display units.  The view and payload are built by
//! [`crate::projection`]; this module is what the session and the fit read
//! from them.

use std::sync::{Arc, Mutex, MutexGuard};

use crate::configuration::data_contract::PlotSchema;
use crate::configuration::error::{ErrorKind, PlotError, Result};
use crate::configuration::handlers::{handler, FitTargetKind};
use crate::configuration::resolve::Defaults;
use crate::configuration::selectors::{CrosshairPoint, NumericRange, RectangleRange, SelectorKind, SelectorState, SelectorValue, Viewport,
                                      normalize_viewport};
use crate::configuration::semantics::{projection_scope, ScopeCoordinate};
use crate::configuration::spec::{CellPlot, PlotSpec, Surface};
use crate::configuration::state::DisplayState;
use crate::configuration::value::{Value, ValueMap};
use crate::fit::engine::evaluate_model;
use crate::fit::classifier::evaluate_component;
use crate::fit::param_units::{FitParameterUnits, FitQuantityUnits};
use crate::fit::radial::{RegularImageFitInput, Samples};
use crate::fit::{FitModelSpec, FitParameterDisplay, FitResult, FitTarget, UnitRelation};
use crate::interaction::controller::SelectorSnapshot;
use crate::kinds::{AxisDomain, AxisRef, PlotKind};
use crate::projection::model::{canonical_scalar, PRIMARY_INDEX};
use crate::projection::fit_selection::{prepare_fit_selection, FitContext, FitRequestFacts, FitScope, PreparedFitSelection, SelectorFacts,
                                       SelectorKind as FitSelectorKind, SelectorValue as FitSelectorValue};
use crate::projection::nd::{Array, DType, Scalar};
use crate::projection::output::{AxisValue, HistogramData, Payload, PulsePayload, QuantityArray, SelectionSubject};
use crate::pulse::PulseTimelineData;
use crate::projection::plan::{Bits, Defaults as PlanDefaults, DisplayFacts, ProjectionPlan, ScopeTerm, ScopeValue, SpecFacts};
use crate::projection::view::{DataView, ViewCarries};
use crate::projection::{HistogramProjection, Projection, ProjectionError, SchemaCache, Snapshot, Source};
use crate::pyfmt::format_g;
use crate::units::{Unit, UnitRegistry};

use super::fit_layer::fit_error;
use super::overlay::{FitEllipseGlyph, FitOverlay, FitPolyline};

/// A projection refusal as the Python exception it is (DataView refusals are
/// ValueErrors).
pub fn projection_error(error: ProjectionError) -> PlotError {
    use crate::projection::ErrorKind as Kind;
    let kind = match error.kind {
        Kind::Type => ErrorKind::Type, Kind::Index => ErrorKind::Index, Kind::Key => ErrorKind::Key,
        Kind::NotImplemented | Kind::Runtime => ErrorKind::Runtime,
        Kind::DataView | Kind::AxisResolution | Kind::Value | Kind::EmptySelection => ErrorKind::Value,
    };
    PlotError::new(kind, error.message)
}

/// One plotted quantity's units.
#[derive(Clone, Debug, PartialEq)]
pub struct Quantity { pub canonical: Arc<Unit>, pub display: Arc<Unit> }

impl Quantity {
    pub fn of(array: &QuantityArray) -> Self { Quantity { canonical: array.canonical_unit.clone(), display: array.display_unit.clone() } }
    /// `_display_scalar_to_canonical`.
    pub fn to_canonical(&self, value: f64) -> f64 { convert(value, &self.display, &self.canonical) }
    /// `_canonical_scalar_to_display`.
    pub fn to_display(&self, value: f64) -> f64 { convert(value, &self.canonical, &self.display) }
    pub fn range_to_canonical(&self, value: NumericRange) -> Result<NumericRange> { NumericRange::new(self.to_canonical(value.low), self.to_canonical(value.high)) }
    pub fn range_to_display(&self, value: NumericRange) -> Result<NumericRange> { NumericRange::new(self.to_display(value.low), self.to_display(value.high)) }
    fn units(&self) -> FitQuantityUnits { FitQuantityUnits { canonical_unit: self.canonical.clone(), display_unit: self.display.clone() } }
}

/// `Unit.convert_value_to` over float64 values of compatible units.
pub fn convert(value: f64, source: &Unit, target: &Unit) -> f64 { if source == target { value } else { target.from_base(source.to_base(value)) } }

/// The exact identity of one input revision (`DatasetRevisionRef`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SnapshotRef { pub block_id: Arc<str>, pub generation: Arc<str>, pub revision: u64 }

impl SnapshotRef {
    pub fn of(source: &Source) -> Self { SnapshotRef { block_id: source.block_id.clone(), generation: source.generation.clone(), revision: source.revision } }
}

/// The owner or worker state one projection is read under (`ProjectionContext`).
#[derive(Clone, Debug)]
pub struct ProjectionContext {
    pub display_state: Arc<DisplayState>,
    pub selectors: SelectorSnapshot,
    pub viewport: Option<Viewport>,
    pub focused_facet_index: Option<usize>,
}

impl ProjectionContext {
    pub fn selector_state(&self, kind: SelectorKind) -> Option<SelectorState> { self.selectors.states().into_iter().find(|state| state.kind() == kind) }
}

/// The library facts every projection of a panel shares.
#[derive(Clone)]
pub struct ProjectionSettings {
    pub registry: Option<Arc<UnitRegistry>>,
    pub defaults: Arc<Defaults>,
}

/// `KeyError(kind)`'s text.
pub fn selector_key_error(kind: SelectorKind) -> PlotError {
    let name = match kind { SelectorKind::XRange => "X_RANGE", SelectorKind::Area => "AREA", SelectorKind::Crosshair => "CROSSHAIR", SelectorKind::Threshold => "THRESHOLD" };
    PlotError::key(format!("<SelectorKind.{name}: '{}'>", kind.value()))
}

// ------------------------------------------------------------------ plan
fn spec_facts(spec: &PlotSpec) -> Result<SpecFacts> {
    Ok(match spec {
        PlotSpec::Curve(s) => SpecFacts::Curve { x: s.x.clone(), group: s.group.clone(), reduction: s.reduction },
        PlotSpec::Image(s) => SpecFacts::Image { x: s.x.clone(), y: s.y.clone(), reduction: s.reduction },
        PlotSpec::Histogram(s) => SpecFacts::Histogram { group: s.group.clone(), reduced: s.reduced.clone(), reduction: s.reduction },
        PlotSpec::Rolling(s) => SpecFacts::Rolling { x: s.x.clone(), group: s.group.clone(), reduction: s.reduction },
        PlotSpec::FacetGrid(grid) => SpecFacts::FacetGrid { facet: grid.facet.clone(), cell: Box::new(spec_facts(&grid.cell.as_spec())?) },
        PlotSpec::PulseTimeline(_) => return Err(PlotError::type_error("PulseTimelinePlot requires PulseTimelineData")),
    })
}

fn display_facts(values: &ValueMap) -> DisplayFacts {
    let text = |name: &str| values.get(name).filter(|value| !value.is_none()).map(Value::py_str);
    let count = |name: &str| values.get(name).and_then(|value| match value { Value::Int(v) => Some(*v as usize), _ => None });
    let real = |name: &str| values.get(name).and_then(Value::real).map(Bits);
    DisplayFacts {
        window: count("window"), uncertainty: values.get("uncertainty").and_then(|value| match value { Value::Bool(v) => Some(*v), _ => None }),
        trailing: count("trailing"), bin_count: count("bin_count"), x_relim_mode: text("x_relim_mode"),
        x_min: real("x_min"), x_max: real("x_max"), x_display_unit: text("x_display_unit"), y_display_unit: text("y_display_unit"),
        facet_display_unit: text("facet_display_unit"), value_display_unit: text("value_display_unit"),
    }
}

/// What one projection reads of a resolved configuration (the plan the
/// projection generator records).
pub fn projection_plan(schema: &PlotSchema, spec: &PlotSpec, values: &ValueMap) -> Result<ProjectionPlan> {
    let scope = projection_scope(schema, spec, None)?.into_iter().map(|(axis, value)| Ok(ScopeTerm {
        axis, value: match value {
            ScopeCoordinate::Latest => ScopeValue::Latest,
            ScopeCoordinate::Value(coordinate) => ScopeValue::Coordinate(coordinate_scalar(&coordinate)),
        },
    })).collect::<Result<Vec<_>>>()?;
    Ok(ProjectionPlan { spec: spec_facts(spec)?, scope, display: display_facts(values), view_display: None, defaults: PlanDefaults::default() })
}

pub(crate) fn coordinate_scalar(coordinate: &crate::configuration::kinds::Coordinate) -> Scalar {
    use crate::configuration::kinds::Coordinate;
    match coordinate {
        Coordinate::Integer(v) => Scalar::Int(*v), Coordinate::Real(v) => Scalar::Float(*v),
        Coordinate::Text(v) => Scalar::Text(v.clone()), Coordinate::Missing => Scalar::None,
    }
}

/// A projection scalar as a wire value.
pub fn scalar_value(scalar: &Scalar) -> Value {
    match scalar {
        Scalar::None => Value::None, Scalar::Bool(v) => Value::Bool(*v), Scalar::Int(v) => Value::Int(*v),
        Scalar::Float(v) => Value::Float(*v), Scalar::Text(v) => Value::Text(v.clone()),
    }
}

// ------------------------------------------------------------------ fit inputs
/// The exact painted samples a fit consumes (`FitSelection`).
#[derive(Clone, Debug)]
pub struct FitSelection {
    pub data_revision: u64,
    pub scope: FitScope,
    pub coordinates: Vec<Arc<[f64]>>,
    pub observations: Arc<[f64]>,
    pub selected_indices: Option<Arc<[i64]>>,
    pub observation_sigma: Option<Arc<[f64]>>,
    pub facet_index: Option<usize>,
    pub selector_kind: Option<SelectorKind>,
    pub regular_image: Option<Arc<RegularImageFitInput>>,
    pub group_key: Vec<AxisValue>,
}

/// One distribution a fit or classifier answers: `(facet, group, key)`.
#[derive(Clone, Debug, PartialEq)]
pub struct FitCellTarget { pub facet_index: Option<usize>, pub group_index: Option<usize>, pub group_key: Vec<AxisValue> }

/// One retained coordinate a batch fit is laid out along (the fit's sample axes).
#[derive(Clone, Debug, PartialEq)]
pub struct SampleAxis {
    pub domain: AxisDomain,
    pub id: Arc<str>,
    pub name: Arc<str>,
    pub role: Arc<str>,
    pub frame: Option<Arc<str>>,
    pub unit: Option<Arc<str>>,
    pub coordinates: Vec<Scalar>,
    pub labels: Option<Vec<Arc<str>>>,
}

impl SampleAxis {
    pub fn size(&self) -> usize { self.coordinates.len() }
}

const FIT_OVERLAY_POINT_LIMIT: usize = 4096;

// ------------------------------------------------------------------ the projection
struct Built {
    source: Arc<Source>,
    revision: u64,
    view: Mutex<DataView>,
    histogram: Option<HistogramProjection>,
    carry: Mutex<Option<ViewCarries>>,
    settings: ProjectionSettings,
    plan: ProjectionPlan,
}

/// What a projection was built of: a dataset read and its view, or an
/// authored pulse timeline (whose payload is the timeline; no view).
enum Input { Dataset(Built), Pulse { data: Arc<PulseTimelineData>, revision: u64 } }

/// A session's scientific input (`OwnedSnapshot | PulseTimelineData`).
#[derive(Clone)]
pub enum PlotInput { Dataset(Arc<Source>), Pulse(Arc<PulseTimelineData>) }

impl PlotInput {
    pub fn source(&self) -> Option<&Arc<Source>> { match self { PlotInput::Dataset(source) => Some(source), PlotInput::Pulse(_) => None } }
    /// `snapshot_schema(data)` of a dataset; a pulse timeline has none.
    pub fn schema(&self) -> Option<Arc<PlotSchema>> { self.source().map(|source| PlotSchema::new(source.schema.clone())) }
    /// `a is b`.
    pub fn same(&self, other: &PlotInput) -> bool {
        match (self, other) {
            (PlotInput::Dataset(a), PlotInput::Dataset(b)) => Arc::ptr_eq(a, b),
            (PlotInput::Pulse(a), PlotInput::Pulse(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
}

/// One built view and payload under one context.  Copies share the built
/// data; [`FitProjection::with_context`] reads it under another context.
#[derive(Clone)]
pub struct FitProjection {
    input: Arc<Input>,
    spec: PlotSpec,
    payload: Arc<Payload>,
    context: Arc<ProjectionContext>,
}

impl FitProjection {
    /// Build the view and payload of one revision (`FitProjection(...)
    /// ._build_view_and_payload()`), carrying the histogram bins and the
    /// view's work of `inherit`.
    pub fn build(source: Arc<Source>, schema: &PlotSchema, spec: PlotSpec, context: Arc<ProjectionContext>, settings: ProjectionSettings,
                 inherit: Option<&FitProjection>, schemas: &mut SchemaCache) -> Result<FitProjection> {
        let plan = projection_plan(schema, &spec, &context.display_state.values)?;
        let mut owner = Projection::new();
        owner.registry = settings.registry.clone();
        if let Some(Input::Dataset(previous)) = inherit.map(|previous| &*previous.input) {
            owner.histogram = previous.histogram.clone();
            owner.carry = previous.carry.lock().expect("carry lock").take();
            owner.pools = previous.view.lock().expect("view lock").pools.clone();
        }
        let _span = crate::render::profile::span("prepare.projection");
        let snapshot = Snapshot::from_source(&source, schemas).map_err(projection_error)?;
        let projected = match owner.project_snapshot(snapshot, &plan) {
            Ok(projected) => projected,
            Err(error) => {
                // The panel keeps its previous projection: it carries on from this revision's shots.
                if let Some(Input::Dataset(previous)) = inherit.map(|previous| &*previous.input) {
                    *previous.carry.lock().expect("carry lock") = owner.carry.take();
                }
                return Err(projection_error(error));
            }
        };
        let built = Built {
            revision: source.revision, source, view: Mutex::new(projected.view), histogram: owner.histogram.take(),
            carry: Mutex::new(owner.carry.take()), settings, plan,
        };
        Ok(FitProjection { input: Arc::new(Input::Dataset(built)), spec, payload: Arc::new(projected.payload), context })
    }

    /// A pulse timeline's projection: no view, the timeline its own payload.
    pub fn pulse(data: Arc<PulseTimelineData>, revision: u64, spec: PlotSpec, context: Arc<ProjectionContext>) -> Result<FitProjection> {
        if !matches!(spec, PlotSpec::PulseTimeline(_)) {
            return Err(PlotError::type_error(format!("{} requires zlc_data.OwnedSnapshot", crate::configuration::semantics::spec_type_name(&spec))));
        }
        let payload = Arc::new(Payload::Pulse(PulsePayload { data: data.clone(), revision }));
        Ok(FitProjection { input: Arc::new(Input::Pulse { data, revision }), spec, payload, context })
    }

    /// The dataset build every view-reading question needs (a pulse
    /// timeline is never asked one: its session takes the view-less branches).
    fn built(&self) -> &Built {
        match &*self.input { Input::Dataset(built) => built, Input::Pulse { .. } => panic!("a pulse timeline has no data view") }
    }

    pub fn is_pulse(&self) -> bool { matches!(&*self.input, Input::Pulse { .. }) }

    /// `_pulse_x_factor`: source time to the displayed unit.
    pub fn pulse_x_factor(&self) -> Result<f64> {
        let Input::Pulse { data, .. } = &*self.input else { return Err(PlotError::type_error("pulse time conversion requires PulseTimelinePlot")) };
        let unit = self.display_state().values.get("x_display_unit").filter(|value| !value.is_none()).map(Value::py_str);
        Ok(data.time_scale(unit.as_deref(), None)?.0)
    }

    /// `_pulse_source_range_to_display`.
    pub fn pulse_source_range_to_display(&self, value: NumericRange) -> Result<NumericRange> {
        let factor = self.pulse_x_factor()?;
        NumericRange::new(value.low * factor, value.high * factor)
    }

    /// `_pulse_display_range_to_source`.
    pub fn pulse_display_range_to_source(&self, value: NumericRange) -> Result<NumericRange> {
        let factor = self.pulse_x_factor()?;
        NumericRange::new(value.low / factor, value.high / factor)
    }

    /// The same built data read under `context` (`_with_context`).
    pub fn with_context(&self, context: Arc<ProjectionContext>) -> FitProjection { FitProjection { context, ..self.clone() } }

    /// The same built data drawn under another specification of the same
    /// projection facts (a label edit).
    pub fn with_spec(&self, spec: PlotSpec) -> FitProjection { FitProjection { spec, ..self.clone() } }

    pub fn data(&self) -> PlotInput {
        match &*self.input { Input::Dataset(built) => PlotInput::Dataset(built.source.clone()), Input::Pulse { data, .. } => PlotInput::Pulse(data.clone()) }
    }
    /// The dataset revision this is of; a pulse timeline has none.
    pub fn source_ref(&self) -> Option<SnapshotRef> {
        match &*self.input { Input::Dataset(built) => Some(SnapshotRef::of(&built.source)), Input::Pulse { .. } => None }
    }
    pub fn data_revision(&self) -> u64 { match &*self.input { Input::Dataset(built) => built.revision, Input::Pulse { revision, .. } => *revision } }
    /// Which run the data came out of; an authored pulse timeline has none.
    pub fn data_generation(&self) -> Option<&Arc<str>> {
        match &*self.input { Input::Dataset(built) => Some(&built.source.generation), Input::Pulse { .. } => None }
    }
    pub fn payload(&self) -> &Arc<Payload> { &self.payload }
    pub fn spec(&self) -> &PlotSpec { &self.spec }
    pub fn context(&self) -> &Arc<ProjectionContext> { &self.context }
    pub fn display_state(&self) -> &Arc<DisplayState> { &self.context.display_state }
    pub fn viewport(&self) -> Option<Viewport> { self.context.viewport }
    pub fn focused_facet_index(&self) -> Option<usize> { self.context.focused_facet_index }
    pub fn plan(&self) -> &ProjectionPlan { &self.built().plan }
    pub fn histogram(&self) -> Option<&HistogramProjection> { self.built().histogram.as_ref() }
    /// Whether two projections are one built revision (`a.data is b.data`
    /// and `a.payload is b.payload` together).
    pub fn same_built(&self, other: &FitProjection) -> bool {
        match (&*self.input, &*other.input) {
            // A pulse timeline's projection is its timeline, whichever copy reads it.
            (Input::Pulse { data: a, .. }, Input::Pulse { data: b, .. }) => Arc::ptr_eq(a, b),
            _ => Arc::ptr_eq(&self.input, &other.input),
        }
    }
    pub fn same_payload(&self, other: &FitProjection) -> bool { Arc::ptr_eq(&self.payload, &other.payload) }
    pub fn view(&self) -> MutexGuard<'_, DataView> { self.built().view.lock().expect("view lock") }

    fn semantic(&self) -> Surface<'_> { self.spec.semantic() }
    pub fn is_histogram_plot(&self) -> bool { self.semantic().kind() == PlotKind::Histogram }

    /// `_fit_target`: the cell kind's fit target.
    pub fn fit_target(&self) -> Option<FitTarget> {
        handler(self.semantic().kind()).fit_target.map(|kind| match kind {
            FitTargetKind::Series => FitTarget::Series, FitTargetKind::Histogram => FitTarget::Histogram, FitTargetKind::Image => FitTarget::Image,
        })
    }

    // ------------------------------------------------------------- quantities
    pub fn coordinate(&self, axis: &AxisRef) -> Result<Quantity> {
        let resolved = self.view().resolve(axis).map_err(projection_error)?;
        Ok(Quantity { canonical: resolved.canonical_unit.clone(), display: resolved.display_unit.clone() })
    }

    /// `_value_quantity`: the accepted plotted quantity's units.
    pub fn value_quantity(&self) -> Quantity {
        let mut payload: &Payload = &self.payload;
        if let Payload::Facet(facet) = payload {
            if let Some(cell) = facet.cells.first() { payload = &cell.payload; }
        }
        match payload {
            Payload::Curve(curve) if !curve.series.is_empty() => Quantity::of(&curve.series[0].y),
            Payload::Image(image) => Quantity::of(&image.z),
            Payload::Histogram(histogram) => Quantity::of(&histogram.edges),
            _ => {
                let view = self.view();
                Quantity { canonical: view.value_unit.clone(), display: view.value_display_unit.clone() }
            }
        }
    }

    pub fn x_ref(&self) -> Result<AxisRef> { self.semantic().x().cloned().ok_or_else(|| PlotError::type_error("this plot has no coordinate x axis")) }

    pub fn y_axis_ref(&self) -> Result<AxisRef> {
        self.semantic().y().cloned().ok_or_else(|| PlotError::type_error("the selected fit model requires a plot y-coordinate axis"))
    }

    fn rolling_x_quantity(&self) -> Result<Quantity> {
        match &*self.focused_payload(None)? {
            Payload::Curve(curve) if !curve.series.is_empty() => Ok(Quantity::of(&curve.series[0].x)),
            _ => Err(PlotError::type_error("rolling x coordinate requires a visible series")),
        }
    }

    pub fn x_quantity(&self) -> Result<Quantity> {
        if self.semantic().kind() == PlotKind::Rolling { return self.rolling_x_quantity(); }
        self.coordinate(&self.x_ref()?)
    }

    /// `_x_selector_source`: what a selector's x is measured in.
    pub fn x_selector_quantity(&self) -> Result<Quantity> {
        if self.is_histogram_plot() { return Ok(self.value_quantity()); }
        if self.semantic().kind() == PlotKind::Rolling { return self.rolling_x_quantity(); }
        self.coordinate(&self.x_ref()?)
    }

    /// `_y_ref_or_value`: an image's y axis, else the plotted value.
    pub fn y_quantity(&self) -> Result<Quantity> {
        match self.semantic().y() { Some(y) => self.coordinate(y), None => Ok(self.value_quantity()) }
    }

    /// `_viewport_quantities`: the quantity each navigated axis is drawn in.
    pub fn viewport_quantities(&self, viewport: Option<Viewport>) -> Result<(Option<Quantity>, Option<Quantity>)> {
        let (x, y) = viewport.unwrap_or((None, None));
        Ok((if x.is_none() { None } else { Some(self.x_selector_quantity()?) },
            if y.is_none() || self.is_histogram_plot() { None } else { Some(self.y_quantity()?) }))
    }

    /// `_display_selector_state`: one selector in the units its axes are
    /// drawn in; an end the display unit has no value for is refused.
    pub fn display_selector_state(&self, state: &SelectorState) -> Result<SelectorState> {
        // A pulse paints its time axis in source units: what the axes show
        // is the state itself (its readouts scale the text alone).
        if self.is_pulse() { return Ok(*state); }
        let shown = |value: f64, quantity: &Quantity, end: &str| -> Result<f64> {
            let result = quantity.to_display(value);
            if !result.is_finite() {
                return Err(PlotError::value(format!("{} selector {end} {} {} has no value in {}: move or clear it",
                    state.kind().value().replace('_', " "), format_g(value, 6), quantity.canonical.symbol, quantity.display.symbol)));
            }
            Ok(result)
        };
        let shown_range = |value: NumericRange, quantity: &Quantity, axis: &str| -> Result<NumericRange> {
            NumericRange::new(shown(value.low, quantity, &format!("{axis} low"))?, shown(value.high, quantity, &format!("{axis} high"))?)
        };
        let value = match state.value {
            SelectorValue::XRange(range) => SelectorValue::XRange(shown_range(range, &self.x_selector_quantity()?, "x")?),
            SelectorValue::Area(area) => SelectorValue::Area(RectangleRange {
                x: shown_range(area.x, &self.x_selector_quantity()?, "x")?,
                y: if self.is_histogram_plot() { area.y } else { shown_range(area.y, &self.y_quantity()?, "y")? },
            }),
            SelectorValue::Crosshair(point) => SelectorValue::Crosshair(CrosshairPoint::new(
                shown(point.x, &self.x_selector_quantity()?, "x")?,
                if self.is_histogram_plot() { point.y } else { shown(point.y, &self.y_quantity()?, "y")? })?),
            SelectorValue::Threshold(level) => SelectorValue::Threshold(shown(level, &self.value_quantity(), "value")?),
        };
        Ok(SelectorState { value, ..*state })
    }

    /// `_viewport_in_canonical(fit=...)`.
    pub fn viewport_in_canonical(&self, fit: bool) -> Result<Option<Viewport>> {
        let (x_range, y_range) = self.viewport().ok_or_else(|| PlotError::runtime("no viewport to convert"))?;
        if self.is_pulse() {
            // A pulse's canonical time is its source unit; its rows carry none.
            return Ok(Some((x_range.map(|x| self.pulse_display_range_to_source(x)).transpose()?, y_range)));
        }
        let canonical = |value: NumericRange, quantity: &Quantity| -> Option<NumericRange> {
            let (mut low, mut high) = (quantity.to_canonical(value.low), quantity.to_canonical(value.high));
            if fit {
                if low.is_nan() || low == f64::NEG_INFINITY { low = -f64::MAX; }
                if high == f64::INFINITY { high = f64::MAX; }
            }
            (low.is_finite() && high.is_finite()).then(|| NumericRange::new(low, high).expect("finite ends"))
        };
        let x = match x_range { None => None, Some(range) => canonical(range, &self.x_selector_quantity()?) };
        let y = match y_range {
            None => None,
            Some(_) if fit && self.semantic().kind() != PlotKind::Image => None,
            Some(range) if self.is_histogram_plot() => Some(range),
            Some(range) => canonical(range, &self.y_quantity()?),
        };
        Ok(normalize_viewport(Some((x, y))))
    }

    // ------------------------------------------------------------- payload access
    /// `_focused_payload(facet_index)`.
    pub fn focused_payload(&self, facet_index: Option<usize>) -> Result<Arc<Payload>> {
        if !self.spec.is_facet_grid() { return Ok(self.payload.clone()); }
        let Payload::Facet(facet) = &*self.payload else { return Err(PlotError::index("facet index is outside the current grid")) };
        let selected = facet_index.or(self.focused_facet_index());
        match selected.and_then(|index| facet.cells.get(index)) {
            Some(cell) => Ok(Arc::new(cell.payload.clone())),
            None => Err(PlotError::index("facet index is outside the current grid")),
        }
    }

    /// `_fit_targets(all_facets=...)`.
    pub fn fit_targets(&self, all_facets: bool) -> Vec<FitCellTarget> {
        let cells: Vec<(Option<usize>, &Payload)> = match (&self.spec, &*self.payload) {
            (PlotSpec::FacetGrid(_), Payload::Facet(facet)) => facet.cells.iter().enumerate()
                .filter(|(index, _)| all_facets || *index == self.focused_facet_index().unwrap_or(0))
                .map(|(index, cell)| (Some(index), &cell.payload)).collect(),
            (_, payload) => vec![(None, payload)],
        };
        let mut targets = Vec::new();
        for (facet_index, payload) in cells {
            match payload {
                Payload::Histogram(histogram) => targets.extend(histogram.group_keys.iter().enumerate()
                    .map(|(group, key)| FitCellTarget { facet_index, group_index: Some(group), group_key: key.clone() })),
                _ => targets.push(FitCellTarget { facet_index, group_index: None, group_key: Vec::new() }),
            }
        }
        targets
    }

    pub fn has_grouped_histogram(&self) -> bool { self.is_histogram_plot() && self.semantic().group().is_some() }

    /// `_pooled_histogram_groups`: the counted bins as one distribution per axes.
    pub fn pooled_histogram_groups(&self) -> Result<FitProjection> {
        if !self.has_grouped_histogram() { return Ok(self.clone()); }
        let pooled = |payload: &Payload| -> Payload {
            let Payload::Histogram(histogram) = payload else { return payload.clone() };
            let shape = histogram.counts.shape();
            let (groups, bins) = (shape[0], shape[1]);
            let counts = histogram.counts.slice::<i64>();
            let mut total = vec![0i64; bins];
            for group in 0..groups { for bin in 0..bins { total[bin] += counts[group * bins + bin]; } }
            Payload::Histogram(HistogramData { counts: Array::from_vec(total, vec![1, bins]), group_keys: vec![Vec::new()], ..histogram.clone() })
        };
        let mut selected = self.clone();
        match &self.spec {
            PlotSpec::FacetGrid(grid) => {
                let mut grid = grid.clone();
                if let CellPlot::Histogram(cell) = &mut grid.cell { cell.group = None; }
                selected.spec = PlotSpec::FacetGrid(grid);
                if let Payload::Facet(facet) = &*self.payload {
                    let mut facet = facet.clone();
                    for cell in &mut facet.cells { cell.payload = pooled(&cell.payload); }
                    selected.payload = Arc::new(Payload::Facet(facet));
                }
            }
            PlotSpec::Histogram(spec) => {
                let mut spec = spec.clone();
                spec.group = None;
                selected.spec = PlotSpec::Histogram(spec);
                selected.payload = Arc::new(pooled(&self.payload));
            }
            _ => {}
        }
        Ok(selected)
    }

    /// `_fit_sample_axes(targets)`: the retained coordinates, in the order of
    /// the flattened fits.
    pub fn fit_sample_axes(&self, targets: &[FitCellTarget]) -> Result<Vec<SampleAxis>> {
        let mut axes: Vec<(AxisRef, Vec<Scalar>)> = Vec::new();
        if let (Some(facet), Payload::Facet(data)) = (self.spec.facet(), &*self.payload) {
            let mut indices: Vec<usize> = Vec::new();
            for target in targets {
                let index = target.facet_index.expect("facet target");
                if !indices.contains(&index) { indices.push(index); }
            }
            axes.push((facet.clone(), indices.iter().map(|index| data.cells[*index].facet_value_canonical.clone()).collect()));
        }
        if self.has_grouped_histogram() {
            let group = self.semantic().group().cloned().expect("grouped histogram");
            let mut values: Vec<Scalar> = Vec::new();
            for target in targets {
                let value = target.group_key[0].canonical.clone();
                if !values.contains(&value) { values.push(value); }
            }
            axes.push((group, values));
        }
        let mut projected = Vec::new();
        for (reference, values) in axes {
            let view = self.view();
            let resolved = view.resolve(&reference).map_err(projection_error)?;
            let original = resolved.contract.axis.clone();
            let unit = resolved.canonical_unit.symbol.clone();
            let labels = match &original.labels {
                None => None,
                Some(labels) => {
                    let domain = view.domain(&reference).map_err(projection_error)?;
                    Some(values.iter().map(|value| {
                        let entry = domain.values().iter().find(|entry| entry.canonical == *value).expect("retained coordinate");
                        labels[entry.index.expect("used index")].clone()
                    }).collect())
                }
            };
            let coordinates = values.into_iter().map(|value| canonical_scalar(value).map_err(projection_error)).collect::<Result<Vec<_>>>()?;
            projected.push(SampleAxis {
                domain: reference.domain, id: original.id.clone(), name: original.name.clone(), role: original.role.clone(),
                frame: original.frame.clone(), unit: (unit != "1").then(|| Arc::from(unit.as_str())), coordinates: normalized(coordinates), labels,
            });
        }
        Ok(projected)
    }

    /// `selection_subject(spec, payload, source_window=...)` of this view.
    pub fn selection_subject(&self, source_window: Option<usize>) -> Result<SelectionSubject> {
        if self.is_pulse() {
            return Ok(SelectionSubject { plot_kind: PlotKind::PulseTimeline, x: None, y: None, x_coordinate_frame: None,
                                         y_coordinate_frame: None, scope: Vec::new(), source_window: None });
        }
        let mut plan = self.built().plan.clone();
        plan.spec = spec_facts(&self.spec)?;
        plan.display.window = source_window;
        Projection::selection_subject(&self.view(), &plan, &self.payload).map_err(projection_error)
    }

    // ------------------------------------------------------------- fit units
    fn relation_quantity(&self, relation: UnitRelation) -> Result<Option<Quantity>> {
        Ok(match relation {
            UnitRelation::Value => Some(self.value_quantity()),
            UnitRelation::Axis0 => Some(if self.is_histogram_plot() { self.value_quantity() } else { self.x_quantity()? }),
            UnitRelation::Axis1 => Some(self.coordinate(&self.y_axis_ref()?)?),
            _ => None,
        })
    }

    /// The fit parameter units of this projection (`_fit_parameter_units_owner`).
    pub fn parameter_units(&self) -> Result<FitParameterUnits<'_>> {
        let rolling_primary = match &self.spec {
            PlotSpec::Rolling(rolling) => self.view().rolling_coordinate_role(rolling.x.as_ref()).map_err(projection_error)? == PRIMARY_INDEX,
            _ => false,
        };
        Ok(FitParameterUnits {
            registry: self.built().settings.registry.as_deref(),
            quantities: Box::new(move |relation| self.relation_quantity(relation).ok().flatten().map(|quantity| quantity.units())),
            histogram: self.is_histogram_plot(),
            rolling_primary,
        })
    }

    /// `_fit_parameter_units(model)`: each parameter's canonical unit.
    pub fn fit_parameter_units(&self, model: &FitModelSpec) -> Result<Vec<(String, String)>> {
        let units = self.parameter_units()?;
        model.parameters.iter().map(|spec| Ok((spec.name.clone(), units.canonical_parameter_unit(spec).map_err(fit_error)?))).collect()
    }

    /// `_fit_model_units_compatible(model)`.
    pub fn fit_model_units_compatible(&self, model: &FitModelSpec) -> bool {
        let compatible = || -> Result<bool> {
            let value = self.value_quantity().canonical;
            let axis_0 = if self.is_histogram_plot() { value.clone() } else { self.x_quantity()?.canonical };
            let axis_1 = if model.independent_arity == 2 { Some(self.coordinate(&self.y_axis_ref()?)?.canonical) } else { None };
            Ok(crate::fit::target::fit_model_units_compatible(model, &axis_0, axis_1.as_deref(), &value))
        };
        compatible().unwrap_or(false)
    }

    // ------------------------------------------------------------- fit selections
    /// What a fit selection reads of this projection's context.
    fn fit_context(&self) -> FitContext {
        let pair = |range: NumericRange| [Bits(range.low), Bits(range.high)];
        let selectors = self.context.selectors.states().into_iter().map(|state| SelectorFacts {
            kind: fit_selector_kind(state.kind()),
            value: match state.value {
                SelectorValue::XRange(range) => FitSelectorValue::Range(pair(range)),
                SelectorValue::Area(area) => FitSelectorValue::Area([pair(area.x), pair(area.y)]),
                SelectorValue::Crosshair(point) => FitSelectorValue::Point([Bits(point.x), Bits(point.y)]),
                SelectorValue::Threshold(level) => FitSelectorValue::Threshold(Bits(level)),
            },
        }).collect();
        let flag = |name: &str| matches!(self.display_state().values.get(name), Some(Value::Bool(true)));
        FitContext {
            selectors, viewport: self.viewport().map(|(x, y)| [x.map(pair), y.map(pair)]), focused_facet_index: self.focused_facet_index(),
            density: flag("density"), cumulative: flag("cumulative"),
        }
    }

    /// `fit_selection(model, selector_kind=, facet_index=, group_index=)`.
    pub fn fit_selection(&self, model: &FitModelSpec, kind: Option<SelectorKind>, facet_index: Option<usize>, group_index: Option<usize>) -> Result<FitSelection> {
        self.prepare_fit_selection(model, kind)?.select(self, facet_index, group_index)
    }

    /// `_prepare_fit_selection`: the request's model, units and domain,
    /// resolved once before a cell loop.
    pub fn prepare_fit_selection(&self, model: &FitModelSpec, kind: Option<SelectorKind>) -> Result<PreparedSelection> {
        let request = FitRequestFacts {
            model_id: model.model_id.clone(),
            targets: model.targets.iter().map(|target| target.value().to_owned()).collect(),
            arity: model.independent_arity,
            regular_image: model.capabilities.contains(crate::fit::spec::REGULAR_IMAGE_RADIAL)
                || model.capabilities.contains(crate::fit::spec::REGULAR_IMAGE_SEPARABLE),
            relations: model.coordinate_relations.iter().map(|relation| relation.value().to_owned()).collect(),
            selector_kind: kind.map(fit_selector_kind),
        };
        let context = self.fit_context();
        let prepared = prepare_fit_selection(&self.view(), &self.built().plan, &self.payload, &request, &context).map_err(projection_error)?;
        Ok(PreparedSelection { prepared, context })
    }


    // ------------------------------------------------------------- fit overlays
    fn display_parameter(&self, units: &FitParameterUnits<'_>, spec: &crate::fit::FitParameterSpec, value: f64, difference: bool,
                         relation: Option<UnitRelation>) -> Result<(f64, String)> {
        units.display_parameter_value(spec, value, difference, relation).map_err(fit_error)
    }

    /// `_display_fit_parameters`: values and uncertainties in painted units.
    pub fn display_fit_parameters(&self, result: &FitResult) -> Result<Vec<FitParameterDisplay>> {
        let units = self.parameter_units()?;
        let mut rows = Vec::new();
        for (index, spec) in result.model.parameters.iter().enumerate() {
            let (value, unit) = self.display_parameter(&units, spec, result.parameter_values[index], false, None)?;
            let fixed = result.fixed_parameter_names.contains(&spec.name);
            let error = if result.covariance_valid && !fixed {
                Some(self.display_parameter(&units, spec, result.standard_errors[index], true, None)?.0.abs())
            } else { None };
            rows.push(FitParameterDisplay::new(&spec.name, spec.display_label.as_deref().unwrap_or(&spec.name), value, error, &unit).map_err(fit_error)?);
        }
        Ok(rows)
    }

    fn overlay_polylines(&self, result: &FitResult, selection: &FitSelection) -> Result<Vec<FitPolyline>> {
        if !result.success || result.model.independent_arity != 1 { return Ok(Vec::new()); }
        let mut finite: Vec<f64> = selection.coordinates[0].iter().copied().filter(|v| v.is_finite()).collect();
        if finite.len() < 2 { return Ok(Vec::new()); }
        let low = finite.iter().copied().fold(f64::INFINITY, f64::min);
        let high = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sample_count = self.built().settings.defaults.style.artists.fit_component_sample_count;
        let dense = if finite.len() > FIT_OVERLAY_POINT_LIMIT {
            linspace(low, high, sample_count.max(FIT_OVERLAY_POINT_LIMIT))
        } else {
            finite.extend(linspace(low, high, sample_count));
            finite.sort_by(f64::total_cmp);
            finite.dedup_by(|a, b| a == b);
            finite
        };
        let source = self.relation_quantity(result.model.coordinate_relations[0])?
            .ok_or_else(|| PlotError::value("fit coordinate display requires physical axis relations"))?;
        let target = self.relation_quantity(UnitRelation::Axis0)?
            .ok_or_else(|| PlotError::value("fit coordinate display requires physical axis relations"))?;
        if !source.canonical.compatible_with(&target.display) { return Err(PlotError::value("fit coordinate solver and display units are incompatible")); }
        let display_x: Arc<[f64]> = dense.iter().map(|v| convert(*v, &source.canonical, &target.display)).collect();
        let histogram = self.is_histogram_plot();
        let value = self.value_quantity();
        let shown = |values: Vec<f64>| -> Arc<[f64]> {
            if histogram { values.into() } else { values.into_iter().map(|v| convert(v, &value.canonical, &value.display)).collect() }
        };
        let coordinates = [dense];
        let presentation = &result.model.presentation;
        if presentation.components.is_empty() {
            let fitted = evaluate_model(&result.model, &coordinates, &result.parameter_values).map_err(fit_error)?;
            return Ok(vec![FitPolyline { x: display_x, y: shown(fitted), role: if histogram { "total" } else { "primary" }, component_index: 0 }]);
        }
        let components = presentation.components.iter().map(|component|
            evaluate_component(&result.model, &component.component_id, &coordinates, &result.parameter_values).map_err(fit_error))
            .collect::<Result<Vec<_>>>()?;
        let mut total = components[0].clone();
        for values in &components[1..] { for (sum, value) in total.iter_mut().zip(values) { *sum += value; } }
        let mut lines: Vec<FitPolyline> = components.into_iter().enumerate()
            .map(|(index, values)| FitPolyline { x: display_x.clone(), y: shown(values), role: "component", component_index: index }).collect();
        lines.push(FitPolyline { x: display_x, y: shown(total), role: "total", component_index: 0 });
        Ok(lines)
    }

    fn overlay_ellipse(&self, result: &FitResult, parameters: &[FitParameterDisplay]) -> Result<Option<FitEllipseGlyph>> {
        let Some(glyph) = &result.model.presentation.ellipse_glyph else { return Ok(None) };
        if !result.success { return Ok(None); }
        let index = |name: &str| result.model.parameter_index(name).map_err(|error| fit_error(error.into()));
        let center_x = parameters[index(&glyph.center_parameters[0])?].value;
        let center_y = parameters[index(&glyph.center_parameters[1])?].value;
        let units = self.parameter_units()?;
        let mut radii = [0.0; 2];
        for (slot, (name, relation)) in glyph.radius_parameters.iter().zip([UnitRelation::Axis0, UnitRelation::Axis1]).enumerate() {
            let position = index(name)?;
            radii[slot] = self.display_parameter(&units, &result.model.parameters[position], result.parameter_values[position], true, Some(relation))?.0.abs();
        }
        Ok(Some(FitEllipseGlyph { center_x, center_y, radius_x: radii[0], radius_y: radii[1] }))
    }

    /// `_make_fit_overlay(result, selection)`.
    pub fn make_fit_overlay(&self, result: &FitResult, selection: &FitSelection) -> Result<FitOverlay> {
        if !result.success {
            return Ok(FitOverlay { success: false, diagnostic: result.message.clone(), facet_index: selection.facet_index,
                                   group_key: selection.group_key.clone(), ..FitOverlay::default() });
        }
        let parameters = self.display_fit_parameters(result)?;
        let headline = parameters.iter().find(|parameter| parameter.name == result.model.headline).cloned();
        let polylines = self.overlay_polylines(result, selection)?;
        let evidence = if result.evidence.is_finite() {
            format!("ΔBIC = {}: {}", crate::fit::format_fixed(result.evidence, 1), if result.reduced { "one population" } else { "two populations" })
        } else { String::new() };
        Ok(FitOverlay {
            polylines, ellipse_glyph: self.overlay_ellipse(result, &parameters)?, success: true,
            formula: result.model.formula.clone().unwrap_or_default(), parameter_display: parameters, diagnostic: result.message.clone(),
            facet_index: selection.facet_index, group_key: selection.group_key.clone(), headline_parameter: headline, evidence,
        })
    }
}

/// `np.linspace(low, high, count)`.
fn linspace(low: f64, high: f64, count: usize) -> Vec<f64> {
    if count == 0 { return Vec::new(); }
    if count == 1 { return vec![low]; }
    let div = (count - 1) as f64;
    let delta = high - low;
    let step = delta / div;
    let mut values: Vec<f64> = (0..count).map(|i| if step == 0.0 { (i as f64 / div) * delta + low } else { i as f64 * step + low }).collect();
    values[count - 1] = high;
    values
}

/// AxisSpec's coordinate normalization of explicit scalars.
fn normalized(values: Vec<Scalar>) -> Vec<Scalar> {
    if !values.iter().all(Scalar::is_numeric) { return values; }
    if !values.iter().any(|value| matches!(value, Scalar::Float(_))) { return values; }
    if values.iter().any(|value| matches!(value, Scalar::Int(i) if i.abs() > 1 << 53)) { return values; }
    let floats: Vec<f64> = values.iter().map(|value| value.as_f64().expect("numeric")).collect();
    if floats.iter().all(|value| value.fract() == 0.0) {
        return floats.into_iter().map(|value| Scalar::Int(value as i128)).collect();
    }
    floats.into_iter().map(|value| Scalar::Float(if value == 0.0 { 0.0 } else { value })).collect()
}

/// An image's samples in the storage the fit reads natively; any other
/// numeric type widened to float64 as the oracle's plane holds it.
fn image_samples(observations: &Array) -> Samples {
    match observations.dtype() {
        DType::U8 => observations.slice::<u8>().to_vec().into(),
        DType::U16 => observations.slice::<u16>().to_vec().into(),
        DType::I16 => observations.slice::<i16>().to_vec().into(),
        DType::F32 => observations.slice::<f32>().to_vec().into(),
        _ => observations.to_f64().into(),
    }
}

/// One request's prepared selection (`_prepare_fit_selection`'s closure).
pub struct PreparedSelection { prepared: PreparedFitSelection, context: FitContext }

impl PreparedSelection {
    /// Select one cell (`select(facet_index, group_index)`).
    pub fn select(&self, projection: &FitProjection, facet_index: Option<usize>, group_index: Option<usize>) -> Result<FitSelection> {
        let selected = self.prepared.select(&projection.view(), &projection.built().plan, &projection.payload, &self.context, facet_index, group_index)
            .map_err(projection_error)?;
        let floats = |array: &Array| -> Arc<[f64]> { array.to_f64().into() };
        let regular_image = match &selected.regular_image {
            None => None,
            Some(image) => {
                let mask = image.valid_mask.as_ref().map(|mask| mask.bools().to_vec());
                Some(Arc::new(RegularImageFitInput::new(image.x.to_f64(), image.y.to_f64(), image_samples(&image.observations), mask, None)
                    .map_err(fit_error)?))
            }
        };
        Ok(FitSelection {
            data_revision: selected.data_revision, scope: selected.scope, coordinates: selected.coordinates.iter().map(floats).collect(),
            observations: floats(&selected.observations), selected_indices: selected.selected_indices.as_ref().map(|indices| indices.slice::<i64>().into()),
            observation_sigma: selected.observation_sigma.as_ref().map(floats), facet_index: selected.facet_index,
            selector_kind: selected.selector_kind.map(selector_kind_of_fit), regular_image, group_key: selected.group_key,
        })
    }
}

fn fit_selector_kind(kind: SelectorKind) -> FitSelectorKind {
    match kind {
        SelectorKind::XRange => FitSelectorKind::XRange, SelectorKind::Area => FitSelectorKind::Area,
        SelectorKind::Crosshair => FitSelectorKind::Crosshair, SelectorKind::Threshold => FitSelectorKind::Threshold,
    }
}

fn selector_kind_of_fit(kind: FitSelectorKind) -> SelectorKind {
    match kind {
        FitSelectorKind::XRange => SelectorKind::XRange, FitSelectorKind::Area => SelectorKind::Area,
        FitSelectorKind::Crosshair => SelectorKind::Crosshair, FitSelectorKind::Threshold => SelectorKind::Threshold,
    }
}
