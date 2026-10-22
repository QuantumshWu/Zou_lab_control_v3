//! What data a fit reads: `FitProjection.fit_selection` -- the selector or
//! viewport authority, and the exact painted points (curve series, histogram
//! bins, image pixels or a regular image) in the model's units.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::nd::{Array, DType, Scalar};
use super::output::{AxisValue, GoldenArrays, Payload, QuantityArray, convert_array};
use super::plan::{Bits, ProjectionPlan, SpecFacts};
use super::view::DataView;
use super::{ErrorKind, ProjectionError, Result};
use crate::units::Unit;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectorKind { XRange, Area, Crosshair, Threshold }

impl SelectorKind {
    pub fn value(self) -> &'static str {
        match self { SelectorKind::XRange => "x_range", SelectorKind::Area => "area", SelectorKind::Crosshair => "crosshair", SelectorKind::Threshold => "threshold" }
    }
}

/// A committed selector's value in canonical units.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectorValue { Range([Bits; 2]), Area([[Bits; 2]; 2]), Point([Bits; 2]), Threshold(Bits) }

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SelectorFacts { pub kind: SelectorKind, pub value: SelectorValue }

/// The context a fit selection reads (`ProjectionContext` + display flags).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FitContext {
    #[serde(default)] pub selectors: Vec<SelectorFacts>,
    /// The drawn viewport per axis, in display units.
    #[serde(default)] pub viewport: Option<[Option<[Bits; 2]>; 2]>,
    #[serde(default)] pub focused_facet_index: Option<usize>,
    #[serde(default)] pub density: bool,
    #[serde(default)] pub cumulative: bool,
}

/// The model facts a selection depends on (owned by the fit layer).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FitRequestFacts {
    pub model_id: String,
    /// `FitTarget` values the model is authored for.
    pub targets: Vec<String>,
    pub arity: usize,
    /// `model.capabilities` intersects the regular-image capabilities.
    pub regular_image: bool,
    /// `UnitRelation` values of the model's coordinates.
    pub relations: Vec<String>,
    pub selector_kind: Option<SelectorKind>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FitScope { Selector, Viewport, All }

impl FitScope { pub fn value(self) -> &'static str { match self { FitScope::Selector => "selector", FitScope::Viewport => "viewport", FitScope::All => "all" } } }

/// `RegularImageFitInput`: a Cartesian image without coordinate grids.
#[derive(Clone, Debug)]
pub struct RegularImage { pub x: Array, pub y: Array, pub observations: Array, pub valid_mask: Option<Array> }

/// `FitSelection`.
#[derive(Clone, Debug)]
pub struct FitSelection {
    pub data_revision: u64,
    pub scope: FitScope,
    pub coordinates: Vec<Array>,
    pub observations: Array,
    pub selected_indices: Option<Array>,
    pub observation_sigma: Option<Array>,
    pub facet_index: Option<usize>,
    pub selector_kind: Option<SelectorKind>,
    pub regular_image: Option<RegularImage>,
    pub group_key: Vec<AxisValue>,
}

impl FitSelection {
    pub fn golden(&self, arrays: &mut GoldenArrays) -> Value {
        let optional = |arrays: &mut GoldenArrays, a: &Option<Array>| a.as_ref().map_or(Value::Null, |a| arrays.array(a));
        let regular = self.regular_image.as_ref().map(|r| json!({"type": "RegularImageFitInput", "x_coordinates": arrays.array(&r.x),
            "y_coordinates": arrays.array(&r.y), "observations": arrays.array(&r.observations),
            "valid_mask": optional(arrays, &r.valid_mask), "selected_indices": Value::Null}));
        json!({"type": "FitSelection", "data_revision": self.data_revision, "scope": self.scope.value(),
               "coordinates": self.coordinates.iter().map(|c| arrays.array(c)).collect::<Vec<_>>(),
               "observations": arrays.array(&self.observations), "selected_indices": optional(arrays, &self.selected_indices),
               "observation_sigma": optional(arrays, &self.observation_sigma), "facet_index": self.facet_index,
               "selector_kind": self.selector_kind.map(SelectorKind::value), "regular_image": regular.unwrap_or(Value::Null),
               "group_key": self.group_key.iter().map(AxisValue::golden).collect::<Vec<_>>()})
    }
}

const FIT_KINDS: [SelectorKind; 3] = [SelectorKind::Area, SelectorKind::XRange, SelectorKind::Threshold];

fn value_error(message: &str) -> ProjectionError { ProjectionError::value(message) }

/// One quantity a fit coordinate or selector speaks in.
struct Quantity { canonical: Arc<Unit>, display: Arc<Unit> }

impl Quantity {
    fn of(q: &QuantityArray) -> Quantity { Quantity { canonical: q.canonical_unit.clone(), display: q.display_unit.clone() } }
    /// `display.convert_value_to([value], canonical)`.
    fn to_canonical(&self, value: f64) -> f64 {
        if self.canonical == self.display { value } else { self.canonical.from_base(self.display.to_base(value)) }
    }
}

struct Selection<'a> { view: &'a DataView, plan: &'a ProjectionPlan, payload: &'a Payload }

impl<'a> Selection<'a> {
    fn semantic(&self) -> &SpecFacts { self.plan.spec.semantic() }
    fn histogram(&self) -> bool { matches!(self.semantic(), SpecFacts::Histogram { .. }) }
    fn first_payload(&self) -> &Payload {
        match self.payload { Payload::Facet(f) if !f.cells.is_empty() => &f.cells[0].payload, other => other }
    }
    /// `_value_quantity()`.
    fn value_quantity(&self) -> Result<Quantity> {
        Ok(match self.first_payload() {
            Payload::Curve(c) if !c.series.is_empty() => Quantity::of(&c.series[0].y),
            Payload::Image(i) => Quantity::of(&i.z),
            Payload::Histogram(h) => Quantity::of(&h.edges),
            _ => Quantity { canonical: self.view.value_unit.clone(), display: self.view.value_display_unit.clone() },
        })
    }
    fn focused(&self, facet_index: Option<usize>, context: &FitContext) -> Result<&Payload> {
        let Payload::Facet(facet) = self.payload else { return Ok(self.payload) };
        let selected = facet_index.or(context.focused_facet_index);
        match selected {
            Some(index) if index < facet.cells.len() => Ok(&facet.cells[index].payload),
            _ => Err(ProjectionError::new(ErrorKind::Index, "facet index is outside the current grid")),
        }
    }
    fn coordinate(&self, axis: &crate::kinds::AxisRef) -> Result<Quantity> {
        let resolved = self.view.resolve(axis)?;
        Ok(Quantity { canonical: resolved.canonical_unit.clone(), display: resolved.display_unit.clone() })
    }
    /// `_require_fit_model_compatible`: the plot kind's fit target, then units.
    fn require_compatible(&self, request: &FitRequestFacts, context: &FitContext) -> Result<()> {
        let target = match self.semantic() {
            SpecFacts::Curve { .. } | SpecFacts::Rolling { .. } => "series",
            SpecFacts::Image { .. } => "image",
            SpecFacts::Histogram { .. } => "histogram",
            _ => "none",
        };
        if !request.targets.iter().any(|t| t == target) {
            return Err(value_error(&format!("fit model {} is not authored for {target} plots", crate::configuration::value::repr_str(&request.model_id))));
        }
        if !self.units_compatible(request, context)? {
            return Err(value_error(&format!("fit model {} is incompatible with the plot axes", crate::configuration::value::repr_str(&request.model_id))));
        }
        Ok(())
    }
    /// `_fit_model_units_compatible` + `fit_model_units_compatible`: every
    /// plotted independent axis has its relation's dimension. Python reads
    /// the quantities under `except (AttributeError, TypeError, ValueError)`.
    fn units_compatible(&self, request: &FitRequestFacts, context: &FitContext) -> Result<bool> {
        let caught = |error: ProjectionError| -> Result<Option<Arc<Unit>>> {
            match error.kind { ErrorKind::Type | ErrorKind::Value | ErrorKind::DataView | ErrorKind::AxisResolution | ErrorKind::EmptySelection => Ok(None), _ => Err(error) }
        };
        let read = |quantity: Result<Quantity>| -> Result<Option<Arc<Unit>>> { match quantity { Ok(q) => Ok(Some(q.canonical)), Err(e) => caught(e) } };
        let Some(value) = read(self.value_quantity())? else { return Ok(false) };
        let axis_0 = if self.histogram() { value.clone() } else {
            let Some(unit) = read(self.x_source(false, context))? else { return Ok(false) };
            unit
        };
        let axis_1 = if request.arity == 2 {
            let Some(y) = self.semantic().y() else { return Ok(false) };
            let Some(unit) = read(self.coordinate(y))? else { return Ok(false) };
            Some(unit)
        } else { None };
        let sources = if request.arity == 1 { vec![axis_0.clone()] } else { match &axis_1 { Some(y) => vec![axis_0.clone(), y.clone()], None => return Ok(false) } };
        if sources.len() != request.relations.len() { return Ok(false); }
        Ok(sources.iter().zip(&request.relations).all(|(source, relation)| match relation.as_str() {
            "axis_0" => source.compatible_with(&axis_0),
            "axis_1" => axis_1.as_ref().is_some_and(|y| source.compatible_with(y)),
            "value" => source.compatible_with(&value),
            _ => false,
        }))
    }
    /// `_x_quantity()` / `_x_selector_source()`.
    fn x_source(&self, selector: bool, context: &FitContext) -> Result<Quantity> {
        if selector && self.histogram() { return self.value_quantity(); }
        match self.semantic() {
            SpecFacts::Rolling { .. } => match self.focused(None, context)? {
                Payload::Curve(c) if !c.series.is_empty() => Ok(Quantity::of(&c.series[0].x)),
                _ => Err(ProjectionError::type_error("rolling x coordinate requires a visible series")),
            },
            other => match other.x() {
                Some(x) => self.coordinate(x),
                None => Err(ProjectionError::type_error("this plot has no coordinate x axis")),
            },
        }
    }
    fn relation_unit(&self, relation: &str, context: &FitContext) -> Result<Arc<Unit>> {
        Ok(match relation {
            "value" => self.value_quantity()?.canonical,
            "axis_0" => if self.histogram() { self.value_quantity()?.canonical } else { self.x_source(false, context)?.canonical },
            "axis_1" => match self.semantic().y() {
                Some(y) => self.coordinate(y)?.canonical,
                None => return Err(ProjectionError::type_error("the selected fit model requires a plot y-coordinate axis")),
            },
            _ => return Err(ProjectionError::type_error("'NoneType' object has no attribute 'canonical_unit'")),
        })
    }
    /// `_viewport_in_canonical(fit=True)`.
    fn viewport_canonical(&self, context: &FitContext) -> Result<Option<[Option<(f64, f64)>; 2]>> {
        let Some([x_range, y_range]) = &context.viewport else { return Ok(None) };
        let canonical = |range: &[Bits; 2], quantity: &Quantity| -> Option<(f64, f64)> {
            let (mut low, mut high) = (quantity.to_canonical(range[0].0), quantity.to_canonical(range[1].0));
            if low.is_nan() || low == f64::NEG_INFINITY { low = -f64::MAX; }
            if high == f64::INFINITY { high = f64::MAX; }
            (low.is_finite() && high.is_finite()).then_some(if high < low { (high, low) } else { (low, high) })
        };
        let x = match x_range { None => None, Some(r) => canonical(r, &self.x_source(true, context)?) };
        let image = matches!(self.semantic(), SpecFacts::Image { .. });
        let y = match y_range {
            Some(r) if image => match self.semantic().y() { Some(axis) => canonical(r, &self.coordinate(axis)?), None => None },
            _ => None,
        };
        Ok(if x.is_none() && y.is_none() { None } else { Some([x, y]) })
    }
}

fn bits_range(range: &[Bits; 2]) -> (f64, f64) { let (a, b) = (range[0].0, range[1].0); if b < a { (b, a) } else { (a, b) } }

fn floats(array: &Array) -> Vec<f64> { array.to_f64() }

fn to_solver(values: Vec<f64>, quantity: &QuantityArray, solver: &Unit) -> Result<Array> {
    if !quantity.canonical_unit.compatible_with(solver) {
        return Err(value_error("fit model coordinate axes require compatible canonical units"));
    }
    let array = Array::vector(values);
    Ok(convert_array(&array, &quantity.canonical_unit, solver).as_f64_array())
}

fn indices_of(valid: &[bool]) -> Array { Array::vector(valid.iter().enumerate().filter(|(_, v)| **v).map(|(i, _)| i as i64).collect::<Vec<i64>>()) }
fn keep(values: &[f64], valid: &[bool]) -> Vec<f64> { values.iter().zip(valid).filter(|(_, k)| **k).map(|(v, _)| *v).collect() }

/// `FitProjection.fit_selection(model, selector_kind=..., facet_index=..., group_index=...)`.
pub fn fit_selection(view: &DataView, plan: &ProjectionPlan, payload: &Payload, request: &FitRequestFacts, context: &FitContext,
                     facet_index: Option<usize>, group_index: Option<usize>) -> Result<FitSelection> {
    prepare_fit_selection(view, plan, payload, request, context)?.select(view, plan, payload, context, facet_index, group_index)
}

/// `_prepare_fit_selection`: one request's authority (the committed
/// selector, else the viewport) and solver units, resolved once before its
/// cell loop.
pub fn prepare_fit_selection(view: &DataView, plan: &ProjectionPlan, payload: &Payload, request: &FitRequestFacts, context: &FitContext)
                             -> Result<PreparedFitSelection> {
    let selection = Selection { view, plan, payload };
    // Python's order: the model's target and units, then the selector.
    selection.require_compatible(request, context)?;
    if request.selector_kind == Some(SelectorKind::Crosshair) { return Err(value_error("crosshair selectors cannot define a fit")); }
    // `_fit_selection_authority`: the committed selector, else the viewport.
    let find = |kind: SelectorKind| context.selectors.iter().find(|s| s.kind == kind);
    let selector = match request.selector_kind {
        Some(kind) if !FIT_KINDS.contains(&kind) => return Err(value_error("crosshair selectors cannot define a fit")),
        Some(kind) => {
            let found = find(kind).ok_or_else(|| ProjectionError::new(ErrorKind::Key, format!("<SelectorKind.{}: '{}'>", kind.value().to_uppercase(), kind.value())))?;
            if !FIT_KINDS.contains(&found.kind) { return Err(value_error("fit selector must contain a numeric selection")); }
            Some(found.clone())
        }
        None => [SelectorKind::Area, SelectorKind::XRange].into_iter().find_map(find).cloned(),
    };
    let viewport = if selector.is_none() { selection.viewport_canonical(context)? } else { None };
    let solver: Vec<Arc<Unit>> = request.relations.iter().map(|r| selection.relation_unit(r, context)).collect::<Result<_>>()?;
    Ok(PreparedFitSelection { arity: request.arity, regular_image: request.regular_image, selector, viewport, solver })
}

/// A request's resolved selection (`_prepare_fit_selection`'s closure).
#[derive(Clone, Debug)]
pub struct PreparedFitSelection {
    arity: usize,
    regular_image: bool,
    selector: Option<SelectorFacts>,
    viewport: Option<[Option<(f64, f64)>; 2]>,
    solver: Vec<Arc<Unit>>,
}

impl PreparedFitSelection {
    /// The painted points of one cell (`select(facet_index, group_index)`).
    pub fn select(&self, view: &DataView, plan: &ProjectionPlan, payload: &Payload, context: &FitContext,
                  facet_index: Option<usize>, group_index: Option<usize>) -> Result<FitSelection> {
        let selection = Selection { view, plan, payload };
        let (selector, viewport, solver) = (self.selector.as_ref(), self.viewport, &self.solver);
        let facet_index = facet_index.or(context.focused_facet_index);
        let focused = selection.focused(facet_index, context)?;
        let kind = selector.map(|s| s.kind);
        let revision = payload.revision();
        match focused {
            Payload::Curve(curve) => {
                if self.arity != 1 { return Err(value_error("curve fit models require exactly one independent axis")); }
                let source = curve.series.first().ok_or_else(|| value_error("painted curve has no series"))?;
                let x = floats(&source.x.canonical);
                let y = floats(&source.y.canonical);
                let mut valid: Vec<bool> = source.valid.bools().iter().zip(&x).zip(&y).map(|((v, a), b)| *v && a.is_finite() && b.is_finite()).collect();
                let scope = match selector {
                    Some(s) => {
                        match &s.value {
                            SelectorValue::Range(r) if s.kind == SelectorKind::XRange => { let (lo, hi) = bits_range(r); for (k, v) in valid.iter_mut().zip(&x) { *k &= *v >= lo && *v <= hi; } }
                            SelectorValue::Area(a) if s.kind == SelectorKind::Area => { let (lo, hi) = bits_range(&a[0]); for (k, v) in valid.iter_mut().zip(&x) { *k &= *v >= lo && *v <= hi; } }
                            SelectorValue::Threshold(t) if s.kind == SelectorKind::Threshold => { for (k, v) in valid.iter_mut().zip(&y) { *k &= *v >= t.0; } }
                            _ => return Err(value_error("selected geometry cannot define a curve fit domain")),
                        }
                        FitScope::Selector
                    }
                    None => match viewport.and_then(|v| v[0]) {
                        Some((lo, hi)) => { for (k, v) in valid.iter_mut().zip(&x) { *k &= *v >= lo && *v <= hi; } FitScope::Viewport }
                        None => FitScope::All,
                    },
                };
                let coordinates = to_solver(keep(&x, &valid), &source.x, &solver[0])?;
                Ok(FitSelection { data_revision: revision, scope, coordinates: vec![coordinates], observations: Array::vector(keep(&y, &valid)),
                    selected_indices: Some(indices_of(&valid)), observation_sigma: source.sem.as_ref().map(|s| Array::vector(keep(&floats(s), &valid))),
                    facet_index, selector_kind: kind, regular_image: None, group_key: vec![] })
            }
            Payload::Image(image) if self.regular_image => {
                let x = to_solver(floats(&image.x.canonical), &image.x, &solver[0])?;
                let y = to_solver(floats(&image.y.canonical), &image.y, &solver[1])?;
                let (valid, observations, scope) = image_domain(image, selector, viewport)?;
                for (name, axis) in [("x", &x), ("y", &y)] {
                    let values = axis.slice::<f64>();
                    if values.iter().any(|v| !v.is_finite()) { return Err(value_error("regular image coordinates must be finite")); }
                    let steps: Vec<f64> = values.windows(2).map(|w| w[1] - w[0]).collect();
                    if !steps.is_empty() && !(steps.iter().all(|d| *d > 0.0) || steps.iter().all(|d| *d < 0.0)) {
                        return Err(value_error(&format!("regular image {name} coordinates must be strictly monotonic")));
                    }
                }
                let storage = if observations.dtype() == DType::Bool { Array::from_vec(observations.bools().iter().map(|b| u8::from(*b)).collect::<Vec<u8>>(), observations.shape().to_vec()) } else { observations };
                let valid_mask = valid.filter(|v| !v.iter().all(|k| *k)).map(|v| Array::from_vec(v, storage.shape().to_vec()));
                Ok(FitSelection { data_revision: revision, scope, coordinates: vec![x.clone(), y.clone()], observations: storage.clone(), selected_indices: None,
                    observation_sigma: None, facet_index, selector_kind: kind,
                    regular_image: Some(RegularImage { x, y, observations: storage, valid_mask }), group_key: vec![] })
            }
            Payload::Histogram(histogram) if selection.histogram() => {
                if context.density || context.cumulative {
                    return Err(value_error("histogram fitting requires count projection; set density=False and cumulative=False"));
                }
                let centers = floats(&histogram.centers.canonical);
                let group = match group_index {
                    Some(g) => g,
                    None if histogram.group_keys.len() != 1 => return Err(value_error("a grouped histogram fit must select each distribution")),
                    None => 0,
                };
                let bins = centers.len();
                let counts: Vec<f64> = histogram.counts.slice::<i64>()[group * bins..(group + 1) * bins].iter().map(|c| *c as f64).collect();
                if !counts.iter().any(|c| *c > 0.0) { return Err(value_error("histogram distribution has no samples")); }
                let mut valid: Vec<bool> = centers.iter().zip(&counts).map(|(c, n)| c.is_finite() && n.is_finite()).collect();
                let scope = match selector {
                    Some(s) => {
                        match &s.value {
                            SelectorValue::Range(r) if s.kind == SelectorKind::XRange => { let (lo, hi) = bits_range(r); for (k, v) in valid.iter_mut().zip(&centers) { *k &= *v >= lo && *v <= hi; } }
                            SelectorValue::Area(a) if s.kind == SelectorKind::Area => { let (lo, hi) = bits_range(&a[0]); for (k, v) in valid.iter_mut().zip(&centers) { *k &= *v >= lo && *v <= hi; } }
                            SelectorValue::Threshold(t) if s.kind == SelectorKind::Threshold => { for (k, v) in valid.iter_mut().zip(&centers) { *k &= *v >= t.0; } }
                            _ => return Err(value_error("selected geometry cannot define a histogram fit domain")),
                        }
                        FitScope::Selector
                    }
                    None => match viewport.and_then(|v| v[0]) {
                        Some((lo, hi)) => { for (k, v) in valid.iter_mut().zip(&centers) { *k &= *v >= lo && *v <= hi; } FitScope::Viewport }
                        None => FitScope::All,
                    },
                };
                let coordinates = to_solver(keep(&centers, &valid), &histogram.centers, &solver[0])?;
                Ok(FitSelection { data_revision: revision, scope, coordinates: vec![coordinates], observations: Array::vector(keep(&counts, &valid)),
                    selected_indices: Some(indices_of(&valid)), observation_sigma: None, facet_index, selector_kind: kind, regular_image: None,
                    group_key: histogram.group_keys[group].clone() })
            }
            _ if selection.histogram() => Err(ProjectionError::new(ErrorKind::Runtime, "histogram projection did not produce histogram data")),
            Payload::Image(image) => {
                if self.arity != 2 { return Err(value_error("image fit models require exactly two independent axes")); }
                let x = to_solver(floats(&image.x.canonical), &image.x, &solver[0])?;
                let y = to_solver(floats(&image.y.canonical), &image.y, &solver[1])?;
                let (mut valid, observations, scope) = image_domain(image, selector, viewport)?;
                let (xs, ys) = (x.slice::<f64>(), y.slice::<f64>());
                let (nx, ny) = (xs.len(), ys.len());
                if !(xs.iter().all(|v| v.is_finite()) && ys.iter().all(|v| v.is_finite())) {
                    let plane: Vec<bool> = (0..ny * nx).map(|p| ys[p / nx].is_finite() && xs[p % nx].is_finite()).collect();
                    valid = Some(match valid { None => plane, Some(v) => v.iter().zip(plane).map(|(a, b)| *a && b).collect() });
                }
                let obs = observations.to_f64();
                let (coordinates, selected, chosen): (Vec<Array>, Vec<f64>, Vec<i64>) = match &valid {
                    None => (vec![Array::vector((0..ny * nx).map(|p| xs[p % nx]).collect::<Vec<f64>>()), Array::vector((0..ny * nx).map(|p| ys[p / nx]).collect::<Vec<f64>>())],
                             obs.clone(), (0..(ny * nx) as i64).collect()),
                    Some(v) => {
                        let positions: Vec<usize> = (0..ny * nx).filter(|p| v[*p]).collect();
                        (vec![Array::vector(positions.iter().map(|p| xs[p % nx]).collect::<Vec<f64>>()), Array::vector(positions.iter().map(|p| ys[p / nx]).collect::<Vec<f64>>())],
                         positions.iter().map(|p| obs[*p]).collect(), positions.iter().map(|p| *p as i64).collect())
                    }
                };
                Ok(FitSelection { data_revision: revision, scope, coordinates, observations: Array::vector(selected), selected_indices: Some(Array::vector(chosen)),
                    observation_sigma: None, facet_index, selector_kind: kind, regular_image: None, group_key: vec![] })
            }
            _ => Err(ProjectionError::type_error("unsupported fit projection payload")),
        }
    }
}

/// `_image_fit_domain`: one canonical mask over the projected image, or None for all.
fn image_domain(image: &super::output::ImageData, selector: Option<&SelectorFacts>, viewport: Option<[Option<(f64, f64)>; 2]>)
    -> Result<(Option<Vec<bool>>, Array, FitScope)> {
    let x = floats(&image.x.canonical);
    let y = floats(&image.y.canonical);
    let (nx, ny) = (x.len(), y.len());
    let observations = image.z.canonical.clone();
    let and = |valid: Option<Vec<bool>>, plane: Vec<bool>| -> Option<Vec<bool>> {
        Some(match valid { None => plane, Some(v) => v.iter().zip(plane).map(|(a, b)| *a && b).collect() })
    };
    let mut valid = if image.valid.constant_value() == Some(Scalar::Bool(true)) { None } else { Some(image.valid.bools().to_vec()) };
    if observations.kind() == 'f' { valid = and(valid, observations.finite_mask()); }
    if !(x.iter().all(|v| v.is_finite()) && y.iter().all(|v| v.is_finite())) {
        valid = and(valid, (0..ny * nx).map(|p| y[p / nx].is_finite() && x[p % nx].is_finite()).collect());
    }
    let box_plane = |columns: Vec<bool>, rows: Vec<bool>| -> Vec<bool> { (0..ny * nx).map(|p| rows[p / nx] && columns[p % nx]).collect() };
    let scope = match selector {
        Some(s) => {
            match &s.value {
                SelectorValue::Range(r) if s.kind == SelectorKind::XRange => {
                    let (lo, hi) = bits_range(r);
                    valid = and(valid, box_plane(x.iter().map(|v| *v >= lo && *v <= hi).collect(), vec![true; ny]));
                }
                SelectorValue::Area(a) if s.kind == SelectorKind::Area => {
                    let ((xl, xh), (yl, yh)) = (bits_range(&a[0]), bits_range(&a[1]));
                    valid = and(valid, box_plane(x.iter().map(|v| *v >= xl && *v <= xh).collect(), y.iter().map(|v| *v >= yl && *v <= yh).collect()));
                }
                SelectorValue::Threshold(t) if s.kind == SelectorKind::Threshold => {
                    valid = and(valid, observations.to_f64().iter().map(|v| *v >= t.0).collect());
                }
                _ => return Err(value_error("selected geometry cannot define an image fit domain")),
            }
            FitScope::Selector
        }
        None => match viewport {
            Some([xr, yr]) => {
                let columns = x.iter().map(|v| xr.is_none_or(|(lo, hi)| *v >= lo && *v <= hi)).collect();
                let rows = y.iter().map(|v| yr.is_none_or(|(lo, hi)| *v >= lo && *v <= hi)).collect();
                valid = and(valid, box_plane(columns, rows));
                FitScope::Viewport
            }
            None => FitScope::All,
        },
    };
    Ok((valid, observations, scope))
}
