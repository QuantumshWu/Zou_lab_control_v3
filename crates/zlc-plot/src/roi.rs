//! `zlc_science.roi`: a coordinate region planned once against the exact
//! source schema an accepted scene was drawn from, and applied to every new
//! read of that source on the native data plane.
//!
//! [`RoiPlan::create`] intersects the selector's region with the subject's
//! explicit scope and source window into one selection and the schema of
//! what survives; [`RoiPlan::apply`] cuts a read (a dense event, or acquired
//! placements with their window) into a snapshot of that schema, exactly as
//! `ROIPlan.apply` then `NativeDataset.from_science` did.

use std::cmp::Ordering;
use std::fmt;
use std::sync::Arc;

use zlc_core::restrict::{compare_numbers, same_scalar, Restriction, RestrictError, Selection, Source, Term};
use zlc_core::runtime::Read;
use zlc_core::schema::{Coordinate, DatasetSchema, Domain, DomainRole};
use zlc_core::dataset::{DataBlock, Validity};
use zlc_core::store::{DatasetStore, Placement, Snapshot};

use crate::configuration::data_contract::indexed_history_layout;
use crate::configuration::selectors::{SelectorState, SelectorValue};
use crate::kinds::{AxisDomain, AxisRef, PlotKind};
use crate::projection::nd::Scalar;
use crate::projection::output::SelectionSubject;

/// The Python exception class a refusal corresponds to.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RoiErrorKind { Unsupported, Replan, Value, Type, Key, Index, EmptySelection }

impl RoiErrorKind {
    pub fn python_name(self) -> &'static str {
        match self {
            Self::Unsupported => "UnsupportedROI", Self::Replan => "ReplanRequired", Self::Value => "ValueError", Self::Type => "TypeError",
            Self::Key => "KeyError", Self::Index => "IndexError", Self::EmptySelection => "EmptySelection",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RoiError { pub kind: RoiErrorKind, pub message: String }

impl RoiError {
    fn new(kind: RoiErrorKind, message: impl Into<String>) -> Self { Self { kind, message: message.into() } }
}

impl fmt::Display for RoiError {
    /// `str(error)`, as the scientific document reported it.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(&self.message) }
}

impl std::error::Error for RoiError {}

impl From<RestrictError> for RoiError {
    fn from(error: RestrictError) -> Self {
        use zlc_core::restrict::ErrorKind as Kind;
        let kind = match error.kind { Kind::Value => RoiErrorKind::Value, Kind::Type => RoiErrorKind::Type, Kind::Index => RoiErrorKind::Index,
                                      Kind::EmptySelection => RoiErrorKind::EmptySelection };
        Self::new(kind, error.message)
    }
}

pub type Result<T> = std::result::Result<T, RoiError>;

fn domain_of(schema: &DatasetSchema, domain: AxisDomain) -> &Arc<Domain> {
    schema.domain(match domain { AxisDomain::Repeat => DomainRole::Repeat, AxisDomain::Point => DomainRole::Point, AxisDomain::CellData => DomainRole::Cell })
}

/// `resolve_axis(schema, ref)`: the axis id and its coordinate frame.
fn resolve(schema: &DatasetSchema, reference: &AxisRef) -> Result<(Arc<str>, Option<Arc<str>>)> {
    let axis = domain_of(schema, reference.domain).axis(&reference.axis_id)
        .ok_or_else(|| RoiError::new(RoiErrorKind::Key, format!("AxisId(value={})", crate::configuration::value::repr_str(&reference.axis_id))))?;
    Ok((axis.id().into(), axis.coordinate_frame().map(Arc::from)))
}

/// `_coordinate_term(schema, ref, low, high, expected_frame=...)`.
fn coordinate_term(schema: &DatasetSchema, reference: &AxisRef, low: &Coordinate, high: &Coordinate, expected: Option<&str>) -> Result<Term> {
    let (axis_id, frame) = resolve(schema, reference)?;
    if frame.as_deref() != expected { return Err(RoiError::new(RoiErrorKind::Value, format!("selection coordinate frame differs from source axis {axis_id}"))); }
    Ok(Term::coordinate_range(axis_id, low, high, frame)?)
}

fn bounds(term: &Term) -> (&Arc<str>, &Coordinate, &Coordinate, &Option<Arc<str>>) {
    match term { Term::Coordinate { axis_id, lower, upper, frame } => (axis_id, lower, upper, frame), _ => unreachable!("ROI terms are coordinate ranges") }
}

/// `_intersect_term(terms, term)`: explicit scopes, window and region on one axis meet.
fn intersect(terms: &mut Vec<Term>, term: Term) -> Result<()> {
    let (axis_id, lower, upper, frame) = bounds(&term);
    let Some(position) = terms.iter().position(|existing| existing.axis_id() == axis_id) else { terms.push(term); return Ok(()) };
    let (_, previous_lower, previous_upper, previous_frame) = bounds(&terms[position]);
    if previous_frame != frame { return Err(RoiError::new(RoiErrorKind::Value, format!("conflicting coordinate frames for {axis_id}"))); }
    let numeric = |value: &Coordinate| matches!(value, Coordinate::Integer(_) | Coordinate::Real(_));
    let empty = || RoiError::new(RoiErrorKind::EmptySelection, format!("ROI and scope do not intersect on {axis_id}"));
    let (low, high) = if [previous_lower, previous_upper, lower, upper].into_iter().all(numeric) {
        // Python's max/min keep the first of equal operands.
        let low = if compare_numbers(lower, previous_lower) == Some(Ordering::Greater) { lower } else { previous_lower };
        let high = if compare_numbers(upper, previous_upper) == Some(Ordering::Less) { upper } else { previous_upper };
        if compare_numbers(high, low) == Some(Ordering::Less) { return Err(empty()); }
        (low.clone(), high.clone())
    } else if [previous_upper, lower, upper].into_iter().all(|value| same_scalar(previous_lower, value)) {
        (lower.clone(), lower.clone())
    } else {
        return Err(empty());
    };
    let replaced = Term::coordinate_range(axis_id.clone(), &low, &high, frame.clone())?;
    terms[position] = replaced;
    Ok(())
}

/// A scope value as a coordinate bound (`canonical_coordinate_scalar` refuses a bool).
fn coordinate_of(value: &Scalar) -> Result<Coordinate> {
    Ok(match value {
        Scalar::None => Coordinate::Missing, Scalar::Int(integer) => Coordinate::Integer(*integer), Scalar::Float(real) => Coordinate::Real(*real),
        Scalar::Text(text) => Coordinate::Text(text.clone()),
        Scalar::Bool(_) => return Err(RoiError::new(RoiErrorKind::Type, "coordinate lower must be null, text, or a finite number")),
    })
}

/// v2 `selection_output_catalog`: what a region derives, by name and label. Every
/// region gives its cut (`roi_frame`); an area also the six reductions of each
/// cell's samples, an x range their mean.
pub const AREA_OUTPUTS: [(&str, &str); 7] = [("roi_frame", "ROI frame"), ("roi_mean", "Mean"), ("roi_sum", "Sum"), ("roi_min", "Min"),
    ("roi_max", "Max"), ("roi_min_10_mean", "Min 10 mean"), ("roi_max_10_mean", "Max 10 mean")];
pub const RANGE_OUTPUTS: [(&str, &str); 2] = [("roi_frame", "ROI frame"), ("roi_mean", "Mean")];

/// v2 `panel_selection_output_catalog`: what a region on this surface would
/// derive. A rolling trace's region names no upstream rows, so it derives nothing;
/// a surface whose y names an axis takes areas, any other x ranges.
pub fn selection_outputs(subject: &SelectionSubject) -> &'static [(&'static str, &'static str)] {
    if subject.plot_kind == PlotKind::Rolling { &[] } else if subject.y.is_some() { &AREA_OUTPUTS } else { &RANGE_OUTPUTS }
}

/// One region's cut, and what it derives: `region` the axis terms, `band` a value
/// band (a bound on the measured value restricts which samples count, v2
/// `_FILTER_DOMAINS`), `outputs` its catalogue.
#[derive(Clone)]
pub struct RoiPlan { restriction: Restriction, band: Option<(f64, f64)>, frame: Arc<DatasetSchema>, outputs: &'static [(&'static str, &'static str)] }

impl RoiPlan {
    /// v2 `PlotSelectionSource._translate` + `SelectionBridge._build_selection`: the
    /// region the operator drew, as the console derives it. A box whose y names no
    /// axis (a curve's value, a histogram's count) is the x range it spans; an x that
    /// names no axis (a histogram's value) is a band on the measured value; a
    /// rolling trace's region derives nothing (`None`).
    pub fn for_region(schema: &Arc<DatasetSchema>, subject: &SelectionSubject, selector: &SelectorState) -> Result<Option<Self>> {
        if subject.plot_kind == PlotKind::Rolling { return Ok(None); }
        let (x_bounds, y_bounds) = match &selector.value {
            SelectorValue::Area(area) => (area.x, subject.y.as_ref().map(|_| area.y)),
            SelectorValue::XRange(range) => (*range, None),
            _ => return Err(RoiError::new(RoiErrorKind::Unsupported, format!("a {} selector marks a point, not a region", selector.kind().value()))),
        };
        if let (Some(x), Some(y), Some(_)) = (&subject.x, &subject.y, y_bounds) {
            if x == y { return Err(RoiError::new(RoiErrorKind::Value, "AREA requires two distinct coordinate axes")); }
        }
        let mut terms = Vec::new();
        let band = match &subject.x {
            Some(x) => { intersect(&mut terms, coordinate_term(schema, x, &Coordinate::Real(x_bounds.low), &Coordinate::Real(x_bounds.high), subject.x_coordinate_frame.as_deref())?)?; None }
            None => Some((x_bounds.low, x_bounds.high)),
        };
        if let (Some(y), Some(bounds)) = (&subject.y, y_bounds) {
            intersect(&mut terms, coordinate_term(schema, y, &Coordinate::Real(bounds.low), &Coordinate::Real(bounds.high), subject.y_coordinate_frame.as_deref())?)?;
        }
        Self::scoped(schema, subject, terms, band, if y_bounds.is_some() { &AREA_OUTPUTS } else { &RANGE_OUTPUTS }).map(Some)
    }

    /// The subject's explicit scope and source window joined to `terms`.
    fn scoped(schema: &Arc<DatasetSchema>, subject: &SelectionSubject, mut terms: Vec<Term>, band: Option<(f64, f64)>,
              outputs: &'static [(&'static str, &'static str)]) -> Result<Self> {
        for (reference, value) in &subject.scope {
            let (_, frame) = resolve(schema, reference)?;
            let coordinate = coordinate_of(value)?;
            intersect(&mut terms, coordinate_term(schema, reference, &coordinate, &coordinate, frame.as_deref())?)?;
        }
        if let Some(window) = subject.source_window {
            let layout = indexed_history_layout(schema).map_err(|error| RoiError::new(RoiErrorKind::Value, error.message.clone()))?
                .ok_or_else(|| RoiError::new(RoiErrorKind::Value, "the subject's source window requires an indexed source schema"))?;
            let keep = window.min(layout.shot_count());
            let axis = schema.domain(DomainRole::Point).axis(&layout.primary_axis_id).expect("primary axis");
            let first = Coordinate::Integer(i128::from(layout.cells[layout.cells.len() - keep]));
            let last = Coordinate::Integer(i128::from(layout.cells[layout.cells.len() - 1]));
            intersect(&mut terms, Term::coordinate_range(axis.id().into(), &first, &last, axis.coordinate_frame().map(Arc::from))?)?;
        }
        // A region of value bounds alone names no rows: every row and cell stays.
        let selection = if terms.is_empty() { None } else { Some(Selection::new(terms)?) };
        let restriction = Restriction::new(schema, selection.as_ref())?;
        let frame = match band { Some(_) => banded_schema(restriction.schema())?, None => restriction.schema().clone() };
        Ok(Self { restriction, band, frame, outputs })
    }

    /// The names and labels this region derives.
    pub fn outputs(&self) -> &'static [(&'static str, &'static str)] { self.outputs }

    /// v2 `_materialize_selection_outputs`: the wanted outputs of one read, by
    /// name. The cut keeps the parent's Repeat and Point; each reduction is one
    /// value per (repeat, point) of the cell samples that count (valid, a
    /// number), invalid where none does.
    pub fn derive(&self, read: &Read, wanted: &[&str]) -> Result<std::collections::BTreeMap<String, Snapshot>> {
        let cut = self.apply(read)?;
        let mut outputs = std::collections::BTreeMap::new();
        let reductions: Vec<&str> = wanted.iter().copied().filter(|name| *name != "roi_frame" && self.outputs.iter().any(|(output, _)| output == name)).collect();
        if !reductions.is_empty() { outputs.extend(reduce(&cut, &reductions)?); }
        if wanted.contains(&"roi_frame") { outputs.insert("roi_frame".to_owned(), cut); }
        Ok(outputs)
    }
    /// `create_plan(schema, subject, selector)`: AREA over two source
    /// coordinates or X_RANGE over one, intersected with the subject's scope
    /// and source window. A facet index is not a scope.
    pub fn create(schema: &Arc<DatasetSchema>, subject: &SelectionSubject, selector: &SelectorState) -> Result<Self> {
        let unsupported = |message: String| RoiError::new(RoiErrorKind::Unsupported, message);
        let ranges = match (&selector.value, &subject.x, &subject.y) {
            (SelectorValue::Area(_), None, _) | (SelectorValue::Area(_), _, None) =>
                return Err(unsupported("AREA requires two source coordinate axes; value-domain filtering is not implemented".into())),
            (SelectorValue::Area(_), Some(x), Some(y)) if x == y => return Err(RoiError::new(RoiErrorKind::Value, "AREA requires two distinct coordinate axes")),
            (SelectorValue::Area(area), Some(x), Some(y)) =>
                vec![(x, area.x, subject.x_coordinate_frame.clone()), (y, area.y, subject.y_coordinate_frame.clone())],
            (SelectorValue::XRange(_), None, _) => return Err(unsupported("X_RANGE has no source coordinate axis; histogram/value filtering is not implemented".into())),
            (SelectorValue::XRange(range), Some(x), _) => vec![(x, *range, subject.x_coordinate_frame.clone())],
            _ => return Err(unsupported(format!("{} is not an implemented coordinate ROI", selector.kind().value()))),
        };
        let area = ranges.len() == 2;
        let mut terms = Vec::new();
        for (reference, interval, frame) in ranges {
            intersect(&mut terms, coordinate_term(schema, reference, &Coordinate::Real(interval.low), &Coordinate::Real(interval.high), frame.as_deref())?)?;
        }
        Self::scoped(schema, subject, terms, None, if area { &AREA_OUTPUTS } else { &RANGE_OUTPUTS })
    }

    /// The exact source schema this plan was made for.
    pub fn source_schema(&self) -> &Arc<DatasetSchema> { self.restriction.source() }
    /// The schema of every snapshot `apply` returns.
    pub fn schema(&self) -> &Arc<DatasetSchema> { &self.frame }

    /// `ROIPlan.apply(read)` adopted as a native snapshot.
    pub fn apply(&self, read: &Read) -> Result<Snapshot> {
        let source = Source::of_read(read).map_err(|message| RoiError::new(RoiErrorKind::Value, message))?;
        self.apply_source(&source)
    }

    /// The same, for planes already in hand.
    pub fn apply_source(&self, source: &Source) -> Result<Snapshot> {
        if **source.schema() != **self.restriction.source() {
            return Err(RoiError::new(RoiErrorKind::Replan, "ROI source schema changed; create a new plan from the accepted subject"));
        }
        let cut = self.restriction.apply(source).map_err(|message| RoiError::new(RoiErrorKind::Value, message))?;
        match self.band { Some(band) => banded(&cut, &self.frame, band), None => Ok(cut) }
    }
}

fn value_error(message: impl Into<String>) -> RoiError { RoiError::new(RoiErrorKind::Value, message) }

/// A band decides sample by sample, so the cut's validity varies along every
/// primary Cell axis (v2: `ValidityContract.components(*cell axes)`); the canonical
/// scalar carrier's value-level validity already is per sample.
fn banded_schema(schema: &Arc<DatasetSchema>) -> Result<Arc<DatasetSchema>> {
    let cell = schema.domain(DomainRole::Cell);
    if cell.axes().iter().all(|axis| axis.role() == "scalar") { return Ok(schema.clone()); }
    let components = cell.axes().iter().filter(|axis| axis.coordinate_of().is_none()).map(|axis| Arc::from(axis.id())).collect();
    let value = schema.value();
    let value = zlc_core::schema::ValueSchema::new(value.dtype(), value.unit().map(Arc::from), value.name().map(Arc::from),
        zlc_core::schema::ValidityContract::Components(components)).map_err(value_error)?;
    DatasetSchema::new(schema.domain(DomainRole::Repeat).clone(), schema.domain(DomainRole::Point).clone(), cell.clone(), value).map_err(value_error)
}

/// Rows (Repeat × Point carriers) and samples per row of one block.
fn layout(schema: &DatasetSchema) -> (usize, usize) {
    let rows = schema.domain(DomainRole::Repeat).size() * schema.domain(DomainRole::Point).size();
    (rows, if rows == 0 { 0 } else { schema.value_len() / rows })
}

/// Whether each sample of a block counts as written: its validity spread over every
/// sample (a per-row flag over the row, a component flag over its components' samples).
fn sample_validity(block: &DataBlock) -> Result<Vec<bool>> {
    let schema = block.schema();
    let (rows, width) = layout(schema);
    let mask = match block.validity() { Validity::Uniform(flag) => return Ok(vec![*flag; rows * width]), Validity::Mask(mask) => mask.as_slice() };
    match schema.value().validity() {
        zlc_core::schema::ValidityContract::Value => Ok((0..rows * width).map(|sample| mask[sample / width.max(1)]).collect()),
        zlc_core::schema::ValidityContract::Components(ids) => {
            let cell = schema.domain(DomainRole::Cell);
            let axes: Vec<(&Arc<zlc_core::schema::CodeMapping>, usize)> = ids.iter().map(|id| Ok((cell.mapping(id).ok_or_else(|| value_error("validity axis is not a Cell axis"))?,
                cell.axis(id).map_or(1, |axis| axis.len())))).collect::<Result<_>>()?;
            let per_row: usize = axes.iter().map(|(_, len)| *len).product();
            (0..rows * width).map(|sample| {
                let (row, at) = (sample / width.max(1), sample % width.max(1));
                let mut component = 0;
                for (mapping, len) in &axes { component = component * len + mapping.code(at).ok_or_else(|| value_error("a sample outside its Cell carrier"))?; }
                Ok(mask[row * per_row + component])
            }).collect()
        }
    }
}

/// The cut with every sample outside `band` no longer counting (v2 `valid & (low <= v <= high)`).
fn banded(cut: &Snapshot, schema: &Arc<DatasetSchema>, (low, high): (f64, f64)) -> Result<Snapshot> {
    let scalar = Arc::ptr_eq(schema, cut.schema());
    let mut placements = Vec::new();
    for segment in cut.segments() {
        let block = segment.data();
        let values = crate::session::overlay::real_values(block.values()).ok_or_else(|| value_error("a value band needs real values"))?;
        let counts: Vec<bool> = sample_validity(block)?.into_iter().zip(&values).map(|(valid, value)| valid && *value >= low && *value <= high).collect();
        let chunk = if scalar { block.schema().clone() } else {
            DatasetSchema::new(block.schema().domain(DomainRole::Repeat).clone(), block.schema().domain(DomainRole::Point).clone(),
                schema.domain(DomainRole::Cell).clone(), schema.value().clone()).map_err(value_error)?
        };
        let data = DataBlock::new(chunk, block.values().clone(), Validity::Mask(zlc_core::OwnedBuffer::from_vec(counts)), block.sigma().cloned()).map_err(value_error)?;
        placements.push(Placement::new(segment.origin(), segment.extent(), data).map_err(value_error)?);
    }
    DatasetStore::new(schema.clone()).commit(placements).and_then(|snapshot| snapshot.with_window(cut.window())).map_err(value_error)
}

/// The six reductions (v2 `_roi_statistics`), each a snapshot over the cut's
/// Repeat and Point with one float64 value per row, in the source's unit.
fn reduce(cut: &Snapshot, names: &[&str]) -> Result<std::collections::BTreeMap<String, Snapshot>> {
    let schema = cut.schema();
    let scalar = crate::document::scalar_domain().map_err(value_error)?;
    let output_schema = |repeat: &Arc<zlc_core::schema::Domain>, point: &Arc<zlc_core::schema::Domain>, name: &str| -> Result<Arc<DatasetSchema>> {
        let value = zlc_core::schema::ValueSchema::new(zlc_core::DType::F64, schema.value().unit().map(Arc::from), Some(Arc::from(name)),
            zlc_core::schema::ValidityContract::Value).map_err(value_error)?;
        DatasetSchema::new(repeat.clone(), point.clone(), scalar.clone(), value).map_err(value_error)
    };
    let mut placements: std::collections::BTreeMap<&str, Vec<Placement>> = names.iter().map(|name| (*name, Vec::new())).collect();
    for segment in cut.segments() {
        let block = segment.data();
        let (rows, width) = layout(block.schema());
        let values = crate::session::overlay::real_values(block.values()).ok_or_else(|| value_error("ROI statistics need real values"))?;
        let valid = sample_validity(block)?;
        let mut answers: Vec<Vec<f64>> = vec![vec![0.0; rows]; names.len()];
        let mut counted = vec![false; rows];
        let mut samples = Vec::with_capacity(width);
        for row in 0..rows {
            samples.clear();
            samples.extend((row * width..(row + 1) * width).filter(|at| valid[*at] && values[*at].is_finite()).map(|at| values[at]));
            if samples.is_empty() { continue; }
            counted[row] = true;
            let total: f64 = samples.iter().sum();
            // The mean of the ten smallest (largest) samples, summed in ascending order.
            let tail = |samples: &[f64], top: bool| -> f64 {
                let count = samples.len().min(10);
                let mut ordered = samples.to_vec();
                ordered.sort_by(f64::total_cmp);
                let taken = if top { &ordered[ordered.len() - count..] } else { &ordered[..count] };
                taken.iter().sum::<f64>() / count as f64
            };
            for (index, name) in names.iter().enumerate() {
                answers[index][row] = match *name {
                    "roi_mean" => total / samples.len() as f64,
                    "roi_sum" => total,
                    "roi_min" => samples.iter().copied().fold(f64::INFINITY, f64::min),
                    "roi_max" => samples.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                    "roi_min_10_mean" => tail(&samples, false),
                    "roi_max_10_mean" => tail(&samples, true),
                    other => return Err(value_error(format!("no ROI statistic {other:?}"))),
                };
            }
        }
        for (index, name) in names.iter().enumerate() {
            let chunk = output_schema(block.schema().domain(DomainRole::Repeat), block.schema().domain(DomainRole::Point), name)?;
            let data = DataBlock::new(chunk, zlc_core::OwnedBuffer::from_vec(std::mem::take(&mut answers[index])).into(),
                Validity::Mask(zlc_core::OwnedBuffer::from_vec(counted.clone())), None).map_err(value_error)?;
            placements.get_mut(name).expect("named").push(Placement::new(segment.origin(), segment.extent(), data).map_err(value_error)?);
        }
    }
    names.iter().map(|name| {
        let whole = output_schema(schema.domain(DomainRole::Repeat), schema.domain(DomainRole::Point), name)?;
        let snapshot = DatasetStore::new(whole).commit(placements.remove(name).unwrap_or_default())
            .and_then(|snapshot| snapshot.with_window(cut.window())).map_err(value_error)?;
        Ok(((*name).to_owned(), snapshot))
    }).collect()
}

/// The plan of a recorded `coordinate-roi` effect: the subject and selector
/// as the scientific document writes them (`json.dumps` of
/// `dataclasses.asdict`), planned against the exact source schema. Ints are
/// read exactly; one beyond the native data plane's 128 bits is refused,
/// never rounded.
pub fn plan_from_metadata(schema: &Arc<DatasetSchema>, metadata: &str) -> Result<RoiPlan> {
    use zlc_archive::Json;
    use crate::configuration::selectors::{CrosshairPoint, NumericRange, RectangleRange};
    let metadata = zlc_archive::json::loads(metadata).map_err(|error| RoiError::new(RoiErrorKind::Value, error.message))?;
    let malformed = |what: &str| RoiError::new(RoiErrorKind::Value, format!("ROI effect {what} is malformed"));
    let field = |value: &'_ Json, key: &str| -> Json { value.get(key).cloned().unwrap_or(Json::Null) };
    let axis = |value: &Json| -> Result<Option<AxisRef>> {
        if value.is_null() { return Ok(None); }
        let domain = field(value, "domain").as_str().and_then(AxisDomain::parse).ok_or_else(|| malformed("axis domain"))?;
        let axis_id = field(value, "axis_id");
        AxisRef::new(domain, axis_id.as_str().ok_or_else(|| malformed("axis id"))?).map(Some).map_err(|message| RoiError::new(RoiErrorKind::Value, message))
    };
    let text = |value: Json| value.as_str().map(Arc::<str>::from);
    let count = |value: &Json| u64::try_from(value.as_int()?).ok();
    let subject = field(&metadata, "subject");
    let scope = field(&subject, "scope");
    let scope = scope.as_array().ok_or_else(|| malformed("scope"))?.iter().map(|term| {
        let [reference, value] = term.as_array().unwrap_or_default() else { return Err(malformed("scope term")) };
        let scalar = match value {
            Json::Null => Scalar::None, Json::Bool(flag) => Scalar::Bool(*flag), Json::Str(text) => Scalar::Text(text.as_str().into()),
            Json::Int(integer) => Scalar::Int(*integer), Json::Float(real) => Scalar::Float(*real),
            _ => return Err(malformed("scope coordinate")),
        };
        Ok((axis(reference)?.ok_or_else(|| malformed("scope axis"))?, scalar))
    }).collect::<Result<Vec<_>>>()?;
    let subject = SelectionSubject {
        plot_kind: field(&subject, "plot_kind").as_str().and_then(PlotKind::parse).ok_or_else(|| malformed("plot kind"))?,
        x: axis(&field(&subject, "x"))?, y: axis(&field(&subject, "y"))?,
        x_coordinate_frame: text(field(&subject, "x_coordinate_frame")), y_coordinate_frame: text(field(&subject, "y_coordinate_frame")),
        scope, source_window: count(&field(&subject, "source_window")).map(|window| window as usize),
    };
    let selector = field(&metadata, "selector");
    let refused = |error: crate::configuration::PlotError| RoiError::new(RoiErrorKind::Value, error.to_string());
    let real = |value: &Json| match value { Json::Float(real) => Ok(*real), Json::Int(integer) => Ok(*integer as f64), _ => Err(malformed("selector value")) };
    let range = |value: &Json| -> Result<NumericRange> { NumericRange::new(real(&field(value, "low"))?, real(&field(value, "high"))?).map_err(refused) };
    let value = field(&selector, "value");
    let selected = match field(&selector, "kind").as_str() {
        Some("x_range") => SelectorValue::XRange(range(&value)?),
        Some("area") => SelectorValue::Area(RectangleRange { x: range(&field(&value, "x"))?, y: range(&field(&value, "y"))? }),
        Some("crosshair") => SelectorValue::Crosshair(CrosshairPoint::new(real(&field(&value, "x"))?, real(&field(&value, "y"))?).map_err(refused)?),
        Some("threshold") => SelectorValue::Threshold(real(&value)?),
        _ => return Err(malformed("selector kind")),
    };
    let revision = count(&field(&selector, "revision")).ok_or_else(|| malformed("selector revision"))?;
    let selector = SelectorState::new(selected, revision, count(&field(&selector, "facet_index"))).map_err(refused)?;
    RoiPlan::create(schema, &subject, &selector)
}
