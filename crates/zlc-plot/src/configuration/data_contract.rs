//! The plotting boundary of the native data plane's schemas
//! (`zlc_plot.data_contract` plus the zlc_data helpers it needs): axis
//! resolution, the axis families defaults read, coordinate alternatives, and
//! the indexed-history layout (`zlc_data.snapshot_projection`).

use std::sync::{Arc, OnceLock};

use zlc_core::schema::{Axis, CodeBase, CodeMapping, Coordinate, Coordinates, DatasetSchema, Domain, DomainRole, ValueSchema};

use crate::units::{resolve_unit, Unit, UnitRegistry};

use super::error::{PlotError, Result};
use super::kinds::{canonical_coordinate, declared_ref, domain_role, AxisDomain, AxisRef, DOMAINS};
use super::value::repr_str;

pub const REPEAT: &str = "repeat";
pub const PRIMARY_INDEX: &str = "primary-index";
pub const SHOT_TIME: &str = "shot-time";
pub const SAMPLE_TIME: &str = "sample-time";
pub const SCAN_POINT: &str = "scan-point";
pub const READOUT_EVENT: &str = "readout-event";
pub const SPATIAL_X: &str = "spatial-x";
pub const SPATIAL_Y: &str = "spatial-y";
pub const SCALAR: &str = "scalar";

/// One dataset schema with the facts the configuration layer reads from it
/// many times, each computed once for this schema value.
pub struct PlotSchema {
    schema: Arc<DatasetSchema>,
    history: OnceLock<Result<Option<Arc<IndexedHistoryLayout>>>>,
}

impl PlotSchema {
    pub fn new(schema: Arc<DatasetSchema>) -> Arc<Self> { Arc::new(Self { schema, history: OnceLock::new() }) }
    pub fn dataset(&self) -> &Arc<DatasetSchema> { &self.schema }
    pub fn domain(&self, domain: AxisDomain) -> &Arc<Domain> { self.schema.domain(domain_role(domain)) }
    /// The indexed-history layout, or `None` without a shot index; a broken
    /// index contract is a producer error (ValueError).
    pub fn history(&self) -> Result<Option<Arc<IndexedHistoryLayout>>> {
        self.history.get_or_init(|| indexed_history_layout(&self.schema).map(|layout| layout.map(Arc::new))).clone()
    }
    pub fn value_unit(&self) -> Option<&str> { self.schema.value().unit() }
    /// The value schema's unit through the registry (`schema_value_unit`).
    pub fn canonical_value_unit(&self, registry: Option<&UnitRegistry>) -> Result<Arc<Unit>> {
        Ok(resolve_unit(self.value_unit().unwrap_or("1"), registry)?)
    }
}

impl PartialEq for PlotSchema {
    fn eq(&self, other: &Self) -> bool { Arc::ptr_eq(&self.schema, &other.schema) || *self.schema == *other.schema }
}

impl std::fmt::Debug for PlotSchema {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str("PlotSchema") }
}

/// One exact schema axis, resolved by its reference.
#[derive(Clone)]
pub struct ResolvedAxis {
    pub axis: Arc<Axis>,
    pub domain: Arc<Domain>,
    pub domain_kind: AxisDomain,
}

impl ResolvedAxis {
    pub fn id(&self) -> &str { self.axis.id() }
    pub fn label(&self) -> &str { self.axis.name() }
    pub fn size(&self) -> usize { self.axis.len() }
    pub fn unit_annotation(&self) -> Option<&str> { self.axis.unit() }
    pub fn coordinate_frame(&self) -> Option<&str> { self.axis.coordinate_frame() }
    pub fn coordinate_labels(&self) -> Option<&[Arc<str>]> { self.axis.labels() }
    pub fn canonical_unit(&self, registry: Option<&UnitRegistry>) -> Result<Arc<Unit>> {
        Ok(resolve_unit(self.axis.unit().unwrap_or("1"), registry)?)
    }
    /// One coordinate as `AxisSpec.coordinate_at` returns it (canonical).
    pub fn coordinate_at(&self, index: usize) -> Option<Coordinate> { coordinate_at(&self.axis, index) }
    pub fn coordinate_position(&self, coordinate: &Coordinate) -> Option<usize> { coordinate_position(&self.axis, coordinate) }
    /// The physical axis whose positions this coordinate names.
    pub fn primary(&self) -> Arc<Axis> { coordinate_axis(&self.domain, &self.axis) }
    /// The primary and every alternative coordinate of it, primary first.
    pub fn family(&self) -> Vec<Arc<Axis>> { coordinate_axes(&self.domain, &self.axis) }
    /// The coordinate index each physical row of its domain names (`source_indices`).
    pub fn source_indices(&self) -> Vec<usize> {
        let mapping = self.domain.mapping(self.axis.id()).expect("domain axis has a mapping");
        (0..mapping.carrier_len()).map(|row| mapping.code(row).expect("row inside the carrier")).collect()
    }
}

/// The canonical coordinate at one position.
pub fn coordinate_at(axis: &Axis, index: usize) -> Option<Coordinate> { axis.coordinate(index).map(|value| canonical_coordinate(&value)) }

/// One coordinate's exact logical position (`AxisSpec.coordinate_position`).
pub fn coordinate_position(axis: &Axis, coordinate: &Coordinate) -> Option<usize> {
    let value = canonical_coordinate(coordinate);
    match (axis.coordinates(), &value) {
        (_, Coordinate::Missing) => None,
        (Coordinates::Index { origin, len }, Coordinate::Integer(target)) => {
            let position = target.checked_sub(*origin)?;
            (0..*len as i128).contains(&position).then_some(position as usize)
        }
        (Coordinates::Index { .. }, _) => None,
        (Coordinates::Numeric(_) | Coordinates::OriginsAndOffsets { .. }, Coordinate::Text(_)) => None,
        _ => (0..axis.len()).find(|index| axis.coordinate(*index).as_ref() == Some(&value)),
    }
}

fn find_axis<'a>(domain: &'a Domain, id: &str) -> Option<&'a Arc<Axis>> { domain.axis(id) }

/// The physical axis whose positions this coordinate names.
pub fn coordinate_axis(domain: &Domain, axis: &Arc<Axis>) -> Arc<Axis> {
    match axis.coordinate_of() {
        Some(primary) => find_axis(domain, primary).cloned().unwrap_or_else(|| axis.clone()),
        None => axis.clone(),
    }
}

pub fn coordinate_axes(domain: &Domain, axis: &Arc<Axis>) -> Vec<Arc<Axis>> {
    let primary = coordinate_axis(domain, axis);
    let mut family = vec![primary.clone()];
    family.extend(domain.axes().iter().filter(|other| other.coordinate_of() == Some(primary.id())).cloned());
    family
}

/// Resolve an exact reference; a reference the schema lacks is a KeyError.
pub fn resolve_axis(schema: &PlotSchema, reference: &AxisRef) -> Result<ResolvedAxis> {
    let domain = schema.domain(reference.domain);
    let axis = find_axis(domain, &reference.axis_id)
        .ok_or_else(|| PlotError::key(format!("AxisId(value={})", repr_str(&reference.axis_id))))?;
    Ok(ResolvedAxis { axis: axis.clone(), domain: domain.clone(), domain_kind: reference.domain })
}

/// One physical-axis identity for all declared coordinate alternatives.
pub fn coordinate_axis_ref(schema: &PlotSchema, reference: &AxisRef) -> Result<AxisRef> {
    let resolved = resolve_axis(schema, reference)?;
    Ok(declared_ref(reference.domain, resolved.primary().id()))
}

/// Every producer-declared information axis, exactly once.
pub fn axis_choices_for_schema(schema: &PlotSchema) -> Vec<AxisRef> {
    let mut choices = Vec::new();
    for domain in DOMAINS {
        for axis in schema.domain(domain).axes() {
            if domain == AxisDomain::CellData && axis.role() == SCALAR { continue; }
            choices.push(declared_ref(domain, axis.id()));
        }
    }
    choices
}

/// How many distinct positions one reference spans.
pub fn axis_size(schema: &PlotSchema, reference: &AxisRef) -> Result<usize> { Ok(resolve_axis(schema, reference)?.size()) }

/// The declared shape grouped as a reader reads it: `(name, size)` per
/// primary axis, for the Repeat, Point and Cell domains.
pub fn schema_structure(schema: &PlotSchema) -> [Vec<(String, usize)>; 3] {
    DOMAINS.map(|domain| schema.domain(domain).axes().iter()
        .filter(|axis| !(domain == AxisDomain::CellData && axis.role() == SCALAR) && axis.coordinate_of().is_none())
        .map(|axis| (axis.name().to_owned(), axis.len())).collect())
}

/// The two cell axes that ARE the image, as (x, y), or `None`.
pub fn image_axes(schema: &PlotSchema) -> Option<(Arc<Axis>, Arc<Axis>)> {
    let axes = schema.domain(AxisDomain::CellData).axes();
    let by_role = |role: &str| axes.iter().rev().find(|axis| axis.role() == role).cloned();
    match (by_role(SPATIAL_X), by_role(SPATIAL_Y)) {
        (Some(x), Some(y)) => return Some((x, y)),
        (None, None) => {}
        _ => return None,
    }
    let significant: Vec<&Arc<Axis>> = axes.iter().filter(|axis| axis.len() > 1).collect();
    if significant.len() != 2 { return None; }
    Some((significant[1].clone(), significant[0].clone()))
}

/// One axis reference with the number of positions it spans.
pub type AxisEntry = (AxisRef, usize);

/// A dataset's axes grouped by what they ARE (see `defaults`).
#[derive(Clone, Debug)]
pub struct AxisFamilies {
    pub repeat: Vec<AxisEntry>,
    pub repeat_size: usize,
    pub history: Option<AxisEntry>,
    pub history_time: Option<AxisEntry>,
    pub sample_time: Option<AxisEntry>,
    pub scan: Vec<AxisEntry>,
    pub events: Vec<AxisEntry>,
    pub picture: Option<(AxisEntry, AxisEntry)>,
    pub content: Vec<AxisEntry>,
    pub data: Vec<AxisEntry>,
    pub has_point_axes: bool,
}

fn live(entries: &[AxisEntry]) -> Vec<AxisEntry> { entries.iter().filter(|entry| entry.1 > 1).cloned().collect() }

impl AxisFamilies {
    pub fn live_scan(&self) -> Vec<AxisEntry> { live(&self.scan) }
    pub fn live_events(&self) -> Vec<AxisEntry> { live(&self.events) }
    pub fn live_content(&self) -> Vec<AxisEntry> { live(&self.content) }
    pub fn live_data(&self) -> Vec<AxisEntry> { live(&self.data) }
}

/// Group a dataset's axes by declared role and place, never by name.
pub fn classify_axes(schema: &PlotSchema) -> AxisFamilies {
    let entry = |domain: AxisDomain, axis: &Arc<Axis>| (declared_ref(domain, axis.id()), axis.len());
    let repeat = schema.domain(AxisDomain::Repeat).axes().iter().map(|axis| entry(AxisDomain::Repeat, axis)).collect();
    let point_axes = schema.domain(AxisDomain::Point).axes();
    let (mut history, mut history_time, mut sample_time) = (None, None, None);
    let (mut scan, mut events, mut point_content) = (Vec::new(), Vec::new(), Vec::new());
    for axis in point_axes {
        let item = entry(AxisDomain::Point, axis);
        match axis.role() {
            PRIMARY_INDEX => { if history.is_none() { history = Some(item); } }
            SHOT_TIME => { if history_time.is_none() { history_time = Some(item); } }
            SAMPLE_TIME => { if sample_time.is_none() { sample_time = Some(item); } }
            SCAN_POINT => scan.push(item),
            READOUT_EVENT => events.push(item),
            _ => point_content.push(item),
        }
    }
    let pair = image_axes(schema);
    let picture = pair.as_ref().map(|(x, y)| (entry(AxisDomain::CellData, x), entry(AxisDomain::CellData, y)));
    let picture_ids: Vec<&str> = pair.as_ref().map(|(x, y)| vec![x.id(), y.id()]).unwrap_or_default();
    let cell_axes: Vec<AxisEntry> = schema.domain(AxisDomain::CellData).axes().iter()
        .filter(|axis| axis.role() != SCALAR).map(|axis| entry(AxisDomain::CellData, axis)).collect();
    let mut content = point_content.clone();
    content.extend(cell_axes.iter().filter(|entry| !picture_ids.contains(&&*entry.0.axis_id)).cloned());
    let mut data = point_content;
    data.extend(cell_axes);
    AxisFamilies {
        repeat, repeat_size: schema.domain(AxisDomain::Repeat).size(), history, history_time, sample_time,
        scan, events, picture, content, data, has_point_axes: !point_axes.is_empty(),
    }
}

/// The Point axes' `(id, name, role, unit, frame)` of a sample-time coordinate.
pub type SampleAxisIdentity = (Arc<str>, Arc<str>, Arc<str>, Option<Arc<str>>, Option<Arc<str>>);

/// What does not change as an indexed history's window slides.
#[derive(Clone, PartialEq)]
pub struct HistoryEvent {
    pub primary_axis_id: Arc<str>,
    pub time_axis_id: Option<Arc<str>>,
    pub repeat: Arc<Domain>,
    pub cell: Arc<Domain>,
    pub value: ValueSchema,
    pub event_size: usize,
    pub event_axes: Vec<Arc<Axis>>,
    pub event_codes: Vec<Vec<usize>>,
    pub sample_axes: Vec<SampleAxisIdentity>,
}

/// The one reading of a Runtime indexed history's Point domain.
#[derive(Clone)]
pub struct IndexedHistoryLayout {
    /// Each shot's relative offset, oldest first; none is above 0.
    pub cells: Vec<i64>,
    pub primary_axis_id: Arc<str>,
    pub time_axis_id: Option<Arc<str>>,
    /// Uniform rows per shot, when uniform.
    pub inner_count: Option<usize>,
    pub event: HistoryEvent,
    pub times: Option<Vec<f64>>,
    /// The Point domain the layout reads (its one mapping owner).
    pub point: Arc<Domain>,
}

impl IndexedHistoryLayout {
    pub fn shot_count(&self) -> usize { self.cells.len() }
    pub fn row_count(&self) -> usize { self.point.size() }
    /// Each requested Point row's shot position, oldest shot first.
    pub fn code(&self, row: usize) -> Option<usize> { self.point.mapping(&self.primary_axis_id)?.code(row) }
    /// The last `window` shots as the suffix `start..row_count` of the Point rows.
    pub fn window_rows(&self, window: i128) -> std::ops::Range<usize> {
        let keep = (window.max(1) as u128).min(self.cells.len() as u128) as usize;
        let first = self.shot_count() - keep;
        let mapping = self.point.mapping(&self.primary_axis_id).expect("primary mapping");
        let start = match mapping.base() {
            CodeBase::Range { .. } => first,
            CodeBase::Explicit(values) => values.as_slice().partition_point(|code| (*code as usize) < first),
        };
        start * mapping.inner()..self.row_count()
    }
    /// The record coordinate role this source identity names.
    pub fn coordinate_role(&self, axis_id: &str) -> Option<&'static str> {
        if *self.primary_axis_id == *axis_id { return Some(PRIMARY_INDEX); }
        if self.time_axis_id.as_deref() == Some(axis_id) { return Some(SHOT_TIME); }
        None
    }
}

fn base_first_last(mapping: &CodeMapping) -> (i128, i128) {
    match mapping.base() {
        CodeBase::Range { start, step, len } => (i128::from(*start), i128::from(*start) + i128::from(*step) * (*len as i128 - 1)),
        CodeBase::Explicit(values) => {
            let values = values.as_slice();
            (i128::from(values[0]), i128::from(values[values.len() - 1]))
        }
    }
}

fn base_value(mapping: &CodeMapping, position: usize) -> i128 {
    match mapping.base() {
        CodeBase::Range { start, step, .. } => i128::from(*start) + i128::from(*step) * position as i128,
        CodeBase::Explicit(values) => i128::from(values.as_slice()[position]),
    }
}

fn domain_codes(mapping: &CodeMapping, rows: usize) -> Vec<usize> {
    (0..rows).map(|row| mapping.code(row).expect("row inside the carrier")).collect()
}

fn history_error(message: &str) -> PlotError { PlotError::value(message) }

/// `indexed_history_layout`: the indexed-history layout of a schema, or
/// `None` without a primary index; a broken index contract is a ValueError.
pub fn indexed_history_layout(schema: &DatasetSchema) -> Result<Option<IndexedHistoryLayout>> {
    let point = schema.domain(DomainRole::Point);
    let primaries: Vec<&Arc<Axis>> = point.axes().iter().filter(|axis| axis.role() == PRIMARY_INDEX).collect();
    if primaries.is_empty() { return Ok(None); }
    if primaries.len() != 1 { return Err(history_error("an indexed history requires exactly one primary-index coordinate")); }
    let primary = primaries[0];
    if primary.coordinate_of().is_some() { return Err(history_error("the primary-index coordinate must own its record axis")); }
    let mut offsets = Vec::with_capacity(primary.len());
    for index in 0..primary.len() {
        match primary.coordinate(index).map(|value| canonical_coordinate(&value)) {
            Some(Coordinate::Integer(value)) if i64::try_from(value).is_ok() => offsets.push(value as i64),
            _ => return Err(history_error("primary-index coordinates must be integer relative offsets")),
        }
    }
    if offsets.iter().any(|offset| *offset > 0) {
        return Err(history_error("relative primary-index coordinates are at most the latest offset 0"));
    }
    if offsets.windows(2).any(|pair| pair[1] <= pair[0]) { return Err(history_error("primary-index offsets must form ordered cells")); }
    let shots = offsets.len();
    let time_axes: Vec<&Arc<Axis>> = point.axes().iter().filter(|axis| axis.role() == SHOT_TIME).collect();
    if time_axes.len() > 1 { return Err(history_error("an indexed history has at most one shot-time coordinate")); }
    let time_axis = time_axes.first().copied();
    if let Some(time) = time_axis {
        if !matches!(time.coordinate_of(), None) && time.coordinate_of() != Some(primary.id()) {
            return Err(history_error("the shot-time coordinate must correspond to the primary-index axis"));
        }
    }
    let mut counts: Option<Vec<usize>> = None;
    let checked: Vec<&Arc<Axis>> = std::iter::once(primary).chain(time_axis).collect();
    for axis in checked {
        let is_primary = Arc::ptr_eq(axis, primary);
        if !is_primary && axis.coordinate_of() == Some(primary.id()) { continue; }
        let mapping = point.mapping(axis.id()).expect("domain axis has a mapping");
        let (first, last) = base_first_last(mapping);
        let (inner, outer, base_len) = (mapping.inner(), mapping.outer(), mapping.base_len());
        let mut ordered = first == 0 && last == shots as i128 - 1 && (outer == 1 || shots == 1);
        let row_counts: Vec<usize> = match mapping.base() {
            CodeBase::Range { step, len, .. } => {
                ordered = ordered && (*len == 1 || *step == 1);
                vec![if shots == 1 { point.size() } else { inner }; shots]
            }
            CodeBase::Explicit(values) => {
                let values = values.as_slice();
                ordered = ordered && values.windows(2).all(|pair| matches!(pair[1] as i128 - pair[0] as i128, 0 | 1));
                let mut starts = vec![0usize];
                starts.extend(values.windows(2).enumerate().filter(|(_, pair)| pair[1] as i128 - pair[0] as i128 > 0).map(|(index, _)| index + 1));
                starts.push(base_len);
                starts.windows(2).map(|pair| (pair[1] - pair[0]) * inner * outer).collect()
            }
        };
        if !ordered || (!is_primary && Some(&row_counts) != counts.as_ref()) {
            return Err(history_error(if is_primary { "primary-index rows must form ordered contiguous cells" }
                                     else { "shot time is one coordinate per shot, on the primary index's rows" }));
        }
        if is_primary { counts = Some(row_counts); }
    }
    let counts = counts.expect("the primary was checked");
    let inner_count = counts[0];
    let uniform = counts.iter().all(|count| *count == inner_count);
    let times = match time_axis {
        Some(axis) => Some((0..axis.len()).map(|index| match axis.coordinate(index) {
            Some(Coordinate::Integer(value)) => Ok(value as f64),
            Some(Coordinate::Real(value)) => Ok(value),
            _ => Err(PlotError::value("shot-time coordinates must be real numbers")),
        }).collect::<Result<Vec<_>>>()?),
        None => None,
    };
    if times.as_ref().is_some_and(|values| values.len() != shots) {
        return Err(history_error("an indexed history stamps every shot or none"));
    }
    let event_axes: Vec<Arc<Axis>> = point.axes().iter()
        .filter(|axis| !Arc::ptr_eq(axis, primary) && !time_axis.is_some_and(|time| Arc::ptr_eq(axis, time)) && axis.role() != SAMPLE_TIME)
        .cloned().collect();
    let sample_axes = point.axes().iter().filter(|axis| axis.role() == SAMPLE_TIME)
        .map(|axis| (Arc::from(axis.id()), Arc::from(axis.name()), Arc::from(axis.role()), axis.unit().map(Arc::from), axis.coordinate_frame().map(Arc::from)))
        .collect();
    let mut repeated = uniform;
    for axis in &event_axes {
        if !repeated || shots == 1 { break; }
        let mapping = point.mapping(axis.id()).expect("domain axis has a mapping");
        let (inner, base_len) = (mapping.inner(), mapping.base_len());
        if inner_count % inner != 0 {
            repeated = match mapping.base() {
                CodeBase::Range { len, .. } => *len == 1,
                CodeBase::Explicit(values) => values.as_slice().iter().all(|value| *value == values.as_slice()[0]),
            };
        } else {
            let shift = (inner_count / inner) % base_len;
            if shift != 0 {
                repeated = match mapping.base() {
                    CodeBase::Range { .. } => false,
                    CodeBase::Explicit(_) => (0..base_len).all(|position| base_value(mapping, position) == base_value(mapping, (position + shift) % base_len)),
                };
            }
        }
    }
    let rows = if repeated && shots > 1 && !event_axes.is_empty() { inner_count } else { point.size() };
    let event_codes = event_axes.iter().map(|axis| domain_codes(point.mapping(axis.id()).unwrap(), rows)).collect();
    let event = HistoryEvent {
        primary_axis_id: Arc::from(primary.id()), time_axis_id: time_axis.map(|axis| Arc::from(axis.id())),
        repeat: schema.domain(DomainRole::Repeat).clone(), cell: schema.domain(DomainRole::Cell).clone(),
        value: schema.value().clone(), event_size: if repeated { inner_count } else { point.size() },
        event_axes, event_codes, sample_axes,
    };
    Ok(Some(IndexedHistoryLayout {
        cells: offsets, primary_axis_id: Arc::from(primary.id()), time_axis_id: time_axis.map(|axis| Arc::from(axis.id())),
        inner_count: uniform.then_some(inner_count), event, times, point: point.clone(),
    }))
}

/// Whether only an indexed Dataset's retained coordinate window changed.
pub fn indexed_schemas_compatible(left: &PlotSchema, right: &PlotSchema) -> Result<bool> {
    let (Some(left), Some(right)) = (left.history()?, right.history()?) else { return Ok(false) };
    Ok(left.inner_count == right.inner_count && left.event == right.event)
}

/// Whether a picture of `previous` takes `following` as new data.
pub fn schemas_continue(previous: &PlotSchema, following: &PlotSchema) -> Result<bool> {
    if previous == following { return Ok(true); }
    indexed_schemas_compatible(previous, following)
}
