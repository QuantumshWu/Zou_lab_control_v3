//! The `zlc_data` dataset model exactly as the Python projection reads it,
//! built from the native data plane (`zlc-core`) without copying planes.
//!
//! `AxisSpec`, `DomainSpec`, `DatasetSchema`, `DataBlock` (materialized or
//! segmented) and `OwnedSnapshot`, with the normalizations the Python
//! constructors apply (coordinate vocabulary, implicit index axes).

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use zlc_core::dataset::{DataBlock, Validity};
use zlc_core::schema::{Axis, CodeBase, Coordinate, Coordinates, DatasetSchema, Domain, DomainRole, ValidityContract};
use zlc_core::runtime::{Publication, Read, ReadKind};
use zlc_core::{AnyBuffer, OwnedBuffer};

use super::nd::{Array, DType, Mask, Scalar, product};
use super::ProjectionError;
use crate::kinds::{AxisDomain, AxisRef};

pub const REPEAT: &str = "repeat";
pub const PRIMARY_INDEX: &str = "primary-index";
pub const SHOT_TIME: &str = "shot-time";
pub const SAMPLE_TIME: &str = "sample-time";
pub const SCALAR: &str = "scalar";

/// An axis's coordinate vector after `zlc_data.axis._coordinates`.
#[derive(Clone, Debug)]
pub enum Coords {
    /// `index_origin + i`.
    Implicit,
    I64(Arc<[i64]>),
    U64(Arc<[u64]>),
    F64(Arc<[f64]>),
    /// Text or mixed coordinates (an object array in numpy).
    Values(Arc<[Scalar]>),
    /// Sample-time records: `origins[record] + offsets[sample]`.
    Factored { origins: Arc<[f64]>, offsets: Arc<[f64]> },
}

#[derive(Debug)]
pub struct AxisSpec {
    pub id: Arc<str>,
    pub name: Arc<str>,
    pub role: Arc<str>,
    pub size: usize,
    pub coords: Coords,
    pub unit: Option<Arc<str>>,
    pub frame: Option<Arc<str>>,
    pub index_origin: i128,
    pub labels: Option<Arc<[Arc<str>]>>,
    pub coordinate_of: Option<Arc<str>>,
    values: OnceLock<Array>,
}

/// `canonical_coordinate_scalar`: integral floats become ints.
pub fn canonical_scalar(value: Scalar) -> Result<Scalar, ProjectionError> {
    Ok(match value {
        Scalar::Float(v) => {
            if !v.is_finite() { return Err(ProjectionError::value("coordinate must be finite")); }
            if v.fract() == 0.0 && v.abs() < 1.7e38 { Scalar::Int(v as i128) } else { Scalar::Float(v) }
        }
        Scalar::Bool(_) => return Err(ProjectionError::type_error("coordinate must be null, text, or a finite number")),
        other => other,
    })
}

fn normalize_floats(values: Vec<f64>) -> Coords {
    if values.iter().all(|v| v.fract() == 0.0) {
        if values.iter().all(|v| *v >= -9.223372036854776e18 && *v < 9.223372036854776e18) {
            return Coords::I64(values.iter().map(|v| *v as i64).collect());
        }
        if values.iter().all(|v| *v >= 0.0 && *v < 1.8446744073709552e19) {
            return Coords::U64(values.iter().map(|v| *v as u64).collect());
        }
        return Coords::Values(values.iter().map(|v| Scalar::Int(*v as i128)).collect());
    }
    Coords::F64(values.into_iter().map(|v| if v == 0.0 { 0.0 } else { v }).collect())
}

impl Coords {
    fn from_ints(values: Vec<i128>) -> Coords {
        if values.iter().all(|v| *v >= i64::MIN as i128 && *v <= i64::MAX as i128) {
            Coords::I64(values.into_iter().map(|v| v as i64).collect())
        } else if values.iter().all(|v| *v >= 0 && *v <= u64::MAX as i128) {
            Coords::U64(values.into_iter().map(|v| v as u64).collect())
        } else {
            Coords::Values(values.into_iter().map(Scalar::Int).collect())
        }
    }

    /// `_coordinates(values)` for a numeric array input.
    fn from_numeric(buffer: &AnyBuffer) -> Coords {
        macro_rules! ints { ($b:expr) => { Coords::from_ints($b.as_slice().iter().map(|v| *v as i128).collect()) }; }
        match buffer {
            AnyBuffer::I64(b) => Coords::I64(b.as_slice().into()),
            AnyBuffer::U64(b) if b.as_slice().iter().any(|v| *v > i64::MAX as u64) => Coords::U64(b.as_slice().into()),
            AnyBuffer::U64(b) => ints!(b), AnyBuffer::U8(b) => ints!(b), AnyBuffer::I8(b) => ints!(b),
            AnyBuffer::U16(b) => ints!(b), AnyBuffer::I16(b) => ints!(b), AnyBuffer::U32(b) => ints!(b), AnyBuffer::I32(b) => ints!(b),
            AnyBuffer::F64(b) => normalize_floats(b.as_slice().to_vec()),
            AnyBuffer::F32(b) => normalize_floats(b.as_slice().iter().map(|v| f64::from(*v)).collect()),
            AnyBuffer::F16(b) => normalize_floats(b.as_slice().iter().map(|v| super::nd::half_to_f64(v.to_bits())).collect()),
            AnyBuffer::Bool(_) | AnyBuffer::C64(_) | AnyBuffer::C128(_) => Coords::Values(Arc::from(Vec::new())),
        }
    }

    /// `_coordinates(tuple)`: canonical scalars, an array where they are all numbers.
    fn from_scalars(values: Vec<Scalar>) -> Result<Coords, ProjectionError> {
        let normalized = values.into_iter().map(canonical_scalar).collect::<Result<Vec<_>, _>>()?;
        if !normalized.iter().all(Scalar::is_numeric) { return Ok(Coords::Values(normalized.into())); }
        let has_float = normalized.iter().any(|v| matches!(v, Scalar::Float(_)));
        if has_float && normalized.iter().any(|v| matches!(v, Scalar::Int(i) if i.abs() > 1 << 53)) {
            return Ok(Coords::Values(normalized.into()));
        }
        if !has_float {
            let ints: Vec<i128> = normalized.iter().map(|v| if let Scalar::Int(i) = v { *i } else { 0 }).collect();
            return Ok(Coords::from_ints(ints));
        }
        Ok(normalize_floats(normalized.iter().map(|v| v.as_f64().unwrap()).collect()))
    }
}

impl AxisSpec {
    #[allow(clippy::too_many_arguments)]
    pub fn new(id: Arc<str>, name: Arc<str>, role: Arc<str>, size: usize, coords: Coords, unit: Option<Arc<str>>,
               frame: Option<Arc<str>>, index_origin: i128, labels: Option<Arc<[Arc<str>]>>, coordinate_of: Option<Arc<str>>) -> Self {
        AxisSpec { id, name, role, size, coords, unit, frame, index_origin, labels, coordinate_of, values: OnceLock::new() }
    }

    fn from_core(axis: &Axis) -> Result<Self, ProjectionError> {
        let (coords, origin) = match axis.coordinates() {
            Coordinates::Index { origin, .. } if *origin >= 0 => (Coords::Implicit, *origin),
            Coordinates::Index { origin, len } => (Coords::from_ints((0..*len).map(|i| origin + i as i128).collect()), 0),
            Coordinates::Numeric(buffer) => (Coords::from_numeric(buffer), 0),
            Coordinates::Values(values) => (Coords::from_scalars(values.iter().map(|value| match value {
                Coordinate::Integer(v) => Scalar::Int(*v),
                Coordinate::Real(v) => Scalar::Float(*v),
                Coordinate::Text(v) => Scalar::Text(v.clone()),
                Coordinate::Missing => Scalar::None,
            }).collect())?, 0),
            Coordinates::OriginsAndOffsets { origins, offsets } =>
                (Coords::Factored { origins: origins.as_slice().into(), offsets: offsets.as_slice().into() }, 0),
        };
        Ok(AxisSpec::new(axis.id().into(), axis.name().into(), axis.role().into(), axis.len(), coords,
            axis.unit().map(Arc::from), axis.coordinate_frame().map(Arc::from), origin,
            axis.labels().map(|labels| labels.iter().cloned().collect()), axis.coordinate_of().map(Arc::from)))
    }

    /// `np.asarray(axis)`: `coordinate_values()`, an object array for scalars.
    pub fn values(&self) -> &Array {
        self.values.get_or_init(|| match &self.coords {
            Coords::Implicit => {
                if self.index_origin + self.size as i128 - 1 > i64::MAX as i128 {
                    Array::objects((0..self.size).map(|i| Scalar::Int(self.index_origin + i as i128)).collect(), vec![self.size])
                } else {
                    Array::vector((0..self.size as i64).map(|i| i + self.index_origin as i64).collect::<Vec<i64>>())
                }
            }
            Coords::I64(v) => Array::vector(v.to_vec()),
            Coords::U64(v) => Array::vector(v.to_vec()),
            Coords::F64(v) => Array::vector(v.to_vec()),
            Coords::Values(v) => Array::objects(v.to_vec(), vec![v.len()]),
            Coords::Factored { origins, offsets } =>
                Array::vector(origins.iter().flat_map(|o| offsets.iter().map(move |s| o + s)).collect::<Vec<f64>>()),
        })
    }

    /// `coordinate_at(index)`: the canonical scalar of one position.
    pub fn coordinate_at(&self, index: usize) -> Scalar {
        match &self.coords {
            Coords::Implicit => Scalar::Int(self.index_origin + index as i128),
            Coords::Factored { origins, offsets } => {
                let value = origins[index / offsets.len()] + offsets[index % offsets.len()];
                canonical_scalar(Scalar::Float(value)).unwrap_or(Scalar::Float(value))
            }
            Coords::F64(v) => canonical_scalar(Scalar::Float(v[index])).unwrap_or(Scalar::Float(v[index])),
            Coords::I64(v) => Scalar::Int(v[index] as i128),
            Coords::U64(v) => Scalar::Int(v[index] as i128),
            Coords::Values(v) => v[index].clone(),
        }
    }

    /// `coordinate_position(value)`: the one position holding this coordinate.
    pub fn coordinate_position(&self, value: &Scalar) -> Option<usize> {
        let value = canonical_scalar(value.clone()).ok()?;
        if let Coords::Implicit = self.coords {
            let Scalar::Int(v) = value else { return None };
            let position = v - self.index_origin;
            return (0..self.size as i128).contains(&position).then_some(position as usize);
        }
        (0..self.size).find(|index| self.coordinate_at(*index) == value && same_kind(&self.coordinate_at(*index), &value))
    }

    pub fn same_spec(&self, other: &AxisSpec) -> bool {
        std::ptr::eq(self, other) || (self.id == other.id && self.name == other.name && self.role == other.role
            && self.size == other.size && self.unit == other.unit && self.frame == other.frame
            && self.index_origin == other.index_origin && self.labels == other.labels
            && self.coordinate_of == other.coordinate_of && coords_equal(&self.coords, &other.coords))
    }
}

fn same_kind(a: &Scalar, b: &Scalar) -> bool { std::mem::discriminant(a) == std::mem::discriminant(b) || (a.is_numeric() && b.is_numeric()) }

fn coords_equal(a: &Coords, b: &Coords) -> bool {
    match (a, b) {
        (Coords::Implicit, Coords::Implicit) => true,
        (Coords::I64(x), Coords::I64(y)) => x == y,
        (Coords::U64(x), Coords::U64(y)) => x == y,
        (Coords::F64(x), Coords::F64(y)) => x.len() == y.len() && x.iter().zip(y.iter()).all(|(l, r)| l == r),
        (Coords::Values(x), Coords::Values(y)) => x == y,
        (Coords::Factored { origins: a, offsets: b }, Coords::Factored { origins: c, offsets: d }) => a == c && b == d,
        _ => false,
    }
}

/// One stored code vector and its declared repetition.
#[derive(Clone, Debug)]
pub enum Base { Range { start: i64, step: i64, len: usize }, Explicit(Arc<[i64]>) }

impl Base {
    pub fn len(&self) -> usize { match self { Base::Range { len, .. } => *len, Base::Explicit(v) => v.len() } }
    pub fn is_empty(&self) -> bool { self.len() == 0 }
    pub fn at(&self, position: usize) -> i64 {
        match self { Base::Range { start, step, .. } => start + position as i64 * step, Base::Explicit(v) => v[position] }
    }
    pub fn same(&self, other: &Base) -> bool {
        match (self, other) {
            (Base::Range { start: a, step: b, len: c }, Base::Range { start: d, step: e, len: f }) =>
                c == f && a == d && (b == e || *c == 1),
            (Base::Explicit(a), Base::Explicit(b)) => a == b,
            _ => false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Mapping { pub base: Base, pub inner: usize, pub outer: usize }

#[derive(Debug)]
pub struct DomainSpec {
    pub shape: Vec<usize>,
    pub axes: Vec<Arc<AxisSpec>>,
    /// None for a dense (Cell) domain: each axis is one physical dimension.
    pub mapping: Option<Vec<Mapping>>,
    codes: Vec<OnceLock<Arc<[i64]>>>,
}

impl DomainSpec {
    pub fn new(shape: Vec<usize>, axes: Vec<Arc<AxisSpec>>, mapping: Option<Vec<Mapping>>) -> Self {
        let count = axes.len();
        DomainSpec { shape, axes, mapping, codes: (0..count).map(|_| OnceLock::new()).collect() }
    }
    pub fn size(&self) -> usize { product(&self.shape) }
    pub fn index_of(&self, axis_id: &str) -> Option<usize> { self.axes.iter().position(|axis| &*axis.id == axis_id) }
    pub fn axis(&self, axis_id: &str) -> Option<&Arc<AxisSpec>> { self.axes.iter().find(|axis| &*axis.id == axis_id) }

    /// `code_mapping(axis_id)`.
    pub fn code_mapping(&self, index: usize) -> Mapping {
        match &self.mapping {
            None => Mapping { base: Base::Range { start: 0, step: 1, len: self.axes[index].size }, inner: 1, outer: 1 },
            Some(mapping) => mapping[index].clone(),
        }
    }
    /// The primary axis this (possibly alternative) coordinate names.
    pub fn coordinate_axis(&self, index: usize) -> usize {
        match &self.axes[index].coordinate_of { None => index, Some(primary) => self.index_of(primary).expect("validated primary") }
    }
    /// `codes(axis_id)`: the complete row code vector, expanded once.
    pub fn codes(&self, index: usize) -> Arc<[i64]> {
        let index = self.coordinate_axis(index);
        self.codes[index].get_or_init(|| {
            let Mapping { base, inner, outer } = self.code_mapping(index);
            let mut codes = Vec::with_capacity(base.len() * inner * outer);
            for _ in 0..outer {
                for position in 0..base.len() {
                    let code = base.at(position);
                    codes.extend(std::iter::repeat_n(code, inner));
                }
            }
            codes.into()
        }).clone()
    }
    /// `codes(axis_id, rows)`.
    pub fn codes_at(&self, index: usize, rows: &[i64]) -> Vec<i64> {
        let Mapping { base, inner, .. } = self.code_mapping(self.coordinate_axis(index));
        rows.iter().map(|row| base.at((*row as usize / inner) % base.len())).collect()
    }
    /// `physical_dimension(axis_id)` within the domain.
    pub fn physical_dimension(&self, index: usize) -> usize { if self.mapping.is_none() { index } else { 0 } }

    pub fn same_spec(&self, other: &DomainSpec) -> bool {
        if std::ptr::eq(self, other) { return true; }
        if self.shape != other.shape || self.axes.len() != other.axes.len()
            || !self.axes.iter().zip(&other.axes).all(|(a, b)| a.same_spec(b)) { return false; }
        match (&self.mapping, &other.mapping) {
            (None, None) => true,
            (Some(a), Some(b)) => a.iter().zip(b).all(|(l, r)| l.inner == r.inner && l.outer == r.outer && l.base.same(&r.base)),
            _ => false,
        }
    }
}

#[derive(Debug)]
pub struct ValueInfo {
    pub dtype: DType,
    pub unit: Option<Arc<str>>,
    pub name: Option<Arc<str>>,
    /// The component axes of a COMPONENTS validity contract, None for VALUE.
    pub components: Option<Vec<Arc<str>>>,
}

impl ValueInfo {
    pub fn same(&self, other: &ValueInfo) -> bool {
        self.dtype == other.dtype && self.unit == other.unit && self.name == other.name && self.components == other.components
    }
}

#[derive(Debug)]
pub struct Schema {
    pub repeat: DomainSpec,
    pub point: DomainSpec,
    pub cell: DomainSpec,
    pub value: ValueInfo,
    pub physical_shape: Vec<usize>,
    pub(crate) layout: OnceLock<Option<Arc<super::history::IndexedHistoryLayout>>>,
}

/// One resolved axis contract (`data_contract.ResolvedAxis`).
#[derive(Clone)]
pub struct AxisContract {
    pub domain: AxisDomain,
    pub index: usize,
    pub axis: Arc<AxisSpec>,
    /// The tensor dimension carrying this axis in `(R, P, *D)`.
    pub dimension: usize,
}

impl Schema {
    pub fn new(repeat: DomainSpec, point: DomainSpec, cell: DomainSpec, value: ValueInfo) -> Self {
        let physical_shape = [&repeat, &point, &cell].iter().flat_map(|d| d.shape.iter().copied()).collect();
        Schema { repeat, point, cell, value, physical_shape, layout: OnceLock::new() }
    }
    pub fn domain(&self, domain: AxisDomain) -> &DomainSpec {
        match domain { AxisDomain::Repeat => &self.repeat, AxisDomain::Point => &self.point, AxisDomain::CellData => &self.cell }
    }
    /// `resolve_axis(schema, ref)`.
    pub fn resolve(&self, axis: &AxisRef) -> Option<AxisContract> {
        let domain = self.domain(axis.domain);
        let index = domain.index_of(&axis.axis_id)?;
        let offset = match axis.domain { AxisDomain::Repeat => 0, AxisDomain::Point => self.repeat.shape.len(),
            AxisDomain::CellData => self.repeat.shape.len() + self.point.shape.len() };
        Some(AxisContract { domain: axis.domain, index, axis: domain.axes[index].clone(), dimension: offset + domain.physical_dimension(index) })
    }
    pub fn domain_of(&self, contract: &AxisContract) -> &DomainSpec { self.domain(contract.domain) }
    /// The compact validity storage shape `(R, P, *components)`.
    pub fn mask_shape(&self) -> Vec<usize> {
        let mut shape = vec![self.repeat.size(), self.point.size()];
        if let Some(components) = &self.value.components {
            for id in components { shape.push(self.cell.axis(id).map_or(1, |axis| axis.size)); }
        }
        shape
    }
    /// The full-plane broadcast shape of a compact mask over `leading` rows.
    pub fn mask_spread(&self, leading: &[usize]) -> Vec<usize> {
        let mut shape = leading.to_vec();
        for axis in &self.cell.axes {
            let named = self.value.components.as_ref().is_some_and(|ids| ids.iter().any(|id| *id == axis.id));
            shape.push(if named { axis.size } else { 1 });
        }
        shape
    }
    pub fn same(&self, other: &Schema) -> bool {
        std::ptr::eq(self, other) || (self.value.same(&other.value) && self.repeat.same_spec(&other.repeat)
            && self.point.same_spec(&other.point) && self.cell.same_spec(&other.cell))
    }
}

/// The planes of one immutable acquired rectangle: one native block, shared
/// by every revision that still holds it (where it sits is per revision).
#[derive(Debug)]
pub struct Segment {
    pub extent: [usize; 2],
    /// `(extent[0], extent[1], *cell_shape)`.
    pub values: Array,
    /// Compact validity `(extent..., *components)`, or uniform.
    pub mask: Mask,
    pub sigma: Option<Array>,
}

/// A segment at its place in one revision's Repeat x Point carriers (its
/// extent copied beside the origin: a layout walk reads no planes).
#[derive(Clone, Debug)]
pub struct Placed { pub origin: [usize; 2], pub extent: [usize; 2], pub segment: Arc<Segment> }

impl Placed {
    pub fn new(origin: [usize; 2], segment: Arc<Segment>) -> Self { Placed { origin, extent: segment.extent, segment } }
}

/// One placed segment as a revision sees it.
#[derive(Clone, Copy, Debug)]
pub struct PlacedRef<'a> { pub origin: [usize; 2], pub extent: [usize; 2], pub segment: &'a Arc<Segment> }

impl std::ops::Deref for PlacedRef<'_> { type Target = Segment; fn deref(&self) -> &Segment { self.segment } }

const PAGE: usize = 64;

/// A revision's placed segments in commit order, kept in shared pages of
/// 64: a window that drops its oldest shots and appends, or a finite run
/// that appends, shares every whole page of its predecessor, so a revision
/// costs what it brought, not a handle per segment it holds (thousands of
/// cold reference counts). Point origins are stored `shift` higher: a
/// window that slides moves every kept segment without rewriting it.
#[derive(Clone, Debug)]
pub struct Segments { pages: Arc<[Arc<[Placed]>]>, skip: usize, len: usize, shift: usize }

impl Segments {
    pub fn new(placed: Vec<Placed>) -> Self {
        let len = placed.len();
        let pages: Vec<Arc<[Placed]>> = placed.chunks(PAGE).map(Arc::from).collect();
        Segments { pages: pages.into(), skip: 0, len, shift: 0 }
    }
    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }
    pub fn get(&self, index: usize) -> PlacedRef<'_> {
        assert!(index < self.len, "segment {index} of {}", self.len);
        let at = self.skip + index;
        let placed = &self.pages[at / PAGE][at % PAGE];
        PlacedRef { origin: [placed.origin[0], placed.origin[1] - self.shift], extent: placed.extent, segment: &placed.segment }
    }
    pub fn iter(&self) -> impl Iterator<Item = PlacedRef<'_>> + '_ {
        self.pages.iter().flat_map(|page| page.iter()).skip(self.skip).take(self.len)
            .map(|placed| PlacedRef { origin: [placed.origin[0], placed.origin[1] - self.shift], extent: placed.extent, segment: &placed.segment })
    }
    /// Segments `drop..drop + keep` of these, seen `slide` lower along
    /// Point, followed by `tail` (at its own origins).
    fn continued(&self, drop: usize, keep: usize, tail: Vec<Placed>, slide: usize) -> Segments {
        let shift = self.shift + slide;
        let begin = self.skip + drop;
        let end = begin + keep;
        let (first, whole) = (begin / PAGE, end / PAGE);
        let added = tail.len();
        let mut pages: Vec<Arc<[Placed]>> = self.pages[first..whole].to_vec();
        // The open page: a partial last page is copied, never written in place.
        let mut open: Vec<Placed> = if end % PAGE != 0 { self.pages[whole][..end % PAGE].to_vec() } else { Vec::with_capacity(PAGE) };
        for mut placed in tail {
            placed.origin[1] += shift;
            open.push(placed);
            if open.len() == PAGE { pages.push(Arc::from(std::mem::replace(&mut open, Vec::with_capacity(PAGE)))); }
        }
        if !open.is_empty() { pages.push(open.into()); }
        Segments { pages: pages.into(), skip: begin - first * PAGE, len: keep + added, shift }
    }
}

#[derive(Clone, Debug)]
pub enum Block {
    /// Complete planes; unacquired cells are already invalid.
    Values { values: Array, mask: Mask, sigma: Option<Array> },
    /// Acquired rectangles in commit order; everything else is a hole.
    Segments(Segments),
}

/// `IndexedWindow`: absolute ordinals of the window this block was read as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Window { pub start: i128, pub latest: i128 }

/// `OwnedSnapshot`: one dataset revision.
#[derive(Clone)]
pub struct Snapshot {
    pub schema: Arc<Schema>,
    pub block: Block,
    pub window: Option<Window>,
    pub block_id: Arc<str>,
    pub generation: Arc<str>,
    pub revision: u64,
}

/// What a native read is: the zlc-core boundary of the projection.
pub struct Source {
    pub schema: Arc<DatasetSchema>,
    pub repeat_counts: Option<Vec<usize>>,
    pub data: SourceData,
    pub block_id: Arc<str>,
    pub generation: Arc<str>,
    pub revision: u64,
    /// `(start, latest)` absolute source ordinals of a history window.
    pub window: Option<(i128, i128)>,
}

pub enum SourceData { Dense(Arc<DataBlock>), Segments(Vec<([usize; 2], Arc<DataBlock>)>) }

/// How `zlc_python` names and lays out a read (`NativeRead.build_science`):
/// its block id, history window, and the placed blocks of a segmented one
/// (`None` for a dense event, whose planes are the block itself).
#[allow(clippy::type_complexity)]
fn read_layout(read: &Read) -> Result<(String, Option<Window>, Option<Vec<([usize; 2], &Arc<DataBlock>)>>), ProjectionError> {
    let key = read.key();
    Ok(match read.kind() {
        ReadKind::Event(_) => (format!("event:{}", key.signal), None, None),
        ReadKind::Canonical(event) => {
            let canonical = event.canonical().expect("canonical read");
            (format!("canonical:{}", key.signal), canonical.window().map(|(start, latest)| Window { start, latest: i128::from(latest) }),
                Some(canonical.segments().map(|p| (p.origin(), p.data())).collect()))
        }
        ReadKind::Window(window) => {
            let ordinal = key.ordinal.ok_or_else(|| ProjectionError::value("A history window requires a real source ordinal"))?;
            let placed = window.entries().iter().map(|entry| {
                let origin = window.placement(entry).ok_or_else(|| ProjectionError::value("Publication is outside its acquired window"))?;
                Ok((origin, entry.data().expect("history event payload")))
            }).collect::<Result<Vec<_>, ProjectionError>>()?;
            (format!("window:{}:{}:{}:{}", key.signal.len(), key.signal, window.width, window.view_epoch),
                Some(Window { start: i128::from(ordinal) + 1 - window.width as i128, latest: i128::from(ordinal) }), Some(placed))
        }
    })
}

fn event_block(read: &Read) -> &Arc<DataBlock> {
    match read.kind() { ReadKind::Event(event) => event.data().expect("checked event payload"), _ => unreachable!("segmented reads place blocks") }
}

impl Source {
    /// A runtime read with nothing carried: what `Snapshot::from_read`
    /// carries over, converted afresh (the reference the carry is held to).
    pub fn from_read(read: &Read) -> Result<Self, ProjectionError> {
        let key = read.key();
        let (block_id, window, placed) = read_layout(read)?;
        let data = match placed {
            None => SourceData::Dense(event_block(read).clone()),
            Some(placed) => SourceData::Segments(placed.into_iter().map(|(origin, block)| (origin, block.clone())).collect()),
        };
        Ok(Source { schema: read.schema().clone(), repeat_counts: Some(read.repeat_counts()), data, block_id: block_id.into(), generation: key.generation.as_uuid().to_string().into(),
            revision: key.revision, window: window.map(|w| (w.start, w.latest)) })
    }
    /// A figure archive's dataset under its recorded identity, dense or
    /// placed as its scientific block is.
    pub fn of_archived(dataset: &zlc_archive::ReadDataset) -> Self {
        let (snapshot, reference) = (&dataset.snapshot, &dataset.reference);
        let data = match dataset.source() {
            zlc_core::restrict::Source::Dense(block) => SourceData::Dense(block),
            zlc_core::restrict::Source::Placed { segments, .. } => SourceData::Segments(segments),
        };
        Source {
            schema: snapshot.schema().clone(), repeat_counts: Some(snapshot.repeat_counts().to_vec()), data,
            block_id: reference.block_id.as_str().into(), generation: reference.stream_generation.as_str().into(), revision: reference.revision,
            window: snapshot.window().map(|(start, latest)| (start, i128::from(latest))),
        }
    }
}

/// What the previous read placed, kept for the next (`NativeRead.from_read_reusing`
/// for the projection): its segments, the address each came from, and its
/// anchor publication.
///
/// The previous read itself is not kept: dropping a history window's entry
/// list costs a cold reference count per shot it holds, and identity does
/// not need it. Within one generation, signal, width and view epoch the
/// runtime's history is one deque that drops from the front, appends at the
/// back and revises only its newest ordinal, so every retained entry is
/// still held by the history (its address cannot have been reused) and the
/// one entry that can be revised, the previous anchor, is held here. A
/// finite run's blocks are held by its store for the generation.
#[derive(Default)]
pub struct SegmentExports { previous: Option<Previous> }

struct Previous { anchor: Arc<Publication>, window: Option<(usize, u64, Arc<DatasetSchema>)>, segments: Segments, ids: Vec<usize> }

impl SegmentExports {
    /// Place `count` entries in read order. `address(i)` names entry `i`
    /// without reading it. When the previous read's entries continue as the
    /// first of these (a window drops its oldest, a finite run keeps them
    /// all), they keep their segments and pages and sit `slide` lower along
    /// Point; only the entries after them are read (`fresh(i)`). Otherwise
    /// every entry is placed afresh. `slide` is `None` when nothing may be
    /// carried.
    fn place(&mut self, read: &Read, window: Option<(usize, u64, Arc<DatasetSchema>)>, count: usize, address: impl Fn(usize) -> usize,
             slide: Option<usize>, fresh: impl Fn(usize) -> Result<Placed, ProjectionError>) -> Result<Segments, ProjectionError> {
        let previous = self.previous.take().filter(|_| slide.is_some());
        let ids: Vec<usize> = (0..count).map(&address).collect();
        let carried = match &previous {
            // The aligned run always ends at the previous anchor, which is held:
            // a reused address cannot complete it.
            Some(Previous { segments, ids: old, .. }) if count > 0 => match old.iter().position(|id| *id == ids[0]) {
                Some(drop) if old.len() - drop <= count && old[drop..] == ids[..old.len() - drop] => Some((drop, old.len() - drop, segments)),
                _ => None,
            },
            _ => None,
        };
        let segments = match carried {
            Some((drop, keep, segments)) => {
                let tail = (keep..count).map(&fresh).collect::<Result<Vec<_>, _>>()?;
                segments.continued(drop, keep, tail, slide.unwrap_or(0))
            }
            None => Segments::new((0..count).map(&fresh).collect::<Result<Vec<_>, _>>()?),
        };
        self.previous = Some(Previous { anchor: read.anchor().clone(), window, segments: segments.clone(), ids });
        Ok(segments)
    }
}

impl Snapshot {
    /// The projection of a runtime read, as `zlc_python` exports it
    /// (`NativeRead.build_science`): planes viewed in place, and what the
    /// previous read already placed carried over.
    pub fn from_read(read: &Read, cache: &mut SchemaCache, exports: &mut SegmentExports) -> Result<Self, ProjectionError> {
        let schema = cache.get(read.schema())?;
        let key = read.key();
        let cell = &schema.cell.shape;
        let previous = exports.previous.as_ref().map(|p| (p.anchor.key(), p.window.clone()))
            .filter(|(old, _)| old.generation == key.generation && old.signal == key.signal);
        let (block, block_id, window) = match read.kind() {
            ReadKind::Event(event) => {
                let data = event.data().expect("checked event payload");
                exports.previous = None;
                (Block::Values { values: Array::from_buffer(data.values().clone(), schema.physical_shape.clone()), mask: mask_from_core(data),
                    sigma: sigma_from_core(data, &schema.physical_shape) }, format!("event:{}", key.signal), None)
            }
            ReadKind::Canonical(event) => {
                let canonical = event.canonical().expect("canonical read");
                let placements: Vec<&zlc_core::store::Placement> = canonical.segments().collect();
                let same = matches!(previous, Some((_, None)));
                let placed = exports.place(read, None, placements.len(), |i| Arc::as_ptr(placements[i].data()) as usize, same.then_some(0),
                    |i| Ok(Placed::new(placements[i].origin(), Arc::new(segment_of(placements[i].data(), cell)))))?;
                (Block::Segments(placed), format!("canonical:{}", key.signal), canonical.window().map(|(start, latest)| Window { start, latest: i128::from(latest) }))
            }
            ReadKind::Window(window) => {
                let ordinal = key.ordinal.ok_or_else(|| ProjectionError::value("A history window requires a real source ordinal"))?;
                let facts = (window.width, window.view_epoch, window.schema.clone());
                // A carried entry kept its ordinal: it sits as many slots lower
                // as the window's latest moved on.
                let slot = window.schema.domain(DomainRole::Point).size() / window.width.max(1);
                let slide = match previous {
                    Some((old, Some((width, epoch, schema)))) if width == facts.0 && epoch == facts.1 && Arc::ptr_eq(&schema, &facts.2) =>
                        old.ordinal.filter(|previous| *previous <= ordinal).map(|previous| (ordinal - previous) as usize * slot),
                    _ => None,
                };
                let entries = window.entries();
                let placed = exports.place(read, Some(facts), entries.len(), |i| Arc::as_ptr(&entries[i]) as usize, slide, |i| {
                    let origin = window.placement(&entries[i]).ok_or_else(|| ProjectionError::value("Publication is outside its acquired window"))?;
                    Ok(Placed::new(origin, Arc::new(segment_of(entries[i].data().expect("history event payload"), cell))))
                })?;
                (Block::Segments(placed), format!("window:{}:{}:{}:{}", key.signal.len(), key.signal, window.width, window.view_epoch),
                    Some(Window { start: i128::from(ordinal) + 1 - window.width as i128, latest: i128::from(ordinal) }))
            }
        };
        Ok(Snapshot { schema, block, window, block_id: block_id.into(), generation: key.generation.as_uuid().to_string().into(), revision: key.revision })
    }
}

/// Converted schemas, reused while the native schema owner is the same.
#[derive(Default)]
pub struct SchemaCache { entries: HashMap<usize, (Arc<DatasetSchema>, Arc<Schema>)> }

impl SchemaCache {
    pub fn get(&mut self, schema: &Arc<DatasetSchema>) -> Result<Arc<Schema>, ProjectionError> {
        let key = Arc::as_ptr(schema) as usize;
        if let Some((owner, converted)) = self.entries.get(&key) {
            if Arc::ptr_eq(owner, schema) { return Ok(converted.clone()); }
        }
        let converted = Arc::new(schema_from_core(schema)?);
        if self.entries.len() > 256 { self.entries.clear(); }
        self.entries.insert(key, (schema.clone(), converted.clone()));
        Ok(converted)
    }
}

fn domain_from_core(domain: &Domain, dense: bool) -> Result<DomainSpec, ProjectionError> {
    let axes = domain.axes().iter().map(|axis| AxisSpec::from_core(axis).map(Arc::new)).collect::<Result<Vec<_>, _>>()?;
    if dense {
        if domain.axes().len() != domain.physical_shape().len() || domain.axes().iter().any(|axis| axis.coordinate_of().is_some()) {
            return Err(ProjectionError::value("cell_domain must use dense implicit axis mapping"));
        }
        return Ok(DomainSpec::new(domain.physical_shape().to_vec(), axes, None));
    }
    if domain.physical_shape().len() != 1 {
        return Err(ProjectionError::value("an explicitly mapped domain must have one physical dimension"));
    }
    let mapping = domain.axes().iter().map(|axis| {
        let mapping = domain.mapping(axis.id()).expect("every axis has a mapping");
        let base = match mapping.base() {
            CodeBase::Range { start, step, len } => Base::Range { start: *start as i64, step: if *len == 1 { 1 } else { *step }, len: *len },
            CodeBase::Explicit(values) => Base::Explicit(values.as_slice().iter().map(|v| *v as i64).collect()),
        };
        Mapping { base, inner: mapping.inner(), outer: mapping.outer() }
    }).collect();
    Ok(DomainSpec::new(domain.physical_shape().to_vec(), axes, Some(mapping)))
}

pub fn schema_from_core(schema: &DatasetSchema) -> Result<Schema, ProjectionError> {
    let dtype = DType::from_core(schema.dtype())
        .ok_or_else(|| ProjectionError::view("complex dataset values require an explicit real-valued transform before plotting"))?;
    let value = schema.value();
    let components = match value.validity() {
        ValidityContract::Value => None,
        ValidityContract::Components(ids) => Some(ids.clone()),
    };
    Ok(Schema::new(
        domain_from_core(schema.domain(DomainRole::Repeat), false)?,
        domain_from_core(schema.domain(DomainRole::Point), false)?,
        domain_from_core(schema.domain(DomainRole::Cell), true)?,
        ValueInfo { dtype, unit: value.unit().map(Arc::from), name: value.name().map(Arc::from), components },
    ))
}

fn mask_from_core(block: &DataBlock) -> Mask {
    match block.validity() {
        Validity::Uniform(value) => Mask::Uniform(*value),
        Validity::Mask(mask) => Mask::Plane(super::nd::Bools::core(mask.clone())),
    }
}

/// One native block's planes, viewed in place.
fn segment_of(data: &DataBlock, cell: &[usize]) -> Segment {
    let extent = [data.schema().domain(DomainRole::Repeat).size(), data.schema().domain(DomainRole::Point).size()];
    let shape: Vec<usize> = extent.iter().copied().chain(cell.iter().copied()).collect();
    Segment { extent, values: Array::from_buffer(data.values().clone(), shape.clone()), mask: mask_from_core(data), sigma: sigma_from_core(data, &shape) }
}

fn sigma_from_core(block: &DataBlock, shape: &[usize]) -> Option<Array> {
    block.sigma().map(|sigma| Array::from_buffer(OwnedBuffer::clone(sigma).into(), shape.to_vec()))
}

impl Snapshot {
    pub fn from_source(source: &Source, cache: &mut SchemaCache) -> Result<Self, ProjectionError> {
        let schema = cache.get(&source.schema)?;
        let cell: Vec<usize> = schema.cell.shape.clone();
        let block = match &source.data {
            SourceData::Dense(data) => Block::Values {
                values: Array::from_buffer(data.values().clone(), schema.physical_shape.clone()),
                mask: mask_from_core(data),
                sigma: sigma_from_core(data, &schema.physical_shape),
            },
            SourceData::Segments(entries) => Block::Segments(Segments::new(entries.iter()
                .map(|(origin, data)| Placed::new(*origin, Arc::new(segment_of(data, &cell)))).collect())),
        };
        Ok(Snapshot {
            schema, block, window: source.window.map(|(start, latest)| Window { start, latest }),
            block_id: source.block_id.clone(), generation: source.generation.clone(), revision: source.revision,
        })
    }
    pub fn is_segmented(&self) -> bool { matches!(self.block, Block::Segments(_)) }
}
