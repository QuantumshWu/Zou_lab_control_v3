//! The single cutter (`zlc_data.selection` and the restriction half of
//! `zlc_data.snapshot_projection`): named selection terms resolved against a
//! schema, the schema of what survives (`restricted_schema`) and the planes of
//! what survives, for one dense block or for acquired placements.
//!
//! Coordinates are read the way the scientific `AxisSpec` holds them (an
//! implicit index, an int64/uint64/float64 vector, a tuple of mixed scalars or
//! factored sample times), so comparisons, normalizations and refusals equal
//! Python's. Errors carry the Python exception class.

use std::cmp::Ordering;
use std::fmt;
use std::sync::Arc;

use crate::dataset::{DataBlock, Validity};
use crate::runtime::{Read, ReadKind};
use crate::schema::{Axis, CodeBase, CodeMapping, Coordinate, Coordinates, DatasetSchema, Domain, DomainRole, ValidityContract};
use crate::store::{DatasetStore, Placement, Snapshot};
use crate::{AnyBuffer, OwnedBuffer};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ErrorKind { Value, Type, Index, EmptySelection }

impl ErrorKind {
    pub fn python_name(self) -> &'static str {
        match self { Self::Value => "ValueError", Self::Type => "TypeError", Self::Index => "IndexError", Self::EmptySelection => "EmptySelection" }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RestrictError { pub kind: ErrorKind, pub message: String }

impl RestrictError {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self { Self { kind, message: message.into() } }
    fn value(message: impl Into<String>) -> Self { Self::new(ErrorKind::Value, message) }
    fn type_error(message: impl Into<String>) -> Self { Self::new(ErrorKind::Type, message) }
    fn empty(message: impl Into<String>) -> Self { Self::new(ErrorKind::EmptySelection, message) }
}

impl fmt::Display for RestrictError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{}: {}", self.kind.python_name(), self.message) }
}

impl std::error::Error for RestrictError {}

pub type Result<T> = std::result::Result<T, RestrictError>;

/// A `range` (step 1) or a `tuple` of ascending indices.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Indices { Range(usize, usize), List(Vec<usize>) }

impl Indices {
    pub fn all(size: usize) -> Self { Self::Range(0, size) }
    pub fn len(&self) -> usize { match self { Self::Range(start, stop) => stop.saturating_sub(*start), Self::List(items) => items.len() } }
    pub fn is_empty(&self) -> bool { self.len() == 0 }
    pub fn at(&self, position: usize) -> usize { match self { Self::Range(start, _) => start + position, Self::List(items) => items[position] } }
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ { (0..self.len()).map(|position| self.at(position)) }
    /// `_keeps_everything`: Python compares ranges as sequences.
    pub fn keeps_everything(&self, size: usize) -> bool {
        match self {
            Self::Range(start, stop) => (*start == 0 && *stop == size) || (stop <= start && size == 0),
            Self::List(items) => items.len() == size && items.iter().enumerate().all(|(position, index)| position == *index),
        }
    }
    /// A range wherever the ascending indices are one contiguous run.
    fn from_sorted(items: Vec<usize>) -> Self {
        match (items.first(), items.last()) {
            (Some(first), Some(last)) if last - first + 1 == items.len() => Self::Range(*first, last + 1),
            _ => Self::List(items),
        }
    }
}

/// `canonical_coordinate_scalar`: an integral float is an int. A float of
/// magnitude 2^127 or more stays a float (Python makes it a larger int).
pub fn canonical_coordinate(value: &Coordinate, field: &str) -> Result<Coordinate> {
    match value {
        Coordinate::Real(real) if !real.is_finite() => Err(RestrictError::value(format!("{field} must be finite"))),
        Coordinate::Real(real) if real.fract() == 0.0 && real.abs() < 1.7014118346046923e38 => Ok(Coordinate::Integer(*real as i128)),
        Coordinate::Real(real) => Ok(Coordinate::Real(if *real == 0.0 { 0.0 } else { *real })),
        other => Ok(other.clone()),
    }
}

fn numeric(value: &Coordinate) -> bool { matches!(value, Coordinate::Integer(_) | Coordinate::Real(_)) }

/// Python's exact comparison of an int with a finite float.
fn compare_integer_real(integer: i128, real: f64) -> Ordering {
    if real >= 1.7014118346046923e38 { return Ordering::Less; }
    if real < -1.7014118346046923e38 { return Ordering::Greater; }
    let whole = real.trunc();
    match integer.cmp(&(whole as i128)) {
        Ordering::Equal => if real > whole { Ordering::Less } else if real < whole { Ordering::Greater } else { Ordering::Equal },
        other => other,
    }
}

/// Python's ordering of two numbers, exact across int and float.
pub fn compare_numbers(left: &Coordinate, right: &Coordinate) -> Option<Ordering> {
    match (left, right) {
        (Coordinate::Integer(a), Coordinate::Integer(b)) => Some(a.cmp(b)),
        (Coordinate::Real(a), Coordinate::Real(b)) => a.partial_cmp(b),
        (Coordinate::Integer(a), Coordinate::Real(b)) => Some(compare_integer_real(*a, *b)),
        (Coordinate::Real(a), Coordinate::Integer(b)) => Some(compare_integer_real(*b, *a).reverse()),
        _ => None,
    }
}

/// Python `==` of two canonical scalars.
pub fn same_scalar(left: &Coordinate, right: &Coordinate) -> bool {
    match (left, right) {
        (Coordinate::Text(a), Coordinate::Text(b)) => a == b,
        (Coordinate::Missing, Coordinate::Missing) => true,
        _ => compare_numbers(left, right) == Some(Ordering::Equal),
    }
}

/// One selection term (`IndexSelection`, `IndexRangeSelection`, `CoordinateRangeSelection`).
#[derive(Clone, Debug, PartialEq)]
pub enum Term {
    Index { axis_id: Arc<str>, index: usize },
    IndexRange { axis_id: Arc<str>, start: usize, stop: usize },
    Coordinate { axis_id: Arc<str>, lower: Coordinate, upper: Coordinate, frame: Option<Arc<str>> },
}

impl Term {
    pub fn index_range(axis_id: Arc<str>, start: usize, stop: usize) -> Result<Self> {
        if stop <= start { return Err(RestrictError::value("index range must be a non-empty half-open interval")); }
        Ok(Self::IndexRange { axis_id, start, stop })
    }

    /// A numeric closed interval, or one exact coordinate of any type.
    pub fn coordinate_range(axis_id: Arc<str>, lower: &Coordinate, upper: &Coordinate, frame: Option<Arc<str>>) -> Result<Self> {
        let lower = canonical_coordinate(lower, "coordinate lower")?;
        let upper = canonical_coordinate(upper, "coordinate upper")?;
        if numeric(&lower) && numeric(&upper) {
            if compare_numbers(&lower, &upper) == Some(Ordering::Greater) {
                return Err(RestrictError::value("coordinate range lower bound cannot exceed upper bound"));
            }
        } else if !same_scalar(&lower, &upper) {
            return Err(RestrictError::value("text or null coordinate selection must name one exact value"));
        }
        Ok(Self::Coordinate { axis_id, lower, upper, frame })
    }

    pub fn axis_id(&self) -> &Arc<str> {
        match self { Self::Index { axis_id, .. } | Self::IndexRange { axis_id, .. } | Self::Coordinate { axis_id, .. } => axis_id }
    }
}

/// One immutable selection over one or more named axes, sorted by axis id.
#[derive(Clone, Debug, PartialEq)]
pub struct Selection { terms: Vec<Term> }

impl Selection {
    pub fn new(mut terms: Vec<Term>) -> Result<Self> {
        if terms.is_empty() { return Err(RestrictError::value("Selection requires at least one term")); }
        let mut ids: Vec<&str> = terms.iter().map(|term| term.axis_id().as_ref()).collect();
        ids.sort_unstable();
        if ids.windows(2).any(|pair| pair[0] == pair[1]) { return Err(RestrictError::value("Selection may name each AxisId only once")); }
        terms.sort_by(|a, b| a.axis_id().cmp(b.axis_id()));
        Ok(Self { terms })
    }
    pub fn terms(&self) -> &[Term] { &self.terms }
}

/// An axis's coordinates as the scientific `AxisSpec` holds them after
/// `_coordinates`: an implicit index, an int64/uint64/float64 vector, a tuple
/// of canonical scalars, or factored sample times.
pub enum Held<'a> {
    Implicit { origin: i128 },
    I64(Vec<i64>),
    U64(Vec<u64>),
    F64(Vec<f64>),
    Values(Vec<Coordinate>),
    Factored { origins: &'a [f64], offsets: &'a [f64] },
}

/// What `_coordinates` makes of an explicit vector.
pub enum Vector { I64(Vec<i64>), U64(Vec<u64>), F64(Vec<f64>), Values(Vec<Coordinate>) }

impl Vector {
    /// `_coordinates(tuple)` of canonical scalars.
    pub fn of_scalars(values: Vec<Coordinate>) -> Self { scalars(values) }
    /// `_coordinates(float64 array)`.
    pub fn of_floats(values: Vec<f64>) -> Self { floats(values) }
    /// The native coordinates `NativeSchema.from_science` adopts.
    pub fn into_coordinates(self) -> Coordinates { coordinates_of(self) }
}

const TWO_63: f64 = 9.223372036854775808e18;
const TWO_64: f64 = 1.8446744073709551616e19;

fn floats(values: Vec<f64>) -> Vector {
    if values.iter().all(|value| value.fract() == 0.0) {
        if values.iter().all(|value| *value >= -TWO_63 && *value < TWO_63) { return Vector::I64(values.iter().map(|value| *value as i64).collect()); }
        if values.iter().all(|value| *value >= 0.0 && *value < TWO_64) { return Vector::U64(values.iter().map(|value| *value as u64).collect()); }
        return Vector::Values(values.iter().map(|value| canonical_coordinate(&Coordinate::Real(*value), "coordinate").expect("finite")).collect());
    }
    Vector::F64(values.into_iter().map(|value| if value == 0.0 { 0.0 } else { value }).collect())
}

fn unsigned(values: Vec<u64>) -> Vector {
    if values.iter().any(|value| *value > i64::MAX as u64) { Vector::U64(values) } else { Vector::I64(values.into_iter().map(|value| value as i64).collect()) }
}

/// `_coordinates(tuple)`: canonical scalars, a numeric vector where they all are numbers.
fn scalars(values: Vec<Coordinate>) -> Vector {
    if !values.iter().all(numeric) { return Vector::Values(values); }
    let any_float = values.iter().any(|value| matches!(value, Coordinate::Real(_)));
    if any_float && values.iter().any(|value| matches!(value, Coordinate::Integer(integer) if integer.unsigned_abs() > 1 << 53)) {
        return Vector::Values(values);
    }
    if !any_float {
        let integers: Vec<i128> = values.iter().map(|value| match value { Coordinate::Integer(integer) => *integer, _ => 0 }).collect();
        if values.is_empty() { return floats(Vec::new()); }
        let (low, high) = (*integers.iter().min().unwrap(), *integers.iter().max().unwrap());
        if low >= i128::from(i64::MIN) && high <= i128::from(i64::MAX) { return Vector::I64(integers.iter().map(|value| *value as i64).collect()); }
        if low >= 0 && high <= i128::from(u64::MAX) { return Vector::U64(integers.iter().map(|value| *value as u64).collect()); }
        return Vector::Values(values);
    }
    floats(values.iter().map(|value| match value { Coordinate::Integer(integer) => *integer as f64, Coordinate::Real(real) => *real, _ => 0.0 }).collect())
}

fn numeric_vector(buffer: &AnyBuffer) -> Vector {
    match buffer {
        AnyBuffer::I64(values) => Vector::I64(values.as_slice().to_vec()),
        AnyBuffer::U64(values) => unsigned(values.as_slice().to_vec()),
        AnyBuffer::F64(values) => floats(values.as_slice().to_vec()),
        AnyBuffer::F32(values) => floats(values.as_slice().iter().map(|value| f64::from(*value)).collect()),
        other => {
            let count = other.len();
            let coordinates = Coordinates::Numeric(other.clone());
            match coordinates.at(0) {
                Some(Coordinate::Real(_)) => floats((0..count).map(|index| match coordinates.at(index) { Some(Coordinate::Real(real)) => real, _ => 0.0 }).collect()),
                _ => Vector::I64((0..count).map(|index| match coordinates.at(index) { Some(Coordinate::Integer(integer)) => integer as i64, _ => 0 }).collect()),
            }
        }
    }
}

impl<'a> Held<'a> {
    pub fn of(axis: &'a Axis) -> Self {
        match axis.coordinates() {
            Coordinates::Index { origin, .. } if *origin >= 0 => Held::Implicit { origin: *origin },
            Coordinates::Index { origin, len } => Held::from_vector(scalars((0..*len).map(|index| Coordinate::Integer(origin + index as i128)).collect())),
            Coordinates::Numeric(buffer) => Held::from_vector(numeric_vector(buffer)),
            Coordinates::Values(values) => Held::from_vector(scalars(values.to_vec())),
            Coordinates::OriginsAndOffsets { origins, offsets } => Held::Factored { origins: origins.as_slice(), offsets: offsets.as_slice() },
        }
    }
    fn from_vector(vector: Vector) -> Self {
        match vector { Vector::I64(v) => Held::I64(v), Vector::U64(v) => Held::U64(v), Vector::F64(v) => Held::F64(v), Vector::Values(v) => Held::Values(v) }
    }

    /// `coordinate_at(index)`.
    pub fn at(&self, index: usize) -> Coordinate {
        let canonical = |real: f64| canonical_coordinate(&Coordinate::Real(real), "coordinate").unwrap_or(Coordinate::Real(real));
        match self {
            Held::Implicit { origin } => Coordinate::Integer(origin + index as i128),
            Held::I64(values) => Coordinate::Integer(i128::from(values[index])),
            Held::U64(values) => Coordinate::Integer(i128::from(values[index])),
            Held::F64(values) => canonical(values[index]),
            Held::Values(values) => values[index].clone(),
            Held::Factored { origins, offsets } => canonical(origins[index / offsets.len()] + offsets[index % offsets.len()]),
        }
    }

    /// `np.asarray(axis)` restricted to `indices` (`coordinate_values()` taken).
    fn taken(&self, size: usize, indices: &Indices) -> Vector {
        match self {
            Held::Implicit { origin } => {
                if origin + size as i128 - 1 > i128::from(i64::MAX) { return scalars(indices.iter().map(|index| self.at(index)).collect()); }
                Vector::I64(indices.iter().map(|index| (origin + index as i128) as i64).collect())
            }
            Held::I64(values) => Vector::I64(indices.iter().map(|index| values[index]).collect()),
            Held::U64(values) => unsigned(indices.iter().map(|index| values[index]).collect()),
            Held::F64(values) => floats(indices.iter().map(|index| values[index]).collect()),
            Held::Values(values) => scalars(indices.iter().map(|index| values[index].clone()).collect()),
            Held::Factored { origins, offsets } => floats(indices.iter().map(|index| origins[index / offsets.len()] + offsets[index % offsets.len()]).collect()),
        }
    }

    /// `coordinate_position(value)` for a canonical value.
    fn position(&self, size: usize, value: &Coordinate) -> Option<usize> {
        match self {
            Held::Implicit { origin } => {
                let Coordinate::Integer(integer) = value else { return None };
                let position = integer - origin;
                (0..size as i128).contains(&position).then_some(position as usize)
            }
            Held::I64(values) => {
                let Coordinate::Integer(integer) = value else { return None };
                let target = i64::try_from(*integer).ok()?;
                values.iter().position(|item| *item == target)
            }
            Held::U64(values) => {
                let Coordinate::Integer(integer) = value else { return None };
                let target = u64::try_from(*integer).ok()?;
                values.iter().position(|item| *item == target)
            }
            Held::F64(_) | Held::Factored { .. } => {
                if !numeric(value) { return None; }
                (0..size).find(|index| same_scalar(&self.at(*index), value))
            }
            Held::Values(values) => values.iter().position(|item| same_scalar(item, value)),
        }
    }
}

/// `float(int)` rounded inward: the nearest double not beyond the bound.
fn inward(bound: &Coordinate, upward: bool) -> f64 {
    match bound {
        Coordinate::Real(real) => *real,
        Coordinate::Integer(integer) => {
            let rounded = *integer as f64;
            let outside = compare_integer_real(*integer, rounded);
            if upward && outside == Ordering::Greater { rounded.next_up() } else if !upward && outside == Ordering::Less { rounded.next_down() } else { rounded }
        }
        _ => f64::NAN,
    }
}

/// numpy's comparison of an integer element with a Python scalar bound.
fn integer_at_least(item: i128, bound: &Coordinate) -> bool {
    match bound { Coordinate::Integer(integer) => item >= *integer, Coordinate::Real(real) => (item as f64) >= *real, _ => false }
}

fn integer_at_most(item: i128, bound: &Coordinate) -> bool {
    match bound { Coordinate::Integer(integer) => item <= *integer, Coordinate::Real(real) => (item as f64) <= *real, _ => false }
}

/// `resolve_selection_indices(axis, term)`.
fn resolve_term(axis: &Axis, term: &Term) -> Result<Indices> {
    let size = axis.len();
    let id = axis.id();
    match term {
        Term::Index { index, .. } => {
            if *index >= size { return Err(RestrictError::new(ErrorKind::Index, format!("selection index {index} is outside axis {id}"))); }
            Ok(Indices::Range(*index, index + 1))
        }
        Term::IndexRange { start, stop, .. } => {
            if *stop > size { return Err(RestrictError::new(ErrorKind::Index, format!("selection range stop {stop} is outside axis {id}"))); }
            Ok(Indices::Range(*start, *stop))
        }
        Term::Coordinate { lower, upper, frame, .. } => {
            if axis.coordinate_frame() != frame.as_deref() { return Err(RestrictError::value(format!("coordinate frame mismatch for axis {id}"))); }
            let empty = || RestrictError::empty(format!("coordinate selection is empty on axis {id}"));
            let held = Held::of(axis);
            if let Held::Implicit { origin } = held {
                if !numeric(lower) || !numeric(upper) { return Err(RestrictError::type_error(format!("implicit axis {id} accepts only numeric coordinates"))); }
                // `ceil(lower - index_origin)`: exact for an int, float arithmetic for a float.
                let shifted = |bound: &Coordinate, up: bool| match bound {
                    Coordinate::Integer(integer) => integer - origin,
                    Coordinate::Real(real) => { let value = real - origin as f64; (if up { value.ceil() } else { value.floor() }) as i128 }
                    _ => unreachable!(),
                };
                let (low, high) = (shifted(lower, true), shifted(upper, false));
                let (low, high) = (low.max(0), high.min(size as i128 - 1));
                if high < low { return Err(empty()); }
                return Ok(Indices::Range(low as usize, high as usize + 1));
            }
            let selected: Vec<usize> = if numeric(lower) && numeric(upper) && !same_scalar(lower, upper) {
                match &held {
                    Held::Values(values) => {
                        if values.iter().any(|value| !matches!(value, Coordinate::Missing) && !numeric(value)) {
                            return Err(RestrictError::type_error(format!("axis {id} coordinates are not entirely numeric")));
                        }
                        values.iter().enumerate().filter(|(_, value)| numeric(value)
                            && compare_numbers(lower, value) != Some(Ordering::Greater)
                            && compare_numbers(value, upper) != Some(Ordering::Greater)).map(|(index, _)| index).collect()
                    }
                    Held::I64(values) => range_of(values.iter().map(|value| i128::from(*value)), lower, upper),
                    Held::U64(values) => range_of(values.iter().map(|value| i128::from(*value)), lower, upper),
                    held => {
                        let values: Vec<f64> = match held {
                            Held::F64(values) => values.clone(),
                            Held::Factored { origins, offsets } => origins.iter().flat_map(|origin| offsets.iter().map(move |offset| origin + offset)).collect(),
                            _ => unreachable!(),
                        };
                        let low = inward(lower, true);
                        let mut chosen: Vec<usize> = (0..values.len()).filter(|index| values[*index] >= low).collect();
                        if !chosen.is_empty() {
                            let high = inward(upper, false);
                            chosen.retain(|index| values[*index] <= high);
                        }
                        chosen
                    }
                }
            } else {
                held.position(size, lower).into_iter().collect()
            };
            if selected.is_empty() { return Err(empty()); }
            Ok(Indices::from_sorted(selected))
        }
    }
}

fn range_of(values: impl Iterator<Item = i128> + Clone, lower: &Coordinate, upper: &Coordinate) -> Vec<usize> {
    let chosen: Vec<usize> = values.clone().enumerate().filter(|(_, value)| integer_at_least(*value, lower)).map(|(index, _)| index).collect();
    if chosen.is_empty() { return chosen; }
    let values: Vec<i128> = values.collect();
    chosen.into_iter().filter(|index| integer_at_most(values[*index], upper)).collect()
}

/// What one selection keeps: physical Repeat rows, Point rows and the
/// logical indices of every Cell axis (in Cell-domain order).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SelectedIndices { pub repeat: Indices, pub point: Indices, pub cells: Vec<Indices> }

impl SelectedIndices {
    /// Repeat/Point row ranges with every Cell coordinate kept.
    pub fn rows(schema: &DatasetSchema, repeat: Indices, point: Indices) -> Self {
        Self { repeat, point, cells: schema.domain(DomainRole::Cell).axes().iter().map(|axis| Indices::all(axis.len())).collect() }
    }
}

fn mapping_of(domain: &Domain, axis: &Axis) -> Arc<CodeMapping> { domain.mapping(axis.id()).expect("every domain axis has a mapping").clone() }

/// `selection_indices(schema, selection)`.
pub fn selection_indices(schema: &DatasetSchema, selection: Option<&Selection>) -> Result<SelectedIndices> {
    let terms: &[Term] = selection.map_or(&[], Selection::terms);
    let roles = [DomainRole::Repeat, DomainRole::Point, DomainRole::Cell];
    for term in terms {
        if !roles.iter().any(|role| schema.domain(*role).axis(term.axis_id()).is_some()) {
            return Err(RestrictError::value(format!("selection axis {} is absent from source schema", term.axis_id())));
        }
    }
    let term_of = |axis: &Axis| terms.iter().find(|term| term.axis_id().as_ref() == axis.id());
    let logical = |axis: &Axis| -> Result<Indices> { term_of(axis).map_or(Ok(Indices::all(axis.len())), |term| resolve_term(axis, term)) };
    let physical = |domain: &Domain| -> Result<Indices> {
        let selected: Vec<&Arc<Axis>> = domain.axes().iter().filter(|axis| term_of(axis).is_some()).collect();
        if selected.is_empty() { return Ok(Indices::all(domain.size())); }
        let nothing = || RestrictError::empty("selection names coordinates no source domain row holds together");
        if selected.len() == 1 {
            let chosen = logical(selected[0])?;
            let mapping = mapping_of(domain, selected[0]);
            if let (Indices::Range(start, stop), CodeBase::Range { start: base, step: 1, len }) = (&chosen, mapping.base()) {
                if mapping.outer() == 1 {
                    let base = i128::from(*base);
                    let first = (*start as i128 - base).max(0);
                    let last = (*stop as i128 - base).min(*len as i128);
                    if last <= first { return Err(nothing()); }
                    return Ok(Indices::Range(first as usize * mapping.inner(), last as usize * mapping.inner()));
                }
            }
        }
        let mut keep = vec![true; domain.size()];
        for axis in selected {
            let chosen = logical(axis)?;
            let mapping = mapping_of(domain, axis);
            let wanted: Option<std::collections::HashSet<usize>> = match &chosen { Indices::List(items) => Some(items.iter().copied().collect()), _ => None };
            for (row, slot) in keep.iter_mut().enumerate() {
                let code = mapping.code(row).expect("row inside carrier");
                *slot &= match (&chosen, &wanted) { (Indices::Range(start, stop), _) => (*start..*stop).contains(&code), (_, Some(set)) => set.contains(&code), _ => false };
            }
        }
        let rows: Vec<usize> = keep.iter().enumerate().filter(|(_, kept)| **kept).map(|(row, _)| row).collect();
        if rows.is_empty() { return Err(nothing()); }
        Ok(Indices::from_sorted(rows))
    };
    let repeat = physical(schema.domain(DomainRole::Repeat))?;
    let point = physical(schema.domain(DomainRole::Point))?;
    let cells = schema.domain(DomainRole::Cell).axes().iter().map(|axis| logical(axis)).collect::<Result<Vec<_>>>()?;
    Ok(SelectedIndices { repeat, point, cells })
}

fn coordinates_of(vector: Vector) -> Coordinates {
    match vector {
        Vector::I64(values) => Coordinates::Numeric(OwnedBuffer::from_vec(values).into()),
        Vector::U64(values) => Coordinates::Numeric(OwnedBuffer::from_vec(values).into()),
        Vector::F64(values) => Coordinates::Numeric(OwnedBuffer::from_vec(values).into()),
        Vector::Values(values) => Coordinates::Values(values.into()),
    }
}

fn invalid(error: String) -> RestrictError { RestrictError::value(error) }

/// `_subset_axis(axis, indices)`.
fn subset_axis(axis: &Arc<Axis>, indices: &Indices) -> Result<Arc<Axis>> {
    if indices.keeps_everything(axis.len()) { return Ok(axis.clone()); }
    let labels = axis.labels().map(|labels| indices.iter().map(|index| labels[index].clone()).collect::<Vec<_>>());
    let held = Held::of(axis);
    let coordinates = match (&held, indices) {
        (Held::Implicit { origin }, Indices::Range(start, _)) => Coordinates::Index { origin: origin + *start as i128, len: indices.len() },
        (Held::Factored { origins, offsets }, Indices::Range(start, stop)) if start % offsets.len() == 0 && stop % offsets.len() == 0 => {
            let width = offsets.len();
            let Coordinates::OriginsAndOffsets { offsets: shared, .. } = axis.coordinates() else { unreachable!() };
            Coordinates::OriginsAndOffsets { origins: OwnedBuffer::from_vec(origins[start / width..stop / width].to_vec()), offsets: shared.clone() }
        }
        _ => coordinates_of(held.taken(axis.len(), indices)),
    };
    Axis::new(axis.id().into(), axis.name().into(), axis.role().into(), coordinates, axis.unit().map(Arc::from),
        axis.coordinate_frame().map(Arc::from), labels, axis.coordinate_of().map(Arc::from)).map_err(invalid)
}

/// The sorted distinct values of `codes` and each code's position among them.
fn unique_inverse(codes: &[u64]) -> (Vec<u64>, Vec<u64>) {
    let mut used = codes.to_vec();
    used.sort_unstable();
    used.dedup();
    let inverse = codes.iter().map(|code| used.binary_search(code).expect("present") as u64).collect();
    (used, inverse)
}

/// `_subset_mapped_domain(domain, indices)`.
fn subset_mapped_domain(domain: &Arc<Domain>, indices: &Indices) -> Result<Arc<Domain>> {
    if indices.keeps_everything(domain.size()) { return Ok(domain.clone()); }
    let mut axes = Vec::with_capacity(domain.axes().len());
    let mut mappings = Vec::with_capacity(domain.axes().len());
    for axis in domain.axes() {
        let mapping = mapping_of(domain, axis);
        let (inner, outer, base_len) = (mapping.inner(), mapping.outer(), mapping.base_len());
        let period = base_len * inner;
        // Kept codes as a range (start, step, len) or a vector, with their repeats.
        enum Kept { Range(i128, i64, usize), Codes(Vec<u64>) }
        let (kept, inner, tile) = match indices {
            Indices::Range(start, stop) if outer == 1 && start % inner == 0 && stop % inner == 0 => {
                let (first, last) = (start / inner, stop / inner);
                (match mapping.base() {
                    CodeBase::Range { start, step, .. } => Kept::Range(i128::from(*start) + first as i128 * i128::from(*step), *step, last - first),
                    CodeBase::Explicit(codes) => Kept::Codes(codes.as_slice()[first..last].to_vec()),
                }, inner, 1)
            }
            Indices::Range(start, stop) if start % period == 0 && stop % period == 0 => (match mapping.base() {
                CodeBase::Range { start, step, len } => Kept::Range(i128::from(*start), *step, *len),
                CodeBase::Explicit(codes) => Kept::Codes(codes.as_slice().to_vec()),
            }, inner, (stop - start) / period),
            _ => (Kept::Codes(indices.iter().map(|row| mapping.code(row).expect("row inside carrier") as u64).collect()), 1, 1),
        };
        let (used, inverse): (Vec<u64>, CodeBase) = match kept {
            Kept::Range(start, step, len) if step > 0 || len <= 1 => {
                ((0..len as i128).map(|position| (start + position * i128::from(step)) as u64).collect(), CodeBase::Range { start: 0, step: 1, len })
            }
            Kept::Range(start, step, len) => {
                let codes: Vec<u64> = (0..len as i128).map(|position| (start + position * i128::from(step)) as u64).collect();
                let (used, inverse) = unique_inverse(&codes);
                (used, natural(inverse))
            }
            Kept::Codes(codes) => {
                let (used, inverse) = unique_inverse(&codes);
                (used, natural(inverse))
            }
        };
        if used.len() == axis.len() {
            axes.push(axis.clone());
        } else {
            let picks: Vec<usize> = used.iter().map(|code| *code as usize).collect();
            axes.push(subset_axis(axis, &Indices::from_sorted(picks))?);
        }
        mappings.push(CodeMapping::new(inverse, inner, tile).map_err(invalid)?);
    }
    Domain::new(vec![indices.len()], axes, mappings).map_err(invalid)
}

/// An inverse vector that is `0..n` is the range it names.
fn natural(inverse: Vec<u64>) -> CodeBase {
    if inverse.iter().enumerate().all(|(position, code)| *code == position as u64) {
        CodeBase::Range { start: 0, step: 1, len: inverse.len() }
    } else {
        CodeBase::Explicit(OwnedBuffer::from_vec(inverse))
    }
}

/// `restricted_schema(schema, repeat, point, data)`: the source's axes, cropped.
pub fn restricted_schema(schema: &Arc<DatasetSchema>, indices: &SelectedIndices) -> Result<Arc<DatasetSchema>> {
    let repeat = subset_mapped_domain(schema.domain(DomainRole::Repeat), &indices.repeat)?;
    let point = subset_mapped_domain(schema.domain(DomainRole::Point), &indices.point)?;
    let cell = schema.domain(DomainRole::Cell);
    let axes = cell.axes().iter().zip(&indices.cells).map(|(axis, picks)| subset_axis(axis, picks)).collect::<Result<Vec<_>>>()?;
    let cell = if axes.iter().zip(cell.axes()).all(|(new, old)| Arc::ptr_eq(new, old)) { cell.clone() } else {
        Domain::dense(axes.iter().map(|axis| axis.len()).collect(), axes).map_err(invalid)?
    };
    if Arc::ptr_eq(&repeat, schema.domain(DomainRole::Repeat)) && Arc::ptr_eq(&point, schema.domain(DomainRole::Point))
        && Arc::ptr_eq(&cell, schema.domain(DomainRole::Cell)) {
        return Ok(schema.clone());
    }
    DatasetSchema::new(repeat, point, cell, schema.value().clone()).map_err(invalid)
}

/// The elements of a C-order array of `shape` kept by `picks` on every axis.
pub fn gather<T: Copy>(source: &[T], shape: &[usize], picks: &[&Indices]) -> Vec<T> {
    let count = shape.len();
    let mut strides = vec![1usize; count];
    for axis in (0..count.saturating_sub(1)).rev() { strides[axis] = strides[axis + 1] * shape[axis + 1]; }
    let mut tail = count;
    while tail > 0 && picks[tail - 1].keeps_everything(shape[tail - 1]) { tail -= 1; }
    if tail == 0 { return source.to_vec(); }
    let total: usize = picks.iter().map(|picks| picks.len()).product();
    let mut out = Vec::with_capacity(total);
    fn walk<T: Copy>(source: &[T], strides: &[usize], picks: &[&Indices], tail: usize, axis: usize, offset: usize, out: &mut Vec<T>) {
        let unit = strides[axis];
        if axis + 1 == tail {
            match picks[axis] {
                Indices::Range(start, stop) => out.extend_from_slice(&source[offset + start * unit..offset + stop * unit]),
                Indices::List(items) => for index in items { out.extend_from_slice(&source[offset + index * unit..offset + (index + 1) * unit]); },
            }
            return;
        }
        for index in picks[axis].iter() { walk(source, strides, picks, tail, axis + 1, offset + index * unit, out); }
    }
    walk(source, &strides, picks, tail, 0, 0, &mut out);
    out
}

macro_rules! each_buffer {
    ($buffer:expr, $values:ident => $body:expr) => {
        match $buffer {
            AnyBuffer::Bool($values) => AnyBuffer::from($body), AnyBuffer::U8($values) => AnyBuffer::from($body),
            AnyBuffer::I8($values) => AnyBuffer::from($body), AnyBuffer::U16($values) => AnyBuffer::from($body),
            AnyBuffer::I16($values) => AnyBuffer::from($body), AnyBuffer::U32($values) => AnyBuffer::from($body),
            AnyBuffer::I32($values) => AnyBuffer::from($body), AnyBuffer::U64($values) => AnyBuffer::from($body),
            AnyBuffer::I64($values) => AnyBuffer::from($body), AnyBuffer::F16($values) => AnyBuffer::from($body),
            AnyBuffer::F32($values) => AnyBuffer::from($body), AnyBuffer::F64($values) => AnyBuffer::from($body),
            AnyBuffer::C64($values) => AnyBuffer::from($body), AnyBuffer::C128($values) => AnyBuffer::from($body),
        }
    };
}

/// `gather` over a typed buffer of any dtype.
pub fn gather_buffer(buffer: &AnyBuffer, shape: &[usize], picks: &[&Indices]) -> AnyBuffer {
    each_buffer!(buffer, values => OwnedBuffer::from_vec(gather(values.as_slice(), shape, picks)))
}

/// The component-axis positions of a schema's validity contract.
fn component_positions(schema: &DatasetSchema) -> Vec<usize> {
    match schema.value().validity() {
        ValidityContract::Value => Vec::new(),
        ValidityContract::Components(ids) => {
            let cell = schema.domain(DomainRole::Cell);
            ids.iter().map(|id| cell.axes().iter().position(|axis| axis.id() == id.as_ref()).expect("validated component")).collect()
        }
    }
}

/// One block's planes cut by `repeat`, `point` and the Cell picks.
struct Cut { values: AnyBuffer, validity: Validity, sigma: Option<OwnedBuffer<f64>> }

fn cut_block(schema: &DatasetSchema, block: &DataBlock, repeat: &Indices, point: &Indices, cells: &[Indices]) -> Cut {
    let block_schema = block.schema();
    let leading = [block_schema.domain(DomainRole::Repeat).size(), block_schema.domain(DomainRole::Point).size()];
    let cell_shape = block_schema.domain(DomainRole::Cell).physical_shape();
    let shape: Vec<usize> = leading.iter().chain(cell_shape).copied().collect();
    let mut picks: Vec<&Indices> = vec![repeat, point];
    picks.extend(cells.iter());
    let values = gather_buffer(block.values(), &shape, &picks);
    let sigma = block.sigma().map(|sigma| OwnedBuffer::from_vec(gather(sigma.as_slice(), &shape, &picks)));
    let validity = match block.validity() {
        Validity::Uniform(flag) => Validity::Uniform(*flag),
        Validity::Mask(mask) => {
            let positions = component_positions(schema);
            let mask_shape: Vec<usize> = leading.iter().copied().chain(positions.iter().map(|position| cell_shape[*position])).collect();
            let mut mask_picks: Vec<&Indices> = vec![repeat, point];
            mask_picks.extend(positions.iter().map(|position| &cells[*position]));
            Validity::Mask(OwnedBuffer::from_vec(gather(mask.as_slice(), &mask_shape, &mask_picks)))
        }
    };
    Cut { values, validity, sigma }
}

/// `compact_dataset_validity` of a compact plane: one value when uniform.
fn compacted(validity: Validity) -> Validity {
    match validity {
        Validity::Mask(mask) if mask.as_slice().iter().all(|flag| *flag) => Validity::Uniform(true),
        Validity::Mask(mask) if !mask.as_slice().iter().any(|flag| *flag) => Validity::Uniform(false),
        other => other,
    }
}

/// Immutable planes as a scientific read holds them: one dense block (an
/// event) or acquired placements in commit order (a finite prefix or a
/// history window), with the window's absolute source ordinals.
#[derive(Clone)]
pub enum Source {
    Dense(Arc<DataBlock>),
    Placed { schema: Arc<DatasetSchema>, segments: Vec<([usize; 2], Arc<DataBlock>)>, window: Option<(i128, u64)> },
}

impl Source {
    /// The planes `zlc_python.NativeRead.to_science` exports for a read.
    pub fn of_read(read: &Read) -> std::result::Result<Self, String> {
        Ok(match read.kind() {
            ReadKind::Event(event) => Source::Dense(event.data().ok_or("Event read has no payload")?.clone()),
            ReadKind::Canonical(event) => {
                let canonical = event.canonical().ok_or("Publication has no finite canonical storage")?;
                Source::Placed { schema: canonical.schema().clone(), segments: canonical.segments().map(|entry| (entry.origin(), entry.data().clone())).collect(),
                    window: canonical.window() }
            }
            ReadKind::Window(window) => {
                let ordinal = window.latest.ordinal.ok_or("A history window requires a real source ordinal")?;
                let segments = window.entries().iter().map(|entry| {
                    let origin = window.placement(entry).ok_or("Publication is outside its acquired window")?;
                    Ok((origin, entry.data().ok_or("History event has no payload")?.clone()))
                }).collect::<std::result::Result<Vec<_>, String>>()?;
                Source::Placed { schema: window.schema.clone(), segments, window: Some((i128::from(ordinal) + 1 - window.width as i128, ordinal)) }
            }
        })
    }

    pub fn of_snapshot(snapshot: &Snapshot) -> Self {
        Source::Placed { schema: snapshot.schema().clone(), segments: snapshot.segments().map(|entry| (entry.origin(), entry.data().clone())).collect(),
            window: snapshot.window() }
    }

    pub fn schema(&self) -> &Arc<DatasetSchema> { match self { Source::Dense(block) => block.schema(), Source::Placed { schema, .. } => schema } }
}

/// One schema's selection, resolved once: its indices and the schema of what survives.
#[derive(Clone)]
pub struct Restriction { source: Arc<DatasetSchema>, indices: SelectedIndices, schema: Arc<DatasetSchema> }

impl Restriction {
    pub fn new(source: &Arc<DatasetSchema>, selection: Option<&Selection>) -> Result<Self> {
        let indices = selection_indices(source, selection)?;
        let schema = restricted_schema(source, &indices)?;
        Ok(Self { source: source.clone(), indices, schema })
    }
    pub fn source(&self) -> &Arc<DatasetSchema> { &self.source }
    pub fn indices(&self) -> &SelectedIndices { &self.indices }
    pub fn schema(&self) -> &Arc<DatasetSchema> { &self.schema }

    /// `restrict_snapshot` then `NativeDataset.from_science`: a dense block
    /// becomes one complete placement in the restricted schema; acquired
    /// placements are cut one by one and keep their order, holes and window.
    pub fn apply(&self, source: &Source) -> std::result::Result<Snapshot, String> {
        if **source.schema() != *self.source { return Err("Restriction source schema differs from its plan".into()); }
        let indices = &self.indices;
        let (placements, window) = match source {
            Source::Dense(block) => {
                let cut = cut_block(&self.source, block, &indices.repeat, &indices.point, &indices.cells);
                let data = DataBlock::new(self.schema.clone(), cut.values, compacted(cut.validity), cut.sigma)?;
                let extent = [self.schema.domain(DomainRole::Repeat).size(), self.schema.domain(DomainRole::Point).size()];
                (vec![Placement::new([0, 0], extent, data)?], None)
            }
            Source::Placed { segments, window, .. } => {
                let keeps_cells = indices.cells.iter().zip(self.source.domain(DomainRole::Cell).axes()).all(|(picks, axis)| picks.keeps_everything(axis.len()));
                let mut placements = Vec::with_capacity(segments.len());
                for (origin, block) in segments {
                    let extent = [block.schema().domain(DomainRole::Repeat).size(), block.schema().domain(DomainRole::Point).size()];
                    let (repeats, repeat_origin) = overlap(&indices.repeat, origin[0], extent[0]);
                    let (points, point_origin) = overlap(&indices.point, origin[1], extent[1]);
                    if repeats.is_empty() || points.is_empty() { continue; }
                    let cut = if repeats.keeps_everything(extent[0]) && points.keeps_everything(extent[1]) && keeps_cells {
                        Cut { values: block.values().clone(), validity: block.validity().clone(), sigma: block.sigma().cloned() }
                    } else {
                        cut_block(&self.source, block, &repeats, &points, &indices.cells)
                    };
                    let chunk = restricted_schema(&self.schema, &SelectedIndices::rows(&self.schema,
                        Indices::Range(repeat_origin, repeat_origin + repeats.len()), Indices::Range(point_origin, point_origin + points.len())))
                        .map_err(|error| error.to_string())?;
                    let data = DataBlock::new(chunk, cut.values, cut.validity, cut.sigma)?;
                    placements.push(Placement::new([repeat_origin, point_origin], [repeats.len(), points.len()], data)?);
                }
                (placements, *window)
            }
        };
        DatasetStore::new(self.schema.clone()).commit(placements)?.with_window(window)
    }
}

/// `overlap(selected, start, size)`: one segment's local picks and where they land.
fn overlap(selected: &Indices, start: usize, size: usize) -> (Indices, usize) {
    match selected {
        Indices::Range(first, stop) => {
            let (low, high) = ((*first).max(start), (*stop).min(start + size));
            (Indices::Range(low - start, low.max(high) - start), low.saturating_sub(*first))
        }
        Indices::List(items) => {
            let left = items.partition_point(|item| *item < start);
            let right = items.partition_point(|item| *item < start + size);
            if left == right { return (Indices::Range(0, 0), left); }
            let local: Vec<usize> = items[left..right].iter().map(|item| item - start).collect();
            (Indices::from_sorted(local), left)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::ValueSchema;
    use crate::DType;

    fn axis(id: &str, role: &str, coordinates: Coordinates) -> Arc<Axis> {
        Axis::new(id.into(), id.into(), role.into(), coordinates, None, None, None, None).unwrap()
    }

    #[test]
    fn coordinate_interval_cuts_an_implicit_image_and_keeps_validity_compact() {
        let rows = axis("y", "spatial-y", Coordinates::Index { origin: 0, len: 4 });
        let columns = axis("x", "spatial-x", Coordinates::Index { origin: 0, len: 5 });
        let one = |id: &str| Domain::new(vec![1], vec![axis(id, if id == "r" { "repeat" } else { "scan-point" }, Coordinates::Index { origin: 0, len: 1 })],
            vec![CodeMapping::new(CodeBase::Range { start: 0, step: 1, len: 1 }, 1, 1).unwrap()]).unwrap();
        let schema = DatasetSchema::new(one("r"), one("p"), Domain::dense(vec![4, 5], vec![rows, columns]).unwrap(),
            ValueSchema::new(DType::F64, None, None, ValidityContract::Value).unwrap()).unwrap();
        let selection = Selection::new(vec![
            Term::coordinate_range("x".into(), &Coordinate::Real(0.5), &Coordinate::Real(3.0), None).unwrap(),
            Term::coordinate_range("y".into(), &Coordinate::Integer(1), &Coordinate::Real(2.5), None).unwrap(),
        ]).unwrap();
        let restriction = Restriction::new(&schema, Some(&selection)).unwrap();
        assert_eq!(restriction.indices().cells, vec![Indices::Range(1, 3), Indices::Range(1, 4)]);
        let values: Vec<f64> = (0..20).map(f64::from).collect();
        let block = DataBlock::new(schema.clone(), OwnedBuffer::from_vec(values).into(), Validity::Mask(OwnedBuffer::from_vec(vec![true])), None).unwrap();
        let cut = restriction.apply(&Source::Dense(block)).unwrap();
        let part = cut.segments().next().unwrap();
        let AnyBuffer::F64(values) = part.data().values() else { panic!() };
        assert_eq!(values.as_slice(), &[6.0, 7.0, 8.0, 11.0, 12.0, 13.0]);
        assert!(matches!(part.data().validity(), Validity::Uniform(true)));
        let empty = Selection::new(vec![Term::coordinate_range("x".into(), &Coordinate::Real(5.5), &Coordinate::Real(9.0), None).unwrap()]).unwrap();
        assert_eq!(Restriction::new(&schema, Some(&empty)).err().unwrap().kind, ErrorKind::EmptySelection);
    }
}
