//! Immutable coordinate and carrier contracts; construction validates, Arc reuse does not.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::sync::Arc;

use crate::{AnyBuffer, DType, OwnedBuffer};

#[derive(Clone, Debug)]
pub enum Coordinate {
    /// Mixed scalar integers are exact through i128; larger inputs must be rejected, never rounded.
    Integer(i128),
    Real(f64),
    Text(Arc<str>),
    Missing,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
enum CoordinateKey {
    Integer(i128),
    Real(u64),
    Text(Arc<str>),
}

impl Coordinate {
    fn key(&self) -> Result<CoordinateKey, String> {
        match self {
            Self::Integer(value) => Ok(CoordinateKey::Integer(*value)),
            Self::Real(value) if value.is_finite() => {
                if value.fract() == 0.0 && *value >= i128::MIN as f64 && *value < -(i128::MIN as f64) {
                    Ok(CoordinateKey::Integer(*value as i128))
                } else {
                    Ok(CoordinateKey::Real(value.to_bits()))
                }
            }
            Self::Text(value) => Ok(CoordinateKey::Text(value.clone())),
            Self::Missing => Err("Axis coordinates cannot be missing".into()),
            Self::Real(_) => Err("Axis coordinates must be finite".into()),
        }
    }
}

impl PartialEq for Coordinate {
    fn eq(&self, other: &Self) -> bool {
        match (self.key(), other.key()) {
            (Ok(left), Ok(right)) => left == right,
            _ => matches!((self, other), (Self::Missing, Self::Missing)),
        }
    }
}

pub enum Coordinates {
    /// An absent coordinate vector denotes exact integer indices, not missing samples.
    Index { origin: i128, len: usize },
    Numeric(AnyBuffer),
    Values(Arc<[Coordinate]>),
    /// Record origins and shared sample offsets stay factored until a coordinate is read.
    OriginsAndOffsets { origins: OwnedBuffer<f64>, offsets: OwnedBuffer<f64> },
}

fn canonical(value: &str, field: &str) -> Result<(), String> {
    if value.is_empty() || value.trim() != value {
        return Err(format!("{field} must be canonical non-empty text"));
    }
    Ok(())
}

fn optional_text(value: &Option<Arc<str>>, field: &str) -> Result<(), String> {
    if let Some(value) = value { canonical(value, field)?; }
    Ok(())
}

fn product(shape: &[usize]) -> Result<usize, String> {
    if shape.is_empty() || shape.contains(&0) {
        return Err("Physical shape must contain positive dimensions".into());
    }
    shape.iter().try_fold(1usize, |n, size| n.checked_mul(*size).ok_or_else(|| "Physical shape overflows usize".into()))
}

fn unique<T: Copy, K: Eq + Hash>(values: &[T], compare: impl Fn(T, T) -> Ordering, key: impl Fn(T) -> K) -> Result<(), String> {
    let (mut increasing, mut decreasing) = (true, true);
    for pair in values.windows(2) {
        match compare(pair[0], pair[1]) {
            Ordering::Equal => return Err("Axis coordinates must be unique".into()),
            Ordering::Less => decreasing = false,
            Ordering::Greater => increasing = false,
        }
    }
    if !increasing && !decreasing {
        let mut seen = HashSet::with_capacity(values.len());
        if !values.iter().all(|value| seen.insert(key(*value))) {
            return Err("Axis coordinates must be unique".into());
        }
    }
    Ok(())
}

fn half_value(bits: u16) -> f64 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = (bits >> 10) & 31;
    let fraction = bits & 1023;
    match exponent {
        0 => sign * f64::from(fraction) * 2.0_f64.powi(-24),
        31 => if fraction == 0 { sign * f64::INFINITY } else { f64::NAN },
        _ => sign * f64::from(1024 + fraction) * 2.0_f64.powi(i32::from(exponent) - 25),
    }
}

fn validate_numeric(values: &AnyBuffer) -> Result<(), String> {
    macro_rules! integers { ($values:expr) => { unique($values.as_slice(), |a, b| a.cmp(&b), |value| value) }; }
    macro_rules! reals {
        ($values:expr) => {{
            let values = $values.as_slice();
            if !values.iter().all(|value| value.is_finite()) { return Err("Axis coordinates must be finite".into()); }
            unique(values, |a, b| a.partial_cmp(&b).unwrap(), |value| if value == 0.0 { 0 } else { value.to_bits() })
        }};
    }
    match values {
        AnyBuffer::U8(v) => integers!(v), AnyBuffer::I8(v) => integers!(v),
        AnyBuffer::U16(v) => integers!(v), AnyBuffer::I16(v) => integers!(v),
        AnyBuffer::U32(v) => integers!(v), AnyBuffer::I32(v) => integers!(v),
        AnyBuffer::U64(v) => integers!(v), AnyBuffer::I64(v) => integers!(v),
        AnyBuffer::F32(v) => reals!(v), AnyBuffer::F64(v) => reals!(v),
        AnyBuffer::F16(v) => {
            if !v.as_slice().iter().all(|value| half_value(value.to_bits()).is_finite()) {
                return Err("Axis coordinates must be finite".into());
            }
            unique(v.as_slice(), |a, b| half_value(a.to_bits()).partial_cmp(&half_value(b.to_bits())).unwrap(),
                |value| if value.to_bits() & 0x7fff == 0 { 0 } else { value.to_bits() })
        }
        AnyBuffer::Bool(_) | AnyBuffer::C64(_) | AnyBuffer::C128(_) => Err("Axis coordinates must be real numbers, not bool or complex".into()),
    }
}

impl Coordinates {
    pub fn len(&self) -> usize {
        match self {
            Self::Index { len, .. } => *len,
            Self::Numeric(values) => values.len(),
            Self::Values(values) => values.len(),
            Self::OriginsAndOffsets { origins, offsets } => origins.len().saturating_mul(offsets.len()),
        }
    }

    pub fn is_empty(&self) -> bool { self.len() == 0 }

    pub fn at(&self, index: usize) -> Option<Coordinate> {
        if index >= self.len() { return None; }
        Some(match self {
            Self::Index { origin, .. } => Coordinate::Integer(origin.checked_add(index as i128)?),
            Self::Values(values) => values[index].clone(),
            Self::OriginsAndOffsets { origins, offsets } => Coordinate::Real(origins.as_slice()[index / offsets.len()] + offsets.as_slice()[index % offsets.len()]),
            Self::Numeric(values) => match values {
                AnyBuffer::U8(v) => Coordinate::Integer(i128::from(v.as_slice()[index])),
                AnyBuffer::I8(v) => Coordinate::Integer(i128::from(v.as_slice()[index])),
                AnyBuffer::U16(v) => Coordinate::Integer(i128::from(v.as_slice()[index])),
                AnyBuffer::I16(v) => Coordinate::Integer(i128::from(v.as_slice()[index])),
                AnyBuffer::U32(v) => Coordinate::Integer(i128::from(v.as_slice()[index])),
                AnyBuffer::I32(v) => Coordinate::Integer(i128::from(v.as_slice()[index])),
                AnyBuffer::U64(v) => Coordinate::Integer(i128::from(v.as_slice()[index])),
                AnyBuffer::I64(v) => Coordinate::Integer(i128::from(v.as_slice()[index])),
                AnyBuffer::F16(v) => Coordinate::Real(half_value(v.as_slice()[index].to_bits())),
                AnyBuffer::F32(v) => Coordinate::Real(f64::from(v.as_slice()[index])),
                AnyBuffer::F64(v) => Coordinate::Real(v.as_slice()[index]),
                AnyBuffer::Bool(_) | AnyBuffer::C64(_) | AnyBuffer::C128(_) => return None,
            },
        })
    }

    fn validate(&self) -> Result<(), String> {
        if self.is_empty() { return Err("Axis coordinates must not be empty".into()); }
        match self {
            Self::Index { origin, len } => {
                origin.checked_add((*len - 1) as i128).ok_or("Implicit coordinate indices overflow i128")?;
                Ok(())
            }
            Self::Numeric(values) => validate_numeric(values),
            Self::Values(values) => {
                let mut seen = HashSet::with_capacity(values.len());
                for value in values.iter() {
                    if !seen.insert(value.key()?) { return Err("Axis coordinates must be unique".into()); }
                }
                Ok(())
            }
            Self::OriginsAndOffsets { origins, offsets } => {
                origins.len().checked_mul(offsets.len()).ok_or("Factored coordinate count overflows usize")?;
                let (origins, offsets) = (origins.as_slice(), offsets.as_slice());
                if !origins.iter().chain(offsets).all(|value| value.is_finite()) { return Err("Axis coordinates must be finite".into()); }
                let extent = origins.iter().map(|v| v.abs()).fold(0.0_f64, f64::max) + offsets.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
                let spacing = f64::from_bits(extent.to_bits() + 1) - extent;
                let ordered = offsets.windows(2).all(|pair| pair[1] - pair[0] > spacing);
                let separated = origins.windows(2).all(|pair| pair[1] + offsets[0] > pair[0] + offsets[offsets.len() - 1]);
                let finite = origins.iter().all(|origin| (origin + offsets[0]).is_finite() && (origin + offsets[offsets.len() - 1]).is_finite());
                if ordered && separated && finite { return Ok(()); }
                let mut seen = HashSet::new();
                for origin in origins {
                    for offset in offsets {
                        if !seen.insert(Coordinate::Real(origin + offset).key()?) { return Err("Axis coordinates must be unique".into()); }
                    }
                }
                Ok(())
            }
        }
    }

    fn same_values(&self, other: &Self) -> bool {
        if std::ptr::eq(self, other) { return true; }
        if self.len() != other.len() { return false; }
        match (self, other) {
            (Self::Index { origin: a, .. }, Self::Index { origin: b, .. }) => return a == b,
            (Self::Numeric(a), Self::Numeric(b)) if a.dtype() == b.dtype() && a.as_bytes() == b.as_bytes() => return true,
            (Self::OriginsAndOffsets { origins: a, offsets: b }, Self::OriginsAndOffsets { origins: c, offsets: d })
                if a.as_slice() == c.as_slice() && b.as_slice() == d.as_slice() => return true,
            _ => {}
        }
        (0..self.len()).all(|index| self.at(index) == other.at(index))
    }
}

pub struct Axis {
    id: Arc<str>,
    name: Arc<str>,
    role: Arc<str>,
    coordinates: Coordinates,
    unit: Option<Arc<str>>,
    coordinate_frame: Option<Arc<str>>,
    labels: Option<Arc<[Arc<str>]>>,
    coordinate_of: Option<Arc<str>>,
}

impl Axis {
    #[allow(clippy::too_many_arguments)]
    pub fn new(id: Arc<str>, name: Arc<str>, role: Arc<str>, coordinates: Coordinates,
        unit: Option<Arc<str>>, coordinate_frame: Option<Arc<str>>, labels: Option<Vec<Arc<str>>>,
        coordinate_of: Option<Arc<str>>) -> Result<Arc<Self>, String> {
        for (text, field) in [(&id, "Axis id"), (&name, "Axis name"), (&role, "Axis role")] { canonical(text, field)?; }
        optional_text(&unit, "Axis unit")?; optional_text(&coordinate_frame, "Coordinate frame")?;
        optional_text(&coordinate_of, "Alternative coordinate primary")?;
        if coordinate_of.as_ref() == Some(&id) { return Err("An alternative coordinate cannot name itself".into()); }
        coordinates.validate()?;
        if let Some(labels) = &labels {
            if labels.len() != coordinates.len() { return Err("Coordinate labels must match axis size".into()); }
            for label in labels { canonical(label, "Coordinate label")?; }
        }
        Ok(Arc::new(Self { id, name, role, coordinates, unit, coordinate_frame,
            labels: labels.map(Arc::from), coordinate_of }))
    }

    pub fn id(&self) -> &str { &self.id }
    pub fn name(&self) -> &str { &self.name }
    pub fn role(&self) -> &str { &self.role }
    pub fn len(&self) -> usize { self.coordinates.len() }
    pub fn is_empty(&self) -> bool { false }
    pub fn coordinates(&self) -> &Coordinates { &self.coordinates }
    pub fn coordinate(&self, index: usize) -> Option<Coordinate> { self.coordinates.at(index) }
    pub fn unit(&self) -> Option<&str> { self.unit.as_deref() }
    pub fn coordinate_frame(&self) -> Option<&str> { self.coordinate_frame.as_deref() }
    pub fn labels(&self) -> Option<&[Arc<str>]> { self.labels.as_deref() }
    pub fn coordinate_of(&self) -> Option<&str> { self.coordinate_of.as_deref() }
}

impl PartialEq for Axis {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other) || (self.id == other.id && self.name == other.name
            && self.role == other.role && self.unit == other.unit
            && self.coordinate_frame == other.coordinate_frame && self.labels == other.labels
            && self.coordinate_of == other.coordinate_of && self.coordinates.same_values(&other.coordinates))
    }
}
impl Eq for Axis {}

pub enum CodeBase {
    Range { start: u64, step: i64, len: usize },
    Explicit(OwnedBuffer<u64>),
}

pub struct CodeMapping {
    base: CodeBase,
    inner: usize,
    outer: usize,
    base_len: usize,
    carrier_len: usize,
    max_code: u64,
}

impl CodeMapping {
    pub fn new(mut base: CodeBase, inner: usize, outer: usize) -> Result<Arc<Self>, String> {
        if inner == 0 || outer == 0 { return Err("Axis code repeat factors must be positive".into()); }
        let (base_len, max_code) = match &mut base {
            CodeBase::Range { start, step, len } => {
                if *len == 0 || *step == 0 { return Err("Code ranges must be nonempty with nonzero step".into()); }
                if *len == 1 { *step = 1; }
                let last = i128::from(*step).checked_mul((*len - 1) as i128).and_then(|v| v.checked_add(i128::from(*start))).ok_or("Code range overflows")?;
                if last < 0 || last > i128::from(i64::MAX) || *start > i64::MAX as u64 { return Err("Axis code lies outside nonnegative int64 indices".into()); }
                (*len, (*start).max(last as u64))
            }
            CodeBase::Explicit(values) => {
                let max = values.as_slice().iter().copied().max().ok_or("Axis code vector must not be empty")?;
                if max > i64::MAX as u64 { return Err("Axis code lies outside nonnegative int64 indices".into()); }
                (values.len(), max)
            }
        };
        let carrier_len = base_len.checked_mul(inner).and_then(|n| n.checked_mul(outer)).ok_or("Axis mapping length overflows usize")?;
        Ok(Arc::new(Self { base, inner, outer, base_len, carrier_len, max_code }))
    }

    pub fn base(&self) -> &CodeBase { &self.base }
    pub fn base_len(&self) -> usize { self.base_len }
    pub fn inner(&self) -> usize { self.inner }
    pub fn outer(&self) -> usize { self.outer }
    pub fn carrier_len(&self) -> usize { self.carrier_len }
    pub fn code(&self, flat_index: usize) -> Option<usize> {
        if flat_index >= self.carrier_len { return None; }
        let position = (flat_index / self.inner) % self.base_len;
        Some(match &self.base {
            CodeBase::Range { start, step, .. } => (i128::from(*start) + position as i128 * i128::from(*step)) as usize,
            CodeBase::Explicit(values) => values.as_slice()[position] as usize,
        })
    }

    fn same_rows(&self, other: &Self) -> bool {
        if std::ptr::eq(self, other) { return true; }
        if self.carrier_len != other.carrier_len { return false; }
        if self.inner == other.inner && self.outer == other.outer && self.base_len == other.base_len {
            match (&self.base, &other.base) {
                (CodeBase::Range { start: a, step: b, .. }, CodeBase::Range { start: c, step: d, .. }) => return a == c && b == d,
                (CodeBase::Explicit(a), CodeBase::Explicit(b)) => return a.as_slice() == b.as_slice(),
                _ => return (0..self.base_len).all(|position| self.code(position * self.inner) == other.code(position * other.inner)),
            }
        }
        (0..self.carrier_len).all(|row| self.code(row) == other.code(row))
    }
}

pub struct Domain {
    physical_shape: Box<[usize]>,
    size: usize,
    axes: Vec<Arc<Axis>>,
    mappings: Vec<Arc<CodeMapping>>,
    by_id: HashMap<Arc<str>, usize>,
}

impl Domain {
    pub fn new(physical_shape: Vec<usize>, axes: Vec<Arc<Axis>>, mut mappings: Vec<Arc<CodeMapping>>) -> Result<Arc<Self>, String> {
        let size = product(&physical_shape)?;
        if axes.len() != mappings.len() { return Err("Every logical axis needs one code mapping".into()); }
        if axes.is_empty() && size != 1 { return Err("A carrier with multiple elements needs a named axis".into()); }
        let mut by_id = HashMap::with_capacity(axes.len());
        for (index, (axis, mapping)) in axes.iter().zip(&mappings).enumerate() {
            if by_id.insert(axis.id.clone(), index).is_some() { return Err("Domain axis ids must be unique".into()); }
            if mapping.carrier_len != size || mapping.max_code as u128 >= axis.len() as u128 {
                return Err("Axis mapping must cover this carrier with valid coordinate codes".into());
            }
        }
        for (index, axis) in axes.iter().enumerate() {
            if let Some(primary) = axis.coordinate_of() {
                let position = *by_id.get(primary).ok_or("Alternative coordinate primary must belong to the same domain")?;
                if axes[position].coordinate_of().is_some() || axis.len() != axes[position].len() {
                    return Err("Alternative coordinates need one same-sized primary axis".into());
                }
                if !Arc::ptr_eq(&mappings[index], &mappings[position]) && !mappings[index].same_rows(&mappings[position]) {
                    return Err("Alternative coordinates must name the same physical elements".into());
                }
                mappings[index] = mappings[position].clone();
            }
        }
        Ok(Arc::new(Self { physical_shape: physical_shape.into_boxed_slice(), size, axes, mappings, by_id }))
    }

    /// Dense tensor axes use the same range/inner/outer formula as mapped flat carriers.
    pub fn dense(physical_shape: Vec<usize>, axes: Vec<Arc<Axis>>) -> Result<Arc<Self>, String> {
        let size = product(&physical_shape)?;
        let primary: Vec<_> = axes.iter().filter(|axis| axis.coordinate_of().is_none()).collect();
        if primary.len() != physical_shape.len() || primary.iter().zip(&physical_shape).any(|(axis, length)| axis.len() != *length) {
            return Err("Dense shape needs one same-sized primary axis per dimension".into());
        }
        let (mut inner, mut outer) = (size, 1usize);
        let mut mapped = HashMap::new();
        for axis in primary {
            inner /= axis.len();
            mapped.insert(axis.id.clone(), CodeMapping::new(CodeBase::Range { start: 0, step: 1, len: axis.len() }, inner, outer)?);
            outer *= axis.len();
        }
        let mappings = axes.iter().map(|axis| mapped.get(axis.coordinate_of().unwrap_or(axis.id())).cloned().ok_or_else(|| "Alternative coordinate primary is absent".to_string())).collect::<Result<Vec<_>, _>>()?;
        Self::new(physical_shape, axes, mappings)
    }

    pub fn physical_shape(&self) -> &[usize] { &self.physical_shape }
    pub fn size(&self) -> usize { self.size }
    pub fn axes(&self) -> &[Arc<Axis>] { &self.axes }
    pub fn axis(&self, id: &str) -> Option<&Arc<Axis>> { self.by_id.get(id).map(|index| &self.axes[*index]) }
    pub fn mapping(&self, id: &str) -> Option<&Arc<CodeMapping>> { self.by_id.get(id).map(|index| &self.mappings[*index]) }

    /// Count distinct coordinates among present carrier rows, fixing every
    /// other primary axis at the current row. Alternative coordinates move
    /// with their primary; scientific validity is deliberately not an input.
    pub fn coordinate_counts(&self, current: usize, rows: impl IntoIterator<Item = usize>) -> Vec<usize> {
        let mut counts: Vec<HashSet<usize>> = self.axes.iter().map(|_| HashSet::new()).collect();
        for row in rows {
            for (target, axis) in self.axes.iter().enumerate() {
                let primary = axis.coordinate_of().unwrap_or(axis.id());
                if self.axes.iter().enumerate().all(|(index, other)|
                    other.coordinate_of().unwrap_or(other.id()) == primary || self.mappings[index].code(row) == self.mappings[index].code(current)) {
                    if let Some(code) = self.mappings[target].code(row) { counts[target].insert(code); }
                }
            }
        }
        counts.into_iter().map(|codes| codes.len()).collect()
    }

    pub fn same_coordinate(&self, left: usize, right: usize) -> bool {
        self.axes.iter().zip(&self.mappings).filter(|(axis, _)| axis.coordinate_of().is_none())
            .all(|(_, mapping)| mapping.code(left) == mapping.code(right))
    }

    /// Candidate rows come from one factored mapping, so regular scans never
    /// scan their entire Point carrier on each acquired point.
    pub fn coordinate_rows(&self, current: usize) -> Vec<usize> {
        let Some(mapping) = self.mappings.iter().max_by_key(|mapping| mapping.base_len) else { return vec![0] };
        let code = mapping.code(current).expect("current row is in the carrier");
        let positions: Vec<usize> = match &mapping.base {
            CodeBase::Range { .. } => vec![(current / mapping.inner) % mapping.base_len],
            CodeBase::Explicit(values) => values.as_slice().iter().enumerate().filter_map(|(index, value)| (*value as usize == code).then_some(index)).collect(),
        };
        (0..mapping.outer).flat_map(|outer| positions.iter().flat_map(move |position|
            (0..mapping.inner).map(move |inner| (outer * mapping.base_len + position) * mapping.inner + inner)))
            .filter(|row| self.same_coordinate(*row, current)).collect()
    }
}

impl PartialEq for Domain {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other) || (self.physical_shape == other.physical_shape
            && self.axes == other.axes && self.mappings.iter().zip(&other.mappings).all(|(left, right)| left.same_rows(right)))
    }
}
impl Eq for Domain {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DomainRole { Repeat, Point, Cell }

#[derive(Clone, Eq, PartialEq)]
pub enum ValidityContract { Value, Components(Vec<Arc<str>>) }

#[derive(Clone, Eq, PartialEq)]
pub struct ValueSchema {
    dtype: DType,
    unit: Option<Arc<str>>,
    name: Option<Arc<str>>,
    validity: ValidityContract,
}

impl ValueSchema {
    pub fn new(dtype: DType, unit: Option<Arc<str>>, name: Option<Arc<str>>, validity: ValidityContract) -> Result<Self, String> {
        optional_text(&unit, "Value unit")?; optional_text(&name, "Value name")?;
        if let ValidityContract::Components(ids) = &validity {
            if ids.is_empty() { return Err("Component validity requires named axes".into()); }
            let mut seen = HashSet::new();
            for id in ids { canonical(id, "Validity component axis")?; if !seen.insert(id) { return Err("Validity axes must be unique".into()); } }
        }
        Ok(Self { dtype, unit, name, validity })
    }
    pub fn dtype(&self) -> DType { self.dtype }
    pub fn unit(&self) -> Option<&str> { self.unit.as_deref() }
    pub fn name(&self) -> Option<&str> { self.name.as_deref() }
    pub fn validity(&self) -> &ValidityContract { &self.validity }
}

pub struct DatasetSchema {
    domains: [Arc<Domain>; 3],
    value: ValueSchema,
    physical_shape: Box<[usize]>,
    value_len: usize,
    component_mask_len: usize,
}

impl DatasetSchema {
    pub fn new(repeat: Arc<Domain>, point: Arc<Domain>, cell: Arc<Domain>, value: ValueSchema) -> Result<Arc<Self>, String> {
        let mut ids = HashSet::new();
        for domain in [&repeat, &point, &cell] {
            for axis in domain.axes() { if !ids.insert(axis.id()) { return Err("Dataset axis ids must be unique across all domains".into()); } }
        }
        let mut mask_len = repeat.size.checked_mul(point.size).ok_or("Validity size overflows usize")?;
        if let ValidityContract::Components(components) = &value.validity {
            let (mut previous, mut primary) = (None, HashSet::new());
            for id in components {
                let index = *cell.by_id.get(id).ok_or("Validity component axes must belong to the Cell domain")?;
                if previous.is_some_and(|prior| prior >= index) { return Err("Validity axes must follow Cell-domain axis order".into()); }
                let axis = &cell.axes[index];
                if !primary.insert(axis.coordinate_of().unwrap_or(axis.id())) { return Err("Validity axes cannot repeat alternative coordinates of the same primary".into()); }
                mask_len = mask_len.checked_mul(axis.len()).ok_or("Validity size overflows usize")?;
                previous = Some(index);
            }
        }
        let physical_shape = [&repeat, &point, &cell].into_iter().flat_map(|domain| domain.physical_shape.iter().copied()).collect::<Vec<_>>();
        let value_len = product(&physical_shape)?;
        Ok(Arc::new(Self { domains: [repeat, point, cell], value, physical_shape: physical_shape.into_boxed_slice(), value_len, component_mask_len: mask_len }))
    }
    pub fn dtype(&self) -> DType { self.value.dtype }
    pub fn value_len(&self) -> usize { self.value_len }
    pub fn domain(&self, role: DomainRole) -> &Arc<Domain> { &self.domains[match role { DomainRole::Repeat => 0, DomainRole::Point => 1, DomainRole::Cell => 2 }] }
    pub fn value(&self) -> &ValueSchema { &self.value }
    pub fn component_mask_len(&self) -> usize { self.component_mask_len }
    pub fn physical_shape(&self) -> &[usize] { &self.physical_shape }
}

impl PartialEq for DatasetSchema {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other) || (self.value == other.value && self.domains == other.domains)
    }
}
impl Eq for DatasetSchema {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_mapping_contract_covers_dense_flat_aliases_and_compact_validity() {
        let axis = |id: &str, len| Axis::new(id.into(), id.into(), "generic".into(), Coordinates::Index { origin: 0, len }, None, None, None, None).unwrap();
        let y = axis("y", 2); let x = axis("x", 3);
        let x_alt = Axis::new("frequency".into(), "Frequency".into(), "coordinate".into(),
            Coordinates::Numeric(OwnedBuffer::from_vec(vec![9_007_199_254_740_993_u64, 9_007_199_254_740_995, 9_007_199_254_740_997]).into()),
            Some("Hz".into()), None, Some(vec!["a".into(), "b".into(), "c".into()]), Some("x".into())).unwrap();
        let dense = Domain::dense(vec![2, 3], vec![y.clone(), x.clone(), x_alt.clone()]).unwrap();
        let rows = CodeMapping::new(CodeBase::Explicit(OwnedBuffer::from_vec(vec![0, 0, 0, 1, 1, 1])), 1, 1).unwrap();
        let columns = CodeMapping::new(CodeBase::Range { start: 0, step: 1, len: 3 }, 1, 2).unwrap();
        let flat = Domain::new(vec![6], vec![y, x, x_alt.clone()], vec![rows, columns.clone(), columns]).unwrap();
        for row in 0..6 { for id in ["y", "x", "frequency"] { assert_eq!(dense.mapping(id).unwrap().code(row), flat.mapping(id).unwrap().code(row)); } }
        let equivalent = Domain::new(vec![2, 3], flat.axes().to_vec(), ["y", "x", "frequency"].iter().map(|id| flat.mapping(id).unwrap().clone()).collect()).unwrap();
        assert!(dense == equivalent);
        assert!(Arc::ptr_eq(dense.mapping("x").unwrap(), dense.mapping("frequency").unwrap()));
        assert_eq!(x_alt.coordinate(0), Some(Coordinate::Integer(9_007_199_254_740_993)));
        assert!(matches!(x_alt.coordinates(), Coordinates::Numeric(AnyBuffer::U64(_))));
        let image = Domain::dense(vec![1200, 1920], vec![axis("pixel-row", 1200), axis("pixel-column", 1920)]).unwrap();
        assert_eq!(image.size(), 2_304_000);
        assert_eq!(image.mapping("pixel-row").unwrap().base_len(), 1200);
        assert_eq!(image.mapping("pixel-column").unwrap().base_len(), 1920);
        assert_eq!(image.mapping("pixel-row").unwrap().code(1921), Some(1));
        assert_eq!(image.mapping("pixel-column").unwrap().code(1921), Some(1));
        let scalar = Domain::new(vec![1], vec![], vec![]).unwrap();
        let value = ValueSchema::new(DType::F64, Some("count".into()), Some("signal".into()), ValidityContract::Components(vec!["y".into()])).unwrap();
        let schema = DatasetSchema::new(scalar.clone(), scalar, dense, value).unwrap();
        assert_eq!(schema.physical_shape(), &[1, 1, 2, 3]);
        assert_eq!(schema.value_len(), 6); assert_eq!(schema.component_mask_len(), 2);
        let reader = schema.clone(); assert!(Arc::ptr_eq(&schema, &reader));
        let equivalent = DatasetSchema::new(schema.domain(DomainRole::Repeat).clone(), schema.domain(DomainRole::Point).clone(), equivalent,
            ValueSchema::new(DType::F64, Some("count".into()), Some("signal".into()), ValidityContract::Components(vec!["y".into()])).unwrap()).unwrap();
        assert!(!Arc::ptr_eq(&schema, &equivalent)); assert!(schema == equivalent);
    }
}
