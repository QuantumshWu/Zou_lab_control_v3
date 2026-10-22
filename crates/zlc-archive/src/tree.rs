//! The dataset schema as `zlc_data.codec` writes it into a manifest, read
//! back through the refusals `AxisSpec`, `DomainSpec`, `ValueSchema` and
//! `DatasetSchema` raise for the same tree.

use std::sync::Arc;

use zlc_core::restrict::{Held, Vector};
use zlc_core::schema::{Axis, CodeBase, CodeMapping, Coordinate, Coordinates, DatasetSchema, Domain, DomainRole, ValidityContract, ValueSchema};
use zlc_core::{DType, OwnedBuffer};

use crate::error::{ArchiveError, Result};
use crate::json::{py_repr, py_repr_names, py_repr_str, Json};
use crate::npy::NpyType;

pub const AXIS_SCHEMA: &str = "zlc_data.AxisSpec";
pub const DOMAIN_SCHEMA: &str = "zlc_data.DomainSpec";
pub const VALUE_SCHEMA: &str = "zlc_data.ValueSchema";
pub const DATASET_SCHEMA: &str = "zlc_data.DatasetSchema";

// ------------------------------------------------------------ validators

/// `str.strip()` would change it (Python's whitespace includes \x1c-\x1f).
pub(crate) fn stripped(text: &str) -> bool {
    let space = |c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c);
    text.starts_with(space) || text.ends_with(space)
}

/// `canonical_text(value, field)`.
pub fn canonical_text<'a>(value: &'a Json, field: &str) -> Result<&'a str> {
    match value { Json::Str(text) if !text.is_empty() && !stripped(text) => Ok(text), _ => Err(ArchiveError::value(format!("{field} must be canonical non-empty text"))) }
}

/// `integer(value, field, minimum=...)`.
pub fn integer(value: &Json, field: &str, minimum: Option<i128>) -> Result<i128> {
    let integer = match value {
        Json::Int(integer) => *integer,
        Json::BigInt(text) => return if text.starts_with('-') && minimum.is_some() { Err(ArchiveError::value(format!("{field} must be at least {}", minimum.unwrap()))) }
            else { Err(ArchiveError::new(crate::ErrorKind::Overflow, format!("{field} is too large"))) },
        _ => return Err(ArchiveError::type_error(format!("{field} must be an integer"))),
    };
    if let Some(minimum) = minimum { if integer < minimum { return Err(ArchiveError::value(format!("{field} must be at least {minimum}"))); } }
    Ok(integer)
}

/// `exact_mapping(value, fields, format_name, discriminator=...)`.
pub fn exact_mapping<'a>(value: &'a Json, fields: &[&str], format_name: &str, discriminator: Option<&str>) -> Result<&'a [(String, Json)]> {
    let pairs = match value.as_object() {
        Some(pairs) if pairs.len() == fields.len() && fields.iter().all(|field| value.get(field).is_some()) => pairs,
        _ => return Err(ArchiveError::value(format!("{format_name} must contain exactly {}", py_repr_names(fields)))),
    };
    if let Some(key) = discriminator {
        let found = value.get(key).unwrap();
        if found.as_str() != Some(format_name) {
            return Err(ArchiveError::value(format!("expected {key} {}, got {}", py_repr_str(format_name), py_repr(found))));
        }
    }
    Ok(pairs)
}

fn field<'a>(value: &'a Json, name: &str) -> &'a Json { value.get(name).expect("exact mapping") }

fn non_canonical(kind: &str) -> ArchiveError { ArchiveError::value(format!("{kind} tree is typed but non-canonical")) }

// ------------------------------------------------------------ coordinates

fn scalar_of(value: &Json) -> Result<Coordinate> {
    Ok(match value {
        Json::Null => Coordinate::Missing,
        Json::Str(text) => Coordinate::Text(text.as_str().into()),
        Json::Int(integer) => Coordinate::Integer(*integer),
        Json::Float(real) => Coordinate::Real(*real),
        Json::BigInt(_) => return Err(ArchiveError::value("coordinates beyond 128-bit integers are not supported natively")),
        Json::Bool(_) => return Err(ArchiveError::type_error("coordinate must be null, text, or a finite number")),
        _ => return Err(ArchiveError::type_error("coordinate must be a real number")),
    })
}

/// `_read_coordinates(values, field)`: canonical scalars only.
fn read_coordinates(values: Option<&Json>, field: &str) -> Result<Option<Vec<Coordinate>>> {
    let Some(values) = values.filter(|values| !values.is_null()) else { return Ok(None) };
    let Json::Array(items) = values else { return Err(ArchiveError::value(format!("AxisSpec {field} must be a list or null"))) };
    items.iter().map(|item| {
        let scalar = scalar_of(item)?;
        // An integral or negative-zero float has another canonical spelling.
        if let Coordinate::Real(real) = scalar { if real.fract() == 0.0 { return Err(non_canonical("AxisSpec")); } }
        Ok(scalar)
    }).collect::<Result<Vec<_>>>().map(Some)
}

fn json_of_real(real: f64) -> Json {
    if real.fract() != 0.0 { return Json::Float(real); }
    if real.abs() < 1.7e38 { Json::Int(real as i128) } else { Json::BigInt(format!("{real:.0}")) }
}

fn json_of_scalar(value: &Coordinate) -> Json {
    match value { Coordinate::Integer(integer) => Json::Int(*integer), Coordinate::Real(real) => json_of_real(*real),
                  Coordinate::Text(text) => Json::str(text.as_ref()), Coordinate::Missing => Json::Null }
}

fn unique(sorted_ok: bool, count_unique: usize, count: usize) -> Result<()> {
    if !sorted_ok && count_unique != count { return Err(ArchiveError::value("axis coordinates must be unique")); }
    Ok(())
}

/// `_unique_coordinates` of a normalized vector.
fn check_unique(vector: &Vector) -> Result<()> {
    fn numeric<T: PartialOrd + Copy>(values: &[T], key: impl Fn(T) -> u64) -> Result<()> {
        let monotonic = values.len() < 2 || values.windows(2).all(|pair| pair[1] > pair[0]) || values.windows(2).all(|pair| pair[1] < pair[0]);
        let mut keys: Vec<u64> = values.iter().map(|value| key(*value)).collect();
        keys.sort_unstable();
        keys.dedup();
        unique(monotonic, keys.len(), values.len())
    }
    match vector {
        Vector::I64(values) => numeric(values, |value| value as u64),
        Vector::U64(values) => numeric(values, |value| value),
        Vector::F64(values) => numeric(values, |value: f64| if value == 0.0 { 0 } else { value.to_bits() }),
        Vector::Values(values) => {
            if values.iter().any(|value| matches!(value, Coordinate::Missing)) { return Err(ArchiveError::value("axis coordinates cannot be missing")); }
            for (index, value) in values.iter().enumerate() {
                if values[..index].iter().any(|earlier| earlier == value) { return Err(ArchiveError::value("axis coordinates must be unique")); }
            }
            Ok(())
        }
    }
}

// ------------------------------------------------------------ AxisSpec

fn axis_tree(axis: &Axis) -> Json {
    let held = Held::of(axis);
    let reals = |values: &[f64]| Json::Array(values.iter().map(|value| json_of_real(*value)).collect());
    let (coordinates, origin, origins) = match &held {
        Held::Implicit { origin } => (Json::Null, *origin, None),
        Held::I64(values) => (Json::Array(values.iter().map(|value| Json::Int(i128::from(*value))).collect()), 0, None),
        Held::U64(values) => (Json::Array(values.iter().map(|value| Json::Int(i128::from(*value))).collect()), 0, None),
        Held::F64(values) => (reals(values), 0, None),
        Held::Values(values) => (Json::Array(values.iter().map(json_of_scalar).collect()), 0, None),
        Held::Factored { origins, offsets } => (reals(offsets), 0, Some(reals(origins))),
    };
    let mut pairs = vec![
        ("schema", Json::str(AXIS_SCHEMA)), ("axis_id", Json::str(axis.id())), ("name", Json::str(axis.name())), ("role", Json::str(axis.role())),
        ("size", Json::Int(axis.len() as i128)), ("coordinates", coordinates), ("unit", axis.unit().map_or(Json::Null, Json::str)),
        ("coordinate_frame", axis.coordinate_frame().map_or(Json::Null, Json::str)), ("index_origin", Json::Int(origin)),
        ("coordinate_labels", axis.labels().map_or(Json::Null, |labels| Json::Array(labels.iter().map(|label| Json::str(label.as_ref())).collect()))),
    ];
    if let Some(primary) = axis.coordinate_of() { pairs.push(("coordinate_of", Json::str(primary))); }
    if let Some(origins) = origins { pairs.push(("coordinate_origins", origins)); }
    Json::object(pairs)
}

fn axis_from_tree(tree: &Json) -> Result<Arc<Axis>> {
    let mut fields = vec!["schema", "axis_id", "name", "role", "size", "coordinates", "unit", "coordinate_frame", "index_origin", "coordinate_labels"];
    let has = |key: &str| tree.as_object().is_some() && tree.get(key).is_some();
    if has("coordinate_of") { fields.push("coordinate_of"); }
    if has("coordinate_origins") { fields.push("coordinate_origins"); }
    exact_mapping(tree, &fields, AXIS_SCHEMA, Some("schema"))?;
    let coordinates = read_coordinates(Some(field(tree, "coordinates")), "coordinates")?;
    let origins = read_coordinates(tree.get("coordinate_origins"), "coordinate_origins")?;
    if tree.get("coordinate_of").is_some_and(Json::is_null) || (tree.get("coordinate_origins").is_some() && origins.is_none()) {
        return Err(non_canonical("AxisSpec"));
    }
    let labels = field(tree, "coordinate_labels");
    if !labels.is_null() && labels.as_array().is_none() { return Err(ArchiveError::value("AxisSpec coordinate_labels must be a list or null")); }
    // The keyword arguments, in order, then `AxisSpec.__post_init__`.
    let axis_id = canonical_text(field(tree, "axis_id"), "AxisId")?;
    let role = canonical_text(field(tree, "role"), "AxisRoleId")?;
    let frame = match field(tree, "coordinate_frame") { Json::Null => None, value => Some(canonical_text(value, "CoordinateFrameId")?) };
    let primary = match tree.get("coordinate_of") { None | Some(Json::Null) => None, Some(value) => Some(canonical_text(value, "AxisId")?) };
    if primary == Some(axis_id) { return Err(ArchiveError::value("an alternative coordinate cannot name itself")); }
    let name = canonical_text(field(tree, "name"), "axis name")?;
    let size = integer(field(tree, "size"), "axis size", Some(1))? as usize;
    let native = match coordinates {
        None => {
            if origins.is_some() { return Err(ArchiveError::value("coordinate origins require explicit sample offsets")); }
            None
        }
        Some(values) => {
            let vector = Vector::of_scalars(values);
            match origins {
                None => {
                    let count = vector_len(&vector);
                    if count != size { return Err(ArchiveError::value(format!("axis coordinates length {count} does not match size {size}"))); }
                    check_unique(&vector)?;
                    Some(vector.into_coordinates())
                }
                Some(origins) => {
                    let offsets = match &vector {
                        _ if role != "sample-time" => return Err(ArchiveError::value("coordinate origins require numeric sample-time offsets")),
                        Vector::I64(values) => values.iter().map(|value| *value as f64).collect::<Vec<_>>(),
                        Vector::U64(values) => values.iter().map(|value| *value as f64).collect(),
                        Vector::F64(values) => values.clone(),
                        Vector::Values(_) => return Err(ArchiveError::value("coordinate origins require numeric sample-time offsets")),
                    };
                    let origins = origins.iter().map(|value| match value {
                        Coordinate::Integer(integer) => Ok(*integer as f64), Coordinate::Real(real) => Ok(*real), Coordinate::Missing => Ok(f64::NAN),
                        Coordinate::Text(text) => Err(ArchiveError::value(format!("could not convert string to float: {}", py_repr_str(text)))),
                    }).collect::<Result<Vec<f64>>>()?;
                    if origins.is_empty() || offsets.is_empty() { return Err(ArchiveError::value("coordinate origins and offsets must be nonempty vectors")); }
                    if !origins.iter().all(|value| value.is_finite()) { return Err(ArchiveError::value("coordinate origins must be finite")); }
                    let count = offsets.len() * origins.len();
                    if count != size { return Err(ArchiveError::value(format!("axis coordinates length {count} does not match size {size}"))); }
                    let first: Vec<f64> = origins.iter().map(|origin| origin + offsets[0]).collect();
                    let last: Vec<f64> = origins.iter().map(|origin| origin + offsets[offsets.len() - 1]).collect();
                    let ordered = offsets.windows(2).all(|pair| pair[1] > pair[0]);
                    let separated = (1..origins.len()).all(|index| first[index] > last[index - 1]);
                    let extent = origins.iter().map(|v| v.abs()).fold(f64::NEG_INFINITY, f64::max) + offsets.iter().map(|v| v.abs()).fold(f64::NEG_INFINITY, f64::max);
                    let spacing = f64::from_bits(extent.abs().to_bits() + 1) - extent.abs();
                    let distinct = offsets.len() == 1 || offsets.windows(2).map(|pair| pair[1] - pair[0]).fold(f64::INFINITY, f64::min) > spacing;
                    if !(ordered && separated && distinct && first.iter().chain(&last).all(|value| value.is_finite())) {
                        let actual: Vec<f64> = origins.iter().flat_map(|origin| offsets.iter().map(move |offset| origin + offset)).collect();
                        if !actual.iter().all(|value| value.is_finite()) { return Err(ArchiveError::value("axis coordinate must be finite")); }
                        check_unique(&Vector::F64(actual))?;
                    }
                    Some(Coordinates::OriginsAndOffsets { origins: OwnedBuffer::from_vec(origins), offsets: OwnedBuffer::from_vec(offsets) })
                }
            }
        }
    };
    let unit = match field(tree, "unit") { Json::Null => None, value => Some(canonical_text(value, "axis unit")?) };
    let origin = integer(field(tree, "index_origin"), "index_origin", Some(0))?;
    if native.is_some() && origin != 0 { return Err(ArchiveError::value("index_origin is only valid for an implicit-coordinate axis")); }
    let labels = match labels {
        Json::Array(items) => {
            let labels = items.iter().map(|item| canonical_text(item, "axis coordinate label").map(Arc::<str>::from)).collect::<Result<Vec<_>>>()?;
            if labels.len() != size { return Err(ArchiveError::value("axis coordinate_labels length must match axis size")); }
            Some(labels)
        }
        _ => None,
    };
    let coordinates = native.unwrap_or(Coordinates::Index { origin, len: size });
    Axis::new(axis_id.into(), name.into(), role.into(), coordinates, unit.map(Arc::from), frame.map(Arc::from), labels, primary.map(Arc::from))
        .map_err(ArchiveError::value)
}

fn vector_len(vector: &Vector) -> usize {
    match vector { Vector::I64(v) => v.len(), Vector::U64(v) => v.len(), Vector::F64(v) => v.len(), Vector::Values(v) => v.len() }
}

// ------------------------------------------------------------ DomainSpec

fn domain_tree(domain: &Domain, mapped: bool) -> Json {
    let axes = Json::Array(domain.axes().iter().map(|axis| axis_tree(axis)).collect());
    let shape = Json::Array(domain.physical_shape().iter().map(|size| Json::Int(*size as i128)).collect());
    let mut pairs = vec![("schema", Json::str(DOMAIN_SCHEMA)), ("shape", shape), ("axes", axes)];
    if !mapped {
        pairs.push(("axis_codes", Json::Null));
        return Json::object(pairs);
    }
    let mappings: Vec<&Arc<CodeMapping>> = domain.axes().iter().map(|axis| domain.mapping(axis.id()).expect("mapped axis")).collect();
    pairs.push(("axis_codes", Json::Array(mappings.iter().map(|mapping| match mapping.base() {
        CodeBase::Range { start, step, len } => {
            let start = i128::from(*start);
            Json::object([("range", Json::Array(vec![Json::Int(start), Json::Int(start + i128::from(*step) * *len as i128), Json::Int(i128::from(*step))]))])
        }
        CodeBase::Explicit(codes) => Json::Array(codes.as_slice().iter().map(|code| Json::Int(i128::from(*code))).collect()),
    }).collect())));
    if mappings.iter().any(|mapping| (mapping.inner(), mapping.outer()) != (1, 1)) {
        pairs.push(("axis_code_repeats", Json::Array(mappings.iter().map(|mapping| Json::Array(vec![Json::Int(mapping.inner() as i128), Json::Int(mapping.outer() as i128)])).collect())));
    }
    Json::object(pairs)
}

/// A code vector as `DomainSpec` normalizes it.
enum Codes { Range { start: i128, stop: i128, step: i128 }, Explicit(Vec<u64>) }

impl Codes {
    fn len(&self) -> usize {
        match self {
            Codes::Range { start, stop, step } => {
                let span = if *step > 0 { stop - start } else { start - stop };
                if span <= 0 { 0 } else { ((span - 1) / step.abs() + 1) as usize }
            }
            Codes::Explicit(codes) => codes.len(),
        }
    }
}

/// `domain_from_tree(tree)`: the domain, and whether it was explicitly mapped.
fn domain_from_tree(tree: &Json) -> Result<(Arc<Domain>, bool)> {
    let mut fields = vec!["schema", "shape", "axes", "axis_codes"];
    let has_repeats = tree.as_object().is_some() && tree.get("axis_code_repeats").is_some();
    if has_repeats { fields.push("axis_code_repeats"); }
    exact_mapping(tree, &fields, DOMAIN_SCHEMA, Some("schema"))?;
    let Json::Array(shape) = field(tree, "shape") else { return Err(ArchiveError::value("DomainSpec shape must be a list")) };
    let Json::Array(axes) = field(tree, "axes") else { return Err(ArchiveError::value("DomainSpec axes must be a list")) };
    let codes = field(tree, "axis_codes");
    if !codes.is_null() && codes.as_array().is_none() { return Err(ArchiveError::value("DomainSpec axis_codes must be a list or null")); }
    // Authored code vectors: a JSON list as written, or a checked range.
    enum Authored<'a> { List(&'a [Json]), Range(i128, i128, i128) }
    let mut authored = Vec::new();
    if let Json::Array(items) = codes {
        for item in items {
            match item {
                Json::Array(values) => authored.push(Authored::List(values)),
                _ => {
                    exact_mapping(item, &["range"], "DomainSpec axis range", None)?;
                    let Json::Array(parts) = field(item, "range") else { return Err(ArchiveError::value("axis code range needs start, stop and step")) };
                    if parts.len() != 3 { return Err(ArchiveError::value("axis code range needs start, stop and step")); }
                    let values = parts.iter().map(|part| integer(part, "axis code range", None)).collect::<Result<Vec<_>>>()?;
                    if values[2] == 0 { return Err(ArchiveError::value("range() arg 3 must not be zero")); }
                    authored.push(Authored::Range(values[0], values[1], values[2]));
                }
            }
        }
    }
    let repeats = tree.get("axis_code_repeats");
    if has_repeats && !matches!(repeats, Some(Json::Array(pairs)) if pairs.iter().all(|pair| pair.as_array().is_some())) {
        return Err(ArchiveError::value("axis_code_repeats must be a list of pairs"));
    }
    let axes = axes.iter().map(axis_from_tree).collect::<Result<Vec<_>>>()?;
    // `DomainSpec.__post_init__`.
    let shape = shape.iter().map(|size| integer(size, "domain dimension size", Some(1)).map(|size| size as usize)).collect::<Result<Vec<_>>>()?;
    if shape.is_empty() { return Err(ArchiveError::value("domain shape must contain at least one dimension")); }
    for (index, axis) in axes.iter().enumerate() {
        if axes[..index].iter().any(|earlier| earlier.id() == axis.id()) { return Err(ArchiveError::value("domain axis ids must be unique")); }
    }
    let mapped = !codes.is_null();
    for axis in &axes {
        let Some(primary) = axis.coordinate_of() else { continue };
        match axes.iter().find(|item| item.id() == primary) {
            Some(primary) if primary.coordinate_of().is_none() && primary.len() == axis.len() => {}
            _ => return Err(ArchiveError::value("alternative coordinates need one same-sized primary axis in their domain")),
        }
        if !mapped { return Err(ArchiveError::value("alternative coordinates require an explicitly mapped domain")); }
    }
    if !mapped {
        if repeats.is_some_and(|value| !value.is_null()) { return Err(ArchiveError::value("code repeats require an explicitly mapped domain")); }
        if axes.len() != shape.len() || axes.iter().zip(&shape).any(|(axis, size)| axis.len() != *size) {
            return Err(ArchiveError::value("a dense domain needs one same-sized axis per physical dimension"));
        }
        return Domain::dense(shape, axes).map(|domain| (domain, false)).map_err(ArchiveError::value);
    }
    if shape.len() != 1 { return Err(ArchiveError::value("an explicitly mapped domain must have one physical dimension")); }
    if authored.len() != axes.len() { return Err(ArchiveError::value("axis_codes must contain one vector per axis")); }
    let carrier = shape[0];
    let raw_pairs: Vec<&[Json]> = match repeats { Some(Json::Array(pairs)) => pairs.iter().map(|pair| pair.as_array().unwrap()).collect(), _ => Vec::new() };
    if repeats.is_some() && raw_pairs.len() != axes.len() { return Err(ArchiveError::value("axis_code_repeats must contain one pair per axis")); }
    let mut factors = Vec::with_capacity(axes.len());
    for index in 0..axes.len() {
        if raw_pairs.is_empty() { factors.push((1usize, 1usize)); continue; }
        let pair = raw_pairs[index];
        if pair.len() != 2 { return Err(ArchiveError::value("each axis code repeat needs inner and outer counts")); }
        let inner = integer(&pair[0], "axis code repeat", Some(1))? as usize;
        let outer = integer(&pair[1], "axis code repeat", Some(1))? as usize;
        factors.push((inner, outer));
    }
    let outside = || ArchiveError::value("axis code is outside its coordinate domain");
    let mismatch = || ArchiveError::value("each axis code vector must match domain size");
    let mut normalized = Vec::with_capacity(axes.len());
    for ((axis, entries), (inner, outer)) in axes.iter().zip(&authored).zip(&factors) {
        normalized.push(match entries {
            Authored::Range(start, stop, step) => {
                let codes = Codes::Range { start: *start, stop: *stop, step: *step };
                let count = codes.len();
                if count * inner * outer != carrier { return Err(mismatch()); }
                let last = start + (count as i128 - 1) * step;
                let (low, high) = ((*start).min(last), (*start).max(last));
                if low < 0 || high >= axis.len() as i128 || high > i128::from(i64::MAX) { return Err(outside()); }
                let step = if count > 1 { *step } else { 1 };
                Codes::Range { start: *start, stop: start + step * count as i128, step }
            }
            Authored::List(values) => {
                if values.iter().any(|value| value.as_array().is_some()) || values.len() * inner * outer != carrier { return Err(mismatch()); }
                let integers = values.iter().map(|value| match value { Json::Int(integer) => Some(*integer), _ => None }).collect::<Option<Vec<_>>>()
                    .ok_or_else(|| ArchiveError::type_error("axis codes must be integers"))?;
                if integers.iter().any(|code| *code < 0 || *code >= axis.len() as i128 || *code > i128::from(i64::MAX)) { return Err(outside()); }
                Codes::Explicit(integers.into_iter().map(|code| code as u64).collect())
            }
        });
    }
    let row_codes = |codes: &Codes, (inner, outer): (usize, usize)| -> Vec<i128> {
        let base: Vec<i128> = match codes {
            Codes::Range { start, step, .. } => (0..codes.len() as i128).map(|position| start + position * step).collect(),
            Codes::Explicit(values) => values.iter().map(|value| i128::from(*value)).collect(),
        };
        let repeated: Vec<i128> = base.iter().flat_map(|code| std::iter::repeat_n(*code, inner)).collect();
        (0..outer).flat_map(|_| repeated.iter().copied()).collect()
    };
    let mut final_factors = factors.clone();
    let mut mappings: Vec<Option<Arc<CodeMapping>>> = vec![None; axes.len()];
    for (index, axis) in axes.iter().enumerate() {
        if let Some(primary) = axis.coordinate_of() {
            let primary = axes.iter().position(|item| item.id() == primary).unwrap();
            // Python compares two ranges as sequences, anything else elementwise.
            let (x, y) = (&normalized[index], &normalized[primary]);
            let same_base = match (x, y) {
                (Codes::Range { start: a, step: c, .. }, Codes::Range { start: d, step: f, .. }) =>
                    x.len() == y.len() && (x.len() == 0 || (a == d && (x.len() == 1 || c == f))),
                _ => row_codes(x, (1, 1)) == row_codes(y, (1, 1)),
            };
            if factors[index] != factors[primary] || !same_base {
                if row_codes(&normalized[index], factors[index]) != row_codes(&normalized[primary], factors[primary]) {
                    return Err(ArchiveError::value("alternative coordinates must name the same physical rows"));
                }
            }
            final_factors[index] = factors[primary];
        }
    }
    if axes.is_empty() && carrier != 1 { return Err(ArchiveError::value("a mapped domain with multiple rows needs a named axis")); }
    // `domain_from_tree`'s canonical checks against what DomainSpec kept.
    let primary_of = |index: usize| axes[index].coordinate_of().map_or(index, |primary| axes.iter().position(|item| item.id() == primary).unwrap());
    let kept_repeats = final_factors.iter().any(|pair| *pair != (1, 1));
    if has_repeats && !kept_repeats { return Err(non_canonical("DomainSpec")); }
    if let Some(Json::Array(pairs)) = repeats {
        let same = pairs.iter().zip(&final_factors).all(|(raw, (inner, outer))| *raw == Json::Array(vec![Json::Int(*inner as i128), Json::Int(*outer as i128)]));
        if !same { return Err(non_canonical("DomainSpec")); }
    }
    for (index, raw) in authored.iter().enumerate() {
        let kept = &normalized[primary_of(index)];
        let canonical = match (raw, kept) {
            (Authored::Range(a, b, c), Codes::Range { start, stop, step }) => (a, b, c) == (start, stop, step),
            (Authored::Range(..), _) | (Authored::List(_), Codes::Range { .. }) => false,
            (Authored::List(_), _) => true,
        };
        if !canonical { return Err(non_canonical("DomainSpec")); }
    }
    for index in 0..axes.len() {
        let primary = primary_of(index);
        if mappings[primary].is_none() {
            let (inner, outer) = final_factors[primary];
            let base = match &normalized[primary] {
                Codes::Range { start, step, .. } => CodeBase::Range { start: *start as u64, step: *step as i64, len: normalized[primary].len() },
                Codes::Explicit(values) => CodeBase::Explicit(OwnedBuffer::from_vec(values.clone())),
            };
            mappings[primary] = Some(CodeMapping::new(base, inner, outer).map_err(ArchiveError::value)?);
        }
        mappings[index] = mappings[primary].clone();
    }
    Domain::new(shape, axes, mappings.into_iter().map(|mapping| mapping.unwrap()).collect()).map(|domain| (domain, true)).map_err(ArchiveError::value)
}

// ------------------------------------------------------------ ValueSchema

/// `str(np.dtype(...))`: numpy's name of a dtype.
pub fn dtype_name(dtype: &NpyType) -> String {
    use crate::npdtype::Kind;
    match &dtype.kind {
        _ if dtype.subarray.is_some() || dtype.has_fields() => dtype.str(),
        Kind::Numeric(kind) if !dtype.swapped => match kind {
            DType::Bool => "bool", DType::U8 => "uint8", DType::I8 => "int8", DType::U16 => "uint16", DType::I16 => "int16",
            DType::U32 => "uint32", DType::I32 => "int32", DType::U64 => "uint64", DType::I64 => "int64", DType::F16 => "float16",
            DType::F32 => "float32", DType::F64 => "float64", DType::C64 => "complex64", DType::C128 => "complex128",
        }.into(),
        Kind::DateTime { code, unit } if !dtype.swapped => format!("{}{unit}", if *code == 'M' { "datetime64" } else { "timedelta64" }),
        Kind::Object if dtype.itemsize == 8 => "object".into(),
        Kind::Object => "StringDType()".into(),
        _ => dtype.str(),
    }
}

fn value_tree(schema: &DatasetSchema) -> Json {
    let value = schema.value();
    let (mode, ids) = match value.validity() {
        ValidityContract::Value => ("VALUE", Vec::new()),
        ValidityContract::Components(ids) => ("COMPONENTS", ids.iter().map(|id| Json::str(id.as_ref())).collect()),
    };
    let mut pairs = vec![
        ("schema", Json::str(VALUE_SCHEMA)),
        ("validity_contract", Json::object([("mode", Json::str(mode)), ("component_axis_ids", Json::Array(ids))])),
        ("dtype", Json::Str(NpyType::native(value.dtype()).str())),
        ("value_unit", value.unit().map_or(Json::Null, Json::str)),
    ];
    if let Some(name) = value.name() { pairs.push(("name", Json::str(name))); }
    Json::object(pairs)
}

fn value_from_tree(tree: &Json) -> Result<ValueSchema> {
    let mut fields = vec!["schema", "validity_contract", "dtype", "value_unit"];
    if tree.as_object().is_some() && tree.get("name").is_some() { fields.push("name"); }
    exact_mapping(tree, &fields, VALUE_SCHEMA, Some("schema"))?;
    let validity = field(tree, "validity_contract");
    if !matches!(validity.as_object(), Some(pairs) if pairs.len() == 2 && validity.get("mode").is_some() && validity.get("component_axis_ids").is_some()) {
        return Err(ArchiveError::value("invalid ValueSchema validity_contract"));
    }
    let mode = match validity.get("mode").unwrap() {
        Json::Str(text) if text == "VALUE" || text == "COMPONENTS" => text.clone(),
        other => return Err(ArchiveError::value(format!("{} is not a valid ValidityMode", py_repr(other)))),
    };
    let Json::Array(ids) = validity.get("component_axis_ids").unwrap() else { return Err(ArchiveError::value("component_axis_ids must be a list")) };
    let ids = ids.iter().map(|id| canonical_text(id, "AxisId").map(Arc::<str>::from)).collect::<Result<Vec<_>>>()?;
    if (1..ids.len()).any(|index| ids[..index].contains(&ids[index])) { return Err(ArchiveError::value("component_axis_ids must be unique")); }
    if mode == "VALUE" && !ids.is_empty() { return Err(ArchiveError::value("VALUE validity cannot declare component axes")); }
    if mode == "COMPONENTS" && ids.is_empty() { return Err(ArchiveError::value("COMPONENTS validity requires at least one component axis")); }
    let unit = field(tree, "value_unit");
    if tree.get("name").is_some_and(Json::is_null) { return Err(non_canonical("ValueSchema")); }
    let text = canonical_text(field(tree, "dtype"), "dtype")?;
    let dtype = crate::npdtype::from_str(text)?;
    let crate::npdtype::Kind::Numeric(kind) = dtype.kind else {
        return Err(ArchiveError::type_error(format!("data values require bool or numeric dtype, got {}", dtype_name(&dtype))));
    };
    if dtype.subarray.is_some() { return Err(ArchiveError::type_error(format!("data values require bool or numeric dtype, got {}", dtype_name(&dtype)))); }
    let unit = match unit { Json::Null => None, value => Some(canonical_text(value, "value_unit")?) };
    let name = match tree.get("name") { None => None, Some(value) => Some(canonical_text(value, "value name")?) };
    if NpyType::native(kind).str() != text { return Err(non_canonical("ValueSchema")); }
    let contract = if mode == "VALUE" { ValidityContract::Value } else { ValidityContract::Components(ids) };
    ValueSchema::new(kind, unit.map(Arc::from), name.map(Arc::from), contract).map_err(ArchiveError::value)
}

// ------------------------------------------------------------ DatasetSchema

/// `dataset_schema_to_tree(schema)`.
pub fn schema_tree(schema: &DatasetSchema) -> Json {
    Json::object([
        ("schema", Json::str(DATASET_SCHEMA)),
        ("repeat_domain", domain_tree(schema.domain(DomainRole::Repeat), true)),
        ("point_domain", domain_tree(schema.domain(DomainRole::Point), true)),
        ("cell_domain", domain_tree(schema.domain(DomainRole::Cell), false)),
        ("value_schema", value_tree(schema)),
    ])
}

/// `dataset_schema_from_tree(tree)`.
pub fn schema_from_tree(tree: &Json) -> Result<Arc<DatasetSchema>> {
    exact_mapping(tree, &["schema", "repeat_domain", "point_domain", "cell_domain", "value_schema"], DATASET_SCHEMA, Some("schema"))?;
    let (repeat, repeat_mapped) = domain_from_tree(field(tree, "repeat_domain"))?;
    let (point, point_mapped) = domain_from_tree(field(tree, "point_domain"))?;
    let (cell, cell_mapped) = domain_from_tree(field(tree, "cell_domain"))?;
    let value = value_from_tree(field(tree, "value_schema"))?;
    if !repeat_mapped || repeat.physical_shape().len() != 1 { return Err(ArchiveError::value("repeat_domain must be an explicitly mapped flat domain")); }
    if repeat.axes().iter().any(|axis| axis.role() != "repeat") { return Err(ArchiveError::value("repeat-domain axes must carry the repeat role")); }
    if !point_mapped || point.physical_shape().len() != 1 { return Err(ArchiveError::value("point_domain must be an explicitly mapped flat domain")); }
    if point.axes().iter().any(|axis| axis.role() == "repeat") { return Err(ArchiveError::value("repeat-role axes belong only to the Repeat domain")); }
    if cell_mapped { return Err(ArchiveError::value("cell_domain must use dense implicit axis mapping")); }
    if cell.axes().iter().any(|axis| axis.role() == "repeat") { return Err(ArchiveError::value("REPEAT role belongs only to the Repeat domain")); }
    let scalar = cell.axes().iter().any(|axis| axis.role() == "scalar");
    if scalar && !scalar_domain(&cell) { return Err(ArchiveError::value("the canonical scalar domain must be the whole Cell domain")); }
    if scalar && *value.validity() != ValidityContract::Value { return Err(ArchiveError::value("the scalar carrier uses value-level validity")); }
    if let ValidityContract::Components(ids) = value.validity() {
        let positions: Option<Vec<usize>> = ids.iter().map(|id| cell.axes().iter().position(|axis| axis.id() == id.as_ref())).collect();
        if !positions.is_some_and(|positions| positions.windows(2).all(|pair| pair[0] <= pair[1])) {
            return Err(ArchiveError::value("validity component axes must be an ordered subset of Cell axes"));
        }
    }
    let ids: Vec<&str> = [&repeat, &point, &cell].iter().flat_map(|domain| domain.axes().iter().map(|axis| axis.id())).collect();
    if (1..ids.len()).any(|index| ids[..index].contains(&ids[index])) { return Err(ArchiveError::value("dataset axis ids must be unique across all domains")); }
    DatasetSchema::new(repeat, point, cell, value).map_err(ArchiveError::value)
}

/// `cell_domain == SCALAR_DOMAIN`: the one `zlc_data.scalar` axis.
fn scalar_domain(cell: &Domain) -> bool {
    let [axis] = cell.axes() else { return false };
    cell.physical_shape() == [1] && axis.id() == "zlc_data.scalar" && axis.name() == "value" && axis.role() == "scalar" && axis.len() == 1
        && axis.unit().is_none() && axis.coordinate_frame().is_none() && axis.labels().is_none() && axis.coordinate_of().is_none()
        && matches!(Held::of(axis), Held::I64(values) if values == [0])
}
