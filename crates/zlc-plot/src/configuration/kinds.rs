//! The shared plot vocabulary (`crate::kinds`) as the configuration layer
//! reads it off the wire, and the canonical coordinate scalar
//! (`zlc_data.canonical_coordinate_scalar`).

pub use zlc_core::schema::Coordinate;

pub use crate::kinds::{AxisDomain, AxisRef, PlotKind, Reduction, RelimMode};

use super::error::{PlotError, Result};
use super::value::{py_strip, repr_str, Value};

/// Every domain, in schema order.
pub const DOMAINS: [AxisDomain; 3] = [AxisDomain::Repeat, AxisDomain::Point, AxisDomain::CellData];

/// `PlotKind(value)`: an exact value, never a guess.
pub fn parse_kind(value: &str) -> Result<PlotKind> {
    PlotKind::parse(value).ok_or_else(|| PlotError::value(format!("{} is not a valid PlotKind", repr_str(value))))
}

/// `PlotKind(str(value))` of a wire value.
pub fn kind_from_value(value: &Value) -> Result<PlotKind> { parse_kind(&value.py_str()) }

pub fn parse_reduction(value: &str) -> Result<Reduction> {
    Reduction::parse(value).ok_or_else(|| PlotError::value(format!("{} is not a valid Reduction", repr_str(value))))
}

/// `Reduction(str(value))`.
pub fn reduction_from_value(value: &Value) -> Result<Reduction> { parse_reduction(&value.py_str()) }

pub fn parse_domain(value: &str) -> Result<AxisDomain> {
    AxisDomain::parse(value).ok_or_else(|| PlotError::value(format!("{} is not a valid AxisDomain", repr_str(value))))
}

/// `AxisDomain(value)` of any wire value: only the exact text names one.
pub fn domain_from_value(value: &Value) -> Result<AxisDomain> {
    match value {
        Value::Text(text) => parse_domain(text),
        other => Err(PlotError::value(format!("{} is not a valid AxisDomain", other.py_repr()))),
    }
}

pub fn domain_role(domain: AxisDomain) -> zlc_core::schema::DomainRole {
    match domain {
        AxisDomain::Repeat => zlc_core::schema::DomainRole::Repeat,
        AxisDomain::Point => zlc_core::schema::DomainRole::Point,
        AxisDomain::CellData => zlc_core::schema::DomainRole::Cell,
    }
}

/// `AxisRef(domain, axis_id)`: stripped as Python strips, never empty.
pub fn axis_ref(domain: AxisDomain, axis_id: &str) -> Result<AxisRef> {
    AxisRef::new(domain, py_strip(axis_id)).map_err(PlotError::value)
}

/// `AxisRef(domain, value)` of a wire value; a non-text id is a ValueError.
pub fn axis_ref_from_value(domain: AxisDomain, axis_id: &Value) -> Result<AxisRef> {
    match axis_id {
        Value::Text(text) => axis_ref(domain, text),
        _ => Err(PlotError::value(format!("{} requires a non-empty axis_id", domain.value()))),
    }
}

/// A reference to an axis a schema declares (its id is already canonical).
pub fn declared_ref(domain: AxisDomain, axis_id: &str) -> AxisRef { AxisRef { domain, axis_id: axis_id.to_owned() } }

/// Python `repr(AxisRef)`, as error messages print it.
pub fn axis_repr(axis: &AxisRef) -> String {
    format!("AxisRef(domain=<AxisDomain.{}: {}>, axis_id={})", match axis.domain {
        AxisDomain::Repeat => "REPEAT", AxisDomain::Point => "POINT", AxisDomain::CellData => "CELL_DATA" },
        repr_str(axis.domain.value()), repr_str(&axis.axis_id))
}

/// `canonical_coordinate_scalar`: null, text, an integer, or a finite float
/// (an integral float is the integer it names).
pub fn canonical_scalar(value: &Value, field: &str) -> Result<Coordinate> {
    match value {
        Value::None => Ok(Coordinate::Missing),
        Value::Text(text) => Ok(Coordinate::Text(text.clone())),
        Value::Int(integer) => Ok(Coordinate::Integer(*integer)),
        Value::Float(float) => canonical_float(*float, field),
        Value::Bool(_) => Err(PlotError::type_error(format!("{field} must be null, text, or a finite number"))),
        _ => Err(PlotError::type_error(format!("{field} must be a real number"))),
    }
}

/// One finite float as its canonical coordinate.
pub fn canonical_float(value: f64, field: &str) -> Result<Coordinate> {
    if !value.is_finite() { return Err(PlotError::value(format!("{field} must be finite"))); }
    // Python makes every integral float its int; past i128 it stays a float here.
    if value.fract() == 0.0 && value.abs() < 2f64.powi(127) { return Ok(Coordinate::Integer(value as i128)); }
    Ok(Coordinate::Real(value))
}

/// A coordinate as the record value Python holds.
pub fn coordinate_value(coordinate: &Coordinate) -> Value {
    match coordinate {
        Coordinate::Integer(value) => Value::Int(*value),
        Coordinate::Real(value) => Value::Float(*value),
        Coordinate::Text(text) => Value::Text(text.clone()),
        Coordinate::Missing => Value::None,
    }
}

/// A coordinate as the canonical scalar (integral reals become integers).
pub fn canonical_coordinate(coordinate: &Coordinate) -> Coordinate {
    match coordinate {
        Coordinate::Real(value) => canonical_float(*value, "coordinate").unwrap_or(Coordinate::Real(*value)),
        other => other.clone(),
    }
}
