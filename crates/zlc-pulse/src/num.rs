//! An authored number as Python held it: an `int` or a `float`, normalized the
//! way the model's `_number` normalizes (an integral float becomes an int).
use crate::error::{Result, type_error, value_error};
use crate::exact::Decimal;
use crate::json::Json;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Num { Int(i128), Float(f64) }

impl Num {
    /// `_number(value, field_name)` over a finite float or int.
    pub fn new(value: f64, field_name: &str) -> Result<Num> {
        if !value.is_finite() { return value_error(format!("{field_name} must be finite")); }
        if value.fract() == 0.0 {
            if value.abs() >= 1.7e38 { return value_error(format!("{field_name} is outside the supported range")); }
            return Ok(Num::Int(value as i128));
        }
        Ok(Num::Float(value))
    }

    /// `_number` of a JSON value: a bool or a non-number is not numeric.
    pub fn from_json(value: &Json, field_name: &str) -> Result<Num> {
        match value {
            Json::Int(int) => Ok(Num::Int(*int)),
            Json::Float(float) => Num::new(*float, field_name),
            _ => type_error(format!("{field_name} must be numeric")),
        }
    }

    pub fn to_json(self) -> Json { match self { Num::Int(int) => Json::Int(int), Num::Float(float) => Json::Float(float) } }

    pub fn to_f64(self) -> f64 { match self { Num::Int(int) => int as f64, Num::Float(float) => float } }

    /// `Fraction(str(value))`.
    pub fn decimal(self) -> Decimal {
        match self { Num::Int(int) => Decimal::from_i128(int), Num::Float(float) => Decimal::from_f64(float) }
    }

    /// `str(value)`.
    pub fn py_str(self) -> String {
        match self { Num::Int(int) => int.to_string(), Num::Float(float) => zlc_core::pyfmt::repr(float) }
    }

    pub fn is_int(self) -> bool { matches!(self, Num::Int(_)) }

    /// The value compared as Python compares an int with a float.
    pub fn same_value(self, other: f64) -> bool { self.to_f64() == other }
}

/// `_number_for(value)`: a float written back as an int when it is integral.
pub fn number_for(value: f64) -> Num {
    if value.fract() == 0.0 && value.abs() < 1.7e38 { Num::Int(value as i128) } else { Num::Float(value) }
}
