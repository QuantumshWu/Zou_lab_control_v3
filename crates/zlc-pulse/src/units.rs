//! The time units a pulse is authored in: the second's engineering ladder from
//! nanoseconds to seconds, nothing finer (the board's tick is tens of
//! nanoseconds) and nothing coarser. A tick is not a unit: it is however long
//! the board's clock says.
//!
//! Spellings follow the product's unit grammar: `us`, `µs` (U+00B5) and `μs`
//! (U+03BC) are one unit, stored as `µs`. The editor crate checks this table
//! against `zlc_plot::units` so the two cannot drift.
use crate::error::{Result, repr, type_error, value_error};
use crate::exact::Decimal;

/// Authorable time units, finest first, in their one stored spelling.
pub const TIME_UNIT_CHOICES: [&str; 4] = ["ns", "µs", "ms", "s"];

/// Decimal exponent of each unit relative to the second.
fn decade(symbol: &str) -> Option<i64> {
    match symbol { "ns" => Some(-9), "µs" => Some(-6), "ms" => Some(-3), "s" => Some(0), _ => None }
}

/// The stored spelling of a spelling the grammar accepts, or None for an unknown unit.
pub fn resolve_time_unit(text: &str) -> Option<&'static str> {
    match text {
        "ns" => Some("ns"),
        "us" | "µs" | "μs" => Some("µs"),
        "ms" => Some("ms"),
        "s" => Some("s"),
        _ => None,
    }
}

/// `canonical_time_unit(value, field_name)`.
pub fn canonical_time_unit(text: &str, field_name: &str) -> Result<&'static str> {
    if text.trim().is_empty() { return type_error(format!("{field_name} must be non-empty text")); }
    match resolve_time_unit(text) {
        Some(symbol) => Ok(symbol),
        None if known_non_time_unit(text) => value_error(format!(
            "{field_name} must be one of ('ns', 'µs', 'ms', 's'), not {}", repr(text))),
        None => value_error(format!("{field_name} is unsupported: {}", repr(text))),
    }
}

/// Units the product grammar knows that are not authorable times (a refusal
/// then says which units are allowed rather than that the unit is unknown).
fn known_non_time_unit(text: &str) -> bool {
    matches!(text, "1" | "V" | "mV" | "A" | "mA" | "Hz" | "kHz" | "MHz" | "GHz" | "m" | "mm" | "um" | "µm" | "nm"
        | "G" | "mG" | "T" | "W" | "mW" | "dBm" | "rad" | "deg" | "%" | "K" | "mK" | "uK" | "µK")
}

/// `nanoseconds_per(unit)`: how many nanoseconds one of `unit` is, as a power of ten.
pub fn nanoseconds_exponent(unit: &str) -> Result<i64> {
    let symbol = canonical_time_unit(unit, "time unit")?;
    Ok(decade(symbol).expect("a canonical unit") + 9)
}

/// `Fraction(str(number)) * nanoseconds_per(unit)`: the exact duration in nanoseconds.
pub fn exact_nanoseconds(value: &Decimal, unit: &str) -> Result<Decimal> {
    Ok(value.scale10(nanoseconds_exponent(unit)?))
}
