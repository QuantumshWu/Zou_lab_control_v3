//! How the editor words its numbers and counts: the one number formatter
//! (`zlc_data.units.format_quantity`, here `zlc_plot::units`), `counted`, and
//! the exact nanosecond arithmetic a total or a grid check needs.
use std::sync::{Arc, OnceLock};

use zlc_plot::units::{Number, Unit, format_quantity, resolve_unit};
use zlc_pulse::binding::convert_time;
use zlc_pulse::exact::Decimal;
use zlc_pulse::num::Num;
use zlc_pulse::units::nanoseconds_exponent;

/// `SUMMARY_SEPARATOR`.
pub const SEPARATOR: &str = " · ";

fn unit(symbol: &'static str) -> &'static Arc<Unit> {
    static ONE: OnceLock<Arc<Unit>> = OnceLock::new();
    static SECOND: OnceLock<Arc<Unit>> = OnceLock::new();
    let cell = if symbol == "1" { &ONE } else { &SECOND };
    cell.get_or_init(|| resolve_unit(symbol, None).expect("a registered unit"))
}

/// `format_quantity(float(value), "1")`: the number as it can be typed back.
pub fn number(value: f64) -> String {
    format_quantity(Number::Float(value), unit("1"), None).unwrap_or_else(|_| zlc_core::pyfmt::repr(value))
}

/// `format_quantity(seconds, "s")`.
pub fn seconds(value: f64) -> String {
    format_quantity(Number::Float(value), unit("s"), None).unwrap_or_else(|_| format!("{} s", zlc_core::pyfmt::repr(value)))
}

/// `counted(count, noun)`: "1 period", "3 periods".
pub fn counted(count: usize, noun: &str) -> String {
    if count == 1 { format!("1 {noun}") } else { format!("{count} {noun}s") }
}

/// `sentence(text)`: ending in a full stop.
pub fn sentence(text: &str) -> String {
    let text = text.trim_end();
    if text.is_empty() || text.ends_with(['.', '?', '!', '…']) { text.to_owned() } else { format!("{text}.") }
}

/// `_nanoseconds(value, unit)`: exact before it is a float.
pub fn nanoseconds(value: Num, unit: &str) -> f64 {
    Decimal::from_f64(value.to_f64()).scale10(nanoseconds_exponent(unit).unwrap_or(0)).to_f64()
}

/// `_readable(nanoseconds)`: one duration in the scale a person would write.
pub fn readable(nanoseconds: f64) -> String { seconds(Decimal::from_f64(nanoseconds).scale10(-9).to_f64()) }

/// `_bound_text(number, unit, binding_unit)`: a value a binding feeds, written
/// the way its field shows it.
pub fn bound_text(value: f64, unit: &str, binding_unit: &str) -> String {
    if unit == binding_unit && (unit == "value" || unit == "count") { return number(value); }
    match convert_time(value, unit, binding_unit) {
        Ok(converted) => format!("{} {binding_unit}", number(converted)),
        Err(_) => format!("{} {unit}", number(value)),
    }
}

/// `figure_padded(value, widest)`: left-padded with figure spaces to `widest`'s width.
pub fn figure_padded(value: &str, widest: &str) -> String { zlc_plot::units::figure_padded(value, widest) }
