//! Selector numbers (`selectors._selector_precision` / `_selector_number`)
//! and the portable series identity (`selectors.series_identity`).

use crate::configuration::error::{ErrorKind, PlotError, Result};
use crate::configuration::value::{repr_str, Value};
use crate::kinds::AxisRef;
use crate::pyfmt::format_g;

/// Decimal precision shared by native and raster selector labels: three
/// decimals finer than the span, never negative.  A span of NaN, infinity or
/// below the smallest normal step raises as `math.log10`/`math.ceil` do.
pub fn selector_precision(span: f64) -> Result<usize> {
    let gap = if span != 0.0 { span.abs() / 1000.0 } else { 0.01 };
    if gap == 0.0 { return Err(PlotError::value("math domain error")); }
    let exponent = gap.log10().ceil();
    if exponent.is_nan() { return Err(PlotError::value("cannot convert float NaN to integer")); }
    if exponent.is_infinite() { return Err(PlotError::new(ErrorKind::Overflow, "cannot convert float infinity to integer")); }
    Ok((-(exponent as i64)).max(0) as usize)
}

/// One selector coordinate at the shared visible-span precision: fixed
/// decimals up to six and below 1e9, `%.6g` otherwise.
pub fn selector_number(value: f64, precision: usize) -> String {
    if precision <= 6 && value.abs() < 1.0e9 { format!("{value:.precision$}") } else { format_g(value, 6) }
}

/// Portable group identity shared by interaction and scientific overlays:
/// `(domain, axis_id, repr(canonical))` of every coordinate value.
pub fn series_identity(group_key: &[(AxisRef, Value)]) -> Vec<(&'static str, String, String)> {
    group_key.iter().map(|(reference, canonical)| (reference.domain.value(), reference.axis_id.clone(), canonical.py_repr())).collect()
}

/// `repr(series_identity(group_key))`, the text the renderer digests.
pub fn series_identity_repr(group_key: &[(AxisRef, Value)]) -> String {
    let items: Vec<String> = series_identity(group_key).iter()
        .map(|(domain, axis_id, canonical)| format!("({}, {}, {})", repr_str(domain), repr_str(axis_id), repr_str(canonical)))
        .collect();
    match items.len() { 0 => "()".into(), 1 => format!("({},)", items[0]), _ => format!("({})", items.join(", ")) }
}
