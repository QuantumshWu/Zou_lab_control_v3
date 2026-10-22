//! The Python forms of the small numeric operations the interaction model is
//! made of, where Rust's own differ: `min`/`max` keep the FIRST argument
//! unless the other is strictly beyond it (so signed zeros and NaN land
//! exactly where CPython puts them), a two-item `sorted`, and the axis-scale
//! inverse that raises `OverflowError` where `float(10.0 ** x)` does.

use crate::configuration::error::{ErrorKind, PlotError, Result};
use crate::render::scale::AxisScale;

/// `min(a, b)`.
pub fn py_min(a: f64, b: f64) -> f64 { if b < a { b } else { a } }

/// `max(a, b)`.
pub fn py_max(a: f64, b: f64) -> f64 { if b > a { b } else { a } }

/// `tuple(sorted((a, b)))`.
pub fn sorted2(a: f64, b: f64) -> (f64, f64) { if b < a { (b, a) } else { (a, b) } }

/// `_axis_scale.axis_value(position, scale)` for one number: a log axis'
/// `float(10.0 ** position)` overflows with `OverflowError` from a finite
/// exponent, and a display-unit lattice raises it from inside its mapping.
pub fn axis_value(scale: &AxisScale, position: f64) -> Result<f64> {
    match scale {
        AxisScale::Log => {
            let value = 10f64.powf(position);
            if value.is_infinite() && position.is_finite() {
                return Err(PlotError::new(ErrorKind::Overflow, "(34, 'Result too large')"));
            }
            Ok(value)
        }
        AxisScale::Converted { mapping, canonical, display } => {
            let inner = axis_value(mapping, position)?;
            Ok(display.convert_value_to(inner, canonical).unwrap_or(f64::NAN))
        }
        other => Ok(other.value(position)),
    }
}

/// `_axis_scale.interpolate(low, high, fraction, scale)`.
pub fn interpolate(scale: &AxisScale, low: f64, high: f64, fraction: f64) -> Result<f64> {
    let (start, stop) = (scale.space(low), scale.space(high));
    axis_value(scale, start + fraction * (stop - start))
}

/// `_axis_scale.midpoint(low, high, scale)`.
pub fn midpoint(scale: &AxisScale, low: f64, high: f64) -> Result<f64> { interpolate(scale, low, high, 0.5) }

/// `_axis_scale.placed(value, limits, scale)`: an end the scale cannot place
/// lies at the low wall, `min(limits)`.
pub fn placed(scale: &AxisScale, value: f64, limits: (f64, f64)) -> f64 {
    if scale.space(value).is_finite() { value } else { py_min(limits.0, limits.1) }
}

/// `np.isclose(a, b, rtol=rtol, atol=atol)` of two numbers.
pub fn isclose(a: f64, b: f64, rtol: f64, atol: f64) -> bool {
    ((a - b).abs() <= atol + rtol * b.abs()) && b.is_finite() || a == b
}

/// `finite_real(value, field)`: the value itself, or `ValueError`.
pub fn finite(value: f64, field: &str) -> Result<f64> {
    if value.is_finite() { Ok(value) } else { Err(PlotError::value(format!("{field} must be finite"))) }
}

/// Python's float `a / b`: `ZeroDivisionError` for any zero divisor.
pub fn divide(a: f64, b: f64) -> Result<f64> {
    if b == 0.0 { return Err(PlotError::new(ErrorKind::ZeroDivision, "float division by zero")); }
    Ok(a / b)
}

/// `math.hypot(x, y)`: CPython's `vector_norm`, a correctly rounded
/// two-coordinate norm (it is not the C library's `hypot`).
pub fn hypot(x: f64, y: f64) -> f64 {
    let coordinates = [x.abs(), y.abs()];
    let found_nan = coordinates.iter().any(|v| v.is_nan());
    let mut max = 0.0f64;
    for &value in &coordinates { if value > max { max = value; } }
    vector_norm(coordinates, max, found_nan)
}

fn vector_norm(mut vec: [f64; 2], max: f64, found_nan: bool) -> f64 {
    if max.is_infinite() { return max; }
    if found_nan { return f64::NAN; }
    if max == 0.0 { return max; }
    let max_e = frexp_exponent(max);
    if max_e < -1023 {
        for value in vec.iter_mut() { *value /= f64::MIN_POSITIVE; }
        return f64::MIN_POSITIVE * vector_norm(vec, max / f64::MIN_POSITIVE, found_nan);
    }
    let scale = power_of_two(-max_e);
    let (mut csum, mut frac1, mut frac2) = (1.0f64, 0.0f64, 0.0f64);
    for value in vec {
        let x = value * scale;
        let (hi, lo) = dl_mul(x, x);
        let (sum, error) = dl_fast_sum(csum, hi);
        csum = sum;
        frac1 += lo;
        frac2 += error;
    }
    let mut h = (csum - 1.0 + (frac1 + frac2)).sqrt();
    let (hi, lo) = dl_mul(-h, h);
    let (sum, error) = dl_fast_sum(csum, hi);
    csum = sum;
    frac1 += lo;
    frac2 += error;
    let x = csum - 1.0 + (frac1 + frac2);
    h += x / (2.0 * h);
    h / scale
}

fn dl_mul(x: f64, y: f64) -> (f64, f64) {
    let z = x * y;
    (z, x.mul_add(y, -z))
}

fn dl_fast_sum(a: f64, b: f64) -> (f64, f64) {
    let x = a + b;
    let z = x - a;
    (x, b - z)
}

/// The `e` of C's `frexp(x)` for a positive finite `x`: `x = m * 2**e`, `0.5 <= m < 1`.
fn frexp_exponent(x: f64) -> i32 {
    let bits = x.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    if exponent != 0 { return exponent - 1022; }
    let mantissa = bits & ((1u64 << 52) - 1);
    64 - mantissa.leading_zeros() as i32 - 1074
}

/// `ldexp(1.0, e)`.
fn power_of_two(e: i32) -> f64 {
    if e >= -1022 { f64::from_bits(((e + 1023) as u64) << 52) } else { f64::from_bits(1u64 << (e + 1074)) }
}

/// Python's float `base ** exponent`: CPython's special cases, then the C
/// `pow`; `OverflowError` where a finite power overflows and
/// `ZeroDivisionError` for zero to a negative power.  A negative base to a
/// fractional power (a complex number in Python) is a `ValueError` here.
pub fn pow(base: f64, exponent: f64) -> Result<f64> {
    let odd = |w: f64| w.is_finite() && w == w.floor() && (w / 2.0).floor() != w / 2.0;
    if exponent == 0.0 { return Ok(1.0); }
    if base.is_nan() { return Ok(base); }
    if exponent.is_nan() { return Ok(if base == 1.0 { 1.0 } else { exponent }); }
    if exponent.is_infinite() {
        let magnitude = base.abs();
        if magnitude == 1.0 { return Ok(1.0); }
        return Ok(if (exponent > 0.0) == (magnitude > 1.0) { exponent.abs() } else { 0.0 });
    }
    if base.is_infinite() {
        return Ok(if exponent > 0.0 { if odd(exponent) { base } else { base.abs() } } else if odd(exponent) { 0.0f64.copysign(base) } else { 0.0 });
    }
    if base == 0.0 {
        if exponent < 0.0 { return Err(PlotError::new(ErrorKind::ZeroDivision, "0.0 cannot be raised to a negative power")); }
        return Ok(if odd(exponent) { base } else { 0.0 });
    }
    let (mut magnitude, mut negate) = (base, false);
    if base < 0.0 {
        if exponent != exponent.floor() { return Err(PlotError::value("negative number raised to a fractional power")); }
        magnitude = -base;
        negate = odd(exponent);
    }
    if magnitude == 1.0 { return Ok(if negate { -1.0 } else { 1.0 }); }
    let result = magnitude.powf(exponent);
    if result.is_infinite() { return Err(PlotError::new(ErrorKind::Overflow, "(34, 'Result too large')")); }
    Ok(if negate { -result } else { result })
}
