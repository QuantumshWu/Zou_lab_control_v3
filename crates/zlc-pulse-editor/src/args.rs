//! What an edit's arguments mean, read the way the v2 presenter read them:
//! a field sends what was typed (text), a notebook or a command sends numbers,
//! and both are the same question (`float(value)`, `str(value)`, `bool(value)`).
use serde_json::Value;
use zlc_pulse::error::repr;
use zlc_pulse::{Error, Kind, Result};

/// `str(value)`.
pub fn py_str(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Null => "None".into(),
        Value::Bool(flag) => if *flag { "True".into() } else { "False".into() },
        Value::Number(number) => match number.as_i64().map(|v| v as i128).or_else(|| number.as_u64().map(|v| v as i128)) {
            Some(int) => int.to_string(),
            None => zlc_core::pyfmt::repr(number.as_f64().unwrap_or(f64::NAN)),
        },
        other => other.to_string(),
    }
}

fn python_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "NoneType", Value::Bool(_) => "bool", Value::Number(n) if n.is_f64() => "float", Value::Number(_) => "int",
        Value::String(_) => "str", Value::Array(_) => "list", Value::Object(_) => "dict",
    }
}

/// `bool(value)`.
pub fn py_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Number(number) => number.as_f64().is_some_and(|v| v != 0.0),
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(items) => !items.is_empty(),
    }
}

/// Python's `float(text)` grammar: an optionally signed decimal, `inf`,
/// `infinity` or `nan`, with `_` allowed between digits and surrounding white space.
fn parse_float_text(text: &str) -> Option<f64> {
    let body = text.trim();
    let (sign, rest) = match body.as_bytes().first() {
        Some(b'+') => (1.0, &body[1..]),
        Some(b'-') => (-1.0, &body[1..]),
        _ => (1.0, body),
    };
    let lower = rest.to_ascii_lowercase();
    if lower == "inf" || lower == "infinity" { return Some(sign * f64::INFINITY); }
    if lower == "nan" { return Some(f64::NAN); }
    let bytes = rest.as_bytes();
    let mut cleaned = String::with_capacity(rest.len());
    let mut digits = 0;
    let mut at = 0;
    let mut previous_digit = false;
    let mut seen_point = false;
    let mut seen_exponent = false;
    while at < bytes.len() {
        let c = bytes[at];
        match c {
            b'0'..=b'9' => { cleaned.push(c as char); digits += 1; previous_digit = true; }
            b'_' => {
                if !previous_digit || !bytes.get(at + 1).is_some_and(u8::is_ascii_digit) { return None; }
                previous_digit = false;
            }
            b'.' if !seen_point && !seen_exponent => { cleaned.push('.'); seen_point = true; previous_digit = false; }
            b'e' | b'E' if !seen_exponent && digits > 0 => {
                cleaned.push('e');
                seen_exponent = true;
                previous_digit = false;
                if matches!(bytes.get(at + 1), Some(b'+' | b'-')) { at += 1; cleaned.push(bytes[at] as char); }
                if !bytes.get(at + 1).is_some_and(u8::is_ascii_digit) { return None; }
            }
            _ => return None,
        }
        at += 1;
    }
    if digits == 0 { return None; }
    cleaned.parse::<f64>().ok().map(|value| sign * value)
}

/// `float(value)`.
pub fn py_float(value: &Value) -> Result<f64> {
    match value {
        Value::Number(number) => Ok(number.as_f64().unwrap_or(f64::NAN)),
        Value::Bool(flag) => Ok(if *flag { 1.0 } else { 0.0 }),
        Value::String(text) => parse_float_text(text)
            .ok_or_else(|| Error::new(Kind::Value, format!("could not convert string to float: {}", repr(text)))),
        other => Err(Error::new(Kind::Type, format!("float() argument must be a string or a real number, not {}", repr(python_type(other))))),
    }
}

/// `int(float_value)`: toward zero.
pub fn py_int_of_float(value: f64) -> Result<i128> {
    if value.is_nan() { return Err(Error::new(Kind::Value, "cannot convert float NaN to integer")); }
    if value.is_infinite() { return Err(Error::new(Kind::Value, "cannot convert float infinity to integer")); }
    if value.abs() >= 1.7e38 { return Err(Error::new(Kind::Value, "integer is outside the supported range")); }
    Ok(value.trunc() as i128)
}

/// A count the model checks as `_nonnegative_int`: a JSON integer, never a bool.
pub fn py_count(value: &Value, field_name: &str) -> Result<i128> {
    match value {
        Value::Number(number) if !number.is_f64() => Ok(number.as_i64().map(|v| v as i128).or_else(|| number.as_u64().map(|v| v as i128)).unwrap_or(0)),
        _ => Err(Error::new(Kind::Type, format!("{field_name} must be an integer"))),
    }
}

/// An optional id: `None if value is None else str(value)`.
pub fn py_optional_str(value: &Value) -> Option<String> { if value.is_null() { None } else { Some(py_str(value)) } }

/// One timeline item `(kind, key)` from a JSON pair.
pub fn item(value: &Value) -> Result<(String, String)> {
    match value.as_array().map(Vec::as_slice) {
        Some([kind, key]) => Ok((py_str(kind), py_str(key))),
        _ => Err(Error::new(Kind::Type, "a schedule item is a [kind, key] pair")),
    }
}

/// A sequence of timeline items.
pub fn items(value: &Value) -> Result<Vec<(String, String)>> {
    value.as_array().ok_or_else(|| Error::new(Kind::Type, "schedule items must be a list of [kind, key] pairs"))?.iter().map(item).collect()
}

/// The argument at `index`, or an error in the words of a missing positional.
pub fn at(args: &[Value], index: usize) -> Result<&Value> {
    args.get(index).ok_or_else(|| Error::new(Kind::Type, format!("missing argument {}", index + 1)))
}
