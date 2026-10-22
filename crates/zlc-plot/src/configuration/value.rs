//! The dynamic values a wire message carries: Python's `None`, `bool`, `int`,
//! `float`, `str`, sequences and string-keyed mappings, with Python's own
//! equality (`1 == 1.0 == True`), `str()` and `repr()`.
//!
//! Everything authored crosses the boundary as one of these; the typed layer
//! (specs, parameters, selectors) decodes them exactly where Python did, so a
//! wrong type is refused with the same exception class at the same step.

use std::fmt::Write as _;
use std::sync::Arc;

use crate::pyfmt::repr as float_repr;

use super::error::{PlotError, Result};

/// One Python value as authored records hold it.  Tuples and lists are one
/// variant: no configuration path distinguishes them.
#[derive(Clone, Debug)]
pub enum Value {
    None,
    Bool(bool),
    Int(i128),
    Float(f64),
    Text(Arc<str>),
    List(Arc<[Value]>),
    Map(ValueMap),
}

/// An insertion-ordered string-keyed mapping (a Python `dict` of a record).
#[derive(Clone, Debug, Default)]
pub struct ValueMap(Arc<Vec<(Arc<str>, Value)>>);

impl ValueMap {
    pub fn new() -> Self { Self::default() }
    pub fn from_pairs<K: Into<Arc<str>>>(pairs: impl IntoIterator<Item = (K, Value)>) -> Self {
        let mut map = Self::new();
        for (key, value) in pairs { map.insert(key.into(), value); }
        map
    }
    pub fn len(&self) -> usize { self.0.len() }
    pub fn is_empty(&self) -> bool { self.0.is_empty() }
    pub fn get(&self, key: &str) -> Option<&Value> { self.0.iter().find(|(name, _)| &**name == key).map(|(_, value)| value) }
    pub fn contains_key(&self, key: &str) -> bool { self.get(key).is_some() }
    pub fn keys(&self) -> impl Iterator<Item = &str> { self.0.iter().map(|(key, _)| &**key) }
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Value)> { self.0.iter().map(|(key, value)| (&**key, value)) }
    /// Python `d[key] = value`: an existing key keeps its position.
    pub fn insert(&mut self, key: impl Into<Arc<str>>, value: Value) {
        let key = key.into();
        let entries = Arc::make_mut(&mut self.0);
        match entries.iter_mut().find(|(name, _)| *name == key) {
            Some(slot) => slot.1 = value,
            None => entries.push((key, value)),
        }
    }
    /// Python `d.setdefault(key, value)`.
    pub fn set_default(&mut self, key: &str, value: Value) {
        if !self.contains_key(key) { self.insert(key, value); }
    }
    pub fn remove(&mut self, key: &str) -> Option<Value> {
        let position = self.0.iter().position(|(name, _)| &**name == key)?;
        Some(Arc::make_mut(&mut self.0).remove(position).1)
    }
    /// Python `d.update(other)`.
    pub fn update(&mut self, other: &ValueMap) {
        for (key, value) in other.0.iter() { self.insert(key.clone(), value.clone()); }
    }
    pub fn retain(&mut self, mut keep: impl FnMut(&str, &Value) -> bool) {
        Arc::make_mut(&mut self.0).retain(|(key, value)| keep(key, value));
    }
}

impl PartialEq for ValueMap {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().all(|(key, value)| other.get(key) == Some(value))
    }
}

impl<K: Into<Arc<str>>> FromIterator<(K, Value)> for ValueMap {
    fn from_iter<I: IntoIterator<Item = (K, Value)>>(iter: I) -> Self { Self::from_pairs(iter) }
}

/// Python's exact comparison of an int with a float.
fn int_equals_float(integer: i128, float: f64) -> bool {
    float.is_finite() && float.fract() == 0.0 && float >= -(2f64.powi(127)) && float < 2f64.powi(127) && float as i128 == integer
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        use Value::*;
        match (self, other) {
            (None, None) => true,
            (Text(a), Text(b)) => a == b,
            (List(a), List(b)) => a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| x == y),
            (Map(a), Map(b)) => a == b,
            (Float(a), Float(b)) => a == b,
            (Float(f), other) | (other, Float(f)) => other.as_exact_int().is_some_and(|i| int_equals_float(i, *f)),
            (a, b) => match (a.as_exact_int(), b.as_exact_int()) { (Some(x), Some(y)) => x == y, _ => false },
        }
    }
}

impl Value {
    pub fn text(value: impl Into<Arc<str>>) -> Self { Value::Text(value.into()) }
    pub fn list(values: impl IntoIterator<Item = Value>) -> Self { Value::List(values.into_iter().collect::<Vec<_>>().into()) }
    pub fn is_none(&self) -> bool { matches!(self, Value::None) }
    pub fn as_str(&self) -> Option<&str> { if let Value::Text(text) = self { Some(text) } else { None } }
    pub fn as_map(&self) -> Option<&ValueMap> { if let Value::Map(map) = self { Some(map) } else { None } }
    pub fn as_list(&self) -> Option<&[Value]> { if let Value::List(list) = self { Some(list) } else { None } }
    /// `bool` and `int` as the integer Python compares them as.
    fn as_exact_int(&self) -> Option<i128> {
        match self { Value::Bool(flag) => Some(i128::from(*flag)), Value::Int(value) => Some(*value), _ => None }
    }
    /// `isinstance(value, Real) and not isinstance(value, bool)`.
    pub fn is_real(&self) -> bool { matches!(self, Value::Int(_) | Value::Float(_)) }
    /// `float(value)` of an int or float.
    pub fn real(&self) -> Option<f64> {
        match self { Value::Int(value) => Some(*value as f64), Value::Float(value) => Some(*value), _ => Option::None }
    }
    /// The Python type name, as error messages print it.
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::None => "NoneType", Value::Bool(_) => "bool", Value::Int(_) => "int", Value::Float(_) => "float",
            Value::Text(_) => "str", Value::List(_) => "list", Value::Map(_) => "dict",
        }
    }

    /// Python `str(value)`.
    pub fn py_str(&self) -> String {
        match self { Value::Text(text) => text.to_string(), other => other.py_repr() }
    }

    /// Python `repr(value)` (a sequence reads as a list).
    pub fn py_repr(&self) -> String {
        match self {
            Value::None => "None".into(),
            Value::Bool(flag) => if *flag { "True" } else { "False" }.into(),
            Value::Int(value) => value.to_string(),
            Value::Float(value) => float_repr(*value),
            Value::Text(text) => repr_str(text),
            Value::List(values) => {
                let mut text = String::from("[");
                for (index, value) in values.iter().enumerate() {
                    if index > 0 { text.push_str(", "); }
                    text.push_str(&value.py_repr());
                }
                text + "]"
            }
            Value::Map(map) => {
                let mut text = String::from("{");
                for (index, (key, value)) in map.iter().enumerate() {
                    if index > 0 { text.push_str(", "); }
                    let _ = write!(text, "{}: {}", repr_str(key), value.py_repr());
                }
                text + "}"
            }
        }
    }

    /// Python's `float(value)`, which also parses text.
    pub fn py_float(&self) -> Result<f64> {
        match self {
            Value::Bool(flag) => Ok(f64::from(u8::from(*flag))),
            Value::Int(value) => Ok(*value as f64),
            Value::Float(value) => Ok(*value),
            Value::Text(text) => parse_python_float(text)
                .ok_or_else(|| PlotError::value(format!("could not convert string to float: {}", repr_str(text)))),
            other => Err(PlotError::type_error(format!(
                "float() argument must be a string or a real number, not '{}'", other.type_name()))),
        }
    }

    /// Plain JSON, as `json.loads` would have produced this value.
    pub fn from_json(value: &serde_json::Value) -> Value {
        match value {
            serde_json::Value::Null => Value::None,
            serde_json::Value::Bool(flag) => Value::Bool(*flag),
            serde_json::Value::Number(number) => {
                if let Some(integer) = number.as_i64() { Value::Int(i128::from(integer)) }
                else if let Some(integer) = number.as_u64() { Value::Int(i128::from(integer)) }
                else { Value::Float(number.as_f64().unwrap_or(f64::NAN)) }
            }
            serde_json::Value::String(text) => Value::text(text.as_str()),
            serde_json::Value::Array(values) => Value::list(values.iter().map(Value::from_json)),
            serde_json::Value::Object(map) => Value::Map(map.iter().map(|(key, value)| (key.as_str(), Value::from_json(value))).collect()),
        }
    }

    /// Plain JSON, as `json.dumps` writes it (a float keeps its float type).
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            Value::None => serde_json::Value::Null,
            Value::Bool(flag) => serde_json::Value::Bool(*flag),
            Value::Int(value) => i64::try_from(*value).map(serde_json::Value::from)
                .or_else(|_| u64::try_from(*value).map(serde_json::Value::from))
                .unwrap_or_else(|_| serde_json::Value::from(*value as f64)),
            Value::Float(value) => serde_json::Number::from_f64(*value).map(serde_json::Value::Number).unwrap_or(serde_json::Value::Null),
            Value::Text(text) => serde_json::Value::String(text.to_string()),
            Value::List(values) => serde_json::Value::Array(values.iter().map(Value::to_json).collect()),
            Value::Map(map) => serde_json::Value::Object(map.iter().map(|(key, value)| (key.to_owned(), value.to_json())).collect()),
        }
    }
}

impl From<bool> for Value { fn from(value: bool) -> Self { Value::Bool(value) } }
impl From<i64> for Value { fn from(value: i64) -> Self { Value::Int(i128::from(value)) } }
impl From<f64> for Value { fn from(value: f64) -> Self { Value::Float(value) } }
impl From<&str> for Value { fn from(value: &str) -> Self { Value::text(value) } }
impl From<String> for Value { fn from(value: String) -> Self { Value::text(value) } }
impl From<Arc<str>> for Value { fn from(value: Arc<str>) -> Self { Value::Text(value) } }
impl<T: Into<Value>> From<Option<T>> for Value {
    fn from(value: Option<T>) -> Self { value.map_or(Value::None, Into::into) }
}

/// Characters Python's `str.isspace` admits that `char::is_whitespace` does not.
fn python_space(character: char) -> bool {
    character.is_whitespace() || matches!(character, '\u{1c}'..='\u{1f}')
}

/// Python `text.strip()`.
pub fn py_strip(text: &str) -> &str { text.trim_matches(python_space) }

/// Python `str.isprintable` for one character, from the Unicode categories
/// that are not printable (Cc, Cf, Cs, Co, Cn, Zl, Zp and every Zs but space),
/// over the ranges a configuration text can plausibly meet.
fn printable(character: char) -> bool {
    let code = character as u32;
    if character == ' ' { return true; }
    !(code < 0x20 || (0x7f..=0xa0).contains(&code) || code == 0xad
        || (0x600..=0x605).contains(&code) || code == 0x61c || code == 0x6dd || code == 0x70f || code == 0x1680
        || code == 0x180e || (0x2000..=0x200f).contains(&code) || (0x2028..=0x202f).contains(&code)
        || (0x205f..=0x2064).contains(&code) || (0x2066..=0x206f).contains(&code) || code == 0x3000
        || (0xd800..=0xf8ff).contains(&code) || code == 0xfeff || (0xfff9..=0xfffb).contains(&code)
        || (0xfffe..=0xffff).contains(&code) || (0xe0000..=0xe007f).contains(&code)
        || (0xf0000..).contains(&code))
}

/// Python `repr(text)`.
pub fn repr_str(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') { '"' } else { '\'' };
    let mut result = String::with_capacity(text.len() + 2);
    result.push(quote);
    for character in text.chars() {
        match character {
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            c if c == quote => { result.push('\\'); result.push(c); }
            c if printable(c) => result.push(c),
            c => {
                let code = c as u32;
                if code < 0x100 { let _ = write!(result, "\\x{code:02x}"); }
                else if code < 0x10000 { let _ = write!(result, "\\u{code:04x}"); }
                else { let _ = write!(result, "\\U{code:08x}"); }
            }
        }
    }
    result.push(quote);
    result
}

/// Python's `float(text)`: surrounding whitespace, an optional sign, decimal
/// digits with single underscores between them, `inf`/`infinity`/`nan`.
pub fn parse_python_float(text: &str) -> Option<f64> {
    let body = py_strip(text);
    let (negative, unsigned) = match body.as_bytes().first() {
        Some(b'-') => (true, &body[1..]),
        Some(b'+') => (false, &body[1..]),
        _ => (false, body),
    };
    let lower = unsigned.to_ascii_lowercase();
    let special = match lower.as_str() { "inf" | "infinity" => Some(f64::INFINITY), "nan" => Some(f64::NAN), _ => None };
    if let Some(value) = special { return Some(if negative { -value } else { value }); }
    // Underscores are legal only between two digits.
    let bytes = unsigned.as_bytes();
    for (index, byte) in bytes.iter().enumerate() {
        if *byte == b'_' {
            let before = index.checked_sub(1).map(|i| bytes[i].is_ascii_digit()).unwrap_or(false);
            let after = bytes.get(index + 1).is_some_and(u8::is_ascii_digit);
            if !before || !after { return None; }
        } else if !(byte.is_ascii_digit() || matches!(byte, b'.' | b'e' | b'E' | b'+' | b'-')) {
            return None;
        }
    }
    let cleaned: String = unsigned.chars().filter(|c| *c != '_').collect();
    if cleaned.is_empty() || cleaned.starts_with(['+', '-']) { return None; }
    let (mantissa, exponent) = match cleaned.find(['e', 'E']) {
        Some(position) => (&cleaned[..position], Some(&cleaned[position + 1..])),
        None => (cleaned.as_str(), None),
    };
    if mantissa.is_empty() || mantissa == "." || mantissa.matches('.').count() > 1 || mantissa.contains(['+', '-']) { return None; }
    if let Some(exponent) = exponent {
        let digits = exponent.strip_prefix(['+', '-']).unwrap_or(exponent);
        if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) { return None; }
    }
    let value: f64 = cleaned.parse().ok()?;
    Some(if negative { -value } else { value })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_equality_and_text() {
        assert_eq!(Value::Int(1), Value::Float(1.0));
        assert_eq!(Value::Bool(true), Value::Int(1));
        assert_ne!(Value::Int(9_007_199_254_740_993), Value::Float(9_007_199_254_740_992.0));
        assert_eq!(Value::Float(-0.0), Value::Int(0));
        assert_eq!(repr_str("it's"), "\"it's\"");
        assert_eq!(repr_str("a'\"b"), "'a\\'\"b'");
        assert_eq!(Value::list([Value::text("scope-value"), Value::Int(3)]).py_repr(), "['scope-value', 3]");
        assert_eq!(parse_python_float(" 1_000.5 "), Some(1000.5));
        assert_eq!(parse_python_float("1__0"), None);
        assert_eq!(parse_python_float("-Infinity"), Some(f64::NEG_INFINITY));
        assert_eq!(parse_python_float(".5e-1"), Some(0.05));
        assert_eq!(parse_python_float("0x10"), None);
        assert_eq!(py_strip("\u{1f} a \n"), "a");
    }
}
