//! JSON as the pulse files and the board protocol use it.
//!
//! Python's reader distinguishes `1000` (an int) from `1000.0` (a float) and a
//! bracket count of `2.0` is refused for not being an integer, so integers and
//! floats stay apart here. Objects keep their order: it is what a file says and
//! what a refusal names first. Three writers: compact with sorted keys (the
//! canonical digest form), compact in order (frames on a socket) and readable
//! (files a person reads, `zlc_durable.readable_json`).
use std::fmt::Write as _;

use crate::error::{Error, Kind, Result, repr};

#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Int(i128),
    Float(f64),
    Str(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

impl Json {
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self { Json::Object(items) => items.iter().find(|(name, _)| name == key).map(|(_, value)| value), _ => None }
    }
    pub fn as_str(&self) -> Option<&str> { if let Json::Str(text) = self { Some(text) } else { None } }
    pub fn as_array(&self) -> Option<&[Json]> { if let Json::Array(items) = self { Some(items) } else { None } }
    pub fn as_object(&self) -> Option<&[(String, Json)]> { if let Json::Object(items) = self { Some(items) } else { None } }
    pub fn as_bool(&self) -> Option<bool> { if let Json::Bool(flag) = self { Some(*flag) } else { None } }
    /// An integer that is not a bool (Python `isinstance(x, int) and not isinstance(x, bool)`).
    pub fn as_int(&self) -> Option<i128> { if let Json::Int(value) = self { Some(*value) } else { None } }
    /// A real number (int or float, not bool).
    pub fn as_f64(&self) -> Option<f64> {
        match self { Json::Int(value) => Some(*value as f64), Json::Float(value) => Some(*value), _ => None }
    }
    pub fn is_null(&self) -> bool { matches!(self, Json::Null) }
    pub fn str(text: impl Into<String>) -> Json { Json::Str(text.into()) }
    pub fn opt_str(text: Option<&str>) -> Json { text.map_or(Json::Null, Json::str) }
    pub fn object(items: Vec<(&str, Json)>) -> Json { Json::Object(items.into_iter().map(|(k, v)| (k.to_owned(), v)).collect()) }
    pub fn strings<S: AsRef<str>>(items: &[S]) -> Json { Json::Array(items.iter().map(|item| Json::str(item.as_ref())).collect()) }
    pub fn ints<T: Copy + Into<i128>>(items: &[T]) -> Json { Json::Array(items.iter().map(|item| Json::Int((*item).into())).collect()) }

    /// Python's type name for a refusal.
    pub fn type_name(&self) -> &'static str {
        match self {
            Json::Null => "NoneType", Json::Bool(_) => "bool", Json::Int(_) => "int", Json::Float(_) => "float",
            Json::Str(_) => "str", Json::Array(_) => "list", Json::Object(_) => "dict",
        }
    }

    // ----------------------------------------------------------------- writers
    /// `json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))`.
    pub fn canonical(&self) -> String { let mut out = String::new(); self.compact(&mut out, true); out }

    /// `json.dumps(value, separators=(",", ":"))` with keys in their order.
    pub fn compact_text(&self) -> String { let mut out = String::new(); self.compact(&mut out, false); out }

    fn compact(&self, out: &mut String, sorted: bool) {
        match self {
            Json::Array(items) => {
                out.push('[');
                for (index, item) in items.iter().enumerate() {
                    if index > 0 { out.push(','); }
                    item.compact(out, sorted);
                }
                out.push(']');
            }
            Json::Object(items) => {
                out.push('{');
                let mut order: Vec<&(String, Json)> = items.iter().collect();
                if sorted { order.sort_by(|a, b| a.0.cmp(&b.0)); }
                for (index, (key, value)) in order.into_iter().enumerate() {
                    if index > 0 { out.push(','); }
                    write_string(out, key);
                    out.push(':');
                    value.compact(out, sorted);
                }
                out.push('}');
            }
            scalar => scalar.scalar(out),
        }
    }

    fn scalar(&self, out: &mut String) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(flag) => out.push_str(if *flag { "true" } else { "false" }),
            Json::Int(value) => { let _ = write!(out, "{value}"); }
            Json::Float(value) => out.push_str(&float_text(*value)),
            Json::Str(text) => write_string(out, text),
            _ => unreachable!("not a scalar"),
        }
    }

    fn is_scalar(&self) -> bool { !matches!(self, Json::Array(_) | Json::Object(_)) }

    /// `zlc_durable.readable_json(tree)`: objects expanded, a list of scalars on
    /// one line or wrapped at 96 columns, a trailing newline.
    pub fn readable(&self) -> String {
        let mut out = String::new();
        self.render(&mut out, 0);
        out.push('\n');
        out
    }

    fn render(&self, out: &mut String, depth: usize) {
        let pad = " ".repeat(2 * depth);
        let inner = " ".repeat(2 * (depth + 1));
        match self {
            Json::Object(items) if items.is_empty() => out.push_str("{}"),
            Json::Object(items) => {
                out.push_str("{\n");
                for (index, (key, value)) in items.iter().enumerate() {
                    if index > 0 { out.push_str(",\n"); }
                    out.push_str(&inner);
                    write_string(out, key);
                    out.push_str(": ");
                    value.render(out, depth + 1);
                }
                out.push('\n');
                out.push_str(&pad);
                out.push('}');
            }
            Json::Array(items) if items.is_empty() => out.push_str("[]"),
            Json::Array(items) if items.iter().all(Json::is_scalar) => {
                let parts: Vec<String> = items.iter().map(|item| { let mut text = String::new(); item.scalar(&mut text); text }).collect();
                let one_line = format!("[{}]", parts.join(", "));
                if one_line.chars().count() + pad.len() <= 96 { out.push_str(&one_line); return; }
                let mut lines = Vec::new();
                let mut current = inner.clone();
                for (index, part) in parts.iter().enumerate() {
                    let piece = if index + 1 < parts.len() { format!("{part},") } else { part.clone() };
                    if current != inner && current.chars().count() + 1 + piece.chars().count() > 96 {
                        lines.push(std::mem::replace(&mut current, inner.clone()));
                    }
                    if current != inner { current.push(' '); }
                    current.push_str(&piece);
                }
                lines.push(current);
                out.push_str("[\n");
                out.push_str(&lines.join("\n"));
                out.push('\n');
                out.push_str(&pad);
                out.push(']');
            }
            Json::Array(items) => {
                out.push_str("[\n");
                for (index, item) in items.iter().enumerate() {
                    if index > 0 { out.push_str(",\n"); }
                    out.push_str(&inner);
                    item.render(out, depth + 1);
                }
                out.push('\n');
                out.push_str(&pad);
                out.push(']');
            }
            scalar => scalar.scalar(out),
        }
    }
}

/// `float.__repr__`, which `json.dumps` writes.
pub fn float_text(value: f64) -> String { zlc_core::pyfmt::repr(value) }

/// A JSON string as `json.dumps(text, ensure_ascii=False)` writes it.
fn write_string(out: &mut String, text: &str) {
    out.push('"');
    for character in text.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => { let _ = write!(out, "\\u{:04x}", c as u32); }
            c => out.push(c),
        }
    }
    out.push('"');
}

// --------------------------------------------------------------------- reader
/// `zlc_durable.strict_json_loads(text, what)`: one JSON document, refusing a
/// duplicated key, a non-finite constant and a number that overflows to infinity.
pub fn parse(text: &str, what: &str) -> Result<Json> {
    let mut parser = Parser { bytes: text.as_bytes(), text, at: 0, what };
    parser.space();
    let value = parser.value()?;
    parser.space();
    if parser.at != parser.bytes.len() { return Err(parser.fail("Extra data")); }
    Ok(value)
}

struct Parser<'a> { bytes: &'a [u8], text: &'a str, at: usize, what: &'a str }

impl Parser<'_> {
    fn fail(&self, message: &str) -> Error {
        let line = self.text[..self.at.min(self.text.len())].matches('\n').count() + 1;
        let column = self.at - self.text[..self.at.min(self.text.len())].rfind('\n').map_or(0, |at| at + 1) + 1;
        Error::new(Kind::Value, format!("{message}: line {line} column {column} (char {})", self.at))
    }

    fn space(&mut self) { while self.at < self.bytes.len() && matches!(self.bytes[self.at], b' ' | b'\t' | b'\n' | b'\r') { self.at += 1; } }

    fn value(&mut self) -> Result<Json> {
        match self.bytes.get(self.at) {
            None => Err(self.fail("Expecting value")),
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b'n') if self.text[self.at..].starts_with("null") => { self.at += 4; Ok(Json::Null) }
            Some(b't') if self.text[self.at..].starts_with("true") => { self.at += 4; Ok(Json::Bool(true)) }
            Some(b'f') if self.text[self.at..].starts_with("false") => { self.at += 5; Ok(Json::Bool(false)) }
            Some(b'N') if self.text[self.at..].starts_with("NaN") => self.constant("NaN"),
            Some(b'I') if self.text[self.at..].starts_with("Infinity") => self.constant("Infinity"),
            Some(b'-') if self.text[self.at..].starts_with("-Infinity") => self.constant("-Infinity"),
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(_) => Err(self.fail("Expecting value")),
        }
    }

    fn constant(&self, name: &str) -> Result<Json> {
        Err(Error::new(Kind::Value, format!("non-finite JSON constant {name} in {}", self.what)))
    }

    fn number(&mut self) -> Result<Json> {
        let start = self.at;
        if self.bytes[self.at] == b'-' { self.at += 1; }
        match self.bytes.get(self.at) {
            Some(b'0') => self.at += 1,
            Some(b'1'..=b'9') => while self.bytes.get(self.at).is_some_and(u8::is_ascii_digit) { self.at += 1; },
            _ => { self.at = start; return Err(self.fail("Expecting value")); }
        }
        let mut float = false;
        if self.bytes.get(self.at) == Some(&b'.') && self.bytes.get(self.at + 1).is_some_and(u8::is_ascii_digit) {
            float = true;
            self.at += 1;
            while self.bytes.get(self.at).is_some_and(u8::is_ascii_digit) { self.at += 1; }
        }
        if matches!(self.bytes.get(self.at), Some(b'e' | b'E')) {
            let mut look = self.at + 1;
            if matches!(self.bytes.get(look), Some(b'+' | b'-')) { look += 1; }
            if self.bytes.get(look).is_some_and(u8::is_ascii_digit) {
                float = true;
                self.at = look;
                while self.bytes.get(self.at).is_some_and(u8::is_ascii_digit) { self.at += 1; }
            }
        }
        let literal = &self.text[start..self.at];
        if float {
            let value: f64 = literal.parse().map_err(|_| self.fail("Expecting value"))?;
            if !value.is_finite() {
                return Err(Error::new(Kind::Value, format!("non-finite JSON number {literal} in {}", self.what)));
            }
            Ok(Json::Float(value))
        } else {
            literal.parse::<i128>().map(Json::Int)
                .map_err(|_| Error::new(Kind::Value, format!("integer {literal} in {} is outside the supported range", self.what)))
        }
    }

    fn string(&mut self) -> Result<String> {
        self.at += 1;
        let mut out = String::new();
        loop {
            let Some(&byte) = self.bytes.get(self.at) else { return Err(self.fail("Unterminated string starting at")); };
            match byte {
                b'"' => { self.at += 1; return Ok(out); }
                b'\\' => {
                    let escape = *self.bytes.get(self.at + 1).ok_or_else(|| self.fail("Unterminated string starting at"))?;
                    self.at += 2;
                    match escape {
                        b'"' => out.push('"'), b'\\' => out.push('\\'), b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'), b'f' => out.push('\u{c}'), b'n' => out.push('\n'),
                        b'r' => out.push('\r'), b't' => out.push('\t'),
                        b'u' => {
                            let first = self.hex4()?;
                            let code = if (0xd800..0xdc00).contains(&first) && self.text[self.at..].starts_with("\\u") {
                                self.at += 2;
                                let second = self.hex4()?;
                                if !(0xdc00..0xe000).contains(&second) { return Err(self.fail("Invalid \\uXXXX escape")); }
                                0x10000 + ((first - 0xd800) << 10) + (second - 0xdc00)
                            } else { first };
                            out.push(char::from_u32(code).ok_or_else(|| self.fail("Invalid \\uXXXX escape"))?);
                        }
                        _ => { self.at -= 1; return Err(self.fail("Invalid \\escape")); }
                    }
                }
                byte if byte < 0x20 => return Err(self.fail("Invalid control character at")),
                _ => {
                    let rest = &self.text[self.at..];
                    let character = rest.chars().next().expect("a character");
                    out.push(character);
                    self.at += character.len_utf8();
                }
            }
        }
    }

    fn hex4(&mut self) -> Result<u32> {
        let digits = self.text.get(self.at..self.at + 4).ok_or_else(|| self.fail("Invalid \\uXXXX escape"))?;
        let value = u32::from_str_radix(digits, 16).map_err(|_| self.fail("Invalid \\uXXXX escape"))?;
        self.at += 4;
        Ok(value)
    }

    fn array(&mut self) -> Result<Json> {
        self.at += 1;
        let mut items = Vec::new();
        self.space();
        if self.bytes.get(self.at) == Some(&b']') { self.at += 1; return Ok(Json::Array(items)); }
        loop {
            self.space();
            items.push(self.value()?);
            self.space();
            match self.bytes.get(self.at) {
                Some(b',') => self.at += 1,
                Some(b']') => { self.at += 1; return Ok(Json::Array(items)); }
                _ => return Err(self.fail("Expecting ',' delimiter")),
            }
        }
    }

    fn object(&mut self) -> Result<Json> {
        self.at += 1;
        let mut items: Vec<(String, Json)> = Vec::new();
        self.space();
        if self.bytes.get(self.at) == Some(&b'}') { self.at += 1; return Ok(Json::Object(items)); }
        loop {
            self.space();
            if self.bytes.get(self.at) != Some(&b'"') { return Err(self.fail("Expecting property name enclosed in double quotes")); }
            let key = self.string()?;
            self.space();
            if self.bytes.get(self.at) != Some(&b':') { return Err(self.fail("Expecting ':' delimiter")); }
            self.at += 1;
            self.space();
            let value = self.value()?;
            if items.iter().any(|(name, _)| *name == key) {
                return Err(Error::new(Kind::Value, format!("duplicate key {} in {}", repr(&key), self.what)));
            }
            items.push((key, value));
            self.space();
            match self.bytes.get(self.at) {
                Some(b',') => self.at += 1,
                Some(b'}') => { self.at += 1; return Ok(Json::Object(items)); }
                _ => return Err(self.fail("Expecting ',' delimiter")),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integers_and_floats_stay_apart_and_strictness_holds() {
        let value = parse(r#"{"a": 1000, "b": 1000.0, "c": [1e-07, -0.5], "d": "µs\n"}"#, "test").unwrap();
        assert_eq!(value.get("a"), Some(&Json::Int(1000)));
        assert_eq!(value.get("b"), Some(&Json::Float(1000.0)));
        assert_eq!(value.canonical(), "{\"a\":1000,\"b\":1000.0,\"c\":[1e-07,-0.5],\"d\":\"µs\\n\"}");
        assert!(parse(r#"{"a": 1, "a": 2}"#, "x").unwrap_err().message.contains("duplicate key 'a' in x"));
        assert!(parse("[NaN]", "x").is_err());
        assert!(parse("[1e999]", "x").unwrap_err().message.contains("non-finite JSON number 1e999"));
    }

    #[test]
    fn readable_layout_matches_zlc_durable() {
        let names: Vec<Json> = (0..40).map(|i| Json::str(format!("ch{i:02}"))).collect();
        let tree = Json::object(vec![("lanes", Json::Array(names)), ("empty", Json::Array(vec![])), ("nested", Json::object(vec![("x", Json::Int(1))]))]);
        let text = tree.readable();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], "{");
        assert_eq!(lines[1], "  \"lanes\": [");
        assert!(lines[2].starts_with("    \"ch00\", \"ch01\""));
        assert!(lines.iter().all(|line| line.chars().count() <= 96));
        assert!(text.ends_with("  \"nested\": {\n    \"x\": 1\n  }\n}\n"));
    }
}
