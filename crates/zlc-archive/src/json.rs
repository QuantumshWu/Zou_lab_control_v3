//! JSON exactly as the scientific code reads and writes it: `json.loads`
//! (CPython's C scanner, with the figure reader's strict hooks or plain)
//! and `json.dumps`, including Python's float `repr`, key order, escaping,
//! lone surrogates (archive text, see [`crate::pystr`]) and the depth at
//! which the interpreter runs out of recursion.

use zlc_core::pyfmt;

use crate::error::{ArchiveError, ErrorKind, Result};
use crate::pystr;

/// One JSON value as `json.loads` returns it; objects keep document order.
#[derive(Clone, Debug)]
pub enum Json {
    Null,
    Bool(bool),
    Int(i128),
    /// An int beyond 128 bits, as its decimal text.
    BigInt(String),
    Float(f64),
    Str(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

impl Drop for Json {
    /// Children are released from a heap stack: a document may nest
    /// thousands of levels.
    fn drop(&mut self) {
        let mut pending: Vec<Json> = match self {
            Json::Array(items) if !items.is_empty() => std::mem::take(items),
            Json::Object(pairs) if !pairs.is_empty() => std::mem::take(pairs).into_iter().map(|(_, value)| value).collect(),
            _ => return,
        };
        while let Some(mut value) = pending.pop() {
            match &mut value {
                Json::Array(items) => pending.append(items),
                Json::Object(pairs) => pending.extend(std::mem::take(pairs).into_iter().map(|(_, value)| value)),
                _ => {}
            }
        }
    }
}

impl Json {
    pub fn object<K: Into<String>>(pairs: impl IntoIterator<Item = (K, Json)>) -> Self { Json::Object(pairs.into_iter().map(|(k, v)| (k.into(), v)).collect()) }
    pub fn str(text: impl Into<String>) -> Self { Json::Str(text.into()) }

    pub fn get(&self, key: &str) -> Option<&Json> {
        match self { Json::Object(pairs) => pairs.iter().find(|(name, _)| name == key).map(|(_, value)| value), _ => None }
    }
    pub fn as_object(&self) -> Option<&[(String, Json)]> { match self { Json::Object(pairs) => Some(pairs), _ => None } }
    pub fn as_array(&self) -> Option<&[Json]> { match self { Json::Array(items) => Some(items), _ => None } }
    pub fn as_str(&self) -> Option<&str> { match self { Json::Str(text) => Some(text), _ => None } }
    pub fn is_null(&self) -> bool { matches!(self, Json::Null) }
    /// `isinstance(value, int) and not isinstance(value, bool)`.
    pub fn as_int(&self) -> Option<i128> { match self { Json::Int(value) => Some(*value), _ => None } }

    /// Insert or replace one key (`dict[key] = value`).
    pub fn insert(&mut self, key: &str, value: Json) {
        if let Json::Object(pairs) = self {
            match pairs.iter_mut().find(|(name, _)| name == key) { Some(slot) => slot.1 = value, None => pairs.push((key.into(), value)) }
        }
    }
    pub fn remove(&mut self, key: &str) -> Option<Json> {
        let Json::Object(pairs) = self else { return None };
        let position = pairs.iter().position(|(name, _)| name == key)?;
        Some(pairs.remove(position).1)
    }

    /// `type(value).__name__` of the Python value.
    pub fn type_name(&self) -> &'static str {
        match self { Json::Null => "NoneType", Json::Bool(_) => "bool", Json::Int(_) | Json::BigInt(_) => "int", Json::Float(_) => "float",
                     Json::Str(_) => "str", Json::Array(_) => "list", Json::Object(_) => "dict" }
    }

    /// The nesting depth: a scalar is 1, a container one more than its deepest item.
    pub fn depth(&self) -> usize {
        let mut deepest = 0;
        let mut pending: Vec<(&Json, usize)> = vec![(self, 1)];
        while let Some((value, depth)) = pending.pop() {
            deepest = deepest.max(depth);
            match value {
                Json::Array(items) => pending.extend(items.iter().map(|item| (item, depth + 1))),
                Json::Object(pairs) => pending.extend(pairs.iter().map(|(_, item)| (item, depth + 1))),
                _ => {}
            }
        }
        deepest
    }

    /// From `serde_json` (a Runtime metadata document).
    pub fn from_serde(value: &serde_json::Value) -> Self {
        match value {
            serde_json::Value::Null => Json::Null,
            serde_json::Value::Bool(flag) => Json::Bool(*flag),
            serde_json::Value::Number(number) => match (number.as_i64(), number.as_u64()) {
                (Some(integer), _) => Json::Int(i128::from(integer)),
                (_, Some(integer)) => Json::Int(i128::from(integer)),
                _ => Json::Float(number.as_f64().unwrap_or(f64::NAN)),
            },
            serde_json::Value::String(text) => Json::Str(pystr::from_rust(text)),
            serde_json::Value::Array(items) => Json::Array(items.iter().map(Json::from_serde).collect()),
            serde_json::Value::Object(map) => Json::Object(map.iter().map(|(key, value)| (pystr::from_rust(key), Json::from_serde(value))).collect()),
        }
    }

    /// To `serde_json`; a 128-bit or larger int becomes its nearest double.
    pub fn to_serde(&self) -> serde_json::Value {
        match self {
            Json::Null => serde_json::Value::Null,
            Json::Bool(flag) => serde_json::Value::Bool(*flag),
            Json::Int(integer) => match (i64::try_from(*integer), u64::try_from(*integer)) {
                (Ok(value), _) => value.into(),
                (_, Ok(value)) => value.into(),
                _ => serde_json::Number::from_f64(*integer as f64).map_or(serde_json::Value::Null, serde_json::Value::Number),
            },
            Json::BigInt(text) => serde_json::Number::from_f64(text.parse().unwrap_or(f64::NAN)).map_or(serde_json::Value::Null, serde_json::Value::Number),
            Json::Float(real) => serde_json::Number::from_f64(*real).map_or(serde_json::Value::Null, serde_json::Value::Number),
            Json::Str(text) => serde_json::Value::String(text.clone()),
            Json::Array(items) => serde_json::Value::Array(items.iter().map(Json::to_serde).collect()),
            Json::Object(pairs) => serde_json::Value::Object(pairs.iter().map(|(key, value)| (key.clone(), value.to_serde())).collect()),
        }
    }
}

fn number(value: &Json) -> Option<f64> {
    match value { Json::Bool(flag) => Some(f64::from(u8::from(*flag))), Json::Float(real) => Some(*real), _ => None }
}

fn integer(value: &Json) -> Option<i128> { match value { Json::Bool(flag) => Some(i128::from(*flag)), Json::Int(integer) => Some(*integer), _ => None } }

/// Python `==` (dicts compare as mappings; bool, int and float as numbers).
impl PartialEq for Json {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Json::Null, Json::Null) => true,
            (Json::Str(a), Json::Str(b)) | (Json::BigInt(a), Json::BigInt(b)) => a == b,
            (Json::Array(a), Json::Array(b)) => a == b,
            (Json::Object(a), Json::Object(b)) => a.len() == b.len() && a.iter().all(|(key, value)| other.get(key) == Some(value)),
            (a, b) => match (integer(a), integer(b)) {
                (Some(x), Some(y)) => x == y,
                _ => match (integer(a), number(b), number(a), integer(b)) {
                    (Some(x), Some(y), _, _) | (_, _, Some(y), Some(x)) => y.fract() == 0.0 && y.abs() < 1.7e38 && y as i128 == x,
                    _ => matches!((number(a), number(b)), (Some(x), Some(y)) if x == y),
                },
            },
        }
    }
}

// ---------------------------------------------------------------- parsing

/// The deepest container CPython's scanner opens before its recursion
/// guard trips, counted from the document root (1), as `read_archive`
/// calls it from a script.
const MAX_CONTAINER_DEPTH: usize = 2998;
/// The deepest object whose pairs hook (a Python call) still runs.
const MAX_HOOK_OBJECT_DEPTH: usize = 2996;
/// The deepest container whose float hook still runs.
const MAX_FLOAT_HOOK_DEPTH: usize = 2995;

/// `str(JSONDecodeError(message, text, index))`, positions in code points.
fn decode_message(text: &str, message: &str, byte: usize) -> String {
    let before = &text[..byte];
    let position = pystr::len(before);
    let line = before.matches('\n').count() + 1;
    let column = match before.rfind('\n') { Some(newline) => position - pystr::len(&text[..newline]), None => position + 1 };
    format!("{message}: line {line} column {column} (char {position})")
}

/// What the scanner reports instead of a value: `StopIteration(idx)`, a
/// `JSONDecodeError`, or another exception.
enum Failure { Stop(usize), Decode(String), Error(ArchiveError) }

type Scan<T> = std::result::Result<T, Failure>;

/// The figure reader's hooks, or plain `json.loads`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode { Figure, Plain }

/// A container being filled.
enum Frame { Array(Vec<Json>), Object(Vec<(String, Json)>, String) }

struct Parser<'a> { text: &'a str, bytes: &'a [u8], mode: Mode }

fn whitespace(byte: u8) -> bool { matches!(byte, b' ' | b'\t' | b'\n' | b'\r') }

fn recursion(message: &str) -> Failure { Failure::Error(ArchiveError::new(ErrorKind::Recursion, message)) }

impl Parser<'_> {
    fn skip(&self, mut index: usize) -> usize {
        while index < self.bytes.len() && whitespace(self.bytes[index]) { index += 1; }
        index
    }
    fn at(&self, index: usize) -> Option<u8> { self.bytes.get(index).copied() }
    fn error(&self, message: &str, index: usize) -> Failure { Failure::Decode(decode_message(self.text, message, index)) }

    fn constant(&self, name: &str, value: f64) -> Scan<Json> {
        match self.mode {
            Mode::Figure => Err(Failure::Error(ArchiveError::value(format!("figure contains non-finite metadata number {name}")))),
            Mode::Plain => Ok(Json::Float(value)),
        }
    }

    /// A scalar at `index` (`scan_once` without containers); `depth` is the
    /// level of the container it sits in.
    fn scalar(&self, index: usize, depth: usize) -> Scan<(Json, usize)> {
        let rest = &self.bytes[index.min(self.bytes.len())..];
        match rest.first() {
            None => Err(Failure::Stop(index)),
            Some(b'"') => self.string(index + 1).map(|(text, end)| (Json::Str(text), end)),
            _ if rest.starts_with(b"null") => Ok((Json::Null, index + 4)),
            _ if rest.starts_with(b"true") => Ok((Json::Bool(true), index + 4)),
            _ if rest.starts_with(b"false") => Ok((Json::Bool(false), index + 5)),
            _ if rest.starts_with(b"NaN") => self.constant("NaN", f64::NAN).map(|value| (value, index + 3)),
            _ if rest.starts_with(b"Infinity") => self.constant("Infinity", f64::INFINITY).map(|value| (value, index + 8)),
            _ if rest.starts_with(b"-Infinity") => self.constant("-Infinity", f64::NEG_INFINITY).map(|value| (value, index + 9)),
            _ => self.number(index, depth),
        }
    }

    fn number(&self, start: usize, depth: usize) -> Scan<(Json, usize)> {
        let digit = |index: usize| self.at(index).is_some_and(|byte| byte.is_ascii_digit());
        let mut index = start;
        if self.at(index) == Some(b'-') { index += 1; }
        match self.at(index) {
            Some(b'1'..=b'9') => { index += 1; while digit(index) { index += 1; } }
            Some(b'0') => index += 1,
            _ => return Err(Failure::Stop(start)),
        }
        let mut real = false;
        if self.at(index) == Some(b'.') && digit(index + 1) {
            real = true;
            index += 2;
            while digit(index) { index += 1; }
        }
        if matches!(self.at(index), Some(b'e' | b'E')) && index + 1 < self.bytes.len() {
            let mark = index;
            index += 1;
            if matches!(self.at(index), Some(b'-' | b'+')) && index + 1 < self.bytes.len() { index += 1; }
            while digit(index) { index += 1; }
            if digit(index - 1) { real = true; } else { index = mark; }
        }
        let text = &self.text[start..index];
        if real {
            let value: f64 = text.parse().expect("scanned number");
            if self.mode == Mode::Figure {
                if depth > MAX_FLOAT_HOOK_DEPTH + 1 { return Err(recursion("maximum recursion depth exceeded")); }
                if depth > MAX_FLOAT_HOOK_DEPTH { return Err(recursion("maximum recursion depth exceeded while calling a Python object")); }
                if !value.is_finite() { return Err(Failure::Error(ArchiveError::value(format!("figure contains non-finite metadata number {text}")))); }
            }
            return Ok((Json::Float(value), index));
        }
        let digits = text.trim_start_matches('-').len();
        if digits > 4300 {
            return Err(Failure::Error(ArchiveError::value(format!(
                "Exceeds the limit (4300 digits) for integer string conversion: value has {digits} digits; use sys.set_int_max_str_digits() to increase the limit"))));
        }
        Ok((text.parse::<i128>().map_or_else(|_| Json::BigInt(text.to_owned()), Json::Int), index))
    }

    fn string(&self, start: usize) -> Scan<(String, usize)> {
        let begin = start - 1;
        let bytes = self.bytes;
        let mut out = String::new();
        let mut end = start;
        loop {
            let mut next = end;
            while next < bytes.len() && bytes[next] != b'"' && bytes[next] != b'\\' {
                if bytes[next] <= 0x1f { return Err(self.error("Invalid control character at", next)); }
                next += 1;
            }
            if next >= bytes.len() { return Err(self.error("Unterminated string starting at", begin)); }
            out.push_str(&self.text[end..next]);
            if bytes[next] == b'"' { return Ok((out, next + 1)); }
            next += 1;
            if next == bytes.len() { return Err(self.error("Unterminated string starting at", begin)); }
            let escape = bytes[next];
            if escape != b'u' {
                end = next + 1;
                out.push(match escape {
                    b'"' => '"', b'\\' => '\\', b'/' => '/', b'b' => '\u{8}', b'f' => '\u{c}', b'n' => '\n', b'r' => '\r', b't' => '\t',
                    _ => return Err(self.error("Invalid \\escape", end - 2)),
                });
                continue;
            }
            next += 1;
            end = next + 4;
            if end >= bytes.len() { return Err(self.error("Invalid \\uXXXX escape", next - 1)); }
            let hex = |from: usize, to: usize| -> Option<u32> {
                let text = std::str::from_utf8(&bytes[from..to]).ok()?;
                if text.bytes().all(|byte| byte.is_ascii_hexdigit()) { u32::from_str_radix(text, 16).ok() } else { None }
            };
            let mut code = hex(next, end).ok_or_else(|| self.error("Invalid \\uXXXX escape", end - 5))?;
            next = end;
            if (0xd800..=0xdbff).contains(&code) && end + 6 < bytes.len() && bytes[next] == b'\\' && bytes[next + 1] == b'u' {
                let low = hex(next + 2, end + 6).ok_or_else(|| self.error("Invalid \\uXXXX escape", end + 1))?;
                if (0xdc00..=0xdfff).contains(&low) {
                    code = 0x10000 + (((code - 0xd800) << 10) | (low - 0xdc00));
                    end += 6;
                }
            }
            pystr::push(&mut out, code);
        }
    }

    /// `object_pairs_hook` of the figure reader (`_metadata_pairs`), or `dict(pairs)`.
    fn finish_object(&self, pairs: Vec<(String, Json)>, depth: usize) -> Scan<Json> {
        if self.mode == Mode::Plain {
            let mut merged: Vec<(String, Json)> = Vec::with_capacity(pairs.len());
            let mut positions: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
            for (key, value) in pairs {
                match positions.get(&key) { Some(at) => merged[*at].1 = value, None => { positions.insert(key.clone(), merged.len()); merged.push((key, value)); } }
            }
            return Ok(Json::Object(merged));
        }
        if depth > MAX_HOOK_OBJECT_DEPTH { return Err(recursion("maximum recursion depth exceeded")); }
        let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::with_capacity(pairs.len());
        for (key, _) in &pairs {
            if !seen.insert(key.as_str()) {
                return Err(Failure::Error(ArchiveError::value(format!("figure contains duplicate metadata key {}", py_repr_str(key)))));
            }
        }
        Ok(Json::Object(pairs))
    }

    /// The key after `{` or `,` at `index`, then its `:`: where its value starts.
    fn key(&self, index: usize) -> Scan<(String, usize)> {
        if self.at(index) != Some(b'"') { return Err(self.error("Expecting property name enclosed in double quotes", index)); }
        let (key, end) = self.string(index + 1)?;
        let index = self.skip(end);
        if self.at(index) != Some(b':') { return Err(self.error("Expecting ':' delimiter", index)); }
        Ok((key, self.skip(index + 1)))
    }

    /// `scan_once(index)` with the containers on a heap stack.
    fn document(&self, mut index: usize) -> Scan<(Json, usize)> {
        let mut stack: Vec<Frame> = Vec::new();
        'value: loop {
            // A value starts at `index`.
            let (mut value, mut end) = match self.at(index) {
                Some(open @ (b'{' | b'[')) => {
                    let object = open == b'{';
                    if stack.len() + 1 > MAX_CONTAINER_DEPTH {
                        return Err(recursion(if object { "maximum recursion depth exceeded while decoding a JSON object from a unicode string" }
                                             else { "maximum recursion depth exceeded while decoding a JSON array from a unicode string" }));
                    }
                    let first = self.skip(index + 1);
                    let close = if object { b'}' } else { b']' };
                    if self.at(first) == Some(close) {
                        let empty = if object { self.finish_object(Vec::new(), stack.len() + 1)? } else { Json::Array(Vec::new()) };
                        (empty, first + 1)
                    } else if object {
                        let (key, start) = self.key(first)?;
                        stack.push(Frame::Object(Vec::new(), key));
                        index = start;
                        continue 'value;
                    } else {
                        stack.push(Frame::Array(Vec::new()));
                        index = first;
                        continue 'value;
                    }
                }
                _ => match self.scalar(index, stack.len()) {
                    Ok(scalar) => scalar,
                    Err(Failure::Stop(at)) if !stack.is_empty() => return Err(self.error("Expecting value", at)),
                    Err(failure) => return Err(failure),
                },
            };
            // Deliver `value` (ending at `end`) to the containers it closes.
            loop {
                let depth = stack.len();
                let Some(frame) = stack.last_mut() else { return Ok((value, end)) };
                let at = self.skip(end);
                match frame {
                    Frame::Array(items) => {
                        items.push(value);
                        match self.at(at) {
                            Some(b']') => {
                                let Some(Frame::Array(items)) = stack.pop() else { unreachable!() };
                                value = Json::Array(items);
                                end = at + 1;
                            }
                            Some(b',') => {
                                let next = self.skip(at + 1);
                                if self.at(next) == Some(b']') { return Err(self.error("Illegal trailing comma before end of array", at)); }
                                index = next;
                                continue 'value;
                            }
                            _ => return Err(self.error("Expecting ',' delimiter", at)),
                        }
                    }
                    Frame::Object(pairs, key) => {
                        pairs.push((std::mem::take(key), value));
                        match self.at(at) {
                            Some(b'}') => {
                                let Some(Frame::Object(pairs, _)) = stack.pop() else { unreachable!() };
                                value = self.finish_object(pairs, depth)?;
                                end = at + 1;
                            }
                            Some(b',') => {
                                let next = self.skip(at + 1);
                                if self.at(next) == Some(b'}') { return Err(self.error("Illegal trailing comma before end of object", at)); }
                                let (new_key, start) = self.key(next)?;
                                *key = new_key;
                                index = start;
                                continue 'value;
                            }
                            _ => return Err(self.error("Expecting ',' delimiter", at)),
                        }
                    }
                }
            }
        }
    }

    /// `json.loads(text)`: one document, then only whitespace.
    fn loads(&self) -> Scan<Json> {
        if self.text.starts_with('\u{feff}') { return Err(self.error("Unexpected UTF-8 BOM (decode using utf-8-sig)", 0)); }
        let start = self.skip(0);
        let (value, end) = match self.document(start) {
            Err(Failure::Stop(at)) => return Err(self.error("Expecting value", at)),
            other => other?,
        };
        let end = self.skip(end);
        if end != self.text.len() { return Err(self.error("Extra data", end)); }
        Ok(value)
    }
}

/// `json.loads(text, object_pairs_hook=<refuse duplicates>,
/// parse_constant=<refuse>, parse_float=<finite>)`; a decode failure reads
/// `invalid figure metadata JSON: ...`, a hook's refusal is its own.
pub fn parse_metadata(text: &str) -> Result<Json> {
    let parser = Parser { text, bytes: text.as_bytes(), mode: Mode::Figure };
    parser.loads().map_err(|failure| match failure {
        Failure::Decode(message) => ArchiveError::value(format!("invalid figure metadata JSON: {message}")),
        Failure::Error(error) => error,
        Failure::Stop(_) => unreachable!("mapped to Expecting value"),
    })
}

/// Plain `json.loads(text)`: duplicate keys keep the last value, `NaN` and
/// friends are floats, ints are exact.
pub fn loads(text: &str) -> Result<Json> {
    let parser = Parser { text, bytes: text.as_bytes(), mode: Mode::Plain };
    parser.loads().map_err(|failure| match failure {
        Failure::Decode(message) => ArchiveError::value(message),
        Failure::Error(error) => error,
        Failure::Stop(_) => unreachable!("mapped to Expecting value"),
    })
}

// ---------------------------------------------------------------- dumping

/// `json.dumps` options this code base uses.
#[derive(Clone, Copy)]
pub enum Style {
    /// `separators=(",", ":"), sort_keys=True, ensure_ascii=False` (the figure info).
    Figure,
    /// `json.dumps(value)`: `", "`, `": "`, document order, ASCII escapes.
    Default,
}

/// Archive text in Python code-point order (`sort_keys`).
fn key_order(a: &str, b: &str) -> std::cmp::Ordering { pystr::code_points(a).cmp(pystr::code_points(b)) }

pub fn dumps(value: &Json, style: Style) -> Result<String> {
    enum Step<'a> { Value(&'a Json), Text(&'static str), Owned(String) }
    let (item_separator, key_separator) = match style { Style::Figure => (",", ":"), Style::Default => (", ", ": ") };
    let mut out = String::new();
    let mut pending = vec![Step::Value(value)];
    while let Some(step) = pending.pop() {
        let value = match step { Step::Text(text) => { out.push_str(text); continue; } Step::Owned(text) => { out.push_str(&text); continue; } Step::Value(value) => value };
        match value {
            Json::Null => out.push_str("null"),
            Json::Bool(flag) => out.push_str(if *flag { "true" } else { "false" }),
            Json::Int(integer) => out.push_str(&integer.to_string()),
            Json::BigInt(text) => out.push_str(text),
            Json::Float(real) => {
                if !real.is_finite() { return Err(ArchiveError::value(format!("Out of range float values are not JSON compliant: {}", pyfmt::repr(*real)))); }
                out.push_str(&pyfmt::repr(*real));
            }
            Json::Str(text) => write_string(text, style, &mut out),
            Json::Array(items) => {
                out.push('[');
                pending.push(Step::Text("]"));
                for (index, item) in items.iter().enumerate().rev() {
                    pending.push(Step::Value(item));
                    if index > 0 { pending.push(Step::Text(item_separator)); }
                }
            }
            Json::Object(pairs) => {
                let mut order: Vec<&(String, Json)> = pairs.iter().collect();
                if let Style::Figure = style { order.sort_by(|a, b| key_order(&a.0, &b.0)); }
                out.push('{');
                pending.push(Step::Text("}"));
                for (index, (key, item)) in order.into_iter().enumerate().rev() {
                    pending.push(Step::Value(item));
                    pending.push(Step::Text(key_separator));
                    let mut quoted = String::new();
                    write_string(key, style, &mut quoted);
                    pending.push(Step::Owned(quoted));
                    if index > 0 { pending.push(Step::Text(item_separator)); }
                }
            }
        }
    }
    Ok(out)
}

fn write_string(text: &str, style: Style, out: &mut String) {
    out.push('"');
    for code in pystr::code_points(text) {
        match code {
            0x22 => out.push_str("\\\""),
            0x5C => out.push_str("\\\\"),
            0x0A => out.push_str("\\n"),
            0x0D => out.push_str("\\r"),
            0x09 => out.push_str("\\t"),
            0x08 => out.push_str("\\b"),
            0x0C => out.push_str("\\f"),
            c if c < 0x20 => out.push_str(&format!("\\u{c:04x}")),
            c if matches!(style, Style::Default) && !(0x20..=0x7e).contains(&c) => {
                if c >= 0x10000 {
                    let v = c - 0x10000;
                    out.push_str(&format!("\\u{:04x}\\u{:04x}", 0xD800 + (v >> 10), 0xDC00 + (v & 0x3FF)));
                } else {
                    out.push_str(&format!("\\u{c:04x}"));
                }
            }
            c => pystr::push(out, c),
        }
    }
    out.push('"');
}

// ---------------------------------------------------------------- repr

fn printable(code: u32) -> bool {
    match code {
        0x20..=0x7e => true,
        0..=0x1f | 0x7f..=0xa0 | 0xad => false,
        0x2028 | 0x2029 | 0x200b..=0x200f | 0x202a..=0x202e | 0x2060..=0x206f | 0xfeff | 0xd800..=0xf8ff | 0xfff9..=0xfffb => false,
        0x1680 | 0x2000..=0x200a | 0x202f | 0x205f | 0x3000 => false,
        0xf0000.. => false,
        _ => true,
    }
}

/// Python `repr(str)` of archive text.
pub fn py_repr_str(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') { '"' } else { '\'' };
    let mut out = String::from(quote);
    for code in pystr::code_points(text) {
        match code {
            0x5C => out.push_str("\\\\"),
            0x0A => out.push_str("\\n"),
            0x0D => out.push_str("\\r"),
            0x09 => out.push_str("\\t"),
            c if c == quote as u32 => { out.push('\\'); out.push(quote); }
            c if printable(c) => out.push(char::from_u32(c).unwrap()),
            c if c < 0x100 => out.push_str(&format!("\\x{c:02x}")),
            c if c < 0x10000 => out.push_str(&format!("\\u{c:04x}")),
            c => out.push_str(&format!("\\U{c:08x}")),
        }
    }
    out.push(quote);
    out
}

/// Python `repr` of the value `json.loads` returned.
pub fn py_repr(value: &Json) -> String {
    match value {
        Json::Null => "None".into(),
        Json::Bool(flag) => if *flag { "True" } else { "False" }.into(),
        Json::Int(integer) => integer.to_string(),
        Json::BigInt(text) => text.clone(),
        Json::Float(real) => pyfmt::repr(*real),
        Json::Str(text) => py_repr_str(text),
        Json::Array(items) => format!("[{}]", items.iter().map(py_repr).collect::<Vec<_>>().join(", ")),
        Json::Object(pairs) => format!("{{{}}}", pairs.iter().map(|(key, value)| format!("{}: {}", py_repr_str(key), py_repr(value))).collect::<Vec<_>>().join(", ")),
    }
}

/// `repr(sorted(names))` of text.
pub fn py_repr_names<S: AsRef<str>>(names: &[S]) -> String {
    let mut sorted: Vec<&str> = names.iter().map(AsRef::as_ref).collect();
    sorted.sort_by(|a, b| key_order(a, b));
    format!("[{}]", sorted.iter().map(|name| py_repr_str(name)).collect::<Vec<_>>().join(", "))
}

/// Python `repr(bytes)`.
pub fn py_repr_bytes(bytes: &[u8]) -> String {
    let quote = if bytes.contains(&b'\'') && !bytes.contains(&b'"') { b'"' } else { b'\'' };
    let mut out = String::from("b");
    out.push(quote as char);
    for byte in bytes {
        match *byte {
            b'\\' => out.push_str("\\\\"),
            b'\n' => out.push_str("\\n"),
            b'\r' => out.push_str("\\r"),
            b'\t' => out.push_str("\\t"),
            b if b == quote => { out.push('\\'); out.push(b as char); }
            0x20..=0x7e => out.push(*byte as char),
            b => out.push_str(&format!("\\x{b:02x}")),
        }
    }
    out.push(quote as char);
    out
}
