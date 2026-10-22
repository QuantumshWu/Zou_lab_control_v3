//! NumPy 2.4's dtypes as an archive meets them (Windows, little-endian):
//! `numpy.dtype(text)` for a descriptor string (typecodes, kind and size,
//! `sctypeDict` names, datetimes, comma strings), `descr_to_dtype` for an
//! NPY header's `descr`, and `.str` of the result. Refusals carry NumPy's
//! exception class and message.

use zlc_core::DType;

use crate::error::{ArchiveError, ErrorKind, Result};
use crate::pyliteral::{literal_eval, EvalError, Int, Obj};

/// The element kind of one dtype.
#[derive(Clone, Debug, PartialEq)]
pub enum Kind {
    Numeric(DType),
    /// `U`: UTF-32 code units.
    Unicode,
    /// `S`: raw bytes.
    Bytes,
    /// `V`: raw bytes, structured when `fields`.
    Void { fields: bool },
    /// `M8`/`m8` with its `[unit]` text (empty for generic).
    DateTime { code: char, unit: String },
    /// `O`, and `T` (StringDType): they hold objects.
    Object,
}

/// One dtype: kind, byte order and itemsize, or a subarray of a base.
#[derive(Clone, Debug, PartialEq)]
pub struct Dtype { pub kind: Kind, pub itemsize: usize, pub swapped: bool, pub subarray: Option<(Box<Dtype>, Vec<usize>)> }

impl Dtype {
    pub fn numeric(dtype: DType, swapped: bool) -> Self { Dtype { kind: Kind::Numeric(dtype), itemsize: numeric_size(dtype), swapped, subarray: None } }
    pub fn unicode(chars: usize, swapped: bool) -> Self { Dtype { kind: Kind::Unicode, itemsize: chars * 4, swapped, subarray: None } }
    pub fn bytes(len: usize) -> Self { Dtype { kind: Kind::Bytes, itemsize: len, swapped: false, subarray: None } }
    fn void(len: usize, fields: bool) -> Self { Dtype { kind: Kind::Void { fields }, itemsize: len, swapped: false, subarray: None } }

    pub fn has_object(&self) -> bool { self.kind == Kind::Object || self.subarray.as_ref().is_some_and(|(base, _)| base.has_object()) }
    pub fn has_fields(&self) -> bool { matches!(self.kind, Kind::Void { fields: true }) }

    /// Whether a byte order applies (`byteorder != '|'`).
    fn ordered(&self) -> bool {
        match &self.kind {
            Kind::Numeric(dtype) => numeric_size(*dtype) > 1,
            Kind::Unicode | Kind::DateTime { .. } => true,
            _ => false,
        }
    }

    /// `dtype.str`.
    pub fn str(&self) -> String {
        if self.subarray.is_some() || self.has_fields() { return format!("|V{}", self.itemsize); }
        let order = if !self.ordered() { '|' } else if self.swapped { '>' } else { '<' };
        match &self.kind {
            Kind::Numeric(dtype) => format!("{order}{}{}", numeric_kind(*dtype), self.itemsize),
            Kind::Unicode => format!("{order}U{}", self.itemsize / 4),
            Kind::Bytes => format!("|S{}", self.itemsize),
            Kind::Void { .. } => format!("|V{}", self.itemsize),
            Kind::DateTime { code, unit } => format!("{order}{code}8{unit}"),
            Kind::Object => "|O".into(),
        }
    }
}

pub fn numeric_size(dtype: DType) -> usize {
    match dtype {
        DType::Bool | DType::U8 | DType::I8 => 1, DType::U16 | DType::I16 | DType::F16 => 2,
        DType::U32 | DType::I32 | DType::F32 => 4, DType::U64 | DType::I64 | DType::F64 | DType::C64 => 8, DType::C128 => 16,
    }
}

fn numeric_kind(dtype: DType) -> char {
    match dtype {
        DType::Bool => 'b', DType::U8 | DType::U16 | DType::U32 | DType::U64 => 'u', DType::I8 | DType::I16 | DType::I32 | DType::I64 => 'i',
        DType::F16 | DType::F32 | DType::F64 => 'f', DType::C64 | DType::C128 => 'c',
    }
}

fn not_understood(text: &str) -> ArchiveError { ArchiveError::type_error(format!("data type {} not understood", crate::json::py_repr_str(text))) }

/// A single-character typecode (`PyArray_DescrFromType`).
fn typecode(c: char) -> Option<Dtype> {
    use DType::*;
    Some(match c {
        '?' => Dtype::numeric(Bool, false), 'b' => Dtype::numeric(I8, false), 'B' => Dtype::numeric(U8, false), 'h' => Dtype::numeric(I16, false),
        'H' => Dtype::numeric(U16, false), 'i' | 'l' => Dtype::numeric(I32, false), 'I' | 'L' => Dtype::numeric(U32, false),
        'q' | 'n' | 'p' => Dtype::numeric(I64, false), 'Q' | 'N' | 'P' => Dtype::numeric(U64, false), 'e' => Dtype::numeric(F16, false),
        'f' => Dtype::numeric(F32, false), 'd' | 'g' => Dtype::numeric(F64, false), 'F' => Dtype::numeric(C64, false),
        'D' | 'G' => Dtype::numeric(C128, false), 'S' | 'a' => Dtype::bytes(0), 'c' => Dtype::bytes(1), 'U' => Dtype::unicode(0, false), 'V' => Dtype::void(0, false),
        'M' => Dtype { kind: Kind::DateTime { code: 'M', unit: String::new() }, itemsize: 8, swapped: false, subarray: None },
        'm' => Dtype { kind: Kind::DateTime { code: 'm', unit: String::new() }, itemsize: 8, swapped: false, subarray: None },
        'O' => Dtype { kind: Kind::Object, itemsize: 8, swapped: false, subarray: None },
        'T' => Dtype { kind: Kind::Object, itemsize: 16, swapped: false, subarray: None },
        _ => return None,
    })
}

/// `PyArray_TypestrConvert(itemsize, kind)`.
fn typestr_convert(size: usize, kind: char) -> Option<Dtype> {
    use DType::*;
    let numeric = |dtype: DType| Some(Dtype::numeric(dtype, false));
    match (kind, size) {
        ('b', 1) => numeric(Bool),
        ('u', 1) => numeric(U8), ('u', 2) => numeric(U16), ('u', 4) => numeric(U32), ('u', 8) => numeric(U64),
        ('i', 1) => numeric(I8), ('i', 2) => numeric(I16), ('i', 4) => numeric(I32), ('i', 8) => numeric(I64),
        ('f', 2) => numeric(F16), ('f', 4) => numeric(F32), ('f', 8) => numeric(F64),
        ('c', 8) => numeric(C64), ('c', 16) => numeric(C128),
        ('O', 4 | 8) => typecode('O'),
        ('M', 8) => typecode('M'), ('m', 8) => typecode('m'),
        _ => None,
    }
}

/// `np.sctypeDict[name]` (keys are full strings, prefix included).
fn sctype(name: &str) -> Option<Dtype> {
    use DType::*;
    let numeric = |dtype: DType| Some(Dtype::numeric(dtype, false));
    match name {
        "a" | "bytes" | "bytes_" => Some(Dtype::bytes(0)),
        "bool" | "bool_" => numeric(Bool),
        "byte" | "int8" => numeric(I8), "ubyte" | "uint8" => numeric(U8),
        "short" | "int16" => numeric(I16), "ushort" | "uint16" => numeric(U16),
        "intc" | "int32" | "long" => numeric(I32), "uintc" | "uint32" | "ulong" => numeric(U32),
        "int" | "int_" | "intp" | "int64" | "longlong" => numeric(I64), "uint" | "uintp" | "uint64" | "ulonglong" => numeric(U64),
        "half" | "float16" => numeric(F16), "single" | "float32" => numeric(F32),
        "double" | "float" | "float64" | "longdouble" => numeric(F64),
        "csingle" | "complex64" => numeric(C64), "cdouble" | "complex" | "complex128" | "clongdouble" => numeric(C128),
        "datetime64" => typecode('M'), "timedelta64" => typecode('m'),
        "object" | "object_" => typecode('O'),
        "str" | "str_" | "unicode" => Some(Dtype::unicode(0, false)),
        "void" => Some(Dtype::void(0, false)),
        _ => None,
    }
}

/// C `strtol(text, &end, 10)` with a 32-bit `long`: value, characters
/// consumed (0 when nothing parsed) and whether it overflowed.
fn strtol(text: &str) -> (i64, usize, bool) {
    let bytes = text.as_bytes();
    let mut at = 0;
    while at < bytes.len() && matches!(bytes[at], b' ' | b'\t' | b'\n' | b'\x0b' | b'\x0c' | b'\r') { at += 1; }
    let negative = matches!(bytes.get(at), Some(b'-'));
    if matches!(bytes.get(at), Some(b'+' | b'-')) { at += 1; }
    let digits_start = at;
    let mut value: i64 = 0;
    let mut overflow = false;
    while at < bytes.len() && bytes[at].is_ascii_digit() {
        value = value.saturating_mul(10).saturating_add(i64::from(bytes[at] - b'0'));
        if value > i64::from(i32::MAX) + 1 { overflow = true; }
        at += 1;
    }
    if at == digits_start { return (0, 0, false); }
    let signed = if negative { -value } else { value };
    if signed > i64::from(i32::MAX) { return (i64::from(i32::MAX), at, true); }
    if signed < i64::from(i32::MIN) { return (i64::from(i32::MIN), at, true); }
    (signed, at, overflow && signed != i64::from(i32::MIN))
}

const UNITS: [&str; 15] = ["Y", "M", "W", "B", "D", "h", "m", "s", "ms", "us", "ns", "ps", "fs", "as", "generic"];

/// `parse_datetime_metadata_from_metastr`: the `[unit]` part of a datetime.
fn datetime_unit(meta: &str) -> Result<String> {
    if meta.is_empty() { return Ok(String::new()); }
    let bad = |at: usize| ArchiveError::type_error(if at > 0 {
        format!("Invalid datetime metadata string \"{meta}\" at position {at}")
    } else {
        format!("Invalid datetime metadata string \"{meta}\"")
    });
    let chars: Vec<char> = meta.chars().collect();
    if chars[0] != '[' { return Err(bad(0)); }
    let Some(close) = chars.iter().position(|c| *c == ']') else { return Err(bad(0)) };
    if close == 1 { return Err(bad(0)); }
    if close + 1 != chars.len() { return Err(bad(close + 1)); }
    let inner: String = chars[1..close].iter().collect();
    // An optional multiplier, the unit, an optional `/divisor`.
    let (number, used, _) = strtol(&inner);
    let mut num: i64 = 1;
    if used > 0 {
        let digits: String = inner.trim_start().chars().take_while(|c| c.is_ascii_digit() || *c == '-' || *c == '+').collect();
        let wide: i128 = digits.parse().unwrap_or(-1);
        if wide > i128::from(i32::MAX) || wide < 0 { return Err(bad(1)); }
        num = number;
    }
    let rest = &inner[used..];
    let unit_end = rest.find('/').unwrap_or(rest.len());
    let unit_text = &rest[..unit_end];
    if unit_text.is_empty() { return Err(bad(1 + used)); }
    let unit_text = if unit_text == "\u{3bc}s" { "us" } else { unit_text };
    let Some(mut base) = UNITS.iter().position(|unit| *unit == unit_text && *unit != "B") else {
        return Err(ArchiveError::type_error(format!("Invalid datetime unit in metadata string \"{meta}\"")));
    };
    if unit_end < rest.len() {
        let denominator_text = &rest[unit_end + 1..];
        let (den, used_den, _) = strtol(denominator_text);
        if used_den == 0 || used_den != denominator_text.len() { return Err(bad(1 + used + unit_end + 1 + used_den)); }
        if den != 1 {
            if base == 14 { return Err(ArchiveError::value("Can't use 'den' divisor with generic units")); }
            const TABLE: [[i64; 4]; 16] = [[12, 52, 365, 0], [1, 2, 4, 0], [4, 30, 720, 0], [2, 4, 5, 0], [7, 168, 10080, 0], [4, 5, 6, 0], [0; 4], [0; 4],
                                           [24, 1440, 86400, 0], [5, 6, 7, 0], [60, 3600, 0, 0], [6, 7, 0, 0], [60, 60000, 0, 0], [7, 8, 0, 0], [1000, 1_000_000, 0, 0], [0; 4]];
            let mut count = if base == 2 { 4 } else if base > 4 { 2 } else { 3 };
            let (tries, units) = if base >= 7 {
                if base == 12 { count = 1; }
                if base == 13 { count = 0; }
                (TABLE[14], [base as i64 + 1, base as i64 + 2, 0, 0])
            } else {
                (TABLE[base * 2], TABLE[base * 2 + 1])
            };
            let found = (0..count).find(|index| den != 0 && tries[*index] % den == 0);
            let Some(index) = found else {
                return Err(ArchiveError::value(format!("divisor ({den}) is not a multiple of a lower-unit in datetime metadata \"{meta}\"")));
            };
            num *= tries[index] / den;
            base = units[index] as usize;
        }
    }
    if base == 14 { return Ok(String::new()); }
    Ok(if num == 1 { format!("[{}]", UNITS[base]) } else { format!("[{num}{}]", UNITS[base]) })
}

/// `_check_for_commastring`.
fn is_commastring(text: &str) -> bool {
    let b = text.as_bytes();
    let order = |c: u8| matches!(c, b'>' | b'<' | b'|' | b'=');
    if b[0].is_ascii_digit() || (b.len() > 1 && order(b[0]) && b[1].is_ascii_digit()) { return true; }
    if (b.len() > 1 && b[0] == b'(' && b[1] == b')') || (b.len() > 3 && order(b[0]) && b[1] == b'(' && b[2] == b')') { return true; }
    let mut brackets = 0i32;
    for c in b {
        match c { b',' if brackets == 0 => return true, b'[' => brackets += 1, b']' => brackets -= 1, _ => {} }
    }
    false
}

/// One item `_commastring` makes of a comma string.
enum CommaItem { Plain(String), Repeated(String, Obj) }

/// `numpy._core._internal._commastring`: its regular expressions by hand.
fn commastring(text: &str) -> Result<(Vec<CommaItem>, bool)> {
    let chars: Vec<char> = text.chars().collect();
    let mut at = 0usize;
    let mut items = Vec::new();
    let mut is_list = false;
    let order = |c: Option<&char>| matches!(c, Some('<' | '>' | '|' | '='));
    while at < chars.len() {
        let order1 = if order(chars.get(at)) { at += 1; chars[at - 1].to_string() } else { String::new() };
        let repeats_start = at;
        while chars.get(at) == Some(&' ') { at += 1; }
        if chars.get(at) == Some(&'(') { at += 1; }
        while chars.get(at).is_some_and(|c| *c == ' ' || *c == ',' || c.is_ascii_digit()) { at += 1; }
        if chars.get(at) == Some(&')') { at += 1; }
        while chars.get(at) == Some(&' ') { at += 1; }
        let repeats: String = chars[repeats_start..at].iter().collect();
        let order2 = if order(chars.get(at)) { at += 1; chars[at - 1].to_string() } else { String::new() };
        let dtype_start = at;
        while chars.get(at).is_some_and(|c| c.is_ascii_alphanumeric() || *c == '.' || *c == '?') { at += 1; }
        if chars.get(at) == Some(&'[') {
            let mut end = at + 1;
            while chars.get(end).is_some_and(|c| c.is_ascii_alphanumeric() || *c == ',' || *c == '.') { end += 1; }
            if end > at + 1 && chars.get(end) == Some(&']') { at = end + 1; }
        }
        let dtype: String = chars[dtype_start..at].iter().collect();
        if at < chars.len() {
            if chars[at..].iter().all(|c| c.is_whitespace()) {
                at = chars.len();
            } else {
                let mut next = at;
                while chars.get(next).is_some_and(|c| c.is_whitespace()) { next += 1; }
                if chars.get(next) != Some(&',') {
                    return Err(ArchiveError::value(format!("format number {} of \"{text}\" is not recognized", items.len() + 1)));
                }
                next += 1;
                while chars.get(next).is_some_and(|c| c.is_whitespace()) { next += 1; }
                at = next;
                is_list = true;
            }
        }
        let byte_order = if order2.is_empty() { order1 } else if order1.is_empty() { order2 } else {
            let native = |o: String| if o == "=" { "<".to_owned() } else { o };
            let (a, b) = (native(order1), native(order2));
            if a != b { return Err(ArchiveError::value(format!("inconsistent byte-order specification {a} and {b}"))); }
            a
        };
        let byte_order = if matches!(byte_order.as_str(), "|" | "=" | "<") { String::new() } else { byte_order };
        let dtype = format!("{byte_order}{dtype}");
        if repeats.is_empty() {
            items.push(CommaItem::Plain(dtype));
        } else {
            let shape = literal_eval(&repeats).map_err(|error| match error {
                EvalError::Syntax => ArchiveError::new(ErrorKind::Syntax, "invalid syntax (<unknown>, line 1)"),
                EvalError::Malformed(message) => ArchiveError::value(message),
                EvalError::Type(message) => ArchiveError::type_error(message),
                EvalError::Recursion => ArchiveError::new(ErrorKind::Recursion, "maximum recursion depth exceeded during ast construction"),
            })?;
            items.push(CommaItem::Repeated(dtype, shape));
        }
        if at == chars.len() { break; }
    }
    Ok((items, is_list))
}

/// `_convert_from_tuple((base, value))`.
fn from_tuple(base: Dtype, value: &Obj) -> Result<Dtype> {
    let flexible = matches!(base.kind, Kind::Bytes | Kind::Unicode | Kind::Void { fields: false }) && base.itemsize == 0 && base.subarray.is_none();
    if flexible {
        let size = match value {
            Obj::Int(Int::Small(size)) if (i128::from(i32::MIN)..=i128::from(i32::MAX)).contains(size) => *size,
            Obj::Bool(flag) => i128::from(*flag),
            _ => return Err(ArchiveError::value("invalid itemsize in generic type tuple")),
        };
        let size = usize::try_from(size).map_err(|_| ArchiveError::value("invalid itemsize in generic type tuple"))?;
        let mut sized = base;
        sized.itemsize = if sized.kind == Kind::Unicode { size << 2 } else { size };
        return Ok(sized);
    }
    let invalid = || ArchiveError::value("invalid shape in fixed-type tuple.");
    let dimension = |item: &Obj| -> Result<i128> {
        match item { Obj::Int(Int::Small(size)) => Ok(*size), Obj::Int(_) => Err(invalid()), _ => Err(invalid()) }
    };
    let shape: Vec<i128> = match value {
        Obj::Tuple(items) => items.iter().map(dimension).collect::<Result<_>>()?,
        Obj::Int(_) => vec![dimension(value)?],
        _ => return Err(invalid()),
    };
    if shape.len() > 64 { return Err(invalid()); }
    if shape.is_empty() && matches!(value, Obj::Tuple(_)) { return Ok(base); }
    if shape.iter().any(|size| *size < 0) { return Err(ArchiveError::value("invalid shape in fixed-type tuple: dimension smaller then zero.")); }
    if shape.iter().any(|size| *size > i128::from(i32::MAX)) { return Err(ArchiveError::value("invalid shape in fixed-type tuple: dimension does not fit into a C int.")); }
    let items = shape.iter().try_fold(1i128, |total, size| total.checked_mul(*size)).filter(|items| *items <= i128::from(i32::MAX));
    let bytes = items.and_then(|items| items.checked_mul(base.itemsize as i128)).filter(|bytes| *bytes <= i128::from(i32::MAX))
        .ok_or_else(|| ArchiveError::value("invalid shape in fixed-type tuple: dtype size in bytes must fit into a C int."))?;
    Ok(Dtype { kind: Kind::Void { fields: false }, itemsize: bytes as usize, swapped: false, subarray: Some((Box::new(base), shape.into_iter().map(|size| size as usize).collect())) })
}

/// `numpy.dtype(text)` for a string.
pub fn from_str(text: &str) -> Result<Dtype> {
    if text.is_empty() { return Err(not_understood(text)); }
    if is_commastring(text) {
        let (items, is_list) = commastring(text)?;
        let convert = |item: &CommaItem| -> Result<Dtype> {
            match item { CommaItem::Plain(dtype) => from_str(dtype), CommaItem::Repeated(dtype, shape) => from_tuple(from_str(dtype)?, shape) }
        };
        if !is_list || items.len() == 1 { return convert(&items[0]); }
        let mut size = 0usize;
        for item in &items { size += convert(item)?.itemsize; }
        return Ok(Dtype::void(size, true));
    }
    let (endian, rest) = match text.chars().next().unwrap() { '<' | '>' | '=' | '|' => (text.chars().next().unwrap(), &text[1..]), _ => ('=', text) };
    if rest.is_empty() { return Err(not_understood(text)); }
    let rest_chars: Vec<char> = rest.chars().collect();
    let datetime = (rest_chars.len() >= 2 && rest_chars[1] == '8' && matches!(rest_chars[0], 'M' | 'm'))
        || rest.starts_with("datetime64") || rest.starts_with("timedelta64");
    if datetime {
        let (code, meta) = if rest_chars[1] == '8' && matches!(rest_chars[0], 'M' | 'm') { (rest_chars[0], &rest[2..]) }
            else if let Some(meta) = rest.strip_prefix("datetime64") { ('M', meta) } else { ('m', &rest["timedelta64".len()..]) };
        let unit = datetime_unit(meta)?;
        return Ok(Dtype { kind: Kind::DateTime { code, unit }, itemsize: 8, swapped: endian == '>', subarray: None });
    }
    let mut resolved: Option<Dtype> = None;
    let mut elsize: usize = 0;
    if rest_chars.len() == 1 {
        resolved = typecode(rest_chars[0]);
    } else {
        let kind = rest_chars[0];
        let tail = &rest[kind.len_utf8()..];
        let (value, used, overflow) = strtol(tail);
        if value > i64::from(i32::MAX) || value < 0 { return Err(not_understood(text)); }
        let parsed = used > 0 && !overflow && used == tail.len();
        if parsed {
            elsize = value as usize;
            resolved = match kind {
                'S' | 'a' => Some(Dtype::bytes(elsize)),
                'U' => {
                    if elsize > (i32::MAX as usize) / 4 { return Err(not_understood(text)); }
                    Some(Dtype::unicode(elsize, false))
                }
                'V' => Some(Dtype::void(elsize, false)),
                _ if elsize == 0 => None,
                _ => typestr_convert(elsize, kind),
            };
            if matches!(kind, 'S' | 'a' | 'U' | 'V') { elsize = 0; }
        }
    }
    let mut dtype = match resolved {
        Some(dtype) => dtype,
        None => {
            if let Some(dtype) = sctype(text) { return Ok(dtype); }
            if ["int0", "uint0", "void0", "object0", "str0", "bytes0", "bool8"].contains(&rest) {
                return Err(ArchiveError::type_error(format!("Alias {} was removed in NumPy 2.0. Use a name without a digit at the end.", crate::json::py_repr_str(text))));
            }
            return Err(not_understood(text));
        }
    };
    let _ = elsize;
    if endian == '>' && dtype.ordered() { dtype.swapped = true; }
    Ok(dtype)
}

fn index_error() -> ArchiveError { ArchiveError::new(ErrorKind::Index, "tuple index out of range") }

/// `descr_to_dtype(descr)`; a TypeError is the header's descr refusal.
pub fn descr_to_dtype(descr: &Obj) -> Result<Dtype> {
    match descr {
        Obj::Str(text) => from_str(text),
        Obj::Tuple(items) => {
            let base = descr_to_dtype(items.first().ok_or_else(index_error)?)?;
            from_tuple(base, items.get(1).ok_or_else(index_error)?)
        }
        _ => {
            let fields: Vec<Obj> = match descr {
                Obj::List(items) | Obj::Set(items) => items.clone(),
                Obj::Dict(pairs) => pairs.iter().map(|(key, _)| key.clone()).collect(),
                Obj::Bytes(bytes) => bytes.iter().map(|byte| Obj::Int(Int::Small(i128::from(*byte)))).collect(),
                other => return Err(ArchiveError::type_error(format!("'{}' object is not iterable", other.type_name()))),
            };
            let mut size = 0usize;
            let mut names: Vec<String> = Vec::new();
            for field in &fields {
                let parts: Vec<Obj> = match field {
                    Obj::Tuple(items) | Obj::List(items) => items.clone(),
                    Obj::Str(text) => text.chars().map(|c| Obj::Str(c.to_string())).collect(),
                    Obj::Bytes(bytes) => bytes.iter().map(|byte| Obj::Int(Int::Small(i128::from(*byte)))).collect(),
                    Obj::Dict(pairs) => pairs.iter().map(|(key, _)| key.clone()).collect(),
                    Obj::Set(items) => items.clone(),
                    other => return Err(ArchiveError::type_error(format!("object of type '{}' has no len()", other.type_name()))),
                };
                let (name, dtype) = match parts.len() {
                    2 => (parts[0].clone(), descr_to_dtype(&parts[1])?),
                    3 => (parts[0].clone(), from_tuple(descr_to_dtype(&parts[1])?, &parts[2])?),
                    count if count < 3 => return Err(ArchiveError::value(format!("not enough values to unpack (expected 3, got {count})"))),
                    _ => return Err(ArchiveError::value("too many values to unpack (expected 3)")),
                };
                let padding = matches!(&name, Obj::Str(text) if text.is_empty()) && matches!(dtype.kind, Kind::Void { fields: false }) && dtype.subarray.is_none();
                if !padding {
                    let name = match &name {
                        Obj::Tuple(pair) if pair.len() == 2 => pair[1].clone(),
                        Obj::Tuple(pair) => return Err(ArchiveError::value(if pair.len() > 2 { "too many values to unpack (expected 2)".to_owned() }
                            else { format!("not enough values to unpack (expected 2, got {})", pair.len()) })),
                        other => other.clone(),
                    };
                    let Obj::Str(name) = name else { return Err(ArchiveError::value("field names must be strings")) };
                    if names.contains(&name) { return Err(ArchiveError::value("name already used as a name or title")); }
                    names.push(name);
                }
                size += dtype.itemsize;
            }
            Ok(Dtype::void(size, true))
        }
    }
}
