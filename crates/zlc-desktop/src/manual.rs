//! The Figure Viewer's data working copies (v2 `zlc_workbench.viewer`'s
//! `_draft_*` rules and `data_editor_intent`): Edit data copies a Figure's
//! Dataset, New data starts a useful Curve, and every edit works on ONE dense
//! logical tensor -- Repeat and Point are expanded here and packed back onto
//! their carriers only when the copy is applied, so every axis obeys the same
//! Add / Edit / Remove and Rows / Columns / Scope rules.  The page only shows
//! what [`Draft::projection`] says and sends the operator's intents back.
use std::collections::BTreeMap;
use std::sync::Arc;

use serde_json::{Value, json};
use zlc_core::dataset::{DataBlock, Validity};
use zlc_core::pyfmt;
use zlc_core::restrict::Source;
use zlc_core::schema::{Axis, CodeBase, CodeMapping, Coordinate, Coordinates, DatasetSchema, Domain, DomainRole, ValidityContract, ValueSchema};
use zlc_core::store::{DatasetStore, Placement, Snapshot};
use zlc_core::{AnyBuffer, Complex128, Complex64, DType, F16, OwnedBuffer};

type Result<T> = std::result::Result<T, String>;

/// v2 `_MANUAL_DTYPES`, in its order, by numpy name.
pub const DTYPES: [(DType, &str); 14] = [
    (DType::Bool, "bool"), (DType::U8, "uint8"), (DType::U16, "uint16"), (DType::U32, "uint32"), (DType::U64, "uint64"),
    (DType::I8, "int8"), (DType::I16, "int16"), (DType::I32, "int32"), (DType::I64, "int64"),
    (DType::F16, "float16"), (DType::F32, "float32"), (DType::F64, "float64"), (DType::C64, "complex64"), (DType::C128, "complex128"),
];
fn dtype_name(dtype: DType) -> &'static str { DTYPES.iter().find(|(kind, _)| *kind == dtype).map_or("float64", |(_, name)| name) }
fn dtype_named(name: &str) -> Result<DType> {
    DTYPES.iter().find(|(_, known)| *known == name).map(|(kind, _)| *kind).ok_or_else(|| format!("unsupported manual data type {name:?}"))
}

const SCALAR_ID: &str = "zlc_data.scalar";
/// `zlc_data.SCALAR_AXIS`: the one trailing data item a scalar value occupies.
fn scalar_axis() -> Arc<Axis> {
    Axis::new(SCALAR_ID.into(), "value".into(), "scalar".into(), Coordinates::Numeric(OwnedBuffer::from_vec(vec![0_i64]).into()),
        None, None, None, None).expect("the scalar carrier axis")
}
fn is_scalar(axis: &Axis) -> bool { axis.role() == "scalar" }

/// The three Dataset domains as the editor names them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Dom { Repeat, Point, Cell }
impl Dom {
    const ALL: [Dom; 3] = [Dom::Repeat, Dom::Point, Dom::Cell];
    fn key(self) -> &'static str { match self { Dom::Repeat => "repeat", Dom::Point => "point", Dom::Cell => "cell_data" } }
    fn label(self) -> &'static str { match self { Dom::Repeat => "Repeat", Dom::Point => "Point", Dom::Cell => "Cell data" } }
    fn role(self) -> &'static str { match self { Dom::Repeat => "repeat", Dom::Point => "scan-point", Dom::Cell => "component" } }
    fn parse(text: &str) -> Result<Dom> {
        Dom::ALL.into_iter().find(|dom| dom.key() == text).ok_or_else(|| format!("unknown Dataset domain {text:?}"))
    }
}

// ------------------------------------------------------------------ scalars

/// One value read out of any dtype, for conversion and comparison.
#[derive(Clone, Copy, Debug)]
enum Num { Bool(bool), Int(i128), Float(f64), Complex(f64, f64) }
impl Num {
    fn same(self, other: Num) -> bool {
        let pair = |num: Num| match num { Num::Bool(flag) => (f64::from(u8::from(flag)), 0.0, None), Num::Int(int) => (int as f64, 0.0, Some(int)),
            Num::Float(real) => (real, 0.0, None), Num::Complex(re, im) => (re, im, None) };
        let (a, b) = (pair(self), pair(other));
        if let (Some(x), Some(y)) = (a.2, b.2) { return x == y; }
        let equal = |x: f64, y: f64| x == y || (x.is_nan() && y.is_nan());
        equal(a.0, b.0) && equal(a.1, b.1)
    }
    fn real(self) -> f64 { match self { Num::Bool(flag) => f64::from(u8::from(flag)), Num::Int(int) => int as f64, Num::Float(real) | Num::Complex(real, _) => real } }
}

/// numpy's `str(scalar)` of one stored value.
trait Element: Copy + Default + Send + Sync + 'static {
    fn num(self) -> Num;
    fn from_num(num: Num) -> Self;
    fn text(self) -> String;
}

macro_rules! integers {
    ($($type:ty),*) => {$(
        impl Element for $type {
            fn num(self) -> Num { Num::Int(self as i128) }
            fn from_num(num: Num) -> Self {
                match num { Num::Bool(flag) => Self::from(flag), Num::Int(int) => int as Self, Num::Float(real) | Num::Complex(real, _) => (real as i128) as Self }
            }
            fn text(self) -> String { self.to_string() }
        }
    )*};
}
integers!(u8, u16, u32, u64, i8, i16, i32, i64);

impl Element for bool {
    fn num(self) -> Num { Num::Bool(self) }
    fn from_num(num: Num) -> Self {
        match num { Num::Bool(flag) => flag, Num::Int(int) => int != 0, Num::Float(real) => real != 0.0, Num::Complex(re, im) => re != 0.0 || im != 0.0 }
    }
    fn text(self) -> String { (if self { "True" } else { "False" }).into() }
}
impl Element for f64 {
    fn num(self) -> Num { Num::Float(self) }
    fn from_num(num: Num) -> Self { num.real() }
    fn text(self) -> String { pyfmt::repr(self) }
}
impl Element for f32 {
    fn num(self) -> Num { Num::Float(f64::from(self)) }
    fn from_num(num: Num) -> Self { num.real() as f32 }
    fn text(self) -> String { short_repr(f64::from(self), &format!("{:e}", self.abs())) }
}
impl Element for F16 {
    fn num(self) -> Num { Num::Float(zlc_plot::projection::nd::half_to_f64(self.to_bits())) }
    fn from_num(num: Num) -> Self { F16::from_bits(half_bits(num.real())) }
    fn text(self) -> String {
        let value = zlc_plot::projection::nd::half_to_f64(self.to_bits());
        short_repr(value, &format!("{:e}", (value as f32).abs()))
    }
}
impl Element for Complex128 {
    fn num(self) -> Num { Num::Complex(self.re, self.im) }
    fn from_num(num: Num) -> Self { match num { Num::Complex(re, im) => Complex128 { re, im }, other => Complex128 { re: other.real(), im: 0.0 } } }
    fn text(self) -> String { complex_text(pyfmt::repr(self.re), pyfmt::repr(self.im), self.re, self.im) }
}
impl Element for Complex64 {
    fn num(self) -> Num { Num::Complex(f64::from(self.re), f64::from(self.im)) }
    fn from_num(num: Num) -> Self { let wide = Complex128::from_num(num); Complex64 { re: wide.re as f32, im: wide.im as f32 } }
    fn text(self) -> String {
        complex_text(f32::text(self.re), f32::text(self.im), f64::from(self.re), f64::from(self.im))
    }
}

/// Python's `repr` layout over the shortest digits of a narrower float
/// (numpy's `str(np.float32(x))`): `digits` is Rust's shortest `{:e}` text.
fn short_repr(value: f64, digits: &str) -> String {
    if value.is_nan() { return "nan".into(); }
    if value.is_infinite() { return if value < 0.0 { "-inf" } else { "inf" }.into(); }
    let (mantissa, exponent) = digits.split_once('e').expect("exponent form");
    let digits = mantissa.replace('.', "");
    let decpt = exponent.parse::<i32>().expect("exponent") + 1;
    let sign = if value.is_sign_negative() { "-" } else { "" };
    if decpt <= -4 || decpt > 16 {
        let tail = if digits.len() > 1 { format!(".{}", &digits[1..]) } else { String::new() };
        let shown = decpt - 1;
        return format!("{sign}{}{tail}e{}{:02}", &digits[..1], if shown < 0 { '-' } else { '+' }, shown.unsigned_abs());
    }
    if decpt <= 0 { format!("{sign}0.{}{digits}", "0".repeat((-decpt) as usize)) }
    else if decpt as usize >= digits.len() { format!("{sign}{digits}{}.0", "0".repeat(decpt as usize - digits.len())) }
    else { format!("{sign}{}.{}", &digits[..decpt as usize], &digits[decpt as usize..]) }
}

/// Python's complex repr: `2j`, `(1+2j)`, components without a trailing `.0`.
fn complex_text(re: String, im: String, real: f64, imaginary: f64) -> String {
    let bare = |text: String| text.strip_suffix(".0").map(str::to_owned).unwrap_or(text);
    let imaginary_text = bare(im);
    if real == 0.0 && !real.is_sign_negative() { return format!("{imaginary_text}j"); }
    let sign = if imaginary_text.starts_with('-') || (imaginary.is_nan() && imaginary.is_sign_negative()) { "" } else { "+" };
    format!("({}{sign}{imaginary_text}j)", bare(re))
}

/// IEEE binary64 to binary16, rounding to nearest even (numpy's cast).
fn half_bits(value: f64) -> u16 {
    let single = value as f32;
    let bits = single.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;
    if exponent == 0xff { return sign | 0x7c00 | if mantissa != 0 { 0x200 } else { 0 }; }
    let unbiased = exponent - 127;
    if unbiased > 15 { return sign | 0x7c00; }
    if unbiased >= -14 {
        let half_exponent = (unbiased + 15) as u32;
        let mut half = (half_exponent << 10) | (mantissa >> 13);
        let rest = mantissa & 0x1fff;
        if rest > 0x1000 || (rest == 0x1000 && half & 1 == 1) { half += 1; }
        return sign | half as u16;
    }
    if unbiased < -25 { return sign; }
    let full = mantissa | 0x80_0000;
    let shift = (-14 - unbiased + 13) as u32;
    let mut half = full >> shift;
    let rest = full & ((1 << shift) - 1);
    let middle = 1 << (shift - 1);
    if rest > middle || (rest == middle && half & 1 == 1) { half += 1; }
    sign | half as u16
}

/// `_parse_value(text, dtype)`: numpy's `np.asarray(text, dtype=dtype).item()`.
fn parse_value(text: &str, dtype: DType) -> Result<Num> {
    let value = text.trim();
    let refused = || format!("{} is not a {} value", python_quote(value), dtype_name(dtype));
    match dtype {
        DType::Bool => match value.to_lowercase().as_str() {
            "true" | "1" | "yes" | "on" => Ok(Num::Bool(true)),
            "false" | "0" | "no" | "off" => Ok(Num::Bool(false)),
            _ => Err(format!("{} is not a boolean value", python_quote(value))),
        },
        DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
            let digits = value.replace('_', "");
            let int = digits.parse::<i128>().map_err(|_| refused())?;
            let (low, high): (i128, i128) = match dtype {
                DType::U8 => (0, u8::MAX.into()), DType::U16 => (0, u16::MAX.into()), DType::U32 => (0, u32::MAX.into()), DType::U64 => (0, u64::MAX.into()),
                DType::I8 => (i8::MIN.into(), i8::MAX.into()), DType::I16 => (i16::MIN.into(), i16::MAX.into()),
                DType::I32 => (i32::MIN.into(), i32::MAX.into()), _ => (i64::MIN.into(), i64::MAX.into()),
            };
            if int < low || int > high { return Err(refused()); }
            Ok(Num::Int(int))
        }
        DType::F16 | DType::F32 | DType::F64 => python_float(value).map(Num::Float).ok_or_else(refused),
        DType::C64 | DType::C128 => python_complex(value).map(|(re, im)| Num::Complex(re, im)).ok_or_else(refused),
    }
}

/// Python's `repr(str)` for a message.
fn python_quote(text: &str) -> String { if text.contains('\'') && !text.contains('"') { format!("\"{text}\"") } else { format!("'{}'", text.replace('\'', "\\'")) } }

/// `float(text)`: decimal or exponent notation, `nan`, `inf`, digit underscores.
fn python_float(text: &str) -> Option<f64> {
    let text = text.trim();
    if text.is_empty() || text.starts_with('_') || text.ends_with('_') || text.contains("__") { return None; }
    let plain = text.replace('_', "");
    let lowered = plain.to_lowercase();
    let unsigned = lowered.trim_start_matches(['+', '-']);
    if matches!(unsigned, "nan" | "inf" | "infinity") { return plain.parse::<f64>().ok().or_else(|| lowered.parse::<f64>().ok()); }
    if !unsigned.chars().next().is_some_and(|first| first.is_ascii_digit() || first == '.') { return None; }
    plain.parse::<f64>().ok()
}

/// `complex(text)`: `a`, `bj`, `a+bj`, `a-bj`, optionally in parentheses.
fn python_complex(text: &str) -> Option<(f64, f64)> {
    let mut text = text.trim();
    if let Some(inner) = text.strip_prefix('(').and_then(|rest| rest.strip_suffix(')')) { text = inner.trim(); }
    if text.is_empty() || text.contains(char::is_whitespace) { return None; }
    let Some(body) = text.strip_suffix(['j', 'J']) else { return python_float(text).map(|re| (re, 0.0)) };
    let bytes = body.as_bytes();
    // The imaginary part starts at the last sign not inside an exponent.
    let split = (1..bytes.len()).rev().find(|&at| matches!(bytes[at], b'+' | b'-') && !matches!(bytes[at - 1], b'e' | b'E'));
    let imaginary = |part: &str| match part { "" | "+" => Some(1.0), "-" => Some(-1.0), other => python_float(other) };
    match split {
        Some(at) => Some((python_float(&body[..at])?, imaginary(&body[at..])?)),
        None => Some((0.0, imaginary(body)?)),
    }
}

// ------------------------------------------------------------------ dense tensors

/// One dense C-order tensor of one dtype.
#[derive(Clone)]
enum Dense { Bool(Vec<bool>), U8(Vec<u8>), U16(Vec<u16>), U32(Vec<u32>), U64(Vec<u64>), I8(Vec<i8>), I16(Vec<i16>), I32(Vec<i32>), I64(Vec<i64>),
             F16(Vec<F16>), F32(Vec<f32>), F64(Vec<f64>), C64(Vec<Complex64>), C128(Vec<Complex128>) }

macro_rules! each {
    ($dense:expr, $values:ident => $body:expr) => {
        match $dense {
            Dense::Bool($values) => $body, Dense::U8($values) => $body, Dense::U16($values) => $body, Dense::U32($values) => $body,
            Dense::U64($values) => $body, Dense::I8($values) => $body, Dense::I16($values) => $body, Dense::I32($values) => $body,
            Dense::I64($values) => $body, Dense::F16($values) => $body, Dense::F32($values) => $body, Dense::F64($values) => $body,
            Dense::C64($values) => $body, Dense::C128($values) => $body,
        }
    };
}
macro_rules! rebuild {
    ($dense:expr, $values:ident => $body:expr) => {
        match $dense {
            Dense::Bool($values) => Dense::Bool($body), Dense::U8($values) => Dense::U8($body), Dense::U16($values) => Dense::U16($body),
            Dense::U32($values) => Dense::U32($body), Dense::U64($values) => Dense::U64($body), Dense::I8($values) => Dense::I8($body),
            Dense::I16($values) => Dense::I16($body), Dense::I32($values) => Dense::I32($body), Dense::I64($values) => Dense::I64($body),
            Dense::F16($values) => Dense::F16($body), Dense::F32($values) => Dense::F32($body), Dense::F64($values) => Dense::F64($body),
            Dense::C64($values) => Dense::C64($body), Dense::C128($values) => Dense::C128($body),
        }
    };
}

impl Dense {
    fn zeros(dtype: DType, len: usize) -> Self {
        match dtype {
            DType::Bool => Dense::Bool(vec![false; len]), DType::U8 => Dense::U8(vec![0; len]), DType::U16 => Dense::U16(vec![0; len]),
            DType::U32 => Dense::U32(vec![0; len]), DType::U64 => Dense::U64(vec![0; len]), DType::I8 => Dense::I8(vec![0; len]),
            DType::I16 => Dense::I16(vec![0; len]), DType::I32 => Dense::I32(vec![0; len]), DType::I64 => Dense::I64(vec![0; len]),
            DType::F16 => Dense::F16(vec![F16::default(); len]), DType::F32 => Dense::F32(vec![0.0; len]), DType::F64 => Dense::F64(vec![0.0; len]),
            DType::C64 => Dense::C64(vec![Complex64::default(); len]), DType::C128 => Dense::C128(vec![Complex128::default(); len]),
        }
    }
    fn of(buffer: &AnyBuffer) -> Self {
        match buffer {
            AnyBuffer::Bool(v) => Dense::Bool(v.as_slice().to_vec()), AnyBuffer::U8(v) => Dense::U8(v.as_slice().to_vec()),
            AnyBuffer::U16(v) => Dense::U16(v.as_slice().to_vec()), AnyBuffer::U32(v) => Dense::U32(v.as_slice().to_vec()),
            AnyBuffer::U64(v) => Dense::U64(v.as_slice().to_vec()), AnyBuffer::I8(v) => Dense::I8(v.as_slice().to_vec()),
            AnyBuffer::I16(v) => Dense::I16(v.as_slice().to_vec()), AnyBuffer::I32(v) => Dense::I32(v.as_slice().to_vec()),
            AnyBuffer::I64(v) => Dense::I64(v.as_slice().to_vec()), AnyBuffer::F16(v) => Dense::F16(v.as_slice().to_vec()),
            AnyBuffer::F32(v) => Dense::F32(v.as_slice().to_vec()), AnyBuffer::F64(v) => Dense::F64(v.as_slice().to_vec()),
            AnyBuffer::C64(v) => Dense::C64(v.as_slice().to_vec()), AnyBuffer::C128(v) => Dense::C128(v.as_slice().to_vec()),
        }
    }
    fn buffer(&self) -> AnyBuffer { each!(self, values => OwnedBuffer::from_vec(values.clone()).into()) }
    fn dtype(&self) -> DType {
        match self {
            Dense::Bool(_) => DType::Bool, Dense::U8(_) => DType::U8, Dense::U16(_) => DType::U16, Dense::U32(_) => DType::U32, Dense::U64(_) => DType::U64,
            Dense::I8(_) => DType::I8, Dense::I16(_) => DType::I16, Dense::I32(_) => DType::I32, Dense::I64(_) => DType::I64, Dense::F16(_) => DType::F16,
            Dense::F32(_) => DType::F32, Dense::F64(_) => DType::F64, Dense::C64(_) => DType::C64, Dense::C128(_) => DType::C128,
        }
    }
    fn num(&self, index: usize) -> Num { each!(self, values => values[index].num()) }
    fn text(&self, index: usize) -> String { each!(self, values => values[index].text()) }
    fn set(&mut self, index: usize, num: Num) { each!(self, values => values[index] = Element::from_num(num)) }
    fn zero(&mut self, index: usize) { each!(self, values => values[index] = Default::default()) }
    /// `astype(dtype)`, element by element as numpy casts.
    fn cast(&self, dtype: DType) -> Dense {
        let mut target = Dense::zeros(dtype, self.len());
        for index in 0..self.len() { target.set(index, self.num(index)); }
        target
    }
    fn len(&self) -> usize { each!(self, values => values.len()) }
    fn remap(&self, shape: &[usize], target: &[usize], source: &dyn Fn(&[usize]) -> Option<usize>) -> Dense {
        rebuild!(self, values => remap(values, shape, target, Default::default(), source))
    }
}

fn product(shape: &[usize]) -> usize { shape.iter().product() }

/// A tensor of `target` shape whose element at each index is `source(index)`'s
/// element of `data` (C order), or `fill` where it names none.
fn remap<T: Copy>(data: &[T], shape: &[usize], target: &[usize], fill: T, source: &dyn Fn(&[usize]) -> Option<usize>) -> Vec<T> {
    let _ = shape;
    let mut out = Vec::with_capacity(product(target));
    let mut index = vec![0usize; target.len()];
    for _ in 0..product(target) {
        out.push(source(&index).map_or(fill, |at| data[at]));
        for dim in (0..target.len()).rev() {
            index[dim] += 1;
            if index[dim] < target[dim] { break; }
            index[dim] = 0;
        }
    }
    out
}

fn flat(shape: &[usize], index: &[usize]) -> usize { index.iter().zip(shape).fold(0, |at, (i, size)| at * size + i) }

/// The tensor reshaped: `dim` resized to `len` (kept where it overlaps, `fill` beyond).
fn resized(shape: &[usize], dim: usize, len: usize) -> (Vec<usize>, impl Fn(&[usize]) -> Option<usize> + '_) {
    let mut target = shape.to_vec();
    target[dim] = len;
    (target, move |index: &[usize]| (index[dim] < shape[dim]).then(|| flat(shape, index)))
}

// ------------------------------------------------------------------ the working copy

/// What a Figure's dataset hands its working copy.
pub struct EditSource {
    pub snapshot: Snapshot,
    pub name: String,
    pub path: String,
    pub dataset: String,
    pub recipe: Option<Value>,
    pub signal: String,
    /// The image's archived point layer (its sites), drawn on the preview while the coordinates match.
    pub overlay: Option<Arc<zlc_plot::session::ImagePointOverlay>>,
}

/// What an Apply publishes and shows.
pub struct Apply {
    pub editor: String,
    pub owner: String,
    pub signal: String,
    pub snapshot: Snapshot,
    pub metadata: Value,
    /// The archive dataset or the previous Apply this one edits.
    pub parent: Option<String>,
    /// The panel already showing this copy, if its kind can still show the edited shape.
    pub panel: Option<String>,
    /// The archive's recipe target for a first panel, when it still describes the data.
    pub recipe: Option<Value>,
    /// v2 `_manual_plot_input`: the archived point layer, kept only while all three domains are the copy's source's.
    pub overlay: Option<Arc<zlc_plot::session::ImagePointOverlay>>,
}

/// What an intent asks of the viewer beyond the copy itself.
/// `Save`: Save as… onto the archive `path` (None: the window's Save dialog asks), writing
/// over files already there only when `overwrite`.
pub enum Effect { None, Apply(Box<Apply>), Save { editor: String, panel: String, path: Option<String>, overwrite: bool, suggested: String }, Discarded { panel: Option<String>, signal: String } }

/// The document `save` a Save as… onto the archive `path` becomes: its picture beside it
/// (v2 `with_suffix(".png")`; a name's own dots, "scan_1.5us", stay), writing over no file
/// already there the operator has not agreed to -- all of them with `overwrite`, else only
/// the archive, when `asked` (the window's Save dialog asked about it as it was chosen).
pub fn save_command(id: &Value, panel: &str, document: u32, path: &str, asked: bool, overwrite: bool) -> Value {
    let image = match path.strip_suffix(".npz").or_else(|| path.strip_suffix(".NPZ")) { Some(stem) => format!("{stem}.png"), None => format!("{path}.png") };
    let agreed = if overwrite { json!(true) } else if asked { json!([path]) } else { json!(false) };
    json!({"id": id, "action": "save", "panel": panel, "path": image, "document": document, "overwrite": agreed})
}

pub struct Draft {
    pub id: String,
    pub serial: u64,
    name: String, initial_name: String,
    dtype: DType,
    unit: Option<String>,
    note: String, initial_note: String,
    source_text: String,
    source_path: Option<String>,
    source_dataset: String,
    pub source_signal: Option<String>,
    source_snapshot: Snapshot,
    source_repeat: Arc<Domain>, source_point: Arc<Domain>,
    recipe: Option<Value>,
    source_overlay: Option<Arc<zlc_plot::session::ImagePointOverlay>>,
    axes: [Vec<Arc<Axis>>; 3],
    validity_contract: ValidityContract,
    value_name: Option<String>,
    values: Dense,
    validity: Vec<bool>,
    sigma: Option<Vec<f64>>,
    shape: Vec<usize>,
    selected_axis: String,
    row_axis: String, column_axis: String,
    scopes: BTreeMap<String, usize>,
    component: &'static str,
    /// The table rows and columns the page shows: `[row, rows, column, columns]`.
    window: [usize; 4],
    modified: bool,
    unsaved: bool,
    message: String,
    severity: &'static str,
    /// What the last Apply published: its snapshot, the values its zeros replaced, its name and note.
    applied: Option<(Snapshot, Option<Dense>, String, String)>,
    pub panel: String,
    pub save_ready: bool,
    /// How many Applies this copy has had (a Save names the one it wrote).
    pub applies: u64,
    /// Its tab is open (an applied copy stays in the Data chooser after its tab closes).
    pub open: bool,
}

/// A coordinate as an axis value reads in a header or a cell.
fn coordinate_text(coordinate: &Coordinate) -> String {
    match coordinate {
        Coordinate::Integer(value) => value.to_string(),
        Coordinate::Real(value) => pyfmt::repr(*value),
        Coordinate::Text(text) => text.to_string(),
        Coordinate::Missing => "None".into(),
    }
}
/// `SemanticCycleChoices` label: `{coordinate:g}` (or its label), then the unit.
fn scope_label(axis: &Axis, index: usize) -> String {
    let words = match (axis.labels(), axis.coordinate(index)) {
        (Some(labels), _) => labels[index].to_string(),
        (None, Some(Coordinate::Integer(value))) => pyfmt::format_g(value as f64, 6),
        (None, Some(Coordinate::Real(value))) => pyfmt::format_g(value, 6),
        (None, Some(Coordinate::Text(text))) => text.to_string(),
        (None, _) => "(null)".into(),
    };
    match axis.unit() { Some(unit) if !unit.is_empty() => format!("{words} {unit}"), _ => words }
}
fn coordinates_of(axis: &Axis) -> Vec<Coordinate> { (0..axis.len()).filter_map(|index| axis.coordinate(index)).collect() }
fn explicit(axis: &Axis) -> bool { !matches!(axis.coordinates(), Coordinates::Index { .. }) }
fn index_origin(axis: &Axis) -> i128 { if let Coordinates::Index { origin, .. } = axis.coordinates() { *origin } else { 0 } }

/// Explicit coordinates as the archive keeps them: an integer or real vector, else values.
fn coordinates(values: Vec<Coordinate>) -> Coordinates {
    if values.iter().all(|value| matches!(value, Coordinate::Integer(int) if i64::try_from(*int).is_ok())) {
        return Coordinates::Numeric(OwnedBuffer::from_vec(values.iter().map(|value| match value { Coordinate::Integer(int) => *int as i64, _ => 0 }).collect::<Vec<i64>>()).into());
    }
    if values.iter().all(|value| matches!(value, Coordinate::Real(_))) {
        return Coordinates::Numeric(OwnedBuffer::from_vec(values.iter().map(|value| match value { Coordinate::Real(real) => *real, _ => 0.0 }).collect::<Vec<f64>>()).into());
    }
    Coordinates::Values(values.into())
}

/// `_parse_axis_value`: quoted text, an integer, a real, else the words.
fn parse_axis_value(text: &str) -> Result<Coordinate> {
    let token = text.trim();
    if token.is_empty() { return Err("axis values cannot be blank".into()); }
    let quoted = token.len() >= 2 && token.starts_with(['\'', '"']) && token.ends_with(&token[..1]);
    let value = if quoted { Coordinate::Text(token[1..token.len() - 1].into()) }
        else if token.trim_start_matches(['+', '-']).chars().all(|c| c.is_ascii_digit()) && token.trim_start_matches(['+', '-']).len() == token.len() - usize::from(token.starts_with(['+', '-'])) && token.len() > usize::from(token.starts_with(['+', '-'])) {
            Coordinate::Integer(token.parse::<i128>().map_err(|_| "axis value is too large".to_string())?)
        } else if let Some(real) = python_float(token) { Coordinate::Real(real) } else { Coordinate::Text(token.into()) };
    match value {
        Coordinate::Real(real) if !real.is_finite() => Err("axis value must be finite".into()),
        Coordinate::Real(real) if real.fract() == 0.0 && real.abs() < 9.0e15 => Ok(Coordinate::Real(real)),
        other => Ok(other),
    }
}

/// `_next_numeric_coordinate`: continue a numeric axis by its own last step.
fn next_numeric(values: &[Coordinate], occupied: &[Coordinate]) -> Result<Coordinate> {
    let add = |a: &Coordinate, b: &Coordinate| -> Coordinate {
        match (a, b) {
            (Coordinate::Integer(x), Coordinate::Integer(y)) => x.checked_add(*y).map_or(Coordinate::Real(*x as f64 + *y as f64), Coordinate::Integer),
            _ => Coordinate::Real(number(a) + number(b)),
        }
    };
    let step = if values.len() >= 2 {
        match (&values[values.len() - 1], &values[values.len() - 2]) {
            (Coordinate::Integer(x), Coordinate::Integer(y)) => Coordinate::Integer(x - y),
            (x, y) => Coordinate::Real(number(x) - number(y)),
        }
    } else { Coordinate::Integer(1) };
    let mut candidate = add(&values[values.len() - 1], &step);
    while occupied.contains(&candidate) {
        let advanced = add(&candidate, &step);
        if advanced == candidate {
            return Err(format!("cannot continue the axis past {} in steps of {}; enter the new coordinates by hand",
                coordinate_text(&candidate), coordinate_text(&step)));
        }
        candidate = advanced;
    }
    Ok(candidate)
}
fn number(value: &Coordinate) -> f64 { match value { Coordinate::Integer(int) => *int as f64, Coordinate::Real(real) => *real, _ => f64::NAN } }

/// `_resized_coordinates`: an explicit axis grown by its own pattern, or cut.
fn resized_coordinates(previous: Option<&Axis>, length: usize) -> Result<Option<Vec<Coordinate>>> {
    let Some(axis) = previous.filter(|axis| explicit(axis)) else { return Ok(None) };
    let mut values: Vec<Coordinate> = coordinates_of(axis).into_iter().take(length).collect();
    let mut occupied = values.clone();
    while values.len() < length {
        let candidate = if !values.is_empty() && values.iter().all(|value| matches!(value, Coordinate::Integer(_) | Coordinate::Real(_))) {
            next_numeric(&values, &occupied)?
        } else {
            let mut serial = values.len();
            let mut text = format!("{}-{serial}", axis.name());
            while occupied.contains(&Coordinate::Text(text.as_str().into())) { serial += 1; text = format!("{}-{serial}", axis.name()); }
            Coordinate::Text(text.into())
        };
        occupied.push(candidate.clone());
        values.push(candidate);
    }
    Ok(Some(values))
}

/// `_axis_spec`: an axis as its fields describe it, keeping what `previous` had.
fn axis_spec(id: &str, name: &str, length: usize, unit: &str, dom: Dom, previous: Option<&Axis>, preserve_role: bool) -> Result<Arc<Axis>> {
    if length == 0 { return Err("axis length must be at least 1".into()); }
    let values = resized_coordinates(previous, length)?;
    let same_length = previous.is_some_and(|axis| axis.len() == length);
    let coordinates = match values {
        Some(values) => coordinates(values),
        None => Coordinates::Index { origin: previous.map_or(0, index_origin), len: length },
    };
    let role = match previous { Some(axis) if preserve_role => axis.role().to_owned(), _ => dom.role().to_owned() };
    let unit = unit.trim();
    Axis::new(id.into(), name.trim().into(), role.into(), coordinates, (!unit.is_empty()).then(|| unit.into()),
        previous.and_then(|axis| axis.coordinate_frame()).map(Into::into),
        previous.filter(|_| same_length).and_then(|axis| axis.labels()).map(|labels| labels.to_vec()),
        previous.and_then(|axis| axis.coordinate_of()).map(Into::into))
}

/// `_mapped_domain`: a flat carrier whose rows are the C-order product of its axes.
fn mapped_domain(axes: &[Arc<Axis>]) -> Result<Arc<Domain>> {
    if axes.is_empty() { return Domain::new(vec![1], vec![], vec![]); }
    let primary: Vec<&Arc<Axis>> = axes.iter().filter(|axis| axis.coordinate_of().is_none()).collect();
    let size: usize = primary.iter().map(|axis| axis.len()).product();
    let (mut inner, mut outer) = (size, 1usize);
    let mut mapped: BTreeMap<String, Arc<CodeMapping>> = BTreeMap::new();
    for axis in &primary {
        inner /= axis.len();
        mapped.insert(axis.id().to_owned(), CodeMapping::new(CodeBase::Range { start: 0, step: 1, len: axis.len() }, inner, outer)?);
        outer *= axis.len();
    }
    let mappings = axes.iter().map(|axis| mapped.get(axis.coordinate_of().unwrap_or(axis.id())).cloned().ok_or_else(|| "Alternative coordinate primary is absent".to_string()))
        .collect::<Result<Vec<_>>>()?;
    Domain::new(vec![size], axes.to_vec(), mappings)
}

/// `_edited_mapped_domain`: keep a carrier's map until the operator changes its topology.
fn edited_domain(axes: &[Arc<Axis>], source: &Domain) -> Result<Arc<Domain>> {
    let signature = |axes: &[Arc<Axis>]| axes.iter().map(|axis| (axis.id().to_owned(), axis.len())).collect::<Vec<_>>();
    if signature(axes) != signature(source.axes()) { return mapped_domain(axes); }
    let mappings = source.axes().iter().map(|axis| source.mapping(axis.id()).cloned().expect("a domain maps its axes")).collect();
    Domain::new(source.physical_shape().to_vec(), axes.to_vec(), mappings)
}

/// `_domain_flat_rows`: each carrier row's position in the domain's logical C-order tensor.
fn domain_rows(domain: &Domain) -> Result<Vec<usize>> {
    let primary: Vec<&Arc<Axis>> = domain.axes().iter().filter(|axis| axis.coordinate_of().is_none()).collect();
    if primary.is_empty() { return Ok(vec![0; domain.size()]); }
    let shape: Vec<usize> = primary.iter().map(|axis| axis.len()).collect();
    let mappings: Vec<&Arc<CodeMapping>> = primary.iter().map(|axis| domain.mapping(axis.id()).expect("a domain maps its axes")).collect();
    let rows: Vec<usize> = (0..domain.size()).map(|row| {
        let index: Vec<usize> = mappings.iter().map(|mapping| mapping.code(row).expect("a mapping covers its carrier")).collect();
        flat(&shape, &index)
    }).collect();
    let mut seen = vec![false; product(&shape)];
    for row in &rows {
        if std::mem::replace(&mut seen[*row], true) { return Err("manual editing requires unique logical rows in each Dataset domain".into()); }
    }
    Ok(rows)
}

impl Draft {
    /// `_new_manual_snapshot`: a useful Curve Dataset, not an empty shape to repair.
    pub fn new_curve(serial: u64) -> Result<Self> {
        let count = 16;
        let repeat = Axis::new("manual.repeat".into(), "repeat".into(), "repeat".into(), Coordinates::Numeric(OwnedBuffer::from_vec(vec![0_i64]).into()), None, None, None, None)?;
        let x = Axis::new("manual.x".into(), "x".into(), "scan-point".into(),
            Coordinates::Numeric(OwnedBuffer::from_vec((0..count as i64).collect::<Vec<i64>>()).into()), None, None, None, None)?;
        let schema = DatasetSchema::new(mapped_domain(&[repeat])?, mapped_domain(&[x])?, Domain::dense(vec![1], vec![scalar_axis()])?,
            ValueSchema::new(DType::F64, None, None, ValidityContract::Value)?)?;
        let block = DataBlock::new(schema.clone(), OwnedBuffer::from_vec(vec![0.0_f64; count]).into(), Validity::Uniform(true), None)?;
        let snapshot = DatasetStore::new(schema).commit(vec![Placement::new([0, 0], [1, count], block)?])?;
        let name = format!("Manual data {serial}");
        Self::from_snapshot(&snapshot, serial, &name, "New manual Dataset".into(), None, String::new(), None, None)
    }

    /// `_draft_from_snapshot`: one working copy of `source`'s dataset.
    pub fn edit(serial: u64, source: EditSource) -> Result<Self> {
        let file = std::path::Path::new(&source.path).file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default();
        let mut draft = Self::from_snapshot(&source.snapshot, serial, &source.name, format!("Copy of {file} · {}", source.dataset), Some(source.path),
            source.dataset, source.recipe, Some(source.signal))?;
        draft.source_overlay = source.overlay;
        Ok(draft)
    }

    #[allow(clippy::too_many_arguments)]
    fn from_snapshot(snapshot: &Snapshot, serial: u64, name: &str, source_text: String, source_path: Option<String>, source_dataset: String,
                     recipe: Option<Value>, source_signal: Option<String>) -> Result<Self> {
        let schema = snapshot.schema().clone();
        let (values, validity, sigma, shape) = expand(snapshot)?;
        let axes = [schema.domain(DomainRole::Repeat).axes().to_vec(), schema.domain(DomainRole::Point).axes().to_vec(), schema.domain(DomainRole::Cell).axes().to_vec()];
        let visible: Vec<String> = [&axes[1], &axes[2], &axes[0]].into_iter().flatten().filter(|axis| !is_scalar(axis)).map(|axis| axis.id().to_owned()).collect();
        let mut draft = Self {
            id: format!("data-{serial}"), serial, name: name.into(), initial_name: name.into(), dtype: schema.dtype(),
            unit: schema.value().unit().map(str::to_owned), note: String::new(), initial_note: String::new(), source_text,
            modified: source_path.is_none(), source_path, source_dataset, source_signal, source_snapshot: snapshot.clone(),
            source_repeat: schema.domain(DomainRole::Repeat).clone(), source_point: schema.domain(DomainRole::Point).clone(), recipe, source_overlay: None,
            axes, validity_contract: schema.value().validity().clone(), value_name: schema.value().name().map(str::to_owned),
            values, validity, sigma, shape, selected_axis: visible.first().cloned().unwrap_or_default(), row_axis: String::new(), column_axis: String::new(),
            scopes: BTreeMap::new(), component: "values", window: [0, 64, 0, 32], unsaved: false, message: String::new(), severity: "idle",
            applied: None, panel: String::new(), save_ready: false, applies: 0, open: true,
        };
        draft.normalize_table();
        Ok(draft)
    }

    pub fn name(&self) -> &str { &self.name }
    pub fn dirty(&self) -> bool { self.modified || self.unsaved }
    pub fn signal(&self) -> String { format!("@figure/manual/{}/data", self.serial) }
    pub fn applied(&self) -> bool { self.applied.is_some() }

    fn storage_axes(&self) -> Vec<Arc<Axis>> { self.axes.iter().flatten().filter(|axis| axis.coordinate_of().is_none()).cloned().collect() }
    fn visible_axes(&self) -> Vec<Arc<Axis>> { self.storage_axes().into_iter().filter(|axis| !is_scalar(axis)).collect() }
    fn primary_count(&self, dom: Dom) -> usize { self.axes[dom as usize].iter().filter(|axis| axis.coordinate_of().is_none()).count() }

    fn location(&self, id: &str) -> Result<(Dom, usize, Arc<Axis>)> {
        for dom in Dom::ALL {
            if let Some(position) = self.axes[dom as usize].iter().position(|axis| axis.id() == id && !is_scalar(axis)) {
                return Ok((dom, position, self.axes[dom as usize][position].clone()));
            }
        }
        Err(format!("unknown Dataset axis {}", python_quote(id)))
    }

    /// `_normalize_table_axes`.
    fn normalize_table(&mut self) {
        let ids: Vec<String> = self.visible_axes().iter().map(|axis| axis.id().to_owned()).collect();
        let mut row = self.row_axis.clone();
        let mut column = self.column_axis.clone();
        if !ids.contains(&row) { row = if ids.len() >= 2 { ids[ids.len() - 2].clone() } else { ids.first().cloned().unwrap_or_default() }; }
        if !ids.contains(&column) || column == row { column = if ids.len() >= 2 && ids[ids.len() - 1] != row { ids[ids.len() - 1].clone() } else { String::new() }; }
        self.row_axis = row;
        self.column_axis = column;
        let sizes: BTreeMap<String, usize> = self.visible_axes().iter().map(|axis| (axis.id().to_owned(), axis.len())).collect();
        self.scopes = sizes.iter().map(|(id, size)| (id.clone(), self.scopes.get(id).copied().unwrap_or(0).min(size - 1))).collect();
    }

    // -------------------------------------------------------------- array edits

    fn reshape(&mut self, target: Vec<usize>, source: &dyn Fn(&[usize]) -> Option<usize>) {
        let shape = self.shape.clone();
        self.values = self.values.remap(&shape, &target, source);
        self.validity = remap(&self.validity, &shape, &target, false, source);
        self.sigma = self.sigma.as_ref().map(|sigma| remap(sigma, &shape, &target, f64::NAN, source));
        self.shape = target;
    }
    /// `_resize_storage_axis`.
    fn resize_dim(&mut self, dim: usize, len: usize) {
        if self.shape[dim] == len { return; }
        let shape = self.shape.clone();
        let (target, source) = resized(&shape, dim, len);
        self.reshape(target, &source);
    }
    /// `_insert_storage_axis`: the existing data at index 0 of the new dimension.
    fn insert_dim(&mut self, dim: usize, len: usize) {
        let shape = self.shape.clone();
        let mut target = shape.clone();
        target.insert(dim, len);
        self.reshape(target, &move |index: &[usize]| {
            if index[dim] != 0 { return None; }
            let mut kept = index.to_vec();
            kept.remove(dim);
            Some(flat(&shape, &kept))
        });
    }
    /// `_take_storage_axis`.
    fn take_dim(&mut self, dim: usize, at: usize) {
        let shape = self.shape.clone();
        let mut target = shape.clone();
        target.remove(dim);
        self.reshape(target, &move |index: &[usize]| {
            let mut full = index.to_vec();
            full.insert(dim, at);
            Some(flat(&shape, &full))
        });
    }
    fn permute(&mut self, order: &[usize]) {
        let shape = self.shape.clone();
        let target: Vec<usize> = order.iter().map(|dim| shape[*dim]).collect();
        let order = order.to_vec();
        self.reshape(target, &move |index: &[usize]| {
            let mut full = vec![0; shape.len()];
            for (position, dim) in order.iter().enumerate() { full[*dim] = index[position]; }
            Some(flat(&shape, &full))
        });
    }

    // -------------------------------------------------------------- axes

    fn new_axis_id(&self, stem: &str) -> String {
        let mut base = String::new();
        let mut replacing = false;
        for character in stem.trim().chars() {
            let kept = character.is_ascii_alphanumeric() || matches!(character, '_' | '.' | '-');
            if kept { base.push(character) } else if !replacing { base.push('-') }
            replacing = !kept;
        }
        let base = base.trim_matches(['.', '-']).to_owned();
        let base = if base.is_empty() { "axis".to_owned() } else { base };
        let occupied: Vec<&str> = self.axes.iter().flatten().map(|axis| axis.id()).collect();
        (1..).map(|serial| if serial == 1 { format!("manual.{base}") } else { format!("manual.{base}.{serial}") })
            .find(|candidate| !occupied.contains(&candidate.as_str())).expect("an unused axis id")
    }

    /// `_add_axis`.
    fn add_axis(&mut self, name: &str, length: usize, unit: &str, dom: Dom) -> Result<()> {
        let axis = axis_spec(&self.new_axis_id(name), name, length, unit, dom, None, false)?;
        let (repeat, point) = (self.primary_count(Dom::Repeat), self.primary_count(Dom::Point));
        let cell = &self.axes[Dom::Cell as usize];
        if dom == Dom::Cell && cell.len() == 1 && is_scalar(&cell[0]) {
            self.axes[Dom::Cell as usize] = vec![axis.clone()];
            self.resize_dim(repeat + point, length);
        } else {
            let at = match dom { Dom::Repeat => repeat, Dom::Point => repeat + point, Dom::Cell => self.storage_axes().len() };
            self.insert_dim(at, length);
            self.axes[dom as usize].push(axis.clone());
        }
        self.selected_axis = axis.id().to_owned();
        self.normalize_table();
        Ok(())
    }

    /// `_edit_axis`: rename, resize or move an axis (with its alternative coordinates).
    fn edit_axis(&mut self, id: &str, name: &str, length: usize, unit: &str, dom: Dom) -> Result<bool> {
        let (old_dom, _position, previous) = self.location(id)?;
        let primary = previous.coordinate_of().unwrap_or(previous.id()).to_owned();
        let family: Vec<Arc<Axis>> = self.axes[old_dom as usize].iter().filter(|axis| axis.coordinate_of().unwrap_or(axis.id()) == primary).cloned().collect();
        if dom == Dom::Cell && family.len() > 1 {
            return Err("Cell-data axes cannot carry alternative coordinates; keep this axis in Repeat or Point".into());
        }
        let mut replacements: Vec<(String, Arc<Axis>)> = Vec::new();
        for axis in &family {
            let mine = axis.id() == previous.id();
            replacements.push((axis.id().to_owned(), axis_spec(axis.id(), if mine { name } else { axis.name() }, length,
                if mine { unit } else { axis.unit().unwrap_or("") }, dom, Some(axis), old_dom == dom)?));
        }
        // Every step works on a scratch copy; the working copy takes the whole result at the end.
        let mut edited = self.scratch();
        let old_storage: Vec<String> = edited.storage_axes().iter().map(|axis| axis.id().to_owned()).collect();
        let old_dim = old_storage.iter().position(|axis| *axis == primary).expect("the primary is stored");
        edited.resize_dim(old_dim, length);
        let replaced = |axis: &Arc<Axis>| replacements.iter().find(|(id, _)| id == axis.id()).map(|(_, new)| new.clone());
        if old_dom == dom {
            let changed = family.iter().any(|axis| replaced(axis).is_some_and(|new| *new != **axis));
            edited.axes[dom as usize] = edited.axes[dom as usize].iter().map(|axis| replaced(axis).unwrap_or_else(|| axis.clone())).collect();
            edited.normalize_table();
            self.take(edited);
            return Ok(changed);
        }
        edited.axes[old_dom as usize].retain(|axis| replaced(axis).is_none());
        let mut storage = old_storage;
        if old_dom == Dom::Cell && edited.axes[old_dom as usize].is_empty() { edited.axes[old_dom as usize] = vec![scalar_axis()]; }
        let target = &edited.axes[dom as usize];
        if dom == Dom::Cell && target.len() == 1 && is_scalar(&target[0]) {
            let scalar_dim = storage.iter().position(|axis| axis == SCALAR_ID).expect("the scalar carrier is stored");
            edited.take_dim(scalar_dim, 0);
            storage.remove(scalar_dim);
            edited.axes[dom as usize].clear();
        }
        edited.axes[dom as usize].extend(replacements.iter().map(|(_, axis)| axis.clone()));
        let desired: Vec<String> = edited.storage_axes().iter().map(|axis| axis.id().to_owned()).filter(|axis| storage.contains(axis)).collect();
        let order: Vec<usize> = desired.iter().map(|axis| storage.iter().position(|kept| kept == axis).expect("a stored axis")).collect();
        edited.permute(&order);
        let scalar_missing = edited.axes[Dom::Cell as usize].iter().any(|axis| is_scalar(axis)) && !storage.iter().any(|axis| axis == SCALAR_ID);
        if scalar_missing { edited.insert_dim(desired.len(), 1); }
        edited.normalize_table();
        self.take(edited);
        Ok(true)
    }

    /// `_delete_axis`: drop an axis, keeping its current Scope slice.
    fn delete_axis(&mut self, id: &str) -> Result<()> {
        let (dom, _position, axis) = self.location(id)?;
        let primary = axis.coordinate_of().unwrap_or(axis.id()).to_owned();
        let dim = self.storage_axes().iter().position(|item| item.id() == primary).expect("the primary is stored");
        self.axes[dom as usize].retain(|item| item.coordinate_of().unwrap_or(item.id()) != primary);
        let kept = self.scopes.get(&primary).copied().unwrap_or(0).min(axis.len() - 1);
        if dom == Dom::Cell && self.axes[dom as usize].is_empty() {
            self.axes[dom as usize] = vec![scalar_axis()];
            self.take_dim(dim, kept);
            self.insert_dim(self.storage_axes().len() - 1, 1);
        } else {
            self.take_dim(dim, kept);
        }
        self.selected_axis = self.visible_axes().first().map(|axis| axis.id().to_owned()).unwrap_or_default();
        self.message = format!("Removed {}; kept coordinate {}", axis.name(),
            axis.coordinate(kept).map_or_else(|| kept.to_string(), |coordinate| match coordinate { Coordinate::Text(text) => python_quote(&text), other => coordinate_text(&other) }));
        self.normalize_table();
        Ok(())
    }

    /// `_set_axis_values`.
    fn set_axis_values(&mut self, id: &str, cells: &[(usize, usize, String)]) -> Result<bool> {
        let (dom, position, axis) = self.location(id)?;
        let mut values = coordinates_of(&axis);
        let mut replacements = Vec::new();
        for (row, column, text) in cells {
            if *column != 0 || *row >= axis.len() { return Err("axis value edit lies outside the selected axis".into()); }
            replacements.push((*row, parse_axis_value(text)?));
        }
        if replacements.is_empty() { return Ok(false); }
        let mut changed = false;
        for (index, value) in replacements {
            changed = changed || values[index] != value || std::mem::discriminant(&values[index]) != std::mem::discriminant(&value);
            values[index] = value;
        }
        for (index, value) in values.iter().enumerate() {
            if values[..index].contains(value) { return Err("axis values must be unique".into()); }
        }
        if !changed { return Ok(false); }
        let replacement = Axis::new(axis.id().into(), axis.name().into(), axis.role().into(), coordinates(values), axis.unit().map(Into::into),
            axis.coordinate_frame().map(Into::into), axis.labels().map(|labels| labels.to_vec()), axis.coordinate_of().map(Into::into))?;
        self.axes[dom as usize][position] = replacement;
        Ok(true)
    }

    /// `_set_table_axis`.
    fn set_table_axis(&mut self, id: &str, mode: &str) -> Result<()> {
        self.location(id)?;
        let (mut row, mut column) = (self.row_axis.clone(), self.column_axis.clone());
        match mode {
            "rows" => { if column == id { column = if !row.is_empty() && row != id { row.clone() } else { String::new() }; } row = id.into(); }
            "columns" => { if row == id { row = if !column.is_empty() && column != id { column.clone() } else { String::new() }; } column = id.into(); }
            "scope" => { if row == id { row.clear(); } if column == id { column.clear(); } }
            other => return Err(format!("unknown table axis mode {}", python_quote(other))),
        }
        (self.row_axis, self.column_axis) = (row, column);
        self.normalize_table();
        Ok(())
    }

    /// `_table_index`: the storage index of one table cell.
    fn table_index(&self, row: usize, column: usize) -> Result<usize> {
        let mut index = Vec::new();
        for axis in self.storage_axes() {
            let at = if is_scalar(&axis) { 0 } else if axis.id() == self.row_axis { row } else if axis.id() == self.column_axis { column }
                else { self.scopes.get(axis.id()).copied().unwrap_or(0) };
            if at >= axis.len() { return Err("data edit lies outside the selected Dataset slice".into()); }
            index.push(at);
        }
        Ok(flat(&self.shape, &index))
    }

    /// `_set_dtype`: refused where a valid value would change.
    fn set_dtype(&mut self, name: &str) -> Result<bool> {
        let dtype = dtype_named(name)?;
        if dtype == self.dtype { return Ok(false); }
        let converted = self.values.cast(dtype);
        let restored = converted.cast(self.dtype);
        if (0..self.validity.len()).any(|index| self.validity[index] && !self.values.num(index).same(restored.num(index))) {
            return Err("that data type would change existing valid values".into());
        }
        self.values = converted;
        self.dtype = dtype;
        Ok(true)
    }

    /// `_set_table_cells`.
    fn set_cells(&mut self, component: &str, cells: &[(usize, usize, String)]) -> Result<bool> {
        if !matches!(component, "values" | "validity" | "sigma") { return Err(format!("unknown data table {}", python_quote(component))); }
        if component == "sigma" && self.sigma.is_none() { return Err("enable sigma before editing it".into()); }
        let mut parsed = Vec::new();
        for (row, column, text) in cells {
            let index = self.table_index(*row, *column)?;
            let raw = text.trim();
            let value = match component {
                "values" => if raw.is_empty() { None } else { Some(parse_value(raw, self.dtype)?) },
                "validity" => Some(Num::Bool(!raw.is_empty() && matches!(parse_value(raw, DType::Bool)?, Num::Bool(true)))),
                _ => {
                    let sigma = if raw.is_empty() { f64::NAN } else { python_float(raw).ok_or_else(|| format!("could not convert string to float: {}", python_quote(raw)))? };
                    if sigma < 0.0 { return Err("sample sigma must be non-negative".into()); }
                    Some(Num::Float(sigma))
                }
            };
            parsed.push((index, value));
        }
        let mut changed = false;
        for (index, value) in parsed {
            match (component, value) {
                ("values", None) => {
                    changed = changed || self.validity[index] || !self.values.num(index).same(Num::Int(0))
                        || self.sigma.as_ref().is_some_and(|sigma| !sigma[index].is_nan());
                    self.values.zero(index);
                    self.validity[index] = false;
                    if let Some(sigma) = &mut self.sigma { sigma[index] = f64::NAN; }
                }
                ("values", Some(value)) => {
                    let mut probe = Dense::zeros(self.dtype, 1);
                    probe.set(0, value);
                    changed = changed || !self.validity[index] || !self.values.num(index).same(probe.num(0));
                    self.values.set(index, value);
                    self.validity[index] = true;
                }
                ("validity", Some(Num::Bool(flag))) => { changed = changed || self.validity[index] != flag; self.validity[index] = flag; }
                (_, Some(Num::Float(sigma))) => {
                    let errors = self.sigma.as_mut().expect("sigma is on");
                    changed = changed || !(errors[index] == sigma || (errors[index].is_nan() && sigma.is_nan()));
                    errors[index] = sigma;
                }
                _ => {}
            }
        }
        Ok(changed)
    }

    fn scratch(&self) -> Self {
        Self { id: self.id.clone(), name: self.name.clone(), initial_name: self.initial_name.clone(), unit: self.unit.clone(), note: self.note.clone(),
            initial_note: self.initial_note.clone(), source_text: self.source_text.clone(), source_path: self.source_path.clone(),
            source_dataset: self.source_dataset.clone(), source_signal: self.source_signal.clone(), source_snapshot: self.source_snapshot.clone(),
            source_repeat: self.source_repeat.clone(), source_point: self.source_point.clone(), recipe: self.recipe.clone(),
            source_overlay: self.source_overlay.clone(), axes: self.axes.clone(),
            validity_contract: self.validity_contract.clone(), value_name: self.value_name.clone(), values: self.values.clone(),
            validity: self.validity.clone(), sigma: self.sigma.clone(), shape: self.shape.clone(), selected_axis: self.selected_axis.clone(),
            row_axis: self.row_axis.clone(), column_axis: self.column_axis.clone(), scopes: self.scopes.clone(), message: self.message.clone(),
            applied: self.applied.clone(), panel: self.panel.clone(), ..*self }
    }
    fn take(&mut self, edited: Self) { *self = edited; }

    // -------------------------------------------------------------- intents

    /// `data_editor_intent`: one operator intent; an error leaves the copy as it was and says why.
    pub fn intent(&mut self, op: &Value) -> Effect {
        self.message.clear();
        self.severity = "idle";
        match self.apply_intent(op) {
            Ok(effect) => effect,
            Err(error) => { self.message = error; self.severity = "error"; Effect::None }
        }
    }

    fn apply_intent(&mut self, op: &Value) -> Result<Effect> {
        let text = |key: &str| op[key].as_str().map(str::to_owned).unwrap_or_default();
        let count = |key: &str| op[key].as_u64().map(|value| value as usize).ok_or_else(|| format!("{key} must be a non-negative integer"));
        let cells = |key: &str| -> Result<Vec<(usize, usize, String)>> {
            op[key].as_array().ok_or("cells must be a list")?.iter().map(|cell| {
                let row = cell[0].as_u64().ok_or("a cell needs its row")? as usize;
                let column = cell[1].as_u64().ok_or("a cell needs its column")? as usize;
                Ok((row, column, cell[2].as_str().unwrap_or("").to_owned()))
            }).collect()
        };
        let mut dirty = true;
        match op["op"].as_str().unwrap_or("") {
            "set_dataset_field" => {
                let value = text("value");
                dirty = match op["field"].as_str().unwrap_or("") {
                    "dtype" => self.set_dtype(&value)?,
                    "name" => {
                        let name = value.trim();
                        if name.is_empty() { return Err("Dataset name cannot be blank".into()); }
                        let changed = name != self.name;
                        self.name = name.into();
                        changed
                    }
                    "unit" => { let unit = (!value.trim().is_empty()).then(|| value.trim().to_owned()); let changed = unit != self.unit; self.unit = unit; changed }
                    "note" => { let note = value.trim().to_owned(); let changed = note != self.note; self.note = note; changed }
                    other => return Err(format!("unknown Dataset field {}", python_quote(other))),
                };
            }
            "select_axis" => { self.location(&text("axis_id"))?; self.selected_axis = text("axis_id"); dirty = false; }
            "add_axis" => self.add_axis(&text("name"), count("length")?, &text("unit"), Dom::parse(&text("domain"))?)?,
            "edit_axis" => dirty = self.edit_axis(&text("axis_id"), &text("name"), count("length")?, &text("unit"), Dom::parse(&text("domain"))?)?,
            "delete_axis" => self.delete_axis(&text("axis_id"))?,
            "set_axis_values" => dirty = self.set_axis_values(&text("axis_id"), &cells("cells")?)?,
            "set_table_axis" => { self.set_table_axis(&text("axis_id"), &text("mode"))?; dirty = false; }
            "set_scope" => {
                let id = text("axis_id");
                let (_, _, axis) = self.location(&id)?;
                let index = count("index")?;
                if index >= axis.len() { return Err("scope position lies outside its Dataset axis".into()); }
                self.scopes.insert(id, index);
                dirty = false;
            }
            "set_component" => {
                self.component = match op["component"].as_str().unwrap_or("") {
                    "values" => "values", "validity" => "validity", "sigma" => "sigma",
                    other => return Err(format!("unknown data component {}", python_quote(other))),
                };
                dirty = false;
            }
            "toggle_sigma" => {
                let enabled = op["enabled"].as_bool().unwrap_or(false);
                dirty = enabled != self.sigma.is_some();
                if enabled && self.sigma.is_none() { self.sigma = Some(vec![f64::NAN; self.validity.len()]); }
                else if !enabled { self.sigma = None; if self.component == "sigma" { self.component = "values"; } }
            }
            "set_cells" => dirty = self.set_cells(&text("component"), &cells("cells")?)?,
            "view" => { self.window = [count("row")?, count("rows")?.min(400), count("column")?, count("columns")?.min(200)]; dirty = false; }
            "discard" => {
                if self.modified { self.restore()?; } else if self.unsaved { return Ok(self.discard_applied()); }
                dirty = false;
            }
            "apply_preview" => { self.note = text("note").trim().to_owned(); return self.prepare_apply().map(|apply| Effect::Apply(Box::new(apply))); }
            "save_as" => {
                self.note = text("note").trim().to_owned();
                if self.modified || self.applied.is_none() { return Err("Apply & preview the current data before saving".into()); }
                if !self.save_ready { return Err("the preview Panel has not accepted this data yet".into()); }
                return Ok(Effect::Save { editor: self.id.clone(), panel: self.panel.clone(), path: op["path"].as_str().map(str::to_owned),
                                        overwrite: op["overwrite"].as_bool().unwrap_or(false), suggested: self.suggested() });
            }
            other => return Err(format!("unknown data edit operation {}", python_quote(other))),
        }
        if dirty { self.modified = true; }
        Ok(Effect::None)
    }

    /// `_restore_data_draft`: back to what was applied (or opened).
    fn restore(&mut self) -> Result<()> {
        let (snapshot, values, name, note) = match &self.applied {
            Some((snapshot, values, name, note)) => (snapshot.clone(), values.clone(), name.clone(), note.clone()),
            None => (self.source_snapshot.clone(), None, self.initial_name.clone(), self.initial_note.clone()),
        };
        let mut restored = Self::from_snapshot(&snapshot, self.serial, &name, self.source_text.clone(), self.source_path.clone(),
            self.source_dataset.clone(), self.recipe.clone(), self.source_signal.clone())?;
        restored.note = note;
        restored.initial_name = self.initial_name.clone();
        restored.initial_note = self.initial_note.clone();
        restored.source_snapshot = self.source_snapshot.clone();
        restored.source_repeat = self.source_repeat.clone();
        restored.source_point = self.source_point.clone();
        restored.source_overlay = self.source_overlay.clone();
        if let Some(values) = values { restored.values = values; }
        restored.applied = self.applied.take();
        restored.panel = std::mem::take(&mut self.panel);
        restored.save_ready = self.save_ready;
        restored.applies = self.applies;
        restored.open = self.open;
        restored.unsaved = self.unsaved;
        restored.modified = false;
        restored.message = "Edits discarded".into();
        *self = restored;
        Ok(())
    }

    /// `_discard_applied_data`: the applied copy and its preview go; the copy is what it was opened as.
    fn discard_applied(&mut self) -> Effect {
        let panel = (!self.panel.is_empty()).then(|| std::mem::take(&mut self.panel));
        self.applied = None;
        self.save_ready = false;
        self.unsaved = false;
        let _ = self.restore();
        self.message = "Applied working copy discarded".into();
        Effect::Discarded { panel, signal: self.signal() }
    }

    /// `close_data_editor` once agreed: unsaved Applies and edits go; the tab closes.
    pub fn close(&mut self) -> Effect {
        let effect = if self.unsaved { self.discard_applied() } else { if self.modified { let _ = self.restore(); } Effect::None };
        self.open = false;
        effect
    }

    fn suggested(&self) -> String {
        let mut base = String::new();
        let mut replacing = false;
        for character in self.name.chars() {
            let kept = character.is_ascii_alphanumeric() || matches!(character, '_' | '.' | '-');
            if kept { base.push(character) } else if !replacing { base.push('-') }
            replacing = !kept;
        }
        let base = base.trim_matches(['.', '-']);
        let file = format!("{}.npz", if base.is_empty() { "figure" } else { base });
        match &self.source_path {
            Some(path) => std::path::Path::new(path).parent().map_or(file.clone(), |folder| folder.join(&file).display().to_string()),
            None => file,
        }
    }

    // -------------------------------------------------------------- apply

    /// `_draft_schema`: the edited shape as a Dataset schema.
    fn schema(&self) -> Result<Arc<DatasetSchema>> {
        let repeat = edited_domain(&self.axes[0], &self.source_repeat)?;
        let point = edited_domain(&self.axes[1], &self.source_point)?;
        let cell_axes = &self.axes[2];
        let cell = Domain::dense(cell_axes.iter().filter(|axis| axis.coordinate_of().is_none()).map(|axis| axis.len()).collect(), cell_axes.clone())?;
        let contract = if cell_axes.len() == 1 && is_scalar(&cell_axes[0]) { ValidityContract::Value } else {
            let mut required: Vec<String> = match &self.validity_contract {
                ValidityContract::Components(ids) => ids.iter().filter(|id| cell_axes.iter().any(|axis| axis.id() == id.as_ref())).map(|id| id.to_string()).collect(),
                ValidityContract::Value => Vec::new(),
            };
            // A Cell axis the validity varies along is one validity resolves.
            let base = self.primary_count(Dom::Repeat) + self.primary_count(Dom::Point);
            for (offset, axis) in cell_axes.iter().filter(|axis| axis.coordinate_of().is_none()).enumerate() {
                let dim = base + offset;
                let varies = (0..self.validity.len()).any(|at| {
                    let mut index = unflat(&self.shape, at);
                    index[dim] = 0;
                    self.validity[at] != self.validity[flat(&self.shape, &index)]
                });
                if varies && !required.iter().any(|id| id == axis.id()) { required.push(axis.id().to_owned()); }
            }
            let ordered: Vec<Arc<str>> = cell_axes.iter().filter(|axis| required.iter().any(|id| id == axis.id())).map(|axis| axis.id().into()).collect();
            if ordered.is_empty() { ValidityContract::Value } else { ValidityContract::Components(ordered) }
        };
        let name = self.value_name.clone().unwrap_or_else(|| self.name.clone());
        DatasetSchema::new(repeat, point, cell, ValueSchema::new(self.dtype, self.unit.as_deref().map(Into::into), Some(name.into()), contract)?)
    }

    /// `_manual_snapshot`: the copy packed onto its carriers (invalid samples published as zero).
    fn snapshot(&self) -> Result<(Snapshot, bool)> {
        if let Some(sigma) = &self.sigma { if sigma.iter().any(|value| value.is_finite() && *value < 0.0) { return Err("sample sigma must be non-negative".into()); } }
        let schema = self.schema()?;
        let repeat_rows = domain_rows(schema.domain(DomainRole::Repeat))?;
        let point_rows = domain_rows(schema.domain(DomainRole::Point))?;
        let (repeats, points) = (repeat_rows.len(), point_rows.len());
        let cells = schema.domain(DomainRole::Cell).size();
        let logical_points: usize = self.axes[1].iter().filter(|axis| axis.coordinate_of().is_none()).map(|axis| axis.len()).product();
        let source = |row: usize, point: usize| (repeat_rows[row] * logical_points + point_rows[point]) * cells;
        let mut values = Dense::zeros(self.dtype, repeats * points * cells);
        let mut validity = vec![false; repeats * points * cells];
        let mut sigma = self.sigma.as_ref().map(|_| vec![f64::NAN; repeats * points * cells]);
        let mut zeroed = false;
        for row in 0..repeats {
            for point in 0..points {
                for cell in 0..cells {
                    let (from, to) = (source(row, point) + cell, (row * points + point) * cells + cell);
                    if self.validity[from] { values.set(to, self.values.num(from)); } else { zeroed = true; }
                    validity[to] = self.validity[from];
                    if let (Some(target), Some(errors)) = (&mut sigma, &self.sigma) { target[to] = errors[from]; }
                }
            }
        }
        let mask = match schema.value().validity() {
            ValidityContract::Value => (0..repeats * points).map(|sample| validity[sample * cells]).collect::<Vec<bool>>(),
            ValidityContract::Components(ids) => {
                let cell_shape = schema.domain(DomainRole::Cell).physical_shape().to_vec();
                let cell_ids: Vec<&str> = schema.domain(DomainRole::Cell).axes().iter().filter(|axis| axis.coordinate_of().is_none()).map(|axis| axis.id()).collect();
                let kept: Vec<usize> = ids.iter().map(|id| cell_ids.iter().position(|axis| *axis == id.as_ref()).expect("a component axis")).collect();
                let marks: usize = kept.iter().map(|dim| cell_shape[*dim]).product();
                let mark_shape: Vec<usize> = kept.iter().map(|dim| cell_shape[*dim]).collect();
                (0..repeats * points * marks).map(|at| {
                    let (sample, mark) = (at / marks, at % marks);
                    let mark_index = unflat(&mark_shape, mark);
                    let mut cell_index = vec![0; cell_shape.len()];
                    for (position, dim) in kept.iter().enumerate() { cell_index[*dim] = mark_index[position]; }
                    validity[sample * cells + flat(&cell_shape, &cell_index)]
                }).collect()
            }
        };
        let validity = if mask.iter().all(|flag| *flag) { Validity::Uniform(true) } else if !mask.iter().any(|flag| *flag) { Validity::Uniform(false) }
            else { Validity::Mask(OwnedBuffer::from_vec(mask)) };
        let block = DataBlock::new(schema.clone(), values.buffer(), validity, sigma.map(OwnedBuffer::from_vec))?;
        Ok((DatasetStore::new(schema).commit(vec![Placement::new([0, 0], [repeats, points], block)?])?, zeroed))
    }

    /// `_apply_data_draft`, the part that is the copy's: what to publish and where it shows.
    fn prepare_apply(&mut self) -> Result<Apply> {
        let (snapshot, _zeroed) = self.snapshot()?;
        let plot = zlc_plot::configuration::PlotSchema::new(snapshot.schema().clone());
        let admits = |target: &Value| -> bool {
            // v2 `_recipe_admits`: the saved spec still describes this Dataset.
            let Some(spec) = target.get("spec") else { return false };
            let Ok(spec) = zlc_plot::configuration::recipe::decode_spec(&zlc_plot::configuration::value::Value::from_json(spec)) else { return false };
            zlc_plot::configuration::describe_semantics(Some(&plot), &spec, &zlc_plot::configuration::Defaults::default().layout).is_ok()
        };
        let recipe = self.recipe.clone().filter(admits);
        if recipe.is_none() && zlc_plot::configuration::fitting_spec(Some(&plot), None, None).ok().flatten().is_none() {
            return Err("no plot kind can show data of this shape".into());
        }
        let operation = if self.applied.is_some() || self.source_signal.is_some() { "manual-edit" } else { "manual-create" };
        let mut record = json!({
            "node": format!("manual-data-{}", self.serial), "operation": operation,
            "software": {"distribution": "zou-lab-control", "version": env!("CARGO_PKG_VERSION")},
            "dataset": {"name": self.name, "shape": snapshot.schema().physical_shape(), "dtype": dtype_name(self.dtype)},
            "note": self.note,
        });
        if let Some(path) = &self.source_path { record["input"] = json!({"archive": path, "dataset": self.source_dataset}); }
        let (old, new) = (self.source_snapshot.schema(), snapshot.schema());
        let same = [DomainRole::Repeat, DomainRole::Point, DomainRole::Cell].into_iter().all(|role| old.domain(role) == new.domain(role));
        let overlay = self.source_overlay.clone().filter(|_| same);
        Ok(Apply { editor: self.id.clone(), owner: format!("manual-data-{}", self.serial), signal: self.signal(), snapshot, overlay,
            metadata: record, parent: if self.applied.is_some() { Some(self.signal()) } else { self.source_signal.clone() },
            panel: (!self.panel.is_empty()).then(|| self.panel.clone()), recipe })
    }

    /// The Apply landed: its preview panel is `panel`.
    pub fn applied_as(&mut self, snapshot: Snapshot, panel: String) {
        let replaced = (0..self.validity.len()).any(|index| !self.validity[index]).then(|| self.values.clone());
        self.applied = Some((snapshot, replaced, self.name.clone(), self.note.clone()));
        self.applies += 1;
        self.modified = false;
        self.unsaved = true;
        self.save_ready = false;
        self.message = "Applied; preparing the preview Panel".into();
        self.severity = "idle";
        self.panel = panel;
    }

    pub fn refused(&mut self, error: String) { self.message = error; self.severity = "error"; }

    /// The preview panel has (or no longer has) drawn the applied data.
    pub fn set_save_ready(&mut self, ready: bool) -> bool {
        if ready == self.save_ready { return false; }
        self.save_ready = ready;
        if ready && !self.modified { self.message = "Preview ready".into(); self.severity = "idle"; }
        true
    }

    pub fn saved(&mut self, archive: &str, current: bool) {
        if current { self.unsaved = false; }
        self.message = if current { format!("Saved {archive}") } else { format!("Saved older preview to {archive}; current data is not saved") };
        self.severity = if current { "idle" } else { "warning" };
    }

    pub fn warn(&mut self, text: &str) { self.message = text.into(); self.severity = "warning"; }

    // -------------------------------------------------------------- projection

    /// `_data_projection`: everything the editor page shows.
    pub fn projection(&mut self) -> Value {
        let axes: Vec<Value> = Dom::ALL.iter().flat_map(|dom| self.axes[*dom as usize].iter().filter(|axis| !is_scalar(axis)).map(move |axis| json!({
            "id": axis.id(), "name": axis.name(), "size": axis.len(), "unit": axis.unit().unwrap_or(""), "domain": dom.key(), "domain_label": dom.label(),
        }))).collect();
        if !axes.iter().any(|axis| axis["id"] == self.selected_axis.as_str()) {
            self.selected_axis = axes.first().map(|axis| axis["id"].as_str().unwrap_or("").to_owned()).unwrap_or_default();
        }
        let coordinates: Vec<String> = self.location(&self.selected_axis).map(|(_, _, axis)| coordinates_of(&axis).iter().map(coordinate_text).collect()).unwrap_or_default();
        // `_validate_draft_arrays`: one value of the declared dtype and one validity flag per element.
        let can_apply = self.values.dtype() == self.dtype && self.values.len() == product(&self.shape) && self.validity.len() == product(&self.shape);
        let message = if self.message.is_empty() { String::new() } else { sentence(&self.message) };
        json!({
            "dataset": {"name": self.name, "dtype": dtype_name(self.dtype), "unit": self.unit.clone().unwrap_or_default(), "note": self.note,
                        "source": self.source_text, "dtype_choices": DTYPES.iter().map(|(_, name)| json!([name, name])).collect::<Vec<_>>()},
            "domain_choices": Dom::ALL.iter().map(|dom| json!([dom.key(), dom.label()])).collect::<Vec<_>>(),
            "axes": axes,
            "selected_axis": self.selected_axis,
            "axis_values": {"values": coordinates, "blank_hint": "Axis values cannot be blank."},
            "table": self.table(),
            "dirty": self.dirty(), "modified": self.modified, "can_apply": can_apply,
            "can_save": self.save_ready && !self.modified, "save_suggested": self.suggested(),
            "message": message, "message_severity": self.severity,
        })
    }

    /// `_table_projection`: the 2-D slice the Data card shows, as text, for the rows and columns on screen.
    fn table(&mut self) -> Value {
        self.normalize_table();
        let storage = self.storage_axes();
        let dim = |id: &str| storage.iter().position(|axis| axis.id() == id);
        let (row_dim, column_dim) = (dim(&self.row_axis), dim(&self.column_axis));
        let axis_of = |at: Option<usize>| at.map(|at| storage[at].clone());
        let (row_axis, column_axis) = (axis_of(row_dim), axis_of(column_dim));
        let (rows, columns) = (row_axis.as_ref().map_or(1, |axis| axis.len()), column_axis.as_ref().map_or(1, |axis| axis.len()));
        let mut base = vec![0; storage.len()];
        for (at, axis) in storage.iter().enumerate() {
            if Some(at) != row_dim && Some(at) != column_dim && !is_scalar(axis) { base[at] = self.scopes.get(axis.id()).copied().unwrap_or(0); }
        }
        let [first_row, row_count, first_column, column_count] = self.window;
        let first_row = first_row.min(rows.saturating_sub(1));
        let first_column = first_column.min(columns.saturating_sub(1));
        let (last_row, last_column) = ((first_row + row_count).min(rows), (first_column + column_count).min(columns));
        let component = if self.component == "sigma" && self.sigma.is_none() { "values" } else { self.component };
        let block: Vec<Vec<String>> = (first_row..last_row).map(|row| (first_column..last_column).map(|column| {
            let mut index = base.clone();
            if let Some(at) = row_dim { index[at] = row; }
            if let Some(at) = column_dim { index[at] = column; }
            let at = flat(&self.shape, &index);
            match component {
                "validity" => self.validity[at].text(),
                "sigma" => {
                    let sigma = self.sigma.as_ref().map_or(f64::NAN, |sigma| sigma[at]);
                    if self.validity[at] && sigma.is_finite() { sigma.text() } else { String::new() }
                }
                _ => if self.validity[at] { self.values.text(at) } else { String::new() },
            }
        }).collect()).collect();
        let headers = |axis: &Option<Arc<Axis>>| axis.as_ref().map_or_else(|| vec!["Value".to_owned()], |axis| coordinates_of(axis).iter().map(coordinate_text).collect());
        let axis_rows: Vec<Value> = self.visible_axes().iter().map(|axis| {
            let id = axis.id();
            let mode = if id == self.row_axis { "rows" } else if id == self.column_axis { "columns" } else { "scope" };
            let index = self.scopes.get(id).copied().unwrap_or(0).min(axis.len() - 1);
            json!({"axis_id": id, "name": axis.name(), "size": axis.len(), "unit": axis.unit().unwrap_or(""), "mode": mode, "index": index,
                   "scope_label": scope_label(axis, index)})
        }).collect();
        let structure: Vec<Vec<Value>> = self.axes.iter().map(|axes| axes.iter().filter(|axis| !is_scalar(axis) && axis.coordinate_of().is_none())
            .map(|axis| json!([axis.name(), axis.len()])).collect()).collect();
        json!({
            "component": component,
            "component_choices": [["values", "Values", true], ["validity", "Validity", true], ["sigma", "Sigma", self.sigma.is_some()]],
            "sigma_enabled": self.sigma.is_some(),
            "shape": [rows, columns],
            "block": {"row": first_row, "column": first_column, "cells": block},
            "blank_help": match component { "validity" => "Blank means False.", "sigma" => "Blank means no stated sigma.", _ => "Blank deletes this sample and marks it invalid." },
            "row_headers": headers(&row_axis), "column_headers": headers(&column_axis),
            "structure": structure, "axes": axis_rows,
        })
    }
}

/// One Figure Viewer's working copies, in the order they were opened, and
/// what their Applies published (the parents the next Apply descends from).
#[derive(Default)]
pub struct Drafts {
    serial: u64,
    pub list: Vec<Draft>,
    pub published: BTreeMap<String, Arc<zlc_core::runtime::Publication>>,
    /// The panel epoch each Apply's preview reaches once it shows that data, by editor.
    pub epochs: BTreeMap<String, u64>,
    /// A Save as… in flight: its request, its editor, which Apply it saves, and the archive it saves to.
    pub saving: Option<(u64, String, u64, String)>,
}

impl Drafts {
    /// `new_data`: one immediately useful Curve working copy.
    pub fn new_data(&mut self) -> Result<&mut Draft> {
        self.serial += 1;
        self.list.push(Draft::new_curve(self.serial)?);
        Ok(self.list.last_mut().expect("just made"))
    }
    /// `edit_data`: the copy already open on `signal` (an applied copy's own, or a
    /// dataset's), else a new copy of the dataset `source` describes.
    pub fn edit(&mut self, signal: &str, source: impl FnOnce() -> Result<EditSource>) -> Result<&mut Draft> {
        if let Some(at) = self.list.iter().position(|draft| (draft.applied() && draft.signal() == signal) || draft.source_signal.as_deref() == Some(signal)) {
            return Ok(&mut self.list[at]);
        }
        self.serial += 1;
        self.list.push(Draft::edit(self.serial, source()?)?);
        Ok(self.list.last_mut().expect("just made"))
    }
    pub fn get_mut(&mut self, editor: &str) -> Option<&mut Draft> { self.list.iter_mut().find(|draft| draft.id == editor) }

    /// The applied copies the Data chooser offers: `(signal, name)`.
    pub fn offered(&self) -> Vec<(String, String)> {
        self.list.iter().filter(|draft| draft.applied()).map(|draft| (draft.signal(), draft.name().to_owned())).collect()
    }
    /// What the viewer page shows of them: one tab each.
    pub fn state(&mut self) -> Value {
        Value::Array(self.list.iter_mut().filter(|draft| draft.open).map(|draft| json!({
            "id": draft.id, "title": format!("Data · {}", draft.name()), "dirty": draft.dirty(), "projection": draft.projection(),
        })).collect())
    }
    /// Whether each preview has drawn the data its copy applied (v2 `_refresh_data_save_states`).
    pub fn settle(&mut self, panels: &Value) -> bool {
        let mut changed = false;
        for draft in &mut self.list {
            let Some(epoch) = self.epochs.get(&draft.id) else { continue };
            let row = panels.as_array().and_then(|rows| rows.iter().find(|row| row["id"] == draft.panel.as_str()));
            let ready = row.is_some_and(|row| row["data_binding_epoch"].as_u64() == Some(*epoch) && !row["frame"].is_null() && row["error"].is_null());
            changed |= draft.set_save_ready(ready);
        }
        changed
    }
}

fn unflat(shape: &[usize], mut at: usize) -> Vec<usize> {
    let mut index = vec![0; shape.len()];
    for dim in (0..shape.len()).rev() { index[dim] = at % shape[dim]; at /= shape[dim]; }
    index
}

/// `zlc_ui.status.sentence`: ends in a full stop.
fn sentence(text: &str) -> String {
    let text = text.trim_end();
    if text.is_empty() || text.ends_with(['.', '?', '!', '…']) { text.into() } else { format!("{text}.") }
}

/// `_expand_snapshot_for_edit`: the dataset's planes as one dense logical tensor
/// (repeat axes, point axes, cell axes), with validity per element.
fn expand(snapshot: &Snapshot) -> Result<(Dense, Vec<bool>, Option<Vec<f64>>, Vec<usize>)> {
    let schema = snapshot.schema();
    let planes = zlc_archive::manifest::dense_planes(&Source::of_snapshot(snapshot));
    let (repeat, point, cell) = (schema.domain(DomainRole::Repeat), schema.domain(DomainRole::Point), schema.domain(DomainRole::Cell));
    let logical = |domain: &Domain| domain.axes().iter().filter(|axis| axis.coordinate_of().is_none()).map(|axis| axis.len()).collect::<Vec<_>>();
    let (repeat_shape, point_shape) = (logical(repeat), logical(point));
    let cell_shape = cell.physical_shape().to_vec();
    let shape: Vec<usize> = repeat_shape.iter().chain(&point_shape).chain(&cell_shape).copied().collect();
    let (repeat_rows, point_rows) = (domain_rows(repeat)?, domain_rows(point)?);
    let (points, cells, logical_points) = (point.size(), cell.size(), product(&point_shape));
    let source_values = Dense::of(&planes.values);
    let mut values = Dense::zeros(schema.dtype(), product(&shape));
    let mut validity = vec![false; product(&shape)];
    let mut sigma = planes.sigma.as_ref().map(|_| vec![f64::NAN; product(&shape)]);
    let marks = planes.mask.as_ref().map(|mask| mask.len() / (repeat.size() * points).max(1));
    let component_dims: Vec<usize> = match schema.value().validity() {
        ValidityContract::Value => Vec::new(),
        ValidityContract::Components(ids) => ids.iter().map(|id| cell.axes().iter().filter(|axis| axis.coordinate_of().is_none())
            .position(|axis| axis.id() == id.as_ref()).expect("a component axis")).collect(),
    };
    let mark_shape: Vec<usize> = component_dims.iter().map(|dim| cell_shape[*dim]).collect();
    for row in 0..repeat.size() {
        for column in 0..points {
            let sample = row * points + column;
            let target = (repeat_rows[row] * logical_points + point_rows[column]) * cells;
            for item in 0..cells {
                let (from, to) = (sample * cells + item, target + item);
                values.set(to, source_values.num(from));
                validity[to] = match (&planes.mask, marks) {
                    (Some(mask), Some(marks)) => {
                        let cell_index = unflat(&cell_shape, item);
                        let mark: Vec<usize> = component_dims.iter().map(|dim| cell_index[*dim]).collect();
                        mask[sample * marks + if mark.is_empty() { 0 } else { flat(&mark_shape, &mark) }]
                    }
                    _ => planes.valid,
                };
                if let (Some(target), Some(errors)) = (&mut sigma, &planes.sigma) { target[to] = errors[from]; }
            }
        }
    }
    Ok((values, validity, sigma, shape))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image() -> Snapshot {
        let axis = |id: &str, role: &str, len: usize| Axis::new(id.into(), id.into(), role.into(), Coordinates::Index { origin: 0, len }, Some("pixel".into()), None, None, None).unwrap();
        let schema = DatasetSchema::new(mapped_domain(&[axis("repeat", "repeat", 1)]).unwrap(), Domain::new(vec![1], vec![], vec![]).unwrap(),
            Domain::dense(vec![2, 3], vec![axis("y", "spatial-y", 2), axis("x", "spatial-x", 3)]).unwrap(),
            ValueSchema::new(DType::F64, None, Some("site_map".into()), ValidityContract::Value).unwrap()).unwrap();
        let block = DataBlock::new(schema.clone(), OwnedBuffer::from_vec(vec![1.5, 2.0, 3.0, 4.0, 5.0, 6.25]).into(), Validity::Uniform(true), None).unwrap();
        DatasetStore::new(schema).commit(vec![Placement::new([0, 0], [1, 1], block).unwrap()]).unwrap()
    }

    fn cells(projection: &Value) -> Vec<Vec<String>> {
        serde_json::from_value(projection["table"]["block"]["cells"].clone()).unwrap()
    }

    #[test]
    fn a_copy_shows_its_slice_and_packs_back_unchanged() {
        let source = EditSource { snapshot: image(), name: "Site map".into(), path: "C:/data/site_map.npz".into(), dataset: "data".into(), recipe: None, signal: "@figure/1/data".into(), overlay: None };
        let mut draft = Draft::edit(1, source).unwrap();
        let projection = draft.projection();
        assert_eq!(projection["dataset"]["source"], "Copy of site_map.npz · data");
        assert_eq!(projection["table"]["shape"], json!([2, 3]));
        assert_eq!(cells(&projection), vec![vec!["1.5", "2.0", "3.0"], vec!["4.0", "5.0", "6.25"]]);
        assert_eq!(projection["axes"][0]["domain_label"], "Repeat");
        let (snapshot, _) = draft.snapshot().unwrap();
        assert!(**snapshot.schema() == **image().schema());
        let (values, ..) = expand(&snapshot).unwrap();
        assert_eq!((0..6).map(|at| values.text(at)).collect::<Vec<_>>(), ["1.5", "2.0", "3.0", "4.0", "5.0", "6.25"]);
    }

    #[test]
    fn edits_follow_v2_rules() {
        let mut draft = Draft::new_curve(2).unwrap();
        assert_eq!(draft.name(), "Manual data 2");
        assert!(draft.dirty());
        // The repeat axis holds the rows, x the columns (v2's 1 x 16 table).
        draft.intent(&json!({"op": "set_cells", "component": "values", "cells": [[0, 0, "2.5"], [0, 1, ""]]}));
        let projection = draft.projection();
        assert_eq!(projection["table"]["shape"], json!([1, 16]));
        assert_eq!(cells(&projection)[0][0], "2.5");
        assert_eq!(cells(&projection)[0][1], "");
        draft.intent(&json!({"op": "add_axis", "name": "y", "length": 3, "unit": "", "domain": "cell_data"}));
        assert_eq!(draft.shape, vec![1, 16, 3]);
        draft.intent(&json!({"op": "set_dataset_field", "field": "dtype", "value": "int8"}));
        assert_eq!(draft.projection()["message"], "that data type would change existing valid values.");
        draft.intent(&json!({"op": "edit_axis", "axis_id": "manual.x", "name": "x", "length": 4, "unit": "ms", "domain": "point"}));
        assert_eq!(draft.shape, vec![1, 4, 3]);
        draft.intent(&json!({"op": "delete_axis", "axis_id": "manual.y"}));
        assert_eq!(draft.shape, vec![1, 4, 1]);
        assert_eq!(draft.projection()["message"], "Removed y; kept coordinate 0.");
        draft.intent(&json!({"op": "set_axis_values", "axis_id": "manual.x", "cells": [[0, 0, "1"], [1, 0, "1"]]}));
        assert_eq!(draft.projection()["message"], "axis values must be unique.");
        let apply = match draft.intent(&json!({"op": "apply_preview", "note": "  why "})) { Effect::Apply(apply) => apply, _ => panic!("apply") };
        assert_eq!(apply.signal, "@figure/manual/2/data");
        assert_eq!(apply.metadata["operation"], "manual-create");
        assert_eq!(apply.metadata["note"], "why");
    }

    /// Save as… writes over no file the operator has not agreed to, as the document writes. The
    /// window's Save dialog agreed to the archive it chose: the picture beside it already there is
    /// refused, named alone, until the window's Overwrite saves that archive again told to; a named
    /// archive (a remote client's) is refused naming every file there until told to overwrite.
    /// A refused save changes no file.
    #[test]
    fn save_as_writes_over_only_what_the_operator_agreed_to() {
        use std::path::Path;
        use zlc_core::runtime::{Bundle, Declaration, Input, Runtime, Storage};
        use zlc_plot::document::{DocumentRead, FigureDocument};

        #[derive(Clone)]
        struct Frame(Arc<zlc_plot::projection::Source>);
        impl DocumentRead for Frame {
            fn source(&self) -> &Arc<zlc_plot::projection::Source> { &self.0 }
            fn provenance_json(&self) -> String { "null".into() }
        }
        // A 2 x 3 image committed to a Runtime and read back as the viewer reads an Apply.
        let axis = |id: &str, role: &str, len: usize| Axis::new(id.into(), id.into(), role.into(), Coordinates::Index { origin: 0, len }, None, None, None, None).unwrap();
        let one = |id: &str, role: &str| Domain::dense(vec![1], vec![axis(id, role, 1)]).unwrap();
        let schema = DatasetSchema::new(one("repeat", "repeat"), one("frame", "readout-event"), Domain::dense(vec![2, 3], vec![axis("y", "spatial-y", 2), axis("x", "spatial-x", 3)]).unwrap(),
                                        ValueSchema::new(DType::F64, None, None, ValidityContract::Value).unwrap()).unwrap();
        let runtime = Runtime::new();
        let generation = runtime.begin("manual", vec![Declaration { signal: "data".into(), schema: schema.clone(), storage: Storage::Latest }], json!({})).unwrap();
        let data = DataBlock::new(schema, OwnedBuffer::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).into(), Validity::Uniform(true), None).unwrap();
        runtime.commit(&generation, &Bundle { inputs: vec![Input::Event { signal: "data".into(), ordinal: 0, data, origin: None }], metadata: json!({}), parents: vec![] }).unwrap();
        let read = runtime.read_coherent_views(&[(generation, Arc::from("data"), None)], &mut Default::default()).pop().unwrap().unwrap();
        let frame = Frame(Arc::new(zlc_plot::projection::Source::from_read(&read).unwrap()));

        let mut document = FigureDocument::new(crate::pulse_host::services(), Some(Arc::new(zlc_plot::figure::ArchiveSaver)));
        let all: std::collections::BTreeSet<String> = ["preview".to_owned()].into();
        document.execute(r#"{"id": 0, "action": "ready", "dpr": 1.0}"#, Default::default(), None, None).unwrap();
        document.finish_presentation(&all, &all);
        let presented = json!({"id": 1, "action": "sync_panels", "inputs": [{"panel": "preview", "kind": "image", "window": 1, "signal": "data", "epoch": 1}],
                               "targets": {"preview": {"semantic": {"kind": "image"}}}});
        document.execute(&presented.to_string(), [("preview".to_owned(), frame)].into(), None, None).unwrap();
        document.finish_presentation(&all, &all);

        let folder = std::env::temp_dir().join(format!("zlc_save_as_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&folder);
        std::fs::create_dir_all(&folder).unwrap();
        let folder = zlc_archive::durable::resolve(&folder).unwrap();
        let (archive, image) = (folder.join("scan.npz"), folder.join("scan.png"));
        let named = |paths: &[&std::path::PathBuf]| json!(paths.iter().map(|path| path.display().to_string()).collect::<Vec<_>>());
        let mut request = 1;
        // The host's save of the archive `archive`, as main.rs queues it, run by the document.
        let mut save = |asked: bool, overwrite: bool| -> Value {
            request += 1;
            let mut command = save_command(&json!(request), "preview", 1, &archive.display().to_string(), asked, overwrite);
            assert_eq!(command["path"], json!(image.display().to_string()), "the picture beside the archive");
            // Which document it is for is the host's to read (main.rs), not the document's.
            assert_eq!(command.as_object_mut().unwrap().remove("document"), Some(json!(1)));
            let state = document.execute(&command.to_string(), Default::default(), None, None).unwrap().state;
            document.finish_presentation(&all, &all);
            assert!(state["error"].is_null(), "{}", state["error"]);
            state["saved"].clone()
        };
        let disk = |path: &Path| std::fs::read(path).ok().map(|bytes| (bytes, std::fs::metadata(path).unwrap().modified().unwrap()));
        let plant = |path: &Path, bytes: &[u8]| std::fs::write(path, bytes).unwrap();
        let real = |path: &Path, magic: &[u8]| disk(path).is_some_and(|(bytes, _)| bytes.starts_with(magic));

        // The window: the Save dialog chose (and asked about) scan.npz; scan.png beside it is there too.
        plant(&archive, b"an older archive");
        plant(&image, b"an older picture");
        let before = (disk(&archive), disk(&image));
        assert_eq!(save(true, false), json!({"exists": named(&[&image])}), "only the picture is asked about");
        assert!((disk(&archive), disk(&image)) == before, "a refused save wrote");
        // Overwrite: the same archive, named, told to.
        assert_eq!(save(false, true), named(&[&image, &archive]), "written once agreed");
        assert!(real(&image, b"\x89PNG") && real(&archive, b"PK"), "a picture and an archive");
        // Only the archive the dialog asked about is there: written.
        std::fs::remove_file(&image).unwrap();
        plant(&archive, b"an older archive");
        assert_eq!(save(true, false), named(&[&image, &archive]), "the dialog's answer is enough");
        assert!(real(&image, b"\x89PNG") && real(&archive, b"PK"), "a picture and an archive");

        // A named archive (a remote client's): refused naming every file there, written when told to.
        plant(&archive, b"an older archive");
        plant(&image, b"an older picture");
        let before = (disk(&archive), disk(&image));
        assert_eq!(save(false, false), json!({"exists": named(&[&image, &archive])}), "both named");
        assert!((disk(&archive), disk(&image)) == before, "a refused save wrote");
        std::fs::remove_file(&image).unwrap();
        let before = disk(&archive);
        assert_eq!(save(false, false), json!({"exists": named(&[&archive])}), "the archive named");
        assert!(disk(&archive) == before && disk(&image).is_none(), "a refused save wrote");
        assert_eq!(save(false, true), named(&[&image, &archive]), "overwrite: true writes");
        assert!(real(&image, b"\x89PNG") && real(&archive, b"PK"), "a picture and an archive");
        let _ = std::fs::remove_dir_all(&folder);
    }

    #[test]
    fn values_read_as_numpy_writes_them() {
        assert_eq!(1e-5_f32.text(), "1e-05");
        assert_eq!(2.6749_f32.text(), "2.6749");
        assert_eq!(Complex128 { re: 1.0, im: 2.0 }.text(), "(1+2j)");
        assert_eq!(Complex128 { re: 0.0, im: -2.5 }.text(), "-2.5j");
        assert!(matches!(python_complex("1-2j"), Some((re, im)) if re == 1.0 && im == -2.0));
        assert!(matches!(parse_value("300", DType::U8), Err(message) if message == "'300' is not a uint8 value"));
        assert_eq!(half_bits(1.0), 0x3c00);
        assert_eq!(half_bits(65504.0), 0x7bff);
        assert_eq!(half_bits(1e-7), 0x0002);
    }
}
