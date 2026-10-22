//! NPY members as `numpy.lib.format` writes and reads them: the padded
//! header dictionary, the data in the array's own byte and memory order,
//! and everything `read_array` does with a member, malformed or not —
//! header parsing (`npdtype`, `pyliteral`), the element count NumPy
//! computes, its allocation, its chunked reads, and its reshape.

use std::borrow::Cow;

use zlc_core::{AnyBuffer, Complex128, Complex64, DType, F16, OwnedBuffer, Scalar};

use crate::error::{ArchiveError, ErrorKind, Result};
use crate::json::{py_repr_bytes, py_repr_str};
use crate::npdtype::{descr_to_dtype, Kind};
use crate::pyliteral::{filter_header, literal_eval, sorted, EvalError, Int, Obj, TokenFailure};
use crate::pystr;

pub use crate::npdtype::Dtype as NpyType;

pub const MAGIC: &[u8; 6] = b"\x93NUMPY";
const ARRAY_ALIGN: usize = 64;
const GROWTH_AXIS_MAX_DIGITS: usize = 21;
const MAX_HEADER_SIZE: usize = 10000;
const BUFFER_SIZE: usize = 1 << 18;

impl NpyType {
    pub fn native(dtype: DType) -> Self { NpyType::numeric(dtype, false) }
}

/// One member's values in C order: numeric planes in native byte order,
/// UTF-32 code units (`U`), or the stored bytes of any other dtype.
#[derive(Clone)]
pub enum NpyData { Numeric(AnyBuffer), Text(Vec<u32>), Raw(Vec<u8>) }

/// One NPY array: its dtype, logical shape and values, and whether NumPy
/// holds it Fortran-ordered (so writes it back that way).
#[derive(Clone)]
pub struct NpyArray { pub dtype: NpyType, pub shape: Vec<usize>, pub data: NpyData, pub fortran: bool }

/// Native little-endian values of a padding-free scalar whose every bit
/// pattern is valid, copied once from their stored bytes.
fn pod<T: Scalar + Default>(bytes: &[u8]) -> Vec<T> {
    let count = bytes.len() / std::mem::size_of::<T>();
    let mut values = vec![T::default(); count];
    // SAFETY: T is one of zlc-core's sealed storage scalars (no padding, any
    // bit pattern valid apart from bool, which never comes here) and the
    // destination holds exactly `count * size_of::<T>()` bytes.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), values.as_mut_ptr().cast::<u8>(), count * std::mem::size_of::<T>()); }
    values
}

macro_rules! from_le {
    ($bytes:expr, $swapped:expr, $ty:ty, $read:expr) => {{
        let size = std::mem::size_of::<$ty>();
        let read = $read;
        if cfg!(target_endian = "little") && !$swapped { return AnyBuffer::from(OwnedBuffer::from_vec(pod::<$ty>($bytes))); }
        let values: Vec<$ty> = $bytes.chunks_exact(size).map(|chunk| {
            let mut raw = [0u8; std::mem::size_of::<$ty>()];
            raw.copy_from_slice(chunk);
            if $swapped { raw.reverse(); }
            read(raw)
        }).collect();
        AnyBuffer::from(OwnedBuffer::from_vec(values))
    }};
}

/// Values of `dtype` from their stored bytes.
pub fn numeric_from_bytes(bytes: &[u8], dtype: DType, swapped: bool) -> AnyBuffer {
    match dtype {
        DType::Bool => AnyBuffer::from(OwnedBuffer::from_vec(bytes.iter().map(|byte| *byte != 0).collect::<Vec<bool>>())),
        DType::U8 => AnyBuffer::from(OwnedBuffer::from_vec(bytes.to_vec())),
        DType::I8 => AnyBuffer::from(OwnedBuffer::from_vec(bytes.iter().map(|byte| *byte as i8).collect::<Vec<i8>>())),
        DType::U16 => from_le!(bytes, swapped, u16, u16::from_le_bytes),
        DType::I16 => from_le!(bytes, swapped, i16, i16::from_le_bytes),
        DType::U32 => from_le!(bytes, swapped, u32, u32::from_le_bytes),
        DType::I32 => from_le!(bytes, swapped, i32, i32::from_le_bytes),
        DType::U64 => from_le!(bytes, swapped, u64, u64::from_le_bytes),
        DType::I64 => from_le!(bytes, swapped, i64, i64::from_le_bytes),
        DType::F16 => from_le!(bytes, swapped, F16, |raw| F16::from_bits(u16::from_le_bytes(raw))),
        DType::F32 => from_le!(bytes, swapped, f32, f32::from_le_bytes),
        DType::F64 => from_le!(bytes, swapped, f64, f64::from_le_bytes),
        DType::C64 if cfg!(target_endian = "little") && !swapped => AnyBuffer::from(OwnedBuffer::from_vec(pod::<Complex64>(bytes))),
        DType::C128 if cfg!(target_endian = "little") && !swapped => AnyBuffer::from(OwnedBuffer::from_vec(pod::<Complex128>(bytes))),
        DType::C64 => {
            let values: Vec<Complex64> = bytes.chunks_exact(8).map(|chunk| {
                let part = |range: std::ops::Range<usize>| { let mut raw = [0u8; 4]; raw.copy_from_slice(&chunk[range]); if swapped { raw.reverse(); } f32::from_le_bytes(raw) };
                Complex64 { re: part(0..4), im: part(4..8) }
            }).collect();
            AnyBuffer::from(OwnedBuffer::from_vec(values))
        }
        DType::C128 => {
            let values: Vec<Complex128> = bytes.chunks_exact(16).map(|chunk| {
                let part = |range: std::ops::Range<usize>| { let mut raw = [0u8; 8]; raw.copy_from_slice(&chunk[range]); if swapped { raw.reverse(); } f64::from_le_bytes(raw) };
                Complex128 { re: part(0..8), im: part(8..16) }
            }).collect();
            AnyBuffer::from(OwnedBuffer::from_vec(values))
        }
    }
}

/// Reverse every `unit`-byte group (a byte-swapped copy).
fn swap_units(bytes: &[u8], unit: usize) -> Vec<u8> {
    let mut out = bytes.to_vec();
    if unit > 1 { for chunk in out.chunks_exact_mut(unit) { chunk.reverse(); } }
    out
}

/// The elements of a C-order plane in the other memory order (`to_fortran`)
/// or back, `itemsize` bytes each, without an index table.
fn reorder(bytes: &[u8], shape: &[usize], itemsize: usize, to_fortran: bool) -> Vec<u8> {
    let rank = shape.len();
    let count: usize = shape.iter().product();
    if rank < 2 || count == 0 || itemsize == 0 { return bytes.to_vec(); }
    let mut out = vec![0u8; bytes.len()];
    // Fortran strides (in elements) of each axis.
    let mut strides = vec![1usize; rank];
    for axis in 1..rank { strides[axis] = strides[axis - 1] * shape[axis - 1]; }
    let mut index = vec![0usize; rank];
    let mut fortran_offset = 0usize;
    for c_offset in 0..count {
        let (from, to) = if to_fortran { (c_offset, fortran_offset) } else { (fortran_offset, c_offset) };
        out[to * itemsize..(to + 1) * itemsize].copy_from_slice(&bytes[from * itemsize..(from + 1) * itemsize]);
        let mut axis = rank;
        while axis > 0 {
            axis -= 1;
            index[axis] += 1;
            fortran_offset += strides[axis];
            if index[axis] < shape[axis] { break; }
            fortran_offset -= strides[axis] * shape[axis];
            index[axis] = 0;
        }
    }
    out
}

impl NpyArray {
    pub fn numeric(values: AnyBuffer, shape: Vec<usize>) -> Self {
        Self { dtype: NpyType::native(values.dtype()), shape, data: NpyData::Numeric(values), fortran: false }
    }

    /// `np.asarray(tuple_of_str, dtype="U")`: the widest item sets the width (at least 1).
    pub fn unicode(items: Vec<String>, shape: Vec<usize>) -> Self {
        let chars = items.iter().map(|item| pystr::len(item)).max().unwrap_or(0).max(1);
        let mut units = Vec::with_capacity(items.len() * chars);
        for item in &items {
            let start = units.len();
            units.extend(pystr::code_points(item));
            units.resize(start + chars, 0);
        }
        Self { dtype: NpyType::unicode(chars, false), shape, data: NpyData::Text(units), fortran: false }
    }

    /// Element count (`array.size`).
    pub fn size(&self) -> usize { self.shape.iter().product() }

    /// Item `index` of a `U` array as Python text (`str(array[index])`,
    /// trailing NULs dropped); CPython refuses a lone unit beyond U+10FFFF.
    pub fn text(&self, index: usize) -> Result<String> {
        let NpyData::Text(units) = &self.data else { return Ok(String::new()) };
        let chars = self.dtype.itemsize / 4;
        let mut item = &units[index * chars..(index + 1) * chars];
        while let [rest @ .., 0] = item { item = rest; }
        if let [unit] = item { if *unit > 0x10FFFF { return Err(ArchiveError::new(ErrorKind::System, "invalid maximum character passed to PyUnicode_New")); } }
        let mut out = String::with_capacity(item.len());
        for unit in item { pystr::push(&mut out, *unit); }
        Ok(out)
    }

    /// Item `index` of an `S` array (trailing NULs dropped).
    pub fn bytes_item(&self, index: usize) -> &[u8] {
        let NpyData::Raw(bytes) = &self.data else { return &[] };
        let size = self.dtype.itemsize;
        let mut item = &bytes[index * size..(index + 1) * size];
        while let [rest @ .., 0] = item { item = rest; }
        item
    }

    /// The values' bytes in C order and the dtype's own byte order.
    fn c_bytes(&self) -> Cow<'_, [u8]> {
        match &self.data {
            NpyData::Numeric(values) => {
                let unit = match self.dtype.kind { Kind::Numeric(DType::C64) => 4, Kind::Numeric(DType::C128) => 8, _ => self.dtype.itemsize };
                if self.dtype.swapped { Cow::Owned(swap_units(values.as_bytes(), unit)) } else { Cow::Borrowed(values.as_bytes()) }
            }
            NpyData::Text(units) => Cow::Owned(units.iter().flat_map(|unit| if self.dtype.swapped { unit.to_be_bytes() } else { unit.to_le_bytes() }).collect()),
            NpyData::Raw(bytes) => Cow::Borrowed(bytes),
        }
    }

    /// The bytes `write_array` stores: Fortran order when the array is.
    pub fn data_bytes(&self) -> Cow<'_, [u8]> {
        let bytes = self.c_bytes();
        if self.fortran { Cow::Owned(reorder(&bytes, &self.shape, self.dtype.itemsize, true)) } else { bytes }
    }

    /// `_write_array_header(fp, header_data_from_array_1_0(array))`.
    pub fn header(&self) -> Vec<u8> {
        let shape = match self.shape.len() {
            0 => "()".to_owned(),
            1 => format!("({},)", self.shape[0]),
            _ => format!("({})", self.shape.iter().map(usize::to_string).collect::<Vec<_>>().join(", ")),
        };
        let fortran = if self.fortran { "True" } else { "False" };
        let mut header = format!("{{'descr': {}, 'fortran_order': {fortran}, 'shape': {shape}, }}", py_repr_str(&self.dtype.str()));
        if let Some(first) = self.shape.first() { header.push_str(&" ".repeat(GROWTH_AXIS_MAX_DIGITS.saturating_sub(first.to_string().len()))); }
        let header_len = header.len() + 1;
        let (version, prefix) = if 10 + header_len < 65536 { (1u8, 10) } else { (2u8, 12) };
        let pad = ARRAY_ALIGN - ((prefix + header_len) % ARRAY_ALIGN);
        let mut out = Vec::with_capacity(prefix + header_len + pad);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&[version, 0]);
        let total = header_len + pad;
        if version == 1 { out.extend_from_slice(&(total as u16).to_le_bytes()); } else { out.extend_from_slice(&(total as u32).to_le_bytes()); }
        out.extend_from_slice(header.as_bytes());
        out.resize(out.len() + pad, b' ');
        out.push(b'\n');
        out
    }

    /// The values' bytes as little-endian C order (what a digest of
    /// `array.astype(dtype.newbyteorder("<"))` covers).
    pub fn little_endian_bytes(&self) -> Cow<'_, [u8]> {
        match (&self.data, &self.dtype.kind) {
            (NpyData::Numeric(values), _) => Cow::Borrowed(values.as_bytes()),
            (NpyData::Text(units), _) => Cow::Owned(units.iter().flat_map(|unit| unit.to_le_bytes()).collect()),
            (NpyData::Raw(bytes), Kind::DateTime { .. }) if self.dtype.swapped => Cow::Owned(swap_units(bytes, 8)),
            (NpyData::Raw(bytes), _) => Cow::Borrowed(bytes),
        }
    }
}

// ------------------------------------------------------------- reading

/// Where `read_array` reads: a zip member or an open file.
pub trait NpySource {
    /// `fp.read(n)`.
    fn read(&mut self, n: usize) -> Result<Vec<u8>>;
    /// `isfileobj(fp)`: data then comes through `numpy.fromfile`.
    fn is_file(&self) -> bool;
}

impl NpySource for crate::zip::ZipExtFile<'_> {
    fn read(&mut self, n: usize) -> Result<Vec<u8>> { crate::zip::ZipExtFile::read(self, n) }
    fn is_file(&self) -> bool { false }
}

/// A plain `.npy` file opened by `np.load`.
pub struct FileSource { pub data: Vec<u8>, pub position: usize }

impl NpySource for FileSource {
    fn read(&mut self, n: usize) -> Result<Vec<u8>> {
        let end = self.position.saturating_add(n).min(self.data.len());
        let out = self.data[self.position..end].to_vec();
        self.position = end;
        Ok(out)
    }
    fn is_file(&self) -> bool { true }
}

/// `_read_bytes(fp, size, template)`.
fn read_bytes(source: &mut dyn NpySource, size: usize, template: &str) -> Result<Vec<u8>> {
    let mut data = Vec::new();
    loop {
        let more = source.read(size - data.len())?;
        let empty = more.is_empty();
        if data.is_empty() { data = more; } else { data.extend(more); }
        if empty || data.len() == size { break; }
    }
    if data.len() != size {
        return Err(ArchiveError::value(format!("EOF: reading {template}, expected {size} bytes got {}", data.len())));
    }
    Ok(data)
}

fn token_error(failure: TokenFailure) -> ArchiveError {
    match failure {
        TokenFailure::Token(message, line, col) => ArchiveError::new(ErrorKind::Token, format!("({}, ({line}, {col}))", py_repr_str(&message))),
        TokenFailure::Indentation(message, line) => ArchiveError::new(ErrorKind::Indentation, format!("{message} (<string>, line {line})")),
        TokenFailure::Tab(line) => ArchiveError::new(ErrorKind::Tab, format!("inconsistent use of tabs and spaces in indentation (<string>, line {line})")),
    }
}

fn eval_error(error: EvalError) -> ArchiveError {
    match error {
        EvalError::Syntax => ArchiveError::new(ErrorKind::Syntax, "invalid syntax"),
        EvalError::Malformed(message) => ArchiveError::value(message),
        EvalError::Type(message) => ArchiveError::type_error(message),
        EvalError::Recursion => ArchiveError::new(ErrorKind::Recursion, "maximum recursion depth exceeded during ast construction"),
    }
}

/// `read_magic`, `_check_version` and `_read_array_header`: shape (Python
/// ints), Fortran order and dtype.
fn read_header(source: &mut dyn NpySource) -> Result<(Vec<Obj>, bool, NpyType)> {
    let magic = read_bytes(source, 8, "magic string")?;
    if &magic[..6] != MAGIC {
        return Err(ArchiveError::value(format!("the magic string is not correct; expected {}, got {}", py_repr_bytes(MAGIC), py_repr_bytes(&magic[..6]))));
    }
    let (major, minor) = (magic[6], magic[7]);
    let (length_size, utf8) = match (major, minor) {
        (1, 0) => (2, false), (2, 0) => (4, false), (3, 0) => (4, true),
        _ => return Err(ArchiveError::value(format!("we only support format version (1,0), (2,0), and (3,0), not ({major}, {minor})"))),
    };
    let raw = read_bytes(source, length_size, "array header length")?;
    let length = if length_size == 2 { usize::from(u16::from_le_bytes([raw[0], raw[1]])) } else { u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]) as usize };
    let bytes = read_bytes(source, length, "array header")?;
    let header: String = if utf8 { pystr::decode_utf8(&bytes)? } else { bytes.iter().map(|byte| *byte as char).collect() };
    let count = pystr::len(&header);
    if count > MAX_HEADER_SIZE {
        return Err(ArchiveError::value(format!("Header info length ({count}) is large and may not be safe to load securely.\nTo allow loading, adjust `max_header_size` or fully trust the `.npy` file using `allow_pickle=True`.\nFor safety against large resource use or crashes, sandboxing may be necessary.")));
    }
    let parsed = match literal_eval(&header) {
        Ok(value) => value,
        Err(EvalError::Syntax) if !utf8 => {
            let filtered = filter_header(&header).map_err(token_error)?;
            match literal_eval(&filtered) {
                Ok(value) => value,
                Err(EvalError::Syntax) => return Err(ArchiveError::value(format!("Cannot parse header: {}", py_repr_str(&filtered)))),
                Err(other) => return Err(eval_error(other)),
            }
        }
        Err(EvalError::Syntax) => return Err(ArchiveError::value(format!("Cannot parse header: {}", py_repr_str(&header)))),
        Err(other) => return Err(eval_error(other)),
    };
    let Obj::Dict(pairs) = &parsed else { return Err(ArchiveError::value(format!("Header is not a dictionary: {}", parsed.repr()))) };
    let expected = ["descr", "fortran_order", "shape"];
    let exact = pairs.len() == 3 && expected.iter().all(|key| pairs.iter().any(|(k, _)| matches!(k, Obj::Str(text) if text == key)));
    if !exact {
        let keys = sorted(pairs.iter().map(|(key, _)| key.clone()).collect()).map_err(ArchiveError::type_error)?;
        return Err(ArchiveError::value(format!("Header does not contain the correct keys: [{}]", keys.iter().map(Obj::repr).collect::<Vec<_>>().join(", "))));
    }
    let field = |name: &str| &pairs.iter().find(|(key, _)| matches!(key, Obj::Str(text) if text == name)).unwrap().1;
    let shape = match field("shape") {
        Obj::Tuple(items) if items.iter().all(|item| matches!(item, Obj::Int(_) | Obj::Bool(_))) => items.clone(),
        other => return Err(ArchiveError::value(format!("shape is not valid: {}", other.repr()))),
    };
    let fortran = match field("fortran_order") {
        Obj::Bool(flag) => *flag,
        other => return Err(ArchiveError::value(format!("fortran_order is not a valid bool: {}", other.repr()))),
    };
    let descr = field("descr");
    let dtype = descr_to_dtype(descr).map_err(|error| match error.kind {
        ErrorKind::Type => ArchiveError::value(format!("descr is not a valid dtype descriptor: {}", descr.repr())),
        _ => error,
    })?;
    Ok((shape, fortran, dtype))
}

/// `numpy.multiply.reduce(shape, dtype=numpy.int64)`: values wrap to int64.
fn element_count(shape: &[Obj]) -> Result<i64> {
    let mut count: i64 = 1;
    for item in shape {
        let value: i64 = match item {
            Obj::Bool(flag) => i64::from(*flag),
            Obj::Int(Int::Small(value)) if (i128::from(i64::MIN)..=i128::from(u64::MAX)).contains(value) => *value as i64,
            _ => return Err(ArchiveError::new(ErrorKind::Overflow, "int too big to convert")),
        };
        count = count.wrapping_mul(value);
    }
    Ok(count)
}

/// `str(dtype)` as an allocation failure names it.
fn dtype_label(dtype: &NpyType) -> String {
    match &dtype.kind {
        Kind::Numeric(kind) if !dtype.swapped => crate::tree::dtype_name(&NpyType::native(*kind)),
        Kind::DateTime { code, unit } if !dtype.swapped => format!("{}{}", if *code == 'M' { "datetime64" } else { "timedelta64" }, unit),
        _ => dtype.str(),
    }
}

/// `_size_to_string(num_bytes)`.
fn size_text(bytes: u128) -> String {
    const UNITS: [&str; 7] = ["bytes", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"];
    let bits = 128 - bytes.leading_zeros() as usize;
    let mut unit = (bits.saturating_sub(1)).max(1) / 10;
    let mut value = bytes as f64 / (1u128 << (unit * 10)) as f64;
    if value.round() == 1024.0 { unit += 1; value /= 1024.0; }
    if unit >= UNITS.len() { value *= (1u128 << ((unit - UNITS.len() + 1) * 10)) as f64; unit = UNITS.len() - 1; }
    if unit == 0 { return format!("{value:.0} {}", UNITS[0]); }
    if value.round() < 1000.0 {
        // `#.3g`: three significant digits, the point kept.
        let scientific = format!("{value:.2e}");
        let (mantissa, exponent) = scientific.split_once('e').unwrap();
        let exponent: i32 = exponent.parse().unwrap();
        let digits: String = mantissa.chars().filter(|c| c.is_ascii_digit()).collect();
        let point = (exponent + 1) as usize;
        let text = if point >= digits.len() { format!("{digits}.") } else { format!("{}.{}", &digits[..point], &digits[point..]) };
        return format!("{text} {}", UNITS[unit]);
    }
    format!("{value:.0}. {}", UNITS[unit])
}

/// `numpy.ndarray(count, dtype)`'s refusals; the buffer the data fills.
fn allocate(count: i64, dtype: &NpyType) -> Result<Vec<u8>> {
    if count < 0 { return Err(ArchiveError::value("negative dimensions are not allowed")); }
    let bytes = (count as i128) * (dtype.itemsize as i128);
    if bytes > i128::from(i64::MAX) {
        return Err(ArchiveError::value("array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size."));
    }
    let mut buffer: Vec<u8> = Vec::new();
    if buffer.try_reserve_exact(bytes as usize).is_err() {
        return Err(ArchiveError::new(ErrorKind::Memory, format!("Unable to allocate {} for an array with shape ({count},) and data type {}", size_text(bytes as u128), dtype_label(dtype))));
    }
    Ok(buffer)
}

/// `convert_shape_to_string(dims, "")` as a reshape refusal prints it.
fn shape_text(dims: &[i128]) -> String {
    let first = dims.iter().position(|dim| *dim >= 0);
    let Some(first) = first else { return "()".into() };
    let mut text = format!("({}", dims[first]);
    for dim in &dims[first + 1..] { if *dim < 0 { text.push_str(",newaxis"); } else { text.push_str(&format!(",{dim}")); } }
    text.push_str(if first == 0 && dims.len() == 1 { ",)" } else { ")" });
    text
}

/// `array.reshape(dims)` of `count` elements of `itemsize` bytes: the shape
/// NumPy accepts, or its refusal.
fn reshape(dims: &[Obj], count: i64, itemsize: usize) -> Result<Vec<usize>> {
    let mut values = Vec::with_capacity(dims.len());
    for dim in dims {
        values.push(match dim {
            Obj::Bool(_) => return Err(ArchiveError::type_error("an integer is required")),
            Obj::Int(Int::Small(value)) if (i128::from(i64::MIN)..=i128::from(i64::MAX)).contains(value) => *value,
            _ => return Err(ArchiveError::value("Maximum allowed dimension exceeded")),
        });
    }
    let mismatch = || ArchiveError::value(format!("cannot reshape array of size {count} into shape {}", shape_text(&values)));
    let mut unknown = None;
    let mut known: i128 = 1;
    for (index, value) in values.iter().enumerate() {
        if *value < 0 {
            if unknown.is_some() { return Err(ArchiveError::value("can only specify one unknown dimension")); }
            unknown = Some(index);
        } else {
            known = known.checked_mul(*value).filter(|product| *product <= i128::from(i64::MAX)).ok_or_else(mismatch)?;
        }
    }
    if unknown.is_some() || known != i128::from(count) { return Err(mismatch()); }
    let mut bytes = itemsize as i128;
    for value in &values {
        if *value == 0 { continue; }
        bytes = bytes.checked_mul(*value).filter(|product| *product <= i128::from(i64::MAX))
            .ok_or_else(|| ArchiveError::value("array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size."))?;
    }
    Ok(values.into_iter().map(|value| value as usize).collect())
}

/// `format.read_array(fp, allow_pickle=False)`.
pub fn read_array(source: &mut dyn NpySource) -> Result<NpyArray> {
    let (shape, fortran, dtype) = read_header(source)?;
    let count = if shape.is_empty() { 1 } else { element_count(&shape)? };
    if dtype.has_object() { return Err(ArchiveError::value("Object arrays cannot be loaded when allow_pickle=False")); }
    let itemsize = dtype.itemsize;
    let mut buffer = allocate(count, &dtype)?;
    let shape_repr = Obj::Tuple(shape.clone()).repr();
    let count_repr = count.to_string();
    let count = count as usize;
    if source.is_file() {
        // `numpy.fromfile`: whatever whole items the file still holds.
        let want = count * itemsize;
        let mut data = Vec::new();
        while data.len() < want { let more = source.read(want - data.len())?; if more.is_empty() { break; } data.extend(more); }
        let items = if itemsize == 0 { count } else { data.len() / itemsize };
        if items != count {
            return Err(ArchiveError::value(format!("Failed to read all data for array. Expected {shape_repr} = {count_repr} elements, could only read {items} elements. (file seems not fully written?)")));
        }
        buffer.extend_from_slice(&data[..items * itemsize]);
    } else if itemsize > 0 {
        let step = BUFFER_SIZE / itemsize.min(BUFFER_SIZE);
        let mut index = 0;
        while index < count {
            let read_count = step.min(count - index);
            buffer.extend(read_bytes(source, read_count * itemsize, "array data")?);
            index += step;
        }
    }
    // A subarray dtype spreads its elements over extra axes.
    let (dtype, size) = match &dtype.subarray {
        Some((base, sub)) => ((**base).clone(), count.saturating_mul(sub.iter().product::<usize>())),
        None => (dtype, count),
    };
    if size != count {
        return Err(ArchiveError::value(format!("Failed to read all data for array. Expected {shape_repr} = {count_repr} elements, could only read {size} elements. (file seems not fully written?)")));
    }
    let shape = if fortran {
        let reversed: Vec<Obj> = shape.iter().rev().cloned().collect();
        let mut dims = reshape(&reversed, count as i64, dtype.itemsize)?;
        dims.reverse();
        dims
    } else {
        reshape(&shape, count as i64, dtype.itemsize)?
    };
    let bytes = if fortran { reorder(&buffer, &shape, dtype.itemsize, false) } else { buffer };
    let data = match &dtype.kind {
        Kind::Numeric(kind) => NpyData::Numeric(numeric_from_bytes(&bytes, *kind, dtype.swapped)),
        Kind::Unicode => NpyData::Text(bytes.chunks_exact(4).map(|unit| {
            let raw = [unit[0], unit[1], unit[2], unit[3]];
            if dtype.swapped { u32::from_be_bytes(raw) } else { u32::from_le_bytes(raw) }
        }).collect()),
        _ => NpyData::Raw(bytes),
    };
    // `flags.c_contiguous` of the transposed read: no elements, no item
    // bytes, or at most one axis longer than one.
    let contiguous_c = count == 0 || dtype.itemsize == 0 || shape.iter().filter(|dim| **dim > 1).count() <= 1;
    Ok(NpyArray { dtype, shape, data, fortran: fortran && !contiguous_c })
}

/// `np.asarray(member_bytes)` for a member without the NPY magic.
pub fn bytes_scalar(content: Vec<u8>) -> NpyArray {
    let len = content.len().max(1);
    let mut raw = content;
    raw.resize(len, 0);
    NpyArray { dtype: NpyType::bytes(len), shape: Vec::new(), data: NpyData::Raw(raw), fortran: false }
}
