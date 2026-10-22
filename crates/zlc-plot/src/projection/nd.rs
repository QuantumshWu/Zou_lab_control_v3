//! Typed immutable C-contiguous arrays: the numpy arrays a projection reads
//! and returns, with numpy's dtype vocabulary. Storage shares zlc-core buffers.

use std::fmt;
use std::sync::Arc;

use zlc_core::{AnyBuffer, F16, OwnedBuffer};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType { Bool, U8, I8, U16, I16, U32, I32, U64, I64, F16, F32, F64, Object }

impl DType {
    /// numpy's `dtype.kind`.
    pub fn kind(self) -> char {
        match self {
            DType::Bool => 'b',
            DType::U8 | DType::U16 | DType::U32 | DType::U64 => 'u',
            DType::I8 | DType::I16 | DType::I32 | DType::I64 => 'i',
            DType::F16 | DType::F32 | DType::F64 => 'f',
            DType::Object => 'O',
        }
    }
    /// How numpy's minimum/maximum loops for this dtype break ties.
    pub fn extremes(self) -> crate::numeric::Extremes { crate::numeric::Extremes::of(self.kind(), self.itemsize()) }
    pub fn itemsize(self) -> usize {
        match self {
            DType::Bool | DType::U8 | DType::I8 => 1,
            DType::U16 | DType::I16 | DType::F16 => 2,
            DType::U32 | DType::I32 | DType::F32 => 4,
            DType::U64 | DType::I64 | DType::F64 | DType::Object => 8,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            DType::Bool => "bool", DType::U8 => "u8", DType::I8 => "i8", DType::U16 => "u16", DType::I16 => "i16",
            DType::U32 => "u32", DType::I32 => "i32", DType::U64 => "u64", DType::I64 => "i64", DType::F16 => "f16",
            DType::F32 => "f32", DType::F64 => "f64", DType::Object => "object",
        }
    }
    pub fn parse(text: &str) -> Option<Self> {
        [DType::Bool, DType::U8, DType::I8, DType::U16, DType::I16, DType::U32, DType::I32, DType::U64, DType::I64,
         DType::F16, DType::F32, DType::F64, DType::Object].into_iter().find(|dtype| dtype.name() == text)
    }
    pub fn from_core(dtype: zlc_core::DType) -> Option<Self> {
        use zlc_core::DType as C;
        Some(match dtype {
            C::Bool => DType::Bool, C::U8 => DType::U8, C::I8 => DType::I8, C::U16 => DType::U16, C::I16 => DType::I16,
            C::U32 => DType::U32, C::I32 => DType::I32, C::U64 => DType::U64, C::I64 => DType::I64, C::F16 => DType::F16,
            C::F32 => DType::F32, C::F64 => DType::F64, C::C64 | C::C128 => return None,
        })
    }
    /// The integer bounds numpy's `iinfo` gives (bool as 1/0).
    pub fn integer_bounds(self) -> Option<(i128, i128)> {
        Some(match self {
            DType::Bool => (0, 1),
            DType::U8 => (0, u8::MAX as i128), DType::I8 => (i8::MIN as i128, i8::MAX as i128),
            DType::U16 => (0, u16::MAX as i128), DType::I16 => (i16::MIN as i128, i16::MAX as i128),
            DType::U32 => (0, u32::MAX as i128), DType::I32 => (i32::MIN as i128, i32::MAX as i128),
            DType::U64 => (0, u64::MAX as i128), DType::I64 => (i64::MIN as i128, i64::MAX as i128),
            _ => return None,
        })
    }
}

/// IEEE binary16 to binary64 (exact).
pub fn half_to_f64(bits: u16) -> f64 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = (bits >> 10) & 31;
    let fraction = bits & 1023;
    match exponent {
        0 => sign * f64::from(fraction) * 2.0_f64.powi(-24),
        31 => if fraction == 0 { sign * f64::INFINITY } else { f64::NAN },
        _ => sign * f64::from(1024 + fraction) * 2.0_f64.powi(i32::from(exponent) - 25),
    }
}

/// A Python scalar as a projection hands it on (`AxisValue.canonical`, a
/// coordinate, a scope value): `None`, `bool`, `int`, `float` or `str`.
#[derive(Clone, Debug)]
pub enum Scalar { None, Bool(bool), Int(i128), Float(f64), Text(Arc<str>) }

impl PartialEq for Scalar {
    /// Python equality: `1 == 1.0`, bit-free float comparison.
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Scalar::None, Scalar::None) => true,
            (Scalar::Bool(a), Scalar::Bool(b)) => a == b,
            (Scalar::Int(a), Scalar::Int(b)) => a == b,
            (Scalar::Float(a), Scalar::Float(b)) => a == b,
            (Scalar::Int(a), Scalar::Float(b)) | (Scalar::Float(b), Scalar::Int(a)) => (*a as f64) == *b && b.fract() == 0.0 && (*b as i128) == *a,
            (Scalar::Text(a), Scalar::Text(b)) => a == b,
            _ => false,
        }
    }
}

impl Scalar {
    /// Python `str(value)`.
    pub fn python_str(&self) -> String {
        match self {
            Scalar::None => "None".into(),
            Scalar::Bool(value) => if *value { "True".into() } else { "False".into() },
            Scalar::Int(value) => value.to_string(),
            Scalar::Float(value) => crate::pyfmt::repr(*value),
            Scalar::Text(value) => value.to_string(),
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self { Scalar::Int(v) => Some(*v as f64), Scalar::Float(v) => Some(*v), Scalar::Bool(v) => Some(f64::from(u8::from(*v))), _ => None }
    }
    pub fn is_numeric(&self) -> bool { matches!(self, Scalar::Int(_) | Scalar::Float(_)) }
}

/// One numpy element type.
pub trait Elem: crate::numeric::Widen + zlc_core::Scalar + 'static {
    const NP: DType;
    /// The exact integer value (floats truncate; only integer dtypes ask).
    fn int(self) -> i128 { self.f64() as i128 }
    fn f64(self) -> f64;
    fn finite(self) -> bool { true }
    fn wrap(buffer: OwnedBuffer<Self>) -> AnyBuffer;
    fn view(buffer: &AnyBuffer) -> Option<&OwnedBuffer<Self>>;
    /// The exact Python scalar `.item()` gives.
    fn scalar(self) -> Scalar;
}

macro_rules! elem_int {
    ($($type:ty => $dtype:ident),*) => {$(
        impl Elem for $type {
            const NP: DType = DType::$dtype;
            #[inline(always)] fn f64(self) -> f64 { self as f64 }
            fn wrap(buffer: OwnedBuffer<Self>) -> AnyBuffer { AnyBuffer::$dtype(buffer) }
            fn view(buffer: &AnyBuffer) -> Option<&OwnedBuffer<Self>> { if let AnyBuffer::$dtype(b) = buffer { Some(b) } else { None } }
            fn int(self) -> i128 { self as i128 }
            fn scalar(self) -> Scalar { Scalar::Int(self as i128) }
        }
    )*};
}
elem_int!(u8 => U8, i8 => I8, u16 => U16, i16 => I16, u32 => U32, i32 => I32, u64 => U64, i64 => I64);

impl Elem for bool {
    const NP: DType = DType::Bool;
    #[inline(always)] fn f64(self) -> f64 { if self { 1.0 } else { 0.0 } }
    fn wrap(buffer: OwnedBuffer<Self>) -> AnyBuffer { AnyBuffer::Bool(buffer) }
    fn view(buffer: &AnyBuffer) -> Option<&OwnedBuffer<Self>> { if let AnyBuffer::Bool(b) = buffer { Some(b) } else { None } }
    fn scalar(self) -> Scalar { Scalar::Bool(self) }
}
impl Elem for f32 {
    const NP: DType = DType::F32;
    #[inline(always)] fn f64(self) -> f64 { f64::from(self) }
    #[inline(always)] fn finite(self) -> bool { self.is_finite() }
    fn wrap(buffer: OwnedBuffer<Self>) -> AnyBuffer { AnyBuffer::F32(buffer) }
    fn view(buffer: &AnyBuffer) -> Option<&OwnedBuffer<Self>> { if let AnyBuffer::F32(b) = buffer { Some(b) } else { None } }
    fn scalar(self) -> Scalar { Scalar::Float(f64::from(self)) }
}
impl Elem for f64 {
    const NP: DType = DType::F64;
    #[inline(always)] fn f64(self) -> f64 { self }
    #[inline(always)] fn finite(self) -> bool { self.is_finite() }
    fn wrap(buffer: OwnedBuffer<Self>) -> AnyBuffer { AnyBuffer::F64(buffer) }
    fn view(buffer: &AnyBuffer) -> Option<&OwnedBuffer<Self>> { if let AnyBuffer::F64(b) = buffer { Some(b) } else { None } }
    fn scalar(self) -> Scalar { Scalar::Float(self) }
}
impl crate::numeric::Widen for F16 { #[inline(always)] fn widen(self) -> f64 { half_to_f64(self.to_bits()) } }
impl Elem for F16 {
    const NP: DType = DType::F16;
    #[inline(always)] fn f64(self) -> f64 { half_to_f64(self.to_bits()) }
    #[inline(always)] fn finite(self) -> bool { self.to_bits() & 0x7c00 != 0x7c00 }
    fn wrap(buffer: OwnedBuffer<Self>) -> AnyBuffer { AnyBuffer::F16(buffer) }
    fn view(buffer: &AnyBuffer) -> Option<&OwnedBuffer<Self>> { if let AnyBuffer::F16(b) = buffer { Some(b) } else { None } }
    fn scalar(self) -> Scalar { Scalar::Float(self.f64()) }
}

#[derive(Clone)]
pub enum Buf { Num(AnyBuffer), Obj(Arc<[Scalar]>), Const(Arc<Constant>) }

/// One value broadcast over an array (numpy's stride-0 plane), written out
/// only if somebody reads the elements.
pub struct Constant { value: AnyBuffer, len: usize, full: std::sync::OnceLock<Buf> }

/// An immutable C-contiguous array view: `shape` elements from `offset`.
#[derive(Clone)]
pub struct Array { buf: Buf, offset: usize, shape: Vec<usize> }

impl fmt::Debug for Array {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Array({}, {:?})", self.dtype().name(), self.shape)
    }
}

pub fn product(shape: &[usize]) -> usize { shape.iter().product() }

/// Dispatch over the numeric element type of an [`Array`], binding the typed
/// slice. Object arrays take the `object` arm.
#[macro_export]
macro_rules! with_numeric {
    ($array:expr, |$slice:ident| $body:expr, object => $object:expr) => {{
        use $crate::projection::nd::Buf as __Buf;
        use zlc_core::AnyBuffer as __Any;
        let __array = &$array;
        let (__start, __stop) = __array.span();
        match __array.buffer() {
            __Buf::Num(__Any::Bool(b)) => { let $slice: &[bool] = &b.as_slice()[__start..__stop]; $body }
            __Buf::Num(__Any::U8(b)) => { let $slice: &[u8] = &b.as_slice()[__start..__stop]; $body }
            __Buf::Num(__Any::I8(b)) => { let $slice: &[i8] = &b.as_slice()[__start..__stop]; $body }
            __Buf::Num(__Any::U16(b)) => { let $slice: &[u16] = &b.as_slice()[__start..__stop]; $body }
            __Buf::Num(__Any::I16(b)) => { let $slice: &[i16] = &b.as_slice()[__start..__stop]; $body }
            __Buf::Num(__Any::U32(b)) => { let $slice: &[u32] = &b.as_slice()[__start..__stop]; $body }
            __Buf::Num(__Any::I32(b)) => { let $slice: &[i32] = &b.as_slice()[__start..__stop]; $body }
            __Buf::Num(__Any::U64(b)) => { let $slice: &[u64] = &b.as_slice()[__start..__stop]; $body }
            __Buf::Num(__Any::I64(b)) => { let $slice: &[i64] = &b.as_slice()[__start..__stop]; $body }
            __Buf::Num(__Any::F32(b)) => { let $slice: &[f32] = &b.as_slice()[__start..__stop]; $body }
            __Buf::Num(__Any::F64(b)) => { let $slice: &[f64] = &b.as_slice()[__start..__stop]; $body }
            __Buf::Num(__Any::F16(b)) => { let $slice: &[zlc_core::F16] = &b.as_slice()[__start..__stop]; $body }
            __Buf::Num(_) => panic!("complex arrays never reach a projection"),
            __Buf::Obj(_) => $object,
            __Buf::Const(_) => unreachable!(),
        }
    }};
}

impl Array {
    pub fn from_vec<T: Elem>(values: Vec<T>, shape: Vec<usize>) -> Self {
        debug_assert_eq!(values.len(), product(&shape));
        Array { buf: Buf::Num(T::wrap(OwnedBuffer::from_vec(values))), offset: 0, shape }
    }
    pub fn from_buffer(buffer: AnyBuffer, shape: Vec<usize>) -> Self {
        debug_assert_eq!(buffer.len(), product(&shape));
        Array { buf: Buf::Num(buffer), offset: 0, shape }
    }
    pub fn objects(values: Vec<Scalar>, shape: Vec<usize>) -> Self {
        Array { buf: Buf::Obj(values.into()), offset: 0, shape }
    }
    pub fn vector<T: Elem>(values: Vec<T>) -> Self { let n = values.len(); Self::from_vec(values, vec![n]) }
    pub fn full<T: Elem>(value: T, shape: Vec<usize>) -> Self { Self::from_vec(vec![value; product(&shape)], shape) }

    pub fn buffer(&self) -> &Buf {
        match &self.buf {
            Buf::Const(constant) => constant.full.get_or_init(|| Buf::Num(crate::with_dtype!(DType::from_core(constant.value.dtype()).unwrap(), T => {
                let value = T::view(&constant.value).unwrap().as_slice()[0];
                T::wrap(OwnedBuffer::from_vec(vec![value; constant.len]))
            }))),
            other => other,
        }
    }
    /// `np.broadcast_to(value, shape)`: nothing is written until read.
    pub fn constant<T: Elem>(value: T, shape: Vec<usize>) -> Self {
        let len = product(&shape);
        Array { buf: Buf::Const(Arc::new(Constant { value: T::wrap(OwnedBuffer::from_vec(vec![value])), len, full: std::sync::OnceLock::new() })), offset: 0, shape }
    }
    /// The broadcast value, when this array is one.
    pub fn constant_value(&self) -> Option<Scalar> {
        if let Buf::Const(constant) = &self.buf { Some(crate::with_dtype!(DType::from_core(constant.value.dtype()).unwrap(), T => T::view(&constant.value).unwrap().as_slice()[0].scalar())) } else { None }
    }
    pub fn span(&self) -> (usize, usize) { (self.offset, self.offset + self.len()) }
    pub fn shape(&self) -> &[usize] { &self.shape }
    pub fn ndim(&self) -> usize { self.shape.len() }
    pub fn len(&self) -> usize { product(&self.shape) }
    pub fn is_empty(&self) -> bool { self.len() == 0 }
    pub fn dtype(&self) -> DType {
        match &self.buf {
            Buf::Obj(_) => DType::Object,
            Buf::Const(constant) => DType::from_core(constant.value.dtype()).unwrap(),
            Buf::Num(buffer) => DType::from_core(buffer.dtype()).expect("complex arrays never reach a projection"),
        }
    }
    pub fn kind(&self) -> char { self.dtype().kind() }

    /// The typed elements; panics on a dtype mismatch (a programming error).
    pub fn slice<T: Elem>(&self) -> &[T] {
        match self.buffer() {
            Buf::Num(buffer) => &T::view(buffer).unwrap_or_else(|| panic!("array is {}, not {}", self.dtype().name(), T::NP.name())).as_slice()[self.offset..self.offset + self.len()],
            Buf::Obj(_) => panic!("object array read as {}", T::NP.name()),
            Buf::Const(_) => unreachable!(),
        }
    }
    pub fn try_slice<T: Elem>(&self) -> Option<&[T]> {
        match self.buffer() { Buf::Num(buffer) => T::view(buffer).map(|b| &b.as_slice()[self.offset..self.offset + self.len()]), Buf::Obj(_) | Buf::Const(_) => None }
    }
    pub fn object_slice(&self) -> Option<&[Scalar]> {
        match self.buffer() { Buf::Obj(values) => Some(&values[self.offset..self.offset + self.len()]), Buf::Num(_) | Buf::Const(_) => None }
    }

    pub fn reshape(&self, shape: Vec<usize>) -> Self {
        assert_eq!(product(&shape), self.len(), "reshape must keep the element count");
        Array { buf: self.buf.clone(), offset: self.offset, shape }
    }
    /// Index the leading axis: a zero-copy view.
    pub fn index0(&self, index: usize) -> Self {
        let inner = product(&self.shape[1..]);
        Array { buf: self.buf.clone(), offset: self.offset + index * inner, shape: self.shape[1..].to_vec() }
    }
    /// Rows `start..stop` of the leading axis: a zero-copy view.
    pub fn rows(&self, start: usize, stop: usize) -> Self {
        let inner = product(&self.shape[1..]);
        let mut shape = self.shape.clone();
        shape[0] = stop - start;
        Array { buf: self.buf.clone(), offset: self.offset + start * inner, shape }
    }
    /// Whether two arrays are the same storage view (numpy `is` for views of one owner).
    pub fn same_storage(&self, other: &Array) -> bool {
        self.offset == other.offset && self.shape == other.shape && match (&self.buf, &other.buf) {
            (Buf::Num(a), Buf::Num(b)) => a.dtype() == b.dtype() && std::ptr::eq(a.as_bytes().as_ptr(), b.as_bytes().as_ptr()),
            (Buf::Obj(a), Buf::Obj(b)) => Arc::ptr_eq(a, b),
            (Buf::Const(a), Buf::Const(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }

    /// Little-endian element bytes (numpy's `tobytes()` of a C-contiguous array).
    pub fn bytes(&self) -> &[u8] {
        match self.buffer() {
            Buf::Num(buffer) => {
                let size = self.dtype().itemsize();
                &buffer.as_bytes()[self.offset * size..(self.offset + self.len()) * size]
            }
            Buf::Obj(_) | Buf::Const(_) => &[],
        }
    }

    /// Element `index` as f64 (numpy `float(a.flat[i])`); object elements by value.
    pub fn get_f64(&self, index: usize) -> f64 {
        with_numeric!(self, |s| s[index].f64(), object => self.object_slice().unwrap()[index].as_f64().unwrap_or(f64::NAN))
    }
    /// Element `index` as the Python scalar `.item()` gives.
    pub fn scalar(&self, index: usize) -> Scalar {
        with_numeric!(self, |s| s[index].scalar(), object => self.object_slice().unwrap()[index].clone())
    }
    /// Every element widened to f64 (numpy `astype(float64)`).
    pub fn to_f64(&self) -> Vec<f64> {
        with_numeric!(self, |s| s.iter().map(|v| v.f64()).collect(), object => self.object_slice().unwrap().iter().map(|v| v.as_f64().unwrap_or(f64::NAN)).collect())
    }
    pub fn as_f64_array(&self) -> Array {
        if self.dtype() == DType::F64 { return self.clone(); }
        Array::from_vec(self.to_f64(), self.shape.clone())
    }
    /// `np.isfinite` over a numeric array (object arrays: all finite).
    pub fn finite_mask(&self) -> Vec<bool> {
        with_numeric!(self, |s| s.iter().map(|v| v.finite()).collect(), object => vec![true; self.len()])
    }
    pub fn all_finite(&self) -> bool {
        with_numeric!(self, |s| s.iter().all(|v| v.finite()), object => true)
    }

    /// Gather flat positions (numpy fancy indexing of a flattened array).
    pub fn take_flat(&self, positions: &[usize], shape: Vec<usize>) -> Array {
        with_numeric!(self, |s| Array::from_vec(positions.iter().map(|&p| s[p]).collect(), shape),
            object => { let s = self.object_slice().unwrap(); Array::objects(positions.iter().map(|&p| s[p].clone()).collect(), shape) })
    }
    /// `np.take(array, order, axis)` for a C-contiguous array.
    pub fn take_axis(&self, order: &[usize], axis: usize) -> Array {
        let outer = product(&self.shape[..axis]);
        let extent = self.shape[axis];
        let inner = product(&self.shape[axis + 1..]);
        let mut positions = Vec::with_capacity(outer * order.len() * inner);
        for o in 0..outer {
            for &k in order {
                let base = (o * extent + k) * inner;
                positions.extend(base..base + inner);
            }
        }
        let mut shape = self.shape.clone();
        shape[axis] = order.len();
        self.take_flat(&positions, shape)
    }
    /// Keep the elements where `mask` (1-D boolean indexing).
    pub fn compress(&self, mask: &[bool]) -> Array {
        let positions: Vec<usize> = mask.iter().enumerate().filter(|(_, keep)| **keep).map(|(i, _)| i).collect();
        let n = positions.len();
        self.take_flat(&positions, vec![n])
    }
    /// `np.moveaxis(a, sources, destinations).reshape(-1, *kept)` made contiguous:
    /// the remaining axes (in order) lead, then `sources` in the given order.
    pub fn move_to_back(&self, sources: &[usize]) -> Array {
        let ndim = self.ndim();
        let mut order: Vec<usize> = (0..ndim).filter(|axis| !sources.contains(axis)).collect();
        order.extend_from_slice(sources);
        if order.iter().enumerate().all(|(i, axis)| i == *axis) {
            return self.clone();
        }
        self.transpose(&order)
    }
    /// `np.transpose(a, order)` made contiguous.
    pub fn transpose(&self, order: &[usize]) -> Array {
        let shape: Vec<usize> = order.iter().map(|axis| self.shape[*axis]).collect();
        let mut strides = vec![1usize; self.ndim()];
        for axis in (0..self.ndim().saturating_sub(1)).rev() { strides[axis] = strides[axis + 1] * self.shape[axis + 1]; }
        let moved: Vec<usize> = order.iter().map(|axis| strides[*axis]).collect();
        let total = product(&shape);
        if self.dtype() != DType::Object && total > 0 {
            return crate::with_dtype!(self.dtype(), T => {
                let source = self.slice::<T>();
                // The innermost output axis is copied as one run when it is contiguous.
                let last = shape.len() - 1;
                let run = if moved[last] == 1 { shape[last] } else { 1 };
                let outer_dims = if run > 1 { last } else { shape.len() };
                let mut out: Vec<T> = Vec::with_capacity(total);
                let mut index = vec![0usize; outer_dims];
                let mut flat = 0usize;
                for _ in 0..total / run {
                    if run > 1 { out.extend_from_slice(&source[flat..flat + run]); } else { out.push(source[flat]); }
                    for axis in (0..outer_dims).rev() {
                        index[axis] += 1;
                        flat += moved[axis];
                        if index[axis] < shape[axis] { break; }
                        flat -= moved[axis] * shape[axis];
                        index[axis] = 0;
                    }
                }
                Array::from_vec(out, shape)
            });
        }
        let mut positions = Vec::with_capacity(total);
        let mut index = vec![0usize; shape.len()];
        let mut flat = 0usize;
        for _ in 0..total {
            positions.push(flat);
            for axis in (0..shape.len()).rev() {
                index[axis] += 1;
                flat += moved[axis];
                if index[axis] < shape[axis] { break; }
                flat -= moved[axis] * shape[axis];
                index[axis] = 0;
            }
        }
        self.take_flat(&positions, shape)
    }
}

/// An immutable bool plane: one computed here, or a zlc-core validity
/// buffer viewed in place (no copy per revision).
#[derive(Clone)]
pub struct Bools(zlc_core::OwnedBuffer<bool>);

impl Bools {
    pub fn core(buffer: zlc_core::OwnedBuffer<bool>) -> Self { Bools(buffer) }
    /// The very same storage (not merely equal contents).
    pub fn ptr_eq(a: &Bools, b: &Bools) -> bool { a.as_ptr() == b.as_ptr() && a.len() == b.len() }
}
impl std::ops::Deref for Bools { type Target = [bool]; fn deref(&self) -> &[bool] { self.0.as_slice() } }
impl From<Vec<bool>> for Bools { fn from(values: Vec<bool>) -> Self { Bools(zlc_core::OwnedBuffer::from_vec(values)) } }
impl From<&[bool]> for Bools { fn from(values: &[bool]) -> Self { values.to_vec().into() } }
impl FromIterator<bool> for Bools { fn from_iter<I: IntoIterator<Item = bool>>(iter: I) -> Self { iter.into_iter().collect::<Vec<bool>>().into() } }
impl PartialEq for Bools { fn eq(&self, other: &Self) -> bool { self[..] == other[..] } }
impl std::fmt::Debug for Bools { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "Bools({})", self.len()) } }

/// A boolean plane, or a constant one that was never materialized
/// (numpy's stride-0 broadcast).
#[derive(Clone, Debug)]
pub enum Mask { Uniform(bool), Plane(Bools) }

impl Mask {
    pub fn get(&self, index: usize) -> bool { match self { Mask::Uniform(v) => *v, Mask::Plane(p) => p[index] } }
    /// `_stride_zero_all_true(mask) or bool(mask.all())`.
    pub fn all_true(&self) -> bool { match self { Mask::Uniform(v) => *v, Mask::Plane(p) => p.iter().all(|v| *v) } }
    pub fn plane(&self, len: usize) -> Bools { match self { Mask::Uniform(v) => vec![*v; len].into(), Mask::Plane(p) => p.clone() } }
    pub fn from_vec(values: Vec<bool>) -> Mask { Mask::Plane(values.into()) }
    /// Uniform when every element agrees, which changes no result.
    pub fn normalized(values: Vec<bool>) -> Mask {
        if values.iter().all(|v| *v) { Mask::Uniform(true) } else { Mask::Plane(values.into()) }
    }
}

/// Bind `$t` to the element type of a numeric dtype.
#[macro_export]
macro_rules! with_dtype {
    ($dtype:expr, $t:ident => $body:expr) => {{
        use $crate::projection::nd::DType as __D;
        match $dtype {
            __D::Bool => { type $t = bool; $body }
            __D::U8 => { type $t = u8; $body }
            __D::I8 => { type $t = i8; $body }
            __D::U16 => { type $t = u16; $body }
            __D::I16 => { type $t = i16; $body }
            __D::U32 => { type $t = u32; $body }
            __D::I32 => { type $t = i32; $body }
            __D::U64 => { type $t = u64; $body }
            __D::I64 => { type $t = i64; $body }
            __D::F16 => { type $t = zlc_core::F16; $body }
            __D::F32 => { type $t = f32; $body }
            __D::F64 => { type $t = f64; $body }
            __D::Object => panic!("object arrays have no numeric element type"),
        }
    }};
}

impl Array {
    /// Join same-dtype arrays end to end (`np.concatenate(..., axis=None)`).
    pub fn concat(parts: &[&Array], shape: Vec<usize>) -> Array {
        let dtype = parts.first().map_or(DType::F64, |part| part.dtype());
        if dtype == DType::Object {
            let mut values = Vec::new();
            for part in parts { values.extend_from_slice(part.object_slice().unwrap()); }
            return Array::objects(values, shape);
        }
        if parts.len() == 1 { return parts[0].reshape(shape); }
        crate::with_dtype!(dtype, T => {
            let mut values: Vec<T> = Vec::with_capacity(product(&shape));
            for part in parts { values.extend_from_slice(part.slice::<T>()); }
            Array::from_vec(values, shape)
        })
    }
    /// Leading-axis rows in `order` (`values[order]`).
    pub fn take_rows(&self, order: &[usize]) -> Array {
        let inner = product(&self.shape[1..]);
        let mut shape = self.shape.clone();
        shape[0] = order.len();
        if self.dtype() == DType::Object {
            let source = self.object_slice().unwrap();
            return Array::objects(order.iter().flat_map(|row| source[row * inner..(row + 1) * inner].iter().cloned()).collect(), shape);
        }
        crate::with_dtype!(self.dtype(), T => {
            let source = self.slice::<T>();
            let mut values: Vec<T> = Vec::with_capacity(order.len() * inner);
            for row in order { values.extend_from_slice(&source[row * inner..(row + 1) * inner]); }
            Array::from_vec(values, shape)
        })
    }
    /// A numpy bool array as a mask plane.
    pub fn bools(&self) -> &[bool] { self.slice::<bool>() }
}

/// IEEE binary32 to binary16, round to nearest even (numpy's `npy_float_to_half`).
pub fn f32_to_half_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;
    if exponent == 0xff {
        return if mantissa == 0 { sign | 0x7c00 } else { sign | 0x7e00 | ((mantissa >> 13) as u16) };
    }
    let unbiased = exponent - 127;
    if unbiased > 15 { return sign | 0x7c00; }
    if unbiased >= -14 {
        // Normal half: round the 13 dropped bits to nearest even.
        let half_exponent = (unbiased + 15) as u32;
        let mut half = (half_exponent << 10) | (mantissa >> 13);
        let dropped = mantissa & 0x1fff;
        if dropped > 0x1000 || (dropped == 0x1000 && half & 1 == 1) { half += 1; }
        return sign | half as u16;
    }
    if unbiased < -25 { return sign; }
    // Subnormal half.
    let full = mantissa | 0x80_0000;
    let shift = (-14 - unbiased + 13) as u32;
    let mut half = full >> shift;
    let dropped = full & ((1 << shift) - 1);
    let halfway = 1 << (shift - 1);
    if dropped > halfway || (dropped == halfway && half & 1 == 1) { half += 1; }
    sign | half as u16
}

/// `a - b` as numpy computes it in an array's own float dtype (NEP 50: a
/// Python float meets a float32/float16 array in that dtype), widened.
pub fn subtract_in_dtype(dtype: DType, a: f64, b: f64) -> f64 {
    match dtype {
        DType::F32 => f64::from(a as f32 - b as f32),
        DType::F16 => {
            let b16 = half_to_f64(f32_to_half_bits(b as f32)) as f32;
            half_to_f64(f32_to_half_bits(a as f32 - b16))
        }
        _ => a - b,
    }
}

/// A plane of one element type that may be one broadcast value.
#[derive(Clone, Debug)]
pub enum Plane<T: Copy> { Uniform(T, usize), Values(Vec<T>) }

impl<T: Copy + PartialEq> Plane<T> {
    pub fn len(&self) -> usize { match self { Plane::Uniform(_, n) => *n, Plane::Values(v) => v.len() } }
    pub fn is_empty(&self) -> bool { self.len() == 0 }
    pub fn get(&self, index: usize) -> T { match self { Plane::Uniform(v, _) => *v, Plane::Values(v) => v[index] } }
    pub fn to_vec(&self) -> Vec<T> { match self { Plane::Uniform(v, n) => vec![*v; *n], Plane::Values(v) => v.clone() } }
    pub fn range(&self, start: usize, stop: usize) -> Plane<T> {
        match self { Plane::Uniform(v, _) => Plane::Uniform(*v, stop - start), Plane::Values(v) => Plane::Values(v[start..stop].to_vec()) }
    }
    pub fn uniform(&self) -> Option<T> {
        match self { Plane::Uniform(v, _) => Some(*v), Plane::Values(_) => None }
    }
    pub fn map<U: Copy>(&self, f: impl Fn(T) -> U) -> Plane<U> {
        match self { Plane::Uniform(v, n) => Plane::Uniform(f(*v), *n), Plane::Values(v) => Plane::Values(v.iter().map(|x| f(*x)).collect()) }
    }
}

/// One recycled allocation per element type: a live panel's packed scratch
/// comes back for the next revision instead of faulting in fresh pages.
#[derive(Default)]
pub struct Pools {
    bool: zlc_core::BufferPool<bool>, u8: zlc_core::BufferPool<u8>, i8: zlc_core::BufferPool<i8>,
    u16: zlc_core::BufferPool<u16>, i16: zlc_core::BufferPool<i16>, u32: zlc_core::BufferPool<u32>,
    i32: zlc_core::BufferPool<i32>, u64: zlc_core::BufferPool<u64>, i64: zlc_core::BufferPool<i64>,
    f16: zlc_core::BufferPool<F16>, f32: zlc_core::BufferPool<f32>, f64: zlc_core::BufferPool<f64>,
}

pub trait Pooled: Elem { fn pool(pools: &Pools) -> &zlc_core::BufferPool<Self>; }
macro_rules! pooled { ($($t:ty => $f:ident),*) => {$(impl Pooled for $t { fn pool(pools: &Pools) -> &zlc_core::BufferPool<Self> { &pools.$f } })*}; }
pooled!(bool => bool, u8 => u8, i8 => i8, u16 => u16, i16 => i16, u32 => u32, i32 => i32, u64 => u64, i64 => i64, F16 => f16, f32 => f32, f64 => f64);

impl Pools {
    /// [`Array::concat`] into a recycled allocation.
    pub fn concat(&self, parts: &[&Array], shape: Vec<usize>) -> Array {
        let dtype = parts.first().map_or(DType::F64, |part| part.dtype());
        if dtype == DType::Object || parts.len() <= 1 { return Array::concat(parts, shape); }
        crate::with_dtype!(dtype, T => {
            let mut builder = <T as Pooled>::pool(self).builder(product(&shape));
            for part in parts { builder.extend_from_slice(part.slice::<T>()); }
            Array::from_buffer(T::wrap(builder.finish()), shape)
        })
    }
}
