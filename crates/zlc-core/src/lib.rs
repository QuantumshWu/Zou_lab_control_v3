//! Immutable numerical storage. Shape, axes and publication identity belong to consumers.

use std::mem::{align_of, offset_of, size_of};
use std::sync::{Arc, Mutex, Weak};

pub mod schema;
pub mod dataset;
pub mod runtime;
pub mod store;
pub mod control;
pub mod pyfmt;
pub mod decimal;
pub mod units;
pub mod restrict;
pub mod checkout;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DType {
    Bool, U8, I8, U16, I16, U32, I32, U64, I64, F16, F32, F64, C64, C128,
}

/// IEEE 754 binary16 storage, preserving every bit pattern without arithmetic.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default)]
pub struct F16(u16);

impl F16 {
    pub const fn from_bits(bits: u16) -> Self { Self(bits) }
    pub const fn to_bits(self) -> u16 { self.0 }
}

/// NumPy complex64 storage: native-endian real then imaginary binary32.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Complex64 { pub re: f32, pub im: f32 }

/// NumPy complex128 storage: native-endian real then imaginary binary64.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Complex128 { pub re: f64, pub im: f64 }

// These are storage types, not numeric conversion/arithmetic implementations.
// u16 and IEEE f32/f64 admit every bit pattern. The fixed representations plus
// size/offset/alignment checks prove there are no internal or trailing padding
// bytes for as_bytes() or for one-copy foreign-buffer initialization.
const _: () = {
    assert!(size_of::<F16>() == 2 && align_of::<F16>() == align_of::<u16>());
    assert!(size_of::<Complex64>() == 8 && align_of::<Complex64>() == align_of::<f32>());
    assert!(offset_of!(Complex64, re) == 0 && offset_of!(Complex64, im) == 4);
    assert!(size_of::<Complex128>() == 16 && align_of::<Complex128>() == align_of::<f64>());
    assert!(offset_of!(Complex128, re) == 0 && offset_of!(Complex128, im) == 8);
};

mod sealed {
    pub trait Sealed {}
}

/// Native-endian storage scalars have no padding. Booleans enter only as valid bools.
pub trait Scalar: sealed::Sealed + Copy + Send + Sync + 'static {
    const DTYPE: DType;
}

macro_rules! scalars {
    ($($type:ty => $dtype:ident),* $(,)?) => {
        $(impl sealed::Sealed for $type {}
        impl Scalar for $type { const DTYPE: DType = DType::$dtype; })*

        /// One dynamic dtype owner. Every variant retains the existing typed
        /// allocation, and Clone shares that owner without copying any data.
        #[derive(Clone)]
        pub enum AnyBuffer { $($dtype(OwnedBuffer<$type>)),* }

        $(impl From<OwnedBuffer<$type>> for AnyBuffer {
            fn from(owner: OwnedBuffer<$type>) -> Self { Self::$dtype(owner) }
        })*

        impl AnyBuffer {
            /// Borrow `len` elements of `dtype` at `byte_offset` of foreign memory without copying.
            pub fn from_foreign(dtype: DType, owner: Arc<dyn ForeignBytes>, byte_offset: usize, len: usize) -> Result<Self, String> {
                match dtype { $(DType::$dtype => OwnedBuffer::<$type>::from_foreign(owner, byte_offset, len).map(Self::$dtype)),* }
            }
            /// The foreign owner and byte offset of these values, when they are borrowed.
            pub fn foreign(&self) -> Option<(&Arc<dyn ForeignBytes>, usize)> {
                match self { $(Self::$dtype(owner) => owner.foreign()),* }
            }
            pub fn dtype(&self) -> DType {
                match self { $(Self::$dtype(owner) => owner.dtype()),* }
            }
            pub fn len(&self) -> usize {
                match self { $(Self::$dtype(owner) => owner.len()),* }
            }
            pub fn is_empty(&self) -> bool { self.len() == 0 }
            pub fn byte_len(&self) -> usize {
                match self { $(Self::$dtype(owner) => owner.byte_len()),* }
            }
            pub fn as_bytes(&self) -> &[u8] {
                match self { $(Self::$dtype(owner) => owner.as_bytes()),* }
            }
            pub fn alignment(&self) -> usize {
                match self { $(Self::$dtype(owner) => owner.alignment()),* }
            }
        }
    };
}
scalars! {
    bool => Bool, u8 => U8, i8 => I8, u16 => U16, i16 => I16, u32 => U32, i32 => I32,
    u64 => U64, i64 => I64, F16 => F16, f32 => F32, f64 => F64,
    Complex64 => C64, Complex128 => C128,
}

/// One optional idle allocation; active readers never belong to this cache.
pub struct BufferPool<T: Scalar> {
    idle: Arc<Mutex<Option<Vec<T>>>>,
}

impl<T: Scalar> Default for BufferPool<T> {
    fn default() -> Self {
        Self { idle: Arc::new(Mutex::new(None)) }
    }
}

impl<T: Scalar> BufferPool<T> {
    pub fn new() -> Self { Self::default() }

    /// Start an empty builder without initializing elements that the producer will overwrite.
    pub fn builder(&self, capacity: usize) -> BufferBuilder<T> {
        let mut values = self.idle.lock().unwrap().take().unwrap_or_default();
        values.clear();
        values.reserve(capacity);
        BufferBuilder { storage: Storage { values, pool: Arc::downgrade(&self.idle), foreign: None } }
    }

    /// The idle allocation as a plain vector (stale contents), for a producer
    /// that fills it in place before [`share`](Self::share) publishes it.
    pub fn take_idle(&self) -> Option<Vec<T>> { self.idle.lock().unwrap().take() }

    /// Publish `values` as a buffer that returns to this pool when its last reader drops it.
    pub fn share(&self, values: Vec<T>) -> OwnedBuffer<T> {
        OwnedBuffer(Arc::new(Storage { values, pool: Arc::downgrade(&self.idle), foreign: None }))
    }

    /// Release the idle allocation. Outstanding buffers may return after this call.
    pub fn clear(&self) {
        let idle = self.idle.lock().unwrap().take();
        drop(idle);
    }
}

struct Storage<T: Scalar> {
    values: Vec<T>,
    pool: Weak<Mutex<Option<Vec<T>>>>,
    /// Values borrowed from memory another owner keeps immutable (`values` is then empty).
    foreign: Option<Foreign>,
}

struct Foreign { owner: Arc<dyn ForeignBytes>, offset: usize, len: usize }

/// Memory that stays mapped and is never written while this owner lives, such
/// as a shared-memory block another process lent (see zlc-plugin-host).
///
/// # Safety
/// `bytes` must return the same memory on every call, and nothing may write
/// it for the whole lifetime of the implementing value.
pub unsafe trait ForeignBytes: Send + Sync + 'static {
    fn bytes(&self) -> &[u8];
    fn as_any(&self) -> &dyn std::any::Any;
}

impl<T: Scalar> Drop for Storage<T> {
    fn drop(&mut self) {
        if let Some(pool) = self.pool.upgrade() {
            let displaced = {
                let mut idle = pool.lock().unwrap();
                if idle.as_ref().is_none_or(|values| values.capacity() < self.values.capacity()) {
                    idle.replace(std::mem::take(&mut self.values))
                } else {
                    None
                }
            };
            drop(displaced);
        }
    }
}

pub struct BufferBuilder<T: Scalar> {
    storage: Storage<T>,
}

impl<T: Scalar> BufferBuilder<T> {
    pub fn push(&mut self, value: T) { self.storage.values.push(value); }

    pub fn extend_from_slice(&mut self, values: &[T]) {
        self.storage.values.extend_from_slice(values);
    }

    /// Explicit initialization for kernels that need a complete mutable output slice.
    pub fn resize(&mut self, len: usize, value: T) { self.storage.values.resize(len, value); }

    pub fn as_mut_slice(&mut self) -> &mut [T] { &mut self.storage.values }

    pub fn len(&self) -> usize { self.storage.values.len() }

    pub fn is_empty(&self) -> bool { self.storage.values.is_empty() }

    pub fn finish(self) -> OwnedBuffer<T> { OwnedBuffer(Arc::new(self.storage)) }
}

impl<T: Scalar> Storage<T> {
    fn slice(&self) -> &[T] {
        match &self.foreign {
            None => &self.values,
            Some(foreign) => {
                let bytes = &foreign.owner.bytes()[foreign.offset..foreign.offset + foreign.len * size_of::<T>()];
                // from_foreign checked bounds, alignment and bool bytes; ForeignBytes keeps them fixed.
                unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<T>(), foreign.len) }
            }
        }
    }
}

/// Cloning shares ownership, not pixels. No public operation yields mutable access.
///
/// ```compile_fail
/// use zlc_core::OwnedBuffer;
/// let mut buffer = OwnedBuffer::from_vec(vec![1_u16, 2]);
/// buffer.as_slice()[0] = 3;
/// ```
pub struct OwnedBuffer<T: Scalar>(Arc<Storage<T>>);

impl<T: Scalar> Clone for OwnedBuffer<T> {
    fn clone(&self) -> Self { Self(Arc::clone(&self.0)) }
}

impl<T: Scalar> OwnedBuffer<T> {
    /// Transfer an existing typed allocation without copying or changing its length.
    pub fn from_vec(values: Vec<T>) -> Self {
        Self(Arc::new(Storage { values, pool: Weak::new(), foreign: None }))
    }

    /// Borrow `len` values at `byte_offset` of foreign immutable memory, without copying.
    pub fn from_foreign(owner: Arc<dyn ForeignBytes>, byte_offset: usize, len: usize) -> Result<Self, String> {
        let size = len.checked_mul(size_of::<T>()).ok_or("Foreign value count overflows")?;
        let bytes = owner.bytes();
        let end = byte_offset.checked_add(size).filter(|end| *end <= bytes.len())
            .ok_or_else(|| format!("{size} bytes at offset {byte_offset} exceed the {}-byte foreign block", bytes.len()))?;
        let region = &bytes[byte_offset..end];
        if region.as_ptr() as usize % align_of::<T>() != 0 {
            return Err(format!("Foreign {:?} values at offset {byte_offset} are not aligned", T::DTYPE));
        }
        // A bool other than 0 or 1 is not a value Rust may read.
        if T::DTYPE == DType::Bool && region.iter().any(|byte| *byte > 1) {
            return Err("Foreign bool values must be 0 or 1".into());
        }
        Ok(Self(Arc::new(Storage { values: Vec::new(), pool: Weak::new(),
            foreign: Some(Foreign { owner, offset: byte_offset, len }) })))
    }

    /// The foreign owner and byte offset when these values are borrowed.
    pub fn foreign(&self) -> Option<(&Arc<dyn ForeignBytes>, usize)> {
        self.0.foreign.as_ref().map(|foreign| (&foreign.owner, foreign.offset))
    }

    pub fn as_slice(&self) -> &[T] { self.0.slice() }

    /// Native-endian representation, excluding unused capacity; codecs own endian conversion.
    pub fn as_bytes(&self) -> &[u8] {
        // Scalar is sealed to initialized, padding-free storage types; the borrow retains its owner.
        unsafe { std::slice::from_raw_parts(self.as_slice().as_ptr().cast(), self.byte_len()) }
    }

    pub fn len(&self) -> usize { self.0.foreign.as_ref().map_or(self.0.values.len(), |foreign| foreign.len) }

    pub fn is_empty(&self) -> bool { self.len() == 0 }

    pub fn byte_len(&self) -> usize { self.len() * size_of::<T>() }

    pub fn dtype(&self) -> DType { T::DTYPE }

    pub fn alignment(&self) -> usize { align_of::<T>() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_ownership_moves_and_readers_share_it() {
        let values = vec![1.25_f64, -2.5, 3.0];
        let address = values.as_ptr();
        let frozen = OwnedBuffer::from_vec(values);
        let reader = frozen.clone();
        assert_eq!(frozen.as_slice().as_ptr(), address);
        assert_eq!(reader.as_slice().as_ptr(), address);
        assert_eq!(address as usize % frozen.alignment(), 0);
        assert_eq!(frozen.dtype(), DType::F64);
        assert_eq!(frozen.len(), 3);
        assert_eq!(frozen.byte_len(), 24);
        assert_eq!(&frozen.as_bytes()[..8], &1.25_f64.to_ne_bytes());
        drop(frozen);
        assert_eq!(reader.as_slice(), &[1.25, -2.5, 3.0]);
        let verdicts = OwnedBuffer::from_vec(vec![false, true]);
        assert_eq!(verdicts.dtype(), DType::Bool);
        assert_eq!(verdicts.as_bytes(), &[0, 1]);

        let halves = AnyBuffer::from(OwnedBuffer::from_vec(vec![F16::from_bits(0x7e01)]));
        assert_eq!(halves.dtype(), DType::F16);
        assert_eq!(halves.as_bytes(), &0x7e01_u16.to_ne_bytes());
        let complex = OwnedBuffer::from_vec(vec![Complex64 {
            re: f32::from_bits(0x7fc0_0001), im: -0.0,
        }]);
        let address = complex.as_slice().as_ptr().cast::<u8>();
        let erased = AnyBuffer::from(complex);
        let reader = erased.clone();
        drop(erased);
        assert_eq!(reader.as_bytes().as_ptr(), address);
        assert_eq!((reader.dtype(), reader.len(), reader.byte_len(), reader.alignment()),
                   (DType::C64, 1, 8, align_of::<f32>()));
        assert!(!reader.is_empty());
        let AnyBuffer::C64(typed) = reader else { panic!("dtype changed"); };
        assert_eq!(typed.as_slice()[0].re.to_bits(), 0x7fc0_0001);
        assert_eq!(typed.as_slice()[0].im.to_bits(), (-0.0_f32).to_bits());
        let wide = AnyBuffer::from(OwnedBuffer::from_vec(vec![Complex128 { re: 1.5, im: -2.25 }]));
        assert_eq!(wide.dtype(), DType::C128);
        assert_eq!(wide.byte_len(), 16);
        assert_eq!(&wide.as_bytes()[8..], &(-2.25_f64).to_ne_bytes());
    }

    #[test]
    fn foreign_memory_is_borrowed_checked_and_released_by_the_last_reader() {
        use std::sync::atomic::{AtomicBool, Ordering};
        struct Region { words: Vec<u64>, dropped: Arc<AtomicBool> }
        unsafe impl ForeignBytes for Region {
            fn bytes(&self) -> &[u8] { unsafe { std::slice::from_raw_parts(self.words.as_ptr().cast(), self.words.len() * 8) } }
            fn as_any(&self) -> &dyn std::any::Any { self }
        }
        impl Drop for Region { fn drop(&mut self) { self.dropped.store(true, Ordering::Release); } }
        let dropped = Arc::new(AtomicBool::new(false));
        let words = vec![u64::from_ne_bytes([1, 0, 2, 0, 3, 0, 4, 0]), u64::from_ne_bytes([0, 1, 1, 0, 9, 9, 9, 9])];
        let owner: Arc<dyn ForeignBytes> = Arc::new(Region { words, dropped: dropped.clone() });
        let base = owner.bytes().as_ptr();
        let values = OwnedBuffer::<u16>::from_foreign(owner.clone(), 0, 4).unwrap();
        assert_eq!(values.as_slice(), &[1, 2, 3, 4]);
        assert_eq!(values.as_bytes().as_ptr(), base);
        assert_eq!(values.foreign().map(|(_, offset)| offset), Some(0));
        let mask = AnyBuffer::from_foreign(DType::Bool, owner.clone(), 8, 4).unwrap();
        assert_eq!(mask.as_bytes(), &[0, 1, 1, 0]);
        assert!(OwnedBuffer::<bool>::from_foreign(owner.clone(), 8, 5).is_err(), "9 is not a bool");
        assert!(OwnedBuffer::<u16>::from_foreign(owner.clone(), 1, 2).is_err(), "misaligned");
        assert!(OwnedBuffer::<u64>::from_foreign(owner.clone(), 8, 2).is_err(), "out of bounds");
        drop(owner);
        let reader = values.clone();
        drop((values, mask));
        assert!(!dropped.load(Ordering::Acquire));
        assert_eq!(reader.len(), 4);
        drop(reader);
        assert!(dropped.load(Ordering::Acquire));
    }

    #[test]
    fn recycle_waits_for_the_last_reader_and_length_is_per_owner() {
        let pool = BufferPool::<u16>::new();
        let mut builder = pool.builder(16);
        builder.extend_from_slice(&[10, 20, 30, 40]);
        let address = builder.as_mut_slice().as_ptr();
        let first = builder.finish();
        let reader = first.clone();
        drop(first);
        let mut other = pool.builder(16);
        other.extend_from_slice(&[90, 91]);
        assert_ne!(other.as_mut_slice().as_ptr(), address);
        assert_eq!(reader.as_slice(), &[10, 20, 30, 40]);
        drop(reader);
        let mut reused = pool.builder(16);
        reused.push(7);
        assert_eq!(reused.as_mut_slice().as_ptr(), address);
        let short = reused.finish();
        assert_eq!(short.len(), 1);
        assert_eq!(short.as_bytes().len(), 2);
        assert_eq!(other.as_mut_slice(), &[90, 91]);
    }

    #[test]
    fn readers_and_unfinished_builders_outlive_the_pool() {
        let pool = BufferPool::<f32>::new();
        let mut builder = pool.builder(8);
        builder.resize(2, 0.0);
        builder.as_mut_slice().copy_from_slice(&[3.0, 4.0]);
        let frozen = builder.finish();
        let mut pending = pool.builder(8);
        pending.push(5.0);
        pool.clear();
        drop(pool);
        assert_eq!(frozen.as_slice(), &[3.0, 4.0]);
        assert_eq!(pending.finish().as_slice(), &[5.0]);
    }
}
