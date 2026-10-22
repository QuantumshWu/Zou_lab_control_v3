//! Reused buffers for the large temporaries of one fit.
//!
//! Windows' process heap decommits freed blocks once more than 64 KiB lie
//! free, so every fresh temporary that size pays a page fault per 4 KiB
//! (about 2.5 ns an element: more than the arithmetic done on it).  Each
//! thread keeps a few such buffers for its next fit instead.

use std::cell::RefCell;
use std::ops::{Deref, DerefMut};
use std::thread::LocalKey;

/// Below this many bytes a buffer is the heap's cheap kind; not kept.
const SMALLEST_BYTES: usize = 32 * 1024;
/// At most this many bytes are kept per thread and element type.
const KEPT_BYTES: usize = 24 * 1024 * 1024;

pub trait Pooled: Copy + 'static {
    fn pool() -> &'static LocalKey<RefCell<Vec<Vec<Self>>>>;
}

macro_rules! pooled {
    ($t:ty, $name:ident) => {
        thread_local! { static $name: RefCell<Vec<Vec<$t>>> = const { RefCell::new(Vec::new()) }; }
        impl Pooled for $t {
            fn pool() -> &'static LocalKey<RefCell<Vec<Vec<$t>>>> { &$name }
        }
    };
}
pooled!(f64, FLOATS);
pooled!(bool, FLAGS);

/// An empty vector able to hold `len` elements, reused when one is kept.
fn take<T: Pooled>(len: usize) -> Vec<T> {
    if len * size_of::<T>() < SMALLEST_BYTES {
        return Vec::with_capacity(len);
    }
    let reused = T::pool()
        .try_with(|pool| {
            let mut pool = pool.borrow_mut();
            let fits = (0..pool.len()).filter(|&i| pool[i].capacity() >= len).min_by_key(|&i| pool[i].capacity());
            let index = fits.or_else(|| (0..pool.len()).max_by_key(|&i| pool[i].capacity()))?;
            Some(pool.swap_remove(index))
        })
        .ok()
        .flatten();
    match reused {
        Some(mut buffer) => {
            buffer.clear();
            buffer.reserve(len);
            buffer
        }
        None => Vec::with_capacity(len),
    }
}

fn keep<T: Pooled>(buffer: Vec<T>) {
    let bytes = buffer.capacity() * size_of::<T>();
    if !(SMALLEST_BYTES..=KEPT_BYTES).contains(&bytes) {
        return;
    }
    let _ = T::pool().try_with(|pool| {
        let mut pool = pool.borrow_mut();
        pool.push(buffer);
        // Over the budget: the smallest go first.
        while pool.iter().map(|b| b.capacity() * size_of::<T>()).sum::<usize>() > KEPT_BYTES {
            let smallest = (0..pool.len()).min_by_key(|&i| pool[i].capacity()).expect("non-empty");
            pool.swap_remove(smallest);
        }
    });
}

/// A vector whose storage goes back to the thread's pool when dropped.
pub struct Scratch<T: Pooled>(Vec<T>);

impl<T: Pooled> Scratch<T> {
    /// Empty, with room for `len`.
    pub fn with_capacity(len: usize) -> Self { Scratch(take(len)) }
    pub fn filled(len: usize, value: T) -> Self {
        let mut buffer = take(len);
        buffer.resize(len, value);
        Scratch(buffer)
    }
    pub fn copied(values: &[T]) -> Self {
        let mut buffer = take(values.len());
        buffer.extend_from_slice(values);
        Scratch(buffer)
    }
    pub fn collected(len: usize, values: impl Iterator<Item = T>) -> Self {
        let mut buffer = take(len);
        buffer.extend(values);
        Scratch(buffer)
    }
    /// The vector itself, leaving the pool's claim (see [`Scratch::adopt`]).
    pub fn into_vec(mut self) -> Vec<T> { std::mem::take(&mut self.0) }
    /// Returns `vector`'s storage to the pool when dropped.
    pub fn adopt(vector: Vec<T>) -> Self { Scratch(vector) }
}

impl<T: Pooled> Drop for Scratch<T> {
    fn drop(&mut self) { keep(std::mem::take(&mut self.0)); }
}

impl<T: Pooled> Deref for Scratch<T> {
    type Target = Vec<T>;
    fn deref(&self) -> &Vec<T> { &self.0 }
}

impl<T: Pooled> DerefMut for Scratch<T> {
    fn deref_mut(&mut self) -> &mut Vec<T> { &mut self.0 }
}
