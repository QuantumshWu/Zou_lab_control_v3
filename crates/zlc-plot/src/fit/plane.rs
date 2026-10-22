//! An image plane in its own storage, and the full-frame passes of the
//! regular-image refinement, each reading the plane once.
//!
//! A camera frame stays in the type it arrived in (u8/u16/i16/f32/f64);
//! every pass converts a sample to f64 (exact) and, in the refinement,
//! centres it on the frame's reference (`sample - reference`) as it reads
//! it -- the numbers the oracle's materialised float64 plane holds, without
//! the plane.  Work splits across threads only where each output keeps its
//! own sequence of operations: products per image row, sums per stripe
//! combined in stripe order.

use std::sync::Arc;

use rayon::prelude::*;

use super::blas;
use super::radial::STRIPE_ROWS;

/// An image's samples, row-major, in their own storage type.
#[derive(Debug, Clone)]
pub enum Samples {
    U8(Arc<[u8]>),
    U16(Arc<[u16]>),
    I16(Arc<[i16]>),
    F32(Arc<[f32]>),
    F64(Arc<[f64]>),
}

macro_rules! samples_from {
    ($($t:ty => $variant:ident),*) => {$(
        impl From<Vec<$t>> for Samples {
            fn from(values: Vec<$t>) -> Self { Samples::$variant(values.into()) }
        }
        impl From<Arc<[$t]>> for Samples {
            fn from(values: Arc<[$t]>) -> Self { Samples::$variant(values) }
        }
    )*};
}
samples_from!(u8 => U8, u16 => U16, i16 => I16, f32 => F32, f64 => F64);

/// Runs `$body` with `$s` bound to the samples' slice, whatever its type.
macro_rules! with_samples {
    ($samples:expr, |$s:ident| $body:expr) => {
        match $samples {
            Samples::U8($s) => $body,
            Samples::U16($s) => $body,
            Samples::I16($s) => $body,
            Samples::F32($s) => $body,
            Samples::F64($s) => $body,
        }
    };
}

impl Samples {
    pub fn len(&self) -> usize { with_samples!(self, |s| s.len()) }
    pub fn is_empty(&self) -> bool { self.len() == 0 }
    /// A float plane may hold non-finite samples; an integer one cannot.
    pub fn is_float(&self) -> bool { matches!(self, Samples::F32(_) | Samples::F64(_)) }
    #[inline]
    pub fn get(&self, index: usize) -> f64 { with_samples!(self, |s| s[index].value()) }
    /// The samples at `indices`, in the same storage.
    pub fn gather(&self, indices: impl Iterator<Item = usize>) -> Samples { with_samples!(self, |s| indices.map(|i| s[i]).collect::<Vec<_>>().into()) }
    pub fn to_f64(&self) -> Vec<f64> { with_samples!(self, |s| s.iter().map(|v| v.value()).collect()) }
}

/// A storage type, read as f64 exactly, with its fused product kernel.
pub trait Sample: Copy + Send + Sync + 'static {
    /// Every value of the type is finite (the integer storages).
    const FINITE: bool;
    fn value(self) -> f64;
    /// For four image rows over one 2048-column chunk and each vector: the
    /// `dgemv_t` kernel's four FMA lanes per row, on `sample - reference`.
    fn group_lanes(rows: [&[Self]; 4], reference: f64, xs: &[&[f64]]) -> Vec<Lanes4>;
}

/// One image row's four FMA lanes, for four rows.
pub type Lanes4 = [[f64; 4]; 4];

fn group_lanes_scalar<T: Sample>(rows: [&[T]; 4], reference: f64, xs: &[&[f64]]) -> Vec<Lanes4> {
    xs.iter()
        .map(|x| {
            let mut l = [[0.0_f64; 4]; 4];
            for i in (0..x.len() & !3).step_by(4) {
                for (k, row) in rows.iter().enumerate() {
                    for lane in 0..4 {
                        l[k][lane] = (row[i + lane].value() - reference).mul_add(x[i + lane], l[k][lane]);
                    }
                }
            }
            l
        })
        .collect()
}

fn avx2_fma() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[cfg(target_arch = "x86_64")]
mod avx {
    use std::arch::x86_64::*;

    use super::Lanes4;

    #[inline(always)]
    fn lanes4(v: __m256d) -> [f64; 4] { unsafe { std::mem::transmute(v) } }

    /// The kernel for one storage type: four converted lanes per row and
    /// step, centred, then one FMA per vector and row.
    macro_rules! group {
        ($name:ident, $t:ty, |$p:ident| $load:expr) => {
            #[inline]
            #[target_feature(enable = "avx2,fma")]
            unsafe fn held<const V: usize>(rows: [&[$t]; 4], reference: f64, xs: &[&[f64]]) -> Vec<Lanes4> {
                let n = xs[0].len() & !3;
                let r = _mm256_set1_pd(reference);
                let mut acc = [[_mm256_setzero_pd(); 4]; V];
                let mut i = 0;
                while i < n {
                    unsafe {
                        let mut cols = [r; 4];
                        for (col, row) in cols.iter_mut().zip(rows) {
                            let $p = row.as_ptr().add(i);
                            *col = _mm256_sub_pd($load, r);
                        }
                        for (v, lanes) in acc.iter_mut().enumerate() {
                            let xv = _mm256_loadu_pd(xs[v].as_ptr().add(i));
                            for (k, value) in lanes.iter_mut().enumerate() {
                                *value = _mm256_fmadd_pd(cols[k], xv, *value);
                            }
                        }
                    }
                    i += 4;
                }
                acc.iter().map(|l| [lanes4(l[0]), lanes4(l[1]), lanes4(l[2]), lanes4(l[3])]).collect()
            }

            #[target_feature(enable = "avx2,fma")]
            pub unsafe fn $name(rows: [&[$t]; 4], reference: f64, xs: &[&[f64]]) -> Vec<Lanes4> {
                unsafe {
                    match xs.len() {
                        1 => held::<1>(rows, reference, xs),
                        2 => held::<2>(rows, reference, xs),
                        3 => held::<3>(rows, reference, xs),
                        _ => xs.iter().flat_map(|x| held::<1>(rows, reference, std::slice::from_ref(x))).collect(),
                    }
                }
            }
        };
    }

    pub mod u8 {
        use super::*;
        group!(group, u8, |p| _mm256_cvtepi32_pd(_mm_cvtepu8_epi32(_mm_cvtsi32_si128((p as *const i32).read_unaligned()))));
    }
    pub mod u16 {
        use super::*;
        group!(group, u16, |p| _mm256_cvtepi32_pd(_mm_cvtepu16_epi32(_mm_loadl_epi64(p as *const __m128i))));
    }
    pub mod i16 {
        use super::*;
        group!(group, i16, |p| _mm256_cvtepi32_pd(_mm_cvtepi16_epi32(_mm_loadl_epi64(p as *const __m128i))));
    }
    pub mod f32 {
        use super::*;
        group!(group, f32, |p| _mm256_cvtps_pd(_mm_loadu_ps(p)));
    }
    pub mod f64 {
        use super::*;
        group!(group, f64, |p| _mm256_loadu_pd(p));
    }
}

macro_rules! sample_type {
    ($t:ident, $finite:expr) => {
        impl Sample for $t {
            const FINITE: bool = $finite;
            #[inline(always)]
            fn value(self) -> f64 { self as f64 }
            fn group_lanes(rows: [&[Self]; 4], reference: f64, xs: &[&[f64]]) -> Vec<Lanes4> {
                #[cfg(target_arch = "x86_64")]
                if avx2_fma() {
                    return unsafe { avx::$t::group(rows, reference, xs) };
                }
                group_lanes_scalar(rows, reference, xs)
            }
        }
    };
}
sample_type!(u8, true);
sample_type!(u16, true);
sample_type!(i16, true);
sample_type!(f32, false);
sample_type!(f64, false);

/// Runs `$body` (generic code using `mul_add` or wide conversions) compiled
/// for AVX2 and FMA when the CPU has them: a closure does not inherit a
/// `target_feature`, so each pass's per-task body goes through this.
macro_rules! fast {
    ($name:ident<$t:ident>($($arg:ident: $ty:ty),*) -> $ret:ty $body:block) => {
        #[inline(always)]
        fn $name<$t: Sample>($($arg: $ty),*) -> $ret $body
        paste_fast!($name<$t>($($arg: $ty),*) -> $ret);
    };
}
macro_rules! paste_fast {
    ($name:ident<$t:ident>($($arg:ident: $ty:ty),*) -> $ret:ty) => {
        mod $name {
            use super::*;
            #[cfg(target_arch = "x86_64")]
            #[target_feature(enable = "avx2,fma")]
            pub unsafe fn wide<$t: Sample>($($arg: $ty),*) -> $ret { super::$name::<$t>($($arg),*) }
            pub fn call<$t: Sample>($($arg: $ty),*) -> $ret {
                #[cfg(target_arch = "x86_64")]
                if avx2_fma() {
                    return unsafe { wide::<$t>($($arg),*) };
                }
                super::$name::<$t>($($arg),*)
            }
        }
    };
}

/// `_compiled_regular_centered_context`'s reference and moments without
/// its plane: the first usable sample, then per stripe the centred sum,
/// sum of squares and count of usable samples (stripes in parallel, as the
/// oracle's `prange`, and four at a time on a thread so their sums overlap),
/// combined stripe by stripe.
pub fn centred_context(samples: &Samples, mask: Option<&[bool]>, width: usize) -> [f64; 4] {
    with_samples!(samples, |s| centred_context_of(s, mask, width))
}

fast!(stripe_moments<T>(s: &[T], mask: Option<&[bool]>, reference: f64, first: usize, count: usize, stripe_size: usize) -> Vec<[f64; 3]> {
    // `count` whole stripes side by side (four at most): each its own
    // sequence, the sequences overlapping in time; a short last stripe
    // alone.
    let points = s.len();
    let mut out = Vec::with_capacity(count);
    if count == 4 && (first + 4) * stripe_size <= points && mask.is_none() {
        let slices: [&[T]; 4] = std::array::from_fn(|k| &s[(first + k) * stripe_size..(first + k + 1) * stripe_size]);
        let mut sums = [[0.0_f64; 2]; 4];
        let mut counts = [0i64; 4];
        for offset in 0..stripe_size {
            for k in 0..4 {
                let value = slices[k][offset].value();
                if T::FINITE || value.is_finite() {
                    let shifted = value - reference;
                    sums[k][0] += shifted;
                    sums[k][1] += shifted * shifted;
                    counts[k] += 1;
                }
            }
        }
        out.extend((0..4).map(|k| [sums[k][0], sums[k][1], counts[k] as f64]));
        return out;
    }
    for stripe in first..first + count {
        let (mut total, mut squares, mut n) = (0.0, 0.0, 0i64);
        for p in stripe * stripe_size..((stripe + 1) * stripe_size).min(points) {
            let value = s[p].value();
            if mask.is_none_or(|m| m[p]) && (T::FINITE || value.is_finite()) {
                let shifted = value - reference;
                total += shifted;
                squares += shifted * shifted;
                n += 1;
            }
        }
        out.push([total, squares, n as f64]);
    }
    out
});

fn centred_context_of<T: Sample>(s: &[T], mask: Option<&[bool]>, width: usize) -> [f64; 4] {
    let points = s.len();
    let usable = |p: usize, value: f64| mask.is_none_or(|m| m[p]) && value.is_finite();
    let reference = (0..points).map(|p| (p, s[p].value())).find(|&(p, v)| usable(p, v)).map_or(0.0, |(_, v)| v);
    let stripe_size = STRIPE_ROWS * width.max(1);
    let stripes = points.div_ceil(stripe_size);
    let moments: Vec<[f64; 3]> = (0..stripes.div_ceil(4))
        .into_par_iter()
        .flat_map_iter(|task| stripe_moments::call(s, mask, reference, 4 * task, 4.min(stripes - 4 * task), stripe_size))
        .collect();
    let mut context = [reference, 0.0, 0.0, 0.0];
    for m in 0..3 {
        let mut total = 0.0;
        for moment in &moments {
            total += moment[m];
        }
        context[m + 1] = total;
    }
    context
}

/// Below this many samples a pass runs on the calling thread.
const PARALLEL_SAMPLES: usize = 1 << 16;

/// `image @ v` for each of `vectors` (numba's `dgemv('T')` on the
/// C-contiguous plane, OpenBLAS's order per image row), the plane read
/// once and centred as it is read.  Image rows are independent outputs:
/// groups of four (the kernel's column blocks) split across threads.
pub fn plane_products(samples: &Samples, reference: f64, width: usize, height: usize, vectors: &[&[f64]]) -> Vec<Vec<f64>> {
    with_samples!(samples, |s| products_of(s, reference, width, height, vectors))
}

fn products_of<T: Sample>(s: &[T], reference: f64, width: usize, height: usize, vectors: &[&[f64]]) -> Vec<Vec<f64>> {
    let m = width;
    let mut out: Vec<Vec<f64>> = vec![vec![0.0; height]; vectors.len()];
    if m == 0 || height == 0 {
        return out;
    }
    // The kernel's 2048-column chunks (each a multiple of four), then the
    // `m & 3` remainder columns.
    const NBMAX: usize = 2048;
    let m3 = m & 3;
    let mut chunks = Vec::new();
    let mut start = 0;
    while start + NBMAX <= m - m3 {
        chunks.push((start, NBMAX));
        start += NBMAX;
    }
    if (m & (NBMAX - 1)) - m3 > 0 && start < m - m3 {
        chunks.push((start, (m & (NBMAX - 1)) - m3));
        start += (m & (NBMAX - 1)) - m3;
    }
    let tail_start = start;
    let row = |j: usize| &s[j * m..(j + 1) * m];
    let value = |j: usize, i: usize| row(j)[i].value() - reference;
    // Each image row's own sequence: its chunks' lane sums, then the
    // remainder columns.
    let remainder = |j: usize, x: &[f64], y: f64| -> f64 {
        let r = tail_start;
        match m3 {
            3 => y + value(j, r + 2).mul_add(x[r + 2], value(j, r).mul_add(x[r], value(j, r + 1) * x[r + 1])),
            2 => y + value(j, r).mul_add(x[r], value(j, r + 1) * x[r + 1]),
            1 => value(j, r).mul_add(x[r], y),
            _ => y,
        }
    };
    let group = |g: usize| -> Vec<[f64; 4]> {
        let mut y = vec![[0.0_f64; 4]; vectors.len()];
        for &(first, length) in &chunks {
            let rows = [&row(4 * g)[first..first + length], &row(4 * g + 1)[first..first + length], &row(4 * g + 2)[first..first + length], &row(4 * g + 3)[first..first + length]];
            let xs: Vec<&[f64]> = vectors.iter().map(|x| &x[first..first + length]).collect();
            for (yv, lanes) in y.iter_mut().zip(T::group_lanes(rows, reference, &xs)) {
                for (k, l) in lanes.iter().enumerate() {
                    yv[k] += (l[0] + l[2]) + (l[1] + l[3]);
                }
            }
        }
        for (yv, x) in y.iter_mut().zip(vectors) {
            for (k, value) in yv.iter_mut().enumerate() {
                *value = remainder(4 * g + k, x, *value);
            }
        }
        y
    };
    let groups = height / 4;
    let results: Vec<Vec<[f64; 4]>> = if s.len() >= PARALLEL_SAMPLES { (0..groups).into_par_iter().map(group).collect() } else { (0..groups).map(group).collect() };
    for (g, y) in results.iter().enumerate() {
        for (column, values) in out.iter_mut().zip(y) {
            column[4 * g..4 * g + 4].copy_from_slice(values);
        }
    }
    // The `n & 3` rows: a 2-lane plain pair, then a 4-lane plain single.
    let mut j = 4 * groups;
    if (height - j) & 2 != 0 {
        for (column, x) in out.iter_mut().zip(vectors) {
            for k in 0..2 {
                let mut y = 0.0;
                for &(first, length) in &chunks {
                    let mut l = [0.0_f64; 2];
                    for i in (first..first + length).step_by(2) {
                        for (lane, value_l) in l.iter_mut().enumerate() {
                            *value_l += value(j + k, i + lane) * x[i + lane];
                        }
                    }
                    y = (l[0] + l[1]).mul_add(1.0, y);
                }
                column[j + k] = remainder(j + k, x, y);
            }
        }
        j += 2;
    }
    if (height - j) & 1 != 0 {
        for (column, x) in out.iter_mut().zip(vectors) {
            let mut y = 0.0;
            for &(first, length) in &chunks {
                let mut l = [0.0_f64; 4];
                for i in (first..first + length).step_by(4) {
                    for (lane, value_l) in l.iter_mut().enumerate() {
                        *value_l += value(j, i + lane) * x[i + lane];
                    }
                }
                y = ((l[0] + l[2]) + (l[1] + l[3])).mul_add(1.0, y);
            }
            column[j] = remainder(j, x, y);
        }
    }
    out
}

/// `_compiled_regular_residual_rss` on the centred plane: per stripe the
/// residuals `amplitude y x + offset - observed` fed straight into `ddot`'s
/// sixteen FMA lanes (and its plain tail), the stripes (in parallel) added
/// in order.
pub fn residual_rss(samples: &Samples, reference: f64, amplitude: f64, offset: f64, x_basis: &[f64], y_basis: &[f64]) -> f64 {
    with_samples!(samples, |s| residual_rss_of(s, reference, amplitude, offset, x_basis, y_basis))
}

fast!(stripe_rss<T>(s: &[T], reference: f64, amplitude: f64, offset: f64, x_basis: &[f64], y_basis: &[f64], start: usize) -> f64 {
    let width = x_basis.len();
    let stop = y_basis.len().min(start + STRIPE_ROWS);
    let length = (stop - start) * width;
    let blocked = length & !15;
    // Residuals `e..` of the stripe, row piece by row piece (a function,
    // not a closure, so it compiles with the caller's features).
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn fill<T: Sample>(s: &[T], reference: f64, amplitude: f64, offset: f64, x_basis: &[f64], y_basis: &[f64], start: usize, from: usize, out: &mut [f64]) {
        let width = x_basis.len();
        let (mut e, mut filled) = (from, 0);
        while filled < out.len() {
            let (row, column) = (start + e / width, e % width);
            let take = (width - column).min(out.len() - filled);
            let scaled = amplitude * y_basis[row];
            let observed = &s[row * width + column..row * width + column + take];
            for ((value, &x), &v) in out[filled..filled + take].iter_mut().zip(&x_basis[column..column + take]).zip(observed) {
                *value = scaled * x + offset - (v.value() - reference);
            }
            e += take;
            filled += take;
        }
    }
    let fill = |from: usize, out: &mut [f64]| fill(s, reference, amplitude, offset, x_basis, y_basis, start, from, out);
    const PIECE: usize = 2048;
    let mut buffer = [0.0_f64; PIECE];
    let mut lanes = [0.0_f64; 16];
    let mut e = 0;
    while e < blocked {
        let n = PIECE.min(blocked - e);
        fill(e, &mut buffer[..n]);
        lanes = blas::ddot_lanes(lanes, &buffer[..n]);
        e += n;
    }
    let mut dot = if blocked > 0 { blas::ddot_reduce(&lanes) } else { 0.0 };
    fill(blocked, &mut buffer[..length - blocked]);
    for &r in &buffer[..length - blocked] {
        dot += r * r;
    }
    dot
});

fn residual_rss_of<T: Sample>(s: &[T], reference: f64, amplitude: f64, offset: f64, x_basis: &[f64], y_basis: &[f64]) -> f64 {
    let starts: Vec<usize> = (0..y_basis.len()).step_by(STRIPE_ROWS).collect();
    let stripe = |start: usize| stripe_rss::call(s, reference, amplitude, offset, x_basis, y_basis, start);
    let parts: Vec<f64> = if s.len() >= PARALLEL_SAMPLES { starts.par_iter().map(|&start| stripe(start)).collect() } else { starts.iter().map(|&start| stripe(start)).collect() };
    let mut rss = 0.0;
    for part in parts {
        rss += part;
    }
    rss
}
