//! The OpenBLAS 0.3.30 "Haswell" level-1/2 kernels numba's LAPACK calls,
//! reproduced operation for operation.  Each was identified by probing the
//! library itself (`scipy_openblas`, the build numba links) with random
//! operands until a model matched every case bit for bit; the probes live
//! with the golden generators.
//!
//! Matrices are column-major views (`Mat`), as LAPACK sees them.

/// A column-major matrix (or a view into one at an offset).
#[derive(Debug, Clone)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    /// Column-major storage, leading dimension `rows`.
    pub data: Vec<f64>,
}

impl Mat {
    pub fn zeros(rows: usize, cols: usize) -> Self { Mat { rows, cols, data: vec![0.0; rows * cols] } }
    pub fn identity(n: usize) -> Self {
        let mut m = Mat::zeros(n, n);
        for i in 0..n {
            m.set(i, i, 1.0);
        }
        m
    }
    /// From a row-major `rows x cols` slice.
    pub fn from_row_major(rows: usize, cols: usize, values: &[f64]) -> Self {
        let mut m = Mat::zeros(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                m.set(i, j, values[i * cols + j]);
            }
        }
        m
    }
    #[inline(always)]
    pub fn get(&self, i: usize, j: usize) -> f64 { self.data[j * self.rows + i] }
    #[inline(always)]
    pub fn set(&mut self, i: usize, j: usize, v: f64) { self.data[j * self.rows + i] = v; }
    pub fn to_row_major(&self) -> Vec<f64> {
        let mut out = vec![0.0; self.rows * self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                out[i * self.cols + j] = self.get(i, j);
            }
        }
        out
    }
}

// --------------------------------------------------------------------------
// Inner loops: scalar, or AVX lanes doing the same IEEE operation per lane
// (LLVM leaves these loops scalar, their order being fixed).
// --------------------------------------------------------------------------

/// The vectorisable inner loops the kernels below share.
struct Lanes {
    /// `lanes[l] = fma(x[i], y[i], lanes[l])` for `i % 16 == l`, over a
    /// length that is a multiple of 16, from the given lanes.
    dot16: unsafe fn([f64; 16], &[f64], &[f64]) -> [f64; 16],
    /// Four columns' four-lane FMA sums against each of `xs` (length a
    /// multiple of 4), the columns read once for all of them.
    fma4x4_multi: unsafe fn(&[&[f64]], [&[f64]; 4]) -> Vec<[[f64; 4]; 4]>,
    /// Two columns' two-lane `l += a * x` sums (length a multiple of 2).
    mul2x2: unsafe fn(&[f64], [&[f64]; 2]) -> [[f64; 2]; 2],
    /// One column's four-lane `l += a * x` sums (length a multiple of 4).
    mul1x4: unsafe fn(&[f64], &[f64]) -> [f64; 4],
    /// `y[i] = fma(a, x[i], y[i])`.
    axpy: unsafe fn(f64, &[f64], &mut [f64]),
    /// `acc[i n + j] = fma(a_r[i], b_r[j], acc[i n + j])`, row after row.
    outer: unsafe fn(&[f64], &[f64], usize, usize, &mut [f64]),
}

mod scalar {
    pub fn dot16(mut lanes: [f64; 16], x: &[f64], y: &[f64]) -> [f64; 16] {
        for (xb, yb) in x.chunks_exact(16).zip(y.chunks_exact(16)) {
            for l in 0..16 {
                lanes[l] = xb[l].mul_add(yb[l], lanes[l]);
            }
        }
        lanes
    }
    pub fn fma4x4_multi(xs: &[&[f64]], a: [&[f64]; 4]) -> Vec<[[f64; 4]; 4]> {
        xs.iter()
            .map(|x| {
                let mut l = [[0.0_f64; 4]; 4];
                for (i, x4) in x.chunks_exact(4).enumerate() {
                    for (k, column) in a.iter().enumerate() {
                        for lane in 0..4 {
                            l[k][lane] = column[4 * i + lane].mul_add(x4[lane], l[k][lane]);
                        }
                    }
                }
                l
            })
            .collect()
    }
    pub fn mul2x2(x: &[f64], a: [&[f64]; 2]) -> [[f64; 2]; 2] {
        let mut l = [[0.0_f64; 2]; 2];
        for (i, x2) in x.chunks_exact(2).enumerate() {
            for (k, column) in a.iter().enumerate() {
                for lane in 0..2 {
                    l[k][lane] += column[2 * i + lane] * x2[lane];
                }
            }
        }
        l
    }
    pub fn mul1x4(x: &[f64], a: &[f64]) -> [f64; 4] {
        let mut l = [0.0_f64; 4];
        for (x4, a4) in x.chunks_exact(4).zip(a.chunks_exact(4)) {
            for lane in 0..4 {
                l[lane] += a4[lane] * x4[lane];
            }
        }
        l
    }
    pub fn axpy(a: f64, x: &[f64], y: &mut [f64]) {
        for (yi, &xi) in y.iter_mut().zip(x) {
            *yi = a.mul_add(xi, *yi);
        }
    }
    pub fn outer(a: &[f64], b: &[f64], m: usize, n: usize, acc: &mut [f64]) {
        for (ar, br) in a.chunks_exact(m).zip(b.chunks_exact(n)) {
            for (i, &ai) in ar.iter().enumerate() {
                for (value, &bj) in acc[i * n..(i + 1) * n].iter_mut().zip(br) {
                    *value = ai.mul_add(bj, *value);
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod avx {
    use std::arch::x86_64::*;

    #[inline(always)]
    fn lanes4(v: __m256d) -> [f64; 4] { unsafe { std::mem::transmute(v) } }

    #[target_feature(enable = "avx,fma")]
    pub unsafe fn dot16(lanes: [f64; 16], x: &[f64], y: &[f64]) -> [f64; 16] {
        let n = x.len().min(y.len()) & !15;
        let (px, py) = (x.as_ptr(), y.as_ptr());
        let mut acc: [__m256d; 4] = std::array::from_fn(|k| unsafe { _mm256_loadu_pd(lanes[4 * k..].as_ptr()) });
        let mut i = 0;
        while i < n {
            for (k, value) in acc.iter_mut().enumerate() {
                unsafe { *value = _mm256_fmadd_pd(_mm256_loadu_pd(px.add(i + 4 * k)), _mm256_loadu_pd(py.add(i + 4 * k)), *value) };
            }
            i += 16;
        }
        let mut out = [0.0; 16];
        for (k, value) in acc.iter().enumerate() {
            out[4 * k..4 * k + 4].copy_from_slice(&lanes4(*value));
        }
        out
    }

    /// `V` vectors at once: the four columns' lanes loaded once per step.
    #[inline]
    #[target_feature(enable = "avx,fma")]
    unsafe fn fma4x4_held<const V: usize>(xs: &[&[f64]], a: [&[f64]; 4]) -> Vec<[[f64; 4]; 4]> {
        let n = a.iter().map(|c| c.len()).chain(xs.iter().map(|x| x.len())).min().unwrap_or(0) & !3;
        let mut acc = [[_mm256_setzero_pd(); 4]; V];
        let mut i = 0;
        while i < n {
            unsafe {
                let cols = [_mm256_loadu_pd(a[0].as_ptr().add(i)), _mm256_loadu_pd(a[1].as_ptr().add(i)), _mm256_loadu_pd(a[2].as_ptr().add(i)), _mm256_loadu_pd(a[3].as_ptr().add(i))];
                for (v, lanes) in acc.iter_mut().enumerate() {
                    let xv = _mm256_loadu_pd(xs[v].as_ptr().add(i));
                    for (k, value) in lanes.iter_mut().enumerate() {
                        *value = _mm256_fmadd_pd(cols[k], xv, *value);
                    }
                }
            }
            i += 4;
        }
        acc.iter().map(|lanes| [lanes4(lanes[0]), lanes4(lanes[1]), lanes4(lanes[2]), lanes4(lanes[3])]).collect()
    }

    #[target_feature(enable = "avx,fma")]
    pub unsafe fn fma4x4_multi(xs: &[&[f64]], a: [&[f64]; 4]) -> Vec<[[f64; 4]; 4]> {
        match xs.len() {
            1 => unsafe { fma4x4_held::<1>(xs, a) },
            2 => unsafe { fma4x4_held::<2>(xs, a) },
            3 => unsafe { fma4x4_held::<3>(xs, a) },
            _ => xs.iter().flat_map(|x| unsafe { fma4x4_held::<1>(std::slice::from_ref(x), a) }).collect(),
        }
    }

    #[target_feature(enable = "avx,fma")]
    pub unsafe fn mul2x2(x: &[f64], a: [&[f64]; 2]) -> [[f64; 2]; 2] {
        let n = x.len().min(a[0].len()).min(a[1].len()) & !1;
        let mut acc = [_mm_setzero_pd(); 2];
        let mut i = 0;
        while i < n {
            let xv = unsafe { _mm_loadu_pd(x.as_ptr().add(i)) };
            for (k, value) in acc.iter_mut().enumerate() {
                unsafe { *value = _mm_add_pd(*value, _mm_mul_pd(_mm_loadu_pd(a[k].as_ptr().add(i)), xv)) };
            }
            i += 2;
        }
        unsafe { std::mem::transmute(acc) }
    }

    #[target_feature(enable = "avx,fma")]
    pub unsafe fn mul1x4(x: &[f64], a: &[f64]) -> [f64; 4] {
        let n = x.len().min(a.len()) & !3;
        let mut acc = _mm256_setzero_pd();
        let mut i = 0;
        while i < n {
            unsafe { acc = _mm256_add_pd(acc, _mm256_mul_pd(_mm256_loadu_pd(a.as_ptr().add(i)), _mm256_loadu_pd(x.as_ptr().add(i)))) };
            i += 4;
        }
        lanes4(acc)
    }

    #[target_feature(enable = "avx,fma")]
    pub unsafe fn axpy(a: f64, x: &[f64], y: &mut [f64]) {
        let n = x.len().min(y.len());
        let av = _mm256_set1_pd(a);
        let mut i = 0;
        while i + 4 <= n {
            unsafe {
                let p = y.as_mut_ptr().add(i);
                _mm256_storeu_pd(p, _mm256_fmadd_pd(av, _mm256_loadu_pd(x.as_ptr().add(i)), _mm256_loadu_pd(p)));
            }
            i += 4;
        }
        for j in i..n {
            y[j] = a.mul_add(x[j], y[j]);
        }
    }

    /// `outer` for `M` rows of `Q` quads, the accumulators held in
    /// registers (`b` rows padded to whole quads; padding lanes discarded).
    #[inline]
    #[target_feature(enable = "avx,fma")]
    unsafe fn outer_held<const M: usize, const Q: usize>(a: &[f64], b: &[f64], n: usize, acc: &mut [f64]) {
        unsafe {
            let mut held = [[_mm256_setzero_pd(); Q]; M];
            let mut padded = [0.0_f64; 8];
            for (i, row) in held.iter_mut().enumerate() {
                for (q, value) in row.iter_mut().enumerate() {
                    padded[..4].fill(0.0);
                    let lanes = (n - 4 * q).min(4);
                    padded[..lanes].copy_from_slice(&acc[i * n + 4 * q..i * n + 4 * q + lanes]);
                    *value = _mm256_loadu_pd(padded.as_ptr());
                }
            }
            for (ar, br) in a.chunks_exact(M).zip(b.chunks_exact(n)) {
                let bq: [__m256d; Q] = if n == 4 * Q {
                    std::array::from_fn(|q| _mm256_loadu_pd(br.as_ptr().add(4 * q)))
                } else {
                    padded[..n].copy_from_slice(br);
                    std::array::from_fn(|q| _mm256_loadu_pd(padded.as_ptr().add(4 * q)))
                };
                for (i, row) in held.iter_mut().enumerate() {
                    let av = _mm256_set1_pd(ar[i]);
                    for (q, value) in row.iter_mut().enumerate() {
                        *value = _mm256_fmadd_pd(av, bq[q], *value);
                    }
                }
            }
            for (i, row) in held.iter().enumerate() {
                for (q, value) in row.iter().enumerate() {
                    let lanes = (n - 4 * q).min(4);
                    acc[i * n + 4 * q..i * n + 4 * q + lanes].copy_from_slice(&lanes4(*value)[..lanes]);
                }
            }
        }
    }

    #[target_feature(enable = "avx,fma")]
    pub unsafe fn outer(a: &[f64], b: &[f64], m: usize, n: usize, acc: &mut [f64]) {
        macro_rules! held {
            ($($m:literal),*) => {
                match (m, n.div_ceil(4)) {
                    $(($m, 1) => return unsafe { outer_held::<$m, 1>(a, b, n, acc) },
                      ($m, 2) => return unsafe { outer_held::<$m, 2>(a, b, n, acc) },)*
                    _ => {}
                }
            };
        }
        held!(1, 2, 3, 4, 5, 6);
        let quads = n / 4;
        for (ar, br) in a.chunks_exact(m).zip(b.chunks_exact(n)) {
            for (i, &ai) in ar.iter().enumerate() {
                let row = &mut acc[i * n..(i + 1) * n];
                let av = _mm256_set1_pd(ai);
                for q in 0..quads {
                    unsafe {
                        let p = row.as_mut_ptr().add(4 * q);
                        _mm256_storeu_pd(p, _mm256_fmadd_pd(av, _mm256_loadu_pd(br.as_ptr().add(4 * q)), _mm256_loadu_pd(p)));
                    }
                }
                for j in 4 * quads..n {
                    row[j] = ai.mul_add(br[j], row[j]);
                }
            }
        }
    }
}

static SCALAR: Lanes = Lanes { dot16: scalar::dot16, fma4x4_multi: scalar::fma4x4_multi, mul2x2: scalar::mul2x2, mul1x4: scalar::mul1x4, axpy: scalar::axpy, outer: scalar::outer };
#[cfg(target_arch = "x86_64")]
static AVX: Lanes = Lanes { dot16: avx::dot16, fma4x4_multi: avx::fma4x4_multi, mul2x2: avx::mul2x2, mul1x4: avx::mul1x4, axpy: avx::axpy, outer: avx::outer };

/// The AVX loops when the CPU has AVX and FMA, else the scalar ones.
fn lanes() -> &'static Lanes {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx") && std::arch::is_x86_feature_detected!("fma") {
        return &AVX;
    }
    &SCALAR
}

/// `ddot`'s sixteen FMA lanes carried on over `x . x` (a multiple of 16
/// long): a sum fed piece by piece, its lanes as one call would have them.
pub fn ddot_lanes(lanes: [f64; 16], x: &[f64]) -> [f64; 16] { unsafe { (self::lanes().dot16)(lanes, x, x) } }

/// `ddot`'s reduction of its sixteen lanes (when it had any blocks).
pub fn ddot_reduce(lanes: &[f64; 16]) -> f64 {
    let acc = |k: usize, l: usize| lanes[4 * k + l];
    let t: [[f64; 2]; 4] = std::array::from_fn(|k| [acc(k, 0) + acc(k, 2), acc(k, 1) + acc(k, 3)]);
    let t4 = [t[0][0] + t[1][0], t[0][1] + t[1][1]];
    let t6 = [t[2][0] + t[3][0], t[2][1] + t[3][1]];
    (t4[0] + t6[0]) + (t4[1] + t6[1])
}

/// `ddot` with unit strides: 16-element blocks into four 4-lane FMA
/// accumulators, reduced lane-pairwise, then an uncontracted tail.
#[inline(always)]
fn ddot_body(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    let n1 = n & !15;
    let mut dot = 0.0_f64;
    if n1 > 0 {
        // Lane 4k + l is accumulator k's lane l.
        dot = ddot_reduce(&unsafe { (lanes().dot16)([0.0; 16], &x[..n1], &y[..n1]) });
    }
    for i in n1..n {
        dot += y[i] * x[i];
    }
    dot
}

/// `daxpy` (`y += a x`): FMA over 16-element blocks, plain tail.
#[inline(always)]
fn daxpy_body(a: f64, x: &[f64], y: &mut [f64]) {
    let n = x.len();
    let n1 = n & !15;
    unsafe { (lanes().axpy)(a, &x[..n1], &mut y[..n1]) };
    for (yi, &xi) in y[n1..n].iter_mut().zip(&x[n1..n]) {
        *yi += a * xi;
    }
}

/// `dnrm2`: the x87 kernel (four accumulators over blocks of eight, the
/// tail into the first, combined `D + ((C + A) + B)`), with Windows' 53-bit
/// x87 precision; the extended exponent range is reproduced by an exact
/// power-of-two scaling when the operands would leave double's range.
pub fn dnrm2(x: &[f64]) -> f64 {
    // The largest magnitude, in four lanes (a maximum is exact in any order).
    let mut largest = [0.0_f64; 4];
    let mut nan = false;
    let quads = x.chunks_exact(4);
    let rest = quads.remainder();
    for quad in quads {
        for lane in 0..4 {
            let a = quad[lane].abs();
            nan |= a.is_nan();
            largest[lane] = if a > largest[lane] { a } else { largest[lane] };
        }
    }
    for &v in rest {
        let a = v.abs();
        nan |= a.is_nan();
        largest[0] = if a > largest[0] { a } else { largest[0] };
    }
    if nan {
        return f64::NAN;
    }
    let largest = largest.iter().fold(0.0_f64, |m, &v| if v > m { v } else { m });
    if largest.is_infinite() {
        return f64::INFINITY;
    }
    if largest == 0.0 {
        return 0.0;
    }
    let exponent = (largest.log2().floor()) as i32;
    let scaled = !(-480..=480).contains(&exponent);
    // `2^k` for every representable `k` (subnormal below -1022).
    let pow2 = |k: i32| if k >= -1022 { f64::from_bits(((k + 1023) as u64) << 52) } else { f64::from_bits(1u64 << (k + 1074)) };
    // Scaled so the largest is near 1: one multiply, or for a subnormal
    // largest (a factor past 2^1023) two, both exact.
    let scale = |v: f64| if -exponent <= 1023 { v * pow2(-exponent) } else { v * pow2(1023) * pow2(-exponent - 1023) };
    let copy: Vec<f64>;
    let values = if scaled {
        copy = x.iter().map(|&v| scale(v)).collect();
        &copy[..]
    } else {
        x
    };
    // Lanes A, B, C, D take elements 0, 1, 2, 3 of every four in the
    // blocks of eight; the tail goes into A.
    let blocked = (values.len() >> 3) << 3;
    let [mut a, b, c, d] = unsafe { (lanes().mul1x4)(&values[..blocked], &values[..blocked]) };
    for &p in &values[blocked..] {
        a += p * p;
    }
    let root = (d + ((c + a) + b)).sqrt();
    if scaled { root * pow2(exponent) } else { root }
}

/// `dsymv` with `uplo='L'`, unit strides, `beta = 0` already applied to
/// `y`: OpenBLAS' 4-column blocks (the scalar branch every matrix under
/// sixteen rows takes), then single columns.  `a(i, j)` reads the lower
/// triangle.
pub fn dsymv_lower(m: usize, alpha: f64, a: &dyn Fn(usize, usize) -> f64, x: &[f64], y: &mut [f64]) {
    let offset = m;
    let offset1 = (offset / 4) * 4;
    let mut j = 0;
    while j < offset1 {
        let t1 = [alpha * x[j], alpha * x[j + 1], alpha * x[j + 2], alpha * x[j + 3]];
        let mut t2 = [0.0_f64; 4];
        for k in 0..4 {
            y[j + k] += t1[k] * a(j + k, j + k);
        }
        let from = j + 1;
        if m - from >= 12 {
            // The 4x4 microkernel branch: never reached by the small
            // Gauss-Newton matrices this port solves.
            dsymv_block_kernel(m, j, &t1, &mut t2, a, x, y);
        } else {
            for k in 0..3 {
                for i in (j + 1 + k)..(j + 4) {
                    y[i] += t1[k] * a(i, j + k);
                    t2[k] += a(i, j + k) * x[i];
                }
            }
            for i in (j + 4)..m {
                for k in 0..4 {
                    y[i] += t1[k] * a(i, j + k);
                    t2[k] += a(i, j + k) * x[i];
                }
            }
        }
        for k in 0..4 {
            y[j + k] += alpha * t2[k];
        }
        j += 4;
    }
    for j in offset1..offset {
        let t1 = alpha * x[j];
        let mut t2 = 0.0;
        y[j] += t1 * a(j, j);
        for i in (j + 1)..m {
            y[i] += t1 * a(i, j);
            t2 += a(i, j) * x[i];
        }
        y[j] += alpha * t2;
    }
}

fn dsymv_block_kernel(m: usize, j: usize, t1: &[f64; 4], t2: &mut [f64; 4], a: &dyn Fn(usize, usize) -> f64, x: &[f64], y: &mut [f64]) {
    // Same ordering as the scalar branch; only reached for m >= j + 13.
    for k in 0..3 {
        for i in (j + 1 + k)..(j + 4) {
            y[i] += t1[k] * a(i, j + k);
            t2[k] += a(i, j + k) * x[i];
        }
    }
    for i in (j + 4)..m {
        for k in 0..4 {
            y[i] += t1[k] * a(i, j + k);
            t2[k] += a(i, j + k) * x[i];
        }
    }
}

/// `dsyr2` lower: per column two axpys, `alpha*x[j]` with `y`, then
/// `alpha*y[j]` with `x`.
pub fn dsyr2_lower(n: usize, alpha: f64, x: &[f64], y: &[f64], a: &mut Mat, r0: usize, c0: usize) {
    let mut column = vec![0.0; n];
    for j in 0..n {
        let len = n - j;
        for (pass, (scale, vector)) in [(alpha * x[j], y), (alpha * y[j], x)].into_iter().enumerate() {
            let _ = pass;
            for k in 0..len {
                column[k] = a.get(r0 + j + k, c0 + j);
            }
            daxpy(scale, &vector[j..n], &mut column[..len]);
            for k in 0..len {
                a.set(r0 + j + k, c0 + j, column[k]);
            }
        }
    }
}

/// `dgemv('T')` on the `m x n` view at `(r0, c0)`, `beta = 0`.
#[inline(always)]
fn dgemv_t_body(m: usize, n: usize, alpha: f64, a: &Mat, r0: usize, c0: usize, x: &[f64]) -> Vec<f64> {
    let columns: Vec<&[f64]> = (0..n).map(|j| &a.data[(c0 + j) * a.rows + r0..(c0 + j) * a.rows + r0 + m]).collect();
    dgemv_t_core(m, &columns, &[x], alpha, 4 * (n / 4)).remove(0)
}

/// `dgemv('T')` for one vector or several, over the given columns (each
/// `m` long): for each 2048-row chunk, the first `grouped` columns (a
/// multiple of four, read once for every vector) in 4-column FMA lanes
/// summed `(l0+l2)+(l1+l3)`, the rest -- the kernel's `n & 3` -- as a
/// 2-lane plain pair and a 4-lane plain single; then the `m & 3` remainder
/// rows.
#[inline(always)]
fn dgemv_t_core(m: usize, columns: &[&[f64]], vectors: &[&[f64]], alpha: f64, grouped: usize) -> Vec<Vec<f64>> {
    let n = columns.len();
    let mut y = vec![vec![0.0_f64; n]; vectors.len()];
    if m < 1 || n < 1 {
        return y;
    }
    const NBMAX: isize = 2048;
    let mi = m as isize;
    let m3 = (m & 3) as isize;
    let mut m1 = mi & -4;
    let m2 = (mi & (NBMAX - 1)) - m3;
    let mut nb = NBMAX;
    let mut row0 = 0usize;
    let kernels = lanes();
    while nb == NBMAX {
        m1 -= nb;
        if m1 < 0 {
            if m2 == 0 {
                break;
            }
            nb = m2;
        }
        let rows = nb as usize;
        let column = |j: usize| &columns[j][row0..row0 + rows];
        let xs: Vec<&[f64]> = vectors.iter().map(|x| &x[row0..row0 + rows]).collect();
        for col in (0..grouped).step_by(4) {
            let l = unsafe { (kernels.fma4x4_multi)(&xs, [column(col), column(col + 1), column(col + 2), column(col + 3)]) };
            for (yv, lv) in y.iter_mut().zip(&l) {
                for (k, lanes) in lv.iter().enumerate() {
                    let t = (lanes[0] + lanes[2]) + (lanes[1] + lanes[3]);
                    yv[col + k] += t * alpha;
                }
            }
        }
        let mut col = grouped;
        if (n - grouped) & 2 != 0 {
            for (yv, x) in y.iter_mut().zip(&xs) {
                let l = unsafe { (kernels.mul2x2)(x, [column(col), column(col + 1)]) };
                for (k, lanes) in l.iter().enumerate() {
                    let t = lanes[0] + lanes[1];
                    yv[col + k] = t.mul_add(alpha, yv[col + k]);
                }
            }
            col += 2;
        }
        if (n - grouped) & 1 != 0 {
            for (yv, x) in y.iter_mut().zip(&xs) {
                let l = unsafe { (kernels.mul1x4)(x, column(col)) };
                let t = (l[0] + l[2]) + (l[1] + l[3]);
                yv[col] = t.mul_add(alpha, yv[col]);
            }
        }
        row0 += rows;
    }
    if m3 == 0 {
        return y;
    }
    let r = row0;
    for (yv, x) in y.iter_mut().zip(vectors) {
        match m3 {
            3 => {
                let xt = [x[r] * alpha, x[r + 1] * alpha, x[r + 2] * alpha];
                for (value, c) in yv.iter_mut().zip(columns) {
                    *value += c[r + 2].mul_add(xt[2], c[r].mul_add(xt[0], c[r + 1] * xt[1]));
                }
            }
            2 => {
                let xt = [x[r] * alpha, x[r + 1] * alpha];
                for (value, c) in yv.iter_mut().zip(columns) {
                    *value += c[r].mul_add(xt[0], c[r + 1] * xt[1]);
                }
            }
            _ => {
                let xt = x[r] * alpha;
                for (value, c) in yv.iter_mut().zip(columns) {
                    *value = c[r].mul_add(xt, *value);
                }
            }
        }
    }
    y
}

/// `dgemv('N')` with a strided `x` (the only form `DLARF('R')` calls),
/// `beta = 0`: groups of four columns through the FMA pair kernel, single
/// columns `y + a*(x*alpha)`, remainder rows as a plain dot product.
#[inline(always)]
fn dgemv_n_strided_body(m: usize, n: usize, alpha: f64, a: &Mat, r0: usize, c0: usize, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0_f64; m];
    if m < 1 || n < 1 {
        return y;
    }
    let get = |i: usize, j: usize| a.get(r0 + i, c0 + j);
    let m3 = m & 3;
    let mm = m - m3;
    if mm > 0 {
        let mut col = 0;
        for _ in 0..(n >> 2) {
            for (i, value) in y.iter_mut().enumerate().take(mm) {
                let mut t4 = get(i, col).mul_add(x[col], 0.0);
                let mut t5 = get(i, col + 1).mul_add(x[col + 1], 0.0);
                t4 = get(i, col + 2).mul_add(x[col + 2], t4);
                t5 = get(i, col + 3).mul_add(x[col + 3], t5);
                *value = alpha.mul_add(t4 + t5, *value);
            }
            col += 4;
        }
        for _ in 0..(n & 3) {
            for (i, value) in y.iter_mut().enumerate().take(mm) {
                *value += get(i, col) * (x[col] * alpha);
            }
            col += 1;
        }
    }
    for (r, value) in y.iter_mut().enumerate().skip(mm) {
        let mut temp = 0.0;
        for (j, &xj) in x.iter().enumerate().take(n) {
            temp += get(r, j) * xj;
        }
        *value += alpha * temp;
    }
    y
}

/// `dger` (`A += alpha x y^T`) on the view at `(r0, c0)`: one axpy per
/// column with `alpha*y[j]`.
#[inline(always)]
fn dger_body(m: usize, n: usize, alpha: f64, x: &[f64], y: &[f64], a: &mut Mat, r0: usize, c0: usize) {
    let rows = a.rows;
    for j in 0..n {
        let start = (c0 + j) * rows + r0;
        daxpy_body(alpha * y[j], &x[..m], &mut a.data[start..start + m]);
    }
}

/// `drot` on one pair of elements: `x' = fma(c, x, s*y)`,
/// `y' = fma(c, y, -(s*x))`.
#[inline(always)]
pub fn drot_pair(c: f64, s: f64, x: f64, y: f64) -> (f64, f64) {
    (c.mul_add(x, s * y), c.mul_add(y, -(s * x)))
}

/// NumPy's `a.T @ b` for two C-contiguous `k x m` / `k x n` matrices
/// (OpenBLAS Haswell `dgemm`, `alpha = 1`, `beta = 0`): each element one
/// FMA accumulator over a block of K, blocks of `GEMM_Q = 128` (the last
/// two split evenly, rounded to the unroll of 4), added in order -- except
/// the kernel's 4x1 tiles (the last row when `m` is odd, against each full
/// group of four columns), which keep four accumulators per block (K mod 4
/// over the block's multiple of eight, the rest into the first) summed as
/// `(l0 + l1) + (l2 + l3)`.  A 1x1 product is `ddot`.  Row-major `m x n`.
#[inline(always)]
fn gemm_at_b_body(k: usize, m: usize, n: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
    if m == 1 && n == 1 {
        return vec![ddot_body(&a[..k], &b[..k])];
    }
    const Q: usize = 128;
    let mut blocks = Vec::new();
    let mut start = 0;
    while start < k {
        let mut length = k - start;
        if length >= 2 * Q {
            length = Q;
        } else if length > Q {
            length = (length / 2).div_ceil(4) * 4;
        }
        blocks.push((start, start + length));
        start += length;
    }
    // Every element's chain runs over the rows in order; all of them at
    // once, so the chains overlap instead of waiting on each other.
    let mut c = vec![0.0; m * n];
    let mut acc = vec![0.0_f64; m * n];
    for &(lo, hi) in &blocks {
        acc.iter_mut().for_each(|v| *v = 0.0);
        unsafe { (lanes().outer)(&a[lo * m..hi * m], &b[lo * n..hi * n], m, n, &mut acc) };
        for (total, &value) in c.iter_mut().zip(&acc) {
            *total += value;
        }
    }
    if m % 2 == 1 {
        let i = m - 1;
        for j in 0..4 * (n / 4) {
            let mut total = 0.0;
            for &(lo, hi) in &blocks {
                let unrolled = lo + ((hi - lo) & !7);
                let mut l = [0.0_f64; 4];
                for r in lo..unrolled {
                    l[(r - lo) % 4] = a[r * m + i].mul_add(b[r * n + j], l[(r - lo) % 4]);
                }
                for r in unrolled..hi {
                    l[0] = a[r * m + i].mul_add(b[r * n + j], l[0]);
                }
                total += (l[0] + l[1]) + (l[2] + l[3]);
            }
            c[i * n + j] = total;
        }
    }
    c
}


/// See `ddot_body`; dispatched to hardware FMA when the CPU has it (the
/// same correctly rounded result as the libm call, one instruction).
pub fn ddot(x: &[f64], y: &[f64]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        #[target_feature(enable = "fma")]
        unsafe fn fast(x: &[f64], y: &[f64]) -> f64 { ddot_body(x, y) }
        if std::arch::is_x86_feature_detected!("fma") {
            return unsafe { fast(x, y) };
        }
    }
    ddot_body(x, y)
}

/// See `daxpy_body`; dispatched to hardware FMA when the CPU has it (the
/// same correctly rounded result as the libm call, one instruction).
pub fn daxpy(a: f64, x: &[f64], y: &mut [f64])  {
    #[cfg(target_arch = "x86_64")]
    {
        #[target_feature(enable = "fma")]
        unsafe fn fast(a: f64, x: &[f64], y: &mut [f64])  { daxpy_body(a, x, y) }
        if std::arch::is_x86_feature_detected!("fma") {
            return unsafe { fast(a, x, y) };
        }
    }
    daxpy_body(a, x, y)
}

/// See `dgemv_t_body`; dispatched to hardware FMA when the CPU has it (the
/// same correctly rounded result as the libm call, one instruction).
pub fn dgemv_t(m: usize, n: usize, alpha: f64, a: &Mat, r0: usize, c0: usize, x: &[f64]) -> Vec<f64> {
    #[cfg(target_arch = "x86_64")]
    {
        #[target_feature(enable = "fma")]
        unsafe fn fast(m: usize, n: usize, alpha: f64, a: &Mat, r0: usize, c0: usize, x: &[f64]) -> Vec<f64> { dgemv_t_body(m, n, alpha, a, r0, c0, x) }
        if std::arch::is_x86_feature_detected!("fma") {
            return unsafe { fast(m, n, alpha, a, r0, c0, x) };
        }
    }
    dgemv_t_body(m, n, alpha, a, r0, c0, x)
}

/// See `dgemv_n_strided_body`; dispatched to hardware FMA when the CPU has it (the
/// same correctly rounded result as the libm call, one instruction).
pub fn dgemv_n_strided(m: usize, n: usize, alpha: f64, a: &Mat, r0: usize, c0: usize, x: &[f64]) -> Vec<f64> {
    #[cfg(target_arch = "x86_64")]
    {
        #[target_feature(enable = "fma")]
        unsafe fn fast(m: usize, n: usize, alpha: f64, a: &Mat, r0: usize, c0: usize, x: &[f64]) -> Vec<f64> { dgemv_n_strided_body(m, n, alpha, a, r0, c0, x) }
        if std::arch::is_x86_feature_detected!("fma") {
            return unsafe { fast(m, n, alpha, a, r0, c0, x) };
        }
    }
    dgemv_n_strided_body(m, n, alpha, a, r0, c0, x)
}

/// See `dger_body`; dispatched to hardware FMA when the CPU has it (the
/// same correctly rounded result as the libm call, one instruction).
pub fn dger(m: usize, n: usize, alpha: f64, x: &[f64], y: &[f64], a: &mut Mat, r0: usize, c0: usize)  {
    #[cfg(target_arch = "x86_64")]
    {
        #[target_feature(enable = "fma")]
        unsafe fn fast(m: usize, n: usize, alpha: f64, x: &[f64], y: &[f64], a: &mut Mat, r0: usize, c0: usize)  { dger_body(m, n, alpha, x, y, a, r0, c0) }
        if std::arch::is_x86_feature_detected!("fma") {
            return unsafe { fast(m, n, alpha, x, y, a, r0, c0) };
        }
    }
    dger_body(m, n, alpha, x, y, a, r0, c0)
}

/// See `gemm_at_b_body`; dispatched to hardware FMA when the CPU has it (the
/// same correctly rounded result as the libm call, one instruction).
pub fn gemm_at_b(k: usize, m: usize, n: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
    #[cfg(target_arch = "x86_64")]
    {
        #[target_feature(enable = "fma")]
        unsafe fn fast(k: usize, m: usize, n: usize, a: &[f64], b: &[f64]) -> Vec<f64> { gemm_at_b_body(k, m, n, a, b) }
        if std::arch::is_x86_feature_detected!("fma") {
            return unsafe { fast(k, m, n, a, b) };
        }
    }
    gemm_at_b_body(k, m, n, a, b)
}
