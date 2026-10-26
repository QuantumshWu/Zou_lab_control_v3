//! Column kernel of the sparse Hilbert stages (AVX2 + FMA, with a scalar
//! fallback).
//!
//! A right product Y = sum_e X_e A_e over the edges e of one target needs,
//! for every output column j, the sum over the nonzero entries A_e[k, j] of
//! A_e[k, j] times field column k of source X_e. Every edge carries the
//! nonzeros of its coefficient by column (starts, rows, values); the kernel
//! walks them in register tiles of 8 field rows, so only nonzero
//! coefficient entries are multiplied, the field is read in place (no
//! packing) and every output entry is written once.
use crate::C;

/// One edge of a target, as addresses: its source block (first field
/// column) and its coefficient's column lists, column j holding the entries
/// start[j]..start[j+1] of `row` and `value`.
#[derive(Clone, Copy)]
pub(super) struct Edge {
    pub block: usize,
    pub start: usize,
    pub row: usize,
    pub value: usize,
}

impl Edge {
    pub fn new(block: usize, start: &[usize], row: &[usize], value: &[C]) -> Self {
        Self { block, start: start.as_ptr() as usize, row: row.as_ptr() as usize, value: value.as_ptr() as usize }
    }

    /// Entries of column j.
    #[inline]
    unsafe fn column(self, j: usize) -> (usize, usize) {
        let start = self.start as *const usize;
        (*start.add(j), *start.add(j + 1))
    }

    /// Row r0 of the field column of entry `index`, and the entry's value.
    #[inline]
    unsafe fn entry(self, index: usize, stride: usize, r0: usize) -> (*const C, C) {
        let row = *(self.row as *const usize).add(index);
        ((self.block as *const C).add(row * stride + r0), *(self.value as *const C).add(index))
    }
}

/// Where the column sums land.
#[derive(Clone, Copy)]
pub(super) enum Epilogue {
    /// out[r + j * stride] += alpha * sum.
    Add { stride: usize },
    /// The rows are r = i + n*b of a mode block with `rhs` right-hand sides;
    /// column j of the sum is added conjugated and transposed:
    /// out[j + n*(b + rhs*i)] += alpha * conj(sum).
    ConjugateTranspose { n: usize, rhs: usize },
}

impl Epilogue {
    #[inline]
    unsafe fn put(self, out: *mut C, r: usize, j: usize, alpha: f64, re: f64, im: f64) {
        match self {
            Epilogue::Add { stride } => {
                let at = out.add(r + j * stride);
                (*at).re += alpha * re;
                (*at).im += alpha * im;
            }
            Epilogue::ConjugateTranspose { n, rhs } => {
                let at = out.add(j + n * (r / n + rhs * (r % n)));
                (*at).re += alpha * re;
                (*at).im -= alpha * im;
            }
        }
    }
}

/// True when the vector kernel can run on this processor.
pub(super) fn available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// For every column j < columns: the sum over all edges and all entries of
/// their column j of value * (field column `row` of the edge's block)[r]
/// for r < rows, written by `epilogue` with `alpha`. Field columns are
/// `stride` entries apart.
///
/// Safety: every edge's block must hold the columns its rows name, each
/// with at least `rows` entries, its lists must hold `columns` + 1 starts,
/// and `out` must hold every position the epilogue addresses for r < rows
/// and j < columns.
pub(super) unsafe fn columns(edges: &[Edge], columns: usize, stride: usize, rows: usize, out: *mut C, epilogue: Epilogue, alpha: f64) {
    #[cfg(target_arch = "x86_64")]
    if available() {
        return x86::columns(edges, columns, stride, rows, out, epilogue, alpha);
    }
    columns_scalar(edges, columns, stride, 0, rows, out, epilogue, alpha)
}

/// Rows first..rows of `columns`, one row at a time.
#[allow(clippy::too_many_arguments)]
unsafe fn columns_scalar(
    edges: &[Edge],
    columns: usize,
    stride: usize,
    first: usize,
    rows: usize,
    out: *mut C,
    epilogue: Epilogue,
    alpha: f64,
) {
    for j in 0..columns {
        for r in first..rows {
            let (mut sr, mut si, mut count) = (0., 0., 0);
            for &edge in edges {
                let (begin, end) = edge.column(j);
                for index in begin..end {
                    let (x, v) = edge.entry(index, stride, r);
                    sr += (*x).re * v.re - (*x).im * v.im;
                    si += (*x).re * v.im + (*x).im * v.re;
                }
                count += end - begin;
            }
            if count > 0 {
                epilogue.put(out, r, j, alpha, sr, si);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::{Edge, Epilogue};
    use crate::C;
    use std::arch::x86_64::*;

    /// 8-row tiles (four vectors of two complex entries), then 2-row tiles,
    /// then a scalar last row.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn columns(edges: &[Edge], columns: usize, stride: usize, rows: usize, out: *mut C, epilogue: Epilogue, alpha: f64) {
        let mut r0 = 0;
        while r0 + 8 <= rows {
            tile::<4>(edges, columns, stride, r0, out, epilogue, alpha);
            r0 += 8;
        }
        while r0 + 2 <= rows {
            tile::<1>(edges, columns, stride, r0, out, epilogue, alpha);
            r0 += 2;
        }
        if r0 < rows {
            super::columns_scalar(edges, columns, stride, r0, rows, out, epilogue, alpha);
        }
    }

    /// Rows r0..r0 + 2V of every column. The field rows times Re(value) and
    /// times Im(value) accumulate separately and combine once per column:
    /// (a + ib)(c + id) = (ac - bd) + i(bc + ad).
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn tile<const V: usize>(
        edges: &[Edge],
        columns: usize,
        stride: usize,
        r0: usize,
        out: *mut C,
        epilogue: Epilogue,
        alpha: f64,
    ) {
        for j in 0..columns {
            let mut re = [_mm256_setzero_pd(); V];
            let mut im = [_mm256_setzero_pd(); V];
            let mut count = 0;
            for &edge in edges {
                let (begin, end) = edge.column(j);
                for index in begin..end {
                    let (x, value) = edge.entry(index, stride, r0);
                    let (x, vr, vi) = (x as *const f64, _mm256_set1_pd(value.re), _mm256_set1_pd(value.im));
                    for v in 0..V {
                        let x = _mm256_loadu_pd(x.add(4 * v));
                        re[v] = _mm256_fmadd_pd(x, vr, re[v]);
                        im[v] = _mm256_fmadd_pd(x, vi, im[v]);
                    }
                }
                count += end - begin;
            }
            if count == 0 {
                continue;
            }
            let mut sums = [_mm256_setzero_pd(); V];
            for v in 0..V {
                sums[v] = _mm256_addsub_pd(re[v], _mm256_permute_pd(im[v], 0b0101));
            }
            match epilogue {
                Epilogue::Add { stride } => {
                    let scale = _mm256_set1_pd(alpha);
                    let y = out.add(r0 + j * stride) as *mut f64;
                    for (v, sum) in sums.iter().enumerate() {
                        let at = y.add(4 * v);
                        _mm256_storeu_pd(at, _mm256_fmadd_pd(*sum, scale, _mm256_loadu_pd(at)));
                    }
                }
                Epilogue::ConjugateTranspose { .. } => {
                    let mut values = [0f64; 16];
                    for (v, sum) in sums.iter().enumerate() {
                        _mm256_storeu_pd(values.as_mut_ptr().add(4 * v), *sum);
                    }
                    for q in 0..2 * V {
                        epilogue.put(out, r0 + q, j, alpha, values[2 * q], values[2 * q + 1]);
                    }
                }
            }
        }
    }
}
