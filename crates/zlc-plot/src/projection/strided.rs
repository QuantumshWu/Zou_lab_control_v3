//! The dense leading reduction without moving the tensor first.
//!
//! `_dense_tensor_projection` lays the tensor out as `(pool, *kept)` with
//! `np.moveaxis(...)` + a contiguous copy, then reduces the leading axis.
//! With at least two kept elements numpy's reduction of that layout is an
//! elementwise accumulation over the pool rows in order, so reading each
//! pool row straight out of the original tensor gives the same bits per
//! output without writing the moved copy. Runs of the innermost kept axis
//! stay contiguous, and outputs are split across threads.

use rayon::prelude::*;

use super::nd::{Elem, product};
use crate::numeric::parallel_parts;

/// `(pool, kept)` addressing of a C-contiguous tensor.
pub struct Moved {
    /// Source offset of each pool row, in pool (C) order.
    pub pool: Vec<usize>,
    /// `(output start, source offset)` of each contiguous kept run.
    pub runs: Vec<(usize, usize)>,
    pub run: usize,
    pub kept: usize,
}

impl Moved {
    pub fn new(shape: &[usize], dims: &[usize]) -> Moved {
        let ndim = shape.len();
        let mut strides = vec![1usize; ndim];
        for axis in (0..ndim.saturating_sub(1)).rev() { strides[axis] = strides[axis + 1] * shape[axis + 1]; }
        let offsets = |axes: &[usize]| -> Vec<usize> {
            let sizes: Vec<usize> = axes.iter().map(|a| shape[*a]).collect();
            let total = product(&sizes);
            let mut out = Vec::with_capacity(total);
            let mut index = vec![0usize; axes.len()];
            let mut flat = 0usize;
            for _ in 0..total {
                out.push(flat);
                for position in (0..axes.len()).rev() {
                    index[position] += 1;
                    flat += strides[axes[position]];
                    if index[position] < sizes[position] { break; }
                    flat -= strides[axes[position]] * sizes[position];
                    index[position] = 0;
                }
            }
            out
        };
        let pool_axes: Vec<usize> = (0..ndim).filter(|a| !dims.contains(a)).collect();
        let pool = offsets(&pool_axes);
        let kept = product(&dims.iter().map(|a| shape[*a]).collect::<Vec<_>>());
        let last = dims.last().copied();
        let (run, run_axes): (usize, &[usize]) = match last {
            Some(axis) if strides[axis] == 1 => (shape[axis], &dims[..dims.len() - 1]),
            _ => (1, dims),
        };
        let runs = offsets(run_axes).into_iter().enumerate().map(|(i, source)| (i * run, source)).collect();
        Moved { pool, runs, run, kept }
    }

    /// `body(output_slice, run_offsets)` over disjoint output ranges, in parallel when large.
    fn split<A: Send>(&self, out: &mut [A], work: usize, body: impl Fn(&mut [A], &[(usize, usize)]) + Sync) {
        let parts = parallel_parts(work);
        if parts == 1 { return body(out, &self.runs); }
        let runs_per_chunk = self.runs.len().div_ceil(parts * 2).max(1);
        out.par_chunks_mut(runs_per_chunk * self.run).zip(self.runs.par_chunks(runs_per_chunk)).for_each(|(part, runs)| body(part, runs));
    }

    /// `out[k] = 0 + sum over pool rows (in order) of x`, counting the valid ones.
    pub fn sum<T: Elem>(&self, values: &[T], mask: Option<&[bool]>) -> (Vec<f64>, Vec<i64>) {
        let mut out = vec![0.0_f64; self.kept];
        let mut counts = vec![0i64; if mask.is_some() { self.kept } else { 0 }];
        let run = self.run;
        let work = self.pool.len() * self.kept;
        match mask {
            None => self.split(&mut out, work, |part, runs| {
                for &p in &self.pool {
                    for (r, (_, source)) in runs.iter().enumerate() {
                        let base = p + source;
                        for (slot, value) in part[r * run..(r + 1) * run].iter_mut().zip(&values[base..base + run]) { *slot += value.f64(); }
                    }
                }
            }),
            Some(mask) => self.split2(&mut out, &mut counts, work, |part, tally, runs| {
                for &p in &self.pool {
                    for (r, (_, source)) in runs.iter().enumerate() {
                        let base = p + source;
                        for j in 0..run {
                            if mask[base + j] { part[r * run + j] += values[base + j].f64(); tally[r * run + j] += 1; }
                        }
                    }
                }
            }),
        }
        (out, counts)
    }

    /// Integer accumulation (numpy's int64/uint64 default), wrapping.
    pub fn int_sum<T: Elem>(&self, values: &[T], mask: Option<&[bool]>) -> Vec<i128> {
        let mut out = vec![0i128; self.kept];
        let run = self.run;
        self.split(&mut out, self.pool.len() * self.kept, |part, runs| {
            for &p in &self.pool {
                for (r, (_, source)) in runs.iter().enumerate() {
                    let base = p + source;
                    for j in 0..run {
                        if mask.is_none_or(|m| m[base + j]) { let slot = &mut part[r * run + j]; *slot = slot.wrapping_add(values[base + j].int()); }
                    }
                }
            }
        });
        out
    }

    /// Masked/unmasked extremes as float64, folding each kept element's
    /// pool rows in order with `step(current, sample)` from ±inf.
    pub fn extreme<T: Elem>(&self, values: &[T], mask: Option<&[bool]>, minimum: bool, step: impl Fn(f64, f64) -> f64 + Sync) -> Vec<f64> {
        let mut out = vec![if minimum { f64::INFINITY } else { f64::NEG_INFINITY }; self.kept];
        let run = self.run;
        self.split(&mut out, self.pool.len() * self.kept, |part, runs| {
            for &p in &self.pool {
                for (r, (_, source)) in runs.iter().enumerate() {
                    let base = p + source;
                    for j in 0..run {
                        if mask.is_some_and(|m| !m[base + j]) { continue; }
                        let slot = &mut part[r * run + j];
                        *slot = step(*slot, values[base + j].f64());
                    }
                }
            }
        });
        out
    }

    /// The first valid pool row's flat source position of each kept element.
    pub fn first(&self, mask: &[bool]) -> Vec<usize> {
        let mut out = vec![usize::MAX; self.kept];
        let run = self.run;
        for &p in &self.pool {
            for (r, (_, source)) in self.runs.iter().enumerate() {
                for j in 0..run {
                    let slot = &mut out[r * run + j];
                    if *slot == usize::MAX && mask[p + source + j] { *slot = p + source + j; }
                }
            }
        }
        let fallback = self.pool[0];
        for (k, slot) in out.iter_mut().enumerate() {
            if *slot == usize::MAX { let (r, j) = (k / run, k % run); *slot = fallback + self.runs[r].1 + j; }
        }
        out
    }

    /// Two output planes split over the same disjoint ranges.
    fn split2<A: Send, B: Send>(&self, first: &mut [A], second: &mut [B], work: usize, body: impl Fn(&mut [A], &mut [B], &[(usize, usize)]) + Sync) {
        let parts = parallel_parts(work);
        if parts == 1 { return body(first, second, &self.runs); }
        let runs_per_chunk = self.runs.len().div_ceil(parts * 2).max(1);
        let width = runs_per_chunk * self.run;
        first.par_chunks_mut(width).zip(second.par_chunks_mut(width)).zip(self.runs.par_chunks(runs_per_chunk))
            .for_each(|((a, b), runs)| body(a, b, runs));
    }

    /// `centred_moment_sums` with `inner = 1`: per kept element, `sum(d)` and `sum(d*d)`.
    pub fn centred<T: Elem>(&self, values: &[T], offsets: &[f64], mask: Option<&[bool]>) -> (Vec<f64>, Vec<f64>) {
        let mut first = vec![0.0_f64; self.kept];
        let mut second = vec![0.0_f64; self.kept];
        let run = self.run;
        self.split2(&mut first, &mut second, self.pool.len() * self.kept, |a, b, runs| {
            for &p in &self.pool {
                for (r, (start, source)) in runs.iter().enumerate() {
                    let base = p + source;
                    let (a, b) = (&mut a[r * run..(r + 1) * run], &mut b[r * run..(r + 1) * run]);
                    let (row, centres) = (&values[base..base + run], &offsets[*start..*start + run]);
                    match mask {
                        None => for j in 0..run {
                            let delta = row[j].f64() - centres[j];
                            a[j] += delta;
                            b[j] += delta * delta;
                        },
                        Some(mask) => for j in 0..run {
                            if !mask[base + j] { continue; }
                            let delta = row[j].f64() - centres[j];
                            a[j] += delta;
                            b[j] += delta * delta;
                        },
                    }
                }
            }
        });
        (first, second)
    }

    /// The kept plane of pool row 0 (`moved[0]`) as source positions.
    pub fn row_positions(&self, row: usize) -> Vec<usize> {
        let p = self.pool[row];
        let mut out = Vec::with_capacity(self.kept);
        for (_, source) in &self.runs { out.extend(p + source..p + source + self.run); }
        out
    }
}
