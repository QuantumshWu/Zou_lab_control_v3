//! The compiled kernels of `zlc_plot._raster_kernels` the projection uses,
//! operation for operation and in the same order (their `prange` is over
//! independent outputs or order-free integer partials).

use super::nd::Elem;
use crate::numeric::parallel_parts;

pub const REDUCE_MEAN: u8 = 0;
pub const REDUCE_SUM: u8 = 1;
pub const REDUCE_MIN: u8 = 2;
pub const REDUCE_MAX: u8 = 3;
pub const REDUCE_FIRST: u8 = 4;
/// `_MASKED_LEADING_MIN_SAMPLES`: below it a float64 leading reduction with
/// holes stays on numpy (`fused_masked_leading_float64` declines).
pub const MASKED_LEADING_MIN_SAMPLES: usize = 32768;
/// `aggregate_axis_codes` operation 5: centred first and second moments.
pub const CENTRED: u8 = 5;

pub struct AxisCodes<'a> {
    pub codes: &'a [Vec<i64>],
    pub sizes: &'a [usize],
    pub domain_sizes: &'a [usize],
    pub strides: &'a [usize],
}

/// A walk's per-bucket state; [`Aggregated::finish`]ed, the kernel's answer.
pub struct Aggregated { pub out: Vec<f64>, pub second: Vec<f64>, pub counts: Vec<i64>, pub presence: Vec<bool> }

impl Aggregated {
    /// Nothing walked yet: each bucket at the operation's start.
    pub fn start(bucket_count: usize, operation: u8) -> Aggregated {
        Aggregated {
            out: vec![match operation { REDUCE_MIN => f64::INFINITY, REDUCE_MAX => f64::NEG_INFINITY, _ => 0.0 }; bucket_count],
            second: vec![0.0_f64; if operation == CENTRED { bucket_count } else { 0 }],
            counts: vec![0i64; bucket_count], presence: vec![false; bucket_count],
        }
    }

    /// The walked state as the kernel answers: a mean (and centred moments)
    /// divided by its count, an empty bucket NaN.
    pub fn finish(mut self, operation: u8) -> Aggregated {
        for bucket in 0..self.counts.len() {
            let count = self.counts[bucket];
            if count > 0 {
                if operation == REDUCE_MEAN { self.out[bucket] /= count as f64; }
                if operation == CENTRED { self.out[bucket] /= count as f64; self.second[bucket] /= count as f64; }
            } else {
                self.out[bucket] = f64::NAN;
                if operation == CENTRED { self.second[bucket] = f64::NAN; }
            }
        }
        self
    }

    /// [`Aggregated::finish`]'s values and counts, the walked state kept (an operation other than `CENTRED`).
    pub fn finished(&self, operation: u8) -> (Vec<f64>, Vec<i64>) {
        let out = self.out.iter().zip(&self.counts).map(|(value, count)| match *count {
            0 => f64::NAN,
            count if operation == REDUCE_MEAN => value / count as f64,
            _ => *value,
        }).collect();
        (out, self.counts.clone())
    }
}

/// `aggregate_axis_codes` before its finish: a row-major walk folding
/// `values` into each bucket of `state` by axis-sized codes. A walk taken up
/// again with later samples is the walk of all of them.
pub fn aggregate_axis_codes<T: Elem>(state: &mut Aggregated, values: &[T], valid: Option<&[bool]>, axes: &AxisCodes<'_>,
                                     operation: u8, offsets: &[f64]) {
    let Aggregated { out, second, counts, presence } = state;
    let axis_count = axes.sizes.len();
    let mut axis_index = vec![0usize; axis_count];
    let mut axis_tick = vec![0usize; axis_count];
    for position in 0..values.len() {
        if position > 0 {
            for axis in 0..axis_count {
                axis_tick[axis] += 1;
                if axis_tick[axis] == axes.strides[axis] {
                    axis_tick[axis] = 0;
                    axis_index[axis] += 1;
                    if axis_index[axis] == axes.sizes[axis] { axis_index[axis] = 0; }
                }
            }
        }
        let mut bucket = 0usize;
        let mut admitted = true;
        for axis in 0..axis_count {
            let code = axes.codes[axis][axis_index[axis]];
            if code < 0 { admitted = false; break; }
            bucket = bucket * axes.domain_sizes[axis] + code as usize;
        }
        if !admitted { continue; }
        presence[bucket] = true;
        if let Some(valid) = valid { if !valid[position] { continue; } }
        let sample = values[position].f64();
        match operation {
            REDUCE_MIN => {
                if sample != sample { out[bucket] = sample; }
                else if out[bucket] == out[bucket] && sample <= out[bucket] { out[bucket] = sample; }
            }
            REDUCE_MAX => {
                if sample != sample { out[bucket] = sample; }
                else if out[bucket] == out[bucket] && sample >= out[bucket] { out[bucket] = sample; }
            }
            REDUCE_FIRST => { if counts[bucket] == 0 { out[bucket] = sample; } }
            CENTRED => {
                let delta = sample - offsets[bucket];
                out[bucket] += delta;
                second[bucket] += delta * delta;
            }
            _ => out[bucket] += sample,
        }
        counts[bucket] += 1;
    }
}

/// `masked_leading_float64`: reduce `(pool, outputs)` under a mask in one pass.
pub fn masked_leading_float64(values: &[f64], valid: &[bool], pool: usize, outputs: usize, reduction: u8) -> (Vec<f64>, Vec<i64>) {
    let column = |c: usize| -> (f64, i64) {
        let mut count = 0i64;
        let mut accumulator = match reduction { REDUCE_MIN => f64::INFINITY, REDUCE_MAX => f64::NEG_INFINITY, _ => 0.0 };
        let mut found = false;
        for row in 0..pool {
            let position = row * outputs + c;
            if !valid[position] { continue; }
            let sample = values[position];
            count += 1;
            match reduction {
                REDUCE_MEAN | REDUCE_SUM => accumulator += sample,
                REDUCE_MIN => if sample <= accumulator { accumulator = sample },
                REDUCE_MAX => if sample >= accumulator { accumulator = sample },
                _ => if !found { accumulator = sample; found = true; },
            }
        }
        let value = if count == 0 { f64::NAN } else if reduction == REDUCE_MEAN { accumulator / count as f64 } else { accumulator };
        (value, count)
    };
    use rayon::prelude::*;
    let parts = parallel_parts(pool * outputs);
    let pairs: Vec<(f64, i64)> = if parts > 1 { (0..outputs).into_par_iter().with_min_len(outputs.div_ceil(parts)).map(column).collect() }
                                 else { (0..outputs).map(column).collect() };
    pairs.into_iter().unzip()
}

/// `centred_moment_sums` over a `(outer, keep, inner)` view: per kept
/// position, `sum(d)` and `sum(d*d)` with `d = x - offsets[k]`.
pub fn centred_moment_sums<T: Elem>(values: &[T], offsets: &[f64], valid: Option<&[bool]>, outer: usize, keep: usize,
                                    inner: usize) -> (Vec<f64>, Vec<f64>) {
    let one = |k: usize| -> (f64, f64) {
        let (mut first, mut second) = (0.0_f64, 0.0_f64);
        let offset = offsets[k];
        for o in 0..outer {
            for i in 0..inner {
                let position = (o * keep + k) * inner + i;
                if let Some(valid) = valid { if !valid[position] { continue; } }
                let delta = values[position].f64() - offset;
                first += delta;
                second += delta * delta;
            }
        }
        (first, second)
    };
    use rayon::prelude::*;
    let parts = parallel_parts(outer * keep * inner);
    let pairs: Vec<(f64, f64)> = if parts > 1 { (0..keep).into_par_iter().with_min_len(keep.div_ceil(parts)).map(one).collect() }
                                 else { (0..keep).map(one).collect() };
    pairs.into_iter().unzip()
}

/// `_combine_moment_summary`: Chan's combination of `(n, mean, M2)` summaries.
#[inline]
fn combine(left: (i64, f64, f64, f64), right: (i64, f64, f64, f64)) -> (i64, f64, f64, f64) {
    let (left_n, left_mean, left_m2, _) = left;
    let (right_n, right_mean, right_m2, _) = right;
    if left_n == 0 { return right; }
    if right_n == 0 { return left; }
    let count = left_n + right_n;
    let delta = right_mean - left_mean;
    let mean = left_mean + delta * right_n as f64 / count as f64;
    let m2 = left_m2 + right_m2 + delta * delta * left_n as f64 * right_n as f64 / count as f64;
    (count, mean, m2, f64::NAN)
}

/// `_trailing_moment_windows`, in place.
pub fn trailing_moment_windows(counts: &mut [i64], means: &mut [f64], m2s: &mut [f64], singles: &mut [f64], span: usize) {
    let total = counts.len();
    let width = span.min(total);
    if width == 0 { return; }
    type Summary = (i64, f64, f64, f64);
    let mut state: Vec<Summary> = (0..total).map(|i| (counts[i], means[i], m2s[i], singles[i])).collect();
    let mut suffix: Vec<Summary> = vec![(0, 0.0, 0.0, 0.0); total];
    // Each span-sized block: its suffix scan, then its prefix scan in place.
    let block = |state: &mut [Summary], suffix: &mut [Summary]| {
        let mut running = (0i64, 0.0, 0.0, f64::NAN);
        for index in (0..state.len()).rev() {
            running = combine(state[index], running);
            suffix[index] = running;
        }
        let mut running = (0i64, 0.0, 0.0, f64::NAN);
        for slot in state.iter_mut() {
            running = combine(running, *slot);
            *slot = running;
        }
    };
    use rayon::prelude::*;
    // A window step (a division chain) costs about eight summed elements.
    let parallel = parallel_parts(total * 8) > 1 && total / width >= 2;
    if parallel {
        state.par_chunks_mut(width).zip(suffix.par_chunks_mut(width)).for_each(|(s, x)| block(s, x));
    } else {
        // Each block's two scans are one dependent chain of divisions; eight
        // complete blocks advance in lock-step so their chains overlap.
        const LANES: usize = 8;
        let full = total / width;
        let mut first_block = 0;
        while first_block + LANES <= full {
            let starts: [usize; LANES] = std::array::from_fn(|lane| (first_block + lane) * width);
            let mut running = [(0i64, 0.0, 0.0, f64::NAN); LANES];
            for step in (0..width).rev() {
                for lane in 0..LANES {
                    let index = starts[lane] + step;
                    running[lane] = combine(state[index], running[lane]);
                    suffix[index] = running[lane];
                }
            }
            let mut running = [(0i64, 0.0, 0.0, f64::NAN); LANES];
            for step in 0..width {
                for lane in 0..LANES {
                    let index = starts[lane] + step;
                    running[lane] = combine(running[lane], state[index]);
                    state[index] = running[lane];
                }
            }
            first_block += LANES;
        }
        let rest = first_block * width;
        state[rest..].chunks_mut(width).zip(suffix[rest..].chunks_mut(width)).for_each(|(s, x)| block(s, x));
    }
    // Every later block position combines the preceding block's suffix with its prefix.
    let finish = |(b, chunk): (usize, &mut [Summary])| {
        if b == 0 { return; }
        let block_start = b * width;
        let stop = (width - 1).min(chunk.len());
        for (offset, slot) in chunk[..stop].iter_mut().enumerate() {
            let left = block_start + offset + 1 - width;
            *slot = combine(suffix[left], *slot);
        }
    };
    if parallel { state.par_chunks_mut(width).enumerate().for_each(finish); } else { state.chunks_mut(width).enumerate().for_each(finish); }
    for (i, (c, m, q, s)) in state.into_iter().enumerate() { counts[i] = c; means[i] = m; m2s[i] = q; singles[i] = s; }
}

/// `uniform_histogram`: counts per (facet, bin), numpy's inclusive last edge
/// and two rounding corrections against the actual edges.
pub fn uniform_histogram<T: Elem>(values: &[T], valid: Option<&[bool]>, facet_codes: &[i64], facet_stride: usize,
                                  edges: &[f64], facets: usize) -> Vec<i64> {
    let bins = edges.len() - 1;
    let first = edges[0];
    let last = edges[bins];
    let denominator = last - first;
    let axis_size = facet_codes.len();
    let count_range = |start: usize, stop: usize| -> Vec<i64> {
        let mut partial = vec![0i64; facets * bins];
        for p in start..stop {
            if let Some(valid) = valid { if !valid[p] { continue; } }
            let mut facet = 0i64;
            if axis_size > 0 { facet = facet_codes[(p / facet_stride) % axis_size]; }
            if facet < 0 { continue; }
            let sample = values[p].f64();
            if !(sample >= first && sample <= last) { continue; }
            let mut index = (((sample - first) / denominator) * bins as f64) as i64;
            if index == bins as i64 { index -= 1; }
            if sample < edges[index as usize] { index -= 1; }
            if sample >= edges[(index + 1) as usize] && index != bins as i64 - 1 { index += 1; }
            partial[facet as usize * bins + index as usize] += 1;
        }
        partial
    };
    let parts = parallel_parts(values.len());
    if parts == 1 { return count_range(0, values.len()); }
    use rayon::prelude::*;
    let chunk = values.len().div_ceil(parts);
    let partials: Vec<Vec<i64>> = (0..parts).into_par_iter()
        .map(|t| count_range((t * chunk).min(values.len()), ((t + 1) * chunk).min(values.len()))).collect();
    let mut total = vec![0i64; facets * bins];
    for partial in partials { for (slot, value) in total.iter_mut().zip(partial) { *slot += value; } }
    total
}

/// `finite_extrema`: `(finite count, min, max, all whole)` of a masked float pool.
pub fn finite_extrema<T: Elem>(values: &[T], valid: Option<&[bool]>) -> (usize, f64, f64, bool) {
    let (mut count, mut low, mut high, mut whole) = (0usize, f64::INFINITY, f64::NEG_INFINITY, true);
    for (p, value) in values.iter().enumerate() {
        if let Some(valid) = valid { if !valid[p] { continue; } }
        if !value.finite() { continue; }
        let sample = value.f64();
        count += 1;
        if sample < low { low = sample; }
        if sample > high { high = sample; }
        if sample != sample.floor() { whole = false; }
    }
    (count, low, high, whole)
}

/// Writes to provably disjoint output buckets from several threads.
struct Shared<T>(*mut T);
unsafe impl<T> Send for Shared<T> {}
unsafe impl<T> Sync for Shared<T> {}
impl<T> Shared<T> {
    /// SAFETY: callers write each index from at most one thread.
    #[inline(always)]
    #[allow(clippy::mut_from_ref)]
    unsafe fn at(&self, index: usize) -> &mut T { unsafe { &mut *self.0.add(index) } }
}

/// [`aggregate_axis_codes`] split over threads without changing any bucket's
/// order: the slices of one tensor dimension are grouped by the codes they
/// carry on that dimension, so different groups reach disjoint buckets, and
/// each group walks its own positions in the original row-major order, into
/// `state`. False, `state` untouched, when no dimension separates the work
/// (the caller walks serially).
#[allow(clippy::too_many_arguments)]
pub fn aggregate_axis_codes_parallel<T: Elem>(state: &mut Aggregated, values: &[T], valid: Option<&[bool]>, shape: &[usize], codes: &[Vec<i64>],
                                              dimensions: &[usize], domain_sizes: &[usize], operation: u8, offsets: &[f64]) -> bool {
    let ndim = shape.len();
    if values.len() < 1 << 16 || codes.iter().zip(dimensions).any(|(c, d)| c.len() != shape[*d]) { return false; }
    // Each axis contributes code * (product of the later domain sizes) to a bucket number.
    let weights: Vec<i64> = (0..codes.len()).map(|a| domain_sizes[a + 1..].iter().product::<usize>() as i64).collect();
    // The splitting dimension: the code-carrying one with the most distinct code tuples.
    let mut best: Option<(usize, Vec<Vec<usize>>)> = None;
    for dimension in 0..ndim {
        let axes: Vec<usize> = (0..codes.len()).filter(|a| dimensions[*a] == dimension).collect();
        if axes.is_empty() { continue; }
        let mut groups: std::collections::BTreeMap<Vec<i64>, Vec<usize>> = Default::default();
        for index in 0..shape[dimension] {
            groups.entry(axes.iter().map(|a| codes[*a][index]).collect()).or_default().push(index);
        }
        let groups: Vec<Vec<usize>> = groups.into_values().collect();
        // The outermost dimension that feeds every thread wins: its slices are
        // the largest contiguous blocks. Otherwise the most groups.
        let enough = parallel_parts(values.len()).max(2);
        let better = match best.as_ref() { None => true, Some((_, b)) => b.len() < enough && groups.len() > b.len() };
        if groups.len() >= 2 && better { best = Some((dimension, groups)); }
    }
    let Some((split, groups)) = best else { return false };
    let contribution = |dimension: usize, index: usize| -> i64 {
        let mut total = 0i64;
        for (a, d) in dimensions.iter().enumerate() {
            if *d == dimension {
                let code = codes[a][index];
                if code < 0 { return -1; }
                total += code * weights[a];
            }
        }
        total
    };
    let walk = |dims: &[usize], first: usize| -> Vec<i64> {
        let total: usize = dims.iter().product();
        let mut index = vec![0usize; dims.len()];
        (0..total).map(|_| {
            let mut sum = 0i64;
            for (k, i) in index.iter().enumerate() {
                let c = contribution(first + k, *i);
                if c < 0 { sum = -1; break; }
                sum += c;
            }
            for k in (0..dims.len()).rev() { index[k] += 1; if index[k] < dims[k] { break; } index[k] = 0; }
            sum
        }).collect()
    };
    let block: usize = shape[split + 1..].iter().product();
    // The slice's bucket offsets, factored: one per row of the dimensions after
    // the split but the last, and one per index of the last (both small).
    let tail = &shape[split + 1..];
    let (rows, last, width) = if tail.is_empty() { (vec![0i64], vec![0i64], 1usize) } else {
        let n = tail.len();
        (walk(&tail[..n - 1], split + 1), (0..tail[n - 1]).map(|i| contribution(split + n, i)).collect::<Vec<i64>>(), tail[n - 1])
    };
    // When the innermost dimension names distinct buckets at every index, its
    // index ranges are disjoint bucket sets too: split them across threads as
    // well, keeping each bucket's own visiting order.
    let injective = {
        let mut seen: Vec<i64> = last.iter().copied().filter(|v| *v >= 0).collect();
        let n = seen.len();
        seen.sort_unstable();
        seen.dedup();
        seen.len() == n
    };
    let threads = parallel_parts(values.len());
    let pieces = if injective && threads > 1 && groups.len() < 2 * threads { (2 * threads).div_ceil(groups.len()).min(width.div_ceil(32)).max(1) } else { 1 };
    let step = width.div_ceil(pieces);
    let tasks: Vec<(usize, usize, usize)> = (0..groups.len()).flat_map(|g| (0..pieces).map(move |k| (g, k * step, ((k + 1) * step).min(width))))
        .filter(|(_, lo, hi)| lo < hi).collect();
    let inner = (rows, last, width);
    let outer_offsets = walk(&shape[..split], 0);
    {
        let Aggregated { out, second, counts, presence } = state;
        let (o, s, c, p) = (Shared(out.as_mut_ptr()), Shared(second.as_mut_ptr()), Shared(counts.as_mut_ptr()), Shared(presence.as_mut_ptr()));
        use rayon::prelude::*;
        let task = |&(g, lo, hi): &(usize, usize, usize)| {
            let group = &groups[g];
            for (o_index, outer_offset) in outer_offsets.iter().enumerate() {
                if *outer_offset < 0 { continue; }
                for &i in group {
                    let slice = contribution(split, i);
                    if slice < 0 { continue; }
                    let base = (o_index * shape[split] + i) * block;
                    let lanes = Lanes { out: &o, second: &s, counts: &c, presence: &p };
                    let base_bucket = outer_offset + slice;
                    match (operation, valid) {
                        (REDUCE_MIN, None) => slice_pass::<T, REDUCE_MIN, false>(values, None, &inner, lo, hi, base, base_bucket, offsets, &lanes),
                        (REDUCE_MIN, Some(v)) => slice_pass::<T, REDUCE_MIN, true>(values, Some(v), &inner, lo, hi, base, base_bucket, offsets, &lanes),
                        (REDUCE_MAX, None) => slice_pass::<T, REDUCE_MAX, false>(values, None, &inner, lo, hi, base, base_bucket, offsets, &lanes),
                        (REDUCE_MAX, Some(v)) => slice_pass::<T, REDUCE_MAX, true>(values, Some(v), &inner, lo, hi, base, base_bucket, offsets, &lanes),
                        (REDUCE_FIRST, None) => slice_pass::<T, REDUCE_FIRST, false>(values, None, &inner, lo, hi, base, base_bucket, offsets, &lanes),
                        (REDUCE_FIRST, Some(v)) => slice_pass::<T, REDUCE_FIRST, true>(values, Some(v), &inner, lo, hi, base, base_bucket, offsets, &lanes),
                        (CENTRED, None) => slice_pass::<T, CENTRED, false>(values, None, &inner, lo, hi, base, base_bucket, offsets, &lanes),
                        (CENTRED, Some(v)) => slice_pass::<T, CENTRED, true>(values, Some(v), &inner, lo, hi, base, base_bucket, offsets, &lanes),
                        (_, None) => slice_pass::<T, REDUCE_SUM, false>(values, None, &inner, lo, hi, base, base_bucket, offsets, &lanes),
                        (_, Some(v)) => slice_pass::<T, REDUCE_SUM, true>(values, Some(v), &inner, lo, hi, base, base_bucket, offsets, &lanes),
                    }
                }
            }
        };
        if threads > 1 { tasks.par_iter().with_min_len(tasks.len().div_ceil(2 * threads)).for_each(task); } else { tasks.iter().for_each(task); }
    }
    true
}

struct Lanes<'a> { out: &'a Shared<f64>, second: &'a Shared<f64>, counts: &'a Shared<i64>, presence: &'a Shared<bool> }

/// One slice of the parallel walk, specialized by operation and masking.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn slice_pass<T: Elem, const OP: u8, const MASKED: bool>(values: &[T], valid: Option<&[bool]>, inner: &(Vec<i64>, Vec<i64>, usize),
                                                         lo: usize, hi: usize, base: usize, base_bucket: i64, offsets: &[f64], lanes: &Lanes<'_>) {
    let (rows, last, width) = inner;
    // The innermost dimension usually names consecutive buckets (bucket = first
    // + x): each row is then one contiguous run of samples into one contiguous
    // run of buckets, which the compiler vectorizes. Same per-bucket order.
    let contiguous = OP != REDUCE_FIRST && hi > lo && last[lo] >= 0 && (lo..hi).all(|x| last[x] == last[lo] + (x - lo) as i64);
    if contiguous {
        let n = hi - lo;
        for (r, row) in rows.iter().enumerate() {
            if *row < 0 { continue; }
            let first = (base_bucket + row + last[lo]) as usize;
            let start = base + r * width + lo;
            let samples = &values[start..start + n];
            // SAFETY: these buckets carry this group's split-dimension codes, which no other group has.
            unsafe {
                let out = std::slice::from_raw_parts_mut(lanes.out.0.add(first), n);
                let counts = std::slice::from_raw_parts_mut(lanes.counts.0.add(first), n);
                std::slice::from_raw_parts_mut(lanes.presence.0.add(first), n).fill(true);
                let marks: &[bool] = if MASKED { &valid.unwrap_unchecked()[start..start + n] } else { &[] };
                match OP {
                    CENTRED => {
                        let second = std::slice::from_raw_parts_mut(lanes.second.0.add(first), n);
                        let centres = &offsets[first..first + n];
                        for j in 0..n {
                            if MASKED && !marks[j] { continue; }
                            let delta = samples[j].f64() - centres[j];
                            out[j] += delta;
                            second[j] += delta * delta;
                            counts[j] += 1;
                        }
                    }
                    REDUCE_MIN | REDUCE_MAX => for j in 0..n {
                        if MASKED && !marks[j] { continue; }
                        let sample = samples[j].f64();
                        let slot = &mut out[j];
                        if OP == REDUCE_MIN {
                            if sample != sample { *slot = sample } else if *slot == *slot && sample <= *slot { *slot = sample }
                        } else if sample != sample { *slot = sample } else if *slot == *slot && sample >= *slot { *slot = sample }
                        counts[j] += 1;
                    },
                    _ => for j in 0..n {
                        if MASKED && !marks[j] { continue; }
                        out[j] += samples[j].f64();
                        counts[j] += 1;
                    },
                }
            }
        }
        return;
    }
    for (r, row) in rows.iter().enumerate() {
        if *row < 0 { continue; }
        for (x, column) in last.iter().enumerate().take(hi).skip(lo) {
            if *column < 0 { continue; }
            let q = r * width + x;
            let bucket = (base_bucket + row + column) as usize;
        // SAFETY: the bucket carries this group's split-dimension codes, which no other group has.
        unsafe {
            *lanes.presence.at(bucket) = true;
            let position = base + q;
            if MASKED && !valid.unwrap_unchecked()[position] { continue; }
            let sample = values[position].f64();
            let slot = lanes.out.at(bucket);
            match OP {
                REDUCE_MIN => if sample != sample { *slot = sample } else if *slot == *slot && sample <= *slot { *slot = sample },
                REDUCE_MAX => if sample != sample { *slot = sample } else if *slot == *slot && sample >= *slot { *slot = sample },
                REDUCE_FIRST => if *lanes.counts.at(bucket) == 0 { *slot = sample },
                CENTRED => { let delta = sample - offsets[bucket]; *slot += delta; *lanes.second.at(bucket) += delta * delta; }
                _ => *slot += sample,
            }
            *lanes.counts.at(bucket) += 1;
            }
        }
    }
}
