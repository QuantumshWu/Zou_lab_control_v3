//! Histograms: the window frequency table of an integer history, the
//! planned (grouped / reduced / windowed) distribution, edges and their
//! retention between revisions (`FitProjection._histogram_bins`).

use std::collections::HashMap;
use std::sync::Arc;

use super::curve::CellSpec;
use super::kernels;
use super::model::{Block, Segment};
use super::nd::{Array, Bools, DType, Mask, product};
use super::output::{AxisValue, FacetCell, FacetData, HistogramData, QuantityArray};
use super::plan::{HistogramProjection, Projection, ProjectionPlan, relim_mode};
use super::reduce::{axis_aggregate, packed_codes};
use super::view::{DataView, Packed, acquired_repeat_count, expand_mask, pack};
use super::{ProjectionError, Result};
use crate::kinds::{AxisRef, Reduction, RelimMode};
use crate::with_dtype;

const FREQUENCY_LEVEL_LIMIT: i128 = 1 << 18;

/// `_HistogramPlan`: reduced samples and coordinate-owned distribution codes.
pub struct HistogramPlan {
    pub values: Array,
    pub valid: Mask,
    pub group_codes: Vec<i64>,
    pub code_axis: usize,
    pub group_keys: Vec<Vec<AxisValue>>,
    /// How many samples the whole pool holds: `values` may be only the rows
    /// a window meets.
    pub operands: usize,
}

/// What a histogram plan packs: the rows it may count, and the whole pool's sample count.
struct Scratch { packed: Arc<Packed>, operands: usize }

/// A still-filling run's counted segments, by the address of their immutable
/// planes: each one's extremes and its counts under `edges`.
pub struct HistogramCarry {
    /// Each group's carrier dimension, codes and value count.
    query: Vec<(usize, Arc<[i64]>, usize)>,
    /// Keeps the identified planes alive.
    #[allow(dead_code)]
    segments: super::model::Segments,
    facts: HashMap<(usize, usize, [usize; 2]), (Extremes, Vec<i64>)>,
    edges: Vec<f64>,
}

/// What `_histogram_bins` reads of a pool's valid samples: an integer pool's
/// least and greatest, or a float pool's finite count, least and greatest
/// (the first met of equal ones) and whether all are whole.
#[derive(Clone, Copy)]
pub struct Extremes { integer: bool, count: usize, low: f64, high: f64, whole: bool }

impl Extremes {
    /// No sample yet.
    pub fn none() -> Extremes { Extremes { integer: true, count: 0, low: f64::INFINITY, high: f64::NEG_INFINITY, whole: true } }

    /// The scan of `values` under `valid`, as `_histogram_bins` makes it.
    pub fn of(values: &Array, valid: &Mask) -> Extremes {
        let n = values.len();
        let marks = if valid.all_true() { None } else { Some(valid.plane(n)) };
        if "biu".contains(values.kind()) {
            let picks: Vec<f64> = (0..n).filter(|p| marks.as_ref().is_none_or(|m| m[*p])).map(|p| values.get_f64(p)).collect();
            Extremes { integer: true, count: picks.len(), low: picks.iter().copied().fold(f64::INFINITY, f64::min),
                       high: picks.iter().copied().fold(f64::NEG_INFINITY, f64::max), whole: true }
        } else {
            let (count, low, high, whole) = with_dtype!(values.dtype(), T => kernels::finite_extrema(values.slice::<T>(), marks.as_deref()));
            Extremes { integer: false, count, low, high, whole }
        }
    }

    /// What `_histogram_bins` takes of them: whether any, least, greatest, integral.
    fn read(self) -> (bool, f64, f64, bool) { (self.count > 0, self.low, self.high, self.integer || self.whole) }

    /// These, then `later`'s: the scan of both pools one after the other.
    pub fn then(self, later: Extremes) -> Extremes {
        if self.count == 0 { return Extremes { whole: self.whole && later.whole, ..later }; }
        if later.count == 0 { return Extremes { whole: self.whole && later.whole, ..self }; }
        let (low, high) = if self.integer { (f64::min(self.low, later.low), f64::max(self.high, later.high)) }
            else { (if later.low < self.low { later.low } else { self.low }, if later.high > self.high { later.high } else { self.high }) };
        Extremes { integer: self.integer, count: self.count + later.count, low, high, whole: self.whole && later.whole }
    }
}

/// `_histogram_plan`'s distribution codes of each sample's group on
/// `packed`: combined along the axes from the first to the last carrying a
/// group, and that last axis (none: one group).
fn group_codes(packed: &Packed, domains: &[(usize, Arc<super::view::UsedDomain>)]) -> (Vec<i64>, usize) {
    if domains.is_empty() { return (vec![0i64], 0); }
    let mut placed: Vec<(usize, Vec<i64>)> = Vec::new();
    for (dimension, domain) in domains {
        match &packed.rows {
            Some((repeat, point)) if *dimension < 2 => {
                let rows = if *dimension == 0 { repeat } else { point };
                placed.push((0, rows.iter().map(|r| domain.codes[*r as usize]).collect()));
            }
            Some(_) => placed.push((dimension - 1, domain.codes.to_vec())),
            None => placed.push((*dimension, domain.codes.to_vec())),
        }
    }
    let first = placed.iter().map(|p| p.0).min().unwrap();
    let code_axis = placed.iter().map(|p| p.0).max().unwrap();
    let code_shape: Vec<usize> = packed.values.shape()[first..=code_axis].to_vec();
    let total = product(&code_shape);
    let mut combined = vec![0i64; total];
    for ((dimension, codes), (_, domain)) in placed.iter().zip(domains) {
        let inner = product(&code_shape[dimension - first + 1..]);
        let extent = code_shape[dimension - first];
        let size = domain.values().len() as i64;
        for (flat, slot) in combined.iter_mut().enumerate() { *slot = *slot * size + codes[(flat / inner) % extent]; }
    }
    (combined, code_axis)
}

/// `_histogram_from_plan`'s counts: each group's samples against `edges`.
fn count_pool(values: &Array, valid: &Mask, group_codes: &[i64], code_axis: usize, groups: usize, edges: &[f64]) -> Vec<i64> {
    let stride = product(&values.shape()[code_axis + 1..]);
    let codes: &[i64] = if group_codes.len() == 1 && groups == 1 { &[] } else { group_codes };
    exact_counts(values, valid, codes, stride, groups, edges)
}

/// `np.linspace(start, stop, num)` in float64.
pub fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    let div = num.saturating_sub(1);
    let delta = stop - start;
    let mut y: Vec<f64> = (0..num).map(|i| i as f64).collect();
    if div > 0 {
        let step = delta / div as f64;
        if step == 0.0 { for v in y.iter_mut() { *v = (*v / div as f64) * delta; } }
        else { for v in y.iter_mut() { *v *= step; } }
    } else {
        for v in y.iter_mut() { *v *= delta; }
    }
    for v in y.iter_mut() { *v += start; }
    if num > 1 { y[num - 1] = stop; }
    y
}

/// `histogram_edges(low, high, bins, integral=...)`.
pub fn histogram_edges(low: f64, high: f64, bins: usize, integral: bool) -> Vec<f64> {
    let bins = bins.max(1);
    let high = if high <= low { low + 1.0 } else { high };
    if !integral { return linspace(low, high, bins + 1); }
    let first = (low + 0.5).floor() as i128;
    let last = (high + 0.5).floor() as i128;
    let covered = (last - first + 1).max(1);
    let width = (true_divide(covered, bins as i128).ceil() as i128).max(1);
    let count = true_divide(covered, width).ceil() as i128;
    (0..=count).map(|i| (first as f64 - 0.5) + width as f64 * i as f64).collect()
}

/// Python's `int / int`: the exact quotient rounded once to float64 (half
/// to even). Spans past 2^53 make `a as f64 / b as f64` round twice. Exact
/// for `|b| < 2^73` (a histogram divides an int64 span by a bin count or width).
pub fn true_divide(a: i128, b: i128) -> f64 {
    let negative = (a < 0) != (b < 0);
    let (n, d) = (a.unsigned_abs(), b.unsigned_abs());
    let signed = |x: f64| if negative { -x } else { x };
    if n < 1 << 53 && d < 1 << 53 { return signed(n as f64 / d as f64); }
    let bits = |x: u128| 128 - x.leading_zeros() as i32;
    // Scale so the integer quotient carries 55 or 56 bits, then round to 53
    // with the remainder as the sticky bit.
    let shift = 55 - (bits(n) - bits(d));
    let (numerator, denominator) = if shift >= 0 { (n << shift, d) } else { (n, d << -shift) };
    let (mut quotient, remainder) = (numerator / denominator, numerator % denominator);
    let extra = bits(quotient) - 53;
    let low = quotient & ((1u128 << extra) - 1);
    let half = 1u128 << (extra - 1);
    quotient >>= extra;
    if low > half || (low == half && (remainder != 0 || quotient & 1 == 1)) { quotient += 1; }
    signed(quotient as f64 * 2f64.powi(extra - shift))
}

/// Python's `round()` of a float to an integer (half to even).
fn round_half_even(value: f64) -> i128 {
    let rounded = value.round();
    if (value - value.trunc()).abs() == 0.5 { (2.0 * (value / 2.0).round()) as i128 } else { rounded as i128 }
}

/// `_aligned_integer_bins(edges)`: `(first, width)` of integer-aligned uniform edges.
pub fn aligned_integer_bins(edges: &[f64]) -> Option<(i128, i128)> {
    if edges.len() < 2 { return None; }
    if edges.windows(2).any(|w| !(w[1] - w[0]).is_finite()) || edges[0] < -9.223372036854776e18 || edges[edges.len() - 1] > 9.223372036854776e18 { return None; }
    let width = round_half_even(edges[1] - edges[0]);
    let first = round_half_even(edges[0] + 0.5);
    if width <= 0 { return None; }
    let expected = (0..edges.len()).map(|i| (first as f64 - 0.5) + width as f64 * i as f64);
    if !edges.iter().zip(expected).all(|(a, b)| *a == b) { return None; }
    Some((first, width))
}

/// `_counts_from_frequency(offset, frequency, edges)`.
pub fn counts_from_frequency(offset: i128, frequency: &[i64], edges: &[f64]) -> Option<Vec<i64>> {
    let (first, width) = aligned_integer_bins(edges)?;
    let size = frequency.len() as i128;
    let positions: Vec<usize> = (0..edges.len()).map(|i| (first + width * i as i128 - offset).clamp(0, size) as usize).collect();
    let mut running = vec![0i64; frequency.len() + 1];
    for (i, f) in frequency.iter().enumerate() { running[i + 1] = running[i] + f; }
    Some(positions.windows(2).map(|p| running[p[1]] - running[p[0]]).collect())
}

/// `_uniform_edges`: the edges are numpy's own linspace of their endpoints.
fn uniform_edges(edges: &[f64]) -> bool {
    if edges.len() < 2 { return false; }
    let (first, last) = (edges[0], edges[edges.len() - 1]);
    if !(first.is_finite() && last.is_finite() && (last - first).is_finite()) || last <= first { return false; }
    linspace(first, last, edges.len()) == edges && edges.windows(2).all(|w| w[0] < w[1])
}

/// Counts against explicit edges with numpy's semantics (last edge inclusive).
pub fn exact_counts(values: &Array, valid: &Mask, group_codes: &[i64], stride: usize, groups: usize, edges: &[f64]) -> Vec<i64> {
    let marks = if valid.all_true() { None } else { Some(valid.plane(values.len())) };
    if uniform_edges(edges) {
        return with_dtype!(values.dtype(), T => kernels::uniform_histogram(values.slice::<T>(), marks.as_deref(), group_codes, stride, edges, groups));
    }
    let bins = edges.len() - 1;
    let mut counts = vec![0i64; groups * bins];
    let last = edges[bins];
    for p in 0..values.len() {
        if marks.as_ref().is_some_and(|m| !m[p]) { continue; }
        let group = if group_codes.is_empty() { 0 } else { group_codes[(p / stride) % group_codes.len()] };
        if group < 0 { continue; }
        let v = values.get_f64(p);
        if !(v >= edges[0] && v <= last) { continue; }
        let bin = if v == last { bins - 1 } else { edges.partition_point(|e| *e <= v) - 1 };
        counts[group as usize * bins + bin] += 1;
    }
    counts
}

impl DataView {
    /// `_reduction_plan(refs)`: dense dimensions and mapped coordinates.
    fn reduction_plan(&self, refs: &[AxisRef]) -> Result<(Vec<usize>, Vec<AxisRef>)> {
        let mut dimensions = Vec::new();
        let mut coordinates = Vec::new();
        for axis in refs {
            let resolved = self.resolve(axis)?;
            let contract = &resolved.contract;
            let domain = self.schema.domain_of(contract);
            let local = domain.physical_dimension(contract.index);
            let siblings = (0..domain.axes.len()).filter(|i| domain.axes[*i].coordinate_of.is_none() && domain.physical_dimension(*i) == local).count();
            if siblings == 1 { if !dimensions.contains(&contract.dimension) { dimensions.push(contract.dimension); } } else { coordinates.push(axis.clone()); }
        }
        dimensions.sort_unstable();
        Ok((dimensions, coordinates))
    }

    /// `_carrier_group_codes(dimension, coordinates)`.
    fn carrier_group_codes(&self, dimension: usize, coordinates: &[AxisRef]) -> Result<(Vec<i64>, usize)> {
        let mut contracts = Vec::new();
        for axis in coordinates { let r = self.resolve(axis)?; if r.dimension() == dimension { contracts.push(r.contract.clone()); } }
        if contracts.is_empty() { return Err(ProjectionError::value("carrier group requires a mapped axis")); }
        if contracts.iter().any(|c| c.domain != contracts[0].domain) { return Err(ProjectionError::value("one physical dimension cannot span two domains")); }
        let domain = self.schema.domain_of(&contracts[0]);
        let local = domain.physical_dimension(contracts[0].index);
        let named: Vec<usize> = contracts.iter().map(|c| domain.coordinate_axis(c.index)).collect();
        let kept: Vec<usize> = (0..domain.axes.len()).filter(|i| domain.axes[*i].coordinate_of.is_none() && !named.contains(i)
            && domain.physical_dimension(*i) == local).collect();
        if kept.is_empty() { return Ok((vec![0; domain.size()], 1)); }
        let mut combined = vec![0i64; domain.size()];
        let mut span = 1usize;
        for axis in kept {
            let codes = domain.codes(axis);
            let size = domain.axes[axis].size as i64;
            for (slot, code) in combined.iter_mut().zip(codes.iter()) { *slot = *slot * size + code; }
            span *= domain.axes[axis].size;
        }
        let mut present = vec![false; span];
        for code in &combined { present[*code as usize] = true; }
        let mut remap = vec![-1i64; span];
        let mut used = 0usize;
        for (code, slot) in remap.iter_mut().enumerate() { if present[code] { *slot = used as i64; used += 1; } }
        Ok((combined.iter().map(|c| remap[*c as usize]).collect(), used))
    }

    /// The rows `_histogram_plan`'s window keeps: `(dimension, start, stop)`
    /// along the history's Point rows or the Repeat axis.
    fn histogram_window(&self, window: usize) -> (usize, usize, usize) {
        if let Some(layout) = &self.layout {
            let (start, stop) = layout.window_rows(&self.schema, window);
            (1, start, stop)
        } else if window > 1 {
            let stop = acquired_repeat_count(&self.snapshot);
            (0, stop.saturating_sub(window), stop)
        } else { (0, 0, self.schema.physical_shape[0]) }
    }

    /// The scratch a histogram plan reads. A still-filling run's window along
    /// Repeat packs only the segments it meets: the rows outside it are masked
    /// out anyway, and packing the whole run on every revision grew with it.
    fn histogram_scratch(&self, dimension: usize, start: usize, stop: usize, restricted: bool) -> Result<Scratch> {
        if let Block::Segments(segments) = &self.snapshot.block {
            if restricted && dimension == 0 && self.layout.is_none() && !self.packs_whole() {
                let selection: Vec<usize> = (0..segments.len()).filter(|i| {
                    let s = segments.get(*i);
                    s.origin[0] < stop && s.origin[0] + s.extent[0] > start
                }).collect();
                let rows: usize = segments.iter().map(|s| s.extent[0] * s.extent[1]).sum();
                let packed = Arc::new(pack(&self.snapshot, Some(&selection), false, Some(&self.pools))?);
                return Ok(Scratch { packed, operands: rows * product(&self.schema.physical_shape[2..]) });
            }
        }
        let packed = self.packed(false)?;
        Ok(Scratch { operands: packed.values.len(), packed })
    }

    /// `_histogram_plan(groups, reduced, aggregation, window)`.
    pub fn histogram_plan(&self, groups: &[AxisRef], reduced: &[AxisRef], aggregation: Reduction, window: usize) -> Result<Arc<HistogramPlan>> {
        if window == 0 { return Err(ProjectionError::value("history window must be positive")); }
        let aggregation = aggregation.statistic();
        self.caches.borrow_mut().frequency = None;
        let key = format!("{groups:?}|{reduced:?}|{aggregation:?}|{window}");
        if let Some((cached, plan)) = &self.caches.borrow().histogram_plan { if *cached == key { return Ok(plan.clone()); } }
        let shape = self.schema.physical_shape.clone();
        let (dimension, start, stop) = self.histogram_window(window);
        let restricted = start != 0 || stop != shape[dimension];
        // A still-filling run's reduction over its whole pool walks on from the previous revision.
        let walked = !restricted && !reduced.is_empty() && self.layout.is_none() && !self.packs_whole();
        let scratch = if walked { None } else { Some(self.histogram_scratch(dimension, start, stop, restricted)?) };
        let mut pool = None;
        if let Some(scratch) = &scratch {
            let packed = &scratch.packed;
            let values = packed.values.clone();
            let mut valid = packed.valid.clone();
            if restricted {
                let n = values.len();
                let leading = values.shape()[0];
                let inner_of = |axis: usize| -> usize { product(&values.shape()[axis + 1..]) };
                let keep: Vec<bool> = match &packed.rows {
                    None => { let inner = inner_of(dimension); let extent = shape[dimension]; (0..n).map(|p| { let i = (p / inner) % extent; i >= start && i < stop }).collect() }
                    Some((repeat, point)) => {
                        let indices = if dimension == 0 { repeat } else { point };
                        let inner = n / leading;
                        (0..n).map(|p| { let i = indices[p / inner] as usize; i >= start && i < stop }).collect()
                    }
                };
                valid = match valid {
                    Mask::Uniform(true) => Mask::Plane(keep.into()),
                    Mask::Uniform(false) => Mask::Uniform(false),
                    Mask::Plane(plane) => Mask::Plane(plane.iter().zip(keep).map(|(a, b)| *a && b).collect()),
                };
            }
            pool = Some((values, valid));
        }
        let (dimensions, coordinates) = self.reduction_plan(reduced)?;
        let mut domains = Vec::new();
        for axis in groups { domains.push((self.resolve(axis)?.dimension(), self.domain(axis)?)); }
        let mut keys: Vec<Vec<AxisValue>> = vec![vec![]];
        if !groups.is_empty() {
            for (_, domain) in &domains {
                keys = keys.iter().flat_map(|prefix| domain.values().iter().map(move |v| { let mut k = prefix.clone(); k.push(v.clone()); k })).collect();
            }
        }
        let (plan_values, plan_valid, combined, code_axis, operands) = if !reduced.is_empty() {
            let collapse = &dimensions;
            let mut grouped: Vec<(usize, Vec<i64>, usize)> = Vec::new();
            let mut coordinate_dims: Vec<usize> = coordinates.iter().map(|c| self.resolve(c).map(|r| r.dimension())).collect::<Result<_>>()?;
            coordinate_dims.sort_unstable();
            coordinate_dims.dedup();
            for dim in coordinate_dims { let (codes, count) = self.carrier_group_codes(dim, &coordinates)?; grouped.push((dim, codes, count)); }
            let extent = |axis: usize| grouped.iter().find(|g| g.0 == axis).map_or(shape[axis], |g| g.2);
            let keep_axes: Vec<usize> = (0..shape.len()).filter(|a| !collapse.contains(a)).collect();
            let out_shape: Vec<usize> = keep_axes.iter().map(|a| extent(*a)).collect();
            let mut strides = vec![0usize; keep_axes.len()];
            let mut stride = 1usize;
            for position in (0..keep_axes.len()).rev() { strides[position] = stride; stride *= out_shape[position]; }
            let count = stride.max(1);
            let codes: Vec<Arc<[i64]>> = keep_axes.iter().map(|a| grouped.iter().find(|g| g.0 == *a).map_or_else(
                || (0..shape[*a] as i64).collect::<Arc<[i64]>>(), |g| g.1.clone().into())).collect();
            let walk_whole = |packed: &Packed, values: &Array, valid: &Mask| {
                let (axis_codes, axes) = packed_codes(packed, &codes, &keep_axes);
                let aggregated = axis_aggregate(values, valid, &axis_codes, &axes, &out_shape, aggregation, None);
                (aggregated.out, aggregated.counts)
            };
            let (out, counts) = match (&scratch, &pool) {
                (Some(scratch), Some((values, valid))) => walk_whole(&scratch.packed, values, valid),
                _ => match self.accumulated(&codes, &keep_axes, &out_shape, aggregation, false)? {
                    Some((out, counts, _)) => (out, counts),
                    None => { let packed = self.packed(false)?; walk_whole(&packed, &packed.values, &packed.valid) }
                },
            };
            let mut combined = vec![0i64; count];
            for (dimension, domain) in &domains {
                let mut domain_codes: Vec<i64> = domain.codes.to_vec();
                if let Some(carrier) = grouped.iter().find(|g| g.0 == *dimension) {
                    let position = keep_axes.iter().position(|a| a == dimension).unwrap();
                    let mut regrouped = vec![-1i64; out_shape[position]];
                    for (row, group) in carrier.1.iter().enumerate() { regrouped[*group as usize] = domain_codes[row]; }
                    domain_codes = regrouped;
                }
                let position = keep_axes.iter().position(|a| a == dimension).unwrap();
                let size = domain.values().len() as i64;
                for (bucket, slot) in combined.iter_mut().enumerate() {
                    *slot = *slot * size + domain_codes[(bucket / strides[position]) % out_shape[position]];
                }
            }
            let valid: Vec<bool> = counts.iter().map(|c| *c > 0).collect();
            let operands = out.len();
            (Array::vector(out), Mask::from_vec(valid), combined, 0usize, operands)
        } else {
            let (scratch, (values, valid)) = (scratch.as_ref().expect("an unreduced pool is packed"), pool.expect("an unreduced pool is packed"));
            let (combined, code_axis) = group_codes(&scratch.packed, &domains);
            (values, valid, combined, code_axis, scratch.operands)
        };
        let plan = Arc::new(HistogramPlan { values: plan_values, valid: plan_valid, group_codes: combined, code_axis, group_keys: keys, operands });
        self.caches.borrow_mut().histogram_plan = Some((key, plan.clone()));
        Ok(plan)
    }

    /// The histogram of a still-filling run's pool (the whole run or a window
    /// of whole segments along Repeat, no reduction), from each segment's
    /// facts kept by [`HistogramCarry`]: the pool's extremes are its segments'
    /// folded in row order (the order a scan of the packed pool meets them),
    /// and its counts are their counts summed while the edges stay the carried
    /// ones; edges that moved count every segment of the pool again. `between`
    /// runs where the whole computation checks the facet. None where the plan
    /// path answers (a layout, a reduction, a segment the window cuts).
    #[allow(clippy::too_many_arguments)]
    pub fn counted_histogram(&self, projection: &mut Projection, plan: &ProjectionPlan, groups: &[AxisRef], reduced: &[AxisRef],
                             window: usize, between: impl FnOnce() -> Result<()>) -> Result<Option<HistogramData>> {
        let dtype = self.schema.value.dtype;
        if !reduced.is_empty() || window == 0 || self.layout.is_some() || !"biuf".contains(dtype.kind()) || dtype == DType::F16 { return Ok(None); }
        let Block::Segments(segments) = &self.snapshot.block else { return Ok(None) };
        if self.packs_whole() { return Ok(None); }
        let points = self.schema.physical_shape[1];
        let (_, start, stop) = self.histogram_window(window);
        let first_row = |s: &super::model::PlacedRef<'_>| s.origin[0] * points + s.origin[1];
        let last_row = |s: &super::model::PlacedRef<'_>| (s.origin[0] + s.extent[0] - 1) * points + s.origin[1] + s.extent[1] - 1;
        let mut chosen: Vec<usize> = (0..segments.len()).filter(|i| { let s = segments.get(*i); s.origin[0] < stop && s.origin[0] + s.extent[0] > start }).collect();
        if chosen.is_empty() || chosen.iter().any(|i| { let s = segments.get(*i); s.origin[0] < start || s.origin[0] + s.extent[0] > stop }) { return Ok(None); }
        chosen.sort_by_key(|i| first_row(&segments.get(*i)));
        if chosen.windows(2).any(|pair| last_row(&segments.get(pair[0])) >= first_row(&segments.get(pair[1]))) { return Ok(None); }
        let identity = |s: &super::model::PlacedRef<'_>| {
            let planes = s.values.bytes();
            (!planes.is_empty()).then(|| (planes.as_ptr() as usize, planes.len(), s.origin))
        };
        let Some(identities) = chosen.iter().map(|i| identity(&segments.get(*i))).collect::<Option<Vec<_>>>() else { return Ok(None) };
        // What `_histogram_plan` reads before the samples, with its refusals.
        self.caches.borrow_mut().frequency = None;
        let mut domains = Vec::new();
        for axis in groups { domains.push((self.resolve(axis)?.dimension(), self.domain(axis)?)); }
        let mut keys: Vec<Vec<AxisValue>> = vec![vec![]];
        for (_, domain) in &domains {
            keys = keys.iter().flat_map(|prefix| domain.values().iter().map(move |v| { let mut k = prefix.clone(); k.push(v.clone()); k })).collect();
        }
        let query: Vec<(usize, Arc<[i64]>, usize)> = domains.iter().map(|(dimension, domain)| (*dimension, domain.codes.clone(), domain.values().len())).collect();
        let carried = self.caches.borrow_mut().counted.take().filter(|carry| carry.query.len() == query.len()
            && carry.query.iter().zip(&query).all(|(old, new)| old.0 == new.0 && old.2 == new.2 && new.1.starts_with(&old.1)));
        let known = |id: &(usize, usize, [usize; 2])| carried.as_ref().and_then(|carry| carry.facts.get(id));
        let mut packs: Vec<Option<Packed>> = chosen.iter().map(|_| None).collect();
        let mut scans = Vec::with_capacity(chosen.len());
        for ((index, id), packed) in chosen.iter().zip(&identities).zip(packs.iter_mut()) {
            scans.push(match known(id) {
                Some((extremes, _)) => *extremes,
                None => {
                    let segment = pack(&self.snapshot, Some(&[*index]), false, Some(&self.pools))?;
                    let extremes = Extremes::of(&segment.values, &segment.valid);
                    *packed = Some(segment);
                    extremes
                }
            });
        }
        let extremes = scans.iter().fold(Extremes::none(), |pool, segment| pool.then(*segment));
        let edges = projection.histogram_bins(self, plan, Binned::Extremes(extremes))?;
        between()?;
        validate_edges(&edges)?;
        let same_edges = carried.as_ref().is_some_and(|carry| carry.edges.len() == edges.len()
            && carry.edges.iter().zip(&edges).all(|(a, b)| a.to_bits() == b.to_bits()));
        let mut counts = vec![0i64; keys.len() * (edges.len() - 1)];
        let mut facts = HashMap::with_capacity(chosen.len());
        for (((index, id), packed), scan) in chosen.iter().zip(identities).zip(packs).zip(scans) {
            let counted = match known(&id) {
                Some((_, counted)) if same_edges => counted.clone(),
                _ => {
                    let segment = match packed { Some(segment) => segment, None => pack(&self.snapshot, Some(&[*index]), false, Some(&self.pools))? };
                    let (codes, code_axis) = group_codes(&segment, &domains);
                    count_pool(&segment.values, &segment.valid, &codes, code_axis, keys.len(), &edges)
                }
            };
            for (total, count) in counts.iter_mut().zip(&counted) { *total += count; }
            facts.insert(id, (scan, counted));
        }
        self.caches.borrow_mut().counted = Some(HistogramCarry { query, segments: segments.clone(), facts, edges: edges.clone() });
        Ok(Some(self.histogram_from_counts(edges, counts, keys)))
    }

    /// `_histogram_from_counts`.
    pub fn histogram_from_counts(&self, edges: Vec<f64>, counts: Vec<i64>, group_keys: Vec<Vec<AxisValue>>) -> HistogramData {
        let bins = edges.len() - 1;
        let centers: Vec<f64> = (0..bins).map(|i| (edges[i] + edges[i + 1]) / 2.0).collect();
        let label = self.value_label();
        let groups = group_keys.len();
        HistogramData {
            revision: self.revision(), generation: self.generation().clone(),
            edges: QuantityArray::new(Array::vector(edges), self.value_unit.clone(), self.value_display_unit.clone(), label.clone()),
            centers: QuantityArray::new(Array::vector(centers), self.value_unit.clone(), self.value_display_unit.clone(), label),
            counts: Array::from_vec(counts, vec![groups, bins]), group_keys,
        }
    }

    /// `_histogram_from_plan(edges, plan)`.
    pub fn histogram_from_plan(&self, edges: &[f64], plan: &HistogramPlan) -> Result<HistogramData> {
        if !"biuf".contains(plan.values.kind()) { return Err(ProjectionError::view("dataset values must be real numeric for this projection")); }
        validate_edges(edges)?;
        let counts = count_pool(&plan.values, &plan.valid, &plan.group_codes, plan.code_axis, plan.group_keys.len(), edges);
        Ok(self.histogram_from_counts(edges.to_vec(), counts, plan.group_keys.clone()))
    }

    /// `window_frequency(window)`: an integer history window's value table,
    /// moved by the immutable ranges that entered and left since the carried
    /// revision instead of recounted (`_advance_frequency`).
    pub fn window_frequency(&self, window: usize) -> Result<Option<(i128, Arc<[i64]>)>> {
        let Some(layout) = self.layout.clone() else { self.caches.borrow_mut().frequency = None; return Ok(None) };
        if self.snapshot.window.is_none() || !"iu".contains(self.schema.value.dtype.kind()) {
            self.caches.borrow_mut().frequency = None;
            return Ok(None);
        }
        if window == 0 { return Err(ProjectionError::value("history window must be positive")); }
        let first = layout.window_rows(&self.schema, window).0;
        let entries = frequency_entries(&self.snapshot, first);
        let carried = self.caches.borrow_mut().frequency.take();
        let mut counted = None;
        if let Some(carry) = carried.as_ref() {
            if carry.generation == self.snapshot.generation && carry.cell_value_same(&self.schema) {
                counted = advance_frequency(self, carry, &entries);
            }
        }
        if counted.is_none() { counted = count_frequency(&self.schema, &entries, true); }
        let Some((offset, table)) = counted else { return Ok(None) };
        let table: Arc<[i64]> = table.into();
        self.caches.borrow_mut().frequency = Some(FrequencyCarry {
            generation: self.snapshot.generation.clone(), schema: self.schema.clone(), entries, offset, table: table.clone(),
        });
        Ok(Some((offset, table)))
    }

    /// Drop the carried window table: this panel counts no window.
    pub fn release_window_frequency(&self) { self.caches.borrow_mut().frequency = None; }
}

fn validate_edges(edges: &[f64]) -> Result<()> {
    if edges.len() < 2 || !edges.iter().all(|e| e.is_finite()) { return Err(ProjectionError::value("histogram edges must be a finite one-dimensional sequence")); }
    if edges.windows(2).any(|w| w[1] - w[0] <= 0.0) { return Err(ProjectionError::value("histogram edges must be strictly increasing")); }
    Ok(())
}

/// What `_histogram_bins` bins: a plan's pool (with the whole pool's sample
/// count), a frequency table, or a pool's extremes found ahead.
pub enum Binned<'a> { Values(&'a Array, &'a Mask, usize), Frequency(i128, &'a [i64]), Extremes(Extremes) }

impl Projection {
    /// `FitProjection._histogram_bins`: stable canonical edges, retained.
    pub fn histogram_bins(&mut self, view: &DataView, plan: &ProjectionPlan, binned: Binned<'_>) -> Result<Vec<f64>> {
        let display = &plan.display;
        let count = display.bin_count.ok_or_else(|| ProjectionError::value("histogram projection needs bin_count"))?;
        let (mut has_values, mut data_low, mut data_high, integral);
        match binned {
            Binned::Frequency(offset, table) => {
                integral = true;
                let occupied: Vec<usize> = (0..table.len()).filter(|i| table[*i] != 0).collect();
                has_values = !occupied.is_empty();
                data_low = if has_values { (offset + occupied[0] as i128) as f64 } else { 0.0 };
                data_high = if has_values { (offset + *occupied.last().unwrap() as i128) as f64 } else { 0.0 };
            }
            Binned::Values(values, valid, n) => {
                // A window's edges come from the samples it keeps, whatever the validity outside it.
                if values.dtype() == DType::F16 && n > 0 {
                    return Err(ProjectionError::new(super::ErrorKind::NotImplemented, "float16"));
                }
                (has_values, data_low, data_high, integral) = Extremes::of(values, valid).read();
            }
            Binned::Extremes(extremes) => (has_values, data_low, data_high, integral) = extremes.read(),
        }
        let fraction = plan.defaults.histogram_domain_padding_fraction.0;
        let mode = relim_mode(display);
        let previous = self.histogram.clone();
        let (mut low, mut high);
        if let Some(previous) = previous.as_ref().filter(|p| p.requested_bins == count && mode == RelimMode::Normal) {
            (low, high) = previous.domain;
            if has_values && (data_low < low || data_high > high) {
                // Python's builtins: a tie keeps the first argument.
                let envelope_low = if data_low < low { data_low } else { low };
                let envelope_high = if data_high > high { data_high } else { high };
                let padding = fraction * (envelope_high - envelope_low);
                if data_low < low { low = data_low - padding; }
                if data_high > high { high = data_high + padding; }
            }
        } else if mode == RelimMode::Fixed {
            if !has_values { data_low = 0.0; data_high = 1.0; }
            let written = |value: Option<super::plan::Bits>, fallback: f64| -> f64 {
                value.map_or(fallback, |v| view.value_display_unit.convert_value_to(v.0, &view.value_unit).unwrap_or(f64::NAN))
            };
            low = written(display.x_min, data_low);
            high = written(display.x_max, data_high);
        } else {
            if !has_values { data_low = 0.0; data_high = 1.0; has_values = false; }
            if data_low == data_high {
                let half = (data_low.abs() * 0.05).max(0.5);
                data_low -= half;
                data_high += half;
            }
            low = data_low;
            high = data_high;
            if mode != RelimMode::Tight {
                let padding = fraction * (high - low);
                low -= padding;
                high += padding;
            }
        }
        let edges = histogram_edges(low, high, count, integral && has_values);
        if !(low.is_finite() && high.is_finite() && low < high) { return Err(ProjectionError::value("histogram projection domain must be increasing")); }
        if !edges.iter().all(|e| e.is_finite()) || edges.windows(2).any(|w| w[1] - w[0] <= 0.0) {
            return Err(ProjectionError::value("histogram projection edges must be finite and increasing"));
        }
        let selected = HistogramProjection { bin_count: edges.len() - 1, requested_bins: count, domain: (low, high), edges };
        if self.histogram.as_ref() != Some(&selected) { self.histogram = Some(selected); }
        Ok(self.histogram.as_ref().unwrap().edges.clone())
    }
}

/// The histogram kind's `build_payload`.
pub fn histogram_payload(projection: &mut Projection, view: &DataView, plan: &ProjectionPlan, group: Option<&AxisRef>,
                         reduced: &[AxisRef], reduction: Reduction, window: usize) -> Result<HistogramData> {
    if reduced.is_empty() && group.is_none() && view.has_primary_index() {
        if let Some((offset, table)) = view.window_frequency(window)? {
            let edges = projection.histogram_bins(view, plan, Binned::Frequency(offset, &table))?;
            validate_edges(&edges)?;
            if let Some(counts) = counts_from_frequency(offset, &table, &edges) {
                return Ok(view.histogram_from_counts(edges, counts, vec![vec![]]));
            }
        }
    }
    let groups: Vec<AxisRef> = group.into_iter().cloned().collect();
    if let Some(histogram) = view.counted_histogram(projection, plan, &groups, reduced, window, || Ok(()))? { return Ok(histogram); }
    let histogram_plan = view.histogram_plan(&groups, reduced, reduction, window)?;
    let edges = projection.histogram_bins(view, plan, Binned::Values(&histogram_plan.values, &histogram_plan.valid, histogram_plan.operands))?;
    view.histogram_from_plan(&edges, &histogram_plan)
}

/// The facet-grid kind's `build_payload` for histogram cells (`_histogram_facet`).
pub fn histogram_facet_payload(projection: &mut Projection, view: &DataView, plan: &ProjectionPlan, facet: Option<&AxisRef>,
                               cell: &CellSpec, window: usize) -> Result<FacetData> {
    let CellSpec::Histogram { group, reduced, reduction } = cell else { unreachable!() };
    let groups: Vec<AxisRef> = facet.into_iter().chain(group.iter()).cloned().collect();
    let histogram = match view.counted_histogram(projection, plan, &groups, reduced, window, || view.validate_facet(facet, cell))? {
        Some(histogram) => histogram,
        None => {
            let pool = view.histogram_plan(&groups, reduced, *reduction, window)?;
            let edges = projection.histogram_bins(view, plan, Binned::Values(&pool.values, &pool.valid, pool.operands))?;
            view.validate_facet(facet, cell)?;
            view.histogram_from_plan(&edges, &pool)?
        }
    };
    let cells = match facet {
        None => vec![FacetCell { facet_index: Some(0), facet_value_canonical: super::nd::Scalar::Int(1),
            facet_value_display: super::nd::Scalar::Int(1), label: "Facet 1".into(), payload: super::output::Payload::Histogram(histogram) }],
        Some(_) => {
            let bins = histogram.edges.canonical.len() - 1;
            let counts = histogram.counts.slice::<i64>();
            let mut cells = Vec::new();
            let mut start = 0;
            let keys = &histogram.group_keys;
            while start < keys.len() {
                let value = keys[start][0].clone();
                let mut stop = start + 1;
                while stop < keys.len() && keys[stop][0] == value { stop += 1; }
                let part = HistogramData { counts: Array::from_vec(counts[start * bins..stop * bins].to_vec(), vec![stop - start, bins]),
                    group_keys: keys[start..stop].iter().map(|k| k[1..].to_vec()).collect(), ..histogram.clone() };
                cells.push(FacetCell { facet_index: Some(cells.len()), facet_value_canonical: value.canonical.clone(),
                    facet_value_display: value.display.clone(), label: value.label.clone(), payload: super::output::Payload::Histogram(part) });
                start = stop;
            }
            cells
        }
    };
    Ok(FacetData { revision: view.revision(), generation: view.generation().clone(), cells })
}

/// One counted contribution: rows `start..` of one immutable segment.
#[derive(Clone)]
pub struct FrequencyEntry { segment: Arc<Segment>, start: usize }

/// `_WindowFrequency`: the table and the immutable ranges that formed it.
/// Holding the segments keeps their planes alive, so an equal plane
/// address names the very same immutable data.
pub struct FrequencyCarry {
    generation: Arc<str>,
    schema: Arc<super::model::Schema>,
    entries: Vec<FrequencyEntry>,
    offset: i128,
    table: Arc<[i64]>,
}

impl FrequencyCarry {
    fn cell_value_same(&self, schema: &super::model::Schema) -> bool {
        self.schema.cell.same_spec(&schema.cell) && self.schema.value.same(&schema.value)
    }
}

fn frequency_entries(snapshot: &super::model::Snapshot, first: usize) -> Vec<FrequencyEntry> {
    let entry = |origin: [usize; 2], extent: [usize; 2], segment: &Arc<Segment>| {
        let start = first.saturating_sub(origin[1]);
        (start < extent[1]).then(|| FrequencyEntry { segment: segment.clone(), start })
    };
    match &snapshot.block {
        Block::Values { values, mask, .. } => {
            let shape = &snapshot.schema.physical_shape;
            let whole = Arc::new(Segment { extent: [shape[0], shape[1]], values: values.clone(), mask: mask.clone(), sigma: None });
            entry([0, 0], whole.extent, &whole).into_iter().collect()
        }
        Block::Segments(segments) => segments.iter().filter_map(|s| entry(s.origin, s.extent, s.segment)).collect(),
    }
}

fn same_planes(a: &Arc<Segment>, b: &Arc<Segment>) -> bool {
    Arc::ptr_eq(a, b) || a.values.same_storage(&b.values) && match (&a.mask, &b.mask) {
        (Mask::Uniform(x), Mask::Uniform(y)) => x == y,
        (Mask::Plane(x), Mask::Plane(y)) => Bools::ptr_eq(x, y) || x == y,
        _ => false,
    }
}

/// Values of rows `start..stop` of one segment, each valid sample once.
fn visit_rows(schema: &super::model::Schema, segment: &Segment, start: usize, stop: usize, mut visit: impl FnMut(i128)) {
    if matches!(segment.mask, Mask::Uniform(false)) || start >= stop { return; }
    let extent = segment.extent;
    let cell = segment.values.len() / (extent[0] * extent[1]).max(1);
    let full = match &segment.mask { Mask::Plane(compact) => Some(expand_mask(schema, compact, extent[0] * extent[1])), _ => None };
    with_dtype!(schema.value.dtype, T => {
        let slice = segment.values.slice::<T>();
        for r in 0..extent[0] {
            for p in start..stop {
                let base = (r * extent[1] + p) * cell;
                match &full {
                    None => for v in &slice[base..base + cell] { visit(super::nd::Elem::int(*v)); },
                    Some(f) => for c in 0..cell { if f[base + c] { visit(super::nd::Elem::int(slice[base + c])); } },
                }
            }
        }
    });
}

/// `_count_frequency`: one table over the ranges (the whole dtype for narrow integers).
fn count_frequency(schema: &super::model::Schema, entries: &[FrequencyEntry], full_dtype: bool) -> Option<(i128, Vec<i64>)> {
    let dtype = schema.value.dtype;
    let (low, high) = dtype.integer_bounds()?;
    let (offset, size) = if full_dtype && dtype.itemsize() <= 2 { (low, (high - low + 1) as usize) } else {
        let (mut lo, mut hi, mut any) = (i128::MAX, i128::MIN, false);
        for entry in entries { visit_rows(schema, &entry.segment, entry.start, entry.segment.extent[1], |v| { lo = lo.min(v); hi = hi.max(v); any = true; }); }
        if !any { (0, 1) } else {
            if lo < i64::MIN as i128 || hi > i64::MAX as i128 || hi - lo + 1 > FREQUENCY_LEVEL_LIMIT { return None; }
            (lo, (hi - lo + 1) as usize)
        }
    };
    let mut table = vec![0i64; size];
    for entry in entries { visit_rows(schema, &entry.segment, entry.start, entry.segment.extent[1], |v| table[(v - offset) as usize] += 1); }
    Some((offset, table))
}

/// `_advance_frequency`: subtract the ranges that left, add those that arrived.
fn advance_frequency(view: &DataView, carry: &FrequencyCarry, entries: &[FrequencyEntry]) -> Option<(i128, Vec<i64>)> {
    if entries.is_empty() { return Some((carry.offset, vec![0; carry.table.len()])); }
    if carry.entries.is_empty() { return None; }
    let mut matched_old = vec![false; carry.entries.len()];
    let mut leaving: Vec<(Arc<Segment>, usize, usize)> = Vec::new();
    let mut arriving: Vec<(Arc<Segment>, usize, usize)> = Vec::new();
    let mut any = false;
    // A window slides and a run appends: search on from the last match first.
    let mut cursor = 0usize;
    for entry in entries {
        let candidate = |i: &usize| !matched_old[*i] && same_planes(&carry.entries[*i].segment, &entry.segment);
        let found = (cursor..carry.entries.len()).take(8).find(candidate).or_else(|| (0..carry.entries.len()).find(candidate))
            .map(|i| { cursor = i + 1; (i, &carry.entries[i]) });
        match found {
            Some((i, old)) => {
                matched_old[i] = true;
                any = true;
                if old.start < entry.start { leaving.push((old.segment.clone(), old.start, entry.start)); }
                if entry.start < old.start { arriving.push((entry.segment.clone(), entry.start, old.start)); }
            }
            None => arriving.push((entry.segment.clone(), entry.start, entry.segment.extent[1])),
        }
    }
    if !any { return None; }
    for (i, old) in carry.entries.iter().enumerate() { if !matched_old[i] { leaving.push((old.segment.clone(), old.start, old.segment.extent[1])); } }
    let schema = &view.schema;
    let mut offset = carry.offset;
    let mut table = carry.table.to_vec();
    for (ranges, sign) in [(&leaving, -1i64), (&arriving, 1i64)] {
        if ranges.is_empty() { continue; }
        let (mut lo, mut hi, mut seen) = (i128::MAX, i128::MIN, false);
        for (segment, start, stop) in ranges.iter() { visit_rows(schema, segment, *start, *stop, |v| { lo = lo.min(v); hi = hi.max(v); seen = true; }); }
        if !seen { continue; }
        let start = offset.min(lo);
        let stop = (offset + table.len() as i128).max(hi + 1);
        if stop - start > FREQUENCY_LEVEL_LIMIT { return None; }
        if start != offset || stop != offset + table.len() as i128 {
            let mut grown = vec![0i64; (stop - start) as usize];
            grown[(offset - start) as usize..(offset - start) as usize + table.len()].copy_from_slice(&table);
            table = grown;
            offset = start;
        }
        for (segment, start, stop) in ranges.iter() { visit_rows(schema, segment, *start, *stop, |v| table[(v - offset) as usize] += sign); }
    }
    Some((offset, table))
}

/// `aligned_histogram_edges(values, bins, limits=None)`: edges for one pool;
/// integer-valued samples (every one of them) get integer-aligned bins.
pub fn aligned_histogram_edges(values: &Array, bins: usize, limits: Option<(f64, f64)>) -> Vec<f64> {
    let float = values.kind() == 'f';
    let flat: Vec<f64> = if float { values.to_f64().into_iter().filter(|v| v.is_finite()).collect() } else { values.to_f64() };
    let (low, high) = match limits {
        Some(limits) => limits,
        None if !flat.is_empty() => {
            let rule = values.dtype().extremes();
            (crate::numeric::reduce_contiguous(flat[0], &flat[1..], rule, true), crate::numeric::reduce_contiguous(flat[0], &flat[1..], rule, false))
        }
        None => (0.0, 1.0),
    };
    let integral = "iub".contains(values.kind()) || (!flat.is_empty() && flat.iter().all(|v| *v == v.floor()));
    histogram_edges(low, high, bins, integral)
}

/// `histogram_counts(values, edges, valid=None)`: numpy's counts for explicit
/// edges (half-open bins, the last one closed), however they are computed.
pub fn histogram_counts(values: &Array, edges: &[f64], valid: Option<&[bool]>) -> Vec<i64> {
    let mask = match valid { None => Mask::Uniform(true), Some(v) => Mask::Plane(v.into()) };
    exact_counts(values, &mask, &[], 1, 1, edges)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edges_and_counts_follow_numpy() {
        // An int64 span past 2^53: Python divides the integers exactly.
        let edges = histogram_edges(-9194953342047675914.0, 8069816977415715667.0, 1000, true);
        assert_eq!(edges.len(), 1001);
        assert_eq!(true_divide(17264770319463392257, 1000).ceil(), 17264770319463392.0);
        assert_eq!(true_divide(17264770319463392257, 1000), 17264770319463392.257_f64);
        assert_eq!(linspace(0.0, 1.0, 5), vec![0.0, 0.25, 0.5, 0.75, 1.0]);
        // Integer samples bin on half-integers; the count shrinks to the span.
        assert_eq!(histogram_edges(2.0, 5.0, 10, true), vec![1.5, 2.5, 3.5, 4.5, 5.5]);
        let values = Array::vector(vec![1u8, 2, 2, 3, 9]);
        let edges = aligned_histogram_edges(&values, 4, None);
        assert_eq!(edges, vec![0.5, 3.5, 6.5, 9.5]);
        assert_eq!(histogram_counts(&values, &edges, None), vec![4, 0, 1]);
        let floats = Array::vector(vec![0.0f64, 0.5, 1.0, f64::NAN]);
        assert_eq!(histogram_counts(&floats, &[0.0, 0.5, 1.0], None), vec![1, 2]);
        assert_eq!(aligned_histogram_edges(&floats, 2, None), vec![0.0, 0.5, 1.0]);
        assert_eq!(counts_from_frequency(0, &[1, 2, 3, 4], &[-0.5, 1.5, 3.5]), Some(vec![3, 7]));
    }
}
