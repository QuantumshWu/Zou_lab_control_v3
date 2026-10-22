//! The three reduction owners of `DataView`, each in its own summation
//! order: the dense tensor reduction (`_dense_tensor_projection`), the
//! factored lattice (`_factored_planes`) and the generic axis-code
//! aggregation (`_segmented_axes`), plus the one SEM rule (`_sem_of_mean`).

use std::sync::Arc;

use super::kernels::{self, AxisCodes};
use super::nd::{Array, DType, Elem, Mask, Plane, product};
use super::model::Block;
use super::view::{pack, DataView, Packed, UsedDomain};
use super::{ErrorKind, Result};
use crate::kinds::{AxisRef, Reduction};
use crate::numeric::{add_reduce, extreme_reduce, int_add_reduce, leading_sum};
use crate::with_dtype;

// ------------------------------------------------------------------- SEM
/// `_sem_from_moments`.
pub fn sem_from_moments(mean: &[f64], mean_of_squares: &[f64], counts: &[i64], mean_sigma_square: Option<&[f64]>) -> Vec<f64> {
    let eps = f64::EPSILON;
    (0..mean.len()).map(|i| {
        let n = counts[i] as f64;
        let squared_mean = mean[i] * mean[i];
        let mut spread = mean_of_squares[i] - squared_mean;
        let roundoff = ((mean_of_squares[i].abs() + squared_mean) * eps) * (n.max(1.0) + 3.0);
        if spread <= roundoff { spread = 0.0; }
        if !spread.is_nan() && spread < 0.0 { spread = 0.0; }
        let mut variance = if n > 1.0 { spread * n / (n - 1.0) } else { f64::NAN };
        if let Some(stated) = mean_sigma_square { if variance.is_nan() { variance = stated[i]; } }
        let sem = (variance / n).sqrt();
        if n < 1.0 { f64::NAN } else { sem }
    }).collect()
}

/// `_sem_of_mean`: per bucket, centred on the bucket's own mean.
pub fn sem_of_mean(means: &[f64], counts: &[i64], samples: &Array, sigma: Option<&Array>,
                   centred: &dyn Fn(&Array, &[f64]) -> Result<(Vec<f64>, Vec<f64>)>) -> Result<Vec<f64>> {
    if sigma.is_none() && !counts.iter().any(|c| *c > 1) { return Ok(vec![f64::NAN; means.len()]); }
    let (first, second) = centred(samples, means)?;
    let stated = match sigma { Some(sigma) => Some(centred(sigma, &vec![0.0; means.len()])?.1), None => None };
    Ok(sem_from_moments(&first, &second, counts, stated.as_deref()))
}

// ------------------------------------------------------- dense reduction
/// `_masked_leading_reduce(values, usable, aggregation)` over `(pool, *kept)`.
pub fn masked_leading_reduce(values: &Array, usable: &Mask, aggregation: Reduction) -> (Array, Plane<i64>) {
    let pool = values.shape()[0];
    let kept_shape: Vec<usize> = values.shape()[1..].to_vec();
    let rest = product(&kept_shape);
    if pool == 1 {
        let valid: Vec<i64> = (0..rest).map(|i| i64::from(usable.get(i))).collect();
        return (values.index0(0), Plane::Values(valid));
    }
    let float = values.kind() == 'f';
    let buffered = values.dtype() != DType::F64;
    if usable.all_true() {
        let counts = Plane::Uniform(pool as i64, rest);
        let result = with_dtype!(values.dtype(), T => {
            let slice = values.slice::<T>();
            match aggregation {
                Reduction::Mean => Array::from_vec(leading_sum(slice, pool, rest, buffered).into_iter().map(|s| s / pool as f64).collect(), kept_shape.clone()),
                Reduction::Sum if float => Array::from_vec(leading_sum(slice, pool, rest, buffered), kept_shape.clone()),
                Reduction::Sum => int_sums(int_add_reduce(slice, &[pool, rest], &[0], None, as_int::<T>), values.dtype(), kept_shape.clone()),
                Reduction::Min | Reduction::Max => Array::from_vec(
                    extreme_reduce(slice, &[pool, rest], &[0], None, None, values.dtype().extremes(), aggregation == Reduction::Min),
                    kept_shape.clone()),
                _ => values.index0(0),
            }
        });
        return (result, counts);
    }
    let marks = usable.plane(values.len());
    if values.dtype() == DType::F64 && values.len() >= kernels::MASKED_LEADING_MIN_SAMPLES && rest >= 2 && aggregation != Reduction::Last {
        let code = match aggregation { Reduction::Mean => kernels::REDUCE_MEAN, Reduction::Sum => kernels::REDUCE_SUM,
            Reduction::Min => kernels::REDUCE_MIN, Reduction::Max => kernels::REDUCE_MAX, _ => kernels::REDUCE_FIRST };
        let (out, counts) = kernels::masked_leading_float64(values.slice::<f64>(), &marks, pool, rest, code);
        return (Array::from_vec(out, kept_shape), Plane::Values(counts));
    }
    let counts: Vec<i64> = int_add_reduce(&marks, &[pool, rest], &[0], Some(&marks), |_| 1).into_iter().map(|v| v as i64).collect();
    let empty_nan = |result: Vec<f64>| -> Vec<f64> { result.into_iter().zip(&counts).map(|(v, c)| if *c > 0 { v } else { f64::NAN }).collect() };
    let result = with_dtype!(values.dtype(), T => {
        let slice = values.slice::<T>();
        match aggregation {
            Reduction::Mean => Array::from_vec(add_reduce(slice, &[pool, rest], &[0], Some(&marks), buffered).into_iter().zip(&counts)
                .map(|(s, c)| s / *c as f64).collect(), kept_shape.clone()),
            Reduction::Sum if float => Array::from_vec(empty_nan(add_reduce(slice, &[pool, rest], &[0], Some(&marks), buffered)), kept_shape.clone()),
            Reduction::Sum => {
                let sums = int_sums(int_add_reduce(slice, &[pool, rest], &[0], Some(&marks), as_int::<T>), values.dtype(), kept_shape.clone());
                Array::from_vec(empty_nan(sums.to_f64()), kept_shape.clone())
            }
            Reduction::Min | Reduction::Max => {
                // Converted to float64 first, then `np.min(where=, initial=±inf)`.
                let minimum = aggregation == Reduction::Min;
                let initial = if minimum { f64::INFINITY } else { f64::NEG_INFINITY };
                let out = extreme_reduce(slice, &[pool, rest], &[0], Some(&marks), Some(initial), DType::F64.extremes(), minimum);
                Array::from_vec(empty_nan(out), kept_shape.clone())
            }
            _ => {
                let first: Vec<usize> = (0..rest).map(|column| (0..pool).find(|row| marks[row * rest + column]).unwrap_or(0)).collect();
                let picked = values.take_flat(&first.iter().enumerate().map(|(column, row)| row * rest + column).collect::<Vec<_>>(), kept_shape.clone());
                where_nan(&picked, &counts.iter().map(|c| *c > 0).collect::<Vec<_>>())
            }
        }
    });
    (result, Plane::Values(counts))
}

pub fn as_int<T: Elem>(value: T) -> i128 { value.int() }

/// numpy's integer sum dtype: int64 (signed, bool) or uint64 (unsigned).
fn int_sums(sums: Vec<i128>, dtype: DType, shape: Vec<usize>) -> Array {
    if dtype.kind() == 'u' { Array::from_vec(sums.into_iter().map(|v| v as u64).collect::<Vec<u64>>(), shape) }
    else { Array::from_vec(sums.into_iter().map(|v| v as i64).collect::<Vec<i64>>(), shape) }
}

/// `np.where(keep, values, np.nan)` with NEP 50 promotion (float32 stays).
pub fn where_nan(values: &Array, keep: &[bool]) -> Array {
    match values.dtype() {
        DType::F32 => Array::from_vec(values.slice::<f32>().iter().zip(keep).map(|(v, k)| if *k { *v } else { f32::NAN }).collect(), values.shape().to_vec()),
        DType::F16 => Array::from_vec(values.slice::<zlc_core::F16>().iter().zip(keep).map(|(v, k)| if *k { *v } else { zlc_core::F16::from_bits(0x7e00) }).collect(), values.shape().to_vec()),
        _ => Array::from_vec(values.to_f64().into_iter().zip(keep).map(|(v, k)| if *k { v } else { f64::NAN }).collect(), values.shape().to_vec()),
    }
}

/// `masked_centred_moment_sums` over `(outer, keep, inner)`, any dtype.
pub fn centred_sums(plane: &Array, offsets: &[f64], marks: Option<&[bool]>, outer: usize, keep: usize, inner: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    // numba types no float16 array: the compiled kernel refuses, so does Python.
    if plane.dtype() == DType::F16 { return Err(super::ProjectionError::new(ErrorKind::NotImplemented, "float16")); }
    Ok(    with_dtype!(plane.dtype(), T => kernels::centred_moment_sums(plane.slice::<T>(), offsets, marks, outer, keep, inner)))
}

pub struct Dense {
    pub domains: Vec<Arc<UsedDomain>>,
    /// Kept-axes plane in the refs' order; native dtype where numpy keeps it.
    pub values: Array,
    pub counts: Plane<i64>,
    pub sem: Option<Vec<f64>>,
    pub valid: Plane<bool>,
}

impl DataView {
    /// `_dense_tensor_projection(refs, aggregation, uncertainty)`.
    pub fn dense_tensor_projection(&self, refs: &[AxisRef], aggregation: Reduction, uncertainty: bool) -> Result<Option<Dense>> {
        if refs.is_empty() { return Ok(None); }
        let aggregation = aggregation.statistic();
        let (domains, mut codes, mut dimensions) = match self.axis_projection(refs) {
            Ok(found) => found,
            Err(error) if error.kind == ErrorKind::AxisResolution => return Ok(None),
            Err(error) => return Err(error),
        };
        if !self.packs_whole() { return Ok(None); }
        let packed = self.packed(uncertainty)?;
        let mut shape = self.schema.physical_shape.clone();
        if let Some(layout) = &self.layout {
            if let Some(inner) = layout.inner_count {
                if domains.iter().zip(&dimensions).any(|(domain, dimension)| *dimension == 1 && domain.size() != shape[1]) {
                    let shots = layout.shot_count();
                    let mut split_codes = Vec::new();
                    let mut split_dimensions = Vec::new();
                    for ((axis_codes, dimension), axis) in codes.iter().zip(&dimensions).zip(refs) {
                        if *dimension != 1 {
                            split_codes.push(axis_codes.clone());
                            split_dimensions.push(dimension + usize::from(*dimension > 1));
                            continue;
                        }
                        let contract = self.resolve(axis)?.contract.clone();
                        let mapping = self.schema.domain_of(&contract).code_mapping(contract.index);
                        let (base, repeat, tile) = (mapping.base.len(), mapping.inner, mapping.outer);
                        if base * repeat == inner && tile == shots {
                            split_codes.push(axis_codes[..inner].into());
                            split_dimensions.push(2);
                        } else if repeat == inner && base * tile == shots {
                            split_codes.push(axis_codes.iter().step_by(inner).copied().collect());
                            split_dimensions.push(1);
                        } else {
                            return Ok(None);
                        }
                    }
                    shape = [vec![shape[0], shots, inner], shape[2..].to_vec()].concat();
                    codes = split_codes;
                    dimensions = split_dimensions;
                }
            }
        }
        if (0..dimensions.len()).any(|i| dimensions[..i].contains(&dimensions[i])) { return Ok(None); }
        let mut orders: Vec<Option<Vec<usize>>> = Vec::new();
        let mut kept_sizes = Vec::new();
        for ((domain, axis_codes), dimension) in domains.iter().zip(&codes).zip(&dimensions) {
            let size = shape[*dimension];
            if domain.size() != size || axis_codes.len() != size || axis_codes.iter().any(|c| *c < 0 || *c as usize >= size)
                || !domain.canonical.all_finite() {
                return Ok(None);
            }
            kept_sizes.push(size);
            if axis_codes.iter().enumerate().all(|(i, c)| *c as usize == i) { orders.push(None); continue; }
            let mut seen = vec![0usize; size];
            for code in axis_codes.iter() { seen[*code as usize] += 1; }
            if seen.iter().any(|count| *count != 1) { return Ok(None); }
            let mut order = vec![0usize; size];
            for (natural, code) in axis_codes.iter().enumerate() { order[*code as usize] = natural; }
            orders.push(Some(order));
        }
        let kept = product(&kept_sizes);
        let total = packed.values.len();
        let pool = total / kept.max(1);
        let all_valid = packed.valid.all_true();
        let marks: Option<super::nd::Bools> = if all_valid { None } else { Some(packed.valid.plane(total)) };
        let source_shape = shape.clone();
        let moved_layout = super::strided::Moved::new(&source_shape, &dimensions);
        let identity_order = dimensions.iter().enumerate().all(|(i, d)| *d == source_shape.len() - dimensions.len() + i);
        // `moved[row]` as numpy writes it, only where a layout is read whole.
        let gather_row = |array: &Array, row: usize| -> Array {
            if identity_order { array.reshape(vec![total]).rows(row * kept, (row + 1) * kept).reshape(kept_sizes.clone()) }
            else { array.reshape(vec![total]).take_flat(&moved_layout.row_positions(row), kept_sizes.clone()) }
        };
        let laid_out = |array: &Array| -> Array {
            let moved = array.reshape(shape.clone()).move_to_back(&dimensions);
            moved.reshape([vec![moved.len() / kept.max(1)], kept_sizes.clone()].concat())
        };
        let (mut values, counts, valid_plane, identity) = if pool == 1 {
            let values = gather_row(&packed.values, 0);
            let valid: Plane<bool> = match &marks {
                None => Plane::Uniform(true, kept),
                Some(m) => Plane::Values(moved_layout.row_positions(0).iter().map(|p| m[*p]).collect()),
            };
            let counts = if all_valid { Plane::Uniform(1, kept) } else { valid.map(i64::from) };
            (values, counts, valid, true)
        } else {
            let (values, counts) = if kept >= 2 {
                strided_reduce(&packed.values, &moved_layout, marks.as_deref(), aggregation, pool, kept_sizes.clone(), &gather_row)
            } else {
                let moved = laid_out(&packed.values);
                let usable = match &marks { None => Mask::Uniform(true), Some(m) => Mask::Plane(laid_out(&Array::from_vec(m.to_vec(), packed.values.shape().to_vec())).bools().into()) };
                masked_leading_reduce(&moved, &usable, aggregation)
            };
            let valid = if counts.uniform().is_some_and(|c| c > 0) && values.all_finite() { Plane::Uniform(true, kept) } else {
                let finite = values.finite_mask();
                Plane::Values((0..kept).map(|i| counts.get(i) > 0 && finite[i]).collect())
            };
            (values, counts, valid, false)
        };
        let (mut counts, mut valid_plane) = (counts, valid_plane);
        let mut sem = None;
        if uncertainty {
            if aggregation != Reduction::Mean { return Ok(None); }
            if identity && packed.sigma.is_none() {
                sem = Some(vec![f64::NAN; kept]);
            } else {
                let counts_vec = counts.to_vec();
                let counts_ref = &counts_vec;
                let centred = |plane: &Array, offsets: &[f64]| -> Result<(Vec<f64>, Vec<f64>)> {
                    // numba types no float16 array: the compiled kernel refuses, so does Python.
                    if plane.dtype() == DType::F16 { return Err(super::ProjectionError::new(ErrorKind::NotImplemented, "float16")); }
                    let (first, second) = with_dtype!(plane.dtype(), T => moved_layout.centred(plane.slice::<T>(), offsets, marks.as_deref()));
                    let scale = |moment: Vec<f64>| moment.into_iter().zip(counts_ref).map(|(m, c)| if *c > 0 { m / *c as f64 } else { f64::NAN }).collect();
                    Ok((scale(first), scale(second)))
                };
                let means = values.to_f64();
                sem = Some(sem_of_mean(&means, &counts_vec, &packed.values, packed.sigma.as_ref(), &centred)?);
            }
        }
        for (axis, order) in orders.iter().enumerate() {
            let Some(order) = order else { continue };
            values = values.reshape(kept_sizes.clone()).take_axis(order, axis);
            if let Plane::Values(plane) = &counts { counts = Plane::Values(Array::from_vec(plane.clone(), kept_sizes.clone()).take_axis(order, axis).slice::<i64>().to_vec()); }
            if let Plane::Values(plane) = &valid_plane { valid_plane = Plane::Values(Array::from_vec(plane.clone(), kept_sizes.clone()).take_axis(order, axis).bools().to_vec()); }
            if let Some(errors) = &sem { sem = Some(Array::from_vec(errors.clone(), kept_sizes.clone()).take_axis(order, axis).slice::<f64>().to_vec()); }
        }
        Ok(Some(Dense { domains, values: values.reshape(kept_sizes), counts, sem, valid: valid_plane }))
    }
}

// --------------------------------------------------------- code buckets
/// `_aggregate_by_codes`, the numpy reference (used by the lattice fold);
/// `rule` is the tie rule of the dtype `values` were widened from.
pub fn aggregate_by_codes(values: &[f64], usable: &[bool], codes: &[i64], buckets: usize, aggregation: Reduction,
                          rule: crate::numeric::Extremes) -> (Vec<f64>, Vec<i64>) {
    if buckets == codes.len() && codes.iter().enumerate().all(|(i, c)| *c == i as i64) {
        let output = values.iter().zip(usable).map(|(v, u)| if *u { *v } else { f64::NAN }).collect();
        return (output, usable.iter().map(|u| i64::from(*u)).collect());
    }
    let mut output = vec![f64::NAN; buckets];
    let mut counts = vec![0i64; buckets];
    let positions: Vec<usize> = (0..codes.len()).filter(|p| usable[*p] && codes[*p] >= 0).collect();
    for p in &positions { counts[codes[*p] as usize] += 1; }
    match aggregation {
        Reduction::Sum | Reduction::Mean => {
            let mut sums = vec![0.0_f64; buckets];
            for p in &positions { sums[codes[*p] as usize] += values[*p]; }
            for bucket in 0..buckets {
                if counts[bucket] > 0 { output[bucket] = if aggregation == Reduction::Mean { sums[bucket] / counts[bucket] as f64 } else { sums[bucket] }; }
            }
        }
        Reduction::First => { for p in positions.iter().rev() { output[codes[*p] as usize] = values[*p]; } }
        _ => {
            // The members stably sorted by bucket, then `ufunc.reduceat`:
            // each bucket reduces its members in their original order.
            let mut start = vec![0usize; buckets + 1];
            for bucket in 0..buckets { start[bucket + 1] = start[bucket] + counts[bucket] as usize; }
            let mut fill = start.clone();
            let mut ordered = vec![0.0_f64; positions.len()];
            for p in &positions { let bucket = codes[*p] as usize; ordered[fill[bucket]] = values[*p]; fill[bucket] += 1; }
            let minimum = aggregation == Reduction::Min;
            for bucket in 0..buckets {
                let (low, high) = (start[bucket], start[bucket + 1]);
                if high > low { output[bucket] = crate::numeric::reduce_contiguous(ordered[low], &ordered[low + 1..high], rule, minimum); }
            }
        }
    }
    (output, counts)
}

/// `_axis_aggregate`: the compiled exact-order axis-code reduction.
pub fn axis_aggregate(values: &Array, valid: &Mask, codes: &[Vec<i64>], dimensions: &[usize], domain_sizes: &[usize],
                      aggregation: Reduction, offsets: Option<&[f64]>) -> kernels::Aggregated {
    let shape = values.shape();
    let marks = if valid.all_true() { None } else { Some(valid.plane(values.len())) };
    let operation = if offsets.is_some() { kernels::CENTRED } else { axis_operation(aggregation) };
    let buckets = product(domain_sizes);
    if values.dtype() == DType::F16 && offsets.is_none() && matches!(aggregation.statistic(), Reduction::Min | Reduction::Max) {
        // numba types no float16 array: `_axis_aggregate` takes numpy's
        // reference, whose `reduceat` keeps the first of equal samples.
        let total = values.len();
        let mut bucket_of = vec![-1i64; total];
        let mut index = vec![0usize; shape.len()];
        for slot in bucket_of.iter_mut() {
            let mut bucket = 0i64;
            for (axis, dimension) in dimensions.iter().enumerate() {
                let code = codes[axis][index[*dimension]];
                if code < 0 { bucket = -1; break; }
                bucket = bucket * domain_sizes[axis] as i64 + code;
            }
            *slot = bucket;
            for k in (0..shape.len()).rev() { index[k] += 1; if index[k] < shape[k] { break; } index[k] = 0; }
        }
        let usable: Vec<bool> = (0..total).map(|p| bucket_of[p] >= 0 && marks.as_ref().is_none_or(|m| m[p])).collect();
        let (out, counts) = aggregate_by_codes(&values.to_f64(), &usable, &bucket_of, buckets, aggregation.statistic(), DType::F16.extremes());
        let mut presence = vec![false; buckets];
        for bucket in &bucket_of { if *bucket >= 0 { presence[*bucket as usize] = true; } }
        return kernels::Aggregated { out, second: vec![], counts, presence };
    }
    let mut state = kernels::Aggregated::start(buckets, operation);
    axis_walk(&mut state, values, marks.as_deref(), codes, dimensions, domain_sizes, operation, offsets.unwrap_or(&[]));
    state.finish(operation)
}

/// The kernel operation `_axis_aggregate` reduces with.
fn axis_operation(aggregation: Reduction) -> u8 {
    match aggregation.statistic() {
        Reduction::Mean => kernels::REDUCE_MEAN, Reduction::Sum => kernels::REDUCE_SUM, Reduction::Min => kernels::REDUCE_MIN,
        Reduction::Max => kernels::REDUCE_MAX, _ => kernels::REDUCE_FIRST,
    }
}

/// The compiled walk of `values` into `state` (parallel where a dimension separates the work).
#[allow(clippy::too_many_arguments)]
fn axis_walk(state: &mut kernels::Aggregated, values: &Array, marks: Option<&[bool]>, codes: &[Vec<i64>], dimensions: &[usize],
             domain_sizes: &[usize], operation: u8, offsets: &[f64]) {
    let shape = values.shape();
    let strides: Vec<usize> = dimensions.iter().map(|d| product(&shape[d + 1..])).collect();
    let sizes: Vec<usize> = codes.iter().map(Vec::len).collect();
    let axes = AxisCodes { codes, sizes: &sizes, domain_sizes, strides: &strides };
    with_dtype!(values.dtype(), T => {
        let slice = values.slice::<T>();
        if !kernels::aggregate_axis_codes_parallel(state, slice, marks, shape, codes, dimensions, domain_sizes, operation, offsets) {
            kernels::aggregate_axis_codes(state, slice, marks, &axes, operation, offsets);
        }
    })
}

pub struct Segmented { pub values: Vec<f64>, pub counts: Vec<i64>, pub presence: Vec<bool>, pub sem: Option<Vec<f64>> }

/// The walked, unfinished state of `_segmented_axes` over the rows of a
/// holey block, by segment identity (the address of its immutable planes,
/// which the carry keeps alive, and its place). A later revision of the same
/// run that only appended segments, every row of them after the walked ones,
/// walks just those rows on from it: each bucket then folds the same samples
/// in the same order as the walk of the whole scratch, so the reduction is
/// the same bits (a mean is still its sum divided once by its count).
pub struct AccumulationCarry {
    query: (Vec<usize>, Vec<usize>, u8),
    codes: Vec<Arc<[i64]>>,
    identities: Vec<(usize, usize, [usize; 2])>,
    /// Keeps the identified planes alive.
    #[allow(dead_code)]
    segments: super::model::Segments,
    /// The last walked row, `repeat * points + point`.
    last: usize,
    state: kernels::Aggregated,
    /// The buckets the codes reach (they decide it alone).
    presence: Vec<bool>,
}

impl DataView {
    /// `_segmented_axes`: the ordinary axis reduction applied to the scratch.
    pub fn segmented_axes(&self, axis_codes: &[Arc<[i64]>], dimensions: &[usize], sizes: &[usize], aggregation: Reduction,
                          uncertainty: bool) -> Result<Segmented> {
        if let Some((values, counts, presence)) = self.accumulated(axis_codes, dimensions, sizes, aggregation, uncertainty)? {
            return Ok(Segmented { values, counts, presence, sem: None });
        }
        let packed = self.packed(uncertainty)?;
        let (codes, kept) = packed_codes(&packed, axis_codes, dimensions);
        let reduced = axis_aggregate(&packed.values, &packed.valid, &codes, &kept, sizes, aggregation, None);
        let presence = self.reachable(axis_codes, dimensions, sizes);
        let sem = if uncertainty {
            let centred = |plane: &Array, offsets: &[f64]| -> Result<(Vec<f64>, Vec<f64>)> {
                let result = axis_aggregate(plane, &packed.valid, &codes, &kept, sizes, Reduction::Sum, Some(offsets));
                Ok((result.out, result.second))
            };
            Some(sem_of_mean(&reduced.out, &reduced.counts, &packed.values, packed.sigma.as_ref(), &centred)?)
        } else { None };
        Ok(Segmented { values: reduced.out, counts: reduced.counts, presence, sem })
    }

    /// The buckets the axis codes reach, whatever the data holds.
    fn reachable(&self, axis_codes: &[Arc<[i64]>], dimensions: &[usize], sizes: &[usize]) -> Vec<bool> {
        let shape = &self.schema.physical_shape;
        let mut reachable = vec![0usize];
        let mut seen_dimensions: Vec<usize> = dimensions.to_vec();
        seen_dimensions.sort_unstable();
        seen_dimensions.dedup();
        for dimension in seen_dimensions {
            let mut contribution = vec![0i64; shape[dimension]];
            let mut admitted = vec![true; shape[dimension]];
            for (index, (axis, carrier)) in axis_codes.iter().zip(dimensions).enumerate() {
                if *carrier != dimension { continue; }
                let stride: usize = sizes[index + 1..].iter().product();
                for (row, code) in axis.iter().enumerate() {
                    admitted[row] &= *code >= 0;
                    contribution[row] += code * stride as i64;
                }
            }
            let mut used: Vec<i64> = contribution.iter().zip(&admitted).filter(|(_, a)| **a).map(|(c, _)| *c).collect();
            used.sort_unstable();
            used.dedup();
            reachable = reachable.iter().flat_map(|r| used.iter().map(move |u| (*r as i64 + u) as usize)).collect();
        }
        let mut presence = vec![false; product(sizes)];
        for r in reachable { presence[r] = true; }
        presence
    }

    /// `_segmented_axes` of a holey block walked on from the previous
    /// revision's [`AccumulationCarry`] (the whole walk when there is none to
    /// take up), carried on to the next: its values, counts and presence.
    /// None where it is not carried: an uncertainty (centred on the means the
    /// new rows move), a whole tensor, numpy's float16 extremes, planes
    /// without an address.
    #[allow(clippy::type_complexity)]
    pub(crate) fn accumulated(&self, axis_codes: &[Arc<[i64]>], dimensions: &[usize], sizes: &[usize], aggregation: Reduction,
                   uncertainty: bool) -> Result<Option<(Vec<f64>, Vec<i64>, Vec<bool>)>> {
        let operation = axis_operation(aggregation);
        let Block::Segments(segments) = &self.snapshot.block else { return Ok(None) };
        if uncertainty || self.packs_whole()
            || (self.schema.value.dtype == DType::F16 && matches!(operation, kernels::REDUCE_MIN | kernels::REDUCE_MAX)) {
            return Ok(None);
        }
        let identity = |s: &super::model::PlacedRef<'_>| {
            let planes = s.values.bytes();
            (!planes.is_empty()).then(|| (planes.as_ptr() as usize, planes.len(), s.origin))
        };
        let Some(identities) = segments.iter().map(|s| identity(&s)).collect::<Option<Vec<_>>>() else { return Ok(None) };
        let points = self.schema.physical_shape[1];
        let first_row = |s: &super::model::PlacedRef<'_>| s.origin[0] * points + s.origin[1];
        let last_row = |s: &super::model::PlacedRef<'_>| (s.origin[0] + s.extent[0] - 1) * points + s.origin[1] + s.extent[1] - 1;
        let query = (dimensions.to_vec(), sizes.to_vec(), operation);
        let carried = self.caches.borrow_mut().accumulation.take().filter(|carry| carry.query == query
            && identities.starts_with(&carry.identities)
            && carry.codes.len() == axis_codes.len() && carry.codes.iter().zip(axis_codes).all(|(old, new)| new.starts_with(old))
            && segments.iter().skip(carry.identities.len()).all(|s| first_row(&s) > carry.last));
        // The codes decide the reachable buckets: unchanged codes, unchanged presence.
        let same_codes = carried.as_ref().is_some_and(|carry| carry.codes.iter().zip(axis_codes).all(|(old, new)| old.len() == new.len()));
        let (mut state, presence, walked) = match carried {
            Some(carry) if carry.identities.len() == identities.len() => (carry.state, same_codes.then_some(carry.presence), None),
            Some(carry) => {
                let brought: Vec<usize> = (carry.identities.len()..segments.len()).collect();
                (carry.state, same_codes.then_some(carry.presence), Some(Arc::new(pack(&self.snapshot, Some(&brought), false, Some(&self.pools))?)))
            }
            None => (kernels::Aggregated::start(product(sizes), operation), None, Some(self.packed(false)?)),
        };
        if let Some(packed) = walked {
            let (codes, kept) = packed_codes(&packed, axis_codes, dimensions);
            let marks = if packed.valid.all_true() { None } else { Some(packed.valid.plane(packed.values.len())) };
            axis_walk(&mut state, &packed.values, marks.as_deref(), &codes, &kept, sizes, operation, &[]);
        }
        let (values, counts) = state.finished(operation);
        let presence = presence.unwrap_or_else(|| self.reachable(axis_codes, dimensions, sizes));
        let last = segments.iter().map(|s| last_row(&s)).max().unwrap_or(0);
        self.caches.borrow_mut().accumulation = Some(AccumulationCarry { query, codes: axis_codes.to_vec(), identities,
            segments: segments.clone(), last, state, presence: presence.clone() });
        Ok(Some((values, counts, presence)))
    }
}

/// Axis codes on the packed scratch: rows of a holey block index codes.
pub fn packed_codes(packed: &Packed, axis_codes: &[Arc<[i64]>], dimensions: &[usize]) -> (Vec<Vec<i64>>, Vec<usize>) {
    match &packed.rows {
        None => (axis_codes.iter().map(|c| c.to_vec()).collect(), dimensions.to_vec()),
        Some((repeat, point)) => (
            axis_codes.iter().zip(dimensions).map(|(axis, dimension)| match dimension {
                0 => repeat.iter().map(|r| axis[*r as usize]).collect(),
                1 => point.iter().map(|p| axis[*p as usize]).collect(),
                _ => axis.to_vec(),
            }).collect(),
            dimensions.iter().map(|d| if *d < 2 { 0 } else { d - 1 }).collect(),
        ),
    }
}

/// The `(pool >= 2, kept >= 2)` dense reduction read in place (see `strided.rs`):
/// the same per-element order as numpy's reduction of the moved copy.
fn strided_reduce(values: &Array, moved: &super::strided::Moved, marks: Option<&[bool]>, aggregation: Reduction, pool: usize,
                  kept_shape: Vec<usize>, gather_row: &dyn Fn(&Array, usize) -> Array) -> (Array, Plane<i64>) {
    let kept = product(&kept_shape);
    let float = values.kind() == 'f';
    let counted = |c: Vec<i64>| -> Plane<i64> { if marks.is_none() { Plane::Uniform(pool as i64, kept) } else { Plane::Values(c) } };
    let nan_empty = |v: Vec<f64>, c: &Plane<i64>| -> Vec<f64> { v.into_iter().enumerate().map(|(k, x)| if c.get(k) > 0 { x } else { f64::NAN }).collect() };
    with_dtype!(values.dtype(), T => {
        let slice = values.slice::<T>();
        match aggregation {
            Reduction::Mean => {
                let (sums, counts) = moved.sum(slice, marks);
                let counts = counted(counts);
                // numpy divides an empty bucket's 0 by 0 (x86's negative NaN);
                // the fused kernel that takes a large float64 tensor with holes
                // writes np.nan.
                let fused = marks.is_some() && values.dtype() == DType::F64 && values.len() >= kernels::MASKED_LEADING_MIN_SAMPLES;
                (Array::from_vec((0..kept).map(|k| if fused && counts.get(k) == 0 { f64::NAN } else { sums[k] / counts.get(k) as f64 }).collect(), kept_shape), counts)
            }
            Reduction::Sum if float => {
                let (sums, counts) = moved.sum(slice, marks);
                let counts = counted(counts);
                let sums = if marks.is_some() { nan_empty(sums, &counts) } else { sums };
                (Array::from_vec(sums, kept_shape), counts)
            }
            Reduction::Sum => {
                let sums = moved.int_sum(slice, marks);
                let unsigned = values.dtype().kind() == 'u';
                if marks.is_none() {
                    let array = if unsigned { Array::from_vec(sums.into_iter().map(|v| v as u64).collect::<Vec<u64>>(), kept_shape) }
                                else { Array::from_vec(sums.into_iter().map(|v| v as i64).collect::<Vec<i64>>(), kept_shape) };
                    (array, Plane::Uniform(pool as i64, kept))
                } else {
                    let counts = counted(moved.sum(slice, marks).1);
                    let as_float: Vec<f64> = sums.into_iter().map(|v| if unsigned { v as u64 as f64 } else { v as i64 as f64 }).collect();
                    (Array::from_vec(nan_empty(as_float, &counts), kept_shape), counts)
                }
            }
            Reduction::Min | Reduction::Max => {
                // `_masked_leading_reduce` on the moved copy: numpy on the
                // native dtype without holes, the fused kernel's `<=` for a
                // large float64 tensor with holes, numpy on float64 otherwise.
                let minimum = aggregation == Reduction::Min;
                let rule = values.dtype().extremes();
                let fused = values.dtype() == DType::F64 && values.len() >= kernels::MASKED_LEADING_MIN_SAMPLES;
                let extremes = match marks {
                    None => moved.extreme(slice, None, minimum, |a, b| rule.step(a, b, minimum)),
                    Some(_) if fused => moved.extreme(slice, marks, minimum, |a, b| if (minimum && b <= a) || (!minimum && b >= a) { b } else { a }),
                    Some(_) => moved.extreme(slice, marks, minimum, |a, b| DType::F64.extremes().step(a, b, minimum)),
                };
                if marks.is_none() { (Array::from_vec(extremes, kept_shape), Plane::Uniform(pool as i64, kept)) } else {
                    let counts = counted(moved.sum(slice, marks).1);
                    (Array::from_vec(nan_empty(extremes, &counts), kept_shape), counts)
                }
            }
            _ => match marks {
                None => (gather_row(values, 0), Plane::Uniform(pool as i64, kept)),
                Some(m) => {
                    let counts = counted(moved.sum(slice, Some(m)).1);
                    let picked = values.reshape(vec![values.len()]).take_flat(&moved.first(m), kept_shape);
                    let keep: Vec<bool> = (0..kept).map(|k| counts.get(k) > 0).collect();
                    (where_nan(&picked, &keep), counts)
                }
            },
        }
    })
}
