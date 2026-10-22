//! `_factored_planes`: moments by tensor axes, then a tiny fold by the
//! combined row key -- the one lattice computation behind curves, scan
//! heatmaps and facets over modest factored axes.

use std::sync::Arc;

use super::nd::{Array, DType, product};
use super::reduce::{aggregate_by_codes, centred_sums, sem_of_mean};
use super::view::{DataView, UsedDomain};
use super::{ErrorKind, Result};
use super::output::QuantityArray;
use crate::kinds::{AxisRef, Reduction};
use crate::numeric::{add_reduce, extreme_reduce, int_add_reduce};
use crate::with_dtype;

pub struct FactoredPlanes {
    pub x_quantity: QuantityArray,
    pub x_labels: Option<Vec<String>>,
    pub row_domains: Vec<Arc<UsedDomain>>,
    pub row_sizes: Vec<usize>,
    pub row_presence: Vec<bool>,
    pub group_domains: Vec<Arc<UsedDomain>>,
    pub group_sizes: Vec<usize>,
    /// `(row_buckets, combos)` row-major.
    pub y_plane: Vec<f64>,
    pub counts_plane: Vec<i64>,
    pub sem_plane: Option<Vec<f64>>,
    pub combos: usize,
}

/// `np.transpose(plane, permutation)` then `np.take(.., order, axis=1+i)`
/// per group, flattened `(rows, combos)` -- for a plane over `kept_shape`.
fn code_ordered<T: Copy>(plane: &[T], kept_shape: &[usize], permutation: &[usize], orders: &[Option<Vec<usize>>]) -> Vec<T> {
    let shape: Vec<usize> = permutation.iter().map(|axis| kept_shape[*axis]).collect();
    let mut strides = vec![1usize; kept_shape.len()];
    for axis in (0..kept_shape.len().saturating_sub(1)).rev() { strides[axis] = strides[axis + 1] * kept_shape[axis + 1]; }
    let total = product(&shape);
    let mut out = Vec::with_capacity(total);
    let mut index = vec![0usize; shape.len()];
    for _ in 0..total {
        let mut flat = 0usize;
        for (position, axis) in permutation.iter().enumerate() {
            let mut i = index[position];
            if position >= 1 { if let Some(order) = &orders[position - 1] { i = order[i]; } }
            flat += i * strides[*axis];
        }
        out.push(plane[flat]);
        for axis in (0..shape.len()).rev() {
            index[axis] += 1;
            if index[axis] < shape[axis] { break; }
            index[axis] = 0;
        }
    }
    out
}

/// `_axis_coordinate_labels(resolved, canonical)`.
pub fn axis_coordinate_labels(view: &DataView, axis: &AxisRef, canonical: &Array) -> Result<Option<Vec<String>>> {
    let resolved = view.resolve(axis)?;
    let Some(declared) = resolved.coordinate_labels() else { return Ok(None) };
    let domain = &resolved.domain_canonical;
    let by: Vec<(super::nd::Scalar, &str)> = (0..domain.len()).map(|i| (domain.scalar(i), declared[i].as_ref())).collect();
    Ok(Some((0..canonical.len()).map(|i| {
        let value = canonical.scalar(i);
        by.iter().find(|(key, _)| *key == value).map(|(_, label)| (*label).to_owned())
            .unwrap_or_else(|| crate::pyfmt::format_g(canonical.get_f64(i), 6))
    }).collect()))
}

impl DataView {
    pub fn x_quantity(&self, axis: &AxisRef, domain: &UsedDomain) -> Result<QuantityArray> {
        let resolved = self.resolve(axis)?;
        Ok(QuantityArray::with_display(domain.canonical.clone(), domain.display.clone(), resolved.canonical_unit.clone(),
            resolved.display_unit.clone(), resolved.label.to_string()))
    }

    /// `_factored_planes(row_refs, groups, aggregation, uncertainty)`.
    pub fn factored_planes(&self, row_refs: &[AxisRef], groups: &[AxisRef], aggregation: Reduction, uncertainty: bool) -> Result<Option<FactoredPlanes>> {
        let aggregation = aggregation.statistic();
        if !matches!(aggregation, Reduction::Mean | Reduction::Sum | Reduction::Min | Reduction::Max) { return Ok(None); }
        if !self.packs_whole() || row_refs.is_empty() { return Ok(None); }
        let packed = self.packed(uncertainty)?;
        let resolve_all = |refs: &[AxisRef]| -> Result<Option<Vec<usize>>> {
            let mut dims = Vec::new();
            for axis in refs {
                match self.resolve(axis) {
                    Ok(resolved) => dims.push(resolved.dimension()),
                    Err(error) if error.kind == ErrorKind::AxisResolution => return Ok(None),
                    Err(error) => return Err(error),
                }
            }
            Ok(Some(dims))
        };
        let Some(row_dims) = resolve_all(row_refs)? else { return Ok(None) };
        let Some(group_dims) = resolve_all(groups)? else { return Ok(None) };
        let x = row_refs.last().unwrap();
        let shape = packed.values.shape().to_vec();
        let row_dimension = *row_dims.last().unwrap();
        if row_dims.iter().any(|d| *d != row_dimension) { return Ok(None); }
        let mut kept_dims: Vec<usize> = Vec::new();
        for dimension in &group_dims {
            if *dimension == row_dimension || kept_dims.contains(dimension) { return Ok(None); }
            kept_dims.push(*dimension);
        }
        let rows = shape[row_dimension];
        let row_domains = row_refs.iter().map(|axis| self.domain(axis)).collect::<Result<Vec<_>>>()?;
        let row_sizes: Vec<usize> = row_domains.iter().map(|d| d.size()).collect();
        if row_sizes.contains(&0) { return Ok(None); }
        let mut combined = vec![0i64; rows];
        let mut row_ok = vec![true; rows];
        for (domain, size) in row_domains.iter().zip(&row_sizes) {
            for row in 0..rows {
                let code = domain.codes[row];
                row_ok[row] &= code >= 0;
                combined[row] = combined[row] * *size as i64 + code.max(0);
            }
        }
        let row_buckets = product(&row_sizes);
        let mut row_presence = vec![false; row_buckets];
        for row in 0..rows { if row_ok[row] { row_presence[combined[row] as usize] = true; } }
        let mut group_domains = Vec::new();
        let mut group_orders = Vec::new();
        for (axis, dimension) in groups.iter().zip(&kept_dims) {
            let domain = self.domain(axis)?;
            if domain.size() != shape[*dimension] || domain.codes.iter().any(|c| *c < 0) { return Ok(None); }
            let natural = domain.codes.iter().enumerate().all(|(i, c)| *c == i as i64);
            group_orders.push(if natural { None } else {
                let mut order = vec![0usize; domain.codes.len()];
                for (i, code) in domain.codes.iter().enumerate() { order[*code as usize] = i; }
                Some(order)
            });
            group_domains.push(domain);
        }
        let reduce_axes: Vec<usize> = (0..shape.len()).filter(|a| *a != row_dimension && !kept_dims.contains(a)).collect();
        let mut remaining: Vec<usize> = std::iter::once(row_dimension).chain(kept_dims.iter().copied()).collect();
        remaining.sort_unstable();
        let kept_shape: Vec<usize> = remaining.iter().map(|a| shape[*a]).collect();
        let all_valid = packed.valid.all_true();
        let marks = if all_valid { None } else { Some(packed.valid.plane(packed.values.len())) };
        let kept_count = product(&kept_shape);
        let counts_pg: Vec<i64> = if all_valid {
            vec![reduce_axes.iter().map(|a| shape[*a]).product::<usize>() as i64; kept_count]
        } else {
            let marks = marks.as_deref().unwrap();
            int_add_reduce(marks, &shape, &reduce_axes, Some(marks), |_| 1).into_iter().map(|v| v as i64).collect()
        };
        let buffered = packed.values.dtype() != DType::F64;
        let moments_pg: Vec<f64> = with_dtype!(packed.values.dtype(), T => {
            let slice = packed.values.slice::<T>();
            match aggregation {
                Reduction::Mean | Reduction::Sum => if reduce_axes.is_empty() { slice.iter().map(|v| super::nd::Elem::f64(*v)).collect() }
                    else { add_reduce(slice, &shape, &reduce_axes, marks.as_deref(), buffered) },
                _ => {
                    // `ufunc(values, axis=reduce_axes[, where=usable, initial=limit])`.
                    let minimum = aggregation == Reduction::Min;
                    let initial = marks.as_ref().map(|_| if minimum { f64::INFINITY } else { f64::NEG_INFINITY });
                    extreme_reduce(slice, &shape, &reduce_axes, marks.as_deref(), initial, packed.values.dtype().extremes(), minimum)
                }
            }
        });
        let permutation: Vec<usize> = std::iter::once(remaining.iter().position(|a| *a == row_dimension).unwrap())
            .chain(kept_dims.iter().map(|d| remaining.iter().position(|a| a == d).unwrap())).collect();
        let group_sizes: Vec<usize> = kept_dims.iter().map(|d| shape[*d]).collect();
        let combos = product(&group_sizes);
        let counts_pg = code_ordered(&counts_pg, &kept_shape, &permutation, &group_orders);
        let moments_pg = code_ordered(&moments_pg, &kept_shape, &permutation, &group_orders);
        let fold_codes: Vec<i64> = (0..rows).flat_map(|row| {
            let ok = row_ok[row];
            let base = combined[row] * combos as i64;
            (0..combos).map(move |c| if ok { base + c as i64 } else { -1 })
        }).collect();
        let buckets = row_buckets * combos;
        let ones = vec![true; fold_codes.len()];
        let counts_as_f64: Vec<f64> = counts_pg.iter().map(|c| *c as f64).collect();
        let float64 = DType::F64.extremes();
        let (counts_fold, _) = aggregate_by_codes(&counts_as_f64, &ones, &fold_codes, buckets, Reduction::Sum, float64);
        let counts: Vec<i64> = counts_fold.iter().map(|c| if c.is_nan() { 0 } else { *c as i64 }).collect();
        let y_flat: Vec<f64> = match aggregation {
            Reduction::Mean | Reduction::Sum => {
                let (sums, _) = aggregate_by_codes(&moments_pg, &ones, &fold_codes, buckets, Reduction::Sum, float64);
                sums.iter().zip(&counts).map(|(s, c)| if *c > 0 { if aggregation == Reduction::Mean { s / *c as f64 } else { *s } } else { f64::NAN }).collect()
            }
            _ => {
                let usable: Vec<bool> = counts_pg.iter().map(|c| *c > 0).collect();
                let (y, _) = aggregate_by_codes(&moments_pg, &usable, &fold_codes, buckets, aggregation, packed.values.dtype().extremes());
                y.iter().zip(&counts).map(|(v, c)| if *c > 0 { *v } else { f64::NAN }).collect()
            }
        };
        let mut sem_flat = None;
        if uncertainty {
            let inverse: Vec<usize> = { let mut inv = vec![0; permutation.len()]; for (i, p) in permutation.iter().enumerate() { inv[*p] = i; } inv };
            // Undo code_ordered for a (rows, combos) plane: back to `remaining` order.
            let from_code_order = |plane: &[f64]| -> Vec<f64> {
                let permuted_shape: Vec<usize> = permutation.iter().map(|axis| kept_shape[*axis]).collect();
                let mut restored = vec![0.0; plane.len()];
                let mut index = vec![0usize; kept_shape.len()];
                let total = plane.len();
                for flat_permuted in 0..total {
                    let mut rest = flat_permuted;
                    for axis in (0..permuted_shape.len()).rev() { index[axis] = rest % permuted_shape[axis]; rest /= permuted_shape[axis]; }
                    let mut source = index.clone();
                    for (position, order) in group_orders.iter().enumerate() {
                        if let Some(order) = order { source[position + 1] = order[source[position + 1]]; }
                    }
                    let mut flat = 0usize;
                    for axis in 0..kept_shape.len() { flat = flat * kept_shape[axis] + source[inverse[axis]]; }
                    restored[flat] = plane[flat_permuted];
                }
                restored
            };
            let span: Vec<usize> = (remaining[0]..=*remaining.last().unwrap()).collect();
            let block = span.iter().all(|a| remaining.contains(a) || shape[*a] == 1);
            let outer: usize = shape[..remaining[0]].iter().product();
            let inner: usize = shape[remaining.last().unwrap() + 1..].iter().product();
            let fold_ref = &fold_codes;
            let counts_ref = &counts;
            let centred = |plane: &Array, offsets: &[f64]| -> Result<(Vec<f64>, Vec<f64>)> {
                let gathered: Vec<f64> = fold_ref.iter().map(|c| offsets[(*c).max(0) as usize]).collect();
                let per_group_offsets = from_code_order(&gathered);
                // `_centred_moment_sums` takes a contiguous mask or none at all.
                let per_group = if block && (marks.is_none() || packed.valid_contiguous) {
                    centred_sums(plane, &per_group_offsets, marks.as_deref(), outer, kept_count, inner)?
                } else {
                    numpy_centred(plane, &per_group_offsets, &shape, &remaining, &reduce_axes, marks.as_deref())
                };
                let fold = |moment: Vec<f64>| -> Vec<f64> {
                    let ordered = code_ordered(&moment, &kept_shape, &permutation, &group_orders);
                    let (total, _) = aggregate_by_codes(&ordered, &vec![true; ordered.len()], fold_ref, buckets, Reduction::Sum, float64);
                    total.iter().zip(counts_ref).map(|(m, c)| if *c > 0 { m / *c as f64 } else { f64::NAN }).collect()
                };
                Ok((fold(per_group.0), fold(per_group.1)))
            };
            sem_flat = Some(sem_of_mean(&y_flat, &counts, &packed.values, packed.sigma.as_ref(), &centred)?);
        }
        let x_domain = row_domains.last().unwrap().clone();
        let x_quantity = self.x_quantity(x, &x_domain)?;
        let x_labels = super::lattice::axis_coordinate_labels(self, x, &x_domain.canonical)?;
        Ok(Some(FactoredPlanes { x_quantity, x_labels, row_domains, row_sizes, row_presence, group_domains, group_sizes,
            y_plane: y_flat, counts_plane: counts, sem_plane: sem_flat, combos }))
    }
}

/// The numpy fallback of `_factored_planes`' centred moments (kept axes not one block).
fn numpy_centred(plane: &Array, offsets: &[f64], shape: &[usize], remaining: &[usize], reduce_axes: &[usize], marks: Option<&[bool]>) -> (Vec<f64>, Vec<f64>) {
    let values = plane.to_f64();
    let mut index = vec![0usize; shape.len()];
    let kept_shape: Vec<usize> = remaining.iter().map(|a| shape[*a]).collect();
    let mut delta = Vec::with_capacity(values.len());
    for (position, value) in values.iter().enumerate() {
        if position > 0 { for axis in (0..shape.len()).rev() { index[axis] += 1; if index[axis] < shape[axis] { break; } index[axis] = 0; } }
        let mut flat = 0usize;
        for (k, axis) in remaining.iter().enumerate() { flat = flat * kept_shape[k] + index[*axis]; }
        delta.push(value - offsets[flat]);
    }
    let first = add_reduce(&delta, shape, reduce_axes, marks, false);
    let squared: Vec<f64> = delta.iter().map(|d| d * d).collect();
    let second = add_reduce(&squared, shape, reduce_axes, marks, false);
    (first, second)
}
