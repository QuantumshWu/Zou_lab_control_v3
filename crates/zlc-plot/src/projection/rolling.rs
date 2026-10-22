//! Rolling histories (`DataView.rolling_history` and its owners) and the
//! rolling payload with trailing moment summaries (`_rolling_payload`,
//! `_trailing_trace`).

use std::sync::Arc;

use super::kernels;
use super::model::{Block, PRIMARY_INDEX, SHOT_TIME, Segments};
use super::nd::{Array, DType};
use super::output::{AxisValue, CurveData, CurveSeries, QuantityArray, RollingHistory, convert_array};
use super::reduce::{axis_aggregate, sem_of_mean};
use super::view::{DataView, acquired_repeat_count, pack};
use super::{ProjectionError, Result};
use crate::kinds::{AxisRef, Reduction};
use crate::numeric::{blas_dot_self, pairwise_widened};
use crate::units::resolve_unit;
use crate::with_dtype;

impl DataView {
    fn history(&self, values: Vec<f64>, counts: Vec<i64>, valid: Vec<bool>, sem: Option<Vec<f64>>, keys: Vec<Vec<AxisValue>>,
               source_indices: Option<Arc<[i64]>>, source_times: Option<Arc<[f64]>>) -> RollingHistory {
        let shape = vec![values.len() / keys.len().max(1), keys.len()];
        RollingHistory {
            revision: self.revision(), generation: self.generation().clone(),
            values: Array::from_vec(values, shape.clone()), valid: Array::from_vec(valid, shape.clone()),
            counts: Array::from_vec(counts, shape.clone()), group_keys: keys,
            source_indices: source_indices.map(|s| Array::vector(s.to_vec())), source_times: source_times.map(|s| Array::vector(s.to_vec())),
            sem: sem.map(|s| Array::from_vec(s, shape)),
        }
    }

    fn group_keys(&self, group: Option<&AxisRef>) -> Result<Vec<Vec<AxisValue>>> {
        Ok(match group { None => vec![vec![]], Some(group) => self.domain(group)?.values().iter().map(|v| vec![v.clone()]).collect() })
    }

    /// `_history_from_axes`.
    #[allow(clippy::too_many_arguments)]
    fn history_from_axes(&self, axis_codes: &[Arc<[i64]>], dimensions: &[usize], sizes: &[usize], keys: Vec<Vec<AxisValue>>,
                         aggregation: Reduction, uncertainty: bool, source_indices: Option<Arc<[i64]>>,
                         source_times: Option<Arc<[f64]>>) -> Result<RollingHistory> {
        let reduced = self.segmented_axes(axis_codes, dimensions, sizes, aggregation, uncertainty && aggregation == Reduction::Mean)?;
        let valid: Vec<bool> = reduced.counts.iter().zip(&reduced.values).map(|(c, v)| *c > 0 && v.is_finite()).collect();
        Ok(self.history(reduced.values, reduced.counts, valid, reduced.sem, keys, source_indices, source_times))
    }

    /// `rolling_history(x, group, aggregation, uncertainty)`.
    pub fn rolling_history(&self, x: Option<&AxisRef>, group: Option<&AxisRef>, aggregation: Reduction, uncertainty: bool) -> Result<RollingHistory> {
        self.validate_rolling(group, x)?;
        let aggregation = aggregation.statistic();
        if let Some(layout) = self.layout.clone() {
            if self.snapshot.is_segmented() {
                if let Some(history) = self.segment_history(group, aggregation, uncertainty)? { return Ok(history); }
            }
            let mut codes: Vec<Arc<[i64]>> = vec![layout.codes(&self.schema)];
            let mut dimensions = vec![1usize];
            let mut sizes = vec![layout.shot_count()];
            if let Some(group) = group {
                let (domains, group_codes, group_dimensions) = self.axis_projection(std::slice::from_ref(group))?;
                codes.extend(group_codes);
                dimensions.extend(group_dimensions);
                sizes.push(domains[0].size());
            }
            return self.history_from_axes(&codes, &dimensions, &sizes, self.group_keys(group)?, aggregation, uncertainty,
                Some(layout.cells.clone()), layout.times.clone());
        }
        let repeats = self.schema.repeat.size();
        if repeats <= 1 { return self.single_revision_history(group, aggregation, uncertainty); }
        let mut history = if self.snapshot.is_segmented() { self.segment_history(group, aggregation, uncertainty)? } else { None };
        if history.is_none() {
            let mut codes: Vec<Arc<[i64]>> = vec![(0..repeats as i64).collect()];
            let mut dimensions = vec![0usize];
            let mut sizes = vec![repeats];
            if let Some(group) = group {
                let (domains, group_codes, group_dimensions) = self.axis_projection(std::slice::from_ref(group))?;
                codes.extend(group_codes);
                dimensions.extend(group_dimensions);
                sizes.push(domains[0].size());
            }
            history = Some(self.history_from_axes(&codes, &dimensions, &sizes, self.group_keys(group)?, aggregation, uncertainty, None, None)?);
        }
        let history = history.unwrap();
        let begun = acquired_repeat_count(&self.snapshot);
        if begun == repeats { return Ok(history); }
        let cut = |array: &Array| array.rows(0, begun);
        Ok(RollingHistory { values: cut(&history.values), valid: cut(&history.valid), counts: cut(&history.counts),
            sem: history.sem.as_ref().map(cut), ..history })
    }

    /// `_single_revision_history`.
    fn single_revision_history(&self, group: Option<&AxisRef>, aggregation: Reduction, uncertainty: bool) -> Result<RollingHistory> {
        let uncertainty = uncertainty && aggregation == Reduction::Mean;
        if let Some(group) = group {
            let (domains, group_codes, group_dimensions) = self.axis_projection(std::slice::from_ref(group))?;
            let codes: Vec<Arc<[i64]>> = std::iter::once(Arc::from(vec![0i64])).chain(group_codes).collect();
            let dimensions: Vec<usize> = std::iter::once(0).chain(group_dimensions).collect();
            return self.history_from_axes(&codes, &dimensions, &[1, domains[0].size()], self.group_keys(Some(group))?,
                aggregation, uncertainty, None, None);
        }
        if self.snapshot.is_segmented() {
            return self.history_from_axes(&[Arc::from(vec![0i64])], &[0], &[1], vec![vec![]], aggregation, uncertainty, None, None);
        }
        let (pooled, pooled_sigma) = self.pooled()?;
        let value = reduce_scalar(&pooled, aggregation);
        let count = pooled.len();
        let sem = if uncertainty {
            let centred = |plane: &Array, offsets: &[f64]| -> Result<(Vec<f64>, Vec<f64>)> {
                if count == 0 { return Ok((vec![f64::NAN], vec![f64::NAN])); }
                let (first, second) = centred_moment_totals(plane, offsets[0]);
                Ok((vec![first / count as f64], vec![second / count as f64]))
            };
            Some(sem_of_mean(&[value], &[count as i64], &pooled, pooled_sigma.as_ref(), &centred)?)
        } else { None };
        Ok(self.history(vec![value], vec![count as i64], vec![count > 0 && value.is_finite()], sem, vec![vec![]], None, None))
    }

    /// `pooled_values()` and `_pooled_sigma()`: every valid sample of a materialized revision.
    pub fn pooled(&self) -> Result<(Array, Option<Array>)> {
        let packed = self.packed(true)?;
        let n = packed.values.len();
        if packed.valid.all_true() {
            return Ok((packed.values.reshape(vec![n]), packed.sigma.as_ref().map(|s| s.reshape(vec![n]))));
        }
        let keep = packed.valid.plane(n);
        Ok((packed.values.reshape(vec![n]).compress(&keep), packed.sigma.as_ref().map(|s| s.reshape(vec![n]).compress(&keep))))
    }

    /// `_segment_history`: rows by immutable segment, reduced once.
    /// `_segment_history`: rows carried by immutable segment identity from
    /// the previous revision's history; only the rows the other segments
    /// touch are reduced. A window drops its oldest shots and appends, a
    /// finite run appends, so a commit costs what it brought.
    fn segment_history(&self, group: Option<&AxisRef>, aggregation: Reduction, uncertainty: bool) -> Result<Option<RollingHistory>> {
        let carried = self.caches.borrow_mut().rolling.take();
        let Block::Segments(segments) = &self.snapshot.block else { return Ok(None) };
        if segments.is_empty() || segments.iter().any(|s| s.extent[0] != 1) { return Ok(None); }
        let layout = self.layout.clone();
        let origins: Vec<[usize; 2]> = segments.iter().map(|s| s.origin).collect();
        let (row_count, rows): (usize, Vec<i64>) = match &layout {
            None => (self.schema.repeat.size(), origins.iter().map(|o| o[0] as i64).collect()),
            Some(layout) => {
                let firsts: Vec<i64> = origins.iter().map(|o| o[1] as i64).collect();
                let lasts: Vec<i64> = segments.iter().map(|s| (s.origin[1] + s.extent[1] - 1) as i64).collect();
                let rows = layout.codes_at(&self.schema, &firsts);
                // A window's shots arrive in order: distinct without a sort.
                let distinct = rows.windows(2).all(|pair| pair[0] < pair[1]) || {
                    let mut unique = rows.clone();
                    unique.sort_unstable();
                    unique.dedup();
                    unique.len() == rows.len()
                };
                if origins.iter().any(|o| o[0] != 0) || rows != layout.codes_at(&self.schema, &lasts) || !distinct {
                    return Ok(None);
                }
                (layout.shot_count(), rows)
            }
        };
        let keys = self.group_keys(group)?;
        let (group_codes, group_dimension) = match group {
            None => (None, None),
            Some(group) => { let (_, codes, dims) = self.axis_projection(std::slice::from_ref(group))?; (Some(codes[0].clone()), Some(dims[0])) }
        };
        let query = (group.cloned(), aggregation, uncertainty);
        let identities: Vec<usize> = segments.iter().map(|s| Arc::as_ptr(s.segment) as usize).collect();
        let groups = keys.len();
        // The carried history holds its segments alive: an equal address is
        // the very same immutable planes.
        let (mut start, mut matched) = (0usize, 0usize);
        if let Some(c) = carried.as_ref().filter(|c| c.query == query && c.keys == keys && c.group_dimension == group_dimension
            && (group_dimension == Some(1) && layout.is_some() || c.group_codes == group_codes)) {
            if let Some(found) = c.identities.iter().position(|id| *id == identities[0]) {
                start = found;
                matched = c.identities.len() - start;
                if matched > segments.len() || c.identities[start..] != identities[..matched]
                    || layout.is_none() && (start != 0 || c.row_count != row_count || c.origins[..matched] != origins[..matched]) {
                    matched = 0;
                }
            }
        }
        let mut values = vec![f64::NAN; row_count * groups];
        let mut counts = vec![0i64; row_count * groups];
        let mut valid = vec![false; row_count * groups];
        let want_sem = uncertainty && aggregation == Reduction::Mean;
        let mut sem = want_sem.then(|| vec![f64::NAN; row_count * groups]);
        let mut pending_rows: Vec<i64>;
        if matched > 0 {
            let c = carried.as_ref().unwrap();
            // (destination row, carried row) pairs.
            let moves: Vec<(usize, usize)> = if layout.is_none() {
                // Repeats that receive a new segment are reduced again whole.
                pending_rows = rows[matched..].to_vec();
                pending_rows.sort_unstable();
                pending_rows.dedup();
                (0..row_count).filter(|r| pending_rows.binary_search(&(*r as i64)).is_err()).map(|r| (r, r)).collect()
            } else {
                let mut kept: Vec<usize> = (0..matched).collect();
                if group_dimension == Some(1) {
                    // A Point group also depends on the placement in the parent.
                    let (now, before) = (group_codes.as_ref().unwrap(), c.group_codes.as_ref().unwrap());
                    kept.retain(|j| (0..segments.get(*j).extent[1]).all(|local|
                        now[origins[*j][1] + local] == before[c.origins[start + *j][1] + local]));
                }
                let moves: Vec<(usize, usize)> = kept.iter().map(|j| (rows[*j] as usize, c.rows[start + *j] as usize)).collect();
                if kept.len() == matched && rows.windows(2).all(|pair| pair[0] < pair[1]) {
                    // Ordered shots, every carried one kept: the rest arrived.
                    pending_rows = rows[matched..].to_vec();
                } else {
                    let mut destination: Vec<i64> = moves.iter().map(|m| m.0 as i64).collect();
                    destination.sort_unstable();
                    pending_rows = rows.clone();
                    pending_rows.sort_unstable();
                    pending_rows.dedup();
                    pending_rows.retain(|r| destination.binary_search(r).is_err());
                }
                moves
            };
            for (to, from) in moves {
                for column in 0..groups {
                    let (d, s) = (to * groups + column, from * groups + column);
                    values[d] = c.values[s];
                    counts[d] = c.counts[s];
                    valid[d] = c.valid[s];
                    if let (Some(sem), Some(old)) = (sem.as_mut(), c.sem.as_ref()) { sem[d] = old[s]; }
                }
            }
        } else {
            pending_rows = rows.clone();
            pending_rows.sort_unstable();
            pending_rows.dedup();
        }
        if !pending_rows.is_empty() {
            let selection: Vec<usize> = (0..segments.len()).filter(|i| pending_rows.binary_search(&rows[*i]).is_ok()).collect();
            let packed = pack(&self.snapshot, Some(&selection), want_sem, Some(&self.pools))?;
            let (selected, mut dimensions): (Vec<i64>, Vec<usize>) = match (&layout, &packed.rows) {
                (None, None) => ((0..row_count as i64).collect(), vec![0]),
                (None, Some((repeat, _))) => (repeat.to_vec(), vec![0]),
                (Some(layout), None) => (layout.codes(&self.schema).to_vec(), vec![1]),
                (Some(layout), Some((_, point))) => (layout.codes_at(&self.schema, point), vec![0]),
            };
            let mut codes = vec![selected.iter().map(|s| pending_rows.partition_point(|p| p < s) as i64).collect::<Vec<i64>>()];
            let mut sizes = vec![pending_rows.len()];
            if let (Some(group_codes), Some(group_dimension)) = (&group_codes, group_dimension) {
                match &packed.rows {
                    Some((repeat, point)) if group_dimension < 2 => {
                        let carrier = if group_dimension == 0 { repeat } else { point };
                        codes.push(carrier.iter().map(|r| group_codes[*r as usize]).collect());
                        dimensions.push(0);
                    }
                    Some(_) => { codes.push(group_codes.to_vec()); dimensions.push(group_dimension - 1); }
                    None => { codes.push(group_codes.to_vec()); dimensions.push(group_dimension); }
                }
                sizes.push(groups);
            }
            let reduced = axis_aggregate(&packed.values, &packed.valid, &codes, &dimensions, &sizes, aggregation, None);
            let errors = if want_sem {
                let centred = |plane: &Array, offsets: &[f64]| -> Result<(Vec<f64>, Vec<f64>)> {
                    let r = axis_aggregate(plane, &packed.valid, &codes, &dimensions, &sizes, Reduction::Sum, Some(offsets));
                    Ok((r.out, r.second))
                };
                Some(sem_of_mean(&reduced.out, &reduced.counts, &packed.values, packed.sigma.as_ref(), &centred)?)
            } else { None };
            for (position, row) in pending_rows.iter().enumerate() {
                for column in 0..groups {
                    let source = position * groups + column;
                    let destination = *row as usize * groups + column;
                    values[destination] = reduced.out[source];
                    counts[destination] = reduced.counts[source];
                    valid[destination] = reduced.counts[source] > 0 && reduced.out[source].is_finite();
                    if let (Some(sem), Some(errors)) = (sem.as_mut(), &errors) { sem[destination] = errors[source]; }
                }
            }
        }
        let history = self.history(values.clone(), counts.clone(), valid.clone(), sem.clone(), keys.clone(), layout.as_ref().map(|l| l.cells.clone()),
            layout.as_ref().and_then(|l| l.times.clone()));
        self.caches.borrow_mut().rolling = Some(RollingCarry { query, identities, segments: segments.clone(), keys, values, counts, valid, sem,
            group_dimension, group_codes, rows, origins, row_count, indexed: layout.is_some() });
        Ok(Some(history))
    }
}

/// `_rolling_carry`: one revision's segment history, by segment identity.
pub struct RollingCarry {
    query: (Option<AxisRef>, Reduction, bool),
    identities: Vec<usize>,
    /// Keeps the identified planes alive.
    #[allow(dead_code)]
    segments: Segments,
    keys: Vec<Vec<AxisValue>>,
    values: Vec<f64>,
    counts: Vec<i64>,
    valid: Vec<bool>,
    sem: Option<Vec<f64>>,
    group_dimension: Option<usize>,
    group_codes: Option<Arc<[i64]>>,
    rows: Vec<i64>,
    origins: Vec<[usize; 2]>,
    row_count: usize,
    /// Whether it was an indexed history (a carry crosses only to the same kind).
    pub(crate) indexed: bool,
}

/// `_reduce_scalar(values, aggregation)` over a flat valid pool.
pub fn reduce_scalar(values: &Array, aggregation: Reduction) -> f64 {
    if values.is_empty() { return f64::NAN; }
    with_dtype!(values.dtype(), T => {
        let slice = values.slice::<T>();
        let sum = || if values.dtype() == DType::F64 { 0.0 + pairwise_widened(slice) } else {
            slice.chunks(crate::numeric::BUFFER).fold(0.0, |total, chunk| total + pairwise_widened(chunk))
        };
        match aggregation {
            Reduction::Mean => sum() / slice.len() as f64,
            Reduction::Sum => sum(),
            Reduction::Min | Reduction::Max => crate::numeric::reduce_contiguous(super::nd::Elem::f64(slice[0]), &slice[1..],
                values.dtype().extremes(), aggregation == Reduction::Min),
            _ => super::nd::Elem::f64(slice[0]),
        }
    })
}

/// `_centred_moment_totals(flat, offset)`: blocked sums of `d` and `d*d`.
fn centred_moment_totals(plane: &Array, offset: f64) -> (f64, f64) {
    const BLOCK: usize = 1 << 17;
    let flat = plane.to_f64();
    let dtype = plane.dtype();
    let (mut first, mut second) = (0.0_f64, 0.0_f64);
    for block in flat.chunks(BLOCK) {
        let piece: Vec<f64> = block.iter().map(|v| super::nd::subtract_in_dtype(dtype, *v, offset)).collect();
        first += 0.0 + pairwise_widened(&piece);
        second += blas_dot_self(&piece);
    }
    (first, second)
}

/// `_trailing_trace(history, column, span, uncertainty)`.
fn trailing_trace(history: &RollingHistory, column: usize, span: usize, uncertainty: bool) -> (Vec<f64>, Option<Vec<f64>>, Vec<bool>) {
    let total = history.len();
    let groups = history.groups();
    let at = |array: &Array| -> Vec<f64> { let plane = array.to_f64(); (0..total).map(|r| plane[r * groups + column]).collect() };
    let values = at(&history.values);
    let valid = history.valid.bools();
    let mut contributing: Vec<bool> = (0..total).map(|r| valid[r * groups + column]).collect();
    let sems: Vec<f64> = match (&history.sem, uncertainty) { (Some(sem), true) => at(sem), _ => vec![f64::NAN; total] };
    let shot_n: Vec<i64> = if history.group_keys[column].is_empty() {
        let plane = history.counts.slice::<i64>();
        let counts: Vec<i64> = (0..total).map(|r| plane[r * groups + column]).collect();
        for (c, n) in contributing.iter_mut().zip(&counts) { *c = *c && *n > 0; }
        contributing.iter().zip(&counts).map(|(c, n)| if *c { *n } else { 0 }).collect()
    } else { contributing.iter().map(|c| i64::from(*c)).collect() };
    let mut means: Vec<f64> = contributing.iter().zip(&values).map(|(c, v)| if *c { *v } else { 0.0 }).collect();
    let mut m2s = vec![0.0_f64; total];
    let mut singles = vec![f64::NAN; total];
    for r in 0..total {
        if contributing[r] && shot_n[r] > 1 && sems[r].is_finite() { m2s[r] = sems[r] * sems[r] * shot_n[r] as f64 * (shot_n[r] - 1) as f64; }
        if contributing[r] && shot_n[r] == 1 && sems[r].is_finite() { singles[r] = sems[r] * sems[r]; }
    }
    let mut counts = shot_n;
    kernels::trailing_moment_windows(&mut counts, &mut means, &mut m2s, &mut singles, span);
    let sem = uncertainty.then(|| (0..total).map(|r| {
        let mut variance = f64::NAN;
        if counts[r] > 1 { variance = m2s[r] / (counts[r] - 1) as f64; }
        if counts[r] == 1 && singles[r].is_finite() { variance = singles[r]; }
        (variance / counts[r] as f64).sqrt()
    }).collect());
    let valid: Vec<bool> = counts.iter().zip(&means).map(|(n, m)| *n > 0 && m.is_finite()).collect();
    let means = means.into_iter().zip(&valid).map(|(m, v)| if *v { m } else { f64::NAN }).collect();
    (means, sem, valid)
}

/// `FitProjection._rolling_payload`: one history series per rolling group.
pub fn rolling_payload(view: &DataView, history: &RollingHistory, along: Option<&AxisRef>, group: Option<&AxisRef>,
                       trailing: usize, uncertainty: bool) -> Result<CurveData> {
    let total = history.len();
    if total == 0 { return Err(ProjectionError::value("rolling history cannot be empty")); }
    let role = view.rolling_coordinate_role(along)?;
    let (x_values, x_unit, x_label) = if role == PRIMARY_INDEX {
        let coordinates: Vec<f64> = match &history.source_indices {
            Some(indices) => indices.to_f64(),
            None => (0..total).map(|i| i as f64 - (total as f64 - 1.0)).collect(),
        };
        (coordinates, resolve_unit("1", None)?, "Shots from latest".to_owned())
    } else if role == SHOT_TIME && history.source_times.is_some() {
        let resolved = view.resolve(along.unwrap())?;
        (history.source_times.as_ref().unwrap().to_f64(), resolved.canonical_unit.clone(), resolved.label.to_string())
    } else {
        return Err(ProjectionError::value(format!("a rolling plot places its shots along the shot index or, for a history that stamps its shots, the shot-time axis; not {along:?}")));
    };
    let x_canonical = Array::vector(x_values);
    let x = match along {
        None => QuantityArray::with_display(x_canonical.clone(), x_canonical, x_unit.clone(), x_unit, x_label),
        Some(axis) => {
            let display_unit = view.resolve(axis)?.display_unit.clone();
            QuantityArray::with_display(x_canonical.clone(), convert_array(&x_canonical, &x_unit, &display_unit), x_unit, display_unit, x_label)
        }
    };
    let groups = history.groups();
    let unit = view.value_display_unit.clone();
    let canonical_unit = view.value_unit.clone();
    let value_label = view.value_label();
    let valid_plane = history.valid.bools();
    let values_plane: Vec<f64> = history.values.to_f64().into_iter().zip(valid_plane).map(|(v, k)| if *k { v } else { f64::NAN }).collect();
    let sem_plane: Option<Vec<f64>> = match (&history.sem, uncertainty) {
        (Some(sem), true) => Some(sem.to_f64().into_iter().zip(valid_plane).map(|(v, k)| if *k { v } else { f64::NAN }).collect()),
        _ => None,
    };
    let series = history.group_keys.iter().enumerate().map(|(column, key)| {
        let (canonical, sem, valid) = if trailing > 1 {
            trailing_trace(history, column, trailing, uncertainty)
        } else {
            let take = |plane: &[f64]| -> Vec<f64> { (0..total).map(|r| plane[r * groups + column]).collect() };
            (take(&values_plane), sem_plane.as_ref().map(|s| take(s)), (0..total).map(|r| valid_plane[r * groups + column]).collect())
        };
        let label = if key.is_empty() { value_label.clone() } else { key.iter().map(|v| v.label.as_str()).collect::<Vec<_>>().join(", ") };
        CurveSeries { x: x.clone(), y: QuantityArray::new(Array::vector(canonical), canonical_unit.clone(), unit.clone(), value_label.clone()),
            valid: Array::vector(valid), sem: sem.map(Array::vector), x_labels: None, group_key: key.clone(), label }
    }).collect();
    Ok(CurveData { revision: history.revision, generation: history.generation.clone(), x_ref: None,
        group_by: group.into_iter().cloned().collect(), series, x_role: Some(role.into()) })
}
