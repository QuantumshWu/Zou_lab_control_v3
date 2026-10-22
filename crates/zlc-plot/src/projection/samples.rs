//! Sample-level reads: `DataView.samples` (every physical sample in the full
//! schema shape), `DataView.coordinate` (a coordinate broadcast over every
//! sample, kept factored as the one line it varies along) and the selector
//! masks of `FitProjection._selector_mask` (X range, area, threshold).

use std::sync::Arc;

use super::fit_selection::{SelectorFacts, SelectorKind, SelectorValue};
use super::nd::{Array, Mask, product};
use super::output::{Payload, QuantityArray};
use super::plan::{ProjectionPlan, SpecFacts};
use super::view::{DataView, acquired_repeat_count};
use super::{ProjectionError, Result};
use crate::kinds::{AxisDomain, AxisRef};
use crate::units::Unit;

/// `SampleProjection`: the full `(R, P, *D)` sample planes of one revision.
#[derive(Clone, Debug)]
pub struct SampleProjection {
    pub revision: u64,
    pub generation: Arc<str>,
    pub shape: Vec<usize>,
    pub value: QuantityArray,
    pub valid_mask: Array,
    pub sigma: Option<Array>,
}

/// `CoordinateArray`: one axis's coordinate at every sample, stored as the
/// line along the tensor dimension it varies on.
#[derive(Clone, Debug)]
pub struct CoordinateArray {
    pub axis: AxisRef,
    pub dimension: usize,
    pub shape: Vec<usize>,
    /// Canonical / display coordinate and declared index of each carrier row.
    pub canonical_line: Array,
    pub display_line: Array,
    pub indices_line: Arc<[i64]>,
    pub canonical_unit: Arc<Unit>,
    pub display_unit: Arc<Unit>,
    pub label: String,
}

impl CoordinateArray {
    fn stride(&self) -> usize { product(&self.shape[self.dimension + 1..]) }
    /// The row along `dimension` that flat sample `index` sits on.
    pub fn row(&self, index: usize) -> usize { (index / self.stride()) % self.shape[self.dimension] }
    pub fn canonical_at(&self, index: usize) -> f64 { self.canonical_line.get_f64(self.row(index)) }
    /// The broadcast plane written out (numpy's `np.asarray(coordinate.canonical)`).
    pub fn canonical_plane(&self) -> Array { self.broadcast(&self.canonical_line) }
    pub fn display_plane(&self) -> Array { self.broadcast(&self.display_line) }
    fn broadcast(&self, line: &Array) -> Array {
        let total = product(&self.shape);
        let positions: Vec<usize> = (0..total).map(|i| self.row(i)).collect();
        line.take_flat(&positions, self.shape.clone())
    }
}

impl DataView {
    /// `DataView.samples`.
    pub fn samples(&self) -> Result<SampleProjection> {
        let packed = self.packed(true)?;
        let shape = self.schema.physical_shape.clone();
        let (values, valid, sigma) = match &packed.rows {
            None => (packed.values.clone(), match &packed.valid {
                Mask::Uniform(v) => Array::constant(*v, shape.clone()),
                Mask::Plane(p) => Array::from_vec(p.to_vec(), shape.clone()),
            }, packed.sigma.clone()),
            Some((repeat, point)) => {
                let cell = product(&shape[2..]);
                let total = product(&shape);
                let mut target = vec![0usize; repeat.len()];
                for (row, (r, p)) in repeat.iter().zip(point.iter()).enumerate() { target[row] = (*r as usize * shape[1] + *p as usize) * cell; }
                let values = crate::with_dtype!(self.schema.value.dtype, T => {
                    let source = packed.values.slice::<T>();
                    let mut whole: Vec<T> = vec![zero::<T>(); total];
                    for (row, start) in target.iter().enumerate() { whole[*start..*start + cell].copy_from_slice(&source[row * cell..(row + 1) * cell]); }
                    Array::from_vec(whole, shape.clone())
                });
                let mut valid = vec![false; total];
                for (row, start) in target.iter().enumerate() { for c in 0..cell { valid[start + c] = packed.valid.get(row * cell + c); } }
                let sigma = packed.sigma.as_ref().map(|s| {
                    let source = s.slice::<f64>();
                    let mut whole = vec![f64::NAN; total];
                    for (row, start) in target.iter().enumerate() { whole[*start..*start + cell].copy_from_slice(&source[row * cell..(row + 1) * cell]); }
                    Array::from_vec(whole, shape.clone())
                });
                (values, Array::from_vec(valid, shape.clone()), sigma)
            }
        };
        Ok(SampleProjection {
            revision: self.revision(), generation: self.generation().clone(), shape,
            value: QuantityArray::new(values, self.value_unit.clone(), self.value_display_unit.clone(), self.value_label()),
            valid_mask: valid, sigma,
        })
    }

    /// `DataView.coordinate(ref)`.
    pub fn coordinate(&self, axis: &AxisRef) -> Result<CoordinateArray> {
        let resolved = self.resolve(axis)?;
        let positions: Vec<usize> = resolved.source_codes.iter().map(|c| *c as usize).collect();
        let n = positions.len();
        Ok(CoordinateArray {
            axis: axis.clone(), dimension: resolved.dimension(), shape: self.schema.physical_shape.clone(),
            canonical_line: resolved.domain_canonical.take_flat(&positions, vec![n]),
            display_line: resolved.domain_display.take_flat(&positions, vec![n]),
            indices_line: resolved.source_codes.clone(),
            canonical_unit: resolved.canonical_unit.clone(), display_unit: resolved.display_unit.clone(), label: resolved.label.to_string(),
        })
    }
}

fn zero<T: super::nd::Elem>() -> T {
    // Every element type's zero is its all-zero bit pattern (numpy's np.zeros).
    unsafe { std::mem::zeroed() }
}

fn bits_range(range: &[super::plan::Bits; 2]) -> (f64, f64) { let (a, b) = (range[0].0, range[1].0); if b < a { (b, a) } else { (a, b) } }

/// `FitProjection._selector_mask(state)` for the numeric selectors: which
/// samples of this revision the selection holds, in the full schema shape.
pub fn selector_mask(view: &DataView, plan: &ProjectionPlan, payload: &Payload, state: &SelectorFacts) -> Result<Array> {
    if state.kind == SelectorKind::Crosshair {
        return Err(ProjectionError::type_error("the crosshair selection needs the drawn axes' transform"));
    }
    let samples = view.samples()?;
    let shape = samples.shape.clone();
    let total = product(&shape);
    let mut mask: Vec<bool> = (0..total).map(|i| samples.valid_mask.get_f64(i) != 0.0).collect();
    let semantic = plan.spec.semantic();
    let rolling = matches!(plan.spec, SpecFacts::Rolling { .. });
    let histogram = matches!(semantic, SpecFacts::Histogram { .. });
    let offsets = if rolling { Some(rolling_sample_offsets(view, plan)?) } else { None };
    if let Some(offsets) = &offsets {
        let shots: Vec<f64> = match payload {
            Payload::Curve(curve) if !curve.series.is_empty() => curve.series[0].x.canonical.to_f64(),
            _ => vec![],
        };
        for (keep, offset) in mask.iter_mut().zip(offsets) { *keep &= shots.contains(offset); }
    }
    let value: Vec<f64> = samples.value.canonical.to_f64();
    let x_test = |mask: &mut Vec<bool>, low: f64, high: f64| -> Result<()> {
        if let Some(offsets) = &offsets {
            for (keep, x) in mask.iter_mut().zip(offsets) { *keep &= *x >= low && *x <= high; }
        } else if histogram {
            for (keep, x) in mask.iter_mut().zip(&value) { *keep &= *x >= low && *x <= high; }
        } else {
            let axis = semantic.x().ok_or_else(|| ProjectionError::type_error("this plot has no coordinate x axis"))?;
            let coordinate = view.coordinate(axis)?;
            let line: Vec<bool> = coordinate.canonical_line.to_f64().iter().map(|x| *x >= low && *x <= high).collect();
            for (index, keep) in mask.iter_mut().enumerate() { *keep &= line[coordinate.row(index)]; }
        }
        Ok(())
    };
    match (&state.kind, &state.value) {
        (SelectorKind::XRange, SelectorValue::Range(range)) => { let (lo, hi) = bits_range(range); x_test(&mut mask, lo, hi)?; }
        (SelectorKind::Area, SelectorValue::Area(area)) => {
            let (lo, hi) = bits_range(&area[0]);
            x_test(&mut mask, lo, hi)?;
            let (ylo, yhi) = bits_range(&area[1]);
            if let SpecFacts::Image { y, .. } = semantic {
                let coordinate = view.coordinate(y)?;
                let line: Vec<bool> = coordinate.canonical_line.to_f64().iter().map(|v| *v >= ylo && *v <= yhi).collect();
                for (index, keep) in mask.iter_mut().enumerate() { *keep &= line[coordinate.row(index)]; }
            } else if !histogram {
                for (keep, v) in mask.iter_mut().zip(&value) { *keep &= *v >= ylo && *v <= yhi; }
            }
        }
        (SelectorKind::Threshold, SelectorValue::Threshold(t)) => { for (keep, v) in mask.iter_mut().zip(&value) { *keep &= *v >= t.0; } }
        _ => return Err(ProjectionError::type_error("selector value does not match its kind")),
    }
    Ok(Array::from_vec(mask, shape))
}

/// `_rolling_sample_offsets`: the rolling x each sample of this revision sits at.
fn rolling_sample_offsets(view: &DataView, plan: &ProjectionPlan) -> Result<Vec<f64>> {
    let shape = view.schema().physical_shape.clone();
    let total = product(&shape);
    if let Some(layout) = view.history_layout() {
        let axis = match &plan.spec { SpecFacts::Rolling { x: Some(x), .. } => x.clone(),
            _ => AxisRef { domain: AxisDomain::Point, axis_id: layout.primary_axis_id.to_string() } };
        let coordinate = view.coordinate(&axis)?;
        let line = coordinate.canonical_line.to_f64();
        return Ok((0..total).map(|i| line[coordinate.row(i)]).collect());
    }
    let begun = acquired_repeat_count(view.snapshot()) as f64;
    let inner = product(&shape[1..]);
    Ok((0..total).map(|i| (i / inner) as f64 - (begun - 1.0)).collect())
}
