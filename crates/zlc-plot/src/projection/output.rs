//! What a plot draws: the projection payloads of `zlc_plot.data_view`, with
//! canonical and display planes, units, labels and group identities.

use std::sync::Arc;

use serde_json::{Value, json};

use super::nd::{Array, DType, Scalar};
use crate::kinds::{AxisRef, PlotKind};
use crate::units::{Conversion, Unit};

/// `QuantityArray`: canonical and display representations of one quantity.
#[derive(Clone, Debug)]
pub struct QuantityArray {
    pub canonical: Array,
    pub display: Array,
    pub canonical_unit: Arc<Unit>,
    pub display_unit: Arc<Unit>,
    pub label: String,
}

impl QuantityArray {
    /// Display computed the way `QuantityArray.display` computes it.
    pub fn new(canonical: Array, canonical_unit: Arc<Unit>, display_unit: Arc<Unit>, label: String) -> Self {
        let display = convert_array(&canonical, &canonical_unit, &display_unit);
        QuantityArray { canonical, display, canonical_unit, display_unit, label }
    }
    pub fn with_display(canonical: Array, display: Array, canonical_unit: Arc<Unit>, display_unit: Arc<Unit>, label: String) -> Self {
        QuantityArray { canonical, display, canonical_unit, display_unit, label }
    }
    /// The same quantity at the positions where `keep`.
    pub fn compress(&self, keep: &[bool]) -> Self {
        QuantityArray { canonical: self.canonical.compress(keep), display: self.display.compress(keep),
            canonical_unit: self.canonical_unit.clone(), display_unit: self.display_unit.clone(), label: self.label.clone() }
    }
}

fn scaled_factor(conversion: &Conversion) -> Option<f64> { if let Conversion::Scaled(f) = conversion { Some(*f) } else { None } }

/// `Unit.convert_value_to` over an array, in numpy's dtypes: the same array
/// when the units are equal; a scaled unit keeps a float32/float16 array in
/// its own precision (NEP 50), everything else is computed in float64.
pub fn convert_array(values: &Array, source: &Unit, target: &Unit) -> Array {
    if source == target { return values.clone(); }
    if let (Some(a), Some(b)) = (scaled_factor(&source.conversion), scaled_factor(&target.conversion)) {
        if values.dtype() == DType::F32 {
            let (a, b) = (a as f32, b as f32);
            return Array::from_vec(values.slice::<f32>().iter().map(|v| (v * a) / b).collect(), values.shape().to_vec());
        }
    }
    let converted: Vec<f64> = values.to_f64().into_iter().map(|v| target.from_base(source.to_base(v))).collect();
    Array::from_vec(converted, values.shape().to_vec())
}

/// `AxisValue`: one labelled coordinate of a group or facet axis.
#[derive(Clone, Debug)]
pub struct AxisValue {
    pub axis: AxisRef,
    pub index: Option<usize>,
    pub canonical: Scalar,
    pub display: Scalar,
    pub label: String,
}

impl PartialEq for AxisValue {
    fn eq(&self, other: &Self) -> bool {
        self.axis == other.axis && self.index == other.index && self.canonical == other.canonical
            && self.display == other.display && self.label == other.label
    }
}

#[derive(Clone, Debug)]
pub struct CurveSeries {
    pub x: QuantityArray,
    pub y: QuantityArray,
    pub valid: Array,
    /// Standard error of each MEAN point, canonical unit.
    pub sem: Option<Array>,
    pub x_labels: Option<Vec<String>>,
    pub group_key: Vec<AxisValue>,
    pub label: String,
}

#[derive(Clone, Debug)]
pub struct CurveData {
    pub revision: u64,
    pub generation: Arc<str>,
    pub x_ref: Option<AxisRef>,
    pub group_by: Vec<AxisRef>,
    pub series: Vec<CurveSeries>,
    pub x_role: Option<Arc<str>>,
}

#[derive(Clone, Debug)]
pub struct ImageData {
    pub revision: u64,
    pub generation: Arc<str>,
    pub x_ref: AxisRef,
    pub y_ref: AxisRef,
    pub x: QuantityArray,
    pub y: QuantityArray,
    pub z: QuantityArray,
    pub valid: Array,
}

#[derive(Clone, Debug)]
pub struct HistogramData {
    pub revision: u64,
    pub generation: Arc<str>,
    pub edges: QuantityArray,
    pub centers: QuantityArray,
    /// `(groups, bins)` int64.
    pub counts: Array,
    pub group_keys: Vec<Vec<AxisValue>>,
}

impl HistogramData {
    pub fn labels(&self) -> Vec<String> {
        self.group_keys.iter().map(|key| if key.is_empty() { self.edges.label.clone() }
            else { key.iter().map(|v| v.label.as_str()).collect::<Vec<_>>().join(", ") }).collect()
    }
}

/// Every shot of one rolling projection, as `(shots, groups)` planes.
#[derive(Clone, Debug)]
pub struct RollingHistory {
    pub revision: u64,
    pub generation: Arc<str>,
    pub values: Array,
    pub valid: Array,
    pub counts: Array,
    pub group_keys: Vec<Vec<AxisValue>>,
    pub source_indices: Option<Array>,
    pub source_times: Option<Array>,
    pub sem: Option<Array>,
}

impl RollingHistory {
    pub fn len(&self) -> usize { self.values.shape()[0] }
    pub fn is_empty(&self) -> bool { self.len() == 0 }
    pub fn groups(&self) -> usize { self.values.shape()[1] }
}

#[derive(Clone, Debug)]
pub struct FacetCell {
    pub facet_index: Option<usize>,
    pub facet_value_canonical: Scalar,
    pub facet_value_display: Scalar,
    pub label: String,
    pub payload: Payload,
}

#[derive(Clone, Debug)]
pub struct FacetData {
    pub revision: u64,
    pub generation: Arc<str>,
    pub cells: Vec<FacetCell>,
}

/// A pulse timeline's payload: the authored timeline itself (no view is
/// built of it) at the session's revision.
#[derive(Clone, Debug)]
pub struct PulsePayload { pub data: Arc<crate::pulse::PulseTimelineData>, pub revision: u64 }

#[derive(Clone, Debug)]
pub enum Payload { Curve(CurveData), Image(ImageData), Histogram(HistogramData), Facet(FacetData), Pulse(PulsePayload) }

impl Payload {
    pub fn revision(&self) -> u64 {
        match self {
            Payload::Curve(p) => p.revision, Payload::Image(p) => p.revision, Payload::Histogram(p) => p.revision, Payload::Facet(p) => p.revision,
            Payload::Pulse(p) => p.revision,
        }
    }
    /// The run the data came out of; an authored pulse timeline has none.
    pub fn generation(&self) -> Option<&Arc<str>> {
        match self {
            Payload::Curve(p) => Some(&p.generation), Payload::Image(p) => Some(&p.generation), Payload::Histogram(p) => Some(&p.generation),
            Payload::Facet(p) => Some(&p.generation), Payload::Pulse(_) => None,
        }
    }
}

/// Exact upstream quantities cut by one accepted plot projection.
#[derive(Clone, Debug, PartialEq)]
pub struct SelectionSubject {
    pub plot_kind: PlotKind,
    pub x: Option<AxisRef>,
    pub y: Option<AxisRef>,
    pub x_coordinate_frame: Option<Arc<str>>,
    pub y_coordinate_frame: Option<Arc<str>>,
    pub scope: Vec<(AxisRef, Scalar)>,
    pub source_window: Option<usize>,
}

// ------------------------------------------------------------- golden form
/// The arrays a golden document refers to as `{"rust": index}`.
#[derive(Default)]
pub struct GoldenArrays(pub Vec<Array>);

impl GoldenArrays {
    pub fn array(&mut self, array: &Array) -> Value { self.0.push(array.clone()); json!({"rust": self.0.len() - 1}) }
    fn optional(&mut self, array: &Option<Array>) -> Value { array.as_ref().map_or(Value::Null, |a| self.array(a)) }
}

/// A Python scalar as `goldens.scalar` writes it.
pub fn scalar_json(value: &Scalar) -> Value {
    match value {
        Scalar::None => Value::Null,
        Scalar::Bool(v) => json!(v),
        Scalar::Int(v) if *v >= i64::MIN as i128 && *v <= i64::MAX as i128 => json!(*v as i64),
        Scalar::Int(v) if *v >= 0 && *v <= u64::MAX as i128 => json!(*v as u64),
        Scalar::Int(v) => json!({"i": v.to_string()}),
        Scalar::Float(v) => json!({"f": hex_bits(*v)}),
        Scalar::Text(v) => json!(v.as_ref()),
    }
}

pub fn hex_bits(value: f64) -> String { value.to_le_bytes().iter().map(|b| format!("{b:02x}")).collect() }

pub fn axis_ref_json(axis: &AxisRef) -> Value { json!({"domain": axis.domain.value(), "axis_id": axis.axis_id}) }
fn optional_ref(axis: &Option<AxisRef>) -> Value { axis.as_ref().map_or(Value::Null, axis_ref_json) }

impl QuantityArray {
    pub fn golden(&self, arrays: &mut GoldenArrays) -> Value {
        json!({"type": "QuantityArray", "canonical": arrays.array(&self.canonical), "display": arrays.array(&self.display),
               "canonical_unit": self.canonical_unit.symbol, "display_unit": self.display_unit.symbol, "label": self.label})
    }
}

impl AxisValue {
    pub fn golden(&self) -> Value {
        json!({"type": "AxisValue", "ref": axis_ref_json(&self.axis), "index": self.index, "canonical": scalar_json(&self.canonical),
               "display": scalar_json(&self.display), "label": self.label})
    }
}

fn keys_json(keys: &[Vec<AxisValue>]) -> Value {
    Value::Array(keys.iter().map(|key| Value::Array(key.iter().map(AxisValue::golden).collect())).collect())
}

impl CurveData {
    pub fn golden(&self, arrays: &mut GoldenArrays) -> Value {
        let series: Vec<Value> = self.series.iter().map(|s| json!({
            "type": "CurveSeries", "x": s.x.golden(arrays), "y": s.y.golden(arrays), "valid": arrays.array(&s.valid),
            "sem": arrays.optional(&s.sem), "x_labels": s.x_labels, "label": s.label,
            "group_key": Value::Array(s.group_key.iter().map(AxisValue::golden).collect()),
        })).collect();
        json!({"type": "CurveData", "revision": self.revision, "generation": self.generation.as_ref(), "x_ref": optional_ref(&self.x_ref),
               "group_by": self.group_by.iter().map(axis_ref_json).collect::<Vec<_>>(), "series": series,
               "x_role": self.x_role.as_deref()})
    }
}

impl ImageData {
    pub fn golden(&self, arrays: &mut GoldenArrays) -> Value {
        json!({"type": "ImageData", "revision": self.revision, "generation": self.generation.as_ref(),
               "x_ref": axis_ref_json(&self.x_ref), "y_ref": axis_ref_json(&self.y_ref), "x": self.x.golden(arrays),
               "y": self.y.golden(arrays), "z": self.z.golden(arrays), "valid": arrays.array(&self.valid)})
    }
}

impl HistogramData {
    pub fn golden(&self, arrays: &mut GoldenArrays) -> Value {
        json!({"type": "HistogramData", "revision": self.revision, "generation": self.generation.as_ref(),
               "edges": self.edges.golden(arrays), "centers": self.centers.golden(arrays), "counts": arrays.array(&self.counts),
               "group_keys": keys_json(&self.group_keys)})
    }
}

impl RollingHistory {
    pub fn golden(&self, arrays: &mut GoldenArrays) -> Value {
        json!({"type": "RollingHistory", "revision": self.revision, "generation": self.generation.as_ref(),
               "values": arrays.array(&self.values), "valid": arrays.array(&self.valid), "counts": arrays.array(&self.counts),
               "group_keys": keys_json(&self.group_keys), "source_indices": arrays.optional(&self.source_indices),
               "source_times": arrays.optional(&self.source_times), "sem": arrays.optional(&self.sem)})
    }
}

impl Payload {
    pub fn golden(&self, arrays: &mut GoldenArrays) -> Value {
        match self {
            Payload::Curve(p) => p.golden(arrays),
            Payload::Image(p) => p.golden(arrays),
            Payload::Histogram(p) => p.golden(arrays),
            Payload::Facet(p) => json!({"type": "FacetData", "revision": p.revision, "generation": p.generation.as_ref(),
                "cells": p.cells.iter().map(|cell| json!({"type": "FacetCell", "facet_index": cell.facet_index,
                    "facet_value_canonical": scalar_json(&cell.facet_value_canonical),
                    "facet_value_display": scalar_json(&cell.facet_value_display), "label": cell.label,
                    "payload": cell.payload.golden(arrays)})).collect::<Vec<_>>()}),
            Payload::Pulse(p) => json!({"type": "PulseTimelineData", "revision": p.revision, "data": p.data.to_json()}),
        }
    }
}

impl SelectionSubject {
    pub fn golden(&self) -> Value {
        json!({"type": "SelectionSubject", "plot_kind": self.plot_kind.value(), "x": optional_ref(&self.x), "y": optional_ref(&self.y),
               "x_coordinate_frame": self.x_coordinate_frame.as_deref(), "y_coordinate_frame": self.y_coordinate_frame.as_deref(),
               "scope": self.scope.iter().map(|(axis, value)| json!([axis_ref_json(axis), scalar_json(value)])).collect::<Vec<_>>(),
               "source_window": self.source_window})
    }
}
