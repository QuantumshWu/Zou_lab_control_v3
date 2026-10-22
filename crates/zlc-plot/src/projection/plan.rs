//! The explicit input boundary (`ProjectionPlan`) and the projection driver
//! (`FitProjection._build_view_and_payload`): scope the source, build the
//! unit-aware view, project the payload, carry state between revisions.

use std::sync::Arc;

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

use super::curve::CellSpec;
use super::model::{SchemaCache, Snapshot, Source};
use super::nd::Scalar;
use super::output::{Payload, SelectionSubject, scalar_json, hex_bits};
use super::view::DataView;
use super::{ProjectionError, Result};
use crate::kinds::{AxisRef, PlotKind, Reduction, RelimMode};
use crate::units::UnitRegistry;

/// A float carried bit for bit (`{"f": "<hex>"}`), or a plain JSON number.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bits(pub f64);

impl Serialize for Bits {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        serde_json::json!({"f": hex_bits(self.0)}).serialize(serializer)
    }
}

pub fn parse_hex_f64(text: &str) -> Option<f64> {
    if text.len() != 16 { return None; }
    let mut bytes = [0u8; 8];
    for (i, byte) in bytes.iter_mut().enumerate() { *byte = u8::from_str_radix(&text[2 * i..2 * i + 2], 16).ok()?; }
    Some(f64::from_le_bytes(bytes))
}

impl<'de> Deserialize<'de> for Bits {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        match &value {
            Value::Number(n) => n.as_f64().map(Bits).ok_or_else(|| D::Error::custom("bad number")),
            Value::Object(o) => o.get("f").and_then(Value::as_str).and_then(parse_hex_f64).map(Bits).ok_or_else(|| D::Error::custom("bad float bits")),
            _ => Err(D::Error::custom("expected a float")),
        }
    }
}

/// A Python scalar in golden JSON (`goldens.scalar`).
pub fn scalar_from_json(value: &Value) -> std::result::Result<Scalar, String> {
    Ok(match value {
        Value::Null => Scalar::None,
        Value::Bool(v) => Scalar::Bool(*v),
        Value::Number(n) => if let Some(i) = n.as_i64() { Scalar::Int(i as i128) } else if let Some(u) = n.as_u64() { Scalar::Int(u as i128) }
                            else { Scalar::Float(n.as_f64().unwrap()) },
        Value::String(s) => Scalar::Text(s.as_str().into()),
        Value::Object(o) => {
            if let Some(f) = o.get("f").and_then(Value::as_str) { Scalar::Float(parse_hex_f64(f).ok_or("bad float")?) }
            else if let Some(i) = o.get("i").and_then(Value::as_str) { Scalar::Int(i.parse().map_err(|_| "bad int")?) }
            else if let Some(v) = o.get("v").and_then(Value::as_bool) { Scalar::Bool(v) }
            else { return Err(format!("unknown scalar {value}")); }
        }
        Value::Array(_) => return Err("array is not a scalar".into()),
    })
}

/// One resolved scope term: an axis pinned to one coordinate, or to its last.
#[derive(Clone, Debug, PartialEq)]
pub enum ScopeValue { Latest, Coordinate(Scalar) }

#[derive(Clone, Debug, PartialEq)]
pub struct ScopeTerm { pub axis: AxisRef, pub value: ScopeValue }

impl Serialize for ScopeTerm {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let value = match &self.value { ScopeValue::Latest => serde_json::json!({"latest": true}),
                                        ScopeValue::Coordinate(c) => serde_json::json!({"coordinate": scalar_json(c)}) };
        serde_json::json!({"axis": self.axis, "value": value}).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ScopeTerm {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        let axis: AxisRef = serde_json::from_value(value["axis"].clone()).map_err(D::Error::custom)?;
        let inner = &value["value"];
        let scope = if inner.get("latest").is_some() { ScopeValue::Latest }
            else { ScopeValue::Coordinate(scalar_from_json(&inner["coordinate"]).map_err(D::Error::custom)?) };
        Ok(ScopeTerm { axis, value: scope })
    }
}

/// The spec facts a projection reads (never labels, scope or coordinates).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SpecFacts {
    Curve { x: AxisRef, group: Option<AxisRef>, reduction: Reduction },
    Image { x: AxisRef, y: AxisRef, reduction: Reduction },
    Histogram { group: Option<AxisRef>, #[serde(default)] reduced: Vec<AxisRef>, reduction: Reduction },
    Rolling { x: Option<AxisRef>, group: Option<AxisRef>, reduction: Reduction },
    FacetGrid { facet: Option<AxisRef>, cell: Box<SpecFacts> },
}

impl SpecFacts {
    /// `semantic_spec`: a facet grid is its cell.
    pub fn semantic(&self) -> &SpecFacts { if let SpecFacts::FacetGrid { cell, .. } = self { cell } else { self } }
    pub fn kind(&self) -> PlotKind {
        match self { SpecFacts::Curve { .. } => PlotKind::Curve, SpecFacts::Image { .. } => PlotKind::Image,
            SpecFacts::Histogram { .. } => PlotKind::Histogram, SpecFacts::Rolling { .. } => PlotKind::Rolling,
            SpecFacts::FacetGrid { .. } => PlotKind::FacetGrid }
    }
    pub fn reduction(&self) -> Reduction {
        match self.semantic() { SpecFacts::Curve { reduction, .. } | SpecFacts::Image { reduction, .. }
            | SpecFacts::Histogram { reduction, .. } | SpecFacts::Rolling { reduction, .. } => *reduction,
            SpecFacts::FacetGrid { .. } => Reduction::Mean }
    }
    pub fn x(&self) -> Option<&AxisRef> {
        match self.semantic() { SpecFacts::Curve { x, .. } | SpecFacts::Image { x, .. } => Some(x), SpecFacts::Rolling { x, .. } => x.as_ref(), _ => None }
    }
    pub fn y(&self) -> Option<&AxisRef> { if let SpecFacts::Image { y, .. } = self.semantic() { Some(y) } else { None } }
    pub fn cell(&self) -> Option<CellSpec> {
        Some(match self {
            SpecFacts::Curve { x, group, reduction } => CellSpec::Curve { x: x.clone(), group: group.clone(), reduction: *reduction },
            SpecFacts::Image { x, y, reduction } => CellSpec::Image { x: x.clone(), y: y.clone(), reduction: *reduction },
            SpecFacts::Histogram { group, reduced, reduction } => CellSpec::Histogram { group: group.clone(), reduced: reduced.clone(), reduction: *reduction },
            _ => return None,
        })
    }
}

/// The display-state facts a projection reads.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DisplayFacts {
    #[serde(default)] pub window: Option<usize>,
    #[serde(default)] pub uncertainty: Option<bool>,
    #[serde(default)] pub trailing: Option<usize>,
    #[serde(default)] pub bin_count: Option<usize>,
    #[serde(default)] pub x_relim_mode: Option<String>,
    #[serde(default)] pub x_min: Option<Bits>,
    #[serde(default)] pub x_max: Option<Bits>,
    #[serde(default)] pub x_display_unit: Option<String>,
    #[serde(default)] pub y_display_unit: Option<String>,
    #[serde(default)] pub facet_display_unit: Option<String>,
    #[serde(default)] pub value_display_unit: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Defaults { pub histogram_domain_padding_fraction: Bits }

impl Default for Defaults { fn default() -> Self { Defaults { histogram_domain_padding_fraction: Bits(0.05) } } }

/// Everything one projection reads from the resolved configuration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProjectionPlan {
    pub spec: SpecFacts,
    /// `projection_scope(schema, spec)`, resolved by the configuration.
    #[serde(default)] pub scope: Vec<ScopeTerm>,
    #[serde(default)] pub display: DisplayFacts,
    /// The display facts the view was built under, when they differ.
    #[serde(default)] pub view_display: Option<DisplayFacts>,
    #[serde(default)] pub defaults: Defaults,
}

/// `HistogramProjection`: the bins a histogram holds across revisions.
#[derive(Clone, Debug, PartialEq)]
pub struct HistogramProjection {
    pub bin_count: usize,
    pub requested_bins: usize,
    pub domain: (f64, f64),
    pub edges: Vec<f64>,
}

/// One projected revision.
pub struct Projected {
    pub view: DataView,
    pub payload: Payload,
}

/// The projection owner of one panel: carries between revisions.
#[derive(Default)]
pub struct Projection {
    pub histogram: Option<HistogramProjection>,
    pub registry: Option<Arc<UnitRegistry>>,
    schemas: SchemaCache,
    exports: super::model::SegmentExports,
    pub(crate) carry: Option<super::view::ViewCarries>,
    pub(crate) pools: Arc<super::nd::Pools>,
}

impl Projection {
    pub fn new() -> Self { Self::default() }

    /// Project one runtime read: its planes viewed in place, the segments
    /// this owner's previous read converted carried over.
    pub fn project_read(&mut self, read: &zlc_core::runtime::Read, plan: &ProjectionPlan) -> Result<Projected> {
        let snapshot = Snapshot::from_read(read, &mut self.schemas, &mut self.exports)?;
        self.project_snapshot(snapshot, plan)
    }

    /// Project one native source (a golden snapshot, an archived dataset).
    pub fn project(&mut self, source: &Source, plan: &ProjectionPlan) -> Result<Projected> {
        let snapshot = Snapshot::from_source(source, &mut self.schemas)?;
        self.project_snapshot(snapshot, plan)
    }

    /// `_build_view_and_payload` for one snapshot.
    pub fn project_snapshot(&mut self, snapshot: Snapshot, plan: &ProjectionPlan) -> Result<Projected> {
        let view_display = plan.view_display.as_ref().unwrap_or(&plan.display);
        let scoped = super::scope::scoped_data(&snapshot, plan, view_display)?;
        let mut overrides: Vec<(AxisRef, Arc<str>)> = Vec::new();
        let mut set = |axis: &AxisRef, unit: &Option<String>| {
            if let Some(unit) = unit {
                overrides.retain(|(key, _)| key != axis);
                overrides.push((axis.clone(), unit.as_str().into()));
            }
        };
        if let SpecFacts::FacetGrid { facet: Some(facet), .. } = &plan.spec { set(facet, &view_display.facet_display_unit); }
        if let Some(x) = plan.spec.x() { set(x, &view_display.x_display_unit); }
        if let Some(y) = plan.spec.y() { set(y, &view_display.y_display_unit); }
        let mut view = DataView::new(scoped, overrides, view_display.value_display_unit.as_deref(), self.registry.clone())?;
        view.pools = self.pools.clone();
        if let Some(carry) = self.carry.take() { view.adopt(carry); }
        // A refused revision still hands on what it learnt of the shots.
        let payload = self.build_payload(&view, plan);
        self.carry = Some(view.release());
        Ok(Projected { view, payload: payload? })
    }

    fn build_payload(&mut self, view: &DataView, plan: &ProjectionPlan) -> Result<Payload> {
        let display = &plan.display;
        let uncertainty_on = display.uncertainty.unwrap_or(false);
        let spec = &plan.spec;
        let statistic_mean = spec.reduction().statistic() == Reduction::Mean;
        if !matches!(spec, SpecFacts::Histogram { .. }) { view.release_window_frequency(); }
        match spec {
            SpecFacts::Curve { x, group, reduction } =>
                Ok(Payload::Curve(view.curve(x, group.as_slice(), *reduction, uncertainty_on && statistic_mean)?)),
            SpecFacts::Image { x, y, reduction } => Ok(Payload::Image(view.image(x, y, *reduction)?)),
            SpecFacts::Rolling { x, group, reduction } => {
                let uncertainty = uncertainty_on && statistic_mean;
                let history = view.rolling_history(x.as_ref(), group.as_ref(), *reduction, uncertainty)?;
                let trailing = if statistic_mean { display.trailing.unwrap_or(1) } else { 1 };
                Ok(Payload::Curve(super::rolling::rolling_payload(view, &history, x.as_ref(), group.as_ref(), trailing, uncertainty)?))
            }
            SpecFacts::Histogram { group, reduced, reduction } => {
                let window = display.window.ok_or_else(|| ProjectionError::value("histogram projection needs the window"))?;
                Ok(Payload::Histogram(super::histogram::histogram_payload(self, view, plan, group.as_ref(), reduced, *reduction, window)?))
            }
            SpecFacts::FacetGrid { facet, cell } => {
                let cell_spec = cell.cell().ok_or_else(|| ProjectionError::type_error("facet cell must be CurvePlot, ImagePlot, or HistogramPlot"))?;
                if let CellSpec::Histogram { .. } = cell_spec {
                    let window = display.window.unwrap_or(1);
                    return Ok(Payload::Facet(super::histogram::histogram_facet_payload(self, view, plan, facet.as_ref(), &cell_spec, window)?));
                }
                let uncertainty = matches!(cell_spec, CellSpec::Curve { .. }) && uncertainty_on && statistic_mean;
                Ok(Payload::Facet(view.facet(facet.as_ref(), &cell_spec, uncertainty)?))
            }
        }
    }

    /// `DataView.selection_subject(spec, payload, source_window=...)`.
    pub fn selection_subject(view: &DataView, plan: &ProjectionPlan, payload: &Payload) -> Result<SelectionSubject> {
        super::scope::selection_subject(view, plan, payload)
    }
}

pub(crate) fn relim_mode(display: &DisplayFacts) -> RelimMode {
    display.x_relim_mode.as_deref().and_then(RelimMode::parse).unwrap_or(RelimMode::Normal)
}
