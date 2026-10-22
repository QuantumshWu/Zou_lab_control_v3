//! Selector and viewport value types (`zlc_plot.selectors`): ranges,
//! rectangles, crosshair points, the committed selector snapshot, viewport
//! normalization and coordinate-addressed classifier threshold targets.
//! Pointer geometry belongs to the session/renderer.

use std::sync::Arc;

use super::error::{PlotError, Result};
use super::kinds::{axis_ref, parse_domain, AxisDomain};
use super::value::{repr_str, Value, ValueMap};
use crate::pyfmt::repr as float_repr;

fn finite(value: f64, field: &str) -> Result<f64> {
    if !value.is_finite() { return Err(PlotError::value(format!("{field} must be finite"))); }
    Ok(value)
}

/// `finite_real(value, field, normalize_zero=False)` of a wire value.
pub(crate) fn finite_value(value: &Value, field: &str) -> Result<f64> {
    match value {
        Value::Int(_) | Value::Float(_) => finite(value.real().unwrap(), field),
        _ => Err(PlotError::type_error(format!("{field} must be a real number"))),
    }
}

/// An ordered numeric interval; reversed ends are swapped.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NumericRange { pub low: f64, pub high: f64 }

impl NumericRange {
    pub fn new(low: f64, high: f64) -> Result<Self> {
        let (low, high) = (finite(low, "range low")?, finite(high, "range high")?);
        Ok(if high < low { Self { low: high, high: low } } else { Self { low, high } })
    }
    pub fn from_values(low: &Value, high: &Value) -> Result<Self> {
        Self::new(finite_value(low, "range low")?, finite_value(high, "range high")?)
    }
    pub fn span(&self) -> f64 { self.high - self.low }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RectangleRange { pub x: NumericRange, pub y: NumericRange }

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CrosshairPoint { pub x: f64, pub y: f64 }

impl CrosshairPoint {
    pub fn new(x: f64, y: f64) -> Result<Self> { Ok(Self { x: finite(x, "crosshair x")?, y: finite(y, "crosshair y")? }) }
}

/// The navigation override of each axis; `None` when neither is set.
pub type Viewport = (Option<NumericRange>, Option<NumericRange>);

/// Navigation overrides only the axes explicitly selected by the operator.
pub fn normalize_viewport(value: Option<Viewport>) -> Option<Viewport> {
    value.filter(|(x, y)| x.is_some() || y.is_some())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub enum SelectorKind { XRange, Area, Crosshair, Threshold }

impl SelectorKind {
    pub const ALL: [SelectorKind; 4] = [SelectorKind::XRange, SelectorKind::Area, SelectorKind::Crosshair, SelectorKind::Threshold];
    pub fn value(self) -> &'static str {
        match self { SelectorKind::XRange => "x_range", SelectorKind::Area => "area", SelectorKind::Crosshair => "crosshair", SelectorKind::Threshold => "threshold" }
    }
    pub fn parse(value: &Value) -> Result<Self> {
        Self::ALL.into_iter().find(|kind| value.as_str() == Some(kind.value()))
            .ok_or_else(|| PlotError::value(format!("{} is not a valid SelectorKind", value.py_repr())))
    }
}

/// A selector's value; its kind is the variant.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SelectorValue { XRange(NumericRange), Area(RectangleRange), Crosshair(CrosshairPoint), Threshold(f64) }

impl SelectorValue {
    pub fn kind(&self) -> SelectorKind {
        match self { SelectorValue::XRange(_) => SelectorKind::XRange, SelectorValue::Area(_) => SelectorKind::Area,
                     SelectorValue::Crosshair(_) => SelectorKind::Crosshair, SelectorValue::Threshold(_) => SelectorKind::Threshold }
    }
}

/// One committed selector.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SelectorState {
    pub value: SelectorValue,
    pub revision: u64,
    pub facet_index: Option<u64>,
}

impl SelectorState {
    pub fn new(value: SelectorValue, revision: u64, facet_index: Option<u64>) -> Result<Self> {
        if let SelectorValue::Threshold(level) = value { finite(level, "threshold")?; }
        Ok(Self { value, revision, facet_index })
    }
    pub fn kind(&self) -> SelectorKind { self.value.kind() }
}

/// At most one state per kind, in kind order (`SelectorSnapshot.committed`).
pub fn committed_selectors(states: &[SelectorState]) -> Result<Vec<SelectorState>> {
    let mut by_kind: Vec<SelectorState> = Vec::new();
    for state in states {
        match by_kind.iter_mut().find(|held| held.kind() == state.kind()) { Some(slot) => *slot = *state, None => by_kind.push(*state) }
    }
    if by_kind.len() != states.len() { return Err(PlotError::value("committed selectors must contain at most one state per kind")); }
    by_kind.sort_by_key(SelectorState::kind);
    Ok(by_kind)
}

/// One coordinate of a classifier threshold scope.
#[derive(Clone, Debug)]
pub enum ThresholdCoordinate { Integer(i128), Real(f64), Text(Arc<str>) }

impl PartialEq for ThresholdCoordinate {
    fn eq(&self, other: &Self) -> bool { self.value() == other.value() }
}

impl ThresholdCoordinate {
    fn from_value(value: &Value) -> Result<Self> {
        match value {
            Value::Bool(_) => Err(PlotError::type_error("classifier threshold coordinates cannot be bool")),
            Value::Int(integer) => Ok(ThresholdCoordinate::Integer(*integer)),
            Value::Float(real) => Ok(ThresholdCoordinate::Real(finite(*real, "classifier threshold coordinates").map_err(|_| PlotError::value("classifier threshold coordinates must be finite"))?)),
            Value::Text(text) => Ok(ThresholdCoordinate::Text(text.clone())),
            _ => Err(PlotError::type_error("classifier threshold coordinates must be numeric or text")),
        }
    }
    pub fn value(&self) -> Value {
        match self { ThresholdCoordinate::Integer(v) => Value::Int(*v), ThresholdCoordinate::Real(v) => Value::Float(*v), ThresholdCoordinate::Text(t) => Value::Text(t.clone()) }
    }
    /// `repr(_threshold_coordinate_key(value))`'s pieces.
    fn key_parts(&self) -> (&'static str, String) {
        match self {
            ThresholdCoordinate::Integer(v) => ("integer", v.to_string()),
            ThresholdCoordinate::Real(v) => ("real", float_repr(*v)),
            ThresholdCoordinate::Text(t) => ("text", repr_str(t)),
        }
    }
    fn key_kind(&self) -> u8 { match self { ThresholdCoordinate::Integer(_) => 0, ThresholdCoordinate::Real(_) => 1, ThresholdCoordinate::Text(_) => 2 } }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ThresholdScope { pub domain: AxisDomain, pub axis_id: Arc<str>, pub coordinate: ThresholdCoordinate }

/// A population-weighted Gaussian pair, as the bimodal fit writes it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GaussianComponents { pub center: f64, pub sigma: f64, pub delta_center: f64, pub sigma_b: f64, pub ratio: f64 }

/// One authored classifier level, addressed by the coordinates it holds at.
#[derive(Clone, Debug, PartialEq)]
pub struct ThresholdTarget {
    pub value: Option<f64>,
    pub scope: Vec<ThresholdScope>,
    /// Present (possibly `None`) only when the record carried the field.
    pub gaussian_components: Option<Option<GaussianComponents>>,
}

const GAUSSIAN_FIELDS: [&str; 5] = ["center", "sigma", "delta_center", "sigma_B", "ratio"];

fn gaussian_components(value: &Value) -> Result<Option<GaussianComponents>> {
    if value.is_none() { return Ok(None); }
    let map = value.as_map().filter(|map| map.len() == 5 && GAUSSIAN_FIELDS.iter().all(|name| map.contains_key(name)))
        .ok_or_else(|| PlotError::value("classifier Gaussian component fields differ"))?;
    let mut numbers = [0.0; 5];
    for (slot, name) in numbers.iter_mut().zip(GAUSSIAN_FIELDS) {
        let item = map.get(name).unwrap();
        let number = item.real().filter(|_| item.is_real()).ok_or_else(|| PlotError::type_error("classifier Gaussian components must be finite numbers"))?;
        if !number.is_finite() { return Err(PlotError::value("classifier Gaussian components must be finite numbers")); }
        *slot = number;
    }
    let [center, sigma, delta_center, sigma_b, ratio] = numbers;
    if delta_center <= 0.0 { return Err(PlotError::value("classifier Gaussian means must be strictly ordered")); }
    if sigma.min(sigma_b) <= 0.0 { return Err(PlotError::value("classifier Gaussian sigmas must be positive")); }
    if !(0.0 < ratio && ratio < 1.0) { return Err(PlotError::value("classifier Gaussian weights must be positive")); }
    Ok(Some(GaussianComponents { center, sigma, delta_center, sigma_b, ratio }))
}

impl ThresholdTarget {
    /// `repr(_classifier_threshold_key(target))`.
    pub fn key_repr(&self) -> String {
        let items: Vec<String> = self.scope.iter().map(|item| {
            let (kind, value) = item.coordinate.key_parts();
            format!("({}, {}, {}, {})", repr_str(item.domain.value()), repr_str(&item.axis_id), repr_str(kind), value)
        }).collect();
        match items.len() { 0 => "()".into(), 1 => format!("({},)", items[0]), _ => format!("({})", items.join(", ")) }
    }
    fn same_identity(&self, other: &Self) -> bool {
        self.scope.len() == other.scope.len() && self.scope.iter().zip(&other.scope).all(|(a, b)| a.domain == b.domain
            && a.axis_id == b.axis_id && a.coordinate.key_kind() == b.coordinate.key_kind() && a.coordinate == b.coordinate)
    }
    /// The record form (`_plain`).
    pub fn wire(&self) -> Value {
        let mut map = ValueMap::from_pairs([
            ("value", self.value.map_or(Value::None, Value::Float)),
            ("scope", Value::list(self.scope.iter().map(|item| Value::Map(ValueMap::from_pairs([
                ("domain", Value::text(item.domain.value())), ("axis_id", Value::Text(item.axis_id.clone())),
                ("coordinate", item.coordinate.value())]))))),
        ]);
        if let Some(components) = &self.gaussian_components {
            map.insert("gaussian_components", components.map_or(Value::None, |g| Value::Map(ValueMap::from_pairs([
                ("center", Value::Float(g.center)), ("sigma", Value::Float(g.sigma)), ("delta_center", Value::Float(g.delta_center)),
                ("sigma_B", Value::Float(g.sigma_b)), ("ratio", Value::Float(g.ratio))]))));
        }
        Value::Map(map)
    }
}

/// Validate and copy coordinate-addressed authored classifier levels.
pub fn normalize_classifier_threshold_targets(targets: &Value) -> Result<Vec<ThresholdTarget>> {
    let Value::List(targets) = targets else { return Err(PlotError::type_error("classifier threshold targets must be a sequence")) };
    let mut normalized: Vec<ThresholdTarget> = Vec::new();
    for target in targets.iter() {
        let map = target.as_map().ok_or_else(|| PlotError::type_error("each classifier threshold target must be an object"))?;
        let required = map.contains_key("value") && map.contains_key("scope");
        if !required || map.keys().any(|key| !matches!(key, "value" | "scope" | "gaussian_components")) {
            return Err(PlotError::value("classifier threshold target fields differ"));
        }
        let threshold = match map.get("value").unwrap() {
            Value::None => None,
            value @ (Value::Int(_) | Value::Float(_)) => {
                let number = value.real().unwrap();
                if !number.is_finite() { return Err(PlotError::value("classifier threshold values must be finite numbers or None")); }
                Some(number)
            }
            _ => return Err(PlotError::type_error("classifier threshold values must be finite numbers or None")),
        };
        let Value::List(scope) = map.get("scope").unwrap() else { return Err(PlotError::type_error("classifier threshold scope must be a sequence")) };
        let mut items: Vec<ThresholdScope> = Vec::new();
        for item in scope.iter() {
            let entry = item.as_map().ok_or_else(|| PlotError::type_error("classifier threshold scope entries must be objects"))?;
            if entry.len() != 3 || !["domain", "axis_id", "coordinate"].iter().all(|name| entry.contains_key(name)) {
                return Err(PlotError::value("classifier threshold scope fields differ"));
            }
            let domain = parse_domain(&entry.get("domain").unwrap().py_str())?;
            let axis_id = entry.get("axis_id").unwrap().as_str().ok_or_else(|| PlotError::type_error("classifier threshold scope axis_id must be text"))?;
            let reference = axis_ref(domain, axis_id)?;
            if items.iter().any(|held| held.domain == reference.domain && *held.axis_id == *reference.axis_id) {
                return Err(PlotError::value("classifier threshold scope repeats an axis"));
            }
            items.push(ThresholdScope { domain: reference.domain, axis_id: Arc::from(reference.axis_id.as_str()), coordinate: ThresholdCoordinate::from_value(entry.get("coordinate").unwrap())? });
        }
        items.sort_by(|a, b| {
            let (ka, va) = a.coordinate.key_parts();
            let (kb, vb) = b.coordinate.key_parts();
            (a.domain.value(), &*a.axis_id, format!("({}, {})", repr_str(ka), va))
                .cmp(&(b.domain.value(), &*b.axis_id, format!("({}, {})", repr_str(kb), vb)))
        });
        let gaussian = map.get("gaussian_components").map(gaussian_components).transpose()?;
        let record = ThresholdTarget { value: threshold, scope: items, gaussian_components: gaussian };
        if normalized.iter().any(|held| held.same_identity(&record)) {
            return Err(PlotError::value("classifier threshold targets repeat a coordinate"));
        }
        normalized.push(record);
    }
    normalized.sort_by_key(ThresholdTarget::key_repr);
    Ok(normalized)
}

/// The record form of normalized threshold targets.
pub fn thresholds_wire(targets: &[ThresholdTarget]) -> Value { Value::list(targets.iter().map(ThresholdTarget::wire)) }

/// One atomic selector front: the committed states (one per kind, in kind
/// order) and the transient pointer candidate that takes over its kind.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SelectorSnapshot {
    pub committed: Vec<SelectorState>,
    pub candidate: Option<SelectorState>,
}

impl SelectorSnapshot {
    pub fn new(committed: &[SelectorState], candidate: Option<SelectorState>) -> Result<Self> {
        Ok(Self { committed: committed_selectors(committed)?, candidate })
    }
    /// The scene's states after the candidate takes ownership of its kind.
    pub fn states(&self) -> Vec<SelectorState> {
        let mut states: Vec<SelectorState> = self.committed.iter().filter(|state| self.candidate.is_none_or(|c| c.kind() != state.kind())).copied().collect();
        states.extend(self.candidate);
        states.sort_by_key(SelectorState::kind);
        states
    }
}
