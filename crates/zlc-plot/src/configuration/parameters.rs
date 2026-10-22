//! Typed display-parameter declarations (`zlc_plot.parameters`): how one
//! parameter is normalized and validated, and which part of a rendered view
//! an accepted change invalidates.

use std::ops::{BitAnd, BitOr, BitOrAssign};

use crate::pyfmt::format_g;
use crate::units::{resolve_unit, DEFAULT_UNITS};

use super::error::{PlotError, Result};
use super::value::{py_strip, repr_str, Value, ValueMap};

/// Independent work caused by an accepted display-parameter change.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct RenderEffect(pub u32);

impl RenderEffect {
    pub const NONE: Self = Self(0);
    pub const VIEW_PROJECTION: Self = Self(1);
    pub const PAYLOAD_PROJECTION: Self = Self(2);
    pub const BASE_GEOMETRY: Self = Self(4);
    pub const BASE_STYLE: Self = Self(8);
    pub const AXIS_TRANSFORM: Self = Self(16);
    pub const TEXT: Self = Self(32);
    pub const CHROME: Self = Self(64);
    pub const OVERLAY: Self = Self(128);
    pub const LAYOUT: Self = Self(256);
    pub const FIT_SELECTION: Self = Self(512);
    pub const INTERACTION_REPROJECT: Self = Self(1024);
    pub fn bits(self) -> u32 { self.0 }
    pub fn contains(self, other: Self) -> bool { self.0 & other.0 == other.0 }
    pub fn intersects(self, other: Self) -> bool { self.0 & other.0 != 0 }
    pub fn is_empty(self) -> bool { self.0 == 0 }
}

impl BitOr for RenderEffect { type Output = Self; fn bitor(self, other: Self) -> Self { Self(self.0 | other.0) } }
impl BitOrAssign for RenderEffect { fn bitor_assign(&mut self, other: Self) { self.0 |= other.0 } }
impl BitAnd for RenderEffect { type Output = Self; fn bitand(self, other: Self) -> Self { Self(self.0 & other.0) } }

/// The runtime type a stored value must have.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ValueType {
    Bool,
    Int,
    /// `(int, float)`.
    Number,
    Text,
}

impl ValueType {
    fn matches(self, value: &Value) -> bool {
        matches!((self, value), (ValueType::Bool, Value::Bool(_)) | (ValueType::Int, Value::Int(_))
            | (ValueType::Number, Value::Int(_) | Value::Float(_)) | (ValueType::Text, Value::Text(_)))
    }
    fn name(self) -> &'static str {
        match self { ValueType::Bool => "bool", ValueType::Int => "int", ValueType::Number => "int or float", ValueType::Text => "str" }
    }
}

/// The boundary normalizers the plot kinds declare.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Normalizer {
    None,
    /// `_label_text_or_none`: text kept as typed (no strip).
    LabelText,
    /// `_unit_text_or_none`: stripped text, blank means automatic.
    UnitText,
    /// `_pulse_time_unit`: one spelling per unit, from the registry.
    PulseTimeUnit,
    /// `_relim_mode`.
    RelimMode,
    /// `_finite_or_none`.
    FiniteOrNone,
    /// `integer(value, "value")`.
    Integer,
    /// `text(value, field="value")`: stripped, non-empty.
    NonEmptyText,
    /// `ImagePresentation(str(value)).value`.
    ImagePresentation,
    /// `_finite_number`: Python `float(value)`, finite.
    FiniteNumber,
    /// The height-bar camera's elevation clamp, [8, 80] degrees.
    CameraElevation,
    /// The height-bar camera's zoom clamp, [0.4, 6].
    CameraZoom,
}

fn text_value(value: &Value) -> Result<&str> {
    value.as_str().ok_or_else(|| PlotError::type_error("value must be text"))
}

fn finite_real(value: &Value, field: &str) -> Result<f64> {
    let number = match value {
        Value::Int(_) | Value::Float(_) => value.real().unwrap(),
        _ => return Err(PlotError::type_error(format!("{field} must be a real number"))),
    };
    if !number.is_finite() { return Err(PlotError::value(format!("{field} must be finite"))); }
    Ok(number)
}

fn finite_number(value: &Value) -> Result<f64> {
    let number = value.py_float()?;
    if !number.is_finite() { return Err(PlotError::value("camera parameters must be finite")); }
    Ok(number)
}

impl Normalizer {
    pub fn apply(self, value: &Value) -> Result<Value> {
        Ok(match self {
            Normalizer::None => value.clone(),
            Normalizer::LabelText => match value { Value::None => Value::None, other => Value::text(text_value(other)?) },
            Normalizer::UnitText => match value {
                Value::None => Value::None,
                other => { let text = py_strip(text_value(other)?); if text.is_empty() { Value::None } else { Value::text(text) } }
            },
            Normalizer::PulseTimeUnit => match Normalizer::UnitText.apply(value)? {
                Value::Text(unit) => match resolve_unit(&unit, None) { Ok(resolved) => Value::text(resolved.symbol.as_str()), Err(_) => Value::Text(unit) },
                other => other,
            },
            Normalizer::RelimMode => Value::text(py_strip(text_value(value)?).to_lowercase()),
            Normalizer::FiniteOrNone => match value { Value::None => Value::None, other => Value::Float(finite_real(other, "value")?) },
            Normalizer::Integer => match value {
                Value::Int(integer) => Value::Int(*integer),
                _ => return Err(PlotError::type_error("value must be an integer")),
            },
            Normalizer::NonEmptyText => {
                let text = py_strip(text_value(value)?);
                if text.is_empty() { return Err(PlotError::value("value must be non-empty")); }
                Value::text(text)
            }
            Normalizer::ImagePresentation => {
                let text = value.py_str();
                if text != "heatmap" && text != "height_bars" {
                    return Err(PlotError::value(format!("{} is not a valid ImagePresentation", repr_str(&text))));
                }
                Value::text(text)
            }
            Normalizer::FiniteNumber => Value::Float(finite_number(value)?),
            Normalizer::CameraElevation => Value::Float(finite_number(value)?.max(8.0).min(80.0)),
            Normalizer::CameraZoom => Value::Float(finite_number(value)?.max(0.4).min(6.0)),
        })
    }
}

/// Declaration for one typed display parameter.
#[derive(Clone, Debug)]
pub struct ParameterSpec {
    pub name: &'static str,
    pub value_type: ValueType,
    pub effects: RenderEffect,
    pub default: Value,
    pub normalizer: Normalizer,
    pub allow_none: bool,
    pub label: String,
    /// `(value, label)`: what may be stored, and how it reads on a control.
    pub choices: Vec<(Value, String)>,
    pub minimum: Option<f64>,
    pub maximum: Option<f64>,
    pub step: Option<f64>,
    /// Whether the value keeps its meaning across a change of plot identity.
    pub portable: bool,
    /// Whether an editor shows `choices` as one segmented control.
    pub segmented: bool,
}

impl ParameterSpec {
    pub(crate) fn declare(name: &'static str, value_type: ValueType, effects: RenderEffect, default: Value) -> Self {
        Self { name, value_type, effects, default, normalizer: Normalizer::None, allow_none: false,
               label: words_title(name), choices: Vec::new(), minimum: None, maximum: None, step: None,
               portable: true, segmented: false }
    }
    pub(crate) fn normalized(mut self, normalizer: Normalizer) -> Self { self.normalizer = normalizer; self }
    pub(crate) fn optional(mut self) -> Self { self.allow_none = true; self }
    pub(crate) fn labelled(mut self, label: impl Into<String>) -> Self { self.label = label.into(); self }
    pub(crate) fn not_portable(mut self) -> Self { self.portable = false; self }
    pub(crate) fn segmented(mut self) -> Self { self.segmented = true; self }
    pub(crate) fn bounded(mut self, minimum: f64, maximum: f64, step: f64) -> Self {
        self.minimum = Some(minimum); self.maximum = Some(maximum); self.step = Some(step); self
    }
    pub(crate) fn with_choices(mut self, choices: Vec<(Value, String)>) -> Self { self.choices = choices; self }
    /// The declaration's own normalization, as `__post_init__` applies it.
    pub(crate) fn finished(mut self) -> Self {
        let choices = std::mem::take(&mut self.choices);
        self.choices = choices.into_iter().map(|(value, label)| (self.prepare_value(&value).expect("declared choice"), label)).collect();
        self.label = self.label.trim().to_owned();
        self.default = self.prepare(&self.default.clone()).expect("declared default");
        self
    }

    pub fn choice_values(&self) -> impl Iterator<Item = &Value> { self.choices.iter().map(|(value, _)| value) }

    /// Normalize and validate a value for storage in a display state.
    pub fn prepare(&self, value: &Value) -> Result<Value> {
        let prepared = self.prepare_value(value)?;
        if prepared.is_none() { return Ok(prepared); }
        if !self.choices.is_empty() && !self.choice_values().any(|choice| *choice == prepared) {
            let choices = self.choice_values().map(Value::py_repr).collect::<Vec<_>>().join(", ");
            return Err(PlotError::value(format!("parameter '{}' must be one of {choices}", self.name)));
        }
        if let Some(number) = prepared.real() {
            if let Some(minimum) = self.minimum {
                if number < minimum { return Err(PlotError::value(format!("parameter '{}' must be at least {}", self.name, format_g(minimum, 6)))); }
            }
            if let Some(maximum) = self.maximum {
                if number > maximum { return Err(PlotError::value(format!("parameter '{}' must be at most {}", self.name, format_g(maximum, 6)))); }
            }
        }
        Ok(prepared)
    }

    fn prepare_value(&self, value: &Value) -> Result<Value> {
        let prepared = self.normalizer.apply(value)?;
        if prepared.is_none() {
            if self.allow_none { return Ok(Value::None); }
            return Err(PlotError::type_error(format!("parameter '{}' does not accept None", self.name)));
        }
        if !self.value_type.matches(&prepared) {
            return Err(PlotError::type_error(format!("parameter '{}' requires {}; got {}", self.name, self.value_type.name(), prepared.type_name())));
        }
        Ok(prepared)
    }
}

/// `name.replace("_", " ").strip().title()`.
fn words_title(name: &str) -> String {
    let mut result = String::new();
    let mut boundary = true;
    for character in name.replace('_', " ").trim().chars() {
        if character.is_alphabetic() {
            if boundary { result.extend(character.to_uppercase()); } else { result.extend(character.to_lowercase()); }
            boundary = false;
        } else {
            result.push(character);
            boundary = true;
        }
    }
    result
}

/// What a schema does to a whole transition beyond each value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TransitionRules {
    /// No limit pair: only authored conflicts are refused.
    Conflicts,
    /// Limit pairs: leaving Fixed clears the pair, Fixed needs both ends.
    Limits,
}

/// One immutable display state: parameter name to value, in schema order.
pub type FrozenParameters = ValueMap;

/// An immutable, ordered registry of parameters for one plot kind.
#[derive(Clone, Debug)]
pub struct ParameterSchema {
    specs: Vec<ParameterSpec>,
    rules: TransitionRules,
}

impl ParameterSchema {
    pub(crate) fn new(specs: Vec<ParameterSpec>, rules: TransitionRules) -> Self {
        for (index, spec) in specs.iter().enumerate() {
            assert!(specs[..index].iter().all(|other| other.name != spec.name), "duplicate parameter name: {:?}", spec.name);
        }
        Self { specs, rules }
    }
    pub fn len(&self) -> usize { self.specs.len() }
    pub fn is_empty(&self) -> bool { self.specs.is_empty() }
    pub fn names(&self) -> impl Iterator<Item = &'static str> + '_ { self.specs.iter().map(|spec| spec.name) }
    pub fn specs(&self) -> &[ParameterSpec] { &self.specs }
    pub fn get(&self, name: &str) -> Option<&ParameterSpec> { self.specs.iter().find(|spec| spec.name == name) }
    pub fn contains(&self, name: &str) -> bool { self.get(name).is_some() }
    pub fn rules(&self) -> TransitionRules { self.rules }
    pub fn same_names(&self, other: &ParameterSchema) -> bool { self.names().eq(other.names()) }

    /// The subset of a persisted appearance this schema declares.
    pub fn declared_subset(&self, values: &ValueMap) -> ValueMap {
        values.iter().filter(|(name, _)| self.contains(name)).map(|(name, value)| (name, value.clone())).collect()
    }

    /// Every update prepared by its declaration; an unknown name is a KeyError.
    pub fn prepare_updates(&self, updates: &ValueMap) -> Result<ValueMap> {
        let unknown: Vec<String> = updates.keys().filter(|name| !self.contains(name)).map(repr_str).collect();
        if !unknown.is_empty() { return Err(PlotError::key(repr_str(&format!("unknown display parameter(s): {}", unknown.join(", "))))); }
        let mut prepared = ValueMap::new();
        for (name, value) in updates.iter() { prepared.insert(name, self.get(name).unwrap().prepare(value)?); }
        Ok(prepared)
    }

    pub fn defaults(&self) -> ValueMap { self.specs.iter().map(|spec| (spec.name, spec.default.clone())).collect() }

    pub fn initial_values(&self, overrides: Option<&ValueMap>) -> Result<FrozenParameters> {
        self.prepare_transition(&self.defaults(), overrides.unwrap_or(&ValueMap::new()))
    }

    /// Normalize updates and validate the resulting complete state.
    pub fn prepare_transition(&self, current: &ValueMap, updates: &ValueMap) -> Result<FrozenParameters> {
        self.transition_prepared(current, &self.prepare_updates(updates)?)
    }

    pub(crate) fn transition_prepared(&self, current: &ValueMap, prepared: &ValueMap) -> Result<FrozenParameters> {
        let mut prepared = prepared.clone();
        if self.rules == TransitionRules::Limits {
            let mut generated = ValueMap::new();
            for (mode, low, high) in LIMIT_PAIRS {
                let Some(updated) = prepared.get(mode) else { continue };
                if updated.as_str() == Some(RelimModeName::FIXED) { continue; }
                if current.contains_key(low) && current.contains_key(high) {
                    for name in [low, high] {
                        if !prepared.contains_key(name) && !generated.contains_key(name) { generated.insert(name, Value::None); }
                    }
                }
            }
            if !generated.is_empty() { prepared.update(&self.prepare_updates(&generated)?); }
        }
        let mut values = current.clone();
        values.update(&prepared);
        self.validate_full_state(&values)?;
        Ok(values)
    }

    fn validate_full_state(&self, values: &ValueMap) -> Result<()> {
        validate_authored_conflicts(values)?;
        if self.rules == TransitionRules::Limits {
            for (mode, low, high) in LIMIT_PAIRS {
                let (Some(low_value), Some(high_value)) = (values.get(low), values.get(high)) else { continue };
                if values.get(mode).and_then(Value::as_str) == Some(RelimModeName::FIXED) && (low_value.is_none() || high_value.is_none()) {
                    return Err(PlotError::value(format!("fixed {mode} requires {low} and {high}")));
                }
            }
        }
        Ok(())
    }

    pub fn effects_for<'a>(&self, names: impl IntoIterator<Item = &'a str>) -> Result<RenderEffect> {
        let mut combined = RenderEffect::NONE;
        let mut unknown = Vec::new();
        for name in names {
            match self.get(name) { Some(spec) => combined |= spec.effects, None => unknown.push(repr_str(name)) }
        }
        if !unknown.is_empty() { return Err(PlotError::key(repr_str(&format!("unknown display parameter(s): {}", unknown.join(", "))))); }
        Ok(combined)
    }
}

struct RelimModeName;
impl RelimModeName { const FIXED: &'static str = "fixed"; }

/// Every authored limit pair, with the mode that governs it.
pub const LIMIT_PAIRS: [(&str, &str, &str); 3] = [
    ("relim_mode", "color_min", "color_max"),
    ("relim_mode", "y_min", "y_max"),
    ("x_relim_mode", "x_min", "x_max"),
];

/// Every relim mode name there is.
pub const LIMIT_MODE_NAMES: [&str; 2] = ["relim_mode", "x_relim_mode"];

/// The (mode, low, high) triple this authored limit belongs to.
pub fn limit_pair_for(name: &str) -> Option<(&'static str, &'static str, &'static str)> {
    LIMIT_PAIRS.into_iter().find(|(_, low, high)| *low == name || *high == name)
}

/// The display states no host could ever accept: an inverted pair, a
/// non-positive fixed log floor, a cumulative threshold classifier.
pub fn validate_authored_conflicts(values: &ValueMap) -> Result<()> {
    let mode = values.get("relim_mode");
    for (_mode, low, high) in LIMIT_PAIRS {
        let (Some(low_value), Some(high_value)) = (values.get(low), values.get(high)) else { continue };
        if let (Some(a), Some(b)) = (low_value.real(), high_value.real()) {
            if a >= b { return Err(PlotError::value(format!("{low} must be smaller than {high}"))); }
        }
    }
    if matches!(values.get("log_y"), Some(Value::Bool(true))) && mode.and_then(Value::as_str) == Some(RelimModeName::FIXED) {
        if let Some(floor) = values.get("y_min").and_then(Value::real) {
            if floor <= 0.0 { return Err(PlotError::value("log count limits require a positive y_min")); }
        }
    }
    if matches!(values.get("threshold_classifier"), Some(Value::Bool(true))) && matches!(values.get("cumulative"), Some(Value::Bool(true))) {
        return Err(PlotError::value("threshold classifier requires a non-cumulative histogram"));
    }
    Ok(())
}

/// Shared `(value, label)` constructors for declarations.
pub(crate) fn spelled_choices<S: AsRef<str>>(values: impl IntoIterator<Item = S>) -> Vec<(Value, String)> {
    values.into_iter().map(|value| (Value::text(value.as_ref()), value.as_ref().to_owned())).collect()
}

/// A snake_case name as the words a control shows, sentence case.
pub(crate) fn words(name: &str) -> String {
    let text = name.replace('_', " ");
    let mut characters = text.chars();
    match characters.next() {
        Some(first) => first.to_uppercase().collect::<String>() + characters.as_str(),
        None => String::new(),
    }
}

pub(crate) fn enum_choices(values: &[&str]) -> Vec<(Value, String)> {
    values.iter().map(|value| (Value::text(*value), words(value))).collect()
}

/// The pulse timeline's time-unit ladder.
pub(crate) fn pulse_time_choices() -> Vec<(Value, String)> {
    let seconds = DEFAULT_UNITS.resolve("s").expect("seconds");
    spelled_choices(DEFAULT_UNITS.display_choices(&seconds).expect("time ladder"))
}
