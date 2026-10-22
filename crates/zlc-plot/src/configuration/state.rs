//! Immutable display revisions (`zlc_plot.state`) and the portable
//! interaction identities a committed gesture leaves behind.

use std::collections::BTreeSet;
use std::sync::Arc;

use super::error::{PlotError, Result};
use super::parameters::{FrozenParameters, RenderEffect};
use super::value::{Value, ValueMap};

/// One series identity: `(domain, axis_id, canonical coordinate repr)` parts.
#[derive(Clone, Debug, PartialEq)]
pub struct SeriesTarget {
    pub key: Vec<[Arc<str>; 3]>,
    pub facet_index: Option<i128>,
}

/// The transient readout frozen with a saved picture.
#[derive(Clone, Debug, PartialEq)]
pub struct SeriesReadout {
    pub target: SeriesTarget,
    pub label: Arc<str>,
    /// `"hover"` or `"locked"`.
    pub mode: Arc<str>,
}

/// Committed gestures (`{"series_lock": ...}`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Interaction { pub series_lock: Option<SeriesTarget> }

/// The presented readout (`{"series_readout": ...}`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Presentation { pub series_readout: Option<SeriesReadout> }

/// A partial interaction mapping: `None` leaves the field as it is.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct InteractionUpdate { pub series_lock: Option<Option<SeriesTarget>> }

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PresentationUpdate { pub series_readout: Option<Option<SeriesReadout>> }

/// Python iteration of a wire value (`tuple(value)`).
pub(crate) fn py_iter(value: &Value) -> Result<Vec<Value>> {
    match value {
        Value::List(items) => Ok(items.to_vec()),
        Value::Text(text) => Ok(text.chars().map(|c| Value::text(c.to_string())).collect()),
        Value::Map(map) => Ok(map.keys().map(Value::text).collect()),
        other => Err(PlotError::type_error(format!("'{}' object is not iterable", other.type_name()))),
    }
}

fn series_target(value: &Value, readout: bool) -> Result<(SeriesTarget, Option<(Arc<str>, Arc<str>)>)> {
    let mismatch = || PlotError::value("series identity fields differ from the interaction contract");
    let map = value.as_map().ok_or_else(mismatch)?;
    let expected: &[&str] = if readout { &["key", "facet_index", "label", "mode"] } else { &["key", "facet_index"] };
    if map.len() != expected.len() || !expected.iter().all(|name| map.contains_key(name)) { return Err(mismatch()); }
    let mut key = Vec::new();
    let mut valid = true;
    for part in py_iter(map.get("key").unwrap())? {
        let items = py_iter(&part)?;
        if items.len() != 3 || !items.iter().all(|item| item.as_str().is_some()) { valid = false; continue; }
        key.push([0, 1, 2].map(|index| Arc::from(items[index].as_str().unwrap())));
    }
    if key.is_empty() || !valid {
        return Err(PlotError::value("series key must contain domain, axis and canonical coordinate identities"));
    }
    let facet_index = match map.get("facet_index").unwrap() {
        Value::None => None,
        Value::Int(index) if *index >= 0 => Some(*index),
        _ => return Err(PlotError::value("series facet_index must be a nonnegative integer or None")),
    };
    let target = SeriesTarget { key, facet_index };
    if !readout { return Ok((target, None)); }
    let mode = map.get("mode").unwrap();
    let label = map.get("label").unwrap();
    match (mode.as_str(), label.as_str()) {
        (Some(mode @ ("hover" | "locked")), Some(label)) => Ok((target, Some((Arc::from(label), Arc::from(mode))))),
        _ => Err(PlotError::value("series readout requires a label and hover/locked mode")),
    }
}

impl InteractionUpdate {
    /// `normalize_interaction(values)` of a wire mapping.
    pub fn from_value(value: &Value) -> Result<Self> {
        let map = value.as_map().filter(|map| map.keys().all(|key| key == "series_lock"))
            .ok_or_else(|| PlotError::value("interaction fields must be series_lock"))?;
        Ok(Self { series_lock: match map.get("series_lock") {
            None => None,
            Some(Value::None) => Some(None),
            Some(target) => Some(Some(series_target(target, false)?.0)),
        } })
    }
    pub fn applied_to(&self, current: &Interaction) -> Interaction {
        Interaction { series_lock: self.series_lock.clone().unwrap_or_else(|| current.series_lock.clone()) }
    }
}

impl PresentationUpdate {
    pub fn from_value(value: &Value) -> Result<Self> {
        let map = value.as_map().filter(|map| map.keys().all(|key| key == "series_readout"))
            .ok_or_else(|| PlotError::value("presentation fields must be series_readout"))?;
        Ok(Self { series_readout: match map.get("series_readout") {
            None => None,
            Some(Value::None) => Some(None),
            Some(target) => {
                let (target, readout) = series_target(target, true)?;
                let (label, mode) = readout.unwrap();
                Some(Some(SeriesReadout { target, label, mode }))
            }
        } })
    }
    pub fn applied_to(&self, current: &Presentation) -> Presentation {
        Presentation { series_readout: self.series_readout.clone().unwrap_or_else(|| current.series_readout.clone()) }
    }
}

impl SeriesTarget {
    fn wire(&self) -> ValueMap {
        ValueMap::from_pairs([
            ("key", Value::list(self.key.iter().map(|part| Value::list(part.iter().map(|item| Value::Text(item.clone())))))),
            ("facet_index", self.facet_index.map_or(Value::None, Value::Int)),
        ])
    }
}

impl Interaction {
    /// The record form (`_plain(normalize_interaction(...))`).
    pub fn wire(&self) -> Value {
        Value::Map(ValueMap::from_pairs([("series_lock", self.series_lock.as_ref().map_or(Value::None, |target| Value::Map(target.wire())))]))
    }
}

impl Presentation {
    pub fn wire(&self) -> Value {
        Value::Map(ValueMap::from_pairs([("series_readout", self.series_readout.as_ref().map_or(Value::None, |readout| {
            let mut map = readout.target.wire();
            map.insert("label", Value::Text(readout.label.clone()));
            map.insert("mode", Value::Text(readout.mode.clone()));
            Value::Map(map)
        }))]))
    }
}

/// One immutable, monotonically numbered display configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct DisplayState {
    pub revision: u64,
    pub values: FrozenParameters,
    pub changed_names: BTreeSet<&'static str>,
    pub effects: RenderEffect,
    pub interaction: Interaction,
}

impl DisplayState {
    pub fn get(&self, name: &str) -> Option<&Value> { self.values.get(name) }
}

/// The owner of the current [`DisplayState`] (`DisplayStateStore`): a
/// prepared transition replaces the complete immutable state at once, so
/// readers may keep an earlier revision.  Synchronization is the owner's.
pub struct DisplayStateStore {
    schema: std::sync::Arc<super::parameters::ParameterSchema>,
    state: std::sync::Arc<DisplayState>,
}

impl DisplayStateStore {
    pub fn new(schema: std::sync::Arc<super::parameters::ParameterSchema>, initial: Option<&ValueMap>, initial_revision: u64,
               initial_effects: RenderEffect, initial_interaction: Interaction) -> Result<Self> {
        let values = schema.initial_values(initial)?;
        let state = DisplayState { revision: initial_revision, values, changed_names: schema.names().collect(),
                                   effects: initial_effects, interaction: initial_interaction };
        Ok(Self { schema, state: std::sync::Arc::new(state) })
    }
    pub fn state(&self) -> &std::sync::Arc<DisplayState> { &self.state }
    pub fn schema(&self) -> &std::sync::Arc<super::parameters::ParameterSchema> { &self.schema }

    /// Commit a transition this store's schema prepared from `previous`.
    pub fn commit_prepared(&mut self, previous: &std::sync::Arc<DisplayState>, candidate: FrozenParameters) -> Result<std::sync::Arc<DisplayState>> {
        if !std::sync::Arc::ptr_eq(&self.state, previous) { return Err(PlotError::runtime("display state changed before prepared commit")); }
        let changed: BTreeSet<&'static str> = self.schema.names().filter(|name| previous.values.get(name) != candidate.get(name)).collect();
        if changed.is_empty() { return Ok(previous.clone()); }
        let effects = self.schema.effects_for(changed.iter().copied())?;
        self.state = std::sync::Arc::new(DisplayState { revision: previous.revision + 1, values: candidate, changed_names: changed,
                                                        effects, interaction: previous.interaction.clone() });
        Ok(self.state.clone())
    }

    /// Commit a gesture's portable identities; an unchanged one is no revision.
    pub fn commit_interaction(&mut self, update: &InteractionUpdate) -> std::sync::Arc<DisplayState> {
        let current = self.state.clone();
        let wanted = update.applied_to(&current.interaction);
        if wanted == current.interaction { return current; }
        self.state = std::sync::Arc::new(DisplayState { revision: current.revision + 1, values: current.values.clone(),
                                                        changed_names: BTreeSet::new(), effects: RenderEffect::OVERLAY, interaction: wanted });
        self.state.clone()
    }

    /// Roll back one unpresented transition under compare-and-swap.
    pub fn restore_prepared(&mut self, current: &std::sync::Arc<DisplayState>, previous: std::sync::Arc<DisplayState>) -> Result<()> {
        if !std::sync::Arc::ptr_eq(&self.state, current) { return Err(PlotError::runtime("display state changed before rollback")); }
        self.state = previous;
        Ok(())
    }
}
