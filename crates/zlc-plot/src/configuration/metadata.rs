//! Schema-only metadata of a configuration (`_configuration_metadata`) and
//! its pure display-parameter transitions (`_configuration_parameters`).

use std::sync::Arc;

use crate::units::{resolve_unit, Unit, UnitRegistry, DEFAULT_UNITS};

use super::data_contract::{resolve_axis, PlotSchema};
use super::display::{parameter_schema_for, FACET_FIT_PARAMETER};
use super::error::{PlotError, Result};
use super::fit::{FitLayer, FitModelInfo, FitQuantity, FitUnits, UnitRelation};
use super::handlers::{handler, label_roles, LabelRole};
use super::kinds::{AxisRef, PlotKind};
use super::ordered::OrderedMap;
use super::parameters::{limit_pair_for, ParameterSchema, RenderEffect, LIMIT_PAIRS};
use super::spec::{LabelSlot, PlotSpec};
use super::state::DisplayState;
use super::value::{repr_str, Value, ValueMap};
use crate::style::PlotStyle;

fn registry_or_default(registry: Option<&UnitRegistry>) -> &UnitRegistry { registry.unwrap_or(&DEFAULT_UNITS) }

/// The canonical unit behind each unit parameter, in parameter order:
/// value, x, y, facet.
pub fn unit_parameter_units(schema: Option<&PlotSchema>, spec: &PlotSpec, registry: Option<&UnitRegistry>) -> Result<OrderedMap<&'static str, Arc<Unit>>> {
    let mut units = OrderedMap::new();
    let Some(schema) = schema else { return Ok(units) };
    let registry = Some(registry_or_default(registry));
    units.insert("value_display_unit", schema.canonical_value_unit(registry)?);
    let semantic = spec.semantic();
    if let Some(x) = semantic.x() { units.insert("x_display_unit", resolve_axis(schema, x)?.canonical_unit(registry)?); }
    if let Some(y) = semantic.y() { units.insert("y_display_unit", resolve_axis(schema, y)?.canonical_unit(registry)?); }
    if let Some(facet) = spec.facet() { units.insert("facet_display_unit", resolve_axis(schema, facet)?.canonical_unit(registry)?); }
    Ok(units)
}

fn slot_roles(spec: &PlotSpec, schema: Option<&PlotSchema>) -> Result<Vec<(LabelSlot, LabelRole)>> {
    let history = match schema { Some(schema) => schema.history()?, None => None };
    label_roles(&spec.semantic_spec(), history.as_deref())
}

/// The plotted unit vocabulary (`configuration_quantities`): "value", and
/// "x"/"y" where the kind plots them.
pub fn configuration_quantities(schema: Option<&PlotSchema>, spec: &PlotSpec, state: &DisplayState, registry: Option<&UnitRegistry>) -> Result<OrderedMap<&'static str, FitQuantity>> {
    let mut result = OrderedMap::new();
    let Some(schema) = schema else { return Ok(result) };
    let registry = Some(registry_or_default(registry));
    let quantity = |canonical: Arc<Unit>, name: &str| -> Result<FitQuantity> {
        let display = match state.values.get(name) {
            None | Some(Value::None) => canonical.clone(),
            Some(selected) => resolve_unit(&selected.py_str(), registry)?,
        };
        Ok(FitQuantity { canonical, display })
    };
    result.insert("value", quantity(schema.canonical_value_unit(registry)?, "value_display_unit")?);
    let semantic = spec.semantic();
    for (slot, role) in slot_roles(spec, Some(schema))? {
        let key = match slot { LabelSlot::X => "x", LabelSlot::Y => "y", _ => continue };
        let entry = match role {
            LabelRole::Value => result.get(&"value").cloned().unwrap(),
            LabelRole::Axis(reference) => quantity(resolve_axis(schema, &reference)?.canonical_unit(registry)?, if key == "x" { "x_display_unit" } else { "y_display_unit" })?,
            LabelRole::X => quantity(resolve_axis(schema, semantic.x().expect("rolling x"))?.canonical_unit(registry)?, "x_display_unit")?,
            LabelRole::Repeat | LabelRole::Count => {
                let unit = resolve_unit("1", registry)?;
                FitQuantity { canonical: unit.clone(), display: unit }
            }
            _ => continue,
        };
        result.insert(key, entry);
    }
    Ok(result)
}

/// The fit layer's unit vocabulary before any sample exists.
pub fn configuration_fit_units(schema: Option<&PlotSchema>, spec: &PlotSpec, state: &DisplayState, registry: Option<&Arc<UnitRegistry>>) -> Result<FitUnits> {
    let quantities = configuration_quantities(schema, spec, state, registry.map(|r| &**r))?;
    let roles = slot_roles(spec, schema)?;
    Ok(FitUnits {
        registry: registry.cloned(),
        value: quantities.get(&"value").cloned(), axis_0: quantities.get(&"x").cloned(), axis_1: quantities.get(&"y").cloned(),
        histogram: spec.semantic().kind() == PlotKind::Histogram,
        rolling_primary: roles.iter().any(|(slot, role)| *slot == LabelSlot::X && *role == LabelRole::Repeat),
    })
}

/// Whether the plotted independent axes can enter this model's units.
pub fn fit_model_units_compatible(model: &FitModelInfo, axis_0: &Arc<Unit>, axis_1: Option<&Arc<Unit>>) -> bool {
    let sources: Vec<Option<&Arc<Unit>>> = if model.independent_arity == 1 { vec![Some(axis_0)] } else { vec![Some(axis_0), axis_1] };
    if sources.len() != model.coordinate_relations.len() { return false; }
    sources.iter().zip(&model.coordinate_relations).all(|(source, relation)| {
        let target = match relation { UnitRelation::Axis0 => Some(axis_0), UnitRelation::Axis1 => axis_1 };
        matches!((source, target), (Some(source), Some(target)) if source.compatible_with(target))
    })
}

/// Fit catalogue, editor choice overrides and automatic unit values.
#[derive(Clone, Debug, Default)]
pub struct ConfigurationMetadata {
    pub fit_models: Vec<Arc<FitModelInfo>>,
    pub parameter_choices: OrderedMap<&'static str, Vec<Value>>,
    pub automatic_values: OrderedMap<&'static str, Value>,
}

pub fn configuration_metadata(schema: Option<&PlotSchema>, spec: &PlotSpec, style: &PlotStyle, fit: &dyn FitLayer,
                              registry: Option<&UnitRegistry>, request_model: Option<&FitModelInfo>) -> Result<ConfigurationMetadata> {
    let parameters = parameter_schema_for(spec, style);
    let mut choices: OrderedMap<&'static str, Vec<Value>> = OrderedMap::new();
    if let PlotSpec::PulseTimeline(_) = spec {
        if let Some(unit) = parameters.get("x_display_unit") {
            choices.insert("x_display_unit", unit.choice_values().map(|value| Value::text(value.py_str())).collect());
        }
        return Ok(ConfigurationMetadata { fit_models: Vec::new(), parameter_choices: choices, automatic_values: OrderedMap::new() });
    }
    let schema = schema.ok_or_else(|| PlotError::type_error("data-backed configuration metadata requires DatasetSchema"))?;
    let registry = registry_or_default(registry);
    let semantic = spec.semantic();
    let cell_handler = handler(semantic.kind());
    let value = schema.canonical_value_unit(Some(registry))?;
    let roles = slot_roles(spec, Some(schema))?;
    let axis_unit = |role: Option<&LabelRole>| -> Result<Option<Arc<Unit>>> {
        Ok(match role {
            Some(LabelRole::Axis(reference)) => Some(resolve_axis(schema, reference)?.canonical_unit(Some(registry))?),
            Some(LabelRole::Value) => Some(value.clone()),
            Some(LabelRole::Repeat) => Some(resolve_unit("1", Some(registry))?),
            Some(LabelRole::X) => Some(resolve_axis(schema, semantic.x().expect("rolling x"))?.canonical_unit(Some(registry))?),
            _ => None,
        })
    };
    let role_of = |slot: LabelSlot| roles.iter().find(|(held, _)| *held == slot).map(|(_, role)| role);
    let axis_0 = axis_unit(role_of(LabelSlot::X))?;
    let axis_1 = match role_of(LabelSlot::Y) { Some(role @ LabelRole::Axis(_)) => axis_unit(Some(role))?, _ => None };
    let models: Vec<Arc<FitModelInfo>> = match (cell_handler.fit_target, &axis_0) {
        (Some(target), Some(axis_0)) => fit.models_for(target).into_iter()
            .filter(|model| fit_model_units_compatible(model, axis_0, axis_1.as_ref())).collect(),
        _ => Vec::new(),
    };
    if parameters.contains(FACET_FIT_PARAMETER) {
        let mut names: Vec<Value> = vec![Value::text("model headline")];
        let mut seen: Vec<&str> = Vec::new();
        let sources: Vec<&FitModelInfo> = match request_model { Some(model) => vec![model], None => models.iter().map(|model| &**model).collect() };
        for model in sources {
            for name in &model.parameter_names {
                if !seen.contains(&name.as_str()) { seen.push(name); names.push(Value::text(name.as_str())); }
            }
        }
        choices.insert(FACET_FIT_PARAMETER, names);
    }
    let mut automatic = OrderedMap::new();
    for (name, unit) in unit_parameter_units(Some(schema), spec, Some(registry))?.iter() {
        if parameters.contains(name) {
            choices.insert(name, registry.display_choices(unit)?.into_iter().map(Value::from).collect());
            automatic.insert(*name, Value::text(unit.symbol.as_str()));
        }
    }
    Ok(ConfigurationMetadata { fit_models: models, parameter_choices: choices, automatic_values: automatic })
}

/// Carry values the new schema admits, trying complete limit pairs first.
pub fn retained_parameters(old: &ValueMap, schema: &ParameterSchema) -> Result<ValueMap> {
    let mut current = schema.initial_values(None)?;
    let mut retained = ValueMap::new();
    for group in LIMIT_PAIRS {
        let names = [group.0, group.1, group.2];
        if names.iter().any(|name| !schema.contains(name) || !old.contains_key(name)) { continue; }
        let updates: ValueMap = names.iter().map(|name| (*name, old.get(name).unwrap().clone())).collect();
        match schema.prepare_transition(&current, &updates) {
            Ok(candidate) => {
                for name in names { retained.insert(name, candidate.get(name).unwrap().clone()); }
                current = candidate;
            }
            Err(error) if error.is_type_value_or_key() => continue,
            Err(error) => return Err(error),
        }
    }
    for name in schema.names() {
        let Some(value) = old.get(name) else { continue };
        if retained.contains_key(name) { continue; }
        match schema.prepare_transition(&current, &ValueMap::from_pairs([(name, value.clone())])) {
            Ok(candidate) => { retained.insert(name, candidate.get(name).unwrap().clone()); current = candidate; }
            Err(error) if error.is_type_value_or_key() => continue,
            Err(error) => return Err(error),
        }
    }
    Ok(retained)
}

/// Coordinate contracts that let a viewport survive a semantic edit.
#[derive(Clone, Debug, PartialEq)]
pub struct ViewportRoles {
    kind: PlotKind,
    x: Option<AxisRef>,
    y: Option<AxisRef>,
    facet: Option<AxisRef>,
    units: Option<(Option<Arc<str>>, Vec<(AxisRef, Option<Arc<str>>, Option<Arc<str>>)>)>,
}

pub fn viewport_roles(spec: &PlotSpec, schema: Option<&PlotSchema>) -> Result<ViewportRoles> {
    let semantic = spec.semantic();
    let (x, y, facet) = (semantic.x().cloned(), semantic.y().cloned(), spec.facet().cloned());
    let units = match schema {
        None => None,
        Some(schema) => {
            let mut contracts = Vec::new();
            for reference in [&x, &y, &facet].into_iter().flatten() {
                let axis = resolve_axis(schema, reference)?;
                contracts.push((reference.clone(), axis.unit_annotation().map(Arc::from), axis.coordinate_frame().map(Arc::from)));
            }
            Some((schema.value_unit().map(Arc::from), contracts))
        }
    };
    Ok(ViewportRoles { kind: spec.kind(), x, y, facet, units })
}

fn materialize_fixed_limits(schema: &ParameterSchema, prepared: &mut ValueMap, previous: &DisplayState, fixed_limits: &OrderedMap<String, (f64, f64)>) -> Result<()> {
    let mut candidate = previous.values.clone();
    candidate.update(prepared);
    for (mode, low, high) in LIMIT_PAIRS {
        if ![mode, low, high].iter().all(|name| schema.contains(name)) { continue; }
        if candidate.get(mode).and_then(Value::as_str) != Some("fixed") { continue; }
        let (low_value, high_value) = (candidate.get(low).unwrap(), candidate.get(high).unwrap());
        if !low_value.is_none() && !high_value.is_none() { continue; }
        let (captured_low, captured_high) = *fixed_limits.get(&low.to_owned())
            .ok_or_else(|| PlotError::value(format!("fixed {mode} requires captured limits for {}", repr_str(low))))?;
        if low_value.is_none() { prepared.insert(low, Value::Float(captured_low)); }
        if high_value.is_none() { prepared.insert(high, Value::Float(captured_high)); }
    }
    Ok(())
}

fn prepare_value_unit_ranges(schema: &ParameterSchema, prepared: &mut ValueMap, previous: &DisplayState, authored_names: &[String],
                             spec: &PlotSpec, units: &OrderedMap<&'static str, Arc<Unit>>, registry: Option<&UnitRegistry>) -> Result<()> {
    let Some(canonical) = units.get(&"value_display_unit") else { return Ok(()) };
    let Some(selected) = prepared.get("value_display_unit").cloned() else { return Ok(()) };
    if selected == previous.values.get("value_display_unit").cloned().unwrap_or(Value::None) { return Ok(()); }
    let value_low = match spec.semantic().kind() {
        PlotKind::Curve | PlotKind::Rolling => "y_min",
        PlotKind::Image => "color_min",
        PlotKind::Histogram => "x_min",
        _ => return Ok(()),
    };
    let (mode, low, high) = limit_pair_for(value_low).expect("declared pair");
    if prepared.get(mode).is_some_and(|value| value.as_str() != Some("fixed")) { return Ok(()); }
    let target = match &selected { Value::None => canonical.clone(), unit => resolve_unit(&unit.py_str(), registry)? };
    let source = match previous.values.get("value_display_unit") { None | Some(Value::None) => canonical.clone(), Some(unit) => resolve_unit(&unit.py_str(), registry)? };
    for name in [low, high] {
        if !schema.contains(name) || authored_names.iter().any(|authored| authored == name) { continue; }
        let current = prepared.get(name).or_else(|| previous.values.get(name)).cloned().unwrap_or(Value::None);
        let Some(number) = current.real() else { continue };
        let converted = source.convert_value_to(number, &target)?;
        if value_low != "color_min" && !converted.is_finite() {
            return Err(PlotError::value(format!("{name} {} {} has no value in {}: set the value limits or leave Fixed",
                crate::pyfmt::format_g(number, 6), source.symbol, target.symbol)));
        }
        prepared.insert(name, Value::Float(converted));
    }
    Ok(())
}

/// One parameter transaction's candidate state, changed names and effects.
pub struct ParameterResolution {
    pub values: ValueMap,
    pub changed: std::collections::BTreeSet<&'static str>,
    pub effects: RenderEffect,
}

#[allow(clippy::too_many_arguments)]
pub fn resolve_parameters(schema: &ParameterSchema, previous: &DisplayState, values: &ValueMap, authored_values: Option<&ValueMap>,
                          fixed_limits: &OrderedMap<String, (f64, f64)>, spec: &PlotSpec, dataset: Option<&PlotSchema>,
                          registry: Option<&UnitRegistry>) -> Result<ParameterResolution> {
    let mut prepared = schema.prepare_updates(values)?;
    let authored = match authored_values { None => prepared.clone(), Some(authored) => schema.prepare_updates(authored)? };
    let authored_names: Vec<String> = authored.iter().filter(|(_, value)| !value.is_none()).map(|(name, _)| name.to_owned()).collect();
    let units = unit_parameter_units(dataset, spec, registry)?;
    for (name, canonical) in units.iter() {
        let Some(selected) = prepared.get(name).filter(|value| !value.is_none()) else { continue };
        let target = resolve_unit(&selected.py_str(), registry)?;
        if !canonical.compatible_with(&target) {
            return Err(PlotError::value(format!("{name} {} is incompatible with {}", repr_str(&target.symbol), repr_str(&canonical.symbol))));
        }
    }
    materialize_fixed_limits(schema, &mut prepared, previous, fixed_limits)?;
    if authored_values.is_some() {
        let mut transition = authored.clone();
        for (mode, low, high) in LIMIT_PAIRS {
            if ![mode, low, high].iter().all(|name| schema.contains(name)) { continue; }
            let current_mode = prepared.get(mode).or_else(|| previous.values.get(mode)).cloned().unwrap_or(Value::None);
            if current_mode.as_str() != Some("fixed") { continue; }
            for name in [low, high] {
                if transition.get(name).is_none_or(Value::is_none) {
                    let value = prepared.get(name).or_else(|| previous.values.get(name)).cloned().unwrap_or(Value::None);
                    transition.insert(name, value);
                }
            }
        }
        let authored_candidate = schema.transition_prepared(&previous.values, &transition)?;
        for name in schema.names() {
            let value = authored_candidate.get(name).unwrap();
            if transition.contains_key(name) || previous.values.get(name) != Some(value) { prepared.insert(name, value.clone()); }
        }
    }
    prepare_value_unit_ranges(schema, &mut prepared, previous, &authored_names, spec, &units, registry)?;
    let candidate = schema.transition_prepared(&previous.values, &prepared)?;
    let changed: std::collections::BTreeSet<&'static str> = schema.names().filter(|name| previous.values.get(name) != candidate.get(name)).collect();
    let effects = if changed.is_empty() { RenderEffect::NONE } else { schema.effects_for(changed.iter().copied())? };
    Ok(ParameterResolution { values: candidate, changed, effects })
}
