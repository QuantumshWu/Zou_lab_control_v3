//! The editor contract: parameter controls (`zlc_plot.ui`) and the panel
//! description a desktop UI receives (`zlc_science.desktop.FigureDocument
//! ._describe` for a resolved configuration).

use serde_json::{json, Value as Json};

use super::data_contract::schema_structure;
use super::error::{PlotError, Result};
use super::ordered::OrderedMap;
use super::parameters::{limit_pair_for, ParameterSchema, RenderEffect, ValueType};
use super::resolve::ResolvedConfiguration;
use super::value::{repr_str, Value, ValueMap};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ControlKind { Boolean, Integer, Number, Text, Choice }

impl ControlKind {
    pub fn value(self) -> &'static str {
        match self { ControlKind::Boolean => "boolean", ControlKind::Integer => "integer", ControlKind::Number => "number",
                     ControlKind::Text => "text", ControlKind::Choice => "choice" }
    }
}

/// One toolkit-independent parameter editor description.
#[derive(Clone, Debug)]
pub struct ParameterControl {
    pub name: &'static str,
    pub label: String,
    pub kind: ControlKind,
    pub value: Value,
    pub allow_none: bool,
    pub choices: Vec<(Value, String)>,
    pub minimum: Option<f64>,
    pub maximum: Option<f64>,
    pub step: Option<f64>,
    pub effects: RenderEffect,
    pub automatic: bool,
    pub automatic_value: Value,
    pub unavailable_reason: String,
    pub segmented: bool,
}

/// Project one canonical schema/state pair into ordered UI controls.
pub fn parameter_controls(schema: &ParameterSchema, values: &ValueMap, choice_overrides: &OrderedMap<&'static str, Vec<Value>>,
                          automatic_values: &OrderedMap<&'static str, Value>) -> Result<Vec<ParameterControl>> {
    let unknown: Vec<String> = choice_overrides.keys().filter(|name| !schema.contains(name)).map(|name| repr_str(name)).collect();
    if !unknown.is_empty() { return Err(PlotError::key(repr_str(&format!("choice override refers to unknown parameter(s): {}", unknown.join(", "))))); }
    let unknown: Vec<String> = automatic_values.keys().filter(|name| !schema.contains(name)).map(|name| repr_str(name)).collect();
    if !unknown.is_empty() { return Err(PlotError::key(repr_str(&format!("automatic value refers to unknown parameter(s): {}", unknown.join(", "))))); }
    let mut result = Vec::new();
    for spec in schema.specs() {
        let value = values.get(spec.name).ok_or_else(|| PlotError::key(format!("display state is missing parameter {}", repr_str(spec.name))))?;
        let choices = match choice_overrides.get(&spec.name) {
            None => spec.choices.clone(),
            Some(overridden) => overridden.iter().map(|value| (value.clone(), value.py_str())).collect(),
        };
        let kind = if !choices.is_empty() { ControlKind::Choice } else {
            match spec.value_type { ValueType::Bool => ControlKind::Boolean, ValueType::Int => ControlKind::Integer,
                                    ValueType::Number => ControlKind::Number, ValueType::Text => ControlKind::Text }
        };
        let pair = limit_pair_for(spec.name);
        let limit_field = pair.is_some_and(|(mode, _, _)| values.contains_key(mode));
        let fixed = pair.is_some_and(|(mode, _, _)| values.get(mode) == Some(&Value::text("fixed")));
        result.push(ParameterControl {
            name: spec.name, label: spec.label.clone(), kind, value: value.clone(), allow_none: spec.allow_none, choices,
            minimum: spec.minimum, maximum: spec.maximum, step: spec.step, effects: spec.effects,
            automatic: spec.allow_none && !limit_field, automatic_value: automatic_values.get(&spec.name).cloned().unwrap_or(Value::None),
            unavailable_reason: if limit_field && !fixed { "Choose Fixed limits to edit.".into() } else { String::new() },
            segmented: spec.segmented,
        });
    }
    Ok(result)
}

fn choices_json(choices: &[(Value, String)]) -> Json {
    Json::Array(choices.iter().map(|(value, label)| json!([value.to_json(), label])).collect())
}

fn number_json(value: Option<f64>) -> Json { value.map_or(Json::Null, |number| json!(number)) }

/// The `controls` list a desktop panel description carries.  `live_window`
/// omits the window control a live input binding owns.
pub fn panel_controls(configuration: &ResolvedConfiguration, live_window: bool) -> Result<Vec<Json>> {
    let mut controls = Vec::new();
    for field in &configuration.semantics.fields {
        let mut item = json!({
            "section": "semantic", "name": field.name, "label": field.label, "kind": "choice", "value": field.value.to_json(),
            "choices": choices_json(&field.choices), "allow_none": !field.required, "unavailable_reason": "",
        });
        if let Some(cycle) = field.cycle_choices.as_ref().filter(|cycle| !cycle.is_empty()) {
            let position = cycle.position_of(&field.value);
            let (value, label) = cycle.get(position.unwrap_or(0)).expect("nonempty cycle");
            item["cycle"] = json!({"count": cycle.len(), "index": position, "label": label, "value": value.to_json()});
        }
        controls.push(item);
    }
    let metadata = &configuration.metadata;
    for control in parameter_controls(&configuration.parameter_schema, &configuration.display_state.values,
                                       &metadata.parameter_choices, &metadata.automatic_values)? {
        // A live panel's Window is its binding's row, held where v2's spec declares it (a curve's
        // before Uncertainty band, a rolling's before Side distribution).
        if live_window && control.name == "window" { controls.push(json!({"name": "window", "section": "display", "bound": true})); continue; }
        controls.push(json!({
            "name": control.name, "label": control.label, "kind": control.kind.value(), "value": control.value.to_json(),
            "allow_none": control.allow_none, "choices": choices_json(&control.choices), "minimum": number_json(control.minimum),
            "maximum": number_json(control.maximum), "step": number_json(control.step), "automatic": control.automatic,
            "automatic_value": control.automatic_value.to_json(), "unavailable_reason": control.unavailable_reason,
            "segmented": control.segmented, "section": "display",
        }));
    }
    Ok(controls)
}

/// The configuration-owned part of a desktop panel description: kind,
/// structure, target, controls, size, size choices and the fit block (the
/// session adds focus, axes and computed fit values).
pub fn describe_panel(configuration: &ResolvedConfiguration, live_window: bool) -> Result<Json> {
    let fit = &configuration.fit_target;
    let expression = configuration.fit_expression()?;
    let expression = match fit.get("expression") { Some(value) => value.to_json(), None => Json::String(expression) };
    let structure = configuration.schema.as_deref().map(|schema| {
        Json::Array(schema_structure(schema).iter().map(|group| Json::Array(group.iter().map(|(name, size)| json!([name, size])).collect())).collect())
    });
    Ok(json!({
        "kind": configuration.kind().value(),
        "structure": structure,
        "target": Value::Map(configuration.target()?).to_json(),
        "controls": panel_controls(configuration, live_window)?,
        "size": configuration.size,
        "size_choices": configuration.size_choices(),
        "fit": {
            "models": configuration.fit_models().iter().map(|model| json!({"key": model.model_id, "label": model.display_name,
                "parameters": model.parameter_names.iter().zip(&model.parameter_symbols).collect::<Vec<_>>()})).collect::<Vec<_>>(),
            "selected": fit.get("model").map_or(Json::String(String::new()), Value::to_json),
            "configuration": Value::Map(fit.clone()).to_json(),
            "expression": expression,
            "error": configuration.fit_expression_error(),
        },
    }))
}

/// Describe a blank authored kind without data or a host: its schema's
/// defaults plus the declared subset of a persisted appearance.
pub fn parameter_controls_for_kind(kind: super::kinds::PlotKind, values: Option<&ValueMap>, facet_cell_kind: Option<super::kinds::PlotKind>,
                                   style: &crate::style::PlotStyle) -> Result<Vec<ParameterControl>> {
    let schema = super::display::parameter_schema_for_kind(kind, style, facet_cell_kind)?;
    let state = schema.initial_values(values.map(|values| schema.declared_subset(values)).as_ref())?;
    parameter_controls(&schema, &state, &OrderedMap::new(), &OrderedMap::new())
}

/// The plot-kind catalogue a desktop state carries (`_PLOT_KINDS`).
pub fn plot_kinds() -> Json {
    Json::Array(super::handlers::HANDLERS.iter().map(|handler| json!({
        "id": handler.kind.value(), "label": handler.display_name, "record_input": handler.record_input,
    })).collect())
}
