//! Authoring fields as plugins declare them (PLUGIN_PROTOCOL §2, v2
//! `AuthoringField`): each value's own rule (type, choices, bounds), drafts
//! that may stand incomplete, and the `Control` rows views render.
pub mod typed;
use serde::Deserialize;
use serde_json::{Map, Value, json};

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct Field {
    pub name: String,
    #[serde(rename = "type", alias = "value_type")]
    pub kind: String,
    pub label: String,
    pub default: Value,
    pub required: bool,
    pub minimum: Option<f64>,
    pub maximum: Option<f64>,
    pub step: Option<f64>,
    pub unit: Option<String>,
    /// The unit family a number is shown in by default, when its owner
    /// declares one (an RF generator's amplitude in Vpp, its unit dBm); the
    /// view picks the prefix in it. Display only: `unit` stays the value's.
    pub display_unit: Option<String>,
    pub choices: Vec<Choice>,
    pub segmented: bool,
    pub blank_label: Option<String>,
    pub description: Option<String>,
    /// `[controlling field, [values]]`: editable only while that field holds
    /// one of the values; otherwise kept as written, not validated, not run.
    pub enabled_when: Option<(String, Vec<Value>)>,
    pub derived: bool,
    /// The form section the field stands in; the form titles each change of it.
    pub group: Option<String>,
    pub file_filter: String,
    pub columns: Vec<Field>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Choice { pub value: Value, pub label: String }

impl Field {
    fn label(&self) -> &str { if self.label.is_empty() { &self.name } else { &self.label } }
    fn vacant(&self, value: &Value) -> bool {
        match value {
            Value::Null => true,
            Value::String(text) => self.required && text.trim().is_empty(),
            Value::Array(items) => self.required && items.is_empty(),
            _ => false,
        }
    }
}

/// Whether `field` is active for the effective values (typed equality: `1`
/// is not `1.0`, `true` is not `1`).
pub fn enabled(field: &Field, effective: &Map<String, Value>) -> bool {
    field.enabled_when.as_ref().is_none_or(|(controller, offered)|
        offered.iter().any(|value| effective.get(controller).unwrap_or(&Value::Null) == value))
}

pub fn effective(fields: &[Field], values: &Map<String, Value>) -> Map<String, Value> {
    fields.iter().map(|field| (field.name.clone(), values.get(&field.name).cloned().unwrap_or_else(|| field.default.clone()))).collect()
}

/// `value` as `field` keeps it, or one sentence naming the field.
/// `complete` reaches only the rows of a rows field (v2 `project_field`).
pub fn project(field: &Field, value: &Value, complete: bool) -> Result<Value, String> {
    typed::project(field, value, complete).map_err(|error| error.message)
}

/// A float as Python prints it in a sentence (`0.0`, `1e-05` aside).
fn py_float(value: f64) -> String {
    if value.fract() == 0.0 && value.abs() < 1e16 { format!("{value:.1}") } else { format!("{value}") }
}

/// Project a draft: every value checks except completeness, which is
/// reported as the second result. With `complete`, a vacant required
/// field refuses (v2 `draft_values` / `project_values`).
pub fn project_values(fields: &[Field], supplied: &Map<String, Value>, complete: bool) -> Result<(Map<String, Value>, bool), String> {
    typed::values(fields, supplied, complete).map_err(|error| error.message)
}

/// The first vacant required field among the active ones, as an issue.
pub fn completeness_issue(fields: &[Field], values: &Map<String, Value>) -> Option<String> {
    let effective = effective(fields, values);
    fields.iter().filter(|field| enabled(field, &effective))
        .find(|field| field.required && field.vacant(&effective[&field.name]))
        .map(|field| if field.kind == "rows" { format!("{} must have at least one row", field.label()) } else { format!("{} must be filled in", field.label()) })
}

/// Only today's declared leaves of a saved draft, columns of rows included
/// (a layout restores what still exists; v2 `_current_authoring_values`).
pub fn current_values(fields: &[Field], saved: &Map<String, Value>) -> Map<String, Value> {
    let mut values = Map::new();
    for field in fields {
        let Some(value) = saved.get(&field.name) else { continue };
        let value = match (field.kind.as_str(), value) {
            ("rows", Value::Array(rows)) => Value::Array(rows.iter().map(|row| match row {
                Value::Object(row) => Value::Object(current_values(&field.columns, row)),
                other => other.clone(),
            }).collect()),
            _ => value.clone(),
        };
        values.insert(field.name.clone(), value);
    }
    values
}

/// A saved draft restored onto today's fields (v2 Layout restore): a value its field
/// refuses is left at the field's default, and its refusal returned for the reader's note.
pub fn restore(fields: &[Field], saved: &Map<String, Value>) -> Result<(Map<String, Value>, Vec<String>), String> {
    let mut kept = current_values(fields, saved);
    let mut refusals = Vec::new();
    for field in fields {
        if let Some(Err(refusal)) = kept.get(&field.name).map(|value| project(field, value, false)) {
            kept.remove(&field.name);
            refusals.push(refusal);
        }
    }
    project_values(fields, &kept, false).map(|(values, _)| (values, refusals))
}

/// The type's defaults as an editable draft.
pub fn defaults(fields: &[Field]) -> Map<String, Value> {
    project_values(fields, &Map::new(), false).map(|(values, _)| values)
        .unwrap_or_else(|_| fields.iter().map(|field| (field.name.clone(), field.default.clone())).collect())
}

/// One `Control` per field (apps/desktop/ui/src/protocol.ts), in declared order.
pub fn controls(fields: &[Field], values: &Map<String, Value>, section: &str, problems: &[(String, String)]) -> Vec<Value> {
    let effective = effective(fields, values);
    fields.iter().map(|field| {
        let kind = match field.kind.as_str() {
            "int" => "integer", "float" => "number", "bool" => "boolean", "choice" => "choice",
            "rows" => "rows", "pair" => "pair", "resource" | "folder" => "path", _ => "text",
        };
        let kind = if !field.choices.is_empty() && kind != "rows" { "choice" } else { kind };
        let kind = if !field.file_filter.is_empty() { "path" } else { kind };
        let mut control = json!({"section": section, "name": field.name, "label": field.label(), "kind": kind,
            "value": effective[&field.name], "enabled": enabled(field, &effective), "required": field.required});
        if !field.choices.is_empty() {
            control["choices"] = Value::Array(field.choices.iter().map(|choice| json!([choice.value, choice.label])).collect());
            control["allow_none"] = json!(!field.required);
        } else if matches!(kind, "integer" | "number") {
            // v2 _project_field: a defaultless number starts vacant, required or not, so its box
            // holds that vacancy; one with a default is never blank (a spin box).
            control["allow_none"] = json!(field.default.is_null());
        }
        for (key, value) in [("minimum", field.minimum), ("maximum", field.maximum), ("step", field.step)] {
            if let Some(value) = value { control[key] = json!(value); }
        }
        for (key, value) in [("unit", &field.unit), ("display_unit", &field.display_unit), ("description", &field.description), ("blank_label", &field.blank_label)] {
            if let Some(value) = value { control[key] = json!(value); }
        }
        if let Some(group) = field.group.as_ref().filter(|group| !group.is_empty()) { control["group"] = json!(group); }
        if field.segmented { control["segmented"] = json!(true); }
        if field.kind == "folder" { control["mode"] = json!("dir"); }
        if !field.file_filter.is_empty() { control["filter"] = json!(field.file_filter); }
        if field.derived { control["derived"] = json!(true); }
        if !field.columns.is_empty() {
            control["columns"] = Value::Array(controls(&field.columns, &Map::new(), section, &[]));
        }
        if let Some((_, problem)) = problems.iter().find(|(name, _)| name == &field.name) { control["problem"] = json!(problem); }
        control
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field(value: Value) -> Field { serde_json::from_value(value).unwrap() }

    #[test]
    fn values_keep_their_field_rule_and_drafts_may_stand_incomplete() {
        let fields = vec![
            field(json!({"name": "mode", "type": "choice", "label": "Mode", "default": "a",
                         "choices": [{"value": "a", "label": "A"}, {"value": "b", "label": "B"}]})),
            field(json!({"name": "exposure", "type": "float", "label": "Exposure", "minimum": 0.0, "unit": "s", "default": 0.1})),
            field(json!({"name": "frames", "type": "int", "label": "Frames", "required": true, "enabled_when": ["mode", ["b"]]})),
            field(json!({"name": "pairs", "type": "rows", "label": "Pairs",
                         "columns": [{"name": "start", "type": "str", "label": "Start"}, {"name": "on", "type": "bool", "label": "On", "default": true}]})),
        ];
        let draft = |values: Value| project_values(&fields, values.as_object().unwrap(), false);
        let (values, complete) = draft(json!({"exposure": "0.5", "pairs": [{"start": "0"}]})).unwrap();
        assert_eq!(values["exposure"], json!(0.5));
        assert_eq!(values["pairs"], json!([{"start": "0", "on": true}]));
        assert!(complete, "frames is disabled while mode is a");
        let (_, complete) = draft(json!({"mode": "b"})).unwrap();
        assert!(!complete);
        assert_eq!(completeness_issue(&fields, &json!({"mode": "b"}).as_object().unwrap().clone()).unwrap(), "Frames must be filled in");
        assert_eq!(draft(json!({"exposure": -1})).unwrap_err(), "Exposure must be at least 0.0 s");
        assert_eq!(draft(json!({"mode": "c"})).unwrap_err(), "Mode must be one of \"a\", \"b\"");
        assert_eq!(draft(json!({"mode": "b", "frames": 2.0})).unwrap_err(), "Frames must be an integer or decimal integer text");
        assert!(draft(json!({"colour": 1})).unwrap_err().starts_with("unknown authoring fields"));
        assert_eq!(draft(json!({"pairs": [1]})).unwrap_err(), "Pairs row 1 must be a mapping of its columns");
        let saved = json!({"mode": "b", "gone": 1, "pairs": [{"start": "1", "old": 2}]});
        assert_eq!(Value::Object(current_values(&fields, saved.as_object().unwrap())), json!({"mode": "b", "pairs": [{"start": "1"}]}));
        let file = field(json!({"name": "config_file", "type": "str", "default": "", "file_filter": "Pulse Config (*.json)"}));
        let controls = controls(std::slice::from_ref(&file), &Map::new(), "parameter", &[]);
        assert_eq!(controls[0]["kind"], "path");
        assert_eq!(controls[0]["filter"], "Pulse Config (*.json)");
        assert_eq!(project(&file, &Value::Null, false).unwrap(), "");
        assert_eq!(project(&file, &json!("C:/config.json"), false).unwrap(), "C:/config.json");
        // A declared display family reaches the row beside the value's own unit; none, no key.
        let amplitude = field(json!({"name": "power", "type": "float", "label": "Power", "unit": "dBm", "display_unit": "Vpp"}));
        let rows = super::controls(&[amplitude, fields[1].clone()], &Map::new(), "tunable", &[]);
        assert_eq!((&rows[0]["unit"], &rows[0]["display_unit"]), (&json!("dBm"), &json!("Vpp")));
        assert!(rows[1].get("display_unit").is_none());
    }
}
