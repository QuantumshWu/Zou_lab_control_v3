//! Parameter value contracts. Presentation widgets and control rows belong
//! to the workbench; these rules know only declared values and constraints.
pub mod typed;
use serde::Deserialize;
use serde_json::{Map,Value};

#[derive(Clone,Debug,Default,Deserialize)]
#[serde(default)]
pub struct Field {
    pub name:String,
    #[serde(rename="type",alias="value_type")]
    pub kind:String,
    pub label:String,
    pub default:Value,
    pub required:bool,
    pub minimum:Option<f64>,
    pub maximum:Option<f64>,
    pub unit:Option<String>,
    pub choices:Vec<Choice>,
    pub enabled_when:Option<(String,Vec<Value>)>,
    pub columns:Vec<Field>,
}
#[derive(Clone,Debug,Deserialize)]
pub struct Choice { pub value:Value }
impl Field {
    fn label(&self)->&str{if self.label.is_empty(){&self.name}else{&self.label}}
    fn vacant(&self,value:&Value)->bool{match value{Value::Null=>true,Value::String(s)=>self.required&&s.trim().is_empty(),Value::Array(v)=>self.required&&v.is_empty(),_=>false}}
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parameter_projection_is_independent_of_control_metadata() {
        let fields:Vec<Field>=serde_json::from_value(json!([
            {"name":"enabled","type":"bool","default":false},
            {"name":"count","type":"int","required":true,"enabled_when":["enabled",[true]],"minimum":1},
            {"name":"value","type":"float","unit":"V","default":0.5,"minimum":0.0}
        ])).unwrap();
        let (values,complete)=project_values(&fields,&Map::new(),false).unwrap();
        assert!(complete);assert_eq!(values["count"],Value::Null);
        let input=json!({"enabled":"true","count":"2","value":"1.25"});
        let(values,complete)=typed::values(&fields,input.as_object().unwrap(),true).unwrap();
        assert!(complete);assert_eq!(values["count"],2);assert_eq!(values["value"],1.25);
        let invalid=json!({"enabled":true,"count":2.0});
        assert_eq!(typed::values(&fields,invalid.as_object().unwrap(),true).unwrap_err().kind,"TypeError");
        assert_eq!(typed::project(&fields[2],&json!(-1),true).unwrap_err().kind,"ValueError");
    }

    #[test]
    fn nested_values_and_saved_defaults_use_the_same_constraints() {
        let fields:Vec<Field>=serde_json::from_value(json!([
            {"name":"rows","type":"rows","columns":[{"name":"choice","type":"choice","default":"a","choices":[{"value":"a"},{"value":"b"}]}]}
        ])).unwrap();
        let input=json!({"rows":[{}]});
        assert_eq!(project_values(&fields,input.as_object().unwrap(),true).unwrap().0["rows"],json!([{"choice":"a"}]));
        let saved=json!({"rows":[{"choice":"b","obsolete":4}],"obsolete":5});
        assert_eq!(current_values(&fields,saved.as_object().unwrap())["rows"],json!([{"choice":"b"}]));
        let invalid=json!({"rows":[{"choice":"missing"}]});
        assert_eq!(restore(&fields,invalid.as_object().unwrap()).unwrap().1.len(),1);
    }
}
