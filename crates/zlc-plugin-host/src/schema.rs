//! Validation of plugin command parameters against the JSON Schema the SDK
//! generated from their fields. Only the subset `zlc_plugin.declare` emits is
//! understood: type, enum, minimum/maximum, items, minItems/maxItems,
//! properties/required/additionalProperties and anyOf. The plugin's own field
//! projection remains the final judge (it answers `invalid`).
use serde_json::Value;

pub fn validate(value: &Value, schema: &Value) -> Result<(), String> { check(value, schema, "params") }

fn check(value: &Value, schema: &Value, path: &str) -> Result<(), String> {
    if let Some(options) = schema.get("anyOf").and_then(Value::as_array) {
        let failures: Vec<String> = options.iter().filter_map(|option| check(value, option, path).err()).collect();
        return if failures.len() < options.len() { Ok(()) } else { Err(failures.join("; or ")) };
    }
    if let Some(choices) = schema.get("enum").and_then(Value::as_array) {
        if !choices.iter().any(|choice| same(choice, value)) {
            return Err(format!("{path} must be one of {}", Value::Array(choices.clone())));
        }
    }
    if let Some(kind) = schema.get("type").and_then(Value::as_str) {
        let fits = match kind {
            "object" => value.is_object(),
            "array" => value.is_array(),
            "string" => value.is_string(),
            "boolean" => value.is_boolean(),
            "null" => value.is_null(),
            "integer" => value.is_i64() || value.is_u64() || value.as_f64().is_some_and(|x| x.fract() == 0.0),
            "number" => value.is_number(),
            _ => true,
        };
        if !fits { return Err(format!("{path} must be {kind}, not {value}")); }
    }
    if let Some(number) = value.as_f64() {
        if schema.get("minimum").and_then(Value::as_f64).is_some_and(|low| number < low) {
            return Err(format!("{path} must be at least {}", schema["minimum"]));
        }
        if schema.get("maximum").and_then(Value::as_f64).is_some_and(|high| number > high) {
            return Err(format!("{path} must be at most {}", schema["maximum"]));
        }
    }
    if let Some(items) = value.as_array() {
        if schema.get("minItems").and_then(Value::as_u64).is_some_and(|low| (items.len() as u64) < low)
            || schema.get("maxItems").and_then(Value::as_u64).is_some_and(|high| items.len() as u64 > high) {
            return Err(format!("{path} has {} items", items.len()));
        }
        if let Some(item) = schema.get("items") {
            for (index, entry) in items.iter().enumerate() { check(entry, item, &format!("{path}[{index}]"))?; }
        }
    }
    if let Some(object) = value.as_object() {
        let properties = schema.get("properties").and_then(Value::as_object);
        for name in schema.get("required").and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_str) {
            if !object.contains_key(name) { return Err(format!("{path}.{name} is required")); }
        }
        for (name, entry) in object {
            match properties.and_then(|properties| properties.get(name)) {
                Some(property) => check(entry, property, &format!("{path}.{name}"))?,
                None if schema.get("additionalProperties") == Some(&Value::Bool(false)) =>
                    return Err(format!("{path} has no parameter {name:?}")),
                None => {}
            }
        }
    }
    Ok(())
}

fn same(left: &Value, right: &Value) -> bool {
    match (left.as_f64(), right.as_f64()) {
        (Some(a), Some(b)) => a == b,
        _ => left == right,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn the_generated_subset_accepts_and_refuses() {
        let schema = json!({"type": "object", "additionalProperties": false, "required": ["seconds"], "properties": {
            "seconds": {"type": "number", "minimum": 0.0},
            "mode": {"anyOf": [{"enum": ["fast", "slow"]}, {"type": "null"}]},
            "shape": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2}}});
        validate(&json!({"seconds": 0.5, "mode": null, "shape": [3, 4]}), &schema).unwrap();
        for bad in [json!({}), json!({"seconds": -1}), json!({"seconds": 1, "mode": "medium"}),
                    json!({"seconds": 1, "shape": [1]}), json!({"seconds": 1, "colour": "red"}), json!({"seconds": 1, "shape": [1.5, 2]})] {
            assert!(validate(&bad, &schema).is_err(), "{bad}");
        }
    }
}
