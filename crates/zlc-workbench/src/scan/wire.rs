//! Application wire DTOs for compiled scan specifications. The core's immutable
//! model deliberately has no JSON parser or authored-bank vocabulary.
use serde::Deserialize;
use serde_json::{Value, json};
use zlc_core::scan::{AxisSpec, PlanSpec, Scope};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Axis {
    port: String,
    identity: String,
    label: String,
    unit: String,
    values: Vec<f64>,
    #[serde(default)]
    coordinate_of: Option<String>,
    host_advanced: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Spec { axes: Vec<Axis>, scan_repeats: usize, run_repeats: usize }

pub fn plan_spec(value: &Value) -> Result<PlanSpec, String> {
    let spec: Spec = serde_json::from_value(value.clone()).map_err(|error| format!("scan plan: {error}"))?;
    Ok(PlanSpec { axes: spec.axes.into_iter().map(|axis| AxisSpec {
        port:axis.port,identity:axis.identity,label:axis.label,unit:axis.unit,values:axis.values,
        coordinate_of:axis.coordinate_of,host_advanced:axis.host_advanced,
    }).collect(), scan_repeats:spec.scan_repeats,run_repeats:spec.run_repeats })
}

pub fn scope(value: &Value) -> Result<Scope, String> {
    match value.as_str() {
        Some("outer") => Ok(Scope::Outer), Some("inner") => Ok(Scope::Inner), Some("all") => Ok(Scope::All),
        _ => Err("scan row scope must be outer, inner or all".into()),
    }
}

pub fn axes(axes: &[AxisSpec]) -> Value {
    Value::Array(axes.iter().map(|axis| json!({
        "port":axis.port,"identity":axis.identity,"label":axis.label,"unit":axis.unit,"values":axis.values,
        "coordinate_of":axis.coordinate_of,"host_advanced":axis.host_advanced,
    })).collect())
}
