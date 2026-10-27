//! Atom Workbench native worker: JSONL requests on stdin, one response line
//! per request on stdout. Every rb87-full/2 solve is served by the population
//! core (`population_core_adapter`). Matrices in the public protocol are
//! row-major, frequencies are rad/us, collapse amplitudes are sqrt(1/us).
use serde::Deserialize;
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};

mod harmonics;
mod population_core_adapter;
mod lattice_core_adapter;
const NATIVE_PROTOCOL: &str = "atom-lindblad-jsonl/2";
const FULL_MODEL_SCHEMA: &str = "rb87-full/2";

#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Operator {
    real: Vec<f64>,
    imag: Vec<f64>,
}
#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Drive {
    omega: f64,
    cos: Operator,
    sin: Operator,
}
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
struct FrequencyCoordinates {
    frequencies:Vec<f64>,
    #[serde(default,rename="frequenciesMHz")]
    frequencies_mhz:Option<Vec<f64>>,
    drive_indices:Vec<Vec<i32>>,
    frame_indices:Vec<Vec<i32>>,
    frame_sectors:Vec<i32>,
}
#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Model {
    dimension: usize,
    #[serde(default)]
    states: Vec<String>,
    h_real: Vec<f64>,
    h_imag: Vec<f64>,
    collapse: Vec<Operator>,
    #[serde(default)]
    initial_index: usize,
    initial_real: Option<Vec<f64>>,
    initial_imag: Option<Vec<f64>>,
    #[serde(default)]
    drives: Vec<Drive>,
    #[serde(default)]
    frequency_coordinates:Option<FrequencyCoordinates>,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Options {
    #[serde(default = "default_duration")]
    duration_us: f64,
    #[serde(default = "default_samples")]
    samples: usize,
    /// Absolute population error on the probability (0..1) scale.
    #[serde(default = "default_population_tolerance")]
    population_tolerance: f64,
    #[serde(default = "default_true")]
    include_asymptotic: bool,
    /// Only the long-time mean (no finite-time trajectory).
    #[serde(default)]
    steady_only: bool,
    /// Observation horizon of the long-time mean (us); absent or null is the
    /// t -> infinity mean.
    #[serde(default)]
    mean_horizon_us: Option<f64>,
    /// Hidden: the population core of a steadyOnly request, "shear" or
    /// "lattice"; overrides ATOM_POPULATION_CORE. Not part of the UI schema.
    #[serde(default)]
    solver_core: Option<String>,
    /// Hidden: lattice-core validation settings (frame, orders, tolerances).
    #[serde(default)]
    lattice: Option<Value>,
}
fn default_duration() -> f64 {
    5.
}
fn default_samples() -> usize {
    200
}
fn default_population_tolerance() -> f64 {
    0.005
}
fn default_true() -> bool {
    true
}
impl Default for Options {
    fn default() -> Self {
        Self {
            duration_us: 5.,
            samples: default_samples(),
            population_tolerance: default_population_tolerance(),
            include_asymptotic: true,
            steady_only: false,
            mean_horizon_us: None,
            solver_core: None,
            lattice: None,
        }
    }
}
impl Options {
    fn validate_population_tolerance(&self) -> Result<(), String> {
        if !self.population_tolerance.is_finite()
            || self.population_tolerance <= 0.
            || self.population_tolerance > 0.02
        {
            return Err("populationTolerance must be finite and in (0,0.02] on the probability scale".into());
        }
        Ok(())
    }
}
#[cfg(test)]
mod population_option_tests {
    use super::*;
    #[test]
    fn population_options_keep_legacy_default_and_requested_probability_scale() {
        let legacy:Options=serde_json::from_value(json!({})).unwrap();
        assert_eq!(legacy.population_tolerance,0.005);
        for value in [0.001,0.005,0.02,0.003] {
            let parsed:Options=serde_json::from_value(json!({"populationTolerance":value})).unwrap();
            assert_eq!(parsed.population_tolerance,value);
            assert!(parsed.validate_population_tolerance().is_ok());
        }
        for value in [0.,-0.001,0.020001,f64::INFINITY,f64::NAN] {
            assert!(Options{population_tolerance:value,..Default::default()}.validate_population_tolerance().is_err());
        }
    }
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Request {
    #[serde(default)]
    id: Value,
    #[serde(default)]
    op: Option<String>,
    #[serde(default)]
    model: Option<Model>,
    #[serde(default)]
    model_schema: Option<String>,
    #[serde(default)]
    options: Options,
    #[serde(default)]
    progress: bool,
}

fn write_line(message: &Value) -> bool {
    let mut out = io::BufWriter::new(io::stdout().lock());
    serde_json::to_writer(&mut out, message).is_ok() && writeln!(out).is_ok() && out.flush().is_ok()
}

/// One protocol response. `capabilities` and `warmup` (the worker start-up
/// LAPACK self-check) are operations; anything else is an rb87-full/2 solve,
/// which may first stream the long-time mean as `asymptotic` progress.
fn respond(request: Request) -> Value {
    let id = request.id;
    match request.op.as_deref() {
        Some("capabilities") => return json!({"id":id,"protocol":NATIVE_PROTOCOL,"capabilities":[FULL_MODEL_SCHEMA]}),
        Some("warmup") => return match population_core_adapter::warmup() {
            Ok(()) => json!({"id":id,"protocol":NATIVE_PROTOCOL,"ready":true}),
            Err(error) => json!({"id":id,"protocol":NATIVE_PROTOCOL,"error":error}),
        },
        Some(_) => return json!({"id":id,"protocol":NATIVE_PROTOCOL,"error":"Unknown native operation"}),
        None => {}
    }
    let stamp = |mut result: Value| {
        result["protocol"] = json!(NATIVE_PROTOCOL);
        result["modelSchema"] = json!(FULL_MODEL_SCHEMA);
        result
    };
    let mut publish_mean = |mean: &Value| {
        if request.progress {
            write_line(&json!({"id":id,"protocol":NATIVE_PROTOCOL,"progress":"asymptotic","result":stamp(mean.clone())}));
        }
    };
    let result = match (request.model_schema.as_deref(), request.model.as_ref()) {
        (Some(FULL_MODEL_SCHEMA), Some(model)) => population_core_adapter::solve(model, &request.options, &mut publish_mean),
        (Some(FULL_MODEL_SCHEMA), None) => Err("A solve request requires a model".into()),
        _ => Err(format!("Unsupported native model schema; this worker serves {FULL_MODEL_SCHEMA}")),
    };
    match result {
        Ok(result) => json!({"id":id,"protocol":NATIVE_PROTOCOL,"result":stamp(result)}),
        Err(error) => json!({"id":id,"protocol":NATIVE_PROTOCOL,"error":error}),
    }
}

#[cfg(feature="static-numerics")]
pub fn embedded_source_manifest()->&'static str{include_str!(env!("ATOM_NUMERIC_MANIFEST"))}

/// Shared JSONL entry used by the ordinary solver executable and the native
/// desktop executable's isolated --worker process. Numerical code is shared.
pub fn main() {
    for line in io::stdin().lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("stdin: {e}");
                break;
            }
        };
        if line.trim().is_empty() {
            continue;
        }
        let output = match serde_json::from_str::<Request>(&line) {
            Ok(request) => respond(request),
            Err(error) => json!({"id":null,"protocol":NATIVE_PROTOCOL,"error":format!("Invalid request: {error}")}),
        };
        if !write_line(&output) {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn request(line: Value) -> Value {
        respond(serde_json::from_value(line).unwrap())
    }
    #[test]
    fn operations_report_the_single_schema_and_a_numerical_self_check() {
        assert_eq!(request(json!({"id":0,"op":"capabilities"}))["capabilities"], json!([FULL_MODEL_SCHEMA]));
        let warm = request(json!({"id":-1,"op":"warmup"}));
        assert_eq!(warm["ready"], true, "{warm}");
        assert!(request(json!({"id":1,"op":"unknown"}))["error"].is_string());
    }
    #[test]
    fn solves_require_the_full_model_schema() {
        let model = json!({"dimension":1,"states":["a"],"hReal":[0],"hImag":[0],"collapse":[]});
        for schema in [Value::Null, json!("generic-lindblad/1")] {
            let out = request(json!({"id":2,"modelSchema":schema,"model":model,"options":{"steadyOnly":true}}));
            assert!(out["error"].as_str().unwrap().contains("Unsupported native model schema"), "{out}");
        }
        let out = request(json!({"id":3,"modelSchema":FULL_MODEL_SCHEMA,"model":model}));
        assert!(out["error"].is_string(), "{out}");
    }
}
