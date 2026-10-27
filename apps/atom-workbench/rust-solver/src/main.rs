//! Atom Workbench native worker: JSONL requests on stdin, one response line
//! per request on stdout. Every rb87-full/2 solve is served by the population
//! core through `lattice_core_adapter`. Matrices in the public protocol are
//! row-major, frequencies are rad/us, collapse amplitudes are sqrt(1/us).
use serde::Deserialize;
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};

mod harmonics;
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
    /// Absolute population error on the probability (0..1) scale: the
    /// core's fixed tolerance, which a request may only restate.
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
}
fn default_duration() -> f64 {
    5.
}
fn default_samples() -> usize {
    200
}
fn default_population_tolerance() -> f64 {
    atom_population_core::lattice::ledger::POPULATION_TOLERANCE
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
        Some("warmup") => return match lattice_core_adapter::warmup() {
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
        (Some(FULL_MODEL_SCHEMA), Some(model)) => lattice_core_adapter::solve(model, &request.options, &mut publish_mean),
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
    #[test]
    fn a_request_restates_the_fixed_tolerance_and_names_a_finite_grid() {
        assert_eq!(serde_json::from_value::<Options>(json!({})).unwrap().population_tolerance, 0.005);
        let model = json!({"dimension":1,"states":["a"],"hReal":[0],"hImag":[0],"collapse":[]});
        let solve = |options: Value| request(json!({"id":4,"modelSchema":FULL_MODEL_SCHEMA,"model":model,"options":options}))["error"].as_str().unwrap().to_owned();
        assert!(solve(json!({"steadyOnly":true,"populationTolerance":0.001})).contains("populationTolerance is fixed"));
        assert!(solve(json!({"samples":1})).contains("Invalid finite population grid"));
        assert!(solve(json!({"durationUs":0})).contains("Invalid finite population grid"));
    }
}
