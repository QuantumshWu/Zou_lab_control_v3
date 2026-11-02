//! Atom Workbench native worker: JSONL requests on stdin, one response line
//! per request on stdout. Every rb87-full/2 solve is served by the population
//! core through `lattice_core_adapter`; one whose model carries motion along an
//! axis (1D) by the motion core through `motion_core_adapter`, and one whose
//! model carries motion in a plane (2D) by the GPU engine through
//! `motion_gpu_adapter`. Matrices in the public protocol are row-major,
//! frequencies are rad/us, collapse amplitudes are sqrt(1/us).
use serde::Deserialize;
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};

mod harmonics;
mod lattice_core_adapter;
mod motion_core_adapter;
mod motion_gpu_adapter;
const NATIVE_PROTOCOL: &str = "atom-lindblad-jsonl/3";
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
    /// Radial motion: the atom in a tweezer moving along one lab axis (motion_core_adapter) or in
    /// the plane perpendicular to the tweezer (motion_gpu_adapter).
    #[serde(default)]
    motion:Option<motion_core_adapter::MotionModel>,
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
    /// The mean p̄(T) with the trajectory (a mean-only request has it
    /// always).
    #[serde(default = "default_true")]
    include_mean: bool,
    /// Only the mean p̄(T) (no trajectory).
    #[serde(default)]
    mean_only: bool,
    /// The time T (us) of the mean the app reads: 1e5 or 1e6 (100 ms or 1 s,
    /// the standard horizons, both of which a mean run returns); 1 s when
    /// absent.
    #[serde(default = "default_mean_horizon")]
    mean_horizon_us: f64,
}
fn default_duration() -> f64 {
    5.
}
fn default_samples() -> usize {
    200
}
fn default_population_tolerance() -> f64 {
    atom_population_core::lattice::accuracy::POPULATION_TOLERANCE
}
fn default_true() -> bool {
    true
}
fn default_mean_horizon() -> f64 {
    1e6
}
impl Default for Options {
    fn default() -> Self {
        Self {
            duration_us: 5.,
            samples: default_samples(),
            population_tolerance: default_population_tolerance(),
            include_mean: true,
            mean_only: false,
            mean_horizon_us: default_mean_horizon(),
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
/// which with `progress` first streams every checkpoint's result as
/// `checkpoint` progress (the same contract, marked `progress`).
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
    let mut publish = |checkpoint: &Value| {
        if request.progress {
            write_line(&json!({"id":id,"protocol":NATIVE_PROTOCOL,"progress":"checkpoint","result":stamp(checkpoint.clone())}));
        }
    };
    let result = match (request.model_schema.as_deref(), request.model.as_ref()) {
        (Some(FULL_MODEL_SCHEMA), Some(model)) => match &model.motion {
            Some(motion) if motion.is_plane() => motion_gpu_adapter::solve(model, motion, &request.options, &mut publish),
            Some(motion) => motion_core_adapter::solve(model, motion, &request.options, &mut publish),
            None => lattice_core_adapter::solve(model, &request.options, &mut publish),
        },
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
            let out = request(json!({"id":2,"modelSchema":schema,"model":model,"options":{"meanOnly":true}}));
            assert!(out["error"].as_str().unwrap().contains("Unsupported native model schema"), "{out}");
        }
        let out = request(json!({"id":3,"modelSchema":FULL_MODEL_SCHEMA,"model":model}));
        assert!(out["error"].is_string(), "{out}");
    }
    #[test]
    fn a_request_restates_the_fixed_tolerance_and_names_a_finite_grid() {
        assert_eq!(serde_json::from_value::<Options>(json!({})).unwrap().population_tolerance, atom_population_core::lattice::accuracy::POPULATION_TOLERANCE);
        let model = json!({"dimension":1,"states":["a"],"hReal":[0],"hImag":[0],"collapse":[]});
        let solve = |options: Value| request(json!({"id":4,"modelSchema":FULL_MODEL_SCHEMA,"model":model,"options":options}))["error"].as_str().unwrap().to_owned();
        assert!(solve(json!({"meanOnly":true,"populationTolerance":0.001})).contains("populationTolerance is fixed"));
        for samples in [1, 2001] {
            assert!(solve(json!({"samples":samples})).contains("A trajectory has 2 to 2000 samples"), "{samples}");
        }
        assert!(solve(json!({"durationUs":0})).contains("must be a positive finite time"));
        assert!(solve(json!({"durationUs":1.5e7})).contains("The trajectory duration is limited to 10 s (requested 15 s)"));
    }
    /// The probe-only fixture without its probe, with a motion block: no
    /// beam, H0 the fixture's diagonal, the spherical collapse operators
    /// standing in for the Cartesian ones; the atom starts in a ground state.
    fn motion_model() -> Value {
        let fixture: Value = serde_json::from_str(include_str!("../tests/fixtures/probe_only_short.json")).unwrap();
        let mut model = fixture["model"].clone();
        let diagonal: Vec<Value> = model["hReal"].as_array().unwrap().iter().enumerate().map(|(k, x)| if k % 33 == 0 { x.clone() } else { json!(0.) }).collect();
        model["hReal"] = json!(diagonal);
        model["hImag"] = json!(vec![0.; 1024]);
        model["drives"] = json!([]);
        model["motion"] = json!({"axis":"z","temperatureUK":30.,"massKg":1.4431608951791763e-25,"fieldVectorG":[0.,0.,3.],
            "hamiltonian":{"real":diagonal,"imag":vec![0.;1024]},"frameFrequenciesMHz":(0..32).map(|i| if i < 8 { 0. } else { 3.8e8 }).collect::<Vec<_>>(),
            "beams":[],"emission":model["collapse"].clone()});
        model
    }
    /// A motion request is a trajectory alone. Without light nothing confines the atom; with
    /// light the engine is not implemented yet.
    #[test]
    fn a_motion_request_is_refused_until_the_engine_is_implemented() {
        let mut model = motion_model();
        let solve = |model: &Value, options: Value| request(json!({"id":6,"modelSchema":FULL_MODEL_SCHEMA,"model":model,"options":options}))["error"].as_str().unwrap_or("").to_owned();
        assert!(solve(&model, json!({"durationUs":10.,"samples":5})).contains("a trajectory alone"));
        let trajectory = json!({"durationUs":10.,"samples":5,"includeMean":false});
        assert_eq!(solve(&model, trajectory.clone()), atom_motion_core::NO_TRAP);
        let zero = json!({"real":vec![0.;1024],"imag":vec![0.;1024]});
        model["motion"]["beams"] = json!([{"id":"tweezer","name":"光镊","direction":[0.,1.,0.],"waveNumberPerUm":7.37,"waistUm":1.2,"intensityMWcm2":1.77e8,
            "frequencyIndex":[0],"coupling":zero,"counterCoupling":zero,"outsideShift":zero}]);
        assert_eq!(solve(&model, trajectory), atom_motion_core::NOT_IMPLEMENTED);
    }
    /// A plane's motion goes to the 2D engine, a trajectory alone as an axis' does. Without light
    /// nothing confines the atom in the plane; on a computer without an NVIDIA GPU the engine says
    /// that instead.
    #[test]
    fn a_plane_s_motion_request_goes_to_the_gpu_engine() {
        let mut model = motion_model();
        model["motion"]["axis"] = json!("xy");
        let solve = |options: Value| request(json!({"id":7,"modelSchema":FULL_MODEL_SCHEMA,"model":model,"options":options}))["error"].as_str().unwrap_or("").to_owned();
        assert!(solve(json!({"durationUs":10.,"samples":5})).contains("a trajectory alone"));
        let refusal = solve(json!({"durationUs":10.,"samples":5,"includeMean":false}));
        assert!(refusal == atom_motion_gpu::solve::NO_TRAP || refusal.starts_with(&atom_motion_gpu::gpu::unavailable("")), "{refusal}");
    }
}
