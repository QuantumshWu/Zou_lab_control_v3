//! Atom Workbench native worker: JSONL requests on stdin, one response line
//! per request on stdout. Every rb87-full/2 solve is served by the population
//! core through `lattice_core_adapter`; one whose model carries motion (3D) by
//! the semiclassical engine through `motion3d_adapter`. Matrices in the public
//! protocol are row-major, frequencies are rad/us, collapse amplitudes are
//! sqrt(1/us).
use serde::Deserialize;
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};

mod harmonics;
mod lattice_core_adapter;
mod motion3d_adapter;
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
    /// Motion: the atom moving in 3D in the tweezer the block names (motion3d_adapter).
    #[serde(default)]
    motion:Option<motion3d_adapter::MotionModel>,
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
    /// The device a motion solve is asked to run on.
    #[serde(default)]
    device: Device,
}
/// A motion solve's device (`options.device`): "gpu" or "cpu", the GPU when absent.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
enum Device {
    #[default]
    Gpu,
    Cpu,
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
            device: Device::Gpu,
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

/// One protocol response. `capabilities`, `warmup` (the worker start-up LAPACK
/// self-check) and `gpu` (starts the motion's GPU, answered when it has, the
/// driver's first compile of the kernels included, or with why it cannot) are
/// operations; anything else is an rb87-full/2 solve, which with `progress` first
/// streams every checkpoint's result as `checkpoint` progress (the same contract,
/// marked `progress`).
fn respond(request: Request) -> Value {
    let id = request.id;
    let ready = |outcome: Result<(), String>| match outcome {
        Ok(()) => json!({"id":id,"protocol":NATIVE_PROTOCOL,"ready":true}),
        Err(error) => json!({"id":id,"protocol":NATIVE_PROTOCOL,"error":error}),
    };
    match request.op.as_deref() {
        Some("capabilities") => return json!({"id":id,"protocol":NATIVE_PROTOCOL,"capabilities":[FULL_MODEL_SCHEMA]}),
        Some("warmup") => return ready(lattice_core_adapter::warmup()),
        Some("gpu") => return ready(motion3d_adapter::start_gpu()),
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
            Some(motion) => motion3d_adapter::solve(model, motion, &request.options, &mut publish),
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
    /// The `gpu` operation answers once the GPU has started -- ready -- or with why it cannot, never
    /// before: a second start finds the same outcome at once (`ATOM_HIDE_NVCUDA` simulates no driver).
    #[test]
    fn the_gpu_operation_answers_when_the_gpu_has_started() {
        let first = request(json!({"id":-2,"op":"gpu"}));
        assert_eq!(first["id"], -2, "{first}");
        assert_eq!(first["protocol"], NATIVE_PROTOCOL, "{first}");
        assert!(first["ready"] == true || !first["error"].as_str().unwrap_or("").is_empty(), "{first}");
        assert_eq!(request(json!({"id":-2,"op":"gpu"})), first);
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
    /// The probe-only fixture without its probe, with a 3D motion block: no beam, H0 the fixture's
    /// diagonal, the spherical collapse operators standing in for the Cartesian ones; the atom
    /// starts in a ground state.
    fn motion_model() -> Value {
        let fixture: Value = serde_json::from_str(include_str!("../tests/fixtures/probe_only_short.json")).unwrap();
        let mut model = fixture["model"].clone();
        let diagonal: Vec<Value> = model["hReal"].as_array().unwrap().iter().enumerate().map(|(k, x)| if k % 33 == 0 { x.clone() } else { json!(0.) }).collect();
        model["hReal"] = json!(diagonal);
        model["hImag"] = json!(vec![0.; 1024]);
        model["drives"] = json!([]);
        model["motion"] = json!({"axis":"3d","trapBeamId":"tweezer","temperatureUK":30.,"massKg":1.4431608951791763e-25,"fieldVectorG":[0.,0.,3.],
            "hamiltonian":{"real":diagonal,"imag":vec![0.;1024]},"frameFrequenciesMHz":(0..32).map(|i| if i < 8 { 0. } else { 3.8e8 }).collect::<Vec<_>>(),
            "beams":[],"emission":model["collapse"].clone()});
        model
    }
    /// The default scene's motion as the app sends it (tests/fixtures/default_scene_motion.json):
    /// the result passes the adapter's checks -- the
    /// engine's rows, before the projection, sum to one (`lattice_core_adapter::row_errors`, the
    /// trace it reports) --, and every published
    /// population lies in [0, 1] (the rows projected onto the probability simplex).
    #[test]
    fn the_default_scene_s_motion_keeps_every_row_s_sum() {
        let fixture: Value = serde_json::from_str(include_str!("../tests/fixtures/default_scene_motion.json")).unwrap();
        let mut line = fixture["payload"].clone();
        line["id"] = json!(8);
        let out = request(line);
        assert!(out["error"].is_null(), "{}", out["error"]);
        let result = &out["result"];
        let trace = result["diagnostics"]["traceError"].as_f64().unwrap();
        assert!(trace <= 1e-7, "a row's sum before the projection is off by {trace:.2e}");
        let rows: Vec<Vec<f64>> = result["populations"].as_array().unwrap().iter().map(|row| row.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()).collect();
        let worst = rows.iter().map(|row| (row.iter().sum::<f64>() - 1.).abs()).fold(0., f64::max);
        assert!(worst <= 1e-7, "a published row's sum is off by {worst:.2e}");
        let outside = rows.iter().flatten().filter(|&&p| !(0. ..=1.).contains(&p)).count();
        assert_eq!(outside, 0, "every published population lies in [0, 1]");
    }
    /// The default scene with a weak D1 probe (tests/fixtures/weak_d1_probe_motion.json): the light's
    /// friction is weak, and a step's change of the mechanical energy is the integrator's own error,
    /// which the friction criterion took for the friction -- the released step search ran away, and
    /// the harmonic control's sums per step asked for terabytes. The step is the one the light's
    /// pattern needs (k v_T ≈ 0.35 rad/us at 20 uK: a few tenths of a microsecond), the result
    /// within its targets.
    #[test]
    fn a_weak_light_s_motion_keeps_the_step_its_pattern_needs() {
        let fixture: Value = serde_json::from_str(include_str!("../tests/fixtures/weak_d1_probe_motion.json")).unwrap();
        let mut line = fixture["payload"].clone();
        line["id"] = json!(9);
        let out = request(line);
        assert!(out["error"].is_null(), "{}", out["error"]);
        let result = &out["result"];
        let dt = result["motion"]["timeStepUs"].as_f64().unwrap();
        assert!(dt >= 0.1, "the step is {dt:e} us");
        let flagged = &result["diagnostics"]["approximations"]["flagged"];
        assert!(flagged.is_null(), "{flagged}");
    }
    /// Every atom lost (tests/fixtures/total_loss_motion.json: the three-beam scene's tweezer at a
    /// millionth of its power, a trap 1 nK deep, its probe and repump, the atom at 30 uK, 100 us): the
    /// initial atoms are the trap's bound ones (motion-3d ensemble.rs), and the light, scattering on,
    /// unbinds every one and pushes it out of the trap's region (motion-3d engine.rs # Loss) within the
    /// run. The result says so -- the loss one from the sample every atom has left by, the rows there (no
    /// held atom to describe) the last held ones, each a probability vector, outside the validated regime,
    /// and the flag from when -- where the released engine's empty rows failed the adapter's trace gate.
    #[test]
    fn a_motion_whose_every_atom_is_lost_reports_the_loss() {
        let fixture: Value = serde_json::from_str(include_str!("../tests/fixtures/total_loss_motion.json")).unwrap();
        let mut line = fixture["payload"].clone();
        line["id"] = json!(10);
        let out = request(line);
        assert!(out["error"].is_null(), "{}", out["error"]);
        let result = &out["result"];
        let loss: Vec<f64> = result["motion"]["lossProbability"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
        assert_eq!(loss[0], 0.);
        assert!(loss.windows(2).all(|w| w[1] >= w[0]), "{loss:?}");
        let gone = loss.iter().position(|&l| l == 1.).unwrap_or_else(|| panic!("every atom leaves within the run: {loss:?}"));
        assert!(loss[gone..].iter().all(|&l| l == 1.), "{loss:?}");
        let rows: Vec<Vec<f64>> = result["populations"].as_array().unwrap().iter().map(|row| row.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()).collect();
        assert!(rows[gone..].iter().all(|row| row == &rows[gone - 1]), "the rows after the loss carry the last held one's");
        let classical = result["motion"]["classical"].as_array().unwrap();
        assert!(classical[gone..].iter().all(|c| c == false), "{classical:?}");
        let flagged = result["diagnostics"]["approximations"]["flagged"].as_str().unwrap_or("");
        assert!(flagged.contains("无原子被俘获"), "{flagged}");
    }
    /// The default scene's tweezer alone (tests/fixtures/tweezer_alone_motion.json: 30 uK in its
    /// 830 uK deep trap, 1 ms): with no other light the light's frame rotates at the tweezer, a near
    /// carrier whose shift lies in its field, and the loss criterion's trap less the trap beam off
    /// kept that field -- a trap 36 uK deep, 84 % of the atoms lost at once. The depth is the
    /// tweezer's whole shift: the atoms stay.
    #[test]
    fn the_tweezer_alone_holds_its_thermal_atom() {
        let fixture: Value = serde_json::from_str(include_str!("../tests/fixtures/tweezer_alone_motion.json")).unwrap();
        let mut line = fixture["payload"].clone();
        line["id"] = json!(11);
        let out = request(line);
        assert!(out["error"].is_null(), "{}", out["error"]);
        let loss: Vec<f64> = out["result"]["motion"]["lossProbability"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
        assert!(loss.iter().all(|&l| l < 0.01), "{loss:?}");
    }
    /// A motion request is a trajectory alone of 3D motion, its trap one of its beams; a tweezer
    /// without light shift confines nothing. The device is the GPU (by default) or the CPU.
    #[test]
    fn a_motion_request_is_3d_motion_in_the_trap_it_names() {
        let mut model = motion_model();
        let solve = |model: &Value, options: Value| request(json!({"id":6,"modelSchema":FULL_MODEL_SCHEMA,"model":model,"options":options}))["error"].as_str().unwrap_or("").to_owned();
        assert!(solve(&model, json!({"durationUs":10.,"samples":5})).contains("a trajectory alone"));
        let trajectory = json!({"durationUs":10.,"samples":5,"includeMean":false,"device":"gpu"});
        assert!(solve(&model, trajectory.clone()).contains("trapBeamId tweezer names none of its beams"));
        for axis in ["z", "xy"] {
            let mut along = model.clone();
            along["motion"]["axis"] = json!(axis);
            assert_eq!(solve(&along, trajectory.clone()), motion3d_adapter::ONLY_3D);
        }
        let zero = json!({"real":vec![0.;1024],"imag":vec![0.;1024]});
        model["motion"]["beams"] = json!([{"id":"tweezer","name":"光镊","direction":[0.,0.,1.],"waveNumberPerUm":7.37,"waistUm":1.2,"intensityMWcm2":1.77e8,
            "frequencyIndex":[0],"coupling":zero,"counterCoupling":zero,"outsideShift":zero}]);
        assert_eq!(solve(&model, trajectory), atom_motion_3d::solve::NOT_CONFINED);
        let device = |name: &str| serde_json::from_value::<Options>(json!({"device":name})).map(|o| o.device);
        assert_eq!((device("gpu").unwrap(), device("cpu").unwrap()), (Device::Gpu, Device::Cpu));
        assert!(device("tpu").is_err());
        assert_eq!(serde_json::from_value::<Options>(json!({})).unwrap().device, Device::Gpu);
    }
}
