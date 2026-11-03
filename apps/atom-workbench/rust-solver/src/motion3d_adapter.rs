//! The protocol bridge to the 3D motion engine (`atom_motion_3d`): an rb87-full/2 request whose
//! model carries the motion block (src/full-master-model.ts `MotionModel`) -- its beams kept apart,
//! their directions and the emission operators on the lab axes x, y, z -- a single atom moving in
//! 3D in the tweezer the block names, semiclassical. A trajectory alone, on the request's grid.
//!
//! # Request
//! The motion block: `axis` "3d" (the only motion served); `trapBeamId`, the id of the beam that
//! traps the atom (its focus is the origin, its axes and depth the trap's); `temperatureUK`,
//! `massKg`, `fieldVectorG`, `hamiltonian` (H0 without the beams' couplings), `frameFrequenciesMHz`,
//! `emission` (three operators, one per lab axis) and `beams`, each with `waistUm` exactly when it
//! is a Gaussian beam focused at the atom (a plane wave without). `options.device`, "cpu" or "gpu",
//! the device asked for ("cpu" when absent).
//!
//! # Result contract
//! `states`, `timesUs`, `populations` (the held atom's, averaged over the trajectories),
//! `populationAccuracy` against the method's own target (`POPULATION_TARGET`, two standard errors
//! per sample), `diagnostics` (`engine`, `traceError`, `initialStateError`, `elapsedMs`, `method`,
//! `device` -- the device used -- with `deviceNote` when it is not the one asked for, and
//! `approximations`: the ledger of the a-priori approximations, `budget`, `used`, `items` of
//! `label`, `estimate` and `action`, and `flagged` when some output times lie beyond the budget),
//! and `motion`, read by `readMotionResult` in src/solver-types.ts:
//! - `axis`: "3d";
//! - per lab axis ("x", "y", "z"): `temperatureUK` and `temperatureErrorUK` (μK, one per sample:
//!   T = m Var(v)/k_B over one trap period of the axis around the sample, and its standard error),
//!   `meanVelocityMps`, `meanOccupation` (both over the same period; n̄ relative to
//!   that axis's `trapFrequencyKHz`, ω/2π at the focus), `momentum` (`velocitiesMps`, `samples`,
//!   `probabilities`) and `position` (`positionsUm`, ...), distributions at listed samples;
//! - `temperatureTotalUK` and `temperatureTotalErrorUK`, one per sample: T_total = (T_x + T_y +
//!   T_z)/3, the temperature of the 3D velocity variance, and its standard error;
//! - `lossProbability`, one per sample, never decreasing;
//! - `classical`, one per sample: whether it lies in the validated regime (k_B T ≥ `CLASSICAL` ħω
//!   on every axis, motion-3d solve.rs); outside it the result is a reference only;
//! - `trajectories` and `timeStepUs`.
//!
//! Every output but `lossProbability` describes the atom still held. Every batch's result is
//! streamed first (`progress`: true) with the engine's estimate of the remaining wall time
//! (`diagnostics.estimatedRemainingMs`).
use crate::{Device,Model,Operator,Options,lattice_core_adapter};
use atom_motion_3d::ledger::Ledger;
use atom_motion_3d::solve::{Distributions,Solution,CONFIDENCE,POPULATION_TARGET};
use atom_motion_3d::{Beam,Complex,Matrix,Request,STATES};
use atom_population_core::Lapack;
use atom_population_core::lattice::pipeline::{LONGEST_TIME_US,MOST_SAMPLES};
use serde::Deserialize;
use serde_json::{json,Value};
use std::time::Instant;

const ENGINE:&str="rust-motion-3d";
const AXES:[&str;3]=["x","y","z"];
/// Why a motion block along an axis or in a plane is refused: only 3D motion is served.
pub const ONLY_3D:&str="运动只计算 3D（半经典）：运动轴需为 3d";
/// Why a request for the GPU runs on the CPU.
pub const GPU_PENDING:&str="尚无 GPU 版本：本次在 CPU 上计算";

/// The motion block of the model, as the TypeScript model writes it.
#[derive(Clone,Deserialize)]
#[serde(rename_all="camelCase",deny_unknown_fields)]
pub struct MotionModel{
    axis:String,
    trap_beam_id:String,
    #[serde(rename="temperatureUK")]
    temperature_uk:f64,
    mass_kg:f64,
    field_vector_g:[f64;3],
    hamiltonian:Operator,
    #[serde(rename="frameFrequenciesMHz")]
    frame_frequencies_mhz:Vec<f64>,
    beams:Vec<MotionBeam>,
    emission:Vec<Operator>,
}
#[derive(Clone,Deserialize)]
#[serde(rename_all="camelCase",deny_unknown_fields)]
struct MotionBeam{
    id:String,name:String,direction:[f64;3],wave_number_per_um:f64,#[serde(default)] waist_um:Option<f64>,outside_shift:Operator,
    #[serde(rename="intensityMWcm2")]
    intensity_mw_cm2:f64,frequency_index:Vec<i32>,coupling:Operator,counter_coupling:Operator,
}

/// A 32 × 32 matrix of the model, its entries finite.
fn matrix(operator:&Operator,what:&str)->Result<Matrix,String>{
    if operator.real.len()!=STATES*STATES||operator.imag.len()!=STATES*STATES{return Err(format!("{what} must be a 32 x 32 row-major matrix"));}
    if operator.real.iter().chain(&operator.imag).any(|x|!x.is_finite()){return Err(format!("{what} has a nonfinite entry"));}
    Ok(Matrix{real:operator.real.clone(),imag:operator.imag.clone()})
}

/// The engine's request: the model's motion block, its initial density and the request's grid.
fn request(model:&Model,motion:&MotionModel,options:&Options)->Result<Request,String>{
    let finite=|x:f64|x.is_finite();
    if motion.axis!="3d"{return Err(ONLY_3D.into());}
    if !(finite(motion.temperature_uk)&&motion.temperature_uk>0.&&finite(motion.mass_kg)&&motion.mass_kg>0.&&motion.field_vector_g.iter().all(|&x|finite(x))){
        return Err("The motion block needs a positive temperature and mass and a finite field".into());}
    if motion.frame_frequencies_mhz.len()!=STATES||!motion.frame_frequencies_mhz.iter().all(|&x|finite(x)){return Err("The motion block needs the 32 states' frame frequencies".into());}
    let frequencies=model.frequency_coordinates.as_ref().and_then(|c|c.frequencies_mhz.clone()).unwrap_or_default();
    let beams=motion.beams.iter().map(|b|{
        if !b.direction.iter().all(|&x|finite(x))||!(finite(b.wave_number_per_um)&&b.wave_number_per_um>0.)||b.waist_um.is_some_and(|w|!(finite(w)&&w>0.))
            ||!(finite(b.intensity_mw_cm2)&&b.intensity_mw_cm2>=0.)||b.frequency_index.len()!=frequencies.len(){
            return Err(format!("Beam {} needs a finite direction, a positive wave number, a positive waist if focused, a nonnegative intensity and an index on the {} frequency axes",b.id,frequencies.len()));}
        Ok(Beam{id:b.id.clone(),name:b.name.clone(),direction:b.direction,wave_number_per_um:b.wave_number_per_um,waist_um:b.waist_um,intensity_mw_cm2:b.intensity_mw_cm2,
            frequency_index:b.frequency_index.clone(),coupling:matrix(&b.coupling,"A beam's coupling")?,counter_coupling:matrix(&b.counter_coupling,"A beam's counter-rotating coupling")?,
            outside_shift:matrix(&b.outside_shift,"A beam's outside shift")?})
    }).collect::<Result<Vec<_>,String>>()?;
    let trap_beam=beams.iter().position(|b|b.id==motion.trap_beam_id).ok_or_else(||format!("The motion block's trapBeamId {} names none of its beams",motion.trap_beam_id))?;
    let [a,b,c]=motion.emission.as_slice()else{return Err("The motion block has three emission operators, one per lab axis".into())};
    let density=lattice_core_adapter::initial_density(model,&Lapack::load()?)?;
    Ok(Request{states:model.states.clone(),hamiltonian:matrix(&motion.hamiltonian,"The motion block's H0")?,frame_frequencies_mhz:motion.frame_frequencies_mhz.clone(),
        frequencies_mhz:frequencies,beams,trap_beam,emission:[matrix(a,"An emission operator")?,matrix(b,"An emission operator")?,matrix(c,"An emission operator")?],
        field_vector_g:motion.field_vector_g,mass_kg:motion.mass_kg,initial_density:density.iter().map(|z|Complex{re:z.re,im:z.im}).collect(),
        temperature_uk:motion.temperature_uk,times_us:lattice_core_adapter::grid(options)})
}

fn per_axis<T:serde::Serialize>(values:&[[T;3]])->Value where T:Copy{
    json!(AXES.iter().enumerate().map(|(a,name)|(name.to_string(),json!(values.iter().map(|v|v[a]).collect::<Vec<T>>()))).collect::<serde_json::Map<String,Value>>())
}
fn distributions(d:&[Distributions;3],grid:&str)->Value{
    json!(AXES.iter().zip(d).map(|(name,x)|(name.to_string(),json!({grid:x.grid,"samples":x.samples,"probabilities":x.probabilities}))).collect::<serde_json::Map<String,Value>>())
}
/// The ledger in the contract's shape.
fn approximations(ledger:&Ledger)->Value{
    let mut out=json!({"budget":ledger.budget,"used":ledger.used(),
        "items":ledger.items.iter().map(|i|json!({"label":i.label,"estimate":i.estimate,"action":i.action})).collect::<Vec<_>>()});
    if let Some(reason)=&ledger.flagged{out["flagged"]=json!(reason);}
    out
}

/// The result in the one contract; `remaining` marks a streamed checkpoint (with the engine's
/// estimate of the wall time to the final result, ms).
fn result(model:&Model,times:&[f64],initial:&[f64],solution:&Solution,groups:&[String],device:Device,remaining:Option<Option<f64>>,started:Instant)->Result<Value,String>{
    if solution.populations.len()!=times.len()||solution.populations.iter().any(|row|row.len()!=STATES||row.iter().any(|x|!x.is_finite())){return Err("The 3D motion engine returned populations that are not 32 finite values per time".into());}
    let(trace,initial_error)=lattice_core_adapter::row_errors(&solution.populations,initial)?;
    let errors:Vec<f64>=solution.sample_errors.iter().map(|e|CONFIDENCE*e).collect();
    let estimated=errors.iter().fold(0_f64,|a,&b|a.max(b));let converged=estimated<=POPULATION_TARGET;
    let mut result=json!({"states":model.states,"timesUs":times,"populations":solution.populations,
        "populationAccuracy":{"status":if converged{"estimated-converged"}else{"estimated-partial"},"requestedTolerance":POPULATION_TARGET,
            "sampleErrors":errors,"estimatedMaximumError":estimated,"scope":{"states":STATES,"samples":times.len(),"groups":groups}},
        "diagnostics":{"engine":ENGINE,"traceError":trace,"initialStateError":initial_error,"elapsedMs":started.elapsed().as_secs_f64()*1000.,"method":solution.method,
            "device":"cpu","approximations":approximations(&solution.approximations)},
        "motion":{"axis":"3d","temperatureUK":per_axis(&solution.temperature_uk),"temperatureErrorUK":per_axis(&solution.temperature_errors_uk),
            "temperatureTotalUK":solution.temperature_total_uk,"temperatureTotalErrorUK":solution.temperature_total_errors_uk,
            "meanVelocityMps":per_axis(&solution.mean_velocity_mps),"meanOccupation":per_axis(&solution.mean_occupation),
            "trapFrequencyKHz":json!({"x":solution.trap_frequency_khz[0],"y":solution.trap_frequency_khz[1],"z":solution.trap_frequency_khz[2]}),
            "lossProbability":solution.loss_probability,"classical":solution.classical,
            "momentum":distributions(&solution.velocity,"velocitiesMps"),"position":distributions(&solution.position,"positionsUm"),
            "trajectories":solution.trajectories,"timeStepUs":solution.time_step_us}});
    // The engine runs on the CPU; a request for the GPU says why it did not get it.
    if device==Device::Gpu{result["diagnostics"]["deviceNote"]=json!(GPU_PENDING);}
    if !converged{result["populationAccuracy"]["reason"]=json!(solution.reason.clone()
        .unwrap_or_else(||format!("The statistical error is {estimated:.2e} against the target {POPULATION_TARGET:e}")));}
    if let Some(remaining)=remaining{
        result["progress"]=json!(true);
        if let Some(ms)=remaining.filter(|ms|ms.is_finite()){result["diagnostics"]["estimatedRemainingMs"]=json!(ms.max(0.));}
    }
    Ok(result)
}

/// A motion request, checked -- a trajectory alone of the complete 32-state model on a finite grid
/// -- through the 3D engine: every batch is published as it is made, the final solution returned.
pub fn solve(model:&Model,motion:&MotionModel,options:&Options,publish:&mut dyn FnMut(&Value))->Result<Value,String>{
    let started=Instant::now();
    if options.mean_only||options.include_mean{return Err("A motion request is a trajectory alone: meanOnly and includeMean are false".into());}
    if !(options.duration_us.is_finite()&&options.duration_us>0.){return Err("The trajectory duration must be a positive finite time".into());}
    if options.duration_us>LONGEST_TIME_US{return Err(format!("The trajectory duration is limited to 10 s (requested {} s)",options.duration_us/1e6));}
    if !(2..=MOST_SAMPLES).contains(&options.samples){return Err(format!("A trajectory has 2 to {MOST_SAMPLES} samples (requested {})",options.samples));}
    if model.dimension!=STATES||model.states.len()!=STATES{return Err("rb87-full/2 requires the complete 32-state Rb87 model".into());}
    let(_,groups)=lattice_core_adapter::groups(model)?;let names:Vec<String>=groups.into_iter().map(|g|g.name).collect();
    let request=request(model,motion,options)?;
    let initial:Vec<f64>=(0..STATES).map(|i|request.initial_density[i*STATES+i].re).collect();
    let mut failure:Option<String>=None;
    let mut observe=|checkpoint:&Solution,remaining:Option<f64>|{if failure.is_some(){return;}
        match result(model,&request.times_us,&initial,checkpoint,&names,options.device,Some(remaining),started){Ok(value)=>publish(&value),Err(e)=>failure=Some(e)}};
    let solution=atom_motion_3d::solve::solve(&request,&mut observe)?;
    if let Some(e)=failure{return Err(e);}
    result(model,&request.times_us,&initial,&solution,&names,options.device,None,started)
}
