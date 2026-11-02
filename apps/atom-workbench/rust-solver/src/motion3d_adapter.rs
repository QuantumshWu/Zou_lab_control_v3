//! The protocol bridge to the 3D motion engine (`atom_motion_3d`): an rb87-full/2 request whose
//! motion block has the axis "3d" -- its beam directions and emission operators on the lab axes
//! x, y, z (src/full-master-model.ts) -- a single atom moving in 3D in the tweezer, semiclassical.
//! A trajectory alone, on the request's grid.
//!
//! # Result contract
//! As motion_core_adapter.rs's (`states`, `timesUs`, `populations`, `populationAccuracy`,
//! `diagnostics`), with `populationAccuracy` against the method's own target (`POPULATION_TARGET`,
//! two standard errors per sample), and `motion`, read by `readMotionResult` in
//! src/solver-types.ts:
//! - `axis`: "3d";
//! - per lab axis ("x", "y", "z"): `temperatureUK` and `temperatureErrorUK` (μK, one per sample:
//!   T = m Var(v)/k_B over one trap period of the axis around the sample, and its standard error),
//!   `meanVelocityMps`, `meanOccupation` (both over the same period; n̄ relative to
//!   that axis's `trapFrequencyKHz`, ω/2π at the focus), `momentum` (`velocitiesMps`, `samples`,
//!   `probabilities`) and `position` (`positionsUm`, ...) as the 1D block's distributions;
//! - `lossProbability`, one per sample, never decreasing;
//! - `classical`, one per sample: whether it lies in the validated regime (k_B T ≥ `CLASSICAL` ħω
//!   on every axis, motion-3d solve.rs); outside it the result is a reference only;
//! - `trajectories` and `timeStepUs`.
//!
//! Every batch's result is streamed first (`progress`: true) with the engine's estimate of the
//! remaining wall time (`diagnostics.estimatedRemainingMs`).
use crate::motion_core_adapter::{self,MotionModel};
use crate::{Model,Options,lattice_core_adapter};
use atom_motion_3d::solve::{Distributions,Solution,CONFIDENCE,POPULATION_TARGET};
use atom_motion_core::STATES;
use atom_population_core::lattice::pipeline::{LONGEST_TIME_US,MOST_SAMPLES};
use serde_json::{json,Value};
use std::time::Instant;

const ENGINE:&str="rust-motion-3d";
const AXES:[&str;3]=["x","y","z"];

fn per_axis<T:serde::Serialize>(values:&[[T;3]])->Value where T:Copy{
    json!(AXES.iter().enumerate().map(|(a,name)|(name.to_string(),json!(values.iter().map(|v|v[a]).collect::<Vec<T>>()))).collect::<serde_json::Map<String,Value>>())
}
fn distributions(d:&[Distributions;3],grid:&str)->Value{
    json!(AXES.iter().zip(d).map(|(name,x)|(name.to_string(),json!({grid:x.grid,"samples":x.samples,"probabilities":x.probabilities}))).collect::<serde_json::Map<String,Value>>())
}

/// The result in the one contract; `remaining` marks a streamed checkpoint (with the engine's
/// estimate of the wall time to the final result, ms).
fn result(model:&Model,times:&[f64],initial:&[f64],solution:&Solution,groups:&[String],remaining:Option<Option<f64>>,started:Instant)->Result<Value,String>{
    if solution.populations.len()!=times.len()||solution.populations.iter().any(|row|row.len()!=STATES||row.iter().any(|x|!x.is_finite())){return Err("The 3D motion engine returned populations that are not 32 finite values per time".into());}
    let(trace,initial_error)=lattice_core_adapter::row_errors(&solution.populations,initial)?;
    let errors:Vec<f64>=solution.sample_errors.iter().map(|e|CONFIDENCE*e).collect();
    let estimated=errors.iter().fold(0_f64,|a,&b|a.max(b));let converged=estimated<=POPULATION_TARGET;
    let mut result=json!({"states":model.states,"timesUs":times,"populations":solution.populations,
        "populationAccuracy":{"status":if converged{"estimated-converged"}else{"estimated-partial"},"requestedTolerance":POPULATION_TARGET,
            "sampleErrors":errors,"estimatedMaximumError":estimated,"scope":{"states":STATES,"samples":times.len(),"groups":groups}},
        "diagnostics":{"engine":ENGINE,"traceError":trace,"initialStateError":initial_error,"elapsedMs":started.elapsed().as_secs_f64()*1000.,"method":solution.method},
        "motion":{"axis":"3d","temperatureUK":per_axis(&solution.temperature_uk),"temperatureErrorUK":per_axis(&solution.temperature_errors_uk),
            "meanVelocityMps":per_axis(&solution.mean_velocity_mps),"meanOccupation":per_axis(&solution.mean_occupation),
            "trapFrequencyKHz":json!({"x":solution.trap_frequency_khz[0],"y":solution.trap_frequency_khz[1],"z":solution.trap_frequency_khz[2]}),
            "lossProbability":solution.loss_probability,"classical":solution.classical,
            "momentum":distributions(&solution.velocity,"velocitiesMps"),"position":distributions(&solution.position,"positionsUm"),
            "trajectories":solution.trajectories,"timeStepUs":solution.time_step_us}});
    if !converged{result["populationAccuracy"]["reason"]=json!(solution.reason.clone()
        .unwrap_or_else(||format!("The statistical error is {estimated:.2e} against the target {POPULATION_TARGET:e}")));}
    if let Some(remaining)=remaining{
        result["progress"]=json!(true);
        if let Some(ms)=remaining.filter(|ms|ms.is_finite()){result["diagnostics"]["estimatedRemainingMs"]=json!(ms.max(0.));}
    }
    Ok(result)
}

/// A 3D motion request: every batch is published as it is made, the final solution returned.
pub fn solve(model:&Model,motion:&MotionModel,options:&Options,publish:&mut dyn FnMut(&Value))->Result<Value,String>{
    let started=Instant::now();
    if options.mean_only||options.include_mean{return Err("A motion request is a trajectory alone: meanOnly and includeMean are false".into());}
    if !(options.duration_us.is_finite()&&options.duration_us>0.){return Err("The trajectory duration must be a positive finite time".into());}
    if options.duration_us>LONGEST_TIME_US{return Err(format!("The trajectory duration is limited to 10 s (requested {} s)",options.duration_us/1e6));}
    if !(2..=MOST_SAMPLES).contains(&options.samples){return Err(format!("A trajectory has 2 to {MOST_SAMPLES} samples (requested {})",options.samples));}
    if model.dimension!=STATES||model.states.len()!=STATES{return Err("rb87-full/2 requires the complete 32-state Rb87 model".into());}
    let(_,groups)=lattice_core_adapter::groups(model)?;let names:Vec<String>=groups.into_iter().map(|g|g.name).collect();
    let request=motion_core_adapter::request(model,motion,options)?;
    let initial:Vec<f64>=(0..STATES).map(|i|request.initial_density[i*STATES+i].re).collect();
    let mut failure:Option<String>=None;
    let mut observe=|checkpoint:&Solution,remaining:Option<f64>|{if failure.is_some(){return;}
        match result(model,&request.times_us,&initial,checkpoint,&names,Some(remaining),started){Ok(value)=>publish(&value),Err(e)=>failure=Some(e)}};
    let solution=atom_motion_3d::solve::solve(&request,&mut observe)?;
    if let Some(e)=failure{return Err(e);}
    result(model,&request.times_us,&initial,&solution,&names,None,started)
}
