//! The protocol bridge to the 2D motion engine (`atom_motion_gpu`): an rb87-full/2 request whose
//! model's motion block has a plane for its axis (src/full-master-model.ts `MotionModel`) -- the
//! atom in a tweezer moving in the plane through the focus perpendicular to the tweezer -- solved by
//! quantum trajectories on an NVIDIA GPU. The request is the motion core's
//! (motion_core_adapter.rs `prepare`) in the frame of the tweezer's axis, whose first two axes are
//! the plane's. It is a trajectory alone -- no long-time mean -- on the request's grid, at the fixed
//! population tolerance.
//!
//! # Result contract
//! The motion core adapter's for the trajectory: `states`, `timesUs`, `populations` (the internal
//! populations averaged over position and over the trajectories), `populationAccuracy` and
//! `diagnostics` (`engine`, `traceError`, `initialStateError`, `elapsedMs`, `method`); and `motion`,
//! read by `readMotionResult` in src/solver-types.ts:
//! - `axis`, the request's plane, and `axes`, its two lab axes in order (a, b);
//! - per axis, one value per sample: `temperatureUK` (T = m Var(v)/k_B along the axis, uK) and
//!   `temperatureErrorUK` (its standard error), `meanVelocityMps` (m/s) and `meanOccupation` (n̄
//!   relative to the axis's trap frequency);
//! - `trapFrequencyKHz`, the trap frequency ω/2π at the focus along each axis (kHz, positive);
//! - `lossProbability`, one per sample: the probability that the atom has left the trap by then,
//!   from 0 to 1 and never decreasing;
//! - `momentum`: per axis the velocity marginal, as the motion core's `momentum` (`velocitiesMps`,
//!   `samples`, `probabilities`), both on one grid at the same samples;
//! - `position`: `positionsUm` (the uniform grid along both axes, from the focus), `samples` and
//!   `probabilities` (one map per listed sample, row-major: a row of the grid's count of
//!   probabilities along a for each position along b, nonnegative, summing to one);
//! - `trajectories`, how many the estimates rest on.
//!
//! The rows of `populations`: the first is the initial density's -- the engine's observation at
//! t = 0 holds the excited states' share of its adiabatic elimination, which the atom acquires only
//! within about 1/Γ of the start -- and every other is the engine's, made in single precision,
//! projected onto the probability simplex as the population core's rows are
//! (lattice_core_adapter.rs). Each sample's error (`sampleErrors`) is the standard error of the
//! trajectory average plus the projection's change; the first row's is zero.
//!
//! Every output but `lossProbability` describes the atom still held.
//!
//! Every batch's merged result is streamed first (`progress`: true) in the same shape, with the
//! run's estimate of the wall time left (`diagnostics.estimatedRemainingMs`, ms) once its errors
//! project the trajectories needed. Between them, at least every `BEAT_SECONDS` however long a
//! batch runs, a `running` message (main.rs): `simulatedUs` of `durationUs` in the running batch,
//! the `trajectories` finished and, once there is one, `estimatedRemainingMs`.
use crate::motion_core_adapter::{self,MotionModel};
use crate::{Model,lattice_core_adapter,Options};
use atom_motion_core::{Request,STATES};
use atom_motion_gpu::solution::{Running,Solution};
use atom_population_core::lattice::accuracy::{self,POPULATION_TOLERANCE};
use atom_population_core::lattice::pipeline;
use serde_json::{json,Value};
use std::time::Instant;

const ENGINE:&str="rust-motion-gpu";

/// The engine's solution, checked against the contract before it is reported.
fn check(solution:&Solution,times:usize)->Result<(),String>{
    let bad=|what:&str|->Result<(),String>{Err(format!("The 2D motion engine returned {what}"))};
    if solution.populations.len()!=times||solution.populations.iter().any(|row|row.len()!=STATES||row.iter().any(|x|!x.is_finite())){return bad("populations that are not 32 finite values per time");}
    if solution.sample_errors.len()!=times||solution.sample_errors.iter().any(|&x|!(x.is_finite()&&x>=0.)){return bad("standard errors that are not one nonnegative value per time");}
    let pairs=|v:&[[f64;2]],nonnegative:bool|v.len()==times&&v.iter().flatten().all(|&x|x.is_finite()&&(!nonnegative||x>=0.));
    if !pairs(&solution.temperature_uk,true)||!pairs(&solution.temperature_errors_uk,true)||!pairs(&solution.mean_velocity_mps,false)||!pairs(&solution.mean_occupation,false){
        return bad("temperatures, their errors, mean velocities or occupations that are not one finite value per axis and time");}
    if solution.trap_frequency_khz.iter().any(|&f|!(f.is_finite()&&f>0.)){return bad("a trap frequency that is not positive");}
    let loss=&solution.loss_probability;
    if loss.len()!=times||loss.iter().any(|&p|!(0. ..=1.).contains(&p))||loss.windows(2).any(|w|w[1]<w[0]){return bad("loss probabilities that are not probabilities never decreasing in time");}
    for m in &solution.velocity{
        if let Err(e)=motion_core_adapter::check_distributions(&m.grid,&m.samples,&m.probabilities,m.grid.len(),times,"velocity"){return bad(&e);}
    }
    if solution.velocity[0].grid!=solution.velocity[1].grid||solution.velocity[0].samples!=solution.velocity[1].samples{return bad("velocity marginals on different grids or at different times");}
    let p=&solution.position;
    if let Err(e)=motion_core_adapter::check_distributions(&p.grid,&p.samples,&p.probabilities,p.grid.len()*p.grid.len(),times,"position"){return bad(&e);}
    let step=(p.grid[p.grid.len()-1]-p.grid[0])/(p.grid.len()-1) as f64;
    if p.grid.iter().enumerate().any(|(i,&x)|(x-(p.grid[0]+i as f64*step)).abs()>1e-9*step){return bad("a position grid that is not uniform");}
    if solution.trajectories==0{return bad("no trajectories");}
    Ok(())
}

/// The result in the one contract; `last` false marks a streamed checkpoint.
fn result(model:&Model,motion:&MotionModel,request:&Request,solution:&Solution,groups:&[lattice_core_adapter::Group],last:bool,started:Instant)->Result<Value,String>{
    let times=request.times_us.len();check(solution,times)?;
    let initial:Vec<f64>=(0..STATES).map(|i|request.initial_density[i*STATES+i].re).collect();
    let indices:Vec<Vec<usize>>=groups.iter().map(|g|g.indices.clone()).collect();
    let mut populations=vec![initial.clone()];let mut errors=solution.sample_errors.clone();errors[0]=0.;
    for(row,error)in solution.populations.iter().zip(&mut errors).skip(1){
        let projected=pipeline::simplex(row);*error+=accuracy::metric(row,&projected,&indices);populations.push(projected);
    }
    let(trace,initial_error)=lattice_core_adapter::row_errors(&populations,&initial)?;
    let estimated=errors.iter().fold(0_f64,|a,&b|a.max(b));let converged=estimated<=POPULATION_TOLERANCE;
    let axes:Vec<String>=motion.axis.chars().map(String::from).collect();
    let per_axis=|v:&[[f64;2]]|[v.iter().map(|x|x[0]).collect::<Vec<f64>>(),v.iter().map(|x|x[1]).collect()];
    let marginal=|d:usize|{let m=&solution.velocity[d];json!({"velocitiesMps":m.grid,"samples":m.samples,"probabilities":m.probabilities})};
    let x=&solution.position;
    let mut result=json!({"states":model.states,"timesUs":request.times_us,"populations":populations,
        "populationAccuracy":{"status":if converged{"estimated-converged"}else{"estimated-partial"},"requestedTolerance":POPULATION_TOLERANCE,
            "sampleErrors":errors,"estimatedMaximumError":estimated,"scope":{"states":STATES,"samples":times,"groups":groups.iter().map(|g|&g.name).collect::<Vec<_>>()}},
        "diagnostics":{"engine":ENGINE,"traceError":trace,"initialStateError":initial_error,"elapsedMs":started.elapsed().as_secs_f64()*1000.,"method":solution.method},
        "motion":{"axis":motion.axis,"axes":axes,"temperatureUK":per_axis(&solution.temperature_uk),"temperatureErrorUK":per_axis(&solution.temperature_errors_uk),
            "meanVelocityMps":per_axis(&solution.mean_velocity_mps),"trapFrequencyKHz":solution.trap_frequency_khz,"meanOccupation":per_axis(&solution.mean_occupation),
            "lossProbability":solution.loss_probability,"momentum":[marginal(0),marginal(1)],
            "position":{"positionsUm":x.grid,"samples":x.samples,"probabilities":x.probabilities},"trajectories":solution.trajectories}});
    // A result above the tolerance says why the run stopped there.
    if !converged{result["populationAccuracy"]["reason"]=json!(motion_core_adapter::above_tolerance(solution.reason.as_ref(),estimated));}
    if !last{
        result["progress"]=json!(true);
        if let Some(ms)=solution.remaining_ms.filter(|ms|ms.is_finite()){result["diagnostics"]["estimatedRemainingMs"]=json!(ms.max(0.));}
    }
    Ok(result)
}

/// A running batch's report as the `running` message's body (main.rs sends it as such).
fn running(report:&Running)->Value{
    let mut body=json!({"simulatedUs":report.simulated_us,"durationUs":report.duration_us,"trajectories":report.trajectories});
    if let Some(ms)=report.remaining_ms.filter(|ms|ms.is_finite()){body["estimatedRemainingMs"]=json!(ms.max(0.));}
    json!({"running":body})
}

/// A 2D motion request through the GPU engine: every batch's merged result is published as it is
/// made (a result that fails its check stops the run), the final one returned. Without an NVIDIA
/// GPU and driver the engine says so, in the words the app shows.
pub fn solve(model:&Model,motion:&MotionModel,options:&Options,publish:&mut dyn FnMut(&Value))->Result<Value,String>{
    let started=Instant::now();
    let(request,groups)=motion_core_adapter::prepare(model,motion,options)?;
    let mut failure:Option<String>=None;let publish=std::cell::RefCell::new(publish);
    let mut observe=|checkpoint:&Solution|match result(model,motion,&request,checkpoint,&groups,false,started){
        Ok(value)=>{(publish.borrow_mut())(&value);true},Err(e)=>{failure=Some(e);false}};
    let mut beat=|report:&Running|(publish.borrow_mut())(&running(report));
    let solution=atom_motion_gpu::solve::solve(&request,&mut observe,&mut beat)?;
    if let Some(e)=failure{return Err(e);}
    result(model,motion,&request,&solution,&groups,true,started)
}
