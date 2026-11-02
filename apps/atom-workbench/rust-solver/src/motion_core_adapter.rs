//! The protocol bridge to the motion core (`atom_motion_core`): an rb87-full/2
//! request whose model carries the motion block (src/full-master-model.ts
//! `MotionModel`), the atom in a tweezer moving radially along one lab axis. It
//! is a trajectory alone -- no long-time mean -- on the request's grid, at the
//! fixed population tolerance. A block with a plane for its axis is 2D motion,
//! served by motion_gpu_adapter.rs on the same request (`prepare`).
//!
//! # Result contract
//! The lattice adapter's for a trajectory (lattice_core_adapter.rs): `states`,
//! `timesUs`, `populations` (the internal populations averaged over position),
//! `populationAccuracy` and `diagnostics` (`engine`, `traceError`,
//! `initialStateError`, `elapsedMs`, `method`), without `meanHorizons`; and
//! `motion`, read by `readMotionResult` in src/solver-types.ts:
//! - `axis`, the request's;
//! - `temperatureUK` (T = m Var(v)/k_B along the axis, uK) and
//!   `meanVelocityMps` (m/s), one per sample;
//! - `trapFrequencyKHz`, the trap frequency ω/2π at the focus along the axis
//!   (kHz, positive), and `meanOccupation`, one n̄ per sample relative to it;
//! - `lossProbability`, one per sample: the probability that the atom has left
//!   the trap by then, from 0 to 1 and never decreasing;
//! - `continuedFromUs`, present when the samples after it are the core's
//!   continuation (its prediction from the simulated history), not simulated:
//!   the last simulated time (us);
//! - `momentum`: `velocitiesMps` (the grid, strictly increasing), `samples`
//!   (increasing sample indices, the last sample among them) and
//!   `probabilities` (one distribution per listed sample: the probability at
//!   each grid velocity, nonnegative, summing to one); `position` likewise,
//!   with `positionsUm` (from the focus).
//!
//! Every output but `lossProbability` describes the atom still held.
//!
//! Every checkpoint of the core's run is streamed first (`progress`: true) in
//! the same shape, with the core's estimate of the remaining wall time
//! (`diagnostics.estimatedRemainingMs`, ms) when it has one, and
//! `diagnostics.estimatedRemainingAtMost`: true while that is the time to the end
//! and the continuation may still stop the run earlier.
use crate::{Model,Operator,Options,lattice_core_adapter};
use atom_motion_core::{Beam,Complex,Matrix,Remaining,Request,Solution,STATES};
use atom_population_core::Lapack;
use atom_population_core::lattice::accuracy::POPULATION_TOLERANCE;
use atom_population_core::lattice::pipeline::{LONGEST_TIME_US,MOST_SAMPLES};
use serde::Deserialize;
use serde_json::{json,Value};
use std::time::Instant;

const ENGINE:&str="rust-motion-core";

/// The motion block of the model, as the TypeScript model writes it.
#[derive(Clone,Deserialize)]
#[serde(rename_all="camelCase",deny_unknown_fields)]
pub struct MotionModel{
    /// A lab axis (`x`, `y`, `z`: 1D motion along it) or a plane (`yz`, `zx`, `xy`: 2D motion in
    /// it, named by the first two axes of its frame).
    pub(crate) axis:String,
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
impl MotionModel{
    /// 2D motion: the block's axis is a plane.
    pub(crate) fn is_plane(&self)->bool{matches!(self.axis.as_str(),"yz"|"zx"|"xy")}
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

/// The core's request: the model's motion block, its initial density and the request's grid.
fn request(model:&Model,motion:&MotionModel,options:&Options)->Result<Request,String>{
    let finite=|x:f64|x.is_finite();
    if !matches!(motion.axis.as_str(),"x"|"y"|"z")&&!motion.is_plane(){return Err("The motion axis is x, y or z, or the plane yz, zx or xy".into());}
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
    let [a,b,c]=motion.emission.as_slice()else{return Err("The motion block has three emission operators, one per axis of the motion frame".into())};
    let density=lattice_core_adapter::initial_density(model,&Lapack::load()?)?;
    Ok(Request{states:model.states.clone(),hamiltonian:matrix(&motion.hamiltonian,"The motion block's H0")?,frame_frequencies_mhz:motion.frame_frequencies_mhz.clone(),
        frequencies_mhz:frequencies,beams,emission:[matrix(a,"An emission operator")?,matrix(b,"An emission operator")?,matrix(c,"An emission operator")?],
        field_vector_g:motion.field_vector_g,mass_kg:motion.mass_kg,initial_density:density.iter().map(|z|Complex{re:z.re,im:z.im}).collect(),
        temperature_uk:motion.temperature_uk,times_us:lattice_core_adapter::grid(options),tolerance:POPULATION_TOLERANCE})
}

/// A motion request, 1D or 2D, checked: a trajectory alone of the complete 32-state model on a
/// finite grid at the fixed tolerance. The engine's request and the population groups its errors
/// cover.
pub(crate) fn prepare(model:&Model,motion:&MotionModel,options:&Options)->Result<(Request,Vec<lattice_core_adapter::Group>),String>{
    if options.mean_only||options.include_mean{return Err("A motion request is a trajectory alone: meanOnly and includeMean are false".into());}
    if options.population_tolerance!=POPULATION_TOLERANCE{return Err(format!("populationTolerance is fixed at {POPULATION_TOLERANCE} (requested {})",options.population_tolerance));}
    if !(options.duration_us.is_finite()&&options.duration_us>0.){return Err("The trajectory duration must be a positive finite time".into());}
    if options.duration_us>LONGEST_TIME_US{return Err(format!("The trajectory duration is limited to 10 s (requested {} s)",options.duration_us/1e6));}
    if !(2..=MOST_SAMPLES).contains(&options.samples){return Err(format!("A trajectory has 2 to {MOST_SAMPLES} samples (requested {})",options.samples));}
    if model.dimension!=STATES||model.states.len()!=STATES{return Err("rb87-full/2 requires the complete 32-state Rb87 model".into());}
    let(_,groups)=lattice_core_adapter::groups(model)?;
    Ok((request(model,motion,options)?,groups))
}
/// Why a result is above the tolerance: the engine's reason, or its largest error.
pub(crate) fn above_tolerance(reason:Option<&String>,estimated:f64)->String{
    reason.cloned().unwrap_or_else(||format!("The error bound is {estimated:.2e} against the tolerance {POPULATION_TOLERANCE:e}"))
}

/// Distributions at listed times, checked against the contract: on a strictly increasing grid,
/// at increasing times ending with the last, each `points` probabilities (the grid's own count,
/// or its square for a map of the plane) summing to one.
pub(crate) fn check_distributions(grid:&[f64],samples:&[usize],probabilities:&[Vec<f64>],points:usize,times:usize,what:&str)->Result<(),String>{
    if grid.len()<2||grid.iter().any(|v|!v.is_finite())||grid.windows(2).any(|w|w[1]<=w[0]){return Err(format!("a {what} grid that is not strictly increasing"));}
    if samples.last()!=Some(&(times-1))||samples.windows(2).any(|w|w[1]<=w[0])||probabilities.len()!=samples.len(){
        return Err(format!("{what} distributions not at increasing times ending with the last"));}
    if probabilities.iter().any(|p|p.len()!=points||p.iter().any(|&x|!(0. ..=1.).contains(&x))||(p.iter().sum::<f64>()-1.).abs()>lattice_core_adapter::TRACE_LIMIT){
        return Err(format!("a {what} distribution that is not a probability on its grid"));}
    Ok(())
}

/// The core's solution, checked against the contract before it is reported.
fn check(solution:&Solution,times:usize)->Result<(),String>{
    let bad=|what:&str|->Result<(),String>{Err(format!("The motion core returned {what}"))};
    let series=|v:&[f64],nonnegative:bool|v.len()==times&&v.iter().all(|&x|x.is_finite()&&(!nonnegative||x>=0.));
    if solution.populations.len()!=times||solution.populations.iter().any(|row|row.len()!=STATES||row.iter().any(|x|!x.is_finite())){return bad("populations that are not 32 finite values per time");}
    if !series(&solution.sample_errors,true)||!series(&solution.temperature_uk,true)||!series(&solution.mean_velocity_mps,false)||!series(&solution.mean_occupation,false){
        return bad("per-time errors, temperatures, mean velocities or occupations that are not one finite value per time");}
    if !(solution.trap_frequency_khz.is_finite()&&solution.trap_frequency_khz>0.){return bad("a trap frequency that is not positive");}
    if solution.continued_from_us.is_some_and(|t|!(t.is_finite()&&t>=0.)){return bad("a continuation start that is not a time of the run");}
    let loss=&solution.loss_probability;
    if !series(loss,true)||loss.iter().any(|&p|p>1.)||loss.windows(2).any(|w|w[1]<w[0]){return bad("loss probabilities that are not probabilities never decreasing in time");}
    for(m,what)in [(&solution.momentum,"momentum"),(&solution.position,"position")]{
        if let Err(e)=check_distributions(&m.grid,&m.samples,&m.probabilities,m.grid.len(),times,what){return bad(&e);}
    }
    Ok(())
}

/// The result in the one contract; `remaining` marks a streamed checkpoint (with the core's
/// estimate of the wall time to the final result when it has one).
fn result(model:&Model,motion:&MotionModel,request:&Request,solution:&Solution,groups:&[String],remaining:Option<Option<Remaining>>,started:Instant)->Result<Value,String>{
    let times=request.times_us.len();check(solution,times)?;
    let initial:Vec<f64>=(0..STATES).map(|i|request.initial_density[i*STATES+i].re).collect();
    let(trace,initial_error)=lattice_core_adapter::row_errors(&solution.populations,&initial)?;
    let estimated=solution.sample_errors.iter().fold(0_f64,|a,&b|a.max(b));let converged=estimated<=POPULATION_TOLERANCE;
    let(m,x)=(&solution.momentum,&solution.position);
    let mut result=json!({"states":model.states,"timesUs":request.times_us,"populations":solution.populations,
        "populationAccuracy":{"status":if converged{"estimated-converged"}else{"estimated-partial"},"requestedTolerance":POPULATION_TOLERANCE,
            "sampleErrors":solution.sample_errors,"estimatedMaximumError":estimated,"scope":{"states":STATES,"samples":times,"groups":groups}},
        "diagnostics":{"engine":ENGINE,"traceError":trace,"initialStateError":initial_error,"elapsedMs":started.elapsed().as_secs_f64()*1000.,"method":solution.method},
        "motion":{"axis":motion.axis,"temperatureUK":solution.temperature_uk,"meanVelocityMps":solution.mean_velocity_mps,
            "trapFrequencyKHz":solution.trap_frequency_khz,"meanOccupation":solution.mean_occupation,"lossProbability":solution.loss_probability,
            "momentum":{"velocitiesMps":m.grid,"samples":m.samples,"probabilities":m.probabilities},
            "position":{"positionsUm":x.grid,"samples":x.samples,"probabilities":x.probabilities}}});
    if let Some(t)=solution.continued_from_us{result["motion"]["continuedFromUs"]=json!(t);}
    // A result above the tolerance says why it stopped there.
    if !converged{result["populationAccuracy"]["reason"]=json!(above_tolerance(solution.reason.as_ref(),estimated));}
    if let Some(remaining)=remaining{
        result["progress"]=json!(true);
        if let Some(r)=remaining.filter(|r|r.ms.is_finite()){result["diagnostics"]["estimatedRemainingMs"]=json!(r.ms.max(0.));
            if r.at_most{result["diagnostics"]["estimatedRemainingAtMost"]=json!(true);}}
    }
    Ok(result)
}

/// A 1D motion request through the motion core: every checkpoint is
/// published as it is made, the final solution returned.
pub fn solve(model:&Model,motion:&MotionModel,options:&Options,publish:&mut dyn FnMut(&Value))->Result<Value,String>{
    let started=Instant::now();
    let(request,groups)=prepare(model,motion,options)?;let names:Vec<String>=groups.into_iter().map(|g|g.name).collect();
    let mut failure:Option<String>=None;
    let mut observe=|checkpoint:&Solution,remaining:Option<Remaining>|{if failure.is_some(){return;}
        match result(model,motion,&request,checkpoint,&names,Some(remaining),started){Ok(value)=>publish(&value),Err(e)=>failure=Some(e)}};
    let solution=atom_motion_core::solve(&request,&mut observe)?;
    if let Some(e)=failure{return Err(e);}
    result(model,motion,&request,&solution,&names,None,started)
}
