//! A 3D motion solve: the model, the time step from a step-doubling check, then trajectories in
//! sequential batches until the standard errors meet the targets at every output time (as QuTiP's
//! mcsolve does with its target tolerance), each batch's result handed out as a checkpoint with
//! the wall time the remaining trajectories will take.
//!
//! # Targets
//! The semiclassical method is accurate to about `TEMPERATURE_TARGET` in T and
//! `POPULATION_TARGET` in the populations in its regime; the trajectories are run until two
//! standard errors of every population and the loss lie within those targets at every output
//! time, and of every temperature (each axis, relative) at the output times in the regime --
//! outside it the method's own error is larger and the temperature is a reference only.
//!
//! # Time step
//! The trap's harmonic part is moved exactly (engine.rs); what the step must resolve is the light's
//! pattern crossed by the moving atom and the internal state's response to it, which carries the
//! light's friction. The step is chosen by step doubling on the deterministic flow (as an adaptive
//! ODE solver's local error, QuTiP's way): states are sampled along a pilot of a few trajectories
//! at a first step (the phase k_max v_T of the light's pattern at the initial temperature advancing
//! sqrt(η) per step, η the a-priori share), and from each one step of dt is compared with two of
//! dt/2, without jumps. The step is accepted when (4/3 of) the differences meet their shares:
//! - the internal state's populations, summed over the steps between two jumps (the pilot's mean
//!   waiting time), within η ε of the app's population tolerance ε -- the internal state's lag
//!   carries the friction, so this also keeps the temperature's step error near η of its target;
//! - the step's light impulse, relative, within η of the temperature target.
//! Both scale as dt², so a step that misses is shrunk by the square root of its excess and checked
//! again; one that meets them with room is grown the same way.

//! # Regime
//! A point particle has no zero-point motion: below a few vibrational quanta per axis the method
//! cools too far. Against the 1D GMW references (the tweezer PGC TW2) T is within 3 % down to
//! k_B T ≈ 4.5 ħω, within 10 % at 3.5 ħω and 13 % low at 3 ħω; against the 2D quantum trajectories
//! of the same scene (x and y free) within 8 % at 4.6 ħω and 15 % low at 3.1 ħω. Each output time
//! says whether it is in the regime k_B T ≥ `CLASSICAL` ħω on every axis.
use crate::engine::{Engine,Settings,Trajectory};
use crate::model::Model;
use crate::stats::{summarize,Estimate,Scales,KT_PER_UK};
use crate::{sample_states,simulate,Initial,Outcome};
use atom_motion_core::{Complex,Request,STATES};
use atom_population_core::lattice::accuracy::{A_PRIORI_SHARE,POPULATION_TOLERANCE};
use std::time::Instant;

/// The method's accuracy targets: T relative, populations and loss absolute.
pub const TEMPERATURE_TARGET:f64=0.10;
pub const POPULATION_TARGET:f64=0.05;
/// Statistics are run until this many standard errors lie within the targets.
pub const CONFIDENCE:f64=2.;
/// The trap ends where its depth has fallen to 1/ESCAPE of the focus's (the 1D engine's choice).
pub const ESCAPE:f64=20.;
/// The validated regime: k_B T ≥ CLASSICAL ħω along every axis (T the engine's own), where the 1D
/// and 2D quantum references put T within about 10 % (validation, see the module documentation).
pub const CLASSICAL:f64=4.;
/// The most trajectories a solve runs; a solve stopped there returns its result with the reason.
pub const MOST_TRAJECTORIES:u64=1<<14;
/// The most trajectories of a group (a task and a jackknife group): a batch is cut into at least
/// two groups per thread, so a small batch fills the threads and the jackknife has enough groups.
const GROUP:u64=8;
fn group(count:u64,threads:u64)->u64{(count/(2*threads)).clamp(1,GROUP)}

pub const NOT_CONFINED:&str="沿某个方向没有约束原子的势阱：3D 运动计算光镊中的单个原子，需要聚焦在原子处的光（光镊）";

/// Distributions along one axis on one grid, at some of the times.
#[derive(Clone,Debug,PartialEq)]
pub struct Distributions{pub grid:Vec<f64>,pub samples:Vec<usize>,pub probabilities:Vec<Vec<f64>>}

/// A 3D motion solve's result, at every time of the request (per axis: the request's frame).
/// Everything but `loss_probability` describes the atom still held.
#[derive(Clone,Debug)]
pub struct Solution{
    pub populations:Vec<Vec<f64>>,
    /// Per time, the largest standard error of the populations and the loss.
    pub sample_errors:Vec<f64>,
    pub temperature_uk:Vec<[f64;3]>,
    pub temperature_errors_uk:Vec<[f64;3]>,
    pub mean_velocity_mps:Vec<[f64;3]>,
    /// ω_a/2π (kHz) at the focus of the reference state's trap, and n̄ per axis relative to it.
    pub trap_frequency_khz:[f64;3],
    pub mean_occupation:Vec<[f64;3]>,
    pub loss_probability:Vec<f64>,
    /// Per axis: velocities (m/s) and positions (um from the focus).
    pub velocity:[Distributions;3],
    pub position:[Distributions;3],
    /// Per time: in the validated regime (k_B T ≥ CLASSICAL ħω on every axis).
    pub classical:Vec<bool>,
    pub trajectories:u64,
    pub time_step_us:f64,
    pub method:String,
    pub reason:Option<String>,
}

/// The largest ratio of CONFIDENCE standard errors to the targets over every output: the
/// populations and the loss everywhere, the temperatures where the method holds (`classical`).
fn worst_ratio(estimates:&[Estimate],classical:&dyn Fn(&Estimate)->bool)->f64{
    estimates.iter().map(|e|{
        // The temperature to its target where the method holds; outside, a reference only.
        let t=if classical(e){(0..3).map(|a|if e.temperature[a]>0.{e.temperature_error[a]/(TEMPERATURE_TARGET*e.temperature[a])}else{0.}).fold(0.,f64::max)}else{0.};
        let p=e.population_errors.iter().fold(e.loss_error,|m,&x|m.max(x))/POPULATION_TARGET;
        CONFIDENCE*t.max(p)
    }).fold(0.,f64::max)
}

/// The distribution sample indices: evenly spread over the outputs, the first and last among them.
fn listed(outputs:usize)->Vec<usize>{
    let count=outputs.min(21);let mut v:Vec<usize>=(0..count).map(|i|((i*(outputs-1)) as f64/(count-1).max(1) as f64).round() as usize).collect();v.dedup();v
}

/// Histograms of one axis of the samples at the listed times on one grid symmetric about 0 that
/// holds all of them (the bins' centres; an odd count, 0 at the centre).
fn histograms(samples:&[Vec<([f64;3],[f64;3])>],listed:&[usize],axis:usize,value:&dyn Fn(&([f64;3],[f64;3]))->f64)->Distributions{
    let reach=samples.iter().flatten().map(|s|value(s).abs()).fold(0.,f64::max).max(f64::MIN_POSITIVE);
    let count=samples.iter().map(|s|s.len()).max().unwrap_or(0);
    // Bins: about the square root of the samples, odd, at least 21.
    let bins=(((count as f64).sqrt() as usize)|1).max(21);let width=2.*reach/(bins-1) as f64;
    let grid:Vec<f64>=(0..bins).map(|i|(i as f64-(bins/2) as f64)*width).collect();
    let probabilities=samples.iter().map(|s|{
        let mut p=vec![0.;bins];if s.is_empty(){p[bins/2]=1.;return p;}
        for x in s{let i=((value(x)/width).round() as isize+(bins/2) as isize).clamp(0,bins as isize-1) as usize;p[i]+=1./s.len() as f64;}p
    }).collect();
    let _=axis;Distributions{grid,samples:listed.to_vec(),probabilities}
}

/// Solves a request. `checkpoint` receives every batch's complete result and the estimated wall
/// time (ms) the rest will take.
pub fn solve(request:&Request,checkpoint:&mut dyn FnMut(&Solution,Option<f64>))->Result<Solution,String>{
    let model=Model::new(request)?;
    if model.focused.is_empty(){return Err(NOT_CONFINED.into());}
    let initial=Initial::new(&model,request)?;
    // The reference state: the largest initial population; its trap sets ω per axis and the escape.
    let(block,state,_)=*initial.states.iter().max_by(|a,b|a.2.total_cmp(&b.2)).unwrap();
    let frequencies=model.trap_frequencies(block,state);
    if frequencies.iter().any(|&w|!(w>0.)){return Err(NOT_CONFINED.into());}
    let times=&request.times_us;let outputs=times.len();
    if outputs<2||times[0]!=0.{return Err("A 3D motion solve needs at least two times, the first at 0".into());}
    let interval=times[1]-times[0];
    if times.windows(2).any(|w|((w[1]-w[0])-interval).abs()>1e-9*interval){return Err("A 3D motion solve needs evenly spaced times".into());}
    let escape=1./ESCAPE;
    let listed=listed(outputs);let scales=Scales{hbar_over_mass:model.hbar_over_mass,frequencies};
    let seed=0x3D_5EED;
    // The first step: the light pattern's phase k_max v_T per step at sqrt(η × the temperature
    // target), or without light the trap's own ω.
    let thermal=(KT_PER_UK*request.temperature_uk*model.hbar_over_mass).sqrt();
    let pattern=model.carriers.iter().map(|c|c.wave_number).fold(0.,f64::max)*thermal;
    let rate=if pattern>0.{pattern}else{frequencies.iter().copied().fold(0.,f64::max)};
    let mut steps=((interval*rate/(A_PRIORI_SHARE*TEMPERATURE_TARGET).sqrt()).ceil() as usize).max(1);
    let threads=rayon::current_num_threads() as u64;let pilot=2*threads;
    let settings=|steps:usize|Settings{dt:interval/steps as f64,free:[true;3],escape_intensity:escape,frequencies};
    let duration=times[outputs-1];
    let(states,jumps)=sample_states(&model,&settings(steps),&initial,seed,pilot,steps,outputs);
    let waiting=pilot as f64*duration/(jumps.max(1) as f64);
    for _ in 0..8{
        let engine=Engine{model:&model,settings:settings(steps)};let excess=step_excess(&engine,&states,waiting);
        let next=((steps as f64*excess.sqrt()).ceil() as usize).max(1);
        if excess<=1.&&next>=steps/2{break;}
        if next==steps{break;}
        steps=next;
    }
    let dt=interval/steps as f64;let settings=settings(steps);
    let clock=Instant::now();
    let first=simulate(&model,&settings,&initial,seed,0,pilot,group(pilot,threads),steps,outputs,&listed);
    let wall=clock.elapsed().as_secs_f64();
    let classical=|e:&Estimate|(0..3).all(|a|KT_PER_UK*e.temperature[a]>=CLASSICAL*frequencies[a]);
    // Batches until the targets are met.
    let mut outcome=first;let mut done=pilot;let mut per_trajectory=wall/pilot as f64;
    loop{
        let estimates=summarize(&outcome.groups,&scales);
        let ratio=worst_ratio(&estimates,&classical);
        let needed=((done as f64*ratio*ratio).ceil() as u64).min(MOST_TRAJECTORIES);
        let mut solution=assemble(&model,&estimates,&outcome,&listed,frequencies,done,dt,&request.initial_density);
        if needed<=done{
            if ratio>1.{solution.reason=Some(format!("统计误差为目标的 {ratio:.1} 倍：轨迹数已达上限 {MOST_TRAJECTORIES}"));}
            return Ok(solution);
        }
        checkpoint(&solution,Some(per_trajectory*(needed-done) as f64*1e3));
        let next=(needed-done).clamp(2*threads,3*done);
        let clock=Instant::now();
        let more=simulate(&model,&settings,&initial,seed,done,next,group(next,threads),steps,outputs,&listed);
        per_trajectory=clock.elapsed().as_secs_f64()/next as f64;
        outcome.groups.extend(more.groups);for(a,b)in outcome.samples.iter_mut().zip(more.samples){a.extend(b);}outcome.jumps+=more.jumps;
        done+=next;
    }
}

/// The step-doubling excess of a step (module documentation): the largest of the internal state's
/// accumulated population error over η ε and the light impulse's relative error over η × the
/// temperature target, from one step against two halves on the sampled states; 1 meets them.
fn step_excess(engine:&Engine,states:&[Trajectory],waiting:f64)->f64{
    let dt=engine.settings.dt;
    let(mut population,mut difference,mut impulse)=(0.,0.,0.);
    for t in states{let(motion,internal)=engine.doubling(t,dt);difference+=motion[0];impulse+=motion[1];population+=internal[0];}
    let richardson=4./3.;
    let accumulated=richardson*population/states.len().max(1) as f64*(waiting/dt).max(1.);
    let relative=if impulse>0.{richardson*difference/impulse}else{0.};
    (accumulated/(A_PRIORI_SHARE*POPULATION_TOLERANCE)).max(relative/(A_PRIORI_SHARE*TEMPERATURE_TARGET))
}

/// The solution of the estimates so far.
/// At t = 0 the populations are the initial state's, as the request gives them (the light acts after).
fn assemble(model:&Model,estimates:&[Estimate],outcome:&Outcome,listed:&[usize],frequencies:[f64;3],trajectories:u64,dt:f64,initial:&[Complex])->Solution{
    let hm=model.hbar_over_mass;
    let velocity=std::array::from_fn(|a|histograms(&outcome.samples,listed,a,&|s|s.1[a]*hm));
    let position=std::array::from_fn(|a|histograms(&outcome.samples,listed,a,&|s|s.0[a]));
    let first:Vec<f64>=(0..STATES).map(|i|initial[i*STATES+i].re).collect();
    Solution{
        populations:estimates.iter().enumerate().map(|(i,e)|if i==0{first.clone()}else{e.populations.clone()}).collect(),
        sample_errors:estimates.iter().enumerate().map(|(i,e)|if i==0{0.}else{e.population_errors.iter().fold(e.loss_error,|m,&x|m.max(x))}).collect(),
        temperature_uk:estimates.iter().map(|e|e.temperature).collect(),
        temperature_errors_uk:estimates.iter().map(|e|e.temperature_error).collect(),
        mean_velocity_mps:estimates.iter().map(|e|e.mean_velocity).collect(),
        trap_frequency_khz:frequencies.map(|w|w/std::f64::consts::TAU*1e3),
        mean_occupation:estimates.iter().map(|e|e.occupation).collect(),
        loss_probability:{let mut worst=0.;estimates.iter().map(|e|{worst=e.loss.max(worst);worst}).collect()},
        velocity,position,
        classical:estimates.iter().map(|e|(0..3).all(|a|KT_PER_UK*e.temperature[a]>=CLASSICAL*frequencies[a])).collect(),
        trajectories,time_step_us:dt,
        method:format!("半经典 3D：经典质点（位置、动量）携带消去激发态后的基态内态，量子跳跃（自发辐射按偶极辐射方向图反冲，吸收按光束反冲并保持精确的平均力），{trajectories} 条轨迹，步长 {dt:.3} μs"),
        reason:None,
    }
}
