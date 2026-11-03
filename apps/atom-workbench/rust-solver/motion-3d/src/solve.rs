//! A 3D motion solve: the model, its approximations' ledger, the time step from a step-doubling
//! check, then trajectories in sequential batches until the standard errors meet the targets at
//! every output time (as QuTiP's mcsolve does with its target tolerance), each batch's result handed
//! out as a checkpoint with the wall time the remaining trajectories will take.
//!
//! # Targets
//! The semiclassical method is accurate to about `TEMPERATURE_TARGET` in T and
//! `POPULATION_TARGET` in the populations in its regime; the trajectories are run until two
//! standard errors of every population and the loss lie within those targets at every output
//! time, and of every temperature (each axis, relative) at the output times in the regime --
//! outside it the method's own error is larger and the temperature is a reference only.
//!
//! # Ledger
//! Every a-priori approximation's estimate enters the solve's ledger (ledger.rs), against the
//! budget, the a-priori share of the population target. The model's are fixed before the run: the
//! far components' and the couplings' folds (ground.rs), the quartic van Vleck shifts left out, the
//! Gaussian beams taken at their peak and the tables (model.rs). Two are measured along the
//! trajectories: the elimination's η², each output's mean excited population (the local expansion
//! parameter, ⟨Y⟩ of each atom's state), and the beams' profiles at the atoms. The first order in
//! the Doppler shift takes what the budget leaves after the model's and the pilot's elimination
//! (the exact response beyond it, engine.rs) -- or, where the elimination alone exceeds the budget,
//! the budget on its own, as no refinement can bring that output within it. The output times whose
//! total exceeds the budget are flagged (`Ledger::flagged`).
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
//!   waiting time), within η ε of the population target ε -- the internal state's lag carries the
//!   friction, so this also keeps the temperature's step error near η of its target;
//! - the step's light impulse, relative, within η of the temperature target.
//! Both scale as dt², so a step that misses is shrunk by the square root of its excess and checked
//! again; one that meets them with room is grown the same way. The step's exponentials are summed
//! to a share η of its own population budget, η² ε dt/waiting, instead of to double precision.
//!
//! # Batches
//! A batch is cut into two groups per thread (lib.rs `Stratum`): each group's initial states are a
//! Latin hypercube, so its mean of the initial energies, and of what they leave at early times,
//! varies far less than independent draws'; the groups are independent, so the jackknife over them
//! measures the error. T, ⟨v⟩, n̄ and ⟨r²⟩ are each atom's means over one trap period of the axis
//! around each output (lib.rs `simulate`): its kinetic energy sloshes into potential energy and back
//! twice a period, so over a period the spread of T between atoms is their energies' alone.
//!
//! # Regime
//! A point particle has no zero-point motion: below a few vibrational quanta per axis the method
//! cools too far. Against the 1D GMW references of the tweezer PGC (the atom on the line through
//! the focus): heating (TW1, k_B T from 6 to 16 ħω over 10 ms) T within 6 % (mean +2 %) and the
//! populations within 0.03; cooling (TW2) T within 4.5 % down to k_B T ≈ 5 ħω, 6 % low at 4.6 ħω,
//! 7-10 % low at 4 ħω and 10-55 % low below 3.5 ħω, the populations within 0.045 but for the first
//! 50 µs of optical pumping (up to 0.066). Against the 2D quantum trajectories of the same scene
//! (x and y free) within 8 % at 4.6 ħω and 15 % low at 3.1 ħω. Each output time says whether it
//! is in the regime k_B T ≥ `CLASSICAL` ħω on every axis (with T the engine's own, which lies below
//! the quantum T where the method cools too far, so the mark comes early rather than late).
use crate::engine::{Engine,Settings,Trajectory};
use crate::ground::Kind;
use crate::ledger::{Ledger,BUDGET};
use crate::model::{Model,Trap};
use crate::stats::{summarize,Estimate,Scales,KT_PER_UK};
use crate::{sample_states,simulate,Initial,Outcome};
use crate::{Complex,Request,STATES};
use atom_population_core::lattice::accuracy::A_PRIORI_SHARE;
use std::time::Instant;

/// The method's accuracy targets: T relative, populations and loss absolute.
pub const TEMPERATURE_TARGET:f64=0.10;
pub const POPULATION_TARGET:f64=0.05;
/// Statistics are run until this many standard errors lie within the targets.
pub const CONFIDENCE:f64=2.;
/// The validated regime: k_B T ≥ CLASSICAL ħω along every axis (T the engine's own), where the 1D
/// and 2D quantum references put T within about 10 % (validation, see the module documentation).
pub const CLASSICAL:f64=4.;
/// The most trajectories a solve runs; a solve stopped there returns its result with the reason.
pub const MOST_TRAJECTORIES:u64=1<<14;
/// A batch's group: a batch is cut into two groups per thread (jackknife groups and Latin
/// hypercubes of initial states, lib.rs `Stratum`), so a batch's groups are as large as its size
/// allows while the jackknife keeps at least that many.
fn group(count:u64,threads:u64)->u64{(count/(2*threads)).max(1)}

pub const NOT_CONFINED:&str="光镊不约束原子：3D 运动计算光镊中的单个原子，需要在焦点处形成势阱的光镊";

/// Distributions along one axis on one grid, at some of the times.
#[derive(Clone,Debug,PartialEq)]
pub struct Distributions{pub grid:Vec<f64>,pub samples:Vec<usize>,pub probabilities:Vec<Vec<f64>>}

/// A 3D motion solve's result, at every time of the request (per lab axis).
/// Everything but `loss_probability` describes the atom still held.
#[derive(Clone,Debug)]
pub struct Solution{
    pub populations:Vec<Vec<f64>>,
    /// Per time, the largest standard error of the populations and the loss.
    pub sample_errors:Vec<f64>,
    pub temperature_uk:Vec<[f64;3]>,
    pub temperature_errors_uk:Vec<[f64;3]>,
    /// T_total = (T_x + T_y + T_z)/3 and its standard error (stats.rs `Estimate`).
    pub temperature_total_uk:Vec<f64>,
    pub temperature_total_errors_uk:Vec<f64>,
    pub mean_velocity_mps:Vec<[f64;3]>,
    /// ω_a/2π (kHz) at the focus of the reference state's trap per lab axis, and n̄ per axis relative to it.
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
    /// The a-priori approximations and their estimated errors (ledger.rs).
    pub approximations:Ledger,
}

/// An estimate in the validated regime: k_B T ≥ CLASSICAL ħω along every axis (ω per axis).
fn classical(e:&Estimate,frequencies:[f64;3])->bool{(0..3).all(|a|KT_PER_UK*e.temperature[a]>=CLASSICAL*frequencies[a])}

/// The largest ratio of CONFIDENCE standard errors to the targets over every output after t = 0
/// (the request's initial state, `start`): the populations and the loss everywhere, the
/// temperatures where the method holds.
fn worst_ratio(estimates:&[Estimate],frequencies:[f64;3])->f64{
    estimates.iter().skip(1).map(|e|{
        // The temperature to its target where the method holds; outside, a reference only.
        let t=if classical(e,frequencies){(0..3).map(|a|if e.temperature[a]>0.{e.temperature_error[a]/(TEMPERATURE_TARGET*e.temperature[a])}else{0.}).fold(0.,f64::max)}else{0.};
        let p=e.population_errors.iter().fold(e.loss_error,|m,&x|m.max(x))/POPULATION_TARGET;
        CONFIDENCE*t.max(p)
    }).fold(0.,f64::max)
}

/// The distribution sample indices: evenly spread over the outputs, the first and last among them.
fn listed(outputs:usize)->Vec<usize>{
    let count=outputs.min(21);let mut v:Vec<usize>=(0..count).map(|i|((i*(outputs-1)) as f64/(count-1).max(1) as f64).round() as usize).collect();v.dedup();v
}

/// Histograms of a value of the samples (a coordinate along one axis) at the listed times on one
/// grid symmetric about 0 that holds all of them (the bins' centres; an odd count, 0 at the centre).
fn histograms(samples:&[Vec<([f64;3],[f64;3])>],listed:&[usize],value:&dyn Fn(&([f64;3],[f64;3]))->f64)->Distributions{
    let reach=samples.iter().flatten().map(|s|value(s).abs()).fold(0.,f64::max).max(f64::MIN_POSITIVE);
    let count=samples.iter().map(|s|s.len()).max().unwrap_or(0);
    // Bins: about the square root of the samples, odd, at least 21.
    let bins=(((count as f64).sqrt() as usize)|1).max(21);let width=2.*reach/(bins-1) as f64;
    let grid:Vec<f64>=(0..bins).map(|i|(i as f64-(bins/2) as f64)*width).collect();
    let probabilities=samples.iter().map(|s|{
        let mut p=vec![0.;bins];if s.is_empty(){p[bins/2]=1.;return p;}
        for x in s{let i=((value(x)/width).round() as isize+(bins/2) as isize).clamp(0,bins as isize-1) as usize;p[i]+=1./s.len() as f64;}p
    }).collect();
    Distributions{grid,samples:listed.to_vec(),probabilities}
}

/// A time in the words of the app (μs below a millisecond, ms above).
fn when(us:f64)->String{if us<1e3{format!("{us:.0} μs")}else{format!("{:.2} ms",us/1e3)}}

/// The model's approximations (fixed before the run): the far components' fold, the couplings'
/// fold, the quartic van Vleck shifts left out, the Gaussian beams' profiles and the tables.
fn model_ledger(model:&Model,request:&Request)->Ledger{
    let g=&model.ground;let mut ledger=Ledger::new();let(far,near)=g.fold_estimates();
    let folded=g.components.iter().filter(|c|c.folded&&c.size>0.).count();
    ledger.add("远失谐分量（光镊、反旋项）",far,format!("二阶折叠（van Vleck 能移），{folded} 个分量"));
    let(kept,fold)=g.items.iter().filter(|i|!matches!(i.kind,Kind::Photons{..})).fold((0,0),|(k,f),i|if i.folded{(k,f+1)}else{(k+1,f)});
    let sizes:Vec<String>=g.clusters.iter().map(|c|c.states.len().to_string()).collect();
    ledger.add("基态之间的耦合（塞曼子能级、拉曼项、两组光的拍频）",near,format!("{fold} 对二阶折叠，{kept} 对显式保留；子空间 {}",sizes.join("+")));
    if model.quartic>0.{ledger.add("经多束光的四阶能移",model.quartic,"略去（一阶折叠）".into());}
    for &(l,error,_)in &model.profiles{ledger.add(&format!("「{}」的光强分布",request.beams[l].name),error,"按峰值强度的平面波".into());}
    for &l in &model.focused[1..]{ledger.add(&format!("「{}」的光强分布",request.beams[l].name),0.,"按位置插值（强度维度）".into());}
    ledger.add("聚焦光的光强插值表",model.table_error,"三次插值".into());
    ledger
}

/// The ledger of a solve's estimates so far: the model's, the run's largest first-order Doppler
/// term and, per output, the elimination's η² and the profiles at the atoms; the outputs whose total
/// exceeds the budget flagged.
fn ledger(model:&Model,base:&Ledger,estimates:&[Estimate],times:&[f64],allowance:f64)->Ledger{
    let mut ledger=base.clone();let g=&model.ground;
    let excited:Vec<f64>=estimates.iter().map(|e|g.excited_states.iter().map(|&i|e.populations[i]).sum::<f64>()).collect();
    let profiles:Vec<f64>=estimates.iter().map(|e|e.profile).collect();let dopplers:Vec<f64>=estimates.iter().map(|e|e.doppler).collect();
    let largest=|v:&[f64]|v.iter().skip(1).copied().fold(0.,f64::max);
    ledger.add("消去激发态",largest(&excited),"零阶绝热消去（激发态布居 η²）".into());
    ledger.add("多普勒与反冲频移",largest(&dopplers),format!("一阶（kv/σ）²；单步超过 {:.1e} 时用精确响应",allowance));
    if !model.profiles.is_empty(){ledger.add("聚焦光在原子处的强度（沿轨迹）",largest(&profiles),"按峰值强度的平面波".into());}
    // The outputs beyond the budget.
    let fixed=base.used()-model.profiles.iter().map(|p|p.1).sum::<f64>();
    let profile=|i:usize|model.profiles.iter().map(|p|p.1).sum::<f64>().max(profiles[i]);
    let over:Vec<usize>=(1..estimates.len()).filter(|&i|fixed+profile(i)+dopplers[i]+excited[i]>BUDGET).collect();
    if !over.is_empty(){
        let(first,last)=(over[0],*over.last().unwrap());
        let span=if over.len()==last-first+1&&last==estimates.len()-1{format!("{} 起",when(times[first]))}else{format!("{} 至 {} 之间 {} 个时刻",when(times[first]),when(times[last]),over.len())};
        let worst=over.iter().copied().max_by(|&a,&b|excited[a].total_cmp(&excited[b])).unwrap();
        ledger.flagged=Some(format!("{span}超出近似误差预算 {:.2}%：消去激发态的误差（激发态布居）最大 {:.2}%，合计 {:.2}%",
            100.*BUDGET,100.*excited[worst],100.*(fixed+profile(worst)+dopplers[worst]+excited[worst])));
    }
    ledger
}

/// Solves a request. `checkpoint` receives every batch's complete result and the estimated wall
/// time (ms) the rest will take.
pub fn solve(request:&Request,checkpoint:&mut dyn FnMut(&Solution,Option<f64>))->Result<Solution,String>{
    let model=Model::new(request)?;
    let initial=Initial::new(&model,request)?;
    // The reference state: the largest initial population; its trap sets ω per axis.
    let reference=(0..initial.states.len()).max_by(|&a,&b|initial.states[a].2.total_cmp(&initial.states[b].2)).unwrap();
    let trap:Trap=initial.traps[reference];
    if trap.frequencies.iter().any(|&w|!(w>0.)){return Err(NOT_CONFINED.into());}
    let times=&request.times_us;let outputs=times.len();
    if outputs<2||times[0]!=0.{return Err("A 3D motion solve needs at least two times, the first at 0".into());}
    let interval=times[1]-times[0];
    if times.windows(2).any(|w|((w[1]-w[0])-interval).abs()>1e-9*interval){return Err("A 3D motion solve needs evenly spaced times".into());}
    let listed=listed(outputs);let scales=Scales{hbar_over_mass:model.hbar_over_mass,frequencies:trap.lab};
    let seed=0x3D_5EED;
    let base=model_ledger(&model,request);
    // The first step: the light pattern's phase k_max v_T per step at sqrt(η × the temperature
    // target), or without light the trap's own ω.
    let thermal=(KT_PER_UK*request.temperature_uk*model.hbar_over_mass).sqrt();
    let pattern=model.carriers.iter().map(|c|c.wave_number).fold(0.,f64::max)*thermal;
    let rate=if pattern>0.{pattern}else{trap.frequencies.iter().copied().fold(0.,f64::max)};
    let mut steps=((interval*rate/(A_PRIORI_SHARE*TEMPERATURE_TARGET).sqrt()).ceil() as usize).max(1);
    let threads=rayon::current_num_threads() as u64;let pilot=2*threads;
    let duration=times[outputs-1];
    let plain=|steps:usize,truncation:f64|Settings::new(interval/steps as f64,&trap,truncation,true);
    let(states,jumps)=sample_states(&model,&plain(steps,f64::EPSILON),&initial,seed,pilot,steps,outputs);
    let waiting=pilot as f64*duration/(jumps.max(1) as f64);
    // The first order in the Doppler shift: what the budget leaves after the model and the pilot's
    // elimination, or the budget itself where the elimination alone exceeds it.
    let eta=states.iter().map(|t|{let p=model.point(t.r,t.kappa);let q=model.populations(t.cluster,&p,&t.psi,t.time);
        model.ground.excited_states.iter().map(|&i|q[i]).sum::<f64>()}).sum::<f64>()/states.len().max(1) as f64;
    let left=BUDGET-base.used()-eta;let allowance=if left>0.{left}else{BUDGET};
    // The exponentials are summed to a share η of the step's own population budget, η² ε dt/waiting.
    let settings=|steps:usize|{let dt=interval/steps as f64;let mut s=plain(steps,A_PRIORI_SHARE*A_PRIORI_SHARE*POPULATION_TARGET*dt/waiting);s.doppler=allowance;s};
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
    // Batches until the targets are met.
    let mut outcome=first;let mut done=pilot;let mut per_trajectory=wall/pilot as f64;
    loop{
        let mut estimates=summarize(&outcome.groups,&scales);start(&mut estimates[0],&model,&initial);
        let ratio=worst_ratio(&estimates,trap.lab);
        let needed=((done as f64*ratio*ratio).ceil() as u64).min(MOST_TRAJECTORIES);
        let ledger=ledger(&model,&base,&estimates,times,allowance);
        let mut solution=assemble(&model,&estimates,&outcome,&listed,trap.lab,done,dt,&request.initial_density,ledger);
        if needed<=done{
            if ratio>1.{solution.reason=Some(format!("统计误差为目标的 {ratio:.1} 倍：轨迹数已达上限 {MOST_TRAJECTORIES}"));}
            return Ok(solution);
        }
        checkpoint(&solution,Some(per_trajectory*(needed-done) as f64*1e3));
        let next=(needed-done).clamp(2*threads,3*done);
        let clock=Instant::now();
        let more=simulate(&model,&settings,&initial,seed,done,next,group(next,threads),steps,outputs,&listed);
        per_trajectory=clock.elapsed().as_secs_f64()/next as f64;
        outcome.groups.extend(more.groups);for(a,b)in outcome.samples.iter_mut().zip(more.samples){a.extend(b);}outcome.jumps+=more.jumps;outcome.doppler=outcome.doppler.max(more.doppler);
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
    (accumulated/(A_PRIORI_SHARE*POPULATION_TARGET)).max(relative/(A_PRIORI_SHARE*TEMPERATURE_TARGET))
}

/// t = 0 is the request's initial state, exactly: per lab axis T is its thermal Wigner temperature
/// T* = (ħω/2k_B) coth(ħω/2k_B T) in each initial state's trap (lib.rs `initial_trajectory`), the
/// mean velocity zero -- the trajectories' estimates there carry only the sampling's error.
fn start(e:&mut Estimate,model:&Model,initial:&Initial){
    let kt=KT_PER_UK*initial.temperature_uk;let mut t=[0.;3];let _=model;
    for(&(_,_,p),trap)in initial.states.iter().zip(&initial.traps){
        // Per lab axis: the principal axes' thermal Wigner temperatures, weighted by their share of it.
        for a in 0..3{for i in 0..3{let w=trap.frequencies[i];let half=w/(2.*kt);
            t[a]+=p*trap.axes[i][a]*trap.axes[i][a]*if half>0.{initial.temperature_uk*half/half.tanh()}else{initial.temperature_uk};}}
    }
    e.temperature=t;e.temperature_error=[0.;3];e.mean_velocity=[0.;3];
    e.temperature_total=t.iter().sum::<f64>()/3.;e.temperature_total_error=0.;
}

/// The solution of the estimates so far.
/// At t = 0 the populations are the initial state's, as the request gives them (the light acts
/// after), and so are T and the mean velocity (`start`).
fn assemble(model:&Model,estimates:&[Estimate],outcome:&Outcome,listed:&[usize],frequencies:[f64;3],trajectories:u64,dt:f64,initial:&[Complex],ledger:Ledger)->Solution{
    let hm=model.hbar_over_mass;
    let velocity=std::array::from_fn(|a|histograms(&outcome.samples,listed,&|s|s.1[a]*hm));
    let position=std::array::from_fn(|a|histograms(&outcome.samples,listed,&|s|s.0[a]));
    let first:Vec<f64>=(0..STATES).map(|i|initial[i*STATES+i].re).collect();
    let sizes:Vec<String>=model.ground.clusters.iter().map(|c|c.states.len().to_string()).collect();
    Solution{
        populations:estimates.iter().enumerate().map(|(i,e)|if i==0{first.clone()}else{e.populations.clone()}).collect(),
        sample_errors:estimates.iter().enumerate().map(|(i,e)|if i==0{0.}else{e.population_errors.iter().fold(e.loss_error,|m,&x|m.max(x))}).collect(),
        temperature_uk:estimates.iter().map(|e|e.temperature).collect(),
        temperature_errors_uk:estimates.iter().map(|e|e.temperature_error).collect(),
        temperature_total_uk:estimates.iter().map(|e|e.temperature_total).collect(),
        temperature_total_errors_uk:estimates.iter().map(|e|e.temperature_total_error).collect(),
        mean_velocity_mps:estimates.iter().map(|e|e.mean_velocity).collect(),
        trap_frequency_khz:frequencies.map(|w|w/std::f64::consts::TAU*1e3),
        mean_occupation:estimates.iter().map(|e|e.occupation).collect(),
        loss_probability:{let mut worst=0.;estimates.iter().map(|e|{worst=e.loss.max(worst);worst}).collect()},
        velocity,position,
        classical:estimates.iter().map(|e|classical(e,frequencies)).collect(),
        trajectories,time_step_us:dt,
        method:format!("半经典 3D：经典质点（位置、动量）携带消去激发态后的基态内态（子空间 {}），量子跳跃（自发辐射按偶极辐射方向图反冲，吸收按光束反冲并保持精确的平均力），机械能高于势阱深度即算损失，{trajectories} 条轨迹，步长 {dt:.3} μs",sizes.join("+")),
        reason:None,
        approximations:ledger,
    }
}
