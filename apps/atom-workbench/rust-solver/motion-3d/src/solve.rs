//! A 3D motion solve: the model, its approximations' ledger, the time step from a step-doubling
//! check, then trajectories in sequential batches until the standard errors meet the targets at
//! every output time (as QuTiP's mcsolve does with its target tolerance), each batch's result handed
//! out as a checkpoint with the wall time the remaining trajectories will take.
//!
//! # Targets
//! The semiclassical method is accurate to about `TEMPERATURE_TARGET` in T and
//! `POPULATION_TARGET` in the populations in its regime; the trajectories are run until each
//! output's total estimated error (below) lies within those targets: every population's and the
//! loss's at every output time, and every temperature's (each axis, relative) at the output times
//! in the regime -- outside it the method's own error is larger and the temperature is a reference
//! only.
//!
//! # Ledger and the total error
//! Every a-priori approximation's estimate enters the solve's ledger (ledger.rs), reported against
//! its budget, the a-priori share of the population target. The decisions take that share: the
//! couplings' and the far components' folds (ground.rs), a Gaussian beam's intensity axis (model.rs),
//! the region the initial positions are drawn in (lib.rs `thermal_position`) and, per step, the
//! first order in the Doppler shift -- what the budget leaves after the model's estimates and the
//! pilot's elimination, or the budget on its own where the elimination alone exceeds it (the exact
//! response beyond, engine.rs). The elimination's η² -- each output's mean
//! excited population, the local expansion parameter ⟨Y⟩ of each atom's state -- is measured along
//! the trajectories and is no decision yet: it enters the total (validated on resonant probes against
//! the population core: optical pumping's population error 0.4–1.1 η², a cycling transition's 2η⁴).
//!
//! An output's total estimated error is the ledger's total there (its η², its mean (kv/σ)² and the
//! profiles at the atoms with the model's estimates), the time step's bias (its step-doubling share)
//! and CONFIDENCE standard errors: the populations' against the population target (absolute), T's
//! against the temperature target (relative, where the method holds) -- the fold's charge in either
//! its own (`fold_charges`: at fourth order the populations' S², T's still S, the force folding at
//! second order). The trajectories are run until
//! the totals meet the targets -- the statistics within what the ledger and the step leave -- and an
//! output whose total exceeds a target all the same, its a-priori share beyond it, is flagged
//! (`Ledger::flagged`; a checkpoint's statistics still run, so only its a-priori excess is).
//!
//! # Time step
//! The trap's harmonic part is moved exactly (engine.rs); what the step must resolve is the light's
//! pattern crossed by the moving atom and the internal state's response to it, which carries the
//! light's friction. The step is chosen by step doubling on the deterministic flow (as an adaptive
//! ODE solver's local error, QuTiP's way): states are sampled along a pilot of a few trajectories
//! at a first step (the phase k_max v_T of the light's pattern at the initial temperature advancing
//! sqrt(η) per step, η the a-priori share), and from each one step of dt is compared with two of
//! dt/2, without jumps. A step meets the targets when (4/3 of) the differences meet their shares:
//! - the internal state's populations after one waiting time between jumps (the pilot's mean),
//!   within η ε of the population target ε. Bounded by one step's difference times the steps of
//!   that time; where the motion's criteria below are met, measured as well: from each of a
//!   subsample of the states (one per pilot trajectory) the jump-free flow over that time (at most
//!   the run) in steps of dt against steps of dt/2, the largest difference of the cluster's
//!   populations, their mean over the subsample plus CONFIDENCE standard errors -- in a gray
//!   molasses the atoms wait 50-140 us in dark states, and the bound over-counts the step's error
//!   there several times;
//! - the step's light impulse, relative, within η of the temperature target;
//! - the light's friction, relative, within η of the temperature target: the sampled states' mean
//!   change of the mechanical energy over the step, which the conservative forces leave alone --
//!   the internal state's lag that carries the friction is a small share of its populations, so
//!   their criterion alone lets T's step error grow past its share (the 3D PGC example: T 2–3 % high
//!   at the populations' step, measured against half that step). Where the friction moves the
//!   energy by less than its thermal value 3 k_B T over the run, T's step error is the energy's step
//!   error summed over the run against that thermal energy, and the step's energy error is taken
//!   against a step's share of it: the change there can be the integrator's own error, which falls
//!   with the step as the step's error does (`step_excess`).
//! All scale as dt², so a step that misses is shrunk by the square root of its excess and checked
//! again, and one that meets them with room is grown the same way, by whatever factor, until a step
//! comes up twice: first by the population criterion's bound, then from the largest step that met
//! the targets by it with the population error measured. Every checked step that met the targets
//! is a candidate; the one kept costs least in prediction (`predicted_cost`): its work per interval
//! -- counted on the sampled states (`interval_work`): its steps' advances, each the generator's and
//! the force's multiply-adds and table pairs of the state's cluster and two half steps' series, and
//! per jump its photons and the two advances it redoes, each unit at a cost measured once (no timing
//! enters a solve: a request gives the same step and the same result on any machine and load) --,
//! times the trajectories the statistics need. Their spread is known only after a batch, but whatever it is, the trajectories a candidate
//! needs relative to the bound rule's step are at most the larger over the targets of (that step's
//! room/the candidate's room)², the room what the target leaves past the a-priori estimates and the
//! step's bias: exact for the target that binds where the candidate's bias is the larger. The bound
//! rule's step is left only for a candidate predicted to cost less. A larger step is kept only where
//! it pays: its bias takes room from the statistics and the work per
//! step grows with it (the exponentials' sub-steps, the jumps' redone steps) -- in the 3D PGC and the
//! tetrahedral MOT the candidates' costs lie within a few per cent, in the D1 gray molasses the
//! measured criterion halves it. The step's exponentials are summed to a share η of its own
//! population budget, η² ε dt/waiting, instead of to double precision. A trajectory takes at most
//! `MOST_STEPS` steps over the run: neither the first step nor the search goes finer, and a step that
//! bound sets is kept even where it misses its shares, its measured bias booked as any step's (the
//! flag, where a total exceeds its target, says why).
//!
//! # Batches
//! Every batch is cut into groups of `TASK` trajectories (lib.rs `Stratum`), a task on the CPU and a
//! block on the GPU: each group's initial states are a Latin hypercube, so its mean of the initial
//! energies, and of what they leave at early times, varies far less than independent draws'; the
//! groups are independent, so the jackknife over them measures the error. One size for every group
//! gives every trajectory the same share of the variance, so the need that a batch's errors predict
//! holds for the trajectories still to come; groups of 8 keep most of the hypercube's gain (the
//! trajectories T needs at early times fall by 10-40% from groups of 1 to groups of 8, little more
//! beyond) while the jackknife's groups grow with the trajectories. T, ⟨v⟩, n̄ and ⟨r²⟩ are each
//! atom's means over one trap period of the axis around each output (lib.rs `simulate`): its kinetic
//! energy sloshes into potential energy and back twice a period, so over a period the spread of T
//! between atoms is their energies' alone.
//!
//! The first standard errors stand on `FIRST_GROUPS` groups on any machine: the pilot's trajectories,
//! each a group of one (with the harmonic control, at the pilot's step), predict the need, and the
//! first batch is that many groups of `TASK`, at most the need the pilot predicts, at least the
//! device's width -- a machine of more threads runs it in fewer rounds, not with more trajectories.
//! Each later batch is the predicted remaining need -- the trajectories the errors so far say the
//! targets need, less those run --, at least the device's width and at most as many as have run: the
//! need is the largest of many outputs' estimated errors, biased up while few groups stand behind it,
//! so it is followed at most to twice the trajectories it rests on. No margin is added above the
//! prediction: a batch more costs the device's tail and the statistics, a few trajectories' time,
//! where a margin of the estimate's own noise would cost tens of percent of the trajectories.
//!
//! # Control variates
//! Each output's κ² and populations are estimated with control variates of exactly known means
//! (control.rs, stats.rs `estimate`): the harmonic twin's for T, always (it costs a few products per
//! output), and the held twin's for the populations. β is the regression on the control over every
//! trajectory so far, held when the jackknife leaves a group out: its errors -- what the stopping rule
//! takes -- are the corrected values' spread over the groups; where a control follows nothing, β ≈ 0.
//! The held
//! twin costs a trajectory of its own: it runs in the first batch, and is kept when the trajectories
//! that batch says the targets need with it -- the populations' or T's, whichever needs more --,
//! times its share of the work (counted as the step's, `held_work`), are fewer than without: where T
//! decides the trajectories its control of the populations saves none. The first batch's groups are
//! Latin hypercubes, so its T need is the run's, not inflated by independent draws' spread at early
//! times.
//!
//! # Regime
//! A point particle has no zero-point motion: below a few vibrational quanta per axis the method
//! cools too far. Against the 1D GMW references of the tweezer PGC (the atom on the line through
//! the focus; measured with v2026.11.2.6's engine, whose 3D PGC example this one reproduces within
//! its standard errors): heating (TW1, k_B T from 6 to 16 ħω over 10 ms) T within 6 % (mean +2 %) and the
//! populations within 0.03; cooling (TW2) T within 4.5 % down to k_B T ≈ 5 ħω, 6 % low at 4.6 ħω,
//! 7-10 % low at 4 ħω and 10-55 % low below 3.5 ħω, the populations within 0.045 but for the first
//! 50 µs of optical pumping (up to 0.066). Against the 2D quantum trajectories of the same scene
//! (x and y free) within 8 % at 4.6 ħω and 15 % low at 3.1 ħω. Each output time says whether it
//! is in the regime k_B T ≥ `CLASSICAL` ħω on every axis (with T the engine's own, which lies below
//! the quantum T where the method cools too far, so the mark comes early rather than late).
//!
//! # Device
//! The batches run on the device asked for (`Device`): the CPU's threads, or the GPU (gpu/mod.rs),
//! which runs a batch's trajectories as the CPU does -- the same groups, initial states, random
//! streams and sums. The two devices make a solve's choices alike: the model, the ledger, the pilot
//! and the time step on the CPU in double precision either way, the statistics and their stopping
//! rule. A batch is whole groups: on the CPU at least one (its threads take the trajectories one at
//! a time, so a batch costs its trajectories and one tail whatever its size), and on the GPU whole
//! waves -- a group in each of its resident blocks --, a wave costing about what one trajectory does,
//! so that there the first batch is usually the last. A GPU that cannot run (none, an older driver, a
//! model beyond its kernel) leaves the solve on the CPU with the reason
//! (`Solution::device_note`); one that fails during the batches has the solve redone on the CPU.
use crate::engine::{column_norm,node_phase,series,Engine,Settings,Trajectory};
use crate::gpu;
use crate::ground::{ExcitedKind,Ground,Kind};
use crate::ledger::{Ledger,BUDGET};
use crate::model::{Model,Trap};
use crate::control::{Harmonic,Twin};
use crate::stats::{summarize,Estimate,Means,Scales,KT_PER_UK};
use crate::{sample_states,simulate,Initial,Outcome,Variates,TASK};
use crate::{Complex,Request,STATES};
use atom_population_core::lattice::accuracy::A_PRIORI_SHARE;
use rayon::prelude::*;
use std::time::Instant;

/// Where a solve's trajectories run (the request's `options.device`).
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum Device{Cpu,Gpu}

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
/// The most steps a trajectory takes over the run, a bound on the work as `MOST_TRAJECTORIES` is (a
/// solve takes at most their product): neither the first step nor the step search goes finer, and a
/// step it bounds that misses its shares is kept with its measured bias, the reason in the flag where
/// a total exceeds its target (`ledger`).
pub const MOST_STEPS:usize=1<<24;
/// The jackknife's groups behind a solve's first standard errors, on any machine (# Batches): the
/// pilot's trajectories, each a group of one, and at most the first batch's groups of `TASK` -- the
/// twin's decision and the first prediction stand on them (formerly two per thread of 16 threads).
pub const FIRST_GROUPS:u64=32;

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
    /// The trajectories the statistics need by the latest batch's estimate: `trajectories` is the
    /// first whole batch at or beyond it (a batch at most triples the trajectories run).
    pub needed:u64,
    pub time_step_us:f64,
    pub method:String,
    pub reason:Option<String>,
    /// The a-priori approximations and their estimated errors (ledger.rs).
    pub approximations:Ledger,
    /// The device the trajectories ran on, and why it is not the one asked for (Chinese).
    pub device:Device,
    pub device_note:Option<String>,
}

/// An estimate in the validated regime: k_B T ≥ CLASSICAL ħω along every axis (ω per axis).
fn classical(e:&Estimate,frequencies:[f64;3])->bool{(0..3).all(|a|KT_PER_UK*e.temperature[a]>=CLASSICAL*frequencies[a])}

/// An output's statistical errors at CONFIDENCE: the populations' and the loss's largest (absolute)
/// and, where the method holds, T's largest relative one over the axes (None outside: a reference).
fn statistics(e:&Estimate,frequencies:[f64;3])->(f64,Option<f64>){
    let p=CONFIDENCE*e.population_errors.iter().fold(e.loss_error,|m,&x|m.max(x));
    let t=classical(e,frequencies).then(||CONFIDENCE*(0..3).map(|a|if e.temperature[a]>0.{e.temperature_error[a]/e.temperature[a]}else{0.}).fold(0.,f64::max));
    (p,t)
}
/// What the a-priori estimates (`Accounts::apriori`) and the step's bias leave of a target: the room
/// the statistics may take -- the whole target where nothing is left (that output is flagged).
fn room(target:f64,taken:f64)->f64{if taken<target{target-taken}else{target}}
/// The largest ratio of the statistical errors to their room over every output after t = 0 (the
/// request's initial state, `start`): 1 meets the targets.
fn worst_ratio(estimates:&[Estimate],accounts:&Accounts,frequencies:[f64;3])->f64{let[p,t]=worst_ratios(estimates,accounts,frequencies);p.max(t)}
/// The same per kind: the populations' and the loss's, and T's.
fn worst_ratios(estimates:&[Estimate],accounts:&Accounts,frequencies:[f64;3])->[f64;2]{
    estimates.iter().enumerate().skip(1).map(|(i,e)|{
        let(p,t)=statistics(e,frequencies);let a=accounts.apriori[i];
        [p/room(POPULATION_TARGET,a[0]+accounts.bias[0]),t.map_or(0.,|t|t/room(TEMPERATURE_TARGET,a[1]+accounts.bias[1]))]
    }).fold([0.;2],|w,r|[w[0].max(r[0]),w[1].max(r[1])])
}

/// The distribution sample indices: evenly spread over the outputs, the first and last among them.
fn listed(outputs:usize)->Vec<usize>{
    let count=outputs.min(21);let mut v:Vec<usize>=(0..count).map(|i|((i*(outputs-1)) as f64/(count-1).max(1) as f64).round() as usize).collect();v.dedup();v
}

/// Histograms of a value of the samples (a coordinate along one axis) at the listed times on one
/// grid symmetric about 0 that holds all of them (the bins' centres; an odd count, 0 at the centre).
/// A listed time where no atom is held carries the one before (`assemble`); the first is t = 0,
/// where every atom is.
fn histograms(samples:&[Vec<([f64;3],[f64;3])>],listed:&[usize],value:&dyn Fn(&([f64;3],[f64;3]))->f64)->Distributions{
    let reach=samples.iter().flatten().map(|s|value(s).abs()).fold(0.,f64::max).max(f64::MIN_POSITIVE);
    let count=samples.iter().map(|s|s.len()).max().unwrap_or(0);
    // Bins: about the square root of the samples, odd, at least 21.
    let bins=(((count as f64).sqrt() as usize)|1).max(21);let width=2.*reach/(bins-1) as f64;
    let grid:Vec<f64>=(0..bins).map(|i|(i as f64-(bins/2) as f64)*width).collect();
    let mut last:Option<Vec<f64>>=None;
    let probabilities=samples.iter().map(|s|{
        if s.is_empty(){return last.clone().unwrap_or_else(||{let mut p=vec![0.;bins];p[bins/2]=1.;p});}
        let mut p=vec![0.;bins];
        for x in s{let i=((value(x)/width).round() as isize+(bins/2) as isize).clamp(0,bins as isize-1) as usize;p[i]+=1./s.len() as f64;}
        last=Some(p.clone());p
    }).collect();
    Distributions{grid,samples:listed.to_vec(),probabilities}
}

/// A time in the words of the app (μs below a millisecond, ms above).
fn when(us:f64)->String{if us<1e3{format!("{us:.0} μs")}else{format!("{:.2} ms",us/1e3)}}

/// The model's approximations (fixed before the run): the far components' fold, the couplings'
/// fold, the quartic van Vleck shifts left out, the Gaussian beams' profiles and the tables.
fn model_ledger(model:&Model,request:&Request)->Ledger{
    let g=&model.ground;let mut ledger=Ledger::new();let(far,near,statics)=g.fold_estimates();
    let folded=g.components.iter().filter(|c|c.folded&&c.size>0.).count();
    ledger.add("远失谐分量（光镊、反旋项）",far,format!("二阶折叠（van Vleck 能移），{folded} 个分量"));
    // The drives' fold, its charge; H's couplings between excited states.
    let explicit:usize=g.clusters.iter().map(|c|c.excited.len()).sum();
    let count=|drive:bool|g.excited_items.iter().filter(|x|matches!(x.kind,ExcitedKind::Drive(_))==drive).fold((0,0),|(k,f),x|if x.folded{(k,f+1)}else{(k+1,f)});
    let(kept,fold)=count(true);
    let order=if model.fourth{"四阶折叠（误差界为饱和参数之和的平方）"}else{"二阶折叠（误差界为饱和参数之和）"};
    ledger.add(FOLD,fold_charge(model,g.charge),if explicit==0{format!("{fold} 个驱动全部{order}，饱和参数按光场图样平均")}
        else{format!("{fold} 个驱动{order}，{kept} 个显式保留、{explicit} 个激发态精确传播：按预计总耗时选取")});
    let(kept,fold)=count(false);
    if kept+fold>0{ledger.add("激发态之间的耦合（光镊混合超精细能级）",statics,format!("{fold} 个二阶折叠，{kept} 个显式保留"));}
    let(kept,fold)=g.items.iter().filter(|i|matches!(i.kind,Kind::Static{..}|Kind::Raman{..})).fold((0,0),|(k,f),i|if i.folded{(k,f+1)}else{(k+1,f)});
    let sizes:Vec<String>=g.clusters.iter().map(|c|c.dimension().to_string()).collect();
    ledger.add("基态之间的耦合（塞曼子能级、拉曼项、两组光的拍频）",near,format!("{fold} 对二阶折叠，{kept} 对显式保留；子空间 {}",sizes.join("+")));
    if model.quartic>0.{ledger.add("经多束光的四阶能移",model.quartic,"略去（一阶折叠）".into());}
    if model.fourth_dropped>0.{ledger.add("四阶折叠的快拍频项",model.fourth_dropped,"略去（与二阶相同的截断频率）".into());}
    for &(l,error,_)in &model.profiles{ledger.add(&format!("「{}」的光强分布",request.beams[l].name),error,"按峰值强度的平面波".into());}
    for &l in &model.focused[1..]{ledger.add(&format!("「{}」的光强分布",request.beams[l].name),0.,"按位置插值（强度维度）".into());}
    ledger.add("聚焦光的光强插值表",model.table_error,"三次插值".into());
    ledger
}

/// A solve's error accounts so far: per output and target (the populations, T) the a-priori estimates'
/// total (the model's, and there the fold's charge as measured, the mean (kv/σ)² and the profiles at
/// the atoms) and the fold's charge in it (`fold_charges`), and the time step's bias on the populations
/// and on T.
struct Accounts{apriori:Vec<[f64;2]>,excited:Vec<[f64;2]>,bias:[f64;2]}

/// The ledger of a solve's estimates so far, and its accounts: the model's items, the elimination's
/// largest η², the largest mean (kv/σ)² and the profiles at the atoms. With `last`, the outputs whose
/// total estimated error -- the a-priori total, the step's bias and the statistics -- exceeds a
/// target are flagged; before it, those whose a-priori total and bias alone do.
fn ledger(model:&Model,base:&Ledger,estimates:&[Estimate],times:&[f64],allowance:f64,bias:[f64;2],frequencies:[f64;3],last:bool)->(Ledger,Accounts){
    let mut ledger=base.clone();let charge=fold_charge(model,model.ground.charge);
    // The fold's error, charged as measured along the trajectories: the folded excited population's
    // 2η², the saturation the atoms met -- what the fold's bound takes on the paths they go (an atom
    // that gathers where the light is bright, as red-detuned molasses holds it at its lattice's
    // bright sites, meets more than the pattern's mean). The a-priori estimate over the reach decided
    // the fold (`Candidate`).
    let measured:Vec<[f64;2]>=estimates.iter().map(|e|fold_charges(model.fourth,2.*e.folded)).collect();
    let profiles:Vec<f64>=estimates.iter().map(|e|e.profile).collect();let dopplers:Vec<f64>=estimates.iter().map(|e|e.doppler).collect();
    let largest=|v:&[f64]|v.iter().skip(1).copied().fold(0.,f64::max);
    ledger.add("多普勒与反冲频移",largest(&dopplers),format!("一阶（kv/σ）²；单步超过 {:.1e} 时用精确响应",allowance));
    if !model.profiles.is_empty(){ledger.add("聚焦光在原子处的强度（沿轨迹）",largest(&profiles),"按峰值强度的平面波".into());}
    let met=largest(&measured.iter().map(|m|m[0]).collect::<Vec<f64>>());
    if let Some(item)=ledger.items.iter_mut().find(|i|i.label==FOLD){
        item.action=format!("{}；计入沿轨迹实测的饱和参数 2η²{}（先验估计 {:.2}%{}）{}",item.action,if model.fourth{" 的平方"}else{""},100.*charge,if met>charge{"，实测高于先验估计"}else{""},
            if model.fourth{"；温度仍计 2η²（力按二阶折叠）"}else{""});item.estimate=met;
    }
    let peak:f64=model.profiles.iter().map(|p|p.1).sum();let fixed=base.used()-peak-charge-cut(base);
    let apriori:Vec<[f64;2]>=(0..estimates.len()).map(|i|if i==0{[0.;2]}else{measured[i].map(|m|fixed+peak.max(profiles[i])+dopplers[i]+m)}).collect();
    let accounts=Accounts{apriori,excited:measured,bias};
    // The outputs beyond a target: (output, the quantity, its total, the a-priori share, the statistics).
    let over:Vec<(usize,&str,f64,f64,f64)>=(1..estimates.len()).filter_map(|i|{
        let(p,t)=statistics(&estimates[i],frequencies);let a=accounts.apriori[i];let(p,t)=if last{(p,t)}else{(0.,t.map(|_|0.))};
        let population=(a[0]+bias[0]+p>POPULATION_TARGET).then_some((i,"布居",a[0]+bias[0]+p,a[0]+bias[0],p));
        let temperature=t.and_then(|t|(a[1]+bias[1]+t>TEMPERATURE_TARGET).then_some((i,"温度",a[1]+bias[1]+t,a[1]+bias[1],t)));
        population.or(temperature)}).collect();
    // The elimination's a-priori bound over the atoms' reach (model.rs `Model::resonance`): beyond the
    // population target, light near resonance somewhere the atoms go -- the measured η² can miss it.
    let resonance=(model.resonance>POPULATION_TARGET).then(||format!("光在原子所到之处接近共振：消去激发态的先验界（基态的激发态权重）达 {:.0}%（目标 {:.0}%），近共振处消去激发态不成立",
        100.*model.resonance,100.*POPULATION_TARGET));
    // The initial ensemble's cut beyond the budget, on its own: the loss and the populations depend on it.
    let share=cut(base);
    let initial=(share>BUDGET).then(||format!("初始热分布有 {:.0}% 落在势阱区之外（丢失与布居依赖这一截断）",100.*share));
    ledger.flagged=match(resonance,initial){(Some(r),Some(i))=>Some(format!("{r}；{i}")),(r,i)=>r.or(i)};
    if !over.is_empty(){
        let(first,last_over)=(over[0].0,over[over.len()-1].0);
        let span=if over.len()==last_over-first+1&&last_over==estimates.len()-1{format!("{} 起",when(times[first]))}
            else{format!("{} 至 {} 之间 {} 个时刻",when(times[first]),when(times[last_over]),over.len())};
        let worst=over.iter().max_by(|a,b|(a.2/if a.1=="布居"{POPULATION_TARGET}else{TEMPERATURE_TARGET}).total_cmp(&(b.2/if b.1=="布居"{POPULATION_TARGET}else{TEMPERATURE_TARGET}))).unwrap();
        let target=if worst.1=="布居"{POPULATION_TARGET}else{TEMPERATURE_TARGET};
        let over=format!("{span}估计误差超出目标：{}误差最大 {:.1}%（目标 {:.0}%；近似与步长 {:.1}%，其中消去激发态 {:.1}%；统计 {:.1}%）",
            worst.1,100.*worst.2,100.*target,100.*worst.3,100.*accounts.excited[worst.0][if worst.1=="布居"{0}else{1}],100.*worst.4);
        ledger.flagged=Some(match ledger.flagged.take(){Some(r)=>format!("{r}；{over}"),None=>over});
    }
    (ledger,accounts)
}

/// Solves a request on the device asked for (module documentation, # Device). `checkpoint` receives
/// every batch's complete result and the estimated wall time (ms) the rest will take.
pub fn solve(request:&Request,device:Device,checkpoint:&mut dyn FnMut(&Solution,Option<f64>))->Result<Solution,String>{
    // The GPU starts in the background while the model is made.
    if device==Device::Gpu{gpu::prepare();}
    match run(request,device,checkpoint){
        Err(Stop::Gpu(error))=>{
            let mut solution=run(request,Device::Cpu,checkpoint).map_err(Stop::message)?;
            solution.device_note=Some(format!("GPU 计算出错（{error}）：本次在 CPU 上计算"));Ok(solution)
        }
        other=>other.map_err(Stop::message),
    }
}

/// Why a solve stopped: its own error, or its GPU's (the solve is then redone on the CPU).
enum Stop{Error(String),Gpu(String)}
impl From<String> for Stop{fn from(error:String)->Stop{Stop::Error(error)}}
impl Stop{fn message(self)->String{match self{Stop::Error(e)|Stop::Gpu(e)=>e}}}

fn run(request:&Request,device:Device,checkpoint:&mut dyn FnMut(&Solution,Option<f64>))->Result<Solution,Stop>{
    let times=&request.times_us;let outputs=times.len();
    if outputs<2||times[0]!=0.{return Err(Stop::Error("A 3D motion solve needs at least two times, the first at 0".into()));}
    let interval=times[1]-times[0];
    if times.windows(2).any(|w|((w[1]-w[0])-interval).abs()>1e-9*interval){return Err(Stop::Error("A 3D motion solve needs evenly spaced times".into()));}
    let listed=listed(outputs);let seed=0x3D_5EED;
    let pilot=FIRST_GROUPS;
    let duration=times[outputs-1];
    // The drift that moves an atom's thermal energy 3 k_B T (a 3D harmonic trap's) over the run, per
    // unit time: a step's share of it is the friction criterion's floor (`step_excess`).
    let drift=3.*KT_PER_UK*request.temperature_uk/duration;
    // The fold of the excited states (module documentation): the rule's candidates from all folded
    // down, their order the a-priori estimates'; the pilot runs the first, and each candidate's
    // charge is what its folded drives meet at the pilot's states (`measured_charge`). The first
    // whose charge leaves the statistics room is the reference, and a later one replaces it where its
    // work times the trajectories its room needs costs less in prediction.
    let mut ladder=Ground::allowances(request)?.into_iter();let mut failed=None;
    // The fourth order is a candidate on the CPU (the GPU's kernel folds at second order).
    let fourth=device==Device::Cpu;
    let Some(mut pilot_candidate)=next_candidate(request,&mut ladder,false,&mut failed)else{return Err(Stop::Error(failed.unwrap_or_else(||"No model of the request".into())))};
    // The reference state: the largest initial population; its trap sets ω per axis.
    let trap=pilot_candidate.trap();
    if trap.frequencies.iter().any(|&w|!(w>0.)){return Err(Stop::Error(NOT_CONFINED.into()));}
    // The first step: the light pattern's phase k_max v_T per step at sqrt(η × the temperature
    // target), or without light the trap's own ω.
    let thermal=(KT_PER_UK*request.temperature_uk*pilot_candidate.model.hbar_over_mass).sqrt();
    let pattern=pilot_candidate.model.carriers.iter().map(|c|c.wave_number).fold(0.,f64::max)*thermal;
    let rate=if pattern>0.{pattern}else{trap.frequencies.iter().copied().fold(0.,f64::max)};
    // The finest step the work bound allows (`MOST_STEPS`), in steps per interval.
    let most=(MOST_STEPS/(outputs-1)).max(1);
    let steps=((interval*rate/(A_PRIORI_SHARE*TEMPERATURE_TARGET).sqrt()).ceil() as usize).clamp(1,most);
    let plain=|steps:usize,truncation:f64|Settings::new(interval/steps as f64,&trap,truncation,true);
    // The pilots' exponentials to double precision; the beats their exact steps leave out within the share
    // the run's take at most, the step's population budget over the whole run (engine.rs `Segment::resolve`).
    let piloting=|steps:usize|{let mut s=plain(steps,f64::EPSILON);s.beats=A_PRIORI_SHARE*A_PRIORI_SHARE*POPULATION_TARGET*s.dt/duration;s.horizon=duration;s};
    let pilot_harmonic=Harmonic::new(&pilot_candidate.model,&piloting(steps),&pilot_candidate.initial,steps,outputs);
    let(pilot_states,pilot_jumps,pilot_groups)=sample_states(&pilot_candidate.model,&piloting(steps),&pilot_candidate.initial,seed,pilot,steps,outputs,&pilot_harmonic);
    let charge=measured_charge(&pilot_candidate.model,&pilot_candidate.model,&pilot_states);pilot_candidate.charged(charge);
    // The targets the fold's choice weighs (`Binding`): the populations' always, T's where the pilot's atoms
    // reach the method's regime -- elsewhere T is a reference only, and its charge must not choose.
    let binds=[true,Binding::of(&pilot_states,pilot_candidate.model.hbar_over_mass,trap.lab).temperature];
    // The pilot's statistics per target (its trajectories each a group of one, with the harmonic control):
    // CONFIDENCE standard errors, the populations' and the loss's largest and T's at the outputs in the
    // regime (`statistics`). The trajectories a model needs over the binding targets are (statistic/room)²
    // times the pilot's -- the target with the larger need is what decides its trajectories.
    let stats:[f64;2]={
        let scales=Scales{hbar_over_mass:pilot_candidate.model.hbar_over_mass,frequencies:trap.lab};
        let means:Vec<Means>=(0..outputs).map(|i|Means{harmonic:pilot_harmonic.mean[i],twin:None}).collect();
        let mut e=summarize(&pilot_groups,&scales,&means);start(&mut e[0],&pilot_candidate.model,&pilot_candidate.initial);
        e.iter().skip(1).map(|x|statistics(x,trap.lab)).fold([0.;2],|m,(p,t)|[m[0].max(p),m[1].max(t.unwrap_or(0.))])
    };
    let need_at=|rooms:&[f64;2]|(0..2).filter(|&k|binds[k]&&stats[k]>0.).map(|k|if rooms[k]>0.{(stats[k]/rooms[k]).powi(2)}else{f64::INFINITY}).fold(0.,f64::max);
    // The trajectories `rooms` need relative to `reference`'s; without statistics (none spread), the larger
    // ratio of the binding targets' rooms.
    let need=|reference:&[f64;2],rooms:&[f64;2]|{let at=need_at(reference);
        if at>0.{need_at(rooms)/at}else{(0..2).filter(|&k|binds[k]).map(|k|if rooms[k]>0.{(reference[k]/rooms[k]).powi(2)}else{f64::INFINITY}).fold(0.,f64::max)}};
    let pilot_waiting=pilot as f64*duration/(pilot_jumps.max(1) as f64);
    let mut reference:Option<Candidate>=None;
    // The ladder is walked for a fit only where the fold's charge is what keeps the pilot's model from
    // one: estimates the fold does not touch (an initial ensemble beyond its trap, say) leave no
    // candidate room, and every rung would only cost its model.
    if !pilot_candidate.fits(binds)&&pilot_candidate.fits_unfolded(binds){
        while let Some(mut c)=next_candidate(request,&mut ladder,false,&mut failed){
            let charge=measured_charge(&pilot_candidate.model,&c.model,&pilot_states);c.charged(charge);let fits=c.fits(binds);reference=Some(c);if fits{break;}
        }
    }
    let better={
        let base=reference.as_ref().unwrap_or(&pilot_candidate);let mut better=None;
        // A pilot whose atoms were all lost before its first output measures no candidate's work.
        if base.fits(binds)&&!pilot_states.is_empty(){
            let rooms=base.rooms;let mut best=1.;let mut work=1_f64;
            // The fewest trajectories any later candidate can need relative to the reference: its rooms
            // at no a-priori estimate at all.
            let fewest=need(&rooms,&TARGETS.map(|target|target-A_PRIORI_SHARE*target));
            // The reference's own folds at fourth order, their charges the same saturation's at that order
            // and their work timed as any candidate's -- built only where they can pay: their rooms at the
            // reference's other estimates leave fewer trajectories than it needs, its step being the second
            // order's and more.
            let charge=fold_charges(true,base.saturation);let others=base.others();
            let promise=need(&rooms,&std::array::from_fn(|k|TARGETS[k]-others-charge[k]-A_PRIORI_SHARE*TARGETS[k]));
            if fourth&&promise<best&&base.model.ground.clusters.iter().any(|c|c.excited.is_empty()){
                if let Ok(mut c)=Candidate::new(request,base.allowance,true){
                    c.charged(base.saturation);
                    if c.fits(binds){
                        let trajectories=need(&rooms,&c.rooms);
                        let(settings,limit)=(piloting(steps),best/trajectories);
                        let w=if !(counted_work(&pilot_candidate.model,&base.model,&c.model,&settings,&pilot_states,interval,pilot_waiting)<=limit){f64::INFINITY}
                            else{model_work(&pilot_candidate.model,&base.model,&c.model,&settings,&pilot_states,pilot as usize,interval,limit)};
                        if w*trajectories<best{best=w*trajectories;better=Some(c);}
                    }
                }
            }
            // Every later candidate's work is at least the last one timed (it keeps more states explicit):
            // each feasible one is timed while even the fewest trajectories could pay for it.
            while work*fewest<best{
                let Some(mut c)=next_candidate(request,&mut ladder,false,&mut failed)else{break};
                let charge=measured_charge(&pilot_candidate.model,&c.model,&pilot_states);c.charged(charge);
                if !c.fits(binds){continue;}
                let trajectories=need(&rooms,&c.rooms);
                let(settings,limit)=(piloting(steps),best/fewest);
                work=work.max(if !(counted_work(&pilot_candidate.model,&base.model,&c.model,&settings,&pilot_states,interval,pilot_waiting)<=limit){f64::INFINITY}
                    else{model_work(&pilot_candidate.model,&base.model,&c.model,&settings,&pilot_states,pilot as usize,interval,limit)});
                if work*trajectories<best{best=work*trajectories;better=Some(c);}
            }
        }
        better
    };
    // The pilot again where the choice is not its model (its trajectories' sums the first prediction's).
    let(chosen,(states,jumps,pilot_groups),pilot_harmonic)=match better.or(reference){
        Some(c)=>{let h=Harmonic::new(&c.model,&piloting(steps),&c.initial,steps,outputs);
            let pilot_run=sample_states(&c.model,&piloting(steps),&c.initial,seed,pilot,steps,outputs,&h);(c,pilot_run,h)}
        None=>(pilot_candidate,(pilot_states,pilot_jumps,pilot_groups),pilot_harmonic),
    };
    let Candidate{model,initial,base,charge,..}=chosen;
    // The generator against its jumps at the pilot's states (`gain`): booked where its decay falls below
    // the jumps' rate.
    let mut base=base;let gain=gain(&model,&states);
    if gain>0.{base.add(GAIN,gain,"沿试探轨迹测得：无跳跃的衰减低于跳跃率（i(H − H†) − ΣL†L 的负本征值对最大跳跃率之比）".into());}
    // The beats the exact steps left out (engine.rs `Segment::resolve`), booked as their bound per unit time
    // over the horizon: within the exponentials' share by construction.
    let pilot_beats=states.iter().map(|t|t.beats).fold(0.,f64::max);
    let scales=Scales{hbar_over_mass:model.hbar_over_mass,frequencies:trap.lab};
    let waiting=pilot as f64*duration/(jumps.max(1) as f64);let horizon=waiting.min(duration);
    // The first order in the Doppler shift: what the budget leaves after the model's other estimates
    // (the fold's charge takes its own share of the target), or the budget itself where they exceed it.
    let left=BUDGET-(base.used()-cut(&base)-fold_charge(&model,model.ground.charge));let allowance=if left>0.{left}else{BUDGET};
    // The exponentials are summed to a share η of the step's own population budget, η² ε dt/waiting.
    let settings=|steps:usize|{let dt=interval/steps as f64;let mut s=plain(steps,A_PRIORI_SHARE*A_PRIORI_SHARE*POPULATION_TARGET*dt/waiting);s.doppler=allowance;s.horizon=waiting;s};
    // The step (module documentation): by the bound's rule first -- from the first step, the next by
    // the square root of the excess (both errors scale as dt²) until it suggests a step already
    // checked --, then, where the population criterion decides that rule's step, from it with the
    // population error measured where it decides and the motion's criteria are met (a step's
    // doubling taken once). Kept: of the checked steps that met the targets the one of least
    // predicted cost, or the last checked when none did.
    let doubled=|steps:usize|{let r=step_excess(&Engine::new(&model,settings(steps)),&states,waiting,drift);Check{steps,population:r[0],motion:r[1]}};
    let search=|first:usize,measured:bool,known:&[Check],checked:&mut Vec<Check>|{
        let mut steps=first;
        for _ in 0..8{
            if checked.iter().any(|c|c.steps==steps){break;}
            let mut c=known.iter().find(|c|c.steps==steps).copied().unwrap_or_else(||doubled(steps));
            if measured&&c.decided_by_population(){c.population=accumulated_excess(&Engine::new(&model,settings(steps)),&states,horizon,pilot as usize);}
            checked.push(c);
            steps=((steps as f64*c.excess().sqrt()).ceil() as usize).clamp(1,most);
        }
    };
    let mut bound=vec![];search(steps,false,&[],&mut bound);
    let grown=*bound.iter().filter(|c|c.met()).min_by_key(|c|c.steps).unwrap_or(&bound[bound.len()-1]);
    let mut measured=vec![];if grown.decided_by_population(){search(grown.steps,true,&bound,&mut measured);}
    let candidates:Vec<Check>=bound.iter().chain(&measured).filter(|c|c.met()).copied().collect();
    // The a-priori total the steps' costs take: the ledger's estimates with the fold's charge as the
    // pilot measured it.
    let apriori=charge.map(|c|base.used()-cut(&base)-fold_charge(&model,model.ground.charge)+c+pilot_beats*horizon);let parts=state_work(&model,&states);
    let chosen=if candidates.len()>1{
        // Against the bound rule's step, or where it missed the targets the smallest candidate step.
        let baseline=*candidates.iter().find(|c|c.steps==grown.steps).or(candidates.iter().max_by_key(|c|c.steps)).unwrap();
        let mut distinct:Vec<usize>=candidates.iter().map(|c|c.steps).filter(|&s|s!=baseline.steps).collect();distinct.sort();distinct.dedup();distinct.insert(0,baseline.steps);
        let work:Vec<f64>=distinct.iter().map(|&s|interval_work(&parts,&settings(s),interval,waiting)).collect();
        let work_of=|c:&Check|work[distinct.iter().position(|&s|s==c.steps).unwrap()]/work[0];
        let cost=|c:&Check|predicted_cost(c,work_of(c),&baseline,apriori);
        // The baseline is left only for a candidate predicted to cost less (ties keep it).
        candidates.iter().fold(baseline,|best,c|if cost(c)<cost(&best){*c}else{best})
    }else{candidates.first().copied().unwrap_or(grown)};
    let(steps,ratios)=(chosen.steps,chosen.ratios());
    let dt=interval/steps as f64;let settings=settings(steps);
    // A step the work bound set that misses its shares: why, for the flag.
    let bounded=(steps==most&&!chosen.met()).then(||format!("步数已达上限（每条轨迹 {MOST_STEPS} 步）：步长 {dt:.3e} μs 的误差为其份额的 {:.1} 倍",chosen.excess()));
    // The step's bias: its share of each target times its measured excess there.
    let bias=[ratios[0]*A_PRIORI_SHARE*POPULATION_TARGET,ratios[1]*A_PRIORI_SHARE*TEMPERATURE_TARGET];
    // The device: the GPU when asked for and able, the model on it; else the CPU, with why.
    let(gpu,note)=match device{
        Device::Cpu=>(None,None),
        Device::Gpu=>match gpu::gpu_within(gpu::START_WAIT).and_then(|g|{let program=g.lock().unwrap_or_else(|e|e.into_inner()).program(&model,&settings)?;Ok((g,program))}){
            Ok(on)=>(Some(on),None),
            Err(reason)=>(None,Some(format!("{reason}：本次在 CPU 上计算"))),
        },
    };
    let used=if gpu.is_some(){Device::Gpu}else{Device::Cpu};
    // The control variates (control.rs): the harmonic twin's, and the held twin where the trap beam's
    // intensity is the pilot's mean, when the generator is static.
    let harmonic=Harmonic::new(&model,&settings,&initial,steps,outputs);
    let intensity=states.iter().map(|t|model.geometry[model.trap_beam].intensity(t.r)).sum::<f64>()/states.len().max(1) as f64;
    let mut twin=Twin::new(&model,intensity);
    let twin_means=twin.as_ref().map(|t|t.means(&model,&request.initial_density,interval,outputs));
    let means=|held:bool|->Vec<Means>{(0..outputs).map(|i|Means{harmonic:harmonic.mean[i],twin:twin_means.as_ref().filter(|_|held).map(|m|m[i])}).collect()};
    let batch=|first:u64,count:u64,twin:Option<&Twin>|->Result<Outcome,Stop>{
        let variates=Variates{harmonic:&harmonic,twin};
        match &gpu{
            None=>Ok(simulate(&model,&settings,&initial,seed,first,count,TASK,steps,outputs,&listed,Some(&variates))),
            Some((g,program))=>g.lock().unwrap_or_else(|e|e.into_inner()).simulate(program,&model,&settings,&initial,seed,first,count,TASK,steps,outputs,&listed,Some(&variates))
                .map_err(Stop::Gpu),
        }
    };
    // A batch is whole groups: on the CPU at least one, on the GPU a group in each of its resident
    // blocks -- such a batch costs about what one trajectory does --, a whole number of its widths.
    let width=match &gpu{Some((g,_))=>g.lock().unwrap_or_else(|e|e.into_inner()).width(1),None=>TASK};
    let unit=if gpu.is_some(){width}else{TASK};
    // The first batch: FIRST_GROUPS groups, at most the pilot's predicted need (its trajectories each
    // a group, with the harmonic control), at least the device's width.
    let pilot_need={
        let means:Vec<Means>=(0..outputs).map(|i|Means{harmonic:pilot_harmonic.mean[i],twin:None}).collect();
        let mut e=summarize(&pilot_groups,&scales,&means);start(&mut e[0],&model,&initial);
        let(_,accounts)=ledger(&model,&booked(&base,pilot_beats*horizon),&e,times,allowance,bias,trap.lab,false);
        (pilot as f64*worst_ratio(&e,&accounts,trap.lab).powi(2)).min(MOST_TRAJECTORIES as f64).ceil() as u64
    };
    let clock=Instant::now();
    let count=(FIRST_GROUPS*TASK).min(pilot_need).max(width).div_ceil(unit)*unit;
    let first=batch(0,count,twin.as_ref())?;
    let wall=clock.elapsed().as_secs_f64();
    // The held twin is kept where its control pays for itself: the trajectories the first batch says
    // the targets need with it, times its share of the work (counted as the step's, `held_work`),
    // against those they need without.
    if twin.is_some(){
        let needed=|means:&[Means]|{let mut e=summarize(&first.groups,&scales,means);start(&mut e[0],&model,&initial);
            let(_,accounts)=ledger(&model,&booked(&base,pilot_beats.max(first.beats)*horizon),&e,times,allowance,bias,trap.lab,false);worst_ratio(&e,&accounts,trap.lab).powi(2)};
        let share=held_work(&parts,&settings,interval,waiting)/interval_work(&parts,&settings,interval,waiting);
        if needed(&means(true))*(1.+share)>=needed(&means(false)){twin=None;}
    }
    let means=means(twin.is_some());
    // Batches until the targets are met.
    let mut outcome=first;let mut done=count;let mut per_trajectory=wall/count as f64;
    loop{
        let mut estimates=summarize(&outcome.groups,&scales,&means);start(&mut estimates[0],&model,&initial);
        let(_,accounts)=ledger(&model,&booked(&base,pilot_beats.max(outcome.beats)*horizon),&estimates,times,allowance,bias,trap.lab,false);
        let ratio=worst_ratio(&estimates,&accounts,trap.lab);
        let wanted=(done as f64*ratio*ratio).ceil() as u64;let needed=wanted.min(MOST_TRAJECTORIES);
        let(mut ledger,_)=ledger(&model,&booked(&base,pilot_beats.max(outcome.beats)*horizon),&estimates,times,allowance,bias,trap.lab,needed<=done);
        if let(Some(note),Some(flag))=(&bounded,&ledger.flagged){ledger.flagged=Some(format!("{note}；{flag}"));}
        let mut solution=assemble(&model,&estimates,&outcome,&listed,times,trap.lab,done,dt,&request.initial_density,ledger,twin.is_some());
        solution.device=used;solution.device_note=note.clone();solution.needed=wanted;
        if needed<=done{
            if ratio>1.{solution.reason=Some(format!("统计误差为目标的 {ratio:.1} 倍：轨迹数已达上限 {MOST_TRAJECTORIES}"));}
            return Ok(solution);
        }
        checkpoint(&solution,Some(per_trajectory*(needed-done) as f64*1e3));
        let next=next_batch(needed,done,width,unit);
        let clock=Instant::now();
        let more=batch(done,next,twin.as_ref())?;
        per_trajectory=clock.elapsed().as_secs_f64()/next as f64;
        outcome.groups.extend(more.groups);for(a,b)in outcome.samples.iter_mut().zip(more.samples){a.extend(b);}outcome.jumps+=more.jumps;outcome.doppler=outcome.doppler.max(more.doppler);outcome.beats=outcome.beats.max(more.beats);
        done+=next;
    }
}

/// The model's ledger with the beats the exact steps left out, `bound` their largest over the horizon.
fn booked(base:&Ledger,bound:f64)->Ledger{
    let mut ledger=base.clone();
    if bound>0.{ledger.add("多速率步略去的残余拍频",bound,"整步影响之界（Duhamel 积分的分部积分）在指数截断份额内的拍频对略去，其余按 CF4 精确传播".into());}
    ledger
}

/// The ledger's item of a generator whose decay falls below its jumps' rate (`gain`).
const GAIN:&str="生成元的增益";
/// A model's gain at sampled states (model.rs `Model::decay_balance`, at most FIRST_GROUPS × 8 of them): the
/// largest −λ of D = i(H − H†) − Σ L†L beyond its eigenvalues' round-off, relative to the largest jump rate
/// met -- the relative error of the jump rate where the no-jump decay misses it (a van Vleck path's
/// dissipative part that no jump carries, the validation's F10); 0 for a Lindblad generator, unbounded
/// for a gain without jumps.
fn gain(model:&Model,states:&[Trajectory])->f64{
    let every=(states.len()/(FIRST_GROUPS as usize*8)).max(1);
    let(gain,rate)=states.par_iter().step_by(every).map(|t|{let(low,rate,scale)=model.decay_balance(t.cluster,&model.point(t.r,t.kappa),t.time);
        let n=model.dimension(t.cluster) as f64;((-low-64.*f64::EPSILON*n*scale).max(0.),rate)}).reduce(||(0.,0.),|a,b|(a.0.max(b.0),a.1.max(b.1)));
    if gain==0.{0.}else if rate>0.{gain/rate}else{f64::INFINITY}
}

/// The ledger's item of the drives' fold (its charge a priori; the solve's ledger charges it as measured).
const FOLD:&str="消去激发态";
/// The ledger's item of the initial ensemble's cut (lib.rs `thermal_position`): the share of the
/// traps' thermal positions beyond their region. The temperature asked for against the trap sets it;
/// no choice of the solve changes it, so it stays out of the totals the solve's choices take (the
/// fold's room, the Doppler allowance, the statistics' room) and out of the population headline of
/// the flag: beyond the budget it is flagged on its own (`ledger`).
const CUT:&str="初始热分布（势阱区之外）";
/// The cut's share in a ledger (`CUT`), zero without one.
fn cut(ledger:&Ledger)->f64{ledger.items.iter().find(|i|i.label==CUT).map_or(0.,|i|i.estimate)}
/// The fold's charge for the saturation S its folded drives meet: its populations' error bound, S at
/// second order and S² at fourth (ground.rs; against the population core 0.2-1.0 S² where the series
/// converges, S ≲ 0.5 -- beyond, the charge exceeds the target and the rule keeps the drives).
fn fold_charge(model:&Model,saturation:f64)->f64{fold_charges(model.fourth,saturation)[0]}
/// The fold's charges per target, the populations' (`fold_charge`) and T's, for the saturation S its
/// folded drives meet at second or fourth order: T's is S at either -- the force folds at second order
/// (against explicit states, the moving 3D PGC's T is 1.1 S off on its axis above 4ħω that differs,
/// its populations within 0.5 S).
fn fold_charges(fourth:bool,saturation:f64)->[f64;2]{[if fourth{saturation*saturation}else{saturation},saturation]}
/// The targets in the order of the rooms: the populations' (absolute) and T's (relative).
const TARGETS:[f64;2]=[POPULATION_TARGET,TEMPERATURE_TARGET];

/// A candidate of the fold (module documentation): the model at one of the rule's allowances, its
/// initial state, its ledger and the rooms its estimates leave the statistics -- each target less the
/// ledger's total, the fold's charge as measured (`Candidate::charged`), and the step's bias at most
/// (its share).
struct Candidate{model:Model,initial:Initial,base:Ledger,allowance:f64,saturation:f64,charge:[f64;2],rooms:[f64;2]}
impl Candidate{
    fn new(request:&Request,allowance:f64,fourth:bool)->Result<Candidate,String>{
        let model=Model::with_order(request,allowance,fourth)?;let initial=Initial::new(&model,request)?;let mut base=model_ledger(&model,request);
        // The initial positions' region (lib.rs `thermal_position`): the share of the traps' thermal
        // positions it leaves out.
        if initial.outside>0.{base.add(CUT,initial.outside,"位置取势阱区（谐振近似低于阱深处）内的玻尔兹曼分布；丢失与布居依赖这一截断".into());}
        let(saturation,charge)=(model.ground.charge,fold_charges(model.fourth,model.ground.charge));let others=base.used()-cut(&base)-charge[0];
        let rooms=std::array::from_fn(|k|TARGETS[k]-others-charge[k]-A_PRIORI_SHARE*TARGETS[k]);
        Ok(Candidate{model,initial,base,allowance,saturation,charge,rooms})
    }
    /// Its rooms with the fold's charge for the saturation `saturation` met in place of the a-priori estimate.
    fn charged(&mut self,saturation:f64){let charge=fold_charges(self.model.fourth,saturation);self.saturation=saturation;self.charge=charge;
        let others=self.others();self.rooms=std::array::from_fn(|k|TARGETS[k]-others-charge[k]-A_PRIORI_SHARE*TARGETS[k]);}
    /// Its ledger's estimates but the fold's.
    fn others(&self)->f64{self.base.used()-cut(&self.base)-fold_charge(&self.model,self.model.ground.charge)}
    /// Its estimates leave the statistics room on the binding targets (`binds`: the populations', T's).
    fn fits(&self,binds:[bool;2])->bool{self.rooms.iter().zip(binds).all(|(&r,b)|!b||r>0.)}
    /// Its estimates but the fold's leave the statistics room on the binding targets: a fold could still make
    /// it fit.
    fn fits_unfolded(&self,binds:[bool;2])->bool{TARGETS.iter().zip(binds).all(|(&target,b)|!b||target-self.others()-A_PRIORI_SHARE*target>0.)}
    /// The trap of its reference state, the largest initial population: ω per axis.
    fn trap(&self)->Trap{let i=&self.initial;i.traps[(0..i.states.len()).max_by(|&a,&b|i.states[a].2.total_cmp(&i.states[b].2)).unwrap()]}
}
/// The next candidate along the rule's allowances that builds (a model that fails, its tables not
/// converging, is passed over with its error kept).
fn next_candidate(request:&Request,ladder:&mut dyn Iterator<Item=f64>,fourth:bool,failed:&mut Option<String>)->Option<Candidate>{
    for a in ladder{match Candidate::new(request,a,fourth){Ok(c)=>return Some(c),Err(e)=>{failed.get_or_insert(e);}}}
    None
}
/// The saturation a candidate's folded drives meet: at each of the pilot's states, carried into its
/// clusters (`carried`), 2η² (twice the folded excited population, whose fold's charge bounds its
/// error, `fold_charge`); per pilot trajectory their mean, and of those the mean plus
/// CONFIDENCE standard errors -- what the atoms meet rather than the light's pattern extrapolated
/// (an atom held in the dark states of a gray molasses meets little of its bright state's
/// saturation; one gathered at a red-detuned lattice's bright sites meets more than its mean).
fn measured_charge(from:&Model,to:&Model,states:&[Trajectory])->f64{
    let mut sums:Vec<(u64,f64,f64)>=vec![];
    for t in states{
        let x=carried(from,to,t);let p=to.point(x.r,x.kappa);let s=2.*to.readout(x.cluster,&p,&x.psi,x.time).1;
        match sums.iter_mut().find(|e|e.0==t.index){Some(e)=>{e.1+=s;e.2+=1.;}None=>sums.push((t.index,s,1.))}
    }
    let means:Vec<f64>=sums.iter().map(|e|e.1/e.2).collect();let n=means.len() as f64;if n==0.{return to.ground.charge;}
    let mean=means.iter().sum::<f64>()/n;let spread=(means.iter().map(|m|(m-mean).powi(2)).sum::<f64>()/(n*(n-1.).max(1.))).sqrt();
    mean+CONFIDENCE*spread
}

/// A candidate's work relative to the reference's, at its upper confidence bound: from each of
/// `subsample` of the pilot's states (of the model `from`), the same state carried into either's
/// clusters (`carried`), an interval of steps of either back to back on one thread. The work is the
/// ratio of the two totals -- a model's cost lies in the states it makes dear (a cluster with explicit
/// states), which a median of the states' ratios would pass over --, its upper bound CONFIDENCE
/// standard errors above it (the ratio's delta method over the states). The reference's states are
/// timed first; the candidate's stop as soon as their total passes `limit` times the reference's --
/// the most work worth knowing, the choice's -- and its work is then infinite: a candidate whose
/// explicit states take a thousand times the reference's steps is not run to the end to be refused.
fn model_work(from:&Model,reference:&Model,candidate:&Model,settings:&Settings,states:&[Trajectory],subsample:usize,interval:f64,limit:f64)->f64{
    use std::sync::atomic::{AtomicBool,AtomicU64,Ordering};
    let engines=[Engine::new(reference,settings.clone()),Engine::new(candidate,settings.clone())];
    let every=(states.len()/subsample.max(1)).max(1);let picked:Vec<&Trajectory>=states.iter().step_by(every).collect();
    let steps=(interval/settings.dt).round() as usize;
    let run=|e:&Engine,mut x:Trajectory,spent:Option<(&AtomicU64,u64,&AtomicBool)>|->f64{
        let clock=Instant::now();
        for _ in 0..steps{
            if let Some((total,most,stop))=spent{if stop.load(Ordering::Relaxed)||total.load(Ordering::Relaxed)+clock.elapsed().as_nanos() as u64>most{stop.store(true,Ordering::Relaxed);break;}}
            if x.alive{e.step(&mut x);}
        }
        let time=clock.elapsed();if let Some((total,_,_))=spent{total.fetch_add(time.as_nanos() as u64,Ordering::Relaxed);}time.as_secs_f64()
    };
    let base:Vec<f64>=picked.par_iter().map(|t|run(&engines[0],carried(from,reference,t),None)).collect();
    let b=base.iter().sum::<f64>().max(f64::MIN_POSITIVE);
    let(total,stop)=(AtomicU64::new(0),AtomicBool::new(false));let most=(limit*b*1e9).min(u64::MAX as f64) as u64;
    let work:Vec<f64>=picked.par_iter().map(|t|run(&engines[1],carried(from,candidate,t),Some((&total,most,&stop)))).collect();
    if stop.load(Ordering::Relaxed){return f64::INFINITY;}
    let n=base.len() as f64;let a=work.iter().sum::<f64>();
    let ratio=a/b;let spread=work.iter().zip(&base).map(|(x,y)|(x-ratio*y).powi(2)).sum::<f64>()/(n-1.).max(1.);
    ratio+CONFIDENCE*(spread/n).sqrt()/(b/n)
}

/// A candidate's counted work relative to the reference's (`interval_work`) on the pilot's states
/// carried into each: a candidate this count puts beyond the work worth knowing is not timed
/// (`model_work`) -- its explicit states' exact steps can take a thousand times the reference's.
fn counted_work(from:&Model,reference:&Model,candidate:&Model,settings:&Settings,states:&[Trajectory],interval:f64,waiting:f64)->f64{
    let into=|m:&Model|->Vec<Trajectory>{states.iter().map(|t|carried(from,m,t)).collect()};
    interval_work(&state_work(candidate,&into(candidate)),settings,interval,waiting)/interval_work(&state_work(reference,&into(reference)),settings,interval,waiting).max(f64::MIN_POSITIVE)
}

/// A trajectory's state carried from one model into another's clusters: into the cluster of its
/// most populated ground state, its amplitudes on that cluster's ground states and on the explicit
/// excited states the two clusters share.
fn carried(from:&Model,to:&Model,t:&Trajectory)->Trajectory{
    let(f,g)=(&from.ground.clusters[t.cluster],&to.ground);let ng=f.states.len();
    let most=(0..ng).max_by(|&a,&b|t.psi[a].norm_sqr().total_cmp(&t.psi[b].norm_sqr())).unwrap();
    let k=g.cluster[f.states[most]];let c=&g.clusters[k];
    let mut out=t.clone();out.cluster=k;out.psi=[Complex::ZERO;crate::model::MOST_MIXED];
    for(x,&b)in f.states.iter().enumerate(){if let Some(y)=c.position(b){out.psi[y]=t.psi[x];}}
    for(x,&e)in f.excited.iter().enumerate(){if let Some(y)=c.excited_position(e){out.psi[y]=t.psi[ng+x];}}
    out
}

/// The batch after `done` trajectories (module documentation, # Batches): the predicted remaining
/// need, at least the device's `width` and at most `done` -- the need followed at most to twice the
/// trajectories it rests on --, in whole `unit`s (groups on the CPU, waves on the GPU).
fn next_batch(needed:u64,done:u64,width:u64,unit:u64)->u64{needed.saturating_sub(done).clamp(width,done.max(width)).div_ceil(unit)*unit}

/// The targets that bind a solve: the populations and the loss at every output; T only at the outputs
/// in the method's regime (k_B T ≥ CLASSICAL ħω on every axis, as `classical`), outside which T is a
/// reference only (仅供参考). Here from a pilot's states at the outputs: T binds where some output's
/// states, their T per axis m⟨δv²⟩/k_B = ħ²⟨δκ²⟩/m, lie in the regime.
#[derive(Clone,Copy,Debug)]
pub struct Binding{pub temperature:bool}
impl Binding{
    pub fn of(states:&[Trajectory],hbar_over_mass:f64,frequencies:[f64;3])->Binding{
        let mut times:Vec<f64>=states.iter().map(|t|t.time).collect();times.sort_by(f64::total_cmp);times.dedup();
        let temperature=times.iter().any(|&time|{
            let at:Vec<&Trajectory>=states.iter().filter(|t|t.time==time).collect();let n=at.len() as f64;
            n>1.&&(0..3).all(|a|{let mean=at.iter().map(|t|t.kappa[a]).sum::<f64>()/n;
                let variance=at.iter().map(|t|(t.kappa[a]-mean).powi(2)).sum::<f64>()/(n-1.);hbar_over_mass*variance>=CLASSICAL*frequencies[a]})
        });
        Binding{temperature}
    }
}

/// A checked step (module documentation): its steps per interval and its ratios to the shares -- the
/// population criterion's (its bound's, or where measured the upper confidence bound of its measured
/// error) and the motion's (the impulse's and the friction's larger); 1 meets each.
#[derive(Clone,Copy,Debug)]
struct Check{steps:usize,population:f64,motion:f64}
impl Check{
    fn ratios(&self)->[f64;2]{[self.population,self.motion]}
    fn excess(&self)->f64{self.population.max(self.motion)}
    fn met(&self)->bool{self.excess()<=1.}
    /// The population criterion decides the step: its ratio the larger, the motion's criteria met.
    fn decided_by_population(&self)->bool{self.motion<=1.&&self.population>self.motion}
}

/// The step-doubling excess of a step (module documentation), from one step against two halves on all
/// the sampled states: the population criterion's bound -- one step's population error times the steps
/// of a waiting time -- and the light impulse's and the friction's larger relative error, each over
/// its share (1 meets it).
///
/// The friction's: the step's error of the mechanical energy's change, against that change where the
/// friction moves the energy by more than its thermal value 3 k_B T over the run -- the equilibrium T
/// it sets is off by the friction's relative error --, else against a step's share of that thermal
/// energy, `drift` dt (`drift` = 3 k_B T/duration): there T is off by the step's energy error summed
/// over the run, and the change is no measure of the step's error -- with the friction weak it is the
/// integrator's own error, whose third order makes one step's difference from two halves three times
/// their change (E₁ − E₂ = 3(E₂ − E₀)) at every step, so that their ratio never falls.
fn step_excess(engine:&Engine,states:&[Trajectory],waiting:f64,drift:f64)->[f64;2]{
    let dt=engine.settings.dt;let richardson=4./3.;let count=states.len().max(1) as f64;
    // The states' doublings on the threads, summed in their order (the sums do not depend on the threads).
    let doubled:Vec<([f64;2],[f64;3],[f64;2])>=states.par_iter().map(|t|engine.doubling(t,dt)).collect();
    let(mut population,mut difference,mut impulse,mut energy,mut friction)=(0.,0.,0.,0.,0.);
    for(motion,internal,work)in &doubled{difference+=motion[0];impulse+=motion[1];population+=internal[0];energy+=work[0];friction+=work[1];}
    let bound=richardson*population/count*(waiting/dt).max(1.);
    let relative=if impulse>0.{richardson*difference/impulse}else{0.};
    let dissipated=richardson*energy.abs()/friction.abs().max(count*drift*dt);
    [bound/(A_PRIORI_SHARE*POPULATION_TARGET),relative.max(dissipated)/(A_PRIORI_SHARE*TEMPERATURE_TARGET)]
}

/// The population error a step accumulates over `horizon` (module documentation), over its share:
/// from each of `subsample` states spread over the sampled ones the jump-free flow in steps of dt
/// against steps of dt/2, the largest difference of the cluster's populations; their mean plus
/// CONFIDENCE standard errors.
fn accumulated_excess(engine:&Engine,states:&[Trajectory],horizon:f64,subsample:usize)->f64{
    let dt=engine.settings.dt;let richardson=4./3.;let count=((horizon/dt).ceil() as usize).max(1);
    let every=(states.len()/subsample.max(1)).max(1);let picked:Vec<&Trajectory>=states.iter().step_by(every).collect();
    let errors:Vec<f64>=picked.par_iter().map(|t|{
        let n=engine.model.dimension(t.cluster);let(one,two)=(engine.coast(t,dt,count),engine.coast(t,0.5*dt,2*count));
        let(a,b)=(one.psi[..n].iter().map(|z|z.norm_sqr()).sum::<f64>(),two.psi[..n].iter().map(|z|z.norm_sqr()).sum::<f64>());
        richardson*(0..n).map(|i|(one.psi[i].norm_sqr()/a-two.psi[i].norm_sqr()/b).abs()).fold(0.,f64::max)
    }).collect();
    let m=errors.len().max(1) as f64;let mean=errors.iter().sum::<f64>()/m;
    let spread=(errors.iter().map(|e|(e-mean).powi(2)).sum::<f64>()/(m*(m-1.).max(1.))).sqrt();
    (mean+CONFIDENCE*spread)/(A_PRIORI_SHARE*POPULATION_TARGET)
}

/// What a unit of each of the step's parts costs (ns), for the work model (`interval_work`): measured
/// once on one performance core of the Core Ultra 7 265H (examples/bench_cost.rs) and fitted over
/// DEFAULT, THREE, PGC3D, TET, GMD1C and PGCP_10, each part within 10–20 % of its measured time; only
/// their ratios enter the choice. An advance without its propagation: per call (the point's fields,
/// the flow, the kick), per multiply-add of the generator and the force (`Model::work`) and per table
/// pair.
const ADVANCE_NS:f64=374.;const MULTIPLY_ADD_NS:f64=0.247;const PAIR_NS:f64=29.9;
/// A half step's propagation: per term of its series and per multiply-add (n² a term).
const TERM_NS:f64=4.4;const TERM_MULTIPLY_ADD_NS:f64=0.469;
/// A whole jump (`Engine::jump`): per call (its point, the class's and the polarization's draws, the
/// dipole pattern, the kicks), per multiply-add of every class's photons and per multiply-add of the
/// picked class's absorbed light (`Model::work`). Timed against the advance in the same interleaved
/// rounds on the loaded machine (each the least of 30 rounds × 3 passes), taken at the advance's cost
/// above: within 10 % but DEFAULT's (+21 %); a jump costs 1.5–2.1 times its photons alone.
const JUMP_NS:f64=970.;const PHOTON_MULTIPLY_ADD_NS:f64=0.594;const ABSORPTION_MULTIPLY_ADD_NS:f64=1.48;
/// The held twin's step besides its two products (control.rs): 30–190 ns over DEFAULT, THREE, PGC3D
/// and GMD1C on the same core.
const HELD_NS:f64=100.;

/// A sampled state's parts of the work model (`interval_work`): its cluster's advance without the
/// propagation and its jump (ns), its generator's norm, its dimension and whether its cluster takes
/// the exact step (explicit excited states).
struct StateWork{advance:f64,jump:f64,norm:f64,n:usize,exact:bool}
fn state_work(model:&Model,states:&[Trajectory])->Vec<StateWork>{
    states.par_iter().map(|t|{
        let w=model.work(t.cluster);let n=model.dimension(t.cluster);let point=model.point(t.r,t.kappa);let exact=model.mixed(t.cluster);
        let norm=if exact{column_norm(&model.generator(t.cluster,&point,t.time).hamiltonian,n)}else{column_norm(&model.local(t.cluster,&point,t.time).hamiltonian,n)};
        StateWork{advance:ADVANCE_NS+MULTIPLY_ADD_NS*(w.generator+w.force)+PAIR_NS*w.pairs as f64,
            jump:JUMP_NS+PHOTON_MULTIPLY_ADD_NS*w.photons+ABSORPTION_MULTIPLY_ADD_NS*w.absorption,norm,n,exact}
    }).collect()
}
/// A step's advance at a state with its propagation (ns): two half steps' series; or the exact step
/// (engine.rs `Engine::exact`), its propagator over the grid's spacing -- n³ per term of its series to
/// double precision -- and per spacing a product and the force's forms (n² each), at the same costs
/// per multiply-add as the series' (not timed on its own).
fn advance_of(x:&StateWork,s:&Settings)->f64{
    if !x.exact{return x.advance+2.*half_step(x,s);}
    let spacings=(x.norm*s.dt/node_phase()).ceil().max(1.);let terms=series(x.norm,s.dt/spacings,f64::EPSILON).1 as f64;let n=x.n as f64;
    x.advance+TERM_MULTIPLY_ADD_NS*(n*n*n*terms+4.*spacings*n*n)
}
/// A half step's propagation at a state (ns): its series' sub-steps × terms at its generator's norm,
/// the step's dt and truncation.
fn half_step(x:&StateWork,s:&Settings)->f64{let(sub,terms)=series(x.norm,0.5*s.dt,s.truncation);(sub*terms) as f64*(TERM_NS+TERM_MULTIPLY_ADD_NS*(x.n*x.n) as f64)}
/// A step's work per interval (ns, `ADVANCE_NS` and the rest), counted on the sampled states -- the
/// same whatever the machine and its load: its steps' advances, each the generator's and the force's
/// multiply-adds and pairs of the state's cluster and two half steps' series; and per jump, one every
/// waiting time, the jump itself -- every class's photons, the picked class's absorbed light and the
/// rest -- and two more advances, the step redone to the jump and its rest. A readout per output is
/// the same at every step and left out.
fn interval_work(states:&[StateWork],s:&Settings,interval:f64,waiting:f64)->f64{
    let(mut advance,mut jump)=(0.,0.);
    for x in states{let a=advance_of(x,s);advance+=a;jump+=x.jump+2.*a;}
    let count=states.len().max(1) as f64;
    (interval/s.dt*advance+interval/waiting*jump)/count
}
/// The held twin's work per interval (control.rs, ns): per step its two products by the half step's
/// propagator (n² multiply-adds each) and its step's own (`HELD_NS`); per jump the step redone to it
/// and its rest by the series.
fn held_work(states:&[StateWork],s:&Settings,interval:f64,waiting:f64)->f64{
    let(mut step,mut jump)=(0.,0.);
    for x in states{step+=HELD_NS+2.*TERM_MULTIPLY_ADD_NS*(x.n*x.n) as f64;jump+=4.*half_step(x,s);}
    let count=states.len().max(1) as f64;
    (interval/s.dt*step+interval/waiting*jump)/count
}

/// A candidate step's predicted cost relative to a reference's (module documentation): its work per
/// interval relative to the reference's, times the trajectories the statistics need relative to the
/// reference's -- at most the largest over the targets of (the reference's room/its room)², the room
/// what a target leaves past the a-priori estimates and the step's bias, as the ledger adds them:
/// exact for the target that binds, whatever the trajectories' spreads.
fn predicted_cost(c:&Check,work:f64,reference:&Check,apriori:[f64;2])->f64{
    let rooms=|c:&Check|[POPULATION_TARGET-apriori[0]-c.population*A_PRIORI_SHARE*POPULATION_TARGET,TEMPERATURE_TARGET-apriori[1]-c.motion*A_PRIORI_SHARE*TEMPERATURE_TARGET];
    let(mine,theirs)=(rooms(c),rooms(reference));
    if mine.iter().any(|&r|!(r>0.)){return f64::INFINITY;}
    work*(0..2).map(|k|(theirs[k].max(0.)/mine[k]).powi(2)).fold(0.,f64::max)
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
/// after), and so are T and the mean velocity (`start`). An output where no atom is held (the loss
/// one) has no held atom to describe: its populations, T, ⟨v⟩, n̄ and distributions are the last
/// output's that had one -- each row still a probability vector, the app's p (1 − L) zero there --,
/// outside the validated regime, and the flag says from when.
fn assemble(model:&Model,estimates:&[Estimate],outcome:&Outcome,listed:&[usize],times:&[f64],frequencies:[f64;3],trajectories:u64,dt:f64,initial:&[Complex],mut ledger:Ledger,twin:bool)->Solution{
    let hm=model.hbar_over_mass;
    let velocity=std::array::from_fn(|a|histograms(&outcome.samples,listed,&|s|s.1[a]*hm));
    let position=std::array::from_fn(|a|histograms(&outcome.samples,listed,&|s|s.0[a]));
    let first:Vec<f64>=(0..STATES).map(|i|initial[i*STATES+i].re).collect();
    let sizes:Vec<String>=model.ground.clusters.iter().map(|c|c.dimension().to_string()).collect();
    let explicit:usize=model.ground.clusters.iter().map(|c|c.excited.len()).sum();
    let internal=if explicit>0{format!("内态（子空间 {}，其中 {explicit} 个激发态显式、精确传播，其余激发态二阶消去）",sizes.join("+"))}else{format!("消去激发态后的基态内态（子空间 {}）",sizes.join("+"))};
    // Each output's source: itself where an atom is held (t = 0 always), else the last that held one.
    let held=|i:usize|i==0||estimates[i].loss<1.;
    let source:Vec<usize>=(0..estimates.len()).scan(0,|last,i|{if held(i){*last=i;}Some(*last)}).collect();
    if let Some(gone)=(0..estimates.len()).find(|&i|!held(i)){
        let note=format!("{} 起已无原子被俘获（丢失概率 1）：此后的布居、温度与分布沿用 {} 的值，仅供参考",when(times[gone]),when(times[source[gone]]));
        ledger.flagged=Some(match ledger.flagged.take(){Some(r)=>format!("{note}；{r}"),None=>note});
    }
    let carried=|f:&dyn Fn(&Estimate)->[f64;3]|->Vec<[f64;3]>{source.iter().map(|&j|f(&estimates[j])).collect()};
    Solution{
        populations:source.iter().map(|&j|if j==0{first.clone()}else{estimates[j].populations.clone()}).collect(),
        sample_errors:estimates.iter().enumerate().map(|(i,e)|if i==0{0.}else{e.population_errors.iter().fold(e.loss_error,|m,&x|m.max(x))}).collect(),
        temperature_uk:carried(&|e|e.temperature),
        temperature_errors_uk:carried(&|e|e.temperature_error),
        temperature_total_uk:source.iter().map(|&j|estimates[j].temperature_total).collect(),
        temperature_total_errors_uk:source.iter().map(|&j|estimates[j].temperature_total_error).collect(),
        mean_velocity_mps:carried(&|e|e.mean_velocity),
        trap_frequency_khz:frequencies.map(|w|w/std::f64::consts::TAU*1e3),
        mean_occupation:carried(&|e|e.occupation),
        loss_probability:{let mut worst=0.;estimates.iter().map(|e|{worst=e.loss.max(worst);worst}).collect()},
        velocity,position,
        classical:estimates.iter().enumerate().map(|(i,e)|held(i)&&classical(e,frequencies)).collect(),
        trajectories,needed:trajectories,time_step_us:dt,
        method:format!("半经典 3D：经典质点（位置、动量）携带{internal}，量子跳跃（自发辐射按偶极辐射方向图反冲，吸收按光束反冲并保持精确的平均力），机械能高于势阱深度即算损失，{trajectories} 条轨迹，步长 {dt:.3} μs；控制变量：谐振孪生（T）{}",if twin{"与静止孪生（布居）"}else{""}),
        reason:None,
        approximations:ledger,
        device:Device::Cpu,device_note:None,
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    /// The batches: the remaining need in whole groups, at least the width, at most doubling; on the
    /// GPU whole waves.
    #[test]
    fn a_batch_is_the_remaining_need_within_its_bounds(){
        assert_eq!(next_batch(300,256,32,TASK),48);
        assert_eq!(next_batch(260,256,32,TASK),32);
        assert_eq!(next_batch(845,256,32,TASK),256);
        assert_eq!(next_batch(700,640,640,640),640);
        assert_eq!(next_batch(4000,1280,640,640),1280);
        assert_eq!(next_batch(1700,1280,640,640),640);
    }
}
