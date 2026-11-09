//! The loss's check after the product (solve.rs # Time step): the step's bias on the loss, read from the step's local
//! errors weighted by the loss's dual -- Szepessy, Tempone and Zouraris's a posteriori weak error --, the dual carried by the
//! energy's own transfer over the product's trajectories.
//!
//! # The dual
//! The loss at the output t_k is the share of atoms lost by then. Its value function u_k(t, x), the chance that an atom at x at
//! t is lost by t_k, applied to a step's local weak error at x and summed over the steps before t_k is the step's bias on it,
//! to leading order. u_k is read on the energy's features F(x): e = E/D, the loss's energy over the trap's depth (engine.rs
//! `Engine::loss_energy`: −1 at the bottom of the reference well, 0 at the trap's far level), on hats at `KNOTS`, and lost.
//! The conservative motion keeps e, so that its local error is the step's heating's, and the loss is e's threshold. Their
//! transfer over an output interval, C_d, is the regression of F at the next output on F at this one over the product's
//! trajectories, with a ridge of one trajectory's weight, the lost state's row its own (absorbing: lost stays lost, however few
//! atoms the product lost by then); u_k(t_d) = C_d ⋯ C_{k−1} e_lost. Not the populations: an internal state's memory is its
//! waiting time between jumps, far below an output interval, and read across outputs their dual is noise. Not per-axis
//! energies, kinetic or harmonic: the step's error in an orbit's phase reads as a change of them of either sign (R5009's T_z
//! 1.6 against 0.07; the kinetic energies' duals read the T of every scene of the gate, the cycling control's, the
//! three-beam MOT's and the tetrahedral MOT's whose truths hold no T step error among them, 3 to 700 times its tolerance). T's
//! step error is read before the product, on its own windows (solve.rs `drift`).
//!
//! # Its local errors: the reads
//! A read at a state: the step's whole dynamics -- its flow, its jumps and its loss's rule at every step's end (engine.rs
//! `Engine::step`) -- over M steps of dt against 2M of dt/2 (`fine`), M the waiting time between jumps in steps, from the state
//! on streams of their own, the same for both: per step the features' change at the ends, times Richardson's factor. Not the
//! jump-free flows with the jump's survival and a jump's sample at the start: in the gray molasses's tail (PGC1D_BIAS_OFF at
//! 1.28 us, e above −0.4) they read a fifth to a quarter of the whole dynamics' error, which takes the jumps' own -- their
//! crossing, their kicks, the transients after them.
//! The reads are drawn in the product's own batches (lib.rs `walk`): per trajectory and output interval at a step drawn
//! uniformly in it, with the inclusion probability of the interval and the state's energy bin (`Sampling`), and weighted by
//! its inverse (Horvitz and Thompson): no state is kept, and the check's trajectories count for the statistics. A batch's
//! inclusion takes `rate` reads per trajectory, half of them alike over the intervals and bins the atoms hold, half as the
//! dual weighs them (`sampling`: the sensitivity of the loss at every later output to the bin's features over its
//! tolerance, times the reads' mean magnitude there) -- the first batch the pilot's dual (its trajectories' energies), the next
//! ones the product's so far; over a product the reads cost at most a pilot's work (solve.rs), unless more are bought.
//!
//! # The verdict
//! Per output the bias b_k and its spread over the groups (stats.rs `Spread`, Student's t at the groups read: the dual refit
//! without each group, its reads left out), its interval at z; z the largest of the outputs' looks, their number the
//! participation ratio of the replicates' correlation over the outputs (Li and Ji's effective number of tests). B⁻ -- the
//! interval's distance from no bias -- and B⁺ -- its far end -- over the tolerance, the largest over the outputs; `could`, both
//! with their errors shrunk to a read at every interval an atom held. What the solve does with them: `Verdict::round`.

use crate::engine::{Engine,Trajectory};
use crate::rng::{Purpose,Rng};
use crate::solve::{richardson,CONFIDENCE,ORDER};
use crate::read::jackknife_error;
use crate::stats::{largest_look,Spread};

/// The energy's hats (e = E/D): the trap's bottom to a fifth of its depth past its far level, by fifths.
pub(crate) const KNOTS:[f64;7]=[-1.,-0.8,-0.6,-0.4,-0.2,0.,0.2];
/// The features: the hats, and lost.
pub(crate) const FEATURES:usize=KNOTS.len()+1;
const LOST:usize=KNOTS.len();
/// The energy's bins: the hats' segments.
pub(crate) const BINS:usize=KNOTS.len()-1;
const SQUARE:usize=FEATURES*FEATURES;

/// A state's features: held at energy e, its hats (e clamped to the knots); lost, lost alone.
pub(crate) fn features(e:Option<f64>)->[f64;FEATURES]{
    let mut f=[0.;FEATURES];
    match e{None=>f[LOST]=1.,Some(e)=>{let x=e.clamp(KNOTS[0],KNOTS[BINS]);let k=bin(x);let w=(x-KNOTS[k])/(KNOTS[k+1]-KNOTS[k]);f[k]=1.-w;f[k+1]=w;}}
    f
}
/// An energy's bin: the hats' segment it lies in, clamped.
pub(crate) fn bin(e:f64)->usize{KNOTS[1..BINS].iter().take_while(|&&k|e>k).count()}

/// A read (`read`): its output interval, its energy bin, its inverse inclusion probability and per step the features' change.
#[derive(Clone,Debug)]
pub struct Read{pub interval:u32,pub bin:u8,pub weight:f64,pub change:[f64;FEATURES]}

/// What a trajectory leaves the check (lib.rs `walk`).
#[derive(Clone,Debug,Default)]
pub struct Trace{
    /// Per output it ran through its energy while held (`Engine::loss_energy`), NaN once lost.
    pub energies:Vec<f32>,
    /// Its reads.
    pub reads:Vec<Read>,
    /// The intervals where a read could have been drawn: the atom held at the drawn step.
    pub chances:u32,
}
impl Trace{
    /// A trajectory's trace through `outputs` outputs, empty.
    pub(crate) fn new(outputs:usize)->Trace{Trace{energies:Vec::with_capacity(outputs),..Default::default()}}
    /// Its state at the next output: its energy while held, NaN once lost.
    pub(crate) fn output(&mut self,engine:&Engine,held:Option<&Trajectory>){self.energies.push(held.map_or(f32::NAN,|t|engine.loss_energy(t) as f32));}
    /// Whether it was lost by the last output it ran through.
    pub(crate) fn lost(&self)->bool{self.energies.iter().any(|e|e.is_nan())}
    /// Its state at the step drawn in output interval i, held (`drawn`, `deviate` its inclusion's): a chance, read (`read`, `fine`
    /// the engine at dt/2) where the deviate falls below its energy bin's inclusion, weighted by its inverse (Horvitz and Thompson).
    pub(crate) fn look(&mut self,engine:&Engine,fine:&Engine,sampling:&Sampling,t:&Trajectory,interval:usize,deviate:f64){
        self.chances+=1;let b=bin(engine.loss_energy(t));let p=sampling.inclusion[interval-1][b];
        if deviate<p{self.reads.push(Read{interval:interval as u32,bin:b as u8,weight:1./p,change:read(engine,fine,t,sampling.span,interval)});}
    }
}

/// A batch's reads (`walk`): per output interval and energy bin a read's inclusion probability, and the reads' span in steps.
#[derive(Clone,Debug)]
pub struct Sampling{pub inclusion:Vec<[f64;BINS]>,pub span:usize}
impl Sampling{
    /// Every read's probability `rate` over the intervals: a trajectory's reads, alike everywhere.
    pub fn alike(intervals:usize,span:usize,rate:f64)->Sampling{let p=(rate/intervals.max(1) as f64).min(1.);Sampling{inclusion:vec![[p;BINS];intervals],span}}
}

/// A trajectory's read in output interval i (the steps after output i − 1): the step it is drawn at and its inclusion's
/// deviate, on the check's stream (event 2i).
pub(crate) fn drawn(seed:u64,index:u64,interval:usize,steps:usize)->(usize,f64){
    let mut r=Rng::new(seed,Purpose::Check,index,2*interval as u64);let at=((r.uniform()*steps as f64) as usize).min(steps.max(1)-1);(at,r.uniform())
}

/// A read at a state in output interval i (module documentation, # Its local errors): the step's whole dynamics over `span`
/// steps of dt against 2 span of dt/2 (`fine`), from the state on streams of their own -- its index with the check's bit and
/// the interval above a trajectory's --, the same for both; per step the features' change at the ends, times Richardson's factor.
pub(crate) fn read(engine:&Engine,fine:&Engine,x:&Trajectory,span:usize,interval:usize)->[f64;FEATURES]{
    let mut coarse=x.clone();coarse.index=x.index|1<<63|(interval as u64)<<32;let mut halves=coarse.clone();
    for _ in 0..span{if coarse.alive{engine.step(&mut coarse);}}
    for _ in 0..2*span{if halves.alive{fine.step(&mut halves);}}
    let at=|t:&Trajectory|features(t.alive.then(||engine.loss_energy(t)));
    let(a,b)=(at(&coarse),at(&halves));let factor=richardson(ORDER)/span as f64;
    std::array::from_fn(|k|factor*(a[k]-b[k]))
}

/// A trace's features at output d: held there, its energy's; lost, lost; None past the outputs it ran through.
fn at(t:&Trace,d:usize)->Option<[f64;FEATURES]>{t.energies.get(d).map(|&e|features((!e.is_nan()).then_some(e as f64)))}

/// The sums the dual and the reads take, per group of the product (`TASK` trajectories in index order): per output interval
/// Σ F Fᵀ at its start, Σ F at its start times F at its end, Σ the reads' weighted changes and the trajectories through it.
#[derive(Clone)]
struct Sums{start:Vec<[f64;SQUARE]>,across:Vec<[f64;SQUARE]>,injection:Vec<[f64;FEATURES]>,through:Vec<f64>}
impl Sums{
    fn new(intervals:usize)->Sums{Sums{start:vec![[0.;SQUARE];intervals],across:vec![[0.;SQUARE];intervals],injection:vec![[0.;FEATURES];intervals],through:vec![0.;intervals]}}
    fn add(&mut self,t:&Trace){
        for d in 0..self.through.len(){
            let(Some(f),Some(g))=(at(t,d),at(t,d+1))else{continue};
            for a in 0..FEATURES{if f[a]==0.{continue;}for b in 0..FEATURES{self.start[d][a*FEATURES+b]+=f[a]*f[b];self.across[d][a*FEATURES+b]+=f[a]*g[b];}}
            self.through[d]+=1.;
        }
        for r in &t.reads{if let Some(x)=self.injection.get_mut(r.interval as usize-1){for a in 0..FEATURES{x[a]+=r.weight*r.change[a];}}}
    }
    fn less(&self,o:&Sums)->Sums{let mut s=self.clone();
        for d in 0..s.through.len(){for k in 0..SQUARE{s.start[d][k]-=o.start[d][k];s.across[d][k]-=o.across[d][k];}for a in 0..FEATURES{s.injection[d][a]-=o.injection[d][a];}s.through[d]-=o.through[d];}s}
    /// Per interval the transfer C_d (row-major, F at its start to F at its end): the held atoms' regressed, the lost's row the
    /// absorbing state's own.
    fn transfers(&self)->Vec<[f64;SQUARE]>{(0..self.through.len()).map(|d|{let mut c=regress(&self.start[d],&self.across[d],self.through[d]);
        for k in 0..FEATURES{c[LOST*FEATURES+k]=if k==LOST{1.}else{0.};}c}).collect()}
    /// The bias per output (`steps` per interval): the loss's part of dF, dF(d + 1) = C_dᵀ (dF(d) + steps m_d).
    fn bias(&self,steps:usize)->Vec<f64>{
        let c=self.transfers();let mut df=[0.;FEATURES];let mut out=vec![0.];
        for d in 0..self.through.len(){
            let n=self.through[d].max(1.);let x:[f64;FEATURES]=std::array::from_fn(|a|df[a]+steps as f64*self.injection[d][a]/n);
            df=std::array::from_fn(|b|(0..FEATURES).map(|a|c[d][a*FEATURES+b]*x[a]).sum());out.push(df[LOST]);
        }
        out
    }
}

/// The least squares of `across` on `start` (F Fᵀ and F Gᵀ summed over `rows`), with a ridge of one row's weight: the transfer.
fn regress(start:&[f64;SQUARE],across:&[f64;SQUARE],rows:f64)->[f64;SQUARE]{
    let n=FEATURES;let trace:f64=(0..n).map(|i|start[i*n+i]).sum();let ridge=trace/(n as f64*rows.max(1.))+f64::MIN_POSITIVE;
    let mut l=[0.;SQUARE];
    for i in 0..n{for j in 0..=i{let mut s=start[i*n+j]+if i==j{ridge}else{0.};for k in 0..j{s-=l[i*n+k]*l[j*n+k];}
        if i==j{l[i*n+i]=s.max(f64::MIN_POSITIVE).sqrt();}else{l[i*n+j]=s/l[j*n+j];}}}
    let mut x=[0.;SQUARE];
    for c in 0..n{let mut y=[0.;FEATURES];for i in 0..n{let mut s=across[i*n+c];for k in 0..i{s-=l[i*n+k]*y[k];}y[i]=s/l[i*n+i];}
        for i in (0..n).rev(){let mut s=y[i];for k in i+1..n{s-=l[k*n+i]*x[k*n+c];}x[i*n+c]=s/l[i*n+i];}}
    x
}

/// What the check read on a run (solve.rs `Solution::loss_check`): its reads, the intervals they could have been at, their span
/// in steps, B⁻, B⁺ and `could` at the step kept, the output B⁺ came from, the steps per interval it refined from, and whether
/// the step stands verified (B⁺ ≤ 1).
#[derive(Clone,Debug)]
pub struct Report{pub reads:u64,pub chances:u64,pub span:usize,pub lower:f64,pub upper:f64,pub could:[f64;2],pub binding:usize,pub refined:Vec<usize>,pub verified:bool}

/// The check's verdict on a product's traces (module documentation, # The verdict): per output the bias and its error at
/// z (`Spread::at`), B⁻, B⁺ and `could` (over each output's tolerance), the output B⁺ came from, and the reads taken against the
/// intervals they could have been.
#[derive(Clone,Debug)]
pub struct Verdict{pub bias:Vec<[f64;2]>,pub lower:f64,pub upper:f64,pub could:[f64;2],pub binding:usize,pub reads:u64,pub chances:u64}
impl Verdict{
    /// B⁻ beyond 1: the step's bias on the loss lies beyond its tolerance by its errors.
    pub fn refines(&self)->bool{self.lower>1.}
    /// B⁺ within 1: the bias within its tolerance by its errors, the step verified.
    pub fn verified(&self)->bool{self.upper<=1.}
    /// Unsettled: B⁻ within 1, B⁺ beyond it, and B⁻ beyond it once every chance were read (none read: nothing settled).
    pub fn unsettled(&self)->bool{!self.refines()&&!self.verified()&&self.could[0]>1.}
    /// The finer step's factor, B⁺^(1/p): the bias's upper side brought within its tolerance.
    pub fn factor(&self)->f64{self.upper.max(1.).powf(1./ORDER)}
    /// The factor reading every chance would leave: B⁺ with its errors shrunk.
    pub fn read_factor(&self)->f64{self.could[1].max(1.).powf(1./ORDER)}
    /// The round after this verdict at `steps` per interval (solve.rs `run_at`, after any reads it bought), at most `most`,
    /// `previous` the [B⁺, B⁻] of the round that refined to it.
    pub fn round(&self,steps:usize,most:usize,previous:Option<[f64;2]>)->Round{
        if self.verified(){return Round::Verified;}
        if !(self.refines()||self.unsettled()){return Round::Unreadable;}
        if previous.is_some_and(|[upper,lower]|self.upper>=upper&&self.lower>=lower){return Round::Stalled;}
        if steps>=most{return Round::Bounded;}
        Round::Refine(((steps as f64*self.factor()).ceil() as usize).clamp(steps+1,most))
    }
}

/// A round of the check (`Verdict::round`): the step verified, the run again at a finer step, or the step standing unverified
/// -- the far end of its bias's interval in the record and the flag saying why (solve.rs `run_at`).
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum Round{
    /// B⁺ within 1.
    Verified,
    /// B⁻ beyond 1, or unsettled: the run again at this many steps per interval, B⁺^(1/p) finer.
    Refine(usize),
    /// Unverified: the check cannot read the bias -- B⁻ within 1 < B⁺, and every chance read could not take B⁻ beyond it.
    Unreadable,
    /// Unverified: the last refinement made no progress, neither B⁺ nor B⁻ below its round's.
    Stalled,
    /// Unverified: the work bound stops the rounds.
    Bounded,
}

/// The verdict on `traces` (in index order, `group` trajectories a jackknife group) at `steps` per output interval against
/// each output's tolerance on the loss.
pub(crate) fn verdict(traces:&[Trace],group:usize,steps:usize,tolerance:&[f64])->Verdict{
    let intervals=tolerance.len()-1;let(reads,chances)=traces.iter().fold((0u64,0u64),|s,t|(s.0+t.reads.len() as u64,s.1+t.chances as u64));
    // No read reads nothing: unread, and what reads could show unknown -- a run that lost atoms before any was drawn buys them.
    if reads==0{return Verdict{bias:vec![[0.,f64::INFINITY];intervals+1],lower:0.,upper:f64::INFINITY,could:[f64::INFINITY;2],binding:intervals,reads,chances};}
    let groups:Vec<Sums>=traces.chunks(group).map(|g|{let mut s=Sums::new(intervals);for t in g{s.add(t);}s}).collect();
    let mut total=Sums::new(intervals);for t in traces{total.add(t);}
    let all=total.bias(steps);
    let replicates:Vec<Vec<f64>>=groups.iter().map(|g|total.less(g).bias(steps)).collect();let n=replicates.len() as f64;
    let spread:Vec<Spread>=(0..=intervals).map(|k|{let v:Vec<f64>=replicates.iter().map(|r|r[k]).collect();Spread{error:jackknife_error(v.iter().copied()),groups:n}}).collect();
    let error:Vec<f64>=spread.iter().map(|s|s.error).collect();
    let z=largest_look(looks(&replicates,&error)).max(CONFIDENCE);
    // What reading every chance could show (`could`, B⁻ and B⁺): each group's deviation shrunk as its reads' error, by √(reads/chances).
    let shrink=(reads as f64/chances.max(1) as f64).sqrt();let shrunk=|s:Spread|Spread{error:shrink*s.error,..s};
    // The interval [b − at, b + at]: its distance from no bias, and its far end from it.
    let near=|k:usize,s:Spread|(all[k].abs()-s.at(z)).max(0.)/tolerance[k];
    let far=|k:usize,s:Spread|(all[k].abs()+s.at(z))/tolerance[k];
    let most=|f:&dyn Fn(usize)->f64|(1..=intervals).map(f).fold(0.,f64::max);
    let binding=(1..=intervals).max_by(|&a,&b|far(a,spread[a]).total_cmp(&far(b,spread[b]))).unwrap_or(intervals);
    Verdict{bias:(0..=intervals).map(|k|[all[k],spread[k].at(z)]).collect(),lower:most(&|k|near(k,spread[k])),upper:most(&|k|far(k,spread[k])),
        could:[most(&|k|near(k,shrunk(spread[k]))),most(&|k|far(k,shrunk(spread[k])))],binding,reads,chances}
}

/// The outputs' effective number of looks: the participation ratio (tr R)²/tr R² of the replicates' correlation R over the
/// outputs that read an error (Li and Ji): one for a single bias read at every output, their count for independent ones. tr R²
/// as the groups' Gram matrix's square norm, small however many outputs.
fn looks(replicates:&[Vec<f64>],error:&[f64])->f64{
    let k:Vec<usize>=(0..error.len()).filter(|&k|error[k]>0.).collect();if k.is_empty(){return 1.;}
    let(g,n)=(replicates.len(),replicates.len() as f64);let mut gram=vec![0.;g*g];
    for &k in &k{let m=replicates.iter().map(|r|r[k]).sum::<f64>()/n;let v:Vec<f64>=replicates.iter().map(|r|r[k]-m).collect();let s=v.iter().map(|x|x*x).sum::<f64>().sqrt();
        for a in 0..g{if v[a]==0.{continue;}for b in 0..g{gram[a*g+b]+=v[a]*v[b]/(s*s);}}}
    (k.len()*k.len()) as f64/gram.iter().map(|x|x*x).sum::<f64>()
}

/// The next batch's sampling (module documentation, # Its local errors): `rate` reads per trajectory, half alike over the
/// intervals and bins the traces' atoms hold, half as the dual weighs each bin -- the largest over later outputs of the loss's
/// sensitivity to its features over the output's tolerance, times the reads' mean magnitude there (one per feature before any
/// read) --, each at most 1.
pub(crate) fn sampling(traces:&[Trace],steps:usize,tolerance:&[f64],span:usize,rate:f64)->Sampling{
    let intervals=tolerance.len()-1;let mut total=Sums::new(intervals);for t in traces{total.add(t);}let c=total.transfers();
    // The sensitivity per interval and feature: steps C_d ⋯ C_{k−1} e_lost over tol_k, the largest over k > d.
    let mut sensitivity=vec![[0.;FEATURES];intervals];
    for k in 1..=intervals{let mut w=[0.;FEATURES];w[LOST]=1.;
        for d in (0..k).rev(){w=std::array::from_fn(|a|(0..FEATURES).map(|b|c[d][a*FEATURES+b]*w[b]).sum());for a in 0..FEATURES{sensitivity[d][a]=f64::max(sensitivity[d][a],steps as f64*w[a].abs()/tolerance[k]);}}}
    // The reads' mean magnitude per bin and feature.
    let mut magnitude=[[0.;FEATURES+1];BINS];
    for t in traces{for r in &t.reads{let m=&mut magnitude[r.bin as usize];for a in 0..FEATURES{m[a]+=r.weight*r.change[a].abs();}m[FEATURES]+=r.weight;}}
    let mean=|b:usize,a:usize|if magnitude[b][FEATURES]>0.{magnitude[b][a]/magnitude[b][FEATURES]}else{1.};
    // The atoms' share per interval and bin.
    let mut held=vec![[0.;BINS];intervals];
    for t in traces{for d in 0..intervals{if let Some(&e)=t.energies.get(d){if !e.is_nan(){held[d][bin(e as f64)]+=1./traces.len() as f64;}}}}
    let weight:Vec<[f64;BINS]>=(0..intervals).map(|d|std::array::from_fn(|b|(0..FEATURES).map(|a|sensitivity[d][a]*mean(b,a)).sum())).collect();
    let(atoms,weighed)=(0..intervals).fold((0.,0.),|s,d|(s.0+held[d].iter().sum::<f64>(),s.1+(0..BINS).map(|b|held[d][b]*weight[d][b]).sum::<f64>()));
    if !(atoms>0.){return Sampling::alike(intervals,span,0.);}
    let inclusion=(0..intervals).map(|d|std::array::from_fn(|b|(rate*if weighed>0.{0.5/atoms+0.5*weight[d][b]/weighed}else{1./atoms}).min(1.))).collect();
    Sampling{inclusion,span}
}

#[cfg(test)]
mod tests{
    use super::*;
    /// The features: hats summing to one over the knots, clamped beyond them; lost alone.
    #[test]
    fn the_features_are_the_energy_s_hats_and_lost(){
        for e in [-1.3,-1.,-0.93,-0.5,-0.2,0.,0.07,0.2,0.9]{let f=features(Some(e));assert!((f.iter().sum::<f64>()-1.).abs()<1e-15&&f[LOST]==0.,"{e}: {f:?}");
            let x=e.clamp(KNOTS[0],KNOTS[BINS]);assert!((KNOTS.iter().zip(&f).map(|(k,w)|k*w).sum::<f64>()-x).abs()<1e-12,"{e}: the hats interpolate e");}
        assert_eq!(features(None),{let mut f=[0.;FEATURES];f[LOST]=1.;f});
        assert_eq!((bin(-1.),bin(-0.81),bin(-0.8),bin(-0.79),bin(-0.01),bin(0.),bin(0.1),bin(0.5)),(0,0,0,1,4,4,5,5));
    }
    /// The read on a made-up product: atoms that each interval move up a hat from the trap's far level's below with the chance of
    /// their energy, the lost staying lost, and reads that move every held atom up by a known share of a hat per step: the bias
    /// is that share carried to the lost, and B⁻ refines where it lies beyond its tolerance by its errors; without reads' changes
    /// the step is verified.
    #[test]
    fn the_verdict_carries_the_reads_to_the_loss(){
        let(atoms,intervals,steps)=(4096usize,6usize,10usize);let mut rng=Rng::new(3,Purpose::Check,0,0);
        let shift=1e-3;let traces:Vec<Trace>=(0..atoms).map(|_|{
            let mut e:Option<f64>=Some(-0.9);let mut t=Trace::default();
            for d in 0..=intervals{t.energies.push(e.map_or(f32::NAN,|x|x as f32));if d==intervals{break;}
                // A read every interval, inclusion 1: per step the hats' change of an energy moved up by `shift`.
                if let Some(x)=e{let(a,b)=(features(Some(x)),features(Some(x+shift)));t.reads.push(Read{interval:d as u32+1,bin:bin(x) as u8,weight:1.,change:std::array::from_fn(|k|b[k]-a[k])});t.chances+=1;}
                e=e.and_then(|x|{let up=x+0.35*rng.uniform();if up>0.{None}else{Some(up)}});}
            t}).collect();
        let tolerance=vec![0.02;intervals+1];let v=verdict(&traces,8,steps,&tolerance);
        let lost=(1..=intervals).map(|k|traces.iter().filter(|t|t.energies[k].is_nan()).count() as f64/atoms as f64).collect::<Vec<_>>();
        assert!(lost[intervals-1]>0.2,"the made-up atoms are lost: {lost:?}");
        // The shift moves the escape ahead by steps × shift per interval: the loss's bias at least its slope's share.
        assert!(v.bias[intervals][0]>0.&&v.refines()&&!v.verified(),"{v:?}");
        let quiet:Vec<Trace>=traces.iter().map(|t|Trace{reads:t.reads.iter().map(|r|Read{change:[0.;FEATURES],..r.clone()}).collect(),..t.clone()}).collect();
        let q=verdict(&quiet,8,steps,&tolerance);assert!(q.bias.iter().all(|b|b[0]==0.)&&!q.refines()&&q.verified(),"{q:?}");
    }
    /// A round (`Verdict::round`): verified within the tolerance; beyond it, or unsettled, refined by B⁺^(1/p); unverified where
    /// the check cannot read the bias, where a refinement made no progress -- neither B⁺ nor B⁻ fell --, and at the work bound.
    #[test]
    fn a_round_without_progress_stands_unverified(){
        let at=|lower:f64,upper:f64,could:f64|Verdict{bias:vec![],lower,upper,could:[could,upper],binding:1,reads:1,chances:1};
        assert_eq!(at(0.,0.9,0.).round(10,100,None),Round::Verified);
        assert_eq!((at(1.5,4.,2.).round(10,100,None),at(0.5,4.,1.2).round(10,100,None)),(Round::Refine(20),Round::Refine(20)));
        assert_eq!(at(0.5,4.,0.8).round(10,100,None),Round::Unreadable);
        assert_eq!(at(1.5,4.,2.).round(20,100,Some([3.,1.2])),Round::Stalled);
        assert_eq!((at(1.5,2.5,2.).round(20,100,Some([3.,1.2])),at(1.1,4.,2.).round(20,100,Some([3.,1.2]))),(Round::Refine(32),Round::Refine(40)));
        assert_eq!(at(1.5,4.,2.).round(100,100,None),Round::Bounded);
    }
    /// The sampling takes `rate` reads per trajectory over the intervals and bins its atoms hold, half alike.
    #[test]
    fn the_sampling_takes_its_rate(){
        let traces:Vec<Trace>=(0..64).map(|i|Trace{energies:(0..=5).map(|d|if i%7==0&&d>3{f32::NAN}else{-0.9+0.1*(d as f32)}).collect(),..Default::default()}).collect();
        let tolerance=vec![0.02;6];let s=sampling(&traces,10,&tolerance,3,2.);
        let mut held=vec![[0.;BINS];5];for t in &traces{for d in 0..5{let e=t.energies[d];if !e.is_nan(){held[d][bin(e as f64)]+=1./64.;}}}
        let expected:f64=(0..5).map(|d|(0..BINS).map(|b|held[d][b]*s.inclusion[d][b]).sum::<f64>()).sum();
        assert!((expected-2.).abs()<1e-12,"{expected}");
        let alike=Sampling::alike(5,3,2.);assert!(alike.inclusion.iter().all(|x|x.iter().all(|&p|(p-0.4).abs()<1e-15)));
    }
}
