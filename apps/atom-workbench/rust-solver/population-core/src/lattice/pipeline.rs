//! The population pipeline: every output of a request is a matrix function
//! of one lattice operator applied to x0, read from one reduced model.
//!
//! 1. The lattice: the explicit set P and its ring Q (`charge_set`), one per
//!    request.
//! 2. The run's operator (`fold`): the request's fold, the pieces whose
//!    action over the run is negligible whatever x0 left out
//!    (`RunFold::screened`, half the a-priori share), and the far shell's
//!    rings beyond the second order (the other half): their first-order
//!    Duhamel responses on x0's evolution at the run's sample times
//!    (`fold::Duhamel`), followed until what the next rings can land is
//!    within half of it, the outermost dropped and the far modes of the
//!    others pruned within the rest; the rings kept become ring targets.
//! 3. One rational Krylov space (`space`): x0, the gated steps'
//!    translations (with no solve), the bordered steady column, then the
//!    real ladder from the fastest scale the outputs need down to 1 / T in a
//!    fixed bit-reversed cycle; before the first cycle, x0's components on
//!    the centre generator's eigenvectors of population weight at least the
//!    a-priori share at the first output time (exact directions of a
//!    one-mode lattice: a strong drive's Rabi oscillation, which real poles
//!    resolve slowly; measured on D1 + D2 at 2000 mW/cm^2, 20 us: 0.24 off
//!    without them, 6.6e-3 with them, the standard set unchanged).
//! 4. Checkpoints after the steady column and after every cycle of the
//!    ladder: the Galerkin model (`reduced`) and its outputs (the trajectory
//!    on the grid, the horizons, the slowest relaxation). An output's
//!    estimate is its change since the previous checkpoint: a cycle visits
//!    every scale of the ladder once, so the change over a cycle is the
//!    convergence check (an estimate, as QuTiP's step control is, not a
//!    bound). A trajectory sample's bound adds the change of its projection
//!    onto the probability simplex.
//! 5. A checkpoint is accepted when every bound is within the tolerance and
//!    its cycle added directions. The run stops there, or when a cycle added
//!    no direction, or the largest change did not shrink over a cycle, or
//!    the basis budget is spent, and returns its best checkpoint (`Answer`):
//!    converged, or partial with the reason and every sample's own estimate.
//!    Every checkpoint is reported to the observer as it is made.
use super::*;
use super::accuracy::{self,A_PRIORI_SHARE,POPULATION_TOLERANCE};
use super::charge_set::{self,ExplicitSet,LatticeSets};
use super::fold::{self,Duhamel,RunFold,Screened,StoredFold};
use super::krylov::Vectors;
use super::poles::{InnerWork,Lattice,Pole};
use super::reduced::{Galerkin,Model,SlowRate};
use super::space::{self,Column,Space};

/// Memory guard: a request takes about BASE_BYTES plus MODE_BYTES per
/// lattice mode of each run, and its runs execute at once (`Request::runs`:
/// a full request's trajectory and mean, each with its own space).
/// Measured (private bytes, with the former error estimator, which took
/// about as much as the basis): D1 + three D2 lasers, 13 modes, 0.62 GB; six
/// lasers without the trap, 63 modes, 2.0 GB. A request beyond the budget
/// stops before its first solve, with the reason. The budget is MEMORY_SHARE
/// of the machine's installed physical memory: a run beyond it pages, and
/// every solve sweeps the lattice and the whole basis, so it would stall the
/// machine rather than slow down; the installed memory rather than the free,
/// so whether a request is solved does not depend on what else runs.
/// MEMORY_SHARE is a deliberate margin: the other half for the system, the
/// app and the lab PC's other programs. What the guard leaves is the runs'
/// bases and their images (`Space::new`).
const BASE_BYTES:f64=5e8;
const MODE_BYTES:f64=3.2e7;
const MEMORY_SHARE:f64=0.5;
/// The memory a request of `runs` runs on lattices of `modes` modes is
/// planned to take (bytes).
fn planned_bytes(runs:usize,modes:usize)->f64{BASE_BYTES+MODE_BYTES*(runs*modes)as f64}
/// The real pole that resolves the solution at time t is about RESOLUTION / t
/// (shift-and-invert: van den Eshof and Hochbruck, SIAM J. Sci. Comput. 27,
/// 1438 (2006), gamma = t / 10).
const RESOLUTION:f64=10.;
/// The longest output time a request may ask for (us, 10 s) and the most
/// samples of its grid: the product's request range.
pub const LONGEST_TIME_US:f64=1e7;
pub const MOST_SAMPLES:usize=2000;

/// What a request asks for: a trajectory or horizons (or both).
pub struct Request<'a>{
    /// The uniform grid from 0 (empty: no trajectory), at most MOST_SAMPLES
    /// times up to LONGEST_TIME_US.
    pub times:&'a[f64],
    /// Horizons T (us) of the laboratory time average, each at most
    /// LONGEST_TIME_US.
    pub horizons:&'a[f64],
    pub groups:&'a[Vec<usize>],
    /// The explicit set shared by the runs of one request from the same rho0
    /// (the first run to need it builds it).
    pub explicit:&'a std::sync::OnceLock<Result<ExplicitSet,String>>,
    /// The runs of the request that execute at once (a full request's
    /// trajectory and mean: 2), which share the memory budget.
    pub runs:usize,
}

/// The error estimate of every output at one checkpoint (population units):
/// per trajectory sample and per horizon.
#[derive(Clone,Debug,Default)]
pub struct Estimate{pub trajectory:Vec<f64>,pub horizons:Vec<f64>}

/// One checkpoint, for the diagnostics.
#[derive(Clone,Debug)]
pub struct Checkpoint{
    pub solves:usize,pub columns:usize,
    /// The estimate and the largest bound over every output.
    pub estimate:Estimate,pub largest:f64,pub accepted:bool,
    pub growing:usize,pub slowest:Option<SlowRate>,
    /// Raw outputs: trajectory rows (row 0 rho0's), horizon rows.
    pub trajectory:Vec<Vec<f64>>,pub horizons:Vec<Vec<f64>>,
    /// The solves so far and the run's timing so far.
    pub inner:InnerWork,pub timing:Timing,
    pub milliseconds:f64,
}

/// The outputs of one checkpoint with their bounds: the run's result (its
/// best checkpoint) and every checkpoint reported as it is made.
#[derive(Clone,Debug)]
pub struct Answer{
    pub solves:usize,
    /// The lattice's modes (the fields' layout).
    pub modes:Vec<Charge>,
    /// Trajectory rows projected onto the probability simplex (row 0 is
    /// rho0's) and the bound of each (the estimate plus the projection's
    /// change).
    pub trajectory:Vec<Vec<f64>>,pub sample_errors:Vec<f64>,
    /// Horizon populations (laboratory time averages, raw), their fields on
    /// `modes` (Hermitian-symmetric) and bounds.
    pub horizons:Vec<Vec<f64>>,pub horizon_fields:Vec<Vec<C>>,pub horizon_errors:Vec<f64>,
    pub estimate:Estimate,pub slowest:Option<SlowRate>,
    /// Every bound within the tolerance.
    pub accepted:bool,
}
impl Answer{
    /// The largest bound over every output.
    pub fn largest(&self)->f64{self.sample_errors.iter().chain(&self.horizon_errors).fold(0_f64,|a,&b|a.max(b))}
}

#[derive(Clone,Debug,Default)]
pub struct Timing{pub eigenbasis_ms:f64,pub explicit_set_ms:f64,pub fold_ms:f64,pub setup_ms:f64,pub solves_ms:f64,pub projection_ms:f64,pub model_ms:f64,pub outputs_ms:f64,pub total_ms:f64}

/// A run's fold: what the screen left out, the far shell's rings (the
/// responses of the rings beyond the second order, the rings kept, the ring
/// targets built) and the time of the far shell's landings, of the
/// responses (the landings included) and of the ring targets.
#[derive(Clone,Debug,Default)]
pub struct FoldReport{pub screened:Screened,pub ring_responses:Vec<f64>,pub rings:usize,pub ring_targets:usize,pub landings_ms:f64,pub responses_ms:f64,pub rings_ms:f64}
impl FoldReport{
    /// What the run's fold leaves out: the screened pieces' bound and the
    /// dropped rings' responses.
    pub fn dropped(&self)->f64{self.screened.spent+self.ring_responses[self.rings..].iter().sum::<f64>()}
}

/// The ladder of a run: its top and bottom (1/us) and the poles of a cycle.
#[derive(Clone,Debug,Default)]
pub struct LadderReport{pub top:f64,pub bottom:f64,pub poles:usize}

pub struct Solution{
    /// The best checkpoint's outputs; `converged` when it was accepted, else
    /// the reason the run stopped.
    pub answer:Answer,pub converged:bool,pub reason:Option<String>,
    /// The request's explicit set, the run's lattice and translations.
    pub explicit:ExplicitSet,pub lattice:LatticeSets,pub translations:Vec<Charge>,
    /// The solves, and every checkpoint reported.
    pub columns:Vec<Column>,pub checkpoints:Vec<Checkpoint>,
    /// The model's dimension, the solves' work.
    pub dimension:usize,pub work:InnerWork,
    pub fold:FoldReport,pub ladder:LadderReport,pub timing:Timing,
}

/// Euclidean projection of real values onto the probability simplex.
pub fn simplex(values:&[f64])->Vec<f64>{
    let mut sorted=values.to_vec();sorted.sort_by(|a,b|b.total_cmp(a));
    let(mut sum,mut threshold)=(0.,0.);
    for(index,&value)in sorted.iter().enumerate(){sum+=value;let candidate=(sum-1.)/(index+1)as f64;if value>candidate{threshold=candidate;}else{break;}}
    values.iter().map(|x|(x-threshold).max(0.)).collect()
}

/// The slowest relaxation of a Galerkin model: its slowest decaying mode of
/// population weight at least `weight` (`Model::slow_candidates`) with the
/// residual |A v - theta v| / |v| of v = W s (the image from the lattice).
fn slowest(model:&Model,space:&Space,vectors:&Vectors,weight:f64,lattice:&mut Lattice<'_>)->Result<Option<SlowRate>,String>{
    let Some(&(j,rate))=model.slow_candidates(weight).first()else{return Ok(None)};
    let r=model.r;let y=&model.vectors[r*j..r*(j+1)];
    let basis:Vec<&[C]>=space.basis.iter().map(|v|v.as_slice()).collect();
    let w=vectors.combinations(&basis,&[y.iter().map(|z|z.re).collect(),y.iter().map(|z|z.im).collect()])?;
    let aw=w.iter().map(|x|{let mut y=vec![ZERO;x.len()];lattice.apply(x,&mut y).map(|_|y)}).collect::<Result<Vec<_>,_>>()?;
    // A v - theta v, real and imaginary parts.
    let theta=model.theta[j];let(wr,wi,ar,ai)=(&w[0],&w[1],&aw[0],&aw[1]);
    let re:Vec<C>=(0..wr.len()).map(|e|C{re:ar[e].re-theta.re*wr[e].re+theta.im*wi[e].re,im:ar[e].im-theta.re*wr[e].im+theta.im*wi[e].im}).collect();
    let im:Vec<C>=(0..wr.len()).map(|e|C{re:ai[e].re-theta.re*wi[e].re-theta.im*wr[e].re,im:ai[e].im-theta.re*wi[e].im-theta.im*wr[e].im}).collect();
    let residual=(vectors.dot(&re,&re)+vectors.dot(&im,&im)).max(0.).sqrt()/(vectors.dot(wr,wr)+vectors.dot(wi,wi)).max(f64::MIN_POSITIVE).sqrt();
    Ok(Some(SlowRate{rate,residual}))
}

/// The responses of the far shell's rings beyond the second order: per
/// source E_j of x0 (`Duhamel::sources`, weighed by |c_j|) the far shell's
/// landing on the lattice (`FoldLibrary::far_shell_sources`), with what it
/// leaves out (trace norm) times T within `tail` and its far blocks within
/// `bytes`; per ring its Duhamel response summed over the landing modes; and
/// the landings' time.
fn ring_responses(prepared:&LatticePrepared,modes:&[Charge],nu:&[f64],duhamel:&Duhamel<'_>,horizon:f64,tail:f64,bytes:f64)->Result<(Vec<f64>,f64),String>{
    if prepared.plan.far().is_empty()||duhamel.count()==0{return Ok((Vec::new(),0.));}
    let d=prepared.generator.n*prepared.generator.n;let nu_of=|q:&[i32]|prepared.nu(q);
    let centre=modes.iter().position(|q|*q==prepared.generator.zero).ok_or("Lattice modes must contain the zero charge")?;
    // Per ring and landing mode the images of every source ([j * d + e]).
    let sources:Vec<(usize,&[C],f64)>=duhamel.sources().zip(duhamel.weights()).map(|(e,w)|(centre,e,w)).collect();let count=sources.len();
    let clock=Instant::now();
    let(landings,_,_)=prepared.fold.far_shell_sources(&prepared.generator,&prepared.plan,&nu_of,modes,&sources,tail/horizon,bytes)?;
    let landings_ms=clock.elapsed().as_secs_f64()*1e3;
    let responses=landings.iter().skip(1).map(|ring|{let mut per:BTreeMap<usize,Vec<C>>=BTreeMap::new();
        for(&(jj,q),block)in ring{per.entry(q).or_insert_with(||vec![ZERO;count*d])[jj*d..(jj+1)*d].copy_from_slice(block);}
        per.iter().map(|(&q,v)|duhamel.response(nu[q],1.,v)).sum()}).collect::<Result<Vec<f64>,String>>()?;
    Ok((responses,landings_ms))
}

/// Every output of a request from rho0 (row-major n x n); `observer` sees
/// every checkpoint with its outputs as it is made.
pub fn solve(prepared:&LatticePrepared,initial:&[C],request:&Request<'_>,mut observer:Option<&mut dyn FnMut(&Checkpoint,&Answer)>)->Result<Solution,String>{
    let started=Instant::now();
    let times=request.times;
    if times.is_empty()&&request.horizons.is_empty(){return Err("A request asks for a trajectory or a horizon".into());}
    if !times.is_empty(){
        if times.len()<2||times[0]!=0.{return Err("A trajectory grid starts at zero with at least two times".into());}
        let(step,end)=(times[1],times[times.len()-1]);
        if !(end>0.)||times.iter().enumerate().any(|(i,&t)|(t-step*i as f64).abs()>1e-12*end.max(1.)){return Err("A trajectory grid must be uniform".into());}
    }
    if request.horizons.iter().any(|&t|!(t.is_finite()&&t>0.)){return Err("A horizon must be a finite positive time".into());}
    if times.len()>MOST_SAMPLES{return Err(format!("A trajectory has at most {MOST_SAMPLES} samples (requested {})",times.len()));}
    let longest=times.last().copied().into_iter().chain(request.horizons.iter().copied()).fold(0_f64,f64::max);
    if longest>LONGEST_TIME_US{return Err(format!("An output time is at most {LONGEST_TIME_US:e} us (10 s; requested {longest:e} us)"));}
    let n=prepared.generator.n;
    if initial.len()!=n*n{return Err("Initial density dimension differs".into());}
    let tolerance=POPULATION_TOLERANCE;let share=A_PRIORI_SHARE*tolerance;
    let mut timing=Timing::default();
    // 1. The lattice.
    let clock=Instant::now();let eigen=prepared.eigenbasis()?;timing.eigenbasis_ms=clock.elapsed().as_secs_f64()*1e3;
    let clock=Instant::now();
    let explicit=request.explicit.get_or_init(||charge_set::explicit_set(prepared,initial,tolerance)).clone()?;
    timing.explicit_set_ms=clock.elapsed().as_secs_f64()*1e3;
    let budget=MEMORY_SHARE*crate::memory::physical_bytes()?;
    let sets=explicit.sets.clone();let modes=sets.modes();
    let planned=planned_bytes(request.runs,modes.len());
    if planned>budget{return Err(format!("The lattice of this request has {} modes and would need about {:.1} GB, above the {:.0} GB memory budget: not solved",modes.len(),planned/1e9,budget/1e9));}
    // 2. The far shell's rings: followed until the tail is within a quarter
    // of the share, the outermost dropped and the others' far modes pruned
    // within another quarter.
    let clock=Instant::now();
    let nu:Vec<f64>=modes.iter().map(|q|prepared.nu(q)).collect::<Result<_,_>>()?;
    let duhamel=Duhamel::new(eigen,initial,&fold::sample_times(times,request.horizons))?;
    let(responses,landings_ms)=ring_responses(prepared,&modes,&nu,&duhamel,longest,0.25*share,(budget-planned)/request.runs.max(1)as f64)?;
    let mut depth=responses.len();
    while depth>0&&responses[depth-1..].iter().sum::<f64>()<=0.25*share{depth-=1;}
    let mut report=FoldReport{ring_responses:responses,rings:depth,landings_ms,responses_ms:clock.elapsed().as_secs_f64()*1e3,..FoldReport::default()};
    let clock=Instant::now();
    let rings:Vec<StoredFold>=if depth==0{Vec::new()}else{
        // The ring targets land on the lattice's edges whose charges the plan
        // folds (and the diagonal), K^(0); what the dropped rings left of the
        // quarter is shared by the kept rings' pruning.
        let known:BTreeSet<Charge>=prepared.plan.targets.iter().chain(&prepared.plan.omitted).map(|t|t.charge.clone()).chain(std::iter::once(prepared.generator.zero.clone())).collect();
        let mut offsets=BTreeSet::new();for p in &modes{for q in &modes{let t=charge_minus(q,p)?;if known.contains(&t){offsets.insert(t);}}}
        let spare=(0.25*share-report.ring_responses[depth..].iter().sum::<f64>())/depth as f64;
        let prune:Vec<f64>=report.ring_responses[..depth].iter().map(|&a|if a>0.{spare/a}else{0.}).collect();
        prepared.fold.ring_targets(&prepared.generator,&prepared.plan,depth,1,&prune,&offsets)?};
    report.ring_targets=rings.len();report.rings_ms=clock.elapsed().as_secs_f64()*1e3;
    timing.fold_ms=report.responses_ms+report.rings_ms;
    // The screen, within the other half of the share.
    let nu_max=nu.iter().map(|w|w.abs()).fold(0.,f64::max);
    let(fold,screened)=RunFold::screened(&prepared.fold,&rings,&prepared.generator.zero,nu_max,longest,0.5*share);
    report.screened=screened;
    // 3. The space.
    let clock=Instant::now();
    let mut lattice=Lattice::new(prepared,fold,sets.clone())?;
    let column=(std::mem::size_of::<C>()*lattice.operator().dimension())as f64;
    let capacity=((budget-planned)/request.runs.max(1)as f64/(2.*column)).max(0.)as usize;
    let mut space=Space::new(&mut lattice,initial,capacity)?;
    for s in charge_set::gated_steps(prepared,tolerance){if lattice.operator().index.contains_key(&s){space.activate(&lattice,&s)?;}}
    timing.setup_ms=clock.elapsed().as_secs_f64()*1e3;
    // Below this rate a mode is conserved over the run.
    let kernel=share/longest;
    space.solve(&mut lattice,Pole::Zero,kernel)?;
    // The ladder: from the fastest scale the outputs need down to 1 / T. That
    // is RESOLUTION / t1 at the earliest output time t1 (the grid's step, or
    // the shortest horizon), or for a trajectory, whose readout sees every
    // mode's rotation, the explicit set's fastest frequency when it is
    // faster: its modes carry x0's amplitude above the a-priori share (the
    // ring's stay below it, `charge_set`). A horizon reads the centre alone.
    // In a fixed cycle: sorted by sigma, the bit-reversed order over the next
    // power of two (the indices beyond the ladder left out), so that every
    // stretch of the cycle spreads over the whole ladder.
    let first=if times.len()>1{times[1]}else{request.horizons.iter().copied().fold(f64::INFINITY,f64::min)};
    let rotation=if times.is_empty(){0.}else{explicit.sets.explicit.iter().map(|q|prepared.nu(q).map(f64::abs)).collect::<Result<Vec<_>,_>>()?.into_iter().fold(0.,f64::max)};
    let ladder=LadderReport{top:(RESOLUTION/first).max(rotation).max(1./longest),bottom:1./longest,poles:0};
    let cycle:Vec<Pole>={
        let mut poles=space::ladder(ladder.top,ladder.bottom);
        poles.sort_by(|a,b|b.sigma().total_cmp(&a.sigma()));
        let m=poles.len();let p2=m.next_power_of_two();let width=p2.trailing_zeros();
        (0..p2).map(|i|if width==0{0}else{i.reverse_bits()>>(usize::BITS-width)}).filter(|&i|i<m).map(|i|poles[i]).collect()};
    let ladder=LadderReport{poles:cycle.len(),..ladder};
    // 4. Checkpoints.
    let mut galerkin=Galerkin::new(&lattice)?;
    let vectors=Vectors::new(lattice.operator())?;
    let mut checkpoints:Vec<Checkpoint>=Vec::new();
    // The previous checkpoint's outputs and basis size, its largest change.
    let mut previous:Option<(Vec<Vec<f64>>,Vec<Vec<f64>>,usize)>=None;let mut last_change:Option<f64>=None;
    let mut best:Option<Answer>=None;let reason;
    loop{
        let clock=Instant::now();
        galerkin.ingest(&space.basis,&mut lattice)?;
        let model_clock=Instant::now();
        let model=galerkin.model(&prepared.lapack,&space.initial,longest)?;timing.model_ms+=model_clock.elapsed().as_secs_f64()*1e3;
        let outputs_clock=Instant::now();
        let mut trajectory=model.trajectory(times);
        if let Some(first)=trajectory.first_mut(){*first=(0..n).map(|i|initial[i*n+i].re).collect();}
        let horizons:Vec<Vec<f64>>=request.horizons.iter().map(|&t|model.horizon(t)).collect();
        // The change since the previous checkpoint (unknown at the first),
        // and whether the cycle added directions.
        let grew=previous.as_ref().is_none_or(|p|space.basis.len()>p.2);
        let estimate=match &previous{
            Some((t0,h0,_))=>Estimate{trajectory:trajectory.iter().zip(t0).map(|(a,b)|accuracy::metric(a,b,request.groups)).collect(),
                horizons:horizons.iter().zip(h0).map(|(a,b)|accuracy::metric(a,b,request.groups)).collect()},
            None=>Estimate{trajectory:vec![1.;trajectory.len()],horizons:vec![1.;horizons.len()]}};
        let projected:Vec<Vec<f64>>=trajectory.iter().map(|p|simplex(p)).collect();
        let sample_errors:Vec<f64>=estimate.trajectory.iter().zip(trajectory.iter().zip(&projected)).map(|(e,(a,b))|e+accuracy::metric(a,b,request.groups)).collect();
        let(s,r)=(&model.vectors,model.r);
        let basis:Vec<&[C]>=space.basis.iter().map(|v|v.as_slice()).collect();
        let coordinates:Vec<Vec<f64>>=request.horizons.iter().map(|&t|{let w=model.weights(t);
            (0..r).map(|i|(0..r).fold(0.,|acc,j|acc+mul(s[i+r*j],w[j]).re)).collect()}).collect();
        let horizon_fields=vectors.combinations(&basis,&coordinates)?;
        let slow=slowest(&model,&space,&vectors,share,&mut lattice)?;
        timing.outputs_ms+=outputs_clock.elapsed().as_secs_f64()*1e3;
        let mut answer=Answer{solves:space.solves(),modes:modes.clone(),trajectory:projected,sample_errors,horizons:horizons.clone(),horizon_fields,
            horizon_errors:estimate.horizons.clone(),estimate:estimate.clone(),slowest:slow,accepted:false};
        let largest=answer.largest();
        answer.accepted=previous.is_some()&&grew&&largest<=tolerance;
        timing.projection_ms=galerkin.milliseconds;
        let point=Checkpoint{solves:space.solves(),columns:space.basis.len(),estimate,largest,accepted:answer.accepted,growing:model.growing,slowest:slow,
            trajectory:trajectory.clone(),horizons:horizons.clone(),inner:lattice.work().clone(),timing:timing.clone(),milliseconds:clock.elapsed().as_secs_f64()*1e3};
        if let Some(o)=observer.as_mut(){o(&point,&answer);}
        checkpoints.push(point);
        // The best checkpoint with an estimate: one within the tolerance over
        // any other, the latest of those, else the least largest bound.
        if previous.is_some()&&best.as_ref().is_none_or(|b|largest<=tolerance||(b.largest()>tolerance&&largest<b.largest())){best=Some(answer.clone());}
        if answer.accepted{reason=None;break;}
        // 5. Stopping.
        let why=if previous.is_some()&&!grew{Some(format!("the last cycle of the ladder added no direction after {} solves",space.solves()))}
            else if last_change.is_some_and(|c|largest>=c){Some(format!("the largest change did not shrink over a cycle of the ladder after {} solves",space.solves()))}
            else if space.full(){Some(format!("the basis memory was spent after {} solves",space.solves()))}else{None};
        if let Some(why)=why{
            let b=best.as_ref().ok_or("A run stopped before its second checkpoint")?;
            let over:Vec<usize>=(0..b.sample_errors.len()).filter(|&i|b.sample_errors[i]>tolerance).collect();
            let samples=match(over.first(),over.last()){(Some(&a),Some(&z))=>format!("; {} of {} samples above it, t = {:.4} to {:.4} us",over.len(),b.sample_errors.len(),times[a],times[z]),_=>String::new()};
            let means=b.horizon_errors.iter().filter(|&&e|e>tolerance).count();
            let means=if means>0{format!("; {means} mean output(s) above it")}else{String::new()};
            reason=Some(format!("Stopped because {why}: the best checkpoint ({} solves) has an error estimate of {:.2e} against the tolerance {tolerance:e}{samples}{means}",b.solves,b.largest()));
            break;
        }
        if previous.is_some(){last_change=Some(largest);}
        previous=Some((trajectory,horizons,space.basis.len()));
        // Before the first cycle, x0's components on the centre generator's
        // eigenvectors that the first output still sees.
        if checkpoints.len()==1{
            let(centre,size)={let op=lattice.operator();(op.centre,op.dimension())};
            let fields:Vec<Vec<C>>=eigen.components(initial,share,first)?.into_iter().map(|b|{let mut x=vec![ZERO;size];x[centre*n*n..(centre+1)*n*n].copy_from_slice(&b);x}).collect();
            space.add(&lattice,fields)?;
        }
        // The next cycle of the ladder.
        for &pole in &cycle{space.solve(&mut lattice,pole,kernel)?;if space.full(){break;}}
    }
    let answer=best.ok_or("A run has its checkpoint")?;
    let work=lattice.work().clone();
    timing.solves_ms=work.milliseconds;timing.projection_ms=galerkin.milliseconds;timing.total_ms=started.elapsed().as_secs_f64()*1e3;
    Ok(Solution{converged:answer.accepted,answer,reason,explicit,lattice:sets,translations:space.translations(),
        columns:space.columns.clone(),dimension:galerkin.columns(),work,fold:report,ladder,timing,checkpoints})
}

#[cfg(test)]
mod tests{
    use super::*;
    use super::super::tests::{fixture,prepared,dense_steady};
    use super::super::operator::LatticeOperator;
    #[test]
    fn lattice_pipeline_follows_the_exact_lattice_dynamics(){
        // The fixture from |0><0|: the trajectory (2 us) and the state at a
        // horizon against the same lattice propagated densely (RK4, real
        // form), and the bordered steady column against the lattice's dense
        // null vector. The far drives at 4% of the fixture's.
        let mut f=fixture();for v in f.far.iter_mut().chain(f.far2.iter_mut()){*v*=0.04;}let p=prepared(&f);let n=4;let d=16;let groups=vec![vec![0,1],vec![2,3]];
        let mut initial=vec![ZERO;d];initial[0]=ONE;
        let times:Vec<f64>=(0..21).map(|i|0.1*i as f64).collect();let horizons=[3.];
        let mut seen=0;let mut observe=|_:&Checkpoint,_:&Answer|{seen+=1;};
        let s=solve(&p,&initial,&Request{times:&times,horizons:&horizons,groups:&groups,explicit:&std::sync::OnceLock::new(),runs:1},Some(&mut observe)).unwrap();
        assert!(s.converged&&s.reason.is_none()&&seen==s.checkpoints.len(),"{:?}",s.reason);
        let a=&s.answer;
        let op=LatticeOperator::new(&p,RunFold::whole(&p.fold),&a.modes).unwrap();let dense=op.dense();let size=op.dimension();
        let rate=|x:&[C]|->Vec<C>{(0..size).map(|r|{let mut v=ZERO;for c in 0..size{mul_add(&mut v,dense[r*size+c],x[c]);}v}).collect()};
        let mut x=vec![ZERO;size];x[op.centre*d]=ONE;
        let readout=|x:&[C],t:f64|->Vec<f64>{(0..n).map(|i|(0..a.modes.len()).map(|q|mul(C{re:(op.nu[q]*t).cos(),im:(op.nu[q]*t).sin()},x[q*d+i*n+i]).re).sum()).collect()};
        let steps=400;let h=0.1/steps as f64;let mut t=0.;
        let axpy=|x:&[C],k:&[C],s:f64|->Vec<C>{x.iter().zip(k).map(|(a,b)|C{re:a.re+s*b.re,im:a.im+s*b.im}).collect()};
        let step=|x:&mut Vec<C>|{let k1=rate(x);let k2=rate(&axpy(x,&k1,h/2.));let k3=rate(&axpy(x,&k2,h/2.));let k4=rate(&axpy(x,&k3,h));
            for e in 0..size{x[e].re+=h/6.*(k1[e].re+2.*k2[e].re+2.*k3[e].re+k4[e].re);x[e].im+=h/6.*(k1[e].im+2.*k2[e].im+2.*k3[e].im+k4[e].im);}};
        for(i,row)in a.trajectory.iter().enumerate(){
            if i>0{for _ in 0..steps{step(&mut x);t+=h;}}
            // Every sample within the tolerance of the exact lattice dynamics,
            // and its bound within the tolerance.
            let error=accuracy::metric(&readout(&x,t),row,&groups);
            assert!(error<=POPULATION_TOLERANCE&&a.sample_errors[i]<=POPULATION_TOLERANCE,"sample {i}: {error:e} against {:e}",a.sample_errors[i]);
        }
        for _ in 0..10*steps{step(&mut x);}
        let centre:Vec<f64>=(0..n).map(|i|x[op.centre*d+i*n+i].re).collect();
        let horizon=accuracy::metric(&centre,&a.horizons[0],&groups);assert!(horizon<=POPULATION_TOLERANCE,"horizon {horizon:e}");
        let exact=dense_steady(&p,&dense,op.centre,n);
        // The bordered column at a tight tolerance is the lattice's null
        // vector.
        let mut lattice=Lattice::new(&p,RunFold::whole(&p.fold),s.lattice.clone()).unwrap();let mut u=vec![ZERO;size];for i in 0..n{u[op.centre*d+i*n+i]=C{re:0.25,im:0.};}
        let(w,_,_)=lattice.solve(Pole::Zero,&u,1e-13).unwrap();let trace:f64=(0..n).map(|i|w[op.centre*d+i*n+i].re).sum();
        let column:Vec<f64>=(0..n).map(|i|w[op.centre*d+i*n+i].re/trace).collect();
        let tight=accuracy::metric(&exact,&column,&groups);assert!(tight<1e-10,"bordered column {tight:e}");
        assert!(a.slowest.is_some());
        // Projected rows: probabilities, the first one rho0's.
        assert!(a.trajectory.iter().all(|r|r.iter().all(|&v|v>=0.)&&(r.iter().sum::<f64>()-1.).abs()<1e-12)&&a.trajectory[0][0]==1.);
    }
    #[test]
    fn a_strong_far_drive_stays_explicit_with_its_micromotion(){
        // H(t) = 200 cos(1000 t) sigma_x (rad/us), no decay, rho0 = |0><0|:
        // p_e(t) = sin^2(0.2 sin 1000 t), 0.0394695 at t = pi/2000 us. The
        // drive's mixing 0.2 is far above the fold's (m^2 = 0.04): it is a
        // lattice step, and the micromotion is in the trajectory.
        use crate::preparation::{DriveView,MatrixView,ModelView,PhaseBasis,FrequencyParts};
        let zero=vec![0.;4];let h=vec![0.;4];let cos=vec![0.,200.,200.,0.];
        let drives=[DriveView{omega:1000.,cosine:MatrixView{real:&cos,imag:&zero},sine:MatrixView{real:&zero,imag:&zero}}];
        let model=ModelView{dimension:2,hamiltonian:MatrixView{real:&h,imag:&zero},collapse:&[],drives:&drives,ground:&[true,false]};
        let p=LatticePrepared::prepare(&model,PhaseBasis::from_parts(&[FrequencyParts{hi:1000.,lo:0.}],&[vec![1]]).unwrap()).unwrap();
        assert!(p.plan.far().is_empty(),"the strong drive is not folded");
        let mut initial=vec![ZERO;4];initial[0]=ONE;
        let end=std::f64::consts::PI/2000.;let times:Vec<f64>=(0..9).map(|i|end*i as f64/8.).collect();
        let s=solve(&p,&initial,&Request{times:&times,horizons:&[],groups:&[],explicit:&std::sync::OnceLock::new(),runs:1},None).unwrap();
        assert!(s.converged,"{:?}",s.reason);
        for(i,&t)in times.iter().enumerate(){let exact=(0.2*(1000.*t).sin()).sin().powi(2);
            assert!((s.answer.trajectory[i][1]-exact).abs()<=POPULATION_TOLERANCE,"t {t:e}: {} against {exact}",s.answer.trajectory[i][1]);}
    }
}
