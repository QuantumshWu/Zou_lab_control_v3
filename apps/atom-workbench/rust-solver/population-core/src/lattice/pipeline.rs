//! The population pipeline: every output of a request is a matrix function
//! of one lattice operator applied to x0, read from one reduced model.
//!
//! 1. The explicit set P and its ring Q (`charge_set`); the lattice P u Q
//!    and its solves at the poles (`poles`). A request that propagates (a
//!    trajectory or a horizon) builds its frame (`frame`: the reduced
//!    model's metric, the estimate's fast part, the slowest relaxation's
//!    classes) on its own thread meanwhile.
//! 2. One rational Krylov space from x0 (`space`): the bordered steady
//!    column first, the translations through the gate active from the start,
//!    then the ladder's poles by their expected gain, re-ranked at every
//!    checkpoint: sigma |p_err(xi)| (the error transform) times the share of
//!    its own transform a pole's last solve removed (a pole whose solve did
//!    not reduce it loses its turn; one never solved keeps the whole), per
//!    real solve (a complex pole costs two); a pole that adds no direction
//!    leaves the ladder. While
//!    the slowest relaxation needs resolving, the real poles near its rate
//!    gain priority: the best population gain (at least the tolerance)
//!    times its relative need, by their log distance, and the share of the
//!    need their last solve removed.
//! 3. Every CHECKPOINT solves (a request that propagates nothing: also
//!    right after the steady column): the Galerkin model in the frame's
//!    metric (`reduced`; a request for the steady state alone takes M = I,
//!    a speed case: it skips about 1 s of frame build, and its steady
//!    column makes the kernel independent of the metric), its outputs (trajectory on the grid, horizons, steady
//!    state, slowest relaxation, conserved count) and the error estimate
//!    (`estimator`). An output's bound is ACCEPTANCE_FACTOR x its estimate,
//!    plus, on a trajectory row, the change of its projection onto the
//!    probability simplex (computed exactly, not estimated). A checkpoint
//!    is accepted when every bound is within the tolerance and the slowest
//!    relaxation is resolved where a horizon needs it (`slow_pairs`, and
//!    every class of x0 a horizon needs: `missing_classes`).
//! 4. Stopping (deterministic, no wall time): accepted; or the largest
//!    bound, above the tolerance, did not improve by PLATEAU_GAIN over the
//!    last PLATEAU checkpoints (likewise the slowest relaxation's need once
//!    every bound is within the tolerance); or the solve budget
//!    (MAXIMUM_SOLVES) or the basis budget (BASIS_BYTES) is spent. The run
//!    returns its best checkpoint (`Answer`), converged or partial with the
//!    reason; every checkpoint is reported to the observer as it is made.
//!
//! The conserved count is the reduced model's kernel (the conserved
//! quantities x0 reaches), and the folded centre block's when the lattice is
//! that one block (a static model: its kernel is known exactly).
use super::*;
use super::charge_set::{self,ExplicitSet};
use super::estimate_api::{ACCEPTANCE_FACTOR,ErrorEstimate,EstimateSetup,Outputs,ReducedView};
use super::ledger::{self,EXPANSION_LIMIT,POPULATION_TOLERANCE,RemainderLedger};
use super::reduced::{Model,Projection,SLOW_ACCURACY,SlowRate};
use super::poles::{InnerWork,Lattice,Pole};
use super::space::{self,Column,Space};
use super::frame::{SwFrame,SwMetric};
use super::estimator::{self,UnifiedEstimate};
use super::eigenbasis::BlockEigenbasis;

/// Solves between checkpoints; the ladder is re-ranked at every checkpoint
/// (from its reduced model) and whenever its queue runs out. Measured at
/// this spacing: the estimates up to acceptance cost 19-35% of the solves on
/// three to five lasers; acceptance is seen at most three solves late.
const CHECKPOINT:usize=4;
/// Most solves of a request, and the basis memory (basis and images).
/// MAXIMUM_SOLVES is a deliberate margin: twice the 24 solves the slowest
/// accepted research case takes (16-24 at 20 us).
const MAXIMUM_SOLVES:usize=48;
const BASIS_BYTES:usize=1<<30;
/// The plateau: the largest bound (or the slowest relaxation's need) did
/// not improve by PLATEAU_GAIN over the last PLATEAU checkpoints. Measured
/// before the rule: four lasers + tweezer at 10 us / 2000 samples held the
/// bound at 5.0-5.7e-3 over 20 solves, five lasers at about 1e-2 over 24.
/// Over two checkpoints four lasers without the trap at 20 us stopped at 16
/// solves (bound 1.2e-2), accepted at 20 over three.
const PLATEAU:usize=3;
const PLATEAU_GAIN:f64=2.;
/// The slowest relaxation: its candidates carry population
/// weight of at least CANDIDATE_WEIGHT x tolerance; a candidate must be
/// resolved when its weight at a horizon T, weight x e^{-rate T}, is at
/// least REQUIRED_CONTRIBUTION x tolerance; it is matched when every class
/// of x0 (weight at least CANDIDATE_WEIGHT x tolerance) slower than its rate
/// / CLASS_GAP has a candidate within CLASS_MATCH of its rate. A class of
/// x0 itself must be resolved (`missing_classes`) when its own weight at a
/// horizon is at least REQUIRED_CONTRIBUTION x tolerance.
/// CANDIDATE_WEIGHT, measured: the spurious slow pairs of a four-laser case
/// carried 1.2e-5 and 1.5e-6, the physical one 4e-2; 1e-2 x tolerance =
/// 5e-5 lies 2-3 orders from both.
const CANDIDATE_WEIGHT:f64=1e-2;
/// A deliberate margin: a pair or class contributing less than a tenth of
/// the tolerance at every horizon cannot move an output beyond it.
const REQUIRED_CONTRIBUTION:f64=0.1;
/// Derived: 1 + CLASS_MATCH, so a class slower than rate / CLASS_GAP is
/// outside the candidate's own match and needs another candidate.
const CLASS_GAP:f64=1.4;
/// A deliberate margin over the measured offset between a class (the
/// frame's second-order block) and its Ritz pair: 18% on a four-laser case
/// (class 0.0418 /us, pair 0.0344 /us).
const CLASS_MATCH:f64=0.4;
/// Grid samples assumed for the ladder's top when a request has no grid,
/// and the longest time of a request for the limit alone (us).
/// LADDER_SAMPLES is a deliberate margin: as if the horizon were sampled
/// like the app's trajectories (161-200 samples on the research scenes).
const LADDER_SAMPLES:f64=200.;
/// A deliberate margin: the app's default horizon, 1 s; the steady column
/// and the slow points (down to `space::SLOW_FLOOR`) serve slower rates.
const LIMIT_REACH:f64=1e6;
/// A decaying class of the frame's block carries none of x0's trace; a
/// class whose trace is at least this share of its weight is the block's
/// kernel. Measured: decaying classes at most 5e-3 of their weight, the
/// kernel 1.6 (trace 1.00, weight 0.63): a margin of 100x and 3x.
const KERNEL_TRACE_SHARE:f64=0.5;
/// Stack of the frame's thread (LAPACK on it).
const FRAME_STACK:usize=1<<26;

/// What a request asks for.
pub struct Request<'a>{
    /// The uniform grid from 0 (empty: no trajectory).
    pub times:&'a[f64],
    /// Horizons T (us) of the laboratory time average.
    pub horizons:&'a[f64],
    /// Whether the steady state (the t -> infinity mean) is reported.
    pub steady:bool,
    pub groups:&'a[Vec<usize>],
    /// The explicit set shared by the runs of one request from the same rho0
    /// (the first run to need it builds it), else built by the run.
    pub explicit:Option<&'a std::sync::OnceLock<Result<ExplicitSet,String>>>,
}

/// One checkpoint, for the diagnostics.
#[derive(Clone,Debug)]
pub struct Checkpoint{
    pub solves:usize,pub columns:usize,pub dimension:usize,
    /// The estimate and the largest bound over every output.
    pub estimate:ErrorEstimate,pub largest:f64,pub accepted:bool,
    pub growing:usize,pub clipped:usize,pub projected:usize,pub conserved:usize,pub slowest:Option<SlowRate>,
    /// Raw outputs: trajectory rows (row 0 rho0's), horizon rows, steady row.
    pub trajectory:Vec<Vec<f64>>,pub horizons:Vec<Vec<f64>>,pub steady:Vec<f64>,
    /// The solves so far (inner steps, block operations, ms) and the run's
    /// timing so far.
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
    /// rho0's), the bound of each (ACCEPTANCE_FACTOR x the estimate plus the
    /// projection's change), and the largest projection change.
    pub trajectory:Vec<Vec<f64>>,pub sample_errors:Vec<f64>,pub projection:f64,
    /// Horizon and steady populations (laboratory time averages, raw), their
    /// fields on `modes` (Hermitian-symmetric) and bounds.
    pub horizons:Vec<Vec<f64>>,pub steady:Vec<f64>,
    pub horizon_fields:Vec<Vec<C>>,pub steady_field:Vec<C>,
    pub horizon_errors:Vec<f64>,pub steady_error:Option<f64>,
    /// |A x| / |x| of the steady field (rad/us).
    pub steady_residual:f64,
    pub estimate:ErrorEstimate,pub conserved:usize,pub slowest:Option<SlowRate>,
    /// Every bound within the tolerance and the slowest relaxation resolved
    /// where a horizon needs it.
    pub accepted:bool,
}
impl Answer{
    /// The largest bound over every output.
    pub fn largest(&self)->f64{self.sample_errors.iter().chain(&self.horizon_errors).chain(self.steady_error.iter()).fold(0_f64,|a,&b|a.max(b))}
}

#[derive(Clone,Debug,Default)]
pub struct Timing{pub pairs_ms:f64,pub observer_ms:f64,pub eigenbasis_ms:f64,pub explicit_set_ms:f64,pub setup_ms:f64,pub estimator_build_ms:f64,pub outputs_ms:f64,pub frame_ms:[f64;3],pub frame_wait_ms:f64,pub solves_ms:f64,pub images_ms:f64,pub projection_ms:f64,pub model_ms:f64,pub ordering_ms:f64,pub estimate_ms:f64,pub total_ms:f64}

pub struct Solution{
    /// The best checkpoint's outputs; `converged` when it was accepted, else
    /// the reason the run stopped.
    pub answer:Answer,pub converged:bool,pub reason:Option<String>,
    pub explicit:ExplicitSet,pub translations:Vec<Charge>,
    pub columns:Vec<Column>,pub checkpoints:Vec<Checkpoint>,
    pub dimension:usize,pub work:InnerWork,pub images:usize,
    pub ledger:RemainderLedger,pub timing:Timing,
}

/// Euclidean projection of real values onto the probability simplex.
pub fn simplex(values:&[f64])->Vec<f64>{
    let mut sorted=values.to_vec();sorted.sort_by(|a,b|b.total_cmp(a));
    let(mut sum,mut threshold)=(0.,0.);
    for(index,&value)in sorted.iter().enumerate(){sum+=value;let candidate=(sum-1.)/(index+1)as f64;if value>candidate{threshold=candidate;}else{break;}}
    values.iter().map(|x|(x-threshold).max(0.)).collect()
}

/// The decaying classes of x0: the eigenvalues of the frame's second-order
/// block (L0' + sum_s K_s, the Raman rates included) on which x0 carries
/// population weight |c_j| max_i |(E V)_ij| (c = V^-1 E^-1 x0) of at least
/// CANDIDATE_WEIGHT x tolerance; (rate, weight), slowest first. The block's
/// kernel is not a class: the block annihilates the trace, so a decaying class
/// carries none of x0's trace |c_j sum_i (E V)_ij| (measured: at most 5e-3
/// of its weight) and a class carrying at least KERNEL_TRACE_SHARE of its
/// weight is the kernel, its value zero within the eigensystem's error over the horizon
/// (three lasers at T = 1 s: 1.4e-8 /us, weight 0.63, trace 1.00; taken as
/// a class, every candidate was unmatched and a horizon needed it).
fn class_rates(frame:&SwFrame<'_>,eigen:&BlockEigenbasis,lapack:&Lapack,initial:&[C],tolerance:f64)->Result<Vec<(f64,f64)>,String>{
    let class=frame.class()?;let d=initial.len();let n=class.readout.len()/d.max(1);
    let coordinates=eigen.coefficients(&[initial])?;
    let mut c=vec![ZERO;d];lapack.complex_gemm().multiply(d,1,d,ONE,&class.inverse,d,&coordinates,d,ZERO,&mut c,d)?;
    let mut classes:Vec<(f64,f64)>=(0..d).filter_map(|j|{
        let column=&class.readout[n*j..n*(j+1)];
        let weight=magnitude(c[j])*column.iter().map(|z|magnitude(*z)).fold(0.,f64::max);
        let trace=magnitude(c[j])*magnitude(column.iter().fold(ZERO,|a,z|C{re:a.re+z.re,im:a.im+z.im}));
        let rate=-class.values[j].re;
        (rate>=super::reduced::KERNEL&&weight>=CANDIDATE_WEIGHT*tolerance&&trace<KERNEL_TRACE_SHARE*weight).then_some((rate,weight))}).collect();
    classes.sort_by(|a,b|a.0.total_cmp(&b.0));Ok(classes)
}

/// The classes of x0 a horizon needs and the space has not resolved,
/// slowest first: a class whose own weight at a horizon T, weight x
/// e^{-rate T}, is at least REQUIRED_CONTRIBUTION x tolerance, without a
/// resolved candidate within CLASS_MATCH of its rate. A candidate's own
/// requirement is taken at its rate, so a class slower than every candidate
/// escaped it, and neither the residual nor the estimate sees a class the
/// space lacks: the default scene (probe + 852 nm tweezer) at T = 100 ms had
/// its classes at 5.6e-7 and 8.7e-7 /us (weights 0.7% and 1.0%) outside a
/// space whose slowest candidate decayed at 5.1e-5 /us, and p(T) was the
/// steady state's, off by 7.8e-3 against a bound of 7e-13 (5e-4 with the
/// horizons' fast part).
fn missing_classes(classes:&[(f64,f64)],pairs:&[(SlowRate,f64)],horizons:&[f64],tolerance:f64)->Vec<f64>{
    classes.iter().filter(|&&(rate,weight)|horizons.iter().any(|&t|weight*(-rate*t).exp()>=REQUIRED_CONTRIBUTION*tolerance)
        &&!pairs.iter().any(|(p,_)|p.resolved()&&(p.rate-rate).abs()<=CLASS_MATCH*rate)).map(|c|c.0).collect()
}

/// The slowest relaxation's candidates of a model (`Model::slow_candidates`),
/// slowest first, each with its population weight and its rate, residual
/// |A W y - theta W y| / |W y| (from the images, which keep its accuracy at
/// slow rates) and whether the classes `classes` slower than its rate /
/// CLASS_GAP are matched (None: not checked): the residual cannot see a
/// slow class the space lacks.
fn slow_pairs(model:&Model,space:&Space,vectors:&super::krylov::Vectors,classes:Option<&[f64]>,tolerance:f64)->Result<Vec<(SlowRate,f64)>,String>{
    let chosen=model.slow_candidates(CANDIDATE_WEIGHT*tolerance);
    if chosen.is_empty(){return Ok(Vec::new());}
    let parts:Vec<Vec<f64>>=chosen.iter().flat_map(|&(j,_)|{let y=model.ritz_coordinates(j);[y.iter().map(|z|z.re).collect(),y.iter().map(|z|z.im).collect()]}).collect();
    let basis:Vec<&[C]>=space.basis.iter().map(|v|v.as_slice()).collect();let images:Vec<&[C]>=space.images.iter().map(|v|v.as_slice()).collect();
    let(w,aw)=(vectors.combinations(&basis,&parts)?,vectors.combinations(&images,&parts)?);
    let found:Vec<(f64,f64,f64)>=chosen.iter().enumerate().map(|(c,&(j,rate))|{
        let theta=model.theta[j];let(wr,wi,ar,ai)=(&w[2*c],&w[2*c+1],&aw[2*c],&aw[2*c+1]);
        // A W y - theta W y, real and imaginary parts.
        let re:Vec<C>=(0..wr.len()).map(|e|C{re:ar[e].re-theta.re*wr[e].re+theta.im*wi[e].re,im:ar[e].im-theta.re*wr[e].im+theta.im*wi[e].im}).collect();
        let im:Vec<C>=(0..wr.len()).map(|e|C{re:ai[e].re-theta.re*wi[e].re-theta.im*wr[e].re,im:ai[e].im-theta.re*wi[e].im-theta.im*wr[e].im}).collect();
        let residual=(vectors.dot(&re,&re)+vectors.dot(&im,&im)).max(0.).sqrt()/(vectors.dot(wr,wr)+vectors.dot(wi,wi)).max(f64::MIN_POSITIVE).sqrt();
        (rate,residual,model.amplitudes[j])}).collect();
    let matched=|rate:f64|classes.map(|classes|classes.iter().filter(|&&class|class<rate/CLASS_GAP).all(|&class|found.iter().any(|f|(f.0-class).abs()<=CLASS_MATCH*class)));
    Ok(found.iter().map(|&(rate,residual,weight)|(SlowRate{rate,residual,matched:matched(rate)},weight)).collect())
}

/// How far a slow pair is from resolved: its residual against SLOW_ACCURACY
/// x its rate (at least one while a slower class is missing), times its
/// rate over the slowest missing class's; and the rate its poles aim at
/// (the missing class's, else its own).
fn need(pair:&SlowRate,classes:&[f64])->(f64,f64){
    let accuracy=pair.residual/(SLOW_ACCURACY*pair.rate);
    match classes.iter().copied().find(|&class|class<pair.rate/CLASS_GAP&&pair.matched==Some(false)){
        Some(class)=>(accuracy.max(1.)*pair.rate/class,class),
        None=>(accuracy,pair.rate)}
}

/// How far a missing class is from resolved: the nearest candidate's
/// residual against SLOW_ACCURACY x its rate (at least one) times its rate's
/// ratio to the class's (at least one); one without a candidate. Its poles
/// aim at the class's rate.
fn class_need(class:f64,pairs:&[(SlowRate,f64)])->(f64,f64){
    let need=pairs.iter().map(|(p,_)|(p.residual/(SLOW_ACCURACY*p.rate)).max(1.)*(p.rate/class).max(class/p.rate)).fold(f64::INFINITY,f64::min);
    (if need.is_finite(){need}else{1.},class)
}

/// Every output of a request from rho0 (row-major n x n); `observer` sees
/// every checkpoint with its outputs as it is made.
pub fn solve(prepared:&LatticePrepared,initial:&[C],request:&Request<'_>,mut observer:Option<&mut dyn FnMut(&Checkpoint,&Answer)>)->Result<Solution,String>{
    let started=Instant::now();let tolerance=POPULATION_TOLERANCE;let n=prepared.generator.n;
    if initial.len()!=n*n{return Err("Initial density dimension differs".into());}
    let times=request.times;
    if !times.is_empty(){
        if times.len()<2||times[0]!=0.{return Err("A trajectory grid starts at zero with at least two times".into());}
        let(step,end)=(times[1],times[times.len()-1]);
        if !(end>0.)||times.iter().enumerate().any(|(i,&t)|(t-step*i as f64).abs()>1e-12*end.max(1.)){return Err("A trajectory grid must be uniform".into());}
    }
    if request.horizons.iter().any(|&t|!(t.is_finite()&&t>0.)){return Err("A horizon must be a finite positive time".into());}
    let longest=times.last().copied().into_iter().chain(request.horizons.iter().copied()).fold(0_f64,f64::max);
    let longest=if longest>0.{longest}else{LIMIT_REACH};
    let t1=if times.len()>1{times[1]}else{longest/LADDER_SAMPLES};
    let propagates=!times.is_empty()||!request.horizons.is_empty();
    let mut timing=Timing::default();
    // 1. The lattice; the frame on its own thread for a request that
    // propagates.
    let clock=Instant::now();let eigen=prepared.eigenbasis()?;timing.eigenbasis_ms=clock.elapsed().as_secs_f64()*1e3;let clock=Instant::now();
    let frame=SwFrame::new(prepared,estimator::longest_time(times,request.horizons));
    std::thread::scope(|scope|->Result<Solution,String>{
    if propagates{std::thread::Builder::new().stack_size(FRAME_STACK).spawn_scoped(scope,||{let _=frame.class();}).map_err(|e|e.to_string())?;}
    let explicit=match request.explicit{Some(shared)=>shared.get_or_init(||charge_set::explicit_set(prepared,initial,tolerance)).clone()?,None=>charge_set::explicit_set(prepared,initial,tolerance)?};timing.explicit_set_ms=clock.elapsed().as_secs_f64()*1e3;
    let clock=Instant::now();
    let mut lattice=Lattice::new(prepared,explicit.sets.clone())?;
    let ledger=lattice.operator().ledger()?;
    if ledger.expansion_parameter>EXPANSION_LIMIT{return Err(format!("Static fold expansion parameter {:e} above {EXPANSION_LIMIT:e}",ledger.expansion_parameter));}
    // 2. The space: translations, the steady column.
    let capacity={let op=lattice.operator();let d=op.n*op.n;BASIS_BYTES/(std::mem::size_of::<C>()*2*d*op.modes.len())};
    let mut space=Space::new(&mut lattice,initial,capacity)?;
    for s in charge_set::gated_steps(prepared,tolerance){if lattice.operator().index.contains_key(&s){space.activate(&mut lattice,&s)?;}}
    timing.setup_ms=clock.elapsed().as_secs_f64()*1e3;
    space.solve(&mut lattice,Pole::Zero)?;
    let modes=lattice.modes.clone();
    let clock=Instant::now();
    let mut estimator=UnifiedEstimate::new(EstimateSetup{prepared,lattice:&explicit.sets,outputs:Outputs{times,horizons:request.horizons,steady:request.steady,groups:request.groups},frame:&frame,ledger:&ledger,initial})?;
    timing.estimator_build_ms=clock.elapsed().as_secs_f64()*1e3;
    // The ladder spans the grid's times when there is one (a horizon is
    // served by the steady column, the slow points and the slowest
    // relaxation's poles): its poles near 1 / T of a 1 s horizon grew
    // spurious near-zero slow Ritz pairs in both metrics (rates 1e-8 to
    // 1e-5, residuals 0.2-3) whose p(1 s) bounds reached 1e1-1e4; without
    // them three, four, five lasers and four without the trap at 20 us were
    // accepted at 16-24 solves (four without the trap partial before) and
    // five lasers at 10 us / 2000 samples reached 7.0e-3 against 2.3e-2.
    let(points,slow_points)=space::ladder(eigen,initial,t1,times.last().copied().unwrap_or(longest),tolerance)?;let mut alive=vec![true;points.len()];
    // The frame's metric (waiting for the frame's thread). A request that
    // propagates nothing (the steady state alone) takes M = I instead, a
    // speed case: it skips about 1 s of frame build (`reduced`).
    let clock=Instant::now();
    let metric=if propagates{Some(SwMetric::new(&frame,eigen,lattice.operator())?)}else{None};
    timing.frame_wait_ms=clock.elapsed().as_secs_f64()*1e3;
    let mut galerkin=Projection::new(lattice.operator())?;
    let lapack=&prepared.lapack;let ops=super::krylov::Vectors::new(lattice.operator(),1)?;
    // A slow pair must be resolved when it contributes REQUIRED_CONTRIBUTION
    // x tolerance at a horizon.
    let required=|pair:&SlowRate,weight:f64|request.horizons.iter().any(|&t|weight*(-pair.rate*t).exp()>=REQUIRED_CONTRIBUTION*tolerance);
    // The classes of x0 (a request that propagates).
    let mut classes:Option<Vec<(f64,f64)>>=None;
    let mut checkpoints:Vec<Checkpoint>=Vec::new();
    // The best checkpoint so far, the latest model, and per checkpoint the
    // progress measure of the stopping rule.
    let mut best:Option<Answer>=None;let mut progress:Vec<f64>=Vec::new();
    let mut reason:Option<String>=None;
    let mut current:Option<(usize,Model)>=None;
    // Per point: its last score, the score it had when last solved, and the
    // share of its transform left after that solve; likewise the slowest
    // relaxation's need when it was last solved and the share left.
    let(mut score,mut solved_at,mut remaining)=(vec![0.;points.len()],vec![None::<f64>;points.len()],vec![0.;points.len()]);
    let(mut slow_at,mut slow_remaining)=(vec![None::<f64>;points.len()],vec![0.;points.len()]);
    // The slowest relaxation while it needs poles: (need, target rate).
    let mut slow_need:Option<(f64,f64)>=None;
    loop{
        let live:Vec<usize>=(0..points.len()).filter(|&i|alive[i]).collect();
        let exhausted=live.is_empty()||space.solves()>=MAXIMUM_SOLVES||space.full();
        if exhausted||space.solves()%CHECKPOINT==0||(!propagates&&space.solves()==1){
            // 3. The checkpoint: model, outputs, estimate, bounds.
            let clock=Instant::now();
            galerkin.ingest(metric.as_ref(),&space.basis,&space.images)?;
            let model=galerkin.model(lapack,space.initial_norm,longest,tolerance)?;timing.model_ms+=clock.elapsed().as_secs_f64()*1e3;
            timing.projection_ms=galerkin.milliseconds;timing.images_ms=space.images_ms;
            let outputs_clock=Instant::now();
            let mut trajectory=model.trajectory(&galerkin,times);
            if let Some(first)=trajectory.first_mut(){*first=(0..n).map(|i|initial[i*n+i].re).collect();}
            let horizons:Vec<Vec<f64>>=request.horizons.iter().map(|&t|model.horizon(&galerkin,t)).collect();
            let steady=model.steady_populations(&galerkin);
            timing.outputs_ms+=outputs_clock.elapsed().as_secs_f64()*1e3;
            let estimate_clock=Instant::now();
            let mut estimate=estimator.estimate(&ReducedView{basis:&space.basis,images:&space.images,r:model.r,vectors:&model.vectors,
                used_theta:&model.used_theta,used_coefficients:&model.used_coefficients,kernel:&model.kernel},&mut lattice)?;
            timing.estimate_ms+=estimate_clock.elapsed().as_secs_f64()*1e3;
            // The growth charge on every output.
            for v in estimate.trajectory.iter_mut().chain(estimate.horizons.iter_mut()).chain(estimate.steady.iter_mut()){*v+=model.growth_charge;}
            // The slowest relaxation (the classes from the frame's block).
            let pairs_clock=Instant::now();
            if classes.is_none()&&propagates{classes=Some(class_rates(&frame,eigen,lapack,initial,tolerance)?);}
            let rates:Option<Vec<f64>>=classes.as_ref().map(|c|c.iter().map(|x|x.0).collect());
            let pairs=slow_pairs(&model,&space,&ops,rates.as_deref(),tolerance)?;
            let unresolved:Vec<&SlowRate>=pairs.iter().filter(|(p,w)|!p.resolved()&&required(p,*w)).map(|(p,_)|p).collect();
            let missing=missing_classes(classes.as_deref().unwrap_or(&[]),&pairs,request.horizons,tolerance);
            timing.pairs_ms+=pairs_clock.elapsed().as_secs_f64()*1e3;
            // The outputs with their bounds.
            let answer={
                let outputs_clock=Instant::now();
                let projected:Vec<Vec<f64>>=trajectory.iter().map(|p|simplex(p)).collect();
                let changes:Vec<f64>=trajectory.iter().zip(&projected).map(|(a,b)|ledger::metric(a,b,request.groups)).collect();
                let sample_errors:Vec<f64>=estimate.trajectory.iter().zip(&changes).map(|(e,p)|ACCEPTANCE_FACTOR*e+p).collect();
                let field=|u:&[f64],images:bool|->Result<Vec<C>,String>{
                    let source:Vec<&[C]>=if images{space.images.iter().map(|v|v.as_slice()).collect()}else{space.basis.iter().map(|v|v.as_slice()).collect()};
                    Ok(ops.combinations(&source,&[u.to_vec()])?.remove(0))};
                let steady_field=field(&model.steady,false)?;let steady_image=field(&model.steady,true)?;
                let steady_residual=ops.norm(&steady_image)/ops.norm(&steady_field).max(f64::MIN_POSITIVE);
                let horizon_fields=request.horizons.iter().map(|&t|field(&model.state(t),false)).collect::<Result<Vec<_>,_>>()?;
                timing.outputs_ms+=outputs_clock.elapsed().as_secs_f64()*1e3;
                let mut answer=Answer{solves:space.solves(),modes:modes.clone(),trajectory:projected,sample_errors,projection:changes.iter().fold(0.,|a:f64,&b|a.max(b)),
                    horizons:horizons.clone(),steady:steady.clone(),horizon_fields,steady_field,
                    horizon_errors:estimate.horizons.iter().map(|e|ACCEPTANCE_FACTOR*e).collect(),steady_error:estimate.steady.map(|e|ACCEPTANCE_FACTOR*e),
                    steady_residual,estimate:estimate.clone(),
                    conserved:if modes.len()==1{eigen.kernel_count(super::reduced::KERNEL).max(1)}else{model.conserved()},
                    slowest:pairs.first().map(|p|p.0),accepted:false};
                answer.accepted=answer.largest()<=tolerance&&unresolved.is_empty()&&missing.is_empty();
                answer};
            let point=Checkpoint{solves:space.solves(),columns:space.basis.len(),dimension:model.r,estimate,largest:answer.largest(),accepted:answer.accepted,
                growing:model.growing,clipped:model.clipped,projected:model.projected,conserved:answer.conserved,slowest:answer.slowest,
                trajectory,horizons,steady,inner:lattice.work().clone(),timing:timing.clone(),milliseconds:clock.elapsed().as_secs_f64()*1e3};
            let observer_clock=Instant::now();
            if let Some(o)=observer.as_mut(){o(&point,&answer);}
            timing.observer_ms+=observer_clock.elapsed().as_secs_f64()*1e3;
            checkpoints.push(point);
            // The best checkpoint: one within the tolerance over any other,
            // the latest of those, else the least largest bound.
            let within=answer.largest()<=tolerance;
            if best.as_ref().is_none_or(|b|if within{true}else{b.largest()>tolerance&&answer.largest()<b.largest()}){best=Some(answer.clone());}
            if answer.accepted{break;}
            // 4. The stopping rule on the bound above the tolerance, else on
            // the slowest relaxation's need.
            // The missing classes first (slowest first), then the candidates.
            let needs:Vec<(f64,f64)>=missing.iter().map(|&class|class_need(class,&pairs))
                .chain(unresolved.iter().map(|p|need(p,rates.as_deref().unwrap_or(&[])))).collect();
            progress.push(if within{needs.iter().map(|x|x.0).fold(0.,f64::max)}else{answer.largest()});
            let phase=|i:usize|checkpoints[i].largest<=tolerance;
            let last=progress.len()-1;
            // Over the checkpoints of the current phase (within the tolerance
            // or not) only: a phase that began less than PLATEAU checkpoints
            // ago has no plateau yet.
            let start=(0..=last).rev().take_while(|&i|phase(i)==within).last().unwrap_or(last);
            let plateau=last-start>=PLATEAU&&{
                let before=progress[start..=last-PLATEAU].iter().copied().fold(f64::INFINITY,f64::min);
                let recent=progress[last+1-PLATEAU..].iter().copied().fold(f64::INFINITY,f64::min);
                recent>before/PLATEAU_GAIN};
            if exhausted||plateau{
                let b=best.as_ref().ok_or("A run has its checkpoint")?;
                let why=if plateau{format!("the error bound stopped improving ({PLATEAU_GAIN}x over {PLATEAU} checkpoints) after {} solves",space.solves())}
                    else if space.full(){format!("the basis memory was spent after {} solves",space.solves())}
                    else if live.is_empty(){format!("the pole ladder was exhausted after {} solves",space.solves())}
                    else{format!("the solve budget ({MAXIMUM_SOLVES}) was spent")};
                reason=Some(if b.largest()<=tolerance{format!("Every output is within the tolerance; the slowest relaxation is not resolved ({why}): it is reported as a range")}
                    else{
                        let over:Vec<usize>=(0..b.sample_errors.len()).filter(|&i|b.sample_errors[i]>tolerance).collect();
                        let samples=match(over.first(),over.last()){(Some(&a),Some(&z))=>format!("; {} of {} samples above it, t = {:.4} to {:.4} us",over.len(),b.sample_errors.len(),times[a],times[z]),_=>String::new()};
                        let means=b.horizon_errors.iter().chain(b.steady_error.iter()).filter(|&&e|e>tolerance).count();
                        let means=if means>0{format!("; {means} mean output(s) above it")}else{String::new()};
                        format!("Stopped because {why}: the best checkpoint ({} solves) has an error bound of {:.2e} against the tolerance {tolerance:e}{samples}{means}",b.solves,b.largest())});
                break;
            }
            slow_need=needs.first().copied();
            current=Some((space.solves(),model));
        }
        // The next pole by its expected gain.
        let clock=Instant::now();
        let model=match current.take(){Some((solves,model))if solves==space.solves()=>model,_=>{
            galerkin.ingest(metric.as_ref(),&space.basis,&space.images)?;galerkin.model(lapack,space.initial_norm,longest,tolerance)?}};
        let xis:Vec<C>=live.iter().map(|&i|points[i].sigma()).collect();
        let transform=model.error_transform(&galerkin,&space.basis,&space.images,eigen,lapack,&xis)?;
        for(&i,p)in live.iter().zip(&transform){if slow_points[i]{continue;}
            score[i]=points[i].sigma().re*p.iter().map(|z|magnitude(*z)).fold(0.,f64::max);
            if let Some(before)=solved_at[i].take(){remaining[i]=if before>0.{(score[i]/before).min(1.)}else{1.};}
        }
        let need=slow_need.map_or(0.,|(need,_)|need.min(1.));
        for &i in &live{if let Some(before)=slow_at[i].take(){slow_remaining[i]=if before>0.{(need/before).min(1.)}else{1.};}}
        // The slowest relaxation weighs as much as the best population gain
        // (at least the tolerance) times its need.
        let gain=|i:usize|->f64{if slow_points[i]{0.}else{score[i]*(1.-remaining[i])/points[i].copies()as f64}};
        let weight=live.iter().map(|&i|gain(i)).fold(tolerance,f64::max);
        let priority=|i:usize|->f64{
            let bonus=match(slow_need,points[i]){(Some((_,target)),Pole::Real(sigma))=>{
                let distance=(sigma/target).ln();weight*need*(1.-slow_remaining[i])/(1.+distance*distance)}
                _=>0.};
            gain(i)+bonus};
        let mut ranked:Vec<(usize,f64)>=live.iter().map(|&i|(i,priority(i))).collect();
        ranked.sort_by(|a,b|b.1.total_cmp(&a.1));
        timing.ordering_ms+=clock.elapsed().as_secs_f64()*1e3;
        // Solve the queue up to the next checkpoint.
        for &(i,_)in &ranked{
            if !alive[i]{continue;}
            solved_at[i]=Some(score[i]);if slow_need.is_some(){slow_at[i]=Some(need);}
            if space.solve(&mut lattice,points[i])?==0{alive[i]=false;}
            if space.solves()%CHECKPOINT==0||space.solves()>=MAXIMUM_SOLVES||space.full(){break;}
        }
    }
    let answer=best.ok_or("A run has its checkpoint")?;
    let work=lattice.work().clone();
    timing.solves_ms=work.milliseconds;timing.images_ms=space.images_ms;timing.projection_ms=galerkin.milliseconds;timing.total_ms=started.elapsed().as_secs_f64()*1e3;
    timing.frame_ms=frame.milliseconds();
    let converged=answer.accepted;
    Ok(Solution{answer,converged,reason:if converged{None}else{reason},translations:space.translations(),
        columns:space.columns.clone(),dimension:galerkin.columns(),images:space.basis.len(),work,ledger:ledger.clone(),timing,checkpoints,explicit:explicit.clone()})
    })
}

#[cfg(test)]
mod tests{
    use super::*;
    use super::super::tests::{fixture,prepared,dense_steady};
    use super::super::operator::LatticeOperator;
    #[test]
    fn lattice_pipeline_follows_the_exact_lattice_dynamics(){
        // The fixture from |0><0|: the trajectory (2 us), the state at a
        // horizon and the steady state against the same lattice P u Q
        // propagated densely (RK4, real form) and its dense null vector. The
        // far drives at 2% of the fixture's.
        let mut f=fixture();for v in f.far.iter_mut().chain(f.far2.iter_mut()){*v*=0.02;}let p=prepared(&f);let n=4;let d=16;let groups=vec![vec![0,1],vec![2,3]];
        let mut initial=vec![ZERO;d];initial[0]=ONE;
        let times:Vec<f64>=(0..21).map(|i|0.1*i as f64).collect();let horizons=[3.];
        let mut seen=0;let mut observe=|_:&Checkpoint,_:&Answer|{seen+=1;};
        let s=solve(&p,&initial,&Request{times:&times,horizons:&horizons,steady:true,groups:&groups,explicit:None},Some(&mut observe)).unwrap();
        assert!(s.converged&&s.reason.is_none()&&seen==s.checkpoints.len(),"{:?}",s.reason);
        let a=&s.answer;
        let op=LatticeOperator::new(&p,&a.modes).unwrap();let dense=op.dense();let size=op.dimension();
        let rate=|x:&[C]|->Vec<C>{(0..size).map(|r|{let mut v=ZERO;for c in 0..size{mul_add(&mut v,dense[r*size+c],x[c]);}v}).collect()};
        let mut x=vec![ZERO;size];x[op.centre*d]=ONE;
        let readout=|x:&[C],t:f64|->Vec<f64>{(0..n).map(|i|(0..a.modes.len()).map(|q|mul(C{re:(op.nu[q]*t).cos(),im:(op.nu[q]*t).sin()},x[q*d+i*n+i]).re).sum()).collect()};
        let steps=400;let h=0.1/steps as f64;let mut t=0.;let mut worst=(0_f64,0_f64);
        let axpy=|x:&[C],k:&[C],s:f64|->Vec<C>{x.iter().zip(k).map(|(a,b)|C{re:a.re+s*b.re,im:a.im+s*b.im}).collect()};
        for(i,row)in a.trajectory.iter().enumerate(){
            if i>0{for _ in 0..steps{let k1=rate(&x);let k2=rate(&axpy(&x,&k1,h/2.));let k3=rate(&axpy(&x,&k2,h/2.));let k4=rate(&axpy(&x,&k3,h));
                for e in 0..size{x[e].re+=h/6.*(k1[e].re+2.*k2[e].re+2.*k3[e].re+k4[e].re);x[e].im+=h/6.*(k1[e].im+2.*k2[e].im+2.*k3[e].im+k4[e].im);}t+=h;}}
            // Every sample within 1 / 0.8 of its own bound (above 2% of the
            // tolerance: the validation's floor; the estimate is no per-time
            // bound), the bounds within the tolerance.
            let error=ledger::metric(&readout(&x,t),row,&groups);worst=(worst.0.max(error),worst.1.max(a.sample_errors[i]));
            assert!(0.8*error<=a.sample_errors[i].max(0.02*POPULATION_TOLERANCE)&&a.sample_errors[i]<=POPULATION_TOLERANCE,"sample {i}: {error:e} against {:e}",a.sample_errors[i]);
        }
        // The time maximum within 1 / 0.9 of the largest bound (measured:
        // 0.95 in the frame's metric, the errors at 4% of the tolerance).
        assert!(0.9*worst.0<=worst.1,"trajectory {:e} against {:e}",worst.0,worst.1);
        for _ in 0..10*steps{let k1=rate(&x);let k2=rate(&axpy(&x,&k1,h/2.));let k3=rate(&axpy(&x,&k2,h/2.));let k4=rate(&axpy(&x,&k3,h));
            for e in 0..size{x[e].re+=h/6.*(k1[e].re+2.*k2[e].re+2.*k3[e].re+k4[e].re);x[e].im+=h/6.*(k1[e].im+2.*k2[e].im+2.*k3[e].im+k4[e].im);}}
        let centre:Vec<f64>=(0..n).map(|i|x[op.centre*d+i*n+i].re).collect();
        let horizon=ledger::metric(&centre,&a.horizons[0],&groups);assert!(horizon<=POPULATION_TOLERANCE,"horizon {horizon:e}");
        let exact=dense_steady(&p,&dense,op.centre,n);
        // The bordered column at a tight tolerance is the lattice's null
        // vector; the reported steady state carries the algebraic error of
        // the bordered solve at STEADY_TOLERANCE.
        let mut lattice=Lattice::new(&p,s.explicit.sets.clone()).unwrap();let mut u=vec![ZERO;size];for i in 0..n{u[op.centre*d+i*n+i]=C{re:0.25,im:0.};}
        let(w,_,_)=lattice.solve(Pole::Zero,&[u],1e-13).unwrap();let trace:f64=(0..n).map(|i|w[0][op.centre*d+i*n+i].re).sum();
        let column:Vec<f64>=(0..n).map(|i|w[0][op.centre*d+i*n+i].re/trace).collect();
        let tight=ledger::metric(&exact,&column,&groups);assert!(tight<1e-10,"bordered column {tight:e}");
        let steady=ledger::metric(&exact,&a.steady,&groups);assert!(steady<1e-4,"steady {steady:e}");
        assert!(a.conserved==1&&a.slowest.is_some());
        // Projected rows: probabilities, the first one rho0's.
        assert!(a.trajectory.iter().all(|r|r.iter().all(|&v|v>=0.)&&(r.iter().sum::<f64>()-1.).abs()<1e-12)&&a.trajectory[0][0]==1.);
    }
}
