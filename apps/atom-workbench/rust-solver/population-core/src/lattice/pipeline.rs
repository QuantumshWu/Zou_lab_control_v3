//! The population pipeline: every output of a request is a matrix function
//! of one lattice operator applied to x0, read from one reduced model.
//!
//! 1. The explicit set P and its ring Q (`charge_set`); the lattice P u Q
//!    and its solves at the poles (`poles`). The request's frame (`frame`:
//!    the reduced model's metric, the estimate's fast part, the slowest
//!    relaxation's classes) is built on its own thread meanwhile.
//! 2. One rational Krylov space from x0 (`space`): the bordered steady
//!    column first, the translations through the gate active from the start,
//!    then the ladder's poles by their expected gain, re-ranked at every
//!    checkpoint: sigma |p_err(xi)| (the error transform) times the share of
//!    its own transform a pole's last solve removed (a pole whose solve did
//!    not reduce it loses its turn; one never solved keeps the whole), per
//!    real solve (a complex pole costs two); a pole that adds no direction
//!    leaves the ladder.
//! 3. Every CHECKPOINT solves: the Galerkin model in the frame's metric
//!    (`reduced`), its outputs (trajectory on the grid, horizons, steady
//!    state, slowest relaxation, conserved count) and the error estimate
//!    (`estimator`). An output's bound is ACCEPTANCE_FACTOR x its estimate,
//!    plus, on a trajectory row, the change of its projection onto the
//!    probability simplex (computed exactly, not estimated). A checkpoint
//!    is accepted when every bound is within the tolerance: the estimate
//!    charges every slow mode's own residual (`estimator`), so a slow class
//!    the space lacks or resolves badly enters the bound itself.
//! 4. Stopping (deterministic, no wall time): accepted; or the largest
//!    bound, above the tolerance, did not improve by PLATEAU_GAIN over the
//!    last PLATEAU checkpoints; or the solve budget
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
use super::reduced::{Model,Projection,SlowRate};
use super::poles::{InnerWork,Lattice,Pole};
use super::space::{self,Column,Space};
use super::frame::{SwFrame,SwMetric};
use super::estimator::UnifiedEstimate;
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
/// Memory guard: a request takes about BASE_BYTES plus MODE_BYTES per lattice
/// mode (measured, private bytes: D1 + three D2 lasers, 13 modes, 0.62 GB;
/// six lasers without the trap, 63 modes, 2.0 GB; the pole batches and the
/// estimator's ring grow with the modes); one whose lattice would need more
/// than MEMORY_BYTES stops before its first solve, with the reason.
const BASE_BYTES:f64=5e8;
const MODE_BYTES:f64=3.2e7;
const MEMORY_BYTES:f64=5e9;
/// The memory a lattice of `modes` modes is planned to take (bytes).
fn planned_bytes(modes:usize)->f64{BASE_BYTES+MODE_BYTES*modes as f64}
/// The plateau: the largest bound did not improve by PLATEAU_GAIN over the
/// last PLATEAU checkpoints. Measured
/// before the rule: four lasers + tweezer at 10 us / 2000 samples held the
/// bound at 5.0-5.7e-3 over 20 solves, five lasers at about 1e-2 over 24.
/// Over two checkpoints four lasers without the trap at 20 us stopped at 16
/// solves (bound 1.2e-2), accepted at 20 over three.
const PLATEAU:usize=3;
const PLATEAU_GAIN:f64=2.;
/// The reported slowest relaxation (`slowestRelaxationUs`, no part of the
/// acceptance): its candidates carry population weight of at least
/// CANDIDATE_WEIGHT x tolerance; a candidate is matched when every class of
/// x0 (weight at least CANDIDATE_WEIGHT x tolerance) slower than its rate /
/// CLASS_GAP has a candidate within CLASS_MATCH of its rate.
/// CANDIDATE_WEIGHT, measured: the spurious slow pairs of a four-laser case
/// carried 1.2e-5 and 1.5e-6, the physical one 4e-2; 1e-2 x tolerance =
/// 5e-5 lies 2-3 orders from both.
const CANDIDATE_WEIGHT:f64=1e-2;
/// Derived: 1 + CLASS_MATCH, so a class slower than rate / CLASS_GAP is
/// outside the candidate's own match and needs another candidate.
const CLASS_GAP:f64=1.+CLASS_MATCH;
/// A deliberate margin over the measured offset between a class (the
/// frame's second-order block) and its Ritz pair: 18% on a four-laser case
/// (class 0.0418 /us, pair 0.0344 /us).
const CLASS_MATCH:f64=0.4;
/// Grid samples assumed for the ladder's top when a request has no grid.
/// LADDER_SAMPLES is a deliberate margin: as if the horizon were sampled
/// like the app's trajectories (161-200 samples on the research scenes).
const LADDER_SAMPLES:f64=200.;
/// A decaying class of the frame's block carries none of x0's trace; a
/// class whose trace is at least this share of its weight is the block's
/// kernel. Measured: decaying classes at most 5e-3 of their weight, the
/// kernel 1.6 (trace 1.00, weight 0.63): a margin of 100x and 3x.
const KERNEL_TRACE_SHARE:f64=0.5;
/// Completion (`solve`): at the first checkpoint, a bound above the
/// tolerance of which the slow part (the ring's feedback and the K^(1) rests
/// the lattice leaves out) carries at least COMPLETION_SHARE is completed.
/// Measured (three lasers at B = 0 and 1 mG, p(1 s)): at the first
/// checkpoint the Krylov part is 1e-11 of bounds of 7.1-7.4e-3, the ring's
/// part 3.8-4.3e-3 and the K^(1) rests' 1.1-1.2e-2 (signed parts, partly
/// cancelling); a trajectory's first checkpoint is the Krylov residual's
/// (2e-1 against 2-8e-3).
const COMPLETION_SHARE:f64=0.5;
/// The K^(1) rests join the operator when their part is at least this
/// share of the tolerance (each alone fails: both lattices' means stayed
/// 1.3-2.0e-3 off with either change alone, 4e-5 and 1.7e-4 with both).
const K1_COMPLETION:f64=0.25;
/// The ring charges promoted: by decreasing attribution until they carry
/// this share of the ring total.
const RING_COMPLETION:f64=0.9;
/// Stack of the frame's thread (LAPACK on it).
const FRAME_STACK:usize=1<<26;

/// What a request asks for: a trajectory or horizons (or both), and with
/// them the steady state.
pub struct Request<'a>{
    /// The uniform grid from 0 (empty: no trajectory).
    pub times:&'a[f64],
    /// Horizons T (us) of the laboratory time average.
    pub horizons:&'a[f64],
    /// Whether the steady state (the t -> infinity mean) is reported.
    pub steady:bool,
    pub groups:&'a[Vec<usize>],
    /// The explicit set shared by the runs of one request from the same rho0
    /// (the first run to need it builds it).
    pub explicit:&'a std::sync::OnceLock<Result<ExplicitSet,String>>,
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
    /// rho0's) and the bound of each (ACCEPTANCE_FACTOR x the estimate plus
    /// the projection's change).
    pub trajectory:Vec<Vec<f64>>,pub sample_errors:Vec<f64>,
    /// Horizon and steady populations (laboratory time averages, raw), their
    /// fields on `modes` (Hermitian-symmetric) and bounds.
    pub horizons:Vec<Vec<f64>>,pub steady:Vec<f64>,
    pub horizon_fields:Vec<Vec<C>>,pub steady_field:Vec<C>,
    pub horizon_errors:Vec<f64>,pub steady_error:Option<f64>,
    /// Per horizon T, the population the reduced model still relaxes at T:
    /// sum over its decaying modes (neither kernel nor projected) of their
    /// largest centre population contribution at T (p(T) reads the centre
    /// block alone), amplitude x e^{Re theta T}: an upper bound on
    /// |p(T) - p(inf)| as the model holds it.
    pub horizon_relaxing:Vec<f64>,
    /// |A x| / |x| of the steady field (rad/us).
    pub steady_residual:f64,
    pub estimate:ErrorEstimate,pub conserved:usize,pub slowest:Option<SlowRate>,
    /// Every bound within the tolerance.
    pub accepted:bool,
}
impl Answer{
    /// The largest bound over every output.
    pub fn largest(&self)->f64{self.sample_errors.iter().chain(&self.horizon_errors).chain(self.steady_error.iter()).fold(0_f64,|a,&b|a.max(b))}
}

#[derive(Clone,Debug,Default)]
pub struct Timing{pub eigenbasis_ms:f64,pub explicit_set_ms:f64,pub setup_ms:f64,pub estimator_build_ms:f64,pub outputs_ms:f64,pub frame_ms:[f64;3],pub frame_wait_ms:f64,pub solves_ms:f64,pub images_ms:f64,pub projection_ms:f64,pub model_ms:f64,pub ordering_ms:f64,pub estimate_ms:f64,pub total_ms:f64}

pub struct Solution{
    /// The best checkpoint's outputs; `converged` when it was accepted, else
    /// the reason the run stopped.
    pub answer:Answer,pub converged:bool,pub reason:Option<String>,
    /// The completion decided at the first checkpoint (also when it changed
    /// nothing: `restarted` false), and the first run's solves when it
    /// restarted the run.
    pub completion:Option<Completion>,
    pub explicit:ExplicitSet,pub translations:Vec<Charge>,
    /// The completed run's solves, and every checkpoint reported (when the
    /// run restarted, the first run's first).
    pub columns:Vec<Column>,pub checkpoints:Vec<Checkpoint>,
    pub dimension:usize,pub work:InnerWork,pub images:usize,
    pub ledger:RemainderLedger,pub timing:Timing,
}

/// A lattice completion: the ring charges promoted into P (one of each
/// pair; their mirrors join with them), the completed explicit set (the
/// run's own when none is) and whether the operator keeps every K^(1) rest;
/// where it was decided, the bound there, its slow part and the K^(1)
/// rests' part (bound units), the ring charges' attribution shares, and
/// whether the run restarted on the completed lattice.
#[derive(Clone,Debug)]
pub struct Completion{
    pub promoted:Vec<Charge>,pub explicit:ExplicitSet,pub keep_k1:bool,pub decided_at:usize,
    pub bound:f64,pub slow_part:f64,pub k1_part:f64,
    pub attribution:Vec<(Charge,f64)>,pub restarted:bool,
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

/// The slowest relaxation's candidates of a model (`Model::slow_candidates`),
/// slowest first, each with its population weight and its rate, residual
/// |A W y - theta W y| / |W y| (from the images, which keep its accuracy at
/// slow rates) and whether the classes `classes` slower than its rate /
/// CLASS_GAP are matched: the residual cannot see a slow class the space
/// lacks.
fn slow_pairs(model:&Model,space:&Space,vectors:&super::krylov::Vectors,classes:&[f64],tolerance:f64)->Result<Vec<(SlowRate,f64)>,String>{
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
    let matched=|rate:f64|classes.iter().filter(|&&class|class<rate/CLASS_GAP).all(|&class|found.iter().any(|f|(f.0-class).abs()<=CLASS_MATCH*class));
    Ok(found.iter().map(|&(rate,residual,weight)|(SlowRate{rate,residual,matched:matched(rate)},weight)).collect())
}


/// Every output of a request from rho0 (row-major n x n); `observer` sees
/// every checkpoint with its outputs as it is made.
///
/// Completion: at the first checkpoint, when the largest bound is above the
/// tolerance and its slow part (the omitted ring's feedback and the K^(1)
/// rests the lattice leaves out) is at least COMPLETION_SHARE of it, the
/// K^(1) rests the lattice leaves out join the operator when the estimate
/// with them alone is at least K1_COMPLETION x tolerance, and the ring
/// charges carrying RING_COMPLETION of the ring's attribution (|a_m| |B y_r|
/// over the kernel and slow modes, from the ring's feedback alone) join P
/// with their mirrors. The run then restarts once on the completed lattice
/// (the frame and the eigenbasis are kept), its checkpoints after the first
/// run's. Every other run is unchanged.
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
    let longest=times.last().copied().into_iter().chain(request.horizons.iter().copied()).fold(0_f64,f64::max);
    let frame=SwFrame::new(prepared,longest);
    std::thread::scope(|scope|->Result<Solution,String>{
        std::thread::Builder::new().stack_size(FRAME_STACK).spawn_scoped(scope,||{let _=frame.class();}).map_err(|e|e.to_string())?;
        match attempt(prepared,initial,request,&mut observer,&frame,None,started)?{
            Ok(solution)=>Ok(solution),
            Err((completion,first))=>{
                let mut solution=attempt(prepared,initial,request,&mut observer,&frame,Some(&completion),started)?
                    .map_err(|_|"A completed lattice asked for another completion".to_string())?;
                solution.checkpoints.splice(0..0,first);Ok(solution)}
        }
    })
}

/// One run of `solve` on the request's lattice or on its completion: the
/// solution, or (Err) the completion its first checkpoint asks for, with
/// that checkpoint.
fn attempt<'a>(prepared:&'a LatticePrepared,initial:&[C],request:&Request<'_>,observer:&mut Option<&mut dyn FnMut(&Checkpoint,&Answer)>,
    frame:&'a SwFrame<'a>,completion:Option<&Completion>,started:Instant)->Result<Result<Solution,(Completion,Vec<Checkpoint>)>,String>{
    let tolerance=POPULATION_TOLERANCE;let n=prepared.generator.n;
    if initial.len()!=n*n{return Err("Initial density dimension differs".into());}
    // The longest output time, the frame's.
    let(times,longest)=(request.times,frame.horizon);
    let t1=if times.len()>1{times[1]}else{longest/LADDER_SAMPLES};
    let mut timing=Timing::default();
    // 1. The lattice (the frame is built on its own thread meanwhile).
    let clock=Instant::now();let eigen=prepared.eigenbasis()?;timing.eigenbasis_ms=clock.elapsed().as_secs_f64()*1e3;let clock=Instant::now();
    let explicit=match completion{Some(c)=>c.explicit.clone(),None=>request.explicit.get_or_init(||charge_set::explicit_set(prepared,initial,tolerance)).clone()?};
    timing.explicit_set_ms=clock.elapsed().as_secs_f64()*1e3;
    let keep_k1=completion.is_some_and(|c|c.keep_k1);
    let clock=Instant::now();
    let mut lattice=Lattice::with_k1_rests(prepared,explicit.sets.clone(),keep_k1)?;
    let ledger=lattice.operator().ledger()?;
    let planned=planned_bytes(lattice.modes.len());
    if planned>MEMORY_BYTES{return Err(format!("The lattice of this request has {} modes and would need about {:.1} GB, above the {:.0} GB memory budget: not solved",lattice.modes.len(),planned/1e9,MEMORY_BYTES/1e9));}
    // A static fold whose expansion parameter p = nu_max / (|nu_far| - ||L0||)
    // exceeds EXPANSION_LIMIT keeps its K^(1) term and is charged its
    // remainder (`ledger::nu_freeze`, a generator bound delta): delta min(t, tau)
    // on every output, tau the slowest relaxation time; at p >= 1 the
    // expansion diverges and the request is not solved.
    if ledger.expansion_parameter>=1.{return Err(format!("Static fold expansion parameter {:e}: the fold's frequency expansion diverges on this lattice, not solved",ledger.expansion_parameter));}
    let freeze=(ledger.expansion_parameter>EXPANSION_LIMIT).then_some(ledger.nu_freeze);
    // 2. The space: translations, the steady column.
    let capacity={let op=lattice.operator();let d=op.n*op.n;BASIS_BYTES/(std::mem::size_of::<C>()*2*d*op.modes.len())};
    let mut space=Space::new(&mut lattice,initial,capacity)?;
    for s in charge_set::gated_steps(prepared,tolerance){if lattice.operator().index.contains_key(&s){space.activate(&mut lattice,&s)?;}}
    timing.setup_ms=clock.elapsed().as_secs_f64()*1e3;
    space.solve(&mut lattice,Pole::Zero)?;
    let modes=lattice.modes.clone();
    let clock=Instant::now();
    let mut estimator=UnifiedEstimate::new(EstimateSetup{prepared,lattice:&explicit.sets,outputs:Outputs{times,horizons:request.horizons,steady:request.steady,groups:request.groups},frame,ledger:&ledger,keep_k1})?;
    timing.estimator_build_ms=clock.elapsed().as_secs_f64()*1e3;
    // The ladder spans the grid's times when there is one (a horizon is
    // served by the steady column and the slow points): its poles near 1 / T of a 1 s horizon grew
    // spurious near-zero slow Ritz pairs in both metrics (rates 1e-8 to
    // 1e-5, residuals 0.2-3) whose p(1 s) bounds reached 1e1-1e4; without
    // them three, four, five lasers and four without the trap at 20 us were
    // accepted at 16-24 solves (four without the trap partial before) and
    // five lasers at 10 us / 2000 samples reached 7.0e-3 against 2.3e-2.
    let(points,slow_points)=space::ladder(eigen,initial,t1,times.last().copied().unwrap_or(longest),tolerance)?;let mut alive=vec![true;points.len()];
    // The frame's metric (waiting for the frame's thread).
    let clock=Instant::now();
    let metric=SwMetric::new(frame,eigen,lattice.operator())?;
    timing.frame_wait_ms=clock.elapsed().as_secs_f64()*1e3;
    let mut galerkin=Projection::new(lattice.operator(),&metric)?;
    let lapack=&prepared.lapack;let ops=super::krylov::Vectors::new(lattice.operator(),1)?;
    // The decaying classes of x0, which the slowest relaxation is checked
    // against (the frame's block).
    let classes:Vec<f64>=class_rates(frame,eigen,lapack,initial,tolerance)?.into_iter().map(|c|c.0).collect();
    let mut checkpoints:Vec<Checkpoint>=Vec::new();
    // The best checkpoint so far, the latest model, and per checkpoint the
    // progress measure of the stopping rule.
    let mut best:Option<Answer>=None;let mut progress:Vec<f64>=Vec::new();
    let mut reason:Option<String>=None;
    let mut current:Option<(usize,Model)>=None;
    // Per point: its last score, the score it had when last solved, and the
    // share of its transform left after that solve.
    let(mut score,mut solved_at,mut remaining)=(vec![0.;points.len()],vec![None::<f64>;points.len()],vec![0.;points.len()]);
    // The first checkpoint's completion decision when it changed nothing.
    let mut decided:Option<Completion>=None;
    loop{
        let live:Vec<usize>=(0..points.len()).filter(|&i|alive[i]).collect();
        let exhausted=live.is_empty()||space.solves()>=MAXIMUM_SOLVES||space.full();
        if exhausted||space.solves()%CHECKPOINT==0{
            // 3. The checkpoint: model, outputs, estimate, bounds.
            let clock=Instant::now();
            galerkin.ingest(&space.basis,&space.images)?;
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
                used_theta:&model.used_theta,used_coefficients:&model.used_coefficients,kernel:&model.kernel,projected:&model.projected_out},&mut lattice)?;
            timing.estimate_ms+=estimate_clock.elapsed().as_secs_f64()*1e3;
            // The growth charge on every output.
            for v in estimate.trajectory.iter_mut().chain(estimate.horizons.iter_mut()).chain(estimate.steady.iter_mut()){*v+=model.growth_charge;}
            // The slowest relaxation.
            let pairs=slow_pairs(&model,&space,&ops,&classes,tolerance)?;
            if let Some(delta)=freeze{let tau=pairs.first().map_or(f64::INFINITY,|(p,_)|1./p.rate);
                for(v,&t)in estimate.trajectory.iter_mut().zip(times){*v+=delta*t.min(tau);}
                for(v,&t)in estimate.horizons.iter_mut().zip(request.horizons){*v+=delta*t.min(tau);}
                for v in estimate.steady.iter_mut(){*v+=delta*tau;}}
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
                let mut answer=Answer{solves:space.solves(),modes:modes.clone(),trajectory:projected,sample_errors,
                    horizons:horizons.clone(),steady:steady.clone(),horizon_fields,steady_field,
                    horizon_errors:estimate.horizons.iter().map(|e|ACCEPTANCE_FACTOR*e).collect(),
                    horizon_relaxing:request.horizons.iter().map(|&t|(0..model.r).filter(|&j|!model.kernel[j]&&!model.projected_out[j])
                        .map(|j|model.centre_amplitudes[j]*(model.used_theta[j].re*t).exp()).sum()).collect(),steady_error:estimate.steady.map(|e|ACCEPTANCE_FACTOR*e),
                    steady_residual,estimate:estimate.clone(),
                    conserved:if modes.len()==1{eigen.kernel_count(super::reduced::KERNEL).max(1)}else{model.conserved()},
                    slowest:pairs.first().map(|p|p.0),accepted:false};
                answer.accepted=answer.largest()<=tolerance;
                answer};
            let point=Checkpoint{solves:space.solves(),columns:space.basis.len(),dimension:model.r,estimate,largest:answer.largest(),accepted:answer.accepted,
                growing:model.growing,clipped:model.clipped,projected:model.projected,conserved:answer.conserved,slowest:answer.slowest,
                trajectory,horizons,steady,inner:lattice.work().clone(),timing:timing.clone(),milliseconds:clock.elapsed().as_secs_f64()*1e3};
            if let Some(o)=observer.as_mut(){o(&point,&answer);}
            checkpoints.push(point);
            // The best checkpoint: one within the tolerance over any other,
            // the latest of those, else the least largest bound.
            let within=answer.largest()<=tolerance;
            if best.as_ref().is_none_or(|b|if within{true}else{b.largest()>tolerance&&answer.largest()<b.largest()}){best=Some(answer.clone());}
            if answer.accepted{break;}
            // 5. Completion, decided at the first checkpoint (see `solve`).
            if completion.is_none()&&checkpoints.len()==1{
                let e=&answer.estimate;let off=answer.sample_errors.len();let horizons_count=answer.horizon_errors.len();
                // Per output: its bound and its slow part (bound units); the steady
                // state's estimate is its slow part.
                let mut outputs:Vec<(f64,f64,usize)>=answer.sample_errors.iter().enumerate()
                    .map(|(i,&b)|(b,ACCEPTANCE_FACTOR*e.slow_trajectory.get(i).copied().unwrap_or(0.),i)).collect();
                outputs.extend(answer.horizon_errors.iter().enumerate().map(|(h,&b)|(b,ACCEPTANCE_FACTOR*e.slow_horizons.get(h).copied().unwrap_or(0.),off+h)));
                if let Some(b)=answer.steady_error{outputs.push((b,b,off+horizons_count));}
                let(bound,part,at)=outputs.iter().copied().max_by(|a,b|a.0.total_cmp(&b.0)).unwrap_or((0.,0.,0));
                if bound>tolerance&&part>=COMPLETION_SHARE*bound{
                    let view=ReducedView{basis:&space.basis,images:&space.images,r:model.r,vectors:&model.vectors,
                        used_theta:&model.used_theta,used_coefficients:&model.used_coefficients,kernel:&model.kernel,projected:&model.projected_out};
                    let pick=|x:&ErrorEstimate|ACCEPTANCE_FACTOR*if at<off{x.trajectory[at]}else if at<off+horizons_count{x.horizons[at-off]}else{x.steady.unwrap_or(0.)};
                    // The K^(1) rests' part, when the lattice leaves any out.
                    let drops=lattice.operator().drops_k1_rests();
                    let k1_part=if drops{pick(&estimator.k1_estimate(&view,&mut lattice)?)}else{0.};
                    let keep=drops&&k1_part>=K1_COMPLETION*tolerance;
                    // The ring charges by attribution, a pair together (the canonical
                    // member, the larger charge, stands for it).
                    let mut pairs:BTreeMap<Charge,f64>=BTreeMap::new();
                    for(c,a)in estimator.ring_attribution(&view)?{let m=opposite(&c);let key=if c>m{c}else{m};*pairs.entry(key).or_default()+=a;}
                    let total:f64=pairs.values().sum();
                    let mut ranked:Vec<(Charge,f64)>=pairs.into_iter().collect();ranked.sort_by(|a,b|b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                    let mut promoted=Vec::new();let mut carried=0.;
                    if total>0.{for(c,a)in &ranked{if carried>=RING_COMPLETION*total{break;}promoted.push(c.clone());carried+=a;}}
                    // The completed lattice must fit the memory budget and keep the
                    // fold's frequency expansion small (its parameter scales with the
                    // largest |nu_q|): else no charge is promoted.
                    let mut grown=None;
                    if !promoted.is_empty(){
                        let set=charge_set::completed(prepared,&explicit,&promoted)?;
                        let nu_max=|modes:&[Charge]|->Result<f64,String>{modes.iter().map(|q|prepared.nu(q).map(f64::abs)).try_fold(0_f64,|a,v|v.map(|v|a.max(v)))};
                        let(before,after)=(nu_max(&explicit.sets.modes())?,nu_max(&set.sets.modes())?);
                        let expansion=if before>0.{ledger.expansion_parameter*after/before}else{ledger.expansion_parameter};
                        if planned_bytes(set.sets.modes().len())>MEMORY_BYTES||expansion>EXPANSION_LIMIT{promoted.clear();}else{grown=Some(set);}
                    }
                    let attribution:Vec<(Charge,f64)>=ranked.iter().map(|(c,a)|(c.clone(),if total>0.{a/total}else{0.})).collect();
                    let restarted=keep||!promoted.is_empty();
                    let decision=Completion{promoted,explicit:grown.unwrap_or_else(||explicit.clone()),keep_k1:keep,decided_at:space.solves(),bound,slow_part:part,k1_part,attribution,restarted};
                    if restarted{return Ok(Err((decision,checkpoints)));}
                    decided=Some(decision);
                }
            }
            // 4. The stopping rule on the bound (a checkpoint within the
            // tolerance is accepted).
            progress.push(answer.largest());
            let last=progress.len()-1;
            let plateau=last>=PLATEAU&&{
                let before=progress[..=last-PLATEAU].iter().copied().fold(f64::INFINITY,f64::min);
                let recent=progress[last+1-PLATEAU..].iter().copied().fold(f64::INFINITY,f64::min);
                recent>before/PLATEAU_GAIN};
            if exhausted||plateau{
                let b=best.as_ref().ok_or("A run has its checkpoint")?;
                let why=if plateau{format!("the error bound stopped improving ({PLATEAU_GAIN}x over {PLATEAU} checkpoints) after {} solves",space.solves())}
                    else if space.full(){format!("the basis memory was spent after {} solves",space.solves())}
                    else if live.is_empty(){format!("the pole ladder was exhausted after {} solves",space.solves())}
                    else{format!("the solve budget ({MAXIMUM_SOLVES}) was spent")};
                reason=Some({
                        let over:Vec<usize>=(0..b.sample_errors.len()).filter(|&i|b.sample_errors[i]>tolerance).collect();
                        let samples=match(over.first(),over.last()){(Some(&a),Some(&z))=>format!("; {} of {} samples above it, t = {:.4} to {:.4} us",over.len(),b.sample_errors.len(),times[a],times[z]),_=>String::new()};
                        let means=b.horizon_errors.iter().chain(b.steady_error.iter()).filter(|&&e|e>tolerance).count();
                        let means=if means>0{format!("; {means} mean output(s) above it")}else{String::new()};
                        format!("Stopped because {why}: the best checkpoint ({} solves) has an error bound of {:.2e} against the tolerance {tolerance:e}{samples}{means}",b.solves,b.largest())});
                break;
            }
            current=Some((space.solves(),model));
        }
        // The next pole by its expected gain.
        let clock=Instant::now();
        let model=match current.take(){Some((solves,model))if solves==space.solves()=>model,_=>{
            galerkin.ingest(&space.basis,&space.images)?;galerkin.model(lapack,space.initial_norm,longest,tolerance)?}};
        let xis:Vec<C>=live.iter().map(|&i|points[i].sigma()).collect();
        let transform=model.error_transform(&galerkin,&space.basis,&space.images,eigen,lapack,&xis)?;
        for(&i,p)in live.iter().zip(&transform){if slow_points[i]{continue;}
            score[i]=points[i].sigma().re*p.iter().map(|z|magnitude(*z)).fold(0.,f64::max);
            if let Some(before)=solved_at[i].take(){remaining[i]=if before>0.{(score[i]/before).min(1.)}else{1.};}
        }
        let gain=|i:usize|->f64{if slow_points[i]{0.}else{score[i]*(1.-remaining[i])/points[i].copies()as f64}};
        let mut ranked:Vec<(usize,f64)>=live.iter().map(|&i|(i,gain(i))).collect();
        ranked.sort_by(|a,b|b.1.total_cmp(&a.1));
        timing.ordering_ms+=clock.elapsed().as_secs_f64()*1e3;
        // Solve the queue up to the next checkpoint.
        for &(i,_)in &ranked{
            if !alive[i]{continue;}
            solved_at[i]=Some(score[i]);
            if space.solve(&mut lattice,points[i])?==0{alive[i]=false;}
            if space.solves()%CHECKPOINT==0||space.solves()>=MAXIMUM_SOLVES||space.full(){break;}
        }
    }
    let answer=best.ok_or("A run has its checkpoint")?;
    let work=lattice.work().clone();
    timing.solves_ms=work.milliseconds;timing.images_ms=space.images_ms;timing.projection_ms=galerkin.milliseconds;timing.total_ms=started.elapsed().as_secs_f64()*1e3;
    timing.frame_ms=frame.milliseconds();
    let converged=answer.accepted;
    Ok(Ok(Solution{answer,converged,reason:if converged{None}else{reason},completion:completion.cloned().or(decided),translations:space.translations(),
        columns:space.columns.clone(),dimension:galerkin.columns(),images:space.basis.len(),work,ledger:ledger.clone(),timing,checkpoints,explicit:explicit.clone()}))
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
        let s=solve(&p,&initial,&Request{times:&times,horizons:&horizons,steady:true,groups:&groups,explicit:&std::sync::OnceLock::new()},Some(&mut observe)).unwrap();
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
