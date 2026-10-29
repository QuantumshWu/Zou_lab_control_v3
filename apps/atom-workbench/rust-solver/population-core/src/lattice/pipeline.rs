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
//! 3. Every CHECKPOINT solves (a run without a grid, the mean's, also right
//!    after the steady column): the Galerkin model in the frame's metric
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
use super::estimator::{Source,UnifiedEstimate};
use super::eigenbasis::BlockEigenbasis;

/// Solves between checkpoints; the ladder is re-ranked at every checkpoint
/// (from its reduced model) and whenever its queue runs out. Measured at
/// this spacing: the estimates up to acceptance cost 19-35% of the solves on
/// three to five lasers; acceptance is seen at most three solves late.
/// A run without a grid (a mean run) takes its first one right after the
/// steady column and decides the completion there. Measured on the
/// necessity review's 9 mean runs: three, four, five lasers, four without
/// the trap, D1 + D2 and three lasers at B = 0 and 1 mG accepted at 1 solve
/// instead of 4, the errors against references (the steady states, four
/// without the trap's mean at 1 s; 1.3e-3 and 2.2e-3 at B = 0 and 1 mG)
/// the same and within their bounds; the default scene, still relaxing at
/// 1 s (bound 9.7e-3 at 1 solve), at 4 as before. Never a trajectory: a
/// trajectory checkpoint at 1 solve accepted four lasers without the trap
/// at 10 ms falsely (the necessity review: bound 3.0e-3, error 7.3e-3).
const CHECKPOINT:usize=4;
/// Most solves of a request, and the basis memory (basis and images).
/// MAXIMUM_SOLVES is a deliberate margin: twice the 24 solves the slowest
/// accepted research case takes (16-24 at 20 us).
const MAXIMUM_SOLVES:usize=48;
const BASIS_BYTES:usize=1<<30;
/// Memory guard: a request takes about BASE_BYTES plus MODE_BYTES per
/// lattice mode of each run, and its runs execute at once (`Request::runs`:
/// a full request's trajectory and mean, each with its own lattice, space
/// and estimator). Measured, private bytes: D1 + three D2 lasers, 13 modes,
/// 0.62 GB; six lasers without the trap, 63 modes, 2.0 GB (the pole batches
/// and the estimator's ring grow with the modes); five lasers at 20 us /
/// 1000 samples, 63 modes: the trajectory alone 1.60 GB, the mean alone
/// 0.58 GB, both 1.65 GB. A run plans every run of its request on a lattice
/// of its own size, so the runs together stay within the budget: a request
/// beyond it stops before its first solve, with the reason, and a
/// completion beyond it is not taken. The budget is MEMORY_SHARE of the
/// machine's installed physical memory, not a fixed size: a run beyond the
/// physical memory pages, and every solve sweeps the lattice and the whole
/// basis, so it would stall the machine rather than slow down; the
/// installed memory rather than the free, so whether a request is solved
/// does not depend on what else runs. MEMORY_SHARE is a deliberate margin:
/// the other half for the system, the app and the lab PC's other programs.
const BASE_BYTES:f64=5e8;
const MODE_BYTES:f64=3.2e7;
const MEMORY_SHARE:f64=0.5;
/// The memory a request of `runs` runs on lattices of `modes` modes is
/// planned to take (bytes).
fn planned_bytes(runs:usize,modes:usize)->f64{BASE_BYTES+MODE_BYTES*(runs*modes)as f64}
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
/// tolerance of which the slow part (the ring's feedback and what the
/// lattice operator leaves out of the second-order far shell, the K^(1)
/// rests it drops among it) carries at least COMPLETION_SHARE is completed.
/// Measured (three lasers at B = 0 and 1 mG, p(1 s)): at the first
/// checkpoint the Krylov part is 1e-11 of bounds of 7.1-7.4e-3, the ring's
/// part 3.8-4.3e-3 and the K^(1) rests' 1.1-1.2e-2 (signed parts, partly
/// cancelling); a trajectory's first checkpoint is the Krylov residual's
/// (2e-1 against 2-8e-3). Without the share, completing wherever a source
/// reaches COMPLETION_FLOOR, the same three lasers at 20 us restarted as
/// well: twice the wall time (1.6 -> 3.5 s), the error at 1 mG 4.5e-4 ->
/// 1.6e-3.
const COMPLETION_SHARE:f64=0.5;
/// A source of the slow part joins the lattice while its part (bound units)
/// is at least COMPLETION_FLOOR x tolerance: the K^(1) rests, kept when the
/// operator drops them and the part of what it leaves out reaches the floor,
/// and the ring charges by decreasing attribution while the
/// ring's part they leave (its unpromoted share of the attribution) reaches
/// it. Measured: K^(1) rests and ring each alone fail (both lattices' means
/// stayed 1.3-2.0e-3 off with either change alone, 4e-5 and 1.7e-4 with
/// both); the floor on the ring left the decisions of the 20 completing
/// runs of the necessity review unchanged (every ring part at least
/// 7.2e-3), and every run without an ablation promotes one pair carrying
/// its whole attribution, as the earlier rule (90% of the attribution) did.
const COMPLETION_FLOOR:f64=0.25;
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
    /// The runs of the request that execute at once (a full request's
    /// trajectory and mean: 2), which share the memory budget.
    pub runs:usize,
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
/// rests' and the ring's parts (bound units), the ring charges' attribution
/// shares, and whether the run restarted on the completed lattice.
#[derive(Clone,Debug)]
pub struct Completion{
    pub promoted:Vec<Charge>,pub explicit:ExplicitSet,pub keep_k1:bool,pub decided_at:usize,
    pub bound:f64,pub slow_part:f64,pub k1_part:f64,pub ring_part:f64,
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
fn slow_pairs(model:&Model,space:&Space,vectors:&super::krylov::Vectors,classes:&[f64],tolerance:f64,lattice:&mut Lattice<'_>)->Result<Vec<(SlowRate,f64)>,String>{
    let chosen=model.slow_candidates(CANDIDATE_WEIGHT*tolerance);
    if chosen.is_empty(){return Ok(Vec::new());}
    let parts:Vec<Vec<f64>>=chosen.iter().flat_map(|&(j,_)|{let y=model.ritz_coordinates(j);[y.iter().map(|z|z.re).collect(),y.iter().map(|z|z.im).collect()]}).collect();
    let basis:Vec<&[C]>=space.basis.iter().map(|v|v.as_slice()).collect();let images:Vec<&[C]>=space.images.iter().map(|v|v.as_slice()).collect();
    let w=vectors.combinations(&basis,&parts)?;
    let aw=if space.images.first().is_some_and(|v|v.is_empty()){w.iter().map(|x|{let mut y=vec![ZERO;x.len()];lattice.apply(x,&mut y).map(|_|y)}).collect::<Result<Vec<_>,_>>()?}else{vectors.combinations(&images,&parts)?};
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
/// rests the lattice leaves out) is at least COMPLETION_SHARE of it, each
/// source is estimated alone and joins while its part reaches
/// COMPLETION_FLOOR x tolerance: the K^(1) rests the lattice leaves out join
/// the operator, and the ring charges join P with their mirrors by
/// decreasing attribution (|a_m| |B y_r| over the kernel and slow modes,
/// from the ring's feedback) while the ring's part they leave reaches the
/// floor. The run then restarts once on the completed lattice
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

/// (research, ATOM_DIAG_EXACT=path at the checkpoints listed in ATOM_DIAG_AT)
/// The model's exact error on its own lattice: x(t) = e^{tA} x0 by Pade (8,8)
/// steps (ATOM_DIAG_SUB per grid interval, default 50; shifted solves to
/// 1e-11 with the run's preconditioner) at the grid times up to ATOM_DIAG_TMAX
/// (default 0.31 us), less the model's field W u(t); per lattice mode its
/// populations Re e^{i nu_q t} E_q[ii]. One JSON line per checkpoint.
fn basis_cols(space:&Space)->Vec<&[C]>{space.basis.iter().map(|v|v.as_slice()).collect()}
#[allow(clippy::too_many_arguments)]
/// (research, S4) Acceptance by convergence: per output (trajectory sample,
/// horizon, steady state), Delta2 = |x_k - x_k-1| and Delta1 = |x_k-1 - x_k-2|
/// over the last three checkpoints, r = Delta2 / Delta1, the estimate
/// Delta2 r / (1 - r); r at or above R_LIMIT is stagnation, reported as at
/// least twice the tolerance (not accepted); Delta2 below FLOOR x tolerance
/// is converged to roundoff. Fewer than three levels: unknown (1). The
/// estimates are returned over ACCEPTANCE_FACTOR (the pipeline multiplies).
const R_LIMIT:f64=0.8;const FLOOR:f64=1e-3;
fn convergence_estimate(h:&[(Vec<Vec<f64>>,Vec<Vec<f64>>,Vec<f64>)],groups:&[Vec<usize>],tolerance:f64,steady:bool)->ErrorEstimate{
    let k=h.len();let unknown=1./ACCEPTANCE_FACTOR;
    // One output's estimates from its components' changes (the trajectory
    // is one output: its ratio from the largest changes over the grid).
    let output=|d1:&[f64],d2:&[f64]|->Vec<f64>{
        let(m1,m2)=(d1.iter().copied().fold(0.,f64::max),d2.iter().copied().fold(0.,f64::max));
        let converged=m2<=FLOOR*tolerance;let r=if m1>0.{m2/m1}else{f64::INFINITY};
        d2.iter().map(|&x|(if converged{x}else if r<R_LIMIT{x*r/(1.-r)}else{(x*R_LIMIT/(1.-R_LIMIT)).max(2.*tolerance)})/ACCEPTANCE_FACTOR).collect()};
    let one=|a:&[f64],b:&[f64],c:&[f64]|->f64{output(&[ledger::metric(b,a,groups)],&[ledger::metric(c,b,groups)])[0]};
    let(t,hz,s)=(&h[k-1].0,&h[k-1].1,&h[k-1].2);
    if k<3{return ErrorEstimate{trajectory:vec![unknown;t.len()],horizons:vec![unknown;hz.len()],steady:steady.then_some(unknown),..Default::default()};}
    let(a,b)=(&h[k-3],&h[k-2]);
    let d1:Vec<f64>=(0..t.len()).map(|i|ledger::metric(&b.0[i],&a.0[i],groups)).collect();let d2:Vec<f64>=(0..t.len()).map(|i|ledger::metric(&t[i],&b.0[i],groups)).collect();
    ErrorEstimate{trajectory:output(&d1,&d2),horizons:(0..hz.len()).map(|i|one(&a.1[i],&b.1[i],&hz[i])).collect(),
        steady:steady.then(||one(&a.2,&b.2,s)),..Default::default()}
}
/// (research, S3) E^-1 coordinates of blocks, column-major d x count.
fn eig_coefficients(eigen:&BlockEigenbasis,blocks:&[&[C]])->Result<Vec<C>,String>{eigen.coefficients(blocks)}
/// (research, S2) e^A for a real column-major r x r matrix: scaling and
/// squaring with the (13,13) Pade approximant (Higham 2005).
fn expm_real(lapack:&Lapack,a:&[f64],r:usize)->Result<Vec<f64>,String>{
    const B:[f64;14]=[64764752532480000.,32382376266240000.,7771770303897600.,1187353796428800.,129060195264000.,10559470521600.,670442572800.,33522128640.,1323241920.,40840800.,960960.,16380.,182.,1.];
    let norm=(0..r).map(|j|(0..r).map(|i|a[i+r*j].abs()).sum::<f64>()).fold(0.,f64::max);
    let s=if norm>5.371920351148152{(norm/5.371920351148152).log2().ceil()as i32}else{0};
    let scale=0.5_f64.powi(s);let a:Vec<f64>=a.iter().map(|x|x*scale).collect();
    let mm=|x:&[f64],y:&[f64]|->Result<Vec<f64>,String>{let mut z=vec![0.;r*r];lapack.gemm(false,false,r,r,r,1.,x,r,y,r,0.,&mut z,r)?;Ok(z)};
    let a2=mm(&a,&a)?;let a4=mm(&a2,&a2)?;let a6=mm(&a2,&a4)?;
    let comb=|c6:f64,c4:f64,c2:f64,c0:f64|->Vec<f64>{let mut z:Vec<f64>=(0..r*r).map(|e|c6*a6[e]+c4*a4[e]+c2*a2[e]).collect();for i in 0..r{z[i+r*i]+=c0;}z};
    let inner_u=mm(&a6,&comb(B[13],B[11],B[9],0.))?;let mut tu=comb(B[7],B[5],B[3],B[1]);for(x,y)in tu.iter_mut().zip(&inner_u){*x+=y;}
    let u=mm(&a,&tu)?;
    let inner_v=mm(&a6,&comb(B[12],B[10],B[8],0.))?;let mut v=comb(B[6],B[4],B[2],B[0]);for(x,y)in v.iter_mut().zip(&inner_v){*x+=y;}
    let p:Vec<f64>=v.iter().zip(&u).map(|(x,y)|x+y).collect();let q:Vec<f64>=v.iter().zip(&u).map(|(x,y)|x-y).collect();
    let(lu,pivots)=lapack.factor(&q,r)?;let mut e=lapack.solve_matrix_factored(&lu,&pivots,&p,r)?;
    for _ in 0..s{e=mm(&e,&e)?;}
    Ok(e)
}
#[allow(clippy::too_many_arguments)]
fn diag_exact(path:&str,prepared:&LatticePrepared,frame:&SwFrame<'_>,lattice:&mut Lattice<'_>,ops:&super::krylov::Vectors,space:&Space,model:&Model,galerkin:&Projection<'_>,times:&[f64],initial:&[C],solves:usize)->Result<(),String>{
    let tmax=std::env::var("ATOM_DIAG_TMAX").ok().and_then(|v|v.parse::<f64>().ok()).unwrap_or(0.31);
    let sub=std::env::var("ATOM_DIAG_SUB").ok().and_then(|v|v.parse::<usize>().ok()).unwrap_or(50);
    let times:Vec<f64>=times.iter().copied().filter(|&t|t<=tmax).collect();
    if times.len()<2{return Ok(());}
    let clock=Instant::now();
    let(n,size,modes,nu,centre,charges)={let op=lattice.operator();(op.n,op.dimension(),op.modes.len(),op.nu.clone(),op.centre,op.modes.clone())};let d=n*n;
    // Pade (8,8): the four roots z (Im z > 0) of the denominator and their residues.
    let roots:[(C,C);4]=[
        (C{re:5.67796789779552302e+00,im:1.27078225972098888e+01},C{re:-1.65193072544915026e+02,im:-2.19157995335393480e+02}),
        (C{re:8.73657843440328818e+00,im:8.82888500094301421e+00},C{re:4.26599883763159505e+03,im:-2.19879464125002841e+02}),
        (C{re:1.04096815812761267e+01,im:5.23235030528406142e+00},C{re:-1.46726502179980143e+04,im:1.30835546536328566e+04}),
        (C{re:1.11757720865251144e+01,im:1.73522889070729636e+00},C{re:1.06438444528428354e+04,im:-3.82471699398737619e+04})];
    let h=(times[1]-times[0])/sub as f64;
    let mut x=vec![ZERO;size];x[centre*d..(centre+1)*d].copy_from_slice(initial);
    let zero=vec![ZERO;size];let mut inner=0usize;
    let mut exact=vec![x.clone()];
    static EXACT:std::sync::Mutex<Option<(Vec<f64>,usize,Vec<Vec<C>>)>>=std::sync::Mutex::new(None);
    let cached=EXACT.lock().ok().and_then(|g|g.as_ref().filter(|(t,s,_)|*t==times&&*s==size).map(|(_,_,e)|e.clone()));
    // (research, S2) ATOM_DIAG_EXACT_CACHE=path: the exact fields on disk
    // (f64: times, size, the times, then every field's re/im).
    let cache_path=std::env::var("ATOM_DIAG_EXACT_CACHE").ok();
    let from_disk=cached.is_none().then(||cache_path.as_ref().and_then(|p|std::fs::read(p).ok())).flatten().and_then(|bytes|{
        let v:Vec<f64>=bytes.chunks_exact(8).map(|c|f64::from_le_bytes(c.try_into().unwrap())).collect();
        if v.len()<2{return None;}let(count,length)=(v[0]as usize,v[1]as usize);
        if count!=times.len()||length!=size||v.len()!=2+count+2*count*length||v[2..2+count].iter().zip(&times).any(|(a,b)|(a-b).abs()>1e-12){return None;}
        Some((0..count).map(|i|{let o=2+count+2*i*length;(0..length).map(|k|C{re:v[o+2*k],im:v[o+2*k+1]}).collect::<Vec<C>>()}).collect::<Vec<Vec<C>>>())});
    let cached=cached.or(from_disk);
    if let Some(e)=cached{exact=e;}else{
    for _ in 1..times.len(){
        for _ in 0..sub{
            // x <- x + sum over the pairs 2 Re[(r/h) (A - z/h)^-1 x] (R(inf) = (-1)^8 = 1).
            let mut y=x.clone();
            for &(z,res)in &roots{
                let(w,steps,_)=lattice.solve(Pole::Complex(C{re:z.re/h,im:z.im/h}),&[x.clone(),zero.clone()],1e-11)?;inner+=steps;
                let(a,b)=(2.*res.re/h,2.*res.im/h);
                for((v,p),q)in y.iter_mut().zip(&w[0]).zip(&w[1]){v.re+=a*p.re-b*q.re;v.im+=a*p.im-b*q.im;}
            }
            x=y;
        }
        exact.push(x.clone());
    }
    if let Some(p)=&cache_path{
        let mut v:Vec<f64>=vec![times.len()as f64,size as f64];v.extend(&times);for f in &exact{for z in f{v.push(z.re);v.push(z.im);}}
        let bytes:Vec<u8>=v.iter().flat_map(|x|x.to_le_bytes()).collect();std::fs::write(p,bytes).map_err(|e|e.to_string())?;
    }
    }
    if let Ok(mut g)=EXACT.lock(){*g=Some((times.clone(),size,exact.clone()));}
    let basis:Vec<&[C]>=space.basis.iter().map(|v|v.as_slice()).collect();
    // The growing modes: the projected ones' initial mismatch E(0) = x0 - W u(0)
    // (physical trace norm / 2: a contraction's bound on its contribution at
    // every time) and the clipped ones' change of the model trajectory,
    // sum a (e^{theta t} - e^{theta' t}) W s, at the diagnosed times.
    let growing={
        let r=model.r;let xm=ops.combinations(&basis,&[model.state(0.)])?.remove(0);
        let mut e0=vec![ZERO;d];for q in 0..modes{for k in 0..d{e0[k].re-=xm[q*d+k].re;e0[k].im-=xm[q*d+k].im;}}
        for k in 0..d{e0[k].re+=initial[k].re;e0[k].im+=initial[k].im;}
        let mut cm=vec![ZERO;d];for i in 0..n{for j in 0..n{cm[i+n*j]=e0[i*n+j];}}
        let(values,_)=prepared.lapack.complex_eigen_right(&cm,n)?;let half=0.5*values.iter().map(|z|z.re.abs()).sum::<f64>();
        let clipped:Vec<usize>=(0..r).filter(|&j|!model.kernel[j]&&!model.projected_out[j]&&model.theta[j].re>0.&&model.used_theta[j].re==0.).collect();
        let change:Vec<f64>=times.iter().map(|&t|->Result<f64,String>{
            if clipped.is_empty(){return Ok(0.);}
            let u:Vec<f64>=(0..r).map(|i|clipped.iter().fold(0.,|acc,&j|{let a=model.used_coefficients[j];
                let dz=C{re:cexp(model.theta[j],t).re-cexp(model.used_theta[j],t).re,im:cexp(model.theta[j],t).im-cexp(model.used_theta[j],t).im};acc+mul(model.vectors[i*r+j],mul(dz,a)).re})).collect();
            let v=ops.combinations(&basis,&[u])?.remove(0);
            let p:Vec<f64>=(0..n).map(|i|(0..modes).map(|q|{let z=v[q*d+i*n+i];let(c,s)=((nu[q]*t).cos(),(nu[q]*t).sin());c*z.re-s*z.im}).sum()).collect();
            Ok(p.iter().map(|x|x.abs()).fold(0.,f64::max))}).collect::<Result<_,_>>()?;
        format!(",\"projected\":{},\"clipped\":{},\"growthCharge\":{:?},\"projectedTraceNorm\":{half:?},\"clippedChange\":{change:?}",model.projected,model.clipped,model.growth_charge)};
    // (research, S2) Best approximations, stable evaluation, conditioning.
    let s2={
        let lapack=&prepared.lapack;let r=model.r;
        let gcol=super::reduced::transpose_rows(&model.g,r,r);
        let spacing=times[1]-times[0];
        let e_h=expm_real(lapack,&gcol.iter().map(|x|x*spacing).collect::<Vec<f64>>(),r)?;
        let c0:Vec<f64>=(0..r).map(|i|space.initial.get(i).copied().unwrap_or(0.)).collect();
        let mut u=c0.clone();let mut stable=vec![u.clone()];
        for _ in 1..times.len(){let mut v=vec![0.;r];lapack.gemm(false,false,r,1,r,1.,&e_h,r,&u,r,0.,&mut v,r)?;u=v;stable.push(u.clone());}
        // (research, S3) the Euclidean Galerkin model on the same space.
        let ge=super::reduced::transpose_rows(&galerkin.euclidean_g(),r,r);
        let eh_e=expm_real(lapack,&ge.iter().map(|x|x*spacing).collect::<Vec<f64>>(),r)?;
        let mut u=c0.clone();let mut euclid=vec![u.clone()];
        for _ in 1..times.len(){let mut v=vec![0.;r];lapack.gemm(false,false,r,1,r,1.,&eh_e,r,&u,r,0.,&mut v,r)?;u=v;euclid.push(u.clone());}
        let mut euclid_pops=Vec::new();
        let(_,_,ev)=lapack.eigen(&gcol,r)?;
        let fro=|m:&[f64]|m.iter().map(|x|x*x).sum::<f64>().sqrt();
        let(lu,pivots)=lapack.factor(&ev,r)?;let mut id=vec![0.;r*r];for i in 0..r{id[i+r*i]=1.;}let inv=lapack.solve_matrix_factored(&lu,&pivots,&id,r)?;
        let kappa=fro(&ev)*fro(&inv);
        let mut amplification=Vec::new();let mut coordinate_gap=Vec::new();let mut stable_pops=Vec::new();let mut best_e=Vec::new();let mut best_m=Vec::new();
        let mut c_e_norm=Vec::new();let mut readout_gap=0_f64;
        // (research, S3) Early times (t <= ATOM_DIAG_EARLY, default 0.1 us):
        // the best approximation's and the Galerkin error per lattice mode
        // (populations), and both split over the L0' eigen-coordinates by
        // decay rate |Re lambda| and laboratory frequency |Im lambda - nu_q|
        // (bins: < 1, 1-10, 10-100, 100-1e3, 1e3-1e4, > 1e4 per us), summed
        // |c_j| max_k |v_j[kk]|.
        let early_limit=std::env::var("ATOM_DIAG_EARLY").ok().and_then(|v|v.parse::<f64>().ok()).unwrap_or(0.1);
        let eig=prepared.eigenbasis()?;let eblocks=eig.eigenvector_blocks();
        let pop_of:Vec<f64>=(0..d).map(|j|(0..n).map(|k|magnitude(eblocks[j*d+k*n+k])).fold(0.,f64::max)).collect();
        let bin=|x:f64|->usize{if x<1.{0}else if x<10.{1}else if x<100.{2}else if x<1e3{3}else if x<1e4{4}else{5}};
        let mut early_modes:Vec<[Vec<Vec<f64>>;2]>=Vec::new();let mut early_bins:Vec<[Vec<Vec<f64>>;2]>=Vec::new();
        let solver=model.trajectory(galerkin,&times);
        for(ti,&t)in times.iter().enumerate(){
            let ue=model.state(t);
            let e:Vec<C>=(0..r).map(|j|mul(cexp(model.used_theta[j],t),model.used_coefficients[j])).collect();
            let sum_abs=(0..r).map(|i|(0..r).map(|j|magnitude(mul(model.vectors[i*r+j],e[j]))).sum::<f64>()).fold(0.,f64::max);
            let top=ue.iter().map(|x|x.abs()).fold(0.,f64::max).max(f64::MIN_POSITIVE);
            amplification.push(sum_abs/top);
            coordinate_gap.push(ue.iter().zip(&stable[ti]).map(|(a,b)|(a-b).abs()).fold(0.,f64::max));
            let ce=ops.gram(&basis,&[exact[ti].as_slice()])?;let cm=galerkin.metric_coordinates(lapack,&basis,&exact[ti])?;
            c_e_norm.push(ce.iter().map(|x|x*x).sum::<f64>().sqrt());
            let fields=ops.combinations(&basis,&[stable[ti].clone(),ce,cm,euclid[ti].clone()])?;
            let read=|v:&[C]|->Vec<f64>{(0..n).map(|i|(0..modes).map(|q|{let z=v[q*d+i*n+i];let(c,s)=((nu[q]*t).cos(),(nu[q]*t).sin());c*z.re-s*z.im}).sum()).collect()};
            stable_pops.push(read(&fields[0]));best_e.push(read(&fields[1]));best_m.push(read(&fields[2]));euclid_pops.push(read(&fields[3]));
            let xm=ops.combinations(&basis,&[ue])?.remove(0);let mp=read(&xm);
            if t>0.&&t<=early_limit+1e-12{
                let mut per=[vec![vec![0.;n];modes],vec![vec![0.;n];modes]];let mut bins=[vec![vec![0.;6];6],vec![vec![0.;6];6]];
                for(w,approx)in[&fields[1],&xm].into_iter().enumerate(){
                    for q in 0..modes{
                        let block:Vec<C>=(0..d).map(|k|C{re:exact[ti][q*d+k].re-approx[q*d+k].re,im:exact[ti][q*d+k].im-approx[q*d+k].im}).collect();
                        let(cs,sn)=((nu[q]*t).cos(),(nu[q]*t).sin());
                        for i in 0..n{let z=block[i*n+i];per[w][q][i]=cs*z.re-sn*z.im;}
                        let c=eig.coefficients(&[&block])?;
                        for j in 0..d{let a=magnitude(c[j])*pop_of[j];if a==0.{continue;}
                            bins[w][bin(eig.lambda[j].re.abs())][bin((eig.lambda[j].im-nu[q]).abs())]+=a;}
                    }
                }
                early_modes.push(per);early_bins.push(bins);
            }
            readout_gap=readout_gap.max(mp.iter().zip(&solver[ti]).map(|(a,b)|(a-b).abs()).fold(0.,f64::max));
        }
        let thetas:Vec<[f64;2]>=model.theta.iter().map(|z|[z.re,z.im]).collect();
        let amps:Vec<f64>=model.used_coefficients.iter().map(|z|magnitude(*z)).collect();
        format!(",\"kappaF\":{kappa:?},\"amplification\":{amplification:?},\"coordinateGap\":{coordinate_gap:?},\"readoutGap\":{readout_gap:?},\"exactCoordinateNorm\":{c_e_norm:?},\"stable\":{stable_pops:?},\"bestE\":{best_e:?},\"bestM\":{best_m:?},\"theta\":{thetas:?},\"amplitudes\":{amps:?},\"earlyModes\":{early_modes:?},\"earlyBins\":{early_bins:?},\"euclid\":{euclid_pops:?}")
    };
    let mut rows=Vec::new();let mut exact_pops=Vec::new();let mut model_pops=Vec::new();
    for(ti,&t)in times.iter().enumerate(){
        let xm=ops.combinations(&basis,&[model.state(t)])?.remove(0);
        let per_mode=|v:&[C]|->Vec<Vec<f64>>{(0..modes).map(|q|{let(c,s)=((nu[q]*t).cos(),(nu[q]*t).sin());(0..n).map(|i|{let z=v[q*d+i*n+i];c*z.re-s*z.im}).collect()}).collect()};
        let e:Vec<C>=exact[ti].iter().zip(&xm).map(|(a,b)|C{re:a.re-b.re,im:a.im-b.im}).collect();
        rows.push(per_mode(&e));
        let total=|p:&Vec<Vec<f64>>|->Vec<f64>{(0..n).map(|i|p.iter().map(|r|r[i]).sum()).collect()};
        exact_pops.push(total(&per_mode(&exact[ti])));model_pops.push(total(&per_mode(&xm)));
    }
    // ATOM_DIAG_EXACT_FULL: the error fields themselves, per time and mode its
    // block as [re, im] pairs (the request frame, row-major n x n).
    let full=if std::env::var("ATOM_DIAG_EXACT_FULL").is_ok(){
        let mut parts=Vec::new();
        for(ti,&t)in times.iter().enumerate(){let xm=ops.combinations(&basis,&[model.state(t)])?.remove(0);
            let blocks:Vec<String>=(0..modes).map(|q|{let v:Vec<String>=(0..d).map(|k|{let z=C{re:exact[ti][q*d+k].re-xm[q*d+k].re,im:exact[ti][q*d+k].im-xm[q*d+k].im};format!("[{:?},{:?}]",z.re,z.im)}).collect();format!("[{}]",v.join(","))}).collect();
            parts.push(format!("[{}]",blocks.join(",")));}
        format!(",\"fields\":[{}]",parts.join(","))}else{String::new()};
    // ATOM_DIAG_CLASSES: the error's populations split by the L0' eigen-
    // coordinates' SW class: near resonant (in a pair above the kernels'
    // cap), strongly coupled (a kept |X~| above 0.3), the rest.
    let classes=if std::env::var("ATOM_DIAG_CLASSES").is_ok(){
        let kernels=frame.kernels()?;let basis=prepared.eigenbasis()?;let vectors=basis.eigenvector_blocks();
        let mut class=vec![2u8;d];let mut strength=vec![0_f64;d];
        for st in &kernels.steps{for &(i,j,x)in &st.entries{let a=magnitude(x);strength[i]=strength[i].max(a);strength[j]=strength[j].max(a);}
            for &(i,j)in &st.resonant{class[i]=0;class[j]=0;}}
        for i in 0..d{if class[i]==2&&strength[i]>0.3{class[i]=1;}}
        let counts=[0u8,1,2].map(|c|class.iter().filter(|&&x|x==c).count());
        let mut split=vec![vec![vec![0_f64;n];times.len()];3];
        for(ti,&t)in times.iter().enumerate(){let xm=ops.combinations(&basis_cols(space),&[model.state(t)])?.remove(0);
            for q in 0..modes{
                let block:Vec<C>=(0..d).map(|k|C{re:exact[ti][q*d+k].re-xm[q*d+k].re,im:exact[ti][q*d+k].im-xm[q*d+k].im}).collect();
                let c=basis.coefficients(&[&block])?;let(cs,sn)=((nu[q]*t).cos(),(nu[q]*t).sin());
                for i in 0..d{if !nonzero(c[i]){continue;}let v=&vectors[i*d..(i+1)*d];
                    for k in 0..n{let z=mul(c[i],v[k*n+k]);split[class[i]as usize][ti][k]+=cs*z.re-sn*z.im;}}
            }
        }
        format!(",\"classCounts\":{counts:?},\"classes\":{split:?}")}else{String::new()};
    // The residual integral: |E(t)|_tr <= |E(0)|_tr + int_0^t |r(s)|_tr ds for a
    // trace-norm contraction, r(s) = A W u(s) - W u'(s) read physically
    // (sum_q e^{i nu_q s} r_q), by the trapezoid rule on 40 points per grid
    // interval.
    let residual_integral=if std::env::var("ATOM_DIAG_RESINT").is_err(){Vec::new()}else{
        let lapack=&prepared.lapack;let r=model.r;let images:Vec<&[C]>=space.images.iter().map(|v|v.as_slice()).collect();
        let derivative=|t:f64|->Vec<f64>{let e:Vec<C>=(0..r).map(|j|mul(model.used_theta[j],mul(cexp(model.used_theta[j],t),model.used_coefficients[j]))).collect();
            (0..r).map(|i|(0..r).fold(0.,|acc,j|acc+mul(model.vectors[i*r+j],e[j]).re)).collect()};
        let physical=|v:&[C],t:f64|->Vec<C>{let mut m=vec![ZERO;d];for q in 0..modes{let ph=cexp(C{re:0.,im:nu[q]},t);for k in 0..d{mul_add(&mut m[k],ph,v[q*d+k]);}}m};
        let trace_norm=|m:&[C]|->Result<f64,String>{let mut cm=vec![ZERO;d];for i in 0..n{for j in 0..n{cm[i+n*j]=m[i*n+j];}}
            let(values,_)=lapack.complex_eigen_right(&cm,n)?;Ok(values.iter().map(|z|z.re.abs()).sum())};
        let x0m=ops.combinations(&basis_cols(space),&[model.state(0.)])?.remove(0);
        let mut e0=vec![ZERO;size];e0[centre*d..(centre+1)*d].copy_from_slice(initial);for(a,b)in e0.iter_mut().zip(&x0m){a.re-=b.re;a.im-=b.im;}
        let start=trace_norm(&physical(&e0,0.))?;
        let per=40usize;let dt=(times[1]-times[0])/per as f64;let mut acc=start;let mut out=vec![start];let mut prev:Option<f64>=None;
        for step in 0..=(times.len()-1)*per{let t=step as f64*dt;
            let v=ops.combinations(&images,&[model.state(t)])?.remove(0);let w=ops.combinations(&basis_cols(space),&[derivative(t)])?.remove(0);
            let res:Vec<C>=v.iter().zip(&w).map(|(a,b)|C{re:a.re-b.re,im:a.im-b.im}).collect();
            let norm=trace_norm(&physical(&res,t))?;
            if let Some(p)=prev{acc+=0.5*dt*(p+norm);}prev=Some(norm);
            if step>0&&step%per==0{out.push(acc);}
        }
        out};
    let line=format!("{{\"solves\":{solves},\"columns\":{},\"times\":{times:?},\"h\":{h:?},\"inner\":{inner},\"seconds\":{:?},\"charges\":{charges:?},\"nu\":{nu:?},\"error\":{rows:?},\"exact\":{exact_pops:?},\"model\":{model_pops:?},\"residualIntegral\":{residual_integral:?}{growing}{s2}{full}{classes}}}",
        space.basis.len(),clock.elapsed().as_secs_f64());
    use std::io::Write;let mut f=std::fs::OpenOptions::new().create(true).append(true).open(path).map_err(|e|e.to_string())?;writeln!(f,"{line}").map_err(|e|e.to_string())?;
    Ok(())
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
    // (research) ATOM_REF_SOLVES=N: a run to N solves, never accepted, no plateau.
    let reference=std::env::var("ATOM_REF_SOLVES").ok().and_then(|v|v.parse::<usize>().ok());
    let maximum_solves=reference.unwrap_or(MAXIMUM_SOLVES);
    let clock=Instant::now();
    let mut lattice=Lattice::with_k1_rests(prepared,explicit.sets.clone(),keep_k1)?;
    let ledger=lattice.operator().ledger()?;
    let budget=MEMORY_SHARE*crate::memory::physical_bytes()?;
    let planned=planned_bytes(request.runs,lattice.modes.len());
    if planned>budget{return Err(format!("The lattice of this request has {} modes and would need about {:.1} GB, above the {:.0} GB memory budget: not solved",lattice.modes.len(),planned/1e9,budget/1e9));}
    // The static fold's frequency expansion (K^(0) + i nu K^(1)) holds while
    // its parameter p = nu_max / (|nu_far| - ||L0||) is small: a lattice with
    // p above EXPANSION_LIMIT is not solved, the limit a completion keeps to
    // as well. Measured: p at most 9.2e-4 on the research requests (21 runs);
    // its remainder charged as delta min(t, tau) instead (`ledger::nu_freeze`,
    // tau the slowest relaxation) was 1.3e-2 to 0.45 and made 3 of 6 runs
    // partial, whatever tau.
    if ledger.expansion_parameter>EXPANSION_LIMIT{return Err(format!("Static fold expansion parameter {:e} is above {EXPANSION_LIMIT:e}: the fold's frequency expansion is not small on this lattice, not solved",ledger.expansion_parameter));}
    // 2. The space: translations, the steady column.
    let capacity={let op=lattice.operator();let d=op.n*op.n;
        if std::env::var("ATOM_BASIS_FROM_GUARD").is_ok(){
            // (research, S3) the guard's headroom over the planned request,
            // shared by its runs; a column: basis, image and the estimator's
            // coordinates (about as large again).
            let headroom=((budget-planned)/request.runs.max(1)as f64).max(0.);
            let column=(std::mem::size_of::<C>()*4*d*op.modes.len())as f64;
            let c=((BASIS_BYTES as f64+headroom)/column)as usize;eprintln!("DBG basis capacity {c} columns (budget {:.1} GB, planned {:.1} GB)",budget/1e9,planned/1e9);c
        }else{BASIS_BYTES/(std::mem::size_of::<C>()*2*d*op.modes.len())}};
    let mut space=Space::new(&mut lattice,initial,capacity)?;
    // (research, S3) ATOM_DIAG_ADJOINT: Re <y, A x> against Re <A^H y, x> on
    // random Hermitian-symmetric fields.
    if std::env::var("ATOM_DIAG_ADJOINT").is_ok(){
        let(n,size,canonical,mirror)={let op=lattice.operator();(op.n,op.dimension(),op.canonical.clone(),op.mirror.clone())};
        let field=|seed:u64|->Vec<C>{let mut st=seed;let mut v:Vec<C>=(0..size).map(|_|{st=st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);let a=((st>>11)as f64/(1u64<<53)as f64)-0.5;
            st=st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);let b=((st>>11)as f64/(1u64<<53)as f64)-0.5;C{re:a,im:b}}).collect();mirror_field(&mut v,n,&canonical,&mirror);v};
        let(x,y)=(field(7),field(11));let mut ax=vec![ZERO;size];lattice.apply(&x,&mut ax)?;let ay=lattice.operator().apply_adjoint(&y)?;
        let dot=|a:&[C],b:&[C]|a.iter().zip(b).map(|(u,v)|u.re*v.re+u.im*v.im).sum::<f64>();
        let(l,r)=(dot(&y,&ax),dot(&ay,&x));let scale=dot(&ax,&ax).sqrt()*dot(&y,&y).sqrt();
        eprintln!("DBG adjoint <y,Ax> {l:.12e} <A^H y,x> {r:.12e} relative {:.2e}",(l-r).abs()/scale);
    }
    for s in charge_set::gated_steps(prepared,tolerance){if lattice.operator().index.contains_key(&s){space.activate(&mut lattice,&s)?;}}
    // (research, S3) ATOM_POLE_INFINITY=m: m columns at the pole at infinity
    // (A x0, A^2 x0, ...) right after x0.
    if let Some(m)=std::env::var("ATOM_POLE_INFINITY").ok().and_then(|v|v.parse::<usize>().ok()){for _ in 0..m{let added=space.infinite(&mut lattice)?;eprintln!("DBG infinite pole added {added}");}}
    // (research, S2) ATOM_TRANSLATION_SUMS: also the sums and differences of
    // every two gated steps on the lattice (one of each pair).
    if std::env::var("ATOM_TRANSLATION_SUMS").is_ok(){
        let gated:Vec<Charge>=charge_set::gated_steps(prepared,tolerance).into_iter().filter(|s|lattice.operator().index.contains_key(s)).collect();
        let mut active:Vec<Charge>=space.translations();
        for i in 0..gated.len(){for j in i+1..gated.len(){for sign in[1i32,-1]{
            let c:Charge=gated[i].iter().zip(&gated[j]).map(|(&a,&b)|a+sign*b).collect();
            let m=opposite(&c);let c=if c>m{c}else{m};
            if active.contains(&c)||!lattice.operator().index.contains_key(&c){continue;}
            space.activate(&mut lattice,&c)?;active.push(c);
        }}}
    }
    timing.setup_ms=clock.elapsed().as_secs_f64()*1e3;
    space.solve(&mut lattice,Pole::Zero)?;
    let modes=lattice.modes.clone();
    let clock=Instant::now();
    // (research, S4) ATOM_CONVERGENCE: no error estimator; the history of
    // the checkpoints' outputs.
    let convergence=std::env::var("ATOM_CONVERGENCE").is_ok();let no_images=std::env::var("ATOM_NO_IMAGES").is_ok();
    let mut history:Vec<(Vec<Vec<f64>>,Vec<Vec<f64>>,Vec<f64>)>=Vec::new();
    let mut estimator=if convergence{None}else{Some(UnifiedEstimate::new(EstimateSetup{prepared,lattice:&explicit.sets,outputs:Outputs{times,horizons:request.horizons,steady:request.steady,groups:request.groups},frame,ledger:&ledger,keep_k1})?)};
    timing.estimator_build_ms=clock.elapsed().as_secs_f64()*1e3;
    // The ladder spans the grid's times when there is one (a horizon is
    // served by the steady column): its poles near 1 / T of a 1 s horizon grew
    // spurious near-zero slow Ritz pairs in both metrics (rates 1e-8 to
    // 1e-5, residuals 0.2-3) whose p(1 s) bounds reached 1e1-1e4; without
    // them three, four, five lasers and four without the trap at 20 us were
    // accepted at 16-24 solves (four without the trap partial before) and
    // five lasers at 10 us / 2000 samples reached 7.0e-3 against 2.3e-2.
    // (research, S3) ATOM_POLES_TRANSLATED: lattice candidates only on the
    // translated modes.
    let translated_only=std::env::var("ATOM_POLES_TRANSLATED").is_ok();
    let shifts_of=|steps:&[Charge],lattice:&Lattice<'_>|->Vec<f64>{let op=lattice.operator();
        steps.iter().filter_map(|s|op.index.get(s).map(|&q|op.nu[q])).flat_map(|v|[v,-v]).collect()};
    let shifts:Vec<f64>=if translated_only{std::iter::once(0.).chain(shifts_of(&space.translations(),&lattice)).collect()}else{lattice.operator().nu.clone()};
    let mut points=space::ladder(eigen,initial,&shifts,t1,times.last().copied().unwrap_or(longest),tolerance)?;
    let untranslated_lattice=std::env::var("ATOM_LATTICE_POLES_UNTRANSLATED").is_ok();
    let centre_poles=space::ladder(eigen,initial,&[0.],t1,times.last().copied().unwrap_or(longest),tolerance)?;
    if std::env::var("ATOM_DIAG_LADDER").is_ok(){eprintln!("DBG ladder {} points: {}",points.len(),points.iter().map(|p|{let z=p.sigma();format!("{:.3e}{:+.4e}i",z.re,z.im)}).collect::<Vec<_>>().join(" "));}let mut alive=vec![true;points.len()];
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
    // (research, S3) ATOM_TRANSLATION_APRIORI: the families' amplitudes from
    // x0 alone - the first- and second-order transfers X~_s c0 and
    // X~_s2 X~_s1 c0 (the slip each lattice mode's family starts with),
    // propagated by L0''s decays e^{lambda_j t} to the output times, read as
    // populations (a pair's weight 2) - against GATE x tolerance; the
    // translations are activated before the first solve.
    if std::env::var("ATOM_TRANSLATION_APRIORI").is_ok()&&!times.is_empty(){
        let kernels=frame.kernels()?;let rows=eigen.population_rows();
        let(n,d,centre,charges,index)={let op=lattice.operator();(op.n,op.n*op.n,op.centre,op.modes.clone(),op.index.clone())};
        let c0=eigen.coefficients(&[initial])?;
        // bound: magnitudes (a bound, no cancellation), the near-resonant
        // pairs (|X~| above the cap) counted as a full transfer.
        let bound=std::env::var("ATOM_TRANSLATION_APRIORI").map(|v|v=="bound").unwrap_or(false);
        let apply=|s:usize,c:&[C]|->Vec<C>{let mut o=vec![ZERO;d];
            if bound{for &(i,j,x)in &kernels.steps[s].entries{o[i].re+=magnitude(x)*magnitude(c[j]);}for &(i,j)in &kernels.steps[s].resonant{o[i].re+=magnitude(c[j]);}}
            else{kernels.subtract_kernel(s,c,&mut o);for v in &mut o{v.re=-v.re;v.im=-v.im;}}o};
        let mut slip:Vec<Vec<C>>=vec![vec![ZERO;d];charges.len()];
        let first:Vec<(usize,Vec<C>)>=kernels.steps.iter().enumerate().filter_map(|(s,st)|index.get(&st.charge).map(|&q|(s,q))).map(|(s,q)|{let v=apply(s,&c0);for(a,b)in slip[q].iter_mut().zip(&v){add(a,*b);}(q,v)}).collect();
        for(q1,v1)in &first{for(s2,st2)in kernels.steps.iter().enumerate(){
            let Ok(target)=charge_plus(&charges[*q1],&st2.charge)else{continue};let Some(&q2)=index.get(&target)else{continue};if q2==centre{continue;}
            let v=apply(s2,v1);for(a,b)in slip[q2].iter_mut().zip(&v){add(a,*b);}}}
        let sample:Vec<f64>={let mut idx:Vec<usize>=Vec::new();let mut i=1usize;while i<times.len(){idx.push(i);i=(i*3/2).max(i+1);}idx.into_iter().map(|i|times[i]).collect()};
        let gate=charge_set::GATE*tolerance;
        for q in 0..charges.len(){
            if q==centre{continue;}
            let mut amplitude=0_f64;
            for &t in &sample{let decayed:Vec<C>=(0..d).map(|j|mul(slip[q][j],cexp(C{re:eigen.lambda[j].re.min(0.),im:eigen.lambda[j].im},t))).collect();
                for i in 0..n{let mut v=ZERO;let mut b=0.;for j in 0..d{mul_add(&mut v,rows[i+n*j],decayed[j]);b+=magnitude(rows[i+n*j])*magnitude(decayed[j]);}amplitude=amplitude.max(2.*if bound{b}else{magnitude(v)});}}
            if std::env::var("ATOM_DIAG_APRIORI").is_ok()&&amplitude>0.1*gate{eprintln!("DBG a-priori {:?} {amplitude:.2e}",charges[q]);}
            let c=charges[q].clone();let m=opposite(&c);let active=space.translations();
            if amplitude<=gate||active.contains(&c)||active.contains(&m){continue;}
            space.activate(&mut lattice,&c)?;eprintln!("DBG translation {c:?} a-priori amplitude {amplitude:.2e}");
        }
        if translated_only{
            let all=space.translations();let sh:Vec<f64>=std::iter::once(0.).chain(shifts_of(&all,&lattice)).collect();
            points=space::ladder(eigen,initial,&sh,t1,times.last().copied().unwrap_or(longest),tolerance)?;alive=vec![true;points.len()];
            eprintln!("DBG ladder {} points",points.len());
        }
    }
    let(mut score,mut solved_at,mut remaining)=(vec![0.;points.len()],vec![None::<f64>;points.len()],vec![0.;points.len()]);
    // The first checkpoint's completion decision when it changed nothing.
    let mut decided:Option<Completion>=None;
    loop{
        let live:Vec<usize>=(0..points.len()).filter(|&i|alive[i]).collect();
        let exhausted=live.is_empty()||space.solves()>=maximum_solves||space.full();
        if exhausted||space.solves()%CHECKPOINT==0||(times.is_empty()&&space.solves()==1){
            // 3. The checkpoint: model, outputs, estimate, bounds.
            let clock=Instant::now();
            if no_images{galerkin.ingest_adjoint(&space.basis,&mut lattice)?;}else{galerkin.ingest(&space.basis,&space.images)?;}
            let model=galerkin.model(lapack,&space.initial,longest,tolerance)?;timing.model_ms+=clock.elapsed().as_secs_f64()*1e3;
            if let Ok(path)=std::env::var("ATOM_DIAG_EXACT"){
                let at:Vec<usize>=std::env::var("ATOM_DIAG_AT").ok().map(|v|v.split(',').filter_map(|x|x.trim().parse().ok()).collect()).unwrap_or_default();
                if at.contains(&space.solves()){diag_exact(&path,prepared,frame,&mut lattice,&ops,&space,&model,&galerkin,times,initial,space.solves())?;}
            }
            timing.projection_ms=galerkin.milliseconds;timing.images_ms=space.images_ms;
            let outputs_clock=Instant::now();
            let mut trajectory=model.trajectory(&galerkin,times);
            if let Some(first)=trajectory.first_mut(){*first=(0..n).map(|i|initial[i*n+i].re).collect();}
            let horizons:Vec<Vec<f64>>=request.horizons.iter().map(|&t|model.horizon(&galerkin,t)).collect();
            let steady=model.steady_populations(&galerkin);
            timing.outputs_ms+=outputs_clock.elapsed().as_secs_f64()*1e3;
            let estimate_clock=Instant::now();
            // (research, S3) ATOM_DIAG_MEMORY: the footprint per checkpoint.
            if std::env::var("ATOM_DIAG_MEMORY").is_ok(){let(fast,pool)=estimator.as_ref().map_or((0,0),|e|e.stored_bytes());let(private,peak)=crate::memory::private_bytes();
                let field=16*space.basis.first().map_or(0,|v|v.len());
                eprintln!("DBG memory solves {} columns {} krylov {} translations {} | basis {:.0} MB images {:.0} MB metric {:.0} MB estimator fast {:.0} MB pool {:.0} MB | process {:.0} MB peak {:.0} MB",space.solves(),space.basis.len(),space.krylov_count(),space.translations().len(),(field*space.basis.len())as f64/1e6,(field*space.images.len())as f64/1e6,galerkin.stored_bytes()as f64/1e6,fast as f64/1e6,pool as f64/1e6,private/1e6,peak/1e6);}
            let mut estimate=if convergence{history.push((trajectory.clone(),horizons.clone(),steady.clone()));convergence_estimate(&history,request.groups,tolerance,request.steady)}else{estimator.as_mut().unwrap().estimate(&ReducedView{basis:&space.basis,images:&space.images,r:model.r,vectors:&model.vectors,
                used_theta:&model.used_theta,used_coefficients:&model.used_coefficients,kernel:&model.kernel,projected:&model.projected_out},&mut lattice)?};
            timing.estimate_ms+=estimate_clock.elapsed().as_secs_f64()*1e3;
            // The growth charge on every output.
            for v in estimate.trajectory.iter_mut().chain(estimate.horizons.iter_mut()).chain(estimate.steady.iter_mut()){*v+=model.growth_charge;}
            // The slowest relaxation.
            let pairs=slow_pairs(&model,&space,&ops,&classes,tolerance,&mut lattice)?;
            // The outputs with their bounds.
            let answer={
                let outputs_clock=Instant::now();
                let projected:Vec<Vec<f64>>=trajectory.iter().map(|p|simplex(p)).collect();
                let changes:Vec<f64>=trajectory.iter().zip(&projected).map(|(a,b)|ledger::metric(a,b,request.groups)).collect();
                let sample_errors:Vec<f64>=estimate.trajectory.iter().zip(&changes).map(|(e,p)|ACCEPTANCE_FACTOR*e+p).collect();
                let field=|u:&[f64],images:bool|->Result<Vec<C>,String>{
                    let source:Vec<&[C]>=if images{space.images.iter().map(|v|v.as_slice()).collect()}else{space.basis.iter().map(|v|v.as_slice()).collect()};
                    Ok(ops.combinations(&source,&[u.to_vec()])?.remove(0))};
                let steady_field=field(&model.steady,false)?;
                let steady_image=if no_images{let mut y=vec![ZERO;steady_field.len()];lattice.apply(&steady_field,&mut y)?;y}else{field(&model.steady,true)?};
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
                answer.accepted=answer.largest()<=tolerance&&reference.is_none();
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
            if completion.is_none()&&checkpoints.len()==1&&!convergence{
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
                    let floor=COMPLETION_FLOOR*tolerance;
                    // The part of what the operator leaves out, when it drops K^(1)
                    // rests (keeping them is what completes that source).
                    let drops=lattice.operator().drops_k1_rests();
                    let estimator=estimator.as_mut().unwrap();let k1_part=if drops{pick(&estimator.estimate_alone(&view,&mut lattice,Source::K1)?)}else{0.};
                    let keep=drops&&k1_part>=floor;
                    // The ring charges by attribution, a pair together (the canonical
                    // member, the larger charge, stands for it), and the ring's part.
                    let mut pairs:BTreeMap<Charge,f64>=BTreeMap::new();
                    for(c,a)in estimator.ring_attribution(&view)?{let m=opposite(&c);let key=if c>m{c}else{m};*pairs.entry(key).or_default()+=a;}
                    let total:f64=pairs.values().sum();
                    let mut ranked:Vec<(Charge,f64)>=pairs.into_iter().collect();ranked.sort_by(|a,b|b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                    let ring_part=if total>0.{pick(&estimator.estimate_alone(&view,&mut lattice,Source::Ring)?)}else{0.};
                    let mut promoted=Vec::new();let mut carried=0.;
                    if total>0.{for(c,a)in &ranked{if ring_part*(1.-carried/total)<floor{break;}promoted.push(c.clone());carried+=a;}}
                    // The completed lattice must fit the memory budget and keep the
                    // fold's frequency expansion small (its parameter scales with the
                    // largest |nu_q|): else no charge is promoted.
                    let mut grown=None;
                    if !promoted.is_empty(){
                        let set=charge_set::completed(prepared,&explicit,&promoted)?;
                        let nu_max=|modes:&[Charge]|->Result<f64,String>{modes.iter().map(|q|prepared.nu(q).map(f64::abs)).try_fold(0_f64,|a,v|v.map(|v|a.max(v)))};
                        let(before,after)=(nu_max(&explicit.sets.modes())?,nu_max(&set.sets.modes())?);
                        let expansion=if before>0.{ledger.expansion_parameter*after/before}else{ledger.expansion_parameter};
                        if planned_bytes(request.runs,set.sets.modes().len())>budget||expansion>EXPANSION_LIMIT{promoted.clear();}else{grown=Some(set);}
                    }
                    let attribution:Vec<(Charge,f64)>=ranked.iter().map(|(c,a)|(c.clone(),if total>0.{a/total}else{0.})).collect();
                    let restarted=keep||!promoted.is_empty();
                    let decision=Completion{promoted,explicit:grown.unwrap_or_else(||explicit.clone()),keep_k1:keep,decided_at:space.solves(),bound,slow_part:part,k1_part,ring_part,attribution,restarted};
                    if restarted{return Ok(Err((decision,checkpoints)));}
                    decided=Some(decision);
                }
            }
            // 4. The stopping rule on the bound (a checkpoint within the
            // tolerance is accepted).
            progress.push(answer.largest());
            let last=progress.len()-1;
            let plateau=reference.is_none()&&last>=PLATEAU&&{
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
            // (research, S2) ATOM_TRANSLATION_AMPLITUDE: a translation along
            // every lattice mode whose population amplitude in the reduced
            // solution exceeds GATE x tolerance (the translation gate's
            // principle, measured), activated retroactively (copies, no solves).
            let mut grown=false;let before_translations=space.translations();
            // (research, S3) ATOM_TRANSLATION_TRANSFER: the a-priori trigger,
            // the first-order transfer of the reduced solution's blocks into
            // every untranslated mode.
            if std::env::var("ATOM_TRANSLATION_TRANSFER").is_ok()&&!times.is_empty(){
                let kernels=frame.kernels()?;let rows=eigen.population_rows();
                let(n,d,centre,charges,nu,index)={let op=lattice.operator();(op.n,op.n*op.n,op.centre,op.modes.clone(),op.nu.clone(),op.index.clone())};
                let basis_refs:Vec<&[C]>=space.basis.iter().map(|v|v.as_slice()).collect();
                let sample:Vec<f64>={let mut idx:Vec<usize>=Vec::new();let mut i=1usize;while i<times.len(){idx.push(i);i=(i*3/2).max(i+1);}idx.into_iter().map(|i|times[i]).collect()};
                let fields=ops.combinations(&basis_refs,&sample.iter().map(|&t|model.state(t)).collect::<Vec<_>>())?;
                let gate=charge_set::GATE*tolerance;
                let mut amplitude=vec![0_f64;charges.len()];
                for(f,&t)in fields.iter().zip(&sample){
                    let blocks:Vec<&[C]>=(0..charges.len()).map(|q|&f[q*d..(q+1)*d]).collect();let c=eig_coefficients(eigen,&blocks)?;
                    let active=space.translations();
                    for q in 0..charges.len(){
                        if q==centre||active.contains(&charges[q])||active.contains(&opposite(&charges[q])){continue;}
                        let mut est=vec![ZERO;d];
                        for(s,st)in kernels.steps.iter().enumerate(){
                            let Ok(src)=charge_minus(&charges[q],&st.charge)else{continue};let Some(&b)=index.get(&src)else{continue};
                            kernels.subtract_kernel(s,&c[b*d..(b+1)*d],&mut est);}
                        let ph=cexp(C{re:0.,im:nu[q]},t);
                        for i in 0..n{let mut v=ZERO;for j in 0..d{mul_add(&mut v,rows[i+n*j],est[j]);}amplitude[q]=amplitude[q].max(2.*mul(ph,v).re.abs());}
                    }
                }
                for q in 0..charges.len(){
                    if q==centre||amplitude[q]<=gate{continue;}
                    let c=charges[q].clone();let m=opposite(&c);
                    let active=space.translations();if active.contains(&c)||active.contains(&m){continue;}
                    space.activate(&mut lattice,&c)?;grown=true;
                    eprintln!("DBG translation {c:?} transfer amplitude {:.2e} at {} solves",amplitude[q],space.solves());
                }
            }
            if std::env::var("ATOM_TRANSLATION_AMPLITUDE").is_ok()&&!times.is_empty(){
                let sample:Vec<f64>=times.iter().copied().step_by((times.len()/100).max(1)).collect();
                let amplitudes=model.mode_amplitudes(&galerkin,&sample);
                let gate=charge_set::GATE*tolerance;
                let(centre,charges)={let op=lattice.operator();(op.centre,op.modes.clone())};
                for(q,a)in amplitudes{
                    if q==centre||a<=gate{continue;}
                    let c=charges[q].clone();let m=opposite(&c);
                    let active=space.translations();if active.contains(&c)||active.contains(&m){continue;}
                    space.activate(&mut lattice,&c)?;grown=true;
                    eprintln!("DBG translation {c:?} amplitude {a:.2e} at {} solves",space.solves());
                }
            }
            if grown&&translated_only{
                // The new translations' lattice candidates join the ladder.
                let fresh:Vec<Charge>=space.translations().into_iter().filter(|c|!before_translations.contains(c)).collect();
                let extra:Vec<Pole>=space::ladder(eigen,initial,&shifts_of(&fresh,&lattice),t1,times.last().copied().unwrap_or(longest),tolerance)?
                    .into_iter().filter(|p|matches!(p,Pole::Complex(_))).collect();
                let added=extra.len();points.extend(extra);alive.extend(vec![true;added]);score.extend(vec![0.;added]);solved_at.extend(vec![None;added]);remaining.extend(vec![0.;added]);
                eprintln!("DBG ladder +{added} points ({} in all)",points.len());
            }
            if !grown{current=Some((space.solves(),model));}
        }
        // The next pole by its expected gain.
        let clock=Instant::now();
        let model=match current.take(){Some((solves,model))if solves==space.solves()=>model,_=>{
            if no_images{galerkin.ingest_adjoint(&space.basis,&mut lattice)?;}else{galerkin.ingest(&space.basis,&space.images)?;}galerkin.model(lapack,&space.initial,longest,tolerance)?}};
        let xis:Vec<C>=live.iter().map(|&i|points[i].sigma()).collect();
        let mut apply=|x:&[C]|->Result<Vec<C>,String>{let mut y=vec![ZERO;x.len()];lattice.apply(x,&mut y)?;Ok(y)};
        let transform=model.error_transform(&galerkin,&space.basis,&space.images,eigen,lapack,&xis,if no_images{Some(&mut apply)}else{None})?;
        for(&i,p)in live.iter().zip(&transform){
            score[i]=points[i].sigma().re*p.iter().map(|z|magnitude(*z)).fold(0.,f64::max);
            if let Some(before)=solved_at[i].take(){remaining[i]=if before>0.{(score[i]/before).min(1.)}else{1.};}
        }
        let gain=|i:usize|->f64{score[i]*(1.-remaining[i])/points[i].copies()as f64};
        let mut ranked:Vec<(usize,f64)>=live.iter().map(|&i|(i,gain(i))).collect();
        ranked.sort_by(|a,b|b.1.total_cmp(&a.1));
        timing.ordering_ms+=clock.elapsed().as_secs_f64()*1e3;
        // Solve the queue up to the next checkpoint.
        for &(i,_)in &ranked{
            if !alive[i]{continue;}
            solved_at[i]=Some(score[i]);
            // (research, S3) ATOM_LATTICE_POLES_UNTRANSLATED: a lattice-frequency
            // pole's vectors already target a family; translations do not copy them.
            space.untranslated_next=untranslated_lattice&&matches!(points[i],Pole::Complex(_))&&!centre_poles.iter().any(|p|{let(a,b)=(p.sigma(),points[i].sigma());a.re==b.re&&a.im==b.im});
            if space.solve(&mut lattice,points[i])?==0{alive[i]=false;}
            space.untranslated_next=false;
            if space.solves()%CHECKPOINT==0||space.solves()>=maximum_solves||space.full(){break;}
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
        let s=solve(&p,&initial,&Request{times:&times,horizons:&horizons,steady:true,groups:&groups,explicit:&std::sync::OnceLock::new(),runs:1},Some(&mut observe)).unwrap();
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
