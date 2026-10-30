//! The population pipeline: every output of a request is a matrix function
//! of one lattice operator applied to x0, read from one reduced model.
//!
//! 1. The lattice: the explicit set P and its ring Q (`charge_set`), and the
//!    a-priori amplitudes of x0 (`apriori`): every charge they carry above
//!    the gate that P u Q misses joins the lattice (the completion). The
//!    request's frame (`frame`: the kernels, the metric's block, the
//!    slowest relaxation's classes) is built on its own thread meanwhile.
//! 2. The fold: the run's own (`fold::FoldLibrary`), one truncation rule for
//!    every piece, the far shell's rings beyond the second order among them:
//!    pieces dropped smallest first while their first-order Duhamel responses
//!    on x0's evolution up to the run's longest output time T
//!    (`apriori::Duhamel`) sum within the fold's budget; the rings kept become
//!    static ring targets (`ring_targets`).
//! 3. One block rational Krylov space (`space`): the start [x0, the slip of
//!    every translation the a-priori rule activates], the translations (the
//!    gated steps, then the a-priori ones, with no solve), the bordered
//!    steady column, then the real ladder in a fixed bit-reversed cycle.
//! 4. Every CHECKPOINT solves (a run without a grid, the mean's, also right
//!    after the steady column): the metric Petrov-Galerkin model (`reduced`),
//!    its outputs (trajectory on the grid, horizons, slowest relaxation).
//!    An output's error estimate is its convergence over the last four
//!    checkpoints (`convergence`); its bound is that estimate plus the growth
//!    charge, and on a trajectory row the change of its projection onto the
//!    probability simplex (computed exactly). A checkpoint is accepted when
//!    every bound is within the tolerance.
//! 5. Stopping (deterministic, no wall time): accepted; or the largest
//!    bound, above the tolerance, did not improve by PLATEAU_GAIN over the
//!    last PLATEAU checkpoints; or the solve budget (MAXIMUM_SOLVES) or the
//!    basis budget (BASIS_BYTES) is spent. The run returns its best
//!    checkpoint (`Answer`), converged or partial with the reason and every
//!    sample's own bound; every checkpoint is reported to the observer as it
//!    is made.
use super::*;
use super::accuracy::{self,POPULATION_TOLERANCE};
use super::apriori::{self,Apriori,Duhamel};
use super::charge_set::{self,ExplicitSet,LatticeSets};
use super::fold::{DroppedPieces,FoldLibrary,FoldRule,ResponseRule,Storage};
use super::reduced::{Model,Projection,SlowRate};
use super::poles::{InnerWork,Lattice,Pole};
use super::space::{self,Column,Space};
use super::frame::{SwFrame,SwMetric};
use super::eigenbasis::BlockEigenbasis;

/// Solves between checkpoints. A run without a grid (a mean run) takes its
/// first one right after the steady column as well (measured on the
/// necessity review's 9 mean runs: accepted at 1 solve instead of 4 where the
/// mean had settled, the errors the same). Never a trajectory: a trajectory
/// checkpoint at 1 solve accepted four lasers without the trap at 10 ms
/// falsely (bound 3.0e-3, error 7.3e-3).
const CHECKPOINT:usize=4;
/// Most solves of a request, and the basis memory (the basis alone: no
/// image is stored). MAXIMUM_SOLVES is a deliberate margin: twice the 24
/// solves the slowest accepted research case takes.
const MAXIMUM_SOLVES:usize=48;
const BASIS_BYTES:usize=1<<30;
/// Memory guard: a request takes about BASE_BYTES plus MODE_BYTES per
/// lattice mode of each run, and its runs execute at once (`Request::runs`:
/// a full request's trajectory and mean, each with its own lattice and
/// space). Measured (private bytes, with the former error estimator, which
/// took about as much as the basis: an upper estimate now): D1 + three D2
/// lasers, 13 modes, 0.62 GB; six lasers without the trap, 63 modes, 2.0 GB.
/// A run plans every run of its request on a lattice of its own size, so the
/// runs together stay within the budget: a request beyond it stops before
/// its first solve, with the reason, and a completion beyond it is not
/// taken. The budget is MEMORY_SHARE of the machine's installed physical
/// memory: a run beyond the physical memory pages, and every solve sweeps
/// the lattice and the whole basis, so it would stall the machine rather
/// than slow down; the installed memory rather than the free, so whether a
/// request is solved does not depend on what else runs. MEMORY_SHARE is a
/// deliberate margin: the other half for the system, the app and the lab
/// PC's other programs.
const BASE_BYTES:f64=5e8;
const MODE_BYTES:f64=3.2e7;
const MEMORY_SHARE:f64=0.5;
/// The memory a request of `runs` runs on lattices of `modes` modes is
/// planned to take (bytes).
fn planned_bytes(runs:usize,modes:usize)->f64{BASE_BYTES+MODE_BYTES*(runs*modes)as f64}
/// The plateau: the largest bound did not improve by PLATEAU_GAIN over the
/// last PLATEAU checkpoints. Measured before the rule: four lasers + tweezer
/// at 10 us / 2000 samples held the bound at 5.0-5.7e-3 over 20 solves.
const PLATEAU:usize=3;
const PLATEAU_GAIN:f64=2.;
/// The convergence estimate (`convergence`): a ratio of successive changes
/// at or above STAGNATION is no convergence (the output is not accepted);
/// a last change below ROUNDOFF x tolerance is converged to roundoff.
/// STAGNATION is a deliberate margin: at r = 0.8 the geometric tail r / (1 -
/// r) is 4 changes, beyond which four levels cannot tell a slow geometric
/// convergence from a stall.
const STAGNATION:f64=0.8;
const ROUNDOFF:f64=1e-3;
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
/// The far shell's rings stop when what the next ones can land (trace norm),
/// times T, is within this share of the fold's budget.
const RING_TOLERANCE_SHARE:f64=1e-3;
/// A ring target's far mode of ring k + 1 is left out when its source is
/// below RING_PRUNE_SHARE x budget / (ring k + 1's response) of its ring's
/// largest: the pruned modes leave a share of the budget well below this (a
/// ring holds at most a few hundred far modes).
const RING_PRUNE_SHARE:f64=1e-6;
/// Stack of the frame's thread (LAPACK on it).
const FRAME_STACK:usize=1<<26;
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
    pub solves:usize,pub columns:usize,pub dimension:usize,
    /// The estimate and the largest bound over every output.
    pub estimate:Estimate,pub largest:f64,pub accepted:bool,
    pub growing:usize,pub clipped:usize,pub projected:usize,pub slowest:Option<SlowRate>,
    /// Raw outputs: trajectory rows (row 0 rho0's), horizon rows.
    pub trajectory:Vec<Vec<f64>>,pub horizons:Vec<Vec<f64>>,
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
    /// rho0's) and the bound of each (the estimate plus the growth charge
    /// plus the projection's change).
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
pub struct Timing{pub eigenbasis_ms:f64,pub explicit_set_ms:f64,pub apriori_ms:f64,pub setup_ms:f64,pub outputs_ms:f64,pub frame_ms:[f64;3],pub frame_wait_ms:f64,pub solves_ms:f64,pub projection_ms:f64,pub model_ms:f64,pub estimate_ms:f64,pub total_ms:f64}

pub struct Solution{
    /// The best checkpoint's outputs; `converged` when it was accepted, else
    /// the reason the run stopped.
    pub answer:Answer,pub converged:bool,pub reason:Option<String>,
    /// The request's explicit set, the run's lattice (the set and the
    /// charges its completion added, one of each pair) and translations.
    pub explicit:ExplicitSet,pub lattice:LatticeSets,pub completed:Vec<Charge>,pub translations:Vec<Charge>,
    /// The solves, and every checkpoint reported.
    pub columns:Vec<Column>,pub checkpoints:Vec<Checkpoint>,
    pub dimension:usize,pub work:InnerWork,pub basis:usize,pub block:usize,
    pub fold:FoldReport,pub timing:Timing,
}

/// A run's fold under the truncation rule: its rule, the orders in nu_q it
/// built and what it dropped (responses, populations); the far shell's
/// depth (rings beyond the second order kept) and the rings' responses; the
/// rings' responses', the build's (theirs included) and the ring targets'
/// time.
#[derive(Clone,Debug,Default)]
pub struct FoldReport{
    pub rule:Option<FoldRule>,pub orders:usize,pub dropped:DroppedPieces,pub targets:usize,
    pub rings:usize,pub ring_responses:Vec<f64>,pub responses_ms:f64,pub build_ms:f64,pub rings_ms:f64,
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
/// CANDIDATE_WEIGHT x tolerance; slowest first. The block's kernel is not a
/// class: the block annihilates the trace, so a decaying class carries none
/// of x0's trace |c_j sum_i (E V)_ij| (measured: at most 5e-3 of its weight)
/// and a class carrying at least KERNEL_TRACE_SHARE of its weight is the
/// kernel.
fn class_rates(frame:&SwFrame<'_>,eigen:&BlockEigenbasis,lapack:&Lapack,initial:&[C],tolerance:f64)->Result<Vec<f64>,String>{
    let class=frame.class()?;let d=initial.len();let n=class.readout.len()/d.max(1);
    let coordinates=eigen.coefficients(&[initial])?;
    let mut c=vec![ZERO;d];lapack.complex_gemm().multiply(d,1,d,ONE,&class.inverse,d,&coordinates,d,ZERO,&mut c,d)?;
    let mut classes:Vec<f64>=(0..d).filter_map(|j|{
        let column=&class.readout[n*j..n*(j+1)];
        let weight=magnitude(c[j])*column.iter().map(|z|magnitude(*z)).fold(0.,f64::max);
        let trace=magnitude(c[j])*magnitude(column.iter().fold(ZERO,|a,z|C{re:a.re+z.re,im:a.im+z.im}));
        let rate=-class.values[j].re;
        (rate>=super::reduced::KERNEL&&weight>=CANDIDATE_WEIGHT*tolerance&&trace<KERNEL_TRACE_SHARE*weight).then_some(rate)}).collect();
    classes.sort_by(f64::total_cmp);Ok(classes)
}

/// The slowest relaxation's candidates of a model (`Model::slow_candidates`),
/// slowest first, each with its rate, residual |A W y - theta W y| / |W y|
/// (the Ritz vector's image from the lattice) and whether the classes
/// `classes` slower than its rate / CLASS_GAP are matched: the residual
/// cannot see a slow class the space lacks.
fn slow_pairs(model:&Model,space:&Space,vectors:&super::krylov::Vectors,classes:&[f64],tolerance:f64,lattice:&mut Lattice<'_>)->Result<Vec<SlowRate>,String>{
    let chosen=model.slow_candidates(CANDIDATE_WEIGHT*tolerance);
    if chosen.is_empty(){return Ok(Vec::new());}
    let parts:Vec<Vec<f64>>=chosen.iter().flat_map(|&(j,_)|{let y=model.ritz_coordinates(j);[y.iter().map(|z|z.re).collect(),y.iter().map(|z|z.im).collect()]}).collect();
    let basis:Vec<&[C]>=space.basis.iter().map(|v|v.as_slice()).collect();
    let w=vectors.combinations(&basis,&parts)?;
    let aw=w.iter().map(|x|{let mut y=vec![ZERO;x.len()];lattice.apply(x,&mut y).map(|_|y)}).collect::<Result<Vec<_>,_>>()?;
    let found:Vec<(f64,f64)>=chosen.iter().enumerate().map(|(c,&(j,rate))|{
        let theta=model.theta[j];let(wr,wi,ar,ai)=(&w[2*c],&w[2*c+1],&aw[2*c],&aw[2*c+1]);
        // A W y - theta W y, real and imaginary parts.
        let re:Vec<C>=(0..wr.len()).map(|e|C{re:ar[e].re-theta.re*wr[e].re+theta.im*wi[e].re,im:ar[e].im-theta.re*wr[e].im+theta.im*wi[e].im}).collect();
        let im:Vec<C>=(0..wr.len()).map(|e|C{re:ai[e].re-theta.re*wi[e].re-theta.im*wr[e].re,im:ai[e].im-theta.re*wi[e].im-theta.im*wr[e].im}).collect();
        let residual=(vectors.dot(&re,&re)+vectors.dot(&im,&im)).max(0.).sqrt()/(vectors.dot(wr,wr)+vectors.dot(wi,wi)).max(f64::MIN_POSITIVE).sqrt();
        (rate,residual)}).collect();
    let matched=|rate:f64|classes.iter().filter(|&&class|class<rate/CLASS_GAP).all(|&class|found.iter().any(|f|(f.0-class).abs()<=CLASS_MATCH*class));
    Ok(found.iter().map(|&(rate,residual)|SlowRate{rate,residual,matched:matched(rate)}).collect())
}

/// Acceptance by convergence: the outputs of the last four checkpoints x_k,
/// their changes D_k = |x_k - x_k-1| (`accuracy::metric`), the rate r =
/// max(|D_k| / |D_k-1|, |D_k-1| / |D_k-2|) (two ratios: a level that has not
/// entered the asymptotic regime, a ratio near or above one, counts as
/// stagnation) and the geometric tail D_k r / (1 - r) as the estimate. The
/// trajectory is one output (its ratios from its largest changes over the
/// grid; each sample's estimate from its own last change), every horizon
/// another. r at or above STAGNATION is no convergence: the estimate is at
/// least twice the tolerance. A last change below ROUNDOFF x tolerance is
/// converged to roundoff (the change itself). Fewer than four checkpoints:
/// unknown (1).
fn convergence(history:&[(Vec<Vec<f64>>,Vec<Vec<f64>>)],groups:&[Vec<usize>],tolerance:f64)->Estimate{
    let k=history.len();let(t,hz)=(&history[k-1].0,&history[k-1].1);
    if k<4{return Estimate{trajectory:vec![1.;t.len()],horizons:vec![1.;hz.len()]};}
    let output=|d0:&[f64],d1:&[f64],d2:&[f64]|->Vec<f64>{
        let(m0,m1,m2)=(d0.iter().copied().fold(0.,f64::max),d1.iter().copied().fold(0.,f64::max),d2.iter().copied().fold(0.,f64::max));
        let converged=m2<=ROUNDOFF*tolerance;let ratio=|a:f64,b:f64|if a>0.{b/a}else if b>0.{f64::INFINITY}else{0.};let r=ratio(m1,m2).max(ratio(m0,m1));
        d2.iter().map(|&x|if converged{x}else if r<STAGNATION{x*r/(1.-r)}else{(x*STAGNATION/(1.-STAGNATION)).max(2.*tolerance)}).collect()};
    let changes=|level:usize|->Vec<f64>{(0..t.len()).map(|i|accuracy::metric(&history[level].0[i],&history[level-1].0[i],groups)).collect()};
    let horizon=|i:usize|->f64{let change=|level:usize|accuracy::metric(&history[level].1[i],&history[level-1].1[i],groups);
        output(&[change(k-3)],&[change(k-2)],&[change(k-1)])[0]};
    Estimate{trajectory:output(&changes(k-3),&changes(k-2),&changes(k-1)),horizons:(0..hz.len()).map(horizon).collect()}
}

/// The responses of the far shell's rings beyond the second order: per
/// source E_j of x0 (`Duhamel::sources`) the far shell's landing on the
/// lattice (`FoldLibrary::far_shell`, the run's fold and the frame's share
/// L0's split), per ring its Duhamel response summed over the landing modes.
fn ring_responses(prepared:&LatticePrepared,modes:&[Charge],nu:&[f64],duhamel:&Duhamel<'_>,horizon:f64,budget:f64)->Result<Vec<f64>,String>{
    if prepared.plan.far().is_empty()||duhamel.count()==0{return Ok(Vec::new());}
    let d=prepared.generator.n*prepared.generator.n;let nu_of=|q:&[i32]|prepared.nu(q);
    let centre=modes.iter().position(|q|*q==prepared.generator.zero).ok_or("Lattice modes must contain the zero charge")?;
    // Per ring and landing mode the images of every source ([j * d + e]).
    let sources:Vec<(usize,&[C])>=duhamel.sources().map(|e|(centre,e)).collect();let count=sources.len();
    let(landings,_,_)=prepared.fold.far_shell_sources(&prepared.generator,&prepared.plan,&nu_of,modes,&sources,RING_TOLERANCE_SHARE*budget/horizon)?;
    landings.iter().skip(1).map(|ring|{let mut per:BTreeMap<usize,Vec<C>>=BTreeMap::new();
        for(&(jj,q),block)in ring{per.entry(q).or_insert_with(||vec![ZERO;count*d])[jj*d..(jj+1)*d].copy_from_slice(block);}
        per.iter().map(|(&q,v)|duhamel.response(nu[q],1.,v)).sum()}).collect()
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
    let frame=SwFrame::new(prepared,longest);
    std::thread::scope(|scope|->Result<Solution,String>{
        std::thread::Builder::new().stack_size(FRAME_STACK).spawn_scoped(scope,||{let _=frame.class();}).map_err(|e|e.to_string())?;
        run(prepared,initial,request,&mut observer,&frame,started)
    })
}

/// One run of `solve` (see the module documentation).
fn run<'a>(prepared:&'a LatticePrepared,initial:&[C],request:&Request<'_>,observer:&mut Option<&mut dyn FnMut(&Checkpoint,&Answer)>,
    frame:&'a SwFrame<'a>,started:Instant)->Result<Solution,String>{
    let tolerance=POPULATION_TOLERANCE;let n=prepared.generator.n;let d=n*n;
    if initial.len()!=d{return Err("Initial density dimension differs".into());}
    let(times,longest)=(request.times,frame.horizon);
    let t1=if times.len()>1{times[1]}else{longest/LADDER_SAMPLES};
    let gate=charge_set::GATE*tolerance;
    let mut timing=Timing::default();
    // 1. The lattice: the explicit set, completed by the a-priori rule.
    let clock=Instant::now();let eigen=prepared.eigenbasis()?;timing.eigenbasis_ms=clock.elapsed().as_secs_f64()*1e3;let clock=Instant::now();
    let explicit=request.explicit.get_or_init(||charge_set::explicit_set(prepared,initial,tolerance)).clone()?;
    timing.explicit_set_ms=clock.elapsed().as_secs_f64()*1e3;
    let clock=Instant::now();
    let samples=apriori::sample_times(times,request.horizons);
    let apriori=Apriori::new(prepared,frame.kernels()?,eigen,initial,&samples)?;
    let budget=MEMORY_SHARE*crate::memory::physical_bytes()?;
    let base=explicit.sets.modes();
    let planned=planned_bytes(request.runs,base.len());
    if planned>budget{return Err(format!("The lattice of this request has {} modes and would need about {:.1} GB, above the {:.0} GB memory budget: not solved",base.len(),planned/1e9,budget/1e9));}
    let mut completed=apriori.missing(&base,gate);
    let mut sets=explicit.sets.completed(&completed);
    if planned_bytes(request.runs,sets.modes().len())>budget{completed.clear();sets=explicit.sets.clone();}
    timing.apriori_ms=clock.elapsed().as_secs_f64()*1e3;
    // 2. The run's fold by the truncation rule, the far shell's rings among
    // its pieces.
    let clock=Instant::now();
    let modes=sets.modes();
    let rule=FoldRule::new(longest);
    let nu:Vec<f64>=modes.iter().map(|q|prepared.nu(q)).collect::<Result<_,_>>()?;
    let duhamel=Duhamel::new(eigen,initial,&samples)?;
    let responses_clock=Instant::now();
    let rings=ring_responses(prepared,&modes,&nu,&duhamel,longest,rule.budget)?;
    let responses_ms=responses_clock.elapsed().as_secs_f64()*1e3;
    let input=ResponseRule{modes:modes.clone(),nu,duhamel:&duhamel,rings};
    let mut fold=FoldLibrary::build(&prepared.generator,&prepared.plan,&prepared.lapack,Storage::Rule(rule,&input))?;
    let build_ms=clock.elapsed().as_secs_f64()*1e3;let clock=Instant::now();
    let depth=fold.diagnostics.dropped.kept_rings;
    if depth>0{
        // The ring targets land on the lattice's edges whose charges the plan
        // folds (and the diagonal), K^(0) alone; a far mode of ring k + 1 is
        // pruned below its share of the budget.
        let known:BTreeSet<Charge>=prepared.plan.targets.iter().chain(&prepared.plan.omitted).map(|t|t.charge.clone()).chain(std::iter::once(prepared.generator.zero.clone())).collect();
        let mut offsets=BTreeSet::new();for p in &modes{for q in &modes{let t=charge_minus(q,p)?;if known.contains(&t){offsets.insert(t);}}}
        let prune:Vec<f64>=(0..depth).map(|k|input.rings.get(k).map_or(0.,|&a|if a>0.{RING_PRUNE_SHARE*rule.budget/a}else{0.})).collect();
        let targets=fold.ring_targets(&prepared.generator,&prepared.plan,depth,1,&prune,&offsets)?;fold.targets.extend(targets);
    }
    let report=FoldReport{rule:Some(rule),orders:fold.diagnostics.orders,dropped:fold.diagnostics.dropped.clone(),targets:fold.targets.len(),rings:depth,ring_responses:input.rings.clone(),responses_ms,build_ms,rings_ms:clock.elapsed().as_secs_f64()*1e3};
    drop(input);
    // 3. The space: the gated translations, the steady column, the a-priori
    // translations with their slips in the block start.
    let clock=Instant::now();
    let mut lattice=Lattice::new(prepared,&fold,sets.clone())?;
    let capacity=BASIS_BYTES/(std::mem::size_of::<C>()*d*lattice.modes.len());
    let mut space=Space::new(&mut lattice,initial,capacity)?;
    for s in charge_set::gated_steps(prepared,tolerance){if lattice.operator().index.contains_key(&s){space.activate(&lattice,&s)?;}}
    timing.setup_ms=clock.elapsed().as_secs_f64()*1e3;
    space.solve(&mut lattice,Pole::Zero)?;
    {
        let(centre,mirror)={let op=lattice.operator();(op.centre,op.mirror.clone())};
        for q in 0..modes.len(){
            if q==centre||apriori.amplitude(&modes[q])<=gate{continue;}
            let(c,m)=(modes[q].clone(),opposite(&modes[q]));let active=space.translations();
            if active.contains(&c)||active.contains(&m){continue;}
            space.activate(&lattice,&c)?;
            space.add_start(&lattice,apriori.slip_field(eigen,&modes,&mirror,q))?;
        }
    }
    // The ladder spans the grid's times when there is one (a horizon is
    // served by the steady column), in a fixed cycle: sorted by sigma, the
    // bit-reversed order over the next power of two (the indices beyond the
    // ladder left out), so that every stretch of the cycle spreads over the
    // whole ladder.
    let cycle:Vec<Pole>={
        let mut ladder=space::ladder(t1,times.last().copied().unwrap_or(longest));
        ladder.sort_by(|a,b|b.sigma().total_cmp(&a.sigma()));
        let m=ladder.len();let p2=m.next_power_of_two();let width=p2.trailing_zeros();
        (0..p2).map(|i|if width==0{0}else{i.reverse_bits()>>(usize::BITS-width)}).filter(|&i|i<m).map(|i|ladder[i]).collect()};
    // The frame's metric (waiting for the frame's thread) and the projection.
    let clock=Instant::now();
    let metric=SwMetric::new(frame,eigen,lattice.operator())?;
    timing.frame_wait_ms=clock.elapsed().as_secs_f64()*1e3;
    let x0={let op=lattice.operator();let mut x=vec![ZERO;op.dimension()];x[op.centre*d..(op.centre+1)*d].copy_from_slice(initial);x};
    let mut galerkin=Projection::new(&lattice,&metric,space::LADDER_TOP/t1,&x0)?;
    let lapack=&prepared.lapack;let ops=super::krylov::Vectors::new(lattice.operator())?;
    // The decaying classes of x0, which the slowest relaxation is checked
    // against (the frame's block).
    let classes=class_rates(frame,eigen,lapack,initial,tolerance)?;
    let mut checkpoints:Vec<Checkpoint>=Vec::new();let mut history=Vec::new();
    // The best checkpoint so far and per checkpoint the stopping rule's
    // progress measure.
    let mut best:Option<Answer>=None;let mut progress:Vec<f64>=Vec::new();
    let mut reason:Option<String>=None;let mut next=0usize;
    loop{
        let exhausted=space.solves()>=MAXIMUM_SOLVES||space.full();
        if exhausted||space.solves()%CHECKPOINT==0||(times.is_empty()&&space.solves()==1){
            // 4. The checkpoint: model, outputs, estimate, bounds.
            let clock=Instant::now();
            galerkin.ingest(&space.basis,&mut lattice)?;
            let model=galerkin.model(lapack,&space.initial,longest,tolerance)?;timing.model_ms+=clock.elapsed().as_secs_f64()*1e3;
            timing.projection_ms=galerkin.milliseconds;
            let outputs_clock=Instant::now();
            let mut trajectory=model.trajectory(&galerkin,times);
            if let Some(first)=trajectory.first_mut(){*first=(0..n).map(|i|initial[i*n+i].re).collect();}
            let horizons:Vec<Vec<f64>>=request.horizons.iter().map(|&t|model.horizon(&galerkin,t)).collect();
            timing.outputs_ms+=outputs_clock.elapsed().as_secs_f64()*1e3;
            let estimate_clock=Instant::now();
            history.push((trajectory.clone(),horizons.clone()));
            let mut estimate=convergence(&history,request.groups,tolerance);
            // The growth charge on every output.
            for v in estimate.trajectory.iter_mut().chain(estimate.horizons.iter_mut()){*v+=model.growth_charge;}
            timing.estimate_ms+=estimate_clock.elapsed().as_secs_f64()*1e3;
            // The slowest relaxation.
            let pairs=slow_pairs(&model,&space,&ops,&classes,tolerance,&mut lattice)?;
            // The outputs with their bounds.
            let answer={
                let outputs_clock=Instant::now();
                let projected:Vec<Vec<f64>>=trajectory.iter().map(|p|simplex(p)).collect();
                let changes:Vec<f64>=trajectory.iter().zip(&projected).map(|(a,b)|accuracy::metric(a,b,request.groups)).collect();
                let sample_errors:Vec<f64>=estimate.trajectory.iter().zip(&changes).map(|(e,p)|e+p).collect();
                let basis:Vec<&[C]>=space.basis.iter().map(|v|v.as_slice()).collect();
                let horizon_fields=request.horizons.iter().map(|&t|Ok(ops.combinations(&basis,&[model.state(t)])?.remove(0))).collect::<Result<Vec<_>,String>>()?;
                timing.outputs_ms+=outputs_clock.elapsed().as_secs_f64()*1e3;
                let mut answer=Answer{solves:space.solves(),modes:modes.clone(),trajectory:projected,sample_errors,
                    horizons:horizons.clone(),horizon_fields,horizon_errors:estimate.horizons.clone(),
                    estimate:estimate.clone(),slowest:pairs.first().copied(),accepted:false};
                answer.accepted=answer.largest()<=tolerance;
                answer};
            let point=Checkpoint{solves:space.solves(),columns:space.basis.len(),dimension:model.r,estimate,largest:answer.largest(),accepted:answer.accepted,
                growing:model.growing,clipped:model.clipped,projected:model.projected,slowest:answer.slowest,
                trajectory,horizons,inner:lattice.work().clone(),timing:timing.clone(),milliseconds:clock.elapsed().as_secs_f64()*1e3};
            if let Some(o)=observer.as_mut(){o(&point,&answer);}
            checkpoints.push(point);
            // The best checkpoint: one within the tolerance over any other,
            // the latest of those, else the least largest bound.
            let within=answer.largest()<=tolerance;
            if best.as_ref().is_none_or(|b|if within{true}else{b.largest()>tolerance&&answer.largest()<b.largest()}){best=Some(answer.clone());}
            if answer.accepted{break;}
            // 5. The stopping rule on the bound.
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
                    else{format!("the solve budget ({MAXIMUM_SOLVES}) was spent")};
                let over:Vec<usize>=(0..b.sample_errors.len()).filter(|&i|b.sample_errors[i]>tolerance).collect();
                let samples=match(over.first(),over.last()){(Some(&a),Some(&z))=>format!("; {} of {} samples above it, t = {:.4} to {:.4} us",over.len(),b.sample_errors.len(),times[a],times[z]),_=>String::new()};
                let means=b.horizon_errors.iter().filter(|&&e|e>tolerance).count();
                let means=if means>0{format!("; {means} mean output(s) above it")}else{String::new()};
                reason=Some(format!("Stopped because {why}: the best checkpoint ({} solves) has an error bound of {:.2e} against the tolerance {tolerance:e}{samples}{means}",b.solves,b.largest()));
                break;
            }
        }
        // The fixed cycle of real poles up to the next checkpoint.
        loop{
            let pole=cycle[next%cycle.len()];next+=1;
            space.solve(&mut lattice,pole)?;
            if space.solves()%CHECKPOINT==0||space.solves()>=MAXIMUM_SOLVES||space.full(){break;}
        }
    }
    let answer=best.ok_or("A run has its checkpoint")?;
    let work=lattice.work().clone();
    timing.solves_ms=work.milliseconds;timing.projection_ms=galerkin.milliseconds;timing.total_ms=started.elapsed().as_secs_f64()*1e3;
    timing.frame_ms=frame.milliseconds();
    let converged=answer.accepted;
    Ok(Solution{answer,converged,reason:if converged{None}else{reason},explicit,lattice:sets,completed,translations:space.translations(),
        columns:space.columns.clone(),dimension:galerkin.columns(),basis:space.basis.len(),block:space.block,work,fold:report,timing,checkpoints})
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
        // null vector. The far drives at 2% of the fixture's.
        let mut f=fixture();for v in f.far.iter_mut().chain(f.far2.iter_mut()){*v*=0.02;}let p=prepared(&f);let n=4;let d=16;let groups=vec![vec![0,1],vec![2,3]];
        let mut initial=vec![ZERO;d];initial[0]=ONE;
        let times:Vec<f64>=(0..21).map(|i|0.1*i as f64).collect();let horizons=[3.];
        let mut seen=0;let mut observe=|_:&Checkpoint,_:&Answer|{seen+=1;};
        let s=solve(&p,&initial,&Request{times:&times,horizons:&horizons,groups:&groups,explicit:&std::sync::OnceLock::new(),runs:1},Some(&mut observe)).unwrap();
        assert!(s.converged&&s.reason.is_none()&&seen==s.checkpoints.len(),"{:?}",s.reason);
        let a=&s.answer;
        let op=LatticeOperator::new(&p,&p.fold,&a.modes).unwrap();let dense=op.dense();let size=op.dimension();
        let rate=|x:&[C]|->Vec<C>{(0..size).map(|r|{let mut v=ZERO;for c in 0..size{mul_add(&mut v,dense[r*size+c],x[c]);}v}).collect()};
        let mut x=vec![ZERO;size];x[op.centre*d]=ONE;
        let readout=|x:&[C],t:f64|->Vec<f64>{(0..n).map(|i|(0..a.modes.len()).map(|q|mul(C{re:(op.nu[q]*t).cos(),im:(op.nu[q]*t).sin()},x[q*d+i*n+i]).re).sum()).collect()};
        let steps=400;let h=0.1/steps as f64;let mut t=0.;
        let axpy=|x:&[C],k:&[C],s:f64|->Vec<C>{x.iter().zip(k).map(|(a,b)|C{re:a.re+s*b.re,im:a.im+s*b.im}).collect()};
        for(i,row)in a.trajectory.iter().enumerate(){
            if i>0{for _ in 0..steps{let k1=rate(&x);let k2=rate(&axpy(&x,&k1,h/2.));let k3=rate(&axpy(&x,&k2,h/2.));let k4=rate(&axpy(&x,&k3,h));
                for e in 0..size{x[e].re+=h/6.*(k1[e].re+2.*k2[e].re+2.*k3[e].re+k4[e].re);x[e].im+=h/6.*(k1[e].im+2.*k2[e].im+2.*k3[e].im+k4[e].im);}t+=h;}}
            // Every sample within the tolerance of the exact lattice dynamics,
            // and its bound within the tolerance.
            let error=accuracy::metric(&readout(&x,t),row,&groups);
            assert!(error<=POPULATION_TOLERANCE&&a.sample_errors[i]<=POPULATION_TOLERANCE,"sample {i}: {error:e} against {:e}",a.sample_errors[i]);
        }
        for _ in 0..10*steps{let k1=rate(&x);let k2=rate(&axpy(&x,&k1,h/2.));let k3=rate(&axpy(&x,&k2,h/2.));let k4=rate(&axpy(&x,&k3,h));
            for e in 0..size{x[e].re+=h/6.*(k1[e].re+2.*k2[e].re+2.*k3[e].re+k4[e].re);x[e].im+=h/6.*(k1[e].im+2.*k2[e].im+2.*k3[e].im+k4[e].im);}}
        let centre:Vec<f64>=(0..n).map(|i|x[op.centre*d+i*n+i].re).collect();
        let horizon=accuracy::metric(&centre,&a.horizons[0],&groups);assert!(horizon<=POPULATION_TOLERANCE,"horizon {horizon:e}");
        let exact=dense_steady(&p,&dense,op.centre,n);
        // The bordered column at a tight tolerance is the lattice's null
        // vector.
        let mut lattice=Lattice::new(&p,&p.fold,s.lattice.clone()).unwrap();let mut u=vec![ZERO;size];for i in 0..n{u[op.centre*d+i*n+i]=C{re:0.25,im:0.};}
        let(w,_,_)=lattice.solve(Pole::Zero,&u,1e-13).unwrap();let trace:f64=(0..n).map(|i|w[op.centre*d+i*n+i].re).sum();
        let column:Vec<f64>=(0..n).map(|i|w[op.centre*d+i*n+i].re/trace).collect();
        let tight=accuracy::metric(&exact,&column,&groups);assert!(tight<1e-10,"bordered column {tight:e}");
        assert!(a.slowest.is_some());
        // Projected rows: probabilities, the first one rho0's.
        assert!(a.trajectory.iter().all(|r|r.iter().all(|&v|v>=0.)&&(r.iter().sum::<f64>()-1.).abs()<1e-12)&&a.trajectory[0][0]==1.);
    }
    #[test]
    fn convergence_estimates_geometric_changes_and_refuses_stagnation(){
        // Changes halving each level: r = 1/2, the estimate the last change.
        // Changes shrinking by 0.9: stagnation, at least twice the tolerance.
        // Fewer than four levels: unknown.
        let groups:Vec<Vec<usize>>=vec![];
        let levels=|ratio:f64|->Vec<(Vec<Vec<f64>>,Vec<Vec<f64>>)>{let mut x=0.;let mut step=0.08;
            (0..5).map(|_|{let row=vec![vec![x,1.-x]];x+=step;step*=ratio;(row.clone(),row)}).collect()};
        let tolerance=POPULATION_TOLERANCE;
        let halving=levels(0.5);let e=convergence(&halving,&groups,tolerance);let last=0.08*0.5_f64.powi(3);
        assert!((e.trajectory[0]-last).abs()<=1e-12&&(e.horizons[0]-last).abs()<=1e-12,"{:?} against {last:e}",e);
        let slow=levels(0.9);let e=convergence(&slow,&groups,tolerance);
        assert!(e.trajectory[0]>=2.*tolerance&&e.horizons[0]>=2.*tolerance,"{e:?}");
        let e=convergence(&halving[..3],&groups,tolerance);assert!(e.trajectory[0]==1.&&e.horizons[0]==1.);
    }
}
