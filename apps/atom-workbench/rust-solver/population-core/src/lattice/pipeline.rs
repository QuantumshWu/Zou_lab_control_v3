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
//!
//! The fold: every run builds its own (`fold::FoldLibrary`, by the truncation
//! rule at the run's longest output time T and its lattice's largest
//! |nu_q|); at the first checkpoint the far shell's depth is decided on the
//! reduced solution with the near ring's completion (one decision, at most
//! one restart); at the end what the fold leaves out is bounded on the
//! reduced solution through the true propagator's trace-norm contraction
//! and added to every bound (`posterior`).
use super::*;
use super::charge_set::{self,ExplicitSet};
use super::estimate_api::{ACCEPTANCE_FACTOR,ErrorEstimate,EstimateSetup,Outputs,ReducedView};
use super::ledger::{self,POPULATION_TOLERANCE};
use super::fold::{DroppedPieces,FoldLibrary,FoldRule,Storage,FOLD_SHARE,field_trace_norm};
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
/// The near ring's completion (`solve`): at the first checkpoint, a bound
/// above the tolerance of which the slow part (the ring's feedback) carries
/// at least COMPLETION_SHARE is completed. Measured (three lasers at B = 0
/// and 1 mG, p(1 s), before the truncation rule): at the first checkpoint
/// the Krylov part is 1e-11 of bounds of 7.1-7.4e-3, the ring's part
/// 3.8-4.3e-3; a trajectory's first checkpoint is the Krylov residual's
/// (2e-1 against 2-8e-3). Without the share, completing wherever the ring
/// reaches COMPLETION_FLOOR, the same three lasers at 20 us restarted as
/// well: twice the wall time (1.6 -> 3.5 s), the error at 1 mG 4.5e-4 ->
/// 1.6e-3.
const COMPLETION_SHARE:f64=0.5;
/// The ring charges join the lattice by decreasing attribution while the
/// ring's part they leave (its unpromoted share of the attribution, bound
/// units) is at least COMPLETION_FLOOR x tolerance. Measured: the floor left
/// the decisions of the 20 completing runs of the necessity review unchanged
/// (every ring part at least 7.2e-3), and every run without an ablation
/// promotes one pair carrying its whole attribution, as the earlier rule
/// (90% of the attribution) did.
const COMPLETION_FLOOR:f64=0.25;
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

/// One checkpoint, for the diagnostics.
#[derive(Clone,Debug)]
pub struct Checkpoint{
    pub solves:usize,pub columns:usize,pub dimension:usize,
    /// The estimate and the largest bound over every output.
    pub estimate:ErrorEstimate,pub largest:f64,pub accepted:bool,
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
    /// rho0's) and the bound of each (ACCEPTANCE_FACTOR x the estimate plus
    /// the projection's change).
    pub trajectory:Vec<Vec<f64>>,pub sample_errors:Vec<f64>,
    /// Horizon populations (laboratory time averages, raw), their fields on
    /// `modes` (Hermitian-symmetric) and bounds.
    pub horizons:Vec<Vec<f64>>,pub horizon_fields:Vec<Vec<C>>,pub horizon_errors:Vec<f64>,
    pub estimate:ErrorEstimate,pub slowest:Option<SlowRate>,
    /// Every bound within the tolerance.
    pub accepted:bool,
}
impl Answer{
    /// The largest bound over every output.
    pub fn largest(&self)->f64{self.sample_errors.iter().chain(&self.horizon_errors).fold(0_f64,|a,&b|a.max(b))}
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
    pub fold:FoldReport,pub timing:Timing,
}

/// A run's fold under the truncation rule: its rule, the orders in nu_q it
/// built and what its build dropped (induced trace norm x T); the far
/// shell's depth built (rings beyond the second order); at the end the
/// a-posteriori term at T (population units, 1/2 int_0^T ||D x(s)||_1 ds on
/// the reduced solution, added to every bound), the dressing (half the far
/// amplitudes' trace norms at t = 0: the initial slip's first-order bound,
/// reported) and their time.
#[derive(Clone,Debug,Default)]
pub struct FoldReport{
    pub rule:Option<FoldRule>,pub orders:usize,pub dropped:DroppedPieces,pub targets:usize,pub build_ms:f64,
    pub rings:usize,pub posterior:f64,pub dressing:f64,pub posterior_ms:f64,
}

/// The first checkpoint's decision: the near ring's completion (the ring
/// charges promoted into P, one of each pair, their mirrors joining; the
/// completed explicit set, the run's own when none is) and the far shell's
/// depth (`rings` beyond the second order, and per ring its action on the
/// reduced solution, int_0^T ||D_k x(s)||_1 ds); where it was decided, the
/// bound there, its slow part and the ring's part (bound units), the ring
/// charges' attribution shares, and whether the run restarted.
#[derive(Clone,Debug)]
pub struct Completion{
    pub promoted:Vec<Charge>,pub explicit:ExplicitSet,pub rings:usize,pub ring_actions:Vec<f64>,pub decided_at:usize,
    pub bound:f64,pub slow_part:f64,pub ring_part:f64,
    pub attribution:Vec<(Charge,f64)>,pub restarted:bool,
}

/// The Ritz modes of a checkpoint's reduced model, what the fold's rule reads
/// of them: per mode its value, its coefficient and its Ritz vector's
/// coordinates on the first `columns` basis vectors (S row-major r x r).
struct RitzModes{r:usize,theta:Vec<C>,coefficients:Vec<C>,vectors:Vec<C>,columns:usize}
impl RitzModes{
    fn of(model:&Model,columns:usize)->Self{Self{r:model.r,theta:model.used_theta.clone(),coefficients:model.used_coefficients.clone(),vectors:model.vectors.clone(),columns}}
    /// x_m = W s_m on the lattice.
    fn field(&self,basis:&[Vec<C>],m:usize)->Vec<C>{
        let mut x=vec![ZERO;basis[0].len()];
        for j in 0..self.columns{let c=self.vectors[j*self.r+m];if !nonzero(c){continue;}for(v,&w)in x.iter_mut().zip(&basis[j]){mul_add(v,c,w);}}
        x}
    /// int_0^t |e^{theta_m s}| ds.
    fn weight(&self,m:usize,t:f64)->f64{let a=self.theta[m].re;if (a*t).abs()<1e-12{t}else{((a*t).min(EXPONENT_CAP).exp()-1.)/a}}
}
/// An exponent above which e^x is taken at e^EXPONENT_CAP (a growing mode's
/// weight is then infinite in effect: its bound carries no information).
const EXPONENT_CAP:f64=700.;
/// A far-shell action's rings stop when what the next ones can land is
/// within this share of the rule's budget (over every mode of a model).
const RING_TOLERANCE_SHARE:f64=1e-3;

/// Per ring k >= 1 of the far shell (the parts of order k + 2), its action
/// on the reduced solution to T, sum_m |a_m| N_1(D_k x_m) phi_m(T) (the
/// rule's units: trace norm x time).
fn ring_actions(fold:&FoldLibrary,prepared:&LatticePrepared,modes:&[Charge],ritz:&RitzModes,basis:&[Vec<C>],horizon:f64,budget:f64)->Result<Vec<f64>,String>{
    let n=prepared.generator.n;let nu_of=|q:&[i32]|prepared.nu(q);
    let weights:Vec<f64>=(0..ritz.r).map(|m|magnitude(ritz.coefficients[m])*ritz.weight(m,horizon)).collect();
    let total:f64=weights.iter().sum();let mut actions:Vec<f64>=Vec::new();
    if !(total>0.){return Ok(actions);}
    let tolerance=RING_TOLERANCE_SHARE*budget/(total*(n as f64).powf(1.5)*modes.len()as f64);
    let execution=crate::parallel::Execution::new()?;
    let per_mode=execution.map((0..ritz.r).filter(|&m|weights[m]>0.).collect(),|m|->Result<Vec<f64>,String>{
        let x=ritz.field(basis,m);
        let shell=fold.far_shell(&prepared.generator,&prepared.plan,&nu_of,modes,&x,ritz.theta[m],tolerance)?;
        shell.landings.iter().skip(1).map(|landing|Ok(weights[m]*field_trace_norm(&prepared.lapack,landing,n)?)).collect()});
    for rings in per_mode{for(k,a)in rings?.into_iter().enumerate(){if k==actions.len(){actions.push(0.);}actions[k]+=a;}}
    Ok(actions)
}
/// The truncation rule's a-posteriori term (`fold`): what the run's fold
/// leaves out of the far shell, by its action on the reduced solution
/// x(s) = sum_m a_m e^{theta_m s} x_m through the true propagator's trace-norm
/// contraction, the populations moving by at most
/// 1/2 sum_m |a_m| N_1(g_m) phi_m(t), g_m = (S(theta_m) - K) x_m the far shell's
/// exact action at the mode's own shift less the applied fold, with the
/// rings' tail (`LatticeOperator::left_out_exact`); what the far shell lands
/// on near charges outside the lattice is the near ring's, which the ring's
/// feedback charges with its detuning (`estimator`), not this bound,
/// phi_m(t) = int_0^t e^{Re theta_m s} ds; per grid time and per horizon,
/// with the dressing (half the far amplitudes' trace norms at t = 0).
fn posterior(lattice:&Lattice<'_>,prepared:&LatticePrepared,ritz:&RitzModes,basis:&[Vec<C>],times:&[f64],horizons:&[f64],horizon:f64,budget:f64)->Result<(Vec<f64>,Vec<f64>,f64),String>{
    let op=lattice.operator();let n=op.n;let lapack=&prepared.lapack;
    let(mut grid,mut means,mut dressing)=(vec![0.;times.len()],vec![0.;horizons.len()],0.);
    let weights:Vec<f64>=(0..ritz.r).map(|m|magnitude(ritz.coefficients[m])*ritz.weight(m,horizon)).collect();
    let total:f64=weights.iter().sum();if !(total>0.){return Ok((grid,means,dressing));}
    let tolerance=RING_TOLERANCE_SHARE*budget/(total*(n as f64).powf(1.5)*op.modes.len()as f64);
    let execution=crate::parallel::Execution::new()?;
    let parts=execution.map((0..ritz.r).filter(|&m|nonzero(ritz.coefficients[m])).collect(),|m|->Result<(usize,f64,f64),String>{
        let x=ritz.field(basis,m);
        let(g,shell)=op.left_out_exact(&x,ritz.theta[m],tolerance)?;
        Ok((m,field_trace_norm(lapack,&g,n)?+shell.tail*(n as f64).powf(1.5),shell.dressing.iter().sum::<f64>()))});
    for part in parts{let(m,size,dressed)=part?;let a=magnitude(ritz.coefficients[m]);
        for(e,&t)in grid.iter_mut().zip(times){*e+=0.5*a*size*ritz.weight(m,t);}
        for(e,&t)in means.iter_mut().zip(horizons){*e+=0.5*a*size*ritz.weight(m,t);}
        dressing+=0.5*a*dressed;
    }
    Ok((grid,means,dressing))
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
/// The first checkpoint's decision, one for both lattices around the run's:
/// the far shell's depth, the fewest rings beyond the second order whose
/// rest, by its action on the reduced solution to T (`ring_actions`), stays
/// within what the build left of the truncation rule's budget; and the near
/// ring's completion, when the largest bound is above the tolerance and its
/// slow part (the omitted ring's feedback) is at least COMPLETION_SHARE of
/// it: the ring charges join P with their mirrors by decreasing attribution
/// (|a_m| |B y_r| over the kernel and slow modes, from the ring's feedback)
/// while the ring's part they leave reaches COMPLETION_FLOOR x tolerance.
/// When either asks for more, the run restarts once on the completed lattice
/// and fold (the frame and the eigenbasis are kept), its checkpoints after
/// the first run's. Every other run is unchanged.
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
    // The run's fold by the truncation rule (T, the lattice's largest
    // |nu_q|), with the far shell's depth a completion decided.
    let clock=Instant::now();
    let lattice_modes=explicit.sets.modes();
    let nu_max=lattice_modes.iter().map(|q|prepared.nu(q).map(f64::abs)).try_fold(0_f64,|a,v|v.map(|v|a.max(v)))?;
    let rule=FoldRule::new(longest,nu_max);
    let mut fold=FoldLibrary::build(&prepared.generator,&prepared.plan,&prepared.lapack,Storage::Rule(rule))?;
    let rings=completion.map_or(0,|c|c.rings);
    if rings>0{
        // The ring targets land on the lattice's edges whose charges the plan
        // folds (and the diagonal).
        let known:BTreeSet<Charge>=prepared.plan.targets.iter().chain(&prepared.plan.omitted).map(|t|t.charge.clone()).chain(std::iter::once(prepared.generator.zero.clone())).collect();
        let mut offsets=BTreeSet::new();for p in &lattice_modes{for q in &lattice_modes{let t=charge_minus(q,p)?;if known.contains(&t){offsets.insert(t);}}}
        let ring_targets=fold.ring_targets(&prepared.generator,&prepared.plan,rings,&offsets)?;fold.targets.extend(ring_targets);
    }
    let mut report=FoldReport{rule:Some(rule),orders:fold.diagnostics.orders,dropped:fold.diagnostics.dropped.clone(),targets:fold.targets.len(),build_ms:clock.elapsed().as_secs_f64()*1e3,rings,..Default::default()};
    let clock=Instant::now();
    let mut lattice=Lattice::new(prepared,&fold,explicit.sets.clone())?;
    let budget=MEMORY_SHARE*crate::memory::physical_bytes()?;
    let planned=planned_bytes(request.runs,lattice.modes.len());
    if planned>budget{return Err(format!("The lattice of this request has {} modes and would need about {:.1} GB, above the {:.0} GB memory budget: not solved",lattice.modes.len(),planned/1e9,budget/1e9));}
    // 2. The space: translations, the steady column.
    let capacity={let op=lattice.operator();let d=op.n*op.n;BASIS_BYTES/(std::mem::size_of::<C>()*2*d*op.modes.len())};
    let mut space=Space::new(&mut lattice,initial,capacity)?;
    for s in charge_set::gated_steps(prepared,tolerance){if lattice.operator().index.contains_key(&s){space.activate(&mut lattice,&s)?;}}
    timing.setup_ms=clock.elapsed().as_secs_f64()*1e3;
    space.solve(&mut lattice,Pole::Zero)?;
    let modes=lattice.modes.clone();
    let clock=Instant::now();
    let mut estimator=UnifiedEstimate::new(EstimateSetup{prepared,lattice:&explicit.sets,outputs:Outputs{times,horizons:request.horizons,groups:request.groups},frame,fold:&fold})?;
    timing.estimator_build_ms=clock.elapsed().as_secs_f64()*1e3;
    // The ladder spans the grid's times when there is one (a horizon is
    // served by the steady column): its poles near 1 / T of a 1 s horizon grew
    // spurious near-zero slow Ritz pairs in both metrics (rates 1e-8 to
    // 1e-5, residuals 0.2-3) whose p(1 s) bounds reached 1e1-1e4; without
    // them three, four, five lasers and four without the trap at 20 us were
    // accepted at 16-24 solves (four without the trap partial before) and
    // five lasers at 10 us / 2000 samples reached 7.0e-3 against 2.3e-2.
    let points=space::ladder(eigen,initial,t1,times.last().copied().unwrap_or(longest),tolerance)?;let mut alive=vec![true;points.len()];
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
    let mut best:Option<(Answer,RitzModes)>=None;let mut progress:Vec<f64>=Vec::new();
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
        if exhausted||space.solves()%CHECKPOINT==0||(times.is_empty()&&space.solves()==1){
            // 3. The checkpoint: model, outputs, estimate, bounds.
            let clock=Instant::now();
            galerkin.ingest(&space.basis,&space.images)?;
            let model=galerkin.model(lapack,&space.initial,longest,tolerance)?;timing.model_ms+=clock.elapsed().as_secs_f64()*1e3;
            timing.projection_ms=galerkin.milliseconds;timing.images_ms=space.images_ms;
            let outputs_clock=Instant::now();
            let mut trajectory=model.trajectory(&galerkin,times);
            if let Some(first)=trajectory.first_mut(){*first=(0..n).map(|i|initial[i*n+i].re).collect();}
            let horizons:Vec<Vec<f64>>=request.horizons.iter().map(|&t|model.horizon(&galerkin,t)).collect();
            timing.outputs_ms+=outputs_clock.elapsed().as_secs_f64()*1e3;
            let estimate_clock=Instant::now();
            let mut estimate=estimator.estimate(&ReducedView{basis:&space.basis,images:&space.images,r:model.r,vectors:&model.vectors,
                used_theta:&model.used_theta,used_coefficients:&model.used_coefficients,kernel:&model.kernel,projected:&model.projected_out},&mut lattice)?;
            timing.estimate_ms+=estimate_clock.elapsed().as_secs_f64()*1e3;
            // The growth charge on every output.
            for v in estimate.trajectory.iter_mut().chain(estimate.horizons.iter_mut()){*v+=model.growth_charge;}
            // The slowest relaxation.
            let pairs=slow_pairs(&model,&space,&ops,&classes,tolerance)?;
            // The outputs with their bounds.
            let answer={
                let outputs_clock=Instant::now();
                let projected:Vec<Vec<f64>>=trajectory.iter().map(|p|simplex(p)).collect();
                let changes:Vec<f64>=trajectory.iter().zip(&projected).map(|(a,b)|ledger::metric(a,b,request.groups)).collect();
                let sample_errors:Vec<f64>=estimate.trajectory.iter().zip(&changes).map(|(e,p)|ACCEPTANCE_FACTOR*e+p).collect();
                let basis:Vec<&[C]>=space.basis.iter().map(|v|v.as_slice()).collect();
                let horizon_fields=request.horizons.iter().map(|&t|Ok(ops.combinations(&basis,&[model.state(t)])?.remove(0))).collect::<Result<Vec<_>,String>>()?;
                timing.outputs_ms+=outputs_clock.elapsed().as_secs_f64()*1e3;
                let mut answer=Answer{solves:space.solves(),modes:modes.clone(),trajectory:projected,sample_errors,
                    horizons:horizons.clone(),horizon_fields,horizon_errors:estimate.horizons.iter().map(|e|ACCEPTANCE_FACTOR*e).collect(),
                    estimate:estimate.clone(),slowest:pairs.first().map(|p|p.0),accepted:false};
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
            if best.as_ref().is_none_or(|(b,_)|if within{true}else{b.largest()>tolerance&&answer.largest()<b.largest()}){best=Some((answer.clone(),RitzModes::of(&model,space.basis.len())));}
            // 5. The first checkpoint's decision (see `solve`): the far shell's
            // depth on the reduced solution, and the near ring's completion.
            if completion.is_none()&&checkpoints.len()==1{
                let ritz=RitzModes::of(&model,space.basis.len());
                let left=FOLD_SHARE*tolerance-report.dropped.total();
                let actions=if prepared.plan.far().is_empty(){Vec::new()}else{ring_actions(&fold,prepared,&modes,&ritz,&space.basis,longest,left)?};
                // The depth: the fewest rings whose rest stays within what the
                // build left of the rule's budget.
                let mut depth=actions.len();while depth>0&&actions[depth-1..].iter().sum::<f64>()<=left{depth-=1;}
                let e=&answer.estimate;let off=answer.sample_errors.len();
                // Per output: its bound and its slow part (bound units); the steady
                // state's estimate is its slow part.
                let mut outputs:Vec<(f64,f64,usize)>=answer.sample_errors.iter().enumerate()
                    .map(|(i,&b)|(b,ACCEPTANCE_FACTOR*e.slow_trajectory.get(i).copied().unwrap_or(0.),i)).collect();
                outputs.extend(answer.horizon_errors.iter().enumerate().map(|(h,&b)|(b,ACCEPTANCE_FACTOR*e.slow_horizons.get(h).copied().unwrap_or(0.),off+h)));
                let(bound,part,at)=outputs.iter().copied().max_by(|a,b|a.0.total_cmp(&b.0)).unwrap_or((0.,0.,0));
                let(mut promoted,mut grown,mut ring_part,mut attribution)=(Vec::new(),None,0.,Vec::new());
                if bound>tolerance&&part>=COMPLETION_SHARE*bound{
                    let view=ReducedView{basis:&space.basis,images:&space.images,r:model.r,vectors:&model.vectors,
                        used_theta:&model.used_theta,used_coefficients:&model.used_coefficients,kernel:&model.kernel,projected:&model.projected_out};
                    let pick=|x:&ErrorEstimate|ACCEPTANCE_FACTOR*if at<off{x.trajectory[at]}else{x.horizons[at-off]};
                    let floor=COMPLETION_FLOOR*tolerance;
                    // The ring charges by attribution, a pair together (the canonical
                    // member, the larger charge, stands for it), and the ring's part.
                    let mut pairs:BTreeMap<Charge,f64>=BTreeMap::new();
                    for(c,a)in estimator.ring_attribution(&view)?{let m=opposite(&c);let key=if c>m{c}else{m};*pairs.entry(key).or_default()+=a;}
                    let total:f64=pairs.values().sum();
                    let mut ranked:Vec<(Charge,f64)>=pairs.into_iter().collect();ranked.sort_by(|a,b|b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                    ring_part=if total>0.{pick(&estimator.estimate_alone(&view,&mut lattice,Source::Ring)?)}else{0.};
                    let mut carried=0.;
                    if total>0.{for(c,a)in &ranked{if ring_part*(1.-carried/total)<floor{break;}promoted.push(c.clone());carried+=a;}}
                    // The completed lattice must fit the memory budget: else no
                    // charge is promoted.
                    if !promoted.is_empty(){
                        let set=charge_set::completed(prepared,&explicit,&promoted)?;
                        if planned_bytes(request.runs,set.sets.modes().len())>budget{promoted.clear();}else{grown=Some(set);}
                    }
                    attribution=ranked.iter().map(|(c,a)|(c.clone(),if total>0.{a/total}else{0.})).collect();
                }
                let restarted=depth>rings||!promoted.is_empty();
                let decision=Completion{promoted,explicit:grown.unwrap_or_else(||explicit.clone()),rings:depth.max(rings),ring_actions:actions,decided_at:space.solves(),bound,slow_part:part,ring_part,attribution,restarted};
                if restarted{return Ok(Err((decision,checkpoints)));}
                decided=Some(decision);
            }
            if answer.accepted{break;}
            // 4. The stopping rule on the bound (a checkpoint within the
            // tolerance is accepted).
            progress.push(answer.largest());
            let last=progress.len()-1;
            let plateau=last>=PLATEAU&&{
                let before=progress[..=last-PLATEAU].iter().copied().fold(f64::INFINITY,f64::min);
                let recent=progress[last+1-PLATEAU..].iter().copied().fold(f64::INFINITY,f64::min);
                recent>before/PLATEAU_GAIN};
            if exhausted||plateau{
                let b=&best.as_ref().ok_or("A run has its checkpoint")?.0;
                let why=if plateau{format!("the error bound stopped improving ({PLATEAU_GAIN}x over {PLATEAU} checkpoints) after {} solves",space.solves())}
                    else if space.full(){format!("the basis memory was spent after {} solves",space.solves())}
                    else if live.is_empty(){format!("the pole ladder was exhausted after {} solves",space.solves())}
                    else{format!("the solve budget ({MAXIMUM_SOLVES}) was spent")};
                reason=Some({
                        let over:Vec<usize>=(0..b.sample_errors.len()).filter(|&i|b.sample_errors[i]>tolerance).collect();
                        let samples=match(over.first(),over.last()){(Some(&a),Some(&z))=>format!("; {} of {} samples above it, t = {:.4} to {:.4} us",over.len(),b.sample_errors.len(),times[a],times[z]),_=>String::new()};
                        let means=b.horizon_errors.iter().filter(|&&e|e>tolerance).count();
                        let means=if means>0{format!("; {means} mean output(s) above it")}else{String::new()};
                        format!("Stopped because {why}: the best checkpoint ({} solves) has an error bound of {:.2e} against the tolerance {tolerance:e}{samples}{means}",b.solves,b.largest())});
                break;
            }
            current=Some((space.solves(),model));
        }
        // The next pole by its expected gain.
        let clock=Instant::now();
        let model=match current.take(){Some((solves,model))if solves==space.solves()=>model,_=>{
            galerkin.ingest(&space.basis,&space.images)?;galerkin.model(lapack,&space.initial,longest,tolerance)?}};
        let xis:Vec<C>=live.iter().map(|&i|points[i].sigma()).collect();
        let transform=model.error_transform(&galerkin,&space.basis,&space.images,eigen,lapack,&xis)?;
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
            if space.solve(&mut lattice,points[i])?==0{alive[i]=false;}
            if space.solves()%CHECKPOINT==0||space.solves()>=MAXIMUM_SOLVES||space.full(){break;}
        }
    }
    let(mut answer,ritz)=best.ok_or("A run has its checkpoint")?;
    // The a-posteriori term on the best checkpoint's reduced solution.
    let clock=Instant::now();
    let(grid,means,dressing)=posterior(&lattice,prepared,&ritz,&space.basis,times,request.horizons,longest,FOLD_SHARE*tolerance)?;
    for(e,v)in answer.sample_errors.iter_mut().zip(&grid){*e+=v;}
    for(e,v)in answer.horizon_errors.iter_mut().zip(&means){*e+=v;}
    answer.accepted=answer.largest()<=tolerance;
    report.posterior=grid.iter().chain(&means).fold(0_f64,|a,&b|a.max(b));report.dressing=dressing;report.posterior_ms=clock.elapsed().as_secs_f64()*1e3;
    if !answer.accepted&&reason.is_none(){reason=Some(format!("The fold's a-posteriori term ({:.2e}) took the best checkpoint's bound to {:.2e}, above the tolerance {tolerance:e}",report.posterior,answer.largest()));}
    let work=lattice.work().clone();
    timing.solves_ms=work.milliseconds;timing.images_ms=space.images_ms;timing.projection_ms=galerkin.milliseconds;timing.total_ms=started.elapsed().as_secs_f64()*1e3;
    timing.frame_ms=frame.milliseconds();
    let converged=answer.accepted;
    Ok(Ok(Solution{answer,converged,reason:if converged{None}else{reason},completion:completion.cloned().or(decided),translations:space.translations(),
        columns:space.columns.clone(),dimension:galerkin.columns(),images:space.basis.len(),work,fold:report,timing,checkpoints,explicit:explicit.clone()}))
}

#[cfg(test)]
mod tests{
    use super::*;
    use super::super::tests::{fixture,prepared,dense_steady};
    use super::super::operator::LatticeOperator;
    #[test]
    fn lattice_pipeline_follows_the_exact_lattice_dynamics(){
        // The fixture from |0><0|: the trajectory (2 us) and the state at a
        // horizon against the same lattice P u Q propagated densely (RK4, real
        // form), and the steady column against its dense null vector. The far
        // drives at 2% of the fixture's.
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
        // vector.
        let mut lattice=Lattice::new(&p,&p.fold,s.explicit.sets.clone()).unwrap();let mut u=vec![ZERO;size];for i in 0..n{u[op.centre*d+i*n+i]=C{re:0.25,im:0.};}
        let(w,_,_)=lattice.solve(Pole::Zero,&[u],1e-13).unwrap();let trace:f64=(0..n).map(|i|w[0][op.centre*d+i*n+i].re).sum();
        let column:Vec<f64>=(0..n).map(|i|w[0][op.centre*d+i*n+i].re/trace).collect();
        let tight=ledger::metric(&exact,&column,&groups);assert!(tight<1e-10,"bordered column {tight:e}");
        assert!(a.slowest.is_some());
        // Projected rows: probabilities, the first one rho0's.
        assert!(a.trajectory.iter().all(|r|r.iter().all(|&v|v>=0.)&&(r.iter().sum::<f64>()-1.).abs()<1e-12)&&a.trajectory[0][0]==1.);
    }
}
