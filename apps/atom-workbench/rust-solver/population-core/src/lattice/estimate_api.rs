//! The interface between the population pipeline and its error estimate.
//!
//! The pipeline builds one rational Krylov space W on the lattice P u Q (the
//! explicit charges P, then their ring Q; every mode set closed under
//! q -> -q) and, at each checkpoint, one reduced model x(t) ~ W X u(t),
//! u' = G u, u(0) = c0, from which every output is read: the trajectory on
//! the grid (lab readout, the local time average), the states at the
//! horizons (centre readout = the laboratory time average at T) and the
//! steady state. The estimator receives the space and the reduced model and
//! returns an error estimate per output; ACCEPTANCE_FACTOR times it is the
//! output's bound, and the pipeline accepts the checkpoint when every bound
//! is at most `ledger::POPULATION_TOLERANCE`.
//!
//! Fields: Hermitian-symmetric (x_-q = x_q^dagger, the centre Hermitian) on
//! the modes of `LatticeSets::modes`, n x n row-major blocks, every mode
//! stored (mirrored blocks consistent). Inner products are real,
//! Re <a, b> summed over every mode, so all reduced quantities are real.
//! Populations of a field at time t: sum_q e^{i nu_q t} diag(x_q) (real
//! part), nu_q = `LatticePrepared::nu(q)`; at a horizon the centre's diagonal
//! alone. Errors are measured with `ledger::metric` (largest difference over
//! the states and the requested groups).
//!
//! Lifetime: the pipeline constructs one estimator per request, after the
//! lattice is fixed, and estimates every checkpoint. Basis columns are append-only: column j of `ReducedView::basis`
//! and `images` never changes once given, so per-column work may be cached
//! across checkpoints (the translated copies are columns like any other).
//! X, G, c0 and the Ritz data are recomputed at every checkpoint.
use super::*;
use super::ledger::RemainderLedger;
use super::frame::SwFrame;

/// An output's bound is ACCEPTANCE_FACTOR x its estimated error; a checkpoint
/// is accepted when every bound is within POPULATION_TOLERANCE. Measured
/// (`pipeline_against_a_reference`, 20 us / 200 samples, every checkpoint
/// whose actual error is above 2% of the tolerance): the smallest ratio of
/// the time-maximum estimate to the actual error is 0.89 over the research
/// requests (probe + tweezer, two lasers without the trap, three, four and
/// five lasers; 1 / 0.89 = 1.12 below the factor). Four lasers without the
/// trap gave 0.77 at its accepted checkpoint (estimate 1.7e-3, "actual"
/// 2.2e-3), but against its older reference, which is accurate only to about
/// 1.2-1.5e-3: that actual is within the reference's own uncertainty, so it
/// is no evidence against the factor.
pub const ACCEPTANCE_FACTOR:f64=1.25;

/// The lattice of a run: `modes()` = explicit (P) then ring (Q), the centre
/// in P. The space lives on these modes; everything beyond them is omitted.
#[derive(Clone,Debug)]
pub struct LatticeSets{pub explicit:Vec<Charge>,pub ring:Vec<Charge>}
impl LatticeSets{
    pub fn modes(&self)->Vec<Charge>{self.explicit.iter().chain(&self.ring).cloned().collect()}
    /// Index of the zero charge in `modes()`.
    pub fn centre(&self)->Option<usize>{self.explicit.iter().position(|q|q.iter().all(|&k|k==0))}
}
/// The charges one structural step (`HarmonicPlan::structural_steps`)
/// outside `set`, sorted by (|q|_1, q): the ring of P is Q, the ring of
/// P u Q the omitted shell R the estimate slaves.
pub fn ring(prepared:&LatticePrepared,set:&[Charge])->Result<Vec<Charge>,String>{
    let inside:BTreeSet<&Charge>=set.iter().collect();let mut out=BTreeSet::new();
    for q in set{for s in &prepared.plan.structural_steps(){let r=charge_plus(q,s)?;if !inside.contains(&r){out.insert(r);}}}
    let mut out:Vec<Charge>=out.into_iter().collect();
    out.sort_by_key(|q|(q.iter().map(|&k|k.unsigned_abs()as u64).sum::<u64>(),q.clone()));
    Ok(out)
}

/// What a request asks for.
#[derive(Clone,Copy,Debug)]
pub struct Outputs<'a>{
    /// The uniform grid, starting at 0 (us).
    pub times:&'a[f64],
    /// The horizons T (us) whose time-averaged state is reported.
    pub horizons:&'a[f64],
    /// Whether the steady state (t -> infinity mean) is reported.
    pub steady:bool,
    /// Requested population groups (state indices), for `ledger::metric`.
    pub groups:&'a[Vec<usize>],
}

/// Everything fixed for one request.
pub struct EstimateSetup<'a>{
    /// Generator, plan, fold, `floquet` (the shared FloquetKernels), the
    /// shared real eigenbasis of L0' (`eigenbasis()`), LAPACK.
    pub prepared:&'a LatticePrepared,
    pub lattice:&'a LatticeSets,
    pub outputs:Outputs<'a>,
    /// The request's Schrieffer-Wolff / Floquet frame (built for the
    /// longest output time).
    pub frame:&'a SwFrame<'a>,
    /// The fold remainders as generator-norm bounds (rad/us) for the
    /// lattice's largest |nu_q|.
    pub ledger:&'a RemainderLedger,
    /// Whether the lattice operator keeps every K^(1) rest (a completed
    /// lattice): the error lattice's outer operator does the same.
    pub keep_k1:bool,
}

/// The reduced model of one checkpoint.
pub struct ReducedView<'a>{
    /// W: orthonormal (Re <w_i, w_j> = delta_ij) fields on the lattice's
    /// modes, w_0 = x0 / |x0|, and their images A W (computed explicitly with
    /// the lattice operator on P u Q).
    pub basis:&'a[Vec<C>],pub images:&'a[Vec<C>],
    /// The Ritz decomposition of the Galerkin model G = Re W^T A W (with the
    /// trace correction; not a pure projection: the residual A W - W G is
    /// what the outputs leave out): G S = S diag(theta), S (r x r,
    /// row-major, column j the eigenvector of theta_j), r = basis.len(). The
    /// outputs use u(t) = S diag(e^{used_theta t}) used_coefficients: the
    /// values and coefficients differ from theta and S^-1 c0 only on growing
    /// modes (small amplitude projected out with theta = a = 0 and its
    /// readout charged by the pipeline, any other clipped to the imaginary
    /// axis: the residual of the clipped pair, read from the images, carries
    /// it) and on the kernel (theta = 0 exactly; `kernel`, never a projected
    /// mode).
    pub r:usize,pub vectors:&'a[C],pub used_theta:&'a[C],pub used_coefficients:&'a[C],
    /// Per Ritz mode: the kernel, and a growing mode projected out (the
    /// estimator reads these flags, never theta or a being zero: a clipped
    /// real growing mode also has theta = 0).
    pub kernel:&'a[bool],pub projected:&'a[bool],
}

/// The pipeline's own lattice P u Q at the steady shift: its operator, its
/// steady-shift preconditioner, the bordered system (A + u tau0^T) y = f,
/// u = I/n on the centre, tau0 the centre trace (as the pole-0 column), and
/// the inner directions of the pipeline's own steady solve. Fields are
/// Hermitian-symmetric on the lattice's modes.
pub trait SteadySolves{
    /// y = A x.
    fn apply(&mut self,x:&[C],y:&mut[C])->Result<(),String>;
    /// y = M0^-1 x: the Floquet frames with the centre Woodbury term and the
    /// trace row at the steady shift.
    fn precondition(&mut self,x:&[C],y:&mut[C])->Result<(),String>;
    /// The pole-0 (steady) solve's inner directions z_j = P^-1 v_j and their
    /// images A z_j (A unbordered) on P u Q, mirror-completed; append-only:
    /// an entry never changes once handed out.
    fn directions(&self)->(&[Vec<C>],&[Vec<C>]);
    /// The K^(1) rests the operator leaves out applied to a
    /// Hermitian-symmetric field (`LatticeOperator::left_out`; None when it
    /// leaves none out).
    fn left_out(&self,x:&[C])->Result<Option<Vec<C>>,String>;
}

/// Work of one estimate, for the diagnostics.
#[derive(Clone,Debug,Default)]
pub struct EstimateWork{
    /// Operator or preconditioner applications and exact block solves,
    /// counted per n x n block (one lattice block = 1).
    pub block_operations:usize,pub block_solves:usize,
    /// Time (ms).
    pub milliseconds:f64,
    /// Slow modes solved, whether the trajectory took the slow part, and
    /// the slowest decay rate the time rule read (1/us).
    pub slow_modes:usize,pub slow_trajectory:bool,pub lambda_slow:f64,
}

/// Estimated errors of one checkpoint's outputs (population metric).
#[derive(Clone,Debug,Default)]
pub struct ErrorEstimate{
    /// One per grid time.
    pub trajectory:Vec<f64>,
    /// One per horizon.
    pub horizons:Vec<f64>,
    /// The steady state (None when not requested).
    pub steady:Option<f64>,
    /// The slow part alone (the ring's feedback and the K^(1) rests the
    /// lattice leaves out, through the slow modes; the kernel's own residual
    /// with them), per grid time and per horizon: what a completed lattice
    /// could remove (`pipeline`'s completion). The steady state's estimate is
    /// its slow part.
    pub slow_trajectory:Vec<f64>,pub slow_horizons:Vec<f64>,
    pub work:EstimateWork,
}
impl ErrorEstimate{
    /// The largest estimate over every output.
    pub fn largest(&self)->f64{
        self.trajectory.iter().chain(&self.horizons).chain(self.steady.iter()).fold(0_f64,|a,&b|a.max(b))
    }
}
