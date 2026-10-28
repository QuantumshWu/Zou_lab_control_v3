//! The interface between the population pipeline and its error estimate.
//!
//! The pipeline builds one rational Krylov space W on the lattice P u Q (the
//! explicit charges P, then their ring Q; every mode set closed under
//! q -> -q) and, at each checkpoint, one reduced model x(t) ~ W X u(t),
//! u' = G u, u(0) = c0, from which every output is read: the trajectory on
//! the grid (lab readout, the local time average), the states at the
//! horizons (centre readout = the laboratory time average at T) and the
//! steady state. The estimator receives the space and the reduced model and
//! returns an error bound per output; the pipeline accepts the checkpoint
//! when ACCEPTANCE_FACTOR times the largest of them is at most
//! `ledger::POPULATION_TOLERANCE`.
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
//! lattice is fixed (`Construct`), and calls `Estimator::estimate` at every
//! checkpoint. Basis columns are append-only: column j of `ReducedView::basis`
//! and `images` never changes once given, so per-column work may be cached
//! across checkpoints (the translated copies are columns like any other).
//! X, G, c0 and the Ritz data are recomputed at every checkpoint.
use super::*;
use super::ledger::RemainderLedger;
use super::frame::SwFrame;

/// Acceptance: ACCEPTANCE_FACTOR x (largest estimated error over the grid,
/// the horizons and the steady state) <= POPULATION_TOLERANCE.
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
    /// Generator, plan, fold, `floquet` (the shared FloquetKernels),
    /// `eigenbasis()` (the shared real eigenbasis of L0' with exact batched
    /// block solves), LAPACK.
    pub prepared:&'a LatticePrepared,
    pub lattice:&'a LatticeSets,
    pub outputs:Outputs<'a>,
    /// The request's Schrieffer-Wolff / Floquet frame (built for the
    /// longest output time).
    pub frame:&'a SwFrame,
    /// The fold remainders as generator-norm bounds (rad/us) for the
    /// lattice's largest |nu_q|: the fold charges.
    pub ledger:&'a RemainderLedger,
    /// rho0 (n x n row-major); x0 is rho0 on the centre mode.
    pub initial:&'a[C],
}

/// The reduced model of one checkpoint.
pub struct ReducedView<'a>{
    /// W: orthonormal (Re <w_i, w_j> = delta_ij) fields on the lattice's
    /// modes, w_0 = x0 / |x0|, and their images A W (computed explicitly with
    /// the lattice operator on P u Q).
    pub basis:&'a[Vec<C>],pub images:&'a[Vec<C>],
    /// Reduced dimension r and the coordinates: x(t) ~ W X u(t) with X
    /// (k x r, row-major, k = basis.len()), u' = G u (r x r, row-major),
    /// u(0) = c0. G is the Galerkin model in the Schrieffer-Wolff / Floquet
    /// frame's inner product with the trace correction (not a pure
    /// projection: the residual A W X - W X G is what the outputs leave
    /// out); X = I there.
    pub r:usize,pub x:&'a[f64],pub g:&'a[f64],pub c0:&'a[f64],
    /// Ritz decomposition G S = S diag(theta): theta (r), S (r x r,
    /// row-major, column j the eigenvector of theta_j), a = S^-1 c0 (r).
    pub theta:&'a[C],pub vectors:&'a[C],pub coefficients:&'a[C],
    /// The values and coefficients the outputs use: u(t) = S diag(e^{used_theta t}) used_coefficients.
    /// They differ from theta, a only on growing modes (small amplitude
    /// projected out with its readout charged in `growth_charge`, any other
    /// clipped to the imaginary axis: the residual of the clipped pair, read
    /// from the images, carries it) and on the kernel (theta = 0 exactly).
    pub used_theta:&'a[C],pub used_coefficients:&'a[C],pub kernel:&'a[bool],
    /// Population size of the modes projected out (charged by the pipeline).
    pub growth_charge:f64,
    /// The reduced steady state u_inf (r; G u_inf = 0 with the centre trace
    /// of W X u_inf equal to that of x0) when the kernel is simple.
    pub steady:Option<&'a[f64]>,
}

/// The pipeline's own lattice P u Q at the steady shift: its operator, its
/// preconditioner at the steady shift (the Floquet frames with the centre
/// Woodbury term and the trace row), the bordered system
/// (A + u tau0^T) y = f (u = I/n on the centre, tau0 the centre trace, as
/// the pole-0 column) and the inner directions of the pipeline's own
/// steady solve. Fields are Hermitian-symmetric on the lattice's modes.
pub trait SteadySolves{
    /// y for f, with the inner steps it took.
    fn solve(&mut self,f:&[C],tolerance:f64)->Result<(Vec<C>,usize),String>;
    /// y = A x.
    fn apply(&mut self,x:&[C],y:&mut[C])->Result<(),String>;
    /// y = M0^-1 x, the steady-shift preconditioner.
    fn precondition(&mut self,x:&[C],y:&mut[C])->Result<(),String>;
    /// The preconditioned inner directions z of the pipeline's steady solve
    /// with their images A z, handed over once (empty afterwards, and before
    /// that solve).
    fn directions(&mut self)->Vec<(Vec<C>,Vec<C>)>;
}

/// Work of one estimate, for the diagnostics.
#[derive(Clone,Debug,Default)]
pub struct EstimateWork{
    /// Operator or preconditioner applications and exact block solves,
    /// counted per n x n block (one lattice block = 1).
    pub block_operations:usize,pub block_solves:usize,
    /// Inner steps of steady solves.
    pub inner_steps:usize,
    pub milliseconds:f64,
}

/// Error bounds of one checkpoint's outputs (population metric).
#[derive(Clone,Debug,Default)]
pub struct ErrorEstimate{
    /// One per grid time.
    pub trajectory:Vec<f64>,
    /// One per horizon.
    pub horizons:Vec<f64>,
    /// The steady state (None when not requested).
    pub steady:Option<f64>,
    pub work:EstimateWork,
}
impl ErrorEstimate{
    /// The largest bound over every output.
    pub fn largest(&self)->f64{
        self.trajectory.iter().chain(&self.horizons).chain(self.steady.iter()).fold(0_f64,|a,&b|a.max(b))
    }
}

/// The error estimate of one run.
pub trait Estimator{
    fn estimate(&mut self,reduced:&ReducedView<'_>,steady:&mut dyn SteadySolves)->Result<ErrorEstimate,String>;
}
/// An estimator's constructor (one per request, after the lattice is fixed).
pub type Construct=for<'a> fn(EstimateSetup<'a>)->Result<Box<dyn Estimator+'a>,String>;
