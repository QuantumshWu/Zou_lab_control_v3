//! Long-time (Cesàro) mean of the complete driven Lindblad equation, computed
//! on the same exact gauge pencil `E x' = F x` as the finite-time engine.
//!
//! # Mathematics
//!
//! The envelope field `x = (x_q)` lives on integer Fourier charges `q`. The
//! model-frame density is `rho(t) = D(theta) R[(E x(t))(theta)] D(theta)^+`
//! with `theta = Omega t`, `R` the Hilbert-basis restore and
//! `D = diag(exp(i g_s . theta))` the state gauge (see `from_envelope_at`).
//!
//! `E` is a Fourier convolution and `F = G - i diag(mu . Omega) E`, so with the
//! index shift `(S_p x)_q = x_(q-p)` the pencil is shift covariant:
//! `E S_p = S_p E` and `F S_p = S_p (F - i p.Omega E)`. Every eigenvalue
//! `lambda` appears as the whole family `lambda - i p.Omega`. The laboratory
//! time average of `rho_ij` (state frame index `f`, ground/excited sector
//! `sigma`) collects the stationary member of every family and equals
//!
//! ```text
//!   M_ij = [sigma_i = sigma_j] R( (E Pi0 X0)_(h_i - h_j) )_ij,   h = f - g,
//! ```
//!
//! where `Pi0` is the spectral projector of the pencil at eigenvalue zero and
//! `X0` is the initial envelope point `y0` placed on EVERY Fourier mode (the
//! `theta = 0` delta function). If the null space is one-dimensional,
//! `Pi0 X0 = w`, the null vector normalised to unit central trace, and the
//! mean does not depend on the initial state. Otherwise the mean depends on
//! the conserved quantities evaluated at `theta = 0`, and
//! `Pi0 X0 = lim_(s -> 0+) s (sE - F)^-1 E X0`. Note that `Pi0` applied to the
//! central initial vector alone would give the mean averaged over the initial
//! laser phases, which differs whenever the null space is degenerate.
//!
//! # Algorithm and error control
//!
//! Conserved quantities and relaxation: the slow-manifold generator `J_eff`
//! (`slow_dual.rs`, a dense eigenproblem without a pole) gives the number of
//! conserved quantities (its modes at zero, up to its numerical resolution)
//! and the slowest relaxation rate of the others, however small. More than
//! one conserved quantity makes the mean initial-dependent; otherwise the
//! limit is unique and `slowest_relaxation_us` says when it is reached.
//!
//! One-dimensional null space (the common case):
//! 1. Trial space: the initial vector (and, when a relaxation is slower than
//!    the pole resolves, the slow generator's conserved and slow modes lifted
//!    to the domain), plus residual-driven
//!    correction columns `delta = (s E - F)^-1 F w` at a near-zero pole `s`
//!    (a thousandth of the slowest relaxation rate, within 1e-10..1e-8/us).
//!    The reduced null vector `w` (Galerkin, exact trace conservation, no
//!    pole) converges in a handful of columns; inner solves only shape the
//!    space.
//! 2. Basis error: `w + delta` is the Newton update of the null vector, so the
//!    laboratory readout of `delta` estimates the error of `w`. A slow mode of
//!    rate `gamma` is resolved by the pole with relative bias `s / gamma`; the
//!    estimate is inflated by `1 + s/gamma` with the slowest rate of the slow
//!    generator (or, without it, of the reduced model above the resolvable
//!    rate `100 s`). The correction solves are accurate (the unresolved part of
//!    an inexact solve sits in the slow, most amplified directions), and every
//!    estimate is checked against the change actually observed after the
//!    correction enters the space; an under-prediction raises the factor
//!    applied to later estimates (self-calibration).
//! 3. Fourier truncation (halo): the residual of `w` on the neighbouring modes
//!    is condensed through the mode-diagonal P0 inverse, fed back into the
//!    domain and propagated by one more correction solve. Per-mode halo
//!    indicators select the modes added when the halo dominates. Every
//!    decision here is made by magnitude, never by exact zeros: the starting
//!    domain follows the steps whose mixing amplitude `|H_q| / |omega_q|` is
//!    significant, the halo covers every step, and halo modes with negligible
//!    transport amplitude are condensed by the transport inverse and charged
//!    twice (a P0 shift factor per mode is the dominant halo cost).
//! 4. The corrected laboratory mean is projected onto the trace-one positive
//!    cone (Frobenius-nearest density matrix, which cannot increase the
//!    Frobenius error) and that change is charged to the budget.
//! 5. Accept when `SAFETY * (basis + halo) + projection <= epsilon`.
//!
//! # Observation horizon
//!
//! With `MeanOptions::horizon_us = Some(T)` the result is the mean at `T`:
//! what has decayed by `T` is gone, oscillations faster than the drives'
//! beats average out exactly as above (the laboratory time average of the
//! state at `T`), and the slow modes are propagated over `T` from the
//! initial state. Decided by size, never by a guessed rate:
//! * the slow generator bounds what is still relaxing at `T`: every
//!   non-conserved mode with its population weight `|R_k c_k|` (initial
//!   central ground block `y0`) and its decay `exp(Re(alpha_k + i nu) T)`
//!   at half its rate (`SlowDual::horizon_bound`; the reduction's rates are
//!   measured within 12% of the exact ones). Below `HORIZON_NEGLIGIBLE`
//!   of the tolerance the result is the limit and the bound is charged;
//! * otherwise the state at `T` comes from the finite-time engine on the
//!   window (its estimate covers the whole window), read with the
//!   laboratory time average; a conserved quantity or a rate of 1e-12 are
//!   then propagated alike, so the result is continuous in the rates and
//!   tends to the limit as `T` grows.
//! Propagating with `exp(J_eff T)` alone was measured 4e-3 off on a toy
//! whose slow rate the reduction gets 12% high, which is why the generator
//! only decides and bounds here.
//!
//! Degenerate null space: a second conserved mode of the slow generator, or
//! (without it, or when every relaxation is resolvable) a second reduced
//! eigenvalue below the resolvable rate or a probe comparing the corrected
//! mean with `Pi0` of a second vector (`X0` for driven models, the mixed
//! state for static ones). `Pi0 X0` is then evaluated with the resolvent at
//! two poles and first-order Richardson extrapolation (refused when a
//! relaxation is slower than the pole resolves, and when a resolvent solve
//! stalls); Fourier convergence is checked by repeating the evaluation with
//! every neighbouring mode along every Fourier step added.
use super::*;
use super::directions::{shift_invert, InnerSolve};
use super::engine::{self, support_domain, FiniteOptions, RequestedGroup};
use super::reduced::ReducedModel;
use super::runtime_core::RuntimeCore;
use super::slow_dual::{Dense, SlowDual, SlowDualOptions, SlowSpectrum};
use super::trial::{DomainOperator, TrialSpace};

/// Largest near-zero pole (1/us). The pole is chosen from the relaxation
/// spectrum of the slow generator `J_eff` (`SlowDual::slow_spectrum`, which
/// resolves arbitrarily small rates without a pole): `POLE_RESOLUTION` times
/// its slowest relaxation rate, within `POLE_FLOOR..=POLE_MAX`.
const POLE_MAX: f64 = 1e-8;
/// Smallest pole. Slower relaxations are not resolved by the pole but by the
/// trial space: it starts with the slow generator's conserved and slow modes
/// (`seed_slow_modes`), so the Galerkin null vector (no pole) carries them,
/// and the pole only shapes corrections whose estimate is inflated by the
/// bias `1 + pole / rate`. (A correction solve at a pole a thousand times
/// below a 1e-9/us rate was observed to lose the slow direction.)
const POLE_FLOOR: f64 = 1e-10;
const POLE_RESOLUTION: f64 = 1e-3;
/// A reduced eigenvalue below this many poles is not separated from zero by
/// the pole (bias above 1%).
const RESOLVABLE_POLES: f64 = 100.;
/// Safety factor on the summed basis and halo estimates. Measured
/// effectivities of the correction readout are 0.92..1.00.
const SAFETY: f64 = 1.25;
/// Correction directions: accurate enough to resolve the slow manifold,
/// which carries the amplified part of the error.
const CORRECTION_SOLVE: InnerSolve = InnerSolve { maximum_steps: 80, relative_residual: 1e-4 };
/// Restarts of the correction solve when its residual stays large.
const CORRECTION_RESTARTS: usize = 3;
/// Inner residual above which an estimate is not trusted for acceptance.
const ESTIMATE_RESIDUAL_LIMIT: f64 = 1e-3;
/// Relative residual of the degeneracy probe per unit of tolerance (the probe
/// resolves differences of a quarter tolerance), and of the degenerate path.
const PROBE_TOLERANCE_PER_EPSILON: f64 = 1e-2;
const RESOLVENT_TOLERANCE: f64 = 1e-9;
/// A resolvent solve whose true relative residual stays above this multiple
/// of the requested one has stalled; its mean is not used.
const RESOLVENT_STALL: f64 = 10.;
/// Halo modes are added until they cover this fraction of the halo weight.
const HALO_COVERAGE: f64 = 0.9;
/// Halo modes whose transport amplitude `|h_q| / |s + i omega_q|` is below
/// this fraction of the largest one are condensed with the transport inverse
/// instead of the exact mode-diagonal inverse (one P0 shift factor per mode).
const FAR_HALO_AMPLITUDE: f64 = 1e-3;

/// Laboratory-frame metadata of the original states, in the SAME reduced
/// frequency basis as the prepared phases.
#[derive(Clone, Debug)]
pub struct LabFrame {
    /// Integer carrier-frame index of every original state.
    pub frame_indices: Vec<Charge>,
    /// Ground (0) or excited (1) sector of every original state.
    pub sectors: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct MeanOptions {
    /// Absolute tolerance on the probability scale (populations, requested
    /// group sums and the spectral norm of the density error).
    pub population_tolerance: f64,
    pub groups: Vec<RequestedGroup>,
    pub lab: LabFrame,
    pub maximum_columns: usize,
    pub maximum_modes: usize,
    pub wall_seconds: f64,
    /// Observation horizon (us); None is the t -> infinity mean.
    pub horizon_us: Option<f64>,
}

impl MeanOptions {
    pub fn new(population_tolerance: f64, groups: Vec<RequestedGroup>, lab: LabFrame) -> Self {
        Self { population_tolerance, groups, lab, maximum_columns: 80, maximum_modes: 600, wall_seconds: 120., horizon_us: None }
    }
}

/// How a horizon mean was obtained (`MeanOptions::horizon_us`).
#[derive(Clone, Debug)]
pub struct HorizonPart {
    pub horizon_us: f64,
    /// "limit": the slow part still relaxing at the horizon is negligible
    /// (its bound is charged); "propagation": the state at the horizon by
    /// the finite-time engine; "propagated-limit": the limit itself by
    /// propagation (the pole-based solve was not reliable).
    pub method: &'static str,
    /// Bound (from the slow generator, decay rates halved) on the largest
    /// population or group still relaxing at the horizon.
    pub decaying: f64,
    /// Estimate of the propagation (0 for "limit").
    pub propagation_estimate: f64,
    /// Charge for projecting the averaged density onto the density cone.
    pub projection_charge: f64,
}

/// How the long-time mean depends on the initial state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeanKind {
    /// One-dimensional null space: the same mean for every initial density.
    Unique,
    /// Several conserved quantities (zero modes of the slow generator; without
    /// it, of the reduced model or the resolvent probe): the mean depends on
    /// the initial density.
    InitialDependent { zero_modes: usize },
}

#[derive(Clone, Debug)]
pub struct MeanCheckpoint {
    pub columns: usize,
    pub modes: usize,
    pub pole: f64,
    pub basis_estimate: f64,
    pub halo_estimate: Option<f64>,
    /// Halo modes condensed with the exact mode-diagonal inverse and with the
    /// transport inverse.
    pub halo_modes: (usize, usize),
    /// Relative residual reached by the correction solve.
    pub inner_residual: f64,
    pub action: &'static str,
}

#[derive(Clone, Debug)]
pub struct MeanDiagnostics {
    pub estimated_error: f64,
    pub basis_estimate: f64,
    pub halo_estimate: f64,
    pub projection_charge: f64,
    pub minimum_eigenvalue_before_projection: f64,
    pub trace_error_before_projection: f64,
    pub hermiticity_error: f64,
    /// `|F x| / |E x|` in REAL coordinates for the returned null vector `x`
    /// (the corrected one, or the extrapolated resolvent vector).
    pub relative_residual: f64,
    pub slowest_rate: Option<f64>,
    /// Slowest relaxation time (us) of the decaying modes: `1 / rate` of the
    /// slowest relaxation of `J_eff`, or of the reduced model when the slow
    /// generator is unavailable; `None` when no decaying mode is resolved.
    /// A unique limit is reached after a few such times.
    pub slowest_relaxation_us: Option<f64>,
    /// Conserved quantities of `J_eff` (None when unavailable).
    pub conserved_quantities: Option<usize>,
    pub zero_modes: usize,
    pub pole: f64,
    pub columns: usize,
    pub modes: usize,
    pub operator_applications: usize,
    pub elapsed_ms: f64,
    /// Milliseconds: domain operators (pencil plans, P0 shifts), correction
    /// solves, halo condensation, degeneracy probe, slow-dual preparation
    /// (when `solve` prepares it).
    pub timing: [f64; 5],
    /// Diagonal of the uncorrected null-vector mean: the quantity whose error
    /// the basis estimate measures (kept for verification of the estimate).
    pub uncorrected_populations: Vec<f64>,
    pub checkpoints: Vec<MeanCheckpoint>,
    /// The horizon part (None for the t -> infinity mean).
    pub horizon: Option<HorizonPart>,
    /// The degeneracy probe's resolvent solve stalled and the slow
    /// generator, which found only the trace conserved, decided uniqueness.
    pub probe_stalled: bool,
    /// For a limit obtained by propagation: why the pole-based solve was not
    /// used (its error, or how far its mean was from the propagated one).
    pub limit_note: Option<String>,
}

pub struct MeanSolution {
    /// Laboratory-frame mean density, row-major `n x n`, trace one, positive.
    pub density: Vec<C>,
    /// Largest model-frame zero-charge coherence between the ground and
    /// excited sectors. It oscillates at an optical frequency in the
    /// laboratory frame: the time average exists but the density limit does
    /// not, although the populations may still converge.
    pub discarded_optical_coherence: f64,
    /// Its diagonal.
    pub populations: Vec<f64>,
    pub kind: MeanKind,
    pub diagnostics: MeanDiagnostics,
}

// ---------------------------------------------------------------------------
// Charge and small dense helpers.

fn charge_plus(a: &[i32], b: &[i32]) -> Charge {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}
fn charge_minus(a: &[i32], b: &[i32]) -> Charge {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}
fn euclid(values: &[f64]) -> f64 {
    values.iter().map(|x| x * x).sum::<f64>().sqrt()
}
fn euclid_complex(values: &[C]) -> f64 {
    values.iter().map(|z| z.re * z.re + z.im * z.im).sum::<f64>().sqrt()
}
fn complex_product(a: C, b: C) -> C {
    C { re: a.re * b.re - a.im * b.im, im: a.re * b.im + a.im * b.re }
}
fn reciprocal(z: C) -> C {
    let norm = z.re * z.re + z.im * z.im;
    C { re: z.re / norm, im: -z.im / norm }
}
fn ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.
}
/// Central-mode trace of a REAL column (its first `n` coordinates are the
/// central populations, each with unit weight).
fn central_trace(column: &[f64], n: usize) -> f64 {
    column[..n].iter().sum()
}
/// `Q a` for a column-major `rows x m` block.
fn lift(lapack: &Lapack, block: &[f64], rows: usize, coefficients: &[f64]) -> Result<Vec<f64>, String> {
    let m = coefficients.len();
    let mut out = vec![0.; rows];
    lapack.gemm(false, false, rows, 1, m, 1., block, rows, coefficients, m, 0., &mut out, rows)?;
    Ok(out)
}

// ---------------------------------------------------------------------------
// Laboratory-frame readout.

/// The charges `h_i - h_j` needed by every same-sector pair and the target
/// index of each pair.
struct Frame {
    n: usize,
    targets: Vec<Charge>,
    /// `(i, j, target)` for every same-sector pair.
    pairs: Vec<(usize, usize, usize)>,
    /// Cross-sector pairs `(i, j)` whose model-frame harmonic is the zero
    /// charge. Their laboratory average vanishes (optical frequency
    /// difference); a nonzero model-frame value means the density itself
    /// does not converge in the laboratory frame.
    cross: Vec<(usize, usize)>,
    zero_target: Option<usize>,
}

impl Frame {
    fn new(gauges: &[Charge], lab: &LabFrame) -> Result<Self, String> {
        let n = gauges.len();
        if lab.frame_indices.len() != n || lab.sectors.len() != n {
            return Err("Laboratory frame metadata must describe every original state".into());
        }
        if lab.frame_indices.iter().zip(gauges).any(|(f, g)| f.len() != g.len()) {
            return Err("Laboratory frame indices use another phase basis than the prepared gauges".into());
        }
        let h: Vec<Charge> = lab.frame_indices.iter().zip(gauges).map(|(f, g)| charge_minus(f, g)).collect();
        let mut index = BTreeMap::<Charge, usize>::new();
        let mut targets = Vec::new();
        let mut pairs = Vec::new();
        let mut cross = Vec::new();
        for i in 0..n {
            for j in 0..n {
                if lab.sectors[i] != lab.sectors[j] {
                    if h[i] == h[j] {
                        cross.push((i, j));
                    }
                    continue;
                }
                let charge = charge_minus(&h[i], &h[j]);
                let target = *index.entry(charge.clone()).or_insert_with(|| {
                    targets.push(charge);
                    targets.len() - 1
                });
                pairs.push((i, j, target));
            }
        }
        let zero_target = targets.iter().position(|q: &Charge| q.iter().all(|&k| k == 0));
        Ok(Self { n, targets, pairs, cross, zero_target })
    }

    /// Laboratory mean from `E x` on the target modes (Hilbert frame).
    fn assemble(&self, physical: &coordinates::PhysicalCoordinates, mass: &[C]) -> Result<Vec<C>, String> {
        let d = self.n * self.n;
        let restored = (0..self.targets.len())
            .map(|t| physical.hilbert_to_states(&mass[t * d..(t + 1) * d]))
            .collect::<Result<Vec<_>, _>>()?;
        let mut density = vec![C::default(); d];
        for &(i, j, t) in &self.pairs {
            density[i * self.n + j] = restored[t][i * self.n + j];
        }
        Ok(density)
    }
}

/// `E x` from one input-mode set onto the laboratory target modes.
struct TargetReadout {
    pencil: OwnedPencil,
    mass: Vec<C>,
    scratch: Vec<C>,
}

impl TargetReadout {
    fn new(core: &RuntimeCore, input: &[Charge], frame: &Frame) -> Result<Self, String> {
        let pencil = core.prepared().pencil(input, &frame.targets, 1)?;
        let length = pencil.output_dimension();
        Ok(Self { pencil, mass: vec![C::default(); length], scratch: vec![C::default(); length] })
    }
    /// Laboratory density of a signed complex field on the input modes.
    fn density(&mut self, physical: &coordinates::PhysicalCoordinates, frame: &Frame, signed: &[C]) -> Result<Vec<C>, String> {
        self.pencil.apply_into(signed, 1, &mut self.mass, &mut self.scratch)?;
        frame.assemble(physical, &self.mass)
    }
    /// Largest discarded cross-sector model-frame DC coherence of the last
    /// evaluated field.
    fn discarded(&self, physical: &coordinates::PhysicalCoordinates, frame: &Frame) -> Result<f64, String> {
        let Some(t) = frame.zero_target else { return Ok(0.) };
        let d = frame.n * frame.n;
        let restored = physical.hilbert_to_states(&self.mass[t * d..(t + 1) * d])?;
        Ok(frame.cross.iter().map(|&(i, j)| restored[i * frame.n + j].re.hypot(restored[i * frame.n + j].im)).fold(0_f64, f64::max))
    }
}

/// Error metric of a density difference: the largest of the state
/// populations, the requested group sums and the spectral norm (the largest
/// population change in ANY basis, which bounds background-basis projections).
fn error_metric(lapack: &Lapack, delta: &[C], n: usize, groups: &[RequestedGroup]) -> Result<f64, String> {
    let mut largest = (0..n).map(|s| delta[s * n + s].re.abs()).fold(0_f64, f64::max);
    for group in groups {
        largest = largest.max(group.indices.iter().map(|&s| delta[s * n + s].re).sum::<f64>().abs());
    }
    let (values, _) = lapack.hermitian_eigen(&hermitian_column_major(delta, n), n)?;
    Ok(largest.max(values.iter().map(|v| v.abs()).fold(0_f64, f64::max)))
}

/// Hermitian part, column-major (LAPACK layout).
fn hermitian_column_major(a: &[C], n: usize) -> Vec<C> {
    let mut out = vec![C::default(); n * n];
    for i in 0..n {
        for j in 0..n {
            let (x, y) = (a[i * n + j], a[j * n + i]);
            out[i + j * n] = C { re: 0.5 * (x.re + y.re), im: 0.5 * (x.im - y.im) };
        }
    }
    out
}

/// Euclidean projection of real values onto the probability simplex.
fn simplex(values: &[f64]) -> Vec<f64> {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let (mut sum, mut threshold) = (0., 0.);
    for (index, &value) in sorted.iter().enumerate() {
        sum += value;
        let candidate = (sum - 1.) / (index + 1) as f64;
        if value > candidate {
            threshold = candidate;
        } else {
            break;
        }
    }
    values.iter().map(|x| (x - threshold).max(0.)).collect()
}

struct Projected {
    density: Vec<C>,
    minimum_eigenvalue: f64,
    trace_error: f64,
    hermiticity_error: f64,
}

/// Frobenius-nearest positive trace-one matrix. The true mean belongs to this
/// convex set, so the projection never increases the Frobenius error.
fn project_density(lapack: &Lapack, m: &[C], n: usize) -> Result<Projected, String> {
    let mut hermiticity_error = 0_f64;
    let mut trace = C::default();
    for i in 0..n {
        trace.re += m[i * n + i].re;
        trace.im += m[i * n + i].im;
        for j in 0..n {
            let (x, y) = (m[i * n + j], m[j * n + i]);
            hermiticity_error = hermiticity_error.max((x.re - y.re).hypot(x.im + y.im));
        }
    }
    let (values, vectors) = lapack.hermitian_eigen(&hermitian_column_major(m, n), n)?;
    let minimum_eigenvalue = values.iter().copied().fold(f64::INFINITY, f64::min);
    let weights = simplex(&values);
    let mut density = vec![C::default(); n * n];
    for (k, &p) in weights.iter().enumerate() {
        if p == 0. {
            continue;
        }
        let v = &vectors[k * n..(k + 1) * n];
        for i in 0..n {
            for j in 0..n {
                // p v_i conj(v_j)
                let (a, b) = (v[i], v[j]);
                density[i * n + j].re += p * (a.re * b.re + a.im * b.im);
                density[i * n + j].im += p * (a.im * b.re - a.re * b.im);
            }
        }
    }
    Ok(Projected { density, minimum_eigenvalue, trace_error: (trace.re - 1.).hypot(trace.im), hermiticity_error })
}

fn add_into(target: &mut [C], other: &[C]) {
    for (a, b) in target.iter_mut().zip(other) {
        a.re += b.re;
        a.im += b.im;
    }
}
fn difference(a: &[C], b: &[C]) -> Vec<C> {
    a.iter().zip(b).map(|(x, y)| C { re: x.re - y.re, im: x.im - y.im }).collect()
}

// ---------------------------------------------------------------------------
// Reduced model analysis.

struct ReducedSpectrum {
    /// Magnitudes of all reduced eigenvalues, ascending. The first one is the
    /// structural zero of exact trace conservation.
    magnitudes: Vec<f64>,
}

impl ReducedSpectrum {
    fn of(model: &ReducedModel, lapack: &Lapack) -> Result<Self, String> {
        let (re, im, _) = lapack.eigen(&model.generator, model.m)?;
        let mut magnitudes: Vec<f64> = re.iter().zip(&im).map(|(a, b)| a.hypot(*b)).collect();
        magnitudes.sort_by(f64::total_cmp);
        Ok(Self { magnitudes })
    }
    /// Smallest nonzero rate, i.e. the second smallest magnitude.
    fn slowest_rate(&self) -> Option<f64> {
        self.magnitudes.get(1).copied()
    }
    /// The structural zero plus every eigenvalue below the resolvable rate.
    fn zero_modes(&self, resolvable: f64) -> usize {
        1 + self.magnitudes.iter().skip(1).filter(|&&x| x < resolvable).count()
    }
    fn slowest_resolved_rate(&self, resolvable: f64) -> Option<f64> {
        self.magnitudes.iter().skip(1).copied().find(|&x| x >= resolvable)
    }
}

/// Reduced null vector `G a = 0` with unit trace `t . a = 1`. The rows of `G`
/// are dependent along `t` (exact trace conservation), so the row where `t`
/// is largest is replaced by the normalisation.
fn reduced_null_vector(model: &ReducedModel, trace: &[f64], lapack: &Lapack) -> Result<Vec<f64>, String> {
    let m = model.m;
    let pivot = (0..m).max_by(|&a, &b| trace[a].abs().total_cmp(&trace[b].abs())).ok_or("Empty reduced model")?;
    let mut matrix = model.generator.clone();
    for j in 0..m {
        matrix[pivot + j * m] = trace[j];
    }
    let mut rhs = vec![0.; m];
    rhs[pivot] = 1.;
    let null = lapack.solve(&matrix, &rhs, m)?;
    if null.iter().any(|x| !x.is_finite()) {
        return Err("Nonfinite reduced null vector".into());
    }
    Ok(null)
}

/// `(pole E - F) x = rhs` by restarted, P0-preconditioned GMRES until the
/// TRUE relative residual is at most the requested one or the restarts are
/// exhausted. Returns the solution and its true relative residual.
fn solve_shifted(op: &mut DomainOperator, rhs: &[f64], settings: InnerSolve, restarts: usize, applications: &mut usize)
    -> Result<(Vec<f64>, f64), String> {
    let rows = op.rows();
    let norm = euclid(rhs).max(f64::MIN_POSITIVE);
    let mut x = vec![0.; rows];
    let mut residual = rhs.to_vec();
    let mut applied = vec![0.; rows];
    let mut relative = 1.;
    for attempt in 0..=restarts {
        let (dx, used) = shift_invert(op, &residual, 1, &[0], settings)?;
        *applications += used;
        for (a, b) in x.iter_mut().zip(dx) {
            *a += b;
        }
        op.shifted(&x, 1, &[0], &mut applied)?;
        *applications += 1;
        for ((r, b), a) in residual.iter_mut().zip(rhs).zip(&applied) {
            *r = b - a;
        }
        let next = euclid(&residual) / norm;
        // Stop when converged or when a restart no longer helps.
        let stalled = attempt > 0 && next > 0.5 * relative;
        relative = next;
        if relative <= settings.relative_residual || stalled {
            break;
        }
    }
    Ok((x, relative))
}

/// Correction `delta` with `(pole E - F) delta = rhs`, with its component
/// along the null vector removed (that component is the amplified trace
/// error of the inexact solve and carries no information).
fn correction(op: &mut DomainOperator, rhs: &[f64], null: &[f64], n: usize, applications: &mut usize)
    -> Result<(Vec<f64>, f64), String> {
    let (mut delta, relative) = solve_shifted(op, rhs, CORRECTION_SOLVE, CORRECTION_RESTARTS, applications)?;
    let trace = central_trace(&delta, n);
    for (d, w) in delta.iter_mut().zip(null) {
        *d -= trace * w;
    }
    Ok((delta, relative))
}

// ---------------------------------------------------------------------------
// Fourier truncation (halo).

/// Modes one coupling step outside the domain.
fn shell_modes(modes: &[Charge], steps: &[Charge]) -> Vec<Charge> {
    let inside: BTreeSet<&Charge> = modes.iter().collect();
    let mut shell = BTreeSet::new();
    for q in modes {
        for s in steps {
            for candidate in [charge_plus(q, s), charge_minus(q, s)] {
                if !inside.contains(&candidate) {
                    shell.insert(candidate);
                }
            }
        }
    }
    shell.into_iter().collect()
}

struct Halo {
    /// Density correction (laboratory frame) from the condensed halo.
    density: Vec<C>,
    /// The part of `density` produced by the transport-condensed modes.
    far_density: Option<Vec<C>>,
    reliable: bool,
    /// Frobenius weight of the condensed field on every halo mode.
    indicators: Vec<(Charge, f64)>,
    /// Modes condensed exactly and by transport.
    counts: (usize, usize),
}

/// Condensed halo: `x_S = (s E - F)_SS^-1 F_SM w` approximated mode-diagonally
/// (approximate shell solution), its feedback `F_MS x_S` corrected inside the
/// domain, and the direct readout of `x_S`. Returns the laboratory density
/// correction.
///
/// The mode-diagonal inverse is the exact P0 on the modes that carry the halo
/// and the transport inverse `1 / (s + i omega_q)` on the far modes whose
/// transport amplitude is below `FAR_HALO_AMPLITUDE` of the largest (a P0
/// shift factor costs a dense factorisation per mode, the transport inverse
/// nothing). The far contribution is computed separately so that it can be
/// charged in full to the estimate.
#[allow(clippy::too_many_arguments)]
fn halo(core: &mut RuntimeCore, op: &mut DomainOperator, frame: &Frame, readout: &mut TargetReadout,
    shell: &[Charge], null: &[f64], pole: f64, applications: &mut usize) -> Result<Halo, String> {
    let n = frame.n;
    let d = n * n;
    let modes = op.modes().to_vec();
    let signed = op.signed_block(null, 1)?;
    // Residual of the domain null vector on the shell.
    let mut outward = core.prepared().pencil(&modes, shell, 1)?;
    let mut e_shell = vec![C::default(); shell.len() * d];
    let mut f_shell = vec![C::default(); shell.len() * d];
    outward.apply_into(&signed, 1, &mut e_shell, &mut f_shell)?;
    let shifts: Vec<C> = {
        let phases = &core.prepared().primitives().hilbert.source.phases;
        shell.iter().map(|q| phases.frequency(q).map(|f| C { re: pole, im: f.hi + f.lo })).collect::<Result<_, _>>()?
    };
    let transport: Vec<f64> = (0..shell.len()).map(|k| euclid_complex(&f_shell[k * d..(k + 1) * d]) / shifts[k].re.hypot(shifts[k].im))
        .collect();
    let largest = transport.iter().copied().fold(0_f64, f64::max);
    let (near, far): (Vec<usize>, Vec<usize>) = (0..shell.len()).partition(|&k| transport[k] > FAR_HALO_AMPLITUDE * largest);
    let mut condensed = vec![C::default(); shell.len() * d];
    if !near.is_empty() {
        let mut gathered = Vec::with_capacity(near.len() * d);
        for &k in &near {
            gathered.extend_from_slice(&f_shell[k * d..(k + 1) * d]);
        }
        let near_shifts: Vec<C> = near.iter().map(|&k| shifts[k]).collect();
        let mut p0 = core.p0_mut().prepare_batch(&near_shifts, 1)?;
        let mut solved = vec![C::default(); gathered.len()];
        p0.apply_into(&gathered, 1, &mut solved)?;
        for (slot, &k) in near.iter().enumerate() {
            condensed[k * d..(k + 1) * d].copy_from_slice(&solved[slot * d..(slot + 1) * d]);
        }
    }
    for &k in &far {
        let inverse = reciprocal(shifts[k]);
        for (x, h) in condensed[k * d..(k + 1) * d].iter_mut().zip(&f_shell[k * d..(k + 1) * d]) {
            *x = complex_product(*h, inverse);
        }
    }
    let indicators: Vec<(Charge, f64)> = shell.iter().enumerate()
        .map(|(k, q)| (q.clone(), euclid_complex(&condensed[k * d..(k + 1) * d])))
        .collect();
    let mut inward = core.prepared().pencil(shell, &modes, 1)?;
    let mut direct = TargetReadout::new(core, shell, frame)?;
    let mut response = |core: &RuntimeCore, field: &[C], applications: &mut usize| -> Result<(Vec<C>, f64), String> {
        // Feedback of the condensed halo on the domain, corrected inside it.
        let mut e_back = vec![C::default(); modes.len() * d];
        let mut f_back = vec![C::default(); modes.len() * d];
        inward.apply_into(field, 1, &mut e_back, &mut f_back)?;
        let mut pulled = vec![C::default(); op.rows()];
        op.map().pull_complex_into(&f_back, 1, &mut pulled)?;
        let feedback: Vec<f64> = pulled.iter().map(|z| z.re).collect();
        let (inner, relative) = correction(op, &feedback, null, n, applications)?;
        let inner_signed = op.signed_block(&inner, 1)?;
        let mut density = readout.density(core.physical(), frame, &inner_signed)?;
        // Direct readout of the halo field itself.
        add_into(&mut density, &direct.density(core.physical(), frame, field)?);
        Ok((density, relative))
    };
    let (density, relative) = response(core, &condensed, applications)?;
    let (far_density, far_relative) = if far.is_empty() {
        (None, 0.)
    } else {
        let mut far_field = vec![C::default(); condensed.len()];
        for &k in &far {
            far_field[k * d..(k + 1) * d].copy_from_slice(&condensed[k * d..(k + 1) * d]);
        }
        let (value, relative) = response(core, &far_field, applications)?;
        (Some(value), relative)
    };
    Ok(Halo { density, far_density, reliable: relative.max(far_relative) <= ESTIMATE_RESIDUAL_LIMIT, indicators,
        counts: (near.len(), far.len()) })
}

/// Halo modes with the largest indicators covering `HALO_COVERAGE` of the
/// total squared weight, with their Hermitian partners.
fn select_halo_modes(indicators: &[(Charge, f64)]) -> Vec<Charge> {
    let total: f64 = indicators.iter().map(|(_, w)| w * w).sum();
    let mut sorted: Vec<&(Charge, f64)> = indicators.iter().collect();
    sorted.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let mut chosen = BTreeSet::new();
    let mut covered = 0.;
    for (q, w) in sorted {
        if covered >= HALO_COVERAGE * total && !chosen.is_empty() {
            break;
        }
        if chosen.insert(q.clone()) {
            covered += w * w;
        }
        let partner: Charge = q.iter().map(|x| -x).collect();
        if chosen.insert(partner) {
            covered += w * w;
        }
    }
    chosen.into_iter().collect()
}

// ---------------------------------------------------------------------------
// Driver.

struct Context {
    n: usize,
    axes: usize,
    /// Every Fourier step (the halo covers all of them).
    steps: Vec<Charge>,
    /// Steps with a significant mixing amplitude: the starting domain and the
    /// domain growth of the degenerate path.
    domain_steps: Vec<Charge>,
    frame: Frame,
    lapack: Arc<Lapack>,
    /// Relaxation spectrum of the slow generator (None when it is
    /// unavailable: then the reduced spectrum and the probe alone decide).
    slow: Option<SlowSpectrum>,
    /// The near-zero pole of this attempt.
    pole: f64,
}

impl Context {
    fn new(core: &RuntimeCore, options: &MeanOptions, slow: Option<SlowSpectrum>) -> Result<Self, String> {
        let source = &core.prepared().primitives().hilbert.source;
        let steps = core.fourier_steps()?;
        let pole = slow.and_then(|s| s.slowest_rate).map_or(POLE_MAX, |rate| (POLE_RESOLUTION * rate).clamp(POLE_FLOOR, POLE_MAX));
        Ok(Self { n: source.n, axes: steps.axes, steps: steps.all, domain_steps: steps.domain,
            frame: Frame::new(&source.gauges, &options.lab)?, lapack: Arc::clone(core.lapack()), slow, pole })
    }
    /// Rates the current pole separates from zero.
    fn resolvable_rate(&self) -> f64 {
        RESOLVABLE_POLES * self.pole
    }
    /// The slow generator found only the trace conserved.
    fn slow_unique(&self) -> bool {
        self.slow.is_some_and(|s| s.conserved == 1)
    }
    fn metric(&self, delta: &[C], groups: &[RequestedGroup]) -> Result<f64, String> {
        error_metric(&self.lapack, delta, self.n, groups)
    }
}

/// Bookkeeping shared by both paths.
struct Work {
    started: Instant,
    applications: usize,
    timing: [f64; 5],
    checkpoints: Vec<MeanCheckpoint>,
    probe_stalled: bool,
}

impl Work {
    fn within_budget(&self, options: &MeanOptions) -> bool {
        self.started.elapsed().as_secs_f64() <= options.wall_seconds
    }
}

/// A converged laboratory mean before the density projection.
struct Candidate {
    /// Uncorrected (null vector or resolvent) laboratory mean.
    raw_mean: Vec<C>,
    /// Defect-corrected laboratory mean.
    corrected: Vec<C>,
    discarded_optical_coherence: f64,
    basis_estimate: f64,
    halo_estimate: f64,
    relative_residual: f64,
    slowest_rate: Option<f64>,
    zero_modes: usize,
    modes: Vec<Charge>,
    pole: f64,
    columns: usize,
}

/// Domain, trial space and readout of the one-dimensional null-space path,
/// kept for the degeneracy probe.
struct NullState {
    space: TrialSpace,
    readout: TargetReadout,
}

enum Outcome {
    Null(Candidate, NullState),
    /// The reduced spectrum shows more than one eigenvalue below the
    /// resolvable rate.
    Degenerate { modes: Vec<Charge>, zero_modes: usize, slowest_rate: Option<f64> },
}

/// The long-time mean, preparing the request's slow dual itself (the mean
/// proceeds without it if that fails).
pub fn solve(core: &mut RuntimeCore, options: &MeanOptions) -> Result<MeanSolution, String> {
    let started = Instant::now();
    let dual = SlowDual::prepare(core.prepared(), SlowDualOptions::default()).ok();
    let mut solution = solve_with(core, options, dual.as_ref())?;
    solution.diagnostics.timing[4] += ms(started) - solution.diagnostics.elapsed_ms;
    solution.diagnostics.elapsed_ms = ms(started);
    Ok(solution)
}

/// `solve` with the request's slow dual (None: unavailable).
///
/// Classification by the slow generator: more than one conserved mode of
/// `J_eff` is a genuinely conserved quantity, and the initial-dependent
/// projection is computed directly. Otherwise the limit is unique however
/// slow its relaxation: the pole is set below the slowest rate (down to
/// `POLE_FLOOR`), the trial space starts with the slow generator's conserved
/// and unresolved slow modes, the reduced spectrum's near-zero eigenvalues
/// are relaxations rather than conserved quantities, and the bias of the
/// estimate uses the slow generator's slowest rate. Without a slow generator
/// the reduced spectrum and the degeneracy probe decide, as before.
pub fn solve_with(core: &mut RuntimeCore, options: &MeanOptions, slow: Option<&SlowDual>) -> Result<MeanSolution, String> {
    match options.horizon_us {
        Some(horizon) => horizon_mean(core, options, slow, horizon),
        None => {
            let limit = limit_mean(core, options, slow);
            // A stalled degeneracy probe means the near-zero pole is beyond
            // what the inner solves resolve; the null vector's corrections
            // used the same pole (measured 6x beyond their estimate on the
            // probe + tweezer scene). The limit then comes from propagation.
            let reliable = limit.as_ref().is_ok_and(|solution| !solution.diagnostics.probe_stalled);
            match slow {
                Some(dual) if !reliable => propagated_limit(core, options, dual, limit),
                _ => limit,
            }
        }
    }
}

/// Points of the propagation grid of a horizon mean (its estimate is the
/// largest over them, which bounds the error of the averaged state at `T`).
const HORIZON_SAMPLES: usize = 65;
/// Share of the tolerance the propagation may use (the rest covers the
/// projection of the averaged density onto the density cone).
const HORIZON_PROPAGATION_SHARE: f64 = 0.9;
/// The slow part is negligible at the horizon when its bound is below this
/// fraction of the tolerance.
const HORIZON_NEGLIGIBLE: f64 = 0.05;
/// The bound takes every decay rate of the slow generator at this fraction
/// (its rates are a reduction: measured 4-12% from the exact ones on a toy).
const HORIZON_RATE_MARGIN: f64 = 0.5;

/// Central ground block (g x g) of the initial frame density `E y0`.
fn initial_ground(core: &RuntimeCore, dual: &SlowDual) -> Result<Dense, String> {
    let (n, g) = (dual.n, dual.g);
    let central = vec![0; dual.slots[0].len()];
    let mut pencil = core.prepared().pencil(std::slice::from_ref(&central), std::slice::from_ref(&central), 1)?;
    let (mut e, mut f) = (vec![C::default(); n * n], vec![C::default(); n * n]);
    pencil.apply_into(core.initial().point(), 1, &mut e, &mut f)?;
    let mut ground = Dense::zeros(g * g, 1);
    for a in 0..g {
        for b in 0..g {
            ground.set(a * g + b, 0, e[a * n + b]);
        }
    }
    Ok(ground)
}

/// Largest state or group population of a per-state bound.
fn bound_size(values: &[f64], groups: &[RequestedGroup]) -> f64 {
    let state = values.iter().copied().fold(0_f64, f64::max);
    let group = groups.iter().map(|group| group.indices.iter().filter(|&&i| i < values.len()).map(|&i| values[i]).sum::<f64>()).fold(0_f64, f64::max);
    state.max(group)
}

/// The mean at the observation horizon `T` (module docs): the limit when the
/// slow generator bounds what is still relaxing at `T` below
/// `HORIZON_NEGLIGIBLE * epsilon` (that bound is charged), otherwise the
/// state at `T` by the finite-time engine (error-controlled over the whole
/// window), read with the laboratory time average.
fn horizon_mean(core: &mut RuntimeCore, options: &MeanOptions, slow: Option<&SlowDual>, horizon: f64) -> Result<MeanSolution, String> {
    let epsilon = options.population_tolerance;
    if !(horizon.is_finite() && horizon > 0.) {
        return Err("The observation horizon must be a finite positive time".into());
    }
    let dual = slow.ok_or("A mean at a finite horizon needs the slow generator, which could not be prepared for this model; the tolerance was not relaxed")?;
    let ground = initial_ground(core, dual)?;
    let remaining = bound_size(&dual.horizon_bound(&ground, horizon, HORIZON_RATE_MARGIN)?, &options.groups);
    if remaining <= HORIZON_NEGLIGIBLE * epsilon {
        let limit = solve_with(core, &MeanOptions { horizon_us: None, ..options.clone() }, slow)?;
        let mut solution = limit;
        let estimate = solution.diagnostics.estimated_error + remaining;
        if estimate > epsilon {
            return Err(format!("The mean at the {horizon:e} us horizon is estimated at {estimate:e} (limit {:e}, still relaxing at most {remaining:e}); the tolerance was not relaxed", solution.diagnostics.estimated_error));
        }
        solution.diagnostics.estimated_error = estimate;
        let method = if solution.diagnostics.horizon.is_some() { "propagated-limit" } else { "limit" };
        solution.diagnostics.horizon = Some(HorizonPart { horizon_us: horizon, method, decaying: remaining, propagation_estimate: 0., projection_charge: 0. });
        return Ok(solution);
    }
    let kind = MeanKind::InitialDependent { zero_modes: dual.slow_spectrum().conserved.max(1) };
    let mut solution = propagate(core, options, dual, horizon, kind)?;
    solution.diagnostics.horizon.as_mut().expect("propagation part").decaying = remaining;
    Ok(solution)
}

/// The limit by propagation: to the first doubling of `10 / rate` (slowest
/// relaxation of the slow generator) at which the bound on what still relaxes
/// is below `HORIZON_NEGLIGIBLE * epsilon`; that bound is charged. `limit` is
/// the pole-based attempt (its diagnostics are kept when it converged).
fn propagated_limit(core: &mut RuntimeCore, options: &MeanOptions, dual: &SlowDual, limit: Result<MeanSolution, String>) -> Result<MeanSolution, String> {
    let epsilon = options.population_tolerance;
    let spectrum = dual.slow_spectrum();
    let ground = initial_ground(core, dual)?;
    let mut horizon = 10. / spectrum.slowest_rate.unwrap_or(1.).max(f64::MIN_POSITIVE);
    let mut remaining = bound_size(&dual.horizon_bound(&ground, horizon, HORIZON_RATE_MARGIN)?, &options.groups);
    for _ in 0..60 {
        if remaining <= HORIZON_NEGLIGIBLE * epsilon {
            break;
        }
        horizon *= 2.;
        remaining = bound_size(&dual.horizon_bound(&ground, horizon, HORIZON_RATE_MARGIN)?, &options.groups);
    }
    let reason = |error: String| match &limit {
        Ok(_) => error,
        Err(first) => format!("{first}; the limit by propagation failed too: {error}"),
    };
    if remaining > HORIZON_NEGLIGIBLE * epsilon {
        return Err(reason(format!("the slow generator still bounds {remaining:e} relaxing at {horizon:e} us")));
    }
    let kind = if spectrum.conserved > 1 { MeanKind::InitialDependent { zero_modes: spectrum.conserved } } else { MeanKind::Unique };
    let mut solution = propagate(core, options, dual, horizon, kind).map_err(reason)?;
    let estimate = solution.diagnostics.estimated_error + remaining;
    if estimate > epsilon {
        return Err(reason(format!("estimate {estimate:e} with {remaining:e} still relaxing at {horizon:e} us")));
    }
    solution.diagnostics.estimated_error = estimate;
    let part = solution.diagnostics.horizon.as_mut().expect("propagation part");
    part.method = "propagated-limit";
    part.decaying = remaining;
    match limit {
        Ok(pole) => {
            solution.diagnostics.relative_residual = pole.diagnostics.relative_residual;
            solution.diagnostics.probe_stalled = pole.diagnostics.probe_stalled;
            solution.diagnostics.limit_note = Some(format!(
                "the near-zero pole was not resolved (its degeneracy probe stalled); its mean differed by {:e}",
                context_metric(core, options, &pole.density, &solution.density)?
            ));
        }
        Err(first) => solution.diagnostics.limit_note = Some(first),
    }
    Ok(solution)
}

fn context_metric(core: &RuntimeCore, options: &MeanOptions, a: &[C], b: &[C]) -> Result<f64, String> {
    error_metric(core.lapack(), &difference(a, b), core.prepared().primitives().hilbert.source.n, &options.groups)
}

/// The laboratory-averaged state at `horizon` by the finite-time engine on a
/// fork of the core (error-controlled over the whole window), projected onto
/// the density cone (charged).
fn propagate(core: &mut RuntimeCore, options: &MeanOptions, dual: &SlowDual, horizon: f64, kind: MeanKind) -> Result<MeanSolution, String> {
    let epsilon = options.population_tolerance;
    let n = dual.n;
    let times: Vec<f64> = (0..HORIZON_SAMPLES).map(|i| horizon * i as f64 / (HORIZON_SAMPLES - 1) as f64).collect();
    let finite = FiniteOptions { population_tolerance: HORIZON_PROPAGATION_SHARE * epsilon, keep_end_state: true, ..FiniteOptions::default() };
    let propagated = engine::solve(core.fork()?, &times, options.groups.clone(), finite)?;
    let (modes, signed) = propagated.end_state.as_ref().ok_or("The propagation to the horizon returned no end state")?;
    let spectrum = dual.slow_spectrum();
    let context = Context::new(core, options, Some(spectrum))?;
    let mut readout = TargetReadout::new(core, modes, &context.frame)?;
    let averaged = readout.density(core.physical(), &context.frame, signed)?;
    let discarded = readout.discarded(core.physical(), &context.frame)?;
    let projected = project_density(&context.lapack, &averaged, n)?;
    let projection_charge = context.metric(&difference(&projected.density, &averaged), &options.groups)?;
    let propagation_estimate = propagated.diagnostics.estimated_error;
    let estimate = propagation_estimate + projection_charge;
    if estimate > epsilon {
        return Err(format!("The propagation to {horizon:e} us is estimated at {estimate:e} (window {propagation_estimate:e}, density projection {projection_charge:e}); the tolerance was not relaxed"));
    }
    let populations = (0..n).map(|s| projected.density[s * n + s].re).collect();
    let checkpoints = propagated
        .diagnostics
        .checkpoints
        .iter()
        .map(|c| MeanCheckpoint {
            columns: c.columns,
            modes: c.modes,
            pole: f64::NAN,
            basis_estimate: c.basis_estimate.unwrap_or(f64::NAN),
            halo_estimate: c.halo_estimate,
            halo_modes: (0, 0),
            inner_residual: f64::NAN,
            action: c.action,
        })
        .collect();
    Ok(MeanSolution {
        density: projected.density,
        discarded_optical_coherence: discarded,
        populations,
        kind,
        diagnostics: MeanDiagnostics {
            estimated_error: estimate,
            basis_estimate: propagated.diagnostics.raw_estimate,
            halo_estimate: 0.,
            projection_charge,
            minimum_eigenvalue_before_projection: projected.minimum_eigenvalue,
            trace_error_before_projection: projected.trace_error,
            hermiticity_error: projected.hermiticity_error,
            // The window has no null-vector residual; a converged pole-based
            // limit replaces it (`propagated_limit`).
            relative_residual: 0.,
            slowest_rate: spectrum.slowest_rate,
            slowest_relaxation_us: spectrum.slowest_rate.map(|rate| 1. / rate),
            conserved_quantities: Some(spectrum.conserved),
            zero_modes: spectrum.conserved,
            pole: f64::NAN,
            columns: propagated.diagnostics.final_columns,
            modes: propagated.diagnostics.final_modes,
            operator_applications: propagated.diagnostics.operator_applications,
            elapsed_ms: propagated.diagnostics.elapsed_ms,
            timing: [0.; 5],
            uncorrected_populations: (0..n).map(|s| averaged[s * n + s].re).collect(),
            checkpoints,
            horizon: Some(HorizonPart { horizon_us: horizon, method: "propagation", decaying: 0., propagation_estimate, projection_charge }),
            probe_stalled: false,
            limit_note: None,
        },
    })
}

/// The t -> infinity mean (`solve_with` without a horizon).
fn limit_mean(core: &mut RuntimeCore, options: &MeanOptions, slow: Option<&SlowDual>) -> Result<MeanSolution, String> {
    let epsilon = options.population_tolerance;
    if !(epsilon > 0. && epsilon <= 0.02) {
        return Err("populationTolerance must be in (0, 0.02]".into());
    }
    let started = Instant::now();
    let slow_dual = slow;
    let slow = slow.map(|dual| dual.slow_spectrum());
    let context = Context::new(core, options, slow)?;
    let mut work = Work { started, applications: 0, timing: [0.; 5], checkpoints: Vec::new(), probe_stalled: false };
    let unresolved = slow.and_then(|s| s.slowest_rate).filter(|&rate| rate < context.resolvable_rate());
    if let Some(conserved) = slow.map(|s| s.conserved).filter(|&c| c > 1) {
        // The resolvent projection keeps a fraction pole / rate of every
        // decaying mode; it is used only when no relaxation is that slow.
        if let Some(rate) = unresolved {
            return Err(format!("{conserved} conserved quantities together with a relaxation of {:e} us cannot be resolved by the near-zero pole; the tolerance was not relaxed", 1. / rate));
        }
        let modes = support_domain(context.axes, &context.domain_steps, 1)?;
        return degenerate(core, options, &context, modes, conserved, None, &mut work);
    }
    match null_vector_mean(core, options, &context, slow_dual, &mut work)? {
        Outcome::Null(candidate, mut state) => {
            // With the slow generator unique and a relaxation the pole cannot
            // resolve, the probe's resolvent would keep that slow mode and
            // read it as a degeneracy: the slow generator decides.
            if context.slow_unique() && unresolved.is_some() {
                return finish(&context, options, candidate, MeanKind::Unique, &mut work);
            }
            let clock = Instant::now();
            // A stalled probe is a numerical failure, not evidence of a second
            // conserved quantity: the slow generator, which found only the
            // trace conserved, decides then.
            let probe = match degeneracy_probe(core, options, &context, &candidate, &mut state, &mut work) {
                Err(reason) if context.slow_unique() && reason.contains("stalled") => {
                    work.probe_stalled = true;
                    Ok(Probe::Unique)
                }
                other => other,
            }?;
            work.timing[3] += ms(clock);
            match probe {
                Probe::Unique => finish(&context, options, candidate, MeanKind::Unique, &mut work),
                Probe::StaticDegenerate => {
                    finish(&context, options, Candidate { zero_modes: 2, ..candidate }, MeanKind::InitialDependent { zero_modes: 2 }, &mut work)
                }
                Probe::Degenerate => {
                    let (modes, rate) = (candidate.modes.clone(), candidate.slowest_rate);
                    degenerate(core, options, &context, modes, 2, rate, &mut work)
                }
            }
        }
        Outcome::Degenerate { modes, zero_modes, slowest_rate } => {
            degenerate(core, options, &context, modes, zero_modes, slowest_rate, &mut work)
        }
    }
}

fn finish(context: &Context, options: &MeanOptions, candidate: Candidate, kind: MeanKind, work: &mut Work) -> Result<MeanSolution, String> {
    let projected = project_density(&context.lapack, &candidate.corrected, context.n)?;
    let charge = context.metric(&difference(&projected.density, &candidate.corrected), &options.groups)?;
    let estimate = SAFETY * (candidate.basis_estimate + candidate.halo_estimate) + charge;
    if estimate > options.population_tolerance {
        return Err(format!("Long-time mean estimate {estimate:e} exceeds the tolerance after the density projection; the tolerance was not relaxed"));
    }
    let n = context.n;
    let populations = (0..n).map(|s| projected.density[s * n + s].re).collect();
    Ok(MeanSolution {
        density: projected.density,
        discarded_optical_coherence: candidate.discarded_optical_coherence,
        populations,
        kind,
        diagnostics: MeanDiagnostics {
            estimated_error: estimate,
            basis_estimate: candidate.basis_estimate,
            halo_estimate: candidate.halo_estimate,
            projection_charge: charge,
            minimum_eigenvalue_before_projection: projected.minimum_eigenvalue,
            trace_error_before_projection: projected.trace_error,
            hermiticity_error: projected.hermiticity_error,
            relative_residual: candidate.relative_residual,
            slowest_rate: candidate.slowest_rate,
            slowest_relaxation_us: context.slow.map_or(candidate.slowest_rate, |s| s.slowest_rate).map(|rate| 1. / rate),
            conserved_quantities: context.slow.map(|s| s.conserved),
            zero_modes: candidate.zero_modes,
            pole: candidate.pole,
            columns: candidate.columns,
            modes: candidate.modes.len(),
            operator_applications: work.applications,
            elapsed_ms: ms(work.started),
            timing: work.timing,
            uncorrected_populations: (0..n).map(|s| candidate.raw_mean[s * n + s].re).collect(),
            checkpoints: std::mem::take(&mut work.checkpoints),
            horizon: None,
            probe_stalled: work.probe_stalled,
            limit_note: None,
        },
    })
}

/// Defect-corrected null vector of the pencil with basis and halo control.
fn null_vector_mean(core: &mut RuntimeCore, options: &MeanOptions, context: &Context, slow_dual: Option<&SlowDual>, work: &mut Work)
    -> Result<Outcome, String> {
    let epsilon = options.population_tolerance;
    let mut modes = support_domain(context.axes, &context.domain_steps, 1)?;
    let pole = context.pole;
    let resolvable = context.resolvable_rate();
    let clock = Instant::now();
    let mut space = TrialSpace::new(core, &modes, &[pole])?;
    let mut readout = TargetReadout::new(core, &modes, &context.frame)?;
    // Only a relaxation the pole cannot resolve needs the slow modes in the
    // space from the start (the corrections resolve every other one).
    let unresolved = context.slow.and_then(|s| s.slowest_rate).is_some_and(|rate| rate < resolvable);
    if let Some(dual) = slow_dual.filter(|_| context.slow_unique() && unresolved) {
        seed_slow_modes(&mut space, dual, resolvable)?;
    }
    work.timing[0] += ms(clock);
    // Set when the correction is already contained in the trial space: the
    // space is invariant and its null vector exact on this domain.
    let mut invariant = false;
    // Previous raw mean and its predicted correction, for self-calibration.
    let mut previous: Option<(Vec<C>, f64)> = None;
    let mut calibration = 1_f64;
    let mut calibrated = false;
    loop {
        if !work.within_budget(options) {
            return Err(format!("Long-time mean work limit reached with {} columns on {} Fourier modes; the tolerance was not relaxed",
                space.columns(), modes.len()));
        }
        let model = ReducedModel::build(&space, 0.)?;
        let spectrum = ReducedSpectrum::of(&model, &context.lapack)?;
        // With the slow generator unique, near-zero reduced eigenvalues are
        // slow relaxations (its conserved count is the zero-mode count).
        let zero_modes = if context.slow_unique() { 1 } else { spectrum.zero_modes(resolvable) };
        if zero_modes > 1 {
            return Ok(Outcome::Degenerate { modes, zero_modes, slowest_rate: spectrum.slowest_resolved_rate(resolvable) });
        }
        let rows = space.rows();
        let null_reduced = reduced_null_vector(&model, space.traces().0, &context.lapack)?;
        let null = lift(&context.lapack, space.q(), rows, &null_reduced)?;
        let residual = lift(&context.lapack, space.fq(), rows, &null_reduced)?;
        let signed = space.op.signed_block(&null, 1)?;
        let raw_mean = readout.density(core.physical(), &context.frame, &signed)?;
        let discarded_optical_coherence = readout.discarded(core.physical(), &context.frame)?;
        // The correction appended in the previous round predicted this change.
        if let Some((old_mean, predicted)) = previous.take() {
            let observed = context.metric(&difference(&raw_mean, &old_mean), &options.groups)?;
            if predicted > 0. && observed.is_finite() {
                calibration = calibration.max(observed / predicted);
                calibrated = true;
            }
        }
        // Newton correction of the null vector and its laboratory readout.
        let clock = Instant::now();
        let (delta, inner) = correction(&mut space.op, &residual, &null, context.n, &mut work.applications)?;
        work.timing[1] += ms(clock);
        let delta_signed = space.op.signed_block(&delta, 1)?;
        let basis_density = readout.density(core.physical(), &context.frame, &delta_signed)?;
        // A slow rate gamma is resolved by the pole with relative bias s/gamma:
        // the slow generator's slowest rate (and a slower resolved reduced
        // one), or without it the reduced spectrum's above the resolvable rate.
        let slowest = if context.slow_unique() {
            let rate = context.slow.and_then(|s| s.slowest_rate).unwrap_or(f64::INFINITY);
            spectrum.slowest_rate().filter(|&r| r >= resolvable).map_or(rate, |r| r.min(rate))
        } else {
            spectrum.slowest_rate().unwrap_or(resolvable).max(resolvable)
        };
        let bias = calibration * (1. + pole / slowest);
        let predicted = context.metric(&basis_density, &options.groups)?;
        let basis_estimate = predicted * bias;
        // An estimate is used for acceptance only after one estimate has been
        // checked against the change it predicted (or the space is invariant).
        let basis_reliable = inner <= ESTIMATE_RESIDUAL_LIMIT && (calibrated || invariant);
        let mut checkpoint = MeanCheckpoint { columns: space.columns(), modes: modes.len(), pole, basis_estimate,
            halo_estimate: None, halo_modes: (0, 0), inner_residual: inner, action: "" };
        if basis_reliable && SAFETY * basis_estimate <= 0.5 * epsilon {
            let shell = shell_modes(&modes, &context.steps);
            let halo = if shell.is_empty() {
                None
            } else {
                let clock = Instant::now();
                let value = halo(core, &mut space.op, &context.frame, &mut readout, &shell, &null, pole, &mut work.applications)?;
                work.timing[2] += ms(clock);
                // The correction itself, plus the transport part once more: the
                // transport inverse may be off by its own size near a resonance.
                let far = value.far_density.as_ref().map_or(Ok(0.), |far| context.metric(far, &options.groups))?;
                let estimate = (context.metric(&value.density, &options.groups)? + far) * bias;
                checkpoint.halo_modes = value.counts;
                Some((value, estimate))
            };
            let halo_estimate = halo.as_ref().map_or(0., |h| h.1);
            checkpoint.halo_estimate = halo.as_ref().map(|h| h.1);
            let mut corrected = raw_mean.clone();
            add_into(&mut corrected, &basis_density);
            if let Some((h, _)) = &halo {
                add_into(&mut corrected, &h.density);
            }
            let projected = project_density(&context.lapack, &corrected, context.n)?;
            let charge = context.metric(&difference(&projected.density, &corrected), &options.groups)?;
            let estimate = SAFETY * (basis_estimate + halo_estimate) + charge;
            if estimate <= epsilon && halo.as_ref().map_or(true, |h| h.0.reliable) {
                checkpoint.action = "accept";
                work.checkpoints.push(checkpoint);
                // The residual of the returned vector: the corrected null vector.
                let returned: Vec<f64> = null.iter().zip(&delta).map(|(w, d)| w + d).collect();
                let relative_residual = relative_null_residual(&mut space.op, &returned, &mut work.applications)?;
                let candidate = Candidate { raw_mean, corrected, discarded_optical_coherence, basis_estimate, halo_estimate,
                    relative_residual, slowest_rate: spectrum.slowest_rate(), zero_modes, modes, pole, columns: space.columns() };
                return Ok(Outcome::Null(candidate, NullState { space, readout }));
            }
            if let Some((h, _)) = halo.filter(|h| h.1 >= basis_estimate) {
                if modes.len() >= options.maximum_modes {
                    return Err(format!("Long-time mean needs more than {} Fourier modes; the tolerance was not relaxed", options.maximum_modes));
                }
                let mut grown: BTreeSet<Charge> = modes.iter().cloned().collect();
                grown.extend(select_halo_modes(&h.indicators));
                modes = grown.into_iter().collect();
                checkpoint.action = "expand-domain";
                work.checkpoints.push(checkpoint);
                let clock = Instant::now();
                space = space.transfer(core, &modes)?;
                readout = TargetReadout::new(core, &modes, &context.frame)?;
                work.timing[0] += ms(clock);
                invariant = false;
                previous = None;
                continue;
            }
        }
        if space.columns() >= options.maximum_columns {
            return Err(format!("Long-time mean trial space limit of {} columns reached; the tolerance was not relaxed",
                options.maximum_columns));
        }
        checkpoint.action = "refine-basis";
        work.checkpoints.push(checkpoint);
        previous = Some((raw_mean, predicted));
        if space.append(delta, 1)?.is_empty() {
            if invariant {
                return Err("The long-time mean trial space cannot be extended further".into());
            }
            invariant = true;
        }
    }
}

/// Appends the slow generator's conserved modes and its modes relaxing
/// slower than `rate` to the trial space (real and imaginary parts of each
/// lifted frame field on the domain's slot charges). The Galerkin null vector
/// then carries the slow manifold however slow its relaxation, which no pole
/// above `rate / 100` resolves in a correction. Returns the columns added.
fn seed_slow_modes(space: &mut TrialSpace, dual: &SlowDual, rate: f64) -> Result<usize, String> {
    let modes = space.op.modes().to_vec();
    let n = space.op.map().physical_dimension();
    let d = n * n;
    let rows = space.rows();
    let mut added = 0;
    for blocks in dual.slow_modes(rate)? {
        let mut signed = vec![C::default(); modes.len() * d];
        for (slot, x) in &blocks {
            let Some(k) = modes.iter().position(|q| q == &dual.slots[*slot]) else { continue };
            for a in 0..n {
                for b in 0..n {
                    signed[k * d + a * n + b] = x.at(a, b);
                }
            }
        }
        let mut pulled = vec![C::default(); rows];
        space.op.map().pull_complex_into(&signed, 1, &mut pulled)?;
        for part in [pulled.iter().map(|z| z.re).collect::<Vec<f64>>(), pulled.iter().map(|z| z.im).collect()] {
            let size = euclid(&part);
            let scale = euclid_complex(&pulled);
            if size > 1e-8 * scale && size.is_finite() {
                added += space.append(part, 1)?.len();
            }
        }
    }
    Ok(added)
}

// ---------------------------------------------------------------------------
// Degenerate null space.

enum Probe {
    Unique,
    /// Static model whose null space is degenerate: the null vector of the
    /// initial trajectory is already Pi0 x0 (there are no Fourier families),
    /// only the classification changes.
    StaticDegenerate,
    /// Driven model: Pi0 X0 differs from the null vector of the initial
    /// trajectory (which is the mean averaged over initial laser phases).
    Degenerate,
}

/// The second probe: `X0` (initial point on every mode) for driven models,
/// the maximally mixed central state for static ones.
fn probe_matrix(core: &RuntimeCore, context: &Context) -> (Vec<C>, bool) {
    let n = context.n;
    if context.axes == 0 {
        let mut mixed = vec![C::default(); n * n];
        for i in 0..n {
            mixed[i * n + i].re = 1. / n as f64;
        }
        (mixed, true)
    } else {
        (core.initial().point().to_vec(), false)
    }
}

/// REAL coordinates of a Hilbert-frame matrix placed on the central mode or
/// on every mode of the operator's domain.
fn placed(op: &DomainOperator, matrix: &[C], central_only: bool) -> Result<Vec<f64>, String> {
    let n = op.map().physical_dimension();
    let d = n * n;
    let mut signed = vec![C::default(); op.modes().len() * d];
    for (k, q) in op.modes().iter().enumerate() {
        if !central_only || q.iter().all(|&x| x == 0) {
            signed[k * d..(k + 1) * d].copy_from_slice(matrix);
        }
    }
    let mut pulled = vec![C::default(); op.rows()];
    op.map().pull_complex_into(&signed, 1, &mut pulled)?;
    let scale = pulled.iter().map(|z| z.re.abs()).fold(1_f64, f64::max);
    if pulled.iter().any(|z| z.im.abs() > 1e-12 * scale) {
        return Err("A probe envelope is not Hermitian".into());
    }
    Ok(pulled.iter().map(|z| z.re).collect())
}

/// Laboratory mean of `v = s (sE - F)^-1 E probe` for the single pole `s` of
/// `op`, and `v` itself (REAL coordinates); it tends to `Pi0 probe` as `s -> 0`.
/// A solve that stalls above `RESOLVENT_STALL` times the requested relative
/// residual is an error: its mean is not used.
fn resolvent_mean(core: &RuntimeCore, context: &Context, op: &mut DomainOperator, readout: &mut TargetReadout, probe: &[C],
    central_only: bool, tolerance: f64, applications: &mut usize) -> Result<(Vec<C>, Vec<f64>), String> {
    let pole = *op.poles().first().ok_or("Resolvent operator without a pole")?;
    let x = placed(op, probe, central_only)?;
    let rows = op.rows();
    let (mut ex, mut fx) = (vec![0.; rows], vec![0.; rows]);
    op.images(&x, 1, &mut ex, &mut fx)?;
    let settings = InnerSolve { maximum_steps: 80, relative_residual: tolerance };
    let (v, relative) = solve_shifted(op, &ex, settings, 12, applications)?;
    if !(relative <= RESOLVENT_STALL * tolerance) {
        return Err(format!("The resolvent solve at pole {pole:e}/us stalled at relative residual {relative:e} (requested {tolerance:e}); the tolerance was not relaxed"));
    }
    let scaled: Vec<f64> = v.iter().map(|value| pole * value).collect();
    let signed = op.signed_block(&scaled, 1)?;
    Ok((readout.density(core.physical(), &context.frame, &signed)?, scaled))
}

/// `|F x| / |E x|` in REAL coordinates: the relative residual of a null-vector
/// candidate `x` of the pencil.
fn relative_null_residual(op: &mut DomainOperator, x: &[f64], applications: &mut usize) -> Result<f64, String> {
    let rows = op.rows();
    let (mut ex, mut fx) = (vec![0.; rows], vec![0.; rows]);
    op.images(x, 1, &mut ex, &mut fx)?;
    *applications += 1;
    Ok(euclid(&fx) / euclid(&ex).max(f64::MIN_POSITIVE))
}

/// Compares the corrected null-vector mean with `Pi0` of the second probe.
/// A one-dimensional null space gives the same density up to the estimated
/// error of the candidate; a degenerate null space changes the mean by the
/// difference between conserved states. The resolvent at pole `s` is biased
/// by `s/gamma` times the probe's component on a slow mode of rate `gamma`
/// (rates down to 1e-6/us occur and the mixed or delta probes excite them
/// strongly), so a failed first comparison is repeated with the first-order
/// Richardson extrapolation of the poles `s` and `2s`.
fn degeneracy_probe(core: &mut RuntimeCore, options: &MeanOptions, context: &Context, candidate: &Candidate, state: &mut NullState,
    work: &mut Work) -> Result<Probe, String> {
    let (probe, central_only) = probe_matrix(core, context);
    let tolerance = (PROBE_TOLERANCE_PER_EPSILON * options.population_tolerance).max(RESOLVENT_TOLERANCE);
    let threshold = 0.25 * options.population_tolerance + SAFETY * (candidate.basis_estimate + candidate.halo_estimate);
    let (first, _) = resolvent_mean(core, context, &mut state.space.op, &mut state.readout, &probe, central_only, tolerance,
        &mut work.applications)?;
    let gap = context.metric(&difference(&first, &candidate.corrected), &options.groups)?;
    work.checkpoints.push(MeanCheckpoint { columns: candidate.columns, modes: candidate.modes.len(), pole: candidate.pole,
        basis_estimate: gap, halo_estimate: None, halo_modes: (0, 0), inner_residual: f64::NAN, action: "degeneracy-probe" });
    if gap <= threshold {
        return Ok(Probe::Unique);
    }
    let mut doubled = DomainOperator::new(core, &candidate.modes, &[2. * candidate.pole])?;
    let (second, _) = resolvent_mean(core, context, &mut doubled, &mut state.readout, &probe, central_only, tolerance,
        &mut work.applications)?;
    let first_order = difference(&first, &second);
    let mut extrapolated = first;
    add_into(&mut extrapolated, &first_order);
    // The remaining second-order term is below a tenth of the first-order one
    // whenever the pole is below a tenth of the slow rate.
    let allowance = 0.1 * context.metric(&first_order, &options.groups)?;
    let extrapolated_gap = context.metric(&difference(&extrapolated, &candidate.corrected), &options.groups)?;
    work.checkpoints.push(MeanCheckpoint { columns: candidate.columns, modes: candidate.modes.len(), pole: 2. * candidate.pole,
        basis_estimate: extrapolated_gap, halo_estimate: Some(allowance), halo_modes: (0, 0), inner_residual: f64::NAN,
        action: "degeneracy-probe-extrapolated" });
    if extrapolated_gap <= threshold + allowance {
        return Ok(Probe::Unique);
    }
    Ok(if context.axes == 0 { Probe::StaticDegenerate } else { Probe::Degenerate })
}

/// `Pi0` of the initial state (`X0` for driven models, the central initial
/// vector for static ones) by the resolvent at the poles `s` and `2s` with
/// first-order Richardson extrapolation; Fourier convergence by the change
/// when every neighbouring mode along EVERY Fourier step is added (weak
/// steps included, as the unique path's halo covers them: a weak leakage
/// channel can decide which conserved state is reached). A static model is
/// then classified by `Pi0` of the maximally mixed state.
fn degenerate(core: &mut RuntimeCore, options: &MeanOptions, context: &Context, mut modes: Vec<Charge>, zero_modes: usize,
    slowest_rate: Option<f64>, work: &mut Work) -> Result<MeanSolution, String> {
    let pole = context.pole;
    let initial = core.initial().point().to_vec();
    let central_only = context.axes == 0;
    // Extrapolated mean, first-order term, discarded optical coherence and the
    // relative residual of the extrapolated vector 2a - b.
    let evaluate = |core: &mut RuntimeCore, modes: &[Charge], matrix: &[C], work: &mut Work| -> Result<(Vec<C>, f64, f64, f64), String> {
        let mut readout = TargetReadout::new(core, modes, &context.frame)?;
        let mut op = DomainOperator::new(core, modes, &[pole])?;
        let (a, va) = resolvent_mean(core, context, &mut op, &mut readout, matrix, central_only, RESOLVENT_TOLERANCE, &mut work.applications)?;
        let discarded = readout.discarded(core.physical(), &context.frame)?;
        let mut op = DomainOperator::new(core, modes, &[2. * pole])?;
        let (b, vb) = resolvent_mean(core, context, &mut op, &mut readout, matrix, central_only, RESOLVENT_TOLERANCE, &mut work.applications)?;
        let first_order = difference(&a, &b);
        let mut extrapolated = a;
        add_into(&mut extrapolated, &first_order);
        let vector: Vec<f64> = va.iter().zip(&vb).map(|(a, b)| 2. * a - b).collect();
        let residual = relative_null_residual(&mut op, &vector, &mut work.applications)?;
        Ok((extrapolated, context.metric(&first_order, &options.groups)?, discarded, residual))
    };
    let clock = Instant::now();
    let (mut mean, mut extrapolation, mut discarded, mut residual) = evaluate(core, &modes, &initial, work)?;
    loop {
        let shell = shell_modes(&modes, &context.steps);
        let mut larger: Vec<Charge> = modes.iter().cloned().chain(shell.iter().cloned()).collect();
        larger.sort();
        let (halo_estimate, next) = if shell.is_empty() {
            (0., None)
        } else {
            let (value, next_extrapolation, next_discarded, next_residual) = evaluate(core, &larger, &initial, work)?;
            (context.metric(&difference(&value, &mean), &options.groups)?, Some((value, next_extrapolation, next_discarded, next_residual)))
        };
        work.checkpoints.push(MeanCheckpoint { columns: 0, modes: modes.len(), pole, basis_estimate: extrapolation,
            halo_estimate: Some(halo_estimate), halo_modes: (shell.len(), 0), inner_residual: f64::NAN, action: "degenerate-resolvent" });
        let projected = project_density(&context.lapack, &mean, context.n)?;
        let charge = context.metric(&difference(&projected.density, &mean), &options.groups)?;
        if SAFETY * (extrapolation + halo_estimate) + charge <= options.population_tolerance {
            work.timing[3] += ms(clock);
            // A static model with a degenerate reduced spectrum may still have
            // a unique stationary density: compare with Pi0 of the mixed state.
            let kind = if central_only {
                let (mixed, _) = probe_matrix(core, context);
                let (other, _, _, _) = evaluate(core, &modes, &mixed, work)?;
                if context.metric(&difference(&other, &mean), &options.groups)? <= 0.25 * options.population_tolerance {
                    MeanKind::Unique
                } else {
                    MeanKind::InitialDependent { zero_modes }
                }
            } else {
                MeanKind::InitialDependent { zero_modes }
            };
            let candidate = Candidate { raw_mean: mean.clone(), corrected: mean, discarded_optical_coherence: discarded,
                basis_estimate: extrapolation, halo_estimate, relative_residual: residual, slowest_rate, zero_modes,
                modes, pole, columns: 0 };
            return finish(context, options, candidate, kind, work);
        }
        match next {
            Some((value, next_extrapolation, next_discarded, next_residual)) if larger.len() <= options.maximum_modes && work.within_budget(options) => {
                modes = larger;
                mean = value;
                extrapolation = next_extrapolation;
                discarded = next_discarded;
                residual = next_residual;
            }
            _ => return Err("Initial-dependent long-time mean did not reach the tolerance; the tolerance was not relaxed".into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preparation::{DriveView, GaugePolicy, MatrixView, ModelView, PhaseBasis};
    use super::super::b0_inverse::P0Policy;
    use super::super::coordinates::InitialView;
    use super::super::KernelPolicy;

    /// A small model in the original state basis, row-major complex matrices,
    /// with at most one phase axis of frequency `nu`.
    #[derive(Clone)]
    struct Toy {
        n: usize,
        ground: Vec<bool>,
        h0: Vec<C>,
        /// (cosine, sine) drive matrices, all at frequency `nu`.
        drives: Vec<(Vec<C>, Vec<C>)>,
        collapse: Vec<Vec<C>>,
        initial: usize,
        nu: f64,
        /// Laboratory frame index of every state (one axis).
        frame: Vec<i32>,
    }

    fn real(x: &[C]) -> Vec<f64> { x.iter().map(|z| z.re).collect() }
    fn imag(x: &[C]) -> Vec<f64> { x.iter().map(|z| z.im).collect() }
    fn c(re: f64) -> C { C { re, im: 0. } }
    fn mul(a: C, b: C) -> C { C { re: a.re * b.re - a.im * b.im, im: a.re * b.im + a.im * b.re } }
    fn conj(a: C) -> C { C { re: a.re, im: -a.im } }
    fn add(a: &mut C, b: C) { a.re += b.re; a.im += b.im; }
    fn sub(a: &mut C, b: C) { a.re -= b.re; a.im -= b.im; }
    fn inverse(a: C) -> C { let norm = a.re * a.re + a.im * a.im; C { re: a.re / norm, im: -a.im / norm } }

    /// The prepared core and mean options of a toy, optionally with one more
    /// drive `(frequency, cosine matrix)` on a second phase axis (laboratory
    /// frame index 0 along it).
    fn toy_core(toy: &Toy, tolerance: f64, second: Option<(f64, &[C])>) -> (RuntimeCore, MeanOptions) {
        let n = toy.n;
        let zero = vec![C::default(); n * n];
        let mut parts: Vec<_> = toy.drives.iter().map(|(cos, sin)| (toy.nu, real(cos), imag(cos), real(sin), imag(sin))).collect();
        if let Some((nu2, cos2)) = second {
            parts.push((nu2, real(cos2), imag(cos2), real(&zero), imag(&zero)));
        }
        let drives: Vec<DriveView> = parts.iter().map(|p| DriveView { omega: p.0,
            cosine: MatrixView { real: &p.1, imag: &p.2 }, sine: MatrixView { real: &p.3, imag: &p.4 } }).collect();
        let channels: Vec<_> = toy.collapse.iter().map(|x| (real(x), imag(x))).collect();
        let collapse: Vec<MatrixView> = channels.iter().map(|(r, i)| MatrixView { real: r, imag: i }).collect();
        let (hr, hi) = (real(&toy.h0), imag(&toy.h0));
        let view = ModelView { dimension: n, hamiltonian: MatrixView { real: &hr, imag: &hi }, collapse: &collapse,
            drives: &drives, ground: &toy.ground, initial_index: toy.initial };
        let (phases, frame): (PhaseBasis, Vec<Charge>) = if parts.is_empty() {
            (PhaseBasis::from_declared(&[], None, &[]).unwrap(), vec![vec![]; n])
        } else if let Some((nu2, _)) = second {
            let charges: Vec<Charge> = toy.drives.iter().map(|_| vec![1, 0]).chain([vec![0, 1]]).collect();
            (PhaseBasis::from_declared(&[toy.nu, nu2], None, &charges).unwrap(), toy.frame.iter().map(|&f| vec![f, 0]).collect())
        } else {
            (PhaseBasis::from_declared(&[toy.nu], None, &vec![vec![1]; toy.drives.len()]).unwrap(),
                toy.frame.iter().map(|&f| vec![f]).collect())
        };
        let core = RuntimeCore::prepare(&view, InitialView { index: toy.initial, real: None, imag: None }, phases,
            GaugePolicy::UnsafeComponentAligned, KernelPolicy::HybridBlas, P0Policy::EigenPreferred).unwrap();
        let lab = LabFrame { frame_indices: frame, sectors: toy.ground.iter().map(|&g| if g { 0 } else { 1 }).collect() };
        let groups = vec![RequestedGroup { name: "excited".into(), indices: (0..n).filter(|&i| !toy.ground[i]).collect() }];
        (core, MeanOptions::new(tolerance, groups, lab))
    }

    fn solve_mean(toy: &Toy, tolerance: f64) -> MeanSolution {
        let (mut core, options) = toy_core(toy, tolerance, None);
        solve(&mut core, &options).unwrap()
    }

    /// Dense reference, independent of the gauge pencil: the ungauged
    /// Fourier-extended Lindblad generator K on charges -order..order and
    /// `s (s - K)^-1 X0` with a tiny `s` by complex Gaussian elimination,
    /// which is `Pi0 X0` for ANY null-space dimension. Laboratory mean.
    fn dense_mean(toy: &Toy, order: i32) -> Vec<C> {
        dense_mean_at(toy, order, 1e-10)
    }

    /// `dense_mean` with the pole `s`; `s = 0` on a static toy is the exact
    /// stationary state of a unique null space (one population equation
    /// replaced by the trace normalisation), free of any pole bias.
    fn dense_mean_at(toy: &Toy, order: i32, s: f64) -> Vec<C> {
        let n = toy.n;
        let d = n * n;
        let modes = (2 * order + 1) as usize;
        let size = modes * d;
        // -i [h, x]
        let commutator = |h: &[C], x: &[C]| -> Vec<C> {
            let mut out = vec![C::default(); d];
            for i in 0..n {
                for j in 0..n {
                    for k in 0..n {
                        add(&mut out[i * n + j], mul(h[i * n + k], x[k * n + j]));
                        sub(&mut out[i * n + j], mul(x[i * n + k], h[k * n + j]));
                    }
                }
            }
            out.iter().map(|z| C { re: z.im, im: -z.re }).collect()
        };
        let lindblad = |x: &[C]| -> Vec<C> {
            let mut out = commutator(&toy.h0, x);
            for l in &toy.collapse {
                for i in 0..n {
                    for j in 0..n {
                        let mut value = C::default();
                        for a in 0..n {
                            for b in 0..n {
                                add(&mut value, mul(mul(l[i * n + a], x[a * n + b]), conj(l[j * n + b])));
                                // -1/2 (L^+ L x + x L^+ L)
                                let left = mul(conj(l[b * n + i]), l[b * n + a]);
                                let right = mul(conj(l[b * n + a]), l[b * n + j]);
                                let anti = mul(left, x[a * n + j]);
                                let anti2 = mul(x[i * n + a], right);
                                value.re -= 0.5 * (anti.re + anti2.re);
                                value.im -= 0.5 * (anti.im + anti2.im);
                            }
                        }
                        add(&mut out[i * n + j], value);
                    }
                }
            }
            out
        };
        // e^{+i nu t}: (cos - i sin)/2 ; e^{-i nu t}: (cos + i sin)/2.
        let plus: Vec<Vec<C>> = toy.drives.iter().map(|(cs, sn)| cs.iter().zip(sn)
            .map(|(a, b)| C { re: 0.5 * (a.re + b.im), im: 0.5 * (a.im - b.re) }).collect()).collect();
        let minus: Vec<Vec<C>> = toy.drives.iter().map(|(cs, sn)| cs.iter().zip(sn)
            .map(|(a, b)| C { re: 0.5 * (a.re - b.im), im: 0.5 * (a.im + b.re) }).collect()).collect();
        // Row-major (s - K).
        let mut matrix = vec![C::default(); size * size];
        for column in 0..size {
            let (q, k) = (column / d, column % d);
            let mut x = vec![C::default(); d];
            x[k] = c(1.);
            let charge = q as f64 - order as f64;
            let mut put = |mode: usize, values: &[C]| {
                for (row, z) in values.iter().enumerate() {
                    sub(&mut matrix[(mode * d + row) * size + column], *z);
                }
            };
            let mut diagonal = lindblad(&x);
            diagonal[k].im -= charge * toy.nu;
            put(q, &diagonal);
            for p in 0..toy.drives.len() {
                if q + 1 < modes {
                    put(q + 1, &commutator(&plus[p], &x));
                }
                if q >= 1 {
                    put(q - 1, &commutator(&minus[p], &x));
                }
            }
            matrix[column * size + column].re += s;
        }
        let mut rhs = vec![C::default(); size];
        for q in 0..modes {
            rhs[q * d + toy.initial * n + toy.initial] = c(s);
        }
        if s == 0. {
            assert!(toy.drives.is_empty() && order == 0, "the exact stationary state is for static toys");
            for j in 0..size {
                matrix[j] = C::default();
            }
            for i in 0..n {
                matrix[i * n + i] = c(1.);
            }
            rhs[0] = c(1.);
        }
        for k in 0..size {
            let magnitude = |z: C| z.re.hypot(z.im);
            let pivot_row = (k..size).max_by(|&a, &b| magnitude(matrix[a * size + k]).total_cmp(&magnitude(matrix[b * size + k]))).unwrap();
            if pivot_row != k {
                for j in 0..size {
                    matrix.swap(k * size + j, pivot_row * size + j);
                }
                rhs.swap(k, pivot_row);
            }
            let pivot = inverse(matrix[k * size + k]);
            for i in k + 1..size {
                let factor = mul(matrix[i * size + k], pivot);
                if factor.re == 0. && factor.im == 0. {
                    continue;
                }
                for j in k..size {
                    let t = mul(factor, matrix[k * size + j]);
                    sub(&mut matrix[i * size + j], t);
                }
                let t = mul(factor, rhs[k]);
                sub(&mut rhs[i], t);
            }
        }
        let mut x = vec![C::default(); size];
        for i in (0..size).rev() {
            let mut sum = rhs[i];
            for j in i + 1..size {
                sub(&mut sum, mul(matrix[i * size + j], x[j]));
            }
            x[i] = mul(sum, inverse(matrix[i * size + i]));
        }
        let mut mean = vec![C::default(); d];
        for i in 0..n {
            for j in 0..n {
                if toy.ground[i] != toy.ground[j] {
                    continue;
                }
                let charge = if toy.drives.is_empty() { 0 } else { toy.frame[i] - toy.frame[j] };
                mean[i * n + j] = x[(charge + order) as usize * d + i * n + j];
            }
        }
        mean
    }

    fn largest_difference(a: &[C], b: &[C]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x.re - y.re).hypot(x.im - y.im)).fold(0., f64::max)
    }

    fn couple(m: &mut [C], n: usize, i: usize, j: usize, v: f64) {
        m[i * n + j] = c(v);
        m[j * n + i] = c(v);
    }

    /// Lambda system: two ground states, one excited state decaying to both.
    /// Driven: the second leg is a drive at nu, so the laboratory ground
    /// coherence is a nonzero model-frame harmonic (frame indices 0, -1).
    fn lambda(driven: bool) -> Toy {
        let n = 3;
        let mut h0 = vec![C::default(); 9];
        h0[4] = c(0.4);
        h0[8] = c(-0.7);
        couple(&mut h0, n, 0, 2, 0.6);
        let mut drives = Vec::new();
        if driven {
            let mut cos = vec![C::default(); 9];
            couple(&mut cos, n, 1, 2, 0.8);
            drives.push((cos, vec![C::default(); 9]));
        } else {
            couple(&mut h0, n, 1, 2, 0.3);
        }
        let mut first = vec![C::default(); 9];
        first[2] = c(0.9);
        let mut second = vec![C::default(); 9];
        second[n + 2] = c(0.7);
        Toy { n, ground: vec![true, true, false], h0, drives, collapse: vec![first, second], initial: 0, nu: 2.3, frame: vec![0, -1, 0] }
    }

    /// Closed periodically driven ground pair plus an undriven excited state
    /// decaying into it: a two-dimensional null space whose conserved
    /// quantities depend on the drive phase (initial-phase averaging differs).
    fn degenerate_driven() -> Toy {
        let n = 3;
        let mut h0 = vec![C::default(); 9];
        h0[0] = c(-0.65);
        h0[4] = c(0.65);
        let mut cos = vec![C::default(); 9];
        couple(&mut cos, n, 0, 1, 0.9);
        let mut decay = vec![C::default(); 9];
        decay[2] = c(1.);
        Toy { n, ground: vec![true, true, false], h0, drives: vec![(cos, vec![C::default(); 9])], collapse: vec![decay],
            initial: 2, nu: 2.1, frame: vec![0, 0, 0] }
    }

    #[test]
    fn static_mean_matches_the_dense_stationary_density() {
        let toy = lambda(false);
        let mean = solve_mean(&toy, 1e-4);
        assert_eq!(mean.kind, MeanKind::Unique);
        let error = largest_difference(&mean.density, &dense_mean(&toy, 0));
        assert!(error < 1e-4, "static error {error:e}");
        assert!(mean.discarded_optical_coherence > 1e-3, "a driven Lambda keeps an optical coherence");
    }

    #[test]
    fn driven_mean_matches_dense_floquet_including_a_laboratory_coherence() {
        let toy = lambda(true);
        let mean = solve_mean(&toy, 1e-4);
        let reference = dense_mean(&toy, 12);
        let error = largest_difference(&mean.density, &reference);
        assert!(error < 1e-4, "driven error {error:e}");
        assert!(reference[1].re.hypot(reference[1].im) > 1e-3, "the ground coherence is a nonzero model-frame harmonic");
        assert!(mean.diagnostics.estimated_error <= 1e-4);
        assert_eq!(mean.diagnostics.conserved_quantities, Some(1));
        let relaxation = mean.diagnostics.slowest_relaxation_us.expect("a decaying mode");
        assert!(relaxation.is_finite() && relaxation > 1. && relaxation < 100., "slowest relaxation {relaxation} us");
    }

    /// The adapter's combined request: the finite engine on one fork of the
    /// core waits for the slow dual that the mean, on another fork, prepares,
    /// and both run concurrently. Each result equals the same solve alone.
    #[test]
    fn a_mean_and_the_finite_engine_run_concurrently_on_forks_of_one_core() {
        use super::super::engine::{self, DualSource, FiniteOptions};
        let toy = lambda(true);
        let (core, options) = toy_core(&toy, 1e-4, None);
        let times: Vec<f64> = (0..21).map(|i| 4. * i as f64 / 20.).collect();
        let finite = FiniteOptions { population_tolerance: 1e-3, ..FiniteOptions::default() };
        let alone_mean = solve(&mut core.fork().unwrap(), &options).unwrap();
        let alone_finite = engine::solve(core.fork().unwrap(), &times, options.groups.clone(), finite.clone()).unwrap();
        let mut mean_core = core.fork().unwrap();
        let cell = std::sync::OnceLock::new();
        let (mean, concurrent) = std::thread::scope(|scope| {
            let finite = scope.spawn(|| engine::solve_with(core, &times, options.groups.clone(), finite, DualSource::Shared(&cell)));
            let dual = cell.get_or_init(|| SlowDual::prepare(mean_core.prepared(), SlowDualOptions::default()));
            let mean = solve_with(&mut mean_core, &options, dual.as_ref().ok()).unwrap();
            (mean, finite.join().unwrap().unwrap())
        });
        let parts = |m: &MeanSolution| m.density.iter().map(|z| (z.re, z.im)).collect::<Vec<_>>();
        assert_eq!(parts(&mean), parts(&alone_mean));
        assert_eq!(concurrent.populations, alone_finite.populations);
        assert_eq!(concurrent.diagnostics.error_control, "dual-weighted-residual");
    }

    #[test]
    fn degenerate_driven_mean_is_the_theta_zero_projection_not_the_phase_average() {
        let toy = degenerate_driven();
        let mean = solve_mean(&toy, 1e-4);
        assert!(matches!(mean.kind, MeanKind::InitialDependent { .. }), "{:?}", mean.kind);
        assert_eq!(mean.diagnostics.conserved_quantities, Some(2), "a conserved quantity besides the trace");
        // Independent Floquet reference (monodromy projector, scratch analysis)
        // and the dense resolvent of the extended generator.
        let expected = [0.673781965169, 0.326218034831, 0.];
        for (p, e) in mean.populations.iter().zip(expected) {
            assert!((p - e).abs() < 1e-4, "{p} vs {e}");
        }
        assert!(largest_difference(&mean.density, &dense_mean(&toy, 14)) < 1e-4);
        let residual = mean.diagnostics.relative_residual;
        assert!(residual.is_finite() && residual < 1e-3, "{residual:e}");
    }

    #[test]
    fn static_degenerate_null_space_is_classified_initial_dependent() {
        // No coupling: the excited state decays into two conserved ground
        // states with branching 0.81 : 0.49.
        let n = 3;
        let mut first = vec![C::default(); 9];
        first[2] = c(0.9);
        let mut second = vec![C::default(); 9];
        second[n + 2] = c(0.7);
        let toy = Toy { n, ground: vec![true, true, false], h0: vec![C::default(); 9], drives: vec![], collapse: vec![first, second],
            initial: 2, nu: 1., frame: vec![0, 0, 0] };
        let mean = solve_mean(&toy, 1e-4);
        assert!(matches!(mean.kind, MeanKind::InitialDependent { .. }), "{:?}", mean.kind);
        assert!((mean.populations[0] - 0.81 / 1.3).abs() < 1e-4 && (mean.populations[1] - 0.49 / 1.3).abs() < 1e-4,
            "{:?}", mean.populations);
    }

    #[test]
    fn slow_unique_pumping_is_resolved_and_not_mistaken_for_degeneracy() {
        // A third ground state couples to the excited state 1000 times more
        // weakly: it is pumped at about 1e-5/us, so the stationary state is
        // unique but a resolvent probe at the near-zero pole is biased by
        // pole/rate on that slow mode.
        let n = 4;
        let mut h0 = vec![C::default(); 16];
        h0[5] = c(0.4);
        h0[10] = c(0.2);
        couple(&mut h0, n, 0, 3, 0.6);
        couple(&mut h0, n, 1, 3, 0.3);
        couple(&mut h0, n, 2, 3, 0.003);
        let mut decays = Vec::new();
        for (ground, amplitude) in [(0, 0.9), (1, 0.7), (2, 0.4)] {
            let mut l = vec![C::default(); 16];
            l[ground * n + 3] = c(amplitude);
            decays.push(l);
        }
        let toy = Toy { n, ground: vec![true, true, true, false], h0, drives: vec![], collapse: decays, initial: 2, nu: 1.,
            frame: vec![0; 4] };
        let reference = dense_mean(&toy, 0);
        for tolerance in [1e-3, 1e-4] {
            let mean = solve_mean(&toy, tolerance);
            assert_eq!(mean.kind, MeanKind::Unique, "tolerance {tolerance}");
            let error = largest_difference(&mean.density, &reference);
            assert!(error <= tolerance, "tolerance {tolerance}: error {error:e}");
        }
    }

    #[test]
    fn a_slow_but_finite_relaxation_reaches_its_unique_limit() {
        // A third ground state is filled through a 1e-4 decay amplitude
        // (branching 1e-8) and emptied through a 3e-6 coupling: it relaxes at
        // about 1.4e-9/us, far below the old resolvable rate (1e-6/us, below
        // which rates were taken as conserved), and the unique limit puts
        // almost all population there. The slow generator resolves the rate
        // and sets the pole below it; the mean is the true limit, not an
        // initial-dependent short-time state.
        let n = 4;
        let mut h0 = vec![C::default(); 16];
        h0[5] = c(0.4);
        h0[10] = c(0.2);
        couple(&mut h0, n, 0, 3, 0.6);
        couple(&mut h0, n, 1, 3, 0.3);
        couple(&mut h0, n, 2, 3, 3e-6);
        let mut decays = Vec::new();
        for (ground, amplitude) in [(0, 0.9), (1, 0.7), (2, 1e-4)] {
            let mut l = vec![C::default(); 16];
            l[ground * n + 3] = c(amplitude);
            decays.push(l);
        }
        let toy = Toy { n, ground: vec![true, true, true, false], h0, drives: vec![], collapse: decays, initial: 0, nu: 1.,
            frame: vec![0; 4] };
        let reference = dense_mean_at(&toy, 0, 0.);
        assert!(reference[2 * n + 2].re > 0.5, "the limit fills the slow state ({})", reference[2 * n + 2].re);
        let mean = solve_mean(&toy, 1e-4);
        assert_eq!(mean.kind, MeanKind::Unique);
        assert_eq!(mean.diagnostics.conserved_quantities, Some(1));
        let relaxation = mean.diagnostics.slowest_relaxation_us.expect("the slow relaxation is resolved");
        assert!((1e8..1e10).contains(&relaxation), "slowest relaxation {relaxation:e} us");
        assert!(mean.diagnostics.pole <= 1e-10 * 1.0001, "pole {:e}", mean.diagnostics.pole);
        let error = largest_difference(&mean.density, &reference);
        assert!(error <= 1e-4, "error {error:e}");
    }

    /// Populations of a static toy at time `t` by the dense exponential of its
    /// real 2 n^2 Lindblad generator (independent of the pencil and of J_eff).
    fn static_populations_at(toy: &Toy, t: f64) -> Vec<f64> {
        assert!(toy.drives.is_empty());
        let n = toy.n;
        let d = n * n;
        let lindblad = |x: &[C]| -> Vec<C> {
            let mut out = vec![C::default(); d];
            for i in 0..n {
                for j in 0..n {
                    // -i [h, x]
                    let mut commutator = C::default();
                    for k in 0..n {
                        add(&mut commutator, mul(toy.h0[i * n + k], x[k * n + j]));
                        sub(&mut commutator, mul(x[i * n + k], toy.h0[k * n + j]));
                    }
                    add(&mut out[i * n + j], C { re: commutator.im, im: -commutator.re });
                    for l in &toy.collapse {
                        for a in 0..n {
                            for b in 0..n {
                                add(&mut out[i * n + j], mul(mul(l[i * n + a], x[a * n + b]), conj(l[j * n + b])));
                                let anti = mul(mul(conj(l[b * n + i]), l[b * n + a]), x[a * n + j]);
                                let anti2 = mul(x[i * n + a], mul(conj(l[b * n + a]), l[b * n + j]));
                                sub(&mut out[i * n + j], C { re: 0.5 * (anti.re + anti2.re), im: 0.5 * (anti.im + anti2.im) });
                            }
                        }
                    }
                }
            }
            out
        };
        // Real coordinates [Re x (d), Im x (d)].
        let size = 2 * d;
        let mut generator = vec![0.; size * size];
        for column in 0..size {
            let mut x = vec![C::default(); d];
            if column < d { x[column].re = 1. } else { x[column - d].im = 1. }
            let image = lindblad(&x);
            for k in 0..d {
                generator[column * size + k] = image[k].re;
                generator[column * size + d + k] = image[k].im;
            }
        }
        let lapack = Lapack::load().unwrap();
        let propagator = crate::dense_expm::exponential(&lapack, &generator, size, t).unwrap();
        let start = toy.initial * n + toy.initial;
        (0..n).map(|s| propagator[start * size + s * n + s]).collect()
    }

    /// The mean at a horizon: exact propagation of the slow relaxation over
    /// the horizon, continuous towards the limit, initial-dependent while the
    /// slow mode still carries population.
    #[test]
    fn a_horizon_mean_propagates_the_slow_relaxation_exactly() {
        // The toy of `a_slow_but_finite_relaxation_reaches_its_unique_limit`:
        // the third ground state fills at about 1.4e-9/us.
        let n = 4;
        let mut h0 = vec![C::default(); 16];
        h0[5] = c(0.4);
        h0[10] = c(0.2);
        couple(&mut h0, n, 0, 3, 0.6);
        couple(&mut h0, n, 1, 3, 0.3);
        couple(&mut h0, n, 2, 3, 3e-6);
        let mut decays = Vec::new();
        for (ground, amplitude) in [(0, 0.9), (1, 0.7), (2, 1e-4)] {
            let mut l = vec![C::default(); 16];
            l[ground * n + 3] = c(amplitude);
            decays.push(l);
        }
        let toy = Toy { n, ground: vec![true, true, true, false], h0, drives: vec![], collapse: decays, initial: 0, nu: 1.,
            frame: vec![0; 4] };
        let limit = solve_mean(&toy, 1e-4);
        for horizon in [1e7, 3e8, 2e9, 1e13] {
            let (mut core, mut options) = toy_core(&toy, 1e-4, None);
            options.horizon_us = Some(horizon);
            let mean = solve(&mut core, &options).unwrap();
            let exact = static_populations_at(&toy, horizon);
            let error = mean.populations.iter().zip(&exact).map(|(a, b)| (a - b).abs()).fold(0_f64, f64::max);
            let part = mean.diagnostics.horizon.as_ref().expect("horizon part");
            println!("horizon {horizon:e}: {} error {error:e}, relaxing bound {:e}, propagation {:e}, projection {:e}, estimate {:e}", part.method,
                part.decaying, part.propagation_estimate, part.projection_charge, mean.diagnostics.estimated_error);
            assert!(mean.diagnostics.estimated_error <= 1e-4);
            if horizon <= 2e9 {
                // (The dense exponential itself loses accuracy far beyond.)
                assert!(error <= mean.diagnostics.estimated_error, "horizon {horizon:e}: error {error:e}");
                assert_eq!(part.method, "propagation");
                assert!(part.decaying > 1e-2, "the slow state is still filling at {horizon:e}");
                assert!(matches!(mean.kind, MeanKind::InitialDependent { .. }));
            } else {
                assert_eq!(part.method, "limit");
                assert_eq!(mean.kind, MeanKind::Unique);
                let gap = mean.populations.iter().zip(&limit.populations).map(|(a, b)| (a - b).abs()).fold(0_f64, f64::max);
                assert!(gap <= 1e-6, "far horizon against the limit {gap:e}");
            }
        }
    }

    #[test]
    fn the_degenerate_branch_checks_every_fourier_step() {
        // The degenerate driven pair plus a weak, far-detuned modulation of
        // the excited energy on a second axis: its mixing amplitude is below
        // the domain threshold, so the starting domain ignores that step, but
        // the Fourier check of the initial-dependent projection must add the
        // neighbours along it too.
        let toy = degenerate_driven();
        let mut modulation = vec![C::default(); 9];
        modulation[8] = c(0.002);
        let (mut core, options) = toy_core(&toy, 1e-4, Some((50., &modulation)));
        let steps = core.fourier_steps().unwrap();
        assert!(steps.all.len() > steps.domain.len(), "the modulation is a weak step");
        let mean = solve(&mut core, &options).unwrap();
        assert!(matches!(mean.kind, MeanKind::InitialDependent { .. }), "{:?}", mean.kind);
        let first = mean.diagnostics.checkpoints.iter().find(|c| c.action == "degenerate-resolvent").expect("the degenerate branch");
        let modes = support_domain(steps.axes, &steps.domain, 1).unwrap();
        assert_eq!(first.halo_modes.0, shell_modes(&modes, &steps.all).len(), "shell over every step");
        assert!(first.halo_modes.0 > shell_modes(&modes, &steps.domain).len());
    }

    #[test]
    fn a_stalled_resolvent_solve_is_not_accepted() {
        let toy = degenerate_driven();
        let (mut core, options) = toy_core(&toy, 1e-4, None);
        let context = Context::new(&core, &options, None).unwrap();
        let modes = support_domain(context.axes, &context.domain_steps, 1).unwrap();
        let mut op = DomainOperator::new(&mut core, &modes, &[context.pole]).unwrap();
        let mut readout = TargetReadout::new(&core, &modes, &context.frame).unwrap();
        let initial = core.initial().point().to_vec();
        let mut applications = 0;
        let unreachable = resolvent_mean(&core, &context, &mut op, &mut readout, &initial, false, 1e-300, &mut applications);
        assert!(unreachable.as_ref().is_err_and(|e| e.contains("stalled")), "{:?}", unreachable.map(|_| ()));
        assert!(resolvent_mean(&core, &context, &mut op, &mut readout, &initial, false, 1e-9, &mut applications).is_ok());
    }

    /// Every exactly-zero forbidden element (E->G collapse, G-E Hamiltonian
    /// and drive couplings) receives a random relative 1e-9..1e-6 value.
    fn perturbed(toy: &Toy, seed: u64) -> Toy {
        let mut toy = toy.clone();
        let mut state = seed.max(1);
        let mut random = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % 1_000_000) as f64 / 1e6
        };
        let n = toy.n;
        let scale = |m: &[C]| m.iter().map(|z| z.re.hypot(z.im)).fold(0., f64::max).max(1.);
        let ground = toy.ground.clone();
        for l in &mut toy.collapse {
            let s = scale(l);
            for g in 0..n {
                for e in 0..n {
                    if ground[g] && !ground[e] && l[g * n + e].re == 0. && l[g * n + e].im == 0. {
                        l[g * n + e] = c(s * 10f64.powf(-9. + 3. * random()));
                    }
                }
            }
        }
        let mut matrices: Vec<&mut Vec<C>> = vec![&mut toy.h0];
        for (cos, _) in &mut toy.drives {
            matrices.push(cos);
        }
        for m in matrices {
            let s = scale(m);
            for i in 0..n {
                for j in 0..n {
                    if ground[i] && !ground[j] && m[i * n + j].re == 0. && m[i * n + j].im == 0. {
                        let v = s * 10f64.powf(-9. + 3. * random());
                        m[i * n + j] = c(v);
                        m[j * n + i] = c(v);
                    }
                }
            }
        }
        toy
    }

    #[test]
    fn forbidden_element_perturbations_keep_the_mean_continuous() {
        for (name, base) in [("static", lambda(false)), ("driven", lambda(true)), ("degenerate", degenerate_driven())] {
            let started = Instant::now();
            let reference = solve_mean(&base, 1e-4);
            let unperturbed_seconds = started.elapsed().as_secs_f64();
            for seed in [3, 11, 29] {
                let started = Instant::now();
                let changed = solve_mean(&perturbed(&base, seed), 1e-4);
                let difference = largest_difference(&changed.density, &reference.density);
                assert!(difference < 2e-4, "{name} seed {seed}: {difference:e}");
                assert!(started.elapsed().as_secs_f64() < 10. * unperturbed_seconds + 5., "{name} seed {seed} runtime");
            }
        }
    }
}
