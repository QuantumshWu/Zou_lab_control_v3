//! Population-error estimates for a reduced candidate, driven by the slow dual.
//!
//! # Error equation
//!
//! The candidate density (frame coordinates) is
//! `Xhat(t) = sum_q exp(i nu_q t) (E Q a(t))_q` with `a' = G a`, `a(0) = a0`.
//! Its residual `r = L Xhat - Xhat'` equals the pencil residual
//! `sum_q exp(i nu_q t) ((F Q - E Q G) a(t))_q`, because the pencil is
//! `E = T` and `F = L o T - Omega.d_theta T` in the same gauged,
//! Hilbert-rotated Fourier representation. The error `e = X - Xhat` obeys
//! `e' = L e + r`, `e(0) = e0` (zero for candidates built from the exact
//! initial state).
//!
//! # Basis estimate (every checkpoint; retained rows only)
//!
//! The slow part of `e` obeys `e_P' = J_eff e_P + P r`. The ground block of the
//! retained residual at charge `k` is fed to the slow slot of charge `k` when
//! `k` is slow, otherwise to the central slot with its physical phase
//! `exp(i nu_k t)` (the frame representation is redundant: a ground field at
//! charge `k` with envelope `f(t)` equals one at charge 0 with envelope
//! `exp(i nu_k t) f(t)`). The non-ground residual at every retained charge
//! `k` is integrated exactly by the dressed charge-0 no-jump dynamics
//! `eta' = (N - i nu_k) eta + r_Q`, `eta(0) = 0` (diagonal in the eigenvectors
//! of `A_L`, `A_R`); its populations are read out directly (phase
//! `exp(i nu_k t)`) and its recycling `R_0 eta_EE` is a slow source, routed
//! like a ground field at `k`. Diagonalizing the reduced generator
//! `G = V diag(beta) V^-1` turns every source into a sum of exponentials, and
//! `int_0^t exp(alpha (t-s)) exp(mu s) ds = (exp(mu t) - exp(alpha t))/(mu - alpha)`
//! is evaluated exactly (entire-function branch when `|mu - alpha| t <= 1`).
//! The cost is independent of the number of output times except for the
//! final `O(K)` readout.
//!
//! # Halo estimate (near acceptance; Fourier-truncation part)
//!
//! For every halo charge `q` (reached from the retained domain by one pencil
//! application) the halo residual `R_q = (F Q - E Q G)_q` drives the envelope
//! response `eta_q' = (N - i nu_q) eta_q + R_q a(t)`, with `N` the complete
//! charge-0 no-jump operator in its dressed eigen-coordinates
//! (`lambda_ik = a_i + b_k`; the diagonal `d_ab = A_L[a,a] + A_R[b,b]` is the
//! fallback when the dressed basis is ill-conditioned). Componentwise:
//!
//! * well separated (`|beta_j + i nu_q - lambda| >= floor`): the particular
//!   response `c / (beta_j + i nu_q - lambda) exp(beta_j t)`;
//! * near resonant: the zero-initial entire response
//!   `c (exp(beta_j t) - exp((lambda - i nu_q) t)) / (beta_j + i nu_q - lambda)`.
//!
//! When `q` is itself a slot its ground block is slow content: it goes to the
//! slot of `q` as a direct `J_eff` source and only the non-ground rest gets
//! the response above. A slow halo charge that is not a slot (reached only
//! through three or more hops) keeps its ground block in the response, where
//! its near-resonant coherences take the entire form and accumulate. The exact identity
//! `e = eta + U(e0 - eta(0)) + int U (r - eta' + L eta)` then gives the halo
//! contribution: `eta(t)` (read out directly), the initial boundary `-eta(0)`
//! (slow part into the central slot, the rest as an exact dressed fast layer),
//! and one Liouvillian hop `L eta` back into the retained domain:
//!
//! * same charge (`p = 0`): the ground block of `(L_0 - N) eta = R_0 eta_EE`;
//! * `p != 0` into a retained or slow charge `k`: the ground block of
//!   `L_p eta` is a slow source; its non-ground block gets the exact dressed
//!   fast response when `k` is slow, and (optionally) Markov slaving plus one
//!   hop when `k` is fast.
//!
//! Evaluated are the halo charges that return into the retained domain or a
//! slot through a significant coupling in one hop, and every slow halo charge
//! (its own populations accumulate even without such a return). Each halo
//! charge pair `{q, -q}` yields its own signed contribution, which is the
//! domain-selection indicator.
use super::*;
use super::real_fourier::RealFourierMap;
use super::slow_dual::{cabs, cadd, cdiv, cmul, complex, csub, frequency_key, robust_eigen, sandwich_all, Dense, LuFactor, Series, SlowDual, TimeGrid, ZERO};
#[cfg(test)]
use super::slow_dual::{cexp, cscale};

/// A requested signed population sum (for example one F manifold).
#[derive(Clone, Debug)]
pub struct PopulationGroup {
    pub name: String,
    pub indices: Vec<usize>,
}

/// Plain-data view of one reduced candidate. All REAL arrays are column-major
/// `[rows, columns]` in the REAL Fourier coordinates of `map`.
pub struct CandidateView<'a> {
    pub map: &'a RealFourierMap,
    pub q: &'a [f64],
    pub eq: &'a [f64],
    pub fq: &'a [f64],
    pub columns: usize,
    /// UNROTATED reduced generator (m x m, column-major): `a' = G a`.
    pub generator: &'a [f64],
    /// Reduced initial coefficients `a(0)`.
    pub initial: &'a [f64],
    /// Output times (us), any order of nonnegative finite values. Uniform
    /// grids starting at zero are evaluated by exponential recurrences.
    pub times: &'a [f64],
}

#[derive(Clone, Debug)]
pub struct PopulationErrorEstimate {
    pub times: Vec<f64>,
    pub states: usize,
    /// Signed estimate of (true - candidate) populations, `[time, state]`.
    pub state_error: Vec<f64>,
    pub group_names: Vec<String>,
    /// Signed group sums, `[time, group]`.
    pub group_error: Vec<f64>,
    pub maximum_state: f64,
    pub maximum_group: f64,
    /// `max(maximum_state, maximum_group) + margin`.
    pub maximum: f64,
    /// Unsigned allowance included in `maximum`: for a resolved estimate the
    /// charge for what the dual leaves out by magnitude (the slow charges only
    /// weak couplings reach, see `unrepresented_charge`), zero when nothing
    /// is left out; 1 (no information on the probability scale) when the
    /// candidate could not be resolved (see `unresolved`).
    pub margin: f64,
    /// Why the estimate is only the trivial conservative bound.
    pub unresolved: Option<String>,
    /// Largest imaginary part discarded by the real readout (consistency).
    pub maximum_imaginary: f64,
    pub elapsed_ms: f64,
}

#[derive(Clone, Debug)]
pub struct HaloModeIndicator {
    /// Canonical member of the signed pair `{charge, -charge}`.
    pub charge: Charge,
    /// Max over time, states and groups of the pair's signed contribution
    /// (its residual response, returns and slow boundary; the shared fast
    /// responses enter the total only), on at most `INDICATOR_TIMES` samples.
    pub indicator: f64,
    /// Frobenius norm of the pair's halo residual coefficients (diagnostic).
    pub residual_norm: f64,
}

#[derive(Clone, Debug)]
pub struct HaloEstimate {
    pub estimate: PopulationErrorEstimate,
    /// Sorted by decreasing indicator.
    pub modes: Vec<HaloModeIndicator>,
    pub halo_modes: usize,
    pub near_resonant_terms: usize,
    /// Near-resonant non-ground returns into a slow charge that were not
    /// integrated (would need a pair-source fast response).
    pub unrouted_resonant_returns: usize,
    pub pencil_ms: f64,
    /// Modal columns per pencil application.
    pub block_columns: usize,
}

/// Near-resonant halo response component `unit * e_t[beta_j, second]`.
struct ResonantTerm {
    column: usize,
    /// Frame matrix of the component (coefficient included).
    unit: Dense,
    /// Envelope node `lambda - i nu_q`.
    second: C,
}

#[derive(Clone, Copy, Debug)]
pub struct HaloOptions {
    /// Modal columns per pencil application. One plan covers every halo mode
    /// (its intermediate modes are computed once per block); the plan and the
    /// halo rows held at a time grow with this width.
    pub maximum_rhs: usize,
    /// Response components with `|beta + i nu - d| < floor` use the
    /// zero-initial entire form. `None` selects the optical decay floor.
    pub resonance_floor: Option<f64>,
    /// Evaluate every halo mode, including those that cannot return into the
    /// retained domain in one Liouvillian hop (they then contribute only their
    /// own oscillating populations, their initial boundary and their
    /// same-charge recycling, which reaches the slow error with a fast phase).
    pub all_halo_modes: bool,
    /// Additionally Markov-slave the non-ground returns that land on fast
    /// retained charges and return them one more drive hop into the slow
    /// slots. Off by default: it never changed a halo estimate that mattered
    /// on the benchmarks while costing about 2-3x the pencil (one exact
    /// charge-block solve per column).
    pub slave_fast_returns: bool,
}

impl Default for HaloOptions {
    fn default() -> Self {
        Self { maximum_rhs: 2, resonance_floor: None, all_halo_modes: false, slave_fast_returns: false }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BasisOptions {
    /// Integrate the non-ground residual at every retained charge exactly
    /// through the dressed no-jump dynamics (early fast transients).
    pub fast_response: bool,
    /// Markov-slave the non-ground residual at fast retained charges and
    /// return it one drive hop into the slow slots (with the second-hop
    /// relaxation there). The dressed fast response stays at its own charge;
    /// this is the drive-mediated path back to the ground block, which carries
    /// the late-time error of strongly driven candidates (without it the
    /// estimate fell to 0.6x the true error on threeNearStrong).
    pub slave_fast_charges: bool,
}

impl Default for BasisOptions {
    fn default() -> Self { Self { fast_response: true, slave_fast_charges: true } }
}

/// Modal columns per block of the basis estimate (bounds its memory).
const BASIS_BLOCK: usize = 8;
/// Samples used for the per-pair halo indicators (a uniform subset of the
/// output times, always including the last one).
const INDICATOR_TIMES: usize = 64;

// ---------------------------------------------------------------------------
// Reduced candidate dynamics
// ---------------------------------------------------------------------------

/// Eigen-decomposition of the reduced generator: `a(t) = V exp(beta t) w`.
struct SmallSource {
    exponents: Vec<C>,
    vectors: Dense,
    weights: Vec<C>,
}

impl SmallSource {
    fn new(lap: &Lapack, generator: &[f64], initial: &[f64], m: usize) -> Result<Self, String> {
        if m == 0 || generator.len() != m * m || initial.len() != m {
            return Err("Reduced generator/initial shape mismatch".into());
        }
        if generator.iter().chain(initial).any(|x| !x.is_finite()) {
            return Err("Nonfinite reduced generator/initial".into());
        }
        let complex_g = Dense { rows: m, cols: m, data: generator.iter().map(|&x| complex(x, 0.)).collect() };
        let eigen = robust_eigen(&complex_g, Some(generator), lap)?;
        let (exponents, vectors) = (eigen.values, eigen.vectors);
        let lu = LuFactor::new(&vectors, lap)?;
        let a0: Vec<C> = initial.iter().map(|&x| complex(x, 0.)).collect();
        let weights = lu.solve(&a0, lap)?;
        // Reconstruct a0 to qualify the decomposition (relative).
        let back = vectors.multiply(&Dense { rows: m, cols: 1, data: weights.clone() }, lap)?;
        let scale = a0.iter().fold(0_f64, |s, z| s.hypot(cabs(*z))).max(f64::MIN_POSITIVE);
        let defect = back.data.iter().zip(&a0).fold(0_f64, |s, (x, y)| s.hypot(cabs(csub(*x, *y))));
        let amplification = weights.iter().map(|z| cabs(*z)).sum::<f64>() / scale;
        if defect > 1e-8 * scale || !amplification.is_finite() || amplification > 1e8 {
            return Err(format!("Reduced generator eigenbasis is unresolved (defect {defect:e}, amplification {amplification:e})"));
        }
        Ok(Self { exponents, vectors, weights })
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn validate(view: &CandidateView<'_>) -> Result<(usize, usize), String> {
    let rows = view.map.dimension();
    let m = view.columns;
    if m == 0 || view.q.len() != rows * m || view.eq.len() != rows * m || view.fq.len() != rows * m {
        return Err("Candidate REAL Q/EQ/FQ must be complete [rows, columns]".into());
    }
    if view.times.is_empty() || view.times.iter().any(|t| !t.is_finite() || *t < 0.) {
        return Err("Estimator needs nonnegative finite output times".into());
    }
    if view.q.iter().chain(view.eq).chain(view.fq).any(|x| !x.is_finite()) {
        return Err("Nonfinite candidate images".into());
    }
    Ok((rows, m))
}

/// Modal columns of the candidate in REAL coordinates and pushed to the
/// complete signed field: `sum_l X[:, l] P[l, k]` for a complex block `P`
/// (m x width), returned row-major `[mode * d + entry, width]`. Real and
/// imaginary parts are pushed separately (the map is real-linear), so only
/// one block of columns is ever held.
fn push_modal(map: &RealFourierMap, terms: &[(&[f64], &Dense)], rows: usize, m: usize, lap: &Lapack) -> Result<Vec<C>, String> {
    let width = terms[0].1.cols;
    let mut re = vec![0.; rows * width];
    let mut im = vec![0.; rows * width];
    for (real, block) in terms {
        let block_re: Vec<f64> = block.data.iter().map(|z| z.re).collect();
        let block_im: Vec<f64> = block.data.iter().map(|z| z.im).collect();
        for (target, part) in [(&mut re, &block_re), (&mut im, &block_im)] {
            let product = lap.multiply(real, part, rows, width, m);
            for (t, v) in target.iter_mut().zip(&product) {
                *t += v;
            }
        }
    }
    let mut out = vec![ZERO; rows * width];
    let mut part = vec![ZERO; rows * width];
    for (values, imaginary) in [(&re, false), (&im, true)] {
        let mut row_major = vec![0.; rows * width];
        for k in 0..width {
            for i in 0..rows {
                row_major[i * width + k] = values[i + k * rows];
            }
        }
        map.push_into(&row_major, width, &mut part)?;
        for (o, v) in out.iter_mut().zip(&part) {
            *o = if imaginary { cadd(*o, complex(-v.im, v.re)) } else { cadd(*o, *v) };
        }
    }
    Ok(out)
}

/// Modal weights `V diag(w)` of the reduced candidate.
fn modal_weights(small: &SmallSource) -> Dense {
    let m = small.weights.len();
    let mut out = small.vectors.clone();
    for j in 0..m {
        for i in 0..m {
            let value = cmul(out.at(i, j), small.weights[j]);
            out.set(i, j, value);
        }
    }
    out
}

/// State and group maxima of complex `[time, state]` populations.
fn maxima(values: &[C], n: usize, groups: &[PopulationGroup]) -> (f64, f64) {
    let mut state = 0_f64;
    let mut group = 0_f64;
    for row in values.chunks_exact(n) {
        for z in row {
            state = state.max(z.re.abs());
        }
        for g in groups {
            let sum: f64 = g.indices.iter().filter(|&&i| i < n).map(|&i| row[i].re).sum();
            group = group.max(sum.abs());
        }
    }
    (state, group)
}

/// Charge for the slow charges that only weak couplings reach (module docs of
/// `slow_dual.rs`): over the horizon `T` their ground blocks gain at most
/// `T ||K||_F ||e_P||` from the slow error `e_P`, and a population read from
/// them is at most that norm (the readout weights `B_ia conj(B_ib)` have unit
/// norm). The slow error's norm is taken as `g` times the largest population
/// error (ground coherences of the populations' size). The charge vanishes
/// with the weak couplings, so the estimate is continuous in them.
fn unrepresented_charge(dual: &SlowDual, horizon: f64, maximum_state: f64) -> f64 {
    horizon * dual.diagnostics.unrepresented_rate * dual.g as f64 * maximum_state
}

fn finish_estimate(
    dual: &SlowDual,
    values: &[C],
    times: &[f64],
    groups: &[PopulationGroup],
    began: Instant,
) -> Result<PopulationErrorEstimate, String> {
    let n = dual.n;
    let nt = times.len();
    if values.len() != n * nt {
        return Err("Population estimate shape mismatch".into());
    }
    for group in groups {
        if group.indices.iter().any(|&i| i >= n) {
            return Err(format!("Population group {} indexes outside the state range", group.name));
        }
    }
    let state_error: Vec<f64> = values.iter().map(|z| z.re).collect();
    let maximum_imaginary = values.iter().map(|z| z.im.abs()).fold(0_f64, f64::max);
    let mut group_error = vec![0.; nt * groups.len()];
    for t in 0..nt {
        for (k, group) in groups.iter().enumerate() {
            group_error[t * groups.len() + k] = group.indices.iter().map(|&i| state_error[t * n + i]).sum();
        }
    }
    let maximum_state = state_error.iter().map(|x| x.abs()).fold(0_f64, f64::max);
    let maximum_group = group_error.iter().map(|x| x.abs()).fold(0_f64, f64::max);
    if !maximum_state.is_finite() || !maximum_group.is_finite() {
        return Err("Nonfinite population error estimate".into());
    }
    let horizon = times.iter().copied().fold(0_f64, f64::max);
    let margin = unrepresented_charge(dual, horizon, maximum_state);
    Ok(PopulationErrorEstimate {
        times: times.to_vec(),
        states: n,
        state_error,
        group_names: groups.iter().map(|g| g.name.clone()).collect(),
        group_error,
        maximum_state,
        maximum_group,
        maximum: maximum_state.max(maximum_group) + margin,
        margin,
        unresolved: None,
        maximum_imaginary,
        elapsed_ms: began.elapsed().as_secs_f64() * 1000.,
    })
}

/// The trivial conservative estimate (`|error| <= 1` on the probability
/// scale) for a candidate whose reduced dynamics could not be resolved: the
/// controller then refines instead of failing.
fn unresolved_estimate(dual: &SlowDual, times: &[f64], groups: &[PopulationGroup], reason: String, began: Instant) -> PopulationErrorEstimate {
    let (n, nt) = (dual.n, times.len());
    PopulationErrorEstimate {
        times: times.to_vec(),
        states: n,
        state_error: vec![0.; nt * n],
        group_names: groups.iter().map(|g| g.name.clone()).collect(),
        group_error: vec![0.; nt * groups.len()],
        maximum_state: 0.,
        maximum_group: 0.,
        maximum: 1.,
        margin: 1.,
        unresolved: Some(reason),
        maximum_imaginary: 0.,
        elapsed_ms: began.elapsed().as_secs_f64() * 1000.,
    }
}

/// Frame-entry columns (row `a*n+b`) as `n x n` blocks.
fn column_blocks(columns: &Dense, n: usize) -> Vec<Dense> {
    (0..columns.cols)
        .map(|j| {
            let mut x = Dense::zeros(n, n);
            for entry in 0..n * n {
                x.set(entry / n, entry % n, columns.at(entry, j));
            }
            x
        })
        .collect()
}

impl SlowDual {
    /// Basis (in-domain) estimate from the retained residual rows only.
    pub fn basis_estimate(&self, view: &CandidateView<'_>, groups: &[PopulationGroup]) -> Result<PopulationErrorEstimate, String> {
        self.basis_estimate_with(view, groups, BasisOptions::default())
    }

    /// Fast response at one charge `q`:
    /// `eta' = (N - i nu_q) eta + Rq_hat a(t)`, `eta(0) = 0`, where `Rq_hat`
    /// is the residual (or returned field) at `q` without its ground block, as
    /// modal columns. In the dressed eigen-coordinates every entry is
    /// `sum_j c_ikj (exp(beta_j t) - exp(lambda_ik t)) / (beta_j - lambda_ik)`,
    /// `lambda_ik = a_i + b_k - i nu_q`. The populations of `eta` are read out
    /// directly (phase `exp(i nu_q t)`); its recycling `R_0 eta_EE` (the only
    /// part of `L_0 - N` at the same charge) is a `J_eff` source for
    /// `target = (slot, shift)`: the slot of `q` with shift 0 when `q` is slow,
    /// otherwise the central slot with the physical phase (`shift = nu_q`).
    fn add_fast_response(&self, series: &mut Series, target: (usize, f64), nu: f64, start: usize, coefficients: &Dense) -> Result<(), String> {
        let dressed = match &self.dressed {
            Some(d) => d,
            None => return Ok(()),
        };
        let lap = &self.lapack;
        let n = self.n;
        let n2 = n * n;
        let m = coefficients.cols;
        let blocks = column_blocks(coefficients, n);
        let refs: Vec<&Dense> = blocks.iter().collect();
        let eigen = sandwich_all(&dressed.left_inverse, &dressed.right, &refs, lap)?;
        let lambda: Vec<C> = (0..n2)
            .map(|ik| csub(cadd(dressed.left_values[ik / n], dressed.right_values[ik % n]), complex(0., nu)))
            .collect();
        let mut forced = Dense::zeros(n2, m);
        let mut homogeneous = vec![ZERO; n2];
        let mut near: Vec<(usize, usize, C)> = Vec::new();
        for (j, xh) in eigen.iter().enumerate() {
            let beta = series.beta[start + j];
            for i in 0..n {
                for k in 0..n {
                    let ik = i * n + k;
                    let c = xh.at(i, k);
                    if c.re == 0. && c.im == 0. {
                        continue;
                    }
                    let delta = csub(beta, lambda[ik]);
                    if cabs(delta) * series.t_max > 1. {
                        let f = cdiv(c, delta);
                        forced.set(ik, j, f);
                        homogeneous[ik] = csub(homogeneous[ik], f);
                    } else {
                        near.push((ik, j, c));
                    }
                }
            }
        }
        // Direct populations: forced families at the physical phase, the
        // homogeneous part exp((lambda + i nu) t) = exp((a_i + b_k) t).
        series.add_direct(nu, start, &dressed.readout.multiply(&forced, lap)?);
        series.add_dressed_direct(&homogeneous);
        for &(ik, j, c) in &near {
            let p: Vec<C> = (0..dressed.readout.rows).map(|s| cmul(dressed.readout.at(s, ik), c)).collect();
            series.direct_near.push((p, cadd(series.beta[start + j], complex(0., nu)), cadd(lambda[ik], complex(0., nu))));
        }
        // Recycling into the target slot (with the routing phase).
        let (slot, shift) = target;
        let recycling = &dressed.modal_recycling[slot];
        series.add_slow(shift, start, &recycling.multiply(&forced, lap)?);
        series.add_dressed_slow(slot, shift - nu, &homogeneous);
        for &(ik, j, c) in &near {
            let v: Vec<C> = (0..recycling.rows).map(|r| cmul(recycling.at(r, ik), c)).collect();
            if v.iter().any(|z| z.re != 0. || z.im != 0.) {
                series.add_pair(cadd(series.beta[start + j], complex(0., shift)), cadd(lambda[ik], complex(0., shift)), v);
            }
        }
        Ok(())
    }

    /// Slow part of a physical initial error `X0` (central charge): its ground
    /// block becomes the initial slow error. Returns the remaining fast layer
    /// `X0_Q - S(P X0)`: the non-slow part minus the pattern the Markov readout
    /// would already attribute to the slow part (it must relax, not appear
    /// instantly).
    fn add_slow_initial(&self, series: &mut Series, initial: &Dense) -> Result<Dense, String> {
        let (n, g) = (self.n, self.g);
        let mut ground = Dense::zeros(g * g, 1);
        for a in 0..g {
            for b in 0..g {
                ground.set(a * g + b, 0, initial.at(a, b));
            }
        }
        let mut layer = initial.clone();
        for a in 0..g {
            for b in 0..g {
                layer.set(a, b, ZERO);
            }
        }
        if ground.frobenius() > 0. {
            series.add_initial(&self.to_modal(0, &ground)?.data);
            for (column, slaved) in self.slaved_central.iter().enumerate().take(g * g) {
                let amount = ground.data[column];
                if amount.re == 0. && amount.im == 0. {
                    continue;
                }
                for k in 0..n * n {
                    layer.data[k] = csub(layer.data[k], cmul(amount, slaved.data[k]));
                }
            }
        }
        Ok(layer)
    }

    /// Exact fast initial layer at the central charge: `Xhat(t) = Xhat(0)
    /// exp(lambda t)` in the dressed no-jump eigen-coordinates; populations are
    /// read out directly and the recycling `R_0 X_EE` feeds the central slot.
    fn add_fast_initial(&self, series: &mut Series, initial: &Dense) -> Result<(), String> {
        let dressed = match &self.dressed {
            Some(d) => d,
            None => return Ok(()),
        };
        let n = self.n;
        let xh = dressed.to_eigen(initial, &self.lapack)?;
        let amplitudes: Vec<C> = (0..n * n).map(|ik| xh.at(ik / n, ik % n)).collect();
        series.add_dressed_direct(&amplitudes);
        series.add_dressed_slow(0, 0., &amplitudes);
        Ok(())
    }

    /// Basis estimate with explicit options (calibration and diagnostics).
    pub fn basis_estimate_with(&self, view: &CandidateView<'_>, groups: &[PopulationGroup], options: BasisOptions) -> Result<PopulationErrorEstimate, String> {
        let began = Instant::now();
        let (rows, m) = validate(view)?;
        let lap = &self.lapack;
        let n = self.n;
        let g = self.g;
        if view.map.physical_dimension() != n {
            return Err("Candidate map dimension differs from the slow dual".into());
        }
        let small = match SmallSource::new(lap, view.generator, view.initial, m) {
            Ok(small) => small,
            Err(reason) => return Ok(unresolved_estimate(self, view.times, groups, reason, began)),
        };
        let grid = TimeGrid::new(view.times);
        let phases = &self.owner.hilbert.source.phases;
        let d = n * n;
        let mut series = Series::new(&small.exponents, grid.maximum);
        let weights = modal_weights(&small);
        // Modal residual columns, one block at a time: the residual is linear
        // in the candidate columns and (F Q - E Q G) V diag(w) =
        // F Q V diag(w) - E Q V diag(w beta), so column k evolves exactly as
        // exp(beta_k t) and blocks never mix.
        let mut start = 0;
        while start < m {
            let width = BASIS_BLOCK.min(m - start);
            let block = weights.block(0..m, start..start + width);
            let mut scaled = block.clone();
            for k in 0..width {
                for l in 0..m {
                    let value = cmul(scaled.at(l, k), small.exponents[start + k]);
                    scaled.set(l, k, csub(ZERO, value));
                }
            }
            let signed = push_modal(view.map, &[(view.fq, &block), (view.eq, &scaled)], rows, m, lap)?;
            // Non-ground residuals at fast charges, slaved together.
            let mut slaving: Vec<(Charge, Dense)> = Vec::new();
            for (mode, charge) in view.map.modes().iter().enumerate() {
                let slow_slot = self.slot_of(charge);
                let nu = phases.frequency(charge).map(|f| f.hi + f.lo)?;
                // Ground block: slot of the charge, or central slot with its phase.
                let (slot, shift) = slow_slot.map_or((0, nu), |s| (s, 0.));
                let mut ground = Dense::zeros(g * g, width);
                let mut field = Dense::zeros(d, width);
                for entry in 0..d {
                    let (a, b) = (entry / n, entry % n);
                    let row = (mode * d + entry) * width;
                    for k in 0..width {
                        let v = signed[row + k];
                        if a < g && b < g {
                            ground.set(a * g + b, k, v);
                        } else {
                            field.set(entry, k, v);
                        }
                    }
                }
                if options.fast_response && self.dressed.is_some() && field.frobenius() > 0. {
                    // Non-ground part: exact dressed fast response at this charge.
                    self.add_fast_response(&mut series, (slot, shift), nu, start, &field)?;
                }
                if options.slave_fast_charges && slow_slot.is_none() && field.frobenius() > 0. {
                    slaving.push((charge.clone(), field));
                }
                if ground.frobenius() > 0. {
                    series.add_slow(shift, start, &self.to_modal(slot, &ground)?);
                }
            }
            if !slaving.is_empty() {
                let fields: Vec<(Charge, &Dense)> = slaving.iter().map(|(k, f)| (k.clone(), f)).collect();
                for (target_slot, returned) in self.slaved_returns_all(&fields, width)? {
                    series.add_slow(0., start, &self.to_modal(target_slot, &returned)?);
                }
            }
            start += width;
        }
        let populations = self.evaluate(&series, &grid)?;
        finish_estimate(self, &populations, view.times, groups, began)
    }

    /// Halo modes of a retained domain: output modes reached from `input` by
    /// one complete pencil application (the same reverse-dependency graph as
    /// `OwnedPrepared::pencil`), excluding the retained modes themselves.
    pub fn halo_modes(prepared: &OwnedPrepared, input: &[Charge]) -> Result<Vec<Charge>, String> {
        let keys = |i: usize| -> Modes { prepared.families[i].keys().cloned().collect() };
        let (s, f, gg, k, r, lgg, log) = (keys(0), keys(1), keys(2), keys(3), keys(4), keys(5), keys(6));
        let axes = prepared.primitives.hilbert.source.phases.axes.len();
        let input: Modes = input.iter().cloned().collect();
        let mut j_support = Modes::from([vec![0; axes]]);
        j_support.extend(s.iter().cloned());
        j_support.extend(combinations(&s, &s, false)?);
        let forward_j = combinations(&input, &j_support, false)?;
        let z = combinations(&forward_j, &k, false)?;
        let mut output = forward_j.clone();
        output.extend(z.iter().cloned());
        for family in [&f, &gg, &r] {
            output.extend(combinations(&forward_j, family, false)?);
        }
        for family in [&lgg, &log] {
            output.extend(combinations(&z, family, false)?);
        }
        Ok(output.into_iter().filter(|q| !input.contains(q)).collect())
    }

    /// Fourier-truncation (halo) estimate and per-halo-pair indicators.
    ///
    /// Every quantity of the halo estimate is linear in the candidate's modal
    /// columns `c_j exp(beta_j t)` and treats them independently, so the halo
    /// is processed in blocks of modal columns: the pencil is applied once per
    /// block to the modal inputs `Q V diag(w)` for ALL halo modes (one plan, its
    /// intermediate modes computed once), giving the modal residual columns
    /// `F (Q v_j) - beta_j E (Q v_j)` directly, and only one block of halo rows
    /// is ever held. The total is one series (dressed fast responses of all
    /// returns summed per target charge and block, one initial layer) evaluated
    /// once; each pair's indicator uses its own residual response, returns and
    /// slow boundary on a subsample of the output times.
    pub fn halo_estimate(
        &self,
        prepared: &OwnedPrepared,
        view: &CandidateView<'_>,
        groups: &[PopulationGroup],
        options: HaloOptions,
    ) -> Result<HaloEstimate, String> {
        let began = Instant::now();
        if !self.belongs_to(prepared) {
            return Err("Slow dual belongs to another request".into());
        }
        let (rows, m) = validate(view)?;
        if options.maximum_rhs == 0 {
            return Err("Halo block width must be positive".into());
        }
        let lap = &self.lapack;
        let (n, g) = (self.n, self.g);
        let d = n * n;
        let floor = options.resonance_floor.unwrap_or(self.diagnostics.optical_decay_floor);
        let phases = &self.owner.hilbert.source.phases;
        let frequency = |q: &[i32]| -> Result<f64, String> { phases.frequency(q).map(|f| f.hi + f.lo) };
        let small = match SmallSource::new(lap, view.generator, view.initial, m) {
            Ok(small) => small,
            Err(reason) => {
                let estimate = unresolved_estimate(self, view.times, groups, reason, began);
                return Ok(HaloEstimate { estimate, modes: Vec::new(), halo_modes: 0, near_resonant_terms: 0, unrouted_resonant_returns: 0, pencil_ms: 0., block_columns: 0 });
            }
        };
        let grid = TimeGrid::new(view.times);
        // Indicator samples: a uniform subset of the output times.
        let stride = view.times.len().div_ceil(INDICATOR_TIMES).max(1);
        let mut indicator_times: Vec<f64> = view.times.iter().step_by(stride).copied().collect();
        if (view.times.len() - 1) % stride != 0 {
            indicator_times.push(view.times[view.times.len() - 1]);
        }
        let indicator_grid = TimeGrid::new(&indicator_times);
        let retained: Vec<Charge> = view.map.modes().to_vec();
        let retained_set: Modes = retained.iter().cloned().collect();
        let halo = Self::halo_modes(prepared, &retained)?;
        // Evaluated completely: the halo modes that return into the retained
        // domain (or a slot) through a significant coupling in one hop, and
        // the slow ones (the slot criterion), whose ground block is slow error
        // content: its populations accumulate whether or not it returns in
        // one hop. A fast mode whose returns are all weak couplings (module
        // docs of `slow_dual.rs`) feeds the slow error only at their strength:
        // its returns are evaluated exactly, while its own populations,
        // boundary and same-charge recycling are left out, as for a fast mode
        // that does not return at all, so that the estimate is continuous as
        // the weak couplings vanish. Other fast modes are left out (their
        // effect needs two hops).
        let returns = |q: &Charge, significant: bool| -> bool {
            self.frame.terms.iter().filter(|term| term.significant == significant).any(|term| {
                add_charge(q, &term.charge, false).is_ok_and(|k| retained_set.contains(&k) || self.slot_of(&k).is_some())
            })
        };
        let mut kept = Vec::with_capacity(halo.len());
        let mut weak_only = Vec::with_capacity(halo.len());
        for q in halo {
            let complete = options.all_halo_modes || returns(&q, true) || self.is_slow(frequency(&q)?);
            if complete || returns(&q, false) {
                weak_only.push(!complete);
                kept.push(q);
            }
        }
        let halo = kept;
        let canonical = |q: &Charge| -> Charge {
            let negative: Charge = q.iter().map(|x| -x).collect();
            if q.iter().copied().find(|&v| v != 0).is_none_or(|v| v > 0) { q.clone() } else { negative }
        };
        let mut pair_index: BTreeMap<Charge, usize> = BTreeMap::new();
        for (q, _) in halo.iter().zip(&weak_only).filter(|(_, &weak)| !weak) {
            let next = pair_index.len();
            pair_index.entry(canonical(q)).or_insert(next);
        }
        let pairs = pair_index.len();
        // Indicators: each pair's block contributions on the indicator grid
        // and its homogeneous slow amplitudes (the common exp(J t) part).
        let mut pair_values: Vec<Vec<C>> = vec![vec![ZERO; indicator_times.len() * n]; pairs];
        let mut pair_homogeneous: Vec<Vec<C>> = vec![vec![ZERO; self.dimension()]; pairs];
        let mut pair_boundary: Vec<Dense> = vec![Dense::zeros(n, n); pairs];
        let mut pair_residual: Vec<f64> = vec![0.; pairs];
        let mut total = Series::new(&small.exponents, grid.maximum);
        let mut boundary_total = Dense::zeros(n, n);
        let mut near_resonant_terms = 0;
        let mut unrouted_resonant_returns = 0;
        let mut pencil_ms = 0.;
        let halo_frequency: Vec<f64> = halo.iter().map(|q| frequency(q)).collect::<Result<_, _>>()?;

        // Modal inputs Q V diag(w), one block at a time.
        let weights = modal_weights(&small);
        // Returning hops of every halo mode, resolved once: (term, target
        // slot and shift, index of the target charge's fast field or None for
        // the same-charge return).
        let mut fast_targets: Vec<(Charge, (usize, f64), f64)> = Vec::new();
        let mut fast_index: BTreeMap<Charge, usize> = BTreeMap::new();
        let mut routes: Vec<Vec<(usize, (usize, f64), Option<usize>)>> = Vec::with_capacity(halo.len());
        for (q, &weak) in halo.iter().zip(&weak_only) {
            let mut list = Vec::new();
            for (t, term) in self.frame.terms.iter().enumerate() {
                let same_charge = term.charge.iter().all(|&v| v == 0);
                let k = add_charge(q, &term.charge, false)?;
                if (same_charge && weak) || (!same_charge && !retained_set.contains(&k) && self.slot_of(&k).is_none()) {
                    continue;
                }
                let target = self.target(&k, &frequency)?;
                let fast = if same_charge {
                    None
                } else {
                    let next = fast_targets.len();
                    let index = *fast_index.entry(k.clone()).or_insert(next);
                    if index == next {
                        fast_targets.push((k.clone(), target, frequency(&k)?));
                    }
                    Some(index)
                };
                list.push((t, target, fast));
            }
            routes.push(list);
        }
        let width_max = options.maximum_rhs.min(m);
        let out_rows = halo.len() * d;
        let mut plan = if halo.is_empty() { None } else { Some(prepared.pencil(&retained, &halo, width_max)?) };
        let mut e_out = vec![ZERO; out_rows * width_max];
        let mut f_out = vec![ZERO; out_rows * width_max];
        let mut start = 0;
        while start < m && plan.is_some() {
            let width = width_max.min(m - start);
            let beta = &small.exponents[start..start + width];
            let input = push_modal(view.map, &[(view.q, &weights.block(0..m, start..start + width))], rows, m, lap)?;
            let pencil_began = Instant::now();
            plan.as_mut().expect("plan").apply_into(&input, width, &mut e_out[..out_rows * width], &mut f_out[..out_rows * width])?;
            pencil_ms += pencil_began.elapsed().as_secs_f64() * 1000.;
            // Non-ground returns of this block, summed per target charge, and
            // ground returns summed per (pair, slot, shift) before their
            // transformation to J_eff coordinates.
            let mut fast_fields: Vec<Option<Dense>> = vec![None; fast_targets.len()];
            // This block's contribution of every pair (block columns only).
            let mut pair_series: BTreeMap<usize, Series> = BTreeMap::new();
            // Keyed by (pair, slot, shift); no pair for weakly returning modes.
            let mut ground_returns: BTreeMap<(Option<usize>, usize, i64), (f64, Dense)> = BTreeMap::new();

            for (local, q) in halo.iter().enumerate() {
                let pair = (!weak_only[local]).then(|| pair_index[&canonical(q)]);
                let nu_q = halo_frequency[local];
                // Modal residual columns R_q[:, k] = F w_k - beta_k E w_k.
                let mut residual = Dense::zeros(d, width);
                for entry in 0..d {
                    let row = (local * d + entry) * width;
                    for k in 0..width {
                        residual.set(entry, k, csub(f_out[row + k], cmul(beta[k], e_out[row + k])));
                    }
                }
                let norm = residual.frobenius();
                if norm == 0. {
                    continue;
                }
                // The pair's part of this block (indicators); weakly returning
                // modes enter the total only.
                let mut series = pair.map(|pair| {
                    pair_residual[pair] = pair_residual[pair].hypot(norm);
                    pair_series.entry(pair).or_insert_with(|| Series::new(beta, grid.maximum))
                });
                // A halo charge that is itself a slow slot: its ground block is
                // slow content, a direct J_eff source (as in the basis estimate);
                // only the non-ground rest gets the fast halo response below. A
                // slot returns through the central term, so it has a pair.
                if let (Some(own), Some(series)) = (self.slot_of(q), series.as_mut()) {
                    let mut block = Dense::zeros(g * g, width);
                    for a in 0..g {
                        for b in 0..g {
                            for k in 0..width {
                                block.set(a * g + b, k, residual.at(a * n + b, k));
                                residual.set(a * n + b, k, ZERO);
                            }
                        }
                    }
                    if block.frobenius() > 0. {
                        let modal = self.to_modal(own, &block)?;
                        total.add_slow(0., start, &modal);
                        series.add_slow(0., 0, &modal);
                    }
                }

                // --- Envelope response eta_q ---------------------------------
                let (particular, resonant) = self.halo_response(&residual, beta, nu_q, floor)?;
                near_resonant_terms += resonant.len();

                if let (Some(pair), Some(series)) = (pair, series.as_mut()) {
                    // --- Direct populations of eta (physical phase exp(i nu_q t))
                    let mut populations = Dense::zeros(n, width);
                    for (k, x) in particular.iter().enumerate() {
                        for (s, v) in self.frame_populations(x).into_iter().enumerate() {
                            populations.set(s, k, v);
                        }
                    }
                    total.add_direct(nu_q, start, &populations);
                    series.add_direct(nu_q, 0, &populations);
                    for term in &resonant {
                        let p = self.frame_populations(&term.unit);
                        let first = cadd(beta[term.column], complex(0., nu_q));
                        let second = cadd(term.second, complex(0., nu_q));
                        total.add_direct_pair(first, second, p.clone());
                        series.add_direct_pair(first, second, p);
                    }

                    // --- Initial boundary -eta(0) (physical matrix at t = 0) --
                    for x in &particular {
                        for k in 0..d {
                            boundary_total.data[k] = csub(boundary_total.data[k], x.data[k]);
                            pair_boundary[pair].data[k] = csub(pair_boundary[pair].data[k], x.data[k]);
                        }
                    }
                }

                // --- One-hop return ------------------------------------------
                // p = 0 (same charge): the ground block of (L_0 - N) eta, which
                // the no-jump reference left out; it feeds slot q or, for a
                // fast q, the central slot with phase exp(i nu_q t).
                // p != 0 into a retained or slow charge k: the complete field
                // L_p eta; its ground block is a slow source, its non-ground
                // block gets the exact dressed fast response at k (summed over
                // the halo, total only).
                let refs: Vec<&Dense> = particular.iter().collect();
                for &(t, (slot, shift), fast) in &routes[local] {
                    let term = &self.frame.terms[t];
                    let same_charge = fast.is_none();
                    let k = add_charge(q, &term.charge, false)?;
                    let ground = &mut ground_returns
                        .entry((pair, slot, frequency_key(shift)))
                        .or_insert_with(|| (shift, Dense::zeros(g * g, width)))
                        .1;
                    if let Some(index) = fast {
                        let images = self.frame.apply_all(term, &refs, lap)?;
                        let field = fast_fields[index].get_or_insert_with(|| Dense::zeros(d, width));
                        for (column, image) in images.iter().enumerate() {
                            for b in 0..n {
                                for a in 0..n {
                                    let v = image.at(a, b);
                                    if a < g && b < g {
                                        ground.add_at(a * g + b, column, v);
                                    } else {
                                        field.add_at(a * n + b, column, v);
                                    }
                                }
                            }
                        }
                        if self.slot_of(&k).is_none() && options.slave_fast_returns {
                            let mut field = Dense::zeros(d, width);
                            for (column, image) in images.iter().enumerate() {
                                for b in 0..n {
                                    for a in 0..n {
                                        field.set(a * n + b, column, image.at(a, b));
                                    }
                                }
                            }
                            for (target_slot, returned) in self.slaved_returns(&k, &field)? {
                                let modal = self.to_modal(target_slot, &returned)?;
                                total.add_slow(0., start, &modal);
                                if let Some(series) = series.as_mut() {
                                    series.add_slow(0., 0, &modal);
                                }
                            }
                        }
                    } else {
                        let returned = self.same_charge_ground_returns(&refs)?;
                        for (entry, v) in ground.data.iter_mut().zip(&returned.data) {
                            *entry = cadd(*entry, *v);
                        }
                    }
                    // Near-resonant terms: zero-initial pair sources. Their
                    // ground block returns exactly; a non-ground return is
                    // slaved when k is fast (optional) and otherwise counted (it
                    // needs a pair-source fast response; rare: resonance
                    // requires a strongly damped candidate mode). Their images
                    // are formed in one batch per route.
                    let units: Vec<&Dense> = resonant.iter().map(|r| &r.unit).collect();
                    let images: Vec<Dense> = if units.is_empty() {
                        Vec::new()
                    } else if same_charge {
                        let returned = self.same_charge_ground_returns(&units)?;
                        (0..units.len())
                            .map(|column| {
                                let mut out = Dense::zeros(n, n);
                                for row in 0..g * g {
                                    out.set(row / g, row % g, returned.at(row, column));
                                }
                                out
                            })
                            .collect()
                    } else {
                        self.frame.apply_all(term, &units, lap)?
                    };
                    for (resonant_term, image) in resonant.iter().zip(&images) {
                        let column = resonant_term.column;
                        let first_node = cadd(beta[column], complex(0., shift));
                        let second_node = cadd(resonant_term.second, complex(0., shift));
                        let mut ground = Dense::zeros(g * g, 1);
                        for ga in 0..g {
                            for gb in 0..g {
                                ground.set(ga * g + gb, 0, image.at(ga, gb));
                            }
                        }
                        if ground.frobenius() > 0. {
                            let modal = self.to_modal(slot, &ground)?;
                            total.add_pair(first_node, second_node, modal.data.clone());
                            if let Some(series) = series.as_mut() {
                                series.add_pair(first_node, second_node, modal.data);
                            }
                        }
                        if same_charge {
                            continue;
                        }
                        let mut rest = Dense::zeros(d, 1);
                        for ea in 0..n {
                            for eb in 0..n {
                                if ea >= g || eb >= g {
                                    rest.set(ea * n + eb, 0, image.at(ea, eb));
                                }
                            }
                        }
                        if rest.frobenius() == 0. {
                            continue;
                        }
                        if self.slot_of(&k).is_some() {
                            unrouted_resonant_returns += 1;
                            continue;
                        }
                        if !options.slave_fast_returns {
                            continue;
                        }
                        for (target_slot, returned) in self.slaved_returns(&k, &rest)? {
                            let modal = self.to_modal(target_slot, &returned)?;
                            total.add_pair(beta[column], resonant_term.second, modal.data.clone());
                            if let Some(series) = series.as_mut() {
                                series.add_pair(beta[column], resonant_term.second, modal.data);
                            }
                        }
                    }
                }
            }
            // --- Ground returns into J_eff, per pair -------------------------
            for ((pair, slot, _), (shift, ground)) in &ground_returns {
                if ground.frobenius() > 0. {
                    let modal = self.to_modal(*slot, ground)?;
                    total.add_slow(*shift, start, &modal);
                    if let Some(pair) = pair {
                        pair_series.entry(*pair).or_insert_with(|| Series::new(beta, grid.maximum)).add_slow(*shift, 0, &modal);
                    }
                }
            }
            // --- Exact dressed fast responses of this block's returns -------
            for (field, (_, target, nu_k)) in fast_fields.iter().zip(&fast_targets) {
                if let Some(field) = field {
                    self.add_fast_response(&mut total, *target, *nu_k, start, field)?;
                }
            }
            for (pair, series) in pair_series {
                let values = self.evaluate_parts(&series, &indicator_grid, Some(&mut pair_homogeneous[pair]))?;
                for (a, b) in pair_values[pair].iter_mut().zip(&values) {
                    *a = cadd(*a, *b);
                }
            }
            start += width;
        }
        drop(plan);
        drop(e_out);
        drop(f_out);

        // --- Indicators: each pair's response, returns and slow boundary ----
        let mut charges: Vec<Charge> = vec![Vec::new(); pairs];
        for (charge, &index) in &pair_index {
            charges[index] = charge.clone();
        }
        let mut modes = Vec::with_capacity(pairs);
        for pair in 0..pairs {
            let mut series = Series::new(&small.exponents, grid.maximum);
            series.add_initial(&pair_homogeneous[pair]);
            if pair_boundary[pair].frobenius() > 0. {
                self.add_slow_initial(&mut series, &pair_boundary[pair])?;
            }
            let mut values = self.evaluate(&series, &indicator_grid)?;
            for (a, b) in values.iter_mut().zip(&pair_values[pair]) {
                *a = cadd(*a, *b);
            }
            let (state, group) = maxima(&values, n, groups);
            modes.push(HaloModeIndicator { charge: charges[pair].clone(), indicator: state.max(group), residual_norm: pair_residual[pair] });
        }
        modes.sort_by(|a, b| b.indicator.total_cmp(&a.indicator));

        // --- Initial boundary: slow part plus exact fast initial layer ------
        if boundary_total.frobenius() > 0. {
            let layer = self.add_slow_initial(&mut total, &boundary_total)?;
            self.add_fast_initial(&mut total, &layer)?;
        }
        let values = self.evaluate(&total, &grid)?;
        let estimate = finish_estimate(self, &values, view.times, groups, began)?;
        Ok(HaloEstimate { estimate, modes, halo_modes: halo.len(), near_resonant_terms, unrouted_resonant_returns, pencil_ms, block_columns: width_max })
    }

    /// Particular envelope response of one halo charge `q` to the modal source
    /// columns `c_j exp(beta_j t)` (frame entries a*n+b in rows):
    /// `eta_j = (beta_j + i nu_q - N)^{-1} c_j`, with `N` the complete charge-0
    /// no-jump operator, diagonal in the dressed eigen-coordinates
    /// (`lambda_ik = a_i + b_k`). Components with `|beta_j + i nu_q - lambda| <
    /// floor` are returned separately as zero-initial terms
    /// `c e_t[beta_j, lambda - i nu_q]` (entire form). Without a dressed frame
    /// the diagonal of `N` is the reference.
    fn halo_response(&self, coefficients: &Dense, exponents: &[C], nu_q: f64, floor: f64) -> Result<(Vec<Dense>, Vec<ResonantTerm>), String> {
        let n = self.n;
        let lap = &self.lapack;
        let mut resonant = Vec::new();
        let blocks = column_blocks(coefficients, n);
        // Componentwise particular response against the reference eigenvalue
        // `lambda` (dressed: a_i + b_k; fallback: the diagonal d_ab); near
        // resonant components are split off as zero-initial terms whose frame
        // matrix `unit` is produced by `to_frame`.
        let mut solve = |j: usize, x: &mut Dense, lambda: &dyn Fn(usize, usize) -> C, to_frame: &dyn Fn(usize, usize, C) -> Dense| -> Result<(), String> {
            let shift = cadd(exponents[j], complex(0., nu_q));
            for b in 0..n {
                for a in 0..n {
                    let c = x.at(a, b);
                    if c.re == 0. && c.im == 0. {
                        continue;
                    }
                    let l = lambda(a, b);
                    let denominator = csub(shift, l);
                    if cabs(denominator) >= floor {
                        x.set(a, b, cdiv(c, denominator));
                    } else {
                        resonant.push(ResonantTerm { column: j, unit: to_frame(a, b, c), second: csub(l, complex(0., nu_q)) });
                        x.set(a, b, ZERO);
                    }
                }
            }
            Ok(())
        };
        match &self.dressed {
            Some(dressed) => {
                // Every column moves to the dressed eigen-coordinates and back
                // with two products each way.
                let refs: Vec<&Dense> = blocks.iter().collect();
                let mut eigen = sandwich_all(&dressed.left_inverse, &dressed.right, &refs, lap)?;
                let lambda = |i: usize, k: usize| cadd(dressed.left_values[i], dressed.right_values[k]);
                // The frame matrix of eigen-coordinate (i, k) is the rank-one
                // c V_L[:, i] V_R^-1[k, :] (a slow halo mode has one per
                // column and ground coherence).
                let to_frame = |i: usize, k: usize, c: C| {
                    let mut unit = Dense::zeros(n, n);
                    for col in 0..n {
                        let right = cmul(c, dressed.right_inverse.at(k, col));
                        for row in 0..n {
                            unit.set(row, col, cmul(dressed.left.at(row, i), right));
                        }
                    }
                    unit
                };
                for (j, xh) in eigen.iter_mut().enumerate() {
                    solve(j, xh, &lambda, &to_frame)?;
                }
                let refs: Vec<&Dense> = eigen.iter().collect();
                let particular = sandwich_all(&dressed.left, &dressed.right_inverse, &refs, lap)?;
                Ok((particular, resonant))
            }
            None => {
                let lambda = |a: usize, b: usize| self.diagonal_reference[a * n + b];
                let to_frame = |a: usize, b: usize, c: C| {
                    let mut unit = Dense::zeros(n, n);
                    unit.set(a, b, c);
                    unit
                };
                let mut particular = blocks;
                for (j, x) in particular.iter_mut().enumerate() {
                    solve(j, x, &lambda, &to_frame)?;
                }
                Ok((particular, resonant))
            }
        }
    }

    /// Ground blocks (`g^2 x m`, row `a*g+b`) of what the halo reference left
    /// out at the same charge: the recycling `R_0 X_EE` for the dressed
    /// no-jump reference, or `(L_0 - D) X` for the diagonal one.
    fn same_charge_ground_returns(&self, xs: &[&Dense]) -> Result<Dense, String> {
        let (n, g) = (self.n, self.g);
        let zero = self.frame.zero_charge();
        let term = match self.frame.term(&zero) {
            Some(t) => t,
            None => return Ok(Dense::zeros(g * g, xs.len())),
        };
        if self.dressed.is_some() {
            return self.frame.recycled(term, xs, &self.lapack);
        }
        let mut out = Dense::zeros(g * g, xs.len());
        for (column, x) in xs.iter().enumerate() {
            let ground = self.frame.apply_ground_output(term, x, &self.lapack)?;
            for a in 0..g {
                for b in 0..g {
                    out.set(a * g + b, column, csub(ground[a * g + b], cmul(self.diagonal_reference[a * n + b], x.at(a, b))));
                }
            }
        }
        Ok(out)
    }

    /// Physical populations `p_i = sum_ab B_ia X_ab conj(B_ib)` of a frame matrix.
    fn frame_populations(&self, x: &Dense) -> Vec<C> {
        let mut out = vec![ZERO; self.n];
        for &(state, a, b, w) in &self.population_readout {
            out[state] = cadd(out[state], cmul(w, x.at(a, b)));
        }
        out
    }

    /// Slow slot receiving a field at charge `k`: its own slot when `k` is slow,
    /// otherwise the central slot with the physical phase `exp(i nu_k t)`.
    fn target(&self, k: &[i32], frequency: &dyn Fn(&[i32]) -> Result<f64, String>) -> Result<(usize, f64), String> {
        Ok(match self.slot_of(k) {
            Some(s) => (s, 0.),
            None => (0, frequency(k)?),
        })
    }
}

/// Combine two estimates (for example basis + halo) into one summary.
pub fn combined_estimate(first: &PopulationErrorEstimate, second: &PopulationErrorEstimate) -> Result<PopulationErrorEstimate, String> {
    if first.times.len() != second.times.len() || first.states != second.states || first.group_names != second.group_names {
        return Err("Estimates use different grids/states/groups".into());
    }
    let state_error: Vec<f64> = first.state_error.iter().zip(&second.state_error).map(|(a, b)| a + b).collect();
    let group_error: Vec<f64> = first.group_error.iter().zip(&second.group_error).map(|(a, b)| a + b).collect();
    let maximum_state = state_error.iter().map(|x| x.abs()).fold(0_f64, f64::max);
    let maximum_group = group_error.iter().map(|x| x.abs()).fold(0_f64, f64::max);
    let margin = first.margin + second.margin;
    Ok(PopulationErrorEstimate {
        times: first.times.clone(),
        states: first.states,
        state_error,
        group_names: first.group_names.clone(),
        group_error,
        maximum_state,
        maximum_group,
        maximum: maximum_state.max(maximum_group) + margin,
        margin,
        unresolved: first.unresolved.clone().or_else(|| second.unresolved.clone()),
        maximum_imaginary: first.maximum_imaginary.max(second.maximum_imaginary),
        elapsed_ms: first.elapsed_ms + second.elapsed_ms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::KernelPolicy;
    use crate::preparation::{DriveView, GaugePolicy, MatrixView, ModelView, PhaseBasis};

    /// Four-level test model: ground {0,1}, excited {2,3}, static optical
    /// couplings, Gamma = 10/us on both excited states (good time-scale
    /// separation from the ~0.1/us optical pumping).
    struct TestModel {
        h: Vec<f64>,
        collapse: Vec<Vec<f64>>,
    }

    fn test_model() -> TestModel {
        let n = 4;
        let mut h = vec![0.; n * n];
        for (i, v) in [0., 0.3, 5., 5.4].iter().enumerate() {
            h[i * n + i] = *v;
        }
        for (i, j, v) in [(0, 2, 0.6), (1, 3, 0.5), (1, 2, 0.3)] {
            h[i * n + j] = v;
            h[j * n + i] = v;
        }
        let mut collapse = Vec::new();
        for (to, from, rate) in [(0, 2, 6.), (1, 2, 4.), (1, 3, 10.)] {
            let mut c = vec![0.; n * n];
            c[to * n + from] = f64::sqrt(rate);
            collapse.push(c);
        }
        TestModel { h, collapse }
    }

    fn prepare(model: &TestModel, axes: &[f64]) -> OwnedPrepared {
        let zero = vec![0.; 16];
        let channels: Vec<_> = model.collapse.iter().map(|c| MatrixView { real: c, imag: &zero }).collect();
        let view = ModelView {
            dimension: 4,
            hamiltonian: MatrixView { real: &model.h, imag: &zero },
            collapse: &channels,
            drives: &[],
            ground: &[true, true, false, false],
            initial_index: 0,
        };
        let phases = PhaseBasis::from_declared(axes, None, &[]).unwrap();
        OwnedPrepared::prepare(&view, phases, GaugePolicy::Identity, KernelPolicy::Csr).unwrap()
    }

    type M4 = [[C; 4]; 4];
    fn zero4() -> M4 { [[ZERO; 4]; 4] }
    fn mul4(a: &M4, b: &M4) -> M4 {
        let mut out = zero4();
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    out[i][j] = cadd(out[i][j], cmul(a[i][k], b[k][j]));
                }
            }
        }
        out
    }
    fn dagger4(a: &M4) -> M4 {
        let mut out = zero4();
        for i in 0..4 {
            for j in 0..4 {
                out[i][j] = complex(a[j][i].re, -a[j][i].im);
            }
        }
        out
    }
    fn axpy4(a: &M4, b: &M4, s: f64) -> M4 {
        let mut o = zero4();
        for i in 0..4 {
            for j in 0..4 {
                o[i][j] = cadd(a[i][j], cscale(b[i][j], s));
            }
        }
        o
    }

    /// Exact physical error for a constant-coefficient residual
    /// `r(t) = sum_q exp(i nu_q t) B R_q B^H` (a(t) = 1), by RK4 on the ORIGINAL
    /// model matrices. Returns populations `[time, state]`.
    fn exact_error(model: &TestModel, basis: &[C], residual: &[(f64, M4)], times: &[f64]) -> Vec<f64> {
        exact_error_driven(model, None, basis, residual, times)
    }

    /// `exact_error` with an optional cosine drive `cos(omega t) C`.
    fn exact_error_driven(model: &TestModel, drive: Option<(&[f64], f64)>, basis: &[C], residual: &[(f64, M4)], times: &[f64]) -> Vec<f64> {
        let n = 4;
        let mut h = zero4();
        for i in 0..n {
            for j in 0..n {
                h[i][j] = complex(model.h[i * n + j], 0.);
            }
        }
        let channels: Vec<M4> = model
            .collapse
            .iter()
            .map(|c| {
                let mut m = zero4();
                for i in 0..n {
                    for j in 0..n {
                        m[i][j] = complex(c[i * n + j], 0.);
                    }
                }
                m
            })
            .collect();
        let mut b = zero4();
        for i in 0..n {
            for a in 0..n {
                b[i][a] = basis[i + a * n];
            }
        }
        let physical: Vec<(f64, M4)> = residual.iter().map(|(nu, r)| (*nu, mul4(&mul4(&b, r), &dagger4(&b)))).collect();
        let mut hc = zero4();
        let omega = drive.map_or(0., |(_, w)| w);
        if let Some((cosine, _)) = drive {
            for i in 0..n {
                for j in 0..n {
                    hc[i][j] = complex(cosine[i * n + j], 0.);
                }
            }
        }
        let derivative = |t: f64, e: &M4| -> M4 {
            let h = axpy4(&h, &hc, (omega * t).cos());
            let he = mul4(&h, e);
            let eh = mul4(e, &h);
            let mut out = zero4();
            for i in 0..4 {
                for j in 0..4 {
                    let commutator = csub(he[i][j], eh[i][j]);
                    out[i][j] = complex(commutator.im, -commutator.re);
                }
            }
            for c in &channels {
                let cd = dagger4(c);
                let gamma = mul4(&cd, c);
                let jump = mul4(&mul4(c, e), &cd);
                let left = mul4(&gamma, e);
                let right = mul4(e, &gamma);
                for i in 0..4 {
                    for j in 0..4 {
                        out[i][j] = cadd(out[i][j], csub(jump[i][j], cscale(cadd(left[i][j], right[i][j]), 0.5)));
                    }
                }
            }
            for (nu, r) in &physical {
                let phase = cexp(complex(0., nu * t));
                for i in 0..4 {
                    for j in 0..4 {
                        out[i][j] = cadd(out[i][j], cmul(phase, r[i][j]));
                    }
                }
            }
            out
        };
        let step: f64 = 2e-4;
        let mut e = zero4();
        let mut t = 0.;
        let mut out = Vec::with_capacity(times.len() * n);
        for &target in times {
            while t < target - 1e-12 {
                let h = step.min(target - t);
                let k1 = derivative(t, &e);
                let k2 = derivative(t + h / 2., &axpy4(&e, &k1, h / 2.));
                let k3 = derivative(t + h / 2., &axpy4(&e, &k2, h / 2.));
                let k4 = derivative(t + h, &axpy4(&e, &k3, h));
                for i in 0..4 {
                    for j in 0..4 {
                        let inc = cadd(cadd(k1[i][j], cscale(k2[i][j], 2.)), cadd(cscale(k3[i][j], 2.), k4[i][j]));
                        e[i][j] = cadd(e[i][j], cscale(inc, h / 6.));
                    }
                }
                t += h;
            }
            for i in 0..n {
                out.push(e[i][i].re);
            }
        }
        out
    }

    /// Basis estimate for a synthetic one-column candidate with a constant
    /// residual: EQ = 0, FQ = R (pulled to REAL coordinates), G = 0, a0 = 1.
    fn estimate(owned: &OwnedPrepared, modes: &[Charge], residual: &[(Charge, M4)], times: &[f64], fast: bool) -> PopulationErrorEstimate {
        let map = owned.real_fourier_map(modes).unwrap();
        let rows = map.dimension();
        let n = 4;
        let d = n * n;
        let mut signed = vec![ZERO; rows];
        for (q, r) in residual {
            let position = map.modes().iter().position(|m| m == q).unwrap();
            for i in 0..n {
                for j in 0..n {
                    signed[position * d + i * n + j] = cadd(signed[position * d + i * n + j], r[i][j]);
                }
            }
        }
        let mut pulled = vec![ZERO; rows];
        map.pull_complex_into(&signed, 1, &mut pulled).unwrap();
        let fq: Vec<f64> = pulled.iter().map(|z| z.re).collect();
        let mut q = vec![0.; rows];
        q[0] = 1.;
        let eq = vec![0.; rows];
        let dual = SlowDual::prepare(owned, Default::default()).unwrap();
        let view = CandidateView { map: &map, q: &q, eq: &eq, fq: &fq, columns: 1, generator: &[0.], initial: &[1.], times };
        dual.basis_estimate_with(&view, &[], BasisOptions { fast_response: fast, ..Default::default() }).unwrap()
    }

    fn worst_relative(estimate: &[f64], exact: &[f64]) -> f64 {
        let scale = exact.iter().map(|x| x.abs()).fold(0_f64, f64::max);
        estimate.iter().zip(exact).map(|(a, b)| (a - b).abs()).fold(0_f64, f64::max) / scale
    }

    #[test]
    fn error_estimate_unresolved_candidate_is_a_conservative_bound_not_an_error() {
        let model = test_model();
        let owned = prepare(&model, &[]);
        let map = owned.real_fourier_map(&[vec![]]).unwrap();
        let rows = map.dimension();
        let mut q = vec![0.; rows];
        q[0] = 1.;
        let zeros = vec![0.; rows];
        let dual = SlowDual::prepare(&owned, Default::default()).unwrap();
        let times = [0., 1., 2.];
        let view = CandidateView { map: &map, q: &q, eq: &zeros, fq: &zeros, columns: 1, generator: &[f64::NAN], initial: &[1.], times: &times };
        let basis = dual.basis_estimate(&view, &[]).expect("never an error");
        assert_eq!(basis.maximum, 1.);
        assert!(basis.unresolved.is_some());
        let halo = dual.halo_estimate(&owned, &view, &[], HaloOptions::default()).expect("never an error");
        let total = combined_estimate(&basis, &halo.estimate).unwrap();
        assert!(total.maximum >= 1.);
    }

    #[test]
    fn error_estimate_secular_ground_population_source_matches_exact() {
        // Trace-zero population source in the ground block: the error grows
        // secularly and is redistributed by optical pumping.
        let model = test_model();
        let owned = prepare(&model, &[]);
        let basis = owned.primitives().hilbert.basis_column_major.clone();
        let mut r = zero4();
        r[0][0] = complex(1e-3, 0.);
        r[1][1] = complex(-1e-3, 0.);
        let times: Vec<f64> = (0..41).map(|i| 0.25 * i as f64).collect();
        let est = estimate(&owned, &[vec![]], &[(vec![], r)], &times, true);
        let exact = exact_error(&model, &basis, &[(0., r)], &times);
        let error = worst_relative(&est.state_error, &exact);
        assert!(error < 0.03, "secular source relative misfit {error:e}");
    }

    #[test]
    fn error_estimate_static_optical_residual_uses_the_fast_response() {
        // A purely optical (off-diagonal) residual still produces population
        // error through the Rabi coupling; the slow-only estimate cannot see
        // its early transient, the dressed fast response integrates it exactly.
        let model = test_model();
        let owned = prepare(&model, &[]);
        let basis = owned.primitives().hilbert.basis_column_major.clone();
        let mut r = zero4();
        r[0][2] = complex(0.02, 0.01);
        r[2][0] = complex(0.02, -0.01);
        let times: Vec<f64> = (0..41).map(|i| 0.2 * i as f64).collect();
        let exact = exact_error(&model, &basis, &[(0., r)], &times);
        let with_fast = estimate(&owned, &[vec![]], &[(vec![], r)], &times, true);
        let slow_only = estimate(&owned, &[vec![]], &[(vec![], r)], &times, false);
        let fast_error = worst_relative(&with_fast.state_error, &exact);
        let slow_error = worst_relative(&slow_only.state_error, &exact);
        assert!(fast_error < 0.03, "optical residual relative misfit {fast_error:e}");
        assert!(slow_error > 0.3, "slow-only estimate unexpectedly captured the optical residual ({slow_error:e})");
    }

    #[test]
    fn error_estimate_optical_residual_at_a_slow_sideband() {
        // A weak drive at 3/us makes the +-1 charges slow slots (|3 +- 0.3| is
        // inside the cutoff); an optical residual oscillating there runs the
        // dressed fast response at a nonzero slot frequency.
        let model = test_model();
        let zero = vec![0.; 16];
        let mut cosine = vec![0.; 16];
        cosine[1 * 4 + 3] = 0.05;
        cosine[3 * 4 + 1] = 0.05;
        let channels: Vec<_> = model.collapse.iter().map(|c| MatrixView { real: c, imag: &zero }).collect();
        let drives = [DriveView { omega: 3., cosine: MatrixView { real: &cosine, imag: &zero }, sine: MatrixView { real: &zero, imag: &zero } }];
        let view = ModelView {
            dimension: 4,
            hamiltonian: MatrixView { real: &model.h, imag: &zero },
            collapse: &channels,
            drives: &drives,
            ground: &[true, true, false, false],
            initial_index: 0,
        };
        let phases = PhaseBasis::from_declared(&[3.], None, &[vec![1]]).unwrap();
        let owned = OwnedPrepared::prepare(&view, phases, GaugePolicy::Identity, KernelPolicy::Csr).unwrap();
        let dual = SlowDual::prepare(&owned, Default::default()).unwrap();
        assert!(dual.slot_of(&[1]).is_some() && dual.slot_of(&[-1]).is_some(), "sidebands must be slow slots");
        let basis = owned.primitives().hilbert.basis_column_major.clone();
        let times: Vec<f64> = (0..41).map(|i| 0.2 * i as f64).collect();
        let modes = vec![vec![0], vec![1], vec![-1]];
        // Purely optical residual, then one that also moves population between
        // the ground and excited blocks (trace-free) and an excited coherence.
        for with_populations in [false, true] {
            let mut r = zero4();
            r[0][2] = complex(0.02, 0.01);
            r[2][0] = complex(0.01, -0.005);
            if with_populations {
                r[2][2] = complex(0.01, 0.002);
                r[1][1] = complex(-0.01, -0.002);
                r[3][2] = complex(0.004, 0.001);
            }
            let l = dagger4(&r);
            let exact = exact_error_driven(&model, Some((&cosine, 3.)), &basis, &[(3., r), (-3., l)], &times);
            let est = estimate(&owned, &modes, &[(vec![1], r), (vec![-1], l)], &times, true);
            let error = worst_relative(&est.state_error, &exact);
            // Before the per-slot expansion point this was 13% with a growing
            // trace defect (Markov elimination at s = 0 for a slot rotating at 3/us).
            assert!(error < 0.05, "slow-sideband residual relative misfit {error:e} (populations: {with_populations})");
            let trace_defect = (0..times.len()).map(|t| est.state_error[t * 4..t * 4 + 4].iter().sum::<f64>().abs()).fold(0_f64, f64::max);
            let scale = exact.iter().map(|x| x.abs()).fold(0_f64, f64::max);
            assert!(trace_defect < 0.05 * scale, "slot trace defect {trace_defect:e} vs scale {scale:e}");
        }
    }

    #[test]
    fn error_estimate_optical_residual_at_a_fast_charge() {
        // An optical residual at the fast charges +-1 (nu = 12/us, not slow):
        // its early transient and populations come from the dressed fast
        // response at that charge, its recycling reaches the central slot with
        // the physical phase.
        let model = test_model();
        let owned = prepare(&model, &[12.]);
        let basis = owned.primitives().hilbert.basis_column_major.clone();
        let mut r = zero4();
        r[0][2] = complex(0.03, 0.01);
        r[2][0] = complex(0.01, -0.02);
        r[3][3] = complex(0.01, 0.);
        r[1][1] = complex(-0.01, 0.);
        let l = dagger4(&r);
        let times: Vec<f64> = (0..81).map(|i| 0.05 * i as f64).collect();
        let modes = vec![vec![0], vec![1], vec![-1]];
        let exact = exact_error(&model, &basis, &[(12., r), (-12., l)], &times);
        // The ground part alone (phase-routed to the central slot) misses the
        // optical and excited response.
        let mut ground_only = zero4();
        ground_only[1][1] = r[1][1];
        let ground_estimate = estimate(&owned, &modes, &[(vec![1], ground_only), (vec![-1], dagger4(&ground_only))], &times, true);
        let est = estimate(&owned, &modes, &[(vec![1], r), (vec![-1], l)], &times, true);
        let error = worst_relative(&est.state_error, &exact);
        let ground_error = worst_relative(&ground_estimate.state_error, &exact);
        assert!(error < 0.05, "fast-charge optical residual relative misfit {error:e}");
        assert!(ground_error > 0.3, "ground routing alone unexpectedly matched ({ground_error:e})");
    }

    #[test]
    fn error_estimate_fast_phase_population_source_averages_out() {
        // A population residual oscillating at 50/us produces only an O(a/nu)
        // error; the central slot receives it with its physical phase.
        let model = test_model();
        let owned = prepare(&model, &[50.]);
        let basis = owned.primitives().hilbert.basis_column_major.clone();
        let mut r = zero4();
        r[0][0] = complex(5e-3, 0.);
        r[1][1] = complex(-5e-3, 0.);
        let times: Vec<f64> = (0..81).map(|i| 0.05 * i as f64).collect();
        let modes = vec![vec![0], vec![1], vec![-1]];
        let est = estimate(&owned, &modes, &[(vec![1], r), (vec![-1], r)], &times, true);
        let exact = exact_error(&model, &basis, &[(50., r), (-50., r)], &times);
        let error = worst_relative(&est.state_error, &exact);
        let scale = exact.iter().map(|x| x.abs()).fold(0_f64, f64::max);
        assert!(scale < 5e-4, "oscillating source should average out ({scale:e})");
        assert!(error < 0.05, "fast-phase source relative misfit {error:e}");
    }

    /// Direct RK4 integration of the ORIGINAL driven Lindblad equation
    /// (H(t) = H0 + cos(omega t) Cos), starting in state 0; populations
    /// `[time, state]`.
    fn exact_driven_populations(model: &TestModel, cosine: &[f64], omega: f64, times: &[f64]) -> Vec<f64> {
        let n = 4;
        let to_m4 = |v: &[f64]| -> M4 {
            let mut m = zero4();
            for i in 0..n {
                for j in 0..n {
                    m[i][j] = complex(v[i * n + j], 0.);
                }
            }
            m
        };
        let h0 = to_m4(&model.h);
        let hc = to_m4(cosine);
        let channels: Vec<M4> = model.collapse.iter().map(|c| to_m4(c)).collect();
        let derivative = |t: f64, rho: &M4| -> M4 {
            let h = axpy4(&h0, &hc, (omega * t).cos());
            let hr = mul4(&h, rho);
            let rh = mul4(rho, &h);
            let mut out = zero4();
            for i in 0..4 {
                for j in 0..4 {
                    let c = csub(hr[i][j], rh[i][j]);
                    out[i][j] = complex(c.im, -c.re);
                }
            }
            for c in &channels {
                let cd = dagger4(c);
                let gamma = mul4(&cd, c);
                let jump = mul4(&mul4(c, rho), &cd);
                let left = mul4(&gamma, rho);
                let right = mul4(rho, &gamma);
                for i in 0..4 {
                    for j in 0..4 {
                        out[i][j] = cadd(out[i][j], csub(jump[i][j], cscale(cadd(left[i][j], right[i][j]), 0.5)));
                    }
                }
            }
            out
        };
        let step: f64 = 1e-4;
        let mut rho = zero4();
        rho[0][0] = complex(1., 0.);
        let mut t = 0.;
        let mut out = Vec::new();
        for &target in times {
            while t < target - 1e-12 {
                let h = step.min(target - t);
                let k1 = derivative(t, &rho);
                let k2 = derivative(t + h / 2., &axpy4(&rho, &k1, h / 2.));
                let k3 = derivative(t + h / 2., &axpy4(&rho, &k2, h / 2.));
                let k4 = derivative(t + h, &axpy4(&rho, &k3, h));
                for i in 0..4 {
                    for j in 0..4 {
                        let inc = cadd(cadd(k1[i][j], cscale(k2[i][j], 2.)), cadd(cscale(k3[i][j], 2.), k4[i][j]));
                        rho[i][j] = cadd(rho[i][j], cscale(inc, h / 6.));
                    }
                }
                t += h;
            }
            for i in 0..n {
                out.push(rho[i][i].re);
            }
        }
        out
    }

    #[test]
    fn error_estimate_halo_predicts_fourier_truncation_error() {
        use super::super::b0_inverse::P0Policy;
        use super::super::coordinates::InitialView;
        use super::super::directions::{shift_invert, InnerSolve};
        use super::super::reduced::ReducedModel;
        use super::super::runtime_core::RuntimeCore;
        use super::super::trial::TrialSpace;
        // Add a near-resonant oscillating drive (sideband at 6/us against the
        // 5/us optical splitting) so that the {0, +-1} domain truncates.
        let model = test_model();
        let zero = vec![0.; 16];
        let mut cosine = vec![0.; 16];
        cosine[2] = 1.2;
        cosine[8] = 1.2;
        let channels: Vec<_> = model.collapse.iter().map(|c| MatrixView { real: c, imag: &zero }).collect();
        let drives = [DriveView { omega: 6., cosine: MatrixView { real: &cosine, imag: &zero }, sine: MatrixView { real: &zero, imag: &zero } }];
        let view = ModelView {
            dimension: 4,
            hamiltonian: MatrixView { real: &model.h, imag: &zero },
            collapse: &channels,
            drives: &drives,
            ground: &[true, true, false, false],
            initial_index: 0,
        };
        let phases = PhaseBasis::from_declared(&[6.], None, &[vec![1]]).unwrap();
        let owned = OwnedPrepared::prepare(&view, phases, GaugePolicy::Identity, KernelPolicy::Csr).unwrap();
        let mut core = RuntimeCore::from_owned(owned, InitialView { index: 0, real: None, imag: None }, P0Policy::EigenPreferred).unwrap();
        let times: Vec<f64> = (0..101).map(|i| 0.1 * i as f64).collect();
        // Rational Krylov space on the {0, +-1} domain (one pole, chained from
        // the newest column), resolved in the basis so that the domain
        // truncation dominates the error.
        let modes: Vec<Charge> = (-1..=1).map(|k| vec![k]).collect();
        let mut space = TrialSpace::new(&mut core, &modes, &[2.]).unwrap();
        let inner = InnerSolve { maximum_steps: 12, relative_residual: 1e-3 };
        while space.columns() < 30 {
            let rows = space.rows();
            let newest = space.columns() - 1;
            let rhs = space.eq()[newest * rows..(newest + 1) * rows].to_vec();
            let (direction, _) = shift_invert(&mut space.op, &rhs, 1, &[0], inner).unwrap();
            if space.append(direction, 1).unwrap().is_empty() {
                break;
            }
        }
        let reduced = ReducedModel::build(&space).unwrap();
        let lapack = Arc::clone(core.lapack());
        let trajectory = reduced.trajectory(&lapack, &times).unwrap();
        let candidate = space.readout().evaluate(&lapack, &trajectory, &times).unwrap();
        let reference = exact_driven_populations(&model, &cosine, 6., &times);
        let owned = core.prepared();
        let dual = SlowDual::prepare(owned, Default::default()).unwrap();
        let view = CandidateView {
            map: space.op.map(),
            q: space.q(),
            eq: space.eq(),
            fq: space.fq(),
            columns: space.columns(),
            generator: &reduced.generator,
            initial: &reduced.initial,
            times: &times,
        };
        let basis = dual.basis_estimate(&view, &[]).unwrap();
        let actual: Vec<f64> = reference.iter().zip(&candidate.values).map(|(a, b)| a - b).collect();
        let actual_max = actual.iter().map(|x| x.abs()).fold(0_f64, f64::max);
        assert!(actual_max > 1e-5, "test domain must truncate visibly ({actual_max:e})");
        // The +-2 shell is itself slow here (|12 -+ 0.3| < 20): its ground
        // block must go to J_eff, not to the diagonal halo response, and the
        // returns into the slow +-1 slots need their non-ground part.
        for all_halo_modes in [true, false] {
            let halo = dual.halo_estimate(owned, &view, &[], HaloOptions { all_halo_modes, ..Default::default() }).unwrap();
            let combined = combined_estimate(&basis, &halo.estimate).unwrap();
            let dot: f64 = actual.iter().zip(&combined.state_error).map(|(a, b)| a * b).sum();
            let norms = actual.iter().map(|x| x * x).sum::<f64>().sqrt() * combined.state_error.iter().map(|x| x * x).sum::<f64>().sqrt();
            let cosine = dot / norms;
            let ratio = combined.maximum_state / actual_max;
            println!("truncation error {actual_max:e}, basis {:e}, halo {:e}, ratio {ratio:.3}, cos {cosine:.3}", basis.maximum_state, halo.estimate.maximum_state);
            assert!((0.7..1.4).contains(&ratio), "halo effectivity {ratio}");
            assert!(cosine > 0.9, "halo shape similarity {cosine}");
            assert_eq!(halo.modes[0].charge, vec![2], "the nearest shell carries the truncation error");
        }
    }

    /// Basis and halo estimates of the driven model of the halo test, plus
    /// (with `weak > 0`) a second drive at 7/us on the forbidden pair 0-3 at
    /// `weak` times the first drive: (dual diagnostics, basis, halo maxima).
    fn driven_estimates(weak: f64) -> (super::super::slow_dual::SlowDualDiagnostics, f64, f64) {
        use super::super::b0_inverse::P0Policy;
        use super::super::coordinates::InitialView;
        use super::super::directions::{shift_invert, InnerSolve};
        use super::super::reduced::ReducedModel;
        use super::super::runtime_core::RuntimeCore;
        use super::super::trial::TrialSpace;
        let model = test_model();
        let zero = vec![0.; 16];
        let mut cosine = vec![0.; 16];
        cosine[2] = 1.2;
        cosine[8] = 1.2;
        let mut forbidden = vec![0.; 16];
        forbidden[3] = 1.2 * weak;
        forbidden[12] = 1.2 * weak;
        let channels: Vec<_> = model.collapse.iter().map(|c| MatrixView { real: c, imag: &zero }).collect();
        let blank = [0.; 16];
        let strong = DriveView { omega: 6., cosine: MatrixView { real: &cosine, imag: &blank }, sine: MatrixView { real: &blank, imag: &blank } };
        let forbidden = DriveView { omega: 7., cosine: MatrixView { real: &forbidden, imag: &blank }, sine: MatrixView { real: &blank, imag: &blank } };
        let (drives, phases, modes) = if weak > 0. {
            let phases = PhaseBasis::from_declared(&[6., 7.], None, &[vec![1, 0], vec![0, 1]]).unwrap();
            (vec![strong, forbidden], phases, (-1..=1).map(|k| vec![k, 0]).collect::<Vec<Charge>>())
        } else {
            (vec![strong], PhaseBasis::from_declared(&[6.], None, &[vec![1]]).unwrap(), (-1..=1).map(|k| vec![k]).collect())
        };
        let view = ModelView {
            dimension: 4,
            hamiltonian: MatrixView { real: &model.h, imag: &zero },
            collapse: &channels,
            drives: &drives,
            ground: &[true, true, false, false],
            initial_index: 0,
        };
        let owned = OwnedPrepared::prepare(&view, phases, GaugePolicy::Identity, KernelPolicy::Csr).unwrap();
        let mut core = RuntimeCore::from_owned(owned, InitialView { index: 0, real: None, imag: None }, P0Policy::EigenPreferred).unwrap();
        let times: Vec<f64> = (0..101).map(|i| 0.1 * i as f64).collect();
        let mut space = TrialSpace::new(&mut core, &modes, &[2.]).unwrap();
        let inner = InnerSolve { maximum_steps: 12, relative_residual: 1e-3 };
        while space.columns() < 30 {
            let rows = space.rows();
            let newest = space.columns() - 1;
            let rhs = space.eq()[newest * rows..(newest + 1) * rows].to_vec();
            let (direction, _) = shift_invert(&mut space.op, &rhs, 1, &[0], inner).unwrap();
            if space.append(direction, 1).unwrap().is_empty() {
                break;
            }
        }
        let reduced = ReducedModel::build(&space).unwrap();
        let owned = core.prepared();
        let dual = SlowDual::prepare(owned, Default::default()).unwrap();
        let view = CandidateView {
            map: space.op.map(),
            q: space.q(),
            eq: space.eq(),
            fq: space.fq(),
            columns: space.columns(),
            generator: &reduced.generator,
            initial: &reduced.initial,
            times: &times,
        };
        let basis = dual.basis_estimate(&view, &[]).unwrap();
        let halo = dual.halo_estimate(owned, &view, &[], HaloOptions::default()).unwrap();
        (dual.diagnostics().clone(), basis.maximum, halo.estimate.maximum)
    }

    #[test]
    fn error_estimate_is_continuous_in_weak_couplings() {
        // A forbidden transition switched on at 1e-7 of the drive adds weak
        // operator charges and slow charges that only they reach (7, 13, 14,
        // -1 rad/us are all below the cutoff here). They must not become
        // slots, and the estimates must move by about the coupling, not by the
        // structure: the slow charges and halo modes they reach are charged
        // or returned at their strength.
        let (plain, basis, halo) = driven_estimates(0.);
        let (perturbed, basis_p, halo_p) = driven_estimates(1e-7);
        println!("basis {basis:e} -> {basis_p:e}, halo {halo:e} -> {halo_p:e}, unrepresented {} at {:e}", perturbed.unrepresented_slow_charges, perturbed.unrepresented_rate);
        assert_eq!(plain.weak_operator_charges, 0);
        assert_eq!(perturbed.weak_operator_charges, 2);
        assert_eq!(perturbed.slow_charges.len(), plain.slow_charges.len(), "weak couplings shape no slots");
        assert!(perturbed.unrepresented_slow_charges > 0, "the weak drive reaches slow charges");
        assert!(perturbed.unrepresented_rate < 1e-5, "coupling into them {:e}", perturbed.unrepresented_rate);
        assert!((basis_p - basis).abs() < 1e-4 * basis, "basis {basis:e} -> {basis_p:e}");
        assert!((halo_p - halo).abs() < 1e-4 * halo, "halo {halo:e} -> {halo_p:e}");
    }
}
