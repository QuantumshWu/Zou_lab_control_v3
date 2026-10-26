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
//! When `q` is itself a slow charge its ground block is slow content: it goes
//! to the slot of `q` as a direct `J_eff` source and only the non-ground rest
//! gets the response above. The exact identity
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
//! Each halo charge pair `{q, -q}` yields its own signed contribution, which is
//! the domain-selection indicator.
use super::*;
use super::real_fourier::RealFourierMap;
use super::slow_dual::{cabs, cadd, cdiv, cexp, cmul, complex, cscale, csub, exp_divided_2, recycle_ground, Dense, LuFactor, ModalSources, SlowDual, NEGLIGIBLE_EXPONENT, ZERO};

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
    /// Output times (us), any order of nonnegative finite values.
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
    pub maximum: f64,
    /// Largest imaginary part discarded by the real readout (consistency).
    pub maximum_imaginary: f64,
    pub elapsed_ms: f64,
}

#[derive(Clone, Debug)]
pub struct HaloModeIndicator {
    /// Canonical member of the signed pair `{charge, -charge}`.
    pub charge: Charge,
    /// Max over time, states and groups of the pair's signed contribution.
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
    /// Halo output modes per pencil application (each application recomputes
    /// its intermediate modes, so larger chunks are cheaper).
    pub chunk_modes: usize,
    /// Columns per pencil application.
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
    fn default() -> Self { Self { chunk_modes: 48, maximum_rhs: 8, resonance_floor: None, all_halo_modes: false, slave_fast_returns: false } }
}

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
        let complex_g: Vec<C> = generator.iter().map(|&x| complex(x, 0.)).collect();
        let (exponents, vectors) = lap.complex_eigen_right(&complex_g, m)?;
        let vectors = Dense { rows: m, cols: m, data: vectors };
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

    /// `R V diag(w)` for a field block `R` (rows x m): column j then evolves
    /// as `exp(beta_j t)`.
    fn modal_columns(&self, block: &Dense, lap: &Lapack) -> Result<Dense, String> {
        let mut out = block.multiply(&self.vectors, lap)?;
        for j in 0..out.cols {
            for i in 0..out.rows {
                out.data[i + j * out.rows] = cmul(out.data[i + j * out.rows], self.weights[j]);
            }
        }
        Ok(out)
    }
}

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

/// Push a column-major REAL block `[rows, m]` to the complete signed field
/// `[mode * d + entry, m]` (row-major in the column index).
fn push_columns(map: &RealFourierMap, real: &[f64], rows: usize, m: usize) -> Result<Vec<C>, String> {
    let mut row_major = vec![0.; rows * m];
    for j in 0..m {
        for i in 0..rows {
            row_major[i * m + j] = real[i + j * rows];
        }
    }
    let mut signed = vec![ZERO; rows * m];
    map.push_into(&row_major, m, &mut signed)?;
    Ok(signed)
}

/// Ground block (g^2 rows, row-major entry order) of one signed mode.
fn ground_block(signed: &[C], mode: usize, n: usize, g: usize, m: usize) -> Dense {
    let d = n * n;
    let mut block = Dense::zeros(g * g, m);
    for a in 0..g {
        for b in 0..g {
            let row = (mode * d + a * n + b) * m;
            for j in 0..m {
                block.set(a * g + b, j, signed[row + j]);
            }
        }
    }
    block
}

/// `sum_ik W[s, ik] exp(lambda_ik t)` for every output time, skipping the
/// (mode, time) pairs where `exp(Re lambda t)` is below `exp(-41.4) ~ 1e-18`:
/// fast dressed modes decay within a few optical lifetimes, so they touch only
/// the earliest samples while the slow (ground-like) modes cover all of them.
fn natural_readout(weights: &Dense, lambda: &[C], times: &[f64]) -> Dense {
    let states = weights.rows;
    let mut values = Dense::zeros(states, times.len());
    for (ik, &l) in lambda.iter().enumerate() {
        let column = &weights.data[ik * states..(ik + 1) * states];
        if column.iter().all(|z| z.re == 0. && z.im == 0.) {
            continue;
        }
        for (it, &t) in times.iter().enumerate() {
            if l.re * t < NEGLIGIBLE_EXPONENT {
                continue;
            }
            let e = cexp(cscale(l, t));
            let out = &mut values.data[it * states..(it + 1) * states];
            for (o, w) in out.iter_mut().zip(column) {
                *o = cadd(*o, cmul(*w, e));
            }
        }
    }
    values
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
    Ok(PopulationErrorEstimate {
        times: times.to_vec(),
        states: n,
        state_error,
        group_names: groups.iter().map(|g| g.name.clone()).collect(),
        group_error,
        maximum_state,
        maximum_group,
        maximum: maximum_state.max(maximum_group),
        maximum_imaginary,
        elapsed_ms: began.elapsed().as_secs_f64() * 1000.,
    })
}

#[derive(Clone, Copy, Debug)]
pub struct BasisOptions {
    /// Integrate the non-ground residual at every retained charge exactly
    /// through the dressed no-jump dynamics (early fast transients).
    pub fast_response: bool,
    /// Additionally Markov-slave the non-ground residual at fast retained
    /// charges and return it one drive hop into the slow slots.
    pub slave_fast_charges: bool,
}

impl Default for BasisOptions {
    fn default() -> Self { Self { fast_response: true, slave_fast_charges: false } }
}

impl SlowDual {
    /// Basis (in-domain) estimate from the retained residual rows only.
    pub fn basis_estimate(&self, view: &CandidateView<'_>, groups: &[PopulationGroup]) -> Result<PopulationErrorEstimate, String> {
        self.basis_estimate_with(view, groups, BasisOptions::default())
    }

    /// Fast response at one charge `q`:
    /// `eta' = (N - i nu_q) eta + Rq_hat a(t)`, `eta(0) = 0`, where `Rq_hat`
    /// is the residual (or returned field) at `q` without its ground block. In
    /// the dressed eigen-coordinates every entry is
    /// `sum_j c_ikj (exp(beta_j t) - exp(lambda_ik t)) / (beta_j - lambda_ik)`,
    /// `lambda_ik = a_i + b_k - i nu_q`. The populations of `eta` are read out
    /// directly (phase `exp(i nu_q t)`); its recycling `R_0 eta_EE` (the only
    /// part of `L_0 - N` at the same charge) is a `J_eff` source for
    /// `target = (slot, shift)`: the slot of `q` with shift 0 when `q` is slow,
    /// otherwise the central slot with the physical phase (`shift = nu_q`).
    #[allow(clippy::too_many_arguments)]
    fn add_fast_response(
        &self,
        target: (usize, f64),
        nu: f64,
        exponents: &[C],
        coefficients: &Dense,
        times: &[f64],
        sources: &mut ModalSources,
        direct: &mut [C],
    ) -> Result<(), String> {
        let dressed = match &self.dressed {
            Some(d) => d,
            None => return Ok(()),
        };
        let lap = &self.lapack;
        let n = self.n;
        let n2 = n * n;
        let m = coefficients.cols;
        let nt = times.len();
        let t_max = times.iter().copied().fold(0_f64, f64::max);
        // Source columns in dressed eigen-coordinates (index i*n+k).
        let mut source = Dense::zeros(n2, m);
        for j in 0..m {
            let mut x = Dense::zeros(n, n);
            for entry in 0..n2 {
                x.set(entry / n, entry % n, coefficients.at(entry, j));
            }
            let xh = dressed.to_eigen(&x, lap)?;
            for i in 0..n {
                for k in 0..n {
                    source.set(i * n + k, j, xh.at(i, k));
                }
            }
        }
        let lambda: Vec<C> = (0..n2)
            .map(|ik| csub(cadd(dressed.left_values[ik / n], dressed.right_values[ik % n]), complex(0., nu)))
            .collect();
        let mut forced = Dense::zeros(n2, m);
        let mut homogeneous = vec![ZERO; n2];
        let mut near: Vec<(usize, usize, C)> = Vec::new();
        for j in 0..m {
            for ik in 0..n2 {
                let c = source.at(ik, j);
                if c.re == 0. && c.im == 0. {
                    continue;
                }
                let delta = csub(exponents[j], lambda[ik]);
                if cabs(delta) * t_max > 1. {
                    let f = cdiv(c, delta);
                    forced.set(ik, j, f);
                    homogeneous[ik] = csub(homogeneous[ik], f);
                } else {
                    near.push((ik, j, c));
                }
            }
        }
        // Direct population readout: rho (F e^{beta t} + H e^{lambda t} + near).
        let states = dressed.readout.rows;
        let readout_forced = dressed.readout.multiply(&forced, lap)?;
        let mut readout_homogeneous = dressed.readout.clone();
        for ik in 0..n2 {
            for s in 0..states {
                let v = readout_homogeneous.at(s, ik);
                readout_homogeneous.set(s, ik, cmul(v, homogeneous[ik]));
            }
        }
        let mut forced_phases = Dense::zeros(m, nt);
        for (it, &t) in times.iter().enumerate() {
            for j in 0..m {
                forced_phases.set(j, it, cexp(cscale(exponents[j], t)));
            }
        }
        let values_forced = readout_forced.multiply(&forced_phases, lap)?;
        let values_natural = natural_readout(&readout_homogeneous, &lambda, times);
        for (it, &t) in times.iter().enumerate() {
            let phase = if nu == 0. { complex(1., 0.) } else { cexp(complex(0., nu * t)) };
            for s in 0..states.min(n) {
                let mut v = cadd(values_forced.at(s, it), values_natural.at(s, it));
                for &(ik, j, c) in &near {
                    let w = dressed.readout.at(s, ik);
                    if w.re != 0. || w.im != 0. {
                        v = cadd(v, cmul(cmul(w, c), exp_divided_2(exponents[j], lambda[ik], t)));
                    }
                }
                direct[it * n + s] = cadd(direct[it * n + s], cmul(phase, v));
            }
        }
        // Recycling return into the target slot (with the routing phase).
        let (slot, shift) = target;
        let rotate = |z: C| cadd(z, complex(0., shift));
        let modal = &dressed.modal_recycling[slot];
        let forced_modal = modal.multiply(&forced, lap)?;
        for j in 0..m {
            sources.add_exponential(rotate(exponents[j]), forced_modal.data[j * forced_modal.rows..(j + 1) * forced_modal.rows].to_vec());
        }
        let dim = modal.rows;
        for ik in 0..n2 {
            let h = homogeneous[ik];
            if h.re == 0. && h.im == 0. {
                continue;
            }
            let column: Vec<C> = (0..dim).map(|r| cmul(modal.at(r, ik), h)).collect();
            if column.iter().all(|z| z.re == 0. && z.im == 0.) {
                continue;
            }
            sources.add_exponential(rotate(lambda[ik]), column);
        }
        for &(ik, j, c) in &near {
            let column: Vec<C> = (0..dim).map(|r| cmul(modal.at(r, ik), c)).collect();
            if column.iter().any(|z| z.re != 0. || z.im != 0.) {
                sources.add_pair(rotate(exponents[j]), rotate(lambda[ik]), column);
            }
        }
        Ok(())
    }

    /// Slow part of a physical initial error `X0` (central charge): its ground
    /// block becomes the initial slow error. Returns the remaining fast layer
    /// `X0_Q - S(P X0)`: the non-slow part minus the pattern the Markov readout
    /// would already attribute to the slow part (it must relax, not appear
    /// instantly).
    fn add_slow_initial(&self, initial: &Dense, sources: &mut ModalSources) -> Result<Dense, String> {
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
            sources.add_initial(&self.to_modal(0, &ground)?.data);
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
    fn add_fast_initial(&self, initial: &Dense, times: &[f64], sources: &mut ModalSources, direct: &mut [C]) -> Result<(), String> {
        let dressed = match &self.dressed {
            Some(d) => d,
            None => return Ok(()),
        };
        let lap = &self.lapack;
        let n = self.n;
        let n2 = n * n;
        let nt = times.len();
        let xh = dressed.to_eigen(initial, lap)?;
        let lambda: Vec<C> = (0..n2).map(|ik| cadd(dressed.left_values[ik / n], dressed.right_values[ik % n])).collect();
        let states = dressed.readout.rows;
        let mut weighted = dressed.readout.clone();
        for ik in 0..n2 {
            let x = xh.at(ik / n, ik % n);
            for s in 0..states {
                let v = weighted.at(s, ik);
                weighted.set(s, ik, cmul(v, x));
            }
        }
        let values = natural_readout(&weighted, &lambda, times);
        for it in 0..nt {
            for s in 0..states.min(n) {
                direct[it * n + s] = cadd(direct[it * n + s], values.at(s, it));
            }
        }
        let modal = &dressed.modal_recycling[0];
        let dim = modal.rows;
        for ik in 0..n2 {
            let x = xh.at(ik / n, ik % n);
            if x.re == 0. && x.im == 0. {
                continue;
            }
            let column: Vec<C> = (0..dim).map(|r| cmul(modal.at(r, ik), x)).collect();
            if column.iter().any(|z| z.re != 0. || z.im != 0.) {
                sources.add_exponential(lambda[ik], column);
            }
        }
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
        let small = SmallSource::new(lap, view.generator, view.initial, m)?;
        // Retained residual rows R = F Q - E Q G (REAL coordinates).
        let mut residual = lap.multiply(view.eq, view.generator, rows, m, m);
        for (r, f) in residual.iter_mut().zip(view.fq) {
            *r = f - *r;
        }
        let signed = push_columns(view.map, &residual, rows, m)?;
        let phases = &self.owner.hilbert.source.phases;
        let d = n * n;
        let mut sources = ModalSources::default();
        let mut direct = vec![ZERO; view.times.len() * n];
        for (mode, charge) in view.map.modes().iter().enumerate() {
            let slow_slot = self.slot_of(charge);
            if options.fast_response && self.dressed.is_some() {
                // Non-ground part of the residual: exact dressed fast response
                // at this charge; its recycling feeds the slot of the charge
                // (slow) or the central slot with the physical phase (fast).
                let mut field = Dense::zeros(d, m);
                for entry in 0..d {
                    let (a, b) = (entry / n, entry % n);
                    if a < g && b < g {
                        continue;
                    }
                    for j in 0..m {
                        field.set(entry, j, signed[(mode * d + entry) * m + j]);
                    }
                }
                if field.frobenius() > 0. {
                    let coefficients = small.modal_columns(&field, lap)?;
                    let nu = phases.frequency(charge).map(|f| f.hi + f.lo)?;
                    let target = slow_slot.map_or((0, nu), |s| (s, 0.));
                    self.add_fast_response(target, nu, &small.exponents, &coefficients, view.times, &mut sources, &mut direct)?;
                }
            }
            if options.slave_fast_charges && slow_slot.is_none() {
                let mut field = Dense::zeros(d, m);
                for entry in 0..d {
                    for j in 0..m {
                        field.set(entry, j, signed[(mode * d + entry) * m + j]);
                    }
                }
                let coefficients = small.modal_columns(&field, lap)?;
                for (target_slot, returned) in self.slaved_returns(charge, &coefficients)? {
                    let modal = self.to_modal(target_slot, &returned)?;
                    for j in 0..m {
                        sources.add_exponential(small.exponents[j], modal.data[j * modal.rows..(j + 1) * modal.rows].to_vec());
                    }
                }
            }
            let block = ground_block(&signed, mode, n, g, m);
            if block.frobenius() == 0. {
                continue;
            }
            let (slot, shift) = match slow_slot {
                Some(s) => (s, 0.),
                None => (0, phases.frequency(charge).map(|f| f.hi + f.lo)?),
            };
            let columns = small.modal_columns(&block, lap)?;
            let modal = self.to_modal(slot, &columns)?;
            for j in 0..m {
                let exponent = cadd(small.exponents[j], complex(0., shift));
                sources.add_exponential(exponent, modal.data[j * modal.rows..(j + 1) * modal.rows].to_vec());
            }
        }
        let amplitudes = self.propagate(&sources, view.times)?;
        let mut populations = self.read_populations(&amplitudes, view.times)?;
        for (p, x) in populations.iter_mut().zip(&direct) {
            *p = cadd(*p, *x);
        }
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
        if options.chunk_modes == 0 || options.maximum_rhs == 0 {
            return Err("Halo chunk sizes must be positive".into());
        }
        let lap = &self.lapack;
        let (n, g) = (self.n, self.g);
        let d = n * n;
        let nt = view.times.len();
        let floor = options.resonance_floor.unwrap_or(self.diagnostics.optical_decay_floor);
        let phases = &self.owner.hilbert.source.phases;
        let frequency = |q: &[i32]| -> Result<f64, String> { phases.frequency(q).map(|f| f.hi + f.lo) };
        let small = SmallSource::new(lap, view.generator, view.initial, m)?;
        let retained: Vec<Charge> = view.map.modes().to_vec();
        let retained_set: Modes = retained.iter().cloned().collect();
        let mut halo = Self::halo_modes(prepared, &retained)?;
        if !options.all_halo_modes {
            // Keep the halo modes that return into the retained domain (or a
            // slow slot) in one Liouvillian hop: only they feed the slow error.
            let returns = |q: &Charge| -> bool {
                self.frame.terms.iter().any(|term| {
                    add_charge(q, &term.charge, false).is_ok_and(|k| retained_set.contains(&k) || self.slot_of(&k).is_some())
                })
            };
            halo.retain(|q| returns(q));
        }
        let signed_q = push_columns(view.map, view.q, rows, m)?;

        // Canonical pair index for every halo charge.
        let canonical = |q: &Charge| -> Charge {
            let negative: Charge = q.iter().map(|x| -x).collect();
            if q.iter().copied().find(|&v| v != 0).is_none_or(|v| v > 0) { q.clone() } else { negative }
        };
        let mut pair_index: BTreeMap<Charge, usize> = BTreeMap::new();
        for q in &halo {
            let key = canonical(q);
            let next = pair_index.len();
            pair_index.entry(key).or_insert(next);
        }
        let pairs = pair_index.len();
        let mut pair_sources: Vec<ModalSources> = (0..pairs).map(|_| ModalSources::default()).collect();
        let mut pair_direct: Vec<Vec<C>> = vec![vec![ZERO; nt * n]; pairs];
        let mut pair_residual: Vec<f64> = vec![0.; pairs];
        let mut pair_boundary: Vec<Dense> = vec![Dense::zeros(n, n); pairs];
        let mut near_resonant_terms = 0;
        let mut unrouted_resonant_returns = 0;
        // Non-ground returns into slow charges, summed per (pair, slot).
        let mut fast_fields: BTreeMap<(usize, Charge), ((usize, f64), f64, Dense)> = BTreeMap::new();
        let mut pencil_ms = 0.;

        let rhs_max = options.maximum_rhs.min(m);
        for chunk in halo.chunks(options.chunk_modes) {
            // --- Halo residual rows R = F Q - E Q G on this chunk ----------
            // Accumulated as R^T (m x rows) one column block at a time: the
            // row-major [row, column] pencil outputs are column-major
            // [column, row] blocks, so F^T fills rows of R^T and
            // -G[block, :]^T E_block^T is one product.
            let pencil_began = Instant::now();
            let mut plan = prepared.pencil(&retained, chunk, rhs_max)?;
            let out_rows = chunk.len() * d;
            let mut residual_t = Dense::zeros(m, out_rows);
            let mut start = 0;
            while start < m {
                let width = rhs_max.min(m - start);
                let mut input = vec![ZERO; rows * width];
                for i in 0..rows {
                    input[i * width..(i + 1) * width].copy_from_slice(&signed_q[i * m + start..i * m + start + width]);
                }
                let mut e_out = vec![ZERO; out_rows * width];
                let mut f_out = vec![ZERO; out_rows * width];
                plan.apply_into(&input, width, &mut e_out, &mut f_out)?;
                let mut generator_block_t = Dense::zeros(m, width);
                for j in 0..m {
                    for k in 0..width {
                        generator_block_t.set(j, k, complex(view.generator[(start + k) + j * m], 0.));
                    }
                }
                let mass_block = Dense { rows: width, cols: out_rows, data: e_out };
                let correction = generator_block_t.multiply(&mass_block, lap)?;
                for row in 0..out_rows {
                    for j in 0..m {
                        let at = j + row * m;
                        residual_t.data[at] = csub(residual_t.data[at], correction.data[at]);
                    }
                    for k in 0..width {
                        let at = (start + k) + row * m;
                        residual_t.data[at] = cadd(residual_t.data[at], f_out[row * width + k]);
                    }
                }
                start += width;
            }
            pencil_ms += pencil_began.elapsed().as_secs_f64() * 1000.;

            for (local, q) in chunk.iter().enumerate() {
                let pair = pair_index[&canonical(q)];
                let nu_q = frequency(q)?;
                // R_q as a d x m Dense (row = frame entry a*n+b).
                let mut residual = Dense::zeros(d, m);
                for entry in 0..d {
                    for j in 0..m {
                        residual.set(entry, j, residual_t.at(j, local * d + entry));
                    }
                }
                let norm = residual.frobenius();
                if norm == 0. {
                    continue;
                }
                pair_residual[pair] = pair_residual[pair].hypot(norm);
                // A halo charge that is itself a slow slot: its ground block is
                // slow content, a direct J_eff source (as in the basis estimate);
                // only the non-ground rest gets the fast halo response below.
                if let Some(own) = self.slot_of(q) {
                    let mut block = Dense::zeros(g * g, m);
                    for a in 0..g {
                        for b in 0..g {
                            for j in 0..m {
                                block.set(a * g + b, j, residual.at(a * n + b, j));
                                residual.set(a * n + b, j, ZERO);
                            }
                        }
                    }
                    if block.frobenius() > 0. {
                        let columns = small.modal_columns(&block, lap)?;
                        let modal = self.to_modal(own, &columns)?;
                        for j in 0..m {
                            pair_sources[pair].add_exponential(small.exponents[j], modal.data[j * modal.rows..(j + 1) * modal.rows].to_vec());
                        }
                    }
                }
                let coefficients = small.modal_columns(&residual, lap)?;

                // --- Envelope response eta_q ---------------------------------
                // Particular columns X[:, j] and near-resonant terms.
                let (particular, resonant) = self.halo_response(&coefficients, &small.exponents, nu_q, floor)?;
                near_resonant_terms += resonant.len();

                // --- Direct readout of eta (physical phase exp(i nu_q t)) ----
                let direct = &mut pair_direct[pair];
                for j in 0..m {
                    let exponent = cadd(small.exponents[j], complex(0., nu_q));
                    let readout = self.frame_populations(&particular[j]);
                    if readout.iter().all(|z| z.re == 0. && z.im == 0.) {
                        continue;
                    }
                    for (it, &t) in view.times.iter().enumerate() {
                        let phase = cexp(cscale(exponent, t));
                        for state in 0..n {
                            direct[it * n + state] = cadd(direct[it * n + state], cmul(readout[state], phase));
                        }
                    }
                }
                for term in &resonant {
                    let readout = self.frame_populations(&term.unit);
                    let first = cadd(small.exponents[term.column], complex(0., nu_q));
                    let second = cadd(term.second, complex(0., nu_q));
                    for (it, &t) in view.times.iter().enumerate() {
                        let value = exp_divided_2(first, second, t);
                        for state in 0..n {
                            direct[it * n + state] = cadd(direct[it * n + state], cmul(readout[state], value));
                        }
                    }
                }

                // --- Initial boundary -eta(0) (physical matrix at t = 0) ------
                let boundary = &mut pair_boundary[pair];
                for x in &particular {
                    for k in 0..d {
                        boundary.data[k] = csub(boundary.data[k], x.data[k]);
                    }
                }

                // --- One-hop return ------------------------------------------
                // p = 0 (same charge): only the ground block of (L_0 - D) eta,
                // i.e. what the diagonal reference D left out (recycling and
                // static couplings into the ground block); it feeds slot q or,
                // for a fast q, the central slot with phase exp(i nu_q t).
                // p != 0 into a retained or slow charge k: the complete field
                // L_p eta; its ground block is a slow source, its non-ground
                // block gets the exact dressed fast response (k slow) or Markov
                // slaving plus one hop (k fast).
                for term in &self.frame.terms {
                    let same_charge = term.charge.iter().all(|&v| v == 0);
                    let k = add_charge(q, &term.charge, false)?;
                    if !same_charge && !retained_set.contains(&k) && self.slot_of(&k).is_none() {
                        continue;
                    }
                    let (slot, shift) = self.target(&k, &frequency)?;
                    let mut field = Dense::zeros(d, m);
                    for (j, x) in particular.iter().enumerate() {
                        if x.frobenius() == 0. {
                            continue;
                        }
                        if same_charge {
                            for (row, v) in self.same_charge_ground_return(x).iter().enumerate() {
                                field.set((row / g) * n + row % g, j, *v);
                            }
                            continue;
                        }
                        let image = self.frame.apply(term, x, lap)?;
                        for a in 0..n {
                            for b in 0..n {
                                field.set(a * n + b, j, image.at(a, b));
                            }
                        }
                    }
                    if field.frobenius() > 0. {
                        let mut block = Dense::zeros(g * g, m);
                        for a in 0..g {
                            for b in 0..g {
                                for j in 0..m {
                                    block.set(a * g + b, j, field.at(a * n + b, j));
                                }
                            }
                        }
                        if block.frobenius() > 0. {
                            let modal = self.to_modal(slot, &block)?;
                            for j in 0..m {
                                let exponent = cadd(small.exponents[j], complex(0., shift));
                                pair_sources[pair].add_exponential(exponent, modal.data[j * modal.rows..(j + 1) * modal.rows].to_vec());
                            }
                        }
                        if !same_charge {
                            // Exact dressed fast response of the non-ground block at k.
                            // Linear in the field: accumulate per (pair, k) and
                            // integrate once after the loop.
                            let mut fast = field.clone();
                            for a in 0..g {
                                for b in 0..g {
                                    for j in 0..m {
                                        fast.set(a * n + b, j, ZERO);
                                    }
                                }
                            }
                            let nu_k = frequency(&k)?;
                            let entry = fast_fields.entry((pair, k.clone())).or_insert_with(|| ((slot, shift), nu_k, Dense::zeros(d, m)));
                            for (acc, v) in entry.2.data.iter_mut().zip(&fast.data) {
                                *acc = cadd(*acc, *v);
                            }
                            if self.slot_of(&k).is_none() && options.slave_fast_returns {
                                for (target_slot, returned) in self.slaved_returns(&k, &field)? {
                                    let modal = self.to_modal(target_slot, &returned)?;
                                    for j in 0..m {
                                        pair_sources[pair].add_exponential(small.exponents[j], modal.data[j * modal.rows..(j + 1) * modal.rows].to_vec());
                                    }
                                }
                            }
                        }
                    }
                    // Near-resonant terms: zero-initial pair sources. Their
                    // ground block returns exactly; a non-ground return is
                    // slaved when k is fast and otherwise counted (it needs a
                    // pair-source fast response; rare: resonance requires a
                    // strongly damped candidate mode).
                    for resonant_term in &resonant {
                        let (j, second) = (resonant_term.column, resonant_term.second);
                        let image = if same_charge {
                            let mut out = Dense::zeros(n, n);
                            for (row, v) in self.same_charge_ground_return(&resonant_term.unit).iter().enumerate() {
                                out.set(row / g, row % g, *v);
                            }
                            out
                        } else {
                            self.frame.apply(term, &resonant_term.unit, lap)?
                        };
                        let first_node = cadd(small.exponents[j], complex(0., shift));
                        let second_node = cadd(second, complex(0., shift));
                        let mut ground = Dense::zeros(g * g, 1);
                        for ga in 0..g {
                            for gb in 0..g {
                                ground.set(ga * g + gb, 0, image.at(ga, gb));
                            }
                        }
                        if ground.frobenius() > 0. {
                            let modal = self.to_modal(slot, &ground)?;
                            pair_sources[pair].add_pair(first_node, second_node, modal.data);
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
                            pair_sources[pair].add_pair(small.exponents[j], second, modal.data);
                        }
                    }
                }
            }
        }

        // --- Exact dressed fast response of the returns into slow charges ---
        for ((pair, _), (target, nu_k, field)) in &fast_fields {
            if field.frobenius() > 0. {
                self.add_fast_response(*target, *nu_k, &small.exponents, field, view.times, &mut pair_sources[*pair], &mut pair_direct[*pair])?;
            }
        }

        // --- Initial boundaries: slow part plus exact fast initial layer -----
        // Pairs whose boundary is negligible share one combined fast layer
        // that enters the total only.
        let largest = pair_boundary.iter().map(|b| b.frobenius()).fold(0_f64, f64::max);
        let mut shared_layer = Dense::zeros(n, n);
        let mut shared_sources = ModalSources::default();
        let mut shared_direct = vec![ZERO; nt * n];
        for pair in 0..pairs {
            let boundary = &pair_boundary[pair];
            let norm = boundary.frobenius();
            if norm == 0. {
                continue;
            }
            let layer = self.add_slow_initial(boundary, &mut pair_sources[pair])?;
            if norm >= 1e-3 * largest {
                self.add_fast_initial(&layer, view.times, &mut pair_sources[pair], &mut pair_direct[pair])?;
            } else {
                for k in 0..d {
                    shared_layer.data[k] = cadd(shared_layer.data[k], layer.data[k]);
                }
            }
        }
        if shared_layer.frobenius() > 0. {
            self.add_fast_initial(&shared_layer, view.times, &mut shared_sources, &mut shared_direct)?;
        }

        // --- Evaluate every pair and the total ------------------------------
        let mut total = vec![ZERO; nt * n];
        if shared_layer.frobenius() > 0. {
            let amplitudes = self.propagate(&shared_sources, view.times)?;
            let layer = self.read_populations(&amplitudes, view.times)?;
            for ((a, b), c) in total.iter_mut().zip(&layer).zip(&shared_direct) {
                *a = cadd(*a, cadd(*b, *c));
            }
        }
        let mut modes = Vec::with_capacity(pairs);
        let mut charges: Vec<Charge> = vec![Vec::new(); pairs];
        for (charge, &index) in &pair_index {
            charges[index] = charge.clone();
        }
        for pair in 0..pairs {
            let amplitudes = self.propagate(&pair_sources[pair], view.times)?;
            let mut contribution = self.read_populations(&amplitudes, view.times)?;
            for (a, b) in contribution.iter_mut().zip(&pair_direct[pair]) {
                *a = cadd(*a, *b);
            }
            let mut indicator = contribution.iter().map(|z| z.re.abs()).fold(0_f64, f64::max);
            for t in 0..nt {
                for group in groups {
                    let sum: f64 = group.indices.iter().filter(|&&i| i < n).map(|&i| contribution[t * n + i].re).sum();
                    indicator = indicator.max(sum.abs());
                }
            }
            for (a, b) in total.iter_mut().zip(&contribution) {
                *a = cadd(*a, *b);
            }
            modes.push(HaloModeIndicator { charge: charges[pair].clone(), indicator, residual_norm: pair_residual[pair] });
        }
        modes.sort_by(|a, b| b.indicator.total_cmp(&a.indicator));
        let estimate = finish_estimate(self, &total, view.times, groups, began)?;
        Ok(HaloEstimate { estimate, modes, halo_modes: halo.len(), near_resonant_terms, unrouted_resonant_returns, pencil_ms })
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
        let m = coefficients.cols;
        let lap = &self.lapack;
        let mut particular = Vec::with_capacity(m);
        let mut resonant = Vec::new();
        for j in 0..m {
            let shift = cadd(exponents[j], complex(0., nu_q));
            let mut x = Dense::zeros(n, n);
            for entry in 0..n * n {
                x.set(entry / n, entry % n, coefficients.at(entry, j));
            }
            if x.frobenius() == 0. {
                particular.push(x);
                continue;
            }
            match &self.dressed {
                Some(dressed) => {
                    let mut xh = dressed.to_eigen(&x, lap)?;
                    for i in 0..n {
                        for k in 0..n {
                            let c = xh.at(i, k);
                            if c.re == 0. && c.im == 0. {
                                continue;
                            }
                            let lambda = cadd(dressed.left_values[i], dressed.right_values[k]);
                            let denominator = csub(shift, lambda);
                            if cabs(denominator) >= floor {
                                xh.set(i, k, cdiv(c, denominator));
                            } else {
                                let mut unit = Dense::zeros(n, n);
                                unit.set(i, k, c);
                                let unit = dressed.from_eigen(&unit, lap)?;
                                resonant.push(ResonantTerm { column: j, unit, second: csub(lambda, complex(0., nu_q)) });
                                xh.set(i, k, ZERO);
                            }
                        }
                    }
                    particular.push(dressed.from_eigen(&xh, lap)?);
                }
                None => {
                    for a in 0..n {
                        for b in 0..n {
                            let c = x.at(a, b);
                            if c.re == 0. && c.im == 0. {
                                continue;
                            }
                            let lambda = self.diagonal_reference[a * n + b];
                            let denominator = csub(shift, lambda);
                            if cabs(denominator) >= floor {
                                x.set(a, b, cdiv(c, denominator));
                            } else {
                                let mut unit = Dense::zeros(n, n);
                                unit.set(a, b, c);
                                resonant.push(ResonantTerm { column: j, unit, second: csub(lambda, complex(0., nu_q)) });
                                x.set(a, b, ZERO);
                            }
                        }
                    }
                    particular.push(x);
                }
            }
        }
        Ok((particular, resonant))
    }

    /// Ground block (row-major g x g) of what the halo reference left out at
    /// the same charge: the recycling `R_0 X_EE` for the dressed no-jump
    /// reference, or `(L_0 - D) X` for the diagonal one.
    fn same_charge_ground_return(&self, x: &Dense) -> Vec<C> {
        let (n, g) = (self.n, self.g);
        let zero = self.frame.zero_charge();
        let term = match self.frame.term(&zero) {
            Some(t) => t,
            None => return vec![ZERO; g * g],
        };
        if self.dressed.is_some() {
            return recycle_ground(&self.frame, &term.recycling, x);
        }
        let mut out = self.frame.apply_ground_output(term, x);
        for a in 0..g {
            for b in 0..g {
                out[a * g + b] = csub(out[a * g + b], cmul(self.diagonal_reference[a * n + b], x.at(a, b)));
            }
        }
        out
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
    Ok(PopulationErrorEstimate {
        times: first.times.clone(),
        states: first.states,
        state_error,
        group_names: first.group_names.clone(),
        group_error,
        maximum_state,
        maximum_group,
        maximum: maximum_state.max(maximum_group),
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
}
