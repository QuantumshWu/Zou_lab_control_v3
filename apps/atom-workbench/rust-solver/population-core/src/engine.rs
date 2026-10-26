//! Finite-time population engine: grows the trial space and the Fourier
//! domain until the population error estimate meets the requested tolerance.
//!
//! Error control (`estimated`): the slow-manifold dual-weighted residual
//! estimate of the basis error is checked at every checkpoint; once it fits
//! the budget, the halo estimate of the Fourier truncation is added and the
//! candidate is accepted when `kappa * |basis + halo| + projection <= eps`.
//! Halo indicators select the few Fourier modes worth adding. If the slow
//! dual cannot be prepared, `successive` compares successive candidates and
//! Fourier orders instead (conservative fallback).

use super::*;
use super::directions::{shift_invert, InnerSolve};
use super::error_estimate::{combined_estimate, BasisOptions, CandidateView, HaloModeIndicator, HaloOptions, PopulationGroup};
use super::reduced::ReducedModel;
use super::runtime_core::RuntimeCore;
use super::slow_dual::{SlowDual, SlowDualOptions};
use super::trial::TrialSpace;
use crate::parallel::Execution;
use std::collections::VecDeque;

/// A named set of state indices whose summed population is also checked.
#[derive(Clone, Debug)]
pub struct RequestedGroup {
    pub name: String,
    pub indices: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct FiniteOptions {
    /// Absolute population tolerance on the probability scale.
    pub population_tolerance: f64,
    /// Rational poles (1/us): `fast_pole`, plus `slow_pole_factor / duration`
    /// when the horizon is long enough for it to be a separate time scale.
    /// Both are independent of the number of output samples.
    pub slow_pole_factor: f64,
    pub fast_pole: f64,
    pub first_columns: usize,
    pub block_columns: usize,
    pub maximum_columns: usize,
    pub maximum_order: usize,
    pub wall_seconds: f64,
    pub inner: InnerSolve,
    /// Multiplies the error estimate before comparing it with the tolerance;
    /// calibrated estimate/true ratios on the benchmarks are 0.75-1.37.
    pub safety_factor: f64,
}

impl Default for FiniteOptions {
    fn default() -> Self {
        Self {
            population_tolerance: 0.005,
            slow_pole_factor: 200.,
            fast_pole: 20.,
            first_columns: 16,
            block_columns: 8,
            maximum_columns: 400,
            maximum_order: 6,
            wall_seconds: 120.,
            inner: InnerSolve { maximum_steps: 24, relative_residual: 1e-5 },
            safety_factor: 1.5,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Checkpoint {
    pub modes: usize,
    pub columns: usize,
    /// Basis error estimate (or successive change in the fallback).
    pub basis_estimate: Option<f64>,
    /// Fourier-truncation estimate (or change against the previous order).
    pub halo_estimate: Option<f64>,
    pub action: &'static str,
    pub elapsed_ms: f64,
}

/// Wall time per engine stage, in milliseconds.
#[derive(Clone, Debug, Default)]
pub struct EngineTiming {
    pub directions: f64,
    pub orthogonalize: f64,
    pub images: f64,
    pub projections: f64,
    pub readout_append: f64,
    pub transfer: f64,
    pub reduced: f64,
    pub readout_evaluate: f64,
    pub dual_prepare: f64,
    pub basis_estimate: f64,
    pub halo_estimate: f64,
}

#[derive(Clone, Debug)]
pub struct FiniteDiagnostics {
    /// `kappa * estimate + projection_charge`, the quantity compared with the
    /// tolerance (the fallback reports its successive changes instead).
    pub estimated_error: f64,
    /// The unscaled population error estimate of the returned candidate.
    pub raw_estimate: f64,
    /// The unsigned part of `raw_estimate` that charges what the estimate
    /// bounds instead of integrating (slow charges only weak couplings reach,
    /// fully clustered near-resonant halo returns).
    pub estimate_margin: f64,
    /// Near-resonant halo return components of the accepted estimate that
    /// were bounded and charged instead of integrated.
    pub bounded_resonant_components: usize,
    pub projection_charge: f64,
    /// "dual-weighted-residual" or "successive-change".
    pub error_control: &'static str,
    pub final_modes: usize,
    pub final_columns: usize,
    pub image_columns: usize,
    pub operator_applications: usize,
    pub peak_trial_bytes: usize,
    pub maximum_imaginary_population: f64,
    pub elapsed_ms: f64,
    pub timing: EngineTiming,
    pub checkpoints: Vec<Checkpoint>,
}

pub struct FiniteSolution {
    /// Probability populations, row-major `[time, state]`.
    pub populations: Vec<f64>,
    pub diagnostics: FiniteDiagnostics,
}

/// Graph ball of the declared drive charges (both signs) of radius `order`.
pub fn support_domain(axes: usize, steps: &[Charge], order: usize) -> Result<Vec<Charge>, String> {
    if steps.iter().any(|q| q.len() != axes) {
        return Err("Drive-charge axes differ".into());
    }
    let mut result: BTreeSet<Charge> = BTreeSet::from([vec![0i32; axes]]);
    let mut directions: BTreeSet<Charge> = BTreeSet::new();
    for step in steps {
        directions.insert(step.clone());
        directions.insert(step.iter().map(|x| -x).collect());
    }
    for _ in 0..order {
        let old: Vec<_> = result.iter().cloned().collect();
        for q in old {
            for step in &directions {
                result.insert(q.iter().zip(step).map(|(a, b)| a + b).collect());
            }
        }
    }
    Ok(result.into_iter().collect())
}

/// Euclidean projection of every population row onto the probability
/// simplex. Returns the projected rows and the largest change of any state
/// or group population, which is charged to the error budget.
pub fn constrain_populations(raw: &[f64], n: usize, groups: &[RequestedGroup]) -> Result<(Vec<f64>, f64), String> {
    let mut output = Vec::with_capacity(raw.len());
    let mut maximum = 0_f64;
    for row in raw.chunks_exact(n) {
        let mut sorted = row.to_vec();
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
        let projected: Vec<f64> = row.iter().map(|x| (x - threshold).max(0.)).collect();
        for (a, b) in projected.iter().zip(row) {
            maximum = maximum.max((a - b).abs());
        }
        for group in groups {
            let change: f64 = group.indices.iter().map(|&i| projected[i] - row[i]).sum();
            maximum = maximum.max(change.abs());
        }
        output.extend(projected);
    }
    Ok((output, maximum))
}

/// Largest change of any state or group population between two candidates.
fn population_change(a: &[f64], b: &[f64], n: usize, groups: &[RequestedGroup]) -> f64 {
    let mut maximum = 0_f64;
    for (x, y) in a.chunks_exact(n).zip(b.chunks_exact(n)) {
        for (p, q) in x.iter().zip(y) {
            maximum = maximum.max((p - q).abs());
        }
        for group in groups {
            let change: f64 = group.indices.iter().map(|&i| x[i] - y[i]).sum();
            maximum = maximum.max(change.abs());
        }
    }
    maximum
}

/// Rational poles for a time horizon. The fast pole resolves the optical and
/// Rabi transients (rates ~20-100 /us); a slow pole ~200/T is added only when
/// it is a separate scale (T > ~40 us), e.g. for millisecond pumping dynamics.
pub fn rational_poles(duration: f64, options: &FiniteOptions) -> Vec<f64> {
    let slow = options.slow_pole_factor / duration;
    if slow < options.fast_pole / 4. {
        vec![options.fast_pole, slow]
    } else {
        vec![options.fast_pole]
    }
}

/// Parallel shift-invert chains, one per rational pole: each column spawns
/// `(pole E - F)^{-1} E q` for the pole of its chain. By partial fractions the
/// union of the chains is the rational Krylov space of all poles.
struct Chain {
    frontier: VecDeque<(usize, usize)>,
    operator_applications: usize,
    directions_ms: f64,
}

const CHAIN_BLOCK: usize = 4;

impl Chain {
    fn new(poles: usize) -> Self {
        Self { frontier: (0..poles).map(|p| (0, p)).collect(), operator_applications: 0, directions_ms: 0. }
    }

    fn grow(&mut self, space: &mut TrialSpace, count: usize, inner: InnerSolve) -> Result<usize, String> {
        let rows = space.rows();
        let poles = space.op.poles().len();
        let mut added = 0;
        while added < count {
            if self.frontier.is_empty() {
                // Every chain broke down: restart all poles from the newest column.
                let newest = space.columns() - 1;
                self.frontier.extend((0..poles).map(|p| (newest, p)));
            }
            let sources: Vec<(usize, usize)> = (0..CHAIN_BLOCK.min(self.frontier.len()))
                .filter_map(|_| self.frontier.pop_front())
                .collect();
            let mut rhs = Vec::with_capacity(rows * sources.len());
            for &(j, _) in &sources {
                rhs.extend_from_slice(&space.eq()[j * rows..(j + 1) * rows]);
            }
            let pole: Vec<usize> = sources.iter().map(|&(_, p)| p).collect();
            let clock = Instant::now();
            let (directions, applications) = shift_invert(&mut space.op, &rhs, sources.len(), &pole, inner)?;
            self.directions_ms += clock.elapsed().as_secs_f64() * 1000.;
            self.operator_applications += applications;
            let before = space.columns();
            let kept = space.append(directions, sources.len())?;
            for (offset, &source) in kept.iter().enumerate() {
                self.frontier.push_back((before + offset, sources[source].1));
            }
            if kept.is_empty() && self.frontier.is_empty() {
                break;
            }
            added += kept.len();
        }
        Ok(added)
    }
}

/// Bookkeeping shared by the error-control policies.
struct Run {
    core: RuntimeCore,
    times: Vec<f64>,
    groups: Vec<RequestedGroup>,
    options: FiniteOptions,
    epsilon: f64,
    n: usize,
    axes: usize,
    steps: Vec<Charge>,
    lapack: Arc<Lapack>,
    space: Option<TrialSpace>,
    chain: Chain,
    started: Instant,
    checkpoints: Vec<Checkpoint>,
    timing: EngineTiming,
    peak_bytes: usize,
}

/// The kappa-scaled basis estimate must fit this share of the tolerance
/// before the (more expensive) halo estimate is evaluated.
const BASIS_SHARE: f64 = 0.6;
/// After adding halo modes, the excluded indicators may still sum to this
/// share of the tolerance (divided by kappa).
const HALO_REMAINDER_SHARE: f64 = 0.25;
/// Signed mode pairs added per domain step at most.
const MAXIMUM_PAIRS_PER_STEP: usize = 24;
/// The screening basis estimate: the full one without the drive-hop slaving
/// of fast charges, at 25-55% of its cost. Measured screen/full ratios are
/// 0.97-1.00 far from acceptance and 0.76-0.98 near it, never above 1.001, so
/// a screen that fails the basis gate is followed by more columns without the
/// full estimate; acceptance always uses the full estimate.
const SCREEN: BasisOptions = BasisOptions { fast_response: true, slave_fast_charges: false };

fn ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.
}

impl Run {
    fn space(&self) -> &TrialSpace {
        self.space.as_ref().expect("trial space is present between steps")
    }

    fn deadline(&self) -> Result<(), String> {
        if self.started.elapsed().as_secs_f64() > self.options.wall_seconds {
            return Err(format!(
                "Population work limit reached after {:.1} s with {} Fourier modes and {} columns; the tolerance was not relaxed",
                self.started.elapsed().as_secs_f64(),
                self.space().modes().len(),
                self.space().columns()
            ));
        }
        Ok(())
    }

    fn checkpoint(&mut self, basis: Option<f64>, halo: Option<f64>, action: &'static str) {
        let (modes, columns) = (self.space().modes().len(), self.space().columns());
        self.checkpoints.push(Checkpoint {
            modes,
            columns,
            basis_estimate: basis,
            halo_estimate: halo,
            action,
            elapsed_ms: ms(self.started),
        });
    }

    /// Adds `count` chain columns. Returns false when no independent direction
    /// is left; that is accepted only when the space is numerically invariant
    /// (vanishing reduced residual).
    fn grow(&mut self, count: usize, model: &ReducedModel) -> Result<bool, String> {
        let space = self.space.as_mut().expect("trial space");
        if space.columns() + count > self.options.maximum_columns {
            return Err(format!(
                "Trial space limit of {} columns reached with {} Fourier modes; the tolerance was not relaxed",
                self.options.maximum_columns,
                space.modes().len()
            ));
        }
        let added = self.chain.grow(space, count, self.options.inner)?;
        self.peak_bytes = self.peak_bytes.max(space.bytes());
        if added > 0 {
            return Ok(true);
        }
        let residual = space.residual(&model.generator)?;
        let scale = space.fq().iter().map(|x| x * x).sum::<f64>().sqrt().max(f64::MIN_POSITIVE);
        let size = residual.iter().map(|x| x * x).sum::<f64>().sqrt();
        if size > 1e-9 * scale {
            return Err("The trial space cannot be extended further".into());
        }
        Ok(false)
    }

    fn transfer(&mut self, modes: &[Charge]) -> Result<(), String> {
        let space = self.space.take().expect("trial space");
        let moved = space.transfer(&mut self.core, modes)?;
        self.peak_bytes = self.peak_bytes.max(moved.bytes());
        self.space = Some(moved);
        Ok(())
    }

    fn populations(&mut self, model: &ReducedModel) -> Result<super::readout::Populations, String> {
        let clock = Instant::now();
        let trajectory = model.trajectory(&self.lapack, &self.times)?;
        self.timing.reduced += ms(clock);
        let clock = Instant::now();
        let populations = self.space().readout().evaluate(&self.lapack, &trajectory, &self.times)?;
        self.timing.readout_evaluate += ms(clock);
        Ok(populations)
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    fn finish(
        mut self, populations: Vec<f64>, maximum_imaginary: f64, raw_estimate: f64, scaled: f64, charge: f64,
        (estimate_margin, bounded_resonant_components): (f64, usize), error_control: &'static str,
    ) -> FiniteSolution {
        let space = self.space.take().expect("trial space");
        let measured = &space.timing;
        self.timing.directions = self.chain.directions_ms;
        self.timing.orthogonalize = measured.orthogonalize_ms;
        self.timing.images = measured.images_ms;
        self.timing.projections = measured.projections_ms;
        self.timing.readout_append = measured.readout_ms;
        self.timing.transfer = measured.transfer_ms;
        FiniteSolution {
            populations,
            diagnostics: FiniteDiagnostics {
                estimated_error: scaled + charge,
                raw_estimate,
                estimate_margin,
                bounded_resonant_components,
                projection_charge: charge,
                error_control,
                final_modes: space.modes().len(),
                final_columns: space.columns(),
                image_columns: space.image_columns(),
                operator_applications: self.chain.operator_applications,
                peak_trial_bytes: self.peak_bytes,
                maximum_imaginary_population: maximum_imaginary,
                elapsed_ms: ms(self.started),
                timing: self.timing,
                checkpoints: self.checkpoints,
            },
        }
    }

    /// Columns to add so that the basis estimate is predicted to reach
    /// `target`, from its geometric decay between the last two checkpoints.
    fn predicted_block(&self, history: &[(usize, f64)], target: f64) -> usize {
        let poles = self.space().op.poles().len().max(1);
        let default = self.options.block_columns;
        let block = match history {
            [.., (m0, e0), (m1, e1)] if m1 > m0 && *e1 < *e0 && *e1 > 0. => {
                let rate = (e1 / e0).ln() / (m1 - m0) as f64;
                let needed = (target / e1).ln() / rate;
                if needed.is_finite() && needed > 0. { needed.ceil() as usize } else { default }
            }
            _ => default,
        };
        block.clamp(poles, 2 * default).div_ceil(poles) * poles
    }

    /// Signed halo modes (both members of each pair) to add: the largest
    /// indicators until the excluded ones sum to at most `remainder`.
    fn halo_modes(&self, indicators: &[HaloModeIndicator], remainder: f64) -> Vec<Charge> {
        let mut excluded: f64 = indicators.iter().map(|m| m.indicator).sum();
        let retained = self.space().modes();
        let mut out = Vec::new();
        for mode in indicators.iter().take(MAXIMUM_PAIRS_PER_STEP) {
            if excluded <= remainder {
                break;
            }
            excluded -= mode.indicator;
            if retained.contains(&mode.charge) {
                continue;
            }
            let negative: Charge = mode.charge.iter().map(|x| -x).collect();
            out.push(mode.charge.clone());
            if negative != mode.charge {
                out.push(negative);
            }
        }
        out
    }

    /// Error control by the slow-manifold dual-weighted residual estimate.
    fn estimated(mut self, dual: SlowDual) -> Result<FiniteSolution, String> {
        let kappa = self.options.safety_factor;
        let epsilon = self.epsilon;
        // Aim a little below the halo threshold: basis estimates are cheap, so
        // a few more checkpoints cost less than overshooting the basis.
        let target = 0.8 * BASIS_SHARE * epsilon / kappa;
        let groups: Vec<PopulationGroup> =
            self.groups.iter().map(|g| PopulationGroup { name: g.name.clone(), indices: g.indices.clone() }).collect();
        let mut history: Vec<(usize, f64)> = Vec::new();
        loop {
            self.deadline()?;
            let clock = Instant::now();
            let model = ReducedModel::build(self.space())?;
            self.timing.reduced += ms(clock);
            // Err: the basis gate failed, with the screen (or full) estimate.
            let estimates = {
                let space = self.space.as_ref().expect("trial space");
                let view = CandidateView {
                    map: space.op.map(),
                    q: space.q(),
                    eq: space.eq(),
                    fq: space.fq(),
                    columns: space.columns(),
                    generator: &model.generator,
                    initial: &model.initial,
                    times: &self.times,
                };
                let clock = Instant::now();
                let screen = dual.basis_estimate_with(&view, &groups, SCREEN)?.maximum;
                let basis = if kappa * screen > BASIS_SHARE * epsilon { Err(screen) } else { Ok(dual.basis_estimate(&view, &groups)?) };
                self.timing.basis_estimate += ms(clock);
                match basis {
                    Ok(basis) if kappa * basis.maximum <= BASIS_SHARE * epsilon => {
                        let clock = Instant::now();
                        let halo = dual.halo_estimate(self.core.prepared(), &view, &groups, HaloOptions::default())?;
                        self.timing.halo_estimate += ms(clock);
                        Ok((basis, halo))
                    }
                    Ok(basis) => Err(basis.maximum),
                    Err(screen) => Err(screen),
                }
            };
            let (basis, halo) = match estimates {
                Ok(pair) => pair,
                Err(basis) => {
                    history.push((self.space().columns(), basis));
                    self.checkpoint(Some(basis), None, "refine-basis");
                    let block = self.predicted_block(&history, target);
                    self.grow(block, &model)?;
                    continue;
                }
            };
            history.push((self.space().columns(), basis.maximum));
            let total = combined_estimate(&basis, &halo.estimate)?;
            if kappa * total.maximum <= epsilon {
                let candidate = self.populations(&model)?;
                let (projected, charge) = constrain_populations(&candidate.values, self.n, &self.groups)?;
                if kappa * total.maximum + charge <= epsilon {
                    self.checkpoint(Some(basis.maximum), Some(halo.estimate.maximum), "accept");
                    let scaled = kappa * total.maximum;
                    return Ok(self.finish(
                        projected,
                        candidate.maximum_imaginary,
                        total.maximum,
                        scaled,
                        charge,
                        (total.margin, halo.unrouted_resonant_returns),
                        "dual-weighted-residual",
                    ));
                }
            }
            if kappa * halo.estimate.maximum > 0.5 * epsilon && halo.estimate.maximum >= basis.maximum {
                let added = self.halo_modes(&halo.modes, HALO_REMAINDER_SHARE * epsilon / kappa);
                if !added.is_empty() {
                    self.checkpoint(Some(basis.maximum), Some(halo.estimate.maximum), "add-modes");
                    let mut modes = self.space().modes().to_vec();
                    modes.extend(added);
                    modes.sort();
                    self.transfer(&modes)?;
                    history.clear();
                    self.grow(self.options.block_columns, &model)?;
                    continue;
                }
            }
            self.checkpoint(Some(basis.maximum), Some(halo.estimate.maximum), "refine-basis");
            let block = self.predicted_block(&history, target);
            self.grow(block, &model)?;
        }
    }

    /// Fallback without the slow dual: successive candidate changes (basis)
    /// and the change against the previous graph-ball order (domain).
    fn successive(mut self) -> Result<FiniteSolution, String> {
        let epsilon = self.epsilon;
        let mut order = 1;
        let mut previous: Option<Vec<f64>> = None;
        let mut previous_domain: Option<Vec<f64>> = None;
        let mut invariant = false;
        loop {
            self.deadline()?;
            let model = ReducedModel::build(self.space())?;
            let candidate = self.populations(&model)?;
            let populations = candidate.values;
            let basis_change = if invariant {
                Some(0.)
            } else {
                previous.as_ref().map(|p| population_change(p, &populations, self.n, &self.groups))
            };
            let domain_change = previous_domain.as_ref().map(|p| population_change(p, &populations, self.n, &self.groups));
            let next = support_domain(self.axes, &self.steps, order + 1)?;
            let exact_domain = next.len() == self.space().modes().len();
            if let Some(b) = basis_change.filter(|&b| b <= 0.25 * epsilon) {
                let domain_error = if exact_domain { Some(0.) } else { domain_change };
                if let Some(d) = domain_error.filter(|&d| b + d <= 0.9 * epsilon) {
                    let (projected, charge) = constrain_populations(&populations, self.n, &self.groups)?;
                    if b + d + charge <= epsilon {
                        self.checkpoint(basis_change, domain_change, "accept");
                        return Ok(self.finish(projected, candidate.maximum_imaginary, b + d, b + d, charge, (0., 0), "successive-change"));
                    }
                }
                if !exact_domain && order < self.options.maximum_order {
                    self.checkpoint(basis_change, domain_change, "expand-domain");
                    order += 1;
                    invariant = false;
                    previous_domain = Some(populations.clone());
                    previous = Some(populations);
                    self.transfer(&next)?;
                    self.grow(self.options.block_columns, &model)?;
                    continue;
                }
            }
            self.checkpoint(basis_change, domain_change, "refine-basis");
            previous = Some(populations);
            invariant = !self.grow(self.options.block_columns, &model)?;
        }
    }
}

pub fn solve(core: RuntimeCore, times: &[f64], groups: Vec<RequestedGroup>, options: FiniteOptions) -> Result<FiniteSolution, String> {
    solve_with(core, times, groups, options, None)
}

/// `solve` with the request's slow dual already prepared (`Some(Ok)`), or
/// its preparation already failed (`Some(Err)`); `None` prepares it here,
/// while the first chain columns are computed.
pub fn solve_with(
    mut core: RuntimeCore, times: &[f64], groups: Vec<RequestedGroup>, options: FiniteOptions, given: Option<Result<SlowDual, String>>,
) -> Result<FiniteSolution, String> {
    let started = Instant::now();
    let epsilon = options.population_tolerance;
    if times.len() < 2 || times[0] != 0. || times.windows(2).any(|w| !(w[1] > w[0])) {
        return Err("Invalid finite population time grid".into());
    }
    if !(epsilon > 0. && epsilon <= 0.02) {
        return Err("populationTolerance must be in (0, 0.02]".into());
    }
    let duration = times[times.len() - 1];
    let poles = rational_poles(duration, &options);
    let n = core.prepared().primitives().hilbert.source.n;
    // Start on the significant Fourier steps; the halo estimate covers all of
    // them and adds the modes its indicators ask for.
    let steps = core.fourier_steps()?;
    let lapack = Arc::clone(core.lapack());
    let mut space = TrialSpace::new(&mut core, &support_domain(steps.axes, &steps.domain, 1)?, &poles)?;
    let mut chain = Chain::new(poles.len());
    // The slow dual depends only on the model: prepare it while the first
    // chain columns are computed.
    let (dual, dual_ms) = match given {
        Some(dual) => {
            chain.grow(&mut space, options.first_columns.saturating_sub(1), options.inner)?;
            (dual, 0.)
        }
        None => {
            let prepared = core.prepared();
            let ((dual, dual_ms), grown) = Execution::new()?.join(
                || {
                    let clock = Instant::now();
                    (SlowDual::prepare(prepared, SlowDualOptions::default()), ms(clock))
                },
                || chain.grow(&mut space, options.first_columns.saturating_sub(1), options.inner),
            );
            grown?;
            (dual, dual_ms)
        }
    };
    let peak_bytes = space.bytes();
    let mut run = Run {
        core,
        times: times.to_vec(),
        groups,
        options,
        epsilon,
        n,
        axes: steps.axes,
        steps: steps.all,
        lapack,
        space: Some(space),
        chain,
        started,
        checkpoints: Vec::new(),
        timing: EngineTiming::default(),
        peak_bytes,
    };
    run.timing.dual_prepare = dual_ms;
    match dual {
        Ok(dual) => run.estimated(dual),
        Err(_) => run.successive(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::b0_inverse::P0Policy;
    use super::super::coordinates::InitialView;

    /// Exact populations of the static fixture model by a dense exponential of
    /// the real 2n^2-dimensional Lindblad superoperator.
    fn exact_populations(times: &[f64]) -> Vec<f64> {
        let n = 4;
        let mut h = vec![0.; 16];
        for (i, v) in [0., 0.6, 3., 4.].iter().enumerate() {
            h[i * n + i] = *v;
        }
        for (i, j, v) in [(0, 2, 0.1), (1, 3, 0.2), (0, 1, 0.02)] {
            h[i * n + j] = v;
            h[j * n + i] = v;
        }
        let mut c = vec![0.; 16];
        c[2] = 0.7;
        c[3] = 0.1;
        c[6] = 0.2;
        c[7] = 0.8;
        let mm = |a: &[f64], b: &[f64]| -> Vec<f64> {
            let mut out = vec![0.; 16];
            for i in 0..n {
                for j in 0..n {
                    for k in 0..n {
                        out[i * n + j] += a[i * n + k] * b[k * n + j];
                    }
                }
            }
            out
        };
        let ct: Vec<f64> = (0..16).map(|i| c[(i % n) * n + i / n]).collect();
        let ctc = mm(&ct, &c);
        // Real vector: [Re rho (16), Im rho (16)]; L acts linearly (H, C real).
        let apply = |x: &[f64]| -> Vec<f64> {
            let (re, im) = (&x[..16], &x[16..]);
            let mut out = vec![0.; 32];
            let (hr, rh, hi, ih) = (mm(&h, re), mm(re, &h), mm(&h, im), mm(im, &h));
            let (cre, cim) = (mm(&mm(&c, re), &ct), mm(&mm(&c, im), &ct));
            let (ar, ra, ai, ia) = (mm(&ctc, re), mm(re, &ctc), mm(&ctc, im), mm(im, &ctc));
            for k in 0..16 {
                // -i[H, rho]: real part = [H, Im rho], imaginary part = -[H, Re rho].
                out[k] = (hi[k] - ih[k]) + cre[k] - 0.5 * (ar[k] + ra[k]);
                out[16 + k] = -(hr[k] - rh[k]) + cim[k] - 0.5 * (ai[k] + ia[k]);
            }
            out
        };
        let mut generator = vec![0.; 32 * 32];
        for j in 0..32 {
            let mut e = vec![0.; 32];
            e[j] = 1.;
            let column = apply(&e);
            generator[j * 32..(j + 1) * 32].copy_from_slice(&column);
        }
        let lapack = Lapack::load().unwrap();
        // propagator[row + col*32]: rho0 = |0><0| is real coordinate 0.
        let mut exact = Vec::new();
        for &t in times {
            let propagator = crate::dense_expm::exponential(&lapack, &generator, 32, t).unwrap();
            for i in 0..n {
                exact.push(propagator[i * n + i]);
            }
        }
        exact
    }

    #[test]
    fn engine_matches_exact_static_evolution() {
        let owned = super::super::tests::fixture();
        let core = RuntimeCore::from_owned(owned, InitialView { index: 0, real: None, imag: None }, P0Policy::EigenPreferred)
            .unwrap();
        let times: Vec<f64> = (0..21).map(|i| 4. * i as f64 / 20.).collect();
        let options = FiniteOptions { population_tolerance: 1e-4, ..FiniteOptions::default() };
        let solution = solve(core, &times, vec![], options).unwrap();
        let exact = exact_populations(&times);
        let error = solution.populations.iter().zip(&exact).map(|(a, b)| (a - b).abs()).fold(0_f64, f64::max);
        assert!(error < 1e-4, "static fixture population error {error}");
    }
}
