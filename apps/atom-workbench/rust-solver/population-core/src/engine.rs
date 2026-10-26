//! Finite-time population engine: grows the trial space and the Fourier
//! domain until the population error estimate meets the requested tolerance.
//!
//! INTERIM CONTROL POLICY: until the dual-weighted residual estimator is
//! integrated, stopping uses successive-candidate population changes (basis)
//! and the change against the previous Fourier order (domain).

use super::*;
use super::directions::{shift_invert, InnerSolve};
use super::reduced::ReducedModel;
use super::runtime_core::RuntimeCore;
use super::trial::TrialSpace;
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
    /// Rational poles (1/us), geometric between `slow_pole_factor / duration`
    /// and `fast_pole`, at most `maximum_poles`; independent of the sampling.
    pub slow_pole_factor: f64,
    pub fast_pole: f64,
    pub maximum_poles: usize,
    /// Development only: explicit pole list overriding the rule above.
    pub poles: Option<Vec<f64>>,
    pub first_columns: usize,
    pub block_columns: usize,
    pub maximum_columns: usize,
    pub maximum_order: usize,
    pub wall_seconds: f64,
    pub inner: InnerSolve,
    /// Development only: build exactly (order, columns) and return.
    pub fixed: Option<(usize, usize)>,
}

impl Default for FiniteOptions {
    fn default() -> Self {
        Self {
            population_tolerance: 0.005,
            slow_pole_factor: 200.,
            fast_pole: 20.,
            maximum_poles: 2,
            poles: None,
            first_columns: 16,
            block_columns: 8,
            maximum_columns: 400,
            maximum_order: 6,
            wall_seconds: 120.,
            inner: InnerSolve { maximum_steps: 12, relative_residual: 1e-3 },
            fixed: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Checkpoint {
    pub order: usize,
    pub modes: usize,
    pub columns: usize,
    pub basis_change: Option<f64>,
    pub domain_change: Option<f64>,
    pub action: &'static str,
    pub elapsed_ms: f64,
}

#[derive(Clone, Debug)]
pub struct FiniteDiagnostics {
    pub estimated_error: f64,
    pub projection_charge: f64,
    pub final_order: usize,
    pub final_modes: usize,
    pub final_columns: usize,
    pub image_columns: usize,
    pub operator_applications: usize,
    pub peak_trial_bytes: usize,
    pub maximum_imaginary_population: f64,
    pub elapsed_ms: f64,
    /// Milliseconds: inner directions, orthogonalization, images, projections,
    /// readout append, transfers, reduced model, readout evaluation.
    pub timing: [f64; 8],
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

/// Rational poles for a time horizon: geometric between the slowest scale
/// that matters on `[0, duration]` and the fast optical/decay scale.
pub fn rational_poles(duration: f64, options: &FiniteOptions) -> Vec<f64> {
    if let Some(poles) = &options.poles {
        return poles.clone();
    }
    let slow = options.slow_pole_factor / duration;
    let fast = options.fast_pole;
    if slow >= fast / 2. {
        return vec![slow];
    }
    let count = (((fast / slow).ln() / 4f64.ln()).ceil() as usize + 1).clamp(2, options.maximum_poles.max(2));
    let ratio = (fast / slow).powf(1. / (count - 1) as f64);
    (0..count).map(|i| fast / ratio.powi(i as i32)).collect()
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
            let (directions, report) = shift_invert(&mut space.op, &rhs, sources.len(), &pole, inner)?;
            self.directions_ms += clock.elapsed().as_secs_f64() * 1000.;
            self.operator_applications += report.operator_applications;
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

pub fn solve(
    mut core: RuntimeCore, times: &[f64], groups: Vec<RequestedGroup>, options: FiniteOptions,
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
    let (axes, steps) = {
        let phases = &core.prepared().primitives().hilbert.source.phases;
        (phases.axes.len(), phases.drive_indices.clone())
    };
    let lapack = Arc::clone(core.lapack());
    let mut order = 1;
    let mut space = TrialSpace::new(&mut core, &support_domain(axes, &steps, order)?, &poles)?;
    let mut chain = Chain::new(poles.len());
    chain.grow(&mut space, options.first_columns.saturating_sub(1), options.inner)?;
    if let Some((fixed_order, columns)) = options.fixed {
        while order < fixed_order {
            order += 1;
            space = space.transfer(&mut core, &support_domain(axes, &steps, order)?)?;
        }
        while space.columns() < columns {
            let want = columns - space.columns();
            if chain.grow(&mut space, want, options.inner)? == 0 {
                break;
            }
        }
        let model = ReducedModel::build(&space)?;
        if std::env::var_os("ATOM_EXP_SPECTRUM").is_some() {
            let (re, im, _) = lapack.eigen(&model.generator, model.m)?;
            let mut pairs: Vec<(f64, f64)> = re.into_iter().zip(im).collect();
            pairs.sort_by(|a, b| b.0.total_cmp(&a.0));
            eprintln!("reduced spectrum m={} top: {:?}", model.m, &pairs[..pairs.len().min(8)]);
        }
        let trajectory = model.trajectory(&lapack, times)?;
        let candidate = space.readout().evaluate(&lapack, &trajectory, times)?;
        return Ok(FiniteSolution {
            populations: candidate.values,
            diagnostics: FiniteDiagnostics {
                estimated_error: 0.,
                projection_charge: 0.,
                final_order: order,
                final_modes: space.modes().len(),
                final_columns: space.columns(),
                image_columns: space.image_columns(),
                operator_applications: chain.operator_applications,
                peak_trial_bytes: space.bytes(),
                maximum_imaginary_population: candidate.maximum_imaginary,
                elapsed_ms: started.elapsed().as_secs_f64() * 1000.,
                timing: [chain.directions_ms, space.timing.orthogonalize_ms, space.timing.images_ms,
                    space.timing.projections_ms, space.timing.readout_ms, space.timing.transfer_ms, 0., 0.],
                checkpoints: vec![],
            },
        });
    }
    let mut checkpoints = Vec::new();
    let (mut reduced_ms, mut evaluate_ms) = (0., 0.);
    let mut previous: Option<Vec<f64>> = None;
    let mut previous_domain: Option<Vec<f64>> = None;
    let mut peak_bytes = space.bytes();
    loop {
        if started.elapsed().as_secs_f64() > options.wall_seconds {
            return Err(format!(
                "Population work limit reached after {:.1} s at Fourier order {order} with {} columns; the tolerance was not relaxed",
                started.elapsed().as_secs_f64(),
                space.columns()
            ));
        }
        let clock = Instant::now();
        let model = ReducedModel::build(&space)?;
        let trajectory = model.trajectory(&lapack, times)?;
        reduced_ms += clock.elapsed().as_secs_f64() * 1000.;
        let clock = Instant::now();
        let candidate = space.readout().evaluate(&lapack, &trajectory, times)?;
        evaluate_ms += clock.elapsed().as_secs_f64() * 1000.;
        let populations = candidate.values;
        let basis_change = previous.as_ref().map(|p| population_change(p, &populations, n, &groups));
        let domain_change = previous_domain.as_ref().map(|p| population_change(p, &populations, n, &groups));
        let exact_domain = support_domain(axes, &steps, order + 1)?.len() == space.modes().len();
        let mut checkpoint = Checkpoint {
            order,
            modes: space.modes().len(),
            columns: space.columns(),
            basis_change,
            domain_change,
            action: "",
            elapsed_ms: started.elapsed().as_secs_f64() * 1000.,
        };
        let basis_resolved = basis_change.is_some_and(|b| b <= 0.25 * epsilon);
        if basis_resolved {
            let b = basis_change.unwrap_or(0.);
            let domain_error = if exact_domain { Some(0.) } else { domain_change };
            if let Some(d) = domain_error.filter(|&d| b + d <= 0.9 * epsilon) {
                let (projected, charge) = constrain_populations(&populations, n, &groups)?;
                if b + d + charge <= epsilon {
                    checkpoint.action = "accept";
                    checkpoints.push(checkpoint);
                    return Ok(FiniteSolution {
                        populations: projected,
                        diagnostics: FiniteDiagnostics {
                            estimated_error: b + d + charge,
                            projection_charge: charge,
                            final_order: order,
                            final_modes: space.modes().len(),
                            final_columns: space.columns(),
                            image_columns: space.image_columns(),
                            operator_applications: chain.operator_applications,
                            peak_trial_bytes: peak_bytes,
                            maximum_imaginary_population: candidate.maximum_imaginary,
                            elapsed_ms: started.elapsed().as_secs_f64() * 1000.,
                            timing: [
                                chain.directions_ms,
                                space.timing.orthogonalize_ms,
                                space.timing.images_ms,
                                space.timing.projections_ms,
                                space.timing.readout_ms,
                                space.timing.transfer_ms,
                                reduced_ms,
                                evaluate_ms,
                            ],
                            checkpoints,
                        },
                    });
                }
            }
            if !exact_domain && order < options.maximum_order {
                checkpoint.action = "expand-domain";
                checkpoints.push(checkpoint);
                order += 1;
                previous_domain = Some(populations.clone());
                previous = Some(populations);
                space = space.transfer(&mut core, &support_domain(axes, &steps, order)?)?;
                chain.grow(&mut space, options.block_columns, options.inner)?;
                peak_bytes = peak_bytes.max(space.bytes());
                continue;
            }
        }
        if space.columns() + options.block_columns > options.maximum_columns {
            return Err(format!(
                "Trial space limit of {} columns reached at Fourier order {order}; the tolerance was not relaxed",
                options.maximum_columns
            ));
        }
        checkpoint.action = "refine-basis";
        checkpoints.push(checkpoint);
        previous = Some(populations);
        if chain.grow(&mut space, options.block_columns, options.inner)? == 0 {
            return Err("The trial space cannot be extended further".into());
        }
        peak_bytes = peak_bytes.max(space.bytes());
    }
}
