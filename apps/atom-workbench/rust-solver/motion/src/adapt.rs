//! Per-trajectory step control by step doubling against tolerances per unit time (SPEC §2.2).
//!
//! # The rule
//! From the same state one step of τ and two of τ/2 (`step::step`). The scheme's local error is O(τ^{p+1}), p = 2
//! (`units::ORDER`), so the two halves' error is their difference over 2^p − 1 = 3. A step that took the beating grid
//! (`step::Step::regrid`) also has the grid's error, which the doubling cannot see (both its steps take the grid's
//! spacing): the full step redone on twice the panels differs from it by (1 − 2⁻⁴) of its grid error (CF4 and Simpson
//! are of fourth order), so the grid's error is 2⁴/(2⁴ − 1) of that difference -- the halves' grid no coarser. The two
//! add into one local error per displayed quantity: per lab axis the momentum, the loss energy E_s, the displayed rows'
//! largest difference (`Tolerances`); within the tolerance of every quantity the halves are taken, else τ is redone.
//!
//! # The next τ and the next grid
//! The doubling's three steps take one spacing: h′ = τ/4m, m = ⌈τ/4h⌉, the full step 2m panels and each half m
//! (`step::panels`), so that their grid errors per unit time are one and their difference is the split's alone.
//!
//! The tolerance is per unit time, so over it the split's error goes as C τ^p and the grid's as D h^q (h the grid's
//! spacing, `step::Trajectory::grid`; q = 4), at a work per unit time A/τ + B/h: A the step's own work (the assemblies,
//! `Dynamics::assembly_work`, the reference, the forms), B/h the grid's (its CF4 nodes, whatever τ). Read on the doubling,
//! C = e_s/τ^p, D = e_g/h′^q, A and B from the counted work (`step::Step::work`). The least work within the tolerance,
//! min A/τ + B/h subject to Cτ^p + Dh^q = 1, is the Lagrange point τ^{p+1} = A/(λpC), h^{q+1} = B/(λqD) with λ the root of
//! a λ^{−p/(p+1)} + b λ^{−q/(q+1)} = 1, a = C^{1/(p+1)}(A/p)^{p/(p+1)}, b = D^{1/(q+1)}(B/q)^{q/(q+1)} (`optimum`); without a
//! grid τ* = C^{−1/p}, the tolerance's own step. After a rejection the next doubling takes (τ*, h*), which lands on the
//! tolerance under the error model; after an acceptance it moves toward them by the exponents p/(p+1) and q/(q+1) of
//! the ratio -- τ (τ*/τ)^{p/(p+1)} = τ (tolerance/error)^{1/(p+1)} for one variable --, approaching the tolerance from
//! below without crossing it. Either way by one factor of two at most, the range the doubling validated (`level`). All
//! from the orders, the doubling's levels and the counted work; no safety factor, no share.
//!
//! # Jumps and the loss
//! The doubling compares the jump-free flows. Accepted, where the halves' propagation crosses the trajectory's jump
//! threshold, the half that holds the root is redone to it -- a step no longer than the half, so within the accepted
//! error -- and the jump made at its end (`step::jump`); the rest of the step is dropped, the next starts from the state
//! the jump left. The loss is tested at every step's end and after every jump (SPEC §1.7).

use crate::model::Generator;
use crate::step::{jump, panels, step, step_work, Dynamics, Segment, Step, Trajectory};
use crate::units::{MOST_STEPS, ORDER};

/// The local errors allowed per µs of a step (SPEC §2.2: the error ≤ s·ε·τ/T_run of each output's scale), which the solve
/// computes from the targets ε, the step share s (`units::STEP_SHARE`), the run's length T_run and the pilot's scales.
/// The errors add over the run as the steps' local errors do, to at most s·ε of each scale.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tolerances {
    /// Per lab axis, κ's error (1/µm) per µs: T_a ∝ ⟨κ_a²⟩ moves by 2δκ/κ_T of itself for an error δκ on the axis's thermal
    /// scale κ_T, so T's share is s·ε_T κ_T/(2 T_run).
    pub momentum: [f64; 3],
    /// E_s's error (rad/µs) per µs: on the thermal energy k_BT_s, s·ε_T k_BT_s/T_run.
    pub energy: f64,
    /// The displayed rows' largest error per µs: s·ε_P/T_run.
    pub population: f64,
}

/// The grid's order in its spacing h: CF4's local error O(h⁵) and composite Simpson's, per unit time O(h⁴).
const GRID_ORDER: i32 = 4;

/// A step's or a spacing's change within the range the doubling validated: it compares τ with τ/2 (h with h/2), one
/// factor of two, so the error model is read on that range and a change beyond one factor of two either way extrapolates
/// it -- a wildly non-asymptotic step (an output interval of a 1 MHz trap) read errors of 10³⁰ of the tolerance, the
/// model asked τ = 10⁻¹³ µs, and there the doubling's difference was the energy's round-off, ever above a tolerance ∝ τ: a
/// spiral to zero. One level at a time, each next doubling validates the range adjacent to the last.
fn level(ratio: f64) -> f64 {
    if ratio.is_nan() { 0.5 } else { ratio.clamp(0.5, 2.0) }
}

/// The least-work step and spacing (module documentation, # The next τ and the next grid): (τ*, h*) minimizing
/// A/τ + B/h subject to Cτ^p + Dh^q = 1, p = `units::ORDER`, q = `GRID_ORDER`. λ solves a λ^{−α} + b λ^{−β} = 1
/// (α = p/(p+1), β = q/(q+1)), decreasing in ln λ, bracketed by where either term alone is 1 and ½, by Newton on ln λ kept
/// in the bracket until it no longer moves. A source without error (C or D zero) takes an unbounded step or spacing.
fn optimum(a_work: f64, b_work: f64, c: f64, d: f64) -> (f64, f64) {
    let (p, q) = (ORDER as f64, GRID_ORDER as f64);
    let (alpha, beta) = (p / (p + 1.0), q / (q + 1.0));
    let a = c.powf(1.0 / (p + 1.0)) * (a_work / p).powf(alpha);
    let b = d.powf(1.0 / (q + 1.0)) * (b_work / q).powf(beta);
    let terms = [(a, alpha), (b, beta)];
    let live = terms.iter().filter(|(w, _)| *w > 0.0);
    let lo = live.clone().map(|(w, e)| w.ln() / e).fold(f64::INFINITY, f64::min);
    let hi = live.map(|(w, e)| (2.0 * w).ln() / e).fold(f64::NEG_INFINITY, f64::max);
    if !(lo.is_finite() && hi.is_finite()) {
        return (f64::INFINITY, f64::INFINITY);
    }
    let g = |mu: f64| terms.iter().map(|(w, e)| w * (-e * mu).exp()).sum::<f64>() - 1.0;
    let slope = |mu: f64| -terms.iter().map(|(w, e)| e * w * (-e * mu).exp()).sum::<f64>();
    let (mut lo, mut hi, mut mu) = (lo, hi, 0.5 * (lo + hi));
    for _ in 0..64 {
        let value = g(mu);
        if value > 0.0 {
            lo = mu;
        } else {
            hi = mu;
        }
        let newton = mu - value / slope(mu);
        let next = if newton > lo && newton < hi { newton } else { 0.5 * (lo + hi) };
        if next == mu {
            break;
        }
        mu = next;
    }
    let lambda = mu.exp();
    ((a_work / (lambda * p * c)).powf(1.0 / (p + 1.0)), (b_work / (lambda * q * d)).powf(1.0 / (q + 1.0)))
}

/// The differences of the displayed quantities of two states at one time, each over its tolerance per µs times τ: per
/// lab axis the momentum, E_s, the displayed rows; [κ_x, κ_y, κ_z, E_s, rows].
fn differences<D: Dynamics>(model: &D, tolerances: &Tolerances, a: &Trajectory, b: &Trajectory, tau: f64) -> [f64; 5] {
    let (pa, pb) = (a.point(), b.point());
    let energy = (model.energy(a.cluster, &pa, &a.psi) - model.energy(b.cluster, &pb, &b.psi)).abs();
    let (ra, rb) = (model.readout(a.cluster, &pa, &a.psi), model.readout(b.cluster, &pb, &b.psi));
    let rows = ra.iter().zip(&rb).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
    let t = tau.abs();
    [
        (a.kappa[0] - b.kappa[0]).abs() / (tolerances.momentum[0] * t),
        (a.kappa[1] - b.kappa[1]).abs() / (tolerances.momentum[1] * t),
        (a.kappa[2] - b.kappa[2]).abs() / (tolerances.momentum[2] * t),
        energy / (tolerances.energy * t),
        rows / (tolerances.population * t),
    ]
}

/// The trajectory's physical state from a step's end, its step control's counters kept.
fn adopt(trajectory: &mut Trajectory, end: &Trajectory) {
    let (tau, grid, steps, capped, work, assemblies) =
        (trajectory.tau, trajectory.grid, trajectory.steps, trajectory.capped, trajectory.work, trajectory.assemblies);
    *trajectory = end.clone();
    (trajectory.tau, trajectory.grid, trajectory.steps, trajectory.capped, trajectory.work, trajectory.assemblies) =
        (tau, grid, steps, capped, work, assemblies);
}

/// A step from the trajectory's state, counted.
fn counted<D: Dynamics>(model: &D, trajectory: &mut Trajectory, from: &Trajectory, tau: f64) -> Option<Step> {
    trajectory.steps += 1;
    let s = step(model, from, tau).ok()?;
    trajectory.work += s.work;
    trajectory.assemblies += s.assemblies;
    Some(s)
}

/// The loss test at the trajectory's state (SPEC §1.7): its time once lost.
fn test_loss<D: Dynamics>(model: &D, trajectory: &mut Trajectory) {
    if trajectory.lost.is_none() && model.lost(trajectory.cluster, &trajectory.point(), &trajectory.psi) {
        trajectory.lost = Some(trajectory.time);
    }
}

/// Takes an accepted step of the trajectory (module documentation, # Jumps and the loss): where it crosses the threshold,
/// the step from the same start redone to the root and the jump at its end; else the step. False where the trajectory
/// stops here: lost, or jumped (the rest of a doubled step dropped).
fn take<D: Dynamics>(model: &D, trajectory: &mut Trajectory, s: &Step, each: &mut dyn FnMut(&Segment)) -> bool {
    match s.crossing {
        Some(root) => {
            let from = trajectory.clone();
            if root > 0.0 {
                if let Some(redone) = counted(model, trajectory, &from, root) {
                    each(&redone.segment);
                    adopt(trajectory, &redone.end);
                    test_loss(model, trajectory);
                }
            }
            if trajectory.lost.is_none() {
                jump(model, trajectory);
            }
            false
        }
        None => {
            each(&s.segment);
            adopt(trajectory, &s.end);
            test_loss(model, trajectory);
            trajectory.lost.is_none()
        }
    }
}

/// Advances a held trajectory to the time `to` (the next output time; the steps never cross it) by doubled steps, its
/// jumps and its loss test (module documentation); `each` takes every accepted step's path (a jump's step is the one
/// redone to its root, the jump at its end). A trajectory lost on the way stops at its loss time. Past `units::MOST_STEPS`
/// the trajectory ends as usual (SPEC §2.2): unchecked steps, each the rest of the output interval (halved while it cannot
/// be made), so that its work stays bounded, and `Trajectory::capped` records from when -- that span is flagged. A step
/// that cannot be made (`step::Refused`) or whose error is not a number is redone at half its length, the doubling's more
/// accurate level.
pub fn advance<D: Dynamics>(model: &D, tolerances: &Tolerances, trajectory: &mut Trajectory, to: f64, each: &mut dyn FnMut(&Segment)) {
    let p = ORDER as f64;
    let assembly = model.assembly_work();
    while trajectory.lost.is_none() && trajectory.time < to {
        let tau = trajectory.tau.min(to - trajectory.time);
        // One spacing for the doubling's three steps: τ/4m, no wider than the grid's (the trajectory's own grid moves only
        // where a step takes one).
        let spacing = tau / (4 * panels(0.5 * tau, trajectory.grid)) as f64;
        let mut from = trajectory.clone();
        from.grid = spacing;
        if trajectory.steps >= MOST_STEPS {
            trajectory.capped.get_or_insert(trajectory.time);
            match counted(model, trajectory, &from, tau) {
                Some(s) => {
                    take(model, trajectory, &s, each);
                    trajectory.tau = f64::INFINITY;
                }
                None => trajectory.tau = 0.5 * tau,
            }
            continue;
        }
        let doubled = counted(model, trajectory, &from, tau).and_then(|full| {
            let first = counted(model, trajectory, &from, 0.5 * tau)?;
            let second = counted(model, trajectory, &first.end, 0.5 * tau)?;
            Some((full, first, second))
        });
        let Some((full, first, second)) = doubled else {
            trajectory.tau = 0.5 * tau;
            continue;
        };
        // The split's error and the grid's, each over the tolerance; the eigen-propagation's rounding (relative to ‖ψ‖, a
        // population's twice that) with the grid's, on the rows.
        let split = differences(model, tolerances, &full.end, &second.end, tau).map(|d| d / ((1 << ORDER) - 1) as f64);
        let mut grid = match full.regrid() {
            Some((finer, work)) => {
                trajectory.work += work;
                let g = (1 << GRID_ORDER) as f64;
                differences(model, tolerances, &full.end, &finer, tau).map(|d| d * g / (g - 1.0))
            }
            None => [0.0; 5],
        };
        grid[4] += 2.0 * [&full, &first, &second].iter().map(|s| s.rounding).fold(0.0, f64::max) / (tolerances.population * tau);
        let total = (0..5).map(|q| split[q] + grid[q]).fold(0.0, f64::max);
        let (e_split, e_grid) = (split.iter().copied().fold(0.0, f64::max), grid.iter().copied().fold(0.0, f64::max));
        if !total.is_finite() {
            trajectory.tau = 0.5 * tau;
            continue;
        }
        // The least-work (τ*, h*) on the error model read here (`optimum`), and the move toward it.
        let a_work = full.work - full.grid_work + full.assemblies as f64 * assembly;
        let b_work = full.grid_work * spacing / tau;
        let (best_tau, best_grid) = optimum(a_work, b_work, e_split / tau.powi(ORDER), e_grid / spacing.powi(GRID_ORDER));
        let accepted = total <= 1.0;
        let q = GRID_ORDER as f64;
        let (time_exponent, grid_exponent) = if accepted { (p / (p + 1.0), q / (q + 1.0)) } else { (1.0, 1.0) };
        trajectory.tau = tau * level((best_tau / tau).powf(time_exponent));
        if full.grid_work > 0.0 {
            trajectory.grid = spacing * level((best_grid / spacing).powf(grid_exponent));
        }
        if accepted && take(model, trajectory, &first, each) {
            take(model, trajectory, &second, each);
        }
    }
}

/// The integrator's counted work of one doubled advance of τ through a cluster's generator at a point, the path at the
/// velocity v (µm/µs), and the generator assemblies it asks: what `advance` adds to `Trajectory::work` and
/// `Trajectory::assemblies` per accepted doubling, priced without taking it (`step::step_work`) -- the full step, its two
/// halves and, on a beating generator, the grid's doubling of the full step; two assemblies a step. The model's price
/// is the assemblies times its `assembly_work`.
pub fn advance_work(generator: &Generator, waves: &[[f64; 3]], velocity: [f64; 3], tau: f64, grid: f64) -> (f64, u64) {
    let (full, regrid) = step_work(generator, waves, velocity, tau, grid);
    let (half, _) = step_work(generator, waves, velocity, 0.5 * tau, grid);
    (full + 2.0 * half + regrid, 6)
}
