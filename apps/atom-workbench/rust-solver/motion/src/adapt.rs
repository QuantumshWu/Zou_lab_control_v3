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
//! The tolerance is per unit time, so the split's error over its tolerance goes as τ^p and the grid's as φ⁴ (φ the grid's
//! phase, `step::Trajectory::grid`): after a rejection τ ← τ (share/error)^{1/p} and φ ← φ (share/error)^{1/4}, which land
//! on the tolerance under the error model; after an acceptance the exponents 1/(p + 1) and 1/5, which approach it from
//! below without crossing it. Both from the orders; no safety factor.
//!
//! The shares: the split's error per unit time goes as τ^p at a work ∝ 1/τ (the assemblies and the reference), the
//! grid's as φ⁴ at a work ∝ 1/φ (its CF4 nodes, whatever τ). The least work at a total error e minimizes W_s/τ + W_g/φ
//! subject to C τ^p + D φ⁴ = e: the Lagrange condition puts the errors in the ratio of the works over the orders,
//! e_s : e_g = W_s/p : W_g/4, measured on each step's counted work (`step::Step::work`, `Dynamics::assembly_work`).
//! Without a grid the split takes the whole tolerance.
//!
//! # Jumps and the loss
//! The doubling compares the jump-free flows. Accepted, where the halves' propagation crosses the trajectory's jump
//! threshold, the half that holds the root is redone to it -- a step no longer than the half, so within the accepted
//! error -- and the jump made at its end (`step::jump`); the rest of the step is dropped, the next starts from the state
//! the jump left. The loss is tested at every step's end and after every jump (SPEC §1.7).

use crate::step::{jump, step, Dynamics, Segment, Step, Trajectory};
use crate::units::{MOST_STEPS, ORDER};

/// The local errors allowed per µs of a step (SPEC §2.2: the error ≤ s·ε·τ/T_run of each output's scale), which the solve
/// computes from the targets ε, the step share s (`units::STEP_SHARE`), the run's length T_run and the pilot's scales.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tolerances {
    /// Per lab axis, κ's error (1/µm) per µs.
    pub momentum: [f64; 3],
    /// E_s's error (rad/µs) per µs.
    pub energy: f64,
    /// The displayed rows' largest error per µs.
    pub population: f64,
}

/// The grid's order in its phase: CF4's local error O(h⁵) and composite Simpson's, per unit time O(φ⁴).
const GRID_ORDER: i32 = 4;

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
/// the steps go on unchecked at the last length and `Trajectory::capped` records from when (SPEC §2.2). A step that cannot
/// be made (`step::Refused`) or whose error is not a number is redone at half its length, the doubling's more accurate
/// level.
pub fn advance<D: Dynamics>(model: &D, tolerances: &Tolerances, trajectory: &mut Trajectory, to: f64, each: &mut dyn FnMut(&Segment)) {
    let p = ORDER as f64;
    while trajectory.lost.is_none() && trajectory.time < to {
        let tau = trajectory.tau.min(to - trajectory.time);
        let from = trajectory.clone();
        if trajectory.steps >= MOST_STEPS {
            trajectory.capped.get_or_insert(trajectory.time);
            match counted(model, trajectory, &from, tau) {
                Some(s) => {
                    take(model, trajectory, &s, each);
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
        // The split's error, the grid's, and their shares by the counted work.
        let split = differences(model, tolerances, &full.end, &second.end, tau).map(|d| d / ((1 << ORDER) - 1) as f64);
        let grid = match full.regrid() {
            Some((finer, work)) => {
                trajectory.work += work;
                let g = (1 << GRID_ORDER) as f64;
                differences(model, tolerances, &full.end, &finer, tau).map(|d| d * g / (g - 1.0))
            }
            None => [0.0; 5],
        };
        let split_work = full.work - full.grid_work + full.assemblies as f64 * model.assembly_work();
        let share = if full.grid_work > 0.0 { split_work / p / (split_work / p + full.grid_work / GRID_ORDER as f64) } else { 1.0 };
        let total = (0..5).map(|q| split[q] + grid[q]).fold(0.0, f64::max);
        let (e_split, e_grid) = (split.iter().copied().fold(0.0, f64::max), grid.iter().copied().fold(0.0, f64::max));
        if !total.is_finite() {
            trajectory.tau = 0.5 * tau;
            continue;
        }
        let accepted = total <= 1.0;
        let (time_exponent, grid_exponent) = if accepted { (1.0 / (p + 1.0), 1.0 / (GRID_ORDER as f64 + 1.0)) } else { (1.0 / p, 1.0 / GRID_ORDER as f64) };
        trajectory.tau = tau * (share / e_split).powf(time_exponent);
        if full.grid_work > 0.0 {
            trajectory.grid *= ((1.0 - share) / e_grid).powf(grid_exponent);
        }
        if accepted && take(model, trajectory, &first, each) {
            take(model, trajectory, &second, each);
        }
    }
}
