//! One constant step for the run, and the read of its error (SPEC §2.2).
//!
//! # The step
//! Every trajectory of a run advances by steps of one τ (`step::step`). The last step of an output interval is its rest
//! (the steps never cross an output time); a step that crosses the trajectory's jump threshold is redone to the root and
//! the jump made at its end (`step::jump`), the next step from there; a step the scheme refuses (`step::Refused`) is
//! taken at its half. The beating grid's spacing is the step's quarter: two panels a step, one a half step
//! (`step::panels`), so that a step and its halves take one spacing.
//!
//! τ is one for the run because the splitting is symplectic in its conservative part and its own inverse: at a constant
//! step the energy of a trap's motion errs by a bounded amount (measured on the reference traps R1: no drift over 10³
//! periods), where a step chosen from the last step's error drifts. The steps are then the run's length over τ and its
//! jumps.
//!
//! # The read
//! Where a trajectory is read (the pilot's, solve.rs), every step it takes is doubled: from the same state the step
//! and two of its halves, the jump-free flows. The scheme's local error is O(τ^{p+1}), p = `units::ORDER`, so the step's
//! own error is their difference times 2^p/(2^p − 1) (Richardson). A step on the beating grid is also redone on twice
//! its panels (`step::Step::regrid`); CF4 and Simpson are of fourth order, so the grid's error is that difference times
//! 2⁴/(2⁴ − 1). The two add, signed, over the steps the trajectory takes, per displayed quantity (`Drift`):
//! - per lab axis a, the errors of the principal axes' energies E_i = (ħ/m)k_i²/2 + g_i y_i + λ_i y_i²/2 of the step's
//!   reference (y_i = e_i·(r − p), k_i = e_i·κ), each mapped by its direction cosine squared e_ia²: T_a is the window
//!   mean of (ħ/m)κ_a² over one trap period (SPEC §1.7), which of a harmonic axis is its energy, while an error of the
//!   orbit's phase moves κ_a by its amplitude and that mean not at all;
//! - the loss energy E_s;
//! - the displayed rows.
//!
//! The ensemble's mean of a quantity's accumulated error at an output is the step's error there (solve.rs). Local errors
//! add along a trajectory wherever the motion does not forget them: every jump cuts a step short, and the cut breaks the
//! regular sequence that bounds a trap's energy error (R3: at ω τ = 2.5, some 40 jumps an atom raise T by half in 400 µs,
//! and the sum reads it). Where the motion does forget them -- a trap without scattering, whose error stays bounded; a
//! damped motion -- the sum reads more than the error: the safe side.

use crate::linalg::dot3;
use crate::model::{Generator, STATES};
use crate::step::{jump, step, step_work, Dynamics, Flow, Segment, Step, Trajectory};
use crate::units::ORDER;

/// The grid's order in its spacing h: CF4's local error O(h⁵) and composite Simpson's, per unit time O(h⁴).
const GRID_ORDER: i32 = 4;

/// The panels a half step takes of the beating grid, a step twice as many (module documentation, # The step): the least
/// grid a step and its halves share.
const HALF_PANELS: usize = 1;

/// A trajectory's accumulated signed step errors (module documentation, # The read): per lab axis the axes' energies'
/// (rad/µs), the loss energy's (rad/µs), the displayed rows'.
#[derive(Clone, Debug, PartialEq)]
pub struct Drift {
    pub axes: [f64; 3],
    pub energy: f64,
    pub rows: [f64; STATES],
}

impl Drift {
    pub const ZERO: Drift = Drift { axes: [0.0; 3], energy: 0.0, rows: [0.0; STATES] };
}

/// What the doubling compares of a state: per principal axis of a reference flow (the step's) the axis's energy, the
/// loss energy E_s, the displayed rows.
struct Marks {
    axes: [f64; 3],
    energy: f64,
    rows: [f64; STATES],
}

fn marks<D: Dynamics>(model: &D, flow: &Flow, t: &Trajectory) -> Marks {
    let at = t.point();
    let d = [t.r[0] - flow.centre[0], t.r[1] - flow.centre[1], t.r[2] - flow.centre[2]];
    let axes = std::array::from_fn(|i| {
        let (y, k) = (dot3(flow.axes[i], d), dot3(flow.axes[i], t.kappa));
        0.5 * flow.hbar_over_mass * k * k + flow.gradient[i] * y + 0.5 * flow.curvatures[i] * y * y
    });
    Marks { axes, energy: model.energy(t.cluster, &at, &t.psi), rows: model.readout(t.cluster, &at, &t.psi) }
}

/// The trajectory's physical state from a step's end, its counters kept.
fn adopt(trajectory: &mut Trajectory, end: &Trajectory) {
    let (tau, grid, steps, capped, work, assemblies) =
        (trajectory.tau, trajectory.grid, trajectory.steps, trajectory.capped, trajectory.work, trajectory.assemblies);
    *trajectory = end.clone();
    (trajectory.tau, trajectory.grid, trajectory.steps, trajectory.capped, trajectory.work, trajectory.assemblies) =
        (tau, grid, steps, capped, work, assemblies);
}

/// A step of `length` from a state on the grid a step of that length takes (module documentation, # The step), counted.
fn counted<D: Dynamics>(model: &D, trajectory: &mut Trajectory, from: &Trajectory, length: f64) -> Option<Step> {
    trajectory.steps += 1;
    let s = step(model, from, length).ok()?;
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

/// A step's error added to the drift (module documentation, # The read): the step `s` taken from `from` less its two
/// halves, and less its regrid where it took the grid, each by its Richardson factor. Halves the scheme refuses where
/// it made the step are not met (the middle's fixed point contracts faster at half the step); the step is then unread.
fn read<D: Dynamics>(model: &D, trajectory: &mut Trajectory, from: &Trajectory, s: &Step, drift: &mut Drift) {
    let half = 0.5 * s.segment.tau;
    let Some(first) = counted(model, trajectory, from, half) else { return };
    let Some(second) = counted(model, trajectory, &first.end, half) else { return };
    let on = s.segment.flow;
    let whole = marks(model, &on, &s.end);
    let mut add = |other: &Marks, factor: f64| {
        for i in 0..3 {
            let e = factor * (whole.axes[i] - other.axes[i]);
            for a in 0..3 {
                drift.axes[a] += on.axes[i][a] * on.axes[i][a] * e;
            }
        }
        drift.energy += factor * (whole.energy - other.energy);
        for q in 0..STATES {
            drift.rows[q] += factor * (whole.rows[q] - other.rows[q]);
        }
    };
    let richardson = |order: i32| (1 << order) as f64 / ((1 << order) - 1) as f64;
    add(&marks(model, &on, &second.end), richardson(ORDER));
    if let Some((finer, work)) = s.regrid() {
        trajectory.work += work;
        add(&marks(model, &on, &finer), richardson(GRID_ORDER));
    }
}

/// Advances a held trajectory to the time `to` (the next output time) by steps of τ, with its jumps and its loss test
/// (module documentation, # The step; the loss is tested at every step's end and after every jump, SPEC §1.7); `each`
/// takes every step's path (a jump's step is the one redone to its root, the jump at its end). A trajectory lost on the
/// way stops at its loss time. With a drift, every step taken is read into it (module documentation, # The read). A
/// state that leaves the floating-point range is refused at every length: the atom is lost there.
pub fn advance<D: Dynamics>(model: &D, tau: f64, trajectory: &mut Trajectory, mut drift: Option<&mut Drift>, to: f64, each: &mut dyn FnMut(&Segment)) {
    let grid = |length: f64| length / (2 * 2 * HALF_PANELS) as f64;
    trajectory.tau = tau;
    while trajectory.lost.is_none() && trajectory.time < to {
        let rest = to - trajectory.time;
        // The step from here: τ or the interval's rest, at its half where the scheme refuses it.
        let mut length = tau.min(rest);
        let made = loop {
            let mut from = trajectory.clone();
            from.grid = grid(length);
            match counted(model, trajectory, &from, length) {
                Some(s) => break Some((from, s)),
                None if trajectory.time + 0.5 * length > trajectory.time => length *= 0.5,
                None => break None,
            }
        };
        let Some((from, whole)) = made else {
            trajectory.lost = Some(trajectory.time);
            break;
        };
        // The step taken: the whole one, or the one to the jump's root (none where the root is the start itself).
        let crossing = whole.crossing;
        let (taken, jumps) = match crossing {
            Some(root) if root > 0.0 => {
                let mut at = from;
                at.grid = grid(root);
                (counted(model, trajectory, &at, root).map(|s| (at, s)), true)
            }
            Some(_) => (None, true),
            None => (Some((from, whole)), false),
        };
        if let Some((at, s)) = &taken {
            if let Some(drift) = drift.as_deref_mut() {
                read(model, trajectory, at, s, drift);
            }
            each(&s.segment);
            adopt(trajectory, &s.end);
            trajectory.grid = at.grid;
            test_loss(model, trajectory);
        }
        if trajectory.lost.is_some() {
            break;
        }
        if jumps {
            jump(model, trajectory);
        } else if length == rest {
            // The step spanned the rest of the interval: it ends on the output time, not an ulp short of it (the sum of
            // the steps' times rounds), which would leave an interval of the time's round-off to be stepped.
            trajectory.time = to;
        }
    }
}

/// The integrator's counted work of one step of τ through a cluster's generator at a point, the path at the velocity v
/// (µm/µs), and the generator assemblies it asks: what `advance` adds to `Trajectory::work` and `Trajectory::assemblies`
/// per step, priced without taking it (`step::step_work`). The model's price is the assemblies times its `assembly_work`.
pub fn advance_work(generator: &Generator, waves: &[[f64; 3]], velocity: [f64; 3], tau: f64) -> (f64, u64) {
    let (work, _) = step_work(generator, waves, velocity, tau, tau / (2 * 2 * HALF_PANELS) as f64);
    (work, 2)
}
