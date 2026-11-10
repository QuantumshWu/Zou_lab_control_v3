//! One constant step for the run, and the read of its error (SPEC §2.2).
//!
//! # The step
//! Every trajectory of a run advances by steps of one τ (`step::step`). The last step of an output interval is its rest
//! (the steps never cross an output time); a step that crosses the trajectory's jump threshold is redone to the root and
//! the jump made at its end (`step::jump`), the next step from there; a step the scheme refuses (`step::Refused`) is
//! taken at its half, and one refused down to the time's round-off ends the advance with its refusal -- a numerical
//! failure the solve counts and flags, never a loss. The beating grid's spacing is the step's quarter: two panels a step, one a half step
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
//! own local error is their difference times 2^p/(2^p − 1) (decide.rs `richardson`). A step on the beating grid is also
//! redone on twice its panels (`step::Step::regrid`); CF4 and Simpson are of fourth order, so the grid's local error is
//! that difference times 2⁴/(2⁴ − 1).
//!
//! A local error is an error of an output only as far as the dynamics carries it there: to first order the step's
//! error of an ensemble mean is Σ_n E[∇u(t_n, X_n)·ℓ_n], ℓ_n the local errors and u(t, x) the exact dynamics'
//! expectation of the output's quantity from the state x at the time t (SPEC §2.2). What carries a local error differs
//! between the internal state and the motion, and the read follows each (measured on a bright scene, six beams' cooling
//! in a tweezer: a plain sum of the local errors reads 17 to 21 % on a population whose values at τ, τ/2 and τ/8 agree
//! within their statistics, and grows with the run's length).
//!
//! - **The internal state** is carried by linear maps: along a trajectory's path and jumps, ψ after is a product of
//!   propagators and jump operators applied to ψ before, whatever ψ is. The read carries a second internal state S
//!   through the same maps -- each step taken once more from the trajectory's position and momentum with S in ψ's
//!   place, each jump's operator (the class and the polarization the trajectory drew) on S, renormalized as ψ is
//!   (`step::jump`) -- and adds every step's local error of ψ to it with its sign turned: S is the trajectory's
//!   internal state with its step errors taken out, to first order, and it forgets them exactly as the internal state
//!   forgets (optical pumping erases what a step did within a few jumps). The rows' error is the trajectory's rows less
//!   S's (`Drift::rows`); it is not a sum.
//! - **The motion** has the three axes' energies for its slow variables. Their local errors are accumulated per lab
//!   axis (`Mark::error`): the doubling's differences of the principal axes' energies E_i = (ħ/m)k_i²/2 + g_i y_i +
//!   λ_i y_i²/2 of the step's reference (y_i = e_i·(r − p), k_i = e_i·κ), each mapped by its direction cosine squared
//!   e_ia² -- T_a is the window mean of (ħ/m)κ_a² over one trap period (SPEC §1.7), which of a harmonic axis is its
//!   energy, while an error of the orbit's phase moves κ_a by its amplitude and that mean not at all --; and the
//!   difference between the trajectory's step and S's, the kick an internal error makes. Beside them are accumulated
//!   the kinetic energy the jumps put in per lab axis (`Mark::input`) and the hazard's integral with its local error
//!   (`Mark::hazard`, `Mark::hazard_error`): the jump-free doubling sees no recoil, and a hazard off by a share makes
//!   that share too many or too few of them. Each mark also holds the atom's own part of T (`Mark::level`, `Trail`):
//!   per lab axis the mean of its kinetic energy (ħ/m)κ_a²/2 over the trap period behind it -- T_a's summand (SPEC
//!   §1.7), half the axis's energy on a harmonic axis, and bounded and smooth in any trap (the harmonic energy about
//!   the trap's centre is not: in a crossed trap's arms and across a lattice's sites it is no energy of the atom's).
//!   The ensemble's regression of them carries the local errors to the outputs (transport.rs).
//! - **The loss energy** E_s keeps the plain sum of its local errors (`Drift::energy`): a trajectory's loss decision
//!   is judged against the sum as it stood (solve.rs) -- the safe side for a motion that forgets.
//!
//! Left out, each second order in the step's error or bounded by a read above: the shift of a jump's time and of its
//! outcome's odds by the internal error (S takes the trajectory's jumps; the hazard's share bounds the times'), the
//! internal error's own part of E_s at an output, and the motion's errors' effect on the rows.
//!
//! A step whose halves or whose carried step the scheme refuses is counted unread (`Drift::unread`), and so is a jump
//! whose operator annihilates S: the trajectory's read is then no number.

use crate::decide::richardson;
use crate::linalg::{dot3, norm_sqr};
use crate::model::{Point, STATES};
use crate::step::{attempt, jump, step, step_work, test_loss, Attempt, Dynamics, Flow, Internal, Middle, Refused, Scratch, Segment, Step, Trajectory};
use crate::transport::Mark;
use crate::units::ORDER;
use std::collections::VecDeque;

/// The grid's order in its spacing h: CF4's local error O(h⁵) and composite Simpson's, per unit time O(h⁴).
const GRID_ORDER: i32 = 4;

/// The step evaluations a read step takes (module documentation, # The read): the step, its two halves, and the carried
/// state's step.
pub const READ_STEPS: u64 = 4;

/// The panels a half step takes of the beating grid, a step twice as many (module documentation, # The step): the least
/// grid a step and its halves share.
const HALF_PANELS: usize = 1;

/// What a read trajectory holds of its step errors at a time (module documentation, # The read): the motion's mark
/// (transport.rs), the loss energy's accumulated local errors (rad/µs), the displayed rows' error -- the trajectory's
/// rows less its carried state's --, and the steps and jumps the read could not follow.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Drift {
    pub mark: Mark,
    pub energy: f64,
    pub rows: [f64; STATES],
    pub unread: f64,
}

impl Drift {
    pub const ZERO: Drift = Drift { mark: Mark::ZERO, energy: 0.0, rows: [0.0; STATES], unread: 0.0 };
}

/// A read trajectory's recent path (module documentation, # The read): per lab axis the window (µs, the axis's trap
/// period), and the steps' paths back to the longest window behind the last one, each with its ∫κ_a² dt.
#[derive(Clone, Debug, PartialEq)]
struct Trail {
    windows: [f64; 3],
    steps: VecDeque<(Segment, [f64; 3])>,
}

impl Trail {
    /// The trail a step longer.
    fn push(&mut self, segment: &Segment) {
        let end = segment.start + segment.tau;
        self.steps.push_back((segment.clone(), segment.moments(segment.start, end).1));
        let longest = self.windows.iter().fold(0.0, |a: f64, &b| a.max(b));
        while self.steps.front().is_some_and(|(s, _)| s.start + s.tau <= end - longest) {
            self.steps.pop_front();
        }
    }

    /// The mean of the kinetic energy (ħ/m)κ_a²/2 per lab axis over the axis's window behind the trail's end, the window
    /// cut where the trail begins (the trajectory's start); exact on the steps' paths (`Segment::moments`).
    fn levels(&self, hbar_over_mass: f64) -> [f64; 3] {
        let Some(end) = self.steps.back().map(|(s, _)| s.start + s.tau) else { return [0.0; 3] };
        std::array::from_fn(|a| {
            let from = end - self.windows[a];
            let (mut integral, mut time) = (0.0, 0.0);
            for (s, whole) in &self.steps {
                let (first, last) = (s.start, s.start + s.tau);
                if last <= from {
                    continue;
                }
                if first >= from {
                    integral += whole[a];
                    time += last - first;
                } else {
                    integral += s.moments(from, last).1[a];
                    time += last - from;
                }
            }
            0.5 * hbar_over_mass * integral / time
        })
    }
}

/// A trajectory's read (module documentation, # The read): what it accumulates (`Reader::drift` gives it with the rows),
/// its recent path, the carried internal state S on the trajectory's cluster -- none before the first step, where S is
/// ψ --, and the states its rows' error is read at: the trajectory's and S's at the end of the last step read, on that
/// step's cluster at its end's point. The rows' error is no sum: it is asked at the outputs alone, and made there.
#[derive(Clone, Debug, PartialEq)]
pub struct Reader {
    drift: Drift,
    trail: Trail,
    carried: Option<Internal>,
    shown: Option<(usize, Point, Internal, Internal)>,
}

impl Reader {
    /// A reader whose marks average the kinetic energy of each lab axis over `windows` (µs): the axes' trap periods.
    pub fn new(windows: [f64; 3]) -> Reader {
        Reader { drift: Drift::ZERO, trail: Trail { windows, steps: VecDeque::new() }, carried: None, shown: None }
    }
    /// What the read holds of the trajectory's step errors: what it accumulated, and the rows' error -- the rows of the
    /// trajectory's state less those of its carried state as the last step read left them (none read yet: no error).
    pub fn drift<D: Dynamics>(&self, model: &D) -> Drift {
        let mut drift = self.drift;
        if let Some((cluster, at, ours, theirs)) = &self.shown {
            let (ours, theirs) = (model.readout(*cluster, at, ours), model.readout(*cluster, at, theirs));
            drift.rows = std::array::from_fn(|q| ours[q] - theirs[q]);
        }
        drift
    }
    /// The loss energy's accumulated local errors (`Drift::energy`).
    pub fn energy(&self) -> f64 {
        self.drift.energy
    }
}

/// What the doubling compares of a state: the principal axes' energies of a reference flow about its expansion point p,
/// (ħ/m)k_i²/2 + g_i y_i + λ_i y_i²/2, y_i = e_i·(r − p), k_i = e_i·κ.
fn energies(flow: &Flow, t: &Trajectory) -> [f64; 3] {
    let d = [t.r[0] - flow.centre[0], t.r[1] - flow.centre[1], t.r[2] - flow.centre[2]];
    std::array::from_fn(|i| {
        let (y, k) = (dot3(flow.axes[i], d), dot3(flow.axes[i], t.kappa));
        0.5 * flow.hbar_over_mass * k * k + flow.gradient[i] * y + 0.5 * flow.curvatures[i] * y * y
    })
}

/// Principal axes' quantities per lab axis: each by its direction cosine squared.
fn lab(flow: &Flow, axes: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|a| (0..3).map(|i| flow.axes[i][a] * flow.axes[i][a] * axes[i]).sum())
}

/// The trajectory's physical state from a step's end, its counters kept.
fn adopt(trajectory: &mut Trajectory, end: &Trajectory) {
    let (tau, grid, steps, capped, work, assemblies) =
        (trajectory.tau, trajectory.grid, trajectory.steps, trajectory.capped, trajectory.work, trajectory.assemblies);
    *trajectory = end.clone();
    (trajectory.tau, trajectory.grid, trajectory.steps, trajectory.capped, trajectory.work, trajectory.assemblies) =
        (tau, grid, steps, capped, work, assemblies);
}

/// A step of `length` from a state on the grid a step of that length takes (module documentation, # The step), counted:
/// the flow between jumps, whatever the threshold.
fn counted<D: Dynamics>(model: &D, trajectory: &mut Trajectory, from: &Trajectory, length: f64, scratch: &mut Scratch) -> Result<Step, Refused> {
    trajectory.steps += 1;
    let s = step(model, from, length, scratch)?;
    trajectory.work += s.work;
    trajectory.assemblies += s.assemblies;
    Ok(s)
}

/// The same step tried for the trajectory's jump (`step::attempt`), counted: whole, or its crossing's time.
fn tried<D: Dynamics>(model: &D, trajectory: &mut Trajectory, from: &Trajectory, length: f64, scratch: &mut Scratch) -> Result<Result<Step, f64>, Refused> {
    trajectory.steps += 1;
    match attempt(model, from, length, scratch)? {
        Attempt::Whole(s) => {
            trajectory.work += s.work;
            trajectory.assemblies += s.assemblies;
            Ok(Ok(s))
        }
        Attempt::Crossing { root, work, assemblies } => {
            trajectory.work += work;
            trajectory.assemblies += assemblies;
            Ok(Err(root))
        }
    }
}

/// A step read (module documentation, # The read): the step `s` taken from `from` against its two halves and, where it
/// took the grid, its regrid, each by its Richardson factor; and the carried state's step. Where the scheme refuses a
/// half or the carried step, the step is counted unread and the carried state begins anew from the trajectory's.
fn read<D: Dynamics>(model: &D, trajectory: &mut Trajectory, from: &Trajectory, s: &Step, reader: &mut Reader, scratch: &mut Scratch) {
    let half = 0.5 * s.segment.tau;
    let halves = counted(model, trajectory, from, half, scratch).and_then(|first| counted(model, trajectory, &first.end, half, scratch));
    let carried = match reader.carried.take() {
        Some(psi) => counted(model, trajectory, &Trajectory { psi, ..from.clone() }, s.segment.tau, scratch).map(Some),
        None => Ok(None),
    };
    let (Ok(second), Ok(carried)) = (halves, carried) else {
        reader.drift.unread += 1.0;
        return;
    };
    let (on, whole, drift) = (s.segment.flow, &s.end, &mut reader.drift);
    let (axes, loss, size) = (energies(&on, whole), model.energy(whole.cluster, &whole.point(), &whole.psi), norm_sqr(&whole.psi));
    drift.mark.hazard += (norm_sqr(&from.psi) / size).ln();
    // The carried state after the step: S's own step where it differs from ψ, then ψ's local errors taken out.
    let mut next = match carried {
        Some(c) => {
            let kick = lab(&on, std::array::from_fn(|i| axes[i] - energies(&on, &c.end)[i]));
            for a in 0..3 {
                drift.mark.error[a] += kick[a];
            }
            c.end.psi
        }
        None => whole.psi,
    };
    // A level finer in the step or in the grid against the step taken: the finer level's parameter is half.
    let mut finer = |other: &Trajectory, factor: f64| {
        let moved = lab(&on, std::array::from_fn(|i| factor * (axes[i] - energies(&on, other)[i])));
        for a in 0..3 {
            drift.mark.error[a] += moved[a];
        }
        drift.energy += factor * (loss - model.energy(other.cluster, &other.point(), &other.psi));
        drift.mark.hazard_error += factor * (norm_sqr(&other.psi) / size).ln();
        for (x, (fine, coarse)) in next.iter_mut().zip(other.psi.iter().zip(whole.psi.iter())) {
            *x = *x + (*fine - *coarse) * factor;
        }
    };
    finer(&second.end, richardson(0.5, ORDER as f64));
    if let Some((regrid, work)) = s.regrid(model, scratch) {
        trajectory.work += work;
        finer(&regrid, richardson(0.5, GRID_ORDER as f64));
    }
    reader.shown = Some((whole.cluster, whole.point(), whole.psi, next));
    reader.carried = Some(next);
}

/// Advances a held trajectory to the time `to` (the next output time) by steps of τ, with its jumps and its loss test
/// (module documentation, # The step; the loss is tested at every step's end and after every jump, SPEC §1.7); `each`
/// takes every step's path (a jump's step is the one redone to its root, the jump at its end). A trajectory lost on the
/// way stops at its loss time. With a reader, every step taken and every jump is read into it (module documentation,
/// # The read). Where the scheme refuses a step at every length down to the time's round-off, the advance ends there
/// with the refusal; so it does where the model gives a jump no photon (`step::jump`). `scratch` is the trajectory's
/// (`step::Scratch`): its steps' memory, made once with it.
pub fn advance<D: Dynamics>(model: &D, tau: f64, trajectory: &mut Trajectory, scratch: &mut Scratch, mut reader: Option<&mut Reader>, to: f64, each: &mut dyn FnMut(&Segment)) -> Result<(), Refused> {
    let grid = |length: f64| length / (2 * 2 * HALF_PANELS) as f64;
    trajectory.tau = tau;
    while trajectory.lost.is_none() && trajectory.time < to {
        let rest = to - trajectory.time;
        // The step taken from here: τ or the interval's rest; the one to the jump's root where it crosses the threshold
        // (none where the root is the start itself); from half the length where the scheme refuses either.
        let mut length = tau.min(rest);
        let (at, taken, jumps) = loop {
            let mut from = trajectory.clone();
            from.grid = grid(length);
            let refusal = match tried(model, trajectory, &from, length, scratch) {
                Ok(Ok(whole)) => break (from, Some(whole), false),
                Ok(Err(root)) if root > 0.0 => {
                    from.grid = grid(root);
                    match counted(model, trajectory, &from, root, scratch) {
                        Ok(s) => break (from, Some(s), true),
                        Err(e) => {
                            length = root;
                            e
                        }
                    }
                }
                Ok(Err(_)) => break (from, None, true),
                Err(e) => e,
            };
            length *= 0.5;
            if !(trajectory.time + length > trajectory.time) {
                return Err(refusal);
            }
        };
        if let Some(s) = &taken {
            if let Some(reader) = reader.as_deref_mut() {
                read(model, trajectory, &at, s, reader, scratch);
                reader.trail.push(&s.segment);
                reader.drift.mark.level = reader.trail.levels(model.hbar_over_mass());
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
            match reader.as_deref_mut() {
                Some(reader) => {
                    // The jump read: the kinetic energy it puts in per lab axis; a carried state its operator
                    // annihilates is no state to carry on.
                    let (before, followed) = (trajectory.kappa, reader.carried.is_some());
                    jump(model, trajectory, &mut reader.carried)?;
                    for a in 0..3 {
                        reader.drift.mark.input[a] += 0.5 * model.hbar_over_mass() * (trajectory.kappa[a] * trajectory.kappa[a] - before[a] * before[a]);
                    }
                    if followed && reader.carried.is_none() {
                        reader.drift.unread += 1.0;
                    }
                }
                None => jump(model, trajectory, &mut None)?,
            }
        } else if length == rest {
            // The step spanned the rest of the interval: it ends on the output time, not an ulp short of it (the sum of
            // the steps' times rounds), which would leave an interval of the time's round-off to be stepped.
            trajectory.time = to;
        }
    }
    Ok(())
}

/// The counted work of one step of τ on the path through a cluster's middle at a momentum and a velocity (µm/µs), and the
/// middles a step asks of the model: what `advance` adds to `Trajectory::work` and `Trajectory::assemblies` per step,
/// priced without taking it and without a middle of its own (`step::step_work`). The model's price of a middle is its
/// `assembly_work`.
pub fn advance_work<M: Middle>(middle: &M, kappa: [f64; 3], velocity: [f64; 3], tau: f64) -> (f64, u64) {
    let (work, _) = step_work(middle, kappa, velocity, tau, tau / (2 * 2 * HALF_PANELS) as f64);
    (work, 1)
}
