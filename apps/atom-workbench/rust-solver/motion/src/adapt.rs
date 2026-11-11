//! One constant step for the run (SPEC §2.2).
//!
//! # The step
//! Every trajectory of a run advances by steps of one τ (`step::step`). The last step of an output interval is its rest
//! (the steps never cross an output time); the step before a jump ends at the jump (# The jumps' times), the next step
//! from there; a step the scheme refuses (`step::Refused`) is taken at its half, and one refused down to the time's
//! round-off ends the advance with its refusal -- a numerical failure the solve counts and flags, never a loss. A step
//! takes its generator's change along the path through its own three nodes, its middle and its ends, whatever its
//! length (step.rs, # The kick): what that leaves is an error of the step's length like the splitting's, and the run's
//! companion at twice the step measures both at the outputs (solve.rs, # The step). The generator's own frequencies
//! are no matter of the step's length: they are integrated exactly.
//!
//! τ is one for the run because the splitting is symplectic in its conservative part and its own inverse: at a constant
//! step the energy of a trap's motion errs by a bounded amount (measured on the reference traps R1: no drift over 10³
//! periods), where a step chosen from the last step's error drifts. The steps are then the run's length over τ and its
//! jumps.
//!
//! # The jumps' times
//! A jump is made at the end of a step whose own propagation reaches the trajectory's threshold there (SPEC §1.6): the
//! step that is taken sets the jump's time, so the time's error is that step's own, a part of the step's error like
//! every other. A step's norm is its own curve -- its generator sits at its middle, half its length ahead on the straight
//! path through it --, so where a longer step's norm crosses the threshold is not when a shorter step reaches it. Timed
//! by a step of the nominal length and made at the end of a shorter one, the jumps took the hazard of another phase of
//! the orbit: at ω τ = 3 a held atom at a standing wave's antinode lost its recoil heating along the wave (R10). So the
//! step to a jump is the fixed point of "the step of the length L
//! reaches the threshold at s(L)", L = s(L):
//! - the length tried first is the time to the threshold at the rate the trajectory has just had -- the last taken
//!   step's fall of ln‖ψ‖² over its length (`Trajectory::rate`) --, no longer than τ or the interval's rest: where jumps
//!   are frequent no step of the nominal length is tried and thrown away;
//! - a step tried says where its own propagation reaches the threshold (`step::Attempt`): nowhere -- it is taken, and
//!   no jump made; at its end, within the tolerance -- it is taken and the jump made at its end; elsewhere -- before
//!   its end, or beyond it, where its propagation is followed on its own path for at most its length again and not
//!   past the nominal end -- it only proposes that length, and the step of that length is tried. Each look at a length
//!   is the step of that length;
//! - the lengths found too long and too short bracket the fixed point: a proposal outside the bracket is replaced by the
//!   bracket's middle, and where the bracket is narrower than the tolerance the step tried is the step taken -- the
//!   search ends by its tolerance or by its bracket, never by a count.
//! The tolerance is `units::JUMP_TIMING` of the step's length (derived there: the second order of the step's share of
//! the tighter target): the time's distance from the fixed point is held at the second order of what the step may
//! take.
//!
//! # The Brownian path
//! The light a trajectory scatters diffuses its momentum (SPEC §1.6; step.rs, # The absorbed light's momentum) along
//! the trajectory's own Brownian path, three independent standard ones. The n-th step taken draws its part of the path
//! from its own stream (`random::Stream::diffusion(seed, index, n)`): two vectors of normal deviates, the path's
//! increment and its first moment about the step's middle at whatever length the step takes (`step::Brownian::over`)
//! -- a step tried at several lengths for its jump takes the same deviates at each, and its norm does not depend on
//! them. A trajectory is a function of its seed, its index and the run's step: the same trajectory at twice the step
//! (solve.rs, # The step) starts from the same atom and draws from the same streams by their keys.

use crate::linalg::norm_sqr;
use crate::random::Stream;
use crate::step::{jump, step_work, test_loss, Attempt, Brownian, Dynamics, Middle, Midway, Refused, Scratch, Segment, Trajectory};
use crate::units::JUMP_TIMING;

/// The trajectory's physical state from the end of a step taken, its counters kept and the step counted among those
/// taken: the next step's part of the Brownian path is the next one's (module documentation, # The Brownian path).
fn adopt(trajectory: &mut Trajectory, end: &Trajectory) {
    let (tau, steps, capped, work, assemblies, taken) = (trajectory.tau, trajectory.steps, trajectory.capped, trajectory.work, trajectory.assemblies, trajectory.taken);
    *trajectory = end.clone();
    (trajectory.tau, trajectory.steps, trajectory.capped, trajectory.work, trajectory.assemblies, trajectory.taken) = (tau, steps, capped, work, assemblies, taken + 1);
}

/// The middle of a step of `length` from a state (`step::Midway`), counted: a step evaluation, and a middle of the model
/// where the step's fixed point holds.
fn midway<'a, D: Dynamics>(model: &'a D, trajectory: &mut Trajectory, from: &Trajectory, length: f64) -> Result<Midway<'a, D>, Refused> {
    trajectory.steps += 1;
    let through = Midway::new(model, from, length)?;
    trajectory.assemblies += 1;
    Ok(through)
}

/// Advances a held trajectory to the time `to` (the next output time) by steps of τ, with its jumps and its loss test
/// (module documentation, # The step, # The jumps' times; the loss is tested at every step's end and after every jump,
/// SPEC §1.7); `each` takes every step's path (a jump's step is the one whose own propagation reaches the threshold at
/// its end, the jump there). A trajectory lost on the way stops at its loss time. Where the scheme refuses a step at
/// every length down to the time's round-off, the advance ends there with the refusal; so it does where the model gives
/// a jump no photon (`step::jump`). `scratch` is the trajectory's (`step::Scratch`): its steps' memory, made once with it.
pub fn advance<D: Dynamics>(model: &D, tau: f64, trajectory: &mut Trajectory, scratch: &mut Scratch, to: f64, each: &mut dyn FnMut(&Segment)) -> Result<(), Refused> {
    trajectory.tau = tau;
    while trajectory.lost.is_none() && trajectory.time < to {
        let rest = to - trajectory.time;
        // The step taken from here (module documentation, # The jumps' times): the longest is τ or the interval's rest;
        // the first tried no longer than the time to the threshold at the last step's rate; then the length each step
        // tried proposes, inside the bracket of the lengths found too short and too long, until one is the step -- none
        // where the threshold is within the time's round-off of the start: the jump is made there. From half the
        // length where the scheme refuses a step, and no longer than that from then on.
        let (mut most, mut bracket) = (tau.min(rest), (0.0, f64::INFINITY));
        // The step's part of the Brownian path: its deviates, the path's shape over whatever length the step takes.
        let mut stream = Stream::diffusion(trajectory.seed, trajectory.index, trajectory.taken);
        let deviates = [stream.normals(), stream.normals()];
        let ahead = if trajectory.rate > 0.0 { (norm_sqr(&trajectory.psi) / trajectory.threshold).ln() / trajectory.rate } else { f64::INFINITY };
        let mut length = most.min(ahead);
        let (at, taken, jumps) = loop {
            let from = trajectory.clone();
            if !(trajectory.time + length > trajectory.time) {
                break (from, None, true);
            }
            let slack = if bracket.1 - bracket.0 <= JUMP_TIMING * length { f64::INFINITY } else { JUMP_TIMING * length };
            let reach = length.min(most - length);
            let brownian = Brownian::over(length, &deviates);
            let tried = midway(model, trajectory, &from, length).and_then(|through| through.attempt(&from, &brownian, slack, reach, scratch));
            let refusal = match tried {
                Ok(Attempt::Whole(whole)) => {
                    trajectory.work += whole.work;
                    break (from, Some(whole), false);
                }
                Ok(Attempt::Reached(whole)) => {
                    trajectory.work += whole.work;
                    break (from, Some(whole), true);
                }
                Ok(Attempt::Proposes { length: proposed, work }) => {
                    trajectory.work += work;
                    if proposed < length {
                        bracket.1 = length.min(bracket.1);
                    } else {
                        bracket.0 = length.max(bracket.0);
                    }
                    length = if proposed > bracket.0 && proposed < bracket.1 || !(trajectory.time + proposed > trajectory.time) { proposed } else { 0.5 * (bracket.0 + bracket.1.min(most)) };
                    continue;
                }
                Err(e) => e,
            };
            length *= 0.5;
            if !(trajectory.time + length > trajectory.time) {
                return Err(refusal);
            }
            (most, bracket) = (length, (0.0, f64::INFINITY));
        };
        if let Some(s) = &taken {
            each(&s.segment);
            let fall = (norm_sqr(&at.psi) / norm_sqr(&s.end.psi)).ln();
            adopt(trajectory, &s.end);
            trajectory.rate = fall / s.segment.tau;
            test_loss(model, trajectory);
        }
        if trajectory.lost.is_some() {
            break;
        }
        if jumps {
            // The jump's carried state is none (`step::jump`): no second internal state rides on the trajectory.
            jump(model, trajectory, &mut None)?;
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
    (step_work(middle, kappa, velocity, tau), 1)
}
