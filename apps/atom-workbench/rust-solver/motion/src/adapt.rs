//! One constant step for the run (SPEC §2.2).
//!
//! # The step
//! Every trajectory of a run advances by steps of one τ (`step::Midway::attempt`), whatever the times it is read at (# The reads):
//! a step ends at no output. The last step of the run is its rest; the step before a jump ends at the jump (# The jumps'
//! times), the next step from there; a step the scheme refuses (`step::Refused`) is taken at its half, and one refused
//! down to the time's round-off ends the advance with its refusal -- a numerical failure the solve counts and flags,
//! never a loss. A step that ends at a bin's end of the trajectory's grid (# The Brownian path) ends at that time, so
//! that a trajectory no jump has moved off the grid stays on it, the sum of its steps' lengths rounding or not. A step
//! takes its generator's change along the path through its own three nodes, its middle and its ends, whatever its
//! length (step.rs, # The kick): what that leaves is an error of the step's length like the splitting's, and the
//! pilot's companion at twice the step checks both at the outputs (solve.rs, # The step). The generator's own frequencies
//! are no matter of the step's length: they are integrated exactly.
//!
//! τ is one for the run because the splitting is symplectic in its conservative part and its own inverse: at a constant
//! step the energy of a trap's motion errs by a bounded amount (measured on the reference traps R1: no drift over 10³
//! periods), where a step chosen from the last step's error drifts. The steps are then the run's length over τ and
//! what its jumps cut off the steps before them.
//!
//! # The reads
//! A trajectory is read at the caller's times (the outputs) where they fall, inside a step, from what the step knows
//! between its ends (step.rs, # The path between a step's ends; `step::Step::at`): the internal state its propagation
//! has there, the position and the momentum on its path with the kick spread over it -- second order in τ like the
//! step's ends, a time at a step's end the end itself. A read is no part of the trajectory: nothing of it enters the
//! next step. So the step follows from what the scheme must resolve alone (solve.rs, # The step), and outputs however
//! dense cost a state at a time each, not a step.
//! - A time read lies in one step: after its start, at its end at the latest. A jump's step ends at the jump, so the
//!   times before a jump are read from the step to it and the times after it from the next.
//! - The loss is a first passage, tested wherever a state of the trajectory is made (SPEC §1.7): at every step's end,
//!   after every jump, and at every read -- else the loss at a time read inside a step would be the last step's end's,
//!   late by the step in the first order. A trajectory found lost at a read is lost at that time: the read is its
//!   last state, and its step counts up to there.
//! - Two levels of a trajectory (solve.rs, # The step) are read at the same times, each on its own steps: their
//!   readings agree as τ → 0, and differ by the scheme's accumulated error, smooth in the time, and by the two reads'
//!   own terms, of the second order with a factor that depends on the time's place in its step ((h² − u²)/2: its
//!   mean over the places is τ²/12, so Richardson's factor holds for it in the mean over the outputs, which is where
//!   the solve takes it). The Brownian path's part of a read is each level's own between its bins' ends, as its part
//!   of a jump's step is: a difference of zero mean, of the variance of what a step's moments do not tell (at most
//!   τ/12 of the diffusion's rate), which falls with τ.
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
//! the trajectory's own Brownian path W, three independent standard ones. The path is one function of the time for a
//! trajectory, whatever step it is taken at (`Path`): the pilot's two levels (solve.rs, # The step) -- the same
//! trajectory at τ and at 2τ -- see the same noise, and differ by what the step does and by nothing drawn anew. The
//! product, at one step, needs the path's law alone; there is one path for both.
//!
//! - The path is made on a tree. Time is cut into cells of one width, the caller's (the companion's step at the
//!   ladder's first rung, solve.rs); a cell's part of the path (its increment and its first moment about its middle,
//!   `step::Brownian::over`) is drawn from the cell's stream
//!   (`random::Stream::brownian`). A cell's two halves are the Brownian bridge between its ends (`step::Brownian::split`)
//!   from the stream of that node of the tree, and so on down: the part over any bin of any rung's grid is a function of
//!   (seed, trajectory, bin), and a bin's part is the sum of its halves' -- a step of 2τ on the grid takes the sum of
//!   the two steps of τ it spans.
//! - A trajectory at the step τ reads the tree down to the bins of its own grid, the cell over 2^depth = the multiples
//!   of τ. A step that ends off the grid -- at a jump, and every step of τ after one -- cuts the bin it ends in: the bridge
//!   between the bin's ends once more, at the cut, from the stream of the cut's time (`random::Stream::bridge`); the rest
//!   of the bin is kept for the step that follows (`Cursor`). A step tried at several lengths for its jump cuts at each
//!   of them, and the one taken moves the trajectory along the path; its norm does not depend on the path.
//! - Every part has the law of a Brownian path's over its stretch, exactly, and parts of stretches that do not overlap
//!   are independent: nothing of the scheme's law knows the tree. What two levels share is the path at the bins' ends
//!   of the coarser one; within a bin each level's cuts are its own.
//! - The tree is read by its branches: the nodes from a cell down to the bin last read are kept, and the next bin's
//!   branch is made from where it leaves them -- two nodes a bin on average, whatever the depth, and the same parts to
//!   the bit as a branch made from its cell.

use crate::linalg::norm_sqr;
use crate::random::Stream;
use crate::step::{jump, step_work, test_loss, Attempt, Brownian, Dynamics, Middle, Midway, Refused, Scratch, Segment, Trajectory};
use crate::units::{rounding, JUMP_TIMING};

/// Where a trajectory stands on its Brownian path (`Path`): the time, the bin of its grid that time lies in, and the
/// path's part from that time to the bin's end.
#[derive(Clone, Copy, Debug)]
pub struct Cursor {
    at: f64,
    bin: u64,
    rest: Brownian,
}

/// A trajectory's Brownian path (module documentation, # The Brownian path) as a trajectory at one step goes along it:
/// the run's seed and the trajectory's number, the cells' width (µs), the depth of the trajectory's grid in a cell's
/// tree (2^depth bins a cell), where the trajectory stands, and the branch of the tree last read -- its cell, the bin's
/// place in it, and the parts of the nodes from the cell down to the bin: a bin's branch is read on from where it
/// leaves the last one's (the next bin's shares all but two nodes with it on average), no part made twice.
#[derive(Clone, Debug)]
pub struct Path {
    seed: u64,
    index: u64,
    cell: f64,
    depth: u32,
    cursor: Cursor,
    branch: (u32, u64, Vec<Brownian>),
}

impl Path {
    /// The path of trajectory `index` of the run `seed` from t = 0, in cells of the width `cell`, for a trajectory whose
    /// grid has 2^depth bins a cell.
    pub fn new(seed: u64, index: u64, cell: f64, depth: u32) -> Path {
        let mut path = Path { seed, index, cell, depth, cursor: Cursor { at: 0.0, bin: 0, rest: Brownian::STILL }, branch: (0, 0, vec![]) };
        path.cursor.rest = path.whole(0);
        path
    }

    /// The end of a bin of the grid.
    fn end(&self, bin: u64) -> f64 {
        (bin + 1) as f64 * (self.cell / (1u64 << self.depth) as f64)
    }

    /// Whether a time is a bin's end: within the rounding of the additions that step a trajectory across a cell (a
    /// time is the sum of its steps' lengths, as many as the bins a cell and two more).
    fn ends(&self, time: f64, end: f64) -> bool {
        (time - end).abs() <= rounding((1usize << self.depth) + 2) * end
    }

    /// The path's part over a whole bin: its cell's, through the halves of the tree's nodes down to it. The node of
    /// the level l on the bin's branch is the one the first l bits of its place name, the 2^l + (place >> (depth − l))-th
    /// of the cell's tree; the levels above the first where the place leaves the last branch's are the last branch's.
    fn whole(&mut self, bin: u64) -> Brownian {
        let (seed, index, depth) = (self.seed, self.index, self.depth);
        let (cell, place) = ((bin >> depth) as u32, bin & ((1u64 << depth) - 1));
        let deviates = |node: u32| {
            let mut stream = Stream::brownian(seed, index, cell, node);
            [stream.normals(), stream.normals(), stream.normals()]
        };
        let (last_cell, last_place, parts) = &mut self.branch;
        if parts.is_empty() || *last_cell != cell {
            let root = deviates(0);
            *parts = vec![Brownian::over(self.cell, &[root[0], root[1]])];
        } else {
            let shared = (0..=depth).rev().find(|&level| place >> (depth - level) == *last_place >> (depth - level)).unwrap_or(0);
            parts.truncate(shared as usize + 1);
        }
        for level in parts.len() as u32 - 1..depth {
            let (width, node, half) = (self.cell / (1u64 << level) as f64, (1u64 << level) + (place >> (depth - level)), (place >> (depth - level - 1)) & 1);
            let part = parts[level as usize].split(width, 0.5 * width, &deviates(node as u32))[half as usize];
            parts.push(part);
        }
        (*last_cell, *last_place) = (cell, place);
        parts[depth as usize]
    }

    /// The path's part from where the trajectory stands to the time `to`, and where it stands then: the rest of its
    /// bin and the bins after it, joined, up to the bin `to` lies in, which is cut there.
    fn reach(&mut self, to: f64) -> (Brownian, Cursor) {
        let mut cursor = self.cursor;
        let (mut part, mut length) = (Brownian::STILL, 0.0);
        loop {
            let end = self.end(cursor.bin);
            let rest = end - cursor.at;
            let ends = self.ends(to, end);
            if ends || to > end {
                (part, length) = (part.joined(length, &cursor.rest, rest), length + rest);
                cursor = Cursor { at: end, bin: cursor.bin + 1, rest: self.whole(cursor.bin + 1) };
                if ends {
                    return (part, cursor);
                }
            } else {
                let piece = to - cursor.at;
                let mut stream = Stream::bridge(self.seed, self.index, to);
                let [head, tail] = cursor.rest.split(rest, piece, &[stream.normals(), stream.normals(), stream.normals()]);
                return (part.joined(length, &head, piece), Cursor { at: to, bin: cursor.bin, rest: tail });
            }
        }
    }
}

/// The trajectory's physical state from the end of a step taken, its counters kept (a step's end holds its start's:
/// the steps tried on the way to it are counted on the trajectory).
fn adopt(trajectory: &mut Trajectory, end: &Trajectory) {
    let (work, assemblies) = (trajectory.work, trajectory.assemblies);
    *trajectory = end.clone();
    (trajectory.work, trajectory.assemblies) = (work, assemblies);
}

/// The middle of a step of `length` from a state (`step::Midway`), counted: a middle of the model where the step's
/// fixed point holds.
fn midway<'a, D: Dynamics>(model: &'a D, trajectory: &mut Trajectory, from: &Trajectory, length: f64) -> Result<Midway<'a, D>, Refused> {
    let through = Midway::new(model, from, length)?;
    trajectory.assemblies += 1;
    Ok(through)
}

/// Advances a held trajectory to the time `to` (the run's end) by steps of τ, with its jumps and its loss test (module
/// documentation, # The step, # The jumps' times; the loss is tested at every step's end, after every jump and at every
/// read, SPEC §1.7), and reads it at the times `at` (increasing) that lie after its time and no later than `to`, each
/// inside the step it falls in (# The reads): `read` takes a time's place in `at` and the state there. `each` takes
/// every step's path with the time the trajectory is on it to -- the step's end, or the read it is found lost at -- (a
/// jump's step is the one whose own propagation reaches the threshold at its end, the jump there). A trajectory lost
/// on the way stops at its loss time. Where the scheme refuses a step at every length down to the time's round-off,
/// the advance ends there with the refusal; so it does where the model gives a jump no photon (`step::jump`).
/// `scratch` is the trajectory's (`step::Scratch`): its steps' memory, made once with it; `path` its Brownian path,
/// which it goes along as it is advanced (`Path`).
#[allow(clippy::too_many_arguments)]
pub fn advance<D: Dynamics>(
    model: &D,
    tau: f64,
    trajectory: &mut Trajectory,
    scratch: &mut Scratch,
    path: &mut Path,
    to: f64,
    at: &[f64],
    read: &mut dyn FnMut(usize, &Trajectory),
    each: &mut dyn FnMut(&Segment, f64),
) -> Result<(), Refused> {
    let mut next = at.partition_point(|&time| time <= trajectory.time);
    while trajectory.lost.is_none() && trajectory.time < to {
        let rest = to - trajectory.time;
        // The step taken from here (module documentation, # The jumps' times): the longest is τ or the run's rest -- the
        // rest too where a step of τ would end within the grid's rounding of the run's end, which is then no step of
        // its own --; the first tried no longer than the time to the threshold at the last step's rate; then the length
        // each step tried proposes, inside the bracket of the lengths found too short and too long, until one is the
        // step -- none where the threshold is within the time's round-off of the start: the jump is made there. From
        // half the length where the scheme refuses a step, and no longer than that from then on.
        let whole = if path.ends(to, trajectory.time + tau) { rest } else { tau.min(rest) };
        let (mut most, mut bracket) = (whole, (0.0, f64::INFINITY));
        let ahead = if trajectory.rate > 0.0 { (norm_sqr(&trajectory.psi) / trajectory.threshold).ln() / trajectory.rate } else { f64::INFINITY };
        let mut length = most.min(ahead);
        let (from, taken, jumps) = loop {
            let from = trajectory.clone();
            if !(trajectory.time + length > trajectory.time) {
                break (from, None, true);
            }
            let slack = if bracket.1 - bracket.0 <= JUMP_TIMING * length { f64::INFINITY } else { JUMP_TIMING * length };
            // Followed beyond its end for its length again at most (not derived: the search's bracket alone, and what
            // bounds the one hop of `spectral::Picture::extend`), and not past the longest step.
            let reach = length.min(most - length);
            // The step's part of the Brownian path, and where the trajectory stands on it at the step's end.
            let (brownian, beyond) = path.reach(trajectory.time + length);
            let tried = midway(model, trajectory, &from, length).and_then(|through| through.attempt(&from, &brownian, slack, reach, scratch));
            let refusal = match tried {
                Ok(Attempt::Whole(whole)) => {
                    trajectory.work += whole.work;
                    break (from, Some((whole, beyond)), false);
                }
                Ok(Attempt::Reached(whole)) => {
                    trajectory.work += whole.work;
                    break (from, Some((whole, beyond)), true);
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
        if let Some((s, beyond)) = &taken {
            // The step's end on the trajectory's clock: the run's end where the step spanned its rest -- not an ulp
            // short of it (the sum of the steps' lengths rounds), which would leave a rest of the time's round-off to
            // be stepped --, else where the path has it: the bin's end it ends at, or its own cut.
            let end = if length == rest { to } else { beyond.at };
            // The times read inside the step (module documentation, # The reads), each with its loss test.
            while next < at.len() && at[next] <= end {
                let mut state = s.at(at[next], scratch);
                test_loss(model, &mut state);
                if state.lost.is_some() {
                    each(&s.segment, state.time);
                    adopt(trajectory, &state);
                    return Ok(());
                }
                read(next, &state);
                next += 1;
            }
            each(&s.segment, end);
            path.cursor = *beyond;
            let fall = (norm_sqr(&from.psi) / norm_sqr(&s.end.psi)).ln();
            adopt(trajectory, &s.end);
            trajectory.time = end;
            trajectory.rate = fall / s.segment.tau;
            test_loss(model, trajectory);
        }
        if trajectory.lost.is_some() {
            break;
        }
        if jumps {
            jump(model, trajectory)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A trajectory goes along its path to the times given, in order: the parts it takes.
    fn walk(path: &mut Path, times: &[f64]) -> Vec<Brownian> {
        times
            .iter()
            .map(|&to| {
                let (part, beyond) = path.reach(to);
                path.cursor = beyond;
                part
            })
            .collect()
    }

    /// The Brownian path is one function of the time whatever the step (module documentation, # The Brownian path).
    /// Over three cells of 8 µs, at the steps 1, 2 and 4 µs (depths 3, 2, 1): every step of 2τ on the grid takes the
    /// two steps of τ it spans, joined, to rounding. A trajectory whose steps end off the grid (a jump's) meets the
    /// same path again at every bin's end: its parts from one end of a bin to the next, joined, are the bin's own. A
    /// bin's part is the same to the bit whatever branch was read before it (the bins in a scattered order on one path
    /// against each on a path of its own). And the parts have a Brownian path's law at any cut: over 4096 trajectories the increment and the first moment of a
    /// stretch of the length s cut out of a bin have the variances s and s³/12 and no covariance, each within its
    /// interval at the confidence z over the looks.
    #[test]
    fn a_trajectory_s_brownian_path_is_the_same_at_every_step() {
        let (cell, cells) = (8.0, 3u64);
        let grid = |depth: u32| -> Vec<f64> { (1..=cells << depth).map(|i| i as f64 * cell / (1u64 << depth) as f64).collect() };
        let close = |a: &Brownian, b: &Brownian| (0..3).all(|x| (a.increment[x] - b.increment[x]).abs() <= 1e-13 && (a.moment[x] - b.moment[x]).abs() <= 1e-12);
        for index in 0..16 {
            let levels: Vec<Vec<Brownian>> = [3, 2, 1].map(|depth| walk(&mut Path::new(5, index, cell, depth), &grid(depth))).into();
            for (fine, coarse) in levels.iter().zip(&levels[1..]) {
                let width = cell * cells as f64 / fine.len() as f64;
                for (i, step) in coarse.iter().enumerate() {
                    assert!(close(step, &fine[2 * i].joined(width, &fine[2 * i + 1], width)), "trajectory {index}: step {i} of {} against its halves", coarse.len());
                }
            }
            // Steps that end off the grid, between the ends of the bins of 2 µs.
            let mut times: Vec<f64> = grid(2);
            times.extend([0.37, 1.999, 2.5, 3.1, 3.3, 7.9, 9.0, 13.25, 20.5, 23.99]);
            times.sort_by(f64::total_cmp);
            let parts = walk(&mut Path::new(5, index, cell, 2), &times);
            let (mut joined, mut length, mut from, mut bin) = (Brownian::STILL, 0.0, 0.0, 0);
            for (part, &to) in parts.iter().zip(&times) {
                (joined, length, from) = (joined.joined(length, part, to - from), length + to - from, to);
                if to == grid(2)[bin] {
                    assert!(close(&joined, &levels[1][bin]), "trajectory {index}: the bin to {to} from its cut parts");
                    (joined, length, bin) = (Brownian::STILL, 0.0, bin + 1);
                }
            }
        }
        let mut one = Path::new(5, 3, cell, 5);
        for bin in [77, 3, 4, 95, 0, 31, 32, 33, 64, 12, 13, 12] {
            assert_eq!(one.whole(bin), Path::new(5, 3, cell, 5).whole(bin), "bin {bin} after another branch");
        }
        let (n, from, to) = (4096, 2.37, 3.81);
        let mut sums = [0.0; 3];
        for index in 0..n {
            let part = walk(&mut Path::new(9, index, cell, 2), &[from, to])[1];
            let (w, z) = (part.increment[1] / (to - from).sqrt(), part.moment[1] / ((to - from).powi(3) / 12.0).sqrt());
            sums = [sums[0] + w * w, sums[1] + z * z, sums[2] + w * z];
        }
        let within = |error: f64| crate::estimate::Spread { error, groups: f64::INFINITY }.interval_over(3.0);
        let [w, z, wz] = sums.map(|s| s / n as f64);
        assert!((w - 1.0).abs() <= within((2.0 / n as f64).sqrt()) && (z - 1.0).abs() <= within((2.0 / n as f64).sqrt()) && wz.abs() <= within((1.0 / n as f64).sqrt()), "a cut stretch's variances over their own: {w}, {z}; covariance {wz}");
    }
}
