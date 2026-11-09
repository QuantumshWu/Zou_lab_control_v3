//! Per-trajectory step control by step doubling against tolerances per unit time (SPEC §2.2).
//!
//! # One rule
//! From the same state one step of τ and two of τ/2 (`step::step`): the scheme's local error is O(τ^{p+1}), p = 2, so
//! the two halves' error is their difference over 2^p − 1 = 3. It is read on what is displayed: per lab axis the momentum,
//! the loss energy E_s, the displayed rows (`Tolerances`). Within the tolerances, the halves are taken; beyond them, τ is
//! redone smaller; either way the next τ ← τ (tolerance/error)^{1/(p+1)}. A step never crosses the output time it
//! advances to. Where the halves' propagation crosses the jump threshold, the half that holds the root is redone to it
//! and the jump made there (`step::jump`); the next step starts from the state the jump left.

use crate::step::{Dynamics, Segment, Trajectory};

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

/// Advances a held trajectory to the time `to` (the next output time; the steps never cross it) by doubled steps, its
/// jumps and its loss test; `each` takes every accepted step's path (a jump's step is the one redone to its root, the
/// jump at its end). A trajectory lost on the way stops at its loss time. Past `units::MOST_STEPS` the steps go on
/// unchecked at the last accepted length and `Trajectory::capped` records from when.
pub fn advance<D: Dynamics>(model: &D, tolerances: &Tolerances, trajectory: &mut Trajectory, to: f64, each: &mut dyn FnMut(&Segment)) {
    let _ = (model, tolerances, trajectory, to, each);
    todo!()
}
