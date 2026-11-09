//! The pipeline: request, reach and model, pilot, decisions, batches, checkpoints, result (SPEC §3, §5, §9).
//!
//! # The pipeline
//! 1. The request's checks; the reach and the model (model.rs).
//! 2. The pilot: an ensemble of its own stream (`random::Purpose::Pilot`) that predicts only (decisions.md A4) -- the
//!    model's units' measured errors and work, the outputs where T is held to its target, the statistics' scale, the
//!    reach the atoms come to. Its result is the first checkpoint.
//! 3. The model's decisions (SPEC §5.2): every unit whose existence gate fails kept, the others folded or kept by the
//!    saved error per work until the model's error leaves the statistics their room, and its falsifier.
//! 4. The product: batches of trajectories, each with its own adaptive steps (adapt.rs), all run to the last output,
//!    until every output's statistics meet the fixed targets the pilot left them, or the trajectories reach their cap.
//! 5. The step's falsifier at a quarter of the tolerance on a few coupled trajectories (SPEC §5.3).
//! 6. The result (contract §2.2), every checkpoint a complete one.
use crate::ledger::Ledger;
use crate::request::{Request, STATES};

/// The population target the app checks a result against by equality (contract §2.2 `requestedTolerance`).
pub use crate::units::POPULATION_TARGET;

/// A distribution along one lab axis on one grid (strictly increasing, at least two points) at some of the output
/// times (`samples`: increasing indices, the last output among them), each the probability at every grid point
/// (contract §2.2 `momentum`, `position`).
#[derive(Clone, Debug, PartialEq)]
pub struct Distributions {
    pub grid: Vec<f64>,
    pub samples: Vec<usize>,
    pub probabilities: Vec<Vec<f64>>,
}

/// A solve's result at every output time of the request, per lab axis x, y, z (contract §2.2). Everything but the
/// loss describes the atom still held.
#[derive(Clone, Debug)]
pub struct Solution {
    /// The held atom's populations, p = P/(1 − L) per output (row 0 the request's initial diagonal), before the
    /// adapter's projection onto the simplex.
    pub populations: Vec<[f64; STATES]>,
    /// Per output the total estimated error of what is shown, the rows as shown and the loss (ledger.rs `Account`).
    pub estimated_errors: Vec<f64>,
    /// T per lab axis (µK) and its standard error, none where fewer than two atoms or groups read it; T_total =
    /// (T_x + T_y + T_z)/3 and its own standard error (estimate.rs `Motion`).
    pub temperature_uk: Vec<[f64; 3]>,
    pub temperature_errors_uk: Vec<[Option<f64>; 3]>,
    pub temperature_total_uk: Vec<f64>,
    pub temperature_total_errors_uk: Vec<Option<f64>>,
    /// The mean velocity (m/s) and n̄ per lab axis, n̄ relative to `trap_frequency_khz`.
    pub mean_velocity_mps: Vec<[f64; 3]>,
    pub mean_occupation: Vec<[f64; 3]>,
    /// ω/2π (kHz) along each lab axis at the bottom of the reference state's trap.
    pub trap_frequency_khz: [f64; 3],
    /// The probability that the atom is lost by each output, never decreasing.
    pub loss_probability: Vec<f64>,
    /// Velocities (m/s) and positions (µm from the focus) per lab axis.
    pub velocity: [Distributions; 3],
    pub position: [Distributions; 3],
    /// Per output: in the validated regime, k_BT ≥ CLASSICAL ħω on every axis (reporting only).
    pub classical: Vec<bool>,
    pub trajectories: u64,
    /// The mean step over the trajectories' steps (µs): each trajectory takes its own (adapt.rs).
    pub time_step_us: f64,
    pub method: String,
    /// Why the result is partial (Chinese): present exactly when some output's total exceeds the population target.
    pub reason: Option<String>,
    pub approximations: Ledger,
}

/// The solve of a request (module documentation): every checkpoint handed to `checkpoint` with the estimated wall time
/// left (ms) where the run's pace gives one, the final solution returned; partial results are never withheld.
pub fn solve(request: &Request, checkpoint: &mut dyn FnMut(&Solution, Option<f64>)) -> Result<Solution, String> {
    let _ = (request, checkpoint);
    todo!("phase 2: the pipeline on the model's and the step's APIs")
}
