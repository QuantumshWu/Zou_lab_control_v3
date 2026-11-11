//! The pipeline: request, reach and model, pilot, decisions, batches, checkpoints, result (SPEC §3, §5, §9).
//!
//! # The pipeline
//! 1. The request's checks; the light; the initial components; the reach and the model with every unit whose existence
//!    gate fails kept (model.rs, SPEC §1.5).
//! 2. The scout: the first member of every pilot group, at the first rung and at twice its step (# The step), screened.
//!    Where its atoms outgrow the reach, or a folded unit is to be kept at its states (`keep`), the attempt ends there
//!    and the next begins with that model: a model is settled for the price of a few trajectories, not of a pilot.
//!    The pilot: the run's first `pilot_size` trajectories (their own numbers, from `MOST_TRAJECTORIES`; the scout's are
//!    its groups' first members and are not run again), at the run's two levels. It screens the step, reads the model's
//!    units' errors at the states of the first held member of every group, the outputs and lab axes where T is held to
//!    its target, and the reach's test. Its trajectories stay in the result: they are trajectories of the run's own
//!    model and step.
//! 3. The model's decisions (SPEC §5.2), from the pilot alone, the step's error taken at its share.
//! 4. The product: batches of whole groups at the settled step, one level, every trajectory run to the last output,
//!    until every held quantity's total lies within its target (# Stopping) or the trajectories reach
//!    `MOST_TRAJECTORIES`.
//! 5. The result (contract §2.2, `assemble`).
//!
//! # The step (SPEC §2.2)
//! One step τ for the run (adapt.rs). A step divides the output interval evenly, and so does its companion's at twice
//! the step: the ladder's rungs are the output interval over the even numbers, and below a rung its halves. A step is
//! admissible where its nodes resolve what the scheme samples along an atom's path (`admissible`): the reference flow,
//! ω τ < π at the prepared trap's largest frequency; and the generator, whose change along the path the propagation
//! reads at nodes half a step apart, (|Ω| + |K| v) τ/2 < π for every residual beat Ω and wave vector K of the model's
//! entries at an atom's speed v. The ladder begins at the largest rung that is admissible for the run and whose double
//! is for the companion (`first_rung`): the run's step at the largest speed of the reach the model is judged over; the
//! companion's at the reach's rms speed.
//!
//! The step's convergence is checked on the pilot, as a refinement is checked by hand: the pilot's trajectories run at
//! τ and, the same trajectory numbers -- the same initial atoms, the same jumps' draws by their number, the same
//! Brownian path (adapt.rs) --, at 2τ (`Levels`). A quantity's step error is e(τ) = [Q(2τ) − Q(τ)]/(2^p − 1), averaged
//! over the run's outputs, its error the delete-one-group jackknife of that average (estimate.rs `between`): one number
//! for the loss, for each row as shown and for T on each lab axis, each carried to the outputs by the dynamics itself.
//! The step's error is a smooth function of the time, and one question whatever the number of outputs.
//!
//! - A comparison of two levels says something only where the coarser one still resolves what the scheme samples: the
//!   first rung's double is admissible for the typical atom, and stable for every atom.
//! - The screen (`screen`, `settle`), on the scout and on the pilot: the step is halved while some average lies clearly
//!   beyond the step's share among the family's few looks (decide.rs `judge`); the level at τ becomes the companion and
//!   only τ/2 is run anew. Down to the step at which a trajectory would take `MOST_STEPS` steps, where the step stands
//!   as it is, flagged.
//! - What enters a quantity's account, at every output alike, is the clear part of its average, |e| − t·SE where
//!   positive, among the same looks (`checked`): nothing where no change is seen, at most the step's share where the
//!   screen let the step stand. It is fixed after the pilot; the product runs one level and reads no step.
//! - The check's resolution -- t·SE of the averages -- is said in the result's method line beside the two steps
//!   compared and the number of pairs (`Checked`). It is in no total and raises no flag.
//!
//! It is a test of "no change seen", not a bound: an error confined to a short stretch of the run is diluted by the
//! run's length in the average, and one below the resolution is not seen at all.
//! A bound per output was built and measured (the product at
//! both levels, a60231dd): the pairs part within 100 µs in six beams' cooling whatever the noise they share, so the
//! difference's error at the late outputs is two independent ensembles', and the measurement cost 30 to 240 % of a run
//! on top of it; pooling it over the outputs books less than the truth where the error grows in time (SPEC §2.2).
//!
//! # Outputs and windows
//! The outputs are the request's times, from 0 and evenly spaced. T's window about an output (estimate.rs `half_window`)
//! takes a trajectory's κ and κ² over its steps' paths (step.rs `Segment::moments`, exact on the path); two outputs'
//! windows never overlap (each at most half an interval on either side), the first is the instant t = 0, and the run goes
//! on through the last output's window (`Plan`).
//!
//! # The model (SPEC §5.2)
//! The units are the model's (model.rs), listed again in every model a keep closes. A unit whose fold does not exist over
//! the reach is kept, and so is one whose measured error leaves an output no room for the statistics. Beyond these the
//! model keeps a unit where keeping it pays (decide.rs `pays`): the run's predicted total work W = w·N, N the trajectories
//! its looks need beside its errors and the step's share (`Look::need` on the pilot's estimates) and w the
//! counted work a trajectory spends per µs (`rate`, the step's count through the model's generator at the pilot's states),
//! falls by a share of itself beyond that share's spread, the jackknife over the pilot's groups (`Pricing`) -- the kept
//! model's W counting the pilot its attempt runs again: every keep is a new attempt, a new model and its own pilot. A
//! kept model is built and rated only for a unit whose keep would pay were it to cost no more per µs than the fold
//! (`Pricing::saving_at_all`: the trajectories its error's removal saves, less the new pilot): a fold is made to cost less than
//! what it folds, so a keep that pays by its rate alone is not looked for -- where one exists the fold is at fault, not
//! the choice.
//!
//! # Stopping (SPEC §3)
//! Every quantity held to a target -- what is shown at an output; T on a lab axis held there -- has the total
//! E² = A² + c² + S² (ledger.rs; `Look`): A the model's error, from the units measured at the pilot's states, and c the
//! step's clear part, both fixed by the pilot; S the statistics' interval. The run stops where every one of them lies
//! within its target: the test the ledger flags an output by (`Look::met`). The stop reads the run's standard errors
//! and nothing else of its estimates. A look that no number of trajectories brings within its target (A² + c² ≥ ε²:
//! the model's decisions have kept what they could) is not chased: the ledger flags it.
//!
//! The trajectories a look needs (`Look::need`): the statistics may take the room R² = ε² − A² − c², and fall as
//! N^{−1/2}; on what is shown they are no less than the unseen share at N (estimate.rs `unseen`, `draws_for`). A batch
//! brings the trajectories to the number the looks predict, at most to twice the number the prediction rests on
//! (`batch`).
//!
//! # T is held per lab axis
//! The semiclassical model is validated where k_BT ≥ `CLASSICAL` ħω and not below (SPEC §0, §8: its one known limit). The
//! test is taken axis by axis: T_a is held to its target -- its step's difference screened against the step's share, its
//! clear part booked, its look met or chased (`looks`), the total flagged where it exceeds the target (ledger.rs) -- at every
//! output where the pilot's T_a may lie in the regime of its own axis; an axis below it is reported, not chased, and
//! takes no other axis with it: a trap's tight axis that cools to a few ħω leaves the numerical error of the loose axes,
//! where the model holds, as controlled as before. The result says per output and axis whether the run's own T_a lies
//! in the regime (`Solution::classical`), whether the axis was held (`Solution::temperature_held`) and its total
//! estimated error (`Solution::temperature_estimated_errors`).
//!
//! The axis's frequency is the reported trap's along the lab axis, ω_a² = (ħ/m) ∂²V/∂x_a² at its bottom (model.rs
//! `Trap::lab`): T_a is the variance of κ along the lab axis over that frequency's period (# Outputs and windows), n̄_a
//! and the result's `trapFrequencyKHz` are read against it, and so is the regime -- one frequency for everything said of
//! an axis; the principal axes' own would test temperatures that are not estimated. In a trap whose principal axes e_i
//! are tilted against the lab axes it is the root mean square of the principal frequencies the axis is made of,
//! ω_a² = Σ_i (e_i·x_a)² ω_i², no less than their mean of the same weights (Jensen): with every mode at its own boundary,
//! k_BT_i = C ħω_i, the axis's T_a = Σ_i (e_i·x_a)² T_i = C ħ Σ_i (e_i·x_a)² ω_i/k_B ≤ C ħω_a/k_B, so a tilted axis is
//! held only where the modes it is made of are inside the regime in that weighted mean, never further out.
//!
//! Which axes are held at which outputs is the pilot's, fixed before the product; the product's own T does not move it.
//!
//! # Checkpoints
//! The decisions take counts, never the wall clock, so a request gives the same result on any machine. The wall clock only
//! times the messages (`Messages`): one clock for the whole solve, set at every message; whenever `CHECKPOINT` has passed
//! since the last one, the stage that is running sends a complete result of the groups it has finished so far in their
//! order (a prefix: whole groups chosen by their number alone). A result is passed on where it rests on two groups at
//! least (the jackknife's least for a spread) and on no fewer trajectories than the attempt's last one -- a pilot's
//! first groups at a finer step are no better than the result before them --; else the last result goes out again, so
//! that the app hears from the solve whatever stage it is in.
use crate::adapt;
use crate::decide::{assured, halved, judge, pays, Judgement, Read};
use crate::ensemble::{Ensemble, Quadrature, Start, Well, GROUP};
use crate::estimate::{between, draws_for, half_window, occupation, spread, summarize, unseen, Between, Estimate, Fate, Group, Held, Scales, Spread};
use crate::field::{Geometry, Light};
use crate::ledger::{percent, span, when, Account, Ledger};
use crate::linalg::V3;
use crate::model::{Envelope, Lights, Model, Run, Sample, Trap, Unit, UnitError, GROUND};
use crate::request::STATES;
use crate::step::{Dynamics, Scratch, Segment, Trajectory};
use crate::units::{hbar_over_mass, CLASSICAL, CONFIDENCE, KT_PER_UK, MOST_STEPS, MOST_TRAJECTORIES, ORDER, STEP_SHARE, TEMPERATURE_TARGET};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::f64::consts::{PI, TAU};
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// The population target the app checks a result against by equality (contract §2.2 `requestedTolerance`).
pub use crate::units::POPULATION_TARGET;
use crate::request::Request;

/// The runs' seed, the key of every trajectory's streams (random.rs): any fixed value, fixed so that a request gives the
/// same result every time.
const SEED: u64 = 0;
/// The output times the distributions are given at (contract §3.4: at most 21, t = 0 and the last among them).
const DISTRIBUTION_TIMES: usize = 21;
/// Half the time the app waits for a message before it kills the worker (contract §2.5: 120 s): the checkpoints' interval.
const CHECKPOINT: Duration = Duration::from_secs(60);
/// What the engine did with a folded unit (the ledger's `action`, shown by the app): its booked error is read at the
/// pilot's states (SPEC §5.2).
const FOLD_MEASURED: &str = "二阶消去（折叠），沿试探轨迹实测";
/// What the engine did with a near table's error (SPEC §5.4): refined over the reach when it was built, and read
/// against the elimination itself at the pilot's states.
const TABLE_MEASURED: &str = "按可达范围加密的表，在试探状态上对照消去本身实测";

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
    /// Per output and lab axis: in the validated regime, k_BT_a ≥ CLASSICAL ħω_a at the run's own T_a (reporting only; the
    /// contract's `classical` is every axis's, contract §2.2).
    pub classical: Vec<[bool; 3]>,
    /// Per output and lab axis: whether the axis's T is held to its target (module documentation, # T is held per lab axis:
    /// fixed by the pilot; an axis below its validated regime is not), and its total estimated error (relative, ledger.rs `Account`:
    /// model, step and statistics, weighed by the held share), none where one of the three is not read.
    pub temperature_held: Vec<[bool; 3]>,
    pub temperature_estimated_errors: Vec<[Option<f64>; 3]>,
    pub trajectories: u64,
    /// The run's step τ (µs, adapt.rs).
    pub time_step_us: f64,
    pub method: String,
    /// What keeps the worst output beyond the population target (Chinese): present where its total exceeds it.
    pub reason: Option<String>,
    pub approximations: Ledger,
}

/// The solve of a request (module documentation): every checkpoint handed to `checkpoint` with the estimated wall time
/// left (ms) where the run's pace gives one, the final solution returned; partial results are never withheld.
pub fn solve(request: &Request, checkpoint: &mut dyn FnMut(&Solution, Option<f64>)) -> Result<Solution, String> {
    request.validate()?;
    let light = Light::new(request);
    let (mut kept, mut reach, mut step) = (vec![], request.temperature_uk, None);
    // The last keep made for its predicted saving (`Attempt::Keep`): the units kept before it, the step then and the
    // saving's share; and the units such a keep has been undone for (`Attempt::Undo`), not priced again.
    let (mut priced, mut barred): (Option<(Vec<Unit>, f64, f64)>, Vec<Unit>) = (None, vec![]);
    let mut messages = Messages { send: checkpoint, clock: Instant::now(), last: None, least: 0 };
    loop {
        // An attempt's results improve on its own last one, from two groups on (module documentation, # Checkpoints).
        messages.least = 2 * GROUP;
        match attempt(request, &light, &mut kept, reach, step, priced.as_ref().map(|p| (p.1, p.2)), &barred, &mut messages)? {
            Attempt::Grow(t, tau) => (reach, step) = (t, Some(tau)),
            Attempt::Keep(u, tau, share) => {
                priced = share.map(|share| (kept.clone(), tau, share));
                barred.extend(share.map(|_| u.clone()));
                kept.push(u);
                step = Some(tau);
            }
            Attempt::Undo => {
                let (before, tau, _) = priced.take().expect("only an attempt told of a priced keep undoes one");
                (kept, step) = (before, Some(tau));
            }
            Attempt::Done(s) => return Ok(s),
        }
    }
}

/// The solve's messages to the app (module documentation, # Checkpoints).
struct Messages<'a> {
    send: &'a mut dyn FnMut(&Solution, Option<f64>),
    /// The last message's time.
    clock: Instant,
    /// The last result sent.
    last: Option<Solution>,
    /// The trajectories a result of this attempt must rest on to be sent.
    least: u64,
}
impl Messages<'_> {
    /// A result of the solve so far, with the wall time left (ms) where the run's pace gives one: sent where it improves
    /// on the attempt's last one, else the last result again.
    fn result(&mut self, s: &Solution, remaining: Option<f64>) {
        if s.trajectories >= self.least {
            self.least = s.trajectories;
            (self.send)(s, remaining);
            self.last = Some(s.clone());
            self.clock = Instant::now();
        } else {
            self.again();
        }
    }
    /// Nothing new: the last result again; with none yet there is nothing to send, and the next look is a full interval on.
    fn again(&mut self) {
        if let Some(s) = &self.last {
            (self.send)(s, None);
        }
        self.clock = Instant::now();
    }
    /// The time until the next message is due.
    fn due(&self) -> Duration {
        CHECKPOINT.saturating_sub(self.clock.elapsed())
    }
}

/// How one attempt at a model ends (`attempt`): the pilot's atoms outgrew the reach the model was judged over (its new
/// temperature, µK), a unit is to be kept -- each with the step the attempt stood at, the next attempt's first rung; a
/// keep with the share of the work it was predicted to save where that is why it is kept (`keep`) --, the last such
/// keep is undone (its model's step, once screened, makes it dearer than the model before it), or the solution.
enum Attempt {
    Grow(f64, f64),
    Keep(Unit, f64, Option<f64>),
    Undo,
    Done(Solution),
}

/// The initial components (SPEC §2.5): the ground states the request's initial density holds and their populations
/// normalized over them -- one at least: a validated request's density is diagonal and on the ground states
/// (request.rs `Request::validate`).
fn components(request: &Request) -> Vec<(usize, f64)> {
    let diagonal: Vec<f64> = (0..GROUND).map(|s| request.initial_density[(s, s)].re).collect();
    let total: f64 = diagonal.iter().sum();
    (0..GROUND).filter(|&s| diagonal[s] > 0.).map(|s| (s, diagonal[s] / total)).collect()
}

/// The ground populations of a basis state alone.
fn alone(state: usize) -> [f64; GROUND] {
    std::array::from_fn(|s| if s == state { 1. } else { 0. })
}

/// One initial component's preparation well (ensemble.rs `Well`) from the model: its prepared potential -- zero at the far
/// level where the trap light is Gaussian beams, the ensemble refusing any other --, the trap, the envelope's beams and peak.
struct Prepared<'a> {
    model: &'a Model,
    light: &'a Light,
    populations: [f64; GROUND],
    lights: Lights,
    trap: Trap,
    envelope: Envelope,
}
impl<'a> Prepared<'a> {
    fn new(model: &'a Model, light: &'a Light, populations: [f64; GROUND]) -> Result<Prepared<'a>, String> {
        Ok(Prepared { trap: model.trap(&populations, &model.prepared())?, envelope: model.envelope(&populations), lights: model.prepared(), model, light, populations })
    }
}
impl Well for Prepared<'_> {
    fn potential(&self, r: V3) -> (f64, V3) {
        let q = self.model.expansion(r, &self.populations, &self.lights);
        (q.value, q.gradient)
    }
    fn trap(&self) -> (V3, [V3; 3], V3) {
        (self.trap.centre, self.trap.axes, self.trap.frequencies)
    }
    fn envelope(&self) -> (Vec<Geometry>, f64) {
        (self.envelope.beams.iter().map(|&l| self.light.beams[l]).collect(), self.envelope.peak())
    }
}

/// Whether a step is admissible for an atom of a speed (module documentation, # The step): its nodes resolve what the
/// scheme samples along the atom's path.
/// - The reference flow, at the prepared trap's largest frequency ω (rad/µs): ω τ < π -- there the two half flows of a
///   step turn the motion by π and any kick between them grows. The same for every atom.
/// - The generator: the propagation reads its change along the path at three nodes, the step's middle and its two
///   ends (step.rs, # The kick), τ/2 apart, and takes it for the parabola through them. An entry that turns at the rate
///   ρ along the path -- its residual beat Ω in the cluster's frames and its residual wave vector K at the atom's
///   velocity, ρ = |Ω + K·v| ≤ |Ω| + |K| v (`sampled`: the model's own entries, per beat the largest |K|) -- advances by
///   ρ τ/2 from node to node; at π the nodes read it as its alias. So (|Ω| + |K| v) τ/2 < π for every entry.
fn admissible(tau: f64, frequency: f64, sampled: &[[f64; 2]], speed: f64) -> bool {
    frequency * tau < PI && sampled.iter().all(|[beat, wave]| 0.5 * (beat + wave * speed) * tau < PI)
}

/// The ladder's first rung (module documentation, # The step): the largest rung -- the output interval over an even
/// number, its double a whole part of the interval too -- that is admissible for the run and whose double is admissible
/// for the companion (`admissible`), at the speeds `fastest` and `typical` (µm/µs) of the reach the model is judged over.
/// (Admissibility is a bound on the step from above: the least even number is found by counting up.)
/// - The run's step is the result's: it is admissible for every atom the model is judged for, at the reach's largest
///   speed (model.rs `Reach::speed`) -- one domain for the model's folds and for the step.
/// - The companion at twice the step is a probe of the step, not a result: it must be stable, which the trap's
///   resonance decides for every atom alike, and resolve the light for the typical atom, at the reach's rms speed
///   (`typical_speed`) -- two steps one of which reads the light's alias at most atoms agree or differ for no reason of
///   the step, while the few fastest atoms beyond its limit weigh in the difference by their share.
/// Where the trap's resonance binds, the first rung is the rung below the largest stable one. Half the output interval
/// at the most: the companion's step is a rung itself, and no step crosses an output.
fn first_rung(interval: f64, frequency: f64, sampled: &[[f64; 2]], fastest: f64, typical: f64) -> f64 {
    let step = |pairs: u64| interval / (2 * pairs) as f64;
    let pairs = (1..).find(|&m| admissible(step(m), frequency, sampled, fastest) && admissible(2. * step(m), frequency, sampled, typical)).expect("a step short enough is admissible");
    step(pairs)
}

/// The rms speed (µm/µs) of atoms at the temperature T (µK): √(3 (ħ/m) k_BT/ħ), the reach's typical atom at the
/// reach's own temperature.
fn typical_speed(hbar_over_mass: f64, temperature_uk: f64) -> f64 {
    (3. * hbar_over_mass * KT_PER_UK * temperature_uk).sqrt()
}

/// The outputs each lab axis's T is averaged over between the pilot's levels (estimate.rs `between`), and the axes
/// that are judged: an axis held to its target somewhere is averaged over the outputs it is held at, and judged; one
/// held nowhere is averaged over every output, for its accounts alone.
fn averaged_over(held: &[[bool; 3]]) -> (Vec<[bool; 3]>, [bool; 3]) {
    let somewhere: [bool; 3] = std::array::from_fn(|a| held.iter().any(|h| h[a]));
    (held.iter().map(|h| std::array::from_fn(|a| h[a] || !somewhere[a])).collect(), somewhere)
}

/// The reads a step is judged by (module documentation, # The step), each with the step's share of its target: the
/// run's averages of the loss, of every row as shown, and of T on each lab axis that is held to its target somewhere
/// (`held`). One family: what is clear of one of them is clear among them all.
fn judged(read: &Between, held: [bool; 3]) -> Vec<(Read, f64)> {
    let [sp, st] = step_bound();
    let mut looks = vec![(read.loss, sp)];
    looks.extend(read.shown.iter().map(|x| (*x, sp)));
    looks.extend((0..3).filter(|&a| held[a]).filter_map(|a| read.temperature[a].map(|t| (t, st))));
    looks
}

/// The screen of a step on the reads between its two levels (module documentation, # The step; decide.rs `judge`):
/// the family of its reads, each against the step's share. The step is halved where the family is beyond.
fn screen(read: &Between, held: [bool; 3]) -> Judgement {
    judge(&judged(read, held), ORDER as f64)
}

/// The step's part of an output's accounts (module documentation, # The step): on what is shown -- the largest of the
/// rows' and the loss's, a probability's error at most 1 -- and on T per lab axis (relative). The step's share until the
/// pilot's two levels are read; then the clear part of their difference's average over the run (`checked`), the same
/// at every output, unbounded on an axis's T that no output reads at both levels.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Bound {
    shown: f64,
    temperature: [f64; 3],
}
impl Bound {
    /// The step's share of each target (`step_bound`).
    fn share() -> Bound {
        let [shown, temperature] = step_bound();
        Bound { shown, temperature: [temperature; 3] }
    }
}

/// The step's check as the result tells it (module documentation, # The step): the companion's step (µs), the pairs
/// compared, and the check's resolution -- the interval t·SE of the pairs' averages over the run, the largest on what is
/// shown (absolute) and on T on the axes held to their target (relative; unbounded where a held axis's is not read).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Checked {
    companion: f64,
    pairs: u64,
    resolution: [f64; 2],
}

/// The pilot's reads between its two levels as the run takes them (module documentation, # The step): the clear part
/// of each average among the looks the screen judges -- |e| − t·SE at the family's interval, where positive --, and the
/// check's resolution, one look's t·SE.
fn checked(read: &Between, held: [bool; 3]) -> (Bound, [f64; 2]) {
    let looks = judged(read, held).len() as f64;
    let clear = |read: &Read| read.clear(looks).max(0.);
    let step = Bound { shown: read.shown.iter().map(clear).fold(clear(&read.loss), f64::max).min(1.), temperature: read.temperature.map(|t| t.map_or(f64::INFINITY, |t| clear(&t))) };
    let shown = read.shown.iter().map(|x| x.spread.interval()).fold(read.loss.spread.interval(), f64::max);
    let temperature = (0..3).filter(|&a| held[a]).map(|a| read.temperature[a].map_or(f64::INFINITY, |t| t.spread.interval())).fold(0., f64::max);
    (step, [shown, temperature])
}

/// The estimates, the listed outputs' atoms and the counts of some groups.
fn pooled(parts: &[&Part], scales: &Scales) -> (Vec<Estimate>, Vec<Vec<(V3, V3)>>, Counts) {
    let estimates = summarize(parts, scales);
    let mut listed: Vec<Vec<(V3, V3)>> = vec![vec![]; parts.first().map_or(0, |p| p.listed.len())];
    let mut counts = Counts::default();
    for p in parts {
        for (l, x) in listed.iter_mut().zip(&p.listed) {
            l.extend_from_slice(x);
        }
        counts.add(&p.counts);
    }
    (estimates, listed, counts)
}

/// A message while the pilot's members come in: the result of what is in so far -- the parts in hand and those of the
/// members now running, each as a group of its own.
fn so_far(context: &Context, have: &[Part], more: &[&Part], messages: &mut Messages) {
    let all: Vec<&Part> = have.iter().chain(more.iter().copied()).collect();
    let (e, l, c) = pooled(&all, &context.scales);
    let n = e.first().map_or(0., |x| x.trajectories + x.failed) as u64;
    messages.result(&assemble(context, &e, &l, &c, n), None);
}

/// The reach's falsifier (SPEC §1.8; decide.rs: a booked bound that a read can only refute): the pilot's T where it has
/// outgrown the reach's temperature `reach` (µK) -- some output's T above it beyond its interval among all the T read
/// (Šidák over the outputs and axes) --; then the largest T read, the reach's new temperature. A T that reads above the
/// reach within its interval leaves the reach standing, as every unreadable falsifier's read does.
fn outgrown(estimates: &[Estimate], reach: f64) -> Option<f64> {
    let read: Vec<&crate::estimate::Motion> = estimates.iter().skip(1).filter_map(|e| e.motion.as_ref()).collect();
    let looks = 3. * read.len() as f64;
    let beyond = read.iter().any(|m| (0..3).any(|a| m.temperature[a] - m.temperature_errors[a].interval_over(looks) > reach));
    beyond.then(|| read.iter().flat_map(|m| m.temperature).fold(reach, f64::max))
}

/// Whether a lab axis's temperature (µK) lies in the validated regime of that axis's frequency (rad/µs), k_BT ≥
/// `CLASSICAL` ħω (SPEC §0, §3): the one test, per axis, of what the result reports as classical and of where the axis's T
/// is held to its target (module documentation, # T is held per lab axis).
fn classical(temperature: f64, frequency: f64) -> bool {
    KT_PER_UK * temperature >= CLASSICAL * frequency
}

/// The outputs and lab axes where T is held to its target (module documentation, # T is held per lab axis): where the T
/// of `estimates` (the pilot's) on the axis has its error read and its upper bound T_a + t·SE lies in the validated
/// regime of the axis's frequency (rad/µs); none at t = 0, the ensemble's exact moment.
fn held_to(estimates: &[Estimate], frequencies: &V3) -> Vec<[bool; 3]> {
    estimates
        .iter()
        .enumerate()
        .map(|(k, e)| {
            let Some(m) = e.motion.as_ref().filter(|_| k > 0) else { return [false; 3] };
            std::array::from_fn(|a| {
                let (t, error) = (m.temperature[a], &m.temperature_errors[a]);
                error.read() && t > 0. && classical(t + error.interval(), frequencies[a])
            })
        })
        .collect()
}

/// The states the model is read at and the folded units' errors there (SPEC §5.2): at every output the first member of
/// every group of `parts` that still holds its atom -- one a group keeps every group in a mean's spread, and a unit's
/// error is a function of the state; the first held one, so that an output some atom reaches is read wherever a group
/// has one there --, with each state's output and group. A state `known` already holds -- the same trajectory at the same
/// output and step, the scout's inside the pilot -- keeps the errors read there: a unit's error is a function of the state.
struct Sampled {
    samples: Vec<State>,
    at: Vec<usize>,
    from: Vec<usize>,
    errors: Vec<Vec<UnitError>>,
}
fn sampled(model: &Model, units: &[Unit], tables: &crate::model::tabulate::Tables, parts: &[&Part], known: Option<&Sampled>) -> Sampled {
    let mut samples: Vec<State> = vec![];
    let (mut at, mut from) = (vec![], vec![]);
    for (g, p) in parts.iter().enumerate() {
        for (k, s) in p.samples.iter().enumerate() {
            if let Some(state) = s.iter().min_by_key(|x| x.index) {
                samples.push(state.clone());
                at.push(k);
                from.push(g);
            }
        }
    }
    let read: std::collections::HashMap<(u64, usize, u64), usize> =
        known.map_or_else(Default::default, |k| k.samples.iter().zip(&k.at).enumerate().map(|(j, (s, &at))| ((s.index, at, s.tau.to_bits()), j)).collect());
    let each: Vec<Vec<UnitError>> = samples
        .par_iter()
        .zip(&at)
        .map(|(s, &k)| match (known, read.get(&(s.index, k, s.tau.to_bits()))) {
            (Some(known), Some(&j)) => known.errors.iter().map(|unit| unit[j]).collect(),
            _ => model.errors_with(units, std::slice::from_ref(&s.sample), Some(tables)).into_iter().map(|e| e[0]).collect(),
        })
        .collect();
    let errors = (0..units.len()).map(|u| each.iter().map(|e| e[u]).collect()).collect();
    Sampled { samples, at, from, errors }
}

/// The folded unit to keep, if any (SPEC §5.2), from the units' errors `measured` at `states` of the trajectories
/// `parts` (their `estimates`) and the errors `fixed` beside them, the step's taken at its share (a pilot's measured
/// bound is its own noise): a unit whose error leaves an output no room for the statistics -- of the
/// outputs left none, the unit of the largest weight there (its errors on what is shown and on T where they were read,
/// weighed by the held share as the totals are, each over its target, added up) --; else, of the units whose error's removal alone would pay
/// for the `again` trajectories its model runs before it stands where this one does, each rated with its kept model,
/// the one whose keeping saves the most predicted work beyond its spread -- with that saving's share, and never a unit
/// of `barred` (a keep of this kind once made is not priced again: undone or not, its price has been read).
#[allow(clippy::too_many_arguments)]
fn keep(
    request: &Request,
    light: &Light,
    kept: &[Unit],
    model: &Model,
    reference: [f64; GROUND],
    centre: V3,
    units: &[Unit],
    measured: &Measured,
    states: &[State],
    parts: &[&Part],
    estimates: &[Estimate],
    fixed: &Fixed,
    scales: &Scales,
    unsampled: [f64; 2],
    again: f64,
    barred: &[Unit],
) -> Option<(usize, Option<f64>)> {
    let means = measured.means(fixed.model.len(), None);
    let held: Vec<f64> = estimates.iter().map(|e| 1. - e.loss).collect();
    // A unit weighs where it was read: an output with no state read carries every unit's bound alike, and tells none
    // of them apart.
    let mut read = vec![false; fixed.model.len()];
    measured.output.iter().for_each(|&k| read[k] = true);
    let crowded: Vec<usize> = crowded(&looks(estimates, fixed, &vec![Bound::share(); fixed.model.len()])).into_iter().filter(|&k| held[k] > 0. && read[k]).collect();
    let weights: Vec<f64> = (0..units.len()).map(|u| crowded.iter().map(|&k| held[k] * (means[u][k][0] / POPULATION_TARGET + means[u][k][1] / TEMPERATURE_TARGET)).sum()).collect();
    if let Some(u) = (0..units.len()).max_by(|&a, &b| weights[a].total_cmp(&weights[b])).filter(|&u| weights[u] > 0.) {
        return Some((u, None));
    }
    let mut pricing = Pricing::new(request, light, kept, model, reference, centre, units, measured, states, parts, fixed, scales, unsampled, again);
    let saving = pricing.saving_at_all();
    let mut ranked: Vec<Candidate> = if saving.is_empty() {
        vec![]
    } else {
        pricing.rate();
        let pricing = &pricing;
        let others = pricing.leave_one_out();
        saving.into_par_iter().filter(|&u| !barred.contains(&units[u]) && pays(0., pricing.saving(&others, u, &pricing.own))).filter_map(|u| pricing.priced(&others, u)).collect()
    };
    ranked.sort_by(|a, b| assured(b.saving).total_cmp(&assured(a.saving)));
    ranked.first().filter(|c| pays(0., c.saving)).map(|c| (c.unit, Some(c.saving.value)))
}

/// One attempt at the solve with the units `kept` and the reach at `reach` µK (module documentation, # The pipeline). The
/// units its model's closure keeps join `kept`, the solve's one set: a unit an attempt keeps after its pilot is a unit of
/// the closed model (a drive of a carrier the closure made near), and the next attempt's model is built from both.
/// `step`: the step the attempt before stood at, this one's first rung. `priced`: the model's last unit was kept for a
/// predicted saving -- the step of the model before it and the saving's share of the work. The price knew the kept
/// model's work a step, not the step it needs (pricing reads no step); where the screen then takes a step at which the
/// kept model costs more than the one before it, (1 − share)·τ_before/τ > 1, the keep is undone (`Attempt::Undo`): the
/// fold's error stands, measured and booked as before.
#[allow(clippy::too_many_arguments)]
fn attempt(
    request: &Request,
    light: &Light,
    kept: &mut Vec<Unit>,
    reach: f64,
    step: Option<f64>,
    priced: Option<(f64, f64)>,
    barred: &[Unit],
    messages: &mut Messages,
) -> Result<Attempt, String> {
    let hm = hbar_over_mass(request.mass_kg);
    let initial = components(request);
    // The reference state (SPEC §2.1): the initial component of the largest population, of equal ones the last.
    let chief = (0..initial.len()).max_by(|&a, &b| initial[a].1.total_cmp(&initial[b].1)).expect("a validated request's initial state holds a ground state");
    let reference = alone(initial[chief].0);
    // The model: every unit whose fold does not exist over the reach kept (SPEC §1.5), closed again until none fails (a
    // model lists only what it folds, model.rs `units`).
    let (model, trap, judged) = loop {
        let model = Model::new(request, light, kept)?;
        let trap = model.trap(&reference, &model.prepared())?;
        let judged = model.reach(&trap, &reference, reach, unseen(MOST_TRAJECTORIES as f64));
        let units = model.units();
        let failing: Vec<Unit> =
            units.iter().zip(model.gates(&units, &judged)).filter(|(_, g)| !g.exists()).map(|(u, _)| u.clone()).collect();
        if failing.is_empty() {
            break (model, trap, judged);
        }
        kept.extend(failing);
    };
    // The trap every reported quantity describes (SPEC §1.3: trapFrequencyKHz, n̄, the validated regime, T's targeting and
    // its windows): the trap light's potential with the other lights' phase-averaged shifts at the reference state
    // (`Model::reported`). Where it has no minimum the descent reaches -- the atoms are not held by the light they run in: a
    // mathematical object that does not exist, not a kind of scene -- the prepared trap's is reported, and flagged.
    let (report, unheld) = match model.trap(&reference, &model.reported()) {
        Ok(t) => (t, false),
        Err(_) => (trap.clone(), true),
    };
    let scales = Scales { hbar_over_mass: hm, frequencies: report.lab };
    let plan = Plan::new(&request.times_us, report.lab);
    let outputs = plan.times.len();
    let wells: Vec<Prepared> = initial.iter().map(|&(s, _)| Prepared::new(&model, light, alone(s))).collect::<Result<_, _>>()?;
    let ensemble = Ensemble::new(&initial.iter().zip(&wells).map(|(&(s, p), w)| (s, p, w as &dyn Well)).collect::<Vec<_>>(), request.temperature_uk, hm)?;
    let row: [f64; STATES] = std::array::from_fn(|s| request.initial_density[(s, s)].re);
    // What the reach the model was judged over leaves out of the initial ensemble (model/reach.rs `Model::left_out`): of
    // the atoms and of their kinetic energy, counted on the initial atoms of the most trajectories a solve runs.
    let unsampled = model.left_out(&judged, (0..MOST_TRAJECTORIES as u64).map(|index| { let start = ensemble.draw(SEED, index); (start.r, start.kappa) }));
    let depth = ensemble.depths()[chief];
    let mut ledger = Ledger::default();
    ledger.note = Some(format!("初始系综：各初态在自己的准备阱中的束缚原子（降低的等温分布乘每轴谐振子的 Wigner 因子，T = {} μK），参考态阱深 {:.1} k_BT", request.temperature_uk, depth));
    if unheld {
        ledger.flag("运行中的势没有极小：阱频与 n̄ 按准备阱报告".into());
    }
    let exact = |kinetic: &[Quadrature]| -> (V3, f64) {
        (ensemble.momentum(&kinetic.iter().map(|k| k.eta).collect::<Vec<_>>()), kinetic.iter().map(|k| k.difference / k.eta).fold(0., f64::max))
    };
    // T(0) and n̄(0) by quadrature (ensemble.rs `Ensemble::kinetic`), twice. Here with no budget: the two levels it takes
    // at least -- an error needs two to differ --, for the results the pilot's stages send. After the pilot with as many
    // evaluations of the potential as the pilot took steps: a step evaluates the potential's expansion at least once (its
    // reference flow's, SPEC §2.1), so the quadrature costs no more than the pilot the solve has just run, in any trap;
    // where that budget ends it before its levels agree, their difference is T(0)'s booked error (`Fixed::start`).
    let (momentum, start) = exact(&ensemble.kinetic(0));
    // The ladder's first rung over the reach the model is judged over: an attempt that begins at the step the one
    // before stood at begins no higher than its own first rung (a reach that has grown lowers it).
    let first_rung = first_rung(plan.times[1] - plan.times[0], trap.frequencies.iter().copied().fold(0., f64::max), &model.sampled(&model.running()), judged.speed, typical_speed(hm, judged.temperature_uk));
    let mut context = Context {
        momentum,
        fixed: Fixed { model: vec![[0.; 2]; outputs], temperature: vec![[false; 3]; outputs], start },
        step: vec![Bound::share(); outputs],
        checked: None,
        tau: step.map_or(first_rung, |step| step.min(first_rung)),
        plan: Plan { cell: 2. * step.map_or(first_rung, |step| step.min(first_rung)), ..plan },
        scales,
        initial: row,
        ledger,
        method: format!("半经典 3D：{}", model.describe().trim_end_matches('。')),
    };

    // The near light's tables (SPEC §2.3, §5.4; model/tabulate.rs): refined over the reach to the largest ensemble's
    // resolution, no grid of more nodes than the middles the pilot's first rung asks of the model at its two levels -- a
    // table dearer than the run it serves is not built, and what it then misses is read at the atoms below.
    let most = (pilot_size() as f64 * (context.plan.end / context.tau + context.plan.end / (2. * context.tau))).ceil() as usize;
    let mut tables = model.tabulate(&judged, most);
    let run = Run::tabulated(&model, &tables, reference);

    // The pilot's groups (module documentation, # The pipeline): the run's first trajectories, at two levels.
    let first = MOST_TRAJECTORIES / GROUP;
    let numbers: Vec<u64> = (first..first + pilot_size() / GROUP).collect();
    // A priced keep's model at the step `step` against the model before it at its own (see `priced`).
    let dearer = |step: f64| priced.is_some_and(|(before, share)| (1. - share) * before / step > 1.);
    let mut levels = Levels::default();

    // The scout (module documentation, # The pipeline): the first member of every pilot group at both levels, screened;
    // where its atoms outgrow the reach or a unit is to be kept at its states, the attempt ends here -- for the price of
    // the scout, not of a pilot. The pilot's read of the same things below stands for the result.
    if settle(&run, &ensemble, &mut context, &mut levels, &numbers, 0..1, &report.lab, &dearer, messages) == Settled::Dearer {
        return Ok(Attempt::Undo);
    }
    let scouted = {
        let tau = context.tau;
        let scout: Vec<&Part> = levels.fine.iter().collect();
        let estimates = pooled(&scout, &context.scales).0;
        if let Some(t) = outgrown(&estimates, reach) {
            return Ok(Attempt::Grow(t, tau));
        }
        let units = model.units();
        let read = sampled(&model, &units, &tables, &scout, None);
        let measured = Measured { errors: &read.errors, output: &read.at, group: &read.from };
        let fixed = Fixed { model: model_error(&sums(&measured.means(outputs, None), unsampled), None, unsampled), temperature: held_to(&estimates, &report.lab), ..context.fixed.clone() };
        if let Some((u, share)) = keep(request, light, kept, &model, reference, trap.centre, &units, &measured, &read.samples, &scout, &estimates, &fixed, &context.scales, unsampled, scout.len() as f64, barred) {
            return Ok(Attempt::Keep(units[u].clone(), tau, share));
        }
        read
    };
    // The pilot: its groups' other members at both levels, screened.
    match settle(&run, &ensemble, &mut context, &mut levels, &numbers, 1..GROUP, &report.lab, &dearer, messages) {
        Settled::Dearer => return Ok(Attempt::Undo),
        Settled::Capped => context.ledger.flag(capped()),
        Settled::Standing => {}
    }
    let tau = context.tau;
    let run = {
        let pilot: Vec<&Part> = levels.fine.iter().collect();
        let (estimates, _, counts) = pooled(&pilot, &context.scales);
        if let Some(t) = outgrown(&estimates, reach) {
            return Ok(Attempt::Grow(t, tau));
        }
        let steps = counts.steps + levels.coarse.iter().map(|p| p.counts.steps).sum::<u64>();
        (context.momentum, context.fixed.start) = exact(&ensemble.kinetic(steps));

        // The model's errors at the pilot's states (SPEC §5.2): each folded unit's per output, the held atoms' mean over
        // the first member of every group -- one a group keeps every group in the mean's spread, and a unit's error is a
        // function of the state, read as well on eleven states an output as on eighty-eight.
        let units = model.units();
        let Sampled { samples, at, from, errors } = sampled(&model, &units, &tables, &pilot, Some(&scouted));
        let measured = Measured { errors: &errors, output: &at, group: &from };
        let means = measured.means(outputs, None);
        let (population, temperature): (Vec<Vec<f64>>, Vec<Vec<f64>>) =
            means.iter().map(|unit| (unit.iter().map(|m| m[0]).collect(), unit.iter().map(|m| m[1]).collect())).unzip();
        context.fixed.model = model_error(&sums(&means, unsampled), None, unsampled);
        // The tables' errors (SPEC §5.4): each table's whole error against the elimination itself, read at one state of
        // each pilot trajectory (`one_each`) and averaged over them -- a state on another cluster reads none of it. A table
        // beyond the tables' shares has the axis of its largest read halved and is read again, until it is within them or
        // its next grid would exceed the work's bound; what it reads then is booked on every output, and flagged where it
        // still lies beyond.
        let read_at = one_each(&samples, &at, outputs);
        let (shown_share, temperature_share) = crate::model::tabulate::shares();
        drop(run);
        let mut stuck = vec![false; tables.tables.len()];
        let booked = loop {
            let reads: Vec<Vec<crate::model::tabulate::TableRead>> = read_at.par_iter().map(|&j| model.table_reads(&tables, std::slice::from_ref(&samples[j].sample))).collect();
            let mean = |f: &dyn Fn(&crate::model::tabulate::TableRead) -> f64, t: usize| reads.iter().map(|r| f(&r[t])).sum::<f64>() / reads.len().max(1) as f64;
            let whole: Vec<[f64; 2]> = (0..tables.tables.len()).map(|t| [mean(&|r| r.whole[0].population, t), mean(&|r| r.whole[0].temperature, t)]).collect();
            let beyond = (0..whole.len()).find(|&t| !stuck[t] && (whole[t][0] > shown_share || whole[t][1] > temperature_share));
            let Some(t) = beyond else { break whole };
            let axes = tables.tables[t].axes();
            let weight = |f: usize| mean(&|r| r.axes[f][0].population / shown_share + r.axes[f][0].temperature / temperature_share, t);
            let worst = (0..axes).max_by(|&a, &b| weight(a).total_cmp(&weight(b)));
            match worst.and_then(|f| model.halved(&tables.tables[t], f, most)) {
                Some(finer) => tables.tables[t].table = finer,
                None => stuck[t] = true,
            }
        };
        for (t, table) in tables.tables.iter().enumerate() {
            let label = format!(
                "近共振光的表（{} 所在子空间{}）",
                model.state_label(model.clusters()[table.layout.cluster].states[0]),
                if table.layout.lights == model.running() { "" } else { "，光镊关闭时" }
            );
            for k in 1..outputs {
                context.fixed.model[k] = [(context.fixed.model[k][0] + booked[t][0]).min(1.), context.fixed.model[k][1] + booked[t][1]];
            }
            context.ledger.item(&label, booked[t][0].max(booked[t][1]), TABLE_MEASURED);
            if stuck[t] {
                context.ledger.flag(format!("{label}的误差超出表的份额，而加密已到工作量的界：按实测记账"));
            }
        }
        context.fixed.temperature = held_to(&estimates, &report.lab);
        // The step's check (module documentation, # The step): the clear part of the pilot's difference into the
        // accounts, its resolution into the result's words.
        let (over, held) = averaged_over(&context.fixed.temperature);
        let (step, resolution) = checked(&between(&pilot, &levels.coarse.iter().collect::<Vec<_>>(), &context.scales, &over), held);
        (context.step, context.checked) = (vec![step; outputs], Some(Checked { companion: 2. * tau, pairs: pilot_size(), resolution }));
        // A unit to keep at the pilot's states (`keep`): its model then runs a pilot again.
        if let Some((u, share)) = keep(request, light, kept, &model, reference, trap.centre, &units, &measured, &samples, &pilot, &estimates, &context.fixed, &context.scales, unsampled, pilot_size() as f64, barred) {
            return Ok(Attempt::Keep(units[u].clone(), tau, share));
        }
        // The ledger's items: each folded unit's error on what is shown and on T (its largest over the outputs, weighed by
        // the pilot's held share as the totals are; an output where the pilot holds no atom weighs nothing: its units'
        // errors are their bounds, `Measured::means`) with the item's place for its unit, and the unsampled share.
        let held: Vec<f64> = estimates.iter().map(|e| 1. - e.loss).collect();
        let weighed = |u: usize, k: usize| -> f64 { if held[k] > 0. { population[u][k].max(temperature[u][k]) * held[k] } else { 0. } };
        for (u, unit) in units.iter().enumerate() {
            context.ledger.item(&model.label(unit), (1..outputs).map(|k| weighed(u, k)).fold(0., f64::max), FOLD_MEASURED);
        }
        context.ledger.item("可达范围之外的初始原子份额", unsampled[0], "按初始系综计数（Wilson 远端），计入布居的模型误差");
        context.ledger.item("可达范围之外的初始原子的动能份额", unsampled[1], "按初始系综计数，计入 T 的模型误差");
        Run::tabulated(&model, &tables, reference)
    };

    // The product (module documentation, # Stopping): on from the pilot's trajectories at the run's step, the run's
    // first; the pilot's companions have done their part.
    let mut parts = levels.fine;
    let (e, l, c) = product(&run, &ensemble, &context, &mut parts, &numbers, messages);
    // A point of the prepared potential below the bottom the ensemble was drawn about (ensemble.rs, # Its draw).
    if ensemble.deeper() > 0 {
        context.ledger.flag("准备势在所取阱底之外还有更深的地方：初始系综按该阱底的深度封顶".into());
    }
    Ok(Attempt::Done(assemble(&context, &e, &l, &c, parts.len() as u64 * GROUP)))
}

/// The flag of a step whose error lies clearly beyond what it may take at the steps' cap (module documentation, # The
/// step): its clear part is booked as measured.
fn capped() -> String {
    format!("时间步长的误差明显超出它可占的部分，而每条轨迹的步数已达上限 {MOST_STEPS}，无法再减半：按读清的部分记账")
}

/// The pilot's two levels (module documentation, # The step): its groups at the step τ and the same groups, in the same
/// order, at 2τ.
#[derive(Default)]
struct Levels {
    fine: Vec<Part>,
    coarse: Vec<Part>,
}
impl Levels {
    /// The finer level's groups and their companions.
    fn paired(&self) -> (Vec<&Part>, Vec<&Part>) {
        (self.fine.iter().collect(), self.coarse.iter().collect())
    }
}

/// The members `members` of the groups `numbers` at the step `tau`, each trajectory through the plan (`run_groups`).
#[allow(clippy::too_many_arguments)]
fn level(run: &Run, ensemble: &Ensemble, plan: &Plan, tau: f64, numbers: &[u64], members: std::ops::Range<u64>, keep: bool, messages: &mut Messages, progress: &mut dyn FnMut(&[&Part], &mut Messages)) -> Vec<Part> {
    run_groups(numbers, members, Shape::of(plan, keep), &|i| trajectory(run, ensemble, tau, plan, keep, i), messages, progress)
}

/// Some groups' further members joined to those a level holds of them (all of them where it holds none).
fn absorb(level: &mut Vec<Part>, more: Vec<Part>) {
    if level.is_empty() {
        *level = more;
    } else {
        level.iter_mut().zip(more).for_each(|(p, m)| p.absorb(m));
    }
}

/// How the pilot's screen leaves the step (`settle`): standing; clearly beyond its share at the steps' cap; or at a
/// step that makes a priced keep dearer than the model before it (`attempt`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Settled {
    Standing,
    Capped,
    Dearer,
}

/// The members `members` of the pilot's groups `numbers` at both levels, joined to those the levels hold, and the
/// screen (module documentation, # The step): while some difference between the levels lies clearly beyond the step's
/// share among all the looks (`screen`), the step is halved -- the level at τ becomes the companion, and the members so
/// far run anew at τ/2 alone. The finer level keeps its states (the model is read at them).
#[allow(clippy::too_many_arguments)]
fn settle(run: &Run, ensemble: &Ensemble, context: &mut Context, levels: &mut Levels, numbers: &[u64], members: std::ops::Range<u64>, frequencies: &V3, dearer: &dyn Fn(f64) -> bool, messages: &mut Messages) -> Settled {
    let more = level(run, ensemble, &context.plan, context.tau, numbers, members.clone(), true, messages, &mut |prefix, messages| so_far(context, &levels.fine, prefix, messages));
    absorb(&mut levels.fine, more);
    let more = level(run, ensemble, &context.plan, 2. * context.tau, numbers, members.clone(), false, messages, &mut |_, messages| so_far(context, &levels.fine, &[], messages));
    absorb(&mut levels.coarse, more);
    loop {
        let beyond = {
            let (fine, coarse) = levels.paired();
            let (over, held) = averaged_over(&held_to(&pooled(&fine, &context.scales).0, frequencies));
            matches!(screen(&between(&fine, &coarse, &context.scales, &over), held), Judgement::Beyond(_))
        };
        let below = halved(context.tau, 1.);
        if !beyond {
            return Settled::Standing;
        } else if context.plan.end / below > MOST_STEPS as f64 {
            return Settled::Capped;
        } else if dearer(below) {
            return Settled::Dearer;
        }
        context.tau = below;
        let finer = level(run, ensemble, &context.plan, below, numbers, 0..members.end, true, messages, &mut |prefix, messages| so_far(context, &[], prefix, messages));
        levels.coarse = std::mem::replace(&mut levels.fine, finer);
    }
}

/// The trajectories a batch brings the product to, for the `need` that the estimates of `have` trajectories predict:
/// whole groups, more than `have`, at most twice as many, and `MOST_TRAJECTORIES` at most.
///
/// At most twice: the need is predicted from variances read on those trajectories' groups (the pilot's eleven at first,
/// whose largest read over the outputs runs high), and a batch takes the product no further than to twice what its
/// prediction rests on. What that bounds: the batches accumulate -- every trajectory of the product is in the result --,
/// so the product ends on the first total that meets the targets. Each total is at most g times one that did not meet
/// them, so the last total is at most g times the least that meets them, whatever was predicted: at g = 2 no more than
/// half of the work is spent past the need, and a need N times the pilot's size is reached in log₂ N batches.
fn batch(need: f64, have: u64) -> u64 {
    let whole = ((need / GROUP as f64).ceil() as u64).saturating_mul(GROUP);
    whole.clamp(have + GROUP, 2 * have).min(MOST_TRAJECTORIES)
}

/// The product (module documentation, # Stopping): batches of whole groups at the run's step, on from the pilot's
/// groups (their numbers `pilot`; the product's own follow in the list's order), to the number the looks predict
/// (`batch`), until every look that some number of trajectories can meet is met, or the trajectories reach
/// `MOST_TRAJECTORIES`. Each batch's result is a checkpoint; returns the estimates, the listed atoms and the counts at
/// its end.
fn product(run: &Run, ensemble: &Ensemble, context: &Context, parts: &mut Vec<Part>, pilot: &[u64], messages: &mut Messages) -> (Vec<Estimate>, Vec<Vec<(V3, V3)>>, Counts) {
    let (started, from) = (Instant::now(), parts.len() as u64 * GROUP);
    // The group at the list's place g.
    let number = |g: u64| pilot.get(g as usize).copied().unwrap_or(g);
    // The wall time (ms) the trajectories from `done` to `want` take at the product's pace so far; none before its own
    // first trajectories are in.
    let left = |done: u64, want: u64| (done > from).then(|| started.elapsed().as_secs_f64() * 1e3 / (done - from) as f64 * (want - done) as f64);
    loop {
        let all: Vec<&Part> = parts.iter().collect();
        let n = all.len() as u64 * GROUP;
        let (e, l, c) = pooled(&all, &context.scales);
        let chased: Vec<Look> = looks(&e, &context.fixed, &context.step).into_iter().filter(Look::fits).collect();
        if chased.iter().all(Look::met) || n >= MOST_TRAJECTORIES {
            return (e, l, c);
        }
        let want = batch(chased.iter().map(Look::need).fold(0., f64::max), n);
        messages.result(&assemble(context, &e, &l, &c, n), left(n, want));
        let numbers: Vec<u64> = (n / GROUP..want / GROUP).map(number).collect();
        let new = level(run, ensemble, &context.plan, context.tau, &numbers, 0..GROUP, false, messages, &mut |prefix, messages| {
            let all: Vec<&Part> = parts.iter().chain(prefix.iter().copied()).collect();
            let done = all.len() as u64 * GROUP;
            let (e, l, c) = pooled(&all, &context.scales);
            messages.result(&assemble(context, &e, &l, &c, done), left(done, want));
        });
        parts.extend(new);
    }
}

/// The pilot's size: the fewest whole groups whose unseen share fits the statistics' room in the population target beside
/// the step's share alone, z²/(N + z²) ≤ √(1 − s²) ε (`room`; estimate.rs `unseen` and its inverse `draws_for`, units.rs
/// `STEP_SHARE`) -- the smallest ensemble that can see a share as large as the target, 88 trajectories.
fn pilot_size() -> u64 {
    (draws_for(room(POPULATION_TARGET, 0., step_bound()[0])) / GROUP as f64).ceil() as u64 * GROUP
}

/// The outputs and T's windows (module documentation, # Outputs and windows): per output and lab axis the window's half
/// width, the outputs the distributions are taken at, and the run's end, the last window's.
struct Plan {
    times: Vec<f64>,
    halves: Vec<V3>,
    listed: Vec<usize>,
    end: f64,
    /// The cells of a trajectory's Brownian path (adapt.rs `Path`), µs: the companion's step at the attempt's first
    /// rung, a whole part of the output interval. Every step of the attempt -- a rung's, its companion's -- is a cell
    /// over a power of two, so every level reads the same path. The output interval until the first rung is known.
    cell: f64,
}
impl Plan {
    /// The plan of the request's times with the trap's frequency per lab axis (rad/µs).
    fn new(times: &[f64], frequencies: V3) -> Plan {
        let (interval, last) = (times[1] - times[0], times.len() - 1);
        let halves: Vec<V3> = times.iter().map(|&t| frequencies.map(|w| half_window(w, interval, t))).collect();
        let count = times.len().min(DISTRIBUTION_TIMES);
        let mut listed: Vec<usize> = (0..count).map(|i| ((i * last) as f64 / (count - 1) as f64).round() as usize).collect();
        listed.dedup();
        Plan { end: times[last] + halves[last].iter().fold(0., |a: f64, &b| a.max(b)), times: times.to_vec(), halves, listed, cell: interval }
    }
}

/// Counts over trajectories: the steps taken, the integrator's counted work (decide.rs `Work`) and its generator
/// assemblies, the time held (µs), and the trajectories the scheme could not advance (adapt.rs `advance`'s refusal) with
/// the earliest time one stopped at.
#[derive(Clone, Copy, Debug, Default)]
struct Counts {
    steps: u64,
    work: f64,
    assemblies: u64,
    held: f64,
    failed: u64,
    since: Option<f64>,
}
impl Counts {
    fn add(&mut self, o: &Counts) {
        self.steps += o.steps;
        self.work += o.work;
        self.assemblies += o.assemblies;
        self.held += o.held;
        self.failed += o.failed;
        self.since = match (self.since, o.since) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
    }
}

/// A pilot trajectory's state at an output (model.rs `Sample`) with the run's step and the trajectory's number: where
/// the units' errors are measured and a model is priced (`rate`).
#[derive(Clone)]
struct State {
    sample: Sample,
    tau: f64,
    index: u64,
}

/// What one trajectory leaves: what became of it by every output (estimate.rs `Fate`), its (r, κ) at the listed outputs
/// where held, its states at the outputs where held (`keep`: the pilot's, for the model's errors and prices), its counts.
struct Record {
    outputs: Vec<Fate>,
    listed: Vec<Option<(V3, V3)>>,
    samples: Vec<Option<State>>,
    counts: Counts,
}

/// One trajectory from its initial state through the plan at the step τ (module documentation, # Outputs and windows).
/// A trajectory the scheme cannot advance (adapt.rs) fails from the first output it does not reach.
fn run<D: Dynamics>(model: &D, start: Start, index: u64, tau: f64, plan: &Plan, keep: bool) -> Record {
    let outputs = plan.times.len();
    let interval = plan.times[1] - plan.times[0];
    let widest = plan.halves.iter().flatten().fold(0., |a: f64, &b| a.max(b));
    let mut t = Trajectory::new(model, start.state, start.r, start.kappa, SEED, index);
    let mut scratch = Scratch::default();
    // The trajectory's Brownian path in the plan's cells, its grid the multiples of τ (a cell over a power of two):
    // the same path at every rung and at its companion.
    let mut path = adapt::Path::new(SEED, index, plan.cell, (plan.cell / tau).log2().round() as u32);
    // Per output and lab axis: the window's time held, ∫κ dt and ∫κ² dt.
    let mut moments = vec![[[0.; 3]; 3]; outputs];
    let mut out = vec![Fate::Lost; outputs];
    let mut listed = vec![None; plan.listed.len()];
    let mut samples: Vec<Option<State>> = vec![None; if keep { outputs } else { 0 }];
    let mut steps = 0;
    // The first output the trajectory does not reach held, and whether the scheme could not advance it.
    let (mut reached, mut failed) = (outputs, false);
    let mut record = |k: usize, t: &Trajectory, out: &mut Vec<Fate>| {
        // The atom at the output's instant; from the first output on its κ and κ² are its window's means (below).
        out[k] = Fate::Held(Held { rows: model.readout(t.cluster, &t.point(), &t.psi), kappa: t.kappa, kappa2: t.kappa.map(|x| x * x) });
        if let Ok(j) = plan.listed.binary_search(&k) {
            listed[j] = Some((t.r, t.kappa));
        }
        if keep {
            let sample = Sample { cluster: t.cluster, at: t.point(), psi: t.psi.to_vec() };
            samples[k] = Some(State { sample, tau, index });
        }
    };
    record(0, &t, &mut out);
    {
        let mut each = |s: &Segment| {
            steps += 1;
            let (s0, s1) = (s.start, s.start + s.tau);
            let first = (((s0 - widest) / interval).floor().max(1.)) as usize;
            let last = (((s1 + widest) / interval).ceil() as usize).min(outputs - 1);
            for k in first..=last {
                // The step's moments over an axis's cut of it (`Segment::moments` gives every axis's): made once for
                // the axes whose windows cut the step alike.
                let mut made: [Option<((f64, f64), (V3, V3))>; 3] = [None; 3];
                for a in 0..3 {
                    let h = plan.halves[k][a];
                    let (lo, hi) = (s0.max(plan.times[k] - h), s1.min(plan.times[k] + h));
                    if hi > lo {
                        let (m1, m2) = match made[..a].iter().flatten().find(|(cut, _)| *cut == (lo, hi)) {
                            Some((_, same)) => *same,
                            None => s.moments(lo, hi),
                        };
                        made[a] = Some(((lo, hi), (m1, m2)));
                        let w = &mut moments[k][a];
                        w[0] += hi - lo;
                        w[1] += m1[a];
                        w[2] += m2[a];
                    }
                }
            }
        };
        for k in 1..outputs {
            failed = adapt::advance(model, tau, &mut t, &mut scratch, &mut path, plan.times[k], &mut each).is_err();
            if failed || t.lost.is_some() {
                reached = k;
                break;
            }
            record(k, &t, &mut out);
        }
        if reached == outputs {
            // Through the last output's window; a failure there leaves every output read.
            failed = adapt::advance(model, tau, &mut t, &mut scratch, &mut path, plan.end, &mut each).is_err();
        }
    }
    // From the output it does not reach: failed, or lost.
    if reached < outputs {
        out[reached..].fill(if failed { Fate::Failed } else { Fate::Lost });
    }
    for (k, o) in out.iter_mut().enumerate().skip(1) {
        if let Fate::Held(h) = o {
            let w = &moments[k];
            h.kappa = std::array::from_fn(|a| w[a][1] / w[a][0]);
            h.kappa2 = std::array::from_fn(|a| w[a][2] / w[a][0]);
        }
    }
    let counts = Counts { steps, work: t.work, assemblies: t.assemblies, held: t.lost.unwrap_or(t.time).min(plan.end), failed: u64::from(failed), since: failed.then_some(t.time) };
    Record { outputs: out, listed, samples, counts }
}

/// What a group of trajectories leaves: its sums (estimate.rs `Group`), its held atoms' (r, κ) at the listed outputs, its
/// states at the outputs (the pilot's), its counts.
struct Part {
    sums: Group,
    listed: Vec<Vec<(V3, V3)>>,
    samples: Vec<Vec<State>>,
    counts: Counts,
}
impl std::borrow::Borrow<Group> for &Part {
    fn borrow(&self) -> &Group {
        &self.sums
    }
}
impl Part {
    /// The same groups' further members, after this part's.
    fn absorb(&mut self, other: Part) {
        self.sums.absorb(other.sums);
        for (l, x) in self.listed.iter_mut().zip(other.listed) {
            l.extend(x);
        }
        for (s, x) in self.samples.iter_mut().zip(other.samples) {
            s.extend(x);
        }
        self.counts.add(&other.counts);
    }
}

/// What a group's records fold into (`Part`): the outputs, the listed outputs, and whether the states are kept.
#[derive(Clone, Copy)]
struct Shape {
    outputs: usize,
    listed: usize,
    keep: bool,
}
impl Shape {
    fn of(plan: &Plan, keep: bool) -> Shape {
        Shape { outputs: plan.times.len(), listed: plan.listed.len(), keep }
    }
    /// A group's records added in their order.
    fn fold(&self, records: impl Iterator<Item = Record>) -> Part {
        let mut part = Part {
            sums: Group::new(self.outputs),
            listed: vec![vec![]; self.listed],
            samples: vec![vec![]; if self.keep { self.outputs } else { 0 }],
            counts: Counts::default(),
        };
        for r in records {
            part.sums.add(&r.outputs);
            for (l, x) in part.listed.iter_mut().zip(r.listed) {
                l.extend(x);
            }
            for (s, x) in part.samples.iter_mut().zip(r.samples) {
                s.extend(x);
            }
            part.counts.add(&r.counts);
        }
        part
    }
}

/// Trajectory `index` of a run at the step τ: its initial state from the ensemble (ensemble.rs `Ensemble::draw`), run
/// through the plan (`run`). A function of the index and the step: the same index at another step is the same atom on
/// the same streams (module documentation, # The step).
fn trajectory(model: &Run, ensemble: &Ensemble, tau: f64, plan: &Plan, keep: bool, index: u64) -> Record {
    run(model, ensemble.draw(SEED, index), index, tau, plan, keep)
}

/// The groups of the numbers `groups`, the members `members` of each (all of them: 0..GROUP), their trajectories run on every core in
/// their numbers' order -- each thread takes the next number, so the groups finish nearly in order and the first after
/// about one trajectory's time --, each group's records added in
/// its members' order on this thread (`Shape::fold`, nothing depends on the threads), the groups returned in their order;
/// whenever the solve's next message is due (`Messages::due`), `progress` takes the groups finished so far in order and
/// sends their result, or with no group finished the last result goes out again (module documentation, # Checkpoints).
fn run_groups(groups: &[u64], members: std::ops::Range<u64>, shape: Shape, record: &(dyn Fn(u64) -> Record + Sync), messages: &mut Messages, progress: &mut dyn FnMut(&[&Part], &mut Messages)) -> Vec<Part> {
    let size = members.end - members.start;
    let count = groups.len();
    let stop = count as u64 * size;
    // The k-th trajectory run: member k mod size of the k / size-th group.
    let number = move |k: u64| groups[(k / size) as usize] * GROUP + members.start + k % size;
    let next = AtomicU64::new(0);
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(stop as usize);
    let mut waiting: Vec<Vec<Option<Record>>> = (0..count).map(|_| (0..size).map(|_| None).collect()).collect();
    let mut parts: Vec<Option<Part>> = (0..count).map(|_| None).collect();
    std::thread::scope(|scope| {
        let (sender, receiver) = mpsc::channel::<(u64, Record)>();
        for _ in 0..threads {
            let (sender, next) = (sender.clone(), &next);
            scope.spawn(move || loop {
                let k = next.fetch_add(1, Ordering::Relaxed);
                if k >= stop || sender.send((k, record(number(k)))).is_err() {
                    break;
                }
            });
        }
        drop(sender);
        let mut done = 0;
        while done < count {
            match receiver.recv_timeout(messages.due()) {
                Ok((k, r)) => {
                    let (g, m) = ((k / size) as usize, (k % size) as usize);
                    waiting[g][m] = Some(r);
                    if waiting[g].iter().all(Option::is_some) {
                        parts[g] = Some(shape.fold(waiting[g].iter_mut().map(|r| r.take().expect("a member's record"))));
                        done += 1;
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    let prefix: Vec<&Part> = parts.iter().map_while(Option::as_ref).collect();
                    if prefix.is_empty() {
                        messages.again();
                    } else {
                        progress(&prefix, messages);
                    }
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
    });
    parts.into_iter().map(|p| p.expect("a group's trajectories ended without their part")).collect()
}

/// What the pilot fixes for the run (module documentation, # Stopping), per output: the model's error on the held atoms
/// (populations absolute, T relative; T's unbounded where no pilot atom reads it), and per lab axis whether that axis's
/// T is held to its target; and T(0)'s relative error, its quadrature's last two levels' difference (ensemble.rs
/// `Ensemble::kinetic`).
#[derive(Clone)]
struct Fixed {
    model: Vec<[f64; 2]>,
    temperature: Vec<[bool; 3]>,
    start: f64,
}

/// The step's share of the targets on what is shown and on T (relative), s·ε (SPEC §2.2, §4): what its error may take
/// -- the pilot is screened against it, the model's decisions take it for the step's error, and a result is booked at
/// it until its two levels are read.
fn step_bound() -> [f64; 2] {
    [STEP_SHARE * POPULATION_TARGET, STEP_SHARE * TEMPERATURE_TARGET]
}

/// The room the statistics may take of a target ε beside a model error A and a step error b, √(ε² − A² − b²) (module
/// documentation, # Stopping); none where the two fill the target.
fn room(target: f64, model: f64, step: f64) -> f64 {
    (target * target - model * model - step * step).max(0.).sqrt()
}

/// One quantity held to its target at one output (module documentation, # Stopping): what is shown, or T on a lab axis
/// held there. Its account (ledger.rs: the model's error A, the step's clear part c, the statistics S, as the ledger
/// books them) against its target ε; and what its need takes beside them: the statistics' jackknife interval and the
/// trajectories it was read on, and on what is shown the unseen share, which the statistics are no less than.
#[derive(Clone, Copy, Debug)]
struct Look {
    output: usize,
    target: f64,
    account: Account,
    interval: f64,
    trajectories: f64,
    shown: bool,
}
impl Look {
    /// Whether its total lies within its target: the ledger's own test of an output (ledger.rs `Ledger::close`).
    fn met(&self) -> bool {
        self.account.total() <= self.target
    }
    /// The room its statistics may take (`room`).
    fn room(&self) -> f64 {
        room(self.target, self.account.model, self.account.step)
    }
    /// Whether some number of trajectories brings it within its target: the model's error and the step's leave a room.
    fn fits(&self) -> bool {
        self.room() > 0.
    }
    /// The trajectories it needs: its interval, falling as N^{−1/2}, within its room; on what is shown no fewer than the
    /// draws that leave the room unseen (estimate.rs `draws_for`). None where no number fits.
    fn need(&self) -> f64 {
        let room = self.room();
        let draws = self.trajectories * (self.interval / room).powi(2);
        if self.shown { draws.max(draws_for(room)) } else { draws }
    }
}

/// The looks of a run (module documentation, # Stopping) by its estimates, the pilot's fixed errors and the step's
/// part: at every output after t = 0 what is shown, and T on each lab axis held to its target there. A held axis
/// whose T has no account -- one of its parts is unread -- is not met; where it is its statistics that are unread, its
/// need is predicted from the normal theory of a variance over the N h atoms held, the relative interval z √(2/(N h)).
fn looks(estimates: &[Estimate], fixed: &Fixed, step: &[Bound]) -> Vec<Look> {
    let (population, temperature) = accounts(estimates, fixed, step);
    let mut looks = vec![];
    for (k, e) in estimates.iter().enumerate().skip(1) {
        let (h, n) = (1. - e.loss, e.trajectories);
        looks.push(Look { output: k, target: POPULATION_TARGET, account: population[k], interval: e.shown_interval(), trajectories: n, shown: true });
        for a in (0..3).filter(|&a| fixed.temperature[k][a]) {
            let account = temperature[k][a].unwrap_or(Account { model: h * fixed.model[k][1], step: h * step[k].temperature[a], statistics: f64::INFINITY });
            let interval = if account.statistics.is_finite() { account.statistics } else { h * CONFIDENCE * (2. / (n * h)).sqrt() };
            looks.push(Look { output: k, target: TEMPERATURE_TARGET, account, interval, trajectories: n, shown: false });
        }
    }
    looks
}

/// The outputs where some look leaves the statistics no room (SPEC §5.2): no number of trajectories fits.
fn crowded(looks: &[Look]) -> Vec<usize> {
    let mut outputs: Vec<usize> = looks.iter().filter(|l| !l.fits()).map(|l| l.output).collect();
    outputs.dedup();
    outputs
}

/// Each output's accounts of what is shown and of T per lab axis (ledger.rs `Account`) by the run's estimates, the model's
/// errors on the held atoms per output (`Fixed`) and the step's part, T's where read: the held share 1 − L weighs the
/// model's error on the rows as shown, and T's whole accounts (module documentation, # Stopping); t = 0 is the
/// request's state, without error, and the ensemble's moments, T's by quadrature: its accounts are the quadrature's
/// error alone (`Fixed::start`), in the discretization's part (SPEC §4). The model's error on what is shown is at most
/// 1, a probability's. An axis's T has an account where its statistics, the model's error and the axis's step error
/// are all read.
fn accounts(estimates: &[Estimate], fixed: &Fixed, step: &[Bound]) -> (Vec<Account>, Vec<[Option<Account>; 3]>) {
    estimates
        .iter()
        .zip(&fixed.model)
        .zip(step)
        .enumerate()
        .map(|(k, ((e, a), b))| {
            if k == 0 {
                return (Account::default(), [Some(Account { step: fixed.start, ..Account::default() }); 3]);
            }
            let h = 1. - e.loss;
            let population = Account { model: (a[0] * h).min(1.), step: b.shown, statistics: e.shown_statistics() };
            let temperature = std::array::from_fn(|x| {
                let relative = e.motion.as_ref().map_or(f64::INFINITY, |m| e.temperature_interval(x, m.temperature[x]));
                let bound = b.temperature[x];
                (relative.is_finite() && a[1].is_finite() && bound.is_finite()).then_some(Account { model: h * a[1], step: h * bound, statistics: h * relative })
            });
            (population, temperature)
        })
        .unzip()
}

/// What the pilot measured of the model's units (SPEC §5.2): each folded unit's errors at the pilot's states, and each
/// state's output and group.
struct Measured<'a> {
    errors: &'a [Vec<UnitError>],
    output: &'a [usize],
    group: &'a [usize],
}
impl Measured<'_> {
    /// Per unit and output the held atoms' mean error (on their populations, absolute; on T, relative) over the pilot's
    /// groups but `left`. Where they hold no atom nothing is measured: the error is its bound there, 1 on a population and
    /// none on T (the run's atoms at such an output are at most the share the pilot left unseen, and T is not held to its
    /// target without a pilot's T).
    fn means(&self, outputs: usize, left: Option<usize>) -> Vec<Vec<[f64; 2]>> {
        self.errors
            .iter()
            .map(|unit| {
                let (mut sum, mut count) = (vec![[0.; 2]; outputs], vec![0.; outputs]);
                for ((e, &k), &g) in unit.iter().zip(self.output).zip(self.group) {
                    if Some(g) != left {
                        sum[k][0] += e.population;
                        sum[k][1] += e.temperature;
                        count[k] += 1.;
                    }
                }
                sum.iter().zip(&count).map(|(s, &c)| if c > 0. { [s[0] / c, s[1] / c] } else { [1., f64::INFINITY] }).collect()
            })
            .collect()
    }
}

/// The units' mean errors added up per output (populations, T; SPEC §4), every folded unit's and what the reach leaves
/// out of the initial ensemble (of the atoms, of their kinetic energy; model/reach.rs `Model::left_out`).
fn sums(means: &[Vec<[f64; 2]>], unsampled: [f64; 2]) -> Vec<[f64; 2]> {
    let outputs = means.first().map_or(0, Vec::len);
    (0..outputs).map(|k| means.iter().fold(unsampled, |a, m| [a[0] + m[k][0], a[1] + m[k][1]])).collect()
}

/// The model's error per output from the sums `whole` of its units' (`sums`), less one unit's where `without` gives its
/// mean errors and the units' number: none at t = 0, the request's state; a population's at most 1. A read error leaves
/// the sum by subtraction -- an output's work, not every unit's again; at an output no state was read at, every unit's
/// error is its bound alike (`Measured::means`), and the others' bounds stand.
fn model_error(whole: &[[f64; 2]], without: Option<(&[[f64; 2]], usize)>, unsampled: [f64; 2]) -> Vec<[f64; 2]> {
    let less = |k: usize, i: usize| match without {
        None => whole[k][i],
        Some((unit, _)) if unit[k][i].is_finite() => whole[k][i] - unit[k][i],
        Some((_, units)) => if units > 1 { f64::INFINITY } else { unsampled[i] },
    };
    (0..whole.len()).map(|k| if k == 0 { [0.; 2] } else { [less(k, 0).min(1.), less(k, 1)] }).collect()
}

/// The counted work (decide.rs `Work`) a trajectory spends per µs at a pilot state under a model: one step's -- the
/// integrator's (adapt.rs `advance_work`) on the path of the state's own momentum, velocity and τ through `middle`, its
/// cluster's middle at the trap's centre (a price needs the generator's sizes, not the state's own point: one middle a
/// cluster serves every state of a model), and its middles' (`assembly`, the model's count of one) -- over its τ.
fn rate(run: &Run, assembly: f64, middle: &crate::model::Middle, state: &State) -> f64 {
    let at = state.sample.at;
    let velocity = at.kappa.map(|k| run.hbar_over_mass() * k);
    let (work, assemblies) = adapt::advance_work(middle, at.kappa, velocity, state.tau);
    (work + assemblies as f64 * assembly) / state.tau
}

/// One state of each pilot trajectory among `states` (their outputs in `output`): its outputs taken in turn by its
/// number -- the trajectory is the pilot's independent unit.
fn one_each(states: &[State], output: &[usize], outputs: usize) -> Vec<usize> {
    (0..states.len()).filter(|&j| output[j] as u64 == 1 + states[j].index % (outputs as u64 - 1)).collect()
}

/// A unit to keep (`Pricing::priced`) and its predicted saving.
struct Candidate {
    unit: usize,
    saving: Read,
}

/// The pilot's groups but one (or all of them) as the pricing reads them (`Pricing`): the estimates and the units' mean
/// errors over them, and the current model's W there.
struct Level {
    left: Option<usize>,
    estimates: Vec<Estimate>,
    means: Vec<Vec<[f64; 2]>>,
    /// The units' mean errors added up (`sums`).
    whole: Vec<[f64; 2]>,
    /// The trajectories the current model needs there.
    need: f64,
}

/// The pricing of keeps (SPEC §5.2; module documentation, # The model): the predicted saving of total work W = w·N by
/// keeping a folded unit, as a share of the current model's -- N the trajectories the looks need at the model's
/// errors and the step's share (`Look::need` on the pilot's estimates), and for the kept model the pilot its new attempt runs as well (the
/// current one's is spent); w the mean of `rate` --, spread by the jackknife over the pilot's groups (each recomputed
/// without one: its estimates, the units' errors, the rates). A share, not the work saved: both models' N are the one
/// variance the pilot estimates over their rooms (N ∝ σ̂²), and σ̂²'s spread, common to both, scales the work saved
/// without ever changing its sign; in the share it cancels, and what spreads is what the keep changes -- the rooms and
/// the rates. The rates are read at one state of each pilot trajectory -- the trajectory is the pilot's independent
/// unit, its outputs taken in turn by its number --, the same states for every model.
struct Pricing<'a> {
    request: &'a Request,
    light: &'a Light,
    kept: &'a [Unit],
    model: &'a Model,
    reference: [f64; GROUND],
    centre: V3,
    units: &'a [Unit],
    states: &'a [State],
    measured: &'a Measured<'a>,
    pilot: &'a [&'a Part],
    fixed: &'a Fixed,
    scales: &'a Scales,
    unsampled: [f64; 2],
    /// The states the rates are read at and the current model's rates there.
    rated: Vec<usize>,
    own: Vec<f64>,
    /// The trajectories a kept model's attempt runs again before it stands where this one does.
    again: f64,
    /// Every group together.
    all: Level,
}

impl<'a> Pricing<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        request: &'a Request,
        light: &'a Light,
        kept: &'a [Unit],
        model: &'a Model,
        reference: [f64; GROUND],
        centre: V3,
        units: &'a [Unit],
        measured: &'a Measured<'a>,
        states: &'a [State],
        pilot: &'a [&'a Part],
        fixed: &'a Fixed,
        scales: &'a Scales,
        unsampled: [f64; 2],
        again: f64,
    ) -> Pricing<'a> {
        let rated = one_each(states, measured.output, fixed.model.len());
        let all = Level { left: None, estimates: vec![], means: vec![], whole: vec![], need: 0. };
        let mut pricing = Pricing { request, light, kept, model, reference, centre, units, states, measured, pilot, fixed, scales, unsampled, rated, own: vec![], again, all };
        pricing.all = pricing.level(None);
        pricing
    }

    /// The current model's rates at the rated states (`own`): read once a unit's keep saves at all (`saving_at_all`
    /// asks no rate), before any saving is priced.
    fn rate(&mut self) {
        let current = Run::new(self.model, self.reference);
        let assembly = current.assembly_work();
        let middles: Vec<crate::model::Middle> = (0..self.model.clusters().len()).map(|c| current.middle(c, self.centre, 0.)).collect();
        self.own = self.rated.iter().map(|&j| rate(&current, assembly, &middles[self.states[j].sample.cluster], &self.states[j])).collect();
    }

    /// The pilot's groups but `left` as the pricing reads them. Its pooling runs on the thread pool: made on the calling
    /// thread, never inside one of the pool's tasks.
    fn level(&self, left: Option<usize>) -> Level {
        let parts: Vec<&Part> = self.pilot.iter().enumerate().filter(|(g, _)| Some(*g) != left).map(|(_, p)| *p).collect();
        let (estimates, means) = (pooled(&parts, self.scales).0, self.measured.means(self.fixed.model.len(), left));
        let whole = sums(&means, self.unsampled);
        let need = self.need(&estimates, &whole, None);
        Level { left, estimates, means, whole, need }
    }

    /// The jackknife's levels: each group left out.
    fn leave_one_out(&self) -> Vec<Level> {
        (0..self.pilot.len()).map(|g| self.level(Some(g))).collect()
    }

    /// The mean of the rates over the rated states of the groups but `left`.
    fn mean(&self, rates: &[f64], left: Option<usize>) -> f64 {
        let (sum, count) = self.rated.iter().zip(rates).filter(|(&j, _)| Some(self.measured.group[j]) != left).fold((0., 0.), |a, (_, r)| (a.0 + r, a.1 + 1.));
        sum / count
    }

    /// The trajectories the looks need beside the units' errors but `without`'s and the step's share.
    fn need(&self, estimates: &[Estimate], whole: &[[f64; 2]], without: Option<&[[f64; 2]]>) -> f64 {
        let fixed = Fixed { model: model_error(whole, without.map(|unit| (unit, self.units.len())), self.unsampled), ..self.fixed.clone() };
        looks(estimates, &fixed, &vec![Bound::share(); fixed.model.len()]).iter().map(Look::need).fold(0., f64::max)
    }

    /// The share of the current model's W at a level that keeping unit `u` saves at the rates `rates` of its kept model:
    /// a model's W (per µs of run) is its mean rate times its product's trajectories (`need`), the kept model's with the
    /// pilot its new attempt runs.
    fn share(&self, level: &Level, u: usize, rates: &[f64]) -> f64 {
        1. - self.mean(rates, level.left) * (self.need(&level.estimates, &level.whole, Some(&level.means[u])) + self.again) / (self.mean(&self.own, level.left) * level.need)
    }

    /// The saving of keeping unit `u` at the rates `rates` of its kept model, spread over the jackknife's levels `others`.
    fn saving(&self, others: &[Level], u: usize, rates: &[f64]) -> Read {
        Read { value: self.share(&self.all, u, rates), spread: spread(others.iter().map(|l| self.share(l, u, rates)), others.len() as f64) }
    }

    /// The units whose keep would save at all were its model to cost no more per µs than the current one -- the
    /// trajectories its error's removal saves, less the new pilot (module documentation, # The model). A saving that is
    /// none before its spread pays at no confidence: only these are read on the jackknife's levels and given a kept
    /// model.
    fn saving_at_all(&self) -> Vec<usize> {
        // No unit's removal saves more than the removal of them all (a look's need falls with the model's error):
        // where even that does not pay for the new pilot, none is tried.
        if self.need(&self.all.estimates, &vec![self.unsampled; self.all.whole.len()], None) + self.again >= self.all.need {
            return vec![];
        }
        (0..self.units.len()).into_par_iter().filter(|&u| self.need(&self.all.estimates, &self.all.whole, Some(&self.all.means[u])) + self.again < self.all.need).collect()
    }

    /// The rates of the model that keeps unit `u`, built for it, at the rated states: a keep costs less per µs than the
    /// fold it replaces or far more (a drive kept across a fine-structure splitting, its generator's norm ~10⁷ rad/µs).
    /// None for a unit the model refuses to keep (its fold must exist, `Model::new`): its fold's error stands.
    fn kept_rates(&self, u: usize) -> Option<Vec<f64>> {
        let with = Model::new(self.request, self.light, &[self.kept, std::slice::from_ref(&self.units[u])].concat()).ok()?;
        let finer = Run::new(&with, self.reference);
        let assembly = finer.assembly_work();
        let middles: Vec<crate::model::Middle> = (0..with.clusters().len()).map(|c| finer.middle(c, self.centre, 0.)).collect();
        Some(self.rated.iter().map(|&j| rate(&finer, assembly, &middles[finer.cluster_of(self.model.clusters()[self.states[j].sample.cluster].states[0])], &self.states[j])).collect())
    }

    /// Unit `u` with its kept model built and rated, its saving spread over the jackknife's levels `others`.
    fn priced(&self, others: &[Level], u: usize) -> Option<Candidate> {
        let rates = self.kept_rates(u)?;
        Some(Candidate { unit: u, saving: self.saving(others, u, &rates) })
    }
}

/// Histograms of one coordinate of the held atoms at the listed outputs on one grid symmetric about 0 that holds them all
/// (the bins' centres): a listed output with no atom carries the one before it. The number of bins is for the display
/// alone -- no estimate reads a histogram --: the square-root choice, ⌊√n⌋ for the n atoms of the fullest listed output
/// (bins of about √n atoms each), made odd so that 0 is a bin's centre, and three at least -- the contract's grid has two
/// points or more (contract §2.2), and the least odd count that has is three.
fn histograms(listed: &[Vec<(V3, V3)>], outputs: &[usize], value: &dyn Fn(&(V3, V3)) -> f64) -> Distributions {
    let reach = listed.iter().flatten().map(|s| value(s).abs()).fold(f64::MIN_POSITIVE, f64::max);
    let most = listed.iter().map(Vec::len).fold(0, usize::max);
    let bins = ((most as f64).sqrt() as usize | 1).max(3);
    let width = 2. * reach / (bins - 1) as f64;
    let grid: Vec<f64> = (0..bins).map(|i| (i as f64 - (bins / 2) as f64) * width).collect();
    let mut last: Vec<f64> = (0..bins).map(|i| if i == bins / 2 { 1. } else { 0. }).collect();
    let probabilities = listed
        .iter()
        .map(|s| {
            if !s.is_empty() {
                last = vec![0.; bins];
                for x in s {
                    let i = ((value(x) / width).round() as isize + (bins / 2) as isize).clamp(0, bins as isize - 1) as usize;
                    last[i] += 1. / s.len() as f64;
                }
            }
            last.clone()
        })
        .collect();
    Distributions { grid, samples: outputs.to_vec(), probabilities }
}

/// What the result's assembly takes beside the estimates: the plan, the run's step (µs), the scales, the request's initial
/// diagonal, the exact moments at t = 0 ((ħ/m)⟨κ_a²⟩ per lab axis, rad/µs), the pilot's fixed errors, the step's part of
/// every output's accounts -- the step's share until the pilot's two levels are `checked` --, the ledger's items and
/// note, and the method's description.
struct Context {
    plan: Plan,
    tau: f64,
    scales: Scales,
    initial: [f64; STATES],
    momentum: V3,
    fixed: Fixed,
    step: Vec<Bound>,
    checked: Option<Checked>,
    ledger: Ledger,
    method: String,
}

/// The result of the estimates of `trajectories` trajectories (contract §2.2): at t = 0 the request's initial state and the
/// ensemble's exact moments (T, and n̄ as of one atom with that mean, estimate.rs `occupation`); an output with no atom
/// held carries the last one's populations, one with fewer than two atoms the last T read (its error unknown); each
/// flagged.
fn assemble(c: &Context, estimates: &[Estimate], listed: &[Vec<(V3, V3)>], counts: &Counts, trajectories: u64) -> Solution {
    let (times, outputs) = (&c.plan.times, c.plan.times.len());
    let hm = c.scales.hbar_over_mass;
    let w = c.scales.frequencies;
    let classical = |t: &V3| -> [bool; 3] { std::array::from_fn(|a| classical(t[a], w[a])) };
    let mut ledger = c.ledger.clone();
    let (population, temperature) = accounts(estimates, &c.fixed, &c.step);
    let held = c.fixed.temperature.clone();
    // The step's item: the largest it is booked at over the accounts (weighed by the held share as the totals are). And
    // the check in the method's words: the two steps compared, the pairs, the resolution -- a check, in no total.
    let mut step = String::new();
    if let Some(check) = &c.checked {
        let largest = (1..outputs).flat_map(|k| temperature[k].iter().flatten().map(|a| a.step).chain([population[k].step])).fold(0., f64::max);
        let share = (1 << ORDER) - 1;
        ledger.item(&format!("时间步长 τ = {} μs（全程不变）", c.tau), largest, &format!("试探的轨迹在 τ 与 2τ 下逐条配对比较：两者之差的 1/{share} 在全程上的平均中读清的部分"));
        let relative = if check.resolution[1].is_finite() { format!("±{}", percent(check.resolution[1])) } else { "读不出".into() };
        step = format!("（收敛检查：与 {} μs 在 {} 对同号轨迹上比较，全程平均的差别能分辨到布居 ±{}、温度 {relative}；这是检查，不是误差界，未计入总误差）", check.companion, check.pairs, percent(check.resolution[0]));
    }
    ledger.close(times, &population, &temperature, &held);
    // T and the motion at t = 0: exact.
    let start_t = c.momentum.map(|m| m / KT_PER_UK);
    let mut s = Solution {
        populations: Vec::with_capacity(outputs),
        estimated_errors: population.iter().map(Account::total).collect(),
        temperature_uk: Vec::with_capacity(outputs),
        temperature_errors_uk: Vec::with_capacity(outputs),
        temperature_total_uk: Vec::with_capacity(outputs),
        temperature_total_errors_uk: Vec::with_capacity(outputs),
        mean_velocity_mps: Vec::with_capacity(outputs),
        mean_occupation: Vec::with_capacity(outputs),
        trap_frequency_khz: w.map(|x| x / TAU * 1e3),
        loss_probability: estimates.iter().enumerate().map(|(k, e)| if k == 0 { 0. } else { e.loss }).collect(),
        velocity: std::array::from_fn(|a| histograms(listed, &c.plan.listed, &|x| x.1[a] * hm)),
        position: std::array::from_fn(|a| histograms(listed, &c.plan.listed, &|x| x.0[a])),
        classical: Vec::with_capacity(outputs),
        temperature_held: held,
        temperature_estimated_errors: temperature.iter().map(|axes| axes.map(|a| a.map(|a| a.total()))).collect(),
        trajectories,
        time_step_us: c.tau,
        method: format!("{}；{trajectories} 条轨迹（每组 {GROUP} 条拉丁超立方），步长 {} μs{step}。", c.method, c.tau),
        reason: None,
        approximations: ledger,
    };
    let (mut empty, mut unread) = (None, vec![]);
    for (k, e) in estimates.iter().enumerate() {
        if k == 0 {
            s.populations.push(c.initial);
            s.temperature_uk.push(start_t);
            s.temperature_errors_uk.push([Some(0.); 3]);
            s.temperature_total_uk.push(start_t.iter().sum::<f64>() / 3.);
            s.temperature_total_errors_uk.push(Some(0.));
            s.mean_velocity_mps.push([0.; 3]);
            s.mean_occupation.push(std::array::from_fn(|a| occupation(c.momentum[a], 1., w[a])));
            s.classical.push(classical(&start_t));
            continue;
        }
        let last = k - 1;
        s.populations.push(if e.held > 0. { e.populations } else { s.populations[last] });
        if e.held == 0. && empty.is_none() {
            empty = Some(k);
        }
        match &e.motion {
            Some(m) => {
                let read = |x: &crate::estimate::Spread| x.read().then_some(x.error);
                s.temperature_uk.push(m.temperature);
                s.temperature_errors_uk.push(m.temperature_errors.each_ref().map(read));
                s.temperature_total_uk.push(m.temperature_total);
                s.temperature_total_errors_uk.push(read(&m.temperature_total_error));
                s.mean_velocity_mps.push(m.mean_velocity);
                s.mean_occupation.push(m.occupation);
                s.classical.push(classical(&m.temperature));
                if m.temperature_errors.iter().any(|x| !x.read()) && e.held > 0. {
                    unread.push(k);
                }
            }
            None => {
                let (t, total, v, n) = (s.temperature_uk[last], s.temperature_total_uk[last], s.mean_velocity_mps[last], s.mean_occupation[last]);
                s.temperature_uk.push(t);
                s.temperature_errors_uk.push([None; 3]);
                s.temperature_total_uk.push(total);
                s.temperature_total_errors_uk.push(None);
                s.mean_velocity_mps.push(v);
                s.mean_occupation.push(n);
                s.classical.push([false; 3]);
                if e.held > 0. {
                    unread.push(k);
                }
            }
        }
    }
    if let Some(k) = empty {
        s.approximations.flag(format!("{} 起已无原子被俘获（丢失概率 1）：此后的布居、温度与分布沿用此前的值，仅供参考", when(times[k])));
    }
    if !unread.is_empty() {
        s.approximations.flag(format!("{}：被俘获的原子不足两个组，温度沿用此前读到的值，误差未知", span(&unread, times)));
    }
    // T shown with its statistics where its model error or its step error is unread.
    let unbooked: Vec<usize> = (1..outputs)
        .filter(|&k| estimates[k].motion.as_ref().is_some_and(|m| m.temperature_errors.iter().all(Spread::read)) && !(c.fixed.model[k][1].is_finite() && c.step[k].temperature.iter().all(|b| b.is_finite())))
        .collect();
    if !unbooked.is_empty() {
        s.approximations.flag(format!("{}：这些时刻被俘获的原子不足，模型或步长对温度的误差读不出", span(&unbooked, times)));
    }
    if let Some(t) = counts.since {
        s.approximations.flag(format!("{} 条轨迹的状态在数值上无法再推进（最早在 {}）：此后不计入结果", counts.failed, when(t)));
    }
    // The worst output beyond the population target: what keeps it there.
    if let Some((k, a)) = population.iter().enumerate().max_by(|x, y| x.1.total().total_cmp(&y.1.total())) {
        if a.total() > POPULATION_TARGET {
            let cause = if a.statistics >= a.model.max(a.step) {
                if trajectories >= MOST_TRAJECTORIES {
                    format!("统计误差为目标的 {:.1} 倍：轨迹数已达上限 {MOST_TRAJECTORIES}", a.statistics / POPULATION_TARGET)
                } else {
                    format!("统计误差 {}（{trajectories} 条轨迹，计算未完成）", percent(a.statistics))
                }
            } else if a.model >= a.step {
                format!("模型近似误差 {}", percent(a.model))
            } else {
                format!("时间步长误差 {}", percent(a.step))
            };
            s.reason = Some(format!("{}估计误差 {}（目标 {}）：{cause}", when(times[k]), percent(a.total()), percent(POPULATION_TARGET)));
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pilot is the smallest ensemble of whole groups that sees a share as large as the statistics' share of the
    /// population target, z²(1/(√(1 − s²) ε) − 1) = 85.4 trajectories at z = 2, s² = 1/5, ε = 0.05: whole groups whose
    /// unseen share fits that room, where a group fewer's does not.
    #[test]
    fn the_pilot_sees_the_target() {
        let (n, room) = (pilot_size(), (1. - STEP_SHARE * STEP_SHARE).sqrt() * POPULATION_TARGET);
        assert!(n % GROUP == 0 && unseen(n as f64) <= room && unseen((n - GROUP) as f64) > room, "{n}");
    }

    /// The first rung is the largest even part of the output interval that is admissible for the run and whose double is
    /// for the companion (`first_rung`). Rb-87 at 30 µK: the reach's largest speed 0.2185 µm/µs, its rms speed 0.0928.
    /// - No light, a 74.65 kHz trap, 10 µs between outputs: π/ω = 6.70 µs bounds the companion, 3.35 µs the run: the
    ///   interval over 4, 2.5 µs. A 1 kHz trap bounds nothing: half the interval.
    /// - A pattern of 2k = 16.1 µm⁻¹: the run below 2π/(K v_max) = 1.79 µs, the companion below 4.20 µs, so the run
    ///   below 1.79 µs: the interval over 6, 1.667 µs. At √10 k the run below 1.13 µs: the interval over 10, 1 µs.
    /// - In every case the rung is admissible with its double, and the next larger even part is not.
    #[test]
    fn the_first_rung_is_admissible_and_its_double_is_for_the_typical_atom() {
        let khz = |f: f64| std::f64::consts::TAU * f * 1e-3;
        let hm = hbar_over_mass(1.4432e-25);
        let fastest = |t: f64| (2. * (MOST_TRAJECTORIES as f64 / (CONFIDENCE * CONFIDENCE) + 1.).ln() * KT_PER_UK * t * hm).sqrt();
        let (fast, typical) = (fastest(30.), typical_speed(hm, 30.));
        assert!((fast - 0.2185).abs() < 5e-4 && (typical - 0.0928).abs() < 5e-4, "{fast} {typical}");
        assert_eq!(first_rung(10., khz(74.65), &[], fast, typical), 2.5);
        assert_eq!(first_rung(10., khz(1.), &[[0., 0.]], fast, typical), 5.);
        assert!((TAU / (16.1 * fast) - 1.79).abs() < 5e-3 && (TAU / (16.1 * typical) - 4.20).abs() < 1e-2);
        assert_eq!(first_rung(10., khz(74.65), &[[0., 16.1]], fast, typical), 10. / 6.);
        let k = 16.106 / 2. * 10f64.sqrt();
        assert!((TAU / (k * fast) - 1.13).abs() < 5e-3 && (TAU / (k * typical) - 2.66).abs() < 1e-2);
        assert_eq!(first_rung(10., khz(100.), &[[0., k]], fast, typical), 1.);
        let both = |tau: f64, w: f64, sampled: &[[f64; 2]], fast: f64, typical: f64| admissible(tau, w, sampled, fast) && admissible(2. * tau, w, sampled, typical);
        for (interval, w, sampled, fast, typical) in [
            (10., khz(910.7), vec![[0., 16.1]], fast, typical),
            (20., khz(79.), vec![[0., 16.1]], fastest(141.), typical_speed(hm, 141.)),
            (16., khz(20.3), vec![[0., 32.2]], fast, typical),
            (10., khz(74.65), vec![[0., 16.1]], 0.0089, 0.0038),
            (10., khz(74.65), vec![[0., 0.], [2., 16.1], [0.5, 8.]], fast, typical),
            (100., khz(100.), vec![[0., k]], fast, typical),
        ] {
            let tau = first_rung(interval, w, &sampled, fast, typical);
            let pairs = (interval / (2. * tau)).round();
            assert!((interval / (2. * pairs) - tau).abs() <= 1e-12 * tau && both(tau, w, &sampled, fast, typical), "{interval} {tau}");
            assert!(pairs == 1. || !both(interval / (2. * (pairs - 1.)), w, &sampled, fast, typical), "{interval} {tau}: a larger rung is admissible");
        }
        assert!(admissible(1.25, khz(74.65), &[[0., 16.1]], fast) && !admissible(2.5, khz(74.65), &[[0., 16.1]], fast) && admissible(2.5, khz(74.65), &[[0., 16.1]], typical));
    }

    /// The screen's outcomes on the run's averages between two levels (unlimited groups: one look's interval is z of
    /// the error). The step stands where every average fits its share, where a small one lies under a wide interval,
    /// and where one lies above its share but not clearly so among the looks; it is halved where an average is clear of
    /// its share: T's on an axis held to its target somewhere (an axis held nowhere is not judged, whatever the others
    /// are), the loss's and a row's as shown against the population's share. The family is the run's few averages: a
    /// difference of 24 % on T with a standard error of 5 % on eleven groups -- one standing wave's Doppler cooling at
    /// 2.5 µs against 5 µs -- is clear among them, where no single output's 6 SE was among seven thousand looks.
    #[test]
    fn the_screen_halves_the_step_on_a_clear_average_alone() {
        let read = |value: f64, error: f64| Read { value, spread: Spread { error, groups: f64::INFINITY } };
        let quiet = read(0., 0.);
        let at = |t: Read| Between { loss: quiet, shown: [quiet; STATES], temperature: [Some(t), Some(quiet), None] };
        let halved = |read: &Between, held: [bool; 3]| matches!(screen(read, held), Judgement::Beyond(_));
        let [sp, st] = step_bound();
        assert!(!halved(&at(read(0.5 * st, 0.1 * st)), [true; 3]));
        assert!(!halved(&at(read(0.5 * st, st)), [true; 3]));
        assert!(!halved(&at(read(2. * st, st)), [true; 3]));
        assert!(halved(&at(read(4. * st, 0.01 * st)), [true; 3]));
        assert!(!halved(&at(read(9. * st, 0.)), [false, true, true]) && halved(&at(read(9. * st, 0.)), [true, false, false]));
        let lost = Between { loss: read(2. * sp, 0.01 * sp), ..at(quiet) };
        let mut row = at(quiet);
        row.shown[3] = read(-2. * sp, 0.01 * sp);
        assert!(halved(&lost, [false; 3]) && halved(&row, [false; 3]));
        let pilot = |value: f64, error: f64| at(Read { value, spread: Spread { error, groups: 11. } });
        assert_eq!(judged(&pilot(0.24, 0.05), [true; 3]).len(), 1 + STATES + 2);
        assert!(halved(&pilot(0.65, 0.11), [true; 3]) && halved(&pilot(0.32, 0.05), [true; 3]) && !halved(&pilot(0.12, 0.04), [true; 3]));
    }

    /// The step's check on the pilot's averages (`checked`): a difference of 12 % on T_x with a standard error of 4 % on
    /// eleven groups leaves nothing clear among the looks (Student's t at ten degrees of freedom and Šidák's tail over
    /// the family) and nothing enters the accounts; of 60 % the clear part does, the number the screen halves the step
    /// for where it exceeds the step's share. The resolution is one look's t·SE: T's on the axes judged alone, what is
    /// shown from the rows and the loss. An axis no output reads has no bound, and no resolution where it is judged.
    /// The outputs an axis is averaged over: those it is held at; all of them for an axis held nowhere, which is not
    /// judged.
    #[test]
    fn the_step_s_check_books_what_is_clear_and_tells_its_resolution() {
        let quiet = Read { value: 0., spread: Spread { error: 0., groups: f64::INFINITY } };
        let pilot = |value: f64, error: f64| Read { value, spread: Spread { error, groups: 11. } };
        let at = |x: Read| Between { loss: quiet, shown: [quiet; STATES], temperature: [Some(x), Some(pilot(0.3, 0.2)), Some(quiet)] };
        let held = [true, false, true];
        let (step, resolution) = checked(&at(pilot(0.12, 0.04)), held);
        assert!(step.shown == 0. && step.temperature[0] == 0. && step.temperature[2] == 0., "{step:?}");
        let one = Spread { error: 0.04, groups: 11. }.interval();
        assert!(resolution == [0., one] && one > CONFIDENCE * 0.04, "{resolution:?}");
        let read = at(pilot(0.6, 0.04));
        let (step, _) = checked(&read, held);
        let family = Spread { error: 0.04, groups: 11. }.interval_over(judged(&read, held).len() as f64);
        assert!((step.temperature[0] - (0.6 - family)).abs() < 1e-15 && step.temperature[0] > step_bound()[1], "{step:?}");
        assert!(matches!(screen(&read, held), Judgement::Beyond(_)));
        let mut lossy = at(quiet);
        (lossy.loss, lossy.shown[2], lossy.temperature[1]) = (Read { value: -0.2, spread: Spread { error: 0.01, groups: f64::INFINITY } }, pilot(0.01, 0.02), None);
        let (step, resolution) = checked(&lossy, [false, true, false]);
        assert!(step.shown > 0.15 && step.shown < 0.2 && step.temperature[1] == f64::INFINITY, "{step:?}");
        assert!((resolution[0] - Spread { error: 0.02, groups: 11. }.interval()).abs() < 1e-15 && resolution[1] == f64::INFINITY, "{resolution:?}");
        let (over, judged) = averaged_over(&[[false; 3], [true, false, false], [false; 3]]);
        assert_eq!((over, judged), (vec![[false, true, true], [true; 3], [false, true, true]], [true, false, false]));
    }

    /// A look on T at 800 trajectories: the model's error, the step's clear part, the statistics' interval (each
    /// relative).
    fn look(model: f64, step: f64, interval: f64) -> Look {
        Look { output: 1, target: TEMPERATURE_TARGET, account: Account { model, step, statistics: interval }, interval, trajectories: 800., shown: false }
    }

    /// The trajectories a look needs (`Look::need`): its interval within the room the model's error and the step's
    /// clear part leave, N₀ (S/R)²; it is met exactly where it needs no more than it has; no number fits where the two
    /// fill the target; on what is shown the need is no less than the draws that leave the room unseen. And a batch
    /// (`batch`): the need in whole groups, more than the product has, at most twice it, at most the cap.
    #[test]
    fn a_look_needs_the_trajectories_that_fit_its_room() {
        for l in [look(0., 0., 0.12), look(0.03, 0.04, 0.08), look(0.03, 0.04, 0.09), look(0.02, 0., 0.3)] {
            let need = 800. * l.interval.powi(2) / (l.target.powi(2) - l.account.model.powi(2) - l.account.step.powi(2));
            assert!(l.fits() && (l.need() / need - 1.).abs() < 1e-12, "{l:?}: {} against {need}", l.need());
            assert_eq!(l.met(), l.need() <= l.trajectories, "{l:?}");
        }
        assert!(!look(0.06, 0.09, 0.01).fits() && !look(0.11, 0., 0.01).fits() && look(0.06, 0.07, 0.01).fits());
        let shown = Look { target: POPULATION_TARGET, shown: true, ..look(0.03, 0.02, 0.) };
        assert_eq!(shown.need(), draws_for(room(POPULATION_TARGET, 0.03, 0.02)));
        assert_eq!((batch(1147.3, 800), batch(300., 800), batch(5000., 800), batch(f64::INFINITY, 800), batch(1e9, MOST_TRAJECTORIES - 8)), (1152, 808, 1600, 1600, MOST_TRAJECTORIES));
    }

    /// An output's estimate with T = `temperature` (µK) and its standard errors per lab axis, read on unlimited groups (the
    /// interval is z of the error), every atom held, nothing else to chase.
    fn estimate(temperature: V3, errors: V3, trajectories: f64) -> Estimate {
        let exact = Spread { error: 0., groups: f64::INFINITY };
        Estimate {
            trajectories,
            held: trajectories,
            loss: 0.,
            loss_error: exact,
            shown: [0.; STATES],
            shown_errors: [exact; STATES],
            populations: [0.; STATES],
            motion: Some(crate::estimate::Motion {
                temperature,
                temperature_errors: errors.map(|error| Spread { error, groups: f64::INFINITY }),
                temperature_total: temperature.iter().sum::<f64>() / 3.,
                temperature_total_error: exact,
                occupation: [0.; 3],
                mean_velocity: [0.; 3],
            }),
            failed: 0.,
        }
    }

    /// T is held per lab axis (module documentation): in a trap of 100, 100 and 19 kHz the regime begins at
    /// CLASSICAL ħω/k_B = 19.2, 19.2 and 3.65 µK. A pilot that reads 25, 6 and 10 µK (± 1 %) holds x and z at their own
    /// T and leaves y; an axis whose T reads below its boundary within its interval is held (its upper bound is inside);
    /// one whose error is unread is not; nothing is held at t = 0.
    #[test]
    fn an_axis_is_held_where_its_own_temperature_is_in_its_regime() {
        let frequencies = [100., 100., 19.].map(|khz: f64| TAU * khz * 1e-3);
        let boundary = frequencies.map(|w| CLASSICAL * w / KT_PER_UK);
        assert!((boundary[0] - 19.2).abs() < 0.05 && (boundary[2] - 3.65).abs() < 0.05, "{boundary:?}");
        let cold = estimate([25., 6., 10.], [0.25, 0.06, 0.1], 800.);
        let held = held_to(&[cold.clone(), cold.clone()], &frequencies);
        assert_eq!(held, [[false; 3], [true, false, true]]);
        let edge = estimate([19., 19., 3.6], [0.2, 0.05, 0.1], 800.);
        assert_eq!(held_to(&[edge.clone(), edge], &frequencies)[1], [true, false, true]);
        let mut unread = cold.clone();
        unread.motion.as_mut().unwrap().temperature_errors[2] = Spread::UNREAD;
        assert_eq!(held_to(&[cold, unread], &frequencies)[1], [true, false, false]);
    }

    /// The looks and the accounts follow each axis's hold. With x and z held and y not (the pilot's 25, 6, 10 µK), a model
    /// error of 3 % and a step's clear part of 2 %, 40 % and 1 % on the axes: there is one look on what is shown and one
    /// on each held axis, each with its own step part and statistics; an interval of 30 % on y neither keeps the run
    /// from stopping nor asks for a trajectory, where the same interval on x does both, by (interval/room)²; y's step
    /// part, beyond any target, crowds no output where on a held axis it would. A model error that fills the target
    /// crowds its output. Every axis has its account, its own step and statistics in it, and the result's per-axis
    /// totals are those.
    #[test]
    fn the_looks_and_the_accounts_follow_each_axis() {
        let fixed = |held: [bool; 3]| Fixed { model: vec![[0.; 2], [0., 0.03]], temperature: vec![[false; 3], held], start: 0. };
        let none = Bound { shown: 0., temperature: [0.; 3] };
        let step = [none, Bound { shown: 0., temperature: [0.02, 0.4, 0.01] }];
        let held = [true, false, true];
        let n = 800.;
        let quiet = estimate([25., 6., 10.], [0.25, 0.9, 0.1], n);
        let run = [quiet.clone(), quiet];
        let all = looks(&run, &fixed(held), &step);
        assert_eq!(all.iter().map(|l| (l.output, l.shown, l.account.step)).collect::<Vec<_>>(), [(1, true, 0.), (1, false, 0.02), (1, false, 0.01)]);
        assert!(all.iter().all(|l| l.met() && l.need() < n) && crowded(&all).is_empty());
        let noisy = estimate([25., 6., 10.], [3.75, 0.06, 0.1], n);
        let run = [noisy.clone(), noisy];
        let x = looks(&run, &fixed(held), &step)[1];
        let need = n * (CONFIDENCE * 3.75 / 25. / room(TEMPERATURE_TARGET, 0.03, 0.02)).powi(2);
        assert!(!x.met() && (x.need() - need).abs() < 1e-9 * need, "{} against {need}", x.need());
        let every = looks(&run, &fixed([true; 3]), &step);
        assert!(!every[2].fits() && crowded(&every) == [1]);
        let mut full = fixed(held);
        full.model[1][1] = 0.11;
        assert!(crowded(&looks(&run, &full, &step)) == [1]);
        let (_, accounts) = accounts(&run, &fixed(held), &step);
        let expected = [(0.02, CONFIDENCE * 3.75 / 25.), (0.4, CONFIDENCE * 0.06 / 6.), (0.01, CONFIDENCE * 0.1 / 10.)];
        for (a, (step, statistics)) in expected.into_iter().enumerate() {
            let account = accounts[1][a].unwrap();
            assert!(account.model == 0.03 && account.step == step && (account.statistics - statistics).abs() < 1e-14, "{account:?}");
        }
    }

    /// The plan: the windows half a period (at most half an interval) cut at t = 0, the listed outputs evenly spread with
    /// the first and the last, the run's end the last window's.
    #[test]
    fn the_plan_holds_the_windows_and_the_listed_outputs() {
        let times: Vec<f64> = (0..101).map(|k| k as f64).collect();
        let w = std::f64::consts::TAU / 0.5;
        let p = Plan::new(&times, [w, 0.1, w]);
        assert_eq!(p.halves[0], [0.; 3]);
        assert_eq!(p.halves[7], [0.25, 0.5, 0.25]);
        assert_eq!((p.listed.len(), p.listed[0], p.listed[20], p.listed[1]), (21, 0, 100, 5));
        assert_eq!(p.end, 100.5);
        let p = Plan::new(&[0., 2.], [w; 3]);
        assert_eq!((p.listed.clone(), p.end), (vec![0, 1], 2.25));
    }

    /// The histograms hold every atom, sum to one, put 0 at a bin's centre, and carry a listed output with no atom.
    #[test]
    fn the_histograms_hold_the_atoms() {
        let listed = vec![vec![([0.; 3], [1., 0., 0.]), ([0.; 3], [-3., 0., 0.]), ([0.; 3], [0.2, 0., 0.])], vec![]];
        let d = histograms(&listed, &[0, 4], &|x| x.1[0]);
        assert_eq!(d.grid.len(), 3);
        assert_eq!(d.grid[1], 0.);
        assert_eq!(d.probabilities[1], d.probabilities[0]);
        assert!((d.probabilities[0].iter().sum::<f64>() - 1.).abs() < 1e-15);
    }

    /// The groups come back in their order whatever the threads do: 37 groups' records (five of a pilot's numbers, then a
    /// product's) -- each trajectory's own, its rows marking its number -- folded by `run_groups` equal the same records
    /// folded one after another.
    #[test]
    fn the_groups_come_back_in_order() {
        let shape = Shape { outputs: 2, listed: 1, keep: false };
        let record = |i: u64| {
            let mut rows = [0.; STATES];
            rows[(i % STATES as u64) as usize] = 1.;
            let held = Held { rows, kappa: [i as f64; 3], kappa2: [(i * i) as f64; 3] };
            let outputs = vec![Fate::Held(held.clone()), if i % 3 > 0 { Fate::Held(held) } else { Fate::Lost }];
            Record { outputs, listed: vec![Some(([i as f64; 3], [0.; 3]))], samples: vec![], counts: Counts { steps: i, ..Counts::default() } }
        };
        let mut silent = |_: &Solution, _: Option<f64>| {};
        let mut messages = Messages { send: &mut silent, clock: Instant::now(), last: None, least: 0 };
        let numbers: Vec<u64> = (2048..2053).chain(5..37).collect();
        let parts = run_groups(&numbers, 0..GROUP, shape, &record, &mut messages, &mut |_, _| {});
        assert_eq!(parts.len(), 37);
        for (part, number) in parts.iter().zip(&numbers) {
            let one = shape.fold((0..GROUP).map(|m| record(number * GROUP + m)));
            assert_eq!(part.sums, one.sums);
            assert_eq!(part.listed, one.listed);
            assert_eq!(part.counts.steps, one.counts.steps);
        }
    }

    /// The initial ensembles of the conservative reference traps (refs/scenes R1, the app's own requests: a lone tweezer, a
    /// standing wave, crossed beams of one frequency, a σ⁺ tweezer) in the model's prepared wells under the model's proven
    /// envelopes: 2¹² draws each read the exact moments (ħ/m)⟨κ_a²⟩ = Σ_b p_b η_b k_BT (S_b²)_aa per lab axis within their
    /// intervals over the groups at the confidence z -- twelve looks, three axes of four traps -- and the quadrature's own
    /// error as the solve books it for T(0), its last two levels' relative difference (`Fixed::start`), here on 2¹⁶
    /// evaluations, fewer than a pilot's steps in these traps (88 trajectories of 800 steps or more). η lies in (0, 1]:
    /// the lowered isothermal's kinetic temperature, below the Boltzmann one.
    #[test]
    fn the_reference_traps_draw_their_exact_moments() {
        let names = ["R1_TWEEZER", "R1_RETRO", "R1_CROSSED", "R1_SIGMA_P"];
        for name in names {
            let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("scenes").join(format!("{name}.json"));
            let line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
            let request = Request::from_native(&line).unwrap();
            let light = Light::new(&request);
            let model = Model::new(&request, &light, &[]).unwrap();
            let initial = components(&request);
            let wells: Vec<Prepared> = initial.iter().map(|&(s, _)| Prepared::new(&model, &light, alone(s)).unwrap()).collect();
            let hm = hbar_over_mass(request.mass_kg);
            let ensemble = Ensemble::new(&initial.iter().zip(&wells).map(|(&(s, p), w)| (s, p, w as &dyn Well)).collect::<Vec<_>>(), request.temperature_uk, hm).unwrap();
            let kinetic = ensemble.kinetic(1 << 16);
            assert!(kinetic.iter().all(|k| k.eta > 0. && k.eta <= 1.), "{name}: η {kinetic:?}");
            let exact = ensemble.momentum(&kinetic.iter().map(|k| k.eta).collect::<Vec<_>>());
            let quadrature = kinetic.iter().map(|k| k.difference / k.eta).fold(0., f64::max);
            let draws: Vec<V3> = (0..1 << 12).map(|i| ensemble.draw(SEED, i).kappa.map(|k| hm * k * k)).collect();
            for a in 0..3 {
                let groups: Vec<f64> = draws.chunks(GROUP as usize).map(|members| members.iter().map(|d| d[a]).sum::<f64>() / GROUP as f64).collect();
                let g = groups.len() as f64;
                let mean = groups.iter().sum::<f64>() / g;
                let error = (groups.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (g - 1.) / g).sqrt();
                let within = Spread { error, groups: g }.interval_over((3 * names.len()) as f64) + quadrature * exact[a];
                assert!((mean - exact[a]).abs() <= within, "{name} axis {a}: {mean} against {} ± {within} (quadrature {quadrature:e})", exact[a]);
            }
        }
    }

}
