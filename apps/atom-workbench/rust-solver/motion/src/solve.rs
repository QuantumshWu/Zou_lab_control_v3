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
//! 4. The product: batches of whole groups at both levels, every trajectory run to the last output. After each batch
//!    the step's error is measured anew between the levels and booked, and the run stops where every held quantity's
//!    total lies within its target (# Stopping), goes on at its step, goes on at half of it -- the level at τ becomes
//!    the companion --, or ends at `MOST_TRAJECTORIES`.
//! 5. The result (contract §2.2, `assemble`).
//!
//! # The step (SPEC §2.2)
//! One step τ for the run (adapt.rs). The ladder's rungs are the output interval over powers of two; its top one the
//! largest below the reference flow's resonance, ω τ < π at the prepared trap's largest frequency (`top_rung`).
//!
//! The step's error is measured on what is shown, not read step by step. The run keeps two levels: its trajectories at
//! τ, and the same trajectory numbers -- the same initial atoms, the same jumps' draws by their number, the same
//! Brownian path (adapt.rs) -- at 2τ (`Levels`). What the levels share cancels in their difference for as long as the
//! pairs stay together: some 50 µs in six beams' cooling, where the jumps' times, which are the step's, and the
//! light's structure on the wavelength's scale part them whatever the noise (with no diffusion at all as fast), and
//! half a millisecond in one standing wave's Doppler cooling (SPEC §2.2). After that the two levels are independent
//! ensembles, and the difference's error is theirs in quadrature.
//! An output's step error is e(τ) = [Q(2τ) − Q(τ)]/(2^p − 1), its error the delete-one-group jackknife of the paired
//! difference (estimate.rs `between`): one instrument for the loss, the rows as shown and T per lab axis, each carried
//! to the output by the dynamics itself. Booked per output is the bound |e| + t·SE(e) (`Bound`). The part of it that
//! more trajectories do not take away is what is clear of the noise, |e| − t·SE where positive (`Booked::clear`); the
//! rest falls as N^{−1/2}, like the statistics.
//!
//! - The first rung is the rung below the top one: its companion at 2τ is then itself below the resonance.
//! - The screen (`screen`, `settle`), on the scout and on the pilot: the step is halved while some difference lies
//!   clearly beyond the step's share among all the looks (decide.rs `judge`); the level at τ becomes the companion and
//!   only τ/2 is run anew. On the pilot's 88 trajectories SE(e) is 3 to 5 % on T and the family's interval is nine of
//!   them: the screen stops gross errors -- a step that heats the atoms severalfold before the model is judged at their
//!   states -- and no more. A step it leaves standing is not thereby within its share.
//! - In the product (`next`), after every batch past the pilot, the step is halved where some look is refuted -- the
//!   clear part of its step error alone fills what the model's error leaves of its target, so that no number of
//!   trajectories fits --, or where the rung below asks less work to the targets: its trajectories at twice the steps,
//!   with every clear part over 2^p, against the trajectories still needed here at both levels. The trajectories
//!   already run at τ are the new step's companion: nothing is run twice, and only the level at 4τ is left behind.
//!   The clear part is one look's; a halving takes the largest of many. On the pilot that reads much by chance, and
//!   the pilot is only screened; from the first batch on a halving asks a clear part of the target's own size beside
//!   t·SE, some four standard errors and more as N grows (`tests::the_step_is_not_halved_for_its_noise`: twenty
//!   independent looks at 176 trajectories halve the step for noise in 5.3 % of 400 batches, one look's tail being
//!   4.6 %; the looks of a run share their trajectories and are fewer than that in effect).
//! - Down to the step at which a trajectory would take `MOST_STEPS` steps: there the bound is booked as measured, and
//!   flagged.
//!
//! What the instrument rests on is the scheme's order between the two levels, e ∝ τ^p. Where the companion does not
//! resolve the dynamics the difference is larger than (2^p − 1) e, and the bound is too large, never too small in the
//! cases measured (six beams' cooling in a tweezer, τ = 1.25 µs with its companion at 2.5 µs: T reads +11 to +28 %
//! where the truth is −3 %; the step is halved for it, and at 0.625 µs with its companion at 1.25 µs the read is the
//! truth's size, SPEC §2.2). A third level would test the order; a run's trajectories do not resolve that ratio.
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
//! E² = A² + b² + S² (ledger.rs; `Look`): A the model's error, from the units measured at the pilot's states and fixed
//! there; b the step's bound as the two levels measure it on all the groups so far; S the statistics' interval. The run
//! stops where every one of them lies within its target: the test the ledger flags an output by (`Look::met`), so a run
//! that stops by it is flagged nowhere it is held.
//!
//! The trajectories a look still needs (`Look::need`): with x = √(N₀/N) from the N₀ it was read on, the statistics are
//! S x (on what is shown no less than the unseen share at N, estimate.rs `unseen`) and the step's bound c + (b − c) x, c
//! its clear part; the least N with A² + (c + (b − c)x)² + S²x² ≤ ε², none where A² + c² ≥ ε². At the rung below c is
//! c/2^p. A batch brings both levels to the number the looks predict, at most to twice the number the prediction rests
//! on (`batch`).
//!
//! The stop reads the run's own data: the bound and the statistics shrink together, and a run stops at the first batch
//! where both happen to read low. Measured on synthetic levels (`tests::the_stop_on_its_own_data_is_early_by_its_noise_at_the_most`:
//! exponential energies, decorrelated levels, a true step error of 2 %, T held to 10 %, 400 runs): the total a stopped
//! run would report with the true variances is 0.95 of the target on average -- the bound's own z·SE and the batches'
//! overshoot outweigh the early stops --, the truth lies within the reported total in 97.8 % of the runs (one look at z
//! allows 95.4 %), T at the stop has no bias its 400 runs resolve (0.08 ± 0.09 of its standard error), and the step is
//! halved for its noise in one run of them. An early stop is bounded by the relative noise of a standard error on the
//! stop's groups, 8.6 % of the target there; what is held on the pilot instead is what the pilot alone can fix: the
//! model's errors and which axes are held.
//!
//! # T is held per lab axis
//! The semiclassical model is validated where k_BT ≥ `CLASSICAL` ħω and not below (SPEC §0, §8: its one known limit). The
//! test is taken axis by axis: T_a is held to its target -- its step's difference screened against the step's share, the
//! bound booked, its look met or chased (`looks`), the total flagged where it exceeds the target (ledger.rs) -- at every
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
//! least (the jackknife's least for a spread) and on no fewer trajectories than the attempt's last one -- a product's
//! first groups, or the first groups at a finer step, are no better than the result before them --; else the last result
//! goes out again, so that the app hears from the solve whatever stage it is in.
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

/// The ladder's top rung (module documentation, # The step): the output interval over the least power of two that
/// brings the step below the reference flow's resonance at the frequency ω (rad/µs), ω τ < π -- there the two half
/// flows of a step turn the motion by π and any kick between them grows.
fn top_rung(interval: f64, frequency: f64) -> f64 {
    let mut tau = interval;
    while frequency * tau >= PI {
        tau *= 0.5;
    }
    tau
}

/// The screen of a step on the reads between its two levels (module documentation, # The step; decide.rs `judge`): the
/// family of every output's loss and rows as shown, and of T on each lab axis where that axis is held to its target
/// (`held`: per output and axis), each against the step's share. The step is halved where the family is beyond.
fn screen(reads: &[Between], held: &[[bool; 3]]) -> Judgement {
    let [sp, st] = step_bound();
    let mut looks: Vec<(Read, f64)> = vec![];
    for (k, r) in reads.iter().enumerate().skip(1) {
        looks.push((r.loss, sp));
        looks.extend(r.shown.iter().map(|x| (*x, sp)));
        if let Some(t) = &r.temperature {
            looks.extend((0..3).filter(|&a| held[k][a]).map(|a| (t[a], st)));
        }
    }
    judge(&looks, ORDER as f64)
}

/// The step's error on one quantity as the run's two levels bound it (module documentation, # The step): the bound
/// |e| + t·SE, and what is clear of it, |e| − t·SE where positive -- the part more trajectories do not take away.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Booked {
    bound: f64,
    clear: f64,
}
impl Booked {
    /// A quantity the levels do not read: no bound, and nothing clear.
    const UNREAD: Booked = Booked { bound: f64::INFINITY, clear: 0. };
    /// A read's bound and its clear part, one look each.
    fn of(read: &Read) -> Booked {
        Booked { bound: read.bound(1.), clear: read.clear(1.).max(0.) }
    }
    /// An error taken to be `share` whatever the trajectories: the step's share, before the levels are read and in the
    /// model's decisions.
    fn held(share: f64) -> Booked {
        Booked { bound: share, clear: share }
    }
    /// The larger bound and the larger clear part of two.
    fn larger(self, other: Booked) -> Booked {
        Booked { bound: self.bound.max(other.bound), clear: self.clear.max(other.clear) }
    }
}

/// The step's error at one output (`booked`): on what is shown -- the largest of the rows' and the loss's, a
/// probability's error at most 1 -- and on T per lab axis (relative).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Bound {
    shown: Booked,
    temperature: [Booked; 3],
}
impl Bound {
    /// The step's share of each target (`step_bound`).
    fn share() -> Bound {
        let [shown, temperature] = step_bound();
        Bound { shown: Booked::held(shown), temperature: [Booked::held(temperature); 3] }
    }
}

/// The step's error per output as the two levels' reads bound it: T's unbounded on every axis where a level holds
/// fewer than two atoms.
fn booked(reads: &[Between]) -> Vec<Bound> {
    reads
        .iter()
        .map(|r| {
            let shown = r.shown.iter().map(Booked::of).fold(Booked::of(&r.loss), Booked::larger);
            Bound { shown: Booked { bound: shown.bound.min(1.), clear: shown.clear.min(1.) }, temperature: r.temperature.as_ref().map_or([Booked::UNREAD; 3], |t| t.each_ref().map(Booked::of)) }
        })
        .collect()
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
/// has one there --, with each state's output and group.
struct Sampled {
    samples: Vec<State>,
    at: Vec<usize>,
    from: Vec<usize>,
    errors: Vec<Vec<UnitError>>,
}
fn sampled(model: &Model, units: &[Unit], tables: &crate::model::tabulate::Tables, parts: &[&Part]) -> Sampled {
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
    let each: Vec<Vec<UnitError>> =
        samples.par_iter().map(|s| model.errors_with(units, std::slice::from_ref(&s.sample), Some(tables)).into_iter().map(|e| e[0]).collect()).collect();
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
    unsampled: f64,
    again: f64,
    barred: &[Unit],
) -> Option<(usize, Option<f64>)> {
    let means = measured.means(fixed.model.len(), None);
    let held: Vec<f64> = estimates.iter().map(|e| 1. - e.loss).collect();
    // A unit weighs where it was read: an output with no state read carries every unit's bound alike, and tells none
    // of them apart.
    let read = |k: usize| measured.output.contains(&k);
    let weight = |u: usize, over: &[usize]| -> f64 {
        over.iter().filter(|&&k| held[k] > 0. && read(k)).map(|&k| held[k] * (means[u][k][0] / POPULATION_TARGET + means[u][k][1] / TEMPERATURE_TARGET)).sum()
    };
    let crowded = crowded(&looks(estimates, fixed, &vec![Bound::share(); fixed.model.len()]));
    if let Some(u) = (0..units.len()).max_by(|&a, &b| weight(a, &crowded).total_cmp(&weight(b, &crowded))).filter(|&u| weight(u, &crowded) > 0.) {
        return Some((u, None));
    }
    let pricing = Pricing::new(request, light, kept, model, reference, centre, units, measured, states, parts, fixed, scales, unsampled, again);
    let saving = pricing.saving_at_all();
    let mut ranked: Vec<Candidate> = if saving.is_empty() {
        vec![]
    } else {
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
    let top = top_rung(plan.times[1] - plan.times[0], trap.frequencies.iter().copied().fold(0., f64::max));
    let mut context = Context {
        momentum,
        fixed: Fixed { model: vec![[0.; 2]; outputs], temperature: vec![[false; 3]; outputs], start },
        step: vec![Bound::share(); outputs],
        measured: false,
        // The first rung (module documentation, # The step): the rung below the ladder's top one.
        tau: step.unwrap_or_else(|| halved(top, 1.)),
        plan,
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

    // The pilot's groups (module documentation, # The pipeline): the run's first trajectories, at the run's two levels.
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
    {
        let tau = context.tau;
        let scout: Vec<&Part> = levels.fine.iter().collect();
        let estimates = pooled(&scout, &context.scales).0;
        if let Some(t) = outgrown(&estimates, reach) {
            return Ok(Attempt::Grow(t, tau));
        }
        let units = model.units();
        let Sampled { samples, at, from, errors } = sampled(&model, &units, &tables, &scout);
        let measured = Measured { errors: &errors, output: &at, group: &from };
        let unsampled = unseen(MOST_TRAJECTORIES as f64);
        let fixed = Fixed { model: model_error(&measured.means(outputs, None), None, unsampled), temperature: held_to(&estimates, &report.lab), ..context.fixed.clone() };
        if let Some((u, share)) = keep(request, light, kept, &model, reference, trap.centre, &units, &measured, &samples, &scout, &estimates, &fixed, &context.scales, unsampled, scout.len() as f64, barred) {
            return Ok(Attempt::Keep(units[u].clone(), tau, share));
        }
    }
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
        let Sampled { samples, at, from, errors } = sampled(&model, &units, &tables, &pilot);
        let measured = Measured { errors: &errors, output: &at, group: &from };
        let means = measured.means(outputs, None);
        let (population, temperature): (Vec<Vec<f64>>, Vec<Vec<f64>>) =
            means.iter().map(|unit| (unit.iter().map(|m| m[0]).collect(), unit.iter().map(|m| m[1]).collect())).unzip();
        let unsampled = unseen(MOST_TRAJECTORIES as f64);
        context.fixed.model = model_error(&means, None, unsampled);
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
        context.ledger.item("折叠判定之外的未采样份额", unsampled, "计入模型误差");
        Run::tabulated(&model, &tables, reference)
    };

    // The product (module documentation, # Stopping): on from the pilot's trajectories, the run's first, at both levels.
    let (e, l, c) = product(&run, &ensemble, &mut context, &mut levels, &numbers, messages);
    // A point of the prepared potential below the bottom the ensemble was drawn about (ensemble.rs, # Its draw).
    if ensemble.deeper() > 0 {
        context.ledger.flag("准备势在所取阱底之外还有更深的地方：初始系综按该阱底的深度封顶".into());
    }
    Ok(Attempt::Done(assemble(&context, &e, &l, &c, levels.fine.len() as u64 * GROUP)))
}

/// The flag of a step whose error lies clearly beyond what it may take at the steps' cap (module documentation, # The
/// step): its bound is booked as measured.
fn capped() -> String {
    format!("时间步长的误差明显超出它可占的部分，而每条轨迹的步数已达上限 {MOST_STEPS}，无法再减半：按实测的界记账")
}

/// The run's two levels (module documentation, # The step): its groups at the run's step τ and the same groups, in the
/// same order, at 2τ -- the pilot's groups first, then the product's. The companion holds every group of the finer
/// level, and after a halving in the product it may hold more.
#[derive(Default)]
struct Levels {
    fine: Vec<Part>,
    coarse: Vec<Part>,
}
impl Levels {
    /// The finer level's groups and their companions.
    fn paired(&self) -> (Vec<&Part>, Vec<&Part>) {
        (self.fine.iter().collect(), self.coarse.iter().take(self.fine.len()).collect())
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
            let held = held_to(&pooled(&fine, &context.scales).0, frequencies);
            matches!(screen(&between(&fine, &coarse, &context.scales), &held), Judgement::Beyond(_))
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

/// The trajectories a batch brings a level to, for the `need` that estimates of `rests` trajectories predict: whole
/// groups, `least` at the least, at most twice the number the prediction rests on, and `MOST_TRAJECTORIES` at most.
///
/// At most twice: the need is predicted from variances read on those trajectories' groups (the pilot's eleven at first,
/// whose largest read over the outputs runs high), and a batch takes the product no further than to twice what its
/// prediction rests on. What that bounds: the batches accumulate -- every trajectory of the product is in the result --,
/// so the product ends on the first total that meets the targets. Each total is at most g times one that did not meet
/// them, so the last total is at most g times the least that meets them, whatever was predicted: at g = 2 no more than
/// half of the work is spent past the need, and a need N times the pilot's size is reached in log₂ N batches.
fn batch(need: f64, least: u64, rests: u64) -> u64 {
    let whole = ((need / GROUP as f64).ceil() as u64).saturating_mul(GROUP);
    whole.clamp(least, least.max(2 * rests)).min(MOST_TRAJECTORIES)
}

/// What the product does after a batch (`next`): it is done -- every look met, or the trajectories at their cap --; it
/// is stuck -- the step's error clearly fills a target and the steps' cap bars the rung below --; it runs on at its step,
/// or at half of it, to so many trajectories.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Next {
    Done,
    Stuck,
    More(u64),
    Halve(u64),
}

/// The product's next move (module documentation, # Stopping, # The step) from its looks, the trajectories its two
/// levels hold (the finer, the companion), the counted work a trajectory takes at each, and whether the rung below is
/// within the steps' cap (`finer`).
///
/// The step is halved where some look is refuted -- no number of trajectories fits beside what is clear of its step
/// error --, or where the rung below asks less work to the targets. The work to the targets here: the trajectories the
/// looks still need (`Look::need`), at both levels. At the rung below: its need, with every clear part over 2^p, run
/// anew at a work a trajectory that is to the finer level's as that is to the companion's -- the levels' own ratio,
/// two steps for one less the steps that end early at a jump at either length; no less than the finer level's own and
/// no more than twice it, the bounds of a step halved, where a companion that does not resolve the dynamics says
/// otherwise --, and the present finer level, which becomes its companion, brought up to it. Where the rung below is
/// predicted refuted in its turn, it is run no further than the least that reads it, the pilot's size.
///
/// Not on a level of the pilot's size: the pilot only screens (`settle`). A clear part there is |e| − t·SE of the
/// largest of hundreds of reads whose SE is half the target, and reads as much by chance; from the first batch on the
/// margin a halving asks beside t·SE -- a clear part of the target's own size -- is several SE more, and grows as √N.
fn next(looks: &[Look], have: [u64; 2], work: [f64; 2], finer: bool) -> Next {
    if looks.iter().all(Look::met) || have[0] >= MOST_TRAJECTORIES {
        return Next::Done;
    }
    let need = |lower: f64| looks.iter().map(|l| l.need(lower)).fold(0., f64::max);
    let (here, below) = (need(1.), need(0.5_f64.powi(ORDER)));
    let judged = have[0] > pilot_size();
    let refuted = judged && looks.iter().any(Look::refuted);
    let more = |need: f64, have: u64| (need - have as f64).max(0.);
    let stay = more(here, have[0]) * work[0] + more(here, have[1]) * work[1];
    let halve = below * work[0] * (work[0] / work[1]).clamp(1., 2.) + more(below, have[0]) * work[0];
    if (refuted || halve < stay) && judged && finer {
        Next::Halve(if below.is_finite() { batch(below, pilot_size(), have[0]) } else { pilot_size() })
    } else if refuted {
        Next::Stuck
    } else {
        Next::More(batch(here, have[0] + GROUP, have[0]))
    }
}

/// The counted work a trajectory of some groups takes on average (decide.rs `Work`): the integrator's and its middles'
/// (`assembly`, the model's count of one).
fn work(parts: &[Part], assembly: f64) -> f64 {
    let (work, assemblies) = parts.iter().fold((0., 0), |a, p| (a.0 + p.counts.work, a.1 + p.counts.assemblies));
    (work + assemblies as f64 * assembly) / (parts.len() as u64 * GROUP) as f64
}

/// The steps some groups took.
fn steps(parts: &[&Part]) -> u64 {
    parts.iter().map(|p| p.counts.steps).sum()
}

/// The product (module documentation, # Stopping, # The step): batches of whole groups at both levels, on from the
/// pilot's groups (their numbers `pilot`; the product's own follow in the list's order). After each batch the step's
/// error is measured between the levels on all the groups so far and booked, and the product's next move taken from it
/// (`next`): where the step is halved, the level at τ becomes the companion and the groups run anew at τ/2. Each
/// batch's result is a checkpoint; returns the estimates, the listed atoms and the counts of the finer level at its end.
fn product(run: &Run, ensemble: &Ensemble, context: &mut Context, levels: &mut Levels, pilot: &[u64], messages: &mut Messages) -> (Vec<Estimate>, Vec<Vec<(V3, V3)>>, Counts) {
    let started = Instant::now();
    let assembly = run.assembly_work();
    // The group at the list's place g.
    let number = |g: u64| pilot.get(g as usize).copied().unwrap_or(g);
    // The wall time (ms) a batch to `to` groups still takes at the product's pace: its time over the steps it has
    // `taken`, times the steps the groups still to run take at each level's steps a group so far; none before a level's
    // first group is in.
    let left = |taken: u64, fine: &[&Part], coarse: &[&Part], to: u64| -> Option<f64> {
        let todo = |level: &[&Part]| to.saturating_sub(level.len() as u64) as f64 * steps(level) as f64 / level.len() as f64;
        Some(started.elapsed().as_secs_f64() * 1e3 / taken as f64 * (todo(fine) + todo(coarse))).filter(|ms| ms.is_finite())
    };
    let mut taken = 0;
    loop {
        let n = levels.fine.len() as u64 * GROUP;
        let (e, l, c, decision) = {
            let (fine, coarse) = levels.paired();
            let (e, l, c) = pooled(&fine, &context.scales);
            (context.step, context.measured) = (booked(&between(&fine, &coarse, &context.scales)), true);
            let have = [n, levels.coarse.len() as u64 * GROUP];
            let finer = context.plan.end / halved(context.tau, 1.) <= MOST_STEPS as f64;
            let decision = next(&looks(&e, &context.fixed, &context.step), have, [work(&levels.fine, assembly), work(&levels.coarse, assembly)], finer);
            (e, l, c, decision)
        };
        let want = match decision {
            Next::Done => return (e, l, c),
            Next::Stuck => {
                context.ledger.flag(capped());
                return (e, l, c);
            }
            Next::More(want) | Next::Halve(want) => want,
        };
        let to = want / GROUP;
        {
            let (fine, coarse) = levels.paired();
            messages.result(&assemble(context, &e, &l, &c, n), if matches!(decision, Next::More(_)) { left(taken, &fine, &coarse, to) } else { None });
        }
        if matches!(decision, Next::Halve(_)) {
            // The level at τ is the companion of the level at τ/2. Until the new levels are read, the results on the way
            // carry the bound last booked: the rung above's, larger by the order's factor where the order holds.
            context.tau = halved(context.tau, 1.);
            levels.coarse = std::mem::take(&mut levels.fine);
        }
        let numbers: Vec<u64> = (levels.fine.len() as u64..to).map(number).collect();
        let new = level(run, ensemble, &context.plan, context.tau, &numbers, 0..GROUP, false, messages, &mut |prefix, messages| {
            let all: Vec<&Part> = levels.fine.iter().chain(prefix.iter().copied()).collect();
            let (e, l, c) = pooled(&all, &context.scales);
            messages.result(&assemble(context, &e, &l, &c, all.len() as u64 * GROUP), left(taken + steps(prefix), &all, &levels.coarse.iter().collect::<Vec<_>>(), to));
        });
        taken += steps(&new.iter().collect::<Vec<_>>());
        levels.fine.extend(new);
        let numbers: Vec<u64> = (levels.coarse.len() as u64..to).map(number).collect();
        let new = level(run, ensemble, &context.plan, 2. * context.tau, &numbers, 0..GROUP, false, messages, &mut |prefix, messages| {
            let (fine, all): (Vec<&Part>, Vec<&Part>) = (levels.fine.iter().collect(), levels.coarse.iter().chain(prefix.iter().copied()).collect());
            let (e, l, c) = pooled(&fine, &context.scales);
            messages.result(&assemble(context, &e, &l, &c, fine.len() as u64 * GROUP), left(taken + steps(prefix), &fine, &all, to));
        });
        taken += steps(&new.iter().collect::<Vec<_>>());
        levels.coarse.extend(new);
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
}
impl Plan {
    /// The plan of the request's times with the trap's frequency per lab axis (rad/µs).
    fn new(times: &[f64], frequencies: V3) -> Plan {
        let (interval, last) = (times[1] - times[0], times.len() - 1);
        let halves: Vec<V3> = times.iter().map(|&t| frequencies.map(|w| half_window(w, interval, t))).collect();
        let count = times.len().min(DISTRIBUTION_TIMES);
        let mut listed: Vec<usize> = (0..count).map(|i| ((i * last) as f64 / (count - 1) as f64).round() as usize).collect();
        listed.dedup();
        Plan { end: times[last] + halves[last].iter().fold(0., |a: f64, &b| a.max(b)), times: times.to_vec(), halves, listed }
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
    // The trajectory's Brownian path in cells of an output interval, its grid the multiples of τ (the ladder's rungs
    // are the interval over powers of two): the same path at every rung.
    let mut path = adapt::Path::new(SEED, index, interval, (interval / tau).log2().round() as u32);
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
/// held there. Its account (ledger.rs: the model's error A, the step's bound b, the statistics S, as the ledger books
/// them) against its target ε; and what its prediction takes beside them: the clear part c of the step's bound, the
/// statistics' jackknife interval and the trajectories N₀ both were read on, and on what is shown the unseen share,
/// which the statistics are no less than.
#[derive(Clone, Copy, Debug)]
struct Look {
    output: usize,
    target: f64,
    account: Account,
    clear: f64,
    interval: f64,
    trajectories: f64,
    shown: bool,
}
impl Look {
    /// Whether its total lies within its target: the ledger's own test of an output (ledger.rs `Ledger::close`).
    fn met(&self) -> bool {
        self.account.total() <= self.target
    }
    /// Whether some number of trajectories fits: what they do not take away -- the model's error, and the step's clear
    /// part times `lower` (1 at this rung; 2^{−p} at the rung below) -- lies within the target.
    fn fits(&self, lower: f64) -> bool {
        room(self.target, self.account.model, self.clear * lower) > 0.
    }
    /// Whether the step refutes it: no number of trajectories fits, and the model's error alone would leave room.
    fn refuted(&self) -> bool {
        !self.fits(1.) && self.fits(0.)
    }
    /// Its predicted total at N trajectories, x = √(N₀/N): the statistics S x -- on what is shown no less than the unseen
    /// share at N --, the step's bound c + (b − c) x, its clear part times `lower`.
    fn total(&self, n: f64, lower: f64) -> f64 {
        let x = (self.trajectories / n).sqrt();
        let statistics = if self.shown { (self.interval * x).max(unseen(n)) } else { self.interval * x };
        Account { model: self.account.model, step: self.clear * lower + (self.account.step - self.clear) * x, statistics }.total()
    }
    /// The trajectories it needs: the least N whose predicted total lies within the target, by bisection to the format's
    /// resolution (the total never rises with N); none where no number fits or a part is unread.
    fn need(&self, lower: f64) -> f64 {
        if !(self.fits(lower) && self.account.step.is_finite() && self.interval.is_finite()) {
            return f64::INFINITY;
        }
        let beyond = |n: f64| self.total(n, lower) > self.target;
        let (mut low, mut high) = (0., self.trajectories);
        while beyond(high) {
            (low, high) = (high, 2. * high);
        }
        loop {
            let m = 0.5 * (low + high);
            if !(m > low && m < high) {
                break high;
            }
            if beyond(m) {
                low = m;
            } else {
                high = m;
            }
        }
    }
}

/// The looks of a run (module documentation, # Stopping) by its estimates, the pilot's fixed errors and the step's
/// bounds: at every output after t = 0 what is shown, and T on each lab axis held to its target there. A held axis
/// whose T has no account -- one of its parts is unread -- is not met; where it is its statistics that are unread, its
/// need is predicted from the normal theory of a variance over the N h atoms held, the relative interval z √(2/(N h)).
fn looks(estimates: &[Estimate], fixed: &Fixed, step: &[Bound]) -> Vec<Look> {
    let (population, temperature) = accounts(estimates, fixed, step);
    let mut looks = vec![];
    for (k, e) in estimates.iter().enumerate().skip(1) {
        let (h, n) = (1. - e.loss, e.trajectories);
        looks.push(Look { output: k, target: POPULATION_TARGET, account: population[k], clear: step[k].shown.clear, interval: e.shown_interval(), trajectories: n, shown: true });
        for a in (0..3).filter(|&a| fixed.temperature[k][a]) {
            let b = step[k].temperature[a];
            let account = temperature[k][a].unwrap_or(Account { model: h * fixed.model[k][1], step: h * b.bound, statistics: f64::INFINITY });
            let interval = if account.statistics.is_finite() { account.statistics } else { h * CONFIDENCE * (2. / (n * h)).sqrt() };
            looks.push(Look { output: k, target: TEMPERATURE_TARGET, account, clear: h * b.clear, interval, trajectories: n, shown: false });
        }
    }
    looks
}

/// The outputs where some look leaves the statistics no room (SPEC §5.2): no number of trajectories fits.
fn crowded(looks: &[Look]) -> Vec<usize> {
    let mut outputs: Vec<usize> = looks.iter().filter(|l| !l.fits(1.)).map(|l| l.output).collect();
    outputs.dedup();
    outputs
}

/// Each output's accounts of what is shown and of T per lab axis (ledger.rs `Account`) by the run's estimates, the model's
/// errors on the held atoms per output (`Fixed`) and the step's bounds, T's where read: the held share 1 − L weighs the
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
            let population = Account { model: (a[0] * h).min(1.), step: b.shown.bound, statistics: e.shown_statistics() };
            let temperature = std::array::from_fn(|x| {
                let relative = e.motion.as_ref().map_or(f64::INFINITY, |m| e.temperature_interval(x, m.temperature[x]));
                let bound = b.temperature[x].bound;
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

/// The model's error per output (populations, T; none at t = 0, the request's state) from the units' mean errors, every
/// folded unit's but `without`'s added (SPEC §4), and the unsampled share; a population's at most 1.
fn model_error(means: &[Vec<[f64; 2]>], without: Option<usize>, unsampled: f64) -> Vec<[f64; 2]> {
    let outputs = means.first().map_or(0, Vec::len);
    (0..outputs)
        .map(|k| {
            if k == 0 {
                return [0.; 2];
            }
            let mut a = [unsampled; 2];
            for (u, m) in means.iter().enumerate() {
                if Some(u) != without {
                    a[0] += m[k][0];
                    a[1] += m[k][1];
                }
            }
            [a[0].min(1.), a[1]]
        })
        .collect()
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
    base: f64,
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
    unsampled: f64,
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
        unsampled: f64,
        again: f64,
    ) -> Pricing<'a> {
        let rated = one_each(states, measured.output, fixed.model.len());
        let current = Run::new(model, reference);
        let assembly = current.assembly_work();
        let middles: Vec<crate::model::Middle> = (0..model.clusters().len()).map(|c| current.middle(c, centre, 0.)).collect();
        let own: Vec<f64> = rated.iter().map(|&j| rate(&current, assembly, &middles[states[j].sample.cluster], &states[j])).collect();
        drop(middles);
        let all = Level { left: None, estimates: vec![], means: vec![], base: 0. };
        let mut pricing = Pricing { request, light, kept, model, reference, centre, units, states, measured, pilot, fixed, scales, unsampled, rated, own, again, all };
        pricing.all = pricing.level(None);
        pricing
    }

    /// The pilot's groups but `left` as the pricing reads them. Its pooling runs on the thread pool: made on the calling
    /// thread, never inside one of the pool's tasks.
    fn level(&self, left: Option<usize>) -> Level {
        let parts: Vec<&Part> = self.pilot.iter().enumerate().filter(|(g, _)| Some(*g) != left).map(|(_, p)| *p).collect();
        let (estimates, means) = (pooled(&parts, self.scales).0, self.measured.means(self.fixed.model.len(), left));
        let base = self.mean(&self.own, left) * self.need(&estimates, &means, None);
        Level { left, estimates, means, base }
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
    fn need(&self, estimates: &[Estimate], means: &[Vec<[f64; 2]>], without: Option<usize>) -> f64 {
        let fixed = Fixed { model: model_error(means, without, self.unsampled), ..self.fixed.clone() };
        looks(estimates, &fixed, &vec![Bound::share(); fixed.model.len()]).iter().map(|l| l.need(1.)).fold(0., f64::max)
    }

    /// The share of the current model's W at a level that keeping unit `u` saves at the rates `rates` of its kept model:
    /// a model's W (per µs of run) is its mean rate times its product's trajectories (`need`), the kept model's with the
    /// pilot its new attempt runs.
    fn share(&self, level: &Level, u: usize, rates: &[f64]) -> f64 {
        1. - self.mean(rates, level.left) * (self.need(&level.estimates, &level.means, Some(u)) + self.again) / level.base
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
        (0..self.units.len()).into_par_iter().filter(|&u| self.share(&self.all, u, &self.own) > 0.).collect()
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
/// diagonal, the exact moments at t = 0 ((ħ/m)⟨κ_a²⟩ per lab axis, rad/µs), the pilot's fixed errors, the step's error
/// per output as last booked -- the step's share until the run's two levels are `measured` --, the ledger's items and
/// note, and the method's description.
struct Context {
    plan: Plan,
    tau: f64,
    scales: Scales,
    initial: [f64; STATES],
    momentum: V3,
    fixed: Fixed,
    step: Vec<Bound>,
    measured: bool,
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
    if c.measured {
        // The step's item: the largest it is booked at over the accounts (weighed by the held share as the totals are).
        let largest = (1..outputs).flat_map(|k| temperature[k].iter().flatten().map(|a| a.step).chain([population[k].step])).fold(0., f64::max);
        let share = (1 << ORDER) - 1;
        ledger.item(&format!("时间步长 τ = {} μs（全程不变）", c.tau), largest, &format!("同一批轨迹在 τ 与 2τ 下逐条配对实测：两者之差的 1/{share}，加其统计区间"));
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
        method: format!("{}；{trajectories} 条轨迹（每组 {GROUP} 条拉丁超立方），步长 {} μs。", c.method, c.tau),
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
        .filter(|&k| estimates[k].motion.as_ref().is_some_and(|m| m.temperature_errors.iter().all(Spread::read)) && !(c.fixed.model[k][1].is_finite() && c.step[k].temperature.iter().all(|b| b.bound.is_finite())))
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

    /// The ladder's top rung: the output interval over the least power of two below the reference flow's resonance
    /// ω τ = π -- of a 10 µs interval, 5 µs at a tweezer's 74.65 kHz (π/ω = 6.7 µs) and 0.3125 µs at a lattice's 910.7 kHz
    /// (0.55 µs); the interval itself in a slow trap.
    #[test]
    fn the_top_rung_lies_below_the_resonance() {
        let khz = |f: f64| std::f64::consts::TAU * f * 1e-3;
        assert_eq!(top_rung(10., khz(74.65)), 5.);
        assert_eq!(top_rung(10., khz(910.7)), 0.3125);
        assert_eq!(top_rung(10., khz(1.)), 10.);
    }

    /// The screen's outcomes on the reads between two levels at one output (unlimited groups: one look's interval is z
    /// of the error). The step stands where every difference fits its share, where a small one lies under a wide
    /// interval, and where one lies above its share but not clearly so among all the looks -- that one is a bound to
    /// book, not a reason to halve; it is halved where a difference is clear of its share: T's on an axis held to its
    /// target (an axis that is not held is not judged, whatever the others are held to, nor is the output at t = 0),
    /// the loss's and a row's as shown against the population's share.
    #[test]
    fn the_screen_halves_the_step_on_a_clear_difference_alone() {
        let read = |value: f64, error: f64| Read { value, spread: Spread { error, groups: f64::INFINITY } };
        let quiet = read(0., 0.);
        let at = |t: Read| Between { loss: quiet, shown: [quiet; STATES], temperature: Some([t, quiet, quiet]) };
        let halved = |reads: &[Between], held: &[[bool; 3]]| matches!(screen(reads, held), Judgement::Beyond(_));
        let held = [[false; 3], [true; 3]];
        let [sp, st] = step_bound();
        assert!(!halved(&[at(quiet), at(read(0.5 * st, 0.1 * st))], &held));
        assert!(!halved(&[at(quiet), at(read(0.5 * st, st))], &held));
        assert!(!halved(&[at(quiet), at(read(2. * st, st))], &held));
        assert!(halved(&[at(quiet), at(read(4. * st, 0.01 * st))], &held));
        assert!(!halved(&[at(quiet), at(read(9. * st, 0.))], &[[false; 3]; 2]));
        assert!(!halved(&[at(read(9. * st, 0.)), at(quiet)], &[[true; 3]; 2]));
        assert!(!halved(&[at(quiet), at(read(9. * st, 0.))], &[[false; 3], [false, true, true]]));
        assert!(halved(&[at(quiet), at(read(9. * st, 0.))], &[[false; 3], [true, false, false]]));
        let lost = Between { loss: read(2. * sp, 0.01 * sp), ..at(quiet) };
        let mut row = at(quiet);
        row.shown[3] = read(-2. * sp, 0.01 * sp);
        assert!(halved(&[at(quiet), lost], &[[false; 3]; 2]) && halved(&[at(quiet), row], &[[false; 3]; 2]));
        // The pilot's own numbers: a difference of 12 % on T with a standard error of 4 % on eleven groups is no clear
        // one among a hundred looks (Student's t at ten degrees of freedom and Šidák's tail); 60 % is.
        let pilot = |value: f64| Between { temperature: Some([Read { value, spread: Spread { error: 0.04, groups: 11. } }, quiet, quiet]), ..at(quiet) };
        let run = |value: f64| -> Vec<Between> { std::iter::once(at(quiet)).chain(std::iter::repeat_n(pilot(value), 100)).collect() };
        assert!(!halved(&run(0.12), &[[true; 3]; 101]) && halved(&run(0.6), &[[true; 3]; 101]));
    }

    /// A look on T at 800 trajectories: the model's error, the step's bound and its clear part, the statistics'
    /// interval (each relative).
    fn look(model: f64, bound: f64, clear: f64, interval: f64) -> Look {
        Look { output: 1, target: TEMPERATURE_TARGET, account: Account { model, step: bound, statistics: interval }, clear, interval, trajectories: 800., shown: false }
    }

    /// The trajectories a look needs (`Look::need`) against the closed form: with x = √(N₀/N), d = b − c and
    /// R² = ε² − A², the positive root of (S² + d²)x² + 2cd x + c² − R² = 0; at the rung below c is c/2^p. None where the
    /// clear part fills the room, or the model's error the target. On what is shown the need is no less than the draws
    /// that leave the room unseen (estimate.rs `draws_for`).
    #[test]
    fn a_look_needs_the_trajectories_that_fit_its_total() {
        let exact = |l: &Look, lower: f64| {
            let (c, d, s) = (l.clear * lower, l.account.step - l.clear, l.interval);
            let room2 = l.target * l.target - l.account.model * l.account.model;
            let x = (-c * d + (c * c * d * d + (s * s + d * d) * (room2 - c * c)).sqrt()) / (s * s + d * d);
            l.trajectories / (x * x)
        };
        for l in [look(0., 0., 0., 0.12), look(0.03, 0.08, 0., 0.08), look(0.02, 0.095, 0.09, 0.06), look(0.03, 0.05, 0.03, 0.09), look(0.01, 0.02, 0.01, 0.03)] {
            for lower in [1., 0.25] {
                let (need, exact) = (l.need(lower), exact(&l, lower));
                assert!((need / exact - 1.).abs() < 1e-12, "{l:?} at {lower}: {need} against {exact}");
            }
            assert_eq!(l.met(), l.need(1.) <= l.trajectories, "{l:?}");
        }
        let (refuted, full) = (look(0.03, 0.12, 0.10, 0.05), look(0.11, 0.02, 0.01, 0.05));
        assert!(refuted.refuted() && refuted.need(1.) == f64::INFINITY && refuted.need(0.25).is_finite());
        assert!(!full.refuted() && full.need(1.) == f64::INFINITY && full.need(0.25) == f64::INFINITY);
        assert_eq!(look(0., f64::INFINITY, 0., 0.05).need(1.), f64::INFINITY);
        let shown = Look { target: POPULATION_TARGET, shown: true, ..look(0.03, 0., 0., 0.) };
        let room = room(POPULATION_TARGET, 0.03, 0.);
        assert!((shown.need(1.) / draws_for(room) - 1.).abs() < 1e-12, "{} against {}", shown.need(1.), draws_for(room));
    }

    /// The product's moves (`next`) at 800 trajectories a level, the rung below at twice the work a trajectory (the
    /// levels' counted work 2 and 1):
    /// - every look met: done; and done at the trajectories' cap whatever the looks;
    /// - statistics alone beyond the target: on at this step to the need, at most to twice the trajectories;
    /// - a bound of pure noise (nothing clear): on at this step -- the rung below asks the same trajectories at more work;
    /// - a clear part that fills the room: the step is halved, to the rung below's need (the pilot's size at the least,
    ///   and no more where the rung below is refuted in its turn); at the steps' cap the product is stuck;
    /// - a model error that fills the target is no reason to halve: on at this step;
    /// - a clear part that leaves room: halved where the rung below asks less work -- its need at twice the work a
    ///   trajectory anew, the companion already there -- and not where it asks more;
    /// - a companion that holds more than the finer level (after a halving) is not paid for again;
    /// - a companion dearer than the finer level (it does not resolve the dynamics) prices the rung below at the finer
    ///   level's own work, no less;
    /// - on a level of the pilot's size the step is neither halved nor found stuck, whatever is clear: on to twice it.
    #[test]
    fn the_product_goes_on_stops_or_halves_the_step() {
        let (n, work) = (800, [2., 1.]);
        let whole = |need: f64| (need / GROUP as f64).ceil() as u64 * GROUP;
        assert_eq!(next(&[look(0.03, 0.04, 0., 0.08)], [n, n], work, true), Next::Done);
        assert_eq!(next(&[look(0., 0., 0., 0.3)], [MOST_TRAJECTORIES; 2], work, true), Next::Done);
        assert_eq!(next(&[look(0., 0., 0., 0.12)], [n, n], work, true), Next::More(1152));
        assert_eq!(next(&[look(0., 0., 0., 0.3)], [n, n], work, true), Next::More(2 * n));
        let noise = look(0., 0.08, 0., 0.08);
        assert_eq!(next(&[noise], [n, n], work, true), Next::More(whole(noise.need(1.))));
        let refuted = look(0.03, 0.12, 0.10, 0.05);
        assert_eq!(next(&[refuted], [n, n], work, true), Next::Halve(whole(refuted.need(0.25)).max(pilot_size())));
        assert_eq!(next(&[refuted], [n, n], work, false), Next::Stuck);
        assert_eq!(next(&[look(0.03, 0.5, 0.45, 0.05)], [n, n], work, true), Next::Halve(pilot_size()));
        assert_eq!(next(&[look(0.11, 0.02, 0.01, 0.05)], [n, n], work, true), Next::More(2 * n));
        // By the work: (need − 800)(2 + 1) here against need' · 4 below.
        let (pays, dear) = (look(0., 0.095, 0.09, 0.06), look(0., 0.05, 0.03, 0.09));
        assert!(pays.need(0.25) * 4. < (pays.need(1.) - 800.) * 3. && dear.need(0.25) * 4. > (dear.need(1.) - 800.) * 3.);
        assert_eq!(next(&[pays], [n, n], work, true), Next::Halve(whole(pays.need(0.25))));
        assert_eq!(next(&[pays], [n, n], work, false), Next::More(whole(pays.need(1.)).min(2 * n)));
        assert_eq!(next(&[dear], [n, n], work, true), Next::More(whole(dear.need(1.))));
        // The companion already holds the trajectories this step still needs: they are paid at the finer level alone,
        // (need − 800)·2 against need'·4.
        let close = look(0., 0.089, 0.085, 0.06);
        let (here, below) = (close.need(1.), close.need(0.25));
        assert!(below * 4. < (here - 800.) * 3. && below * 4. > (here - 800.) * 2.);
        assert!(matches!(next(&[close], [n, n], work, true), Next::Halve(_)));
        assert!(matches!(next(&[close], [n, 4 * n], work, true), Next::More(_)));
        // Where steps end early at jumps at either length the levels' work is nearly one, and the rung below is cheap:
        // (need − 800)·2.05 here against need'·1.05·1.05 below. A companion dearer than the finer level prices the rung
        // below at the finer level's work, not below it: need'·2 against (need − 800)·(2 + 3).
        let jumps = look(0., 0.08, 0.075, 0.07);
        assert!(matches!(next(&[jumps], [n, n], work, true), Next::More(_)) && matches!(next(&[jumps], [n, n], [1.05, 1.], true), Next::Halve(_)));
        let slight = look(0., 0.077, 0.072, 0.07);
        let (here, below) = (slight.need(1.), slight.need(0.25));
        assert!(below * 2. > (here - 800.) * 5. && below * 2. * (2. / 3.) < (here - 800.) * 5.);
        assert!(matches!(next(&[slight], [n, n], [2., 3.], true), Next::More(_)));
        // The pilot's size: its reads only screen.
        let pilot = |l: Look| Look { trajectories: pilot_size() as f64, ..l };
        for l in [refuted, pays, look(0.03, 0.5, 0.45, 0.05)] {
            for finer in [true, false] {
                assert_eq!(next(&[pilot(l)], [pilot_size(); 2], work, finer), Next::More(2 * pilot_size()));
            }
        }
    }

    /// The step is not halved for its noise (module documentation, # The step): twenty looks on T at one batch past the
    /// pilot (22 groups), independent of each other -- more than the outputs of a run are, which share their
    /// trajectories --, each from an exponential energy at the finer level and an independent one at its companion with
    /// a step error of 2 %. Over 400 such batches `next` asks for the rung below, or finds the step refuted, no more
    /// often than one look's tail at z, with that count's own interval.
    #[test]
    fn the_step_is_not_halved_for_its_noise() {
        let (bias, repetitions, outputs) = (0.02, 400, 21);
        let scales = Scales { hbar_over_mass: KT_PER_UK, frequencies: [1.; 3] };
        let fixed = Fixed { model: vec![[0.; 2]; outputs], temperature: (0..outputs).map(|k| [k > 0, false, false]).collect(), start: 0. };
        let mut energies = exponentials(0x0F1E_2D3C_4B5A_6978);
        let mut group = |factor: f64| {
            let mut g = Group::new(outputs);
            for _ in 0..GROUP {
                g.add(&(0..outputs).map(|_| held(factor * energies())).collect::<Vec<_>>());
            }
            g
        };
        let n = 2 * pilot_size();
        let halved = (0..repetitions)
            .filter(|_| {
                let (fine, coarse): (Vec<Group>, Vec<Group>) = (0..n / GROUP).map(|_| (group(1.), group(1. + 3. * bias))).unzip();
                let looks = looks(&summarize(&fine, &scales), &fixed, &booked(&between(&fine, &coarse, &scales)));
                !matches!(next(&looks, [n; 2], [2., 1.], true), Next::More(_))
            })
            .count() as f64
            / repetitions as f64;
        let p = crate::special::upper(0.5, 0.5 * CONFIDENCE * CONFIDENCE);
        let within = Spread { error: (p * (1. - p) / repetitions as f64).sqrt(), groups: repetitions as f64 }.interval();
        eprintln!("the step halved or refuted on noise in {halved:.4} of {repetitions} batches of {} looks at {n} trajectories (one look's tail {p:.4})", outputs - 1);
        assert!(halved <= p + within, "the step is halved or refuted in {halved} of the batches, one look's tail is {p}");
    }

    /// Exponential deviates of mean 1 from a splitmix64 stream of the seed `state` (the tests' own).
    fn exponentials(mut state: u64) -> impl FnMut() -> f64 {
        move || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            -((((z ^ (z >> 31)) >> 11) as f64 + 0.5) / (1u64 << 53) as f64).ln()
        }
    }

    /// An atom held in the first state with ⟨κ⟩ = 0 and ⟨κ²⟩ = `energy` on every axis.
    fn held(energy: f64) -> Fate {
        let mut rows = [0.; STATES];
        rows[0] = 1.;
        Fate::Held(Held { rows, kappa: [0.; 3], kappa2: [energy; 3] })
    }

    /// The stop reads the run's own data (module documentation, # Stopping): measured on synthetic levels. Each atom has
    /// an exponential energy E at the finer level (T = 1) and (1 + 3b)E' at its companion, E' independent of E -- the
    /// levels decorrelated, the read's noise at its largest --, so the step's error is b = 2 % and the T of a vanishing
    /// step is 1 − b; T is held to 10 %, nothing else is. From the pilot's eleven groups the product goes on by its own
    /// rule (`next`) until it stops, 400 times.
    /// - The step is not halved for its noise: `next` asks for the rung below, or finds the step refuted, in no more of
    ///   the runs than one look's tail at z, with that count's own interval.
    /// - The truth is covered: |T̂ − (1 − b)| lies within the total the run reports as often as one look at z allows,
    ///   less that count's own interval.
    /// - The stop is early by its own noise at the most: the total it would report with the true variances at its
    ///   trajectories, over the target, is on average no more than one plus the relative standard deviation of a
    ///   standard error read on the stop's groups, √((2 + 6/GROUP)/g)/2 at the mean number g of groups (a group's
    ///   mean of GROUP exponentials has the excess kurtosis 6/GROUP).
    /// - T̂ itself is not biased by the stop: its mean over the runs is 1 within its interval.
    #[test]
    fn the_stop_on_its_own_data_is_early_by_its_noise_at_the_most() {
        let (bias, repetitions) = (0.02, 400);
        let scales = Scales { hbar_over_mass: KT_PER_UK, frequencies: [1.; 3] };
        let fixed = Fixed { model: vec![[0.; 2]; 2], temperature: vec![[false; 3], [true, false, false]], start: 0. };
        let mut energies = exponentials(0x1234_5678_9ABC_DEF1);
        let mut group = |factor: f64| {
            let mut g = Group::new(2);
            for _ in 0..GROUP {
                let atom = held(factor * energies());
                g.add(&[atom.clone(), atom]);
            }
            g
        };
        // The total a run would report at N trajectories with the true variances: T's relative standard error 1/√N, the
        // read's (1 + 3b)√2/(3√N) (two independent levels over 2^p − 1), each at z.
        let truly = |n: f64| ((CONFIDENCE / n.sqrt()).powi(2) + (bias + CONFIDENCE * (1. + 3. * bias) * 2f64.sqrt() / (3. * n.sqrt())).powi(2)).sqrt();
        let (mut halved, mut stopped, mut covered, mut early, mut groups, mut mean, mut squares) = (0, 0, 0, 0., 0., 0., 0.);
        for _ in 0..repetitions {
            let (mut fine, mut coarse): (Vec<Group>, Vec<Group>) = (vec![], vec![]);
            let (mut want, mut asked) = (pilot_size(), false);
            let end = loop {
                while (fine.len() as u64) < want / GROUP {
                    fine.push(group(1.));
                    coarse.push(group(1. + 3. * bias));
                }
                let estimates = summarize(&fine, &scales);
                let looks = looks(&estimates, &fixed, &booked(&between(&fine, &coarse, &scales)));
                asked |= matches!(next(&looks, [want; 2], [2., 1.], true), Next::Halve(_));
                match next(&looks, [want; 2], [2., 1.], false) {
                    Next::More(more) => want = more,
                    Next::Done => break Some((estimates[1].motion.unwrap().temperature[0], looks.iter().find(|l| !l.shown).unwrap().account.total())),
                    Next::Stuck => break None,
                    Next::Halve(_) => unreachable!("no rung below"),
                }
            };
            halved += usize::from(asked || end.is_none());
            let Some((t, total)) = end else { continue };
            let n = want as f64;
            stopped += 1;
            covered += usize::from((t - (1. - bias)).abs() <= total * t);
            early += truly(n) / TEMPERATURE_TARGET;
            groups += n / GROUP as f64;
            mean += (t - 1.) * n.sqrt();
            squares += (t - 1.) * (t - 1.) * n;
        }
        let (r, s) = (repetitions as f64, stopped as f64);
        let (halved, coverage, early, groups, mean) = (halved as f64 / r, covered as f64 / s, early / s, groups / s, mean / s);
        let p = crate::special::upper(0.5, 0.5 * CONFIDENCE * CONFIDENCE);
        let noise = 0.5 * ((2. + 6. / GROUP as f64) / groups).sqrt();
        let deviation = (squares / s - mean * mean).sqrt();
        eprintln!("the stop on synthetic data: {stopped} of {repetitions} runs stop, at {groups:.1} groups on average; the step halved or refuted in {halved:.4} of the runs; the truth covered in {coverage:.4} of the stops (one look {:.4}); the true total over the target {early:.4} (the estimate's own noise {noise:.4}); (T̂ − 1)√N has the mean {mean:+.4} and the deviation {deviation:.4}", 1. - p);
        let within = |error: f64, runs: f64| Spread { error, groups: runs }.interval();
        assert!(halved <= p + within((p * (1. - p) / r).sqrt(), r), "the step is halved or refuted in {halved} of the runs, one look's tail is {p}");
        assert!(coverage >= 1. - p - within((p * (1. - p) / s).sqrt(), s), "the truth is covered in {coverage} of the stops, one look's share is {}", 1. - p);
        assert!(early <= 1. + noise, "the true total at the stop is {early} of the target, the estimate's noise allows {}", 1. + noise);
        assert!(mean.abs() <= within(deviation / s.sqrt(), s), "T̂ at the stop is biased: (T̂ − 1)√N has the mean {mean} ± {}", within(deviation / s.sqrt(), s));
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
    /// error of 3 % and a step bound of 2 %, 40 % and 1 % on the axes: there is one look on what is shown and one on each
    /// held axis, each with its own step bound and statistics; an interval of 30 % on y neither keeps the run from
    /// stopping nor asks for a trajectory, where the same interval on x does both, by (interval/room)²; y's step bound,
    /// beyond any target, crowds no output where on a held axis it would -- where it is clear: a bound of noise leaves
    /// room, and the trajectories it asks are finite. A model error that fills the target crowds its output, and is no
    /// refutation of the step. Every axis has its account, its own step and statistics in it, and the result's per-axis
    /// totals are those.
    #[test]
    fn the_looks_and_the_accounts_follow_each_axis() {
        let fixed = |held: [bool; 3]| Fixed { model: vec![[0.; 2], [0., 0.03]], temperature: vec![[false; 3], held], start: 0. };
        let none = Bound { shown: Booked::held(0.), temperature: [Booked::held(0.); 3] };
        let step = [none, Bound { temperature: [0.02, 0.4, 0.01].map(Booked::held), ..none }];
        let held = [true, false, true];
        let n = 800.;
        let quiet = estimate([25., 6., 10.], [0.25, 0.9, 0.1], n);
        let run = [quiet.clone(), quiet];
        let all = looks(&run, &fixed(held), &step);
        assert_eq!(all.iter().map(|l| (l.output, l.shown, l.account.step)).collect::<Vec<_>>(), [(1, true, 0.), (1, false, 0.02), (1, false, 0.01)]);
        assert!(all.iter().all(|l| l.met() && l.need(1.) < n) && crowded(&all).is_empty());
        let noisy = estimate([25., 6., 10.], [3.75, 0.06, 0.1], n);
        let run = [noisy.clone(), noisy];
        let x = looks(&run, &fixed(held), &step)[1];
        let need = n * (CONFIDENCE * 3.75 / 25. / room(TEMPERATURE_TARGET, 0.03, 0.02)).powi(2);
        assert!(!x.met() && (x.need(1.) - need).abs() < 1e-9 * need, "{} against {need}", x.need(1.));
        let every = looks(&run, &fixed([true; 3]), &step);
        assert!(every[2].refuted() && crowded(&every) == [1]);
        let noise = [none, Bound { temperature: [Booked { bound: 0.4, clear: 0. }; 3], ..none }];
        let every = looks(&run, &fixed([true; 3]), &noise);
        assert!(crowded(&every).is_empty() && every.iter().all(|l| l.need(1.).is_finite()) && !every[2].met());
        let mut full = fixed(held);
        full.model[1][1] = 0.11;
        let crowd = looks(&run, &full, &step);
        assert!(crowded(&crowd) == [1] && crowd.iter().all(|l| !l.refuted()));
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
