//! The pipeline: request, reach and model, pilot, decisions, batches, checkpoints, result (SPEC §3, §5, §9).
//!
//! # The pipeline
//! 1. The request's checks; the light; the initial components; the reach and the model with every unit whose existence
//!    gate fails kept (model.rs, SPEC §1.5).
//! 2. The pilot (`pilot_size` trajectories, numbered from `MOST_TRAJECTORIES` so that none of its streams is a run's):
//!    it predicts only (decisions.md A4) -- the run's step and its error (# The step), the model's units' errors measured
//!    at its states, the outputs where T is held to its target and T's scale there, the held share. Its result is the
//!    first checkpoint.
//! 3. The model's decisions (SPEC §5.2), from the pilot alone.
//! 4. The product: batches of whole groups at the pilot's step (adapt.rs), every trajectory run to the last output,
//!    until every output's statistics meet the targets the pilot fixed (`Targets`), or the trajectories reach
//!    `MOST_TRAJECTORIES`.
//! 5. The step's falsifier (SPEC §5.3): the product's first `pilot_size` trajectories again at half the step on the same
//!    streams, and the product's trajectories within the step's error of the escape; an output read clearly beyond the
//!    step's share takes the step down by the measured ratio, and the attempt runs again.
//! 6. The result (contract §2.2, `assemble`).
//!
//! # The step (SPEC §2.2)
//! One step τ for the run (adapt.rs). The ladder's rungs are the output interval over powers of two; its top one the
//! largest below the reference flow's resonance, ω τ < π at the prepared trap's largest frequency (`top_rung`). The pilot
//! is read at a rung -- every step doubled, the signed differences accumulated (adapt.rs `Drift`) -- and its step error
//! per output (estimate.rs `stepped`) decides it against the step's share (`rung`; decide.rs): within everywhere, the
//! rung is the run's step; beyond somewhere, the rung at which the measured error meets the share; unreadable, the next
//! rung; down to the step at which a trajectory would take `MOST_STEPS` steps, where the readable bound is booked and
//! flagged. What is booked per output is the read's bound on the rows as shown and on T (`Fixed::step`); the loss's is the
//! share of the product's trajectories whose loss energy lies within the pilot's accumulated error of the escape -- its
//! mean and z of the atoms' spread (`Fixed::windows`). That share's statistics are the output's own (what no trajectory
//! saw is booked there, estimate.rs `unseen`); it refutes the step where Wilson's near end at its count lies beyond the
//! step's share.
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
//! the statistics need in the room √(ε² − A² − b²) its errors leave (`Targets::need` on the pilot's estimates) and w the
//! counted work a trajectory spends per µs (`rate`, the step's count through the model's generator at the pilot's states),
//! falls by a share of itself beyond that share's spread, the jackknife over the pilot's groups (`candidates`) -- the kept
//! model's W counting the pilot its attempt runs again: every keep is a new attempt, a new model and its own pilot. The
//! model's falsifier keeps one more unit on the product's first trajectories (common random numbers): the unit of the
//! largest booked error among those whose pairs cost no more than the product they check -- a check dearer than what it
//! checks is not one. What the pairs read clearly beyond the fold's booked error (`cleared`) is booked in its place; the
//! unit is kept where that leaves an output no room for the statistics -- the rule the pilot's measured errors keep a unit
//! by --, and else the product goes on until its statistics fit the room left. An unreadable read leaves the booked
//! errors standing.
//!
//! # Stopping (SPEC §3)
//! Each output's total error against its target is E² = A² + b² + S² (ledger.rs): the statistics may take the room
//! R² = ε² − A² − b² the model and the step leave. The targets the stopping rule reads are fixed by the pilot alone (`Fixed`),
//! never by the run's own estimates, so that the stopping depends on the run's standard errors and nothing else: the
//! model's error A from the units measured at the pilot's states, the held share by which T's error counts (the user's
//! definition: T describes the atoms held), and T's scale -- the pilot's T, at the outputs where its upper bound
//! T + t·SE lies in the validated regime on every axis (`CLASSICAL`; elsewhere T is reported, not chased). A batch brings
//! the trajectories to the number the latest estimates predict every output needs (SE ∝ N^{−1/2}; the shown rows at least
//! the count whose unseen share fits the room, estimate.rs `unseen`).
//!
//! # Checkpoints
//! The decisions take counts, never the wall clock, so a request gives the same result on any machine. The wall clock only
//! times the messages (`Messages`): one clock for the whole solve, set at every message; whenever `CHECKPOINT` has passed
//! since the last one, the stage that is running sends a complete result of the groups it has finished so far in their
//! order (a prefix: whole groups chosen by their number alone). A result is passed on where it rests on two groups at
//! least (the jackknife's least for a spread) and on no fewer trajectories than the attempt's last one -- a product's
//! first groups, or a pilot at a finer rung, are no better than the result before them --; else the last result goes out
//! again, so that the app hears from the solve whatever stage it is in.
use crate::adapt::{self, Drift};
use crate::decide::{halved, judge, pays, Judgement, Read};
use crate::ensemble::{Ensemble, Quadrature, Start, Well, GROUP};
use crate::estimate::{half_window, paired, spread, stepped, summarize, unseen, wilson, Estimate, Fate, Group, Held, Output, Scales, Spread, Stepped};
use crate::field::{Geometry, Light};
use crate::ledger::{percent, span, when, Account, Ledger};
use crate::linalg::V3;
use crate::model::{Envelope, Lights, Model, Run, Sample, Trap, Unit, UnitError, GROUND};
use crate::request::STATES;
use crate::step::{Dynamics, Segment, Trajectory};
use crate::units::{hbar_over_mass, CLASSICAL, CONFIDENCE, KT_PER_UK, MOST_STEPS, MOST_TRAJECTORIES, ORDER, STEP_SHARE, TEMPERATURE_TARGET};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::f64::consts::PI;
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
    let mut messages = Messages { send: checkpoint, clock: Instant::now(), last: None, least: 0 };
    loop {
        // An attempt's results improve on its own last one, from two groups on (module documentation, # Checkpoints).
        messages.least = 2 * GROUP;
        match attempt(request, &light, &kept, reach, step, &mut messages)? {
            Attempt::Grow(t, tau) => (reach, step) = (t, Some(tau)),
            Attempt::Keep(u, tau) => {
                kept.push(u);
                step = Some(tau);
            }
            Attempt::Refine(tau) => step = Some(tau),
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
/// temperature, µK), a unit is to be kept -- each with the step the attempt's pilot took, the next attempt's first rung --,
/// the step is refuted (the step to take), or the solution.
enum Attempt {
    Grow(f64, f64),
    Keep(Unit, f64),
    Refine(f64),
    Done(Solution),
}

/// The initial components (SPEC §2.5): the ground states the request's initial density holds (diagonal, validated) and their
/// populations normalized; refused where it holds an excited state, whose motion no prepared well describes.
fn components(request: &Request) -> Result<Vec<(usize, f64)>, String> {
    let diagonal: Vec<f64> = (0..STATES).map(|s| request.initial_density[(s, s)].re).collect();
    if diagonal[GROUND..].iter().any(|&p| p > 0.) {
        return Err("运动计算的初态需在基态上".into());
    }
    let total: f64 = diagonal[..GROUND].iter().sum();
    Ok((0..GROUND).filter(|&s| diagonal[s] > 0.).map(|s| (s, diagonal[s] / total)).collect())
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

/// A rung's outcome on the pilot's step error (module documentation, # The step; decide.rs `judge`): the family of every
/// output's rows as shown, and of T per lab axis where it is held to its target, each against the step's share, at the
/// scheme's order -- beyond, the factor on τ at which the largest measured error meets its share.
fn rung(reads: &[Stepped], held_to: &[Option<V3>]) -> Judgement {
    let [sp, st] = step_bound();
    let mut looks: Vec<(Read, f64)> = vec![];
    for (k, r) in reads.iter().enumerate().skip(1) {
        looks.extend(r.shown.iter().map(|x| (*x, sp)));
        if let (Some(_), Some(t)) = (held_to[k], &r.temperature) {
            looks.extend(t.iter().map(|x| (*x, st)));
        }
    }
    judge(&looks, ORDER as f64)
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

/// Whether temperatures per lab axis (µK) lie in the validated regime of the trap's frequencies (rad/µs) on every axis,
/// k_BT ≥ `CLASSICAL` ħω (SPEC §0, §3): the one test of what the result reports as classical and of where T is held to
/// its target.
fn classical(temperature: &V3, frequencies: &V3) -> bool {
    (0..3).all(|a| KT_PER_UK * temperature[a] >= CLASSICAL * frequencies[a])
}

/// One attempt at the solve with the units `kept` and the reach at `reach` µK (module documentation, # The pipeline).
fn attempt(request: &Request, light: &Light, kept: &[Unit], reach: f64, step: Option<f64>, messages: &mut Messages) -> Result<Attempt, String> {
    let hm = hbar_over_mass(request.mass_kg);
    let initial = components(request)?;
    let reference = alone(initial.iter().max_by(|a, b| a.1.total_cmp(&b.1)).ok_or("运动计算的初态需在基态上")?.0);
    // The model: every unit whose fold does not exist over the reach kept (SPEC §1.5), closed again until none fails (a
    // model lists only what it folds, model.rs `units`).
    let mut kept = kept.to_vec();
    let (model, trap) = loop {
        let model = Model::new(request, light, &kept)?;
        let trap = model.trap(&reference, &model.prepared())?;
        let judged = model.reach(&trap, &reference, reach, unseen(MOST_TRAJECTORIES as f64));
        let units = model.units();
        let failing: Vec<Unit> =
            units.iter().zip(model.gates(&units, &judged)).filter(|(_, g)| !g.exists()).map(|(u, _)| u.clone()).collect();
        if failing.is_empty() {
            break (model, trap);
        }
        kept.extend(failing);
    };
    let run = Run::new(&model, reference);
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
    let depth = ensemble.depths()[initial.iter().position(|c| alone(c.0) == reference).unwrap_or(0)];
    let mut ledger = Ledger::default();
    ledger.note = Some(format!("初始系综：各初态在自己的准备阱中的束缚原子（降低的等温分布乘每轴谐振子的 Wigner 因子，T = {} μK），参考态阱深 {:.1} k_BT", request.temperature_uk, depth));
    if unheld {
        ledger.flag("运行中的势没有极小：阱频与 n̄ 按准备阱报告".into());
    }
    let exact = |kinetic: &[Quadrature]| -> (V3, f64) {
        (ensemble.momentum(&kinetic.iter().map(|k| k.eta).collect::<Vec<_>>()), kinetic.iter().map(|k| k.difference / k.eta).fold(0., f64::max))
    };
    let first = ensemble.kinetic(0);
    let (momentum, start) = exact(&first);
    let mut context = Context {
        momentum,
        fixed: Fixed { model: vec![[0.; 2]; outputs], cleared: vec![0.; outputs], held: vec![1.; outputs], temperature: vec![None; outputs], step: vec![step_bound(); outputs], windows: vec![], start },
        tau: step.unwrap_or_else(|| top_rung(plan.times[1] - plan.times[0], trap.frequencies.iter().copied().fold(0., f64::max))),
        plan,
        scales,
        initial: row,
        ledger,
        method: format!("半经典 3D：{}", model.describe().trim_end_matches('。')),
    };

    // The outputs where T is held to its target, and its scale there (module documentation, # Stopping): the pilot's T
    // where it is read on every axis and its upper bound lies in the validated regime.
    let held_to = |estimates: &[Estimate]| -> Vec<Option<V3>> {
        estimates
            .iter()
            .enumerate()
            .map(|(k, e)| {
                let m = e.motion.as_ref().filter(|m| k > 0 && m.temperature_errors.iter().all(Spread::read))?;
                let upper: V3 = std::array::from_fn(|a| m.temperature[a] + m.temperature_errors[a].interval());
                (m.temperature.iter().all(|&t| t > 0.) && classical(&upper, &report.lab)).then_some(m.temperature)
            })
            .collect()
    };

    // The pilot (module documentation, # The pipeline, # The step): its own trajectory numbers, read at the ladder's rungs
    // from its first one until its step error fits the step's share.
    let first = MOST_TRAJECTORIES / GROUP;
    let (pilot, reads) = loop {
        let tau = context.tau;
        let pilot = run_groups(first, first + pilot_size() / GROUP, Shape::of(&context.plan, true), &|i| trajectory(&run, &ensemble, tau, true, &[], &context.plan, true, i), messages, &mut |prefix, messages| {
            let (e, l, c) = pooled(prefix, &context.scales);
            messages.result(&assemble(&context, &e, &l, &c, prefix.len() as u64 * GROUP), None);
        });
        let parts: Vec<&Part> = pilot.iter().collect();
        let reads = stepped(&parts, &context.scales);
        let next = match rung(&reads, &held_to(&pooled(&parts, &context.scales).0)) {
            Judgement::Within => break (pilot, reads),
            Judgement::Beyond(factor) => halved(tau, factor),
            Judgement::Unreadable => halved(tau, 1.),
        };
        if context.plan.end / next > MOST_STEPS as f64 {
            context.ledger.flag(format!("时间步长的误差在每条轨迹 {MOST_STEPS} 步的上限内读不到份额以内：按读到的界记账"));
            break (pilot, reads);
        }
        context.tau = next;
    };
    let tau = context.tau;
    let pilot: Vec<&Part> = pilot.iter().collect();
    let (estimates, listed, counts) = pooled(&pilot, &context.scales);
    if let Some(t) = outgrown(&estimates, reach) {
        return Ok(Attempt::Grow(t, tau));
    }
    (context.momentum, context.fixed.start) = exact(&ensemble.kinetic(counts.steps));
    // The step's error booked per output: the read's bound on the rows as shown -- a probability's error is at most 1,
    // which an unreadable one is booked at, flagged -- and on T, unbounded where it is not read (T's account is then none,
    // flagged where the run shows T, `assemble`); and the loss energy's window, unbounded where no trajectory reads it.
    let bound = |x: &Read| x.bound(1.);
    let rows: Vec<f64> = reads.iter().map(|r| r.shown.iter().map(bound).fold(0., f64::max)).collect();
    let unread: Vec<usize> = (1..outputs).filter(|&k| !rows[k].is_finite()).collect();
    if !unread.is_empty() {
        context.ledger.flag(format!("{}：步长对布居的误差读不出，按 1 记账", span(&unread, &context.plan.times)));
    }
    context.fixed.step = reads
        .iter()
        .zip(&rows)
        .map(|(r, &shown)| [shown.min(1.), r.temperature.as_ref().map_or(f64::INFINITY, |t| t.iter().map(bound).fold(0., f64::max))])
        .collect();
    context.fixed.windows = reads.iter().map(|r| r.energy.map_or(f64::INFINITY, |e| e[0].abs() + CONFIDENCE * e[1])).collect();

    // The model's errors at the pilot's states (SPEC §5.2): each folded unit's per output, the held atoms' mean.
    let units = model.units();
    let mut samples: Vec<State> = vec![];
    let (mut at, mut from) = (vec![], vec![]);
    for (g, p) in pilot.iter().enumerate() {
        for (k, s) in p.samples.iter().enumerate() {
            samples.extend(s.iter().cloned());
            at.extend(std::iter::repeat_n(k, s.len()));
            from.extend(std::iter::repeat_n(g, s.len()));
        }
    }
    let each: Vec<Vec<UnitError>> = samples.par_iter().map(|s| model.errors(&units, std::slice::from_ref(&s.sample)).into_iter().map(|e| e[0]).collect()).collect();
    let errors: Vec<Vec<UnitError>> = (0..units.len()).map(|u| each.iter().map(|e| e[u]).collect()).collect();
    let measured = Measured { errors: &errors, output: &at, group: &from };
    let means = measured.means(outputs, None);
    let (population, temperature): (Vec<Vec<f64>>, Vec<Vec<f64>>) =
        means.iter().map(|unit| (unit.iter().map(|m| m[0]).collect(), unit.iter().map(|m| m[1]).collect())).unzip();
    let unsampled = unseen(MOST_TRAJECTORIES as f64);
    context.fixed.model = model_error(&means, None, unsampled);
    context.fixed.held = estimates.iter().map(|e| 1. - e.loss).collect();
    context.fixed.temperature = held_to(&estimates);
    let mut targets = Targets::new(&context.fixed);
    // A unit whose error leaves an output no room for the statistics is kept (SPEC §5.2): of the outputs left none, the unit
    // whose errors there, each relative to its target and held share, add up to the most.
    let crowded: Vec<usize> = (1..outputs).filter(|&k| targets.population[k] == 0. || targets.temperature[k].is_some_and(|t| t.1 == 0.)).collect();
    // A unit's errors at an output on what is shown and on T, weighed by the pilot's held share as the totals are (an
    // output where the pilot holds no atom weighs nothing: its units' errors are their bounds, `Measured::means`).
    let held = context.fixed.held.clone();
    let weighed = |u: usize, k: usize| -> [f64; 2] { if held[k] > 0. { [population[u][k] * held[k], temperature[u][k] * held[k]] } else { [0.; 2] } };
    let weight = |u: usize| -> f64 { crowded.iter().map(|&k| weighed(u, k)[0] / POPULATION_TARGET + weighed(u, k)[1] / TEMPERATURE_TARGET).sum() };
    if let Some(u) = (0..units.len()).max_by(|&a, &b| weight(a).total_cmp(&weight(b))).filter(|&u| weight(u) > 0.) {
        return Ok(Attempt::Keep(units[u].clone(), tau));
    }
    // The greedy keep by pays (SPEC §5.2): the unit whose keeping saves the most predicted work beyond its spread.
    let ranked = candidates(request, light, &kept, &model, reference, &units, &measured, &samples, &pilot, &context.fixed, &context.scales, unsampled);
    if let Some(c) = ranked.first().filter(|c| pays(0., c.saving)) {
        return Ok(Attempt::Keep(units[c.unit].clone(), tau));
    }
    // The ledger's items: each folded unit's error on what is shown and on T (its largest over the outputs, weighed by the
    // held share as the totals are), the unsampled share, the step.
    for (u, unit) in units.iter().enumerate() {
        let shown = (1..outputs).map(|k| weighed(u, k)[0].max(weighed(u, k)[1])).fold(0., f64::max);
        if shown > 0. {
            context.ledger.item(&model.label(unit), shown, "二阶消去（折叠），沿试探轨迹实测");
        }
    }
    context.ledger.item("折叠判定之外的未采样份额", unsampled, "计入模型误差");
    let booked = context.fixed.step.iter().skip(1).map(|b| b[0].max(b[1])).fold(0., f64::max);
    context.ledger.item(&format!("时间步长 τ = {tau} μs（全程不变）"), booked, "试探上每步减半实测、沿轨迹累加；产品上以 τ/2 否证");
    messages.result(&assemble(&context, &estimates, &listed, &counts, pilot_size()), None);

    // The product (module documentation, # Stopping).
    let mut parts: Vec<Part> = vec![];
    let started = Instant::now();
    product(&run, &ensemble, &context, &targets, &mut parts, batch(targets.need(&estimates), pilot_size()), started, messages);
    // The step's falsifier (SPEC §5.3) on the product's first groups at half the step, and the product's trajectories
    // within the step's error of the escape; while it runs, the product's result is the checkpoint. Refuted, the attempt
    // runs again at the rung the measured ratio gives.
    let checked = pilot_size() / GROUP;
    let all: Vec<&Part> = parts.iter().collect();
    let (e, l, c) = pooled(&all, &context.scales);
    let current = assemble(&context, &e, &l, &c, all.len() as u64 * GROUP);
    let fine = run_groups(0, checked, Shape::of(&context.plan, false), &|i| trajectory(&run, &ensemble, 0.5 * tau, false, &context.fixed.windows, &context.plan, false, i), messages, &mut |_, messages| messages.result(&current, None));
    let share = step_bound();
    let pairs = refuted(&all[..checked as usize], &fine, &context.scales, &targets, 0.5, ORDER as f64, &|_| share);
    let escape = e.iter().skip(1).any(|x| x.judged > 0. && 1. - wilson(x.judged - x.near, x.judged) > share[0]).then_some(1.);
    if let Some(factor) = [pairs, escape].into_iter().flatten().reduce(f64::min) {
        let next = halved(tau, factor);
        if context.plan.end / next <= MOST_STEPS as f64 {
            return Ok(Attempt::Refine(next));
        }
        context.ledger.flag(format!("步长否证读到超出步长的份额，而每条轨迹的步数已达上限 {MOST_STEPS}，无法再减半"));
    }
    let result = assemble(&context, &e, &l, &c, all.len() as u64 * GROUP);
    // The model's falsifier (SPEC §5.2; module documentation, # The model): one more unit kept on the product's first groups,
    // the unit of the largest booked error (its measured errors on what is shown and on T, each over its target and weighed by
    // the held share) among those whose pairs cost no more than the product.
    let total = |u: usize| -> f64 { (1..outputs).map(|k| weighed(u, k)[0] / POPULATION_TARGET + weighed(u, k)[1] / TEMPERATURE_TARGET).sum() };
    let next = ranked
        .iter()
        .filter(|c| total(c.unit) > 0. && (checked * GROUP) as f64 * c.ratio <= all.len() as f64 * GROUP as f64)
        .max_by(|a, b| total(a.unit).total_cmp(&total(b.unit)))
        .map(|c| c.unit);
    let Some(u) = next else {
        return Ok(Attempt::Done(result));
    };
    let with = Model::new(request, light, &[kept.as_slice(), std::slice::from_ref(&units[u])].concat())?;
    let finer = Run::new(&with, reference);
    let fine = run_groups(0, checked, Shape::of(&context.plan, false), &|i| trajectory(&finer, &ensemble, tau, false, &context.fixed.windows, &context.plan, false, i), messages, &mut |_, messages| messages.result(&result, None));
    // What the pairs read clearly beyond the fold's booked error is booked in its place: on the rows as shown as it is
    // read there, and on T.
    let clear = cleared(&all[..checked as usize], &fine, &context.scales, &targets);
    let mut largest = 0.;
    for k in 1..outputs {
        let excess = [clear[k][0] - weighed(u, k)[0], clear[k][1] - temperature[u][k]].map(|x| x.max(0.));
        context.fixed.cleared[k] += excess[0];
        context.fixed.model[k][1] += excess[1];
        largest = f64::max(largest, (weighed(u, k)[0] + excess[0]).max(weighed(u, k)[1] + excess[1] * held[k]));
    }
    targets = Targets::new(&context.fixed);
    if (1..outputs).any(|k| targets.population[k] == 0. || targets.temperature[k].is_some_and(|t| t.1 == 0.)) {
        return Ok(Attempt::Keep(units[u].clone(), tau));
    }
    let label = model.label(&units[u]);
    match context.ledger.items.iter_mut().find(|i| i.label == label) {
        Some(item) => item.estimate = item.estimate.max(largest),
        None => context.ledger.item(&label, largest, "二阶消去（折叠），沿试探轨迹实测"),
    }
    // The statistics against the room the re-booked errors leave.
    let n = parts.len() as u64 * GROUP;
    if !(1..outputs).all(|k| targets.met(&e, k)) && n < MOST_TRAJECTORIES {
        product(&run, &ensemble, &context, &targets, &mut parts, batch(targets.need(&e), n), started, messages);
    }
    let all: Vec<&Part> = parts.iter().collect();
    let (e, l, c) = pooled(&all, &context.scales);
    Ok(Attempt::Done(assemble(&context, &e, &l, &c, all.len() as u64 * GROUP)))
}

/// The trajectories a batch brings a run of `have` to, for the `need` its estimates predict: whole groups, more than it
/// has, at most twice as many -- the need is predicted from a variance read on the groups so far (the pilot's eleven at
/// first, whose largest read over the outputs runs high), and a batch that only doubles them leaves at most half of the
/// work to a wrong prediction --, and `MOST_TRAJECTORIES` at most.
fn batch(need: f64, have: u64) -> u64 {
    ((need / GROUP as f64).ceil() as u64 * GROUP).clamp(have + GROUP, 2 * have).min(MOST_TRAJECTORIES)
}

/// The product (module documentation, # Stopping): batches of whole groups at the run's step, from the trajectories it
/// has to `want` and on by the estimates' prediction (`batch`), until every output's statistics meet the targets or the
/// trajectories reach `MOST_TRAJECTORIES`; each batch's result is a checkpoint.
#[allow(clippy::too_many_arguments)]
fn product(run: &Run, ensemble: &Ensemble, context: &Context, targets: &Targets, parts: &mut Vec<Part>, mut want: u64, started: Instant, messages: &mut Messages) {
    let outputs = context.plan.times.len();
    loop {
        let have = parts.len() as u64 * GROUP;
        let pace = |done: u64| (done > 0).then(|| started.elapsed().as_secs_f64() * 1e3 / done as f64);
        let new = run_groups(have / GROUP, want / GROUP, Shape::of(&context.plan, false), &|i| trajectory(run, ensemble, context.tau, false, &context.fixed.windows, &context.plan, false, i), messages, &mut |prefix, messages| {
            let all: Vec<&Part> = parts.iter().chain(prefix.iter().copied()).collect();
            let n = all.len() as u64 * GROUP;
            let (e, l, c) = pooled(&all, &context.scales);
            messages.result(&assemble(context, &e, &l, &c, n), pace(n).map(|p| p * (want - n) as f64));
        });
        parts.extend(new);
        let all: Vec<&Part> = parts.iter().collect();
        let n = all.len() as u64 * GROUP;
        let (e, l, c) = pooled(&all, &context.scales);
        if (1..outputs).all(|k| targets.met(&e, k)) || n >= MOST_TRAJECTORIES {
            break;
        }
        want = batch(targets.need(&e), n);
        messages.result(&assemble(context, &e, &l, &c, n), pace(n).map(|p| p * (want - n) as f64));
    }
}

/// The pilot's size: the fewest whole groups whose unseen share fits the statistics' share of the population target,
/// z²/(N + z²) ≤ √(1 − s²) ε (estimate.rs `unseen`, units.rs `STEP_SHARE`) -- the smallest ensemble that can see a share as
/// large as the target, 88 trajectories. The falsifiers take as many.
fn pilot_size() -> u64 {
    let room = (1. - STEP_SHARE * STEP_SHARE).sqrt() * POPULATION_TARGET;
    let n = CONFIDENCE * CONFIDENCE * (1. / room - 1.);
    (n / GROUP as f64).ceil() as u64 * GROUP
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

/// What one trajectory leaves: what it is at every output (estimate.rs `Output`), its (r, κ) at the listed outputs where
/// held, its states at the outputs where held (`keep`: the pilot's, for the model's errors and prices), its counts.
struct Record {
    outputs: Vec<Output>,
    listed: Vec<Option<(V3, V3)>>,
    samples: Vec<Option<State>>,
    counts: Counts,
}

/// One trajectory from its initial state through the plan at the step τ (module documentation, # Outputs and windows):
/// `read`, its steps are doubled and their errors accumulated with it (adapt.rs `Drift`, the pilot's); with the loss
/// energy's `windows` per output, it is judged where `margin` -- its escape energy less its loss energy -- lies within
/// them: at an output where held, and from its loss on at the state it was lost in (module documentation, # The step). A
/// trajectory the scheme cannot advance (adapt.rs) fails from the first output it does not reach.
#[allow(clippy::too_many_arguments)]
fn run<D: Dynamics>(model: &D, start: Start, index: u64, tau: f64, read: bool, windows: &[f64], margin: &dyn Fn(&Trajectory) -> f64, plan: &Plan, keep: bool) -> Record {
    let outputs = plan.times.len();
    let interval = plan.times[1] - plan.times[0];
    let widest = plan.halves.iter().flatten().fold(0., |a: f64, &b| a.max(b));
    let mut t = Trajectory::new(model, start.state, start.r, start.kappa, SEED, index);
    let mut drift = read.then_some(Drift::ZERO);
    // Per output and lab axis: the window's time held, ∫κ dt and ∫κ² dt.
    let mut moments = vec![[[0.; 3]; 3]; outputs];
    let mut out = vec![Output { fate: Fate::Lost, near: None, energy: None }; outputs];
    let mut listed = vec![None; plan.listed.len()];
    let mut samples: Vec<Option<State>> = vec![None; if keep { outputs } else { 0 }];
    let mut steps = 0;
    // The first output the trajectory does not reach held, and whether the scheme could not advance it.
    let (mut reached, mut failed) = (outputs, false);
    let mut record = |k: usize, t: &Trajectory, drift: &Option<Drift>, out: &mut Vec<Output>| {
        // The atom at the output's instant; from the first output on its κ and κ² are its window's means (below).
        let held = Held { rows: model.readout(t.cluster, &t.point(), &t.psi), kappa: t.kappa, kappa2: t.kappa.map(|x| x * x), drift: drift.clone() };
        out[k] = Output { fate: Fate::Held(held), near: windows.get(k).map(|w| margin(t).abs() <= *w), energy: drift.as_ref().map(|d| d.energy) };
        if let Ok(j) = plan.listed.binary_search(&k) {
            listed[j] = Some((t.r, t.kappa));
        }
        if keep {
            let sample = Sample { cluster: t.cluster, at: t.point(), psi: t.psi.clone() };
            samples[k] = Some(State { sample, tau, index });
        }
    };
    record(0, &t, &drift, &mut out);
    {
        let mut each = |s: &Segment| {
            steps += 1;
            let (s0, s1) = (s.start, s.start + s.tau);
            let first = (((s0 - widest) / interval).floor().max(1.)) as usize;
            let last = (((s1 + widest) / interval).ceil() as usize).min(outputs - 1);
            for k in first..=last {
                for a in 0..3 {
                    let h = plan.halves[k][a];
                    let (lo, hi) = (s0.max(plan.times[k] - h), s1.min(plan.times[k] + h));
                    if hi > lo {
                        let (m1, m2) = s.moments(lo, hi);
                        let w = &mut moments[k][a];
                        w[0] += hi - lo;
                        w[1] += m1[a];
                        w[2] += m2[a];
                    }
                }
            }
        };
        for k in 1..outputs {
            failed = adapt::advance(model, tau, &mut t, drift.as_mut(), plan.times[k], &mut each).is_err();
            if failed || t.lost.is_some() {
                reached = k;
                break;
            }
            record(k, &t, &drift, &mut out);
        }
        if reached == outputs {
            // Through the last output's window; a failure there leaves every output read.
            failed = adapt::advance(model, tau, &mut t, drift.as_mut(), plan.end, &mut each).is_err();
        }
    }
    // From the output it does not reach: failed, or lost -- judged at the state it was lost in, its loss energy's error as
    // it stood there.
    let gone = if failed {
        Output { fate: Fate::Failed, near: None, energy: None }
    } else {
        Output { fate: Fate::Lost, near: windows.get(reached).map(|w| margin(&t).abs() <= *w), energy: drift.as_ref().map(|d| d.energy) }
    };
    out[reached.min(outputs)..].fill(gone);
    for (k, o) in out.iter_mut().enumerate().skip(1) {
        if let Fate::Held(h) = &mut o.fate {
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

/// Trajectory `index` of a run: its initial state from the ensemble (ensemble.rs `Ensemble::draw`), run through the plan
/// (`run`), its margin the running potential's far value less its loss energy (model.rs `Model::lost`).
#[allow(clippy::too_many_arguments)]
fn trajectory(model: &Run, ensemble: &Ensemble, tau: f64, read: bool, windows: &[f64], plan: &Plan, keep: bool, index: u64) -> Record {
    let margin = |t: &Trajectory| {
        let far = model.model.asymptote(&model.model.ground_populations(t.cluster, &t.psi), &model.model.running());
        far - model.energy(t.cluster, &t.point(), &t.psi)
    };
    run(model, ensemble.draw(SEED, index), index, tau, read, windows, &margin, plan, keep)
}

/// Groups `first..end`, their trajectories run on every core in their numbers' order -- each thread takes the next number,
/// so the groups finish nearly in order and the first after about one trajectory's time --, each group's records added in
/// its members' order on this thread (`Shape::fold`, nothing depends on the threads), the groups returned in their order;
/// whenever the solve's next message is due (`Messages::due`), `progress` takes the groups finished so far in order and
/// sends their result, or with no group finished the last result goes out again (module documentation, # Checkpoints).
fn run_groups(first: u64, end: u64, shape: Shape, record: &(dyn Fn(u64) -> Record + Sync), messages: &mut Messages, progress: &mut dyn FnMut(&[&Part], &mut Messages)) -> Vec<Part> {
    let (start, stop) = (first * GROUP, end * GROUP);
    let count = (end - first) as usize;
    let next = AtomicU64::new(start);
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min((stop - start) as usize);
    let mut waiting: Vec<Vec<Option<Record>>> = (0..count).map(|_| (0..GROUP).map(|_| None).collect()).collect();
    let mut parts: Vec<Option<Part>> = (0..count).map(|_| None).collect();
    std::thread::scope(|scope| {
        let (sender, receiver) = mpsc::channel::<(u64, Record)>();
        for _ in 0..threads {
            let (sender, next) = (sender.clone(), &next);
            scope.spawn(move || loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= stop || sender.send((i, record(i))).is_err() {
                    break;
                }
            });
        }
        drop(sender);
        let mut done = 0;
        while done < count {
            match receiver.recv_timeout(messages.due()) {
                Ok((i, r)) => {
                    let (g, m) = (((i - start) / GROUP) as usize, ((i - start) % GROUP) as usize);
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
/// (populations absolute, T relative; T's unbounded where no pilot atom reads it) and what the model's falsifier read
/// clearly beyond it on the rows as shown, the held share, and T's scale per axis (µK) where T is held to its target; the
/// step's error read on the rows as shown and on T (relative; unbounded where not read), and the loss energy's window
/// (rad/µs; module documentation, # The step; none before the pilot is read); and T(0)'s relative error, its
/// quadrature's last two levels' difference (ensemble.rs `Ensemble::kinetic`).
#[derive(Clone)]
struct Fixed {
    model: Vec<[f64; 2]>,
    cleared: Vec<f64>,
    held: Vec<f64>,
    temperature: Vec<Option<V3>>,
    step: Vec<[f64; 2]>,
    windows: Vec<f64>,
    start: f64,
}

/// The step's share of the targets on what is shown and on T (relative), s·ε (SPEC §2.2, §4): what its error may take,
/// and what the statistics' room leaves it.
fn step_bound() -> [f64; 2] {
    [STEP_SHARE * POPULATION_TARGET, STEP_SHARE * TEMPERATURE_TARGET]
}

/// The statistics' targets of the run (module documentation, # Stopping): per output the room of the shown rows' and the
/// loss's interval, and where T is held to its target its scale and the room of its relative interval.
struct Targets {
    population: Vec<f64>,
    temperature: Vec<Option<(V3, f64)>>,
    held: Vec<f64>,
}
impl Targets {
    fn new(fixed: &Fixed) -> Targets {
        let [bp, bt] = step_bound();
        let room = |eps: f64, a: f64, b: f64| (eps * eps - a * a - b * b).max(0.).sqrt();
        Targets {
            population: fixed.model.iter().zip(&fixed.held).zip(&fixed.cleared).map(|((a, h), c)| room(POPULATION_TARGET, a[0] * h + c, bp)).collect(),
            temperature: fixed
                .temperature
                .iter()
                .zip(&fixed.model)
                .zip(&fixed.held)
                .map(|((t, a), h)| t.map(|scale| (scale, room(TEMPERATURE_TARGET / h, a[1], bt))))
                .collect(),
            held: fixed.held.clone(),
        }
    }
    /// Whether output k's statistics meet their targets.
    fn met(&self, estimates: &[Estimate], k: usize) -> bool {
        let e = &estimates[k];
        e.shown_statistics() <= self.population[k] && self.temperature[k].is_none_or(|(scale, room)| e.temperature_interval(scale) <= room)
    }
    /// The trajectories each output needs by the estimates of N (module documentation, # Stopping), their largest.
    fn need(&self, estimates: &[Estimate]) -> f64 {
        let z2 = CONFIDENCE * CONFIDENCE;
        estimates
            .iter()
            .enumerate()
            .skip(1)
            .map(|(k, e)| {
                let n = e.trajectories;
                let room = self.population[k];
                let shown = (n * (e.shown_interval() / room).powi(2)).max(z2 * (1. / room - 1.));
                let temperature = match self.temperature[k] {
                    Some((scale, room)) => {
                        let relative = e.temperature_interval(scale);
                        if relative.is_finite() {
                            n * (relative / room).powi(2)
                        } else {
                            // Unread: the normal theory of a variance over N h atoms, relative interval z √(2/(N h)).
                            2. * z2 / (self.held[k] * room * room)
                        }
                    }
                    None => 0.,
                };
                shown.max(temperature)
            })
            .fold(0., f64::max)
    }
}

/// Each output's accounts of what is shown and of T (ledger.rs `Account`) by the run's estimates, the model's errors on the
/// held atoms per output and the step's (`Fixed`), T's where read: the held share 1 − L weighs the model's error on the rows as shown, and
/// T's whole account (module documentation, # Stopping); t = 0 is the request's state and the ensemble's moments, T's
/// by quadrature (`Fixed::start`). The model's error on what is shown is at most 1, a probability's. The step's error on
/// what is shown is the larger of the rows' and the loss's -- the share of the judged trajectories within the loss
/// energy's window of the escape (module documentation, # The step), the step's share where none is judged yet (a
/// pilot's result). T has an account where its statistics, its model error and its step error are all read.
fn accounts(estimates: &[Estimate], fixed: &Fixed) -> (Vec<Account>, Vec<Option<Account>>) {
    estimates
        .iter()
        .zip(&fixed.model)
        .zip(&fixed.step)
        .enumerate()
        .map(|(k, ((e, a), &[bp, bt]))| {
            if k == 0 {
                return (Account::default(), Some(Account { step: fixed.start, ..Account::default() }));
            }
            let h = 1. - e.loss;
            let escape = if e.judged > 0. { e.near / e.judged } else { step_bound()[0] };
            let population = Account { model: (a[0] * h + fixed.cleared[k]).min(1.), step: bp.max(escape), statistics: e.shown_statistics() };
            let temperature = e.motion.and_then(|m| {
                let relative = (0..3).map(|x| m.temperature_errors[x].interval() / m.temperature[x]).fold(0., f64::max);
                (relative.is_finite() && a[1].is_finite() && bt.is_finite()).then_some(Account { model: h * a[1], step: h * bt, statistics: h * relative })
            });
            (population, temperature)
        })
        .unzip()
}

/// A falsifier's looks (SPEC §5.2, §5.3; decide.rs): a coarse and a finer level on the same trajectories, read on every
/// row as shown, the loss and T where held to its target (relative to its scale) -- per look its output, whether it is
/// T's, and the coarse level's own error from the difference at the finer level's parameter `ratio` and the level's
/// `order` (`Read::level`).
fn looks(coarse: &[&Part], fine: &[Part], scales: &Scales, targets: &Targets, ratio: f64, order: f64) -> Vec<(usize, bool, Read)> {
    let reads = paired(coarse, &fine.iter().collect::<Vec<_>>(), scales);
    let mut looks = vec![];
    for (k, p) in reads.iter().enumerate().skip(1) {
        looks.extend(p.shown.iter().chain([&p.loss]).map(|r| (k, false, r.level(ratio, order))));
        if let (Some((scale, _)), Some(t)) = (targets.temperature[k], p.temperature) {
            for a in 0..3 {
                let r = t[a].level(ratio, order);
                looks.push((k, true, Read { value: r.value / scale[a], spread: Spread { error: r.spread.error / scale[a], groups: r.spread.groups } }));
            }
        }
    }
    looks
}

/// A falsifier's refutation: the family of its looks, each against what the coarse level may take there (`share`: per
/// output, on what is shown and on T), judged as every family is (decide.rs `judge`). Beyond: the factor on the coarse
/// level's parameter at which the largest measured error meets its share; else none -- a falsifier's read only refutes.
fn refuted(coarse: &[&Part], fine: &[Part], scales: &Scales, targets: &Targets, ratio: f64, order: f64, share: &dyn Fn(usize) -> [f64; 2]) -> Option<f64> {
    let reads: Vec<(Read, f64)> = looks(coarse, fine, scales, targets, ratio, order).into_iter().map(|(k, t, r)| (r, share(k)[usize::from(t)])).collect();
    match judge(&reads, order) {
        Judgement::Beyond(factor) => Some(factor),
        Judgement::Within | Judgement::Unreadable => None,
    }
}

/// What a falsifier's looks read clearly of the coarse level's error, the finer level taken as the exact one: per output,
/// on what is shown and on T, the largest of what its looks show clearly among them all (decide.rs `Read::clear`, the
/// family's interval as for a read beyond), none below zero.
fn cleared(coarse: &[&Part], fine: &[Part], scales: &Scales, targets: &Targets) -> Vec<[f64; 2]> {
    let looks = looks(coarse, fine, scales, targets, 0., 1.);
    let m = looks.len() as f64;
    let mut clear = vec![[0f64; 2]; targets.held.len()];
    for (k, t, r) in &looks {
        let c = &mut clear[*k][usize::from(*t)];
        *c = c.max(r.clear(m));
    }
    clear
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

/// The counted work (decide.rs `Work`) a trajectory spends per µs at a pilot state under a model, the state on the model's
/// `cluster`: one step's -- the integrator's (adapt.rs `advance_work`, through the model's generator at the state's
/// point, its velocity and τ) and its assemblies' (`assembly`, the model's count of one) -- over its τ.
fn rate(run: &Run, assembly: f64, cluster: usize, state: &State) -> f64 {
    let at = state.sample.at;
    let generator = run.generator(cluster, &at);
    let velocity = at.kappa.map(|k| run.hbar_over_mass() * k);
    let (work, assemblies) = adapt::advance_work(&generator, &run.cluster(cluster).waves, velocity, state.tau);
    (work + assemblies as f64 * assembly) / state.tau
}

/// A unit to keep (`candidates`): its predicted saving, and the kept model's work per µs over the current one's.
struct Candidate {
    unit: usize,
    saving: Read,
    ratio: f64,
}

/// The candidates to keep by pays (SPEC §5.2; module documentation, # The model), best margin first: for every folded unit
/// the predicted saving of total work W = w·N by keeping it, as a share of the current model's -- N the trajectories the
/// statistics need at the model's errors (`Targets::need` on the pilot's estimates), and for the kept model the pilot its
/// new attempt runs as well (the current one's is spent); w the mean of `rate` --, spread by the jackknife over the
/// pilot's groups (each recomputed without one: its estimates, the units' errors, the rates). A share, not the work saved:
/// both models' N are the one variance the pilot estimates over their rooms (N ∝ σ̂²), and σ̂²'s spread, common to both,
/// scales the work saved without ever changing its sign; in the share it cancels, and what spreads is what the keep
/// changes -- the rooms and the rates. Every unit the model can keep is priced, a keep costing less per µs than the fold
/// it replaces or far more (a drive kept across a fine-structure splitting, its generator's norm ~10⁷ rad/µs): at one state
/// of each pilot trajectory -- the trajectory is the pilot's independent unit, its outputs taken in turn by its number --,
/// the same states for every model.
#[allow(clippy::too_many_arguments)]
fn candidates(
    request: &Request,
    light: &Light,
    kept: &[Unit],
    model: &Model,
    reference: [f64; GROUND],
    units: &[Unit],
    measured: &Measured,
    states: &[State],
    pilot: &[&Part],
    fixed: &Fixed,
    scales: &Scales,
    unsampled: f64,
) -> Vec<Candidate> {
    let outputs = fixed.model.len();
    let groups = pilot.len();
    // Per level -- every group (None), then each left out -- the pilot's estimates and the units' mean errors.
    let levels: Vec<Option<usize>> = std::iter::once(None).chain((0..groups).map(Some)).collect();
    let data: Vec<(Vec<Estimate>, Vec<Vec<[f64; 2]>>)> = levels
        .iter()
        .map(|&left| {
            let parts: Vec<&Part> = pilot.iter().enumerate().filter(|(g, _)| Some(*g) != left).map(|(_, p)| *p).collect();
            (pooled(&parts, scales).0, measured.means(outputs, left))
        })
        .collect();
    let priced: Vec<usize> = (0..states.len()).filter(|&j| measured.output[j] as u64 == 1 + states[j].index % (outputs as u64 - 1)).collect();
    let mean = |rates: &[f64], left: Option<usize>| {
        let (sum, count) = priced.iter().zip(rates).filter(|(&j, _)| Some(measured.group[j]) != left).fold((0., 0.), |a, (_, r)| (a.0 + r, a.1 + 1.));
        sum / count
    };
    // A model's W at a level (per µs of run): its mean rate times its product's trajectories and `pilot` more.
    let work = |level: usize, without: Option<usize>, rates: &[f64], pilot: f64| {
        let (estimates, means) = &data[level];
        let f = Fixed { model: model_error(means, without, unsampled), ..fixed.clone() };
        mean(rates, levels[level]) * (Targets::new(&f).need(estimates) + pilot)
    };
    let again = (groups as u64 * GROUP) as f64;
    let current = Run::new(model, reference);
    let assembly = current.assembly_work();
    let own: Vec<f64> = priced.par_iter().map(|&j| rate(&current, assembly, states[j].sample.cluster, &states[j])).collect();
    let base: Vec<f64> = (0..levels.len()).map(|i| work(i, None, &own, 0.)).collect();
    let mut out: Vec<Candidate> = (0..units.len())
        .into_par_iter()
        .filter_map(|u| {
            // A unit the model refuses to keep (its fold must exist, `Model::new`) is no candidate: its fold's error stands.
            let with = Model::new(request, light, &[kept, std::slice::from_ref(&units[u])].concat()).ok()?;
            let finer = Run::new(&with, reference);
            let assembly = finer.assembly_work();
            let rates: Vec<f64> = priced
                .par_iter()
                .map(|&j| rate(&finer, assembly, finer.cluster_of(model.clusters()[states[j].sample.cluster].states[0]), &states[j]))
                .collect();
            let x: Vec<f64> = (0..levels.len()).map(|i| 1. - work(i, Some(u), &rates, again) / base[i]).collect();
            let saving = Read { value: x[0], spread: spread(x[1..].iter().copied(), groups as f64) };
            Some(Candidate { unit: u, saving, ratio: mean(&rates, None) / mean(&own, None) })
        })
        .collect();
    out.sort_by(|a, b| (b.saving.value - b.saving.spread.interval()).total_cmp(&(a.saving.value - a.saving.spread.interval())));
    out
}

/// Histograms of one coordinate of the held atoms at the listed outputs on one grid symmetric about 0 that holds them all
/// (the bins' centres; an odd count about √n, n the most atoms at a listed output, at least three): a listed output with no
/// atom carries the one before it.
fn histograms(listed: &[Vec<(V3, V3)>], outputs: &[usize], value: &dyn Fn(&(V3, V3)) -> f64) -> Distributions {
    let reach = listed.iter().flatten().map(|s| value(s).abs()).fold(f64::MIN_POSITIVE, f64::max);
    let most = listed.iter().map(Vec::len).max().unwrap_or(0);
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
/// diagonal, the exact moments at t = 0 ((ħ/m)⟨κ_a²⟩ per lab axis, rad/µs), the pilot's fixed errors, the ledger's items
/// and note, and the method's description.
struct Context {
    plan: Plan,
    tau: f64,
    scales: Scales,
    initial: [f64; STATES],
    momentum: V3,
    fixed: Fixed,
    ledger: Ledger,
    method: String,
}

/// The result of the estimates of `trajectories` trajectories (contract §2.2): at t = 0 the request's initial state and the
/// ensemble's exact moments; an output with no atom held carries the last one's populations, one with fewer than two
/// atoms the last T read (its error unknown); each flagged.
fn assemble(c: &Context, estimates: &[Estimate], listed: &[Vec<(V3, V3)>], counts: &Counts, trajectories: u64) -> Solution {
    let (times, outputs) = (&c.plan.times, c.plan.times.len());
    let hm = c.scales.hbar_over_mass;
    let w = c.scales.frequencies;
    let classical = |t: &V3| classical(t, &w);
    let mut ledger = c.ledger.clone();
    let (population, temperature) = accounts(estimates, &c.fixed);
    ledger.close(times, &population, &temperature, &c.fixed.temperature.iter().map(Option::is_some).collect::<Vec<_>>());
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
        trap_frequency_khz: w.map(|x| x / std::f64::consts::TAU * 1e3),
        loss_probability: estimates.iter().enumerate().map(|(k, e)| if k == 0 { 0. } else { e.loss }).collect(),
        velocity: std::array::from_fn(|a| histograms(listed, &c.plan.listed, &|x| x.1[a] * hm)),
        position: std::array::from_fn(|a| histograms(listed, &c.plan.listed, &|x| x.0[a])),
        classical: Vec::with_capacity(outputs),
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
            s.mean_occupation.push(std::array::from_fn(|a| c.momentum[a] / w[a] - 0.5));
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
                s.mean_velocity_mps.push(m.mean_velocity.map(|v| v));
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
                s.classical.push(false);
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
    // T shown with its statistics where the pilot left its model error or its step error unread.
    let unbooked: Vec<usize> = (1..outputs)
        .filter(|&k| estimates[k].motion.as_ref().is_some_and(|m| m.temperature_errors.iter().all(Spread::read)) && !(c.fixed.model[k][1].is_finite() && c.fixed.step[k][1].is_finite()))
        .collect();
    if !unbooked.is_empty() {
        s.approximations.flag(format!("{}：试探在这些时刻被俘获的原子不足，模型与步长对温度的误差读不出", span(&unbooked, times)));
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
    /// population target: z²(1/(√(1 − s²) ε) − 1) = 85.4 at z = 2, s² = 1/5, ε = 0.05, eleven groups of eight.
    #[test]
    fn the_pilot_sees_the_target() {
        assert_eq!(pilot_size(), 88);
        let room = (1. - STEP_SHARE * STEP_SHARE).sqrt() * POPULATION_TARGET;
        assert!(unseen(88.) <= room && unseen(80.) > room);
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

    /// A rung's outcomes on T's read at one output (unlimited groups: one look's interval is z of the error): within
    /// where the error and its interval fit the share; an error four times the share, clear among all the looks, takes
    /// the step to its half (second order); a small error under a wide interval is unreadable; T that is not held to
    /// its target is not judged, and neither is the output at t = 0.
    #[test]
    fn a_rung_has_three_outcomes() {
        let read = |value: f64, error: f64| Read { value, spread: Spread { error, groups: f64::INFINITY } };
        let quiet = read(0., 0.);
        let at = |t: Read| Stepped { temperature: Some([t, quiet, quiet]), shown: [quiet; STATES], energy: None };
        let held = [None, Some([30.; 3])];
        let st = step_bound()[1];
        assert_eq!(rung(&[at(quiet), at(read(0.5 * st, 0.1 * st))], &held), Judgement::Within);
        match rung(&[at(quiet), at(read(4. * st, 0.01 * st))], &held) {
            Judgement::Beyond(factor) => assert!((factor - 0.5).abs() < 1e-12, "{factor}"),
            other => panic!("an error four times the share is beyond it: {other:?}"),
        }
        assert_eq!(rung(&[at(quiet), at(read(0.5 * st, st))], &held), Judgement::Unreadable);
        assert_eq!(rung(&[at(quiet), at(read(9. * st, 0.))], &[None, None]), Judgement::Within);
        assert_eq!(rung(&[at(read(9. * st, 0.)), at(quiet)], &[Some([30.; 3]), Some([30.; 3])]), Judgement::Within);
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

    /// The groups come back in their order whatever the threads do: 37 groups' records -- each trajectory's own, its rows
    /// marking its number -- folded by `run_groups` equal the same records folded one after another.
    #[test]
    fn the_groups_come_back_in_order() {
        let shape = Shape { outputs: 2, listed: 1, keep: false };
        let record = |i: u64| {
            let mut rows = [0.; STATES];
            rows[(i % STATES as u64) as usize] = 1.;
            let held = Held { rows, kappa: [i as f64; 3], kappa2: [(i * i) as f64; 3], drift: None };
            let at = |fate: Fate| Output { fate, near: None, energy: None };
            let outputs = vec![at(Fate::Held(held.clone())), at(if i % 3 > 0 { Fate::Held(held) } else { Fate::Lost })];
            Record { outputs, listed: vec![Some(([i as f64; 3], [0.; 3]))], samples: vec![], counts: Counts { steps: i, ..Counts::default() } }
        };
        let mut silent = |_: &Solution, _: Option<f64>| {};
        let mut messages = Messages { send: &mut silent, clock: Instant::now(), last: None, least: 0 };
        let parts = run_groups(5, 42, shape, &record, &mut messages, &mut |_, _| {});
        assert_eq!(parts.len(), 37);
        for (g, part) in parts.iter().enumerate() {
            let one = shape.fold((0..GROUP).map(|m| record((5 + g as u64) * GROUP + m)));
            assert_eq!(part.sums, one.sums);
            assert_eq!(part.listed, one.listed);
            assert_eq!(part.counts.steps, one.counts.steps);
        }
    }

    /// The initial ensembles of the conservative reference traps (refs/scenes R1, the app's own requests: a lone tweezer, a
    /// standing wave, crossed beams of one frequency, a σ⁺ tweezer) in the model's prepared wells under the model's proven
    /// envelopes: 2¹² draws each -- every one within its envelope (the draw's debug check) -- read the exact moments
    /// (ħ/m)⟨κ_a²⟩ = Σ_b p_b η_b k_BT (S_b²)_aa per lab axis within four of their standard errors, η from the quadrature on
    /// a pilot's budget of evaluations (2¹⁶), its last two levels within a tenth of the draws' relative error.
    #[test]
    fn the_reference_traps_draw_their_exact_moments() {
        for name in ["R1_TWEEZER", "R1_RETRO", "R1_CROSSED", "R1_SIGMA_P"] {
            let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("scenes").join(format!("{name}.json"));
            let line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
            let request = Request::from_native(&line).unwrap();
            let light = Light::new(&request);
            let model = Model::new(&request, &light, &[]).unwrap();
            let initial = components(&request).unwrap();
            let wells: Vec<Prepared> = initial.iter().map(|&(s, _)| Prepared::new(&model, &light, alone(s)).unwrap()).collect();
            let hm = hbar_over_mass(request.mass_kg);
            let ensemble = Ensemble::new(&initial.iter().zip(&wells).map(|(&(s, p), w)| (s, p, w as &dyn Well)).collect::<Vec<_>>(), request.temperature_uk, hm).unwrap();
            let kinetic = ensemble.kinetic(1 << 16);
            assert!(kinetic.iter().all(|k| k.eta > 0. && k.eta <= 1. && k.difference < 1.5e-3 * k.eta), "{name}: η {kinetic:?}");
            let exact = ensemble.momentum(&kinetic.iter().map(|k| k.eta).collect::<Vec<_>>());
            let draws: Vec<V3> = (0..1 << 12).map(|i| ensemble.draw(SEED, i).kappa.map(|k| hm * k * k)).collect();
            let n = draws.len() as f64;
            for a in 0..3 {
                let mean = draws.iter().map(|d| d[a]).sum::<f64>() / n;
                let se = (draws.iter().map(|d| (d[a] - mean).powi(2)).sum::<f64>() / (n - 1.) / n).sqrt();
                assert!((mean - exact[a]).abs() <= 4. * se, "{name} axis {a}: {mean} against {} ± {se}", exact[a]);
            }
        }
    }

    /// A falsifier refutes only a clear excess (`refuted`): two levels on the same 16 groups of 8 atoms, the finer one moving
    /// one state's row by δ at one output for every atom alike (the pair's difference has no spread). At the step's level
    /// (half the step, order 2) the coarse level's error is 4δ/3: δ = 0.3 against a share of 0.05 is refuted with the
    /// factor √(0.05/0.4) on the step; δ = 0.01 is within. What the pairs read clearly (`cleared`) is δ itself at that
    /// output.
    #[test]
    fn a_falsifier_refutes_only_a_clear_excess() {
        let outputs = 3;
        let level = |shift: f64| -> Vec<Part> {
            (0..16)
                .map(|g| {
                    let mut sums = Group::new(outputs);
                    for m in 0..8 {
                        let x = ((g * 8 + m) as f64 * 0.37).fract();
                        let held: Vec<Output> = (0..outputs)
                            .map(|k| {
                                let mut rows = [0.; STATES];
                                let d = if k == 2 { shift } else { 0. };
                                (rows[0], rows[1]) = (0.5 + 0.1 * x + d, 0.5 - 0.1 * x - d);
                                Output { fate: Fate::Held(Held { rows, kappa: [x; 3], kappa2: [1. + x * x; 3], drift: None }), near: None, energy: None }
                            })
                            .collect();
                        sums.add(&held);
                    }
                    Part { sums, listed: vec![], samples: vec![], counts: Counts::default() }
                })
                .collect()
        };
        let coarse = level(0.);
        let coarse: Vec<&Part> = coarse.iter().collect();
        let targets = Targets { population: vec![1.; outputs], temperature: vec![None; outputs], held: vec![1.; outputs] };
        let scales = Scales { hbar_over_mass: KT_PER_UK, frequencies: [1.; 3] };
        let factor = refuted(&coarse, &level(-0.3), &scales, &targets, 0.5, 2., &|_| [0.05, 0.1]).unwrap();
        assert!((factor - (0.05_f64 / 0.4).sqrt()).abs() < 1e-12, "{factor}");
        assert!(refuted(&coarse, &level(-0.01), &scales, &targets, 0.5, 2., &|_| [0.05, 0.1]).is_none());
        // What the pairs read clearly, the finer level exact: the whole shift where it is (no spread), nothing elsewhere.
        let clear = cleared(&coarse, &level(-0.3), &scales, &targets);
        assert!((clear[2][0] - 0.3).abs() < 1e-12 && clear[1] == [0.; 2] && clear[2][1] == 0., "{clear:?}");
    }
}
