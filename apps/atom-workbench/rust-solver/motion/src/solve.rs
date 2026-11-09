//! The pipeline: request, reach and model, pilot, decisions, batches, checkpoints, result (SPEC §3, §5, §9).
//!
//! # The pipeline
//! 1. The request's checks; the light; the initial components; the reach and the model with every unit whose existence
//!    gate fails kept (model.rs, SPEC §1.5).
//! 2. The pilot (`pilot_size` trajectories, numbered from `MOST_TRAJECTORIES` so that none of its streams is a run's):
//!    it predicts only (decisions.md A4) -- the model's units' errors measured at its states, the outputs where T is held
//!    to its target and T's scale there, the held share, the step's scales. Its result is the first checkpoint.
//! 3. The model's decisions (SPEC §5.2) and the step's tolerances (SPEC §2.2), from the pilot alone.
//! 4. The product: batches of whole groups, each trajectory with its own adaptive steps (adapt.rs), every one run to the
//!    last output, until every output's statistics meet the targets the pilot fixed (`Targets`), or the trajectories
//!    reach `MOST_TRAJECTORIES`.
//! 5. The step's falsifier (SPEC §5.3): the product's first `pilot_size` trajectories again at a quarter of the tolerance
//!    on the same streams; an output read clearly beyond the step's share tightens the tolerance by the measured ratio and
//!    the product runs again.
//! 6. The result (contract §2.2, `assemble`).
//!
//! # Outputs and windows
//! The outputs are the request's times, from 0 and evenly spaced. T's window about an output (estimate.rs `half_window`)
//! takes a trajectory's κ and κ² over its steps' paths (step.rs `Segment::moments`, exact on the path); two outputs'
//! windows never overlap (each at most half an interval on either side), the first is the instant t = 0, and the run goes
//! on through the last output's window (`Plan`).
//!
//! # The model (SPEC §5.2)
//! The units are the model's (model.rs), listed again in every model a keep closes. A unit whose fold does not exist over
//! the reach is kept, and so is one whose measured error leaves an output no room for the statistics; every keep is a new
//! attempt -- a new model, its own pilot.
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
//! times the checkpoints: one after the pilot, then whenever `CHECKPOINT` has passed, each a complete result of the groups
//! finished so far in their order (a prefix: whole groups chosen by their number alone).
use crate::adapt::{self, Tolerances};
use crate::decide::{Outcome, Read};
use crate::ensemble::{Ensemble, Start, Well, GROUP};
use crate::estimate::{half_window, paired, summarize, unseen, Estimate, Group, Held, Scales, Spread};
use crate::field::{Geometry, Light};
use crate::ledger::{percent, span, when, Account, Ledger};
use crate::linalg::V3;
use crate::model::{Envelope, Lights, Model, Run, Sample, Trap, Unit, UnitError, GROUND};
use crate::request::STATES;
use crate::step::{Dynamics, Segment, Trajectory};
use crate::units::{hbar_over_mass, CLASSICAL, CONFIDENCE, KT_PER_UK, MOST_STEPS, MOST_TRAJECTORIES, STEP_SHARE, TEMPERATURE_TARGET};
use std::sync::atomic::{AtomicU64, Ordering};
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
    /// The mean accepted step over the trajectories (µs): each trajectory takes its own (adapt.rs).
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
    let (mut kept, mut reach) = (vec![], request.temperature_uk);
    loop {
        match attempt(request, &light, &kept, reach, checkpoint)? {
            Attempt::Grow(t) => reach = t,
            Attempt::Keep(u) => kept.push(u),
            Attempt::Done(s) => return Ok(s),
        }
    }
}

/// How one attempt at a model ends (`attempt`): the pilot's atoms outgrew the reach the model was judged over (its new
/// temperature, µK), a unit is to be kept, or the solution.
enum Attempt {
    Grow(f64),
    Keep(Unit),
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

/// The step's tolerances per µs of the run (SPEC §2.2: each output's error ≤ s·ε·τ/T_run of its scale, adapt.rs), from the
/// temperature scales (µK) per lab axis and of the energy: the momentum's against T's target (T ∝ κ², so κ's error δ moves
/// T by 2δ/κ_T relative: δ ≤ s ε_T κ_T/2, κ_T² = k_BT/(ħ/m)); the loss energy's against the population target (the loss
/// moves by the share of atoms within δE of the escape, at most δE/k_BT: δE ≤ s ε k_BT; T moves by δE/k_BT, within it);
/// the displayed rows' against the population target.
fn tolerances(momentum: V3, energy: f64, run: f64, hbar_over_mass: f64) -> Tolerances {
    let [bp, bt] = step_bound();
    Tolerances {
        momentum: momentum.map(|t| bt * (KT_PER_UK * t / hbar_over_mass).sqrt() / (2. * run)),
        energy: bp * KT_PER_UK * energy / run,
        population: bp / run,
    }
}
/// Tolerances tightened (or loosened) by a factor.
fn scaled(t: &Tolerances, factor: f64) -> Tolerances {
    Tolerances { momentum: t.momentum.map(|x| x * factor), energy: t.energy * factor, population: t.population * factor }
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

/// The pilot's T where it has outgrown the reach's temperature `reach` (µK, SPEC §1.8): some output's T above it beyond its
/// interval among all the T read (Šidák over the outputs and axes); then the largest T read, the reach's new temperature.
fn outgrown(estimates: &[Estimate], reach: f64) -> Option<f64> {
    let read: Vec<&crate::estimate::Motion> = estimates.iter().skip(1).filter_map(|e| e.motion.as_ref()).collect();
    let looks = 3. * read.len() as f64;
    let beyond = read.iter().any(|m| (0..3).any(|a| m.temperature[a] - m.temperature_errors[a].interval_over(looks) > reach));
    beyond.then(|| read.iter().flat_map(|m| m.temperature).fold(reach, f64::max))
}

/// One attempt at the solve with the units `kept` and the reach at `reach` µK (module documentation, # The pipeline).
fn attempt(request: &Request, light: &Light, kept: &[Unit], reach: f64, checkpoint: &mut dyn FnMut(&Solution, Option<f64>)) -> Result<Attempt, String> {
    let hm = hbar_over_mass(request.mass_kg);
    let initial = components(request)?;
    let reference = alone(initial.iter().max_by(|a, b| a.1.total_cmp(&b.1)).ok_or("运动计算的初态需在基态上")?.0);
    // The model: every unit whose fold does not exist over the reach kept (SPEC §1.5), closed again until none fails that is
    // not kept already (a unit kept is moot whether a later model lists it or not).
    let mut kept = kept.to_vec();
    let (model, trap) = loop {
        let model = Model::new(request, light, &kept)?;
        let trap = model.trap(&reference, &model.prepared())?;
        let judged = model.reach(&trap, &reference, reach, unseen(MOST_TRAJECTORIES as f64));
        let units = model.units();
        let failing: Vec<Unit> =
            units.iter().zip(model.gates(&units, &judged)).filter(|(u, g)| !g.exists() && !kept.contains(u)).map(|(u, _)| u.clone()).collect();
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
    let exact = |kinetic: Vec<(f64, f64)>| -> (V3, f64) {
        (ensemble.momentum(&kinetic.iter().map(|k| k.0).collect::<Vec<_>>()), kinetic.iter().map(|k| k.1 / k.0).fold(0., f64::max))
    };
    let (momentum, start) = exact(ensemble.kinetic(0));
    let mut context = Context {
        momentum,
        fixed: Fixed { model: vec![[0.; 2]; outputs], held: vec![1.; outputs], temperature: vec![None; outputs], step: step_bound(), start },
        plan,
        scales,
        initial: row,
        ledger,
        method: format!("半经典 3D：{}", model.describe().trim_end_matches('。')),
    };

    // The pilot (module documentation, # The pipeline): its own trajectory numbers, the step at the request's temperature.
    let pilot_tolerances = tolerances([request.temperature_uk; 3], request.temperature_uk, context.plan.end, hm);
    let first = MOST_TRAJECTORIES / GROUP;
    let pilot = run_groups(first, first + pilot_size() / GROUP, Shape::of(&context.plan, true), &|i| trajectory(&run, &ensemble, &pilot_tolerances, &context.plan, true, i), &mut |prefix| {
        let (e, l, c) = pooled(prefix, &context.scales);
        checkpoint(&assemble(&context, &e, &l, &c, prefix.len() as u64 * GROUP), None);
    });
    let pilot: Vec<&Part> = pilot.iter().collect();
    let (estimates, listed, counts) = pooled(&pilot, &context.scales);
    if let Some(t) = outgrown(&estimates, reach) {
        return Ok(Attempt::Grow(t));
    }
    (context.momentum, context.fixed.start) = exact(ensemble.kinetic(counts.steps));

    // The model's errors at the pilot's states (SPEC §5.2): each folded unit's per output, the held atoms' mean.
    let units = model.units();
    let mut samples: Vec<Sample> = vec![];
    let mut at = vec![];
    for p in &pilot {
        for (k, s) in p.samples.iter().enumerate() {
            samples.extend(s.iter().cloned());
            at.extend(std::iter::repeat_n(k, s.len()));
        }
    }
    let errors = model.errors(&units, &samples);
    let per_output = |f: &dyn Fn(&UnitError) -> f64| -> Vec<Vec<f64>> {
        errors
            .iter()
            .map(|unit| {
                let (mut sum, mut count) = (vec![0.; outputs], vec![0.; outputs]);
                for (e, &k) in unit.iter().zip(&at) {
                    sum[k] += f(e);
                    count[k] += 1.;
                }
                sum.iter().zip(&count).map(|(s, c)| if *c > 0. { s / c } else { 0. }).collect()
            })
            .collect()
    };
    let (population, temperature) = (per_output(&|e| e.population), per_output(&|e| e.temperature));
    let unsampled = unseen(MOST_TRAJECTORIES as f64);
    context.fixed.model = (0..outputs)
        .map(|k| if k == 0 { [0.; 2] } else { [population.iter().map(|u| u[k]).sum::<f64>() + unsampled, temperature.iter().map(|u| u[k]).sum::<f64>() + unsampled] })
        .collect();
    context.fixed.held = estimates.iter().map(|e| 1. - e.loss).collect();
    context.fixed.temperature = estimates
        .iter()
        .enumerate()
        .map(|(k, e)| {
            let m = e.motion.as_ref().filter(|_| k > 0)?;
            let upper: V3 = std::array::from_fn(|a| m.temperature[a] + m.temperature_errors[a].interval());
            (m.temperature.iter().all(|&t| t > 0.) && (0..3).all(|a| KT_PER_UK * upper[a] >= CLASSICAL * report.lab[a])).then_some(m.temperature)
        })
        .collect();
    let targets = Targets::new(&context.fixed);
    // A unit whose error leaves an output no room for the statistics is kept (SPEC §5.2): of the outputs left none, the unit
    // whose errors there, each relative to its target and held share, add up to the most.
    let crowded: Vec<usize> = (1..outputs).filter(|&k| targets.population[k] == 0. || targets.temperature[k].is_some_and(|t| t.1 == 0.)).collect();
    let weight = |u: usize| -> f64 {
        crowded.iter().map(|&k| context.fixed.held[k] * (population[u][k] / POPULATION_TARGET + temperature[u][k] / TEMPERATURE_TARGET)).sum()
    };
    if let Some(u) = (0..units.len()).max_by(|&a, &b| weight(a).total_cmp(&weight(b))).filter(|&u| weight(u) > 0.) {
        return Ok(Attempt::Keep(units[u].clone()));
    }
    // The ledger's items: each folded unit's error on what is shown and on T (its largest over the outputs, weighed by the
    // held share as the totals are), the unsampled share, the step.
    for (u, unit) in units.iter().enumerate() {
        let shown = (1..outputs).map(|k| (population[u][k] * context.fixed.held[k]).max(temperature[u][k] * context.fixed.held[k])).fold(0., f64::max);
        if shown > 0. {
            context.ledger.item(&model.label(unit), shown, "二阶消去（折叠），沿试探轨迹实测");
        }
    }
    context.ledger.item("折叠判定之外的未采样份额", unsampled, "计入模型误差");
    context.ledger.item("时间步长（逐轨迹自适应，步长减半检验）", context.fixed.step[0], "容差 s·ε；容差的 1/4 否证");
    checkpoint(&assemble(&context, &estimates, &listed, &counts, pilot_size()), None);

    // The product (module documentation, # Stopping), and the step's falsifier.
    let coldest = |f: &dyn Fn(&crate::estimate::Motion) -> f64| estimates.iter().skip(1).filter_map(|e| e.motion.as_ref()).map(f).filter(|t| *t > 0.).fold(f64::INFINITY, f64::min);
    let held_to: Vec<&V3> = context.fixed.temperature.iter().flatten().collect();
    let momentum: V3 = std::array::from_fn(|a| held_to.iter().map(|t| t[a]).fold(f64::INFINITY, f64::min));
    let momentum = momentum.map(|t| if t.is_finite() { t } else { request.temperature_uk });
    let energy = coldest(&|m| m.temperature_total).min(request.temperature_uk);
    let mut tolerance = tolerances(momentum, energy, context.plan.end, hm);
    let round = |n: f64| ((n / GROUP as f64).ceil() as u64 * GROUP).clamp(pilot_size(), MOST_TRAJECTORIES);
    let mut want = round(targets.need(&estimates));
    let mut parts: Vec<Part> = vec![];
    let started = Instant::now();
    loop {
        loop {
            let have = parts.len() as u64 * GROUP;
            let pace = |done: u64| (done > 0).then(|| started.elapsed().as_secs_f64() * 1e3 / done as f64);
            let new = run_groups(have / GROUP, want / GROUP, Shape::of(&context.plan, false), &|i| trajectory(&run, &ensemble, &tolerance, &context.plan, false, i), &mut |prefix| {
                let all: Vec<&Part> = parts.iter().chain(prefix.iter().copied()).collect();
                let n = all.len() as u64 * GROUP;
                let (e, l, c) = pooled(&all, &context.scales);
                checkpoint(&assemble(&context, &e, &l, &c, n), pace(n).map(|p| p * (want - n) as f64));
            });
            parts.extend(new);
            let all: Vec<&Part> = parts.iter().collect();
            let n = all.len() as u64 * GROUP;
            let (e, l, c) = pooled(&all, &context.scales);
            if (1..outputs).all(|k| targets.met(&e, k)) || n >= MOST_TRAJECTORIES {
                break;
            }
            want = round(targets.need(&e)).max(n + GROUP).min(MOST_TRAJECTORIES);
            checkpoint(&assemble(&context, &e, &l, &c, n), pace(n).map(|p| p * (want - n) as f64));
        }
        // The step's falsifier on the product's first groups; while it runs, the product's result is the checkpoint.
        let checked = pilot_size() / GROUP;
        let all: Vec<&Part> = parts.iter().collect();
        let (e, l, c) = pooled(&all, &context.scales);
        let current = assemble(&context, &e, &l, &c, all.len() as u64 * GROUP);
        let fine = run_groups(0, checked, Shape::of(&context.plan, false), &|i| trajectory(&run, &ensemble, &scaled(&tolerance, 0.25), &context.plan, false, i), &mut |_| checkpoint(&current, None));
        let coarse: Vec<&Part> = parts[..checked as usize].iter().collect();
        match refuted(&coarse, &fine, &context.scales, &targets, 0.25, 1., &|_| context.fixed.step) {
            Some(factor) if c.capped.is_none() => {
                tolerance = scaled(&tolerance, factor);
                want = parts.len() as u64 * GROUP;
                parts.clear();
            }
            Some(_) => {
                context.ledger.flag("步长否证（容差的 1/4）读到超出步长的份额，步数已达上限，无法再收紧".into());
                break;
            }
            None => break,
        }
    }
    let all: Vec<&Part> = parts.iter().collect();
    let (e, l, c) = pooled(&all, &context.scales);
    Ok(Attempt::Done(assemble(&context, &e, &l, &c, all.len() as u64 * GROUP)))
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

/// Counts over trajectories: the accepted steps, the integrator's counted work (decide.rs `Work`) and its generator
/// assemblies, the time held (µs), and the earliest time from which a trajectory's steps went unchecked (the step cap).
#[derive(Clone, Copy, Debug, Default)]
struct Counts {
    steps: u64,
    work: f64,
    assemblies: u64,
    held: f64,
    capped: Option<f64>,
}
impl Counts {
    fn add(&mut self, o: &Counts) {
        self.steps += o.steps;
        self.work += o.work;
        self.assemblies += o.assemblies;
        self.held += o.held;
        self.capped = match (self.capped, o.capped) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
    }
}

/// What one trajectory leaves: per output its atom where held (estimate.rs `Held`), its (r, κ) at the listed outputs
/// where held, its states at the outputs where held (`keep`: the pilot's, for the model's errors), its counts.
struct Record {
    held: Vec<Option<Held>>,
    listed: Vec<Option<(V3, V3)>>,
    samples: Vec<Option<Sample>>,
    counts: Counts,
}

/// One trajectory from its initial state through the plan (module documentation, # Outputs and windows).
fn run<D: Dynamics>(model: &D, start: Start, index: u64, tolerances: &Tolerances, plan: &Plan, keep: bool) -> Record {
    let outputs = plan.times.len();
    let interval = plan.times[1] - plan.times[0];
    let widest = plan.halves.iter().flatten().fold(0., |a: f64, &b| a.max(b));
    let mut t = Trajectory::new(model, start.state, start.r, start.kappa, SEED, index);
    // Per output and lab axis: the window's time held, ∫κ dt and ∫κ² dt.
    let mut windows = vec![[[0.; 3]; 3]; outputs];
    let mut rows: Vec<Option<[f64; STATES]>> = vec![None; outputs];
    let mut listed = vec![None; plan.listed.len()];
    let mut samples: Vec<Option<Sample>> = vec![None; if keep { outputs } else { 0 }];
    let mut steps = 0;
    let mut record = |k: usize, t: &Trajectory, rows: &mut Vec<Option<[f64; STATES]>>| {
        rows[k] = Some(model.readout(t.cluster, &t.point(), &t.psi));
        if let Ok(j) = plan.listed.binary_search(&k) {
            listed[j] = Some((t.r, t.kappa));
        }
        if keep {
            samples[k] = Some(Sample { cluster: t.cluster, at: t.point(), psi: t.psi.clone() });
        }
    };
    record(0, &t, &mut rows);
    let start_kappa = t.kappa;
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
                        let w = &mut windows[k][a];
                        w[0] += hi - lo;
                        w[1] += m1[a];
                        w[2] += m2[a];
                    }
                }
            }
        };
        for k in 1..outputs {
            adapt::advance(model, tolerances, &mut t, plan.times[k], &mut each);
            if t.lost.is_some() {
                break;
            }
            record(k, &t, &mut rows);
        }
        if t.lost.is_none() {
            adapt::advance(model, tolerances, &mut t, plan.end, &mut each);
        }
    }
    let held = rows
        .into_iter()
        .enumerate()
        .map(|(k, r)| {
            r.map(|rows| {
                if k == 0 {
                    Held { rows, kappa: start_kappa, kappa2: start_kappa.map(|x| x * x) }
                } else {
                    let w = &windows[k];
                    Held { rows, kappa: std::array::from_fn(|a| w[a][1] / w[a][0]), kappa2: std::array::from_fn(|a| w[a][2] / w[a][0]) }
                }
            })
        })
        .collect();
    let counts = Counts { steps, work: t.work, assemblies: t.assemblies, held: t.lost.unwrap_or(plan.end).min(plan.end), capped: t.capped };
    Record { held, listed, samples, counts }
}

/// What a group of trajectories leaves: its sums (estimate.rs `Group`), its held atoms' (r, κ) at the listed outputs, its
/// states at the outputs (the pilot's), its counts.
struct Part {
    sums: Group,
    listed: Vec<Vec<(V3, V3)>>,
    samples: Vec<Vec<Sample>>,
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
            part.sums.add(&r.held);
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

/// Trajectory `index` of a run: its initial state from the ensemble (ensemble.rs `Ensemble::draw`), run through the plan.
fn trajectory<D: Dynamics>(model: &D, ensemble: &Ensemble, tolerances: &Tolerances, plan: &Plan, keep: bool, index: u64) -> Record {
    run(model, ensemble.draw(SEED, index), index, tolerances, plan, keep)
}

/// Groups `first..end`, their trajectories run on every core in their numbers' order -- each thread takes the next number,
/// so the groups finish nearly in order and the first after about one trajectory's time --, each group's records added in
/// its members' order on this thread (`Shape::fold`, nothing depends on the threads), the groups returned in their order;
/// whenever `CHECKPOINT` has passed, `progress` takes the groups finished so far in order (module documentation,
/// # Checkpoints).
fn run_groups(first: u64, end: u64, shape: Shape, record: &(dyn Fn(u64) -> Record + Sync), progress: &mut dyn FnMut(&[&Part])) -> Vec<Part> {
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
        let (mut done, mut last) = (0, Instant::now());
        while done < count {
            match receiver.recv_timeout(CHECKPOINT.saturating_sub(last.elapsed())) {
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
                    progress(&prefix);
                    last = Instant::now();
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
    });
    parts.into_iter().map(|p| p.expect("a group's trajectories ended without their part")).collect()
}

/// What the pilot fixes for the run (module documentation, # Stopping), per output: the model's error on the held atoms
/// (populations absolute, T relative), the held share, and T's scale per axis (µK) where T is held to its target; the
/// step's error booked on what is shown and on T (`step_bound`); and T(0)'s relative error, its quadrature's last two
/// levels' difference (ensemble.rs `Ensemble::kinetic`).
#[derive(Clone)]
struct Fixed {
    model: Vec<[f64; 2]>,
    held: Vec<f64>,
    temperature: Vec<Option<V3>>,
    step: [f64; 2],
    start: f64,
}

/// The step's error booked on what is shown and on T (relative): the bound its tolerance gives, s·ε (SPEC §2.2, §4).
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
        let [bp, bt] = fixed.step;
        let room = |eps: f64, a: f64, b: f64| (eps * eps - a * a - b * b).max(0.).sqrt();
        Targets {
            population: fixed.model.iter().zip(&fixed.held).map(|(a, h)| room(POPULATION_TARGET, a[0] * h, bp)).collect(),
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
/// by quadrature (`Fixed::start`).
fn accounts(estimates: &[Estimate], fixed: &Fixed) -> (Vec<Account>, Vec<Option<Account>>) {
    let [bp, bt] = fixed.step;
    estimates
        .iter()
        .zip(&fixed.model)
        .enumerate()
        .map(|(k, (e, a))| {
            if k == 0 {
                return (Account::default(), Some(Account { step: fixed.start, ..Account::default() }));
            }
            let h = 1. - e.loss;
            let population = Account { model: a[0] * h, step: bp, statistics: e.shown_statistics() };
            let temperature = e.motion.and_then(|m| {
                let relative = (0..3).map(|x| m.temperature_errors[x].interval() / m.temperature[x]).fold(0., f64::max);
                relative.is_finite().then_some(Account { model: h * a[1], step: h * bt, statistics: h * relative })
            });
            (population, temperature)
        })
        .unzip()
}

/// A falsifier's refutation (SPEC §5.2, §5.3; decide.rs): a coarse and a finer level on the same trajectories, read on
/// every row as shown, the loss and T where held to its target (relative to its scale) against what is booked for the
/// coarse level there (`share`: per output, on what is shown and on T), the coarse level's own error from the difference at
/// the finer level's parameter `ratio` and the level's `order` (`Read::level`), Šidák over all the looks. Beyond somewhere:
/// the factor on the coarse level's parameter at which the largest measured error meets its share; else none.
fn refuted(coarse: &[&Part], fine: &[Part], scales: &Scales, targets: &Targets, ratio: f64, order: f64, share: &dyn Fn(usize) -> [f64; 2]) -> Option<f64> {
    let reads = paired(coarse, &fine.iter().collect::<Vec<_>>(), scales);
    let mut looks: Vec<(Read, f64)> = vec![];
    for (k, p) in reads.iter().enumerate().skip(1) {
        let [sp, st] = share(k);
        looks.extend(p.shown.iter().chain([&p.loss]).map(|r| (r.level(ratio, order), sp)));
        if let (Some((scale, _)), Some(t)) = (targets.temperature[k], p.temperature) {
            for a in 0..3 {
                let r = t[a].level(ratio, order);
                looks.push((Read { value: r.value / scale[a], spread: Spread { error: r.spread.error / scale[a], groups: r.spread.groups } }, st));
            }
        }
    }
    let m = looks.len() as f64;
    looks
        .iter()
        .filter(|(r, share)| r.outcome(*share, m) == Outcome::Beyond)
        .map(|(r, share)| r.refined(*share, order))
        .fold(None, |f: Option<f64>, x| Some(f.map_or(x, |f| f.min(x))))
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

/// What the result's assembly takes beside the estimates: the plan, the scales, the request's initial diagonal, the exact
/// moments at t = 0 ((ħ/m)⟨κ_a²⟩ per lab axis, rad/µs), the pilot's fixed errors, the ledger's items and note, and the
/// method's description.
struct Context {
    plan: Plan,
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
    let classical = |t: &V3| (0..3).all(|a| KT_PER_UK * t[a] >= CLASSICAL * w[a]);
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
        time_step_us: if counts.steps > 0 { counts.held / counts.steps as f64 } else { c.plan.end },
        method: format!("{}；{trajectories} 条轨迹（每组 {GROUP} 条拉丁超立方），逐轨迹自适应步长。", c.method),
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
    if let Some(t) = counts.capped {
        s.approximations.flag(format!("{} 起有轨迹的步数达到上限 {MOST_STEPS}：其后的步长未受检", when(t)));
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
            let held = Held { rows, kappa: [i as f64; 3], kappa2: [(i * i) as f64; 3] };
            Record { held: vec![Some(held.clone()), (i % 3 > 0).then_some(held)], listed: vec![Some(([i as f64; 3], [0.; 3]))], samples: vec![], counts: Counts { steps: i, ..Counts::default() } }
        };
        let parts = run_groups(5, 42, shape, &record, &mut |_| {});
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
            assert!(kinetic.iter().all(|k| k.0 > 0. && k.0 <= 1. && k.1 < 1.5e-3 * k.0), "{name}: η {kinetic:?}");
            let exact = ensemble.momentum(&kinetic.iter().map(|k| k.0).collect::<Vec<_>>());
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
    /// (a quarter of the tolerance, order 1) the coarse level's error is 4δ/3: δ = 0.3 against a share of 0.05 is refuted
    /// with the factor 0.05/0.4 on the tolerance; δ = 0.01 is within.
    #[test]
    fn a_falsifier_refutes_only_a_clear_excess() {
        let outputs = 3;
        let level = |shift: f64| -> Vec<Part> {
            (0..16)
                .map(|g| {
                    let mut sums = Group::new(outputs);
                    for m in 0..8 {
                        let x = ((g * 8 + m) as f64 * 0.37).fract();
                        let held: Vec<Option<Held>> = (0..outputs)
                            .map(|k| {
                                let mut rows = [0.; STATES];
                                let d = if k == 2 { shift } else { 0. };
                                (rows[0], rows[1]) = (0.5 + 0.1 * x + d, 0.5 - 0.1 * x - d);
                                Some(Held { rows, kappa: [x; 3], kappa2: [1. + x * x; 3] })
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
        let factor = refuted(&coarse, &level(-0.3), &scales, &targets, 0.25, 1., &|_| [0.05, 0.1]).unwrap();
        assert!((factor - 0.05 / 0.4).abs() < 1e-12, "{factor}");
        assert!(refuted(&coarse, &level(-0.01), &scales, &targets, 0.25, 1., &|_| [0.05, 0.1]).is_none());
    }
}
