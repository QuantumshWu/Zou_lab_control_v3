//! The two-point state selection (RUSTPLAN §1.1, its corrections at the end of RUSTPLAN.md; PSTATE): P judged at two
//! points against the full reference, the rest of the 32 folded by RS (`model::rs`). The two points: the centre (tweezer
//! at full intensity, v = 0) and the edge (tweezer intensity 0, v = 0, the centre's lattice phase). The only place P is
//! decided.
//!
//! # The gate
//! At each point, the model's readouts against the reference's (td_s.cmp with the force as a vector):
//! - populations by level, absolute: max_g |δp_g|;
//! - the scattering rate, relative: |δR| / R (every jump channel, the tweezer's folded ones included, in both);
//! - the force vector: |δF⃗| / max(|F⃗|, kR) -- the scale kR is the force a photon's momentum per scattering would give,
//!   so a vanishing F⃗ (a balanced point) is judged against the scattering it could produce, not against zero.
//!
//! Each ≤ `GATE` at both points passes; a measure that is not a number passes no gate. The diffusion is not judged (the
//! user: out of the model's gate), so only ρ is solved. Every solve the gate reads is certified (krylov.rs: its
//! backward error at the rounding floor); one that is not is an error of the selection, never a number read.
//!
//! # Points without light
//! A point no light reaches -- every first-order amplitude of every beam there, the tweezer's too, each frequency's
//! beams summed (`reach`), below the rounding, a < ε (the rounding rule's, `f64::EPSILON`) -- is not judged (`lit`):
//! - the reference has no unique steady state there: nothing pumps, so every mixture of the ground states is
//!   stationary (its system is singular, and `krylov::solve` refuses it);
//! - every P is exact there: the fold's corrections are products of the couplings, every one of them zero to rounding,
//!   so every model is the bare atom the reference is.
//!
//! There is nothing to compare: such a point's errors read zero (`PointError::unlit`), its reference none
//! (`Selection::reference`), and the report says so (`Selection::notes`). A scene whose two points are both dark has
//! the ground states -- the first whole-level candidate -- for its P. (R10's node scenes, the trap at a node of the
//! standing wave, and R3's, the trap at a node of its two beams: their edge, the tweezer off, is dark.)
//!
//! # The decision
//! One procedure in two stages:
//! 1. **The base** (`decide`): from the ground states, one whole level more at a time in the order of its largest
//!    first-order amplitude at the two points (`ranking`, scene2.ranking); the base is the first candidate passing.
//!    The references at both points, then the candidates in turn, each at both points at once (rayon). A candidate
//!    beyond the base is never solved: its model is larger than every one below it (a level more), and the largest
//!    (every level kept, the reference's size) would cost the most time and memory of all (GMD1C: the base 8 passes,
//!    its candidates run to 32). No candidate passing is an error, never an empty P.
//! 2. **Leave one out** (`leave_one_out`, `descend`; PFINE's, its prototype pfine/predict.py): each explicit excited
//!    state i of the base folded alone -- the base less i judged at both points, all of them at once (rayon) --, its
//!    e_i that model's worst error (the largest of the three measures over both points, against the full reference:
//!    the error of the base less i, not an increment over the base's). Then from the base the states are removed in
//!    increasing e_i (ties in the base's order) while the predicted error -- the base's worst plus the removed states'
//!    e_i -- stays within the gate; that set is checked at both points, and while it fails, the removed state of the
//!    largest e_i (the first of equal ones) is put back and the set checked again. P is the first set passing; every
//!    state put back, the base (which passed). The ground states are never removed.
//!
//! The prediction adds the e_i as if the removals' errors added up; each e_i holds the base's own error once more, so
//! the sum counts the base's error once per removed state above the once it starts with. Whatever it predicts, P is a
//! set measured within the gate at both points: the errors over removals are not monotone (a mirror pair with one side
//! kept loses its cancellation; a folded state of amplitude a ≳ 0.25–0.3 is outside RS), so a removal the prediction
//! allows may fail its check -- a check more and a state put back, never the gate (PSTATE's decision history: the gate
//! is judged at the two points alone, the user's rule of 11-19). The descent ends: each failed check puts a state back,
//! so there are at most as many checks as removed states, and a set of one removed state is its leave-one-out model,
//! measured at e_i ≤ the gate less the base's worst (its prediction).

use crate::floquet::{assemble, readouts, select, Frame, Liouville, Readouts};
use crate::krylov::{Route, Unsolved};
use crate::linalg::{C64, Mat};
use crate::model::rs::{self, decay, levels, ModelAt, Rs};
use crate::request::{GROUND, STATES};
use crate::scene::{Components, Scene};
use crate::units::SPEED_OF_LIGHT;
use rayon::prelude::*;
use std::time::Instant;

/// The gate on every readout's error: 1 % (the user's two-point threshold, RUSTPLAN §1.1).
pub const GATE: f64 = 0.01;

/// The two points: the centre (0) and the edge (1).
pub const POINTS: usize = 2;

/// The largest of some errors; not a number where one of them is not (# The gate).
fn largest(errors: &[f64]) -> f64 {
    errors.iter().fold(0.0, |m: f64, &x| if m.is_nan() || x.is_nan() { f64::NAN } else { m.max(x) })
}

/// One point's comparison of a candidate P's model with the full reference.
#[derive(Clone, Debug, PartialEq)]
pub struct PointError {
    /// The largest absolute population error over the levels, R's relative error, F⃗'s error |δF⃗| / max(|F⃗|, kR).
    pub populations: f64,
    pub rate: f64,
    pub force: f64,
    /// The model's solve's route; none at a point not judged.
    pub route: Option<Route>,
}

impl PointError {
    /// The largest of the three (`largest`).
    pub fn worst(&self) -> f64 {
        largest(&[self.populations, self.rate, self.force])
    }

    /// A point not judged (module documentation, # Points without light): every model exact there, so every error
    /// zero -- derived, not measured --, no solve.
    pub fn unlit() -> PointError {
        PointError { populations: 0.0, rate: 0.0, force: 0.0, route: None }
    }
}

/// One candidate P: its states and its errors at the centre and the edge.
#[derive(Clone, Debug, PartialEq)]
pub struct Candidate {
    pub keep: Vec<usize>,
    pub centre: PointError,
    pub edge: PointError,
}

impl Candidate {
    /// The larger of the two points' worst errors (`largest`).
    pub fn worst(&self) -> f64 {
        largest(&[self.centre.worst(), self.edge.worst()])
    }

    /// Every error at both points within `gate`.
    pub fn within(&self, gate: f64) -> bool {
        self.worst() <= gate
    }

    /// Within the gate (`GATE`) at both points.
    pub fn passes(&self) -> bool {
        self.within(GATE)
    }
}

/// One explicit excited state of the base folded alone (module documentation, # The decision, 2): the state and the
/// base less it, judged at both points.
#[derive(Clone, Debug, PartialEq)]
pub struct LeftOut {
    pub state: usize,
    pub candidate: Candidate,
}

impl LeftOut {
    /// Its e_i: the base less the state, its worst error at both points.
    pub fn e(&self) -> f64 {
        self.candidate.worst()
    }
}

/// The descent from the base at a gate (`descend`).
#[derive(Clone, Debug, PartialEq)]
pub struct Descent {
    /// P: the base less `removed`.
    pub keep: Vec<usize>,
    /// The states P leaves out of the base, in increasing e_i.
    pub removed: Vec<usize>,
    /// The first removal's predicted error: the base's worst plus its states' e_i.
    pub predicted: f64,
    /// The sets checked at both points in turn; none where nothing could be removed.
    pub checks: Vec<Candidate>,
}

/// The selection's report: the chosen P; the sets judged at both points on the way, with their errors; the full
/// reference's readouts and routes; the time.
#[derive(Clone, Debug, PartialEq)]
pub struct Selection {
    /// P.
    pub keep: Vec<usize>,
    /// Every set judged at both points in turn but the leave-one-out ones: the whole-level candidates up to the base
    /// (the first `whole` of them, the base the last), then the descent's checks. The last is P's.
    pub candidates: Vec<Candidate>,
    pub whole: usize,
    /// The base's explicit excited states each folded alone, in the base's order (`leave_one_out`).
    pub left_out: Vec<LeftOut>,
    /// The states P leaves out of the base and the first removal's predicted error (`Descent`).
    pub removed: Vec<usize>,
    pub predicted: f64,
    /// The full reference's readouts and route at each point; none at a point not judged (# Points without light).
    pub reference: Solved2,
    /// The time (s): the whole and its stages -- the base (the references and the whole levels), the leave-one-out, the
    /// descent's checks.
    pub seconds: f64,
    pub stages: [f64; 3],
}

impl Selection {
    /// The base: the first whole-level candidate passing (# The decision, 1).
    pub fn base(&self) -> &Candidate {
        &self.candidates[self.whole - 1]
    }

    /// What the report says beside the numbers (Chinese): each point not judged (# Points without light).
    pub fn notes(&self) -> Vec<String> {
        ["中心", "边缘"].iter().zip(&self.reference).filter(|(_, r)| r.is_none()).map(|(name, _)| format!("{name}点无光、无唯一稳态，不判")).collect()
    }
}

/// The full reference's readouts and route at each point, none at a point not judged: a point is judged where its
/// reference is.
pub type Solved2 = [Option<(Readouts, Route)>; POINTS];

/// The model's errors against the reference at one point; `k` the photon wave number of the force's scale kR.
pub fn errors(model: &Readouts, reference: &Readouts, k: f64, route: Route) -> PointError {
    let populations =
        model.levels.iter().zip(&reference.levels).map(|(m, r)| (m - r).abs()).fold(0.0, f64::max);
    let rate = (model.rate - reference.rate).abs() / reference.rate;
    let norm = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let delta = [0, 1, 2].map(|i| model.force[i] - reference.force[i]);
    let force = norm(delta) / norm(reference.force).max(k * reference.rate);
    PointError { populations, rate, force, route: Some(route) }
}

/// A set judged: its models' readouts and routes at the two points against the references' -- a point without its
/// reference not judged (`PointError::unlit`).
fn judged(keep: &[usize], models: &[Option<(Readouts, Route)>], reference: &Solved2, k: f64) -> Candidate {
    let point = |p: usize| match (&models[p], &reference[p]) {
        (Some(m), Some(r)) => errors(&m.0, &r.0, k, m.1),
        _ => PointError::unlit(),
    };
    Candidate { keep: keep.to_vec(), centre: point(0), edge: point(1) }
}

/// The points judged: those with a reference.
fn judged_points(reference: &Solved2) -> [bool; POINTS] {
    reference.each_ref().map(|r| r.is_some())
}

/// A set's models at the judged points, both at once; none at a point not judged.
fn pair<F>(keep: &[usize], points: [bool; POINTS], solve: &F) -> Result<Vec<Option<(Readouts, Route)>>, String>
where
    F: Fn(usize, &[usize]) -> Result<(Readouts, Route), String> + Sync,
{
    (0..POINTS).into_par_iter().map(|p| if points[p] { solve(p, keep).map(Some) } else { Ok(None) }).collect()
}

/// The base (module documentation, # The decision, 1) over ranked candidates (each `keep` a superset of the one
/// before), both points judged (`decide_on`).
pub fn decide<F>(candidates: &[Vec<usize>], k: f64, solve: F) -> Result<Selection, String>
where
    F: Fn(usize, Option<usize>) -> Result<(Readouts, Route), String> + Sync,
{
    decide_on(candidates, [true; POINTS], k, solve)
}

/// The base over ranked candidates, the points `lit` judged (# Points without light): `solve(point, None)` is the
/// full reference at a point, `solve(point, Some(c))` candidate c's model there, each giving the readouts and the
/// route, never asked at a point not judged. The references, then the candidates in turn, each at the judged points
/// at once; the base is the first candidate passing there. The report's P is the base, nothing left out yet (`refine`
/// takes it on).
pub fn decide_on<F>(candidates: &[Vec<usize>], lit: [bool; POINTS], k: f64, solve: F) -> Result<Selection, String>
where
    F: Fn(usize, Option<usize>) -> Result<(Readouts, Route), String> + Sync,
{
    let clock = Instant::now();
    let both = |c: Option<usize>| -> Result<Vec<Option<(Readouts, Route)>>, String> {
        (0..POINTS).into_par_iter().map(|p| if lit[p] { solve(p, c).map(Some) } else { Ok(None) }).collect()
    };
    let reference = both(None)?;
    let reference = [reference[0].clone(), reference[1].clone()];
    let mut tried: Vec<Candidate> = Vec::new();
    for (c, keep) in candidates.iter().enumerate() {
        let candidate = judged(keep, &both(Some(c))?, &reference, k);
        let passes = candidate.passes();
        tried.push(candidate);
        if passes {
            let seconds = clock.elapsed().as_secs_f64();
            let predicted = tried[c].worst();
            return Ok(Selection {
                keep: keep.clone(),
                whole: tried.len(),
                candidates: tried,
                left_out: Vec::new(),
                removed: Vec::new(),
                predicted,
                reference,
                seconds,
                stages: [seconds, 0.0, 0.0],
            });
        }
    }
    let largest = candidates.last().map_or(0, |k| k.len());
    Err(format!("no candidate P passes the {GATE} gate at the points judged (the largest has {largest} states)"))
}

/// Stage 2's measurements (module documentation, # The decision, 2): the base less each of its explicit excited states,
/// every set at the judged points at once (rayon); `solve(point, keep)` a model's readouts and route at a point. In
/// the base's order.
pub fn leave_one_out<F>(base: &Candidate, reference: &Solved2, k: f64, solve: &F) -> Result<Vec<LeftOut>, String>
where
    F: Fn(usize, &[usize]) -> Result<(Readouts, Route), String> + Sync,
{
    let lit = judged_points(reference);
    let states: Vec<usize> = base.keep.iter().copied().filter(|&s| s >= GROUND).collect();
    let sets: Vec<Vec<usize>> = states.iter().map(|&i| base.keep.iter().copied().filter(|&s| s != i).collect()).collect();
    let models: Vec<Option<(Readouts, Route)>> = (0..sets.len() * POINTS)
        .into_par_iter()
        .map(|j| if lit[j % POINTS] { solve(j % POINTS, &sets[j / POINTS]).map(Some) } else { Ok(None) })
        .collect::<Result<_, _>>()?;
    Ok(states
        .iter()
        .zip(&sets)
        .zip(models.chunks(POINTS))
        .map(|((&state, keep), models)| LeftOut { state, candidate: judged(keep, models, reference, k) })
        .collect())
}

/// Stage 2's decision (module documentation, # The decision, 2) at `gate` from the base, which passes it, and its
/// leave-one-out: the removal the prediction allows, then the checks, each at the judged points at once, with the
/// put-backs. P the first set passing or, every state put back, the base. Err where the base does not pass the gate.
pub fn descend<F>(base: &Candidate, left_out: &[LeftOut], gate: f64, reference: &Solved2, k: f64, solve: &F) -> Result<Descent, String>
where
    F: Fn(usize, &[usize]) -> Result<(Readouts, Route), String> + Sync,
{
    if !base.within(gate) {
        return Err(format!("the base ({} states) does not pass the {gate} gate", base.keep.len()));
    }
    let mut order: Vec<&LeftOut> = left_out.iter().collect();
    order.sort_by(|a, b| a.e().total_cmp(&b.e()));
    let mut predicted = base.worst();
    let mut removed: Vec<&LeftOut> = Vec::new();
    for state in order {
        if predicted + state.e() <= gate {
            predicted += state.e();
            removed.push(state);
        }
    }
    let mut checks: Vec<Candidate> = Vec::new();
    while !removed.is_empty() {
        let keep: Vec<usize> = base.keep.iter().copied().filter(|s| removed.iter().all(|r| r.state != *s)).collect();
        let candidate = judged(&keep, &pair(&keep, judged_points(reference), solve)?, reference, k);
        let passes = candidate.within(gate);
        checks.push(candidate);
        if passes {
            break;
        }
        let back = (0..removed.len()).fold(0, |b, j| if removed[j].e() > removed[b].e() { j } else { b });
        removed.remove(back);
    }
    let keep = if removed.is_empty() { base.keep.clone() } else { checks.last().expect("a removal is checked").keep.clone() };
    Ok(Descent { keep, removed: removed.iter().map(|r| r.state).collect(), predicted, checks })
}

/// Stage 2 on the base's selection (`decide`) at the gate (`GATE`): its leave-one-out and its descent, the report's P
/// the descent's, its checks after the whole levels -- and the base's measurement once more where every removed state
/// was put back, so that the last is P's.
pub fn refine<F>(mut selection: Selection, k: f64, solve: &F) -> Result<Selection, String>
where
    F: Fn(usize, &[usize]) -> Result<(Readouts, Route), String> + Sync,
{
    let clock = Instant::now();
    let base = selection.base().clone();
    let left_out = leave_one_out(&base, &selection.reference, k, solve)?;
    let looked = clock.elapsed().as_secs_f64();
    let descent = descend(&base, &left_out, GATE, &selection.reference, k, solve)?;
    let checked = clock.elapsed().as_secs_f64() - looked;
    selection.candidates.extend(descent.checks);
    if selection.candidates.last().map(|c| &c.keep) != Some(&descent.keep) {
        selection.candidates.push(base);
    }
    selection.keep = descent.keep;
    selection.left_out = left_out;
    selection.removed = descent.removed;
    selection.predicted = descent.predicted;
    selection.stages[1] = looked;
    selection.stages[2] = checked;
    selection.seconds += looked + checked;
    Ok(selection)
}

/// The two points' components (module documentation): the centre, the tweezer full, and the edge, dark; no Doppler
/// shift, the focus' lattice phase.
pub fn points(scene: &Scene) -> [Components; POINTS] {
    let zero = vec![0.0; scene.request.beams.len()];
    [scene.components([0.0; 3], &zero, 1.0), scene.components([0.0; 3], &zero, 0.0)]
}

/// The light's reach at a point (scene2.ranking): per excited state e the largest first-order amplitude
/// |V_eg| / |E_g + f − E_e + iΓ_e/2| over the ground states g and the beams `beams` takes -- V the coupling of a
/// frequency's beams together (its field: the beams of one frequency index summed, so that beams cancelling at the point
/// cancel), at f and its conjugate at −f; E the reference's static energies at the point (the tweezer's shift in), Γ its
/// decay. A beam with no component at the point couples nothing there.
pub fn reach(scene: &Scene, reference: &ModelAt, point: &Components, beams: &dyn Fn(usize) -> bool) -> [f64; STATES] {
    let request = &scene.request;
    let still = &reference.still().value;
    let gamma = decay(&reference.jumps, STATES).0;
    let mut score = [0.0_f64; STATES];
    let mut seen: Vec<&Vec<i32>> = Vec::new();
    for (l, beam) in request.beams.iter().enumerate().filter(|&(l, _)| beams(l)) {
        if seen.contains(&&beam.frequency_index) {
            continue;
        }
        seen.push(&beam.frequency_index);
        let mut v = Mat::zeros(STATES, STATES);
        for m in (0..request.beams.len()).filter(|&m| beams(m) && request.beams[m].frequency_index == beam.frequency_index) {
            let key = scene.unit(m);
            if let Some(c) = point.parts.iter().find(|c| c.key == key) {
                v.add_scaled(C64::ONE, &c.value);
            }
        }
        let f = scene.generators[l].frequency;
        for (x, f) in [(v.clone(), f), (v.adjoint(), -f)] {
            for e in GROUND..STATES {
                for g in 0..GROUND {
                    if x[(e, g)] != C64::ZERO {
                        let away = C64::new(still[(g, g)].re + f - still[(e, e)].re, 0.5 * gamma[(e, e)].re);
                        score[e] = score[e].max(x[(e, g)].abs() / away.abs());
                    }
                }
            }
        }
    }
    score
}

/// Whether light reaches the atom at a point (module documentation, # Points without light): some first-order
/// amplitude of every beam's light there (`reach`, the tweezer's included) at or above the rounding, ε.
pub fn lit(scene: &Scene, reference: &ModelAt, point: &Components) -> bool {
    reach(scene, reference, point, &|_| true).iter().any(|&a| a >= f64::EPSILON)
}

/// The candidate levels in the order of their largest first-order amplitude at the two points (scene2.ranking): per
/// excited state the largest over the points of the reach of the light other than the tweezer (`reach`); a level's
/// the largest of its states'. Each level its states (indices into the 32), the best first (a tie in the levels'
/// order).
pub fn ranking(scene: &Scene, references: &[ModelAt], points: &[Components]) -> Vec<Vec<usize>> {
    let request = &scene.request;
    let tw = request.trap_beam;
    let mut score = [0.0_f64; STATES];
    for (reference, point) in references.iter().zip(points) {
        for (s, a) in score.iter_mut().zip(reach(scene, reference, point, &|l| l != tw)) {
            *s = s.max(a);
        }
    }
    let all: Vec<usize> = (0..STATES).collect();
    let mut found: Vec<(f64, Vec<usize>)> = levels(&request.states, &all)
        .into_iter()
        .filter(|level| level[0] >= GROUND)
        .map(|level| (level.iter().map(|&e| score[e]).fold(0.0, f64::max), level))
        .collect();
    found.sort_by(|a, b| b.0.total_cmp(&a.0));
    found.into_iter().map(|(_, level)| level).collect()
}

/// One point's solve (the reference's or a model's): the Hilbert model through the one Liouville assembly, the rounding
/// rule's elements and the solve (floquet, krylov), its readouts and route. A system with no solution, or a solve that
/// is not certified, is an error (module documentation, # The gate): no number is read off it.
fn solved(frame: &Frame, model: &ModelAt) -> Result<(Readouts, Route), String> {
    let liouville = Liouville::new(frame, &model.hilbert());
    let system = assemble(&liouville, &select(&liouville));
    let solved = crate::krylov::solve(&system).map_err(|e| match e {
        Unsolved::Inconsistent(i) => format!("the checkpoint's system is inconsistent at {} rows", i.rows.len()),
        Unsolved::NoIterate { attempts } => format!(
            "the checkpoint's system has no solution: no route gave a finite iterate ({})",
            attempts.iter().map(|a| format!("{:?}{}", a.route, a.unfactorised.as_ref().map_or(String::new(), |u| format!(": {u}")))).collect::<Vec<_>>().join("; ")
        ),
    })?;
    if !solved.certified {
        return Err(format!("the checkpoint's solve is not certified (route {:?}, ζ {:.3e})", solved.route, solved.certificate));
    }
    Ok((readouts(&liouville, &system, &solved.rho), solved.route))
}

/// What every solve of a scene's selection reads: the two points (`points`), the full reference at each
/// (`rs::reference`) and its frame (`Frame::new`: the reference there, no Doppler shift), and whether light reaches
/// the atom there (`lit`: the points judged); the whole-level candidates -- the ground states, then one level more at
/// a time in `ranking`'s order --; the force's scale k, the carrier's wave number (ν/c, td_s's kref).
pub struct TwoPoints<'a> {
    scene: &'a Scene,
    points: [Components; POINTS],
    references: Vec<ModelAt>,
    frames: Vec<Frame>,
    pub lit: [bool; POINTS],
    pub candidates: Vec<Vec<usize>>,
    pub k: f64,
}

impl<'a> TwoPoints<'a> {
    pub fn new(scene: &'a Scene) -> TwoPoints<'a> {
        let points = points(scene);
        let references: Vec<ModelAt> = points.iter().map(|p| rs::reference(scene, p)).collect();
        let zero = vec![0.0; scene.request.beams.len()];
        let frames: Vec<Frame> = references.iter().map(|r| Frame::new(scene, &r.hilbert(), &zero)).collect();
        let mut keep: Vec<usize> = (0..GROUND).collect();
        let mut candidates = vec![keep.clone()];
        for level in ranking(scene, &references, &points) {
            keep.extend(level);
            keep.sort_unstable();
            candidates.push(keep.clone());
        }
        let k = scene.generators.last().expect("the carrier").frequency / SPEED_OF_LIGHT;
        let lit = std::array::from_fn(|p| lit(scene, &references[p], &points[p]));
        TwoPoints { scene, points, references, frames, lit, candidates, k }
    }

    /// The full reference's readouts and route at point p.
    pub fn reference(&self, p: usize) -> Result<(Readouts, Route), String> {
        solved(&self.frames[p], &self.references[p])
    }

    /// The RS model on `keep` (indices into the 32) at point p, on that point's frame's sectors (`Rs::new`, `Rs::at`).
    pub fn model(&self, p: usize, keep: &[usize]) -> Result<(Readouts, Route), String> {
        solved(&self.frames[p], &Rs::new(self.scene, keep, self.frames[p].sectors()).at(self.scene, &self.points[p]))
    }
}

/// Chooses P for a scene (module documentation, # The decision): on the two points (`TwoPoints`), those light reaches
/// judged (# Points without light), the base over the whole-level candidates (`decide_on`), then its leave-one-out
/// and descent (`refine`); every solve the same (`solved`).
pub fn choose(scene: &Scene) -> Result<Selection, String> {
    let two = TwoPoints::new(scene);
    let base = decide_on(&two.candidates, two.lit, two.k, |p, c| match c {
        None => two.reference(p),
        Some(c) => two.model(p, &two.candidates[c]),
    })?;
    refine(base, two.k, &|p: usize, keep: &[usize]| two.model(p, keep))
}

#[cfg(test)]
mod tests;
