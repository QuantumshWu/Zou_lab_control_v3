//! The selection of P (PIPELINE_V3_FINAL §二 A.3, A.5–A.6, B.7–B.11, C.13–C.15): the candidates by structure, each
//! judged by what it does to the run's readings along the atoms' own paths against the 32-state model (paths.rs), and
//! the candidate of least total work chosen among those whose model error A leaves the targets room.
//! The trajectories' parts (the pilot the paths come from, T's memory read on the run's own trajectories, the work)
//! are the solve's (solve.rs, # The selection); what is here is the model's side and the accounts.
//!
//! # The candidates
//! Every excited state e has an importance: its first-order excitation Σ_g p_g Σ_ω |V_eg|²/|E_g + f − E_e + iΓ_e/2|²
//! by the light other than the tweezer's (`excitation`: the population it would hold to the first order, the fold's
//! leading term weighed by the ground states its atoms come from), the largest over the places the pilot's atoms were
//! seen at with their ground populations there (`importance`; before the pilot, `amplitudes`: over the three structural
//! points -- the tweezer's focus with its light full (s = 1), the focus with it off (s = 0), and the place of half its
//! intensity along the direction the other light changes fastest (`middle`) -- from every ground state alike). No
//! equation is solved for it: it is the coupling over the detuning, where the atoms are. (By the amplitude alone,
//! |V|/|Δ + iΓ/2|, PGC3D's repumper resonant on F = 1 → F′ = 2 put F′ = 2 first, 0.27 against 0.06–0.11 for F′ = 3,
//! while 90 % of the atoms are in F = 2: the three-point choice of 2026-11-21 kept F′ = 3 m = 0, ±1 and folded every
//! F′ = 2 state.) The chain (`chain`) starts at the model in which every
//! level is whole -- the 32 states, nothing folded -- and folds the excited states one at a time in increasing
//! importance, those equal to the rounding together (a mirror pair is folded as a pair: one side kept would lose the
//! pair's cancellation), down to the full-ground model G, every excited state folded. The chain is nested and made by the
//! structure alone: no candidate is judged at a point, no gate is read (the three-point check, its 1 % gate, the
//! leave-one-out and its stopping at the gate, and the fold condition s < ½ of model/reach.rs are gone: the user's of
//! 2026-11-21, PIPELINE_V3_FINAL §二 A.6). What stays is the runaway's check of the steps (step.rs, # The domain): a fold
//! that diverges where an atom goes scatters beyond every state of the atom there and is booked whole.
//!
//! # The paths
//! The candidates are judged along `WALKS` of the pilot's trajectories (solve.rs `walks`; P1-PATHS): each is advanced
//! by the candidate and by the 32-state model in the ground-holding exact blocks' density matrices along the same path
//! (paths.rs `Form::along`), by the exponential scheme in H's eigenbasis (paths.rs, # The substeps), and their readings compared
//! in windows of the memory time (paths.rs `windows`, `compare`). A path is read at its start for L = min(T, 5 τ_mem),
//! τ_mem the slowest decay of the full-ground model's generator, the longest over the paths' starts (`memory`; P1 §5: it
//! differs with the place by up to 2.4 times) -- the start's transient and the internal state's forgetting of it --, and,
//! where the run is longer than 2L, again over its last L from the walk's own start state (`Paths::late_from`): its last
//! window reads the model where the atoms are late in the run, which the start's windows cannot (the motion's own
//! memory is the longer: R4's atoms heat through the run).
//!
//! The scheme's setting (`Setting`) -- the substeps a turn of the fastest coherence over a stretch's opening (the
//! start's transient, paths.rs # The start), the substeps a µs after it, and the pieces a step -- is settled by its own
//! error, not fixed
//! (`settle`): on the first path's first `CALIBRATION` µs the reference is advanced at the setting and at each knob
//! doubled, each knob's part of the setting's error the largest change of a displayed row from doubling it alone
//! (Richardson's 2^p/(2^p − 1), p = 2: the amplitudes' linear interpolation at one piece a step, R's at the substep's
//! middle -- paths.rs # The nodes, # The substeps; a rate no lower than one substep a piece, so
//! that where twice the pieces raise a rate to that floor the pieces' part is read at the raised rates,
//! `refinements`); from
//! the coarsest (`COARSEST`), the knob of the largest part is doubled while the parts' sum is beyond `TOLERANCE` -- s⁴
//! of the population target, the share the numerical tolerances inside a step take of theirs (units.rs `QUADRATURE`)
//! --, to `FINEST` at the most (a bound on the work, not on the accuracy: stopped there short of the tolerance, the
//! setting is said not converged and the reference's parts are booked whole in every candidate's A,
//! `Settled::unmet`). Every path, every candidate and the scans take that setting. The
//! same refinements of each candidate on the same piece give its calibration (`Calibration`): the change of its rows'
//! difference to the reference, read alike, booked beside the direct part. (Until 2026-11-22 the setting was fixed at
//! 128 substeps a µs and two pieces, P1-PATHS's measure on R4, and the calibration read against the converged
//! `Scheme::sparse`, whose segments follow the fastest coherence: on S1 -- its 32 states hold levels far off the light's
//! frame -- 0.5 µs of it took more than 190 s, where the whole setting ladder takes a second. The refinements
//! converge to the same limit: H exact in each piece's eigenbasis as the pieces shorten, the exponential Runge–Kutta
//! as the substeps do. Until SELCOST the substeps were Lawson's, whose rate had to resolve the detunings, paths.rs #
//! The substeps.)
//!
//! The reference is the 32-state model itself (B.8), not a whole-level base: under P1's scheme its path cost 10–20 % more
//! than the base's (P1 R2.3), and with it nothing remains of a base's own error to check request by request.
//!
//! # The direct part
//! Per output time t, its window's (`paths::window_of`) error of each displayed row, |mean| + z·se over the paths
//! (`paths::Mean`; z the run's `CONFIDENCE`), the largest over the rows, and the scheme's calibration beside it; past L
//! the larger of the start's last window's and the late segment's last window's, and past the late segment's start the
//! latter's.
//!
//! # The feedback through the motion: zero
//! The model's internal error moves the atoms, and where they go moves every reading. Its effect is not booked: the
//! user's of 2026-11-22, "更严格的上界，尝试解决这个问题，如果确实解决不了，那么取消掉反馈这块，直接认为反馈这点等于0" --
//! the strict bound by the Duhamel identity and the model-only momentum-scaling response (P3-SELECT's first rounds)
//! read 5–200 times PAIR's truth on PGC3D and R4 and kept every scene off its full-ground model, and FOLDTEMP's third
//! round found no tighter bound to be had. The temperature and the loss have no model error; the run says so once (the
//! ledger's item, solve.rs `NO_FEEDBACK`).
//!
//! ## T's memory, read on the run's own trajectories
//! One flag is left of the momentum-scaling response: T's memory not shorter than the run (FOLDTEMP §R2.3, "缩放响应在
//! 运行末仍可读出"): whether the run's end still remembers its initial state. It is read on the product's own
//! trajectories, no trajectory more (solve.rs `memory_at_the_end`; until the coordinator's item 7 of 2026-11-22 two more
//! runs of the pilot's atoms at 1 ± η did it, 4–52 % of the chosen W):
//! - Any perturbation of the initial state not orthogonal to the slow modes asks it. A perturbation p_ε of the draws'
//!   density moves a reading by R_O(t) = d/dε ∫ O(t; x₀) p_ε(x₀) dx₀ = E[O(t) s(x₀)] (the likelihood-ratio identity), s =
//!   ∂_ε log p_ε at ε = 0 the draw's score, the dynamics after t = 0 unchanged; E[s] = 0, so R_O = Cov(O(t), s), and
//!   for an estimate that is a smooth function of the run's sums -- T, a U-statistic of the held atoms' κ and κ² over
//!   their number -- R is its derivative with every trajectory reweighed by 1 + εs.
//! - The perturbation is the initial kinetic energy's exponential tilt, p_ε ∝ p e^{−εK} normalized, K the classical
//!   draw's kinetic energy in k_BT (ensemble.rs, # Its draw; `Ensemble::draw_kinetic`): s = ⟨K⟩ − K. Like the momenta's
//!   scaling it changes the initial state's temperature, and overlaps the slow modes as much; unlike it, its score is
//!   bounded -- the draw's density (e^{y−K} − 1)₊ holds 0 ≤ K ≤ y, y the depth of the place, at most the well's -- and its
//!   variance finite. (The momenta's scaling, by the same identity, had s = 2K e^{y−K}/(e^{y−K} − 1) − 3: it grows as
//!   2y/(y − K) at the escape energy, where the density vanishes linearly, and its second moment diverges as the
//!   logarithm; the groups' jackknife of such a heavy tail reads too small an error. The coordinator's item 8.) The
//!   Wigner map (# The Wigner factor) is a fixed map of the classical draw, so the mapped draws are tilted by the same
//!   factor, and a mixture's component is the trajectory's internal state: the tilt is of each component's own density.
//! - ⟨K⟩ is the run's own mean K̄ over its trajectories (failed or not, at t = 0), so Σ s = 0 over the run: its sums'
//!   counts are not reweighed to the first order. A ratio of reweighed sums is unmoved by a constant added to s, so K̄'s
//!   own sampling error enters only through the U-statistic's (n − 1) -- a relative 1/n of the response --, and the
//!   sample covariance's bias is the same 1/n against the jackknife's 1/√n: both negligible.
//! - The question's scale (`decide`): the momenta's scaling by a unit moved T(0) by 2T(0), and the flag asked whether
//!   that much at t = 0 still moves the end's T by more than its allowance. The tilt's response is put on that scale by
//!   its own start: the end's memory m = 2T(0) R_T(end)/R_T(0), T(0) the run's, its error the groups' jackknife of the
//!   ratio itself. Read there (|m| > z·se), flagged; within the allowance at its upper end, or not told, no flag.
//!
//! # The region left out
//! The chosen candidate is scanned along the prepared trap's three principal axes, both ways, from its centre at the
//! pilot's thermal speed against the reference (`scan`), each scan to the turning point of the pilot's most
//! energetic held atom on the reference state's prepared potential, or to the trap region's edge (`REGION_INTENSITY` of
//! the tweezer's peak), whichever is nearer -- the reach the pilot's atoms show (solve.rs `choose_on_the_pilot`): where
//! the paths did not go -- the tweezer's intensity s below the least the paths reached --, the readings where a row errs
//! by a quarter of its target or more mark the region {s ≤ s*}, s* the most intense of them, its error e the largest
//! row's error there; below the least intensity the scans and the paths reached nothing is read, and an atom there is unknown, its
//! error whole (`Region`). The run books every atom in either (solve.rs `booked_at`): a share f of them errs by at most e
//! (one) on what is shown and moves T by the bound set's `Estimate::moved(f)`. (Only the chosen one: a scan of the
//! 32-state reference to the trap's region's edge took R4 50 s at its pilot's step, the candidates' own a minute more.)
//!
//! A scan is read every `SCAN_READ` of the pilot's steps, its knots there (a straight line at a constant speed has no
//! step ends of its own), and each scan's scheme is settled on its own first `CALIBRATION` µs by the error of what it
//! reads, the candidate's rows' difference to the reference's (`settle`, the paths' rule): along a line H changes as
//! the light does there -- along R4's tweezer axis only with its
//! intensity, over z_R = 92 µm --, not as along the paths, whose setting (taken until SELCOST: four pieces a 0.12 µs step
//! there, a node every 2.7 nm) the scans then followed.
use crate::field::Light;
use crate::linalg::{larger, symmetric3_eigen, C64, Mat, V3};
use crate::model::rs::{self, decay, ModelAt};
use crate::model::selected::Selected;
use crate::paths::{self, Form, Scheme, Series, Walk, Window};
use crate::request::{GROUND, STATES};
use crate::scene::{Components, Scene};
use crate::step::Dynamics;
use crate::units::{CONFIDENCE, POPULATION_TARGET, STEP_SHARE};
use rayon::prelude::*;
use std::time::Instant;

/// The points the structural amplitudes are read at: the centre (0), the edge (1), the middle (2) (module
/// documentation, # The candidates).
pub const POINTS: usize = 3;

/// The pilot's paths a candidate is judged on (BRIEF_P3_SELECT item 2: P1-PATHS's path-to-path spread is at most 2e-3,
/// one or two paths reach a quarter of the target, four cover the start's draws).
pub const WALKS: usize = 4;

/// The paths' scheme's tolerance (module documentation, # The paths): the largest error of a displayed row the
/// reference may take from the scheme's setting on the calibration's piece -- s⁴ of the population target, as the
/// numerical tolerances inside a step take s⁴ of theirs (units.rs `QUADRATURE`): 2.0e-3 at p = 2.
pub const TOLERANCE: f64 = STEP_SHARE * STEP_SHARE * STEP_SHARE * STEP_SHARE * POPULATION_TARGET;

/// A setting of the paths' scheme (module documentation, # The paths): the substeps a turn of the fastest coherence over
/// a stretch's opening, the substeps a µs after it (paths.rs, # The start), and the pieces a step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Setting {
    pub turns: f64,
    pub rate: f64,
    pub pieces: usize,
}

/// The knobs of a setting (`Setting`): the opening's substeps, the run's substeps, the pieces.
pub const KNOBS: usize = 3;

impl Setting {
    /// The setting with its knob `k` (`KNOBS`) doubled.
    fn doubled(self, k: usize) -> Setting {
        let mut s = self;
        match k {
            0 => s.turns *= 2.0,
            1 => s.rate *= 2.0,
            _ => s.pieces *= 2,
        }
        s
    }

    /// Whether its knob `k` is below `FINEST`'s.
    fn open(&self, k: usize) -> bool {
        match k {
            0 => self.turns < FINEST.turns,
            1 => self.rate < FINEST.rate,
            _ => self.pieces < FINEST.pieces,
        }
    }
}

/// The setting the paths' scheme starts from (module documentation, # The paths): a substep a turn of the fastest
/// coherence over the opening (its Nyquist rate, paths.rs # The start), 8 a µs after it, one piece a step -- each knob
/// doubled from there as its error asks.
pub const COARSEST: Setting = Setting { turns: 1.0, rate: 8.0, pieces: 1 };

/// The finest setting (module documentation, # The paths): 64 substeps a turn, 4096 a µs (32 times P1-PATHS's 128), 64
/// pieces a step -- a bound on the work, never a measure of accuracy: a setting settled there short of `TOLERANCE` is
/// said ("到顶未收敛") and its reference's error parts booked whole in every candidate's A (`Settled::unmet`).
pub const FINEST: Setting = Setting { turns: 64.0, rate: 4096.0, pieces: 64 };

/// A refinement's change read as the coarser setting's error (module documentation, # The paths): Richardson's
/// 2^p/(2^p − 1) at the scheme's lowest order p = 2 -- the amplitudes' linear interpolation at one piece a step, the
/// recycling's amplitudes at the substep's middle (the cubic interpolation and the exponential Runge–Kutta err at order
/// 3 or more).
const RICHARDSON: f64 = 4.0 / 3.0;

/// The paths' scheme at a setting, read at every `read`-th knot.
fn scheme(setting: Setting, read: usize) -> Scheme {
    Scheme::Exponential { turns: setting.turns, rate: setting.rate, pieces: setting.pieces, read }
}

/// The calibration's piece (µs): the first path's start, where the readings change fastest (P1-PATHS: one path, 10 µs).
pub const CALIBRATION: f64 = 10.0;

/// The measurements' trace (SELCOST): `SELCOST_TRACE` set, each job's work said.
#[cfg(test)]
pub(crate) fn traced() -> bool {
    std::env::var_os("SELCOST_TRACE").is_some()
}

/// One job's work in the trace (`traced`): its wall seconds and its series' counts.
#[cfg(test)]
pub(crate) fn trace_job(what: &str, s: &Series, wall: f64) {
    if traced() {
        let c = &s.counts;
        eprintln!(
            "    [trace] {what}: {:.1} µs, wall {wall:.2} s; thread s {:.2} (freeze {:.2} + propagation {:.2} + readouts {:.2}); {} nodes, {} substeps, {} products, {} readouts",
            c.time, c.seconds(), c.node_seconds, c.propagation_seconds, c.readout_seconds, c.nodes, c.segments, c.products, c.readouts
        );
    }
}

/// A run's states: P, the structural chain it is one of (module documentation, # The candidates), the states'
/// structural amplitudes, the chain's time, and what the choice made of the chain where it was made (`Choice`: after the
/// pilot, solve.rs # The selection).
#[derive(Clone, Debug)]
pub struct Selection {
    /// P (indices into the 32).
    pub keep: Vec<usize>,
    /// The chain from the full-ground model G (first) up to the 32 states (last), each sorted.
    pub chain: Vec<Vec<usize>>,
    /// Per state its structural amplitude (zero on the ground states).
    pub amplitudes: [f64; STATES],
    pub seconds: f64,
    pub choice: Option<std::sync::Arc<Choice>>,
}

impl Selection {
    /// The full-ground model's states: the chain's first, the model the scout and the pilot run.
    pub fn ground(&self) -> &[usize] {
        &self.chain[0]
    }

    /// What the report says beside the numbers (Chinese): the chain, and the choice where it was made.
    pub fn notes(&self) -> Vec<String> {
        let mut notes = vec![format!("结构链 {} 档：{}", self.chain.len(), self.chain.iter().map(|c| c.len().to_string()).collect::<Vec<_>>().join(" → "))];
        if let Some(choice) = &self.choice {
            notes.extend(choice.report.iter().cloned());
        }
        notes
    }
}

/// The selection's chain for a scene (module documentation, # The candidates): its states' structural amplitudes and the
/// chain; P the full-ground model, until the choice (solve.rs, # The selection) takes another.
pub fn choose(scene: &Scene) -> Result<Selection, String> {
    let clock = Instant::now();
    let amplitudes = amplitudes(scene);
    let chain = chain(&amplitudes);
    Ok(Selection { keep: chain[0].clone(), chain, amplitudes, seconds: clock.elapsed().as_secs_f64(), choice: None })
}

/// The chain (module documentation, # The candidates) by the states' structural amplitudes: from the ground states up,
/// the excited states added in decreasing amplitude, those equal to the rounding at once, to all 32.
pub fn chain(amplitudes: &[f64; STATES]) -> Vec<Vec<usize>> {
    let mut excited: Vec<usize> = (GROUND..STATES).collect();
    excited.sort_by(|&x, &y| amplitudes[y].total_cmp(&amplitudes[x]).then(x.cmp(&y)));
    let mut keep: Vec<usize> = (0..GROUND).collect();
    let mut chain = vec![keep.clone()];
    let tie = crate::units::rounding(STATES);
    let mut i = 0;
    while i < excited.len() {
        let top = amplitudes[excited[i]];
        let mut j = i + 1;
        while j < excited.len() && (top - amplitudes[excited[j]]).abs() <= tie * larger(top.abs(), amplitudes[excited[j]].abs()) {
            j += 1;
        }
        keep.extend(&excited[i..j]);
        keep.sort_unstable();
        chain.push(keep.clone());
        i = j;
    }
    chain
}

/// The structural points' components (module documentation, # The candidates), the atom at rest (no Doppler shift): the
/// centre, the tweezer full, and the edge, dark, both at the focus' place; the middle at `middle`, every light there as
/// it is (the tweezer at its half intensity).
pub fn points(scene: &Scene) -> [Components; POINTS] {
    let zero = vec![0.0; scene.request.beams.len()];
    [scene.components([0.0; 3], &zero, 1.0), scene.components([0.0; 3], &zero, 0.0), scene.components(middle(scene), &zero, 1.0)]
}

/// The middle point's place (µm from the focus): n̂ the top eigenvector of M = Σ_w Re(∇E_w ∇E_w†) at the focus over the
/// light other than the tweezer's -- the direction its pattern changes fastest (Rayleigh–Ritz; Horn & Johnson, Thm
/// 4.2.2), the tweezer's axis where M = 0 --, its sign the one that makes its largest component positive; along it the
/// distance where the tweezer's paraxial Gaussian intensity is half its peak, its one root found by bisection to the
/// rounding (the intensity falls strictly along every ray from the focus).
pub fn middle(scene: &Scene) -> V3 {
    let light = Light::new(&scene.request);
    let mut m = [[0.0_f64; 3]; 3];
    for frequency in light.frequencies.iter().enumerate().filter(|&(w, _)| w != light.trap).map(|(_, f)| f) {
        let mut g = [[C64::ZERO; 3]; 3];
        for (&l, field) in frequency.beams.iter().zip(&frequency.fields) {
            let a = light.beams[l].amplitude_to([0.0; 3], 1).gradient;
            for (gi, ai) in g.iter_mut().zip(a) {
                for (x, e) in gi.iter_mut().zip(field) {
                    *x += ai * *e;
                }
            }
        }
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] += (0..3).map(|q| (g[i][q] * g[j][q].conj()).re).sum::<f64>();
            }
        }
    }
    let tweezer = &light.beams[light.frequencies[light.trap].beams[0]];
    let (values, vectors) = symmetric3_eigen(&m);
    let n = if values[2] > 0.0 {
        let v = vectors[2];
        let big = (0..3).fold(0, |b, i| if v[i].abs() > v[b].abs() { i } else { b });
        if v[big] < 0.0 { v.map(|x| -x) } else { v }
    } else {
        tweezer.direction
    };
    let below = |t: f64| tweezer.amplitude_to(n.map(|x| t * x), 0).value.norm_sqr() - 0.5;
    let (mut lo, mut hi) = (0.0, tweezer.rayleigh().expect("the tweezer is a Gaussian beam (Request::validate)"));
    while below(hi) >= 0.0 {
        (lo, hi) = (hi, 2.0 * hi);
    }
    loop {
        let mid = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi {
            break;
        }
        if below(mid) >= 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let t = if below(lo).abs() <= below(hi).abs() { lo } else { hi };
    n.map(|x| t * x)
}

/// Per state its importance at the structural points (module documentation, # The candidates), before any atom is
/// seen: the largest over the three of its first-order excitation by the light other than the tweezer's from every
/// ground state alike (`excitation`, p_g = 1/8); zero on the ground states. The chain the pilot's choice is made on takes
/// the pilot's states instead (`importance`).
pub fn amplitudes(scene: &Scene) -> [f64; STATES] {
    let tw = scene.request.trap_beam;
    let even = [1.0 / GROUND as f64; GROUND];
    let mut score = [0.0_f64; STATES];
    for point in points(scene) {
        let reference = rs::reference(scene, &point);
        for (s, a) in score.iter_mut().zip(excitation(scene, &reference, &point, &|l| l != tw, &even)) {
            *s = larger(*s, a);
        }
    }
    score
}

/// The first-order excitation of each excited state at a point by the light `beams` takes, from the ground populations
/// `p`: Σ_g p_g Σ_± |V^±_eg|²/|E_g ± f − E_e + iΓ_e/2|² -- V the coupling of a frequency's beams together (its field:
/// the beams of one frequency index summed, so that beams cancelling at the point cancel), at f and its conjugate at −f;
/// E the reference's static energies at the point (the tweezer's shift in), Γ its decay; a beam with no component at the
/// point couples nothing there --: the population the state would hold to the first order, the fold's leading term
/// weighed by the ground states it takes its atoms from.
pub fn excitation(scene: &Scene, reference: &ModelAt, point: &Components, beams: &dyn Fn(usize) -> bool, p: &[f64; GROUND]) -> [f64; STATES] {
    let request = &scene.request;
    let still = &reference.still().value;
    let gamma = decay(&reference.jumps, STATES).0;
    let mut out = [0.0_f64; STATES];
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
                        out[e] += p[g] * x[(e, g)].norm_sqr() / away.norm_sqr();
                    }
                }
            }
        }
    }
    out
}

/// Each excited state's importance on the chain (module documentation, # The candidates): the largest, over the places
/// `places` an atom was seen at with its ground populations there (the pilot's states: an atom at rest there, the
/// tweezer's light as it is), of its first-order excitation by the light other than the tweezer's (`excitation`). Where
/// the atoms go and from the states they are in: a repumper resonant on F = 1 → F′ = 2 gives F′ = 2 the larger
/// amplitude (PGC3D: 0.27 against 0.06–0.11 for F′ = 3), but the atoms are in F = 2 (90 %), whose F′ = 3 holds the
/// excitation; a lattice's polarization reaches states between the structural points (R4's F′ = 3 stretched states).
pub fn importance(scene: &Scene, places: &[(V3, [f64; GROUND])]) -> [f64; STATES] {
    let tw = scene.request.trap_beam;
    let zero = vec![0.0; scene.request.beams.len()];
    places
        .par_iter()
        .map(|(r, p)| {
            let point = scene.components(*r, &zero, 1.0);
            excitation(scene, &rs::reference(scene, &point), &point, &|l| l != tw, p)
        })
        .reduce(|| [0.0; STATES], |a, b| std::array::from_fn(|s| larger(a[s], b[s])))
}

// ---------------------------------------------------------------------------------------------------------------------
// The paths and the candidates' judgment.

/// The tweezer's intensity at a place, relative to its peak (s = |a_tw(r)|²).
pub fn intensity(light: &Light, r: V3) -> f64 {
    light.beams[light.frequencies[light.trap].beams[0]].amplitude_to(r, 0).value.norm_sqr()
}

//// What every candidate is judged against (module documentation, # The paths): the walks cut at L, their windows, the
/// reference's form, the scheme's setting (`settle`), the reference's series along the walks at it and on the
/// calibration's piece at it and its two refinements, the least tweezer intensity the walks reached, the memory time,
/// and the work it took.
pub struct Paths<'a> {
    pub walks: Vec<Walk>,
    pub length: f64,
    pub memory: f64,
    pub windows: Vec<(f64, f64)>,
    pub reference: Form<'a>,
    pub settled: Settled,
    series: Vec<Series>,
    calibration: (f64, Vec<Series>),
    /// The late segment (module documentation, # The paths): its start, its windows and the reference's series there;
    /// none where the run is no longer than twice L.
    late: Option<(f64, Vec<(f64, f64)>, Vec<Series>)>,
    pub least: f64,
    pub seconds: f64,
    pub products: u64,
}

/// The paths' scheme as settled (`settle`; module documentation, # The paths): its setting, the calibration's opening (µs), the
/// reference's error parts there on the calibration's piece (per knob, `KNOBS`), the calibration's runs -- the setting
/// first, its refinements after -- and per knob the two of them its part is read between (`refinements`), the settings
/// run, and their work and time.
#[derive(Clone, Debug)]
pub struct Settled {
    pub setting: Setting,
    pub opening: f64,
    pub errors: [f64; KNOBS],
    pub runs: Vec<Setting>,
    pub pairs: [(usize, usize); KNOBS],
    pub tried: usize,
    pub products: u64,
    pub seconds: f64,
}

impl Settled {
    /// Where the setting stopped at `FINEST` with its parts still beyond `TOLERANCE` (the ladder's top, not
    /// converged): their sum, booked in every candidate's A beside its own calibration (`judge`, `Paths::itself`); none
    /// where the parts met the tolerance.
    pub fn unmet(&self) -> Option<f64> {
        let sum: f64 = self.errors.iter().sum();
        (sum > TOLERANCE).then_some(sum)
    }

    /// The scheme at the setting, read at every `read`-th knot.
    pub fn scheme(&self, read: usize) -> Scheme {
        scheme(self.setting, read)
    }
}

/// A setting's refinements (module documentation, # The paths): its runs -- the setting, and each knob doubled, raised
/// to one substep a piece (`own`) --, and per knob the two runs its part is read between: the substeps' doubled at the
/// setting's pieces; the pieces' doubled at one rate, the doubled pieces' own -- where twice the pieces raise the rate
/// to that floor, against the setting's pieces at the raised rate, one run more where that is none of the others. (Read against the setting itself, the pieces' part held the substeps' wherever the floor doubled them too:
/// TET's ladder doubled the pieces to 16 while its substeps erred, the pieces' own part at 4 a step 1.24e-3, SELCOST.)
fn refinements(setting: Setting, own: &dyn Fn(Setting) -> Setting) -> (Vec<Setting>, [(usize, usize); KNOBS]) {
    let mut runs: Vec<Setting> = std::iter::once(setting).chain((0..KNOBS).map(|k| own(setting.doubled(k)))).collect();
    let pieces = KNOBS;
    let partner = Setting { rate: runs[pieces].rate, ..setting };
    let at = runs.iter().position(|&s| s == partner).unwrap_or_else(|| {
        runs.push(partner);
        runs.len() - 1
    });
    (runs, std::array::from_fn(|k| if k + 1 == pieces { (at, pieces) } else { (0, k + 1) }))
}

/// The scheme's setting by its error (module documentation, # The paths): from `COARSEST`, on the walk's first `piece`
/// µs, the models `forms` at the setting and its refinements (`refinements`), each knob's part of the setting's error
/// the largest change between its two runs (`change`: of a displayed row of the reference, the paths'; of a row's
/// difference of the candidate to the reference, a scan's), read as its error (`RICHARDSON`); while the parts' sum is
/// beyond `TOLERANCE`, the knob of the largest part doubled (of those below `FINEST`), all at the finest the last. A
/// setting's rate is the substeps' own: no less than a piece of the path's step `step` takes (`Form::along` never takes
/// fewer than one substep a piece; below that a doubled rate would change nothing, and read as no error). The setting
/// with its parts and runs, and the first model's series at its runs (the calibration's).
fn settle(forms: &[&Form], walk: &Walk, piece: f64, step: f64) -> Result<(Settled, Vec<Series>), String> {
    let clock = Instant::now();
    let run = |s: Setting| -> Result<Vec<Series>, String> { forms.iter().map(|f| f.along(walk, &walk.start, 0.0, piece, scheme(s, 1))).collect() };
    let own = |s: Setting| Setting { rate: larger(s.rate, s.pieces as f64 / step), ..s };
    let mut setting = own(COARSEST);
    let mut at = run(setting)?;
    let work = |x: &[Series]| x.iter().map(|s| s.counts.products).sum::<u64>();
    let (mut tried, mut products) = (1, work(&at));
    loop {
        let (runs, pairs) = refinements(setting, &own);
        let made: Vec<Vec<Series>> = runs[1..].par_iter().map(|&s| run(s)).collect::<Result<_, _>>()?;
        tried += made.len();
        products += made.iter().map(|s| work(s)).sum::<u64>();
        let series: Vec<Vec<Series>> = std::iter::once(at).chain(made).collect();
        let errors = pairs.map(|(i, j)| RICHARDSON * change(&series[i], &series[j]));
        #[cfg(test)]
        if traced() {
            eprintln!("    [trace] settle {setting:?}: parts {:?}", errors.map(|e| format!("{e:.2e}")));
        }
        let open: [bool; KNOBS] = std::array::from_fn(|k| setting.open(k));
        if errors.iter().sum::<f64>() <= TOLERANCE || !open.iter().any(|&o| o) {
            let opening = series[0][0].counts.opening;
            let first = series.into_iter().map(|mut x| x.swap_remove(0)).collect();
            return Ok((Settled { setting, opening, errors, runs, pairs, tried, products, seconds: clock.elapsed().as_secs_f64() }, first));
        }
        let k = (0..KNOBS).filter(|&k| open[k]).max_by(|&a, &b| errors[a].total_cmp(&errors[b])).expect("an open knob");
        setting = runs[1 + k];
        at = series.into_iter().nth(1 + k).expect("the refinement");
    }
}

/// The largest change between two runs, along one path at the times both read: of a displayed row of one model (one
/// series each), or of a row's difference of the second model to the first (two).
fn change(a: &[Series], b: &[Series]) -> f64 {
    let at = |s: &Series, t: f64| s.readings.iter().find(|y| y.t == t).map(|y| y.rows);
    let mut e = 0.0;
    for x in &a[0].readings {
        let Some(y) = at(&b[0], x.t) else { continue };
        let (u, v) = match (a.get(1), b.get(1)) {
            (Some(p), Some(q)) => match (at(p, x.t), at(q, x.t)) {
                (Some(u), Some(v)) => (u, v),
                _ => continue,
            },
            _ => ([0.0; STATES], [0.0; STATES]),
        };
        for k in 0..STATES {
            e = larger(e, ((u[k] - x.rows[k]) - (v[k] - y[k])).abs());
        }
    }
    e
}

/// A scan's readings (module documentation, # The region left out): one every `SCAN_READ` of the pilot's steps.
pub const SCAN_READ: f64 = 4.0;

/// The scans' walks (module documentation, # The region left out): per principal axis of the trap and each way along
/// it, from the trap's centre at the speed `speed` (µm/µs) to where `beyond` first holds -- the pilot's most energetic
/// held atom's turning point, or the trap region's edge --, its knots `step` apart (µs: its readings), the atom's state at its start
/// `start`.
pub fn scan_walks(beyond: &dyn Fn(V3) -> bool, centre: V3, axes: &[V3; 3], speed: f64, step: f64, hbar_over_mass: f64, start: &[(usize, C64)]) -> Vec<Walk> {
    let mut walks = Vec::new();
    for (i, axis) in axes.iter().enumerate() {
        for sign in [1.0, -1.0] {
            let e = axis.map(|x| sign * x);
            let at = |d: f64| -> V3 { std::array::from_fn(|k| centre[k] + d * e[k]) };
            // The edge: the first distance where `beyond` holds, by doubling then bisection.
            let (mut lo, mut hi) = (0.0, step * speed);
            while !beyond(at(hi)) && hi < 1e6 {
                (lo, hi) = (hi, 2.0 * hi);
            }
            for _ in 0..60 {
                let mid = 0.5 * (lo + hi);
                if beyond(at(mid)) {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            let time = hi / speed;
            let count = (time / step).ceil().max(1.0) as usize;
            let kappa = e.map(|x| x * speed / hbar_over_mass);
            let ends = (0..=count)
                .map(|k| {
                    let t = time * k as f64 / count as f64;
                    let r = at(speed * t);
                    [t, r[0], r[1], r[2], kappa[0], kappa[1], kappa[2]]
                })
                .collect();
            walks.push(Walk { index: (2 * i + usize::from(sign < 0.0)) as u64, start: start.to_vec(), ends });
        }
    }
    walks
}

impl<'a> Paths<'a> {
    /// The reference's part (`Paths`): the walks `walks` (the pilot's, from their own starts, through the run) at their
    /// start, [0, L], L = min(`run`, 5 τ_mem) -- `memory` the memory time (`paths::memory`, the longest over the walks'
    /// starts) --, and at the run's end, [run − L, run], each from the walk's own start state (the late segment: its
    /// last window has forgotten that start), where the run is longer than twice L; the windows of their step `step`;
    /// the scheme's setting on the first walk's piece (`settle`), and the reference along the walks at it.
    pub fn new(reference: &'a Selected, at: V3, walks: Vec<Walk>, memory: f64, run: f64, step: f64, light: &Light) -> Result<Paths<'a>, String> {
        let clock = Instant::now();
        let form = Form::new(reference, at);
        let length = run.min(5.0 * memory);
        let windows = paths::windows(length, memory, step);
        let piece = CALIBRATION.min(length).min(walks[0].end());
        let late_from = (run > 2.0 * length).then_some(run - length);
        let (settled, calibrated) = settle(&[&form], &walks[0], piece, step)?;
        let scheme = settled.scheme(1);
        #[cfg(test)]
        if traced() {
            eprintln!("    [trace] reference form: {}; beats H {:?}, R {:?}", form.shape(), form.beat_values().0, form.beat_values().1);
            eprintln!("    [trace] walks: {:?} ends, step {step}, ends at {:?}", walks.iter().map(|w| w.ends.len()).collect::<Vec<_>>(), walks.iter().map(|w| w.end()).collect::<Vec<_>>());
        }
        #[cfg(test)]
        let pool = Instant::now();
        #[cfg(test)]
        let _ = crate::paths::freeze_parts();
        // One job a walk's start and its end; on the pool.
        let n = walks.len();
        let jobs: Vec<usize> = (0..n * (1 + usize::from(late_from.is_some()))).collect();
        let done: Vec<Result<Series, String>> = jobs
            .par_iter()
            .map(|&j| {
                #[cfg(test)]
                let clock = Instant::now();
                let out = if j < n {
                    let w = &walks[j];
                    form.along(w, &w.start, 0.0, length.min(w.end()), scheme)
                } else {
                    // A walk that ended before the late segment (its atom lost) reads nothing there.
                    let (w, from) = (&walks[j - n], late_from.expect("a late segment"));
                    if from < w.end() { form.along(w, &w.start, from, run.min(w.end()), scheme) } else { Ok(Series::default()) }
                };
                #[cfg(test)]
                if let Ok(s) = &out {
                    trace_job(&format!("reference walk {} {}", j % n, if j < n { "start" } else { "late" }), s, clock.elapsed().as_secs_f64());
                }
                out
            })
            .collect();
        #[cfg(test)]
        if traced() {
            eprintln!("    [trace] reference jobs: {:.2} s wall on {} threads; freezes' parts (nodes, amplitudes, eigen, weights, R and ΔH) {:.2?} s", pool.elapsed().as_secs_f64(), rayon::current_num_threads(), crate::paths::freeze_parts());
        }
        let mut done = done.into_iter();
        let series: Vec<Series> = (0..n).map(|_| done.next().expect("a walk")).collect::<Result<_, _>>()?;
        let late_series: Vec<Series> = done.collect::<Result<_, _>>()?;
        let late = late_from.map(|from| (from, windows.iter().map(|&(a, b)| (from + a, from + b)).collect(), late_series));
        let least = walks.iter().flat_map(|w| w.ends.iter().map(|e| intensity(light, [e[1], e[2], e[3]]))).fold(f64::INFINITY, crate::linalg::smaller);
        let products = series.iter().chain(late.iter().flat_map(|l| l.2.iter())).map(|s| s.counts.products).sum::<u64>() + settled.products;
        Ok(Paths { walks, length, memory, windows, reference: form, settled, series, calibration: (piece, calibrated), late, least, seconds: clock.elapsed().as_secs_f64(), products })
    }

    /// Where the late segment starts (µs), where there is one.
    pub fn late_from(&self) -> Option<f64> {
        self.late.as_ref().map(|l| l.0)
    }

    /// The reference judged against itself (`Judged`): every error none -- the 32 states are the reference.
    pub fn itself(&self, keep: Vec<usize>) -> Judged {
        let labels = &self.reference.model.scene.request.states;
        let hm = self.reference.model.hbar_over_mass();
        let windows = paths::compare(&self.series, &self.series, &self.windows, labels, hm);
        let late = self.late.as_ref().map_or(vec![], |(_, w, s)| paths::compare(s, s, w, labels, hm));
        let calibration = Calibration { rows: self.settled.unmet().unwrap_or(0.0), rate: 0.0 };
        Judged { keep, windows, late, calibration, region: Region::NONE, seconds: 0.0, products: 0 }
    }
}

/// The scheme's error on the fold's error (module documentation, # The paths): on the calibration's piece the largest
/// change of a row's difference to the reference under each refinement of the setting, read as its error part
/// (`RICHARDSON`), the two parts added -- booked beside the direct part --, and of the total rate's relative difference
/// alike (said).
#[derive(Clone, Copy, Debug, Default)]
pub struct Calibration {
    pub rows: f64,
    pub rate: f64,
}

/// The region left out (module documentation, # The region left out): {s ≤ `intensity`} where a scan found a row's
/// error of a quarter of its target or more beyond the paths' least intensity, `error` the largest there (at most one);
/// none flagged, the intensity −∞. Below the least intensity the scans and the paths reached, `edge`, nothing was read: a state there
/// is unknown, its error whole (one). `largest` the largest row's error the scans read beyond the paths' reach, flagged
/// or not.
#[derive(Clone, Copy, Debug)]
pub struct Region {
    pub intensity: f64,
    pub error: f64,
    pub edge: f64,
    pub largest: f64,
}

impl Region {
    /// No region: nothing flagged, nothing unread.
    pub const NONE: Region = Region { intensity: f64::NEG_INFINITY, error: 0.0, edge: f64::NEG_INFINITY, largest: 0.0 };

    /// The error of a place of the tweezer's intensity `s` on what is shown: one below the scans' edge, the region's in
    /// it, none outside (where it is some, the place's share moves T).
    pub fn error_at(&self, s: f64) -> Option<f64> {
        if s < self.edge {
            Some(1.0)
        } else if s <= self.intensity {
            Some(self.error)
        } else {
            None
        }
    }
}

/// One candidate judged along the paths (`judge`): its states, its windows' errors, its calibration, its region (the
/// scans', `scan`; none until scanned), and the work and time its judgment took.
#[derive(Clone, Debug)]
pub struct Judged {
    pub keep: Vec<usize>,
    pub windows: Vec<Window>,
    /// The late segment's windows (their times the run's), none where it has none.
    pub late: Vec<Window>,
    pub calibration: Calibration,
    pub region: Region,
    pub seconds: f64,
    pub products: u64,
}

/// A candidate model judged against the reference along the paths (`Judged`): its series along the walks at the paths'
/// setting and on the calibration's piece at the setting's two refinements, every job on the pool.
pub fn judge(paths: &Paths, model: &Selected, at: V3) -> Result<Judged, String> {
    let clock = Instant::now();
    let form = Form::new(model, at);
    let (piece, ref calibrated) = paths.calibration;
    let scheme = paths.settled.scheme(1);
    // The calibration's refinements (the setting's own series is the first walk's, cut to the piece).
    let refined: Vec<Scheme> = paths.settled.runs[1..].iter().map(|&s| self::scheme(s, 1)).collect();
    let r = refined.len();
    let w0 = &paths.walks[0];
    let n = paths.walks.len();
    let run = paths.late.as_ref().map_or(0.0, |(from, ..)| from + paths.length);
    let jobs: Vec<usize> = (0..r + n * (1 + usize::from(paths.late.is_some()))).collect();
    let done: Vec<Result<Series, String>> = jobs
        .par_iter()
        .map(|&j| {
            #[cfg(test)]
            let clock = Instant::now();
            let out = match j {
                j if j < r => form.along(w0, &w0.start, 0.0, piece, refined[j]),
                j if j < r + n => {
                    let w = &paths.walks[j - r];
                    form.along(w, &w.start, 0.0, paths.length.min(w.end()), scheme)
                }
                j => {
                    let (w, from) = (&paths.walks[j - r - n], paths.late.as_ref().expect("a late segment").0);
                    if from < w.end() { form.along(w, &w.start, from, run.min(w.end()), scheme) } else { Ok(Series::default()) }
                }
            };
            #[cfg(test)]
            if let Ok(s) = &out {
                trace_job(&format!("P{} job {j}", model.rs.keep.len()), s, clock.elapsed().as_secs_f64());
            }
            out
        })
        .collect();
    let mut done = done.into_iter();
    let finer: Vec<Series> = (0..r).map(|_| done.next().expect("a refinement")).collect::<Result<_, _>>()?;
    let series: Vec<Series> = (0..n).map(|_| done.next().expect("a walk")).collect::<Result<_, _>>()?;
    let late_series: Vec<Series> = done.collect::<Result<_, _>>()?;
    let labels = &model.scene.request.states;
    let hm = model.hbar_over_mass();
    let windows = paths::compare(&series, &paths.series, &paths.windows, labels, hm);
    let late = paths.late.as_ref().map_or(vec![], |(_, w, s)| paths::compare(&late_series, s, w, labels, hm));
    // The calibration: the first walk's own series at the setting on the piece (the same scheme from the same start)
    // and its two refinements, against the reference's.
    let cut = Series { readings: series[0].readings.iter().filter(|r| r.t <= piece).cloned().collect(), ..Series::default() };
    let candidate: Vec<&Series> = std::iter::once(&cut).chain(&finer).collect();
    let mut calibration = calibrate(&candidate, calibrated, &paths.settled.pairs);
    calibration.rows += paths.settled.unmet().unwrap_or(0.0);
    let products = series.iter().chain(&late_series).chain(&finer).map(|s| s.counts.products).sum();
    Ok(Judged { keep: model.rs.keep.clone(), windows, late, calibration, region: Region::NONE, seconds: clock.elapsed().as_secs_f64(), products })
}

/// The region left out of a candidate (module documentation, # The region left out; `Region`): each scan's scheme
/// settled on its own start by the error of what it reads (`settle`: the candidate's rows' difference to the
/// reference's), the candidate and the reference along the scans `scans`
/// at their own, read at every knot, every job on the pool;
/// beyond the paths' least intensity `least`, each reading's largest row error, a quarter of the target or more
/// marking. With the time it took.
pub fn scan(reference: &Selected, model: &Selected, at: V3, scans: &[Walk], least: f64, light: &Light) -> Result<(Region, Vec<Setting>, f64), String> {
    let clock = Instant::now();
    let forms = [Form::new(reference, at), Form::new(model, at)];
    // Each scan's setting on its own first `CALIBRATION` µs (its knots its steps).
    let settled: Vec<Settled> = scans
        .par_iter()
        .map(|w| {
            let step = w.ends.get(1).map_or(w.end(), |e| e[0] - w.ends[0][0]);
            settle(&[&forms[0], &forms[1]], w, CALIBRATION.min(w.end()), step).map(|x| x.0)
        })
        .collect::<Result<_, _>>()?;
    let jobs: Vec<(usize, usize)> = (0..2).flat_map(|m| (0..scans.len()).map(move |w| (m, w))).collect();
    #[cfg(test)]
    let _ = crate::paths::freeze_parts();
    #[cfg(test)]
    if traced() {
        for (k, w) in scans.iter().enumerate() {
            let e = w.ends.last().unwrap();
            eprintln!("    [trace] scan {k}: {:.2} µs, {} knots, end r {:?}, s {:.3e}", w.end(), w.ends.len(), [e[1], e[2], e[3]], intensity(light, [e[1], e[2], e[3]]));
        }
        eprintln!("    [trace] candidate form: {}; beats H {:?}, R {:?}", forms[1].shape(), forms[1].beat_values().0, forms[1].beat_values().1);
    }
    let done: Vec<Result<Series, String>> = jobs
        .par_iter()
        .map(|&(m, w)| {
            #[cfg(test)]
            let clock = Instant::now();
            #[cfg(test)]
            let fixed: Option<Setting> = std::env::var("SELCOST_SCAN_SETTING").ok().filter(|x| !x.is_empty()).map(|x| {
                let p: Vec<f64> = x.split('/').map(|v| v.parse().unwrap()).collect();
                Setting { turns: p[0], rate: p[1], pieces: p[2] as usize }
            });
            #[cfg(not(test))]
            let fixed: Option<Setting> = None;
            let out = forms[m].along(&scans[w], &scans[w].start, 0.0, scans[w].end(), fixed.map_or(settled[w].scheme(1), |s| scheme(s, 1)));
            #[cfg(test)]
            if let Ok(s) = &out {
                trace_job(&format!("scan {w} {}", if m == 0 { "reference" } else { "candidate" }), s, clock.elapsed().as_secs_f64());
            }
            out
        })
        .collect();
    #[cfg(test)]
    if traced() {
        eprintln!("    [trace] scans: freezes' parts (nodes, amplitudes, eigen, weights, R and ΔH) {:.2?} s", crate::paths::freeze_parts());
    }
    let done: Vec<Series> = done.into_iter().collect::<Result<_, _>>()?;
    let (reference, candidate) = done.split_at(scans.len());
    let quarter = 0.25 * POPULATION_TARGET;
    let mut region = Region::NONE;
    // Unread: below the least intensity both the scans and the paths reached (the paths read where they went).
    region.edge = scans.iter().filter_map(|w| w.ends.last()).map(|e| intensity(light, [e[1], e[2], e[3]])).fold(least, crate::linalg::smaller);
    for (_line, (r, c)) in reference.iter().zip(candidate).enumerate() {
        #[cfg(test)]
        let mut worst = (0.0_f64, 0.0_f64, 0.0_f64);
        for (c, r) in c.readings.iter().zip(&r.readings) {
            let s = intensity(light, c.r);
            if s >= least {
                continue;
            }
            let e = (0..STATES).map(|k| (c.rows[k] - r.rows[k]).abs()).fold(0.0, larger);
            #[cfg(test)]
            if e > worst.0 {
                worst = (e, c.t, s);
            }
            region.largest = larger(region.largest, e);
            if e >= quarter {
                region.intensity = larger(region.intensity, s);
                region.error = larger(region.error, e.min(1.0));
            }
        }
        #[cfg(test)]
        if traced() {
            eprintln!("    [trace] scan {_line}: largest error {:.3e} at {:.2} µs, s {:.4}", worst.0, worst.1, worst.2);
        }
    }
    Ok((region, settled.iter().map(|s| s.setting).collect(), clock.elapsed().as_secs_f64()))
}

/// The calibration (`Calibration`) of a candidate's series on the calibration's piece at the paths' runs
/// (`candidate`: the setting first, its refinements after), against the reference's alike (`reference`), read at the
/// times all read (the knots): per knob the largest change of a row's difference to the reference between its two runs
/// (`pairs`, `refinements`), and of the total rate's relative difference, read as its error part (`RICHARDSON`), the
/// parts added.
fn calibrate(candidate: &[&Series], reference: &[Series], pairs: &[(usize, usize); KNOBS]) -> Calibration {
    fn at(s: &Series, t: f64) -> Option<&paths::Reading> {
        s.readings.iter().find(|r| r.t == t)
    }
    let total = |r: &paths::Reading| r.rates.iter().sum::<f64>();
    let mut parts = [Calibration::default(); KNOBS];
    for x in &candidate[0].readings {
        for (part, &(i, j)) in parts.iter_mut().zip(pairs) {
            let (Some(a), Some(b), Some(u), Some(v)) = (at(candidate[i], x.t), at(&reference[i], x.t), at(candidate[j], x.t), at(&reference[j], x.t)) else { continue };
            for q in 0..STATES {
                part.rows = larger(part.rows, ((a.rows[q] - b.rows[q]) - (u.rows[q] - v.rows[q])).abs());
            }
            part.rate = larger(part.rate, ((total(a) / total(b) - 1.0) - (total(u) / total(v) - 1.0)).abs());
        }
    }
    Calibration { rows: RICHARDSON * parts.iter().map(|p| p.rows).sum::<f64>(), rate: RICHARDSON * parts.iter().map(|p| p.rate).sum::<f64>() }
}

// ---------------------------------------------------------------------------------------------------------------------
// The accounts.

/// Whether T's memory is not shorter than the run, of its memory at the run's end `r` (µK, on the momenta's-scaling
/// scale: T(0) moved by 2T(0); module documentation, # T's memory, read on the run's own trajectories) and its standard
/// error `se`, T's allowance there `allowance` (µK): yes where it is read there, |r| > z·se (FOLDTEMP §R2.3: "缩放响应
/// 在运行末仍可读出"); no where even its upper end |r| + z·se is within the allowance -- a unit kick of the initial
/// state's temperature moves the end's T by less than its target; none otherwise (no flag: the run's groups do not tell).
pub fn decide(r: f64, se: f64, allowance: f64) -> Option<bool> {
    if !(r.is_finite() && se.is_finite()) {
        None
    } else if r.abs() > CONFIDENCE * se {
        Some(true)
    } else if r.abs() + CONFIDENCE * se < allowance {
        Some(false)
    } else {
        None
    }
}

/// The direct part of a window: per row |mean| + z·se.
fn direct(w: &Window) -> Vec<f64> {
    w.rows.iter().map(|m| m.value.abs() + CONFIDENCE * m.error.abs()).collect()
}

/// A candidate's direct error at the time t (module documentation, # The direct part): the largest row's |mean| + z·se
/// of the time's window (`paths::window_of` on the start's windows `windows`), past L the larger of the start's last
/// window's and the late segment's last window's, past the late segment's start the latter's; the scheme's calibration
/// beside it. None at t = 0, where every model starts in the same state.
pub fn direct_at(judged: &Judged, windows: &[(f64, f64)], t: f64) -> f64 {
    if t <= 0.0 || judged.windows.is_empty() {
        return 0.0;
    }
    let length = windows.last().map_or(0.0, |w| w.1);
    let own = match (judged.late.first().map(|w| w.from), judged.windows.last(), judged.late.last()) {
        (Some(from), Some(last), Some(late_last)) if t > length => {
            let (a, b) = (direct(last), direct(late_last));
            if t > from { b } else { a.iter().zip(&b).map(|(x, y)| larger(*x, *y)).collect() }
        }
        _ => direct(&judged.windows[paths::window_of(windows, t).unwrap_or(0)]),
    };
    own.iter().copied().fold(0.0, larger) + judged.calibration.rows
}

/// What the choice made of the chain (solve.rs, # The selection): P, its chain, its judgment, the paths' windows, the
/// memory time and L, and the report's lines -- every candidate judged with its direct part, its cost and why it was or
/// was not chosen, the stages' times.
#[derive(Clone, Debug)]
pub struct Choice {
    pub keep: Vec<usize>,
    /// The chain the choice was made on: by the importance at the pilot's states (`importance`).
    pub chain: Vec<Vec<usize>>,
    pub judged: Judged,
    pub windows: Vec<(f64, f64)>,
    pub memory: f64,
    pub length: f64,
    pub report: Vec<String>,
}

impl Choice {
    /// P's direct error at the time t (`direct_at`): what is shown; T and the loss have none (module documentation, # The
    /// feedback through the motion: zero).
    pub fn at(&self, t: f64) -> f64 {
        direct_at(&self.judged, &self.windows, t)
    }
}

/// The key the choice is kept under (BRIEF_P3_SELECT item 6: the scene's physics): the request without its times --
/// its length, its outputs and the samples they make. The worker is one process for every request (main.rs), so a scene
/// solved again with other times takes its choice from here (`CHOICES`).
pub fn key(request: &crate::request::Request) -> String {
    let mut r = request.clone();
    r.times_us = Vec::new();
    format!("{r:?}")
}

#[cfg(test)]
thread_local! {
    /// Candidates beside the chain a test asks the choice to judge and weigh for comparison, never to choose (solve.rs
    /// `choose_on_the_pilot`): each its states.
    pub(crate) static ALSO: std::cell::RefCell<Vec<Vec<usize>>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// The choices made (`key`), in the order they were made.
static CHOICES: std::sync::Mutex<Vec<(String, std::sync::Arc<Choice>)>> = std::sync::Mutex::new(Vec::new());

/// The choice kept for a request (`key`), where one was made.
pub fn kept(request: &crate::request::Request) -> Option<std::sync::Arc<Choice>> {
    let key = key(request);
    CHOICES.lock().unwrap_or_else(|e| e.into_inner()).iter().find(|(k, _)| *k == key).map(|(_, c)| c.clone())
}

/// A choice kept for its request (`key`).
pub fn keep(request: &crate::request::Request, choice: std::sync::Arc<Choice>) {
    let key = key(request);
    let mut list = CHOICES.lock().unwrap_or_else(|e| e.into_inner());
    list.retain(|(k, _)| *k != key);
    list.push((key, choice));
}

#[cfg(test)]
pub(crate) mod tests;
