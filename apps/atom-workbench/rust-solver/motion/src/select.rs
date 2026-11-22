//! The selection of P (PIPELINE_V3_FINAL §二 A.3, A.5–A.6, B.7–B.11, C.13–C.15): the candidates by structure, each
//! judged by what it does to the run's readings along the atoms' own paths against the 32-state model (paths.rs), and
//! the candidate of least total work chosen among those whose model error A leaves the targets room.
//! The trajectories' parts (the pilot the paths come from, the model-only scaling response, the work) are the
//! solve's (solve.rs, # The selection); what is here is the model's side and the accounts.
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
//! (paths.rs `Form::along`), by Lawson's scheme in H's eigenbasis (`LAWSON`: P1-PATHS R2), and their readings compared
//! in windows of the memory time (paths.rs `windows`, `compare`). A path is read at its start for L = min(T, 5 τ_mem),
//! τ_mem the slowest decay of the full-ground model's generator, the longest over the paths' starts (`memory`; P1 §5: it
//! differs with the place by up to 2.4 times) -- the start's transient and the internal state's forgetting of it --, and,
//! where the run is longer than 2L, again over its last L from the walk's own start state (`Paths::late_from`): its last
//! window reads the model where the atoms are late in the run, which the start's windows cannot (the motion's own
//! memory is the longer: R4's atoms heat through the run). The scheme's own error on the fold's error is read once per
//! candidate against the converged `Scheme::sparse` on the first path's first `CALIBRATION` µs (`Calibration`), and
//! the rows' booked beside the direct part.
//!
//! The reference is the 32-state model itself (B.8), not a whole-level base: under Lawson its path costs 10–20 % more
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
//! ledger's item, solve.rs `NO_FEEDBACK`). The scaling response is kept for one flag alone, T's memory not shorter than
//! the run (`Response::collapsed`, solve.rs `response`).
//!
//! # The region left out
//! The chosen candidate is scanned along the prepared trap's three principal axes, both ways, from its centre at the
//! pilot's thermal speed against the reference (`scan`, `SCANNING`), each scan to the turning point of the pilot's most
//! energetic held atom on the reference state's prepared potential, or to the trap region's edge (`REGION_INTENSITY` of
//! the tweezer's peak), whichever is nearer -- the reach the pilot's atoms show (solve.rs `choose_on_the_pilot`): where
//! the paths did not go -- the tweezer's intensity s below the least the paths reached --, the readings where a row errs
//! by a quarter of its target or more mark the region {s ≤ s*}, s* the most intense of them, its error e the largest
//! row's error there; below the least intensity the scans and the paths reached nothing is read, and an atom there is unknown, its
//! error whole (`Region`). The run books every atom in either (solve.rs `booked_at`): a share f of them errs by at most e
//! (one) on what is shown and moves T by the bound set's `Estimate::moved(f)`. (Only the chosen one: a scan of the
//! 32-state reference to the trap's region's edge took R4 50 s at its pilot's step, the candidates' own a minute more.)
use crate::field::Light;
use crate::linalg::{larger, symmetric3_eigen, C64, Mat, V3};
use crate::model::rs::{self, decay, ModelAt};
use crate::model::selected::Selected;
use crate::paths::{self, Form, Scheme, Series, Walk, Window};
use crate::request::{GROUND, STATES};
use crate::scene::{Components, Scene};
use crate::step::Dynamics;
use crate::units::{CONFIDENCE, POPULATION_TARGET};
use rayon::prelude::*;
use std::time::Instant;

/// The points the structural amplitudes are read at: the centre (0), the edge (1), the middle (2) (module
/// documentation, # The candidates).
pub const POINTS: usize = 3;

/// The pilot's paths a candidate is judged on (BRIEF_P3_SELECT item 2: P1-PATHS's path-to-path spread is at most 2e-3,
/// one or two paths reach a quarter of the target, four cover the start's draws).
pub const WALKS: usize = 4;

/// The path scheme (P1-PATHS R2): each step of the path in two pieces, H exact in its eigenbasis there, the recycling
/// and the beats by RK4 in 128 substeps a µs (set by the decay, about 3.4 a lifetime of the D lines; P1's 64 erred by
/// 3e-3, 128 by 2.4e-4 on R4). Its error on the fold's error is read and booked (`Calibration`).
pub const LAWSON: Scheme = Scheme::Lawson { per_us: 128.0, pieces: 2, read: 1 };

/// The calibration's piece (µs): the first path's start, where the readings change fastest (P1-PATHS: one path, 10 µs).
pub const CALIBRATION: f64 = 10.0;

/// The momenta's relative kick of the scaling response (PATHGAP's, measured on PGC3D, TET, R4 and tilt0): its symmetric
/// difference errs by O(η²).
pub const KICK: f64 = 0.25;

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
/// reference's form and its series along them by `LAWSON` and on the calibration's piece by `Scheme::sparse`, the least
/// tweezer intensity the walks reached, the memory time, and the work it took.
pub struct Paths<'a> {
    pub walks: Vec<Walk>,
    pub length: f64,
    pub memory: f64,
    pub windows: Vec<(f64, f64)>,
    pub reference: Form<'a>,
    series: Vec<Series>,
    calibration: (f64, Series),
    /// The late segment (module documentation, # The paths): its start, its windows and the reference's series there;
    /// none where the run is no longer than twice L.
    late: Option<(f64, Vec<(f64, f64)>, Vec<Series>)>,
    pub least: f64,
    pub seconds: f64,
    pub products: u64,
}

/// The scans' scheme (module documentation, # The region left out): `LAWSON` in one piece a step (P1-PATHS R2.1: on R4
/// one piece and two err alike on the populations, 2.4e-4 and 2.5e-4, a fiftieth of a quarter of the target the scans
/// mark at), read at every fourth step's end.
pub const SCANNING: Scheme = Scheme::Lawson { per_us: 128.0, pieces: 1, read: 4 };

/// The scans' walks (module documentation, # The region left out): per principal axis of the trap and each way along
/// it, from the trap's centre at the speed `speed` (µm/µs) to where `beyond` first holds -- the pilot's most energetic
/// held atom's turning point, or the trap region's edge --, its knots `step` apart (µs), the atom's state at its start
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
    /// the reference along them and on the calibration's piece.
    pub fn new(reference: &'a Selected, at: V3, walks: Vec<Walk>, memory: f64, run: f64, step: f64, light: &Light) -> Result<Paths<'a>, String> {
        let clock = Instant::now();
        let form = Form::new(reference, at);
        let length = run.min(5.0 * memory);
        let windows = paths::windows(length, memory, step);
        let piece = CALIBRATION.min(length);
        let late_from = (run > 2.0 * length).then_some(run - length);
        // One job the calibration, a walk's start and its end; on the pool.
        let n = walks.len();
        let jobs: Vec<usize> = (0..1 + n * (1 + usize::from(late_from.is_some()))).collect();
        let done: Vec<Result<Series, String>> = jobs
            .par_iter()
            .map(|&j| {
                if j == 0 {
                    form.along(&walks[0], &walks[0].start, 0.0, piece.min(walks[0].end()), Scheme::sparse(4))
                } else if j <= n {
                    let w = &walks[j - 1];
                    form.along(w, &w.start, 0.0, length.min(w.end()), LAWSON)
                } else {
                    // A walk that ended before the late segment (its atom lost) reads nothing there.
                    let (w, from) = (&walks[j - 1 - n], late_from.expect("a late segment"));
                    if from < w.end() { form.along(w, &w.start, from, run.min(w.end()), LAWSON) } else { Ok(Series::default()) }
                }
            })
            .collect();
        let mut done = done.into_iter();
        let calibration = (piece, done.next().expect("the calibration")?);
        let series: Vec<Series> = (0..n).map(|_| done.next().expect("a walk")).collect::<Result<_, _>>()?;
        let late_series: Vec<Series> = done.collect::<Result<_, _>>()?;
        let late = late_from.map(|from| (from, windows.iter().map(|&(a, b)| (from + a, from + b)).collect(), late_series));
        let least = walks.iter().flat_map(|w| w.ends.iter().map(|e| intensity(light, [e[1], e[2], e[3]]))).fold(f64::INFINITY, crate::linalg::smaller);
        let products = series.iter().chain(late.iter().flat_map(|l| l.2.iter())).chain(std::iter::once(&calibration.1)).map(|s| s.counts.products).sum();
        Ok(Paths { walks, length, memory, windows, reference: form, series, calibration, late, least, seconds: clock.elapsed().as_secs_f64(), products })
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
        Judged { keep, windows, late, calibration: Calibration::default(), region: Region::NONE, seconds: 0.0, products: 0 }
    }
}

/// The scheme's error on the fold's error (module documentation, # The paths): on the calibration's piece the largest
/// change, between `LAWSON` and the converged `Scheme::sparse`, of a row's difference to the reference -- booked beside
/// the direct part -- and of the total rate's relative difference (said).
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

/// A candidate model judged against the reference along the paths (`Judged`): its series along the walks by `LAWSON`
/// and on the calibration's piece by `Scheme::sparse`, every job on the pool.
pub fn judge(paths: &Paths, model: &Selected, at: V3) -> Result<Judged, String> {
    let clock = Instant::now();
    let form = Form::new(model, at);
    let (piece, ref calibrated) = paths.calibration;
    let w0 = &paths.walks[0];
    let n = paths.walks.len();
    let run = paths.late.as_ref().map_or(0.0, |(from, ..)| from + paths.length);
    let jobs: Vec<usize> = (0..1 + n * (1 + usize::from(paths.late.is_some()))).collect();
    let done: Vec<Result<Series, String>> = jobs
        .par_iter()
        .map(|&j| match j {
            0 => form.along(w0, &w0.start, 0.0, piece.min(w0.end()), Scheme::sparse(4)),
            j if j <= n => {
                let w = &paths.walks[j - 1];
                form.along(w, &w.start, 0.0, paths.length.min(w.end()), LAWSON)
            }
            j => {
                let (w, from) = (&paths.walks[j - 1 - n], paths.late.as_ref().expect("a late segment").0);
                if from < w.end() { form.along(w, &w.start, from, run.min(w.end()), LAWSON) } else { Ok(Series::default()) }
            }
        })
        .collect();
    let mut done = done.into_iter();
    let sparse = done.next().expect("the calibration")?;
    let series: Vec<Series> = (0..n).map(|_| done.next().expect("a walk")).collect::<Result<_, _>>()?;
    let late_series: Vec<Series> = done.collect::<Result<_, _>>()?;
    let labels = &model.scene.request.states;
    let hm = model.hbar_over_mass();
    let windows = paths::compare(&series, &paths.series, &paths.windows, labels, hm);
    let late = paths.late.as_ref().map_or(vec![], |(_, w, s)| paths::compare(&late_series, s, w, labels, hm));
    // The calibration: the converged scheme's difference to the reference against Lawson's, on the same piece -- the
    // first walk's own Lawson series there (the same scheme from the same start), the candidate's and the reference's.
    let cut = |s: &Series| Series { readings: s.readings.iter().filter(|r| r.t <= piece).cloned().collect(), ..Series::default() };
    let calibration = calibrate(&cut(&series[0]), &cut(&paths.series[0]), &sparse, calibrated);
    let products = series.iter().chain(&late_series).chain([&sparse]).map(|s| s.counts.products).sum();
    Ok(Judged { keep: model.rs.keep.clone(), windows, late, calibration, region: Region::NONE, seconds: clock.elapsed().as_secs_f64(), products })
}

/// The region left out of a candidate (module documentation, # The region left out; `Region`): the candidate and the
/// reference along the scans `scans` by `SCANNING`, every job on the pool; beyond the paths' least intensity `least`,
/// each reading's largest row error, a quarter of the target or more marking. With the time it took.
pub fn scan(reference: &Selected, model: &Selected, at: V3, scans: &[Walk], least: f64, light: &Light) -> Result<(Region, f64), String> {
    let clock = Instant::now();
    let forms = [Form::new(reference, at), Form::new(model, at)];
    let jobs: Vec<(usize, usize)> = (0..2).flat_map(|m| (0..scans.len()).map(move |w| (m, w))).collect();
    let done: Vec<Result<Series, String>> = jobs.par_iter().map(|&(m, w)| forms[m].along(&scans[w], &scans[w].start, 0.0, scans[w].end(), SCANNING)).collect();
    let done: Vec<Series> = done.into_iter().collect::<Result<_, _>>()?;
    let (reference, candidate) = done.split_at(scans.len());
    let quarter = 0.25 * POPULATION_TARGET;
    let mut region = Region::NONE;
    // Unread: below the least intensity both the scans and the paths reached (the paths read where they went).
    region.edge = scans.iter().filter_map(|w| w.ends.last()).map(|e| intensity(light, [e[1], e[2], e[3]])).fold(least, crate::linalg::smaller);
    for (r, c) in reference.iter().zip(candidate) {
        for (c, r) in c.readings.iter().zip(&r.readings) {
            let s = intensity(light, c.r);
            if s >= least {
                continue;
            }
            let e = (0..STATES).map(|k| (c.rows[k] - r.rows[k]).abs()).fold(0.0, larger);
            region.largest = larger(region.largest, e);
            if e >= quarter {
                region.intensity = larger(region.intensity, s);
                region.error = larger(region.error, e.min(1.0));
            }
        }
    }
    Ok((region, clock.elapsed().as_secs_f64()))
}

/// The calibration (`Calibration`) of a candidate's Lawson series `lawson` and the reference's `reference` against their
/// converged series `sparse` and `converged` on the same piece, read at the times both read (the knots).
fn calibrate(lawson: &Series, reference: &Series, sparse: &Series, converged: &Series) -> Calibration {
    let mut c = Calibration::default();
    let at = |s: &Series, t: f64| s.readings.iter().find(|r| r.t == t).cloned();
    for x in &lawson.readings {
        let (Some(y), Some(u), Some(v)) = (at(reference, x.t), at(sparse, x.t), at(converged, x.t)) else { continue };
        let (y, u, v) = (&y, &u, &v);
        for k in 0..STATES {
            c.rows = larger(c.rows, ((x.rows[k] - y.rows[k]) - (u.rows[k] - v.rows[k])).abs());
        }
        let total = |r: &paths::Reading| r.rates.iter().sum::<f64>();
        c.rate = larger(c.rate, ((total(x) / total(y) - 1.0) - (total(u) / total(v) - 1.0)).abs());
    }
    c
}

// ---------------------------------------------------------------------------------------------------------------------
// The accounts.

/// The model-only scaling response (module documentation, # The feedback through the motion: zero): at the times `times` (µs, from t = 0)
/// per reading its R_O = (O₊ − O₋)/2η of the model's own atoms with their initial momenta scaled by 1 ± η -- the loss,
/// each row as shown, the total T (µK) --, the group mean; ⟨κ²⟩ = Σ_a⟨κ_a²⟩ at the kick; and T's response's own
/// statistical interval at the last time (whether it is still read there: the memory not shorter than the run).
#[derive(Clone, Debug)]
pub struct Response {
    pub times: Vec<f64>,
    pub loss: Vec<f64>,
    pub rows: Vec<[f64; STATES]>,
    pub temperature: Vec<f64>,
    /// Each one's standard error (the groups' jackknife), at the same times.
    pub errors: Vec<[f64; STATES + 2]>,
    pub kappa2: f64,
    pub last_interval: f64,
}

impl Response {
    /// The size |R| of the reading q (0 the loss, 1 + s the row s, 1 + `STATES` T) at the k-th time, its noise taken out:
    /// √max(R̄² − se², 0), R̄² being R² + se² on average -- a response the noise hides reads none, where |R̄| would read
    /// the noise (88 atoms: a row's ±η difference has a noise of about 0.09 a unit η, PGC3D's rows' |R̄| integrated to
    /// 71 µs over 1 ms of it).
    fn size(&self, q: usize, k: usize) -> f64 {
        let r = if q == 0 { self.loss[k] } else if q <= STATES { self.rows[k][q - 1] } else { self.temperature[k] };
        let e = self.errors[k][q];
        let d = r * r - if e.is_finite() { e * e } else { 0.0 };
        if d > 0.0 { d.sqrt() } else { 0.0 }
    }

    /// ∫₀ᵗ |R(τ)| dτ of one reading's response size `r` (`size`) at the response's times (the trapezoid; beyond its last
    /// time R held at its last value).
    fn integral(&self, r: &dyn Fn(usize) -> f64, t: f64) -> f64 {
        let mut sum = 0.0;
        for k in 1..self.times.len() {
            let (a, b) = (self.times[k - 1], self.times[k]);
            if a >= t {
                return sum;
            }
            let (ra, rb) = (r(k - 1).abs(), r(k).abs());
            if b <= t {
                sum += 0.5 * (b - a) * (ra + rb);
            } else {
                let rt = ra + (rb - ra) * (t - a) / (b - a);
                return sum + 0.5 * (t - a) * (ra + rt);
            }
        }
        let last = self.times.len() - 1;
        sum + (t - self.times[last]).max(0.0) * r(last).abs()
    }

    /// Λ_T = ∫₀ᵗ|R_T| / R_T(0): T's memory time (µs) to the time t.
    pub fn memory(&self, t: f64) -> f64 {
        self.integral(&|k| self.size(1 + STATES, k), t) / self.temperature[0].abs()
    }

    /// The rows' and the loss's ∫₀ᵗ|R| (µs per unit η): the largest row's and the loss's -- the feedback bound's
    /// weights on what is shown, for the report.
    pub fn shown_weights(&self, t: f64) -> [f64; 2] {
        [(0..STATES).map(|s| self.integral(&|k| self.size(1 + s, k), t)).fold(0.0, larger), self.integral(&|k| self.size(0, k), t)]
    }

    /// Whether T's response is still read at its last time, beyond its interval: no memory shorter than the run (the
    /// flag "Λ 塌掉", FOLDTEMP §R2.3).
    pub fn collapsed(&self) -> bool {
        self.temperature.last().is_some_and(|r| r.abs() > self.last_interval)
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

/// What the choice made of the chain (solve.rs, # The selection): P, its chain, its judgment, the response (its flag
/// alone), the paths' windows, the memory time and L, and the report's lines -- every candidate judged with its direct
/// part, its cost and why it was or was not chosen, the stages' times.
#[derive(Clone, Debug)]
pub struct Choice {
    pub keep: Vec<usize>,
    /// The chain the choice was made on: by the importance at the pilot's states (`importance`).
    pub chain: Vec<Vec<usize>>,
    pub judged: Judged,
    pub response: Response,
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
