//! The two-point state selection (RUSTPLAN §1.1, its corrections at the end of RUSTPLAN.md): P from the ground states,
//! adding one whole level at a time in the order of its largest first-order amplitude at the two points
//! (scene2.ranking), the rest folded by RS (`model::rs`); P is the smallest whose model meets the gate against the full
//! reference at both points. The two points: the centre (tweezer at full intensity, v = 0) and the edge (tweezer
//! intensity 0, v = 0, the centre's lattice phase). The only place P is decided.
//!
//! # The gate
//! At each point, the model's readouts against the reference's (td_s.cmp with the force as a vector):
//! - populations by level, absolute: max_g |δp_g|;
//! - the scattering rate, relative: |δR| / R (every jump channel, the tweezer's folded ones included, in both);
//! - the force vector: |δF⃗| / max(|F⃗|, kR) -- the scale kR is the force a photon's momentum per scattering would give,
//!   so a vanishing F⃗ (a balanced point) is judged against the scattering it could produce, not against zero.
//!
//! Each ≤ `GATE` at both points passes. The diffusion is not judged (the user: out of the model's gate), so only ρ is
//! solved.
//!
//! # The decision
//! The references at both points, then the candidates in turn, each at both points at once (rayon); P the first passing.
//! A candidate beyond it is never solved: its model is larger than every one below it (a level more), and the largest
//! (every level kept, the reference's size) would cost the most time and memory of all (GMD1C: P = 8 passes, its
//! candidates run to 32). No candidate passing is an error, never an empty P.

use crate::floquet::{assemble, readouts, select, Frame, Liouville, Readouts};
use crate::krylov::Route;
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

/// One point's comparison of a candidate P's model with the full reference.
#[derive(Clone, Debug, PartialEq)]
pub struct PointError {
    /// The largest absolute population error over the levels, R's relative error, F⃗'s error |δF⃗| / max(|F⃗|, kR).
    pub populations: f64,
    pub rate: f64,
    pub force: f64,
    pub route: Route,
}

impl PointError {
    /// The largest of the three.
    pub fn worst(&self) -> f64 {
        self.populations.max(self.rate).max(self.force)
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
    pub fn passes(&self) -> bool {
        self.centre.worst() <= GATE && self.edge.worst() <= GATE
    }
}

/// The selection's report: the chosen P, the candidates up to it with their errors, the full reference's readouts and
/// routes, the time.
#[derive(Clone, Debug, PartialEq)]
pub struct Selection {
    pub keep: Vec<usize>,
    pub candidates: Vec<Candidate>,
    pub reference: [(Readouts, Route); 2],
    pub seconds: f64,
}

/// The model's errors against the reference at one point; `k` the photon wave number of the force's scale kR.
pub fn errors(model: &Readouts, reference: &Readouts, k: f64, route: Route) -> PointError {
    let populations =
        model.levels.iter().zip(&reference.levels).map(|(m, r)| (m - r).abs()).fold(0.0, f64::max);
    let rate = (model.rate - reference.rate).abs() / reference.rate;
    let norm = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let delta = [0, 1, 2].map(|i| model.force[i] - reference.force[i]);
    let force = norm(delta) / norm(reference.force).max(k * reference.rate);
    PointError { populations, rate, force, route }
}

/// The decision over ranked candidates (each `keep` a superset of the one before): `solve(point, None)` is the full
/// reference at a point, `solve(point, Some(c))` candidate c's model there, each giving the readouts and the route.
/// The references, then the candidates in turn, each at both points at once; P is the first candidate passing at both
/// points (module documentation, # The decision).
pub fn decide<F>(candidates: &[Vec<usize>], k: f64, solve: F) -> Result<Selection, String>
where
    F: Fn(usize, Option<usize>) -> Result<(Readouts, Route), String> + Sync,
{
    let clock = Instant::now();
    let pair =|c: Option<usize>| -> Result<Vec<(Readouts, Route)>, String> { (0..POINTS).into_par_iter().map(|p| solve(p, c)).collect() };
    let reference = pair(None)?;
    let mut tried: Vec<Candidate> = Vec::new();
    for (c, keep) in candidates.iter().enumerate() {
        let models = pair(Some(c))?;
        let point = |p: usize| errors(&models[p].0, &reference[p].0, k, models[p].1);
        let candidate = Candidate { keep: keep.clone(), centre: point(0), edge: point(1) };
        let passes = candidate.passes();
        tried.push(candidate);
        if passes {
            return Ok(Selection {
                keep: keep.clone(),
                candidates: tried,
                reference: [reference[0].clone(), reference[1].clone()],
                seconds: clock.elapsed().as_secs_f64(),
            });
        }
    }
    let largest = candidates.last().map_or(0, |k| k.len());
    Err(format!("no candidate P passes the {GATE} gate at both points (the largest has {largest} states)"))
}

/// The two points' components (module documentation): the centre, the tweezer full, and the edge, dark; no Doppler
/// shift, the focus' lattice phase.
pub fn points(scene: &Scene) -> [Components; POINTS] {
    let zero = vec![0.0; scene.request.beams.len()];
    [scene.components([0.0; 3], &zero, 1.0), scene.components([0.0; 3], &zero, 0.0)]
}

/// The candidate levels in the order of their largest first-order amplitude at the two points (scene2.ranking): per
/// excited state e the largest over the points of |V_eg| / |E_g + f − E_e + iΓ_e/2| over the ground states g and the
/// light other than the tweezer -- V the coupling of a frequency's beams together (its field: the beams of one
/// frequency index summed), at f and its conjugate at −f; E the reference's static energies at the point (the tweezer's
/// shift in), Γ its decay --; a level's the largest of its states'. Each level its states (indices into the 32), the
/// best first (a tie in the levels' order).
pub fn ranking(scene: &Scene, references: &[ModelAt], points: &[Components]) -> Vec<Vec<usize>> {
    let request = &scene.request;
    let tw = request.trap_beam;
    let mut score = [0.0_f64; STATES];
    for (reference, point) in references.iter().zip(points) {
        let still = &reference.still().value;
        let gamma = decay(&reference.jumps, STATES).0;
        let mut seen: Vec<&Vec<i32>> = Vec::new();
        for (l, beam) in request.beams.iter().enumerate().filter(|&(l, _)| l != tw) {
            if seen.contains(&&beam.frequency_index) {
                continue;
            }
            seen.push(&beam.frequency_index);
            let mut v = Mat::zeros(STATES, STATES);
            for m in (0..request.beams.len()).filter(|&m| m != tw && request.beams[m].frequency_index == beam.frequency_index) {
                let key = scene.unit(m);
                v.add_scaled(C64::ONE, &point.parts.iter().find(|c| c.key == key).expect("a beam's component").value);
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
/// rule's elements and the solve (floquet, krylov), its readouts and route.
fn solved(frame: &Frame, model: &ModelAt) -> Result<(Readouts, Route), String> {
    let liouville = Liouville::new(frame, &model.hilbert());
    let system = assemble(&liouville, &select(&liouville));
    let solved = crate::krylov::solve(&system).map_err(|e| format!("the checkpoint's system is inconsistent at {} rows", e.rows.len()))?;
    Ok((readouts(&liouville, &system, &solved.rho), solved.route))
}

/// Chooses P for a scene (module documentation): at the two points (`points`) the full reference (`rs::reference`) and
/// its frame (`Frame::new`: the reference there, no Doppler shift); the candidates the ground states and then one level
/// more at a time (`ranking`); each candidate's RS model at each point on that point's frame's sectors
/// (`Rs::at`); every solve the same (`solved`); then `decide`, the force's scale k the carrier's wave number (ν/c,
/// td_s's kref).
pub fn choose(scene: &Scene) -> Result<Selection, String> {
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
    decide(&candidates, k, |p, c| match c {
        None => solved(&frames[p], &references[p]),
        Some(c) => solved(&frames[p], &Rs::new(scene, &candidates[c], frames[p].sectors()).at(scene, &points[p])),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn readouts(levels: &[f64], rate: f64, force: [f64; 3]) -> Readouts {
        Readouts { levels: levels.to_vec(), rate, force, ..Default::default() }
    }

    /// The gate's three measures, by hand: populations 0.004 absolute, R 0.02 relative, F⃗ |(0.3, 0.4, 0)| = 0.5 over
    /// max(|F⃗| = 10, kR = 2·4 = 8) = 0.05; at a balanced point F⃗ = 0 the scale is kR = 8: |(0, 0, 0.08)| / 8 = 0.01.
    #[test]
    fn select_errors_by_hand() {
        let reference = readouts(&[0.5, 0.5], 4.0, [6.0, 8.0, 0.0]);
        let model = readouts(&[0.504, 0.496], 4.08, [6.3, 8.4, 0.0]);
        let e = errors(&model, &reference, 2.0, Route::Krylov);
        assert!((e.populations - 0.004).abs() < 1e-15 && (e.rate - 0.02).abs() < 1e-15 && (e.force - 0.05).abs() < 1e-15);
        let balanced = readouts(&[0.5, 0.5], 4.0, [0.0; 3]);
        let e = errors(&readouts(&[0.5, 0.5], 4.0, [0.0, 0.0, 0.08]), &balanced, 2.0, Route::Krylov);
        assert!((e.force - 0.01).abs() < 1e-15);
    }

    /// The decision takes the smallest candidate passing at both points, though a larger one passes too and a
    /// smaller one passes at one point only; the report keeps the candidates up to it. None passing is an error.
    #[test]
    fn select_decide_takes_the_smallest_passing() {
        let candidates = vec![vec![0], vec![0, 1], vec![0, 1, 2]];
        // R's relative error per (point, candidate): candidate 0 fails at the edge, 1 and 2 pass both
        let error = |p: usize, c: usize| [[0.001, 0.005, 0.0], [0.03, 0.009, 0.0]][p][c];
        let out = decide(&candidates, 1.0, |p, c| {
            Ok((readouts(&[1.0], 1.0 + c.map_or(0.0, |c| error(p, c)), [1.0, 0.0, 0.0]), Route::Krylov))
        })
        .unwrap();
        assert_eq!(out.keep, vec![0, 1]);
        assert_eq!(out.candidates.len(), 2);
        assert!(!out.candidates[0].passes() && out.candidates[1].passes());
        let none = decide(&candidates, 1.0, |_, c| {
            Ok((readouts(&[1.0], 1.0 + c.map_or(0.0, |_| 0.5), [1.0, 0.0, 0.0]), Route::Krylov))
        });
        assert!(none.is_err());
    }

    /// A level's name in the prototype's readouts (td_s: "g1", "g2", "e<line><F′>"), of its first state's label
    /// ("ground-F-m", "D<line>-F′-m").
    fn name(label: &str) -> String {
        let mut parts = label.splitn(3, '-');
        let (head, f) = (parts.next().unwrap(), parts.next().unwrap());
        if head == "ground" { format!("g{f}") } else { format!("e{}{f}", &head[1..]) }
    }

    /// `choose` is the prototype's selection (PYEXP2, scratchpad pyexp2/sel_<scene>.json: PYSEL's procedure on each scene's
    /// own request, reqprec/req, its exports pysel/export2) on the six scenes' requests (the directory `V3D_REQ`, the
    /// selections' `V3D_SEL`): the levels' ranking and P the same; the full reference's readouts at both points -- F⃗,
    /// R, the levels' populations, each as the gate reads it -- the prototype's to the solves' accuracy; and per candidate
    /// the gate's three errors, which differ by what the two models differ by (the prototype folds the model's fast
    /// elements in Liouville space, fold2; this one in Hilbert space by RS, model/rs.rs: the same second order, apart in
    /// the next, v3c2.status.md §2.3), said.
    ///
    /// Measured: the references' readouts within 3.4·10⁻¹³ on five scenes, the same element sets (GMD1C 4634 and 4428 as
    /// the prototype's); GMD1C's F⃗ 2.0·10⁻⁷ of its scale max(|F⃗|, kR) ≈ 6·10⁻⁴ (its force near balance), all of it the
    /// prototype's F_y: −1.2·10⁻¹⁰ at the centre where the scene's symmetry has none (here 5.7·10⁻¹⁵) -- the prototype's
    /// solve there agrees with its own direct LU to 10⁻¹¹ only (pyexp2.status.md, its condition). The bound read is 10⁻⁶
    /// of the gate's scale: four orders below the gate, the decisions the same where no error sits within it of 1 %.
    #[test]
    fn choose_is_pysel() {
        let dir = std::path::PathBuf::from(std::env::var("V3D_REQ").expect("V3D_REQ: the six scenes' requests (scratchpad reqprec/req)"));
        let sel = std::path::PathBuf::from(std::env::var("V3D_SEL").expect("V3D_SEL: the prototype's selections (scratchpad pyexp2)"));
        let json = |p: std::path::PathBuf| -> serde_json::Value { serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))).unwrap() };
        let mut worst = [0.0_f64; 2];
        for (file, scene_name) in [("tilt0.json", "tilt0"), ("tilt2.json", "tilt2"), ("R4_LIN_4.json", "r4"), ("GMD1C_1ms.json", "gmd1c"), ("PGC3D_1ms.json", "pgc3d"), ("TET_1ms.json", "tet")] {
            let mut line = json(dir.join(file));
            line["options"]["device"] = serde_json::json!("cpu");
            for form in line["model"]["motion"]["outsideForms"].as_array_mut().unwrap() {
                form.as_object_mut().unwrap().entry("transitions").or_insert(serde_json::json!([]));
            }
            let scene = Scene::new(&crate::request::Request::from_native(&line).unwrap()).unwrap();
            let python = json(sel.join(format!("sel_{scene_name}.json")));
            let labels = &scene.request.states;
            // The ranking.
            let points = points(&scene);
            let references: Vec<ModelAt> = points.iter().map(|p| rs::reference(&scene, p)).collect();
            let ours: Vec<String> = ranking(&scene, &references, &points).iter().map(|l| name(&labels[l[0]])).collect();
            let theirs: Vec<String> = python["ranking"].as_array().unwrap().iter().map(|r| format!("e{}{}", &r[0].as_str().unwrap()[1..], r[1])).collect();
            assert_eq!(ours, theirs, "{scene_name}: the ranking");
            // The selection.
            let selection = choose(&scene).unwrap();
            let p = python["P"].as_u64().unwrap() as usize;
            eprintln!("{scene_name}: P {} (prototype {p}), {} candidates, {:.2} s", selection.keep.len(), selection.candidates.len(), selection.seconds);
            assert_eq!(selection.keep.len(), p, "{scene_name}: P");
            // The full reference's readouts.
            let groups = levels(labels, &(0..STATES).collect::<Vec<_>>());
            for (i, point) in ["centre", "edge"].into_iter().enumerate() {
                let (ours, route) = &selection.reference[i];
                let theirs = &python["full"][point];
                let f: Vec<f64> = theirs["F"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
                let r = theirs["R"].as_f64().unwrap();
                let scale = (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt().max(selection_k(&scene) * r);
                let df = (0..3).map(|a| (ours.force[a] - f[a]).powi(2)).sum::<f64>().sqrt() / scale;
                let dr = (ours.rate - r).abs() / r;
                let dp = groups.iter().zip(&ours.levels).map(|(g, x)| (x - theirs["pops"][name(&labels[g[0]])].as_f64().unwrap()).abs()).fold(0.0, f64::max);
                eprintln!("  {point} reference ({route:?}): F⃗ {df:.1e} R {dr:.1e} populations {dp:.1e}");
                worst[0] = worst[0].max(df.max(dr).max(dp));
                assert!(df.max(dr).max(dp) <= 1e-6, "{scene_name} {point}: the reference's readouts");
            }
            // Each candidate's gate errors.
            for (c, step) in selection.candidates.iter().zip(python["steps"].as_array().unwrap()) {
                assert_eq!(c.keep.len() as u64, step["P"].as_u64().unwrap(), "{scene_name}: the candidates' sizes");
                for (point, ours) in [("centre", &c.centre), ("edge", &c.edge)] {
                    let e = &step["err"][point];
                    let (f, r, pop) = (e["F"].as_f64().unwrap(), e["R"].as_f64().unwrap(), e["P"].as_f64().unwrap());
                    eprintln!(
                        "  P {:2} {point}: F⃗ {:.4e} ({:.4e}) R {:.4e} ({:.4e}) populations {:.4e} ({:.4e}) -- ours (prototype's); {:?}",
                        c.keep.len(), ours.force, f, ours.rate, r, ours.populations, pop, ours.route
                    );
                    worst[1] = worst[1].max(((ours.force - f).abs()).max((ours.rate - r).abs()).max((ours.populations - pop).abs()));
                }
            }
        }
        eprintln!("largest: the reference's readouts {:.1e}; the gate's errors apart by {:.1e}", worst[0], worst[1]);
    }

    fn selection_k(scene: &Scene) -> f64 {
        scene.generators.last().unwrap().frequency / SPEED_OF_LIGHT
    }
}
