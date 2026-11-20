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
//! The references and every candidate's models at both points are independent solves: all run at once (rayon), and
//! the smallest passing candidate is P -- the same P as adding levels one by one and stopping at the first pass, for
//! the wall clock of the slowest solve instead of their sum. No candidate passing is an error, never an empty P.

use crate::floquet::Readouts;
use crate::krylov::Route;
use crate::scene::Scene;
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
/// All solves run at once; P is the first candidate passing at both points.
pub fn decide<F>(candidates: &[Vec<usize>], k: f64, solve: F) -> Result<Selection, String>
where
    F: Fn(usize, Option<usize>) -> Result<(Readouts, Route), String> + Sync,
{
    let clock = Instant::now();
    let jobs: Vec<(usize, Option<usize>)> = (0..POINTS)
        .flat_map(|p| std::iter::once((p, None)).chain((0..candidates.len()).map(move |c| (p, Some(c)))))
        .collect();
    let solved: Vec<Result<(Readouts, Route), String>> = jobs.par_iter().map(|&(p, c)| solve(p, c)).collect();
    let mut table: Vec<Vec<(Readouts, Route)>> = vec![Vec::new(); POINTS];
    for ((p, _), r) in jobs.iter().zip(solved) {
        table[*p].push(r?);
    }
    let point = |p: usize, c: usize| {
        let (model, route) = &table[p][c + 1];
        errors(model, &table[p][0].0, k, *route)
    };
    let all: Vec<Candidate> = candidates
        .iter()
        .enumerate()
        .map(|(c, keep)| Candidate { keep: keep.clone(), centre: point(0, c), edge: point(1, c) })
        .collect();
    let chosen = all.iter().position(|c| c.passes()).ok_or_else(|| {
        let largest = candidates.last().map_or(0, |k| k.len());
        format!("no candidate P passes the {GATE} gate at both points (the largest has {largest} states)")
    })?;
    let reference = [table[0][0].clone(), table[1][0].clone()];
    Ok(Selection {
        keep: all[chosen].keep.clone(),
        candidates: all[..=chosen].to_vec(),
        reference,
        seconds: clock.elapsed().as_secs_f64(),
    })
}

/// Chooses P for a scene: the ranking and each point's reference and models (`Scene::components`, `rs::reference`,
/// `rs::fold`, `floquet`, `krylov::solve`), then `decide`. Waits for `model::rs` (V3C).
pub fn choose(scene: &Scene) -> Result<Selection, String> {
    let _ = scene;
    todo!()
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
}
