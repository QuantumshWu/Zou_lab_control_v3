//! The two-point state selection (RUSTPLAN §1.1): P from the ground states, adding one whole level at a time in the order
//! of its largest first-order amplitude at the two points (scene2.ranking), the rest folded by RS (`model::rs`); P is the
//! smallest whose model meets the 1 % gate against the full reference at both points -- populations by level (absolute),
//! R (relative), F (|δF| / max(|F|, kR)); D not judged. The two points: the centre (tweezer at full intensity, v = 0) and
//! the edge (tweezer intensity 0, v = 0, the centre's lattice phase); the analysis axis z. The only place P is decided.

use crate::floquet::Readouts;
use crate::krylov::Route;
use crate::scene::Scene;

/// One point's comparison of a candidate P's model with the full reference.
#[derive(Clone, Debug, PartialEq)]
pub struct PointError {
    /// The largest absolute population error over the levels, R's relative error, F's error |δF| / max(|F|, kR).
    pub populations: f64,
    pub rate: f64,
    pub force: f64,
    /// D's relative error (reported, not judged).
    pub diffusion: f64,
    pub route: Route,
}

/// One candidate P: its states and its errors at the centre and the edge.
#[derive(Clone, Debug, PartialEq)]
pub struct Candidate {
    pub keep: Vec<usize>,
    pub centre: PointError,
    pub edge: PointError,
}

/// The selection's report: the chosen P, every candidate's errors, the full reference's readouts and routes, the time.
#[derive(Clone, Debug, PartialEq)]
pub struct Selection {
    pub keep: Vec<usize>,
    pub candidates: Vec<Candidate>,
    pub reference: [(Readouts, Route); 2],
    pub seconds: f64,
}

/// Chooses P for a scene.
pub fn choose(scene: &Scene) -> Result<Selection, String> {
    let _ = scene;
    todo!()
}
