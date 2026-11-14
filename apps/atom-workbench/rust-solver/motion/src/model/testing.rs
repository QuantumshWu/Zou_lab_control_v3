//! What the model's tests share (compiled with the tests alone): the reference scenes' requests, the units that fold a
//! light's drives at the atom, a scene's model closed as the solve closes it, a plan's near forms from its table or the
//! exact elimination, the generator's side of the hazard's identity, and normal deviates.
use super::assemble::{NearValues, Plan};
use super::tabulate::NearTable;
use super::{Component, Model, Reach, Trap, Unit, GROUND};
use crate::field::{Amplitude, Light};
use crate::linalg::{inner, norm_sqr, Mat, C64};
use crate::random::Stream;
use crate::request::Request;

/// A scene's request through the one parser: a reference scene by name (refs/scenes, the app's native request), or any
/// native request file by its path.
pub(crate) fn scene(name: &str) -> Request {
    let path = if name.ends_with(".json") { std::path::PathBuf::from(name) } else { std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("scenes").join(format!("{name}.json")) };
    let line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    Request::from_native(&line).unwrap()
}

/// The units whose keeping folds every drive of the co-rotating components of the frequencies `of` with the levels at the
/// atom (`Unit::Bare`: kept, a drive's fold takes the levels there).
pub(crate) fn at_the_atom(request: &Request, light: &Light, of: &[usize]) -> Vec<Unit> {
    let bare = Model::new(request, light, &[]).unwrap();
    of.iter().flat_map(|&frequency| bare.bare_units(Component { frequency, counter: false })).collect()
}

/// The rate at which a generator H lowers a state's norm, ⟨ψ|i(H − H†)|ψ⟩/‖ψ‖²: by the decay's construction (photon.rs)
/// the hazard, Σ_φ Σ_a ‖J_{a,φ}ψ‖²/‖ψ‖² of the photons. The anti-Hermitian part i(H − H†) is formed entry by entry before
/// it meets ψ, so that the Hermitian part's scale (the states' energies) does not round the rate.
pub(crate) fn decay_rate(h: &Mat, psi: &[C64]) -> f64 {
    let decay = Mat::from_fn(h.rows, h.cols, |x, y| C64::I * (h[(x, y)] - h[(y, x)].conj()));
    inner(psi, &decay.apply(psi)).re / norm_sqr(psi)
}

/// A standard normal deviate from two uniform ones (Box–Muller).
pub(crate) fn normal(stream: &mut Stream) -> f64 {
    let (u, v) = (stream.uniform(), stream.uniform());
    (-2.0 * (1.0 - u).ln()).sqrt() * (std::f64::consts::TAU * v).cos()
}

impl Model {
    /// The units that fold a component's drives with the bare levels (`Unit::Bare`), whatever their level in this model:
    /// kept, each of them is folded with the levels at the atom.
    pub(crate) fn bare_units(&self, component: Component) -> Vec<Unit> {
        self.driven.iter().filter(|d| d.0 == component).map(|&(component, excited, ground)| Unit::Bare { component, excited, ground }).collect()
    }

    /// A scene's model as the solve closes it (solve.rs `closed`: the prior at the atoms of the reach of `temperature_uk`,
    /// the ground state `state` the initial state) from the units `kept`: the model, the units it keeps, its prepared trap
    /// and its reach for that state.
    pub(crate) fn closed_at(request: &Request, light: &Light, mut kept: Vec<Unit>, state: usize, temperature_uk: f64) -> (Model, Vec<Unit>, Trap, Reach) {
        let model = crate::solve::closed(request, light, &mut kept, &[(state, 1.0)], temperature_uk).expect("the closed model");
        let reference: [f64; GROUND] = std::array::from_fn(|s| if s == state { 1.0 } else { 0.0 });
        let trap = model.trap(&reference, &model.prepared()).unwrap();
        let reach = model.reach(&trap, &reference, temperature_uk, crate::estimate::unseen(crate::units::MOST_TRAJECTORIES as f64));
        (model, kept, trap, reach)
    }

    /// A plan's near forms at a point, the sections asked each to its order (0 or 1): from its table where one is given and
    /// reads there (tabulate.rs `Model::near_tabulated`), else the exact elimination's to the largest order (assemble.rs
    /// `Model::near_exact`) -- as the run's middles take them.
    pub(crate) fn near_values(&self, plan: &Plan, amplitudes: &[Amplitude], sections: &[(std::ops::Range<usize>, usize)], table: Option<&NearTable>) -> NearValues {
        table.and_then(|table| self.near_tabulated(table, amplitudes, sections)).unwrap_or_else(|| self.near_exact(plan, amplitudes, sections.iter().map(|s| s.1).max().unwrap_or(0)))
    }
}
