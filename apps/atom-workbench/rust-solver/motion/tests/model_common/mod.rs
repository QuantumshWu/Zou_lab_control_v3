//! What the model's tests share (tests/model_*.rs): the scenes' requests, the units that fold a light's drives at the atom, a
//! model closed by its units' measured errors, the generator's side of the hazard's identity, and normal deviates.
#![allow(dead_code)]
use atom_motion::field::Light;
use atom_motion::linalg::{inner, norm_sqr, Mat, C64};
use atom_motion::estimate::unseen;
use atom_motion::model::{Component, Model, Point, Reach, Sample, Trap, Unit, GROUND};
use atom_motion::units::MOST_TRAJECTORIES;
use atom_motion::random::Stream;
use atom_motion::request::Request;
use serde_json::Value;
use std::path::PathBuf;

/// A scene's request through the one parser: a reference scene by name (refs/scenes, the app's native request), or any native
/// request file by its path.
pub fn scene(name: &str) -> Request {
    let path = if name.ends_with(".json") { PathBuf::from(name) } else { PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("scenes").join(format!("{name}.json")) };
    let line: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    Request::from_native(&line).unwrap()
}

/// The units whose keeping folds every drive of the co-rotating components of the frequencies `of` with the levels at the atom
/// (`Unit::Bare`: kept, a drive's fold takes the levels there).
pub fn at_the_atom(request: &Request, light: &Light, of: &[usize]) -> Vec<Unit> {
    let bare = Model::new(request, light, &[]).unwrap();
    of.iter().flat_map(|&frequency| bare.bare_units(Component { frequency, counter: false })).collect()
}

/// A scene's model closed by its units' errors at its reference state (the solve keeps by the errors measured at its
/// pilot's states, solve.rs, # The model): from `kept`, every folded unit -- the counter-rotating drives, only ever folded,
/// left out -- whose error at the ground state `state` at the trap's centre, at rest and at the reach's largest speed along
/// each lab axis, is ½ or more on its states' populations or on T, or whose small parameter there is (no fold of that error
/// holds there, no series of that parameter converges) kept, until none is;
/// the model, the units kept, its trap and its reach at `temperature_uk`.
pub fn closed(request: &Request, light: &Light, mut kept: Vec<Unit>, state: usize, temperature_uk: f64) -> (Model, Vec<Unit>, Trap, Reach) {
    let reference: [f64; GROUND] = std::array::from_fn(|s| if s == state { 1.0 } else { 0.0 });
    loop {
        let model = Model::new(request, light, &kept).unwrap();
        let trap = model.trap(&reference, &model.prepared()).unwrap();
        let reach = model.reach(&trap, &reference, temperature_uk, unseen(MOST_TRAJECTORIES as f64));
        let cluster = model.cluster_of(state);
        let psi: Vec<C64> = model.clusters()[cluster].states.iter().map(|&s| if s == state { C64::ONE } else { C64::ZERO }).collect();
        let speed = reach.speed / model.hbar_over_mass();
        let samples: Vec<Sample> = (0..4).map(|a| Sample { cluster, at: Point { r: trap.centre, kappa: std::array::from_fn(|b| if a == b + 1 { speed } else { 0.0 }), t: 0.0 }, psi: psi.clone() }).collect();
        let units: Vec<Unit> = model.units().into_iter().filter(|u| !matches!(u, Unit::Bare { component, .. } if component.counter)).collect();
        let errors = model.errors(&units, &samples);
        let failing: Vec<Unit> = units.iter().zip(&errors).filter(|(_, e)| e.iter().any(|e| !(e.population < 0.5 && e.temperature < 0.5 && e.parameter < 0.5))).map(|(u, _)| u.clone()).collect();
        if failing.is_empty() {
            return (model, kept, trap, reach);
        }
        kept.extend(failing);
    }
}

/// The rate at which a generator H lowers a state's norm, ⟨ψ|i(H − H†)|ψ⟩/‖ψ‖²: by the decay's construction (model/photon.rs)
/// the hazard, Σ_φ Σ_a ‖J_{a,φ}ψ‖²/‖ψ‖² of the photons. The anti-Hermitian part i(H − H†) is formed entry by entry before it
/// meets ψ, so that the Hermitian part's scale (the states' energies) does not round the rate.
pub fn decay_rate(h: &Mat, psi: &[C64]) -> f64 {
    let decay = Mat::from_fn(h.rows, h.cols, |x, y| C64::I * (h[(x, y)] - h[(y, x)].conj()));
    inner(psi, &decay.apply(psi)).re / norm_sqr(psi)
}

/// A standard normal deviate from two uniform ones (Box–Muller).
pub fn normal(stream: &mut Stream) -> f64 {
    let (u, v) = (stream.uniform(), stream.uniform());
    (-2.0 * (1.0 - u).ln()).sqrt() * (std::f64::consts::TAU * v).cos()
}
