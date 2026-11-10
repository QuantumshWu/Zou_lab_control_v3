//! What the model's tests share (tests/model_*.rs): the scenes' requests, the generator's side of the hazard's identity, and
//! normal deviates.
#![allow(dead_code)]
use atom_motion::linalg::{inner, norm_sqr, Mat, C64};
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
