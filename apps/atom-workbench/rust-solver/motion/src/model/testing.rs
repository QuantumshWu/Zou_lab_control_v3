//! What the model's tests share (compiled with the tests alone): the reference scenes' requests, the generator's side
//! of the hazard's identity.
use crate::linalg::{inner, norm_sqr, Mat, C64};
use crate::request::Request;

/// A scene's request through the one parser: a reference scene by name (refs/scenes, the app's native request), or any
/// native request file by its path.
pub(crate) fn scene(name: &str) -> Request {
    let path = if name.ends_with(".json") { std::path::PathBuf::from(name) } else { std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("scenes").join(format!("{name}.json")) };
    let line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    Request::from_native(&line).unwrap()
}

/// The rate at which a generator H lowers a state's norm, ⟨ψ|i(H − H†)|ψ⟩/‖ψ‖²: by the decay's construction
/// (model/selected.rs, # The generator by beats) the hazard, Σ_φ Σ_a ‖J_{a,φ}ψ‖²/‖ψ‖² of the photons. The anti-Hermitian
/// part i(H − H†) is formed entry by entry before it meets ψ, so that the Hermitian part's scale (the states' energies)
/// does not round the rate.
pub(crate) fn decay_rate(h: &Mat, psi: &[C64]) -> f64 {
    let decay = Mat::from_fn(h.rows, h.cols, |x, y| C64::I * (h[(x, y)] - h[(y, x)].conj()));
    inner(psi, &decay.apply(psi)).re / norm_sqr(psi)
}
