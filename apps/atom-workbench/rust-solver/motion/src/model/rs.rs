//! The one fold (RUSTPLAN §2.1–§2.3 and its correction "one fold formula"): Reiter–Sørensen second order with dissipation
//! and harmonics (td_s.hilbert_rs: each kept column a at E_a + f, M_f = (E_a + f − G)⁻¹ V_f, H_eff = H_slow + ½ Σ (V† M
//! + h.c.) over the pairs whose sum is slow, L = C_kept + Σ_f C M_f per fast channel). The reference and the model differ
//! only in what is folded (`fold`). Used by the checkpoint (`ModelAt::hilbert`) and by the trajectory (`model::selected`).
//! INTERFACE ONLY here (V3A's skeleton for V3C, who writes the bodies).

use crate::floquet::{Frame, Hilbert};
use crate::linalg::{C64, Mat, V3};
use crate::request::STATES;
use crate::scene::{Components, Key, Scene};

/// One frequency component of a model operator: `value` e^{−i f t}.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelPart {
    pub key: Key,
    /// f (rad/µs), rest frame.
    pub frequency: f64,
    pub value: Mat,
    /// ∂value/∂r_i, i = x, y, z.
    pub gradient: [Mat; 3],
}

/// One jump channel: the request's Cartesian emission C_a (a = x, y, z) folded at one fast index n (`Frame::fast_index`,
/// n = 0 the kept states' own decay, n ≠ 0 a folded sideband's scattering). Parts within a channel interfere; channels
/// do not (their cross terms beat at a fast frequency and are left out with their part of −½L†L: the model stays a
/// Lindblad form).
#[derive(Clone, Debug, PartialEq)]
pub struct Channel {
    pub fast: [i64; 3],
    /// L_a by components (n × n), √Γ included.
    pub emission: [Vec<ModelPart>; 3],
    /// (k ∘ C_a) folded the same way: the photon's momentum per element.
    pub kicks: [Vec<ModelPart>; 3],
}

/// A fold at a point: H_eff, the jump channels and the lift, by frequency components, on the kept states.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelAt {
    /// The kept states (indices into the 32, sorted); n = keep.len().
    pub keep: Vec<usize>,
    /// H_eff's components (n × n); each key with its −key's conjugate listed.
    pub effective: Vec<ModelPart>,
    pub channels: Vec<Channel>,
    /// M_f by components: the folded rows' amplitudes (a row per folded state or sideband; `rows` its state).
    pub lift: Vec<ModelPart>,
    pub rows: Vec<usize>,
}

/// The one fold at a point: keeps `keep` (indices into the 32, sorted) at their slow components and folds by the RS
/// formula everything else -- the other states Q, and on the kept states the tweezer's sidebands and, given a frame, the
/// hyperfine sidebands (n_φ ≠ 0).
/// - The reference: `fold(scene, parts, &[0..32], None)` -- only the tweezer's sidebands folded, told by the key's
///   tweezer and carrier coordinates (n_tw, n_ν ≠ 0; the tweezer has an axis of its own, scene.rs), which needs no frame.
/// - The model: `fold(scene, parts, P, Some(frame))`, the frame chosen on the reference at the same point.
/// Denominators as td_s.hilbert_rs: column a at E_a + f_c, E_a = Re (static part)_aa, the folded block's
/// G = (static part) − iΓ/2 (Γ = Σ_a C_a†C_a).
pub fn fold(scene: &Scene, components: &Components, keep: &[usize], frame: Option<&Frame>) -> ModelAt {
    let _ = (scene, components, keep, frame);
    todo!()
}

/// The RS fold on a kept set P for the trajectory: what does not depend on the point made once.
#[derive(Clone, Debug, PartialEq)]
pub struct Rs {
    /// P: indices into the 32, sorted (the ground states first).
    pub keep: Vec<usize>,
}

impl Rs {
    pub fn new(scene: &Scene, frame: &Frame, keep: &[usize]) -> Rs {
        let _ = (scene, frame, keep);
        todo!()
    }

    /// The model at a point, from the scene's components there (`Scene::components_at`): the same as `fold`.
    pub fn at(&self, scene: &Scene, components: &Components) -> ModelAt {
        let _ = (scene, components);
        todo!()
    }
}

impl ModelAt {
    /// The mean force on ρ (n × n, row-major): the adjoint contraction −Re tr(∇H_eff ρ) and the jumps' part.
    pub fn force(&self, rho: &[C64]) -> V3 {
        let _ = rho;
        todo!()
    }

    /// The 32 states' populations of ρ lifted through M_f (the two-point check's lift).
    pub fn lift(&self, rho: &[C64]) -> [f64; STATES] {
        let _ = rho;
        todo!()
    }

    /// The fold as a Hilbert model for the checkpoint, along the analysis axis u: each channel's C_a rotated onto
    /// (e1, e2, u) with weights 2/5, 2/5, 1/5 (floquet::Hilbert), gradients along u.
    pub fn hilbert(&self, axis: V3) -> Hilbert {
        let _ = axis;
        todo!()
    }
}
