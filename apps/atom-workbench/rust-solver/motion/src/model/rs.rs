//! The one fold (RUSTPLAN §2.1–§2.3): Reiter–Sørensen second order with harmonics on P = `keep` (td_s.hilbert_rs, with
//! dissipation, each column at E_a + f): M_f = (E_a + f − G_QQ)⁻¹ H_QS,f, H_eff = H_SS + ½ Σ (H_SQ M_f + h.c.),
//! L_a = C_a,SS + Σ_f C_a,SQ M_f. f the component's rest-frame frequency (D2: each beam its own Doppler). Used by the
//! checkpoint's model (`hilbert`) and by the trajectory (`model::selected`) alike.

use crate::floquet::Hilbert;
use crate::linalg::{C64, Mat, V3};
use crate::request::STATES;
use crate::scene::{Key, Scene};

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

/// The RS fold on a kept set P.
#[derive(Clone, Debug, PartialEq)]
pub struct Rs {
    /// P: indices into the 32, sorted (the ground states first).
    pub keep: Vec<usize>,
    /// Q: the folded states, the rest of the 32.
    pub folded: Vec<usize>,
}

/// The model at a point: H_eff, the jumps and their photon-weighted forms, and the lift, by frequency components.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelAt {
    /// H_eff's components on P (n × n); each key with its −key's conjugate listed.
    pub effective: Vec<ModelPart>,
    /// L_a, a = x, y, z (the request's Cartesian emission, √Γ included), by components (n × n).
    pub jumps: [Vec<ModelPart>; 3],
    /// (k ∘ C_a) folded the same way: the photon's momentum per element (rad/µm · 1/√µs).
    pub kicks: [Vec<ModelPart>; 3],
    /// M_f (|Q| × n) by components: the folded states' amplitudes.
    pub lift: Vec<ModelPart>,
}

impl Rs {
    pub fn new(scene: &Scene, keep: &[usize]) -> Rs {
        let _ = (scene, keep);
        todo!()
    }

    /// The model at r for the momentum κ.
    pub fn at(&self, scene: &Scene, r: V3, kappa: V3) -> ModelAt {
        let _ = (scene, r, kappa);
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

    /// The model as a Hilbert model for the checkpoint, along the analysis axis u (jumps rotated onto e1, e2, u).
    pub fn hilbert(&self, axis: V3) -> Hilbert {
        let _ = axis;
        todo!()
    }
}
