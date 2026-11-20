//! The scene: the one place the request's Hamiltonian is built (RUSTPLAN §1.1, §2.1, §2.3).
//!
//! H(r, κ, t) = H0 − held(O(0)) + Σ_ω O_ω(E_ω(r)) + T(r) + Σ_l [a_l(r) X_l e^{−i f_l t} + h.c.]:
//! - every non-tweezer beam l: its co-rotating coupling X_l (RWA), all 32 states (both D lines, both ground F), at its
//!   rest-frame frequency f_l = ω_l − w_l(κ), w_l(κ) = (ħ/m)(k_l·κ + k_l²/2) (D2);
//! - the tweezer (the one allowed distinction): its static second order T(r) = I(r)(F₀ + w_tw F₁) through the 32 states,
//!   co- with co- and counter- with counter-rotating, built once at unit intensity (D3);
//! - the outside forms O_ω at each frequency's summed field, the focus's diagonal held in H0; the interference of two
//!   beams of one frequency at different Doppler shifts is a component at their beat;
//! - the jumps: the request's Cartesian emission (√Γ included) and each element's photon wave number k_eg = ω_eg/c.
//!
//! Allowed approximations (BRIEF_V3IMPL): the tweezer/non-tweezer split; RWA on non-tweezer light; no non-tweezer coupling
//! outside the 32 states; non-tweezer light on D1/D2 only (`request.rs` `validate`). Everything else to the rounding rule.

use crate::linalg::{Mat, V3};
use crate::model::Quadratic;
use crate::request::{Request, GROUND};

/// A frequency as an exact integer vector over the scene's generators (`Scene::generators`): one per non-tweezer beam,
/// its rest-frame frequency f_l = ω_l − w_l(κ). Equal frequencies of different beams are different keys; the floquet frame
/// (floquet/frame.rs) reduces keys to the request's axes and the Doppler classes exactly.
pub type Key = Vec<i32>;

/// One non-tweezer beam as a frequency generator: its frequency on the request's axes and its wave vector.
#[derive(Clone, Debug, PartialEq)]
pub struct Generator {
    /// The beam's index in `Request::beams`.
    pub beam: usize,
    /// Its frequency on the request's frequency axes (`Beam::frequency_index`), exact.
    pub frequency_index: Vec<i32>,
    /// ω_l = frequency_index · 2π `Request::frequencies_mhz` (rad/µs), in the frames of `Request::frame_frequencies_mhz`.
    pub frequency: f64,
    /// k_l (rad/µm on the lab axes).
    pub wave_vector: V3,
}

/// One frequency component of H at a point: H holds `value` e^{−i f t} (+ h.c. for a nonzero key).
#[derive(Clone, Debug, PartialEq)]
pub struct Component {
    /// Its frequency's key; the zero key is the static part (Hermitian), every other key appears once, its Hermitian
    /// conjugate at −key implied.
    pub key: Key,
    /// f = Σ_l key_l f_l (rad/µs): the frequency in the atom's rest frame at the momentum asked for.
    pub frequency: f64,
    /// The 32 × 32 matrix at r (rad/µs), the beams' amplitudes a_l(r) (phase e^{ik·r} included) in it.
    pub value: Mat,
    /// ∂value/∂r_i, i = x, y, z (rad/µs/µm).
    pub gradient: [Mat; 3],
}

/// H(r, κ) by its frequency components.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Components {
    pub parts: Vec<Component>,
}

/// The jumps: the request's emission and the photon wave number of each element.
#[derive(Clone, Debug, PartialEq)]
pub struct Jumps {
    /// C_a, a = x, y, z (1/√µs, 32 × 32, ground ← excited): Σ_a C_a†C_a = Γ on the excited states.
    pub emission: [Mat; 3],
    /// k_eg = ω_eg / c per element (rad/µm; real, stored on the same 32 × 32 positions as C_a), ω_eg the transition's
    /// frequency H0_ee − H0_gg + 2π(ν_e − ν_g) with the frames.
    pub wave_number: Mat,
}

/// The scene built from a request: frames and frequency generators, the tweezer's F₀ and F₁, the outside forms, the jumps.
#[derive(Clone, Debug)]
pub struct Scene {
    pub request: Request,
    /// The non-tweezer beams as frequency generators (the keys' coordinates).
    pub generators: Vec<Generator>,
    pub jumps: Jumps,
}

impl Scene {
    /// Builds the scene; Err for a request outside the allowed approximations (`Request::validate`).
    pub fn new(request: &Request) -> Result<Scene, String> {
        let _ = request;
        todo!()
    }

    /// H's components at r for the momentum κ (1/µm): values, gradients and rest-frame frequencies.
    pub fn components_at(&self, r: V3, kappa: V3) -> Components {
        let _ = (r, kappa);
        todo!()
    }

    /// The jump operators and the photons' wave numbers.
    pub fn jumps(&self) -> &Jumps {
        &self.jumps
    }

    /// The trap potential of each ground state at r (rad/µs): the tweezer's static second order and the outside forms'
    /// ground diagonal, value, gradient and Hessian -- the one source of V_ref, the prepared wells, the loss energy and
    /// the reported trap (D3).
    pub fn trap_potential(&self, r: V3) -> [Quadratic; GROUND] {
        let _ = r;
        todo!()
    }
}
