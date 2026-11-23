//! A Hilbert model by its frequency components (the full 32-state reference from `Scene`, or the RS model from
//! `model::rs`) and its frame (`frame`, RUSTPLAN §1): the fold's sectors and the trajectory model's frame vectors.
//! (Until 2026-11-23 also the static checkpoint's Floquet lattice, the rounding rule's elements, the one Liouville
//! assembly and its steady state by GMRES (the krylov module, with faer): the old two-point check's, the tests' alone
//! after 2026-11-22, deleted once no test read them.)

pub mod frame;
#[cfg(test)]
mod tests;

use crate::linalg::Mat;
use crate::scene::{Components, Key, Scene};

pub use frame::{Frame, Sectors};

/// One frequency component of an operator of a Hilbert model: `value` e^{−i f t}, f the key's frequency.
#[derive(Clone, Debug, PartialEq)]
pub struct Part {
    pub key: Key,
    /// n × n.
    pub value: Mat,
    /// ∂value/∂r_i, i = x, y, z (n × n).
    pub gradient: [Mat; 3],
}

/// A Hilbert model at a point: n states (a subset `states` of the 32), H, the jumps and their photon-weighted forms by
/// frequency components; for the RS model the lift M_f onto the folded states.
#[derive(Clone, Debug, PartialEq)]
pub struct Hilbert {
    /// The model's states as indices into the 32 (sorted); n = states.len().
    pub states: Vec<usize>,
    /// H's components, each key with its −key's conjugate listed too (td_s.hilbert_full).
    pub hamiltonian: Vec<Part>,
    /// Per jump channel (model/rs.rs `Channel`: one Cartesian emission axis, one fast index) its components; parts within a
    /// channel interfere, channels do not. `Part::gradient` holds ∇L.
    pub jumps: Vec<Vec<Part>>,
    /// The levels (populations read per level): each level's states as indices into the 32.
    pub groups: Vec<Vec<usize>>,
    /// The fold's lift: M_f per key, a row per folded state or sideband.
    pub lift: Option<Lift>,
}

/// The fold's amplitudes M_f = (E_a + f − G)⁻¹ V_f on the folded rows.
#[derive(Clone, Debug, PartialEq)]
pub struct Lift {
    /// Each row's state (an index into the 32).
    pub folded: Vec<usize>,
    pub amplitudes: Vec<(Key, Mat)>,
}

impl Hilbert {
    /// The reference at a point: the 32 states with the tweezer's sidebands folded (the one fold, model/rs.rs `fold`),
    /// everything else explicit.
    pub fn full(scene: &Scene, components: &Components) -> Hilbert {
        crate::model::rs::reference(scene, components).hilbert()
    }
}
