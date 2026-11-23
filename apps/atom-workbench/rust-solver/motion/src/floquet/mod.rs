//! The static checkpoint's Floquet lattice (RUSTPLAN §1): a Hilbert model by its frequency components (the full 32-state
//! reference from `Scene`, or the RS model from `model::rs`), its frame and slow lattice (`frame`), the rounding rule's
//! elements (`select`), and the one Liouville assembly with its readouts (`assemble`). The full reference and the model
//! go through the same functions.
//!
//! The product reads the frame alone (the fold's sectors, the trajectory model's frame vectors); the lattice, the
//! elements, the assembly and its steady state (krylov) are the tests' static reference, compiled with them only (until
//! 2026-11-22 in the product, where the old two-point check had solved them).

pub mod frame;
#[cfg(test)]
pub mod select;
#[cfg(test)]
pub mod assemble;
#[cfg(test)]
mod tests;

use crate::linalg::Mat;
use crate::scene::{Components, Key, Scene};

#[cfg(test)]
pub use assemble::{assemble, readouts, Liouville, Readouts, System};
#[cfg(test)]
pub use frame::Lattice;
pub use frame::{Frame, Sectors};
#[cfg(test)]
pub use select::select;

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

/// An element of the lattice: slow coordinates m (over the lattice basis), fast n = (n_φ, n_tw, n_ν), and the Liouville
/// index x = a·n + b of ρ_ab.
#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Element {
    pub m: Vec<i64>,
    pub n: [i64; 3],
    pub x: usize,
}

/// The elements the rounding rule admits, in the order the rule took them (best first); `amplitudes[i]` the first-order
/// amplitude of `order[i]`.
#[cfg(test)]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Elements {
    pub order: Vec<Element>,
    pub amplitudes: Vec<f64>,
}

/// A sparse complex matrix in compressed rows.
#[cfg(test)]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Sparse {
    pub rows: usize,
    pub cols: usize,
    /// Row i holds entries indptr[i]..indptr[i + 1] of `indices` (columns) and `values`.
    pub indptr: Vec<usize>,
    pub indices: Vec<usize>,
    pub values: Vec<crate::linalg::C64>,
}

impl Hilbert {
    /// The reference at a point: the 32 states with the tweezer's sidebands folded (the one fold, model/rs.rs `fold`),
    /// everything else explicit.
    pub fn full(scene: &Scene, components: &Components) -> Hilbert {
        crate::model::rs::reference(scene, components).hilbert()
    }
}
