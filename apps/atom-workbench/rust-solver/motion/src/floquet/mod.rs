//! The static checkpoint's Floquet lattice (RUSTPLAN §1): a Hilbert model by its frequency components (the full 32-state
//! reference from `Scene`, or the RS model from `model::rs`), its frame and slow lattice (`frame`), the rounding rule's
//! elements (`select`), and the one Liouville assembly with its readouts (`assemble`). The full reference and the model
//! go through the same functions.

pub mod frame;
pub mod select;
pub mod assemble;

use crate::linalg::{C64, Mat};
use crate::scene::Key;

pub use assemble::{assemble, readouts, Liouville, Readouts, System};
pub use frame::{Frame, Lattice};
pub use select::select;

/// One frequency component of an operator of a Hilbert model: `value` e^{−i f t}, f the key's frequency.
#[derive(Clone, Debug, PartialEq)]
pub struct Part {
    pub key: Key,
    /// n × n.
    pub value: Mat,
    /// ∂value/∂u along the analysis axis u (n × n).
    pub gradient: Mat,
}

/// A Hilbert model at a point: n states (a subset `states` of the 32), H, the jumps and their photon-weighted forms by
/// frequency components; for the RS model the lift M_f onto the folded states.
#[derive(Clone, Debug, PartialEq)]
pub struct Hilbert {
    /// The model's states as indices into the 32 (sorted); n = states.len().
    pub states: Vec<usize>,
    /// H's components, each key with its −key's conjugate listed too (td_s.hilbert_full).
    pub hamiltonian: Vec<Part>,
    /// Per jump channel (e1, e2, u: the emission rotated onto the analysis axis u, weights 2/5, 2/5, 1/5) its components;
    /// `Part::gradient` holds ∂L/∂u (zero for the full reference).
    pub jumps: Vec<Vec<Part>>,
    /// Per jump channel √w (K ∘ C) by components: the photon's momentum along u (scene_x `jumpk`).
    pub kicks: Vec<Vec<Part>>,
    /// The levels (populations read per level): indices into `states`.
    pub groups: Vec<Vec<usize>>,
    /// The RS lift (None for the full reference): the folded states Q (indices into the 32) and M_f (|Q| × n) per key.
    pub lift: Option<Lift>,
}

/// The RS model's folded states and their amplitudes M_f = (E_a + f − G_QQ)⁻¹ H_QS,f.
#[derive(Clone, Debug, PartialEq)]
pub struct Lift {
    pub folded: Vec<usize>,
    pub amplitudes: Vec<(Key, Mat)>,
}

/// An element of the lattice: slow coordinates m (over the lattice basis), fast n = (n_φ, n_tw, n_ν), and the Liouville
/// index x = a·n + b of ρ_ab.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Element {
    pub m: Vec<i64>,
    pub n: [i64; 3],
    pub x: usize,
}

/// The elements the rounding rule admits, in the order the rule took them (best first); `amplitudes[i]` the first-order
/// amplitude of `order[i]`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Elements {
    pub order: Vec<Element>,
    pub amplitudes: Vec<f64>,
}

/// A sparse complex matrix in compressed rows.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Sparse {
    pub rows: usize,
    pub cols: usize,
    /// Row i holds entries indptr[i]..indptr[i + 1] of `indices` (columns) and `values`.
    pub indptr: Vec<usize>,
    pub indices: Vec<usize>,
    pub values: Vec<C64>,
}

impl Hilbert {
    /// The full 32-state reference at r for the momentum κ, its gradients and jumps along the analysis axis u
    /// (td_s.hilbert_full on scene_x.build).
    pub fn full(scene: &crate::scene::Scene, r: crate::linalg::V3, kappa: crate::linalg::V3, axis: crate::linalg::V3) -> Hilbert {
        let _ = (scene, r, kappa, axis);
        todo!()
    }
}
