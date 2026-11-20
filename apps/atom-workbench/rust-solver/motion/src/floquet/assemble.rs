//! Lindblad → sparse Liouville by lattice components, the bordered system on the elements, and the readouts (td_s.Liou,
//! mdlat.assemble / readouts, fr.assemble / readouts / rhs). One function for the full reference and the model.
//!
//! L = −i[H, ·] + Σ_j (C_j · C_j† − ½{C_j†C_j, ·}) in the frame; K1 the force superoperator (−½{∂H, ·} and the jumps'
//! gradient terms), K2 the recoil diffusion (½ Σ_j K_j · K_j†). The system on the elements y = (m, n, x):
//! [[L + i f(m, n), τ₀*], [τ₀ᵀ, 0]] (ρ; λ) = (0; 1), τ₀ the k = 0 trace; F = τ₀·K1ρ, D = τ₀·K2ρ + τ₀·K1y.

use std::collections::HashMap;

use super::{Elements, Frame, Hilbert, Lattice, Sparse};
use crate::linalg::C64;

/// One lattice component of the Liouvillian: L, K1, K2 (d × d, d = n²), each absent where zero.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Superoperators {
    pub l: Option<Sparse>,
    pub k1: Option<Sparse>,
    pub k2: Option<Sparse>,
}

/// A Hilbert model's Liouvillian on its lattice, and its readout rows.
#[derive(Clone, Debug)]
pub struct Liouville {
    /// n states; d = n².
    pub states: usize,
    pub frame: Frame,
    pub lattice: Lattice,
    /// The components by lattice key (m, n).
    pub components: HashMap<(Vec<i64>, [i64; 3]), Superoperators>,
    /// The readout rows by lattice key: populations per state, the lift's folded populations, the rates per jump
    /// channel, the force (td_s.Liou names), each a row over d.
    pub rows: HashMap<(Vec<i64>, [i64; 3]), Vec<Vec<C64>>>,
    /// The readouts' names, the rows' order.
    pub names: Vec<String>,
    /// The levels (indices into the model's states), copied from the Hilbert model.
    pub groups: Vec<Vec<usize>>,
}

impl Liouville {
    /// The Liouvillian of a Hilbert model in the frame of its point (td_s.Liou + mdlat.PerM): the reference and the model
    /// at one point take the same frame (`Frame::new` on the reference).
    pub fn new(frame: &Frame, hilbert: &Hilbert) -> Liouville {
        let _ = (frame, hilbert);
        todo!()
    }
}

/// The bordered sparse system on the elements.
#[derive(Clone, Debug, PartialEq)]
pub struct System {
    /// The unknowns' elements (the bordering multiplier is the last unknown, index elements.len()).
    pub elements: Elements,
    /// [[L + i f, τ₀*], [τ₀ᵀ, 0]], (N + 1) × (N + 1), N = elements' count.
    pub matrix: Sparse,
    /// K1 and K2 on the elements (N × N).
    pub k1: Sparse,
    pub k2: Sparse,
    /// τ₀ over the elements: 1 at the k = 0 populations.
    pub trace: Vec<C64>,
    /// Each unknown's harmonic, N + 1 of them (fr.harmonic_ids): the index of its slow lattice point m (not the (m, n)
    /// sector) among the system's sorted distinct m; the multiplier's row takes the populations' m₀.
    pub harmonic: Vec<usize>,
}

/// The system on the admitted elements.
pub fn assemble(parts: &Liouville, elements: &Elements) -> System {
    let _ = (parts, elements);
    todo!()
}

/// What a solved point reads (td_s.pack): populations per state and per level, the rates, F, D (along the analysis axis).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Readouts {
    pub populations: Vec<f64>,
    pub levels: Vec<f64>,
    /// The folded states' populations (the model's lift), empty for the full reference.
    pub folded: Vec<f64>,
    /// The scattering rate per jump channel (1/µs).
    pub rates: Vec<f64>,
    /// The force F = τ₀·K1ρ (ħ/µm/µs) and the diffusion D.
    pub force: f64,
    pub diffusion: f64,
}

/// The readouts of a solved point: ρ (the system's solution without the multiplier) and y (the force's response).
pub fn readouts(parts: &Liouville, system: &System, rho: &[C64], y: &[C64]) -> Readouts {
    let _ = (parts, system, rho, y);
    todo!()
}
