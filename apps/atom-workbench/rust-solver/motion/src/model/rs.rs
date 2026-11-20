//! The one fold (RUSTPLAN correction "one fold formula"; v3c.status.md §4): Reiter–Sørensen second order with dissipation
//! and harmonics. The reference and the model differ only in the kept set and the sectors.
//! INTERFACE STUB in V3A's branch (V3C writes this file; its signatures as v3c.status.md §4, `reference` on the scene's
//! components so that the checkpoint's points -- w = 0, the edge's tweezer dark -- are its input too).

use crate::floquet::{Hilbert, Sectors};
use crate::linalg::{Mat, V3};
use crate::scene::{Components, Key, Scene};

/// One frequency component of a model operator: `value` e^{−i f t}.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelPart {
    pub key: Key,
    pub frequency: f64,
    pub value: Mat,
    pub gradient: [Mat; 3],
}

/// A jump channel: one emission axis, one fast index.
#[derive(Clone, Debug, PartialEq)]
pub struct Channel {
    pub emission: usize,
    pub fast: [i64; 2],
    pub parts: Vec<ModelPart>,
    pub kicks: Vec<ModelPart>,
}

/// A model at a point.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelAt {
    pub states: Vec<usize>,
    pub folded: Vec<usize>,
    pub groups: Vec<Vec<usize>>,
    pub effective: Vec<ModelPart>,
    pub jumps: Vec<Channel>,
    pub lift: Vec<ModelPart>,
}

/// The one fold.
pub fn fold(source: &ModelAt, keep: &[usize], sectors: &Sectors) -> ModelAt {
    let _ = (source, keep, sectors);
    todo!()
}

/// The reference at a point: the scene's components and the Cartesian emission, all 32 states kept, the tweezer's sidebands
/// folded (`Sectors::tweezer`).
pub fn reference(scene: &Scene, components: &Components) -> ModelAt {
    let _ = (scene, components);
    todo!()
}

/// The RS model on P for the trajectory.
#[derive(Clone, Debug, PartialEq)]
pub struct Rs {
    pub keep: Vec<usize>,
}

impl ModelAt {
    /// The model as a Hilbert model for the checkpoint along the analysis axis u (`channels(axis)`, `levels`).
    pub fn hilbert(&self, axis: V3) -> Hilbert {
        let _ = axis;
        todo!()
    }
}
