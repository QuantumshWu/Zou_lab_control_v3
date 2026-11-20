//! The trajectory's model (RUSTPLAN §3.2): the selected P and its RS fold as `step::Dynamics` / `step::Middle` for the
//! onefold integrator. One cluster (all of P), frame β of floquet/frame.rs; generator from `Rs::at` at the step's nodes;
//! landing by the plain L_q ψ (D5); readout by `ModelAt::lift`; reference, energy and loss from `Scene::trap_potential`.

use crate::model::rs::Rs;
use crate::scene::Scene;
use crate::select::Selection;

pub struct Selected {
    pub scene: Scene,
    pub rs: Rs,
    pub report: Selection,
}
