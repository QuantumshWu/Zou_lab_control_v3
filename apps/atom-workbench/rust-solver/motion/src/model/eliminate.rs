//! The near light's elimination (SPEC §1.4): second-order Reiter–Sørensen with Brillouin–Wigner dressing and first-order
//! Doppler, for the co-rotating components the units keep near (the near carriers).
//!
//! This stage holds the carriers' list and the hooks the far light's code calls (the near potential and envelope, the near
//! units, gates, errors and edges, the near spread); a model with a near carrier is refused until the elimination is in.

use super::cluster::Edge;
use super::potential::Expansion;
use super::reach::Span;
use super::{Component, Gate, Lights, Model, Point, Quadratic, Reach, Sample, Unit, UnitError};
use crate::field::Amplitude;
use crate::linalg::C64;
use crate::request::{Request, GROUND};

/// A near carrier: a co-rotating component the units keep near.
#[derive(Clone, Debug)]
pub(crate) struct Carrier {
    pub(crate) frequency: usize,
}

impl Carrier {
    /// The near carriers of a model: every co-rotating component not folded far.
    pub(crate) fn all(model: &Model, _request: &Request) -> Result<Vec<Carrier>, String> {
        let carriers: Vec<Carrier> = (0..model.light.frequencies.len())
            .filter(|&f| !model.folded(Component { frequency: f, counter: false }))
            .map(|frequency| Carrier { frequency })
            .collect();
        if !carriers.is_empty() {
            return Err("近共振载波的消去尚未接入".into());
        }
        Ok(carriers)
    }
}

impl Model {
    /// The near carriers' part of V_L(r; p) (potential.rs): their static shifts on the ground states with the trap light on,
    /// less with it off.
    pub(crate) fn near_potential(&self, _r: [f64; 3], _amplitudes: &[Amplitude], _p: &[f64; GROUND], _lights: &Lights) -> Expansion {
        Expansion { quadratic: Quadratic { value: 0.0, gradient: [0.0; 3], hessian: [[0.0; 3]; 3] }, scale: 0.0, terms: 0 }
    }
    /// The near trap light's light-shift bound per beam pair of the trap frequency (potential.rs `Model::envelope`).
    pub(crate) fn near_envelope(&self, _p: &[f64; GROUND]) -> Vec<f64> {
        vec![]
    }
    /// The near carriers' units: their drives and the couplings of two ground states they make.
    pub(crate) fn near_units(&self) -> Vec<Unit> {
        vec![]
    }
    /// A near unit's gate over the reach (a photon coherence's: photon.rs).
    pub(crate) fn near_gate(&self, unit: &Unit, reach: &Reach) -> Gate {
        match unit {
            Unit::Coherence { .. } => self.coherence_gate(unit, reach),
            _ => Gate { saturation: f64::INFINITY, sweep: f64::INFINITY },
        }
    }
    /// A near unit's error at a sample (a photon coherence's: photon.rs).
    pub(crate) fn near_error(&self, unit: &Unit, sample: &Sample) -> UnitError {
        match unit {
            Unit::Coherence { .. } => self.coherence_error(unit, sample),
            _ => UnitError { parameter: f64::INFINITY, population: f64::INFINITY, temperature: f64::INFINITY },
        }
    }
    /// A kept near unit's edge.
    pub(crate) fn near_edge(&self, _unit: &Unit) -> Option<Edge> {
        None
    }
    /// The bound of the near light's differential shift of two states over a cell, which the static gates' separations take.
    pub(crate) fn near_spread(&self, _a: usize, _b: usize, _spans: &[Span]) -> f64 {
        0.0
    }
    /// The frequency whose kept drives feed an explicit state's photons (photon.rs).
    pub(crate) fn driving_frequency(&self, _source: &super::photon::Source) -> Option<usize> {
        None
    }
    /// The near light's terms of a cluster's generator at a point (assemble.rs).
    pub(crate) fn add_near(&self, _parts: &mut super::assemble::Parts, _cluster: usize, _at: &Point, _amplitudes: &[Amplitude], _lights: &Lights) {}
    /// The near carriers' folded channels on a cluster at a point (readout.rs).
    pub(crate) fn near_channels(&self, _cluster: usize, _at: &Point, _amplitudes: &[Amplitude], _lights: &Lights) -> Vec<super::readout::Channels> {
        vec![]
    }
    /// The near light's part of V_L far from the trap (readout.rs `Model::asymptote`).
    pub(crate) fn near_asymptote(&self, _p: &[f64; GROUND], _lights: &Lights) -> f64 {
        0.0
    }
    /// The near carriers' photon sources (photon.rs).
    pub(crate) fn near_sources(&self) -> Vec<super::photon::Source> {
        vec![]
    }
    /// The explicit sources' recoil terms of a photon (photon.rs): per drive its share of the amplitudes and ∇ln σ.
    pub(crate) fn explicit_recoil(&self, _cluster: usize, _at: &Point, _source: usize, _own: [Vec<C64>; 3]) -> Vec<([Vec<C64>; 3], [C64; 3])> {
        vec![]
    }
}
