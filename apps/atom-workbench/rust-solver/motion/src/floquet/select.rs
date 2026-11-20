//! The rounding rule's elements (td_s.select_all, mdlat.select_all): best first from the k = 0 populations, each
//! (harmonic, element) by one rule -- a_y = a_x c / max(|λ0_y|, c), admitted iff a_y ≥ ε (ε the unit roundoff's
//! `f64::EPSILON`), c the coupling's magnitude (L, K1/k_ph, K2/k_ph²), λ0_y = i f(m, n) + L_xx.

use super::{Elements, Liouville};

/// The admitted elements of a Liouvillian; `photon` the reference photon wave number k_ph (rad/µm) that weighs K1, K2.
pub fn select(parts: &Liouville, photon: f64) -> Elements {
    let _ = (parts, photon);
    todo!()
}
