//! The solve and its fallback chain (fr.solve): ρ, then the force's response y, each certified (ζ ≤ 1) or solved again by
//! the next route.

use crate::floquet::System;
use crate::linalg::C64;

/// The route a solve took.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    /// GMRES on the harmonic-block Schur preconditioner, certified.
    Krylov,
    /// The exact Schur complement.
    ExactSchur,
    /// Direct sparse LU of the whole system.
    Direct,
}

/// A solved system: ρ and y on the system's elements (the multiplier dropped), the route and the GMRES rounds.
#[derive(Clone, Debug, PartialEq)]
pub struct Solved {
    pub rho: Vec<C64>,
    pub y: Vec<C64>,
    pub route: Route,
    pub rounds: usize,
    /// The certificate ζ of the accepted solution.
    pub certificate: f64,
}

/// fr.solve in full.
pub fn solve(system: &System) -> Solved {
    let _ = system;
    todo!()
}
