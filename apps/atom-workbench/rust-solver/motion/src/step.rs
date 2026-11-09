//! One step: A(τ/2) B(τ) A(τ/2) with the mid-step reference, the internal propagation along the path, the impulse, the
//! jump's root and the jump (SPEC §2.1, §1.6).
//!
//! # The motion and its splitting
//! A trajectory is (r, κ, ψ): position (µm), momentum κ = p/ħ (1/µm) and the cluster's unnormalized internal state between
//! jumps: ṙ = (ħ/m)κ, κ̇ = −⟨ψ|G|ψ⟩/⟨ψ|ψ⟩, iψ̇ = Hψ, H the cluster's generator and G = ∇H_h (`Dynamics::generator`).
//! Without the decay this is the Hamiltonian system 𝓗 = (ħ/m)κ²/2 + ⟨ψ|H_h|ψ⟩/⟨ψ|ψ⟩ on the product phase space. A step
//! splits it as 𝓗 = A + B, A = (ħ/m)κ²/2 + V_ref(r) and B the rest, and composes their exact flows A(τ/2) B(τ) A(τ/2):
//! symplectic in its conservative part, its own inverse, second order (propagation.md §2.1).
//!
//! # The reference (A)
//! V_ref is the second-order expansion of the fixed reference state's trap potential (`Dynamics::reference`) at the
//! step's middle p: V_ref(r) = V + g·(r − p) + ½(r − p)ᵀK(r − p). Its flow is exact along K's principal axes -- a rotation
//! at ω² = (ħ/m)λ for a curvature λ > 0, a hyperbolic flow for λ < 0, free flight under the constant force for λ = 0, one
//! formula for all three (`Flow`) --, so a step is not bound to the trap's period. p is the fixed point "the first half
//! flow on V_ref(p) reaches p": the step's reverse then takes the same reference, and the step is its own inverse (Hut,
//! Makino & McMillan 1995, the time-symmetrized leapfrog; propagation.md §2.2). Any geometry, one rule.
//!
//! # The kick (B)
//! The position stays at r_h; the internal state goes along the straight path r_h + v_m(t − t_m) with the path's phases
//! exact: in the cluster's Doppler frame (`Dynamics::cluster`) its generator is Σ_j M_j e^{iω_j(t − t_m)} + diag(K_s·v_m),
//! ω_j = Ω_j + q_j·v_m, static where every pair of states holds one wave vector. The impulse is the conservative force at
//! r_h on the start state (a shear: it depends on r_h and ψ₀ alone), plus what the state's change along the path adds
//! (the evolved state's force along the path less the start state's), less the reference's force. v_m is the step's
//! middle momentum, κ_h plus half the kick (propagation.md §2.3).
//!
//! # Jumps (SPEC §1.6)
//! ‖ψ‖² falls monotonically between jumps (the decay is −(i/2)ΣJ†J by the model's construction). Where it reaches the
//! trajectory's threshold θ within a step, the step is redone to that root and the jump made at its end (`jump`); the
//! rest is a step of its own. A jump draws from its own stream (`random::Stream::jump`) in a fixed order: the next
//! threshold, the class, the polarization, the fluctuation's signs, and the direction's rejection loop last.

use crate::linalg::C64;
use crate::model::{Cluster, Generator, Photon, Point, Quadratic, Recoil, STATES};

/// What one step needs from the model (SPEC §2.1, §1.6): defined here by its consumer, implemented by the model for its
/// `Model` with the run's lights and the reference state. Each method is one model function; the integrator defines
/// none of these quantities.
pub trait Dynamics {
    /// ħ/m (µm²/µs): the velocity of κ is (ħ/m)κ, its kinetic energy (ħ/m)κ²/2 (rad/µs).
    fn hbar_over_mass(&self) -> f64;

    /// The reference potential's second-order expansion at r (SPEC §2.1): the fixed reference state's trap potential,
    /// `Model::expansion(r, p_ref, &model.prepared())`. The splitting is exact for any reference; this one moves the trap's
    /// harmonic part, and its gradient and curvature at the step's middle, exactly.
    fn reference(&self, r: [f64; 3]) -> Quadratic;

    /// A cluster: its states and its Doppler frame K_s (`Model::clusters()[c]`).
    fn cluster(&self, cluster: usize) -> &Cluster;

    /// The cluster a ground state belongs to (`Model::cluster_of`).
    fn cluster_of(&self, ground: usize) -> usize;

    /// The cluster's generator at a point with every light on (`Model::assemble(cluster, at, &model.running())`): its
    /// parts M_j, G_j at (at.r, at.t) for the momentum at.kappa (the folded responses' first-order Doppler term), on the
    /// path H(t) = Σ_j M_j e^{iω_j(t − at.t)}, ω_j = Ω_j + q_j·v, in the cluster's Doppler frame.
    fn generator(&self, cluster: usize, at: &Point) -> Generator;

    /// The photon classes of the cluster's state ψ (unnormalized) at a point (`Model::photons`).
    fn photons(&self, cluster: usize, at: &Point, psi: &[C64]) -> Vec<Photon>;

    /// The absorption recoil's moments of a photon into the drawn real polarization ε (`Photon::recoil`).
    fn recoil(&self, photon: &Photon, polarization: &[f64; 3]) -> Recoil;

    /// The loss energy E_s = (ħ/m)κ²/2 + V_run(r; p(ψ)) of the cluster's state at a point (`Model::energy`, rad/µs).
    fn energy(&self, cluster: usize, at: &Point, psi: &[C64]) -> f64;

    /// The loss test at a point (`Model::lost`): outside the region and E_s above its far value.
    fn lost(&self, cluster: usize, at: &Point, psi: &[C64]) -> bool;

    /// The displayed rows of the cluster's state at a point (`Model::readout`): the step control's population norm.
    fn readout(&self, cluster: usize, at: &Point, psi: &[C64]) -> [f64; STATES];
}

/// A trajectory's state (SPEC §1.6, §1.7, §2.2).
#[derive(Clone, Debug, PartialEq)]
pub struct Trajectory {
    /// Position (µm).
    pub r: [f64; 3],
    /// Momentum over ħ (1/µm).
    pub kappa: [f64; 3],
    /// The internal state on the cluster's states, unnormalized between jumps: ‖ψ‖² is the probability of no jump since
    /// the last one.
    pub psi: Vec<C64>,
    /// The cluster ψ lives on.
    pub cluster: usize,
    /// Time (µs).
    pub time: f64,
    /// The waiting time's threshold θ ∈ (0, 1]: the next jump is where ‖ψ‖² reaches it.
    pub threshold: f64,
    /// The jumps made: the next one draws from `Stream::jump(seed, index, jumps + 1)`.
    pub jumps: u32,
    /// The time the atom was lost (SPEC §1.7: checked at every step's end and after every jump); None while held.
    pub lost: Option<f64>,
    /// The run's seed and the trajectory's number: its jumps' streams.
    pub seed: u64,
    pub index: u64,
    /// The step control's state (`adapt`): the next step's proposed length (µs); the steps made (every `step`, the
    /// doubling's three and the refused ones); the time from which the step cap left the steps unchecked (SPEC §2.2:
    /// that span is flagged).
    pub tau: f64,
    pub steps: u64,
    pub capped: Option<f64>,
    /// The work counted (`decide::Work`, real multiply-adds) of the integrator's own algorithms over the trajectory, and
    /// the model's generator assemblies it asked for (each costed by the model).
    pub work: f64,
    pub assemblies: u64,
}

impl Trajectory {
    /// A trajectory at a basis state (a ground state of the model basis) at (r, κ) and time 0: ψ the unit vector of that
    /// state on its cluster, the first threshold from jump 0's stream (SPEC §2.4).
    pub fn new<D: Dynamics>(model: &D, state: usize, r: [f64; 3], kappa: [f64; 3], seed: u64, index: u64) -> Trajectory {
        let _ = (model, state, r, kappa, seed, index);
        todo!()
    }
    /// The trajectory's point.
    pub fn point(&self) -> Point {
        Point { r: self.r, kappa: self.kappa, t: self.time }
    }
}

/// The reference's exact flow (A) of a step: the expansion point p (µm), the principal axes (unit rows), per axis the
/// curvature λ (rad/µs/µm²) and the gradient g at p along it (rad/µs/µm), and ħ/m.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Flow {
    pub centre: [f64; 3],
    pub axes: [[f64; 3]; 3],
    pub curvatures: [f64; 3],
    pub gradient: [f64; 3],
    pub hbar_over_mass: f64,
}

/// One step's path (`step`): the A flow from its start to the middle t_m, the kick there, the A flow from the kicked
/// middle to its end -- the splitting's continuous path, second order in τ like the step. Its window integrals of κ
/// (`Segment::moments`) are T's window means' (SPEC §1.7), exact on this path.
#[derive(Clone, Debug, PartialEq)]
pub struct Segment {
    /// The step's start time and its length (µs; negative for a reverse step).
    pub start: f64,
    pub tau: f64,
    /// The reference flow of the step.
    pub flow: Flow,
    /// (r, κ) at the start, and at the middle after the kick.
    pub from: ([f64; 3], [f64; 3]),
    pub kicked: ([f64; 3], [f64; 3]),
}

impl Segment {
    /// (r, κ) at the time t within the step.
    pub fn at(&self, t: f64) -> ([f64; 3], [f64; 3]) {
        let _ = t;
        todo!()
    }
    /// ∫κ_a dt and ∫κ_a² dt (1/µm·µs, 1/µm²·µs) per lab axis over [from, to] within the step, on its path.
    pub fn moments(&self, from: f64, to: f64) -> ([f64; 3], [f64; 3]) {
        let _ = (from, to);
        todo!()
    }
}

/// One step of τ from a trajectory's state (`step`).
#[derive(Clone, Debug)]
pub struct Step {
    /// The state at the step's end: the propagation runs to the end whatever the threshold; jumps, the loss and the step
    /// control's counters are the caller's.
    pub end: Trajectory,
    /// The first time (µs from the step's start) at which ‖ψ‖² reaches the threshold within the step, if it does.
    pub crossing: Option<f64>,
    /// The step's path.
    pub segment: Segment,
}

/// Why a step could not be made: the reference's fixed point did not contract within its derived bound, or the state left
/// the floating-point range. The step control halves τ (its more accurate level).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Refused {
    Reference,
    Overflow,
}

/// One step of τ (negative: the reverse step) from a trajectory's state (module documentation).
pub fn step<D: Dynamics>(model: &D, from: &Trajectory, tau: f64) -> Result<Step, Refused> {
    let _ = (model, from, tau);
    todo!()
}

/// The jump of a trajectory at its state (SPEC §1.6): from `Stream::jump(seed, index, jumps + 1)` the next threshold, the
/// class ∝ Σ_a‖u_a‖², the polarization across the direction by ‖Σ_a ε_a u_a‖², the fluctuation's signs, and the direction
/// from the dipole pattern tr G − n·Re(G)·n by exact rejection from the uniform sphere; then ψ ← f/‖f‖ on the landing
/// cluster and κ += −k_φ n + b_J + ξ, Cov ξ = M − b_J b_Jᵀ (`Dynamics::recoil`); then the loss test.
pub fn jump<D: Dynamics>(model: &D, trajectory: &mut Trajectory) {
    let _ = (model, trajectory);
    todo!()
}
