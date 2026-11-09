//! The atom's internal model at a point of its motion (SPEC §1.2–§1.8): the far light H_far, the one potential V_L, the trap and
//! its analytic Hessian, the reach; the units and their existence gates; the clusters and their frames; the elimination
//! (Reiter–Sørensen with Brillouin–Wigner dressing, first-order Doppler); the one generator assembly (H, G = ∇H_h, jump operators);
//! the readout, the hazard, the photons and their recoil.
//!
//! # Notation (SPEC §0 units)
//! States s: the 32 of the model basis, the ground states first (`GROUND` of them), H0 diagonal in it (energies E_s, rad/µs, in
//! the frame where the excited states rotate at ν_frame). The emission operators C_a (a = x, y, z, ground ← excited, 1/√µs),
//! D = Σ_a C_a†C_a on the excited states. Each frequency ω of the light is one coherent field E_ω(r) = Σ_{l∈ω} a_l(r) E_l (`field`)
//! and drives two components, each an excited ← ground coupling linear in that field: the co-rotating V_ω(r) = Σ_k E_ω,k(r) U_ω,k
//! at ν = δ_ω and the counter-rotating W_ω(r) = Σ_k E_ω,k(r)* U′_ω,k at ν = −(δ_ω + 2ν_frame) (`Component`). Frequencies are exact
//! integer coordinates on the request's frequency axes and ν_frame (`Beat`).
//!
//! # What is exact and what is folded
//! Every coupling is kept (it enters the cluster's generator as it is and the propagator resolves it) or folded at second order;
//! which is decided per unit (`Unit`, SPEC §5.2) by `solve`, within the existence gates (`Gate`, SPEC §1.5) this module evaluates.
//! A folded far component enters through its van Vleck shift and its Reiter–Sørensen jumps (`far`), linear in T_ω = E_ω E_ω† and
//! exact at the atom; a folded drive of a near carrier through its dressed response (`eliminate`), whose nonlinear dependence on
//! the far light and the other carriers is tabulated over the field's chart (`table`).
//!
//! # A generator along a step's path (propagation.md §1.2–§1.3)
//! Every term is a field's value at the point times plane-wave phases: absorbing beam l and emitting beam l′ it holds
//! e^{i(k_l − k_l′)·r}. On the step's straight path r_h + v(t − t_m) a cluster's generator is Σ_j M_j e^{iω_j(t − t_m)},
//! ω_j = Ω_j + q_j·v (`Part`): Ω_j the term's beat in the cluster's frames, q_j its wave vector less the Doppler frames'.
//! The envelopes and the tables are held at the point.
pub mod far;
pub mod potential;
pub mod reach;
pub mod units;
pub mod cluster;
pub mod eliminate;
pub mod assemble;
pub mod readout;
pub mod photon;

use crate::field::Light;
use crate::linalg::{C64, Mat};
use crate::request::Request;
pub use crate::request::{GROUND, STATES};

/// A frequency, exactly: integer coordinates on the request's frequency axes, then on the excited states' frame frequency ν_frame;
/// its value is 2π Σ_k n_k f_k (rad/µs, `Model::omega`). Equal frequencies have equal coordinates: a frame or a beat that is zero
/// in exact arithmetic is zero here (SPEC §1.4 frames), whatever the floating point makes of its value.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct Beat(pub Vec<i64>);

/// One component of the light: frequency `frequency` (an index into the field's frequencies), its co-rotating coupling
/// (`counter` false: V_ω e^{−iνt}, ν = δ_ω) or its counter-rotating one (W_ω e^{−iνt}, ν = −(δ_ω + 2ν_frame)).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Component { pub frequency: usize, pub counter: bool }

/// The lights on, per frequency of the field: L in V_L (SPEC §1.3). `Model::running` is every light, `Model::prepared` the trap
/// beam's frequency alone (before t = 0).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Lights(pub Vec<bool>);

/// A point of a trajectory: position r (µm), momentum κ = p/ħ (1/µm), time t (µs).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point { pub r: [f64; 3], pub kappa: [f64; 3], pub t: f64 }

/// A scalar's second-order expansion at a point, analytic: value, gradient, Hessian (rad/µs, rad/µs/µm, rad/µs/µm²).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quadratic { pub value: f64, pub gradient: [f64; 3], pub hessian: [[f64; 3]; 3] }

/// The trap (SPEC §1.3): the local minimum of the prepared potential the descent from the focus reaches, and its analytic
/// Hessian's eigenvectors and frequencies there.
#[derive(Clone, Debug, PartialEq)]
pub struct Trap {
    /// The trap's centre (µm from the focus).
    pub centre: [f64; 3],
    /// V_prep at the centre (rad/µs): V_min, the reach's floor.
    pub minimum: f64,
    /// The principal axes: unit rows, the Hessian's eigenvectors.
    pub axes: [[f64; 3]; 3],
    /// The angular frequencies along them (rad/µs): ω_i = √(λ_i ħ/m), λ_i > 0 the Hessian's eigenvalues (rad/µs/µm²).
    pub frequencies: [f64; 3],
}

/// The atoms' reach (SPEC §1.8): the sublevel set {r : V_prep(r) − V_min ≤ ln(1/s) k_B T_r} of the prepared potential, with the
/// momenta of that kinetic energy. Its extrema are provable bounds: interval subdivision of `cell` with Lipschitz remainders.
#[derive(Clone, Debug, PartialEq)]
pub struct Reach {
    /// T_r (µK): the temperature the atoms reach (never lowered; the solve's trials read it).
    pub temperature_uk: f64,
    /// s: the unsampled share (SPEC §3).
    pub share: f64,
    /// ln(1/s) k_B T_r (rad/µs): the energy above V_min the reach holds, in position and in kinetic energy.
    pub energy: f64,
    /// V_min (rad/µs).
    pub minimum: f64,
    /// A box (µm, [low, high] per lab axis) that provably holds the sublevel set: where the subdivisions start.
    pub cell: [[f64; 2]; 3],
    /// The largest speed |v| (µm/µs) in the reach: √(2 (ħ/m) energy).
    pub speed: f64,
}

/// A unit (SPEC §5.2): one coupling the solve keeps or folds. The more exact level of every fold is keeping it.
#[derive(Clone, Debug, PartialEq)]
pub enum Unit {
    /// A drive of near carrier `carrier` (a co-rotating component) from ground state `ground` into excited state `excited`.
    /// Kept: the excited state is explicit and the drive enters the generator as it is.
    Drive { carrier: Component, excited: usize, ground: usize },
    /// A coupling between ground states `a` ← `b` at one beat: the second-order term absorbing carrier `from` and emitting
    /// carrier `to` (a Raman or a Zeeman coupling of the light, one state's beat of two carriers), or with both None a static
    /// coupling of the far light. Between two clusters, kept joins them; within one, kept resolves its beat (a far beat).
    Term { a: usize, b: usize, from: Option<Component>, to: Option<Component> },
    /// A far component's fold (its van Vleck shift and Reiter–Sørensen jumps). Kept: a co-rotating component is a near carrier;
    /// a counter-rotating one cannot be kept (its fold must exist, else the request is refused).
    Far { component: Component },
    /// The beat T^{ωω′} e^{−i(ν − ν′)t} of two far components' fields in the far light's shifts. Kept: a beat of the generator;
    /// folded: dropped.
    FarBeat { first: Component, second: Component },
    /// A static coupling of an explicit excited state `excited` to another excited state `other`. Kept: `other` is explicit too.
    Excited { excited: usize, other: usize },
    /// A folded drive (`carrier`, `excited` ← `ground`) into an explicit excited state: a pair of channels with that state's kept
    /// drives. Kept: the drive is kept.
    Channel { carrier: Component, excited: usize, ground: usize },
    /// The coherence of the photons landing in ground state `landing` at two frame frequencies `first` and `second` (in the
    /// landing cluster's frames). Kept: one class, their interference with its beat; folded: two classes.
    Coherence { landing: usize, first: Beat, second: Beat },
}

/// A unit's existence gate over the reach (SPEC §1.5): provable upper bounds over the reach in phase space. The fold exists only
/// where both are below ½ (`Gate::exists`); else the unit is kept.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gate {
    /// The static Stewart bound: the coherent peak's upper bound of S = 2m² (a drive's: its ground state's total S_b over its
    /// folded drives).
    pub saturation: f64,
    /// The motional bound: the upper bound of a = |dΔE/dt|/|ΔE|², the sweep v·∇ΔE + k·v̇.
    pub sweep: f64,
}

/// A folded unit's errors on the outputs at one trajectory state (SPEC §1.5 → §5.2): its small parameter measured there and mapped
/// by the leading-order relations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnitError {
    /// The small parameter measured: S (a drive's ground state), m² (a coupling's mixing), a (the sweep), (w/σ)² (the Doppler order).
    pub parameter: f64,
    /// The absolute error it makes on the display rows' populations: ≤ S_b; m²(p_a + p_b); m²√(p_i p_j); m².
    pub population: f64,
    /// The relative error it makes on T: ≈ √2 S; a; (w/σ)².
    pub temperature: f64,
}

/// The counted work of a unit kept and folded (SPEC §5.1: the multiply-adds the engine's algorithms count, per µs of one
/// trajectory at the samples' pace): kept − folded is the price of keeping it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnitWork { pub kept: f64, pub folded: f64 }

/// A trajectory state the units' errors are measured at (the solve's trial states).
#[derive(Clone, Debug, PartialEq)]
pub struct Sample { pub cluster: usize, pub at: Point, pub psi: Vec<C64> }

/// A cluster (SPEC §1.4): the states the kept couplings join, its ground states first and then its explicit excited states;
/// a trajectory's state space between two jumps.
#[derive(Clone, Debug, PartialEq)]
pub struct Cluster {
    /// Its states (indices of the model basis): the ground states, then the explicit excited states.
    pub states: Vec<usize>,
    /// How many of them are ground states.
    pub ground: usize,
    /// Each state's frame θ_s, exactly: its amplitude x_s = e^{iθ_s t} ψ_s rotates at E_s − θ_s in the generator.
    pub frames: Vec<Beat>,
    /// Each state's Doppler frame K_s (1/µm): on a path r_h + v(t − t_m) the state is x_s e^{iK_s·v(t − t_m)}, a term a ← b of
    /// wave vector q at q − (K_a − K_b). Exact where every pair of its states holds one wave vector; the rest are path beats.
    pub waves: Vec<[f64; 3]>,
}

/// One part of a cluster's generator at a point (SPEC §1.4): on the step's path H(t) = Σ_j M_j e^{iω_j(t − t_m)},
/// ω_j = Ω_j + q_j·v, and the gradient of its Hermitian part G(t) = Σ_j G_j e^{iω_j(t − t_m)}. Parts come in pairs (Ω, q),
/// (−Ω, −q) (one part where both vanish), their Hermitian parts paired, G_{−j} = G_j†.
#[derive(Clone, Debug, PartialEq)]
pub struct Part {
    /// Its beat in the cluster's frames, exactly.
    pub beat: Beat,
    /// Ω (rad/µs): the beat's value.
    pub omega: f64,
    /// q (1/µm): its wave vector less the Doppler frames' K_a − K_b.
    pub wave: [f64; 3],
    /// M_j at the point, n × n on the cluster's states (rad/µs), its phase e^{i(Ω t + q·r)} there: the Hermitian part of that
    /// beat with its decay −(i/2) Σ J†J.
    pub generator: Mat,
    /// G_j, the gradient of its Hermitian part at the point (rad/µs/µm), per lab axis.
    pub force: [Mat; 3],
}

/// A cluster's generator at a point with some lights on: its parts, from the one assembly (`Model::assemble`).
#[derive(Clone, Debug, PartialEq)]
pub struct Generator { pub cluster: usize, pub at: Point, pub parts: Vec<Part> }

/// One class of photons (SPEC §1.6) of a cluster's state at a point: the photons of one frame frequency landing in one cluster,
/// folded and explicit amplitudes added. (A class of one frequency landing in two clusters is two of these: the joint law of
/// direction, polarization and landing is the same either way, ∝ ‖P_C Σ_a ε_a u_a‖².)
#[derive(Clone, Debug, PartialEq)]
pub struct Photon {
    /// The cluster the jump lands the atom in.
    pub landing: usize,
    /// |k_φ| (1/µm): the emission recoil is −k_φ n.
    pub wave_number: f64,
    /// u_a = J_{a,φ} ψ (a = x, y, z) on the landing cluster's states: the class's probability ∝ Σ_a ‖u_a‖², the direction's
    /// density ∝ tr G − n·Re(G)·n (G_ab = ⟨u_a|u_b⟩), the polarization ε across n by ‖Σ_a ε_a u_a‖², the state after f = Σ_a ε_a u_a.
    pub amplitudes: [Vec<C64>; 3],
    /// ∇u_a of the folded part, whose absorption's mean the jump carries (`Photon::recoil`): gradients[i][a], i the lab axis.
    pub gradients: [[Vec<C64>; 3]; 3],
    /// The explicit states' absorption sources, whose mean the Ehrenfest force carries: per source its amplitudes (u_a's part
    /// the source holds, with the square root of its share) and its log-gradient ∇ln σ (1/µm, complex per lab axis).
    pub explicit: Vec<([Vec<C64>; 3], [C64; 3])>,
}

/// The absorption recoil's first two moments of a jump into f = Σ_a ε_a u_a (SPEC §1.6, one rule): the mean the jump carries
/// (the folded part's, Im⟨f|∇f_F⟩/‖f‖²) and the total second moment M (1/µm²), the local second Kramers–Moyal coefficient
/// (the folded part's Re⟨∇f_F|∇f_F⟩/‖f‖² and the explicit sources' Re[∇ln σ*⊗∇ln σ] by their shares). The kick is the mean plus
/// a zero-mean fluctuation of covariance M − mean meanᵀ, positive semidefinite by Cauchy–Schwarz.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Recoil { pub mean: [f64; 3], pub second: [[f64; 3]; 3] }

/// The model: a request's states and light with a set of units kept (SPEC §1).
pub struct Model {
    // Built by `Model::new`: the far light, the near carriers, the clusters, the eliminations.
}

impl Model {
    /// The candidate units of a request (structure only; their gates need a model and a reach, `Model::gates`).
    pub fn units(request: &Request, light: &Light) -> Vec<Unit> { let _ = (request, light); todo!() }

    /// The model of a request with the units `kept` (one flag per unit): the clusters they join, the frames, the eliminations of
    /// what they fold (SPEC §1.4). Refused where a fold cannot be made: a counter-rotating component whose fold does not exist.
    pub fn new(request: &Request, light: &Light, units: &[Unit], kept: &[bool]) -> Result<Model, String> {
        let _ = (request, light, units, kept); todo!()
    }

    /// Every light: L_run.
    pub fn running(&self) -> Lights { todo!() }
    /// The trap beam's frequency alone: L_prep.
    pub fn prepared(&self) -> Lights { todo!() }

    /// V_L(r; p) = Σ_b p_b [h_L(r) − h_{L∖trap}(r)]_bb (SPEC §1.3, rad/µs): h the eliminated generator's static Hermitian part from
    /// the one assembly, p the ground states' populations (normalized on them), L `lights`, the trap light the whole field of the
    /// trap beam's frequency. The one potential: the loss energy, the prepared potential, the trap and the reach take it.
    pub fn potential(&self, r: [f64; 3], p: &[f64; GROUND], lights: &Lights) -> f64 { let _ = (r, p, lights); todo!() }

    /// V_L(r; p)'s second-order expansion at r, analytic: the far light's from the field's exact derivatives, the near light's from
    /// its table's interpolant. The integrator's reference takes it for `prepared()` at its reference state; the trap its Hessian.
    pub fn expansion(&self, r: [f64; 3], p: &[f64; GROUND], lights: &Lights) -> Quadratic { let _ = (r, p, lights); todo!() }

    /// V_L(r; p) far from the trap: every focused beam's amplitude zero. A lost atom's energy is above it (SPEC §1.7).
    pub fn asymptote(&self, p: &[f64; GROUND], lights: &Lights) -> f64 { let _ = (p, lights); todo!() }

    /// The loss energy E_s = ħ²κ²/2m + V_run(r; p(ψ)) of a cluster's state at a point (SPEC §1.3), p(ψ) its ground populations.
    pub fn energy(&self, cluster: usize, at: &Point, psi: &[C64]) -> f64 { let _ = (cluster, at, psi); todo!() }

    /// The loss test (SPEC §1.7): outside the region (the field's intensity rule) and E_s above its asymptote.
    pub fn lost(&self, cluster: usize, at: &Point, psi: &[C64]) -> bool { let _ = (cluster, at, psi); todo!() }

    /// The trap of the prepared potential for the ground populations p (SPEC §1.3): the descent from the focus to the local
    /// minimum of V_prep(r; p), the analytic Hessian's eigenvectors and frequencies there. Refused where the descent reaches no
    /// minimum ("光镊不约束原子"), never the focus instead.
    pub fn trap(&self, p: &[f64; GROUND]) -> Result<Trap, String> { let _ = p; todo!() }

    /// The reach of a trap at temperature T_r (µK) and unsampled share s (SPEC §1.8).
    pub fn reach(&self, trap: &Trap, p: &[f64; GROUND], temperature_uk: f64, share: f64) -> Reach { let _ = (trap, p, temperature_uk, share); todo!() }

    /// Each unit's existence gate over the reach (SPEC §1.5), provable bounds, at this model's clusters and dressing.
    pub fn gates(&self, units: &[Unit], reach: &Reach) -> Vec<Gate> { let _ = (units, reach); todo!() }

    /// Each folded unit's errors at each sample, [unit][sample] (kept units: zero): its small parameter measured where its error
    /// is made, mapped to the outputs (SPEC §1.5).
    pub fn errors(&self, units: &[Unit], samples: &[Sample]) -> Vec<Vec<UnitError>> { let _ = (units, samples); todo!() }

    /// Each unit's counted work kept and folded at the samples' pace.
    pub fn work(&self, units: &[Unit], samples: &[Sample]) -> Vec<UnitWork> { let _ = (units, samples); todo!() }

    /// The clusters.
    pub fn clusters(&self) -> &[Cluster] { todo!() }

    /// The cluster a ground state belongs to.
    pub fn cluster_of(&self, ground: usize) -> usize { let _ = ground; todo!() }

    /// The one generator assembly (SPEC §1.4): a cluster's generator at a point with `lights` on, its parts by beat and wave vector,
    /// H with the decay −(i/2) Σ J†J of its photons (Lindblad by construction) and G = ∇H_h. The dynamics, the force, the hazard,
    /// the readout and the potential all take it.
    pub fn assemble(&self, cluster: usize, at: &Point, lights: &Lights) -> Generator { let _ = (cluster, at, lights); todo!() }

    /// The photon classes of a cluster's state ψ at a point, every light on (SPEC §1.6).
    pub fn photons(&self, cluster: usize, at: &Point, psi: &[C64]) -> Vec<Photon> { let _ = (cluster, at, psi); todo!() }

    /// The readout (SPEC §1.7): the 32 populations of a cluster's state ψ at a point, ρ − ½{Y, ρ} on its states plus the folded
    /// excited populations, ρ = ψψ†/‖ψ‖².
    pub fn readout(&self, cluster: usize, at: &Point, psi: &[C64]) -> [f64; STATES] { let _ = (cluster, at, psi); todo!() }

    /// ħ/m (µm²/µs).
    pub fn hbar_over_mass(&self) -> f64 { todo!() }

    /// A frequency's value (rad/µs).
    pub fn omega(&self, f: &Beat) -> f64 { let _ = f; todo!() }
}

impl Photon {
    /// The absorption recoil's moments of the jump into f = Σ_a ε_a u_a (`Recoil`), ε the drawn real polarization.
    pub fn recoil(&self, polarization: &[f64; 3]) -> Recoil { let _ = polarization; todo!() }
}

/// The hazard λ = Σ_φ Σ_a ‖J_{a,φ}ψ‖²/‖ψ‖² (SPEC §1.4, the one definition): ⟨ψ|i(H − H†)|ψ⟩/‖ψ‖² for the generator H at that instant,
/// the decay being the jumps' Σ J†J by construction (tested, never checked at run time).
pub fn hazard(h: &Mat, psi: &[C64]) -> f64 { let _ = (h, psi); todo!() }
