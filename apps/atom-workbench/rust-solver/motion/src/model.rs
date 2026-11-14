//! The atom's internal model at a point of its motion (SPEC §1.2–§1.8): the far light H_far, the one potential V_L, the trap and
//! its analytic Hessian, the reach; the units the solve keeps or folds; the clusters and their frames; the elimination (the
//! one fold, elim.rs, and its normalized series; first-order Doppler); the one generator assembly (H, G = ∇H_h, jump operators);
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
//! which is decided per unit (`Unit`, SPEC §5.2) by `solve`: the prior's closure at the reach's atoms (elim.rs `Model::prior`,
//! criteria one and two) and the units' errors read at the run's states (elim.rs `Model::readings`), the model's domain
//! checked here (`Model::settled`). No light is far or near: each drive -- one component's coupling of one ground state to
//! one excited state -- is folded with the bare levels, folded with the levels at the atom, or explicit (`Level`). A drive
//! folded with the bare levels enters through its second-order shift and its Reiter–Sørensen jumps (`far`), linear in
//! T_ω = E_ω E_ω† and exact at the atom; one folded with the levels at the atom through its dressed response (`eliminate`), a
//! nonlinear function of the constant forms and the other carriers at the atom. That elimination is evaluated at every point
//! as it stands (`Run::new`), or taken from its tables over the coordinates it depends on, refined over the atoms' reach to
//! the resolution of the largest ensemble (`tabulate`, `Run::tabulated`; SPEC §2.3): one assembly either way (`assemble`).
//!
//! # A generator along a step's path (propagation.md §1.2–§1.3)
//! Every term is a field's value at the point times plane-wave phases: absorbing beam l and emitting beam l′ it holds
//! e^{i(k_l − k_l′)·r}. On the step's straight path r_h + v(t − t_m) each beam's plane wave turns by e^{ik_l·v(t − t_m)}, each
//! state's Doppler frame by e^{iK_s·v(t − t_m)}, and each term by its beat in the cluster's frames; the envelopes and the tables
//! are held at the point. The generator on the path is given by its fields at a time (assemble.rs `Middle` and `Path`, the
//! step's `step::Middle`): the point's values once, the momentum's Doppler weights and path rates, and at a time the sum of
//! its terms there -- no list of beating matrices is made, whatever the number of beams.
pub mod far;
pub mod potential;
pub mod reach;
pub mod units;
pub mod cluster;
pub mod eliminate;
pub(crate) mod elim;
pub mod assemble;
pub mod readout;
pub mod photon;
pub mod tabulate;
pub(crate) mod optical;
pub(crate) mod jet;
pub(crate) mod own;
#[cfg(test)]
pub(crate) mod testing;
#[cfg(test)]
mod near_tests;
#[cfg(test)]
mod table_tests;
#[cfg(test)]
mod trap_tests;

use crate::field::Light;
use crate::linalg::{C64, Mat};
use crate::request::Request;
use std::sync::{Arc, Mutex};
pub use assemble::Middle;
pub use crate::request::{GROUND, STATES};

/// A frequency, exactly: rational coordinates on the request's frequency axes, then on the excited states' frame frequency
/// ν_frame -- integer numerators n_k over one positive denominator d, in lowest terms --; its value is 2π Σ_k n_k f_k / d
/// (rad/µs, `Model::omega`). Equal frequencies have equal coordinates: a frame or a beat that is zero in exact arithmetic is
/// zero here (SPEC §1.4 frames), whatever the floating point makes of its value. The light's frequencies and their beats are
/// whole (d = 1); a frame is a fraction where the couplings it is taken from close a cycle off resonance, the cycle's
/// detuning shared by its |C| couplings (cluster.rs, # Frames): d divides a product of cycle lengths.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Beat {
    n: Vec<i64>,
    d: i64,
}

impl Beat {
    /// The frequency of the integer coordinates `n`.
    pub fn whole(n: Vec<i64>) -> Beat {
        Beat { n, d: 1 }
    }
    /// n/d in lowest terms, d positive (d ≠ 0).
    fn reduced(mut n: Vec<i64>, mut d: i64) -> Beat {
        let gcd = |mut a: i64, mut b: i64| {
            while b != 0 {
                (a, b) = (b, a % b);
            }
            a.abs()
        };
        let g = n.iter().fold(d, |g, &x| gcd(g, x));
        if g > 1 {
            n.iter_mut().for_each(|x| *x /= g);
            d /= g;
        }
        if d < 0 {
            n.iter_mut().for_each(|x| *x = -*x);
            d = -d;
        }
        Beat { n, d }
    }
    /// self + s·other.
    pub fn plus(&self, s: i64, other: &Beat) -> Beat {
        if self.d == other.d {
            return Beat::reduced(self.n.iter().zip(&other.n).map(|(&a, &b)| a + s * b).collect(), self.d);
        }
        Beat::reduced(self.n.iter().zip(&other.n).map(|(&a, &b)| a * other.d + s * b * self.d).collect(), self.d * other.d)
    }
    pub fn negated(&self) -> Beat {
        Beat { n: self.n.iter().map(|&a| -a).collect(), d: self.d }
    }
    pub fn is_zero(&self) -> bool {
        self.n.iter().all(|&a| a == 0)
    }
    /// The sign of its first coordinate that is not zero (zero for the zero frequency): a beat's orientation.
    pub fn sign(&self) -> i64 {
        self.n.iter().find(|&&x| x != 0).map_or(0, |x| x.signum())
    }
    /// Its value on the coordinates' values `axes`: Σ_k n_k a_k / d, from the integer numerators (never a difference of
    /// two values).
    pub fn value(&self, axes: &[f64]) -> f64 {
        self.n.iter().zip(axes).map(|(&n, &a)| n as f64 * a).sum::<f64>() / self.d as f64
    }
}

impl cluster::Exact for Beat {
    fn plus(&self, s: i64, other: &Beat) -> Beat {
        Beat::plus(self, s, other)
    }
}

/// One component of the light: frequency `frequency` (an index into the field's frequencies), its co-rotating coupling
/// (`counter` false: V_ω e^{−iνt}, ν = δ_ω) or its counter-rotating one (W_ω e^{−iνt}, ν = −(δ_ω + 2ν_frame)).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Component { pub frequency: usize, pub counter: bool }

impl Component {
    /// Whether the model can take a drive of this component more exactly than its fold with the bare levels (the model's
    /// domain, `Model::settled`): a co-rotating one, at the atom or explicit. A counter-rotating one is only ever folded with
    /// the bare levels -- a keep of it is refused (`Model::new`).
    pub fn keepable(&self) -> bool {
        !self.counter
    }
}

/// How one frequency's light is on (`Lights`): off; on as its coherent field; or on with its beams' relative phases averaged --
/// each beam with its own intensity, no interference between beams: the smooth shift of a light whose fringes the atoms fly over
/// (the reported trap's, SPEC §1.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lit {
    Off,
    Coherent,
    Averaged,
}

/// The lights, per frequency of the field: L in V_L (SPEC §1.3) -- the light set as data. `Model::running` has every light
/// coherent, `Model::prepared` the trap beam's frequency alone (before t = 0), `Model::reported` the trap light coherent and the
/// others averaged (the reported trap).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Lights(pub Vec<Lit>);

impl Lights {
    /// Whether a frequency's light is on.
    pub fn on(&self, frequency: usize) -> bool {
        self.0[frequency] != Lit::Off
    }
    /// Whether the beam pair (j, i) of one frequency (its beam order) enters: any pair of a coherent field, a beam with itself of
    /// an averaged one.
    pub fn pair(&self, frequency: usize, j: usize, i: usize) -> bool {
        match self.0[frequency] {
            Lit::Off => false,
            Lit::Coherent => true,
            Lit::Averaged => j == i,
        }
    }
    /// Whether two frequencies' cross terms enter: both coherent.
    pub fn cross(&self, first: usize, second: usize) -> bool {
        self.0[first] == Lit::Coherent && self.0[second] == Lit::Coherent
    }
}

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
    /// The angular frequencies along the lab axes x, y, z (rad/µs): √(∂²V/∂x_a² ħ/m) at the centre, the curvature along each
    /// lab axis (the T windows' and n̄'s periods, the classical flag).
    pub lab: [f64; 3],
}

/// A proven envelope of the prepared potential (the initial ensemble's exact rejection sampling): with the trap light's
/// static form a†Q_b a in its beams' amplitudes a_l (far.rs: V_prep,b = Σ_l'l a_l'* a_l [P_l'l]_bb), −V_prep(r; p) ≤
/// Σ_l'l |Q_l'l| √(A_l'(r) A_l(r)) ≤ (Σ_l'l |Q_l'l|) max_l A_l(r), since |a†Qa| ≤ Σ|Q_l'l||a_l'||a_l| and a paraxial beam's
/// |a_l(r)|² ≤ A_l(r) = 1/(1 + u_l(r)), u_l = 2ρ²/w₀² + ζ²/z_R² (|a|² = e^{−t/(1+s)}/(1 + s), s = ζ²/z_R², t = 2ρ²/w₀², and
/// e^{−x} ≤ 1/(1 + x) give |a|² ≤ 1/(1 + s + t); a plane wave's A = 1).
#[derive(Clone, Debug, PartialEq)]
pub struct Envelope {
    /// The trap light's beams (indices into the request's beams).
    pub beams: Vec<usize>,
    /// |Q_l'l| (rad/µs), row-major over `beams`.
    pub weights: Vec<f64>,
    /// The far static form's own Q_l'l = [P_l'l]_p (rad/µs), row-major over `beams`, and the near light shift's bound per
    /// pair beside it (eliminate.rs `Model::near_envelope`): how far V_prep moves over a cell is read of the far part's own
    /// form (reach.rs `prepared_own`).
    pub(crate) form: Vec<crate::linalg::C64>,
    pub(crate) near: Vec<f64>,
}

/// The atoms' reach (SPEC §1.8): the sublevel set {(r, κ) : V_prep(r) + (ħ/m)κ²/2 − V_min ≤ L k_B T_r} of the prepared
/// energy, L the level a thermal ensemble's share s lies above (reach.rs, # The reach's level) -- within the trap's region
/// where that level is at or above the potential's far value, and outside the region the sublevel set of a level below
/// it (reach.rs, # The trap's region). Its extrema are provable bounds over its cells, made side by side (reach.rs, # A
/// cell's sides); what it leaves out of the initial ensemble beyond that is counted on the ensemble (reach.rs
/// `Model::left_out`).
#[derive(Clone, Debug, PartialEq)]
pub struct Reach {
    /// T_r (µK): the temperature the atoms reach (never lowered; the solve's trials read it).
    pub temperature_uk: f64,
    /// The kinetic temperature (µK) along each principal axis of the trap of the ensemble the reach is made for,
    /// s_i² T_r with the Wigner factor's s_i² = x_i coth x_i, x_i = ħω_i/2k_BT_r (reach.rs, # The reach's level): T_r
    /// where ħω_i ≪ k_BT_r, the zero point's ħω_i/2k_B where it is cold. Their mean is the ensemble's rms speed's
    /// temperature, their largest what no lab axis's T exceeds while the atoms are the ensemble's.
    pub kinetic_uk: [f64; 3],
    /// L k_B T_r (rad/µs): the energy above V_min the reach holds, potential and kinetic together -- the (1 − s) quantile
    /// of an atom's energy in the ensemble as drawn at T_r, s the share the most trajectories of a solve leave unseen
    /// (SPEC §3; reach.rs, # The reach's level: classically Q(3, L) = s, L = 12.89, where the trap holds along three
    /// axes).
    pub energy: f64,
    /// The energy above V_min it holds outside the trap's region (rad/µs; field.rs, # The region): `energy` itself where
    /// the level is below the far value -- the sublevel set whole --, else that of a level below the far value, or none
    /// (−∞): an atom outside the region above the far value is lost (SPEC §1.7), and one below it is bound.
    pub outside: f64,
    /// V_min (rad/µs).
    pub minimum: f64,
    /// A box (µm, [low, high] per lab axis) that provably holds the reach: where the subdivisions start.
    pub cell: [[f64; 2]; 3],
    /// The largest speed |v| (µm/µs) in the reach: √(2 (ħ/m) energy).
    pub speed: f64,
    /// The ground populations p of the prepared potential it is the sublevel set of.
    pub populations: [f64; GROUND],
}

/// A unit (SPEC §5.2): one coupling the solve keeps or folds. The more exact level of every fold is keeping it. No unit names a
/// kind of light: every drive and every coupling of two states is its own unit, judged at its own distance (elim.rs: criteria
/// one and two).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Unit {
    /// The drive of one component of the light from ground state `ground` into excited state `excited`, folded with the bare
    /// levels: its shift and its jumps are constants (far.rs). Kept: its fold takes the levels at the atom (`Level::Atom`).
    Bare { component: Component, excited: usize, ground: usize },
    /// The same drive folded with the levels at the atom (eliminate.rs: the dressed response, tabulated). Kept: the excited
    /// state is explicit and the drive enters the generator as it is. Folded onto a state another kept drive makes explicit, it
    /// is one leg of a Raman pair through that state, read across the two-photon distance d (far.rs, # A fold onto an
    /// explicit state): a P-layer link, explicit where criterion two's 2(a/d)² is beyond the share (`Model::settled`).
    Drive { component: Component, excited: usize, ground: usize },
    /// Every second-order coupling `a` ← `b` at one exact beat (∝ e^{i beat t}): of two ground states through the excited states
    /// -- one component's own (beat zero: the light's static couplings, the request's outside forms with them) or two
    /// components' (a Raman pair's; where a = b a level's beating shift) --, or of two excited states through the ground states.
    /// Kept: it is in the generator, and its two states are one cluster's, their frames the beat apart.
    Term { a: usize, b: usize, beat: Beat },
    /// The coherence of the photons landing in ground state `landing` at two frame frequencies `first` and `second` (in the
    /// landing cluster's frames). Kept: one class, their interference with its beat; folded: two classes.
    Coherence { landing: usize, first: Beat, second: Beat },
    /// An excited hyperfine block F′ (elim.rs `Object::Block`, `Model::blocks`) for one co-rotating component of the near
    /// light, `component`, and the ground states `grounds` (a set, bit g for ground state g: the cluster whose state read
    /// it, elim.rs `Model::readings`): kept, the block's states are explicit and every drive of that component into them
    /// from those ground states is kept as it is (`Model::block_drives`, `Unit::Drive`); its frame is the frames' (Karp's,
    /// cluster.rs) on the explicit set. The block is explicit at the replica of that component in the group that holds
    /// its part (ELIMINATION §2–§3, v2 5; elim.rs # Groups): a drive of another component, or from another group -- the
    /// other ground hyperfine level, 6.8 GHz away --, turns at its own frequency, where the block is a Q state folded like
    /// any other.
    Block { block: usize, component: Component, grounds: u32 },
    /// The fold's series about its reference at its first order (assemble.rs, # The normalized generator: H₀ − N₁E₀):
    /// folded, the evaluation takes order 0; kept, order 1. Its error is what order 1 takes away of the evaluation's error
    /// against the definition, measured at the run's states (elim.rs `Model::readings`); a higher order is the nearest
    /// block's keep (`Unit::Block`).
    Series,
    /// The normalization's series S⁻¹ = Σ_{j≤J} b_j X^j at the order `order` (assemble.rs `normal`): kept, the
    /// evaluation takes J = `order`; its error is what order J takes away of the evaluation's error against the
    /// definition, measured at the run's states (elim.rs `Model::readings`).
    Normalization { order: usize },
}

/// How a drive is taken (SPEC §1.4): folded with the bare levels, folded with the levels at the atom, or explicit. A drive has
/// exactly one, the highest its kept units give it (`Model::new`): each is the more exact level of the one before.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Level {
    Bare,
    Atom,
    Explicit,
}

/// A folded unit's errors on the outputs at one trajectory state (SPEC §1.5 → §5.2): the part its keep would take away of
/// the model's error measured there (elim.rs `Model::readings`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnitError {
    /// The absolute error it makes on the display rows' populations.
    pub population: f64,
    /// The relative error it makes on T.
    pub temperature: f64,
}
impl UnitError {
    /// No error.
    pub const NONE: UnitError = UnitError { population: 0.0, temperature: 0.0 };
}

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
    /// Each state's frame θ_s, exactly: ψ_s = e^{−iθ_s t} x_s, so that x_s holds E_s − θ_s on the diagonal and a term a ← b of
    /// beat Ω (∝ e^{iΩt}) is at Ω + θ_a − θ_b.
    pub frames: Vec<Beat>,
    /// Each state's Doppler frame along the kept edges' tree (1/µm; cluster.rs, # Doppler frames): the frames K_s where every
    /// pair of its states holds one wave vector and the loops close -- on a path r_h + v(t − t_m) the state is
    /// ψ_s = e^{iK_s·v(t − t_m)} x_s --, and what a plan's frames correct (assemble.rs `Plan::waves`) where they do not.
    pub waves: Vec<[f64; 3]>,
    /// The tree's Doppler frames exactly (`Wave`).
    pub(crate) waves_exact: Vec<Beat>,
}

/// One class of photons (SPEC §1.6) of a cluster's state at a point: the photons of one frame frequency landing in one cluster,
/// folded and explicit amplitudes added. (A class of one frequency landing in two clusters is two of these: the joint law of
/// direction, polarization and landing is the same either way, ∝ ‖P_C Σ_a ε_a u_a‖².)
#[derive(Clone, Debug, PartialEq)]
pub struct Photon {
    /// The class's place among its cluster's classes: the same for every state of the cluster (a state the class's
    /// operator annihilates has no photon of it).
    pub class: usize,
    /// The cluster the jump lands the atom in.
    pub landing: usize,
    /// |k_φ| (1/µm), 2π/c times the class's photon frequency (photon.rs, # A class's photon): the emission recoil is −k_φ n.
    pub wave_number: f64,
    /// k_c (1/µm): the largest wave number of the beams whose light the class's folded jumps absorb -- what one absorbed
    /// photon carries at most, the bound of the mean a jump carries (`Photon::moments`); zero for a class of explicit
    /// states' photons alone, whose absorption's mean is the Ehrenfest force's.
    pub absorbed: f64,
    /// u_a = J_{a,φ} ψ (a = x, y, z) on the landing cluster's states: the class's probability ∝ Σ_a ‖u_a‖², the direction's
    /// density ∝ tr G − n·Re(G)·n (G_ab = ⟨u_a|u_b⟩), the polarization ε across n by ‖Σ_a ε_a u_a‖², the state after f = Σ_a ε_a u_a.
    pub amplitudes: [Vec<C64>; 3],
    /// ∇u_a of the folded part (`Photon::moments`): gradients[i][a], i the lab axis; empty where none were asked.
    pub gradients: [[Vec<C64>; 3]; 3],
}

/// What the light a class's photons absorb does to the momentum per unit time, at the state the photons were made of and
/// in its norm (divide by ‖ψ‖²; SPEC §1.6): the local Kramers–Moyal coefficients of the method's own expansion, none of
/// them divided by the class's rate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Moments {
    /// W = Σ_a ‖u_a‖²: the class's rate.
    pub weight: f64,
    /// Σ_a Im⟨u_a|∇u_a⟩ (1/µm): the folded part's scattering force.
    pub mean: [f64; 3],
    /// Σ_a Re⟨∇u_a|⊗|∇u_a⟩ (1/µm²): the second moment's rate.
    pub second: [[f64; 3]; 3],
    /// s b (1/µm): what a jump of the class carries of its conditional mean b = mean/W -- all of it up to one absorbed
    /// photon's momentum, s = min(1, k_c/|b|).
    pub kick: [f64; 3],
}

/// The light's action on the momentum per unit time beside the Ehrenfest force and the jumps' kicks (SPEC §1.6), at a
/// state and a point: the scattering force the jumps do not carry (1/µm/µs), the second moment's rate they do not carry,
/// 2D (1/µm²/µs; positive semidefinite by Cauchy–Schwarz), and the rounding of 2D's entries as they are summed -- `size`
/// the largest sum of a diagonal entry's terms' moduli, `links` the operations behind an entry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Scattering { pub force: [f64; 3], pub diffusion: [[f64; 3]; 3], pub size: f64, pub links: usize }

impl Scattering {
    /// No light scattered.
    pub const NONE: Scattering = Scattering { force: [0.0; 3], diffusion: [[0.0; 3]; 3], size: 0.0, links: 0 };
}

/// The model: a request's states and light with a set of units kept (SPEC §1).
pub struct Model {
    pub(crate) light: Light,
    pub(crate) axes: far::Axes,
    /// The light's distinct wave vectors up to their sign (1/µm): the coordinates of every exact wave vector (`Wave`); per request
    /// beam its own with its sign -- a beam and its counter-propagating one are ±1 on one coordinate, so that what their wave vectors
    /// cancel in is exactly zero. Two beams' wave vectors are one where they differ by their own rounding (`one_wave`).
    pub(crate) waves: Vec<[f64; 3]>,
    pub(crate) beam_wave: Vec<(usize, i64)>,
    pub(crate) hbar_over_mass: f64,
    /// The bare energies E_s (H0's diagonal).
    pub(crate) energies: Vec<f64>,
    /// H0 less every frequency's outside form at the focus: the states with every light off.
    pub(crate) base: Mat,
    pub(crate) emission: [Mat; 3],
    /// D on the excited states.
    pub(crate) decay: Mat,
    /// Per frequency its light as the folds with the bare levels take it (far.rs).
    pub(crate) tones: Vec<far::Tone>,
    /// The units kept.
    pub(crate) kept: Vec<Unit>,
    pub(crate) clusters: Vec<Cluster>,
    /// Per ground state its cluster.
    pub(crate) membership: [usize; GROUND],
    /// The carriers: the co-rotating components with a drive not bare (eliminate.rs).
    pub(crate) carriers: Vec<eliminate::Carrier>,
    /// Per cluster the blocks of the excited states its eliminations can mix (assemble.rs `Model::mix`).
    pub(crate) mixing: Vec<Vec<usize>>,
    /// The photons' sources (photon.rs) and per cluster its classes.
    pub(crate) sources: Vec<photon::Source>,
    pub(crate) classes: Vec<Vec<photon::Class>>,
    /// The clusters' assemblies under the light sets asked so far, each made once where it is first asked (assemble.rs
    /// `Model::plan`).
    pub(crate) plans: Mutex<Vec<(usize, Lights, Arc<std::sync::OnceLock<Arc<assemble::Plan>>>)>>,
    /// The spread of what moves the folds' spectra (units.rs `Model::spread`), and the constant forms of two frequencies asked so
    /// far (units.rs `Model::constants`).
    pub(crate) spread: f64,
    pub(crate) crosses: Mutex<std::collections::HashMap<(usize, usize), Arc<units::Cross>>>,
    /// Per component (2 frequency + counter) and drive ((excited − GROUND) GROUND + ground) its level; and the drives themselves,
    /// the couplings that are not zero: (component, excited, ground).
    pub(crate) levels: Vec<Vec<Level>>,
    pub(crate) driven: Vec<(Component, usize, usize)>,
    /// The model's beats, each with its pairs of frequencies (units.rs `Model::beats_made`).
    pub(crate) beats: Vec<(Beat, Vec<(usize, usize)>)>,
    /// Per cluster the classes of its readout's places under the running lights -- per channel and ground place the first
    /// of the places co-resonant with it --, found at its first readout (readout.rs `Model::readout_of`).
    pub(crate) resonances: Vec<std::sync::OnceLock<Vec<usize>>>,
    /// The states' labels and the beams' names (the ledger's and the method's words).
    pub(crate) labels: Vec<String>,
    pub(crate) names: Vec<String>,
    /// Per light set with an averaged light and ground populations asked so far, the model its potential is read on
    /// (potential.rs `Model::smooth`).
    pub(crate) smooth: Mutex<Vec<(Lights, [f64; GROUND], Arc<Model>)>>,
    /// The constants of the fold's series every evaluation takes (elim.rs `Series`): read at the focus where the model
    /// is made, and at the atoms of its run where the run reads them (`Model::referenced`).
    pub(crate) series: elim::Series,
    /// The transitions to the levels outside the model of the light's frequencies, each with its frequency (an index of
    /// the light's; request.rs `Transition`): the errors of their second-order terms, read at the states (elim.rs
    /// `errors`) -- no unit: those levels are not in the model, and nothing of them can be kept.
    pub(crate) outside: Arc<Vec<(usize, crate::request::Transition)>>,
    /// The request itself: what the prior's closure reads at a point (elim.rs `system`, `Model::prior`).
    pub(crate) request: Arc<Request>,
}

/// What a request gives a model whatever units are kept (`Model::new`, `Model::rekept`): the light, the exact frequencies and
/// wave vectors, the states' energies and decay, each frequency's light with nothing folded yet (far.rs `Tone::new`; a fold
/// reads only what no fold changes, so a folded one serves as well), and the drives.
struct Given {
    light: Light,
    axes: far::Axes,
    waves: Vec<[f64; 3]>,
    beam_wave: Vec<(usize, i64)>,
    hbar_over_mass: f64,
    energies: Vec<f64>,
    base: Mat,
    emission: [Mat; 3],
    decay: Mat,
    tones: Vec<far::Tone>,
    driven: Vec<(Component, usize, usize)>,
    labels: Vec<String>,
    names: Vec<String>,
    outside: Arc<Vec<(usize, crate::request::Transition)>>,
    request: Arc<Request>,
}

/// An exact wave vector: integer coordinates on the light's distinct wave vectors (`Model::waves`), with a `Beat`'s integer
/// arithmetic, its value Σ_w n_w k_w. Equal wave vectors have equal coordinates: a Doppler frame exact in exact arithmetic leaves
/// a residual of exactly zero.
pub(crate) type Wave = Beat;

impl Model {
    /// The model of a request with the units `kept` (SPEC §1.4): every drive at the level its kept units give it, the bare
    /// folds' constants, the carriers' eliminations, the clusters the kept couplings join with their frames. Any set of units
    /// makes a model -- a unit kept twice over, or one this request has no coupling for, changes nothing --, but for the model's
    /// domain: a counter-rotating drive kept, or one whose bare fold does not exist, is refused.
    pub fn new(request: &Request, light: &Light, kept: &[Unit]) -> Result<Model, String> {
        let axes = far::Axes::new(request)?;
        let mut waves: Vec<[f64; 3]> = vec![];
        let beam_wave = light
            .beams
            .iter()
            .map(|b| {
                let k = b.wave_vector();
                match [1_i64, -1].into_iter().find_map(|sign| waves.iter().position(|w| one_wave(w, &k, sign as f64)).map(|at| (at, sign))) {
                    Some(found) => found,
                    None => {
                        waves.push(k);
                        (waves.len() - 1, 1)
                    }
                }
            })
            .collect();
        // The drives, and their levels as the kept units give them: one level a drive, the highest of its kept units
        // (`Level`). A unit that names no drive of this request is none of the model's.
        let mut tones = vec![];
        for frequency in 0..light.frequencies.len() {
            tones.push(far::Tone::new(request, light, &axes, frequency)?);
        }
        let driven: Vec<(Component, usize, usize)> = (0..tones.len())
            .flat_map(|frequency| [false, true].map(|counter| Component { frequency, counter }))
            .flat_map(|c| (GROUND..STATES).flat_map(move |e| (0..GROUND).map(move |g| (c, e, g))))
            .filter(|&(c, e, g)| tones[c.frequency].drives(c.counter, e, g))
            .collect();
        let mut base = request.hamiltonian.clone();
        for t in &tones {
            base.add_scaled(-C64::ONE, &t.focus);
        }
        let given = Given {
            light: light.clone(),
            axes,
            waves,
            beam_wave,
            hbar_over_mass: crate::units::hbar_over_mass(request.mass_kg),
            energies: far::energies(request),
            base,
            emission: request.emission.clone(),
            decay: far::decay(request),
            tones,
            driven,
            labels: request.states.clone(),
            names: request.beams.iter().map(|b| b.name.clone()).collect(),
            outside: Arc::new(light.frequencies.iter().enumerate().flat_map(|(w, f)| request.outside_forms[f.form].transitions.iter().map(move |t| (w, t.clone()))).collect()),
            request: Arc::new(request.clone()),
        };
        Model::settled(given, kept)
    }

    /// The model of the same request with the units `kept` in its own's stead (`Model::new`).
    pub(crate) fn rekept(&self, kept: &[Unit]) -> Result<Model, String> {
        let given = Given {
            light: self.light.clone(),
            axes: self.axes.clone(),
            waves: self.waves.clone(),
            beam_wave: self.beam_wave.clone(),
            hbar_over_mass: self.hbar_over_mass,
            energies: self.energies.clone(),
            base: self.base.clone(),
            emission: self.emission.clone(),
            decay: self.decay.clone(),
            tones: self.tones.clone(),
            driven: self.driven.clone(),
            labels: self.labels.clone(),
            names: self.names.clone(),
            outside: self.outside.clone(),
            request: self.request.clone(),
        };
        Model::settled(given, kept)
    }

    /// What a request gives with the units `kept` (`Model::new`).
    fn settled(given: Given, kept: &[Unit]) -> Result<Model, String> {
        // A kept block is its drives (`Unit::Block`).
        let kept: Vec<Unit> = kept.iter().flat_map(|u| match u {
            Unit::Block { block, component, grounds } => block_drives(&given, *block, *component, *grounds),
            _ => vec![u.clone()],
        }).collect();
        let kept = &kept[..];
        let Given { light, axes, waves, beam_wave, hbar_over_mass, energies, base, emission, decay, tones, driven, labels, names, outside, request } = given;
        let mut levels: Vec<Vec<Level>> = (0..2 * tones.len()).map(|_| vec![Level::Bare; far::EXCITED * GROUND]).collect();
        for unit in kept {
            let (component, excited, ground, level) = match *unit {
                Unit::Bare { component, excited, ground } => (component, excited, ground, Level::Atom),
                Unit::Drive { component, excited, ground } => (component, excited, ground, Level::Explicit),
                _ => continue,
            };
            if !driven.contains(&(component, excited, ground)) {
                continue;
            }
            if !component.keepable() {
                return Err("反旋转项的驱动只能按裸能级绝热消去，不能显式跟踪（模型的适用范围）".into());
            }
            let at = &mut levels[2 * component.frequency][(excited - GROUND) * GROUND + ground];
            *at = (*at).max(level);
        }
        let mut model = Model {
            light,
            axes,
            waves,
            beam_wave,
            hbar_over_mass,
            energies,
            base,
            emission,
            decay,
            tones,
            kept: kept.to_vec(),
            clusters: vec![],
            membership: [0; GROUND],
            carriers: vec![],
            mixing: vec![],
            sources: vec![],
            classes: vec![],
            plans: Mutex::new(vec![]),
            spread: 0.0,
            crosses: Mutex::new(Default::default()),
            levels,
            driven,
            beats: vec![],
            resonances: vec![],
            labels,
            names,
            smooth: Mutex::new(vec![]),
            series: elim::Series::none(0),
            outside,
            request,
        };
        model.beats = model.beats_made();
        // The clusters of the kept edges: the explicit drives and the kept couplings of two states (units.rs `Model::edges`),
        // a coupling's strength its entry's with every drive bare.
        let bare: Vec<far::Tone> = if kept.iter().any(|u| matches!(u, Unit::Term { .. })) { model.tones.iter().map(|t| t.folded(&|_, _, _| Level::Bare, &model.energies[GROUND..])).collect() } else { vec![] };
        let edges = model.edges(&bare);
        let (mut clusters, membership) = cluster::clusters(STATES, &edges, &model.axes.zero(), &model.no_wave());
        for c in clusters.iter_mut() {
            c.waves = c.waves_exact.iter().map(|w| model.wave_value(w)).collect();
        }
        model.resonances = clusters.iter().map(|_| std::sync::OnceLock::new()).collect();
        model.clusters = clusters;
        model.membership = membership;
        // A drive between two states of one cluster at a residual beat of exactly zero is in the generator as it is: explicit
        // without a unit kept for it.
        for &(component, e, g) in model.driven.iter().filter(|d| !d.0.counter) {
            let beat = &model.tones[component.frequency].beat;
            let held = model.clusters.iter().any(|k| {
                let place = |s: usize| k.states.iter().position(|&x| x == s);
                matches!((place(e), place(g)), (Some(pe), Some(pg)) if k.frames[pe].plus(-1, &k.frames[pg]) == *beat)
            });
            if held {
                model.levels[2 * component.frequency][(e - GROUND) * GROUND + g] = Level::Explicit;
            }
        }
        // Each frequency's drives folded at their levels (far.rs `Tone::folded`); the carriers, the components with a drive not
        // bare. A drive folded with the bare levels onto an explicit state is read at the state's energy among its cluster's
        // ground states (`read`; far.rs, # A fold onto an explicit state): a direct coupling of two P states, its bare
        // fold's distance d the two-photon one of the pair it makes with the state's kept drives (far.rs `second_order`).
        // That fold is the resolvent's series in the drive's coupling a across d, and a constant taken wherever the light
        // is: it exists where its ratio, Stewart's 2m = 2a/|d| with a every beam at its peak (|a_l| ≤ 1, in phase), is
        // below one (`elim::converges`) -- decided here, as the model is made. Where it is not -- at d = 0 (Zeeman
        // sublevels at B = 0) as at any d not large beside a -- the drive goes up one level, to its fold at the atom
        // (`Unit::Bare`); a counter-rotating drive has no level above (its keep is refused: the model's domain). The bare
        // fold's pole is never read. How well a fold that exists does is read where the atoms are, as every fold's: its next
        // order 2(a/d)² at the state's own amplitude and two-photon distance (elim.rs `Model::bare_mixing`), at the prior's
        // atoms (`Model::prior`) and the run's states (`Model::readings`), on its `Unit::Bare`.
        let levels = model.levels.clone();
        let read = model.read_made();
        let linked: Vec<Unit> = model
            .driven
            .iter()
            .filter(|&&(c, e, w)| model.level(c, e, w) == Level::Bare && model.explicit(e))
            .filter(|&&(c, e, w)| {
                let (tone, side) = (&model.tones[c.frequency], c.counter as usize);
                let a: f64 = tone.couplings[side].iter().map(|x| x[(e, w)].abs()).sum();
                let d = read[e - GROUND] - model.energies[w] - tone.nu[side];
                !elim::converges(2.0 * a / d.abs())
            })
            .map(|&(component, excited, ground)| Unit::Bare { component, excited, ground })
            .collect();
        if !linked.is_empty() {
            return model.rekept(&[kept, &linked[..]].concat());
        }
        model.tones = model.tones.iter().enumerate().map(|(frequency, tone)| tone.folded(&|side, e, g| levels[2 * frequency + side][(e - GROUND) * GROUND + g], &read)).collect();
        model.carriers = eliminate::Carrier::all(&model);
        model.spread = model.spread_made();
        // The model's domain: a counter-rotating drive is only ever folded with the bare levels, so that fold must exist
        // wherever the light is. Of a ground state g against the excited states its counter-rotating drives reach, the
        // invariant subspace exists and the second order converges where m = ‖W‖/sep < ½ (Stewart; ELIMINATION §3), W
        // the couplings and sep the distance: per drive e ← g its coupling at most Σ_l |W_l[e, g]| (every beam at its
        // peak, |a_l| ≤ 1, and in phase) across its bare denominator |E_e − E_g − ν − iΓ_e/2| (ν ≈ −2ω: no light shift
        // moves it), m_e the ratio, and g's drives together S = Σ_e 2 m_e² < 2 (½)² = ½: the ratio 2m_F = √(2S), m_F
        // = (Σ_e m_e²)^{1/2}, below one (`elim::converges`). It is read over the whole light, as the fold is a constant
        // taken everywhere; where it is not below one there is no level above the bare one -- an explicit
        // counter-rotating drive would resolve the optical frequency --, and the request is outside the model's domain.
        for g in 0..GROUND {
            let s: f64 = model
                .driven
                .iter()
                .filter(|d| d.0.counter && d.2 == g)
                .map(|&(c, e, _)| {
                    let tone = &model.tones[c.frequency];
                    let size: f64 = tone.couplings[1].iter().map(|x| x[(e, g)].abs()).sum();
                    let distance = C64::new(model.energies[e] - model.energies[g] - tone.nu[1], 0.5 * model.decay[(e - GROUND, e - GROUND)].re).abs();
                    2.0 * (size / distance).powi(2)
                })
                .sum();
            if !elim::converges((2.0 * s).sqrt()) {
                return Err(format!("反旋转项不能绝热消去：同一个基态的各个反旋转项驱动的饱和参数之和 S = {:.2}（各束光取峰值强度、同相叠加）不小于 0.5，光强超出模型的适用范围", s));
            }
        }
        model.mixing = (0..model.clusters.len()).map(|c| model.mix(c)).collect();
        model.sources = model.all_sources();
        model.classes = (0..model.clusters.len()).map(|c| model.classes(c)).collect();
        model.series = elim::Series::none(model.clusters.len());
        model.series = model.series_at(&[([0.0; 3], [0.0; 3])]);
        Ok(model)
    }

    /// The model with its series constants read at the atoms `points` (position, velocity; elim.rs `Model::series_at`):
    /// what the run evaluates its folds with. Nothing made of the constants before is kept (the plans, the averaged
    /// lights' models).
    pub(crate) fn referenced(&mut self, points: &[([f64; 3], [f64; 3])]) {
        self.series = self.series_at(points);
        self.plans = Mutex::new(vec![]);
        self.smooth = Mutex::new(vec![]);
    }

    /// The energy at which a cluster's ground states read its state at `place`, from the states' levels `levels` (far.rs,
    /// # A fold onto an explicit state): in the cluster's frames its ground states are slow at E_a − θ_a and its explicit
    /// excited states follow them, so the excited amplitude at `place` turns at θ_place + (E_a − θ_a) -- the mean over the
    /// cluster's ground states, which its kept couplings hold near one another (what one of them lies off the mean is in
    /// the pair's two-photon distance as criterion two reads it, `Model::settled`). `value`: the levels are energies (with
    /// the frames); else a
    /// derivative's part of their jet (the frames are constants).
    pub(crate) fn read(&self, cluster: &Cluster, place: usize, levels: &[f64], value: bool) -> f64 {
        let frame = |p: usize| if value { self.axes.omega(&cluster.frames[p]) } else { 0.0 };
        frame(place) + (0..cluster.ground).map(|p| levels[cluster.states[p]] - frame(p)).sum::<f64>() / cluster.ground as f64
    }

    /// Per excited state (less GROUND) the energy a bare fold onto it is read at (`read`, with the bare levels): an
    /// explicit state's by its cluster's ground states; another's its own -- no cluster's state reads it, and what folds
    /// it takes its own energy there (eliminate.rs).
    fn read_made(&self) -> Vec<f64> {
        let mut read = self.energies[GROUND..].to_vec();
        for cluster in self.clusters.iter().filter(|c| c.ground > 0) {
            for place in cluster.ground..cluster.states.len() {
                read[cluster.states[place] - GROUND] = self.read(cluster, place, &self.energies, true);
            }
        }
        read
    }

    /// A drive's level (`Level`).
    pub(crate) fn level(&self, component: Component, excited: usize, ground: usize) -> Level {
        self.levels[2 * component.frequency + component.counter as usize][(excited - GROUND) * GROUND + ground]
    }

    /// Where a jump's bare state f lands in a plan's cluster at a point (elim.rs `Fold::transients`; assemble.rs, # The
    /// normalized generator): the slow part of (f, 0) in the normalized coordinates, S (1 + Σ_f S_f T_f)⁻¹ f, the left
    /// and the right wave operators' overlap Σ_f S_f T_f to the series' leading order Σ_f W′ R₀² W = Y -- not
    /// normalized, its norm N against the next jump's threshold (step.rs `jump`): N < 1, the fast part takes the rest
    /// away at once (a photon more); N > 1, the next photon is late (antibunching). The same map is the light's switching
    /// on (ELIMINATION v3 4). Both factors to the normalization's order J (assemble.rs `normal`): at J = 0 the map is
    /// the identity. Its error is measured against the definition's map with the generator's, and each order's part booked
    /// on the unit that takes it away (elim.rs `Fold::landing_error`, `Model::readings`). Where the cluster has no fold the
    /// state is f.
    pub(crate) fn landed(&self, plan: &assemble::Plan, table: Option<&tabulate::NearTable>, at: &Point, f: &[C64]) -> crate::step::Internal {
        let Some(fourth) = plan.fourth.as_ref().filter(|fourth| fourth.bends()) else { return f.iter().copied().collect() };
        let (n, size) = (fourth.n, fourth.n * fourth.n);
        let mut site = assemble::Held::<assemble::Site>::take();
        let table = self.pointed(plan, table, at.r, &mut site);
        self.quartic(plan, fourth, table, Some(at.r), &mut site, false, true);
        Model::quartic_read(|q| {
            let landing = &q.data[assemble::LANDING * size..(assemble::LANDING + 1) * size];
            (0..n).map(|x| (0..n).fold(C64::ZERO, |z, y| z + landing[x * n + y] * f[y])).collect()
        })
    }

    /// A component's exact frequency ν: the co-rotating δ_ω, the counter-rotating −(δ_ω + 2ν_frame).
    pub(crate) fn frequency(&self, component: Component) -> Beat {
        self.axes.component(&self.light, component)
    }

    /// Every light, coherent: L_run.
    pub fn running(&self) -> Lights {
        Lights(vec![Lit::Coherent; self.light.frequencies.len()])
    }
    /// The trap beam's frequency alone: L_prep.
    pub fn prepared(&self) -> Lights {
        Lights((0..self.light.frequencies.len()).map(|w| if w == self.light.trap { Lit::Coherent } else { Lit::Off }).collect())
    }
    /// The trap light coherent (its own interference kept: a retro-reflected tweezer reports its site) and every other light
    /// averaged (its smooth shift, no fringes): the reported trap's lights (SPEC §1.3).
    pub fn reported(&self) -> Lights {
        Lights((0..self.light.frequencies.len()).map(|w| if w == self.light.trap { Lit::Coherent } else { Lit::Averaged }).collect())
    }
    /// The clusters.
    pub fn clusters(&self) -> &[Cluster] {
        &self.clusters
    }
    /// The cluster a ground state belongs to.
    pub fn cluster_of(&self, ground: usize) -> usize {
        self.membership[ground]
    }
    /// ħ/m (µm²/µs).
    pub fn hbar_over_mass(&self) -> f64 {
        self.hbar_over_mass
    }
    /// A frequency's value (rad/µs).
    pub fn omega(&self, f: &Beat) -> f64 {
        self.axes.omega(f)
    }
    /// An exact wave vector's value (1/µm), summed from its numerators in their order (a wave vector is whole: the
    /// light's own and their integer sums, `Wave`).
    pub(crate) fn wave_value(&self, w: &Wave) -> [f64; 3] {
        let mut out = [0.0; 3];
        for (&n, k) in w.n.iter().zip(&self.waves) {
            if n != 0 {
                for i in 0..3 {
                    out[i] += n as f64 * k[i];
                }
            }
        }
        if w.d != 1 {
            out.iter_mut().for_each(|x| *x /= w.d as f64);
        }
        out
    }
    /// A request beam's wave vector, exactly.
    pub(crate) fn beam_wave(&self, beam: usize) -> Wave {
        let mut w = vec![0; self.waves.len()];
        w[self.beam_wave[beam].0] = self.beam_wave[beam].1;
        Beat::whole(w)
    }
    /// The zero wave vector.
    pub(crate) fn no_wave(&self) -> Wave {
        Beat::whole(vec![0; self.waves.len()])
    }
    /// The loss energy E_s = (ħ/m)κ²/2 + V_run(r; p(ψ)) (rad/µs).
    pub fn energy(&self, cluster: usize, at: &Point, psi: &[C64]) -> f64 {
        self.energy_with(cluster, at, psi, None)
    }
    /// The loss energy with the near light from its tables where they serve (tabulate.rs `Model::near`).
    pub(crate) fn energy_with(&self, cluster: usize, at: &Point, psi: &[C64], tables: Option<&tabulate::Tables>) -> f64 {
        let kinetic = 0.5 * self.hbar_over_mass * at.kappa.iter().map(|k| k * k).sum::<f64>();
        kinetic + self.expand(at.r, &self.ground_populations(cluster, psi), &self.running(), 0, tables).quadratic.value
    }
    /// The loss test (SPEC §1.7): outside the trap's region and E_s above V_run's far value.
    pub fn lost(&self, cluster: usize, at: &Point, psi: &[C64]) -> bool {
        self.lost_with(cluster, at, psi, None)
    }
    /// The loss test with the near light from its tables where they serve.
    pub(crate) fn lost_with(&self, cluster: usize, at: &Point, psi: &[C64], tables: Option<&tabulate::Tables>) -> bool {
        self.light.outside(at.r)
            && self.energy_with(cluster, at, psi, tables) > self.asymptote(&self.ground_populations(cluster, psi), &self.running())
    }
    /// p(ψ), the ground populations of a cluster's state (SPEC §1.3): those it holds, and those its explicit excited states
    /// decay to -- p_b = (|ψ_b|² + Σ_e β_be |ψ_e|²)/‖ψ‖², β_be = Σ_a |C_a[b, e]|²/D_ee the branching of e. An explicit excited state
    /// lives 1/Γ, far shorter than any motion: the potential its atom moves in is that of the ground states it lands in. Linear in
    /// the populations, of sum one, defined for every ψ ≠ 0 -- a state with no ground population has its landing states'.
    pub(crate) fn ground_populations(&self, cluster: usize, psi: &[C64]) -> [f64; GROUND] {
        let total = crate::linalg::norm_sqr(psi);
        let mut p = [0.0; GROUND];
        for (z, &s) in psi.iter().zip(&self.clusters[cluster].states) {
            let weight = z.norm_sqr() / total;
            if s < GROUND {
                p[s] += weight;
            } else if weight != 0.0 {
                let rate = self.decay[(s - GROUND, s - GROUND)].re;
                for (b, landing) in p.iter_mut().enumerate() {
                    *landing += weight * self.emission.iter().map(|c| c[(b, s)].norm_sqr()).sum::<f64>() / rate;
                }
            }
        }
        p
    }
}

/// The drives a kept excited block is (`Unit::Block`): every drive of the component `component` from the ground states
/// `grounds` (bit g for ground state g) into the block's states (`Unit::Drive`). A drive of another component, or from
/// another ground state, is no part of it: it drives the block at its own frequency, where the block is folded.
fn block_drives(given: &Given, block: usize, component: Component, grounds: u32) -> Vec<Unit> {
    let blocks = elim::excited_blocks(&given.labels);
    let Some(states) = blocks.get(block) else { return vec![] };
    given.driven.iter().filter(|d| d.0 == component && states.contains(&(d.1 - GROUND)) && grounds >> d.2 & 1 == 1).map(|&(component, excited, ground)| Unit::Drive { component, excited, ground }).collect()
}

/// Whether two beams' wave vectors are one up to the sign `sign` (k = sign·w): every component within the rounding of the
/// vectors themselves. Each is its request's direction normalized -- three squares, two sums, a root, a quotient -- times its wave
/// number, with the request's own decimal rounding nine roundings a component: two that are one vector differ by at most
/// 2γ₉ max(|k|, |w|) a component, and two that physically differ, by a beat's Δω/c or a direction, by many orders more.
fn one_wave(w: &[f64; 3], k: &[f64; 3], sign: f64) -> bool {
    let size = crate::linalg::norm3(*w).max(crate::linalg::norm3(*k));
    (0..3).all(|i| (k[i] - sign * w[i]).abs() <= 2.0 * crate::units::rounding(9) * size)
}

/// The model as one run's integrator sees it (`step::Dynamics`): the model, its near tables where it has them (tabulate.rs), its
/// prepared lights, the reference state's ground populations, and per cluster its plan under the running lights (assemble.rs)
/// with that plan's table.
pub struct Run<'a> {
    pub model: &'a Model,
    pub tables: Option<&'a tabulate::Tables>,
    pub reference: [f64; GROUND],
    prepared: Lights,
    plans: Vec<(Arc<assemble::Plan>, Option<&'a tabulate::NearTable>)>,
    /// The counted real multiply-adds of one jump's photon on a site of its own (`decide::COMPLEX` per complex one,
    /// assemble.rs # Work; photon.rs `Model::emission_of`: the site once with its matrices and the landing that follows,
    /// every class's rate, the class drawn alone with its gradients: `Model::emission_counted`): the largest over the
    /// clusters at the focus, every light on, made once, a thread's matrices forgotten before it (a jump's point is its
    /// own). A step's middle and its scattering count themselves where they are made (step.rs `Price`).
    photons: f64,
}

impl<'a> Run<'a> {
    /// A cluster's plan and its table's axes and nodes, described (the step's cost harness, step/tests.rs).
    #[cfg(test)]
    pub(crate) fn described(&self, cluster: usize) -> String {
        let (plan, table) = &self.plans[cluster];
        let grid = table.map_or("no table".to_string(), |t| format!("table axes {} nodes {} width {}", t.axes(), crate::table::Table::nodes(t.table.axes()), t.table.width()));
        format!("{} | {grid}", plan.described())
    }
    /// The run of a model with the exact elimination at every point.
    pub fn new(model: &'a Model, reference: [f64; GROUND]) -> Run<'a> {
        Run::with(model, None, reference)
    }
    /// The run of a model with its near tables (`Model::tabulate`).
    pub fn tabulated(model: &'a Model, tables: &'a tabulate::Tables, reference: [f64; GROUND]) -> Run<'a> {
        Run::with(model, Some(tables), reference)
    }
    fn with(model: &'a Model, tables: Option<&'a tabulate::Tables>, reference: [f64; GROUND]) -> Run<'a> {
        let running = model.running();
        let at = Point { r: [0.0; 3], kappa: [0.0; 3], t: 0.0 };
        let plans = (0..model.clusters.len()).map(|c| (model.plan(c, &running), tables.and_then(|t| t.of(c, &running)))).collect();
        let plans: Vec<(Arc<assemble::Plan>, Option<&tabulate::NearTable>)> = plans;
        let first = |c: usize| -> Vec<C64> { (0..model.clusters[c].states.len()).map(|x| if x == 0 { C64::ONE } else { C64::ZERO }).collect() };
        let photons = (0..model.clusters.len())
            .map(|c| {
                Model::forget_quartic();
                model.emission_counted(c, &at, &first(c), tables)
            })
            .fold(0.0, f64::max)
            * crate::decide::COMPLEX;
        Run { model, tables, reference, prepared: model.prepared(), plans, photons }
    }
    /// The photon classes of a cluster's state at a point with their amplitudes alone (photon.rs `Gradients::None`): the
    /// classes a jump draws among, by their rates (`Photon::weight`; a jump makes the rates and the drawn class's photon
    /// alone, photon.rs `Model::emission_of`).
    pub fn photon_amplitudes(&self, cluster: usize, at: &Point, psi: &[C64]) -> Vec<Photon> {
        let (plan, table) = &self.plans[cluster];
        self.model.photons_of(plan, *table, at, psi, photon::Gradients::None).0
    }
    /// The scattering of a cluster's state at a point per unit time (photon.rs `Model::scattering_of`).
    pub fn scattering_at(&self, cluster: usize, at: &Point, psi: &[C64]) -> Scattering {
        let (plan, table) = &self.plans[cluster];
        self.model.scattering_of(plan, *table, at, psi).0
    }
    /// Where a jump's bare state `f` lands in a cluster at a point (assemble.rs, # The normalized generator; `Model::landed`): the
    /// slow state it continues in, with its norm.
    pub fn landed(&self, cluster: usize, at: &Point, f: &[C64]) -> crate::step::Internal {
        let (plan, table) = &self.plans[cluster];
        self.model.landed(plan, *table, at, f)
    }
    /// The displayed rows of a cluster's state at a point (readout.rs `Model::readout_of`).
    pub fn readout(&self, cluster: usize, at: &Point, psi: &[C64]) -> [f64; STATES] {
        let (plan, table) = &self.plans[cluster];
        self.model.readout_of(plan, *table, at, psi)
    }
    /// A cluster's generator at a position and time, for every momentum and along every path through the point (assemble.rs
    /// `Middle`): a step's one middle.
    pub fn middle(&self, cluster: usize, r: [f64; 3], t: f64) -> Middle<'a> {
        let (plan, table) = &self.plans[cluster];
        self.model.middle(plan, r, t, *table)
    }
}

/// What a run of a model counts (`Run::with`) priced from its plans alone, each cluster's near forms read from a table on
/// given axes (`Model::priced`; solve.rs `Pricing`): no table is made and nothing eliminated -- a count is the
/// definition's, whatever the numbers. Per cluster its counts at a node (`Middle::work`); a middle with a step's
/// scattering beside it, and a jump's photons, the largest over the clusters (real multiply-adds).
pub(crate) struct Priced {
    pub(crate) costs: Vec<[f64; 3]>,
    pub(crate) assembly: f64,
    pub(crate) photons: f64,
}

impl Model {
    /// `Priced` with cluster c's near forms from a table on `axes[c].0` axes whose coordinates' values at a point count
    /// `axes[c].1` (tabulate.rs `Model::table_axes`, `Model::priced_axes`), counted as `Run::with` counts at the focus:
    /// the scattering on the middle's site with the normalized generator's matrices made of its laid forms, and a jump's
    /// photon on a site of its own with its own matrices and the landing that follows (photon.rs `Plan::sited_work`); the
    /// matrices change along the lab axes the plan's beams change along there.
    pub(crate) fn priced(&self, axes: &[(usize, f64)]) -> Priced {
        use photon::Gradients;
        let running = self.running();
        let plans: Vec<Arc<assemble::Plan>> = (0..self.clusters.len()).map(|c| self.plan(c, &running)).collect();
        let focus = self.light.amplitudes([0.0; 3]);
        let live = |p: &assemble::Plan| (0..3).filter(|&a| p.beams.iter().any(|&l| focus[l].gradient[a] != C64::ZERO)).count();
        let largest = |count: &dyn Fn(&assemble::Plan, usize, f64) -> f64| plans.iter().zip(axes).map(|(p, &(d, place))| count(p, d, place)).fold(0.0, f64::max) * crate::decide::COMPLEX;
        let scattering = largest(&|p, d, place| p.sited_work(d, place, 1, true) + p.classes_work(d, Gradients::All, live(p)));
        let photons = largest(&|p, d, place| p.sited_work(d, place, 1, false) + p.classes_work(d, Gradients::None, live(p)) + (0..p.classes.len()).map(|class| p.classes_work(d, Gradients::Of(class), live(p))).fold(0.0, f64::max));
        let assembly = largest(&|p, d, place| p.middle_work(d, place)) + scattering;
        Priced { costs: plans.iter().zip(axes).map(|(p, &(d, _))| p.costs(d)).collect(), assembly, photons }
    }
}

impl crate::step::Dynamics for Run<'_> {
    type Middle<'a>
        = Middle<'a>
    where
        Self: 'a;
    fn hbar_over_mass(&self) -> f64 {
        self.model.hbar_over_mass
    }
    fn photon_work(&self) -> f64 {
        self.photons
    }
    fn reference(&self, r: [f64; 3]) -> Quadratic {
        self.model.expansion(r, &self.reference, &self.prepared)
    }
    fn cluster(&self, cluster: usize) -> &Cluster {
        &self.model.clusters[cluster]
    }
    fn cluster_of(&self, ground: usize) -> usize {
        self.model.membership[ground]
    }
    fn middle(&self, cluster: usize, r: [f64; 3], t: f64) -> Middle<'_> {
        Run::middle(self, cluster, r, t)
    }
    fn emit(&self, cluster: usize, at: &Point, psi: &[C64], draw: &mut dyn FnMut(&[f64]) -> Option<usize>) -> Option<crate::step::Emitted> {
        let (plan, table) = &self.plans[cluster];
        self.model.emission_of(plan, *table, at, psi, draw)
    }
    fn landed(&self, cluster: usize, at: &Point, f: &[C64]) -> crate::step::Internal {
        Run::landed(self, cluster, at, f)
    }
    fn energy(&self, cluster: usize, at: &Point, psi: &[C64]) -> f64 {
        self.model.energy_with(cluster, at, psi, self.tables)
    }
    fn lost(&self, cluster: usize, at: &Point, psi: &[C64]) -> bool {
        self.model.lost_with(cluster, at, psi, self.tables)
    }
    fn readout(&self, cluster: usize, at: &Point, psi: &[C64]) -> [f64; STATES] {
        Run::readout(self, cluster, at, psi)
    }
}
