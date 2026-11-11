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
//! No light is far or near: each drive -- one component's coupling of one ground state to one excited state -- is folded with
//! the bare levels, folded with the levels at the atom, or explicit (`Level`), as its own denominator needs (`units`). A drive
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
pub mod assemble;
pub mod readout;
pub mod photon;
pub mod tabulate;
pub(crate) mod jet;

use crate::field::Light;
use crate::linalg::{C64, Mat};
use crate::request::Request;
use std::sync::{Arc, Mutex};
pub use assemble::Middle;
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
}

/// The atoms' reach (SPEC §1.8): the sublevel set {(r, κ) : V_prep(r) + (ħ/m)κ²/2 − V_min ≤ L k_B T_r} of the prepared
/// energy, L the level a thermal ensemble's share s lies above (reach.rs, # The reach's level). Its extrema are provable
/// bounds over its cells, made side by side (reach.rs, # A cell's sides); what it leaves out of the initial ensemble
/// beyond that is counted on the ensemble (reach.rs `Model::left_out`).
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
    /// V_min (rad/µs).
    pub minimum: f64,
    /// A box (µm, [low, high] per lab axis) that provably holds the sublevel set: where the subdivisions start.
    pub cell: [[f64; 2]; 3],
    /// The largest speed |v| (µm/µs) in the reach: √(2 (ħ/m) energy).
    pub speed: f64,
    /// The ground populations p of the prepared potential it is the sublevel set of.
    pub populations: [f64; GROUND],
}

/// A unit (SPEC §5.2): one coupling the solve keeps or folds. The more exact level of every fold is keeping it. No unit names a
/// kind of light: every drive and every coupling of two states is its own unit, judged at its own distance (units.rs).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Unit {
    /// The drive of one component of the light from ground state `ground` into excited state `excited`, folded with the bare
    /// levels: its shift and its jumps are constants (far.rs). Kept: its fold takes the levels at the atom (`Unit::Drive`).
    Bare { component: Component, excited: usize, ground: usize },
    /// The same drive folded with the levels at the atom (eliminate.rs: the dressed response, tabulated). Kept: the excited
    /// state is explicit and the drive enters the generator as it is. Folded onto a state another kept drive makes explicit, it
    /// is one leg of a Raman pair through that state: its value, its gate and its error are that pair's, across the
    /// two-photon distance (far.rs, # A fold onto an explicit state; units.rs `Model::pair_gate`; eliminate.rs
    /// `Model::pair_error`) -- at either level, the bare fold's too.
    Drive { component: Component, excited: usize, ground: usize },
    /// The second order of the fold at the atom of the cluster that holds ground state `ground` (assemble.rs, # The fourth
    /// order): its drives folded at the atom leave out the fourth-order light shift and the jumps' next order. Kept: that
    /// cluster's fold is the fourth order's -- the fold's next level, where it has one (`Model::fold_frequency`).
    Second { ground: usize },
    /// Every second-order coupling `a` ← `b` at one exact beat (∝ e^{i beat t}): of two ground states through the excited states
    /// -- one component's own (beat zero: the light's static couplings, the request's outside forms with them) or two
    /// components' (a Raman pair's; where a = b a level's beating shift) --, or of two excited states through the ground states.
    /// Kept: it is in the generator, and its two states are one cluster's, their frames the beat apart.
    Term { a: usize, b: usize, beat: Beat },
    /// A carrier's dressing of the other channels (its light shifts of their ground states and its folded channels'
    /// self-energy) with its polarization units' T̄, their interference averaged (eliminate.rs). Kept: its coherent field at the
    /// point.
    Dressing { carrier: Component },
    /// The coherence of the photons landing in ground state `landing` at two frame frequencies `first` and `second` (in the
    /// landing cluster's frames). Kept: one class, their interference with its beat; folded: two classes.
    Coherence { landing: usize, first: Beat, second: Beat },
}

/// How a drive is taken (SPEC §1.4): folded with the bare levels, folded with the levels at the atom, or explicit. A drive has
/// exactly one, the highest its kept units give it (`Model::new`): each is the more exact level of the one before.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Level {
    Bare,
    Atom,
    Explicit,
}

/// A unit's existence gate over the reach (SPEC §1.5): provable upper bounds over the reach in phase space. The fold exists only
/// where both are below ½ (`Gate::exists`); else the unit is kept. Every fold has both, by one statement (units.rs `gate`); a
/// dressing (`Unit::Dressing`), which folds nothing, has neither.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gate {
    /// The static Stewart bound: the coherent peak's upper bound of S = 2m² (a drive's: of its ground state's folded drives
    /// that are not larger than it, units.rs # The tail rule).
    pub saturation: f64,
    /// The motional bound: the upper bound of a = |dΔE/dt|/|ΔE|², the sweep v·∇ΔE + q·v̇ of the fold's levels and wave vector.
    pub sweep: f64,
}

/// A folded unit's errors on the outputs at one trajectory state (SPEC §1.5 → §5.2): its small parameter measured there and mapped
/// by the leading-order relations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnitError {
    /// The small parameter measured: s (a drive's share of its ground state), ε (a bare fold's denominator's first order),
    /// m² (a coupling's mixing), a carrier's dressing's relative rate error.
    pub parameter: f64,
    /// The absolute error it makes on the display rows' populations: s p_g; ε times its part of the state's rates;
    /// m²(p_a + p_b); m²√(p_i p_j).
    pub population: f64,
    /// The relative error it makes on T: s (the fourth order's 3/2 s²); a; (w/σ)².
    pub temperature: f64,
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
    /// The explicit states' part of the second moment's rate, in the amplitudes' norm (1/µm² times ‖u‖²'s unit; eliminate.rs
    /// `Model::explicit_moment`): their absorption's mean is the Ehrenfest force's. Zero where no gradients were asked.
    pub explicit: [[f64; 3]; 3],
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
    /// Σ_a Re⟨∇u_a|⊗|∇u_a⟩ and the explicit states' part (1/µm²): the second moment's rate.
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
    /// The carriers: the co-rotating components with a drive not bare (eliminate.rs); per excited state (less GROUND)
    /// its kept drives (eliminate.rs `Model::legs_made`).
    pub(crate) carriers: Vec<eliminate::Carrier>,
    pub(crate) legs: Vec<Vec<(usize, usize)>>,
    /// Per component, excited state and ground state the legs of the Raman pairs that component's drive makes folded
    /// onto the excited state (eliminate.rs `Model::pairs_made`, read by `Model::pair_legs`).
    pub(crate) pairs: Vec<Vec<(usize, usize)>>,
    /// Per cluster the blocks of the excited states its eliminations can mix (assemble.rs `Model::mix`).
    pub(crate) mixing: Vec<Vec<usize>>,
    /// Per cluster the frequency whose drives folded at the atom are taken to the fourth order (`Unit::Second` kept;
    /// assemble.rs, # The fourth order), none where its fold is the second order's.
    pub(crate) fourth: Vec<Option<usize>>,
    /// The photons' sources (photon.rs) and per cluster its classes.
    pub(crate) sources: Vec<photon::Source>,
    pub(crate) classes: Vec<Vec<photon::Class>>,
    /// The clusters' assemblies under the light sets asked so far (assemble.rs `Model::plan`).
    pub(crate) plans: Mutex<Vec<Arc<assemble::Plan>>>,
    /// The spread of what moves the folds' spectra (units.rs `Model::spread`), and the constant forms of two frequencies asked so
    /// far (units.rs `Model::constants`).
    pub(crate) spread: f64,
    pub(crate) crosses: Mutex<std::collections::HashMap<(usize, usize), Arc<units::Cross>>>,
    /// Per component (2 frequency + counter) and drive ((excited − GROUND) GROUND + ground) its level; and the drives themselves,
    /// the couplings that are not zero: (component, excited, ground).
    pub(crate) levels: Vec<Vec<Level>>,
    pub(crate) driven: Vec<(Component, usize, usize)>,
    /// Per ground state its drives in the tail rule's order (units.rs `Model::ranked_made`); the model's beats, each with its
    /// pairs of frequencies (units.rs `Model::beats_made`); and what the regions' bounds read of the model's forms (reach.rs).
    pub(crate) ranked: Vec<Vec<units::Ranked>>,
    pub(crate) beats: Vec<(Beat, Vec<(usize, usize)>)>,
    pub(crate) forms: Arc<reach::Forms>,
    /// Per cluster the co-resonant pairs of its folded channels' places under the running lights, found at its first readout
    /// (readout.rs `Model::readout_of`).
    pub(crate) resonances: Vec<std::sync::OnceLock<Vec<[usize; 4]>>>,
    /// The states' labels and the beams' names (the ledger's and the method's words).
    pub(crate) labels: Vec<String>,
    pub(crate) names: Vec<String>,
    /// Per light set with an averaged light and ground populations asked so far, the model its potential is read on
    /// (potential.rs `Model::smooth`).
    pub(crate) smooth: Mutex<Vec<(Lights, [f64; GROUND], Arc<Model>)>>,
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
        // (units.rs, # Levels). A unit that names no drive of this request is none of the model's.
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
        };
        Model::settled(given, kept)
    }

    /// What a request gives with the units `kept` (`Model::new`).
    fn settled(given: Given, kept: &[Unit]) -> Result<Model, String> {
        let Given { light, axes, waves, beam_wave, hbar_over_mass, energies, base, emission, decay, tones, driven, labels, names } = given;
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
            if component.counter {
                return Err("反旋分量的驱动只按裸能级折叠（模型的适用范围），不能保留".into());
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
            legs: vec![],
            pairs: vec![],
            mixing: vec![],
            fourth: vec![],
            sources: vec![],
            classes: vec![],
            plans: Mutex::new(vec![]),
            spread: 0.0,
            crosses: Mutex::new(Default::default()),
            levels,
            driven,
            ranked: vec![],
            beats: vec![],
            forms: Arc::new(reach::Forms::new()),
            resonances: vec![],
            labels,
            names,
            smooth: Mutex::new(vec![]),
        };
        model.ranked = model.ranked_made();
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
        // bare.
        let levels = model.levels.clone();
        let read = model.read_made();
        model.tones = model.tones.iter().enumerate().map(|(frequency, tone)| tone.folded(&|side, e, g| levels[2 * frequency + side][(e - GROUND) * GROUND + g], &read)).collect();
        model.carriers = eliminate::Carrier::all(&model);
        model.legs = model.legs_made();
        model.pairs = model.pairs_made();
        model.spread = model.spread_made();
        // The model's domain: a counter-rotating drive is only ever folded with the bare levels, so that fold must exist --
        // the one gate (units.rs `gate`), every beam at its peak, a ground state's drives together.
        for g in 0..GROUND {
            let saturation: f64 = model.driven.iter().filter(|d| d.0.counter && d.2 == g).map(|&(c, e, _)| model.share_anywhere(c, e, g)).sum();
            if !(saturation < units::EXISTS) {
                return Err(format!("反旋分量的折叠不存在（S = {:.2}）：请求超出模型的适用范围", saturation));
            }
        }
        model.fourth = (0..model.clusters.len())
            .map(|c| model.fold_frequency(c).filter(|_| kept.iter().any(|u| matches!(u, Unit::Second { ground } if *ground < GROUND && model.membership[*ground] == c))))
            .collect();
        model.mixing = (0..model.clusters.len()).map(|c| model.mix(c)).collect();
        model.sources = model.all_sources();
        model.classes = (0..model.clusters.len()).map(|c| model.classes(c)).collect();
        Ok(model)
    }

    /// The energy at which a cluster's ground states read its state at `place`, from the states' levels `levels` (far.rs,
    /// # A fold onto an explicit state): in the cluster's frames its ground states are slow at E_a − θ_a and its explicit
    /// excited states follow them, so the excited amplitude at `place` turns at θ_place + (E_a − θ_a) -- the mean over the
    /// cluster's ground states, which its kept couplings hold near one another (what one of them lies off the mean is in
    /// the pair's error, eliminate.rs `Model::pair_error`). `value`: the levels are energies (with the frames); else a
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

    /// The frequency whose fold at the atom on a cluster has a fourth order (assemble.rs, # The fourth order): the cluster
    /// holds ground states alone, and the drives folded at the atom from them are one frequency's co-rotating ones. None
    /// where there is none, or the expansion is not this one's: beside explicit states the fold's next order has their
    /// couplings in it, and two frequencies' have their beats.
    pub(crate) fn fold_frequency(&self, cluster: usize) -> Option<usize> {
        let c = &self.clusters[cluster];
        if c.states.len() != c.ground {
            return None;
        }
        let mut of = self.driven.iter().filter(|&&(component, e, g)| c.states.contains(&g) && self.level(component, e, g) == Level::Atom).map(|d| d.0.frequency);
        let first = of.next()?;
        of.all(|frequency| frequency == first).then_some(first)
    }

    /// Where a jump's bare ground state f lands in a plan's cluster at a point (assemble.rs, # The fourth order): the
    /// slow state Π f, Π = 1 + ½X − Y in the fold's coordinates, with its norm. A bare state is the slow state and a
    /// fast part that is gone within 1/Γ, and the norm of what it becomes is ‖e^{−iHt} f‖² → ‖Π f‖² e^{−γt} beyond that
    /// time, γ the slow state's rate: ‖Π f‖²/‖f‖² = 1 − p is the bare state's survival extrapolated back to the jump.
    /// On one line p = 2 Re Y − X = (s/2)(Δ² − 3Γ²/4)/(Δ² + Γ²/4) + O(s²), of either sign. Far from the line p > 0, s/2:
    /// the fast part emits one more photon at once (the ringing). Within 0.866 Γ of it p < 0, −3s/2 on resonance: after
    /// a photon the atom is in the ground state and its excited amplitude takes 1/Γ to build up -- the next photon is
    /// late (antibunching). Both are the one statement that the trajectory goes on in Π f as it is, not normalized,
    /// against the next jump's threshold (step.rs `jump`): the waiting time's law is P(T > t) = min(1, (1 − p) e^{−γt})
    /// -- a norm below one is at the threshold already with the probability p, a norm above one reaches it ln(1 − p)/γ
    /// later. The mean interval is (1 − p)/γ for p ≥ 0 and (1 + ln(1 − p))/γ = (1 − p − p²/2 + …)/γ for p < 0, the exact
    /// one's 1/γ_exact to the fourth order either way (two levels: units.rs `ORDER_RATES`). Clamped at zero, p < 0 left
    /// the rate too high by −p, 1.52 s on resonance, more than the second order's s. Where the cluster's fold is the
    /// second order's the slow state is f: there the fast part is within the fold's own order.
    pub(crate) fn landed(&self, plan: &assemble::Plan, table: Option<&tabulate::NearTable>, at: &Point, f: &[C64]) -> crate::step::Internal {
        let Some(fourth) = &plan.fourth else { return f.iter().copied().collect() };
        let (n, size) = (fourth.n, fourth.n * fourth.n);
        let mut site = assemble::Held::<assemble::Site>::take();
        self.light.amplitudes_into(at.r, 1, |_| true, &mut site.beams);
        self.quartic(plan, fourth, table, Some(at.r), &mut site, false);
        Model::quartic_read(|q| {
            let (partner, metric) = (&q.data[4 * size..5 * size], &q.data[8 * size..9 * size]);
            (0..n).map(|x| (0..n).fold(f[x], |z, y| z + (metric[x * n + y] * 0.5 - partner[x * n + y]) * f[y])).collect()
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
    /// An exact wave vector's value (1/µm), summed from its coordinates in their order.
    pub(crate) fn wave_value(&self, w: &Wave) -> [f64; 3] {
        let mut out = [0.0; 3];
        for (&n, k) in w.0.iter().zip(&self.waves) {
            if n != 0 {
                for i in 0..3 {
                    out[i] += n as f64 * k[i];
                }
            }
        }
        out
    }
    /// A request beam's wave vector, exactly.
    pub(crate) fn beam_wave(&self, beam: usize) -> Wave {
        let mut w = vec![0; self.waves.len()];
        w[self.beam_wave[beam].0] = self.beam_wave[beam].1;
        Beat(w)
    }
    /// The zero wave vector.
    pub(crate) fn no_wave(&self) -> Wave {
        Beat(vec![0; self.waves.len()])
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
    /// The counted real multiply-adds of one middle (`decide::COMPLEX` per complex one, assemble.rs # Work) with its state's
    /// scattering (photon.rs `Model::scattering_of`: every class's photons with their gradients, once a step), and of one
    /// jump's photons (photon.rs `Model::photons_counted`): the largest over the clusters at the focus, every light on, made
    /// once.
    assembly: f64,
    photons: f64,
}

impl<'a> Run<'a> {
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
        let photons = (0..model.clusters.len())
            .map(|c| {
                let psi: Vec<C64> = (0..model.clusters[c].states.len()).map(|x| if x == 0 { C64::ONE } else { C64::ZERO }).collect();
                model.photons_counted(c, &at, &psi, tables).1
            })
            .fold(0.0, f64::max)
            * crate::decide::COMPLEX;
        let assembly = plans.iter().map(|(plan, table)| model.middle(plan, at.r, at.t, *table).work).fold(0.0, f64::max) * crate::decide::COMPLEX + photons;
        Run { model, tables, reference, prepared: model.prepared(), plans, assembly, photons }
    }
    /// The photon classes of a cluster's state at a point with their amplitudes alone (photon.rs `Gradients::None`): the
    /// classes a jump draws among, by their rates (`Photon::weight`; a jump makes the rates and the drawn class's photon
    /// alone, photon.rs `Model::emission_of`).
    pub fn photon_amplitudes(&self, cluster: usize, at: &Point, psi: &[C64]) -> Vec<Photon> {
        let (plan, table) = &self.plans[cluster];
        self.model.photons_of(plan, *table, at, psi, photon::Gradients::None).0
    }
    /// The scattering of a cluster's state at a point per unit time, an explicit state's decay at the populations
    /// `populations` of unit sum (None: the state's own; photon.rs `Model::scattering_of`).
    pub fn scattering_at(&self, cluster: usize, at: &Point, psi: &[C64], populations: Option<&[f64]>) -> Scattering {
        let (plan, table) = &self.plans[cluster];
        self.model.scattering_of(plan, *table, at, psi, populations).0
    }
    /// Where a jump's bare state `f` lands in a cluster at a point (assemble.rs, # The fourth order; `Model::landed`): the
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

impl crate::step::Dynamics for Run<'_> {
    type Middle<'a>
        = Middle<'a>
    where
        Self: 'a;
    fn hbar_over_mass(&self) -> f64 {
        self.model.hbar_over_mass
    }
    fn assembly_work(&self) -> f64 {
        self.assembly
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
    fn scattering(&self, cluster: usize, at: &Point, psi: &[C64], populations: &[f64]) -> Scattering {
        self.scattering_at(cluster, at, psi, Some(populations))
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
