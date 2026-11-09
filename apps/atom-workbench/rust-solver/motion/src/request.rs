//! The request (SPEC §9; contract §1.4): a single Rb87 atom in an optical tweezer, its internal state and its motion in
//! 3D -- the scene's rb87-full/2 model with its beams kept apart, their geometry on the lab axes, each beam's field at the
//! focus, the outside forms of every frequency, and the spontaneous emission resolved on the lab axes, as the
//! TypeScript model writes it (src/full-master-model.ts `MotionModel`, the only owner of atomic physics). The native
//! adapter parses the JSON into these types (every field of the motion block required, `electricField` and
//! `outsideForms` included: no request from before them is served); `Request::validate` holds the checks that need
//! the request alone (contract §2.4).
//!
//! At the position r (µm from the focus, the origin) the Hamiltonian in the model's frames is (rad/µs)
//!
//! ```text
//! H(r, t) = H0 + Σ_ω Σ_i (x_{ω,i}(r) − x_{ω,i}(0)) O_{ω,i}
//!              + Σ_l [ a_l(r) X_l e^{−iΔ_l t} + a_l(r)* R_l e^{+i(Δ_l + 2ν_frame) t} + h.c. ]
//! ```
//!
//! with a_l(r) the beam's amplitude relative to its focus value (field.rs), X_l and R_l its co- and counter-rotating
//! couplings at the focus, linear in its field E_l and E_l*, x_{ω,i} the coherence coordinates of the frequency's summed
//! field (field.rs) and O_{ω,i} its outside form: the shift through the states outside the model and the Rb⁺ core, of
//! which H0 already holds the focus value. Δ_l = 2π Σ_k frequency_index_l[k] frequencies_mhz[k] is the beam's
//! detuning from the excited states' frame. Spontaneous emission is Σ_a D[C_a] over the lowering dipole's lab-axis
//! components (`emission`).

use crate::linalg::{C64, Mat};

/// The internal states: the 32 Rb87 states of the model basis.
pub const STATES: usize = 32;
/// The ground manifold's states (5S₁/₂, F = 1 and 2).
pub const GROUND: usize = 8;

/// A state's manifold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Manifold {
    Ground,
    D1,
    D2,
}

/// A state's label `manifold-F-m` read (src/physics.ts: `${manifold}-${F}-${m}`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Level {
    pub manifold: Manifold,
    pub f: u8,
    pub m: i8,
}

impl Level {
    /// The label read, or why it is not `manifold-F-m` with a manifold ground, D1 or D2.
    pub fn parse(label: &str) -> Result<Level, String> {
        let mut parts = label.splitn(3, '-');
        let manifold = match parts.next() {
            Some("ground") => Some(Manifold::Ground),
            Some("D1") => Some(Manifold::D1),
            Some("D2") => Some(Manifold::D2),
            _ => None,
        };
        match (manifold, parts.next().and_then(|f| f.parse().ok()), parts.next().and_then(|m| m.parse().ok())) {
            (Some(manifold), Some(f), Some(m)) => Ok(Level { manifold, f, m }),
            _ => Err(format!("The state label {label} is not manifold-F-m")),
        }
    }
    pub fn is_ground(&self) -> bool {
        self.manifold == Manifold::Ground
    }
}

/// One laser beam. Beams are kept apart: two counter-propagating beams of one frequency are two beams, each with its
/// own spatial phase; the beams of one frequency are one coherent field (SPEC §1.2).
#[derive(Clone, Debug, PartialEq)]
pub struct Beam {
    /// The laser's name as the scene shows it (messages).
    pub name: String,
    /// The direction of propagation n on the lab axes x, y, z (the order of `Request::emission`); its length is
    /// read as a direction only (field.rs normalizes it).
    pub direction: [f64; 3],
    /// |k| = 2πν/c (rad/µm).
    pub wave_number: f64,
    /// A Gaussian beam focused at the origin: its waist, the 1/e² intensity radius at the focus (µm). None: a plane
    /// wave.
    pub waist: Option<f64>,
    /// The beam's frequency on the request's frequency axes (`Request::frequencies_mhz`): exact integer coordinates,
    /// so that equal frequencies are equal and no beat is rounded.
    pub frequency_index: Vec<i32>,
    /// X, the co-rotating excited ← ground coupling at the focus (rad/µs, 32 × 32, rows excited, columns ground).
    pub coupling: Mat,
    /// R, the counter-rotating coupling at the focus (rad/µs, 32 × 32, excited ← ground).
    pub counter_coupling: Mat,
    /// E_l, the beam's field at the focus (V/m) on the spherical components q = −1, 0, +1 of the quantization axis,
    /// its polarization and initial phase in it: X is linear in E_l, R in E_l* (SPEC §1.2).
    pub field: [C64; 3],
}

/// The outside shift of the beams of one frequency as a form in their summed field E (rad/µs/(V/m)²; src/spectroscopy.ts
/// `outsideFieldForm`): O(E) = Σ_i x_i(E) O_i with x = (|E_q|² for q = −1, 0, +1, then per pair p < q in the order
/// (−1, 0), (−1, +1), (0, +1) Re(E_p* E_q), Im(E_p* E_q)) -- field.rs `Coherence`. Each O_i is Hermitian.
#[derive(Clone, Debug, PartialEq)]
pub struct OutsideForm {
    pub frequency_index: Vec<i32>,
    pub matrices: [Mat; 9],
}

/// A motion solve: the model, the initial state and the output times.
#[derive(Clone, Debug, PartialEq)]
pub struct Request {
    /// The model basis: 32 labels `manifold-F-m`, the eigenstates of H0.
    pub states: Vec<String>,
    /// H0 in the frames (rad/µs, 32 × 32): the magnetic field and the beams' shifts through the states outside the
    /// model at the focus, without the couplings; diagonal in the model basis.
    pub hamiltonian: Mat,
    /// The frequency each state's frame rotates at (MHz): 0 for the ground states, ν_frame for the excited ones.
    pub frame_frequencies_mhz: Vec<f64>,
    /// The frequency axes (MHz) the beams' `frequency_index` counts on.
    pub frequencies_mhz: Vec<f64>,
    pub beams: Vec<Beam>,
    /// One outside form per distinct frequency of the beams.
    pub outside_forms: Vec<OutsideForm>,
    /// The trap beam (an index into `beams`): the tweezer, a Gaussian beam whose focus is the origin.
    pub trap_beam: usize,
    /// The lowering dipole's components on the lab axes x, y, z (1/√µs, 32 × 32, ground ← excited): Σ_a D[C_a] is
    /// the model's spontaneous emission.
    pub emission: [Mat; 3],
    pub mass_kg: f64,
    /// The initial motion's temperature (µK).
    pub temperature_uk: f64,
    /// The initial internal density (32 × 32), diagonal in the model basis.
    pub initial_density: Mat,
    /// The output times (µs): from 0, evenly spaced.
    pub times_us: Vec<f64>,
}

/// Why a tweezer without a waist is refused (contract §2.4; shown in the app).
pub const NO_WAIST: &str = "光镊（trapBeamId 所指的光束）需要束腰：3D 运动计算聚焦在原子处的光镊中的单个原子";

impl Request {
    /// The states' levels, or why a label is not one.
    pub fn levels(&self) -> Result<Vec<Level>, String> {
        self.states.iter().map(|s| Level::parse(s)).collect()
    }

    /// The checks that need the request alone (contract §2.4), in the order a request is read: the states, H0, the
    /// frames, the beams and their forms, the tweezer, the emission, the atom, the initial state, the times. The
    /// messages the app shows are Chinese as today; the others name the field.
    pub fn validate(&self) -> Result<(), String> {
        let _ = self;
        todo!("phase 2")
    }
}
