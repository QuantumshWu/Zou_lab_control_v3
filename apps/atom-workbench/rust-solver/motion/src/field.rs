//! The light (SPEC §1.2, §1.3): each beam's amplitude a_l(r) relative to its focus value with its analytic gradient
//! and Hessian; one coherent field per frequency, E_ω(r) = Σ_{l∈ω} a_l(r) E_l, with its derivatives; the coherence
//! coordinates x_{ω,i} of T_ω = E_ω E_ω† with their gradients and Hessians; the bounded chart of each frequency's field
//! with its gradient; and the trap's region by intensity. The potential, the trap and the reach are the model's.
//!
//! # Beams
//! A plane wave a = e^{ikζ}; a paraxial Gaussian beam focused at the origin a = e^{ikζ} exp(−ρ²/(w₀²q))/q,
//! q = 1 + iζ/z_R, z_R = k w₀²/2, ζ = n·r, ρ² = |r|² − ζ²: the Gouy phase −atan(ζ/z_R) and the wavefront's curvature
//! are in q. Its logarithmic gradient g = ∇ln a = ik n − 2t/(w₀²q) + (i/z_R)(ρ²/(w₀²q²) − 1/q) n (t = r − ζn, the
//! transverse part) stays finite where a underflows; ∇a = a g, ∇∇a = a (g gᵀ + ∇g) with
//! ∇g = −2(1 − nnᵀ)/(w₀²q) + 2i(ntᵀ + tnᵀ)/(z_R w₀² q²) − (1/q² − 2ρ²/(w₀²q³)) nnᵀ/z_R² (symmetric).
//! |a| ≤ 1/|q| ≤ 1 everywhere: a beam is brightest at its focus. Paraxial: no longitudinal field, the polarization the
//! same across the beam (SPEC §8).
//!
//! # One field per frequency
//! The beams of one frequency are one coherent field, their relative phases the request's (SPEC §1.2, §8); the
//! couplings are linear in it: V_ω(r) = Σ_l a_l(r) X_l = Σ_q E_{ω,q}(r) D_q. The span of the beams' focus fields has
//! an orthonormal basis u_k (k < r ≤ 3, the eigenvectors of the Gram matrix G = Σ_l E_l E_l† whose eigenvalues σ_k²
//! lie above G's rounding), E_l = Σ_k γ_lk u_k with γ_lk = u_k†E_l, so E_ω(r) = Σ_k e_k(r) u_k, e_k = Σ_l a_l γ_lk.
//! Since Σ_l γ_lk* γ_lk' = σ_k² δ_kk', a quantity Y_l linear in each E_l, Y_l = Σ_k γ_lk Y_k (the couplings X_l), has
//! the basis components Y_k = Σ_l γ_lk* Y_l/σ_k² (`Frequency::dual`), and one antilinear in E_l (R_l = Σ_k γ_lk* Ỹ_k)
//! Ỹ_k = Σ_l γ_lk R_l/σ_k²: the model's check that the couplings agree with the fields is the residual of these.
//!
//! # Coherence coordinates
//! T = E E† (T_pq = E_p E_q*) is Hermitian 3 × 3: nine real numbers x = (|E_q|² for q = −1, 0, +1; then per pair
//! p < q in the order (−1, 0), (−1, +1), (0, +1) Re(E_p* E_q), Im(E_p* E_q)) in (V/m)², the outside forms' order. The far
//! light's static shift is linear in them, exact at the atom (SPEC §1.2): their gradients and Hessians are exact
//! quadratic forms in E, ∇E, ∇∇E.
//!
//! # The chart
//! Each frequency's field in bounded coordinates (SPEC §1.2; field.md §1.1): the intensity I = |E|²/N ∈ [0, 1],
//! N = (Σ_l ‖E_l‖)² (|E| ≤ Σ_l |a_l|‖E_l‖ and |a_l| ≤ 1); and the polarization w = E/|E| on the basis, its components in
//! the chart's order o (the pivot o₀ first, the basis vector of the largest focus field, then the others by their focus
//! fields) as hyperspherical angles α_i = atan2(√Σ_{j≥i}|e_{o_j}|², |e_{o_{i−1}}|) ∈ [0, π/2] and relative phases
//! φ_i = arg(e_{o_i} e_{o₀}*) ∈ (−π, π], i = 1 … r − 1: 2r − 1 coordinates in the order (I, α₁, φ₁, α₂, φ₂). Every
//! coordinate value is some field's, so a table on the whole ranges meets no atom at an edge; the phases are periodic.
//! The field is a trigonometric polynomial in the coordinates (`Frequency::chart_field`): w_{o₀} = cos α₁,
//! w_{o₁} = sin α₁ cos α₂ e^{iφ₁}, w_{o₂} = sin α₁ sin α₂ e^{iφ₂} (the last without its cosine), e = √(IN) w -- smooth
//! through the chart's poles, where a component vanishes and its phase is any. At a pole the chart's gradient is not
//! defined; its vanishing magnitude's terms are taken as zero there (a tabulated function's derivative along the
//! undefined coordinate vanishes with the magnitude, so the force stays bounded).
//!
//! # The region
//! An atom is outside the trap's region where every focused beam's intensity is below `REGION_INTENSITY` of its focus
//! value, |a_l(r)|² < 1/20 (SPEC §1.3, the UI's definition of loss).

use crate::linalg::{C64, V3};
use crate::request::Request;

/// The coherence coordinates per frequency.
pub const COHERENCE: usize = 9;

/// A beam's geometry: its unit direction n (lab axes), its wave number k (rad/µm), its waist w₀ (µm; None: a plane
/// wave).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Geometry {
    pub direction: V3,
    pub wave_number: f64,
    pub waist: Option<f64>,
}

/// A beam's amplitude at a point, relative to its focus value, and its analytic gradient (1/µm) and Hessian (1/µm²).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Amplitude {
    pub value: C64,
    pub gradient: [C64; 3],
    pub hessian: [[C64; 3]; 3],
}

impl Geometry {
    /// The request beam's geometry, its direction normalized.
    pub fn new(direction: V3, wave_number: f64, waist: Option<f64>) -> Geometry {
        let l = crate::linalg::norm3(direction);
        Geometry { direction: direction.map(|x| x / l), wave_number, waist }
    }
    /// k n (rad/µm).
    pub fn wave_vector(&self) -> V3 {
        self.direction.map(|x| x * self.wave_number)
    }
    /// The Rayleigh range z_R = k w₀²/2 of a Gaussian beam (µm).
    pub fn rayleigh(&self) -> Option<f64> {
        self.waist.map(|w| self.wave_number * w * w / 2.0)
    }
    /// a(r), ∇a and ∇∇a (module documentation, # Beams).
    pub fn amplitude(&self, r: V3) -> Amplitude {
        let _ = r;
        todo!("phase 2")
    }
}

/// One frequency's coherent field at a point: E (V/m, spherical components q = −1, 0, +1), ∇E (`gradient[a][q]` =
/// ∂_a E_q) and ∇∇E (`hessian[a][b][q]` = ∂_a ∂_b E_q).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FieldAt {
    pub value: [C64; 3],
    pub gradient: [[C64; 3]; 3],
    pub hessian: [[[C64; 3]; 3]; 3],
}

/// The coherence coordinates x of a field (module documentation, # Coherence coordinates; (V/m)²), ∇x
/// (`gradient[i][a]` = ∂_a x_i) and ∇∇x (`hessian[i][a][b]`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Coherence {
    pub value: [f64; COHERENCE],
    pub gradient: [[f64; 3]; COHERENCE],
    pub hessian: [[[f64; 3]; 3]; COHERENCE],
}

/// The coherence coordinates of a field value alone.
pub fn coordinates(e: &[C64; 3]) -> [f64; COHERENCE] {
    let _ = e;
    todo!("phase 2")
}

impl FieldAt {
    /// Its coherence coordinates with their gradients and Hessians.
    pub fn coherence(&self) -> Coherence {
        todo!("phase 2")
    }
}

/// A frequency's chart at a point (module documentation, # The chart): the 2r − 1 coordinates (I, α₁, φ₁, α₂, φ₂)
/// and their gradients (`gradient[c][a]` = ∂_a s_c).
#[derive(Clone, Debug, PartialEq)]
pub struct Chart {
    pub value: Vec<f64>,
    pub gradient: Vec<[f64; 3]>,
}

/// The range of the chart's coordinate c and whether it is periodic: the intensity [0, 1], an angle [0, π/2], a
/// phase [−π, π) (periodic).
pub fn range(c: usize) -> (f64, f64, bool) {
    use std::f64::consts::{FRAC_PI_2, PI};
    if c == 0 {
        (0.0, 1.0, false)
    } else if c % 2 == 1 {
        (0.0, FRAC_PI_2, false)
    } else {
        (-PI, PI, true)
    }
}

/// One frequency's field: its beams, their focus fields, the basis of their span and the chart's data.
#[derive(Clone, Debug, PartialEq)]
pub struct Frequency {
    /// The frequency's coordinates on the request's frequency axes.
    pub index: Vec<i32>,
    /// Its beams (indices into the request's beams).
    pub beams: Vec<usize>,
    /// Its beams' focus fields E_l (V/m), in `beams` order.
    pub fields: Vec<[C64; 3]>,
    /// Its outside form (an index into the request's outside forms).
    pub form: usize,
    /// N = (Σ_l ‖E_l‖)² ((V/m)²): the chart's intensity scale.
    pub norm: f64,
    /// The orthonormal basis u_k of the span of the focus fields (spherical components), k < r ≤ 3.
    pub basis: Vec<[C64; 3]>,
    /// σ_k², the Gram matrix's eigenvalue of u_k ((V/m)²).
    pub gram: Vec<f64>,
    /// γ_lk = u_k†E_l: per beam (in `beams` order) its components on the basis.
    pub coefficients: Vec<Vec<C64>>,
    /// The chart's component order o: the pivot (the largest focus component Σ_l γ_lk) first.
    pub order: Vec<usize>,
}

impl Frequency {
    /// r, the dimension of the span of its beams' fields.
    pub fn rank(&self) -> usize {
        self.basis.len()
    }
    /// The chart's dimension 2r − 1.
    pub fn chart_dimension(&self) -> usize {
        2 * self.rank() - 1
    }
    /// The weight of beam j (in `beams` order) in basis component k of a quantity linear in the beams' fields:
    /// γ_jk*/σ_k² (module documentation, # One field per frequency).
    pub fn dual(&self, j: usize, k: usize) -> C64 {
        self.coefficients[j][k].conj() / self.gram[k]
    }
    /// Its field at a point from all beams' amplitudes there (indexed as the request's beams).
    pub fn field(&self, amplitudes: &[Amplitude]) -> FieldAt {
        let _ = amplitudes;
        todo!("phase 2")
    }
    /// Its field's components on the basis, e_k = u_k†E, and their gradients (`[a][k]`).
    pub fn components(&self, field: &FieldAt) -> (Vec<C64>, Vec<[C64; 3]>) {
        let _ = field;
        todo!("phase 2")
    }
    /// Its chart at a point, from its field there.
    pub fn chart(&self, field: &FieldAt) -> Chart {
        let _ = field;
        todo!("phase 2")
    }
    /// The field E (V/m, spherical components) at chart coordinates s: a table node's field.
    pub fn chart_field(&self, s: &[f64]) -> [C64; 3] {
        let _ = s;
        todo!("phase 2")
    }
}

/// The light of a request: its beams' geometries and one coherent field per frequency.
#[derive(Clone, Debug, PartialEq)]
pub struct Light {
    /// Per request beam its geometry.
    pub beams: Vec<Geometry>,
    /// Per distinct frequency of the beams its field, in the order of their first beams.
    pub frequencies: Vec<Frequency>,
    /// The trap's frequency (an index into `frequencies`): the one of the tweezer, whose whole field is the trap light
    /// (SPEC §1.3).
    pub trap: usize,
}

impl Light {
    /// The light of a validated request (`Request::validate`).
    pub fn new(request: &Request) -> Light {
        let _ = request;
        todo!("phase 2")
    }
    /// Every beam's amplitude at r (µm).
    pub fn amplitudes(&self, r: V3) -> Vec<Amplitude> {
        self.beams.iter().map(|b| b.amplitude(r)).collect()
    }
    /// Whether a point is outside the trap's region: every focused beam's intensity there below `REGION_INTENSITY` of
    /// its focus value (module documentation, # The region).
    pub fn outside(&self, amplitudes: &[Amplitude]) -> bool {
        self.beams
            .iter()
            .zip(amplitudes)
            .filter(|(b, _)| b.waist.is_some())
            .all(|(_, a)| a.value.norm_sqr() < crate::units::REGION_INTENSITY)
    }
}
