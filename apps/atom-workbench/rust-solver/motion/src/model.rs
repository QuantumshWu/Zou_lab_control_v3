//! The model's shared types and its parts. The trajectory's model is the P the two-point check selected (select.rs) and
//! its one Reiter–Sørensen fold (`rs`), put in the integrator's frame and terms (`selected`), its terms judged once (`classify`: zero, drop, fold,
//! keep); the optical sector's
//! momentum noise beside it (`optical`); the trap's descent on a potential's expansion (`potential`). The types here are
//! what the integrator (step.rs) and the solve read of any model: a frequency exactly (`Beat`), a trajectory's point, a
//! scalar's second-order expansion, the trap, a cluster and its frames, a class of photons and the one photon rule
//! (`Moments`), the scattering beside the Ehrenfest force.
//!
//! # Notation (SPEC §0 units)
//! States s: the 32 of the model basis, the ground states first (`GROUND` of them), H0 diagonal in it (energies E_s, rad/µs, in
//! the frame where the excited states rotate at ν_frame). The emission operators C_a (a = x, y, z, ground ← excited, 1/√µs).
pub mod classify;
pub mod potential;
pub(crate) mod optical;
pub mod rs;
pub mod selected;
#[cfg(test)]
pub(crate) mod testing;
#[cfg(test)]
mod trap_tests;

use crate::linalg::C64;
pub use crate::request::{GROUND, STATES};

/// A frequency, exactly: rational coordinates on the request's frequency axes, then on the excited states' frame frequency
/// ν_frame -- integer numerators n_k over one positive denominator d, in lowest terms --; its value is 2π Σ_k n_k f_k / d
/// (rad/µs). Equal frequencies have equal coordinates: a frame or a beat that is zero in exact arithmetic is zero here,
/// whatever the floating point makes of its value. A cluster's frames (`Cluster::frames`) are these; the trajectory's
/// model holds its own frame exactly (floquet/frame.rs) and gives its clusters the zero frame.
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
    /// The size of the terms its value is the sum of, Σ_k |n_k a_k| / |d|: what the value's rounding is of.
    pub fn size(&self, axes: &[f64]) -> f64 {
        self.n.iter().zip(axes).map(|(&n, &a)| (n as f64 * a).abs()).sum::<f64>() / (self.d as f64).abs()
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

/// A trajectory state as the solve keeps it (its trial states, solve.rs).
#[derive(Clone, Debug, PartialEq)]
pub struct Sample { pub cluster: usize, pub at: Point, pub psi: Vec<C64> }

/// A cluster (SPEC §1.4): a trajectory's state space between two jumps, its ground states first and then its explicit
/// excited states. The trajectory's model has one, P (model/selected.rs, # One cluster, one frame).
#[derive(Clone, Debug, PartialEq)]
pub struct Cluster {
    /// Its states (indices of the model basis): the ground states, then the explicit excited states.
    pub states: Vec<usize>,
    /// How many of them are ground states.
    pub ground: usize,
    /// Each state's frame θ_s, exactly: ψ_s = e^{−iθ_s t} x_s, so that x_s holds E_s − θ_s on the diagonal and a term a ← b of
    /// beat Ω (∝ e^{iΩt}) is at Ω + θ_a − θ_b.
    pub frames: Vec<Beat>,
    /// Each state's Doppler frame K_s (1/µm): on a path r_h + v(t − t_m) the state is ψ_s = e^{iK_s·v(t − t_m)} x_s. The
    /// trajectory's model takes its time frame's spatial twin (model/selected.rs, # One cluster, one frame).
    pub waves: Vec<[f64; 3]>,
    /// The Doppler frames exactly, as integer coordinates on the light's wave vectors (`Beat`'s arithmetic).
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
    /// |k_φ| (1/µm), 2π/c times the class's photon frequency: the emission recoil is −k_φ n.
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

/// What the light a class's photons absorb does to the momentum per unit time (SPEC §1.6, the one rule; every model's
/// photons are read by it: model/selected.rs, and the classes `Photon` holds), at the state the photons were made of: the
/// local Kramers–Moyal coefficients of the method's own expansion, summed over the emitted photon's polarizations a --
/// the scattering force Σ_a Im⟨u_a|∇u_a⟩ and the second moment's rate Σ_a Re⟨∇u_a|⊗|∇u_a⟩, neither divided by the
/// class's rate W = Σ_a ‖u_a‖² (both smooth and finite where the light interferes away). A jump of the class carries its
/// conditional mean b = Σ_a Im⟨u_a|∇u_a⟩/W as far as one absorbed photon's momentum goes, s b with s = min(1, k_c/|b|):
/// b is the beams' wave vectors weighted, within their convex hull while the weights are positive, and beyond k_c only
/// where they interfere to both signs -- there it is no photon's momentum. What the jumps do not carry is the step's
/// (`Scattering::of_moments`). In the state's norm as made (`Moments::of`), or over ‖ψ‖² (`Moments::over`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Moments {
    /// W = Σ_a ‖u_a‖²: the class's rate.
    pub weight: f64,
    /// Σ_a Im⟨u_a|∇u_a⟩ (1/µm): the folded part's scattering force.
    pub mean: [f64; 3],
    /// Σ_a Re⟨∇u_a|⊗|∇u_a⟩ (1/µm²): the second moment's rate.
    pub second: [[f64; 3]; 3],
    /// s = min(1, k_c/|b|): the share of b a jump of the class carries (a ratio: the same in any norm).
    pub share: f64,
    /// s b (1/µm): what a jump of the class carries of its conditional mean b = mean/W.
    pub kick: [f64; 3],
    /// n: the landing states u_a lives on (the rounding's count, `Scattering::of_moments`).
    pub states: usize,
}

impl Moments {
    /// The moments of W, the mean and the second moment (in one norm) with k_c = `absorbed` the bound of what a jump
    /// carries: s and s b. Of a class without a rate the mean is nothing (its sum's every term holds an amplitude), and
    /// so is what its jumps, of which there are none, carry.
    pub fn new(weight: f64, mean: [f64; 3], second: [[f64; 3]; 3], absorbed: f64, states: usize) -> Moments {
        // W is a sum of squares: zero is the one rate without a mean, and a W that is not a number goes on.
        let b: [f64; 3] = mean.map(|x| if weight != 0.0 { x / weight } else { 0.0 });
        let size = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        let share = if size > absorbed { absorbed / size } else { 1.0 };
        Moments { weight, mean, second, share, kick: b.map(|x| share * x), states }
    }
    /// The moments of a class's amplitudes u_a and their gradients ∇_i u_a (`du[i][a]`; empty where none were made: no
    /// mean and no second moment) in the state's norm, k_c = `absorbed`.
    pub fn of(u: &[Vec<C64>; 3], du: &[[Vec<C64>; 3]; 3], absorbed: f64) -> Moments {
        use crate::linalg::{inner, norm_sqr};
        let weight: f64 = (0..3).map(|a| norm_sqr(&u[a])).sum();
        let mean: [f64; 3] = std::array::from_fn(|i| (0..3).map(|a| inner(&u[a], &du[i][a]).im).sum());
        let second: [[f64; 3]; 3] = std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|a| inner(&du[i][a], &du[j][a]).re).sum()));
        Moments::new(weight, mean, second, absorbed, u[0].len())
    }
    /// Over the state's norm ‖ψ‖² = `norm`: W, the mean and the second moment divided by it; s and s b as they are.
    pub fn over(self, norm: f64) -> Moments {
        Moments { weight: self.weight / norm, mean: self.mean.map(|x| x / norm), second: self.second.map(|row| row.map(|x| x / norm)), ..self }
    }
}

/// The light's action on the momentum per unit time beside the Ehrenfest force and the jumps' kicks (SPEC §1.6), at a
/// state and a point: the scattering force the jumps do not carry (1/µm/µs), the second moment's rate they do not carry,
/// 2D (1/µm²/µs; positive semidefinite by Cauchy–Schwarz), and the rounding of 2D's entries as they are summed -- `size`
/// the largest sum of a diagonal entry's terms' moduli, `links` the operations behind an entry (`of_moments`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Scattering { pub force: [f64; 3], pub diffusion: [[f64; 3]; 3], pub size: f64, pub links: usize }

impl Scattering {
    /// No light scattered.
    pub const NONE: Scattering = Scattering { force: [0.0; 3], diffusion: [[0.0; 3]; 3], size: 0.0, links: 0 };

    /// What a state's classes' photons give the momentum beside their jumps (`Moments`, each in the state's norm 1:
    /// `Moments::over`): the force Σ_c (1 − s_c) Σ_a Im⟨u_a|∇u_a⟩ and the diffusion 2D = Σ_c [Σ_a Re⟨∇u_a|⊗|∇u_a⟩ −
    /// s_c² W_c b_c b_cᵀ]. A plane wave's jump carries its k and leaves nothing; a standing wave's carries nothing and
    /// leaves k² R₀ sin² kz, the dipole force's fluctuations.
    ///
    /// The rounding of 2D's entries (step.rs `factor`). A class's three sums -- W, the mean m_i, the second moment
    /// M_ij, each of 3n products on its n landing states -- err by γ_{3n} of their products' moduli: W and M_ii
    /// themselves (their terms are positive), √(W M_ii) for m_i (Cauchy–Schwarz). Its diagonal entry M_ii −
    /// s² m_i²/W then errs by at most γ_{3n} M_ii for the second moment, 2γ_{3n} M_ii for the two means in the
    /// subtracted term (|m_i| ≤ √(W M_ii)) and the few operations of b, s and their products on s² W b_i²: within
    /// γ_{9n + 6} of M_ii + s² W b_i², the moduli of its terms (an off-diagonal entry within the geometric mean of two
    /// diagonal ones', Cauchy–Schwarz). Summed over the C classes: `size` = max_i Σ_c (M_ii + s_c² W_c b_ci²), the
    /// diagonal's terms of every class, and `links` = max_c (9 n_c + 6) + C − 1, the longest chain of operations behind
    /// an entry with the classes' sum.
    pub fn of_moments(classes: &[Moments]) -> Scattering {
        let (mut force, mut diffusion, mut sizes) = ([0.0; 3], [[0.0; 3]; 3], [0.0; 3]);
        let mut chain = 0;
        for m in classes {
            let b: [f64; 3] = m.mean.map(|x| if m.weight != 0.0 { x / m.weight } else { 0.0 });
            for i in 0..3 {
                force[i] += (1.0 - m.share) * m.mean[i];
                for j in 0..3 {
                    diffusion[i][j] += m.second[i][j] - m.share * m.share * m.weight * b[i] * b[j];
                }
                sizes[i] += m.second[i][i].abs() + m.share * m.share * m.weight * b[i] * b[i];
            }
            chain = chain.max(9 * m.states + 6);
        }
        let links = if classes.is_empty() { 0 } else { chain + classes.len() - 1 };
        Scattering { force, diffusion, size: crate::linalg::largest(0.0, sizes), links }
    }

    /// The scattering of a state of the norm ‖ψ‖² = `norm` from its classes' photons made with their gradients
    /// (`of_moments` of their moments over the norm).
    pub fn of(photons: &[Photon], norm: f64) -> Scattering {
        Scattering::of_moments(&photons.iter().map(|p| p.moments().over(norm)).collect::<Vec<_>>())
    }
}

impl Photon {
    /// The class's rate W = Σ_a ‖u_a‖² in its state's norm (`Moments::of` sums it the same).
    pub fn weight(&self) -> f64 {
        self.amplitudes.iter().map(|u| crate::linalg::norm_sqr(u)).sum()
    }
    /// The absorbed light's moments of the class (`Moments::of`), in its state's norm.
    pub fn moments(&self) -> Moments {
        Moments::of(&self.amplitudes, &self.gradients, self.absorbed)
    }
    /// The mean a jump of the class carries (`Moments::kick`): at most one absorbed photon's momentum k_c.
    pub fn kick(&self) -> [f64; 3] {
        self.moments().kick
    }
}
