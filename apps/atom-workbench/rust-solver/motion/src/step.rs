//! One step: A(τ/2) B(τ) A(τ/2) with the mid-step reference, the internal propagation along the path, the impulse, the
//! jump's root and the jump (SPEC §2.1, §1.6).
//!
//! # The motion and its splitting
//! A trajectory is (r, κ, ψ): position (µm), momentum κ = p/ħ (1/µm) and the cluster's unnormalized internal state between
//! jumps: ṙ = (ħ/m)κ, κ̇ = −⟨ψ|G|ψ⟩/⟨ψ|ψ⟩, iψ̇ = Hψ, H the cluster's generator and G = ∇H_h (`Dynamics::middle`).
//! Without the decay this is the Hamiltonian system 𝓗 = (ħ/m)κ²/2 + ⟨ψ|H_h|ψ⟩/⟨ψ|ψ⟩ on the product phase space. A step
//! splits it as 𝓗 = A + B, A = (ħ/m)κ²/2 + V_ref(r) and B the rest, and composes their flows A(τ/2) B(τ) A(τ/2):
//! symplectic in its conservative part, its own inverse, second order (propagation.md §2.1).
//!
//! # The reference (A)
//! V_ref is the second-order expansion of the fixed reference state's trap potential (`Dynamics::reference`) at the
//! step's middle p: V_ref(r) = V + g·(r − p) + ½(r − p)ᵀK(r − p). Its flow is exact along K's principal axes: with
//! y = e·(r − p), k = e·κ, curvature λ and gradient g along the axis e, ẏ = (ħ/m)k, k̇ = −(g + λy), solved over t by
//! y(t) = y c₀ + (ħ/m)k t c₁ − (ħ/m)g t² c₂ and k(t) = k c₀ − (λy + g) t c₁, the Stumpff functions c_k at x = (ħ/m)λt²
//! (`stumpff`): a rotation at ω² = (ħ/m)λ for λ > 0, a hyperbolic flow for λ < 0, free flight under the constant force for
//! λ = 0 -- one formula, entire in x (`Flow`). A step is therefore not bound to the trap's period. p is the fixed point
//! "the first half flow on V_ref(p) reaches p" (`middle`): the step's reverse takes the same reference, so the step is its
//! own inverse (Hut, Makino & McMillan 1995, the time-symmetrized leapfrog; propagation.md §2.2). Any geometry, one rule.
//!
//! # The kick (B)
//! The position stays at r_h; the internal state goes along the straight path r_h + v_m(t − t_m), the path's phases
//! exact: with ψ_s = e^{iK_s·v(t − t_m)} x_s in the cluster's Doppler frame (`Cluster::waves`) x evolves under Ĥ(t), the
//! generator on the path in that frame with the frame's diag(K_s·v). The model gives it by its fields, not by a list of
//! beating matrices (`Middle`): at the middle point, once a step, everything that depends on neither the momentum nor the
//! time; per momentum the beams' Doppler weights and path rates (`Middle::path`); and at a time s from the middle Ĥ(s)
//! itself, each beam's plane wave turned by e^{ik_l·v s} (`Middle::generator`). Ĥ is the same at every time where no
//! term beats in the frames (`Middle::still`: every pair of states holds one wave vector): then it is propagated through
//! its eigendecomposition at any time exactly (`spectral`), else by CF4 on a grid (`beating`). The impulse is exact in
//! its decomposition about the state at the step's middle, held in the frame, x_m = x(t_m) (= ψ(t_m), U(t_m) = 1):
//! - the shear: the conservative force at the fixed r_h on x_m, −∫⟨x_m|G(r_h, t)|x_m⟩dt/‖x_m‖². At a fixed point the
//!   gradient beats in time alone, G(r_h, t) = Σ_m G_m e^{iΩ_m(t − t_m)} over its tones (`Middle::tones`), each
//!   τ sinc(Ω_mτ/2) (`shear`). Of a state that does not change it depends on r_h alone: the conservative kick is a shear
//!   and the step symplectic (propagation.md §2.3);
//! - what the state's change along the path adds: the evolved state's force along the path less x_m's held in the frame,
//!   −∫(⟨x(t)|Ĝ(t)|x(t)⟩/‖x(t)‖² − ⟨x_m|Ĝ(t)|x_m⟩/‖x_m‖²)dt, Ĝ(t) the gradient on the path, one quadrature of the
//!   difference on the propagation's nodes (B1's `force_change`): it vanishes at every node, and with it its
//!   quadrature's error, where the state does not change -- a lone ground state in a standing wave's far light takes no
//!   grid error, however coarse its grid. The lag that friction and the Sisyphus effect live in;
//! - less the reference's force: + τ∇V_ref(r_h), since V_ref's force is the A flow's.
//!
//! The first two are read together (`impulse`): a mean ⟨x|G|x⟩/‖x‖² is tr(Gρ) of the state's density ρ, which the model
//! contracts without forming G (`Middle::force`). With ρ_m the middle state's and (s_k, w_k, ρ_k) the nodes' times from
//! the middle, weights and densities they are −Σ_m τ sinc(Ω_mτ/2) Re tr(G_m ρ_m) − Σ_k w_k Re tr(Ĝ(s_k)(ρ_k − ρ_m)); where
//! Ĥ is still, Ĝ is one matrix and the two are its one form with Q = τρ_m + Σ_k w_k(ρ_k − ρ_m).
//!
//! A generator's scalar part, tr/n -- the trap's light shift and the decay every state shares --, is a phase and a
//! factor on the whole state: the propagations take it out exactly (`traceless`, `linalg::exp_apply`), so that their
//! work and their rounding go with the states' differences alone.
//!
//! Held in the frame, not in the lab: a dressed state's coherences co-rotate with the light's phases along the path, so
//! its force along the path is the frame's, constant where Ĥ is static (a running wave's radiation pressure, steady on the
//! moving atom); held in the lab its coherence would slip against the light at K·v and its path force average away
//! (sinc(k·vτ/2)), counting the steady force twice -- a running wave's pressure came out doubled. Shear and path then differ
//! only by the path's beats, the light pattern's spatial modulation the atom crosses, which is what the shear holds at r_h.
//!
//! The middle state, not the start's: the step's reverse propagates back through the same x_m, so the decomposition is the
//! same both ways and the step its own inverse with the internal state changing too. About the start state the reverse
//! took its own start, x(t₀ + τ), and Q(x) = ∫[F(r_h, x) − F(path, x)]dt, the midpoint rule's O(τ³(v·∇)²F) on that state,
//! differed between the two: a two-level atom in a far-detuned standing wave missed its start by two hundred times the
//! step's own error (tests). B1's PGC read the same T on the middle state as on the start state (propagation.md §2.3).
//!
//! # The path's momentum
//! v_m = (ħ/m)(κ_h + Δκ/2), the mean of the momenta before and after the kick: the reverse step goes back on the same line
//! only at that velocity (propagation.md §2.3). It depends on the kick it makes, so it is found as the fixed point
//! v = (ħ/m)(κ_h + Δκ(v)/2) by one predictor and one correction: the predictor's kick, the shear and the reference's alone,
//! errs by τ ΔF/2 (ΔF the force's change over the step through the state), O(1) times τ where the step does not resolve the
//! internal state (a jump's bare state relaxing in 1/Γ ≪ τ) -- a first-order error in the velocity the friction acts on.
//! The correction multiplies it by the kick's sensitivity to v_m, (τ/2)|∂Δκ/∂κ_m| = O(τ), so one correction makes it
//! O(τ²) and the kick's error O(τ³), the scheme's own local order: p − 1 = 1 correction (`step`). The generator is taken
//! at the predictor's momentum κ_h for the shear and the first propagation, and at the corrected one for the step: one
//! middle, two paths through it, two propagations.
//!
//! # Jumps (SPEC §1.6)
//! ‖ψ‖² falls monotonically between jumps (the decay is −(i/2)ΣJ†J by the model's construction). Where it reaches the
//! trajectory's threshold θ within a step the root s* of ln‖ψ(s)‖² = ln θ is found by bracketed inverse interpolation, to
//! the norm's rounding (`root`); the caller redoes the step to it and jumps at its end (`jump`). A jump draws from its own stream
//! (`random::Stream::jump`) in a fixed order: the next threshold, the class, the polarization, the fluctuation's three
//! signs, and the direction's rejection loop last, so that the loop's length moves no other draw.

use std::f64::consts::PI;
use std::sync::OnceLock;

use crate::decide::COMPLEX;
use crate::linalg::{self, inner, norm_sqr, C64, Mat};
use crate::model::{Cluster, Photon, Point, Quadratic, Recoil, STATES};
use crate::random::Stream;
use crate::units::{rounding, UNIT_ROUNDOFF};

/// What one step needs from the model (SPEC §2.1, §1.6): defined here by its consumer, implemented by the model for its
/// `Model` with the run's lights and the reference state. Each method is one model function; the integrator defines
/// none of these quantities.
pub trait Dynamics {
    /// A cluster's generator at a position and a time (`Dynamics::middle`).
    type Middle<'a>: Middle
    where
        Self: 'a;

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

    /// The cluster's generator at the position r and the time t with every light on, for every momentum and along every
    /// straight path through the point (`Middle`): made once a step, at its middle.
    fn middle(&self, cluster: usize, r: [f64; 3], t: f64) -> Self::Middle<'_>;

    /// The photon classes of the cluster's state ψ (unnormalized) at a point (`Model::photons`).
    fn photons(&self, cluster: usize, at: &Point, psi: &[C64]) -> Vec<Photon>;

    /// The absorption recoil's moments of a photon into the drawn real polarization ε (`Photon::recoil`).
    fn recoil(&self, photon: &Photon, polarization: &[f64; 3]) -> Recoil;

    /// The loss energy E_s = (ħ/m)κ²/2 + V_run(r; p(ψ)) of the cluster's state at a point (`Model::energy`, rad/µs).
    fn energy(&self, cluster: usize, at: &Point, psi: &[C64]) -> f64;

    /// The loss test at a point (`Model::lost`): outside the region and E_s above its far value.
    fn lost(&self, cluster: usize, at: &Point, psi: &[C64]) -> bool;

    /// The displayed rows of the cluster's state at a point (`Model::readout`).
    fn readout(&self, cluster: usize, at: &Point, psi: &[C64]) -> [f64; STATES];

    /// The counted work of one middle (`decide::Work`, real multiply-adds), made once per run: the price of a step's
    /// middle beside its own work and its nodes' (`step_work`).
    fn assembly_work(&self) -> f64;

    /// The counted real multiply-adds of one jump's photons (`Dynamics::photons`), made once per run: `jump` adds it to the
    /// trajectory's work.
    fn photon_work(&self) -> f64;
}

/// A cluster's generator at a point (r_h, t_m) (module documentation, # The kick): what of it depends on neither the
/// momentum nor the time is made with it; a path through the point adds the momentum's part; and the generator and its
/// gradient's forms are asked at times along that path. In the cluster's Doppler frames throughout.
pub trait Middle {
    /// A path through the point: the Doppler weights at the momentum the generator is taken at, and the rates at which
    /// the light's plane waves and the Doppler frames turn at the path's velocity. Plain numbers.
    type Path;

    /// The cluster's states.
    fn states(&self) -> usize;

    /// Whether Ĥ is the same at every time along every path: no term beats in the cluster's frames, in time or along a
    /// path. Exact (the model's integer frequencies and wave vectors), not a comparison of numbers.
    fn still(&self) -> bool;

    /// The path at the velocity v (µm/µs), the generator taken at the momentum κ (1/µm).
    fn path(&self, kappa: [f64; 3], velocity: [f64; 3]) -> Self::Path;

    /// The Doppler frames' rates K_s·v on a path, per state (rad/µs).
    fn gauge(&self, path: &Self::Path, rates: &mut [f64]);

    /// Ĥ(s) at the time s from the middle along the path, into `h` (n × n; rad/µs), the frames' diag(K_s·v) with it.
    fn generator(&self, path: &Self::Path, s: f64, h: &mut Mat);

    /// Re tr(Ĝ_a(s) ρ) per lab axis a (rad/µs/µm): the gradient of Ĥ's Hermitian part at the time s from the middle along
    /// the path, contracted with a Hermitian ρ on the states (n × n, row by row).
    fn force(&self, path: &Self::Path, s: f64, rho: &[C64]) -> [f64; 3];

    /// The gradient at the fixed point by its tones, G(r_h, t) = Σ_m G_m e^{iΩ_m(t − t_m)}: to `each` every Ω_m (rad/µs)
    /// with Re tr(G_m ρ) per lab axis, for a Hermitian ρ.
    fn tones(&self, path: &Self::Path, rho: &[C64], each: &mut dyn FnMut(f64, [f64; 3]));

    /// The counted complex multiply-adds of one `generator`, one `force` and one `tones`.
    fn work(&self) -> [f64; 3];
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
    /// The step control's own (`adapt`): its step (µs), the grid's largest spacing (µs, `beating`), its count of steps and
    /// the time its cap was met.
    pub tau: f64,
    pub grid: f64,
    pub steps: u64,
    pub capped: Option<f64>,
    /// The work counted (`decide::Work`, real multiply-adds) of the integrator's own algorithms over the trajectory --
    /// exponentials, eigen-propagation, CF4 nodes with the model's generators and forces there, the doubling and the
    /// grid's doubling (`adapt` adds each step's) --, the middles it asked the model for (each costed by
    /// `Dynamics::assembly_work`) and its jumps' photons (`Dynamics::photon_work`).
    pub work: f64,
    pub assemblies: u64,
}

impl Trajectory {
    /// A trajectory at a basis state (a ground state of the model basis) at (r, κ) and time 0: ψ the unit vector of that
    /// state on its cluster, the first threshold from jump 0's stream (SPEC §2.4); the step control's own unset.
    pub fn new<D: Dynamics>(model: &D, state: usize, r: [f64; 3], kappa: [f64; 3], seed: u64, index: u64) -> Trajectory {
        let cluster = model.cluster_of(state);
        let states = &model.cluster(cluster).states;
        let at = states.iter().position(|&s| s == state).expect("a ground state lies on its cluster");
        let mut psi = vec![C64::ZERO; states.len()];
        psi[at] = C64::ONE;
        let threshold = Stream::jump(seed, index, 0).open_uniform();
        Trajectory {
            r,
            kappa,
            psi,
            cluster,
            time: 0.0,
            threshold,
            jumps: 0,
            lost: None,
            seed,
            index,
            tau: f64::INFINITY,
            grid: f64::INFINITY,
            steps: 0,
            capped: None,
            work: 0.0,
            assemblies: 0,
        }
    }
    /// The trajectory's point.
    pub fn point(&self) -> Point {
        Point { r: self.r, kappa: self.kappa, t: self.time }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// The reference flow (A).

/// sin(s)/s, or sinh(s)/s when `hyperbolic`, of s ≥ 0: the series 1 ∓ s²/6 where its next term s⁴/120 is within the unit
/// round-off (derived: there the series is exact to round-off), the quotient beyond (accurate wherever s ≠ 0).
fn ratio(s: f64, hyperbolic: bool) -> f64 {
    if s * s * s * s <= 120.0 * UNIT_ROUNDOFF {
        if hyperbolic { 1.0 + s * s / 6.0 } else { 1.0 - s * s / 6.0 }
    } else if hyperbolic {
        s.sinh() / s
    } else {
        s.sin() / s
    }
}

/// The Stumpff functions (c₀, c₁, c₂) at x (Stumpff 1959): c_k(x) = Σ_j (−x)^j/(k + 2j)!, entire in x -- cos √x,
/// sin √x/√x and (1 − cos √x)/x for x > 0, their hyperbolic forms for x < 0, (1, 1, ½) at 0 --, without cancellation:
/// c₂(x) = ½ c₁(x/4)².
pub fn stumpff(x: f64) -> (f64, f64, f64) {
    let hyperbolic = x < 0.0;
    let s = x.abs().sqrt();
    let c0 = if hyperbolic { s.cosh() } else { s.cos() };
    let half = ratio(0.5 * s, hyperbolic);
    (c0, ratio(s, hyperbolic), 0.5 * half * half)
}

/// sin(x)/x of a real x: c₁(x²).
fn sinc(x: f64) -> f64 {
    stumpff(x * x).1
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

impl Flow {
    /// The flow of the expansion `q` at p.
    fn new(q: &Quadratic, centre: [f64; 3], hbar_over_mass: f64) -> Flow {
        let (curvatures, axes) = linalg::symmetric3_eigen(&q.hessian);
        let gradient = std::array::from_fn(|i| linalg::dot3(axes[i], q.gradient));
        Flow { centre, axes, curvatures, gradient, hbar_over_mass }
    }
    /// (r, κ) after the time t (either sign) from (r, κ) (module documentation, # The reference).
    pub fn advance(&self, from: ([f64; 3], [f64; 3]), t: f64) -> ([f64; 3], [f64; 3]) {
        self.reach(from, t).0
    }
    /// `advance`, and the sizes its position's and its momentum's coordinates are summed from (the centre's and every
    /// axis's terms' moduli): `units::rounding` of a coordinate's operations (`REACH`) times them bounds its rounding.
    fn reach(&self, (r, kappa): ([f64; 3], [f64; 3]), t: f64) -> (([f64; 3], [f64; 3]), [f64; 2]) {
        let hm = self.hbar_over_mass;
        let d = [r[0] - self.centre[0], r[1] - self.centre[1], r[2] - self.centre[2]];
        let (mut r1, mut k1) = (self.centre, [0.0; 3]);
        let mut sizes = [self.centre.iter().fold(0.0, |m: f64, x| m.max(x.abs())), 0.0];
        for i in 0..3 {
            let e = self.axes[i];
            let (y, k, lambda, g) = (linalg::dot3(e, d), linalg::dot3(e, kappa), self.curvatures[i], self.gradient[i]);
            let (c0, c1, c2) = stumpff(hm * lambda * t * t);
            let position = [y * c0, hm * k * t * c1, -hm * g * t * t * c2];
            let momentum = [k * c0, -(lambda * y + g) * t * c1];
            let (y1, k_1) = (position[0] + position[1] + position[2], momentum[0] + momentum[1]);
            sizes[0] += position.iter().map(|x| x.abs()).sum::<f64>();
            sizes[1] += momentum.iter().map(|x| x.abs()).sum::<f64>();
            for a in 0..3 {
                r1[a] += y1 * e[a];
                k1[a] += k_1 * e[a];
            }
        }
        ((r1, k1), sizes)
    }
    /// ∇V_ref at r (rad/µs/µm): Σ_i e_i (g_i + λ_i e_i·(r − p)).
    fn gradient_at(&self, r: [f64; 3]) -> [f64; 3] {
        let d = [r[0] - self.centre[0], r[1] - self.centre[1], r[2] - self.centre[2]];
        let mut out = [0.0; 3];
        for i in 0..3 {
            let e = self.axes[i];
            let along = self.gradient[i] + self.curvatures[i] * linalg::dot3(e, d);
            for a in 0..3 {
                out[a] += along * e[a];
            }
        }
        out
    }
    /// The largest rate of the flow, max_i √((ħ/m)|λ_i|) (rad/µs): κ's frequency along its axes (a hyperbolic axis' rate).
    fn rate(&self) -> f64 {
        self.curvatures.iter().map(|l| (self.hbar_over_mass * l.abs()).sqrt()).fold(0.0, f64::max)
    }
}

/// Rounds of an iteration that halves its distance each round at least -- a bracket's bisection (`root`), and the
/// middle's rounds in a stable step, whose contraction is no less (propagation.md §2.2) --: the unit round-off is
/// 2⁻⁵³, the mantissa's bits, so after as many rounds the distance is the round-off's.
const ROUNDS: usize = f64::MANTISSA_DIGITS as usize;

/// The operations between a flow's inputs and a coordinate of its reach (`Flow::reach`): the coordinate adds three
/// axes' displacements to the centre (6), each the Stumpff combination of three terms (10) of the axis's coordinate (a
/// difference and a product of three: 6) and of its x (3) through its functions (3): 28. `units::rounding` of them
/// times the sizes the coordinate is summed from is the rounding of a reach.
const REACH: usize = 28;

/// The step's middle (module documentation, # The reference): the fixed point p of "the first half flow on V_ref(p)
/// reaches p" from (r, κ) over `half`, from the free flight's middle; the flow and (r_h, κ_h). A round moves p by the
/// reference's change over the move times the flow's response, (ħ/m)|∇K|·|r − p|·half²/3 of the last move: a contraction
/// while the step resolves the potential's third derivative, by a factor of 10⁻² to 10⁻⁴ a round at the steps a trap
/// takes (so that an accelerated iteration -- Broyden's, Anderson's -- gains less than one round of some six).
///
/// Converged is: the reach's move within its rounding (`REACH`) -- p is its own reach as far as the flow tells --, or
/// the next round's moves within it. The step's results depend on the centre p only through that third derivative:
/// taken at p with the reach (r_h, κ_h) they differ from the fixed point's by the next round's moves of the reach's
/// position and momentum, the last ones times the contraction, for which the largest measured stands (the centre itself
/// then lies a move, not a rounding, from its reach, which no result sees). Everything else is refused: a move that
/// does not shrink (the map is no contraction at this step: no fixed point is being reached), or `ROUNDS` rounds of a
/// contraction too slow for a stable step.
fn middle<D: Dynamics>(model: &D, r: [f64; 3], kappa: [f64; 3], half: f64) -> Result<(Flow, ([f64; 3], [f64; 3])), Refused> {
    let hm = model.hbar_over_mass();
    let apart = |a: [f64; 3], b: [f64; 3]| (0..3).map(|i| (a[i] - b[i]).abs()).fold(0.0, f64::max);
    let mut p: [f64; 3] = std::array::from_fn(|a| r[a] + hm * kappa[a] * half);
    // The last move of the reach's position, the momentum it reached then, and the largest contraction measured (the
    // first round measures none: its move over no move before it is zero).
    let (mut last, mut before, mut contraction) = (f64::INFINITY, kappa, 0.0_f64);
    for _ in 0..ROUNDS {
        let flow = Flow::new(&model.reference(p), p, hm);
        let (reached, sizes) = flow.reach((r, kappa), half);
        let moved = apart(reached.0, p);
        if !moved.is_finite() || !reached.1.iter().all(|k| k.is_finite()) {
            return Err(Refused::Overflow);
        }
        contraction = contraction.max(moved / last);
        let within = sizes.map(|size| rounding(REACH) * size);
        let next = [contraction * moved, contraction * apart(reached.1, before)];
        if moved <= within[0] || (last.is_finite() && next[0] <= within[0] && next[1] <= within[1]) {
            return Ok((flow, reached));
        }
        if moved >= last {
            return Err(Refused::Reference);
        }
        (last, before, p) = (moved, reached.1, reached.0);
    }
    Err(Refused::Reference)
}

// ---------------------------------------------------------------------------------------------------------------------
// Quadrature to round-off.

/// A Gauss–Legendre rule of m nodes on [−1, 1] (nodes, weights) and its panel phase x_m: the largest ν w at which its
/// error on a component e^{iνs} over a panel of width w, w (νw)^{2m} (m!)⁴/((2m + 1)((2m)!)³) (Davis & Rabinowitz
/// 2.7.12), is within the unit round-off times w -- the panel a rule integrates an entire function of that bandwidth on to
/// round-off.
struct Rule {
    nodes: Vec<f64>,
    weights: Vec<f64>,
    phase: f64,
}

impl Rule {
    fn new(m: usize) -> Rule {
        let (mut nodes, mut weights) = (Vec::with_capacity(m), Vec::with_capacity(m));
        for i in 1..=m {
            // Newton on P_m from cos(π(i − ¼)/(m + ½)), until the step stops shrinking (the round-off's floor).
            let mut x = (PI * (i as f64 - 0.25) / (m as f64 + 0.5)).cos();
            let mut last = f64::INFINITY;
            let derivative = loop {
                let (p, dp) = legendre(m, x);
                let dx = p / dp;
                if !(dx.abs() < last) {
                    break dp;
                }
                last = dx.abs();
                x -= dx;
            };
            nodes.push(x);
            weights.push(2.0 / ((1.0 - x * x) * derivative * derivative));
        }
        let ln_factorial = |n: usize| (1..=n).map(|k| (k as f64).ln()).sum::<f64>();
        let ln = UNIT_ROUNDOFF.ln() + ((2 * m + 1) as f64).ln() + 3.0 * ln_factorial(2 * m) - 4.0 * ln_factorial(m);
        Rule { nodes, weights, phase: (ln / (2 * m) as f64).exp() }
    }
}

/// P_m(x) and P_m′(x) by the three-term recurrence.
fn legendre(m: usize, x: f64) -> (f64, f64) {
    let (mut p0, mut p1) = (1.0, x);
    for j in 2..=m {
        let p2 = ((2 * j - 1) as f64 * x * p1 - (j - 1) as f64 * p0) / j as f64;
        p0 = p1;
        p1 = p2;
    }
    (p1, m as f64 * (x * p1 - p0) / (x * x - 1.0))
}

/// The rules of 2, 4, …, 64 nodes (two nodes at least: exact for the cubics a free flight's κ² is).
fn rules() -> &'static [Rule] {
    static RULES: OnceLock<Vec<Rule>> = OnceLock::new();
    RULES.get_or_init(|| (1..=6).map(|k| Rule::new(1 << k)).collect())
}

/// The rule and the panels over a width of an entire integrand of bandwidth ν to round-off: the fewest nodes, each panel
/// of width w within its rule's phase, νw ≤ x_m. Its node count is the quadrature's work (`count::spectral`).
fn plan(width: f64, bandwidth: f64) -> (&'static Rule, usize) {
    rules()
        .iter()
        .map(|rule| (rule, ((bandwidth * width / rule.phase).ceil() as usize).max(1)))
        .min_by_key(|(rule, panels)| rule.nodes.len() * panels)
        .unwrap()
}

/// The nodes and weights over [a, b] of an entire integrand of bandwidth ν to round-off (`plan`), panel by panel as they
/// are read: a stiff generator's millions of nodes are never held at once.
fn quadrature(a: f64, b: f64, bandwidth: f64) -> impl Iterator<Item = (f64, f64)> {
    let (rule, panels) = plan((b - a).abs(), bandwidth);
    let w = (b - a) / panels as f64;
    (0..panels).flat_map(move |p| {
        let centre = a + (p as f64 + 0.5) * w;
        rule.nodes.iter().zip(&rule.weights).map(move |(x, weight)| (centre + 0.5 * w * x, 0.5 * w * weight))
    })
}

// ---------------------------------------------------------------------------------------------------------------------
// The step's path and its moments.

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
    /// The middle's time t_m.
    fn middle(&self) -> f64 {
        self.start + 0.5 * self.tau
    }
    /// ∫κ_a dt and ∫κ_a² dt (1/µm·µs, 1/µm²·µs) per lab axis over [from, to] within the step, on its path: per side of the
    /// middle, Gauss–Legendre to round-off on the flow, κ² of bandwidth twice the flow's rate (`quadrature`).
    pub fn moments(&self, from: f64, to: f64) -> ([f64; 3], [f64; 3]) {
        let (m, (lo, hi)) = (self.middle(), (from.min(to), from.max(to)));
        let sign = if to >= from { 1.0 } else { -1.0 };
        let (first, last) = (self.start.min(self.start + self.tau), self.start.max(self.start + self.tau));
        let bandwidth = 2.0 * self.flow.rate();
        let (mut one, mut two) = ([0.0; 3], [0.0; 3]);
        // The step's two sides in time order, each with the flow's start and its time origin.
        let sides = if self.tau >= 0.0 {
            [(first, m, self.from, self.start), (m, last, self.kicked, m)]
        } else {
            [(first, m, self.kicked, m), (m, last, self.from, self.start)]
        };
        for (a, b, origin, at) in sides {
            let (a, b) = (a.max(lo), b.min(hi));
            if !(b > a) {
                continue;
            }
            for (t, w) in quadrature(a, b, bandwidth) {
                let k = self.flow.advance(origin, t - at).1;
                for x in 0..3 {
                    one[x] += sign * w * k[x];
                    two[x] += sign * w * k[x] * k[x];
                }
            }
        }
        (one, two)
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// The generator along the path (B).

/// The cluster's generator along the step's path in its Doppler frame (module documentation, # The kick): the model's
/// generator at the middle (`Middle`) with the path through it, and per state the gauge's phase over half the step --
/// the frame's state at the step's start is x₀ = e^{iK_s·v τ/2} ψ₀, and at its end ψ = e^{iK_s·v τ/2} x.
struct Along<'a, M: Middle> {
    middle: &'a M,
    path: M::Path,
    half: [C64; STATES],
    n: usize,
}

impl<'a, M: Middle> Along<'a, M> {
    /// The path through a middle at the velocity v, its generator at the momentum κ, for a step of τ.
    fn new(middle: &'a M, kappa: [f64; 3], velocity: [f64; 3], tau: f64) -> Along<'a, M> {
        let (n, path) = (middle.states(), middle.path(kappa, velocity));
        let mut gauge = [0.0; STATES];
        middle.gauge(&path, &mut gauge[..n]);
        let mut half = [C64::ZERO; STATES];
        for s in 0..n {
            half[s] = C64::cis(0.5 * gauge[s] * tau);
        }
        Along { middle, path, half, n }
    }
}

/// The panels of two spacings over τ with spacings no wider than h (`beating`): ⌈|τ|/2h⌉, at least one; a quotient within
/// its rounding of a whole number is that number (a spacing that divides the step).
pub fn panels(tau: f64, spacing: f64) -> usize {
    let x = tau.abs() / (2.0 * spacing);
    let whole = x.round();
    let panels = if (x - whole).abs() <= rounding(4) * x { whole } else { x.ceil() };
    (panels as usize).max(1)
}

/// A state's density matrix ρ = |x⟩⟨x|/⟨x|x⟩, ρ_ab = x_a x_b*/‖x‖², row by row: a mean ⟨x|G|x⟩/⟨x|x⟩ is tr(Gρ), which the
/// model contracts (`Middle::force`, `Middle::tones`).
fn density(x: &[C64], rho: &mut [C64]) {
    let (n, scale) = (x.len(), 1.0 / norm_sqr(x));
    for (a, left) in x.iter().enumerate() {
        let left = *left * scale;
        for (b, right) in x.iter().enumerate() {
            rho[a * n + b] = left * right.conj();
        }
    }
}

/// The shear (module documentation, # The kick): −∫⟨ψ|G(r_h, t)|ψ⟩dt/‖ψ‖² over the step of a state held, at the fixed r_h
/// each tone at its beat Ω_m alone, ∫e^{iΩ_m(t − t_m)}dt = τ sinc(Ω_mτ/2). `rho` takes the state's density.
fn shear<M: Middle>(along: &Along<M>, rho: &mut [C64], psi: &[C64], tau: f64) -> [f64; 3] {
    density(psi, rho);
    let mut f = [0.0; 3];
    along.middle.tones(&along.path, rho, &mut |omega, form| {
        let w = tau * sinc(0.5 * omega * tau);
        for a in 0..3 {
            f[a] -= w * form[a];
        }
    });
    f
}

/// A step's scratch memory, made once and fitted to each use: the matrix of an exponential (or the still Ĥ₀), the
/// generator at a CF4 spacing's two nodes, and one array carved (`Scratch::carve`) into a grid's node states, the middle
/// state's density, a node's change of density, their weighted sum (`impulse`), and an exponential's two work vectors
/// (`linalg::exp_apply`).
struct Scratch {
    matrix: Mat,
    nodes: [Mat; 2],
    rest: Vec<C64>,
}

struct Carved<'a> {
    matrix: &'a mut Mat,
    generators: &'a mut [Mat; 2],
    states: &'a mut [C64],
    held: &'a mut [C64],
    change: &'a mut [C64],
    sum: &'a mut [C64],
    work: &'a mut [C64],
}

impl Scratch {
    fn new(n: usize) -> Scratch {
        Scratch { matrix: Mat::zeros(n, n), nodes: [Mat::zeros(n, n), Mat::zeros(n, n)], rest: Vec::new() }
    }
    /// The scratch for a grid of `nodes` spacings.
    fn carve(&mut self, nodes: usize) -> Carved<'_> {
        let n = self.matrix.rows;
        let sizes = [(nodes + 1) * n, n * n, n * n, n * n];
        self.rest.resize(sizes.iter().sum::<usize>() + 2 * n, C64::ZERO);
        let (states, rest) = self.rest.split_at_mut(sizes[0]);
        let (held, rest) = rest.split_at_mut(sizes[1]);
        let (change, rest) = rest.split_at_mut(sizes[2]);
        let (sum, work) = rest.split_at_mut(sizes[3]);
        Carved { matrix: &mut self.matrix, generators: &mut self.nodes, states, held, change, sum, work }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// The propagations.

/// What a propagation of the frame's state over the step made: ψ at the step's end (lab; the cluster's states first),
/// the kick's part of the state (`impulse`), the first crossing of the threshold (from the step's start), the panels of
/// its grid (none: through the eigendecomposition), and its counted work.
struct Propagated {
    end: [C64; STATES],
    impulse: [f64; 3],
    crossing: Option<f64>,
    panels: Option<usize>,
    work: f64,
}

/// The counted work (`decide::Work`, real multiply-adds) of the integrator's operations on n states, one formula each,
/// the model's own counts of its generator, its force and its tones at a node with them (`Middle::work`): what the
/// propagations add up and what `step_work` prices a step by.
mod count {
    use super::COMPLEX;
    /// The eigen-propagation through `nodes` quadrature nodes: the still generator, the decomposition's own count
    /// (`linalg::Eigen::work`), and per node n exponentials and the state R d of n².
    pub fn spectral(n: usize, model: [f64; 3], decomposition: f64, nodes: usize) -> f64 {
        COMPLEX * (model[0] + decomposition + (nodes * (n + n * n)) as f64)
    }
    /// One spacing of CF4: the generator at its two nodes, their two combinations of n² each, and the two exponentials'
    /// `products` matrix-vector products of n² in all.
    pub fn cf4(n: usize, model: [f64; 3], products: usize) -> f64 {
        COMPLEX * (2.0 * model[0] + ((4 + products) * n * n) as f64)
    }
    /// The kick's part of the state through `nodes` nodes (`impulse`): the middle state's density; per node its density
    /// and its change (2n²); and of a still generator the changes' sum (n² a node) and its one form, of a beating one
    /// the middle state's tones and a form a node.
    pub fn impulse(n: usize, model: [f64; 3], nodes: usize, still: bool) -> f64 {
        let forms = if still { (nodes * n * n) as f64 + model[1] } else { model[2] + nodes as f64 * model[1] };
        COMPLEX * (((1 + 2 * nodes) * n * n) as f64 + forms)
    }
    /// The predictor's shear: the start state's density and its tones.
    pub fn shear(n: usize, model: [f64; 3]) -> f64 {
        COMPLEX * ((n * n) as f64 + model[2])
    }
}

/// The node count of a quadrature over τ at a bandwidth (`plan`).
fn nodes(tau: f64, bandwidth: f64) -> usize {
    let (rule, panels) = plan(tau.abs(), bandwidth);
    rule.nodes.len() * panels
}

/// A generator's scalar part c = tr(A)/n, taken out of it: e^{−iAs} = e^{−ics} e^{−i(A − c)s} exactly, c a phase and a
/// decay every state shares -- the trap's light shift, hundreds of rad/µs on every state alike, is one. The propagations
/// take the rest, whose norm sizes their work and their rounding, and the scalar's exponential beside it.
fn traceless(a: &mut Mat) -> C64 {
    let scalar = a.trace() / a.rows as f64;
    for i in 0..a.rows {
        a[(i, i)] -= scalar;
    }
    scalar
}

/// The kick's part of the state (module documentation, # The kick): the shear of the middle state and the state's
/// change's force integrated on the propagation's nodes, −Σ_m τ sinc(Ω_mτ/2) Re tr(G_m ρ_m) − Σ_k w_k Re tr(Ĝ(s_k)
/// (ρ_k − ρ_m)): ρ_m the middle state's density, held in the frame, and (s_k, w_k, x_k) the nodes' times from the middle,
/// their weights and their states, which `visit` gives in turn. Where Ĥ is still its gradient is one matrix at every
/// time, and the whole is its one form with Q = τρ_m + Σ_k w_k(ρ_k − ρ_m) (into `sum`). The change vanishes at every node
/// where the state does not.
fn impulse<M: Middle>(along: &Along<M>, held: &mut [C64], change: &mut [C64], sum: &mut [C64], tau: f64, middle: &[C64], visit: &mut dyn FnMut(&mut dyn FnMut(f64, f64, &[C64]))) -> [f64; 3] {
    if along.middle.still() {
        density(middle, held);
        for (q, rho) in sum.iter_mut().zip(held.iter()) {
            *q = *rho * tau;
        }
        visit(&mut |_, w, x| {
            density(x, change);
            for ((q, d), rho) in sum.iter_mut().zip(change.iter()).zip(held.iter()) {
                *q += (*d - *rho) * w;
            }
        });
        return along.middle.force(&along.path, 0.0, sum).map(|f| -f);
    }
    let mut out = shear(along, held, middle, tau);
    visit(&mut |s, w, x| {
        density(x, change);
        for (d, rho) in change.iter_mut().zip(held.iter()) {
            *d -= *rho;
        }
        let form = along.middle.force(&along.path, s, change);
        for a in 0..3 {
            out[a] -= w * form[a];
        }
    });
    out
}

/// A static Ĥ₀'s eigendecomposition for its propagation: Ĥ₀ = c + RΛR⁻¹ with the scalar part c apart (`traceless`), the
/// eigenvalues c + Λ, the bandwidth max|λ_b* − λ_c| of the forms on a state propagated through them (their frequencies
/// and decay rates), its quadrature's nodes over the step, and the rounding bound of a state propagated through it
/// relative to its norm, κ(R)·u grown by the decay rates' spread over the step (`linalg::Eigen`).
struct Spectrum {
    eigen: linalg::Eigen,
    lambda: [C64; STATES],
    bandwidth: f64,
    nodes: usize,
    rounding: f64,
}

/// The route of a propagation over τ, one rule for the propagation and for its price (`propagate`, `step_work`): through
/// the eigendecomposition where Ĥ is still along the path, its decomposition exists, and a state propagated through it
/// rounds no more than through the exponential of Ĥ₀ on the vector, the other exact evaluation of the same state
/// (`linalg::Chebyshev::rounding` of its plan over the step) -- towards an exceptional point κ(R) grows without bound and
/// the exponential, whose rounding does not know it, takes over; on the grid otherwise (None), whose error its own
/// reading books (`Step::regrid`). `matrix` is left Ĥ₀'s traceless part; `sums` holds n numbers.
fn route<M: Middle>(along: &Along<M>, matrix: &mut Mat, sums: &mut [C64], tau: f64) -> Option<Spectrum> {
    if !along.middle.still() {
        return None;
    }
    let n = along.n;
    along.middle.generator(&along.path, 0.0, matrix);
    let scalar = traceless(matrix);
    let (rho, gamma) = linalg::bounds(matrix, sums);
    let eigen = linalg::eigen(matrix)?;
    let mut lambda = [C64::ZERO; STATES];
    for b in 0..n {
        lambda[b] = eigen.values[b] + scalar;
    }
    let values = &lambda[..n];
    let bandwidth = values.iter().flat_map(|a| values.iter().map(move |b| (a.conj() - *b).abs())).fold(0.0, f64::max);
    let growth = values.iter().map(|l| l.im).fold(f64::NEG_INFINITY, f64::max) - values.iter().map(|l| l.im).fold(f64::INFINITY, f64::min);
    let rounding = eigen.condition * UNIT_ROUNDOFF * (1.0 + tau.abs() * growth);
    let exponential = linalg::Chebyshev::plan(rho * tau.abs(), gamma * tau.abs()).rounding(n);
    (rounding <= exponential).then(|| Spectrum { eigen, lambda, bandwidth, nodes: nodes(tau, bandwidth), rounding })
}

/// The propagation of a static Ĥ₀ through its eigendecomposition (`Spectrum`; exact at any time): in the coordinates
/// d = R⁻¹x, d(s) = e^{−i(c + Λ)s}d₀ and x(s) = R d(s), the impulse by Gauss–Legendre at the bandwidth max|λ_b* − λ_c|
/// (`quadrature`), the norm's root on ‖x(s)‖² (`root`).
fn spectral<M: Middle>(path: &Along<M>, scratch: &mut Scratch, spectrum: &Spectrum, psi: &[C64], tau: f64, threshold: Option<f64>) -> Propagated {
    let n = path.n;
    let model = path.middle.work();
    let Carved { held, change, sum, .. } = scratch.carve(0);
    let (e, lambda) = (&spectrum.eigen, &spectrum.lambda[..n]);
    let mut d0 = [C64::ZERO; STATES];
    for b in 0..n {
        d0[b] = (0..n).map(|c| e.inverse[(b, c)] * (psi[c] * path.half[c])).sum();
    }
    let at = |s: f64, x: &mut [C64]| {
        let mut d = [C64::ZERO; STATES];
        for b in 0..n {
            d[b] = d0[b] * (C64::new(0.0, -s) * lambda[b]).exp();
        }
        for (a, x) in x.iter_mut().enumerate() {
            *x = e.vectors.data[a * n..(a + 1) * n].iter().zip(&d).map(|(r, d)| *r * *d).sum();
        }
    };
    let (mut middle, mut end, mut x) = ([C64::ZERO; STATES], [C64::ZERO; STATES], [C64::ZERO; STATES]);
    at(0.5 * tau, &mut middle[..n]);
    let impulse = impulse(path, held, change, sum, tau, &middle[..n], &mut |each: &mut dyn FnMut(f64, f64, &[C64])| {
        for (s, w) in quadrature(0.0, tau, spectrum.bandwidth) {
            at(s, &mut x[..n]);
            each(s - 0.5 * tau, w, &x[..n]);
        }
    });
    at(tau, &mut end[..n]);
    let crossing = threshold.filter(|&theta| tau > 0.0 && norm_sqr(&end[..n]) <= theta).map(|theta| {
        let norm = |s: f64| {
            let mut x = [C64::ZERO; STATES];
            at(s, &mut x[..n]);
            norm_sqr(&x[..n])
        };
        root(norm, (0.0, norm_sqr(psi)), (tau, norm_sqr(&end[..n])), theta, 2.0 * spectrum.rounding + rounding(n + 1))
    });
    for (z, phase) in end[..n].iter_mut().zip(&path.half) {
        *z *= *phase;
    }
    let work = count::spectral(n, model, e.work, spectrum.nodes) + count::impulse(n, model, spectrum.nodes, true);
    Propagated { end, impulse, crossing, panels: None, work }
}

/// CF4, the fourth-order commutator-free Magnus scheme of two exponentials (Blanes & Moan 2006): over h from the time s
/// (from the middle), x ← e^{−ih(w₂Ĥ₁ + w₁Ĥ₂)} e^{−ih(w₁Ĥ₁ + w₂Ĥ₂)} x, Ĥ_i = Ĥ(s + c_i h) at the Gauss nodes c = ½ ∓ √3/6,
/// w = ¼ ± √3/6; exact where Ĥ is still (w₁ + w₂ = ½ each). The generator at the two nodes is made once, into
/// `generators`; each exponential on the vector to round-off (`linalg::exp_apply`, which takes the combination's scalar
/// part out and overwrites `a`; `vectors` its work). Its counted work and its rounding of x relative to its norm
/// (`linalg::Chebyshev::rounding`).
fn cf4<M: Middle>(path: &Along<M>, a: &mut Mat, generators: &mut [Mat; 2], vectors: &mut [C64], x: &mut [C64], s: f64, h: f64) -> (f64, f64) {
    let r = 3.0_f64.sqrt() / 6.0;
    path.middle.generator(&path.path, s + (0.5 - r) * h, &mut generators[0]);
    path.middle.generator(&path.path, s + (0.5 + r) * h, &mut generators[1]);
    let (mut products, mut bound) = (0, 0.0);
    for [w1, w2] in [[0.25 + r, 0.25 - r], [0.25 - r, 0.25 + r]] {
        for (z, (one, two)) in a.data.iter_mut().zip(generators[0].data.iter().zip(&generators[1].data)) {
            *z = *one * w1 + *two * w2;
        }
        let plan = linalg::exp_apply(a, C64::new(0.0, -h), x, vectors);
        products += plan.parts * plan.terms;
        bound += plan.rounding(path.n);
    }
    (count::cf4(path.n, path.middle.work(), products), bound)
}

/// The propagation of Ĥ(t) by CF4 (`cf4`) on 2P spacings h, P = `panels`, the impulse by composite Simpson on its
/// nodes: both of fourth order in νh, ν the force's frequency bound, an error the step's doubling does not see (its
/// steps take the same spacing), read by doubling the panels (`Step::regrid`). The middle state is the middle node's,
/// whose change is none. The norm's root between the nodes that bracket it, by CF4 from the earlier node.
fn beating<M: Middle>(path: &Along<M>, scratch: &mut Scratch, psi: &[C64], tau: f64, panels: usize, threshold: Option<f64>) -> Propagated {
    let (n, nodes) = (path.n, 2 * panels);
    let h = tau / nodes as f64;
    let Carved { matrix, generators, states, held, change, sum, work: vectors } = scratch.carve(nodes);
    for (x, (z, phase)) in states.iter_mut().zip(psi.iter().zip(&path.half)) {
        *x = *z * *phase;
    }
    let (mut work, mut crossing) = (0.0, None);
    for k in 0..nodes {
        let s = k as f64 * h - 0.5 * tau;
        let (before, after) = states[k * n..(k + 2) * n].split_at_mut(n);
        after.copy_from_slice(before);
        let (exponentials, bound) = cf4(path, matrix, generators, vectors, after, s, h);
        work += exponentials;
        if let Some(theta) = threshold.filter(|&theta| crossing.is_none() && tau > 0.0 && norm_sqr(after) <= theta) {
            let from = k as f64 * h;
            let norm = |to: f64| {
                let mut y = [C64::ZERO; STATES];
                y[..n].copy_from_slice(before);
                cf4(path, matrix, generators, vectors, &mut y[..n], s, to - from);
                norm_sqr(&y[..n])
            };
            crossing = Some(root(norm, (from, norm_sqr(before)), (from + h, norm_sqr(after)), theta, 2.0 * bound + rounding(n + 1)));
        }
    }
    // Composite Simpson on the nodes, weights 1, 4, 2, …, 4, 1 over h/3.
    let states = &*states;
    let impulse = impulse(path, held, change, sum, tau, &states[panels * n..(panels + 1) * n], &mut |each: &mut dyn FnMut(f64, f64, &[C64])| {
        for k in (0..=nodes).filter(|&k| k != panels) {
            let weight = if k == 0 || k == nodes { 1.0 } else if k % 2 == 1 { 4.0 } else { 2.0 };
            each(k as f64 * h - 0.5 * tau, h / 3.0 * weight, &states[k * n..(k + 1) * n]);
        }
    });
    let mut end = [C64::ZERO; STATES];
    for (z, (x, phase)) in end.iter_mut().zip(states[nodes * n..].iter().zip(&path.half)) {
        *z = *x * *phase;
    }
    work += count::impulse(n, path.middle.work(), nodes, false);
    Propagated { end, impulse, crossing, panels: Some(panels), work }
}

/// A root in [lo, hi] of ln D(s) = ln θ for a non-increasing D with D(hi) ≤ θ, D from `norm` and D at the bracket's ends
/// given (lo itself where D(lo) ≤ θ already): by the inverse interpolation of ln D − ln θ through its last three points
/// (their parabola's root; their last two's line where three do not tell), kept inside the bracket (its middle where
/// both step out), until ln D is within its rounding of ln θ or the iteration no longer moves; and where `ROUNDS` of
/// them have not got there, by the bracket's halving until its ends are neighbours -- the root ends by its rounding or
/// by its bracket, never by a count.
///
/// `floor` is the rounding of ln D as it is computed -- twice the state's relative to its norm, and the norm's own sum --,
/// the two logarithms' beside it: within it s is the root as far as the norm tells, and the interpolation's next point,
/// taken without another evaluation, is the root to the norm's actual rounding over its rate. (Beyond it the points
/// wander in that rounding until the bracket closes on neighbouring numbers: some ten evaluations more.) The norm's rate
/// 2 Im⟨x|Ĥ|x⟩ is not asked: on a grid it is not the slope of the grid's own norm, and a Newton iteration with it
/// converged at the grid's error's pace, in twice the evaluations.
fn root(mut norm: impl FnMut(f64) -> f64, (lo, at_lo): (f64, f64), (hi, at_hi): (f64, f64), theta: f64, floor: f64) -> f64 {
    let target = theta.ln();
    let floor = floor + 2.0 * UNIT_ROUNDOFF * target.abs();
    let (a, b) = (at_lo.ln() - target, at_hi.ln() - target);
    if !(a > floor) {
        return lo;
    }
    if !(b < -floor) {
        return hi;
    }
    // The last three points (s, ln D − ln θ), the latest first, and their interpolation's root where it lies inside.
    let interpolated = |[(x1, y1), (x2, y2), (x3, y3)]: [(f64, f64); 3], lo: f64, hi: f64| {
        let parabola = x1 * y2 * y3 / ((y1 - y2) * (y1 - y3)) + x2 * y1 * y3 / ((y2 - y1) * (y2 - y3)) + x3 * y1 * y2 / ((y3 - y1) * (y3 - y2));
        let line = x1 - y1 * (x1 - x2) / (y1 - y2);
        [parabola, line].into_iter().find(|&s| s > lo && s < hi)
    };
    let (mut lo, mut hi) = (lo, hi);
    let mut points = [(hi, b), (lo, a), (f64::NAN, f64::NAN)];
    let mut s = interpolated(points, lo, hi).unwrap_or(0.5 * (lo + hi));
    for _ in 0..ROUNDS {
        let g = norm(s).ln() - target;
        if g > 0.0 {
            lo = s;
        } else {
            hi = s;
        }
        points = [(s, g), points[0], points[1]];
        let refined = interpolated(points, lo, hi);
        if g.abs() <= floor {
            return refined.unwrap_or(s);
        }
        let next = refined.unwrap_or(0.5 * (lo + hi));
        if next == s || !(hi > lo) {
            return s;
        }
        s = next;
    }
    loop {
        let middle = 0.5 * (lo + hi);
        if !(middle > lo && middle < hi) {
            return hi;
        }
        if norm(middle).ln() - target > 0.0 {
            lo = middle;
        } else {
            hi = middle;
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// The step.

/// One step of τ from a trajectory's state (`step`).
#[derive(Clone, Debug)]
pub struct Step {
    /// The state at the step's end: the propagation runs to the end whatever the threshold; jumps, the loss and the
    /// trajectory's counters are the caller's.
    pub end: Trajectory,
    /// The first time (µs from the step's start) at which ‖ψ‖² reaches the threshold within the step, if it does.
    pub crossing: Option<f64>,
    /// The step's path.
    pub segment: Segment,
    /// The step's counted work (real multiply-adds: its own and its nodes' generators and forces) and the middles it
    /// asked of the model.
    pub work: f64,
    pub assemblies: u64,
    /// The step's propagation to be redone on twice its grid's panels (`Step::regrid`); none through the
    /// eigendecomposition.
    retry: Option<Retry>,
}

/// What a propagation on the grid needs to be redone on a finer one: the path's momentum and velocity through the step's
/// middle, the start state (the cluster's states first), its panels, and its impulse.
#[derive(Clone, Debug)]
struct Retry {
    kappa: [f64; 3],
    velocity: [f64; 3],
    psi: [C64; STATES],
    panels: usize,
    impulse: [f64; 3],
}

impl Step {
    /// The step's end with its propagation redone on twice its grid's panels (the same middle of the same model, the same
    /// path, the same kick otherwise), and that work: the grid's error read, the step's grid erring by 2⁴/(2⁴ − 1) of the
    /// difference (fourth order). None where the step took no grid.
    pub fn regrid<D: Dynamics>(&self, model: &D) -> Option<(Trajectory, f64)> {
        let r = self.retry.as_ref()?;
        let (n, s) = (self.end.psi.len(), &self.segment);
        let middle = model.middle(self.end.cluster, s.kicked.0, s.middle());
        let finer = beating(&Along::new(&middle, r.kappa, r.velocity, s.tau), &mut Scratch::new(n), &r.psi[..n], s.tau, 2 * r.panels, None);
        let kicked = (s.kicked.0, std::array::from_fn(|a| s.kicked.1[a] + finer.impulse[a] - r.impulse[a]));
        let (r1, kappa1) = s.flow.advance(kicked, 0.5 * s.tau);
        let mut end = self.end.clone();
        end.r = r1;
        end.kappa = kappa1;
        end.psi.copy_from_slice(&finer.end[..n]);
        Some((end, finer.work))
    }
}

/// Why a step or a jump could not be made: the reference's fixed point is not a contraction at this step (`middle`); the
/// state left the floating-point range; or the norm reached its threshold with no photon to emit (`jump`: the model's
/// decay and its photons contradict each other). The step control halves τ at the first two (its more accurate level).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Refused {
    Reference,
    Overflow,
    Emission,
}

/// One propagation of B on the path through a middle at the velocity v, the generator at the momentum κ (module
/// documentation, # The kick), by its route (`route`), and its kick: the propagation's impulse and the reference's
/// τ∇V_ref(r_h).
#[allow(clippy::too_many_arguments)]
fn propagate<M: Middle>(scratch: &mut Scratch, middle: &M, from: &Trajectory, reference: [f64; 3], kappa: [f64; 3], velocity: [f64; 3], tau: f64, threshold: Option<f64>) -> (Propagated, [f64; 3]) {
    let along = Along::new(middle, kappa, velocity, tau);
    let Carved { matrix, work: sums, .. } = scratch.carve(0);
    let propagated = match route(&along, matrix, sums, tau) {
        Some(spectrum) => spectral(&along, scratch, &spectrum, &from.psi, tau, threshold),
        None => beating(&along, scratch, &from.psi, tau, panels(tau, from.grid), threshold),
    };
    let kick = std::array::from_fn(|a| propagated.impulse[a] + tau * reference[a]);
    (propagated, kick)
}

/// One step of τ (negative: the reverse step) from a trajectory's state (module documentation): the middle and its
/// reference (`middle`); the model's generator there (`Dynamics::middle`); the predictor's momentum from the shear at κ_h
/// and the reference's force, the first propagation on its path; the corrected momentum κ_h + Δκ/2 and the step's
/// propagation at it and on its path; the kick and the second half flow.
pub fn step<D: Dynamics>(model: &D, from: &Trajectory, tau: f64) -> Result<Step, Refused> {
    let half = 0.5 * tau;
    let hm = model.hbar_over_mass();
    let n = from.psi.len();
    let (flow, (r_h, kappa_h)) = middle(model, from.r, from.kappa, half)?;
    let reference = flow.gradient_at(r_h);
    let mut scratch = Scratch::new(n);
    let generator = model.middle(from.cluster, r_h, from.time + half);
    // The predictor: the shear and the reference's force.
    let held = shear(&Along::new(&generator, kappa_h, [0.0; 3], tau), scratch.carve(0).held, &from.psi, tau);
    let predicted: [f64; 3] = std::array::from_fn(|a| kappa_h[a] + 0.5 * (held[a] + tau * reference[a]));
    let (one, kick_one) = propagate(&mut scratch, &generator, from, reference, kappa_h, predicted.map(|k| hm * k), tau, None);
    // The correction: the middle momentum of the first propagation's kick.
    let corrected: [f64; 3] = std::array::from_fn(|a| kappa_h[a] + 0.5 * kick_one[a]);
    let velocity = corrected.map(|k| hm * k);
    let (two, kick) = propagate(&mut scratch, &generator, from, reference, corrected, velocity, tau, Some(from.threshold));
    let kicked = (r_h, std::array::from_fn(|a| kappa_h[a] + kick[a]));
    let (r1, kappa1) = flow.advance(kicked, half);
    if !(r1.iter().chain(&kappa1).all(|x| x.is_finite()) && two.end[..n].iter().all(|z| z.is_finite())) {
        return Err(Refused::Overflow);
    }
    let mut end = from.clone();
    end.r = r1;
    end.kappa = kappa1;
    end.psi.copy_from_slice(&two.end[..n]);
    end.time = from.time + tau;
    let work = one.work + two.work + count::shear(n, generator.work());
    let retry = two.panels.map(|panels| {
        let mut psi = [C64::ZERO; STATES];
        psi[..n].copy_from_slice(&from.psi);
        Retry { kappa: corrected, velocity, psi, panels, impulse: two.impulse }
    });
    Ok(Step { end, crossing: two.crossing, segment: Segment { start: from.time, tau, flow, from: (from.r, from.kappa), kicked }, work, assemblies: 1, retry })
}

/// The counted work of one step of τ on the path through a middle at a momentum and a velocity, and of that step's grid
/// doubling (zero without a grid): what `Step::work` and `Step::regrid` count, priced without taking the step -- the
/// solve's price of a model it has not run, at its pilot's states and steps. It makes no middle: the caller's is a
/// state's own, or one of the cluster made once -- the nodes' counts are the cluster's whatever the point
/// (`Middle::work`), and the point gives the route alone. Both propagations by the generator's own route (`route`: the
/// eigen-propagation's decomposition and Gauss–Legendre nodes, or the grid's CF4 spacings no wider than `grid`, each
/// exponential's Chebyshev plan at the bounds of Ĥ's traceless part at the middle over half a spacing, its two nodes'
/// weights' sum) and the shear; the middle itself is the model's (`Dynamics::assembly_work`, one a step).
pub fn step_work<M: Middle>(middle: &M, kappa: [f64; 3], velocity: [f64; 3], tau: f64, grid: f64) -> (f64, f64) {
    let (n, costs) = (middle.states(), middle.work());
    let along = Along::new(middle, kappa, velocity, tau);
    let (mut a, mut sums) = (Mat::zeros(n, n), vec![C64::ZERO; n]);
    let (propagation, regrid) = match route(&along, &mut a, &mut sums, tau) {
        Some(spectrum) => (count::spectral(n, costs, spectrum.eigen.work, spectrum.nodes) + count::impulse(n, costs, spectrum.nodes, true), 0.0),
        None => {
            middle.generator(&along.path, 0.0, &mut a);
            traceless(&mut a);
            let (rho, gamma) = linalg::bounds(&a, &mut sums);
            let grid_work = |panels: usize| {
                let h = (tau / (2 * panels) as f64).abs();
                let plan = linalg::Chebyshev::plan(0.5 * rho * h, 0.5 * gamma * h);
                (2 * panels) as f64 * count::cf4(n, costs, 2 * plan.parts * plan.terms) + count::impulse(n, costs, 2 * panels, false)
            };
            let panels = panels(tau, grid);
            (grid_work(panels), grid_work(2 * panels))
        }
    };
    (2.0 * propagation + count::shear(n, costs), regrid)
}

// ---------------------------------------------------------------------------------------------------------------------
// The jump.

/// A zero-mean kick of covariance C (positive semidefinite up to its rounding): C's Cholesky columns, each with its own
/// sign, E[L s sᵀ Lᵀ] = L Lᵀ = C; a pivot not above zero (C's null directions and their rounding) leaves its column out.
fn fluctuation(c: &[[f64; 3]; 3], signs: [f64; 3]) -> [f64; 3] {
    let mut l = [[0.0; 3]; 3];
    for j in 0..3 {
        let pivot = c[j][j] - (0..j).map(|k| l[j][k] * l[j][k]).sum::<f64>();
        if !(pivot > 0.0) {
            continue;
        }
        let root = pivot.sqrt();
        l[j][j] = root;
        for i in j + 1..3 {
            l[i][j] = (c[i][j] - (0..j).map(|k| l[i][k] * l[j][k]).sum::<f64>()) / root;
        }
    }
    std::array::from_fn(|i| (0..3).map(|j| l[i][j] * signs[j]).sum())
}

/// The jump of a trajectory at its state (SPEC §1.6): from `Stream::jump(seed, index, jumps + 1)` the next threshold, the
/// class ∝ Σ_a‖u_a‖², the polarization across the direction by ‖Σ_a ε_a u_a‖², the fluctuation's signs, and the direction
/// from the dipole pattern tr G − n·Re(G)·n (at most tr G, as Re G ≥ 0) by exact rejection from the uniform sphere; then
/// ψ ← f/‖f‖ on the landing cluster and κ += −k_φ n + b_J + ξ, Cov ξ = M − b_J b_Jᵀ (`Dynamics::recoil`); then the loss
/// test (`test_loss`). A norm that reached its threshold with no photon to emit -- every class's weight zero, or not a
/// number -- is refused (`Refused::Emission`) with the trajectory as it was: the model's decay and its photons
/// contradict each other, and no jump is made of that.
///
/// `carried` is a second internal state of the trajectory's cluster taken through the same jump (adapt.rs, # The read):
/// the picked class's operator into the picked polarization on it, renormalized as ψ is -- the jump is a linear map of
/// the internal state. One the operator annihilates is dropped (None).
pub fn jump<D: Dynamics>(model: &D, trajectory: &mut Trajectory, carried: &mut Option<Vec<C64>>) -> Result<(), Refused> {
    let j = trajectory.jumps + 1;
    let mut stream = Stream::jump(trajectory.seed, trajectory.index, j);
    let threshold = stream.open_uniform();
    let (class, polarization) = (stream.uniform(), stream.uniform());
    let signs: [f64; 3] = std::array::from_fn(|_| if stream.uniform() < 0.5 { 1.0 } else { -1.0 });
    let at = trajectory.point();
    let photons = model.photons(trajectory.cluster, &at, &trajectory.psi);
    let weights: Vec<f64> = photons.iter().map(|p| p.amplitudes.iter().map(|u| norm_sqr(u)).sum()).collect();
    let total: f64 = weights.iter().sum();
    if !(total > 0.0) {
        return Err(Refused::Emission);
    }
    trajectory.jumps = j;
    trajectory.threshold = threshold;
    trajectory.work += model.photon_work();
    let mut x = class * total;
    let mut pick = weights.len() - 1;
    for (i, &w) in weights.iter().enumerate() {
        if x < w {
            pick = i;
            break;
        }
        x -= w;
    }
    let photon = &photons[pick];
    let u = &photon.amplitudes;
    let gram: [[f64; 3]; 3] = std::array::from_fn(|a| std::array::from_fn(|b| inner(&u[a], &u[b]).re));
    let trace = gram[0][0] + gram[1][1] + gram[2][2];
    let direction = loop {
        let n = stream.direction();
        let q: f64 = (0..3).map(|a| (0..3).map(|b| n[a] * gram[a][b] * n[b]).sum::<f64>()).sum();
        if stream.uniform() * trace <= trace - q {
            break n;
        }
    };
    let (e1, e2) = linalg::perpendicular(direction);
    let along = |e: [f64; 3]| -> Vec<C64> { (0..u[0].len()).map(|i| (0..3).map(|a| u[a][i] * e[a]).sum()).collect() };
    let (f1, f2) = (along(e1), along(e2));
    let (w1, w2) = (norm_sqr(&f1), norm_sqr(&f2));
    let (epsilon, f, w) = if polarization * (w1 + w2) < w1 { (e1, f1, w1) } else { (e2, f2, w2) };
    let recoil = model.recoil(photon, &epsilon);
    let b = recoil.mean;
    let spread: [[f64; 3]; 3] = std::array::from_fn(|i| std::array::from_fn(|k| recoil.second[i][k] - b[i] * b[k]));
    let xi = fluctuation(&spread, signs);
    for a in 0..3 {
        trajectory.kappa[a] += -photon.wave_number * direction[a] + b[a] + xi[a];
    }
    if let Some(other) = carried.take() {
        *carried = model.photons(trajectory.cluster, &at, &other).iter().find(|p| p.class == photon.class).and_then(|p| {
            let g: Vec<C64> = (0..p.amplitudes[0].len()).map(|i| (0..3).map(|a| p.amplitudes[a][i] * epsilon[a]).sum()).collect();
            let size = norm_sqr(&g);
            (size > 0.0).then(|| g.iter().map(|z| *z * (1.0 / size.sqrt())).collect())
        });
    }
    let scale = 1.0 / w.sqrt();
    trajectory.psi = f.iter().map(|z| *z * scale).collect();
    trajectory.cluster = photon.landing;
    test_loss(model, trajectory);
    Ok(())
}

/// The loss test at a trajectory's state (SPEC §1.7: at every step's end and after every jump), the one definition: its
/// time once the model finds it lost (`Dynamics::lost`).
pub fn test_loss<D: Dynamics>(model: &D, trajectory: &mut Trajectory) {
    if trajectory.lost.is_none() && model.lost(trajectory.cluster, &trajectory.point(), &trajectory.psi) {
        trajectory.lost = Some(trajectory.time);
    }
}

#[cfg(test)]
mod tests;
