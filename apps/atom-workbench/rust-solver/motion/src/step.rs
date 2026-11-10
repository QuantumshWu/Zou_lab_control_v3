//! One step: A(τ/2) B(τ) A(τ/2) with the mid-step reference, the internal propagation along the path, the impulse, the
//! jump's root and the jump (SPEC §2.1, §1.6).
//!
//! # The motion and its splitting
//! A trajectory is (r, κ, ψ): position (µm), momentum κ = p/ħ (1/µm) and the cluster's unnormalized internal state between
//! jumps: ṙ = (ħ/m)κ, κ̇ = −⟨ψ|G|ψ⟩/⟨ψ|ψ⟩, iψ̇ = Hψ, H the cluster's generator and G = ∇H_h (`Dynamics::generator`).
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
//! exact: with ψ_s = e^{iK_s·v(t − t_m)} x_s in the cluster's Doppler frame (`Cluster::waves`) x evolves under
//! Ĥ(t) = Σ_j M_j e^{iω_j(t − t_m)} + diag(K_s·v), ω_j = Ω_j + q_j·v (`Path`), static where every pair of states holds one
//! wave vector. A static Ĥ is propagated through its eigendecomposition at any time exactly (`spectral`), a beating one by
//! CF4 on a grid (`beating`). The impulse is exact in its decomposition about the state at the step's middle, held in the
//! frame, x_m = x(t_m) (= ψ(t_m), U(t_m) = 1):
//! - the shear: the conservative force at the fixed r_h on x_m, −∫⟨x_m|G(r_h, t)|x_m⟩dt/‖x_m‖², its parts' beats at Ω_j
//!   alone (r is fixed): τ sinc(Ω_jτ/2) each (`shear`). Of a state that does not change it depends on r_h alone: the
//!   conservative kick is a shear and the step symplectic (propagation.md §2.3);
//! - what the state's change along the path adds: the evolved state's force along the path less x_m's held in the frame,
//!   −∫(⟨x(t)|Ĝ(t)|x(t)⟩/‖x(t)‖² − ⟨x_m|Ĝ(t)|x_m⟩/‖x_m‖²)dt, Ĝ(t) = Σ_j G_j e^{iω_j(t − t_m)}, one quadrature of the
//!   difference on the propagation's nodes (B1's `force_change`): it vanishes at every node, and with it its
//!   quadrature's error, where the state does not change -- a lone ground state in a standing wave's far light takes no
//!   grid error, however coarse its grid. The lag that friction and the Sisyphus effect live in;
//! - less the reference's force: + τ∇V_ref(r_h), since V_ref's force is the A flow's.
//!
//! The first two are read together (`impulse`): a mean ⟨x|G|x⟩/‖x‖² is tr(Gρ) of the state's density ρ, so with ρ_m the
//! middle state's and (s_k, w_k, ρ_k) the nodes' times from the middle, weights and densities they are
//! −Σ_j Re tr(G_j Q_j), Q_j = τ sinc(Ω_jτ/2) ρ_m + Σ_k w_k e^{iω_j s_k}(ρ_k − ρ_m): the forms of the generator's parts are
//! made once a propagation, on the sums over its nodes. Ĝ(t) is Hermitian, its parts paired (`model::Part`), and so are
//! the Q_j: the traces are read on the entries on and above the diagonal (`contraction`).
//!
//! A generator's scalar part, tr/n -- the trap's light shift and the decay every state shares --, is a phase and a
//! factor on the whole state: the propagations take it out exactly (`traceless`, `linalg::exp_apply`), so that their
//! work and their rounding go with the states' differences alone.
//!
//! Held in the frame, not in the lab: a dressed state's coherences co-rotate with the light's phases along the path, so
//! its force along the path is the frame's, constant where Ĥ is static (a running wave's radiation pressure, steady on the
//! moving atom); held in the lab its coherence would slip against the light at K·v and its path force average away
//! (sinc(k·vτ/2)), counting the steady force twice -- a running wave's pressure came out doubled. Shear and path then differ
//! only by the frame's beats, the light pattern's spatial modulation the atom crosses, which is what the shear holds at r_h.
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
//! O(τ²) and the kick's error O(τ³), the scheme's own local order: p − 1 = 1 correction (`step`). The generator is assembled
//! at the predictor's momentum κ_h for the shear and the first propagation, and at the corrected one for the step: two
//! assemblies, two propagations.
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
use crate::model::{Cluster, Generator, Part, Photon, Point, Quadratic, Recoil, STATES};
use crate::random::Stream;
use crate::units::{rounding, UNIT_ROUNDOFF};

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
    /// path H(t) = Σ_j M_j e^{iω_j(t − at.t)}, ω_j = Ω_j + q_j·v, in the cluster's Doppler frame. The static part is the one
    /// whose beat and wave vector are both exactly zero.
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

    /// The counted work of one generator assembly (`decide::Work`, real multiply-adds), made once per run: the step control
    /// weighs the split's error against the beating grid's by it (`adapt`).
    fn assembly_work(&self) -> f64;

    /// The counted real multiply-adds of one jump's photons (`Dynamics::photons`), made once per run: `jump` adds it to the
    /// trajectory's work.
    fn photon_work(&self) -> f64;
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
    /// The step control's state (`adapt`): the next step's proposed length (µs); the beating grid's largest spacing (µs,
    /// `beating`); the steps made (every `step`, the doubling's three and the refused ones); the time from which the step
    /// cap left the steps unchecked (SPEC §2.2: that span is flagged).
    pub tau: f64,
    pub grid: f64,
    pub steps: u64,
    pub capped: Option<f64>,
    /// The work counted (`decide::Work`, real multiply-adds) of the integrator's own algorithms over the trajectory --
    /// exponentials, eigen-propagation, CF4 nodes, the doubling and the grid's doubling (`adapt` adds each step's) --, the
    /// model's generator assemblies it asked for (each costed by `Dynamics::assembly_work`) and its jumps' photons
    /// (`Dynamics::photon_work`).
    pub work: f64,
    pub assemblies: u64,
}

impl Trajectory {
    /// A trajectory at a basis state (a ground state of the model basis) at (r, κ) and time 0: ψ the unit vector of that
    /// state on its cluster, the first threshold from jump 0's stream (SPEC §2.4). Its first step is proposed as long as
    /// its first output interval allows and its beating grid no finer than its step's quarter (`adapt` refines both from
    /// there).
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
    pub fn advance(&self, (r, kappa): ([f64; 3], [f64; 3]), t: f64) -> ([f64; 3], [f64; 3]) {
        let hm = self.hbar_over_mass;
        let d = [r[0] - self.centre[0], r[1] - self.centre[1], r[2] - self.centre[2]];
        let (mut r1, mut k1) = (self.centre, [0.0; 3]);
        for i in 0..3 {
            let e = self.axes[i];
            let (y, k, lambda, g) = (linalg::dot3(e, d), linalg::dot3(e, kappa), self.curvatures[i], self.gradient[i]);
            let (c0, c1, c2) = stumpff(hm * lambda * t * t);
            let y1 = y * c0 + hm * k * t * c1 - hm * g * t * t * c2;
            let k_1 = k * c0 - (lambda * y + g) * t * c1;
            for a in 0..3 {
                r1[a] += y1 * e[a];
                k1[a] += k_1 * e[a];
            }
        }
        (r1, k1)
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

/// Rounds of a fixed-point iteration that halves its distance each round at least: 2⁻⁶⁴ is below the unit round-off, so
/// after them the distance is the round-off's (derived; a map that contracts less within them refuses the step).
const ROUNDS: usize = 64;

/// The operations between a flow's inputs and a coordinate of its reach (`Flow::advance`): the coordinate adds three
/// axes' displacements to the centre (6), each the Stumpff combination of three terms (10) of the axis's coordinate (a
/// difference and a product of three: 6) and of its x (3) through its functions (3): 28. `units::rounding` of them
/// times the positions' size is the rounding of a reach.
const REACH: usize = 28;

/// The step's middle (module documentation, # The reference): the fixed point p of "the first half flow on V_ref(p)
/// reaches p" from (r, κ) over `half`, from the free flight's middle; the flow and (r_h, κ_h). A round moves p by the
/// reference's change over the move times the flow's response, (ħ/m)|∇K|·|r − p|·half²/3 of the last move: a contraction
/// while the step resolves the potential's third derivative, by a factor of 10⁻² to 10⁻⁴ a round at the steps a trap
/// takes (so that an accelerated iteration -- Broyden's, Anderson's -- gains less than one round of some six).
///
/// The step's results depend on the centre p only through that third derivative: taken at p with the reach (r_h, κ_h)
/// they differ from the fixed point's by the next round's moves of the reach's position and momentum, the last moves'
/// contraction. So the rounds end where those next moves, the last ones times the largest contraction measured, are
/// within the rounding of a reach (`REACH`) -- the centre itself then lies a move, not a rounding, from its reach, which
/// no result sees --; or where the move is zero or stops shrinking, the round-off's floor that a contraction ends at.
fn middle<D: Dynamics>(model: &D, r: [f64; 3], kappa: [f64; 3], half: f64) -> Result<(Flow, ([f64; 3], [f64; 3])), Refused> {
    let hm = model.hbar_over_mass();
    let largest = |a: [f64; 3], b: [f64; 3]| (0..3).map(|i| a[i].abs().max(b[i].abs())).fold(0.0, f64::max);
    let apart = |a: [f64; 3], b: [f64; 3]| (0..3).map(|i| (a[i] - b[i]).abs()).fold(0.0, f64::max);
    let mut p: [f64; 3] = std::array::from_fn(|a| r[a] + hm * kappa[a] * half);
    // The last move of the reach's position, the momentum it reached then, and the largest contraction measured (the
    // first round measures none: its move over no move before it is zero).
    let (mut last, mut before, mut contraction) = (f64::INFINITY, kappa, 0.0_f64);
    for _ in 0..ROUNDS {
        let flow = Flow::new(&model.reference(p), p, hm);
        let reached = flow.advance((r, kappa), half);
        let moved = apart(reached.0, p);
        if !moved.is_finite() || !reached.1.iter().all(|k| k.is_finite()) {
            return Err(Refused::Overflow);
        }
        contraction = contraction.max(moved / last);
        let next = [(moved, largest(r, p)), (apart(reached.1, before), largest(kappa, reached.1))];
        if moved == 0.0 || moved >= last || (last.is_finite() && next.iter().all(|(move_, size)| contraction * move_ <= rounding(REACH) * size)) {
            return Ok((flow, reached));
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
    /// (r, κ) at the time t within the step: the flow from the start before the middle, from the kicked middle after it.
    pub fn at(&self, t: f64) -> ([f64; 3], [f64; 3]) {
        let m = self.middle();
        if (t - self.start).abs() < (m - self.start).abs() {
            self.flow.advance(self.from, t - self.start)
        } else {
            self.flow.advance(self.kicked, t - m)
        }
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

/// The cluster's generator along the step's path in its Doppler frame (module documentation, # The kick):
/// Ĥ(t) = Σ_j M_j e^{iω_j(t − t_m)} + diag(K_s·v) over the generator's parts, ω_j = Ω_j + q_j·v -- zero, exactly, of the
/// static part --, the middle t_m, and per state the gauge's rate K_s·v and its phase over half the step: the frame's
/// state at the step's start is x₀ = e^{iK_s·v τ/2} ψ₀, and at its end ψ = e^{iK_s·v τ/2} x.
struct Path<'a> {
    parts: &'a [Part],
    velocity: [f64; 3],
    middle: f64,
    gauge: [f64; STATES],
    half: [C64; STATES],
    n: usize,
}

/// Whether a part is the static one: its beat and its wave vector exactly zero (`Dynamics::generator`).
fn is_static(part: &Part) -> bool {
    part.beat.0.iter().all(|&k| k == 0) && part.wave == [0.0; 3]
}

/// The gauge's rates K_s·v of a cluster's Doppler frames at a path velocity.
fn gauge(waves: &[[f64; 3]], velocity: [f64; 3]) -> [f64; STATES] {
    let mut rates = [0.0; STATES];
    for (rate, k) in rates.iter_mut().zip(waves) {
        *rate = linalg::dot3(*k, velocity);
    }
    rates
}

impl<'a> Path<'a> {
    fn new(parts: &'a [Part], n: usize, gauge: [f64; STATES], velocity: [f64; 3], middle: f64, tau: f64) -> Path<'a> {
        let mut half = [C64::ZERO; STATES];
        for s in 0..n {
            half[s] = C64::cis(0.5 * gauge[s] * tau);
        }
        Path { parts, velocity, middle, gauge, half, n }
    }
    /// ω_j of a part.
    fn omega(&self, part: &Part) -> f64 {
        part.omega + linalg::dot3(part.wave, self.velocity)
    }
    /// Whether Ĥ is static along the path: every part is.
    fn is_static(&self) -> bool {
        self.parts.iter().all(is_static)
    }
    /// a ← Σ_j c_j M_j + d diag(K_s·v): a combination of Ĥ's values over times, c_j its weights' sum on the j-th part's
    /// phases and d on the static ones.
    fn combination(&self, a: &mut Mat, coefficient: impl Fn(usize) -> C64, diagonal: f64) {
        a.data.fill(C64::ZERO);
        for (j, part) in self.parts.iter().enumerate() {
            a.add_scaled(coefficient(j), &part.generator);
        }
        for s in 0..self.n {
            a[(s, s)] += C64::real(diagonal * self.gauge[s]);
        }
    }
}

/// The panels of two spacings over τ with spacings no wider than h (`beating`): ⌈|τ|/2h⌉, at least one; a quotient within
/// its rounding of a whole number is that number (the step control sets h to divide its steps, `adapt`).
pub fn panels(tau: f64, spacing: f64) -> usize {
    let x = tau.abs() / (2.0 * spacing);
    let whole = x.round();
    let panels = if (x - whole).abs() <= rounding(4) * x { whole } else { x.ceil() };
    (panels as usize).max(1)
}

/// The entries on and above the diagonal of a Hermitian matrix on n states.
fn upper(n: usize) -> usize {
    n * (n + 1) / 2
}

/// A state's density matrix ρ = |x⟩⟨x|/⟨x|x⟩ as its entries ρ_cb = x_b* x_c/‖x‖² of b ≤ c, row by row: a mean
/// ⟨x|G|x⟩/⟨x|x⟩ = Σ_bc G_bc ρ_cb is their `contraction` with G.
fn density(x: &[C64], rho: &mut [C64]) {
    let scale = 1.0 / norm_sqr(x);
    let mut k = 0;
    for (b, left) in x.iter().enumerate() {
        let left = left.conj() * scale;
        for right in &x[b..] {
            rho[k] = left * *right;
            k += 1;
        }
    }
}

/// Re Σ_bc G_bc Q_cb = Σ_b Re(G_bb Q_bb) + 2 Σ_{b<c} Re(G_bc Q_cb) of Hermitian G and Q, Q as `density` lays it out. A
/// generator's parts pair, G_{−j} = G_j† (`model::Part`: G(t) is Hermitian at every time), and so do the Q_j they are
/// contracted with here (weights e^{iω_j s} and their conjugates on Hermitian densities): each part read on and above the
/// diagonal, the parts' sum is Σ_j Re tr(G_j Q_j).
fn contraction(g: &Mat, q: &[C64]) -> f64 {
    let n = g.rows;
    let (mut k, mut sum) = (0, 0.0);
    for b in 0..n {
        let row = &g.data[b * n + b..(b + 1) * n];
        sum += (row[0] * q[k]).re;
        for (x, y) in row[1..].iter().zip(&q[k + 1..]) {
            sum += 2.0 * (*x * *y).re;
        }
        k += n - b;
    }
    sum
}

/// The shear (module documentation, # The kick): −∫⟨ψ|G(r_h, t)|ψ⟩dt/‖ψ‖² over the step of a state held, at the fixed r_h
/// each part at its beat Ω_j alone, ∫e^{iΩ_j(t − t_m)}dt = τ sinc(Ω_jτ/2). `rho` takes the state's density.
fn shear(generator: &Generator, rho: &mut [C64], psi: &[C64], tau: f64) -> [f64; 3] {
    density(psi, rho);
    let mut f = [0.0; 3];
    for part in &generator.parts {
        let w = tau * sinc(0.5 * part.omega * tau);
        for a in 0..3 {
            f[a] -= w * contraction(&part.force[a], rho);
        }
    }
    f
}

/// A step's scratch memory, made once and fitted to each use: the matrix of an exponential (or the static Ĥ₀), and one
/// array carved (`Scratch::carve`) into a grid's node states, the middle state's density, a node's change of density,
/// every part's sum of them (`impulse`), the parts' phases at a CF4 sub-step's two times, and an exponential's two work
/// vectors (`linalg::exp_apply`).
struct Scratch {
    matrix: Mat,
    rest: Vec<C64>,
}

struct Carved<'a> {
    matrix: &'a mut Mat,
    states: &'a mut [C64],
    held: &'a mut [C64],
    change: &'a mut [C64],
    sums: &'a mut [C64],
    phases: &'a mut [C64],
    work: &'a mut [C64],
}

impl Scratch {
    fn new(n: usize) -> Scratch {
        Scratch { matrix: Mat::zeros(n, n), rest: Vec::new() }
    }
    /// The scratch for a generator of `parts` parts and a grid of `nodes` spacings.
    fn carve(&mut self, parts: usize, nodes: usize) -> Carved<'_> {
        let n = self.matrix.rows;
        let sizes = [(nodes + 1) * n, upper(n), upper(n), parts * upper(n), 2 * parts];
        self.rest.resize(sizes.iter().sum::<usize>() + 2 * n, C64::ZERO);
        let (states, rest) = self.rest.split_at_mut(sizes[0]);
        let (held, rest) = rest.split_at_mut(sizes[1]);
        let (change, rest) = rest.split_at_mut(sizes[2]);
        let (sums, rest) = rest.split_at_mut(sizes[3]);
        let (phases, work) = rest.split_at_mut(sizes[4]);
        Carved { matrix: &mut self.matrix, states, held, change, sums, phases, work }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// The propagations.

/// What a propagation of the frame's state over the step made: ψ at the step's end (lab; the cluster's states first),
/// the kick's part of the state (`impulse`), the first crossing of the threshold (from the step's start), the rounding
/// bound of ψ relative to its norm, the panels of its grid (none: through the eigendecomposition), and its counted work.
struct Propagated {
    end: [C64; STATES],
    impulse: [f64; 3],
    crossing: Option<f64>,
    rounding: f64,
    panels: Option<usize>,
    work: f64,
}

/// The general eigendecomposition's work in n³ complex multiply-adds (Golub & Van Loan §7.5: Hessenberg reduction, the
/// shifted QR to Schur form with its vectors, back substitution; R⁻¹ by LU), the count its algorithm takes.
const EIGEN_WORK: f64 = 25.0;

/// The counted work (`decide::Work`, real multiply-adds) of the integrator's operations on n states, one formula each:
/// what the propagations add up and what `step_work` prices a step by.
mod count {
    use super::{upper, COMPLEX, EIGEN_WORK};
    /// The eigen-propagation through `nodes` quadrature nodes: the decomposition, and per node n exponentials and the
    /// state R d of n².
    pub fn spectral(n: usize, nodes: usize) -> f64 {
        COMPLEX * (EIGEN_WORK * (n * n * n) as f64 + (nodes * (n + n * n)) as f64)
    }
    /// One exponential of CF4 on the vector: its plan's products and the combination of the generator's `parts`
    /// parts, n² each.
    pub fn exponential(n: usize, parts: usize, products: usize) -> f64 {
        COMPLEX * ((products + parts) * n * n) as f64
    }
    /// The kick's part of the state through `nodes` nodes (`impulse`), in entries on and above the diagonal: the middle
    /// state's density and each part's share of it; per node its change of density and each part's sum; each part's three
    /// contractions.
    pub fn impulse(n: usize, parts: usize, nodes: usize) -> f64 {
        COMPLEX * ((1 + parts + nodes * (1 + parts) + 3 * parts) * upper(n)) as f64
    }
    /// The predictor's shear: the start state's density and each part's three contractions.
    pub fn shear(n: usize, parts: usize) -> f64 {
        COMPLEX * ((1 + 3 * parts) * upper(n)) as f64
    }
}

/// The node count of a quadrature over τ at a bandwidth (`plan`).
fn nodes(tau: f64, bandwidth: f64) -> usize {
    let (rule, panels) = plan(tau.abs(), bandwidth);
    rule.nodes.len() * panels
}

/// The bandwidth of the forms on a state propagated through the eigenvalues λ: max|λ_b* − λ_c| (their frequencies and
/// decay rates).
fn spread(lambda: &[C64]) -> f64 {
    lambda.iter().flat_map(|a| lambda.iter().map(move |b| (a.conj() - *b).abs())).fold(0.0, f64::max)
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
/// change's force integrated on the propagation's nodes, together −Σ_j Re tr(G_j Q_j) (`contraction`) with
/// Q_j = τ sinc(Ω_jτ/2) ρ_m + Σ_k w_k e^{iω_j s_k} (ρ_k − ρ_m): ρ_m the middle state's density, held in the frame, and
/// (s_k, w_k, x_k) the nodes' times from the middle, their weights and their states, which `visit` gives in turn. Every
/// part's forms are so made once, on its sum over the nodes; the change vanishes at every node where the state does not.
fn impulse(path: &Path, held: &mut [C64], change: &mut [C64], sums: &mut [C64], tau: f64, middle: &[C64], visit: &mut dyn FnMut(&mut dyn FnMut(f64, f64, &[C64]))) -> [f64; 3] {
    let m = upper(path.n);
    density(middle, held);
    for (part, q) in path.parts.iter().zip(sums.chunks_exact_mut(m)) {
        let shear = tau * sinc(0.5 * part.omega * tau);
        for (q, rho) in q.iter_mut().zip(held.iter()) {
            *q = *rho * shear;
        }
    }
    visit(&mut |s, w, x| {
        density(x, change);
        for (d, rho) in change.iter_mut().zip(held.iter()) {
            *d -= *rho;
        }
        for (part, q) in path.parts.iter().zip(sums.chunks_exact_mut(m)) {
            let z = C64::cis(path.omega(part) * s) * w;
            for (q, d) in q.iter_mut().zip(change.iter()) {
                *q += z * *d;
            }
        }
    });
    let mut out = [0.0; 3];
    for (part, q) in path.parts.iter().zip(sums.chunks_exact(m)) {
        for a in 0..3 {
            out[a] -= contraction(&part.force[a], q);
        }
    }
    out
}

/// The propagation of a static Ĥ₀ through its eigendecomposition Ĥ₀ = c + RΛR⁻¹ (its scalar part c apart, `traceless`;
/// exact at any time, its rounding κ(R)·u grown by the decay rates' spread over the step, `linalg::Eigen`): in the
/// coordinates d = R⁻¹x, d(s) = e^{−i(c + Λ)s}d₀ and x(s) = R d(s), the impulse by Gauss–Legendre to round-off at the
/// bandwidth max|λ_b* − λ_c| (`quadrature`), the norm's root on ‖x(s)‖² (`root`). None where the decomposition
/// fails.
fn spectral(path: &Path, scratch: &mut Scratch, psi: &[C64], tau: f64, threshold: Option<f64>) -> Option<Propagated> {
    let n = path.n;
    let Carved { matrix, held, change, sums, .. } = scratch.carve(path.parts.len(), 0);
    path.combination(matrix, |_| C64::ONE, 1.0);
    let scalar = traceless(matrix);
    let e = linalg::eigen(matrix)?;
    let (mut lambda, mut d0) = ([C64::ZERO; STATES], [C64::ZERO; STATES]);
    for b in 0..n {
        lambda[b] = e.values[b] + scalar;
        d0[b] = (0..n).map(|c| e.inverse[(b, c)] * (psi[c] * path.half[c])).sum();
    }
    let lambda = &lambda[..n];
    let at = |s: f64, x: &mut [C64]| {
        let mut d = [C64::ZERO; STATES];
        for b in 0..n {
            d[b] = d0[b] * (C64::new(0.0, -s) * lambda[b]).exp();
        }
        for (a, x) in x.iter_mut().enumerate() {
            *x = e.vectors.data[a * n..(a + 1) * n].iter().zip(&d).map(|(r, d)| *r * *d).sum();
        }
    };
    let bandwidth = spread(lambda);
    let count = nodes(tau, bandwidth);
    let growth = lambda.iter().map(|l| l.im).fold(f64::NEG_INFINITY, f64::max) - lambda.iter().map(|l| l.im).fold(f64::INFINITY, f64::min);
    let (mut middle, mut end, mut x) = ([C64::ZERO; STATES], [C64::ZERO; STATES], [C64::ZERO; STATES]);
    at(0.5 * tau, &mut middle[..n]);
    let impulse = impulse(path, held, change, sums, tau, &middle[..n], &mut |each: &mut dyn FnMut(f64, f64, &[C64])| {
        for (s, w) in quadrature(0.0, tau, bandwidth) {
            at(s, &mut x[..n]);
            each(s - 0.5 * tau, w, &x[..n]);
        }
    });
    at(tau, &mut end[..n]);
    let bound = e.condition * UNIT_ROUNDOFF * (1.0 + tau.abs() * growth);
    let crossing = threshold.filter(|&theta| tau > 0.0 && norm_sqr(&end[..n]) <= theta).map(|theta| {
        let norm = |s: f64| {
            let mut x = [C64::ZERO; STATES];
            at(s, &mut x[..n]);
            norm_sqr(&x[..n])
        };
        root(norm, (0.0, norm_sqr(psi)), (tau, norm_sqr(&end[..n])), theta, 2.0 * bound + rounding(n + 1))
    });
    for (z, phase) in end[..n].iter_mut().zip(&path.half) {
        *z *= *phase;
    }
    Some(Propagated {
        end,
        impulse,
        crossing,
        rounding: bound,
        panels: None,
        work: count::spectral(n, count) + count::impulse(n, path.parts.len(), count),
    })
}

/// CF4, the fourth-order commutator-free Magnus scheme of two exponentials (Blanes & Moan 2006): over h from t,
/// x ← e^{−ih(w₂Ĥ₁ + w₁Ĥ₂)} e^{−ih(w₁Ĥ₁ + w₂Ĥ₂)} x, Ĥ_i = Ĥ(t + c_i h) at the Gauss nodes c = ½ ∓ √3/6, w = ¼ ± √3/6;
/// exact where Ĥ is static (w₁ + w₂ = ½ each). Each exponential on the vector to round-off (`linalg::exp_apply`, which
/// takes the combination's scalar part out and overwrites `a`; `vectors` its work); the parts' phases at the two
/// nodes are made once, into `phases`. Its counted work and its rounding of x relative to its norm
/// (`linalg::Chebyshev::rounding`).
fn cf4(path: &Path, a: &mut Mat, phases: &mut [C64], vectors: &mut [C64], x: &mut [C64], t: f64, h: f64) -> (f64, f64) {
    let r = 3.0_f64.sqrt() / 6.0;
    let times = [t + (0.5 - r) * h - path.middle, t + (0.5 + r) * h - path.middle];
    for (part, phases) in path.parts.iter().zip(phases.chunks_exact_mut(2)) {
        let omega = path.omega(part);
        phases[0] = C64::cis(omega * times[0]);
        phases[1] = C64::cis(omega * times[1]);
    }
    let (mut work, mut bound) = (0.0, 0.0);
    for [w1, w2] in [[0.25 + r, 0.25 - r], [0.25 - r, 0.25 + r]] {
        path.combination(a, |j| phases[2 * j] * w1 + phases[2 * j + 1] * w2, w1 + w2);
        let plan = linalg::exp_apply(a, C64::new(0.0, -h), x, vectors);
        work += count::exponential(path.n, path.parts.len(), plan.parts * plan.terms);
        bound += plan.rounding(path.n);
    }
    (work, bound)
}

/// The propagation of a beating Ĥ(t) by CF4 (`cf4`) on 2P spacings h, P = `panels`, the impulse by composite Simpson on
/// its nodes: both of fourth order in νh, ν the force's frequency bound, an error the step's doubling does not see (its
/// steps take the same spacing, `adapt`), read by doubling the panels (`Step::regrid`). The middle state is the middle
/// node's, whose change is none. The norm's root between the nodes that bracket it, by CF4 from the earlier node.
fn beating(path: &Path, scratch: &mut Scratch, psi: &[C64], start: f64, tau: f64, panels: usize, threshold: Option<f64>) -> Propagated {
    let (n, nodes) = (path.n, 2 * panels);
    let h = tau / nodes as f64;
    let Carved { matrix, states, held, change, sums, phases, work: vectors } = scratch.carve(path.parts.len(), nodes);
    for (x, (z, phase)) in states.iter_mut().zip(psi.iter().zip(&path.half)) {
        *x = *z * *phase;
    }
    let (mut work, mut crossing) = (0.0, None);
    for k in 0..nodes {
        let t = start + k as f64 * h;
        let (before, after) = states[k * n..(k + 2) * n].split_at_mut(n);
        after.copy_from_slice(before);
        let (exponentials, bound) = cf4(path, matrix, phases, vectors, after, t, h);
        work += exponentials;
        if let Some(theta) = threshold.filter(|&theta| crossing.is_none() && tau > 0.0 && norm_sqr(after) <= theta) {
            let from = k as f64 * h;
            let norm = |s: f64| {
                let mut y = [C64::ZERO; STATES];
                y[..n].copy_from_slice(before);
                cf4(path, matrix, phases, vectors, &mut y[..n], t, s - from);
                norm_sqr(&y[..n])
            };
            crossing = Some(root(norm, (from, norm_sqr(before)), (from + h, norm_sqr(after)), theta, 2.0 * bound + rounding(n + 1)));
        }
    }
    // Composite Simpson on the nodes, weights 1, 4, 2, …, 4, 1 over h/3.
    let states = &*states;
    let impulse = impulse(path, held, change, sums, tau, &states[panels * n..(panels + 1) * n], &mut |each: &mut dyn FnMut(f64, f64, &[C64])| {
        for k in (0..=nodes).filter(|&k| k != panels) {
            let weight = if k == 0 || k == nodes { 1.0 } else if k % 2 == 1 { 4.0 } else { 2.0 };
            each(k as f64 * h - 0.5 * tau, h / 3.0 * weight, &states[k * n..(k + 1) * n]);
        }
    });
    let mut end = [C64::ZERO; STATES];
    for (z, (x, phase)) in end.iter_mut().zip(states[nodes * n..].iter().zip(&path.half)) {
        *z = *x * *phase;
    }
    work += count::impulse(n, path.parts.len(), nodes);
    Propagated { end, impulse, crossing, rounding: 0.0, panels: Some(panels), work }
}

/// A root in [lo, hi] of ln D(s) = ln θ for a non-increasing D with D(hi) ≤ θ, D from `norm` and D at the bracket's ends
/// given (lo itself where D(lo) ≤ θ already): by the inverse interpolation of ln D − ln θ through its last three points
/// (their parabola's root; their last two's line where three do not tell), kept inside the bracket (its middle where
/// both step out), until ln D is within its rounding of ln θ or the iteration no longer moves, at most `ROUNDS` times.
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
    s
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
    /// The propagation's rounding bound of ψ relative to its norm (`spectral`; zero on a grid, whose error is its own).
    pub rounding: f64,
    /// The step's counted work (real multiply-adds) and the generator assemblies it took; of them the grid's work
    /// (`beating`), zero through the eigendecomposition.
    pub work: f64,
    pub assemblies: u64,
    pub grid_work: f64,
    /// The step's propagation to be redone on twice its grid's panels (`Step::regrid`); none through the
    /// eigendecomposition.
    retry: Option<Retry>,
}

/// What a beating propagation needs to be redone on a finer grid: its generator, its path's velocity and gauge, the start
/// state (the cluster's states first), the step, its panels, and its impulse.
#[derive(Clone, Debug)]
struct Retry {
    generator: Generator,
    velocity: [f64; 3],
    gauge: [f64; STATES],
    psi: [C64; STATES],
    start: f64,
    tau: f64,
    panels: usize,
    impulse: [f64; 3],
}

impl Step {
    /// The step's end with its propagation redone on twice its grid's panels (the same path, the same kick otherwise),
    /// and that work: the grid's error read, the step's grid erring by 2⁴/(2⁴ − 1) of the difference (fourth order).
    /// None where the step took no grid.
    pub fn regrid(&self) -> Option<(Trajectory, f64)> {
        let r = self.retry.as_ref()?;
        let n = self.end.psi.len();
        let path = Path::new(&r.generator.parts, n, r.gauge, r.velocity, r.start + 0.5 * r.tau, r.tau);
        let finer = beating(&path, &mut Scratch::new(n), &r.psi[..n], r.start, r.tau, 2 * r.panels, None);
        let s = &self.segment;
        let kicked = (s.kicked.0, std::array::from_fn(|a| s.kicked.1[a] + finer.impulse[a] - r.impulse[a]));
        let (r1, kappa1) = s.flow.advance(kicked, 0.5 * s.tau);
        let mut end = self.end.clone();
        end.r = r1;
        end.kappa = kappa1;
        end.psi.copy_from_slice(&finer.end[..n]);
        Some((end, finer.work))
    }
}

/// Why a step could not be made: the reference's fixed point did not contract within its derived bound, or the state left
/// the floating-point range. The step control halves τ (its more accurate level).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Refused {
    Reference,
    Overflow,
}

/// One propagation of B at the path velocity v (module documentation, # The kick), through the eigendecomposition where
/// Ĥ is static and on the trajectory's grid where it beats (or the decomposition fails), and its kick: the propagation's
/// impulse and the reference's τ∇V_ref(r_h).
fn propagate<D: Dynamics>(model: &D, scratch: &mut Scratch, generator: &Generator, from: &Trajectory, reference: [f64; 3], velocity: [f64; 3], tau: f64, threshold: Option<f64>) -> (Propagated, [f64; 3]) {
    let waves = &model.cluster(from.cluster).waves;
    let path = Path::new(&generator.parts, waves.len(), gauge(waves, velocity), velocity, from.time + 0.5 * tau, tau);
    let propagated = match path.is_static().then(|| spectral(&path, scratch, &from.psi, tau, threshold)).flatten() {
        Some(propagated) => propagated,
        None => beating(&path, scratch, &from.psi, from.time, tau, panels(tau, from.grid), threshold),
    };
    let kick = std::array::from_fn(|a| propagated.impulse[a] + tau * reference[a]);
    (propagated, kick)
}

/// One step of τ (negative: the reverse step) from a trajectory's state (module documentation): the middle and its
/// reference (`middle`); the generator at (r_h, κ_h, t_m), the predictor's momentum from its shear and the reference's
/// force, the first propagation there; the corrected momentum κ_h + Δκ/2, the generator there and the step's propagation;
/// the kick and the second half flow.
pub fn step<D: Dynamics>(model: &D, from: &Trajectory, tau: f64) -> Result<Step, Refused> {
    let half = 0.5 * tau;
    let hm = model.hbar_over_mass();
    let n = from.psi.len();
    let (flow, (r_h, kappa_h)) = middle(model, from.r, from.kappa, half)?;
    let t_m = from.time + half;
    let reference = flow.gradient_at(r_h);
    let mut scratch = Scratch::new(n);
    // The predictor: the shear and the reference's force.
    let first = model.generator(from.cluster, &Point { r: r_h, kappa: kappa_h, t: t_m });
    let held = shear(&first, scratch.carve(first.parts.len(), 0).held, &from.psi, tau);
    let predicted: [f64; 3] = std::array::from_fn(|a| kappa_h[a] + 0.5 * (held[a] + tau * reference[a]));
    let (one, kick_one) = propagate(model, &mut scratch, &first, from, reference, predicted.map(|k| hm * k), tau, None);
    // The correction: the middle momentum of the first propagation's kick.
    let corrected: [f64; 3] = std::array::from_fn(|a| kappa_h[a] + 0.5 * kick_one[a]);
    let velocity = corrected.map(|k| hm * k);
    let second = model.generator(from.cluster, &Point { r: r_h, kappa: corrected, t: t_m });
    let (two, kick) = propagate(model, &mut scratch, &second, from, reference, velocity, tau, Some(from.threshold));
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
    let work = one.work + two.work + count::shear(n, first.parts.len());
    let grid_work = [&one, &two].iter().filter(|p| p.panels.is_some()).map(|p| p.work).sum();
    let retry = two.panels.map(|panels| {
        let mut psi = [C64::ZERO; STATES];
        psi[..n].copy_from_slice(&from.psi);
        Retry { generator: second, velocity, gauge: gauge(&model.cluster(from.cluster).waves, velocity), psi, start: from.time, tau, panels, impulse: two.impulse }
    });
    Ok(Step {
        end,
        crossing: two.crossing,
        segment: Segment { start: from.time, tau, flow, from: (from.r, from.kappa), kicked },
        rounding: two.rounding,
        work,
        assemblies: 2,
        grid_work,
        retry,
    })
}

/// The integrator's counted work of one step of τ through a cluster's generator at a point, the path at the velocity v
/// (µm/µs), and of that step's grid doubling (zero without a grid): what `Step::work` and `Step::regrid` count, priced
/// without taking the step -- the solve's price of a model it has not run, at its pilot's states and steps. Both
/// propagations through the generator's own path (the eigen-propagation's Gauss–Legendre nodes at its spectrum's
/// bandwidth, or the grid's CF4 sub-steps at spacings no wider than `grid`, each exponential's Chebyshev plan at the
/// bounds of the traceless (Ĥ₀ + Σ_j M_j)/2 over a spacing, which a sub-step's combination does not exceed) and the shear.
pub fn step_work(generator: &Generator, waves: &[[f64; 3]], velocity: [f64; 3], tau: f64, grid: f64) -> (f64, f64) {
    let (n, parts) = (waves.len(), generator.parts.len());
    let path = Path::new(&generator.parts, n, gauge(waves, velocity), velocity, 0.0, tau);
    // Each part's traceless bounds (`linalg::bounds`), the static one with the gauge's diagonal (the last).
    let (mut a, mut sums) = (Mat::zeros(n, n), vec![C64::ZERO; n]);
    let (mut rho, mut gamma) = (0.0, 0.0);
    let mut add = |a: &mut Mat| {
        let scalar = traceless(a);
        let (hermitian, decay) = linalg::bounds(a, &mut sums);
        (rho, gamma) = (rho + 0.5 * hermitian, gamma + 0.5 * decay);
        scalar
    };
    for j in (0..parts).filter(|&j| !is_static(&generator.parts[j])) {
        path.combination(&mut a, |i| if i == j { C64::ONE } else { C64::ZERO }, 0.0);
        add(&mut a);
    }
    path.combination(&mut a, |i| if is_static(&generator.parts[i]) { C64::ONE } else { C64::ZERO }, 1.0);
    let scalar = add(&mut a);
    let grid_work = |panels: usize| {
        let h = (tau / (2 * panels) as f64).abs();
        let plan = linalg::Chebyshev::plan(rho * h, gamma * h);
        (2 * panels) as f64 * 2.0 * count::exponential(n, parts, plan.parts * plan.terms) + count::impulse(n, parts, 2 * panels)
    };
    let spectral = path.is_static().then(|| linalg::eigen(&a)).flatten();
    let (propagation, regrid) = match spectral {
        Some(e) => {
            let lambda: Vec<C64> = e.values.iter().map(|l| *l + scalar).collect();
            let nodes = nodes(tau, spread(&lambda));
            (count::spectral(n, nodes) + count::impulse(n, parts, nodes), 0.0)
        }
        None => {
            let panels = panels(tau, grid);
            (grid_work(panels), grid_work(2 * panels))
        }
    };
    (2.0 * propagation + count::shear(n, parts), regrid)
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
/// test. Where no photon can be emitted (every class's weight zero) only the threshold is drawn anew.
pub fn jump<D: Dynamics>(model: &D, trajectory: &mut Trajectory) {
    let j = trajectory.jumps + 1;
    let mut stream = Stream::jump(trajectory.seed, trajectory.index, j);
    let threshold = stream.open_uniform();
    let (class, polarization) = (stream.uniform(), stream.uniform());
    let signs: [f64; 3] = std::array::from_fn(|_| if stream.uniform() < 0.5 { 1.0 } else { -1.0 });
    trajectory.jumps = j;
    trajectory.threshold = threshold;
    let at = trajectory.point();
    let photons = model.photons(trajectory.cluster, &at, &trajectory.psi);
    trajectory.work += model.photon_work();
    let weights: Vec<f64> = photons.iter().map(|p| p.amplitudes.iter().map(|u| norm_sqr(u)).sum()).collect();
    let total: f64 = weights.iter().sum();
    if !(total > 0.0) {
        return;
    }
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
    let scale = 1.0 / w.sqrt();
    trajectory.psi = f.iter().map(|z| *z * scale).collect();
    trajectory.cluster = photon.landing;
    if trajectory.lost.is_none() && model.lost(trajectory.cluster, &trajectory.point(), &trajectory.psi) {
        trajectory.lost = Some(trajectory.time);
    }
}

#[cfg(test)]
mod tests;
