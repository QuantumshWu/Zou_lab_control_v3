//! The initial ensemble (SPEC §2.5): each initial component in its own preparation well, the lowered isothermal distribution
//! times the per-axis harmonic oscillator's Wigner factor, drawn by exact rejection sampling under a proven envelope;
//! Latin-hypercube groups of `GROUP` trajectories, the groups independent.
//!
//! # The lowered isothermal
//! f(r, κ) ∝ e^{−E/k_BT} − 1 for E < 0, zero above (Michie 1963; King 1966): E = (ħ/m)κ²/2 + V(r) the mechanical energy in
//! the component's prepared potential relative to its far level (`Well`; SPEC §1.3: the trap light alone, before t = 0), T
//! the temperature asked for. The bound states of a Gaussian beam fill an unbounded phase space -- along its axis |V| falls
//! as (z_R/z)² while the beam widens as z --, so a Boltzmann factor cut at the far level is not normalizable; the lowered
//! isothermal, the quasi-stationary state of a thermalizing trap that loses its unbound atoms (an absorbing boundary at the
//! escape energy), vanishes linearly there and its tail falls as z⁻³. It is a function of E alone, so the conservative
//! motion keeps it (Jeans), and a trap many k_BT deep holds the Boltzmann distribution to e^{−D/k_BT}.
//!
//! # Its draw
//! The position's weight is w(y(r)), y = −V(r)/k_BT the depth in k_BT and w(y) = kept(3/2, y, y) the momenta below the
//! escape (`kept`; ∫(e^{y−K} − 1)₊ K^{1/2} dK over the kinetic energy K in k_BT), increasing in y. The well's proven envelope
//! −V(r) ≤ B max_l 1/(1 + u_l(r)) over the trap light's Gaussian beams l (u_l = 2ρ_l²/w_l² + ζ_l²/z_{R,l}²: e^{−x} ≤ 1/(1 + x)
//! on the Gaussian factor of |a_l|² = e^{−2ρ²/(w₀²(1+s²))}/(1 + s²), s = ζ/z_R, gives |a_l|² ≤ 1/(1 + u_l)) gives
//! y(r) ≤ a max_l 1/(1 + u_l), a = B/k_BT, and so w(y(r)) ≤ Σ_l E(u_l(r)) with E(u) ≥ w(a/(1 + u)):
//! - u ≤ 1: w(y) ≤ e^y (P ≤ 1 in `kept`) and a/(1 + u) ≤ a(1 − u/2) (the chord of the convex 1/(1 + u)): E = e^{a(1 − u/2)};
//! - u > 1: w(y) ≤ e^y y^{5/2}/Γ(7/2) (e^{y−K} − 1 ≤ (y − K) e^y under the integral), y ≤ a/2 and y ≤ a/u:
//!   E = e^{a/2} a^{5/2} u^{−5/2}/Γ(7/2).
//! In a beam's whitened coordinates ξ (u = |ξ|², d³r = V_l d³ξ, V_l = w_l² z_{R,l}/2) the first piece is a Gaussian of variance
//! 1/a per axis within |ξ| ≤ 1, the second ∝ |ξ|⁻⁵ beyond it: the proposal takes a beam and a piece by their masses
//! (`Component::pieces`), a point from it, and keeps it with probability w(y(r))/Σ_l E(u_l(r)) ≤ 1.
//!
//! The beams' bound is reached only where every beam is at its focus and all are in phase. A well whose bottom lies
//! elsewhere -- a retro-reflected beam's antinode a mirror phase away from the focus, beams that cannot all be in phase --
//! is shallower than B: its depth is c = −V(bottom)/k_BT < a, and under the envelope above an atom would be kept with
//! probability e^{−(a−c)} at best (never, for a cold atom: a − c is the trap's depth times a few 10⁻³, thousands of k_BT
//! at 1 nK). Nothing is deeper than the bottom, so y(r) ≤ min(a/(1 + u), c) and the envelope is flat at the well's own
//! depth where the beams' bound exceeds it:
//! - u ≤ u_c = 2(1 − c/a) (at most 1): E = e^c, a uniform ball in ξ;
//! - u_c < u ≤ 1: E = e^{c − a(u − u_c)/2} g(u), g = (√u_c + √(u − u_c))/√u ≥ 1 -- the chord again, times a factor of at
//!   most √2 that makes its radial law two laws drawn exactly: an exponential and a gamma of 3/2 in a(u − u_c)/2;
//! - u > 1: as before with e^{min(a/2, c)} for e^{a/2}.
//! With the bottom at the focus and in phase, c = a, u_c = 0 and g = 1: the envelope above. Every weight is carried
//! relative to e^c, so no depth overflows the format. The acceptance then falls only as the bottom's thermal volume
//! over the ball's, a power of (a − c). The bottom is the one the descent from the focus reaches (the model's); a
//! proposal that reads deeper than it is counted (`Ensemble::deeper`) and kept as at the ceiling -- the solve flags it.
//!
//! The momentum at depth y:
//! K with density ∝ (e^{y−K} − 1) K^{1/2} on [0, y] by inverting its distribution kept(3/2, y, K)/kept(3/2, y, y) (`invert`),
//! the direction uniform. Every draw is exact; the envelope's looseness sets only the acceptance.
//!
//! # The Wigner factor
//! Along each principal axis i of the component's well (frequency ω_i at its bottom c) the harmonic oscillator's thermal
//! Wigner function is the classical one at T*_i = (ħω_i/2k_B) coth(ħω_i/2k_BT): a classical draw's momentum is scaled by
//! s_i = √(T*_i/T) along e_i, and its displacement from the bottom of its own well likewise, the displacement taken as
//! K⁻¹∇V(r) with K the Hessian at c -- r − c itself in a harmonic well, the displacement from its own site in a standing
//! wave whose sites are as stiff, no site to find. The draw is that map of an exact classical draw; nothing is drawn again.
//!
//! # The exact moments at t = 0
//! Before the map the momenta at each r are isotropic, so the map's ⟨κκᵀ⟩ = S² ⟨κ²⟩/3 with S = Σ_i s_i e_i e_iᵀ, and
//! (ħ/m)⟨κ²⟩/3 = η k_BT, η = ∫ kept(5/2, y(r)) d³r / ∫ kept(3/2, y(r)) d³r the classical ensemble's kinetic temperature
//! relative to T (K's mean is (3/2) kept(5/2)/kept(3/2) at each r). η by quadrature (`Ensemble::kinetic`), its nodes where
//! the ensemble is: about the component's own bottom c, along the principal axes e_i of the potential's Hessian there,
//! in units of the drawn ensemble's widths σ_i = s_i √(k_BT/λ_i) (λ_i the curvatures, s_i the Wigner scales) --
//! r = c + ρ Σ_i σ_i n_i e_i, spherical in (ρ, n): Gauss–Legendre in cos θ, the trapezoid rule in the azimuth (spectral
//! for a periodic integrand), Gauss–Legendre in τ with ρ = τ/(1 − τ). The cloud is at τ = ½ at any temperature and
//! wherever the bottom lies (a cold atom's cloud is nanometres wide, a tilted or retro-reflected well's bottom is off
//! the beams' focus: nodes on the beams' scale about the focus had none in it), five widths -- where a level cuts the
//! ensemble -- at τ = 5/6; the well's far places are at τ → 1, where the integrand σ₁σ₂σ₃ τ²(1 − τ)⁻⁴ kept(s, y) goes
//! to zero smoothly (kept ∝ y^{s+1} as y → 0, and y falls as 1/ρ² at least: (1 − τ)^{2s − 2}) --, every node count
//! doubled from the least that reads a curvature (two per
//! coordinate) until two levels agree within their sums' rounding. What the nodes do not follow -- a density with
//! structure far below the cloud's own scale, the other sites of a lattice as deep as the bottom's -- is in the two
//! last levels' difference, the quadrature's own error.
//!
//! # What a level leaves out
//! A set of phase space given by the kinetic energy it holds at each place, {(r, κ) : (ħ/m)κ²/2 ≤ A(r)} (the model's
//! reach: A = V_min + L − V(r), reach.rs), leaves out a share of the ensemble as it is drawn -- the classical draw
//! (r, κ) mapped by the Wigner factor to (r′, Sκ), r′ = r + Σ_i (s_i − 1)(e_i·K⁻¹∇V(r)) e_i, S = Σ_i s_i e_i e_iᵀ. At r the
//! classical momenta are isotropic with the kinetic energy K (in k_BT) of density ∝ (e^{y−K} − 1) K^{1/2} on [0, y]; a
//! momentum along n (its components on the well's axes) has the mapped kinetic energy q(n) K, q = Σ_i s_i² n_i², and
//! lies beyond the set where K > k₀(n) = A(r′)/(k_BT q(n)). With kept(s, y, k) the momenta below k (`kept`: their
//! number at s = 3/2, their energy at 5/2, both to one constant) the shares left out are
//!
//! ```text
//! atoms:           ∫d³r ⟨kept(3/2, y, y) − kept(3/2, y, min(y, k₀(n)))⟩_n     /  ∫d³r kept(3/2, y, y)
//! kinetic energy:  ∫d³r ⟨q(n) [kept(5/2, y, y) − kept(5/2, y, min(y, k₀(n)))]⟩_n  /  ⟨q⟩ ∫d³r kept(5/2, y, y)
//! ```
//!
//! ⟨·⟩_n the mean over the directions, ⟨q⟩ = Σ_i s_i²/3; over the components by their populations (each component's
//! integrals over its own number ∫kept(3/2); the energy's share the components' energies summed, beyond over all). No
//! atom is drawn: the integrals over r are `kinetic`'s (the same nodes about the bottom, `summed`), the mean
//! over n on one octant of the sphere (q is even in every n_i) by Gauss–Legendre in cos θ on (0, 1) and the midpoint
//! rule in the azimuth on (0, π/2), with the level's node count; every count is doubled as `kinetic`'s are, and the
//! last two levels' difference is the share's own error (`Share`). In a classical harmonic well (s_i = 1, V quadratic)
//! the two are Q(3, L) and Q(4, L) (reach.rs, # The reach's level).
//!
//! # Latin hypercube
//! A group's members draw each of the `DEVIATES` uniform deviates of a first proposal and a momentum -- the component, the
//! piece, the radial coordinate, the direction's two, the kinetic energy's quantile, the momentum direction's two -- in
//! another of `GROUP` equal strata each (the members' order a random permutation of the strata, new for every variable, the
//! same for every member), at a uniform point inside it. A rejected first proposal is followed by proposals from the member's
//! own stream; the momentum keeps its strata. Each member is distributed as an independent draw; the group spreads over every
//! variable's distribution, so its means vary less than independent draws'. Groups are independent of each other, so the
//! jackknife over them measures the error (estimate.rs).
use crate::field::Geometry;
use crate::linalg::{dot3, perpendicular, V3};
use crate::random::{Purpose, Stream};
use crate::special::{by_series, gamma, series, upper};
use crate::units::{rounding, KT_PER_UK};
use rayon::prelude::*;
use std::f64::consts::TAU;

/// The trajectories of one Latin-hypercube group (SPEC §2.5), and with it the block that the delete-one-group jackknife
/// leaves out (estimate.rs). A structural size (SPEC §0): every member is distributed as an independent draw and the
/// groups are independent of each other, so at any size the means the estimates are made of have the ensemble's
/// expectations and the jackknife's variance is that of the groups' means. At a given number of trajectories the size only
/// trades the strata a group spreads over each variable's distribution (as many as its members) and the sums kept (one
/// set a group and output) against the groups the spread is read on (Student's degrees of freedom, one fewer than they).
/// It divides `MOST_TRAJECTORIES`, so that the pilot's own numbers, which start there (solve.rs), start a group.
pub const GROUP: u64 = 8;
const _: () = assert!(crate::units::MOST_TRAJECTORIES % GROUP == 0);
/// The uniform deviates a member draws in its group's strata (module documentation, # Latin hypercube).
const DEVIATES: usize = 8;

/// Why a trap light with a plane wave in it is refused (SPEC §9; shown in the app): a plane wave's potential does not
/// fall off, so the well has no far level to bind atoms below and no envelope of Gaussian beams.
pub const PLANE_WAVE: &str = "与光镊同频的光束都需要束腰：平面波的势在远处不降为零，被俘获原子的初始系综无从定义";
/// Why an initial component whose well's envelope is not positive is refused (SPEC §9; shown in the app): the trap light
/// does not attract that state anywhere.
pub const NO_WELL: &str = "光镊不约束原子：阱光对初态的势没有吸引的部分";

/// What the ensemble needs of one initial component's preparation well (SPEC §1.3, §2.5): defined here by its consumer,
/// implemented by the model for each initial ground state.
pub trait Well: Sync {
    /// The prepared potential relative to its far level, V(r) = V_prep(r) − V_prep(∞) (rad/µs), and its gradient
    /// (rad/µs/µm): an atom is bound where (ħ/m)κ²/2 + V < 0.
    fn potential(&self, r: V3) -> (f64, V3);
    /// The well's bottom c (µm), its principal axes (unit rows) and the frequencies ω_i > 0 (rad/µs) along them: a
    /// minimum of the potential, every curvature positive (the model refuses a light that holds no such bottom, SPEC §1.3).
    fn trap(&self) -> (V3, [V3; 3], V3);
    /// The proven envelope: the trap light's Gaussian beams and B (rad/µs) with −V(r) ≤ B max_l 1/(1 + u_l(r)) everywhere
    /// (module documentation, # Its draw).
    fn envelope(&self) -> (Vec<Geometry>, f64);
}

/// e^{y−shift} P(s, k) − e^{−shift} k^s/Γ(s + 1) = e^{−shift} ∫₀^k (e^{y−q} − 1) q^{s−1} dq/Γ(s) for 0 ≤ k ≤ y: the lowered
/// isothermal's momenta below kinetic energy k at depth y (both in k_BT; s = 3/2 for their number, 5/2 for their energy), P the
/// regularized lower incomplete gamma function (special.rs). To the format's round-off: where P is taken by its series,
/// e^k P(s, k) = k^s/Γ(s + 1) (1 + Σ_{n≥1} k^n/((s + 1)…(s + n))), the lowering is subtracted from the series' first term
/// exactly; elsewhere P is one less the upper function.
pub fn kept(s: f64, y: f64, k: f64, shift: f64) -> f64 {
    if !(k > 0.) {
        return 0.;
    }
    let g = gamma(s + 1.);
    let base = s * k.ln() - shift;
    if by_series(s, k) {
        let rest = series(s, k);
        let (high, low) = ((base + y - k).exp() / g, base.exp() / g);
        // e^{y−k}(1 + rest) − 1 times the power, its −1 against e^{y−k} without cancellation.
        high * rest + if y - k < 1. { low * (y - k).exp_m1() } else { high - low }
    } else {
        (y - shift).exp() * (1. - upper(s, k)) - base.exp() / g
    }
}
/// P(s, x), the regularized lower incomplete gamma function.
fn regularized(s: f64, x: f64) -> f64 {
    kept(s, x, x, x) + (s * x.ln() - x).exp() / gamma(s + 1.)
}

/// The root at `target` of an increasing function on [lo, hi] given with its slope: Newton's steps kept within the bracket,
/// bisection where one leaves it, until a step or the bracket is within the format's round-off.
fn root(f: impl Fn(f64) -> (f64, f64), target: f64, mut lo: f64, mut hi: f64) -> f64 {
    let mut v = 0.5 * (lo + hi);
    loop {
        let (value, slope) = f(v);
        let g = value - target;
        if g == 0. {
            return v;
        }
        if g < 0. {
            lo = v;
        } else {
            hi = v;
        }
        let newton = v - g / slope;
        let next = if slope > 0. && newton > lo && newton < hi { newton } else { 0.5 * (lo + hi) };
        if (next - v).abs() <= f64::EPSILON * v.abs() || !(hi - lo > f64::EPSILON * hi.abs()) {
            return next;
        }
        v = next;
    }
}

/// The kinetic energy (in k_BT) whose distribution at depth y is u: kept(3/2, y, K)/kept(3/2, y, y) = u, its density
/// ∝ (e^{y−K} − 1) K^{1/2} -- solved in v = K^{3/2}, smooth at 0, where d kept/dv = (e^{−K} − e^{−y})/Γ(5/2) at shift y.
fn invert(y: f64, u: f64) -> f64 {
    let s = 1.5;
    let target = u * kept(s, y, y, y);
    root(|v| { let k = v.powf(1. / s); (kept(s, y, k, y), ((-k).exp() - (-y).exp()) / gamma(s + 1.)) }, target, 0., y.powf(s)).powf(1. / s)
}

/// A Gaussian beam of the envelope (module documentation, # Its draw): its frame (two axes across it, its direction), its
/// waist and Rayleigh range.
#[derive(Clone, Debug)]
struct Mode {
    frame: [V3; 3],
    waist: f64,
    rayleigh: f64,
}
impl Mode {
    fn new(g: &Geometry) -> Result<Mode, String> {
        let (Some(waist), Some(rayleigh)) = (g.waist, g.rayleigh()) else {
            return Err(PLANE_WAVE.into());
        };
        let n = g.direction;
        let (e1, e2) = perpendicular(n);
        Ok(Mode { frame: [e1, e2, n], waist, rayleigh })
    }
    /// u = 2ρ²/w₀² + ζ²/z_R² at r.
    fn coordinate(&self, r: V3) -> f64 {
        let p = self.frame.map(|e| dot3(e, r));
        2. * (p[0] * p[0] + p[1] * p[1]) / (self.waist * self.waist) + p[2] * p[2] / (self.rayleigh * self.rayleigh)
    }
    /// r at the whitened point ξ, u(r) = |ξ|².
    fn point(&self, xi: V3) -> V3 {
        let w = self.waist / 2f64.sqrt();
        std::array::from_fn(|i| w * xi[0] * self.frame[0][i] + w * xi[1] * self.frame[1][i] + self.rayleigh * xi[2] * self.frame[2][i])
    }
    /// d³r/d³ξ.
    fn volume(&self) -> f64 {
        0.5 * self.waist * self.waist * self.rayleigh
    }
}

/// The unit vector of cos θ = μ and azimuth φ.
fn direction(mu: f64, phi: f64) -> V3 {
    let (s, (sp, cp)) = ((1. - mu * mu).max(0.).sqrt(), phi.sin_cos());
    [s * cp, s * sp, mu]
}

/// One initial component: its basis state and its share, its well, the envelope's top a = B/k_BT, its ceiling c (the
/// bottom's depth in k_BT, at most a) and beams, the cumulative masses of its pieces relative to e^c (per beam the flat
/// ball, the two laws of the chord, the tail; module documentation, # Its draw), and its well's bottom, axes, curvatures
/// λ_i = ω_i²/(ħ/m) and Wigner scales s_i.
struct Component<'a> {
    state: usize,
    weight: f64,
    well: &'a dyn Well,
    top: f64,
    ceiling: f64,
    modes: Vec<Mode>,
    pieces: Vec<f64>,
    centre: V3,
    axes: [V3; 3],
    curvatures: V3,
    scales: V3,
}

impl Component<'_> {
    /// u_c, where the beams' bound meets the ceiling: 2(1 − c/a), at most 1.
    fn flat(&self) -> f64 {
        (2. * (1. - self.ceiling / self.top)).min(1.)
    }
    /// The masses of a beam's four pieces over e^c and its volume (module documentation, # Its draw): the flat ball, the
    /// chord's exponential and its gamma law, the tail.
    fn masses(&self) -> [f64; 4] {
        let (a, flat) = (self.top, self.flat());
        let span = 0.5 * a * (1. - flat);
        [
            2. * TAU / 3. * flat.powf(1.5),
            TAU * flat.sqrt() * (2. / a) * -(-span).exp_m1(),
            TAU * (2. / a).powf(1.5) * gamma(1.5) * if span > 0. { regularized(1.5, span) } else { 0. },
            TAU * ((0.5 * a).min(self.ceiling) - self.ceiling).exp() * a.powf(2.5) / gamma(3.5),
        ]
    }
    /// Σ_l E(u_l(r)) e^{−c} (module documentation, # Its draw).
    fn envelope(&self, r: V3) -> f64 {
        let (a, flat) = (self.top, self.flat());
        self.modes
            .iter()
            .map(|m| {
                let u = m.coordinate(r);
                if u <= flat {
                    1.
                } else if u <= 1. {
                    (-0.5 * a * (u - flat)).exp() * (flat.sqrt() + (u - flat).sqrt()) / u.sqrt()
                } else {
                    ((0.5 * a).min(self.ceiling) - self.ceiling).exp() * a.powf(2.5) * u.powf(-2.5) / gamma(3.5)
                }
            })
            .sum()
    }
    /// A proposal from the piece of the deviate `piece`, its radial coordinate and direction from the next three.
    fn propose(&self, piece: f64, radial: f64, mu: f64, phi: f64) -> V3 {
        let total = self.pieces[self.pieces.len() - 1];
        let j = self.pieces.partition_point(|&c| c <= piece * total).min(self.pieces.len() - 1);
        let (mode, a, flat) = (&self.modes[j / 4], self.top, self.flat());
        let span = 0.5 * a * (1. - flat);
        let size = match j % 4 {
            // The flat ball: uniform in |ξ|² ≤ u_c.
            0 => flat.sqrt() * radial.cbrt(),
            // The chord's exponential: x = a(u − u_c)/2 with density ∝ e^{−x} on [0, span].
            1 => (flat - 2. / a * (radial * (-span).exp_m1()).ln_1p()).sqrt(),
            // Its gamma law: density ∝ x^{1/2} e^{−x} on [0, span], P(3/2, x) = radial P(3/2, span), in v = x^{3/2}.
            2 => {
                let target = radial * regularized(1.5, span);
                let x = root(|v| { let x = v.powf(2. / 3.); (regularized(1.5, x), (-x).exp() / gamma(2.5)) }, target, 0., span.powf(1.5)).powf(2. / 3.);
                (flat + 2. * x / a).sqrt()
            }
            // The tail: |ξ| with density ∝ |ξ|⁻³ on (1, ∞).
            _ => (1. - radial).powf(-0.5),
        };
        mode.point(direction(2. * mu - 1., TAU * phi).map(|x| x * size))
    }
    /// The Wigner map of a classical draw (module documentation, # The Wigner factor): r with its potential's gradient there, κ.
    fn wigner(&self, r: V3, gradient: V3, kappa: V3) -> (V3, V3) {
        let mut out = (r, [0.; 3]);
        for i in 0..3 {
            let e = self.axes[i];
            let (shift, along) = ((self.scales[i] - 1.) * dot3(e, gradient) / self.curvatures[i], self.scales[i] * dot3(e, kappa));
            for c in 0..3 {
                out.0[c] += shift * e[c];
                out.1[c] += along * e[c];
            }
        }
        out
    }
}

/// A trajectory's initial state: its basis state, position (µm) and momentum κ (1/µm).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Start {
    pub state: usize,
    pub r: V3,
    pub kappa: V3,
}

/// A component's quadrature at t = 0 (`Ensemble::kinetic`): η and its difference between the last two levels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quadrature {
    pub eta: f64,
    pub difference: f64,
}

/// A share of the ensemble by quadrature (`Ensemble::beyond`): its value at the last level and its difference from the
/// level before, the quadrature's own error.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Share {
    pub value: f64,
    pub difference: f64,
}

/// The initial ensemble of a request (module documentation).
pub struct Ensemble<'a> {
    components: Vec<Component<'a>>,
    kt: f64,
    hbar_over_mass: f64,
    /// The proposals that read deeper than their well's bottom (`deeper`).
    over: std::sync::atomic::AtomicU64,
    /// The draws made, the potential's evaluations they read (one a proposal), and the quadratures' (`evaluations`).
    drawn: std::sync::atomic::AtomicU64,
    proposed: std::sync::atomic::AtomicU64,
    integrated: std::sync::atomic::AtomicU64,
}

/// What an ensemble's draws and quadratures cost its model so far, in evaluations of the prepared potential
/// (`Ensemble::evaluations`): per draw the proposals it read -- one at the least, where none was drawn yet --, and the
/// quadratures' evaluations in all (`summed`) -- what a model's potential costs it beside its steps (solve.rs `Formula`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Evaluations {
    pub per_draw: f64,
    pub integrated: f64,
}

impl<'a> Ensemble<'a> {
    /// The ensemble of the initial components (basis state, population, its well) at the temperature T (µK), with ħ/m
    /// (µm²/µs); the populations normalized over them. Refused where a well holds no atom (its envelope's B is not
    /// positive, `NO_WELL`) or its envelope is not one of Gaussian beams (`PLANE_WAVE`).
    pub fn new(initial: &[(usize, f64, &'a dyn Well)], temperature_uk: f64, hbar_over_mass: f64) -> Result<Ensemble<'a>, String> {
        let kt = KT_PER_UK * temperature_uk;
        let total: f64 = initial.iter().map(|c| c.1).sum();
        let components = initial
            .iter()
            .map(|&(state, p, well)| {
                let (beams, bound) = well.envelope();
                if !(bound > 0.) || beams.is_empty() {
                    return Err(NO_WELL.to_string());
                }
                let modes = beams.iter().map(Mode::new).collect::<Result<Vec<_>, _>>()?;
                let a = bound / kt;
                let (centre, axes, frequencies) = well.trap();
                let ceiling = (-well.potential(centre).0 / kt).min(a);
                if !(ceiling > 0.) {
                    return Err(NO_WELL.to_string());
                }
                let curvatures = frequencies.map(|w| w * w / hbar_over_mass);
                let scales = frequencies.map(|w| {
                    let half = 0.5 * w / kt;
                    (half / half.tanh()).sqrt()
                });
                let mut component = Component { state, weight: p / total, well, top: a, ceiling, modes, pieces: vec![], centre, axes, curvatures, scales };
                let masses = component.masses();
                let mut sum = 0.;
                component.pieces = component.modes.iter().flat_map(|m| masses.map(|mass| mass * m.volume())).map(|mass| { sum += mass; sum }).collect();
                Ok(component)
            })
            .collect::<Result<Vec<_>, String>>()?;
        let zero = || std::sync::atomic::AtomicU64::new(0);
        Ok(Ensemble { components, kt, hbar_over_mass, over: zero(), drawn: zero(), proposed: zero(), integrated: zero() })
    }

    /// The initial state of trajectory `index` of the run `seed`: its group the `GROUP` trajectories from index − index mod
    /// GROUP, its strata from that group's Latin stream, the rest from its own (module documentation, # Latin hypercube).
    pub fn draw(&self, seed: u64, index: u64) -> Start {
        let (first, member) = (index - index % GROUP, (index % GROUP) as usize);
        let mut shared = Stream::new(seed, Purpose::Latin, first, 0);
        let mut own = Stream::new(seed, Purpose::Initial, index, 0);
        let deviates: [f64; DEVIATES] = std::array::from_fn(|_| {
            let mut order: Vec<u64> = (0..GROUP).collect();
            for i in (1..GROUP as usize).rev() {
                let j = ((shared.next_u64() as u128 * (i as u128 + 1)) >> 64) as usize;
                order.swap(i, j);
            }
            (order[member] as f64 + own.interior()) / GROUP as f64
        });
        let mut pick = deviates[0];
        let c = self.components.iter().position(|c| { let inside = pick < c.weight; pick -= c.weight; inside }).unwrap_or(self.components.len() - 1);
        let c = &self.components[c];
        let mut proposal = [deviates[1], deviates[2], deviates[3], deviates[4]];
        self.drawn.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let (r, y, gradient) = loop {
            let r = c.propose(proposal[0], proposal[1], proposal[2], proposal[3]);
            self.proposed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let (v, gradient) = c.well.potential(r);
            let y = -v / self.kt;
            if y > c.ceiling + rounding(self.components.len()) * c.top {
                self.over.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            if y > 0. && own.open_uniform() * c.envelope(r) <= kept(1.5, y, y, c.ceiling) {
                break (r, y, gradient);
            }
            proposal = std::array::from_fn(|_| own.interior());
        };
        let k = invert(y, deviates[5]);
        let size = (2. * k * self.kt / self.hbar_over_mass).sqrt();
        let kappa = direction(2. * deviates[6] - 1., TAU * deviates[7]).map(|x| x * size);
        let (r, kappa) = c.wigner(r, gradient, kappa);
        Start { state: c.state, r, kappa }
    }

    /// Each component's quadrature at t = 0 (module documentation, # The exact moments): every
    /// node count doubled until η's integrals at two levels agree within their sums' rounding, or the next level would take
    /// more than `most` evaluations of the potential (two levels at least).
    pub fn kinetic(&self, most: u64) -> Vec<Quadrature> {
        self.components
            .iter()
            .map(|c| {
                let eta = |x: &[f64; 2]| x[1] / x[0];
                let mut n = 2;
                let mut last = self.integrals(c, n);
                loop {
                    n *= 2;
                    let next = self.integrals(c, n);
                    let within = (0..2).all(|s| (next.0[s] - last.0[s]).abs() <= rounding(next.1) * next.0[s] + rounding(last.1) * last.0[s]);
                    let read = Quadrature { eta: eta(&next.0), difference: (eta(&next.0) - eta(&last.0)).abs() };
                    last = next;
                    if within || (2 * n as u64).pow(3) > most {
                        break read;
                    }
                }
            })
            .collect()
    }
    /// ∫ kept(3/2, y) d³r and ∫ kept(5/2, y) d³r (at shift c), with n nodes per coordinate per beam, and the evaluations.
    fn integrals(&self, c: &Component, n: usize) -> ([f64; 2], usize) {
        self.summed(c, n, &|_, y, _| [kept(1.5, y, y, c.ceiling), kept(2.5, y, y, c.ceiling)])
    }
    /// ∫ f d³r over a component's bound places (y > 0) of the functions `of` of a place, its depth y and the potential's
    /// gradient there, with n nodes per coordinate about the component's bottom in units of its drawn widths (module
    /// documentation, # The exact moments), and the evaluations of the potential.
    fn summed<const K: usize>(&self, c: &Component, n: usize, of: &(dyn Fn(V3, f64, V3) -> [f64; K] + Sync)) -> ([f64; K], usize) {
        let nodes = crate::special::gauss_legendre(n);
        let widths: V3 = std::array::from_fn(|i| c.scales[i] * (self.kt / c.curvatures[i]).sqrt());
        let volume = widths[0] * widths[1] * widths[2];
        let rows: Vec<[f64; K]> = (0..n)
            .into_par_iter()
            .map(|row| {
                let (mu, w_mu) = nodes[row];
                let mut sums = [0.; K];
                for j in 0..n {
                    let unit = direction(mu, TAU * (j as f64 + 0.5) / n as f64);
                    for &(z, w_z) in &nodes {
                        let tau = 0.5 * (z + 1.);
                        let rho = tau / (1. - tau);
                        let r: V3 = std::array::from_fn(|a| c.centre[a] + rho * (0..3).map(|i| widths[i] * unit[i] * c.axes[i][a]).sum::<f64>());
                        let (v, gradient) = c.well.potential(r);
                        let y = -v / self.kt;
                        if y > 0. {
                            let weight = volume * w_mu * (TAU / n as f64) * 0.5 * w_z * rho * rho / ((1. - tau) * (1. - tau));
                            for (sum, value) in sums.iter_mut().zip(of(r, y, gradient)) {
                                *sum += weight * value;
                            }
                        }
                    }
                }
                sums
            })
            .collect();
        // The rows summed in their order: the result does not depend on the threads.
        self.integrated.fetch_add((n * n * n) as u64, std::sync::atomic::Ordering::Relaxed);
        (rows.iter().fold([0.; K], |s, x| std::array::from_fn(|i| s[i] + x[i])), n * n * n)
    }

    /// What the set {(r, κ) : (ħ/m)κ²/2 ≤ `room`(r)} leaves out of the ensemble (module documentation, # What a level
    /// leaves out; `room` in rad/µs): the share of the atoms and the share of their kinetic energy, by quadrature -- per
    /// component the node counts doubled as `kinetic`'s, until the integrals at two levels agree within their sums'
    /// rounding or the next level would take more than `most` evaluations of the component's potential (two levels at
    /// least) --, each with its difference between the last two levels.
    pub fn beyond(&self, room: &(dyn Fn(V3) -> f64 + Sync), most: u64) -> [Share; 2] {
        // Per component at a level: the atoms beyond over its number, their mapped kinetic energy and all of it over its
        // number (in k_BT, to the energy's one constant).
        let level = |c: &Component, n: usize| -> ([f64; 4], usize) {
            let nodes = crate::special::gauss_legendre(n);
            let squares = c.scales.map(|s| s * s);
            let octant: Vec<(f64, f64)> = nodes
                .iter()
                .flat_map(|&(x, w)| (0..n).map(move |j| (0.5 * (x + 1.), 0.25 * TAU * (j as f64 + 0.5) / n as f64, 0.5 * w / n as f64)))
                .map(|(mu, phi, weight)| {
                    let unit = direction(mu, phi);
                    ((0..3).map(|i| squares[i] * unit[i] * unit[i]).sum::<f64>(), weight)
                })
                .collect();
            self.summed(c, n, &|r, y, gradient| {
                let held = (room(c.wigner(r, gradient, [0.; 3]).0) / self.kt).max(0.);
                let (number, energy) = (kept(1.5, y, y, c.ceiling), kept(2.5, y, y, c.ceiling));
                let mut out = [number, energy, 0., 0.];
                for &(q, weight) in &octant {
                    let k = (held / q).min(y);
                    out[2] += weight * (number - kept(1.5, y, k, c.ceiling));
                    out[3] += weight * q * (energy - kept(2.5, y, k, c.ceiling));
                }
                out
            })
        };
        // Per component its last two levels.
        let levels: Vec<[[f64; 4]; 2]> = self
            .components
            .iter()
            .map(|c| {
                let mut n = 2;
                let mut last = level(c, n);
                loop {
                    n *= 2;
                    let next = level(c, n);
                    let within = (0..4).all(|s| (next.0[s] - last.0[s]).abs() <= rounding(next.1) * next.0[s] + rounding(last.1) * last.0[s]);
                    let read = [last.0, next.0];
                    last = next;
                    if within || (2 * n as u64).pow(3) > most {
                        break read;
                    }
                }
            })
            .collect();
        let shares = |at: usize| -> [f64; 2] {
            let (mut atoms, mut beyond, mut all) = (0., 0., 0.);
            for (c, sums) in self.components.iter().zip(&levels) {
                let [number, energy, out, out_energy] = sums[at];
                atoms += c.weight * out / number;
                beyond += c.weight * out_energy / number;
                all += c.weight * c.scales.iter().map(|s| s * s).sum::<f64>() / 3. * energy / number;
            }
            [atoms, beyond / all]
        };
        let (before, now) = (shares(0), shares(1));
        std::array::from_fn(|i| Share { value: now[i], difference: (now[i] - before[i]).abs() })
    }

    /// (ħ/m)⟨κ_a²⟩ per lab axis (rad/µs) of the ensemble with the components' η (`kinetic`): Σ_b p_b η_b k_BT (S_b²)_aa (module
    /// documentation, # The exact moments); T_a(0) is it over k_B, n̄_a(0) it over ħω_a less ½.
    pub fn momentum(&self, kinetic: &[f64]) -> V3 {
        let mut out = [0.; 3];
        for (c, eta) in self.components.iter().zip(kinetic) {
            for i in 0..3 {
                for (a, o) in out.iter_mut().enumerate() {
                    *o += c.weight * eta * self.kt * c.scales[i] * c.scales[i] * c.axes[i][a] * c.axes[i][a];
                }
            }
        }
        out
    }

    /// The prepared potential's evaluations so far (`Evaluations`): per draw, and the quadratures'.
    pub fn evaluations(&self) -> Evaluations {
        let (drawn, proposed) = (self.drawn.load(std::sync::atomic::Ordering::Relaxed), self.proposed.load(std::sync::atomic::Ordering::Relaxed));
        Evaluations { per_draw: if drawn > 0 { proposed as f64 / drawn as f64 } else { 1. }, integrated: self.integrated.load(std::sync::atomic::Ordering::Relaxed) as f64 }
    }

    /// The proposals so far that read deeper than their well's bottom (module documentation, # Its draw): none where the
    /// bottom is the potential's lowest point.
    pub fn deeper(&self) -> u64 {
        self.over.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// The depth at each component's bottom in k_BT: what the setup's note says of the ensemble (ledger.rs).
    pub fn depths(&self) -> Vec<f64> {
        self.components.iter().map(|c| -c.well.potential(c.centre).0 / self.kt).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `kept` against its definition by quadrature, at depths and reaches from 10⁻⁴ to 40 k_BT and both sides of its two
    /// branches, for the number and the energy (s = 3/2, 5/2) and the others the series handles.
    #[test]
    fn the_kept_momenta_are_their_integral() {
        let nodes = crate::special::gauss_legendre(64);
        for s in [0.5, 1., 1.5, 2., 2.5] {
            for y in [1e-4, 0.3, 1.4, 2.6, 3.6, 7., 40.] {
                for frac in [0.3, 0.99, 1.] {
                    let k = y * frac;
                    // ∫₀^k (e^{y−q} − 1) q^{s−1} dq/Γ(s) in q = k v² (smooth at 0), on 8 panels.
                    let mut sum = 0.;
                    for p in 0..8 {
                        let (a, b) = (p as f64 / 8., (p + 1) as f64 / 8.);
                        for &(x, w) in &nodes {
                            let v = 0.5 * (a + b) + 0.5 * (b - a) * x;
                            let q = k * v * v;
                            sum += w * 0.5 * (b - a) * 2. * k * v * (y - q).exp_m1() * q.powf(s - 1.);
                        }
                    }
                    let exact = sum / gamma(s);
                    let value = kept(s, y, k, 0.);
                    assert!((value - exact).abs() <= 1e-11 * exact.abs().max(1e-300), "s {s} y {y} k {k}: {value:e} against {exact:e}");
                }
            }
        }
    }

    /// The kinetic energy's inverse distribution against `kept`, to round-off, across the deviates and depths.
    #[test]
    fn the_kinetic_energy_inverts_its_distribution() {
        for y in [1e-3, 0.7, 4.3, 30., 300.] {
            for u in [1e-9, 0.01, 0.3, 0.5, 0.97, 1. - 1e-12] {
                let k = invert(y, u);
                let back = kept(1.5, y, k, y) / kept(1.5, y, y, y);
                assert!(k >= 0. && k <= y * (1. + 1e-12) && (back - u).abs() <= 1e-11, "y {y} u {u}: k {k}, distribution {back}");
            }
        }
    }

    /// A harmonic well cut at its far level, V = −D (1 − u)₊ with u = 2ρ²/w² + ζ²/z_R² of one beam (rad/µs): quadratic up
    /// to the escape, its curvatures 4D/w² across and 2D/z_R² along the beam, under the beam's own envelope
    /// (1 − u ≤ 1/(1 + u)), B = D.
    struct Harmonic {
        depth: f64,
        waist: f64,
        rayleigh: f64,
        hbar_over_mass: f64,
    }
    impl Harmonic {
        fn frequencies(&self) -> V3 {
            let across = (self.hbar_over_mass * 4. * self.depth / (self.waist * self.waist)).sqrt();
            [across, across, (self.hbar_over_mass * 2. * self.depth / (self.rayleigh * self.rayleigh)).sqrt()]
        }
    }
    impl Well for Harmonic {
        fn potential(&self, r: V3) -> (f64, V3) {
            let u = 2. * (r[0] * r[0] + r[1] * r[1]) / (self.waist * self.waist) + r[2] * r[2] / (self.rayleigh * self.rayleigh);
            if u >= 1. {
                return (0., [0.; 3]);
            }
            (-self.depth * (1. - u), [4. * self.depth * r[0] / (self.waist * self.waist), 4. * self.depth * r[1] / (self.waist * self.waist), 2. * self.depth * r[2] / (self.rayleigh * self.rayleigh)])
        }
        fn trap(&self) -> (V3, [V3; 3], V3) {
            ([0.; 3], [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], self.frequencies())
        }
        fn envelope(&self) -> (Vec<Geometry>, f64) {
            (vec![Geometry::new([0., 0., 1.], TAU / (std::f64::consts::PI * self.waist * self.waist / self.rayleigh), Some(self.waist))], self.depth)
        }
    }

    /// What a level leaves out of the ensemble (`Ensemble::beyond`) against the harmonic well's closed forms. A well
    /// quadratic to its escape (`Harmonic`), 30 k_BT deep (a millikelvin tweezer at 30 µK; the lowering is e^{−(30−L)} of
    /// the Boltzmann factor at the level) and 30000 k_BT deep (the same well at 0.03 µK: the cloud a thousandth of the
    /// beam's scale in each direction, where nodes on the beam's scale had none in it), the set
    /// {V + (ħ/m)κ²/2 ≤ V_min + L k_BT}:
    /// - classical (ħω ≪ k_BT: the Wigner scales are one): the energy of six quadratic degrees of freedom over k_BT
    ///   follows Γ(3), so the atoms beyond are Q(3, L) = e^{−L}(1 + L + L²/2) and their share of the kinetic energy
    ///   Q(4, L) = Q(3, L) + e^{−L} L³/6, at L = 6, 9 and 12.89 (where Q(3, L) = z²/(2¹⁴ + z²));
    /// - with the Wigner factor (ħω_ρ = 3 k_BT, ω_ζ 0.24 of it: s_ρ² = 1.66, s_ζ² = 1.04): each axis's energy is
    ///   s_i² times an exponential's, X = s_ρ² G₂ + s_ζ² G₁ (G_k of Γ(k)): the atoms beyond P(X > L) = e^{−L/a}(1 + L/a) +
    ///   e^{−L/b}(1 − e^{−cL}(1 + cL))/(a c)², a = s_ρ², b = s_ζ², c = 1/a − 1/b, and the kinetic energy's share
    ///   E[X; X > L]/E[X] = (L P(X > L) + ∫_L^∞ P(X > x) dx)/(2a + b) (the kinetic half of each axis's energy is half of
    ///   it in the mean, beyond any level of the whole).
    /// The quadrature with 2¹⁶ evaluations a level (32 nodes a coordinate) gives each to a tenth of itself at either
    /// depth, the same numbers in the cloud's own units (measured: the atoms' share −2 %, +0.1 % and +8 % at the three
    /// levels, the energy's within 1 %): the momenta's cut at the level is a kink of the radial integrand, which a
    /// Gauss rule takes algebraically (a share of 2.4e-4 known to a tenth is 2e-5 of the ensemble; two levels' difference
    /// is its booked error, and may be smaller than the error itself where two levels happen to agree).
    #[test]
    fn what_a_level_leaves_out_is_the_harmonic_well_s_closed_form() {
        let (kt, waist, rayleigh) = (KT_PER_UK * 1., 1.0, 3.0);
        let tail = |a: f64, b: f64, level: f64| -> f64 {
            let c = 1. / a - 1. / b;
            (-level / a).exp() * (1. + level / a) + (-level / b).exp() * (1. - (-c * level).exp() * (1. + c * level)) / (a * c).powi(2)
        };
        for (depth, quantum) in [(30. * kt, false), (30. * kt, true), (30000. * kt, false), (30000. * kt, true)] {
            // ħω_ρ = 3 k_BT: (ħ/m) 4D/w² = (3 k_BT)².
            let hbar_over_mass = if quantum { 9. * kt * kt * waist * waist / (4. * depth) } else { 1e-12 };
            let well = Harmonic { depth, waist, rayleigh, hbar_over_mass };
            let ensemble = Ensemble::new(&[(0, 1., &well as &dyn Well)], 1., hbar_over_mass).unwrap();
            let squares = ensemble.components[0].scales.map(|s| s * s);
            assert!(if quantum { (squares[0] - 1.657).abs() < 1e-3 && (squares[2] - 1.041).abs() < 1e-3 } else { squares.iter().all(|s| (s - 1.).abs() < 1e-6) }, "the Wigner scales {squares:?}");
            for level in [6., 9., 12.89] {
                let room = |r: V3| level * kt - (well.potential(r).0 + depth);
                let [atoms, energy] = ensemble.beyond(&room, 1 << 16);
                let exact = if quantum {
                    let (a, b) = (squares[0], squares[2]);
                    // ∫_L^∞ P(X > x) dx by Simpson to 60 of the larger scale beyond L.
                    let (panels, top) = (20000, 60. * a.max(b));
                    let integral: f64 = (0..=2 * panels).map(|i| (if i == 0 || i == 2 * panels { 1. } else if i % 2 == 1 { 4. } else { 2. }) * tail(a, b, level + top * i as f64 / (2 * panels) as f64)).sum::<f64>() * top / (6 * panels) as f64;
                    [tail(a, b, level), (level * tail(a, b, level) + integral) / (2. * a + b)]
                } else {
                    let q3 = (-level).exp() * (1. + level + 0.5 * level * level);
                    [q3, q3 + (-level).exp() * level.powi(3) / 6.]
                };
                eprintln!("depth {} k_BT, quantum {quantum}, L = {level}: atoms {:.4e} ± {:.1e} against {:.4e}; kinetic energy {:.4e} ± {:.1e} against {:.4e}", depth / kt, atoms.value, atoms.difference, exact[0], energy.value, energy.difference, exact[1]);
                for (share, exact) in [atoms, energy].iter().zip(exact) {
                    assert!((share.value - exact).abs() <= 0.1 * exact, "depth {} k_BT, quantum {quantum}, L = {level}: {:e} ± {:e} against {exact:e}", depth / kt, share.value, share.difference);
                }
            }
        }
    }

    /// An isotropic Lorentzian well V = −D/(1 + |r|²/R²) (rad/µs), its envelope tight: one beam whose u is |r|²/R²
    /// (w₀ = √2 R, z_R = R, so k = 1/R), B = D. Its bottom at the origin with curvature 2D/R² on every axis.
    struct Lorentzian {
        depth: f64,
        radius: f64,
        hbar_over_mass: f64,
    }
    impl Well for Lorentzian {
        fn potential(&self, r: V3) -> (f64, V3) {
            let q = 1. + dot3(r, r) / (self.radius * self.radius);
            (-self.depth / q, r.map(|x| 2. * self.depth * x / (self.radius * self.radius * q * q)))
        }
        fn trap(&self) -> (V3, [V3; 3], V3) {
            let w = (self.hbar_over_mass * 2. * self.depth / (self.radius * self.radius)).sqrt();
            ([0.; 3], [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], [w; 3])
        }
        fn envelope(&self) -> (Vec<Geometry>, f64) {
            (vec![Geometry::new([0., 0., 1.], 1. / self.radius, Some(2f64.sqrt() * self.radius))], self.depth)
        }
    }

    /// The Lorentzian well's radial integrals at a = D/k_BT by a 1D rule independent of the module's 3D one: ∫ g(y) kept(3/2, y)
    /// √u du for g = 1, y and the kept(5/2, y) one, y = a/(1 + u), in u = τ²/(1 − τ²) (√u du = 2τ²(1 − τ²)^{−5/2} dτ,
    /// y = a(1 − τ²)) by composite Simpson on 20000 panels, at the shift a as the module's; at τ = 1 the integrands' limits,
    /// e^{−a} a^{5/2}/Γ(7/2) for the number (kept(s, y) → y^{s+1}/Γ(s + 2)) and zero for the others.
    fn lorentzian(a: f64) -> [f64; 3] {
        let panels = 20000;
        let mut sums = [0.; 3];
        for i in 1..=2 * panels {
            let tau = i as f64 / (2 * panels) as f64;
            let w = if i == 2 * panels { 1. } else if i % 2 == 1 { 4. } else { 2. };
            if i == 2 * panels {
                sums[0] += w * (-a).exp() * a.powf(2.5) / gamma(3.5);
            } else {
                let (t, y) = (tau * tau, a * (1. - tau * tau));
                let measure = t * (1. - t).powf(-2.5);
                sums[0] += w * measure * kept(1.5, y, y, a);
                sums[1] += w * measure * kept(2.5, y, y, a);
                sums[2] += w * measure * y * kept(1.5, y, y, a);
            }
        }
        sums
    }
    /// η of the Lorentzian well (`lorentzian`).
    fn lorentzian_eta(a: f64) -> f64 {
        let s = lorentzian(a);
        s[1] / s[0]
    }

    /// η by the module's 3D quadrature meets the independent 1D rule to 1e-9 at a shallow and a deep top, the levels converged
    /// within the rounding.
    #[test]
    fn the_kinetic_temperature_is_its_integral() {
        for a in [3., 40.] {
            let well = Lorentzian { depth: a * KT_PER_UK, radius: 1.3, hbar_over_mass: 1e-6 };
            let ensemble = Ensemble::new(&[(0, 1., &well)], 1., 1e-6).unwrap();
            let Quadrature { eta, difference, .. } = ensemble.kinetic(u64::MAX)[0];
            let reference = lorentzian_eta(a);
            assert!((eta - reference).abs() <= 1e-9 && difference < 1e-9, "a {a}: η {eta} against {reference} (levels {difference})");
        }
    }

    /// A deep trap holds the Boltzmann distribution, whose η is 1, to e^{−D/k_BT} (module documentation). Exactly:
    /// kept(5/2, y) − kept(3/2, y) = e^y (P(7/2, y) − P(5/2, y)) = −y^{5/2}/Γ(7/2) at every point, so
    /// 1 − η = ∫ y^{5/2} d³r / (Γ(7/2) ∫ kept(3/2, y) d³r) > 0. In the Lorentzian well, y = a/(1 + u) with d³r ∝ √u du:
    /// ∫ y^{5/2} √u du = a^{5/2} B(3/2, 1) = (2/3) a^{5/2}, and the number's integral ∫ e^y P(5/2, y) √u du is at least its
    /// part on u ≤ 1, where e^{a/(1+u)} ≥ e^{a(1−u)} and P(5/2, y) ≥ P(5/2, a/2): e^a a^{−3/2} Γ(3/2) P(3/2, a) P(5/2, a/2).
    /// Hence 0 < 1 − η ≤ (2/3) a⁴ e^{−a}/(Γ(7/2) Γ(3/2) P(3/2, a) P(5/2, a/2)), the e^{−D/k_BT} times the phase space's a⁴:
    /// 7.5e-5 at a = 20 and 2.5e-12 at a = 40, both far above the quadrature's last two levels' difference; at a = 400 it
    /// is below the format, and so is everything the two integrals differ by: η is 1 itself.
    #[test]
    fn a_deep_well_s_kinetic_temperature_is_the_boltzmann_one_less_the_lowering() {
        for a in [20., 40., 400.] {
            let well = Lorentzian { depth: a * KT_PER_UK, radius: 1.3, hbar_over_mass: 1e-6 };
            let Quadrature { eta, difference } = Ensemble::new(&[(0, 1., &well)], 1., 1e-6).unwrap().kinetic(u64::MAX)[0];
            let lowering = 2. / 3. * a.powi(4) * (-a).exp() / (gamma(3.5) * gamma(1.5) * regularized(1.5, a) * regularized(2.5, 0.5 * a));
            let resolved = lowering > f64::EPSILON;
            assert!(1. - eta <= lowering && if resolved { 1. - eta > difference } else { eta == 1. }, "a {a}: 1 − η = {:e} against {lowering:e} (levels {difference:e})", 1. - eta);
        }
    }

    /// The mean of the groups' values and its interval at the confidence z among `looks` reads: the standard error of the
    /// mean over the groups, the independent draws, with Student's t at one fewer (estimate.rs `Spread::interval_over`).
    fn read(groups: &[f64], looks: f64) -> (f64, f64) {
        let g = groups.len() as f64;
        let mean = groups.iter().sum::<f64>() / g;
        let error = (groups.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (g - 1.) / g).sqrt();
        (mean, crate::estimate::Spread { error, groups: g }.interval_over(looks))
    }

    /// The draws are the ensemble: 2¹⁶ of them in a shallow Lorentzian well (a = 3, the Wigner factor negligible) read the
    /// kinetic energy's mean (3/2) η k_BT and the depth's mean -- both by the quadrature, η's and the depth's own 1D rule --
    /// within their intervals over the groups. And the groups' means of the kinetic energy vary less than independent
    /// draws' would (the Latin hypercube): per group, the members' mean square deviation from the ensemble's mean over the
    /// group's size, less the square of the group mean's deviation, is on average the variance the strata take away, and
    /// it is read above zero beyond its interval. Three looks.
    #[test]
    fn the_draws_are_the_ensemble() {
        let a = 3.;
        let (radius, hm) = (1.3, 1e-9);
        let well = Lorentzian { depth: a * KT_PER_UK, radius, hbar_over_mass: hm };
        let ensemble = Ensemble::new(&[(5, 1., &well)], 1., hm).unwrap();
        let groups = 1 << 13;
        let (mut kinetic, mut depth) = (vec![], vec![]);
        for g in 0..groups {
            let members: Vec<Start> = (0..GROUP).map(|m| ensemble.draw(7, g * GROUP + m)).collect();
            assert!(members.iter().all(|s| s.state == 5));
            kinetic.push(members.iter().map(|s| 0.5 * hm * dot3(s.kappa, s.kappa) / KT_PER_UK).collect::<Vec<f64>>());
            depth.push(members.iter().map(|s| -well.potential(s.r).0 / KT_PER_UK).sum::<f64>() / GROUP as f64);
        }
        let looks = 3.;
        let means: Vec<f64> = kinetic.iter().map(|k| k.iter().sum::<f64>() / GROUP as f64).collect();
        let (mean, within) = read(&means, looks);
        let exact = 1.5 * lorentzian_eta(a);
        assert!((mean - exact).abs() <= within, "⟨K⟩ {mean} against {exact} ± {within}");
        let gains: Vec<f64> = kinetic.iter().zip(&means).map(|(k, m)| k.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (GROUP * GROUP) as f64 - (m - mean).powi(2)).collect();
        let (gain, within) = read(&gains, looks);
        assert!(gain - within > 0., "the strata take {gain} ± {within} off the variance of a group's mean");
        // The depth's mean by the 1D rule: ∫ y w(y) √u du / ∫ w(y) √u du.
        let sums = lorentzian(a);
        let (mean, within) = read(&depth, looks);
        assert!((mean - sums[2] / sums[0]).abs() <= within, "⟨y⟩ {mean} against {} ± {within}", sums[2] / sums[0]);
    }

    /// A standing wave in a Gaussian envelope whose antinode lies off the focus (a mirror's phase): V = −D cos²(k(z − z₀))
    /// /(1 + |r|²/R²), bound B = D by the beam's envelope, bottom near z₀ at a depth below B.
    struct Shifted {
        depth: f64,
        radius: f64,
        wave: f64,
        site: f64,
        hbar_over_mass: f64,
    }
    impl Shifted {
        /// The bottom on the axis by Newton from the antinode.
        fn bottom(&self) -> f64 {
            let mut z = self.site;
            for _ in 0..50 {
                let h = 1e-6 / self.wave;
                let g = |z: f64| self.potential([0., 0., z]).1[2];
                z -= g(z) * 2. * h / (g(z + h) - g(z - h));
            }
            z
        }
    }
    impl Well for Shifted {
        fn potential(&self, r: V3) -> (f64, V3) {
            let q = 1. + dot3(r, r) / (self.radius * self.radius);
            let (s, c) = (self.wave * (r[2] - self.site)).sin_cos();
            let mut gradient = r.map(|x| 2. * self.depth * c * c * x / (self.radius * self.radius * q * q));
            gradient[2] += 2. * self.depth * self.wave * s * c / q;
            (-self.depth * c * c / q, gradient)
        }
        fn trap(&self) -> (V3, [V3; 3], V3) {
            let z = self.bottom();
            let q = 1. + z * z / (self.radius * self.radius);
            let across = (self.hbar_over_mass * 2. * self.depth / (self.radius * self.radius * q * q)).sqrt();
            let along = (self.hbar_over_mass * 2. * self.depth * self.wave * self.wave / q).sqrt();
            ([0., 0., z], [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], [across, across, along])
        }
        fn envelope(&self) -> (Vec<Geometry>, f64) {
            (vec![Geometry::new([0., 0., 1.], 1. / self.radius, Some(2f64.sqrt() * self.radius))], self.depth)
        }
    }

    /// A cold atom in a well whose bottom is off the focus is drawn at the bottom: at 5000 k_BT of depth and an antinode
    /// 0.07 R from the focus (25 k_BT below the beams' bound, where the bound alone would keep one proposal in e^{12}),
    /// every draw lies in the bottom's site, the positions' spread across and along are the harmonic well's
    /// √(k_BT/λ_i) within the groups' interval, and no proposal reads deeper than the bottom.
    #[test]
    fn a_cold_atom_is_drawn_at_a_bottom_off_the_focus() {
        let hm = 1e-12;
        let well = Shifted { depth: 5000. * KT_PER_UK, radius: 1., wave: 20., site: -0.07, hbar_over_mass: hm };
        let ensemble = Ensemble::new(&[(0, 1., &well)], 1., hm).unwrap();
        let (centre, _, frequencies) = well.trap();
        let draws: Vec<Start> = (0..1 << 10).map(|i| ensemble.draw(11, i)).collect();
        assert_eq!(ensemble.deeper(), 0);
        assert!(draws.iter().all(|s| (s.r[2] - centre[2]).abs() < 0.5 * std::f64::consts::FRAC_PI_2 / well.wave));
        for axis in [0, 2] {
            let exact = KT_PER_UK * hm / (frequencies[axis] * frequencies[axis]);
            let squares: Vec<f64> = draws.chunks(GROUP as usize).map(|g| g.iter().map(|s| (s.r[axis] - centre[axis]).powi(2)).sum::<f64>() / GROUP as f64).collect();
            let (mean, within) = read(&squares, 2.);
            assert!((mean - exact).abs() <= within + 0.01 * exact, "axis {axis}: ⟨δr²⟩ {mean} against {exact} ± {within}");
        }
    }

    /// The Wigner factor: in a deep well with ħω = k_BT the momentum's second moment per lab axis is η k_BT (x/2) coth(x/2)
    /// (x = 1), the draws' mean within its interval over the groups; the map stretches a draw about the well's bottom, not
    /// the focus.
    #[test]
    fn the_wigner_factor_scales_the_draws() {
        let (radius, depth) = (0.9, 60. * KT_PER_UK);
        // ħω = k_BT at T = 1 µK: ω² = (ħ/m) 2D/R² = KT_PER_UK².
        let hm = KT_PER_UK * KT_PER_UK * radius * radius / (2. * depth);
        let well = Lorentzian { depth, radius, hbar_over_mass: hm };
        let ensemble = Ensemble::new(&[(0, 1., &well)], 1., hm).unwrap();
        let eta = ensemble.kinetic(u64::MAX)[0].eta;
        let factor = 0.5 / 0.5f64.tanh();
        let exact = ensemble.momentum(&[eta]);
        assert!((exact[2] - eta * KT_PER_UK * factor).abs() < 1e-12 * exact[2]);
        let draws: Vec<f64> = (0..1 << 15).map(|i| { let s = ensemble.draw(3, i); hm * s.kappa[2] * s.kappa[2] }).collect();
        let groups: Vec<f64> = draws.chunks(GROUP as usize).map(|members| members.iter().sum::<f64>() / GROUP as f64).collect();
        let (mean, within) = read(&groups, 1.);
        assert!((mean - exact[2]).abs() <= within, "{mean} against {} ± {within}", exact[2]);
    }
}
