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
//! (`Component::pieces`), a point from it, and keeps it with probability w(y(r))/Σ_l E(u_l(r)) ≤ 1. The momentum at depth y:
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
//! relative to T (K's mean is (3/2) kept(5/2)/kept(3/2) at each r). η by quadrature (`Ensemble::kinetic`): per beam of the
//! envelope its share (1 + u_l)⁻¹/Σ_l′(1 + u_l′)⁻¹ of the integrand (a smooth partition of unity) in its whitened spherical
//! coordinates -- Gauss–Legendre in cos θ, the trapezoid rule in the azimuth (spectral for a periodic integrand), Gauss–Legendre
//! in τ with u = τ²/(1 − τ²), where the integrand V_l τ²(1 − τ²)^{−5/2} kept(s, y) is smooth at both ends (kept ∝ y^{s+1} as
//! y → 0, and y ≤ a(1 − τ²)) --, every node count doubled from the least that reads a curvature (two per coordinate)
//! until two levels agree within their sums' rounding.
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
use std::f64::consts::{PI, TAU};

/// The trajectories of one Latin-hypercube group (SPEC §2.5), and with it the block that the delete-one-group jackknife
/// leaves out (estimate.rs). A structural size (SPEC §0), which no result's expectation depends on: every member is
/// distributed as an independent draw and the groups are independent of each other, so at any size the estimates are
/// unbiased and the jackknife's variance is that of the groups' means. At a given number of trajectories the size only
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

/// Gauss–Legendre's nodes and weights on [−1, 1]: Newton's iteration on P_n from Tricomi's start, to round-off.
fn legendre(n: usize) -> Vec<(f64, f64)> {
    (0..n)
        .map(|i| {
            let mut x = (PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
            loop {
                let (mut p0, mut p1) = (1., x);
                for k in 2..=n {
                    let p2 = ((2 * k - 1) as f64 * x * p1 - (k - 1) as f64 * p0) / k as f64;
                    p0 = p1;
                    p1 = p2;
                }
                let slope = n as f64 * (x * p1 - p0) / (x * x - 1.);
                let dx = p1 / slope;
                x -= dx;
                if dx.abs() <= 2. * f64::EPSILON {
                    return (x, 2. / ((1. - x * x) * slope * slope));
                }
            }
        })
        .collect()
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

/// One initial component: its basis state and its share, its well, the envelope's top a = B/k_BT and beams, the cumulative
/// masses of its pieces (per beam the core, then the tail; module documentation, # Its draw), and its well's bottom, axes,
/// curvatures λ_i = ω_i²/(ħ/m) and Wigner scales s_i.
struct Component<'a> {
    state: usize,
    weight: f64,
    well: &'a dyn Well,
    top: f64,
    modes: Vec<Mode>,
    pieces: Vec<f64>,
    centre: V3,
    axes: [V3; 3],
    curvatures: V3,
    scales: V3,
}

impl Component<'_> {
    /// Σ_l E(u_l(r)) e^{−a} (module documentation, # Its draw).
    fn envelope(&self, r: V3) -> f64 {
        let a = self.top;
        self.modes
            .iter()
            .map(|m| {
                let u = m.coordinate(r);
                if u <= 1. { (-0.5 * a * u).exp() } else { (-0.5 * a).exp() * a.powf(2.5) * u.powf(-2.5) / gamma(3.5) }
            })
            .sum()
    }
    /// A proposal from the piece of the deviate `piece`, its radial coordinate and direction from the next three.
    fn propose(&self, piece: f64, radial: f64, mu: f64, phi: f64) -> V3 {
        let total = self.pieces[self.pieces.len() - 1];
        let j = self.pieces.partition_point(|&c| c <= piece * total).min(self.pieces.len() - 1);
        let (mode, a) = (&self.modes[j / 2], self.top);
        let size = if j % 2 == 0 {
            // The core: x = a u/2 with density ∝ x^{1/2} e^{−x} on [0, a/2], P(3/2, x) = radial P(3/2, a/2), in v = x^{3/2}.
            let target = radial * regularized(1.5, 0.5 * a);
            let x = root(|v| { let x = v.powf(2. / 3.); (regularized(1.5, x), (-x).exp() / gamma(2.5)) }, target, 0., (0.5 * a).powf(1.5)).powf(2. / 3.);
            (2. * x / a).sqrt()
        } else {
            // The tail: |ξ| with density ∝ |ξ|⁻³ on (1, ∞).
            (1. - radial).powf(-0.5)
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

/// The initial ensemble of a request (module documentation).
pub struct Ensemble<'a> {
    components: Vec<Component<'a>>,
    kt: f64,
    hbar_over_mass: f64,
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
                let core = (TAU / a).powf(1.5) * regularized(1.5, 0.5 * a);
                let tail = (-0.5 * a).exp() * a.powf(2.5) / gamma(3.5) * TAU;
                let mut pieces = Vec::with_capacity(2 * modes.len());
                let mut sum = 0.;
                for m in &modes {
                    for mass in [core, tail] {
                        sum += mass * m.volume();
                        pieces.push(sum);
                    }
                }
                let (centre, axes, frequencies) = well.trap();
                let curvatures = frequencies.map(|w| w * w / hbar_over_mass);
                let scales = frequencies.map(|w| {
                    let half = 0.5 * w / kt;
                    (half / half.tanh()).sqrt()
                });
                Ok(Component { state, weight: p / total, well, top: a, modes, pieces, centre, axes, curvatures, scales })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Ensemble { components, kt, hbar_over_mass })
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
        let (r, y, gradient) = loop {
            let r = c.propose(proposal[0], proposal[1], proposal[2], proposal[3]);
            let (v, gradient) = c.well.potential(r);
            let y = -v / self.kt;
            if y > 0. && own.open_uniform() * c.envelope(r) <= kept(1.5, y, y, c.top) {
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
                    if within || (2 * n as u64).pow(3) * c.modes.len() as u64 > most {
                        break read;
                    }
                }
            })
            .collect()
    }
    /// ∫ kept(3/2, y) d³r and ∫ kept(5/2, y) d³r (at shift a), with n nodes per coordinate per beam, and the evaluations.
    fn integrals(&self, c: &Component, n: usize) -> ([f64; 2], usize) {
        let nodes = legendre(n);
        let rows: Vec<[f64; 2]> = (0..c.modes.len() * n)
            .into_par_iter()
            .map(|row| {
                let (mode, (mu, w_mu)) = (&c.modes[row / n], nodes[row % n]);
                let mut sums = [0.; 2];
                for j in 0..n {
                    let unit = direction(mu, TAU * (j as f64 + 0.5) / n as f64);
                    for &(z, w_z) in &nodes {
                        let tau = 0.5 * (z + 1.);
                        let t = tau * tau;
                        let r = mode.point(unit.map(|x| x * (t / (1. - t)).sqrt()));
                        let y = -c.well.potential(r).0 / self.kt;
                        if y > 0. {
                            let own = 1. / (1. + mode.coordinate(r));
                            let share = own / c.modes.iter().map(|m| 1. / (1. + m.coordinate(r))).sum::<f64>();
                            let weight = mode.volume() * w_mu * (TAU / n as f64) * 0.5 * w_z * t * (1. - t).powf(-2.5) * share;
                            sums[0] += weight * kept(1.5, y, y, c.top);
                            sums[1] += weight * kept(2.5, y, y, c.top);
                        }
                    }
                }
                sums
            })
            .collect();
        // The rows summed in their order: the result does not depend on the threads.
        (rows.iter().fold([0.; 2], |s, x| [s[0] + x[0], s[1] + x[1]]), c.modes.len() * n * n * n)
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
        let nodes = legendre(64);
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
