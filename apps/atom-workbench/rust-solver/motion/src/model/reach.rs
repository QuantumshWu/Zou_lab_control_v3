//! The fold condition over the reach (DESIGN2 §2.4, §5.1, §5.3; HIFOLD §4; BRIEF_JUDGE item 3): a level folded in part or
//! whole is foldable where the RS series of its folded states converges at every place and momentum a held atom reaches
//! -- the two-level fold's radius, |V|/|Δ + iΓ/2| = ½, summed over the folded states of the level (CLUSTER: over the
//! folded m-states alone, not the whole level) --, a strict precondition of the selection (select.rs: a candidate that
//! fails it is not judged).
//!
//! # The quantity
//! Per excited level ℓ, its folded states E and a ground state g (all of them in P),
//!   s_ℓ,g(r, κ)² = Σ_ω Σ_{e∈E} Σ_± |Σ_{l∈ω} a_l(r) Y_l^±(e, g) / (Δ^±_eg,ω + s(r) σ_eg + δ_l(κ) + iγ_e)|²,
//! ω the frequencies of the light other than the tweezer's (its beams summed coherently, a_l their amplitudes),
//! Y⁺ = X_l[e, g] at Δ⁺ = E_g + f_ω − E_e and Y⁻ = X_l[g, e] at Δ⁻ = E_g − f_ω − E_e, E the bare levels (`Scene::dark`),
//! the denominators tuned by the tweezer's intensity s = |a_tw|² (σ = S_g − S_e, S its unit fold's static shift,
//! `Tweezer::still`), δ_l = (ħ/m) k_l·κ a moving atom's Doppler shift, γ_e = Γ_e/2. The level is foldable where
//! sup over the reach of max_g s_ℓ,g is below ½.
//!
//! # The reach
//! A held atom's energy is below its escape value, zero (E_cap = 0; the run's error account takes the same, DESIGN2
//! §2.3): its kinetic energy is below s U₀ (U₀ the deepest well of the prepared potential, every ground state's well the
//! tweezer's intensity times its value at the focus), so |κ| ≤ κ √s with κ = √(2U₀/(ħ/m)), and its place is anywhere --
//! s takes every value in [0, 1], s → 0 far from the focus, where the tweezer is off and the denominators bare.
//!
//! # The bound (strict)
//! Over s in an interval I = [s_a, s_b]:
//! - every term's weight 1/|Δ + sσ + iγ|² is at most its value at the least |Δ + sσ| over I (`weight`); the Doppler
//!   shifts δ_l ≤ (ħ/m)|k_l| κ √s_b enter by Minkowski's inequality, s ≤ √(F₀) + ‖e‖₂ with F₀ the sum without them and
//!   e_t = Σ_l |Y_lt| δ_l / (|d|(|d| − δ_l)), |d| ≥ √(min|Δ + sσ|² + γ²) (`doppler`);
//! - the frequencies do not interfere: F₀ ≤ Σ_ω sup over its own phases (each group's torus apart: an upper bound);
//! - a group of plane waves of one wave number has its beams' phases on a lattice, φ_l = n_l·θ, θ = k B r over a basis
//!   of the beams' direction differences (`lattice`, integer, by the Hermite normal form; CLUSTER's prototype,
//!   extremum2.py): F_ω(θ) = Σ_t w_t |Σ_l Y_lt e^{i n_l·θ}|² = Σ_q c_q e^{iq·θ} is a trigonometric polynomial and its sup
//!   is certified by branch and bound on boxes of θ: on a box of centre c and half-widths h,
//!   F ≤ F(c) + Σ_i |∂_iF(c)| h_i + ½ Σ_q |c_q| (Σ_i |q_i| h_i)² (`torus`);
//! - where s ≥ s_a > 0 the atom is within the tweezer's region of intensity s_a about its focus, so each θ_i = b_i·r is
//!   within |b_i·u| ζ(s_a) + |b_i⊥| ρ(s_a) of its focus value (zero): the large-s part of the bound is taken in real
//!   space, θ on that box rather than the whole torus (`span`);
//! - a group of one beam, or of beams on one direction, is its coherent sum exactly; a group with a Gaussian beam among
//!   several, or of directions without a rational lattice, by the triangle inequality, (Σ_l |Y_lt|)² (strict, larger);
//! - the intervals of s are halved where their bound is ½ or more, down to a width of `NARROWEST`; the condition holds
//!   where every interval's bound is below ½. The triangle inequality's bound (every group's phases free) is read
//!   first: where it is below ½ it is the bound, the lattices' not needed.
//! The bound is read once per folded set and ground state (`Foldability::bound`, kept).

use std::collections::HashMap;
use std::sync::Mutex;

use crate::linalg::{C64, V3};
use crate::request::{GROUND, STATES};
use crate::scene::Scene;

/// The narrowest interval of s the bound halves (`Foldability::bound`).
const NARROWEST: f64 = 1.0 / 4096.0;
/// The initial intervals of s.
const INTERVALS: usize = 16;
/// The most boxes one torus's certificate evaluates (`torus`): beyond, its bound is the largest box's.
const BOXES: usize = 1 << 15;

/// A group's phases (module documentation, # The bound): on a lattice -- its torus's coordinates' rows b_i (θ_i = b_i·r,
/// 1/µm) and per beam its integer vector n_l --, or bounded by the triangle inequality.
#[derive(Clone, Debug)]
enum Phases {
    Lattice { rows: Vec<V3>, n: Vec<Vec<i64>> },
    Triangle,
}

/// A frequency of the light other than the tweezer's: its frequency (rad/µs), its beams (the request's), their phases,
/// per beam (ħ/m)|k_l| (its Doppler shift per unit momentum).
#[derive(Clone, Debug)]
struct Group {
    frequency: f64,
    beams: Vec<usize>,
    phases: Phases,
    doppler: Vec<f64>,
}

/// One term of s² (module documentation, # The quantity): its group, per beam of the group Y_lt, Δ, σ, γ.
#[derive(Clone, Debug)]
struct Term {
    group: usize,
    y: Vec<C64>,
    detuning: f64,
    slope: f64,
    width: f64,
}

/// The fold condition's view of a scene (module documentation).
pub struct Foldability {
    energy: Vec<f64>,
    shift: Vec<f64>,
    width: Vec<f64>,
    groups: Vec<Group>,
    /// Per beam its couplings X_l (32 × 32).
    couplings: Vec<crate::linalg::Mat>,
    /// κ (1/µm): the largest momentum at the focus, κ_max(s) = κ √s.
    kappa: f64,
    /// The tweezer's direction, waist and Rayleigh range (µm).
    tweezer: (V3, f64, f64),
    /// The excited levels: each its states (indices into the 32), its name.
    levels: Vec<(String, Vec<usize>)>,
    /// The bounds read: per folded set and ground state.
    kept: Mutex<HashMap<(Vec<usize>, usize), f64>>,
    /// The states' labels (the report's).
    labels: Vec<String>,
}

/// The integer row Hermite normal form of A (its rows a basis of the lattice A's rows span).
fn hermite(mut a: Vec<Vec<i64>>) -> Vec<Vec<i64>> {
    let (m, n) = (a.len(), a.first().map_or(0, Vec::len));
    let mut pivot = 0;
    for j in 0..n {
        if pivot >= m {
            break;
        }
        loop {
            let Some(i0) = (pivot..m).filter(|&i| a[i][j] != 0).min_by_key(|&i| a[i][j].abs()) else { break };
            a.swap(pivot, i0);
            let others: Vec<usize> = (pivot + 1..m).filter(|&i| a[i][j] != 0).collect();
            if others.is_empty() {
                break;
            }
            for i in others {
                let q = a[i][j].div_euclid(a[pivot][j]);
                let row = a[pivot].clone();
                for (x, r) in a[i].iter_mut().zip(&row) {
                    *x -= q * r;
                }
            }
        }
        if a[pivot][j] != 0 {
            if a[pivot][j] < 0 {
                a[pivot].iter_mut().for_each(|x| *x = -*x);
            }
            pivot += 1;
        }
    }
    a.truncate(pivot);
    a
}

/// The best rational approximation p/q of x with q ≤ 48 (continued fractions), where it is within 10⁻⁹ of x.
fn rational(x: f64) -> Option<(i64, i64)> {
    let (mut h0, mut h1, mut k0, mut k1) = (0_i64, 1_i64, 1_i64, 0_i64);
    let mut y = x;
    for _ in 0..40 {
        let a = y.floor();
        let (h2, k2) = (a as i64 * h1 + h0, a as i64 * k1 + k0);
        if k2 > 48 {
            break;
        }
        (h0, h1, k0, k1) = (h1, h2, k1, k2);
        if (h1 as f64 / k1 as f64 - x).abs() <= 1e-12 || (y - a).abs() < 1e-15 {
            break;
        }
        y = 1.0 / (y - a);
    }
    ((h1 as f64 / k1 as f64 - x).abs() <= 1e-9).then_some((h1, k1))
}

/// The lattice of a group's plane waves of one wave number k (module documentation, # The bound): the beams' directions
/// u_l, their differences to the first u_l − u_0 over an integer basis -- rows B (d × 3, dimensionless) with
/// u_l − u_0 = n_l·B --, so that the phases relative to the first beam's are n_l·θ, θ = k B r. None where a difference
/// has no rational coordinates (denominators to 48) on a basis of them.
fn lattice(directions: &[V3]) -> Option<(Vec<V3>, Vec<Vec<i64>>)> {
    let gens: Vec<V3> = directions[1..].iter().map(|u| std::array::from_fn(|a| u[a] - directions[0][a])).collect();
    // A basis among the differences (Gram–Schmidt on them in turn).
    let mut basis: Vec<V3> = Vec::new();
    let mut ortho: Vec<V3> = Vec::new();
    for g in &gens {
        let mut v = *g;
        for o in &ortho {
            let p: f64 = (0..3).map(|a| v[a] * o[a]).sum();
            v = std::array::from_fn(|a| v[a] - p * o[a]);
        }
        let size = crate::linalg::norm3(v);
        if size > 1e-9 * crate::linalg::norm3(*g).max(1.0) {
            ortho.push(v.map(|x| x / size));
            basis.push(*g);
        }
    }
    let d0 = basis.len();
    if d0 == 0 {
        return Some((Vec::new(), vec![Vec::new(); directions.len()]));
    }
    // Each difference's coordinates on the basis (the normal equations, d0 ≤ 3), rational.
    let gram: Vec<Vec<f64>> = (0..d0).map(|i| (0..d0).map(|j| (0..3).map(|a| basis[i][a] * basis[j][a]).sum()).collect()).collect();
    let solve = |b: Vec<f64>| -> Vec<f64> {
        let mut m: Vec<Vec<f64>> = gram.iter().zip(&b).map(|(r, &x)| r.iter().copied().chain(std::iter::once(x)).collect()).collect();
        for k in 0..d0 {
            let p = (k..d0).max_by(|&i, &j| m[i][k].abs().total_cmp(&m[j][k].abs())).unwrap();
            m.swap(k, p);
            for i in 0..d0 {
                if i != k {
                    let f = m[i][k] / m[k][k];
                    for j in k..=d0 {
                        m[i][j] -= f * m[k][j];
                    }
                }
            }
        }
        (0..d0).map(|k| m[k][d0] / m[k][k]).collect()
    };
    let mut fractions: Vec<Vec<(i64, i64)>> = Vec::new();
    for g in &gens {
        let c = solve((0..d0).map(|i| (0..3).map(|a| basis[i][a] * g[a]).sum()).collect());
        let back: V3 = std::array::from_fn(|a| (0..d0).map(|i| c[i] * basis[i][a]).sum());
        if (0..3).any(|a| (back[a] - g[a]).abs() > 1e-9) {
            return None;
        }
        fractions.push(c.iter().map(|&x| rational(x)).collect::<Option<Vec<_>>>()?);
    }
    let gcd = |mut a: i64, mut b: i64| {
        (a, b) = (a.abs(), b.abs());
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    };
    let common = fractions.iter().flatten().fold(1_i64, |l, &(_, q)| l / gcd(l, q) * q);
    let integers: Vec<Vec<i64>> = fractions.iter().map(|f| f.iter().map(|&(p, q)| p * (common / q)).collect()).collect();
    let h = hermite(integers.clone());
    if h.len() != d0 {
        return None;
    }
    let rows: Vec<V3> = h.iter().map(|r| std::array::from_fn(|a| (0..d0).map(|i| r[i] as f64 / common as f64 * basis[i][a]).sum())).collect();
    // n_l with n_l·H = integers_l (H square, d0 × d0).
    let hf: Vec<Vec<f64>> = h.iter().map(|r| r.iter().map(|&x| x as f64).collect()).collect();
    let mut n = vec![vec![0_i64; d0]];
    for (row, g) in integers.iter().zip(&gens) {
        // Solve x H = row: Hᵀ xᵀ = rowᵀ by elimination.
        let mut m: Vec<Vec<f64>> = (0..d0).map(|j| (0..d0).map(|i| hf[i][j]).chain(std::iter::once(row[j] as f64)).collect()).collect();
        for k in 0..d0 {
            let p = (k..d0).max_by(|&i, &j| m[i][k].abs().total_cmp(&m[j][k].abs())).unwrap();
            m.swap(k, p);
            for i in 0..d0 {
                if i != k {
                    let f = m[i][k] / m[k][k];
                    for j in k..=d0 {
                        m[i][j] -= f * m[k][j];
                    }
                }
            }
        }
        let x: Vec<f64> = (0..d0).map(|k| m[k][d0] / m[k][k]).collect();
        let xi: Vec<i64> = x.iter().map(|v| v.round() as i64).collect();
        if x.iter().zip(&xi).any(|(v, r)| (v - *r as f64).abs() > 1e-9) {
            return None;
        }
        let back: V3 = std::array::from_fn(|a| (0..d0).map(|i| xi[i] as f64 * rows[i][a]).sum());
        if (0..3).any(|a| (back[a] - g[a]).abs() > 1e-9) {
            return None;
        }
        n.push(xi);
    }
    Some((rows, n))
}

/// The least |Δ + sσ| over s in [a, b] (affine in s).
fn least(detuning: f64, slope: f64, a: f64, b: f64) -> f64 {
    let (x, y) = (detuning + a * slope, detuning + b * slope);
    if x * y <= 0.0 { 0.0 } else { x.abs().min(y.abs()) }
}

/// A box of a torus's coordinates: its centre and half-widths.
struct Box {
    centre: Vec<f64>,
    half: Vec<f64>,
    bound: f64,
}

/// A lattice group's F(θ) = Σ_q c_q e^{iq·θ} on its frequencies q (`Phases::Lattice`): its value and gradient.
fn value(q: &[(Vec<i64>, C64)], theta: &[f64]) -> (f64, Vec<f64>) {
    let mut f = 0.0;
    let mut g = vec![0.0; theta.len()];
    for (k, c) in q {
        let phase: f64 = k.iter().zip(theta).map(|(&a, t)| a as f64 * t).sum();
        let z = *c * C64::cis(phase);
        f += z.re;
        for (gi, &ki) in g.iter_mut().zip(k) {
            *gi -= ki as f64 * z.im;
        }
    }
    (f, g)
}

/// The certified sup of a lattice group's F over θ in the box of half-widths `span` about zero (π: the whole torus;
/// module documentation, # The bound): boxes of θ, each bounded by F(c) + Σ_i |∂_iF(c)| h_i + ½ Σ_q |c_q|(Σ_i |q_i|h_i)²,
/// the largest halved until it is within 10⁻³ of the largest value met (or `BOXES` are made).
fn torus(q: &[(Vec<i64>, C64)], span: &[f64]) -> f64 {
    let d = span.len();
    let bound = |centre: &[f64], half: &[f64]| -> (f64, f64) {
        let (f, g) = value(q, centre);
        let linear: f64 = g.iter().zip(half).map(|(gi, h)| gi.abs() * h).sum();
        let second: f64 = q.iter().map(|(k, c)| c.abs() * k.iter().zip(half).map(|(&a, h)| a.abs() as f64 * h).sum::<f64>().powi(2)).sum();
        (f, f + linear + 0.5 * second)
    };
    let per = 8usize;
    let mut boxes: Vec<Box> = Vec::new();
    let mut best = f64::NEG_INFINITY;
    let count = per.pow(d as u32);
    for index in 0..count {
        let mut rest = index;
        let centre: Vec<f64> = (0..d)
            .map(|i| {
                let j = rest % per;
                rest /= per;
                -span[i] + (2 * j + 1) as f64 * span[i] / per as f64
            })
            .collect();
        let half: Vec<f64> = span.iter().map(|s| s / per as f64).collect();
        let (f, b) = bound(&centre, &half);
        best = best.max(f);
        boxes.push(Box { centre, half, bound: b });
    }
    let mut made = count;
    loop {
        let (top, at) = boxes.iter().enumerate().fold((f64::NEG_INFINITY, 0), |(t, a), (i, b)| if b.bound > t { (b.bound, i) } else { (t, a) });
        if top <= best + 1e-3 * best.abs() || made >= BOXES || !top.is_finite() {
            return top;
        }
        let b = boxes.swap_remove(at);
        boxes.retain(|x| x.bound > best);
        for corner in 0..(1usize << d) {
            let half: Vec<f64> = b.half.iter().map(|h| 0.5 * h).collect();
            let centre: Vec<f64> = (0..d).map(|i| b.centre[i] + if corner >> i & 1 == 1 { half[i] } else { -half[i] }).collect();
            let (f, bd) = bound(&centre, &half);
            best = best.max(f);
            made += 1;
            boxes.push(Box { centre, half, bound: bd });
        }
    }
}

impl Foldability {
    /// The fold condition's view of a scene (module documentation): its bare levels, the tweezer's unit fold's static
    /// shifts, the decay rates, the light other than the tweezer's by frequency with each group's phases, and the
    /// reach's κ from the prepared wells' deepest.
    pub fn new(scene: &Scene) -> Foldability {
        let request = &scene.request;
        let dark = scene.dark();
        let still = crate::model::rs::Tweezer::new(scene).still();
        let decay = scene.jumps().decay();
        let energy: Vec<f64> = (0..STATES).map(|s| dark[(s, s)].re).collect();
        let shift: Vec<f64> = (0..STATES).map(|s| still[(s, s)].re).collect();
        let width: Vec<f64> = (0..STATES).map(|s| 0.5 * decay[(s, s)].re).collect();
        let hbar_over_mass = crate::units::hbar_over_mass(request.mass_kg);
        let depth: [f64; GROUND] = std::array::from_fn(|g| still[(g, g)].re);
        let wells = scene.trap_potential([0.0; 3], &depth);
        let deepest = wells.iter().map(|q| -q.value).fold(0.0, crate::linalg::larger);
        let kappa = (2.0 * deepest / hbar_over_mass).sqrt();
        let tw = scene.tweezer();
        let mut groups: Vec<Group> = Vec::new();
        for (l, beam) in request.beams.iter().enumerate().filter(|(l, _)| *l != tw) {
            let doppler = hbar_over_mass * scene.geometry[l].wave_number;
            match groups.iter_mut().find(|g| request.beams[g.beams[0]].frequency_index == beam.frequency_index) {
                Some(g) => {
                    g.beams.push(l);
                    g.doppler.push(doppler);
                }
                None => groups.push(Group { frequency: scene.generators[l].frequency, beams: vec![l], phases: Phases::Triangle, doppler: vec![doppler] }),
            }
        }
        for g in groups.iter_mut() {
            let geometry: Vec<&crate::field::Geometry> = g.beams.iter().map(|&l| &scene.geometry[l]).collect();
            let k = geometry[0].wave_number;
            let plane = g.beams.len() == 1 || geometry.iter().all(|x| x.waist.is_none() && (x.wave_number - k).abs() <= 1e-12 * k);
            if plane {
                if let Some((rows, n)) = lattice(&geometry.iter().map(|x| x.direction).collect::<Vec<_>>()) {
                    g.phases = Phases::Lattice { rows: rows.iter().map(|r| r.map(|x| k * x)).collect(), n };
                }
            }
        }
        let couplings = request.beams.iter().map(|b| b.coupling.clone()).collect();
        let t = &scene.geometry[tw];
        let all: Vec<usize> = (0..STATES).collect();
        let levels = crate::model::rs::levels(&request.states, &all)
            .into_iter()
            .filter(|l| l[0] >= GROUND)
            .map(|l| {
                let name = request.states[l[0]].rsplit_once('-').map_or(request.states[l[0]].clone(), |(a, _)| a.trim_end_matches('-').to_string());
                (name, l)
            })
            .collect();
        Foldability {
            energy,
            shift,
            width,
            groups,
            couplings,
            kappa,
            tweezer: (t.direction, t.waist.unwrap_or(f64::INFINITY), t.rayleigh().unwrap_or(f64::INFINITY)),
            levels,
            kept: Mutex::new(HashMap::new()),
            labels: request.states.clone(),
        }
    }

    /// Whether every excited level's states that `keep` (indices into the 32) leaves out are foldable: Err with the
    /// first level that is not, its folded states and its bound.
    pub fn check(&self, keep: &[usize]) -> Result<(), String> {
        for (name, states) in &self.levels {
            let folded: Vec<usize> = states.iter().copied().filter(|s| !keep.contains(s)).collect();
            if folded.is_empty() {
                continue;
            }
            let bounds: Vec<f64> = (0..GROUND).map(|g| self.bound(&folded, g)).collect();
            let g = (0..GROUND).fold(0, |b, i| if bounds[i] > bounds[b] || bounds[i].is_nan() { i } else { b });
            if !(bounds[g] < 0.5) {
                let states = folded.iter().map(|&s| self.labels[s].as_str()).collect::<Vec<_>>().join("、");
                return Err(format!("{name} 折掉 {states} 不满足可折条件：从 {} 起 s ≤ {:.4}，不在 ½ 之下", self.labels[g], bounds[g]));
            }
        }
        Ok(())
    }

    /// The bound of sup over the reach of s_ℓ,g (module documentation, # The bound) for the folded states `folded` and
    /// the ground state g: the largest over the intervals of s, those of a bound of ½ or more halved down to
    /// `NARROWEST`. Kept per set and ground state.
    pub fn bound(&self, folded: &[usize], g: usize) -> f64 {
        if let Some(&b) = self.kept.lock().unwrap_or_else(|e| e.into_inner()).get(&(folded.to_vec(), g)) {
            return b;
        }
        let terms = self.terms(folded, g);
        // The triangle inequality's bound first (every group's phases free of each other): where it is below ½, it is
        // the bound; else the lattices'.
        let first = (0..INTERVALS).map(|i| self.over(&terms, i as f64 / INTERVALS as f64, (i + 1) as f64 / INTERVALS as f64, false)).fold(0.0, crate::linalg::larger);
        if first < 0.5 {
            self.kept.lock().unwrap_or_else(|e| e.into_inner()).insert((folded.to_vec(), g), first);
            return first;
        }
        let mut intervals: Vec<(f64, f64, f64)> = (0..INTERVALS)
            .map(|i| {
                let (a, b) = (i as f64 / INTERVALS as f64, (i + 1) as f64 / INTERVALS as f64);
                (a, b, self.over(&terms, a, b, true))
            })
            .collect();
        loop {
            let (at, worst) = intervals.iter().enumerate().fold((0, f64::NEG_INFINITY), |(i, w), (j, x)| if x.2 > w { (j, x.2) } else { (i, w) });
            let (a, b, _) = intervals[at];
            if worst < 0.5 || b - a <= NARROWEST || !worst.is_finite() {
                self.kept.lock().unwrap_or_else(|e| e.into_inner()).insert((folded.to_vec(), g), worst);
                return worst;
            }
            let m = 0.5 * (a + b);
            intervals[at] = (a, m, self.over(&terms, a, m, true));
            intervals.push((m, b, self.over(&terms, m, b, true)));
        }
    }

    /// The terms of s_ℓ,g² (module documentation, # The quantity) of the folded states and the ground state g.
    fn terms(&self, folded: &[usize], g: usize) -> Vec<Term> {
        let mut out = Vec::new();
        for (gi, group) in self.groups.iter().enumerate() {
            for &e in folded {
                for sign in [1.0, -1.0] {
                    let y: Vec<C64> = group.beams.iter().map(|&l| if sign > 0.0 { self.couplings[l][(e, g)] } else { self.couplings[l][(g, e)] }).collect();
                    if y.iter().all(|z| *z == C64::ZERO) {
                        continue;
                    }
                    out.push(Term { group: gi, y, detuning: self.energy[g] + sign * group.frequency - self.energy[e], slope: self.shift[g] - self.shift[e], width: self.width[e] });
                }
            }
        }
        out
    }

    /// The bound of s_ℓ,g over s in [a, b] (module documentation, # The bound): √(F₀) + ‖e‖₂, the groups' lattices
    /// read where `lattices` (else every group by the triangle inequality).
    fn over(&self, terms: &[Term], a: f64, b: f64, lattices: bool) -> f64 {
        let span = self.span(a);
        let mut f = 0.0;
        let mut e2 = 0.0;
        for (gi, group) in self.groups.iter().enumerate() {
            let mine: Vec<&Term> = terms.iter().filter(|t| t.group == gi).collect();
            if mine.is_empty() {
                continue;
            }
            let weights: Vec<f64> = mine.iter().map(|t| 1.0 / (least(t.detuning, t.slope, a, b).powi(2) + t.width * t.width)).collect();
            for (t, w) in mine.iter().zip(&weights) {
                let d = w.sqrt().recip();
                let mut e = 0.0;
                for (y, dl) in t.y.iter().zip(&group.doppler) {
                    let delta = dl * self.kappa * b.sqrt();
                    if !(delta < d) {
                        return f64::INFINITY;
                    }
                    e += y.abs() * delta / (d * (d - delta));
                }
                e2 += e * e;
            }
            match if lattices { &group.phases } else { &Phases::Triangle } {
                Phases::Triangle => f += mine.iter().zip(&weights).map(|(t, w)| w * t.y.iter().map(|y| y.abs()).sum::<f64>().powi(2)).sum::<f64>(),
                Phases::Lattice { rows, .. } if rows.is_empty() => f += mine.iter().zip(&weights).map(|(t, w)| w * t.y.iter().copied().sum::<C64>().norm_sqr()).sum::<f64>(),
                Phases::Lattice { rows, n } => {
                    // c_q = Σ_t w_t Σ_{n_m − n_l = q} conj(Y_lt) Y_mt.
                    let mut q: Vec<(Vec<i64>, C64)> = Vec::new();
                    for (t, w) in mine.iter().zip(&weights) {
                        for (l, yl) in t.y.iter().enumerate() {
                            for (m, ym) in t.y.iter().enumerate() {
                                let k: Vec<i64> = n[m].iter().zip(&n[l]).map(|(x, y)| x - y).collect();
                                let c = yl.conj() * *ym * *w;
                                match q.iter_mut().find(|x| x.0 == k) {
                                    Some(x) => x.1 += c,
                                    None => q.push((k, c)),
                                }
                            }
                        }
                    }
                    let widths: Vec<f64> = rows.iter().map(|r| span.map_or(std::f64::consts::PI, |(along, across)| {
                        let (u, _, _) = self.tweezer;
                        let p: f64 = (0..3).map(|i| r[i] * u[i]).sum();
                        let perp = (crate::linalg::norm3(*r).powi(2) - p * p).max(0.0).sqrt();
                        (p.abs() * along + perp * across).min(std::f64::consts::PI)
                    })).collect();
                    f += torus(&q, &widths);
                }
            }
        }
        f.max(0.0).sqrt() + e2.sqrt()
    }

    /// The tweezer's region of intensity at least s_a > 0 about its focus (module documentation, # The bound): the
    /// largest distance along its axis z_R √(1/s_a − 1) and across it w₀ √(ρ²/w₀²), ρ² ≤ (w₀²/2) max over the axis of
    /// (1 + u²) ln(1/(s_a(1 + u²))) -- 1/(e s_a) where s_a ≤ 1/e, else ln(1/s_a). None at s_a = 0 (anywhere).
    fn span(&self, a: f64) -> Option<(f64, f64)> {
        if !(a > 0.0) {
            return None;
        }
        let (_, w, zr) = self.tweezer;
        let along = zr * (1.0 / a - 1.0).max(0.0).sqrt();
        let rho2 = 0.5 * w * w * if a <= (-1.0_f64).exp() { 1.0 / (std::f64::consts::E * a) } else { (1.0 / a).ln() };
        Some((along, rho2.sqrt()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The lattice of six counter-propagating plane waves along x, y, z (PGC3D's): three coordinates, every difference
    /// an integer vector on them; of two along ±z (R4's), one.
    #[test]
    fn the_lattices_of_the_standing_waves() {
        let six: Vec<V3> = vec![[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]];
        let (rows, n) = lattice(&six).expect("a lattice");
        assert_eq!(rows.len(), 3);
        for (u, nl) in six.iter().zip(&n) {
            let back: V3 = std::array::from_fn(|a| six[0][a] + (0..3).map(|i| nl.get(i).copied().unwrap_or(0) as f64 * rows[i][a]).sum::<f64>());
            assert!((0..3).all(|a| (back[a] - u[a]).abs() < 1e-12), "{u:?} {back:?}");
        }
        let (rows, _) = lattice(&[[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]).expect("a lattice");
        assert_eq!(rows.len(), 1);
    }

    /// The fold condition on the six scenes (a measurement; ignored, run by name; V3D_REQ, PSTATE_SCENE one of them): per
    /// scene the P chosen with it, the sets it left out and why, and per excited level P folds in part or whole its
    /// bound (the largest over the ground states) with the time it took.
    #[test]
    #[ignore]
    fn fold_condition_on_the_six_scenes() {
        for (name, request) in crate::select::tests::requests() {
            let scene = Scene::new(&request).unwrap();
            let clock = std::time::Instant::now();
            let fold = Foldability::new(&scene);
            let made = clock.elapsed().as_secs_f64();
            let selection = crate::select::choose(&scene).unwrap();
            let labels = &scene.request.states;
            eprintln!("FOLD {name}: P {} ({:.2} s); the view made in {:.3} s; groups {:?}", selection.keep.len(), selection.seconds, made, fold.groups.iter().map(|g| (g.beams.len(), match &g.phases { Phases::Lattice { rows, .. } => rows.len() as i64, Phases::Triangle => -1 })).collect::<Vec<_>>());
            for (keep, why) in &selection.excluded {
                eprintln!("FOLD {name}: left out P {}: {why}", keep.len());
            }
            for (level, states) in &fold.levels {
                let folded: Vec<usize> = states.iter().copied().filter(|s| !selection.keep.contains(s)).collect();
                if folded.is_empty() {
                    continue;
                }
                let clock = std::time::Instant::now();
                let bounds: Vec<f64> = (0..GROUND).map(|g| fold.bound(&folded, g)).collect();
                let g = (0..GROUND).fold(0, |b, i| if bounds[i] > bounds[b] { i } else { b });
                eprintln!("FOLD {name} {level}: folded {:?} ({}/{}): s ≤ {:.4} at {} ({:.3} s)", folded.iter().map(|&s| labels[s].as_str()).collect::<Vec<_>>(), folded.len(), states.len(), bounds[g], labels[g], clock.elapsed().as_secs_f64());
            }
        }
    }

    /// The torus's certificate against a dense grid: a two-beam standing wave's |1 + 0.6 e^{iθ}|² + 0.3|1 − e^{iθ}|²
    /// has its sup 2.56 at θ = 0; the certificate is above every grid value and within 10⁻⁵ of the sup.
    #[test]
    fn the_torus_bound_is_the_sup() {
        let q = vec![(vec![0], C64::real(1.0 + 0.36 + 0.3 * 2.0)), (vec![1], C64::real(0.6 - 0.3)), (vec![-1], C64::real(0.6 - 0.3))];
        let b = torus(&q, &[std::f64::consts::PI]);
        let grid = (0..10000).map(|i| value(&q, &[-std::f64::consts::PI + i as f64 * 2.0 * std::f64::consts::PI / 10000.0]).0).fold(f64::NEG_INFINITY, f64::max);
        assert!(b >= grid && b - grid <= 1e-5 * grid, "{b} against {grid}");
    }
}
