//! The atoms' reach (SPEC §1.8) and provable extrema over it.
//!
//! # The reach
//! The sublevel set {r : V_prep(r; p) ≤ V_min + E_r} of the prepared potential, E_r = ln(1/s) k_B T_r, with the momenta of at most
//! that kinetic energy, |v| ≤ √(2 (ħ/m) E_r) (the energy above V_min bounds the kinetic one everywhere in it). Its level
//! V_min + E_r is below the far value 0 where the trap holds the reach; there −V_prep ≥ −(V_min + E_r) > 0 on it, and the proven
//! envelope −V_prep ≤ B max_l A_l(r) (model.rs `Envelope`) puts it inside the union of the ellipsoids u_l ≤ B/(−V_min − E_r) − 1
//! of the trap light's focused beams, u_l = xᵀ M_l x, M_l = (2/w₀²)(1 − nnᵀ) + nnᵀ/z_R²: their lab-axis extents √(U (M_l⁻¹)_aa),
//! M_l⁻¹ = (w₀²/2)(1 − nnᵀ) + z_R² nnᵀ, bound `Reach::cell`. A level at or above the far value -- or a plane wave in the trap
//! light -- leaves the reach unbounded: the whole space, its bounds the global ones.
//!
//! # Provable extrema (`Model::bound_over`)
//! A quantity's bound over the reach is the largest of its upper bounds over cells covering the reach's box, a cell taken where
//! it may meet the reach (a lower bound of V_prep over it at most the level). Each upper bound comes from the cell's centre and
//! the beams' Lipschitz constants (`lipschitz`): |a_l(r) − a_l(c)| ≤ L_l |r − c| and |a_l| ≤ 1. The cell of the largest bound is
//! halved along its longest side until the bound decides the question asked -- below the threshold everywhere (the largest
//! remaining bound is the answer), or reached at a centre in the reach (attained) -- or until its bound is within the rounding
//! of the value attained at its centre. No point is sampled as a bound.

use super::{Model, Reach, Trap};
use crate::field::Geometry;
use crate::linalg::{dot3, norm3, C64};
use crate::request::GROUND;
use crate::units::{rounding, KT_PER_UK};
use std::collections::BinaryHeap;

/// A beam's envelope Lipschitz constant L_b = sup |∇b| (1/µm), a = e^{ik·r} b the amplitude with its plane-wave phase taken out:
/// 0 for a plane wave; for a paraxial Gaussian beam ∇b = b g_b, g_b = −2t/(w₀²q) + (i/z_R)(ρ²/(w₀²q²) − 1/q) n (field.rs, less ik n),
/// and with |b| = e^{−x}/√(1 + s), x = ρ²/(w₀²(1 + s)), s = ζ²/z_R²: |b| 2ρ/(w₀²|q|) = 2√x e^{−x}/(w₀ √(1 + s)) ≤ √(2/e)/w₀
/// (√x e^{−x} ≤ 1/√(2e)), |b| |ρ²/(w₀²q²) − 1/q|/z_R ≤ (x e^{−x} + e^{−x})/z_R ≤ (1 + 1/e)/z_R: L_b = √(2/e)/w₀ + (1 + 1/e)/z_R.
pub(crate) fn envelope_lipschitz(beam: &Geometry) -> f64 {
    match (beam.waist, beam.rayleigh()) {
        (Some(w), Some(z)) => {
            let e = std::f64::consts::E;
            (2.0 / e).sqrt() / w + (1.0 + 1.0 / e) / z
        }
        _ => 0.0,
    }
}

/// A beam's Lipschitz constant L = sup |∇a| ≤ k + L_b (`envelope_lipschitz`).
pub(crate) fn lipschitz(beam: &Geometry) -> f64 {
    beam.wave_number + envelope_lipschitz(beam)
}

/// A beam's envelope curvature bound M_b = sup |∇∇b| (1/µm²): 0 for a plane wave; for a paraxial Gaussian beam ∇∇b = b(g_b g_bᵀ + ∇g)
/// with ∇g = −2(1 − nnᵀ)/(w₀²q) + 2i(ntᵀ + tnᵀ)/(z_R w₀² q²) − (1/q² − 2ρ²/(w₀²q³)) nnᵀ/z_R² (field.rs). With x, s as in
/// `envelope_lipschitz`, |g_b| ≤ 2√x/w₀ + (x + 1)/z_R, so |b||g_b|² ≤ (2e^{−1/2}/w₀ + 2e^{−1/2}/z_R)² (√x e^{−x/2} ≤ e^{−1/2},
/// (x + 1)e^{−x/2} ≤ 2e^{−1/2}); and |b||∇g| ≤ e^{−x}(2/w₀² + 4√x/(z_R w₀) + (1 + 2x)/z_R²) ≤ 2/w₀² + 4/(√(2e) z_R w₀) + 2e^{−1/2}/z_R²
/// ((1 + 2x)e^{−x} ≤ 2e^{−1/2}).
pub(crate) fn envelope_curvature(beam: &Geometry) -> f64 {
    match (beam.waist, beam.rayleigh()) {
        (Some(w), Some(z)) => {
            let h = (-0.5_f64).exp();
            let e = std::f64::consts::E;
            (2.0 * h / w + 2.0 * h / z).powi(2) + 2.0 / (w * w) + 4.0 / ((2.0 * e).sqrt() * z * w) + 2.0 * h / (z * z)
        }
        _ => 0.0,
    }
}

/// A box of positions: its centre and half-widths (µm); an infinite half-width is the whole axis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Cell {
    pub(crate) centre: [f64; 3],
    pub(crate) half: [f64; 3],
}

impl Cell {
    /// Its half-diagonal ρ: every point of it within ρ of its centre (infinite for an unbounded cell).
    pub(crate) fn radius(&self) -> f64 {
        self.half.iter().map(|h| h * h).sum::<f64>().sqrt()
    }
    pub(crate) fn bounded(&self) -> bool {
        self.radius().is_finite()
    }
    /// Its halves along its longest side.
    fn halves(&self) -> [Cell; 2] {
        let a = (0..3).max_by(|&i, &j| self.half[i].total_cmp(&self.half[j])).unwrap();
        let mut half = self.half;
        half[a] *= 0.5;
        [-1.0, 1.0].map(|s| {
            let mut centre = self.centre;
            centre[a] += s * half[a];
            Cell { centre, half }
        })
    }
}

/// What a cell gives a bounded quantity: an upper bound over the cell, and the value at its centre (−∞ for an unbounded cell).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Bounds {
    pub(crate) upper: f64,
    pub(crate) centre: f64,
}

/// A beam over a cell, its amplitude a = e^{ik·r} b written by its envelope b (module documentation of units.rs): a and ∇a at the
/// centre, its wave vector, the bound of |a| = |b| over the cell, how far b moves within it (L_b ρ), the bound of |∇b| over it
/// (|∇b(c)| + M_b ρ, at most L_b), and the cell's radius ρ (infinite: the whole space, its global bounds).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Span {
    pub(crate) centre: C64,
    pub(crate) gradient: [C64; 3],
    pub(crate) wave: [f64; 3],
    pub(crate) most: f64,
    pub(crate) spread: f64,
    pub(crate) slope: f64,
    pub(crate) rho: f64,
}

impl Span {
    /// The same beam on the cell's centre point alone: no spread, its envelope's slope there.
    pub(crate) fn point(&self) -> Span {
        let k = self.wave;
        let slope = (0..3).map(|i| (self.gradient[i] - C64::new(0.0, k[i]) * self.centre).norm_sqr()).sum::<f64>().sqrt();
        Span { most: self.centre.abs(), spread: 0.0, slope, rho: 0.0, ..*self }
    }
}

/// A beam pair's coefficient a_j'* a_i = e^{iq·r} b_j* b_i (q = k_i − k_j) over a cell: its value at the centre, the radius of its
/// range (|q|ρ max|b_j| max|b_i| + the envelopes' moves, at most |c₀| + max|a_j| max|a_i|), and the bound of its gradient
/// (|q| max|b_j| max|b_i| + max|∇b_j| max|b_i| + max|b_j| max|∇b_i|; exact at a point: conj(∇a_j) a_i + conj(a_j) ∇a_i).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PairSpan {
    pub(crate) centre: C64,
    pub(crate) radius: f64,
    pub(crate) slope: f64,
}

/// The pair (j, i) of beams over a cell (`PairSpan`).
pub(crate) fn pair_span(spans: &[Span], j: usize, i: usize) -> PairSpan {
    let (a, b) = (&spans[j], &spans[i]);
    let q = norm3(std::array::from_fn(|x| b.wave[x] - a.wave[x]));
    if !a.rho.is_finite() {
        return PairSpan { centre: C64::ZERO, radius: a.most * b.most, slope: q * a.most * b.most + a.slope * b.most + a.most * b.slope };
    }
    let centre = a.centre.conj() * b.centre;
    if a.rho == 0.0 {
        let g: [C64; 3] = std::array::from_fn(|x| a.gradient[x].conj() * b.centre + a.centre.conj() * b.gradient[x]);
        return PairSpan { centre, radius: 0.0, slope: g.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt() };
    }
    let radius = (q * a.rho * a.most * b.most + a.spread * b.most + a.centre.abs() * b.spread).min(centre.abs() + a.most * b.most);
    PairSpan { centre, radius, slope: q * a.most * b.most + a.slope * b.most + a.most * b.slope }
}

/// A cell in the bound's queue, ordered by its upper bound.
struct Queued {
    upper: f64,
    cell: Cell,
    centre: f64,
}
impl PartialEq for Queued {
    fn eq(&self, o: &Self) -> bool {
        self.upper == o.upper
    }
}
impl Eq for Queued {}
impl PartialOrd for Queued {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Queued {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.upper.total_cmp(&o.upper)
    }
}

impl Model {
    /// The reach of a trap at temperature T_r (µK) and unsampled share s (module documentation).
    pub fn reach(&self, trap: &Trap, p: &[f64; GROUND], temperature_uk: f64, share: f64) -> Reach {
        let energy = (1.0 / share).ln() * KT_PER_UK * temperature_uk;
        let level = trap.minimum + energy;
        let envelope = self.envelope(p);
        let b = envelope.peak();
        let n = envelope.beams.len();
        let mut extent = [0.0_f64; 3];
        for (j, &l) in envelope.beams.iter().enumerate() {
            let weight: f64 = (0..n).map(|i| envelope.weights[j * n + i] + envelope.weights[i * n + j]).sum();
            if weight == 0.0 {
                continue;
            }
            let beam = &self.light.beams[l];
            match (beam.waist, beam.rayleigh()) {
                (Some(w), Some(z)) if level < 0.0 => {
                    let u = b / -level - 1.0;
                    for a in 0..3 {
                        let na = beam.direction[a];
                        extent[a] = extent[a].max((u.max(0.0) * (0.5 * w * w * (1.0 - na * na) + z * z * na * na)).sqrt());
                    }
                }
                _ => extent = [f64::INFINITY; 3],
            }
        }
        Reach {
            temperature_uk,
            share,
            energy,
            minimum: trap.minimum,
            cell: std::array::from_fn(|a| [-extent[a], extent[a]]),
            speed: (2.0 * self.hbar_over_mass * energy).sqrt(),
            populations: *p,
        }
    }

    /// Every beam over a cell (`Span`).
    pub(crate) fn spans(&self, cell: &Cell) -> Vec<Span> {
        let rho = cell.radius();
        if !rho.is_finite() {
            return self
                .light
                .beams
                .iter()
                .map(|b| Span { centre: C64::ZERO, gradient: [C64::ZERO; 3], wave: b.wave_vector(), most: 1.0, spread: f64::INFINITY, slope: envelope_lipschitz(b), rho })
                .collect();
        }
        let amplitudes = self.light.amplitudes(cell.centre);
        self.light
            .beams
            .iter()
            .zip(&amplitudes)
            .map(|(beam, a)| {
                let spread = envelope_lipschitz(beam) * rho;
                let k = beam.wave_vector();
                let envelope = (0..3).map(|i| (a.gradient[i] - C64::new(0.0, k[i]) * a.value).norm_sqr()).sum::<f64>().sqrt();
                Span {
                    centre: a.value,
                    gradient: a.gradient,
                    wave: k,
                    most: (a.value.abs() + spread).min(1.0),
                    spread,
                    slope: (envelope + envelope_curvature(beam) * rho).min(envelope_lipschitz(beam)),
                    rho,
                }
            })
            .collect()
    }

    /// A lower bound of V_prep(r; p) over a cell from its centre: each beam pair's coefficient moves by at most its radius
    /// (`pair_span`), weighted by the envelope's |Q_l'l| (the near trap light's included, potential.rs).
    pub(crate) fn prepared_below(&self, cell: &Cell, spans: &[Span], p: &[f64; GROUND]) -> f64 {
        if !cell.bounded() {
            return f64::NEG_INFINITY;
        }
        let at = self.potential(cell.centre, p, &self.prepared());
        let envelope = self.envelope(p);
        let n = envelope.beams.len();
        let mut slack = 0.0;
        for (j, &lj) in envelope.beams.iter().enumerate() {
            for (i, &li) in envelope.beams.iter().enumerate() {
                let w = envelope.weights[j * n + i];
                if w != 0.0 {
                    slack += w * pair_span(spans, lj, li).radius;
                }
            }
        }
        at - slack
    }

    /// The provable bound over the reach of a quantity bounded per cell by `f` (module documentation, # Provable extrema), asked
    /// against `threshold`: the largest upper bound of the cells left when it decides (below the threshold everywhere, or
    /// attained at or above it at a centre in the reach, or within the rounding of its centre's value).
    pub(crate) fn bound_over(&self, reach: &Reach, threshold: f64, f: &dyn Fn(&Cell, &[Span]) -> Bounds) -> f64 {
        let half = reach.cell.map(|[lo, hi]| 0.5 * (hi - lo));
        let start = Cell { centre: reach.cell.map(|[lo, hi]| if (hi - lo).is_finite() { 0.5 * (lo + hi) } else { 0.0 }), half };
        let level = reach.minimum + reach.energy;
        let mut queue = BinaryHeap::new();
        let push = |queue: &mut BinaryHeap<Queued>, cell: Cell| {
            let spans = self.spans(&cell);
            if cell.bounded() && self.prepared_below(&cell, &spans, &reach.populations) > level {
                return;
            }
            let b = f(&cell, &spans);
            queue.push(Queued { upper: b.upper, cell, centre: b.centre });
        };
        push(&mut queue, start);
        loop {
            let Some(top) = queue.pop() else { return 0.0 };
            let attained = top.cell.bounded() && self.potential(top.cell.centre, &reach.populations, &self.prepared()) <= level;
            if top.upper < threshold
                || !top.cell.bounded()
                || (attained && top.centre >= threshold)
                || top.upper - top.centre <= rounding(1) * threshold.abs().max(top.centre.abs())
            {
                return top.upper;
            }
            for half in top.cell.halves() {
                push(&mut queue, half);
            }
        }
    }
}

/// The distance of a range [lo, hi] from zero: 0 where it holds zero.
pub(crate) fn distance(lo: f64, hi: f64) -> f64 {
    if lo <= 0.0 && hi >= 0.0 {
        0.0
    } else {
        lo.abs().min(hi.abs())
    }
}

/// The dot product's bound |k·v| ≤ |k| v over every velocity of the reach.
pub(crate) fn doppler(k: [f64; 3], speed: f64) -> f64 {
    dot3(k, k).sqrt() * speed
}
