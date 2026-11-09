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
use crate::linalg::dot3;
use crate::request::GROUND;
use crate::units::{rounding, KT_PER_UK};
use std::collections::BinaryHeap;

/// A beam's Lipschitz constant L = sup |∇a| (1/µm): k for a plane wave; for a paraxial Gaussian beam, with ∇a = a g,
/// g = ik n − 2t/(w₀²q) + (i/z_R)(ρ²/(w₀²q²) − 1/q) n (field.rs) and |a| = e^{−x}/√(1 + s), x = ρ²/(w₀²(1 + s)), s = ζ²/z_R²:
/// |a| k ≤ k; |a| 2ρ/(w₀²|q|) = 2√x e^{−x}/(w₀ √(1 + s)) ≤ √(2/e)/w₀ (√x e^{−x} ≤ 1/√(2e)); and |a| |ρ²/(w₀²q²) − 1/q|/z_R ≤
/// (x e^{−x} + e^{−x})/z_R ≤ (1/e + 1)/z_R -- so L = k + √(2/e)/w₀ + (1 + 1/e)/z_R.
pub(crate) fn lipschitz(beam: &Geometry) -> f64 {
    match (beam.waist, beam.rayleigh()) {
        (Some(w), Some(z)) => {
            let e = std::f64::consts::E;
            beam.wave_number + (2.0 / e).sqrt() / w + (1.0 + 1.0 / e) / z
        }
        _ => beam.wave_number,
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

/// A beam's amplitude over a cell: its value at the centre and the bound of |a_l| over the cell.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Span {
    pub(crate) centre: crate::linalg::C64,
    pub(crate) most: f64,
    /// L_l ρ: how far a_l moves from its centre value within the cell.
    pub(crate) spread: f64,
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

    /// Every beam's amplitude over a cell (`Span`).
    pub(crate) fn spans(&self, cell: &Cell) -> Vec<Span> {
        let rho = cell.radius();
        if !rho.is_finite() {
            return self.light.beams.iter().map(|_| Span { centre: crate::linalg::C64::ZERO, most: 1.0, spread: f64::INFINITY }).collect();
        }
        let amplitudes = self.light.amplitudes(cell.centre);
        self.light
            .beams
            .iter()
            .zip(&amplitudes)
            .map(|(beam, a)| {
                let spread = lipschitz(beam) * rho;
                Span { centre: a.value, most: (a.value.abs() + spread).min(1.0), spread }
            })
            .collect()
    }

    /// A lower bound of V_prep(r; p) over a cell from its centre: each beam pair's coefficient a_l'* a_l moves by at most
    /// L_l' ρ max|a_l| + |a_l'(c)| L_l ρ (far.rs), its weight |[P_l'l]_p|; the near part's (eliminate.rs) bound added.
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
                    slack += w * (spans[lj].spread * spans[li].most + spans[lj].centre.abs() * spans[li].spread);
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
