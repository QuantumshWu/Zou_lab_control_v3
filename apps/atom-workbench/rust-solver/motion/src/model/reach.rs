//! The atoms' reach (SPEC §1.8) and provable extrema over it.
//!
//! # The reach
//! The sublevel set {(r, v) : V_prep(r; p) + v²/(2 ħ/m) ≤ V_min + E_r} of the prepared energy, E_r = ln(1/s) k_B T_r: at r the
//! speeds |v| ≤ √(2 (ħ/m)(V_min + E_r − V_prep(r))), at most √(2 (ħ/m) E_r) (`Reach::speed`), the positions where that is real. Its level
//! V_min + E_r is below the far value 0 where the trap holds the reach; there −V_prep ≥ −(V_min + E_r) > 0 on it, and the proven
//! envelope −V_prep ≤ B max_l A_l(r) (model.rs `Envelope`) puts it inside the union of the ellipsoids u_l ≤ B/(−V_min − E_r) − 1
//! of the trap light's focused beams, u_l = xᵀ M_l x, M_l = (2/w₀²)(1 − nnᵀ) + nnᵀ/z_R²: their lab-axis extents √(U (M_l⁻¹)_aa),
//! M_l⁻¹ = (w₀²/2)(1 − nnᵀ) + z_R² nnᵀ, bound `Reach::cell`. A level at or above the far value -- or a plane wave in the trap
//! light -- leaves the reach unbounded: the whole space, its bounds the global ones.
//!
//! # Provable extrema (`Model::bounds_over`)
//! A quantity's bound over the reach is the largest of its upper bounds over cells covering the reach's box, a cell taken where
//! it may meet the reach (a lower bound of V_prep over it at most the level). A quantity is one function of a region (`Region`):
//! a cell's beams -- each one's centre value and Lipschitz constants (`lipschitz`): |a_l(r) − a_l(c)| ≤ L_l |r − c|, |a_l| ≤ 1 --
//! or its centre point's, its radius zero. Monotone in the beams' bounds, on the cell it bounds the quantity over the cell and
//! on the point it is the quantity's bound there (a drive's coherent peak: every beam at its amplitude there, in phase); the
//! first tends to the second as the cell shrinks. The cell of a quantity's largest bound is halved along its longest side until
//! the bound answers the question asked -- a threshold decided: below it everywhere (the largest remaining bound is the answer), or
//! reached at a centre in the reach (attained); or an extent known: within a resolution of a value attained at a centre in the
//! reach -- or until it is within the rounding of its centre's value. No point is sampled as a bound. (A threshold near the
//! quantity's true largest value is the expensive question: every cell along the place of that value has to shrink to their
//! difference. A range is asked as an extent, never by thresholds closing in on it.) All the quantities of a model share one tree of cells: a cell's beams, beam pairs and the parts the units' bounds
//! have in common (the far light's energies, the carriers' dressings, the clusters' force bounds, the drives' bounds per ground
//! state) are made once, for whichever quantity asks first, and each quantity halves only the cells it still needs -- one decided
//! on the coarse cells costs nothing further.

use super::eliminate::{CellEnergies, Couplings};
use super::{Model, Reach, Trap};
use crate::field::Geometry;
use crate::linalg::{dot3, norm3, C64};
use crate::request::GROUND;
use crate::units::{rounding, KT_PER_UK};
use std::cell::OnceCell;
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

/// A beam over a cell, its amplitude a = e^{ik·r} b written by its envelope b (module documentation of units.rs): a and ∇a at the
/// centre, its wave vector, the bound of |a| = |b| over the cell, the bound of |∇b| over it (|∇b(c)| + M_b ρ, at most L_b), the
/// envelope's curvature bound M_b (`envelope_curvature`), and the cell's radius ρ (infinite: the whole space, its global bounds).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Span {
    pub(crate) centre: C64,
    pub(crate) gradient: [C64; 3],
    pub(crate) wave: [f64; 3],
    pub(crate) most: f64,
    pub(crate) slope: f64,
    pub(crate) curvature: f64,
    pub(crate) rho: f64,
}

impl Span {
    /// The same beam on the cell's centre point alone: its envelope's slope there.
    pub(crate) fn point(&self) -> Span {
        let k = self.wave;
        let slope = (0..3).map(|i| (self.gradient[i] - C64::new(0.0, k[i]) * self.centre).norm_sqr()).sum::<f64>().sqrt();
        Span { most: self.centre.abs(), slope, rho: 0.0, ..*self }
    }
    /// The bound of |∇a| = |e^{ik·r}(ik b + ∇b)| over the cell: |k| max|b| + max|∇b|.
    pub(crate) fn lipschitz(&self) -> f64 {
        norm3(self.wave) * self.most + self.slope
    }
}

/// A beam pair's coefficient c = a_j'* a_i = e^{iq·r} B, B = b_j* b_i (q = k_i − k_j), over a cell: its value and gradient at the
/// centre; the bound of |∇∇c| over the cell, q²|B| + 2q|∇B| + |∇∇B| with |∇B| ≤ max|∇b_j| max|b_i| + max|b_j| max|∇b_i| and
/// |∇∇B| ≤ M_j max|b_i| + 2 max|∇b_j| max|∇b_i| + max|b_j| M_i; the bound of |∇c| over the cell, |∇c(c)| + ρ max|∇∇c|, at most
/// q|B| + |∇B|'s bound; and the radius of its range, ρ max|∇c|, at most |c(c)| + max|a_j| max|a_i|. On a point: its value,
/// its gradient's norm, radius zero.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PairSpan {
    pub(crate) centre: C64,
    pub(crate) gradient: [C64; 3],
    pub(crate) radius: f64,
    pub(crate) slope: f64,
    pub(crate) hessian: f64,
}

/// The pair (j, i) of beams over a cell (`PairSpan`).
pub(crate) fn pair_span(spans: &[Span], j: usize, i: usize) -> PairSpan {
    let (a, b) = (&spans[j], &spans[i]);
    let q = norm3(std::array::from_fn(|x| b.wave[x] - a.wave[x]));
    let moves = a.slope * b.most + a.most * b.slope;
    let blind = q * a.most * b.most + moves;
    let hessian = q * q * a.most * b.most + 2.0 * q * moves + a.curvature * b.most + 2.0 * a.slope * b.slope + a.most * b.curvature;
    if !a.rho.is_finite() {
        return PairSpan { centre: C64::ZERO, gradient: [C64::ZERO; 3], radius: a.most * b.most, slope: blind, hessian };
    }
    let centre = a.centre.conj() * b.centre;
    let gradient: [C64; 3] = std::array::from_fn(|x| a.gradient[x].conj() * b.centre + a.centre.conj() * b.gradient[x]);
    let slope = (gradient.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt() + a.rho * hessian).min(blind);
    PairSpan { centre, gradient, radius: (a.rho * slope).min(centre.abs() + a.most * b.most), slope, hessian }
}

/// A cell's beams, or its centre point's (its radius zero), with what the units' bounds share over it, each part made once for
/// every quantity that asks (module documentation, # Provable extrema): the reach's speeds over it, the beam pairs per frequency,
/// the far light's energies,
/// the carriers' coherent couplings and dressings of the ground states, the clusters' force bounds and the ground states' drive
/// bounds (eliminate.rs).
pub(crate) struct Region {
    pub(crate) spans: Vec<Span>,
    /// The cell's radius (0: a point).
    pub(crate) rho: f64,
    /// The bound of the atoms' speed over it (µm/µs): the reach's at its least V_prep (module documentation, # The reach); a
    /// sample's own on its point.
    pub(crate) speed: f64,
    /// Per frequency, its beam pairs (j, i) row-major (`PairSpan`).
    pub(crate) pairs: Vec<Vec<PairSpan>>,
    pub(crate) energies: OnceCell<CellEnergies>,
    /// Per carrier.
    pub(crate) couplings: OnceCell<Vec<Couplings>>,
    /// [carrier][ground state].
    pub(crate) dressings: OnceCell<Vec<Vec<f64>>>,
    /// Per cluster.
    pub(crate) forces: Vec<OnceCell<f64>>,
    /// Per ground state: its folded drives' saturation and sweep.
    pub(crate) drives: Vec<OnceCell<(f64, f64)>>,
}

/// A cell of the proofs' tree: its box, its region and its centre point's (none on an unbounded cell), whether its centre is in
/// the reach, and its halves once made (none where a half cannot meet the reach).
struct Node {
    cell: Cell,
    over: Region,
    at: Option<Region>,
    inside: bool,
    halves: Option<[Option<usize>; 2]>,
}

/// A cell in a quantity's queue, ordered by its upper bound.
struct Queued {
    upper: f64,
    node: usize,
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
                .map(|b| Span { centre: C64::ZERO, gradient: [C64::ZERO; 3], wave: b.wave_vector(), most: 1.0, slope: envelope_lipschitz(b), curvature: envelope_curvature(b), rho })
                .collect();
        }
        let amplitudes = self.light.amplitudes(cell.centre);
        self.light
            .beams
            .iter()
            .zip(&amplitudes)
            .map(|(beam, a)| {
                let k = beam.wave_vector();
                let envelope = (0..3).map(|i| (a.gradient[i] - C64::new(0.0, k[i]) * a.value).norm_sqr()).sum::<f64>().sqrt();
                let curvature = envelope_curvature(beam);
                Span {
                    centre: a.value,
                    gradient: a.gradient,
                    wave: k,
                    most: (a.value.abs() + envelope_lipschitz(beam) * rho).min(1.0),
                    slope: (envelope + curvature * rho).min(envelope_lipschitz(beam)),
                    curvature,
                    rho,
                }
            })
            .collect()
    }

    /// How far V_prep(r; p) moves over a bounded cell from its centre's value: each beam pair's coefficient by at most its radius
    /// (`pair_span`), weighted by the envelope's |Q_l'l| (the near trap light's included, potential.rs).
    fn prepared_slack(&self, spans: &[Span], p: &[f64; GROUND]) -> f64 {
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
        slack
    }

    /// A region of the beams `spans` at speeds up to `speed` (`Region`): its beam pairs made, the rest when asked.
    pub(crate) fn region(&self, spans: Vec<Span>, speed: f64) -> Region {
        let pairs = self
            .light
            .frequencies
            .iter()
            .map(|f| f.beams.iter().flat_map(|&j| f.beams.iter().map(move |&i| (j, i))).map(|(j, i)| pair_span(&spans, j, i)).collect())
            .collect();
        Region {
            pairs,
            rho: spans.first().map_or(0.0, |s| s.rho),
            speed,
            spans,
            energies: OnceCell::new(),
            couplings: OnceCell::new(),
            dressings: OnceCell::new(),
            forces: (0..self.clusters.len()).map(|_| OnceCell::new()).collect(),
            drives: (0..GROUND).map(|_| OnceCell::new()).collect(),
        }
    }

    /// The region of a point alone at one speed (a sample's: the errors measured there).
    pub(crate) fn point_region(&self, r: [f64; 3], speed: f64) -> Region {
        self.region(self.spans(&Cell { centre: r, half: [0.0; 3] }), speed)
    }

    /// The provable bounds over the reach of quantities given per region, each asked against `threshold` (module documentation,
    /// # Provable extrema): each one's largest upper bound of the cells left when it is decided -- below the threshold everywhere,
    /// or attained at or above it at a centre in the reach --, or within the rounding of its centre's value; 0 where no cell meets
    /// the reach.
    pub(crate) fn bounds_over(&self, reach: &Reach, threshold: f64, quantities: &[&dyn Fn(&Region) -> f64]) -> Vec<f64> {
        self.proven_over(reach, &|upper, attained| upper < threshold || attained >= threshold, threshold.abs(), quantities)
    }

    /// The provable bounds over the reach of quantities given per region, each to `resolution`: each one's largest upper bound of
    /// the cells left once it is within the resolution of a value attained at a centre in the reach.
    pub(crate) fn extents_over(&self, reach: &Reach, resolution: f64, quantities: &[&dyn Fn(&Region) -> f64]) -> Vec<f64> {
        self.proven_over(reach, &|upper, attained| upper - attained <= resolution, resolution, quantities)
    }

    /// The proofs' subdivision (module documentation, # Provable extrema): per quantity the cell of its largest upper bound is
    /// halved until `settled(that bound, the largest value attained at a centre in the reach so far)`, the cell is unbounded, or
    /// the bound is within the rounding of its centre's value (of `scale`, the question's own size, where that is larger).
    fn proven_over(&self, reach: &Reach, settled: &dyn Fn(f64, f64) -> bool, scale: f64, quantities: &[&dyn Fn(&Region) -> f64]) -> Vec<f64> {
        let level = reach.minimum + reach.energy;
        // The reach's speeds where V_prep is at least v.
        let speed = |v: f64| reach.speed.min((2.0 * self.hbar_over_mass * (level - v).max(0.0)).sqrt());
        let make = |cell: Cell| -> Option<Node> {
            let spans = self.spans(&cell);
            if !cell.bounded() {
                return Some(Node { cell, over: self.region(spans, reach.speed), at: None, inside: false, halves: None });
            }
            let centre = self.potential(cell.centre, &reach.populations, &self.prepared());
            let below = centre - self.prepared_slack(&spans, &reach.populations);
            if below > level {
                return None;
            }
            let at = self.region(spans.iter().map(Span::point).collect(), speed(centre));
            Some(Node { cell, over: self.region(spans, speed(below)), at: Some(at), inside: centre <= level, halves: None })
        };
        let half = reach.cell.map(|[lo, hi]| 0.5 * (hi - lo));
        let start = Cell { centre: reach.cell.map(|[lo, hi]| if (hi - lo).is_finite() { 0.5 * (lo + hi) } else { 0.0 }), half };
        let mut nodes: Vec<Node> = make(start).into_iter().collect();
        let mut out = vec![];
        for q in quantities {
            // The cells left, by their upper bounds, and the largest value at a centre in the reach among them and their parents.
            let (mut queue, mut attained) = (BinaryHeap::new(), f64::NEG_INFINITY);
            let push = |queue: &mut BinaryHeap<Queued>, attained: &mut f64, nodes: &[Node], at: usize| {
                let n = &nodes[at];
                let centre = n.at.as_ref().map_or(f64::NEG_INFINITY, |r| q(r));
                if n.inside {
                    *attained = attained.max(centre);
                }
                queue.push(Queued { upper: q(&n.over), node: at, centre });
            };
            if !nodes.is_empty() {
                push(&mut queue, &mut attained, &nodes, 0);
            }
            out.push(loop {
                let Some(top) = queue.pop() else { break 0.0 };
                let cell = nodes[top.node].cell;
                if settled(top.upper, attained) || !cell.bounded() || top.upper - top.centre <= rounding(1) * scale.max(top.centre.abs()) {
                    break top.upper;
                }
                let halves = match nodes[top.node].halves {
                    Some(h) => h,
                    None => {
                        let h = cell.halves().map(|c| {
                            make(c).map(|x| {
                                nodes.push(x);
                                nodes.len() - 1
                            })
                        });
                        nodes[top.node].halves = Some(h);
                        h
                    }
                };
                for h in halves.into_iter().flatten() {
                    push(&mut queue, &mut attained, &nodes, h);
                }
            });
        }
        out
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::Light;
    use crate::model::far::{pair, EXCITED};
    use crate::model::{Component, Unit};
    use crate::request::{Request, STATES};

    /// R8_R5051 (a six-beam lattice of cooling light with its repump, in a tweezer) with every light but the trap's near.
    fn lattice() -> Model {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("scenes").join("R8_R5051.json");
        let line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let request = Request::from_native(&line).unwrap();
        let light = Light::new(&request);
        let near: Vec<Unit> = (0..light.frequencies.len()).filter(|&f| f != light.trap).map(|frequency| Unit::Far { component: Component { frequency, counter: false } }).collect();
        Model::new(&request, &light, &near).unwrap()
    }

    /// Points spread through a cell: its centre and a 5³ grid on and inside its faces.
    fn points(cell: &Cell) -> Vec<[f64; 3]> {
        let mut out = vec![cell.centre];
        for i in 0..5 {
            for j in 0..5 {
                for k in 0..5 {
                    let u = [i, j, k].map(|x| x as f64 / 2.0 - 1.0);
                    out.push(std::array::from_fn(|a| cell.centre[a] + u[a] * cell.half[a]));
                }
            }
        }
        out
    }

    /// A region's bounds hold at every point of its cell and are the values on its centre point (module documentation,
    /// # Provable extrema): on the six-beam lattice, each beam pair's coefficient within its radius of the centre's and its
    /// gradient within its slope; every carrier's coherent coupling |Σ_l a_l X^f_l[e,s]|, its columns and its units' columns within
    /// their bounds; the far light's energies within their slack and their differences' gradients within `energy_slope_over` --
    /// and on a cell a thousand times smaller the couplings' bounds within that factor of the centre's values.
    #[test]
    fn a_region_s_bounds_hold_over_its_cell_and_are_the_values_at_its_centre() {
        let model = lattice();
        let within = |value: f64, bound: f64| value <= bound * (1.0 + 1e-12) + 1e-12;
        let norm = |v: &[C64]| v.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        for cell in [Cell { centre: [0.13, -0.21, 0.07], half: [0.05, 0.08, 0.03] }, Cell { centre: [-0.4, 0.3, 0.25], half: [0.3, 0.2, 0.25] }] {
            let region = model.region(model.spans(&cell), 0.0);
            let (energies, couplings) = (model.energies(&region), model.couplings(&region));
            for r in points(&cell) {
                let a = model.light.amplitudes(r);
                let mut energy: Vec<f64> = (0..STATES).map(|s| model.base[(s, s)].re).collect();
                let mut gradient = vec![[0.0; 3]; STATES];
                for (w, f) in model.light.frequencies.iter().enumerate() {
                    let n = f.beams.len();
                    for x in 0..n * n {
                        let (c, bound) = (pair(&a[f.beams[x / n]], &a[f.beams[x % n]]), &region.pairs[w][x]);
                        assert!(within((c.value - bound.centre).abs(), bound.radius), "pair {w}/{x} at {r:?}: moved {} beyond {}", (c.value - bound.centre).abs(), bound.radius);
                        let g = c.gradient.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
                        assert!(within(g, bound.slope), "pair {w}/{x} at {r:?}: |∇c| {g} beyond {}", bound.slope);
                        let p = &model.tones[w].pairs[x];
                        for s in 0..STATES {
                            energy[s] += (c.value * p[(s, s)]).re;
                            for i in 0..3 {
                                gradient[s][i] += (c.gradient[i] * p[(s, s)]).re;
                            }
                        }
                    }
                }
                for s in 0..STATES {
                    assert!(within((energy[s] - energies.centre[s]).abs(), energies.slack[s]), "E_{s} at {r:?}");
                }
                for (x, y) in [(0, 3), (GROUND, 1), (GROUND + 5, 6), (STATES - 1, GROUND)] {
                    let g = (0..3).map(|i| (gradient[x][i] - gradient[y][i]).powi(2)).sum::<f64>().sqrt();
                    assert!(within(g, model.energy_slope_over(x, y, &region)), "|∇(E_{x} − E_{y})| {g} at {r:?}");
                }
                for (cc, bound) in model.carriers.iter().zip(couplings) {
                    let mut units: Vec<usize> = cc.units.clone();
                    units.sort();
                    units.dedup();
                    for s in 0..GROUND {
                        let field = |beams: &dyn Fn(usize) -> bool| -> Vec<C64> {
                            (0..EXCITED).map(|e| cc.beams.iter().enumerate().filter(|(l, _)| beams(*l)).map(|(l, &b)| a[b].value * cc.folded[l][(GROUND + e, s)]).sum()).collect()
                        };
                        let u = field(&|_| true);
                        for e in 0..EXCITED {
                            assert!(within(u[e].abs(), bound.entry[s][e]), "|u_{s}[{e}]| {} beyond {} at {r:?}", u[e].abs(), bound.entry[s][e]);
                        }
                        assert!(within(norm(&u), bound.column[s]), "‖u_{s}‖ at {r:?}");
                        for (ui, &unit) in units.iter().enumerate() {
                            assert!(within(norm(&field(&|l| cc.units[l] == unit)), bound.units[ui][s]), "unit {ui}'s ‖u_{s}‖ at {r:?}");
                        }
                    }
                }
            }
            // On the centre point, the values; on a cell a thousand times smaller, within that factor of them.
            let point = model.region(region.spans.iter().map(Span::point).collect(), 0.0);
            let small = Cell { centre: cell.centre, half: cell.half.map(|h| h * 1e-3) };
            let tiny = model.region(model.spans(&small), 0.0);
            let a = model.light.amplitudes(cell.centre);
            for (ci, cc) in model.carriers.iter().enumerate() {
                for s in 0..GROUND {
                    for e in 0..EXCITED {
                        let exact = cc.beams.iter().enumerate().map(|(l, &b)| a[b].value * cc.folded[l][(GROUND + e, s)]).sum::<C64>().abs();
                        let (at, near) = (model.couplings(&point)[ci].entry[s][e], model.couplings(&tiny)[ci].entry[s][e]);
                        assert!((at - exact).abs() <= 1e-12 * exact + 1e-15, "the centre's |u| {at} against {exact}");
                        let scale = model.couplings(&region)[ci].entry[s][e] - exact;
                        assert!(near - exact <= 2e-3 * scale + 1e-12 * exact.max(1e-3), "a thousandth of the cell: {} over the centre's, the cell's {scale}", near - exact);
                    }
                }
            }
        }
    }

    /// One tree of cells for many quantities proves each as its own proof would: the shared cells change no bound (R3_DOPPLER's
    /// cooling pair near, its drives' saturations and sweep from two ground states).
    #[test]
    fn one_tree_proves_each_quantity_as_its_own_proof_would() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("scenes").join("R3_DOPPLER.json");
        let line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let request = Request::from_native(&line).unwrap();
        let light = Light::new(&request);
        let near: Vec<Unit> = (0..light.frequencies.len()).filter(|&f| f != light.trap).map(|frequency| Unit::Far { component: Component { frequency, counter: false } }).collect();
        let model = Model::new(&request, &light, &near).unwrap();
        let b = 7;
        let p = {
            let mut p = [0.0; GROUND];
            p[b] = 1.0;
            p
        };
        let trap = model.trap(&p, &model.prepared()).unwrap();
        let reach = model.reach(&trap, &p, 20.0, 2e-3);
        let cluster = model.membership[b];
        let s = |r: &Region| model.drive_bound(b, cluster, r).0;
        let a = |r: &Region| model.drive_bound(b, cluster, r).1;
        let other = |r: &Region| model.drive_bound(3, model.membership[3], r).0;
        let together = model.bounds_over(&reach, 0.5, &[&s, &a, &other]);
        let alone: Vec<f64> = [&s as &dyn Fn(&Region) -> f64, &a, &other].iter().map(|q| model.bounds_over(&reach, 0.5, &[*q])[0]).collect();
        assert_eq!(together, alone);
    }
}
