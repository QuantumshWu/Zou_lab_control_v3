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
//! a cell's beams -- each one's centre value and Lipschitz constants (`Span`): |a_l(r) − a_l(c)| ≤ L_l |r − c|, |a_l| ≤ 1 --
//! or its centre point's, its radius zero. Monotone in the beams' bounds, on the cell it bounds the quantity over the cell and
//! on the point it is the quantity's bound there (a drive's coherent peak: every beam at its amplitude there, in phase); the
//! first tends to the second as the cell shrinks. The cell of a quantity's largest bound is halved along its longest side until
//! the bound answers the question asked -- a threshold decided: below it everywhere (the largest remaining bound is the answer), or
//! reached at a centre in the reach (attained); or an extent known: within a resolution of a value attained at a centre in the
//! reach -- or until it is within the rounding of its centre's value. No point is sampled as a bound. (A threshold near the
//! quantity's true largest value is the expensive question: every cell along the place of that value has to shrink to their
//! difference. A range is asked as an extent, never by thresholds closing in on it.)
//!
//! All the quantities of a model share one tree of cells: a cell's beams, beam pairs and the parts the units' bounds have in
//! common (the constant forms' energies and their slopes, the carriers' couplings, resolvents and dressings, the clusters'
//! force bounds, each ground state's drives' gates, the levels' slope and the atoms' acceleration) are made once, for
//! whichever quantity asks first, and each quantity halves only the cells it still needs -- one decided on the coarse cells
//! costs nothing further. What the bounds read of the model itself -- its forms' entries that are not zero, with their moduli
//! (`Forms`) -- is made once for the model.
//!
//! # Rounds (`Model::proven_over`)
//! A quantity's proof is a sequence -- its next cell is the one of its largest bound among those its earlier halvings left --,
//! but what a cell holds is a function of the cell alone, and so is a quantity's value on it. The proofs therefore go on in
//! rounds. In a round every proof still open goes on, in its own order, as far as the halves already made for it reach, and
//! then names the cell its order has come to. Where the quantities may be asked from any thread (`AtOnce`:
//! `Model::bounds_over_in_parallel`) it names, ahead of its order, every half the round before made for it whose bound is not
//! settled: whatever the order, each of these has to be halved before the proof can end with its bound settled everywhere, so
//! the cells are halved level by level and the rounds are as many as the cells are deep. The cells named are halved once each,
//! for all the quantities that named them, one after another or on all the cores at once, as the proofs' own steps are. Each
//! proof takes its cells in the order it would take them alone, through the same queue, so its bound is the same number
//! whatever was halved ahead of it and on however many cores. Only a proof that ends otherwise has had cells halved in vain.
//! One whose bound is attained at a centre: the cells not settled above the level at which a centre attains it, for the halves
//! made hold that centre's value from then on (`Proof::sighted`), every cell is settled against it, and the proof asks ahead no
//! more. One that ends at a cell whose bound is within the rounding of its centre's value: the cells it had asked for by then.

use super::eliminate::{CellEnergies, Couplings};
use super::far::EXCITED;
use super::{Component, Envelope, Level, Lights, Model, Reach, Trap};
use crate::field::Geometry;
use crate::linalg::{norm3, Mat, C64};
use crate::request::{GROUND, STATES};
use crate::units::{rounding, KT_PER_UK};
use rayon::prelude::*;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::{Arc, Mutex, OnceLock};

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

/// One weighting w of the far light's static forms P_l'l (an entry's, a diagonal difference's), as a sum over the beam pairs
/// reads it: per pair of non-zero weight, in the pairs' order, its frequency and its place among the frequency's pairs, w(P_l'l)
/// and |w(P_l'l)|.
pub(crate) type Weights = Vec<(usize, usize, C64, f64)>;

/// A carrier's folded couplings X^f_l as the regions' bounds read them (eliminate.rs `Model::couplings`, `Model::shift_slope`).
pub(crate) struct Folded {
    /// Its polarization units, distinct and ascending.
    pub(crate) units: Vec<usize>,
    /// [ground state][beam]: ‖X^f_l[:, s]‖.
    pub(crate) columns: Vec<Vec<f64>>,
    /// [ground state][unit]: the unit's beams of a non-zero column, in their order -- the beam's place in the carrier's, and its
    /// column's entries that are not zero: the excited state (less GROUND), X^f_l[e, s] and its modulus.
    pub(crate) beams: Vec<Vec<Vec<(usize, Vec<(usize, C64, f64)>)>>>,
    /// [ground state]: the excited states (less GROUND) its beams' couplings reach from it, ascending.
    pub(crate) reached: Vec<Vec<usize>>,
}

/// One drive of a ground state as its gate reads it (units.rs `Model::folds`), in the tail rule's order.
pub(crate) struct DriveForm {
    /// Its component (2 frequency + counter), its excited state (less GROUND), its level and its place in the tail rule's
    /// order (units.rs `Ranked::group`).
    pub(crate) component: usize,
    pub(crate) excited: usize,
    pub(crate) level: Level,
    pub(crate) group: usize,
    /// Its beams' entries that are not zero: the request's beam, X_l[e, g] (of a counter-rotating one its conjugate, as its
    /// field enters) and its modulus.
    pub(crate) beams: Vec<(usize, C64, f64)>,
    /// With the bare levels: its detuning E_e − E_g − ν and its denominator's modulus |E_e − E_g − ν − iΓ_e/2|; Γ_e.
    pub(crate) detuning: f64,
    pub(crate) bare: f64,
    pub(crate) gamma: f64,
}

/// One excited state through which two frequencies' drives couple two ground states at the atom (`TermForm::legs`): the
/// emitted frequency's drive e ← a and the absorbed one's e ← b, each by its beams' entries (the request's beam, X_l and its
/// modulus) and, where it is folded with the levels at the atom, its frequency's detuning δ.
pub(crate) struct TermLeg {
    pub(crate) excited: usize,
    pub(crate) to: Vec<(usize, C64, f64)>,
    pub(crate) from: Vec<(usize, C64, f64)>,
    pub(crate) to_atom: Option<f64>,
    pub(crate) from_atom: Option<f64>,
}

/// A coupling of two states a ← b at one of the model's beats (`Unit::Term`) as its bound reads it (units.rs
/// `Model::term_bound`).
pub(crate) struct TermForm {
    /// Its constant part: at the zero beat the two states' own coupling with every light off (`Model::base`); and per pair
    /// of beams whose entry of the frequencies' constant form (far.rs `constant`) is not zero, the emitted beam and the
    /// absorbed one (the request's), C_ji[a, b] and its modulus.
    pub(crate) base: C64,
    pub(crate) pairs: Vec<(usize, usize, C64, f64)>,
    /// Its part at the atom, of two ground states: per excited state a drive folded there reaches from one of them and a
    /// folded drive of the other frequency from the other.
    pub(crate) legs: Vec<TermLeg>,
    /// The largest |k_i − k_j| of its beam pairs (1/µm).
    pub(crate) wave: f64,
}

impl TermForm {
    /// Whether nothing couples the two states at the beat.
    pub(crate) fn is_empty(&self) -> bool {
        self.base == C64::ZERO && self.pairs.is_empty() && self.legs.is_empty()
    }

    /// An entry of two frequencies' constant form within the rounding of the form's own sums is that rounding, not a
    /// coupling (units.rs `rounding`: a quantity within γ_N of its operands' scale) -- a field's component that is zero comes
    /// as the last bits of its vector's, and so does every product with it: per pair of basis vectors two coefficients times
    /// an entry, itself the outside forms' nine terms and two folds' sums over the excited states, of the forms' largest
    /// entry.
    fn rounding_of(model: &Model, to: usize, from: usize, pairs: &[Mat]) -> f64 {
        let terms = model.tones[to].basis[0].len() * model.tones[from].basis[0].len() * (2 * EXCITED + 11);
        rounding(terms) * pairs.iter().map(|m| m.max_abs()).fold(0.0, f64::max)
    }

    fn new(model: &Model, a: usize, b: usize, beat: usize) -> TermForm {
        let mut out = TermForm { base: C64::ZERO, pairs: vec![], legs: vec![], wave: 0.0 };
        let between = |lj: usize, li: usize| norm3(model.wave_value(&model.pair_wave(lj, li)));
        // The states' own coupling with every light off is H0, diagonal, less the outside forms at the focus: theirs.
        let own = model.base[(a, b)];
        if model.beats[beat].0.is_zero() && own.abs() > (0..model.tones.len()).map(|f| TermForm::rounding_of(model, f, f, &model.tones[f].pairs)).fold(0.0, f64::max) {
            out.base = own;
        }
        for &(to, from) in &model.beats[beat].1 {
            let (bt, bf) = (&model.light.frequencies[to].beams, &model.light.frequencies[from].beams);
            let constants = model.constants(to, from);
            let least = TermForm::rounding_of(model, to, from, &constants.pairs);
            for (x, m) in constants.pairs.iter().enumerate() {
                let v = m[(a, b)];
                if v.abs() > least {
                    let (lj, li) = (bt[x / bf.len()], bf[x % bf.len()]);
                    out.pairs.push((lj, li, v, v.abs()));
                    out.wave = out.wave.max(between(lj, li));
                }
            }
            if a >= GROUND || b >= GROUND || (a == b && to == from) {
                continue;
            }
            let (ct, cf) = (Component { frequency: to, counter: false }, Component { frequency: from, counter: false });
            let entries = |c: Component, e: usize, s: usize| -> Vec<(usize, C64, f64)> {
                let beams = &model.light.frequencies[c.frequency].beams;
                beams.iter().zip(&model.tones[c.frequency].couplings[0]).map(|(&l, x)| (l, x[(e, s)], x[(e, s)].abs())).filter(|entry| entry.2 != 0.0).collect()
            };
            for e in GROUND..STATES {
                let (to_beams, from_beams) = (entries(ct, e, a), entries(cf, e, b));
                if to_beams.is_empty() || from_beams.is_empty() {
                    continue;
                }
                let (lt, lf) = (model.level(ct, e, a), model.level(cf, e, b));
                if lt == Level::Explicit || lf == Level::Explicit || (lt == Level::Bare && lf == Level::Bare) {
                    continue;
                }
                for &(lj, ..) in &to_beams {
                    for &(li, ..) in &from_beams {
                        out.wave = out.wave.max(between(lj, li));
                    }
                }
                out.legs.push(TermLeg {
                    excited: e - GROUND,
                    to: to_beams,
                    from: from_beams,
                    to_atom: (lt == Level::Atom).then(|| model.tones[to].detuning),
                    from_atom: (lf == Level::Atom).then(|| model.tones[from].detuning),
                });
            }
        }
        out
    }
}

/// What the regions' bounds read of the model's forms -- the constant forms P_l'l of each frequency (far.rs), the base, the
/// carriers' couplings folded at the atom (eliminate.rs), the ground states' drives and the couplings of two states (units.rs)
/// --, each as its bound sums it: the entries that are not zero, with their moduli. The same for every region of a model, it
/// is the model's (`Model::forms`), each part made when first asked.
pub(crate) struct Forms {
    /// [frequency][pair][state]: [P_l'l]_ss and its modulus.
    levels: OnceLock<Vec<Vec<[(C64, f64); STATES]>>>,
    /// Per excited state x (less GROUND) the excited states y ≠ x it is coupled to somewhere -- by the base or by a form --,
    /// ascending, each with the base's entry.
    mixed: OnceLock<Vec<Vec<(usize, C64)>>>,
    /// Per pair of states (x, y), row-major: the weighting of the forms' entry [P_l'l]_xy.
    entries: Vec<OnceLock<Weights>>,
    /// Per pair of states (x, y), row-major: the weighting of the forms' diagonal difference [P_l'l]_xx − [P_l'l]_yy (y = x: of
    /// the diagonal [P_l'l]_xx).
    differences: Vec<OnceLock<Weights>>,
    /// Per carrier.
    folded: OnceLock<Vec<Folded>>,
    /// Per ground state its drives in the tail rule's order (units.rs `Model::ranked`); and the excited states (less GROUND)
    /// its folded drives reach, ascending.
    drives: OnceLock<(Vec<Vec<DriveForm>>, Vec<Vec<usize>>)>,
    /// Per coupling asked.
    terms: Mutex<HashMap<Coupling, Arc<TermForm>>>,
    /// Γ_min (eliminate.rs `Model::least_decay`).
    decay: OnceLock<f64>,
}

impl Forms {
    pub(crate) fn new() -> Forms {
        let cells = || (0..STATES * STATES).map(|_| OnceLock::new()).collect();
        Forms { levels: OnceLock::new(), mixed: OnceLock::new(), entries: cells(), differences: cells(), folded: OnceLock::new(), drives: OnceLock::new(), terms: Mutex::new(HashMap::new()), decay: OnceLock::new() }
    }

    /// A weighting of the model's far static forms (`Weights`).
    fn weights(model: &Model, w: &dyn Fn(&Mat) -> C64) -> Weights {
        let mut out = vec![];
        for (frequency, tone) in model.tones.iter().enumerate() {
            for (pair, p) in tone.pairs.iter().enumerate() {
                let x = w(p);
                if x != C64::ZERO {
                    out.push((frequency, pair, x, x.abs()));
                }
            }
        }
        out
    }

    /// The far static forms' diagonals with their moduli, [frequency][pair][state].
    pub(crate) fn levels(&self, model: &Model) -> &[Vec<[(C64, f64); STATES]>] {
        self.levels.get_or_init(|| model.tones.iter().map(|t| t.pairs.iter().map(|p| std::array::from_fn(|s| (p[(s, s)], p[(s, s)].abs()))).collect()).collect())
    }

    /// Per excited state (less GROUND) the excited states it is coupled to somewhere, with the base's entries (`Forms::mixed`).
    pub(crate) fn mixed(&self, model: &Model) -> &[Vec<(usize, C64)>] {
        self.mixed.get_or_init(|| {
            (GROUND..STATES)
                .map(|x| (GROUND..STATES).filter(|&y| y != x).map(|y| (y, model.base[(x, y)])).filter(|&(y, base)| base != C64::ZERO || !self.entry(model, x, y).is_empty()).collect())
                .collect()
        })
    }

    /// The weighting of the forms' entry [P_l'l]_xy.
    pub(crate) fn entry(&self, model: &Model, x: usize, y: usize) -> &Weights {
        self.entries[x * STATES + y].get_or_init(|| Forms::weights(model, &|p| p[(x, y)]))
    }

    /// The weighting of the forms' diagonal difference [P_l'l]_xx − [P_l'l]_yy; of the diagonal [P_l'l]_xx where y = x.
    pub(crate) fn difference(&self, model: &Model, x: usize, y: usize) -> &Weights {
        self.differences[x * STATES + y].get_or_init(|| Forms::weights(model, &|p| p[(x, x)] - if y == x { C64::ZERO } else { p[(y, y)] }))
    }

    /// Per carrier its couplings folded at the atom (`Folded`).
    pub(crate) fn folded(&self, model: &Model) -> &[Folded] {
        self.folded.get_or_init(|| {
            model
                .carriers
                .iter()
                .map(|cc| {
                    let mut units = cc.units.clone();
                    units.sort();
                    units.dedup();
                    let entries = |l: usize, s: usize| -> Vec<(usize, C64, f64)> {
                        (0..EXCITED).map(|e| (e, cc.folded[l][(GROUND + e, s)])).filter(|&(_, v)| v != C64::ZERO).map(|(e, v)| (e, v, v.abs())).collect()
                    };
                    let columns: Vec<Vec<f64>> =
                        (0..GROUND).map(|s| cc.folded.iter().map(|x| (0..EXCITED).map(|e| x[(GROUND + e, s)].norm_sqr()).sum::<f64>().sqrt()).collect()).collect();
                    let beams = (0..GROUND)
                        .map(|s| units.iter().map(|&unit| (0..cc.folded.len()).filter(|&l| cc.units[l] == unit && columns[s][l] != 0.0).map(|l| (l, entries(l, s))).collect()).collect())
                        .collect();
                    let reached = (0..GROUND).map(|s| (0..EXCITED).filter(|&e| cc.folded.iter().any(|m| m[(GROUND + e, s)] != C64::ZERO)).collect()).collect();
                    Folded { units, columns, beams, reached }
                })
                .collect()
        })
    }

    fn driven(&self, model: &Model) -> &(Vec<Vec<DriveForm>>, Vec<Vec<usize>>) {
        self.drives.get_or_init(|| {
            let drives: Vec<Vec<DriveForm>> = model
                .ranked
                .iter()
                .enumerate()
                .map(|(g, ranked)| {
                    ranked
                        .iter()
                        .map(|d| {
                            let side = d.component.counter as usize;
                            let (f, tone) = (&model.light.frequencies[d.component.frequency], &model.tones[d.component.frequency]);
                            let beams = f.beams.iter().zip(&tone.couplings[side]).map(|(&l, x)| (l, x[(d.excited, g)])).filter(|entry| entry.1 != C64::ZERO);
                            DriveForm {
                                component: 2 * d.component.frequency + side,
                                excited: d.excited - GROUND,
                                level: model.level(d.component, d.excited, g),
                                group: d.group,
                                beams: beams.map(|(l, x)| (l, if side == 1 { x.conj() } else { x }, x.abs())).collect(),
                                detuning: model.energies[d.excited] - model.energies[g] - tone.nu[side],
                                bare: model.bare_distance(d.component, d.excited, g),
                                gamma: model.decay[(d.excited - GROUND, d.excited - GROUND)].re,
                            }
                        })
                        .collect()
                })
                .collect();
            let reached = drives.iter().map(|of| (0..EXCITED).filter(|&e| of.iter().any(|d| d.excited == e && d.level != Level::Explicit)).collect()).collect();
            (drives, reached)
        })
    }

    /// Per ground state its drives in the tail rule's order (`DriveForm`).
    pub(crate) fn drives(&self, model: &Model) -> &[Vec<DriveForm>] {
        &self.driven(model).0
    }

    /// Per ground state the excited states (less GROUND) its folded drives reach, ascending.
    pub(crate) fn reached(&self, model: &Model) -> &[Vec<usize>] {
        &self.driven(model).1
    }

    /// What the bound of the coupling a ← b at the model's beat `beat` reads (`TermForm`), made once.
    pub(crate) fn term(&self, model: &Model, a: usize, b: usize, beat: usize) -> Arc<TermForm> {
        if let Some(made) = self.terms.lock().expect("the couplings' forms' lock").get(&(a, b, beat)) {
            return made.clone();
        }
        let made = Arc::new(TermForm::new(model, a, b, beat));
        self.terms.lock().expect("the couplings' forms' lock").entry((a, b, beat)).or_insert(made).clone()
    }

    /// The least decay rate of the excited states, Γ_min.
    pub(crate) fn least_decay(&self, model: &Model) -> f64 {
        *self.decay.get_or_init(|| model.least_decay())
    }

}

/// A cell's beams, or its centre point's (its radius zero), with what the units' bounds share over it, each part made once for
/// every quantity that asks (module documentation, # Provable extrema): the reach's speeds over it, the beam pairs per frequency,
/// the constant forms' energies and their slopes, the carriers' coherent couplings, resolvents and dressings of the ground
/// states, the clusters' force bounds (eliminate.rs) and the ground states' drives' gates (units.rs).
pub(crate) struct Region {
    pub(crate) spans: Vec<Span>,
    /// The cell's radius (0: a point).
    pub(crate) rho: f64,
    /// The bound of the atoms' speed over it (µm/µs): the reach's at its least V_prep (module documentation, # The reach); a
    /// sample's own on its point.
    pub(crate) speed: f64,
    /// Per frequency, its beam pairs (j, i) row-major (`PairSpan`).
    pub(crate) pairs: Vec<Vec<PairSpan>>,
    /// What its bounds read of the model's forms.
    pub(crate) forms: Arc<Forms>,
    pub(crate) energies: OnceLock<Box<CellEnergies>>,
    /// Per carrier.
    pub(crate) couplings: OnceLock<Vec<Couplings>>,
    /// [carrier][ground state]: the bound of its dressing's resolvent (eliminate.rs `Model::resolvent`).
    pub(crate) resolvents: OnceLock<Vec<[f64; GROUND]>>,
    /// [carrier][ground state].
    pub(crate) dressings: OnceLock<Vec<[f64; GROUND]>>,
    /// Per cluster.
    pub(crate) forces: Vec<OnceLock<f64>>,
    /// Per ground state the first places of its drives' gates in the tail rule's order (units.rs `Model::folds`).
    pub(crate) drives: [OnceLock<Vec<[f64; 4]>>; GROUND],
    /// Per ground state: the slope of its level under the near light (units.rs `Model::near_slope`).
    pub(crate) slopes: [OnceLock<f64>; GROUND],
    /// Per ground state: the slopes of the constant forms' levels from it to the excited states its folded drives reach
    /// (units.rs `Model::level_slopes`).
    pub(crate) levels: Vec<OnceLock<[f64; EXCITED]>>,
    /// Per coupling of two states asked its saturation and sweep (units.rs `Model::term_bound`).
    pub(crate) terms: Mutex<HashMap<Coupling, (f64, f64)>>,
    /// The bounds of the slope of any two levels' difference and of an atom's acceleration (units.rs `Model::swept_over`).
    pub(crate) moving: OnceLock<(f64, f64)>,
}

/// A coupling of two states a ← b at a beat, its place among the model's beats (`Unit::Term`'s identity in a model).
pub(crate) type Coupling = (usize, usize, usize);

impl Region {
    /// A coupling's saturation and sweep over the region, made once for both of its gates' bounds.
    pub(crate) fn term(&self, coupling: Coupling, make: impl FnOnce() -> (f64, f64)) -> (f64, f64) {
        let terms = || self.terms.lock().expect("the couplings' lock");
        if let Some(made) = terms().get(&coupling) {
            return *made;
        }
        let made = make();
        terms().insert(coupling, made);
        made
    }

    /// The bound over the region of |Σ_l a_l x_l| of beams' entries (the request's beam, x_l, |x_l|), the beams coherent at the
    /// request's phases: its value at the centre plus the cell's radius times Σ_l |x_l| max|∇a_l|, at most every beam at its
    /// largest amplitude and in phase; on a point, the value there.
    pub(crate) fn coherent(&self, entries: &[(usize, C64, f64)]) -> f64 {
        let (mut centre, mut lipschitz, mut blind) = (C64::ZERO, 0.0, 0.0);
        for &(l, x, size) in entries {
            let span = &self.spans[l];
            centre += span.centre * x;
            lipschitz += size * span.lipschitz();
            blind += size * span.most;
        }
        if self.rho.is_finite() { (centre.abs() + self.rho * lipschitz).min(blind) } else { blind }
    }
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

/// A cell in a quantity's queue, ordered by its upper bound: the bound, the cell's node, the quantity's value at its centre, and
/// whether the centre is in the reach.
#[derive(Clone, Copy)]
struct Queued {
    upper: f64,
    node: usize,
    centre: f64,
    inside: bool,
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

/// What the cells of one proving are made from: the model and the reach, and the prepared potential's lights and envelope.
struct Cells<'a> {
    model: &'a Model,
    reach: &'a Reach,
    prepared: Lights,
    envelope: Envelope,
}

impl Cells<'_> {
    /// A cell's node, none where the cell cannot meet the reach: a lower bound of V_prep over it above the reach's level.
    fn make(&self, cell: Cell) -> Option<Node> {
        let (model, reach) = (self.model, self.reach);
        let level = reach.minimum + reach.energy;
        // The reach's speeds where V_prep is at least v.
        let speed = |v: f64| reach.speed.min((2.0 * model.hbar_over_mass * (level - v).max(0.0)).sqrt());
        let spans = model.spans(&cell);
        if !cell.bounded() {
            return Some(Node { cell, over: model.region(spans, reach.speed), at: None, inside: false, halves: None });
        }
        let centre = model.potential(cell.centre, &reach.populations, &self.prepared);
        let below = centre - prepared_slack(&self.envelope, &spans);
        if below > level {
            return None;
        }
        let at = model.region(spans.iter().map(Span::point).collect(), speed(centre));
        Some(Node { cell, over: model.region(spans, speed(below)), at: Some(at), inside: centre <= level, halves: None })
    }
}

/// How far V_prep(r; p) moves over a bounded cell from its centre's value: each beam pair's coefficient by at most its radius
/// (`pair_span`), weighted by the envelope's |Q_l'l| (the near trap light's included, potential.rs).
fn prepared_slack(envelope: &Envelope, spans: &[Span]) -> f64 {
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

/// A quantity on a cell: its value at the centre (−∞ on an unbounded cell, which has none) and its upper bound over the cell.
fn valued<Q: Fn(&Region) -> f64 + ?Sized>(q: &Q, node: &Node) -> (f64, f64) {
    (node.at.as_ref().map_or(f64::NEG_INFINITY, |r| q(r)), q(&node.over))
}

/// What ends a proof (`Model::proven_over`): `settled(the largest upper bound of the cells left, the largest value attained at a
/// centre in the reach)`, and the question's own size.
#[derive(Clone, Copy)]
struct Rule<'a> {
    settled: &'a (dyn Fn(f64, f64) -> bool + Sync),
    scale: f64,
}

/// One quantity's proof (module documentation, # Provable extrema, # Rounds): the cells left, by their upper bounds, and the
/// largest value at a centre in the reach among them and their parents; the halves made for it, by cell, each as it enters the
/// queue, until its order comes to their cell; the largest value at a centre in the reach among all the halves made for it,
/// those waiting too; the cells it has asked to be halved; and its bound once it is decided.
struct Proof {
    queue: BinaryHeap<Queued>,
    attained: f64,
    halved: HashMap<usize, [Option<Queued>; 2]>,
    sighted: f64,
    asked: HashSet<usize>,
    bound: Option<f64>,
}

impl Proof {
    /// A quantity's proof at the proofs' first cell (none: no cell meets the reach, the bound is 0); on an unbounded cell, which
    /// is not halved, the bound is the quantity's over it.
    fn new<Q: Fn(&Region) -> f64 + ?Sized>(q: &Q, first: Option<&Node>) -> Proof {
        let mut proof = Proof { queue: BinaryHeap::new(), attained: f64::NEG_INFINITY, halved: HashMap::new(), sighted: f64::NEG_INFINITY, asked: HashSet::new(), bound: None };
        match first {
            None => proof.bound = Some(0.0),
            Some(node) => {
                let (centre, upper) = valued(q, node);
                if node.cell.bounded() {
                    proof.push(Queued { upper, node: 0, centre, inside: node.inside });
                } else {
                    proof.bound = Some(upper);
                }
            }
        }
        proof
    }

    /// A cell into the queue, its centre's value attained where the centre is in the reach.
    fn push(&mut self, cell: Queued) {
        if cell.inside {
            self.attained = self.attained.max(cell.centre);
        }
        self.queue.push(cell);
    }

    /// The proof as far as the halves made for it reach -- with `made`, those the round before made --, and the cells it asks
    /// to be halved next, none once it is decided (module documentation, # Rounds): the cell its order has come to and, with
    /// `ahead`, each of the halves just made that is not settled against the largest value the halves made hold.
    fn advance(&mut self, made: Vec<(usize, [Option<Queued>; 2])>, rule: Rule, ahead: bool) -> Vec<usize> {
        let new: Vec<Queued> = made.iter().flat_map(|(_, halves)| halves.iter().flatten().copied()).collect();
        self.sighted = new.iter().filter(|half| half.inside).fold(self.sighted, |sighted, half| sighted.max(half.centre));
        self.halved.extend(made);
        while self.bound.is_none() {
            let Some(&top) = self.queue.peek() else {
                self.bound = Some(0.0);
                break;
            };
            if (rule.settled)(top.upper, self.attained) || top.upper - top.centre <= rounding(1) * rule.scale.max(top.centre.abs()) {
                self.bound = Some(top.upper);
                break;
            }
            let Some(halves) = self.halved.remove(&top.node) else {
                let unsettled = new.iter().filter(|half| ahead && !(rule.settled)(half.upper, self.sighted)).map(|half| half.node);
                return [top.node].into_iter().chain(unsettled).filter(|&cell| self.asked.insert(cell)).collect();
            };
            self.queue.pop();
            halves.into_iter().flatten().for_each(|half| self.push(half));
        }
        vec![]
    }
}

/// A cell that proofs ask to be halved in a round: its node, and the quantities asking.
struct Asked {
    node: usize,
    quantities: Vec<usize>,
}

/// A cell halved in a round: its halves where this round made them (none where a half cannot meet the reach), whether each
/// half's centre is in the reach, and per quantity asking its values on each half (`valued`).
struct Halved {
    made: Option<[Option<Node>; 2]>,
    inside: [bool; 2],
    values: Vec<[Option<(f64, f64)>; 2]>,
}

/// A round of the proofs (module documentation, # Rounds): what its cells are made from, the cells made before it, the quantities.
struct Round<'a, Q: ?Sized> {
    cells: &'a Cells<'a>,
    nodes: &'a [Node],
    quantities: &'a [&'a Q],
}

impl<Q: Fn(&Region) -> f64 + ?Sized> Round<'_, Q> {
    /// A cell halved for the quantities asking: its halves made where they are not yet, and those quantities' values on them.
    fn halve(&self, asked: &Asked) -> Halved {
        let node = &self.nodes[asked.node];
        let made = node.halves.is_none().then(|| node.cell.halves().map(|half| self.cells.make(half)));
        let halves: [Option<&Node>; 2] = match &made {
            Some(made) => made.each_ref().map(Option::as_ref),
            None => node.halves.expect("the halves made in an earlier round").map(|at| at.map(|at| &self.nodes[at])),
        };
        let values = asked.quantities.iter().map(|&q| halves.map(|half| half.map(|half| valued(self.quantities[q], half)))).collect();
        Halved { inside: halves.map(|half| half.is_some_and(|half| half.inside)), made, values }
    }
}

/// The pace of a proving (module documentation, # Rounds): whether its proofs ask for cells ahead of their order, and how a
/// round's independent pieces are gone through -- the proofs' own steps (each with what the round before made for it), the
/// halving of the cells asked for --, and the cells let go in the end.
trait Pace<Q: ?Sized> {
    const AHEAD: bool;
    fn proofs<T: Send, R: Send>(proofs: &mut [Proof], made: Vec<T>, step: impl Fn(&mut Proof, T) -> R + Sync) -> Vec<R>;
    fn cells(round: &Round<Q>, asked: &[Asked]) -> Vec<Halved>;
    fn release(nodes: Vec<Node>);
}

/// One after another, no cell ahead of a proof's order: the pace of quantities that only their own thread may ask.
struct InTurn;

impl<Q: Fn(&Region) -> f64 + ?Sized> Pace<Q> for InTurn {
    const AHEAD: bool = false;
    fn proofs<T: Send, R: Send>(proofs: &mut [Proof], made: Vec<T>, step: impl Fn(&mut Proof, T) -> R + Sync) -> Vec<R> {
        proofs.iter_mut().zip(made).map(|(proof, made)| step(proof, made)).collect()
    }
    fn cells(round: &Round<Q>, asked: &[Asked]) -> Vec<Halved> {
        asked.iter().map(|cell| round.halve(cell)).collect()
    }
    fn release(nodes: Vec<Node>) {
        drop(nodes)
    }
}

/// On all the cores at once, each proof asking ahead of its order: the pace of quantities that any thread may ask.
struct AtOnce;

impl<Q: Fn(&Region) -> f64 + Sync + ?Sized> Pace<Q> for AtOnce {
    const AHEAD: bool = true;
    fn proofs<T: Send, R: Send>(proofs: &mut [Proof], made: Vec<T>, step: impl Fn(&mut Proof, T) -> R + Sync) -> Vec<R> {
        proofs.par_iter_mut().zip(made).map(|(proof, made)| step(proof, made)).collect()
    }
    fn cells(round: &Round<Q>, asked: &[Asked]) -> Vec<Halved> {
        asked.par_iter().map(|cell| round.halve(cell)).collect()
    }
    fn release(nodes: Vec<Node>) {
        nodes.into_par_iter().for_each(drop)
    }
}

/// Whether a quantity's proof against a threshold is decided (`Model::bounds_over`), by the largest upper bound of the cells left
/// and the largest value attained at a centre in the reach: below the threshold everywhere, or attained at or above it.
fn decided(threshold: f64) -> impl Fn(f64, f64) -> bool + Sync {
    move |upper, attained| upper < threshold || attained >= threshold
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
            forms: self.forms.clone(),
            energies: OnceLock::new(),
            couplings: OnceLock::new(),
            resolvents: OnceLock::new(),
            dressings: OnceLock::new(),
            forces: (0..self.clusters.len()).map(|_| OnceLock::new()).collect(),
            drives: std::array::from_fn(|_| OnceLock::new()),
            slopes: std::array::from_fn(|_| OnceLock::new()),
            levels: (0..GROUND).map(|_| OnceLock::new()).collect(),
            terms: Mutex::new(HashMap::new()),
            moving: OnceLock::new(),
        }
    }

    /// The region of a point alone at one speed (a sample's: the errors measured there).
    pub(crate) fn point_region(&self, r: [f64; 3], speed: f64) -> Region {
        self.region(self.spans(&Cell { centre: r, half: [0.0; 3] }), speed)
    }

    /// The provable bounds over the reach of quantities given per region, each asked against `threshold` (module documentation,
    /// # Provable extrema): each one's largest upper bound of the cells left when it is decided -- below the threshold everywhere,
    /// or attained at or above it at a centre in the reach --, or within the rounding of its centre's value; 0 where no cell meets
    /// the reach. The cells are halved one after another (`InTurn`).
    pub(crate) fn bounds_over(&self, reach: &Reach, threshold: f64, quantities: &[&dyn Fn(&Region) -> f64]) -> Vec<f64> {
        self.proven_over::<_, InTurn>(reach, Rule { settled: &decided(threshold), scale: threshold.abs() }, quantities)
    }

    /// The bounds against `threshold` (`bounds_over`) of quantities any thread may ask, proven on all the cores (`AtOnce`): the
    /// same bounds, to the bit.
    pub(crate) fn bounds_over_in_parallel(&self, reach: &Reach, threshold: f64, quantities: &[&(dyn Fn(&Region) -> f64 + Sync)]) -> Vec<f64> {
        self.proven_over::<_, AtOnce>(reach, Rule { settled: &decided(threshold), scale: threshold.abs() }, quantities)
    }

    /// The provable bounds over the reach of quantities given per region, each to `resolution`: each one's largest upper bound of
    /// the cells left once it is within the resolution of a value attained at a centre in the reach. The cells are halved one
    /// after another (`InTurn`).
    pub(crate) fn extents_over(&self, reach: &Reach, resolution: f64, quantities: &[&dyn Fn(&Region) -> f64]) -> Vec<f64> {
        self.proven_over::<_, InTurn>(reach, Rule { settled: &|upper, attained| upper - attained <= resolution, scale: resolution }, quantities)
    }

    /// The proofs' subdivision (module documentation, # Provable extrema, # Rounds): per quantity the cell of its largest upper
    /// bound is halved until the bound is settled against the largest value attained at a centre in the reach so far, the cell
    /// is unbounded, or the bound is within the rounding of its centre's value (of the question's own size, where that is
    /// larger: `rule`). The proofs go on in rounds, at the pace `P`.
    fn proven_over<Q: Fn(&Region) -> f64 + ?Sized, P: Pace<Q>>(&self, reach: &Reach, rule: Rule, quantities: &[&Q]) -> Vec<f64> {
        let cells = Cells { model: self, reach, prepared: self.prepared(), envelope: self.envelope(&reach.populations) };
        let half = reach.cell.map(|[lo, hi]| 0.5 * (hi - lo));
        let start = Cell { centre: reach.cell.map(|[lo, hi]| if (hi - lo).is_finite() { 0.5 * (lo + hi) } else { 0.0 }), half };
        let mut nodes: Vec<Node> = cells.make(start).into_iter().collect();
        let mut proofs: Vec<Proof> = quantities.iter().map(|q| Proof::new(*q, nodes.first())).collect();
        // Per proof, the halves the round before made for it.
        let nothing = || -> Vec<Vec<(usize, [Option<Queued>; 2])>> { quantities.iter().map(|_| vec![]).collect() };
        let mut made = nothing();
        loop {
            let asking = P::proofs(&mut proofs, made, |proof, made| proof.advance(made, rule, P::AHEAD));
            // The cells asked for, each with the quantities asking.
            let mut by_cell: Vec<(usize, usize)> = asking.iter().enumerate().flat_map(|(q, cells)| cells.iter().map(move |&node| (node, q))).collect();
            if by_cell.is_empty() {
                break;
            }
            by_cell.sort_unstable();
            let mut asked: Vec<Asked> = vec![];
            for (node, q) in by_cell {
                match asked.last_mut() {
                    Some(cell) if cell.node == node => cell.quantities.push(q),
                    _ => asked.push(Asked { node, quantities: vec![q] }),
                }
            }
            let halved = P::cells(&Round { cells: &cells, nodes: &nodes, quantities }, &asked);
            made = nothing();
            for (cell, halved) in asked.iter().zip(halved) {
                let halves = match halved.made {
                    Some(new) => {
                        let at = new.map(|half| {
                            half.map(|half| {
                                nodes.push(half);
                                nodes.len() - 1
                            })
                        });
                        nodes[cell.node].halves = Some(at);
                        at
                    }
                    None => nodes[cell.node].halves.expect("the halves made in an earlier round"),
                };
                for (&q, values) in cell.quantities.iter().zip(halved.values) {
                    let queued = std::array::from_fn(|x| halves[x].zip(values[x]).map(|(node, (centre, upper))| Queued { upper, node, centre, inside: halved.inside[x] }));
                    made[q].push((cell.node, queued));
                }
            }
        }
        P::release(nodes);
        proofs.into_iter().map(|proof| proof.bound.expect("a proof left open asks for a cell")).collect()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::Light;
    use crate::model::far::{pair, EXCITED};
    use crate::model::{Component, Unit};
    use crate::request::{Request, STATES};

    /// A reference scene's model with every drive of every co-rotating component but the trap's folded with the levels at
    /// the atom.
    fn at_the_atom(scene: &str) -> Model {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("scenes").join(format!("{scene}.json"));
        let line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let request = Request::from_native(&line).unwrap();
        let light = Light::new(&request);
        let bare = Model::new(&request, &light, &[]).unwrap();
        let kept: Vec<Unit> = (0..light.frequencies.len()).filter(|&f| f != light.trap).flat_map(|frequency| bare.bare_units(Component { frequency, counter: false })).collect();
        Model::new(&request, &light, &kept).unwrap()
    }

    /// R8_R5051: a six-beam lattice of cooling light with its repump, in a tweezer.
    fn lattice() -> Model {
        at_the_atom("R8_R5051")
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

    /// One tree of cells for many quantities proves each as its own proof would, and so do its rounds' cells halved on all the
    /// cores, the proofs asking ahead of their order: the shared cells and the cells made ahead change no bound (R3_DOPPLER's
    /// cooling pair's drives at the atom: the strongest drive's saturation and sweep from two ground states).
    #[test]
    fn one_tree_proves_each_quantity_as_its_own_proof_would() {
        let model = at_the_atom("R3_DOPPLER");
        let b = 7;
        let p = {
            let mut p = [0.0; GROUND];
            p[b] = 1.0;
            p
        };
        let trap = model.trap(&p, &model.prepared()).unwrap();
        let reach = model.reach(&trap, &p, 20.0, 2e-3);
        let s = |r: &Region| model.folds(b, 0, r, usize::MAX, 0.0)[0];
        let a = |r: &Region| model.folds(b, 0, r, usize::MAX, 0.0)[1];
        let other = |r: &Region| model.folds(3, 0, r, usize::MAX, 0.0)[0];
        let together = model.bounds_over(&reach, 0.5, &[&s, &a, &other]);
        let alone: Vec<f64> = [&s as &dyn Fn(&Region) -> f64, &a, &other].iter().map(|q| model.bounds_over(&reach, 0.5, &[*q])[0]).collect();
        assert_eq!(together, alone);
        assert_eq!(model.bounds_over_in_parallel(&reach, 0.5, &[&s, &a, &other]), alone);
    }
}
