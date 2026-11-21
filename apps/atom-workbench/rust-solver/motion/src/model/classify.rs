//! The model's one judgment (DESIGN2 §4.1–4.3, BRIEF_JUDGE item 1): every term of the run's model in one of four tiers
//! -- zero, drop, fold, keep -- made once, when the model is made, and read by everything that once judged a small
//! number on its own: the beats a node holds (`model/selected.rs`, what `prune` was), the exact blocks (what `split`'s
//! joins were), the candidate tones (the beats kept), and the fold's input (`input`, what `rs.rs` `significant` was).
//! A step judges nothing; it reads the tiers.
//!
//! # The tiers
//! - **Zero**: what the computation itself cannot tell from nothing. On the input side the user's rule: a coupling
//!   entry whose first-order amplitude a = |M_ij| / |E_i − E_j − f + i(Γ_i + Γ_j)/2| is below ε (`input`). On the output
//!   side -- the model's node, P's slow elements -- the computation's own rounding and nothing else: an entry x that is
//!   the sum of K terms whose moduli sum to S is a rounding zero where |x| ≤ γ_K S (Higham's γ_K = Ku/(1 − Ku), units.rs
//!   `rounding`; `zero`). There is no second definition: an entry below ε times some matrix's largest is no zero by
//!   that alone (until 2026-11-21 the beats read "ε times the largest of every beat's" and the blocks "ε times its
//!   own slot's largest", one quantity with two maxima). S and K are read where the structure is read, the node at
//!   the trap's centre at rest at unit amplitude (`Selected::structure`, rs/fold.rs `Out::scales`); every beam's
//!   amplitude is one there, at its focus, its largest anywhere (field.rs), so S is the sum of the terms' moduli at
//!   their largest. A beat is zero where every entry of its value and of its gradient is.
//! - **Drop**: a whole beat pair ±Ω (the H_eff part with its share of the decay) left out of the model under a strict
//!   bound of what leaving it out does to every readout, booked in the run's error (`Judgment::dropped`; item 2).
//! - **Fold**: what the RS fold takes to second order -- Q, the states the selection leaves out (select.rs, with the
//!   strict fold condition before its points are judged), and P's fast elements (the frame's n ≠ 0, rs.rs # The fold).
//!   Both are decided before the model's node exists: the node holds P's slow elements alone, which are of the other
//!   three tiers.
//! - **Keep**: the rest.
//!
//! # What follows from the tiers, judged nowhere else
//! - the node's beats: the static one and every beat that is neither zero nor dropped;
//! - the exact blocks: P's connected parts under the kept beats' entries that are not zero, of the value and of the
//!   gradient along each lab axis, off the diagonal (an entry that is not a number joins);
//! - the candidate tones: the kept beats other than the static one, with their opposites (model/selected.rs `tones`).
//!
//! The zero tier is read at the trap's centre at rest, as the structure was before: an entry whose terms cancel there
//! alone and not elsewhere would read as zero there; the tests read the blocks at points off it
//! (model/selected/tests.rs `the_exact_blocks_hold_at_every_point`).

use crate::linalg::{C64, Mat};
use crate::model::selected::Node;
use crate::units::rounding;

/// The rounding scales of the structure's node (`Selected::structure`): per generator slot and per gradient slot and
/// axis, per place (n × n, row by row) the sum of the moduli of the terms its value is the sum of, and their count.
pub(crate) struct Scales {
    pub generator: Vec<(Vec<f64>, Vec<usize>)>,
    pub gradient: Vec<[(Vec<f64>, Vec<usize>); 3]>,
}

/// A term's tier (module documentation, # The tiers): the node's beats are zero, dropped or kept (the fold's own are
/// decided before the node exists).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    Zero,
    Drop,
    Keep,
}

/// Whether a computed entry x, the sum of `terms` terms whose moduli sum to `scale`, is a rounding zero: |x| ≤ γ_K S
/// (module documentation, # The tiers). One that is not a number is not.
pub fn zero(x: C64, scale: f64, terms: usize) -> bool {
    x.abs() <= rounding(terms) * scale
}

/// The fold's input by the zero tier's input side, the user's rule (module documentation, # The tiers): a coupling
/// of the fold's input at the frequency f (a term of H ∝ e^{−ift}, `Scene::terms` at rest) whose first-order
/// amplitude a = |M_ij| / |E_i − E_j − f + i(Γ_i + Γ_j)/2| is below the format's ε is no coupling and is zero -- on the
/// fixed matrix and its own denominator (the bare levels E = diag H0, the decay Γ = Σ_a C_a†C_a's diagonal), not on its
/// size against the matrix's largest entry: a small entry close to its resonance is an amplitude like any other, and a
/// near-resonant denominator would carry it out of the rounding. A coupling at its resonance (no denominator) always
/// counts.
pub fn input(m: &Mat, f: f64, energy: &[f64], width: &[f64]) -> Mat {
    Mat::from_fn(m.rows, m.cols, |i, j| {
        let z = m[(i, j)];
        let detuning = C64::new(energy[i] - energy[j] - f, 0.5 * (width[i] + width[j]));
        if z.abs() < f64::EPSILON * detuning.abs() { C64::ZERO } else { z }
    })
}

/// The run's judgment of its node's beats (module documentation): per slot of the node's generator its tier, P's exact
/// blocks under the kept beats, and what the dropped beats cost (`dropped`).
#[derive(Clone, Debug, PartialEq)]
pub struct Judgment {
    /// Per slot of the structure's node (the layout's order) its beat and tier; the static beat (slot 0) is kept.
    pub tiers: Vec<(usize, Tier)>,
    /// P's exact blocks: places in P, each sorted, in the order of their first place.
    pub blocks: Vec<Vec<usize>>,
    /// Y: the bound of the rate at which the dropped beats move the run's state in trace distance (1/µs); zero where
    /// nothing is dropped.
    pub dropped: f64,
}

impl Judgment {
    /// The judgment of the structure's node (`Selected::structure`: the trap's centre at rest, unit amplitude, the
    /// static beat with its frame's θ back on its diagonal) and its rounding scales: each beat zero where every entry of
    /// its value and of its gradient is a rounding zero (`zero`), else kept; the blocks of the kept beats. A node whose
    /// entries are not numbers is refused here (a model at unit amplitude is numbers).
    pub(crate) fn new(node: &Node, scales: &Scales) -> Judgment {
        let finite = |m: &Mat| m.data.iter().all(|z| z.re.is_finite() && z.im.is_finite());
        assert!(node.generator.iter().all(|(_, m)| finite(m)) && node.gradient.iter().all(|(_, g)| g.iter().all(finite)), "the node's coefficients at unit amplitude are not finite");
        let tiers: Vec<(usize, Tier)> = node
            .generator
            .iter()
            .enumerate()
            .map(|(k, (beat, m))| {
                let held = |m: &Mat, (scale, terms): &(Vec<f64>, Vec<usize>)| m.data.iter().zip(scale).zip(terms).any(|((x, s), t)| !zero(*x, *s, *t));
                let moving = node.gradient.iter().position(|g| g.0 == *beat).is_some_and(|d| node.gradient[d].1.iter().zip(&scales.gradient[d]).any(|(g, s)| held(g, s)));
                (*beat, if k == 0 || held(m, &scales.generator[k]) || moving { Tier::Keep } else { Tier::Zero })
            })
            .collect();
        let mut judgment = Judgment { tiers, blocks: Vec::new(), dropped: 0.0 };
        judgment.blocks = judgment.joined(node, scales);
        judgment
    }

    /// Whether slot k is kept.
    pub fn kept(&self, k: usize) -> bool {
        self.tiers[k].1 == Tier::Keep
    }

    /// P's connected parts under the kept beats' entries that are not rounding zeros, of the value and of the gradient
    /// along each lab axis, off the diagonal (module documentation, # What follows from the tiers).
    fn joined(&self, node: &Node, scales: &Scales) -> Vec<Vec<usize>> {
        let n = node.generator[0].1.rows;
        let mut root: Vec<usize> = (0..n).collect();
        fn find(root: &mut [usize], mut i: usize) -> usize {
            while root[i] != i {
                root[i] = root[root[i]];
                i = root[i];
            }
            i
        }
        let mut read = |m: &Mat, (scale, terms): &(Vec<f64>, Vec<usize>)| {
            for a in 0..n {
                for b in (0..n).filter(|&b| b != a) {
                    if !zero(m[(a, b)], scale[a * n + b], terms[a * n + b]) {
                        let (x, y) = (find(&mut root, a), find(&mut root, b));
                        if x != y {
                            root[x.max(y)] = x.min(y);
                        }
                    }
                }
            }
        };
        for (k, (beat, m)) in node.generator.iter().enumerate().filter(|(k, _)| self.kept(*k)) {
            read(m, &scales.generator[k]);
            if let Some(d) = node.gradient.iter().position(|g| g.0 == *beat) {
                for (g, s) in node.gradient[d].1.iter().zip(&scales.gradient[d]) {
                    read(g, s);
                }
            }
        }
        let mut parts: std::collections::BTreeMap<usize, Vec<usize>> = std::collections::BTreeMap::new();
        for i in 0..n {
            let r = find(&mut root, i);
            parts.entry(r).or_default().push(i);
        }
        parts.into_values().collect()
    }
}
