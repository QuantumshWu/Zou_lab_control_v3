//! The model's one judgment (DESIGN2 §4.1–4.3, BRIEF_JUDGE item 1): every term of the run's model in one of four tiers
//! -- zero, drop, fold, keep -- made once, when the model is made, and read by everything that once judged a small
//! number on its own: the beats a node holds (`model/selected.rs`, what `prune` was), the exact blocks (what `split`'s
//! joins were), the candidate tones (the beats kept), and the fold's input (`input`, what `rs.rs` `significant` was).
//! A step judges nothing; it reads the tiers.
//!
//! # The tiers
//! - **Zero**: what the computation itself cannot tell from nothing. On the input side the user's rule: a coupling
//!   entry whose first-order amplitude a = |M_ij| / |E_i − E_j − f + i(Γ_i + Γ_j)/2| is below ε (`input`), a jump's
//!   amplitude whose share of its state's decay amplitude is (`emission_input`). On the output side -- every matrix
//!   the engine makes of those, the fixed matrices of the folds (rs.rs `Scaled`: the photon forms, the decay, the
//!   spherical channels, the tweezer's unit fold) and the model's node, P's slow elements -- the computation's own
//!   rounding and nothing else (until 2026-11-21 the fixed matrices' was u times their largest entry): an entry x that is
//!   the sum of K terms whose moduli sum to S is a rounding zero where |x| ≤ γ_K S (Higham's γ_K = Ku/(1 − Ku), units.rs
//!   `rounding`; `zero`). There is no second definition: an entry below ε times some matrix's largest is no zero by
//!   that alone (until 2026-11-21 the beats read "ε times the largest of every beat's" and the blocks "ε times its
//!   own slot's largest", one quantity with two maxima). S and K are read where the structure is read, the node at
//!   the trap's centre at rest at unit amplitude (`Selected::structure`, rs/fold.rs `Out::scales`); every beam's
//!   amplitude is one there, at its focus, its largest anywhere (field.rs), so S is the sum of the terms' moduli at
//!   their largest. A beat is zero where every entry of its value and of its gradient is.
//! - **Drop**: a whole beat pair ±Ω (the H_eff part with its share of the decay) left out of the model under a strict
//!   bound of what leaving it out does to every readout, booked in the run's error (`Judgment::dropped`; # The drop
//!   tier).
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
//! # The drop tier
//! The bound (DROPRES §2 with its corrections; DESIGN2 §2.1–2.3). The run's model M and M′, M less the dropped beats'
//! slots of the generator Ĥ = H_eff − (i/2)ΣL†L, both on the internal states ⊗ the centre of mass (taken quantum), the
//! loss a transfer into a lost sector the same in both: M's propagator Φ on held ⊕ lost is CPTP and contracts the trace
//! norm. Leaving a beat's decay share without its channels' components makes M′ no Lindblad form (Φ′ need not
//! contract), so with M's Φ and ρ′(t), Δ(T) = ρ′ − ρ = ∫ Φ(T, t)(𝓛′ − 𝓛)ρ′(t) dt and
//! ‖(𝓛′ − 𝓛)ρ‖₁ = ‖i(δĤρ − ρδĤ†)‖₁ ≤ 2 sup‖δĤ‖ ‖ρ‖₁ (δĤ(r̂) a multiplication operator, its norm the sup over the reach
//! of the matrix's), Grönwall: ‖Δ(T)‖₁ ≤ e^{2YT} − 1, the trace distance D(T) = ½‖Δ‖₁ ≤ ½(e^{2YT} − 1) = YT(1 + O(YT)),
//! Y = Σ_dropped sup‖G_k‖ over their slots. No T² term: the dropped force is in [δĤ(r̂), ρ]; the exact propagation's
//! contraction takes every later consequence.
//! - **The H_eff part** of a slot: its entries' bounds over the reach (rs/fold.rs `Out::bounds`): the sums of their terms'
//!   moduli with every beam's amplitude at most one (its focus value: per beam, strict; the joint extremum of their
//!   interference is lower, DESIGN2 §5.3, and not needed), every coefficient at its largest, and the folded blocks'
//!   amplitudes at their bound over the reach (`Plan::reach`: the resolvent over every intensity in [0, 1] and every
//!   momentum below κ_max(s), by boxes and the Neumann series about each box's centre). ‖G‖ ≤ √(‖S‖₁‖S‖_∞) of the
//!   nonnegative bound S ≥ |G| entrywise.
//! - **The decay's share** of a slot: Σ over the decay's products ℓ_l†ℓ_r landing there of ½‖ℓ_l‖‖ℓ_r‖, each spherical
//!   channel component's norm by its own entries' bounds over the reach (the spherical fold's, whose fixed matrices hold
//!   the polarizations' combination exactly: the cancellation of the Cartesian axes is in the matrices, not in the
//!   bound). The products of a high beat are of a large component and a small one (tilt2's ±3ν: 10 × 10⁻¹³), so the
//!   bound by the channels' components is tight where the scale S of the summed products is not (CLUSTER: 10⁹ too
//!   large).
//! - **The readouts** (DROPRES §2.1, its corrections): a row as shown and the loss move by at most D; T by the solve's
//!   bound of a share D/(n − D) of the held atoms moved within the bound set (estimate.rs `Estimate::moved`; at rest and
//!   at T_sc, about c_T D of its allowance, c_T = 2K_max/(k_B T_sc), K_max = U₀ the prepared well's depth, E_cap = 0),
//!   T_sc at least 30 k_BT_rec of the emission's shortest wave number (estimate.rs `semiclassical`). The solve books
//!   them (solve.rs `bounded`, D by `distance`).
//! - **Which beats** (DESIGN2 §2.1, the marginal rule): the candidates in increasing Y, each prefix of them dropped or
//!   none; the drop the one of least W = w · max_l (E_l² − A_l²)^{−(1 + 1/2p)}, E the user's targets (units.rs: the
//!   populations absolute, T's allowance), A the selection's measured error with the drop's bound added (linearly: both
//!   are biases), w the step's counted work the beats change (the propagation's tone classes, spectral.rs # The
//!   second order by tones, beside the three nodes'). A gap of orders between the beats makes the choice "all below the
//!   gap": the drop's share of A is a tiny fraction of the target and any saving pays for it.
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

/// The integration-by-parts lemma (PIPELINE_V3_FINAL A.2), for a complex rate μ = −γ + iω, γ ≥ 0:
/// |∫_a^b c(u) e^{μu} du| = |[c e^{μu}/μ]_a^b − ∫ ċ e^{μu} du/μ| ≤ (|c(a)| + |c(b)| + ∫_a^b |ċ| du)/|μ|, |e^{μu}| ≤ 1 on
/// u ≥ 0 -- the turning and the decay together, |μ| = √(γ² + ω²) (for a real frequency Ω, |μ| = Ω). Of the ends'
/// sizes `ends` = |c(a)| + |c(b)|, the variation `variation` = ∫|ċ|, and the rate. Its one implementation, every bound
/// it gives computed here: a stretch's opening's booked modes (paths.rs `Form::opening`); a step's cut between two
/// groups (step.rs `Groups::choose`: on an envelope of the band β the lemma to every order, |c⁽ᵏ⁾| ≤ βᵏ sup|c| by
/// Bernstein, its ends' terms summed to 2 sup|c| at the rate Ω − β, within Bohr–Favard's π sup|c| taken as the ends);
/// the split's turning integrals of a constant envelope (spectral.rs `followed`, `Picture::remainder`). The fold's fast
/// elements and the photons' classes by their fast index are its consequence on the frame's lattice: they are decided
/// by the index (n ≠ 0, floquet/frame.rs `Frame::fast_index`) and compute no bound.
pub fn by_parts(ends: f64, variation: f64, rate: C64) -> f64 {
    (ends + variation) / rate.abs()
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

/// The emission by the zero tier's input side (module documentation, # The tiers): a jump's amplitude C_a[g, e] whose
/// share of its excited state's whole decay amplitude, |C_a[g, e]| / √Γ_e (Γ_e = Σ_a Σ_g |C_a[g, e]|², the state's
/// decay rate), is below the format's ε is no jump and is zero -- the first-order rule of `input` for a jump, whose
/// natural size is its state's decay, not a detuning. The request's matrix, given entry by entry: no computation of the
/// engine's makes its entries, so this is its one rule.
pub fn emission_input(c: &[Mat; 3]) -> [Mat; 3] {
    let n = c[0].cols;
    let decay: Vec<f64> = (0..n).map(|e| c.iter().map(|m| (0..m.rows).map(|g| m[(g, e)].norm_sqr()).sum::<f64>()).sum()).collect();
    c.each_ref().map(|m| Mat::from_fn(m.rows, m.cols, |g, e| if m[(g, e)].abs() < f64::EPSILON * decay[e].sqrt() { C64::ZERO } else { m[(g, e)] }))
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
    /// The drop tier's candidate pairs: per pair its slots and Y, the bound of its slots' sup norms over the reach
    /// (# The drop tier); none where the node holds no beat but the static one.
    pub pairs: Vec<(Vec<usize>, f64)>,
}

/// The trace distance by which dropped terms of the rate Y (1/µs) may move the state after a time t (µs), strict
/// (# The drop tier): ½(e^{2Yt} − 1) = tY(1 + O(tY)). The one implementation: the solve books it (solve.rs `Fixed`),
/// the drop tier weighs it (`Account`).
pub fn distance(rate: f64, t: f64) -> f64 {
    0.5 * (2.0 * rate * t).exp_m1()
}

/// A drop's readouts' errors and costs (# The drop tier): the selection's measured error -- on the populations
/// (absolute) and on T (relative; `Selected::error`) --, the run's length (µs), and what T's bound of a share of the
/// held atoms moved reads (estimate.rs `Estimate::moved`, the solve's own bound): the held atoms' hold (`Escape`, the
/// prepared well's depth) and T's semiclassical scale at its least (µK), where T's allowance is the scale itself.
#[derive(Clone, Copy, Debug)]
pub struct Account {
    pub populations: f64,
    pub temperature: f64,
    pub duration: f64,
    pub escape: crate::estimate::Escape,
    pub floor: f64,
}

impl Account {
    /// The trace distance after the run's length at the rate Y (`distance`).
    pub fn distance(&self, rate: f64) -> f64 {
        distance(rate, self.duration)
    }

    /// T's bound, relative to T, where the drop at the rate Y moves the held atoms by D (`distance`): the solve's
    /// (estimate.rs `Estimate::moved`, the share D/(n − D) of the held atoms, here n = 1) at T at its scale T_sc,
    /// the atoms at rest on average -- in units of T's allowance there, which is at least T_sc, at most that.
    fn temperature_moved(&self, d: f64) -> f64 {
        use crate::estimate::{Estimate, Motion, Spread};
        let unread = Spread::UNREAD;
        let motion = Motion { temperature: [self.floor; 3], temperature_errors: [unread; 3], temperature_total: self.floor, temperature_total_error: unread, occupation: [0.0; 3], mean_velocity: [0.0; 3], mean_speed: [0.0; 3] };
        let at = Estimate {
            trajectories: 0.0,
            held: 1.0,
            loss: 0.0,
            loss_error: unread,
            shown_errors: [unread; crate::request::STATES],
            populations: [0.0; crate::request::STATES],
            motion: Some(motion),
            failed: 0.0,
            uncounted: 0.0,
            temperature_target: crate::units::TEMPERATURE_TARGET,
            unresolved: 0.0,
        };
        at.moved(if d < 1.0 { d / (1.0 - d) } else { 1.0 }, &self.escape)
    }

    /// W's model-dependent factor max_l (E_l² − A_l²)^{−(1 + 1/2p)} with the drop at the rate Y (# The drop tier): on
    /// the populations E = `POPULATION_TARGET`, A the measured error plus D; on T, in units of its allowance (at least
    /// `TEMPERATURE_TARGET` of T and at least T_sc), E = 1, A the measured relative error over `TEMPERATURE_TARGET` plus
    /// the drop's bound (`temperature_moved`). Unbounded where an A reaches its E, or is not a number.
    pub fn factor(&self, rate: f64) -> f64 {
        use crate::units::{ORDER, POPULATION_TARGET, TEMPERATURE_TARGET};
        let d = self.distance(rate);
        let power = 1.0 + 1.0 / (2.0 * ORDER as f64);
        let room = [POPULATION_TARGET.powi(2) - (self.populations + d).powi(2), 1.0 - (self.temperature / TEMPERATURE_TARGET + self.temperature_moved(d)).powi(2)];
        room.iter().map(|&r| if r > 0.0 { r.powf(-power) } else { f64::INFINITY }).fold(0.0, crate::linalg::larger)
    }
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
        let mut judgment = Judgment { tiers, blocks: Vec::new(), dropped: 0.0, pairs: Vec::new() };
        judgment.blocks = judgment.joined(node, scales);
        judgment
    }

    /// The drop tier (# The drop tier): of the candidate pairs -- per pair its slots and Y, the bound of its slots' sup
    /// norms over the reach -- the prefix in increasing Y of least W, `work` the step's counted work with the slots left
    /// out (a pair's slots by their places); those slots dropped, the blocks made again of the beats kept, `dropped` their
    /// Y. A pair whose Y is not a number is never dropped.
    pub(crate) fn drop(&mut self, node: &Node, scales: &Scales, pairs: &[(Vec<usize>, f64)], account: &Account, work: &dyn Fn(&[usize]) -> f64) {
        let mut order: Vec<usize> = (0..pairs.len()).filter(|&i| pairs[i].1.is_finite()).collect();
        order.sort_by(|&a, &b| pairs[a].1.total_cmp(&pairs[b].1));
        let (mut best, mut least) = (0, work(&[]) * account.factor(0.0));
        let (mut rate, mut slots) = (0.0, Vec::new());
        for (m, &i) in order.iter().enumerate() {
            rate += pairs[i].1;
            slots.extend_from_slice(&pairs[i].0);
            let w = work(&slots) * account.factor(rate);
            if w < least {
                (best, least) = (m + 1, w);
            }
        }
        self.dropping(node, scales, pairs, &order[..best]);
    }

    /// The pairs `chosen` (indices into `pairs`) dropped: their slots' tiers, `dropped` their Y, the blocks of the beats
    /// kept.
    pub(crate) fn dropping(&mut self, node: &Node, scales: &Scales, pairs: &[(Vec<usize>, f64)], chosen: &[usize]) {
        self.pairs = pairs.to_vec();
        self.dropped = 0.0;
        for &i in chosen {
            self.dropped += pairs[i].1;
            for &k in &pairs[i].0 {
                self.tiers[k].1 = Tier::Drop;
            }
        }
        self.blocks = self.joined(node, scales);
    }

    /// The candidates' slots for the drop tier: the slots kept other than the static one.
    pub fn candidates(&self) -> Vec<usize> {
        (1..self.tiers.len()).filter(|&k| self.kept(k)).collect()
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
