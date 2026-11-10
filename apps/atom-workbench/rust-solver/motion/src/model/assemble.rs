//! The one generator assembly (SPEC §1.4): what a cluster's generator is made of under a light set, worked out once (`Plan`), and
//! its evaluation at a point. The dynamics, the force, the hazard, the photons, the readout and the potential take its terms.
//!
//! # Terms
//! Every term of the generator is a product of at most two components of the light's fields at the point times a matrix on the
//! cluster's states, at a base beat Ω (∝ e^{iΩt}), all exact (`Beat`, `Wave`). A field's component is u(r) = Σ_l a_l(r) c_l over
//! beams of one frequency (`Scalar`): its coherent field's component on its basis, e_k = Σ_l a_l γ_lk (field.rs; an averaged
//! light's, each beam's alone), a beam's own amplitude (the far beats'), or the conjugate of one. The terms:
//! - the base (H0 less the outside forms at the focus: every light off) and the frames' −θ_s, constant;
//! - each lit frequency's far static form, e_k'* e_k A_k'k (far.rs);
//! - the kept far beats, a_l₂* a_l₁ C_l₂l₁ and their adjoints (units.rs `Model::far_cross`);
//! - the near light's shifts and couplings of the ground states, e_k'* e_k S_k'k (eliminate.rs), the explicit states' van Vleck
//!   shift by the folded drives, and the kept drives, e_k U_k[e, b] and their adjoints;
//! - the decay −(i/2) Σ_φ Σ_a J_{a,φ}†J_{a,φ}: per class and pair of its members their jumps' own product (photon.rs).
//! A term's entry a ← b lies at the residual beat Ω + θ_a − θ_b in the cluster's frames (cluster.rs), its tone, and each product
//! of two beams' plane waves in it at the residual wave vector q − (K_a − K_b). A Hermitian term adds c M to the generator and
//! ∇(c M) to its gradient; a decay term adds no force (the absorption's mean force is the jumps'). Each Hermitian term comes
//! with its adjoint, so the generator's Hermitian part and its gradient are Hermitian at every time.
//!
//! # Near forms
//! The near light enters through its elimination's responses on each carrier's field basis, T_k and T2_k (eliminate.rs), by a few
//! complex numbers that are linear in them (`NearLayout`): the shifts' S_k'k = −½(U^f_k'†T_k + T_k'†U^f_k) and their Doppler
//! partners D_k'k = ½ U^f_k'†T2_k on the cluster's ground places; the jumps C_a T_k and C_a T2_k; the explicit states' van Vleck
//! shifts; and the responses themselves (the readout's). Each exists where it can differ from zero: the responses of a ground
//! state reach the excited states its coupling reaches through the excited states' own couplings (`Model::reached`). At a point
//! they are the exact elimination's there with their exact derivatives (`Model::near_exact`), or its table's interpolant
//! (tabulate.rs): one definition, tabulated or not.
//!
//! # First-order Doppler
//! A beam's absorption at w_l = (ħ/m)(k_l·κ + k_l²/2) takes T_l − w_l T2_l: with a carrier's field e_k, its Doppler-weighted
//! field Σ_l w_l a_l γ_lk multiplies the partner form. A term's value at a beam pair is therefore v₀ + w_r v_r + w_l v_l +
//! w_l w_r v_lr over the factors that are a near carrier's (`PLAIN`, `RIGHT`, `LEFT`, `BOTH`), the jumps' product exactly: the
//! generator is a Lindblad one at every momentum.
//!
//! # At a point, and along a path through it
//! `Middle`: the plan's beams' amplitudes with their gradients, the near forms there, and from them every term's values --
//! neither the momentum nor the time along a path enters. The Hermitian values' gradients are carried as slopes along a few
//! directions -- a table's along its axes' coordinates' gradients, the exact elimination's along the lab axes --, so that a table
//! of one axis sums one slope where the gradient has three components.
//!
//! A path through the point (`Path`) adds the momentum's part: each beam's Doppler shift w_l at the momentum the generator is
//! taken at, the rate k_l·v at which its plane wave turns at the path's velocity, and the rates K_s·v of the cluster's Doppler
//! frames. At a time s from the middle along the path (a node) every field component is its beams' sum with their plane waves
//! turned by e^{ik_l·v s}; the envelopes and the near forms are the middle's (model.rs, # A generator along a step's path). The
//! generator there is the sum over the blocks of the product of two components times the block's values -- per flag the
//! Hermitian ones and the decay's together --, each entry at its tone's phase e^{iΩ(t_m + s)} and in its states' Doppler frames,
//! e^{−i(K_x − K_y)·v s}, with the frames' diag(K_s·v); the form of its Hermitian part's gradient with a Hermitian ρ, Re tr(Ĝ ρ),
//! is the same sum of the blocks' own forms Σ V[x, y] ρ[y, x], times the gradient of the two components' product, and of their
//! slopes' forms times the product. No matrix per beam pair, per beat or per wave vector is made: what grows with the beams is
//! the components' sums alone. At the fixed point the same forms by tone are the gradient's tones (the step's shear).
//!
//! # Work
//! The complex multiply-adds are counted where they are made (`Middle::work`, `Plan::node`): at a point the near forms (the
//! table's interpolation or the elimination) and the jumps' products (a product's adjoint, the same jumps exchanged, is its
//! conjugate and is not made again); at a node the components from their beams and, per block and flag, its entries. A node's
//! counts are the plan's: no generator is made to know them.

use super::eliminate::Responses;
use super::far::EXCITED;
use super::photon::{Origin, Source};
use super::tabulate::NearTable;
use super::{jet, Beat, Component, Lights, Lit, Model, Unit, Wave};
use crate::field::Amplitude;
use crate::linalg::{dot3, Mat, C64};
use crate::request::{GROUND, STATES};
use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

/// Which factors of a term take their beams' Doppler shifts (module documentation, # First-order Doppler).
pub(crate) const PLAIN: usize = 0;
pub(crate) const RIGHT: usize = 1;
pub(crate) const LEFT: usize = 2;
pub(crate) const BOTH: usize = 3;

/// A near form or a constant: what a term's value is read from (module documentation, # Near forms).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Operand {
    /// S of a pair entry.
    Pair(usize),
    /// D of a pair entry.
    Doppler(usize),
    /// An explicit pair's van Vleck shift.
    Shift(usize),
    /// A jump entry's C_a T, and its C_a T2.
    Jump(usize),
    Jump2(usize),
    /// A constant of the plan.
    Constant(usize),
}

/// One cluster's near forms under a light set (module documentation, # Near forms), in the order of a point's values and of a
/// table's node: S, D, the shifts, the jumps, their Doppler partners, the responses, theirs.
#[derive(Clone, Debug, Default)]
pub(crate) struct NearLayout {
    /// (left carrier, its basis vector k', right carrier, its k, places x, y): S and D of the pair's entry x ← y.
    pub(crate) pairs: Vec<[usize; 6]>,
    /// (carrier, k', k, places x, y of two explicit states).
    pub(crate) shifts: Vec<[usize; 5]>,
    /// (carrier, k, axis a, landing ground state b, place y): (C_a T_k)[b, y].
    pub(crate) jumps: Vec<[usize; 5]>,
    /// (carrier, k, excited state e, place y): T_k[e, y].
    pub(crate) responses: Vec<[usize; 4]>,
}

impl NearLayout {
    /// The complex numbers of a point.
    pub(crate) fn len(&self) -> usize {
        2 * self.pairs.len() + self.shifts.len() + 2 * self.jumps.len() + 2 * self.responses.len()
    }
    /// Where an operand lies among a point's values and, after them, the plan's constants.
    fn at(&self, o: Operand) -> usize {
        let (p, v, j) = (self.pairs.len(), self.shifts.len(), self.jumps.len());
        match o {
            Operand::Pair(i) => i,
            Operand::Doppler(i) => p + i,
            Operand::Shift(i) => 2 * p + i,
            Operand::Jump(i) => 2 * p + v + i,
            Operand::Jump2(i) => 2 * p + v + j + i,
            Operand::Constant(i) => self.len() + i,
        }
    }
    /// The Hermitian forms (S, D, the shifts), the jumps with their partners, the responses with theirs.
    pub(crate) fn hermitian(&self) -> Range<usize> {
        0..2 * self.pairs.len() + self.shifts.len()
    }
    pub(crate) fn jumping(&self) -> Range<usize> {
        self.hermitian().end..self.hermitian().end + 2 * self.jumps.len()
    }
    pub(crate) fn responding(&self) -> Range<usize> {
        self.jumping().end..self.len()
    }
}

/// The near forms at a point over the entries before `len` (zero outside the sections evaluated), to an order, and the complex
/// multiply-adds that made them: jets in the position (jet.rs: `data[part * len + entry]`), or, from a table, the values and to
/// first order their slopes along its axes, whose coordinates' gradients are `directions` (`data[(1 + f) * len + entry]` =
/// ∂v/∂s_f).
#[derive(Clone, Debug)]
pub(crate) struct NearValues {
    pub(crate) len: usize,
    pub(crate) order: usize,
    pub(crate) data: Vec<C64>,
    pub(crate) directions: Option<Vec<[f64; 3]>>,
    pub(crate) work: f64,
}

impl NearValues {
    /// An entry's value and its gradient in the position (of forms to first order).
    pub(crate) fn first_order(&self, entry: usize) -> [C64; 4] {
        match &self.directions {
            None => std::array::from_fn(|p| self.data[p * self.len + entry]),
            Some(directions) => {
                let mut out = [self.data[entry], C64::ZERO, C64::ZERO, C64::ZERO];
                for (f, d) in directions.iter().enumerate() {
                    let slope = self.data[(1 + f) * self.len + entry];
                    for i in 0..3 {
                        out[1 + i] += slope * d[i];
                    }
                }
                out
            }
        }
    }
    /// An entry's jet in the position to the forms' order, into `out` (its parts).
    pub(crate) fn jet(&self, entry: usize, out: &mut [C64]) {
        match (&self.directions, self.order) {
            (None, _) => out.iter_mut().enumerate().for_each(|(p, z)| *z = self.data[p * self.len + entry]),
            (Some(_), 0) => out[0] = self.data[entry],
            (Some(_), _) => out.copy_from_slice(&self.first_order(entry)[..out.len()]),
        }
    }
    /// The directions the forms' first-order parts are slopes along: a table's coordinates' gradients, or the lab axes.
    fn slopes(&self) -> Vec<[f64; 3]> {
        match (&self.directions, self.order) {
            (_, 0) => vec![],
            (Some(directions), _) => directions.clone(),
            (None, _) => (0..3).map(|i| std::array::from_fn(|j| if i == j { 1.0 } else { 0.0 })).collect(),
        }
    }
}

/// A component of the light's fields at a point (module documentation, # Terms): u(r) = Σ_l a_l(r) c_l over beams of one
/// frequency, conjugated as a whole where `conjugate` (its weights are stored conjugated, its beams' wave vectors enter negated).
#[derive(Clone, Debug)]
pub(crate) struct Scalar {
    frequency: usize,
    /// None: the frequency's every beam; Some(j): its beam j alone.
    group: Option<usize>,
    /// The basis vector k of the frequency's field (c_l = γ_lk), or None: the beam's amplitude itself (c_l = 1).
    component: Option<usize>,
    pub(crate) conjugate: bool,
    /// (the beam's place among the plan's beams, c_l).
    pub(crate) beams: Vec<(usize, C64)>,
}

/// The terms of one product of field components at one base beat (module documentation, # Terms): per flag (`PLAIN`, `RIGHT`,
/// `LEFT`; and `BOTH` for the decay) where its Hermitian values and its decay's start among a point's values (one per entry),
/// and where the two together start among a point's merged values.
#[derive(Clone, Debug)]
struct Block {
    left: Option<usize>,
    right: Option<usize>,
    hermitian: [Option<usize>; 3],
    decay: [Option<usize>; 4],
    merged: [Option<usize>; 4],
    /// Per entry x ← y: its place x n + y on the cluster's states, and its tone -- the residual beat of the block's base beat in
    /// the two states' frames, an index into the plan's tones.
    places: Vec<usize>,
    tones: Vec<usize>,
    /// The entries' tone where they share one.
    tone: Option<usize>,
    /// Per Hermitian flag its weight in a form with a Hermitian ρ (`Middle::contract`). A flag's values and their adjoints
    /// -- the two components conjugated, the base beat negated, the entries transposed and conjugated, the Doppler-weighted
    /// side exchanged -- make conjugate forms: of a block and its adjoint the first is taken twice and the other left out,
    /// and of a block that is its own adjoint the flag `RIGHT` twice for `LEFT`.
    weights: [f64; 3],
}

/// One member of a photon class at a point (photon.rs): its source, its field component (None: 1), whether its jumps take the
/// Doppler partner, its relative beat's value in a merged class (rad/µs), and its entries (axis a, landing place, source place,
/// where its jump and its partner's lie among a point's values and constants).
#[derive(Clone, Debug)]
pub(crate) struct MemberPlan {
    pub(crate) source: usize,
    pub(crate) scalar: Option<usize>,
    pub(crate) doppler: bool,
    pub(crate) phase: f64,
    pub(crate) entries: Vec<(usize, usize, usize, usize, usize)>,
}

/// One photon class of the cluster under the plan's lights (photon.rs `Class`): its landing cluster, its wave number, its lit
/// members.
#[derive(Clone, Debug)]
pub(crate) struct ClassPlan {
    pub(crate) landing: usize,
    pub(crate) wave_number: f64,
    pub(crate) members: Vec<MemberPlan>,
}

/// A Hermitian near form's place in the generator: the block, flag and entry (x, y) it is the value of, its slot among a
/// point's values, the form's place, and whether it enters conjugated.
#[derive(Clone, Debug)]
pub(crate) struct Link {
    pub(crate) block: usize,
    pub(crate) flag: usize,
    pub(crate) place: (usize, usize),
    slot: usize,
    pub(crate) operand: usize,
    pub(crate) conjugate: bool,
}

/// A cluster's assembly under a light set, worked out once (module documentation): the near forms, the field components and
/// their beams, the blocks of terms, the tones, the photon classes.
#[derive(Clone, Debug)]
pub(crate) struct Plan {
    pub(crate) cluster: usize,
    pub(crate) lights: Lights,
    states: usize,
    pub(crate) near: NearLayout,
    /// The constants the jumps' products and the photons take, after a point's near forms.
    pub(crate) constants: Vec<C64>,
    /// The request beams the plan's field components sum.
    pub(crate) beams: Vec<usize>,
    pub(crate) scalars: Vec<Scalar>,
    blocks: Vec<Block>,
    /// Per block whether its base beat is exactly zero (the potential's static shifts).
    pub(crate) statics: Vec<bool>,
    /// A point's values without the near light: the Hermitian slots first.
    template: Vec<C64>,
    hermitian: usize,
    pub(crate) links: Vec<Link>,
    /// The jumps' products conj(first) second, by sums (each a range of them), and the decay's slots: (slot, its sum, whether
    /// it takes the sum's conjugate -- the adjoint's, the same jumps exchanged --, the flag's factor −(i/2)(−1)^{Doppler partners}).
    products: Vec<(usize, usize)>,
    raws: Vec<Range<usize>>,
    sums: Vec<(usize, usize, bool, C64)>,
    /// A point's merged values' count (`Block::merged`).
    merged: usize,
    /// The distinct residual beats of the terms' entries in the cluster's frames (rad/µs; the first is zero): at a fixed point
    /// the generator is Σ_m (its entries of tone m) e^{iΩ_m t}.
    tones: Vec<f64>,
    /// Whether the generator is the same at every time along every path: no entry beats in the frames, in time or along a path
    /// (every residual beat and wave vector exactly zero).
    still: bool,
    /// The complex multiply-adds at a node (module documentation, # Work): of the generator; of the force's form without its
    /// values' slopes; and of those per direction of them.
    node: [f64; 3],
    pub(crate) classes: Vec<ClassPlan>,
    /// The complex multiply-adds of a point's values from its near forms: the jumps' products and their sums, the merged values.
    values: f64,
}

/// A block under construction: its factors, base beat and entries, per flag each entry's constant and near form (Hermitian) or
/// constant and products (decay), and which flags hold anything.
struct Draft {
    left: Option<usize>,
    right: Option<usize>,
    beat: Beat,
    entries: Vec<(usize, usize)>,
    hermitian: [Vec<(C64, Option<(Operand, bool)>)>; 3],
    decay: [Vec<(C64, Vec<(Operand, Operand)>)>; 4],
    used: ([bool; 3], [bool; 4]),
}

/// A jump's value at a point: a constant, or a near form with its Doppler partner.
#[derive(Clone, Copy)]
enum Jump {
    Constant(C64),
    Near(usize),
}

/// The decay's factor of a flag: −(i/2) for each product, the Doppler partner's amplitude −w C_a T2 turning its sign.
fn decay_factor(flag: usize) -> C64 {
    C64::new(0.0, if flag == RIGHT || flag == LEFT { 0.5 } else { -0.5 })
}

/// A flag with its two sides exchanged, where a block's factors are (`Builder::block`).
fn flip(flag: usize, exchanged: bool) -> usize {
    if exchanged { [PLAIN, LEFT, RIGHT, BOTH][flag] } else { flag }
}

struct Builder<'a> {
    model: &'a Model,
    cluster: usize,
    lights: &'a Lights,
    near: NearLayout,
    pair_index: HashMap<[usize; 6], usize>,
    jump_index: HashMap<[usize; 5], usize>,
    constants: Vec<C64>,
    beams: Vec<usize>,
    scalars: Vec<Scalar>,
    drafts: Vec<Draft>,
}

impl<'a> Builder<'a> {
    fn states(&self) -> &'a [usize] {
        &self.model.clusters[self.cluster].states
    }
    fn ground(&self) -> usize {
        self.model.clusters[self.cluster].ground
    }
    /// The beam groups of a frequency whose interference its light keeps: every beam of a coherent light together, each beam of
    /// an averaged one alone, none of a dark one.
    fn groups(&self, frequency: usize) -> Vec<Option<usize>> {
        match self.lights.0[frequency] {
            Lit::Off => vec![],
            Lit::Coherent => vec![None],
            Lit::Averaged => (0..self.model.light.frequencies[frequency].beams.len()).map(Some).collect(),
        }
    }
    /// A field component (`Scalar`), found or made; None where it vanishes (no beam of the group holds the basis vector).
    fn scalar(&mut self, frequency: usize, group: Option<usize>, component: Option<usize>, conjugate: bool) -> Option<usize> {
        if let Some(at) = self.scalars.iter().position(|s| (s.frequency, s.group, s.component, s.conjugate) == (frequency, group, component, conjugate)) {
            return Some(at);
        }
        let f = &self.model.light.frequencies[frequency];
        let members: Vec<usize> = group.map_or((0..f.beams.len()).collect(), |j| vec![j]);
        let weights: Vec<(usize, C64)> = members.iter().map(|&j| (f.beams[j], component.map_or(C64::ONE, |k| f.coefficients[j][k]))).filter(|w| w.1 != C64::ZERO).collect();
        if weights.is_empty() {
            return None;
        }
        let mut beams = vec![];
        for (l, c) in weights {
            let at = self.beams.iter().position(|&x| x == l).unwrap_or_else(|| {
                self.beams.push(l);
                self.beams.len() - 1
            });
            beams.push((at, if conjugate { c.conj() } else { c }));
        }
        self.scalars.push(Scalar { frequency, group, component, conjugate, beams });
        Some(self.scalars.len() - 1)
    }
    /// The block of a product of two field components at a base beat, found or made, and whether its factors are exchanged in it:
    /// a product is one block whatever the order of its factors (the lesser component first, none before any), so that a
    /// counter-rotating component's e_k e_k'* joins the co-rotating e_k'* e_k -- the same beam pairs, expanded once.
    fn block(&mut self, left: Option<usize>, right: Option<usize>, beat: &Beat) -> (usize, bool) {
        let exchanged = left > right;
        let (left, right) = if exchanged { (right, left) } else { (left, right) };
        let at = self.drafts.iter().position(|d| d.left == left && d.right == right && d.beat == *beat).unwrap_or_else(|| {
            self.drafts.push(Draft { left, right, beat: beat.clone(), entries: vec![], hermitian: Default::default(), decay: Default::default(), used: Default::default() });
            self.drafts.len() - 1
        });
        (at, exchanged)
    }
    /// A block's entry x ← y, found or made.
    fn entry(&mut self, block: usize, x: usize, y: usize) -> usize {
        let d = &mut self.drafts[block];
        if let Some(at) = d.entries.iter().position(|e| *e == (x, y)) {
            return at;
        }
        d.entries.push((x, y));
        d.hermitian.iter_mut().for_each(|v| v.push((C64::ZERO, None)));
        d.decay.iter_mut().for_each(|v| v.push((C64::ZERO, vec![])));
        d.entries.len() - 1
    }
    /// Adds a Hermitian value to a block's entry: a constant and, where given, a near form (conjugated where the flag says).
    fn hermitian(&mut self, (block, exchanged): (usize, bool), flag: usize, x: usize, y: usize, constant: C64, form: Option<(Operand, bool)>) {
        let (at, flag) = (self.entry(block, x, y), flip(flag, exchanged));
        let d = &mut self.drafts[block];
        d.used.0[flag] = true;
        let slot = &mut d.hermitian[flag][at];
        slot.0 += constant;
        if form.is_some() {
            assert!(slot.1.is_none(), "an entry of a block takes one near form");
            slot.1 = form;
        }
    }
    /// Adds a product of two jumps, conj(first) second, to a block's decay.
    fn decay(&mut self, (block, exchanged): (usize, bool), flag: usize, x: usize, y: usize, first: Jump, second: Jump, partners: (bool, bool)) {
        let (at, flag) = (self.entry(block, x, y), flip(flag, exchanged));
        let operand = |b: &mut Builder, j: Jump, partner: bool| match j {
            Jump::Constant(v) => {
                let found = b.constants.iter().position(|c| *c == v).unwrap_or_else(|| {
                    b.constants.push(v);
                    b.constants.len() - 1
                });
                Operand::Constant(found)
            }
            Jump::Near(i) => if partner { Operand::Jump2(i) } else { Operand::Jump(i) },
        };
        if let (Jump::Constant(a), Jump::Constant(b)) = (first, second) {
            let d = &mut self.drafts[block];
            d.used.1[flag] = true;
            d.decay[flag][at].0 += decay_factor(flag) * a.conj() * b;
            return;
        }
        let (a, b) = (operand(self, first, partners.0), operand(self, second, partners.1));
        let d = &mut self.drafts[block];
        d.used.1[flag] = true;
        d.decay[flag][at].1.push((a, b));
    }

    /// Whether U^f_k'†T_k of a pair can differ from zero on x ← y: the left coupling's excited states from x among those the
    /// right response reaches from y.
    fn touches(&self, left: (usize, usize), right: (usize, usize), x: usize, y: usize) -> bool {
        let reach = self.model.reached(self.cluster, right.0, right.1, self.states()[y]);
        let u = &self.model.carriers[left.0].basis[left.1];
        (0..EXCITED).any(|e| reach[e] && u[(GROUND + e, self.states()[x])] != C64::ZERO)
    }
    /// The pair entry of S and D on x ← y, found or made with its adjoint's (the same forms' y ← x of the pair exchanged); None
    /// where neither can differ from zero.
    fn pair(&mut self, left: (usize, usize), right: (usize, usize), x: usize, y: usize) -> Option<usize> {
        let key = [left.0, left.1, right.0, right.1, x, y];
        if let Some(&at) = self.pair_index.get(&key) {
            return Some(at);
        }
        if !self.touches(left, right, x, y) && !self.touches(right, left, y, x) {
            return None;
        }
        for k in [key, [right.0, right.1, left.0, left.1, y, x]] {
            if !self.pair_index.contains_key(&k) {
                self.pair_index.insert(k, self.near.pairs.len());
                self.near.pairs.push(k);
            }
        }
        Some(self.pair_index[&key])
    }
    /// The jump entry (C_a T_k)[b, y], found or made; None where it cannot differ from zero.
    fn jump(&mut self, carrier: usize, k: usize, a: usize, b: usize, y: usize) -> Option<usize> {
        let key = [carrier, k, a, b, y];
        if let Some(&at) = self.jump_index.get(&key) {
            return Some(at);
        }
        let reach = self.model.reached(self.cluster, carrier, k, self.states()[y]);
        if !(0..EXCITED).any(|e| reach[e] && self.model.emission[a][(b, GROUND + e)] != C64::ZERO) {
            return None;
        }
        self.jump_index.insert(key, self.near.jumps.len());
        self.near.jumps.push(key);
        Some(self.near.jumps.len() - 1)
    }
    /// A source's field component, conjugated or not (None: 1); the outer None where the component vanishes.
    fn source_scalar(&mut self, source: &Source, conjugated: bool) -> Option<Option<usize>> {
        match source.origin {
            Origin::Far { frequency, side, component } => self.scalar(frequency, None, Some(component), (side == 1) != conjugated).map(Some),
            Origin::Near { carrier, component } => self.scalar(self.model.carriers[carrier].frequency, None, Some(component), conjugated).map(Some),
            Origin::Explicit { .. } => Some(None),
        }
    }
    /// A source's jump b ← the state at `place` on axis a; None where it is zero.
    fn source_jump(&mut self, source: &Source, a: usize, b: usize, place: usize) -> Option<Jump> {
        let s = self.states()[place];
        match source.origin {
            Origin::Near { carrier, component } => self.jump(carrier, component, a, b, place).map(Jump::Near),
            _ => Some(self.model.constant_jump(source, a, b, s)).filter(|v| *v != C64::ZERO).map(Jump::Constant),
        }
    }

    /// The base, the frames and the far static forms (module documentation, # Terms).
    fn statics(&mut self) {
        let model = self.model;
        let (states, n) = (self.states(), self.states().len());
        let zero = model.axes.zero();
        let still = self.block(None, None, &zero);
        for x in 0..n {
            for y in 0..n {
                let v = model.base[(states[x], states[y])];
                if v != C64::ZERO {
                    self.hermitian(still, PLAIN, x, y, v, None);
                }
            }
        }
        for (x, theta) in model.clusters[self.cluster].frames.iter().enumerate() {
            self.hermitian(still, PLAIN, x, x, C64::real(-model.omega(theta)), None);
        }
        for (w, (f, tone)) in model.light.frequencies.iter().zip(&model.tones).enumerate() {
            let r = f.rank();
            for group in self.groups(w) {
                for kl in 0..r {
                    for kr in 0..r {
                        let form = &tone.forms[kl * r + kr];
                        let entries: Vec<(usize, usize, C64)> =
                            (0..n * n).map(|i| (i / n, i % n, form[(states[i / n], states[i % n])])).filter(|e| e.2 != C64::ZERO).collect();
                        if entries.is_empty() {
                            continue;
                        }
                        let (Some(left), Some(right)) = (self.scalar(w, group, Some(kl), true), self.scalar(w, group, Some(kr), false)) else { continue };
                        let block = self.block(Some(left), Some(right), &zero);
                        for (x, y, v) in entries {
                            self.hermitian(block, PLAIN, x, y, v, None);
                        }
                    }
                }
            }
        }
    }

    /// The kept far beats' cross terms and their adjoints (units.rs `Model::far_cross`): absorbing `first` and emitting `second` at
    /// the base beat ν₂ − ν₁, coefficient a_l₂* a_l₁ (co-rotating; a_l₁* a_l₂ counter-rotating).
    fn far_beats(&mut self) {
        let model = self.model;
        let (states, n) = (self.states(), self.states().len());
        for unit in &model.kept {
            let Unit::FarBeat { first, second } = unit else { continue };
            if !self.lights.cross(first.frequency, second.frequency) {
                continue;
            }
            let (n1, n2) = (model.light.frequencies[first.frequency].beams.len(), model.light.frequencies[second.frequency].beams.len());
            let beat = model.axes.component(&model.light, *second).plus(-1, &model.axes.component(&model.light, *first));
            let cross = model.far_cross(*first, *second);
            for j in 0..n2 {
                for i in 0..n1 {
                    let m = &cross[j * n1 + i];
                    let (conjugated, plain) = if first.counter { ((first.frequency, i), (second.frequency, j)) } else { ((second.frequency, j), (first.frequency, i)) };
                    let term = (self.scalar(conjugated.0, Some(conjugated.1), None, true), self.scalar(plain.0, Some(plain.1), None, false));
                    let adjoint = (self.scalar(plain.0, Some(plain.1), None, true), self.scalar(conjugated.0, Some(conjugated.1), None, false));
                    let (there, back) = (self.block(term.0, term.1, &beat), self.block(adjoint.0, adjoint.1, &beat.negated()));
                    for x in 0..n {
                        for y in 0..n {
                            let v = m[(states[x], states[y])];
                            if v != C64::ZERO {
                                self.hermitian(there, PLAIN, x, y, v, None);
                                self.hermitian(back, PLAIN, y, x, v.conj(), None);
                            }
                        }
                    }
                }
            }
        }
    }

    /// The near light's terms (module documentation, # Terms): per pair of lit carriers the shifts and couplings of the ground
    /// places at the beat δ_c' − δ_c -- an entry at a residual beat of zero in the frames, or its kept unit (eliminate.rs
    /// `Model::term_included`) --, the explicit states' van Vleck shifts, and the kept drives with their adjoints.
    fn near(&mut self) {
        let model = self.model;
        let (states, n, g) = (self.states(), self.states().len(), self.ground());
        let lit: Vec<usize> = (0..model.carriers.len()).filter(|&c| self.lights.on(model.carriers[c].frequency)).collect();
        let zero = model.axes.zero();
        for &cj in &lit {
            for &ci in &lit {
                let (cp, cc) = (&model.carriers[cj], &model.carriers[ci]);
                if ci != cj && !self.lights.cross(cp.frequency, cc.frequency) {
                    continue;
                }
                let beat = cp.beat.plus(-1, &cc.beat);
                let (from, to) = (Component { frequency: cc.frequency, counter: false }, Component { frequency: cp.frequency, counter: false });
                let groups: Vec<(Option<usize>, Option<usize>)> = if ci == cj { self.groups(cc.frequency).into_iter().map(|x| (x, x)).collect() } else { vec![(None, None)] };
                for kl in 0..cp.basis.len() {
                    for kr in 0..cc.basis.len() {
                        for x in 0..g {
                            for y in 0..g {
                                if !model.term_included(self.cluster, x, y, from, to, &beat) {
                                    continue;
                                }
                                let Some(at) = self.pair((cj, kl), (ci, kr), x, y) else { continue };
                                let exchanged = self.pair((ci, kr), (cj, kl), y, x).expect("a pair's entry comes with its adjoint's");
                                for &(gl, gr) in &groups {
                                    let (Some(left), Some(right)) = (self.scalar(cp.frequency, gl, Some(kl), true), self.scalar(cc.frequency, gr, Some(kr), false)) else { continue };
                                    let block = self.block(Some(left), Some(right), &beat);
                                    self.hermitian(block, PLAIN, x, y, C64::ZERO, Some((Operand::Pair(at), false)));
                                    self.hermitian(block, RIGHT, x, y, C64::ZERO, Some((Operand::Doppler(at), false)));
                                    self.hermitian(block, LEFT, x, y, C64::ZERO, Some((Operand::Doppler(exchanged), true)));
                                }
                            }
                        }
                    }
                }
            }
        }
        for &ci in &lit {
            let cc = &model.carriers[ci];
            let unfolded = &model.tones[cc.frequency].basis[0];
            for kl in 0..cc.basis.len() {
                for kr in 0..cc.basis.len() {
                    for x in g..n {
                        for y in g..n {
                            let (ul, ur) = (&cc.basis[kl], &cc.basis[kr]);
                            if !(0..GROUND).any(|b| ur[(states[x], b)] != C64::ZERO && ul[(states[y], b)] != C64::ZERO) {
                                continue;
                            }
                            self.near.shifts.push([ci, kl, kr, x, y]);
                            let at = self.near.shifts.len() - 1;
                            for group in self.groups(cc.frequency) {
                                let (Some(left), Some(right)) = (self.scalar(cc.frequency, group, Some(kl), true), self.scalar(cc.frequency, group, Some(kr), false)) else { continue };
                                let block = self.block(Some(left), Some(right), &zero);
                                self.hermitian(block, PLAIN, x, y, C64::ZERO, Some((Operand::Shift(at), false)));
                            }
                        }
                    }
                }
            }
            for (k, u) in unfolded.iter().enumerate() {
                let kept: Vec<(usize, usize, C64)> = (g..n)
                    .flat_map(|p| (0..g).map(move |q| (p, q)))
                    .map(|(p, q)| (p, q, u[(states[p], states[q])]))
                    .filter(|&(p, q, v)| v != C64::ZERO && model.drive_kept(ci, states[p], states[q]))
                    .collect();
                if kept.is_empty() {
                    continue;
                }
                let (Some(plain), Some(conjugated)) = (self.scalar(cc.frequency, None, Some(k), false), self.scalar(cc.frequency, None, Some(k), true)) else { continue };
                let (up, down) = (self.block(None, Some(plain), &cc.beat.negated()), self.block(Some(conjugated), None, &cc.beat));
                for (p, q, v) in kept {
                    self.hermitian(up, PLAIN, p, q, v, None);
                    self.hermitian(down, PLAIN, q, p, v.conj(), None);
                }
            }
        }
    }

    /// The decay (photon.rs, # The decay): per class and pair of lit members (j, i), on each axis and common landing state the
    /// product of their jumps conj(L_j) L_i, at the base beat Ω_i − Ω_j.
    fn decays(&mut self) {
        let model = self.model;
        for class in &model.classes[self.cluster] {
            let landing = &model.clusters[class.landing];
            for mj in &class.members {
                let sj = &model.sources[mj.source];
                if !model.lit(sj, self.lights) {
                    continue;
                }
                for mi in &class.members {
                    let si = &model.sources[mi.source];
                    if !model.lit(si, self.lights) {
                        continue;
                    }
                    let (Some(left), Some(right)) = (self.source_scalar(sj, true), self.source_scalar(si, false)) else { continue };
                    let block = self.block(left, right, &si.beat.plus(-1, &sj.beat));
                    let partners = |s: &Source| if matches!(s.origin, Origin::Near { .. }) { vec![false, true] } else { vec![false] };
                    for &(bj, xj) in &mj.entries {
                        for &(bi, xi) in &mi.entries {
                            if bj != bi {
                                continue;
                            }
                            for a in 0..3 {
                                let (Some(first), Some(second)) = (self.source_jump(sj, a, landing.states[bj], xj), self.source_jump(si, a, landing.states[bi], xi)) else { continue };
                                for pj in partners(sj) {
                                    for pi in partners(si) {
                                        self.decay(block, (pi as usize) * RIGHT + (pj as usize) * LEFT, xj, xi, first, second, (pj, pi));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// The photon classes' lit members with their jumps (`ClassPlan`), the entries' operands still to place.
    fn classes(&mut self) -> Vec<(usize, f64, Vec<(usize, Option<usize>, bool, f64, Vec<(usize, usize, usize, Operand, Operand)>)>)> {
        let model = self.model;
        let mut out = vec![];
        for class in &model.classes[self.cluster] {
            let landing = &model.clusters[class.landing];
            let mut members = vec![];
            for m in &class.members {
                let source = &model.sources[m.source];
                if !model.lit(source, self.lights) {
                    continue;
                }
                let Some(scalar) = self.source_scalar(source, false) else { continue };
                let mut entries = vec![];
                for &(bp, sp) in &m.entries {
                    for a in 0..3 {
                        match self.source_jump(source, a, landing.states[bp], sp) {
                            None => {}
                            Some(Jump::Near(i)) => entries.push((a, bp, sp, Operand::Jump(i), Operand::Jump2(i))),
                            Some(Jump::Constant(v)) => {
                                self.constants.push(v);
                                let c = Operand::Constant(self.constants.len() - 1);
                                entries.push((a, bp, sp, c, c));
                            }
                        }
                    }
                }
                members.push((m.source, scalar, matches!(source.origin, Origin::Near { .. }), model.omega(&m.phase), entries));
            }
            out.push((class.landing, model.class_wave_number(self.cluster, class), members));
        }
        out
    }

    /// The plan: the responses' entries, the near forms' places, the blocks' slots, beam pairs and parts.
    fn finish(mut self) -> Plan {
        let classes = self.classes();
        let model = self.model;
        let cluster = &model.clusters[self.cluster];
        let (states, n, g) = (&cluster.states, cluster.states.len(), cluster.ground);
        for (ci, cc) in model.carriers.iter().enumerate() {
            if !self.lights.on(cc.frequency) {
                continue;
            }
            for k in 0..cc.basis.len() {
                for y in 0..g {
                    let reach = model.reached(self.cluster, ci, k, states[y]);
                    self.near.responses.extend((0..EXCITED).filter(|&e| reach[e]).map(|e| [ci, k, e, y]));
                }
            }
        }
        let near = self.near;
        // The slots: every block's Hermitian values first, then the decay's.
        let mut template = vec![];
        let mut links = vec![];
        let mut hermitian: Vec<[Option<usize>; 3]> = vec![[None; 3]; self.drafts.len()];
        for (b, d) in self.drafts.iter().enumerate() {
            for flag in 0..3 {
                if !d.used.0[flag] {
                    continue;
                }
                hermitian[b][flag] = Some(template.len());
                for (e, (constant, form)) in d.hermitian[flag].iter().enumerate() {
                    if let Some((operand, conjugate)) = form {
                        links.push(Link { block: b, flag, place: d.entries[e], slot: template.len(), operand: near.at(*operand), conjugate: *conjugate });
                    }
                    template.push(*constant);
                }
            }
        }
        let slots = template.len();
        let (mut sums, mut products, mut raws) = (vec![], vec![], vec![]);
        let mut made: HashMap<Vec<(usize, usize)>, usize> = HashMap::new();
        let mut decay: Vec<[Option<usize>; 4]> = vec![[None; 4]; self.drafts.len()];
        for (b, d) in self.drafts.iter().enumerate() {
            for flag in 0..4 {
                if !d.used.1[flag] {
                    continue;
                }
                decay[b][flag] = Some(template.len());
                for (constant, terms) in &d.decay[flag] {
                    if !terms.is_empty() {
                        let mut list: Vec<(usize, usize)> = terms.iter().map(|(x, y)| (near.at(*x), near.at(*y))).collect();
                        list.sort();
                        let mut exchanged: Vec<(usize, usize)> = list.iter().map(|&(x, y)| (y, x)).collect();
                        exchanged.sort();
                        let (raw, conjugated) = match (made.get(&list), made.get(&exchanged)) {
                            (Some(&at), _) => (at, false),
                            (None, Some(&at)) => (at, true),
                            (None, None) => {
                                raws.push(products.len()..products.len() + list.len());
                                products.extend(list.iter().copied());
                                made.insert(list, raws.len() - 1);
                                (raws.len() - 1, false)
                            }
                        };
                        sums.push((template.len(), raw, conjugated, decay_factor(flag)));
                    }
                    template.push(*constant);
                }
            }
        }
        // The blocks' entries, their places and tones; whether anything beats -- a residual beat in time, or along a path the
        // residual wave vector of a pair of the two components' beams in the entry's Doppler frames --; and the merged values.
        let mut blocks: Vec<Block> = vec![];
        let mut tone_index: HashMap<Beat, usize> = HashMap::from([(model.axes.zero(), 0)]);
        let mut tones = vec![0.0];
        let (mut still, mut merged_count) = (true, 0);
        for (b, d) in self.drafts.iter().enumerate() {
            let waves = |s: Option<usize>| -> Vec<Wave> {
                s.map_or(vec![model.no_wave()], |i| self.scalars[i].beams.iter().map(|&(at, _)| model.no_wave().plus(if self.scalars[i].conjugate { -1 } else { 1 }, &model.beam_wave(self.beams[at]))).collect())
            };
            let (left, right) = (waves(d.left), waves(d.right));
            let mut block_tones = vec![];
            for &(x, y) in &d.entries {
                let beat = d.beat.plus(1, &cluster.frames[x]).plus(-1, &cluster.frames[y]);
                let frames = cluster.waves_exact[x].plus(-1, &cluster.waves_exact[y]);
                still &= beat.is_zero() && left.iter().all(|l| right.iter().all(|r| l.plus(1, r) == frames));
                block_tones.push(*tone_index.entry(beat.clone()).or_insert_with(|| {
                    tones.push(model.omega(&beat));
                    tones.len() - 1
                }));
            }
            let merged = std::array::from_fn(|flag| {
                (hermitian[b].get(flag).copied().flatten().is_some() || decay[b][flag].is_some()).then(|| {
                    merged_count += d.entries.len();
                    merged_count - d.entries.len()
                })
            });
            let tone = block_tones.first().copied().filter(|t| block_tones.iter().all(|x| x == t));
            blocks.push(Block { left: d.left, right: d.right, hermitian: hermitian[b], decay: decay[b], merged, places: d.entries.iter().map(|&(x, y)| x * n + y).collect(), tones: block_tones, tone, weights: [1.0; 3] });
        }
        // The Hermitian blocks' adjoints (`Block::weights`): the block of the two components' conjugates at the negated beat.
        let conjugate = |s: Option<usize>| match s {
            None => Some(None),
            Some(i) => {
                let x = &self.scalars[i];
                self.scalars.iter().position(|y| (y.frequency, y.group, y.component, y.conjugate) == (x.frequency, x.group, x.component, !x.conjugate)).map(Some)
            }
        };
        for (b, d) in self.drafts.iter().enumerate() {
            let adjoint = conjugate(d.left).zip(conjugate(d.right)).and_then(|(l, r)| {
                let (l, r) = if l > r { (r, l) } else { (l, r) };
                self.drafts.iter().position(|t| t.left == l && t.right == r && t.beat == d.beat.negated())
            });
            blocks[b].weights = match adjoint.filter(|&t| hermitian[t].iter().any(|h| h.is_some())) {
                Some(t) if t > b => [2.0; 3],
                Some(t) if t < b => [0.0; 3],
                Some(_) => [1.0, 2.0, 0.0],
                None => [1.0; 3],
            };
        }
        // A node: the field components from their beams (plain and Doppler-weighted; with their gradients, fourfold), each
        // block's entries per flag -- in a form those of the Hermitian flags taken (`Block::weights`) --, the frames' phases
        // on the n² entries.
        let fields: usize = self.scalars.iter().map(|s| 2 * s.beams.len()).sum();
        let forms: usize = blocks.iter().map(|b| (0..3).filter(|&flag| b.hermitian[flag].is_some() && b.weights[flag] > 0.0).count() * b.places.len()).sum();
        let node = [(fields + merged_count + 2 * n * n) as f64, (4 * fields + forms + n * n) as f64, forms as f64];
        let statics = self.drafts.iter().map(|d| d.beat.is_zero()).collect();
        let classes = classes
            .into_iter()
            .map(|(landing, wave_number, members)| ClassPlan {
                landing,
                wave_number,
                members: members
                    .into_iter()
                    .map(|(source, scalar, doppler, phase, entries)| MemberPlan { source, scalar, doppler, phase, entries: entries.into_iter().map(|(a, bp, sp, j, j2)| (a, bp, sp, near.at(j), near.at(j2))).collect() })
                    .collect(),
            })
            .collect();
        let values = (products.len() + sums.len() + merged_count) as f64;
        Plan {
            raws,
            cluster: self.cluster,
            lights: self.lights.clone(),
            states: n,
            near,
            constants: self.constants,
            beams: self.beams,
            scalars: self.scalars,
            blocks,
            statics,
            template,
            hermitian: slots,
            links,
            sums,
            products,
            merged: merged_count,
            still,
            node,
            tones,
            classes,
            values,
        }
    }
}

impl Plan {
    fn new(model: &Model, cluster: usize, lights: &Lights) -> Plan {
        let mut b = Builder { model, cluster, lights, near: NearLayout::default(), pair_index: HashMap::new(), jump_index: HashMap::new(), constants: vec![], beams: vec![], scalars: vec![], drafts: vec![] };
        b.statics();
        b.far_beats();
        b.near();
        b.decays();
        b.finish()
    }
    /// A block's two field components.
    pub(crate) fn factors(&self, block: usize) -> (Option<usize>, Option<usize>) {
        (self.blocks[block].left, self.blocks[block].right)
    }
    /// The plan's beams' amplitudes with their gradients, from every beam's.
    pub(crate) fn amplitudes(&self, all: &[Amplitude]) -> Vec<[C64; 4]> {
        self.beams.iter().map(|&l| [all[l].value, all[l].gradient[0], all[l].gradient[1], all[l].gradient[2]]).collect()
    }
}

/// A plan's field components at a point (`Model::fields`): per component its jet to an order, plain and weighted by its beams'
/// Doppler shifts (each `parts` numbers a component).
pub(crate) struct Fields {
    parts: usize,
    plain: Vec<C64>,
    doppler: Vec<C64>,
}

impl Fields {
    /// A component's jet.
    pub(crate) fn plain(&self, scalar: usize) -> &[C64] {
        &self.plain[scalar * self.parts..(scalar + 1) * self.parts]
    }
    /// The same weighted by its beams' Doppler shifts.
    pub(crate) fn doppler(&self, scalar: usize) -> &[C64] {
        &self.doppler[scalar * self.parts..(scalar + 1) * self.parts]
    }
}

/// A cluster's generator at a position and a time, for every momentum and along every path through the point (module
/// documentation, # At a point, and along a path through it).
pub struct Middle<'a> {
    model: &'a Model,
    plan: Arc<Plan>,
    t: f64,
    /// Per beam of the plan: a_l and ∇a_l.
    amplitudes: Vec<[C64; 4]>,
    /// Every block's values per flag and entry, the Hermitian ones first; and per block and flag the Hermitian value and the
    /// decay's together (`Block::merged`), the generator's.
    values: Vec<C64>,
    merged: Vec<C64>,
    /// The directions the Hermitian values' gradients are slopes along (1/µm per unit of each), and per Hermitian value its slopes
    /// (`slopes[slot * directions + f]`).
    directions: Vec<[f64; 3]>,
    slopes: Vec<C64>,
    /// The complex multiply-adds that made it.
    pub(crate) work: f64,
    /// Its nodes' scratch, one for all its paths: a node's call fills what it reads.
    scratch: RefCell<Scratch>,
}

impl Model {
    /// A cluster's plan under a light set, made once.
    pub(crate) fn plan(&self, cluster: usize, lights: &Lights) -> Arc<Plan> {
        let mut plans = self.plans.lock().expect("the plans' lock");
        if let Some(p) = plans.iter().find(|p| p.cluster == cluster && p.lights == *lights) {
            return p.clone();
        }
        let plan = Arc::new(Plan::new(self, cluster, lights));
        plans.push(plan.clone());
        plan
    }

    /// The excited states the response of a carrier's basis coupling k from ground state b can reach on a cluster: those in a
    /// block (`Model::mixing`) with a state U^f_k couples b to.
    pub(crate) fn reached(&self, cluster: usize, carrier: usize, k: usize, b: usize) -> [bool; EXCITED] {
        let (u, blocks) = (&self.carriers[carrier].basis[k], &self.mixing[cluster]);
        let mut out = [false; EXCITED];
        for from in 0..EXCITED {
            if u[(GROUND + from, b)] != C64::ZERO {
                for e in 0..EXCITED {
                    out[e] |= blocks[e] == blocks[from];
                }
            }
        }
        out
    }

    /// The blocks of the excited states that an elimination on a cluster can mix, per excited state its block: the classes of
    /// the couplings among them in K + Σ (eliminate.rs) under any light -- the base's, every far form's and the decay's entries,
    /// and the self-energy's v v† of each folded channel to a ground state outside the cluster. A response's support is a union of
    /// blocks: the factorization of K + Σ − z and its solves leave every other entry exactly zero.
    pub(crate) fn mix(&self, cluster: usize) -> Vec<usize> {
        let mut block: Vec<usize> = (0..EXCITED).collect();
        let join = |block: &mut Vec<usize>, a: usize, b: usize| {
            let (x, y) = (block[a], block[b]);
            if x != y {
                block.iter_mut().filter(|z| **z == y).for_each(|z| *z = x);
            }
        };
        for a in 0..EXCITED {
            for b in 0..EXCITED {
                let at = (GROUND + a, GROUND + b);
                if self.base[at] != C64::ZERO || self.decay[(a, b)] != C64::ZERO || self.tones.iter().any(|t| t.pairs.iter().any(|p| p[at] != C64::ZERO)) {
                    join(&mut block, a, b);
                }
            }
        }
        for cc in &self.carriers {
            for s in (0..GROUND).filter(|&s| self.membership[s] != cluster) {
                let reached: Vec<usize> = (0..EXCITED).filter(|&e| cc.folded.iter().any(|x| x[(GROUND + e, s)] != C64::ZERO)).collect();
                for &e in &reached {
                    join(&mut block, reached[0], e);
                }
            }
        }
        block
    }

    /// A plan's near forms from the elimination's responses and the far light's energies (jets, [part][state]), into
    /// `out[part * len + entry]` (module documentation, # Near forms); and their complex multiply-adds (per part each pair's two
    /// sums and its partner's, each jump's two, each shift's two per ground state).
    pub(crate) fn forms(&self, plan: &Plan, responses: &Responses, energies: &[Vec<f64>], out: &mut [C64]) -> f64 {
        let states = &self.clusters[plan.cluster].states;
        let (near, len) = (&plan.near, plan.near.len());
        for part in 0..energies.len() {
            let out = &mut out[part * len..(part + 1) * len];
            let (first, second) = (|c: usize, k: usize| &responses.first[c][k][part], |c: usize, k: usize| &responses.second[c][k][part]);
            for (i, &[cl, kl, cr, kr, x, y]) in near.pairs.iter().enumerate() {
                let (ul, ur) = (&self.carriers[cl].basis[kl], &self.carriers[cr].basis[kr]);
                let (mut s, mut d) = (C64::ZERO, C64::ZERO);
                for e in 0..EXCITED {
                    let left = ul[(GROUND + e, states[x])].conj();
                    s += left * first(cr, kr)[(e, y)] + first(cl, kl)[(e, x)].conj() * ur[(GROUND + e, states[y])];
                    d += left * second(cr, kr)[(e, y)];
                }
                out[near.at(Operand::Pair(i))] = s * -0.5;
                out[near.at(Operand::Doppler(i))] = d * 0.5;
            }
            for (i, &[c, k, a, b, y]) in near.jumps.iter().enumerate() {
                let emission = &self.emission[a];
                let (mut j, mut j2) = (C64::ZERO, C64::ZERO);
                for e in 0..EXCITED {
                    let x = emission[(b, GROUND + e)];
                    if x != C64::ZERO {
                        j += x * first(c, k)[(e, y)];
                        j2 += x * second(c, k)[(e, y)];
                    }
                }
                out[near.at(Operand::Jump(i))] = j;
                out[near.at(Operand::Jump2(i))] = j2;
            }
            let responding = near.responding().start;
            for (i, &[c, k, e, y]) in near.responses.iter().enumerate() {
                out[responding + i] = first(c, k)[(e, y)];
                out[responding + near.responses.len() + i] = second(c, k)[(e, y)];
            }
        }
        // The explicit states' van Vleck shift by the folded drives, ½ Σ_b U^f_k[e,b] U^f_k'[f,b]* (1/(E_e − E_b − δ) + 1/(E_f − E_b − δ)).
        for (i, &[c, kl, kr, x, y]) in near.shifts.iter().enumerate() {
            let cc = &self.carriers[c];
            let (e, f) = (states[x], states[y]);
            let mut shift = vec![C64::ZERO; energies.len()];
            for b in 0..GROUND {
                let v = cc.basis[kr][(e, b)] * cc.basis[kl][(f, b)].conj();
                if v == C64::ZERO {
                    continue;
                }
                for s in [e, f] {
                    let mut gap: Vec<f64> = energies.iter().map(|h| h[s] - h[b]).collect();
                    gap[0] -= cc.detuning;
                    for (z, r) in shift.iter_mut().zip(jet::reciprocal(&gap)) {
                        *z += v * (0.5 * r);
                    }
                }
            }
            for (part, z) in shift.into_iter().enumerate() {
                out[part * len + near.at(Operand::Shift(i))] = z;
            }
        }
        (energies.len() * ((3 * near.pairs.len() + 2 * near.jumps.len()) * EXCITED + 2 * GROUND * near.shifts.len())) as f64
    }

    /// A plan's near forms at a point from the exact elimination there, jets to `order`, each carrier's dressing coherent or not as
    /// `coherent` says (eliminate.rs).
    pub(crate) fn near_exact(&self, plan: &Plan, amplitudes: &[Amplitude], order: usize, coherent: &[bool]) -> NearValues {
        let len = plan.near.len();
        let mut data = vec![C64::ZERO; jet::parts(order) * len];
        if len == 0 {
            return NearValues { len, order, data, directions: None, work: 0.0 };
        }
        let h = self.far_hamiltonian(amplitudes, &plan.lights, order);
        let responses = self.eliminate(plan.cluster, &h, amplitudes, &plan.lights, order, coherent);
        let energies: Vec<Vec<f64>> = h.iter().map(|m| (0..STATES).map(|s| m[(s, s)].re).collect()).collect();
        let work = self.forms(plan, &responses, &energies, &mut data);
        NearValues { len, order, data, directions: None, work: responses.work + self.far_work(&plan.lights, order) + work }
    }

    /// A plan's near forms at a point, the sections asked each to its order (0 or 1): from its table where one is given
    /// (tabulate.rs), else the exact elimination's to the largest order.
    pub(crate) fn near_values(&self, plan: &Plan, amplitudes: &[Amplitude], sections: &[(Range<usize>, usize)], table: Option<&NearTable>) -> NearValues {
        match table {
            Some(table) => self.near_tabulated(table, amplitudes, sections),
            None => self.near_exact(plan, amplitudes, sections.iter().map(|s| s.1).max().unwrap_or(0), &self.coherent_dressing()),
        }
    }

    /// A plan's field components at a point, jets to `order`, plain and weighted by their beams' Doppler shifts at a momentum.
    pub(crate) fn fields(&self, plan: &Plan, amplitudes: &[Amplitude], kappa: [f64; 3], order: usize) -> Fields {
        let parts = jet::parts(order);
        let mut out = Fields { parts, plain: vec![C64::ZERO; parts * plan.scalars.len()], doppler: vec![C64::ZERO; parts * plan.scalars.len()] };
        for (s, (plain, doppler)) in plan.scalars.iter().zip(out.plain.chunks_mut(parts).zip(out.doppler.chunks_mut(parts))) {
            for &(at, c) in &s.beams {
                let (beam, shift) = (jet::whole(&amplitudes[plan.beams[at]]), self.doppler_shift(plan.beams[at], kappa));
                for p in 0..parts {
                    let z = (if s.conjugate { beam[p].conj() } else { beam[p] }) * c;
                    plain[p] += z;
                    doppler[p] += z * shift;
                }
            }
        }
        out
    }

    /// A beam's first-order Doppler shift w_l = (ħ/m)(k_l·κ + k_l²/2) at a momentum (eliminate.rs, # First-order Doppler).
    pub(crate) fn doppler_shift(&self, beam: usize, kappa: [f64; 3]) -> f64 {
        let k = self.light.beams[beam].wave_vector();
        self.hbar_over_mass * (dot3(k, kappa) + 0.5 * dot3(k, k))
    }

    /// A cluster's generator at a position and time under a plan (`Middle`), the near forms from a table where one is given.
    pub(crate) fn middle(&self, plan: &Arc<Plan>, r: [f64; 3], t: f64, table: Option<&NearTable>) -> Middle<'_> {
        if plan.near.len() == 0 {
            let amplitudes = plan.beams.iter().map(|&l| self.light.beams[l].amplitude(r)).map(|a| [a.value, a.gradient[0], a.gradient[1], a.gradient[2]]).collect();
            return self.middle_of(plan, t, amplitudes, None);
        }
        let all = self.light.amplitudes(r);
        let near = self.near_values(plan, &all, &[(plan.near.hermitian(), 1), (plan.near.jumping(), 0)], table);
        self.middle_of(plan, t, plan.amplitudes(&all), Some(&near))
    }

    /// The same from the plan's beams' amplitudes and the near forms there (the Hermitian ones to first order): every block's
    /// values, a Hermitian slot its constant and its near form with its slopes, a decay slot its constant and its jumps'
    /// products. Without near forms (None): the terms that hold none.
    pub(crate) fn middle_of(&self, plan: &Arc<Plan>, t: f64, amplitudes: Vec<[C64; 4]>, near: Option<&NearValues>) -> Middle<'_> {
        let len = plan.near.len();
        let mut values = plan.template.clone();
        let directions = near.map_or(vec![], |n| n.slopes());
        let q = directions.len();
        let mut slopes = vec![C64::ZERO; plan.hermitian * q];
        let mut work = 0.0;
        if let Some(near) = near {
            work = near.work + plan.values;
            for l in &plan.links {
                let turn = |z: C64| if l.conjugate { z.conj() } else { z };
                values[l.slot] += turn(near.data[l.operand]);
                for f in 0..q {
                    slopes[l.slot * q + f] = turn(near.data[(1 + f) * near.len + l.operand]);
                }
            }
            let value = |at: usize| if at < len { near.data[at] } else { plan.constants[at - len] };
            let raws: Vec<C64> = plan.raws.iter().map(|products| plan.products[products.clone()].iter().map(|&(x, y)| value(x).conj() * value(y)).sum()).collect();
            for &(slot, raw, conjugated, factor) in &plan.sums {
                values[slot] += factor * if conjugated { raws[raw].conj() } else { raws[raw] };
            }
        }
        let mut merged = Vec::with_capacity(plan.merged);
        for block in &plan.blocks {
            for flag in (0..4).filter(|&flag| block.merged[flag].is_some()) {
                let (hermitian, decay) = (block.hermitian.get(flag).copied().flatten(), block.decay[flag]);
                merged.extend((0..block.places.len()).map(|i| hermitian.map_or(C64::ZERO, |at| values[at + i]) + decay.map_or(C64::ZERO, |at| values[at + i])));
            }
        }
        Middle { model: self, plan: plan.clone(), t, amplitudes, values, merged, directions, slopes, work, scratch: RefCell::default() }
    }

    /// The one generator assembly (module documentation): a cluster's generator at a position and a time under a light set, the
    /// exact elimination at the point.
    pub fn assemble(&self, cluster: usize, r: [f64; 3], t: f64, lights: &Lights) -> Middle<'_> {
        self.middle(&self.plan(cluster, lights), r, t, None)
    }
}

/// The momentum's part of a cluster's generator along a straight path through its middle (`Middle::path`): per beam of the
/// plan its Doppler shift w_l at the momentum the generator is taken at (rad/µs) and its path rate k_l·v at the path's
/// velocity (rad/µs); per state of the cluster its Doppler frame's rate K_s·v, and whether any frame turns.
#[derive(Clone, Debug)]
pub struct Path {
    shifts: Vec<f64>,
    rates: Vec<f64>,
    gauge: Vec<f64>,
    turning: bool,
}

/// A node's scratch (`Middle::scratch`): the beams' amplitudes turned along the path, the plan's field components there (each
/// with its gradient, plain and weighted by its beams' Doppler shifts), the tones' phases, the Doppler frames' turns, a ρ laid
/// out as the entries are, per tone its three sums, and a form's slopes.
#[derive(Clone, Debug, Default)]
struct Scratch {
    turned: Vec<[C64; 4]>,
    plain: Vec<[C64; 4]>,
    doppler: Vec<[C64; 4]>,
    phases: Vec<C64>,
    turns: Vec<C64>,
    weights: Vec<C64>,
    sums: Vec<[f64; 3]>,
    slopes: Vec<C64>,
}

impl crate::step::Middle for Middle<'_> {
    type Path = Path;

    fn states(&self) -> usize {
        self.plan.states
    }

    /// `Plan::still`.
    fn still(&self) -> bool {
        self.plan.still
    }

    fn path(&self, kappa: [f64; 3], velocity: [f64; 3]) -> Path {
        let (model, plan) = (self.model, &*self.plan);
        let gauge: Vec<f64> = model.clusters[plan.cluster].waves.iter().map(|k| dot3(*k, velocity)).collect();
        Path {
            shifts: plan.beams.iter().map(|&l| model.doppler_shift(l, kappa)).collect(),
            rates: plan.beams.iter().map(|&l| dot3(model.light.beams[l].wave_vector(), velocity)).collect(),
            turning: gauge.iter().any(|rate| *rate != 0.0),
            gauge,
        }
    }

    fn gauge(&self, path: &Path, rates: &mut [f64]) {
        rates.copy_from_slice(&path.gauge);
    }

    /// Module documentation, # At a point, and along a path through it.
    fn generator(&self, path: &Path, s: f64, h: &mut Mat) {
        let (plan, n) = (&*self.plan, self.plan.states);
        let mut scratch = self.scratch.borrow_mut();
        let scratch = &mut *scratch;
        self.along(path, Some(s), 1, scratch);
        h.data.fill(C64::ZERO);
        for block in &plan.blocks {
            let (l, lw, r, rw) = (side(&scratch.plain, block.left), side(&scratch.doppler, block.left), side(&scratch.plain, block.right), side(&scratch.doppler, block.right));
            let c = [l[0] * r[0], l[0] * rw[0], lw[0] * r[0], lw[0] * rw[0]];
            for flag in 0..4 {
                let Some(at) = block.merged[flag] else { continue };
                let values = &self.merged[at..at + block.places.len()];
                match block.tone {
                    Some(tone) => {
                        let scale = c[flag] * scratch.phases[tone];
                        for (&place, v) in block.places.iter().zip(values) {
                            h.data[place] += scale * *v;
                        }
                    }
                    None => {
                        for ((&place, &tone), v) in block.places.iter().zip(&block.tones).zip(values) {
                            h.data[place] += c[flag] * *v * scratch.phases[tone];
                        }
                    }
                }
            }
        }
        if path.turning {
            for x in 0..n {
                let turn = scratch.turns[x];
                for y in 0..n {
                    h.data[x * n + y] *= turn;
                    h.data[y * n + x] *= turn.conj();
                }
                h.data[x * n + x] += C64::real(path.gauge[x]);
            }
        }
    }

    fn force(&self, path: &Path, s: f64, rho: &[C64]) -> [f64; 3] {
        let mut out = [0.0; 3];
        self.contract(path, Some(s), rho, &mut |_, f| out = f);
        out
    }

    fn tones(&self, path: &Path, rho: &[C64], each: &mut dyn FnMut(f64, [f64; 3])) {
        self.contract(path, None, rho, each)
    }

    fn work(&self) -> [f64; 3] {
        let node = self.plan.node;
        let form = node[1] + self.directions.len() as f64 * node[2];
        [node[0], form, form]
    }
}

/// A block's field component on one side (1 where it has none).
fn side(fields: &[[C64; 4]], scalar: Option<usize>) -> [C64; 4] {
    scalar.map_or([C64::ONE, C64::ZERO, C64::ZERO, C64::ZERO], |s| fields[s])
}

impl Middle<'_> {
    /// A node's field components into its scratch, their first `parts` parts (the value; with the gradient, four): at the
    /// time s from the middle along a path (`on`) each beam's amplitude and gradient at the middle times its plane wave's
    /// phase over the path, e^{i k_l·v s}, with the tones' phases at that time and the Doppler frames' turns; at the fixed
    /// middle point (None) the middle's own, at its time.
    fn along(&self, path: &Path, on: Option<f64>, parts: usize, scratch: &mut Scratch) {
        let plan = &*self.plan;
        let s = on.unwrap_or(0.0);
        scratch.turned.clear();
        scratch.turned.extend(self.amplitudes.iter().zip(&path.rates).map(|(a, rate)| {
            let phase = if on.is_some() { C64::cis(rate * s) } else { C64::ONE };
            std::array::from_fn(|part| if part < parts { a[part] * phase } else { C64::ZERO })
        }));
        scratch.plain.clear();
        scratch.doppler.clear();
        for scalar in &plan.scalars {
            let (mut plain, mut doppler) = ([C64::ZERO; 4], [C64::ZERO; 4]);
            for &(at, c) in &scalar.beams {
                for part in 0..parts {
                    let z = (if scalar.conjugate { scratch.turned[at][part].conj() } else { scratch.turned[at][part] }) * c;
                    plain[part] += z;
                    doppler[part] += z * path.shifts[at];
                }
            }
            scratch.plain.push(plain);
            scratch.doppler.push(doppler);
        }
        scratch.phases.clear();
        scratch.phases.extend(plan.tones.iter().map(|&omega| if omega == 0.0 { C64::ONE } else { C64::cis(omega * (self.t + s)) }));
        scratch.turns.clear();
        scratch.turns.extend(path.gauge.iter().map(|rate| if on.is_some() && path.turning { C64::cis(-rate * s) } else { C64::ONE }));
    }

    /// Re tr(Ĝ_a(s) ρ) per lab axis a: the gradient of the generator's Hermitian part along a path at the time s from the
    /// middle (`on`), or at the fixed middle point (None: every tone in turn, to `each` with its beat Ω, its entries at their
    /// phase at the middle's time), contracted with a Hermitian ρ on the cluster's states (n × n, row by row) without forming
    /// it: per block and flag Σ_entries V[x, y] ρ[y, x] and the same of its slopes, times the fields' product and its gradient
    /// -- of a flag's values and their adjoints, whose forms with a Hermitian ρ are conjugates, one, twice (`Block::weights`).
    fn contract(&self, path: &Path, on: Option<f64>, rho: &[C64], each: &mut dyn FnMut(f64, [f64; 3])) {
        let (plan, n, q) = (&*self.plan, self.plan.states, self.directions.len());
        let mut scratch = self.scratch.borrow_mut();
        let scratch = &mut *scratch;
        self.along(path, on, 4, scratch);
        // ρ[y, x] in the Doppler frames along the path, laid out as the entries x ← y are.
        scratch.weights.clear();
        scratch.weights.extend((0..n * n).map(|place| rho[(place % n) * n + place / n] * scratch.turns[place / n] * scratch.turns[place % n].conj()));
        scratch.sums.clear();
        scratch.sums.resize(if on.is_some() { 1 } else { plan.tones.len() }, [0.0; 3]);
        scratch.slopes.clear();
        scratch.slopes.resize(q, C64::ZERO);
        let (weights, phases, slopes) = (&scratch.weights, &scratch.phases, &mut scratch.slopes);
        for block in &plan.blocks {
            if (0..3).all(|flag| block.hermitian[flag].is_none() || block.weights[flag] == 0.0) {
                continue;
            }
            let (l, lw, r, rw) = (side(&scratch.plain, block.left), side(&scratch.doppler, block.left), side(&scratch.plain, block.right), side(&scratch.doppler, block.right));
            let product = |l: &[C64; 4], r: &[C64; 4]| -> [C64; 4] { std::array::from_fn(|part| if part == 0 { l[0] * r[0] } else { l[part] * r[0] + l[0] * r[part] }) };
            let c = [product(&l, &r), product(&l, &rw), product(&lw, &r)];
            // The block's tones in turn: all its entries at once along a path, or where they share one.
            let whole = on.is_some() || block.tone.is_some();
            let mut tone = block.tone.unwrap_or(0);
            loop {
                let mut next = None;
                for (flag, at) in block.hermitian.iter().enumerate() {
                    let Some(at) = at.filter(|_| block.weights[flag] > 0.0) else { continue };
                    let mut value = C64::ZERO;
                    slopes.fill(C64::ZERO);
                    match block.tone {
                        // One tone: its phase is the sums' common factor.
                        Some(shared) => {
                            for (i, &place) in block.places.iter().enumerate() {
                                let w = weights[place];
                                value += self.values[at + i] * w;
                                for (f, slope) in slopes.iter_mut().enumerate() {
                                    *slope += self.slopes[(at + i) * q + f] * w;
                                }
                            }
                            value = value * phases[shared];
                            slopes.iter_mut().for_each(|slope| *slope = *slope * phases[shared]);
                        }
                        None => {
                            for (i, (&place, &entry)) in block.places.iter().zip(&block.tones).enumerate() {
                                if !whole && entry != tone {
                                    if entry > tone && next.is_none_or(|t| entry < t) {
                                        next = Some(entry);
                                    }
                                    continue;
                                }
                                let w = weights[place] * phases[entry];
                                value += self.values[at + i] * w;
                                for (f, slope) in slopes.iter_mut().enumerate() {
                                    *slope += self.slopes[(at + i) * q + f] * w;
                                }
                            }
                        }
                    }
                    let sum = &mut scratch.sums[if on.is_some() { 0 } else { tone }];
                    for a in 0..3 {
                        let along: C64 = slopes.iter().zip(&self.directions).map(|(slope, direction)| *slope * direction[a]).sum();
                        sum[a] += block.weights[flag] * (c[flag][1 + a] * value + c[flag][0] * along).re;
                    }
                }
                match next.filter(|_| !whole) {
                    Some(t) => tone = t,
                    None => break,
                }
            }
        }
        for (m, sum) in scratch.sums.iter().enumerate() {
            each(if on.is_some() { 0.0 } else { plan.tones[m] }, *sum);
        }
    }
}

#[cfg(test)]
mod tests;
