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
//! A term's entry a ← b lies at the residual beat Ω + θ_a − θ_b and, each product of two beams' plane waves apart, at the residual
//! wave vector q − (K_a − K_b) in the cluster's frames (cluster.rs): a part of the generator (model.rs `Part`). A Hermitian term
//! adds c M to its part and ∇(c M) to the part's force; a decay term adds no force (the absorption's mean force is the jumps').
//! Each Hermitian term comes with its adjoint, so the parts pair, (Ω, q) with (−Ω, −q), and G_{−j} = G_j†.
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
//! # At a point
//! `Middle`: the plan's beams' amplitudes with their gradients, the near forms there, and from them every term's values --
//! the momentum does not enter. `Middle::generator` expands them at a momentum into the parts: a term's two field components are
//! sums over beams, so its value multiplies every pair of their beams' amplitudes, a_l' a_l (each conjugated or not) -- the pair's
//! plane waves give its wave vector, its two beams' Doppler shifts its weights. Per pair of beams the terms of every block that
//! holds it are summed first, each with its two components' weights γ; one product with the pair's amplitudes and its gradient then
//! gives the pair's entries of the parts. The Hermitian values' gradients are carried as slopes along a few directions -- a table's
//! along its axes' coordinates' gradients, the exact elimination's along the lab axes --, so that a table of one axis sums one
//! slope where the gradient has three components.
//!
//! # Work
//! The complex multiply-adds are counted where they are made (`Middle::work`, `Plan::expansion`): the near forms (the table's
//! interpolation or the elimination), the jumps' products (a product's adjoint, the same jumps exchanged, is its conjugate and is
//! not made again), and per beam pair, block and entry the values, then per beam pair and entry its part's.

use super::eliminate::Responses;
use super::far::EXCITED;
use super::photon::{Origin, Source};
use super::tabulate::{NearTable, Tables};
use super::{jet, Beat, Component, Generator, Lights, Lit, Model, Part, Point, Sample, Unit, Wave, Work};
use crate::field::Amplitude;
use crate::linalg::{dot3, Mat, C64};
use crate::request::{GROUND, STATES};
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
    /// An entry's jet in the position to the forms' order.
    pub(crate) fn jet(&self, entry: usize) -> Vec<C64> {
        match (&self.directions, self.order) {
            (None, _) => (0..jet::parts(self.order)).map(|p| self.data[p * self.len + entry]).collect(),
            (Some(_), 0) => vec![self.data[entry]],
            (Some(_), _) => self.first_order(entry).to_vec(),
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

/// A pair of beams whose amplitudes' product multiplies terms (module documentation, # At a point): the two beams (place among
/// the plan's beams, conjugated or not; None for a factor that is 1), the entries x ← y its blocks hold with their parts and
/// places x n + y there, and per block the two field components' weights' product and where each of the block's entries lies
/// among the pair's.
#[derive(Clone, Debug)]
struct Pair {
    left: Option<(usize, bool)>,
    right: Option<(usize, bool)>,
    places: Vec<(usize, usize)>,
    blocks: Vec<(usize, C64, Vec<usize>)>,
}

/// The terms of one product of field components at one base beat (module documentation, # Terms): its entries' count and, per
/// flag (`PLAIN`, `RIGHT`, `LEFT`; and `BOTH` for the decay), where its Hermitian values and its decay's start among a point's
/// values (one per entry).
#[derive(Clone, Debug)]
struct Block {
    left: Option<usize>,
    right: Option<usize>,
    entries: usize,
    hermitian: [Option<usize>; 3],
    decay: [Option<usize>; 4],
}

/// A part's key: its exact residual beat with its value Ω (rad/µs) and its residual wave vector (1/µm).
#[derive(Clone, Debug)]
struct Key {
    beat: Beat,
    omega: f64,
    wave: [f64; 3],
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
/// their beams, the blocks of terms, the parts, the photon classes.
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
    pairs: Vec<Pair>,
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
    keys: Vec<Key>,
    pub(crate) classes: Vec<ClassPlan>,
    /// The complex multiply-adds of one expansion (`Middle::generator`): without the gradients' slopes, and per direction of
    /// them; and of a point's values from its near forms.
    expansion: [f64; 2],
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
            out.push((class.landing, model.class_wave_number(class), members));
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
        // The parts, the static one first; and the beam pairs, each with the blocks whose two field components hold its beams.
        let mut index: HashMap<(Beat, Wave), usize> = HashMap::new();
        let mut keys: Vec<Key> = vec![];
        let mut part = |beat: Beat, wave: Wave| -> usize {
            *index.entry((beat.clone(), wave.clone())).or_insert_with(|| {
                keys.push(Key { omega: model.omega(&beat), wave: model.wave_value(&wave), beat });
                keys.len() - 1
            })
        };
        part(model.axes.zero(), model.no_wave());
        let mut pairs: Vec<Pair> = vec![];
        let mut found: HashMap<(Option<(usize, bool)>, Option<(usize, bool)>), usize> = HashMap::new();
        let mut placed: Vec<HashMap<(usize, usize), usize>> = vec![];
        let mut blocks = vec![];
        // The multiply-adds of an expansion: per beam pair, block and entry each flag's value and the pair's weight on their sum
        // (the Hermitian ones', the decay's; and per direction the Hermitian slopes'), then per beam pair and entry its part's value
        // and, of Hermitian terms, its three forces' (and per direction the slopes' share of them).
        let mut expansion = [0.0; 2];
        for (b, d) in self.drafts.iter().enumerate() {
            let sides = |s: Option<usize>| -> Vec<Option<(usize, bool, C64)>> { s.map_or(vec![None], |i| self.scalars[i].beams.iter().map(|&(at, c)| Some((at, self.scalars[i].conjugate, c))).collect()) };
            let wave_of = |beam: &Option<(usize, bool, C64)>| -> Wave { beam.map_or(model.no_wave(), |(at, conjugate, _)| model.no_wave().plus(if conjugate { -1 } else { 1 }, &model.beam_wave(self.beams[at]))) };
            let count = |used: &[bool]| used.iter().filter(|u| **u).count();
            let each = |flags: usize| if flags > 0 { (d.entries.len() * (flags + 1)) as f64 } else { 0.0 };
            for l in sides(d.left) {
                for r in sides(d.right) {
                    let key = (l.map(|x| (x.0, x.1)), r.map(|x| (x.0, x.1)));
                    let at = *found.entry(key).or_insert_with(|| {
                        pairs.push(Pair { left: key.0, right: key.1, places: vec![], blocks: vec![] });
                        placed.push(HashMap::new());
                        pairs.len() - 1
                    });
                    let wave = wave_of(&l).plus(1, &wave_of(&r));
                    let weight = l.map_or(C64::ONE, |x| x.2) * r.map_or(C64::ONE, |x| x.2);
                    let mut map = vec![];
                    for &(x, y) in &d.entries {
                        let beat = d.beat.plus(1, &cluster.frames[x]).plus(-1, &cluster.frames[y]);
                        let residual = wave.plus(-1, &cluster.waves_exact[x]).plus(1, &cluster.waves_exact[y]);
                        let place = (part(beat, residual), x * n + y);
                        let places = &mut pairs[at].places;
                        map.push(*placed[at].entry(place).or_insert_with(|| {
                            places.push(place);
                            places.len() - 1
                        }));
                    }
                    pairs[at].blocks.push((b, weight, map));
                    expansion[0] += each(count(&d.used.0)) + each(count(&d.used.1));
                    expansion[1] += each(count(&d.used.0));
                }
            }
            blocks.push(Block { left: d.left, right: d.right, entries: d.entries.len(), hermitian: hermitian[b], decay: decay[b] });
        }
        for pair in &pairs {
            let own = pair.blocks.iter().any(|(b, _, _)| blocks[*b].hermitian.iter().any(|h| h.is_some()));
            expansion[0] += (pair.places.len() * if own { 4 } else { 1 }) as f64;
            expansion[1] += (pair.places.len() * if own { 3 } else { 0 }) as f64;
        }
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
        let values = (products.len() + sums.len()) as f64;
        Plan {
            pairs,
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
            keys,
            classes,
            expansion,
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

/// A plan's field components at a point (`Model::scalars`): per component its jet to an order, plain and weighted by its beams'
/// Doppler shifts.
pub(crate) struct Fields {
    pub(crate) plain: Vec<Vec<C64>>,
    pub(crate) doppler: Vec<Vec<C64>>,
}

/// A cluster's generator at a position and a time, for every momentum (module documentation, # At a point).
pub struct Middle<'a> {
    model: &'a Model,
    plan: Arc<Plan>,
    r: [f64; 3],
    t: f64,
    /// Per beam of the plan: a_l and ∇a_l.
    amplitudes: Vec<[C64; 4]>,
    /// Every block's values per flag and entry, the Hermitian ones first.
    values: Vec<C64>,
    /// The directions the Hermitian values' gradients are slopes along (1/µm per unit of each), and per Hermitian value its slopes
    /// (`slopes[slot * directions + f]`).
    directions: Vec<[f64; 3]>,
    slopes: Vec<C64>,
    /// The complex multiply-adds that made it.
    pub(crate) work: f64,
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

    /// TEMPORARY: a plan's sizes.
    pub fn plan_summary(&self, cluster: usize) -> String {
        let plan = self.plan(cluster, &self.running());
        let entry_terms: usize = plan.pairs.iter().map(|p| p.blocks.iter().map(|b| b.2.len()).sum::<usize>()).sum();
        let pairs: usize = plan.pairs.len();
        format!(
            "near: pairs {} shifts {} jumps {} responses {} (len {}); constants {}; beams {}; scalars {}; blocks {} (beam pairs {}, entry-terms {}); slots {} (hermitian {}), links {}, sums {}, products {}; parts {}; classes {} (member entries {}); mixing blocks {}",
            plan.near.pairs.len(), plan.near.shifts.len(), plan.near.jumps.len(), plan.near.responses.len(), plan.near.len(), plan.constants.len(), plan.beams.len(), plan.scalars.len(), plan.blocks.len(), pairs, entry_terms,
            plan.template.len(), plan.hermitian, plan.links.len(), plan.sums.len(), plan.products.len(), plan.keys.len(), plan.classes.len(), plan.classes.iter().flat_map(|c| c.members.iter().map(|m| m.entries.len())).sum::<usize>(),
            {
                let count = |f: &dyn Fn(usize, usize) -> bool| {
                    let mut block: Vec<usize> = (0..EXCITED).collect();
                    for a in 0..EXCITED { for b in 0..EXCITED { if f(a, b) { let (x, y) = (block[a], block[b]); if x != y { block.iter_mut().filter(|z| **z == y).for_each(|z| *z = x); } } } }
                    block.sort(); block.dedup(); block.len()
                };
                let mut b = self.mixing[cluster].clone(); b.sort(); b.dedup();
                format!("{} (base alone {}, decay alone {}, far forms alone {}, off-diagonal decay max {:.2e} of {:.2e})", b.len(),
                    count(&|a, b| self.base[(GROUND + a, GROUND + b)] != C64::ZERO),
                    count(&|a, b| self.decay[(a, b)] != C64::ZERO),
                    count(&|a, b| self.tones.iter().any(|t| t.pairs.iter().any(|p| p[(GROUND + a, GROUND + b)] != C64::ZERO))),
                    (0..EXCITED * EXCITED).filter(|i| i / EXCITED != i % EXCITED).map(|i| self.decay[(i / EXCITED, i % EXCITED)].abs()).fold(0.0, f64::max), self.decay[(0, 0)].abs())
            }
        )
    }

    /// A plan's field components at a point, jets to `order`, plain and weighted by their beams' Doppler shifts at a momentum.
    pub(crate) fn fields(&self, plan: &Plan, amplitudes: &[Amplitude], kappa: [f64; 3], order: usize) -> Fields {
        let shifts: Vec<f64> = plan.beams.iter().map(|&l| self.doppler_shift(l, kappa)).collect();
        let mut out = Fields { plain: vec![], doppler: vec![] };
        for s in &plan.scalars {
            let (mut plain, mut doppler) = (vec![C64::ZERO; jet::parts(order)], vec![C64::ZERO; jet::parts(order)]);
            for &(at, c) in &s.beams {
                for (p, a) in jet::of(&amplitudes[plan.beams[at]], order).into_iter().enumerate() {
                    let z = (if s.conjugate { a.conj() } else { a }) * c;
                    plain[p] += z;
                    doppler[p] += z * shifts[at];
                }
            }
            out.plain.push(plain);
            out.doppler.push(doppler);
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
            return self.middle_of(plan, r, t, amplitudes, None);
        }
        let all = self.light.amplitudes(r);
        let near = self.near_values(plan, &all, &[(plan.near.hermitian(), 1), (plan.near.jumping(), 0)], table);
        self.middle_of(plan, r, t, plan.amplitudes(&all), Some(&near))
    }

    /// The same from the plan's beams' amplitudes and the near forms there (the Hermitian ones to first order): every block's
    /// values, a Hermitian slot its constant and its near form with its slopes, a decay slot its constant and its jumps'
    /// products. Without near forms (None): the terms that hold none.
    pub(crate) fn middle_of(&self, plan: &Arc<Plan>, r: [f64; 3], t: f64, amplitudes: Vec<[C64; 4]>, near: Option<&NearValues>) -> Middle<'_> {
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
        Middle { model: self, plan: plan.clone(), r, t, amplitudes, values, directions, slopes, work }
    }

    /// The one generator assembly (module documentation), the exact elimination at the point.
    pub fn assemble(&self, cluster: usize, at: &Point, lights: &Lights) -> Generator {
        self.assembled(cluster, at, lights, None).0
    }

    /// The assembly and the multiply-adds it counted, the near light from its tables where they serve.
    pub(crate) fn assembled(&self, cluster: usize, at: &Point, lights: &Lights, tables: Option<&Tables>) -> (Generator, f64) {
        let plan = self.plan(cluster, lights);
        let middle = self.middle(&plan, at.r, at.t, tables.and_then(|t| t.of(cluster, lights)));
        (middle.generator(at.kappa), middle.work + plan.expansion[0] + middle.directions.len() as f64 * plan.expansion[1])
    }

    /// The counted work at trajectory states (model.rs `Work`): the assembly's multiply-adds, the clusters' dimension and parts,
    /// means over the samples, every light on.
    pub fn work(&self, samples: &[Sample]) -> Work {
        let lights = self.running();
        let mut out = Work { assembly: 0.0, states: 0.0, parts: 0.0 };
        for s in samples {
            let (g, w) = self.assembled(s.cluster, &s.at, &lights, None);
            out.assembly += w;
            out.states += self.clusters[s.cluster].states.len() as f64;
            out.parts += g.parts.len() as f64;
        }
        let n = samples.len().max(1) as f64;
        Work { assembly: out.assembly / n, states: out.states / n, parts: out.parts / n }
    }
}

impl Middle<'_> {
    /// The generator's parts at a momentum (module documentation, # At a point): per pair of beams the values of its blocks'
    /// entries at the pair's Doppler shifts, summed with the blocks' weights; then the product of the two beams' amplitudes with
    /// its gradient on each entry, into the entry's part; a part at a beat Ω takes its phase e^{iΩt}.
    pub fn generator(&self, kappa: [f64; 3]) -> Generator {
        let (plan, n, q) = (&*self.plan, self.plan.states, self.directions.len());
        let shifts: Vec<f64> = plan.beams.iter().map(|&l| self.model.doppler_shift(l, kappa)).collect();
        let mut parts: Vec<Part> = plan.keys.iter().map(|k| Part { beat: k.beat.clone(), omega: k.omega, wave: k.wave, generator: Mat::zeros(n, n), force: std::array::from_fn(|_| Mat::zeros(n, n)) }).collect();
        let side = |beam: &Option<(usize, bool)>| -> ([C64; 4], f64) {
            match beam {
                Some((at, conjugate)) => (self.amplitudes[*at].map(|z| if *conjugate { z.conj() } else { z }), shifts[*at]),
                None => ([C64::ONE, C64::ZERO, C64::ZERO, C64::ZERO], 0.0),
            }
        };
        // Per entry of the pair: the Hermitian values' sum, the decay's, and the Hermitian slopes' per direction.
        let (mut hermitian, mut decay, mut slopes): (Vec<C64>, Vec<C64>, Vec<C64>) = (vec![], vec![], vec![]);
        for pair in &plan.pairs {
            let ((l, wl), (r, wr)) = (side(&pair.left), side(&pair.right));
            let c = l[0] * r[0];
            let gc: [C64; 3] = std::array::from_fn(|i| l[1 + i] * r[0] + l[0] * r[1 + i]);
            let mut weights = [1.0; 4];
            (weights[RIGHT], weights[LEFT], weights[BOTH]) = (wr, wl, wl * wr);
            let places = pair.places.len();
            hermitian.clear();
            hermitian.resize(places, C64::ZERO);
            decay.clear();
            decay.resize(places, C64::ZERO);
            slopes.clear();
            slopes.resize(places * q, C64::ZERO);
            let mut own = false;
            for (block, weight, map) in &pair.blocks {
                let block = &plan.blocks[*block];
                for (flag, at) in block.hermitian.iter().enumerate() {
                    let Some(at) = at else { continue };
                    own = true;
                    let scale = *weight * weights[flag];
                    for (i, &place) in map.iter().enumerate() {
                        hermitian[place] += scale * self.values[at + i];
                        for f in 0..q {
                            slopes[place * q + f] += scale * self.slopes[(at + i) * q + f];
                        }
                    }
                }
                for (flag, at) in block.decay.iter().enumerate() {
                    let Some(at) = at else { continue };
                    let scale = *weight * weights[flag];
                    for (i, &place) in map.iter().enumerate() {
                        decay[place] += scale * self.values[at + i];
                    }
                }
            }
            for (i, &(part, at)) in pair.places.iter().enumerate() {
                let p = &mut parts[part];
                p.generator.data[at] += c * (hermitian[i] + decay[i]);
                if own {
                    for a in 0..3 {
                        let along: C64 = (0..q).map(|f| slopes[i * q + f] * self.directions[f][a]).sum();
                        p.force[a].data[at] += gc[a] * hermitian[i] + c * along;
                    }
                }
            }
        }
        for (key, part) in plan.keys.iter().zip(parts.iter_mut()) {
            if !key.beat.is_zero() {
                let phase = C64::cis(key.omega * self.t);
                part.generator = part.generator.scaled(phase);
                part.force = std::array::from_fn(|a| part.force[a].scaled(phase));
            }
        }
        Generator { cluster: plan.cluster, at: Point { r: self.r, kappa, t: self.t }, parts }
    }
}
