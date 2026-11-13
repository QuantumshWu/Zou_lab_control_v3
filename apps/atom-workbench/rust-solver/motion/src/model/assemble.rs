//! The one generator assembly (SPEC §1.4): what a cluster's generator is made of under a light set, worked out once (`Plan`), and
//! its evaluation at a point. The dynamics, the force, the hazard, the photons, the readout and the potential take its terms.
//!
//! # Terms
//! Every term of the generator is a product of at most two components of the light's fields at the point times a matrix on the
//! cluster's states, at a base beat Ω (∝ e^{iΩt}), all exact (`Beat`, `Wave`). A field's component is u(r) = Σ_l a_l(r) c_l over
//! beams of one frequency (`Scalar`): its coherent field's component on its basis, e_k = Σ_l a_l γ_lk (field.rs; an averaged
//! light's, each beam's alone), or the conjugate of one. The terms:
//! - the base (H0 less the outside forms at the focus: every light off) and the frames' −θ_s, constant;
//! - each pair of lit frequencies' constant form at their beat, e^ω'_k'* e^ω_k C_k'k (far.rs `constant`; a frequency's own is
//!   its static form): the folds with the bare levels;
//! - the same pairs' part at the atom, e^ω'_k'* e^ω_k S_k'k on the ground states (eliminate.rs), the explicit states' van Vleck
//!   shift by the drives folded there, and the explicit drives, e_k U_k[e, b] and their adjoints;
//! - the decay −(i/2) Σ_φ Σ_a J_{a,φ}†J_{a,φ}: per class and pair of its members their jumps' own product (photon.rs).
//! One rule takes an entry a ← b of the first two kinds into the generator, whatever light makes it (eliminate.rs
//! `Model::term_included`): its residual beat in the frames exactly zero, or its unit kept (`Unit::Term`).
//! A term's entry a ← b lies at the residual beat Ω + θ_a − θ_b in the cluster's frames (cluster.rs), its tone, and each product
//! of two beams' plane waves in it at the residual wave vector q − (K_a − K_b). A Hermitian term adds c M to the generator and
//! ∇(c M) to its gradient; a decay term adds no force (the absorption's mean force is the jumps'). Each Hermitian term comes
//! with its adjoint, so the generator's Hermitian part and its gradient are Hermitian at every time.
//!
//! # Near forms
//! The drives folded with the levels at the atom enter through their elimination's responses on each carrier's field basis,
//! T_k and T2_k (eliminate.rs), by a few complex numbers that are linear in them (`NearLayout`): the shifts' S_k'k =
//! −½(U^f_k'†T_k + T_k'†U^f_k) and their Doppler partners D_k'k = ½ U^f_k'†T2_k on the cluster's ground places, U^f a
//! frequency's every folded coupling (far.rs `Tone::coupled`) and T its response at the atom, none where it is no carrier; the
//! jumps C_a T_k and C_a T2_k; the explicit states' van Vleck shifts; and the responses themselves (the readout's). Each exists where it can differ from zero: the responses of a ground
//! state reach the excited states its coupling reaches through the excited states' own couplings (`Model::reached`). At a point
//! they are the exact elimination's there with their exact derivatives (`Model::near_exact`), or its table's interpolant
//! (tabulate.rs): one definition, tabulated or not.
//!
//! # First-order Doppler
//! A beam's absorption at w_l = (ħ/m)(k_l·κ + k_l²/2) takes T_l − w_l T2_l: with a carrier's field e_k, its Doppler-weighted
//! field Σ_l w_l a_l γ_lk multiplies the partner form. A term's value at a beam pair is therefore v₀ + w_r v_r + w_l v_l +
//! w_l w_r v_lr over the factors that are a near carrier's (`PLAIN`, `RIGHT`, `LEFT`, `BOTH`), the jumps' product exactly: the
//! generator is a Lindblad one at every momentum (from a table: to the table's error, # Dense terms).
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
//! # Dense terms
//! At a node the generator is Σ over (block, flag, tone) of one complex number -- the two components' product, plain or
//! Doppler-weighted as the flag says, times the tone's phase -- times that term's values on the cluster's places x n + y. Each
//! is a dense term (`Term`): its values the real parts and then the imaginary parts on the span of places that hold any, whole
//! vectors of `LANES` reals, so that a node is sums over contiguous reals and nothing else (linalg.rs `axpy`, `dot`). A term
//! of constants keeps its values in the plan; one that a near form or the decay enters has them at the point (`Site`): the
//! generator's values, the Hermitian one and the decay's together, and for the forms with ρ the Hermitian values alone with
//! their slopes. A point's values are their constant part, what the near forms lay there (`Plan::lay`, linear in the forms)
//! and the decay's sums of the jumps' products (`Plan::decay`). A table holds both at its nodes in this same layout
//! (tabulate.rs, # A node's reals), so a middle interpolates each number straight to its place onto its constant and
//! multiplies no jumps: its decay is the jumps' product to the table's error, and exactly so at the table's nodes.
//!
//! # The jumps' matrices
//! A state's photons (photon.rs) are its classes' jumps on the state: per class the vector u on the rows a n + b' -- the
//! three axes a of the emitted photon's polarization times the landing cluster's n states b' --, the sum over the class's
//! members of the member's field component times its jumps' matrix on the state. A member's matrix is laid once, a
//! column per source place, for the lanes (`laid`): per vector of `LANES` rows each column's real parts and then its
//! imaginary parts, the columns in turn -- so that a matrix on a state, and the vectors' combinations after it, are sums
//! over contiguous reals and nothing else (linalg.rs `combine`). A constant member's matrix is the plan's
//! (`Plan::emitted`); one that holds a near form -- a carrier's member where its response at the atom reaches -- is a
//! point's (`Site::jumps`): its near jumps laid onto their constant part (`Plan::lay_jumps`), its Doppler partner's
//! matrix after its own, and with them their slopes along the point's directions. A table holds a point's matrices at
//! its nodes in this layout (tabulate.rs, # A node's reals): a point takes them and their slopes in one pass over the
//! stencil's rows, the same interpolant of the same jumps as the forms' own.
//!
//! A middle's numbers and a path's are a thread's spares (`Held`): a step takes the vectors the step before left, and the
//! model's part of a step asks the heap for nothing.
//!
//! # The fourth order
//! The fold's next order on a cluster of ground states whose drives folded at the atom are one frequency's
//! (`Model::fourth`; derivation: fourth.status.md). On the slow invariant subspace Ψ = (x, W x) of the no-jump generator,
//! W = −R + W3, in the orthonormal coordinates y = (1 + X)^{1/2} x, with the second order's own matrices on the cluster at
//! the point -- N = −V†R the near generator (its Hermitian part the shifts S, its anti-Hermitian part −½ L, L = R†D R the
//! folded jumps' product), Y = V†R2 (twice the Doppler partner D) and X = R†R (the metric) --
//!
//! ```text
//! W3 = −R2 N       H⁽⁴⁾ = Herm[−Y N + ½ [X, N]]       L̃_a = J_a (1 − ½X) + J2_a N       decay −(i/2) Σ L̃†L̃
//! ```
//!
//! W3 with one denominator a column: exact where the cluster's ground levels lie within much less than the drives'
//! distances of each other (the rest is the level's error, units.rs). The three matrices are sums over the pairs of the
//! frequency's field components of e_k'* e_k times a form: S and D, and on such a cluster the pairs' X and L with them
//! (`NearLayout::fourth`), tabulated as S is. They are made once a point with their gradients (`Model::quartic`: a
//! thread's last point is kept, so a step's middle and its scattering share one), and from them:
//! - the middle's H⁽⁴⁾ and its gradient, 3 and 6 products of n × n matrices (the commutator's left out where the folded
//!   excited states share one linewidth, L = Γ X: it is nothing there), and the decay's change: with the generator's own
//!   decay at the point by its flags -- Λ = Σ J†J, Λ_R = Σ J†J2, Λ_B = Σ J2†J2, read off its dense terms' values --,
//!   Σ L̃†L̃ − Λ = B†(Λ B + Λ_R N) + N†(Λ_R†B + Λ_B N) − Λ, B = 1 − ½X. Each is the anti-Hermitian read of the
//!   generator's own terms. With the first-order Doppler the response is R − w R2 and the jumps J − w J2, so the terms
//!   whose absorbed component takes the shift w sum to G_R = ½ V^f†R2 + (i/2) Σ J†J2 and those whose emitted one does to
//!   G_L = ½ R2†V^f + (i/2) Σ J2†J: the Hermitian coupling's partner ½ V^f†R2 is in both, G_L† = ½ V^f†R2 − (i/2) Λ_R, and
//!   Λ_R = −i (G_R − G_L†) whatever V^f holds -- as Λ = i (P − P†). (Not G_R less the fold's own Y: a kept pair of two
//!   frequencies has the other frequency's bare leg in V^f and not in Y, and its Hermitian partner was read as decay --
//!   on R6_RAMAN_CO with the pair kept the generator's decay 6e-3 off its photons' rate, of either sign by which leg is
//!   at the atom.) Both are held along a step's path as the point's
//!   forms are, in the cluster's Doppler frames (`Site::quartic`): a node adds n² numbers, a force n² a lab axis;
//! - a state's photons: the jumps' matrices on (1 − ½X)ψ and their partners' on Nψ (photon.rs), every jump of the
//!   cluster on the bent state: the ground amplitude is (1 − ½X) y whatever light it scatters.
//! So the force is still the gradient of the generator's Hermitian part and the decay still the jumps' own product.
//! The fourth order's own first-order Doppler partner and the bare drives' part in N are of the next order and left out;
//! terms that beat in the cluster's frames keep their second order.
//!
//! # Work
//! The complex multiply-adds are counted where they are made (`Middle::work`, `Plan::node`): at a point the near forms (the
//! table's interpolation or the elimination) and the jumps' products (a product's adjoint, the same jumps exchanged, is its
//! conjugate and is not made again); at a node the components from their beams and, per block and flag, its entries. A node's
//! counts are the plan's: no generator is made to know them. They count the definition's operations -- the forms a point
//! takes, an entry per block and flag --, whatever layout the sums are made in.

use super::eliminate::{half, jump, shift, Leg, Responses};
use super::far::EXCITED;
use super::photon::{Origin, Source};
use super::tabulate::NearTable;
use super::{jet, Beat, Component, Level, Lights, Lit, Model, Scattering, Wave};
use crate::field::Amplitude;
use crate::linalg::{axpy, dot, dot3, Mat, C64, LANES};
use crate::request::{GROUND, STATES};
use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::{Deref, DerefMut, Range};
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
    /// X and L of a pair entry (the fourth order's, `NearLayout::fourth`).
    Metric(usize),
    Loss(usize),
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
    /// (left frequency, its basis vector k', right frequency, its k, places x, y): S and D of the pair's entry x ← y.
    pub(crate) pairs: Vec<[usize; 6]>,
    /// (carrier, k', k, places x, y of two explicit states).
    pub(crate) shifts: Vec<[usize; 5]>,
    /// (carrier, k, axis a, landing ground state b, place y): (C_a T_k)[b, y].
    pub(crate) jumps: Vec<[usize; 5]>,
    /// (carrier, k, excited state e, place y): T_k[e, y].
    pub(crate) responses: Vec<[usize; 4]>,
    /// The fourth order's forms of each pair entry, after its S and D (module documentation, # The fourth order): none;
    /// the metric X = T_k'†T_k on x ← y; or X and the loss L = T_k'†D T_k -- one where the folded excited states share
    /// one linewidth Γ, L = Γ X.
    pub(crate) fourth: usize,
}

impl NearLayout {
    /// The complex numbers of a point.
    pub(crate) fn len(&self) -> usize {
        self.forms() * self.pairs.len() + self.shifts.len() + 2 * self.jumps.len() + 2 * self.responses.len()
    }
    /// The forms a pair entry holds.
    fn forms(&self) -> usize {
        2 + self.fourth
    }
    /// Where an operand lies among a point's values and, after them, the plan's constants.
    fn at(&self, o: Operand) -> usize {
        let (p, v, j) = (self.pairs.len(), self.shifts.len(), self.jumps.len());
        match o {
            Operand::Pair(i) => i,
            Operand::Doppler(i) => p + i,
            Operand::Metric(i) => 2 * p + i,
            Operand::Loss(i) => 3 * p + i,
            Operand::Shift(i) => self.forms() * p + i,
            Operand::Jump(i) => self.forms() * p + v + i,
            Operand::Jump2(i) => self.forms() * p + v + j + i,
            Operand::Constant(i) => self.len() + i,
        }
    }
    /// The Hermitian forms (S, D, the shifts), the jumps with their partners, the responses with theirs.
    pub(crate) fn hermitian(&self) -> Range<usize> {
        0..self.forms() * self.pairs.len() + self.shifts.len()
    }
    pub(crate) fn jumping(&self) -> Range<usize> {
        self.hermitian().end..self.hermitian().end + 2 * self.jumps.len()
    }
    pub(crate) fn responding(&self) -> Range<usize> {
        self.jumping().end..self.len()
    }
}

/// The near forms at a point over the entries before `len` (zero outside the sections and the places evaluated), to an order, and the complex
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
    /// The basis vector k of the frequency's field (c_l = γ_lk).
    component: usize,
    pub(crate) conjugate: bool,
    /// (the beam's place among the plan's beams, c_l).
    pub(crate) beams: Vec<(usize, C64)>,
}

/// The terms of one product of field components at one base beat (module documentation, # Terms): its two components
/// (None: 1).
#[derive(Clone, Debug)]
struct Block {
    left: Option<usize>,
    right: Option<usize>,
}

/// One dense term (module documentation, # Dense terms): the entries of one block and flag (`PLAIN`, `RIGHT`, `LEFT`; and
/// `BOTH` for the decay) at one tone -- the residual beat of the block's base beat in the entries' two states' frames, an index
/// into the plan's tones --, on the span of the cluster's places x n + y from `first` on that holds them (`len` of them, whole
/// vectors; zero where the term has no entry). Its values are `len` real parts and then `len` imaginary parts from `at` on:
/// among a point's values where a near form or a decay's sum enters one (`near`), else among the plan's constants.
#[derive(Clone, Debug)]
struct Term {
    block: usize,
    flag: usize,
    tone: usize,
    first: usize,
    len: usize,
    at: usize,
    near: bool,
    /// In a form with a Hermitian ρ (`Middle::contract`), its weight. A flag's values and their adjoints -- the two
    /// components conjugated, the base beat negated, the entries transposed and conjugated, the Doppler-weighted side
    /// exchanged -- make conjugate forms: of a block and its adjoint the first is taken twice and the other left out, and of a
    /// block that is its own adjoint the flag `RIGHT` twice for `LEFT`.
    weight: f64,
}

/// The dense terms of the generator, or of a form with a Hermitian ρ, as they are laid (`Builder::finish`): the terms, the
/// constant part of a point's values, and each term by its block, flag and tone.
#[derive(Default)]
struct Laid {
    terms: Vec<Term>,
    start: Vec<f64>,
    index: HashMap<(usize, usize, usize), usize>,
}

impl Laid {
    /// The term of a block, flag and tone from its entries -- (place, constant, whether a near form or a decay's sum enters
    /// it) --, on the span of those that hold anything; none where none does. A constant term's values go to `constant`.
    fn push(&mut self, constant: &mut Vec<f64>, key: (usize, usize, usize), weight: f64, entries: &[(usize, C64, bool)]) {
        let held: Vec<&(usize, C64, bool)> = entries.iter().filter(|e| e.1 != C64::ZERO || e.2).collect();
        let (Some(low), Some(high)) = (held.iter().map(|e| e.0).min(), held.iter().map(|e| e.0).max()) else { return };
        let first = low / LANES * LANES;
        let len = (high + 1).div_ceil(LANES) * LANES - first;
        let near = held.iter().any(|e| e.2);
        let values = if near { &mut self.start } else { constant };
        let at = values.len();
        values.resize(at + 2 * len, 0.0);
        for e in held {
            values[at + e.0 - first] += e.1.re;
            values[at + len + e.0 - first] += e.1.im;
        }
        self.index.insert(key, self.terms.len());
        self.terms.push(Term { block: key.0, flag: key.1, tone: key.2, first, len, at, near, weight });
    }
    /// Where the entry at `place` of a near term's values lies among a point's: its real part and its imaginary part.
    fn place(&self, key: (usize, usize, usize), place: usize) -> Option<[usize; 2]> {
        self.index.get(&key).map(|&t| &self.terms[t]).map(|t| [t.at + place - t.first, t.at + t.len + place - t.first])
    }
}

/// One member of a photon class at a point (photon.rs): its field component (None: 1), whether its jumps take the
/// Doppler partner, its relative beat's value in a merged class (rad/µs), and its entries (axis a, landing place, source place,
/// where its jump and its partner's lie among a point's values and constants). And the same jumps as one matrix (module
/// documentation, # The jumps' matrices): the source places that hold an entry (`columns`) and where its columns lie
/// (`at`) -- among a point's jumps where any of them is a near form (`near`; its partner's matrix after its own), else
/// among the plan's: a carrier's member whose jumps are all its bare drives' is a constant one, its partner nothing. An
/// explicit state's member is not `folded`.
#[derive(Clone, Debug)]
pub(crate) struct MemberPlan {
    pub(crate) scalar: Option<usize>,
    pub(crate) doppler: bool,
    pub(crate) phase: f64,
    pub(crate) entries: Vec<(usize, usize, usize, usize, usize)>,
    pub(crate) folded: bool,
    pub(crate) near: bool,
    pub(crate) columns: Vec<usize>,
    pub(crate) at: usize,
}

/// One photon class of the cluster under the plan's lights (photon.rs `Class`): its landing cluster, its wave number, the
/// largest wave number of the beams its folded members absorb from (`Photon::absorbed`; zero without one), its lit members;
/// and the rows of its jumps' matrices (module documentation, # The jumps' matrices): the landing cluster's `states`
/// states on each of the three axes, row a `states` + b', as whole vectors (`rows`).
#[derive(Clone, Debug)]
pub(crate) struct ClassPlan {
    pub(crate) landing: usize,
    pub(crate) wave_number: f64,
    pub(crate) absorbed: f64,
    pub(crate) members: Vec<MemberPlan>,
    pub(crate) states: usize,
    pub(crate) rows: usize,
}

/// A Hermitian near form's place in the generator: the block, flag and entry (x, y) it is the value of, the form's place,
/// whether it enters conjugated, and where it lies among a point's values of the generator and of the forms with a
/// Hermitian ρ (its real and its imaginary part; none where its term is left to its adjoint, `Term::weight`).
#[derive(Clone, Debug)]
pub(crate) struct Link {
    pub(crate) block: usize,
    pub(crate) flag: usize,
    pub(crate) place: (usize, usize),
    pub(crate) operand: usize,
    pub(crate) conjugate: bool,
    generator: [usize; 2],
    form: Option<[usize; 2]>,
}

/// What the fourth order of a cluster's fold reads of its plan (module documentation, # The fourth order): its field's
/// basis size m and the cluster's n places; per basis vector the request beams that make its component and their
/// weights; the frequency's pair entries -- (its place among the pairs, k' m + k, x n + y) --; and the folded excited
/// states' one linewidth, where they share one (`Fourth::uniform`).
#[derive(Clone, Debug)]
pub(crate) struct Fourth {
    frequency: usize,
    m: usize,
    pub(crate) n: usize,
    fields: Vec<Vec<(usize, C64)>>,
    entries: Vec<[usize; 3]>,
    uniform: Option<f64>,
}

impl Fourth {
    /// The one linewidth of the excited states a frequency's responses reach from a cluster, where they share one and
    /// no decay couples two of them: there R†D R = Γ R†R. Read of D itself to its rounding: an entry D[e, o] =
    /// Σ_a Σ_b C_a[b, e]* C_a[b, o] is a sum of 3 `GROUND` products and errs by at most γ of the sum of their moduli
    /// (units.rs `rounding`); it is nothing where it is within that, and two diagonal entries are one where they differ
    /// by no more than their two bounds. (On ⁸⁷Rb's D lines in a tweezer there is none: the responses reach D1 and D2
    /// states, 36.13 and 38.12 rad/µs, and the request's D2 widths differ in the fifth digit.)
    fn uniform(model: &Model, cluster: usize, frequency: usize) -> Option<f64> {
        let carrier = model.carrier_of(frequency)?;
        let c = &model.clusters[cluster];
        let mut reached = [false; EXCITED];
        for k in 0..model.carriers[carrier].basis.len() {
            for &b in &c.states[..c.ground] {
                let of = model.reached(cluster, carrier, k, b);
                reached.iter_mut().zip(of).for_each(|(r, more)| *r |= more);
            }
        }
        let reached: Vec<usize> = (0..EXCITED).filter(|&e| reached[e]).collect();
        let bound = |e: usize, o: usize| crate::units::rounding(3 * GROUND + 1) * model.emission.iter().map(|m| (0..GROUND).map(|b| m[(b, GROUND + e)].abs() * m[(b, GROUND + o)].abs()).sum::<f64>()).sum::<f64>();
        let first = *reached.first()?;
        let rate = model.decay[(first, first)].re;
        let one = reached.iter().all(|&e| reached.iter().all(|&o| if e == o { (model.decay[(e, e)].re - rate).abs() <= bound(e, e) + bound(first, first) } else { model.decay[(e, o)].abs() <= bound(e, o) }));
        one.then_some(rate)
    }

    fn new(model: &Model, plan: &Plan, frequency: usize) -> Option<Fourth> {
        let carrier = model.carrier_of(frequency)?;
        let (near, f) = (&plan.near, &model.light.frequencies[frequency]);
        let (m, n) = (model.carriers[carrier].basis.len(), plan.states);
        let fields = (0..m).map(|k| f.beams.iter().zip(&f.coefficients).map(|(&l, c)| (l, c[k])).collect()).collect();
        let entries = near.pairs.iter().enumerate().filter(|(_, p)| p[0] == frequency && p[2] == frequency).map(|(i, p)| [i, p[1] * m + p[3], p[4] * n + p[5]]).collect();
        Some(Fourth { frequency, m, n, fields, entries, uniform: (near.fourth == 1).then(|| Fourth::uniform(model, plan.cluster, frequency)).flatten() })
    }
}

/// The fourth order's matrices of a cluster at a point (module documentation, # The fourth order): N, Y and X on the
/// cluster's places, each its value and its gradient along the lab axes -- `data[(4 form + part) n² + x n + y]` --, the
/// point they are of (the plan, its table's grid, the position), and the complex multiply-adds that made them. A
/// thread's last (`Model::quartic`); the rest is its scratch.
#[derive(Default)]
pub(crate) struct Quartic {
    key: Option<(u64, u64, [u64; 3])>,
    pub(crate) data: Vec<C64>,
    pub(crate) work: f64,
    values: Vec<f64>,
    slopes: Vec<f64>,
    sums: Vec<C64>,
    laid: Vec<f64>,
}

thread_local! {
    static QUARTIC: RefCell<Quartic> = RefCell::new(Quartic::default());
}

/// The plans made so far: each one's own number (`Plan::id`).
static PLANS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// out ← out + scale a b on n × n matrices, row by row.
fn product_onto(out: &mut [C64], a: &[C64], b: &[C64], n: usize, scale: C64) {
    for x in 0..n {
        for c in 0..n {
            let z = a[x * n + c] * scale;
            if z != C64::ZERO {
                for y in 0..n {
                    out[x * n + y] += z * b[c * n + y];
                }
            }
        }
    }
}

/// A cluster's assembly under a light set, worked out once (module documentation): the near forms, the field components and
/// their beams, the blocks of terms, the tones, the photon classes.
#[derive(Clone, Debug)]
pub(crate) struct Plan {
    pub(crate) cluster: usize,
    pub(crate) lights: Lights,
    states: usize,
    pub(crate) near: NearLayout,
    /// The constants the jumps' products and the photons take, after a point's near forms; and per jump entry of the near
    /// forms its constant part, the same field component's bare drives' jump (photon.rs `Origin::Near`).
    pub(crate) constants: Vec<C64>,
    offsets: Vec<C64>,
    /// The request beams the plan's field components sum.
    pub(crate) beams: Vec<usize>,
    pub(crate) scalars: Vec<Scalar>,
    blocks: Vec<Block>,
    /// Per block whether its base beat is exactly zero (the potential's static shifts).
    pub(crate) statics: Vec<bool>,
    /// The cluster's n² places as whole vectors of the dense terms' sums.
    padded: usize,
    /// The generator's dense terms -- per block, flag and tone its Hermitian values and its decay's together --, and those of
    /// a form with a Hermitian ρ: the Hermitian values of the flags that weigh in it. Each in the blocks' order.
    generator: Vec<Term>,
    form: Vec<Term>,
    /// The constant terms' values, and the constant part of a point's values of the generator's terms and of the forms'.
    constant: Vec<f64>,
    pub(crate) generator_start: Vec<f64>,
    pub(crate) form_start: Vec<f64>,
    pub(crate) links: Vec<Link>,
    /// The jumps' products conj(first) second, by sums (each a range of them), and the decay's values: (where among a point's
    /// values of the generator, its sum, whether it takes the sum's conjugate -- the adjoint's, the same jumps exchanged --,
    /// the flag's factor −(i/2)(−1)^{Doppler partners}).
    products: Vec<(usize, usize)>,
    raws: Vec<Range<usize>>,
    sums: Vec<([usize; 2], usize, bool, C64)>,
    /// The distinct residual beats of the terms' entries in the cluster's frames (rad/µs; the first is zero): at a fixed point
    /// the generator is Σ_m (its entries of tone m) e^{iΩ_m t}.
    tones: Vec<f64>,
    /// Each state's Doppler frame K_s (1/µm; cluster.rs, # Doppler frames): on a path r_h + v(t − t_m) the state is
    /// ψ_s = e^{iK_s·v(t − t_m)} x_s, so that x_s holds K_s·v on the diagonal and a plane wave q of an entry a ← b turns at
    /// q − (K_a − K_b). The frames of the least mean square of these under the plan's lights; exact where every pair of
    /// states holds one wave vector and the loops close.
    pub(crate) waves: Vec<[f64; 3]>,
    /// Whether the generator is the same at every time along every path: no entry beats in the frames, in time or along a path
    /// (every residual beat and wave vector exactly zero).
    still: bool,
    /// What a step's nodes sample of the generator along a path: per tone its rate |Ω| (rad/µs) and the largest residual
    /// wave vector |K| (1/µm) of its entries in the cluster's frames -- an entry of the tone turns at Ω + K·v on a path
    /// of the velocity v.
    pub(crate) sampled: Vec<[f64; 2]>,
    /// The complex multiply-adds at a node (module documentation, # Work): of the generator; of the force's form without its
    /// values' slopes; and of those per direction of them.
    node: [f64; 3],
    pub(crate) classes: Vec<ClassPlan>,
    /// The classes' members' jumps as matrices (module documentation, # The jumps' matrices): the constant members'; the
    /// constant part of a point's (the bare drives' jumps of the same field component); and where a point's near jumps
    /// lie among them -- (its real part's place, its imaginary part's, the form's place among a point's values).
    pub(crate) emitted: Vec<f64>,
    pub(crate) jumps_start: Vec<f64>,
    jump_links: Vec<([usize; 2], usize)>,
    /// The complex multiply-adds of a point's values from its near forms: the jumps' products and their sums, the merged values.
    values: f64,
    /// What the fourth order of the cluster's fold reads (module documentation, # The fourth order); none where its fold
    /// is the second order's under these lights. And the plan's own number among the plans made.
    pub(crate) fourth: Option<Fourth>,
    pub(crate) id: u64,
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

/// A jump's value at a point: a constant, or a near form (with its constant part, `Plan::offsets`) and its Doppler partner.
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
    offsets: Vec<C64>,
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
    fn scalar(&mut self, frequency: usize, group: Option<usize>, component: usize, conjugate: bool) -> Option<usize> {
        if let Some(at) = self.scalars.iter().position(|s| (s.frequency, s.group, s.component, s.conjugate) == (frequency, group, component, conjugate)) {
            return Some(at);
        }
        let f = &self.model.light.frequencies[frequency];
        let members: Vec<usize> = group.map_or((0..f.beams.len()).collect(), |j| vec![j]);
        let weights: Vec<(usize, C64)> = members.iter().map(|&j| (f.beams[j], f.coefficients[j][component])).filter(|w| w.1 != C64::ZERO).collect();
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

    /// Whether U^f_k'†T_k of a pair of (frequency, basis vector) can differ from zero on x ← y: the left coupling's excited
    /// states from x among those the right response reaches from y (none where the right frequency is no carrier).
    fn touches(&self, left: (usize, usize), right: (usize, usize), x: usize, y: usize) -> bool {
        let Some(carrier) = self.model.carrier_of(right.0) else { return false };
        let reach = self.model.reached(self.cluster, carrier, right.1, self.states()[y]);
        let u = &self.model.tones[left.0].coupled[0][left.1];
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
        let bare = &self.model.tones[self.model.carriers[carrier].frequency].jumps[0];
        self.offsets.push(bare.get(k).map_or(C64::ZERO, |jumps| jumps[a][(b, self.states()[y])]));
        Some(self.near.jumps.len() - 1)
    }
    /// A source's field component, conjugated or not (None: 1); the outer None where the component vanishes.
    fn source_scalar(&mut self, source: &Source, conjugated: bool) -> Option<Option<usize>> {
        match source.origin {
            Origin::Far { frequency, side, component } => self.scalar(frequency, None, component, (side == 1) != conjugated).map(Some),
            Origin::Near { carrier, component } => self.scalar(self.model.carriers[carrier].frequency, None, component, conjugated).map(Some),
            Origin::Explicit { .. } => Some(None),
        }
    }
    /// A source's jump b ← the state at `place` on axis a; None where it is zero. A carrier's: its near form where its
    /// response at the atom reaches, its constant part with it; elsewhere its constant part alone.
    fn source_jump(&mut self, source: &Source, a: usize, b: usize, place: usize) -> Option<Jump> {
        let s = self.states()[place];
        let near = match source.origin {
            Origin::Near { carrier, component } if place < self.ground() => self.jump(carrier, component, a, b, place),
            _ => None,
        };
        near.map(Jump::Near).or_else(|| Some(self.model.constant_jump(source, a, b, s)).filter(|v| *v != C64::ZERO).map(Jump::Constant))
    }

    /// The base, the frames, and every pair of lit frequencies' constant form at their beat (module documentation, # Terms;
    /// far.rs `constant`): emitting ω' and absorbing ω, e^ω'_k'* e^ω_k C_k'k at δ_ω' − δ_ω, each entry by the one rule -- the
    /// base's too, a static one. The pair exchanged is the adjoint's.
    fn constants(&mut self) {
        let model = self.model;
        let (states, n) = (self.states(), self.states().len());
        let zero = model.axes.zero();
        let still = self.block(None, None, &zero);
        for x in 0..n {
            for y in 0..n {
                let v = model.base[(states[x], states[y])];
                if v != C64::ZERO && model.term_included(self.cluster, x, y, &zero) {
                    self.hermitian(still, PLAIN, x, y, v, None);
                }
            }
        }
        for (x, theta) in model.clusters[self.cluster].frames.iter().enumerate() {
            self.hermitian(still, PLAIN, x, x, C64::real(-model.omega(theta)), None);
        }
        for (to, emitted) in model.tones.iter().enumerate() {
            for (from, absorbed) in model.tones.iter().enumerate() {
                let groups: Vec<(Option<usize>, Option<usize>)> = if to == from { self.groups(to).into_iter().map(|x| (x, x)).collect() } else if self.lights.cross(to, from) { vec![(None, None)] } else { vec![] };
                if groups.is_empty() {
                    continue;
                }
                let beat = emitted.beat.plus(-1, &absorbed.beat);
                let (forms, r) = (model.constants(to, from), absorbed.basis[0].len());
                for (gl, gr) in groups {
                    for (at, form) in forms.forms.iter().enumerate() {
                        let entries: Vec<(usize, usize, C64)> = (0..n * n)
                            .map(|i| (i / n, i % n, form[(states[i / n], states[i % n])]))
                            .filter(|&(x, y, v)| v != C64::ZERO && model.term_included(self.cluster, x, y, &beat))
                            .collect();
                        if entries.is_empty() {
                            continue;
                        }
                        let (Some(left), Some(right)) = (self.scalar(to, gl, at / r, true), self.scalar(from, gr, at % r, false)) else { continue };
                        let block = self.block(Some(left), Some(right), &beat);
                        for (x, y, v) in entries {
                            self.hermitian(block, PLAIN, x, y, v, None);
                        }
                    }
                }
            }
        }
    }

    /// The terms at the atom (module documentation, # Terms): per pair of lit frequencies the ground places' shifts and
    /// couplings at the beat δ_ω' − δ_ω where either's response reaches -- each entry by the one rule (eliminate.rs
    /// `Model::term_included`) --, and per lit carrier the explicit states' van Vleck shifts and the explicit drives with
    /// their adjoints.
    fn near(&mut self) {
        let model = self.model;
        let (states, n, g) = (self.states(), self.states().len(), self.ground());
        let zero = model.axes.zero();
        for (wj, emitted) in model.tones.iter().enumerate() {
            for (wi, absorbed) in model.tones.iter().enumerate() {
                let groups: Vec<(Option<usize>, Option<usize>)> = if wi == wj { self.groups(wi).into_iter().map(|x| (x, x)).collect() } else if self.lights.cross(wj, wi) { vec![(None, None)] } else { vec![] };
                if groups.is_empty() {
                    continue;
                }
                let beat = emitted.beat.plus(-1, &absorbed.beat);
                for kl in 0..emitted.basis[0].len() {
                    for kr in 0..absorbed.basis[0].len() {
                        for x in 0..g {
                            for y in 0..g {
                                if !model.term_included(self.cluster, x, y, &beat) {
                                    continue;
                                }
                                let Some(at) = self.pair((wj, kl), (wi, kr), x, y) else { continue };
                                let exchanged = self.pair((wi, kr), (wj, kl), y, x).expect("a pair's entry comes with its adjoint's");
                                for &(gl, gr) in &groups {
                                    let (Some(left), Some(right)) = (self.scalar(wj, gl, kl, true), self.scalar(wi, gr, kr, false)) else { continue };
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
        for (ci, cc) in model.carriers.iter().enumerate() {
            if !self.lights.on(cc.frequency) {
                continue;
            }
            let (component, tone) = (Component { frequency: cc.frequency, counter: false }, &model.tones[cc.frequency]);
            let folded = &tone.coupled[0];
            for kl in 0..cc.basis.len() {
                for kr in 0..cc.basis.len() {
                    for x in g..n {
                        for y in g..n {
                            // The entry e ← f's halves that can differ from zero: a drive folded at the atom of one leg against
                            // a folded one of the other, from one ground state (`Model::forms`).
                            let (e, f) = (states[x], states[y]);
                            let through = |b: usize| (cc.basis[kr][(e, b)] != C64::ZERO && folded[kl][(f, b)] != C64::ZERO) || (folded[kr][(e, b)] != C64::ZERO && cc.basis[kl][(f, b)] != C64::ZERO);
                            if !(0..GROUND).any(through) {
                                continue;
                            }
                            self.near.shifts.push([ci, kl, kr, x, y]);
                            let at = self.near.shifts.len() - 1;
                            for group in self.groups(cc.frequency) {
                                let (Some(left), Some(right)) = (self.scalar(cc.frequency, group, kl, true), self.scalar(cc.frequency, group, kr, false)) else { continue };
                                let block = self.block(Some(left), Some(right), &zero);
                                self.hermitian(block, PLAIN, x, y, C64::ZERO, Some((Operand::Shift(at), false)));
                            }
                        }
                    }
                }
            }
            for (k, u) in tone.basis[0].iter().enumerate() {
                let explicit: Vec<(usize, usize, C64)> = (g..n)
                    .flat_map(|p| (0..g).map(move |q| (p, q)))
                    .map(|(p, q)| (p, q, u[(states[p], states[q])]))
                    .filter(|&(p, q, v)| v != C64::ZERO && model.level(component, states[p], states[q]) == Level::Explicit)
                    .collect();
                if explicit.is_empty() {
                    continue;
                }
                let (Some(plain), Some(conjugated)) = (self.scalar(cc.frequency, None, k, false), self.scalar(cc.frequency, None, k, true)) else { continue };
                let (up, down) = (self.block(None, Some(plain), &cc.beat.negated()), self.block(Some(conjugated), None, &cc.beat));
                for (p, q, v) in explicit {
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
                    // A jump at the atom comes with its Doppler partner; a constant has none.
                    let partners = |jump: &Jump| if matches!(jump, Jump::Near(_)) { vec![false, true] } else { vec![false] };
                    for &(bj, xj) in &mj.entries {
                        for &(bi, xi) in &mi.entries {
                            if bj != bi {
                                continue;
                            }
                            for a in 0..3 {
                                let (Some(first), Some(second)) = (self.source_jump(sj, a, landing.states[bj], xj), self.source_jump(si, a, landing.states[bi], xi)) else { continue };
                                for pj in partners(&first) {
                                    for pi in partners(&second) {
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
    fn classes(&mut self) -> Vec<(usize, [f64; 2], Vec<(usize, Option<usize>, bool, f64, Vec<(usize, usize, usize, Operand, Operand)>)>)> {
        let model = self.model;
        let mut out = vec![];
        for class in &model.classes[self.cluster] {
            let landing = &model.clusters[class.landing];
            let (mut members, mut absorbed) = (vec![], 0.0_f64);
            for m in &class.members {
                let source = &model.sources[m.source];
                if !model.lit(source, self.lights) {
                    continue;
                }
                let Some(scalar) = self.source_scalar(source, false) else { continue };
                if let Some(frequency) = model.source_frequency(source) {
                    absorbed = model.light.frequencies[frequency].beams.iter().map(|&l| model.light.beams[l].wave_number).fold(absorbed, f64::max);
                }
                let mut entries = vec![];
                for &(bp, sp) in &m.entries {
                    for a in 0..3 {
                        match self.source_jump(source, a, landing.states[bp], sp) {
                            None => {}
                            Some(Jump::Near(i)) => entries.push((a, bp, sp, Operand::Jump(i), Operand::Jump2(i))),
                            Some(Jump::Constant(v)) => {
                                // A constant has no Doppler partner: a carrier's member reads zero for it.
                                self.constants.extend([v, C64::ZERO]);
                                let c = Operand::Constant(self.constants.len() - 2);
                                let partner = if matches!(source.origin, Origin::Near { .. }) { Operand::Constant(self.constants.len() - 1) } else { c };
                                entries.push((a, bp, sp, c, partner));
                            }
                        }
                    }
                }
                members.push((m.source, scalar, matches!(source.origin, Origin::Near { .. }), model.omega(&m.phase), entries));
            }
            out.push((class.landing, [model.class_wave_number(self.cluster, class), absorbed], members));
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
        // The entries' tones, and every plane wave of every entry at a tone, once: (x, y, tone, its residual wave vector in
        // the tree's frames, exactly) -- the wave vector of a pair of the block's two components' beams.
        let mut tone_index: HashMap<Beat, usize> = HashMap::from([(model.axes.zero(), 0)]);
        let mut tones = vec![0.0];
        let mut held: Vec<(usize, usize, usize, Wave)> = vec![];
        let mut entry_tones: Vec<Vec<usize>> = vec![];
        for d in &self.drafts {
            let waves = |s: Option<usize>| -> Vec<Wave> {
                s.map_or(vec![model.no_wave()], |i| self.scalars[i].beams.iter().map(|&(at, _)| model.no_wave().plus(if self.scalars[i].conjugate { -1 } else { 1 }, &model.beam_wave(self.beams[at]))).collect())
            };
            let (left, right) = (waves(d.left), waves(d.right));
            let mut pairs: Vec<Wave> = left.iter().flat_map(|l| right.iter().map(move |r| l.plus(1, r))).collect();
            pairs.sort();
            pairs.dedup();
            let mut block_tones = vec![];
            for &(x, y) in &d.entries {
                let beat = d.beat.plus(1, &cluster.frames[x]).plus(-1, &cluster.frames[y]);
                let frames = cluster.waves_exact[x].plus(-1, &cluster.waves_exact[y]);
                let tone = *tone_index.entry(beat.clone()).or_insert_with(|| {
                    tones.push(model.omega(&beat));
                    tones.len() - 1
                });
                held.extend(pairs.iter().map(|q| (x, y, tone, q.plus(-1, &frames))));
                block_tones.push(tone);
            }
            entry_tones.push(block_tones);
        }
        held.sort();
        held.dedup();
        // The Doppler frames (cluster.rs, # Doppler frames): the tree's with the least squares' correction of what it leaves;
        // whether anything beats -- a residual beat in time, or along a path a residual wave vector --; and per tone the
        // largest residual wave vector of its entries (`Plan::sampled`).
        let correction = super::cluster::correction(n, held.iter().map(|h| (h.0, h.1, &h.3)), &model.no_wave(), |w| model.wave_value(w));
        let waves: Vec<[f64; 3]> = cluster.waves.iter().zip(&correction).map(|(k, more)| std::array::from_fn(|i| k[i] + more[i])).collect();
        let still = held.iter().all(|h| h.2 == 0 && h.3.is_zero());
        let mut waves_sampled = vec![0.0_f64; tones.len()];
        for (x, y, tone, residual) in &held {
            let (q, k) = (model.wave_value(residual), [correction[*x], correction[*y]]);
            let left: [f64; 3] = std::array::from_fn(|i| q[i] - (k[0][i] - k[1][i]));
            waves_sampled[*tone] = waves_sampled[*tone].max(dot3(left, left).sqrt());
        }
        // The Hermitian blocks' adjoints (`Term::weight`): the block of the two components' conjugates at the negated beat.
        let conjugate = |s: Option<usize>| match s {
            None => Some(None),
            Some(i) => {
                let x = &self.scalars[i];
                self.scalars.iter().position(|y| (y.frequency, y.group, y.component, y.conjugate) == (x.frequency, x.group, x.component, !x.conjugate)).map(Some)
            }
        };
        let weights: Vec<[f64; 3]> = self
            .drafts
            .iter()
            .enumerate()
            .map(|(b, d)| {
                let adjoint = conjugate(d.left).zip(conjugate(d.right)).and_then(|(l, r)| {
                    let (l, r) = if l > r { (r, l) } else { (l, r) };
                    self.drafts.iter().position(|t| t.left == l && t.right == r && t.beat == d.beat.negated())
                });
                match adjoint.filter(|&t| self.drafts[t].used.0.iter().any(|used| *used)) {
                    Some(t) if t > b => [2.0; 3],
                    Some(t) if t < b => [0.0; 3],
                    Some(_) => [1.0, 2.0, 0.0],
                    None => [1.0; 3],
                }
            })
            .collect();
        // The dense terms (module documentation, # Dense terms): per block, flag and tone the generator's -- the Hermitian
        // value and the decay's of each entry together -- and, of the Hermitian flags that weigh in a form, the form's.
        let (mut generator, mut form, mut constant) = (Laid::default(), Laid::default(), vec![]);
        let (mut merged, mut forms) = (0, 0);
        for (b, d) in self.drafts.iter().enumerate() {
            let place = |e: usize| d.entries[e].0 * n + d.entries[e].1;
            let mut distinct: Vec<usize> = vec![];
            for &tone in &entry_tones[b] {
                if !distinct.contains(&tone) {
                    distinct.push(tone);
                }
            }
            for flag in 0..4 {
                let hermitian = (flag < 3 && d.used.0[flag]).then(|| &d.hermitian[flag]);
                let weighs = hermitian.is_some() && weights[b][flag] > 0.0;
                if hermitian.is_some() || d.used.1[flag] {
                    merged += d.entries.len();
                }
                if weighs {
                    forms += d.entries.len();
                }
                for &tone in &distinct {
                    let of = (0..d.entries.len()).filter(|&e| entry_tones[b][e] == tone);
                    let whole: Vec<(usize, C64, bool)> = of
                        .clone()
                        .map(|e| {
                            let (value, held) = hermitian.map_or((C64::ZERO, false), |h| (h[e].0, h[e].1.is_some()));
                            (place(e), value + d.decay[flag][e].0, held || !d.decay[flag][e].1.is_empty())
                        })
                        .collect();
                    generator.push(&mut constant, (b, flag, tone), 1.0, &whole);
                    if let Some(h) = hermitian.filter(|_| weighs) {
                        let own: Vec<(usize, C64, bool)> = of.map(|e| (place(e), h[e].0, h[e].1.is_some())).collect();
                        form.push(&mut constant, (b, flag, tone), weights[b][flag], &own);
                    }
                }
            }
        }
        // The near forms' places among a point's values.
        let mut links = vec![];
        for (b, d) in self.drafts.iter().enumerate() {
            for flag in (0..3).filter(|&flag| d.used.0[flag]) {
                for (e, (_, held)) in d.hermitian[flag].iter().enumerate() {
                    let Some((operand, conjugate)) = held else { continue };
                    let (key, place) = ((b, flag, entry_tones[b][e]), d.entries[e].0 * n + d.entries[e].1);
                    links.push(Link {
                        block: b,
                        flag,
                        place: d.entries[e],
                        operand: near.at(*operand),
                        conjugate: *conjugate,
                        generator: generator.place(key, place).expect("a near form's term"),
                        form: form.place(key, place),
                    });
                }
            }
        }
        // The decay's sums of products: a sum's adjoint, the same jumps exchanged, is its conjugate and is not made again.
        let (mut sums, mut products, mut raws) = (vec![], vec![], vec![]);
        let mut made: HashMap<Vec<(usize, usize)>, usize> = HashMap::new();
        for (b, d) in self.drafts.iter().enumerate() {
            for flag in (0..4).filter(|&flag| d.used.1[flag]) {
                for (e, (_, terms)) in d.decay[flag].iter().enumerate() {
                    if terms.is_empty() {
                        continue;
                    }
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
                    let place = generator.place((b, flag, entry_tones[b][e]), d.entries[e].0 * n + d.entries[e].1).expect("a decay's term");
                    sums.push((place, raw, conjugated, decay_factor(flag)));
                }
            }
        }
        // A node's counted work: the field components from their beams (plain and Doppler-weighted; with their gradients,
        // fourfold), each block's entries per flag -- in a form those of the Hermitian flags taken (`Term::weight`) --, the
        // frames' phases on the n² entries.
        let fields: usize = self.scalars.iter().map(|s| 2 * s.beams.len()).sum();
        let node = [(fields + merged + 2 * n * n) as f64, (4 * fields + forms + n * n) as f64, forms as f64];
        let statics = self.drafts.iter().map(|d| d.beat.is_zero()).collect();
        // The classes' members with their jumps as matrices (module documentation, # The jumps' matrices): one that
        // holds a near form among a point's values -- each near form laid onto its constant part, its partner's after
        // it --, any other among the plan's.
        let (len, jumping) = (near.len(), near.jumping().start);
        let (constants, offsets) = (&self.constants, &self.offsets);
        let (mut emitted, mut jumps_start, mut jump_links) = (vec![], vec![], vec![]);
        let classes = classes
            .into_iter()
            .map(|(landing, [wave_number, absorbed], members)| {
                let states = model.clusters[landing].states.len();
                let rows = (3 * states).div_ceil(LANES) * LANES;
                let members = members
                    .into_iter()
                    .map(|(source, scalar, doppler, phase, entries)| {
                        let entries: Vec<(usize, usize, usize, usize, usize)> = entries.into_iter().map(|(a, bp, sp, j, j2)| (a, bp, sp, near.at(j), near.at(j2))).collect();
                        let mut columns: Vec<usize> = entries.iter().map(|e| e.2).collect();
                        columns.sort();
                        columns.dedup();
                        let varies = entries.iter().any(|e| e.3 < len || e.4 < len);
                        let (matrix, values) = (2 * rows * columns.len(), if varies { &mut jumps_start } else { &mut emitted });
                        let at = values.len();
                        values.resize(at + if varies { 2 * matrix } else { matrix }, 0.0);
                        let folded = !matches!(model.sources[source].origin, Origin::Explicit { .. });
                        for &(a, bp, sp, first, second) in &entries {
                            let column = columns.iter().position(|&x| x == sp).expect("an entry's column");
                            let own = at + laid(a * states + bp, column, columns.len());
                            // A constant member's partner is nothing: a carrier's constant jump has none.
                            assert!(varies || !doppler || constants[second - len] == C64::ZERO, "a constant jump's Doppler partner");
                            for (place, slot) in [(first, own)].into_iter().chain(varies.then_some((second, own + matrix))) {
                                let constant = if place >= len {
                                    constants[place - len]
                                } else {
                                    jump_links.push(([slot, slot + LANES], place));
                                    place.checked_sub(jumping).and_then(|jump| offsets.get(jump)).copied().unwrap_or(C64::ZERO)
                                };
                                values[slot] += constant.re;
                                values[slot + LANES] += constant.im;
                            }
                        }
                        MemberPlan { scalar, doppler, phase, entries, folded, near: varies, columns, at }
                    })
                    .collect();
                ClassPlan { landing, wave_number, absorbed, members, states, rows }
            })
            .collect();
        let values = (products.len() + sums.len() + merged) as f64;
        Plan {
            raws,
            cluster: self.cluster,
            lights: self.lights.clone(),
            states: n,
            near,
            constants: self.constants,
            offsets: self.offsets,
            beams: self.beams,
            scalars: self.scalars,
            blocks: self.drafts.iter().map(|d| Block { left: d.left, right: d.right }).collect(),
            statics,
            padded: (n * n).div_ceil(LANES) * LANES,
            generator: generator.terms,
            form: form.terms,
            constant,
            generator_start: generator.start,
            form_start: form.start,
            links,
            sums,
            products,
            waves,
            still,
            sampled: tones.iter().zip(&waves_sampled).map(|(omega, wave)| [omega.abs(), *wave]).collect(),
            node,
            tones,
            classes,
            emitted,
            jumps_start,
            jump_links,
            values,
            fourth: None,
            id: PLANS.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        }
    }
}

impl Plan {
    fn new(model: &Model, cluster: usize, lights: &Lights) -> Plan {
        // The frequency whose fold on the cluster is the fourth order's under the lights: the model's, its light coherent.
        let fourth = model.fourth[cluster].filter(|&frequency| lights.0[frequency] == Lit::Coherent);
        let mut b = Builder { model, cluster, lights, near: NearLayout { fourth: fourth.map_or(0, |frequency| if Fourth::uniform(model, cluster, frequency).is_some() { 1 } else { 2 }), ..NearLayout::default() }, pair_index: HashMap::new(), jump_index: HashMap::new(), constants: vec![], offsets: vec![], beams: vec![], scalars: vec![], drafts: vec![] };
        b.constants();
        b.near();
        b.decays();
        let mut plan = b.finish();
        plan.fourth = fourth.and_then(|frequency| Fourth::new(model, &plan, frequency));
        plan
    }
    /// The constant part of a near form at `place` among a point's values: a jump entry's (`Plan::offsets`), zero of any
    /// other form.
    pub(crate) fn bare(&self, place: usize) -> C64 {
        place.checked_sub(self.near.jumping().start).and_then(|jump| self.offsets.get(jump)).copied().unwrap_or(C64::ZERO)
    }
    /// A block's two field components.
    pub(crate) fn factors(&self, block: usize) -> (Option<usize>, Option<usize>) {
        (self.blocks[block].left, self.blocks[block].right)
    }
    /// The Hermitian near forms into a point's values (module documentation, # Dense terms): each form -- `form(place)`, a
    /// point's or a table node's, or their derivative along a direction -- added where it lies among the generator's values
    /// (where given) and among the forms'.
    pub(crate) fn lay(&self, form: impl Fn(usize) -> C64, mut generator: Option<&mut [f64]>, forms: &mut [f64]) {
        for l in &self.links {
            let z = if l.conjugate { form(l.operand).conj() } else { form(l.operand) };
            if let Some(g) = generator.as_deref_mut() {
                g[l.generator[0]] += z.re;
                g[l.generator[1]] += z.im;
            }
            if let Some([re, im]) = l.form {
                forms[re] += z.re;
                forms[im] += z.im;
            }
        }
    }
    /// The decay into a point's values of the generator: the jumps' products conj(first) second summed (`sums`: their
    /// scratch), each sum with its flag's factor where it lies. `jump(place)` is a jump's value there: a near form with its
    /// constant part, or a constant.
    pub(crate) fn decay(&self, jump: impl Fn(usize) -> C64, sums: &mut Vec<C64>, generator: &mut [f64]) {
        sums.clear();
        sums.extend(self.raws.iter().map(|products| self.products[products.clone()].iter().map(|&(x, y)| jump(x).conj() * jump(y)).sum::<C64>()));
        for &([re, im], sum, conjugated, factor) in &self.sums {
            let z = factor * if conjugated { sums[sum].conj() } else { sums[sum] };
            generator[re] += z.re;
            generator[im] += z.im;
        }
    }
    /// The near jumps into a point's jumps (module documentation, # The jumps' matrices): each form -- `form(place)`, a
    /// point's or a table node's, or their derivative along a direction -- added where it lies in its member's matrix.
    pub(crate) fn lay_jumps(&self, form: impl Fn(usize) -> C64, jumps: &mut [f64]) {
        for &([re, im], place) in &self.jump_links {
            let z = form(place);
            jumps[re] += z.re;
            jumps[im] += z.im;
        }
    }
    /// A dense term's values, their real parts and their imaginary parts: the plan's constants, or a point's (`point`).
    fn values<'a>(&'a self, term: &Term, point: &'a [f64]) -> (&'a [f64], &'a [f64]) {
        (if term.near { point } else { &self.constant[..] })[term.at..term.at + 2 * term.len].split_at(term.len)
    }
    /// Per state of the cluster, the bound of what the constant Hermitian terms' plane waves add to the curvature of
    /// ⟨x|Ĥ_h|x⟩ on a state x held in the cluster's Doppler frames (reach.rs, # The kick's curvature; rad/µs/µm²), the
    /// beams' moduli at most `most` (per request beam). A term's entry x ← y is its value v times the product of its
    /// two field components, a sum over their beams of c_l c_r a_l a_r; in the frames each product turns along a
    /// displacement at q = ±k_l ± k_r − (K_x − K_y) (a conjugated component's wave vectors negated), and its second
    /// derivative along any direction is at most |q|² of it. In a form with a Hermitian ρ = |x⟩⟨x| the entry weighs
    /// |x_x||x_y| ≤ ½(|x_x|² + |x_y|²): half its bound to each of its two states' rows, by the form's own weights
    /// (`Term::weight`: an adjoint left out is its block taken twice) -- Gershgorin's row sums of the symmetrized
    /// moduli. Left out: the trap light's own form on a state with itself -- its curvature is the trap's, taken whole
    /// and with its sign where the trap is (potential.rs `Model::stiffer`) --; and an entry between a ground state and
    /// an excited one, an explicit drive's coupling, whose weight |x_e||x_g| is the drive's own answer and not a held
    /// state's (reach.rs: what the bound does not hold). What a near form or the first-order Doppler partner enters
    /// is not a constant: none of it is here.
    pub(crate) fn fringes(&self, model: &Model, most: &[f64]) -> Vec<f64> {
        let (n, ground) = (self.states, model.clusters[self.cluster].ground);
        // A component's beams: per beam its place among the plan's and whether it is conjugated, and c_l; of no component, 1.
        let side = |component: Option<usize>| -> Vec<(Option<(usize, bool)>, C64)> {
            component.map_or(vec![(None, C64::ONE)], |s| self.scalars[s].beams.iter().map(|&(at, c)| (Some((at, self.scalars[s].conjugate)), c)).collect())
        };
        // Per entry (x, y), tone and pair of beams: the coefficient of the pair's plane waves, every term's at that entry and
        // tone summed -- the blocks of one frequency's basis components are one field's, whose sum the basis does not change.
        let mut pairs: std::collections::BTreeMap<(usize, usize, usize, Option<(usize, bool)>, Option<(usize, bool)>), C64> = std::collections::BTreeMap::new();
        for term in self.form.iter().filter(|term| term.flag == PLAIN) {
            let block = &self.blocks[term.block];
            let (left, right) = (side(block.left), side(block.right));
            let trap = [block.left, block.right].iter().all(|s| s.is_some_and(|s| self.scalars[s].frequency == model.light.trap));
            let (re, im) = self.values(term, &self.form_start);
            for (i, (re, im)) in re.iter().zip(im).enumerate().filter(|(_, (re, im))| **re != 0.0 || **im != 0.0) {
                let (x, y) = ((term.first + i) / n, (term.first + i) % n);
                if (trap && x == y) || (x < ground) != (y < ground) {
                    continue;
                }
                let value = C64::new(*re, *im) * term.weight;
                for &(l, a) in &left {
                    for &(r, b) in &right {
                        *pairs.entry((x, y, term.tone, l, r)).or_insert(C64::ZERO) += value * a * b;
                    }
                }
            }
        }
        // A beam's wave vector as its product takes it, and its largest modulus.
        let wave = |beam: Option<(usize, bool)>| beam.map_or([0.0; 3], |(at, conjugate)| model.light.beams[self.beams[at]].wave_vector().map(|k| if conjugate { -k } else { k }));
        let modulus = |beam: Option<(usize, bool)>| beam.map_or(1.0, |(at, _)| most[self.beams[at]]);
        let mut rows = vec![0.0; n];
        for ((x, y, _, l, r), c) in pairs {
            let (k, q) = (wave(l), wave(r));
            let turned: [f64; 3] = std::array::from_fn(|j| k[j] + q[j] - (self.waves[x][j] - self.waves[y][j]));
            let half = 0.5 * c.abs() * modulus(l) * modulus(r) * dot3(turned, turned);
            rows[x] += half;
            rows[y] += half;
        }
        rows
    }
}

/// What a thread keeps from one use to the next (`Held`): a middle's numbers, a path's. A step's middle and its paths take
/// what the step before left -- vectors of the sizes they need already --, so the model's part of a step asks the heap for
/// nothing.
pub(crate) trait Spare: Default + 'static {
    fn kept() -> &'static std::thread::LocalKey<RefCell<Vec<Box<Self>>>>;
}

thread_local! {
    static SITES: RefCell<Vec<Box<Site>>> = const { RefCell::new(Vec::new()) };
    static RATES: RefCell<Vec<Box<Rates>>> = const { RefCell::new(Vec::new()) };
}

impl Spare for Site {
    fn kept() -> &'static std::thread::LocalKey<RefCell<Vec<Box<Site>>>> {
        &SITES
    }
}

impl Spare for Rates {
    fn kept() -> &'static std::thread::LocalKey<RefCell<Vec<Box<Rates>>>> {
        &RATES
    }
}

/// One of a thread's spare values, held: taken from the thread's kept ones (made where it has none) and given back when
/// dropped.
pub(crate) struct Held<T: Spare>(Option<Box<T>>);

impl<T: Spare> Held<T> {
    pub(crate) fn take() -> Held<T> {
        Held(Some(T::kept().try_with(|kept| kept.borrow_mut().pop()).ok().flatten().unwrap_or_default()))
    }
}

impl<T: Spare> Drop for Held<T> {
    fn drop(&mut self) {
        if let Some(held) = self.0.take() {
            // A thread that is ending keeps nothing: the value goes with it.
            let _ = T::kept().try_with(|kept| kept.borrow_mut().push(held));
        }
    }
}

impl<T: Spare> Deref for Held<T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.0.as_deref().expect("held until dropped")
    }
}

impl<T: Spare> DerefMut for Held<T> {
    fn deref_mut(&mut self) -> &mut T {
        self.0.as_deref_mut().expect("held until dropped")
    }
}

/// Where row `row` of column `column` lies among `count` columns laid for the lanes (module documentation, # The jumps'
/// matrices), from their start: its real part, its imaginary part `LANES` after it.
pub(crate) fn laid(row: usize, column: usize, count: usize) -> usize {
    ((row / LANES) * count + column) * 2 * LANES + row % LANES
}

/// A vector at a length, whatever it holds: its caller writes every element.
pub(crate) fn sized<T: Clone + Default>(v: &mut Vec<T>, len: usize) {
    if v.len() != len {
        v.clear();
        v.resize(len, T::default());
    }
}

/// A middle's numbers (`Middle`; a thread's spare, `Held`).
#[derive(Default)]
pub(crate) struct Site {
    /// Every request beam's amplitude at the point.
    pub(crate) beams: Vec<Amplitude>,
    /// Per beam of the plan: a_l and ∇a_l.
    amplitudes: Vec<[C64; 4]>,
    /// The near terms' values (module documentation, # Dense terms): the generator's, and the forms' with their slopes along
    /// `directions` (1/µm per unit of each; `slopes[f * forms + i]` the slope of value i along direction f).
    pub(crate) generator: Vec<f64>,
    pub(crate) forms: Vec<f64>,
    pub(crate) slopes: Vec<f64>,
    pub(crate) directions: Vec<[f64; 3]>,
    /// A table's coordinates at the point, and the point's place on its grid with the coordinates' counted work
    /// (tabulate.rs `Model::placed`: every read of the table at the point takes it); the jumps' products' sums
    /// (`formed`).
    pub(crate) coordinates: Vec<f64>,
    pub(crate) place: Option<(crate::table::Place, f64)>,
    sums: Vec<C64>,
    /// What a state's photons read (photon.rs `Model::photons_of`): the point's jumps' matrices (module documentation,
    /// # The jumps' matrices) with their slopes along `directions` (`jump_slopes[f * jumps + i]`), the plan's field
    /// components, and the sums in the making, laid as the matrices' columns are: a member's matrices on the state
    /// (`member`) and a class's u and ∇u (`class`).
    pub(crate) jumps: Vec<f64>,
    pub(crate) jump_slopes: Vec<f64>,
    pub(crate) fields: Fields,
    pub(crate) member: Vec<f64>,
    pub(crate) class: Vec<f64>,
    /// What a jump's draw reads (photon.rs `Model::emission_of`): the classes its state emits, their places among the
    /// plan's classes and their rates.
    pub(crate) emitting: Vec<usize>,
    pub(crate) rates: Vec<f64>,
    /// What the readout reads (readout.rs `Model::readout_of`): a table's responses at the point (tabulate.rs
    /// `Model::near_responses`), the carriers' field components (eliminate.rs `Model::responses_each`), and its channels'
    /// responses, one after another, each row-major on the excited states and the cluster's ground places.
    pub(crate) responses: Vec<f64>,
    pub(crate) weights: Vec<(C64, C64)>,
    pub(crate) channels: Vec<C64>,
    /// Per channel and ground place its excited amplitudes, and per class of co-resonant places, at its first place,
    /// their sum (readout.rs, # The readout).
    pub(crate) excited: Vec<C64>,
    /// The fourth order's (module documentation, # The fourth order): H⁽⁴⁾ less (i/2) the decay's change on the cluster's
    /// places, then H⁽⁴⁾'s gradient along the three lab axes; and its products' scratch.
    pub(crate) quartic: Vec<C64>,
    made: Vec<C64>,
    /// Its nodes' scratch, one for all its paths: a node's call fills what it reads.
    scratch: RefCell<Scratch>,
}

impl Site {
    /// The plan's beams' amplitudes with their gradients, from every beam's (`beams`).
    fn select(&mut self, plan: &Plan) {
        self.amplitudes.clear();
        self.amplitudes.extend(plan.beams.iter().map(|&l| &self.beams[l]).map(|a| [a.value, a.gradient[0], a.gradient[1], a.gradient[2]]));
    }
    /// The near terms' values from the near forms at the point (the Hermitian ones to first order): a Hermitian value its
    /// constant and its near form with its slopes, a decay's its constant and its jumps' products. Without near forms (None):
    /// the constants alone.
    fn formed(&mut self, plan: &Plan, near: Option<&NearValues>) {
        self.generator.clear();
        self.generator.extend_from_slice(&plan.generator_start);
        self.forms.clear();
        self.forms.extend_from_slice(&plan.form_start);
        self.directions.clear();
        self.slopes.clear();
        let Some(near) = near else { return };
        self.directions.extend(near.slopes());
        let (len, width) = (plan.near.len(), self.forms.len());
        self.slopes.resize(self.directions.len() * width, 0.0);
        plan.lay(|at| near.data[at], Some(&mut self.generator), &mut self.forms);
        for (f, slopes) in self.slopes.chunks_mut(width.max(1)).enumerate() {
            plan.lay(|at| near.data[(1 + f) * near.len + at], None, slopes);
        }
        plan.decay(|at| if at < len { near.data[at] + plan.bare(at) } else { plan.constants[at - len] }, &mut self.sums, &mut self.generator);
    }
    /// The point's jumps' matrices from the near forms there (module documentation, # The jumps' matrices), to the forms'
    /// order: their constant part and the near jumps, and to first order the jumps' slopes along the forms' directions.
    /// Without near forms (None): the constant part alone.
    pub(crate) fn jumped(&mut self, plan: &Plan, near: Option<&NearValues>) {
        self.jumps.clear();
        self.jumps.extend_from_slice(&plan.jumps_start);
        self.directions.clear();
        self.jump_slopes.clear();
        let Some(near) = near else { return };
        self.directions.extend(near.slopes());
        plan.lay_jumps(|at| near.data[at], &mut self.jumps);
        let width = self.jumps.len();
        self.jump_slopes.resize(self.directions.len() * width, 0.0);
        for (f, slopes) in self.jump_slopes.chunks_mut(width.max(1)).enumerate() {
            plan.lay_jumps(|at| near.data[(1 + f) * near.len + at], slopes);
        }
    }
}

/// A plan's field components at a point (`Model::fields`): per component its jet to an order, plain and weighted by its beams'
/// Doppler shifts (each `parts` numbers a component).
#[derive(Default)]
pub(crate) struct Fields {
    parts: usize,
    plain: Vec<C64>,
    doppler: Vec<C64>,
    /// The plan's beams' Doppler shifts at the momentum the components were weighted at.
    shifts: Vec<f64>,
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
/// documentation, # At a point, and along a path through it); and what the light a state scatters there gives its
/// momentum (photon.rs `Model::scattering_on`): the point is made once, whatever a step asks of it.
pub struct Middle<'a> {
    model: &'a Model,
    plan: Arc<Plan>,
    /// The position, of a middle the model made at one (`Model::middle`; none of a middle of given amplitudes,
    /// `Model::middle_of`), its time, and the table its near forms are read from, where one is.
    r: Option<[f64; 3]>,
    t: f64,
    table: Option<&'a NearTable>,
    /// Its numbers: the beams' amplitudes, the point's place on the table and the near terms' values at the point;
    /// and, once a state's scattering is asked there, the jumps' matrices and the photons' sums.
    point: RefCell<Held<Site>>,
    /// The complex multiply-adds that made it.
    pub(crate) work: f64,
}

impl Model {
    /// What a step's nodes sample of the clusters' generators along a path under a light set (`Plan::sampled`): per
    /// residual beat its rate |Ω| (rad/µs) and the largest wave vector |K| (1/µm) that turns with it.
    pub fn sampled(&self, lights: &Lights) -> Vec<[f64; 2]> {
        (0..self.clusters.len()).flat_map(|c| self.plan(c, lights).sampled.clone()).collect()
    }

    /// A cluster's plan under a light set, made once: the plans' lock is held to find its place or to give it one, and
    /// the plan is made in that place alone -- whoever asks for the same plan meanwhile waits for it, and two plans
    /// asked side by side are made side by side.
    pub(crate) fn plan(&self, cluster: usize, lights: &Lights) -> Arc<Plan> {
        let place = {
            let mut plans = self.plans.lock().expect("the plans' lock");
            match plans.iter().find(|of| of.0 == cluster && of.1 == *lights) {
                Some(found) => found.2.clone(),
                None => {
                    plans.push((cluster, lights.clone(), Arc::default()));
                    plans.last().expect("the place just given").2.clone()
                }
            }
        };
        place.get_or_init(|| Arc::new(Plan::new(self, cluster, lights))).clone()
    }

    /// The excited states the response of a carrier's basis coupling k from ground state b can reach on a cluster: those in a
    /// block (`Model::mixing`) with a state its drives folded at the atom couple b to.
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
    /// and the self-energy's v v† of each folded channel to a ground state outside the cluster and of each kept channel. A
    /// response's support is a union of blocks: the factorization of K + Σ − z and its solves leave every other entry exactly zero.
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
            for s in 0..GROUND {
                let outside = self.membership[s] != cluster;
                let reached: Vec<usize> = (0..EXCITED).filter(|&e| (outside && cc.folded.iter().any(|x| x[(GROUND + e, s)] != C64::ZERO)) || cc.explicit.iter().any(|x| x[(GROUND + e, s)] != C64::ZERO)).collect();
                for &e in &reached {
                    join(&mut block, reached[0], e);
                }
            }
        }
        block
    }

    /// A plan's near forms from the elimination's responses and the constant forms' energies (jets, [part][state]), into
    /// `out[part * len + entry]` (module documentation, # Near forms); and their complex multiply-adds (per part each pair's two
    /// sums and its partner's, each jump's two, each shift's two per ground state). The forms of the ground places
    /// `places` takes -- those the responses were made of (eliminate.rs `Model::eliminate`): a form that reads another
    /// place's response is left as `out` holds it.
    pub(crate) fn forms(&self, plan: &Plan, responses: &Responses, energies: &[Vec<f64>], out: &mut [C64], places: &dyn Fn(usize) -> bool) -> f64 {
        let cluster = &self.clusters[plan.cluster];
        let states = &cluster.states;
        let (near, len) = (&plan.near, plan.near.len());
        // The pairs and the jumps made, a part's.
        let (pairs, jumps) = (near.pairs.iter().filter(|pair| places(pair[4]) && places(pair[5])).count(), near.jumps.iter().filter(|jump| places(jump[4])).count());
        // A frequency's carrier; one that is none has no response at the atom.
        let carriers: Vec<Option<usize>> = (0..self.tones.len()).map(|w| self.carrier_of(w)).collect();
        let none = Mat::zeros(EXCITED, cluster.ground);
        for part in 0..energies.len() {
            let out = &mut out[part * len..(part + 1) * len];
            let (first, second) = (|c: usize, k: usize| &responses.first[c][k][part], |c: usize, k: usize| &responses.second[c][k][part]);
            // The fold's one contraction (eliminate.rs `shift`, `jump`) of the legs -- each its frequency's every folded
            // coupling and its response at the point --; a Doppler partner is the absorbed half with −T2 for T.
            for (i, &[wl, kl, wr, kr, x, y]) in near.pairs.iter().enumerate().filter(|(_, pair)| places(pair[4]) && places(pair[5])) {
                let (ul, ur) = (&self.tones[wl].coupled[0][kl], &self.tones[wr].coupled[0][kr]);
                let emitted = Leg { coupling: ul, state: states[x], response: carriers[wl].map_or(&none, |c| first(c, kl)), place: x };
                let absorbed = Leg { coupling: ur, state: states[y], response: carriers[wr].map_or(&none, |c| first(c, kr)), place: y };
                out[near.at(Operand::Pair(i))] = shift(emitted, absorbed);
                out[near.at(Operand::Doppler(i))] = -half(ul, states[x], carriers[wr].map_or(&none, |c| second(c, kr)), y);
            }
            for (i, &[c, k, a, b, y]) in near.jumps.iter().enumerate().filter(|(_, jump)| places(jump[4])) {
                out[near.at(Operand::Jump(i))] = jump(&self.emission[a], b, first(c, k), y);
                out[near.at(Operand::Jump2(i))] = jump(&self.emission[a], b, second(c, k), y);
            }
            let responding = near.responding().start;
            for (i, &[c, k, e, y]) in near.responses.iter().enumerate().filter(|(_, response)| places(response[3])) {
                out[responding + i] = first(c, k)[(e, y)];
                out[responding + near.responses.len() + i] = second(c, k)[(e, y)];
            }
        }
        // The fourth order's forms of a pair (module documentation, # The fourth order): X = T_k'†T_k and L = T_k'†D T_k on
        // x ← y, products of two responses' jets.
        let mut fourth = 0;
        if near.fourth > 0 {
            let np = energies.len();
            let rates: Vec<(usize, usize, C64)> = (0..EXCITED * EXCITED).map(|i| (i / EXCITED, i % EXCITED, self.decay[(i / EXCITED, i % EXCITED)])).filter(|r| r.2 != C64::ZERO).collect();
            for (i, &[wl, kl, wr, kr, x, y]) in near.pairs.iter().enumerate().filter(|(_, pair)| places(pair[4]) && places(pair[5])) {
                let (Some(cl), Some(cr)) = (carriers[wl], carriers[wr]) else { continue };
                if responses.first[cl].is_empty() || responses.first[cr].is_empty() {
                    continue;
                }
                let (mut metric, mut loss) = ([C64::ZERO; jet::WHOLE], [C64::ZERO; jet::WHOLE]);
                for &(e, f, rate) in &rates {
                    let left: [C64; jet::WHOLE] = std::array::from_fn(|p| if p < np { responses.first[cl][kl][p][(e, x)].conj() } else { C64::ZERO });
                    let right: [C64; jet::WHOLE] = std::array::from_fn(|p| if p < np { responses.first[cr][kr][p][(f, y)] } else { C64::ZERO });
                    if left[..np].iter().all(|z| *z == C64::ZERO) || right[..np].iter().all(|z| *z == C64::ZERO) {
                        continue;
                    }
                    let mut product = [C64::ZERO; jet::WHOLE];
                    jet::multiply(&left[..np], &right[..np], &mut product[..np]);
                    for p in 0..np {
                        loss[p] += product[p] * rate;
                        if e == f {
                            metric[p] += product[p];
                        }
                    }
                    fourth += 1;
                }
                for p in 0..np {
                    out[p * len + near.at(Operand::Metric(i))] = metric[p];
                    if near.fourth == 2 {
                        out[p * len + near.at(Operand::Loss(i))] = loss[p];
                    }
                }
            }
        }
        // The explicit states' van Vleck shift by the drives folded at the atom: each half its own drive's denominator there
        // against the other leg's folded coupling (far.rs `second_order`, where the bare drives' halves are),
        // ½ Σ_b (U^a_k[e,b] U^f_k'[f,b]*/(z_e − E_b − δ) + U^f_k[e,b] U^a_k'[f,b]*/(z_f − E_b − δ)), each explicit state at
        // the energy z the cluster's ground states read it at (model.rs `Model::read`, with the levels at the atom): the
        // two-photon distance of the Raman pair through it (far.rs, # A fold onto an explicit state).
        for (i, &[c, kl, kr, x, y]) in near.shifts.iter().enumerate() {
            let cc = &self.carriers[c];
            let folded = &self.tones[cc.frequency].coupled[0];
            let (e, f) = (states[x], states[y]);
            let mut shift = vec![C64::ZERO; energies.len()];
            let read = |place: usize| -> Vec<f64> { energies.iter().enumerate().map(|(part, h)| self.read(cluster, place, h, part == 0)).collect() };
            let read = [read(x), read(y)];
            for b in 0..GROUND {
                for (z, v) in [(&read[0], cc.basis[kr][(e, b)] * folded[kl][(f, b)].conj()), (&read[1], folded[kr][(e, b)] * cc.basis[kl][(f, b)].conj())] {
                    if v == C64::ZERO {
                        continue;
                    }
                    let mut gap: Vec<f64> = z.iter().zip(energies).map(|(z, h)| z - h[b]).collect();
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
        (energies.len() * ((3 * pairs + 2 * jumps) * EXCITED + 2 * GROUND * near.shifts.len())) as f64 + fourth as f64 * 3.0 * jet::product_work(energies.len())
    }

    /// A plan's near forms at a point from the exact elimination there, jets to `order`, each carrier's dressing coherent or not as
    /// `coherent` says (eliminate.rs).
    pub(crate) fn near_exact(&self, plan: &Plan, amplitudes: &[Amplitude], order: usize, coherent: &[bool]) -> NearValues {
        self.near_exact_on(plan, amplitudes, order, coherent, &|_| true)
    }

    /// `near_exact` of the cluster's ground places `places` takes: the responses from those places alone and the forms
    /// that read no other's (zero elsewhere) -- what a reader of a few places' forms asks for (the potential: the
    /// populated states' own shifts). A form made is the same number whichever places are taken with it.
    pub(crate) fn near_exact_on(&self, plan: &Plan, amplitudes: &[Amplitude], order: usize, coherent: &[bool], places: &dyn Fn(usize) -> bool) -> NearValues {
        let len = plan.near.len();
        let mut data = vec![C64::ZERO; jet::parts(order) * len];
        if len == 0 {
            return NearValues { len, order, data, directions: None, work: 0.0 };
        }
        let h = self.far_hamiltonian(amplitudes, &plan.lights, order, coherent);
        let responses = self.eliminate(plan.cluster, &h, amplitudes, &plan.lights, order, coherent, places);
        let energies: Vec<Vec<f64>> = h.iter().map(|m| (0..STATES).map(|s| m[(s, s)].re).collect()).collect();
        let work = self.forms(plan, &responses, &energies, &mut data, places);
        NearValues { len, order, data, directions: None, work: responses.work + self.far_work(&plan.lights, order, coherent) + work }
    }

    /// A plan's near forms at a point, the sections asked each to its order (0 or 1): from its table where one is given
    /// (tabulate.rs), else the exact elimination's to the largest order.
    pub(crate) fn near_values(&self, plan: &Plan, amplitudes: &[Amplitude], sections: &[(Range<usize>, usize)], table: Option<&NearTable>) -> NearValues {
        match table {
            Some(table) => self.near_tabulated(table, amplitudes, sections),
            None => self.near_exact(plan, amplitudes, sections.iter().map(|s| s.1).max().unwrap_or(0), &self.coherent_dressing()),
        }
    }

    /// The fourth order's matrices of a plan's cluster at a point (`Quartic`) into the thread's, and the counted work
    /// where they were made for it: a thread keeps its last point's -- of the same plan, the same table's grid and the
    /// same position `at` they are the same numbers, so a step's middle and its scattering make them once (None: made,
    /// and kept for no other). `point` holds the point's beams; `laid`: and its forms with their slopes (a middle's:
    /// `Site::formed`, tabulate.rs `Model::near_point`), which are read here and else laid here.
    pub(crate) fn quartic(&self, plan: &Plan, fourth: &Fourth, table: Option<&NearTable>, at: Option<[f64; 3]>, point: &mut Site, laid: bool) -> f64 {
        QUARTIC.with(|kept| {
            let mut q = kept.borrow_mut();
            let grid = table.map_or(0, |t| t.table.axes().iter().fold(1 + crate::table::Table::nodes(t.table.axes()) as u64, |h, a| h.rotate_left(13) ^ a.spacing().to_bits() ^ a.node(0).to_bits().rotate_left(29)));
            let key = at.map(|r| (plan.id, grid, r.map(f64::to_bits)));
            if key.is_some() && q.key == key {
                return 0.0;
            }
            self.quartic_made(plan, fourth, table, None, point, laid, &mut q);
            q.key = key;
            q.work
        })
    }

    /// The thread's fourth-order matrices (`quartic`), read.
    pub(crate) fn quartic_read<R>(read: impl FnOnce(&Quartic) -> R) -> R {
        QUARTIC.with(|kept| read(&kept.borrow()))
    }

    /// `quartic`'s matrices. N's Hermitian part and Y are the point's own laid forms summed with the frequency's plain
    /// components by the nodes' kernel (module documentation, # Dense terms): of the forms' terms -- which hold one of a
    /// term and its adjoint (`Term::weight`) -- the shifts' and, where the other is left out, their adjoints; the Doppler
    /// partners' where the absorbed component takes the shifts, and the adjoints of those where the emitted one does;
    /// each with its slopes along the point's directions and the components' gradients. X (and L where it is not Γ X)
    /// are their pair forms' sums: a table's interpolant with its slopes, the forms `given` (those the point's were
    /// laid from), or the exact elimination's.
    fn quartic_made(&self, plan: &Plan, fourth: &Fourth, table: Option<&NearTable>, given: Option<&NearValues>, point: &mut Site, laid: bool, q: &mut Quartic) {
        let (m, n, pairs) = (fourth.m, fourth.n, plan.near.pairs.len());
        let (size, padded) = (n * n, plan.padded);
        let Quartic { data, work, values, slopes, sums, laid: lanes, .. } = q;
        *work = 0.0;
        // The exact elimination's forms where no table holds them and none are given.
        let made = (table.is_none() && given.is_none()).then(|| self.near_exact(plan, &point.beams, 1, &self.coherent_dressing()));
        let exact = given.or(made.as_ref());
        if !laid {
            point.select(plan);
            *work += match (table, exact) {
                (Some(table), _) => self.near_point(table, point) + plan.values,
                (None, Some(near)) => {
                    point.formed(plan, Some(near));
                    near.work + plan.values
                }
                (None, None) => unreachable!("the exact elimination's forms are made where no table is"),
            };
        }
        let dn = point.directions.len();
        let parts = 4 + dn;
        // The plan's plain components at the point with their gradients.
        let mut scalars = [[C64::ZERO; 4]; 64];
        assert!(plan.scalars.len() <= scalars.len(), "a plan's field components");
        for (z, scalar) in scalars.iter_mut().zip(&plan.scalars) {
            for &(at, c) in &scalar.beams {
                for part in 0..4 {
                    z[part] += (if scalar.conjugate { point.amplitudes[at][part].conj() } else { point.amplitudes[at][part] }) * c;
                }
            }
        }
        let own = |s: Option<usize>| s.is_some_and(|s| plan.scalars[s].frequency == fourth.frequency);
        // Four sums, each its value, its slopes' along each direction and the components' gradients' on it, real parts
        // and imaginary parts on the cluster's places: the shifts whose adjoints the forms hold not, and hold; the
        // partners of an absorbed component's shifts, and of an emitted one's.
        lanes.clear();
        lanes.resize(4 * parts * 2 * padded, 0.0);
        let width = point.forms.len();
        let mut summed = 0;
        for term in plan.form.iter().filter(|term| term.near && plan.tones[term.tone] == 0.0) {
            let block = &plan.blocks[term.block];
            if !own(block.left) || !own(block.right) {
                continue;
            }
            let to = match term.flag {
                PLAIN => if term.weight == 2.0 { 1 } else { 0 },
                flag => {
                    let shifted = if flag == RIGHT { block.right } else { block.left };
                    if shifted.is_some_and(|s| plan.scalars[s].conjugate) { 3 } else { 2 }
                }
            };
            let (l, r) = (side(&scalars, block.left), side(&scalars, block.right));
            let c: [C64; 4] = std::array::from_fn(|part| if part == 0 { l[0] * r[0] } else { l[part] * r[0] + l[0] * r[part] });
            let span = term.first..term.first + term.len;
            let lane = |part: usize| (to * parts + part) * 2 * padded;
            let (real, imaginary) = plan.values(term, &point.forms);
            // A shift's constant part (the bare folds' of the same components) is not the fold's at the atom.
            let start = (term.flag == PLAIN).then(|| plan.form_start[term.at..term.at + 2 * term.len].split_at(term.len));
            let mut add = |part: usize, c: C64, real: &[f64], imaginary: &[f64]| {
                let (re, im) = lanes[lane(part)..lane(part) + 2 * padded].split_at_mut(padded);
                axpy(c, real, imaginary, &mut re[span.clone()], &mut im[span.clone()]);
            };
            for (part, c) in [(0, c[0]), (1 + dn, c[1]), (2 + dn, c[2]), (3 + dn, c[3])] {
                add(part, c, real, imaginary);
                if let Some((real, imaginary)) = start {
                    add(part, -c, real, imaginary);
                }
            }
            for f in 0..dn {
                let (real, imaginary) = point.slopes[f * width + term.at..f * width + term.at + 2 * term.len].split_at(term.len);
                add(1 + f, c[0], real, imaginary);
            }
            summed += term.len * (4 + dn + if start.is_some() { 4 } else { 0 });
        }
        // X, and L where it is not Γ X: Σ e_k'* e_k of the pairs' forms, as the shifts' sums are parted.
        let forms = plan.near.fourth;
        let (first, count) = (2 * pairs, forms * pairs);
        sized(values, 2 * count);
        sized(slopes, dn * 2 * count);
        match (table, exact) {
            (Some(table), _) => {
                table.table.read(&point.place.expect("a site placed on its table").0, 2 * first, None, values, Some(slopes));
                *work += (4usize.pow(dn as u32) * (1 + dn) * 2 * count) as f64 / crate::decide::COMPLEX;
            }
            (None, Some(near)) => {
                for e in 0..count {
                    (values[2 * e], values[2 * e + 1]) = (near.data[first + e].re, near.data[first + e].im);
                    for f in 0..dn {
                        let z = near.data[(1 + f) * near.len + first + e];
                        (slopes[(f * count + e) * 2], slopes[(f * count + e) * 2 + 1]) = (z.re, z.im);
                    }
                }
            }
            (None, None) => unreachable!("the exact elimination's forms are made where no table is"),
        }
        let value = |e: usize| C64::new(values[2 * e], values[2 * e + 1]);
        let slope = |f: usize, e: usize| C64::new(slopes[(f * count + e) * 2], slopes[(f * count + e) * 2 + 1]);
        let mut field = [[C64::ZERO; 4]; 8];
        for (e, beams) in field.iter_mut().zip(&fourth.fields) {
            for &(l, c) in beams {
                let a = &point.beams[l];
                e[0] += a.value * c;
                for i in 0..3 {
                    e[1 + i] += a.gradient[i] * c;
                }
            }
        }
        sums.clear();
        sums.resize(2 * parts * size, C64::ZERO);
        let mut of = (usize::MAX, [C64::ZERO; 4]);
        for &[i, kk, xy] in &fourth.entries {
            if of.0 != kk {
                let (l, r) = (&field[kk / m], &field[kk % m]);
                of = (kk, std::array::from_fn(|part| if part == 0 { l[0].conj() * r[0] } else { l[part].conj() * r[0] + l[0].conj() * r[part] }));
            }
            let c = &of.1;
            for form in 0..forms {
                let (e, to) = (form * pairs + i, form * parts * size + xy);
                let v = value(e);
                sums[to] += c[0] * v;
                for f in 0..dn {
                    sums[to + (1 + f) * size] += c[0] * slope(f, e);
                }
                for a in 0..3 {
                    sums[to + (1 + dn + a) * size] += c[1 + a] * v;
                }
            }
        }
        // To the lab axes: a part's value, or along an axis the components' gradients' sum and the slopes' by their
        // directions.
        let directions = &point.directions;
        let turned = |of: &dyn Fn(usize) -> C64, part: usize| -> C64 {
            match part {
                0 => of(0),
                _ => (0..dn).fold(of(dn + part), |g, f| g + of(1 + f) * directions[f][part - 1]),
            }
        };
        let lane = |sum: usize, part: usize, place: usize| -> C64 {
            let at = (sum * parts + part) * 2 * padded + place;
            C64::new(lanes[at], lanes[at + padded])
        };
        // N = S − (i/2) L, Y = 2 D, X.
        sized(data, 12 * size);
        for part in 0..4 {
            for x in 0..n {
                for y in 0..n {
                    let (xy, yx) = (x * n + y, y * n + x);
                    let shift = turned(&|p| lane(0, p, xy) + lane(1, p, xy) + lane(1, p, yx).conj(), part);
                    let partner = turned(&|p| lane(2, p, xy) + lane(3, p, yx).conj(), part);
                    let metric = turned(&|p| sums[p * size + xy], part);
                    let loss = fourth.uniform.map_or_else(|| turned(&|p| sums[(parts + p) * size + xy], part), |rate| metric * rate);
                    data[part * size + xy] = shift - loss * C64::new(0.0, 0.5);
                    data[(4 + part) * size + xy] = partner * 2.0;
                    data[(8 + part) * size + xy] = metric;
                }
            }
        }
        *work += (summed + fourth.entries.len() * forms * parts + 12 * size * (1 + dn)) as f64;
    }

    /// The fourth order's part of a middle (module documentation, # The fourth order) into the point's `quartic`, from
    /// the point's matrices (`quartic`) and its generator's values: H⁽⁴⁾ with its gradient, and the decay's change. With
    /// the complex multiply-adds that made it (the matrices' where this point made them). The matrices are the position
    /// `at`'s with the table's forms or the exact elimination's (`quartic`), or made of the forms `given`, the ones the
    /// point's were laid from.
    fn fourth_site(&self, plan: &Plan, fourth: &Fourth, table: Option<&NearTable>, given: Option<&NearValues>, at: Option<[f64; 3]>, point: &mut Site) -> f64 {
        let (n, size) = (fourth.n, fourth.n * fourth.n);
        let matrices = match given {
            None => self.quartic(plan, fourth, table, at, point, true),
            Some(_) => QUARTIC.with(|kept| {
                let mut q = kept.borrow_mut();
                self.quartic_made(plan, fourth, None, given, point, true, &mut q);
                q.key = None;
                q.work
            }),
        };
        let Site { amplitudes, generator, quartic, made, .. } = point;
        sized(quartic, 4 * size);
        made.clear();
        made.resize(17 * size, C64::ZERO);
        Model::quartic_read(|q| {
            let (near, y, x) = (&q.data[..4 * size], &q.data[4 * size..8 * size], &q.data[8 * size..]);
            let part = |of: &[C64], p: usize| -> std::ops::Range<usize> { let _ = of; p * size..(p + 1) * size };
            let (raw, rest) = made.split_at_mut(4 * size);
            let (left, rest) = rest.split_at_mut(4 * size);
            // −Y + ½X (the commutator's half where it is anything), value and gradient.
            let half = if fourth.uniform.is_some() { 0.0 } else { 0.5 };
            for (i, to) in left.iter_mut().enumerate() {
                *to = x[i] * half - y[i];
            }
            let mut products = 0;
            for p in 0..4 {
                let out = &mut raw[part(near, p)];
                product_onto(out, &left[part(near, p)], &near[..size], n, C64::ONE);
                products += 1;
                if p > 0 {
                    product_onto(out, &left[..size], &near[part(near, p)], n, C64::ONE);
                    products += 1;
                }
                if fourth.uniform.is_none() {
                    product_onto(out, &near[part(near, p)], &x[..size], n, C64::real(-0.5));
                    products += 1;
                    if p > 0 {
                        product_onto(out, &near[..size], &x[part(near, p)], n, C64::real(-0.5));
                        products += 1;
                    }
                }
                for a in 0..n {
                    for b in 0..n {
                        quartic[p * size + a * n + b] = (raw[p * size + a * n + b] + raw[p * size + b * n + a].conj()) * 0.5;
                    }
                }
            }
            // The generator's decay at the point by its flags, read off its dense terms at rest: Λ = i(P − P†) of the
            // plain terms' sum P; Λ_R = −i(G_R − G_L†) of the sums of the terms whose absorbed component takes the Doppler
            // shifts, G_R, and whose emitted one does, G_L (module documentation, # The fourth order); Λ_B from the
            // terms of both, −(i/2)Λ_B. A term that beats in the frames is left out.
            let (sums, rest) = rest.split_at_mut(4 * size);
            let mut scalars = [C64::ZERO; 64];
            for (z, scalar) in scalars.iter_mut().zip(&plan.scalars) {
                *z = scalar.beams.iter().map(|&(at, c)| (if scalar.conjugate { amplitudes[at][0].conj() } else { amplitudes[at][0] }) * c).sum();
            }
            assert!(plan.scalars.len() <= scalars.len(), "a plan's field components");
            let side = |scalar: Option<usize>| scalar.map_or(C64::ONE, |s| scalars[s]);
            let mut summed = 0;
            for term in plan.generator.iter().filter(|term| plan.tones[term.tone] == 0.0) {
                let block = &plan.blocks[term.block];
                let to = match term.flag {
                    PLAIN => 0,
                    BOTH => 2,
                    flag => {
                        let shifted = if flag == RIGHT { block.right } else { block.left };
                        if shifted.is_some_and(|s| plan.scalars[s].conjugate) { 3 } else { 1 }
                    }
                };
                let (c, (real, imaginary)) = (side(block.left) * side(block.right), plan.values(term, generator));
                for (k, place) in (term.first..(term.first + term.len).min(size)).enumerate() {
                    sums[to * size + place] += c * C64::new(real[k], imaginary[k]);
                }
                summed += term.len;
            }
            let (bend, rest) = rest.split_at_mut(size);
            let (decays, turned) = rest.split_at_mut(3 * size);
            // decays: Λ, Λ_R, Λ_B; bend: B = 1 − ½X.
            for a in 0..n {
                for b in 0..n {
                    let at = a * n + b;
                    decays[at] = (sums[at] - sums[b * n + a].conj()) * C64::I;
                    decays[size + at] = (sums[size + at] - sums[3 * size + b * n + a].conj()) * C64::new(0.0, -1.0);
                    decays[2 * size + at] = sums[2 * size + at] * C64::new(0.0, 2.0);
                    bend[at] = x[at] * -0.5 + if a == b { C64::ONE } else { C64::ZERO };
                }
            }
            // Σ L̃†L̃ = B†(Λ B + Λ_R N) + N†(Λ_R†B + Λ_B N), into the sums' place.
            let (first, second) = sums.split_at_mut(size);
            let (second, changed) = second.split_at_mut(size);
            let changed = &mut changed[..size];
            for z in first.iter_mut().chain(second.iter_mut()).chain(changed.iter_mut()) {
                *z = C64::ZERO;
            }
            let adjoint = |of: &[C64], out: &mut [C64]| {
                for a in 0..n {
                    for b in 0..n {
                        out[a * n + b] = of[b * n + a].conj();
                    }
                }
            };
            product_onto(first, &decays[..size], bend, n, C64::ONE);
            product_onto(first, &decays[size..2 * size], &near[..size], n, C64::ONE);
            adjoint(&decays[size..2 * size], &mut turned[..size]);
            product_onto(second, &turned[..size], bend, n, C64::ONE);
            product_onto(second, &decays[2 * size..], &near[..size], n, C64::ONE);
            adjoint(bend, &mut turned[..size]);
            product_onto(changed, &turned[..size], first, n, C64::ONE);
            adjoint(&near[..size], &mut turned[..size]);
            product_onto(changed, &turned[..size], second, n, C64::ONE);
            for at in 0..size {
                quartic[at] -= (changed[at] - decays[at]) * C64::new(0.0, 0.5);
            }
            matrices + ((products + 6) * n * size + summed + 8 * size) as f64
        })
    }

    /// A plan's field components at a point, jets to `order`, plain and weighted by their beams' Doppler shifts at a momentum.
    pub(crate) fn fields(&self, plan: &Plan, amplitudes: &[Amplitude], kappa: [f64; 3], order: usize) -> Fields {
        let mut out = Fields::default();
        self.fields_into(plan, amplitudes, kappa, order, &mut out);
        out
    }

    /// The same into a caller's components, kept from one use to the next.
    pub(crate) fn fields_into(&self, plan: &Plan, amplitudes: &[Amplitude], kappa: [f64; 3], order: usize, out: &mut Fields) {
        let parts = jet::parts(order);
        out.parts = parts;
        for values in [&mut out.plain, &mut out.doppler] {
            values.clear();
            values.resize(parts * plan.scalars.len(), C64::ZERO);
        }
        // Each beam of the plan once: its Doppler shift, and its jet's parts (to first order without its Hessian's).
        out.shifts.clear();
        out.shifts.extend(plan.beams.iter().map(|&l| self.doppler_shift(l, kappa)));
        for (s, (plain, doppler)) in plan.scalars.iter().zip(out.plain.chunks_mut(parts).zip(out.doppler.chunks_mut(parts))) {
            for &(at, c) in &s.beams {
                let (a, shift) = (&amplitudes[plan.beams[at]], out.shifts[at]);
                let whole = if parts > 4 { jet::whole(a) } else { [C64::ZERO; jet::WHOLE] };
                for p in 0..parts {
                    let part = match p {
                        0 => a.value,
                        1..=3 => a.gradient[p - 1],
                        _ => whole[p],
                    };
                    let z = (if s.conjugate { part.conj() } else { part }) * c;
                    plain[p] += z;
                    doppler[p] += z * shift;
                }
            }
        }
    }

    /// A beam's first-order Doppler shift w_l = (ħ/m)(k_l·κ + k_l²/2) at a momentum (eliminate.rs, # First-order Doppler).
    pub(crate) fn doppler_shift(&self, beam: usize, kappa: [f64; 3]) -> f64 {
        let k = self.light.beams[beam].wave_vector();
        self.hbar_over_mass * (dot3(k, kappa) + 0.5 * dot3(k, k))
    }

    /// A site's point (`Site`): every beam's amplitude at r with its gradient -- no part of a middle or of a state's
    /// photons reads a second derivative -- and, where a plan's near forms are a table's, the point's place on it
    /// (tabulate.rs `Model::placed`). What is read of the model at the point is read on a site that holds it.
    pub(crate) fn pointed(&self, plan: &Plan, table: Option<&NearTable>, r: [f64; 3], site: &mut Site) {
        self.light.amplitudes_into(r, 1, |_| true, &mut site.beams);
        if let Some(table) = table.filter(|_| plan.near.len() > 0) {
            self.placed(table, site);
        }
    }

    /// A cluster's generator at a position and time under a plan (`Middle`), the near forms from a table where one is given.
    pub(crate) fn middle<'a>(&'a self, plan: &Arc<Plan>, r: [f64; 3], t: f64, table: Option<&'a NearTable>) -> Middle<'a> {
        let mut point = Held::<Site>::take();
        self.pointed(plan, table, r, &mut point);
        point.select(plan);
        let work = match table {
            _ if plan.near.len() == 0 => {
                point.formed(plan, None);
                0.0
            }
            Some(table) => self.near_point(table, &mut point) + plan.values,
            None => {
                let near = self.near_exact(plan, &point.beams, 1, &self.coherent_dressing());
                point.formed(plan, Some(&near));
                near.work + plan.values
            }
        };
        let work = work + plan.fourth.as_ref().map_or(0.0, |fourth| self.fourth_site(plan, fourth, table, None, Some(r), &mut point));
        Middle { model: self, plan: plan.clone(), r: Some(r), t, table, point: RefCell::new(point), work }
    }

    /// The same from every beam's amplitude and the near forms there (the Hermitian ones to first order; `Site::formed`),
    /// the fourth order's part made of the same forms. Without near forms (None): the terms that hold none, and no
    /// fourth order. Of forms to order 0 the generator is the same, bit for bit -- a form's value is the same number
    /// at every order --, and the force's forms are without their values' slopes: what a reader of the generator
    /// alone asks for (tabulate.rs `Model::table_reads`).
    pub(crate) fn middle_of(&self, plan: &Arc<Plan>, t: f64, amplitudes: &[Amplitude], near: Option<&NearValues>) -> Middle<'_> {
        let mut point = Held::<Site>::take();
        point.beams.clear();
        point.beams.extend_from_slice(amplitudes);
        point.select(plan);
        point.formed(plan, near);
        // A thread's spare point holds its last middle's: the fourth order's part is this point's, or none.
        point.quartic.clear();
        let mut work = near.map_or(0.0, |n| n.work + plan.values);
        if let (Some(fourth), Some(_)) = (plan.fourth.as_ref(), near) {
            work += self.fourth_site(plan, fourth, None, near, None, &mut point);
        }
        Middle { model: self, plan: plan.clone(), r: None, t, table: None, point: RefCell::new(point), work }
    }

    /// The one generator assembly (module documentation): a cluster's generator at a position and a time under a light set, the
    /// exact elimination at the point.
    pub fn assemble(&self, cluster: usize, r: [f64; 3], t: f64, lights: &Lights) -> Middle<'_> {
        self.middle(&self.plan(cluster, lights), r, t, None)
    }
}

/// The momentum's part of a cluster's generator along a straight path through its middle (`Middle::path`): the momentum
/// the generator is taken at (1/µm); per beam of the plan its Doppler shift w_l at it (rad/µs) and its path rate k_l·v
/// at the path's velocity (rad/µs); per state of the cluster its Doppler frame's rate K_s·v, and whether any frame
/// turns.
pub struct Path {
    kappa: [f64; 3],
    rates: Held<Rates>,
    turning: bool,
}

/// A path's numbers (`Path`; a thread's spare, `Held`).
#[derive(Default)]
pub(crate) struct Rates {
    shifts: Vec<f64>,
    path: Vec<f64>,
    gauge: Vec<f64>,
}

/// A node's scratch (`Site::scratch`): the beams' amplitudes turned along the path, the plan's field components there (each
/// with its gradient, plain and weighted by its beams' Doppler shifts), the tones' phases, the Doppler frames' turns; a ρ laid
/// out as the entries are and a generator's sum -- each the real parts on the cluster's places (`Plan::padded`) and then the
/// imaginary parts --; and per tone its three sums.
#[derive(Default)]
struct Scratch {
    turned: Vec<[C64; 4]>,
    plain: Vec<[C64; 4]>,
    doppler: Vec<[C64; 4]>,
    /// The scalars' derivatives along the path, where they are asked (`along`).
    plain_slope: Vec<[C64; 4]>,
    doppler_slope: Vec<[C64; 4]>,
    /// The gradient's sums, per axis its real and its imaginary parts, then their slopes' (`gradient`).
    gradients: Vec<f64>,
    phases: Vec<C64>,
    turns: Vec<C64>,
    weights: Vec<f64>,
    sum: Vec<f64>,
    sums: Vec<[f64; 3]>,
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
        let mut rates = Held::<Rates>::take();
        let Rates { shifts, path, gauge } = &mut *rates;
        shifts.clear();
        shifts.extend(plan.beams.iter().map(|&l| model.doppler_shift(l, kappa)));
        path.clear();
        path.extend(plan.beams.iter().map(|&l| dot3(model.light.beams[l].wave_vector(), velocity)));
        gauge.clear();
        gauge.extend(plan.waves.iter().map(|k| dot3(*k, velocity)));
        Path { kappa, turning: gauge.iter().any(|rate| *rate != 0.0), rates }
    }

    fn gauge(&self, path: &Path, rates: &mut [f64]) {
        rates.copy_from_slice(&path.rates.gauge);
    }

    /// Module documentation, # At a point, and along a path through it; # Dense terms.
    fn generator(&self, path: &Path, s: f64, h: &mut Mat) {
        let (plan, n, point) = (&*self.plan, self.plan.states, self.point.borrow());
        let mut scratch = point.scratch.borrow_mut();
        self.along(&point.amplitudes, path, Some(s), 1, false, &mut scratch);
        let Scratch { plain, doppler, phases, turns, sum, .. } = &mut *scratch;
        sum.clear();
        sum.resize(2 * plan.padded, 0.0);
        let (re, im) = sum.split_at_mut(plan.padded);
        // A block's products of its two components, per flag: made once for its terms.
        let mut of = (usize::MAX, [C64::ZERO; 4]);
        for term in &plan.generator {
            if of.0 != term.block {
                let block = &plan.blocks[term.block];
                let (l, lw, r, rw) = (side(plain, block.left)[0], side(doppler, block.left)[0], side(plain, block.right)[0], side(doppler, block.right)[0]);
                of = (term.block, [l * r, l * rw, lw * r, lw * rw]);
            }
            let (real, imaginary) = plan.values(term, &point.generator);
            let span = term.first..term.first + term.len;
            axpy(of.1[term.flag] * phases[term.tone], real, imaginary, &mut re[span.clone()], &mut im[span]);
        }
        for (z, (re, im)) in h.data.iter_mut().zip(re.iter().zip(im.iter())) {
            *z = C64::new(*re, *im);
        }
        if path.turning {
            for x in 0..n {
                let turn = turns[x];
                for y in 0..n {
                    h.data[x * n + y] *= turn;
                    h.data[y * n + x] *= turn.conj();
                }
                h.data[x * n + x] += C64::real(path.rates.gauge[x]);
            }
        }
        // The fourth order's, held along the path in the cluster's Doppler frames.
        if plan.fourth.is_some() && point.quartic.len() >= n * n {
            for (z, more) in h.data.iter_mut().zip(&point.quartic[..n * n]) {
                *z += *more;
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

    /// photon.rs `Model::scattering_on` this middle's own site, at the path's momentum and the middle's time.
    fn scattering(&self, path: &Path, psi: &[C64]) -> Scattering {
        self.model.scattering_on(&self.plan, self.table, self.r, path.kappa, self.t, psi, &mut self.point.borrow_mut(), true).0
    }

    /// Ĝ_a(s) as matrices: the forms' own terms (`contract`), each where it lies -- a flag's values times the gradient
    /// of its components' product, and its slopes along the point's directions times the product, at its tone's phase
    /// and in its states' Doppler frames --, then the Hermitian part of their weighted sum (of a block and its adjoint the
    /// form takes one twice: its Hermitian part is the pair's), and the fourth order's gradient. A term onto the three
    /// axes' sums at once, their real and imaginary parts apart (`linalg::axpy`). Its slope along the path, where asked,
    /// in the same pass: along the path every factor of a term is a plane wave's or a tone's phase e^{iωs} or a sum of
    /// them, whose derivative is iω times each (`along`), a block's product by the product rule, the frames' turns
    /// e^{−iK_x·v s} theirs; the fourth order is held along the path.
    fn gradient(&self, path: &Path, s: f64, g: &mut [Vec<C64>; 3], slope: Option<&mut [Vec<C64>; 3]>) {
        let (plan, n, point) = (&*self.plan, self.plan.states, self.point.borrow());
        let mut scratch = point.scratch.borrow_mut();
        let sloped = slope.is_some();
        self.along(&point.amplitudes, path, Some(s), 4, sloped, &mut scratch);
        let Scratch { plain, doppler, plain_slope, doppler_slope, gradients, phases, turns, .. } = &mut *scratch;
        let padded = plan.padded;
        gradients.clear();
        gradients.resize(if sloped { 12 } else { 6 } * padded, 0.0);
        let (sums, slopes) = gradients.split_at_mut(6 * padded);
        let width = point.forms.len();
        // A block's products of its two components with their gradients per Hermitian flag, and their slopes: made once
        // for its terms.
        let product = |l: &[C64; 4], r: &[C64; 4]| -> [C64; 4] { std::array::from_fn(|part| if part == 0 { l[0] * r[0] } else { l[part] * r[0] + l[0] * r[part] }) };
        let none = [C64::ZERO; 4];
        let mut of = (usize::MAX, [[C64::ZERO; 4]; 3], [[C64::ZERO; 4]; 3]);
        for term in &plan.form {
            if of.0 != term.block {
                let block = &plan.blocks[term.block];
                let (l, lw, r, rw) = (side(plain, block.left), side(doppler, block.left), side(plain, block.right), side(doppler, block.right));
                let mut turning = [[C64::ZERO; 4]; 3];
                if sloped {
                    let along = |fields: &[[C64; 4]], scalar: Option<usize>| scalar.map_or(none, |s| fields[s]);
                    let (l1, lw1, r1, rw1) = (along(plain_slope, block.left), along(doppler_slope, block.left), along(plain_slope, block.right), along(doppler_slope, block.right));
                    let rule = |l: &[C64; 4], l1: &[C64; 4], r: &[C64; 4], r1: &[C64; 4]| -> [C64; 4] {
                        let (a, b) = (product(l1, r), product(l, r1));
                        std::array::from_fn(|part| a[part] + b[part])
                    };
                    turning = [rule(&l, &l1, &r, &r1), rule(&l, &l1, &rw, &rw1), rule(&lw, &lw1, &r, &r1)];
                }
                of = (term.block, [product(&l, &r), product(&l, &rw), product(&lw, &r)], turning);
            }
            let (real, imaginary) = plan.values(term, &point.forms);
            let (c, phase) = (of.1[term.flag], phases[term.tone] * term.weight);
            let span = term.first..term.first + term.len;
            let onto = |sums: &mut [f64], a: usize, z: C64, real: &[f64], imaginary: &[f64]| {
                let (re, im) = sums[2 * a * padded..(2 * a + 2) * padded].split_at_mut(padded);
                axpy(z, real, imaginary, &mut re[span.clone()], &mut im[span.clone()]);
            };
            // d/ds of the tone's phase e^{iΩ(t + s)}: iΩ times it.
            let turned = phase * C64::new(0.0, plan.tones[term.tone]);
            for a in 0..3 {
                onto(sums, a, c[1 + a] * phase, real, imaginary);
                if sloped {
                    onto(slopes, a, of.2[term.flag][1 + a] * phase + c[1 + a] * turned, real, imaginary);
                }
            }
            for (f, direction) in point.directions.iter().enumerate().filter(|_| term.near) {
                let (real, imaginary) = point.slopes[f * width + term.at..f * width + term.at + 2 * term.len].split_at(term.len);
                let (value, along) = (c[0] * phase, of.2[term.flag][0] * phase + c[0] * turned);
                for a in (0..3).filter(|&a| direction[a] != 0.0) {
                    onto(sums, a, value * direction[a], real, imaginary);
                    if sloped {
                        onto(slopes, a, along * direction[a], real, imaginary);
                    }
                }
            }
        }
        // The entry x ← y in the frames: `contract` turns ρ[y, x] by turns[x] turns[y]*, whose slope is i(K_y − K_x)·v
        // times it.
        let gauge = &path.rates.gauge;
        let matrix = |sums: &[f64], a: usize, m: &mut Vec<C64>| {
            m.clear();
            m.extend((0..n * n).map(|place| C64::new(sums[2 * a * padded + place], sums[(2 * a + 1) * padded + place]) * turns[place / n] * turns[place % n].conj()));
        };
        for (a, m) in g.iter_mut().enumerate() {
            matrix(sums, a, m);
        }
        if let Some(slope) = slope {
            for (a, m) in slope.iter_mut().enumerate() {
                matrix(slopes, a, m);
                if path.turning {
                    for (place, z) in m.iter_mut().enumerate() {
                        *z += C64::new(0.0, gauge[place % n] - gauge[place / n]) * g[a][place];
                    }
                }
                hermitian(m, n);
            }
        }
        for (a, m) in g.iter_mut().enumerate() {
            if plan.fourth.is_some() && point.quartic.len() >= 4 * n * n {
                for (z, more) in m.iter_mut().zip(&point.quartic[(1 + a) * n * n..(2 + a) * n * n]) {
                    *z += *more;
                }
            }
            hermitian(m, n);
        }
    }

    /// The cluster's optical sector on the path (optical.rs; step.rs, # The optical sector's noise): its ground states,
    /// the least of its excited states' decay over two, and where it beats on the path the light pattern's fastest beat
    /// there, the largest (k_l − k_m)·v over the plan's beams. None for a cluster without excited states.
    fn optical(&self, path: &Path) -> Option<crate::step::Optical> {
        let (model, plan) = (self.model, &*self.plan);
        let c = &model.clusters[plan.cluster];
        if c.ground == plan.states {
            return None;
        }
        let rates = &path.rates.path;
        let (most, least) = rates.iter().fold((f64::NEG_INFINITY, f64::INFINITY), |(most, least), x| (most.max(*x), least.min(*x)));
        let beat = if !plan.still && most >= least { most - least } else { 0.0 };
        let decay = c.states[c.ground..].iter().map(|&e| 0.5 * model.decay[(e - GROUND, e - GROUND)].re).fold(f64::INFINITY, f64::min);
        Some(crate::step::Optical { ground: c.ground, decay, beat })
    }

    fn work(&self) -> [f64; 3] {
        let node = self.plan.node;
        // The fourth order's: its matrix at a node, its gradient's three forms.
        let (matrix, gradient) = self.plan.fourth.as_ref().map_or((0.0, 0.0), |fourth| ((fourth.n * fourth.n) as f64, (3 * fourth.n * fourth.n) as f64));
        let form = node[1] + self.point.borrow().directions.len() as f64 * node[2] + gradient;
        [node[0] + matrix, form, form]
    }
}

/// A block's field component on one side (1 where it has none).
fn side(fields: &[[C64; 4]], scalar: Option<usize>) -> [C64; 4] {
    scalar.map_or([C64::ONE, C64::ZERO, C64::ZERO, C64::ZERO], |s| fields[s])
}

/// An n × n matrix (row by row) replaced by its Hermitian part.
fn hermitian(m: &mut [C64], n: usize) {
    for x in 0..n {
        for y in x..n {
            let z = (m[x * n + y] + m[y * n + x].conj()) * 0.5;
            m[x * n + y] = z;
            m[y * n + x] = z.conj();
        }
    }
}

impl Middle<'_> {
    /// This middle with its generator's near values moved by `by` (a table's error read of them, tabulate.rs).
    pub(crate) fn moved(&self, by: &[f64]) -> Middle<'_> {
        let (mut point, from) = (Held::<Site>::take(), self.point.borrow());
        assert_eq!(by.len(), from.generator.len(), "an error read of the generator's near values");
        point.beams.clone_from(&from.beams);
        point.amplitudes.clone_from(&from.amplitudes);
        point.place = from.place;
        point.generator.clear();
        point.generator.extend(from.generator.iter().zip(by).map(|(value, error)| value + error));
        point.forms.clone_from(&from.forms);
        point.slopes.clone_from(&from.slopes);
        point.directions.clone_from(&from.directions);
        point.quartic.clone_from(&from.quartic);
        Middle { model: self.model, plan: self.plan.clone(), r: self.r, t: self.t, table: self.table, point: RefCell::new(point), work: self.work }
    }

    /// A node's field components into its scratch, their first `parts` parts (the value; with the gradient, four),
    /// from the plan's beams' amplitudes and gradients at the middle (`Site::amplitudes`): at the
    /// time s from the middle along a path (`on`) each beam's amplitude and gradient at the middle times its plane wave's
    /// phase over the path, e^{i k_l·v s}, with the tones' phases at that time and the Doppler frames' turns; at the fixed
    /// middle point (None) the middle's own, at its time.
    fn along(&self, amplitudes: &[[C64; 4]], path: &Path, on: Option<f64>, parts: usize, sloped: bool, scratch: &mut Scratch) {
        let plan = &*self.plan;
        let s = on.unwrap_or(0.0);
        scratch.turned.clear();
        scratch.turned.extend(amplitudes.iter().zip(&path.rates.path).map(|(a, rate)| {
            let phase = if on.is_some() { C64::cis(rate * s) } else { C64::ONE };
            std::array::from_fn(|part| if part < parts { a[part] * phase } else { C64::ZERO })
        }));
        scratch.plain.clear();
        scratch.doppler.clear();
        scratch.plain_slope.clear();
        scratch.doppler_slope.clear();
        for scalar in &plan.scalars {
            let (mut plain, mut doppler, mut plain_slope, mut doppler_slope) = ([C64::ZERO; 4], [C64::ZERO; 4], [C64::ZERO; 4], [C64::ZERO; 4]);
            for &(at, c) in &scalar.beams {
                // Along the path a beam turns at its rate, e^{i rate s}, its conjugate the other way.
                let turn = C64::new(0.0, if scalar.conjugate { -path.rates.path[at] } else { path.rates.path[at] });
                for part in 0..parts {
                    let z = (if scalar.conjugate { scratch.turned[at][part].conj() } else { scratch.turned[at][part] }) * c;
                    plain[part] += z;
                    doppler[part] += z * path.rates.shifts[at];
                    if sloped {
                        plain_slope[part] += z * turn;
                        doppler_slope[part] += z * turn * path.rates.shifts[at];
                    }
                }
            }
            scratch.plain.push(plain);
            scratch.doppler.push(doppler);
            if sloped {
                scratch.plain_slope.push(plain_slope);
                scratch.doppler_slope.push(doppler_slope);
            }
        }
        scratch.phases.clear();
        scratch.phases.extend(plan.tones.iter().map(|&omega| if omega == 0.0 { C64::ONE } else { C64::cis(omega * (self.t + s)) }));
        scratch.turns.clear();
        scratch.turns.extend(path.rates.gauge.iter().map(|rate| if on.is_some() && path.turning { C64::cis(-rate * s) } else { C64::ONE }));
    }

    /// Re tr(Ĝ_a(s) ρ) per lab axis a: the gradient of the generator's Hermitian part along a path at the time s from the
    /// middle (`on`), or at the fixed middle point (None: every tone in turn, to `each` with its beat Ω, its entries at their
    /// phase at the middle's time), contracted with a Hermitian ρ on the cluster's states (n × n, row by row) without forming
    /// it: per dense term Σ_entries V[x, y] ρ[y, x] and the same of its slopes, times the fields' product and its gradient
    /// -- of a flag's values and their adjoints, whose forms with a Hermitian ρ are conjugates, one, twice (`Term::weight`).
    fn contract(&self, path: &Path, on: Option<f64>, rho: &[C64], each: &mut dyn FnMut(f64, [f64; 3])) {
        let (plan, n, point) = (&*self.plan, self.plan.states, self.point.borrow());
        let mut scratch = point.scratch.borrow_mut();
        self.along(&point.amplitudes, path, on, 4, false, &mut scratch);
        let Scratch { plain, doppler, phases, turns, weights, sums, .. } = &mut *scratch;
        // ρ[y, x] in the Doppler frames along the path, laid out as the entries x ← y are.
        weights.clear();
        weights.resize(2 * plan.padded, 0.0);
        let (re, im) = weights.split_at_mut(plan.padded);
        for (x, (re, im)) in re[..n * n].chunks_exact_mut(n).zip(im[..n * n].chunks_exact_mut(n)).enumerate() {
            for (y, (re, im)) in re.iter_mut().zip(im.iter_mut()).enumerate() {
                let w = rho[y * n + x] * turns[x] * turns[y].conj();
                (*re, *im) = (w.re, w.im);
            }
        }
        sums.clear();
        sums.resize(if on.is_some() { 1 } else { plan.tones.len() }, [0.0; 3]);
        let width = point.forms.len();
        // A block's products of its two components with their gradients, per Hermitian flag: made once for its terms.
        let mut of = (usize::MAX, [[C64::ZERO; 4]; 3]);
        for term in &plan.form {
            if of.0 != term.block {
                let block = &plan.blocks[term.block];
                let (l, lw, r, rw) = (side(plain, block.left), side(doppler, block.left), side(plain, block.right), side(doppler, block.right));
                let product = |l: &[C64; 4], r: &[C64; 4]| -> [C64; 4] { std::array::from_fn(|part| if part == 0 { l[0] * r[0] } else { l[part] * r[0] + l[0] * r[part] }) };
                of = (term.block, [product(&l, &r), product(&l, &rw), product(&lw, &r)]);
            }
            let span = term.first..term.first + term.len;
            let (real, imaginary) = plan.values(term, &point.forms);
            let value = dot(real, imaginary, &re[span.clone()], &im[span.clone()]) * phases[term.tone];
            // The value's gradient by its slopes: a constant term has none.
            let mut gradient = [C64::ZERO; 3];
            for (f, direction) in point.directions.iter().enumerate().filter(|_| term.near) {
                let (real, imaginary) = point.slopes[f * width + term.at..f * width + term.at + 2 * term.len].split_at(term.len);
                let slope = dot(real, imaginary, &re[span.clone()], &im[span.clone()]) * phases[term.tone];
                for a in 0..3 {
                    gradient[a] += slope * direction[a];
                }
            }
            let (c, sum) = (&of.1[term.flag], &mut sums[if on.is_some() { 0 } else { term.tone }]);
            for a in 0..3 {
                sum[a] += term.weight * (c[1 + a] * value + c[0] * gradient[a]).re;
            }
        }
        // The fourth order's gradient, a static term's.
        if plan.fourth.is_some() && point.quartic.len() >= 4 * n * n {
            for a in 0..3 {
                let gradient = &point.quartic[(1 + a) * n * n..(2 + a) * n * n];
                sums[0][a] += (0..n * n).map(|place| (gradient[place] * rho[(place % n) * n + place / n]).re).sum::<f64>();
            }
        }
        for (m, sum) in sums.iter().enumerate() {
            each(if on.is_some() { 0.0 } else { plan.tones[m] }, *sum);
        }
    }
}

#[cfg(test)]
mod tests;
