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
//! - the same pairs' part at the atom, e^ω'_k'* e^ω_k S_k'k on the ground states (eliminate.rs), the same contraction of
//!   the explicit states' static couplings to Q (eliminate.rs `Responses::held`) with a carrier's legs, e_k on an explicit
//!   and a ground place, and with each other on two explicit places, and the explicit drives, e_k U_k[e, b] and their
//!   adjoints;
//! - the decay −(i/2) Σ_ρ 𝒯_ρ† D 𝒯_ρ of the excited amplitude (# The excited amplitude), made at each node of the node's
//!   own 𝒯: no term of its own.
//! One rule takes an entry a ← b of the first two kinds into the generator, whatever light makes it (eliminate.rs
//! `Model::term_included`): its residual beat in the frames exactly zero, or its unit kept (`Unit::Term`).
//! A term's entry a ← b lies at the residual beat Ω + θ_a − θ_b in the cluster's frames (cluster.rs), its tone, and each product
//! of two beams' plane waves in it at the residual wave vector q − (K_a − K_b). A Hermitian term adds c M to the generator and
//! ∇(c M) to its gradient; the decay adds no force (the absorption's mean force is the photons'). Each Hermitian term comes
//! with its adjoint, so the generator's Hermitian part and its gradient are Hermitian at every time.
//!
//! # Near forms
//! The drives folded with the levels at the atom enter through their elimination's responses on each carrier's field basis,
//! T_k and T2_k (eliminate.rs), by a few complex numbers that are linear in them (`NearLayout`): the shifts' S_k'k =
//! −½(U^f_k'†T_k + T_k'†U^f_k) and their Doppler partners D_k'k = ½ U^f_k'†T2_k on the cluster's ground places, U^f a
//! frequency's every folded coupling (far.rs `Tone::coupled`) and T its response at the atom, none where it is no carrier; the
//! first order's crossed forms of the other reference classes (`NearLayout::crossed`); the same of the explicit states'
//! static couplings to Q, W[:, x] = K[Q, x] (eliminate.rs `Responses::held`: a pair side of the frequency `tones.len()`,
//! no component, no Doppler partner, its coupling the far light's at the point -- a pair's form its coupling's and its
//! response's product, part by part); the reference Z₀; and the responding forms -- the responses themselves, their partners and the mixed responses
//! (`NearLayout::mixed`) --, of which the excited amplitude is made (# The excited amplitude). Each exists where it can
//! differ from zero: the responses of a ground state reach the excited states its coupling reaches through the excited
//! states' own couplings (`Model::reached`, the responses' one support). At a point
//! they are the exact elimination's there with their exact derivatives (`Model::near_exact`), or its table's interpolant
//! (tabulate.rs): one definition, tabulated or not.
//!
//! # First-order Doppler
//! A beam's absorption at w_l = (ħ/m)(k_l·κ + k_l²/2) takes T_l − w_l T2_l: with a carrier's field e_k, its Doppler-weighted
//! field Σ_l w_l a_l γ_lk multiplies the partner form. A term's value at a beam pair is therefore v₀ + w_r v_r + w_l v_l +
//! w_l w_r v_lr over the factors that are a near carrier's (`PLAIN`, `RIGHT`, `LEFT`; the decay's both-sided part is the
//! product of 𝒯(κ) with itself, # The excited amplitude): the generator is a Lindblad one at every momentum, from a table
//! or not.
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
//! generator there is the sum over the blocks of the product of two components times the block's values, and the decay of
//! the node's 𝒯 (# The excited amplitude), each entry at its tone's phase e^{iΩ(t_m + s)} and in its states' Doppler frames,
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
//! of constants keeps its values in the plan; one that a near form enters has them at the point (`Site`): the generator's
//! values and, for the forms with ρ, the same with their slopes. A point's values are their constant part and what the near
//! forms lay there (`Plan::lay`, linear in the forms). A table holds them at its nodes in this same layout (tabulate.rs,
//! # A node's reals), so a middle interpolates each number straight to its place onto its constant.
//!
//! # The excited amplitude
//! The slow state's excited amplitude is one object: per excited replica ρ (photon.rs, # Replicas and classes; `ReplicaPlan`)
//! the fold's wave operator 𝒯_ρ on the cluster's places, a sum over its legs (`DecayLeg`) -- e_k T_k − e^w_k T2_k of each
//! lit carrier's response, e_k R of each bare drive's constant response, −1 of each explicit state of its frame (the
//! photon's sign, photon.rs) --, and at the series' first order 𝒯⁽¹⁾[:, s] = Σ_t M⁽ᵃ⁽ˢ⁾⁾[:, t] E[t, s] (# The normalized
//! generator). Of it:
//! - the decay Λ = Σ_ρ 𝒯_ρ† D 𝒯_ρ, bilinear in the field components: per pair of a replica's legs one complex pair form of
//!   their vectors (`DecayForm`, `Builder::decays`: a_l†D a_r and its Doppler partners' products) on the block of the two
//!   components and its flag (`PLAIN`, `RIGHT`, `LEFT`, `BOTH`), laid in the generator's dense terms beside the shifts'
//!   -- the shifts and the decay one set of complex pair forms, the Hermitian and the anti-Hermitian part --, with a merged
//!   class's cross terms of its feeds' replicas at their beat (its C_B the kernel). At a node the terms' sum is
//!   H − (i/2)Λ; where the generator bends, Λ = i(h − h†) taken to S⁻¹(Λ + Λ⁽¹⁾)S⁻¹ (`Middle::generator`);
//! - the photons: a class's u_a = Σ_feeds C_a[b', :] 𝒯_ρ S⁻¹ψ (photon.rs, # A state's photons at a point), the replicas'
//!   ε_ρ = 𝒯_ρ S⁻¹ψ on their rows made once a state, term by term of 𝒯 (`Plan::amplitudes`: 𝒯 itself is never laid)
//!   with their gradients;
//! - the metric X = Σ_ρ 𝒯_ρ†𝒯_ρ and the loss L = Σ_ρ 𝒯_ρ† D 𝒯_ρ of its near part: the same pair forms' sums
//!   (`Plan::sums`, `Fourth::sum_into`), the normalization's.
//! The compression of the no-jump generator onto the slow subspace (1; 𝒯) in orthonormal coordinates has i(G − G†) = Σ L†L
//! for any 𝒯 and any Hermitian S (ELIMINATION v12): the decay is the photons' rate on every state -- to the rounding of
//! the sums where the forms are the exact elimination's (and at a table's nodes), to the table's error between its
//! nodes, where the pair forms and the responses are each interpolated (tabulate.rs, # The table's error).
//!
//! A middle's numbers and a path's are a thread's spares (`Held`): a step takes the vectors the step before left, and the
//! model's part of a step asks the heap for nothing.
//!
//! # The normalized generator
//! The near forms are the fold's series about the reference Z₀ = diag(z_y) (elim.rs, # The definition and the route it
//! is evaluated on; eliminate.rs `Model::eliminate`): per pair of field components the order-0 form W′R₀(y)W (S, its
//! Hermitian part laid in the generator), the Doppler partner D = ½ W′R₀(y)²W, the first order's crossed forms
//! Y⁽ᵃ⁾ = W′R₀(z_a)R₀(z_y)W of each reference class a that is not y's (`NearLayout::crossed`; 2D where it is), and Z₀
//! itself, a number a place -- all linear in the products e_k'* e_k with coefficients of the far light alone, tabulated
//! as S is -- and the responding forms (# The excited amplitude): the metric X and the loss L are its near part's. The
//! places' reference classes (eliminate.rs `Model::references`) are the resolvents the first order is of: column s of
//! it at s's own reference, the others' at theirs (ELIMINATION v11, v12). On the
//! slow invariant subspace Ψ = (x, T x) of the no-jump generator, T_f = Σ_j (−1)^j R₀^{j+1} W (H − z₀)^j and H the
//! fixed point of H = A + Σ_j (−1)^j N_j (H − z₀)^j, to the series' order K (model.rs `Model::series`: K = 1 where
//! `Unit::Series` is kept, the evaluation's error measured at the run's states against the definition, elim.rs
//! `Model::readings`;
//! the series' ratio q < 1 only says it converges). In the orthonormal coordinates y = S x, S = (1 + X)^{1/2} -- the Schrieffer–Wolff
//! coordinates, the unitary block-diagonalization's nearest the bare basis (ELIMINATION v6, v7) -- the generator is the
//! des Cloizeaux one, S H S⁻¹, and the jumps L_a = C_a 𝒯 S⁻¹, whose own product is its decay. Evaluated by the
//! Schrieffer–Wolff series, products of the forms alone (`normal`): S^{±1} = Σ_{j≤J} binom(±½, j) X^j to the
//! normalization's order J (`Unit::Normalization`), the Hermitian part the series' own (elim.rs `Series`), with H_h the
//! static Hermitian generator the forms lay and P_a the places of reference class a:
//!
//! ```text
//! E₀ = H_h − (i/2) L − Z₀     H¹ = −Σ_a Y⁽ᵃ⁾ E₀ P_a (K = 1; nothing at K = 0)     E = E₀ + H¹
//! held = Herm H¹     𝒯 = 𝒯⁽⁰⁾(κ) + 𝒯⁽¹⁾, 𝒯⁽¹⁾[:, s] = Σ_t M⁽ᵃ⁽ˢ⁾⁾[:, t] E[t, s]     decay S⁻¹ Σ 𝒯†D𝒯 S⁻¹
//! L_a = C_a 𝒯 S⁻¹     landing S (1 + Y_L)⁻¹, Y_L[x, y] = Y⁽ᵃ⁽ˣ⁾⁾[x, y]
//! ```
//!
//! The normalized part varies along a step's path as the light's pattern does: it is made at each node of the step of
//! the node's own forms -- the middle's near forms with the node's field components, its plane waves turned
//! (`Model::node_normal`) --, as every other term's value is. (Held at the middle, it was the midpoint rule of a
//! quantity that turns at the pattern's K v: a step error of (Kvτ)²/24 of it, which a step's companion at 2τ reads
//! four times over -- PGC3D's pilot read its third state's population at 1.44 of the step's share squared against
//! the released engine's 0.65, and ran 528 trajectories more to see through it.)
//! - a node's generator (`Middle::generator`): the decay of the node's own 𝒯 (# The excited amplitude) at the path's
//!   momentum, S⁻¹(Σ 𝒯†D𝒯)S⁻¹, and the held part, in the entries' own coordinates before the frames turn them as every
//!   entry is;
//! - a node's force (`Middle::gradient`, `Middle::contract`): the held part's gradient there, where it is anything;
//! - a state's photons: C_a 𝒯 S⁻¹ψ of the point's own 𝒯 and S⁻¹ (`Quartic`; a thread's last point is kept, so a step's
//!   middle and its scattering share one); a jump's landing (model.rs `Model::landed`): S(1 + Σ S_f T_f)⁻¹ f, the left
//!   and right wave operators' overlap to the series' leading order Y_L; the readout of the slow state x = S⁻¹ψ itself
//!   (readout.rs).
//! So the force is still the gradient of the generator's Hermitian part and the decay still the photons' own rate.
//! Terms that beat in the cluster's frames are laid as the forms give them, and enter none of X and H: a replica's own
//! norm is its static part (the cross terms are the sectors', folded at their time mean: elim.rs # Sectors).
//!
//! # Exact tones
//! Along a path every piece of a dense term -- an entry and a pair of its block's two components' beams -- turns at one
//! beat: its tone's residual beat Ω and its two plane waves' residual wave vector in the cluster's Doppler frames Q,
//! e^{i(Ω + Q·v)s} at the time s from the middle, its value and the beams' amplitudes the middle's. The plan groups the
//! pieces by (Ω, Q) exactly (`Group`: Q the field's wave vectors less the tree's frames, an exact lattice vector); the
//! static pieces, of no beat in time or along the path, are in none. A group and its adjoint's, (−Ω, −Q), are a tone
//! pair, its first standing for it (`Plan::firsts`): on a path its beat ν = Ω + Q·v (`Middle::beats`), its members'
//! envelopes the groups' pieces at the middle (`Middle::envelopes`: constant along the path), the generator's part of
//! the pair at the time s e^{iνs}G⁺ + e^{−iνs}G⁻ exactly. Which of them the propagation integrates at their beats and
//! which it takes through the step's nodes with the rest of the generator is the propagation's own choice (spectral.rs,
//! # Which beats are tones). The force's form by the same groups (`Middle::tone_force`): of a pair at the beat ν, on
//! σ = Σ_k w_k e^{iνs_k} ρ_k, the part of Σ_k w_k Re tr(Ĝ(s_k) ρ_k) its pieces make -- the first's on σ, its partner's on
//! σ† --, by the form's own weights (`Term::weight`), so that the pair's part of the step's impulse is integrated at its
//! beat as its part of the state is. The normalized generator's part (# The normalized generator) is in no group: its
//! beats are its forms' products, a part X of the beats' own size, taken with the static group. It is a correction of
//! the second order: the forms are the fold's responses and their metric, of the order (g/Δ)² against the generator
//! (g a coupling, Δ its detuning from the folded state), and the series' terms beyond the leading one products of them.
//! What taking its beats through the three nodes leaves -- of a tone the hops' second Magnus term goes as the square of
//! the tone's amplitude (spectral.rs, # Which beats are tones), its alias beyond νh = π its own size -- is therefore
//! (g/Δ)² of what the same beat leaves in the leading part, a small share of the step's error, which the check reads;
//! splitting it into groups would cost the series' products per group at every node for that share.
//!
//! # Work
//! The complex multiply-adds are counted where they are made (`Middle::work`, `Plan::node`), the operations done: at a
//! point the near terms -- a table's reals as it reads them (`Plan::near_read`: the laid values and the responding forms,
//! which the table holds as its nodes do, no value made of them), or the exact elimination and the values laid of its
//! forms (`Plan::formed_work`) --; at a node the components from their beams and, per block and flag, its entries, and the
//! decay's pair forms among the dense terms (`Builder::decays`); a site's photons as its amplitudes and the
//! classes' rows make them (photon.rs `Plan::classes_work`). A node's counts are the plan's: no generator is made to
//! know them. (A table's point once counted the values the exact forms make beside its read, and a site's classes the
//! definition's terms an entry: PGC3D's and GMD1C's middle and scattering read 2.4 and 1.5 times what they made.)

use super::eliminate::{half, Responses};
use super::far::EXCITED;
use super::photon::Channel;
use super::tabulate::NearTable;
use super::elim::{binomial, Orders};
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
/// Both factors Doppler-weighted: the decay's product of two partners (module documentation, # The excited amplitude).
pub(crate) const BOTH: usize = 3;

/// The jet of a leg of no field component (an explicit state's own: its coupling the levels', eliminate.rs
/// `Responses::held`): one, its gradient none.
const UNIT: [C64; 4] = [C64::ONE, C64::ZERO, C64::ZERO, C64::ZERO];

/// A leg of a replica's excited amplitude (module documentation, # The excited amplitude; photon.rs `Replica`): a lit
/// carrier's response at the atom on its basis vector k (the near forms' responses), or a constant vector of the plan's
/// (`Plan::legs`: a bare component's response, an explicit state's own amplitude −e_x).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum DecayLeg {
    Near { carrier: usize, k: usize },
    Constant(usize),
}

/// What a decay sum is of (`DecaySum`): the metric X, the loss L, the first order's 𝒯⁽⁰⁾†K M⁽ᵃ⁾ (per reference class a),
/// its M⁽ᵇ⁾†K M⁽ᵃ⁾ (per pair: b·count + a).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Sum {
    Metric,
    Loss,
    First,
    Second,
}

/// A decay form's entry in one of the normalized generator's sums (`Builder::decays`): its kind, block and flag (the
/// components' product it takes: `PLAIN`, or `LEFT` the left one Doppler-weighted), its entry x ← y and tone, the form
/// (`NearLayout::decays`), its slot (a class, or a pair of them) and factor.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DecaySum {
    kind: Sum,
    block: usize,
    flag: usize,
    x: usize,
    y: usize,
    tone: usize,
    form: usize,
    slot: usize,
    factor: C64,
}

/// A decay form (module documentation, # The excited amplitude): v_l† K v_r of two legs' vectors -- each leg's of an order
/// (0 its amplitude, T or the constant; 1 its Doppler partner T2; 2 + a its mixed response of reference class a, the
/// series' first order's, `NearLayout::mixed`) at its place, x the left one's and y the right one's --, K the plan's kernel
/// (`Plan::kernels`: 0 the identity, the metric's; 1 the decay D; 2 + c the merged class c's C_B).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct DecayForm {
    pub(crate) left: (DecayLeg, usize),
    pub(crate) right: (DecayLeg, usize),
    pub(crate) x: usize,
    pub(crate) y: usize,
    pub(crate) kernel: usize,
}

/// A near form or a constant: what a term's value is read from (module documentation, # Near forms).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Operand {
    /// S of a pair entry.
    Pair(usize),
    /// D of a pair entry.
    Doppler(usize),
    /// A pair entry's first-order form of another place's reference (`NearLayout::crossed`).
    Crossed(usize),
    /// The series' reference z_y of a place (eliminate.rs `Responses::reference`).
    Reference(usize),
    /// A decay form (`NearLayout::decays`).
    Decay(usize),
}

/// One cluster's near forms under a light set (module documentation, # Near forms), in the order of a point's values and of a
/// table's node: the Hermitian forms -- S, D, the crossed forms, the reference, the decay forms -- and the responding ones -- the
/// responses, their Doppler partners, the mixed responses.
#[derive(Clone, Debug, Default)]
pub(crate) struct NearLayout {
    /// (left frequency, its basis vector k', right frequency, its k, places x, y): S and D of the pair's entry x ← y.
    pub(crate) pairs: Vec<[usize; 6]>,
    /// (pair, reference class a): the first order's form of the pair's entry x ← y at the reference of class a, where that is
    /// not y's own (module documentation, # The normalized generator): Y⁽ᵃ⁾[x, y] = U^f_k'† R₀(z_a)R₀(z_y) U_k, its partner
    /// 2D where a is y's class.
    pub(crate) crossed: Vec<[usize; 2]>,
    /// (carrier, k, excited state e, place y): T_k[e, y], and after them T2_k[e, y].
    pub(crate) responses: Vec<[usize; 4]>,
    /// (response, reference class a): R₀(z_a)R₀(z_y) U_k[e, y] of a response's entry at the reference of class a, where that
    /// is not y's own (its partner T2 there).
    pub(crate) mixed: Vec<[usize; 2]>,
    /// The reference Z₀, a number a place of the cluster (or none): where a carrier is lit.
    pub(crate) reference: usize,
    /// The decay forms (`DecayForm`): of every pair of a replica's legs that holds a near one, and of two merged feeds' --
    /// first the `normalizing_decays` the normalized generator's sums read (`Plan::sums`), then those the generator alone
    /// takes.
    pub(crate) decays: Vec<DecayForm>,
    pub(crate) normalizing_decays: usize,
}

impl NearLayout {
    /// The complex numbers of a point.
    pub(crate) fn len(&self) -> usize {
        self.responding().end
    }
    /// Where an operand lies among a point's values.
    fn at(&self, o: Operand) -> usize {
        let (p, c, r) = (self.pairs.len(), self.crossed.len(), self.reference);
        match o {
            Operand::Pair(i) => i,
            Operand::Doppler(i) => p + i,
            Operand::Crossed(i) => 2 * p + i,
            Operand::Reference(i) => 2 * p + c + i,
            Operand::Decay(i) => 2 * p + c + r + i,
        }
    }
    /// The forms laid in the generator (S, D, the crossed forms, the reference, the decay forms); the responses
    /// with their partners and the mixed ones.
    pub(crate) fn hermitian(&self) -> Range<usize> {
        0..2 * self.pairs.len() + self.crossed.len() + self.reference + self.decays.len()
    }
    pub(crate) fn responding(&self) -> Range<usize> {
        self.hermitian().end..self.hermitian().end + 2 * self.responses.len() + self.mixed.len()
    }
    /// The crossed forms, the reference and the decay forms: what the normalized generator reads besides the generator's
    /// (`Site::normal`): its metric X, loss L and the first order's decay.
    pub(crate) fn normalizing(&self) -> Range<usize> {
        2 * self.pairs.len()..2 * self.pairs.len() + self.crossed.len() + self.reference + self.normalizing_decays
    }
}

// The near forms at a point over the entries before `len` (zero outside the sections and the places evaluated), to an order, and the complex
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
    /// The places of the entries it holds (n × n row by row), within its span: its pieces' entries (`Group`).
    held: Vec<usize>,
    /// In a form with a Hermitian ρ (`Middle::contract`), its weight. A flag's values and their adjoints -- the two
    /// components conjugated, the base beat negated, the entries transposed and conjugated, the Doppler-weighted side
    /// exchanged -- make conjugate forms: of a block and its adjoint the first is taken twice and the other left out, and of a
    /// block that is its own adjoint the flag `RIGHT` twice for `LEFT`.
    weight: f64,
}

/// A beating group of the generator's terms (module documentation, # Exact tones): the pieces of the dense terms -- an
/// entry and a pair of its block's two components' beams -- of one tone whose plane waves' residual wave vector in the
/// cluster's Doppler frames is one Q (1/µm): on a path of the velocity v all of them turn at the one beat Ω + Q·v. Its
/// partner is the group of the adjoint pieces (−Ω, −Q), none where there is none; of the two the first stands for the
/// pair.
#[derive(Clone, Debug)]
pub(crate) struct Group {
    tone: usize,
    wave: [f64; 3],
    /// Its exact label (step.rs `Middle::beats`): its tone's beat's and its residual wave vector's integers over the
    /// plan's common denominator.
    key: Vec<i64>,
    partner: Option<usize>,
    first: bool,
    generator: Vec<Piece>,
    form: Vec<Piece>,
}

/// A piece of a dense term (`Group`): the term (its index among the generator's or the form's), the entry's offset in its
/// span, and the beams of the block's left and right components it takes (their places among the components' beams;
/// none of an absent component).
#[derive(Clone, Copy, Debug)]
struct Piece {
    term: usize,
    offset: usize,
    left: Option<usize>,
    right: Option<usize>,
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
        for e in &held {
            values[at + e.0 - first] += e.1.re;
            values[at + len + e.0 - first] += e.1.im;
        }
        self.index.insert(key, self.terms.len());
        let held = held.iter().map(|e| e.0).collect();
        self.terms.push(Term { block: key.0, flag: key.1, tone: key.2, first, len, at, near, held, weight });
    }
    /// Where the entry at `place` of a near term's values lies among a point's: its real part and its imaginary part.
    fn place(&self, key: (usize, usize, usize), place: usize) -> Option<[usize; 2]> {
        self.index.get(&key).map(|&t| &self.terms[t]).map(|t| [t.at + place - t.first, t.at + t.len + place - t.first])
    }
}

/// One excited replica of the cluster under the plan's lights (photon.rs `Replica`; module documentation, # The excited
/// amplitude): its rows' vectors on the excited states (less GROUND; `Builder::replicas`) -- its `own` rows first, each an
/// excited state's unit vector, then a row per bare response, the response itself at its one place --, its
/// place among the plan's amplitude rows (`at`: the rows of the replicas before it), and what makes 𝒯_ρ on its rows j and
/// the cluster's places y -- per lit carrier's response entry (its place among the near forms' responses, its field
/// component, j, y): e_k T_k − e^w_k T2_k; per bare drive's (its field component, j, y, one): e_k; per explicit state
/// (j, y): −1, its own amplitude (the photon's sign, photon.rs). What a state's photons are read on; the decay is the
/// plan's pair forms (`Builder::decays`).
#[derive(Clone, Debug)]
pub(crate) struct ReplicaPlan {
    pub(crate) size: usize,
    basis: Vec<[C64; EXCITED]>,
    pub(crate) at: usize,
    near: Vec<(usize, Option<usize>, usize, usize)>,
    bare: Vec<(usize, usize, usize, C64)>,
    explicit: Vec<(usize, usize)>,
}

/// One photon class of the cluster under the plan's lights (photon.rs `Class`): its landing cluster, its wave number, the
/// largest wave number of the beams its folded amplitudes absorb from (`Photon::absorbed`; zero without one), its landing
/// cluster's `states` states, the lanes' rows of its jumps (`rows`: row a `states` + b' as whole vectors), and its entries
/// -- per row (its place among the lanes' sums, `laid` of its first column: column c a further c 2 LANES on), feed and
/// excited state of the feed's replica its place among the amplitude rows, its relative beat's value
/// in a merged class (rad/µs) and the emission's C_a[b', e] (photon.rs, # A state's photons at a point).
#[derive(Clone, Debug)]
pub(crate) struct ClassPlan {
    pub(crate) landing: usize,
    pub(crate) wave_number: f64,
    pub(crate) absorbed: f64,
    pub(crate) states: usize,
    pub(crate) rows: usize,
    pub(crate) entries: Vec<(usize, usize, f64, C64)>,
    /// Its entries at a relative beat (a merged class's feeds'): each one product more, its phase factor's.
    pub(crate) turning: usize,
}

/// A Hermitian near form's place in the generator: the block, flag and entry (x, y) it is the value of, its tone, the form's
/// place, whether it enters conjugated, and where it lies among a point's values of the generator and of the forms with a
/// Hermitian ρ (its real and its imaginary part; none where its term is left to its adjoint, `Term::weight`).
#[derive(Clone, Debug)]
pub(crate) struct Link {
    pub(crate) block: usize,
    pub(crate) flag: usize,
    pub(crate) place: (usize, usize),
    pub(crate) tone: usize,
    pub(crate) operand: usize,
    pub(crate) conjugate: bool,
    /// What the form is taken times: one for a Hermitian form, the decay's −(i/2) part for a decay form's; and whether it
    /// is a Hermitian form's (the potential's and the force's: `Model::potential`).
    pub(crate) factor: C64,
    pub(crate) hermitian: bool,
    generator: [usize; 2],
    form: Option<[usize; 2]>,
}

/// What the normalized generator of a cluster's fold reads of its plan (module documentation, # The normalized
/// generator): the cluster's n places; each place's reference class (`Plan::references`) and their number; per static
/// pair entry's crossed form, block by block -- per block its two field components and the end of its entries
/// (`groups`): (the crossed form, its class a, the entry x n + y) --, the block's product of components once for its
/// entries; and the evaluation's orders (model.rs `Model::series`, elim.rs `Series::orders`).
#[derive(Clone, Debug)]
pub(crate) struct Fourth {
    pub(crate) n: usize,
    classes: Vec<usize>,
    count: usize,
    groups: Vec<(Option<usize>, Option<usize>, usize)>,
    entries: Vec<(usize, usize, usize)>,
    /// The decay sums (`DecaySum`) by kind -- the metric X, the loss L, the first order's two (`Sum`) --, each in runs of
    /// one block, flag and tone: (left component, right component, flag, tone, the run's end) and its entries (the form's
    /// place among the normalizing forms, the slot, the entry x n + y, the factor).
    decays: [DecayRuns; 4],
    pub(crate) orders: Orders,
}

/// One kind of decay sums' runs (`Fourth::decays`).
#[derive(Clone, Debug, Default)]
pub(crate) struct DecayRuns {
    heads: Vec<(Option<usize>, Option<usize>, usize, usize, usize)>,
    entries: Vec<(usize, usize, usize, C64)>,
}

impl DecayRuns {
    /// The runs: each its components, flag, tone and entries.
    fn runs(&self) -> impl Iterator<Item = (Option<usize>, Option<usize>, usize, usize, &[(usize, usize, usize, C64)])> {
        let mut start = 0;
        self.heads.iter().map(move |&(left, right, flag, tone, end)| {
            let run = &self.entries[start..end];
            start = end;
            (left, right, flag, tone, run)
        })
    }
}

impl Fourth {
    fn new(plan: &Plan, orders: Orders) -> Option<Fourth> {
        if plan.near.reference == 0 {
            return None;
        }
        let (n, p) = (plan.states, plan.near.pairs.len());
        let mut linked: Vec<(usize, usize, usize, usize)> = vec![];
        for l in plan.links.iter().filter(|l| l.flag == PLAIN && !l.conjugate && l.operand < p && plan.tones[l.tone] == 0.0) {
            for (c, &[pair, a]) in plan.near.crossed.iter().enumerate() {
                if pair == l.operand {
                    linked.push((l.block, c, a, l.place.0 * n + l.place.1));
                }
            }
        }
        linked.sort_by_key(|link| link.0);
        let (mut groups, mut entries) = (Vec::<(Option<usize>, Option<usize>, usize)>::new(), Vec::with_capacity(linked.len()));
        for (k, &(block, crossed, a, xy)) in linked.iter().enumerate() {
            if k == 0 || linked[k - 1].0 != block {
                groups.push((plan.blocks[block].left, plan.blocks[block].right, 0));
            }
            entries.push((crossed, a, xy));
            groups.last_mut().expect("a block's group").2 = entries.len();
        }
        // The decay sums by kind, in runs of one block, flag and tone.
        let offset = plan.near.crossed.len() + plan.near.reference;
        let mut decays: [DecayRuns; 4] = Default::default();
        for (k, kind) in [Sum::Metric, Sum::Loss, Sum::First, Sum::Second].into_iter().enumerate() {
            let mut of: Vec<&DecaySum> = plan.sums.iter().filter(|sum| sum.kind == kind).collect();
            of.sort_by_key(|sum| (sum.block, sum.flag, sum.tone));
            let runs = &mut decays[k];
            for (i, sum) in of.iter().enumerate() {
                if i == 0 || (of[i - 1].block, of[i - 1].flag, of[i - 1].tone) != (sum.block, sum.flag, sum.tone) {
                    runs.heads.push((plan.blocks[sum.block].left, plan.blocks[sum.block].right, sum.flag, sum.tone, 0));
                }
                runs.entries.push((offset + sum.form, sum.slot, sum.x * n + sum.y, sum.factor));
                runs.heads.last_mut().expect("a run").4 = runs.entries.len();
            }
        }
        Some(Fourth { n, classes: plan.references.clone(), count: plan.reference_classes, groups, entries, decays, orders })
    }

    /// The decay sums of kind `kind` (`Fourth::decays`) added into `out`, per slot `parts` lanes of n² (row by row) after
    /// one another: each entry's form value (`value` of its place among the normalizing forms) times its block's components'
    /// product by its flag -- `PLAIN` the plain ones (`plain`), `LEFT` the left one Doppler-weighted (`doppler`) -- at its
    /// tone's phase (`phases`; none: the static tone alone), times its factor; with `slopes` (the forms' slopes along dn
    /// directions) and `gradients` the lanes [value, its slopes along the directions, the components' gradients along the
    /// lab axes] as `Model::normal_at` lays them (parts 4 + dn), else the value alone.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sum_into(&self, kind: usize, plain: &[[C64; 4]], doppler: Option<&[[C64; 4]]>, phases: Option<&[C64]>, value: &dyn Fn(usize) -> C64, slopes: Option<(&dyn Fn(usize, usize) -> C64, usize)>, gradients: bool, out: &mut [C64]) {
        let size = self.n * self.n;
        let dn = slopes.map_or(0, |s| s.1);
        let parts = if gradients { 4 + dn } else { 1 };
        for (left, right, flag, tone, run) in self.decays[kind].runs() {
            let phase = match phases {
                Some(phases) => phases[tone],
                None if tone == 0 => C64::ONE,
                None => continue,
            };
            let l = if flag == LEFT { side(doppler.expect("the Doppler-weighted components"), left) } else { side(plain, left) };
            let r = if flag == RIGHT { side(doppler.expect("the Doppler-weighted components"), right) } else { side(plain, right) };
            let c: [C64; 4] = std::array::from_fn(|p| phase * if p == 0 { l[0] * r[0] } else { l[p] * r[0] + l[0] * r[p] });
            for &(at, slot, xy, factor) in run {
                let (v, base) = (value(at) * factor, slot * parts * size + xy);
                out[base] += c[0] * v;
                if gradients {
                    if let Some((slope, _)) = slopes {
                        for f in 0..dn {
                            out[base + (1 + f) * size] += c[0] * slope(f, at) * factor;
                        }
                    }
                    for k in 0..3 {
                        out[base + (1 + dn + k) * size] += c[1 + k] * v;
                    }
                }
            }
        }
    }
    /// The series' first order's decay Λ⁽¹⁾ added into `lambda` (n × n, row by row; module documentation, # The normalized
    /// generator): 𝒯⁽¹⁾ = Σ_a M⁽ᵃ⁾ E P_a of the node's E (`e`), so Λ⁽¹⁾ = Σ_a (R⁽ᵃ⁾ E P_a + its adjoint) + Σ_{b,a} P_b E†
    /// B⁽ᵇᵃ⁾ E P_a with R⁽ᵃ⁾ = 𝒯⁽⁰⁾†K M⁽ᵃ⁾ and B⁽ᵇᵃ⁾ = M⁽ᵇ⁾†K M⁽ᵃ⁾ the decay sums of the node's components at its tones'
    /// phases (`Sum::First`, `Sum::Second`). `scratch` holds the sums.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn first_decay(&self, plain: &[[C64; 4]], doppler: &[[C64; 4]], phases: &[C64], value: &dyn Fn(usize) -> C64, e: &[C64], scratch: &mut Vec<C64>, lambda: &mut [C64]) {
        let (n, size, classes) = (self.n, self.n * self.n, self.count.max(1));
        let class = |s: usize| if self.classes.is_empty() { 0 } else { self.classes[s] };
        sized(scratch, (classes + classes * classes + 2) * size);
        scratch.fill(C64::ZERO);
        let (sums, rest) = scratch.split_at_mut((classes + classes * classes) * size);
        let (r_sums, b_sums) = sums.split_at_mut(classes * size);
        self.sum_into(2, plain, Some(doppler), Some(phases), value, None, false, r_sums);
        self.sum_into(3, plain, Some(doppler), Some(phases), value, None, false, b_sums);
        let (ep, t) = rest.split_at_mut(size);
        // R⁽ᵃ⁾ E P_a and its adjoint.
        for a in 0..classes {
            let r = &r_sums[a * size..(a + 1) * size];
            for x in 0..n {
                for s in (0..n).filter(|&s| class(s) == a) {
                    let z: C64 = (0..n).map(|t| r[x * n + t] * e[t * n + s]).sum();
                    lambda[x * n + s] += z;
                    lambda[s * n + x] += z.conj();
                }
            }
        }
        // P_b E† B⁽ᵇᵃ⁾ E P_a: per pair the columns of E of class a, then the rows of E† of class b.
        for b in 0..classes {
            for a in 0..classes {
                let bb = &b_sums[(b * classes + a) * size..(b * classes + a + 1) * size];
                for t in 0..n {
                    for s in 0..n {
                        ep[t * n + s] = if class(s) == a { (0..n).map(|u| bb[t * n + u] * e[u * n + s]).sum() } else { C64::ZERO };
                    }
                }
                for x in (0..n).filter(|&x| class(x) == b) {
                    for s in 0..n {
                        t[x * n + s] = (0..n).map(|u| e[u * n + x].conj() * ep[u * n + s]).sum();
                        lambda[x * n + s] += t[x * n + s];
                    }
                }
            }
        }
    }
    /// The counted complex multiply-adds of `first_decay`.
    pub(crate) fn first_decay_work(&self) -> f64 {
        let (n, classes) = (self.n, self.count.max(1));
        self.sum_work(2, None) + self.sum_work(3, None) + (2 * n * n * n + 2 * classes * classes * n * n * n) as f64
    }
    /// The counted complex multiply-adds of `sum_into` of kind `kind`, with `gradients` along `dn` slopes' directions.
    pub(crate) fn sum_work(&self, kind: usize, gradients: Option<usize>) -> f64 {
        let entries = self.decays[kind].entries.len();
        let heads = self.decays[kind].heads.len();
        (heads * if gradients.is_some() { 7 } else { 1 } + entries * gradients.map_or(1, |dn| 4 + dn)) as f64
    }

    /// Whether the held part is anything (`normal`): the series' first order -- at order 0 the generator's Hermitian part
    /// is H_h, exactly (the Hermitian part is the series' own).
    pub(crate) fn held(&self) -> bool {
        self.orders.series >= 1
    }

    /// Whether the normalized generator moves anything (`normal`): the held part, or S^{±1} at an order J ≥ 1 -- at J = 0
    /// S⁻¹ = 1 and the landing map 1, the decay the excited amplitudes' own product, a state's photons on ψ itself.
    pub(crate) fn bends(&self) -> bool {
        self.held() || self.orders.normalization >= 1
    }

    /// The crossed forms block by block: the block's two field components and its entries (the form, its class, the entry
    /// x n + y).
    fn groups(&self) -> impl Iterator<Item = (Option<usize>, Option<usize>, &[(usize, usize, usize)])> {
        let mut start = 0;
        self.groups.iter().map(move |&(left, right, end)| {
            let group = &self.entries[start..end];
            start = end;
            (left, right, group)
        })
    }
}

/// The most directions a site's forms change along (a table's axes, or the three lab axes).
const DIRECTIONS: usize = if crate::table::MOST_AXES > 3 { crate::table::MOST_AXES } else { 3 };

/// The normalized generator's matrices of a cluster at a point (module documentation, # The normalized generator), each
/// n × n row by row: S⁻¹, the held part Herm(H¹), E = H − Z₀ of the series' generator H = H_h − (i/2)L + H¹ (four parts
/// each: the value and the gradient along the three lab axes), the landing map S (1 + Y_L)⁻¹ and E₀ = H_h − (i/2)L − Z₀,
/// their values: `data[(4 matrix + part) n² + x n + y]` for the first three, `data[12 n² + …]` and `data[13 n² + …]`
/// for the next; then the first order's forms Y⁽ᵃ⁾ of each reference class, their values (`data[(14 + a) n² + …]`). The point they are of (the plan, its table's grid and the position) and the complex
/// multiply-adds that made them. A thread's last (`Model::quartic`); the rest is its scratch.
#[derive(Default)]
pub(crate) struct Quartic {
    key: Option<(u64, u64, [u64; 3])>,
    /// Whether the landing map is made (`Model::quartic`'s `landing`).
    landing: bool,
    pub(crate) data: Vec<C64>,
    pub(crate) work: f64,
    /// Whether `work` is counted: matrices a step's middle node made (`Model::node_normal`) are counted by the step's
    /// scattering, which reads them next (`Model::quartic`).
    counted: bool,
    sums: Vec<C64>,
    laid: Vec<f64>,
    /// The matrices `normal` is made of, and its scratch.
    arena: Vec<C64>,
    scratch: Vec<C64>,
}

/// The places of `Quartic::data`: S⁻¹, the held part, E (four parts each), the landing map, E₀, the forms Y⁽ᵃ⁾.
pub(crate) const BENT: usize = 0;
pub(crate) const HELD: usize = 4;
pub(crate) const OFFSET: usize = 8;
pub(crate) const LANDING: usize = 12;
pub(crate) const ORIGIN: usize = 13;
pub(crate) const FIRST: usize = 14;

thread_local! {
    static QUARTIC: RefCell<Quartic> = RefCell::new(Quartic::default());
}

/// The plans made so far: each one's own number (`Plan::id`).
static PLANS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// A cluster's assembly under a light set, worked out once (module documentation): the near forms, the field components and
/// their beams, the blocks of terms, the tones, the excited replicas and the photon classes.
#[derive(Clone, Debug)]
pub(crate) struct Plan {
    pub(crate) cluster: usize,
    pub(crate) lights: Lights,
    pub(crate) states: usize,
    pub(crate) near: NearLayout,
    /// The request beams the plan's field components sum.
    pub(crate) beams: Vec<usize>,
    pub(crate) scalars: Vec<Scalar>,
    blocks: Vec<Block>,
    /// Per block whether its base beat is exactly zero (the potential's static shifts).
    pub(crate) statics: Vec<bool>,
    /// The cluster's n² places as whole vectors of the dense terms' sums.
    padded: usize,
    /// The generator's dense terms -- per block, flag and tone its Hermitian values --, and those of a form with a Hermitian
    /// ρ: the Hermitian values of the flags that weigh in it. Each in the blocks' order.
    generator: Vec<Term>,
    form: Vec<Term>,
    /// The constant terms' values, and the constant part of a point's values of the generator's terms and of the forms'.
    constant: Vec<f64>,
    pub(crate) generator_start: Vec<f64>,
    pub(crate) form_start: Vec<f64>,
    pub(crate) links: Vec<Link>,
    /// The distinct residual beats of the terms' entries in the cluster's frames (rad/µs; the first is zero): at a fixed point
    /// the generator is Σ_m (its entries of tone m) e^{iΩ_m t}.
    tones: Vec<f64>,
    /// Each state's Doppler frame K_s (1/µm; cluster.rs, # Doppler frames): on a path r_h + v(t − t_m) the state is
    /// ψ_s = e^{iK_s·v(t − t_m)} x_s, so that x_s holds K_s·v on the diagonal and a plane wave q of an entry a ← b turns at
    /// q − (K_a − K_b). The frames of the least mean square of these under the plan's lights; exact where every pair of
    /// states holds one wave vector and the loops close.
    pub(crate) waves: Vec<[f64; 3]>,
    /// Whether the generator is the same at every time along every path: no entry beats in the frames, in time or along a path
    /// (every residual beat and wave vector exactly zero) -- the excited amplitudes' product as much as the Hermitian terms.
    still: bool,
    /// The complex multiply-adds at a node (module documentation, # Work): of the generator; of the force's form without its
    /// values' slopes; and of those per direction of them.
    node: [f64; 3],
    /// The excited replicas (module documentation, # The excited amplitude), their rows' number, the merged classes' cross
    /// terms; each place's reference class (eliminate.rs `Model::references`) and their number; and per response entry and
    /// reference class the place of its first-order response among the responding forms (T2 where the class is its place's
    /// own, else its mixed response).
    pub(crate) replicas: Vec<ReplicaPlan>,
    pub(crate) amplitude_len: usize,
    /// The decay's constant legs' vectors (a bare response, an explicit state's −e_x), its kernels (sparse: the identity,
    /// D, each merged pair of replicas' C_B; `DecayForm`) and the normalized generator's decay sums (`DecaySum`).
    pub(crate) legs: Vec<[C64; EXCITED]>,
    pub(crate) kernels: Vec<Vec<(usize, usize, C64)>>,
    pub(crate) sums: Vec<DecaySum>,
    pub(crate) references: Vec<usize>,
    pub(crate) reference_classes: usize,
    first_place: Vec<usize>,
    pub(crate) classes: Vec<ClassPlan>,
    /// The complex multiply-adds of a point's values from its exact near forms (`Site::formed`, `formed_work`): each near
    /// form laid onto the generator's values and, where it weighs in the form, onto the form's. And of the form's along
    /// each direction of the forms' slopes: the near forms that weigh in it. A table holds the values as its nodes do: a
    /// point read from one makes none (`Plan::near_read`).
    values: f64,
    sloped: f64,
    /// What the normalized generator of the cluster's fold reads (module documentation, # The normalized generator); none
    /// where no carrier is lit under these lights. And the plan's own number among the plans made.
    pub(crate) fourth: Option<Fourth>,
    pub(crate) id: u64,
    /// The generator's beating groups and the first of each tone pair among them (module documentation, # Exact tones).
    groups: Vec<Group>,
    firsts: Vec<usize>,
}

/// A block under construction: its factors, base beat and entries, per flag each entry's constant and near form, and which
/// flags hold anything.
struct Draft {
    left: Option<usize>,
    right: Option<usize>,
    beat: Beat,
    entries: Vec<(usize, usize)>,
    hermitian: [Vec<(C64, Option<(Operand, bool)>)>; 3],
    /// Per flag and entry the decay's part of the generator, −(i/2) Λ: a constant and the decay forms it takes (each
    /// conjugated or not, times its factor).
    decay: [Vec<(C64, Vec<(Operand, bool, C64)>)>; 4],
    used: [bool; 4],
}

/// A flag with its two sides exchanged, where a block's factors are (`Builder::block`).
fn flip(flag: usize, exchanged: bool) -> usize {
    if exchanged { [PLAIN, LEFT, RIGHT, BOTH][flag] } else { flag }
}

struct Builder<'a> {
    model: &'a Model,
    cluster: usize,
    lights: &'a Lights,
    orders: Orders,
    near: NearLayout,
    pair_index: HashMap<[usize; 6], usize>,
    decay_index: HashMap<DecayForm, usize>,
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
    /// Adds to a block's entry its part of the decay's −(i/2) Λ: a constant, or a decay form (conjugated where asked) times
    /// `factor`.
    fn decay(&mut self, (block, exchanged): (usize, bool), flag: usize, x: usize, y: usize, constant: C64, form: Option<(Operand, bool, C64)>) {
        let (at, flag) = (self.entry(block, x, y), flip(flag, exchanged));
        let d = &mut self.drafts[block];
        d.used[flag] = true;
        let slot = &mut d.decay[flag][at];
        slot.0 += constant;
        slot.1.extend(form);
    }
    /// Adds a Hermitian value to a block's entry: a constant and, where given, a near form (conjugated where the flag says).
    fn hermitian(&mut self, (block, exchanged): (usize, bool), flag: usize, x: usize, y: usize, constant: C64, form: Option<(Operand, bool)>) {
        let (at, flag) = (self.entry(block, x, y), flip(flag, exchanged));
        let d = &mut self.drafts[block];
        d.used[flag] = true;
        let slot = &mut d.hermitian[flag][at];
        slot.0 += constant;
        if form.is_some() {
            assert!(slot.1.is_none(), "an entry of a block takes one near form");
            slot.1 = form;
        }
    }

    /// Whether U^f_k'†T_k of a pair of (frequency, basis vector) can differ from zero on x ← y: the left coupling's excited
    /// states from x among those the right response reaches from y (none where the right frequency is no carrier). The
    /// frequency `tones.len()` (basis vector 0) is the explicit states' static couplings (eliminate.rs `Responses::held`,
    /// `Model::held_support`), beside which a carrier's coupling is its drives' folded at the atom (`Carrier::basis`: the
    /// fold's W, ELIMINATION v12).
    fn touches(&self, left: (usize, usize), right: (usize, usize), x: usize, y: usize) -> bool {
        let (model, held) = (self.model, self.model.tones.len());
        let reach = if right.0 == held {
            model.held_support(self.cluster, y, self.lights).1
        } else {
            let Some(carrier) = model.carrier_of(right.0) else { return false };
            model.reached(self.cluster, carrier, right.1, self.states()[y])
        };
        let coupled: [bool; EXCITED] = if left.0 == held {
            model.held_support(self.cluster, x, self.lights).0
        } else {
            let u = if right.0 == held { &model.carriers[model.carrier_of(left.0).expect("a carrier beside the explicit states' couplings")].basis[left.1] } else { &model.tones[left.0].coupled[0][left.1] };
            std::array::from_fn(|e| u[(GROUND + e, self.states()[x])] != C64::ZERO)
        };
        (0..EXCITED).any(|e| reach[e] && coupled[e])
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
    /// `Model::term_included`) --, per lit carrier the explicit drives with their adjoints, and the pairs of an explicit
    /// state's static couplings' leg.
    fn near(&mut self) {
        let model = self.model;
        let (states, n, g) = (self.states(), self.states().len(), self.ground());
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
        // The explicit states' static couplings to Q (eliminate.rs `Responses::held`; ELIMINATION v12): the same
        // contraction −½(W′_l T_r + T_l†W_r) of every pair with an explicit place's leg -- an explicit place x and a
        // ground place y of a lit carrier's basis vector, x ← y at the carrier's beat negated (the absorbed leg its
        // Doppler partner's) and y ← x at its beat (the emitted one's), and two explicit places, static --, each entry
        // by the one rule.
        let held = model.tones.len();
        let holds = (g..n).any(|x| model.held_support(self.cluster, x, self.lights).1.iter().any(|r| *r));
        if holds {
            for cc in model.carriers.iter().filter(|cc| self.lights.on(cc.frequency)) {
                let w = cc.frequency;
                for k in 0..cc.basis.len() {
                    let (Some(plain), Some(conjugated)) = (self.scalar(w, None, k, false), self.scalar(w, None, k, true)) else { continue };
                    for x in g..n {
                        for y in 0..g {
                            let absorbed = cc.beat.negated();
                            if model.term_included(self.cluster, x, y, &absorbed) {
                                if let Some(at) = self.pair((held, 0), (w, k), x, y) {
                                    let block = self.block(None, Some(plain), &absorbed);
                                    self.hermitian(block, PLAIN, x, y, C64::ZERO, Some((Operand::Pair(at), false)));
                                    self.hermitian(block, RIGHT, x, y, C64::ZERO, Some((Operand::Doppler(at), false)));
                                }
                            }
                            if model.term_included(self.cluster, y, x, &cc.beat) {
                                if let Some(at) = self.pair((w, k), (held, 0), y, x) {
                                    let exchanged = self.pair((held, 0), (w, k), x, y).expect("a pair's entry comes with its adjoint's");
                                    let block = self.block(Some(conjugated), None, &cc.beat);
                                    self.hermitian(block, PLAIN, y, x, C64::ZERO, Some((Operand::Pair(at), false)));
                                    self.hermitian(block, LEFT, y, x, C64::ZERO, Some((Operand::Doppler(exchanged), true)));
                                }
                            }
                        }
                    }
                }
            }
            let zero = model.axes.zero();
            for x in g..n {
                for y in g..n {
                    if model.term_included(self.cluster, x, y, &zero) {
                        if let Some(at) = self.pair((held, 0), (held, 0), x, y) {
                            let still = self.block(None, None, &zero);
                            self.hermitian(still, PLAIN, x, y, C64::ZERO, Some((Operand::Pair(at), false)));
                        }
                    }
                }
            }
        }
        // The reference, where a carrier is lit or an explicit state couples to Q (module documentation, # The normalized
        // generator).
        if holds || model.carriers.iter().any(|cc| self.lights.on(cc.frequency)) {
            self.near.reference = n;
        }
        for cc in model.carriers.iter() {
            if !self.lights.on(cc.frequency) {
                continue;
            }
            let (component, tone) = (Component { frequency: cc.frequency, counter: false }, &model.tones[cc.frequency]);
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

    /// The excited replicas under the plan's lights (`ReplicaPlan`; photon.rs `Replica`): of each of the model's, its lit
    /// channels' responses -- a carrier's at the atom by the near forms' response entries, a bare component's by its
    /// constant response -- and its explicit states, on its rows: first its own, the excited states a carrier's response at
    /// the atom reaches and its explicit states, each its unit vector; then a row per bare response, the response itself at
    /// its one place -- what a state's photons are read on (photon.rs, # A state's photons at a point; the decay is the
    /// plan's pair forms, `Builder::decays`). The response entries are made here (`NearLayout::responses`): per lit carrier,
    /// basis vector and ground place the excited states its response reaches.
    fn replicas(&mut self) -> Vec<ReplicaPlan> {
        let model = self.model;
        let (states, g) = (self.states(), self.ground());
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
        // The explicit states' static couplings' responses (eliminate.rs `Responses::held`), in the slot after the carriers.
        let held = model.carriers.len();
        for x in g..states.len() {
            let reach = model.held_support(self.cluster, x, self.lights).1;
            self.near.responses.extend((0..EXCITED).filter(|&e| reach[e]).map(|e| [held, 0, e, x]));
        }
        let index: HashMap<[usize; 4], usize> = self.near.responses.iter().enumerate().map(|(i, r)| (*r, i)).collect();
        let mut at = 0;
        let mut out = vec![];
        for replica in &model.replicas[self.cluster] {
            // The rows of its own: the near responses' and the explicit states'.
            let mut own: Vec<usize> = vec![];
            let row = |own: &mut Vec<usize>, e: usize| {
                own.iter().position(|&x| x == e).unwrap_or_else(|| {
                    own.push(e);
                    own.len() - 1
                })
            };
            let (mut near, mut bare, mut explicit) = (vec![], vec![], vec![]);
            let mut vectors: Vec<(usize, usize, [C64; EXCITED])> = vec![];
            for &(channel, y) in &replica.channels {
                match channel {
                    Channel::Near { carrier } => {
                        let cc = &model.carriers[carrier];
                        if !self.lights.on(cc.frequency) {
                            continue;
                        }
                        for k in 0..cc.basis.len() {
                            let Some(scalar) = self.scalar(cc.frequency, None, k, false) else { continue };
                            for &e in &replica.rows {
                                if let Some(&i) = index.get(&[carrier, k, e, y]) {
                                    near.push((i, Some(scalar), row(&mut own, e), y));
                                }
                            }
                        }
                    }
                    Channel::Bare { frequency, side } => {
                        if !self.lights.on(frequency) {
                            continue;
                        }
                        for (k, response) in model.tones[frequency].bare_response[side].iter().enumerate() {
                            let Some(scalar) = self.scalar(frequency, None, k, side == 1) else { continue };
                            let v: [C64; EXCITED] = std::array::from_fn(|e| response[(e, states[y])]);
                            if v.iter().any(|z| *z != C64::ZERO) {
                                vectors.push((scalar, y, v));
                            }
                        }
                    }
                }
            }
            for &y in &replica.explicit {
                explicit.push((row(&mut own, states[y] - GROUND), y));
                for &e in &replica.rows {
                    if let Some(&i) = index.get(&[held, 0, e, y]) {
                        near.push((i, None, row(&mut own, e), y));
                    }
                }
            }
            let mut basis: Vec<[C64; EXCITED]> = own.iter().map(|&e| std::array::from_fn(|x| if x == e { C64::ONE } else { C64::ZERO })).collect();
            for (scalar, y, v) in vectors {
                bare.push((scalar, basis.len(), y, C64::ONE));
                basis.push(v);
            }
            let size = basis.len();
            out.push(ReplicaPlan { size, basis, at, near, bare, explicit });
            at += size;
        }
        out
    }

    /// The decay's pair forms (module documentation, # The excited amplitude; `DecayForm`): Λ = Σ_ρ 𝒯_ρ†D𝒯_ρ is bilinear
    /// in the field components, 𝒯_ρ = Σ_legs (e_l a_l − e^w_l T2_l) of its legs (`DecayLeg`: a carrier's response at the atom on
    /// a basis vector, with its Doppler partner; a bare component's constant response; an explicit state's −e_x), so per
    /// pair of a replica's legs -- the left one's place x, the right one's y -- the generator's −(i/2)Λ takes on the block
    /// of the left component conjugated and the right one (an explicit leg has none) the products a_l†D a_r (`PLAIN`),
    /// −a_l†D T2_r (`RIGHT`), −T2_l†D a_r (`LEFT`: the exchanged pair's `RIGHT` form conjugated) and T2_l†D T2_r
    /// (`BOTH`), each a decay form where a leg is near and a constant where both are constants -- the shifts' blocks and
    /// flags, one set of complex pair forms with them. A merged class's two feeds' replicas take the same of their legs
    /// with its C_B (`Plan::kernels`), at their beat. Of the near legs of one replica also the metric's T_l†T_r and the
    /// loss's T_l†D T_r on the plain components (`Plan::sums`: X and L, the normalization's); at the series' first order
    /// the first order's decay -- 𝒯⁽¹⁾ = Σ_a M⁽ᵃ⁾ E P_a: per reference class a the forms of 𝒯⁽⁰⁾†K M⁽ᵃ⁾ (its plain and its
    /// left partner's product) and of M⁽ᵇ⁾†K M⁽ᵃ⁾, M⁽ᵃ⁾ at a place of class a its partner T2, else its mixed response.
    /// Returns the constant legs' vectors, the kernels (sparse: e, f, K[e, f]) and the sums.
    fn decays(&mut self, references: &[usize]) -> (Vec<[C64; EXCITED]>, Vec<Vec<(usize, usize, C64)>>, Vec<DecaySum>) {
        let model = self.model;
        let states = self.states();
        let index: HashMap<[usize; 4], usize> = self.near.responses.iter().enumerate().map(|(i, r)| (*r, i)).collect();
        let mixed: HashMap<[usize; 2], usize> = self.near.mixed.iter().enumerate().map(|(m, &key)| (key, m)).collect();
        // The kernels: the identity, D, and each merged pair of replicas' C_B over the landing places both feed.
        let dense = |m: &[C64]| -> Vec<(usize, usize, C64)> { (0..EXCITED * EXCITED).filter(|&i| m[i] != C64::ZERO).map(|i| (i / EXCITED, i % EXCITED, m[i])).collect() };
        let identity: Vec<C64> = (0..EXCITED * EXCITED).map(|i| if i / EXCITED == i % EXCITED { C64::ONE } else { C64::ZERO }).collect();
        let decay: Vec<C64> = (0..EXCITED * EXCITED).map(|i| model.decay[(i / EXCITED, i % EXCITED)]).collect();
        let mut full: Vec<Vec<C64>> = vec![identity, decay];
        let mut merged: HashMap<(usize, usize), usize> = HashMap::new();
        for class in &model.classes[self.cluster] {
            let landing = &model.clusters[class.landing];
            for (i, &(r1, bp1, _)) in class.feeds.iter().enumerate() {
                for &(r2, bp2, _) in &class.feeds[i + 1..] {
                    if bp1 != bp2 || r1 == r2 {
                        continue;
                    }
                    let b = landing.states[bp1];
                    for key in [(r1, r2), (r2, r1)] {
                        let k = *merged.entry(key).or_insert_with(|| {
                            full.push(vec![C64::ZERO; EXCITED * EXCITED]);
                            full.len() - 1
                        });
                        for e in 0..EXCITED {
                            for f in 0..EXCITED {
                                full[k][e * EXCITED + f] += (0..3).map(|a| model.emission[a][(b, GROUND + e)].conj() * model.emission[a][(b, GROUND + f)]).sum::<C64>();
                            }
                        }
                    }
                }
            }
        }
        let kernels: Vec<Vec<(usize, usize, C64)>> = full.iter().map(|m| dense(m)).collect();
        // Each replica's legs: (leg, its component, place, beat); the constant ones' vectors.
        let mut constants: Vec<[C64; EXCITED]> = vec![];
        let mut legs: Vec<Vec<(DecayLeg, Option<usize>, usize, Beat)>> = vec![];
        for replica in &model.replicas[self.cluster] {
            let mut of = vec![];
            for &(channel, y) in &replica.channels {
                let beat = model.frequency(model.channel_component(channel));
                match channel {
                    Channel::Near { carrier } => {
                        let cc = &model.carriers[carrier];
                        if !self.lights.on(cc.frequency) {
                            continue;
                        }
                        for k in 0..cc.basis.len() {
                            let Some(scalar) = self.scalar(cc.frequency, None, k, false) else { continue };
                            if (0..EXCITED).any(|e| index.contains_key(&[carrier, k, e, y])) {
                                of.push((DecayLeg::Near { carrier, k }, Some(scalar), y, beat.clone()));
                            }
                        }
                    }
                    Channel::Bare { frequency, side } => {
                        if !self.lights.on(frequency) {
                            continue;
                        }
                        for (k, response) in model.tones[frequency].bare_response[side].iter().enumerate() {
                            let Some(scalar) = self.scalar(frequency, None, k, side == 1) else { continue };
                            let v: [C64; EXCITED] = std::array::from_fn(|e| response[(e, states[y])]);
                            if v.iter().any(|z| *z != C64::ZERO) {
                                constants.push(v);
                                of.push((DecayLeg::Constant(constants.len() - 1), Some(scalar), y, beat.clone()));
                            }
                        }
                    }
                }
            }
            for &x in &replica.explicit {
                constants.push(std::array::from_fn(|e| if GROUND + e == states[x] { -C64::ONE } else { C64::ZERO }));
                of.push((DecayLeg::Constant(constants.len() - 1), None, x, model.axes.zero()));
                // Its static couplings' response (eliminate.rs `Responses::held`): no component, no Doppler partner.
                let leg = DecayLeg::Near { carrier: model.carriers.len(), k: 0 };
                if (0..EXCITED).any(|e| index.contains_key(&[model.carriers.len(), 0, e, x])) {
                    of.push((leg, None, x, model.axes.zero()));
                }
            }
            legs.push(of);
        }
        // A near leg's first-order order at a place of class a: its partner at its own class, its mixed response else
        // (none where the leg has none there).
        let first = |leg: DecayLeg, place: usize, a: usize| -> Option<usize> {
            let DecayLeg::Near { carrier, k } = leg else { return None };
            if references.is_empty() || references[place] == a {
                return Some(1);
            }
            (0..EXCITED).filter_map(|e| index.get(&[carrier, k, e, place])).find_map(|&i| mixed.get(&[i, a])).map(|_| 2 + a)
        };
        let half = C64::new(0.0, 0.5);
        let held = self.orders.series >= 1;
        let count = references.iter().max().map_or(1, |m| m + 1);
        let mut sums = vec![];
        let mut pairs: Vec<(usize, usize, usize)> = (0..legs.len()).map(|r| (r, r, 1)).collect();
        let mut crossings: Vec<((usize, usize), usize)> = merged.into_iter().collect();
        crossings.sort();
        pairs.extend(crossings.into_iter().map(|((r1, r2), k)| (r1, r2, k)));
        for (r1, r2, kernel) in pairs {
            for li in 0..legs[r1].len() {
                for ri in 0..legs[r2].len() {
                    let ((l, ls, x, lb), (r, rs, y, rb)) = (legs[r1][li].clone(), legs[r2][ri].clone());
                    let left = match ls {
                        None => None,
                        Some(i) => {
                            let s = &self.scalars[i];
                            let (frequency, component, conjugate) = (s.frequency, s.component, s.conjugate);
                            let Some(c) = self.scalar(frequency, None, component, !conjugate) else { continue };
                            Some(c)
                        }
                    };
                    let block = self.block(left, rs, &lb.plus(-1, &rb));
                    // Near: a response at the atom (its forms, its first order); Doppler: of a carrier's beams.
                    let (ln, rn) = (matches!(l, DecayLeg::Near { .. }), matches!(r, DecayLeg::Near { .. }));
                    let doppler = |leg: DecayLeg| matches!(leg, DecayLeg::Near { carrier, .. } if carrier < model.carriers.len());
                    let (ld, rd) = (doppler(l), doppler(r));
                    let of = |lo: usize, ro: usize| DecayForm { left: (l, lo), right: (r, ro), x, y, kernel };
                    if let (DecayLeg::Constant(i), DecayLeg::Constant(j)) = (l, r) {
                        let v: C64 = kernels[kernel].iter().map(|&(e, f, k)| constants[i][e].conj() * k * constants[j][f]).sum();
                        self.decay(block, PLAIN, x, y, -half * v, None);
                        continue;
                    }
                    let plain = self.decay_form(of(0, 0));
                    self.decay(block, PLAIN, x, y, C64::ZERO, Some((Operand::Decay(plain), false, -half)));
                    if rd {
                        let right = self.decay_form(of(0, 1));
                        self.decay(block, RIGHT, x, y, C64::ZERO, Some((Operand::Decay(right), false, half)));
                    }
                    if ld {
                        // The exchanged pair's RIGHT form, conjugated.
                        let exchanged = self.decay_form(DecayForm { left: (r, 0), right: (l, 1), x: y, y: x, kernel });
                        self.decay(block, LEFT, x, y, C64::ZERO, Some((Operand::Decay(exchanged), true, half)));
                    }
                    if ld && rd {
                        let both = self.decay_form(of(1, 1));
                        self.decay(block, BOTH, x, y, C64::ZERO, Some((Operand::Decay(both), false, -half)));
                    }
                    if ln && rn && kernel == 1 {
                        let metric = self.decay_form(DecayForm { kernel: 0, ..of(0, 0) });
                        sums.push(DecaySum { kind: Sum::Metric, block: block.0, flag: PLAIN, x, y, tone: 0, form: metric, slot: 0, factor: C64::ONE });
                        sums.push(DecaySum { kind: Sum::Loss, block: block.0, flag: PLAIN, x, y, tone: 0, form: plain, slot: 0, factor: C64::ONE });
                    }
                    // The first order's decay: 𝒯⁽⁰⁾†K M⁽ᵃ⁾ (plain, and the left partner's −T2†K M⁽ᵃ⁾), M⁽ᵇ⁾†K M⁽ᵃ⁾.
                    if held && rn {
                        for a in 0..count {
                            let Some(ra) = first(r, y, a) else { continue };
                            let f = self.decay_form(of(0, ra));
                            sums.push(DecaySum { kind: Sum::First, block: block.0, flag: PLAIN, x, y, tone: 0, form: f, slot: a, factor: C64::ONE });
                            if ld {
                                let f = self.decay_form(of(1, ra));
                                sums.push(DecaySum { kind: Sum::First, block: block.0, flag: flip(LEFT, block.1), x, y, tone: 0, form: f, slot: a, factor: -C64::ONE });
                            }
                            if ln {
                                for b in 0..count {
                                    let Some(lb) = first(l, x, b) else { continue };
                                    let f = self.decay_form(of(lb, ra));
                                    sums.push(DecaySum { kind: Sum::Second, block: block.0, flag: PLAIN, x, y, tone: 0, form: f, slot: b * count + a, factor: C64::ONE });
                                }
                            }
                        }
                    }
                }
            }
        }
        // The forms the normalized generator's sums read first (`NearLayout::normalizing`: read at a point with their
        // slopes), then those the generator's terms alone take (laid in its values).
        let mut order: Vec<usize> = vec![];
        for sum in &sums {
            if !order.contains(&sum.form) {
                order.push(sum.form);
            }
        }
        self.near.normalizing_decays = order.len();
        order.extend((0..self.near.decays.len()).filter(|i| !sums.iter().any(|sum| sum.form == *i)));
        let mut moved = vec![0; order.len()];
        for (new, &old) in order.iter().enumerate() {
            moved[old] = new;
        }
        self.near.decays = order.iter().map(|&old| self.near.decays[old]).collect();
        self.decay_index = self.near.decays.iter().enumerate().map(|(i, d)| (*d, i)).collect();
        for sum in sums.iter_mut() {
            sum.form = moved[sum.form];
        }
        for d in self.drafts.iter_mut() {
            for slot in d.decay.iter_mut().flatten() {
                for (operand, _, _) in slot.1.iter_mut() {
                    if let Operand::Decay(i) = operand {
                        *i = moved[*i];
                    }
                }
            }
        }
        (constants, kernels, sums)
    }

    /// A decay form's place among the plan's (`NearLayout::decays`), found or made.
    fn decay_form(&mut self, f: DecayForm) -> usize {
        if let Some(&at) = self.decay_index.get(&f) {
            return at;
        }
        self.decay_index.insert(f, self.near.decays.len());
        self.near.decays.push(f);
        self.near.decays.len() - 1
    }

    /// The first order's forms of other places' references (`NearLayout::crossed`, `NearLayout::mixed`), where the
    /// evaluation takes the series' first order or the normalization's (the landing's overlap) and the cluster's places
    /// have more than one reference class: per pair entry and per response entry, each class that is not its right
    /// place's.
    /// An explicit state's leg on the right has no Doppler partner, so its forms are crossed at every class, its own
    /// too (Y = W′R₀(z_y)²K there, which a carrier's right leg reads from its partner's term).
    fn crossed(&mut self, references: &[usize], count: usize) {
        let o = self.orders;
        if o.series == 0 && o.normalization == 0 {
            return;
        }
        let held = self.model.tones.len();
        for (i, p) in self.near.pairs.iter().enumerate() {
            if p[2] == held {
                self.near.crossed.extend((0..count.max(1)).map(|a| [i, a]));
            } else if count > 1 && self.model.carrier_of(p[2]).is_some() {
                self.near.crossed.extend((0..count).filter(|&a| a != references[p[5]]).map(|a| [i, a]));
            }
        }
        if count <= 1 {
            return;
        }
        if o.series >= 1 {
            for (i, r) in self.near.responses.iter().enumerate() {
                self.near.mixed.extend((0..count).filter(|&a| a != references[r[3]]).map(|a| [i, a]));
            }
        }
    }

    /// The plan: the responses' entries, the near forms' places, the blocks' slots, beam pairs and parts, the replicas and
    /// the classes.
    fn finish(mut self) -> Plan {
        let replicas = self.replicas();
        let model = self.model;
        let cluster = &model.clusters[self.cluster];
        let n = cluster.states.len();
        let references = model.references(self.cluster, self.lights);
        let reference_classes = references.iter().max().map_or(0, |m| m + 1);
        self.crossed(&references, reference_classes);
        let (legs, kernels, mut sums) = self.decays(&references);
        // The model's classes on the plan's replicas, their entries the emission's rows on the feeds' rows.
        let mut classes = vec![];
        for class in &model.classes[self.cluster] {
            let landing = &model.clusters[class.landing];
            let states = landing.states.len();
            let rows = (3 * states).div_ceil(LANES) * LANES;
            let mut entries = vec![];
            let mut absorbed = 0.0_f64;
            for &(r, bp, ref phase) in &class.feeds {
                let replica = &model.replicas[self.cluster][r];
                for &(channel, _) in &replica.channels {
                    let frequency = model.channel_component(channel).frequency;
                    if self.lights.on(frequency) {
                        absorbed = model.light.frequencies[frequency].beams.iter().map(|&l| model.light.beams[l].wave_number).fold(absorbed, f64::max);
                    }
                }
                // The emission's rows on the replica's rows (photon.rs, # A state's photons at a point): an own row's an
                // entry of C_a, a bare row's C_a on its response.
                let b = landing.states[bp];
                for a in 0..3 {
                    for (j, q) in replicas[r].basis.iter().enumerate() {
                        let c: C64 = q.iter().enumerate().map(|(e, z)| model.emission[a][(b, GROUND + e)] * *z).sum();
                        if c != C64::ZERO {
                            entries.push((laid(a * states + bp, 0, 4), replicas[r].at + j, model.omega(phase), c));
                        }
                    }
                }
            }
            let turning = entries.iter().filter(|e| e.2 != 0.0).count();
            classes.push(ClassPlan { landing: class.landing, wave_number: model.class_wave_number(self.cluster, class), absorbed, states, rows, entries, turning });
        }
        let amplitude_len = replicas.iter().map(|r| r.size).sum();
        let near = self.near;
        // The responses' first-order places (`Plan::first_place`): T2 where the class is the place's own, else the mixed.
        let (len, start, kinds) = (near.responses.len(), near.responding().start, reference_classes.max(1));
        let mixed: HashMap<[usize; 2], usize> = near.mixed.iter().enumerate().map(|(m, &key)| (key, m)).collect();
        let first_place: Vec<usize> = (0..len * kinds)
            .map(|x| {
                let (i, a) = (x / kinds, x % kinds);
                // A class that is not the place's own has its mixed response where the evaluation takes the series' first
                // order; none is read where it does not.
                if reference_classes == 0 || a == references[near.responses[i][3]] { start + len + i } else { mixed.get(&[i, a]).map_or(start + len + i, |&m| start + 2 * len + m) }
            })
            .collect();
        // The entries' tones, and every plane wave of every entry at a tone, once: (x, y, tone, its residual wave vector in
        // the tree's frames, exactly) -- the wave vector of a pair of the block's two components' beams.
        let mut tone_index: HashMap<Beat, usize> = HashMap::from([(model.axes.zero(), 0)]);
        let mut tones = vec![0.0];
        let mut held: Vec<(usize, usize, usize, Wave)> = vec![];
        let mut entry_tones: Vec<Vec<usize>> = vec![];
        let scalar_waves = |s: Option<usize>| -> Vec<Wave> {
            s.map_or(vec![model.no_wave()], |i| self.scalars[i].beams.iter().map(|&(at, _)| model.no_wave().plus(if self.scalars[i].conjugate { -1 } else { 1 }, &model.beam_wave(self.beams[at]))).collect())
        };
        for d in &self.drafts {
            let (left, right) = (scalar_waves(d.left), scalar_waves(d.right));
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
        // The decay sums' tones: their entries'.
        for sum in sums.iter_mut() {
            let e = self.drafts[sum.block].entries.iter().position(|&xy| xy == (sum.x, sum.y)).expect("a decay sum's entry");
            sum.tone = entry_tones[sum.block][e];
        }
        // The Doppler frames (cluster.rs, # Doppler frames): the tree's; and whether anything beats -- a residual beat in
        // time, or along a path a residual wave vector, of any entry's: the Hermitian terms' and the decay's alike.
        let waves = cluster.waves.clone();
        let still = held.iter().all(|h| h.2 == 0 && h.3.is_zero());
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
                match adjoint.filter(|&t| self.drafts[t].used.iter().any(|used| *used)) {
                    Some(t) if t > b => [2.0; 3],
                    Some(t) if t < b => [0.0; 3],
                    Some(_) => [1.0, 2.0, 0.0],
                    None => [1.0; 3],
                }
            })
            .collect();
        // The dense terms (module documentation, # Dense terms): per block, flag and tone the generator's Hermitian values and,
        // of the flags that weigh in a form, the form's.
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
            for flag in (0..4).filter(|&flag| d.used[flag]) {
                let weighs = flag < 3 && weights[b][flag] > 0.0;
                merged += d.entries.len();
                if weighs {
                    forms += d.entries.len();
                }
                for &tone in &distinct {
                    // The generator's entry: its Hermitian value and its decay's; the form's: the Hermitian one alone.
                    let hermitian = |e: usize| if flag < 3 { d.hermitian[flag][e] } else { (C64::ZERO, None) };
                    let at_tone = (0..d.entries.len()).filter(|&e| entry_tones[b][e] == tone);
                    let own: Vec<(usize, C64, bool)> = at_tone.clone().map(|e| (place(e), hermitian(e).0 + d.decay[flag][e].0, hermitian(e).1.is_some() || !d.decay[flag][e].1.is_empty())).collect();
                    generator.push(&mut constant, (b, flag, tone), 1.0, &own);
                    if weighs {
                        let own: Vec<(usize, C64, bool)> = at_tone.map(|e| (place(e), hermitian(e).0, hermitian(e).1.is_some())).collect();
                        form.push(&mut constant, (b, flag, tone), weights[b][flag], &own);
                    }
                }
            }
        }
        // The near forms' places among a point's values.
        let mut links = vec![];
        for (b, d) in self.drafts.iter().enumerate() {
            for flag in (0..4).filter(|&flag| d.used[flag]) {
                for e in 0..d.entries.len() {
                    let (key, place) = ((b, flag, entry_tones[b][e]), d.entries[e].0 * n + d.entries[e].1);
                    let hermitian = if flag < 3 { d.hermitian[flag][e].1.map(|(operand, conjugate)| (operand, conjugate, C64::ONE, true)) } else { None };
                    for (operand, conjugate, factor, formed) in hermitian.into_iter().chain(d.decay[flag][e].1.iter().map(|&(o, c, f)| (o, c, f, false))) {
                        links.push(Link {
                            block: b,
                            flag,
                            place: d.entries[e],
                            tone: entry_tones[b][e],
                            operand: near.at(operand),
                            conjugate,
                            factor,
                            hermitian: formed,
                            generator: generator.place(key, place).expect("a near form's term"),
                            form: if formed { form.place(key, place) } else { None },
                        });
                    }
                }
            }
        }
        // A node's counted work: the field components from their beams (plain and Doppler-weighted; with their gradients,
        // fourfold), each block's entries per flag -- in a form those of the Hermitian flags taken (`Term::weight`) --, the
        // frames' phases on the n² entries -- the decay's pair forms among the blocks' entries.
        let fields: usize = self.scalars.iter().map(|s| 2 * s.beams.len()).sum();
        let node = [(fields + merged + 2 * n * n) as f64, (4 * fields + forms + n * n) as f64, forms as f64];
        let statics = self.drafts.iter().map(|d| d.beat.is_zero()).collect();
        let sloped = links.iter().filter(|link| link.form.is_some()).count();
        let values = (links.len() + sloped) as f64;
        // The generator's beating groups (module documentation, # Exact tones): each piece of a dense term -- the generator's,
        // the form's -- by its tone and its two beams' plane waves' residual wave vector in the cluster's Doppler frames,
        // exactly. The static pieces, of no beat in time or along a path, in none.
        let mut beats_of = vec![model.axes.zero(); tones.len()];
        for (beat, &t) in &tone_index {
            beats_of[t] = beat.clone();
        }
        let (mut found, mut groups, mut keys): (HashMap<(Beat, Wave), usize>, Vec<Group>, Vec<(Beat, Wave)>) = Default::default();
        for (formed, terms) in [(false, &generator.terms), (true, &form.terms)] {
            for (index, term) in terms.iter().enumerate() {
                let d = &self.drafts[term.block];
                let sides = |s: Option<usize>| -> Vec<(Option<usize>, Wave)> {
                    s.map_or(vec![(None, model.no_wave())], |i| {
                        let scalar = &self.scalars[i];
                        scalar.beams.iter().enumerate().map(|(j, &(at, _))| (Some(j), model.no_wave().plus(if scalar.conjugate { -1 } else { 1 }, &model.beam_wave(self.beams[at])))).collect()
                    })
                };
                let (left, right) = (sides(d.left), sides(d.right));
                // The term's own entries: of its block's at its tone, those its flag holds (`Laid::push`).
                for &place in &term.held {
                    let (x, y) = (place / n, place % n);
                    let (offset, frames) = (place - term.first, cluster.waves_exact[x].plus(-1, &cluster.waves_exact[y]));
                    for (l, lw) in &left {
                        for (r, rw) in &right {
                            let wave = lw.plus(1, rw).plus(-1, &frames);
                            if beats_of[term.tone].is_zero() && wave.is_zero() {
                                continue;
                            }
                            let key = (beats_of[term.tone].clone(), wave);
                            let g = match found.get(&key) {
                                Some(&g) => g,
                                None => {
                                    let value = model.wave_value(&key.1);
                                    groups.push(Group { tone: term.tone, wave: value, key: vec![], partner: None, first: true, generator: vec![], form: vec![] });
                                    (found.insert(key.clone(), groups.len() - 1), keys.push(key));
                                    groups.len() - 1
                                }
                            };
                            let piece = Piece { term: index, offset, left: *l, right: *r };
                            if formed { groups[g].form.push(piece) } else { groups[g].generator.push(piece) }
                        }
                    }
                }
            }
        }
        // The groups' labels: their beats' and wave vectors' integers over one denominator, so that sums are exact.
        let gcd = |mut a: i64, mut b: i64| {
            while b != 0 {
                (a, b) = (b, a % b);
            }
            a.abs()
        };
        let common = keys.iter().flat_map(|(beat, wave)| [beat.d, wave.d]).fold(1, |l, d| l / gcd(l, d) * d);
        for (group, (beat, wave)) in groups.iter_mut().zip(&keys) {
            group.key = beat.n.iter().map(|x| x * (common / beat.d)).chain(wave.n.iter().map(|x| x * (common / wave.d))).collect();
        }
        for (g, (beat, wave)) in keys.iter().enumerate() {
            let adjoint = (beat.negated(), wave.negated());
            groups[g].partner = found.get(&adjoint).copied();
            groups[g].first = groups[g].partner.is_none() || keys[g] > adjoint;
        }
        let firsts = (0..groups.len()).filter(|&g| groups[g].first).collect();
        Plan {
            cluster: self.cluster,
            lights: self.lights.clone(),
            states: n,
            near,
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
            waves,
            still,
            node,
            tones,
            replicas,
            amplitude_len,
            legs,
            kernels,
            sums,
            references,
            reference_classes,
            first_place,
            classes,
            values,
            sloped: sloped as f64,
            fourth: None,
            id: PLANS.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            groups,
            firsts,
        }
    }
}

impl Plan {
    /// What a node and a point of the plan are made of (the step's cost harness, step/tests.rs): its states, beams, field
    /// components, blocks and tones; the generator's and the force's dense terms with their reals; a point's near forms
    /// and the reals of its values and forms; the replicas with their rows and the classes with their entries; the
    /// normalized generator.
    #[cfg(test)]
    pub(crate) fn described(&self) -> String {
        let reals = |terms: &[Term]| terms.iter().map(|t| 2 * t.len).sum::<usize>();
        let near = |terms: &[Term]| terms.iter().filter(|t| t.near).count();
        let entries: usize = self.classes.iter().map(|c| c.entries.len()).sum();
        let rows: usize = self.classes.iter().map(|c| c.rows).sum();
        format!(
            "n {} beams {} components {} blocks {} tones {} | generator terms {} ({} near, {} reals) form terms {} ({} near, {} reals) | point: near forms {} values {} forms {} | replicas {} rows {} (own {} near {} bare {} explicit {}) decay forms {} (normalizing {}) | classes {} entries {} rows {} | fourth {} references {}",
            self.states,
            self.beams.len(),
            self.scalars.len(),
            self.blocks.len(),
            self.tones.len(),
            self.generator.len(),
            near(&self.generator),
            reals(&self.generator),
            self.form.len(),
            near(&self.form),
            reals(&self.form),
            self.near.len(),
            self.generator_start.len(),
            self.form_start.len(),
            self.replicas.len(),
            self.amplitude_len,
            self.replicas.iter().map(|r| r.size - r.bare.len()).sum::<usize>(),
            self.replicas.iter().map(|r| r.near.len()).sum::<usize>(),
            self.replicas.iter().map(|r| r.bare.len()).sum::<usize>(),
            self.replicas.iter().map(|r| r.explicit.len()).sum::<usize>(),
            self.near.decays.len(),
            self.near.normalizing_decays,
            self.classes.len(),
            entries,
            rows,
            self.fourth.is_some(),
            self.reference_classes
        )
    }

    /// Whether the plan lays forms of two places' resolvents (`NearLayout::crossed`, `NearLayout::mixed`: the first order's
    /// and the landing's, where the cluster's places have more than one reference class): what the elimination makes
    /// for it (eliminate.rs `Model::eliminate`).
    pub(crate) fn mixes(&self) -> bool {
        !self.near.mixed.is_empty() || self.near.crossed.iter().any(|&[i, a]| !self.references.is_empty() && self.references[self.near.pairs[i][5]] != a)
    }

    /// The plan of a cluster under a light set at the evaluation's orders `orders` (elim.rs `Orders`: the run's, the
    /// model's series' -- `Model::plan` --, or the readings' next ones, `Model::plan_at`).
    fn new(model: &Model, cluster: usize, lights: &Lights, orders: Orders) -> Plan {
        let mut b = Builder { model, cluster, lights, orders, near: NearLayout::default(), pair_index: HashMap::new(), decay_index: HashMap::new(), beams: vec![], scalars: vec![], drafts: vec![] };
        b.constants();
        b.near();
        let mut plan = b.finish();
        plan.fourth = Fourth::new(&plan, orders);
        plan
    }
    /// The counts at a node (`Middle::work`): of the generator -- its dense terms, the decay's pair forms among them --
    /// with the normalized part of the node's own forms (`Plan::node_normal_work`),
    /// of the force's form with its values' slopes along `directions` directions (a table's axes) and the held part's, and
    /// of the tones'. Per node of a step's (step.rs `count`: one on a still path, else three, the middle among them): the
    /// middle's normalized part is the point's own -- the scattering's matrices with their gradients, counted there
    /// (`Model::node_normal`) -- and its held part's force its gradient's trace; the others make their matrices, and their
    /// held part's force is its form (`Plan::held_force_work`). The tones are the middle's.
    pub(crate) fn costs(&self, directions: usize) -> [f64; 3] {
        let base = self.node[1] + directions as f64 * self.node[2];
        let Some(fourth) = &self.fourth else { return [self.node[0], base, base] };
        let (nodes, off_middle) = if self.still { (1.0, 0.0) } else { (3.0, 2.0) };
        let share = off_middle / nodes;
        let [made, products] = self.node_normal_work(fourth);
        let trace = if fourth.held() { (3 * fourth.n * fourth.n) as f64 } else { 0.0 };
        let held = if fourth.held() { share * self.held_force_work(fourth, directions) + (1.0 - share) * trace } else { 0.0 };
        [self.node[0] + share * made + products, base + held, base + trace]
    }
    /// The counted complex multiply-adds of the normalized part at a node (`Model::node_normal`, `Middle::generator`):
    /// the node's matrices, of their values alone -- the metric's decay sum with S⁻¹ by Horner's rule (J − 1 products of
    /// n × n matrices), or where the held part is anything `quartic_work`'s --, made off the middle where the part is the
    /// point's (`Plan::costs`); and at every node where it bends the decay's parting from the terms and its two products
    /// with S⁻¹, and at the series' order 1 the first order's decay (`Fourth::first_decay_work`).
    pub(crate) fn node_normal_work(&self, fourth: &Fourth) -> [f64; 2] {
        let (n, size) = (fourth.n, fourth.n * fourth.n);
        let made = if fourth.held() {
            self.quartic_work(fourth, None, false)
        } else {
            fourth.sum_work(0, None) + (fourth.orders.normalization.saturating_sub(1) * n * size) as f64
        };
        let first = if fourth.held() { fourth.first_decay_work() } else { 0.0 };
        [made, if fourth.bends() { (2 * n * size + 3 * size) as f64 } else { 0.0 } + first]
    }
    /// The counted complex multiply-adds of the held part's force form at a node (`Middle::held_force`) with the forms'
    /// slopes along `directions` directions: ρ's products with E₀ and the first order's forms per reference class, the
    /// static Hermitian terms' and partners' values with their slopes on the density, the crossed forms', the near part of
    /// 𝒯 with its gradients and the loss's contraction.
    pub(crate) fn held_force_work(&self, fourth: &Fourth, directions: usize) -> f64 {
        let (n, size) = (fourth.n, fourth.n * fourth.n);
        let summed: usize = self.form.iter().filter(|term| self.tones[term.tone] == 0.0).map(|term| term.len * (1 + if term.near { directions } else { 0 })).sum();
        let classes = fourth.count.max(1);
        ((2 + 2 * classes) * n * size + summed + fourth.entries.len() * (1 + directions) + fourth.groups.len() * (13 + 3 * directions) + 3 * (1 + directions) * size) as f64 + fourth.sum_work(1, Some(directions))
    }
    /// The counted complex multiply-adds of a point's near terms from a table on `axes` axes (tabulate.rs
    /// `Model::near_point`), the coordinates' apart: the reals it reads -- the generator's values to order 0, the form's
    /// and the responding forms to first order.
    pub(crate) fn near_read(&self, axes: usize) -> f64 {
        super::tabulate::read_work(axes, 0, self.generator_start.len()) + super::tabulate::read_work(axes, 1, self.form_start.len()) + super::tabulate::read_work(axes, 1, 2 * self.near.responding().len())
    }
    /// The complex multiply-adds of a point's values from its exact near forms with their slopes along `directions`
    /// directions (`Site::formed`, `values`).
    pub(crate) fn formed_work(&self, directions: usize) -> f64 {
        self.values + directions as f64 * self.sloped
    }
    /// The counted complex multiply-adds of the normalized generator's forms read from a table on `axes` axes
    /// (`Model::normal_read`): the crossed forms and Z₀, to first order where their `slopes` are read.
    pub(crate) fn normal_read_work(&self, axes: usize, slopes: bool) -> f64 {
        super::tabulate::read_work(axes, usize::from(slopes), 2 * self.near.normalizing().len())
    }
    /// The counted complex multiply-adds of a middle at a point, its near terms from a table on `axes` axes whose
    /// coordinates' values there count `place` (`Model::middle`): their interpolation -- the normalized generator's forms
    /// among them (`Model::normal_read`), its part the nodes' (`Plan::costs`); the values are the table's.
    pub(crate) fn middle_work(&self, axes: usize, place: f64) -> f64 {
        let normal = if self.fourth.is_some() { self.normal_read_work(axes, true) } else { 0.0 };
        if self.near.len() == 0 { 0.0 } else { place + self.near_read(axes) + normal }
    }
    /// The counted complex multiply-adds of the normalized generator's matrices (`Model::normal_at`) beside their forms'
    /// reads: a point's on `directions` directions (Some) with their gradients, or a node's values alone (None): the
    /// static Hermitian terms (with their slopes and gradients) summed by the components where the held part reads them,
    /// the crossed forms by their blocks' components, the near part of 𝒯 and its metric and loss, the matrices to the lab
    /// axes, and `normal`'s products of n × n matrices -- each with its three gradients' two at a point: the series' order
    /// 1 per reference class, S⁻¹ by Horner's rule (J − 1: its first step is none); and the landing (S, J − 1, and
    /// (1 + Y_L)⁻¹, J, and their product).
    pub(crate) fn quartic_work(&self, fourth: &Fourth, directions: Option<usize>, landing: bool) -> f64 {
        let (n, size, o) = (fourth.n, fourth.n * fourth.n, fourth.orders);
        let forms = usize::from(fourth.held());
        let (k, j) = (o.series, o.normalization);
        let statics = self.form.iter().filter(|term| self.tones[term.tone] == 0.0);
        let parts = if directions.is_some() { 4 } else { 1 };
        let (summed, entries, matrices, g) = match directions {
            Some(d) => (statics.map(|term| term.len * (4 + if term.near { d } else { 0 })).sum::<usize>(), fourth.entries.len() * (4 + d) + 7 * fourth.groups.len(), 12 * size * (1 + d), 7),
            None => (statics.map(|term| term.len).sum::<usize>(), fourth.entries.len() + fourth.groups.len(), 2 * size, 1),
        };
        let first = if fourth.held() || landing { usize::from(k >= 1) * fourth.count.max(1) } else { 0 };
        let products = g * (first + j.saturating_sub(1)) + if landing { 2 * j } else { 0 };
        let sums = match directions {
            Some(d) => fourth.sum_work(0, Some(d)) + fourth.sum_work(1, Some(d)),
            None => fourth.sum_work(0, None) + fourth.sum_work(1, None),
        };
        (forms * summed + entries + matrices + products * n * size) as f64 + sums + (2 * parts * size) as f64
    }
    /// The counted complex multiply-adds of the normalized generator's matrices made on a site of its own placed on a table of
    /// `axes` axes, whose coordinates there count `place` (`Model::quartic_made`, not laid: a jump's): the point's near
    /// terms read (`Plan::near_read`), and the matrices with their gradients and the landing that follows there
    /// (`quartic_work`).
    pub(crate) fn quartic_made_work(&self, fourth: &Fourth, axes: usize, place: f64) -> f64 {
        place + self.near_read(axes) + self.normal_read_work(axes, true) + self.quartic_work(fourth, Some(axes), true)
    }
    /// The counted complex multiply-adds of the plan's field components from their beams to an order (photon.rs `sited`),
    /// plain and Doppler-weighted.
    pub(crate) fn fields_work(&self, order: usize) -> f64 {
        self.scalars.iter().map(|s| (2 * jet::parts(order) * s.beams.len()) as f64).sum()
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
            let z = l.factor * if l.conjugate { form(l.operand).conj() } else { form(l.operand) };
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

    /// The excited amplitudes ε_ρ = 𝒯_ρ ψ̂ of every replica (photon.rs, # A state's photons at a point) into `out`, laid
    /// [part][amplitude row], term by term of 𝒯 as each is made -- never 𝒯 itself (module documentation, # The excited
    /// amplitude): e_k T_k − e^w_k T2_k of each lit carrier's response entry (an explicit state's static couplings' T
    /// alone) times ψ̂ at its place, each bare drive's e_k R, each explicit state's −1; and at the series' first order
    /// (`first`: E = E₀ + H¹ of the point, parts × n², the thread's matrices, `Model::quartic`) 𝒯⁽¹⁾ψ̂ =
    /// Σ_a Σ_t M⁽ᵃ⁾[:, t] (E P_a ψ̂)_t, M⁽ᵃ⁾ = Σ_k e_k R₀(z_a)R₀(z_t)U_k of the response entries at their first-order places
    /// (`Plan::first_place`; the partner T2 where a is t's own class), of the plain components alone. `turned`: the
    /// responding forms with their lab gradients (`Site::turn`: entry i's T_k, `responses.len()` on its T2_k);
    /// `plain`, `doppler`: the field components' jets; `bent`: ψ̂ = S⁻¹ψ and ∂_a S⁻¹ψ along the lab axes -- all to
    /// `parts` (1: the values; 4: with the lab gradients, by the product rule). `scratch` holds E P_a ψ̂. Returns the
    /// complex multiply-adds (`amplitudes_work`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn amplitudes(&self, turned: &[[C64; 4]], plain: &dyn Fn(usize) -> [C64; 4], doppler: &dyn Fn(usize) -> [C64; 4], first: Option<&[C64]>, bent: &[[C64; STATES]; 4], parts: usize, scratch: &mut Vec<C64>, out: &mut Vec<C64>) -> f64 {
        let (n, count, len) = (self.states, self.near.responses.len(), self.amplitude_len);
        out.clear();
        out.resize(parts * len, C64::ZERO);
        // The product rule of two jets to `parts`.
        let times = |a: &[C64; 4], b: &[C64; 4]| -> [C64; 4] { std::array::from_fn(|p| if p == 0 { a[0] * b[0] } else if p < parts { a[p] * b[0] + a[0] * b[p] } else { C64::ZERO }) };
        let psi = |y: usize| -> [C64; 4] { std::array::from_fn(|p| if p < parts { bent[p][y] } else { C64::ZERO }) };
        let add = |out: &mut [C64], row: usize, z: [C64; 4]| {
            for (p, z) in z.iter().enumerate().take(parts) {
                out[p * len + row] += *z;
            }
        };
        for r in &self.replicas {
            for &(i, s, j, y) in &r.near {
                let mut w = times(&s.map_or(UNIT, plain), &turned[i]);
                if let Some(s) = s {
                    let d = times(&doppler(s), &turned[count + i]);
                    w.iter_mut().zip(d).for_each(|(w, d)| *w -= d);
                }
                add(out, r.at + j, times(&w, &psi(y)));
            }
            for &(s, j, y, v) in &r.bare {
                add(out, r.at + j, times(&plain(s).map(|c| c * v), &psi(y)));
            }
            for &(j, y) in &r.explicit {
                add(out, r.at + j, psi(y).map(|z| -z));
            }
        }
        if let Some(e) = first {
            let (classes, size, responding) = (self.reference_classes.max(1), n * n, self.near.responding().start);
            let class = |s: usize| if self.reference_classes == 0 { 0 } else { self.references[s] };
            // φ_a = E P_a ψ̂ with its gradient, [a][part][t].
            scratch.clear();
            scratch.resize(classes * parts * n, C64::ZERO);
            for t in 0..n {
                for s in 0..n {
                    let a = class(s);
                    for p in 0..parts {
                        let mut z = e[p * size + t * n + s] * bent[0][s];
                        if p > 0 {
                            z += e[t * n + s] * bent[p][s];
                        }
                        scratch[(a * parts + p) * n + t] += z;
                    }
                }
            }
            for r in self.replicas.iter() {
                for &(i, s, j, t) in &r.near {
                    let c = s.map_or(UNIT, plain);
                    for a in 0..classes {
                        let w = times(&c, &turned[self.first_place[i * classes + a] - responding]);
                        let phi: [C64; 4] = std::array::from_fn(|p| if p < parts { scratch[(a * parts + p) * n + t] } else { C64::ZERO });
                        add(out, r.at + j, times(&w, &phi));
                    }
                }
            }
        }
        self.amplitudes_work(parts, first.is_some())
    }
    /// The counted complex multiply-adds of `amplitudes` to `parts` (with the series' first order where `first`): per
    /// near term its product with its component and with ψ̂ -- 2 parts − 1 each by the product rule --, a carrier's
    /// Doppler partner's product again; per bare term its component's product (parts) and ψ̂'s; at the first order
    /// E P_a ψ̂ (n² of them by the product rule) and per near term and class its two products.
    pub(crate) fn amplitudes_work(&self, parts: usize, first: bool) -> f64 {
        let each = (2 * parts - 1) as f64;
        let near: f64 = self.replicas.iter().flat_map(|r| &r.near).map(|x| if x.1.is_some() { 3.0 } else { 2.0 }).sum();
        let bare: usize = self.replicas.iter().map(|r| r.bare.len()).sum();
        let mut work = near * each + bare as f64 * (parts as f64 + each);
        if first {
            let terms: usize = self.replicas.iter().map(|r| r.near.len()).sum();
            work += (self.states * self.states) as f64 * each + (terms * self.reference_classes.max(1)) as f64 * 2.0 * each;
        }
        work
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
/// The counted complex multiply-adds of a responding form's read at a point (`Site::response`) to `parts`: its value, and
/// to first order its slope along each of `directions` directions onto the three lab axes, a real times a complex each
/// (half a complex multiply-add).
pub(crate) fn read_work(parts: usize, directions: usize) -> f64 {
    if parts > 1 { 1.5 * directions as f64 } else { 0.0 }
}

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
    /// (tabulate.rs `Model::placed`: every read of the table at the point takes it).
    pub(crate) coordinates: Vec<f64>,
    pub(crate) place: Option<(crate::table::Place, f64)>,
    /// The responding forms at the point (`NearLayout::responding`: the responses, their partners, the mixed responses),
    /// and their slopes along `directions` (`response_slopes[f * len + i]`), where read to first order (`Model::responded`).
    pub(crate) responses: Vec<C64>,
    pub(crate) response_slopes: Vec<C64>,
    /// The responding forms with their gradients along the lab axes, of their slopes along `directions` (`Site::turn`):
    /// what 𝒯 at the point is made of (`Model::excited_at`).
    pub(crate) turned: Vec<[C64; 4]>,
    /// What a state's photons read (photon.rs `Model::photons_of`): the plan's field components; the excited amplitudes
    /// of a state (`Plan::amplitudes`, of the responding forms turned, `turned`) and a class's sums in the making
    /// (`class`, laid for the lanes).
    pub(crate) fields: Fields,
    pub(crate) member: Vec<C64>,
    pub(crate) class: Vec<f64>,
    /// What a jump's draw reads (photon.rs `Model::emission_of`): the classes its state emits, their places among the
    /// plan's classes and their rates.
    pub(crate) emitting: Vec<usize>,
    pub(crate) rates: Vec<f64>,
    /// What the readout reads (readout.rs `Model::readout_of`): the carriers' field components (eliminate.rs
    /// `Model::responses_each`), and its channels' responses, one after another, each row-major on the excited states and
    /// the cluster's ground places.
    pub(crate) weights: Vec<(C64, C64)>,
    pub(crate) channels: Vec<C64>,
    /// Per channel and ground place its excited amplitudes, and per class of co-resonant places, at its first place,
    /// their sum (readout.rs, # The readout).
    pub(crate) excited_rows: Vec<C64>,
    /// The normalized generator's forms at the point (module documentation, # The normalized generator): the crossed
    /// forms and Z₀, real and imaginary parts, with their slopes along `directions` (`normal_slopes[(f count + e) 2]`), read
    /// once a point (`Model::normal_read`).
    pub(crate) normal: Vec<f64>,
    pub(crate) normal_slopes: Vec<f64>,
    /// Its nodes' scratch, one for all its paths: a node's call fills what it reads.
    scratch: RefCell<Scratch>,
}

impl Site {
    /// The plan's beams' amplitudes with their gradients, from every beam's (`beams`).
    fn select(&mut self, plan: &Plan) {
        // A node's fields are of the beams' amplitudes: none kept from another point's.
        for node in &mut self.scratch.get_mut().nodes {
            node.key = None;
        }
        self.amplitudes.clear();
        self.amplitudes.extend(plan.beams.iter().map(|&l| &self.beams[l]).map(|a| [a.value, a.gradient[0], a.gradient[1], a.gradient[2]]));
    }
    /// The near terms' values from the near forms at the point (the Hermitian ones to first order): a Hermitian value its
    /// constant and its near form with its slopes; and the responding forms with their slopes. Without near forms (None):
    /// the constants alone.
    fn formed(&mut self, plan: &Plan, near: Option<&NearValues>) {
        self.generator.clear();
        self.generator.extend_from_slice(&plan.generator_start);
        self.forms.clear();
        self.forms.extend_from_slice(&plan.form_start);
        self.directions.clear();
        self.slopes.clear();
        self.responses.clear();
        self.response_slopes.clear();
        let Some(near) = near else { return };
        self.directions.extend(near.slopes());
        let width = self.forms.len();
        self.slopes.resize(self.directions.len() * width, 0.0);
        plan.lay(|at| near.data[at], Some(&mut self.generator), &mut self.forms);
        for (f, slopes) in self.slopes.chunks_mut(width.max(1)).enumerate() {
            plan.lay(|at| near.data[(1 + f) * near.len + at], None, slopes);
        }
        self.responded(plan, near);
    }
    /// The responding forms from the near forms at the point (`Site::responses`), with their slopes where they have any.
    fn responded(&mut self, plan: &Plan, near: &NearValues) {
        let range = plan.near.responding();
        self.responses.clear();
        self.responses.extend_from_slice(&near.data[range.clone()]);
        self.response_slopes.clear();
        for f in 0..self.directions.len() {
            self.response_slopes.extend_from_slice(&near.data[(1 + f) * near.len + range.start..(1 + f) * near.len + range.end]);
        }
    }
    /// The responding forms (`Site::responses`) with their gradients along the lab axes, of their slopes along the site's
    /// directions (none where they have none: their values alone), into `turned`, one pass: a form's read (`read_work`).
    fn turn(&mut self) {
        let (len, directions) = (self.responses.len(), self.directions.len());
        let sloped = len > 0 && self.response_slopes.len() >= len * directions;
        self.turned.clear();
        self.turned.extend(self.responses.iter().enumerate().map(|(i, &value)| {
            let mut out = [value, C64::ZERO, C64::ZERO, C64::ZERO];
            if sloped {
                for (f, d) in self.directions.iter().enumerate() {
                    let slope = self.response_slopes[f * len + i];
                    for a in 0..3 {
                        out[1 + a] += slope * d[a];
                    }
                }
            }
            out
        }));
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
    /// A cluster's plan under a light set, made once: the plans' lock is held to find its place or to give it one, and
    /// the plan is made in that place alone -- whoever asks for the same plan meanwhile waits for it, and two plans
    /// asked side by side are made side by side.
    pub(crate) fn plan(&self, cluster: usize, lights: &Lights) -> Arc<Plan> {
        self.plan_at(cluster, lights, self.orders_of(cluster))
    }

    /// `plan` at the evaluation's orders `orders`: the run's are the cluster's own (eliminate.rs `Model::orders_of`, `plan`), the readings' the next
    /// ones too (elim.rs `Model::readings`: the one evaluation, `Model::evaluations`).
    pub(crate) fn plan_at(&self, cluster: usize, lights: &Lights, orders: Orders) -> Arc<Plan> {
        let place = {
            let mut plans = self.plans.lock().expect("the plans' lock");
            match plans.iter().find(|of| of.0 == cluster && of.1 == *lights && of.2 == orders) {
                Some(found) => found.3.clone(),
                None => {
                    plans.push((cluster, lights.clone(), orders, Arc::default()));
                    plans.last().expect("the place just given").3.clone()
                }
            }
        };
        place.get_or_init(|| Arc::new(Plan::new(self, cluster, lights, orders))).clone()
    }

    /// The excited states the response of a carrier's basis coupling k from ground state b can reach on a cluster: those in a
    /// block (`Model::mixing`) with a state its drives folded at the atom couple b to, less the cluster's explicit states
    /// (P at every frequency, ELIMINATION v12: no response holds them).
    pub(crate) fn reached(&self, cluster: usize, carrier: usize, k: usize, b: usize) -> [bool; EXCITED] {
        let u = &self.carriers[carrier].basis[k];
        self.blocks_of(cluster, std::array::from_fn(|e| u[(GROUND + e, b)] != C64::ZERO))
    }

    /// The explicit states' static couplings to Q on a cluster under a light set (eliminate.rs `Responses::held`): of its
    /// explicit place x the excited states (less GROUND) that K[Q, x] can couple it to -- the base's, the decay's and each
    /// far beam pair's form the lights hold (eliminate.rs `Model::level_pairs`) -- and those its response R₀(x)K[Q, x]
    /// reaches (`blocks_of`).
    pub(crate) fn held_support(&self, cluster: usize, x: usize, lights: &Lights) -> ([bool; EXCITED], [bool; EXCITED]) {
        let c = &self.clusters[cluster];
        let s = c.states[x];
        let pairs = self.level_pairs(lights);
        let explicit = |e: usize| c.states[c.ground..].contains(&(GROUND + e));
        let coupled: [bool; EXCITED] = std::array::from_fn(|e| {
            !explicit(e)
                && (self.base[(GROUND + e, s)] != C64::ZERO
                    || self.decay[(e, s - GROUND)] != C64::ZERO
                    || pairs.iter().enumerate().any(|(w, of)| of.iter().enumerate().any(|(p, &held)| held && self.tones[w].pairs[p][(GROUND + e, s)] != C64::ZERO)))
        });
        (coupled, self.blocks_of(cluster, coupled))
    }

    /// The excited states in a block (`Model::mixing`) with one of `from`, less the cluster's explicit states: what a
    /// response of couplings into `from` can reach -- the factorization of K − z on Q and its solves leave every other
    /// entry exactly zero.
    fn blocks_of(&self, cluster: usize, from: [bool; EXCITED]) -> [bool; EXCITED] {
        let (c, blocks) = (&self.clusters[cluster], &self.mixing[cluster]);
        let mut out = [false; EXCITED];
        for f in (0..EXCITED).filter(|&f| from[f]) {
            for e in 0..EXCITED {
                out[e] |= blocks[e] == blocks[f];
            }
        }
        for &x in &c.states[c.ground..] {
            out[x - GROUND] = false;
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
    /// sums and its partner's, each crossed form's, each shift's two per ground state). The forms of the ground places
    /// `places` takes -- those the responses were made of (eliminate.rs `Model::eliminate`): a form that reads another
    /// place's response is left as `out` holds it.
    pub(crate) fn forms(&self, plan: &Plan, responses: &Responses, energies: &[Vec<f64>], out: &mut [C64], places: &dyn Fn(usize) -> bool) -> f64 {
        let cluster = &self.clusters[plan.cluster];
        let states = &cluster.states;
        let (near, len) = (&plan.near, plan.near.len());
        // The pairs and the crossed forms made, a part's.
        let pairs = near.pairs.iter().filter(|pair| places(pair[4]) && places(pair[5])).count();
        let crossed = near.crossed.iter().filter(|&&[i, _]| places(near.pairs[i][4]) && places(near.pairs[i][5])).count();
        // A pair side's response slot -- a frequency's carrier (one that is none has no response at the atom), the explicit
        // states' after the carriers for the frequency `tones.len()` (`Builder::touches`) -- and its coupling's part p
        // (none where it is zero): the explicit states' at the point (`Responses::held`, the far light's jet), beside them a
        // carrier's drives folded at the atom, else its frequency's every folded coupling, constants.
        let held = self.tones.len();
        let slot = |w: usize| if w == held { Some(self.carriers.len()) } else { self.carrier_of(w) };
        let coupling = |w: usize, k: usize, other: usize, p: usize| -> Option<&Mat> {
            if w == held {
                responses.held.get(p)
            } else if p > 0 {
                None
            } else if other == held {
                Some(&self.carriers[self.carrier_of(w).expect("a carrier beside the explicit states' couplings")].basis[k])
            } else {
                Some(&self.tones[w].coupled[0][k])
            }
        };
        // The product rule's pairs of parts (coupling, response) of a part.
        let rule = |part: usize| -> Vec<(usize, usize)> {
            match part {
                0 => vec![(0, 0)],
                p if p < 4 => vec![(p, 0), (0, p)],
                p => {
                    let (a, b) = jet::PAIRS[p - 4];
                    vec![(p, 0), (1 + a, 1 + b), (1 + b, 1 + a), (0, p)]
                }
            }
        };
        // −½ c† r of a coupling's part and a response's, summed over a part's product rule.
        let contract = |part: usize, w: usize, k: usize, other: usize, x: usize, response: Option<&[Mat]>, y: usize| -> C64 {
            rule(part).into_iter().filter_map(|(a, b)| Some(half(coupling(w, k, other, a)?, states[x], response?.get(b)?, y))).sum()
        };
        for part in 0..energies.len() {
            let out = &mut out[part * len..(part + 1) * len];
            // The fold's one contraction (eliminate.rs `shift`) of the legs -- each its coupling and its response at the
            // point --; a Doppler partner is the absorbed half with −T2 for T.
            for (i, &[wl, kl, wr, kr, x, y]) in near.pairs.iter().enumerate().filter(|(_, pair)| places(pair[4]) && places(pair[5])) {
                let first = |w: usize, k: usize| slot(w).map(|c| responses.first[c][k].as_slice());
                let second = |w: usize, k: usize| slot(w).map(|c| responses.second[c][k].as_slice());
                out[near.at(Operand::Pair(i))] = contract(part, wl, kl, wr, x, first(wr, kr), y) + contract(part, wr, kr, wl, y, first(wl, kl), x).conj();
                out[near.at(Operand::Doppler(i))] = -contract(part, wl, kl, wr, x, second(wr, kr), y);
            }
            // The crossed forms (`NearLayout::crossed`): U^f_k'† R₀(z_a)R₀(z_y) U_k on x ← y -- at y's own class R₀(z_y)²U_k.
            for (c, &[i, a]) in near.crossed.iter().enumerate().filter(|(_, &[i, _])| places(near.pairs[i][4]) && places(near.pairs[i][5])) {
                let [wl, kl, wr, kr, x, y] = near.pairs[i];
                let Some(cr) = slot(wr) else { continue };
                let own = plan.references.is_empty() || plan.references[y] == a;
                let mixed = if own { &responses.second[cr][kr] } else { &responses.mixed[cr][kr][a] };
                out[near.at(Operand::Crossed(c))] = contract(part, wl, kl, wr, x, Some(mixed), y) * -2.0;
            }
            let responding = near.responding().start;
            let count = near.responses.len();
            for (i, &[c, k, e, y]) in near.responses.iter().enumerate().filter(|(_, response)| places(response[3])) {
                out[responding + i] = responses.first[c][k][part][(e, y)];
                out[responding + count + i] = responses.second[c][k][part][(e, y)];
            }
            for (m, &[i, a]) in near.mixed.iter().enumerate() {
                let [c, k, e, y] = near.responses[i];
                if places(y) {
                    out[responding + 2 * count + m] = responses.mixed[c][k][a][part][(e, y)];
                }
            }
            for i in 0..near.reference {
                out[near.at(Operand::Reference(i))] = C64::real(responses.reference[part][i]);
            }
        }
        // The decay forms (`DecayForm`): v_l† K v_r of the two legs' vectors at their places, each part's by the product
        // rule of the jets (a constant leg's derivatives none); K v_r made once a right vector and kernel.
        let parts = energies.len();
        let mut decayed = 0;
        let vector = |leg: (DecayLeg, usize), place: usize, part: usize| -> Option<[C64; EXCITED]> {
            match leg {
                (DecayLeg::Constant(j), 0) => Some(if part == 0 { plan.legs[j] } else { [C64::ZERO; EXCITED] }),
                (DecayLeg::Constant(_), _) => None,
                (DecayLeg::Near { carrier, k }, order) => {
                    let m = match order {
                        0 => &responses.first[carrier][k][part],
                        1 => &responses.second[carrier][k][part],
                        o => &responses.mixed.get(carrier)?.get(k)?.get(o - 2)?[part],
                    };
                    Some(std::array::from_fn(|e| m[(e, place)]))
                }
            }
        };
        let mut kept: HashMap<((DecayLeg, usize), usize, usize, usize), [C64; EXCITED]> = HashMap::new();
        for (i, d) in near.decays.iter().enumerate().filter(|(_, d)| places(d.x) && places(d.y)) {
            let kernel = &plan.kernels[d.kernel];
            for part in 0..parts {
                // The product rule's pairs of parts: (left, right).
                let terms: Vec<(usize, usize)> = match part {
                    0 => vec![(0, 0)],
                    p if p < 4 => vec![(p, 0), (0, p)],
                    p => {
                        let (a, b) = jet::PAIRS[p - 4];
                        vec![(p, 0), (1 + a, 1 + b), (1 + b, 1 + a), (0, p)]
                    }
                };
                let mut z = C64::ZERO;
                for (pl, pr) in terms {
                    let (Some(a), Some(b)) = (vector(d.left, d.x, pl), vector(d.right, d.y, pr)) else { continue };
                    let kb = kept.entry((d.right, d.y, d.kernel, pr)).or_insert_with(|| {
                        let mut out = [C64::ZERO; EXCITED];
                        for &(e, f, v) in kernel {
                            out[e] += v * b[f];
                        }
                        decayed += kernel.len();
                        out
                    });
                    z += a.iter().zip(kb.iter()).map(|(x, y)| x.conj() * *y).sum::<C64>();
                    decayed += EXCITED;
                }
                out[part * len + near.at(Operand::Decay(i))] = z;
            }
        }
        (energies.len() * (3 * pairs + crossed) * EXCITED + decayed) as f64
    }

    /// A plan's near forms at a point from the exact elimination there, jets to `order` (eliminate.rs).
    pub(crate) fn near_exact(&self, plan: &Plan, amplitudes: &[Amplitude], order: usize) -> NearValues {
        self.near_exact_on(plan, amplitudes, order, &|_| true)
    }

    /// `near_exact` of the cluster's ground places `places` takes: the responses from those places alone and the forms
    /// that read no other's (zero elsewhere) -- what a reader of a few places' forms asks for (the potential: the
    /// populated states' own shifts). A form made is the same number whichever places are taken with it.
    pub(crate) fn near_exact_on(&self, plan: &Plan, amplitudes: &[Amplitude], order: usize, places: &dyn Fn(usize) -> bool) -> NearValues {
        let len = plan.near.len();
        let mut data = vec![C64::ZERO; jet::parts(order) * len];
        if len == 0 {
            return NearValues { len, order, data, directions: None, work: 0.0 };
        }
        let h = self.far_hamiltonian(amplitudes, &plan.lights, order);
        let responses = self.eliminate(plan.cluster, &h, &plan.lights, order, plan.mixes());
        let energies: Vec<Vec<f64>> = h.iter().map(|m| (0..STATES).map(|s| m[(s, s)].re).collect()).collect();
        let work = self.forms(plan, &responses, &energies, &mut data, places);
        NearValues { len, order, data, directions: None, work: responses.work + self.far_work(&plan.lights, order) + work }
    }

    /// The normalized generator's matrices of a plan's cluster at a point (`Quartic`) into the thread's, and the counted work
    /// where they were made for it: a thread keeps its last point's -- of the same plan, the same table's grid and the
    /// same position `at` they are the same numbers, so a step's middle and its scattering make them once (None: made,
    /// and kept for no other). `point` holds the point's beams and its responding forms (`Model::responded`); `laid`: and
    /// its forms with their slopes (a middle's: `Site::formed`, tabulate.rs `Model::near_point`), which are read here and
    /// else laid here.
    pub(crate) fn quartic(&self, plan: &Plan, fourth: &Fourth, table: Option<&NearTable>, at: Option<[f64; 3]>, point: &mut Site, laid: bool, landing: bool) -> f64 {
        QUARTIC.with(|kept| {
            let mut q = kept.borrow_mut();
            let key = at.map(|r| quartic_key(plan, table, r));
            if key.is_some() && q.key == key && (q.landing || !landing) {
                // Made for the step's middle node (`Model::node_normal`): counted here, once.
                return if std::mem::replace(&mut q.counted, true) { 0.0 } else { q.work };
            }
            q.landing = landing;
            self.quartic_made(plan, fourth, table, None, point, laid, &mut q);
            (q.key, q.counted) = (key, true);
            q.work
        })
    }

    /// The thread's matrices (`quartic`) of a middle's point at `r` -- its site laid, its forms read (`Model::middle`) --,
    /// made unless they are its already; their work left for the step's scattering to count.
    fn quartic_kept(&self, plan: &Plan, fourth: &Fourth, table: Option<&NearTable>, r: [f64; 3], point: &Site, q: &mut Quartic) {
        let key = Some(quartic_key(plan, table, r));
        if q.key == key {
            return;
        }
        q.landing = false;
        q.work = self.quartic_matrices(plan, fourth, point, q);
        (q.key, q.counted) = (key, false);
    }

    /// The thread's fourth-order matrices (`quartic`), read.
    pub(crate) fn quartic_read<R>(read: impl FnOnce(&Quartic) -> R) -> R {
        QUARTIC.with(|kept| read(&kept.borrow()))
    }

    /// The thread's matrices forgotten: the next site makes its own (a count of what a site makes, model.rs `Run::with`).
    pub(crate) fn forget_quartic() {
        QUARTIC.with(|kept| kept.borrow_mut().key = None);
    }

    /// `quartic`'s matrices (module documentation, # The normalized generator): on a site of its own the point's forms
    /// laid and its normalization forms read (`normal_read`) -- a middle's site (`laid`) holds both --, then the matrices
    /// of the point's own field components (`normal_at`).
    fn quartic_made(&self, plan: &Plan, fourth: &Fourth, table: Option<&NearTable>, given: Option<&NearValues>, point: &mut Site, laid: bool, q: &mut Quartic) {
        q.work = 0.0;
        if !laid {
            // The exact elimination's forms where no table holds them and none are given.
            let made = (table.is_none() && given.is_none()).then(|| self.near_exact(plan, &point.beams, 1));
            let exact = given.or(made.as_ref());
            point.select(plan);
            q.work += match (table, exact) {
                (Some(table), _) => self.near_point(table, point),
                (None, Some(near)) => {
                    point.formed(plan, Some(near));
                    near.work + plan.formed_work(point.directions.len())
                }
                (None, None) => unreachable!("the exact elimination's forms are made where no table is"),
            };
            q.work += self.normal_read(plan, table, exact, point, true);
        }
        q.work += self.quartic_matrices(plan, fourth, point, q);
    }

    /// The matrices of a site whose forms are laid and read (`normal_at` at the point's own field components, with
    /// their gradients and, where asked, the landing map), and their counted work.
    fn quartic_matrices(&self, plan: &Plan, fourth: &Fourth, point: &Site, q: &mut Quartic) -> f64 {
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
        self.normal_at(plan, fourth, point, &scalars[..plan.scalars.len()], true, true, q)
    }

    /// The normalized generator's forms at a site's point into the site (`Site::normal`; module documentation, # The
    /// normalized generator): the crossed forms and Z₀ with their slopes along the site's directions -- the table's
    /// interpolant, or the exact elimination's `exact`. Read once a point: a middle's nodes take them (`Model::node_normal`).
    /// Their counted work: the table's read of them (`Plan::normal_read_work`); the exact elimination's are copied, its
    /// own count theirs. Without `slopes` their values alone.
    pub(crate) fn normal_read(&self, plan: &Plan, table: Option<&NearTable>, exact: Option<&NearValues>, point: &mut Site, slopes: bool) -> f64 {
        let range = plan.near.normalizing();
        let (first, count) = (range.start, range.len());
        let dn = if slopes { point.directions.len() } else { 0 };
        let Site { normal, normal_slopes, place, .. } = point;
        sized(normal, 2 * count);
        sized(normal_slopes, dn * 2 * count);
        match (table, exact) {
            (Some(table), _) => {
                table.table.read(&place.expect("a site placed on its table").0, 2 * first, None, normal, slopes.then_some(&mut normal_slopes[..]));
                plan.normal_read_work(table.axes(), slopes)
            }
            (None, Some(near)) => {
                for e in 0..count {
                    (normal[2 * e], normal[2 * e + 1]) = (near.data[first + e].re, near.data[first + e].im);
                    for f in 0..dn {
                        let z = near.data[(1 + f) * near.len + first + e];
                        (normal_slopes[(f * count + e) * 2], normal_slopes[(f * count + e) * 2 + 1]) = (z.re, z.im);
                    }
                }
                0.0
            }
            (None, None) => unreachable!("the exact elimination's forms are made where no table is"),
        }
    }

    /// The normalized generator's matrices (`Quartic::data`; module documentation, # The normalized generator) of a site's
    /// point at the plan's field components `scalars` (each its value and its gradient along the lab axes): the point's
    /// own, or a node's along a step's path through it -- its components the point's beams turned by their plane waves
    /// (`Middle::along`), its forms and responses the point's. The static Hermitian generator H_h and the first order's
    /// forms are the point's laid forms summed with the components by the nodes' kernel (module documentation, # Dense
    /// terms) -- of the forms' terms, which hold one of a term and its adjoint (`Term::weight`), every static one's and,
    /// where the other is left out, its adjoint; the Doppler partners' (2D = Y⁽ᵃ⁾ at y's own class a) where the absorbed
    /// component takes the shifts, and the adjoints of those where the emitted one does; the crossed forms (Y⁽ᵃ⁾ at the
    /// other classes) --, each with its slopes along the point's directions and the components' gradients. X and L are the
    /// near part of 𝒯's metric and loss (`Plan::excite`, `Plan::metric`), Z₀ the reference form (`Site::normal`). From them
    /// `normal`: at a point (`at_point`) with the landing map where `Quartic::landing` asks; without `gradients` the values
    /// alone. Returns the counted work.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn normal_at(&self, plan: &Plan, fourth: &Fourth, point: &Site, scalars: &[[C64; 4]], gradients: bool, at_point: bool, q: &mut Quartic) -> f64 {
        let n = fourth.n;
        let (size, padded) = (n * n, plan.padded);
        let Quartic { data, sums, laid: lanes, landing, arena, scratch, .. } = q;
        let landing = at_point && *landing;
        let dn = if gradients { point.directions.len() } else { 0 };
        let (parts, made) = (4 + dn, if gradients { 4 } else { 1 });
        let classes = fourth.count.max(1);
        // H_h and the first order's forms enter the held part and E₀ (H_h), the held part and the landing (the forms).
        let forms_needed = fourth.held();
        let first_needed = fourth.held() || landing;
        // Four sums, each its value, its slopes' along each direction and the components' gradients' on it, real parts
        // and imaginary parts on the cluster's places: the static Hermitian terms whose adjoints the forms hold not, and
        // hold; the partners of an absorbed component's shifts, and of an emitted one's.
        lanes.clear();
        lanes.resize(4 * parts * 2 * padded, 0.0);
        let width = point.forms.len();
        for term in plan.form.iter().filter(|term| plan.tones[term.tone] == 0.0 && if term.flag == PLAIN { forms_needed } else { first_needed }) {
            let block = &plan.blocks[term.block];
            let to = match term.flag {
                PLAIN => if term.weight == 2.0 { 1 } else { 0 },
                flag => {
                    let shifted = if flag == RIGHT { block.right } else { block.left };
                    if shifted.is_some_and(|s| plan.scalars[s].conjugate) { 3 } else { 2 }
                }
            };
            let (l, r) = (side(scalars, block.left), side(scalars, block.right));
            let c: [C64; 4] = std::array::from_fn(|part| if part == 0 { l[0] * r[0] } else if gradients { l[part] * r[0] + l[0] * r[part] } else { C64::ZERO });
            let span = term.first..term.first + term.len;
            let lane = |part: usize| (to * parts + part) * 2 * padded;
            let (real, imaginary) = plan.values(term, &point.forms);
            let mut add = |part: usize, c: C64, real: &[f64], imaginary: &[f64]| {
                let (re, im) = lanes[lane(part)..lane(part) + 2 * padded].split_at_mut(padded);
                axpy(c, real, imaginary, &mut re[span.clone()], &mut im[span.clone()]);
            };
            for (part, c) in [(0, c[0]), (1 + dn, c[1]), (2 + dn, c[2]), (3 + dn, c[3])].into_iter().take(made) {
                add(part, c, real, imaginary);
            }
            if term.near && gradients {
                for f in 0..dn {
                    let (real, imaginary) = point.slopes[f * width + term.at..f * width + term.at + 2 * term.len].split_at(term.len);
                    add(1 + f, c[0], real, imaginary);
                }
            }
        }
        // The crossed forms: Σ e_k'* e_k of their pair entries into each class's Y⁽ᵃ⁾, as the shifts' sums are parted; and Z₀.
        let count = plan.near.normalizing().len();
        let (values, slopes) = (&point.normal, &point.normal_slopes);
        let value = |e: usize| C64::new(values[2 * e], values[2 * e + 1]);
        let slope = |f: usize, e: usize| C64::new(slopes[(f * count + e) * 2], slopes[(f * count + e) * 2 + 1]);
        sums.clear();
        sums.resize(classes * parts * size, C64::ZERO);
        if first_needed {
            for (left, right, group) in fourth.groups() {
                let (l, r) = (side(scalars, left), side(scalars, right));
                let c: [C64; 4] = std::array::from_fn(|part| if part == 0 { l[0] * r[0] } else if gradients { l[part] * r[0] + l[0] * r[part] } else { C64::ZERO });
                for &(e, a, xy) in group {
                    let (v, to) = (value(e), a * parts * size + xy);
                    sums[to] += c[0] * v;
                    if gradients {
                        for f in 0..dn {
                            sums[to + (1 + f) * size] += c[0] * slope(f, e);
                        }
                        for k in 0..3 {
                            sums[to + (1 + dn + k) * size] += c[1 + k] * v;
                        }
                    }
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
        let reference = plan.near.crossed.len();
        let z0: Vec<[f64; 4]> = (0..plan.near.reference)
            .map(|i| {
                std::array::from_fn(|part| match part {
                    0 => value(reference + i).re,
                    _ if gradients => (0..dn).map(|f| slope(f, reference + i).re * directions[f][part - 1]).sum(),
                    _ => 0.0,
                })
            })
            .collect();
        // H_h, the forms Y⁽ᵃ⁾ class by class, X and L, four parts each, n × n row by row.
        sized(arena, (12 + 4 * classes) * size);
        let (hermitian, rest) = arena.split_at_mut(4 * size);
        let (metric, rest) = rest.split_at_mut(4 * size);
        let (loss, ys) = rest.split_at_mut(4 * size);
        let references = &fourth.classes;
        for part in 0..made {
            for xy in 0..size {
                let (at, yx) = (part * size + xy, (xy % n) * n + xy / n);
                hermitian[at] = if forms_needed { turned(&|p| lane(0, p, xy) + lane(1, p, xy) + lane(1, p, yx).conj(), part) } else { C64::ZERO };
                let partner = if first_needed { turned(&|p| lane(2, p, xy) + lane(3, p, yx).conj(), part) * 2.0 } else { C64::ZERO };
                for a in 0..classes {
                    let own = if references.is_empty() || references[xy % n] == a { partner } else { C64::ZERO };
                    ys[(a * 4 + part) * size + xy] = own + turned(&|p| sums[(a * parts + p) * size + xy], part);
                }
            }
        }
        // X and L of the near part of 𝒯 at the components (the decay sums, `Sum::Metric`, `Sum::Loss`).
        let mut lanes = vec![C64::ZERO; parts * size];
        for (k, out) in [(0, &mut *metric), (1, &mut *loss)] {
            lanes.fill(C64::ZERO);
            fourth.sum_into(k, scalars, None, None, &value, Some((&slope, dn)), gradients, &mut lanes);
            for part in 0..made {
                for xy in 0..size {
                    out[part * size + xy] = turned(&|p| lanes[p * size + xy], part);
                }
            }
        }
        sized(data, (FIRST + classes) * size);
        normal(hermitian, ys, references, classes, metric, loss, n, &z0, fourth.orders, gradients, landing, data, scratch);
        plan.quartic_work(fourth, gradients.then_some(dn), landing)
    }

    /// The normalized generator's part at a node of a step's path through a middle (`Node::normal`; module documentation,
    /// # The normalized generator), of the node's own field components and the middle's forms and responses: where the held
    /// part is nothing, S⁻¹ = Σ_{j≤J} b_j X^j of the node's X alone (the near part of 𝒯 at the node's components); else the
    /// node's matrices (`normal_at`, its values alone or with their gradients) -- S⁻¹, E₀, the held part with its gradient
    /// and the first order's forms (`node_of`). At the middle (`centre`: the middle's table and position) the node's
    /// components are the point's on every path through it: the part is the point's own, the step's scattering's matrices
    /// with their gradients (`Model::quartic`), made once a point. Made once a node.
    #[allow(clippy::too_many_arguments)]
    fn node_normal(&self, plan: &Plan, fourth: &Fourth, point: &Site, node: &mut Node, quartic: &mut Quartic, centre: Option<(Option<&NearTable>, [f64; 3])>, gradients: bool) {
        if node.normal_made && (node.normal_gradients || !gradients) {
            return;
        }
        if let Some((table, r)) = centre {
            QUARTIC.with(|kept| {
                let mut q = kept.borrow_mut();
                self.quartic_kept(plan, fourth, table, r, point, &mut q);
                node_of(fourth, node, &q);
            });
            (node.normal_made, node.normal_gradients) = (true, true);
            return;
        }
        if fourth.held() {
            quartic.landing = false;
            self.normal_at(plan, fourth, point, &node.plain, gradients, false, quartic);
            node_of(fourth, node, quartic);
        } else {
            self.bent_of(fourth, point, &node.plain, &mut quartic.sums, &mut node.normal);
        }
        (node.normal_made, node.normal_gradients) = (true, gradients);
    }

    /// S⁻¹ = Σ_{j≤J} b_j X^j (n × n) into `out` of the plan's plain field components `plain` (their values) and a site's
    /// normalizing forms (`Site::normal`): X the metric's decay sum there (`Sum::Metric`; module documentation, # The
    /// normalized generator). A node's where the held part is nothing (`Model::node_normal`). `scratch` holds what it makes.
    fn bent_of(&self, fourth: &Fourth, point: &Site, plain: &[[C64; 4]], scratch: &mut Vec<C64>, out: &mut Vec<C64>) {
        let size = fourth.n * fourth.n;
        let value = |e: usize| C64::new(point.normal[2 * e], point.normal[2 * e + 1]);
        sized(scratch, 2 * size);
        let (x, t) = scratch.split_at_mut(size);
        x.fill(C64::ZERO);
        fourth.sum_into(0, plain, None, None, &value, None, false, x);
        sized(out, size);
        powers(-0.5, fourth.orders.normalization, x, t, out, fourth.n);
    }

    /// A site's responding forms (`Site::responses`) at its point to `order` (0, or 1 with their slopes along the site's
    /// directions): a table's interpolant where the site is placed on one (tabulate.rs `Model::placed`), else the exact
    /// elimination's (its directions the lab axes); and their counted work.
    pub(crate) fn responded(&self, plan: &Plan, table: Option<&NearTable>, order: usize, site: &mut Site) -> f64 {
        if plan.near.responding().is_empty() {
            site.responses.clear();
            site.response_slopes.clear();
            return 0.0;
        }
        match table {
            Some(table) => self.near_responses(table, order, site),
            None => {
                let near = self.near_exact(plan, &site.beams, order);
                site.directions.clear();
                site.directions.extend(near.slopes());
                site.responded(plan, &near);
                near.work
            }
        }
    }

    /// A site's responding forms turned to the lab axes (`Site::turn`), what a state's excited amplitudes are made of
    /// (`Plan::amplitudes`), once a point to `order` (0, or 1 with their gradients): their reads' counted work
    /// (`read_work`).
    pub(crate) fn turned(&self, plan: &Plan, site: &mut Site, order: usize) -> f64 {
        site.turn();
        plan.near.responding().len() as f64 * read_work(if order >= 1 { 4 } else { 1 }, site.directions.len())
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
    /// (tabulate.rs `Model::placed`). What is read of the model at the point is read on a site that holds it. Returns the
    /// table the point's forms are read from: the plan's where it holds the point, else none -- the exact elimination's
    /// there (tabulate.rs, # Axes).
    pub(crate) fn pointed<'t>(&self, plan: &Plan, table: Option<&'t NearTable>, r: [f64; 3], site: &mut Site) -> Option<&'t NearTable> {
        self.light.amplitudes_into(r, 1, |_| true, &mut site.beams);
        table.filter(|table| plan.near.len() > 0 && self.placed(table, site))
    }

    /// A cluster's generator at a position and time under a plan (`Middle`), the near forms from a table where one is given.
    pub(crate) fn middle<'a>(&'a self, plan: &Arc<Plan>, r: [f64; 3], t: f64, table: Option<&'a NearTable>) -> Middle<'a> {
        let mut point = Held::<Site>::take();
        let table = self.pointed(plan, table, r, &mut point);
        point.select(plan);
        let exact = (table.is_none() && plan.near.len() > 0).then(|| self.near_exact(plan, &point.beams, 1));
        let work = match (table, &exact) {
            _ if plan.near.len() == 0 => {
                point.formed(plan, None);
                0.0
            }
            (Some(table), _) => self.near_point(table, &mut point),
            (None, Some(near)) => {
                point.formed(plan, Some(near));
                near.work + plan.formed_work(point.directions.len())
            }
            (None, None) => unreachable!("the exact elimination's forms are made where no table is"),
        };
        // The normalized generator's forms, which the step's nodes take (`Model::node_normal`).
        point.normal.clear();
        let work = work + plan.fourth.as_ref().map_or(0.0, |_| self.normal_read(plan, table, exact.as_ref(), &mut point, true));
        Middle { model: self, plan: plan.clone(), r: Some(r), t, table, point: RefCell::new(point), work }
    }

    /// The same from every beam's amplitude and the near forms there (the Hermitian ones to first order; `Site::formed`),
    /// the normalized generator's part made of the same forms. Without near forms (None): the terms that hold none, and no
    /// normalized generator. Of forms to order 0 the generator is the same, bit for bit -- a form's value is the same number
    /// at every order --, and the force's forms are without their values' slopes: what a reader of the generator
    /// alone asks for (tabulate.rs `Model::table_reads`).
    pub(crate) fn middle_of(&self, plan: &Arc<Plan>, t: f64, amplitudes: &[Amplitude], near: Option<&NearValues>) -> Middle<'_> {
        let mut point = Held::<Site>::take();
        point.beams.clear();
        point.beams.extend_from_slice(amplitudes);
        point.select(plan);
        point.formed(plan, near);
        // A thread's spare point holds its last middle's: the normalized generator's forms are this point's, or none.
        point.normal.clear();
        let mut work = near.map_or(0.0, |n| n.work + plan.formed_work(point.directions.len()));
        if let (Some(_), Some(near)) = (plan.fourth.as_ref(), near) {
            work += self.normal_read(plan, None, Some(near), &mut point, true);
        }
        Middle { model: self, plan: plan.clone(), r: None, t, table: None, point: RefCell::new(point), work }
    }

    /// The one generator assembly (module documentation): a cluster's generator at a position and a time under a light set, the
    /// exact elimination at the point.
    pub fn assemble(&self, cluster: usize, r: [f64; 3], t: f64, lights: &Lights) -> Middle<'_> {
        self.middle(&self.plan(cluster, lights), r, t, None)
    }

    /// The run's own evaluation at a point, the one criterion one reads (elim.rs `Model::readings`): of plans of one
    /// cluster under one light set -- the run's, and those at the readings' next orders (`Model::plan_at`) -- each the
    /// static normalized generator at rest at `r` (`Middle::generated`'s time mean: every tone but the static one at its
    /// mean, nothing) and where the plan bends the landing map S (1 + Y_L)⁻¹ (`Quartic`), else none: the identity. The
    /// exact elimination at the point (criterion one measures the evaluation's truncation against the definition, both
    /// at the point: never a table's values, whose interpolation is the tables' own reading), once for every plan -- one
    /// set of responses, each plan's forms laid of it (`Model::forms`) -- and its values alone: the readings ask for no
    /// derivative (the run's force does).
    pub(crate) fn evaluations(&self, plans: &[Arc<Plan>], r: [f64; 3]) -> Vec<(Mat, Option<Mat>)> {
        let Some(first) = plans.first() else { return vec![] };
        let mut amplitudes = vec![];
        self.light.amplitudes_into(r, 0, |_| true, &mut amplitudes);
        let near = plans.iter().any(|plan| plan.near.len() > 0);
        let h = self.far_hamiltonian(&amplitudes, &first.lights, 0);
        let responses = near.then(|| self.eliminate(first.cluster, &h, &first.lights, 0, plans.iter().any(|plan| plan.mixes())));
        let energies: Vec<Vec<f64>> = h.iter().map(|m| (0..STATES).map(|s| m[(s, s)].re).collect()).collect();
        plans
            .iter()
            .map(|plan| {
                let n = plan.states;
                let near = responses.as_ref().filter(|_| plan.near.len() > 0).map(|responses| {
                    let len = plan.near.len();
                    let mut data = vec![C64::ZERO; len];
                    let work = self.forms(plan, responses, &energies, &mut data, &|_| true);
                    NearValues { len, order: 0, data, directions: None, work }
                });
                let middle = self.middle_of(plan, 0.0, &amplitudes, near.as_ref());
                let path = crate::step::Middle::path(&middle, [0.0; 3], [0.0; 3]);
                let mut generator = Mat::zeros(n, n);
                middle.generated(&path, 0.0, true, &mut generator);
                let landing = plan.fourth.as_ref().filter(|fourth| fourth.bends()).map(|fourth| {
                    let mut site = Held::<Site>::take();
                    site.beams.clear();
                    site.beams.extend_from_slice(&amplitudes);
                    QUARTIC.with(|kept| {
                        let mut q = kept.borrow_mut();
                        q.landing = true;
                        self.quartic_made(plan, fourth, None, near.as_ref(), &mut site, false, &mut q);
                        // Made of no keyed point: no site's read finds these matrices its own.
                        q.key = None;
                        let size = n * n;
                        Mat::from_fn(n, n, |x, y| q.data[LANDING * size + x * n + y])
                    })
                });
                (generator, landing)
            })
            .collect()
    }
}

/// The momentum's part of a cluster's generator along a straight path through its middle (`Middle::path`): the momentum
/// the generator is taken at (1/µm); per beam of the plan its Doppler shift w_l at it (rad/µs) and its path rate k_l·v
/// at the path's velocity (rad/µs); per state of the cluster its Doppler frame's rate K_s·v, and whether any frame
/// turns.
pub struct Path {
    kappa: [f64; 3],
    velocity: [f64; 3],
    rates: Held<Rates>,
    turning: bool,
    /// The path's own number among the paths made: what a node's fields along it are kept by (`Scratch::nodes`).
    id: u64,
}

/// The paths made so far: each one's own number (`Path::id`).
static PATHS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// A path's numbers (`Path`; a thread's spare, `Held`).
#[derive(Default)]
pub(crate) struct Rates {
    shifts: Vec<f64>,
    path: Vec<f64>,
    gauge: Vec<f64>,
}

/// A node's fields along a path (`Middle::along`): the beams' amplitudes turned along the path, the plan's field components
/// there (each with its gradient, plain and weighted by its beams' Doppler shifts) and, where asked, their derivatives along
/// the path; the tones' phases, the Doppler frames' turns. Kept by the path and the node's time (`key`).
#[derive(Default)]
struct Node {
    key: Option<(u64, u64, bool)>,
    turned: Vec<[C64; 4]>,
    plain: Vec<[C64; 4]>,
    doppler: Vec<[C64; 4]>,
    plain_slope: Vec<[C64; 4]>,
    doppler_slope: Vec<[C64; 4]>,
    phases: Vec<C64>,
    turns: Vec<C64>,
    /// The normalized generator's part at the node (`Model::node_normal`), made once its generator or force asks: S⁻¹
    /// there, and where the held part is anything E, the held part with its gradient, Y and E₀.
    normal: Vec<C64>,
    normal_made: bool,
    normal_gradients: bool,
}

/// The nodes a site keeps (`Scratch::nodes`): a step's path's three -- its middle, where the fixed point's tones are read
/// too, and its two ends -- and the one of the path its momentum is predicted on.
const NODES: usize = 4;

/// A site's scratch (`Site::scratch`): the nodes' fields along the paths through it, kept so that a step's generators and
/// forms at one node make them once (`next`: the place the next node takes, the oldest's); a ρ laid out as the entries are
/// and a generator's sum -- each the real parts on the cluster's places (`Plan::padded`) and then the imaginary parts --;
/// and per tone its three sums.
#[derive(Default)]
struct Scratch {
    nodes: [Node; NODES],
    next: usize,
    /// The normalized generator's matrices at a node where its held part is anything (`Model::node_normal`), and a
    /// node's normalized part in the making (`Middle::generator`).
    quartic: Quartic,
    temporary: Vec<C64>,
    /// The first order's decay sums at a node (`Middle::generator`: `Fourth::first_decay`).
    decayed: Vec<C64>,
    /// The gradient's sums, per axis its real and its imaginary parts, then their slopes' (`gradient`).
    gradients: Vec<f64>,
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
        Path { kappa, velocity, turning: gauge.iter().any(|rate| *rate != 0.0), rates, id: PATHS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) }
    }

    fn gauge(&self, path: &Path, rates: &mut [f64]) {
        rates.copy_from_slice(&path.rates.gauge);
    }

    /// Module documentation, # At a point, and along a path through it; # Dense terms; # The excited amplitude.
    fn generator(&self, path: &Path, s: f64, h: &mut Mat) {
        self.generated(path, s, false, h);
    }

    fn force(&self, path: &Path, s: f64, rho: &[C64]) -> [f64; 3] {
        let mut out = [0.0; 3];
        self.contract(path, Some(s), rho, None, &mut |_, f| out = f);
        out
    }

    fn tones(&self, path: &Path, rho: &[C64], each: &mut dyn FnMut(f64, [f64; 3])) {
        self.contract(path, None, rho, None, each)
    }

    /// `contract` at the middle with the fixed point's tones on `held` (`step::Middle::centre`).
    fn centre(&self, path: &Path, held: &[C64], weight: &dyn Fn(f64) -> f64, rho: &[C64]) -> [f64; 3] {
        let mut out = [0.0; 3];
        self.contract(path, Some(0.0), rho, Some((held, weight)), &mut |_, f| out = f);
        out
    }

    /// photon.rs `Model::scattering_on` this middle's own site, at the path's momentum and the middle's time, with the
    /// real multiply-adds it counted.
    fn scattering(&self, path: &Path, psi: &[C64]) -> (Scattering, f64) {
        let (scattering, work) = self.model.scattering_on(&self.plan, self.table, self.r, path.kappa, self.t, psi, &mut self.point.borrow_mut(), true);
        (scattering, work * crate::decide::COMPLEX)
    }

    /// Ĝ_a(s) as matrices: the forms' own terms (`contract`), each where it lies -- a flag's values times the gradient
    /// of its components' product, and its slopes along the point's directions times the product, at its tone's phase
    /// and in its states' Doppler frames --, then the Hermitian part of their weighted sum (of a block and its adjoint the
    /// form takes one twice: its Hermitian part is the pair's), and the normalized generator's gradient. A term onto the three
    /// axes' sums at once, their real and imaginary parts apart (`linalg::axpy`). Its slope along the path, where asked,
    /// in the same pass: along the path every factor of a term is a plane wave's or a tone's phase e^{iωs} or a sum of
    /// them, whose derivative is iω times each (`along`), a block's product by the product rule, the frames' turns
    /// e^{−iK_x·v s} theirs; the normalized generator is held along the path.
    /// Counted as three of the force's forms, six with the slope (`Middle::work`).
    fn gradient(&self, path: &Path, s: f64, g: &mut [Vec<C64>; 3], slope: Option<&mut [Vec<C64>; 3]>) -> f64 {
        let counted = if slope.is_some() { 6.0 } else { 3.0 } * crate::step::Middle::work(self)[1];
        let (plan, n, point) = (&*self.plan, self.plan.states, self.point.borrow());
        let mut scratch = point.scratch.borrow_mut();
        let sloped = slope.is_some();
        let at = self.along(&point.amplitudes, path, s, sloped, &mut scratch);
        let held = plan.fourth.as_ref().filter(|fourth| fourth.held() && !point.normal.is_empty());
        let Scratch { nodes, gradients, quartic, .. } = &mut *scratch;
        if let Some(fourth) = held {
            self.model.node_normal(plan, fourth, &point, &mut nodes[at], quartic, self.centre(s), true);
        }
        let Node { plain, doppler, plain_slope, doppler_slope, phases, turns, normal, .. } = &nodes[at];
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
        // The held part's gradient at the node, of its own forms, in the frames as the terms' entries.
        for (a, m) in g.iter_mut().enumerate() {
            if held.is_some() {
                for (place, z) in m.iter_mut().enumerate() {
                    *z += normal[(3 + a) * n * n + place] * turns[place / n] * turns[place % n].conj();
                }
            }
            hermitian(m, n);
        }
        counted
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
        self.plan.costs(self.point.borrow().directions.len())
    }

    /// The complex multiply-adds that made it (`Middle::work`), real.
    fn made(&self) -> f64 {
        self.work * crate::decide::COMPLEX
    }

    /// Module documentation, # Exact tones: each tone pair's beat Ω + Q·v on the path.
    fn beats(&self, path: &Path, beats: &mut Vec<f64>, keys: &mut Vec<Vec<i64>>) {
        let plan = &*self.plan;
        beats.extend(plan.firsts.iter().map(|&g| plan.tones[plan.groups[g].tone] + dot3(plan.groups[g].wave, path.velocity)));
        keys.extend(plan.firsts.iter().map(|&g| plan.groups[g].key.clone()));
    }

    /// The pair's first group's envelope and its partner's (`envelope`), constant along the path.
    fn envelopes(&self, path: &Path, m: usize, _: f64, plus: &mut Mat, minus: &mut Mat) {
        let plan = &*self.plan;
        let group = &plan.groups[plan.firsts[m]];
        self.envelope(path, group, plus);
        match group.partner {
            Some(p) => self.envelope(path, &plan.groups[p], minus),
            None => minus.data.fill(C64::ZERO),
        }
    }

    /// The pair's part of the force's forms: its first group's pieces on σ = re + i im, its partner's on σ† = re − i im
    /// (`group_force`), σ the nodes' sums summed (its envelopes are the same at every time of the path: `envelopes`).
    fn tone_force(&self, path: &Path, m: usize, sums: &[(f64, &[C64], &[C64])]) -> [f64; 3] {
        let (re, im) = crate::step::summed(sums);
        let (re, im) = (&re[..], &im[..]);
        let plan = &*self.plan;
        let group = &plan.groups[plan.firsts[m]];
        let mut out = self.group_force(path, group, re, im, 1.0);
        if let Some(p) = group.partner {
            let more = self.group_force(path, &plan.groups[p], re, im, -1.0);
            (0..3).for_each(|a| out[a] += more[a]);
        }
        out
    }

    /// Per piece of the pair's two groups: of an envelope its value, its two beams' product and the phase (four complex
    /// multiply-adds); of the force its value and slopes on σ and the product's four parts (eight, and two a direction).
    fn tone_work(&self, m: usize) -> [f64; 2] {
        let plan = &*self.plan;
        let group = &plan.groups[plan.firsts[m]];
        let pair = [Some(group), group.partner.map(|p| &plan.groups[p])];
        let count = |pieces: fn(&Group) -> &Vec<Piece>| pair.iter().flatten().map(|g| pieces(g).len()).sum::<usize>() as f64;
        let directions = self.point.borrow().directions.len() as f64;
        [4.0 * count(|g| &g.generator), (8.0 + 2.0 * directions) * count(|g| &g.form)]
    }
}

/// A piece's factor (module documentation, # Exact tones): its two beams' parts of the block's components at a node --
/// each beam's amplitude turned there, its weight, conjugated with its component --, plain or Doppler-weighted as the
/// flag says, and their product with its gradient (parts 1–3: the gradient of each times the other).
fn factor(plan: &Plan, node: &Node, path: &Path, block: usize, flag: usize, piece: &Piece) -> [C64; 4] {
    let side = |scalar: Option<usize>, beam: Option<usize>, doppler: bool| -> [C64; 4] {
        match scalar.zip(beam) {
            Some((s, j)) => {
                let scalar = &plan.scalars[s];
                let (at, c) = scalar.beams[j];
                let w = if doppler { path.rates.shifts[at] } else { 1.0 };
                std::array::from_fn(|part| (if scalar.conjugate { node.turned[at][part].conj() } else { node.turned[at][part] }) * c * w)
            }
            None => [C64::ONE, C64::ZERO, C64::ZERO, C64::ZERO],
        }
    };
    let b = &plan.blocks[block];
    let (l, r) = (side(b.left, piece.left, flag == LEFT), side(b.right, piece.right, flag == RIGHT));
    std::array::from_fn(|part| if part == 0 { l[0] * r[0] } else { l[part] * r[0] + l[0] * r[part] })
}

/// The normalized generator's matrices (`Quartic`; module documentation, # The normalized generator) from a point's (or a
/// node's) static Hermitian generator H_h, the first order's forms Y⁽ᵃ⁾ of each reference class a (`ys`, four parts each
/// after one another; `classes` each place's class, `count` their number), the metric X and the loss L (four parts each,
/// n × n row by row: the value and the gradient along the three lab axes) and the reference Z₀ (a place's value and
/// gradient), at the orders `orders` (elim.rs `Orders`) -- the Schrieffer–Wolff series, products of the forms alone
/// (ELIMINATION v7, v12):
/// - E₀ = H_h − (i/2) L − Z₀; at the series' order 1 the first order H¹ = −Σ_a Y⁽ᵃ⁾ E₀ P_a (P_a the places of class a:
///   column s of the first order is of s's own reference, Y⁽ᵃ⁾[x, t] = W′_x R₀(z_a)R₀(z_t) W_t), its gradient by the
///   product rule, and the held part Herm H¹ -- the Hermitian part is the series' own: what the normalization would do to
///   it is in the evaluation's error against the definition, measured where the atoms went and booked (elim.rs
///   `Model::readings`); nothing at the series' order 0; E = E₀ + H¹, what the first order of 𝒯 is of
///   (`Plan::excite_first`);
/// - S⁻¹ = Σ_{j≤J} binom(−½, j) X^j and S = Σ_{j≤J} binom(½, j) X^j at the normalization's order J, by Horner's rule
///   with the gradient's product rule (`series`) -- no eigendecomposition --;
/// - the landing map S (1 + Y_L)⁻¹ with (1 + Y_L)⁻¹ = Σ_{j≤J} (−Y_L)^j, Y_L[x, y] = Y⁽ᵃ⁾[x, y] at x's class a (the left and
///   right wave operators' overlap to the series' leading order; its error against the definition's map measured
///   where the atoms went, elim.rs `Fold::landing_error`).
/// Without `gradients` the values alone (the gradients' parts zero); the landing map only where asked (`landing`), else
/// zero. Into `data` (`Quartic::data`'s places); `scratch` holds what it makes on the way.
#[allow(clippy::too_many_arguments)]
pub(crate) fn normal(hermitian: &[C64], ys: &[C64], classes: &[usize], count: usize, metric: &[C64], loss: &[C64], n: usize, z0: &[[f64; 4]], orders: Orders, gradients: bool, landing: bool, data: &mut [C64], scratch: &mut Vec<C64>) {
    let size = n * n;
    let parts = if gradients { 4 } else { 1 };
    let order = orders.series;
    assert!(order <= 1, "the series' order 1 is `Unit::Series`, a higher one the nearest block's keep");
    sized(scratch, 20 * size);
    let (e, rest) = scratch.split_at_mut(4 * size);
    let (first, rest) = rest.split_at_mut(4 * size);
    let (inverse, rest) = rest.split_at_mut(4 * size);
    let (outer, rest) = rest.split_at_mut(4 * size);
    let (t, rest) = rest.split_at_mut(size);
    let (u, rest) = rest.split_at_mut(size);
    let w = &mut rest[..size];
    // E₀ = H_h − (i/2) L − Z₀.
    for i in 0..parts * size {
        e[i] = hermitian[i] - loss[i] * C64::new(0.0, 0.5);
    }
    for part in 0..parts {
        for (i, z) in z0.iter().enumerate().take(n) {
            e[part * size + i * n + i] -= C64::real(z[part]);
        }
    }
    // The series' first order H¹ = −Σ_a Y⁽ᵃ⁾ E₀ P_a, ∇H¹ = −Σ_a (∇Y⁽ᵃ⁾ E₀ + Y⁽ᵃ⁾ ∇E₀) P_a.
    first.fill(C64::ZERO);
    let class = |s: usize| if classes.is_empty() { 0 } else { classes[s] };
    if order >= 1 {
        for a in 0..count.max(1) {
            let y = &ys[a * 4 * size..(a * 4 + 4) * size];
            for part in 0..parts {
                product(&y[..size], &e[part * size..(part + 1) * size], t, n);
                if part > 0 {
                    product(&y[part * size..(part + 1) * size], &e[..size], u, n);
                    t.iter_mut().zip(u.iter()).for_each(|(z, more)| *z += *more);
                }
                for x in 0..n {
                    for s in (0..n).filter(|&s| class(s) == a) {
                        first[part * size + x * n + s] -= t[x * n + s];
                    }
                }
            }
        }
    }
    series(-0.5, orders.normalization, metric, inverse, t, n, gradients);
    data.fill(C64::ZERO);
    for part in 0..parts {
        let of = &first[part * size..(part + 1) * size];
        let held = &mut data[(HELD + part) * size..(HELD + part + 1) * size];
        for x in 0..n {
            for y in 0..n {
                held[x * n + y] = (of[x * n + y] + of[y * n + x].conj()) * 0.5;
            }
        }
    }
    data[BENT * size..(BENT + 4) * size].copy_from_slice(inverse);
    data[ORIGIN * size..(ORIGIN + 1) * size].copy_from_slice(&e[..size]);
    for (z, (e, h)) in data[OFFSET * size..(OFFSET + 4) * size].iter_mut().zip(e.iter().zip(first.iter())) {
        *z = *e + *h;
    }
    for a in 0..count.max(1) {
        data[(FIRST + a) * size..(FIRST + a + 1) * size].copy_from_slice(&ys[a * 4 * size..(a * 4 + 1) * size]);
    }
    // The landing map S Σ_{j≤J} (−Y_L)^j, by Horner's rule v ← 1 − v Y_L.
    if landing {
        let overlap = &mut outer[size..2 * size];
        for x in 0..n {
            let y = &ys[class(x) * 4 * size..];
            overlap[x * n..(x + 1) * n].copy_from_slice(&y[x * n..(x + 1) * n]);
        }
        let overlap = overlap.to_vec();
        series(0.5, orders.normalization, metric, outer, t, n, false);
        w.fill(C64::ZERO);
        for i in 0..n {
            w[i * n + i] = C64::ONE;
        }
        for _ in 0..orders.normalization {
            product(w, &overlap, t, n);
            for (z, y) in w.iter_mut().zip(t.iter()) {
                *z = -*y;
            }
            for i in 0..n {
                w[i * n + i] += C64::ONE;
            }
        }
        product(&outer[..size], w, &mut data[LANDING * size..(LANDING + 1) * size], n);
    }
}

/// (1 + X)^α to order J by Horner's rule, p ← p X + binom(α, j) from j = J down -- its first step b_J X + b_{J−1}
/// without a product --, into `out` (four parts: the value and, where `gradients`, ∇p ← ∇p X + p ∇X along the three
/// lab axes, else nothing); `t` its scratch.
fn series(alpha: f64, order: usize, x: &[C64], out: &mut [C64], t: &mut [C64], n: usize, gradients: bool) {
    let size = n * n;
    out.fill(C64::ZERO);
    let parts = if gradients { 4 } else { 1 };
    if order >= 1 {
        let top = binomial(alpha, order);
        for (z, y) in out[..parts * size].iter_mut().zip(&x[..parts * size]) {
            *z = *y * top;
        }
    }
    for i in 0..n {
        out[i * n + i] += C64::real(binomial(alpha, order.saturating_sub(1)));
    }
    for j in (0..order.saturating_sub(1)).rev() {
        if gradients {
            for a in 1..4 {
                let (value, rest) = out.split_at_mut(size);
                let d = &mut rest[(a - 1) * size..a * size];
                product(d, &x[..size], t, n);
                d.copy_from_slice(t);
                product(value, &x[a * size..(a + 1) * size], t, n);
                for (z, y) in d.iter_mut().zip(t.iter()) {
                    *z += *y;
                }
            }
        }
        product(&out[..size], &x[..size], t, n);
        out[..size].copy_from_slice(t);
        for i in 0..n {
            out[i * n + i] += C64::real(binomial(alpha, j));
        }
    }
}

/// Σ_{j≤J} binom(α, j) X^j of an n × n matrix X (row by row) into `out`, by Horner's rule p ← p X + c_j from j = J down
/// (`t` its scratch; its first step c_J X + c_{J−1} without a product): the normalization's series (`Model::bent_of`).
fn powers(alpha: f64, order: usize, x: &[C64], t: &mut [C64], out: &mut [C64], n: usize) {
    let size = n * n;
    out[..size].fill(C64::ZERO);
    if order >= 1 {
        let top = binomial(alpha, order);
        for (z, y) in out[..size].iter_mut().zip(&x[..size]) {
            *z = *y * top;
        }
    }
    for i in 0..n {
        out[i * n + i] += C64::real(binomial(alpha, order.saturating_sub(1)));
    }
    for j in (0..order.saturating_sub(1)).rev() {
        product(&out[..size], x, t, n);
        out[..size].copy_from_slice(&t[..size]);
        for i in 0..n {
            out[i * n + i] += C64::real(binomial(alpha, j));
        }
    }
}

/// c = a b of n × n matrices row by row.
fn product(a: &[C64], b: &[C64], c: &mut [C64], n: usize) {
    for i in 0..n {
        let row = &mut c[i * n..(i + 1) * n];
        row.fill(C64::ZERO);
        for k in 0..n {
            let x = a[i * n + k];
            if x == C64::ZERO {
                continue;
            }
            for (z, y) in row.iter_mut().zip(&b[k * n..(k + 1) * n]) {
                *z += x * *y;
            }
        }
    }
}

/// A node's normalized part (`Node::normal`) of the matrices `q` made of its components (`Model::node_normal`): S⁻¹
/// where the held part is nothing; else S⁻¹, E₀, the held part with its gradient, the first order's forms Y⁽ᵃ⁾ of each
/// reference class (`Middle::held_force`) and E (`Middle::generator`: the first order of 𝒯).
fn node_of(fourth: &Fourth, node: &mut Node, q: &Quartic) {
    let size = fourth.n * fourth.n;
    let data = &q.data;
    if !fourth.held() {
        sized(&mut node.normal, size);
        node.normal.copy_from_slice(&data[BENT * size..(BENT + 1) * size]);
        return;
    }
    let classes = fourth.count.max(1);
    sized(&mut node.normal, (7 + classes) * size);
    let normal = &mut node.normal;
    normal[..size].copy_from_slice(&data[BENT * size..(BENT + 1) * size]);
    normal[size..2 * size].copy_from_slice(&data[ORIGIN * size..(ORIGIN + 1) * size]);
    normal[2 * size..6 * size].copy_from_slice(&data[HELD * size..(HELD + 4) * size]);
    normal[6 * size..(6 + classes) * size].copy_from_slice(&data[FIRST * size..(FIRST + classes) * size]);
    normal[(6 + classes) * size..(7 + classes) * size].copy_from_slice(&data[OFFSET * size..(OFFSET + 1) * size]);
}

/// The key of a point's matrices (`Model::quartic`): the plan, the table's grid and the position.
fn quartic_key(plan: &Plan, table: Option<&NearTable>, r: [f64; 3]) -> (u64, u64, [u64; 3]) {
    let grid = table.map_or(0, |t| t.table.axes().iter().fold(1 + crate::table::Table::nodes(t.table.axes()) as u64, |h, a| h.rotate_left(13) ^ a.spacing().to_bits() ^ a.node(0).to_bits().rotate_left(29)));
    (plan.id, grid, r.map(f64::to_bits))
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
    /// The generator at a node (`generator`), or where `mean` its time mean: every tone but the static one at its mean,
    /// nothing -- the run's static normalized generator, what the readings measure (`Model::evaluations`).
    fn generated(&self, path: &Path, s: f64, mean: bool, h: &mut Mat) {
        let (plan, n, point) = (&*self.plan, self.plan.states, self.point.borrow());
        let mut scratch = point.scratch.borrow_mut();
        let at = self.along(&point.amplitudes, path, s, false, &mut scratch);
        let normalized = plan.fourth.as_ref().filter(|fourth| fourth.bends() && !point.normal.is_empty());
        let Scratch { nodes, sum, quartic, temporary, decayed, .. } = &mut *scratch;
        if let Some(fourth) = normalized {
            self.model.node_normal(plan, fourth, &point, &mut nodes[at], quartic, self.centre(s), false);
        }
        let Node { plain, doppler, phases, turns, normal, .. } = &nodes[at];
        let means: Vec<C64>;
        let phases = if mean {
            means = (0..plan.tones.len()).map(|tone| if tone == 0 { C64::ONE } else { C64::ZERO }).collect();
            &means
        } else {
            phases
        };
        let padded = plan.padded;
        sum.clear();
        sum.resize(2 * padded, 0.0);
        let (re, im) = sum.split_at_mut(padded);
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
        // The decay (module documentation, # The excited amplitude): the terms' anti-Hermitian part −(i/2)Λ, Λ = i(h − h†)
        // -- the pair forms' Σ 𝒯†D𝒯 of the node's components at the path's momentum --; where the generator bends, in the
        // normalized coordinates S⁻¹(Λ + Λ⁽¹⁾)S⁻¹ with the series' first order's Λ⁽¹⁾ of the node's E (`Fourth::first_decay`)
        // and the held part beside it -- in the entries' own coordinates before the frames turn them, as every entry is.
        if let Some(fourth) = normalized {
            let size = n * n;
            sized(temporary, 3 * size);
            let (lambda, rest) = temporary.split_at_mut(size);
            let (left, right) = rest.split_at_mut(size);
            for x in 0..n {
                for y in 0..n {
                    let (a, b) = (h.data[x * n + y], h.data[y * n + x].conj());
                    lambda[x * n + y] = C64::I * (a - b);
                    left[x * n + y] = (a + b) * 0.5;
                }
            }
            h.data.copy_from_slice(left);
            if fourth.held() {
                let e = (6 + fourth.count.max(1)) * size;
                let value = |at: usize| C64::new(point.normal[2 * at], point.normal[2 * at + 1]);
                fourth.first_decay(plain, doppler, phases, &value, &normal[e..e + size], decayed, lambda);
                for (z, more) in h.data.iter_mut().zip(&normal[2 * size..3 * size]) {
                    *z += *more;
                }
            }
            let bent = &normal[..size];
            product(bent, lambda, left, n);
            product(left, bent, right, n);
            for (z, l) in h.data.iter_mut().zip(right.iter()) {
                *z -= *l * C64::new(0.0, 0.5);
            }
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
    }

    /// This middle with its generator's near values moved by `by` and its responding forms by `responses` (a table's error
    /// read of them, tabulate.rs; real and imaginary parts).
    pub(crate) fn moved(&self, by: &[f64], responses: &[f64]) -> Middle<'_> {
        let (mut point, from) = (Held::<Site>::take(), self.point.borrow());
        assert_eq!(by.len(), from.generator.len(), "an error read of the generator's near values");
        assert_eq!(responses.len(), 2 * from.responses.len(), "an error read of the responding forms");
        point.beams.clone_from(&from.beams);
        point.amplitudes.clone_from(&from.amplitudes);
        point.place = from.place;
        point.generator.clear();
        point.generator.extend(from.generator.iter().zip(by).map(|(value, error)| value + error));
        point.forms.clone_from(&from.forms);
        point.slopes.clone_from(&from.slopes);
        point.directions.clone_from(&from.directions);
        point.responses.clear();
        point.responses.extend(from.responses.iter().zip(responses.chunks_exact(2)).map(|(value, error)| *value + C64::new(error[0], error[1])));
        point.response_slopes.clone_from(&from.response_slopes);
        point.normal.clone_from(&from.normal);
        point.normal_slopes.clone_from(&from.normal_slopes);
        Middle { model: self.model, plan: self.plan.clone(), r: self.r, t: self.t, table: self.table, point: RefCell::new(point), work: self.work }
    }

    /// The middle's table and position where the node at the time s is the middle (`Model::node_normal`).
    fn centre(&self, s: f64) -> Option<(Option<&NearTable>, [f64; 3])> {
        self.r.filter(|_| s == 0.0).map(|r| (self.table, r))
    }

    /// The held part's force form at a node (module documentation, # The normalized generator): Re tr(∇_a held ρ) on the
    /// density `laid` as the terms' entries are (`contract`: ρ_t[y, x] the laid entry x ← y), of the node's own forms. At
    /// the series' order 1 the held part is Herm H¹, H¹ = −Σ_a Y⁽ᵃ⁾ E₀ P_a, E₀ = H_h − (i/2)L − Z₀ (`normal`), so with ρ_t
    /// Hermitian
    ///   Re tr(∇held ρ_t) = −Σ_a Re tr(∇Y⁽ᵃ⁾ A_a) − Re tr(∇H_h Herm B) + Re tr((i/2)∇L B) + Σ_y ∇z_y Re B[y, y],
    ///   A_a = E₀ P_a ρ_t, B = Σ_a P_a ρ_t Y⁽ᵃ⁾:
    /// the forms' gradients contracted with these matrices term by term, as the force's own form is (`contract`) -- the
    /// static Hermitian terms with Herm B by their weights; the partners (Y⁽ᵃ⁾ at the entry's right place's own class a,
    /// 2 Σ of the absorbed component's terms and the emitted one's adjoints, `normal_at`) with A' and A'† twice, A'[t, x] =
    /// A_{a(t)}[t, x]; the crossed forms with their own class's A_a; the loss with (i/2)B through the near part of 𝒯 and
    /// its gradient (`Plan::metric`); each place's reference's slope with B's diagonal there --: no gradient of the
    /// generator is made.
    #[allow(clippy::too_many_arguments)]
    fn held_force(&self, fourth: &Fourth, point: &Site, plain: &[[C64; 4]], normal: &[C64], laid: &[f64], temporary: &mut Vec<C64>, lays: &mut Vec<f64>) -> [f64; 3] {
        let (plan, n) = (&*self.plan, fourth.n);
        let (size, padded) = (n * n, plan.padded);
        let classes = fourth.count.max(1);
        let class = |s: usize| if fourth.classes.is_empty() { 0 } else { fourth.classes[s] };
        let e0 = &normal[size..2 * size];
        let ys = &normal[6 * size..(6 + classes) * size];
        sized(temporary, (3 + classes) * size);
        let (rho, rest) = temporary.split_at_mut(size);
        let (b, rest) = rest.split_at_mut(size);
        let (prime, a_all) = rest.split_at_mut(size);
        for x in 0..n {
            for z in 0..n {
                rho[z * n + x] = C64::new(laid[x * n + z], laid[padded + x * n + z]);
            }
        }
        // A_a = E₀ P_a ρ, A' of the rows' own classes, B = Σ_a P_a ρ Y⁽ᵃ⁾.
        for a in 0..classes {
            let at = &mut a_all[a * size..(a + 1) * size];
            for t in 0..n {
                for x in 0..n {
                    at[t * n + x] = (0..n).filter(|&s| class(s) == a).map(|s| e0[t * n + s] * rho[s * n + x]).sum();
                }
            }
        }
        for t in 0..n {
            for x in 0..n {
                prime[t * n + x] = a_all[class(t) * size + t * n + x];
            }
        }
        for s in 0..n {
            let y = &ys[class(s) * size..(class(s) + 1) * size];
            for t in 0..n {
                b[s * n + t] = (0..n).map(|x| rho[s * n + x] * y[x * n + t]).sum();
            }
        }
        // The three matrices laid as ρ is (entry x ← y holds M[y, x]), real parts and then imaginary parts: Herm B, A', A'†.
        sized(lays, 6 * padded);
        for x in 0..n {
            for z in 0..n {
                let (at, h) = (x * n + z, (b[z * n + x] + b[x * n + z].conj()) * 0.5);
                (lays[at], lays[padded + at]) = (h.re, h.im);
                (lays[2 * padded + at], lays[3 * padded + at]) = (prime[z * n + x].re, prime[z * n + x].im);
                let adjoint = prime[x * n + z].conj();
                (lays[4 * padded + at], lays[5 * padded + at]) = (adjoint.re, adjoint.im);
            }
        }
        let width = point.forms.len();
        let mut out = [0.0; 3];
        for term in plan.form.iter().filter(|term| plan.tones[term.tone] == 0.0) {
            let block = &plan.blocks[term.block];
            let (which, factor) = match term.flag {
                PLAIN => (0, -term.weight),
                flag => {
                    let shifted = if flag == RIGHT { block.right } else { block.left };
                    (if shifted.is_some_and(|s| plan.scalars[s].conjugate) { 2 } else { 1 }, -2.0)
                }
            };
            let (re, im) = lays[2 * which * padded..2 * (which + 1) * padded].split_at(padded);
            let span = term.first..term.first + term.len;
            let (real, imaginary) = plan.values(term, &point.forms);
            let value = dot(real, imaginary, &re[span.clone()], &im[span.clone()]);
            let (l, r) = (side(plain, block.left), side(plain, block.right));
            let c: [C64; 4] = std::array::from_fn(|part| if part == 0 { l[0] * r[0] } else { l[part] * r[0] + l[0] * r[part] });
            let mut gradient = [C64::ZERO; 3];
            for (f, direction) in point.directions.iter().enumerate().filter(|_| term.near) {
                let (real, imaginary) = point.slopes[f * width + term.at..f * width + term.at + 2 * term.len].split_at(term.len);
                let slope = dot(real, imaginary, &re[span.clone()], &im[span.clone()]);
                for i in 0..3 {
                    gradient[i] += slope * direction[i];
                }
            }
            for i in 0..3 {
                out[i] += factor * (c[1 + i] * value + c[0] * gradient[i]).re;
            }
        }
        // The crossed forms' entries, each with its own class's A_a: −Re Σ ∇Y⁽ᵃ⁾[x, y] A_a[y, x], block by block of their
        // components' product and gradients once.
        let count = plan.near.normalizing().len();
        let (values, slopes) = (&point.normal, &point.normal_slopes);
        let dn = point.directions.len();
        assert!(dn <= DIRECTIONS, "a site's directions");
        for (left, right, group) in fourth.groups() {
            let (mut value, mut along) = (C64::ZERO, [C64::ZERO; DIRECTIONS]);
            for &(e, a, xy) in group {
                let weight = a_all[a * size + (xy % n) * n + xy / n];
                value += C64::new(values[2 * e], values[2 * e + 1]) * weight;
                for (f, sum) in along.iter_mut().enumerate().take(dn) {
                    *sum += C64::new(slopes[(f * count + e) * 2], slopes[(f * count + e) * 2 + 1]) * weight;
                }
            }
            let (l, r) = (side(plain, left), side(plain, right));
            let c0 = l[0] * r[0];
            for k in 0..3 {
                let gradient = l[1 + k] * r[0] + l[0] * r[1 + k];
                let slope = point.directions.iter().zip(&along).fold(C64::ZERO, |sum, (d, a)| sum + *a * d[k]);
                out[k] -= (gradient * value + c0 * slope).re;
            }
        }
        // The loss: +Re tr((i/2) ∇L B), ∇L of the loss's decay sum at the node's components (`Sum::Loss`): the components'
        // gradients and the forms' slopes along the site's directions.
        let value = |e: usize| C64::new(values[2 * e], values[2 * e + 1]);
        let slope = |f: usize, e: usize| C64::new(slopes[(f * count + e) * 2], slopes[(f * count + e) * 2 + 1]);
        let mut loss = vec![C64::ZERO; (4 + dn) * size];
        fourth.sum_into(1, plain, None, None, &value, Some((&slope, dn)), true, &mut loss);
        for k in 0..3 {
            let trace: C64 = (0..n)
                .flat_map(|x| (0..n).map(move |y| (x, y)))
                .map(|(x, y)| {
                    let xy = x * n + y;
                    let gradient = (0..dn).fold(loss[(1 + dn + k) * size + xy], |g, f| g + loss[(1 + f) * size + xy] * point.directions[f][k]);
                    gradient * b[y * n + x]
                })
                .sum();
            out[k] += (trace * C64::new(0.0, 0.5)).re;
        }
        let reference = plan.near.crossed.len();
        for k in 0..3 {
            for i in 0..plan.near.reference {
                let slope: f64 = point.directions.iter().enumerate().map(|(f, d)| slopes[(f * count + reference + i) * 2] * d[k]).sum();
                out[k] += slope * b[i * n + i].re;
            }
        }
        out
    }

    /// A node's fields (`Node`) at the time s from the middle along a path, made into the site's scratch unless it keeps
    /// them, and their place there: each beam's amplitude and gradient at the middle times its plane wave's phase over
    /// the path, e^{i k_l·v s}, the field components of those -- and where `sloped`, their derivatives along the path --,
    /// with the tones' phases at that time and the Doppler frames' turns. At s = 0 they are the fixed middle point's own,
    /// at its time (every phase over the path is one there): the fixed point's tones read them too.
    /// A beating group's envelope (module documentation, # Exact tones): over its pieces the term's value at the entry
    /// times its two beams' product (`factor`) and the tone's phase, at the middle -- the plane waves and the frames' turns
    /// at s = 0 --, into `out` (n × n): on the path the group's part of Ĥ at the time s is this times e^{iνs}.
    fn envelope(&self, path: &Path, group: &Group, out: &mut Mat) {
        let (plan, point) = (&*self.plan, self.point.borrow());
        let mut scratch = point.scratch.borrow_mut();
        let at = self.along(&point.amplitudes, path, 0.0, false, &mut scratch);
        let node = &scratch.nodes[at];
        out.data.fill(C64::ZERO);
        for piece in &group.generator {
            let term = &plan.generator[piece.term];
            let (real, imaginary) = plan.values(term, &point.generator);
            let value = C64::new(real[piece.offset], imaginary[piece.offset]);
            out.data[term.first + piece.offset] += value * factor(plan, node, path, term.block, term.flag, piece)[0] * node.phases[term.tone];
        }
    }

    /// A beating group's part of the force's form on a density σ = re + i sign im (n × n row by row each, re and im
    /// Hermitian), per lab axis: as `contract` takes a term at the middle -- each piece's value and its slopes on σ[y, x]
    /// at its tone's phase, times its two beams' product and its gradient (`factor`), Re of their sum by the term's
    /// weight.
    fn group_force(&self, path: &Path, group: &Group, re: &[C64], im: &[C64], sign: f64) -> [f64; 3] {
        let (plan, n, point) = (&*self.plan, self.plan.states, self.point.borrow());
        let mut scratch = point.scratch.borrow_mut();
        let at = self.along(&point.amplitudes, path, 0.0, false, &mut scratch);
        let node = &scratch.nodes[at];
        let width = point.forms.len();
        let mut out = [0.0; 3];
        for piece in &group.form {
            let term = &plan.form[piece.term];
            let place = term.first + piece.offset;
            let (x, y) = (place / n, place % n);
            let sigma = (re[y * n + x] + im[y * n + x] * C64::new(0.0, sign)) * node.phases[term.tone];
            let (real, imaginary) = plan.values(term, &point.forms);
            let value = C64::new(real[piece.offset], imaginary[piece.offset]) * sigma;
            let mut gradient = [C64::ZERO; 3];
            for (f, direction) in point.directions.iter().enumerate().filter(|_| term.near) {
                let slopes = &point.slopes[f * width + term.at..f * width + term.at + 2 * term.len];
                let slope = C64::new(slopes[piece.offset], slopes[term.len + piece.offset]) * sigma;
                for a in 0..3 {
                    gradient[a] += slope * direction[a];
                }
            }
            let c = factor(plan, node, path, term.block, term.flag, piece);
            for a in 0..3 {
                out[a] += term.weight * (c[1 + a] * value + c[0] * gradient[a]).re;
            }
        }
        out
    }

    fn along(&self, amplitudes: &[[C64; 4]], path: &Path, s: f64, sloped: bool, scratch: &mut Scratch) -> usize {
        let plan = &*self.plan;
        let key = (path.id, s.to_bits());
        if let Some(at) = scratch.nodes.iter().position(|node| node.key.is_some_and(|(id, time, slopes)| (id, time) == key && (slopes || !sloped))) {
            return at;
        }
        let at = scratch.next;
        scratch.next = (at + 1) % NODES;
        let node = &mut scratch.nodes[at];
        node.key = Some((key.0, key.1, sloped));
        node.normal_made = false;
        node.turned.clear();
        node.turned.extend(amplitudes.iter().zip(&path.rates.path).map(|(a, rate)| {
            let phase = if s == 0.0 { C64::ONE } else { C64::cis(rate * s) };
            a.map(|part| part * phase)
        }));
        node.plain.clear();
        node.doppler.clear();
        node.plain_slope.clear();
        node.doppler_slope.clear();
        for scalar in &plan.scalars {
            let (mut plain, mut doppler, mut plain_slope, mut doppler_slope) = ([C64::ZERO; 4], [C64::ZERO; 4], [C64::ZERO; 4], [C64::ZERO; 4]);
            for &(at, c) in &scalar.beams {
                // Along the path a beam turns at its rate, e^{i rate s}, its conjugate the other way.
                let turn = C64::new(0.0, if scalar.conjugate { -path.rates.path[at] } else { path.rates.path[at] });
                for part in 0..4 {
                    let z = (if scalar.conjugate { node.turned[at][part].conj() } else { node.turned[at][part] }) * c;
                    plain[part] += z;
                    doppler[part] += z * path.rates.shifts[at];
                    if sloped {
                        plain_slope[part] += z * turn;
                        doppler_slope[part] += z * turn * path.rates.shifts[at];
                    }
                }
            }
            node.plain.push(plain);
            node.doppler.push(doppler);
            if sloped {
                node.plain_slope.push(plain_slope);
                node.doppler_slope.push(doppler_slope);
            }
        }
        node.phases.clear();
        node.phases.extend(plan.tones.iter().map(|&omega| if omega == 0.0 { C64::ONE } else { C64::cis(omega * (self.t + s)) }));
        node.turns.clear();
        node.turns.extend(path.rates.gauge.iter().map(|rate| if s != 0.0 && path.turning { C64::cis(-rate * s) } else { C64::ONE }));
        at
    }

    /// Re tr(Ĝ_a(s) ρ) per lab axis a: the gradient of the generator's Hermitian part along a path at the time s from the
    /// middle (`on`), or at the fixed middle point (None: every tone in turn, to `each` with its beat Ω, its entries at their
    /// phase at the middle's time), contracted with a Hermitian ρ on the cluster's states (n × n, row by row) without forming
    /// it: per dense term Σ_entries V[x, y] ρ[y, x] and the same of its slopes, times the fields' product and its gradient
    /// -- of a flag's values and their adjoints, whose forms with a Hermitian ρ are conjugates, one, twice (`Term::weight`).
    /// With `held` = (σ, w) the middle's tones on σ as well, each tone m's entries contracted with ρ + w(Ω_m) σ: at s = 0
    /// the fixed point's tones are the path's own forms there (`along`), so the two are one pass of the terms.
    fn contract(&self, path: &Path, on: Option<f64>, rho: &[C64], held: Option<(&[C64], &dyn Fn(f64) -> f64)>, each: &mut dyn FnMut(f64, [f64; 3])) {
        let (plan, n, point) = (&*self.plan, self.plan.states, self.point.borrow());
        let mut scratch = point.scratch.borrow_mut();
        let s = on.unwrap_or(0.0);
        assert!(held.is_none() || s == 0.0, "the fixed point's tones with the forms at the middle alone");
        let at = self.along(&point.amplitudes, path, s, false, &mut scratch);
        let normalized = plan.fourth.as_ref().filter(|fourth| fourth.held() && !point.normal.is_empty());
        let Scratch { nodes, weights, sums, quartic, temporary, sum: lays, .. } = &mut *scratch;
        if let Some(fourth) = normalized {
            self.model.node_normal(plan, fourth, &point, &mut nodes[at], quartic, self.centre(s), false);
        }
        let Node { plain, doppler, phases, turns, normal, normal_gradients, .. } = &nodes[at];
        // ρ[y, x] in the Doppler frames along the path, laid out as the entries x ← y are: one, or one a tone with σ.
        let laid = if held.is_some() { plan.tones.len() } else { 1 };
        weights.clear();
        weights.resize(laid * 2 * plan.padded, 0.0);
        for (m, laid) in weights.chunks_exact_mut(2 * plan.padded).enumerate() {
            let (re, im) = laid.split_at_mut(plan.padded);
            let more = held.map(|(sigma, w)| (sigma, w(plan.tones[m])));
            for (x, (re, im)) in re[..n * n].chunks_exact_mut(n).zip(im[..n * n].chunks_exact_mut(n)).enumerate() {
                for (y, (re, im)) in re.iter_mut().zip(im.iter_mut()).enumerate() {
                    let at = y * n + x;
                    let density = more.map_or(rho[at], |(sigma, w)| rho[at] + sigma[at] * w);
                    let w = density * turns[x] * turns[y].conj();
                    (*re, *im) = (w.re, w.im);
                }
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
            let laid = &weights[if held.is_some() { term.tone } else { 0 } * 2 * plan.padded..][..2 * plan.padded];
            let (re, im) = laid.split_at(plan.padded);
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
        // The held part's gradient at the node, of its own forms, a static term's (its one tone, the first), on the
        // density laid in the frames as the terms' entries are: its matrices' trace where the node holds them (the
        // middle's, the point's own: `Model::node_normal`), else its force form (`held_force`).
        if let Some(fourth) = normalized {
            let laid = &weights[..2 * plan.padded];
            let force = if *normal_gradients {
                let (re, im) = laid.split_at(plan.padded);
                std::array::from_fn(|a| (0..n * n).map(|place| (normal[(3 + a) * n * n + place] * C64::new(re[place], im[place])).re).sum())
            } else {
                self.held_force(fourth, &point, plain, normal, laid, temporary, lays)
            };
            for a in 0..3 {
                sums[0][a] += force[a];
            }
        }
        for (m, sum) in sums.iter().enumerate() {
            each(if on.is_some() { 0.0 } else { plan.tones[m] }, *sum);
        }
    }
}

#[cfg(test)]
mod tests;
