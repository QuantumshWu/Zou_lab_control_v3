//! The fold at the atom (DESIGN.md, final 2026-11-15; onefold step 1, not yet the run's): at a point of the motion the
//! Hamiltonian of the light there, grouped by laser frequency; every Floquet node off P -- each component's excited replica and
//! the ground states' copies its emission reaches -- folded by one Schur complement at one energy, one block per node offset;
//! and of the one excited amplitude it gives, the light shifts and the decay (the compression's Hermitian and anti-Hermitian
//! parts), the jumps, the metric, the force (by the adjoint) and the fold's own error at every node.
//!
//! # The Hamiltonian at a point, by laser frequency
//! The ground states G (`GROUND`) in the lab frame, the excited states E (`EXCITED`) in the request's frame ν_frame. At r,
//! H_GG(r) = H0_GG + Σ_ω O_ω,GG(E_ω(r)) and 𝒦(r) = H0_EE + Σ_ω O_ω,EE(E_ω(r)) − (i/2)D, the outside forms O_ω quadratic in
//! each frequency's coherent field (H0 holds their diagonal at the focus: request.rs `held`). Every frequency's light is two components c (`Part`): the
//! co-rotating V_c = Σ_l a_l X_l at e^{−iν_c t}, ν_c = δ_ω, and the counter-rotating Σ_l a_l* R_l at ν_c = −(δ_ω + 2ν_frame);
//! one rule for both and for every frequency, the tweezer's too (DESIGN §1.4). Absorbing beam l takes the momentum q_l = k_l
//! (the counter-rotating −k_l) and raises the excited state's energy by w_l = (ħ/m)(q_l·κ + q_l²/2) (DESIGN §1.2, onefold 1).
//!
//! # P, sets and frames (DESIGN §1.1)
//! Every coupling e (light joining s to s′, or a fold's Raman coupling) has a mismatch Δ_e = E_s − E_s′ − ω_e and a
//! strength a_e that no frame changes; a frame only shares Δ_e out between the two diagonals and the edge's residual, and a
//! loop's residuals sum to its beat whatever the frames. The one frame-free weight is the mixing m_e = |a_e|/|Δ_e|, and one
//! rule decides with it (`Choice::of`): a coupling whose mixing w(m) = ½(1 − 1/√(1 + 4m²)) -- a copy's population, the
//! copies' own reading -- is beyond the share cannot be folded, is static, and joins its two ends; one within it is a copy,
//! folded with its residual by (ii). The sets are the connected parts of the couplings that cannot be folded -- a forest,
//! not one tree over all states (a tree over all would make the 6.8 GHz couplings of F = 1 and F = 2 static: a set whose
//! diagonal spans GHz, the root of the per-state references and their r² mixed forms). Inside a set the mixing's maximum
//! spanning tree is static (cluster.rs `frames`); its other couplings carry their loops' beats, the integrator's. Each set's
//! whole frame is the exact frequency -- zero or a beat of two components -- that centres its diagonal.
//!
//! P holds one set's ground states, each in its frame θ_s (ψ_s = e^{−iθ_s t} x_s), and the explicit excited states, each at
//! its node's offset. A node is a level (a ground F or an excited F′) at an offset (exact, `Beat`): component c from ground
//! state s reaches the excited level at o = θ_s + ν_c; the excited node at o, emitting component g, reaches the ground copy
//! at ω = o − ν_g -- a copy of state t wherever t is not P's or θ_t ≠ ω. The lattice stops there (second order, DESIGN
//! §1.1; the copies' reabsorption is read, `Reading::truncation`).
//!
//! # Blocks
//! Each offset o is a block (`Block`): its explicit states (P, the E rows), its other excited states and its copies (the Q
//! rows). Its matrix in its frame, Ĥ_o: 𝒦 on the excited rows less o; H_GG on each copy offset's rows less ω; V_g between an
//! excited row and a copy of g. Its sources: V_c from each ground state s with θ_s + ν_c = o (beam by beam), 𝒦 and V_g† from its
//! E rows, H_GG from a ground state at a copy's offset. Every Q node -- an excited level or a copy of a ground level -- has
//! the same tiers: solved self-consistently in the block's decomposition (`Status::Decomposed`), taken at the block's
//! centre z₀ (`Centred`: one pass less, its rows C eliminated first, R₀ = (z₀ − Ĥ_CC)⁻¹, K_eff = Ĥ_SS + Ĥ_SC R₀ Ĥ_CS, every
//! split's source on S dressed by Ĥ_SC R₀ B_C), left out of the block (`Omitted`), or kept in P (`Kept`: a copy's tone, an
//! excited level's explicit place); one evaluation path whatever the choice, and each choice's error read by the same
//! function (`Reading`, DESIGN §1.2, M18). An excited node's centring moves its response by exactly (z₀ − z)R_N(z₀)T_N.
//!
//! # One linear solve per pass
//! For the static P generator H (x coordinates), each block's response T_o (Q rows × n) solves, per source split (a beam with
//! its Doppler shift w, or the static part), the Sylvester equation T H − (K_eff + w) T = B -- each eigenstate at its own
//! eigenvalue, z = λ + o − w, without H's eigenvectors (DESIGN §1.5, onefold 3): with H = U t U† and K_eff = V T_K V† (complex
//! Schur forms, part by part), Z = V† T U solves column by column a triangular system (`sylvester`), at an exceptional
//! point too. The sources go through the decomposition by the field's basis, a beam only combining them (the ranks); the
//! splits' solutions are summed in the Schur coordinates and taken back once. The centred copies' rows follow,
//! T_C = R₀ (Ĥ_CS T_S + B_C).
//!
//! # One decomposition along the path
//! What a block decomposes, K_eff + o, changes along a path only with the light that reaches it: the outside forms' fields
//! (the far light's intensity, slowly) and the copies' couplings (`Path`). Its Schur form V T V† made at one point serves the
//! next ones with its diagonal moved to first order, T_kk + (V†δK V)_kk -- exact in z, in o, in every beam's Doppler shift,
//! which only move the diagonal --, and what that leaves, the part of δK̃ = V†δK V off the diagonal, is read by the
//! evaluation order's function, the block perturbation R(z) δK T, bounded row by row: the first-order error of row i of a
//! column is (Σ_{k≠i} δK̃_ik Z_k)/(c − κ_i), at most Σ_P ‖δK̃_i,P∖i‖ ‖Z_P∖i‖ / |c − κ_i| over the decomposition's connected
//! parts P, relative to the column's whole response (the readings' unit). Beyond the share the decomposition is made again
//! at the point and the passes solved again. Blocks of the same rows nothing centred dresses decompose the same 𝒦 and share
//! one. A Schur form's entries off the diagonal within its rounding are zero, what they held read with the rest.
//!
//! # The exact copies after the excited block (DESIGN §3.4 of the replan)
//! A copy taken exactly (`Status::Decomposed`) is not in the decomposition kept along the path: its couplings to the
//! excited rows are the near light's and change every step. The block is the same, its elimination order the other one --
//! the excited rows first, in the kept basis (their resolvent triangular there), then the copies by their n_D × n_D Schur
//! complement (c − K_DD) − Ỹ′(c − T_E)⁻¹Ỹ at every energy c (Woodbury), Ỹ = V_E†K_ED V_D and Ỹ′ = V_D†K_DE V_E taken at the
//! point. The same block solve as one decomposition of the whole, exactly.
//!
//! # The compression
//! With Φ = [1; T_o] on the lifted space, A = Φ†ĤΦ = Ĥ_PP + Σ_o (Ĥ_Po T_o + T_o†Ĥ_oP + T_o†Ĥ_oo T_o) and M = 1 + Σ_o T_o†T_o
//! (the copies' time-averaged population with it); S = M^{1/2}, G = S⁻¹AS⁻¹. Its anti-Hermitian part is
//! −(i/2) S⁻¹(Σ_o ε_o†D ε_o)S⁻¹, ε_o the block's excited amplitude (its E rows the explicit states themselves, its excited Q rows
//! T_o): Σ_q L_q†L_q with L_q = C_q Σ_o e^{−iot} ε_o S⁻¹, for any T and any Hermitian S (DESIGN §1.3, §2.6).
//!
//! # Tones
//! What stays in P and is not static: an entry between P states of different frames (H_GG, or 𝒦 between explicit states of two
//! offsets, its Hermitian part) at θ_t − θ_s whose coupling cannot be folded (`Fold::turning`: its peak strength over the
//! states' energy difference read as the sets are, `unfoldable`; one that can is a copy left out, its population within
//! the share); a
//! kept copy's emission Γ_t†V_g†ε_o at θ_t − ω and its partner; and the beat of the two blocks a kept copy relates,
//! ε_o'†(𝒦 − ν̄)ε_o at o' − o (o' = θ_t + ν_g; ν̄ the columns' offsets from their own frames, gauge invariant, `Fold::cross`):
//! the copy's excited node is reached both from block o and from its state's sideband through its own block, one Floquet node, so their interference (the rate's, with
//! the decay -(i/2)ε_o'†Dε_o) is physics. Two blocks no kept copy relates are different Floquet nodes of the Hamiltonian:
//! their cross terms ε_o'†(𝒦 − ν̄)ε_o are not taken -- with the beating metric dropped (M8) they would turn at |o − o'|
//! with a size |o − o'|·|ε_o'†ε_o|/2 and a second-order shift of nothing in the exact Floquet problem (onefold report, step
//! 1); what one block's light does to another's ground states is the copies'.
//!
//! Not so the jumps. A photon is no laser component: the jump operators are three, one per emission axis q, over every
//! block, L_q(t) = C_q Σ_o e^{−iot}ε_o S⁻¹ (DESIGN §1.3; FOLDAPI §0), each term at its exact phase and the whole of it
//! taken at a jump's time (model/fold/run.rs): two blocks' landings on the states s, s′ of one landing set interfere at
//! their exact residual, a static one too -- tilt0's Larmor-compensated σ⁺σ⁻, whose two beams' blocks land on m − 1 and
//! m + 1 with one photon frequency: the ground coherence the rotating polarization pumps, its friction (BRIEF_RAMAN). The
//! decay between jumps is Σ_q L_q†L_q: each block's own, L_o†L_o, static in G; and between two blocks their cross
//! products L_o†L_o′ at e^{i(o − o′)t}, terms of the generator that turn as the Hamiltonian's tones do, under the one
//! rule (FOLDAPI §2; `Fold::beating`): the term's size a over its beat |o − o′|, read as `Choice::of` reads a coupling
//! (`unfoldable`) -- within the share it is averaged out (its mean, nothing, kept; its reading the rest), beyond it a tone
//! of the integrator (the run's candidates), so that where every cross term is a tone the decay is the jumps' product at
//! every time by construction, positive semidefinite (BRIEF_WIRE3 item 3). GMD1C's cross decay of its F = 1 and F = 2
//! blocks turns at the hyperfine 42943 rad/µs and is averaged; tilt0's of its two beams' blocks, at 26 rad/µs, is a tone.
//! (A kept copy's related blocks keep their cross term whole: not the run's, whose choice keeps no copy in P.)
//!
//! # The passes as the next one reads (DESIGN §1.5, M1)
//! Two passes, a third only as the share asks (the cheapest within the share, the evaluation order's rule): after the
//! second the next one's change is read by the same block perturbation, bounded -- a column's response moves by at most
//! Σ_i |δH_ij| ‖Z_i‖ over its distance from the block's diagonal, δH the Bloch generator less the H solved at --, and the
//! third is solved only beyond the share. Two at the least: the existence (the passes' ratio) is read from two changes; one
//! pass reads nothing where the self-consistency does not exist (the Λ at its two-photon resonance, ⑥: one pass's next
//! reads zero there, its ratio after three is above one).
//!
//! # Two passes and the read of the next (DESIGN §1.5, M1)
//! Pass 0 at H = diag(Ĥ_PP), then at H = diag of the Bloch generator it gives (the diagonal fold), pass 1 at the Bloch
//! generator H⁽¹⁾ = Ĥ_PP + Σ_o Ĥ_Po T_o⁽⁰⁾; the read: H⁽²⁾ from T⁽¹⁾, δT solving the same equation with source −T⁽¹⁾ δH, and the
//! ratio ‖H⁽²⁾ − H⁽¹⁾‖/‖H⁽¹⁾ − H⁽⁰⁾‖ (at or above one the self-consistency does not exist).
//!
//! # The force (DESIGN §1.3)
//! −∇ Re tr(Ḡ†G) by the adjoint: the cotangents pass back through the compression, the centred rows, each pass's Sylvester
//! equation (its adjoint, the same Schur forms solved backwards, `sylvester_adjoint`) and the Bloch generators to the
//! lattice's matrices, and from them to H_GG, 𝒦 and every beam's amplitude, contracted with their analytic gradients.
// Not yet the run's (step 2 wires it): what only the tests reach is not dead.
#![cfg_attr(not(test), allow(dead_code))]

use super::far::{decay, outside_form, Axes, Bases, EXCITED};
use super::{Beat, Component as Part};
use crate::field::{Amplitude, Light};
use crate::linalg::{hermitian_eigen, schur, Mat, C64};
use std::rc::Rc;
use crate::request::{Manifold, Request, GROUND, STATES};

mod step;
mod proto;
mod decide;
pub(crate) mod run;
#[cfg(test)]
mod tests;

/// The lab axes.
const AXES: usize = 3;

fn mul(a: &Mat, b: &Mat, w: &mut f64) -> Mat {
    *w += (a.rows * a.cols * b.cols) as f64;
    a.product(b)
}
/// a† b.
fn mul_adj(a: &Mat, b: &Mat, w: &mut f64) -> Mat {
    *w += (a.rows * a.cols * b.cols) as f64;
    a.adjoint_product(b)
}
/// a b†.
fn mul_by_adj(a: &Mat, b: &Mat, w: &mut f64) -> Mat {
    *w += (a.rows * a.cols * b.rows) as f64;
    a.product(&b.adjoint())
}
/// a b, of a's entries that are not zero alone (a structurally sparse a: the lattice's frame matrix, emission, sources).
fn mul_sparse(a: &Mat, b: &Mat, w: &mut f64) -> Mat {
    let mut out = Mat::zeros(a.rows, b.cols);
    for i in 0..a.rows {
        for k in 0..a.cols {
            let x = a[(i, k)];
            if x == C64::ZERO {
                continue;
            }
            for j in 0..b.cols {
                out[(i, j)] += x * b[(k, j)];
            }
            *w += b.cols as f64;
        }
    }
    out
}
fn plus(a: &mut Mat, b: &Mat) {
    a.add_scaled(C64::ONE, b);
}
fn rate(rho: f64) -> f64 {
    2.0 * rho + rho * rho
}

/// One component of the light (module documentation): its part, exact frequency and value, and per beam (request
/// indices) its coupling at the focus (EXCITED × GROUND) and the momentum its absorption gives.
pub(crate) struct Component {
    pub(crate) part: Part,
    pub(crate) beat: Beat,
    pub(crate) nu: f64,
    beams: Vec<usize>,
    couplings: Vec<Mat>,
    kicks: Vec<[f64; 3]>,
    /// Its couplings on its field's basis U_k (EXCITED × GROUND, far.rs `Bases`) and per beam its coefficients on them
    /// (γ_lk, conjugated counter-rotating): X_l = Σ_k gamma[l][k] U_k.
    basis: Vec<Mat>,
    gamma: Vec<Vec<C64>>,
}

impl Component {
    /// A beam's factor in it: a_l, or a_l* counter-rotating.
    fn factor(&self, a: C64) -> C64 {
        if self.part.counter {
            a.conj()
        } else {
            a
        }
    }
}

/// How a copy is taken (module documentation, # Blocks).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Status {
    Decomposed,
    Centred,
    Omitted,
    Kept,
}

/// A row of a block: an excited state, or the copy of ground state `state` by the emission of component `component`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Row {
    Excited(usize),
    Copy { state: usize, component: usize },
}

/// The run's choice of nodes (DESIGN §1.1, §3, §11): the ground states' frames and sets (the rule's, `Choice::of`), the
/// explicit excited levels (a level at the offset component `part` reaches from ground state `ground`: an excited node kept
/// in P), and every other Q node's tier (the same tiers for every node, module documentation # Blocks), keyed by an
/// absorbed component and one of the block's own sources (the block): a copy's by the emitting component and its ground
/// level F (unlisted: centred), an excited node's by its level (`excited`; unlisted: solved) -- a later entry overriding an
/// earlier one. `driven`: the nodes of an explicit level at another offset (its key's) that cannot be folded there by the
/// readings (decide.rs, the start's decision): the drive there is a tone of Ĥ_PP (`Fold::driven`), as where the coupling
/// cannot be folded at its peak.
#[derive(Clone, Debug, Default)]
pub(crate) struct Choice {
    pub(crate) frames: Vec<Beat>,
    pub(crate) sets: Vec<Vec<usize>>,
    pub(crate) explicit: Vec<(usize, Part, usize)>,
    pub(crate) driven: Vec<(usize, Part, usize)>,
    pub(crate) copies: Vec<((Part, usize, Part, u8), Status)>,
    pub(crate) excited: Vec<((Part, usize, usize), Status)>,
}

/// Whether a coupling of mixing m = |a|/|Δ| (its strength over its mismatch, both frame-free) cannot be folded (DESIGN
/// §1.1): its copy's population w(m) beyond the share in rate. The one reading of a coupling: the sets' (`Choice::of`)
/// and the tones' (`Fold::turning`).
fn unfoldable(m: f64, share: f64) -> bool {
    rate(admixture(m).sqrt()) > share
}

/// A copy's population at mixing m, the dressed two-state admixture ½(1 − 1/√(1 + 4m²)) (½ where m is infinite: a
/// coupling at no mismatch).
fn admixture(m: f64) -> f64 {
    if m.is_finite() {
        0.5 * (1.0 - 1.0 / (1.0 + 4.0 * m * m).sqrt())
    } else {
        0.5
    }
}

/// The parts of `n` states joined by the edges (a, b) (union by the least), each in increasing order.
fn connected(n: usize, edges: &[(usize, usize)]) -> Vec<Vec<usize>> {
    let mut part: Vec<usize> = (0..n).collect();
    fn find(part: &mut [usize], mut x: usize) -> usize {
        while part[x] != x {
            part[x] = part[part[x]];
            x = part[x];
        }
        x
    }
    for &(a, b) in edges {
        let (a, b) = (find(&mut part, a), find(&mut part, b));
        part[a.max(b)] = a.min(b);
    }
    let mut out: Vec<Vec<usize>> = vec![];
    let mut roots: Vec<usize> = vec![];
    for x in 0..n {
        let r = find(&mut part, x);
        match roots.iter().position(|&y| y == r) {
            Some(at) => out[at].push(x),
            None => {
                roots.push(r);
                out.push(vec![x]);
            }
        }
    }
    out
}

impl Choice {
    /// The rule's structure of a request (module documentation, # P, sets and frames) at its peak light (every beam at its
    /// focus value, in phase), every node folded and every copy centred, with the share `share`: the ground states' couplings
    /// -- through an excited state, absorbing f and emitting g, a = Σ_x |V_g[x,t]| |V_f[x,s]| / |E_s + ν_f − 𝒦_xx|, at the
    /// mismatch Δ = E_t − E_s − (ν_f − ν_g); and the outside forms', Σ_ω Σ_i |O_ω,i[t,s]| N_ω at Δ = E_t − E_s -- of an
    /// admixture beyond the share join their ends into a set, and the mixing's maximum spanning tree of those is static,
    /// each set then moved by the exact frequency (zero or a beat ν_f − ν_g) nearest its diagonal's mean.
    pub(crate) fn of(request: &Request, share: f64) -> Result<Choice, String> {
        let light = Light::new(request);
        let axes = Axes::new(request)?;
        let components = components(request, &light, &axes)?;
        let e: Vec<f64> = (0..STATES).map(|s| request.hamiltonian[(s, s)].re).collect();
        let d = decay(request);
        let peak: Vec<Mat> = components
            .iter()
            .map(|c| Mat::from_fn(EXCITED, GROUND, |x, g| C64::real(c.couplings.iter().map(|m| m[(x, g)].abs()).sum())))
            .collect();
        let forms = |a: usize, b: usize| -> f64 {
            light.frequencies.iter().map(|f| f.norm * request.outside_forms[f.form].matrices.iter().map(|m| m[(a, b)].abs()).sum::<f64>()).sum()
        };
        let mut edges: Vec<(usize, usize, Beat, f64)> = vec![];
        for (f, cf) in components.iter().enumerate() {
            for (g, cg) in components.iter().enumerate() {
                for s in 0..GROUND {
                    for t in (0..GROUND).filter(|&t| t != s) {
                        let a: f64 = (0..EXCITED).map(|x| peak[g][(x, t)].re * peak[f][(x, s)].re / C64::new(e[s] + cf.nu - e[GROUND + x], 0.5 * d[(x, x)].re).abs()).sum();
                        if a > 0.0 {
                            let delta = e[t] - e[s] - (cf.nu - cg.nu);
                            edges.push((t, s, cf.beat.plus(-1, &cg.beat), a / delta.abs()));
                        }
                    }
                }
            }
        }
        for s in 0..GROUND {
            for t in (0..GROUND).filter(|&t| t != s) {
                let a = forms(t, s) + request.hamiltonian[(t, s)].abs();
                if a > 0.0 {
                    edges.push((t, s, axes.zero(), a / (e[t] - e[s]).abs()));
                }
            }
        }
        let kept: Vec<(usize, usize, Beat, f64)> = edges.into_iter().filter(|x| unfoldable(x.3, share)).collect();
        let sets = connected(GROUND, &kept.iter().map(|x| (x.0, x.1)).collect::<Vec<_>>());
        let mut frames = super::cluster::frames(GROUND, &kept, &axes.zero());
        let mut candidates = vec![axes.zero()];
        for cf in &components {
            for cg in &components {
                candidates.push(cf.beat.plus(-1, &cg.beat));
            }
        }
        for set in &sets {
            let mean = set.iter().map(|&s| e[s] - axes.omega(&frames[s])).sum::<f64>() / set.len() as f64;
            let shift = candidates.iter().min_by(|a, b| (axes.omega(a) - mean).abs().total_cmp(&(axes.omega(b) - mean).abs())).unwrap().clone();
            for &s in set {
                frames[s] = frames[s].plus(1, &shift);
            }
        }
        Ok(Choice { frames, sets, explicit: vec![], driven: vec![], copies: vec![], excited: vec![] })
    }

    /// The rule's tiers at the focus with the explicit levels `explicit` kept in P (M18, the evaluation order's rule): the
    /// rule's sets, frames and parts (`Choice::of`); every copy decomposed (exact: a centring can be singular, a copy at the
    /// centre itself), then each at the cheapest evaluation its readings allow within the share: of a block's copies, and of
    /// its excited levels, those whose omission readings sum within the share left out (the smallest first), of the others
    /// those whose centring reads within it centred, the rest decomposed or solved -- to the fixed point (at most four
    /// rounds a set). The readings are in both units, the larger (`Fold::readings`): the column's whole emission and its
    /// emission out of the set (the populations').
    pub(crate) fn tiered(request: &Request, explicit: Vec<(usize, Part, usize)>) -> Result<Choice, String> {
        Choice::tiered_of(request, explicit, vec![], &|_| true)
    }

    /// `tiered` on the sets `of` picks alone (the others' statuses left unread: no key of theirs is another set's, so a set's
    /// fold is the same), the explicit levels' drives `driven` tones.
    fn tiered_of(request: &Request, explicit: Vec<(usize, Part, usize)>, driven: Vec<(usize, Part, usize)>, of: &dyn Fn(&[usize]) -> bool) -> Result<Choice, String> {
        let share = crate::model::elim::share();
        let parts: Vec<Part> = components(request, &Light::new(request), &Axes::new(request)?)?.iter().map(|c| c.part).collect();
        let mut copies = vec![];
        for &a in &parts {
            for s in 0..GROUND {
                for &e in &parts {
                    for f in [1, 2] {
                        copies.push(((a, s, e, f), Status::Decomposed));
                    }
                }
            }
        }
        let mut choice = Choice { copies, explicit, driven, ..Choice::of(request, share)? };
        for set in Fold::sets(request, &choice)?.into_iter().filter(|set| of(set)) {
            for _ in 0..4 {
                let fold = Fold::new(request, &choice, set[0])?;
                if fold.blocks.is_empty() {
                    break;
                }
                let f = fold.read_at([0.0; 3]).ok_or("焦点处的折叠不存在")?;
                let mut changed = false;
                // A block's copies left out together: their omission readings summed, the smallest first, as its excited
                // levels' (each alone within the share, tilt0's three tweezer copies left out together were not:
                // dragchk.status.md section 4).
                for (o, b) in fold.blocks.iter().enumerate() {
                    let Some(&(s0, a0)) = b.sources.first() else { continue };
                    let mut nodes: Vec<(u8, usize, f64, f64)> = f.readings.iter().filter(|r| r.block == o).filter_map(|r| if let Node::Copy { level, component } = r.node { Some((level, component, r.omission, r.centring)) } else { None }).collect();
                    nodes.sort_by(|a, b| a.2.total_cmp(&b.2));
                    let mut sum = 0.0;
                    for (level, component, omission, centring) in nodes {
                        let next = if rate((sum + omission).sqrt()) <= share {
                            sum += omission;
                            Status::Omitted
                        } else if rate(centring.sqrt()) <= share {
                            Status::Centred
                        } else {
                            Status::Decomposed
                        };
                        let key = (fold.components[a0].part, fold.ground[s0], fold.components[component].part, level);
                        let status = choice.copies.iter().rev().find(|x| x.0 == key).map_or(Status::Centred, |x| x.1);
                        if next != status {
                            for &(s, a) in &b.sources {
                                let key = (fold.components[a].part, fold.ground[s], fold.components[component].part, level);
                                choice.copies.retain(|x| x.0 != key);
                                choice.copies.push((key, next));
                            }
                            changed = true;
                        }
                    }
                }
                for (o, b) in fold.blocks.iter().enumerate() {
                    let mut nodes: Vec<(usize, f64, f64)> = f.readings.iter().filter(|r| r.block == o).filter_map(|r| if let Node::Excited(level) = r.node { Some((level, r.omission, r.centring)) } else { None }).collect();
                    nodes.sort_by(|a, b| a.1.total_cmp(&b.1));
                    let mut sum = 0.0;
                    for (level, omission, centring) in nodes {
                        let next = if rate((sum + omission).sqrt()) <= share {
                            sum += omission;
                            Status::Omitted
                        } else if rate(centring.sqrt()) <= share {
                            Status::Centred
                        } else {
                            Status::Decomposed
                        };
                        let Some(&(s0, a0)) = b.sources.first() else { continue };
                        let now = choice.excited.iter().rev().find(|x| x.0 == (fold.components[a0].part, fold.ground[s0], level)).map_or(Status::Decomposed, |x| x.1);
                        if next != now {
                            for &(s, a) in &b.sources {
                                let key = (fold.components[a].part, fold.ground[s], level);
                                choice.excited.retain(|x| x.0 != key);
                                choice.excited.push((key, next));
                            }
                            changed = true;
                        }
                    }
                }
                if !changed {
                    break;
                }
            }
        }
        Ok(choice)
    }

    /// The run's choice of nodes with the explicit levels `explicit` kept in P (DESIGN §3.2, §11; the prototype's, proto.rs):
    /// the rule's tiers at the focus (`tiered`), and in every block that is not far its centred excited rows solved too (a
    /// mode of the kept decomposition, not a pass); a block is far -- its rows all centred, kept so -- where it solves no
    /// row and holds nothing but its excited rows (`Block::far`: an explicit state or a copy is no far form's, proto.rs);
    /// every excited level of a block with rows, and of a block with none at the focus, read at every point (proto.rs # The
    /// levels kept and left out). Which excited nodes are explicit is the run's at every point, not the focus's (run.rs #
    /// The nodes risen and fallen).
    pub(crate) fn base(request: &Request, explicit: Vec<(usize, Part, usize)>) -> Result<Choice, String> {
        Choice::base_of(request, explicit, vec![], &|_| true)
    }

    /// `base` on the sets `of` picks alone (`tiered_of`): a cluster's choice reads its own ground set's tiers, not every
    /// set's (model/fold/run.rs `Clusters::cluster_with`); the explicit levels' drives `driven` tones (`Choice::driven`).
    pub(crate) fn base_of(request: &Request, explicit: Vec<(usize, Part, usize)>, driven: Vec<(usize, Part, usize)>, of: &dyn Fn(&[usize]) -> bool) -> Result<Choice, String> {
        let mut choice = Choice::tiered_of(request, explicit, driven, of)?;
        for set in Fold::sets(request, &choice)?.into_iter().filter(|set| of(set)) {
            let fold = Fold::new(request, &choice, set[0])?;
            let mut add = vec![];
            for b in fold.blocks.iter().filter(|b| !b.far()) {
                for r in &b.rows[b.solved..] {
                    if let Row::Excited(x) = *r {
                        for &(p, c) in &b.sources {
                            add.push(((fold.components[c].part, fold.ground[p], fold.excited_level[x]), Status::Decomposed));
                        }
                    }
                }
            }
            choice.excited.extend(add);
            for b in fold.blocks.iter().filter(|b| !b.rows.is_empty()) {
                let status = if b.far() { Status::Centred } else { Status::Decomposed };
                for &(p, c) in &b.sources {
                    for x in choice.excited.iter_mut().filter(|x| x.1 == Status::Omitted && x.0 .0 == fold.components[c].part && x.0 .1 == fold.ground[p]) {
                        x.1 = status;
                    }
                }
            }
            for x in choice.excited.iter_mut().filter(|x| x.1 == Status::Omitted && set.contains(&x.0 .1)) {
                x.1 = Status::Decomposed;
            }
        }
        Ok(choice)
    }

    /// The excited nodes of a fold that may be kept in P (DESIGN §11.1): per block with sources, each excited level among
    /// its Q rows -- its key (level, the absorbed component's part, its first source's ground state; `Choice::explicit`)
    /// with the block and its rows.
    pub(crate) fn candidates(fold: &Fold) -> Vec<((usize, Part, usize), usize, Vec<usize>)> {
        let mut out: Vec<((usize, Part, usize), usize, Vec<usize>)> = vec![];
        for (o, b) in fold.blocks.iter().enumerate() {
            let Some(&(p, c)) = b.sources.first() else { continue };
            for (r, row) in b.rows.iter().enumerate() {
                let Row::Excited(x) = *row else { continue };
                let key = (fold.excited_level[x], fold.components[c].part, fold.ground[p]);
                match out.iter_mut().find(|n| n.0 == key) {
                    Some(n) => n.2.push(r),
                    None => out.push((key, o, vec![r])),
                }
            }
        }
        out
    }
}

/// A block (module documentation, # Blocks): its offset, its explicit states' P places, its Q rows (the solved ones first:
/// excited states, then decomposed copies; then the centred ones, excited states and copies), its ground sources (place,
/// component), its kept copies (state, component, exact residual θ_t − ω) and its omitted ones.
pub(crate) struct Block {
    pub(crate) offset: Beat,
    pub(crate) value: f64,
    pub(crate) explicit: Vec<usize>,
    pub(crate) rows: Vec<Row>,
    solved: usize,
    sources: Vec<(usize, usize)>,
    pub(crate) kept: Vec<(usize, usize, Beat)>,
    pub(crate) omitted: Vec<(usize, usize)>,
}

impl Block {
    /// Whether the block is far (proto.rs, a block whose rows are all centred: its response R₀B, pair forms of the field):
    /// it solves no row, and holds no explicit state and no copy row -- what a far form does not hold.
    fn far(&self) -> bool {
        self.solved == 0 && self.explicit.is_empty() && self.rows.iter().all(|r| matches!(r, Row::Excited(_)))
    }

    /// The excited rows of its amplitude ε_o: its explicit states, then its excited Q rows (excited state indices).
    fn excited(&self, fold: &Fold) -> Vec<usize> {
        let mut out: Vec<usize> = self.explicit.iter().map(|&p| fold.explicit[p - fold.ground.len()]).collect();
        out.extend(self.rows.iter().filter_map(|r| if let Row::Excited(e) = r { Some(*e) } else { None }));
        out
    }
}

/// The fold of a request with a choice of nodes (module documentation).
pub(crate) struct Fold {
    light: Light,
    /// The local light (run.rs `Onefold::decide`: the start's worst case over the atoms' local physics): per request beam a
    /// factor on its amplitude everywhere -- the trap light's √s, each beam's phase --; empty, none (every factor one).
    pub(crate) local: Vec<C64>,
    /// The explicit states' copies at other offsets whose drive is a tone (`Fold::build`): (excited state, the block's
    /// offset). The others are their blocks' Q rows.
    pub(crate) driven: Vec<(usize, Beat)>,
    hbar_over_mass: f64,
    /// k_BT of the request's temperature (rad/µs): the potential's unit in the omission's reading.
    thermal: f64,
    pub(crate) components: Vec<Component>,
    /// P's ground states (place p < ng holds ground state ground[p]).
    pub(crate) ground: Vec<usize>,
    /// Per P place its frame, exact and its value (rad/µs).
    pub(crate) frames: Vec<Beat>,
    frame_values: Vec<f64>,
    /// Per explicit place j (P place ng + j) its excited state.
    pub(crate) explicit: Vec<usize>,
    /// Per entry (p, q) of P (n × n, row by row) whether it is a tone (module documentation, # Tones): two places of one
    /// kind and different frames whose coupling cannot be folded.
    pub(crate) turning: Vec<bool>,
    /// Per two ground states (GROUND × GROUND) whether their static coupling cannot be folded (`turning`'s reading): a
    /// copy at a P place's frame whose coupling to that place can be folded is a copy row, that coupling left out as P
    /// leaves it out (proto.rs `prototype_workspace`; BRIEF_COPIES item 4).
    pub(crate) ground_tone: Vec<bool>,
    pub(crate) blocks: Vec<Block>,
    /// Per two blocks (blocks × blocks, row by row, symmetric) the reading of their cross decay L_o†L_o′ (module
    /// documentation, # Tones; `Fold::crossing`): a tone where it is beyond the share (`Fold::beating`), else averaged out.
    pub(crate) crossing: Vec<f64>,
    /// H0 less what it holds of the outside forms at the focus (request.rs `held`): its ground block, and its excited block with −(i/2)D.
    base_ground: Mat,
    base_excited: Mat,
    /// The emission operators (GROUND × EXCITED).
    emission: [Mat; 3],
    /// The ground sets a photon lands in (`Fold::sets`): the readings' populations unit reads each transfer into another
    /// set against itself.
    landing: Vec<Vec<usize>>,
    /// Per frequency and (p, q) of its outside form O_pq in the spherical components (STATES × STATES) the entries that are
    /// not zero.
    entries: Vec<Vec<(usize, usize, Vec<(usize, usize, C64)>)>>,
    /// The same of the excited states no block solves and no place holds (the readings' bare excited block alone).
    others: Vec<Vec<(usize, usize, Vec<(usize, usize, C64)>)>>,
    /// The levels: per ground state its F, per excited state its level, the excited levels (manifold, F′).
    pub(crate) ground_level: [u8; GROUND],
    pub(crate) excited_level: Vec<usize>,
    pub(crate) levels: Vec<(Manifold, u8)>,
}

/// The components of a request's light that couple anything, in the order of the frequencies, co-rotating first.
fn components(request: &Request, light: &Light, axes: &Axes) -> Result<Vec<Component>, String> {
    let mut out = vec![];
    let ground: Vec<usize> = (0..GROUND).collect();
    let excited: Vec<usize> = (GROUND..STATES).collect();
    for (frequency, f) in light.frequencies.iter().enumerate() {
        let bases = Bases::new(request, light, frequency)?;
        for counter in [false, true] {
            let part = Part { frequency, counter };
            let beat = axes.component(light, part);
            let couplings: Vec<Mat> = f.beams.iter().map(|&l| (if counter { &request.beams[l].counter_coupling } else { &request.beams[l].coupling }).block(&excited, &ground)).collect();
            if couplings.iter().all(|m| m.data.iter().all(|z| *z == C64::ZERO)) {
                continue;
            }
            let kicks = f.beams.iter().map(|&l| light.beams[l].wave_vector().map(|k| if counter { -k } else { k })).collect();
            let basis = (if counter { &bases.counter } else { &bases.co }).iter().map(|u| u.block(&excited, &ground)).collect();
            let gamma = f.coefficients.iter().map(|g| g.iter().map(|z| if counter { z.conj() } else { *z }).collect()).collect();
            out.push(Component { part, nu: axes.omega(&beat), beat, beams: f.beams.clone(), couplings, kicks, basis, gamma });
        }
    }
    Ok(out)
}

impl Fold {
    /// The state sets of a choice of nodes (DESIGN §3.2): its sets, joined further by its kept copies (a kept copy's level
    /// and the block's source state carry a tone between them) and by its explicit levels (an explicit level and its source).
    pub(crate) fn sets(request: &Request, choice: &Choice) -> Result<Vec<Vec<usize>>, String> {
        let labels = request.levels()?;
        let mut joins = vec![];
        for set in &choice.sets {
            for w in set.windows(2) {
                joins.push((w[0], w[1]));
            }
        }
        for ((_, s, _, f), status) in &choice.copies {
            if *status == Status::Kept {
                joins.extend((0..GROUND).filter(|&t| labels[t].f == *f).map(|t| (*s, t)));
            }
        }
        Ok(connected(GROUND, &joins))
    }

    /// The fold of the state set holding ground state `state` (`Fold::sets`) of a request with a choice of nodes.
    pub(crate) fn new(request: &Request, choice: &Choice, state: usize) -> Result<Fold, String> {
        let sets = Fold::sets(request, choice)?;
        let set = sets.iter().find(|s| s.contains(&state)).ok_or("没有这个基态")?;
        Fold::build(request, choice, set)
    }

    /// The fold of a request with a choice of nodes on the ground states `ground`: its components, P with its frames, and the
    /// blocks of every offset the sources reach with their rows (module documentation).
    fn build(request: &Request, choice: &Choice, ground: &[usize]) -> Result<Fold, String> {
        let light = Light::new(request);
        let axes = Axes::new(request)?;
        let components = components(request, &light, &axes)?;
        let labels = request.levels()?;
        let mut levels: Vec<(Manifold, u8)> = vec![];
        let excited_level: Vec<usize> = (GROUND..STATES)
            .map(|s| {
                let key = (labels[s].manifold, labels[s].f);
                levels.iter().position(|l| *l == key).unwrap_or_else(|| {
                    levels.push(key);
                    levels.len() - 1
                })
            })
            .collect();
        let ground_level: [u8; GROUND] = std::array::from_fn(|s| labels[s].f);
        let find = |p: Part| components.iter().position(|c| c.part == p).ok_or_else(|| format!("没有这个光的分量：{p:?}"));
        // P: the ground states, then the explicit levels' states at their offsets.
        if choice.frames.len() != GROUND {
            return Err("每个基态需要一个帧".into());
        }
        let state_frames = choice.frames.clone();
        let ng = ground.len();
        let mut frames: Vec<Beat> = ground.iter().map(|&s| state_frames[s].clone()).collect();
        // H0 less what it holds of the outside forms at the focus (the field there Σ_l E_l; request.rs `held`).
        let outside: Vec<[[Mat; 3]; 3]> = light.frequencies.iter().map(|f| outside_form(&request.outside_forms[f.form].matrices)).collect();
        let mut base = request.hamiltonian.clone();
        for (w, f) in light.frequencies.iter().enumerate() {
            let e0: [C64; 3] = std::array::from_fn(|q| f.fields.iter().map(|e| e[q]).sum());
            let mut at_focus = Mat::zeros(STATES, STATES);
            for p in 0..3 {
                for q in 0..3 {
                    at_focus.add_scaled(e0[p].conj() * e0[q], &outside[w][p][q]);
                }
            }
            base.add_scaled(-C64::ONE, &crate::request::held(&at_focus));
        }
        let all: Vec<usize> = (0..GROUND).collect();
        let excited: Vec<usize> = (GROUND..STATES).collect();
        let d = decay(request);
        let mut base_excited = base.block(&excited, &excited);
        base_excited.add_scaled(C64::new(0.0, -0.5), &d);
        // The excited states the outside forms couple: each frequency's form on the field components its beams have (a
        // beam's polarization is the same everywhere, field.rs: paraxial), E_p* E_q for p, q among them -- not a component
        // no beam of it has (tilt0's π tweezer couples no m′ to m′ ± 2: such a row would be a copy at an offset that drives it
        // only through a zero, near resonant with an explicit state's column there).
        let mut coupled = vec![false; EXCITED * EXCITED];
        for (o, f) in outside.iter().zip(&light.frequencies) {
            let has: Vec<bool> = (0..3).map(|p| f.fields.iter().any(|e| e[p] != C64::ZERO)).collect();
            for (p, row) in o.iter().enumerate().filter(|(p, _)| has[*p]) {
                for (_, m) in row.iter().enumerate().filter(|(q, _)| has[*q]) {
                    for x in 0..EXCITED {
                        for y in 0..EXCITED {
                            if m[(GROUND + x, GROUND + y)] != C64::ZERO {
                                coupled[x * EXCITED + y] = true;
                                coupled[y * EXCITED + x] = true;
                            }
                        }
                    }
                }
            }
        }
        // The excited states an offset drives: a component from a ground state of the set reaches them there.
        let driven_at = |offset: &Beat| -> Vec<bool> {
            let mut out = vec![false; EXCITED];
            for &s in ground {
                for c in components.iter().filter(|c| state_frames[s].plus(1, &c.beat) == *offset) {
                    for m in &c.couplings {
                        (0..EXCITED).filter(|&x| m[(x, s)] != C64::ZERO).for_each(|x| out[x] = true);
                    }
                }
            }
            out
        };
        // P's explicit states (module documentation, # P, sets and frames): a node is a level at an offset, and holds the
        // level's states that offset drives (`driven_at`), each at the node's offset -- not the level's other states, which
        // other offsets drive (tilt0's F' = 3 m′ at the PGC's offset from m′ + 1, a Zeeman beat apart one from the next:
        // the level whole at one offset made its drives from the other ground states strong tones of a near-resonant
        // transition and its states Q rows of their own blocks besides). One excited state is one place at one frame:
        // a node whose offset would hold a state explicit at another is refused (the start's decision takes its drive
        // there as a tone, `Choice::driven`); a node another key names again adds nothing.
        let mut explicit = vec![];
        for &(level, part, source) in choice.explicit.iter().filter(|x| ground.contains(&x.2)) {
            let offset = state_frames[source].plus(1, &components[find(part)?].beat);
            let held = driven_at(&offset);
            for x in (0..EXCITED).filter(|&x| excited_level[x] == level && held[x]) {
                match explicit.iter().position(|&y| y == x) {
                    Some(j) if frames[ng + j] == offset => {}
                    Some(_) => return Err("一个激发态只能在一个偏移上显式".into()),
                    None => {
                        explicit.push(x);
                        frames.push(offset.clone());
                    }
                }
            }
        }
        let frame_values: Vec<f64> = frames.iter().map(|b| axes.omega(b)).collect();
        // The tones (module documentation, # Tones): an entry of two places of one kind and different frames whose coupling
        // cannot be folded -- its peak strength (H0's, the decay's, and each frequency's outside forms at its intensity
        // scale, as `Choice::of` reads a coupling) over the states' energy difference, frame-free --; one that can is a copy
        // left out (its reading within the share).
        let n = ground.len() + explicit.len();
        let state = |p: usize| if p < ng { ground[p] } else { GROUND + explicit[p - ng] };
        let cannot_fold = |s: usize, t: usize| {
            let decayed = if s < GROUND { 0.0 } else { 0.5 * d[(s - GROUND, t - GROUND)].abs() };
            let peak = request.hamiltonian[(s, t)].abs() + decayed + light.frequencies.iter().map(|f| f.norm * request.outside_forms[f.form].matrices.iter().map(|m| m[(s, t)].abs()).sum::<f64>()).sum::<f64>();
            let delta = request.hamiltonian[(s, s)].re - request.hamiltonian[(t, t)].re;
            unfoldable(peak / delta.abs(), crate::model::elim::share())
        };
        let turning: Vec<bool> = (0..n * n)
            .map(|x| {
                let (p, q) = (x / n, x % n);
                if p == q || (p < ng) != (q < ng) || frames[p] == frames[q] {
                    return false;
                }
                cannot_fold(state(p), state(q))
            })
            .collect();
        // Every two ground states' static coupling read the same way: a copy at a P place's frame is reached from that
        // place by it too (`Fold::lay`'s static split).
        let ground_tone: Vec<bool> = (0..GROUND * GROUND).map(|x| x / GROUND != x % GROUND && cannot_fold(x / GROUND, x % GROUND)).collect();
        // The blocks: every offset a ground state's component reaches, and every explicit state's.
        let mut offsets: Vec<Beat> = vec![];
        let mut sources: Vec<Vec<(usize, usize)>> = vec![];
        let add = |offset: Beat, source: Option<(usize, usize)>, offsets: &mut Vec<Beat>, sources: &mut Vec<Vec<(usize, usize)>>| {
            let at = offsets.iter().position(|o| *o == offset).unwrap_or_else(|| {
                offsets.push(offset);
                sources.push(vec![]);
                offsets.len() - 1
            });
            sources[at].extend(source);
        };
        for (p, &s) in ground.iter().enumerate() {
            for (c, comp) in components.iter().enumerate() {
                if comp.couplings.iter().any(|m| (0..EXCITED).any(|x| m[(x, s)] != C64::ZERO)) {
                    add(frames[p].plus(1, &comp.beat), Some((p, c)), &mut offsets, &mut sources);
                }
            }
        }
        for j in 0..explicit.len() {
            add(frames[ng + j].clone(), None, &mut offsets, &mut sources);
        }
        let mut blocks = vec![];
        let mut driven: Vec<(usize, Beat)> = vec![];
        for (offset, sources) in offsets.into_iter().zip(sources) {
            let value = axes.omega(&offset);
            let mine: Vec<usize> = (0..explicit.len()).filter(|&j| frames[ng + j] == offset).map(|j| ng + j).collect();
            let theirs: Vec<usize> = mine.iter().map(|&p| explicit[p - ng]).collect();
            // The excited Q rows: the excited states a source or an explicit state of the block reaches, closed under 𝒦's
            // couplings (the request's and the outside forms'), each at its level's tier (solved unless chosen otherwise).
            // An explicit state is one object at one frame (FOLDAPI A, coordinator 2026-11-18). Its copy at this block's
            // offset is a Floquet node like any other, under the one rule: where this block's sources drive it beyond
            // the share (their peak strength Σ |V| over the beat between the offset and its frame, `unfoldable`) the
            // drive is an entry of Ĥ_PP that turns (`Fold::driven`; proto.rs # The tones: the drives at another offset)
            // and the block neither holds it nor reaches through it; within the share the copy is a Q row of the block,
            // folded with its light shift, its gradient and its reading (losschk.status.md: the tweezer's far drive of
            // tilt0's explicit F' = 3, a third of the trap's force, was lost as an averaged entry). The readings judge the
            // copy as they judge any node: where the start's decision reads it beyond the share at a point, its node is
            // `Choice::driven` and the drive here is a tone the same way.
            let toned = |x: usize| -> bool {
                let Some(j) = explicit.iter().position(|&y| y == x) else { return false };
                let peak: f64 = sources.iter().map(|&(p, c)| components[c].couplings.iter().map(|m| m[(x, ground[p])].abs()).sum::<f64>()).sum();
                let decided = choice.driven.iter().any(|&(level, part, source)| {
                    level == excited_level[x] && ground.contains(&source) && components.iter().find(|c| c.part == part).is_some_and(|c| state_frames[source].plus(1, &c.beat) == offset)
                });
                decided || (peak > 0.0 && unfoldable(peak / (value - frame_values[ng + j]).abs(), crate::model::elim::share()))
            };
            let foreign = |x: usize| explicit.contains(&x) && !theirs.contains(&x) && toned(x);
            for x in (0..EXCITED).filter(|&x| foreign(x)) {
                driven.push((x, offset.clone()));
            }
            let mut reached = vec![false; EXCITED];
            for &(p, c) in &sources {
                for m in &components[c].couplings {
                    (0..EXCITED).filter(|&x| m[(x, ground[p])] != C64::ZERO && !foreign(x)).for_each(|x| reached[x] = true);
                }
            }
            theirs.iter().for_each(|&x| reached[x] = true);
            loop {
                let before = reached.iter().filter(|&&r| r).count();
                for x in 0..EXCITED {
                    for y in (0..EXCITED).filter(|&y| !foreign(y)) {
                        if reached[x] && !reached[y] && (base_excited[(x, y)] != C64::ZERO || base_excited[(y, x)] != C64::ZERO || coupled[x * EXCITED + y]) {
                            reached[y] = true;
                        }
                    }
                }
                if reached.iter().filter(|&&r| r).count() == before {
                    break;
                }
            }
            let tier = |x: usize| {
                choice
                    .excited
                    .iter()
                    .rev()
                    .find(|((absorbed, s, level), _)| *level == excited_level[x] && find(*absorbed).is_ok_and(|a| sources.iter().any(|&(p, c)| ground[p] == *s && c == a)))
                    .map_or(Status::Decomposed, |x| x.1)
            };
            let q: Vec<usize> = (0..EXCITED).filter(|&x| reached[x] && !theirs.contains(&x) && tier(x) == Status::Decomposed).collect();
            let centred_excited: Vec<usize> = (0..EXCITED).filter(|&x| reached[x] && !theirs.contains(&x) && tier(x) == Status::Centred).collect();
            let rows_excited: Vec<usize> = theirs.iter().chain(&q).chain(&centred_excited).copied().collect();
            // The copies: per emitting component, the ground states it reaches from the block's excited rows off their frames.
            let mut decomposed = vec![];
            let mut centred = vec![];
            let mut kept = vec![];
            let mut omitted = vec![];
            for (g, comp) in components.iter().enumerate() {
                let omega = offset.plus(-1, &comp.beat);
                for t in (0..GROUND).filter(|&t| !ground.contains(&t) || state_frames[t] != omega) {
                    if !comp.couplings.iter().any(|m| rows_excited.iter().any(|&x| m[(x, t)] != C64::ZERO)) {
                        continue;
                    }
                    let status = choice
                        .copies
                        .iter()
                        .rev()
                        .find(|((absorbed, s, emitted, f), _)| *emitted == comp.part && *f == ground_level[t] && find(*absorbed).is_ok_and(|a| sources.iter().any(|&(p, c)| ground[p] == *s && c == a)))
                        .map_or(Status::Centred, |x| x.1);
                    let row = Row::Copy { state: t, component: g };
                    match status {
                        Status::Decomposed => decomposed.push(row),
                        Status::Centred => centred.push(row),
                        Status::Omitted => omitted.push((t, g)),
                        Status::Kept => kept.push((t, g, state_frames[t].plus(-1, &omega))),
                    }
                }
            }
            let mut rows: Vec<Row> = q.iter().map(|&x| Row::Excited(x)).collect();
            rows.extend(decomposed);
            let solved = rows.len();
            rows.extend(centred_excited.iter().map(|&x| Row::Excited(x)));
            rows.extend(centred);
            blocks.push(Block { offset, value, explicit: mine, rows, solved, sources, kept, omitted });
        }
        // The cross decays' readings (module documentation, # Tones).
        let crossing = Fold::crossing(request, &components, &axes, ground, &explicit, &blocks);
        // The outside forms' entries that are not zero, of the states the fold uses (every ground state; the excited states a
        // block solves or a place holds) and of the others.
        let mut used = [true; STATES];
        for x in 0..EXCITED {
            used[GROUND + x] = explicit.contains(&x) || blocks.iter().any(|b| b.rows.contains(&Row::Excited(x)));
        }
        let (mut entries, mut others) = (vec![], vec![]);
        for o in &outside {
            let (mut mine, mut theirs) = (vec![], vec![]);
            for p in 0..3 {
                for q in 0..3 {
                    let m = &o[p][q];
                    let all: Vec<(usize, usize, C64)> = (0..STATES).flat_map(|i| (0..STATES).map(move |j| (i, j))).filter(|&(i, j)| m[(i, j)] != C64::ZERO).map(|(i, j)| (i, j, m[(i, j)])).collect();
                    let (a, b): (Vec<_>, Vec<_>) = all.into_iter().partition(|&(i, j, _)| used[i] && used[j]);
                    if !a.is_empty() {
                        mine.push((p, q, a));
                    }
                    if !b.is_empty() {
                        theirs.push((p, q, b));
                    }
                }
            }
            entries.push(mine);
            others.push(theirs);
        }
        Ok(Fold {
            driven,
            local: vec![],
            hbar_over_mass: crate::units::hbar_over_mass(request.mass_kg),
            thermal: crate::units::KT_PER_UK * request.temperature_uk,
            light,
            components,
            ground: ground.to_vec(),
            frames,
            frame_values,
            explicit,
            turning,
            ground_tone,
            blocks,
            crossing,
            base_ground: base.block(&all, &all),
            base_excited,
            emission: std::array::from_fn(|q| request.emission[q].block(&all, &excited)),
            landing: Fold::sets(request, choice)?,
            entries,
            others,
            ground_level,
            excited_level,
            levels,
        })
    }

    /// Per two blocks the reading of their cross decay (module documentation, # Tones; FOLDAPI §2): the generator's term
    /// −(i/2)e^{i(o − o′)t} Σ_q L_o,q†L_o′,q read as any coupling is (`unfoldable`, `Choice::of`, `turning`) -- its size at
    /// the peak light a = ½ Σ_q Σ_s l_o(q, s) l_o′(q, s) over its beat |o − o′|, exact in the frames' arithmetic. A block's
    /// bare landing amplitude on the ground state s along q at the peak light, as `Choice::of` reads a coupling: l_o(q, s)
    /// = Σ_x |C_q[s, x]| e_o(x), e_o(x) = Σ_(p, c) Σ_l |X_l[x, p]| / |E_p + ν_c − E_x + iΓ_x/2| over its sources (place p,
    /// component c), an explicit state's own row one.
    fn crossing(request: &Request, components: &[Component], axes: &Axes, ground: &[usize], explicit: &[usize], blocks: &[Block]) -> Vec<f64> {
        let d = decay(request);
        let energy = |s: usize| request.hamiltonian[(s, s)].re;
        let ng = ground.len();
        let landing: Vec<[[f64; GROUND]; 3]> = blocks
            .iter()
            .map(|b| {
                let mut e = [0.0; EXCITED];
                for &(p, c) in &b.sources {
                    let (s, comp) = (ground[p], &components[c]);
                    for (x, ex) in e.iter_mut().enumerate() {
                        let v: f64 = comp.couplings.iter().map(|m| m[(x, s)].abs()).sum();
                        if v > 0.0 {
                            *ex += v / C64::new(energy(s) + comp.nu - energy(GROUND + x), 0.5 * d[(x, x)].re).abs();
                        }
                    }
                }
                for &p in &b.explicit {
                    e[explicit[p - ng]] = 1.0;
                }
                std::array::from_fn(|q| std::array::from_fn(|s| (0..EXCITED).map(|x| request.emission[q][(s, GROUND + x)].abs() * e[x]).sum::<f64>()))
            })
            .collect();
        let nb = blocks.len();
        let mut out = vec![0.0; nb * nb];
        for i in 0..nb {
            for j in i + 1..nb {
                let a = 0.5 * (0..3).map(|q| (0..GROUND).map(|s| landing[i][q][s] * landing[j][q][s]).sum::<f64>()).sum::<f64>();
                let m = if a > 0.0 { a / axes.omega(&blocks[i].offset.plus(-1, &blocks[j].offset)).abs() } else { 0.0 };
                out[i * nb + j] = m;
                out[j * nb + i] = m;
            }
        }
        out
    }

    /// Whether the cross decay of blocks i and j is a tone (`Fold::crossing`): its reading beyond the share.
    pub(crate) fn beating(&self, i: usize, j: usize) -> bool {
        let m = self.crossing[i * self.blocks.len() + j];
        m > 0.0 && unfoldable(m, crate::model::elim::share())
    }

    /// n, P's states.
    pub(crate) fn size(&self) -> usize {
        self.ground.len() + self.explicit.len()
    }

    /// The light's fields at a point and their gradients (module documentation, # The Hamiltonian at a point).
    /// The outside forms of the entries `list` at the beams' amplitudes `a`, added to H_GG and 𝒦.
    #[allow(clippy::type_complexity)]
    fn forms(&self, list: &[Vec<(usize, usize, Vec<(usize, usize, C64)>)>], a: &[Amplitude], ground: &mut Mat, excited: &mut Mat, work: &mut f64) {
        for (w, f) in self.light.frequencies.iter().enumerate() {
            let field = f.field(a);
            for (p, q, entries) in &list[w] {
                let x = field.value[*p].conj() * field.value[*q];
                for &(i, j, v) in entries {
                    match (i < GROUND, j < GROUND) {
                        (true, true) => ground[(i, j)] += x * v,
                        (false, false) => excited[(i - GROUND, j - GROUND)] += x * v,
                        _ => {}
                    }
                }
                *work += entries.len() as f64;
            }
        }
    }

    /// Beam l's amplitude at r to an order (field.rs `Geometry::amplitude_to`), times its local factor (`Fold::local`).
    pub(crate) fn amplitude_to(&self, l: usize, r: [f64; 3], order: usize) -> Amplitude {
        let a = self.light.beams[l].amplitude_to(r, order);
        match self.local.get(l) {
            Some(&f) => Amplitude { value: f * a.value, gradient: a.gradient.map(|z| f * z), hessian: a.hessian.map(|row| row.map(|z| f * z)) },
            None => a,
        }
    }

    fn site(&self, r: [f64; 3], kappa: [f64; 3], work: &mut f64) -> Site {
        let a: Vec<Amplitude> = (0..self.light.beams.len()).map(|l| self.amplitude_to(l, r, 1)).collect();
        let mut ground = self.base_ground.clone();
        let mut excited = self.base_excited.clone();
        self.forms(&self.entries, &a, &mut ground, &mut excited, work);
        let drives = self
            .components
            .iter()
            .map(|c| {
                let mut v = Mat::zeros(EXCITED, GROUND);
                for (&l, x) in c.beams.iter().zip(&c.couplings) {
                    v.add_scaled(c.factor(a[l].value), x);
                }
                *work += (c.beams.len() * EXCITED * GROUND) as f64;
                v
            })
            .collect();
        let doppler = self
            .components
            .iter()
            .map(|c| c.kicks.iter().map(|q| self.hbar_over_mass * (q.iter().zip(&kappa).map(|(q, k)| q * k).sum::<f64>() + 0.5 * q.iter().map(|q| q * q).sum::<f64>())).collect())
            .collect();
        Site { a, ground, excited, drives, doppler }
    }

    /// A block's matrices at a point (module documentation, # Blocks): its frame matrix on its Q rows, its sources per split
    /// (the static one first, then one per source beam with its Doppler shift), its emission into P, its part of Ĥ_PP, its
    /// centre z₀.
    fn lay(&self, o: usize, site: &Site, work: &mut f64) -> Laid {
        let b = &self.blocks[o];
        let n = self.size();
        let ng = self.ground.len();
        let m = b.rows.len();
        let theta = &self.frame_values;
        let mut inner = Mat::zeros(m, m);
        for (i, ri) in b.rows.iter().enumerate() {
            for (j, rj) in b.rows.iter().enumerate() {
                inner[(i, j)] = match (*ri, *rj) {
                    (Row::Excited(x), Row::Excited(y)) => site.excited[(x, y)] - if i == j { C64::real(b.value) } else { C64::ZERO },
                    (Row::Excited(x), Row::Copy { state, component }) => site.drives[component][(x, state)],
                    (Row::Copy { state, component }, Row::Excited(y)) => site.drives[component][(y, state)].conj(),
                    (Row::Copy { state: s, component: g }, Row::Copy { state: t, component: h }) if g == h => {
                        site.ground[(s, t)] - if s == t { C64::real(b.value - self.components[g].nu) } else { C64::ZERO }
                    }
                    _ => C64::ZERO,
                };
            }
        }
        // The static split: the E rows' couplings and the ground states at a copy's offset.
        let mut absorb = vec![Mat::zeros(m, n)];
        let mut shifts = vec![0.0];
        let mut beams = vec![None];
        for (i, r) in b.rows.iter().enumerate() {
            for &p in &b.explicit {
                let x = self.explicit[p - ng];
                absorb[0][(i, p)] = match *r {
                    Row::Excited(y) => site.excited[(y, x)],
                    Row::Copy { state, component } => site.drives[component][(x, state)].conj(),
                };
            }
            if let Row::Copy { state, component } = *r {
                for p in (0..ng).filter(|&p| self.frames[p] == self.blocks[o].offset.plus(-1, &self.components[component].beat)) {
                    absorb[0][(i, p)] = site.ground[(state, self.ground[p])];
                }
            }
        }
        // One split a beam, every source column of its component at once (they share its Doppler shift).
        for (c, comp) in self.components.iter().enumerate() {
            let places: Vec<usize> = b.sources.iter().filter(|x| x.1 == c).map(|x| x.0).collect();
            if places.is_empty() {
                continue;
            }
            for (j, (&l, x)) in comp.beams.iter().zip(&comp.couplings).enumerate() {
                let f = comp.factor(site.a[l].value);
                let mut split = Mat::zeros(m, n);
                for &p in &places {
                    for (i, r) in b.rows.iter().enumerate() {
                        if let Row::Excited(y) = *r {
                            split[(i, p)] = f * x[(y, self.ground[p])];
                        }
                    }
                }
                absorb.push(split);
                shifts.push(site.doppler[c][j]);
                beams.push(Some((c, j)));
            }
        }
        // The emission into P: a ground state at its frame (a component's emission from the excited rows, H_GG from a copy at
        // its frame), an explicit state of the block (𝒦 from the excited rows, V_g from a copy).
        let mut emit = Mat::zeros(n, m);
        for (i, r) in b.rows.iter().enumerate() {
            match *r {
                Row::Excited(y) => {
                    for p in 0..ng {
                        for (g, comp) in self.components.iter().enumerate() {
                            if self.frames[p].plus(1, &comp.beat) == b.offset {
                                emit[(p, i)] += site.drives[g][(y, self.ground[p])].conj();
                            }
                        }
                    }
                    for &p in &b.explicit {
                        emit[(p, i)] = site.excited[(self.explicit[p - ng], y)];
                    }
                }
                Row::Copy { state, component } => {
                    let omega = b.offset.plus(-1, &self.components[component].beat);
                    for p in (0..ng).filter(|&p| self.frames[p] == omega) {
                        emit[(p, i)] = site.ground[(self.ground[p], state)];
                    }
                    for &p in &b.explicit {
                        emit[(p, i)] = site.drives[component][(self.explicit[p - ng], state)];
                    }
                }
            }
        }
        // The block's part of Ĥ_PP: its explicit states among themselves (less o) and with the ground states.
        let mut pp = Mat::zeros(n, n);
        for &p in &b.explicit {
            let x = self.explicit[p - ng];
            for &q in &b.explicit {
                pp[(p, q)] = site.excited[(x, self.explicit[q - ng])] - if p == q { C64::real(b.value) } else { C64::ZERO };
            }
            for q in 0..ng {
                for (g, comp) in self.components.iter().enumerate() {
                    if self.frames[q].plus(1, &comp.beat) == b.offset {
                        pp[(p, q)] += site.drives[g][(x, self.ground[q])];
                        pp[(q, p)] += site.drives[g][(x, self.ground[q])].conj();
                    }
                }
            }
        }
        let centre: Vec<f64> = b.sources.iter().map(|&(p, _)| site.ground[(self.ground[p], self.ground[p])].re - theta[p]).chain(b.explicit.iter().map(|&p| pp[(p, p)].re)).collect();
        let z0 = centre.iter().sum::<f64>() / centre.len().max(1) as f64;
        *work += (m * m + m * n * absorb.len()) as f64;
        Laid { inner, absorb, shifts, beams, emit, pp, z0 }
    }

    /// The static P matrix Ĥ_PP (module documentation, # Blocks): the ground states of one frame among themselves (less
    /// their frame), and every block's part.
    fn p_matrix(&self, site: &Site, laid: &[Laid]) -> Mat {
        let n = self.size();
        let mut h = Mat::zeros(n, n);
        for p in 0..self.ground.len() {
            for q in (0..self.ground.len()).filter(|&q| self.frames[q] == self.frames[p]) {
                h[(p, q)] = site.ground[(self.ground[p], self.ground[q])] - if p == q { C64::real(self.frame_values[p]) } else { C64::ZERO };
            }
        }
        for l in laid {
            plus(&mut h, &l.pp);
        }
        h
    }

    /// The fold at a point (r, κ) on a path (module documentation; `Path`, one per fold): no readings (`Fold::read`, at the
    /// analysis times). None where a Schur form does not converge or a centring is singular (a copy at the centre itself:
    /// the rule does not centre it).
    pub(crate) fn at(&self, r: [f64; 3], kappa: [f64; 3], path: &mut Path) -> Option<Folded> {
        let mut work = 0.0;
        let mut phases: Vec<(&'static str, f64)> = vec![];
        let phase = |name: &'static str, work: f64, phases: &mut Vec<(&'static str, f64)>| {
            let before: f64 = phases.iter().map(|x| x.1).sum();
            phases.push((name, work - before));
        };
        let site = self.site(r, kappa, &mut work);
        phase("site", work, &mut phases);
        let n = self.size();
        let laid: Vec<Laid> = (0..self.blocks.len()).map(|o| self.lay(o, &site, &mut work)).collect();
        let hpp = self.p_matrix(&site, &laid);
        phase("lay", work, &mut phases);
        let share = super::elim::share();
        let diag = |h: &Mat| Mat::from_fn(n, n, |i, j| if i == j { h[(i, i)] } else { C64::ZERO });
        let difference = |a: &Mat, b: &Mat| {
            let mut d = a.clone();
            d.add_scaled(-C64::ONE, b);
            d.frobenius()
        };
        // Each block's evaluation order on the path's bases, the passes, and the bases' reading at the energies solved at:
        // beyond the share those bases are made again at the point and the passes solved again.
        let mut renew: Vec<(Vec<Row>, Option<usize>)> = vec![];
        let mut round = 0;
        let (eliminated, stages, h0, h1, h2) = loop {
            let made = path.made;
            let mut eliminated = vec![];
            let mut moved: Vec<Option<(Vec<C64>, Vec<Vec<f64>>)>> = vec![None; path.bases.len()];
            for (o, l) in laid.iter().enumerate() {
                let e = self.effective(o, l, &mut work)?;
                eliminated.push(self.eliminate(o, e, path, &renew, &mut moved, &mut work)?);
            }
            phase("eliminate", work, &mut phases);
            // The passes: at diag(Ĥ_PP), at the diagonal of its Bloch generator, at the Bloch generator; each next one only
            // where the last one's next reads beyond the share.
            let mut stages: Vec<Stage> = vec![];
            let mut hs: Vec<Mat> = vec![diag(&hpp)];
            let bloch = loop {
                let h = hs.last().unwrap().clone();
                let s = self.solve(&site, &laid, &eliminated, &h, stages.len() < 2, &mut work)?;
                let b = self.bloch(&hpp, &laid, &s, &mut work);
                let next = s.next(&eliminated, &h, &b, &mut work);
                stages.push(s);
                if (next <= share && stages.len() == 2) || stages.len() == 3 {
                    break b;
                }
                hs.push(if stages.len() == 1 { diag(&b) } else { b });
            };
            let last = stages.last_mut().unwrap();
            last.read_reuse(&eliminated, &laid, &mut work);
            let mut beyond: Vec<(Vec<Row>, Option<usize>)> = vec![];
            for (e, r) in eliminated.iter().zip(last.reuse()) {
                let x = &path.bases[e.basis];
                if r > share && !beyond.contains(&(x.rows.clone(), x.own)) {
                    beyond.push((x.rows.clone(), x.own));
                }
            }
            phase("passes", work, &mut phases);
            // Beyond the share: the nearest kept decomposition, then a new one; a new one beyond it too is read as it is.
            if beyond.is_empty() || round == 2 || path.made > made {
                let h1 = hs.pop().unwrap();
                let h0 = hs.pop().unwrap_or_else(|| h1.clone());
                break (eliminated, stages, h0, h1, bloch);
            }
            renew.extend(beyond);
            round += 1;
        };
        // The passes' ratio (the self-consistency's existence, ≥ 1 none): of the last two changes.
        let ratio = difference(&h2, &h1) / difference(&h1, &h0);
        let last = stages.last().unwrap();
        // The compression.
        let mut a = hpp.clone();
        let mut metric = Mat::zeros(n, n);
        for (l, t) in laid.iter().zip(&last.responses) {
            plus(&mut a, &mul_sparse(&l.emit, t, &mut work));
            let total = l.absorb.iter().fold(Mat::zeros(t.rows, n), |mut x, y| {
                plus(&mut x, y);
                x
            });
            plus(&mut a, &mul_sparse(&total.adjoint(), t, &mut work).adjoint());
            plus(&mut a, &mul_adj(t, &mul_sparse(&l.inner, t, &mut work), &mut work));
            plus(&mut metric, &mul_adj(t, t, &mut work));
        }
        phase("compression", work, &mut phases);
        let x = hermitian_eigen(&metric);
        work += 10.0 * (n * n * n) as f64;
        let s: Vec<f64> = x.values.iter().map(|&xi| (1.0 + xi).sqrt()).collect();
        let normal = Mat::from_fn(n, n, |i, j| (0..n).map(|k| x.vectors[(i, k)] * x.vectors[(j, k)].conj() / s[k]).sum());
        work += (n * n * n) as f64;
        let generator = mul(&mul(&normal, &a, &mut work), &normal, &mut work);
        // ε per block: its explicit rows, then its excited Q rows.
        let amplitudes: Vec<Mat> = self.blocks.iter().zip(&last.responses).map(|(b, t)| self.amplitude(b, t)).collect();
        let tones = self.tones(&site, &amplitudes, &normal, &mut work);
        let excitation = {
            let mut q = Mat::zeros(n, n);
            for (b, t) in self.blocks.iter().zip(&last.responses) {
                let rows: Vec<usize> = (0..b.rows.len()).filter(|&i| matches!(b.rows[i], Row::Excited(_))).collect();
                let tq = t.block(&rows, &(0..n).collect::<Vec<_>>());
                plus(&mut q, &mul_adj(&tq, &tq, &mut work));
            }
            hermitian_eigen(&q).values.iter().fold(0.0_f64, |m, &v| m.max(v))
        };
        let reuse = last.reuse();
        phase("normal, tones, jumps", work, &mut phases);
        let passes = stages.len();
        Some(Folded { n, a, metric, normal, generator, amplitudes, tones, jumps: vec![], excitation, ratio, readings: vec![], work, reading_work: 0.0, reuse, phases, passes, site, laid, eliminated, stages, hpp, h: [h0, h1, h2], m_eigen: (x.values, x.vectors) })
    }

    /// The fold at a point at rest on a fresh path (its decompositions made there), with its jumps and its readings: what
    /// the choice of nodes reads at the focus (`Choice::tiered`, `Choice::base`).
    pub(crate) fn read_at(&self, r: [f64; 3]) -> Option<Folded> {
        let mut f = self.at(r, [0.0; 3], &mut Path::default())?;
        self.jumps(&mut f);
        self.read(&mut f, 0.0);
        Some(f)
    }

    /// The jumps at a fold (module documentation, # The compression): per block C_q ε_o S⁻¹ -- where a jump is drawn, not
    /// every step (the decay is G's own); their work added to the fold's.
    pub(crate) fn jumps(&self, f: &mut Folded) {
        let mut work = 0.0;
        let es: Vec<usize> = (0..GROUND).collect();
        f.jumps = self
            .blocks
            .iter()
            .zip(&f.amplitudes)
            .map(|(b, e)| {
                let rows = b.excited(self);
                std::array::from_fn(|q| mul(&mul_sparse(&self.emission[q].block(&es, &rows), e, &mut work), &f.normal, &mut work))
            })
            .collect();
        f.work += work;
    }

    /// The readings at a fold (DESIGN §3.1; `Reading`), the light pattern's offset `pattern` (2 k_max |v|, rad/µs) read
    /// beside them: at the analysis times, not every step.
    pub(crate) fn read(&self, f: &mut Folded, pattern: f64) {
        let mut work = 0.0;
        f.readings = self.readings(f, pattern, &mut work);
        f.reading_work = work;
    }

    /// A block's excited amplitude ε_o (module documentation, # The compression): its explicit states' unit rows, then its
    /// excited Q rows of T_o.
    fn amplitude(&self, b: &Block, t: &Mat) -> Mat {
        let n = self.size();
        let excited: Vec<usize> = (0..b.rows.len()).filter(|&i| matches!(b.rows[i], Row::Excited(_))).collect();
        let mut e = Mat::zeros(b.explicit.len() + excited.len(), n);
        for (k, &p) in b.explicit.iter().enumerate() {
            e[(k, p)] = C64::ONE;
        }
        for (k, &i) in excited.iter().enumerate() {
            for j in 0..n {
                e[(b.explicit.len() + k, j)] = t[(i, j)];
            }
        }
        e
    }

    /// One pass (module documentation, # One linear solve per pass): at the static P generator h, every block's response per
    /// split in the Schur coordinates (Z = V†TU) with its source there, in all on its Q rows (its centred rows following),
    /// and per column the reuse's error and the response's size. `diagonal`: h is diagonal (its Schur form itself).
    fn solve(&self, site: &Site, laid: &[Laid], eliminated: &[Eliminated], h: &Mat, diagonal: bool, work: &mut f64) -> Option<Stage> {
        let n = self.size();
        let triangle = if diagonal { Triangle { t: h.clone(), u: Mat::identity(n) } } else { Triangle::of(h, work)? };
        let (mut splits, mut sources, mut responses, mut sizes) = (vec![], vec![], vec![], vec![]);
        for (l, e) in laid.iter().zip(eliminated) {
            // The ranks in H's Schur basis (H diagonal: as they are).
            let turned: Option<Vec<Vec<Mat>>> = (!diagonal).then(|| e.ranks.iter().map(|(_, per)| per.iter().map(|w| mul(w, &triangle.u, work)).collect()).collect());
            let (mut per, mut src) = (vec![], vec![]);
            let mut total = Mat::zeros(e.solved, n);
            for (split, &w) in l.shifts.iter().enumerate() {
                let b = match l.beams[split] {
                    None => match &e.to_static {
                        Some(x) if diagonal => x.clone(),
                        Some(x) => mul(x, &triangle.u, work),
                        None => {
                            per.push(Mat::zeros(e.solved, n));
                            src.push(Mat::zeros(e.solved, n));
                            continue;
                        }
                    },
                    Some((c, j)) => {
                        let comp = &self.components[c];
                        let f = comp.factor(site.a[comp.beams[j]].value);
                        let ci = e.ranks.iter().position(|x| x.0 == c).unwrap();
                        let mut out = Mat::zeros(e.solved, n);
                        for k in 0..comp.basis.len() {
                            let w = match &turned {
                                Some(t) => &t[ci][k],
                                None => &e.ranks[ci].1[k],
                            };
                            out.add_scaled(f * comp.gamma[j][k], w);
                        }
                        *work += (comp.basis.len() * e.solved * n) as f64;
                        if let Some(d) = &e.to_dressed[split] {
                            plus(&mut out, &if diagonal { d.clone() } else { mul(d, &triangle.u, work) });
                        }
                        out
                    }
                };
                let z = e.sylvester(&triangle.t, w, &b, work);
                plus(&mut total, &z);
                per.push(z);
                src.push(b);
            }
            let t = e.from(&total, work);
            let t = if diagonal { t } else { mul_by_adj(&t, &triangle.u, work) };
            responses.push(e.rows(l, &t, work));
            splits.push(per);
            sources.push(src);
            sizes.push((0..n).map(|j| (0..total.rows).map(|k| total[(k, j)].norm_sqr()).sum::<f64>()).collect());
        }
        Some(Stage { triangle, splits, sources, responses, errors: vec![], sizes })
    }

    /// The Bloch generator of a pass's responses: Ĥ_PP + Σ_o Ĥ_Po T_o.
    fn bloch(&self, hpp: &Mat, laid: &[Laid], stage: &Stage, work: &mut f64) -> Mat {
        let mut h = hpp.clone();
        for (l, t) in laid.iter().zip(&stage.responses) {
            plus(&mut h, &mul_sparse(&l.emit, t, work));
        }
        h
    }

    /// The tones (module documentation, # Tones) in the normalized coordinates, each at its exact frequency, ordered by it:
    /// the frame mismatches, the kept copies, the decay's beats.
    /// The ordered pairs of blocks a kept copy relates (module documentation, # Tones): block o and o + (θ_t − ω).
    fn related(&self) -> Vec<(usize, usize)> {
        let mut pairs = vec![];
        for (i, b) in self.blocks.iter().enumerate() {
            for (_, _, r) in &b.kept {
                let target = b.offset.plus(1, r);
                if let Some(j) = self.blocks.iter().position(|c| c.offset == target) {
                    for pair in [(i, j), (j, i)] {
                        if !pairs.contains(&pair) {
                            pairs.push(pair);
                        }
                    }
                }
            }
        }
        pairs
    }

    /// 𝒦's entry between the explicit places p and q (of different offsets) less its decay −(i/2)D: the decay between the
    /// offsets is their group's jumps' cross products, L_o†L_o′ at e^{i(o − o′)t} (module documentation, # Tones), D =
    /// Σ_q C_q†C_q the emission's.
    pub(crate) fn hermitian_excited(&self, excited: &Mat, p: usize, q: usize) -> C64 {
        let ng = self.ground.len();
        let (x, y) = (self.explicit[p - ng], self.explicit[q - ng]);
        let decay: C64 = self.emission.iter().map(|c| (0..GROUND).map(|g| c[(g, x)].conj() * c[(g, y)]).sum::<C64>()).sum();
        excited[(x, y)] + C64::new(0.0, 0.5) * decay
    }

    /// The cross term of two blocks a kept copy relates (module documentation, # Tones), ε_i†(𝒦 − ν̄)ε_j with the frame part of
    /// the Galerkin term taken from each column's own offset: entry (p, q) holds ε_i†𝒦ε_j less ½[(o_i − θ_p) + (o_j − θ_q)] of
    /// the shared states' overlap ε_i†ε_j -- the offsets of the columns' excited amplitudes from their own frames, what the
    /// dropped beating metric (M8) would carry of a common frame (gauge invariant: a shift of every frame by c shifts every
    /// offset by c). Its matrices: 𝒦 on the two blocks' excited rows, the overlap's (1 on the states they share), and the
    /// per-entry offsets.
    fn cross(&self, site: &Site, i: usize, j: usize) -> (Mat, Mat, Mat) {
        let (ri, rj) = (self.blocks[i].excited(self), self.blocks[j].excited(self));
        let k = Mat::from_fn(ri.len(), rj.len(), |a, b| site.excited[(ri[a], rj[b])]);
        let e = Mat::from_fn(ri.len(), rj.len(), |a, b| if ri[a] == rj[b] { C64::ONE } else { C64::ZERO });
        let n = self.size();
        let (oi, oj) = (self.blocks[i].value, self.blocks[j].value);
        let offsets = Mat::from_fn(n, n, |p, q| C64::real(0.5 * ((oi - self.frame_values[p]) + (oj - self.frame_values[q]))));
        (k, e, offsets)
    }

    fn tones(&self, site: &Site, amplitudes: &[Mat], normal: &Mat, work: &mut f64) -> Vec<Tone> {
        let n = self.size();
        let mut out: Vec<Tone> = vec![];
        let add = |beat: Beat, value: f64, m: Mat, out: &mut Vec<Tone>| match out.iter_mut().find(|t| t.beat == beat) {
            Some(t) => plus(&mut t.a, &m),
            None => out.push(Tone { beat, value, a: m, generator: Mat::default() }),
        };
        let ng = self.ground.len();
        // P's entries of places of one kind and different frames whose coupling cannot be folded (`Fold::turning`).
        for x in (0..n * n).filter(|&x| self.turning[x]) {
            let (p, q) = (x / n, x % n);
            let z = if p < ng { site.ground[(self.ground[p], self.ground[q])] } else { self.hermitian_excited(&site.excited, p, q) };
            if z != C64::ZERO {
                let mut m = Mat::zeros(n, n);
                m[(p, q)] = z;
                add(self.frames[p].plus(-1, &self.frames[q]), self.frame_values[p] - self.frame_values[q], m, &mut out);
            }
        }
        for (b, e) in self.blocks.iter().zip(amplitudes) {
            let rows = b.excited(self);
            for &(t, g, ref beat) in &b.kept {
                let Some(pt) = self.ground.iter().position(|&s| s == t) else { continue };
                let v = Mat::from_fn(rows.len(), 1, |i, _| site.drives[g][(rows[i], t)]);
                let row = mul_adj(&v, e, work);
                let mut m = Mat::zeros(n, n);
                for j in 0..n {
                    m[(pt, j)] = row[(0, j)];
                }
                let value = self.frame_values[pt] - (b.value - self.components[g].nu);
                add(beat.clone(), value, m.clone(), &mut out);
                add(beat.negated(), -value, m.adjoint(), &mut out);
            }
        }
        // The beats of the blocks a kept copy relates.
        for (i, j) in self.related() {
            let (k, e, offsets) = self.cross(site, i, j);
            let mut m = mul_adj(&amplitudes[i], &mul(&k, &amplitudes[j], work), work);
            let overlap = mul_adj(&amplitudes[i], &mul(&e, &amplitudes[j], work), work);
            for (x, (y, d)) in m.data.iter_mut().zip(overlap.data.iter().zip(&offsets.data)) {
                *x -= *y * *d;
            }
            add(self.blocks[i].offset.plus(-1, &self.blocks[j].offset), self.blocks[i].value - self.blocks[j].value, m, &mut out);
        }
        for t in out.iter_mut() {
            t.generator = mul(&mul(normal, &t.a, work), normal, work);
        }
        out.sort_by(|x, y| x.value.total_cmp(&y.value));
        out
    }
}

/// The light's fields at a point (`Fold::site`): the beams' amplitudes, H_GG and 𝒦 with their gradients, each component's
/// drive V_c, and per component and beam its Doppler shift w (rad/µs).
struct Site {
    a: Vec<Amplitude>,
    ground: Mat,
    excited: Mat,
    drives: Vec<Mat>,
    doppler: Vec<Vec<f64>>,
}

/// A block's matrices at a point (`Fold::lay`): its frame matrix (Q rows), its sources per split with the split's Doppler
/// shift and beam (component, beam), its emission into P, its part of Ĥ_PP, its centre.
struct Laid {
    inner: Mat,
    absorb: Vec<Mat>,
    shifts: Vec<f64>,
    beams: Vec<Option<(usize, usize)>>,
    emit: Mat,
    pp: Mat,
    z0: f64,
}

/// H = U T U† (complex Schur form): T upper triangular, U unitary.
#[derive(Clone)]
struct Triangle {
    t: Mat,
    u: Mat,
}

impl Triangle {
    fn of(h: &Mat, work: &mut f64) -> Option<Triangle> {
        let n = h.rows;
        if n == 0 {
            return Some(Triangle { t: Mat::zeros(0, 0), u: Mat::zeros(0, 0) });
        }
        let mut t = h.data.clone();
        let (mut q, mut v, mut s) = (vec![C64::ZERO; n * n], vec![C64::ZERO; n], vec![C64::ZERO; n]);
        let (w, _) = schur(&mut t, &mut q, &mut v, &mut s, n)?;
        *work += w as f64;
        let mut t = Mat { rows: n, cols: n, data: t };
        for i in 0..n {
            for j in 0..i {
                t[(i, j)] = C64::ZERO;
            }
        }
        Some(Triangle { t, u: Mat::from_fn(n, n, |i, k| q[k * n + i]) })
    }
}

/// A block's matrix in Schur form part by part: its connected parts (the rows its entries join), each K_c = U_c T_c U_c†,
/// laid side by side -- T block diagonal (each part's rows contiguous, `ends` each row's reach: one past its last entry),
/// V the parts' U on their rows. One decomposition whatever the parts (a part is the whole where all rows join); the parts
/// save work alone (the m blocks of DESIGN §6.2), the result the dense one's. An entry off the diagonal within the
/// decomposition's rounding (γ_m ‖K_c‖_F) is zero, the norm of those dropped kept (`dropped`).
struct Factored {
    t: Mat,
    ends: Vec<usize>,
    parts: Vec<(Vec<usize>, usize, Mat)>,
    dropped: Vec<f64>,
}

impl Factored {
    fn of(k: &Mat, work: &mut f64) -> Option<Factored> {
        let m = k.rows;
        let mut part: Vec<usize> = (0..m).collect();
        fn find(part: &mut [usize], mut x: usize) -> usize {
            while part[x] != x {
                part[x] = part[part[x]];
                x = part[x];
            }
            x
        }
        for i in 0..m {
            for j in 0..m {
                if i != j && k[(i, j)] != C64::ZERO {
                    let (a, b) = (find(&mut part, i), find(&mut part, j));
                    part[a.max(b)] = a.min(b);
                }
            }
        }
        let mut groups: Vec<Vec<usize>> = vec![];
        let mut roots: Vec<usize> = vec![];
        for i in 0..m {
            let r = find(&mut part, i);
            match roots.iter().position(|&x| x == r) {
                Some(at) => groups[at].push(i),
                None => {
                    roots.push(r);
                    groups.push(vec![i]);
                }
            }
        }
        let mut t = Mat::zeros(m, m);
        let mut ends = vec![0; m];
        let mut parts = vec![];
        let mut at = 0;
        let mut dropped = vec![];
        for idx in groups {
            let kc = k.block(&idx, &idx);
            let tri = Triangle::of(&kc, work)?;
            let len = idx.len();
            let floor = crate::units::rounding(len) * kc.frobenius();
            let mut lost = 0.0;
            for a in 0..len {
                ends[at + a] = at + a + 1;
                for b in 0..len {
                    let x = tri.t[(a, b)];
                    if b > a && x.abs() <= floor {
                        lost += x.norm_sqr();
                        continue;
                    }
                    t[(at + a, at + b)] = x;
                    if b > a && x != C64::ZERO {
                        ends[at + a] = at + b + 1;
                    }
                }
            }
            parts.push((idx, at, tri.u));
            dropped.push(f64::sqrt(lost));
            at += len;
        }
        Some(Factored { t, ends, parts, dropped })
    }

    /// V† m (m's rows the block's), part by part.
    fn to(&self, m: &Mat, work: &mut f64) -> Mat {
        let mut out = Mat::zeros(m.rows, m.cols);
        for (idx, at, u) in &self.parts {
            for b in 0..idx.len() {
                for (a, &i) in idx.iter().enumerate() {
                    let z = u[(a, b)].conj();
                    if z != C64::ZERO {
                        for j in 0..m.cols {
                            out[(at + b, j)] += z * m[(i, j)];
                        }
                    }
                }
            }
            *work += (idx.len() * idx.len() * m.cols) as f64;
        }
        out
    }

    /// V z (z's rows the parts' laid side by side), part by part.
    fn from(&self, z: &Mat, work: &mut f64) -> Mat {
        let mut out = Mat::zeros(z.rows, z.cols);
        for (idx, at, u) in &self.parts {
            for (a, &i) in idx.iter().enumerate() {
                for b in 0..idx.len() {
                    let c = u[(a, b)];
                    if c != C64::ZERO {
                        for j in 0..z.cols {
                            out[(i, j)] += c * z[(at + b, j)];
                        }
                    }
                }
            }
            *work += (idx.len() * idx.len() * z.cols) as f64;
        }
        out
    }

    /// The triangular solve (c − T) z = r with T's diagonal `diag`, part by part (a column of `sylvester`).
    fn back(&self, diag: &[C64], c: C64, r: &[C64], work: &mut f64) -> Vec<C64> {
        let m = r.len();
        let mut z = vec![C64::ZERO; m];
        for k in (0..m).rev() {
            let mut acc = r[k];
            for l in k + 1..self.ends[k] {
                acc += self.t[(k, l)] * z[l];
            }
            z[k] = if acc == C64::ZERO { C64::ZERO } else { acc / (c - diag[k]) };
            *work += (self.ends[k] - k) as f64;
        }
        z
    }
}

/// The decompositions kept (module documentation, # One decomposition along the path), one cache per fold and shared by
/// every atom of it: a block's K_eff + o depends on the place only through the light that reaches it, not on the atom, so
/// a decomposition made by one atom serves any other where its reading is within the share. Per basis the matrix it
/// decomposed (a block's K_eff + o on its solved rows, the exact copies' and the lines' couplings left out), its Schur form
/// and per block its sources' ranks in it (per component of its sources, per basis vector U_k, V† of the unit source on
/// its source columns). `current`: per key (solved rows, the block where its copies dress it) the basis the atom stands
/// on -- an atom's own; beyond the share it looks for the nearest kept one (‖K − K_basis‖_F) and makes one only where that
/// one is beyond the share too. `made`, `looked`: the decompositions made and the kept ones taken up again.
#[derive(Default)]
pub(crate) struct Path {
    bases: Vec<Basis>,
    pub(crate) current: Vec<((Vec<Row>, Option<usize>), usize)>,
    pub(crate) made: usize,
    pub(crate) looked: usize,
    /// The largest condition of a made decomposition's eigenvectors (1: a normal one).
    pub(crate) condition: f64,
}

impl Path {
    /// The decompositions kept.
    pub(crate) fn bases_len(&self) -> usize {
        self.bases.len()
    }
}

struct Basis {
    rows: Vec<Row>,
    own: Option<usize>,
    k: Mat,
    factored: Rc<Factored>,
    ranks: Vec<Option<Rc<Vec<(usize, Vec<Mat>)>>>>,
}

/// A block's matrices at the point before its decomposition: K_eff + o on its solved rows (its centred rows eliminated at
/// z₀, R₀ = (z₀ − Ĥ_CC)⁻¹), R₀, its static split's effective source (None where it has none), and per beam split the
/// dressing of its source through the centred rows, Ĥ_SC R₀ B_C (None where its source has no centred rows).
struct Effective {
    k: Mat,
    resolvent: Mat,
    source: Option<Mat>,
    dressed: Vec<Option<Mat>>,
}

/// A block's evaluation order at a point (module documentation, # Blocks): its solved rows, R₀, its basis (`Path`) with
/// the diagonal at the point (first order, less o), per row of the basis and part the norm of what the diagonal leaves of
/// the row on the part (`off`: ‖δK̃_i,P∖i‖, its own part's with what the Schur form dropped), its static split's source and
/// its ranks in the basis.
struct Eliminated {
    solved: usize,
    resolvent: Mat,
    basis: usize,
    factored: Rc<Factored>,
    diag: Vec<C64>,
    off: Vec<Vec<f64>>,
    to_static: Option<Mat>,
    to_dressed: Vec<Option<Mat>>,
    ranks: Rc<Vec<(usize, Vec<Mat>)>>,
    /// The exact copies: their count (the solved rows' last), and in the basis their couplings with the excited rows,
    /// Ỹ = V_E†K_ED V_D and Ỹ′ = V_D†K_DE V_E (module documentation, # The exact copies).
    exact: usize,
    coupling: Mat,
    back_coupling: Mat,
}

impl Eliminated {
    fn to(&self, m: &Mat, work: &mut f64) -> Mat {
        self.factored.to(m, work)
    }
    fn from(&self, z: &Mat, work: &mut f64) -> Mat {
        self.factored.from(z, work)
    }
    fn back(&self, c: C64, r: &[C64], work: &mut f64) -> Vec<C64> {
        if self.exact == 0 {
            return self.factored.back(&self.diag, c, r, work);
        }
        self.column(c, r, false, work)
    }
    fn sylvester(&self, th: &Mat, w: f64, b: &Mat, work: &mut f64) -> Mat {
        if self.exact == 0 {
            return sylvester(&self.factored.t, &self.diag, &self.factored.ends, th, w, b, work);
        }
        let (m, n) = (b.rows, b.cols);
        let mut z = Mat::zeros(m, n);
        for j in 0..n {
            let mut rhs: Vec<C64> = (0..m).map(|k| b[(k, j)]).collect();
            for i in 0..j {
                let t = th[(i, j)];
                if t != C64::ZERO {
                    (0..m).for_each(|k| rhs[k] -= z[(k, i)] * t);
                }
            }
            let col = self.column(th[(j, j)] - C64::real(w), &rhs, false, work);
            (0..m).for_each(|k| z[(k, j)] = col[k]);
        }
        *work += (m * n * n.saturating_sub(1) / 2) as f64;
        z
    }

    /// The triangle's solve on the rows `lo..hi` of the basis: (c − T)z = r, or (c − T†)z = r (`adjoint`, c already the
    /// adjoint's), each part's own; a row with nothing to answer answers nothing.
    fn triangle_on(&self, c: C64, r: &[C64], lo: usize, hi: usize, adjoint: bool, work: &mut f64) -> Vec<C64> {
        let (t, ends) = (&self.factored.t, &self.factored.ends);
        let mut z = vec![C64::ZERO; hi - lo];
        if adjoint {
            for k in lo..hi {
                let mut acc = r[k - lo];
                for l in (lo..k).filter(|&l| ends[l] > k) {
                    acc += t[(l, k)].conj() * z[l - lo];
                }
                z[k - lo] = if acc == C64::ZERO { C64::ZERO } else { acc / (c - self.diag[k].conj()) };
            }
        } else {
            for k in (lo..hi).rev() {
                let mut acc = r[k - lo];
                for l in k + 1..ends[k].min(hi) {
                    acc += t[(k, l)] * z[l - lo];
                }
                z[k - lo] = if acc == C64::ZERO { C64::ZERO } else { acc / (c - self.diag[k]) };
            }
        }
        *work += (lo..hi).map(|k| ends[k] - k).sum::<usize>() as f64;
        z
    }

    /// One column of the block's solve with exact copies (module documentation, # The exact copies): (c − K)z = r, or
    /// (c − K†)z = r, the excited rows first, then the copies' Schur complement.
    fn column(&self, c: C64, r: &[C64], adjoint: bool, work: &mut f64) -> Vec<C64> {
        let (ne, nd) = (self.solved - self.exact, self.exact);
        // (B, C): the couplings into the excited rows and out of them, as the system takes them.
        let (b, cc) = if adjoint { (self.back_coupling.adjoint(), self.coupling.adjoint()) } else { (self.coupling.clone(), self.back_coupling.clone()) };
        let y = self.triangle_on(c, &r[..ne], 0, ne, adjoint, work);
        let p: Vec<Vec<C64>> = (0..nd).map(|a| self.triangle_on(c, &(0..ne).map(|k| b[(k, a)]).collect::<Vec<_>>(), 0, ne, adjoint, work)).collect();
        let t = &self.factored.t;
        let schur = Mat::from_fn(nd, nd, |a, d| {
            let (i, j) = (ne + a, ne + d);
            let own = if a == d {
                c - if adjoint { self.diag[i].conj() } else { self.diag[i] }
            } else if adjoint {
                -t[(j, i)].conj()
            } else {
                -t[(i, j)]
            };
            own - (0..ne).map(|k| cc[(a, k)] * p[d][k]).sum::<C64>()
        });
        *work += (nd * nd * ne) as f64;
        let rhs = Mat::from_fn(nd, 1, |a, _| r[ne + a] + (0..ne).map(|k| cc[(a, k)] * y[k]).sum::<C64>());
        let x = match inverse(&schur, work) {
            Some(inv) => mul(&inv, &rhs, work),
            None => Mat::from_fn(nd, 1, |_, _| C64::new(f64::NAN, f64::NAN)),
        };
        let mut out = y;
        for (a, pa) in p.iter().enumerate() {
            for k in 0..ne {
                out[k] += pa[k] * x[(a, 0)];
            }
        }
        out.extend((0..nd).map(|a| x[(a, 0)]));
        *work += (2 * nd * ne) as f64;
        out
    }

    /// The reuse's error of a split's solution Z (Schur coordinates) at H's Schur diagonal `th` less w, per row and column,
    /// added to `err` (module documentation, # One decomposition along the path): Σ_L ‖δK̃_i,L∖i‖ ‖Z_L∖i‖ / |c − κ_i|.
    fn reuse(&self, th: &Mat, w: f64, z: &Mat, err: &mut [Vec<f64>], work: &mut f64) {
        let parts = &self.factored.parts;
        for j in 0..z.cols {
            let c = th[(j, j)] - C64::real(w);
            let norms: Vec<f64> = parts.iter().map(|(idx, at, _)| (*at..at + idx.len()).map(|k| z[(k, j)].norm_sqr()).sum()).collect();
            for (p, (idx, at, _)) in parts.iter().enumerate() {
                for i in *at..at + idx.len() {
                    let x: f64 = (0..parts.len()).map(|q| self.off[i][q] * if q == p { (norms[q] - z[(i, j)].norm_sqr()).max(0.0) } else { norms[q] }.sqrt()).sum();
                    err[i][j] += match x {
                        0.0 => 0.0,
                        x => x / (c - self.diag[i]).abs(),
                    };
                }
            }
            *work += (z.rows * (2 + parts.len())) as f64;
        }
    }
    fn adjoint(&self, th: &Mat, w: f64, b: &Mat, work: &mut f64) -> Mat {
        if self.exact == 0 {
            return sylvester_adjoint(&self.factored.t, &self.diag, &self.factored.ends, th, w, b, work);
        }
        let (m, n) = (b.rows, b.cols);
        let mut z = Mat::zeros(m, n);
        for j in (0..n).rev() {
            let mut rhs: Vec<C64> = (0..m).map(|k| b[(k, j)]).collect();
            for i in j + 1..n {
                let t = th[(j, i)].conj();
                if t != C64::ZERO {
                    (0..m).for_each(|k| rhs[k] -= z[(k, i)] * t);
                }
            }
            let col = self.column(th[(j, j)].conj() - C64::real(w), &rhs, true, work);
            (0..m).for_each(|k| z[(k, j)] = col[k]);
        }
        *work += (m * n * n.saturating_sub(1) / 2) as f64;
        z
    }

    /// The block's response on all its Q rows from that on its solved rows: the centred ones T_C = R₀(Ĥ_CS T_S + B_C), B_C
    /// every split's source on them.
    fn rows(&self, l: &Laid, t: &Mat, work: &mut f64) -> Mat {
        let m = l.inner.rows;
        let n = t.cols;
        let (s, c): (Vec<usize>, Vec<usize>) = ((0..self.solved).collect(), (self.solved..m).collect());
        if c.is_empty() {
            return t.clone();
        }
        let all: Vec<usize> = (0..n).collect();
        let mut tc = mul_sparse(&l.inner.block(&c, &s), t, work);
        for a in &l.absorb {
            plus(&mut tc, &a.block(&c, &all));
        }
        let tc = mul(&self.resolvent, &tc, work);
        Mat::from_fn(m, n, |i, j| if i < self.solved { t[(i, j)] } else { tc[(i - self.solved, j)] })
    }
}

impl Fold {
    /// A block's matrices before its decomposition (`Effective`). None where R₀ is singular.
    fn effective(&self, o: usize, l: &Laid, work: &mut f64) -> Option<Effective> {
        let b = &self.blocks[o];
        let m = b.rows.len();
        let n = l.absorb[0].cols;
        let (s, c): (Vec<usize>, Vec<usize>) = ((0..b.solved).collect(), (b.solved..m).collect());
        let all: Vec<usize> = (0..n).collect();
        let resolvent = {
            let r = Mat::from_fn(c.len(), c.len(), |i, j| if i == j { C64::real(l.z0) } else { C64::ZERO } - l.inner[(c[i], c[j])]);
            inverse(&r, work)?
        };
        let mut k = l.inner.block(&s, &s);
        let mut source = l.absorb[0].block(&s, &all);
        if !c.is_empty() {
            let rc = mul(&l.inner.block(&s, &c), &resolvent, work);
            plus(&mut k, &mul(&rc, &l.inner.block(&c, &s), work));
            plus(&mut source, &mul(&rc, &l.absorb[0].block(&c, &all), work));
        }
        for i in 0..b.solved {
            k[(i, i)] += C64::real(b.value);
        }
        let source = source.data.iter().any(|z| *z != C64::ZERO).then_some(source);
        let mut dressed = vec![None];
        for a in &l.absorb[1..] {
            let ac = a.block(&c, &all);
            dressed.push((!c.is_empty() && ac.data.iter().any(|z| *z != C64::ZERO)).then(|| mul(&mul(&l.inner.block(&s, &c), &resolvent, work), &ac, work)));
        }
        Some(Effective { k, resolvent, source, dressed })
    }

    /// The block's evaluation order at the point from its matrices and the path's basis: the basis found, or made where it
    /// is missing or to be made again (`renew`); the diagonal moved by the change's first order; the remainder.
    #[allow(clippy::type_complexity)]
    #[allow(clippy::type_complexity)]
    fn eliminate(&self, o: usize, e: Effective, path: &mut Path, renew: &[(Vec<Row>, Option<usize>)], moved: &mut Vec<Option<(Vec<C64>, Vec<Vec<f64>>)>>, work: &mut f64) -> Option<Eliminated> {
        let b = &self.blocks[o];
        let rows = b.rows[..b.solved].to_vec();
        let dressed = b.rows.len() > b.solved || rows.iter().any(|r| matches!(r, Row::Copy { .. }));
        let own = dressed.then_some(o);
        // The exact copies' couplings with the excited rows are not decomposed (module documentation, # The exact copies).
        let ne = rows.iter().take_while(|r| matches!(r, Row::Excited(_))).count();
        let mut kept = e.k.clone();
        for i in 0..b.solved {
            for j in 0..b.solved {
                if (i < ne) != (j < ne) {
                    kept[(i, j)] = C64::ZERO;
                }
            }
        }
        let k = kept.clone();
        // The basis: the atom's own; beyond the share (`renew`, the keys whose reading passed it) first the nearest kept one
        // of the key, then a new one.
        let key = (rows.clone(), own);
        let current = path.current.iter().position(|x| x.0 == key);
        let tries = renew.iter().filter(|x| **x == key).count();
        let nearest = |path: &Path, but: Option<usize>, work: &mut f64| -> Option<usize> {
            let mut best: Option<(usize, f64)> = None;
            for (i, x) in path.bases.iter().enumerate().filter(|(i, x)| x.rows == rows && x.own == own && Some(*i) != but) {
                let mut d = kept.clone();
                d.add_scaled(-C64::ONE, &x.k);
                let size = d.frobenius();
                *work += (d.rows * d.cols) as f64;
                if best.is_none_or(|b| size < b.1) {
                    best = Some((i, size));
                }
            }
            best.map(|b| b.0)
        };
        let standing = current.map(|c| path.current[c].1);
        let chosen = match (standing, tries) {
            (Some(at), 0) => Some(at),
            (None, 0) => nearest(path, None, work),
            (_, 1) => nearest(path, standing, work),
            _ => None,
        };
        if chosen.is_some() && chosen != standing {
            path.looked += 1;
        }
        let at = match chosen {
            Some(at) => at,
            None => {
                let factored = Rc::new(Factored::of(&k, work)?);
                path.made += 1;
                for (idx, _, _) in &factored.parts {
                    let mut x = crate::linalg::Eigen::default();
                    if x.decompose(&k.block(idx, idx)) {
                        path.condition = path.condition.max(x.condition);
                    }
                }
                path.bases.push(Basis { rows, own, k, factored, ranks: vec![None; self.blocks.len()] });
                moved.push(None);
                path.bases.len() - 1
            }
        };
        match current {
            Some(c) => path.current[c].1 = at,
            None => path.current.push((key, at)),
        }
        while moved.len() < path.bases.len() {
            moved.push(None);
        }
        let factored = path.bases[at].factored.clone();
        // The change since the basis was made in the basis, δK̃ = V†δV: its diagonal the first order, each row's rest its
        // reading's norm, with what the form dropped.
        if moved[at].is_none() {
            let mut delta = kept.clone();
            delta.add_scaled(-C64::ONE, &path.bases[at].k);
            let tilde = factored.to(&factored.to(&delta, work).adjoint(), work).adjoint();
            let d: Vec<C64> = (0..b.solved).map(|i| tilde[(i, i)]).collect();
            let mut off = vec![vec![0.0; factored.parts.len()]; b.solved];
            for (p, (idx, at, _)) in factored.parts.iter().enumerate() {
                for i in *at..at + idx.len() {
                    for (q, (jdx, bt, _)) in factored.parts.iter().enumerate() {
                        let sum: f64 = (*bt..bt + jdx.len()).filter(|&k| k != i).map(|k| tilde[(i, k)].norm_sqr()).sum();
                        off[i][q] = sum.sqrt() + if q == p { factored.dropped[p] } else { 0.0 };
                    }
                }
            }
            *work += (delta.rows * delta.cols) as f64;
            moved[at] = Some((d, off));
        }
        let (d, off) = moved[at].clone().unwrap();
        let diag: Vec<C64> = (0..b.solved).map(|i| factored.t[(i, i)] + d[i] - C64::real(b.value)).collect();
        // The sources' ranks in the basis, made with it, kept with it.
        if path.bases[at].ranks[o].is_none() {
            let n = self.size();
            let mut ranks = vec![];
            for (ci, comp) in self.components.iter().enumerate() {
                let places: Vec<usize> = b.sources.iter().filter(|x| x.1 == ci).map(|x| x.0).collect();
                if places.is_empty() {
                    continue;
                }
                let per: Vec<Mat> = comp
                    .basis
                    .iter()
                    .map(|u| {
                        let mut r = Mat::zeros(b.solved, n);
                        for &p in &places {
                            for i in 0..b.solved {
                                if let Row::Excited(y) = b.rows[i] {
                                    r[(i, p)] = u[(y, self.ground[p])];
                                }
                            }
                        }
                        factored.to(&r, work)
                    })
                    .collect();
                ranks.push((ci, per));
            }
            path.bases[at].ranks[o] = Some(Rc::new(ranks));
        }
        let ranks = path.bases[at].ranks[o].clone().unwrap();
        let to_static = e.source.as_ref().map(|x| factored.to(x, work));
        let to_dressed: Vec<Option<Mat>> = e.dressed.iter().map(|d| d.as_ref().map(|x| factored.to(x, work))).collect();
        // The exact copies' couplings in the basis at the point.
        let nd = b.solved - ne;
        let (coupling, back_coupling) = if nd > 0 {
            let mut ed = Mat::zeros(b.solved, b.solved);
            for i in 0..b.solved {
                for j in 0..b.solved {
                    if (i < ne) != (j < ne) {
                        ed[(i, j)] = e.k[(i, j)];
                    }
                }
            }
            let tilde = factored.to(&factored.to(&ed, work).adjoint(), work).adjoint();
            let (es, ds): (Vec<usize>, Vec<usize>) = ((0..ne).collect(), (ne..b.solved).collect());
            (tilde.block(&es, &ds), tilde.block(&ds, &es))
        } else {
            (Mat::zeros(0, 0), Mat::zeros(0, 0))
        };
        Some(Eliminated { solved: b.solved, resolvent: e.resolvent, basis: at, factored: path.bases[at].factored.clone(), diag, off, to_static, to_dressed, ranks, exact: nd, coupling, back_coupling })
    }
}

/// A square matrix's inverse by LU (None where it is singular).
fn inverse(a: &Mat, work: &mut f64) -> Option<Mat> {
    let n = a.rows;
    if n == 0 {
        return Some(Mat::zeros(0, 0));
    }
    let mut f = a.data.clone();
    let mut pivots = vec![0; n];
    if !crate::linalg::factor(&mut f, &mut pivots, n) {
        return None;
    }
    let mut b = Mat::identity(n);
    crate::linalg::solve(&f, &pivots, n, &mut b.data, n);
    *work += (n * n * n) as f64 * 4.0 / 3.0;
    Some(b)
}

/// Z with Z T_H − (T_K + w) Z = B, T_H (n × n) and T_K (m × m) upper triangular: column by column the triangular system
/// (t_jj − w − T_K) z_j = B_j − Σ_{i<j} z_i t_ij (module documentation, # One linear solve per pass).
/// `diag` T_K's diagonal (its strictly upper part `tk`'s).
fn sylvester(tk: &Mat, diag: &[C64], ends: &[usize], th: &Mat, w: f64, b: &Mat, work: &mut f64) -> Mat {
    let (m, n) = (b.rows, b.cols);
    let mut z = Mat::zeros(m, n);
    let mut rhs = vec![C64::ZERO; m];
    for j in 0..n {
        for k in 0..m {
            rhs[k] = b[(k, j)];
        }
        for i in 0..j {
            let t = th[(i, j)];
            if t != C64::ZERO {
                for k in 0..m {
                    rhs[k] -= z[(k, i)] * t;
                }
            }
        }
        let c = th[(j, j)] - C64::real(w);
        for k in (0..m).rev() {
            let mut acc = rhs[k];
            for l in k + 1..ends[k] {
                acc += tk[(k, l)] * z[(l, j)];
            }
            // A row with nothing to answer answers nothing (a decoupled state, even where its energy is the column's).
            z[(k, j)] = if acc == C64::ZERO { C64::ZERO } else { acc / (c - diag[k]) };
        }
    }
    *work += (m * n * n.saturating_sub(1) / 2 + n * (0..m).map(|k| ends[k] - k).sum::<usize>()) as f64;
    z
}

/// Z with Z T_H† − (T_K + w)† Z = B: the adjoint of `sylvester`, the same triangular forms solved from the last column and
/// the first row.
fn sylvester_adjoint(tk: &Mat, diag: &[C64], ends: &[usize], th: &Mat, w: f64, b: &Mat, work: &mut f64) -> Mat {
    let (m, n) = (b.rows, b.cols);
    let mut z = Mat::zeros(m, n);
    let mut rhs = vec![C64::ZERO; m];
    for j in (0..n).rev() {
        for k in 0..m {
            rhs[k] = b[(k, j)];
        }
        for i in j + 1..n {
            let t = th[(j, i)].conj();
            if t != C64::ZERO {
                for k in 0..m {
                    rhs[k] -= z[(k, i)] * t;
                }
            }
        }
        let c = th[(j, j)].conj() - C64::real(w);
        for k in 0..m {
            let mut acc = rhs[k];
            for l in (0..k).filter(|&l| ends[l] > k) {
                acc += tk[(l, k)].conj() * z[(l, j)];
            }
            z[(k, j)] = if acc == C64::ZERO { C64::ZERO } else { acc / (c - diag[k].conj()) };
        }
    }
    *work += (m * n * n.saturating_sub(1) / 2 + n * (0..m).map(|k| ends[k] - k).sum::<usize>()) as f64;
    z
}

/// One pass: its H's Schur form; per block and split the response on the solved rows and its source in the Schur
/// coordinates (Z = V†TU, V†BU); per block the response on all Q rows, and per column the reuse's error (`Eliminated::reuse`)
/// and the solved rows' squared response.
struct Stage {
    triangle: Triangle,
    splits: Vec<Vec<Mat>>,
    sources: Vec<Vec<Mat>>,
    responses: Vec<Mat>,
    errors: Vec<Vec<f64>>,
    sizes: Vec<Vec<f64>>,
}

impl Stage {
    /// The next pass's reading (rate units; module documentation, # The passes as the next one reads): the largest over the
    /// columns of Σ_i |δH_ij| ‖Z_i‖ over the column's distance from each block's diagonal, relative to its whole response,
    /// δH = `bloch` − `h` in H's Schur basis.
    fn next(&self, eliminated: &[Eliminated], h: &Mat, bloch: &Mat, work: &mut f64) -> f64 {
        let n = h.rows;
        // A response that is not finite (an energy on the diagonal itself) reads beyond any share.
        if self.sizes.iter().flatten().any(|x| !x.is_finite()) {
            return f64::INFINITY;
        }
        let mut dh = bloch.clone();
        dh.add_scaled(-C64::ONE, h);
        let u = &self.triangle.u;
        let dh = mul(&mul_adj(u, &dh, work), u, work);
        let whole: Vec<f64> = (0..n).map(|j| self.sizes.iter().map(|x| x[j]).sum::<f64>()).collect();
        let floor = (crate::units::rounding(n) * whole.iter().sum::<f64>().sqrt()).powi(2);
        let mut err = vec![0.0; n];
        for (e, size) in eliminated.iter().zip(&self.sizes) {
            for j in 0..n {
                let gap = e.diag.iter().map(|d| (self.triangle.t[(j, j)] - *d).abs()).fold(f64::INFINITY, f64::min);
                // At the diagonal itself (a gap of zero) a change reads infinite, no change nothing.
                let x = match (0..n).map(|i| dh[(i, j)].abs() * size[i].sqrt()).sum::<f64>() {
                    0.0 => 0.0,
                    x => x / gap,
                };
                err[j] += x * x;
            }
            *work += (n * (n + e.solved)) as f64;
        }
        (0..n).filter(|&j| whole[j] > floor).map(|j| rate((err[j] / whole[j]).sqrt())).fold(0.0, f64::max)
    }

    /// The reuse's errors of its splits (`Eliminated::reuse`), per block and column.
    fn read_reuse(&mut self, eliminated: &[Eliminated], laid: &[Laid], work: &mut f64) {
        let n = self.triangle.t.rows;
        self.errors = eliminated
            .iter()
            .zip(laid)
            .zip(&self.splits)
            .map(|((e, l), per)| {
                let mut err = vec![vec![0.0; n]; e.solved];
                for (z, &w) in per.iter().zip(&l.shifts) {
                    e.reuse(&self.triangle.t, w, z, &mut err, work);
                }
                (0..n).map(|j| err.iter().map(|x| x[j] * x[j]).sum::<f64>().sqrt()).collect()
            })
            .collect();
    }

    /// Per block the reuse's reading (rate units): the largest over the columns of its error relative to the column's whole
    /// response (every block's; a column at the responses' rounding reads nothing).
    fn reuse(&self) -> Vec<f64> {
        let n = self.triangle.t.rows;
        let whole: Vec<f64> = (0..n).map(|j| self.sizes.iter().map(|x| x[j]).sum::<f64>()).collect();
        let floor = (crate::units::rounding(n) * whole.iter().sum::<f64>().sqrt()).powi(2);
        self.errors.iter().map(|e| (0..n).filter(|&j| whole[j] > floor).map(|j| rate(e[j] / whole[j].sqrt())).fold(0.0, f64::max)).collect()
    }
}

/// A tone: its exact frequency and value (the term e^{iΩt}), its matrix in x coordinates and in the normalized ones.
pub(crate) struct Tone {
    pub(crate) beat: Beat,
    pub(crate) value: f64,
    pub(crate) a: Mat,
    pub(crate) generator: Mat,
}

/// A node's readings (DESIGN §3.1), each the largest over H's Schur columns of the squared relative amplitude error ρ²
/// (the column's whole excited response its denominator): the next pass (`next`), the light pattern (`pattern`), the
/// tones' sidebands (`sideband`, a kept copy's), the evaluation order (`order`, its status's: of a centred copy its energy
/// dependence, of an omitted one its whole dressing, of a decomposed one nothing), the second order's truncation
/// (`truncation`, a folded copy's reabsorption), and a copy's mixing w = ½(1 − 1/√(1 + 4|c|²)) (`mixing`); beside them, of
/// every copy, what centring it and what leaving it out would make (`centring`, `omission`): one function, the block
/// perturbation R(z_j) δK T_j, δK = Σ(z) − Σ(z₀) or Σ(z) -- the evaluation order's rule reads them (DESIGN M18). An
/// excited row's error is relative to its column's whole excited response (a rate's), a copy row's absolute (a
/// population's, per unit P amplitude).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Reading {
    pub(crate) block: usize,
    pub(crate) node: Node,
    pub(crate) next: f64,
    pub(crate) pattern: f64,
    pub(crate) sideband: f64,
    pub(crate) order: f64,
    pub(crate) truncation: f64,
    pub(crate) mixing: f64,
    pub(crate) centring: f64,
    pub(crate) omission: f64,
}

impl Reading {
    /// Its rate units' total, rate(ρ) of the parts' sum (`elim::rate`'s 2ρ + ρ²).
    pub(crate) fn total(&self) -> f64 {
        rate((self.next + self.pattern + self.sideband + self.order + self.truncation).sqrt())
    }
}

/// A node of a block: an excited level, or a copy of a ground level F by an emitting component.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Node {
    Excited(usize),
    Copy { level: u8, component: usize },
}

/// The fold at a point (`Fold::at`): P's size; the static compression A, the metric less one X = M − 1, S⁻¹ and G = S⁻¹AS⁻¹; per block its
/// excited amplitude ε_o; the tones; per block its jumps C_q ε_o S⁻¹ (GROUND × n, at e^{−iot}; `Fold::jumps`); the excitation x (the largest
/// eigenvalue of Σ T†T on the excited Q rows); the passes' ratio; the readings (`Fold::read`); the complex multiply-adds made,
/// the fold's and its readings'. And what the adjoint, the rise and the fall read again.
pub(crate) struct Folded {
    pub(crate) n: usize,
    pub(crate) a: Mat,
    pub(crate) metric: Mat,
    pub(crate) normal: Mat,
    pub(crate) generator: Mat,
    pub(crate) amplitudes: Vec<Mat>,
    pub(crate) tones: Vec<Tone>,
    pub(crate) jumps: Vec<[Mat; 3]>,
    pub(crate) excitation: f64,
    pub(crate) ratio: f64,
    pub(crate) readings: Vec<Reading>,
    pub(crate) work: f64,
    pub(crate) reading_work: f64,
    /// Per block the reading of its decomposition's reuse (rate units), and the work by phase.
    pub(crate) reuse: Vec<f64>,
    pub(crate) phases: Vec<(&'static str, f64)>,
    /// The passes solved (two or three, as the next one's reading asks).
    pub(crate) passes: usize,
    site: Site,
    laid: Vec<Laid>,
    eliminated: Vec<Eliminated>,
    stages: Vec<Stage>,
    hpp: Mat,
    h: [Mat; 3],
    m_eigen: (Vec<f64>, Mat),
}

impl Fold {
    /// The model's state of P place p (a ground state's index, an explicit one's GROUND + its own).
    pub(crate) fn place_state(&self, p: usize) -> usize {
        if p < self.ground.len() {
            self.ground[p]
        } else {
            GROUND + self.explicit[p - self.ground.len()]
        }
    }

    /// The model's state of block o's row r: an excited row's, a copy row's ground state.
    pub(crate) fn row_state(&self, o: usize, r: usize) -> usize {
        match self.blocks[o].rows[r] {
            Row::Excited(e) => GROUND + e,
            Row::Copy { state, .. } => state,
        }
    }
}

impl Fold {
    /// The readings at a fold (DESIGN §3.1; `Reading`).
    fn readings(&self, f: &Folded, pattern: f64, work: &mut f64) -> Vec<Reading> {
        let n = f.n;
        let last = f.stages.last().unwrap();
        let tri = &last.triangle;
        // The response of a block to the source columns `cols` (Schur coordinates of H, the solved rows' Schur coordinates of
        // K_eff) at an energy shifted by δ from each column's t_jj: the triangular solve of each column alone.
        let respond = |o: usize, b: &[Mat], delta: &dyn Fn(usize) -> C64, j: usize, work: &mut f64| -> Vec<C64> {
            let e = &f.eliminated[o];
            let m = e.solved;
            let mut out = vec![C64::ZERO; m];
            for (bt, &w) in b.iter().zip(&f.laid[o].shifts) {
                let c = tri.t[(j, j)] + delta(j) - C64::real(w);
                let r: Vec<C64> = (0..m).map(|k| bt[(k, j)]).collect();
                let z = e.back(c, &r, work);
                for k in 0..m {
                    out[k] += z[k];
                }
            }
            out
        };
        // The sources in the Schur coordinates (K_eff's on the left, H's on the right), per block and split.
        let tilde = &last.sources;
        // Each column's whole excited response (the denominator).
        let tu: Vec<Mat> = last.responses.iter().map(|t| mul(t, &tri.u, work)).collect();
        // A column whose response is at the rounding of the responses' own (an idle state's) reads nothing: its relative
        // error is the rounding's.
        let floor = (crate::units::rounding(n) * tu.iter().map(|t| t.frobenius()).sum::<f64>()).powi(2);
        let whole: Vec<f64> = (0..n)
            .map(|j| self.blocks.iter().zip(&tu).map(|(b, t)| (0..b.rows.len()).filter(|&i| matches!(b.rows[i], Row::Excited(_))).map(|i| t[(i, j)].norm_sqr()).sum::<f64>()).sum::<f64>())
            .map(|x| if x > floor { x } else { 0.0 })
            .collect();
        // The populations' unit (dragchk.status.md section 4): a ground set's populations are set by what its states emit
        // into the other sets, not by the whole emission -- tilt0's depumping is a hundredth of its scattering and sets
        // the F = 1 population. Per column j and landing set c (another set) its transfer W_cj = Σ_o Σ_q Σ_{b∈c}
        // |(C_q ε_o,j)_b|², each read against itself (a landing class's rate, the population it feeds); a transfer
        // below the rounding reads none.
        let others: Vec<&Vec<usize>> = self.landing.iter().filter(|set| !set.iter().any(|g| self.ground.contains(g))).collect();
        // Per landing set c (another set) the emission of block o's rows v into it, Σ_q Σ_{b∈c} |(C_q v)_b|².
        let transfer = |o: usize, v: &dyn Fn(usize) -> C64| -> Vec<f64> {
            let b = &self.blocks[o];
            others
                .iter()
                .map(|set| {
                    let mut sum = 0.0;
                    for q in 0..3 {
                        for &g in set.iter() {
                            let mut u = C64::ZERO;
                            for (i, row) in b.rows.iter().enumerate() {
                                if let Row::Excited(x) = *row {
                                    let c = self.emission[q][(g, x)];
                                    if c != C64::ZERO {
                                        u += c * v(i);
                                    }
                                }
                            }
                            sum += u.norm_sqr();
                        }
                    }
                    sum
                })
                .collect()
        };
        // Per column and landing set its transfer (every block's), a set's below the rounding none.
        let out_of: Vec<Vec<f64>> = (0..n)
            .map(|j| {
                let mut w = vec![0.0; others.len()];
                for o in 0..self.blocks.len() {
                    w.iter_mut().zip(transfer(o, &|i| tu[o][(i, j)])).for_each(|(a, b)| *a += b);
                }
                w.into_iter().map(|x| if x > floor { x } else { 0.0 }).collect()
            })
            .collect();
        *work += (3 * GROUND * self.blocks.iter().map(|b| b.rows.len()).sum::<usize>() * n) as f64;
        // An excited node's error v on block o's rows `rows` in the populations' unit at column j: the largest over the
        // landing sets of its transfer's error against the transfer.
        let populations = |o: usize, rows: &dyn Fn(usize) -> bool, v: &dyn Fn(usize) -> C64, j: usize| -> f64 {
            transfer(o, &|i| if rows(i) { v(i) } else { C64::ZERO }).iter().zip(&out_of[j]).filter(|(_, w)| **w > 0.0).map(|(d, w)| d / w).fold(0.0, f64::max)
        };
        // A row's error in its reading's units: an excited row's relative to the column's whole excited response (a rate's),
        // the larger of that and its populations' unit (above); a copy's absolute (a population's, per unit P amplitude:
        // the Schur columns are of unit norm).
        let per = |nd: Node, j: usize| match nd {
            Node::Excited(_) => whole[j],
            Node::Copy { .. } => 1.0,
        };
        let node = |b: &Block, i: usize| match b.rows[i] {
            Row::Excited(x) => Node::Excited(self.excited_level[x]),
            Row::Copy { state, component } => Node::Copy { level: self.ground_level[state], component },
        };
        let mut out: Vec<Reading> = vec![];
        let put = |o: usize, nd: Node, kind: usize, value: f64, out: &mut Vec<Reading>| {
            let r = match out.iter_mut().find(|r| r.block == o && r.node == nd) {
                Some(r) => r,
                None => {
                    out.push(Reading { block: o, node: nd, next: 0.0, pattern: 0.0, sideband: 0.0, order: 0.0, truncation: 0.0, mixing: 0.0, centring: 0.0, omission: 0.0 });
                    out.last_mut().unwrap()
                }
            };
            let slot = match kind {
                0 => &mut r.next,
                1 => &mut r.pattern,
                2 => &mut r.sideband,
                3 => &mut r.order,
                4 => &mut r.truncation,
                5 => &mut r.mixing,
                6 => &mut r.centring,
                _ => &mut r.omission,
            };
            *slot = slot.max(value);
        };
        // The error vector of a block's solved rows (in K_eff's Schur coordinates) back on its rows, split by node.
        let split = |o: usize, v: &[C64], j: usize, kind: usize, scale: f64, out: &mut Vec<Reading>, work: &mut f64| {
            let e = &f.eliminated[o];
            let b = &self.blocks[o];
            let back = e.from(&Mat::from_fn(v.len(), 1, |i, _| v[i]), work).data;
            let mut parts: Vec<(Node, f64)> = vec![];
            for (i, x) in back.iter().enumerate() {
                let nd = node(b, i);
                match parts.iter_mut().find(|p| p.0 == nd) {
                    Some(p) => p.1 += x.norm_sqr(),
                    None => parts.push((nd, x.norm_sqr())),
                }
            }
            for (nd, x) in parts {
                if per(nd, j) > 0.0 {
                    let pop = if matches!(nd, Node::Excited(_)) { populations(o, &|i| i < back.len() && node(b, i) == nd, &|i| back.get(i).copied().unwrap_or(C64::ZERO), j) } else { 0.0 };
                    put(o, nd, kind, scale * (x / per(nd, j)).max(pop), out);
                }
            }
        };
        // The next pass: δH = H⁽²⁾ − H⁽¹⁾ (the Bloch generator of this pass's responses less the one they were solved at),
        // δT solving the same equation with the source −T δH, each split its own.
        let mut dh = f.h[2].clone();
        dh.add_scaled(-C64::ONE, &f.h[1]);
        let dht = mul(&mul_adj(&tri.u, &dh, work), &tri.u, work);
        for (o, e) in f.eliminated.iter().enumerate() {
            let mut dz = Mat::zeros(e.solved, n);
            for (k, zt) in last.splits[o].iter().enumerate() {
                let src = mul(zt, &dht, work).scaled(-C64::ONE);
                plus(&mut dz, &e.sylvester(&tri.t, f.laid[o].shifts[k], &src, work));
            }
            for j in 0..n {
                let col: Vec<C64> = (0..e.solved).map(|i| dz[(i, j)]).collect();
                split(o, &col, j, 0, 1.0, &mut out, work);
            }
        }
        // An excited node left out of its block, read in both of the readings' units, the larger: its response's share of the
        // column's whole excited response (the rates'), and the potential it carries, ΔV_j = 2|Re(U†Ĥ_P,node T_node U)_jj|
        // -- the column's light shift through the node --, against T's share of k_BT (a conservative change moves a kinetic
        // energy by at most itself: tabulate.rs # Refinement): ρ with rate(ρ) = ΔV/(2k_BT), so that rate(ρ) within the
        // populations' share is ΔV/k_BT within T's, twice it (elim::share). A far node's rate is nothing, its shift the trap.
        for (o, b) in self.blocks.iter().enumerate() {
            let ue = mul_adj(&tri.u, &f.laid[o].emit, work);
            for j in (0..n).filter(|&j| whole[j] > 0.0) {
                let mut parts: Vec<(Node, f64, C64)> = vec![];
                for i in (0..b.rows.len()).filter(|&i| matches!(b.rows[i], Row::Excited(_))) {
                    let nd = node(b, i);
                    let shift = ue[(j, i)] * tu[o][(i, j)];
                    match parts.iter_mut().find(|p| p.0 == nd) {
                        Some(p) => {
                            p.1 += tu[o][(i, j)].norm_sqr();
                            p.2 += shift;
                        }
                        None => parts.push((nd, tu[o][(i, j)].norm_sqr(), shift)),
                    }
                }
                for (nd, x, shift) in parts {
                    let potential = ((1.0 + (2.0 * shift.re).abs() / (2.0 * self.thermal)).sqrt() - 1.0).powi(2);
                    let pop = populations(o, &|i| matches!(b.rows[i], Row::Excited(_)) && node(b, i) == nd, &|i| tu[o][(i, j)], j);
                    put(o, nd, 7, (x / whole[j]).max(pop).max(potential), &mut out);
                }
            }
        }
        // Every node's centring on its own response (module documentation, # Blocks): taken at z₀ instead of each column's
        // own energy t_jj, its response moves by exactly (z₀ − t_jj) R_N(z₀) T_N,j, R_N(z₀) = (z₀ − Ĥ_NN)⁻¹ on its rows -- an
        // excited node's relative to the column's whole excited response, a copy's absolute (`per`); the order's reading
        // where it is centred. (A copy's effect through the solved rows is read below, the larger kept.)
        for (o, b) in self.blocks.iter().enumerate() {
            let l = &f.laid[o];
            let mut nodes: Vec<Node> = (0..b.rows.len()).map(|i| node(b, i)).collect();
            nodes.dedup();
            let mut seen: Vec<Node> = vec![];
            for nd in nodes {
                if seen.contains(&nd) {
                    continue;
                }
                seen.push(nd);
                let rows: Vec<usize> = (0..b.rows.len()).filter(|&i| node(b, i) == nd).collect();
                let centred = rows.iter().any(|&i| i >= b.solved);
                let Some(r0) = inverse(&Mat::from_fn(rows.len(), rows.len(), |a, c| if a == c { C64::real(l.z0) } else { C64::ZERO } - l.inner[(rows[a], rows[c])]), work) else { continue };
                for j in (0..n).filter(|&j| whole[j] > 0.0) {
                    let col = Mat::from_fn(rows.len(), 1, |a, _| tu[o][(rows[a], j)]);
                    let moved = mul(&r0, &col, work);
                    let shift = (C64::real(l.z0) - tri.t[(j, j)]).norm_sqr();
                    let mut size = shift * moved.data.iter().map(|x| x.norm_sqr()).sum::<f64>() / per(nd, j);
                    if matches!(nd, Node::Excited(_)) {
                        size = size.max(shift * populations(o, &|i| rows.contains(&i), &|i| rows.iter().position(|&r| r == i).map_or(C64::ZERO, |a| moved.data[a]), j));
                    }
                    put(o, nd, 6, size, &mut out);
                    if centred {
                        put(o, nd, 3, size, &mut out);
                    }
                }
            }
        }
        // The light pattern's offsets ±pattern on each column's own response, the larger.
        if pattern > 0.0 {
            for o in 0..self.blocks.len() {
                for j in 0..n {
                    let base = respond(o, &tilde[o], &|_| C64::ZERO, j, work);
                    let mut best: (f64, Vec<C64>) = (0.0, vec![]);
                    for sign in [1.0, -1.0] {
                        let shifted = respond(o, &tilde[o], &|_| C64::real(sign * pattern), j, work);
                        let d: Vec<C64> = shifted.iter().zip(&base).map(|(a, b)| *a - *b).collect();
                        let size: f64 = d.iter().map(|x| x.norm_sqr()).sum();
                        if size >= best.0 {
                            best = (size, d);
                        }
                    }
                    split(o, &best.1, j, 1, 1.0, &mut out, work);
                }
            }
        }
        // The tones' sidebands: each tone's entry (q, p) in H's Schur basis, δ = t_pp − Ω − t_qq, its mixing w, and the error
        // √w [T(t_qq) − T(t_pp − Ω)] u_q on every block.
        for tone in &f.tones {
            let at = mul(&mul_adj(&tri.u, &tone.a, work), &tri.u, work);
            for p in 0..n {
                for q in 0..n {
                    let a = at[(q, p)];
                    if a == C64::ZERO {
                        continue;
                    }
                    let delta = tri.t[(p, p)] - C64::real(tone.value) - tri.t[(q, q)];
                    let w = 0.5 * (1.0 - delta.abs() / (delta.norm_sqr() + 4.0 * a.norm_sqr()).sqrt());
                    for o in 0..self.blocks.len() {
                        let base = respond(o, &tilde[o], &|_| C64::ZERO, q, work);
                        let shifted = respond(o, &tilde[o], &|_| delta, q, work);
                        let d: Vec<C64> = shifted.iter().zip(&base).map(|(x, y)| *x - *y).collect();
                        split(o, &d, q, 2, w, &mut out, work);
                    }
                }
            }
        }
        // The bare excited block's eigendecomposition, for the reabsorption's resolvent (no lattice node holds it).
        let mut bare = crate::linalg::Eigen::default();
        let full = {
            let (mut g, mut e) = (Mat::zeros(GROUND, GROUND), f.site.excited.clone());
            self.forms(&self.others, &f.site.a, &mut g, &mut e, work);
            e
        };
        let bare = bare.decompose(&full).then_some(bare);
        if let Some(b) = &bare {
            *work += b.work;
        }
        // A node's dressing δK_EE = V δR V† of a block's explicit places (P; V its coupling to them, δR its resolvent's
        // change) read at column j as any node's is, in both units, the larger: the potential it carries -- the column's
        // light shift through it, ΔV_j = 2|Re(u_E,j† δK_EE u_E,j)|, against T's share of k_BT, as an excited node left out
        // -- and the rate's: what it moves of the column's excited amplitudes in P, P's own response to it -- the reduced
        // resolvent (t_jj − t)⁻¹ in H's Schur coordinates, the column's own direction dropped (its energy's change is the
        // potential) -- against the column's whole excited response, P's explicit amplitudes with the Q rows'.
        let explicit_dressing = |v: &Mat, dr: &Mat, places: &[usize], j: usize, work: &mut f64| -> f64 {
            let ng = self.ground.len();
            let ue = Mat::from_fn(places.len(), 1, |i, _| tri.u[(places[i], j)]);
            let dk = mul(v, &mul(dr, &mul_adj(v, &ue, work), work), work);
            let shift: C64 = (0..places.len()).map(|i| ue[(i, 0)].conj() * dk[(i, 0)]).sum();
            let potential = ((1.0 + (2.0 * shift.re).abs() / (2.0 * self.thermal)).sqrt() - 1.0).powi(2);
            let d: Vec<C64> = (0..n).map(|k| (0..places.len()).map(|i| tri.u[(places[i], k)].conj() * dk[(i, 0)]).sum()).collect();
            let mut y = vec![C64::ZERO; n];
            for k in (0..n).rev().filter(|&k| k != j) {
                let mut s = d[k];
                for l in k + 1..n {
                    s += tri.t[(k, l)] * y[l];
                }
                y[k] = s / (tri.t[(j, j)] - tri.t[(k, k)]);
            }
            *work += (n * n * (places.len() + 2)) as f64;
            let moved: f64 = (ng..n).map(|p| (0..n).map(|k| tri.u[(p, k)] * y[k]).sum::<C64>().norm_sqr()).sum();
            let response = whole[j] + (ng..n).map(|p| tri.u[(p, j)].norm_sqr()).sum::<f64>();
            potential.max(if response > 0.0 { moved / response } else { 0.0 })
        };
        // The copies: the evaluation order's error by the block perturbation δK on the solved rows, R(z_j) δK T_j (one
        // function for centring, its prospect and omission), and on the block's explicit places (`explicit_dressing`); the
        // reabsorption a third order would add; the mixing.
        for (o, b) in self.blocks.iter().enumerate() {
            let l = &f.laid[o];
            let e = &f.eliminated[o];
            let groups: Vec<usize> = {
                let mut g: Vec<usize> = b.rows.iter().filter_map(|r| if let Row::Copy { component, .. } = r { Some(*component) } else { None }).collect();
                g.extend(b.omitted.iter().map(|x| x.1));
                g.sort();
                g.dedup();
                g
            };
            let exc: Vec<usize> = (0..b.rows.len()).filter(|&i| matches!(b.rows[i], Row::Excited(_))).collect();
            for g in groups {
                let level_of = |s: usize| self.ground_level[s];
                let copies: Vec<usize> = (0..GROUND)
                    .filter(|&s| b.rows.iter().any(|r| *r == Row::Copy { state: s, component: g }) || b.omitted.contains(&(s, g)))
                    .collect();
                let mut levels: Vec<u8> = copies.iter().map(|&s| level_of(s)).collect();
                levels.sort();
                levels.dedup();
                for level in levels {
                    let states: Vec<usize> = copies.iter().copied().filter(|&s| level_of(s) == level).collect();
                    let omega = b.value - self.components[g].nu;
                    let status = if b.omitted.iter().any(|x| x.1 == g && states.contains(&x.0)) {
                        Status::Omitted
                    } else if b.rows[b.solved..].iter().any(|r| matches!(r, Row::Copy { state, component } if *component == g && states.contains(state))) {
                        Status::Centred
                    } else {
                        Status::Decomposed
                    };
                    // Its coupling to the excited Q rows and its own matrix: V (excited × copies), H_c (copies, less ω).
                    let v = Mat::from_fn(exc.len(), states.len(), |i, k| {
                        let Row::Excited(x) = b.rows[exc[i]] else { unreachable!() };
                        f.site.drives[g][(x, states[k])]
                    });
                    let hc = Mat::from_fn(states.len(), states.len(), |i, k| f.site.ground[(states[i], states[k])] - if i == k { C64::real(omega) } else { C64::ZERO });
                    let r_at = |z: C64, work: &mut f64| inverse(&Mat::from_fn(hc.rows, hc.cols, |i, k| if i == k { z } else { C64::ZERO } - hc[(i, k)]), work);
                    let r0 = r_at(C64::real(l.z0), work);
                    let tu_exc = Mat::from_fn(exc.len(), n, |i, j| tu[o][(exc[i], j)]);
                    // The block's explicit places (P) the copy dresses too, Ĥ_PC T_C on their rows: their light shift
                    // through it (tilt0's tweezer copies of F = 2 shift its explicit F' = 3 m′ = ±1 by 95 rad/µs) -- V on them.
                    let places = &b.explicit;
                    let v_e = Mat::from_fn(places.len(), states.len(), |i, k| f.site.drives[g][(self.explicit[places[i] - self.ground.len()], states[k])]);
                    for j in 0..n {
                        let z = tri.t[(j, j)];
                        let Some(rz) = r_at(z, work) else { continue };
                        let Some(r0) = r0.clone() else { continue };
                        // δK = Σ(z) − Σ(z₀) (centring) and Σ(z) (omission), on the excited rows V(R(z) − R(z₀))V† T_j and V R(z) V† T_j:
                        // their errors R(z_j) δK T_j on the solved rows (in K_eff's Schur coordinates, solved at t_jj).
                        let col = Mat::from_fn(exc.len(), 1, |i, _| tu_exc[(i, j)]);
                        let mut centring = rz.clone();
                        centring.add_scaled(-C64::ONE, &r0);
                        let mut sizes = [0.0; 2];
                        for (k, dr) in [&centring, &rz].into_iter().enumerate() {
                            let dk = mul(&v, &mul(dr, &mul_adj(&v, &col, work), work), work);
                            let src = Mat::from_fn(e.solved, 1, |i, _| if i < exc.len() { dk[(i, 0)] } else { C64::ZERO });
                            let st = e.to(&src, work);
                            let zz = e.back(z, &st.data, work);
                            let back = e.from(&Mat::from_fn(zz.len(), 1, |i, _| zz[i]), work).data;
                            let rate_unit: f64 = back.iter().enumerate().map(|(i, x)| match per(node(b, i), j) { d if d > 0.0 => x.norm_sqr() / d, _ => 0.0 }).sum();
                            sizes[k] = rate_unit.max(populations(o, &|i| matches!(b.rows[i], Row::Excited(_)), &|i| back.get(i).copied().unwrap_or(C64::ZERO), j));
                            if !places.is_empty() {
                                sizes[k] = sizes[k].max(explicit_dressing(&v_e, dr, places, j, work));
                            }
                        }
                        let nd = Node::Copy { level, component: g };
                        put(o, nd, 6, sizes[0], &mut out);
                        put(o, nd, 7, sizes[1], &mut out);
                        match status {
                            Status::Centred => put(o, nd, 3, sizes[0], &mut out),
                            Status::Omitted => put(o, nd, 3, sizes[1], &mut out),
                            _ => {}
                        }
                        // The copy's amplitude (its rows of T, or R(z)V†T on an omitted one) and its mixing.
                        let amp: Vec<C64> = if status == Status::Omitted {
                            let a = mul(&rz, &mul_adj(&v, &col, work), work);
                            (0..states.len()).map(|i| a[(i, 0)]).collect()
                        } else {
                            states.iter().map(|&s| (0..b.rows.len()).find(|&i| b.rows[i] == Row::Copy { state: s, component: g }).map_or(C64::ZERO, |i| tu[o][(i, j)])).collect()
                        };
                        let c2: f64 = amp.iter().map(|x| x.norm_sqr()).sum();
                        put(o, Node::Copy { level, component: g }, 5, 0.5 * (1.0 - 1.0 / (1.0 + 4.0 * c2).sqrt()), &mut out);
                        // The reabsorption: from the copy at ω, any other component h to the excited states at ω + ν_h, on
                        // the bare excited block (no lattice node holds it): ‖(z + ω + ν_h − 𝒦)⁻¹ V_h c‖².
                        if status != Status::Omitted {
                            for (h, comp) in self.components.iter().enumerate().filter(|&(h, _)| h != g) {
                                let vh = Mat::from_fn(EXCITED, states.len(), |x, k| f.site.drives[h][(x, states[k])]);
                                if vh.data.iter().all(|x| *x == C64::ZERO) {
                                    continue;
                                }
                                let Some(bare) = &bare else { continue };
                                let cvec = Mat::from_fn(states.len(), 1, |i, _| amp[i]);
                                let src = mul(&bare.inverse, &mul(&vh, &cvec, work), work);
                                let at = z + C64::real(omega + comp.nu);
                                let third = Mat::from_fn(EXCITED, 1, |x, _| src[(x, 0)] / (at - bare.values[x]));
                                let size: f64 = mul(&bare.vectors, &third, work).data.iter().map(|x| x.norm_sqr()).sum();
                                if whole[j] > 0.0 {
                                    put(o, Node::Copy { level, component: g }, 4, size / whole[j], &mut out);
                                }
                            }
                        }
                    }
                }
            }
        }
        out
    }

    /// The force −∇ Re[tr(Ḡ_s†G) + Σ_k tr(Ḡ_k†G_k)] at the fold's point (module documentation, # The force): `ground` the
    /// static generator's cotangent (for a state's density ρ, ρ: −∇⟨G_h⟩), `tones` the tones' (in their order, any of them
    /// left empty). By the adjoint through the compression, the passes and the lattice, contracted with the fields'
    /// analytic gradients; the work of it returned beside.
    pub(crate) fn force(&self, f: &Folded, ground: &Mat, tones: &[Mat]) -> ([f64; 3], f64) {
        let mut work = 0.0;
        let w = &mut work;
        let n = f.n;
        let ng = self.ground.len();
        let (xi, v) = &f.m_eigen;
        let s: Vec<f64> = xi.iter().map(|&x| (1.0 + x).sqrt()).collect();
        // G = S⁻¹AS⁻¹ for the static part and each tone: Ā = S⁻¹ḠS⁻¹, S̄⁻¹ = ḠS⁻¹A† + A†S⁻¹Ḡ.
        let mut bar_normal = Mat::zeros(n, n);
        let mut bar_tones: Vec<Mat> = vec![];
        let mut through = |g: &Mat, a: &Mat, w: &mut f64| -> Mat {
            plus(&mut bar_normal, &mul_by_adj(&mul(g, &f.normal, w), a, w));
            plus(&mut bar_normal, &mul_adj(a, &mul(&f.normal, g, w), w));
            mul(&mul(&f.normal, g, w), &f.normal, w)
        };
        let bar_a = through(ground, &f.a, w);
        for (t, g) in f.tones.iter().zip(tones) {
            if g.rows == n {
                bar_tones.push(through(g, &t.a, w));
            } else {
                bar_tones.push(Mat::zeros(n, n));
            }
        }
        while bar_tones.len() < f.tones.len() {
            bar_tones.push(Mat::zeros(n, n));
        }
        // S⁻¹ = (1 + X)^{−1/2}: M̄ = V(D∘(V†S̄V))V†, D_ij = −1/(s_i s_j (s_i + s_j)).
        let vs = mul(&mul_adj(v, &bar_normal, w), v, w);
        let dm = Mat::from_fn(n, n, |i, j| vs[(i, j)] * (-1.0 / (s[i] * s[j] * (s[i] + s[j]))));
        let bar_m = mul_by_adj(&mul(v, &dm, w), v, w);
        // The lattice's cotangents: Ĥ_PP, and per block its frame matrix, its sources per split, its emission; per block the
        // response's (all Q rows) of the last pass.
        let mut bar_hpp = bar_a.clone();
        let mut bar_inner: Vec<Mat> = f.laid.iter().map(|l| Mat::zeros(l.inner.rows, l.inner.cols)).collect();
        let mut bar_absorb: Vec<Vec<Mat>> = f.laid.iter().map(|l| l.absorb.iter().map(|a| Mat::zeros(a.rows, a.cols)).collect()).collect();
        let mut bar_emit: Vec<Mat> = f.laid.iter().map(|l| Mat::zeros(l.emit.rows, l.emit.cols)).collect();
        let mut bar_ground = Mat::zeros(GROUND, GROUND);
        let mut bar_excited = Mat::zeros(EXCITED, EXCITED);
        let mut bar_drives: Vec<Mat> = self.components.iter().map(|_| Mat::zeros(EXCITED, GROUND)).collect();
        let last = f.stages.last().unwrap();
        let mut bar_t: Vec<Mat> = vec![];
        for (o, (l, t)) in f.laid.iter().zip(&last.responses).enumerate() {
            let total = l.absorb.iter().fold(Mat::zeros(t.rows, n), |mut x, y| {
                plus(&mut x, y);
                x
            });
            // A ∋ emit T + T†absorb + T†inner T; M ∋ T†T.
            let mut bt = mul_adj(&l.emit, &bar_a, w);
            plus(&mut bt, &mul_by_adj(&total, &bar_a, w));
            plus(&mut bt, &mul_by_adj(&mul(&l.inner, t, w), &bar_a, w));
            plus(&mut bt, &mul(&mul_adj(&l.inner, t, w), &bar_a, w));
            let mut sym = bar_m.clone();
            plus(&mut sym, &bar_m.adjoint());
            plus(&mut bt, &mul(t, &sym, w));
            plus(&mut bar_emit[o], &mul_by_adj(&bar_a, t, w));
            let ta = mul(t, &bar_a, w);
            for a in bar_absorb[o].iter_mut() {
                plus(a, &ta);
            }
            plus(&mut bar_inner[o], &mul_by_adj(&ta, t, w));
            bar_t.push(bt);
        }
        // The tones: frame mismatches (fields alone), kept copies (Γ_t†V_g†ε and its partner), the decay's beats.
        let mut bar_eps: Vec<Mat> = f.amplitudes.iter().map(|e| Mat::zeros(e.rows, e.cols)).collect();
        for (tone, bar) in f.tones.iter().zip(&bar_tones) {
            for p in 0..ng {
                for q in (0..ng).filter(|&q| self.frames[q] != self.frames[p] && f.site.ground[(self.ground[p], self.ground[q])] != C64::ZERO) {
                    if self.frames[p].plus(-1, &self.frames[q]) == tone.beat {
                        bar_ground[(self.ground[p], self.ground[q])] += bar[(p, q)];
                    }
                }
            }
            for p in ng..n {
                for q in (ng..n).filter(|&q| self.frames[q] != self.frames[p]) {
                    if self.frames[p].plus(-1, &self.frames[q]) == tone.beat {
                        bar_excited[(self.explicit[p - ng], self.explicit[q - ng])] += bar[(p, q)];
                    }
                }
            }
            for (o, b) in self.blocks.iter().enumerate() {
                let rows = b.excited(self);
                let e = &f.amplitudes[o];
                for &(t, g, ref beat) in &b.kept {
                    let Some(pt) = self.ground.iter().position(|&s| s == t) else { continue };
                    let vcol = Mat::from_fn(rows.len(), 1, |i, _| f.site.drives[g][(rows[i], t)]);
                    if *beat == tone.beat {
                        // A[t, :] = v†ε: ε̄ += v Ā[t,:], v̄ += ε Ā[t,:]†.
                        let row = Mat::from_fn(1, n, |_, j| bar[(pt, j)]);
                        plus(&mut bar_eps[o], &mul(&vcol, &row, w));
                        let vb = mul_by_adj(e, &row, w);
                        for i in 0..rows.len() {
                            bar_drives[g][(rows[i], t)] += vb[(i, 0)];
                        }
                    }
                    if beat.negated() == tone.beat {
                        // A[:, t] = ε†v: ε̄ += v Ā[:,t]†, v̄ += ε Ā[:,t].
                        let col = Mat::from_fn(n, 1, |j, _| bar[(j, pt)]);
                        plus(&mut bar_eps[o], &mul_by_adj(&vcol, &col, w));
                        let vb = mul(e, &col, w);
                        for i in 0..rows.len() {
                            bar_drives[g][(rows[i], t)] += vb[(i, 0)];
                        }
                    }
                }
            }
            for (i, j) in self.related() {
                if self.blocks[i].offset.plus(-1, &self.blocks[j].offset) != tone.beat {
                    continue;
                }
                // A = ε_i† 𝒦 ε_j − D∘(ε_i† E ε_j): with C̄ = Ā (on 𝒦) and Ȳ = D∘Ā (on the overlap, D real):
                // ε̄_i += 𝒦 ε_j Ā† − E ε_j Ȳ†, ε̄_j += 𝒦† ε_i Ā − E† ε_i Ȳ, 𝒦̄ += ε_i Ā ε_j†.
                let (k, e, offsets) = self.cross(&f.site, i, j);
                let ybar = Mat { rows: n, cols: n, data: bar.data.iter().zip(&offsets.data).map(|(a, d)| *a * d.re).collect() };
                plus(&mut bar_eps[i], &mul_by_adj(&mul(&k, &f.amplitudes[j], w), bar, w));
                plus(&mut bar_eps[j], &mul(&mul_adj(&k, &f.amplitudes[i], w), bar, w));
                bar_eps[i].add_scaled(-C64::ONE, &mul_by_adj(&mul(&e, &f.amplitudes[j], w), &ybar, w));
                bar_eps[j].add_scaled(-C64::ONE, &mul(&mul_adj(&e, &f.amplitudes[i], w), &ybar, w));
                let cb = mul_by_adj(&mul(&f.amplitudes[i], bar, w), &f.amplitudes[j], w);
                let (ri, rj) = (self.blocks[i].excited(self), self.blocks[j].excited(self));
                for (a, &x) in ri.iter().enumerate() {
                    for (b, &y) in rj.iter().enumerate() {
                        bar_excited[(x, y)] += cb[(a, b)];
                    }
                }
            }
        }
        for (o, b) in self.blocks.iter().enumerate() {
            let exc: Vec<usize> = (0..b.rows.len()).filter(|&i| matches!(b.rows[i], Row::Excited(_))).collect();
            for (k, &i) in exc.iter().enumerate() {
                for j in 0..n {
                    bar_t[o][(i, j)] += bar_eps[o][(b.explicit.len() + k, j)];
                }
            }
        }
        // Back through the passes, the last first: the centred rows, each split's Sylvester equation (its adjoint), the
        // Bloch generator (full, or its diagonal) to the previous pass's responses.
        let mut bar_k: Vec<Mat> = f.eliminated.iter().map(|e| Mat::zeros(e.solved, e.solved)).collect();
        let mut bar_r0: Vec<Mat> = f.eliminated.iter().map(|e| Mat::zeros(e.resolvent.rows, e.resolvent.cols)).collect();
        let mut bar_source: Vec<Vec<Mat>> = f.eliminated.iter().zip(&f.laid).map(|(e, l)| l.shifts.iter().map(|_| Mat::zeros(e.solved, n)).collect()).collect();
        // A beam's dressed source (Ĥ_SC R₀ B_C), apart from its rows' own (those reach the beam through its ranks).
        let mut bar_dressed: Vec<Vec<Mat>> = bar_source.clone();
        let mut kappa: Vec<Vec<C64>> = self.components.iter().map(|c| vec![C64::ZERO; c.beams.len()]).collect();
        for k in (0..f.stages.len()).rev() {
            let stage = &f.stages[k];
            let mut bar_h = Mat::zeros(n, n);
            for (o, (l, e)) in f.laid.iter().zip(&f.eliminated).enumerate() {
                let m = l.inner.rows;
                let (sr, cr): (Vec<usize>, Vec<usize>) = ((0..e.solved).collect(), (e.solved..m).collect());
                let all: Vec<usize> = (0..n).collect();
                // T_C = R₀(Ĥ_CS T_S + B_C).
                let bt = &bar_t[o];
                let bar_ts_full = bt.block(&sr, &all);
                let mut bar_ts = bar_ts_full.clone();
                if !cr.is_empty() {
                    let btc = bt.block(&cr, &all);
                    let ts = stage.responses[o].block(&sr, &all);
                    let inner_cs = l.inner.block(&cr, &sr);
                    let mut arg = mul(&inner_cs, &ts, w);
                    for a in &l.absorb {
                        plus(&mut arg, &a.block(&cr, &all));
                    }
                    let r0b = mul_adj(&e.resolvent, &btc, w);
                    plus(&mut bar_ts, &mul_adj(&inner_cs, &r0b, w));
                    plus(&mut bar_r0[o], &mul_by_adj(&btc, &arg, w));
                    let ics = mul_by_adj(&r0b, &ts, w);
                    for (a, &ci) in cr.iter().enumerate() {
                        for (b2, &si) in sr.iter().enumerate() {
                            bar_inner[o][(ci, si)] += ics[(a, b2)];
                        }
                        for ba in bar_absorb[o].iter_mut() {
                            for j in 0..n {
                                ba[(ci, j)] += r0b[(a, j)];
                            }
                        }
                    }
                }
                // Each split, in the Schur coordinates: Λ̃ = sylvester_adjoint(V† T̄ U), Λ = V Λ̃ U†; K̄ += V(Σ Λ̃Z†)V†,
                // H̄ −= U(Σ Z†Λ̃)U†; a beam's source B̄ onto its amplitude through its ranks, κ̄ += Σ_k γ_jk ⟨Λ̃, V†S_k U⟩,
                // the static one's back on its rows.
                let tri = &stage.triangle;
                let g = mul(&e.to(&bar_ts, w), &tri.u, w);
                let mut kz = Mat::zeros(e.solved, e.solved);
                let mut hz = Mat::zeros(n, n);
                // A pass at a diagonal H passes back its diagonal alone (`k` ≤ 1); K̄ is taken part by part (its couplings
                // across parts are left out, as the decomposition leaves them).
                let whole_h = k >= 2;
                let ranks_u: Vec<Vec<Mat>> = e.ranks.iter().map(|(_, per)| per.iter().map(|rk| mul(rk, &tri.u, w)).collect()).collect();
                for (sp, (z, &shift)) in stage.splits[o].iter().zip(&l.shifts).enumerate() {
                    let lz = e.adjoint(&tri.t, shift, &g, w);
                    for (idx, at, _) in &e.factored.parts {
                        for a in *at..at + idx.len() {
                            for b in *at..at + idx.len() {
                                kz[(a, b)] += (0..n).map(|j| lz[(a, j)] * z[(b, j)].conj()).sum::<C64>();
                            }
                        }
                        *w += (idx.len() * idx.len() * n) as f64;
                    }
                    // The exact copies' couplings with the excited rows (solved, not decomposed).
                    let ne = e.solved - e.exact;
                    for a in 0..e.solved {
                        for b in 0..e.solved {
                            if (a < ne) != (b < ne) {
                                kz[(a, b)] += (0..n).map(|j| lz[(a, j)] * z[(b, j)].conj()).sum::<C64>();
                            }
                        }
                    }
                    *w += (2 * ne * e.exact * n) as f64;
                    if whole_h {
                        plus(&mut hz, &mul_adj(z, &lz, w));
                    } else {
                        for i in 0..n {
                            hz[(i, i)] += (0..z.rows).map(|a| z[(a, i)].conj() * lz[(a, i)]).sum::<C64>();
                        }
                        *w += (z.rows * n) as f64;
                    }
                    match l.beams[sp] {
                        None => {
                            if e.to_static.is_some() {
                                plus(&mut bar_source[o][sp], &mul_by_adj(&e.from(&lz, w), &tri.u, w));
                            }
                        }
                        Some((c, j)) => {
                            if e.to_dressed[sp].is_some() {
                                plus(&mut bar_dressed[o][sp], &mul_by_adj(&e.from(&lz, w), &tri.u, w));
                            }
                            let ci = e.ranks.iter().position(|x| x.0 == c).unwrap();
                            for (k, ru) in ranks_u[ci].iter().enumerate() {
                                let inner: C64 = (0..lz.data.len()).map(|i| lz.data[i].conj() * ru.data[i]).sum();
                                kappa[c][j] += inner * self.components[c].gamma[j][k];
                                *w += lz.data.len() as f64;
                            }
                        }
                    }
                }
                let kb = e.from(&e.from(&kz.adjoint(), w).adjoint(), w);
                plus(&mut bar_k[o], &kb);
                bar_h.add_scaled(-C64::ONE, &mul_by_adj(&mul(&tri.u, &hz, w), &tri.u, w));
            }
            if k == 0 {
                // H⁽⁻¹⁾ = diag(Ĥ_PP).
                for i in 0..n {
                    bar_hpp[(i, i)] += bar_h[(i, i)];
                }
                break;
            }
            // The previous pass's Bloch generator: full for the last pass, its diagonal before.
            if k == 1 {
                bar_h = Mat::from_fn(n, n, |i, j| if i == j { bar_h[(i, i)] } else { C64::ZERO });
            }
            plus(&mut bar_hpp, &bar_h);
            let prev = &f.stages[k - 1];
            bar_t = vec![];
            for (o, l) in f.laid.iter().enumerate() {
                plus(&mut bar_emit[o], &mul_by_adj(&bar_h, &prev.responses[o], w));
                bar_t.push(mul_adj(&l.emit, &bar_h, w));
            }
        }
        // Each block's evaluation order: sources_eff = absorb_S + Ĥ_SC R₀ absorb_C (every split), K_eff = Ĥ_SS + Ĥ_SC R₀ Ĥ_CS,
        // R₀ = (z₀ − Ĥ_CC)⁻¹.
        let mut bar_z0: Vec<f64> = vec![0.0; self.blocks.len()];
        for (o, (l, e)) in f.laid.iter().zip(&f.eliminated).enumerate() {
            let m = l.inner.rows;
            let (sr, cr): (Vec<usize>, Vec<usize>) = ((0..e.solved).collect(), (e.solved..m).collect());
            let all: Vec<usize> = (0..n).collect();
            for (sp, bs) in bar_source[o].iter().enumerate() {
                for (a, &si) in sr.iter().enumerate() {
                    for j in 0..n {
                        bar_absorb[o][sp][(si, j)] += bs[(a, j)];
                    }
                }
            }
            for (a, &si) in sr.iter().enumerate() {
                for (b2, &sj) in sr.iter().enumerate() {
                    bar_inner[o][(si, sj)] += bar_k[o][(a, b2)];
                }
            }
            if !cr.is_empty() {
                let sc = l.inner.block(&sr, &cr);
                let cs = l.inner.block(&cr, &sr);
                let r0 = &e.resolvent;
                // source_sp ∋ Ĥ_SC R₀ B_C,sp (the static split's and every beam's dressing); K ∋ Ĥ_SC R₀ Ĥ_CS.
                let mut bar_sc = mul_by_adj(&bar_k[o], &mul(r0, &cs, w), w);
                let mut br0 = mul(&mul_adj(&sc, &bar_k[o], w), &cs.adjoint(), w);
                let scr = mul(&sc, r0, w);
                for sp in 0..l.absorb.len() {
                    let bs = if sp == 0 { &bar_source[o][0] } else { &bar_dressed[o][sp] };
                    if bs.data.iter().all(|z| *z == C64::ZERO) {
                        continue;
                    }
                    let ac = l.absorb[sp].block(&cr, &all);
                    plus(&mut bar_sc, &mul_by_adj(bs, &mul(r0, &ac, w), w));
                    plus(&mut br0, &mul(&mul_adj(&sc, bs, w), &ac.adjoint(), w));
                    let bac = mul_adj(&scr, bs, w);
                    for (a, &ci) in cr.iter().enumerate() {
                        for j in 0..n {
                            bar_absorb[o][sp][(ci, j)] += bac[(a, j)];
                        }
                    }
                }
                plus(&mut bar_r0[o], &br0);
                let bcs = mul_adj(&scr, &bar_k[o], w);
                for (a, &ci) in cr.iter().enumerate() {
                    for (b2, &si) in sr.iter().enumerate() {
                        bar_inner[o][(ci, si)] += bcs[(a, b2)];
                        bar_inner[o][(si, ci)] += bar_sc[(b2, a)];
                    }
                }
                // R₀ = (z₀ − Ĥ_CC)⁻¹: dR₀ = R₀ dĤ_CC R₀ − dz₀ R₀².
                let bcc = mul(&mul_adj(r0, &bar_r0[o], w), &r0.adjoint(), w);
                for (a, &ci) in cr.iter().enumerate() {
                    for (b2, &cj) in cr.iter().enumerate() {
                        bar_inner[o][(ci, cj)] += bcc[(a, b2)];
                    }
                }
                let r2 = mul(r0, r0, w);
                bar_z0[o] = -(0..r2.data.len()).map(|i| (bar_r0[o].data[i].conj() * r2.data[i]).re).sum::<f64>();
            }
        }
        // To the fields: Ĥ_PP, each block's frame matrix, sources and emission, and z₀, onto H_GG, 𝒦, V_c and the beams.
        for p in 0..ng {
            for q in (0..ng).filter(|&q| self.frames[q] == self.frames[p]) {
                bar_ground[(self.ground[p], self.ground[q])] += bar_hpp[(p, q)];
            }
        }
        for (o, b) in self.blocks.iter().enumerate() {
            let l = &f.laid[o];
            for (i, ri) in b.rows.iter().enumerate() {
                for (j, rj) in b.rows.iter().enumerate() {
                    let x = bar_inner[o][(i, j)];
                    match (*ri, *rj) {
                        (Row::Excited(a), Row::Excited(c)) => bar_excited[(a, c)] += x,
                        (Row::Excited(a), Row::Copy { state, component }) => bar_drives[component][(a, state)] += x,
                        (Row::Copy { state, component }, Row::Excited(c)) => bar_drives[component][(c, state)] += x.conj(),
                        (Row::Copy { state: s1, component: g }, Row::Copy { state: s2, component: h }) if g == h => bar_ground[(s1, s2)] += x,
                        _ => {}
                    }
                }
            }
            for (i, r) in b.rows.iter().enumerate() {
                for &p in &b.explicit {
                    let x = self.explicit[p - ng];
                    let y = bar_absorb[o][0][(i, p)];
                    match *r {
                        Row::Excited(e) => bar_excited[(e, x)] += y,
                        Row::Copy { state, component } => bar_drives[component][(x, state)] += y.conj(),
                    }
                    let y = bar_emit[o][(p, i)];
                    match *r {
                        Row::Excited(e) => bar_excited[(x, e)] += y,
                        Row::Copy { state, component } => bar_drives[component][(x, state)] += y,
                    }
                }
                if let Row::Copy { state, component } = *r {
                    let omega = b.offset.plus(-1, &self.components[component].beat);
                    for p in (0..ng).filter(|&p| self.frames[p] == omega) {
                        bar_ground[(state, self.ground[p])] += bar_absorb[o][0][(i, p)];
                        bar_ground[(self.ground[p], state)] += bar_emit[o][(p, i)];
                    }
                }
                if let Row::Excited(y) = *r {
                    for p in 0..ng {
                        for (g, comp) in self.components.iter().enumerate() {
                            if self.frames[p].plus(1, &comp.beat) == b.offset {
                                bar_drives[g][(y, self.ground[p])] += bar_emit[o][(p, i)].conj();
                            }
                        }
                    }
                }
            }
            for (sp, beam) in l.beams.iter().enumerate() {
                let Some((c, j)) = *beam else { continue };
                let x = &self.components[c].couplings[j];
                for &(p, _) in b.sources.iter().filter(|x| x.1 == c) {
                    for (i, r) in b.rows.iter().enumerate() {
                        if let Row::Excited(y) = *r {
                            kappa[c][j] += bar_absorb[o][sp][(i, p)].conj() * x[(y, self.ground[p])];
                        }
                    }
                }
            }
            for &p in &b.explicit {
                let x = self.explicit[p - ng];
                for &q in &b.explicit {
                    bar_excited[(x, self.explicit[q - ng])] += bar_hpp[(p, q)];
                }
                for q in 0..ng {
                    for (g, comp) in self.components.iter().enumerate() {
                        if self.frames[q].plus(1, &comp.beat) == b.offset {
                            bar_drives[g][(x, self.ground[q])] += bar_hpp[(p, q)] + bar_hpp[(q, p)].conj();
                        }
                    }
                }
            }
            // z₀: the mean of the sources' and the explicit states' diagonal (real parts).
            let count = b.sources.len() + b.explicit.len();
            if count > 0 && bar_z0[o] != 0.0 {
                for &(p, _) in &b.sources {
                    bar_ground[(self.ground[p], self.ground[p])] += C64::real(bar_z0[o] / count as f64);
                }
                for &p in &b.explicit {
                    let x = self.explicit[p - ng];
                    bar_excited[(x, x)] += C64::real(bar_z0[o] / count as f64);
                }
            }
        }
        // Contracted with the gradients: Re tr(H̄†∇H) for H_GG and 𝒦, and per beam Re(κ̄ ∇factor) with κ̄ = tr(V̄_c†X_l).
        for (c, comp) in self.components.iter().enumerate() {
            for (j, x) in comp.couplings.iter().enumerate() {
                kappa[c][j] += (0..x.data.len()).map(|i| bar_drives[c].data[i].conj() * x.data[i]).sum::<C64>();
            }
        }
        let mut force = [0.0; 3];
        for ax in 0..AXES {
            let mut g = 0.0;
            for (wi, fr) in self.light.frequencies.iter().enumerate() {
                let field = fr.field(&f.site.a);
                for (p, q, entries) in &self.entries[wi] {
                    let dx = field.gradient[ax][*p].conj() * field.value[*q] + field.value[*p].conj() * field.gradient[ax][*q];
                    let s: C64 = entries
                        .iter()
                        .map(|&(i, j, v)| match (i < GROUND, j < GROUND) {
                            (true, true) => bar_ground[(i, j)].conj() * v,
                            (false, false) => bar_excited[(i - GROUND, j - GROUND)].conj() * v,
                            _ => C64::ZERO,
                        })
                        .sum();
                    g += (s * dx).re;
                    *w += entries.len() as f64;
                }
            }
            for (c, comp) in self.components.iter().enumerate() {
                for (j, &l) in comp.beams.iter().enumerate() {
                    let d = comp.factor(f.site.a[l].gradient[ax]);
                    g += (kappa[c][j] * d).re;
                }
            }
            force[ax] = -g;
        }
        (force, work)
    }
}
