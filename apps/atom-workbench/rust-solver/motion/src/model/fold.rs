//! The fold at the atom (DESIGN.md, final 2026-11-15; onefold step 1, not yet the run's): at a point of the motion the
//! Hamiltonian of the light there, grouped by laser frequency; every Floquet node off P -- each component's excited replica and
//! the ground states' copies its emission reaches -- folded by one Schur complement at one energy, one block per node offset;
//! and of the one excited amplitude it gives, the light shifts and the decay (the compression's Hermitian and anti-Hermitian
//! parts), the jumps, the metric, the force (by the adjoint) and the fold's own error at every node.
//!
//! # The Hamiltonian at a point, by laser frequency
//! The ground states G (`GROUND`) in the lab frame, the excited states E (`EXCITED`) in the request's frame ν_frame. At r,
//! H_GG(r) = H0_GG + Σ_ω O_ω,GG(E_ω(r)) and 𝒦(r) = H0_EE + Σ_ω O_ω,EE(E_ω(r)) − (i/2)D, the outside forms O_ω quadratic in
//! each frequency's coherent field (H0 holds them at the focus). Every frequency's light is two components c (`Part`): the
//! co-rotating V_c = Σ_l a_l X_l at e^{−iν_c t}, ν_c = δ_ω, and the counter-rotating Σ_l a_l* R_l at ν_c = −(δ_ω + 2ν_frame);
//! one rule for both and for every frequency, the tweezer's too (DESIGN §1.4). Absorbing beam l takes the momentum q_l = k_l
//! (the counter-rotating −k_l) and raises the excited state's energy by w_l = (ħ/m)(q_l·κ + q_l²/2) (DESIGN §1.2, onefold 1).
//!
//! # P, frames and the Floquet nodes
//! P holds one state set's ground states (the connected parts of the ground states by the static couplings and the tones the
//! choice makes, `sets`; DESIGN §3.2: derived, not decided), each in its frame θ_s (ψ_s = e^{−iθ_s t} x_s; `mixing_frames`, DESIGN §1.1), and the explicit
//! excited states, each at its node's offset. A node is a level (a ground F or an excited F′) at an offset (exact, `Beat`):
//! component c from ground state s reaches the excited level at o = θ_s + ν_c; the excited node at o, emitting component g,
//! reaches the ground copy at ω = o − ν_g -- a copy of state t wherever θ_t ≠ ω (where θ_t = ω it is P itself). The lattice
//! stops there (second order, DESIGN §1.1; the copies' reabsorption is read, `Reading::truncation`).
//!
//! # Blocks
//! Each offset o is a block (`Block`): its explicit states (P, the E rows), its other excited states and its copies (the Q
//! rows). Its matrix in its frame, Ĥ_o: 𝒦 on the excited rows less o; H_GG on each copy offset's rows less ω; V_g between an
//! excited row and a copy of g. Its sources: V_c from each ground state s with θ_s + ν_c = o (beam by beam), 𝒦 and V_g† from its
//! E rows, H_GG from a ground state at a copy's offset. A copy is folded in the block's decomposition, folded at the block's
//! centre z₀ first (`Status::Centred`: R₀ = (z₀ − Ĥ_CC)⁻¹, K_eff = Ĥ_SS + Ĥ_SC R₀ Ĥ_CS), left out of the block (`Omitted`), or
//! kept in P (`Kept`, a tone); one evaluation path whatever the choice, and each choice's error read by the same function
//! (`Reading`, DESIGN §1.2, M18).
//!
//! # One linear solve per pass
//! For the static P generator H (x coordinates), each block's response T_o (Q rows × n) solves, per source split (a beam with
//! its Doppler shift w, or the static part), the Sylvester equation T H − (K_eff + w) T = B -- each eigenstate at its own
//! eigenvalue, z = λ + o − w, without H's eigenvectors (DESIGN §1.5, onefold 3): with H = U t U† and K_eff = V T_K V† (complex
//! Schur forms), Z = V† T U solves column by column a triangular system (`sylvester`), at an exceptional point too. The
//! centred copies' rows follow, T_C = R₀ (Ĥ_CS T_S + B_C).
//!
//! # The compression
//! With Φ = [1; T_o] on the lifted space, A = Φ†ĤΦ = Ĥ_PP + Σ_o (Ĥ_Po T_o + T_o†Ĥ_oP + T_o†Ĥ_oo T_o) and M = 1 + Σ_o T_o†T_o
//! (the copies' time-averaged population with it); S = M^{1/2}, G = S⁻¹AS⁻¹. Its anti-Hermitian part is
//! −(i/2) S⁻¹(Σ_o ε_o†D ε_o)S⁻¹, ε_o the block's excited amplitude (its E rows the explicit states themselves, its excited Q rows
//! T_o): Σ_q L_q†L_q with L_q = C_q Σ_o e^{−iot} ε_o S⁻¹, for any T and any Hermitian S (DESIGN §1.3, §2.6).
//!
//! # Tones
//! What stays in P and is not static: an entry between P states of different frames (H_GG, or 𝒦 between explicit states of two
//! offsets) at θ_t − θ_s; a kept copy's emission Γ_t†V_g†ε_o at θ_t − ω and its partner; and the beat of the two blocks a kept
//! copy relates, ε_o'†(𝒦 − ν̄)ε_o at o' − o (o' = θ_t + ν_g; ν̄ the columns' offsets from their own frames, gauge invariant,
//! `Fold::cross`): the copy's excited node is reached both from
//! block o and from its state's sideband through its own block, one Floquet node, so their interference (the rate's, with
//! the decay -(i/2)ε_o'†Dε_o) is physics. Two blocks no kept copy relates are different Floquet nodes: their cross terms
//! are not taken -- with the beating metric dropped (M8) they would turn at |o − o'| with a size |o − o'|·|ε_o'†ε_o|/2 and
//! a second-order shift of nothing in the exact Floquet problem (onefold report, step 1); the hazard is to take the jumps'
//! cross terms the same way (step 2).
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
use crate::request::{Manifold, Request, GROUND, STATES};

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

/// The run's choice of nodes (DESIGN §3, §11): the ground states' frames, the explicit excited levels (a level at the
/// offset component `part` reaches from ground state `ground`), and the copies' statuses keyed by an absorbed component and
/// one of the block's own sources (the block), the emitting component and the copy's ground level F -- a later entry
/// overriding an earlier one; unlisted: centred.
#[derive(Clone, Debug, Default)]
pub(crate) struct Choice {
    pub(crate) frames: Vec<Beat>,
    pub(crate) explicit: Vec<(usize, Part, usize)>,
    pub(crate) copies: Vec<((Part, usize, Part, u8), Status)>,
}

impl Choice {
    /// Every node folded, every copy centred, at the frames `frames` (the ground states').
    pub(crate) fn folded(frames: Vec<Beat>) -> Choice {
        Choice { frames, explicit: vec![], copies: vec![] }
    }
}

/// A block (module documentation, # Blocks): its offset, its explicit states' P places, its Q rows (the solved ones first:
/// excited states, then decomposed copies; then the centred ones), its ground sources (place, component), its kept copies
/// (state, component, exact residual θ_t − ω) and its omitted ones.
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
    hbar_over_mass: f64,
    pub(crate) components: Vec<Component>,
    /// P's ground states (place p < ng holds ground state ground[p]).
    pub(crate) ground: Vec<usize>,
    /// Per P place its frame, exact and its value (rad/µs).
    pub(crate) frames: Vec<Beat>,
    frame_values: Vec<f64>,
    /// Per explicit place j (P place ng + j) its excited state.
    pub(crate) explicit: Vec<usize>,
    pub(crate) blocks: Vec<Block>,
    /// H0 less the outside forms at the focus: its ground block, and its excited block with −(i/2)D.
    base_ground: Mat,
    base_excited: Mat,
    /// The emission operators (GROUND × EXCITED).
    emission: [Mat; 3],
    /// Per frequency its outside form O_pq in the spherical components (STATES × STATES).
    outside: Vec<[[Mat; 3]; 3]>,
    /// The levels: per ground state its F, per excited state its level, the excited levels (manifold, F′).
    pub(crate) ground_level: [u8; GROUND],
    pub(crate) excited_level: Vec<usize>,
    pub(crate) levels: Vec<(Manifold, u8)>,
}

/// The ground states' frames by the mixing's maximum spanning tree (DESIGN §1.1, M5), at the focus at rest: a coupling of
/// ground states t ← s through the excited states, absorbing component f and emitting g (f = g: the light's own Raman and
/// Zeeman couplings), b = Σ_e V_g[e,t]* V_f[e,s]/(E_s + ν_f − 𝒦_ee), is static where θ_t − θ_s = ν_f − ν_g; its mixing is
/// |b/δ| with δ = E_t − E_s − (ν_f − ν_g), its detuning when static. The tree of the largest mixings is held static
/// (cluster.rs `frames`, Kruskal), each joined set's first state at zero.
pub(crate) fn mixing_frames(request: &Request) -> Result<Vec<Beat>, String> {
    let light = Light::new(request);
    let axes = Axes::new(request)?;
    let components = components(request, &light, &axes);
    let e: Vec<f64> = (0..STATES).map(|s| request.hamiltonian[(s, s)].re).collect();
    let d = decay(request);
    let at_focus: Vec<Mat> = components.iter().map(|c| c.couplings.iter().fold(Mat::zeros(EXCITED, GROUND), |mut m, x| {
        plus(&mut m, x);
        m
    })).collect();
    let mut edges: Vec<(usize, usize, Beat, f64)> = vec![];
    for (f, cf) in components.iter().enumerate() {
        for (g, cg) in components.iter().enumerate() {
            for s in 0..GROUND {
                for t in (0..GROUND).filter(|&t| t != s) {
                    let b: C64 = (0..EXCITED)
                        .map(|x| at_focus[g][(x, t)].conj() * at_focus[f][(x, s)] / C64::new(e[s] + cf.nu - e[GROUND + x], 0.5 * d[(x, x)].re))
                        .sum();
                    if b == C64::ZERO {
                        continue;
                    }
                    let delta = e[t] - e[s] - (cf.nu - cg.nu);
                    edges.push((t, s, cf.beat.plus(-1, &cg.beat), b.abs() / delta.abs()));
                }
            }
        }
    }
    Ok(super::cluster::frames(GROUND, &edges, &axes.zero()))
}

/// The components of a request's light that couple anything, in the order of the frequencies, co-rotating first.
fn components(request: &Request, light: &Light, axes: &Axes) -> Vec<Component> {
    let mut out = vec![];
    let ground: Vec<usize> = (0..GROUND).collect();
    let excited: Vec<usize> = (GROUND..STATES).collect();
    for (frequency, f) in light.frequencies.iter().enumerate() {
        for counter in [false, true] {
            let part = Part { frequency, counter };
            let beat = axes.component(light, part);
            let couplings: Vec<Mat> = f.beams.iter().map(|&l| (if counter { &request.beams[l].counter_coupling } else { &request.beams[l].coupling }).block(&excited, &ground)).collect();
            if couplings.iter().all(|m| m.data.iter().all(|z| *z == C64::ZERO)) {
                continue;
            }
            let kicks = f.beams.iter().map(|&l| light.beams[l].wave_vector().map(|k| if counter { -k } else { k })).collect();
            out.push(Component { part, nu: axes.omega(&beat), beat, beams: f.beams.clone(), couplings, kicks });
        }
    }
    out
}

impl Fold {
    /// The state sets of a request with a choice of nodes (DESIGN §3.2): the ground states joined by a block they both
    /// source (its emission into each is static), by a kept copy (a tone), by H_GG (static or a tone), and by 𝒦 between the
    /// explicit states of two blocks (a tone) -- the connected parts, each in increasing order.
    pub(crate) fn sets(request: &Request, choice: &Choice) -> Result<Vec<Vec<usize>>, String> {
        let all = Fold::build(request, choice, &(0..GROUND).collect::<Vec<_>>())?;
        let mut part: Vec<usize> = (0..GROUND).collect();
        fn find(part: &mut [usize], mut x: usize) -> usize {
            while part[x] != x {
                part[x] = part[part[x]];
                x = part[x];
            }
            x
        }
        let join = |a: usize, b: usize, part: &mut Vec<usize>| {
            let (a, b) = (find(part, a), find(part, b));
            part[a.max(b)] = a.min(b);
        };
        let first: Vec<Option<usize>> = all.blocks.iter().map(|b| b.sources.first().map(|x| x.0)).collect();
        for (b, f) in all.blocks.iter().zip(&first) {
            let Some(f) = *f else { continue };
            for &(s, _) in &b.sources {
                join(f, s, &mut part);
            }
            for &(t, _, _) in &b.kept {
                join(f, t, &mut part);
            }
        }
        for t in 0..GROUND {
            for u in 0..GROUND {
                if t != u && (all.base_ground[(t, u)] != C64::ZERO || all.outside.iter().any(|o| o.iter().flatten().any(|m| m[(t, u)] != C64::ZERO))) {
                    join(t, u, &mut part);
                }
            }
        }
        for (i, bi) in all.blocks.iter().enumerate() {
            for (j, bj) in all.blocks.iter().enumerate() {
                let couple = bi.explicit.iter().any(|&p| bj.explicit.iter().any(|&q| all.base_excited[(all.explicit[p - GROUND], all.explicit[q - GROUND])] != C64::ZERO));
                if i != j && couple {
                    if let (Some(a), Some(b)) = (first[i], first[j]) {
                        join(a, b, &mut part);
                    }
                }
            }
        }
        let mut out: Vec<Vec<usize>> = vec![];
        let mut roots: Vec<usize> = vec![];
        for s in 0..GROUND {
            let r = find(&mut part, s);
            match roots.iter().position(|&x| x == r) {
                Some(at) => out[at].push(s),
                None => {
                    roots.push(r);
                    out.push(vec![s]);
                }
            }
        }
        Ok(out)
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
        for w in 0..light.frequencies.len() {
            Bases::new(request, &light, w)?;
        }
        let components = components(request, &light, &axes);
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
        let mut explicit = vec![];
        for &(level, part, source) in choice.explicit.iter().filter(|x| ground.contains(&x.2)) {
            let offset = state_frames[source].plus(1, &components[find(part)?].beat);
            for x in (0..EXCITED).filter(|&x| excited_level[x] == level) {
                if explicit.contains(&x) {
                    return Err("一个激发能级只能在一个偏移上显式".into());
                }
                explicit.push(x);
                frames.push(offset.clone());
            }
        }
        let frame_values: Vec<f64> = frames.iter().map(|b| axes.omega(b)).collect();
        // H0 less the outside forms at the focus (the field there Σ_l E_l).
        let outside: Vec<[[Mat; 3]; 3]> = light.frequencies.iter().map(|f| outside_form(&request.outside_forms[f.form].matrices)).collect();
        let mut base = request.hamiltonian.clone();
        for (w, f) in light.frequencies.iter().enumerate() {
            let e0: [C64; 3] = std::array::from_fn(|q| f.fields.iter().map(|e| e[q]).sum());
            for p in 0..3 {
                for q in 0..3 {
                    base.add_scaled(-(e0[p].conj() * e0[q]), &outside[w][p][q]);
                }
            }
        }
        let all: Vec<usize> = (0..GROUND).collect();
        let excited: Vec<usize> = (GROUND..STATES).collect();
        let d = decay(request);
        let mut base_excited = base.block(&excited, &excited);
        base_excited.add_scaled(C64::new(0.0, -0.5), &d);
        // The excited block's pattern: the base, the decay, every outside form.
        let pattern = |x: usize, y: usize| base_excited[(x, y)] != C64::ZERO || outside.iter().any(|o| o.iter().flatten().any(|m| m[(GROUND + x, GROUND + y)] != C64::ZERO));
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
        for (offset, sources) in offsets.into_iter().zip(sources) {
            let value = axes.omega(&offset);
            let mine: Vec<usize> = (0..explicit.len()).filter(|&j| frames[ng + j] == offset).map(|j| ng + j).collect();
            let theirs: Vec<usize> = mine.iter().map(|&p| explicit[p - ng]).collect();
            // The excited Q rows: reached from a source or an explicit state of the block, closed under 𝒦's pattern.
            let mut reached = vec![false; EXCITED];
            for &(p, c) in &sources {
                for m in &components[c].couplings {
                    (0..EXCITED).filter(|&x| m[(x, ground[p])] != C64::ZERO).for_each(|x| reached[x] = true);
                }
            }
            for &x in &theirs {
                (0..EXCITED).filter(|&y| pattern(y, x)).for_each(|y| reached[y] = true);
            }
            let mut changed = true;
            while changed {
                changed = false;
                for x in 0..EXCITED {
                    if !reached[x] || theirs.contains(&x) {
                        continue;
                    }
                    for y in 0..EXCITED {
                        if !reached[y] && (pattern(y, x) || pattern(x, y)) {
                            reached[y] = true;
                            changed = true;
                        }
                    }
                }
            }
            let q: Vec<usize> = (0..EXCITED).filter(|&x| reached[x] && !theirs.contains(&x)).collect();
            let rows_excited: Vec<usize> = theirs.iter().chain(&q).copied().collect();
            // The copies: per emitting component, the ground states it reaches from the block's excited rows off their frames.
            let mut decomposed = vec![];
            let mut centred = vec![];
            let mut kept = vec![];
            let mut omitted = vec![];
            for (g, comp) in components.iter().enumerate() {
                let omega = offset.plus(-1, &comp.beat);
                for t in (0..GROUND).filter(|&t| state_frames[t] != omega) {
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
            rows.extend(centred);
            blocks.push(Block { offset, value, explicit: mine, rows, solved, sources, kept, omitted });
        }
        Ok(Fold {
            hbar_over_mass: crate::units::hbar_over_mass(request.mass_kg),
            light,
            components,
            ground: ground.to_vec(),
            frames,
            frame_values,
            explicit,
            blocks,
            base_ground: base.block(&all, &all),
            base_excited,
            emission: std::array::from_fn(|q| request.emission[q].block(&all, &excited)),
            outside,
            ground_level,
            excited_level,
            levels,
        })
    }

    /// n, P's states.
    pub(crate) fn size(&self) -> usize {
        self.ground.len() + self.explicit.len()
    }

    /// The light's fields at a point and their gradients (module documentation, # The Hamiltonian at a point).
    fn site(&self, r: [f64; 3], kappa: [f64; 3], work: &mut f64) -> Site {
        let a: Vec<Amplitude> = self.light.beams.iter().map(|b| b.amplitude_to(r, 1)).collect();
        let mut ground = self.base_ground.clone();
        let mut excited = self.base_excited.clone();
        let mut d_ground: [Mat; 3] = std::array::from_fn(|_| Mat::zeros(GROUND, GROUND));
        let mut d_excited: [Mat; 3] = std::array::from_fn(|_| Mat::zeros(EXCITED, EXCITED));
        let gs: Vec<usize> = (0..GROUND).collect();
        let es: Vec<usize> = (GROUND..STATES).collect();
        for (w, f) in self.light.frequencies.iter().enumerate() {
            let field = f.field(&a);
            for p in 0..3 {
                for q in 0..3 {
                    let o = &self.outside[w][p][q];
                    if o.data.iter().all(|z| *z == C64::ZERO) {
                        continue;
                    }
                    let (og, oe) = (o.block(&gs, &gs), o.block(&es, &es));
                    let x = field.value[p].conj() * field.value[q];
                    ground.add_scaled(x, &og);
                    excited.add_scaled(x, &oe);
                    for ax in 0..AXES {
                        let dx = field.gradient[ax][p].conj() * field.value[q] + field.value[p].conj() * field.gradient[ax][q];
                        d_ground[ax].add_scaled(dx, &og);
                        d_excited[ax].add_scaled(dx, &oe);
                    }
                    *work += 4.0 * (GROUND * GROUND + EXCITED * EXCITED) as f64;
                }
            }
        }
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
        Site { a, ground, d_ground, excited, d_excited, drives, doppler }
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

    /// The fold at a point (r, κ), the light pattern's offset `pattern` (2 k_max |v|, rad/µs) read beside it (module
    /// documentation). None where a Schur form does not converge.
    pub(crate) fn at(&self, r: [f64; 3], kappa: [f64; 3], pattern: f64) -> Option<Folded> {
        let mut work = 0.0;
        let site = self.site(r, kappa, &mut work);
        let n = self.size();
        let laid: Vec<Laid> = (0..self.blocks.len()).map(|o| self.lay(o, &site, &mut work)).collect();
        let hpp = self.p_matrix(&site, &laid);
        // Each block's evaluation order: its centred copies eliminated at z₀, its solved rows' Schur form.
        let mut eliminated = vec![];
        for (b, l) in self.blocks.iter().zip(&laid) {
            eliminated.push(Eliminated::of(b, l, &mut work)?);
        }
        // The passes.
        let diag = |h: &Mat| Mat::from_fn(n, n, |i, j| if i == j { h[(i, i)] } else { C64::ZERO });
        let mut stages: Vec<Stage> = vec![];
        let h_minus = diag(&hpp);
        let s0 = self.solve(&laid, &eliminated, &h_minus, true, &mut work)?;
        let h0 = diag(&self.bloch(&hpp, &laid, &s0, &mut work));
        stages.push(s0);
        let s1 = self.solve(&laid, &eliminated, &h0, true, &mut work)?;
        let h1 = self.bloch(&hpp, &laid, &s1, &mut work);
        stages.push(s1);
        let s2 = self.solve(&laid, &eliminated, &h1, false, &mut work)?;
        let h2 = self.bloch(&hpp, &laid, &s2, &mut work);
        let difference = |a: &Mat, b: &Mat| {
            let mut d = a.clone();
            d.add_scaled(-C64::ONE, b);
            d.frobenius()
        };
        let ratio = difference(&h2, &h1) / difference(&h1, &h0);
        stages.push(s2);
        let last = stages.last().unwrap();
        // The compression.
        let mut a = hpp.clone();
        let mut metric = Mat::zeros(n, n);
        for (l, t) in laid.iter().zip(&last.responses) {
            plus(&mut a, &mul(&l.emit, t, &mut work));
            let total = l.absorb.iter().fold(Mat::zeros(t.rows, n), |mut x, y| {
                plus(&mut x, y);
                x
            });
            plus(&mut a, &mul_adj(t, &total, &mut work));
            plus(&mut a, &mul_adj(t, &mul(&l.inner, t, &mut work), &mut work));
            plus(&mut metric, &mul_adj(t, t, &mut work));
        }
        let x = hermitian_eigen(&metric);
        work += 10.0 * (n * n * n) as f64;
        let s: Vec<f64> = x.values.iter().map(|&xi| (1.0 + xi).sqrt()).collect();
        let normal = Mat::from_fn(n, n, |i, j| (0..n).map(|k| x.vectors[(i, k)] * x.vectors[(j, k)].conj() / s[k]).sum());
        work += (n * n * n) as f64;
        let generator = mul(&mul(&normal, &a, &mut work), &normal, &mut work);
        // ε per block: its explicit rows, then its excited Q rows.
        let amplitudes: Vec<Mat> = self.blocks.iter().zip(&last.responses).map(|(b, t)| self.amplitude(b, t)).collect();
        let tones = self.tones(&site, &amplitudes, &normal, &mut work);
        let jumps = self
            .blocks
            .iter()
            .zip(&amplitudes)
            .map(|(b, e)| {
                let rows = b.excited(self);
                let es: Vec<usize> = (0..GROUND).collect();
                std::array::from_fn(|q| mul(&mul(&self.emission[q].block(&es, &rows), e, &mut work), &normal, &mut work))
            })
            .collect();
        let excitation = {
            let mut q = Mat::zeros(n, n);
            for (b, t) in self.blocks.iter().zip(&last.responses) {
                let rows: Vec<usize> = (0..b.rows.len()).filter(|&i| matches!(b.rows[i], Row::Excited(_))).collect();
                let tq = t.block(&rows, &(0..n).collect::<Vec<_>>());
                plus(&mut q, &mul_adj(&tq, &tq, &mut work));
            }
            hermitian_eigen(&q).values.iter().fold(0.0_f64, |m, &v| m.max(v))
        };
        let mut folded = Folded { n, a, metric, normal, generator, amplitudes, tones, jumps, excitation, ratio, readings: vec![], work: 0.0, reading_work: 0.0, site, laid, eliminated, stages, hpp, h: [h0, h1, h2], m_eigen: (x.values, x.vectors) };
        let mut reading_work = 0.0;
        folded.readings = self.readings(&folded, pattern, &mut reading_work);
        (folded.work, folded.reading_work) = (work, reading_work);
        Some(folded)
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
    /// split and in all, its centred rows following. `diagonal`: h is diagonal (its Schur form itself).
    fn solve(&self, laid: &[Laid], eliminated: &[Eliminated], h: &Mat, diagonal: bool, work: &mut f64) -> Option<Stage> {
        let n = self.size();
        let triangle = if diagonal { Triangle { t: h.clone(), u: Mat::identity(n) } } else { Triangle::of(h, work)? };
        let mut splits = vec![];
        let mut responses = vec![];
        for (l, e) in laid.iter().zip(eliminated) {
            let mut per = vec![];
            let mut total = Mat::zeros(e.solved, n);
            for (source, &w) in e.sources.iter().zip(&l.shifts) {
                let b = mul(&e.block.to(source, work), &triangle.u, work);
                let z = sylvester(&e.block.t, &e.block.ends, &triangle.t, w, &b, work);
                let t = mul_by_adj(&e.block.from(&z, work), &triangle.u, work);
                plus(&mut total, &t);
                per.push(t);
            }
            responses.push(e.rows(l, &total, work));
            splits.push(per);
        }
        Some(Stage { triangle, splits, responses })
    }

    /// The Bloch generator of a pass's responses: Ĥ_PP + Σ_o Ĥ_Po T_o.
    fn bloch(&self, hpp: &Mat, laid: &[Laid], stage: &Stage, work: &mut f64) -> Mat {
        let mut h = hpp.clone();
        for (l, t) in laid.iter().zip(&stage.responses) {
            plus(&mut h, &mul(&l.emit, t, work));
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
        for p in 0..ng {
            for q in (0..ng).filter(|&q| self.frames[q] != self.frames[p] && site.ground[(self.ground[p], self.ground[q])] != C64::ZERO) {
                let mut m = Mat::zeros(n, n);
                m[(p, q)] = site.ground[(self.ground[p], self.ground[q])];
                add(self.frames[p].plus(-1, &self.frames[q]), self.frame_values[p] - self.frame_values[q], m, &mut out);
            }
        }
        for p in ng..n {
            for q in (ng..n).filter(|&q| self.frames[q] != self.frames[p]) {
                let z = site.excited[(self.explicit[p - ng], self.explicit[q - ng])];
                if z != C64::ZERO {
                    let mut m = Mat::zeros(n, n);
                    m[(p, q)] = z;
                    add(self.frames[p].plus(-1, &self.frames[q]), self.frame_values[p] - self.frame_values[q], m, &mut out);
                }
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
    d_ground: [Mat; 3],
    excited: Mat,
    d_excited: [Mat; 3],
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
/// laid side by side -- T block diagonal (each part's rows contiguous, `ends` each row's part's end), V the parts' U on
/// their rows. One decomposition whatever the parts (a part is the whole where all rows join); the parts save work alone
/// (the m blocks of DESIGN §6.2), the result the dense one's.
struct Factored {
    t: Mat,
    ends: Vec<usize>,
    parts: Vec<(Vec<usize>, usize, Mat)>,
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
        for idx in groups {
            let tri = Triangle::of(&k.block(&idx, &idx), work)?;
            let len = idx.len();
            for a in 0..len {
                ends[at + a] = at + len;
                for b in 0..len {
                    t[(at + a, at + b)] = tri.t[(a, b)];
                }
            }
            parts.push((idx, at, tri.u));
            at += len;
        }
        Some(Factored { t, ends, parts })
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

    /// The triangular solve (c − T) z = r, part by part (a column of `sylvester`).
    fn back(&self, c: C64, r: &[C64], work: &mut f64) -> Vec<C64> {
        let m = r.len();
        let mut z = vec![C64::ZERO; m];
        for k in (0..m).rev() {
            let mut acc = r[k];
            for l in k + 1..self.ends[k] {
                acc += self.t[(k, l)] * z[l];
            }
            z[k] = acc / (c - self.t[(k, k)]);
            *work += (self.ends[k] - k) as f64;
        }
        z
    }
}

/// A block's evaluation order at a point (module documentation, # Blocks): its centred copies eliminated at z₀,
/// R₀ = (z₀ − Ĥ_CC)⁻¹; K_eff on the solved rows and its Schur form; per split its effective source.
struct Eliminated {
    solved: usize,
    resolvent: Mat,
    block: Factored,
    sources: Vec<Mat>,
}

impl Eliminated {
    fn of(b: &Block, l: &Laid, work: &mut f64) -> Option<Eliminated> {
        let m = b.rows.len();
        let (s, c): (Vec<usize>, Vec<usize>) = ((0..b.solved).collect(), (b.solved..m).collect());
        let mut k = l.inner.block(&s, &s);
        let resolvent = {
            let r = Mat::from_fn(c.len(), c.len(), |i, j| if i == j { C64::real(l.z0) } else { C64::ZERO } - l.inner[(c[i], c[j])]);
            inverse(&r, work)?
        };
        let all: Vec<usize> = (0..l.absorb[0].cols).collect();
        let sc = l.inner.block(&s, &c);
        let rc = mul(&sc, &resolvent, work);
        plus(&mut k, &mul(&rc, &l.inner.block(&c, &s), work));
        let mut sources: Vec<Mat> = l.absorb.iter().map(|a| a.block(&s, &all)).collect();
        plus(&mut sources[0], &mul(&rc, &l.absorb[0].block(&c, &all), work));
        Some(Eliminated { solved: b.solved, resolvent, block: Factored::of(&k, work)?, sources })
    }

    /// The block's response on all its Q rows from that on its solved rows: the centred ones T_C = R₀(Ĥ_CS T_S + B_C).
    fn rows(&self, l: &Laid, t: &Mat, work: &mut f64) -> Mat {
        let m = l.inner.rows;
        let n = t.cols;
        let (s, c): (Vec<usize>, Vec<usize>) = ((0..self.solved).collect(), (self.solved..m).collect());
        let all: Vec<usize> = (0..n).collect();
        let mut tc = mul(&l.inner.block(&c, &s), t, work);
        plus(&mut tc, &l.absorb[0].block(&c, &all));
        let tc = mul(&self.resolvent, &tc, work);
        Mat::from_fn(m, n, |i, j| if i < self.solved { t[(i, j)] } else { tc[(i - self.solved, j)] })
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
fn sylvester(tk: &Mat, ends: &[usize], th: &Mat, w: f64, b: &Mat, work: &mut f64) -> Mat {
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
            z[(k, j)] = acc / (c - tk[(k, k)]);
        }
    }
    *work += (m * n * n.saturating_sub(1) / 2 + n * (0..m).map(|k| ends[k] - k).sum::<usize>()) as f64;
    z
}

/// Z with Z T_H† − (T_K + w)† Z = B: the adjoint of `sylvester`, the same triangular forms solved from the last column and
/// the first row.
fn sylvester_adjoint(tk: &Mat, ends: &[usize], th: &Mat, w: f64, b: &Mat, work: &mut f64) -> Mat {
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
            z[(k, j)] = acc / (c - tk[(k, k)].conj());
        }
    }
    *work += (m * n * n.saturating_sub(1) / 2 + n * (0..m).map(|k| ends[k] - k).sum::<usize>()) as f64;
    z
}

/// One pass: its H's Schur form, per block and split the response on the solved rows, per block the response on all Q rows.
struct Stage {
    triangle: Triangle,
    splits: Vec<Vec<Mat>>,
    responses: Vec<Mat>,
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
/// excited amplitude ε_o; the tones; per block its jumps C_q ε_o S⁻¹ (GROUND × n, at e^{−iot}); the excitation x (the largest
/// eigenvalue of Σ T†T on the excited Q rows); the passes' ratio; the readings; the complex multiply-adds made, the fold's and
/// its readings'. And what the adjoint, the rise and the fall read again.
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
    site: Site,
    laid: Vec<Laid>,
    eliminated: Vec<Eliminated>,
    stages: Vec<Stage>,
    hpp: Mat,
    h: [Mat; 3],
    m_eigen: (Vec<f64>, Mat),
}

impl Fold {
    /// A node's rows in its block (DESIGN §11.1): of an excited level or a ground level's copy by a component.
    pub(crate) fn node_rows(&self, block: usize, node: Node) -> Vec<usize> {
        let b = &self.blocks[block];
        (0..b.rows.len())
            .filter(|&i| match (b.rows[i], node) {
                (Row::Excited(x), Node::Excited(level)) => self.excited_level[x] == level,
                (Row::Copy { state, component }, Node::Copy { level, component: g }) => component == g && self.ground_level[state] == level,
                _ => false,
            })
            .collect()
    }
}

impl Folded {
    /// The rise of a node (DESIGN §11.1): its amplitude x_X = T_X x of the P state x (Bloch coordinates, x = S⁻¹y), its
    /// rows `rows` of its block's response -- exact: the metric held T_X†T_X x as the explicit amplitude's square.
    pub(crate) fn rise(&self, block: usize, rows: &[usize], x: &[C64]) -> Vec<C64> {
        let t = &self.stages.last().unwrap().responses[block];
        rows.iter().map(|&i| (0..self.n).map(|j| t[(i, j)] * x[j]).sum()).collect()
    }

    /// The fall of risen nodes (DESIGN §11.1): the state Ψ -- P's x, each risen node its own amplitude (block, rows, x_X), every
    /// other Q row following T -- projected onto the fold's span and normalized, y = S⁻¹Φ†Ψ = S⁻¹(M x + Σ T_X†(x_X − T_X x)),
    /// with the fast part it drops, δp = ‖Ψ‖² − ‖y‖² (what does not follow T; zero where every x_X = T_X x).
    pub(crate) fn fall(&self, x: &[C64], risen: &[(usize, Vec<usize>, Vec<C64>)]) -> (Vec<C64>, f64) {
        let n = self.n;
        let mut overlap: Vec<C64> = self.metric.apply(x).iter().zip(x).map(|(m, a)| *m + *a).collect();
        let mut norm: f64 = x.iter().zip(&overlap).map(|(a, b)| (a.conj() * *b).re).sum();
        let t = &self.stages.last().unwrap().responses;
        for (block, rows, amplitude) in risen {
            let following = self.rise(*block, rows, x);
            for ((&i, a), f) in rows.iter().zip(amplitude).zip(&following) {
                let d = *a - *f;
                for j in 0..n {
                    overlap[j] += t[*block][(i, j)].conj() * d;
                }
                norm += a.norm_sqr() - f.norm_sqr();
            }
        }
        let y = self.normal.apply(&overlap);
        let kept: f64 = y.iter().map(|z| z.norm_sqr()).sum();
        (y, norm - kept)
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
                let z = e.block.back(c, &r, work);
                for k in 0..m {
                    out[k] += z[k];
                }
            }
            out
        };
        // The sources in the Schur coordinates (K_eff's on the left, H's on the right), per block and split.
        let tilde: Vec<Vec<Mat>> = f.eliminated.iter().map(|e| e.sources.iter().map(|s| mul(&e.block.to(s, work), &tri.u, work)).collect()).collect();
        // Each column's whole excited response (the denominator).
        let tu: Vec<Mat> = last.responses.iter().map(|t| mul(t, &tri.u, work)).collect();
        // A column whose response is at the rounding of the responses' own (an idle state's) reads nothing: its relative
        // error is the rounding's.
        let floor = (crate::units::rounding(n) * tu.iter().map(|t| t.frobenius()).sum::<f64>()).powi(2);
        let whole: Vec<f64> = (0..n)
            .map(|j| self.blocks.iter().zip(&tu).map(|(b, t)| (0..b.rows.len()).filter(|&i| matches!(b.rows[i], Row::Excited(_))).map(|i| t[(i, j)].norm_sqr()).sum::<f64>()).sum::<f64>())
            .map(|x| if x > floor { x } else { 0.0 })
            .collect();
        // A row's error in its reading's units: an excited row's relative to the column's whole excited response (a rate's),
        // a copy's absolute (a population's, per unit P amplitude: the Schur columns are of unit norm).
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
            let back = e.block.from(&Mat::from_fn(v.len(), 1, |i, _| v[i]), work).data;
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
                    put(o, nd, kind, scale * x / per(nd, j), out);
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
            for (k, t) in last.splits[o].iter().enumerate() {
                let zt = mul(&e.block.to(t, work), &tri.u, work);
                let src = mul(&zt, &dht, work).scaled(-C64::ONE);
                plus(&mut dz, &sylvester(&e.block.t, &e.block.ends, &tri.t, f.laid[o].shifts[k], &src, work));
            }
            for j in 0..n {
                let col: Vec<C64> = (0..e.solved).map(|i| dz[(i, j)]).collect();
                split(o, &col, j, 0, 1.0, &mut out, work);
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
        let bare = bare.decompose(&f.site.excited).then_some(bare);
        if let Some(b) = &bare {
            *work += b.work;
        }
        // The copies: the evaluation order's error by the block perturbation δK on the solved rows, R(z_j) δK T_j (one
        // function for centring, its prospect and omission); the reabsorption a third order would add; the mixing.
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
                            let st = e.block.to(&src, work);
                            let zz = e.block.back(z, &st.data, work);
                            let back = e.block.from(&Mat::from_fn(zz.len(), 1, |i, _| zz[i]), work).data;
                            sizes[k] = back.iter().enumerate().map(|(i, x)| match per(node(b, i), j) { d if d > 0.0 => x.norm_sqr() / d, _ => 0.0 }).sum();
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
        let mut bar_source: Vec<Vec<Mat>> = f.eliminated.iter().map(|e| e.sources.iter().map(|s| Mat::zeros(s.rows, s.cols)).collect()).collect();
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
                    plus(&mut arg, &l.absorb[0].block(&cr, &all));
                    let r0b = mul_adj(&e.resolvent, &btc, w);
                    plus(&mut bar_ts, &mul_adj(&inner_cs, &r0b, w));
                    plus(&mut bar_r0[o], &mul_by_adj(&btc, &arg, w));
                    let ics = mul_by_adj(&r0b, &ts, w);
                    for (a, &ci) in cr.iter().enumerate() {
                        for (b2, &si) in sr.iter().enumerate() {
                            bar_inner[o][(ci, si)] += ics[(a, b2)];
                        }
                        for j in 0..n {
                            bar_absorb[o][0][(ci, j)] += r0b[(a, j)];
                        }
                    }
                }
                // Each split: Λ = V Z' U†, Z' = sylvester_adjoint(V† T̄ U); B̄ += Λ, K̄ += Λ T†, H̄ −= T†Λ.
                let tri = &stage.triangle;
                let g = mul(&e.block.to(&bar_ts, w), &tri.u, w);
                for (sp, (t, &shift)) in stage.splits[o].iter().zip(&l.shifts).enumerate() {
                    let z = sylvester_adjoint(&e.block.t, &e.block.ends, &tri.t, shift, &g, w);
                    let lambda = mul_by_adj(&e.block.from(&z, w), &tri.u, w);
                    plus(&mut bar_source[o][sp], &lambda);
                    plus(&mut bar_k[o], &mul_by_adj(&lambda, t, w));
                    let th = mul_adj(t, &lambda, w);
                    bar_h.add_scaled(-C64::ONE, &th);
                }
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
        // Each block's evaluation order: sources_eff = absorb_S (+ Ĥ_SC R₀ absorb_C on the static split), K_eff = Ĥ_SS +
        // Ĥ_SC R₀ Ĥ_CS, R₀ = (z₀ − Ĥ_CC)⁻¹.
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
                let ac = l.absorb[0].block(&cr, &all);
                let r0 = &e.resolvent;
                let bs0 = &bar_source[o][0];
                // source0 ∋ Ĥ_SC R₀ B_C; K ∋ Ĥ_SC R₀ Ĥ_CS.
                let mut bar_sc = mul_by_adj(bs0, &mul(r0, &ac, w), w);
                plus(&mut bar_sc, &mul_by_adj(&bar_k[o], &mul(r0, &cs, w), w));
                let mut br0 = mul(&mul_adj(&sc, bs0, w), &ac.adjoint(), w);
                plus(&mut br0, &mul(&mul_adj(&sc, &bar_k[o], w), &cs.adjoint(), w));
                plus(&mut bar_r0[o], &br0);
                let bac = mul_adj(&mul(&sc, r0, w), bs0, w);
                let bcs = mul_adj(&mul(&sc, r0, w), &bar_k[o], w);
                for (a, &ci) in cr.iter().enumerate() {
                    for j in 0..n {
                        bar_absorb[o][0][(ci, j)] += bac[(a, j)];
                    }
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
        let mut kappa: Vec<Vec<C64>> = self.components.iter().map(|c| vec![C64::ZERO; c.beams.len()]).collect();
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
            let tr = |bar: &Mat, d: &Mat| (0..bar.data.len()).map(|i| (bar.data[i].conj() * d.data[i]).re).sum::<f64>();
            let mut g = tr(&bar_ground, &f.site.d_ground[ax]) + tr(&bar_excited, &f.site.d_excited[ax]);
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
