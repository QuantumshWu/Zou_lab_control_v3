//! The start's decision (FOLDAPI §3, §7.5): one representation for the whole run -- per ground set the excited nodes kept
//! in P -- from the worst case over the atoms' local physics, made once before any trajectory; the run evaluates on it
//! and never changes it (run.rs # The representation and the decision times).
//!
//! # The local physics
//! A fold's readings depend on the light at the atom and on its velocity alone, not on where the atom is: the point is
//! the focus, and the light there is replaced by every light an atom can see (`Local`) --
//! - s, the trap light's intensity relative to its focus value, 0 to 1 (the trap's beams' amplitudes times √s);
//! - φ, each beam's phase relative to its frequency's first beam, all of [0, 2π) (the interference patterns: a
//!   standing wave's antinode, node and every point between; they drift in the experiment);
//! - v, the velocity, |v| ≤ √(2 V₀/m), V₀ the trap's depth (an atom held by the trap has no more kinetic energy).
//!
//! The phases of beams of different frequencies are not taken: they are each frequency's time origin, which turns the
//! tones' phases and nothing static (a loop of couplings returns every frequency it absorbs).
//!
//! # The search
//! Sobol points in the unit cube of these coordinates (`Sobol`), their number doubled until the decision -- per ground
//! set its explicit nodes -- does not change over a doubling, from 2^d points (d the coordinates: every orthant of the
//! cube once, every pair of halves of every coordinate seen before "unchanged" can be said); then about each set's
//! worst point a box of the global points' spacing, searched by the same doubling. No threshold of the search's own.
//!
//! # At each point
//! The fold of each set on the decision so far, at the focus with the point's light and momentum, a decision's forward
//! (`Fold::prototype`, its readings in the potential channel's floor: the point's own speed never loosens them) with
//! the set's candidate nodes offered (`Choice::candidates`): where the fold does not exist there, where its walk
//! raises a node, or where its landing does not exist (`Fold::prototype_left`, the node its attribution names), that
//! node joins P -- the union over the points --, and the point is read again on the larger P. A round's points are
//! read in parallel on one decision, then every node asked for joins at once and the points that asked are read again:
//! the decision does not depend on the threads' order.
//!
//! # The terms (fixedrep.status.md, 逐点读数 H; 项清单完成 I)
//! What joins two ground states, turns, or is a copy of an explicit state is a term of one list, each with its size a(r)
//! at the point's light and its distance Δ(r) from P, read by one function (`reading`): ρ = √w(m), m = Σ|a|/|Δ|, w(m)
//! the dressed two-state admixture, rate(ρ) against the share. The ground states' couplings (G) come first, on the light
//! alone and before any fold:
//! - two ground states s, t (`Link`): through an excited state x, absorbing beam l of component f from s and emitting
//!   into beam l′ of g onto t, a = |f_l X_l[x, s]| |f_l′ X_l′[x, t]| over the absorption's optical detuning
//!   |H_ss(r) + ν_f − 𝒦_xx(r) − w_l|, at the two-photon mismatch Δ = H_tt(r) − H_ss(r) − (ν_f − ν_g) + (ħ/m)((q_l −
//!   q_l′)·κ + |q_l − q_l′|²/2) (the copy's kinetic energy at κ + q_l − q_l′); and the direct entry |H_GG(r)[t, s]| at
//!   H_tt(r) − H_ss(r). m sums them per link (t ← s at the beat β = ν_f − ν_g, the direct one's zero), the larger of its
//!   two directions. Beyond the share at any point read the link is static: its two ends one set (`Choice::skeleton`),
//!   the frames along the kept links' maximum spanning tree by their largest m over the points read; within it, t is a
//!   copy of s's fold. A skeleton that changes merges its sets' decisions -- their nodes and their terms, those named by
//!   an offset of a set whose frames moved read again --, and every point read so far is read again on it.
//!
//! The terms of a set's fold follow, every point that asks for one read again on it, as a node joins:
//! - an entry of P between two places of one kind and different frames (`Term::Pair`): a = |Ĥ_PP(r)[p, q]| (two explicit
//!   states' with half their decay's cross term), Δ = the difference of the two diagonals there (its beat with the
//!   frames' diagonals: frame-free); a copy of a ground state at a P place's frame reaches that place by the same entry;
//!   beyond: a tone;
//! - an explicit state x's copy at a block's offset (`Term::Drive`): m = Σ over the block's sources (s, c) and beams l of
//!   |f_l X_l[x, s]| / |H_ss(r) + ν_c − 𝒦_xx(r) − w_l| -- the copy's place in the block, its optical detuning with the
//!   light shifts, the Doppler shift and −iΓ/2, not the beat of its offset and frame (its beat is Δ only once it is P's);
//!   beyond: its drive a tone of P, no row of the block;
//! - two blocks' cross decay (`Term::Cross`): a = ½ Σ_q Σ_s l_o l_o′, each block's landing amplitude at the point,
//!   l_o(q, s) = Σ_x |C_q[s, x]| e_o(x), e_o(x) = Σ |f_l X_l[x, s_p]| over the same optical detuning (an explicit row
//!   one), Δ = |o − o′|; beyond: a tone.
//! The terms are read on the light alone, before the forward: where a point asks for a term, it asks for no node in that
//! round (the forward is read on the decision with it).
//!
//! # What is made once
//! A set's fold on a decision -- its tiers (`Choice::base_of`, the costly part), the fold, its candidates -- is made once,
//! by the first thread that asks for it while the others wait, and shared (`Built`); each thread reads on its own copy of
//! the fold with its own workspace, taken from the fold's pool and given back. A round's joins are planned on the nodes'
//! placement alone (`Search::placed`: a node holding a state explicit at another offset is refused), and only each
//! set's last decision of the round is made, the sets' at once; where it cannot be made
//! the keys join one by one again, each made (`Search::join`) -- the same as joining them one by one wherever every one
//! of them can be made with those before it. What a point asks of a set's decision is the point's and the decision's
//! alone: a point read again on a set whose nodes did not change since it was read there -- the other sets of a point
//! read again for one set's join, the final reading of a point whose last round was on the final decision -- answers
//! what it answered (`Search::asked`), not read again.
//!
//! # The report (`Decision`)
//! Per set its explicit nodes, the worst point and its reading on the final decision (rate units: the bound), the
//! points beyond the share where no node is offered; the rounds' counts, the time.
use super::proto::{Cache, Cell, Taken, Workspace};
use super::{rate, Beat, Choice, Fold, Term, EXCITED, GROUND};
use crate::field::Light;
use crate::linalg::C64;
use crate::request::Request;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// An explicit excited node (run.rs `Key`, `Choice::explicit`): its level, the absorbed component, a source's ground state.
pub(crate) type Key = (usize, crate::model::Component, usize);

/// A set's decision: its explicit nodes and the terms it keeps (`Choice::toned`), each sorted.
pub(crate) type Decided = (Vec<Key>, Vec<Term>);

/// A point's reading of one set on the decision it was read on (`Search::settle`): its reading (rate units; infinite where
/// it asks for what cannot join or its fold does not exist with none to raise), beyond the share with no node offered,
/// asking for what cannot join, no fold and no node.
type Answer = (f64, bool, bool, bool);

/// A static coupling of two ground states (module documentation, # The terms, G): (a, b, β), a < b, static in the frames
/// where θ_a − θ_b = β exactly.
pub(crate) type Link = (usize, usize, Beat);

/// A copy's population at mixing m, the dressed two-state admixture ½(1 − 1/√(1 + 4m²)) (½ where m is infinite: a
/// coupling at no mismatch).
fn admixture(m: f64) -> f64 {
    if m.is_finite() {
        0.5 * (1.0 - 1.0 / (1.0 + 4.0 * m * m).sqrt())
    } else {
        0.5
    }
}

/// The one reading of a term of mixing m (module documentation, # The terms): rate(√w(m)), rate units.
fn reading(m: f64) -> f64 {
    rate(admixture(m).sqrt())
}

/// The fold of every ground state on no link (`Choice::skeleton`): the light at a point for the G terms (`links`).
fn whole(request: &Request) -> Result<Fold, String> {
    Fold::build(request, &Choice::skeleton(request, &[])?, &(0..GROUND).collect::<Vec<_>>())
}

/// The G terms of every two ground states at a point (module documentation, # The terms, G), read on the light there by
/// the fold of every ground state `whole` (its `local` the point's light), its momentum `kappa`: each link with its m.
pub(crate) fn links(whole: &Fold, kappa: [f64; 3]) -> Vec<(Link, f64)> {
    let mut work = 0.0;
    let site = whole.site([0.0; 3], kappa, &mut work);
    let zero = whole.frames[0].plus(-1, &whole.frames[0]);
    let dot = |a: &[f64; 3], b: &[f64; 3]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    // Per direction t ← s and beat: the sum of its routes' a/|Δ|.
    let mut directed: HashMap<(usize, usize, Beat), f64> = HashMap::new();
    for (f, cf) in whole.components.iter().enumerate() {
        // The absorption of beam j of f from s: per excited state its drive over its optical detuning.
        let absorbed: Vec<Vec<Vec<(usize, f64)>>> = (0..cf.beams.len())
            .map(|j| {
                let fl = cf.factor(site.a[cf.beams[j]].value);
                (0..GROUND)
                    .map(|s| {
                        let detuning = C64::real(site.ground[(s, s)].re + cf.nu - site.doppler[f][j]);
                        (0..EXCITED)
                            .filter_map(|x| {
                                let a = (fl * cf.couplings[j][(x, s)]).abs();
                                (a > 0.0).then(|| (x, a / (detuning - site.excited[(x, x)]).abs()))
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();
        for cg in &whole.components {
            let mut sum = [[0.0_f64; GROUND]; GROUND];
            for (j, by_state) in absorbed.iter().enumerate() {
                for (jj, &ll) in cg.beams.iter().enumerate() {
                    let gl = cg.factor(site.a[ll].value);
                    let dq: [f64; 3] = std::array::from_fn(|i| cf.kicks[j][i] - cg.kicks[jj][i]);
                    let kinetic = whole.hbar_over_mass * (dot(&dq, &kappa) + 0.5 * dot(&dq, &dq));
                    for (s, u) in by_state.iter().enumerate().filter(|(_, u)| !u.is_empty()) {
                        for t in (0..GROUND).filter(|&t| t != s) {
                            let a: f64 = u.iter().map(|&(x, ux)| ux * (gl * cg.couplings[jj][(x, t)]).abs()).sum();
                            if a > 0.0 {
                                let delta = site.ground[(t, t)].re - site.ground[(s, s)].re - (cf.nu - cg.nu) + kinetic;
                                sum[t][s] += a / delta.abs();
                            }
                        }
                    }
                }
            }
            let beta = cf.beat.plus(-1, &cg.beat);
            for (t, row) in sum.iter().enumerate() {
                for (s, &m) in row.iter().enumerate().filter(|(_, m)| **m > 0.0) {
                    *directed.entry((t, s, beta.clone())).or_insert(0.0) += m;
                }
            }
        }
    }
    for s in 0..GROUND {
        for t in (0..GROUND).filter(|&t| t != s) {
            let a = site.ground[(t, s)].abs();
            if a > 0.0 {
                *directed.entry((t, s, zero.clone())).or_insert(0.0) += a / (site.ground[(t, t)].re - site.ground[(s, s)].re).abs();
            }
        }
    }
    let mut out: Vec<(Link, f64)> = vec![];
    for ((t, s, beta), m) in directed {
        let link = if t < s { (t, s, beta) } else { (s, t, beta.negated()) };
        match out.iter_mut().find(|x| x.0 == link) {
            Some(x) => x.1 = x.1.max(m),
            None => out.push((link, m)),
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// The links beyond the share among `read` with their mixing, as the skeleton takes them (`Choice::skeleton`).
fn kept(read: &[(Link, f64)]) -> Vec<(usize, usize, Beat, f64)> {
    let share = crate::model::elim::share();
    read.iter().filter(|x| reading(x.1) > share).map(|(l, m)| (l.0, l.1, l.2.clone(), *m)).collect()
}

/// The links read at one point `local` alone (module documentation, # The terms, G), each with its m and its reading.
pub(crate) fn links_at(request: &Request, local: &Local) -> Result<Vec<(Link, f64, f64)>, String> {
    let mut whole = whole(request)?;
    whole.local = local.factors(&Light::new(request));
    Ok(links(&whole, local.kappa).into_iter().map(|(l, m)| (l, m, reading(m))).collect())
}

/// The skeleton of the links read at one point `local` alone (module documentation, # The terms, G): the decision's rule
/// at that point -- the reference's at the focus (run.rs `Reference::states`: a reference is exact for any potential, so
/// its structure needs no worst case), the tests' fixture's.
pub(crate) fn skeleton_at(request: &Request, local: &Local) -> Result<Choice, String> {
    let read: Vec<(Link, f64)> = links_at(request, local)?.into_iter().map(|x| (x.0, x.1)).collect();
    Choice::skeleton(request, &kept(&read))
}

/// The dressed two-state admixture (`admixture`), for the tests' measures.
#[cfg(test)]
pub(crate) fn admixture_of(m: f64) -> f64 {
    admixture(m)
}

/// The skeleton at the focus (`skeleton_at`, `Local::focus`).
pub(crate) fn focus(request: &Request) -> Result<Choice, String> {
    skeleton_at(request, &Local::focus(&Light::new(request)))
}

/// Every term of a set's fold at a point (module documentation, # The terms), its light the fold's `local` and its
/// momentum `kappa`, with its reading (rate units): one entry per term, the largest where two name one.
pub(crate) fn terms(fold: &Fold, kappa: [f64; 3]) -> Vec<(Term, f64)> {
    let mut work = 0.0;
    let site = fold.site([0.0; 3], kappa, &mut work);
    let (ng, n) = (fold.ground.len(), fold.size());
    let excited = super::EXCITED;
    let mut out: Vec<(Term, f64)> = vec![];
    let mut add = |term: Term, m: f64| {
        if m > 0.0 {
            let r = reading(m);
            match out.iter_mut().find(|x| x.0 == term) {
                Some(x) => x.1 = x.1.max(r),
                None => out.push((term, r)),
            }
        }
    };
    // The decay's cross term of two excited states, Σ_q (C_q†C_q)_xy.
    let decay = |x: usize, y: usize| -> crate::linalg::C64 { fold.emission.iter().map(|c| (0..GROUND).map(|g| c[(g, x)].conj() * c[(g, y)]).sum::<crate::linalg::C64>()).sum() };
    let ground_pair = |s: usize, t: usize| site.ground[(s, t)].abs() / (site.ground[(s, s)].re - site.ground[(t, t)].re).abs();
    // P's entries of two places of one kind and different frames.
    for p in 0..n {
        for q in (p + 1..n).filter(|&q| (p < ng) == (q < ng) && fold.frames[p] != fold.frames[q]) {
            if p < ng {
                add(Term::pair(fold.ground[p], fold.ground[q]), ground_pair(fold.ground[p], fold.ground[q]));
            } else {
                let (x, y) = (fold.explicit[p - ng], fold.explicit[q - ng]);
                let a = fold.hermitian_excited(&site.excited, p, q).abs() + 0.5 * decay(x, y).abs();
                add(Term::pair(GROUND + x, GROUND + y), a / (site.excited[(x, x)].re - site.excited[(y, y)].re).abs());
            }
        }
    }
    // A copy of a ground state at a P place's frame: the place's entry with it.
    for b in &fold.blocks {
        for r in &b.rows {
            if let super::Row::Copy { state, component } = *r {
                let omega = b.offset.plus(-1, &fold.components[component].beat);
                for p in (0..ng).filter(|&p| fold.frames[p] == omega && fold.ground[p] != state) {
                    add(Term::pair(state, fold.ground[p]), ground_pair(state, fold.ground[p]));
                }
            }
        }
    }
    // A block's sources' mixing of each excited state, per beam at its optical detuning there.
    let mixing = |b: &super::Block, x: usize| -> f64 {
        let mut m = 0.0;
        for &(p, c) in &b.sources {
            let (s, comp) = (fold.ground[p], &fold.components[c]);
            for (j, (&l, coupling)) in comp.beams.iter().zip(&comp.couplings).enumerate() {
                let a = (comp.factor(site.a[l].value) * coupling[(x, s)]).abs();
                if a > 0.0 {
                    m += a / (crate::linalg::C64::real(site.ground[(s, s)].re + comp.nu - site.doppler[c][j]) - site.excited[(x, x)]).abs();
                }
            }
        }
        m
    };
    // An explicit state's copy at another block's offset.
    for b in &fold.blocks {
        for (j, &x) in fold.explicit.iter().enumerate().filter(|(j, _)| fold.frames[ng + j] != b.offset) {
            let _ = j;
            add(Term::Drive(x, b.offset.clone()), mixing(b, x));
        }
    }
    // Two blocks' cross decay.
    let nb = fold.blocks.len();
    let landing: Vec<Vec<f64>> = fold
        .blocks
        .iter()
        .map(|b| {
            let e: Vec<f64> = (0..excited).map(|x| if b.explicit.iter().any(|&p| fold.explicit[p - ng] == x) { 1.0 } else { mixing(b, x) }).collect();
            (0..3 * GROUND).map(|k| (0..excited).map(|x| fold.emission[k / GROUND][(k % GROUND, x)].abs() * e[x]).sum::<f64>()).collect()
        })
        .collect();
    for i in 0..nb {
        for j in (i + 1..nb).filter(|&j| fold.crossing[i * nb + j]) {
            let a = 0.5 * landing[i].iter().zip(&landing[j]).map(|(x, y)| x * y).sum::<f64>();
            add(Term::cross(&fold.blocks[i].offset, &fold.blocks[j].offset), a / (fold.blocks[i].value - fold.blocks[j].value).abs());
        }
    }
    out
}

/// The terms of a request's sets on the explicit nodes `explicit` read at one point `local` (decide.rs # The terms), each
/// beyond the share kept and the sets read again until none is added: the decision's loop at one point (tests: the
/// focus's, `Local::focus`).
pub(crate) fn at_point(request: &Request, explicit: Vec<Key>, local: &Local) -> Result<Choice, String> {
    let share = crate::model::elim::share();
    let light = Light::new(request);
    let skeleton = skeleton_at(request, local)?;
    let mut toned: Vec<Term> = vec![];
    loop {
        let choice = Choice::base_of(request, &skeleton, explicit.clone(), toned.clone(), &|_| true)?;
        let mut more = false;
        for set in Fold::sets(request, &choice)? {
            let mut fold = Fold::new(request, &choice, set[0])?;
            fold.local = local.factors(&light);
            for (term, r) in terms(&fold, local.kappa) {
                if r > share && !toned.contains(&term) {
                    toned.push(term);
                    more = true;
                }
            }
        }
        if !more {
            return Ok(choice);
        }
        toned.sort();
    }
}

/// A point of the local physics (module documentation): the trap light's relative intensity, every beam's phase (zero for
/// each frequency's first), the momentum κ (rad/µm: the velocity over ħ/m).
#[derive(Clone, Debug, Default)]
pub(crate) struct Local {
    pub(crate) s: f64,
    pub(crate) phases: Vec<f64>,
    pub(crate) kappa: [f64; 3],
}

impl Local {
    /// The focus: every beam at its focus value, in phase, at rest.
    pub(crate) fn focus(light: &Light) -> Local {
        Local { s: 1.0, phases: vec![0.0; light.beams.len()], kappa: [0.0; 3] }
    }

    /// Its coordinates' bits: the point's identity (`Search::asked`).
    fn bits(&self) -> Vec<u64> {
        std::iter::once(self.s).chain(self.phases.iter().copied()).chain(self.kappa).map(f64::to_bits).collect()
    }

    /// Every beam's factor on its amplitude (fold.rs `Fold::local`): √s for the trap's beams, times its phase.
    pub(crate) fn factors(&self, light: &Light) -> Vec<C64> {
        (0..light.beams.len())
            .map(|l| {
                let trap = light.frequencies[light.trap].beams.contains(&l);
                C64::cis(self.phases[l]) * if trap { self.s.sqrt() } else { 1.0 }
            })
            .collect()
    }
}

/// The decision of one ground set (`Decision`): its ground states, its explicit nodes, its worst point on the final
/// decision and the reading there (rate units); the points beyond the share where no node is offered; the nodes asked for
/// whose cluster cannot be made with the others (a state explicit at two offsets: run.rs `Clusters::cluster_with`), each
/// with the count of the points that asked for it, and the points of the final reading that ask for one with the first of
/// them; and the points where its fold does not exist and nothing is left to raise.
#[derive(Clone, Debug, Default)]
pub(crate) struct SetDecision {
    pub(crate) ground: Vec<usize>,
    pub(crate) nodes: Vec<Key>,
    /// The terms it keeps (`Choice::toned`), sorted; those asked for whose cluster cannot be made with the others, each
    /// with the count of the points that asked; the largest reading of a term within the share on the final decision.
    pub(crate) toned: Vec<Term>,
    pub(crate) refused_terms: Vec<(Term, usize)>,
    pub(crate) terms_worst: f64,
    pub(crate) worst: Option<(Local, f64)>,
    pub(crate) beyond: usize,
    pub(crate) refused: Vec<(Key, usize)>,
    pub(crate) unheld: usize,
    pub(crate) unheld_at: Option<Local>,
    pub(crate) nowhere: usize,
    /// Its same-state pairs' metric cross reading (`crossed`): the largest over the points read and where, and the count
    /// of the readings beyond the share (`reading`; each point's every reading counted, its re-reads too).
    pub(crate) crossed: (f64, Option<Local>),
    pub(crate) crossed_beyond: usize,
    /// The links kept between its states (G, # The terms), and the largest reading of a link of one of its states
    /// within the share over the points read.
    pub(crate) links: usize,
    pub(crate) links_worst: f64,
}

/// The start's decision (module documentation): per ground set its own, the rounds (the points read in each and the
/// explicit nodes after it), the points read, the seconds.
#[derive(Clone, Debug, Default)]
pub(crate) struct Decision {
    /// The skeleton's frames (`Choice::skeleton` on the links kept: G, # The terms).
    pub(crate) frames: Vec<Beat>,
    pub(crate) sets: Vec<SetDecision>,
    pub(crate) rounds: Vec<(usize, usize)>,
    pub(crate) points: usize,
    pub(crate) seconds: f64,
    pub(crate) dimensions: usize,
}

impl Decision {
    /// Every set's explicit nodes, one list (`Choice::base`'s).
    pub(crate) fn explicit(&self) -> Vec<Key> {
        let mut out: Vec<Key> = self.sets.iter().flat_map(|s| s.nodes.iter().copied()).collect();
        out.sort();
        out
    }

    /// Every set's decision (`Search::merged`'s input).
    pub(crate) fn decided(&self) -> Vec<Decided> {
        self.sets.iter().map(|s| (s.nodes.clone(), s.toned.clone())).collect()
    }
}

/// The coordinates of the local physics (module documentation): s, then per frequency its beams but the first (their
/// phases), then the momentum's three -- and a unit-cube point's `Local`, the momentum in the ball of radius `most`
/// (rad/µm), uniform in its volume.
struct Coordinates {
    phased: Vec<usize>,
    beams: usize,
    most: f64,
}

impl Coordinates {
    fn new(light: &Light, most: f64) -> Coordinates {
        let phased = light.frequencies.iter().flat_map(|f| f.beams.iter().skip(1).copied()).collect();
        Coordinates { phased, beams: light.beams.len(), most }
    }

    fn dimensions(&self) -> usize {
        1 + self.phased.len() + 3
    }

    fn local(&self, u: &[f64]) -> Local {
        let mut phases = vec![0.0; self.beams];
        for (k, &l) in self.phased.iter().enumerate() {
            phases[l] = std::f64::consts::TAU * u[1 + k];
        }
        let v = &u[1 + self.phased.len()..];
        let (z, angle) = (2.0 * v[0] - 1.0, std::f64::consts::TAU * v[1]);
        let rho = (1.0 - z * z).max(0.0).sqrt();
        let radius = self.most * v[2].cbrt();
        Local { s: u[0], phases, kappa: [radius * rho * angle.cos(), radius * rho * angle.sin(), radius * z] }
    }
}

/// A Sobol sequence (Joe and Kuo's direction numbers, new-joe-kuo-6.21201, its first 21 dimensions), point i in Gray-code
/// order: every block of 2^m points from the first the same net.
struct Sobol {
    directions: Vec<[u32; 32]>,
}

/// Per dimension after the first: its primitive polynomial's degree s and coefficients a, and its initial m_1 ... m_s.
const JOE_KUO: [(u32, u32, &[u32]); 20] = [
    (1, 0, &[1]),
    (2, 1, &[1, 3]),
    (3, 1, &[1, 3, 1]),
    (3, 2, &[1, 1, 1]),
    (4, 1, &[1, 1, 3, 3]),
    (4, 4, &[1, 3, 5, 13]),
    (5, 2, &[1, 1, 5, 5, 17]),
    (5, 4, &[1, 1, 5, 5, 5]),
    (5, 7, &[1, 1, 7, 11, 19]),
    (5, 11, &[1, 1, 5, 1, 1]),
    (5, 13, &[1, 1, 1, 3, 11]),
    (5, 14, &[1, 3, 5, 5, 31]),
    (6, 1, &[1, 3, 3, 9, 7, 49]),
    (6, 13, &[1, 1, 1, 15, 21, 21]),
    (6, 16, &[1, 3, 1, 13, 27, 49]),
    (6, 19, &[1, 1, 1, 15, 7, 5]),
    (6, 22, &[1, 3, 1, 15, 13, 25]),
    (6, 25, &[1, 1, 5, 5, 19, 61]),
    (7, 1, &[1, 3, 7, 11, 23, 15, 103]),
    (7, 4, &[1, 3, 7, 13, 13, 15, 69]),
];

impl Sobol {
    fn new(dimensions: usize) -> Result<Sobol, String> {
        if dimensions > 1 + JOE_KUO.len() {
            return Err(format!("开头决定：局部量 {dimensions} 维，超出 Sobol 方向数表的 {} 维", 1 + JOE_KUO.len()));
        }
        let mut directions = vec![std::array::from_fn(|j| 1u32 << (31 - j))];
        for &(s, a, m) in JOE_KUO.iter().take(dimensions.saturating_sub(1)) {
            let s = s as usize;
            let mut v = [0u32; 32];
            for j in 0..32 {
                v[j] = if j < s {
                    m[j] << (31 - j)
                } else {
                    let mut x = v[j - s] ^ (v[j - s] >> s);
                    for k in 1..s {
                        x ^= ((a >> (s - 1 - k)) & 1) * v[j - k];
                    }
                    x
                };
            }
            directions.push(v);
        }
        Ok(Sobol { directions })
    }

    /// Point i (i < 2³²) in [0, 1)^d.
    fn point(&self, i: u64) -> Vec<f64> {
        let gray = i ^ (i >> 1);
        self.directions
            .iter()
            .map(|v| {
                let x = (0..32).filter(|&j| gray >> j & 1 == 1).fold(0u32, |x, j| x ^ v[j]);
                x as f64 / 4294967296.0
            })
            .collect()
    }
}

/// What a point asks of one set's decision: nothing (its reading, rate units; whether it stays beyond the share with no
/// node offered; its same-state pairs' metric cross reading, `Search::crossed`; its terms' largest reading), terms beyond
/// the share (# The terms), a node (its key), or nothing where its fold does not exist and no node is left to raise.
#[derive(Clone)]
enum Asked {
    Within(f64, bool, f64, f64),
    Terms(Vec<Term>),
    Node(Key),
    Nowhere,
}

/// One set's fold on a decision (module documentation, # What is made once): its choice (`Choice::base_of` with those
/// nodes, its own set's tiers), its fold and candidates, and the threads' copies of the fold with their workspaces, taken
/// and given back.
struct Built {
    choice: Choice,
    fold: Fold,
    candidates: Vec<(Key, Taken)>,
    pool: Mutex<Vec<(Fold, Workspace)>>,
}

impl Built {
    /// A copy of the fold with a workspace: one given back, else made.
    fn take(&self) -> (Fold, Workspace) {
        let taken = self.pool.lock().unwrap_or_else(|e| e.into_inner()).pop();
        taken.unwrap_or_else(|| (self.fold.clone(), self.fold.prototype_workspace().expect("a built fold's workspace")))
    }

    fn give(&self, held: (Fold, Workspace)) {
        self.pool.lock().unwrap_or_else(|e| e.into_inner()).push(held);
    }
}

/// A set's fold on a decision, once made: Err where its cluster cannot be made (run.rs `Clusters::cluster_with`).
type Slot = Arc<OnceLock<Result<Arc<Built>, String>>>;

/// The metric cross reading of a fold's same-state node pairs at its last forward (FOLDAPI §2; dragchk.status 第二项,
/// 给 A 的实现说明 4): one physical excited state at two offsets -- an explicit place and a Q row of another block, or Q
/// rows of two blocks -- is two nodes whose cross term in the physical norm, Σ e^{i(o − o′)t} ε_n†ε_n′, the metric drops
/// (M8: the blocks apart, `Workspace::populations`; the beating metric). Its size, the coupling's mixing in the rule's
/// reading (dragchk (B1); in the static metric's gauge the Hamiltonian cross term's a/|Δ|), is ‖ε_n‖ ‖ε_n′‖ on the P
/// columns (an explicit place's row the unit one): the largest over every pair. Within the share it is averaged out (the
/// blocks apart); beyond it the pair's cross terms must be kept in the transients (dragchk (B3)), which this fold does not
/// hold: the decision reports it.
fn crossed(fold: &Fold, cache: &Cache, ws: &super::proto::Workspace) -> f64 {
    let n = fold.size();
    // Every block's rows' ε on the P columns: per block, per row, its squared norm over the columns.
    let mut norms: Vec<Vec<f64>> = fold.blocks.iter().map(|_| vec![]).collect();
    for j in 0..n {
        let mut x = vec![C64::ZERO; n];
        x[j] = C64::ONE;
        for (o, rows) in ws.amplitudes(fold, cache, &x).into_iter().enumerate() {
            norms[o].resize(rows.len(), 0.0);
            for (r, z) in rows.into_iter().enumerate() {
                norms[o][r] += z.norm_sqr();
            }
        }
    }
    // Per excited state its nodes' norms: its explicit place (one), and each block's row of it.
    let mut nodes: Vec<Vec<(usize, f64)>> = vec![vec![]; crate::request::STATES];
    for p in fold.ground.len()..n {
        nodes[fold.place_state(p)].push((usize::MAX, 1.0));
    }
    for (o, rows) in norms.iter().enumerate() {
        for (r, &w) in rows.iter().enumerate() {
            let state = fold.row_state(o, r);
            if state >= crate::request::GROUND {
                nodes[state].push((o, w.sqrt()));
            }
        }
    }
    let mut most = 0.0_f64;
    for of in &nodes {
        for (a, &(oa, wa)) in of.iter().enumerate() {
            for &(ob, wb) in &of[a + 1..] {
                if oa != ob {
                    most = most.max(wa * wb);
                }
            }
        }
    }
    most
}

/// The decision's inputs (module documentation), and what it makes once and keeps (# What is made once): each set's
/// fold per decision (`Built`) on the skeleton so far (# The terms, G), the points read and their G terms.
pub(crate) struct Search<'a> {
    request: &'a Request,
    light: &'a Light,
    /// The skeleton on the links kept so far (`Choice::skeleton`), its ground sets (`Fold::sets`), the links with their
    /// largest mixing over the points read.
    skeleton: Choice,
    grounds: Vec<Vec<usize>>,
    links: Vec<(Link, f64)>,
    /// Every ground state's fold (`whole`): the G terms' light.
    whole: Fold,
    /// The points read so far, in their order, with their indices by their bits, each one's G terms, and its reading per
    /// set on the decision it was last read on (`settle`); every link's largest m over them.
    seen: Vec<Local>,
    index: HashMap<Vec<u64>, usize>,
    read: HashMap<Vec<u64>, Vec<(Link, f64)>>,
    largest: HashMap<Link, f64>,
    rows: Vec<Vec<Answer>>,
    /// The potential channel's floor (rad/µs): every reading's unit.
    floor: f64,
    /// The largest momentum an atom held by the trap has (rad/µm).
    most: f64,
    built: Mutex<Vec<((usize, Decided), Slot)>>,
    /// What each point asked of each set's decision (module documentation, # What is made once), by the point's bits
    /// (`Local::bits`), the set and its nodes.
    asked: Mutex<HashMap<(Vec<u64>, usize, Decided), Asked>>,
}

impl<'a> Search<'a> {
    /// The search of a request's run (module documentation): its light, the potential channel's floor (rad/µs) and the
    /// largest momentum an atom held by the trap has (rad/µm); no link kept yet, every ground state its own set.
    pub(crate) fn new(request: &'a Request, light: &'a Light, floor: f64, most: f64) -> Result<Search<'a>, String> {
        let skeleton = Choice::skeleton(request, &[])?;
        let grounds = Fold::sets(request, &skeleton)?;
        let whole = whole(request)?;
        Ok(Search { request, light, skeleton, grounds, links: vec![], whole, seen: vec![], index: HashMap::new(), read: HashMap::new(), largest: HashMap::new(), rows: vec![], floor, most, built: Default::default(), asked: Default::default() })
    }

    /// One choice of every set's explicit nodes `nodes`, made of each set's own (`Built`, made once and at once): the
    /// rule's sets and frames, every set's explicit nodes (one list in its order),
    /// and per set its tiers, its own keys' (a key's source state is its set's) in its choice's order -- each set's fold
    /// on it the one on its own choice (`Fold::build` reads a set's keys alone): what `Choice::base_of` makes of all the
    /// sets at once (run.rs `Onefold::new`).
    pub(crate) fn merged(&self, nodes: &[Decided]) -> Result<Choice, String> {
        self.build(&nodes.iter().cloned().enumerate().collect::<Vec<_>>());
        let built: Vec<Arc<Built>> = nodes.iter().enumerate().map(|(k, x)| self.built(k, x)).collect::<Result<_, _>>()?;
        let first = &built.first().ok_or("没有基态集合")?.choice;
        let mut explicit: Vec<Key> = nodes.iter().flat_map(|x| x.0.iter().copied()).collect();
        explicit.sort();
        let mut toned: Vec<Term> = nodes.iter().flat_map(|x| x.1.iter().cloned()).collect();
        toned.sort();
        toned.dedup();
        let mut out = Choice { frames: first.frames.clone(), sets: first.sets.clone(), explicit, copies: vec![], excited: vec![], toned };
        for (b, ground) in built.iter().zip(&self.grounds) {
            out.copies.extend(b.choice.copies.iter().filter(|x| ground.contains(&x.0 .1)).cloned());
            out.excited.extend(b.choice.excited.iter().filter(|x| ground.contains(&x.0 .1)).cloned());
        }
        Ok(out)
    }

    /// Set k's fold with the explicit nodes `nodes` (`Built`: `Choice::base_of`, its own set's tiers; the fold and its
    /// workspace), made once -- the first thread to ask makes it, the others wait for it.
    fn built(&self, k: usize, nodes: &Decided) -> Result<Arc<Built>, String> {
        let slot: Slot = {
            let mut built = self.built.lock().unwrap_or_else(|e| e.into_inner());
            match built.iter().find(|x| x.0 .0 == k && x.0 .1 == *nodes) {
                Some(x) => x.1.clone(),
                None => {
                    let slot: Slot = Default::default();
                    built.push(((k, nodes.clone()), slot.clone()));
                    slot
                }
            }
        };
        slot.get_or_init(|| {
            let choice = Choice::base_of(self.request, &self.skeleton, nodes.0.clone(), nodes.1.clone(), &|set| set == &self.grounds[k][..])?;
            let fold = Fold::new(self.request, &choice, self.grounds[k][0])?;
            let ws = fold.prototype_workspace()?;
            let candidates = Choice::candidates(&fold).into_iter().map(|(key, block, rows)| (key, Taken { block, rows })).collect();
            Ok(Arc::new(Built { choice, pool: Mutex::new(vec![(fold.clone(), ws)]), fold, candidates }))
        })
        .clone()
    }

    /// Every set's fold on its nodes `wanted` (set, nodes) made at once, a thread each.
    fn build(&self, wanted: &[(usize, Decided)]) {
        std::thread::scope(|scope| {
            for (k, nodes) in wanted {
                scope.spawn(move || {
                    let _ = self.built(*k, nodes);
                });
            }
        });
    }

    /// Whether set k's nodes `nodes` are placed (`Fold::build`: a node holding a state explicit at another offset is
    /// refused there) -- what can refuse a cluster before its tiers are read; read on the skeleton, as the tiers' first
    /// fold is (`Choice::tiered_of`).
    fn placed(&self, k: usize, nodes: &Vec<Key>) -> bool {
        let choice = Choice { explicit: nodes.clone(), ..self.skeleton.clone() };
        Fold::sets(self.request, &choice).is_ok() && Fold::new(self.request, &choice, self.grounds[k][0]).is_ok()
    }

    /// What the point `local` asks of a set's fold `built`, read on the thread's copy `held` (`Asked`; module
    /// documentation, # At each point).
    fn ask(&self, built: &Result<Arc<Built>, String>, held: Option<&mut (Fold, Workspace)>, local: &Local, refused: &[Term]) -> Asked {
        let (Ok(built), Some((fold, ws))) = (built, held) else { return Asked::Nowhere };
        let candidates = &built.candidates;
        if fold.blocks.is_empty() {
            return Asked::Within(0.0, false, 0.0, 0.0);
        }
        fold.local = local.factors(self.light);
        // The terms first (# The terms): those beyond the share the decision does not keep and has not refused.
        let share = crate::model::elim::share();
        let mut worst = 0.0_f64;
        let mut asks: Vec<Term> = vec![];
        for (term, r) in terms(fold, local.kappa) {
            if built.choice.toned.contains(&term) {
                continue;
            }
            if r > share && !refused.contains(&term) {
                asks.push(term);
            } else {
                worst = worst.max(r);
            }
        }
        if !asks.is_empty() {
            return Asked::Terms(asks);
        }
        ws.cell = Cell { nodes: candidates.iter().map(|c| c.1.clone()).collect(), here: true, ..Cell::default() };
        let mut cache = Cache::default();
        let step = fold.prototype([0.0; 3], local.kappa, self.floor, &mut cache, ws).map(|s| (s.reading, s.beyond));
        let mut chosen = ws.cell.chosen;
        if let (Some(_), None) = (step, chosen) {
            if !fold.prototype_left(&cache, ws) {
                chosen = ws.attributed(&cache);
            }
        }
        match (step, chosen) {
            (_, Some(c)) => Asked::Node(candidates[c].0),
            (Some((reading, beyond)), None) => Asked::Within(reading, beyond, crossed(fold, &cache, ws), worst),
            (None, None) => Asked::Nowhere,
        }
    }

    /// The points `points` read on the decision `nodes` (per set), in parallel: per point and set what it asks.
    fn round(&self, points: &[Local], nodes: &[Decided], refused: &[Vec<Term>]) -> Vec<Vec<Asked>> {
        self.build(&nodes.iter().cloned().enumerate().collect::<Vec<_>>());
        let built: Vec<Result<Arc<Built>, String>> = nodes.iter().enumerate().map(|(k, x)| self.built(k, x)).collect();
        let built = &built;
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(points.len().max(1));
        let chunk = points.len().div_ceil(threads);
        std::thread::scope(|scope| {
            let handles: Vec<_> = points
                .chunks(chunk.max(1))
                .map(|part| {
                    scope.spawn(move || {
                        // The thread's copies, taken where a point is read on them first.
                        let mut held: Vec<Option<(Fold, Workspace)>> = built.iter().map(|_| None).collect();
                        let mut answer = |k: usize, local: &Local| -> Asked {
                            let key = (local.bits(), k, nodes[k].clone());
                            if let Some(a) = self.asked.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
                                return a.clone();
                            }
                            if held[k].is_none() {
                                held[k] = built[k].as_ref().ok().map(|b| b.take());
                            }
                            let a = self.ask(&built[k], held[k].as_mut(), local, &refused[k]);
                            self.asked.lock().unwrap_or_else(|e| e.into_inner()).insert(key, a.clone());
                            a
                        };
                        let out = part.iter().map(|local| (0..nodes.len()).map(|k| answer(k, local)).collect::<Vec<Asked>>()).collect::<Vec<_>>();
                        for (b, h) in built.iter().zip(held) {
                            if let (Ok(b), Some(h)) = (b, h) {
                                b.give(h);
                            }
                        }
                        out
                    })
                })
                .collect();
            handles.into_iter().flat_map(|h| h.join().expect("a decision's thread")).collect()
        })
    }

    /// Whether set k's cluster with the nodes `nodes` can be made (run.rs `Clusters::cluster_with`: refused where a state
    /// is explicit at two offsets, or a block is not the prototype's).
    fn feasible(&self, k: usize, nodes: &Decided) -> bool {
        self.built(k, nodes).is_ok()
    }

    /// The keys `keys` joined to a set's decision `set` in their order (`settle`), each where `can` makes its cluster with
    /// those before it, else refused: the nodes, and the keys refused.
    fn join(&self, keys: &[Key], set: &SetDecision, can: &dyn Fn(&Vec<Key>) -> bool) -> (Vec<Key>, Vec<Key>) {
        let mut now = set.nodes.clone();
        let mut refused: Vec<Key> = vec![];
        for &key in keys {
            if now.contains(&key) || set.refused.iter().any(|r| r.0 == key) || refused.contains(&key) {
                continue;
            }
            let mut more = now.clone();
            more.push(key);
            more.sort();
            if can(&more) {
                now = more;
            } else {
                refused.push(key);
            }
        }
        (now, refused)
    }

    /// The points read until none asks for a node that can join (`round`; the nodes asked for joining at once, in their
    /// keys' order, each where its cluster can be made with those before it -- else refused, kept so --, the points
    /// whose node's level had a node join read again: every read again follows a join, so the reading ends), into the
    /// decision; each point's reading per set on the final decision: (its
    /// reading, rate units, infinite where it asked for a refused node or its fold does not exist with none to raise;
    /// beyond the share with no node offered; asked for a refused node; no fold and no node).
    fn settle(&mut self, points: &[Local], decision: &mut Decision) -> Vec<usize> {
        // The points' indices among those seen (a point seen before is the same point).
        let mut indices: Vec<usize> = vec![];
        for local in points {
            let at = *self.index.entry(local.bits()).or_insert_with(|| {
                self.seen.push(local.clone());
                self.rows.push(vec![]);
                self.seen.len() - 1
            });
            indices.push(at);
        }
        let mut open: Vec<usize> = indices.clone();
        open.sort();
        open.dedup();
        while !open.is_empty() {
            // The G terms (# The terms): every point's read once, on the light alone; where the links kept or their tree
            // move the skeleton, every point seen is read again on it.
            self.read_links(&open);
            if self.restructure(decision) {
                open = (0..self.seen.len()).collect();
            }
            let sets = decision.sets.len();
            let points: Vec<Local> = open.iter().map(|&i| self.seen[i].clone()).collect();
            let nodes: Vec<Vec<Key>> = decision.sets.iter().map(|s| s.nodes.clone()).collect();
            let toned: Vec<Vec<Term>> = decision.sets.iter().map(|s| s.toned.clone()).collect();
            let refused: Vec<Vec<Term>> = decision.sets.iter().map(|s| s.refused_terms.iter().map(|r| r.0.clone()).collect()).collect();
            let decided: Vec<Decided> = nodes.iter().cloned().zip(toned.iter().cloned()).collect();
            let asked = self.round(&points, &decided, &refused);
            decision.points += open.len();
            // The nodes asked, per set, joining in the order of the points that ask for them (the most first; a tie by the
            // keys' order), each where its cluster can be made: a node is a level at an offset holding the states that
            // offset reaches (fold.rs `Fold::build`), and one excited state is one place at one frame -- a node that would
            // hold a state already explicit at another offset is refused (its drive there is the one rule's, fold.rs
            // `toned`; the candidates offer no such node, `Choice::candidates`, so only two keys of one round that each
            // hold the state meet it). The joins planned on the placements, each set's last decision made, the sets' at
            // once (# What is made once).
            let keys: Vec<Vec<Key>> = (0..sets)
                .map(|k| {
                    let mut asks: Vec<Key> = asked.iter().filter_map(|a| if let Asked::Node(key) = a[k] { Some(key) } else { None }).collect();
                    asks.sort();
                    let mut keys: Vec<(Key, usize)> = vec![];
                    for key in asks {
                        match keys.iter_mut().find(|x| x.0 == key) {
                            Some(x) => x.1 += 1,
                            None => keys.push((key, 1)),
                        }
                    }
                    keys.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
                    keys.into_iter().map(|x| x.0).collect()
                })
                .collect();
            // The terms asked, per set, joining at once with the nodes (each set's terms in their order, sorted); where the
            // cluster cannot be made with them all, one by one, each where it can be made with those before it, else
            // refused and kept so.
            let terms: Vec<Vec<Term>> = (0..sets)
                .map(|k| {
                    let mut asks: Vec<Term> = asked.iter().filter_map(|a| if let Asked::Terms(t) = &a[k] { Some(t.clone()) } else { None }).flatten().collect();
                    asks.sort();
                    asks.dedup();
                    asks
                })
                .collect();
            let with = |k: usize, more: &[Term]| -> Vec<Term> {
                let mut t: Vec<Term> = toned[k].iter().chain(more).cloned().collect();
                t.sort();
                t.dedup();
                t
            };
            let plans: Vec<(Vec<Key>, Vec<Key>)> = (0..sets).map(|k| self.join(&keys[k], &decision.sets[k], &|x| self.placed(k, x))).collect();
            self.build(&plans.iter().enumerate().map(|(k, p)| (k, (p.0.clone(), with(k, &terms[k])))).collect::<Vec<_>>());
            for (k, plan) in plans.into_iter().enumerate() {
                let all = with(k, &terms[k]);
                let (now, refused) = if self.feasible(k, &(plan.0.clone(), all.clone())) { plan } else { self.join(&keys[k], &decision.sets[k], &|x| self.feasible(k, &(x.clone(), toned[k].clone()))) };
                let mut kept = toned[k].clone();
                let mut refused_terms: Vec<Term> = vec![];
                if self.feasible(k, &(now.clone(), all.clone())) {
                    kept = all;
                } else {
                    for t in &terms[k] {
                        let more = with(k, &kept.iter().chain(std::iter::once(t)).cloned().collect::<Vec<_>>());
                        if self.feasible(k, &(now.clone(), more.clone())) {
                            kept = more;
                        } else {
                            refused_terms.push(t.clone());
                        }
                    }
                }
                let set = &mut decision.sets[k];
                set.nodes = now;
                set.toned = kept;
                set.refused.extend(refused.into_iter().map(|key| (key, 0)));
                set.refused_terms.extend(refused_terms.into_iter().map(|t| (t, 0)));
            }
            let mut again = vec![];
            // Whether set k had a node of the level joined in this round.
            let joined = |k: usize, level: usize, sets: &[SetDecision]| sets[k].nodes.iter().any(|n| n.0 == level && !nodes[k].contains(n));
            for ((&i, asks), point) in open.iter().zip(asked).zip(&points) {
                let mut more = false;
                let mut row = vec![(0.0, false, false, false); sets];
                for (k, ask) in asks.into_iter().enumerate() {
                    match ask {
                        Asked::Within(read, beyond, cross, worst) => {
                            row[k] = (read, beyond, false, false);
                            decision.sets[k].terms_worst = decision.sets[k].terms_worst.max(worst);
                            let set = &mut decision.sets[k];
                            if cross > set.crossed.0 {
                                set.crossed = (cross, Some(point.clone()));
                            }
                            set.crossed_beyond += usize::from(reading(cross) > crate::model::elim::share());
                        }
                        Asked::Nowhere => row[k] = (f64::INFINITY, false, false, true),
                        Asked::Terms(asks) if asks.iter().any(|t| decision.sets[k].toned.contains(t)) => more = true,
                        Asked::Terms(asks) => {
                            row[k] = (f64::INFINITY, false, true, false);
                            for r in decision.sets[k].refused_terms.iter_mut().filter(|r| asks.contains(&r.0)) {
                                r.1 += 1;
                            }
                        }
                        Asked::Node(key) if joined(k, key.0, &decision.sets) => more = true,
                        Asked::Node(key) => {
                            row[k] = (f64::INFINITY, false, true, false);
                            if let Some(r) = decision.sets[k].refused.iter_mut().find(|r| r.0 == key) {
                                r.1 += 1;
                            }
                        }
                    }
                }
                self.rows[i] = row;
                if more {
                    again.push(i);
                }
            }
            open = again;
        }
        indices
    }

    /// The G terms (module documentation, # The terms) of the points seen at `open` that are not read yet, each on the
    /// whole fold at its light, in parallel.
    fn read_links(&mut self, open: &[usize]) {
        let todo: Vec<Local> = open.iter().map(|&i| self.seen[i].clone()).filter(|l| !self.read.contains_key(&l.bits())).collect();
        if todo.is_empty() {
            return;
        }
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(todo.len());
        let chunk = todo.len().div_ceil(threads);
        let (whole, light) = (&self.whole, self.light);
        let read: Vec<(Vec<u64>, Vec<(Link, f64)>)> = std::thread::scope(|scope| {
            let handles: Vec<_> = todo
                .chunks(chunk)
                .map(|part| {
                    scope.spawn(move || {
                        let mut fold = whole.clone();
                        part.iter()
                            .map(|local| {
                                fold.local = local.factors(light);
                                (local.bits(), links(&fold, local.kappa))
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            handles.into_iter().flat_map(|h| h.join().expect("a decision's thread")).collect()
        });
        for (bits, links) in read {
            for (link, m) in &links {
                let x = self.largest.entry(link.clone()).or_insert(0.0);
                *x = x.max(*m);
            }
            self.read.insert(bits, links);
        }
    }

    /// The skeleton of the links beyond the share at a point seen, each with its largest mixing over the points seen
    /// (module documentation, # The terms, G); where its sets or frames are not the search's, the search takes it: its
    /// sets' decisions merged into the new sets' -- the nodes, refusals and P's pairs of the sets each holds, the terms
    /// an offset names (a drive's, a cross decay's) only where no state of the set moved its frame --, what was made on
    /// the old one forgotten. Whether it was taken.
    fn restructure(&mut self, decision: &mut Decision) -> bool {
        let share = crate::model::elim::share();
        let mut links: Vec<(Link, f64)> = self.largest.iter().filter(|x| reading(*x.1) > share).map(|(l, m)| (l.clone(), *m)).collect();
        links.sort_by(|a, b| a.0.cmp(&b.0));
        let skeleton = Choice::skeleton(self.request, &links.iter().map(|(l, m)| (l.0, l.1, l.2.clone(), *m)).collect::<Vec<_>>()).expect("the request's skeleton");
        let grounds = Fold::sets(self.request, &skeleton).expect("the skeleton's sets");
        let same = skeleton.frames == self.skeleton.frames && grounds == self.grounds && !decision.frames.is_empty();
        self.links = links;
        if same {
            return false;
        }
        let moved = |s: usize| decision.frames.get(s).is_some_and(|f| *f != skeleton.frames[s]);
        let sets: Vec<SetDecision> = grounds
            .iter()
            .map(|g| {
                let parts: Vec<&SetDecision> = decision.sets.iter().filter(|d| d.ground.iter().all(|s| g.contains(s))).collect();
                let still = !g.iter().any(|&s| moved(s));
                let mut nodes: Vec<Key> = parts.iter().flat_map(|d| d.nodes.iter().copied()).collect();
                nodes.sort();
                nodes.dedup();
                let mut toned: Vec<Term> = parts.iter().flat_map(|d| d.toned.iter().cloned()).filter(|t| still || matches!(t, Term::Pair(..))).collect();
                toned.sort();
                toned.dedup();
                let refused = parts.iter().flat_map(|d| d.refused.iter().copied()).collect();
                let refused_terms = parts.iter().flat_map(|d| d.refused_terms.iter().cloned()).filter(|t| still || matches!(t.0, Term::Pair(..))).collect();
                SetDecision { ground: g.clone(), nodes, toned, refused, refused_terms, ..SetDecision::default() }
            })
            .collect();
        decision.sets = sets;
        decision.frames = skeleton.frames.clone();
        self.skeleton = skeleton;
        self.grounds = grounds;
        self.built.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.asked.lock().unwrap_or_else(|e| e.into_inner()).clear();
        true
    }

    /// Per set its links kept and the largest reading of a link of its states within the share over the points seen
    /// (`SetDecision::links`, `links_worst`).
    fn links_read(&self, decision: &mut Decision) {
        let share = crate::model::elim::share();
        for set in decision.sets.iter_mut() {
            set.links = self.links.iter().filter(|(l, _)| set.ground.contains(&l.0) && set.ground.contains(&l.1)).count();
            set.links_worst = self
                .seen
                .iter()
                .flat_map(|local| self.read[&local.bits()].iter())
                .filter(|(l, _)| set.ground.contains(&l.0) || set.ground.contains(&l.1))
                .map(|x| reading(x.1))
                .filter(|&r| r <= share)
                .fold(0.0, f64::max);
        }
    }

    /// The doubling (module documentation, # The search) over the points `point(i)`: from 2^d points, doubled until the
    /// decision is the same over a doubling; the readings of every point read, with the points.
    fn doubled(&mut self, decision: &mut Decision, point: &dyn Fn(u64) -> Local) -> Vec<usize> {
        let mut read: Vec<usize> = vec![];
        let mut count = 1u64 << decision.dimensions;
        let mut from = 0u64;
        loop {
            let before = (decision.frames.clone(), decision.decided());
            let points: Vec<Local> = (from..count).map(point).collect();
            read.extend(self.settle(&points, decision));
            decision.rounds.push((points.len(), decision.sets.iter().map(|s| s.nodes.len()).sum()));
            let same = (decision.frames.clone(), decision.decided()) == before;
            if same && from > 0 {
                return read;
            }
            (from, count) = (count, 2 * count);
        }
    }

    /// Of the points seen at `indices`, the one whose reading on set k is the largest and finite, with it.
    fn worst(&self, indices: &[usize], k: usize) -> Option<(Local, f64)> {
        indices.iter().filter(|&&i| self.rows[i].get(k).is_some_and(|r| r.0.is_finite())).max_by(|&&a, &&b| self.rows[a][k].0.total_cmp(&self.rows[b][k].0)).map(|&i| (self.seen[i].clone(), self.rows[i][k].0))
    }

    /// The start's decision (module documentation).
    pub(crate) fn decide(&mut self) -> Result<Decision, String> {
        let clock = std::time::Instant::now();
        let coordinates = Coordinates::new(self.light, self.most);
        let d = coordinates.dimensions();
        let sobol = Sobol::new(d)?;
        let mut decision = Decision { sets: self.grounds.iter().map(|g| SetDecision { ground: g.clone(), ..SetDecision::default() }).collect(), dimensions: d, ..Decision::default() };
        let global = self.doubled(&mut decision, &|i| coordinates.local(&sobol.point(i)));
        // The global points' spacing; each set's worst point's box of it, searched by the same doubling (a set the box's
        // search merges reads its own box no more: the sets are the skeleton's at the box's start).
        let spacing = (global.len() as f64).powf(-1.0 / d as f64);
        let mut refined: Vec<usize> = vec![];
        let mut k = 0;
        while k < decision.sets.len() {
            let Some((at, _)) = self.worst(&global, k) else {
                k += 1;
                continue;
            };
            let centre = Search::unit(&coordinates, &at);
            let local = |i: u64| {
                let u: Vec<f64> = sobol.point(i).iter().zip(&centre).enumerate().map(|(c, (&w, &x))| Search::within(&coordinates, c, x + spacing * (2.0 * w - 1.0))).collect();
                coordinates.local(&u)
            };
            refined.extend(self.doubled(&mut decision, &local));
            k += 1;
        }
        // The readings on the final decision: every point seen, read on it (a point read on a smaller P reads on a larger
        // one again at the next round's start, and those of the last round are on it). (Where one asks for a node even
        // so, it joins: the union.)
        decision.sets.iter_mut().for_each(|s| s.terms_worst = 0.0);
        let all: Vec<usize> = global.iter().chain(&refined).copied().collect();
        let points: Vec<Local> = all.iter().map(|&i| self.seen[i].clone()).collect();
        let finals = self.settle(&points, &mut decision);
        self.links_read(&mut decision);
        for (k, set) in decision.sets.iter_mut().enumerate() {
            let row = |i: &usize| self.rows[*i][k];
            set.beyond = finals.iter().filter(|i| row(i).1).count();
            set.unheld = finals.iter().filter(|i| row(i).2).count();
            set.nowhere = finals.iter().filter(|i| row(i).3).count();
            set.worst = self.worst(&finals, k);
            set.unheld_at = finals.iter().find(|i| row(i).2).map(|&i| self.seen[i].clone());
        }
        decision.seconds = clock.elapsed().as_secs_f64();
        Ok(decision)
    }

    /// A point's unit-cube coordinates (`Coordinates::local`'s inverse).
    fn unit(coordinates: &Coordinates, at: &Local) -> Vec<f64> {
        let mut u = vec![at.s];
        u.extend(coordinates.phased.iter().map(|&l| at.phases[l] / std::f64::consts::TAU));
        let r = at.kappa.iter().map(|x| x * x).sum::<f64>().sqrt();
        let z = if r > 0.0 { at.kappa[2] / r } else { 0.0 };
        let angle = at.kappa[1].atan2(at.kappa[0]).rem_euclid(std::f64::consts::TAU);
        u.extend([(z + 1.0) / 2.0, angle / std::f64::consts::TAU, if coordinates.most > 0.0 { (r / coordinates.most).powi(3) } else { 0.0 }]);
        u
    }

    /// Coordinate c's value x taken into the cube: a phase or the momentum's azimuth turned, the others held at its edges.
    fn within(coordinates: &Coordinates, c: usize, x: f64) -> f64 {
        let turns = (1..=coordinates.phased.len()).contains(&c) || c == coordinates.dimensions() - 2;
        if turns {
            x.rem_euclid(1.0)
        } else {
            x.clamp(0.0, 1.0 - f64::EPSILON)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Sobol;

    /// Every dimension's first 2^m points take each of the 2^m intervals once (its generator's leading m × m block unit
    /// upper triangular), and every pair of the first eight dimensions' first 2^(2m+4) points each of the 2^m × 2^m
    /// squares (a (t, 2)-net with t ≤ 4 on these pairs; dimensions 1 and 7 have t = 3 at 64 points: 8 of the 16 squares).
    #[test]
    fn the_sobol_points_stratify_each_coordinate() {
        let sobol = Sobol::new(21).unwrap();
        for m in [1u32, 3, 6] {
            let n = 1u64 << m;
            for d in 0..21 {
                let mut seen = vec![0; n as usize];
                for i in 0..n {
                    seen[(sobol.point(i)[d] * n as f64) as usize] += 1;
                }
                assert!(seen.iter().all(|&c| c == 1), "dimension {d}, 2^{m} points: {seen:?}");
            }
        }
        let m = 2u32;
        let cells = 1usize << m;
        for a in 0..8 {
            for b in (a + 1)..8 {
                let mut seen = vec![0; cells * cells];
                for i in 0..(1u64 << (2 * m + 4)) {
                    let p = sobol.point(i);
                    seen[(p[a] * cells as f64) as usize * cells + (p[b] * cells as f64) as usize] += 1;
                }
                assert!(seen.iter().all(|&c| c > 0), "dimensions {a}, {b}: {seen:?}");
            }
        }
    }
}
