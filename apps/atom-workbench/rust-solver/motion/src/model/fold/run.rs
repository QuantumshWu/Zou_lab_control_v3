//! The run on the fold at the atom (DESIGN §5.3 `model/run.rs`, §15.4 step 2; BRIEF_WIRE 2a): the integrator's model
//! (`step::Dynamics`, `step::Middle`) given by the prototype (proto.rs) -- at a step's middle and its two outer nodes the
//! generator G, its jumps, its photons and its force, all of one solve and its one derivative.
//!
//! # The state sets
//! The run's ground sets are the rule's (`Choice::base`): every ground state of F = 1 and F = 2 in one vector, the sets
//! the connected parts of what cannot be folded (DESIGN §1.1, the forest). A cluster of the integrator (`Cluster`) is a
//! ground set with a set of explicit excited nodes kept in P (# The nodes risen and fallen): its ground states, then its
//! explicit excited states, each in its frame; made where a trajectory first asks for it and kept for the run (`Arena`),
//! each on its own choice (`Choice::base` with its explicit nodes). No Doppler frame is taken (each state's K_s = 0):
//! the generator at the outer nodes is the light at the path's own points there (`Fold::prototype_nodes`), its plane
//! waves turned, and the integrator takes its change through the three nodes.
//!
//! # The representation: the start's (FOLDAPI §3, §7.5; decide.rs)
//! Which excited nodes are explicit is decided once, before any trajectory, from the worst case over the atoms' local
//! physics (`Onefold::new`, decide.rs): per ground set one cluster (`Set`), the k-th cluster the k-th set's. Nothing
//! rises or falls in the run: a trajectory starts on its set's cluster (its bare state landed there, `Onefold::landed_at`;
//! nowhere where the landing does not exist, counted), a jump into a set lands on that set's cluster. A decision time
//! reads and makes the decompositions again (below); a reading beyond the share there, or a fold that does not exist
//! there, is counted and said with each result and at the solve's end (adapt.rs `counted`), the representation unchanged.
//!
//! # The representation and the decision times
//! An atom's kept decompositions (proto.rs `Cache`), per set, are its own: the trajectory carries them as a value
//! (`step::Held`). They change only where the run says a decision is made (`Dynamics::middle` with `decide`; DESIGN
//! §11.3): there the middle's first evaluation joins or lets go the levels and makes the decompositions again as its
//! readings ask, and what it decides is the trajectory's from then on (`Middle::held`). Every other evaluation -- the
//! step's path at its own momentum, a set a jump lands in, the photon and the landing at a jump -- starts from the
//! trajectory's representation and holds it (proto.rs # The decisions: no level joins or leaves, no decomposition is
//! made again; its readings read). A decision is made where a trajectory starts (`Dynamics::started`: the landing of
//! its bare state, the held forms there -- its first decomposition made once) and at the decision times, the ends of
//! the coarsest level's cells (adapt.rs # The decision times: at the end of the step that reaches one, on its end state
//! at its own point, a middle made there deciding -- step.rs `Midway::decided`; its readings and its switch the fold's
//! there, not τ/2 before). What a holding forward lacks -- a set's first decomposition (a jump into a set the
//! trajectory has none of), or a fold that does not exist on what it holds -- it makes as a decision does, and that is
//! the trajectory's (`Middle::forward`): no choice. A fold that does not exist even so rises at the step's start
//! (step.rs `Midway::new`, `Middle::risen`). The representation is per set a shared value (`Caches`): a decision
//! replaces its own set's.
//!
//! # The held forms
//! A step's path momentum is predicted from the start state's held forms (step.rs # The path's momentum): the end state's
//! at the last step's middle, in the same contraction as the step's kick (`Middle::nodes`: one more density at the
//! middle, its static G's form on a channel of its own and each tone's complex amplitude on one each, referenced to the
//! middle's time; the integrator takes the reference's gradient from them); at a point -- a trajectory's start, a jump at
//! a step's end into another set, a switch at a decision -- one evaluation there, its tangent and the contraction at the point, less the
//! reference's gradient there (`Onefold::held_forms`).
//!
//! # The tones
//! What stays in P and beats -- an entry of two P states of different frames (proto.rs # The tones) -- is the
//! generator's turning part: at the time t the generator is G + Σ_m e^{iν_m t} G_m, each tone's envelope G_m at the
//! step's three nodes its own (the light at the path's points there, the middle's S). The integrator's candidates
//! (`Middle::beats`) are the generator's whole parts at each exact beat and its opposite -- the tones' pairs, ν and −ν,
//! and the cross decays kept at those beats (# The jumps' coherence; `Candidate`) --, each keyed by its
//! beat's exact coordinates; their envelopes carry the middle's phase e^{±iν t_m}. Their forms are the same contraction
//! as the static G's (proto.rs `Fold::prototype_force`):
//! at the nodes with the static ones at their exact phases (`Middle::nodes`, `Middle::force`), on the integrator's sums
//! at their beats (`Middle::tone_force`), on the held state at the beats' weights (`Middle::nodes`, `Middle::tones`).
//!
//! # The jumps on the step's middle (DESIGN §13.2)
//! A jump inside a step is made on the middles the step made: the photon at the jump's time s from the middle on the
//! emitting set's (`Middle::emitted`: the light turned to the jump's place, the responses per beam and their tangent the
//! middle's turned with it, S the middle's -- proto.rs `Fold::prototype_place`, no fold at the place), the landing
//! S N⁻¹f on the landing set's (`Middle::landing`, S and N the middle's). At a point (a trajectory's start) a middle is made
//! there and the jump made on it at s = 0: one way for both.
//!
//! # The jumps' coherence (BRIEF_WIRE3 item 3; FOLDAPI §0, §2)
//! The jumps are three operators, one per emission axis q, over every block of the set: L_q(t) = C_q Σ_o e^{−iot}ε_o S⁻¹
//! (DESIGN §1.3), each block at its exact phase -- the photons' classes are the landing sets (proto.rs
//! `Fold::prototype_photons`), a jump's amplitudes its class's photons' own, every block summed at its phase at the jump's
//! time (`Middle::emitted`, proto.rs `Workspace::jumped`): the landings' interference whole, a static one's too. The
//! decay between jumps is Σ_q L_q†L_q: its blocks' own terms static in G, and between two blocks their cross products at
//! the beat o − o′, each a term of the generator under the one rule (fold.rs `Fold::beating`): within the share it is
//! averaged out, beyond it added to the generator at its beat at every position (`step::Middle::generator`), so that
//! where every cross term is a tone i(G − G†) = Σ_q L_q(t)†L_q(t) at every time, static and beating, by construction.
//! Each cross decay kept is part of the integrator's candidate of its exact beat -- its key the offsets' exact
//! coordinates o − o′, with the tone of P between the two blocks' explicit states where there is one, at the beat's same
//! key --, so that the integrator takes the decay's beat as it takes a tone (spectral.rs, # Which beats are tones) and the
//! survival's hazard at its exact phase (step.rs, # Jumps): not through the step's three nodes, which alias it beyond
//! νh = π.
//!
//! # The averaged terms folded (FOLDAPI §2; DESIGN §1.2, §3.2)
//! A term of the generator that turns and that the rule averages out -- an entry of Ĥ_PP between places of different
//! frames within the share (proto.rs `Workspace::averaged`), a cross decay of two blocks within it (fold.rs
//! `Fold::beating`) -- is a Floquet sideband node, folded as a far level or a ground copy is, not dropped: its static
//! second order is kept. Of the generator's Fourier coefficients G_{±ν} at each averaged beat ν (every averaged term at
//! that exact beat summed, the Hamiltonian's and the decay's alike), the high-frequency expansion's second order
//! [G_{+ν}, G_{−ν}]/ν is added to the static generator at every position (`Middle::generator`, `folded_part`, `Fourier`):
//! made at a decision's point and held with the decompositions (proto.rs `Cache::folded`; its change over the cell is its
//! own size times the light's, beyond the order it is of); what it leaves is
//! the next order, its reading m = a/|ν| (the rule's, within the share) once more smaller. Of a cross decay the
//! coefficients are −(i/2) Σ_q L_o,q†L_o′,q and its adjoint: their commutator is Hermitian, a shift, no decay -- the
//! decay stays the jumps' products'. (Its force -- the shift's gradient -- is not taken: the contraction holds the static
//! A and the tones alone.)
//!
//! # The jumps' phase
//! A jump lands on the lab frame's ground states, C_q Σ e^{−iot} ε_o S⁻¹x: in the landing set's coordinates, each of its
//! ground states s at its own frame θ_s, the landed amplitude of s is Σ_o e^{i(θ_s − o)t} (C_q ε_o S⁻¹x)_s -- every block
//! at its offset's phase, and e^{iθ_s t} what a set of several frames holds between its states (`Dynamics::emit`). The rates, the force and the diffusion are sums over the states of |u_s|² and
//! of their gradients' products, each its own phase's: none of them turns.
//!
//! # The rest of the model (BRIEF_STEP3C item 3)
//! The splitting's reference potential (A) and the loss energy are the fold's, of the same request with a subset of its
//! light lit (`lit`: every other frequency's beams' fields and couplings zero) -- the same object and the same rule
//! (`Choice::base`), the light its input:
//! - the reference, V_ref(r) = Re Σ e_a*(r)e_b(r) W_ab, is what the folds of the trap's frequency alone hold of the field
//!   at the focus on the reference state's ground populations: Ĥ_PP's outside forms and the far blocks' U_a†R₀U_b, R₀
//!   held there (proto.rs `Fold::prototype_held`; `Reference`). Bilinear in the field's coefficients e = ε_k (ε_k*
//!   counter-rotating), ε_k = Σ_l γ_lk a_l(r), so a form of the beam pairs a_l'*a_l as the old model's far potential:
//!   its gradient and Hessian are the beams' amplitudes' jets, exactly (equal to the old prepared potential to 10⁻¹³
//!   where the trap light is far: tests). A
//!   near-resonant trap light has no far block and holds none: its force stays in the model's, less the reference's, as
//!   any part of the force does (a reference is exact for any potential: step.rs # The reference);
//! - the loss energy, (ħ/m)κ²/2 + V_run(r; p), V_run(r; p) = Σ_b p_b [G_L(r) − G_{L∖trap}(r)]_bb of the folds of every
//!   light and of every light but the trap's (the run's clusters, and those of its request with the trap's light off:
//!   `Clusters`), each ground set's on the cluster the bare state √p holds at the point at rest -- with nothing explicit
//!   where its fold exists there, else risen until it does, the rule of a trajectory's start (`Clusters::risen`,
//!   existence its test) -- its diagonal in the lab (G's less its frame), p the state's ground populations (an explicit
//!   excited state's by its decay's branching). On a risen cluster the risen node's light shift is in its coupling to
//!   the ground places, not on their diagonal; so both light sets stand on one set of explicit nodes, a node risen for
//!   either risen in both (`Clusters::difference`): its shift is left out of both alike and their difference keeps the
//!   trap's (BRIEF_LANDIMPL item 4: risen in one alone, the difference jumped by that node's whole light shift where the
//!   other's fold ceased to exist). Its far value is zero: the trap's frequency holds no plane wave
//!   (ensemble.rs `PLANE_WAVE` refuses one).
//! The displayed rows are the fold's time-averaged populations at the point (`Dynamics::readout`): the populations of
//! the slow modes between two jumps (proto.rs `Workspace::populations`) -- not the optical Bloch equations' ensemble
//! ρ_ee, which holds each jump's transient over 1/Γ that the fold's landing leaves out (within the share, read by the
//! ring's reading).
//!
//! # The optical sector (step.rs # The optical sector's noise; DESIGN §1.3, R23)
//! A set that holds explicit excited states has an optical sector (`Middle::optical`): its ground states, γ_Q the least
//! of its explicit states' decay over two, and the light pattern's fastest beat along the path over its light's beams.
//! Its K is of the gradient's matrices Ĝ_a (`Middle::gradient`), not of their contraction with one density: the same
//! derivative of the step as the force's, in its forward mode (proto.rs `Fold::prototype_gradient`: the step's tangent
//! through the compression) -- the force contracts it in the reverse mode, each where it is the cheap one. Their slope
//! along the path is the parabola through the step's outer nodes the propagation takes, a tone's phase exact. A set of
//! ground states alone has none: its fold's photons hold that part.
use super::proto::{At, Bar, Cache, Moments, Workspace, KINDS};
#[cfg(test)]
pub(crate) use super::proto::PROF;
use super::{Choice, Fold};
use crate::field::Light;
use crate::linalg::{Mat, C64};
use crate::model::{Cluster, Point, Quadratic, Scattering};
use crate::request::{Request, GROUND, STATES};
use crate::step::{Decided, Emitted, Forms, Held, Internal, Landed, Landing, Middle as _, Optical, Owed};
use std::cell::{RefCell, RefMut};
use std::sync::{Arc, Mutex, OnceLock};

/// A decision's time and states, counted when it ends (adapt.rs `Counts`).
struct Timed(std::time::Instant, usize);
impl Drop for Timed {
    fn drop(&mut self) {
        crate::adapt::count(crate::adapt::DECISIONS, 1);
        crate::adapt::count(crate::adapt::DECIDING, self.0.elapsed().as_nanos() as u64);
        crate::adapt::count(crate::adapt::STATES, self.1 as u64);
    }
}

/// An explicit excited node: its level, the absorbed component's part and a source's ground state (`Choice::explicit`).
type Key = (usize, crate::model::Component, usize);

/// One cluster of the run (# The state sets): its ground set's index and its explicit nodes, its fold, its cluster, the
/// ways a decision may take from it (`Rise`, `Fall`), and the workspaces its evaluations take in turn. Nothing is counted
/// at the focus: a representation's price is its sizes' formula where it stands (proto/price.rs).
struct Set {
    ground_set: usize,
    fold: Fold,
    cluster: Cluster,
    pool: Mutex<Vec<Workspace>>,
    /// The tones' values ν_m (rad/µs; proto.rs `Workspace::tones`' order); the integrator's candidates (`Candidate`).
    tones: Vec<f64>,
    candidates: Vec<Candidate>,
    /// The pairs of the jumps' blocks (their order: the near blocks', then the far ones', as the fold's blocks) whose cross
    /// decay is a tone (fold.rs `Fold::beating`), each with its beat o − o′ (rad/µs: its candidate's, `Candidate::cross`):
    /// the decay between their offsets (# The jumps' coherence).
    coherent: Vec<(usize, usize, f64)>,
    /// The averaged terms' Fourier coefficients by beat (# The averaged terms folded).
    folded: Vec<Fourier>,
    /// Its optical sector (# The optical sector): γ_Q, the least of its explicit excited states' decay over two (none
    /// where it holds none), and its light's beams' momenta (the fold's components' beams: the pattern's beats).
    optical: Option<f64>,
    waves: Vec<[f64; 3]>,
}

/// A candidate tone of the integrator (step.rs `Middle::beats`): the generator's whole part at one exact beat and its
/// opposite (# The tones, # The jumps' coherence) -- a pair of the set's tones at ν and −ν (either may be none) and the
/// cross decays kept at them (the indices of `Set::coherent`, at ν and at −ν) --, the beat's exact key (its
/// coordinates on the set's common denominator); its counts are its sizes' formula where it stands (`Middle::tone_work`).
struct Candidate {
    beat: f64,
    plus: Option<usize>,
    minus: Option<usize>,
    cross: [Vec<usize>; 2],
    key: Vec<i64>,
}

/// One averaged beat ν of a set's generator, its exact beat's orientation (# The averaged terms folded; `Beat::sign`): at
/// +ν and at −ν, Ĥ_PP's averaged entries
/// there (indices of proto.rs `Workspace::averaged`, their G_ν by `Workspace::averaged_at`) and the blocks' cross decays there
/// (pairs of the jumps' blocks, `Set::coherent`'s order).
struct Fourier {
    value: f64,
    entries: [Vec<usize>; 2],
    pairs: [Vec<(usize, usize)>; 2],
}

/// An atom's representation (`step::Held`): per cluster its kept decompositions, each shared -- a decision replaces one
/// cluster's, a middle reads its own without copying the others' --; per ground set the cluster it last took there (where
/// a jump into that set lands). No count is kept: a cluster's price is its sizes' formula (proto/price.rs).
#[derive(Clone, Default)]
struct Caches {
    sets: Vec<Arc<Cache>>,
}

impl Caches {
    /// A cluster's decompositions.
    fn keep(&mut self, cluster: usize, cache: Arc<Cache>) {
        if self.sets.len() <= cluster {
            self.sets.resize(cluster + 1, Arc::default());
        }
        self.sets[cluster] = cache;
    }

}

/// The run's clusters (# The state sets), made where first asked for and kept: an append-only list whose entries stay
/// where they are made (chunk c holds 2^c of them), so that a reference to one lives as long as the run; and every key
/// asked for, its ground set and explicit nodes, with its cluster or why it was refused -- a refusal kept as a cluster
/// is, so that no key is made twice. A cluster is its key: its number is the run's handle of it, given in the order the
/// trajectories' threads first ask (a ground set's base cluster that set's, made first), and nothing reads a number but
/// to find its cluster or to tell two apart -- no order, no tie, no output --, so a request gives the same result to the
/// bit whatever order its threads ask in. A cluster is made under the keys' lock: its making runs on no thread pool.
struct Arena {
    chunks: [OnceLock<Box<[OnceLock<Set>]>>; 32],
    keys: Mutex<Vec<((usize, Vec<Key>), Result<usize, String>)>>,
}

impl Arena {
    fn slot(&self, i: usize) -> &OnceLock<Set> {
        let chunk = (usize::BITS - 1 - (i + 1).leading_zeros()) as usize;
        &self.chunks[chunk].get_or_init(|| (0..1_usize << chunk).map(|_| OnceLock::new()).collect())[i + 1 - (1 << chunk)]
    }

    fn get(&self, i: usize) -> &Set {
        self.slot(i).get().expect("a cluster made")
    }

    /// The cluster of a key, made by `make` where it was not asked for yet; refused as it was where it was.
    fn index(&self, key: (usize, Vec<Key>), make: impl FnOnce() -> Result<Set, String>) -> Result<usize, String> {
        let mut keys = self.keys.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((_, at)) = keys.iter().find(|k| k.0 == key) {
            return at.clone();
        }
        let at = make().map(|set| {
            let i = keys.iter().filter(|k| k.1.is_ok()).count();
            let _ = self.slot(i).set(set);
            i
        });
        keys.push((key, at.clone()));
        at
    }
}

/// The clusters of one request (# The state sets): the request, every cluster asked for (`Arena`; the first per ground set
/// its base, the base choice's explicit nodes of that set -- none: `Onefold::new`'s), the base choice, and the ground sets
/// (the photons' landing partition) with each ground state's. The run's, and the loss energy's light subset's (# The rest
/// of the model): one object, one rule.
pub(super) struct Clusters {
    request: Request,
    sets: Arena,
    base: Choice,
    grounds: Vec<Vec<usize>>,
    membership: [usize; GROUND],
    /// The potential channel's unit at rest (rad/µs; `Onefold::mechanical`'s floor, k_B T_sc).
    floor: f64,
}

/// The run's model on the fold at the atom (module documentation).
pub(crate) struct Onefold {
    pub(super) clusters: Clusters,
    hbar_over_mass: f64,
    /// The bare ground energies, and the excited states' frame frequency (rad/µs): the photons' frequencies.
    energies: Vec<f64>,
    frame: f64,
    /// The splitting's reference potential and the loss energy's light subset (module documentation, # The rest of the
    /// model): every light but the trap's, its clusters made where first asked for (every light's are the run's own).
    light: Light,
    states: Vec<Reference>,
    reference: Reference,
    /// V_ref at the focus (rad/µs): the bottom an atom's mechanical energy is counted from (`Onefold::mechanical`).
    bottom: f64,
    untrapped: OnceLock<Result<Clusters, String>>,
    /// The start's decision (decide.rs; # The representation and the decision times): the run's clusters stand on it.
    decision: super::decide::Decision,
}

// The run is one object the trajectories' threads share (solve.rs `Hand::record`): Send + Sync, by the compiler.
const _: () = {
    const fn shared<T: Send + Sync>() {}
    shared::<Onefold>();
};

impl Set {
    /// The cluster of ground set `k` (`ground`) with the explicit nodes `nodes` on a choice of nodes that keeps them
    /// (module documentation, # The state sets): its fold and cluster, its tones' candidates, cross decays and optical
    /// sector -- its structure alone, no evaluation made.
    fn new(request: &Request, choice: &Choice, k: usize, ground: &[usize]) -> Result<Set, String> {
        let decay = super::decay(request);
        let fold = Fold::new(request, choice, ground[0])?;
        let mut states: Vec<usize> = fold.ground.clone();
        states.extend(fold.explicit.iter().map(|&x| GROUND + x));
        let cluster = Cluster { ground: fold.ground.len(), frames: fold.frames.clone(), waves: vec![[0.0; 3]; states.len()], waves_exact: vec![crate::model::Beat::whole(vec![]); states.len()], states };
        let (pool, tones, candidates, coherent, folded) = if fold.blocks.is_empty() {
            (vec![], vec![], vec![], vec![], vec![])
        } else {
            let ws = fold.prototype_workspace()?;
            let (near, far) = ws.layout();
            let blocks: Vec<usize> = near.iter().map(|x| x.0).chain(far.iter().map(|x| x.0)).collect();
            let beats: Vec<(crate::model::Beat, f64)> = ws.tones().map(|(b, v)| (b.clone(), v)).collect();
            // The pairs of blocks whose cross decay is a tone, each with its exact beat o − o′ and its value.
            let pairs: Vec<(usize, usize)> = (0..blocks.len()).flat_map(|i| (0..blocks.len()).map(move |j| (i, j))).filter(|&(i, j)| i != j && fold.beating(blocks[i], blocks[j])).collect();
            let crossing: Vec<(crate::model::Beat, f64)> = pairs.iter().map(|&(i, j)| (fold.blocks[blocks[i]].offset.plus(-1, &fold.blocks[blocks[j]].offset), fold.blocks[blocks[i]].value - fold.blocks[blocks[j]].value)).collect();
            let candidates = candidates(&beats, &crossing);
            // Each pair turned at its candidate's beat, as the candidate's members are (`Middle::generator`).
            let mut coherent: Vec<(usize, usize, f64)> = pairs.iter().map(|&(i, j)| (i, j, 0.0)).collect();
            for c in &candidates {
                for (side, sign) in c.cross.iter().zip([1.0, -1.0]) {
                    side.iter().for_each(|&k| coherent[k].2 = sign * c.beat);
                }
            }
            // The averaged terms by their exact beat, oriented (# The averaged terms folded): Ĥ_PP's, and the cross decays
            // the rule does not keep (a pair of blocks with a common landing, `Fold::crossing` not zero).
            let mut folded: Vec<(crate::model::Beat, Fourier)> = vec![];
            let at = |folded: &mut Vec<(crate::model::Beat, Fourier)>, beat: crate::model::Beat, value: f64| -> (usize, usize) {
                let (oriented, side) = if beat.sign() > 0 { (beat, 0) } else { (beat.negated(), 1) };
                let m = folded.iter().position(|f| f.0 == oriented).unwrap_or_else(|| {
                    folded.push((oriented, Fourier { value: if side == 0 { value } else { -value }, entries: [vec![], vec![]], pairs: [vec![], vec![]] }));
                    folded.len() - 1
                });
                (m, side)
            };
            for (m, beat, value) in ws.averaged() {
                let (k, side) = at(&mut folded, beat.clone(), value);
                folded[k].1.entries[side].push(m);
            }
            let nb = fold.blocks.len();
            for i in 0..blocks.len() {
                for j in (0..blocks.len()).filter(|&j| j != i && fold.crossing[blocks[i] * nb + blocks[j]] && !fold.beating(blocks[i], blocks[j])) {
                    let (k, side) = at(&mut folded, fold.blocks[blocks[i]].offset.plus(-1, &fold.blocks[blocks[j]].offset), fold.blocks[blocks[i]].value - fold.blocks[blocks[j]].value);
                    folded[k].1.pairs[side].push((i, j));
                }
            }
            (vec![ws], beats.iter().map(|b| b.1).collect(), candidates, coherent, folded.into_iter().map(|f| f.1).collect())
        };
        let optical = fold.explicit.iter().map(|&x| 0.5 * decay[(x, x)].re).reduce(f64::min);
        let waves = fold.components.iter().flat_map(|c| c.kicks.iter().copied()).collect();
        Ok(Set { ground_set: k, fold, cluster, pool: Mutex::new(pool), tones, candidates, coherent, folded, optical, waves })
    }
}

impl Onefold {
    /// The run of a request (module documentation), the splitting's reference state's ground populations `reference`:
    /// every ground set's cluster with nothing explicit first (`Choice::base`), then the start's decision's. Each set's
    /// choice is made once, by the decision's search, and the run's choice is theirs (decide.rs `Search::merged`): the
    /// rule's ground sets are its structure's (`Choice::of`; the tiers keep no copy, which alone would join two sets,
    /// `Fold::sets`), its light the request's.
    pub(crate) fn new(request: &Request, reference: [f64; GROUND]) -> Result<Onefold, String> {
        let light = Light::new(request);
        let grounds = Fold::sets(request, &Choice::of(request, crate::model::elim::share())?)?;
        let mut search = super::decide::Search::new(request, &light, &grounds, 0.0, 0.0);
        let base = Onefold::of(request, &search.merged(&vec![(vec![], vec![]); grounds.len()])?, reference)?;
        // The start's decision (decide.rs): the worst case over the local physics, an atom held by the trap at most its
        // depth's kinetic energy; every reading in the potential channel's floor.
        let depth = (-base.bottom).max(0.0);
        search.bound(base.clusters.floor, (2.0 * depth / base.hbar_over_mass).sqrt());
        let decision = search.decide()?;
        let mut run = if decision.decided().iter().all(|d| d.0.is_empty() && d.1.is_empty()) {
            base
        } else {
            Onefold::of(request, &search.merged(&decision.decided())?, reference)?
        };
        run.decision = decision;
        Ok(run)
    }

    /// The start's decision's explicit nodes (tests).
    #[cfg(test)]
    pub(crate) fn explicit(&self) -> Vec<Key> {
        self.decision.explicit()
    }

    /// The run's choice: the start's decision's nodes, terms and tiers (tests).
    #[cfg(test)]
    pub(crate) fn choice(&self) -> &Choice {
        &self.clusters.base
    }

    /// The start's decision's report (FOLDAPI §3, §7.5; decide.rs): per ground set its explicit nodes, its tones and the
    /// terms averaged and folded (their beats, rad/µs), its worst point and the reading there (the bound), the points
    /// beyond the share with no node offered; the search's points, rounds and time.
    pub(crate) fn report(&self) -> String {
        let d = &self.decision;
        let labels = self.clusters.request.levels().unwrap_or_default();
        let label = |s: usize| labels.get(s).map_or(format!("{s}"), |l| format!("F={} m={}", l.f, l.m));
        let mut out = format!(
            "开头决定：{} 个点（{} 维），{:.2} s；各轮（点数，显式结点数）{:?}；份额 {:.4}\n",
            d.points,
            d.dimensions,
            d.seconds,
            d.rounds,
            crate::model::elim::share()
        );
        for (k, sd) in d.sets.iter().enumerate() {
            let set = self.clusters.sets.get(k);
            let ground: Vec<String> = sd.ground.iter().map(|&s| label(s)).collect();
            let named = |keys: &[Key]| -> Vec<String> {
                keys.iter()
                    .map(|&(level, part, source)| {
                        let x = set.fold.excited_level.iter().position(|&l| l == level);
                        format!("{} via {:?} from {}", x.map_or(format!("level {level}"), |x| label(GROUND + x).replace("F=", "F'=")), part, label(source))
                    })
                    .collect()
            };
            out += &format!("  集合 {k} {ground:?}
    显式：{:?}
", named(&sd.nodes));
            let kinds = |f: &dyn Fn(&super::Term) -> bool| sd.toned.iter().filter(|t| f(t)).count();
            out += &format!(
                "    保留的项（逐点读数超份额）：P 内项 {}，驱动 {}，交叉衰减 {}；份额内的项最大读数 {:.4e}\n",
                kinds(&|t| matches!(t, super::Term::Pair(..))),
                kinds(&|t| matches!(t, super::Term::Drive(..))),
                kinds(&|t| matches!(t, super::Term::Cross(..))),
                sd.terms_worst
            );
            if !sd.refused_terms.is_empty() {
                out += &format!("    要求保留而簇做不成的项 {:?}\n", sd.refused_terms.iter().map(|(t, c)| format!("{t:?}：{c} 点")).collect::<Vec<_>>());
            }
            out += &format!("    音调：{:?}\n", set.tones.iter().map(|v| format!("{v:.4}")).collect::<Vec<_>>());
            out += &format!("    作音调的交叉衰减：{:?}\n", set.coherent.iter().map(|c| format!("{:.4}", c.2)).collect::<Vec<_>>());
            out += &format!("    平均并折叠：{:?}\n", set.folded.iter().map(|f| format!("{:.4e}", f.value)).collect::<Vec<_>>());
            match &sd.worst {
                Some((at, reading)) => {
                    let v: Vec<f64> = at.kappa.iter().map(|k| k * self.hbar_over_mass).collect();
                    let phases: Vec<String> = at.phases.iter().enumerate().filter(|(_, p)| **p != 0.0).map(|(l, p)| format!("{}:{:.2}", self.clusters.request.beams[l].name, p)).collect();
                    out += &format!("    最坏点：s {:.3}，相位 {phases:?}，v [{:.3}, {:.3}, {:.3}] m/s；界（读数）{reading:.4e}\n", at.s, v[0], v[1], v[2]);
                }
                None => out += "    最坏点：无\n",
            }
            out += &format!("    超份额而无结点可升的点 {}，折叠不存在且无结点可升的点 {}\n", sd.beyond, sd.nowhere);
            out += &format!("    同态结点对的度规交叉读数：最大 {:.3e}（份额内的混入上限 {:.3e}），超份额的读数 {} 次", sd.crossed.0, crate::model::elim::share() / 2.0, sd.crossed_beyond);
            if let Some(at) = &sd.crossed.1 {
                let v: Vec<f64> = at.kappa.iter().map(|k| k * self.hbar_over_mass).collect();
                out += &format!("，最大处 s {:.3} v [{:.3}, {:.3}, {:.3}] m/s", at.s, v[0], v[1], v[2]);
            }
            out += "\n";
            if !sd.refused.is_empty() {
                let refused: Vec<String> = sd
                    .refused
                    .iter()
                    .map(|&((level, part, source), count)| {
                        let x = set.fold.excited_level.iter().position(|&l| l == level);
                        format!("{} via {:?} from {}：{count} 点", x.map_or(format!("level {level}"), |x| label(GROUND + x).replace("F=", "F'=")), part, label(source))
                    })
                    .collect();
                out += &format!("    要求升而簇做不成的结点（同一能级第二个偏置）：{refused:?}；终读中要求它们的点 {}", sd.unheld);
                if let Some(at) = &sd.unheld_at {
                    let v: Vec<f64> = at.kappa.iter().map(|k| k * self.hbar_over_mass).collect();
                    out += &format!("，其一 s {:.3} v [{:.3}, {:.3}, {:.3}] m/s 相位 {:?}", at.s, v[0], v[1], v[2], at.phases.iter().map(|p| format!("{p:.2}")).collect::<Vec<_>>());
                }
                out += "\n";
            }
        }
        out
    }

    /// The run of a request on a base choice of nodes (`new`: nothing explicit): its clusters (`Clusters::new`). Refused
    /// where a set is not the prototype's (proto.rs `prototype_workspace`).
    pub(crate) fn of(request: &Request, choice: &Choice, reference: [f64; GROUND]) -> Result<Onefold, String> {
        let light = Light::new(request);
        let axes = super::Axes::new(request)?;
        let energies: Vec<f64> = (0..GROUND).map(|s| request.hamiltonian[(s, s)].re).collect();
        let hbar_over_mass = crate::units::hbar_over_mass(request.mass_kg);
        let states = Reference::states(request, &light)?;
        let reference = Reference::of(&states, &reference);
        // The potential channel's floor, k_B T_sc (unify2 §2.1; estimate.rs `semiclassical`): the larger of 30 recoils at
        // the beams' mean k² and the reference's zero point ħω̄/2 at the focus (rad/µs).
        let k2 = light.beams.iter().map(|b| b.wave_number * b.wave_number).sum::<f64>() / light.beams.len().max(1) as f64;
        let at = reference.at(&light, [0.0; 3]);
        let zero_point = (0..3).map(|a| (hbar_over_mass * at.hessian[a][a]).max(0.0).sqrt()).sum::<f64>() / 6.0;
        let floor = (crate::units::SUB_DOPPLER_RECOILS * hbar_over_mass * k2).max(zero_point);
        let run = Onefold {
            clusters: Clusters::new(request, choice, floor)?,
            hbar_over_mass,
            frame: axes.omega(&axes.frame()),
            energies,
            bottom: at.value,
            reference,
            states,
            untrapped: OnceLock::new(),
            light,
            decision: super::decide::Decision::default(),
        };
        Ok(run)
    }

    /// The ground sets' count (tests).
    #[cfg(test)]
    pub(crate) fn sets_len(&self) -> usize {
        self.clusters.grounds.len()
    }

    /// An atom's mechanical energy at (r, κ), the potential channel's unit (unify2 §2.1: each atom's error within a share of
    /// its own energy keeps the ensemble's within the same share of its whole): (ħ/m)κ²/2 + V_ref(r) − V_ref(focus), k_B T_sc
    /// at the least (rad/µs).
    fn mechanical(&self, r: [f64; 3], kappa: [f64; 3]) -> f64 {
        let kinetic = 0.5 * self.hbar_over_mass * kappa.iter().map(|k| k * k).sum::<f64>();
        (kinetic + self.reference.at(&self.light, r).value - self.bottom).max(self.clusters.floor)
    }

    /// A cluster's kept decompositions in a representation (none yet: a new atom's).
    fn cache(&self, held: &Held, set: usize) -> Arc<Cache> {
        caches(held).and_then(|c| c.sets.get(set).cloned()).unwrap_or_default()
    }

    /// One evaluation at a point that decides nothing (module documentation, # The representation): the set's forward on
    /// a copy of the trajectory's decompositions, then `then` on it with the forward's work. None where the fold does not
    /// exist there.
    fn evaluate<R>(&self, set: usize, r: [f64; 3], kappa: [f64; 3], held: &Held, then: impl FnOnce(&Fold, &Cache, &mut Workspace, f64) -> R) -> Option<R> {
        let fold = &self.clusters.sets.get(set).fold;
        let held = self.cache(held, set);
        let mut cache = (*held).clone();
        let mut ws = self.clusters.take(set);
        // Held first; where it does not exist on what the atom holds, made as a decision's, as a middle's forward is
        // (`Middle::forward`) -- on the copy: it decides nothing.
        let energy = self.mechanical(r, kappa);
        let mut exists = fold.prototype_within(r, kappa, energy, &mut cache, &mut ws, false).is_some();
        let mut spent = ws.spent();
        if !exists {
            cache = (*held).clone();
            exists = fold.prototype_within(r, kappa, energy, &mut cache, &mut ws, true).is_some();
            spent += ws.spent();
        }
        let out = exists.then(|| then(fold, &cache, &mut ws, spent));
        self.clusters.give(set, ws);
        out
    }

    /// The held forms of a set's state at a point (module documentation, # The held forms), on the set's forward made
    /// there (`cache`, `ws`; none where the set holds no light): its tangent, and in one contraction at the point the
    /// static G's form and each tone's complex amplitude at the point's time, less the reference's gradient there (step.rs
    /// `Forms`); where `land`, of the bare state's landing (`Workspace::taken`, nothing following: S N⁻¹ψ, its left
    /// response made: `Fold::prototype_left`). With the state they are of; no number where the landing does not exist.
    fn held_forms(&self, set: usize, at: &Point, psi: &[C64], made: Option<(&Cache, &mut Workspace)>, land: bool) -> (Vec<C64>, Forms) {
        let gradient = self.reference.at(&self.light, at.r).gradient;
        let Some((cache, ws)) = made else { return (psi.to_vec(), Forms { time: at.t, ..Forms::default() }.less(gradient)) };
        let fold = &self.clusters.sets.get(set).fold;
        let state = if land { ws.taken(fold, cache, psi, &[], false).map(|x| x.0) } else { Some(psi.to_vec()) };
        let Some(state) = state else { return (psi.iter().map(|_| C64::new(f64::NAN, f64::NAN)).collect(), Forms { time: at.t, held: [f64::NAN; 3], tones: vec![] }) };
        let forms = forms_on(fold, cache, ws, &self.clusters.sets.get(set).tones, at.t, &state);
        (state, Forms { time: at.t, ..forms }.less(gradient))
    }

    /// p(ψ), the ground populations of a set's state (model.rs `Model::ground_populations`, on the set's states): an
    /// explicit excited state's by its decay's branching.
    fn populations(&self, set: usize, psi: &[C64]) -> [f64; GROUND] {
        let total = crate::linalg::norm_sqr(psi);
        let mut p = [0.0; GROUND];
        for (z, &s) in psi.iter().zip(&self.clusters.sets.get(set).cluster.states) {
            let weight = z.norm_sqr() / total;
            if s < GROUND {
                p[s] += weight;
            } else if weight != 0.0 {
                let rate: f64 = (0..GROUND).map(|b| self.clusters.request.emission.iter().map(|c| c[(b, s)].norm_sqr()).sum::<f64>()).sum();
                for (b, landing) in p.iter_mut().enumerate() {
                    *landing += weight * self.clusters.request.emission.iter().map(|c| c[(b, s)].norm_sqr()).sum::<f64>() / rate;
                }
            }
        }
        p
    }

    /// A bare state f landed at a point on cluster `cluster` (unify2 §4.1: a trajectory's start, a jump into another set;
    /// # The representation): a decision's forward there on the representation `held` (its decompositions made at the
    /// point), f landed on it (`Workspace::taken`, nothing following) against the threshold θ (its clock,
    /// `Workspace::clock`: proto/transient.rs) with its held forms there; the representation with that decision's
    /// decompositions. None where the fold, the landing or its modes do not exist there.
    fn landed_at(&self, cluster: usize, at: &Point, f: &[C64], from: &[usize], held: &Held, threshold: f64) -> Option<(usize, Landed, Forms, Caches)> {
        let mut caches = caches(held).cloned().unwrap_or_default();
        let set = self.clusters.sets.get(cluster);
        let bare: Vec<C64> = set.cluster.states.iter().map(|s| from.iter().position(|x| x == s).map_or(C64::ZERO, |i| f[i])).collect();
        if set.fold.blocks.is_empty() {
            let (state, forms) = self.held_forms(cluster, at, &bare, None, false);
            return Some((cluster, Landed::plain(state.into_iter().collect(), threshold), forms, caches));
        }
        let mut cache = (*caches.sets.get(cluster).cloned().unwrap_or_default()).clone();
        let mut ws = self.clusters.take(cluster);
        let exists = set.fold.prototype(at.r, at.kappa, self.mechanical(at.r, at.kappa), &mut cache, &mut ws).is_some();
        let taken = exists.then(|| ws.taken(&set.fold, &cache, &bare, &[], false)).flatten().filter(|(state, _)| state.iter().all(|z| z.is_finite()));
        let landed = taken.and_then(|(state, a)| {
            // The held forms make the step's tangent there (`forms_on`), the clock's kick its.
            let (state, forms) = self.held_forms(cluster, at, &state, Some((&cache, &mut ws)), false);
            let clock = ws.clock(&set.fold, &cache, &state, &a, threshold, at.t)?;
            Some((landed_of(state.into_iter().collect(), clock), forms))
        });
        self.clusters.give(cluster, ws);
        caches.keep(cluster, Arc::new(cache));
        landed.map(|(landed, forms)| (cluster, landed, forms, caches))
    }

    /// The prepared potential of the ground populations p (module documentation, # The rest of the model; `Reference::of`):
    /// the trap light's alone, as the splitting's reference is of the reference state's.
    pub(crate) fn prepared(&self, p: &[f64; GROUND]) -> Reference {
        Reference::of(&self.states, p)
    }

    /// The run's light.
    pub(crate) fn light(&self) -> &Light {
        &self.light
    }

    /// The books' profile of a cluster's state on the trajectory's representation (`step::Held`; unify2 §2.7): per kind
    /// the reading its cluster's last decision left and their whole, rate units (proto.rs `Cache::profile`) -- what each
    /// step books over its cell (`Middle::books`, `step::Trajectory::model`) --, no forward made; a set that holds no
    /// light books none. None where the representation holds no decision on the cluster.
    pub(crate) fn profile(&self, cluster: usize, held: &Held) -> Option<[f64; KINDS + 1]> {
        if self.clusters.sets.get(cluster).fold.blocks.is_empty() {
            return Some([0.0; KINDS + 1]);
        }
        self.cache(held, cluster).profile()
    }

    /// The reported trap's potential (SPEC §1.3; module documentation, # The rest of the model) of the ground populations
    /// p: the held forms of the run's base folds at the focus (`Fold::prototype_held`, every light on, nothing explicit),
    /// the trap's frequency's on all its beam pairs and every other frequency's on each beam's own pair alone -- its
    /// phases averaged: a pair of two beams of one frequency is their interference pattern, which an atom moving across it
    /// at any speed sees averaged out over the trap. A near light holds no form (its response is no form of the field)
    /// and is not in it. None where a far block does not exist at the focus.
    pub(crate) fn reported(&self, p: &[f64; GROUND]) -> Result<Reference, String> {
        let mut out = Reference { pairs: vec![] };
        for k in (0..self.clusters.grounds.len()).filter(|&k| self.clusters.grounds[k].iter().any(|&s| p[s] != 0.0)) {
            let fold = &self.clusters.sets.get(k).fold;
            if fold.blocks.is_empty() {
                continue;
            }
            let weight: Vec<f64> = (0..fold.size()).map(|x| fold.ground.get(x).map_or(0.0, |&s| p[s])).collect();
            let mut ws = self.clusters.take(k);
            let forms = fold.prototype_held(&mut ws, &weight);
            self.clusters.give(k, ws);
            let trap = self.light.trap;
            out.add(Reference::laid(&self.light, &forms.ok_or("报告阱：远块在焦点处不存在")?, &|w, first, second| w == trap || first == second).pairs.into_iter());
        }
        Ok(out)
    }
}

impl Clusters {
    /// A request's clusters on a base choice of nodes (# The state sets): per ground set its cluster with the choice's
    /// explicit nodes of that set first (cluster k the k-th set's), any other made where it is first asked for
    /// (`cluster_with`).
    fn new(request: &Request, choice: &Choice, floor: f64) -> Result<Clusters, String> {
        let grounds = Fold::sets(request, choice)?;
        let mut membership = [0; GROUND];
        for (k, set) in grounds.iter().enumerate() {
            set.iter().for_each(|&g| membership[g] = k);
        }
        let out = Clusters { request: request.clone(), sets: Arena { chunks: std::array::from_fn(|_| OnceLock::new()), keys: Mutex::new(vec![]) }, base: choice.clone(), grounds, membership, floor };
        for k in 0..out.grounds.len() {
            let mut nodes: Vec<Key> = choice.explicit.iter().copied().filter(|x| out.grounds[k].contains(&x.2)).collect();
            nodes.sort();
            out.sets.index((k, nodes), || Set::new(request, choice, k, &out.grounds[k]))?;
        }
        Ok(out)
    }

    /// The cluster of ground set k with the explicit nodes `nodes` (# The state sets): made where it is not yet, on the
    /// base choice of that set with those nodes kept in P
    /// (`Choice::base_of`: its own set's tiers, no other's); refused where a fold of it cannot be made (a state explicit at
    /// two offsets, a block the prototype does not hold), and kept refused.
    pub(super) fn cluster_with(&self, k: usize, mut nodes: Vec<Key>) -> Result<usize, String> {
        nodes.sort();
        self.sets.index((k, nodes.clone()), || {
            let mut explicit: Vec<Key> = self.base.explicit.iter().copied().filter(|x| !self.grounds[k].contains(&x.2)).collect();
            explicit.extend(nodes.iter().copied());
            let choice = Choice::base_of(&self.request, explicit, self.base.toned.clone(), &|set| set == &self.grounds[k][..])?;
            Set::new(&self.request, &choice, k, &self.grounds[k])
        })
    }

    /// A workspace of a cluster, taken from its pool (made where none is free).
    fn take(&self, set: usize) -> Workspace {
        let taken = self.sets.get(set).pool.lock().unwrap_or_else(|e| e.into_inner()).pop();
        taken.unwrap_or_else(|| self.sets.get(set).fold.prototype_workspace().expect("the set's workspace"))
    }

    fn give(&self, set: usize, ws: Workspace) {
        self.sets.get(set).pool.lock().unwrap_or_else(|e| e.into_inner()).push(ws);
    }

    /// Σ_b p_b [G(r) − G′(r)]_bb in the lab (the diagonals less their frames) of these clusters' light and `other`'s, over
    /// the ground sets p populates (# The rest of the model), both on one set of explicit nodes: the run's (the start's
    /// decision, its base choice's), each light set's of its own ground states and components. None where either fold does
    /// not exist at r at rest.
    fn difference(&self, other: &Clusters, r: [f64; 3], p: &[f64; GROUND]) -> Option<f64> {
        let mut out = 0.0;
        for (sign, side) in [(1.0, self), (-1.0, other)] {
            for k in (0..side.grounds.len()).filter(|&k| side.grounds[k].iter().any(|&s| p[s] != 0.0)) {
                let base = &side.sets.get(k).fold;
                let on = |key: &Key| side.grounds[k].contains(&key.2) && base.components.iter().any(|c| c.part == key.1);
                let nodes: Vec<Key> = self.base.explicit.iter().copied().filter(on).collect();
                let cluster = side.cluster_with(k, nodes).ok()?;
                let set = side.sets.get(cluster);
                let mut cache = Cache::default();
                let mut ws = side.take(cluster);
                let exists = set.fold.prototype(r, [0.0; 3], side.floor, &mut cache, &mut ws).is_some();
                let fold = &set.fold;
                let (n, g) = (fold.size(), ws.generator());
                let part: f64 = fold.ground.iter().enumerate().map(|(x, &s)| p[s] * (g.data[x * n + x].re + fold.frame_values[x])).sum();
                side.give(cluster, ws);
                if !exists {
                    return None;
                }
                out += sign * part;
            }
        }
        Some(out)
    }
}

/// The held forms of a state x at a fold's middle, its tangent made (`Fold::prototype_force` in one contraction): the
/// static G's form and each tone's complex amplitude at the time t (step.rs `Forms`, the model's).
fn forms_on(fold: &Fold, cache: &Cache, ws: &mut Workspace, tones: &[f64], t: f64, state: &[C64]) -> Forms {
    let n = fold.size();
    fold.prototype_tangent(cache, ws);
    let mut rho = vec![C64::ZERO; n * n];
    crate::step::density(state, &mut rho);
    let phased: Vec<Vec<C64>> = tones.iter().map(|&nu| rho.iter().map(|z| *z * C64::cis(nu * t)).collect()).collect();
    let mut bars = vec![Bar { channel: 0, position: 0, tone: None, matrix: &rho }];
    bars.extend(phased.iter().enumerate().map(|(m, x)| Bar { channel: 1 + m, position: 0, tone: Some(m), matrix: x }));
    let f = fold.prototype_force(cache, ws, &bars, 1 + tones.len()).forms;
    Forms { time: 0.0, held: real(f[0]), tones: tones.iter().zip(&f[1..]).map(|(&nu, a)| (nu, *a)).collect() }
}

/// The averaged terms' second order at the last step's middle (# The averaged terms folded): Σ_ν [G_{+ν}, G_{−ν}]/ν, each
/// coefficient Ĥ_PP's averaged entries' G_ν there (proto.rs `Workspace::averaged_at`) and its cross decays' −(i/2) Σ_q
/// L_o,q†L_o′,q (the jumps there, where the set has a cross decay averaged).
fn folded_part(set: &Set, ws: &Workspace, jumps: Option<&[[Mat; 3]]>) -> Mat {
    let n = set.fold.size();
    let mut out = Mat::zeros(n, n);
    for f in &set.folded {
        let [plus, minus] = [0, 1].map(|side| {
            let mut g = Mat::zeros(n, n);
            for &m in &f.entries[side] {
                g.add_scaled(C64::ONE, &ws.averaged_at(m));
            }
            for &(i, j) in &f.pairs[side] {
                let jumps = jumps.expect("the jumps of an averaged cross decay's blocks");
                for q in 0..3 {
                    g.add_scaled(C64::new(0.0, -0.5), &jumps[i][q].adjoint().product(&jumps[j][q]));
                }
            }
            g
        });
        out.add_scaled(C64::real(1.0 / f.value), &plus.product(&minus));
        out.add_scaled(C64::real(-1.0 / f.value), &minus.product(&plus));
    }
    out
}

/// A representation's per-cluster decompositions and chosen clusters (`Caches`), where it holds the run's.
fn caches(held: &Held) -> Option<&Caches> {
    held.0.as_ref().and_then(|h| h.downcast_ref::<Caches>())
}

/// A set's candidate tones (`Candidate`; module documentation, # The tones, # The jumps' coherence) from the exact beats
/// and values of its tones and of its cross decays kept (`crossing`): each exact beat with its opposite
/// once -- its tones there and its cross decays there --, in the order of their first beat of positive orientation
/// (`Beat::sign`; of a beat whose opposite is none, its own); its beat the one of positive orientation, of its tone's
/// value where it has one; its key that beat's coordinates on the common denominator of them all.
fn candidates(beats: &[(crate::model::Beat, f64)], crossing: &[(crate::model::Beat, f64)]) -> Vec<Candidate> {
    let all = || beats.iter().chain(crossing);
    let common = all().fold(1_i64, |l, (b, _)| {
        let (mut a, mut c) = (l, b.d);
        while c != 0 {
            (a, c) = (c, a % c);
        }
        l / a * b.d
    });
    let mut out: Vec<Candidate> = vec![];
    let mut made: Vec<crate::model::Beat> = vec![];
    for (b, value) in all() {
        let oriented = if b.sign() > 0 { b.clone() } else { b.negated() };
        if (b.sign() < 0 && all().any(|x| x.0 == oriented)) || made.contains(&oriented) {
            continue;
        }
        let opposite = oriented.negated();
        let (plus, minus) = (beats.iter().position(|x| x.0 == oriented), beats.iter().position(|x| x.0 == opposite));
        let cross = [&oriented, &opposite].map(|beat| (0..crossing.len()).filter(|&k| crossing[k].0 == *beat).collect::<Vec<usize>>());
        let beat = match (plus, minus) {
            (Some(m), _) => beats[m].1,
            (None, Some(m)) => -beats[m].1,
            (None, None) => *value * b.sign() as f64,
        };
        let key = oriented.n.iter().map(|x| x * (common / oriented.d)).collect();
        out.push(Candidate { beat, plus, minus, cross, key });
        made.push(oriented);
    }
    out
}

/// The request with only the frequencies `on` lit (module documentation, # The rest of the model): every other
/// frequency's beams' fields and couplings zero, and what H0 holds of its outside shift at the focus taken out of H0
/// (request.rs `held`: H = H0 − held(O(0)) + O(r)).
pub(super) fn lit(request: &Request, light: &Light, on: impl Fn(usize) -> bool) -> Request {
    let mut out = request.clone();
    for (_, f) in light.frequencies.iter().enumerate().filter(|(w, _)| !on(*w)) {
        let focus: [C64; 3] = std::array::from_fn(|q| f.beams.iter().map(|&l| request.beams[l].field[q]).sum());
        let mut at_focus = Mat::zeros(STATES, STATES);
        for (x, o) in crate::request::coordinates(&focus).iter().zip(&request.outside_forms[f.form].matrices) {
            at_focus.add_scaled(C64::real(*x), o);
        }
        out.hamiltonian.add_scaled(-C64::ONE, &crate::request::held(&at_focus));
        for &l in &f.beams {
            let beam = &mut out.beams[l];
            beam.field = [C64::ZERO; 3];
            beam.coupling = Mat::zeros(STATES, STATES);
            beam.counter_coupling = Mat::zeros(STATES, STATES);
        }
    }
    out
}

/// The splitting's reference potential (module documentation, # The rest of the model): V_ref(r) = Re Σ B_l'l a_l'*(r)a_l(r)
/// over the beam pairs of the lit frequencies -- the held forms' terms e_a* e_b W with each coefficient on its beams,
/// e = ε_k = Σ_l γ_lk a_l (field.rs `Frequency::coefficients`), its conjugate counter-rotating --, each pair's product with
/// its gradient and Hessian (far.rs `pair`), as the old model's far potential takes them. The held forms are linear in
/// the ground populations they are weighed by (`Fold::prototype_held`), and so is V_ref: a population's is its ground
/// states' alone weighed by it (`Reference::states`, `Reference::of`) -- the prepared potential of every initial
/// component and of every ground state (solve.rs: the wells, the kick's curvature) the same object as the run's.
pub(crate) struct Reference {
    pairs: Vec<(usize, usize, C64)>,
}

impl Reference {
    /// Every ground state's reference alone, in the ground states' order: the held forms of the folds of the request
    /// with the trap's frequency alone lit (`lit`) at the focus, each ground set's with nothing explicit
    /// (`Choice::base`), at each of its ground places alone (`Fold::prototype_held`), each term laid on its beam pairs.
    /// Refused where a far block does not exist at the focus.
    pub(crate) fn states(request: &Request, light: &Light) -> Result<Vec<Reference>, String> {
        let prepared = lit(request, light, |w| w == light.trap);
        // Nothing explicit and no term kept: the reference reads the held forms at the focus, which no term changes.
        let choice = Choice::base(&prepared, vec![], vec![])?;
        let mut out: Vec<Reference> = (0..GROUND).map(|_| Reference { pairs: vec![] }).collect();
        for set in Fold::sets(&prepared, &choice)? {
            let fold = Fold::new(&prepared, &choice, set[0])?;
            let mut ws = fold.prototype_workspace()?;
            for (x, &s) in fold.ground.iter().enumerate() {
                let weight: Vec<f64> = (0..fold.size()).map(|y| if y == x { 1.0 } else { 0.0 }).collect();
                out[s] = Reference::laid(light, &fold.prototype_held(&mut ws, &weight).ok_or("参考势：阱光的远块在焦点处不存在")?, &|_, _, _| true);
            }
        }
        Ok(out)
    }

    /// The reference of the ground populations p, Σ_s p_s V_s of the ground states' own (`states`).
    pub(crate) fn of(states: &[Reference], p: &[f64; GROUND]) -> Reference {
        let mut out = Reference { pairs: vec![] };
        for (s, state) in states.iter().enumerate().filter(|(s, _)| p[*s] != 0.0) {
            out.add(state.pairs.iter().map(|&(first, second, value)| (first, second, value * p[s])));
        }
        out
    }

    /// Terms added to their beam pairs.
    fn add(&mut self, terms: impl Iterator<Item = (usize, usize, C64)>) {
        for (first, second, value) in terms {
            match self.pairs.iter_mut().find(|x| x.0 == first && x.1 == second) {
                Some(x) => x.2 += value,
                None => self.pairs.push((first, second, value)),
            }
        }
    }

    /// Held forms (`Fold::prototype_held`) laid on their beam pairs: each form of one frequency's coefficients, its
    /// beam pairs (l', l) those `keep` keeps of that frequency's.
    fn laid(light: &Light, forms: &[([(usize, usize, bool); 2], C64)], keep: &dyn Fn(usize, usize, usize) -> bool) -> Reference {
        let mut out = Reference { pairs: vec![] };
        for &([a, b], w) in forms {
            // e_a* e_b of one rotation: co-rotating Σ γ_ja* γ_j'b a_j* a_j', counter-rotating its conjugates' (the
            // pair of a_j' and a_j). Two coefficients of opposite rotations meet at one offset only at twice the
            // optical frequency: no static form holds them.
            assert_eq!(a.2, b.2, "a static form of coefficients of one rotation");
            let (fa, fb) = (&light.frequencies[a.0], &light.frequencies[b.0]);
            for (j, &l) in fa.beams.iter().enumerate() {
                for (jj, &ll) in fb.beams.iter().enumerate() {
                    let (ga, gb) = (fa.coefficients[j][a.1], fb.coefficients[jj][b.1]);
                    let (first, second, value) = if a.2 { (ll, l, ga * gb.conj() * w) } else { (l, ll, ga.conj() * gb * w) };
                    if a.0 == b.0 && keep(a.0, first, second) {
                        out.add(std::iter::once((first, second, value)));
                    }
                }
            }
        }
        out
    }

    /// V_ref's second-order expansion at r: each beam's amplitude to the second order (field.rs `Beam::amplitude_to`),
    /// each pair's product a_l'* a_l (far.rs `pair`), times its weight, the real part.
    pub(crate) fn at(&self, light: &Light, r: [f64; 3]) -> Quadratic {
        self.rounded(light, r).0
    }

    /// V_ref's second-order expansion at r (`at`) with the rounding of its value, `rounding(terms)` times the sum of its
    /// terms' moduli (potential.rs `descend`).
    pub(crate) fn rounded(&self, light: &Light, r: [f64; 3]) -> (Quadratic, f64) {
        let mut out = Quadratic { value: 0.0, gradient: [0.0; 3], hessian: [[0.0; 3]; 3] };
        let mut scale = 0.0;
        for &(first, second, weight) in &self.pairs {
            let c = crate::model::far::pair(&light.beams[first].amplitude_to(r, 2), &light.beams[second].amplitude_to(r, 2));
            out.value += (c.value * weight).re;
            scale += (c.value * weight).abs();
            for x in 0..3 {
                out.gradient[x] += (c.gradient[x] * weight).re;
                for y in 0..3 {
                    out.hessian[x][y] += (c.hessian[x][y] * weight).re;
                }
            }
        }
        (out, crate::units::rounding(self.pairs.len().max(1)) * scale)
    }

    /// B = Σ_l'l |B_l'l|: −V_ref(r) ≤ B max_l |a_l(r)|² (ensemble.rs `Well::envelope`), each pair's |a_l'* a_l| at most
    /// the larger of its beams' intensities.
    pub(crate) fn peak(&self) -> f64 {
        self.pairs.iter().map(|x| x.2.abs()).sum()
    }
}

/// The position of the time s from a step's middle among the prototype's (proto.rs `Workspace::tone`): 0 the middle, 1
/// the outer node at −h, 2 the one at h.
fn position(s: f64) -> usize {
    if s == 0.0 {
        0
    } else if s < 0.0 {
        1
    } else {
        2
    }
}

/// A set's generator at a point (`step::Middle`; module documentation): the forward made at a path's momentum and
/// velocity where they are first asked for (its first a decision where the middle decides; a path is its numbers, so a
/// path asked for again is the one made), the outer nodes where a step asks for them, the tangent where a form or the
/// photons are asked; the middle's time t (the tones' phases).
pub(crate) struct Middle<'a> {
    run: &'a Onefold,
    set: usize,
    r: [f64; 3],
    t: f64,
    decide: bool,
    caches: Option<Arc<Caches>>,
    inner: RefCell<Inner>,
}

/// What a middle holds of its evaluations: the set's decompositions as the trajectory gave them with what its decision
/// made (`committed`), and those its last forward stood on (`eval`); the workspace; the path the workspace holds (its
/// momentum and velocity), the outer nodes' offset it holds them at, whether it holds the tangent; whether the decision
/// was made; the generator at the middle; the forwards' counted work, made or not (real multiply-adds,
/// `step::Middle::made`), the last one's -- the trajectory's count of the cluster (`Caches::counts`) --, and its reading
/// (rate units: its decision's, proto.rs `Step::reading`).
struct Inner {
    committed: Arc<Cache>,
    eval: Cache,
    ws: Option<Workspace>,
    current: Option<([f64; 3], [f64; 3])>,
    nodes: Option<f64>,
    /// The offset the outer nodes were last made at, on any path: the step's half (its picture's ends).
    half: Option<f64>,
    tangent: bool,
    decided: bool,
    exists: bool,
    generator: Mat,
    made: f64,
    reading: f64,
    /// Whether the last decision's walk left a column beyond its budget.
    beyond: bool,
    /// The last forward's accounts by kind (ρ²; proto.rs `Step::kinds`), its cell's ends (`Step::started`, `Step::ended`)
    /// and whether it is unresolved.
    kinds: [f64; KINDS],
    started: [f64; KINDS],
    ended: [f64; KINDS],
    unresolved: bool,
}

/// A path through a middle: its momentum and velocity.
pub(crate) struct Path {
    kappa: [f64; 3],
    velocity: [f64; 3],
}

impl Drop for Middle<'_> {
    fn drop(&mut self) {
        if let Some(ws) = self.inner.get_mut().ws.take() {
            self.run.clusters.give(self.set, ws);
        }
    }
}

impl Middle<'_> {
    /// The averaged terms' second order at the middle and the static generator's there (# The averaged terms folded), their
    /// Frobenius norms, and the averaged beats' count (a measurement, tests).
    #[cfg(test)]
    pub(crate) fn folded(&self, path: &Path) -> (f64, f64, usize) {
        let set = self.run.clusters.sets.get(self.set);
        let inner = self.made(path, None, false);
        let Some(ws) = inner.ws.as_ref().filter(|_| inner.exists) else { return (0.0, 0.0, 0) };
        let jumps = set.fold.prototype_jumps(&inner.eval, ws, At::Middle);
        (folded_part(set, ws, Some(&jumps)).frobenius(), inner.generator.frobenius(), set.folded.len())
    }

    /// The forward at the path's momentum, where the workspace does not hold it (module documentation, # The representation).
    fn forward(&self, path: &Path) -> RefMut<'_, Inner> {
        let mut inner = self.inner.borrow_mut();
        if inner.current != Some((path.kappa, path.velocity)) {
            let fold = &self.run.clusters.sets.get(self.set).fold;
            let decide = self.decide && !inner.decided;
            let energy = self.run.mechanical(self.r, path.kappa);
            let Inner { committed, eval, ws, generator, .. } = &mut *inner;
            let ws = ws.as_mut().expect("a workspace");
            // A forward that decides nothing holds (# The representation and the decision times): what it would lack -- a
            // block's first decomposition -- it makes on its own copy and leaves (unify2 §4.3: no representation changes
            // between the decision times); a decision's is the trajectory's.
            *eval = (**committed).clone();
            let made = fold.prototype_within(self.r, path.kappa, energy, eval, ws, decide).map(|step| {
                generator.data.copy_from_slice(&step.generator.data);
                (step.reading, step.kinds, step.started, step.ended, step.unresolved)
            });
            // The averaged terms' second order (# The averaged terms folded): made at a decision's point, or where the
            // atom holds none yet, and held with the decompositions.
            let set = self.run.clusters.sets.get(self.set);
            if made.is_some() && !set.folded.is_empty() && (decide || eval.folded.is_none()) {
                let pairs = set.folded.iter().any(|f| f.pairs.iter().any(|p| !p.is_empty()));
                let jumps = pairs.then(|| fold.prototype_jumps(eval, ws, At::Middle));
                let folded = folded_part(set, ws, jumps.as_deref());
                // Its reading: held over the cell and out of the force, what it leaves is at most its own size against the
                // generator's (its change over the cell and its gradient are its own size times the light's relative
                // change): ‖folded‖/‖G‖, against the share.
                let reading = folded.frobenius() / generator.frobenius().max(f64::MIN_POSITIVE);
                crate::adapt::most(crate::adapt::FOLDED, reading);
                crate::adapt::count(crate::adapt::FOLDED_BEYOND, u64::from(reading > crate::model::elim::share()));
                eval.folded = Some(folded);
            }
            let spent = ws.spent() * crate::decide::COMPLEX;
            let beyond = decide && made.is_some() && ws.step_beyond();
            if decide {
                *committed = Arc::new(eval.clone());
            }
            inner.decided |= decide;
            inner.exists = made.is_some();
            inner.made += spent;
            inner.beyond = beyond;
            if let Some((reading, kinds, started, ended, unresolved)) = made {
                (inner.reading, inner.kinds, inner.started, inner.ended, inner.unresolved) = (reading, kinds, started, ended, unresolved);
            }
            (inner.current, inner.nodes, inner.tangent) = (Some((path.kappa, path.velocity)), None, false);
        }
        inner
    }

    /// The forward, the outer nodes at ±h where asked, and the tangent where asked.
    fn made(&self, path: &Path, h: Option<f64>, tangent: bool) -> RefMut<'_, Inner> {
        let mut inner = self.forward(path);
        let fold = &self.run.clusters.sets.get(self.set).fold;
        if !inner.exists {
            return inner;
        }
        if let Some(h) = h.filter(|&h| inner.nodes != Some(h)) {
            let Inner { eval, ws, .. } = &mut *inner;
            fold.prototype_nodes(self.r, path.kappa, path.velocity, [-h, h], eval, ws.as_mut().unwrap());
            (inner.nodes, inner.half, inner.tangent) = (Some(h), Some(h), false);
        }
        if tangent && !inner.tangent {
            let Inner { eval, ws, .. } = &mut *inner;
            fold.prototype_tangent(eval, ws.as_mut().unwrap());
            inner.tangent = true;
        }
        inner
    }

    /// The forms of one contraction of the step's derivative (proto.rs `Fold::prototype_force`): per channel the sum of its
    /// bars, the outer nodes at ±h made where `h`.
    fn forms(&self, path: &Path, h: Option<f64>, bars: &[Bar], channels: usize) -> Vec<[C64; 3]> {
        let fold = &self.run.clusters.sets.get(self.set).fold;
        let mut inner = self.made(path, h, true);
        if !inner.exists {
            return vec![[C64::new(f64::NAN, f64::NAN); 3]; channels];
        }
        let Inner { eval, ws, .. } = &mut *inner;
        fold.prototype_force(eval, ws.as_mut().unwrap(), bars, channels).forms
    }

    /// A jump's photon of the state ψ at the time s from the middle along the path (`step::Dynamics::emit`; DESIGN §13.2):
    /// the forward and the tangent the step made at the path (made here where it made none), the place there -- the light
    /// turned to r + vs, the responses turned lane by lane, S the middle's (proto.rs `Fold::prototype_place`): no fold at
    /// the place --, every class's rate there, the class `draw` picks and its kick alone (the tangent turned there, on its
    /// landing set's rows: `Fold::prototype_photons`); its jump's amplitudes L(s)ψ on the landing set's states -- the
    /// photons' own u = Σ_o e^{−iot} C ε_o S⁻¹ψ over every block (`Workspace::jumped`) --, each at its frame's phase at the time t_m + s
    /// (module documentation, # The jumps' phase), and the photon's frequency. None where the set has no light (no
    /// blocks), the fold does not exist there, or `draw` picks none.
    pub(crate) fn emitted(&self, path: &Path, s: f64, psi: &[C64], draw: &mut dyn FnMut(&[f64]) -> Option<usize>) -> Option<Emitted> {
        let (run, set) = (self.run, &self.run.clusters.sets.get(self.set));
        let fold = &set.fold;
        if fold.blocks.is_empty() {
            return None;
        }
        let mut inner = self.made(path, None, true);
        if !inner.exists {
            return None;
        }
        let Inner { eval, ws, .. } = &mut *inner;
        let ws = ws.as_mut().expect("a workspace");
        let place = fold.prototype_place(self.r, path.kappa, path.velocity, s, ws);
        let t = self.t + s;
        let photons = fold.prototype_photons(eval, ws, psi, &run.clusters.grounds, At::Place, t, Moments::Drawn(draw));
        let class = &photons.classes[photons.drawn?];
        // The class's jump: every block's at its offset's phase e^{−iot} (# The jumps' coherence), as its photons summed
        // them, on the landing cluster's states, each at its frame's phase e^{iθ_s t} (# The jumps' phase).
        let landing = self.landing_cluster(class.landing);
        let to = run.clusters.sets.get(landing);
        let mut amplitudes = [Internal::zeros(to.cluster.states.len()); 3];
        for (a, amplitude) in amplitudes.iter_mut().enumerate() {
            for ((z, &s), &theta) in amplitude.iter_mut().zip(&to.cluster.states).zip(&to.fold.frame_values) {
                if s < GROUND {
                    *z = ws.jumped(s, a) * C64::cis(theta * t);
                }
            }
        }
        // The photon's frequency: the excited amplitude's (the frame's, its offset's and the set's slow ground energy) less
        // the landing state's energy; the blocks' offsets at their shares of the class's rate (`Class::offset`).
        let first = fold.ground[0];
        let omega = run.frame + class.offset + run.energies[first] - fold.frame_values[0] - run.energies[run.clusters.grounds[class.landing][0]];
        Some(Emitted { landing, wave_number: crate::units::wave_number((omega / crate::units::RAD_PER_US_PER_MHZ).abs()), amplitudes, kick: class.kick, work: crate::decide::COMPLEX * (place + photons.work) })
    }

    /// A jump's photon of the state ψ with the photon its last landing's transient owes it (step.rs `Dynamics::transient`;
    /// proto/transient.rs `Clock::owed`: the fast part's amplitudes u_q at the jump, every block at its offset's phase), at
    /// the time s from the middle along the path: the whole state's photon, its class drawn by the rates of the two summed
    /// on each landing set's rows, its amplitudes the two summed on the landing cluster's states (each at its frame's
    /// phase); its kick the class's of the state's own photon (`Fold::prototype_photons`, every class's moments). None
    /// where the set has no light, the fold does not exist there, or `draw` picks none.
    pub(crate) fn transient(&self, path: &Path, s: f64, psi: &[C64], owed: &Owed, draw: &mut dyn FnMut(&[f64]) -> Option<usize>) -> Option<Emitted> {
        let (run, set) = (self.run, &self.run.clusters.sets.get(self.set));
        let fold = &set.fold;
        if fold.blocks.is_empty() {
            return None;
        }
        let mut inner = self.made(path, None, true);
        if !inner.exists {
            return None;
        }
        let Inner { eval, ws, .. } = &mut *inner;
        let ws = ws.as_mut().expect("a workspace");
        let place = fold.prototype_place(self.r, path.kappa, path.velocity, s, ws);
        let t = self.t + s;
        let photons = fold.prototype_photons(eval, ws, psi, &run.clusters.grounds, At::Place, t, Moments::Every);
        let whole = |b: usize, q: usize| ws.jumped(b, q) + owed.photon[3 * b + q];
        let rates: Vec<f64> = run.clusters.grounds.iter().map(|g| g.iter().map(|&b| (0..3).map(|q| whole(b, q).norm_sqr()).sum::<f64>()).sum()).collect();
        let k = draw(&rates)?;
        let class = &photons.classes[k];
        // The kick: the drawn class's own (of ψ's photon), its mean over the classes moved to the one the landing booked --
        // the slow photon's mean, as the step's own jumps carry it (`Clock::kick`: K_∞ + δg booked the rest); none where
        // the landing booked the cycle's whole (at once).
        let mean = |w: &dyn Fn(usize) -> f64| -> [f64; 3] {
            let total: f64 = (0..rates.len()).map(w).sum();
            std::array::from_fn(|a| if total > 0.0 { (0..rates.len()).map(|c| w(c) * photons.classes[c].kick[a]).sum::<f64>() / total } else { 0.0 })
        };
        let (booked, drawn) = (if owed.whole { [0.0; 3] } else { mean(&|c| photons.classes[c].rate) }, mean(&|c| rates[c]));
        let kick: [f64; 3] = std::array::from_fn(|a| class.kick[a] + booked[a] - drawn[a]);
        let landing = self.landing_cluster(class.landing);
        let to = run.clusters.sets.get(landing);
        let mut amplitudes = [Internal::zeros(to.cluster.states.len()); 3];
        for (a, amplitude) in amplitudes.iter_mut().enumerate() {
            for ((z, &st), &theta) in amplitude.iter_mut().zip(&to.cluster.states).zip(&to.fold.frame_values) {
                if st < GROUND {
                    *z = whole(st, a) * C64::cis(theta * t);
                }
            }
        }
        let first = fold.ground[0];
        let omega = run.frame + class.offset + run.energies[first] - fold.frame_values[0] - run.energies[run.clusters.grounds[class.landing][0]];
        Some(Emitted { landing, wave_number: crate::units::wave_number((omega / crate::units::RAD_PER_US_PER_MHZ).abs()), amplitudes, kick, work: crate::decide::COMPLEX * (place + photons.work + (6 * GROUND) as f64) })
    }

    /// The cluster a jump into ground set k lands in (# The nodes risen and fallen): this middle's where k is its own set,
    /// else the one the trajectory last took there, else k's with nothing explicit.
    fn landing_cluster(&self, k: usize) -> usize {
        if k == self.run.clusters.sets.get(self.set).ground_set {
            return self.set;
        }
        k
    }

    /// Where a jump's bare state f lands in this middle's set at the time t (`step::Landing`): its slow part S N⁻¹f along its
    /// fast one, S and N the middle's (DESIGN §1.3, §13.2, the landing; the forward the step made at the path, its left
    /// response made where a landing first asks for it, its count this middle's) -- the switch's formula with nothing
    /// following (`Workspace::taken`: a fall at x_X = 0) --, against the trajectory's threshold θ: its clock on the
    /// landing's modes (`Workspace::clock`, proto/transient.rs: the threshold θ′ the slow part's norm meets where the
    /// exact survival meets θ, the photon the transient owes that jump). The state itself where the set has no light.
    /// Nowhere where the landing does not exist -- the forward, or the left response or N (`Workspace::taken`), or its
    /// modes, the fold's one exit: a jump into another set is a decision there (`Onefold::landed_at`), one into this set
    /// lands on what the cell's look saw exist (unify2 §4.1, §4.3).
    pub(super) fn landing(&self, path: &Path, f: &[C64], threshold: f64, t: f64) -> Landing {
        let set = self.run.clusters.sets.get(self.set);
        if set.fold.blocks.is_empty() {
            return Landing::On(Landed::plain(f.iter().copied().collect(), threshold));
        }
        let mut inner = self.made(path, None, true);
        let Inner { eval, ws, made, exists, .. } = &mut *inner;
        let ws = ws.as_mut().expect("a workspace");
        if *exists {
            let landed = ws.taken(&set.fold, eval, f, &[], false);
            *made += ws.left_work() * crate::decide::COMPLEX;
            if let Some((psi, a)) = landed {
                if let Some(clock) = ws.clock(&set.fold, eval, &psi, &a, threshold, t) {
                    *made += clock.work * crate::decide::COMPLEX;
                    return Landing::On(landed_of(psi.into_iter().collect(), clock));
                }
            }
        }
        Landing::Nowhere
    }

    /// The trajectory's representation with this cluster's decompositions as this middle's decision made them.
    fn kept(&self) -> Caches {
        let mut caches = self.caches.as_ref().map_or_else(Caches::default, |c| (**c).clone());
        caches.keep(self.set, self.inner.borrow().committed.clone());
        caches
    }

    /// The representation this middle leaves (`step::Middle::held`): its decompositions (`kept`), and this cluster the one
    /// taken in its ground set.
    fn leaves(&self) -> Caches {
        self.kept()
    }

    /// The model's counts at a node on this middle's representation (`step::Middle::work`; proto/price.rs `Sizes::costs`):
    /// the formula on the sizes its forward stood on, none where it holds no light or does not exist.
    fn costs(&self) -> Option<(super::proto::price::Sizes, super::proto::price::Route)> {
        let inner = self.inner.borrow();
        let ws = inner.ws.as_ref().filter(|_| inner.exists && inner.current.is_some())?;
        let fold = &self.run.clusters.sets.get(self.set).fold;
        Some((ws.sizes(fold, &inner.eval, &self.run.clusters.grounds), ws.route()))
    }

    /// e^{iν_m (t_m + s)} ρ: tone m's X at the time s from the middle (module documentation, # The tones).
    fn phased(&self, m: usize, s: f64, rho: &[C64]) -> Vec<C64> {
        let phase = C64::cis(self.run.clusters.sets.get(self.set).tones[m] * (self.t + s));
        rho.iter().map(|z| *z * phase).collect()
    }
}

/// A landing on its clock (proto/transient.rs `Clock`; step.rs `Landed`): the slow state, the threshold θ′, the photon owed
/// to that jump and whether the landing booked its cycle's whole momentum (at once), the kick.
fn landed_of(psi: Internal, clock: super::proto::transient::Clock) -> Landed {
    Landed { psi, threshold: clock.threshold, owed: clock.owed.map(|photon| Owed { photon, whole: clock.once }), kick: clock.kick }
}

/// The real parts of a form.
fn real(form: [C64; 3]) -> [f64; 3] {
    form.map(|z| z.re)
}

impl crate::step::Middle for Middle<'_> {
    type Path = Path;

    fn states(&self) -> usize {
        self.run.clusters.sets.get(self.set).fold.size()
    }

    fn still(&self) -> bool {
        false
    }

    fn path(&self, kappa: [f64; 3], velocity: [f64; 3]) -> Path {
        Path { kappa, velocity }
    }

    fn gauge(&self, _: &Path, rates: &mut [f64]) {
        rates.fill(0.0);
    }

    /// The static G at the time s from the middle and every tone at its phase there, G + Σ_m e^{iν_m(t_m + s)} G_m(s)
    /// (module documentation, # The tones).
    fn generator(&self, path: &Path, s: f64, h: &mut Mat) {
        let inner = self.made(path, (s != 0.0).then_some(s.abs()), false);
        if !inner.exists {
            h.data.fill(C64::new(f64::NAN, f64::NAN));
            return;
        }
        let ws = inner.ws.as_ref().unwrap();
        let source = if s == 0.0 { &inner.generator } else { ws.node_generator(usize::from(s > 0.0)) };
        h.data.copy_from_slice(&source.data);
        let position = position(s);
        let set = &self.run.clusters.sets.get(self.set);
        for (m, &nu) in set.tones.iter().enumerate() {
            let phase = C64::cis(nu * (self.t + s));
            for (z, g) in h.data.iter_mut().zip(&ws.tone(m, position).data) {
                *z += phase * *g;
            }
        }
        // The decay between two blocks' offsets, kept: −(i/2) e^{i(o − o′)t} Σ_q L_o,q†L_o′,q of its jumps there (# The
        // jumps' coherence).
        if !set.coherent.is_empty() {
            let at = match position {
                0 => At::Middle,
                j => At::Node(j - 1),
            };
            let jumps = set.fold.prototype_jumps(&inner.eval, ws, at);
            for &(i, j, beat) in &set.coherent {
                let factor = C64::new(0.0, -0.5) * C64::cis(beat * (self.t + s));
                for q in 0..3 {
                    let cross = jumps[i][q].adjoint().product(&jumps[j][q]);
                    for (z, c) in h.data.iter_mut().zip(&cross.data) {
                        *z += factor * *c;
                    }
                }
            }
        }
        // The averaged terms' second order (# The averaged terms folded), as the trajectory holds it.
        if let Some(folded) = &inner.eval.folded {
            for (z, c) in h.data.iter_mut().zip(&folded.data) {
                *z += *c;
            }
        }
    }

    /// The static G's form at the time s and every tone's there at its exact phase, one contraction.
    fn force(&self, path: &Path, s: f64, rho: &[C64]) -> [f64; 3] {
        let j = position(s);
        let phased: Vec<Vec<C64>> = (0..self.run.clusters.sets.get(self.set).tones.len()).map(|m| self.phased(m, s, rho)).collect();
        let mut bars = vec![Bar { channel: 0, position: j, tone: None, matrix: rho }];
        bars.extend(phased.iter().enumerate().map(|(m, x)| Bar { channel: 0, position: j, tone: Some(m), matrix: x }));
        real(self.forms(path, (s != 0.0).then_some(s.abs()), &bars, 1)[0])
    }

    /// The static G's at zero, each tone's at its beat ν_m: the middle's G_m at its phase e^{iν_m t_m} (module
    /// documentation, # The tones), each its own channel of one contraction.
    fn tones(&self, path: &Path, rho: &[C64], each: &mut dyn FnMut(f64, [f64; 3])) {
        let tones = &self.run.clusters.sets.get(self.set).tones;
        let phased: Vec<Vec<C64>> = (0..tones.len()).map(|m| self.phased(m, 0.0, rho)).collect();
        let mut bars = vec![Bar { channel: 0, position: 0, tone: None, matrix: rho }];
        bars.extend(phased.iter().enumerate().map(|(m, x)| Bar { channel: 1 + m, position: 0, tone: Some(m), matrix: x }));
        let forms = self.forms(path, None, &bars, 1 + tones.len());
        each(0.0, real(forms[0]));
        for (m, &nu) in tones.iter().enumerate() {
            each(nu, real(forms[1 + m]));
        }
    }

    /// The three nodes' static forms and every tone's there at its exact phase, the held state's at the beats' weights
    /// with the middle's (module documentation, # The tones); and where `end` is given, the end state's held forms at the
    /// middle, the static G's on a channel of its own and each tone's complex amplitude, referenced to the middle's time,
    /// on one each (# The held forms) -- in one contraction.
    fn nodes(&self, path: &Path, h: f64, rho: [&[C64]; 3], held: Option<(&[C64], &dyn Fn(f64) -> f64)>, end: Option<(&[C64], &Path)>) -> ([[f64; 3]; 3], Option<Forms>) {
        // The tones are the light's beats in time (the path's turns are its envelopes'): the middle's own, no path at rest.
        let end = end.map(|(rho, _)| rho);
        let tones = &self.run.clusters.sets.get(self.set).tones;
        let at = |nu: f64| -> Vec<C64> {
            match held {
                Some((held, weight)) => rho[1].iter().zip(held).map(|(r, x)| *r + *x * weight(nu)).collect(),
                None => rho[1].to_vec(),
            }
        };
        let middle = at(0.0);
        let phased: Vec<[Vec<C64>; 3]> = tones.iter().enumerate().map(|(m, &nu)| [self.phased(m, 0.0, &at(nu)), self.phased(m, -h, rho[0]), self.phased(m, h, rho[2])]).collect();
        let mut bars: Vec<Bar> = [&middle[..], rho[0], rho[2]].into_iter().enumerate().map(|(j, x)| Bar { channel: 0, position: j, tone: None, matrix: x }).collect();
        for (m, xs) in phased.iter().enumerate() {
            bars.extend(xs.iter().enumerate().map(|(j, x)| Bar { channel: 0, position: j, tone: Some(m), matrix: x }));
        }
        let ended: Vec<Vec<C64>> = end.map_or(vec![], |rho| (0..tones.len()).map(|m| self.phased(m, 0.0, rho)).collect());
        if let Some(rho) = end {
            bars.push(Bar { channel: 1, position: 0, tone: None, matrix: rho });
            bars.extend(ended.iter().enumerate().map(|(m, x)| Bar { channel: 2 + m, position: 0, tone: Some(m), matrix: x }));
        }
        let f = self.forms(path, Some(h), &bars, if end.is_some() { 2 + tones.len() } else { 1 });
        let held = end.map(|_| Forms { time: 0.0, held: real(f[1]), tones: tones.iter().zip(&f[2..]).map(|(&nu, a)| (nu, *a)).collect() });
        ([real(f[0]), [0.0; 3], [0.0; 3]], held)
    }

    /// The candidates' beats and keys (module documentation, # The tones).
    fn beats(&self, _: &Path, beats: &mut Vec<f64>, keys: &mut Vec<Vec<i64>>) {
        for c in &self.run.clusters.sets.get(self.set).candidates {
            beats.push(c.beat);
            keys.push(c.key.clone());
        }
    }

    /// Candidate m's members at the time s from the middle (`Candidate`): e^{iνt_m} times its tone G_+(s) and its cross
    /// decays at ν there, −(i/2) Σ_q L_o,q†L_o′,q (# The jumps' coherence), and e^{−iνt_m} times the same at −ν (none: zero).
    fn envelopes(&self, path: &Path, m: usize, s: f64, plus: &mut Mat, minus: &mut Mat) {
        let inner = self.made(path, (s != 0.0).then_some(s.abs()), false);
        let set = self.run.clusters.sets.get(self.set);
        let c = &set.candidates[m];
        if !inner.exists {
            plus.data.fill(C64::new(f64::NAN, f64::NAN));
            minus.data.fill(C64::new(f64::NAN, f64::NAN));
            return;
        }
        let ws = inner.ws.as_ref().unwrap();
        let at = match position(s) {
            0 => At::Middle,
            j => At::Node(j - 1),
        };
        let jumps = (!c.cross.iter().all(Vec::is_empty)).then(|| set.fold.prototype_jumps(&inner.eval, ws, at));
        for ((member, cross), out, sign) in [((c.plus, &c.cross[0]), plus, 1.0), ((c.minus, &c.cross[1]), minus, -1.0)] {
            let phase = C64::cis(sign * c.beat * self.t);
            out.data.fill(C64::ZERO);
            if let Some(k) = member {
                for (z, g) in out.data.iter_mut().zip(&ws.tone(k, position(s)).data) {
                    *z = phase * *g;
                }
            }
            for &pair in cross {
                let (i, j, _) = set.coherent[pair];
                let jumps = jumps.as_ref().expect("the jumps of a cross decay's blocks");
                let factor = C64::new(0.0, -0.5) * phase;
                for q in 0..3 {
                    let product = jumps[i][q].adjoint().product(&jumps[j][q]);
                    for (z, x) in out.data.iter_mut().zip(&product.data) {
                        *z += factor * *x;
                    }
                }
            }
        }
    }

    /// Candidate m's forms on the integrator's sums σ_j = re + i im at its beat, node by node (step.rs
    /// `Middle::tone_force`): at each node s_j its + tone's G_+(s_j) on e^{iνt_m}σ_j, its − tone's on e^{−iνt_m}σ_j† --
    /// the envelopes the propagation takes through the nodes (`envelopes`) and the hazard with them (step.rs # Jumps), one
    /// formula; a cross decay has no Hermitian part, and no form. One contraction (the outer nodes made at the sums'
    /// offset where asked).
    fn tone_force(&self, path: &Path, m: usize, sums: &[(f64, &[C64], &[C64])]) -> [f64; 3] {
        let c = &self.run.clusters.sets.get(self.set).candidates[m];
        let n = self.states();
        let members: Vec<(usize, usize, Vec<C64>)> = sums
            .iter()
            .flat_map(|&(s, re, im)| {
                [(c.plus, 1.0), (c.minus, -1.0)].into_iter().filter_map(move |(k, sign)| {
                    let phase = C64::cis(sign * c.beat * self.t);
                    k.map(|k| (k, position(s), (0..n * n).map(|i| (re[i] + C64::new(0.0, sign) * im[i]) * phase).collect()))
                })
            })
            .collect();
        if members.is_empty() {
            return [0.0; 3];
        }
        let h = sums.iter().map(|x| x.0.abs()).fold(0.0, f64::max);
        let bars: Vec<Bar> = members.iter().map(|(k, j, x)| Bar { channel: 0, position: *j, tone: Some(*k), matrix: x }).collect();
        real(self.forms(path, (h != 0.0).then_some(h), &bars, 1)[0])
    }

    fn tone_work(&self, m: usize) -> [f64; 2] {
        let c = &self.run.clusters.sets.get(self.set).candidates[m];
        let (members, cross) = (usize::from(c.plus.is_some()) + usize::from(c.minus.is_some()), c.cross[0].len() + c.cross[1].len());
        self.costs().map_or([0.0; 2], |(sizes, _)| sizes.candidate(members, cross, c.plus.or(c.minus).unwrap_or(0)))
    }

    fn scattering(&self, path: &Path, psi: &[C64]) -> (Scattering, f64) {
        let fold = &self.run.clusters.sets.get(self.set).fold;
        let mut inner = self.made(path, None, true);
        if !inner.exists {
            return (Scattering { force: [f64::NAN; 3], diffusion: [[f64::NAN; 3]; 3], size: 0.0, links: 0 }, 0.0);
        }
        let Inner { eval, ws, .. } = &mut *inner;
        let photons = fold.prototype_photons(eval, ws.as_mut().unwrap(), psi, &self.run.clusters.grounds, At::Middle, self.t, Moments::Every);
        // The terms of a diagonal entry: per class its second moment and its kick's W b², at most the entry and twice
        // the kicks' part; the products behind it: S⁻¹ψ, the responses, the emission, a class's Gram over its rows.
        let kicks: [f64; 3] = std::array::from_fn(|a| photons.classes.iter().map(|c| c.rate * c.kick[a] * c.kick[a]).sum());
        let size = (0..3).map(|a| photons.diffusion[a][a].abs() + 2.0 * kicks[a]).fold(0.0, f64::max);
        let links = fold.size() + STATES + 3 * GROUND;
        (Scattering { force: photons.force, diffusion: photons.diffusion, size, links }, photons.work * crate::decide::COMPLEX)
    }

    fn work(&self) -> [f64; 3] {
        self.costs().map_or([0.0; 3], |(sizes, route)| sizes.costs(&route))
    }

    fn jump_work(&self) -> f64 {
        self.costs().map_or(0.0, |(sizes, _)| sizes.jump())
    }

    /// Its forwards' counted work (`Inner::made`): a step's model is priced where it is made, its forward at the point
    /// with its sweeps (`step::Price`), its nodes and contractions per node (`work`).
    fn made(&self) -> f64 {
        self.inner.borrow().made
    }

    /// The set's optical sector on the path (module documentation, # The optical sector): its ground states, γ_Q -- the
    /// least of its explicit excited states' decay over two --, and the light pattern's fastest beat along the path, the
    /// largest (k_l − k_m)·v over its light's beams. None where the set holds no explicit state (a fold's photons hold that
    /// part: step.rs # The optical sector's noise).
    fn optical(&self, path: &Path) -> Option<Optical> {
        let set = &self.run.clusters.sets.get(self.set);
        let decay = set.optical?;
        let (most, least) = set.waves.iter().map(|k| (0..3).map(|a| k[a] * path.velocity[a]).sum::<f64>()).fold((f64::NEG_INFINITY, f64::INFINITY), |(most, least), x| (most.max(x), least.min(x)));
        Some(Optical { ground: set.cluster.ground, decay, beat: if most >= least { most - least } else { 0.0 } })
    }

    /// Ĝ_a(s) as matrices (module documentation, # The optical sector): the static G's and every tone's at its phase there,
    /// e^{iν_m(t_m + s)} (`Middle::force`'s, its contraction), of the step's derivative in its forward mode (proto.rs
    /// `Fold::prototype_gradient`; at a time s off the middle the nodes made there, as `generator`'s); and where asked
    /// their slope along the path at the middle (the step's, `step::optical`), through the step's outer nodes -- the
    /// parabola the propagation takes (spectral.rs `Picture::slope`): the static G's and each tone's envelope by their
    /// chord (Ĝ(h) − Ĝ(−h))/2h, each tone's phase by its derivative iν_m. In the set's frames, the lab's (no Doppler
    /// frame: `gauge`).
    fn gradient(&self, path: &Path, s: f64, g: &mut [Vec<C64>; 3], slope: Option<&mut [Vec<C64>; 3]>) -> f64 {
        assert!(s == 0.0 || slope.is_none(), "the gradient's slope at the middle (the step's optical sector's)");
        let set = &self.run.clusters.sets.get(self.set);
        let n = self.states();
        let ends = if s != 0.0 { Some(s.abs()) } else { slope.as_ref().map(|_| self.inner.borrow().half.expect("the step's outer nodes: its picture's ends")) };
        let mut inner = self.made(path, ends, true);
        if !inner.exists {
            for m in g.iter_mut().chain(slope.into_iter().flat_map(|x| x.iter_mut())) {
                m.clear();
                m.resize(n * n, C64::new(f64::NAN, f64::NAN));
            }
            return 0.0;
        }
        let Inner { eval, ws, .. } = &mut *inner;
        let ws = ws.as_mut().expect("a workspace");
        let phases = |u: f64, turned: bool| -> Vec<C64> { set.tones.iter().map(|&nu| C64::cis(nu * (self.t + u)) * if turned { C64::new(0.0, nu) } else { C64::ONE }).collect() };
        let mut counted = set.fold.prototype_gradient(eval, ws, position(s), true, &phases(s, false), g);
        if let Some(slope) = slope {
            let h = ends.expect("the outer nodes");
            let mut minus: [Vec<C64>; 3] = Default::default();
            counted += set.fold.prototype_gradient(eval, ws, 1, true, &phases(0.0, false), &mut minus);
            counted += set.fold.prototype_gradient(eval, ws, 2, true, &phases(0.0, false), slope);
            for (plus, minus) in slope.iter_mut().zip(&minus) {
                plus.iter_mut().zip(minus).for_each(|(p, m)| *p = (*p - *m) / (2.0 * h));
            }
            if !set.tones.is_empty() {
                let mut turned: [Vec<C64>; 3] = Default::default();
                counted += set.fold.prototype_gradient(eval, ws, 0, false, &phases(0.0, true), &mut turned);
                for (o, t) in slope.iter_mut().zip(&turned) {
                    o.iter_mut().zip(t).for_each(|(o, t)| *o += *t);
                }
            }
        }
        counted
    }

    /// Its left response forgotten (`step::Middle::forget_left`).
    #[cfg(test)]
    fn forget_left(&self) {
        if let Some(ws) = self.inner.borrow_mut().ws.as_mut() {
            ws.forget_left();
        }
    }

    fn books(&self, path: &Path, psi: &[C64]) -> Option<([f64; KINDS], bool, f64)> {
        let inner = self.made(path, None, false);
        if !inner.exists || self.run.clusters.sets.get(self.set).fold.blocks.is_empty() {
            return None;
        }
        let gamma = -2.0 * psi.iter().zip(inner.generator.apply(psi)).map(|(a, b)| a.conj() * b).sum::<C64>().im / crate::linalg::norm_sqr(psi);
        Some((inner.kinds, inner.unresolved, gamma))
    }

    fn held(&self) -> Option<Held> {
        if !self.inner.borrow().decided {
            return None;
        }
        Some(Held(Some(Arc::new(self.leaves()))))
    }

    fn exists(&self, path: &Path) -> bool {
        self.run.clusters.sets.get(self.set).fold.blocks.is_empty() || self.made(path, None, false).exists
    }

    /// The decision at a step's end, on a middle made deciding at the end's point (# The representation and the decision
    /// times; step.rs `Midway::decided`): its forward reads and makes its decompositions again at the point, the
    /// trajectory's from here; the cell's books' end. A reading beyond the share there, or a fold that does not exist
    /// there, is counted (adapt.rs `counted`): the representation is the start's (# The representation). Nothing where this middle
    /// does not decide or holds no light.
    fn decide(&self, path: &Path, psi: &[C64], interval: f64) -> Decided {
        let mut out = Decided { held: None, work: 0.0, booked: [0.0; crate::step::MODEL] };
        let set = self.run.clusters.sets.get(self.set);
        if !self.decide || set.fold.blocks.is_empty() {
            return out;
        }
        let _timed = Timed(std::time::Instant::now(), set.cluster.states.len());
        let exists = self.made(path, None, false).exists;
        if exists {
            let inner = self.inner.borrow();
            // The state's decay rate on the generator, i⟨ψ|G − G†|ψ⟩/‖ψ‖².
            let gamma = -2.0 * psi.iter().zip(inner.generator.apply(psi)).map(|(a, b)| a.conj() * b).sum::<C64>().im / crate::linalg::norm_sqr(psi);
            // The cell this decision ends (unify2 §2.7): its end measured, the difference from its start's booked linearly.
            let r = |rho2: f64| 2.0 * rho2.sqrt() + rho2;
            for k in 0..KINDS {
                out.booked[k] = 0.5 * (r(inner.ended[k]) - r(inner.started[k])) * gamma * interval;
            }
            out.booked[KINDS] = 0.5 * (r(inner.ended.iter().sum()) - r(inner.started.iter().sum())) * gamma * interval;
            crate::adapt::count(crate::adapt::BEYOND, u64::from(inner.beyond));
        } else {
            crate::adapt::count(crate::adapt::ABSENT, 1);
        }
        out.held = Some(Held(Some(Arc::new(self.leaves()))));
        out
    }
}

impl crate::step::Dynamics for Onefold {
    type Middle<'a>
        = Middle<'a>
    where
        Self: 'a;

    fn hbar_over_mass(&self) -> f64 {
        self.hbar_over_mass
    }

    /// V_ref's second-order expansion at r (module documentation, # The rest of the model; `Reference`).
    fn reference(&self, r: [f64; 3]) -> Quadratic {
        self.reference.at(&self.light, r)
    }

    fn cluster(&self, cluster: usize) -> &Cluster {
        &self.clusters.sets.get(cluster).cluster
    }

    fn cluster_of(&self, ground: usize) -> usize {
        self.clusters.membership[ground]
    }

    fn middle(&self, cluster: usize, r: [f64; 3], t: f64, held: &Held, decide: bool) -> Middle<'_> {
        let committed = self.cache(held, cluster);
        let n = self.clusters.sets.get(cluster).fold.size();
        let ws = (!self.clusters.sets.get(cluster).fold.blocks.is_empty()).then(|| self.clusters.take(cluster));
        let caches = held.0.as_ref().and_then(|h| h.clone().downcast::<Caches>().ok());
        Middle {
            run: self,
            set: cluster,
            r,
            t,
            decide,
            caches,
            inner: RefCell::new(Inner {
                eval: Cache::default(),
                committed,
                ws,
                current: None,
                nodes: None,
                half: None,
                tangent: false,
                decided: false,
                exists: true,
                generator: Mat::zeros(n, n),
                made: 0.0,
                reading: 0.0,
                beyond: false,
                kinds: [0.0; KINDS],
                started: [0.0; KINDS],
                ended: [0.0; KINDS],
                unresolved: false,
            }),
        }
    }

    /// A jump's photon (`Middle::emitted`): on the step's middle at the jump's time from it, or at a point (a trajectory's
    /// start) on a middle made there, at its own time.
    fn emit<'m>(&'m self, cluster: usize, at: &Point, on: Option<(&Middle<'m>, &Path, f64)>, psi: &[C64], held: &Held, draw: &mut dyn FnMut(&[f64]) -> Option<usize>) -> Option<Emitted> {
        match on {
            Some((middle, path, s)) => middle.emitted(path, s, psi, draw),
            None => {
                let middle = self.middle(cluster, at.r, at.t, held, false);
                let path = middle.path(at.kappa, [0.0; 3]);
                middle.emitted(&path, 0.0, psi, draw)
            }
        }
    }

    /// A jump's photon with what its landing's transient owes it (`Middle::transient`): on the step's middle at the jump's
    /// time, or at a point on a middle made there.
    fn transient<'m>(&'m self, cluster: usize, at: &Point, on: Option<(&Middle<'m>, &Path, f64)>, psi: &[C64], owed: &Owed, held: &Held, draw: &mut dyn FnMut(&[f64]) -> Option<usize>) -> Option<Emitted> {
        match on {
            Some((middle, path, s)) => middle.transient(path, s, psi, owed, draw),
            None => {
                let middle = self.middle(cluster, at.r, at.t, held, false);
                let path = middle.path(at.kappa, [0.0; 3]);
                middle.transient(&path, 0.0, psi, owed, draw)
            }
        }
    }

    /// The held forms at a point, deciding nothing (# The held forms): on one evaluation there.
    fn forms(&self, cluster: usize, at: &Point, psi: &[C64], held: &Held) -> Forms {
        if self.clusters.sets.get(cluster).fold.blocks.is_empty() {
            return self.held_forms(cluster, at, psi, None, false).1;
        }
        self.evaluate(cluster, at.r, at.kappa, held, |_, cache, ws, _| self.held_forms(cluster, at, psi, Some((cache, ws)), false).1).unwrap_or_else(|| Forms { time: at.t, held: [f64::NAN; 3], tones: vec![] })
    }

    /// The bare state's landing and its held forms in one evaluation, the trajectory's first decision (unify2 §4.1): its
    /// cluster -- its set's with nothing explicit, the walk's rises taken there with the look over the first cell `cell`
    /// (`Onefold::landed_at`) --, and its decompositions made there its representation (# The representation and the
    /// decision times).
    fn started(&self, cluster: usize, at: &Point, f: &[C64], _cell: f64, threshold: f64) -> (usize, Landed, Forms, Held) {
        let k = self.clusters.sets.get(cluster).ground_set;
        let from = self.clusters.sets.get(cluster).cluster.states.clone();
        match self.landed_at(k, at, f, &from, &Held::default(), threshold) {
            Some((cluster, landed, forms, caches)) => (cluster, landed, forms, Held(Some(Arc::new(caches)))),
            None => (cluster, Landed::plain(f.iter().map(|_| C64::new(f64::NAN, f64::NAN)).collect(), threshold), Forms { time: at.t, held: [f64::NAN; 3], tones: vec![] }, Held::default()),
        }
    }

    /// Where a jump's bare state lands in its own set against the threshold θ (`Middle::landing`): on the landing set's
    /// middle in the step, or at a point on a middle made there; nowhere where it does not exist (`step::Landing`).
    fn landed<'m>(&'m self, cluster: usize, at: &Point, on: Option<(&Middle<'m>, &Path)>, f: &[C64], held: &Held, threshold: f64) -> Landing {
        match on {
            Some((middle, path)) => middle.landing(path, f, threshold, at.t),
            None => {
                let middle = self.middle(cluster, at.r, at.t, held, false);
                let path = middle.path(at.kappa, [0.0; 3]);
                middle.landing(&path, f, threshold, at.t)
            }
        }
    }

    /// (ħ/m)κ²/2 + V_run(r; p(ψ)) (module documentation, # The rest of the model): the two light sets' on one set of
    /// explicit nodes, risen where either fold does not exist at the point (`Clusters::difference`); NaN only where no node
    /// is left to rise, or the request with the trap's light off makes no clusters.
    fn energy(&self, cluster: usize, at: &Point, psi: &[C64]) -> f64 {
        let kinetic = 0.5 * self.hbar_over_mass * at.kappa.iter().map(|k| k * k).sum::<f64>();
        let p = self.populations(cluster, psi);
        let untrapped = self.untrapped.get_or_init(|| {
            let request = lit(&self.clusters.request, &self.light, |w| w != self.light.trap);
            // The run's terms: both light sets on one set of explicit nodes and kept terms (`Clusters::difference`).
            Clusters::new(&request, &Choice::base(&request, vec![], self.clusters.base.toned.clone())?, self.clusters.floor)
        });
        let potential = untrapped.as_ref().ok().and_then(|u| self.clusters.difference(u, at.r, &p)).unwrap_or(f64::NAN);
        kinetic + potential
    }

    /// Outside the trap's region with the loss energy above its far value, zero (module documentation, # The rest of the
    /// model).
    fn lost(&self, cluster: usize, at: &Point, psi: &[C64]) -> bool {
        self.light.outside(at.r) && self.energy(cluster, at, psi) > 0.0
    }

    /// The fold's time-averaged populations of the state at the point (DESIGN §1.3; proto.rs `Workspace::populations`):
    /// the populations of the slow modes between two jumps -- not the optical Bloch equations' ensemble ρ_ee, which holds
    /// each jump's transient that the fold's landing leaves out (module documentation, # The rest of the model) --; one
    /// forward there on the trajectory's representation, deciding nothing; a set with no light its states' own. Over their
    /// sum, ψ†ψ by construction.
    fn readout(&self, cluster: usize, at: &Point, psi: &[C64], held: &Held) -> [f64; STATES] {
        let mut out = [0.0; STATES];
        let set = &self.clusters.sets.get(cluster);
        let total = if set.fold.blocks.is_empty() {
            for (z, &s) in psi.iter().zip(&set.cluster.states) {
                out[s] = z.norm_sqr();
            }
            crate::linalg::norm_sqr(psi)
        } else {
            self.evaluate(cluster, at.r, at.kappa, held, |fold, cache, ws, _| ws.populations(fold, cache, psi, &mut out)).unwrap_or(f64::NAN)
        };
        out.map(|x| x / total)
    }

    /// A jump into another set (unify2 §4.1: a decision time at the jump's own point and time): the decision there on the
    /// cluster the trajectory last held for that set (`Middle::landing_cluster`'s rule), its look over the cell to the
    /// next decision (`until`), its rises taken, the bare state landed on it (`Onefold::landed_at`). None where the jump
    /// stays in the emitting cluster's set (its landing the step's: `landed`).
    fn entered(&self, from: usize, cluster: usize, at: &Point, f: &[C64], held: &Held, _until: f64, threshold: f64) -> Option<Option<(usize, Internal, Landed, Held)>> {
        let k = self.clusters.sets.get(cluster).ground_set;
        if k == self.clusters.sets.get(from).ground_set {
            return None;
        }
        let states = self.clusters.sets.get(cluster).cluster.states.clone();
        Some(self.landed_at(k, at, f, &states, held, threshold).map(|(c, landed, _, caches)| {
            let to = &self.clusters.sets.get(c).cluster.states;
            let bare: Internal = to.iter().map(|s| states.iter().position(|x| x == s).map_or(C64::ZERO, |i| f[i])).collect();
            (c, bare, landed, Held(Some(Arc::new(caches))))
        }))
    }
}
