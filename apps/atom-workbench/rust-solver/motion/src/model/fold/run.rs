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
//! # The nodes risen and fallen (DESIGN §11; BRIEF_STEP3C item 1)
//! Which excited nodes are explicit is decided at the run's decision times (a middle made deciding: `Middle::decide`),
//! by the evaluation order's rule (M18) on P and Q: the cheapest representation within the share. The fold is within the
//! share wherever it exists (its series sweeps until its rest reads within it), so the choice is the count: a step's
//! price on each side (the integrator's, `step::Offer`), a switch priced as a restart -- the other side's forward at the
//! point and the held forms there. The fold's count grows without bound as its series' ratio goes to one (the sweeps
//! go as ln(share)/ln r), so a node rises before its fold ceases to exist; no constant holds a side. A decision offers
//! two: the rise of the Q node that holds the most of the state (its population there), and the fall of the explicit
//! node that holds the least -- each among the nodes whose cluster can be made: a refused cluster is kept refused
//! (`Arena`), its price infinite, so no decision offers it again. Each side is priced by the count of a forward the
//! trajectory last measured on it (`Caches::counts`; where it measured none, its cluster's at the focus, made or not):
//! the step's own forward on its cluster at every decision, and an attempt's. A decision takes the cheaper offer where
//! it is cheaper than staying by these counts, its forward made within the budget that keeps it so
//! (`Fold::prototype_within`); the attempt is a count like any other and is kept: where it gives the side within the
//! share, what it counted; where it gives none within its budget (over the budget, the fold not existing there, or
//! reading beyond the share), the budget or what it spent, whichever is more -- the least that side may cost by what
//! was measured, at which its offer is no cheaper than staying was. So a side found dearer is tried again only where
//! staying has grown dearer than it was then, and an attempt wastes at most the budget, within a step's own price: no
//! constant, no hysteresis. A fall is taken only where what it drops is within the share
//! (DESIGN §11.4): with the trajectory's owed balance b (`Trajectory::owed`: what earlier switches dropped and the
//! share has not paid since, each interval paying (share − ρ)γΔt off it), b + δp + ρ′γΔt ≤ share·γΔt, ρ′ the folded
//! side's reading, γ the state's decay rate, Δt the interval -- so a trajectory that flips books δp at each fall and its
//! balance stops the next one: the flips stop by what they book, no hysteresis.
//!
//! A switch is one formula (`Workspace::taken`; proto.rs # The landing): the state lifted on the side it leaves, taken
//! onto the other's slow span along its fast one, ψ′ = S′a′, a′ = N′⁻¹Φ_L′†Ψ at the middle, not normalized. Risen, the
//! node's rows become P amplitudes x_X = T_X x, every Q row following T: exact (N′x, so S′x: the metric held T_X†T_X as
//! the explicit amplitude's square). Fallen, its rows are given their explicit amplitudes, a′ = x + N′⁻¹T_L,X†(x_X −
//! T_X x), and the norm's change δp = ‖ψ‖² − ‖ψ′‖² over ‖ψ‖² -- the fast part's net emission, of either sign -- is
//! booked (`Trajectory::dropped`); a norm at the trajectory's threshold or below is that photon at once (step.rs
//! `Midway::decided`). A jump's landing is the same formula with nothing following (a bare state, S N⁻¹f): a fall at
//! x_X = 0. A jump lands in its emitting cluster where it lands in that ground set, else in the
//! cluster the trajectory last held for the landing set (`Caches::chosen`), else in that set's with nothing explicit.
//! A trajectory starts on its set's cluster with nothing explicit where its fold exists there, else with the candidate
//! nodes risen in turn until it does (`Dynamics::started`).
//!
//! # The representation and the decision times
//! An atom's kept decompositions (proto.rs `Cache`), per set, are its own: the trajectory carries them as a value
//! (`step::Held`). They change only where the run says a decision is made (`Dynamics::middle` with `decide`; DESIGN
//! §11.3): there the middle's first evaluation joins or lets go the levels and makes the decompositions again as its
//! readings ask, and what it decides is the trajectory's from then on (`Middle::held`). Every other evaluation -- the
//! step's path at its own momentum, a set a jump lands in, the photon and the landing at a jump -- starts from the
//! trajectory's representation and decides nothing that outlives it. Here a decision is made at every step's middle,
//! where a trajectory starts (`Dynamics::started`: the landing of its bare state, the held forms there -- its first
//! decomposition made once), and where a jump lands it in another set (step.rs `Places`: the landing set's middle, on
//! which the step goes on -- its first decomposition there made once and kept); the grid of the coarsest level's step
//! ends is step 3's. The representation is per set a shared value (`Caches`): a decision replaces its own set's.
//!
//! # The held forms
//! A step's path momentum is predicted from the start state's held forms (step.rs # The path's momentum): the end state's
//! at the last step's middle, in the same contraction as the step's kick (`Middle::nodes`: one more density at the
//! middle, its static G's form on a channel of its own and each tone's complex amplitude on one each, referenced to the
//! middle's time; the integrator takes the reference's gradient from them); at a point -- a trajectory's start, a jump at
//! a step's end into another set -- one evaluation there, its tangent and the contraction at the point, less the
//! reference's gradient there (`Onefold::held_forms`).
//!
//! # The tones
//! What stays in P and beats -- an entry of two P states of different frames (proto.rs # The tones) -- is the
//! generator's turning part: at the time t the generator is G + Σ_m e^{iν_m t} G_m, each tone's envelope G_m at the
//! step's three nodes its own (the light at the path's points there, the middle's S). The integrator's candidates
//! (`Middle::beats`) are the generator's whole parts at each exact beat and its opposite -- the tones' pairs, ν and −ν,
//! and the coherence groups' cross decays at those beats (# The jumps' coherence; `Candidate`) --, each keyed by its
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
//! # The jumps' coherence (BRIEF_WIRE3 item 3)
//! A set's blocks whose explicit states P holds a tone between are one coherence group (fold.rs `Fold::groups`, # Tones):
//! their jumps are one operator per landing set, L_q(t) = C_q Σ_o e^{−iot}ε_o S⁻¹ over the group (DESIGN §1.3), so the
//! photons' classes are per group (proto.rs `Fold::prototype_photons` at the time) and a jump's amplitudes its class's
//! photons' own, its group's blocks summed at their phases (`Middle::emitted`, proto.rs `Workspace::jumped`); and the
//! decay between the offsets is its cross products, added to the
//! generator at their beats at every position (`step::Middle::generator`): i(G − G†) = Σ_q L_q(t)†L_q(t) at every time,
//! static and beating, by construction. Each cross decay is part of the integrator's candidate of its exact beat -- its
//! key the offsets' exact coordinates o − o′, with the tone of P between the two blocks' explicit states where there is
//! one, at the beat's same key --, so that the integrator takes the decay's beat as it takes a tone (spectral.rs, # Which
//! beats are tones) and the survival's hazard at its exact phase (step.rs, # Jumps): not through the step's three nodes,
//! which alias it beyond νh = π. Every block of the scenes is a group of its own.
//!
//! # The jumps' phase
//! A jump lands on the lab frame's ground states, C_q Σ e^{−iot} ε_o S⁻¹x: in the landing set's coordinates, each of its
//! ground states s at its own frame θ_s, the landed amplitude of s is e^{i(θ_s − o)t} (C_q ε_o S⁻¹x)_s -- one block a
//! class (fold.rs # Tones), so e^{−iot} is the state's global phase and e^{iθ_s t} what a set of several frames holds
//! between its states (`Dynamics::emit`). The rates, the force and the diffusion are sums over the states of |u_s|² and
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
//!   `Clusters`), each ground set's on the cluster the bare state √p first holds at the point at rest -- with nothing
//!   explicit where its fold exists there, else risen until it does, the rule of a trajectory's start
//!   (`Clusters::rise`, `Clusters::diagonal`) -- its diagonal in the lab (G's less its frame), p the state's ground
//!   populations (an explicit excited state's by its decay's branching). On a risen cluster the risen node's light shift
//!   is in its coupling to the ground places, not on their diagonal: of both light sets alike where both rise it, so
//!   that their difference keeps the trap's. Its far value is zero: the trap's frequency holds no plane wave
//!   (ensemble.rs `PLANE_WAVE` refuses one).
//! The displayed rows are the fold's time-averaged populations at the point (`Dynamics::readout`): the populations of
//! the slow modes between two jumps (proto.rs `Workspace::populations`) -- not the optical Bloch equations' ensemble
//! ρ_ee, which holds each jump's transient over 1/Γ that the fold's landing leaves out (within the share, read by the
//! landing's reading).
//!
//! # The optical sector (step.rs # The optical sector's noise; DESIGN §1.3, R23)
//! A set that holds explicit excited states has an optical sector (`Middle::optical`): its ground states, γ_Q the least
//! of its explicit states' decay over two, and the light pattern's fastest beat along the path over its light's beams.
//! Its K is of the gradient's matrices Ĝ_a (`Middle::gradient`), not of their contraction with one density: the same
//! derivative of the step as the force's, in its forward mode (proto.rs `Fold::prototype_gradient`: the step's tangent
//! through the compression) -- the force contracts it in the reverse mode, each where it is the cheap one. Their slope
//! along the path is the parabola through the step's outer nodes the propagation takes, a tone's phase exact. A set of
//! ground states alone has none: its fold's photons hold that part.
use super::proto::{At, Bar, Cache, Moments, Workspace};
use super::{Choice, Fold};
use crate::field::Light;
use crate::linalg::{Mat, C64};
use crate::model::{Cluster, Point, Quadratic, Scattering};
use crate::request::{Request, GROUND, STATES};
use crate::step::{Decided, Dynamics as _, Emitted, Forms, Held, Internal, Middle as _, Offer, Optical, Switched};
use std::cell::{RefCell, RefMut};
use std::sync::{Arc, Mutex, OnceLock};

/// An explicit excited node: its level, the absorbed component's part and a source's ground state (`Choice::explicit`).
type Key = (usize, crate::model::Component, usize);

/// One cluster of the run (# The state sets): its ground set's index and its explicit nodes, its fold, its cluster, the
/// counted work of a step's middle at the focus (`step::Middle::work`: a node's generator, a node's force and tones, a
/// form; and the forward's own, `Middle::made`, made or not -- a trajectory's price of the cluster where it measured none
/// there, # The nodes risen and fallen), its levels' bandwidth (`step::Offer`), the ways a decision may take from it
/// (`Rise`, `Fall`), and the workspaces its evaluations take in turn.
struct Set {
    ground_set: usize,
    nodes: Vec<Key>,
    fold: Fold,
    cluster: Cluster,
    work: [f64; 3],
    made: f64,
    bandwidth: f64,
    rises: Vec<Rise>,
    falls: Vec<Fall>,
    pool: Mutex<Vec<Workspace>>,
    /// The tones' values ν_m (rad/µs; proto.rs `Workspace::tones`' order); the integrator's candidates (`Candidate`).
    tones: Vec<f64>,
    candidates: Vec<Candidate>,
    /// The pairs of the jumps' blocks (their order: the near blocks', then the far ones', as the fold's blocks) of one
    /// coherence group (fold.rs `Fold::groups`), each with its beat o − o′ (rad/µs: its candidate's, `Candidate::cross`):
    /// the decay between their offsets (# The jumps' coherence).
    coherent: Vec<(usize, usize, f64)>,
    /// Its optical sector (# The optical sector): γ_Q, the least of its explicit excited states' decay over two (none
    /// where it holds none), and its light's beams' momenta (the fold's components' beams: the pattern's beats).
    optical: Option<f64>,
    waves: Vec<[f64; 3]>,
}

/// A Q node a decision may raise (# The nodes risen and fallen; `Choice::candidates`): its key, its block and rows, and the
/// cluster it rises to -- made where first asked for and kept with it, None where refused.
struct Rise {
    key: Key,
    block: usize,
    rows: Vec<usize>,
    cluster: OnceLock<Option<usize>>,
}

/// An explicit node a decision may let fall: its key, its P places, and the cluster without it (as `Rise`'s).
struct Fall {
    key: Key,
    places: Vec<usize>,
    cluster: OnceLock<Option<usize>>,
}

/// A candidate tone of the integrator (step.rs `Middle::beats`): the generator's whole part at one exact beat and its
/// opposite (# The tones, # The jumps' coherence) -- a pair of the set's tones at ν and −ν (either may be none) and the
/// coherence groups' cross decays at them (the indices of `Set::coherent`, at ν and at −ν) --, the beat's exact key (its
/// coordinates on the set's common denominator), and the counted complex multiply-adds of its envelopes (the three
/// nodes', the middle's phase) and of its forms (`Middle::tone_work`).
struct Candidate {
    beat: f64,
    plus: Option<usize>,
    minus: Option<usize>,
    cross: [Vec<usize>; 2],
    key: Vec<i64>,
    work: [f64; 2],
}

/// An atom's representation (`step::Held`): per cluster its kept decompositions, each shared -- a decision replaces one
/// cluster's, a middle reads its own without copying the others' --, and the count of a forward the trajectory last
/// measured there (real multiply-adds: a decision's price of it, # The nodes risen and fallen); per ground set the
/// cluster it last took there (where a jump into that set lands).
#[derive(Clone, Default)]
struct Caches {
    sets: Vec<Arc<Cache>>,
    counts: Vec<Option<f64>>,
    chosen: Vec<Option<usize>>,
}

impl Caches {
    /// A cluster's decompositions and its count.
    fn keep(&mut self, cluster: usize, cache: Arc<Cache>, count: f64) {
        if self.sets.len() <= cluster {
            self.sets.resize(cluster + 1, Arc::default());
            self.counts.resize(cluster + 1, None);
        }
        (self.sets[cluster], self.counts[cluster]) = (cache, Some(count));
    }

    /// The cluster taken in ground set k.
    fn choose(&mut self, k: usize, cluster: usize) {
        if self.chosen.len() <= k {
            self.chosen.resize(k + 1, None);
        }
        self.chosen[k] = Some(cluster);
    }
}

/// The run's clusters (# The state sets), made where first asked for and kept: an append-only list whose entries stay
/// where they are made (chunk c holds 2^c of them), so that a reference to one lives as long as the run; and every key
/// asked for, its ground set and explicit nodes, with its cluster or why it was refused -- a refusal kept as a cluster
/// is, so that no key is made twice.
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
    reference: Reference,
    untrapped: OnceLock<Result<Clusters, String>>,
    photons: f64,
}

impl Set {
    /// The cluster of ground set `k` (`ground`) with the explicit nodes `nodes` on a choice of nodes that keeps them
    /// (module documentation, # The state sets): its fold and cluster, its counts at the focus at rest, its tones'
    /// candidates, coherence groups and optical sector.
    fn new(request: &Request, choice: &Choice, k: usize, ground: &[usize], nodes: Vec<Key>) -> Result<Set, String> {
        let decay = super::decay(request);
        let fold = Fold::new(request, choice, ground[0])?;
        let mut states: Vec<usize> = fold.ground.clone();
        states.extend(fold.explicit.iter().map(|&x| GROUND + x));
        // The levels' bandwidth: the largest |λ_a* − λ_b| of the states' energies in their frames, the excited ones' with
        // their decay (solve.rs `bandwidth`).
        let levels: Vec<C64> = states
            .iter()
            .zip(&fold.frame_values)
            .map(|(&s, theta)| C64::new(request.hamiltonian[(s, s)].re - theta, if s < GROUND { 0.0 } else { -0.5 * decay[(s - GROUND, s - GROUND)].re }))
            .collect();
        let bandwidth = levels.iter().flat_map(|a| levels.iter().map(move |b| (a.conj() - *b).abs())).fold(0.0, f64::max);
        let cluster = Cluster { ground: fold.ground.len(), frames: fold.frames.clone(), waves: vec![[0.0; 3]; states.len()], waves_exact: vec![crate::model::Beat::whole(vec![]); states.len()], states };
        let (pool, work, made, tones, candidates, coherent) = if fold.blocks.is_empty() {
            (vec![], [0.0; 3], 0.0, vec![], vec![], vec![])
        } else {
            let mut ws = fold.prototype_workspace()?;
            let (near, far) = ws.layout();
            let blocks: Vec<usize> = near.iter().map(|x| x.0).chain(far.iter().map(|x| x.0)).collect();
            let n = fold.size();
            let beats: Vec<(crate::model::Beat, f64)> = ws.tones().map(|(b, v)| (b.clone(), v)).collect();
            // The pairs of one coherence group's blocks, each with its exact beat o − o′ and its value.
            let pairs: Vec<(usize, usize)> = (0..blocks.len()).flat_map(|i| (0..blocks.len()).map(move |j| (i, j))).filter(|&(i, j)| i != j && fold.groups[blocks[i]] == fold.groups[blocks[j]]).collect();
            let crossing: Vec<(crate::model::Beat, f64)> = pairs.iter().map(|&(i, j)| (fold.blocks[blocks[i]].offset.plus(-1, &fold.blocks[blocks[j]].offset), fold.blocks[blocks[i]].value - fold.blocks[blocks[j]].value)).collect();
            // The counts of a middle at the focus at rest: its forward (made or not) and, where it is made, its nodes, its
            // tangent and force with every tone's at the three nodes; a tone's forms on the integrator's sums.
            let mut cache = Cache::default();
            let exists = fold.prototype([0.0; 3], [0.0; 3], &mut cache, &mut ws).is_some();
            let made = ws.spent() * crate::decide::COMPLEX;
            let counted = exists.then(|| {
                let nodes = fold.prototype_nodes([0.0; 3], [0.0; 3], [0.0; 3], [-0.5, 0.5], &cache, &mut ws);
                let tangent = fold.prototype_tangent(&cache, &mut ws);
                let zero = vec![C64::ZERO; n * n];
                let bar = |position: usize, tone: Option<usize>| Bar { channel: 0, position, tone, matrix: &zero };
                let mut bars: Vec<Bar> = (0..3).map(|j| bar(j, None)).collect();
                bars.extend((0..beats.len()).flat_map(|m| (0..3).map(move |j| (m, j))).map(|(m, j)| bar(j, Some(m))));
                let force = fold.prototype_force(&cache, &mut ws, &bars, 1).work;
                // A candidate's envelopes: its tones' at the three nodes, and per cross decay its three products L†L there
                // (GROUND × n each, `Middle::envelopes`); its forms: its tones' at the three nodes (a cross decay has no
                // Hermitian part).
                let candidates = candidates(&beats, &crossing, |plus, minus, cross| {
                    let one: Vec<Bar> = [plus, minus].iter().filter(|&&x| x).flat_map(|_| (0..3).map(|j| bar(j, Some(0)))).collect();
                    let forms = if one.is_empty() { 0.0 } else { fold.prototype_force(&cache, &mut ws, &one, 1).work };
                    [(3 * one.len() * n * n + 9 * cross * (GROUND + 1) * n * n) as f64, forms]
                });
                ([nodes / 3.0, (tangent + force) / 3.0, tangent + force], candidates)
            });
            let (work, candidates) = counted.unwrap_or_else(|| ([0.0; 3], candidates(&beats, &crossing, |_, _, _| [0.0; 2])));
            // Each pair turned at its candidate's beat, as the candidate's members are (`Middle::generator`).
            let mut coherent: Vec<(usize, usize, f64)> = pairs.iter().map(|&(i, j)| (i, j, 0.0)).collect();
            for c in &candidates {
                for (side, sign) in c.cross.iter().zip([1.0, -1.0]) {
                    side.iter().for_each(|&k| coherent[k].2 = sign * c.beat);
                }
            }
            (vec![ws], work, made, beats.iter().map(|b| b.1).collect(), candidates, coherent)
        };
        let optical = fold.explicit.iter().map(|&x| 0.5 * decay[(x, x)].re).reduce(f64::min);
        let waves = fold.components.iter().flat_map(|c| c.kicks.iter().copied()).collect();
        let rises = Choice::candidates(&fold).into_iter().map(|(key, block, rows)| Rise { key, block, rows, cluster: OnceLock::new() }).collect();
        let ng = fold.ground.len();
        let falls = nodes.iter().map(|&key| Fall { key, places: (ng..fold.size()).filter(|&p| fold.excited_level[fold.explicit[p - ng]] == key.0).collect(), cluster: OnceLock::new() }).collect();
        Ok(Set { ground_set: k, nodes, fold, cluster, work, made, bandwidth, rises, falls, pool: Mutex::new(pool), tones, candidates, coherent, optical, waves })
    }

    /// Its price's counts (`step::Offer`): its states, ground states, bandwidth and counts at the focus, with a forward's
    /// count `made`.
    fn offer(&self, made: f64) -> Offer {
        Offer { states: self.cluster.states.len(), ground: self.cluster.ground, bandwidth: self.bandwidth, costs: self.work, made }
    }
}

impl Onefold {
    /// The run of a request (module documentation), the splitting's reference state's ground populations `reference`:
    /// every ground set's cluster with nothing explicit first (`Choice::base`).
    pub(crate) fn new(request: &Request, reference: [f64; GROUND]) -> Result<Onefold, String> {
        Onefold::of(request, &Choice::base(request, vec![])?, reference)
    }

    /// The run of a request on a base choice of nodes (`new`: nothing explicit): its clusters (`Clusters::new`). Refused
    /// where a set is not the prototype's (proto.rs `prototype_workspace`).
    pub(crate) fn of(request: &Request, choice: &Choice, reference: [f64; GROUND]) -> Result<Onefold, String> {
        let light = Light::new(request);
        let axes = super::Axes::new(request)?;
        let energies: Vec<f64> = (0..GROUND).map(|s| request.hamiltonian[(s, s)].re).collect();
        let mut run = Onefold {
            clusters: Clusters::new(request, choice)?,
            hbar_over_mass: crate::units::hbar_over_mass(request.mass_kg),
            frame: axes.omega(&axes.frame()),
            energies,
            reference: Reference::new(&lit(request, &light, |w| w == light.trap), &light, &reference)?,
            untrapped: OnceLock::new(),
            light,
            photons: 0.0,
        };
        // A jump's photons at the focus, the dearest base cluster's, as a jump at a point makes them (`Dynamics::emit`):
        // its forward, tangent, the place and the photons there -- every class's rate and the dearest class's moments,
        // the class a jump draws (`Fold::prototype_photons`) -- (`Dynamics::photon_work`).
        run.photons = (0..run.clusters.grounds.len())
            .filter(|&k| !run.clusters.sets.get(k).fold.blocks.is_empty())
            .map(|k| {
                let n = run.clusters.sets.get(k).fold.size();
                let psi: Vec<C64> = (0..n).map(|x| if x == 0 { C64::ONE } else { C64::ZERO }).collect();
                run.evaluate(k, [0.0; 3], [0.0; 3], &Held::default(), |fold, cache, ws, forward| {
                    let made = forward + fold.prototype_tangent(cache, ws) + fold.prototype_place([0.0; 3], [0.0; 3], [0.0; 3], 0.0, ws);
                    let (mut dearest, mut class, mut classes) = (0.0_f64, 0, 1);
                    while class < classes {
                        let mut drawn = |rates: &[f64]| {
                            classes = rates.len();
                            (class < classes).then_some(class)
                        };
                        dearest = dearest.max(fold.prototype_photons(cache, ws, &psi, &run.clusters.grounds, At::Place, 0.0, Moments::Drawn(&mut drawn)).work);
                        class += 1;
                    }
                    made + dearest
                })
                .unwrap_or(0.0)
            })
            .fold(0.0, f64::max)
            * crate::decide::COMPLEX;
        Ok(run)
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
        let mut cache = (*self.cache(held, set)).clone();
        let mut ws = self.clusters.take(set);
        let exists = fold.prototype(r, kappa, &mut cache, &mut ws).is_some();
        let spent = ws.spent();
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

    /// Where a trajectory's bare state f at a point first lands (`Dynamics::started`; # The nodes risen and fallen): ground
    /// set k's cluster with nothing explicit where its fold exists there within the share (its landing's reading within
    /// it, proto.rs # The landing's reading), else risen until one is (`Clusters::rise`) -- the landing and its held forms
    /// on the forward that found it within, and the representation it starts with. None where no node is left to rise.
    fn first(&self, k: usize, at: &Point, f: &[C64]) -> Option<(usize, Vec<C64>, Forms, Caches)> {
        let share = crate::model::elim::share();
        let holds = |fold: &Fold, cache: &Cache, ws: &mut Workspace| ws.readings().landing <= share && fold.prototype_left(cache, ws);
        let (cluster, (state, forms), mut caches) = self.clusters.rise(k, at, f, holds, |cluster, bare, cache, ws| {
            let light = !self.clusters.sets.get(cluster).fold.blocks.is_empty();
            self.held_forms(cluster, at, bare, light.then_some((cache, ws)), true)
        })?;
        caches.choose(k, cluster);
        Some((cluster, state, forms, caches))
    }
}

impl Clusters {
    /// A request's clusters on a base choice of nodes (# The state sets): per ground set its cluster with the choice's
    /// explicit nodes of that set first (cluster k the k-th set's), any other made where it is first asked for
    /// (`cluster_with`).
    fn new(request: &Request, choice: &Choice) -> Result<Clusters, String> {
        let grounds = Fold::sets(request, choice)?;
        let mut membership = [0; GROUND];
        for (k, set) in grounds.iter().enumerate() {
            set.iter().for_each(|&g| membership[g] = k);
        }
        let out = Clusters { request: request.clone(), sets: Arena { chunks: std::array::from_fn(|_| OnceLock::new()), keys: Mutex::new(vec![]) }, base: choice.clone(), grounds, membership };
        for k in 0..out.grounds.len() {
            let mut nodes: Vec<Key> = choice.explicit.iter().copied().filter(|x| out.grounds[k].contains(&x.2)).collect();
            nodes.sort();
            out.sets.index((k, nodes.clone()), || Set::new(request, choice, k, &out.grounds[k], nodes))?;
        }
        Ok(out)
    }

    /// The cluster of ground set k with the explicit nodes `nodes` (# The state sets): made where it is not yet, on the
    /// base choice with those nodes of that set kept in P (`Choice::base`); refused where a fold of it cannot be made (a
    /// level explicit at two offsets, a block the prototype does not hold), and kept refused.
    pub(super) fn cluster_with(&self, k: usize, mut nodes: Vec<Key>) -> Result<usize, String> {
        nodes.sort();
        self.sets.index((k, nodes.clone()), || {
            let mut explicit: Vec<Key> = self.base.explicit.iter().copied().filter(|x| !self.grounds[k].contains(&x.2)).collect();
            explicit.extend(nodes.iter().copied());
            let choice = Choice::base(&self.request, explicit)?;
            Set::new(&self.request, &choice, k, &self.grounds[k], nodes)
        })
    }

    /// The cluster a way leads to (`Rise`, `Fall`) from cluster `from`, with the explicit nodes `nodes`: asked for once and
    /// kept with the way, None where refused (its price infinite: # The nodes risen and fallen).
    fn way(&self, from: &Set, cell: &OnceLock<Option<usize>>, nodes: impl FnOnce() -> Vec<Key>) -> Option<usize> {
        *cell.get_or_init(|| self.cluster_with(from.ground_set, nodes()).ok())
    }

    /// The rise of a cluster for a state (# The nodes risen and fallen): its Q node that holds the most of the state, by
    /// the state's amplitudes on its blocks' rows (`Workspace::amplitudes`), among those whose cluster can be made -- that
    /// cluster.
    fn risen(&self, set: &Set, amplitudes: &[Vec<C64>]) -> Option<usize> {
        let held = |r: &Rise| -> f64 { r.rows.iter().map(|&row| amplitudes[r.block].get(row).map_or(0.0, |z| z.norm_sqr())).sum() };
        let mut order: Vec<(&Rise, f64)> = set.rises.iter().map(|r| (r, held(r))).filter(|x| x.1 > 0.0).collect();
        order.sort_by(|a, b| b.1.total_cmp(&a.1));
        order.into_iter().find_map(|(r, _)| {
            self.way(set, &r.cluster, || {
                let mut nodes = set.nodes.clone();
                nodes.push(r.key);
                nodes
            })
        })
    }

    /// The fall of a cluster for a state's P amplitudes x: its explicit node that holds the least of the state (its
    /// places' population), among those whose cluster can be made -- that cluster.
    fn fallen(&self, set: &Set, x: &[C64]) -> Option<usize> {
        let mut order: Vec<(&Fall, f64)> = set.falls.iter().map(|f| (f, f.places.iter().map(|&p| x[p].norm_sqr()).sum())).collect();
        order.sort_by(|a, b| a.1.total_cmp(&b.1));
        order.into_iter().find_map(|(f, _)| self.way(set, &f.cluster, || set.nodes.iter().copied().filter(|k| *k != f.key).collect()))
    }

    /// A workspace of a cluster, taken from its pool (made where none is free).
    fn take(&self, set: usize) -> Workspace {
        let taken = self.sets.get(set).pool.lock().unwrap_or_else(|e| e.into_inner()).pop();
        taken.unwrap_or_else(|| self.sets.get(set).fold.prototype_workspace().expect("the set's workspace"))
    }

    fn give(&self, set: usize, ws: Workspace) {
        self.sets.get(set).pool.lock().unwrap_or_else(|e| e.into_inner()).push(ws);
    }

    /// Where a bare state f of ground set k at a point is first evaluated (# The nodes risen and fallen): k's cluster with
    /// nothing explicit where its forward there `holds`; else -- its fold not existing there, or not holding -- that
    /// cluster's rise for f in that forward (its last sweep's responses, where it failed: `risen`), until one holds:
    /// `then` on that forward (the cluster, f on its states, the forward's decompositions and workspace), with the
    /// representation the attempts made -- every forward's decompositions and count (made or not: no budget). None where
    /// no node is left to rise.
    fn rise<R>(&self, k: usize, at: &Point, f: &[C64], holds: impl Fn(&Fold, &Cache, &mut Workspace) -> bool, then: impl FnOnce(usize, &[C64], &Cache, &mut Workspace) -> R) -> Option<(usize, R, Caches)> {
        let mut caches = Caches::default();
        let mut cluster = k;
        loop {
            let set = self.sets.get(cluster);
            let mut bare = vec![C64::ZERO; set.cluster.states.len()];
            let ground = f.len().min(set.cluster.ground);
            bare[..ground].copy_from_slice(&f[..ground]);
            let mut cache = Cache::default();
            let mut ws = self.take(cluster);
            let held = set.fold.prototype(at.r, at.kappa, &mut cache, &mut ws).is_some() && holds(&set.fold, &cache, &mut ws);
            let count = ws.spent() * crate::decide::COMPLEX;
            if held {
                let out = then(cluster, &bare, &cache, &mut ws);
                self.give(cluster, ws);
                caches.keep(cluster, Arc::new(cache), count);
                return Some((cluster, out, caches));
            }
            let amplitudes = ws.amplitudes(&set.fold, &cache, &bare);
            self.give(cluster, ws);
            caches.keep(cluster, Arc::new(cache), count);
            cluster = self.risen(set, &amplitudes)?;
        }
    }

    /// Σ_b p_b [G(r)]_bb in the lab (the diagonal less its frame) of the ground sets p populates (# The rest of the model):
    /// each set's on the cluster its bare state √p first holds at r at rest -- its fold with nothing explicit where it
    /// exists there, else risen until it does (`rise`: the rule of a trajectory's start, existence its test). None where
    /// no node is left to rise.
    fn diagonal(&self, r: [f64; 3], p: &[f64; GROUND]) -> Option<f64> {
        let at = Point { r, kappa: [0.0; 3], t: 0.0 };
        let mut out = 0.0;
        for k in (0..self.grounds.len()).filter(|&k| self.grounds[k].iter().any(|&s| p[s] != 0.0)) {
            let f: Vec<C64> = self.sets.get(k).fold.ground.iter().map(|&s| C64::real(p[s].sqrt())).collect();
            let (_, part, _) = self.rise(k, &at, &f, |_, _, _| true, |cluster, _, _, ws| {
                let fold = &self.sets.get(cluster).fold;
                let (n, g) = (fold.size(), ws.generator());
                fold.ground.iter().enumerate().map(|(x, &s)| p[s] * (g.data[x * n + x].re + fold.frame_values[x])).sum::<f64>()
            })?;
            out += part;
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

/// A representation's per-cluster decompositions and chosen clusters (`Caches`), where it holds the run's.
fn caches(held: &Held) -> Option<&Caches> {
    held.0.as_ref().and_then(|h| h.downcast_ref::<Caches>())
}

/// A set's candidate tones (`Candidate`; module documentation, # The tones, # The jumps' coherence) from the exact beats
/// and values of its tones and of its coherence groups' cross decays (`crossing`): each exact beat with its opposite
/// once -- its tones there and its cross decays there --, in the order of their first beat of positive orientation
/// (`Beat::sign`; of a beat whose opposite is none, its own); its beat the one of positive orientation, of its tone's
/// value where it has one; its key that beat's coordinates on the common denominator of them all; `work` the counts of a
/// candidate of so many tones and cross decays.
fn candidates(beats: &[(crate::model::Beat, f64)], crossing: &[(crate::model::Beat, f64)], mut work: impl FnMut(bool, bool, usize) -> [f64; 2]) -> Vec<Candidate> {
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
        let work = work(plus.is_some(), minus.is_some(), cross[0].len() + cross[1].len());
        out.push(Candidate { beat, plus, minus, cross, key, work });
        made.push(oriented);
    }
    out
}

/// The request with only the frequencies `on` lit (module documentation, # The rest of the model): every other
/// frequency's beams' fields and couplings zero, and its outside shift at the focus taken out of H0, which holds it
/// (request.rs: H = H0 + Σ_ω Σ_i (x_ω,i(r) − x_ω,i(0)) O_ω,i).
pub(super) fn lit(request: &Request, light: &Light, on: impl Fn(usize) -> bool) -> Request {
    let mut out = request.clone();
    for (_, f) in light.frequencies.iter().enumerate().filter(|(w, _)| !on(*w)) {
        let focus: [C64; 3] = std::array::from_fn(|q| f.beams.iter().map(|&l| request.beams[l].field[q]).sum());
        for (x, o) in crate::request::coordinates(&focus).iter().zip(&request.outside_forms[f.form].matrices) {
            out.hamiltonian.add_scaled(C64::real(-x), o);
        }
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
/// its gradient and Hessian (far.rs `pair`), as the old model's far potential takes them.
struct Reference {
    pairs: Vec<(usize, usize, C64)>,
}

impl Reference {
    /// The held forms of the folds of `prepared` (the request with the trap's frequency alone lit) at the focus, on the
    /// reference state's ground populations p: each ground set's with nothing explicit (`Choice::base`), its ground
    /// places weighted by their populations (`Fold::prototype_held`), each term laid on its beam pairs. Refused where a
    /// far block does not exist at the focus.
    fn new(prepared: &Request, light: &Light, p: &[f64; GROUND]) -> Result<Reference, String> {
        let choice = Choice::base(prepared, vec![])?;
        let mut pairs: Vec<(usize, usize, C64)> = vec![];
        for set in Fold::sets(prepared, &choice)?.iter().filter(|set| set.iter().any(|&s| p[s] != 0.0)) {
            let fold = Fold::new(prepared, &choice, set[0])?;
            let weight: Vec<f64> = (0..fold.size()).map(|x| fold.ground.get(x).map_or(0.0, |&s| p[s])).collect();
            let mut ws = fold.prototype_workspace()?;
            for ([a, b], w) in fold.prototype_held(&mut ws, &weight).ok_or("参考势：阱光的远块在焦点处不存在")? {
                // e_a* e_b of one rotation: co-rotating Σ γ_ja* γ_j'b a_j* a_j', counter-rotating its conjugates' (the
                // pair of a_j' and a_j). Two coefficients of opposite rotations meet at one offset only at twice the
                // optical frequency: no static form holds them.
                assert_eq!(a.2, b.2, "a static form of coefficients of one rotation");
                let (fa, fb) = (&light.frequencies[a.0], &light.frequencies[b.0]);
                for (j, &l) in fa.beams.iter().enumerate() {
                    for (jj, &ll) in fb.beams.iter().enumerate() {
                        let (ga, gb) = (fa.coefficients[j][a.1], fb.coefficients[jj][b.1]);
                        let (first, second, value) = if a.2 { (ll, l, ga * gb.conj() * w) } else { (l, ll, ga.conj() * gb * w) };
                        match pairs.iter_mut().find(|x| x.0 == first && x.1 == second) {
                            Some(x) => x.2 += value,
                            None => pairs.push((first, second, value)),
                        }
                    }
                }
            }
        }
        Ok(Reference { pairs })
    }

    /// V_ref's second-order expansion at r: each beam's amplitude to the second order (field.rs `Beam::amplitude_to`),
    /// each pair's product a_l'* a_l (far.rs `pair`), times its weight, the real part.
    fn at(&self, light: &Light, r: [f64; 3]) -> Quadratic {
        let mut out = Quadratic { value: 0.0, gradient: [0.0; 3], hessian: [[0.0; 3]; 3] };
        for &(first, second, weight) in &self.pairs {
            let c = crate::model::far::pair(&light.beams[first].amplitude_to(r, 2), &light.beams[second].amplitude_to(r, 2));
            out.value += (c.value * weight).re;
            for x in 0..3 {
                out.gradient[x] += (c.gradient[x] * weight).re;
                for y in 0..3 {
                    out.hessian[x][y] += (c.hessian[x][y] * weight).re;
                }
            }
        }
        out
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
/// `step::Middle::made`), the last one's -- the trajectory's count of the cluster (`Caches::counts`) --, its reading (rate
/// units: the largest of its readings, each within the share) and the budget a forward is made within
/// (`Fold::prototype_within`; a switch's, # The nodes risen and fallen).
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
    counted: f64,
    reading: f64,
    landing: f64,
    budget: f64,
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
    /// The forward at the path's momentum, where the workspace does not hold it (module documentation, # The representation).
    fn forward(&self, path: &Path) -> RefMut<'_, Inner> {
        let mut inner = self.inner.borrow_mut();
        if inner.current != Some((path.kappa, path.velocity)) {
            let fold = &self.run.clusters.sets.get(self.set).fold;
            let decide = self.decide && !inner.decided;
            let Inner { committed, eval, ws, generator, budget, .. } = &mut *inner;
            *eval = (**committed).clone();
            let ws = ws.as_mut().expect("a workspace");
            let made = fold.prototype_within(self.r, path.kappa, eval, ws, *budget).map(|step| {
                generator.data.copy_from_slice(&step.generator.data);
                (step.readings().most(), step.landing)
            });
            let spent = ws.spent() * crate::decide::COMPLEX;
            if decide {
                *committed = Arc::new(eval.clone());
            }
            inner.decided |= decide;
            inner.exists = made.is_some();
            (inner.made, inner.counted) = (inner.made + spent, spent);
            if let Some((reading, landing)) = made {
                (inner.reading, inner.landing) = (reading, landing);
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
    /// the place --, every class's rate there, the class `draw` picks and its kick alone (its group's tangent turned there,
    /// on its landing set's rows: `Fold::prototype_photons`); its jump's amplitudes L(s)ψ on the landing set's states --
    /// the photons' own u = C ε S⁻¹ψ of its group (`Workspace::jumped`) --, each at its frame's phase at the time t_m + s
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
        fold.prototype_place(self.r, path.kappa, path.velocity, s, ws);
        let t = self.t + s;
        let photons = fold.prototype_photons(eval, ws, psi, &run.clusters.grounds, At::Place, t, Moments::Drawn(draw));
        let class = &photons.classes[photons.drawn?];
        // The class's jump: its group's blocks' at their offsets' phases e^{−iot} (# The jumps' coherence), as its photons
        // summed them, on the landing cluster's states, each at its frame's phase e^{iθ_s t} (# The jumps' phase).
        let landing = self.landing_cluster(class.landing);
        let to = run.clusters.sets.get(landing);
        let mut amplitudes = [Internal::zeros(to.cluster.states.len()); 3];
        for (a, amplitude) in amplitudes.iter_mut().enumerate() {
            for ((z, &s), &theta) in amplitude.iter_mut().zip(&to.cluster.states).zip(&to.fold.frame_values) {
                if s < GROUND {
                    *z = ws.jumped(class.slot, s, a) * C64::cis(theta * t);
                }
            }
        }
        // The photon's frequency: the excited amplitude's (the frame's, its offset's and the set's slow ground energy) less
        // the landing state's energy; of a group, its blocks' offsets at their shares of the rate (`Class::offset`).
        let first = fold.ground[0];
        let omega = run.frame + class.offset + run.energies[first] - fold.frame_values[0] - run.energies[run.clusters.grounds[class.landing][0]];
        Some(Emitted { landing, wave_number: crate::units::wave_number((omega / crate::units::RAD_PER_US_PER_MHZ).abs()), amplitudes, kick: class.kick })
    }

    /// The cluster a jump into ground set k lands in (# The nodes risen and fallen): this middle's where k is its own set,
    /// else the one the trajectory last took there, else k's with nothing explicit.
    fn landing_cluster(&self, k: usize) -> usize {
        if k == self.run.clusters.sets.get(self.set).ground_set {
            return self.set;
        }
        self.caches.as_ref().and_then(|c| c.chosen.get(k).copied().flatten()).unwrap_or(k)
    }

    /// Where a jump's bare state f lands in this middle's set: its slow part S N⁻¹f along its fast one, S and N the
    /// middle's (DESIGN §1.3, §13.2, the landing; the forward the step made at the path, its left response made where a
    /// landing first asks for it, its count this middle's) -- the switch's formula with nothing following
    /// (`Workspace::taken`: a fall at x_X = 0), its norm the jump clock's. The state itself where the set has no light; no
    /// number where the landing does not exist.
    pub(super) fn landing(&self, path: &Path, f: &[C64]) -> Internal {
        if self.run.clusters.sets.get(self.set).fold.blocks.is_empty() {
            return f.iter().copied().collect();
        }
        let mut inner = self.made(path, None, false);
        if !inner.exists {
            return f.iter().map(|_| C64::new(f64::NAN, f64::NAN)).collect();
        }
        let Inner { eval, ws, made, .. } = &mut *inner;
        let ws = ws.as_mut().expect("a workspace");
        let landed = ws.taken(&self.run.clusters.sets.get(self.set).fold, eval, f, &[], false);
        *made += ws.left_work() * crate::decide::COMPLEX;
        match landed {
            Some((psi, _)) => psi.into_iter().collect(),
            None => f.iter().map(|_| C64::new(f64::NAN, f64::NAN)).collect(),
        }
    }

    /// The trajectory's representation with this cluster's decompositions as this middle's decision made them, and the
    /// count of its last forward (# The nodes risen and fallen).
    fn kept(&self) -> Caches {
        let mut caches = self.caches.as_ref().map_or_else(Caches::default, |c| (**c).clone());
        let inner = self.inner.borrow();
        caches.keep(self.set, inner.committed.clone(), inner.counted);
        caches
    }

    /// The representation this middle leaves (`step::Middle::held`): its decompositions and count (`kept`), and this
    /// cluster the one taken in its ground set.
    fn leaves(&self) -> Caches {
        let mut caches = self.kept();
        caches.choose(self.run.clusters.sets.get(self.set).ground_set, self.set);
        caches
    }

    /// A cluster's price by the trajectory's count of it (# The nodes risen and fallen): the count of a forward it last
    /// measured there, else the cluster's at the focus.
    fn priced(&self, cluster: usize) -> Offer {
        let set = self.run.clusters.sets.get(cluster);
        set.offer(self.caches.as_ref().and_then(|c| c.counts.get(cluster).copied().flatten()).unwrap_or(set.made))
    }

    /// The offers of a decision (# The nodes risen and fallen) for the state's P amplitudes x at this middle: the rise of
    /// the Q node that holds the most of the state (`Clusters::risen`), the fall of the explicit node that holds the least
    /// (`Clusters::fallen`); each its cluster.
    fn offers(&self, x: &[C64]) -> Vec<usize> {
        let (run, set) = (self.run, self.run.clusters.sets.get(self.set));
        let inner = self.inner.borrow();
        let amplitudes = inner.ws.as_ref().expect("a workspace").amplitudes(&set.fold, &inner.eval, x);
        run.clusters.risen(set, &amplitudes).into_iter().chain(run.clusters.fallen(set, x)).collect()
    }

    /// The state ψ of this middle taken onto the cluster `to` at its middle, `other` (# The nodes risen and fallen; one
    /// formula, `Workspace::taken`): each of `to`'s P places the state's amplitude there -- a P place's own, or a risen
    /// node's row's (εx) --; each of its Q rows that is this cluster's P place given that place's amplitude; every other
    /// Q row following. With what it drops, δp over ‖ψ‖², and the left response's count where it was made here (real
    /// multiply-adds). None where `other`'s left response does not exist (`Workspace::taken`).
    pub(super) fn take_onto(&self, psi: &[C64], to: usize, other: &Middle) -> Option<(Vec<C64>, f64, f64)> {
        let (run, mine) = (self.run, self.run.clusters.sets.get(self.set));
        let (theirs_set, from) = (run.clusters.sets.get(to), &mine.fold);
        let target = &theirs_set.fold;
        let (inner, mut theirs) = (self.inner.borrow(), other.inner.borrow_mut());
        let ws = inner.ws.as_ref().expect("a workspace");
        let x = ws.normal().apply(psi);
        // The state's amplitude on every model state this middle holds: its P places, its blocks' rows (a level at one
        // offset only: the explicit ones are no row of another's block in `to`, `Set::rises`).
        let amplitudes = ws.amplitudes(from, &inner.eval, &x);
        let mut of: Vec<Option<C64>> = vec![None; STATES];
        for (p, z) in x.iter().enumerate() {
            of[from.place_state(p)] = Some(*z);
        }
        let risen: Vec<(usize, usize)> = mine.rises.iter().filter(|c| theirs_set.nodes.contains(&c.key)).flat_map(|c| c.rows.iter().map(move |&r| (c.block, r))).collect();
        for &(o, r) in &risen {
            of[from.row_state(o, r)] = Some(amplitudes[o][r]);
        }
        let xp: Vec<C64> = (0..target.size()).map(|p| of[target.place_state(p)].unwrap_or(C64::ZERO)).collect();
        let fallen: Vec<(usize, usize, C64)> = theirs_set
            .rises
            .iter()
            .filter(|c| mine.nodes.contains(&c.key))
            .flat_map(|c| c.rows.iter().map(move |&r| (c.block, r)))
            .filter_map(|(o, r)| of[target.row_state(o, r)].map(|z| (o, r, z)))
            .collect();
        let Inner { eval, ws, .. } = &mut *theirs;
        let ws = ws.as_mut().expect("a workspace");
        let taken = ws.taken(target, eval, &xp, &fallen, true);
        let work = ws.left_work() * crate::decide::COMPLEX;
        let (taken, _) = taken?;
        let norm = crate::linalg::norm_sqr(psi);
        let dropped = (norm - crate::linalg::norm_sqr(&taken)) / norm;
        Some((taken, dropped, work))
    }

    /// e^{iν_m (t_m + s)} ρ: tone m's X at the time s from the middle (module documentation, # The tones).
    fn phased(&self, m: usize, s: f64, rho: &[C64]) -> Vec<C64> {
        let phase = C64::cis(self.run.clusters.sets.get(self.set).tones[m] * (self.t + s));
        rho.iter().map(|z| *z * phase).collect()
    }
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
        // The decay between a coherence group's offsets: −(i/2) e^{i(o − o′)t} Σ_q L_o,q†L_o′,q of its jumps there (# The
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
        self.run.clusters.sets.get(self.set).candidates[m].work
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
        self.run.clusters.sets.get(self.set).work
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

    fn held(&self) -> Option<Held> {
        if !self.inner.borrow().decided {
            return None;
        }
        Some(Held(Some(Arc::new(self.leaves()))))
    }

    /// The decision at a step's end (module documentation, # The nodes risen and fallen): the balance paid by this
    /// interval, (share − ρ)γΔt; the offers (`Middle::offers`) priced by the trajectory's counts of them (`priced`) with
    /// their restarts -- the other side's forward at the point and its held forms -- against staying (this middle's
    /// forwards as made, `made`); the cheapest that is cheaper made here within the budget that keeps it so, its count kept
    /// (where it gives no side within the share, at least that budget), and the state taken onto it
    /// (`Middle::take_onto`); a fall taken only where the balance, its δp and its fold's ργΔt are within the share's γΔt.
    /// Nothing where this middle does not decide, holds no light, or its fold does not exist here.
    fn decide(&self, path: &Path, psi: &[C64], interval: f64, owed: f64, price: &dyn Fn(&Offer) -> f64) -> Decided {
        let mut out = Decided { owed, held: None, switched: None, work: 0.0 };
        let (run, set) = (self.run, self.run.clusters.sets.get(self.set));
        if !self.decide || set.fold.blocks.is_empty() || !self.made(path, None, false).exists {
            return out;
        }
        let share = crate::model::elim::share();
        // A state's decay rate on a generator, i⟨ψ|G − G†|ψ⟩/‖ψ‖².
        let rate = |g: &Mat, psi: &[C64]| -2.0 * psi.iter().zip(g.apply(psi)).map(|(a, b)| a.conj() * b).sum::<C64>().im / crate::linalg::norm_sqr(psi);
        let (x, gamma, reading, landing, made) = {
            let inner = self.inner.borrow();
            (inner.ws.as_ref().expect("a workspace").normal().apply(psi), rate(&inner.generator, psi), inner.reading, inner.landing, inner.made)
        };
        out.owed = (owed - (share - reading.min(share)) * gamma * interval).max(0.0);
        let here = price(&set.offer(made));
        let restart = |o: &Offer| o.made + crate::decide::COMPLEX * o.costs[2];
        let offers = self.offers(&x);
        // Where the landing reads beyond the share the fold is not within it here: its rise, whatever its price (no
        // budget); else the cheapest offer cheaper than staying, its forward within the budget that keeps it so (its count
        // in its price and in its restart; real multiply-adds).
        let (to, budget) = if landing > share {
            let Some(&to) = offers.iter().find(|&&c| run.clusters.sets.get(c).nodes.len() > set.nodes.len()) else { return out };
            (to, None)
        } else {
            let best = offers.into_iter().map(|c| (c, self.priced(c))).map(|(c, o)| (c, o, price(&o) + restart(&o))).filter(|x| x.2 < here).min_by(|a, b| a.2.total_cmp(&b.2));
            let Some((to, offer, _)) = best else { return out };
            (to, Some((here - price(&Offer { made: 0.0, ..offer }) - crate::decide::COMPLEX * offer.costs[2]) / 2.0))
        };
        let other = run.middle(to, self.r, self.t, &Held(Some(Arc::new(self.leaves()))), true);
        other.inner.borrow_mut().budget = budget.map_or(f64::INFINITY, |b| b / crate::decide::COMPLEX);
        let (exists, within) = {
            let theirs = other.made(path, None, false);
            (theirs.exists, theirs.exists && theirs.reading <= share)
        };
        {
            // The attempt's count (# The nodes risen and fallen): where it gives no side within the share, at least the
            // budget it was made within.
            let mut theirs = other.inner.borrow_mut();
            theirs.budget = f64::INFINITY;
            if let Some(b) = budget.filter(|_| !within) {
                theirs.counted = theirs.counted.max(b);
            }
        }
        out.work += other.inner.borrow().made;
        // The decompositions it made are the trajectory's, whatever it takes (a decision's evaluation, DESIGN §11.3).
        out.held = Some(Held(Some(Arc::new(other.kept()))));
        if !exists {
            return out;
        }
        let Some((taken, dropped, left)) = self.take_onto(psi, to, &other) else { return out };
        out.work += left;
        if set.nodes.len() > run.clusters.sets.get(to).nodes.len() {
            let theirs = other.inner.borrow();
            let (rho, gamma) = (theirs.reading, rate(&theirs.generator, &taken));
            if theirs.landing > share || out.owed + dropped + rho * gamma * interval > share * gamma * interval {
                return out;
            }
        }
        // Its held forms at the middle: the static G's and each tone's at the middle's phase, one contraction.
        let tones = &run.clusters.sets.get(to).tones;
        let n = taken.len();
        let mut rho = vec![C64::ZERO; n * n];
        crate::step::density(&taken, &mut rho);
        let phased: Vec<Vec<C64>> = (0..tones.len()).map(|m| other.phased(m, 0.0, &rho)).collect();
        let mut bars = vec![Bar { channel: 0, position: 0, tone: None, matrix: &rho }];
        bars.extend(phased.iter().enumerate().map(|(m, x)| Bar { channel: 1 + m, position: 0, tone: Some(m), matrix: x }));
        let f = other.forms(path, None, &bars, 1 + tones.len());
        out.work += crate::decide::COMPLEX * run.clusters.sets.get(to).work[2];
        let forms = Forms { time: 0.0, held: real(f[0]), tones: tones.iter().zip(&f[1..]).map(|(&nu, a)| (nu, *a)).collect() };
        out.held = Some(Held(Some(Arc::new(other.leaves()))));
        out.switched = Some(Switched { cluster: to, psi: taken.into_iter().collect(), dropped, forms });
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
                counted: 0.0,
                reading: 0.0,
                landing: 0.0,
                budget: f64::INFINITY,
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

    /// The held forms at a point, deciding nothing (# The held forms): on one evaluation there.
    fn forms(&self, cluster: usize, at: &Point, psi: &[C64], held: &Held) -> Forms {
        if self.clusters.sets.get(cluster).fold.blocks.is_empty() {
            return self.held_forms(cluster, at, psi, None, false).1;
        }
        self.evaluate(cluster, at.r, at.kappa, held, |_, cache, ws, _| self.held_forms(cluster, at, psi, Some((cache, ws)), false).1).unwrap_or_else(|| Forms { time: at.t, held: [f64::NAN; 3], tones: vec![] })
    }

    /// The bare state's landing and its held forms in one evaluation, the trajectory's first decision: its cluster
    /// (`Onefold::first`, `Clusters::rise`), and its decompositions made there its representation (# The representation and the decision
    /// times).
    fn started(&self, cluster: usize, at: &Point, f: &[C64]) -> (usize, Internal, Forms, Held) {
        match self.first(self.clusters.sets.get(cluster).ground_set, at, f) {
            Some((cluster, landed, forms, caches)) => (cluster, landed.into_iter().collect(), forms, Held(Some(Arc::new(caches)))),
            None => (cluster, f.iter().map(|_| C64::new(f64::NAN, f64::NAN)).collect(), Forms { time: at.t, held: [f64::NAN; 3], tones: vec![] }, Held::default()),
        }
    }

    /// Where a jump's bare state lands (`Middle::landing`): on the landing set's middle in the step, or at a point on a
    /// middle made there.
    fn landed<'m>(&'m self, cluster: usize, at: &Point, on: Option<(&Middle<'m>, &Path)>, f: &[C64], held: &Held) -> Internal {
        match on {
            Some((middle, path)) => middle.landing(path, f),
            None => {
                let middle = self.middle(cluster, at.r, at.t, held, false);
                let path = middle.path(at.kappa, [0.0; 3]);
                middle.landing(&path, f)
            }
        }
    }

    /// (ħ/m)κ²/2 + V_run(r; p(ψ)) (module documentation, # The rest of the model): each light set's on the clusters its
    /// folds hold at the point, risen where they do not exist (`Clusters::diagonal`); NaN only where no node is left to
    /// rise, or the request with the trap's light off makes no clusters.
    fn energy(&self, cluster: usize, at: &Point, psi: &[C64]) -> f64 {
        let kinetic = 0.5 * self.hbar_over_mass * at.kappa.iter().map(|k| k * k).sum::<f64>();
        let p = self.populations(cluster, psi);
        let untrapped = self.untrapped.get_or_init(|| {
            let request = lit(&self.clusters.request, &self.light, |w| w != self.light.trap);
            Clusters::new(&request, &Choice::base(&request, vec![])?)
        });
        let potential = self.clusters.diagonal(at.r, &p).zip(untrapped.as_ref().ok().and_then(|u| u.diagonal(at.r, &p))).map_or(f64::NAN, |(on, off)| on - off);
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

    fn photon_work(&self) -> f64 {
        self.photons
    }
}
