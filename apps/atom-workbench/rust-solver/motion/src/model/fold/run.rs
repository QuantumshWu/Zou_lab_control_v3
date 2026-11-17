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
//! Which excited nodes are explicit is decided at the run's decision times (a middle made deciding at the decision's
//! point: `Middle::decide`),
//! by the evaluation order's rule (M18) on P and Q: the cheapest representation within the share. A decision's forward
//! chooses its evaluation within the share by its accounts (proto.rs # The decisions), so the choice is the count: a
//! step's price on each side (the integrator's, `step::Offer`), a switch priced as a restart -- the other side's forward
//! at the point and the held forms there. A decision offers two: the rise of the Q node that holds the most of the state
//! (its population there), and the fall of the explicit node that holds the least -- each among the nodes whose cluster
//! can be made: a refused cluster is kept refused (`Arena`), its price infinite, so no decision offers it again. Each
//! side is priced by the count of a forward the trajectory last measured on it (`Caches::counts`; where it measured
//! none, its cluster's at the focus, made or not): the step's own forward on its cluster at every decision, and an
//! attempt's. A decision takes the cheaper offer where it is cheaper than staying by these counts; the attempt is a
//! count like any other and is kept: what it spent. Where the side reads beyond the share or does not exist at the
//! point, that is no price: its count is what it cost, and the next decision reads it again at its own point. The
//! fold's side is within the share where its reading ρ is -- its accounts combined, the ring's with them (proto.rs # The
//! ring: the transient a jump's landing leaves) --; where ρ reads beyond the share (an account no evaluation remedies)
//! the fold is not within it at the point, and the decision is its rise, whatever its price. A fall is taken only where
//! what it leaves is within the share (DESIGN §11.4): with
//! the trajectory's owed balance b (`Trajectory::owed`: what earlier switches booked and the share has not paid since,
//! each interval paying (share − ρ)γΔt off it), b + E + ρ′γΔt ≤ share·γΔt, E the ring of the fall's fast part (proto.rs
//! `Workspace::ring`: its transient's error, zero for a state that follows T), ρ′ the folded side's readings (within the
//! share), γ the state's decay rate, Δt the interval -- so a trajectory that flips books E at each fall and its balance
//! stops the next one: the flips stop by what they book, no hysteresis.
//!
//! A switch is one formula (`Workspace::taken`; proto.rs # The landing): the state lifted on the side it leaves, taken
//! onto the other's slow span along its fast one, ψ′ = S′a′, a′ = N′⁻¹Φ_L′†Ψ at the middle, not normalized. Risen, the
//! node's rows become P amplitudes x_X = T_X x, every Q row following T: exact (N′x, so S′x: the metric held T_X†T_X as
//! the explicit amplitude's square). Fallen, its rows are given their explicit amplitudes, a′ = x + N′⁻¹T_L,X†(x_X −
//! T_X x), and what the fast part leaves -- its ring E (proto.rs `Workspace::ring`) -- is booked (`Trajectory::dropped`);
//! its norm's change, ‖ψ‖² − ‖ψ′‖², is the fast part's net emission, of either sign, the clock's: a norm at the
//! trajectory's threshold or below is that photon at once (step.rs `Midway::decided`). A jump's landing is the same formula with nothing following (a bare state, S N⁻¹f): a fall at
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
use super::proto::{At, Bar, Cache, Cell, Moments, Taken, Workspace, KINDS};
use super::{Choice, Fold};
use crate::field::Light;
use crate::linalg::{Mat, C64};
use crate::model::{Cluster, Point, Quadratic, Scattering};
use crate::request::{Request, GROUND, STATES};
use crate::step::{Decided, Dynamics as _, Emitted, Forms, Held, Internal, Landing, Middle as _, Optical, Switched};
use std::cell::{RefCell, RefMut};
use std::sync::{Arc, Mutex, OnceLock};

/// An explicit excited node: its level, the absorbed component's part and a source's ground state (`Choice::explicit`).
type Key = (usize, crate::model::Component, usize);

/// One cluster of the run (# The state sets): its ground set's index and its explicit nodes, its fold, its cluster, the
/// ways a decision may take from it (`Rise`, `Fall`), and the workspaces its evaluations take in turn. Nothing is counted
/// at the focus: a representation's price is its sizes' formula where it stands (proto/price.rs).
struct Set {
    ground_set: usize,
    nodes: Vec<Key>,
    fold: Fold,
    cluster: Cluster,
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
/// coordinates on the set's common denominator); its counts are its sizes' formula where it stands (`Middle::tone_work`).
struct Candidate {
    beat: f64,
    plus: Option<usize>,
    minus: Option<usize>,
    cross: [Vec<usize>; 2],
    key: Vec<i64>,
}

/// An atom's representation (`step::Held`): per cluster its kept decompositions, each shared -- a decision replaces one
/// cluster's, a middle reads its own without copying the others' --; per ground set the cluster it last took there (where
/// a jump into that set lands). No count is kept: a cluster's price is its sizes' formula (proto/price.rs).
#[derive(Clone, Default)]
struct Caches {
    sets: Vec<Arc<Cache>>,
    chosen: Vec<Option<usize>>,
}

impl Caches {
    /// A cluster's decompositions.
    fn keep(&mut self, cluster: usize, cache: Arc<Cache>) {
        if self.sets.len() <= cluster {
            self.sets.resize(cluster + 1, Arc::default());
        }
        self.sets[cluster] = cache;
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
}

// The run is one object the trajectories' threads share (solve.rs `Hand::record`): Send + Sync, by the compiler.
const _: () = {
    const fn shared<T: Send + Sync>() {}
    shared::<Onefold>();
};

impl Set {
    /// The cluster of ground set `k` (`ground`) with the explicit nodes `nodes` on a choice of nodes that keeps them
    /// (module documentation, # The state sets): its fold and cluster, its tones' candidates, coherence groups and optical
    /// sector -- its structure alone, no evaluation made.
    fn new(request: &Request, choice: &Choice, k: usize, ground: &[usize], nodes: Vec<Key>) -> Result<Set, String> {
        let decay = super::decay(request);
        let fold = Fold::new(request, choice, ground[0])?;
        let mut states: Vec<usize> = fold.ground.clone();
        states.extend(fold.explicit.iter().map(|&x| GROUND + x));
        let cluster = Cluster { ground: fold.ground.len(), frames: fold.frames.clone(), waves: vec![[0.0; 3]; states.len()], waves_exact: vec![crate::model::Beat::whole(vec![]); states.len()], states };
        let (pool, tones, candidates, coherent) = if fold.blocks.is_empty() {
            (vec![], vec![], vec![], vec![])
        } else {
            let ws = fold.prototype_workspace()?;
            let (near, far) = ws.layout();
            let blocks: Vec<usize> = near.iter().map(|x| x.0).chain(far.iter().map(|x| x.0)).collect();
            let beats: Vec<(crate::model::Beat, f64)> = ws.tones().map(|(b, v)| (b.clone(), v)).collect();
            // The pairs of one coherence group's blocks, each with its exact beat o − o′ and its value.
            let pairs: Vec<(usize, usize)> = (0..blocks.len()).flat_map(|i| (0..blocks.len()).map(move |j| (i, j))).filter(|&(i, j)| i != j && fold.groups[blocks[i]] == fold.groups[blocks[j]]).collect();
            let crossing: Vec<(crate::model::Beat, f64)> = pairs.iter().map(|&(i, j)| (fold.blocks[blocks[i]].offset.plus(-1, &fold.blocks[blocks[j]].offset), fold.blocks[blocks[i]].value - fold.blocks[blocks[j]].value)).collect();
            let candidates = candidates(&beats, &crossing);
            // Each pair turned at its candidate's beat, as the candidate's members are (`Middle::generator`).
            let mut coherent: Vec<(usize, usize, f64)> = pairs.iter().map(|&(i, j)| (i, j, 0.0)).collect();
            for c in &candidates {
                for (side, sign) in c.cross.iter().zip([1.0, -1.0]) {
                    side.iter().for_each(|&k| coherent[k].2 = sign * c.beat);
                }
            }
            (vec![ws], beats.iter().map(|b| b.1).collect(), candidates, coherent)
        };
        let optical = fold.explicit.iter().map(|&x| 0.5 * decay[(x, x)].re).reduce(f64::min);
        let waves = fold.components.iter().flat_map(|c| c.kicks.iter().copied()).collect();
        let rises = Choice::candidates(&fold).into_iter().map(|(key, block, rows)| Rise { key, block, rows, cluster: OnceLock::new() }).collect();
        let ng = fold.ground.len();
        let falls = nodes.iter().map(|&key| Fall { key, places: (ng..fold.size()).filter(|&p| fold.excited_level[fold.explicit[p - ng]] == key.0).collect(), cluster: OnceLock::new() }).collect();
        Ok(Set { ground_set: k, nodes, fold, cluster, rises, falls, pool: Mutex::new(pool), tones, candidates, coherent, optical, waves })
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
        };
        Ok(run)
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

    /// A decision at a point on a bare state f (unify2 §4.1: a trajectory's start, a jump into another set), from the
    /// cluster `cluster` on the representation `held`: its decision there (`Fold::prototype` with the cell `width` ahead
    /// for its look, `Onefold::cell`) -- where its walk chose a node's rise, or its fold or its landing does not exist there
    /// (the node its attribution names, `proto::Workspace::attributed`), that node's cluster's, until one chooses none
    /// (each rise holds one node more: it ends) --, and f landed on it (`Workspace::taken`, nothing following) with its held
    /// forms there; the representation with that decision's decompositions and the cluster chosen in its set. None where
    /// no node is left to rise.
    fn landed_at(&self, cluster: usize, at: &Point, f: &[C64], from: &[usize], held: &Held, width: f64) -> Option<(usize, Vec<C64>, Forms, Caches)> {
        let mut caches = caches(held).cloned().unwrap_or_default();
        let energy = self.mechanical(at.r, at.kappa);
        let mut cluster = cluster;
        loop {
            let set = self.clusters.sets.get(cluster);
            let bare: Vec<C64> = set.cluster.states.iter().map(|s| from.iter().position(|x| x == s).map_or(C64::ZERO, |i| f[i])).collect();
            if set.fold.blocks.is_empty() {
                caches.choose(set.ground_set, cluster);
                let (state, forms) = self.held_forms(cluster, at, &bare, None, false);
                return Some((cluster, state, forms, caches));
            }
            let mut cache = (*caches.sets.get(cluster).cloned().unwrap_or_default()).clone();
            let mut ws = self.clusters.take(cluster);
            ws.cell = self.cell(cluster, at.r, at.kappa, width, 1.0);
            let exists = set.fold.prototype(at.r, at.kappa, energy, &mut cache, &mut ws).is_some();
            let mut chosen = ws.cell.chosen;
            if exists && chosen.is_none() {
                let (state, forms) = self.held_forms(cluster, at, &bare, Some((&cache, &mut ws)), true);
                if state.iter().all(|z| z.is_finite()) {
                    self.clusters.give(cluster, ws);
                    caches.keep(cluster, Arc::new(cache));
                    caches.choose(set.ground_set, cluster);
                    return Some((cluster, state, forms, caches));
                }
                chosen = ws.attributed(&cache);
            }
            self.clusters.give(cluster, ws);
            caches.keep(cluster, Arc::new(cache));
            cluster = self.clusters.rise_to(set, chosen?)?;
        }
    }

    /// What a decision on cluster `cluster` at (r, κ) prices and looks over (`proto::Cell`; unify2 §3.3, §4.2): the cell of
    /// `width` (µs) ahead -- the steps its price counts (`steps`, the model's own counts each: step.rs `Midway::decided`),
    /// the reference flow's path from the point for its look (none where the width is not known) --, the landing sets, and
    /// the nodes the cluster may raise (its rises, by their blocks and rows).
    fn cell(&self, cluster: usize, r: [f64; 3], kappa: [f64; 3], width: f64, steps: f64) -> Cell {
        let set = self.clusters.sets.get(cluster);
        let flow = crate::step::Flow::new(&self.reference.at(&self.light, r), r, self.hbar_over_mass);
        Cell {
            steps,
            jumps: 0.0,
            path: (width > 0.0).then_some(super::proto::Path { flow, from: (r, kappa), width }),
            sets: self.clusters.grounds.clone(),
            nodes: set.rises.iter().map(|x| Taken { block: x.block, rows: x.rows.clone() }).collect(),
            chosen: None,
        }
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
            out.sets.index((k, nodes.clone()), || Set::new(request, choice, k, &out.grounds[k], nodes))?;
        }
        Ok(out)
    }

    /// The cluster of ground set k with the explicit nodes `nodes` (# The state sets): made where it is not yet, on the
    /// base choice of that set with those nodes kept in P (`Choice::base_of`: its own set's tiers, no other's); refused
    /// where a fold of it cannot be made (a level explicit at two offsets, a block the prototype does not hold), and kept
    /// refused.
    pub(super) fn cluster_with(&self, k: usize, mut nodes: Vec<Key>) -> Result<usize, String> {
        nodes.sort();
        self.sets.index((k, nodes.clone()), || {
            let mut explicit: Vec<Key> = self.base.explicit.iter().copied().filter(|x| !self.grounds[k].contains(&x.2)).collect();
            explicit.extend(nodes.iter().copied());
            let choice = Choice::base_of(&self.request, explicit, &|set| set == &self.grounds[k][..])?;
            Set::new(&self.request, &choice, k, &self.grounds[k], nodes)
        })
    }

    /// The cluster a way leads to (`Rise`, `Fall`) from cluster `from`, with the explicit nodes `nodes`: asked for once and
    /// kept with the way, None where refused (its price infinite: # The nodes risen and fallen).
    fn way(&self, from: &Set, cell: &OnceLock<Option<usize>>, nodes: impl FnOnce() -> Vec<Key>) -> Option<usize> {
        *cell.get_or_init(|| self.cluster_with(from.ground_set, nodes()).ok())
    }

    /// The cluster a decision's walk raises a node to (`proto::Cell::chosen`, an index of the set's rises): made where it is
    /// first asked for, None where refused.
    fn rise_to(&self, set: &Set, m: usize) -> Option<usize> {
        let rise = set.rises.get(m)?;
        self.way(set, &rise.cluster, || {
            let mut nodes = set.nodes.clone();
            nodes.push(rise.key);
            nodes
        })
    }

    /// The cluster an explicit node falls to (as `rise_to`).
    fn fall_to(&self, set: &Set, fall: &Fall) -> Option<usize> {
        self.way(set, &fall.cluster, || set.nodes.iter().copied().filter(|k| *k != fall.key).collect())
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
    /// the ground sets p populates (# The rest of the model), both on one set of explicit nodes: none where both folds'
    /// decisions at r at rest raise nothing; where either raises a node or its fold does not exist there, that node (its
    /// walk's attribution, `proto::Fold::walk`: the rule of a trajectory's start), risen in both -- in each light set where
    /// its base cluster offers it (a node of a light the other set has not is risen in this alone) --, until neither
    /// raises one. None where no node is left to rise.
    fn difference(&self, other: &Clusters, r: [f64; 3], p: &[f64; GROUND]) -> Option<f64> {
        let mut nodes: Vec<Key> = vec![];
        'rise: loop {
            let mut out = 0.0;
            for (sign, side) in [(1.0, self), (-1.0, other)] {
                for k in (0..side.grounds.len()).filter(|&k| side.grounds[k].iter().any(|&s| p[s] != 0.0)) {
                    // The nodes this light set's set k holds: of its ground states, by its light's components (a node
                    // that a risen cluster of it offers, whether its base does or not: each rise holds one more, so the
                    // rises end).
                    let mine: Vec<Key> = nodes.iter().copied().filter(|key| side.grounds[k].contains(&key.2) && side.sets.get(k).fold.components.iter().any(|c| c.part == key.1)).collect();
                    let cluster = side.cluster_with(k, mine).ok()?;
                    let set = side.sets.get(cluster);
                    let mut cache = Cache::default();
                    let mut ws = side.take(cluster);
                    ws.cell = Cell { nodes: set.rises.iter().map(|x| Taken { block: x.block, rows: x.rows.clone() }).collect(), ..Cell::default() };
                    let exists = set.fold.prototype(r, [0.0; 3], side.floor, &mut cache, &mut ws).is_some();
                    let fold = &set.fold;
                    let (n, g) = (fold.size(), ws.generator());
                    let part: f64 = fold.ground.iter().enumerate().map(|(x, &s)| p[s] * (g.data[x * n + x].re + fold.frame_values[x])).sum();
                    // Where its decision's walk raises a node there, or the fold does not exist there: that node (the rule of a
                    // trajectory's start, `Onefold::landed_at`).
                    let chosen = (!exists || ws.cell.chosen.is_some()).then_some(ws.cell.chosen);
                    side.give(cluster, ws);
                    if let Some(chosen) = chosen {
                        let risen = side.rise_to(set, chosen?)?;
                        let more: Vec<Key> = side.sets.get(risen).nodes.iter().copied().filter(|key| !nodes.contains(key)).collect();
                        nodes.extend(more);
                        continue 'rise;
                    }
                    out += sign * part;
                }
            }
            return Some(out);
        }
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
        let choice = Choice::base(&prepared, vec![])?;
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
    /// The last decision's walk: the rise it chose (an index of the set's rises), whether a column stays beyond its budget.
    rise: Option<usize>,
    beyond: bool,
    /// The last forward's accounts by kind (ρ²; proto.rs `Step::kinds`), its cell's ends (`Step::started`, `Step::ended`)
    /// and whether it is unresolved.
    kinds: [f64; 5],
    started: [f64; 5],
    ended: [f64; 5],
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
            let spent = ws.spent() * crate::decide::COMPLEX;
            let (rise, beyond) = (ws.cell.chosen, decide && made.is_some() && ws.step_beyond());
            if decide {
                *committed = Arc::new(eval.clone());
            }
            inner.decided |= decide;
            inner.exists = made.is_some();
            inner.made += spent;
            (inner.rise, inner.beyond) = (if decide { rise } else { None }, beyond);
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
        let place = fold.prototype_place(self.r, path.kappa, path.velocity, s, ws);
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
        Some(Emitted { landing, wave_number: crate::units::wave_number((omega / crate::units::RAD_PER_US_PER_MHZ).abs()), amplitudes, kick: class.kick, work: crate::decide::COMPLEX * (place + photons.work) })
    }

    /// The cluster a jump into ground set k lands in (# The nodes risen and fallen): this middle's where k is its own set,
    /// else the one the trajectory last took there, else k's with nothing explicit.
    fn landing_cluster(&self, k: usize) -> usize {
        if k == self.run.clusters.sets.get(self.set).ground_set {
            return self.set;
        }
        self.caches.as_ref().and_then(|c| c.chosen.get(k).copied().flatten()).unwrap_or(k)
    }

    /// Where a jump's bare state f lands in this middle's set (`step::Landing`): its slow part S N⁻¹f along its fast one, S
    /// and N the middle's (DESIGN §1.3, §13.2, the landing; the forward the step made at the path, its left response made
    /// where a landing first asks for it, its count this middle's) -- the switch's formula with nothing following
    /// (`Workspace::taken`: a fall at x_X = 0), its norm the jump clock's. The state itself where the set has no light.
    /// Nowhere where the landing does not exist -- the forward, or the left response or N (`Workspace::taken`), the fold's
    /// one exit: a jump into another set is a decision there (`Onefold::landed_at`), one into this set lands on what the
    /// cell's look saw exist (unify2 §4.1, §4.3).
    pub(super) fn landing(&self, path: &Path, f: &[C64]) -> Landing {
        let set = self.run.clusters.sets.get(self.set);
        if set.fold.blocks.is_empty() {
            return Landing::On(f.iter().copied().collect());
        }
        let mut inner = self.made(path, None, false);
        let Inner { eval, ws, made, exists, .. } = &mut *inner;
        let ws = ws.as_mut().expect("a workspace");
        if *exists {
            let landed = ws.taken(&set.fold, eval, f, &[], false);
            *made += ws.left_work() * crate::decide::COMPLEX;
            if let Some((psi, _)) = landed {
                return Landing::On(psi.into_iter().collect());
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
        let mut caches = self.kept();
        caches.choose(self.run.clusters.sets.get(self.set).ground_set, self.set);
        caches
    }

    /// The cell a decision on this middle prices and looks over (`Onefold::cell`), set before its forward.
    fn priced(&self, cell: Cell) {
        if let Some(ws) = self.inner.borrow_mut().ws.as_mut() {
            ws.cell = cell;
        }
    }

    /// The explicit node whose fall the formula prices cheapest over the decision's cell, where it saves (unify2 §3.3, §3.4:
    /// the cell's price on the sizes the fall leaves less this one's, with the switch once; proto.rs
    /// `Workspace::fall_price`): its cluster. None where none saves.
    fn fall(&self) -> Option<usize> {
        let set = self.run.clusters.sets.get(self.set);
        let inner = self.inner.borrow();
        let ws = inner.ws.as_ref()?;
        let mut best: Option<(f64, usize)> = None;
        for f in &set.falls {
            let Some(price) = ws.fall_price(&set.fold, &inner.eval, &f.places) else { continue };
            if price < 0.0 && best.is_none_or(|b| price < b.0) {
                if let Some(to) = self.run.clusters.fall_to(set, f) {
                    best = Some((price, to));
                }
            }
        }
        best.map(|b| b.1)
    }

    /// The model's counts at a node on this middle's representation (`step::Middle::work`; proto/price.rs `Sizes::costs`):
    /// the formula on the sizes its forward stood on, none where it holds no light or does not exist.
    fn costs(&self) -> Option<(super::proto::price::Sizes, super::proto::price::Route)> {
        let inner = self.inner.borrow();
        let ws = inner.ws.as_ref().filter(|_| inner.exists && inner.current.is_some())?;
        let fold = &self.run.clusters.sets.get(self.set).fold;
        Some((ws.sizes(fold, &inner.eval, &self.run.clusters.grounds), ws.route()))
    }

    /// The state ψ of this middle taken onto the cluster `to` at its middle, `other` (# The nodes risen and fallen; one
    /// formula, `Workspace::taken`): each of `to`'s P places the state's amplitude there -- a P place's own, or a risen
    /// node's row's (εx) --; each of its Q rows that is this cluster's P place given that place's amplitude; every other
    /// Q row following. With the ring of what it leaves (`Workspace::ring`: zero for a state that follows T, a rise), and
    /// the left response's count where it was made here (real multiply-adds). None where `other`'s left response does not
    /// exist (`Workspace::taken`).
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
        let (taken, slow) = taken?;
        let ring = ws.ring(target, eval, &xp, &fallen, true, &slow, &taken);
        Some((taken, ring, work))
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

    fn books(&self, path: &Path, psi: &[C64]) -> Option<([f64; 5], bool, f64)> {
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

    /// The decision at a step's end, on a middle made deciding at the end's point (module documentation, # The nodes
    /// risen and fallen; step.rs `Midway::decided`; unify2 §3.4): the balance paid by this interval, (share − ρ)γΔt, and
    /// the cell's books' end; the walk of its forward on its cell (`Onefold::cell`: the formula's prices over the `steps`
    /// steps the cell counts, the look over the cell) -- the remedies its accounts' attribution asks --;
    /// where the walk chose a node's rise, the state taken onto its cluster (`Middle::take_onto`; where this side's fold
    /// does not exist at the point, its amplitudes landed there, `Middle::landing`) and that cluster's decision there,
    /// until one chooses none (each rise holds one node more: it ends); where it chose none and every column is within,
    /// the explicit node whose fall the formula prices cheapest over the cell with its switch (`Middle::fall`) tried --
    /// its cluster's decision at the point --, taken only where that side is within over its cell and asks for no rise,
    /// and the balance with what the fall books (the ring of its fast part, C + |E|: `Workspace::ring`) and its fold's
    /// ρ′γΔt is within the share's γΔt (DESIGN §11.4). What it decides is the trajectory's, switched or not; nothing where
    /// this middle does not decide or holds no light.
    fn decide(&self, path: &Path, psi: &[C64], interval: f64, owed: f64, steps: f64) -> Decided {
        let mut out = Decided { owed, held: None, switched: None, work: 0.0, booked: [0.0; crate::step::MODEL] };
        let (run, set) = (self.run, self.run.clusters.sets.get(self.set));
        if !self.decide || set.fold.blocks.is_empty() {
            return out;
        }
        let share = crate::model::elim::share();
        // A state's decay rate on a generator, i⟨ψ|G − G†|ψ⟩/‖ψ‖².
        let rate = |g: &Mat, psi: &[C64]| -2.0 * psi.iter().zip(g.apply(psi)).map(|(a, b)| a.conj() * b).sum::<C64>().im / crate::linalg::norm_sqr(psi);
        self.priced(run.cell(self.set, self.r, path.kappa, interval, steps));
        let exists = self.made(path, None, false).exists;
        if exists {
            let inner = self.inner.borrow();
            let gamma = rate(&inner.generator, psi);
            out.owed = (owed - (share - inner.reading.min(share)) * gamma * interval).max(0.0);
            // The cell this decision ends (unify2 §2.7): its end measured, the difference from its start's booked linearly.
            let r = |rho2: f64| 2.0 * rho2.sqrt() + rho2;
            for k in 0..5 {
                out.booked[k] = 0.5 * (r(inner.ended[k]) - r(inner.started[k])) * gamma * interval;
            }
            out.booked[5] = 0.5 * (r(inner.ended.iter().sum()) - r(inner.started.iter().sum())) * gamma * interval;
        }
        // The walk's rises, cluster after cluster.
        let mut chain: Vec<Middle<'_>> = vec![];
        let (mut state, mut dropped) = (psi.to_vec(), 0.0);
        loop {
            let current = chain.last().unwrap_or(self);
            let Some(m) = current.inner.borrow().rise else { break };
            let Some(to) = run.clusters.rise_to(run.clusters.sets.get(current.set), m) else { break };
            let other = run.middle(to, self.r, self.t, &Held(Some(Arc::new(current.leaves()))), true);
            other.priced(run.cell(to, self.r, path.kappa, interval, steps));
            let there = other.made(path, None, false).exists;
            out.work += other.inner.borrow().made;
            if !there {
                break;
            }
            let taken = if current.inner.borrow().exists {
                current.take_onto(&state, to, &other)
            } else {
                let (from, onto) = (&run.clusters.sets.get(current.set).cluster.states, &run.clusters.sets.get(to).cluster.states);
                let f: Vec<C64> = onto.iter().map(|s| from.iter().position(|x| x == s).map_or(C64::ZERO, |i| state[i])).collect();
                match other.landing(path, &f) {
                    Landing::On(x) => Some((x.to_vec(), 0.0, 0.0)),
                    _ => None,
                }
            };
            let Some((taken, ring, left)) = taken else { break };
            out.work += left;
            (state, dropped) = (taken, dropped + ring);
            chain.push(other);
        }
        // A fall, where every column is within over the cell and the formula says its cell saves.
        let within = exists && chain.is_empty() && !self.inner.borrow().beyond;
        if let Some(to) = within.then(|| self.fall()).flatten() {
            let other = run.middle(to, self.r, self.t, &Held(Some(Arc::new(self.leaves()))), true);
            other.priced(run.cell(to, self.r, path.kappa, interval, steps));
            let (holds, rho) = {
                let theirs = other.made(path, None, false);
                (theirs.exists && theirs.rise.is_none() && !theirs.beyond, theirs.reading)
            };
            out.work += other.inner.borrow().made;
            if let Some((taken, ring, left)) = holds.then(|| self.take_onto(psi, to, &other)).flatten() {
                out.work += left;
                let gamma = rate(&other.inner.borrow().generator, &taken);
                if out.owed + ring + rho * gamma * interval <= share * gamma * interval {
                    (state, dropped) = (taken, ring);
                    chain.push(other);
                }
            }
        }
        let last = chain.last().unwrap_or(self);
        out.held = Some(Held(Some(Arc::new(last.leaves()))));
        if chain.is_empty() {
            return out;
        }
        // Its held forms at the middle: the static G's and each tone's at the middle's phase, one contraction.
        let to = last.set;
        let tones = &run.clusters.sets.get(to).tones;
        let n = state.len();
        let mut rho = vec![C64::ZERO; n * n];
        crate::step::density(&state, &mut rho);
        let phased: Vec<Vec<C64>> = (0..tones.len()).map(|m| last.phased(m, 0.0, &rho)).collect();
        let mut bars = vec![Bar { channel: 0, position: 0, tone: None, matrix: &rho }];
        bars.extend(phased.iter().enumerate().map(|(m, x)| Bar { channel: 1 + m, position: 0, tone: Some(m), matrix: x }));
        let f = last.forms(path, None, &bars, 1 + tones.len());
        out.work += crate::decide::COMPLEX * last.costs().map_or(0.0, |(sizes, route)| sizes.costs(&route)[2]);
        let forms = Forms { time: 0.0, held: real(f[0]), tones: tones.iter().zip(&f[1..]).map(|(&nu, a)| (nu, *a)).collect() };
        out.switched = Some(Switched { cluster: to, psi: state.into_iter().collect(), dropped, forms });
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
                rise: None,
                beyond: false,
                kinds: [0.0; 5],
                started: [0.0; 5],
                ended: [0.0; 5],
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
    fn started(&self, cluster: usize, at: &Point, f: &[C64], cell: f64) -> (usize, Internal, Forms, Held) {
        let k = self.clusters.sets.get(cluster).ground_set;
        let from = self.clusters.sets.get(cluster).cluster.states.clone();
        match self.landed_at(k, at, f, &from, &Held::default(), cell) {
            Some((cluster, landed, forms, caches)) => (cluster, landed.into_iter().collect(), forms, Held(Some(Arc::new(caches)))),
            None => (cluster, f.iter().map(|_| C64::new(f64::NAN, f64::NAN)).collect(), Forms { time: at.t, held: [f64::NAN; 3], tones: vec![] }, Held::default()),
        }
    }

    /// Where a jump's bare state lands in its own set (`Middle::landing`): on the landing set's middle in the step, or at a
    /// point on a middle made there; nowhere where it does not exist (`step::Landing`).
    fn landed<'m>(&'m self, cluster: usize, at: &Point, on: Option<(&Middle<'m>, &Path)>, f: &[C64], held: &Held) -> Landing {
        match on {
            Some((middle, path)) => middle.landing(path, f),
            None => {
                let middle = self.middle(cluster, at.r, at.t, held, false);
                let path = middle.path(at.kappa, [0.0; 3]);
                middle.landing(&path, f)
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
            Clusters::new(&request, &Choice::base(&request, vec![])?, self.clusters.floor)
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
    fn entered(&self, from: usize, cluster: usize, at: &Point, f: &[C64], held: &Held, until: f64) -> Option<Option<(usize, Internal, Internal, Held)>> {
        let k = self.clusters.sets.get(cluster).ground_set;
        if k == self.clusters.sets.get(from).ground_set {
            return None;
        }
        let states = self.clusters.sets.get(cluster).cluster.states.clone();
        let start = caches(held).and_then(|c| c.chosen.get(k).copied().flatten()).unwrap_or(k);
        Some(self.landed_at(start, at, f, &states, held, until - at.t).map(|(c, landed, _, caches)| {
            let to = &self.clusters.sets.get(c).cluster.states;
            let bare: Internal = to.iter().map(|s| states.iter().position(|x| x == s).map_or(C64::ZERO, |i| f[i])).collect();
            (c, bare, landed.into_iter().collect(), Held(Some(Arc::new(caches))))
        }))
    }
}
