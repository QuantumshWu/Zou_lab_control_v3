//! The run on the fold at the atom (DESIGN §5.3 `model/run.rs`, §15.4 step 2; BRIEF_WIRE 2a): the integrator's model
//! (`step::Dynamics`, `step::Middle`) given by the prototype (proto.rs) -- at a step's middle and its two outer nodes the
//! generator G, its jumps, its photons and its force, all of one solve and its one derivative.
//!
//! # The state sets
//! The run's choice of nodes is the rule's at the focus (`Choice::running`): every ground state of F = 1 and F = 2 in one
//! vector, the sets the connected parts of what cannot be folded (DESIGN §1.1, the forest), joined by the explicit levels
//! it keeps in P (`Fold::sets`). A set is a cluster of the integrator (`Cluster`): its ground states, then its explicit
//! excited states, each in its frame. No Doppler frame is taken (each state's K_s = 0): the generator at the outer nodes
//! is the light at the path's own points there (`Fold::prototype_nodes`), its plane waves turned, and the integrator
//! takes its change through the three nodes.
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
//! (`Middle::beats`) are the tones' pairs, ν and −ν, each keyed by its beat's exact coordinates; their envelopes carry the
//! middle's phase e^{±iν t_m}. Their forms are the same contraction as the static G's (proto.rs `Fold::prototype_force`):
//! at the nodes with the static ones at their exact phases (`Middle::nodes`, `Middle::force`), on the integrator's sums
//! at their beats (`Middle::tone_force`), on the held state at the beats' weights (`Middle::nodes`, `Middle::tones`).
//!
//! # The jumps on the step's middle (DESIGN §13.2)
//! A jump inside a step is made on the middles the step made: the photon at the jump's time s from the middle on the
//! emitting set's (`Middle::emitted`: the light turned to the jump's place, the responses per beam and their tangent the
//! middle's turned with it, S the middle's -- proto.rs `Fold::prototype_place`, no fold at the place), the landing
//! S⁻¹f on the landing set's (`Middle::landing`, S the middle's). At a point (a trajectory's start) a middle is made
//! there and the jump made on it at s = 0: one way for both.
//!
//! # The jumps' coherence (BRIEF_WIRE3 item 3)
//! A set's blocks whose explicit states P holds a tone between are one coherence group (fold.rs `Fold::groups`, # Tones):
//! their jumps are one operator per landing set, L_q(t) = C_q Σ_o e^{−iot}ε_o S⁻¹ over the group (DESIGN §1.3), so the
//! photons' classes are per group (proto.rs `Fold::prototype_photons` at the time) and a jump's amplitudes its class's
//! photons' own, its group's blocks summed at their phases (`Middle::emitted`, proto.rs `Workspace::jumped`); and the
//! decay between the offsets is its cross products, added to the
//! generator at their beats at every position (`step::Middle::generator`): i(G − G†) = Σ_q L_q(t)†L_q(t) at every time,
//! static and beating, by construction. Every block of the scenes is a group of its own.
//!
//! # The jumps' phase
//! A jump lands on the lab frame's ground states, C_q Σ e^{−iot} ε_o S⁻¹x: in the landing set's coordinates, each of its
//! ground states s at its own frame θ_s, the landed amplitude of s is e^{i(θ_s − o)t} (C_q ε_o S⁻¹x)_s -- one block a
//! class (fold.rs # Tones), so e^{−iot} is the state's global phase and e^{iθ_s t} what a set of several frames holds
//! between its states (`Dynamics::emit`). The rates, the force and the diffusion are sums over the states of |u_s|² and
//! of their gradients' products, each its own phase's: none of them turns.
//!
//! # The rest of the model
//! The reference potential of the splitting (A), the loss energy and its test are the request's prepared and running
//! potentials as model.rs makes them (the folds with every drive bare: a reference is exact for any potential); the
//! displayed rows are the fold's time-averaged populations at the point (`Dynamics::readout`). The photons' and the
//! landing's own forms (DESIGN step 3) and the explicit sets' optical sector are not yet here.
use super::proto::{At, Bar, Cache, Moments, Workspace};
use super::{Choice, Fold};
use crate::field::Light;
use crate::linalg::{Mat, C64};
use crate::model::{Cluster, Lights, Model, Point, Quadratic, Scattering};
use crate::request::{Request, GROUND, STATES};
use crate::step::{Emitted, Forms, Held, Internal, Middle as _};
use std::cell::{RefCell, RefMut};
use std::sync::{Arc, Mutex};

/// One state set of the run: its fold, its cluster, the counted work of a step's middle at the focus
/// (`step::Middle::work`), and the workspaces its evaluations take in turn.
struct Set {
    fold: Fold,
    cluster: Cluster,
    work: [f64; 3],
    pool: Mutex<Vec<Workspace>>,
    /// The tones' values ν_m (rad/µs; proto.rs `Workspace::tones`' order); the integrator's candidates (`Candidate`).
    tones: Vec<f64>,
    candidates: Vec<Candidate>,
    /// The pairs of the jumps' blocks (their order: the near blocks', then the far ones', as the fold's blocks) of one
    /// coherence group (fold.rs `Fold::groups`), each with its beat o − o′ (rad/µs): the decay between their offsets (# The
    /// jumps' coherence).
    coherent: Vec<(usize, usize, f64)>,
}

/// A candidate tone of the integrator (step.rs `Middle::beats`): a pair of the set's tones at ν and −ν (either may be
/// none), the beat's exact key (its coordinates on the set's common denominator), and the counted complex multiply-adds
/// of its envelopes (the three nodes', the middle's phase) and of its forms (`Middle::tone_work`).
struct Candidate {
    beat: f64,
    plus: Option<usize>,
    minus: Option<usize>,
    key: Vec<i64>,
    work: [f64; 2],
}

/// An atom's representation (`step::Held`): per set its kept decompositions, each shared -- a decision replaces one set's,
/// a middle reads its own set's without copying the others'.
#[derive(Clone)]
struct Caches(Vec<Arc<Cache>>);

/// The run's model on the fold at the atom (module documentation).
pub(crate) struct Onefold {
    sets: Vec<Set>,
    /// The sets' ground states (the photons' landing partition).
    grounds: Vec<Vec<usize>>,
    membership: [usize; GROUND],
    hbar_over_mass: f64,
    /// The bare ground energies, and the excited states' frame frequency (rad/µs): the photons' frequencies.
    energies: Vec<f64>,
    frame: f64,
    /// The potentials of the splitting's reference and of the loss (module documentation, # The rest of the model).
    potentials: Model,
    reference: [f64; GROUND],
    prepared: Lights,
    running: Lights,
    photons: f64,
}

impl Onefold {
    /// The run of a request (module documentation), the splitting's reference state's ground populations `reference`.
    /// Refused where the fold at the focus has a tone, or a set is not the prototype's (proto.rs `prototype_workspace`).
    pub(crate) fn new(request: &Request, reference: [f64; GROUND]) -> Result<Onefold, String> {
        Onefold::of(request, &Choice::running(request)?, reference)
    }

    /// The run of a request on a choice of nodes (`new`: the rule's at the focus, `Choice::running`).
    pub(crate) fn of(request: &Request, choice: &Choice, reference: [f64; GROUND]) -> Result<Onefold, String> {
        let grounds = Fold::sets(request, choice)?;
        let light = Light::new(request);
        let potentials = Model::new(request, &light, &[])?;
        let axes = super::Axes::new(request)?;
        let mut membership = [0; GROUND];
        let mut sets = vec![];
        for (k, set) in grounds.iter().enumerate() {
            set.iter().for_each(|&g| membership[g] = k);
            let fold = Fold::new(request, choice, set[0])?;
            let mut states: Vec<usize> = fold.ground.clone();
            states.extend(fold.explicit.iter().map(|&x| GROUND + x));
            let cluster = Cluster { ground: fold.ground.len(), frames: fold.frames.clone(), waves: vec![[0.0; 3]; states.len()], waves_exact: vec![potentials.no_wave(); states.len()], states };
            let (pool, blocks, work, tones, candidates) = if fold.blocks.is_empty() {
                (vec![], vec![], [0.0; 3], vec![], vec![])
            } else {
                let mut ws = fold.prototype_workspace()?;
                let (near, far) = ws.layout();
                let blocks = near.iter().map(|x| x.0).chain(far.iter().map(|x| x.0)).collect();
                let n = fold.size();
                let beats: Vec<(crate::model::Beat, f64)> = ws.tones().map(|(b, v)| (b.clone(), v)).collect();
                // The counts of a middle at the focus at rest: its forward and nodes, its tangent and force with every
                // tone's at the three nodes; a tone's forms on the integrator's sums.
                let mut cache = Cache::default();
                let forward: f64 = fold.prototype([0.0; 3], [0.0; 3], &mut cache, &mut ws).ok_or("焦点处的折叠不存在")?.work.iter().sum();
                let nodes = fold.prototype_nodes([0.0; 3], [0.0; 3], [0.0; 3], [-0.5, 0.5], &cache, &mut ws);
                let tangent = fold.prototype_tangent(&cache, &mut ws);
                let zero = vec![C64::ZERO; n * n];
                let bar = |position: usize, tone: Option<usize>| Bar { channel: 0, position, tone, matrix: &zero };
                let mut bars: Vec<Bar> = (0..3).map(|j| bar(j, None)).collect();
                bars.extend((0..beats.len()).flat_map(|m| (0..3).map(move |j| (m, j))).map(|(m, j)| bar(j, Some(m))));
                let force = fold.prototype_force(&cache, &mut ws, &bars, 1).work;
                let candidates = candidates(&beats, |plus, minus| {
                    let one: Vec<Bar> = [plus, minus].iter().filter(|&&x| x).map(|_| bar(0, Some(0))).collect();
                    [(3 * one.len() * n * n) as f64, fold.prototype_force(&cache, &mut ws, &one, 1).work]
                });
                (vec![ws], blocks, [(forward + nodes) / 3.0, (tangent + force) / 3.0, tangent + force], beats.iter().map(|b| b.1).collect(), candidates)
            };
            let coherent = (0..blocks.len())
                .flat_map(|i| (0..blocks.len()).map(move |j| (i, j)))
                .filter(|&(i, j)| i != j && fold.groups[blocks[i]] == fold.groups[blocks[j]])
                .map(|(i, j)| (i, j, fold.blocks[blocks[i]].value - fold.blocks[blocks[j]].value))
                .collect();
            sets.push(Set { fold, cluster, work, pool: Mutex::new(pool), tones, candidates, coherent });
        }
        let energies: Vec<f64> = (0..GROUND).map(|s| request.hamiltonian[(s, s)].re).collect();
        let mut run = Onefold {
            hbar_over_mass: crate::units::hbar_over_mass(request.mass_kg),
            frame: axes.omega(&axes.frame()),
            energies,
            reference,
            prepared: potentials.prepared(),
            running: potentials.running(),
            potentials,
            grounds,
            membership,
            sets,
            photons: 0.0,
        };
        // A jump's photons at the focus, the dearest set's, as a jump at a point makes them (`Dynamics::emit`): its
        // forward, tangent, the place and the photons there -- every class's rate and the dearest class's moments, the
        // class a jump draws (`Fold::prototype_photons`) -- (`Dynamics::photon_work`).
        run.photons = (0..run.sets.len())
            .filter(|&k| !run.sets[k].fold.blocks.is_empty())
            .map(|k| {
                let n = run.sets[k].fold.size();
                let psi: Vec<C64> = (0..n).map(|x| if x == 0 { C64::ONE } else { C64::ZERO }).collect();
                run.evaluate(k, [0.0; 3], [0.0; 3], &Held::default(), |fold, cache, ws, forward| {
                    let made = forward + fold.prototype_tangent(cache, ws) + fold.prototype_place([0.0; 3], [0.0; 3], [0.0; 3], 0.0, ws);
                    let (mut dearest, mut class, mut classes) = (0.0_f64, 0, 1);
                    while class < classes {
                        let mut drawn = |rates: &[f64]| {
                            classes = rates.len();
                            (class < classes).then_some(class)
                        };
                        dearest = dearest.max(fold.prototype_photons(cache, ws, &psi, &run.grounds, At::Place, 0.0, Moments::Drawn(&mut drawn)).work);
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

    /// A set's kept decompositions in a representation (none yet: a new atom's).
    fn cache(&self, held: &Held, set: usize) -> Arc<Cache> {
        held.0.as_ref().and_then(|h| h.downcast_ref::<Caches>()).and_then(|c| c.0.get(set).cloned()).unwrap_or_default()
    }

    /// A workspace of a set, taken from its pool (made where none is free).
    fn take(&self, set: usize) -> Workspace {
        let taken = self.sets[set].pool.lock().unwrap_or_else(|e| e.into_inner()).pop();
        taken.unwrap_or_else(|| self.sets[set].fold.prototype_workspace().expect("the set's workspace"))
    }

    fn give(&self, set: usize, ws: Workspace) {
        self.sets[set].pool.lock().unwrap_or_else(|e| e.into_inner()).push(ws);
    }

    /// One evaluation at a point that decides nothing (module documentation, # The representation): the set's forward on
    /// a copy of the trajectory's decompositions, then `then` on it with the forward's work. None where the fold does not
    /// exist there.
    fn evaluate<R>(&self, set: usize, r: [f64; 3], kappa: [f64; 3], held: &Held, then: impl FnOnce(&Fold, &Cache, &mut Workspace, f64) -> R) -> Option<R> {
        let fold = &self.sets[set].fold;
        let mut cache = (*self.cache(held, set)).clone();
        let mut ws = self.take(set);
        let forward = fold.prototype(r, kappa, &mut cache, &mut ws).map(|step| step.work.iter().sum::<f64>());
        let out = forward.map(|w| then(fold, &cache, &mut ws, w));
        self.give(set, ws);
        out
    }

    /// The held forms of a set's state at a point (module documentation, # The held forms): the set's forward on the
    /// trajectory's decompositions, its tangent, and in one contraction at the point the static G's form and each tone's
    /// complex amplitude at the point's time, less the reference's gradient there (step.rs `Forms`); where `land`, of the
    /// bare state's landing ψ = S⁻¹ψ. With the state they are of and the decompositions the forward stood on. None where
    /// the fold does not exist there.
    fn held_forms(&self, set: usize, at: &Point, psi: &[C64], held: &Held, land: bool) -> Option<(Vec<C64>, Forms, Cache)> {
        let fold = &self.sets[set].fold;
        let (n, tones) = (fold.size(), &self.sets[set].tones);
        let gradient = self.potentials.expansion(at.r, &self.reference, &self.prepared).gradient;
        if fold.blocks.is_empty() {
            return Some((psi.to_vec(), Forms { time: at.t, ..Forms::default() }.less(gradient), Cache::default()));
        }
        let mut cache = (*self.cache(held, set)).clone();
        let mut ws = self.take(set);
        let out = fold.prototype(at.r, at.kappa, &mut cache, &mut ws).is_some().then(|| {
            let state = if land { ws.normal().product(&Mat { rows: n, cols: 1, data: psi.to_vec() }).data } else { psi.to_vec() };
            fold.prototype_tangent(&cache, &mut ws);
            let mut rho = vec![C64::ZERO; n * n];
            crate::step::density(&state, &mut rho);
            let phased: Vec<Vec<C64>> = tones.iter().map(|&nu| rho.iter().map(|z| *z * C64::cis(nu * at.t)).collect()).collect();
            let mut bars = vec![Bar { channel: 0, position: 0, tone: None, matrix: &rho }];
            bars.extend(phased.iter().enumerate().map(|(m, x)| Bar { channel: 1 + m, position: 0, tone: Some(m), matrix: x }));
            let f = fold.prototype_force(&cache, &mut ws, &bars, 1 + tones.len()).forms;
            (state, Forms { time: at.t, held: real(f[0]), tones: tones.iter().zip(&f[1..]).map(|(&nu, a)| (nu, *a)).collect() }.less(gradient))
        });
        self.give(set, ws);
        out.map(|(state, forms)| (state, forms, cache))
    }

    /// p(ψ), the ground populations of a set's state (model.rs `Model::ground_populations`, on the set's states).
    fn populations(&self, set: usize, psi: &[C64]) -> [f64; GROUND] {
        let total = crate::linalg::norm_sqr(psi);
        let mut p = [0.0; GROUND];
        let model = &self.potentials;
        for (z, &s) in psi.iter().zip(&self.sets[set].cluster.states) {
            let weight = z.norm_sqr() / total;
            if s < GROUND {
                p[s] += weight;
            } else if weight != 0.0 {
                let rate = model.decay[(s - GROUND, s - GROUND)].re;
                for (b, landing) in p.iter_mut().enumerate() {
                    *landing += weight * model.emission.iter().map(|c| c[(b, s)].norm_sqr()).sum::<f64>() / rate;
                }
            }
        }
        p
    }
}

/// A set's candidate tones (`Candidate`; module documentation, # The tones) from its tones' exact beats and values: each
/// pair ν, −ν once, its beat the one of positive orientation (`Beat::sign`), its key that beat's coordinates on the
/// tones' common denominator; `work` the counts of a candidate of so many members.
fn candidates(beats: &[(crate::model::Beat, f64)], mut work: impl FnMut(bool, bool) -> [f64; 2]) -> Vec<Candidate> {
    let common = beats.iter().fold(1_i64, |l, (b, _)| {
        let (mut a, mut c) = (l, b.d);
        while c != 0 {
            (a, c) = (c, a % c);
        }
        l / a * b.d
    });
    let mut out = vec![];
    for (m, (b, value)) in beats.iter().enumerate() {
        let partner = beats.iter().position(|x| x.0 == b.negated());
        if b.sign() < 0 && partner.is_some() {
            continue;
        }
        let (plus, minus, beat) = if b.sign() > 0 { (Some(m), partner, *value) } else { (None, Some(m), -*value) };
        let key = (if b.sign() > 0 { b.clone() } else { b.negated() }).n.iter().map(|x| x * (common / b.d)).collect();
        out.push(Candidate { beat, plus, minus, key, work: work(plus.is_some(), minus.is_some()) });
    }
    out
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
/// was made; and the generator and S⁻¹ at the middle.
struct Inner {
    committed: Arc<Cache>,
    eval: Cache,
    ws: Option<Workspace>,
    current: Option<([f64; 3], [f64; 3])>,
    nodes: Option<f64>,
    tangent: bool,
    decided: bool,
    exists: bool,
    generator: Mat,
}

/// A path through a middle: its momentum and velocity.
pub(crate) struct Path {
    kappa: [f64; 3],
    velocity: [f64; 3],
}

impl Drop for Middle<'_> {
    fn drop(&mut self) {
        if let Some(ws) = self.inner.get_mut().ws.take() {
            self.run.give(self.set, ws);
        }
    }
}

impl Middle<'_> {
    /// The forward at the path's momentum, where the workspace does not hold it (module documentation, # The representation).
    fn forward(&self, path: &Path) -> RefMut<'_, Inner> {
        let mut inner = self.inner.borrow_mut();
        if inner.current != Some((path.kappa, path.velocity)) {
            let fold = &self.run.sets[self.set].fold;
            let decide = self.decide && !inner.decided;
            let Inner { committed, eval, ws, generator, .. } = &mut *inner;
            *eval = (**committed).clone();
            let made = fold.prototype(self.r, path.kappa, eval, ws.as_mut().expect("a workspace")).map(|step| generator.data.copy_from_slice(&step.generator.data));
            if decide {
                *committed = Arc::new(eval.clone());
            }
            inner.decided |= decide;
            inner.exists = made.is_some();
            (inner.current, inner.nodes, inner.tangent) = (Some((path.kappa, path.velocity)), None, false);
        }
        inner
    }

    /// The forward, the outer nodes at ±h where asked, and the tangent where asked.
    fn made(&self, path: &Path, h: Option<f64>, tangent: bool) -> RefMut<'_, Inner> {
        let mut inner = self.forward(path);
        let fold = &self.run.sets[self.set].fold;
        if !inner.exists {
            return inner;
        }
        if let Some(h) = h.filter(|&h| inner.nodes != Some(h)) {
            let Inner { eval, ws, .. } = &mut *inner;
            fold.prototype_nodes(self.r, path.kappa, path.velocity, [-h, h], eval, ws.as_mut().unwrap());
            (inner.nodes, inner.tangent) = (Some(h), false);
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
        let fold = &self.run.sets[self.set].fold;
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
        let (run, set) = (self.run, &self.run.sets[self.set]);
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
        let photons = fold.prototype_photons(eval, ws, psi, &run.grounds, At::Place, t, Moments::Drawn(draw));
        let class = &photons.classes[photons.drawn?];
        // The class's jump: its group's blocks' at their offsets' phases e^{−iot} (# The jumps' coherence), as its photons
        // summed them, on the landing set's states, each at its frame's phase e^{iθ_s t} (# The jumps' phase).
        let to = &run.sets[class.landing];
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
        let omega = run.frame + class.offset + run.energies[first] - fold.frame_values[0] - run.energies[run.grounds[class.landing][0]];
        Some(Emitted { landing: class.landing, wave_number: crate::units::wave_number((omega / crate::units::RAD_PER_US_PER_MHZ).abs()), amplitudes, kick: class.kick })
    }

    /// Where a jump's bare state f lands in this middle's set: its slow part S⁻¹f, S the middle's (DESIGN §1.3, §13.2, the
    /// landing; the forward the step made at the path). The state itself where the set has no light.
    fn landing(&self, path: &Path, f: &[C64]) -> Internal {
        if self.run.sets[self.set].fold.blocks.is_empty() {
            return f.iter().copied().collect();
        }
        let inner = self.made(path, None, false);
        if !inner.exists {
            return f.iter().map(|_| C64::new(f64::NAN, f64::NAN)).collect();
        }
        let n = f.len();
        inner.ws.as_ref().expect("a workspace").normal().product(&Mat { rows: n, cols: 1, data: f.to_vec() }).data.into_iter().collect()
    }

    /// e^{iν_m (t_m + s)} ρ: tone m's X at the time s from the middle (module documentation, # The tones).
    fn phased(&self, m: usize, s: f64, rho: &[C64]) -> Vec<C64> {
        let phase = C64::cis(self.run.sets[self.set].tones[m] * (self.t + s));
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
        self.run.sets[self.set].fold.size()
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
        let set = &self.run.sets[self.set];
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
        let phased: Vec<Vec<C64>> = (0..self.run.sets[self.set].tones.len()).map(|m| self.phased(m, s, rho)).collect();
        let mut bars = vec![Bar { channel: 0, position: j, tone: None, matrix: rho }];
        bars.extend(phased.iter().enumerate().map(|(m, x)| Bar { channel: 0, position: j, tone: Some(m), matrix: x }));
        real(self.forms(path, (s != 0.0).then_some(s.abs()), &bars, 1)[0])
    }

    /// The static G's at zero, each tone's at its beat ν_m: the middle's G_m at its phase e^{iν_m t_m} (module
    /// documentation, # The tones), each its own channel of one contraction.
    fn tones(&self, path: &Path, rho: &[C64], each: &mut dyn FnMut(f64, [f64; 3])) {
        let tones = &self.run.sets[self.set].tones;
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
        let tones = &self.run.sets[self.set].tones;
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
        for c in &self.run.sets[self.set].candidates {
            beats.push(c.beat);
            keys.push(c.key.clone());
        }
    }

    /// Candidate m's members at the time s from the middle: e^{iνt_m} G_+(s) and e^{−iνt_m} G_−(s) (none: zero).
    fn envelopes(&self, path: &Path, m: usize, s: f64, plus: &mut Mat, minus: &mut Mat) {
        let inner = self.made(path, (s != 0.0).then_some(s.abs()), false);
        let c = &self.run.sets[self.set].candidates[m];
        for (member, out, sign) in [(c.plus, plus, 1.0), (c.minus, minus, -1.0)] {
            match member.filter(|_| inner.exists) {
                Some(k) => {
                    let phase = C64::cis(sign * c.beat * self.t);
                    for (z, g) in out.data.iter_mut().zip(&inner.ws.as_ref().unwrap().tone(k, position(s)).data) {
                        *z = phase * *g;
                    }
                }
                None => out.data.fill(if inner.exists { C64::ZERO } else { C64::new(f64::NAN, f64::NAN) }),
            }
        }
    }

    /// Candidate m's forms on the integrator's sums σ = re + i im at its beat: its + member's middle G_+ on e^{iνt_m}σ, its
    /// − member's on e^{−iνt_m}σ† (step.rs `Middle::tone_force`).
    fn tone_force(&self, path: &Path, m: usize, re: &[C64], im: &[C64]) -> [f64; 3] {
        let c = &self.run.sets[self.set].candidates[m];
        let n = self.states();
        let members: Vec<(usize, Vec<C64>)> = [(c.plus, 1.0), (c.minus, -1.0)]
            .iter()
            .filter_map(|&(k, sign)| {
                let phase = C64::cis(sign * c.beat * self.t);
                k.map(|k| (k, (0..n * n).map(|i| (re[i] + C64::new(0.0, sign) * im[i]) * phase).collect()))
            })
            .collect();
        let bars: Vec<Bar> = members.iter().map(|(k, x)| Bar { channel: 0, position: 0, tone: Some(*k), matrix: x }).collect();
        real(self.forms(path, None, &bars, 1)[0])
    }

    fn tone_work(&self, m: usize) -> [f64; 2] {
        self.run.sets[self.set].candidates[m].work
    }

    fn scattering(&self, path: &Path, psi: &[C64]) -> (Scattering, f64) {
        let fold = &self.run.sets[self.set].fold;
        let mut inner = self.made(path, None, true);
        if !inner.exists {
            return (Scattering { force: [f64::NAN; 3], diffusion: [[f64::NAN; 3]; 3], size: 0.0, links: 0 }, 0.0);
        }
        let Inner { eval, ws, .. } = &mut *inner;
        let photons = fold.prototype_photons(eval, ws.as_mut().unwrap(), psi, &self.run.grounds, At::Middle, self.t, Moments::Every);
        // The terms of a diagonal entry: per class its second moment and its kick's W b², at most the entry and twice
        // the kicks' part; the products behind it: S⁻¹ψ, the responses, the emission, a class's Gram over its rows.
        let kicks: [f64; 3] = std::array::from_fn(|a| photons.classes.iter().map(|c| c.rate * c.kick[a] * c.kick[a]).sum());
        let size = (0..3).map(|a| photons.diffusion[a][a].abs() + 2.0 * kicks[a]).fold(0.0, f64::max);
        let links = fold.size() + STATES + 3 * GROUND;
        (Scattering { force: photons.force, diffusion: photons.diffusion, size, links }, photons.work * crate::decide::COMPLEX)
    }

    fn work(&self) -> [f64; 3] {
        self.run.sets[self.set].work
    }

    fn made(&self) -> f64 {
        0.0
    }

    fn gradient(&self, _: &Path, _: f64, g: &mut [Vec<C64>; 3], slope: Option<&mut [Vec<C64>; 3]>) {
        // No explicit set's optical sector here (module documentation, # The rest of the model): `optical` is None.
        let n = self.states();
        g.iter_mut().for_each(|m| m.resize(n * n, C64::ZERO));
        if let Some(slope) = slope {
            slope.iter_mut().for_each(|m| m.resize(n * n, C64::ZERO));
        }
    }

    fn held(&self) -> Option<Held> {
        let inner = self.inner.borrow();
        if !inner.decided {
            return None;
        }
        let mut caches = self.caches.as_ref().map_or_else(|| Caches(vec![Arc::default(); self.run.sets.len()]), |c| (**c).clone());
        caches.0[self.set] = inner.committed.clone();
        Some(Held(Some(Arc::new(caches))))
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

    fn reference(&self, r: [f64; 3]) -> Quadratic {
        self.potentials.expansion(r, &self.reference, &self.prepared)
    }

    fn cluster(&self, cluster: usize) -> &Cluster {
        &self.sets[cluster].cluster
    }

    fn cluster_of(&self, ground: usize) -> usize {
        self.membership[ground]
    }

    fn middle(&self, cluster: usize, r: [f64; 3], t: f64, held: &Held, decide: bool) -> Middle<'_> {
        let committed = self.cache(held, cluster);
        let n = self.sets[cluster].fold.size();
        let ws = (!self.sets[cluster].fold.blocks.is_empty()).then(|| self.take(cluster));
        let caches = held.0.as_ref().and_then(|h| h.clone().downcast::<Caches>().ok());
        Middle {
            run: self,
            set: cluster,
            r,
            t,
            decide,
            caches,
            inner: RefCell::new(Inner { eval: Cache::default(), committed, ws, current: None, nodes: None, tangent: false, decided: false, exists: true, generator: Mat::zeros(n, n) }),
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

    /// The held forms at a point, deciding nothing (# The held forms).
    fn forms(&self, cluster: usize, at: &Point, psi: &[C64], held: &Held) -> Forms {
        self.held_forms(cluster, at, psi, held, false).map_or_else(|| Forms { time: at.t, held: [f64::NAN; 3], tones: vec![] }, |x| x.1)
    }

    /// The bare state's landing and its held forms in one evaluation, the trajectory's first decision: its
    /// decompositions made there are its representation (# The representation and the decision times).
    fn started(&self, cluster: usize, at: &Point, f: &[C64]) -> (Internal, Forms, Held) {
        match self.held_forms(cluster, at, f, &Held::default(), true) {
            Some((landed, forms, cache)) => {
                let mut caches = Caches(vec![Arc::default(); self.sets.len()]);
                caches.0[cluster] = Arc::new(cache);
                (landed.into_iter().collect(), forms, Held(Some(Arc::new(caches))))
            }
            None => (f.iter().map(|_| C64::new(f64::NAN, f64::NAN)).collect(), Forms { time: at.t, held: [f64::NAN; 3], tones: vec![] }, Held::default()),
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

    fn energy(&self, cluster: usize, at: &Point, psi: &[C64]) -> f64 {
        let kinetic = 0.5 * self.hbar_over_mass * at.kappa.iter().map(|k| k * k).sum::<f64>();
        kinetic + self.potentials.expand(at.r, &self.populations(cluster, psi), &self.running, 0, None).quadratic.value
    }

    fn lost(&self, cluster: usize, at: &Point, psi: &[C64]) -> bool {
        self.potentials.light.outside(at.r) && self.energy(cluster, at, psi) > self.potentials.asymptote(&self.populations(cluster, psi), &self.running)
    }

    /// The fold's time-averaged populations of the state at the point (DESIGN §1.3; proto.rs `Workspace::populations`):
    /// one forward there on the trajectory's representation, deciding nothing; a set with no light its states' own. Over
    /// their sum, ψ†ψ by construction.
    fn readout(&self, cluster: usize, at: &Point, psi: &[C64], held: &Held) -> [f64; STATES] {
        let mut out = [0.0; STATES];
        let set = &self.sets[cluster];
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
