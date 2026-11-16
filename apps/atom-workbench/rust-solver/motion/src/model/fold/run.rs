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
//! trajectory's representation and decides nothing that outlives it. Here a decision is made at every step's middle;
//! the grid of the coarsest level's step ends is step 3's.
//!
//! # The tones
//! What stays in P and beats (an entry of two P states of different frames, a kept copy's emission) is no part of the
//! prototype: a request whose fold at the focus has a tone is refused here (`Onefold::new`), and the integrator gets no
//! candidate tones.
//!
//! # The rest of the model
//! The reference potential of the splitting (A), the loss energy and its test are the request's prepared and running
//! potentials as model.rs makes them (the folds with every drive bare: a reference is exact for any potential); the
//! displayed rows are the set's own populations. The photons' and the landing's own forms (DESIGN step 3) and the
//! explicit sets' optical sector are not yet here.
use super::proto::{Cache, Workspace};
use super::{Choice, Fold};
use crate::field::Light;
use crate::linalg::{Mat, C64};
use crate::model::{Cluster, Lights, Model, Point, Quadratic, Scattering};
use crate::request::{Request, GROUND, STATES};
use crate::step::{Emitted, Held, Internal};
use std::cell::{Cell, RefCell, RefMut};
use std::sync::{Arc, Mutex};

/// One state set of the run: its fold, its cluster, the near and far blocks in the jumps' order as the fold's blocks, the
/// counted work of a step's middle at the focus (`step::Middle::work`), and the workspaces its evaluations take in turn.
struct Set {
    fold: Fold,
    cluster: Cluster,
    blocks: Vec<usize>,
    work: [f64; 3],
    pool: Mutex<Vec<Workspace>>,
}

/// An atom's representation (`step::Held`): per set its kept decompositions.
#[derive(Clone)]
struct Caches(Vec<Cache>);

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
        let choice = Choice::running(request)?;
        let grounds = Fold::sets(request, &choice)?;
        let light = Light::new(request);
        let potentials = Model::new(request, &light, &[])?;
        let axes = super::Axes::new(request)?;
        let mut membership = [0; GROUND];
        let mut sets = vec![];
        for (k, set) in grounds.iter().enumerate() {
            set.iter().for_each(|&g| membership[g] = k);
            let fold = Fold::new(request, &choice, set[0])?;
            if let Some(f) = fold.at([0.0; 3], [0.0; 3], &mut super::Path::default()) {
                if !f.tones.is_empty() {
                    let tones: Vec<String> = f.tones.iter().map(|t| format!("{:.2} rad/µs（最大元 {:.2e}）", t.value, t.generator.max_abs())).collect();
                    return Err(format!("态集合 {set:?} 在焦点处有音调 {tones:?}：原型不含音调"));
                }
            }
            let mut states: Vec<usize> = fold.ground.clone();
            states.extend(fold.explicit.iter().map(|&x| GROUND + x));
            let cluster = Cluster { ground: fold.ground.len(), frames: fold.frames.clone(), waves: vec![[0.0; 3]; states.len()], waves_exact: vec![potentials.no_wave(); states.len()], states };
            let (pool, blocks, work) = if fold.blocks.is_empty() {
                (vec![], vec![], [0.0; 3])
            } else {
                let mut ws = fold.prototype_workspace()?;
                let (near, far) = ws.layout();
                let blocks = near.iter().map(|x| x.0).chain(far.iter().map(|x| x.0)).collect();
                // The counts of a middle at the focus at rest: its forward and nodes, its tangent and force.
                let mut cache = Cache::default();
                let forward: f64 = fold.prototype([0.0; 3], [0.0; 3], &mut cache, &mut ws).ok_or("焦点处的折叠不存在")?.work.iter().sum();
                let nodes = fold.prototype_nodes([0.0; 3], [0.0; 3], [0.0; 3], [-0.5, 0.5], &cache, &mut ws);
                let tangent = fold.prototype_tangent(&cache, &mut ws);
                let n = fold.size();
                let zero = Mat::zeros(n, n);
                let force = fold.prototype_force(&cache, &mut ws, [&zero, &zero, &zero]).work;
                (vec![ws], blocks, [(forward + nodes) / 3.0, (tangent + force) / 3.0, tangent + force])
            };
            sets.push(Set { fold, cluster, blocks, work, pool: Mutex::new(pool) });
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
        // A jump's photons at the focus, the dearest set's: its forward, tangent and photons (`Dynamics::photon_work`).
        run.photons = (0..run.sets.len())
            .filter(|&k| !run.sets[k].fold.blocks.is_empty())
            .map(|k| {
                let n = run.sets[k].fold.size();
                let psi: Vec<C64> = (0..n).map(|x| if x == 0 { C64::ONE } else { C64::ZERO }).collect();
                run.evaluate(k, [0.0; 3], [0.0; 3], &Held::default(), |fold, cache, ws, forward| forward + fold.prototype_tangent(cache, ws) + fold.prototype_photons(cache, ws, &psi, &run.grounds).work).unwrap_or(0.0)
            })
            .fold(0.0, f64::max)
            * crate::decide::COMPLEX;
        Ok(run)
    }

    /// A set's kept decompositions in a representation (none yet: a new atom's).
    fn cache(&self, held: &Held, set: usize) -> Cache {
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
        let mut cache = self.cache(held, set);
        let mut ws = self.take(set);
        let forward = fold.prototype(r, kappa, &mut cache, &mut ws).map(|step| step.work.iter().sum::<f64>());
        let out = forward.map(|w| then(fold, &cache, &mut ws, w));
        self.give(set, ws);
        out
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

/// A set's generator at a point (`step::Middle`; module documentation): the forward made at a path's momentum where the
/// path is first asked for (its first a decision where the middle decides), the outer nodes where a step asks for them,
/// the tangent where a form or the photons are asked.
pub(crate) struct Middle<'a> {
    run: &'a Onefold,
    set: usize,
    r: [f64; 3],
    decide: bool,
    caches: Option<Arc<Caches>>,
    paths: Cell<u64>,
    inner: RefCell<Inner>,
}

/// What a middle holds of its evaluations: the set's decompositions as the trajectory gave them with what its decision
/// made (`committed`), and those its last forward stood on (`eval`); the workspace; the path the workspace holds, the
/// outer nodes' offset it holds them at, whether it holds the tangent; whether the decision was made; and the generator
/// and S⁻¹ at the middle.
struct Inner {
    committed: Cache,
    eval: Cache,
    ws: Option<Workspace>,
    current: u64,
    nodes: Option<f64>,
    tangent: bool,
    decided: bool,
    exists: bool,
    generator: Mat,
}

/// A path through a middle: its momentum and velocity, and its place among the middle's paths.
pub(crate) struct Path {
    id: u64,
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
        if inner.current != path.id {
            let fold = &self.run.sets[self.set].fold;
            let decide = self.decide && !inner.decided;
            let Inner { committed, eval, ws, generator, .. } = &mut *inner;
            *eval = committed.clone();
            let made = fold.prototype(self.r, path.kappa, eval, ws.as_mut().expect("a workspace")).map(|step| generator.data.copy_from_slice(&step.generator.data));
            if decide {
                *committed = eval.clone();
            }
            inner.decided |= decide;
            inner.exists = made.is_some();
            (inner.current, inner.nodes, inner.tangent) = (path.id, None, false);
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

    /// Σ_j Re tr(Ĝ(s_j) ρ_j) per lab axis over the middle (`bars[0]`) and the outer nodes at ±h (`bars[1]`, `bars[2]`,
    /// where `h`), at once (proto.rs `Fold::prototype_force`, the force's negative).
    fn forms(&self, path: &Path, h: Option<f64>, bars: [&[C64]; 3]) -> [f64; 3] {
        let fold = &self.run.sets[self.set].fold;
        let n = fold.size();
        let mut inner = self.made(path, h, true);
        if !inner.exists {
            return [f64::NAN; 3];
        }
        let mats: Vec<Mat> = bars.iter().map(|b| Mat { rows: n, cols: n, data: b.to_vec() }).collect();
        let Inner { eval, ws, .. } = &mut *inner;
        fold.prototype_force(eval, ws.as_mut().unwrap(), [&mats[0], &mats[1], &mats[2]]).force.map(|f| -f)
    }
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
        let id = self.paths.get() + 1;
        self.paths.set(id);
        Path { id, kappa, velocity }
    }

    fn gauge(&self, _: &Path, rates: &mut [f64]) {
        rates.fill(0.0);
    }

    fn generator(&self, path: &Path, s: f64, h: &mut Mat) {
        let inner = self.made(path, (s != 0.0).then_some(s.abs()), false);
        if !inner.exists {
            h.data.fill(C64::new(f64::NAN, f64::NAN));
            return;
        }
        let source = if s == 0.0 { &inner.generator } else { inner.ws.as_ref().unwrap().node_generator(usize::from(s > 0.0)) };
        h.data.copy_from_slice(&source.data);
    }

    fn force(&self, path: &Path, s: f64, rho: &[C64]) -> [f64; 3] {
        let zero = vec![C64::ZERO; rho.len()];
        if s == 0.0 {
            self.forms(path, None, [rho, &zero, &zero])
        } else if s < 0.0 {
            self.forms(path, Some(-s), [&zero, rho, &zero])
        } else {
            self.forms(path, Some(s), [&zero, &zero, rho])
        }
    }

    fn tones(&self, path: &Path, rho: &[C64], each: &mut dyn FnMut(f64, [f64; 3])) {
        each(0.0, self.force(path, 0.0, rho));
    }

    fn nodes(&self, path: &Path, h: f64, rho: [&[C64]; 3], held: Option<(&[C64], &dyn Fn(f64) -> f64)>) -> [[f64; 3]; 3] {
        // The fixed point's one tone (at zero) on the held state at its weight goes with the middle's form.
        let middle: Vec<C64> = match held {
            Some((held, weight)) => rho[1].iter().zip(held).map(|(r, x)| *r + *x * weight(0.0)).collect(),
            None => rho[1].to_vec(),
        };
        [self.forms(path, Some(h), [&middle, rho[0], rho[2]]), [0.0; 3], [0.0; 3]]
    }

    fn scattering(&self, path: &Path, psi: &[C64]) -> (Scattering, f64) {
        let fold = &self.run.sets[self.set].fold;
        let mut inner = self.made(path, None, true);
        if !inner.exists {
            return (Scattering { force: [f64::NAN; 3], diffusion: [[f64::NAN; 3]; 3], size: 0.0, links: 0 }, 0.0);
        }
        let Inner { eval, ws, .. } = &mut *inner;
        let photons = fold.prototype_photons(eval, ws.as_mut().unwrap(), psi, &self.run.grounds);
        // The terms of a diagonal entry: per class its second moment and its kick's W b², at most the entry and twice
        // the kicks' part; the products behind it: S⁻¹ψ, the responses, the emission, a class's Gram over its rows.
        let kicks: [f64; 3] = std::array::from_fn(|a| photons.classes.iter().map(|c| c.2 * c.3[a] * c.3[a]).sum());
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
        let mut caches = self.caches.as_ref().map_or_else(|| Caches(vec![Cache::default(); self.run.sets.len()]), |c| (**c).clone());
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

    fn middle(&self, cluster: usize, r: [f64; 3], _: f64, held: &Held, decide: bool) -> Middle<'_> {
        let committed = self.cache(held, cluster);
        let n = self.sets[cluster].fold.size();
        let ws = (!self.sets[cluster].fold.blocks.is_empty()).then(|| self.take(cluster));
        let caches = held.0.as_ref().and_then(|h| h.clone().downcast::<Caches>().ok());
        Middle {
            run: self,
            set: cluster,
            r,
            decide,
            caches,
            paths: Cell::new(0),
            inner: RefCell::new(Inner { eval: committed.clone(), committed, ws, current: 0, nodes: None, tangent: false, decided: false, exists: true, generator: Mat::zeros(n, n) }),
        }
    }

    fn emit(&self, cluster: usize, at: &Point, psi: &[C64], held: &Held, draw: &mut dyn FnMut(&[f64]) -> Option<usize>) -> Option<Emitted> {
        let set = &self.sets[cluster];
        let (classes, jumps) = self.evaluate(cluster, at.r, at.kappa, held, |fold, cache, ws, _| {
            fold.prototype_tangent(cache, ws);
            let photons = fold.prototype_photons(cache, ws, psi, &self.grounds);
            (photons.classes, fold.prototype_jumps(cache, ws, None))
        })?;
        let rates: Vec<f64> = classes.iter().map(|c| c.2).collect();
        let (block, landing, _, kick) = classes[draw(&rates)?];
        let states = &self.sets[landing].cluster.states;
        let amplitudes: [Internal; 3] = std::array::from_fn(|a| {
            let u = jumps[block][a].product(&Mat { rows: psi.len(), cols: 1, data: psi.to_vec() });
            states.iter().map(|&s| if s < GROUND { u.data[s] } else { C64::ZERO }).collect()
        });
        // The photon's frequency: the block's excited amplitude's (the frame's, its offset's and the set's slow ground
        // energy) less the landing state's energy.
        let (fold, first) = (&set.fold, set.fold.ground[0]);
        let omega = self.frame + fold.blocks[set.blocks[block]].value + self.energies[first] - fold.frame_values[0] - self.energies[self.grounds[landing][0]];
        Some(Emitted { landing, wave_number: crate::units::wave_number((omega / crate::units::RAD_PER_US_PER_MHZ).abs()), amplitudes, kick })
    }

    fn landed(&self, cluster: usize, at: &Point, f: &[C64], held: &Held) -> Internal {
        if self.sets[cluster].fold.blocks.is_empty() {
            return f.iter().copied().collect();
        }
        // The slow part of the bare state: S⁻¹f (DESIGN §1.3, the landing).
        let n = f.len();
        self.evaluate(cluster, at.r, at.kappa, held, |_, _, ws, _| ws.normal().product(&Mat { rows: n, cols: 1, data: f.to_vec() }).data.into_iter().collect())
            .unwrap_or_else(|| f.iter().map(|_| C64::new(f64::NAN, f64::NAN)).collect())
    }

    fn energy(&self, cluster: usize, at: &Point, psi: &[C64]) -> f64 {
        let kinetic = 0.5 * self.hbar_over_mass * at.kappa.iter().map(|k| k * k).sum::<f64>();
        kinetic + self.potentials.expand(at.r, &self.populations(cluster, psi), &self.running, 0, None).quadratic.value
    }

    fn lost(&self, cluster: usize, at: &Point, psi: &[C64]) -> bool {
        self.potentials.light.outside(at.r) && self.energy(cluster, at, psi) > self.potentials.asymptote(&self.populations(cluster, psi), &self.running)
    }

    fn readout(&self, cluster: usize, _: &Point, psi: &[C64]) -> [f64; STATES] {
        let total = crate::linalg::norm_sqr(psi);
        let mut out = [0.0; STATES];
        for (z, &s) in psi.iter().zip(&self.sets[cluster].cluster.states) {
            out[s] = z.norm_sqr() / total;
        }
        out
    }

    fn photon_work(&self) -> f64 {
        self.photons
    }
}
