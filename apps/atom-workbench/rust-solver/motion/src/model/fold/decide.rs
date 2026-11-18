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
use super::{Choice, Fold};
use crate::field::Light;
use crate::linalg::C64;
use crate::request::Request;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// An explicit excited node (run.rs `Key`, `Choice::explicit`): its level, the absorbed component, a source's ground state.
pub(crate) type Key = (usize, crate::model::Component, usize);

/// A point of the local physics (module documentation): the trap light's relative intensity, every beam's phase (zero for
/// each frequency's first), the momentum κ (rad/µm: the velocity over ħ/m).
#[derive(Clone, Debug, Default)]
pub(crate) struct Local {
    pub(crate) s: f64,
    pub(crate) phases: Vec<f64>,
    pub(crate) kappa: [f64; 3],
}

impl Local {
    /// Its coordinates' bits: the point's identity (`Search::asked`).
    fn bits(&self) -> Vec<u64> {
        std::iter::once(self.s).chain(self.phases.iter().copied()).chain(self.kappa).map(f64::to_bits).collect()
    }

    /// Every beam's factor on its amplitude (fold.rs `Fold::local`): √s for the trap's beams, times its phase.
    fn factors(&self, light: &Light) -> Vec<C64> {
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
    pub(crate) worst: Option<(Local, f64)>,
    pub(crate) beyond: usize,
    pub(crate) refused: Vec<(Key, usize)>,
    pub(crate) unheld: usize,
    pub(crate) unheld_at: Option<Local>,
    pub(crate) nowhere: usize,
    /// Its same-state pairs' metric cross reading (`crossed`): the largest over the points read and where, and the count
    /// of the readings beyond the share (`unfoldable`; each point's every reading counted, its re-reads too).
    pub(crate) crossed: (f64, Option<Local>),
    pub(crate) crossed_beyond: usize,
}

/// The start's decision (module documentation): per ground set its own, the rounds (the points read in each and the
/// explicit nodes after it), the points read, the seconds.
#[derive(Clone, Debug, Default)]
pub(crate) struct Decision {
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
/// node offered; its same-state pairs' metric cross reading, `Search::crossed`), a node (its key), or nothing where its
/// fold does not exist and no node is left to raise.
#[derive(Clone)]
enum Asked {
    Within(f64, bool, f64),
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
/// fold per decision (`Built`), and the rule's structure with nothing explicit (`Choice::of`, the placements' reading).
pub(crate) struct Search<'a> {
    request: &'a Request,
    light: &'a Light,
    grounds: &'a [Vec<usize>],
    /// The potential channel's floor (rad/µs): every reading's unit.
    floor: f64,
    /// The largest momentum an atom held by the trap has (rad/µm).
    most: f64,
    built: Mutex<Vec<((usize, Vec<Key>), Slot)>>,
    of: OnceLock<Result<Choice, String>>,
    /// What each point asked of each set's decision (module documentation, # What is made once), by the point's bits
    /// (`Local::bits`), the set and its nodes.
    asked: Mutex<HashMap<(Vec<u64>, usize, Vec<Key>), Asked>>,
}

impl<'a> Search<'a> {
    /// The search of a request's run (module documentation): its light, its ground sets, the potential channel's floor
    /// (rad/µs) and the largest momentum an atom held by the trap has (rad/µm).
    pub(crate) fn new(request: &'a Request, light: &'a Light, grounds: &'a [Vec<usize>], floor: f64, most: f64) -> Search<'a> {
        Search { request, light, grounds, floor, most, built: Default::default(), of: OnceLock::new(), asked: Default::default() }
    }

    /// The floor and the largest momentum (`new`) given later: the sets' folds are made without them (`merged`, the base
    /// run's), the points' reads with them.
    pub(crate) fn bound(&mut self, floor: f64, most: f64) {
        (self.floor, self.most) = (floor, most);
    }

    /// One choice of every set's explicit nodes `nodes`, made of each set's own (`Built`, made once and at once): the
    /// rule's sets and frames, every set's explicit nodes (one list in its order),
    /// and per set its tiers, its own keys' (a key's source state is its set's) in its choice's order -- each set's fold
    /// on it the one on its own choice (`Fold::build` reads a set's keys alone): what `Choice::base_of` makes of all the
    /// sets at once (run.rs `Onefold::new`).
    pub(crate) fn merged(&self, nodes: &[Vec<Key>]) -> Result<Choice, String> {
        self.build(&nodes.iter().cloned().enumerate().collect::<Vec<_>>());
        let built: Vec<Arc<Built>> = nodes.iter().enumerate().map(|(k, x)| self.built(k, x)).collect::<Result<_, _>>()?;
        let first = &built.first().ok_or("没有基态集合")?.choice;
        let mut explicit: Vec<Key> = nodes.iter().flat_map(|x| x.iter().copied()).collect();
        explicit.sort();
        let mut out = Choice { frames: first.frames.clone(), sets: first.sets.clone(), explicit, copies: vec![], excited: vec![] };
        for (b, ground) in built.iter().zip(self.grounds) {
            out.copies.extend(b.choice.copies.iter().filter(|x| ground.contains(&x.0 .1)).cloned());
            out.excited.extend(b.choice.excited.iter().filter(|x| ground.contains(&x.0 .1)).cloned());
        }
        Ok(out)
    }

    /// Set k's fold with the explicit nodes `nodes` (`Built`: `Choice::base_of`, its own set's tiers; the fold and its
    /// workspace), made once -- the first thread to ask makes it, the others wait for it.
    fn built(&self, k: usize, nodes: &Vec<Key>) -> Result<Arc<Built>, String> {
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
            let choice = Choice::base_of(self.request, nodes.clone(), &|set| set == &self.grounds[k][..])?;
            let fold = Fold::new(self.request, &choice, self.grounds[k][0])?;
            let ws = fold.prototype_workspace()?;
            let candidates = Choice::candidates(&fold).into_iter().map(|(key, block, rows)| (key, Taken { block, rows })).collect();
            Ok(Arc::new(Built { choice, pool: Mutex::new(vec![(fold.clone(), ws)]), fold, candidates }))
        })
        .clone()
    }

    /// Every set's fold on its nodes `wanted` (set, nodes) made at once, a thread each.
    fn build(&self, wanted: &[(usize, Vec<Key>)]) {
        std::thread::scope(|scope| {
            for (k, nodes) in wanted {
                scope.spawn(move || {
                    let _ = self.built(*k, nodes);
                });
            }
        });
    }

    /// Whether set k's nodes `nodes` are placed (`Fold::build`: a node holding a state explicit at another offset is
    /// refused there) -- what can refuse a cluster before its tiers are read; read on the rule's structure, as the tiers'
    /// first fold is (`Choice::tiered_of`).
    fn placed(&self, k: usize, nodes: &Vec<Key>) -> bool {
        let Ok(of) = self.of.get_or_init(|| Choice::of(self.request, crate::model::elim::share())) else { return false };
        let choice = Choice { explicit: nodes.clone(), ..of.clone() };
        Fold::sets(self.request, &choice).is_ok() && Fold::new(self.request, &choice, self.grounds[k][0]).is_ok()
    }

    /// What the point `local` asks of a set's fold `built`, read on the thread's copy `held` (`Asked`; module
    /// documentation, # At each point).
    fn ask(&self, built: &Result<Arc<Built>, String>, held: Option<&mut (Fold, Workspace)>, local: &Local) -> Asked {
        let (Ok(built), Some((fold, ws))) = (built, held) else { return Asked::Nowhere };
        let candidates = &built.candidates;
        if fold.blocks.is_empty() {
            return Asked::Within(0.0, false, 0.0);
        }
        fold.local = local.factors(self.light);
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
            (Some((reading, beyond)), None) => Asked::Within(reading, beyond, crossed(fold, &cache, ws)),
            (None, None) => Asked::Nowhere,
        }
    }

    /// The points `points` read on the decision `nodes` (per set), in parallel: per point and set what it asks.
    fn round(&self, points: &[Local], nodes: &[Vec<Key>]) -> Vec<Vec<Asked>> {
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
                            let a = self.ask(&built[k], held[k].as_mut(), local);
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
    fn feasible(&self, k: usize, nodes: &Vec<Key>) -> bool {
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
    fn settle(&self, points: &[Local], decision: &mut Decision) -> Vec<Vec<(f64, bool, bool, bool)>> {
        let sets = decision.sets.len();
        let mut out: Vec<Vec<(f64, bool, bool, bool)>> = vec![vec![(0.0, false, false, false); sets]; points.len()];
        let mut open: Vec<usize> = (0..points.len()).collect();
        while !open.is_empty() {
            let nodes: Vec<Vec<Key>> = decision.sets.iter().map(|s| s.nodes.clone()).collect();
            let asked = self.round(&open.iter().map(|&i| points[i].clone()).collect::<Vec<_>>(), &nodes);
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
            let plans: Vec<(Vec<Key>, Vec<Key>)> = (0..sets).map(|k| self.join(&keys[k], &decision.sets[k], &|x| self.placed(k, x))).collect();
            self.build(&plans.iter().map(|p| p.0.clone()).enumerate().collect::<Vec<_>>());
            for (k, plan) in plans.into_iter().enumerate() {
                let (now, refused) = if self.feasible(k, &plan.0) { plan } else { self.join(&keys[k], &decision.sets[k], &|x| self.feasible(k, x)) };
                let set = &mut decision.sets[k];
                set.nodes = now;
                set.refused.extend(refused.into_iter().map(|key| (key, 0)));
            }
            let mut again = vec![];
            // Whether set k had a node of the level joined in this round.
            let joined = |k: usize, level: usize, sets: &[SetDecision]| sets[k].nodes.iter().any(|n| n.0 == level && !nodes[k].contains(n));
            for (&i, asks) in open.iter().zip(asked) {
                let mut more = false;
                for (k, ask) in asks.into_iter().enumerate() {
                    match ask {
                        Asked::Within(reading, beyond, cross) => {
                            out[i][k] = (reading, beyond, false, false);
                            let set = &mut decision.sets[k];
                            if cross > set.crossed.0 {
                                set.crossed = (cross, Some(points[i].clone()));
                            }
                            set.crossed_beyond += usize::from(super::unfoldable(cross, crate::model::elim::share()));
                        }
                        Asked::Nowhere => out[i][k] = (f64::INFINITY, false, false, true),
                        Asked::Node(key) if joined(k, key.0, &decision.sets) => more = true,
                        Asked::Node(key) => {
                            out[i][k] = (f64::INFINITY, false, true, false);
                            if let Some(r) = decision.sets[k].refused.iter_mut().find(|r| r.0 == key) {
                                r.1 += 1;
                            }
                        }
                    }
                }
                if more {
                    again.push(i);
                }
            }
            open = again;
        }
        out
    }

    /// The doubling (module documentation, # The search) over the points `point(i)`: from 2^d points, doubled until the
    /// decision is the same over a doubling; the readings of every point read, with the points.
    fn doubled(&self, decision: &mut Decision, point: &dyn Fn(u64) -> Local) -> Vec<(Local, Vec<(f64, bool, bool, bool)>)> {
        let mut read: Vec<(Local, Vec<(f64, bool, bool, bool)>)> = vec![];
        let mut count = 1u64 << decision.dimensions;
        let mut from = 0u64;
        loop {
            let before: Vec<Vec<Key>> = decision.sets.iter().map(|s| s.nodes.clone()).collect();
            let points: Vec<Local> = (from..count).map(point).collect();
            let readings = self.settle(&points, decision);
            decision.rounds.push((points.len(), decision.sets.iter().map(|s| s.nodes.len()).sum()));
            read.extend(points.into_iter().zip(readings));
            let same = decision.sets.iter().zip(&before).all(|(s, b)| s.nodes == *b);
            if same && from > 0 {
                return read;
            }
            (from, count) = (count, 2 * count);
        }
    }

    /// The start's decision (module documentation).
    pub(crate) fn decide(&self) -> Result<Decision, String> {
        let clock = std::time::Instant::now();
        let coordinates = Coordinates::new(self.light, self.most);
        let d = coordinates.dimensions();
        let sobol = Sobol::new(d)?;
        let mut decision = Decision { sets: self.grounds.iter().map(|g| SetDecision { ground: g.clone(), ..SetDecision::default() }).collect(), dimensions: d, ..Decision::default() };
        let global = self.doubled(&mut decision, &|i| coordinates.local(&sobol.point(i)));
        // The global points' spacing; each set's worst point's box of it, searched by the same doubling.
        let spacing = (global.len() as f64).powf(-1.0 / d as f64);
        let worst_of = |read: &[(Local, Vec<(f64, bool, bool, bool)>)], k: usize| read.iter().filter(|(_, r)| r[k].0.is_finite()).max_by(|a, b| a.1[k].0.total_cmp(&b.1[k].0)).map(|(l, _)| l.clone());
        let mut refined: Vec<(Local, Vec<(f64, bool, bool, bool)>)> = vec![];
        for k in 0..decision.sets.len() {
            let Some(at) = worst_of(&global, k) else { continue };
            let centre = self.unit(&coordinates, &at);
            let local = |i: u64| {
                let u: Vec<f64> = sobol.point(i).iter().zip(&centre).enumerate().map(|(c, (&w, &x))| self.within(&coordinates, c, x + spacing * (2.0 * w - 1.0))).collect();
                coordinates.local(&u)
            };
            refined.extend(self.doubled(&mut decision, &local));
        }
        // The readings on the final decision: the points of the last round of each doubling, read on it (a point read on
        // a smaller P reads on a larger one again at the next round's start, and those of the last round are on it).
        // (Where one asks for a node even so, it joins: the union.)
        let finals = self.settle(&global.iter().chain(&refined).map(|x| x.0.clone()).collect::<Vec<_>>(), &mut decision);
        let all: Vec<Local> = global.iter().chain(&refined).map(|x| x.0.clone()).collect();
        for (k, set) in decision.sets.iter_mut().enumerate() {
            set.beyond = finals.iter().filter(|r| r[k].1).count();
            set.unheld = finals.iter().filter(|r| r[k].2).count();
            set.nowhere = finals.iter().filter(|r| r[k].3).count();
            set.worst = all.iter().zip(&finals).filter(|(_, r)| r[k].0.is_finite()).max_by(|a, b| a.1[k].0.total_cmp(&b.1[k].0)).map(|(l, r)| (l.clone(), r[k].0));
            set.unheld_at = all.iter().zip(&finals).find(|(_, r)| r[k].2).map(|(l, _)| l.clone());
        }
        decision.seconds = clock.elapsed().as_secs_f64();
        Ok(decision)
    }

    /// A point's unit-cube coordinates (`Coordinates::local`'s inverse).
    fn unit(&self, coordinates: &Coordinates, at: &Local) -> Vec<f64> {
        let mut u = vec![at.s];
        u.extend(coordinates.phased.iter().map(|&l| at.phases[l] / std::f64::consts::TAU));
        let r = at.kappa.iter().map(|x| x * x).sum::<f64>().sqrt();
        let z = if r > 0.0 { at.kappa[2] / r } else { 0.0 };
        let angle = at.kappa[1].atan2(at.kappa[0]).rem_euclid(std::f64::consts::TAU);
        u.extend([(z + 1.0) / 2.0, angle / std::f64::consts::TAU, if coordinates.most > 0.0 { (r / coordinates.most).powi(3) } else { 0.0 }]);
        u
    }

    /// Coordinate c's value x taken into the cube: a phase or the momentum's azimuth turned, the others held at its edges.
    fn within(&self, coordinates: &Coordinates, c: usize, x: f64) -> f64 {
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
