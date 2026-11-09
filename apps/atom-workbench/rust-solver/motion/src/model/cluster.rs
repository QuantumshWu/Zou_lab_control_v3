//! Clusters and their frames (SPEC §1.4).
//!
//! # Clusters
//! The states the kept couplings join (`Edge`: a kept drive, a kept coupling between two ground states, a kept static coupling
//! of two excited states, a kept channel into an explicit state) form a cluster: a trajectory's state space between two jumps.
//! Every ground state is in one, alone where nothing joins it; an excited state is in one where a kept coupling joins it, then
//! explicit. Its basis: its ground states, then its explicit excited states, each in increasing order.
//!
//! # Frames
//! Each state of a cluster takes a frame θ_s, exactly an integer combination of the light's frequencies (`Beat`): ψ_s = e^{−iθ_s t}
//! x_s puts a term a ← b of beat Ω (∝ e^{iΩt}) at Ω + θ_a − θ_b, so that an edge is static when θ_a − θ_b is its own frame
//! difference. Along a spanning tree of the kept edges every one of them is static, exactly; an edge off the tree, or a term
//! the tree leaves, keeps its residual beat, which the propagator resolves. The tree is the maximum one by strength (Kruskal):
//! the frames are exact for any tree, and the strongest couplings static leave the propagator the least to resolve (a choice
//! of cost, not of accuracy). The same tree gives each state its Doppler frame K_s (model.rs `Cluster::waves`): a drive's
//! strongest beam's k_l, a coupling's strongest beam pair's k_l − k_l'; exact where every pair of states holds one wave vector.

use super::{Beat, Cluster, Wave};
use crate::request::GROUND;

/// A kept coupling between two states a ← b: its frame difference θ_a − θ_b and its Doppler frames' K_a − K_b (its strongest
/// part's wave vector), exactly; its strength (rad/µs, the spanning tree's order).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Edge {
    pub(crate) a: usize,
    pub(crate) b: usize,
    pub(crate) beat: Beat,
    pub(crate) wave: Wave,
    pub(crate) strength: f64,
}

/// A union of sets over 0..n.
struct Union {
    parent: Vec<usize>,
}

impl Union {
    fn new(n: usize) -> Union {
        Union { parent: (0..n).collect() }
    }
    fn find(&mut self, x: usize) -> usize {
        let mut x = x;
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }
    /// Joins the sets of a and b; whether they were apart.
    fn join(&mut self, a: usize, b: usize) -> bool {
        let (a, b) = (self.find(a), self.find(b));
        if a == b {
            return false;
        }
        self.parent[a.max(b)] = a.min(b);
        true
    }
}

/// The clusters of the kept edges over `states` states, with their exact frames (module documentation; the Doppler frames'
/// values are the model's to take), and each ground state's cluster. `zero` and `still` are the zero beat and wave vector.
pub(crate) fn clusters(states: usize, edges: &[Edge], zero: &Beat, still: &Wave) -> (Vec<Cluster>, [usize; GROUND]) {
    let mut joined = Union::new(states);
    let mut order: Vec<usize> = (0..edges.len()).collect();
    order.sort_by(|&x, &y| edges[y].strength.total_cmp(&edges[x].strength));
    let mut tree: Vec<&Edge> = vec![];
    for &x in &order {
        if joined.join(edges[x].a, edges[x].b) {
            tree.push(&edges[x]);
        }
    }
    let in_edge: Vec<bool> = (0..states).map(|s| s < GROUND || edges.iter().any(|e| e.a == s || e.b == s)).collect();
    let mut roots: Vec<usize> = vec![];
    let mut out: Vec<Cluster> = vec![];
    for s in (0..states).filter(|&s| in_edge[s]) {
        let root = joined.find(s);
        let at = match roots.iter().position(|&x| x == root) {
            Some(at) => at,
            None => {
                roots.push(root);
                out.push(Cluster { states: vec![], ground: 0, frames: vec![], waves: vec![], waves_exact: vec![] });
                roots.len() - 1
            }
        };
        out[at].states.push(s);
    }
    let mut membership = [0; GROUND];
    for (c, cluster) in out.iter_mut().enumerate() {
        cluster.ground = cluster.states.iter().filter(|&&s| s < GROUND).count();
        for &s in cluster.states.iter().filter(|&&s| s < GROUND) {
            membership[s] = c;
        }
        // Frames along the tree from the cluster's first state.
        let n = cluster.states.len();
        let place = |s: usize| cluster.states.iter().position(|&x| x == s);
        let mut frames: Vec<Option<(Beat, Wave)>> = vec![None; n];
        frames[0] = Some((zero.clone(), still.clone()));
        let mut changed = true;
        while changed {
            changed = false;
            for e in &tree {
                let (Some(a), Some(b)) = (place(e.a), place(e.b)) else { continue };
                match (&frames[a], &frames[b]) {
                    (None, Some((tb, kb))) => {
                        frames[a] = Some((tb.plus(1, &e.beat), kb.plus(1, &e.wave)));
                        changed = true;
                    }
                    (Some((ta, ka)), None) => {
                        frames[b] = Some((ta.plus(-1, &e.beat), ka.plus(-1, &e.wave)));
                        changed = true;
                    }
                    _ => {}
                }
            }
        }
        let (frames, waves): (Vec<Beat>, Vec<Wave>) = frames.into_iter().map(|f| f.expect("a cluster is connected by its tree")).unzip();
        cluster.frames = frames;
        cluster.waves_exact = waves;
    }
    (out, membership)
}
