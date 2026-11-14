//! Clusters and their frames (SPEC §1.4).
//!
//! # Clusters
//! The states the kept couplings join (`Edge`, units.rs `Model::edges`: a kept drive at the atom, or a kept coupling of two
//! states at a beat -- two ground states, or an explicit excited state and another excited one) form a cluster: a trajectory's
//! state space between two jumps.
//! Every ground state is in one, alone where nothing joins it; an excited state is in one where a kept coupling joins it, then
//! explicit. Its basis: its ground states, then its explicit excited states, each in increasing order.
//!
//! # Frames
//! Each state of a cluster takes a frame θ_s, exactly (`Beat`): ψ_s = e^{−iθ_s t} x_s puts an edge a ← b (static where
//! θ_a − θ_b is its `beat`) at the residual beat `beat` − (θ_a − θ_b), and every other term at its own residual. The
//! frames are a gauge: the propagation integrates a residual beat exactly with its entries' own turns, at their combined
//! frequencies, which no frame changes (spectral.rs, # Tones; ELIMINATION v8). What the frames decide is which couplings
//! are the generator's static part -- taken in the step's exact exponential, whatever their size -- and which turn: of
//! a turning coupling the propagation's leading remainder is its hops' second Magnus term, of the coupling's square. So
//! the frames keep the strongest couplings static (`frames`): the kept edges' maximum spanning tree by strength is held
//! static, and every edge off the tree -- by the tree's cycle property the weakest of the cycle it closes -- turns at what
//! its cycle misses, the cycle's whole detuning Σ_{e∈C} ±β_e on that one edge. A chain is static, exactly; so is a tree;
//! a ring that misses by D turns at D on its weakest edge. Each joined set's first state takes the frame zero (a phase of
//! the whole state).
//!
//! # Doppler frames
//! On a straight path r_h + v s each state also takes a Doppler frame K_s, ψ_s = e^{iK_s·v s} x_s: a plane wave e^{iq·r} of an
//! entry a ← b then turns at the residual wave vector Q = q − (K_a − K_b), and the diagonal holds K_s·v. Any K_s is an exact
//! change of variables -- the motion holds none of them. They are the frames along the kept edges' strength tree -- a
//! drive's strongest beam's k_l, a coupling's strongest beam pair's k_l − k_l', exact integers (`Cluster::waves`) --, so
//! that what the tree holds static it holds static along every path too; every other plane wave turns at its residual Q,
//! exact, which the propagation integrates at its combined frequencies (spectral.rs, # Tones). The residuals are the
//! field's own wave vectors less the tree's: exact lattice vectors, so that one plane wave of the field on two entries
//! the tree's frames turn alike is one tone (assemble.rs, # Exact tones). (A least squares' correction of the frames,
//! the residuals' sum made least, made the generator change least along a path for a step's nodes, which took its
//! change as a polynomial; with the beats integrated exactly it served nothing, and its continuous shifts set every
//! entry at a residual of its own: GMD1C's cluster of eight states held 92 tones a step, their products 17 000 classes.)
//! The tree is the maximum one by strength (Kruskal).

use super::{Beat, Cluster, Wave};
use crate::request::GROUND;

/// A kept coupling between two states a ← b: the frame difference θ_a − θ_b at which it is static and its Doppler frames'
/// K_a − K_b (its strongest part's wave vector), exactly; its strength (rad/µs, the Doppler frames' tree's order).
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

/// What frames are taken in (`frames`): a frequency as exactly as its kind holds it -- a value (rad/µs) in the
/// elimination's closure (elim.rs), a `Beat` in a model's clusters -- with the sums the frames are made of.
pub(crate) trait Exact: Clone {
    /// self + s·other.
    fn plus(&self, s: i64, other: &Self) -> Self;
}

impl Exact for f64 {
    fn plus(&self, s: i64, other: &f64) -> f64 {
        self + s as f64 * other
    }
}

/// The frames of n states from the couplings `components` (module documentation, # Frames): each a ← b at the beat β
/// with its strength (rad/µs), static in the frames where θ_a − θ_b = β. The couplings' maximum spanning tree by strength
/// is held static -- Kruskal's: the strongest first, each that joins two sets kept, ties in their order; every coupling
/// off the tree is then the weakest of the cycle it closes (the cut and cycle properties) --, its frames along it,
/// θ_a = θ_b + β, from each joined set's first state at zero, exactly in the frequencies' own arithmetic (`Exact`).
/// Every coupling off the tree turns at what its cycle misses, β − (θ_a − θ_b): the residual a loop cannot be rid of
/// sits on its weakest edge. A coupling of one state is left out. `zero`: the zero frequency.
pub(crate) fn frames<F: Exact>(n: usize, components: &[(usize, usize, F, f64)], zero: &F) -> Vec<F> {
    let mut order: Vec<usize> = (0..components.len()).filter(|&i| components[i].0 != components[i].1).collect();
    order.sort_by(|&x, &y| components[y].3.total_cmp(&components[x].3));
    let mut joined = Union::new(n);
    let tree: Vec<usize> = order.into_iter().filter(|&i| joined.join(components[i].0, components[i].1)).collect();
    let mut theta: Vec<Option<F>> = (0..n).map(|s| (joined.find(s) == s).then(|| zero.clone())).collect();
    let mut changed = true;
    while changed {
        changed = false;
        for &i in &tree {
            let (a, b, beta, _) = &components[i];
            match (&theta[*a], &theta[*b]) {
                (None, Some(tb)) => (theta[*a], changed) = (Some(tb.plus(1, beta)), true),
                (Some(ta), None) => (theta[*b], changed) = (Some(ta.plus(-1, beta)), true),
                _ => {}
            }
        }
    }
    theta.into_iter().map(|t| t.expect("a state of a joined set")).collect()
}

/// The clusters of the kept edges over `states` states, with their exact frames (module documentation, # Frames: the
/// strongest edges static), the Doppler frames along the edges' tree (the model's to take their values), and each ground
/// state's cluster. `zero` and `still` are the zero beat and wave vector.
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
        let n = cluster.states.len();
        let place = |s: usize| cluster.states.iter().position(|&x| x == s);
        let components: Vec<(usize, usize, Beat, f64)> = edges.iter().filter_map(|e| Some((place(e.a)?, place(e.b)?, e.beat.clone(), e.strength))).collect();
        let frames = frames(n, &components, zero);
        // The Doppler frames along the tree from the cluster's first state.
        let mut waves: Vec<Option<Wave>> = vec![None; n];
        waves[0] = Some(still.clone());
        let mut changed = true;
        while changed {
            changed = false;
            for e in &tree {
                let (Some(a), Some(b)) = (place(e.a), place(e.b)) else { continue };
                match (&waves[a], &waves[b]) {
                    (None, Some(kb)) => {
                        waves[a] = Some(kb.plus(1, &e.wave));
                        changed = true;
                    }
                    (Some(ka), None) => {
                        waves[b] = Some(ka.plus(-1, &e.wave));
                        changed = true;
                    }
                    _ => {}
                }
            }
        }
        cluster.frames = frames;
        cluster.waves_exact = waves.into_iter().map(|k| k.expect("a cluster is connected by its tree")).collect();
    }
    (out, membership)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two frequency axes (rad/µs) a beat's value is read on: a Larmor splitting and a probe's detuning.
    const BEATS: [f64; 2] = [26.37, 3.2];
    fn beat(a: i64, b: i64) -> Beat {
        Beat::whole(vec![a, b])
    }
    fn omega(x: &Beat) -> f64 {
        x.value(&BEATS)
    }
    /// A component's residual β − (θ_a − θ_b) in the frames, exactly.
    fn left_by(c: &(usize, usize, Beat), theta: &[Beat]) -> Beat {
        c.2.plus(-1, &theta[c.0]).plus(1, &theta[c.1])
    }

    /// The frames (module documentation, # Frames; ELIMINATION v8): the strongest couplings static, exactly. A chain is
    /// static -- its residuals the zero beat, not values near it --, and the first state and a state of no coupling at
    /// zero. A ring of three that misses by D = (1, −2) holds all of D on its weakest coupling, the other two static; the
    /// values' frames (the elimination's closure) are the same to rounding. Two couplings of one pair at two beats: the
    /// stronger static, the weaker at their difference.
    #[test]
    fn the_frames_leave_a_chain_static_and_a_ring_its_detuning_on_its_weakest_edge() {
        let chain = vec![(1, 0, beat(1, 0), 3.0), (2, 1, beat(1, 1), 2.0), (3, 2, beat(0, -1), 1.0)];
        let theta = frames(5, &chain, &beat(0, 0));
        assert!(theta[0].is_zero() && theta[4].is_zero(), "the first state and a state of no component at zero");
        for c in &chain {
            let c = (c.0, c.1, c.2.clone());
            assert!(left_by(&c, &theta).is_zero(), "{} ← {}: {:?}", c.0, c.1, left_by(&c, &theta));
        }
        let ring = vec![(1, 0, beat(1, 0), 3.0), (2, 1, beat(0, 1), 0.5), (0, 2, beat(0, -3), 2.0)];
        let theta = frames(3, &ring, &beat(0, 0));
        for (k, c) in ring.iter().enumerate() {
            let left = left_by(&(c.0, c.1, c.2.clone()), &theta);
            assert_eq!(left, if k == 1 { beat(1, -2) } else { beat(0, 0) }, "{} ← {}", c.0, c.1);
        }
        let values: Vec<(usize, usize, f64, f64)> = ring.iter().map(|c| (c.0, c.1, omega(&c.2), c.3)).collect();
        let theta_values = frames(3, &values, &0.0);
        for (c, v) in ring.iter().zip(&values) {
            let exact = omega(&left_by(&(c.0, c.1, c.2.clone()), &theta));
            assert!((v.2 - theta_values[v.0] + theta_values[v.1] - exact).abs() <= 1e-13 * omega(&beat(1, -2)).abs(), "{} ← {}", c.0, c.1);
        }
        let pair = [(1, 0, beat(0, 0), 1.0), (1, 0, beat(1, 0), 2.0)];
        let theta = frames(2, &pair, &beat(0, 0));
        assert_eq!(theta[1], beat(1, 0));
        assert_eq!(left_by(&(1, 0, beat(0, 0)), &theta), beat(-1, 0));
    }

    /// The tree is the maximum one by strength (Kruskal), so every coupling off it is the weakest of the cycle it closes
    /// in the tree (the cycle property): on 200 random graphs of 2 to 8 states and up to 16 couplings of random beats and
    /// strengths, the static couplings join every set the couplings join, and each turning coupling is no stronger than
    /// the weakest static one on the widest static path between its two states (the tree's path is one).
    #[test]
    fn every_coupling_off_the_tree_is_the_weakest_of_its_cycle() {
        let mut s = crate::random::Stream::new(5, crate::random::Purpose::Initial, 0, 0);
        for case in 0..200 {
            let n = 2 + (7.0 * s.uniform()) as usize;
            let count = (16.0 * s.uniform()) as usize;
            let mut pick = || ((n as f64) * s.uniform()) as usize % n;
            let mut couplings = vec![];
            for _ in 0..count {
                let (a, b) = (pick(), pick());
                couplings.push((a, b));
            }
            let components: Vec<(usize, usize, Beat, f64)> = couplings.iter().map(|&(a, b)| (a, b, beat((7.0 * s.uniform()) as i64 - 3, (5.0 * s.uniform()) as i64 - 2), s.uniform())).collect();
            let theta = frames(n, &components, &beat(0, 0));
            let still: Vec<bool> = components.iter().map(|c| c.0 != c.1 && left_by(&(c.0, c.1, c.2.clone()), &theta).is_zero()).collect();
            // The static couplings' graph: its path between two states, and the weakest strength on it.
            let tree: Vec<(usize, usize, f64)> = components.iter().zip(&still).filter(|(_, s)| **s).map(|(c, _)| (c.0, c.1, c.3)).collect();
            // The widest path's weakest strength between two states over the static couplings (the tree's path among them).
            let widest = |from: usize, to: usize| -> Option<f64> {
                let mut best: Vec<Option<f64>> = vec![None; n];
                best[from] = Some(f64::INFINITY);
                let mut changed = true;
                while changed {
                    changed = false;
                    for &(a, b, w) in &tree {
                        for (x, y) in [(a, b), (b, a)] {
                            if let Some(v) = best[x] {
                                let through = v.min(w);
                                if best[y].is_none_or(|old| through > old) {
                                    (best[y], changed) = (Some(through), true);
                                }
                            }
                        }
                    }
                }
                best[to]
            };
            for (c, s) in components.iter().zip(&still) {
                if c.0 == c.1 || *s {
                    continue;
                }
                let path = widest(c.0, c.1).unwrap_or_else(|| panic!("case {case}: {} and {} joined by no static coupling", c.0, c.1));
                assert!(c.3 <= path, "case {case}: the turning {} ← {} of strength {} beside a static path at {path}", c.0, c.1, c.3);
            }
        }
    }
}
