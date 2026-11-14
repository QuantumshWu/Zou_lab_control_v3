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
//! θ_a − θ_b is its `beat`) at the residual beat `beat` − (θ_a − θ_b), and every other term at its own residual, which the
//! propagator resolves; the static part, whatever its size, is the step's exact exponential (ELIMINATION v4). The frames are
//! the ones that leave the kept edges turning least (`frames`): the least largest residual t* is the largest mean detuning
//! |Σ_{e∈C} ±β_e|/|C| of a cycle C of the edges (the linear program's dual, Karp's minimum mean cycle), and every edge of
//! that cycle turns at t* -- its detuning shared equally --; the cycle is then one state, and what is left is made least the
//! same way, cycle by cycle (max-balancing, Schneider & Schneider 1991: the next cycle's mean is no larger, and the frames
//! that make each in turn least are unique up to one constant a joined set), until what is left has no cycle off resonance
//! and is static. A chain is static, exactly; a ring of n edges that misses by D turns at D/n on each of them, its frames
//! fractions with the denominator n. The cluster's first state takes the frame zero (a phase of the whole state).
//!
//! # Doppler frames
//! On a straight path r_h + v s each state also takes a Doppler frame K_s, ψ_s = e^{iK_s·v s} x_s: a plane wave e^{iq·r} of an
//! entry a ← b then turns at the residual wave vector Q = q − (K_a − K_b), and the diagonal holds K_s·v. Any K_s is an exact
//! change of variables -- the motion holds none of them. What they decide is how fast the generator changes along the path in
//! the frame, which is all a step's nodes approximate (step.rs, # The kick): with Ĥ = Σ A_Q e^{iQ·r} over its entries' plane
//! waves, dĤ/ds = Σ i(Q·v) A_Q e^{iQ·r}, whose mean square over the fringes and the directions of v is (v²/3) Σ |A_Q|² |Q|².
//! A plan's frames are the ones that make it least with every plane wave an entry holds at a tone counted once,
//!
//! ```text
//! K = argmin Σ_Q |q − (K_a − K_b)|²        L K = b,  L the entries' graph Laplacian,  b_s = Σ_{Q: s ← ·} q − Σ_{Q: · ← s} q
//! ```
//!
//! -- which plane waves an entry holds is the plan's own and exact, their amplitudes are a point's (a near form's with the
//! levels there); counted so, the frames are linear in the wave vectors with rational weights, the same in any coordinates.
//! An entry alone would take the mean of its plane waves; the Laplacian shares what the loops do not close. The minimum is
//! zero exactly where every pair of states holds one wave vector and they add up around every loop: there the generator is
//! the same along every path (`Plan::still`), by the same rule. Where they do not close no frame is: the diagonal entries'
//! plane waves, the light's own pattern, turn in every frame, and the frames take the least of the rest (the largest
//! residual is not what they minimise: its minimum is neither one frame nor a closed form; assemble.rs `Plan::sampled`
//! says what is left).
//!
//! They are made as the frames along a spanning tree of the kept edges -- a drive's strongest beam's k_l, a coupling's
//! strongest beam pair's k_l − k_l', exact integers (`Cluster::waves`) -- plus a correction δ (`correction`): L δ = the sums
//! of the tree's residuals, which are integers too. Where the tree's frames leave nothing the sums are exactly zero and so
//! is δ, whatever the rounding of the solve; any tree gives the same K. The tree is the maximum one by strength (Kruskal):
//! a choice of where the integers start, not of the result.

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

/// The correction δ_s (1/µm) of n states' Doppler frames (module documentation, # Doppler frames): `held` is every plane wave
/// of every entry x ← y, once -- (x, y, its residual wave vector in the frames to correct, exactly) --, `value` an exact
/// wave vector's value. The least squares of the residuals Q − (δ_x − δ_y): L δ = b, b_s = Σ_{s ← ·} Q − Σ_{· ← s} Q summed
/// exactly, each set of states the entries join with its first state's δ = 0 (the frames of a set are free by one common
/// K, a phase of the whole state; two sets no entry joins share no term). L is positive definite on the rest: eliminated
/// without pivoting. A b of exact zeros gives exact zeros.
pub(crate) fn correction<'a>(n: usize, held: impl Iterator<Item = (usize, usize, &'a Wave)>, still: &Wave, value: impl Fn(&Wave) -> [f64; 3]) -> Vec<[f64; 3]> {
    let (mut laplacian, mut sums, mut joined) = (vec![0.0; n * n], vec![still.clone(); n], Union::new(n));
    for (x, y, residual) in held.filter(|h| h.0 != h.1) {
        for (a, b) in [(x, y), (y, x)] {
            laplacian[a * n + a] += 1.0;
            laplacian[a * n + b] -= 1.0;
        }
        (sums[x], sums[y]) = (sums[x].plus(1, residual), sums[y].plus(-1, residual));
        joined.join(x, y);
    }
    let mut delta: Vec<[f64; 3]> = sums.iter().map(&value).collect();
    for s in (0..n).filter(|&s| joined.find(s) == s) {
        for other in 0..n {
            (laplacian[s * n + other], laplacian[other * n + s]) = (0.0, 0.0);
        }
        (laplacian[s * n + s], delta[s]) = (1.0, [0.0; 3]);
    }
    for k in 0..n {
        for row in k + 1..n {
            let factor = laplacian[row * n + k] / laplacian[k * n + k];
            if factor != 0.0 {
                for column in k..n {
                    laplacian[row * n + column] -= factor * laplacian[k * n + column];
                }
                for i in 0..3 {
                    delta[row][i] -= factor * delta[k][i];
                }
            }
        }
    }
    for k in (0..n).rev() {
        for column in k + 1..n {
            for i in 0..3 {
                delta[k][i] -= laplacian[k * n + column] * delta[column][i];
            }
        }
        for i in 0..3 {
            delta[k][i] /= laplacian[k * n + k];
        }
    }
    delta
}

/// What frames are taken in (`frames`): a frequency as exactly as its kind holds it -- a value (rad/µs) in the
/// elimination's closure (elim.rs), a `Beat` in a model's clusters -- with the sums and the one division, by a cycle's
/// length, the frames are made of.
pub(crate) trait Exact: Clone {
    /// self + s·other.
    fn plus(&self, s: i64, other: &Self) -> Self;
    /// self/k.
    fn over(&self, k: i64) -> Self;
    fn is_zero(&self) -> bool;
}

impl Exact for f64 {
    fn plus(&self, s: i64, other: &f64) -> f64 {
        self + s as f64 * other
    }
    fn over(&self, k: i64) -> f64 {
        self / k as f64
    }
    fn is_zero(&self) -> bool {
        *self == 0.0
    }
}

/// The frames of n states that leave the couplings `components` turning least (module documentation, # Frames): each
/// component a ← b at β turns at β − (θ_a − θ_b) in the frames θ (static where θ_a − θ_b = β). As a graph: an arc b → a of
/// weight β and an arc a → b of −β per component; frames with θ_v ≤ θ_u + w + t on every arc u → v are exactly those that
/// leave every component within ±t. The least such t is minus the least mean weight of a cycle (every cycle holds both
/// ways round: its least mean is ≤ 0), by Karp's walks (1978): D_k(v) the least weight of a walk of k arcs from any state,
/// λ* = min_v max_k (D_m(v) − D_k(v))/(m − k) over the m states; and every cycle of the least walk of m arcs to the
/// minimizing v has the mean λ* (with the weights less λ*, D_m(v) is v's least walk, and a cycle taken out of it leaves a
/// walk of m − |C| arcs no lighter, so the cycle weighs nothing) -- t* = −Σ_C w/|C| of the first one found back along it,
/// exactly, in the frequencies' own arithmetic (`Exact`). Each arc of that cycle is held at θ_v = θ_u + w + t*, and the
/// cycle is one state from there on (max-balancing, Schneider & Schneider 1991). Where the least cycle's t* is not above
/// zero -- every cycle left on resonance -- the rest is static: the frames of the least walks at t = 0, θ_v − θ_u = w on
/// every arc, exactly where the cycles close exactly. A component of one state turns at its own β in every frame and is
/// left out. `value`: a frequency's value (rad/µs); `zero`: the zero frequency. Returns the frames -- each set of states
/// the components join with its first state at zero -- and t*.
pub(crate) fn frames<F: Exact>(n: usize, components: &[(usize, usize, F)], value: &dyn Fn(&F) -> f64, zero: &F) -> (Vec<F>, F) {
    // Each state's set (a state made one with a cycle: its set's first state) and its frame relative to the set's.
    let (mut set, mut relative): (Vec<usize>, Vec<F>) = ((0..n).collect(), vec![zero.clone(); n]);
    // Each state's node among the sets, and the arcs between the sets: β + θ_b − θ_a of their own frames.
    let contracted = |set: &[usize], relative: &[F]| -> (Vec<usize>, Vec<(usize, usize, F)>) {
        let sets: Vec<usize> = (0..n).filter(|&s| set[s] == s).collect();
        let node: Vec<usize> = (0..n).map(|s| sets.iter().position(|&x| x == set[s]).expect("a state's set")).collect();
        let arcs = components
            .iter()
            .filter(|c| set[c.0] != set[c.1])
            .flat_map(|(a, b, beta)| {
                let w = beta.plus(1, &relative[*b]).plus(-1, &relative[*a]);
                [(node[*b], node[*a], w.clone()), (node[*a], node[*b], zero.plus(-1, &w))]
            })
            .collect();
        (node, arcs)
    };
    let mut residual = zero.clone();
    loop {
        let (node, arcs) = contracted(&set, &relative);
        if arcs.is_empty() {
            break;
        }
        let m = node.iter().copied().max().map_or(0, |x| x + 1);
        let weights: Vec<f64> = arcs.iter().map(|a| value(&a.2)).collect();
        // Karp's walks, each with its last arc.
        let (mut walks, mut last) = (vec![vec![f64::INFINITY; m]; m + 1], vec![vec![usize::MAX; m]; m + 1]);
        walks[0].iter_mut().for_each(|x| *x = 0.0);
        for k in 1..=m {
            for (i, (from, to, _)) in arcs.iter().enumerate() {
                let x = walks[k - 1][*from] + weights[i];
                if x < walks[k][*to] {
                    (walks[k][*to], last[k][*to]) = (x, i);
                }
            }
        }
        let mut least = (f64::INFINITY, usize::MAX);
        for v in (0..m).filter(|&v| walks[m][v].is_finite()) {
            let worst = (0..m).filter(|&k| walks[k][v].is_finite()).map(|k| (walks[m][v] - walks[k][v]) / (m - k) as f64).fold(f64::NEG_INFINITY, f64::max);
            if worst < least.0 {
                least = (worst, v);
            }
        }
        // The least walk of m arcs to v, back to its first repeated state: its cycle's arcs in order.
        let mut at = vec![least.1; m + 1];
        let mut seen = vec![usize::MAX; m];
        seen[at[m]] = m;
        let mut cycle = vec![];
        for k in (1..=m).rev() {
            at[k - 1] = arcs[last[k][at[k]]].0;
            if seen[at[k - 1]] != usize::MAX {
                cycle = (k..=seen[at[k - 1]]).map(|l| last[l][at[l]]).collect();
                break;
            }
            seen[at[k - 1]] = k - 1;
        }
        let t = zero.plus(-1, &cycle.iter().fold(zero.clone(), |sum, &i| sum.plus(1, &arcs[i].2))).over(cycle.len() as i64);
        if t.is_zero() || !(value(&t) > 0.0) {
            break;
        }
        if value(&t) > value(&residual) {
            residual = t.clone();
        }
        // The cycle's frames from its first state, and its sets made one.
        let mut theta: Vec<Option<F>> = vec![None; m];
        theta[arcs[cycle[0]].0] = Some(zero.clone());
        for &i in &cycle {
            let (from, to, w) = &arcs[i];
            let next = theta[*from].as_ref().expect("the cycle's arcs in order").plus(1, w).plus(1, &t);
            theta[*to].get_or_insert(next);
        }
        let first = (0..n).find(|&s| node[s] == arcs[cycle[0]].0).map(|s| set[s]).expect("the cycle's first set");
        for s in 0..n {
            if let Some(th) = &theta[node[s]] {
                (relative[s], set[s]) = (th.plus(1, &relative[s]), first);
            }
        }
    }
    // The rest static: the least walks from any set at t = 0.
    let (node, arcs) = contracted(&set, &relative);
    let m = node.iter().copied().max().map_or(0, |x| x + 1);
    let (mut theta, mut at): (Vec<F>, Vec<f64>) = (vec![zero.clone(); m], vec![0.0; m]);
    for _ in 0..m {
        let mut moved = false;
        for (from, to, w) in &arcs {
            let x = at[*from] + value(w);
            if x < at[*to] {
                (theta[*to], at[*to], moved) = (theta[*from].plus(1, w), x, true);
            }
        }
        if !moved {
            break;
        }
    }
    let mut out: Vec<F> = (0..n).map(|s| theta[node[s]].plus(1, &relative[s])).collect();
    // Each joined set's first state at zero.
    let mut joined = Union::new(n);
    for c in components {
        joined.join(c.0, c.1);
    }
    let anchors: Vec<F> = (0..n).map(|s| out[joined.find(s)].clone()).collect();
    out.iter_mut().zip(&anchors).for_each(|(x, a)| *x = x.plus(-1, a));
    (out, residual)
}

/// The clusters of the kept edges over `states` states, with their exact frames (module documentation, # Frames: the
/// edges turning least, `value` a beat's value), the Doppler frames along the edges' tree (the model's to take their
/// values), and each ground state's cluster. `zero` and `still` are the zero beat and wave vector.
pub(crate) fn clusters(states: usize, edges: &[Edge], zero: &Beat, still: &Wave, value: &dyn Fn(&Beat) -> f64) -> (Vec<Cluster>, [usize; GROUND]) {
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
        let components: Vec<(usize, usize, Beat)> = edges.iter().filter_map(|e| Some((place(e.a)?, place(e.b)?, e.beat.clone()))).collect();
        let frames = frames(n, &components, value, zero).0;
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
    use crate::linalg::{dot3, norm3};
    use crate::units::rounding;

    /// Two wave vectors that are not orthogonal (1/µm): the coordinates of the tests' exact ones.
    const AXES: [[f64; 3]; 2] = [[8.0, 0.0, 0.0], [3.0, 5.0, 1.0]];

    fn wave(a: i64, b: i64) -> Wave {
        Beat::whole(vec![a, b])
    }
    fn value(w: &Wave) -> [f64; 3] {
        std::array::from_fn(|i| w.value(&[AXES[0][i], AXES[1][i]]))
    }
    /// The plane waves x ← y with their adjoints' (y ← x, the wave vectors negated), as a plan holds them.
    fn with_adjoints(held: &[(usize, usize, Wave)]) -> Vec<(usize, usize, Wave)> {
        held.iter().flat_map(|(x, y, q)| [(*x, *y, q.clone()), (*y, *x, q.negated())]).collect()
    }
    fn corrected(n: usize, held: &[(usize, usize, Wave)]) -> Vec<[f64; 3]> {
        correction(n, held.iter().map(|h| (h.0, h.1, &h.2)), &wave(0, 0), value)
    }
    /// A plane wave's residual in the corrected frames, Q − (δ_x − δ_y).
    fn left(h: &(usize, usize, Wave), delta: &[[f64; 3]]) -> [f64; 3] {
        let q = value(&h.2);
        std::array::from_fn(|i| q[i] - (delta[h.0][i] - delta[h.1][i]))
    }
    fn squares(held: &[(usize, usize, Wave)], delta: &[[f64; 3]]) -> f64 {
        held.iter().map(|h| dot3(left(h, delta), left(h, delta))).sum()
    }
    /// The solve's rounding on n states of wave vectors of the size `scale`: the elimination's backward error γ_3n
    /// (Higham §10.1) times the condition of a grounded Laplacian of equal counts, below 2n·n², within γ of n⁴ operations.
    fn within(n: usize, scale: f64) -> f64 {
        rounding(n.pow(4)) * scale
    }

    /// A closing loop -- every pair of states one wave vector, a + b = c around the triangle -- has frames that leave
    /// nothing. From the frames that already close it the correction is exactly zero; from no frame at all it is those
    /// frames, K = (0, a, a + b), to the solve's rounding: one rule, whatever it starts from.
    #[test]
    fn a_closing_loop_s_frames_leave_nothing() {
        let (a, b) = (wave(1, 0), wave(0, 1));
        let closed = with_adjoints(&[(1, 0, a.clone()), (2, 1, b.clone()), (2, 0, a.plus(1, &b)), (0, 0, wave(0, 0)), (1, 1, wave(0, 0)), (2, 2, wave(0, 0))]);
        let frames = [wave(0, 0), a.clone(), a.plus(1, &b)];
        let in_the_frames: Vec<(usize, usize, Wave)> = closed.iter().map(|(x, y, q)| (*x, *y, q.plus(-1, &frames[*x]).plus(1, &frames[*y]))).collect();
        assert!(in_the_frames.iter().all(|h| h.2.is_zero()));
        assert_eq!(corrected(3, &in_the_frames), vec![[0.0; 3]; 3]);
        let delta = corrected(3, &closed);
        let scale = norm3(value(&a.plus(1, &b)));
        for (s, frame) in frames.iter().enumerate() {
            let apart = norm3(std::array::from_fn(|i| delta[s][i] - value(frame)[i]));
            assert!(apart <= within(3, scale), "state {s}: {apart:e} from its frame");
        }
        assert!(closed.iter().all(|h| norm3(left(h, &delta)) <= within(3, scale)));
    }

    /// Where the wave vectors do not close, the frames are the least squares of what is left.
    /// - A pair of states that holds several plane waves takes their mean: K_1 − K_0 = (q_1 + q_2 + q_3)/3.
    /// - A triangle whose loop misses by d = c − a − b shares it equally: each of its three plane waves keeps d/3
    ///   (K_1 − K_0 = a + d/3, K_2 − K_1 = b + d/3, K_2 − K_0 = c − d/3), the sum of squares |d|²/3 -- with the
    ///   adjoints', twice that; no frame leaves less: the sum is convex and stationary there, and every other frame tried
    ///   leaves more.
    /// - The diagonal's plane waves are no frame's to move: they change nothing.
    /// - Two sets of states no entry joins are corrected apart, each from its first state.
    #[test]
    fn where_the_loops_do_not_close_the_frames_are_the_least_squares() {
        let several = with_adjoints(&[(1, 0, wave(2, 0)), (1, 0, wave(0, 1)), (1, 0, wave(-1, 1)), (0, 0, wave(1, -1)), (1, 1, wave(2, 0))]);
        let delta = corrected(2, &several);
        let mean = value(&wave(1, 2)).map(|x| x / 3.0);
        assert_eq!(delta[0], [0.0; 3]);
        assert!(norm3(std::array::from_fn(|i| delta[1][i] - mean[i])) <= within(2, norm3(value(&wave(2, 1)))), "{delta:?}");

        let (a, b, d) = (wave(1, 0), wave(0, 1), wave(1, -2));
        let open = with_adjoints(&[(1, 0, a.clone()), (2, 1, b.clone()), (2, 0, a.plus(1, &b).plus(1, &d)), (1, 1, wave(1, 1))]);
        let delta = corrected(3, &open);
        let (third, scale) = (value(&d).map(|x| x / 3.0), norm3(value(&a.plus(1, &b).plus(1, &d))));
        for h in open.iter().filter(|h| h.0 != h.1) {
            let sign = if (h.0, h.1) == (2, 0) || (h.0, h.1) == (0, 1) || (h.0, h.1) == (1, 2) { 1.0 } else { -1.0 };
            let apart = norm3(std::array::from_fn(|i| left(h, &delta)[i] - sign * third[i]));
            assert!(apart <= within(3, scale), "{} ← {}: {apart:e} from ±d/3", h.0, h.1);
        }
        let (least, diagonal) = (squares(&open, &delta), 2.0 * dot3(value(&wave(1, 1)), value(&wave(1, 1))));
        assert!((least - diagonal - 2.0 * dot3(value(&d), value(&d)) / 3.0).abs() <= within(3, scale * scale), "{least}");
        for moved in 0..3 * 3 * 2 {
            let mut other = delta.clone();
            other[moved / 6][moved / 2 % 3] += if moved % 2 == 0 { 1.0 } else { -1.0 } * scale / 64.0;
            assert!(squares(&open, &other) > least, "the frame moved ({moved}) leaves less");
        }
        assert_eq!(corrected(3, &open[..6]), delta, "the diagonal's plane waves moved the frames");

        let apart = with_adjoints(&[(1, 0, wave(1, 0)), (1, 0, wave(0, 1)), (3, 2, wave(1, 1)), (3, 2, wave(-1, 1)), (3, 2, wave(0, 4))]);
        let delta = corrected(4, &apart);
        assert!(delta[0] == [0.0; 3] && delta[2] == [0.0; 3], "{delta:?}");
        let means = [value(&wave(1, 1)).map(|x| x / 2.0), value(&wave(0, 6)).map(|x| x / 3.0)];
        for (s, mean) in [(1, means[0]), (3, means[1])] {
            assert!(norm3(std::array::from_fn(|i| delta[s][i] - mean[i])) <= within(4, norm3(value(&wave(0, 6)))), "{delta:?}");
        }
    }

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

    /// The frames (module documentation, # Frames; ELIMINATION v4): a chain of components is static in its frames, exactly
    /// -- the residual is the zero beat, not a value near it --; a ring of three that misses by D is left D/3 on each of
    /// its components, exactly (a fraction of the axes with the denominator 3), which no frames beat: the residuals'
    /// signed sum around the ring is D whatever the frames, so one of the three is at least |D|/3. The values' frames
    /// (the elimination's closure) are the same to rounding.
    #[test]
    fn the_frames_leave_a_chain_static_and_a_ring_its_detuning_over_three() {
        let chain = vec![(1, 0, beat(1, 0)), (2, 1, beat(1, 1)), (3, 2, beat(0, -1))];
        let (theta, residual) = frames(5, &chain, &omega, &beat(0, 0));
        assert!(residual.is_zero(), "{residual:?}");
        assert!(theta[0].is_zero() && theta[4].is_zero(), "the first state and a state of no component at zero");
        for c in &chain {
            assert!(left_by(c, &theta).is_zero(), "{} ← {}: {:?}", c.0, c.1, left_by(c, &theta));
        }
        // a + b + c = D = (1, −2): each component left D/3 in the sense of the ring.
        let ring = vec![(1, 0, beat(1, 0)), (2, 1, beat(0, 1)), (0, 2, beat(0, -3))];
        let (theta, residual) = frames(3, &ring, &omega, &beat(0, 0));
        let third = beat(1, -2).over(3);
        assert!(residual == third || residual == third.negated(), "{residual:?} against ±{third:?}");
        assert!((omega(&residual) - omega(&beat(1, -2)).abs() / 3.0).abs() <= 1e-14 * omega(&residual));
        for c in &ring {
            assert_eq!(left_by(c, &theta), third, "{} ← {}", c.0, c.1);
        }
        let values: Vec<(usize, usize, f64)> = ring.iter().map(|c| (c.0, c.1, omega(&c.2))).collect();
        let (theta_values, residual_value) = frames(3, &values, &|x| *x, &0.0);
        assert!((residual_value - omega(&residual)).abs() <= 1e-14 * residual_value);
        for (c, v) in ring.iter().zip(&values) {
            assert!((v.2 - theta_values[v.0] + theta_values[v.1] - omega(&left_by(c, &theta))).abs() <= 1e-13 * residual_value);
        }
        // Two components of one pair at two beats: a ring of two, each left half the difference.
        let (theta, residual) = frames(2, &[(1, 0, beat(0, 0)), (1, 0, beat(1, 0))], &omega, &beat(0, 0));
        assert_eq!(residual, beat(1, 0).over(2));
        assert_eq!(theta[1], beat(1, 0).over(2));
    }

    /// The frames past the largest cycle (max-balancing): a chain hung on a ring stays static, and a second ring that
    /// misses by less keeps its own smaller share -- no component turns faster than its own cycles ask, nor at all where
    /// none asks. Two rings of 3 and 2 joined by a chain: D₁/3 on the first, D₂/2 on the second, the chain static.
    #[test]
    fn past_the_largest_cycle_each_ring_keeps_its_own_share_and_a_chain_stays_static() {
        let mut components = vec![(1, 0, beat(1, 0)), (2, 1, beat(0, 1)), (0, 2, beat(0, -3))];
        components.extend([(3, 2, beat(2, 1)), (4, 3, beat(-1, 0))]);
        components.extend([(5, 4, beat(0, 1)), (4, 5, beat(0, 0))]);
        let (theta, residual) = frames(6, &components, &omega, &beat(0, 0));
        let (first, second) = (beat(1, -2).over(3), beat(0, 1).over(2));
        assert!(omega(&first).abs() > omega(&second).abs());
        assert!(residual == first || residual == first.negated(), "{residual:?}");
        for c in &components[..3] {
            assert_eq!(left_by(c, &theta), first, "the first ring's {} ← {}", c.0, c.1);
        }
        for c in &components[3..5] {
            assert!(left_by(c, &theta).is_zero(), "the chain's {} ← {}: {:?}", c.0, c.1, left_by(c, &theta));
        }
        let shares: Vec<Beat> = components[5..].iter().map(|c| left_by(c, &theta)).collect();
        assert!(shares.iter().all(|x| *x == second || *x == second.negated()), "the second ring's {shares:?}");
        assert!(theta[0].is_zero());
    }
}
