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
//! residual is not what they minimise: its minimum is neither one frame nor a closed form; the propagation integrates
//! what is left at its combined frequencies, spectral.rs, # Tones).
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
