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
//! Each state of a cluster takes a frame θ_s, exactly an integer combination of the light's frequencies (`Beat`): ψ_s = e^{−iθ_s t}
//! x_s puts a term a ← b of beat Ω (∝ e^{iΩt}) at Ω + θ_a − θ_b, so that an edge is static when θ_a − θ_b is its own frame
//! difference. Along a spanning tree of the kept edges every one of them is static, exactly; an edge off the tree, or a term
//! the tree leaves, keeps its residual beat, which the propagator resolves. The tree is the maximum one by strength (Kruskal):
//! the frames are exact for any tree, and the strongest couplings static leave the propagator the least to resolve (a choice
//! of cost, not of accuracy).
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
//! They are made as the tree's frames -- a drive's strongest beam's k_l, a coupling's strongest beam pair's k_l − k_l'
//! along the tree, exact integers (`Cluster::waves`) -- plus a correction δ (`correction`): L δ = the sums of the tree's
//! residuals, which are integers too. Where the tree's frames leave nothing the sums are exactly zero and so is δ, whatever
//! the rounding of the solve; any tree gives the same K.

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

/// The clusters of the kept edges over `states` states, with their exact frames along the tree (module documentation; the
/// Doppler frames' values are the model's to take), and each ground state's cluster. `zero` and `still` are the zero beat
/// and wave vector.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::{dot3, norm3};
    use crate::units::rounding;

    /// Two wave vectors that are not orthogonal (1/µm): the coordinates of the tests' exact ones.
    const AXES: [[f64; 3]; 2] = [[8.0, 0.0, 0.0], [3.0, 5.0, 1.0]];

    fn wave(a: i64, b: i64) -> Wave {
        Beat(vec![a, b])
    }
    fn value(w: &Wave) -> [f64; 3] {
        std::array::from_fn(|i| w.0[0] as f64 * AXES[0][i] + w.0[1] as f64 * AXES[1][i])
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
}
