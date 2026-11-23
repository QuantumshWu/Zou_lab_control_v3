//! The folded block's spectrum (model/rs.rs, # At a point; lin.status.md 1.1, `lin.spectrum`): G = V diag(c + μ′) V⁻¹
//! with the block first decoupled, exactly to rounding, at the gaps of its diagonal that pay, and each leaf decomposed
//! about its own centre c. One decomposition of a whole block mixes the D1 levels' 4.5·10⁷ rad/µs (the frame's) with
//! the D2 levels' denominators of 10²–10³ rad/µs: an eigenvalue's absolute error is the block's scale, εmax|G| ≈ 10⁻⁸
//! rad/µs, 3·10⁴ ε on a D2 denominator (measured in the prototype). Decoupled, each leaf's eigenvalues err by its own
//! scale.
//!
//! # The decoupling
//! At a split of the indices into J and K (`split_into`), G = [[A₁₁, A₁₂], [A₂₁, A₂₂]] in that order; the transform
//! T = [[I, Y], [X, I + XY]], T⁻¹ = [[I + YX, −Y], [−X, I]] makes T⁻¹GT = diag(B₁₁, B₂₂), B₁₁ = A₁₁ + A₁₂X,
//! B₂₂ = A₂₂ − XA₁₂, where X solves the Riccati equation A₂₂X − XA₁₁ + A₂₁ − XA₁₂X = 0 and Y the Sylvester equation
//! B₁₁Y − YB₂₂ = −A₁₂. Both are solved by the iteration about the diagonals, X ← (−A₂₁ + XA₁₂X − N₂X + XN₁) ⊘ Δ, Δ_ij =
//! d₂ᵢ − d₁ⱼ (N the off-diagonal parts, d the diagonals), until a sweep no longer moves X by more than its rounding (or
//! no longer moves it less than the sweep before: the rounding's floor). A split is taken where it pays (`split_into`): the
//! gap between the two sides' diagonals is wider than both sides' spreads (their diagonals' range and twice their
//! largest off-diagonal row sum) and than twice the coupling across: the iteration's ratio is then below one half. Which
//! split is taken chooses only which exact transform is used.
//!
//! # Its basis from the leaves'
//! With V_J, V_K the two sides' bases (their leaves' eigenvectors, each side decoupled in turn), the block's is
//! V = T diag(V_J, V_K) and V⁻¹ = diag(V_J⁻¹, V_K⁻¹) T⁻¹, made by blocks without T:
//!   V = [[V_J, Y V_K], [X V_J, V_K + X (Y V_K)]],  V⁻¹ = [[V_J⁻¹ + (V_J⁻¹Y) X, −V_J⁻¹Y], [−V_K⁻¹X, V_K⁻¹]],
//! six products of 3|J||K|(|J| + |K|) multiply-adds in all; a leaf of one state is its own value, V = V⁻¹ = 1. Every
//! product is the product kernel's (`linalg::fused_product`) on the spectrum's own buffers, kept by depth of the
//! decoupling (each depth's made for the largest block it can hold where it is first reached): a decomposition
//! allocates nothing once its depths and leaves have been met.

use crate::linalg::{self, larger, smaller, C64, Mat};
use crate::units::UNIT_ROUNDOFF;

/// The spectrum of a block (module documentation): V, V⁻¹, and per eigen index its leaf's centre c and μ′ (the
/// eigenvalue c + μ′), the leaves' sizes, and the counted complex multiply-adds; its buffers kept for the next.
#[derive(Clone, Debug, Default)]
pub struct Spectrum {
    pub vectors: Mat,
    pub inverse: Mat,
    pub centre: Vec<f64>,
    pub values: Vec<C64>,
    pub leaves: Vec<usize>,
    pub work: f64,
    /// The leaves' decompositions, by leaf (`Eigen` keeps its memory), and the decoupling's buffers by depth.
    eigens: Vec<linalg::Eigen>,
    levels: Vec<Level>,
    /// `split_into`'s: the diagonal, its order, the gaps.
    diagonal: Vec<f64>,
    order: Vec<usize>,
    gaps: Vec<(f64, usize)>,
}

/// One depth of the decoupling (`decouple`): its split, the four blocks, X and Y and their iteration's, the two sides'
/// generators B₁₁, B₂₂ and bases, the products', and a leaf's shifted block.
#[derive(Clone, Debug, Default)]
struct Level {
    made: bool,
    left: Vec<usize>,
    right: Vec<usize>,
    a11: Vec<C64>,
    a12: Vec<C64>,
    a21: Vec<C64>,
    a22: Vec<C64>,
    x: Vec<C64>,
    y: Vec<C64>,
    c: Vec<C64>,
    rhs: Vec<C64>,
    delta: Vec<C64>,
    near: Vec<C64>,
    far: Vec<C64>,
    t: Vec<C64>,
    b11: Vec<C64>,
    b22: Vec<C64>,
    vj: Vec<C64>,
    vji: Vec<C64>,
    vk: Vec<C64>,
    vki: Vec<C64>,
    u: Vec<C64>,
    w: Vec<C64>,
    shifted: Mat,
}

impl Level {
    /// The depth's buffers for blocks up to m states, every one at its largest at once.
    fn make(&mut self, m: usize) {
        if self.made {
            return;
        }
        self.made = true;
        self.left.reserve(m);
        self.right.reserve(m);
        for v in [
            &mut self.a11, &mut self.a12, &mut self.a21, &mut self.a22, &mut self.x, &mut self.y, &mut self.c, &mut self.rhs, &mut self.delta, &mut self.near, &mut self.far, &mut self.t,
            &mut self.b11, &mut self.b22, &mut self.vj, &mut self.vji, &mut self.vk, &mut self.vki, &mut self.u, &mut self.w, &mut self.shifted.data,
        ] {
            v.reserve(m * m);
        }
    }
}

/// `v` as `len` zeros.
fn zeroed(v: &mut Vec<C64>, len: usize) {
    v.clear();
    v.resize(len, C64::ZERO);
}

/// The widest gap of G's diagonal (real parts) at which a split pays (module documentation): the two sides, each sorted,
/// into `left` and `right`; false where none pays. G is m × m, row-major; `d` and `order` its buffers.
fn split_into(g: &[C64], m: usize, (d, order, gaps): (&mut Vec<f64>, &mut Vec<usize>, &mut Vec<(f64, usize)>), left: &mut Vec<usize>, right: &mut Vec<usize>) -> bool {
    if m < 2 {
        return false;
    }
    d.clear();
    d.extend((0..m).map(|i| g[i * m + i].re));
    order.clear();
    order.extend(0..m);
    order.sort_by(|&a, &b| d[a].total_cmp(&d[b]));
    let row_sum = |i: usize, side: &[usize]| -> f64 { side.iter().filter(|&&j| j != i).map(|&j| g[i * m + j].abs()).sum() };
    let spread = |side: &[usize]| -> f64 {
        let (lo, hi) = side.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &i| (smaller(lo, d[i]), larger(hi, d[i])));
        let off = if side.len() > 1 { side.iter().map(|&i| row_sum(i, side)).fold(0.0, larger) } else { 0.0 };
        hi - lo + 2.0 * off
    };
    // The widest gap that pays, the first of equal ones: the gaps tried widest first, the first that pays taken (a gap
    // that is not a number pays nowhere).
    gaps.clear();
    gaps.extend((1..m).map(|c| (d[order[c]] - d[order[c - 1]], c)));
    gaps.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let pays = |&(gap, c): &(f64, usize)| {
        let (l, r) = order.split_at(c);
        let cross = l.iter().flat_map(|&i| r.iter().map(move |&j| larger(g[i * m + j].abs(), g[j * m + i].abs()))).fold(0.0, larger);
        gap > 2.0 * cross && gap > spread(l) + spread(r)
    };
    let Some(&(_, c)) = gaps.iter().find(|x| pays(x)) else { return false };
    left.clear();
    left.extend_from_slice(&order[..c]);
    right.clear();
    right.extend_from_slice(&order[c..]);
    left.sort_unstable();
    right.sort_unstable();
    true
}

/// An iteration's buffers (`iterate`): the denominators, −N_L, N_R, the right-hand side, a product's.
struct Sweep<'a> {
    delta: &'a mut Vec<C64>,
    near: &'a mut Vec<C64>,
    far: &'a mut Vec<C64>,
    rhs: &'a mut Vec<C64>,
    t: &'a mut Vec<C64>,
}

/// The iteration about the diagonals (module documentation): X ← (C + f(X) − N_L X + X N_R) ⊘ (d_Lᵢ − d_Rⱼ) of X
/// (p × q, into `x`) of the left block L (p × p) and the right R (q × q), f(X) = X A X with `quadratic` A (q × p; the
/// Riccati's A₁₂, none for a Sylvester equation), until the sweep's change is within X's rounding or no longer shrinks.
/// Returns the counted complex multiply-adds.
fn iterate(left: &[C64], right: &[C64], c: &[C64], (p, q): (usize, usize), quadratic: Option<&[C64]>, x: &mut Vec<C64>, sweep: Sweep) -> f64 {
    let Sweep { delta, near, far, rhs, t } = sweep;
    near.clear();
    near.extend(left.iter().enumerate().map(|(at, z)| if at / p == at % p { C64::ZERO } else { -*z }));
    far.clear();
    far.extend(right.iter().enumerate().map(|(at, z)| if at / q == at % q { C64::ZERO } else { *z }));
    delta.clear();
    delta.extend((0..p * q).map(|at| left[(at / q) * (p + 1)] - right[(at % q) * (q + 1)]));
    x.clear();
    x.extend(c.iter().zip(delta.iter()).map(|(a, b)| *a / *b));
    let (mut last, mut work) = (f64::INFINITY, 0.0);
    loop {
        rhs.clear();
        rhs.extend_from_slice(c);
        if let Some(a) = quadratic {
            zeroed(t, p * p);
            linalg::fused_product(x, a, (p, q, p), t);
            linalg::fused_product_onto(t, x, (p, p, q), rhs);
            work += (2 * p * p * q) as f64;
        }
        linalg::fused_product_onto(near, x, (p, p, q), rhs);
        linalg::fused_product_onto(x, far, (p, q, q), rhs);
        work += (p * p * q + p * q * q) as f64;
        // The next iterate in place, its change from this one, and its largest entry.
        let (mut change, mut top) = (0.0_f64, 0.0_f64);
        for ((x, r), d) in x.iter_mut().zip(rhs.iter()).zip(delta.iter()) {
            let next = *r / *d;
            change = larger(change, (next - *x).abs());
            top = larger(top, next.abs());
            *x = next;
        }
        // Done at the rounding or where the change stops falling; a change that is not a number ends it too (the
        // iterate carries it out), where both comparisons would be false and the iteration would not end.
        if !(change > 2.0 * UNIT_ROUNDOFF * top) || !(change < last) {
            return work;
        }
        last = change;
    }
}

/// What every depth of the decoupling shares: the leaves' decompositions, `split_into`'s buffers, the leaves' sizes, the
/// count.
struct Shared<'a> {
    eigens: &'a mut Vec<linalg::Eigen>,
    diagonal: &'a mut Vec<f64>,
    order: &'a mut Vec<usize>,
    gaps: &'a mut Vec<(f64, usize)>,
    leaves: &'a mut Vec<usize>,
    work: f64,
}

/// G (m × m, row-major) = V diag(c + μ′) V⁻¹ on the decoupling's buffers from this depth on (`levels`, the first this
/// depth's; module documentation): V into `v` (its rows G's indices, its columns the eigen indices in the leaves'
/// order), V⁻¹ into `vi`, each eigen index's centre and μ′ into `centre` and `values`, the leaves' sizes onto the
/// shared list. False where a leaf has no decomposition.
#[allow(clippy::too_many_arguments)]
fn decouple(g: &[C64], m: usize, levels: &mut [Level], shared: &mut Shared, v: &mut [C64], vi: &mut [C64], centre: &mut [f64], values: &mut [C64]) -> bool {
    // The depth d's blocks hold at most n − d states, the depths from it on as many.
    let most = levels.len();
    let (here, below) = levels.split_first_mut().expect("a depth for every split (at most the block's size)");
    here.make(most);
    if !split_into(g, m, (&mut *shared.diagonal, &mut *shared.order, &mut *shared.gaps), &mut here.left, &mut here.right) {
        // A leaf: about its own centre c, the mean of Re G_ii; a single state its own value, V = V⁻¹ = 1.
        let c = (0..m).map(|i| g[i * m + i].re).sum::<f64>() / m.max(1) as f64;
        centre.fill(c);
        let leaf = shared.leaves.len();
        shared.leaves.push(m);
        if m == 1 {
            values[0] = g[0] - C64::real(c);
            (v[0], vi[0]) = (C64::ONE, C64::ONE);
            return true;
        }
        if shared.eigens.len() <= leaf {
            shared.eigens.resize_with(leaf + 1, linalg::Eigen::default);
        }
        let shifted = &mut here.shifted;
        (shifted.rows, shifted.cols) = (m, m);
        shifted.data.clear();
        shifted.data.extend_from_slice(&g[..m * m]);
        for i in 0..m {
            shifted.data[i * m + i] -= C64::real(c);
        }
        let eigen = &mut shared.eigens[leaf];
        if !eigen.decompose(shifted) {
            return false;
        }
        shared.work += eigen.work;
        values.copy_from_slice(&eigen.values[..m]);
        v.copy_from_slice(&eigen.vectors.data);
        vi.copy_from_slice(&eigen.inverse.data);
        return true;
    }
    let (nj, nk) = (here.left.len(), here.right.len());
    let Level { left, right, a11, a12, a21, a22, x, y, c, rhs, delta, near, far, t, b11, b22, vj, vji, vk, vki, u, w, .. } = here;
    let block = |out: &mut Vec<C64>, rows: &[usize], cols: &[usize]| {
        out.clear();
        out.extend(rows.iter().flat_map(|&i| cols.iter().map(move |&j| g[i * m + j])));
    };
    block(a11, left, left);
    block(a12, left, right);
    block(a21, right, left);
    block(a22, right, right);
    // X (|K| × |J|): A₂₂X − XA₁₁ = −A₂₁ + XA₁₂X.
    c.clear();
    c.extend(a21.iter().map(|z| -*z));
    shared.work += iterate(a22, a11, c, (nk, nj), Some(a12), x, Sweep { delta, near, far, rhs, t });
    // B₁₁ = A₁₁ + A₁₂X, B₂₂ = A₂₂ − XA₁₂.
    b11.clear();
    b11.extend_from_slice(a11);
    linalg::fused_product_onto(a12, x, (nj, nk, nj), b11);
    zeroed(t, nk * nk);
    linalg::fused_product(x, a12, (nk, nj, nk), t);
    b22.clear();
    b22.extend(a22.iter().zip(t.iter()).map(|(a, b)| *a - *b));
    // Y (|J| × |K|): B₁₁Y − YB₂₂ = −A₁₂.
    c.clear();
    c.extend(a12.iter().map(|z| -*z));
    shared.work += iterate(b11, b22, c, (nj, nk), None, y, Sweep { delta, near, far, rhs, t });
    shared.work += (2 * nj * nk * (nj + nk)) as f64;
    // Each side's basis, the right's on the depths below once the left's are done with them.
    zeroed(vj, nj * nj);
    zeroed(vji, nj * nj);
    zeroed(vk, nk * nk);
    zeroed(vki, nk * nk);
    if !decouple(b11, nj, below, shared, vj, vji, &mut centre[..nj], &mut values[..nj]) {
        return false;
    }
    if !decouple(b22, nk, below, shared, vk, vki, &mut centre[nj..], &mut values[nj..]) {
        return false;
    }
    // V by blocks (module documentation, # Its basis from the leaves'), its rows put back at G's indices: row J_p is
    // [V_J | Y V_K]_p, row K_q [X V_J | V_K + X (Y V_K)]_q.
    zeroed(u, nj * nk);
    linalg::fused_product(y, vk, (nj, nk, nk), u);
    zeroed(w, nk * nk);
    w.copy_from_slice(vk);
    linalg::fused_product_onto(x, u, (nk, nj, nk), w);
    for (p, &row) in left.iter().enumerate() {
        v[row * m..row * m + nj].copy_from_slice(&vj[p * nj..(p + 1) * nj]);
        v[row * m + nj..(row + 1) * m].copy_from_slice(&u[p * nk..(p + 1) * nk]);
    }
    for (q, &row) in right.iter().enumerate() {
        v[row * m + nj..(row + 1) * m].copy_from_slice(&w[q * nk..(q + 1) * nk]);
    }
    zeroed(t, nk * nj);
    linalg::fused_product(x, vj, (nk, nj, nj), t);
    for (q, &row) in right.iter().enumerate() {
        v[row * m..row * m + nj].copy_from_slice(&t[q * nj..(q + 1) * nj]);
    }
    // V⁻¹ by blocks, its columns put back at G's indices: rows 0..|J| are [V_J⁻¹ + (V_J⁻¹Y) X | −V_J⁻¹Y], the rest
    // [−V_K⁻¹X | V_K⁻¹].
    zeroed(u, nj * nk);
    linalg::fused_product(vji, y, (nj, nj, nk), u);
    zeroed(w, nj * nj);
    w.copy_from_slice(vji);
    linalg::fused_product_onto(u, x, (nj, nk, nj), w);
    zeroed(t, nk * nj);
    linalg::fused_product(vki, x, (nk, nk, nj), t);
    for r in 0..nj {
        for (p, &col) in left.iter().enumerate() {
            vi[r * m + col] = w[r * nj + p];
        }
        for (q, &col) in right.iter().enumerate() {
            vi[r * m + col] = -u[r * nk + q];
        }
    }
    for r in 0..nk {
        for (p, &col) in left.iter().enumerate() {
            vi[(nj + r) * m + col] = -t[r * nj + p];
        }
        for (q, &col) in right.iter().enumerate() {
            vi[(nj + r) * m + col] = vki[r * nk + q];
        }
    }
    shared.work += (3 * nj * nk * (nj + nk)) as f64;
    true
}

impl Spectrum {
    /// G's spectrum (module documentation); false where a leaf has no eigendecomposition.
    pub fn decompose(&mut self, g: &Mat) -> bool {
        let n = g.rows;
        assert_eq!(n, g.cols, "the spectrum of a {}x{} block", g.rows, g.cols);
        // A depth for every split: each takes a state at least off the block below it.
        if self.levels.len() < n.max(1) {
            self.levels.resize_with(n.max(1), Level::default);
        }
        for m in [&mut self.vectors, &mut self.inverse] {
            (m.rows, m.cols) = (n, n);
            zeroed(&mut m.data, n * n);
        }
        self.centre.clear();
        self.centre.resize(n, 0.0);
        self.values.clear();
        self.values.resize(n, C64::ZERO);
        self.leaves.clear();
        let mut shared = Shared { eigens: &mut self.eigens, diagonal: &mut self.diagonal, order: &mut self.order, gaps: &mut self.gaps, leaves: &mut self.leaves, work: 0.0 };
        let made = decouple(&g.data, n, &mut self.levels[..n.max(1)], &mut shared, &mut self.vectors.data, &mut self.inverse.data, &mut self.centre, &mut self.values);
        self.work = shared.work;
        made
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A block of two well separated groups -- one far off (a D1-like level at −4.5·10⁷) with a weak coupling across
    /// (7·10⁻⁴, the decay's cross terms) -- decomposed decoupled: G V = V diag(c + μ′) to the rounding of each leaf's own
    /// scale, V V⁻¹ = I, and the near group's eigenvalues against the near block decomposed alone with the coupling's
    /// second order folded in.
    #[test]
    fn the_decoupled_spectrum_is_exact_to_each_leaf_s_rounding() {
        let n = 6;
        let mut g = Mat::zeros(n, n);
        let far = [-4.5e7, -4.5e7 + 5.0e3, -4.5e7 + 5.1e3];
        let near = [120.0, 410.0, 900.0];
        for i in 0..3 {
            g[(i, i)] = C64::new(far[i], -19.0);
            g[(3 + i, 3 + i)] = C64::new(near[i], -19.0);
        }
        g[(0, 1)] = C64::new(3.0, 0.5);
        g[(1, 0)] = C64::new(3.0, -0.5);
        g[(4, 5)] = C64::new(2.0, 0.1);
        g[(5, 4)] = C64::new(2.0, -0.1);
        g[(2, 4)] = C64::new(0.0, -7e-4);
        g[(4, 2)] = C64::new(0.0, -7e-4);
        let mut s = Spectrum::default();
        assert!(s.decompose(&g));
        assert!(s.leaves.len() >= 2, "{:?}", s.leaves);
        let product = s.vectors.product(&s.inverse);
        let unit = (0..n).flat_map(|i| (0..n).map(move |j| (i, j))).map(|(i, j)| (product[(i, j)] - if i == j { C64::ONE } else { C64::ZERO }).abs()).fold(0.0, larger);
        assert!(unit < 1e-14, "{unit:e}");
        // The near leaf's residual G v − λ v per eigen index, relative to the near group's scale.
        for k in 0..n {
            let lambda = C64::real(s.centre[k]) + s.values[k];
            let v: Vec<C64> = (0..n).map(|i| s.vectors[(i, k)]).collect();
            let gv = g.apply(&v);
            let r = (0..n).map(|i| (gv[i] - lambda * v[i]).abs()).fold(0.0, larger);
            let scale = if s.centre[k] < 0.0 { 4.5e7 } else { 1e3 };
            eprintln!("k {k}: λ {lambda:?} residual {r:.2e} (scale {scale:e})");
            if s.centre[k] > 0.0 {
                assert!(r < 64.0 * UNIT_ROUNDOFF * scale, "{r:e}");
            }
        }
    }
}
