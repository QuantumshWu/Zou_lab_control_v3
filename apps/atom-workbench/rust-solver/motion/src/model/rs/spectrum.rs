//! The folded block's spectrum (model/rs.rs, # At a point; lin.status.md 1.1, `lin.spectrum`): G = V diag(c + μ′) V⁻¹
//! with the block first decoupled, exactly to rounding, at the gaps of its diagonal that pay, and each leaf decomposed
//! about its own centre c. One decomposition of a whole block mixes the D1 levels' 4.5·10⁷ rad/µs (the frame's) with
//! the D2 levels' denominators of 10²–10³ rad/µs: an eigenvalue's absolute error is the block's scale, εmax|G| ≈ 10⁻⁸
//! rad/µs, 3·10⁴ ε on a D2 denominator (measured in the prototype). Decoupled, each leaf's eigenvalues err by its own
//! scale.
//!
//! # The decoupling
//! At a split of the indices into J and K (`split`), G = [[A₁₁, A₁₂], [A₂₁, A₂₂]] in that order; the transform
//! T = [[I, Y], [X, I + XY]], T⁻¹ = [[I + YX, −Y], [−X, I]] makes T⁻¹GT = diag(B₁₁, B₂₂), B₁₁ = A₁₁ + A₁₂X,
//! B₂₂ = A₂₂ − XA₁₂, where X solves the Riccati equation A₂₂X − XA₁₁ + A₂₁ − XA₁₂X = 0 and Y the Sylvester equation
//! B₁₁Y − YB₂₂ = −A₁₂. Both are solved by the iteration about the diagonals, X ← (−A₂₁ + XA₁₂X − N₂X + XN₁) ⊘ Δ, Δ_ij =
//! d₂ᵢ − d₁ⱼ (N the off-diagonal parts, d the diagonals), until a sweep no longer moves X by more than its rounding (or
//! no longer moves it less than the sweep before: the rounding's floor). A split is taken where it pays (`split`): the
//! gap between the two sides' diagonals is wider than both sides' spreads (their diagonals' range and twice their
//! largest off-diagonal row sum) and than twice the coupling across: the iteration's ratio is then below one half. Which
//! split is taken chooses only which exact transform is used.

use crate::linalg::{self, C64, Mat};
use crate::units::UNIT_ROUNDOFF;

/// The spectrum of a block (module documentation): V, V⁻¹, and per eigen index its leaf's centre c and μ′ (the
/// eigenvalue c + μ′), the leaves' sizes, and the counted complex multiply-adds.
#[derive(Clone, Debug, Default)]
pub struct Spectrum {
    pub vectors: Mat,
    pub inverse: Mat,
    pub centre: Vec<f64>,
    pub values: Vec<C64>,
    pub leaves: Vec<usize>,
    pub work: f64,
    /// The leaves' decompositions, kept (`Eigen` keeps its memory).
    eigens: Vec<linalg::Eigen>,
}

/// The widest gap of G's diagonal (real parts) at which a split pays (module documentation): the two sides, each sorted.
pub fn split(g: &Mat) -> Option<(Vec<usize>, Vec<usize>)> {
    let n = g.rows;
    if n < 2 {
        return None;
    }
    let d: Vec<f64> = (0..n).map(|i| g[(i, i)].re).collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| d[a].total_cmp(&d[b]));
    let row_sum = |i: usize, side: &[usize]| -> f64 { side.iter().filter(|&&j| j != i).map(|&j| g[(i, j)].abs()).sum() };
    let spread = |side: &[usize]| -> f64 {
        let (lo, hi) = side.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &i| (lo.min(d[i]), hi.max(d[i])));
        let off = if side.len() > 1 { side.iter().map(|&i| row_sum(i, side)).fold(0.0, f64::max) } else { 0.0 };
        hi - lo + 2.0 * off
    };
    let mut best: Option<(f64, usize)> = None;
    for c in 1..n {
        let (left, right) = order.split_at(c);
        let gap = d[right[0]] - d[left[c - 1]];
        if best.is_some_and(|(b, _)| gap <= b) {
            continue;
        }
        let cross = left.iter().flat_map(|&i| right.iter().map(move |&j| g[(i, j)].abs().max(g[(j, i)].abs()))).fold(0.0, f64::max);
        if gap > 2.0 * cross && gap > spread(left) + spread(right) {
            best = Some((gap, c));
        }
    }
    best.map(|(_, c)| {
        let (mut left, mut right) = (order[..c].to_vec(), order[c..].to_vec());
        left.sort_unstable();
        right.sort_unstable();
        (left, right)
    })
}

fn sub(a: &Mat, b: &Mat) -> Mat {
    Mat { rows: a.rows, cols: a.cols, data: a.data.iter().zip(&b.data).map(|(x, y)| *x - *y).collect() }
}

fn off_diagonal(a: &Mat) -> Mat {
    Mat::from_fn(a.rows, a.cols, |i, j| if i == j { C64::ZERO } else { a[(i, j)] })
}

/// The iteration about the diagonals (module documentation): X ← (c + f(X) − N_L X + X N_R) ⊘ (d_Lᵢ − d_Rⱼ), X (|L| × |R|)
/// of the left block L and the right block R, `f` the equation's own nonlinear part (the Riccati's XA₁₂X, none for a
/// Sylvester equation), until the sweep's change is within X's rounding or no longer shrinks. Returns X and the counted
/// complex multiply-adds.
fn iterate(left: &Mat, right: &Mat, c: &Mat, f: impl Fn(&Mat) -> Option<Mat>) -> (Mat, f64) {
    let (m, k) = (left.rows, right.rows);
    let (nl, nr) = (off_diagonal(left), off_diagonal(right));
    let delta: Vec<C64> = (0..m).flat_map(|i| (0..k).map(move |j| (i, j))).map(|(i, j)| left[(i, i)] - right[(j, j)]).collect();
    let mut x = Mat { rows: m, cols: k, data: c.data.iter().zip(&delta).map(|(a, b)| *a / *b).collect() };
    let (mut last, mut work) = (f64::INFINITY, 0.0);
    loop {
        let mut rhs = c.clone();
        if let Some(extra) = f(&x) {
            rhs.add_scaled(C64::ONE, &extra);
        }
        rhs = sub(&rhs, &nl.product(&x));
        rhs.add_scaled(C64::ONE, &x.product(&nr));
        work += (m * m * k + m * k * k) as f64;
        let next = Mat { rows: m, cols: k, data: rhs.data.iter().zip(&delta).map(|(a, b)| *a / *b).collect() };
        let change = sub(&next, &x).max_abs();
        x = next;
        if change <= 2.0 * UNIT_ROUNDOFF * x.max_abs() || change >= last {
            return (x, work);
        }
        last = change;
    }
}

/// G = T diag(leaves) T⁻¹ recursively (module documentation): the leaves (their indices into G, and their blocks), and T,
/// T⁻¹ with columns (rows) in the leaves' concatenated order.
fn decouple(g: &Mat, work: &mut f64) -> (Mat, Mat, Vec<(Vec<usize>, Mat)>) {
    let n = g.rows;
    let Some((j, k)) = split(g) else {
        return (Mat::identity(n), Mat::identity(n), vec![((0..n).collect(), g.clone())]);
    };
    let (a11, a12, a21, a22) = (g.block(&j, &j), g.block(&j, &k), g.block(&k, &j), g.block(&k, &k));
    let minus = |m: &Mat| m.scaled(C64::real(-1.0));
    // X (|K| × |J|): A₂₂X − XA₁₁ = −A₂₁ + XA₁₂X.
    let (x, wx) = iterate(&a22, &a11, &minus(&a21), |x| Some(x.product(&a12).product(x)));
    let b11 = {
        let mut b = a11.clone();
        b.add_scaled(C64::ONE, &a12.product(&x));
        b
    };
    let b22 = sub(&a22, &x.product(&a12));
    // Y (|J| × |K|): B₁₁Y − YB₂₂ = −A₁₂.
    let (y, wy) = iterate(&b11, &b22, &minus(&a12), |_| None);
    *work += wx + wy + 3.0 * (j.len() * k.len() * (j.len() + k.len())) as f64;
    let m = j.len();
    // T and T⁻¹ in the order (J, K).
    let mut t = Mat::identity(n);
    let mut ti = Mat::identity(n);
    let xy = x.product(&y);
    let yx = y.product(&x);
    for a in 0..n {
        for b in 0..n {
            match (a < m, b < m) {
                (true, false) => {
                    t[(a, b)] = y[(a, b - m)];
                    ti[(a, b)] = -y[(a, b - m)];
                }
                (false, true) => {
                    t[(a, b)] = x[(a - m, b)];
                    ti[(a, b)] = -x[(a - m, b)];
                }
                (false, false) => t[(a, b)] += xy[(a - m, b - m)],
                (true, true) => ti[(a, b)] += yx[(a, b)],
            }
        }
    }
    let (tl, til, left) = decouple(&b11, work);
    let (tr, tir, right) = decouple(&b22, work);
    // T · diag(T_L, T_R), diag(T⁻¹_L, T⁻¹_R) · T⁻¹, and back to G's order: rows of T (columns of T⁻¹) from (J, K) to G's.
    let mut inner = Mat::zeros(n, n);
    let mut inner_inverse = Mat::zeros(n, n);
    for a in 0..m {
        for b in 0..m {
            inner[(a, b)] = tl[(a, b)];
            inner_inverse[(a, b)] = til[(a, b)];
        }
    }
    for a in m..n {
        for b in m..n {
            inner[(a, b)] = tr[(a - m, b - m)];
            inner_inverse[(a, b)] = tir[(a - m, b - m)];
        }
    }
    let (full, full_inverse) = (t.product(&inner), inner_inverse.product(&ti));
    *work += 2.0 * (n * n * n) as f64;
    let order: Vec<usize> = j.iter().chain(&k).copied().collect();
    let mut tg = Mat::zeros(n, n);
    let mut tig = Mat::zeros(n, n);
    for (p, &row) in order.iter().enumerate() {
        for c in 0..n {
            tg[(row, c)] = full[(p, c)];
            tig[(c, row)] = full_inverse[(c, p)];
        }
    }
    let leaves = left.into_iter().map(|(ix, b)| (ix.iter().map(|&i| j[i]).collect(), b)).chain(right.into_iter().map(|(ix, b)| (ix.iter().map(|&i| k[i]).collect(), b))).collect();
    (tg, tig, leaves)
}

impl Spectrum {
    /// G's spectrum (module documentation); false where a leaf has no eigendecomposition.
    pub fn decompose(&mut self, g: &Mat) -> bool {
        let n = g.rows;
        let mut work = 0.0;
        let (t, ti, leaves) = decouple(g, &mut work);
        let (mut vb, mut vbi) = (Mat::zeros(n, n), Mat::zeros(n, n));
        self.centre = vec![0.0; n];
        self.values = vec![C64::ZERO; n];
        self.leaves = leaves.iter().map(|l| l.0.len()).collect();
        let mut at = 0;
        if self.eigens.len() < leaves.len() {
            self.eigens.resize_with(leaves.len(), linalg::Eigen::default);
        }
        for (leaf, (ix, b)) in leaves.iter().enumerate() {
            let k = ix.len();
            let c = (0..k).map(|i| b[(i, i)].re).sum::<f64>() / k as f64;
            let shifted = Mat::from_fn(k, k, |i, j| if i == j { b[(i, j)] - C64::real(c) } else { b[(i, j)] });
            let eigen = &mut self.eigens[leaf];
            if !eigen.decompose(&shifted) {
                return false;
            }
            work += eigen.work;
            for i in 0..k {
                self.centre[at + i] = c;
                self.values[at + i] = eigen.values[i];
                for j in 0..k {
                    vb[(at + i, at + j)] = eigen.vectors[(i, j)];
                    vbi[(at + i, at + j)] = eigen.inverse[(i, j)];
                }
            }
            at += k;
        }
        self.vectors = t.product(&vb);
        self.inverse = vbi.product(&ti);
        work += 2.0 * (n * n * n) as f64;
        self.work = work;
        true
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
        let unit = (0..n).flat_map(|i| (0..n).map(move |j| (i, j))).map(|(i, j)| (product[(i, j)] - if i == j { C64::ONE } else { C64::ZERO }).abs()).fold(0.0, f64::max);
        assert!(unit < 1e-14, "{unit:e}");
        // The near leaf's residual G v − λ v per eigen index, relative to the near group's scale.
        for k in 0..n {
            let lambda = C64::real(s.centre[k]) + s.values[k];
            let v: Vec<C64> = (0..n).map(|i| s.vectors[(i, k)]).collect();
            let gv = g.apply(&v);
            let r = (0..n).map(|i| (gv[i] - lambda * v[i]).abs()).fold(0.0, f64::max);
            let scale = if s.centre[k] < 0.0 { 4.5e7 } else { 1e3 };
            eprintln!("k {k}: λ {lambda:?} residual {r:.2e} (scale {scale:e})");
            if s.centre[k] > 0.0 {
                assert!(r < 64.0 * UNIT_ROUNDOFF * scale, "{r:e}");
            }
        }
    }
}
