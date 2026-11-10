//! Taylor jets in the position: a quantity's value with its gradient (order 1) or also its Hessian (order 2), exact -- the force
//! and the potential's expansion take the elimination's derivatives through them (eliminate.rs).
//!
//! A jet holds 1, 4 or 10 parts: the value, ∂_x, ∂_y, ∂_z, then ∂_xx, ∂_xy, ∂_xz, ∂_yy, ∂_yz, ∂_zz. Products follow Leibniz's
//! rule, (ab)_ij = a_ij b + a_i b_j + a_j b_i + a b_ij; a resolvent x = M⁻¹b with one factorization of the value M₀:
//! x_i = M₀⁻¹(b_i − M_i x), x_ij = M₀⁻¹(b_ij − M_i x_j − M_j x_i − M_ij x) -- the derivatives of M x = b, exact.

use crate::field::Amplitude;
use crate::linalg::{Lu, Mat, C64};

/// The parts of a jet of an order (0, 1 or 2).
pub(crate) fn parts(order: usize) -> usize {
    [1, 4, 10][order]
}

/// The complex multiply-adds of an n × n LU factorization (linalg.rs `Lu::new`): at column k, n − k − 1 multipliers and
/// (n − k − 1)² updates, Σ = (n³ − n)/3.
pub(crate) fn factor_work(n: usize) -> f64 {
    ((n * n * n - n) / 3) as f64
}

/// The complex multiply-adds of `solve` on n unknowns with `parts` parts: n² per substitution (linalg.rs `Lu::solve`), a matrix
/// product M_i x (n²) and a substitution per first derivative, three products and a substitution per second.
pub(crate) fn solve_work(n: usize, parts: usize) -> f64 {
    let n2 = (n * n) as f64;
    n2 * (1.0 + if parts >= 4 { 6.0 } else { 0.0 } + if parts >= 10 { 24.0 } else { 0.0 })
}

/// The complex multiply-adds of `product` of two jets of `parts` parts: one, two per first derivative, four per second.
pub(crate) fn product_work(parts: usize) -> f64 {
    1.0 + if parts >= 4 { 6.0 } else { 0.0 } + if parts >= 10 { 24.0 } else { 0.0 }
}

/// The part of the second derivative ∂_i ∂_j.
pub(crate) fn second(i: usize, j: usize) -> usize {
    4 + [[0, 1, 2], [1, 3, 4], [2, 4, 5]][i][j]
}

/// The axis pairs (i, j), i ≤ j, of the second parts in their order.
pub(crate) const PAIRS: [(usize, usize); 6] = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)];

/// An amplitude's whole jet: its value, its gradient, its Hessian's six parts.
pub(crate) fn whole(c: &Amplitude) -> [C64; 10] {
    std::array::from_fn(|p| match p {
        0 => c.value,
        1..=3 => c.gradient[p - 1],
        _ => c.hessian[PAIRS[p - 4].0][PAIRS[p - 4].1],
    })
}

/// The amplitude of a whole jet (`whole`'s inverse: the Hessian symmetric).
pub(crate) fn amplitude(jet: &[C64; 10]) -> Amplitude {
    Amplitude { value: jet[0], gradient: std::array::from_fn(|i| jet[1 + i]), hessian: std::array::from_fn(|i| std::array::from_fn(|j| jet[second(i, j)])) }
}

/// A beam coefficient's jet (an amplitude or a pair's product) to an order.
pub(crate) fn of(c: &Amplitude, order: usize) -> Vec<C64> {
    whole(c)[..parts(order)].to_vec()
}

/// The product jet of two scalar jets into `out`, as many parts as it holds: Leibniz's rule (module documentation), written once.
pub(crate) fn multiply(a: &[C64], b: &[C64], out: &mut [C64]) {
    out[0] = a[0] * b[0];
    if out.len() >= 4 {
        for i in 0..3 {
            out[1 + i] = a[1 + i] * b[0] + a[0] * b[1 + i];
        }
    }
    if out.len() >= 10 {
        for (k, &(i, j)) in PAIRS.iter().enumerate() {
            out[4 + k] = a[4 + k] * b[0] + a[1 + i] * b[1 + j] + a[1 + j] * b[1 + i] + a[0] * b[4 + k];
        }
    }
}

/// The product jet of two scalar jets.
pub(crate) fn product(a: &[C64], b: &[C64]) -> Vec<C64> {
    let mut out = vec![C64::ZERO; a.len()];
    multiply(a, b, &mut out);
    out
}

/// The jet of 1/d for a real jet d.
pub(crate) fn reciprocal(d: &[f64]) -> Vec<f64> {
    let n = d.len();
    let g = 1.0 / d[0];
    let mut out = vec![g];
    if n >= 4 {
        for i in 0..3 {
            out.push(-d[1 + i] * g * g);
        }
    }
    if n >= 10 {
        for (k, &(i, j)) in PAIRS.iter().enumerate() {
            out.push(2.0 * d[1 + i] * d[1 + j] * g * g * g - d[4 + k] * g * g);
        }
    }
    out
}

/// The jet of M⁻¹ b for a matrix jet M (its value factored as `lu`) and a vector jet b (module documentation).
pub(crate) fn solve(lu: &Lu, m: &[Mat], b: &[Vec<C64>]) -> Vec<Vec<C64>> {
    let n = b.len();
    let mut x0 = b[0].clone();
    lu.solve(&mut x0);
    let mut out = vec![x0];
    if n >= 4 {
        for i in 0..3 {
            let mi = m[1 + i].apply(&out[0]);
            let mut y: Vec<C64> = b[1 + i].iter().zip(&mi).map(|(p, q)| *p - *q).collect();
            lu.solve(&mut y);
            out.push(y);
        }
    }
    if n >= 10 {
        for (k, &(i, j)) in PAIRS.iter().enumerate() {
            let (a, c, d) = (m[1 + i].apply(&out[1 + j]), m[1 + j].apply(&out[1 + i]), m[4 + k].apply(&out[0]));
            let mut y: Vec<C64> = (0..b[4 + k].len()).map(|x| b[4 + k][x] - a[x] - c[x] - d[x]).collect();
            lu.solve(&mut y);
            out.push(y);
        }
    }
    out
}

/// Σ_parts of a scalar jet times a constant matrix, added to a matrix jet: out_p += c_p m.
pub(crate) fn add_scaled(out: &mut [Mat], c: &[C64], m: &Mat) {
    for (o, z) in out.iter_mut().zip(c) {
        if *z != C64::ZERO {
            o.add_scaled(*z, m);
        }
    }
}
