//! Taylor jets in the position: a quantity's value with its gradient (order 1) or also its Hessian (order 2), exact -- the force
//! and the potential's expansion take the elimination's derivatives through them (eliminate.rs).
//!
//! A jet holds 1, 4 or 10 parts: the value, ∂_x, ∂_y, ∂_z, then ∂_xx, ∂_xy, ∂_xz, ∂_yy, ∂_yz, ∂_zz. Products follow Leibniz's
//! rule, (ab)_ij = a_ij b + a_i b_j + a_j b_i + a b_ij; a resolvent x = M⁻¹b with one factorization of the value M₀:
//! x_i = M₀⁻¹(b_i − M_i x), x_ij = M₀⁻¹(b_ij − M_i x_j − M_j x_i − M_ij x) -- the derivatives of M x = b, exact.

use crate::field::Amplitude;
use crate::linalg::{Mat, C64};

/// The parts of a jet of an order (0, 1 or 2).
pub(crate) const fn parts(order: usize) -> usize {
    [1, 4, 10][order]
}

/// The parts of a whole jet, the second order's.
pub(crate) const WHOLE: usize = parts(2);

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
pub(crate) fn whole(c: &Amplitude) -> [C64; WHOLE] {
    std::array::from_fn(|p| match p {
        0 => c.value,
        1..=3 => c.gradient[p - 1],
        _ => c.hessian[PAIRS[p - 4].0][PAIRS[p - 4].1],
    })
}

/// The amplitude of a whole jet (`whole`'s inverse: the Hessian symmetric).
pub(crate) fn amplitude(jet: &[C64; WHOLE]) -> Amplitude {
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

/// Σ_parts of a scalar jet times a constant matrix, added to a matrix jet: out_p += c_p m.
pub(crate) fn add_scaled(out: &mut [Mat], c: &[C64], m: &Mat) {
    for (o, z) in out.iter_mut().zip(c) {
        if *z != C64::ZERO {
            o.add_scaled(*z, m);
        }
    }
}
