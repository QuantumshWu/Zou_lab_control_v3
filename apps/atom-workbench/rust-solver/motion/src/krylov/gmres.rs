//! Right-preconditioned GMRES to the rounding floor, and its certificate (FASTREF.md §§ 2.2, 3; `fr.gmres`, `fr.Rows`
//! with the root fix of `frx.py`).
//!
//! # The norm it minimises
//! Rows are equilibrated: W = diag(1/a_i), a_i = Σ_j |A_ij| (LAPACK xGEEQU/xGERFS; Higham 2002, §§ 7.2–7.3). The
//! system's rows span 32 to 4.5e7 in 1-norm (optical coherences of another line), so an unscaled norm would let the
//! large rows hide the residual of the small ones. GMRES minimises ‖W r‖₂ over x0 + P⁻¹K_k (right preconditioning:
//! the residual is the original system's), modified Gram–Schmidt Arnoldi (backward stable: Paige, Rozložník &
//! Strakoš, SIMAX 28 (2006) 264), the small least-squares problem by Givens rotations.
//!
//! # The stop: the rounding floor
//! ‖W r‖₂ of GMRES's iterates is non-increasing in exact arithmetic. In floating point it stops decreasing exactly
//! when the residual has reached the floor of its own evaluation: the iteration runs while it decreases and returns
//! the iterate of least ‖W r‖₂ -- no target number. (Stopping at the certificate instead leaves 1.2e-11 against the
//! direct LU where the problem's condition is 1e5, GMD1C at B = 0; two or three more rounds to the floor bring it to
//! 6e-13: FASTREF.md §3.) It also ends at an Arnoldi breakdown (exact in exact arithmetic) and when the Krylov space
//! is the whole space (k = n).
//!
//! # The certificate
//! The computed residual of row i itself errs by |Δr_i| ≤ γ (|A||x| + |b|)_i ≤ γ (a_i ‖x‖∞ + |b_i|) (Higham 2002,
//! Lemma 3.1 and §12.1), so a residual below that cannot be computed: the iterate is certified iff every row's
//! backward error is at that level,
//!
//! ```text
//!     ζ = max_i |r_i| / (γ_i (a_i ‖x‖∞ + |b_i|)) ≤ 1.
//! ```
//!
//! Row i's residual b_i − Σ_j A_ij x_j is a sum of m_i + 1 complex terms (m_i stored entries); each complex product's
//! real and imaginary parts are a sum of two real products, so each part is a real sum of 2(m_i + 1) rounded
//! operations: γ_i = γ_{2(m_i+1)} = 2(m_i + 1)u / (1 − 2(m_i + 1)u) (`units::rounding`), which is the prototype's
//! (m_i + 1)ε / (1 − (m_i + 1)ε) with ε = 2u, digit for digit.

use crate::linalg::larger;
use super::{product, row, Schur};
use crate::floquet::Sparse;
use crate::linalg::C64;
use crate::units::rounding;

/// The row-equilibration of a matrix: the row 1-norms a_i, the weights w_i = 1/a_i (1 on a zero row), each row's
/// rounding level γ_i.
pub struct Rows {
    norm: Vec<f64>,
    weight: Vec<f64>,
    level: Vec<f64>,
}

impl Rows {
    pub fn new(a: &Sparse) -> Rows {
        let n = a.rows;
        let norm: Vec<f64> = (0..n).map(|i| row(a, i).1.iter().map(|v| v.abs()).sum()).collect();
        let weight = norm.iter().map(|&s| if s == 0.0 { 1.0 } else { 1.0 / s }).collect();
        let level = (0..n).map(|i| rounding(2 * (a.indptr[i + 1] - a.indptr[i] + 1))).collect();
        Rows { norm, weight, level }
    }

    /// The zero rows whose right-hand side is not zero (0 = b_i: no solution).
    pub fn inconsistent(&self, b: &[C64]) -> Vec<usize> {
        (0..b.len()).filter(|&i| self.norm[i] == 0.0 && b[i] != C64::ZERO).collect()
    }

    /// (η, ζ): the row-equilibrated backward error max_i |r_i| / (a_i ‖x‖∞ + |b_i|), and the same over each row's
    /// rounding level. A row with a_i ‖x‖∞ + |b_i| = 0 has r_i = 0 exactly and counts 0.
    pub fn backward(&self, x: &[C64], b: &[C64], r: &[C64]) -> (f64, f64) {
        let size = x.iter().map(|v| v.abs()).fold(0.0, larger);
        let (mut eta, mut zeta) = (0.0f64, 0.0f64);
        for i in 0..b.len() {
            let scale = self.norm[i] * size + b[i].abs();
            if scale > 0.0 {
                let e = r[i].abs() / scale;
                eta = eta.max(e);
                zeta = zeta.max(e / self.level[i]);
            }
        }
        (eta, zeta)
    }
}

/// Why GMRES stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stop {
    /// b = 0: x = 0, exact, no iteration.
    ZeroRhs,
    /// ‖W r‖₂ did not decrease: the rounding floor.
    Stagnation,
    /// h_{k+1,k} = 0: the Krylov space is invariant.
    Breakdown,
    /// The Krylov space is the whole space (k = n).
    Exhausted,
    /// The preconditioner's application was not finite (a singular pivot of a block).
    NonFinite,
}

/// One GMRES run: the returned iterate (least ‖W r‖₂), its index, the first certified iterate, its backward errors,
/// the stop, ζ per iteration, and the complex multiply-adds spent (the prototype's account: two products by A, one
/// application of P⁻¹ without its factors' solves, the Gram–Schmidt sweep, the residual's weights).
#[derive(Clone, Debug, PartialEq)]
pub struct Outcome {
    pub x: Vec<C64>,
    pub iterations: usize,
    pub best: usize,
    pub first_certified: Option<usize>,
    pub eta: f64,
    pub zeta: f64,
    pub certified: bool,
    pub stop: Stop,
    pub history: Vec<f64>,
    pub work: f64,
}

/// GMRES on A x = b with the right preconditioner P, minimising ‖W r‖₂ (`Rows`), x0 = 0, to the floor.
pub fn gmres(a: &Sparse, b: &[C64], p: &Schur, rows: &Rows) -> Outcome {
    let n = b.len();
    let w = &rows.weight;
    let bw: Vec<C64> = b.iter().zip(w).map(|(&v, &s)| v * s).collect();
    let beta = norm(&bw);
    let mut out = Outcome {
        x: vec![C64::ZERO; n],
        iterations: 0,
        best: 0,
        first_certified: None,
        eta: 0.0,
        zeta: 0.0,
        certified: true,
        stop: Stop::ZeroRhs,
        history: Vec::new(),
        work: 0.0,
    };
    if beta == 0.0 {
        return out;
    }
    out.eta = f64::INFINITY;
    out.zeta = f64::INFINITY;
    out.certified = false;
    out.stop = Stop::Exhausted;
    let mut v: Vec<Vec<C64>> = vec![bw.iter().map(|&x| x / beta).collect()];
    let mut z: Vec<Vec<C64>> = Vec::new();
    // the Hessenberg matrix's columns after the rotations (upper triangular R), the rotations, the rotated β e₁
    let mut r_cols: Vec<Vec<C64>> = Vec::new();
    let mut rotations: Vec<(f64, C64)> = Vec::new();
    let mut g: Vec<C64> = vec![C64::real(beta)];
    let mut best_rn = f64::INFINITY;
    let mut previous = f64::INFINITY;
    let mut az = vec![C64::ZERO; n];
    let mut x = vec![C64::ZERO; n];
    let mut residual = vec![C64::ZERO; n];
    let nnz = a.indices.len() as f64;
    for k in 0..n {
        // z_k = P⁻¹ W⁻¹ v_k
        let scaled: Vec<C64> = v[k].iter().zip(w).map(|(&x, &s)| x / s).collect();
        let mut zk = vec![C64::ZERO; n];
        p.apply(&scaled, &mut zk);
        if !zk.iter().all(|c| c.is_finite()) {
            out.stop = Stop::NonFinite;
            break;
        }
        product(a, &zk, &mut az);
        let mut u: Vec<C64> = az.iter().zip(w).map(|(&x, &s)| x * s).collect();
        z.push(zk);
        // modified Gram–Schmidt
        let mut h = vec![C64::ZERO; k + 2];
        for (i, vi) in v.iter().enumerate() {
            let hi: C64 = vi.iter().zip(&u).map(|(&p, &q)| p.conj() * q).sum();
            for (uj, &vj) in u.iter_mut().zip(vi) {
                *uj -= hi * vj;
            }
            h[i] = hi;
        }
        let next = norm(&u);
        h[k + 1] = C64::real(next);
        // the previous rotations, then the one that zeroes h_{k+1}
        for (i, &(c, s)) in rotations.iter().enumerate() {
            let (p, q) = (h[i], h[i + 1]);
            h[i] = p * c + s * q;
            h[i + 1] = q * c - s.conj() * p;
        }
        let (c, s, top) = givens(h[k], next);
        h[k] = top;
        rotations.push((c, s));
        let gk = g[k];
        g[k] = gk * c;
        g.push(-(s.conj() * gk));
        r_cols.push(h[..k + 1].to_vec());
        // y = R⁻¹ g, x = Z y
        let mut y = g[..k + 1].to_vec();
        for i in (0..=k).rev() {
            let mut t = y[i];
            for j in i + 1..=k {
                t -= r_cols[j][i] * y[j];
            }
            y[i] = t / r_cols[i][i];
        }
        x.iter_mut().for_each(|e| *e = C64::ZERO);
        for (zj, &yj) in z.iter().zip(&y) {
            for (e, &q) in x.iter_mut().zip(zj) {
                *e += q * yj;
            }
        }
        product(a, &x, &mut residual);
        for (ri, &bi) in residual.iter_mut().zip(b) {
            *ri = bi - *ri;
        }
        let weighted: f64 = residual.iter().zip(w).map(|(&q, &s)| (q * s).norm_sqr()).sum::<f64>().sqrt();
        let rn = weighted / beta;
        let (eta, zeta) = rows.backward(&x, b, &residual);
        out.work += nnz + p.apply_madds + (2 * (k + 1) * n) as f64 + nnz + n as f64;
        out.iterations = k + 1;
        out.history.push(zeta);
        if rn < best_rn {
            best_rn = rn;
            out.best = k + 1;
            out.x.copy_from_slice(&x);
            out.eta = eta;
            out.zeta = zeta;
        }
        if zeta <= 1.0 && out.first_certified.is_none() {
            out.first_certified = Some(k + 1);
        }
        if next == 0.0 {
            out.stop = Stop::Breakdown;
            break;
        }
        if rn >= previous {
            out.stop = Stop::Stagnation;
            break;
        }
        previous = rn;
        v.push(u.iter().map(|&q| q / next).collect());
    }
    out.certified = out.zeta <= 1.0;
    out
}

/// ‖v‖₂.
fn norm(v: &[C64]) -> f64 {
    v.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt()
}

/// The complex Givens rotation that maps (a, b), b real ≥ 0, to (ρ, 0): (c, s, ρ) with c real,
/// [c s; −s̄ c] [a; b] = [ρ; 0], c² + |s|² = 1.
fn givens(a: C64, b: f64) -> (f64, C64, C64) {
    let size = a.abs();
    if size == 0.0 {
        return (0.0, C64::ONE, C64::real(b));
    }
    let r = crate::linalg::hypot(size, b);
    let phase = a / size;
    (size / r, phase * (b / r), phase * r)
}
