//! The static Floquet–Liouville reference solver of the two-point check (RUSTPLAN §1.1; FASTREF.md, the prototype
//! `fastref\fr.py` `fr.solve` with the zero-norm root fix of `vlat\frx.py`): the bordered steady-state system
//!
//! ```text
//!     Bd [ρ; μ] = [0; 1],    Bd = [[KK, τ*], [τᵀ, 0]],
//! ```
//!
//! solved to its rounding floor by right-preconditioned GMRES, certified by a row-wise backward error, with a chain of
//! exact fallbacks. The module takes `floquet::System` as given (the matrix, each unknown's harmonic); the readouts
//! are `floquet::readouts`. Only ρ is solved (`solve`).
//!
//! # The method (FASTREF.md §§ 2, 3, 5, 8)
//! 1. **Split** (`split`): slow = the unknowns whose Gershgorin disc lies in the connected component of the disc union
//!    that contains 0, fast = the rest. By the separation form of Gershgorin's theorem (Varga 2004, Thm 1.6) that
//!    component holds exactly as many eigenvalues as unknowns: a gap of the spectrum read off the matrix, no label and
//!    no number.
//! 2. **Preconditioner** (`precond`): P = A with the fast-fast block replaced by its diagonal D_f, applied through the
//!    slow Schur complement S = A_ss − A_sf D_f⁻¹ A_fs; S itself replaced by its harmonic-diagonal blocks S_jj, each
//!    factorised exactly (sparse LU). A − P = diag(S_cross, O_f): the error is the cross-harmonic coupling against a
//!    harmonic's own gap and the fast-fast coupling against the fast gap (measured ρ ≤ 0.28 and ≤ 9e-3).
//! 3. **GMRES** (`gmres`): right preconditioned, x0 = 0, minimising the row-equilibrated residual ‖W r‖₂,
//!    W = diag(1/a_i), a_i = Σ_j |A_ij|; it runs while ‖W r‖₂ decreases (monotone in exact arithmetic) and returns the
//!    iterate of least ‖W r‖₂: the rounding floor, no target number.
//! 4. **Certificate**: ζ = max_i |r_i| / (γ_i (a_i ‖x‖∞ + |b_i|)) ≤ 1, γ_i the rounding level of row i's own residual
//!    evaluation (Higham 2002, §3.1 Lemma 3.1, §12.1).
//! 5. **Route** (`solve`): not certified → the same GMRES with the exact S (one block; skipped when S has no
//!    cross-harmonic entry, then it is the same operator) → the direct LU of the whole matrix (one block, every
//!    unknown slow: P = A). Every attempt is reported. The split and the blocks change P only, never the system, the
//!    stop or the certificate: every route returns the solution of the same equations to the same backward error.
//!
//! # The zero-norm root fix (VLAT.md §3, `frx.py`): every norm used as a divisor can be exactly zero
//! - β = ‖W b‖ = 0: x = 0 solves A x = b exactly; returned at once, certified (ζ = 0), no iteration.
//! - h_{k+1,k} = 0 (Arnoldi breakdown): the Krylov space is invariant, this step's least-squares iterate is exact in
//!   exact arithmetic; the iteration ends there.
//! - a_i = 0 (a zero row): the row reads 0 = b_i. b_i = 0: no information, its residual is identically 0, weight 1.
//!   b_i ≠ 0: the system is inconsistent, reported (`Inconsistent`), never solved around.
//! - A singular pivot of a preconditioner block (a block may be singular where S or A is not): its application is not
//!   finite; the attempt stops uncertified and the route goes on.

pub mod gmres;
pub mod precond;
pub mod solve;
pub mod split;

pub use gmres::{gmres, Outcome, Rows, Stop};
pub use precond::{Schur, Unfactorised};
pub use solve::{solve, Attempt, Route, Solved};
pub use split::split;

use crate::floquet::Sparse;
use crate::linalg::C64;
use rayon::prelude::*;

/// Row i of a compressed-row matrix: its columns and values.
pub fn row(a: &Sparse, i: usize) -> (&[usize], &[C64]) {
    let r = a.indptr[i]..a.indptr[i + 1];
    (&a.indices[r.clone()], &a.values[r])
}

/// The diagonal of a square matrix (0 where not stored).
pub fn diagonal(a: &Sparse) -> Vec<C64> {
    (0..a.rows)
        .map(|i| {
            let (c, v) = row(a, i);
            c.iter().zip(v).filter(|(&j, _)| j == i).map(|(_, &x)| x).sum()
        })
        .collect()
}

/// y = A x, rows in parallel (each row's sum in stored order: the result does not depend on the thread count).
pub fn product(a: &Sparse, x: &[C64], y: &mut [C64]) {
    y.par_iter_mut().enumerate().for_each(|(i, yi)| {
        let (c, v) = row(a, i);
        *yi = c.iter().zip(v).map(|(&j, &e)| e * x[j]).sum();
    });
}

/// A zero row of the matrix with a nonzero right-hand side: 0 = b_i has no solution (the rows).
#[derive(Clone, Debug, PartialEq)]
pub struct Inconsistent {
    pub rows: Vec<usize>,
}

#[cfg(test)]
mod tests;
