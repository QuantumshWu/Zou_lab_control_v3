//! The static Floquet–Liouville reference solver of the two-point check (FASTREF.md, the prototype `fastref\fr.py`
//! `fr.solve` with the zero-norm root fix of `vlat\frx.py`): the bordered steady-state system
//!
//! ```text
//!     Bd [ρ; μ] = [0; 1],    Bd = [[KK, τ*], [τᵀ, 0]]
//! ```
//!
//! solved to its rounding floor by right-preconditioned GMRES, certified by a row-wise backward error, with a chain
//! of exact fallbacks. The module takes the system as given (sparse matrix, right-hand side, each unknown's harmonic,
//! the readout rows); building it is the lattice's business.
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

mod gmres;
mod precond;
mod solve;
mod split;

pub use gmres::{gmres, Outcome, Rows, Stop};
pub use precond::{Schur, Unfactorised};
pub use solve::{solve, Attempt, Level, Solved};
pub use split::split;

use crate::linalg::C64;
use rayon::prelude::*;

/// A square sparse complex matrix in compressed rows: row i's entries are `column[start[i]..start[i + 1]]` and
/// `value[...]`, any order within a row.
#[derive(Clone, Debug)]
pub struct Csr {
    pub start: Vec<usize>,
    pub column: Vec<usize>,
    pub value: Vec<C64>,
}

impl Csr {
    pub fn dim(&self) -> usize {
        self.start.len() - 1
    }
    pub fn nnz(&self) -> usize {
        self.column.len()
    }
    pub fn row(&self, i: usize) -> (&[usize], &[C64]) {
        let r = self.start[i]..self.start[i + 1];
        (&self.column[r.clone()], &self.value[r])
    }
    /// The diagonal (0 where not stored).
    pub fn diagonal(&self) -> Vec<C64> {
        (0..self.dim())
            .map(|i| {
                let (c, v) = self.row(i);
                c.iter().zip(v).filter(|(&j, _)| j == i).map(|(_, &x)| x).sum()
            })
            .collect()
    }
    /// y = A x, rows in parallel (each row's sum in stored order: the result does not depend on the thread count).
    pub fn apply(&self, x: &[C64], y: &mut [C64]) {
        y.par_iter_mut().enumerate().for_each(|(i, yi)| {
            let (c, v) = self.row(i);
            *yi = c.iter().zip(v).map(|(&j, &a)| a * x[j]).sum();
        });
    }
}

/// A sparse row vector (a readout's linear form over the unknowns).
#[derive(Clone, Debug, Default)]
pub struct Row {
    pub index: Vec<usize>,
    pub value: Vec<C64>,
}

impl Row {
    /// Re(row · x).
    pub fn read(&self, x: &[C64]) -> f64 {
        self.index.iter().zip(&self.value).map(|(&i, &v)| v * x[i]).sum::<C64>().re
    }
}

/// The readouts as linear forms of the solution: the force F = Re(f·x) (τᵀK1ρ), the scattering rate R = Re(r·x) (the
/// rate channels' sum), and per level group the lifted, unnormalised population Re(p_g·x); the groups partition the
/// states, so the populations are normalised by their own sum.
#[derive(Clone, Debug, Default)]
pub struct Readouts {
    pub force: Row,
    pub rate: Row,
    pub populations: Vec<(String, Row)>,
}

impl Readouts {
    /// The readouts of x (populations normalised by their sum; left as they are when the sum is 0 -- x = 0, the
    /// zero-norm class: no sum to divide by).
    pub fn read(&self, x: &[C64]) -> Readings {
        let raw: Vec<(String, f64)> = self.populations.iter().map(|(name, row)| (name.clone(), row.read(x))).collect();
        let total: f64 = raw.iter().map(|(_, v)| v).sum();
        let populations = raw.into_iter().map(|(name, v)| (name, if total == 0.0 { v } else { v / total })).collect();
        Readings { force: self.force.read(x), rate: self.rate.read(x), populations }
    }
}

/// The readouts' values: F, R, the populations by group.
#[derive(Clone, Debug, Default)]
pub struct Readings {
    pub force: f64,
    pub rate: f64,
    pub populations: Vec<(String, f64)>,
}

/// The bordered steady-state system: the matrix (the last unknown is the border's μ), the right-hand side, every
/// unknown's harmonic (the slow-lattice index j of its element key, any integer labelling; the border row goes with
/// the populations it sums), and the readouts.
#[derive(Clone, Debug)]
pub struct System {
    pub matrix: Csr,
    pub rhs: Vec<C64>,
    pub harmonic: Vec<usize>,
    pub readouts: Readouts,
}

/// A zero row of the matrix with a nonzero right-hand side: 0 = b_i has no solution (the rows, at most a few).
#[derive(Clone, Debug, PartialEq)]
pub struct Inconsistent {
    pub rows: Vec<usize>,
}

#[cfg(test)]
mod tests;
