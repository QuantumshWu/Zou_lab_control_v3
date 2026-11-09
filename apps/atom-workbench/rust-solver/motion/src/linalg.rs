//! Complex numbers and small dense complex matrices (n ≤ 32), what the model and the integrator need (SPEC §2.1):
//! products and the adjoint, the LU solve, the Hermitian eigendecomposition, the general (non-Hermitian)
//! eigendecomposition with its condition, the matrix exponential and its action on a vector to round-off, and 3 × 3 real
//! helpers. One implementation per operation; a real symmetric matrix is a Hermitian one.
//!
//! # To round-off
//! Every iteration here runs until the floating-point format, not a tolerance, ends it: its cost is logarithmic in the
//! accuracy (SPEC §2.1), so there is no tolerance to choose. "At round-off" means backward stable: the result is the
//! exact one of data perturbed relatively by a modest multiple of the unit round-off u (`units::UNIT_ROUNDOFF`); its
//! forward error is then the problem's own condition times that.
//!
//! # The Taylor plan of the exponential (`Taylor`)
//! e^X for ‖X‖₁ ≤ x is summed as Taylor polynomials of degree m on q parts Y = X/q of norm y = x/q: sub-steps on a
//! vector, or q = 2^s and s squarings for the matrix. Per part, on a vector of norm one:
//! - the truncation is at most R_m(y) = Σ_{k>m} y^k/k! ≤ (y^{m+1}/(m+1)!)/(1 − y/(m+2));
//! - the rounding of the sum is at most γ Σ_k ‖Y^k v‖/k! ≤ γ e^y (γ the products' and the sum's γ_{n+m}, Higham §3.1).
//! Over the q = x/y parts the rounding adds up to (x/y) γ e^y: per unit of ‖X‖ it is γ e^y/y, least at y* = 1
//! (d/dy (e^y/y) = e^y (y − 1)/y²). So the parts are the fewest of norm ≤ 1, q = ⌈x⌉ on a vector, 2^⌈log₂ x⌉ for the
//! matrix, and each part's degree the least m whose truncation R_m(y) is within the unit round-off u: at y = 1, m = 18.
//! Both follow from the floating-point format alone; the result's error is then a modest multiple of u‖X‖, the
//! problem's own sensitivity to the rounding of X (backward stable). The squarings keep it so for a contraction
//! (‖e^{tX}‖ ≤ 1 for t ≥ 0, as e^{−iH_eff t} with H_eff's anti-Hermitian part negative semidefinite): no hump.
//! The vector route costs q·m matrix-vector products (n² each), the matrix route (m − 1 + s) products (n³ each) and
//! one application; the caller picks by its cost.

use crate::units::UNIT_ROUNDOFF;
use std::iter::Sum;
use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};

// ---------------------------------------------------------------------------------------------------------------------
// Complex numbers.

/// A complex number.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct C64 {
    pub re: f64,
    pub im: f64,
}

impl C64 {
    pub const ZERO: C64 = C64 { re: 0.0, im: 0.0 };
    pub const ONE: C64 = C64 { re: 1.0, im: 0.0 };
    pub const I: C64 = C64 { re: 0.0, im: 1.0 };
    pub const fn new(re: f64, im: f64) -> C64 {
        C64 { re, im }
    }
    pub const fn real(re: f64) -> C64 {
        C64 { re, im: 0.0 }
    }
    pub fn conj(self) -> C64 {
        C64 { re: self.re, im: -self.im }
    }
    /// |z|².
    pub fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }
    /// |z|, without overflow or underflow of the square.
    pub fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
    /// arg z in (−π, π].
    pub fn arg(self) -> f64 {
        self.im.atan2(self.re)
    }
    /// e^{iφ}.
    pub fn cis(phase: f64) -> C64 {
        let (s, c) = phase.sin_cos();
        C64 { re: c, im: s }
    }
    /// e^z.
    pub fn exp(self) -> C64 {
        C64::cis(self.im) * self.re.exp()
    }
    /// √z on the principal branch (Re √z ≥ 0), without cancellation.
    pub fn sqrt(self) -> C64 {
        let r = self.abs();
        if r == 0.0 {
            return C64::ZERO;
        }
        let re = ((r + self.re) * 0.5).sqrt();
        let im = ((r - self.re) * 0.5).sqrt();
        C64 { re, im: if self.im < 0.0 { -im } else { im } }
    }
    /// 1/z.
    pub fn recip(self) -> C64 {
        C64::ONE / self
    }
    /// |re| + |im|: a norm within √2 of |z|, the one the matrix norms here bound with.
    pub fn abs1(self) -> f64 {
        self.re.abs() + self.im.abs()
    }
    pub fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
}

impl Add for C64 {
    type Output = C64;
    fn add(self, b: C64) -> C64 {
        C64 { re: self.re + b.re, im: self.im + b.im }
    }
}
impl Sub for C64 {
    type Output = C64;
    fn sub(self, b: C64) -> C64 {
        C64 { re: self.re - b.re, im: self.im - b.im }
    }
}
impl Mul for C64 {
    type Output = C64;
    fn mul(self, b: C64) -> C64 {
        C64 { re: self.re * b.re - self.im * b.im, im: self.re * b.im + self.im * b.re }
    }
}
impl Mul<f64> for C64 {
    type Output = C64;
    fn mul(self, s: f64) -> C64 {
        C64 { re: self.re * s, im: self.im * s }
    }
}
impl Div for C64 {
    /// Smith's division: no overflow of |b|² for large b.
    type Output = C64;
    fn div(self, b: C64) -> C64 {
        if b.re.abs() >= b.im.abs() {
            let (r, d) = (b.im / b.re, b.re + b.im * (b.im / b.re));
            C64 { re: (self.re + self.im * r) / d, im: (self.im - self.re * r) / d }
        } else {
            let (r, d) = (b.re / b.im, b.re * (b.re / b.im) + b.im);
            C64 { re: (self.re * r + self.im) / d, im: (self.im * r - self.re) / d }
        }
    }
}
impl Div<f64> for C64 {
    type Output = C64;
    fn div(self, s: f64) -> C64 {
        C64 { re: self.re / s, im: self.im / s }
    }
}
impl Neg for C64 {
    type Output = C64;
    fn neg(self) -> C64 {
        C64 { re: -self.re, im: -self.im }
    }
}
impl AddAssign for C64 {
    fn add_assign(&mut self, b: C64) {
        *self = *self + b;
    }
}
impl SubAssign for C64 {
    fn sub_assign(&mut self, b: C64) {
        *self = *self - b;
    }
}
impl MulAssign for C64 {
    fn mul_assign(&mut self, b: C64) {
        *self = *self * b;
    }
}
impl MulAssign<f64> for C64 {
    fn mul_assign(&mut self, s: f64) {
        *self = *self * s;
    }
}
impl DivAssign for C64 {
    fn div_assign(&mut self, b: C64) {
        *self = *self / b;
    }
}
impl Sum for C64 {
    fn sum<I: Iterator<Item = C64>>(iter: I) -> C64 {
        iter.fold(C64::ZERO, |a, b| a + b)
    }
}

/// ⟨a, b⟩ = Σ_i a_i* b_i.
pub fn inner(a: &[C64], b: &[C64]) -> C64 {
    a.iter().zip(b).map(|(x, y)| x.conj() * *y).sum()
}
/// ‖v‖² = Σ_i |v_i|².
pub fn norm_sqr(v: &[C64]) -> f64 {
    v.iter().map(|z| z.norm_sqr()).sum()
}

// ---------------------------------------------------------------------------------------------------------------------
// Dense matrices.

/// A dense complex matrix, row-major: entry (i, j) at `data[i * cols + j]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<C64>,
}

impl Index<(usize, usize)> for Mat {
    type Output = C64;
    fn index(&self, (i, j): (usize, usize)) -> &C64 {
        &self.data[i * self.cols + j]
    }
}
impl IndexMut<(usize, usize)> for Mat {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut C64 {
        &mut self.data[i * self.cols + j]
    }
}

impl Mat {
    pub fn zeros(rows: usize, cols: usize) -> Mat {
        Mat { rows, cols, data: vec![C64::ZERO; rows * cols] }
    }
    pub fn identity(n: usize) -> Mat {
        Mat::from_fn(n, n, |i, j| if i == j { C64::ONE } else { C64::ZERO })
    }
    pub fn from_fn(rows: usize, cols: usize, mut entry: impl FnMut(usize, usize) -> C64) -> Mat {
        Mat { rows, cols, data: (0..rows * cols).map(|k| entry(k / cols, k % cols)).collect() }
    }
    /// diag(d).
    pub fn diagonal(d: &[C64]) -> Mat {
        Mat::from_fn(d.len(), d.len(), |i, j| if i == j { d[i] } else { C64::ZERO })
    }
    /// The rows × cols block at the given row and column indices.
    pub fn block(&self, rows: &[usize], cols: &[usize]) -> Mat {
        Mat::from_fn(rows.len(), cols.len(), |i, j| self[(rows[i], cols[j])])
    }
    /// A†.
    pub fn adjoint(&self) -> Mat {
        Mat::from_fn(self.cols, self.rows, |i, j| self[(j, i)].conj())
    }
    /// A B (rows × cols · cols × b.cols complex multiply-adds).
    pub fn product(&self, b: &Mat) -> Mat {
        assert_eq!(self.cols, b.rows, "product of {}x{} and {}x{}", self.rows, self.cols, b.rows, b.cols);
        let mut out = Mat::zeros(self.rows, b.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let x = self[(i, k)];
                for j in 0..b.cols {
                    out.data[i * b.cols + j] += x * b.data[k * b.cols + j];
                }
            }
        }
        out
    }
    /// A† B, without forming A†.
    pub fn adjoint_product(&self, b: &Mat) -> Mat {
        assert_eq!(self.rows, b.rows, "adjoint product of {}x{} and {}x{}", self.rows, self.cols, b.rows, b.cols);
        let mut out = Mat::zeros(self.cols, b.cols);
        for k in 0..self.rows {
            for i in 0..self.cols {
                let x = self[(k, i)].conj();
                for j in 0..b.cols {
                    out.data[i * b.cols + j] += x * b.data[k * b.cols + j];
                }
            }
        }
        out
    }
    /// A v.
    pub fn apply(&self, v: &[C64]) -> Vec<C64> {
        assert_eq!(self.cols, v.len());
        (0..self.rows).map(|i| self.data[i * self.cols..(i + 1) * self.cols].iter().zip(v).map(|(a, x)| *a * *x).sum()).collect()
    }
    /// z A.
    pub fn scaled(&self, z: C64) -> Mat {
        Mat { rows: self.rows, cols: self.cols, data: self.data.iter().map(|x| z * *x).collect() }
    }
    /// A ← A + z B.
    pub fn add_scaled(&mut self, z: C64, b: &Mat) {
        assert_eq!((self.rows, self.cols), (b.rows, b.cols));
        for (x, y) in self.data.iter_mut().zip(&b.data) {
            *x += z * *y;
        }
    }
    pub fn trace(&self) -> C64 {
        (0..self.rows.min(self.cols)).map(|i| self[(i, i)]).sum()
    }
    /// ‖A‖₁ bounded above: the largest column sum of |re| + |im| (within √2 of the 1-norm, above it).
    pub fn norm1(&self) -> f64 {
        (0..self.cols).map(|j| (0..self.rows).map(|i| self[(i, j)].abs1()).sum::<f64>()).fold(0.0, f64::max)
    }
    /// The Frobenius norm √Σ|a_ij|².
    pub fn frobenius(&self) -> f64 {
        norm_sqr(&self.data).sqrt()
    }
    /// The largest entry's modulus.
    pub fn max_abs(&self) -> f64 {
        self.data.iter().fold(0.0, |a, z| a.max(z.abs()))
    }
    /// (A + A†)/2.
    pub fn hermitian_part(&self) -> Mat {
        Mat::from_fn(self.rows, self.cols, |i, j| (self[(i, j)] + self[(j, i)].conj()) * 0.5)
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// LU.

/// The LU factorization with partial pivoting of a square matrix, P A = L U (L unit lower, U upper, in one array).
/// Backward stable: the computed factors are those of A + ΔA, |ΔA| ≤ γ_n |L||U| (Higham, Theorem 9.3).
/// Cost n³/3 complex multiply-adds; a solve n².
#[derive(Clone, Debug)]
pub struct Lu {
    factors: Mat,
    pivots: Vec<usize>,
}

impl Lu {
    /// The factorization; None when a pivot column is exactly zero, so that no factorization exists. A nearly
    /// singular A factors, its solutions large: its condition is the caller's to read (`Lu::inverse`).
    pub fn new(a: &Mat) -> Option<Lu> {
        let n = a.rows;
        assert_eq!(n, a.cols, "LU of a {}x{} matrix", a.rows, a.cols);
        let mut f = a.clone();
        let mut pivots = vec![0; n];
        for k in 0..n {
            let p = (k..n).max_by(|&i, &j| f[(i, k)].abs1().total_cmp(&f[(j, k)].abs1())).unwrap();
            if f[(p, k)] == C64::ZERO {
                return None;
            }
            pivots[k] = p;
            if p != k {
                for j in 0..n {
                    f.data.swap(k * n + j, p * n + j);
                }
            }
            let pivot = f[(k, k)];
            for i in k + 1..n {
                let l = f[(i, k)] / pivot;
                f[(i, k)] = l;
                for j in k + 1..n {
                    let u = f.data[k * n + j];
                    f.data[i * n + j] -= l * u;
                }
            }
        }
        Some(Lu { factors: f, pivots })
    }
    /// b ← A⁻¹ b.
    pub fn solve(&self, b: &mut [C64]) {
        let (f, n) = (&self.factors, self.factors.rows);
        for (k, &p) in self.pivots.iter().enumerate() {
            b.swap(k, p);
        }
        for i in 0..n {
            let mut x = b[i];
            for j in 0..i {
                x -= f.data[i * n + j] * b[j];
            }
            b[i] = x;
        }
        for i in (0..n).rev() {
            let mut x = b[i];
            for j in i + 1..n {
                x -= f.data[i * n + j] * b[j];
            }
            b[i] = x / f.data[i * n + i];
        }
    }
    /// A⁻¹ B, column by column.
    pub fn solve_columns(&self, b: &Mat) -> Mat {
        let mut out = b.clone();
        let mut column = vec![C64::ZERO; b.rows];
        for j in 0..b.cols {
            for i in 0..b.rows {
                column[i] = b[(i, j)];
            }
            self.solve(&mut column);
            for i in 0..b.rows {
                out[(i, j)] = column[i];
            }
        }
        out
    }
    /// A⁻¹.
    pub fn inverse(&self) -> Mat {
        self.solve_columns(&Mat::identity(self.factors.rows))
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Eigendecompositions.

/// H = U diag(λ) U† of a Hermitian matrix: the eigenvalues ascending, U unitary (its columns orthonormal to round-off,
/// in a degenerate eigenspace too).
#[derive(Clone, Debug)]
pub struct Hermitian {
    pub values: Vec<f64>,
    pub vectors: Mat,
}

/// The eigendecomposition of the Hermitian part of h, (h + h†)/2, by cyclic complex Jacobi rotations: each rotation
/// unitary up to its own rounding, so U stays orthonormal to round-off, in a degenerate eigenspace too. The sweeps run
/// until the off-diagonal part is within the unit round-off of the matrix, ‖off(A)‖_F ≤ u‖A‖_F, or a sweep no longer
/// shrinks it (the floating-point format's floor; the convergence is quadratic). Each eigenvalue is then within
/// ‖off(A)‖₂ ≤ u‖A‖_F of the exact one (Weyl). Cost about 6n³ complex multiply-adds per sweep, some 6-10 sweeps.
pub fn hermitian_eigen(h: &Mat) -> Hermitian {
    let n = h.rows;
    assert_eq!(n, h.cols, "the eigendecomposition of a {}x{} matrix", h.rows, h.cols);
    let mut a = h.hermitian_part();
    let mut u = Mat::identity(n);
    let floor = UNIT_ROUNDOFF * UNIT_ROUNDOFF * norm_sqr(&a.data);
    let off = |a: &Mat| -> f64 { (0..n).map(|i| (0..n).filter(|&j| j != i).map(|j| a[(i, j)].norm_sqr()).sum::<f64>()).sum() };
    let mut previous = f64::INFINITY;
    loop {
        let left = off(&a);
        if left <= floor || !(left < previous) {
            break;
        }
        previous = left;
        for p in 0..n {
            for q in p + 1..n {
                let modulus = a[(p, q)].abs();
                if modulus == 0.0 {
                    continue;
                }
                // A phase on q makes a_pq real and positive (dividing by the modulus: its reciprocal would overflow
                // when it is subnormal); a real rotation in (p, q) then zeroes it.
                let phase = a[(p, q)] / modulus;
                for k in 0..n {
                    a[(k, q)] = a[(k, q)] * phase.conj();
                    u[(k, q)] = u[(k, q)] * phase.conj();
                }
                for k in 0..n {
                    a[(q, k)] = a[(q, k)] * phase;
                }
                // tan θ, the smaller root of t² + 2τt − 1 = 0 (Rutishauser): |θ| ≤ π/4.
                let tau = (a[(q, q)].re - a[(p, p)].re) / (2.0 * modulus);
                let t = tau.signum() / (tau.abs() + (1.0 + tau * tau).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (x, y) = (a[(k, p)], a[(k, q)]);
                    a[(k, p)] = x * c - y * s;
                    a[(k, q)] = x * s + y * c;
                    let (x, y) = (u[(k, p)], u[(k, q)]);
                    u[(k, p)] = x * c - y * s;
                    u[(k, q)] = x * s + y * c;
                }
                for k in 0..n {
                    let (x, y) = (a[(p, k)], a[(q, k)]);
                    a[(p, k)] = x * c - y * s;
                    a[(q, k)] = x * s + y * c;
                }
                a[(p, q)] = C64::ZERO;
                a[(q, p)] = C64::ZERO;
            }
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| a[(i, i)].re.total_cmp(&a[(j, j)].re));
    Hermitian { values: order.iter().map(|&i| a[(i, i)].re).collect(), vectors: Mat::from_fn(n, n, |i, k| u[(i, order[k])]) }
}

/// A = R diag(λ) R⁻¹ of a general square matrix: the eigenvalues (in the complex Schur form's order), R with columns
/// of unit 2-norm, R⁻¹, and the condition κ(R) = ‖R‖₁‖R⁻¹‖₁, which bounds what is computed through the
/// decomposition: e^{tA}v through it errs by about κ(R)·u·(1 + |t|·(max Re λ − min Re λ)) relative to ‖v‖ (`apply_exp`),
/// and κ grows without bound towards an exceptional point.
#[derive(Clone, Debug)]
pub struct Eigen {
    pub values: Vec<C64>,
    pub vectors: Mat,
    pub inverse: Mat,
    pub condition: f64,
}

/// The Givens rotation G = [[c, s], [−s*, c]] (c real) with G [x, y]ᵀ = [r, 0]ᵀ.
fn givens(x: C64, y: C64) -> (f64, C64) {
    let (ax, ay) = (x.abs(), y.abs());
    if ay == 0.0 {
        return (1.0, C64::ZERO);
    }
    if ax == 0.0 {
        return (0.0, y.conj() / ay);
    }
    let norm = ax.hypot(ay);
    (ax / norm, (x / ax) * y.conj() / norm)
}

/// The 1-norm, the largest column sum of |a_ij|.
fn one_norm(a: &Mat) -> f64 {
    (0..a.cols).map(|j| (0..a.rows).map(|i| a[(i, j)].abs()).sum::<f64>()).fold(0.0, f64::max)
}

/// The eigendecomposition of a general complex matrix: Householder reduction to Hessenberg form; the single-shift
/// implicit QR iteration to the complex Schur form T = Q†AQ, with Wilkinson shifts (an exceptional shift every tenth
/// iteration) and deflation where a subdiagonal entry is within the precision of its two diagonal neighbours (LAPACK
/// zlahqr's criterion); the eigenvectors of T by back substitution (a difference of eigenvalues below the precision of
/// T's scale taken at it, LAPACK ztrevc's rule); R = QY with unit columns, and R⁻¹ by LU. None when the QR iteration
/// does not converge within LAPACK zlahqr's 30·max(10, n) iterations per eigenvalue, or R is singular.
pub fn eigen(a: &Mat) -> Option<Eigen> {
    let n = a.rows;
    assert_eq!(n, a.cols, "the eigendecomposition of a {}x{} matrix", a.rows, a.cols);
    let mut h = a.clone();
    let mut q = Mat::identity(n);
    // Householder reduction: H ← P H P, Q ← Q P, P = I − β v v† zeroing column k below its subdiagonal.
    let mut v = vec![C64::ZERO; n];
    for k in 0..n.saturating_sub(2) {
        let alpha = (k + 1..n).map(|i| h[(i, k)].norm_sqr()).sum::<f64>().sqrt();
        if alpha == 0.0 {
            continue;
        }
        let x0 = h[(k + 1, k)];
        let phase = if x0 == C64::ZERO { C64::ONE } else { x0 / x0.abs() };
        for i in 0..n {
            v[i] = if i > k { h[(i, k)] } else { C64::ZERO };
        }
        v[k + 1] += phase * alpha;
        let beta = 2.0 / (k + 1..n).map(|i| v[i].norm_sqr()).sum::<f64>();
        for j in 0..n {
            let s = (k + 1..n).map(|i| v[i].conj() * h[(i, j)]).sum::<C64>() * beta;
            for i in k + 1..n {
                h[(i, j)] -= v[i] * s;
            }
        }
        for m in [&mut h, &mut q] {
            for i in 0..n {
                let s = (k + 1..n).map(|j| m[(i, j)] * v[j]).sum::<C64>() * beta;
                for j in k + 1..n {
                    m[(i, j)] -= s * v[j].conj();
                }
            }
        }
        for i in k + 2..n {
            h[(i, k)] = C64::ZERO;
        }
    }
    // The QR iteration on the active block [lo, hi), the rotations accumulated into Q.
    let scale = h.data.iter().map(|z| z.abs1()).fold(0.0, f64::max).max(f64::MIN_POSITIVE);
    let most = 30 * n.max(10);
    let (mut hi, mut iterations) = (n, 0);
    while hi > 1 {
        let mut lo = hi - 1;
        while lo > 0 {
            let s = h[(lo - 1, lo - 1)].abs1() + h[(lo, lo)].abs1();
            if h[(lo, lo - 1)].abs1() <= f64::EPSILON * if s > 0.0 { s } else { scale } {
                h[(lo, lo - 1)] = C64::ZERO;
                break;
            }
            lo -= 1;
        }
        if lo == hi - 1 {
            hi -= 1;
            iterations = 0;
            continue;
        }
        iterations += 1;
        if iterations > most {
            return None;
        }
        // The Wilkinson shift: the eigenvalue of the trailing 2 × 2 block nearer its corner.
        let (p, pq, qp, s) = (h[(hi - 2, hi - 2)], h[(hi - 2, hi - 1)], h[(hi - 1, hi - 2)], h[(hi - 1, hi - 1)]);
        let mut mu = if iterations % 10 == 0 {
            s + C64::real(qp.abs1())
        } else {
            let half = (p - s) * 0.5;
            let d = (half * half + pq * qp).sqrt();
            let (m1, m2) = (s + half - d, s + half + d);
            if (m1 - s).abs1() <= (m2 - s).abs1() {
                m1
            } else {
                m2
            }
        };
        if !mu.is_finite() {
            mu = s;
        }
        // The bulge chased down the block: rows by G, columns by G†.
        for k in lo..hi - 1 {
            let (x, y) = if k == lo { (h[(k, k)] - mu, h[(k + 1, k)]) } else { (h[(k, k - 1)], h[(k + 1, k - 1)]) };
            let (c, sn) = givens(x, y);
            for j in if k == lo { lo } else { k - 1 }..n {
                let (u, w) = (h[(k, j)], h[(k + 1, j)]);
                h[(k, j)] = u * c + w * sn;
                h[(k + 1, j)] = w * c - u * sn.conj();
            }
            for i in 0..(k + 3).min(hi) {
                let (u, w) = (h[(i, k)], h[(i, k + 1)]);
                h[(i, k)] = u * c + w * sn.conj();
                h[(i, k + 1)] = w * c - u * sn;
            }
            for i in 0..n {
                let (u, w) = (q[(i, k)], q[(i, k + 1)]);
                q[(i, k)] = u * c + w * sn.conj();
                q[(i, k + 1)] = w * c - u * sn;
            }
            if k > lo {
                h[(k + 1, k - 1)] = C64::ZERO;
            }
        }
    }
    // T's eigenvectors: y_kk = 1, upwards by back substitution.
    let values: Vec<C64> = (0..n).map(|i| h[(i, i)]).collect();
    let floor = f64::EPSILON * scale;
    let mut y = Mat::zeros(n, n);
    for k in 0..n {
        y[(k, k)] = C64::ONE;
        for j in (0..k).rev() {
            let s: C64 = (j + 1..=k).map(|m| h[(j, m)] * y[(m, k)]).sum();
            let mut d = h[(j, j)] - values[k];
            if d.abs1() < floor {
                d = C64::real(floor);
            }
            y[(j, k)] = -(s / d);
        }
    }
    let mut r = q.product(&y);
    for k in 0..n {
        let norm = (0..n).map(|i| r[(i, k)].norm_sqr()).sum::<f64>().sqrt();
        if !(norm > 0.0 && norm.is_finite()) {
            return None;
        }
        for i in 0..n {
            r[(i, k)] = r[(i, k)] / norm;
        }
    }
    let inverse = Lu::new(&r)?.inverse();
    let condition = one_norm(&r) * one_norm(&inverse);
    condition.is_finite().then_some(Eigen { values, vectors: r, inverse, condition })
}

impl Eigen {
    /// v ← R e^{tΛ} R⁻¹ v: e^{tA} v through the decomposition, exact at any t up to the rounding `Eigen` states.
    pub fn apply_exp(&self, t: C64, v: &mut [C64]) {
        let mut w = self.inverse.apply(v);
        for (x, &l) in w.iter_mut().zip(&self.values) {
            *x = *x * (l * t).exp();
        }
        v.copy_from_slice(&self.vectors.apply(&w));
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// The exponential.

/// The Taylor plan of e^X for ‖X‖₁ ≤ x (module documentation, # The Taylor plan): `parts` of norm ≤ x/parts ≤ 1, each
/// summed through degree `terms`. Its work: parts × terms matrix-vector products (n² each), or for the matrix
/// exponential terms − 1 + log₂ parts matrix products (n³ each).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Taylor {
    pub parts: usize,
    pub terms: usize,
}

/// The widest part y* = 1: where the rounding per unit of norm, γ e^y/y, is least (module documentation).
const WIDEST: f64 = 1.0;

impl Taylor {
    /// The least degree m whose truncation R_m(y) ≤ (y^{m+1}/(m+1)!)/(1 − y/(m+2)) is within the unit round-off,
    /// for y ≤ `WIDEST` (so y < m + 2).
    fn degree(y: f64) -> usize {
        let mut m = 0;
        let mut bound = y;
        while !(bound / (1.0 - y / (m as f64 + 2.0)) <= UNIT_ROUNDOFF) {
            m += 1;
            bound *= y / (m as f64 + 1.0);
        }
        m
    }
    /// The plan on a vector: the fewest parts of norm within y*, ⌈x/y*⌉, and the least degree for their norm.
    pub fn steps(x: f64) -> Taylor {
        assert!(x.is_finite() && x >= 0.0, "the exponential of a matrix of norm {x}");
        let parts = ((x / WIDEST).ceil() as usize).max(1);
        Taylor { parts, terms: Taylor::degree(x / parts as f64) }
    }
    /// The plan of the matrix exponential: the parts the fewest power of two whose norm is within y* (its squarings
    /// log₂ parts), and the least degree for their norm.
    pub fn squarings(x: f64) -> Taylor {
        assert!(x.is_finite() && x >= 0.0, "the exponential of a matrix of norm {x}");
        let mut parts = 1;
        while (parts as f64) * WIDEST < x {
            parts *= 2;
        }
        Taylor { parts, terms: Taylor::degree(x / parts as f64) }
    }
}

/// e^{tA} to round-off: the Taylor polynomial of tA/2^s (`Taylor::squarings` of ‖tA‖₁) squared s times.
pub fn exp(a: &Mat, t: C64) -> Mat {
    let n = a.rows;
    assert_eq!(n, a.cols, "the exponential of a {}x{} matrix", a.rows, a.cols);
    let plan = Taylor::squarings(a.norm1() * t.abs());
    let mut sum = Mat::identity(n);
    if plan.terms > 0 {
        let x = a.scaled(t / plan.parts as f64);
        let mut term = x.clone();
        sum.add_scaled(C64::ONE, &x);
        for k in 2..=plan.terms {
            term = term.product(&x);
            for z in &mut term.data {
                *z *= 1.0 / k as f64;
            }
            sum.add_scaled(C64::ONE, &term);
        }
    }
    let mut parts = plan.parts;
    while parts > 1 {
        sum = sum.product(&sum);
        parts /= 2;
    }
    sum
}

/// v ← e^{tA} v to round-off: the Taylor series on the vector in the parts of `Taylor::steps` of ‖tA‖₁; no matrix is
/// formed.
pub fn exp_apply(a: &Mat, t: C64, v: &mut [C64]) {
    let n = v.len();
    assert_eq!((a.rows, a.cols), (n, n), "the exponential of a {}x{} matrix on a vector of {n}", a.rows, a.cols);
    let plan = Taylor::steps(a.norm1() * t.abs());
    let step = t / plan.parts as f64;
    let (mut term, mut next) = (vec![C64::ZERO; n], vec![C64::ZERO; n]);
    for _ in 0..plan.parts {
        term.copy_from_slice(v);
        for k in 1..=plan.terms {
            let scale = step / k as f64;
            for (i, x) in next.iter_mut().enumerate() {
                *x = a.data[i * n..(i + 1) * n].iter().zip(&term).map(|(h, y)| *h * *y).sum::<C64>() * scale;
            }
            std::mem::swap(&mut term, &mut next);
            for (x, y) in v.iter_mut().zip(&term) {
                *x += *y;
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// 3 × 3 real helpers.

/// A real 3-vector (lab axes x, y, z).
pub type V3 = [f64; 3];
/// A real 3 × 3 matrix, rows first.
pub type M3 = [[f64; 3]; 3];

pub fn dot3(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
pub fn norm3(a: V3) -> f64 {
    dot3(a, a).sqrt()
}
pub fn cross3(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
/// M v.
pub fn apply3(m: &M3, v: V3) -> V3 {
    [dot3(m[0], v), dot3(m[1], v), dot3(m[2], v)]
}
/// Two unit vectors that complete the unit vector n to a right-handed orthonormal basis (e₁, e₂, n): e₁ the lab axis
/// least along n less its part along n, normalized; e₂ = n × e₁.
pub fn perpendicular(n: V3) -> (V3, V3) {
    let least = (0..3).min_by(|&i, &j| n[i].abs().total_cmp(&n[j].abs())).unwrap();
    let mut e = [0.0; 3];
    e[least] = 1.0;
    let along = dot3(e, n);
    let e1 = [e[0] - along * n[0], e[1] - along * n[1], e[2] - along * n[2]];
    let l = norm3(e1);
    let e1 = [e1[0] / l, e1[1] / l, e1[2] / l];
    (e1, cross3(n, e1))
}
/// The eigendecomposition of a real symmetric 3 × 3 matrix (its symmetric part): eigenvalues ascending and their
/// orthonormal eigenvectors (rows), by `hermitian_eigen` (the one implementation; real rotations keep them real).
pub fn symmetric3_eigen(m: &M3) -> (V3, [V3; 3]) {
    let h = Mat::from_fn(3, 3, |i, j| C64::real(0.5 * (m[i][j] + m[j][i])));
    let e = hermitian_eigen(&h);
    let vectors = std::array::from_fn(|k| std::array::from_fn(|i| e.vectors[(i, k)].re));
    ([e.values[0], e.values[1], e.values[2]], vectors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::{Purpose, Stream};
    use crate::units::rounding;

    /// A reproducible matrix of entries uniform in [−½, ½] + i[−½, ½].
    fn sample(rows: usize, cols: usize, seed: u64) -> Mat {
        let mut s = Stream::new(seed, Purpose::Initial, rows as u64, cols as u32);
        Mat::from_fn(rows, cols, |_, _| C64::new(s.uniform() - 0.5, s.uniform() - 0.5))
    }
    /// A unitary matrix: the Cayley transform (I + iH)⁻¹(I − iH) of a random Hermitian H, unitary to round-off.
    fn unitary(n: usize, seed: u64) -> Mat {
        let h = sample(n, n, seed).hermitian_part();
        let (mut plus, mut minus) = (h.scaled(C64::I), h.scaled(-C64::I));
        for i in 0..n {
            plus[(i, i)] += C64::ONE;
            minus[(i, i)] += C64::ONE;
        }
        Lu::new(&plus).unwrap().solve_columns(&minus)
    }
    fn difference(a: &Mat, b: &Mat) -> f64 {
        a.data.iter().zip(&b.data).fold(0.0, |m, (x, y)| m.max((*x - *y).abs()))
    }

    /// LU with partial pivoting is backward stable: the residual of a random 24 × 24 system is within its normwise
    /// backward error bound γ_{3n}·ρ‖A‖‖x‖ (Higham Thm 9.3; the growth ρ of a random matrix below n).
    #[test]
    fn the_lu_solves_within_its_backward_error() {
        let n = 24;
        let (a, b) = (sample(n, n, 1), sample(n, 3, 2));
        let x = Lu::new(&a).unwrap().solve_columns(&b);
        let mut r = a.product(&x);
        r.add_scaled(-C64::ONE, &b);
        let bound = rounding(3 * n) * n as f64 * a.norm1() * x.max_abs();
        assert!(r.max_abs() <= bound, "{:e} > {bound:e}", r.max_abs());
    }

    /// The Hermitian eigendecomposition of U diag(d) U† with known d (repeated values included, a degenerate
    /// eigenspace): the eigenvalues are d within γ‖A‖_F (Weyl's bound of the off-diagonal left at u‖A‖_F plus the
    /// rotations' rounding), U is orthonormal and A U = U Λ to round-off, for n = 1 … 32.
    #[test]
    fn the_hermitian_eigen_finds_known_spectra() {
        for n in [1, 2, 3, 5, 8, 16, 32] {
            let mut s = Stream::new(3, Purpose::Initial, n as u64, 0);
            let mut d: Vec<f64> = (0..n).map(|i| if i % 3 == 2 { 7.0 } else { 40.0 * (s.uniform() - 0.5) }).collect();
            let u = unitary(n, 10 + n as u64);
            let a = u.product(&Mat::diagonal(&d.iter().map(|&x| C64::real(x)).collect::<Vec<_>>())).product(&u.adjoint());
            let e = hermitian_eigen(&a);
            d.sort_by(f64::total_cmp);
            let scale = a.frobenius();
            let worst = e.values.iter().zip(&d).fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()));
            assert!(worst <= rounding(8 * n) * scale, "n {n}: eigenvalues off by {worst:e} of {scale:e}");
            let gram = e.vectors.adjoint_product(&e.vectors);
            let off = difference(&gram, &Mat::identity(n));
            assert!(off <= rounding(8 * n), "n {n}: U†U − I {off:e}");
            let lambda = Mat::diagonal(&e.values.iter().map(|&x| C64::real(x)).collect::<Vec<_>>());
            let residual = difference(&a.product(&e.vectors), &e.vectors.product(&lambda));
            assert!(residual <= rounding(8 * n) * scale, "n {n}: residual {residual:e}");
        }
    }

    /// The general eigendecomposition gives the matrix back, R Λ R⁻¹ = A to γ κ(R) ‖A‖, for random generators of the
    /// shape H_eff = H − (i/2)Γ (Hermitian plus a negative anti-Hermitian part) of 1 to 32 states; it finds the known
    /// eigenvalues of S diag(λ) S⁻¹; and towards the exceptional point of a two-level atom, Ω = Γ/2 in
    /// [[0, Ω/2], [Ω/2, −iΓ/2]], its condition grows as the eigenvectors merge.
    #[test]
    fn the_eigendecomposition_gives_the_matrix_back() {
        for n in [1, 2, 3, 5, 8, 12, 13, 20, 32] {
            for seed in 1..=3 {
                let mut h = sample(n, n, 100 * seed + n as u64).hermitian_part().scaled(C64::real(50.0));
                let mut s = Stream::new(seed, Purpose::Latin, n as u64, 1);
                for i in 0..n {
                    h[(i, i)] += C64::new(400.0 * (s.uniform() - 0.5), -19.0 * (s.uniform() + 0.5));
                }
                let e = eigen(&h).expect("converges");
                let back = e.vectors.product(&Mat::diagonal(&e.values)).product(&e.inverse);
                let bound = rounding(4 * n * n) * e.condition * h.max_abs();
                let error = difference(&back, &h);
                assert!(error <= bound, "n {n} seed {seed}: {error:e} > {bound:e}");
            }
        }
        // Known eigenvalues: S diag(λ) S⁻¹ with S = I + a random part of norm about ½.
        let n = 6;
        let lambda: Vec<C64> = (0..n).map(|k| C64::new(3.0 * k as f64 - 7.0, -0.5 * k as f64)).collect();
        let mut s = sample(n, n, 7).scaled(C64::real(1.0 / n as f64));
        for i in 0..n {
            s[(i, i)] += C64::ONE;
        }
        let a = s.product(&Mat::diagonal(&lambda)).product(&Lu::new(&s).unwrap().inverse());
        let e = eigen(&a).unwrap();
        let mut found = e.values.clone();
        found.sort_by(|x, y| x.re.total_cmp(&y.re));
        let worst = found.iter().zip(&lambda).fold(0.0_f64, |m, (x, y)| m.max((*x - *y).abs()));
        assert!(worst <= rounding(4 * n * n) * e.condition * a.max_abs(), "{worst:e}");
        let g = 38.0;
        let at = |omega: f64| {
            let h = Mat::from_fn(2, 2, |i, j| match (i, j) {
                (1, 1) => C64::new(0.0, -g / 2.0),
                (0, 0) => C64::ZERO,
                _ => C64::real(omega / 2.0),
            });
            eigen(&h).unwrap().condition
        };
        assert!(at(g / 2.0 * (1.0 + 1e-8)) > 1e3 * at(2.0 * g), "{} {}", at(g / 2.0 * (1.0 + 1e-8)), at(2.0 * g));
    }

    /// The Taylor plan: at the widest part y = 1 the least degree whose truncation is within u is 18 (1/19! = 8.2e-18,
    /// while 1/18!·19/18 = 1.65e-16 > u); zero norm takes nothing; the vector takes ⌈x⌉ parts, the matrix 2^⌈log₂ x⌉.
    #[test]
    fn the_taylor_plan_follows_from_the_round_off() {
        assert_eq!(Taylor::steps(1.0), Taylor { parts: 1, terms: 18 });
        assert_eq!(Taylor::steps(0.0), Taylor { parts: 1, terms: 0 });
        assert_eq!(Taylor::steps(2.5).parts, 3);
        assert_eq!(Taylor::squarings(2.5).parts, 4);
        assert!(Taylor::steps(1e-3).terms <= 6);
    }

    /// e^{−iHτ} of a Hermitian H with a known spectrum, U e^{−idτ} U†: the matrix exponential and its action on a
    /// vector agree with it to the problem's own rounding, γ (1 + ‖H‖τ), from ‖H‖τ = 0.3 to 3000; e^{−iθσ_x} =
    /// cos θ − i sin θ σ_x likewise; and for a decaying non-normal generator the three routes (matrix, vector, eigen)
    /// agree within the eigen route's κ(R) γ (1 + ‖H‖τ).
    #[test]
    fn the_exponential_is_exact_to_round_off() {
        let n = 8;
        let u = unitary(n, 5);
        let d: Vec<f64> = (0..n).map(|k| 10.0 * k as f64 - 33.0).collect();
        let h = u.product(&Mat::diagonal(&d.iter().map(|&x| C64::real(x)).collect::<Vec<_>>())).product(&u.adjoint());
        let v: Vec<C64> = sample(n, 1, 9).data;
        for tau in [0.0, 0.3 / 40.0, 7.0 / 40.0, 3000.0 / 40.0] {
            let exact = u.product(&Mat::diagonal(&d.iter().map(|&x| C64::cis(-x * tau)).collect::<Vec<_>>())).product(&u.adjoint());
            let bound = rounding(4 * n * n) * (1.0 + h.norm1() * tau);
            let m = exp(&h, C64::new(0.0, -tau));
            assert!(difference(&m, &exact) <= bound, "τ {tau}: matrix {:e} > {bound:e}", difference(&m, &exact));
            let mut w = v.clone();
            exp_apply(&h, C64::new(0.0, -tau), &mut w);
            let want = exact.apply(&v);
            let error = w.iter().zip(&want).fold(0.0_f64, |m, (x, y)| m.max((*x - *y).abs()));
            assert!(error <= bound, "τ {tau}: vector {error:e} > {bound:e}");
        }
        for theta in [0.1, 2.0, 50.0] {
            let sx = Mat::from_fn(2, 2, |i, j| if i != j { C64::ONE } else { C64::ZERO });
            let m = exp(&sx, C64::new(0.0, -theta));
            let exact = Mat::from_fn(2, 2, |i, j| if i == j { C64::real(theta.cos()) } else { C64::new(0.0, -theta.sin()) });
            assert!(difference(&m, &exact) <= rounding(16) * (1.0 + theta), "θ {theta}: {:e}", difference(&m, &exact));
        }
        let mut k = h.clone();
        for i in 0..n {
            k[(i, i)] += C64::new(0.0, -3.0 * (i + 1) as f64);
        }
        k[(0, n - 1)] += C64::real(25.0);
        let e = eigen(&k).unwrap();
        for tau in [0.05, 1.0] {
            let t = C64::new(0.0, -tau);
            let m = exp(&k, t);
            let (mut a, mut b) = (v.clone(), v.clone());
            exp_apply(&k, t, &mut a);
            e.apply_exp(t, &mut b);
            let want = m.apply(&v);
            let bound = rounding(4 * n * n) * e.condition * (1.0 + k.norm1() * tau);
            for (route, x) in [("vector", &a), ("eigen", &b)] {
                let error = x.iter().zip(&want).fold(0.0_f64, |m, (p, q)| m.max((*p - *q).abs()));
                assert!(error <= bound, "τ {tau} {route}: {error:e} > {bound:e}");
            }
        }
    }

    /// The 3 × 3 helpers: a symmetric matrix's eigenvectors are orthonormal and diagonalize it; the perpendicular pair
    /// completes n to a right-handed orthonormal basis.
    #[test]
    fn the_three_by_three_helpers_are_orthonormal() {
        let m = [[2.0, 0.3, -1.0], [0.3, -4.0, 0.5], [-1.0, 0.5, 1.0]];
        let (values, vectors) = symmetric3_eigen(&m);
        for k in 0..3 {
            let mv = apply3(&m, vectors[k]);
            for i in 0..3 {
                assert!((mv[i] - values[k] * vectors[k][i]).abs() <= rounding(24) * 6.0);
            }
            for l in 0..3 {
                assert!((dot3(vectors[k], vectors[l]) - if k == l { 1.0 } else { 0.0 }).abs() <= rounding(24));
            }
        }
        assert!(values[0] <= values[1] && values[1] <= values[2]);
        let n = [0.6, 0.0, -0.8];
        let (a, b) = perpendicular(n);
        assert!(dot3(a, n).abs() <= rounding(3) && dot3(b, n).abs() <= rounding(3) && dot3(a, b).abs() <= rounding(3));
        assert!((norm3(a) - 1.0).abs() <= rounding(3) && (norm3(b) - 1.0).abs() <= rounding(3));
        let c = cross3(a, b);
        assert!((0..3).all(|i| (c[i] - n[i]).abs() <= rounding(6)));
    }
}
