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
//! e^X for ‖X‖₁ ≤ x is summed as Taylor polynomials of degree m on parts of norm y = x/q (q parts: sub-steps on a
//! vector, q = 2^s squarings for the matrix). With all parts commuting, the truncation is a backward error: the degree
//! m polynomial T_m(Y) = e^{Y + ΔY} with ‖ΔY‖ ≤ e^y y^{m+1}/(m+1)! (the remainder's bound relative to ‖e^{-Y}‖ ≤ e^y),
//! and the q parts compose to e^{X + qΔY}, the same relative backward error ‖ΔY‖/y. The rounding of a part's sum is at
//! most γ_m Σ_k y^k/k! ≤ γ_m e^y of its result's scale; over q parts it is q γ_m e^y = (x/y) γ_m e^y, a backward error
//! γ_m e^y/y relative to ‖X‖. The plan takes the part norm y* that minimizes the total relative backward error
//! (truncation + rounding) per unit of ‖X‖, with m the least degree whose truncation is within the unit round-off:
//! both follow from u alone, computed once (`Taylor::widest`), no table of θ_m.

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

/// The eigendecomposition of the Hermitian part of h by cyclic complex Jacobi rotations, swept until a sweep changes
/// nothing (quadratic convergence; each rotation exactly unitary up to its own rounding, so U stays orthonormal).
/// High relative accuracy: each eigenvalue to γ ‖H‖ (Demmel & Veselić). Cost about 8 n³ complex multiply-adds per sweep,
/// some 6-10 sweeps.
pub fn hermitian_eigen(h: &Mat) -> Hermitian {
    let _ = h;
    todo!("phase 2")
}

/// A = R diag(λ) R⁻¹ of a general square matrix: the eigenvalues (in the complex Schur form's order), R with columns
/// of unit 2-norm, R⁻¹, and the condition κ(R) = ‖R‖₁‖R⁻¹‖₁ (`Mat::norm1`), which bounds what is computed through
/// the decomposition: e^{tA}v through it errs by about κ(R)·u·(1 + |t|·(max Re λ − min Re λ)) relative to ‖v‖, and κ
/// grows without bound towards an exceptional point.
#[derive(Clone, Debug)]
pub struct Eigen {
    pub values: Vec<C64>,
    pub vectors: Mat,
    pub inverse: Mat,
    pub condition: f64,
}

/// The eigendecomposition of a general complex matrix: Householder reduction to Hessenberg form, the shifted QR
/// iteration to the complex Schur form T = Q†AQ (Wilkinson shifts, deflation where a subdiagonal entry is within u of
/// its diagonal neighbours, LAPACK zlahqr's criterion), the eigenvectors of T by back substitution, R = QY normalized,
/// and R⁻¹ by LU. None when the QR iteration does not converge within LAPACK's 30 iterations per eigenvalue
/// (exceptional shifts every tenth), or R is singular.
pub fn eigen(a: &Mat) -> Option<Eigen> {
    let _ = a;
    todo!("phase 2")
}

impl Eigen {
    /// v ← R e^{tΛ} R⁻¹ v: e^{tA} v through the decomposition, exact at any t up to the rounding `Eigen` states.
    pub fn apply_exp(&self, t: C64, v: &mut [C64]) {
        let _ = (t, v);
        todo!("phase 2")
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// The exponential.

/// The Taylor plan of e^X for ‖X‖₁ ≤ x (module documentation, # The Taylor plan): `parts` of norm ≤ x/parts, each
/// summed through degree `terms`. Its work: parts × terms products (matrix-vector n², or for the matrix exponential
/// terms − 1 + log₂ parts matrix products of n³).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Taylor {
    pub parts: usize,
    pub terms: usize,
}

impl Taylor {
    /// The plan on a vector: the fewest parts of norm within the widest y*, the least degree for their norm.
    pub fn steps(x: f64) -> Taylor {
        let _ = x;
        todo!("phase 2")
    }
    /// The plan of the matrix exponential: parts a power of two (its squarings), the least degree for their norm.
    pub fn squarings(x: f64) -> Taylor {
        let _ = x;
        todo!("phase 2")
    }
}

/// e^{tA} to round-off: scaling and squaring of the Taylor polynomial by `Taylor::squarings` of ‖tA‖₁.
pub fn exp(a: &Mat, t: C64) -> Mat {
    let _ = (a, t);
    todo!("phase 2")
}

/// v ← e^{tA} v to round-off: the Taylor series on the vector in the parts of `Taylor::steps` of ‖tA‖₁; no matrix is
/// formed.
pub fn exp_apply(a: &Mat, t: C64, v: &mut [C64]) {
    let _ = (a, t, v);
    todo!("phase 2")
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
