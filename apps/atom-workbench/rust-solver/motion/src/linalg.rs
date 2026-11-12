//! Complex numbers and small dense complex matrices (n ≤ 32), what the model and the integrator need (SPEC §2.1):
//! products and the adjoint, the LU solve, the Hermitian eigendecomposition, the general (non-Hermitian)
//! eigendecomposition with its condition, the exponential's action on a vector to round-off, and 3 × 3 real
//! helpers. One implementation per operation; a real symmetric matrix is a Hermitian one.
//!
//! # To round-off
//! Every iteration here runs until the floating-point format, not a tolerance, ends it (the QR iteration besides within
//! LAPACK's count, beyond which the decomposition is refused): its cost is logarithmic in the
//! accuracy (SPEC §2.1), so there is no tolerance to choose. "At round-off" means backward stable: the result is the
//! exact one of data perturbed relatively by a modest multiple of the unit round-off u (`units::UNIT_ROUNDOFF`); its
//! forward error is then the problem's own condition times that.
//!
//! # The Chebyshev plan of the exponential's action (`Chebyshev`)
//! On a vector, e^{tA}v = e^{−iz} e^{−iX}v with X = itA − z = H + iK, H and K Hermitian and z a scalar: of a generator,
//! tA = −iH_eff s, H is its Hermitian part's s and K its decay's, each less its part of z -- the middle of its spectrum
//! as its bound knows it (`bounds`). The scalar part goes out exactly: the trap's light shift, hundreds of rad/µs on
//! every state alike, costs nothing, nor does a detuning that all of a cluster's excited states share cost more than
//! half of itself. For the rest
//! the Jacobi–Anger expansion e^{−iσx} = J₀(σ) + 2 Σ_{k≥1} (−i)^k J_k(σ) T_k(x) at x = X/σ sums the exponential in
//! Chebyshev polynomials of X/σ. Of a Hermitian X with σ ≥ ‖X‖₂, ‖T_k(X/σ)‖ ≤ 1 and the terms fall as J_k(σ),
//! |J_k(σ)| ≤ (σ/2)^k/k!, once k is beyond σ: about σ products of n², where Taylor polynomials on parts of norm one take
//! 18 σ.
//! - The bounds: ρ ≥ ‖H‖₂ and γ ≥ ‖K‖₂, of each the lesser of its Gershgorin hull's half width about the hull's
//!   middle and its Frobenius norm about its diagonal's mean (`bounds`), and σ = max(ρ, γ).
//! - The truncation: the numerical range of X/σ lies in the rectangle |Re| ≤ ρ/σ, |Im| ≤ γ/σ, inside the ellipse of foci
//!   ±1 through its corner w = (ρ + iγ)/σ, on which |T_k| ≤ r^k, r = |w + √(w² − 1)| (1 for a Hermitian X); so
//!   ‖T_k(X/σ)‖ ≤ (1 + √2) r^k (Crouzeix & Palencia 2017), and on a part of σ/q the terms beyond K sum to at most
//!   2(1 + √2) Σ_{k>K} y^k/k! ‖v‖ with y = σr/2q, a sum within its first term over 1 − y/(K + 2): K is the least for
//!   which this is within u.
//! - The rounding: the terms reach r^{σ/q} ≤ e^{(σ/q)(r − 1)}, and σ(r − 1) ≤ γ + √(2σγ + γ²) = Y: a hump where the
//!   states' decay rates differ, which the rounding of a part's sum scales with while the parts' roundings add up: over
//!   q parts q e^{Y/q}, least where a part's exponent is of order one (its derivative in q vanishes at q = Y). So the
//!   parts are the fewest with σγ/q² ≤ 1, q = ⌈√(σγ)⌉ (γ/q ≤ 1 with it, as σ ≥ γ) -- one part for a Hermitian X of any
//!   norm. (A part's y is also kept within ln(f64::MAX): the plan's own y^k/k! stay in range.)
//! - The sum: Clenshaw's recurrence b_k = c_k v + 2(X/σ) b_{k+1} − b_{k+2} downwards from K, the part's sum
//!   c₀v + (X/σ) b₁ − b₂; the coefficients' J_k(σ/q) by Miller's backward recurrence J_{k−1} = (2k/z) J_k − J_{k+1} in
//!   step with it, from K + 2 and unnormalized -- the sum is linear in them and is divided by J₀ + 2 Σ_m J_{2m} = 1 at its
//!   end --, so no coefficient is stored. K products a part.

use crate::units::{rounding, UNIT_ROUNDOFF};
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

// The kernels below (`axpy`, `dot`, `combine`; table.rs `blend`) are written once, for 256-bit vectors of reals with fused
// multiply-add, and the whole build takes those instructions (apps/atom-workbench/.cargo/config.toml; a build that sets
// RUSTFLAGS itself names them there): no second, scalar form and no choice at run time. Without the features a fused
// multiply-add is a library call per number -- so such a build is refused here instead of running many times slower.
#[cfg(not(all(target_feature = "avx2", target_feature = "fma")))]
compile_error!("atom-motion's kernels are written for AVX2 with FMA: build with -C target-feature=+avx2,+fma (apps/atom-workbench/.cargo/config.toml)");

/// The reals of one vector of the split kernels' sums (`axpy`, `dot`, `combine`): their vectors' lengths are multiples
/// of it.
pub const LANES: usize = 4;

/// y ← y + c x on complex vectors given by their real parts and their imaginary parts apart (x: `real`, `imaginary`;
/// y: `re`, `im`), whole vectors of `LANES`: lane by lane, two fused multiply-adds a real, no element depending on
/// another.
pub fn axpy(c: C64, real: &[f64], imaginary: &[f64], re: &mut [f64], im: &mut [f64]) {
    debug_assert!(re.len() % LANES == 0 && [real.len(), imaginary.len(), im.len()] == [re.len(); 3]);
    for (((re, im), real), imaginary) in re.chunks_exact_mut(LANES).zip(im.chunks_exact_mut(LANES)).zip(real.chunks_exact(LANES)).zip(imaginary.chunks_exact(LANES)) {
        for l in 0..LANES {
            re[l] = c.re.mul_add(real[l], (-c.im).mul_add(imaginary[l], re[l]));
            im[l] = c.re.mul_add(imaginary[l], c.im.mul_add(real[l], im[l]));
        }
    }
}

/// Σ_i x_i y_i (no conjugate) of two complex vectors given by their real parts and their imaginary parts apart, whole
/// vectors of `LANES`: one partial sum a lane, so the lanes add side by side.
pub fn dot(xr: &[f64], xi: &[f64], yr: &[f64], yi: &[f64]) -> C64 {
    debug_assert!(xr.len() % LANES == 0 && [xi.len(), yr.len(), yi.len()] == [xr.len(); 3]);
    let (mut re, mut im) = ([0.0; LANES], [0.0; LANES]);
    for (((xr, xi), yr), yi) in xr.chunks_exact(LANES).zip(xi.chunks_exact(LANES)).zip(yr.chunks_exact(LANES)).zip(yi.chunks_exact(LANES)) {
        for l in 0..LANES {
            re[l] = xr[l].mul_add(yr[l], (-xi[l]).mul_add(yi[l], re[l]));
            im[l] = xr[l].mul_add(yi[l], xi[l].mul_add(yr[l], im[l]));
        }
    }
    C64::new((re[0] + re[1]) + (re[2] + re[3]), (im[0] + im[1]) + (im[2] + im[3]))
}

/// Sums of complex columns laid for the lanes: per vector of `LANES` rows each column's real parts and then its imaginary
/// parts, the columns in turn (`columns`: `count` of them; `sums`: `width`, of the same rows). The sums from `at` on, one
/// per `count` weights of `c`, are y_o ← Σ_k c_ok x_k, or onto what they hold (`onto`) y_o + Σ_k c_ok x_k: a matrix on a
/// vector, or vectors' combinations. Vector by vector a sum is held in place and written once, two fused multiply-adds a
/// real and column, no lane depending on another. Its sums are of a few vectors: it is inlined where it is called (a
/// call would save and restore more registers than they fill), and its places are counted by sums and products alone,
/// never a quotient.
#[inline(always)]
pub fn combine(onto: bool, c: &[C64], count: usize, columns: &[f64], sums: &mut [f64], width: usize, at: usize) {
    const PAIR: usize = 2 * LANES;
    debug_assert!(count > 0 && c.len() % count == 0 && at + c.len() / count <= width && columns.len() % (PAIR * count) == 0 && sums.len() / width == columns.len() / count);
    let (mut from, mut to) = (0, PAIR * at);
    while from < columns.len() {
        let x = &columns[from..from + PAIR * count];
        let (mut first, mut sum) = (0, to);
        while first < c.len() {
            let (c, y) = (&c[first..first + count], &mut sums[sum..sum + PAIR]);
            let (mut re, mut im) = ([0.0; LANES], [0.0; LANES]);
            if onto {
                re.copy_from_slice(&y[..LANES]);
                im.copy_from_slice(&y[LANES..]);
            }
            for (c, x) in c.iter().zip(x.chunks_exact(PAIR)) {
                for l in 0..LANES {
                    re[l] = c.re.mul_add(x[l], (-c.im).mul_add(x[LANES + l], re[l]));
                    im[l] = c.re.mul_add(x[LANES + l], c.im.mul_add(x[l], im[l]));
                }
            }
            y[..LANES].copy_from_slice(&re);
            y[LANES..].copy_from_slice(&im);
            first += count;
            sum += PAIR;
        }
        from += PAIR * count;
        to += PAIR * width;
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
#[derive(Clone, Debug, Default, PartialEq)]
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
    /// ‖A‖₁: the largest column sum of the entries' moduli.
    pub fn norm1(&self) -> f64 {
        (0..self.cols).map(|j| (0..self.rows).map(|i| self[(i, j)].abs()).sum::<f64>()).fold(0.0, f64::max)
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

/// P A = L U of the n × n matrix in `f` (row by row), in its place: L unit lower, U upper, partial pivoting, the pivots'
/// rows. Backward stable: the computed factors are those of A + ΔA, |ΔA| ≤ γ_n |L||U| (Higham, Theorem 9.3). Cost n³/3
/// complex multiply-adds. Returns whether no pivot column was exactly zero: with one there is no factorization (the
/// factors then hold what a division by zero makes). A nearly singular A factors, its solutions large: its condition is
/// the caller's to read.
pub fn factor(f: &mut [C64], pivots: &mut [usize], n: usize) -> bool {
    let mut regular = true;
    for k in 0..n {
        let p = (k..n).max_by(|&i, &j| f[i * n + k].abs1().total_cmp(&f[j * n + k].abs1())).unwrap();
        regular &= f[p * n + k] != C64::ZERO;
        pivots[k] = p;
        if p != k {
            let (above, below) = f.split_at_mut(p * n);
            above[k * n..(k + 1) * n].swap_with_slice(&mut below[..n]);
        }
        let (above, below) = f.split_at_mut((k + 1) * n);
        let pivot_row = &above[k * n..];
        for row in below.chunks_exact_mut(n) {
            let l = row[k] / pivot_row[k];
            row[k] = l;
            for (z, upper) in row[k + 1..].iter_mut().zip(&pivot_row[k + 1..]) {
                *z -= l * *upper;
            }
        }
    }
    regular
}

/// B ← A⁻¹B by A's factors (`factor`), B of n rows of `columns` numbers, row by row: n² complex multiply-adds a column.
pub fn solve(f: &[C64], pivots: &[usize], n: usize, b: &mut [C64], columns: usize) {
    for (k, &p) in pivots.iter().enumerate() {
        if p != k {
            let (above, below) = b.split_at_mut(p * columns);
            above[k * columns..(k + 1) * columns].swap_with_slice(&mut below[..columns]);
        }
    }
    for i in 0..n {
        let (above, row) = b.split_at_mut(i * columns);
        for (l, other) in f[i * n..i * n + i].iter().zip(above.chunks_exact(columns)) {
            for (z, other) in row[..columns].iter_mut().zip(other) {
                *z -= *l * *other;
            }
        }
    }
    for i in (0..n).rev() {
        let (rest, below) = b.split_at_mut((i + 1) * columns);
        let row = &mut rest[i * columns..];
        for (upper, other) in f[i * n + i + 1..(i + 1) * n].iter().zip(below.chunks_exact(columns)) {
            for (z, other) in row.iter_mut().zip(other) {
                *z -= *upper * *other;
            }
        }
        let diagonal = f[i * n + i];
        for z in row.iter_mut() {
            *z = *z / diagonal;
        }
    }
}

/// The LU factorization with partial pivoting of a square matrix (`factor`), held.
#[derive(Clone, Debug)]
pub struct Lu {
    factors: Mat,
    pivots: Vec<usize>,
}

impl Lu {
    /// The factorization; None when a pivot column is exactly zero, so that no factorization exists.
    pub fn new(a: &Mat) -> Option<Lu> {
        let n = a.rows;
        assert_eq!(n, a.cols, "LU of a {}x{} matrix", a.rows, a.cols);
        let mut lu = Lu { factors: a.clone(), pivots: vec![0; n] };
        factor(&mut lu.factors.data, &mut lu.pivots, n).then_some(lu)
    }
    /// b ← A⁻¹ b.
    pub fn solve(&self, b: &mut [C64]) {
        solve(&self.factors.data, &self.pivots, self.factors.rows, b, 1);
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

/// The numbers of `jacobi`: the complex ones, or the reals a real symmetric matrix stays in (the same rotations, the
/// same operations on its real parts: the one implementation).
trait Scalar: Copy + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Mul<f64, Output = Self> + Div<f64, Output = Self> {
    const ZERO: Self;
    fn conj(self) -> Self;
    fn re(self) -> f64;
    fn norm_sqr(self) -> f64;
    fn abs(self) -> f64;
}

impl Scalar for C64 {
    const ZERO: C64 = C64::ZERO;
    fn conj(self) -> C64 {
        C64::conj(self)
    }
    fn re(self) -> f64 {
        self.re
    }
    fn norm_sqr(self) -> f64 {
        C64::norm_sqr(self)
    }
    fn abs(self) -> f64 {
        C64::abs(self)
    }
}

impl Scalar for f64 {
    const ZERO: f64 = 0.0;
    fn conj(self) -> f64 {
        self
    }
    fn re(self) -> f64 {
        self
    }
    fn norm_sqr(self) -> f64 {
        self * self
    }
    fn abs(self) -> f64 {
        f64::abs(self)
    }
}

/// The eigendecomposition of a Hermitian matrix a (n × n, row-major) by cyclic Jacobi rotations, in place: a becomes
/// diagonal (its diagonal the eigenvalues, unordered) and u, the identity on entry, its eigenvectors' columns. Each
/// rotation is unitary up to its own rounding, so U stays orthonormal to round-off, in a degenerate eigenspace too. The
/// sweeps run until the off-diagonal part is within the unit round-off of the matrix, ‖off(A)‖_F ≤ u‖A‖_F, or a sweep no
/// longer shrinks it (the floating-point format's floor; the convergence is quadratic). Each eigenvalue is then within
/// ‖off(A)‖₂ ≤ u‖A‖_F of the exact one (Weyl). Cost about 6n³ multiply-adds per sweep, some 6-10 sweeps.
fn jacobi<S: Scalar>(a: &mut [S], u: &mut [S], n: usize) {
    let floor = UNIT_ROUNDOFF * UNIT_ROUNDOFF * a.iter().map(|z| z.norm_sqr()).sum::<f64>();
    let off = |a: &[S]| -> f64 { (0..n).map(|i| (0..n).filter(|&j| j != i).map(|j| a[i * n + j].norm_sqr()).sum::<f64>()).sum() };
    let mut previous = f64::INFINITY;
    loop {
        let left = off(a);
        if left <= floor || !(left < previous) {
            break;
        }
        previous = left;
        for p in 0..n {
            for q in p + 1..n {
                let modulus = a[p * n + q].abs();
                if modulus == 0.0 {
                    continue;
                }
                // A phase on q makes a_pq real and positive (dividing by the modulus: its reciprocal would overflow
                // when it is subnormal); a real rotation in (p, q) then zeroes it.
                let phase = a[p * n + q] / modulus;
                for k in 0..n {
                    a[k * n + q] = a[k * n + q] * phase.conj();
                    u[k * n + q] = u[k * n + q] * phase.conj();
                }
                for k in 0..n {
                    a[q * n + k] = a[q * n + k] * phase;
                }
                // tan θ, the smaller root of t² + 2τt − 1 = 0 (Rutishauser): |θ| ≤ π/4.
                let tau = (a[q * n + q].re() - a[p * n + p].re()) / (2.0 * modulus);
                let t = tau.signum() / (tau.abs() + (1.0 + tau * tau).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (x, y) = (a[k * n + p], a[k * n + q]);
                    a[k * n + p] = x * c - y * s;
                    a[k * n + q] = x * s + y * c;
                    let (x, y) = (u[k * n + p], u[k * n + q]);
                    u[k * n + p] = x * c - y * s;
                    u[k * n + q] = x * s + y * c;
                }
                for k in 0..n {
                    let (x, y) = (a[p * n + k], a[q * n + k]);
                    a[p * n + k] = x * c - y * s;
                    a[q * n + k] = x * s + y * c;
                }
                a[p * n + q] = S::ZERO;
                a[q * n + p] = S::ZERO;
            }
        }
    }
}

/// The eigendecomposition of the Hermitian part of h, (h + h†)/2 (`jacobi`), the eigenvalues ascending.
pub fn hermitian_eigen(h: &Mat) -> Hermitian {
    let n = h.rows;
    assert_eq!(n, h.cols, "the eigendecomposition of a {}x{} matrix", h.rows, h.cols);
    let mut a = h.hermitian_part();
    let mut u = Mat::identity(n);
    jacobi(&mut a.data, &mut u.data, n);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| a[(i, i)].re.total_cmp(&a[(j, j)].re));
    Hermitian { values: order.iter().map(|&i| a[(i, i)].re).collect(), vectors: Mat::from_fn(n, n, |i, k| u[(i, order[k])]) }
}

/// A = R diag(λ) R⁻¹ of a general square matrix: the eigenvalues (in the complex Schur form's order), R with columns
/// of unit 2-norm, R⁻¹, and the condition κ(R) = ‖R‖₁‖R⁻¹‖₁, which bounds what is computed through the
/// decomposition: e^{tA}v through it errs by about κ(R)·u·(1 + |t|·(max Re λ − min Re λ)) relative to ‖v‖,
/// and κ grows without bound towards an exceptional point. `work` is the count of its complex multiply-adds as its
/// algorithm made them (`decide::Work` counts, SPEC §5.1): the reduction's, the QR iteration's rotations', the back
/// substitution's, R's and its inverse's. It keeps its memory: decomposing again (`Eigen::decompose`) asks the heap for
/// nothing while the size stays.
#[derive(Clone, Debug, Default)]
pub struct Eigen {
    pub values: Vec<C64>,
    pub vectors: Mat,
    pub inverse: Mat,
    pub condition: f64,
    pub work: f64,
    /// The decomposition's own: H (the Hessenberg and then the Schur form, row by row); Q, T's eigenvectors Y and R,
    /// each column by column (a column's numbers side by side, as the rotations and the sums take them); a
    /// reflection's vector and its sums; R's factors and their pivots.
    schur: Vec<C64>,
    columns: [Vec<C64>; 3],
    reflection: [Vec<C64>; 2],
    factors: Vec<C64>,
    pivots: Vec<usize>,
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

/// (u, w) ← (c u + s w, c w − s* u) on two vectors, number by number.
fn rotate(first: &mut [C64], second: &mut [C64], c: f64, s: C64) {
    for (x, y) in first.iter_mut().zip(second.iter_mut()) {
        let (u, w) = (*x, *y);
        *x = u * c + w * s;
        *y = w * c - u * s.conj();
    }
}

/// The eigendecomposition of a general complex matrix (`Eigen::decompose`), on memory of its own: the tests' entry (a
/// step's picture keeps its own, spectral.rs).
#[cfg(test)]
pub(crate) fn eigen(a: &Mat) -> Option<Eigen> {
    let mut e = Eigen::default();
    e.decompose(a).then_some(e)
}

impl Eigen {
    /// R column by column (its transpose row by row): a product's right factor as `combine` takes its weights.
    pub fn columns(&self) -> &[C64] {
        &self.columns[2]
    }

    /// The eigendecomposition of a general complex matrix: Householder reduction to Hessenberg form; the single-shift
    /// implicit QR iteration to the complex Schur form T = Q†AQ, with Wilkinson shifts (an exceptional shift every tenth
    /// iteration) and deflation where a subdiagonal entry is within the precision of its two diagonal neighbours (LAPACK
    /// zlahqr's criterion); the eigenvectors of T by back substitution (a difference of eigenvalues below the precision of
    /// T's scale taken at it, LAPACK ztrevc's rule); R = QY with unit columns, and R⁻¹ by LU. False -- and nothing of
    /// the decomposition to read -- when the QR iteration does not converge within LAPACK zlahqr's 30·max(10, n)
    /// iterations per eigenvalue, or R is singular.
    pub fn decompose(&mut self, a: &Mat) -> bool {
        let n = a.rows;
        assert_eq!(n, a.cols, "the eigendecomposition of a {}x{} matrix", a.rows, a.cols);
        if self.values.len() != n {
            self.values.resize(n, C64::ZERO);
            for m in [&mut self.vectors, &mut self.inverse] {
                (m.rows, m.cols) = (n, n);
                m.data.resize(n * n, C64::ZERO);
            }
            for v in self.columns.iter_mut().chain([&mut self.schur, &mut self.factors]) {
                v.resize(n * n, C64::ZERO);
            }
            self.reflection.iter_mut().for_each(|v| v.resize(n, C64::ZERO));
            self.pivots.resize(n, 0);
        }
        let h = &mut self.schur[..];
        let [q, y, r] = &mut self.columns;
        let [v, sums] = &mut self.reflection;
        h.copy_from_slice(&a.data);
        let mut work = 0;
        for (k, column) in q.chunks_exact_mut(n).enumerate() {
            column.fill(C64::ZERO);
            column[k] = C64::ONE;
        }
        // Householder reduction: H ← P H P, Q ← Q P, P = I − β v v† zeroing column k below its subdiagonal.
        for k in 0..n.saturating_sub(2) {
            let alpha = (k + 1..n).map(|i| h[i * n + k].norm_sqr()).sum::<f64>().sqrt();
            if alpha == 0.0 {
                continue;
            }
            let x0 = h[(k + 1) * n + k];
            let phase = if x0 == C64::ZERO { C64::ONE } else { x0 / x0.abs() };
            for i in 0..n {
                v[i] = if i > k { h[i * n + k] } else { C64::ZERO };
            }
            v[k + 1] += phase * alpha;
            let beta = 2.0 / (k + 1..n).map(|i| v[i].norm_sqr()).sum::<f64>();
            // The two norms, the reflection on H's rows, and on H's and Q's columns: of n − k − 1 terms each.
            work += (2 + 6 * n) * (n - k - 1);
            // H ← H − v (β v†H): a column's sum over the rows below k, row after row.
            sums.fill(C64::ZERO);
            for (v, row) in v[k + 1..].iter().zip(h[(k + 1) * n..].chunks_exact(n)) {
                for (s, z) in sums.iter_mut().zip(row) {
                    *s += v.conj() * *z;
                }
            }
            for s in sums.iter_mut() {
                *s = *s * beta;
            }
            for (v, row) in v[k + 1..].iter().zip(h[(k + 1) * n..].chunks_exact_mut(n)) {
                for (z, s) in row.iter_mut().zip(sums.iter()) {
                    *z -= *v * *s;
                }
            }
            // H ← H − (β H v) v†: a row's sum over the columns beyond k.
            for row in h.chunks_exact_mut(n) {
                let s = row[k + 1..].iter().zip(&v[k + 1..]).map(|(z, v)| *z * *v).sum::<C64>() * beta;
                for (z, v) in row[k + 1..].iter_mut().zip(&v[k + 1..]) {
                    *z -= s * v.conj();
                }
            }
            // Q ← Q − (β Q v) v†: a row's sum over Q's columns beyond k, column after column.
            sums.fill(C64::ZERO);
            for (v, column) in v[k + 1..].iter().zip(q[(k + 1) * n..].chunks_exact(n)) {
                for (s, z) in sums.iter_mut().zip(column) {
                    *s += *z * *v;
                }
            }
            for s in sums.iter_mut() {
                *s = *s * beta;
            }
            for (v, column) in v[k + 1..].iter().zip(q[(k + 1) * n..].chunks_exact_mut(n)) {
                for (z, s) in column.iter_mut().zip(sums.iter()) {
                    *z -= *s * v.conj();
                }
            }
            for i in k + 2..n {
                h[i * n + k] = C64::ZERO;
            }
        }
        // The QR iteration on the active block [lo, hi), the rotations accumulated into Q.
        let scale = h.iter().map(|z| z.abs1()).fold(0.0, f64::max).max(f64::MIN_POSITIVE);
        let most = 30 * n.max(10);
        let (mut hi, mut iterations) = (n, 0);
        while hi > 1 {
            let mut lo = hi - 1;
            while lo > 0 {
                let s = h[(lo - 1) * n + lo - 1].abs1() + h[lo * n + lo].abs1();
                if h[lo * n + lo - 1].abs1() <= f64::EPSILON * if s > 0.0 { s } else { scale } {
                    h[lo * n + lo - 1] = C64::ZERO;
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
                return false;
            }
            // The Wilkinson shift: the eigenvalue of the trailing 2 × 2 block nearer its corner.
            let (p, pq, qp, s) = (h[(hi - 2) * n + hi - 2], h[(hi - 2) * n + hi - 1], h[(hi - 1) * n + hi - 2], h[(hi - 1) * n + hi - 1]);
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
                let from = if k == lo { lo } else { k - 1 };
                let (x, y) = if k == lo { (h[k * n + k] - mu, h[(k + 1) * n + k]) } else { (h[k * n + k - 1], h[(k + 1) * n + k - 1]) };
                let (c, sn) = givens(x, y);
                // Two products an entry of each rotated pair: H's rows from the bulge on, H's columns above it, Q's columns.
                work += 4 * ((n - from) + (k + 3).min(hi) + n);
                let (upper, lower) = h[k * n..(k + 2) * n].split_at_mut(n);
                rotate(&mut upper[from..], &mut lower[from..], c, sn);
                for row in h.chunks_exact_mut(n).take((k + 3).min(hi)) {
                    let (u, w) = (row[k], row[k + 1]);
                    row[k] = u * c + w * sn.conj();
                    row[k + 1] = w * c - u * sn;
                }
                let (first, second) = q[k * n..(k + 2) * n].split_at_mut(n);
                rotate(first, second, c, sn.conj());
                if k > lo {
                    h[(k + 1) * n + k - 1] = C64::ZERO;
                }
            }
        }
        // T's eigenvectors: y_kk = 1, upwards by back substitution.
        for (i, value) in self.values.iter_mut().enumerate() {
            *value = h[i * n + i];
        }
        let values = &self.values;
        let floor = f64::EPSILON * scale;
        for (k, y) in y.chunks_exact_mut(n).enumerate() {
            y.fill(C64::ZERO);
            y[k] = C64::ONE;
            for j in (0..k).rev() {
                let s: C64 = h[j * n + j + 1..=j * n + k].iter().zip(&y[j + 1..=k]).map(|(t, y)| *t * *y).sum();
                work += k - j + 1;
                let mut d = h[j * n + j] - values[k];
                if d.abs1() < floor {
                    d = C64::real(floor);
                }
                y[j] = -(s / d);
            }
        }
        // R = QY and its columns' norms and scaling; its LU (Σ_k (n − k − 1)(n − k)), its inverse (n solves of n²), the norms.
        work += n * n * n + 2 * n * n + (0..n).map(|k| (n - k - 1) * (n - k)).sum::<usize>() + n * n * n + 2 * n * n;
        // A column of R: Q's columns up to its own by its column of Y (upper triangular), then of unit norm.
        let mut one = 0.0_f64;
        for (k, (r, y)) in r.chunks_exact_mut(n).zip(y.chunks_exact(n)).enumerate() {
            r.fill(C64::ZERO);
            for (y, q) in y[..=k].iter().zip(q.chunks_exact(n)) {
                for (z, q) in r.iter_mut().zip(q) {
                    *z += *q * *y;
                }
            }
            let norm = r.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
            if !(norm > 0.0 && norm.is_finite()) {
                return false;
            }
            for z in r.iter_mut() {
                *z = *z / norm;
            }
            one = one.max(r.iter().map(|z| z.abs()).sum::<f64>());
            for (i, z) in r.iter().enumerate() {
                self.vectors.data[i * n + k] = *z;
            }
        }
        self.factors.copy_from_slice(&self.vectors.data);
        if !factor(&mut self.factors, &mut self.pivots, n) {
            return false;
        }
        for (i, row) in self.inverse.data.chunks_exact_mut(n).enumerate() {
            row.fill(C64::ZERO);
            row[i] = C64::ONE;
        }
        solve(&self.factors, &self.pivots, n, &mut self.inverse.data, n);
        self.condition = one * self.inverse.norm1();
        self.work = work as f64;
        self.condition.is_finite()
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// The exponential.

/// The Chebyshev plan of e^{−iX} on a vector (module documentation, # The Chebyshev plan): `parts` equal parts, each
/// summed through degree `terms` -- parts × terms matrix-vector products (n² each).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Chebyshev {
    pub parts: usize,
    pub terms: usize,
}

impl Chebyshev {
    /// The plan for X = H + iK with ρ ≥ ‖H‖₂ and γ ≥ ‖K‖₂: σ = max(ρ, γ); the fewest parts q with σγ/q² ≤ 1 (and a
    /// part's y = σr/2q within ln(f64::MAX)); the least degree K with 2(1 + √2) Σ_{k>K} y^k/k! ≤ u, the sum within its
    /// first term over 1 − y/(K + 2). Nothing to sum where X is zero.
    pub fn plan(rho: f64, gamma: f64) -> Chebyshev {
        let sigma = rho.max(gamma);
        assert!(sigma.is_finite() && rho >= 0.0 && gamma >= 0.0, "the exponential of a matrix of bounds {rho}, {gamma}");
        if sigma == 0.0 {
            return Chebyshev { parts: 1, terms: 0 };
        }
        let corner = C64::new(rho / sigma, gamma / sigma);
        let r = (corner + (corner * corner - C64::ONE).sqrt()).abs();
        let parts = ((sigma * gamma).sqrt().max(0.5 * sigma * r / f64::MAX.ln()).ceil() as usize).max(1);
        let y = 0.5 * sigma * r / parts as f64;
        let (mut terms, mut next) = (0, y);
        while !((terms + 2) as f64 > y && 2.0 * (1.0 + std::f64::consts::SQRT_2) * next / (1.0 - y / (terms + 2) as f64) <= UNIT_ROUNDOFF) {
            terms += 1;
            next *= y / (terms + 1) as f64;
        }
        Chebyshev { parts, terms }
    }
    /// The rounding of the plan's sum on n states relative to ‖v‖, to first order in u: every step of a part rounds a
    /// product of n terms and two additions, on vectors of ‖v‖'s size within the part's hump, and enters the sum through
    /// a Chebyshev polynomial of X/σ; the scalar part's exponential and its product round once more. γ of that many
    /// operations (`units::rounding`).
    pub fn rounding(&self, n: usize) -> f64 {
        rounding((n + 2) * self.parts * self.terms + 2)
    }
}

/// The bounds of a matrix less its scalar part (`bounds`): ρ ≥ ‖H‖₂ and γ ≥ ‖K‖₂ of X − z = H + iK, and z.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bounds {
    pub rho: f64,
    pub gamma: f64,
    pub scalar: C64,
}

/// (ρ, γ) ≥ (‖H‖₂, ‖K‖₂) of X − z = H + iK, H and K Hermitian, at the scalar z that makes each least: of X's Hermitian
/// part (X + X†)/2 less Re z and of (X − X†)/2i less Im z the lesser of
/// - the half width of its spectrum's Gershgorin hull about the hull's middle: a Hermitian matrix's eigenvalues lie in
///   the union of [d_i − r_i, d_i + r_i], d its diagonal and r_i its row's off-diagonal moduli -- close where few
///   entries couple a state, and half the distance of two groups of levels whatever the group's sizes;
/// - its Frobenius norm about its diagonal's mean, where that norm is least -- close where the spectrum is spread.
/// z's two parts are the middles taken. `sums` holds the n rows' sums.
pub fn bounds(x: &Mat, sums: &mut [C64]) -> Bounds {
    let n = x.rows;
    sums[..n].fill(C64::ZERO);
    let mut squares = C64::ZERO;
    for i in 0..n {
        for j in i + 1..n {
            let (above, below) = (x[(i, j)], x[(j, i)].conj());
            let pair = C64::new((above + below).norm_sqr(), (above - below).norm_sqr()) * 0.25;
            squares += pair * 2.0;
            let pair = C64::new(pair.re.sqrt(), pair.im.sqrt());
            sums[i] += pair;
            sums[j] += pair;
        }
    }
    let mean = x.trace() / n as f64;
    // Of the two parts (the complex numbers' real and imaginary ones): the hull's ends and the diagonal's spread.
    let (mut lo, mut hi, mut spread) = (C64::new(f64::INFINITY, f64::INFINITY), C64::new(f64::NEG_INFINITY, f64::NEG_INFINITY), C64::ZERO);
    for i in 0..n {
        let (d, r) = (x[(i, i)], sums[i]);
        lo = C64::new(lo.re.min(d.re - r.re), lo.im.min(d.im - r.im));
        hi = C64::new(hi.re.max(d.re + r.re), hi.im.max(d.im + r.im));
        spread += C64::new((d.re - mean.re).powi(2), (d.im - mean.im).powi(2));
    }
    let least = |lo: f64, hi: f64, squares: f64, mean: f64| {
        let (hull, frobenius) = (0.5 * (hi - lo), squares.sqrt());
        if hull <= frobenius { (hull, 0.5 * (lo + hi)) } else { (frobenius, mean) }
    };
    let ((rho, re), (gamma, im)) = (least(lo.re, hi.re, squares.re + spread.re, mean.re), least(lo.im, hi.im, squares.im + spread.im, mean.im));
    Bounds { rho, gamma, scalar: C64::new(re, im) }
}

/// v ← e^{tA} v to round-off (module documentation, # The Chebyshev plan), and the plan it took. A is overwritten (by
/// 2X/σ, the matrix of Clenshaw's recurrence); `work` holds 2n numbers.
pub fn exp_apply(a: &mut Mat, t: C64, v: &mut [C64], work: &mut [C64]) -> Chebyshev {
    let n = v.len();
    assert_eq!((a.rows, a.cols), (n, n), "the exponential of a {}x{} matrix on a vector of {n}", a.rows, a.cols);
    // X = itA − z, its bounds and its plan.
    let turn = C64::I * t;
    for x in &mut a.data {
        *x = turn * *x;
    }
    let Bounds { rho, gamma, scalar } = bounds(a, work);
    for i in 0..n {
        a[(i, i)] -= scalar;
    }
    let plan = Chebyshev::plan(rho, gamma);
    if plan.terms > 0 {
        let sigma = rho.max(gamma);
        let z = sigma / plan.parts as f64;
        for x in &mut a.data {
            *x *= 2.0 / sigma;
        }
        let (mut b1, mut b2) = work[..2 * n].split_at_mut(n);
        for _ in 0..plan.parts {
            // Downwards from K + 2: (J_{k+1}, J_k) unnormalized, the even orders' sum, and (b_{k+1}, b_{k+2}).
            let (mut above, mut here, mut even) = (0.0, 1.0, 0.0);
            b1.fill(C64::ZERO);
            b2.fill(C64::ZERO);
            for k in (1..=plan.terms + 2).rev() {
                if k <= plan.terms {
                    // c_k = 2(−i)^k J_k; b_k = c_k v + (2X/σ) b_{k+1} − b_{k+2}, into b_{k+2}'s place. (b_{K+1} = 0: no product.)
                    let c = [C64::ONE, -C64::I, -C64::ONE, C64::I][k % 4] * (2.0 * here);
                    for i in 0..n {
                        let product: C64 = if k < plan.terms { a.data[i * n..(i + 1) * n].iter().zip(b1.iter()).map(|(m, b)| *m * *b).sum() } else { C64::ZERO };
                        b2[i] = c * v[i] + product - b2[i];
                    }
                    std::mem::swap(&mut b1, &mut b2);
                }
                if k % 2 == 0 {
                    even += here;
                }
                (above, here) = (here, 2.0 * k as f64 / z * here - above);
            }
            // The part's sum J₀v + (X/σ) b₁ − b₂ over J₀ + 2 Σ_m J_{2m}.
            let norm = here + 2.0 * even;
            for i in 0..n {
                let product: C64 = a.data[i * n..(i + 1) * n].iter().zip(b1.iter()).map(|(m, b)| *m * *b).sum();
                b2[i] = (v[i] * here + product * 0.5 - b2[i]) / norm;
            }
            v.copy_from_slice(b2);
        }
    }
    let phase = (-C64::I * scalar).exp();
    for x in v.iter_mut() {
        *x *= phase;
    }
    plan
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
/// orthonormal eigenvectors (rows), by `jacobi` on the reals (the one implementation, on the stack).
pub fn symmetric3_eigen(m: &M3) -> (V3, [V3; 3]) {
    let mut a: [f64; 9] = std::array::from_fn(|k| 0.5 * (m[k / 3][k % 3] + m[k % 3][k / 3]));
    let mut u: [f64; 9] = std::array::from_fn(|k| if k / 3 == k % 3 { 1.0 } else { 0.0 });
    jacobi(&mut a, &mut u, 3);
    let mut order = [0, 1, 2];
    order.sort_by(|&i, &j| a[4 * i].total_cmp(&a[4 * j]));
    (order.map(|i| a[4 * i]), order.map(|k| std::array::from_fn(|i| u[3 * i + k])))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::random::{Purpose, Stream};
    use crate::units::rounding;

    /// A⁻¹ B by a factorization.
    fn solve_columns(lu: &Lu, b: &Mat) -> Mat {
        let mut out = b.clone();
        solve(&lu.factors.data, &lu.pivots, lu.factors.rows, &mut out.data, b.cols);
        out
    }

    /// The split kernels (`axpy`, `dot`, `combine`) are the complex sums they stand for, to the rounding of a sum of
    /// their length; and their pace (printed: ns a complex multiply-add on spans of 28, a five-state cluster's places;
    /// `combine`'s on five columns of 16 rows, a five-state cluster's jumps on its state).
    #[test]
    fn the_split_kernels_are_their_complex_sums() {
        let mut s = Stream::new(11, Purpose::Initial, 0, 0);
        let n = 28;
        let mut draw = |count: usize| -> Vec<f64> { (0..count).map(|_| s.uniform() - 0.5).collect() };
        let (xr, xi, yr, yi) = (draw(n), draw(n), draw(n), draw(n));
        let c = C64::new(0.3, -0.7);
        let (x, y): (Vec<C64>, Vec<C64>) = ((0..n).map(|i| C64::new(xr[i], xi[i])).collect(), (0..n).map(|i| C64::new(yr[i], yi[i])).collect());
        let (mut re, mut im) = (yr.clone(), yi.clone());
        axpy(c, &xr, &xi, &mut re, &mut im);
        for i in 0..n {
            assert!((C64::new(re[i], im[i]) - (y[i] + c * x[i])).abs() <= rounding(3) * (y[i].abs() + c.abs() * x[i].abs()), "axpy at {i}");
        }
        let (sum, size): (C64, f64) = ((0..n).map(|i| x[i] * y[i]).sum(), (0..n).map(|i| x[i].abs() * y[i].abs()).sum());
        assert!((dot(&xr, &xi, &yr, &yi) - sum).abs() <= rounding(2 * n) * size, "dot");
        // `combine`: two of three sums of five columns on sixteen rows, the middle one left as it is; alone and onto
        // what the sums hold.
        let (rows, count, width) = (16, 5, 3);
        let (columns, weights, held) = (draw(2 * rows * count), draw(2 * 2 * count), draw(2 * rows * width));
        let weights: Vec<C64> = weights.chunks(2).map(|w| C64::new(w[0], w[1])).collect();
        let entry = |values: &[f64], of: usize, k: usize, row: usize| {
            let at = ((row / LANES) * of + k) * 2 * LANES + row % LANES;
            C64::new(values[at], values[at + LANES])
        };
        for onto in [false, true] {
            let mut sums = held.clone();
            combine(onto, &weights[..count], count, &columns, &mut sums, width, 0);
            combine(onto, &weights[count..], count, &columns, &mut sums, width, 2);
            for row in 0..rows {
                assert_eq!(entry(&sums, width, 1, row), entry(&held, width, 1, row), "combine beside its sums");
                for (o, c) in [(0, &weights[..count]), (2, &weights[count..])] {
                    let start = if onto { entry(&held, width, o, row) } else { C64::ZERO };
                    let (sum, size): (C64, f64) = ((0..count).map(|k| c[k] * entry(&columns, count, k, row)).sum(), (0..count).map(|k| c[k].abs() * entry(&columns, count, k, row).abs()).sum());
                    assert!((entry(&sums, width, o, row) - (start + sum)).abs() <= rounding(2 * count + 1) * (start.abs() + size), "combine at {row}, sum {o}, onto {onto}");
                }
            }
        }
        let mut sums = vec![0.0; 2 * rows];
        let rounds = 200_000;
        let started = std::time::Instant::now();
        for _ in 0..rounds {
            axpy(std::hint::black_box(c), std::hint::black_box(&xr), &xi, &mut re, &mut im);
        }
        let first = started.elapsed().as_secs_f64() * 1e9 / (rounds * n) as f64;
        let started = std::time::Instant::now();
        let mut total = C64::ZERO;
        for _ in 0..rounds {
            total += dot(std::hint::black_box(&xr), &xi, &yr, &yi);
        }
        let second = started.elapsed().as_secs_f64() * 1e9 / (rounds * n) as f64;
        let started = std::time::Instant::now();
        for _ in 0..rounds {
            combine(false, std::hint::black_box(&weights[..count]), count, std::hint::black_box(&columns), &mut sums, 1, 0);
        }
        std::hint::black_box((total, re, im, sums));
        eprintln!("axpy {first:.3} ns, dot {second:.3} ns, combine {:.3} ns a complex multiply-add", started.elapsed().as_secs_f64() * 1e9 / (rounds * rows * count) as f64);
    }

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
        solve_columns(&Lu::new(&plus).unwrap(), &minus)
    }
    fn difference(a: &Mat, b: &Mat) -> f64 {
        a.data.iter().zip(&b.data).fold(0.0, |m, (x, y)| m.max((*x - *y).abs()))
    }

    /// The tests' dense reference of an exponential: e^{tA} by the Taylor polynomial of tA/2^s, of norm ≤ 1 and degree 18
    /// (1/19! = 8.2e-18 is within u, 1/18! is not), squared s times. n³ a product: for references only.
    pub(crate) fn exp(a: &Mat, t: C64) -> Mat {
        let mut parts = 1;
        while (parts as f64) < a.norm1() * t.abs() {
            parts *= 2;
        }
        let x = a.scaled(t / parts as f64);
        let (mut sum, mut term) = (Mat::identity(a.rows), Mat::identity(a.rows));
        for k in 1..=18 {
            term = term.product(&x).scaled(C64::real(1.0 / k as f64));
            sum.add_scaled(C64::ONE, &term);
        }
        while parts > 1 {
            sum = sum.product(&sum);
            parts /= 2;
        }
        sum
    }

    /// LU with partial pivoting is backward stable: the residual of a random 24 × 24 system is within its normwise
    /// backward error bound γ_{3n}·ρ‖A‖‖x‖ (Higham Thm 9.3; the growth ρ of a random matrix below n).
    #[test]
    fn the_lu_solves_within_its_backward_error() {
        let n = 24;
        let (a, b) = (sample(n, n, 1), sample(n, 3, 2));
        let x = solve_columns(&Lu::new(&a).unwrap(), &b);
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
        let a = s.product(&Mat::diagonal(&lambda)).product(&solve_columns(&Lu::new(&s).unwrap(), &Mat::identity(n)));
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

    /// The vector's Chebyshev plan: nothing for a zero matrix; for a Hermitian one a single part, of degree 15 at σ = 1
    /// (2(1 + √2)(½)¹⁶/16! = 3.5e-18 within u, (½)¹⁵/15! not) and about σ beyond it (between σ and 2σ at σ = 100, where
    /// Taylor polynomials on parts of norm one take 1800 products); where the decay rates differ the fewest parts with
    /// σγ/q² ≤ 1; and parts that keep y within ln(f64::MAX) (three at σ = 3000).
    #[test]
    fn the_plan_follows_from_the_round_off() {
        assert_eq!(Chebyshev::plan(0.0, 0.0), Chebyshev { parts: 1, terms: 0 });
        assert_eq!(Chebyshev::plan(1.0, 0.0), Chebyshev { parts: 1, terms: 15 });
        assert!(Chebyshev::plan(1e-3, 0.0).terms <= 5);
        let wide = Chebyshev::plan(100.0, 0.0);
        assert!(wide.parts == 1 && wide.terms > 100 && wide.terms < 200, "{wide:?}");
        assert_eq!(Chebyshev::plan(28.0, 0.3).parts, 3);
        assert_eq!(Chebyshev::plan(0.0, 5.0).parts, 5);
        assert_eq!(Chebyshev::plan(3000.0, 0.0).parts, 3);
    }

    /// The exponential's bounds (`bounds`): of a random matrix X, ρ and γ bound its two Hermitian parts' spectra about
    /// the scalar taken -- against the spectra themselves --; two groups of levels Δ apart, weakly coupled, cost half
    /// their distance and the couplings whatever the groups' sizes (their mean would cost the smaller group Δ times
    /// the larger one's share), a common level nothing; and one state coupled to eight others (norm √8 of a coupling)
    /// its Frobenius norm, 4 of it, where its row's sum is 8.
    #[test]
    fn the_bounds_hold_about_the_spectrum_s_middle() {
        let n = 9;
        let x = sample(n, n, 21);
        let found = bounds(&x, &mut vec![C64::ZERO; n]);
        let (h, k) = (x.hermitian_part(), Mat::from_fn(n, n, |i, j| (x[(i, j)] - x[(j, i)].conj()) * C64::new(0.0, -0.5)));
        for (part, bound, scalar) in [(&h, found.rho, found.scalar.re), (&k, found.gamma, found.scalar.im)] {
            let most = hermitian_eigen(part).values.iter().map(|l| (l - scalar).abs()).fold(0.0, f64::max);
            assert!(most <= bound && bound <= part.norm1() + scalar.abs(), "{most} against the bound {bound}");
        }
        let (delta, coupling, level) = (165.0, 3.0, 4000.0);
        let groups = Mat::from_fn(13, 13, |i, j| match (i < 8, j < 8) {
            (true, true) | (false, false) if i == j => C64::real(level + if i < 8 { 0.0 } else { delta }),
            (true, false) | (false, true) if (i + j) % 5 == 0 => C64::real(coupling),
            _ => C64::ZERO,
        });
        let found = bounds(&groups, &mut vec![C64::ZERO; 13]);
        assert!(found.rho >= 0.5 * delta && found.rho <= 0.5 * delta + 3.0 * coupling && found.gamma == 0.0, "{found:?}");
        assert!((found.scalar - C64::real(level + 0.5 * delta)).abs() <= 3.0 * coupling, "{found:?}");
        let star = Mat::from_fn(n, n, |i, j| if (i == 0) != (j == 0) { C64::real(0.4) } else { C64::ZERO });
        let found = bounds(&star, &mut vec![C64::ZERO; n]);
        assert!((found.rho - 1.6).abs() <= 1e-12 && found.scalar == C64::ZERO, "{found:?}");
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
            exp_apply(&mut h.clone(), C64::new(0.0, -tau), &mut w, &mut vec![C64::ZERO; 2 * n]);
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
            exp_apply(&mut k.clone(), t, &mut a, &mut vec![C64::ZERO; 2 * n]);
            let mut d = e.inverse.apply(&b);
            for (x, l) in d.iter_mut().zip(&e.values) {
                *x = *x * (*l * t).exp();
            }
            b = e.vectors.apply(&d);
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
            let mv: V3 = std::array::from_fn(|i| dot3(m[i], vectors[k]));
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
