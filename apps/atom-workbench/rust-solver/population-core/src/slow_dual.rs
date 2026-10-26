//! Slow-manifold dual for population-error estimation.
//!
//! # Setting
//!
//! The finite-time solver represents the density in the gauged, Hilbert-rotated
//! "frame" coordinates `X(theta, t) = sum_c exp(i c.theta) X_c(t)`, evaluated on
//! the physical characteristic `theta = omega t`. The frame Liouvillian has
//! Fourier charges `L(theta) = sum_p exp(i p.theta) L_p` with
//!
//! ```text
//! L_p X = -i (H_p X - X H_p) - 1/2 (Gamma_p X + X Gamma_p) + R_p(X_EE),
//! ```
//!
//! where `H_p`, `Gamma_p` (excited block only) and the recycling map `R_p`
//! (excited block -> ground block) are the complete prepared coefficients. The
//! Fourier-envelope equation of any field (density, error, auxiliary) is
//!
//! ```text
//! X_c' = -i nu_c X_c + sum_p L_p X_{c-p} + source_c(t),     nu_c = c.omega.
//! ```
//!
//! # Slow manifold
//!
//! All optical (ground-excited) coherences decay at least at `Gamma_min / 2`
//! (~18/us for Rb87) and all excited-block entries at least at `Gamma_min`;
//! only the ground block (populations, Zeeman/hyperfine coherences) evolves
//! slowly. The slow space `P` is therefore the complete 8x8 ground block at
//! the central charge, plus the ground block at every "Raman-slow" charge `c`
//! for which some ground coherence has `|nu_c + eps_a - eps_b| < cutoff`
//! (the gauged hyperfine coherence of a two-photon resonance). Everything else
//! (`Q`) is eliminated adiabatically (Markov Schur complement), one Fourier
//! hop away from `P`:
//!
//! ```text
//! J_eff[c', c] = -i nu_c delta_{c'c} + P L_{c'-c} P
//!              + sum_{c''} P L_{c'-c''} Q R_{c''} Q L_{c''-c} P
//!              + sum_{c'' fast, c''' slow} P L_{c'-c'''} Q R_{c'''} Q L_{c'''-c''} R_{c''} L_{c''-c} P,
//! R_k = (i (nu_k - r_c) - L_0)_{QQ}^{-1}   (exact single-charge inverse).
//! ```
//!
//! The last (two-hop) line is the real-excitation path through a drive
//! charge: ground -> optical coherence at a fast charge -> excited population
//! (or optical coherence) back at a slow charge, which relaxes there and is
//! recycled into the ground block. It is the Neumann term `D_s^-1 O_sf D_f^-1`
//! of the eliminated-space inverse with the fast->slow-charge couplings
//! `O_sf` kept; dropping only fast->fast, slow->fast and slow->slow
//! inter-charge couplings of `Q` keeps the left trace vector of every single
//! slow charge exact (`l_0 J_eff = 0` with one slot), so the population trace
//! is conserved by construction instead of leaking at the pumping rate of the
//! drive-charge couplings.
//!
//! The expansion point `s_c = -i r_c` of column slot `c` is the envelope
//! frequency its slow content actually carries: `r_c = nu_c` when the
//! populations of charge `c` are themselves slow (`|nu_c| < cutoff`; this keeps
//! the charge-`c` trace exactly conserved, `l_c J = -i nu_c l_c` for
//! `l_c = tr` at charge `c` including the slaved excited block), otherwise
//! `r_c = nu_c + eps_a - eps_b` of its slowest (Raman-resonant) coherence. The
//! central slot has `r_0 = 0`, the ordinary zero-frequency Markov limit. The
//! Schur complement is then accurate for the slow content up to
//! `O(|J_eff + i r_c| / gap)`, not merely up to `O(nu_c / gap)`.
//!
//! For a "fast" intermediate charge `c''` (no slow ground coherence) the whole
//! charge-`c''` block is eliminated with the exact structured inverse
//! `(i nu - L_0)^{-1}`: a complex-Schur triangular Sylvester solve for the
//! no-jump part `A_L X + X A_R` plus a rank-`g^2` Woodbury correction for the
//! excited->ground recycling. For a slow charge only the non-ground part is
//! eliminated: the excited block is condensed onto the optical coherences and
//! the resulting dense optical Schur complement is LU factored.
//!
//! No coefficient is dropped because it is zero or small; all choices above
//! are made from magnitudes (decay rates, frequency separations).
//!
//! # Use
//!
//! `J_eff` is the generator of the error's slow component. With a residual
//! source `s(t)` (see `error_estimate.rs`) the slow error is
//! `e_P(t) = exp(J t) e_P(0) + int_0^t exp(J (t-s)) s(s) ds`, evaluated exactly in
//! the eigenbasis of `J` (entire-function formulas for near-resonant terms).
//! Populations are read out through the Hilbert basis `B`
//! (`p_i = sum_ab B_ia X_ab conj(B_ib)`; the diagonal gauge cancels). Excited
//! populations are read out from the adiabatically slaved excited block.
//!
//! # Weak couplings
//!
//! An operator charge whose couplings are all below `STRUCTURAL_COUPLING_FLOOR`
//! of the largest coupling of their family (Hamiltonian, damping, recycling)
//! is weak. Exactly as in the preparation, a weak coupling never shapes the
//! structure: the slots are the slow charges two *significant* couplings
//! reach, so a forbidden transition perturbed to 1e-7 of the drive does not
//! triple the slow dimension. Weak couplings still act exactly in every
//! elimination, slaving, return and residual (a halo mode that returns only
//! through weak couplings contributes just those returns, see
//! `error_estimate.rs`). The slow charges that only a
//! path through a weak coupling reaches stay unrepresented: their ground
//! block is held out of the eliminations (it is slow, not fast), and the
//! couplings `K` from the slow columns into it are collected while `J_eff` is
//! assembled. Over a horizon `T` such a charge gains at most `T ||K|| ||e_P||`
//! from the slow error, which every estimate charges (`error_estimate.rs`).
use super::*;
use crate::preparation::STRUCTURAL_COUPLING_FLOOR;

// ---------------------------------------------------------------------------
// Small complex helpers
// ---------------------------------------------------------------------------

#[inline]
pub(crate) fn cadd(a: C, b: C) -> C { C { re: a.re + b.re, im: a.im + b.im } }
#[inline]
pub(crate) fn csub(a: C, b: C) -> C { C { re: a.re - b.re, im: a.im - b.im } }
#[inline]
pub(crate) fn cmul(a: C, b: C) -> C { C { re: a.re * b.re - a.im * b.im, im: a.re * b.im + a.im * b.re } }
#[inline]
pub(crate) fn cconj(a: C) -> C { C { re: a.re, im: -a.im } }
#[inline]
pub(crate) fn cscale(a: C, s: f64) -> C { C { re: a.re * s, im: a.im * s } }
#[inline]
pub(crate) fn cabs(a: C) -> f64 { a.re.hypot(a.im) }
#[inline]
pub(crate) fn cdiv(a: C, b: C) -> C {
    // Smith's algorithm keeps the quotient finite for widely scaled operands.
    if b.re.abs() >= b.im.abs() {
        let r = b.im / b.re;
        let d = b.re + b.im * r;
        C { re: (a.re + a.im * r) / d, im: (a.im - a.re * r) / d }
    } else {
        let r = b.re / b.im;
        let d = b.re * r + b.im;
        C { re: (a.re * r + a.im) / d, im: (a.im * r - a.re) / d }
    }
}
#[inline]
pub(crate) fn cexp(a: C) -> C {
    let m = a.re.exp();
    let (s, c) = a.im.sin_cos();
    C { re: m * c, im: m * s }
}
#[inline]
pub(crate) fn complex(re: f64, im: f64) -> C { C { re, im } }
pub(crate) const ZERO: C = C { re: 0., im: 0. };
/// `exp(x)` with `x` below this is negligible (< 1e-18) next to O(1) terms.
pub(crate) const NEGLIGIBLE_EXPONENT: f64 = -41.4;
pub(crate) const ONE: C = C { re: 1., im: 0. };
#[inline]
fn minus_i(a: C) -> C { C { re: a.im, im: -a.re } }

/// `(exp(z) - 1) / z`, entire; the series branch is used for `|z| <= 1`.
pub(crate) fn exprel(z: C) -> C {
    if cabs(z) > 1. {
        return cdiv(csub(cexp(z), ONE), z);
    }
    // sum_{k>=0} z^k / (k+1)!; 22 terms leave a tail below 1e-21 for |z|<=1.
    let mut term = ONE;
    let mut sum = ONE;
    for k in 1..22 {
        term = cscale(cmul(term, z), 1. / (k as f64 + 1.));
        sum = cadd(sum, term);
    }
    sum
}

/// First divided difference of `x -> exp(x t)`: `(exp(x t) - exp(y t)) / (x - y)`.
pub(crate) fn exp_divided_2(x: C, y: C, t: f64) -> C {
    let delta = csub(x, y);
    if cabs(delta) * t <= 1. {
        cscale(cmul(cexp(cscale(y, t)), exprel(cscale(delta, t))), t)
    } else {
        cdiv(csub(cexp(cscale(x, t)), cexp(cscale(y, t))), delta)
    }
}

/// Second divided difference of `x -> exp(x t)` at three nodes. Separated
/// nodes use the recursive definition; a fully clustered triple uses the
/// Opitz formula (exponential of the 3x3 bidiagonal node matrix).
pub(crate) fn exp_divided_3(x: C, y: C, z: C, t: f64) -> C {
    let far = |a: C, b: C| cabs(csub(a, b)) * t > 1.;
    if far(x, z) {
        return cdiv(csub(exp_divided_2(x, y, t), exp_divided_2(y, z, t)), csub(x, z));
    }
    if far(y, z) {
        return cdiv(csub(exp_divided_2(y, x, t), exp_divided_2(x, z, t)), csub(y, z));
    }
    if far(x, y) {
        return cdiv(csub(exp_divided_2(x, z, t), exp_divided_2(z, y, t)), csub(x, y));
    }
    opitz_3(x, y, z, t)
}

/// `exp(t Z)[0, 2]` for `Z = [[x, 1, 0], [0, y, 1], [0, 0, z]]`.
fn opitz_3(x: C, y: C, z: C, t: f64) -> C {
    // Shift by the mean node so the remaining matrix is small, then Taylor.
    let mean = cscale(cadd(cadd(x, y), z), 1. / 3.);
    let mut m = [[ZERO; 3]; 3];
    m[0][0] = cscale(csub(x, mean), t);
    m[1][1] = cscale(csub(y, mean), t);
    m[2][2] = cscale(csub(z, mean), t);
    m[0][1] = complex(t, 0.);
    m[1][2] = complex(t, 0.);
    // The diagonal is now O(1); the superdiagonal t enters only polynomially
    // (nilpotent part), so a plain Taylor series with squaring is stable.
    let norm = (0..3).map(|i| (0..3).map(|j| cabs(m[i][j])).sum::<f64>()).fold(0_f64, f64::max);
    let squarings = if norm > 0.5 { (norm / 0.5).log2().ceil() as i32 } else { 0 };
    let scale = 0.5_f64.powi(squarings);
    for row in m.iter_mut() {
        for v in row.iter_mut() {
            *v = cscale(*v, scale);
        }
    }
    let mut result = [[ZERO; 3]; 3];
    let mut term = [[ZERO; 3]; 3];
    for i in 0..3 {
        result[i][i] = ONE;
        term[i][i] = ONE;
    }
    for k in 1..30 {
        let mut next = [[ZERO; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                let mut s = ZERO;
                for l in 0..3 {
                    s = cadd(s, cmul(term[i][l], m[l][j]));
                }
                next[i][j] = cscale(s, 1. / k as f64);
            }
        }
        term = next;
        for i in 0..3 {
            for j in 0..3 {
                result[i][j] = cadd(result[i][j], term[i][j]);
            }
        }
    }
    for _ in 0..squarings {
        let mut squared = [[ZERO; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                let mut s = ZERO;
                for l in 0..3 {
                    s = cadd(s, cmul(result[i][l], result[l][j]));
                }
                squared[i][j] = s;
            }
        }
        result = squared;
    }
    // exp(t Z) = exp(t mean) exp(t (Z - mean)); the (0,2) entry carries the
    // divided difference f[x,y,z] with f(u) = exp(u t) (superdiagonal = 1, the
    // t factors are already inside the shifted matrix).
    cmul(cexp(cscale(mean, t)), result[0][2])
}

// ---------------------------------------------------------------------------
// Dense column-major complex matrices
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub(crate) struct Dense {
    pub rows: usize,
    pub cols: usize,
    /// Column-major: entry (i, j) is `data[i + j * rows]`.
    pub data: Vec<C>,
}

impl Dense {
    pub fn zeros(rows: usize, cols: usize) -> Self { Self { rows, cols, data: vec![ZERO; rows * cols] } }
    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.data[i + i * n] = ONE;
        }
        m
    }
    pub fn from_row_major(values: &[C], rows: usize, cols: usize) -> Self {
        let mut m = Self::zeros(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                m.data[i + j * rows] = values[i * cols + j];
            }
        }
        m
    }
    #[inline]
    pub fn at(&self, i: usize, j: usize) -> C { self.data[i + j * self.rows] }
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, v: C) { self.data[i + j * self.rows] = v; }
    #[inline]
    pub fn add_at(&mut self, i: usize, j: usize, v: C) {
        let x = &mut self.data[i + j * self.rows];
        x.re += v.re;
        x.im += v.im;
    }
    pub fn adjoint(&self) -> Self {
        let mut m = Self::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                m.data[j + i * self.cols] = cconj(self.at(i, j));
            }
        }
        m
    }
    pub fn block(&self, rows: std::ops::Range<usize>, cols: std::ops::Range<usize>) -> Self {
        let mut m = Self::zeros(rows.len(), cols.len());
        for (jj, j) in cols.clone().enumerate() {
            for (ii, i) in rows.clone().enumerate() {
                m.data[ii + jj * m.rows] = self.at(i, j);
            }
        }
        m
    }
    pub fn multiply(&self, other: &Self, lap: &Lapack) -> Result<Self, String> {
        if self.cols != other.rows {
            return Err("Dense product shape mismatch".into());
        }
        if self.rows == 0 || other.cols == 0 {
            return Ok(Self::zeros(self.rows, other.cols));
        }
        if self.cols == 0 {
            return Ok(Self::zeros(self.rows, other.cols));
        }
        let mut out = Self::zeros(self.rows, other.cols);
        lap.multiply_complex_into(&self.data, &other.data, &mut out.data, self.rows, other.cols, self.cols)?;
        Ok(out)
    }
    pub fn frobenius(&self) -> f64 { self.data.iter().fold(0_f64, |s, z| s.hypot(cabs(*z))) }
    pub fn is_finite(&self) -> bool { self.data.iter().all(|z| z.re.is_finite() && z.im.is_finite()) }
}

/// `[X_1 | X_2 | ...]`: square column-major blocks side by side are just their
/// concatenated storage.
fn side_by_side(xs: &[&Dense], n: usize) -> Dense {
    let mut out = Dense::zeros(n, n * xs.len());
    for (j, x) in xs.iter().enumerate() {
        out.data[j * n * n..(j + 1) * n * n].copy_from_slice(&x.data);
    }
    out
}

/// `[X_1; X_2; ...]` from side-by-side blocks (element `(j n + a, b)`).
fn stacked(side: &Dense, n: usize, m: usize) -> Dense {
    let mut out = Dense::zeros(n * m, n);
    for j in 0..m {
        for b in 0..n {
            for a in 0..n {
                out.data[j * n + a + b * n * m] = side.data[j * n * n + a + b * n];
            }
        }
    }
    out
}

/// Block `j` of a stacked `[Y_1; Y_2; ...]`.
fn stacked_block(stack: &Dense, n: usize, m: usize, j: usize) -> Dense {
    let mut out = Dense::zeros(n, n);
    for b in 0..n {
        for a in 0..n {
            out.data[a + b * n] = stack.data[j * n + a + b * n * m];
        }
    }
    out
}

/// `L X_j R` for every block with two products instead of `2 m`.
pub(crate) fn sandwich_all(left: &Dense, right: &Dense, xs: &[&Dense], lap: &Lapack) -> Result<Vec<Dense>, String> {
    let n = left.rows;
    let m = xs.len();
    if m == 0 {
        return Ok(Vec::new());
    }
    let product = left.multiply(&side_by_side(xs, n), lap)?;
    let result = stacked(&product, n, m).multiply(right, lap)?;
    Ok((0..m).map(|j| stacked_block(&result, n, m, j)).collect())
}

/// LU factorization of a square complex matrix with an inexpensive
/// reciprocal-condition diagnostic (1-norm of A times 1-norm of A^{-1}
/// estimated on a few probe vectors).
pub(crate) struct LuFactor {
    n: usize,
    lu: Vec<C>,
    pivots: Vec<i32>,
}

impl LuFactor {
    pub fn new(a: &Dense, lap: &Lapack) -> Result<Self, String> {
        if a.rows != a.cols || a.rows == 0 {
            return Err("LU requires a nonempty square matrix".into());
        }
        let (lu, pivots) = lap.complex_factor(&a.data, a.rows)?;
        Ok(Self { n: a.rows, lu, pivots })
    }
    pub fn solve(&self, rhs: &[C], lap: &Lapack) -> Result<Vec<C>, String> {
        lap.complex_solve_factored(&self.lu, &self.pivots, rhs, self.n)
    }
    pub fn solve_matrix(&self, rhs: &Dense, lap: &Lapack) -> Result<Dense, String> {
        let mut out = Dense::zeros(rhs.rows, rhs.cols);
        for j in 0..rhs.cols {
            let column = self.solve(&rhs.data[j * rhs.rows..(j + 1) * rhs.rows], lap)?;
            out.data[j * rhs.rows..(j + 1) * rhs.rows].copy_from_slice(&column);
        }
        Ok(out)
    }
    pub fn inverse(&self, lap: &Lapack) -> Result<Dense, String> { self.solve_matrix(&Dense::identity(self.n), lap) }
}

fn one_norm(a: &Dense) -> f64 {
    (0..a.cols).map(|j| (0..a.rows).map(|i| cabs(a.at(i, j))).sum::<f64>()).fold(0_f64, f64::max)
}

// ---------------------------------------------------------------------------
// Triangular Sylvester solves  z X - A_L X - X A_R = Y
// ---------------------------------------------------------------------------

/// Complex Schur factors `A = U T U^H` of a square matrix.
#[derive(Clone)]
pub(crate) struct Schur {
    pub t: Dense,
    pub u: Dense,
    pub uh: Dense,
}

impl Schur {
    pub fn new(a: &Dense, lap: &Lapack) -> Result<Self, String> {
        if a.rows != a.cols {
            return Err("Schur factor requires a square matrix".into());
        }
        if a.rows == 0 {
            return Ok(Self { t: Dense::zeros(0, 0), u: Dense::zeros(0, 0), uh: Dense::zeros(0, 0) });
        }
        let (t, q) = lap.complex_schur(&a.data, a.rows)?;
        let t = Dense { rows: a.rows, cols: a.rows, data: t };
        let u = Dense { rows: a.rows, cols: a.rows, data: q };
        let uh = u.adjoint();
        Ok(Self { t, u, uh })
    }
    pub fn dimension(&self) -> usize { self.t.rows }
}

// ---------------------------------------------------------------------------
// Qualified eigen-decompositions that never depend on one QR run
// ---------------------------------------------------------------------------

/// `A = V diag(values) V^-1`, qualified by its residual.
pub(crate) struct Eigen {
    pub values: Vec<C>,
    pub vectors: Dense,
    pub inverse: Dense,
    /// `||V||_1 ||V^-1||_1`.
    pub condition: f64,
    /// `||A V - V diag(values)||_F / (||A||_F ||V||_F)`.
    pub residual: f64,
    /// Which attempt produced the decomposition (diagnostic).
    pub method: &'static str,
}

/// Residual bound accepted without looking further.
const EIGEN_RESIDUAL: f64 = 1e-10;
/// Residual bound accepted when no attempt reaches `EIGEN_RESIDUAL`.
const EIGEN_RESIDUAL_LAST: f64 = 1e-7;

/// Eigen-decomposition of a square matrix that survives a non-converging
/// Hessenberg QR. Every attempt is an exact transformation of the same
/// problem, so the first one whose residual qualifies is used:
///
/// 1. `zgeev` of `A + i s I` (a purely imaginary shift, `s ~ ||A||`): the
///    eigenvectors are unchanged and the eigenvalues move by exactly `i s`,
///    but the iteration no longer runs on a real matrix stored as complex,
///    whose conjugate-pair stagnation is what fails in practice. It comes
///    first because a failing plain `zgeev` is slow (seconds on m = 40: it
///    exhausts its iteration limit) while the shifted one converges at once;
/// 2. `zgeev` of `A`;
/// 3. `dgeev` of `A` when `A` is real (real double-shift QR);
/// 4. `zgeev` of `Q A Q^T` with a fixed pseudo-random orthogonal `Q` (a
///    different Hessenberg form), mapped back with `V = Q^T V'`;
/// 5. the complex Schur form `A = U T U^H` with eigenvectors from triangular
///    back substitution.
pub(crate) fn robust_eigen(a: &Dense, real: Option<&[f64]>, lap: &Lapack) -> Result<Eigen, String> {
    robust_eigen_from(a, real, lap, 0)
}

/// `robust_eigen` starting at attempt `first` (tests exercise each path).
pub(crate) fn robust_eigen_from(a: &Dense, real: Option<&[f64]>, lap: &Lapack, first: usize) -> Result<Eigen, String> {
    let n = a.rows;
    if n == 0 || a.cols != n || !a.is_finite() {
        return Err("Eigen-decomposition needs a finite nonempty square matrix".into());
    }
    let norm = a.frobenius().max(f64::MIN_POSITIVE);
    let mut best: Option<Eigen> = None;
    let mut failures: Vec<String> = Vec::new();
    for attempt in first..5 {
        let raw: Result<(Vec<C>, Dense, &'static str), String> = match attempt {
            1 => lap.complex_eigen_right(&a.data, n).map(|(w, v)| (w, Dense { rows: n, cols: n, data: v }, "zgeev")),
            0 => {
                let shift = 0.318_309_886 * norm;
                let mut shifted = a.clone();
                for i in 0..n {
                    shifted.add_at(i, i, complex(0., shift));
                }
                lap.complex_eigen_right(&shifted.data, n).map(|(w, v)| {
                    let w = w.into_iter().map(|z| csub(z, complex(0., shift))).collect();
                    (w, Dense { rows: n, cols: n, data: v }, "zgeev-imaginary-shift")
                })
            }
            2 => match real {
                Some(values) if values.len() == n * n => real_eigen(values, n, lap).map(|(w, v)| (w, v, "dgeev")),
                _ => continue,
            },
            3 => {
                let q = pseudo_random_orthogonal(n);
                let rotated = q.multiply(a, lap).and_then(|qa| qa.multiply(&q.adjoint(), lap));
                rotated.and_then(|r| lap.complex_eigen_right(&r.data, n)).and_then(|(w, v)| {
                    let v = q.adjoint().multiply(&Dense { rows: n, cols: n, data: v }, lap)?;
                    Ok((w, v, "zgeev-similarity"))
                })
            }
            _ => schur_eigen(a, lap).map(|(w, v)| (w, v, "schur")),
        };
        let (values, vectors, method) = match raw {
            Ok(r) => r,
            Err(e) => {
                failures.push(e);
                continue;
            }
        };
        match qualify(a, values, vectors, method, norm, lap) {
            Ok(eigen) if eigen.residual <= EIGEN_RESIDUAL => return Ok(eigen),
            Ok(eigen) => {
                if best.as_ref().is_none_or(|b| eigen.residual < b.residual) {
                    best = Some(eigen);
                }
            }
            Err(e) => failures.push(e),
        }
    }
    match best {
        Some(eigen) if eigen.residual <= EIGEN_RESIDUAL_LAST => Ok(eigen),
        Some(eigen) => Err(format!("Eigen-decomposition unresolved: best residual {:e} ({})", eigen.residual, eigen.method)),
        None => Err(format!("Eigen-decomposition unresolved: {}", failures.join("; "))),
    }
}

fn qualify(a: &Dense, values: Vec<C>, mut vectors: Dense, method: &'static str, norm: f64, lap: &Lapack) -> Result<Eigen, String> {
    let n = a.rows;
    if values.len() != n || !vectors.is_finite() || values.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) {
        return Err(format!("{method}: nonfinite eigenpairs"));
    }
    // Unit columns keep the inverse well scaled.
    for j in 0..n {
        let length = (0..n).fold(0_f64, |s, i| s.hypot(cabs(vectors.at(i, j))));
        if !(length > 0.) {
            return Err(format!("{method}: zero eigenvector"));
        }
        for i in 0..n {
            vectors.set(i, j, cscale(vectors.at(i, j), 1. / length));
        }
    }
    let inverse = LuFactor::new(&vectors, lap)?.inverse(lap)?;
    let condition = one_norm(&vectors) * one_norm(&inverse);
    let image = a.multiply(&vectors, lap)?;
    let mut defect = 0_f64;
    for j in 0..n {
        for i in 0..n {
            defect = defect.hypot(cabs(csub(image.at(i, j), cmul(vectors.at(i, j), values[j]))));
        }
    }
    let residual = defect / (norm * (n as f64).sqrt());
    if !(condition.is_finite() && residual.is_finite() && inverse.is_finite()) {
        return Err(format!("{method}: singular eigenvector basis"));
    }
    Ok(Eigen { values, vectors, inverse, condition, residual, method })
}

/// Real `dgeev`, unpacked to complex pairs (`v_j = vr_j + i vr_{j+1}`).
fn real_eigen(values: &[f64], n: usize, lap: &Lapack) -> Result<(Vec<C>, Dense), String> {
    let (wr, wi, vr) = lap.eigen(values, n)?;
    let mut w = Vec::with_capacity(n);
    let mut v = Dense::zeros(n, n);
    let mut j = 0;
    while j < n {
        if wi[j] != 0. && j + 1 < n {
            for i in 0..n {
                let (re, im) = (vr[i + j * n], vr[i + (j + 1) * n]);
                v.set(i, j, complex(re, im));
                v.set(i, j + 1, complex(re, -im));
            }
            w.push(complex(wr[j], wi[j]));
            w.push(complex(wr[j + 1], wi[j + 1]));
            j += 2;
        } else {
            for i in 0..n {
                v.set(i, j, complex(vr[i + j * n], 0.));
            }
            w.push(complex(wr[j], wi[j]));
            j += 1;
        }
    }
    Ok((w, v))
}

/// Eigenpairs from the complex Schur form: `T x = t_kk x` by back
/// substitution (`x_k = 1`), with a relative floor on the denominators so a
/// repeated eigenvalue yields a finite (ill-conditioned) vector, then `U x`.
fn schur_eigen(a: &Dense, lap: &Lapack) -> Result<(Vec<C>, Dense), String> {
    let n = a.rows;
    let schur = Schur::new(a, lap)?;
    let t = &schur.t;
    let small = f64::EPSILON * t.frobenius().max(f64::MIN_POSITIVE);
    let values: Vec<C> = (0..n).map(|k| t.at(k, k)).collect();
    let mut x = Dense::zeros(n, n);
    for k in 0..n {
        x.set(k, k, ONE);
        for i in (0..k).rev() {
            let mut sum = ZERO;
            for j in i + 1..=k {
                sum = cadd(sum, cmul(t.at(i, j), x.at(j, k)));
            }
            let mut denominator = csub(values[k], t.at(i, i));
            if cabs(denominator) < small {
                denominator = complex(small, 0.);
            }
            x.set(i, k, cdiv(sum, denominator));
        }
    }
    Ok((values, schur.u.multiply(&x, lap)?))
}

/// A fixed, well-mixed orthogonal matrix: the product of three Householder
/// reflectors with pseudo-random (deterministic) vectors.
fn pseudo_random_orthogonal(n: usize) -> Dense {
    let mut q = Dense::identity(n);
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    for _ in 0..3 {
        let v: Vec<f64> = (0..n)
            .map(|_| {
                state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                ((state >> 11) as f64 / (1u64 << 53) as f64) - 0.5
            })
            .collect();
        let vv: f64 = v.iter().map(|x| x * x).sum();
        if !(vv > 0.) {
            continue;
        }
        // q <- (I - 2 v v^T / v^T v) q
        for j in 0..n {
            let dot = (0..n).fold(ZERO, |s, i| cadd(s, cscale(q.at(i, j), v[i])));
            for i in 0..n {
                let value = csub(q.at(i, j), cscale(dot, 2. * v[i] / vv));
                q.set(i, j, value);
            }
        }
    }
    q
}

/// Solver for `z X - A_L X - X A_R = Y` with `X` of shape `p x r`, using the
/// Schur forms of `A_L` (p x p) and `A_R` (r x r). In Schur coordinates
/// `X = U_L Xt U_R^H` the equation becomes `z Xt - T_L Xt - Xt T_R = Yt` with
/// both `T` upper triangular; it is solved column by column (forward in the
/// column index because `T_R` is upper triangular) and by back substitution
/// within each column.
#[derive(Clone)]
pub(crate) struct SylvesterFactor {
    pub left: Schur,
    pub right: Schur,
}

impl SylvesterFactor {
    /// Smallest `|z - T_L[i,i] - T_R[j,j]|`, i.e. the smallest modulus of the
    /// Sylvester operator's eigenvalues at shift `z`.
    pub fn minimum_denominator(&self, z: C) -> f64 {
        let mut smallest = f64::INFINITY;
        for i in 0..self.left.dimension() {
            for j in 0..self.right.dimension() {
                let d = csub(csub(z, self.left.t.at(i, i)), self.right.t.at(j, j));
                smallest = smallest.min(cabs(d));
            }
        }
        smallest
    }
    pub fn solve(&self, z: C, y: &Dense, lap: &Lapack) -> Result<Dense, String> {
        let p = self.left.dimension();
        let r = self.right.dimension();
        if y.rows != p || y.cols != r {
            return Err("Sylvester right-hand side shape mismatch".into());
        }
        if p == 0 || r == 0 {
            return Ok(Dense::zeros(p, r));
        }
        let yt = self.left.uh.multiply(y, lap)?.multiply(&self.right.u, lap)?;
        let x = self.triangular(z, &yt)?;
        let out = self.left.u.multiply(&x, lap)?.multiply(&self.right.uh, lap)?;
        if !out.is_finite() {
            return Err("Nonfinite Sylvester solution".into());
        }
        Ok(out)
    }

    /// `solve` for many square right-hand sides (p = r), with the Schur-basis
    /// changes of all of them in two products each way.
    pub fn solve_all(&self, z: C, ys: &[&Dense], lap: &Lapack) -> Result<Vec<Dense>, String> {
        if ys.is_empty() {
            return Ok(Vec::new());
        }
        if self.left.dimension() != self.right.dimension() || ys.iter().any(|y| y.rows != self.left.dimension() || y.cols != self.right.dimension()) {
            return Err("Batched Sylvester solves need square right-hand sides".into());
        }
        let transformed = sandwich_all(&self.left.uh, &self.right.u, ys, lap)?;
        let solved: Vec<Dense> = transformed.iter().map(|yt| self.triangular(z, yt)).collect::<Result<_, _>>()?;
        let refs: Vec<&Dense> = solved.iter().collect();
        let out = sandwich_all(&self.left.u, &self.right.uh, &refs, lap)?;
        if out.iter().any(|x| !x.is_finite()) {
            return Err("Nonfinite Sylvester solution".into());
        }
        Ok(out)
    }

    /// The triangular core in Schur coordinates.
    fn triangular(&self, z: C, yt: &Dense) -> Result<Dense, String> {
        let p = self.left.dimension();
        let r = self.right.dimension();
        let tl = &self.left.t;
        let tr = &self.right.t;
        let mut x = Dense::zeros(p, r);
        let mut rhs = vec![ZERO; p];
        for j in 0..r {
            // Column j: (z - T_R[j,j] - T_L) x_j = yt_j + sum_{k<j} T_R[k,j] x_k.
            for i in 0..p {
                let mut v = yt.at(i, j);
                for k in 0..j {
                    v = cadd(v, cmul(tr.at(k, j), x.at(i, k)));
                }
                rhs[i] = v;
            }
            let shift = csub(z, tr.at(j, j));
            for i in (0..p).rev() {
                let mut v = rhs[i];
                for l in i + 1..p {
                    v = cadd(v, cmul(tl.at(i, l), x.at(l, j)));
                }
                let denominator = csub(shift, tl.at(i, i));
                if cabs(denominator) == 0. {
                    return Err("Singular triangular Sylvester denominator".into());
                }
                x.set(i, j, cdiv(v, denominator));
            }
        }
        Ok(x)
    }
}

// ---------------------------------------------------------------------------
// Frame Liouvillian
// ---------------------------------------------------------------------------

/// One Fourier charge `p` of the complete frame Liouvillian.
pub(crate) struct ChargeTerm {
    pub charge: Charge,
    /// `H_p`, n x n (absent when the model has no Hamiltonian at this charge).
    pub hamiltonian: Option<Dense>,
    /// `Gamma_p` embedded in the excited block, n x n.
    pub gamma: Option<Dense>,
    /// Recycling `R_p`: (ground entry a*g+b, excited entry i*e+j, coefficient).
    pub recycling: Vec<(usize, usize, C)>,
    /// `R_p` as a dense `g^2 x e^2` matrix (rows a*g+b, columns i*e+j) when
    /// at least a third of it is stored: a perturbed Hilbert basis fills every
    /// entry, and one matrix product then costs what the entries of the
    /// unperturbed (sparse) map cost one by one.
    pub recycling_matrix: Option<Dense>,
    /// Stored entries `(row, column, value)` of `H_p` and `Gamma_p` when they
    /// are sparse enough for entrywise products (dipole couplings are).
    pub hamiltonian_entries: Option<Vec<(usize, usize, C)>>,
    pub gamma_entries: Option<Vec<(usize, usize, C)>>,
    /// False for a weak term (module docs): every coupling below
    /// `STRUCTURAL_COUPLING_FLOOR` of the largest of its family. The central
    /// term is always significant.
    pub significant: bool,
}

/// Stored entries of a matrix that is sparse enough (under a third of its
/// storage) for entrywise products to beat dense ones.
fn sparse_entries(a: &Dense) -> Option<Vec<(usize, usize, C)>> {
    let mut entries = Vec::new();
    for j in 0..a.cols {
        for i in 0..a.rows {
            let v = a.at(i, j);
            if v.re != 0. || v.im != 0. {
                entries.push((i, j, v));
            }
        }
    }
    (3 * entries.len() < a.rows * a.cols).then_some(entries)
}

/// Largest entry of a frame term in each family: Hamiltonian, damping,
/// recycling. At the central charge the Hamiltonian diagonal (the energies)
/// is not a coupling and is left out, as in the preparation's
/// `largest_coupling`.
fn coupling_sizes(term: &ChargeTerm, central: bool) -> [f64; 3] {
    let largest = |m: &Option<Dense>, skip_diagonal: bool| -> f64 {
        m.as_ref().map_or(0., |m| {
            let mut x = 0_f64;
            for j in 0..m.cols {
                for i in 0..m.rows {
                    if !(skip_diagonal && i == j) {
                        x = x.max(cabs(m.at(i, j)));
                    }
                }
            }
            x
        })
    };
    let recycling = term.recycling.iter().map(|&(_, _, z)| cabs(z)).fold(0_f64, f64::max);
    [largest(&term.hamiltonian, central), largest(&term.gamma, false), recycling]
}

pub(crate) struct FrameLiouvillian {
    pub n: usize,
    pub g: usize,
    pub e: usize,
    pub terms: Vec<ChargeTerm>,
    pub axes: usize,
}

impl FrameLiouvillian {
    pub fn from_primitives(p: &PrimitivePreparation) -> Result<Self, String> {
        let source = &p.hilbert.source;
        let n = source.n;
        let g = source.ground.len();
        let e = n - g;
        if source.ground != (0..g).collect::<Vec<_>>() || source.excited != (g..n).collect::<Vec<_>>() {
            return Err("Frame coordinates must list ground states first".into());
        }
        let axes = source.phases.axes.len();
        let mut charges: BTreeSet<Charge> = BTreeSet::new();
        charges.extend(source.hamiltonian.keys().cloned());
        charges.extend(p.hilbert.radiation.gamma.keys().cloned());
        charges.extend(p.hilbert.radiation.recycling.keys().cloned());
        charges.insert(vec![0; axes]);
        let mut terms = Vec::new();
        for charge in charges {
            let hamiltonian = source.hamiltonian.get(&charge).map(|h| Dense::from_row_major(h, n, n));
            let gamma = p.hilbert.radiation.gamma.get(&charge).map(|block| {
                let mut full = Dense::zeros(n, n);
                for i in 0..e {
                    for j in 0..e {
                        full.set(g + i, g + j, block[i * e + j]);
                    }
                }
                full
            });
            let mut recycling = Vec::new();
            if let Some(r) = p.hilbert.radiation.recycling.get(&charge) {
                for ab in 0..g * g {
                    for ij in 0..e * e {
                        let v = r[ab * e * e + ij];
                        // Storage-zero skipping only; every represented entry is kept.
                        if v.re != 0. || v.im != 0. {
                            recycling.push((ab, ij, v));
                        }
                    }
                }
            }
            let recycling_matrix = (!recycling.is_empty() && 3 * recycling.len() >= g * g * e * e).then(|| {
                let mut r = Dense::zeros(g * g, e * e);
                for &(ab, ij, v) in &recycling {
                    r.set(ab, ij, v);
                }
                r
            });
            let hamiltonian_entries = hamiltonian.as_ref().and_then(sparse_entries);
            let gamma_entries = gamma.as_ref().and_then(sparse_entries);
            terms.push(ChargeTerm { charge, hamiltonian, gamma, recycling, recycling_matrix, hamiltonian_entries, gamma_entries, significant: true });
        }
        let sizes: Vec<[f64; 3]> = terms.iter().map(|t| coupling_sizes(t, t.charge.iter().all(|&v| v == 0))).collect();
        let reference: Vec<f64> = (0..3).map(|k| sizes.iter().map(|s| s[k]).fold(0_f64, f64::max)).collect();
        for (term, size) in terms.iter_mut().zip(&sizes) {
            let central = term.charge.iter().all(|&v| v == 0);
            term.significant = central || (0..3).any(|k| reference[k] > 0. && size[k] >= STRUCTURAL_COUPLING_FLOOR * reference[k]);
        }
        Ok(Self { n, g, e, terms, axes })
    }

    pub fn zero_charge(&self) -> Charge { vec![0; self.axes] }

    pub fn term(&self, charge: &[i32]) -> Option<&ChargeTerm> { self.terms.iter().find(|t| t.charge == charge) }

    /// Complete `L_p X_j` for many blocks: the left products `H X_j` share one
    /// product with the side-by-side blocks, the right products `X_j H` one
    /// with the stacked blocks.
    pub fn apply_all(&self, term: &ChargeTerm, xs: &[&Dense], lap: &Lapack) -> Result<Vec<Dense>, String> {
        let n = self.n;
        let m = xs.len();
        let mut out: Vec<Dense> = (0..m).map(|_| Dense::zeros(n, n)).collect();
        if m == 0 {
            return Ok(out);
        }
        // Sparse terms: (A X)[r, b] += A[r, k] X[k, b] and (X A)[r, b] +=
        // X[r, k] A[k, b] entry by entry, scaled by `left` and `right`.
        let sparse = |entries: &[(usize, usize, C)], left: C, right: C, out: &mut Vec<Dense>| {
            for (x, o) in xs.iter().zip(out.iter_mut()) {
                for &(r, k, v) in entries {
                    let (lv, rv) = (cmul(left, v), cmul(right, v));
                    for b in 0..n {
                        // left product: row r of the output, column b
                        o.data[r + b * n] = cadd(o.data[r + b * n], cmul(lv, x.data[k + b * n]));
                    }
                    for a in 0..n {
                        // right product: X[a, r] A[r, k] lands at (a, k)
                        o.data[a + k * n] = cadd(o.data[a + k * n], cmul(rv, x.data[a + r * n]));
                    }
                }
            }
        };
        let dense_needed = (term.hamiltonian.is_some() && term.hamiltonian_entries.is_none()) || (term.gamma.is_some() && term.gamma_entries.is_none());
        if !dense_needed {
            // -i (H X - X H) and -1/2 (Gamma X + X Gamma)
            if let Some(entries) = &term.hamiltonian_entries {
                sparse(entries, complex(0., -1.), complex(0., 1.), &mut out);
            }
            if let Some(entries) = &term.gamma_entries {
                sparse(entries, complex(-0.5, 0.), complex(-0.5, 0.), &mut out);
            }
            self.add_recycling(term, xs, &mut out, lap)?;
            return Ok(out);
        }
        let side = side_by_side(xs, n);
        let stack = stacked(&side, n, m);
        let both = |a: &Dense, f: &dyn Fn(C, C) -> C, out: &mut Vec<Dense>| -> Result<(), String> {
            let left = a.multiply(&side, lap)?;
            let right = stack.multiply(a, lap)?;
            for (j, o) in out.iter_mut().enumerate() {
                for b in 0..n {
                    for r in 0..n {
                        let l = left.data[j * n * n + r + b * n];
                        let q = right.data[j * n + r + b * n * m];
                        o.data[r + b * n] = cadd(o.data[r + b * n], f(l, q));
                    }
                }
            }
            Ok(())
        };
        if let Some(h) = &term.hamiltonian {
            both(h, &|l, r| minus_i(csub(l, r)), &mut out)?;
        }
        if let Some(gm) = &term.gamma {
            both(gm, &|l, r| cscale(cadd(l, r), -0.5), &mut out)?;
        }
        self.add_recycling(term, xs, &mut out, lap)?;
        Ok(out)
    }

    /// Complete `L_p X` for one charge term.
    pub fn apply(&self, term: &ChargeTerm, x: &Dense, lap: &Lapack) -> Result<Dense, String> {
        let n = self.n;
        let mut out = Dense::zeros(n, n);
        if let Some(h) = &term.hamiltonian {
            let hx = h.multiply(x, lap)?;
            let xh = x.multiply(h, lap)?;
            for k in 0..n * n {
                out.data[k] = cadd(out.data[k], minus_i(csub(hx.data[k], xh.data[k])));
            }
        }
        if let Some(gm) = &term.gamma {
            let gx = gm.multiply(x, lap)?;
            let xg = x.multiply(gm, lap)?;
            for k in 0..n * n {
                out.data[k] = csub(out.data[k], cscale(cadd(gx.data[k], xg.data[k]), 0.5));
            }
        }
        let mut outs = vec![out];
        self.add_recycling(term, &[x], &mut outs, lap)?;
        Ok(outs.pop().expect("one block"))
    }

    /// Recycled ground blocks `R_p(X_EE)` of many blocks: `g^2 x m`, row
    /// `a*g+b`, one column per block. A dense map is one matrix product.
    pub fn recycled(&self, term: &ChargeTerm, xs: &[&Dense], lap: &Lapack) -> Result<Dense, String> {
        let (g, e) = (self.g, self.e);
        let m = xs.len();
        if let Some(r) = &term.recycling_matrix {
            let mut excited = Dense::zeros(e * e, m);
            for (column, x) in xs.iter().enumerate() {
                for j in 0..e {
                    for i in 0..e {
                        excited.set(i * e + j, column, x.at(g + i, g + j));
                    }
                }
            }
            return r.multiply(&excited, lap);
        }
        let mut out = Dense::zeros(g * g, m);
        for (column, x) in xs.iter().enumerate() {
            for &(ab, ij, v) in &term.recycling {
                out.add_at(ab, column, cmul(v, x.at(g + ij / e, g + ij % e)));
            }
        }
        Ok(out)
    }

    fn add_recycling(&self, term: &ChargeTerm, xs: &[&Dense], out: &mut [Dense], lap: &Lapack) -> Result<(), String> {
        if term.recycling.is_empty() {
            return Ok(());
        }
        let g = self.g;
        let recycled = self.recycled(term, xs, lap)?;
        for (column, o) in out.iter_mut().enumerate() {
            for a in 0..g {
                for b in 0..g {
                    o.add_at(a, b, recycled.at(a * g + b, column));
                }
            }
        }
        Ok(())
    }

    /// Ground block of `L_p X` only (row-major g x g). `Gamma` acts on the
    /// excited block and cannot reach it; the commutator and recycling can.
    pub fn apply_ground_output(&self, term: &ChargeTerm, x: &Dense, lap: &Lapack) -> Result<Vec<C>, String> {
        let (n, g) = (self.n, self.g);
        let mut out = vec![ZERO; g * g];
        if let Some(h) = &term.hamiltonian {
            for a in 0..g {
                for b in 0..g {
                    let mut s = ZERO;
                    for k in 0..n {
                        s = cadd(s, cmul(h.at(a, k), x.at(k, b)));
                        s = csub(s, cmul(x.at(a, k), h.at(k, b)));
                    }
                    out[a * g + b] = minus_i(s);
                }
            }
        }
        if !term.recycling.is_empty() {
            for (ab, v) in self.recycled(term, &[x], lap)?.data.iter().enumerate() {
                out[ab] = cadd(out[ab], *v);
            }
        }
        Ok(out)
    }

    /// `L_p E_ab` for the ground unit matrix `E_ab`: only the commutator acts
    /// (Gamma and recycling vanish on the ground block).
    pub fn apply_to_ground_unit(&self, term: &ChargeTerm, a: usize, b: usize) -> Dense {
        let n = self.n;
        let mut out = Dense::zeros(n, n);
        if let Some(h) = &term.hamiltonian {
            for k in 0..n {
                // -i (H E_ab - E_ab H): column a of H into column b, minus row b of H into row a.
                out.add_at(k, b, minus_i(h.at(k, a)));
                out.add_at(a, k, csub(ZERO, minus_i(h.at(b, k))));
            }
        }
        out
    }

    /// The charge-0 no-jump operators `A_L = -i H_0 - Gamma_0/2` and
    /// `A_R = i H_0 - Gamma_0/2`, so that `L_0 X = A_L X + X A_R + R_0 X_EE`
    /// exactly (no Hermiticity of the stored coefficients is assumed).
    pub fn no_jump(&self) -> (Dense, Dense) {
        let n = self.n;
        let zero = self.zero_charge();
        let term = self.term(&zero);
        let mut left = Dense::zeros(n, n);
        let mut right = Dense::zeros(n, n);
        if let Some(term) = term {
            if let Some(h) = &term.hamiltonian {
                for k in 0..n * n {
                    left.data[k] = minus_i(h.data[k]);
                    right.data[k] = csub(ZERO, minus_i(h.data[k]));
                }
            }
            if let Some(gm) = &term.gamma {
                for k in 0..n * n {
                    left.data[k] = csub(left.data[k], cscale(gm.data[k], 0.5));
                    right.data[k] = csub(right.data[k], cscale(gm.data[k], 0.5));
                }
            }
        }
        (left, right)
    }
}

// ---------------------------------------------------------------------------
// Charge-block solvers for (z - L_0) restricted to the eliminated space Q
// ---------------------------------------------------------------------------

/// Exact inverse of the complete charge block `z - L_0` (all n^2 entries).
/// No-jump part by a full triangular Sylvester solve; recycling by Woodbury.
struct FullBlockSolver {
    z: C,
    sylvester: Arc<SylvesterFactor>,
    /// LU of `K = I - V_R (z - N)^{-1} U` (g^2 x g^2), `U` the ground-entry
    /// embedding: `(z - L_0)^{-1} y = S^{-1} (y + U K^{-1} V_R S^{-1} y)` with
    /// `S = z - N`, two Sylvester solves and no stored images.
    woodbury: Option<LuFactor>,
    minimum_denominator: f64,
    woodbury_condition: f64,
}

/// Inverse of `z - L_0` restricted to the non-ground entries (ground entries
/// are slow and kept in P): excited block condensed onto the optical block,
/// optical Schur complement factored densely.
struct NonGroundSolver {
    z: C,
    blocks: Arc<NoJumpBlocks>,
    lu: LuFactor,
    minimum_denominator: f64,
    condition_estimate: f64,
}

enum BlockSolver {
    Full(FullBlockSolver),
    NonGround(NonGroundSolver),
}

/// Block pieces of the charge-0 no-jump operators.
struct NoJumpBlocks {
    g: usize,
    e: usize,
    left_ge: Dense,  // A_L[G, E]
    left_eg: Dense,  // A_L[E, G]
    right_ge: Dense, // A_R[G, E]
    right_eg: Dense, // A_R[E, G]
    ge: SylvesterFactor, // (A_L[G,G], A_R[E,E])
    eg: SylvesterFactor, // (A_L[E,E], A_R[G,G])
    ee: SylvesterFactor, // (A_L[E,E], A_R[E,E])
    left_gg: Dense,
    left_ee: Dense,
    right_gg: Dense,
    right_ee: Dense,
}

impl NoJumpBlocks {
    fn new(left: &Dense, right: &Dense, g: usize, e: usize, lap: &Lapack) -> Result<Self, String> {
        let n = g + e;
        let left_gg = left.block(0..g, 0..g);
        let left_ee = left.block(g..n, g..n);
        let right_gg = right.block(0..g, 0..g);
        let right_ee = right.block(g..n, g..n);
        let schur_left_gg = Schur::new(&left_gg, lap)?;
        let schur_left_ee = Schur::new(&left_ee, lap)?;
        let schur_right_gg = Schur::new(&right_gg, lap)?;
        let schur_right_ee = Schur::new(&right_ee, lap)?;
        Ok(Self {
            g,
            e,
            left_ge: left.block(0..g, g..n),
            left_eg: left.block(g..n, 0..g),
            right_ge: right.block(0..g, g..n),
            right_eg: right.block(g..n, 0..g),
            ge: SylvesterFactor { left: schur_left_gg.clone(), right: schur_right_ee.clone() },
            eg: SylvesterFactor { left: schur_left_ee.clone(), right: schur_right_gg },
            ee: SylvesterFactor { left: schur_left_ee, right: schur_right_ee },
            left_gg,
            left_ee,
            right_gg,
            right_ee,
        })
    }

    /// `z X - A1 X - X A2` (forward Sylvester product).
    fn forward(z: C, a1: &Dense, a2: &Dense, x: &Dense, lap: &Lapack) -> Result<Dense, String> {
        let left = a1.multiply(x, lap)?;
        let right = x.multiply(a2, lap)?;
        let mut out = x.clone();
        for k in 0..out.data.len() {
            out.data[k] = csub(csub(cmul(z, x.data[k]), left.data[k]), right.data[k]);
        }
        Ok(out)
    }

    /// Excited block driven by the optical coherences:
    /// `x_EE = N_EE^{-1}(y_EE + A_L[E,G] x_GE + x_EG A_R[G,E])`.
    fn excited_from_optical(&self, z: C, y_ee: &Dense, x_ge: &Dense, x_eg: &Dense, lap: &Lapack) -> Result<Dense, String> {
        let a = self.left_eg.multiply(x_ge, lap)?;
        let b = x_eg.multiply(&self.right_ge, lap)?;
        let mut rhs = y_ee.clone();
        for k in 0..rhs.data.len() {
            rhs.data[k] = cadd(cadd(rhs.data[k], a.data[k]), b.data[k]);
        }
        self.ee.solve(z, &rhs, lap)
    }

    fn pack(&self, x_ge: &Dense, x_eg: &Dense) -> Vec<C> {
        let mut v = Vec::with_capacity(2 * self.g * self.e);
        v.extend_from_slice(&x_ge.data);
        v.extend_from_slice(&x_eg.data);
        v
    }
    fn unpack(&self, v: &[C]) -> (Dense, Dense) {
        let half = self.g * self.e;
        (
            Dense { rows: self.g, cols: self.e, data: v[..half].to_vec() },
            Dense { rows: self.e, cols: self.g, data: v[half..].to_vec() },
        )
    }

    /// Optical Schur complement applied to `x_O` (excited block eliminated).
    fn optical_schur(&self, z: C, x_ge: &Dense, x_eg: &Dense, lap: &Lapack) -> Result<Vec<C>, String> {
        let zero_ee = Dense::zeros(self.e, self.e);
        let x_ee = self.excited_from_optical(z, &zero_ee, x_ge, x_eg, lap)?;
        let mut out_ge = Self::forward(z, &self.left_gg, &self.right_ee, x_ge, lap)?;
        let coupling_ge = self.left_ge.multiply(&x_ee, lap)?;
        for k in 0..out_ge.data.len() {
            out_ge.data[k] = csub(out_ge.data[k], coupling_ge.data[k]);
        }
        let mut out_eg = Self::forward(z, &self.left_ee, &self.right_gg, x_eg, lap)?;
        let coupling_eg = x_ee.multiply(&self.right_eg, lap)?;
        for k in 0..out_eg.data.len() {
            out_eg.data[k] = csub(out_eg.data[k], coupling_eg.data[k]);
        }
        Ok(self.pack(&out_ge, &out_eg))
    }
}

impl NonGroundSolver {
    fn new(z: C, blocks: Arc<NoJumpBlocks>, lap: &Lapack) -> Result<Self, String> {
        let (g, e) = (blocks.g, blocks.e);
        let size = 2 * g * e;
        let minimum_denominator = blocks
            .ge
            .minimum_denominator(z)
            .min(blocks.eg.minimum_denominator(z))
            .min(blocks.ee.minimum_denominator(z));
        let mut schur = Dense::zeros(size, size);
        for column in 0..size {
            let mut unit = vec![ZERO; size];
            unit[column] = ONE;
            let (x_ge, x_eg) = blocks.unpack(&unit);
            let image = blocks.optical_schur(z, &x_ge, &x_eg, lap)?;
            schur.data[column * size..(column + 1) * size].copy_from_slice(&image);
        }
        let lu = LuFactor::new(&schur, lap)?;
        // Cheap condition indicator: ||S||_1 ||S^{-1} v|| / ||v|| on a probe.
        let probe: Vec<C> = (0..size).map(|k| complex(1. + (k % 7) as f64 * 0.1, (k % 5) as f64 * 0.07)).collect();
        let solved = lu.solve(&probe, lap)?;
        let ratio = solved.iter().map(|z| cabs(*z)).sum::<f64>() / probe.iter().map(|z| cabs(*z)).sum::<f64>();
        Ok(Self { z, blocks, lu, minimum_denominator, condition_estimate: one_norm(&schur) * ratio })
    }

    /// Solve `(z - L_0) x = y` on the non-ground entries; returns `x` with a
    /// zero ground block. The ground block of `y` is ignored (it is P).
    fn solve(&self, y: &Dense, lap: &Lapack) -> Result<Dense, String> {
        let b = &self.blocks;
        let (g, e) = (b.g, b.e);
        let n = g + e;
        let y_ge = y.block(0..g, g..n);
        let y_eg = y.block(g..n, 0..g);
        let y_ee = y.block(g..n, g..n);
        // Right-hand side of the optical Schur system.
        let w = b.ee.solve(self.z, &y_ee, lap)?;
        let mut rhs_ge = y_ge.clone();
        let c_ge = b.left_ge.multiply(&w, lap)?;
        for k in 0..rhs_ge.data.len() {
            rhs_ge.data[k] = cadd(rhs_ge.data[k], c_ge.data[k]);
        }
        let mut rhs_eg = y_eg.clone();
        let c_eg = w.multiply(&b.right_eg, lap)?;
        for k in 0..rhs_eg.data.len() {
            rhs_eg.data[k] = cadd(rhs_eg.data[k], c_eg.data[k]);
        }
        let solution = self.lu.solve(&b.pack(&rhs_ge, &rhs_eg), lap)?;
        let (x_ge, x_eg) = b.unpack(&solution);
        let x_ee = b.excited_from_optical(self.z, &y_ee, &x_ge, &x_eg, lap)?;
        let mut x = Dense::zeros(n, n);
        for i in 0..g {
            for j in 0..e {
                x.set(i, g + j, x_ge.at(i, j));
                x.set(g + j, i, x_eg.at(j, i));
            }
        }
        for i in 0..e {
            for j in 0..e {
                x.set(g + i, g + j, x_ee.at(i, j));
            }
        }
        if !x.is_finite() {
            return Err("Nonfinite non-ground block solve".into());
        }
        Ok(x)
    }
}

impl FullBlockSolver {
    fn new(z: C, sylvester: Arc<SylvesterFactor>, frame: &FrameLiouvillian, lap: &Lapack) -> Result<Self, String> {
        let (n, g) = (frame.n, frame.g);
        let minimum_denominator = sylvester.minimum_denominator(z);
        let zero = frame.zero_charge();
        let mut woodbury = None;
        let mut woodbury_condition = 1.;
        if let Some(term) = frame.term(&zero).filter(|t| !t.recycling.is_empty()) {
            let rank = g * g;
            let units: Vec<Dense> = (0..rank)
                .map(|ab| {
                    let mut unit = Dense::zeros(n, n);
                    unit.set(ab / g, ab % g, ONE);
                    unit
                })
                .collect();
            let refs: Vec<&Dense> = units.iter().collect();
            let ground_images = sylvester.solve_all(z, &refs, lap)?;
            // K = I - V_R W with V_R X = recycling functional of X_EE.
            let images: Vec<&Dense> = ground_images.iter().collect();
            let recycled = frame.recycled(term, &images, lap)?;
            let mut k = Dense::identity(rank);
            for (entry, v) in k.data.iter_mut().zip(&recycled.data) {
                *entry = csub(*entry, *v);
            }
            let factor = LuFactor::new(&k, lap)?;
            let inverse = factor.inverse(lap)?;
            woodbury_condition = one_norm(&k) * one_norm(&inverse);
            woodbury = Some(factor);
        }
        Ok(Self { z, sylvester, woodbury, minimum_denominator, woodbury_condition })
    }

    fn solve(&self, y: &Dense, frame: &FrameLiouvillian, lap: &Lapack) -> Result<Dense, String> {
        Ok(self.solve_all(&[y], frame, lap)?.pop().expect("one solution"))
    }

    /// `(z - L_0)^{-1} y` for many right-hand sides.
    fn solve_all(&self, ys: &[&Dense], frame: &FrameLiouvillian, lap: &Lapack) -> Result<Vec<Dense>, String> {
        let first = self.sylvester.solve_all(self.z, ys, lap)?;
        let Some(factor) = &self.woodbury else { return Ok(first) };
        let g = frame.g;
        let zero = frame.zero_charge();
        let term = frame.term(&zero).expect("charge-0 term");
        let refs: Vec<&Dense> = first.iter().collect();
        let amounts = factor.solve_matrix(&frame.recycled(term, &refs, lap)?, lap)?;
        let mut corrected: Vec<Dense> = Vec::with_capacity(ys.len());
        for (column, y) in ys.iter().enumerate() {
            let mut y2 = (*y).clone();
            for ab in 0..g * g {
                y2.add_at(ab / g, ab % g, amounts.at(ab, column));
            }
            corrected.push(y2);
        }
        let refs: Vec<&Dense> = corrected.iter().collect();
        let out = self.sylvester.solve_all(self.z, &refs, lap)?;
        if out.iter().any(|x| !x.is_finite()) {
            return Err("Nonfinite full charge-block solve".into());
        }
        Ok(out)
    }
}

impl BlockSolver {
    fn solve(&self, y: &Dense, frame: &FrameLiouvillian, lap: &Lapack) -> Result<Dense, String> {
        match self {
            BlockSolver::Full(s) => s.solve(y, frame, lap),
            BlockSolver::NonGround(s) => s.solve(y, lap),
        }
    }
}

// ---------------------------------------------------------------------------
// The slow dual
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct SlowDualOptions {
    /// A ground coherence at a nonzero charge is "slow" (kept in P) when its
    /// envelope frequency `|nu_c + eps_a - eps_b|` is below this cutoff
    /// (rad/us). `None` selects four times the slowest optical decay rate.
    pub slow_frequency_cutoff: Option<f64>,
}

impl Default for SlowDualOptions {
    fn default() -> Self { Self { slow_frequency_cutoff: None } }
}

#[derive(Clone, Debug)]
pub struct SlowDualDiagnostics {
    pub slow_charges: Vec<Charge>,
    pub slow_dimension: usize,
    pub operator_charges: usize,
    /// Operator charges whose couplings are all below
    /// `STRUCTURAL_COUPLING_FLOOR` of the largest of their family.
    pub weak_operator_charges: usize,
    /// Slow charges that only paths through a weak coupling reach (not
    /// slots) and that the slow columns actually couple into.
    pub unrepresented_slow_charges: usize,
    /// Frobenius norm of those couplings (rad/us); charged to every estimate.
    pub unrepresented_rate: f64,
    pub eliminated_charge_blocks: usize,
    /// Expansion point `r_c` of every slot (`s_c = -i r_c`, rad/us).
    pub slot_rotation: Vec<f64>,
    /// Slowest damping of the eliminated optical coherences (rad/us).
    pub optical_decay_floor: f64,
    pub slow_frequency_cutoff: f64,
    /// Smallest `|z - lambda|` of any eliminated block (well-posedness).
    pub minimum_eliminated_denominator: f64,
    pub maximum_woodbury_condition: f64,
    pub maximum_optical_schur_condition: f64,
    pub generator_eigenvector_condition: f64,
    pub generator_reconstruction_relative: f64,
    /// Which eigen-decomposition attempt resolved `J_eff` (see `robust_eigen`).
    pub generator_eigen_method: &'static str,
    pub generator_spectral_abscissa: f64,
    /// Eigenvector condition of the dressed no-jump frame (None: unavailable).
    pub dressed_condition: Option<f64>,
    pub preparation_ms: f64,
}

/// Prepared once per request: the slow generator `J_eff`, its eigenbasis and
/// the population readout. Independent of the trial basis and Fourier domain.
pub struct SlowDual {
    pub(crate) owner: Arc<PrimitivePreparation>,
    pub(crate) lapack: Arc<Lapack>,
    pub(crate) frame: FrameLiouvillian,
    pub(crate) n: usize,
    pub(crate) g: usize,
    /// Slow charges; `slots[0]` is the central charge.
    pub(crate) slots: Vec<Charge>,
    pub(crate) slot_frequency: Vec<f64>,
    /// `J_eff`, column-major, dimension `slots.len() * g^2`.
    pub(crate) generator: Dense,
    pub(crate) eigenvalues: Vec<C>,
    pub(crate) inverse_vectors: Dense,
    /// Readout groups `(nu, R V)`: physical populations
    /// `p(t) = Re sum_groups exp(i nu t) (R V) xhat(t)`.
    pub(crate) readout: Vec<(f64, Dense)>,
    /// Frame no-jump diagonal (reference rates for halo responses):
    /// `d_ab = A_L[a,a] + A_R[b,b]`, row-major n x n.
    pub(crate) diagonal_reference: Vec<C>,
    /// Physical population readout of a frame matrix: (state, a, b, weight).
    pub(crate) population_readout: Vec<(usize, usize, usize, C)>,
    /// Exact eigen-coordinates of the charge-0 no-jump operator (fast response).
    pub(crate) dressed: Option<DressedFrame>,
    /// Slaved charge-0 non-ground block per slow column (n x n frame matrices).
    pub(crate) slaved_central: Vec<Dense>,
    /// Exact charge-block inverses, built on demand for Markov slaving of
    /// fields at fast charges (None only for synthetic test duals).
    elimination: Option<Elimination>,
    pub(crate) diagnostics: SlowDualDiagnostics,
    /// Ground "energies" `eps_a` (rad/us) of the slot criterion.
    pub(crate) ground_energy: Vec<f64>,
    /// `[(0, R)]` with `||R xhat|| = ||K V xhat||` for the couplings `K` into
    /// the unrepresented slow charges (compressed to their range; empty when
    /// there are none): the readout of the norm every estimate charges.
    pub(crate) unrepresented_readout: Vec<(f64, Dense)>,
}

/// Rows `R` with `||R xhat|| = ||K V xhat||` for the stacked couplings `K`
/// (`g^2 x dim` per unrepresented charge) and the `J_eff` eigenvectors `V`:
/// `R = diag(sqrt(mu)) W^H` from the Gram matrix `(K V)^H K V = W diag(mu)
/// W^H`, keeping the eigenvalues above 1e-24 of the largest (1e-12 in the
/// norm), so the readout has at most `dim` rows.
fn unrepresented_norm_readout(coupling: &BTreeMap<Charge, Dense>, vectors: &Dense, dim: usize, lap: &Lapack) -> Result<Vec<(f64, Dense)>, String> {
    if coupling.is_empty() {
        return Ok(Vec::new());
    }
    let mut gram = Dense::zeros(dim, dim);
    for k in coupling.values() {
        let kv = k.multiply(vectors, lap)?;
        for j in 0..dim {
            for i in 0..dim {
                let mut sum = ZERO;
                for r in 0..kv.rows {
                    sum = cadd(sum, cmul(cconj(kv.at(r, i)), kv.at(r, j)));
                }
                gram.add_at(i, j, sum);
            }
        }
    }
    let (values, w) = lap.hermitian_eigen(&gram.data, dim)?;
    let largest = values.iter().copied().fold(0_f64, f64::max);
    if !(largest > 0.) {
        return Ok(Vec::new());
    }
    let kept: Vec<usize> = (0..dim).filter(|&k| values[k] > 1e-24 * largest).collect();
    let mut rows = Dense::zeros(kept.len(), dim);
    for (row, &k) in kept.iter().enumerate() {
        let scale = values[k].sqrt();
        for j in 0..dim {
            rows.set(row, j, cscale(cconj(w[k * dim + j]), scale));
        }
    }
    Ok(vec![(0., rows)])
}

/// The slot criterion: some ground coherence at envelope frequency `nu` has
/// `|nu + eps_a - eps_b| < cutoff`.
fn slow_ground_coherence(nu: f64, ground_energy: &[f64], cutoff: f64) -> bool {
    ground_energy.iter().any(|a| ground_energy.iter().any(|b| (nu + a - b).abs() < cutoff))
}

/// Relaxation spectrum of `J_eff` (`SlowDual::slow_spectrum`).
#[derive(Clone, Copy, Debug)]
pub struct SlowSpectrum {
    /// Conserved quantities: the trace and any other mode at zero.
    pub conserved: usize,
    /// Slowest relaxation rate of the decaying modes (1/us).
    pub slowest_rate: Option<f64>,
    /// Rates at or below this are numerically zero (1/us).
    pub floor: f64,
}

/// Factored no-jump structure plus a cache of fast charge-block inverses
/// `(i nu_k - L_0)^{-1}` keyed by charge.
struct Elimination {
    blocks: Arc<NoJumpBlocks>,
    sylvester: Arc<SylvesterFactor>,
    cache: std::sync::Mutex<BTreeMap<Charge, Arc<BlockSolver>>>,
    slow_cache: std::sync::Mutex<BTreeMap<usize, Arc<BlockSolver>>>,
}

/// Cache key of a shifted solve: the charge plus the shift rounded to 1e-9
/// rad/us (shifts computed along different charge paths agree to rounding).
fn shift_key(c: &Charge, z_im: f64) -> (Charge, i64) { (c.clone(), (z_im * 1e9).round() as i64) }

/// Eigen-coordinates of the charge-0 no-jump operator `N X = A_L X + X A_R`:
/// with `A_L = V_L diag(a) V_L^-1`, `A_R = V_R diag(b) V_R^-1` and
/// `Xhat = V_L^-1 X V_R`, `N` is diagonal: `(N X)hat_ik = (a_i + b_k) Xhat_ik`.
/// This includes the complete static Hamiltonian (strong static Rabi
/// couplings) and the complete charge-0 radiative damping, so every early
/// fast transient of an optical/excited residual is integrated exactly.
pub(crate) struct DressedFrame {
    pub left_values: Vec<C>,
    pub right_values: Vec<C>,
    pub left: Dense,          // V_L
    pub left_inverse: Dense,  // V_L^-1
    pub right: Dense,         // V_R
    pub right_inverse: Dense, // V_R^-1
    /// Population readout `rho[s, i*n+k]` of an eigen-coordinate matrix.
    pub readout: Dense,
    /// `W[:, slot block] * recycling` for every slot (dim x n^2).
    pub modal_recycling: Vec<Dense>,
    pub condition: f64,
}

impl SlowDual {
    pub fn prepare(prepared: &OwnedPrepared, options: SlowDualOptions) -> Result<Self, String> {
        let began = Instant::now();
        let p = Arc::clone(&prepared.primitives);
        let lap = Arc::clone(&prepared.preparation_lapack);
        let frame = FrameLiouvillian::from_primitives(&p)?;
        let (n, g, e) = (frame.n, frame.g, frame.e);
        let phases = &p.hilbert.source.phases;
        let frequency = |q: &[i32]| -> Result<f64, String> { phases.frequency(q).map(|f| f.hi + f.lo) };
        let zero = frame.zero_charge();

        // --- Charge-0 no-jump operators and time scales -------------------
        let (left, right) = frame.no_jump();
        let blocks = Arc::new(NoJumpBlocks::new(&left, &right, g, e, &lap)?);
        // Slowest damping of an optical coherence: -Re of the (G,E) Sylvester
        // eigenvalues T_L[G,G] + T_R[E,E].
        let mut optical_decay_floor = f64::INFINITY;
        for i in 0..g {
            for j in 0..e {
                let lambda = cadd(blocks.ge.left.t.at(i, i), blocks.ge.right.t.at(j, j));
                optical_decay_floor = optical_decay_floor.min(-lambda.re);
            }
        }
        if !(optical_decay_floor.is_finite() && optical_decay_floor > 0.) {
            return Err("No damped optical coherences: the slow-manifold dual needs radiative decay".into());
        }
        let cutoff = options.slow_frequency_cutoff.unwrap_or(4. * optical_decay_floor);
        if !(cutoff.is_finite() && cutoff > 0.) {
            return Err("Invalid slow-frequency cutoff".into());
        }
        // Ground "energies" eps_a: the bare ground block is -i H_GG, so its
        // Schur diagonal is -i eps_a.
        let ground_energy: Vec<f64> = (0..g).map(|a| -blocks.ge.left.t.at(a, a).im).collect();

        // --- Slow charges -------------------------------------------------
        // Slots are the slow charges two significant couplings reach from the
        // central charge (module docs, "Weak couplings"). The slow charges
        // that only paths through a weak coupling reach stay unrepresented:
        // the couplings into them are collected below and charged.
        let two_hops = |charges: &[&Charge]| -> Result<BTreeSet<Charge>, String> {
            let mut out = BTreeSet::new();
            for &p1 in charges.iter().chain(std::iter::once(&&zero)) {
                for &p2 in charges.iter().chain(std::iter::once(&&zero)) {
                    out.insert(add_charge(p1, p2, false)?);
                }
            }
            Ok(out)
        };
        let is_slow = |c: &Charge| -> Result<bool, String> { Ok(*c != zero && slow_ground_coherence(frequency(c)?, &ground_energy, cutoff)) };
        let strong: Vec<&Charge> = frame.terms.iter().filter(|t| t.significant).map(|t| &t.charge).collect();
        let all: Vec<&Charge> = frame.terms.iter().map(|t| &t.charge).collect();
        let candidates = two_hops(&strong)?;
        let mut slots = vec![zero.clone()];
        for c in &candidates {
            if is_slow(c)? {
                slots.push(c.clone());
            }
        }
        let mut unrepresented: BTreeSet<Charge> = BTreeSet::new();
        for c in two_hops(&all)? {
            if !candidates.contains(&c) && is_slow(&c)? {
                unrepresented.insert(c);
            }
        }
        let slot_frequency: Vec<f64> = slots.iter().map(|c| frequency(c)).collect::<Result<_, _>>()?;
        let slot_of = |c: &Charge| slots.iter().position(|s| s == c);
        let g2 = g * g;
        let dim = slots.len() * g2;
        // Expansion point per slot (module docs): co-rotate with the slot's
        // populations when they are slow, else with its slowest coherence.
        let slot_rotation: Vec<f64> = slot_frequency
            .iter()
            .enumerate()
            .map(|(s, &nu)| {
                if s == 0 {
                    return 0.;
                }
                if nu.abs() < cutoff {
                    return nu;
                }
                let mut best = nu;
                for a in 0..g {
                    for b in 0..g {
                        let candidate = nu + ground_energy[a] - ground_energy[b];
                        if candidate.abs() < best.abs() {
                            best = candidate;
                        }
                    }
                }
                best
            })
            .collect();

        // --- Elimination solvers per intermediate charge -----------------
        let full_sylvester = Arc::new(SylvesterFactor { left: Schur::new(&left, &lap)?, right: Schur::new(&right, &lap)? });
        // One block inverse at a time (z = i (nu_{c''} - r_c)): built, applied
        // to every slow column, dropped. Keeping them all would cost about
        // 1-2.5 MB per intermediate charge.
        let mut built: BTreeSet<(Charge, i64)> = BTreeSet::new();
        let mut minimum_denominator = f64::INFINITY;
        let mut woodbury_condition = 1_f64;
        let mut schur_condition = 1_f64;
        let mut build_solver = |c: &Charge, z_im: f64| -> Result<BlockSolver, String> {
            built.insert(shift_key(c, z_im));
            let z = complex(0., z_im);
            // A slow charge keeps its ground block: only the rest is eliminated.
            let solver = if slot_of(c).is_some() || unrepresented.contains(c) {
                let s = NonGroundSolver::new(z, Arc::clone(&blocks), &lap)?;
                minimum_denominator = minimum_denominator.min(s.minimum_denominator);
                schur_condition = schur_condition.max(s.condition_estimate);
                BlockSolver::NonGround(s)
            } else {
                let s = FullBlockSolver::new(z, Arc::clone(&full_sylvester), &frame, &lap)?;
                minimum_denominator = minimum_denominator.min(s.minimum_denominator);
                woodbury_condition = woodbury_condition.max(s.woodbury_condition);
                BlockSolver::Full(s)
            };
            Ok(solver)
        };

        // --- Assemble J_eff ------------------------------------------------
        let mut generator = Dense::zeros(dim, dim);
        // Adiabatically slaved charge-0 non-ground block for each slow column:
        // X_Q(0) = sum_c (0 - L_0)_QQ^-1 Q L_{-c} P e_P(c). Used for the excited
        // readout and for the fast initial layer.
        let mut slaved_central: Vec<Dense> = vec![Dense::zeros(n, n); dim];
        // Excited-population readout of the slaved block at every slot charge:
        // slaved_population[s2] is (state x column).
        let excited_readout: Vec<(usize, usize, usize, C)> = {
            let basis = &p.hilbert.basis_column_major;
            let mut out = Vec::new();
            for i in 0..n {
                for a in g..n {
                    for b in g..n {
                        let w = cmul(basis[i + a * n], cconj(basis[i + b * n]));
                        if w.re != 0. || w.im != 0. {
                            out.push((i, a, b, w));
                        }
                    }
                }
            }
            out
        };
        let mut slaved_population: Vec<Dense> = vec![Dense::zeros(n, dim); slots.len()];
        // Record an eliminated (slaved) block x at slot s2 for one column.
        let record_slaved = |x: &Dense, s2: usize, column: usize, slaved_central: &mut Vec<Dense>, slaved_population: &mut Vec<Dense>| {
            if s2 == 0 {
                let target = &mut slaved_central[column];
                for k in 0..n * n {
                    target.data[k] = cadd(target.data[k], x.data[k]);
                }
            }
            for &(i, a, b, w) in &excited_readout {
                slaved_population[s2].add_at(i, column, cmul(w, x.at(a, b)));
            }
        };
        // Ground rows of the couplings into every unrepresented slow charge
        // (`g^2 x dim` per charge; their Frobenius norm is charged).
        let mut unrepresented_coupling: BTreeMap<Charge, Dense> = BTreeMap::new();
        let mut couple_unrepresented = |c: &Charge, column: usize, ground: &dyn Fn(usize, usize) -> C| {
            let k = unrepresented_coupling.entry(c.clone()).or_insert_with(|| Dense::zeros(g2, dim));
            for a in 0..g {
                for b in 0..g {
                    k.add_at(a * g + b, column, ground(a, b));
                }
            }
        };
        for (s, c) in slots.iter().enumerate() {
            for ab in 0..g2 {
                let column = s * g2 + ab;
                generator.add_at(column, column, complex(0., -slot_frequency[s]));
            }
            // Non-ground fields arriving at slow charges after the second hop,
            // accumulated per (column, target slot) before their elimination.
            let mut second_hop: BTreeMap<(usize, usize), Dense> = BTreeMap::new();
            for term in &frame.terms {
                let c2 = add_charge(c, &term.charge, false)?;
                let target_slot = slot_of(&c2);
                let held = target_slot.is_some() || unrepresented.contains(&c2);
                let mut solver: Option<BlockSolver> = None;
                for ab in 0..g2 {
                    let column = s * g2 + ab;
                    let mut y = frame.apply_to_ground_unit(term, ab / g, ab % g);
                    if held {
                        // Direct P -> P coupling (or into an unrepresented slow
                        // charge), then eliminate only the rest.
                        match target_slot {
                            Some(s2) => {
                                for a2 in 0..g {
                                    for b2 in 0..g {
                                        generator.add_at(s2 * g2 + a2 * g + b2, column, y.at(a2, b2));
                                    }
                                }
                            }
                            None => couple_unrepresented(&c2, column, &|a2, b2| y.at(a2, b2)),
                        }
                        for a2 in 0..g {
                            for b2 in 0..g {
                                y.set(a2, b2, ZERO);
                            }
                        }
                    }
                    if y.frobenius() == 0. {
                        continue;
                    }
                    if solver.is_none() {
                        solver = Some(build_solver(&c2, frequency(&c2)? - slot_rotation[s])?);
                    }
                    let x = solver.as_ref().expect("solver").solve(&y, &frame, &lap)?;
                    if let Some(s2) = target_slot {
                        record_slaved(&x, s2, column, &mut slaved_central, &mut slaved_population);
                    }
                    for term2 in &frame.terms {
                        let c3 = add_charge(&c2, &term2.charge, false)?;
                        let Some(s3) = slot_of(&c3) else {
                            if unrepresented.contains(&c3) {
                                let image = frame.apply(term2, &x, &lap)?;
                                couple_unrepresented(&c3, column, &|a, b| image.at(a, b));
                            }
                            continue;
                        };
                        let image = frame.apply(term2, &x, &lap)?;
                        for a in 0..g {
                            for b in 0..g {
                                generator.add_at(s3 * g2 + a * g + b, column, image.at(a, b));
                            }
                        }
                        if !held {
                            // Second hop from a fast charge into a slow one.
                            let w = second_hop.entry((ab, s3)).or_insert_with(|| Dense::zeros(n, n));
                            for a in 0..n {
                                for b in 0..n {
                                    if a >= g || b >= g {
                                        w.add_at(a, b, image.at(a, b));
                                    }
                                }
                            }
                        }
                    }
                }
            }
            for s3 in 0..slots.len() {
                let c3 = &slots[s3];
                let mut solver: Option<BlockSolver> = None;
                for ab in 0..g2 {
                    let Some(w) = second_hop.remove(&(ab, s3)) else { continue };
                    if w.frobenius() == 0. {
                        continue;
                    }
                    let column = s * g2 + ab;
                    if solver.is_none() {
                        solver = Some(build_solver(c3, slot_frequency[s3] - slot_rotation[s])?);
                    }
                    let x2 = solver.as_ref().expect("solver").solve(&w, &frame, &lap)?;
                    record_slaved(&x2, s3, column, &mut slaved_central, &mut slaved_population);
                    for term3 in &frame.terms {
                        let c4 = add_charge(c3, &term3.charge, false)?;
                        if let Some(s4) = slot_of(&c4) {
                            let ground = frame.apply_ground_output(term3, &x2, &lap)?;
                            for (k, v) in ground.iter().enumerate() {
                                generator.add_at(s4 * g2 + k, column, *v);
                            }
                        } else if unrepresented.contains(&c4) {
                            let ground = frame.apply_ground_output(term3, &x2, &lap)?;
                            couple_unrepresented(&c4, column, &|a, b| ground[a * g + b]);
                        }
                    }
                }
            }
        }
        if !generator.is_finite() {
            return Err("Nonfinite slow generator".into());
        }
        let unrepresented_rate = unrepresented_coupling.values().map(|k| k.frobenius().powi(2)).sum::<f64>().sqrt();
        if !unrepresented_rate.is_finite() {
            return Err("Nonfinite coupling into the unrepresented slow charges".into());
        }

        // --- Eigenbasis of J_eff -------------------------------------------
        let eigen = robust_eigen(&generator, None, &lap)?;
        let (eigenvalues, vectors, inverse_vectors) = (eigen.values, eigen.vectors, eigen.inverse);
        let eigenvector_condition = eigen.condition;
        let reconstruction = eigen.residual;
        let generator_eigen_method = eigen.method;
        if !(eigenvector_condition.is_finite() && eigenvector_condition < 1e10 && reconstruction < 1e-8) {
            return Err(format!(
                "Slow generator eigenbasis unresolved (condition {eigenvector_condition:e}, reconstruction {reconstruction:e})"
            ));
        }
        let spectral_abscissa = eigenvalues.iter().map(|z| z.re).fold(f64::NEG_INFINITY, f64::max);
        let unrepresented_readout = unrepresented_norm_readout(&unrepresented_coupling, &vectors, dim, &lap)?;

        // --- Population readout ---------------------------------------------
        let basis = &p.hilbert.basis_column_major; // B[i + a n]
        let weight = |i: usize, a: usize, b: usize| cmul(basis[i + a * n], cconj(basis[i + b * n]));
        let mut population_readout = Vec::new();
        let scale = (0..n * n).map(|k| cabs(basis[k])).fold(0_f64, f64::max).powi(2);
        for i in 0..n {
            for (lo, hi) in [(0, g), (g, n)] {
                for a in lo..hi {
                    for b in lo..hi {
                        let w = weight(i, a, b);
                        if cabs(w) > 1e-15 * scale {
                            population_readout.push((i, a, b, w));
                        }
                    }
                }
            }
        }
        // Every slot group: its ground readout plus the slaved excited block
        // at that charge (keeps each charge's trace consistent).
        let mut readout = Vec::with_capacity(slots.len());
        for s in 0..slots.len() {
            let mut r = slaved_population[s].clone();
            for &(i, a, b, w) in &population_readout {
                if a < g && b < g {
                    r.add_at(i, s * g2 + a * g + b, w);
                }
            }
            readout.push((slot_frequency[s], r.multiply(&vectors, &lap)?));
        }

        let mut diagonal_reference = vec![ZERO; n * n];
        for a in 0..n {
            for b in 0..n {
                diagonal_reference[a * n + b] = cadd(left.at(a, a), right.at(b, b));
            }
        }
        let recycling0 = frame.term(&zero).map(|t| t.recycling.clone()).unwrap_or_default();
        // Without the dressed frame the fast responses would be lost and the
        // estimate could underestimate; fail the preparation instead (callers
        // then use their fallback error control).
        let dressed = Some(DressedFrame::new(&left, &right, &population_readout, &recycling0, g, &inverse_vectors, slots.len(), &lap)?);

        let diagnostics = SlowDualDiagnostics {
            slow_charges: slots.clone(),
            slow_dimension: dim,
            operator_charges: frame.terms.len(),
            weak_operator_charges: frame.terms.iter().filter(|t| !t.significant).count(),
            unrepresented_slow_charges: unrepresented_coupling.len(),
            unrepresented_rate,
            eliminated_charge_blocks: built.len(),
            slot_rotation: slot_rotation.clone(),
            optical_decay_floor,
            slow_frequency_cutoff: cutoff,
            minimum_eliminated_denominator: minimum_denominator,
            maximum_woodbury_condition: woodbury_condition,
            maximum_optical_schur_condition: schur_condition,
            generator_eigenvector_condition: eigenvector_condition,
            generator_reconstruction_relative: reconstruction,
            generator_eigen_method,
            generator_spectral_abscissa: spectral_abscissa,
            dressed_condition: dressed.as_ref().map(|d| d.condition),
            preparation_ms: began.elapsed().as_secs_f64() * 1000.,
        };
        Ok(Self {
            owner: p,
            lapack: lap,
            frame,
            n,
            g,
            slots,
            slot_frequency,
            generator,
            eigenvalues,
            inverse_vectors,
            readout,
            diagonal_reference,
            population_readout,
            dressed,
            slaved_central,
            elimination: Some(Elimination { blocks: Arc::clone(&blocks), sylvester: full_sylvester, cache: Default::default(), slow_cache: Default::default() }),
            diagnostics,
            ground_energy,
            unrepresented_readout,
        })
    }

    pub fn diagnostics(&self) -> &SlowDualDiagnostics { &self.diagnostics }

    /// The modes of `J_eff` that are conserved or relax slower than `rate`
    /// (1/us), as frame matrices per slot: the right eigenvector's ground
    /// block at its slot, plus at the central slot the adiabatically slaved
    /// non-ground block of every slow column (`slaved_central`). They span
    /// the slow manifold a pole cannot resolve (see `asymptotic.rs`).
    pub(crate) fn slow_modes(&self, rate: f64) -> Result<Vec<Vec<(usize, Dense)>>, String> {
        let spectrum = self.slow_spectrum();
        let lap = &self.lapack;
        let vectors = LuFactor::new(&self.inverse_vectors, lap)?.inverse(lap)?;
        let (n, g) = (self.n, self.g);
        let g2 = g * g;
        let mut out = Vec::new();
        for (k, &alpha) in self.eigenvalues.iter().enumerate() {
            let conserved = self.slot_frequency.iter().any(|&nu| cabs(cadd(alpha, complex(0., nu))) <= spectrum.floor);
            if !conserved && -alpha.re >= rate {
                continue;
            }
            let mut blocks = Vec::with_capacity(self.slots.len());
            for s in 0..self.slots.len() {
                let mut x = Dense::zeros(n, n);
                for a in 0..g {
                    for b in 0..g {
                        x.set(a, b, vectors.at(s * g2 + a * g + b, k));
                    }
                }
                if s == 0 {
                    for (column, slaved) in self.slaved_central.iter().enumerate() {
                        let c = vectors.at(column, k);
                        for (entry, value) in x.data.iter_mut().zip(&slaved.data) {
                            *entry = cadd(*entry, cmul(c, *value));
                        }
                    }
                }
                blocks.push((s, x));
            }
            out.push(blocks);
        }
        Ok(out)
    }

    /// Relaxation spectrum of `J_eff` for the long-time limit. Its
    /// eigenvalues resolve arbitrarily small rates (a dense eigenproblem of
    /// dimension `slots * g^2`, no pole). A mode is conserved when
    /// `|alpha + i nu_s| <= floor` for some slot `s` (it then has a zero
    /// member in its Fourier family and enters the mean); `floor` is 1000
    /// times the numerical size of the exactly conserved trace mode (the
    /// smallest `|alpha|`, at least `eps ||J_eff||_F`), the resolution of the
    /// eigenvalues near zero. Every other mode with `-Re alpha > floor`
    /// relaxes; the slowest such rate sets `slowest_rate`. Undamped modes at
    /// other frequencies average out of the mean and are neither.
    pub fn slow_spectrum(&self) -> SlowSpectrum {
        let size = |z: C| cabs(z);
        let trace = self.eigenvalues.iter().map(|z| size(*z)).fold(f64::INFINITY, f64::min);
        let floor = 1e3 * trace.max(f64::EPSILON * self.generator.frobenius());
        let mut conserved = 0;
        let mut slowest: Option<f64> = None;
        for &alpha in &self.eigenvalues {
            if self.slot_frequency.iter().any(|&nu| size(cadd(alpha, complex(0., nu))) <= floor) {
                conserved += 1;
            } else if -alpha.re > floor {
                slowest = Some(slowest.map_or(-alpha.re, |r: f64| r.min(-alpha.re)));
            }
        }
        SlowSpectrum { conserved, slowest_rate: slowest, floor }
    }
    /// Bound per state on the population an initial central ground block
    /// `ground` (frame entries, g x g) still carries at the horizon `T` (us)
    /// in the slow generator's non-conserved modes: the sum over modes `k`
    /// and readout groups `nu` of `|(R V)_k c_k| exp(f Re(alpha_k + i nu) T)`
    /// with every decay rate taken at the fraction `f` (a margin for the
    /// reduction), over the terms that do not average out
    /// (`|Im(alpha_k) + nu| T <= 1`), divided by the trace the readout gives
    /// the initial block (its slaved part adds to it).
    pub(crate) fn horizon_bound(&self, ground: &Dense, horizon: f64, fraction: f64) -> Result<Vec<f64>, String> {
        if !(horizon.is_finite() && horizon > 0.) {
            return Err("The observation horizon must be a finite positive time".into());
        }
        let modal = self.to_modal(0, ground)?;
        let floor = self.slow_spectrum().floor;
        let n = self.n;
        let mut bound = vec![0.; n];
        let mut trace = 0.;
        for (nu, readout) in &self.readout {
            for (k, &alpha) in self.eigenvalues.iter().enumerate() {
                let c = modal.data[k];
                if cabs(c) == 0. {
                    continue;
                }
                let total = cadd(alpha, complex(0., *nu));
                for s in 0..readout.rows.min(n) {
                    trace += cmul(readout.at(s, k), c).re;
                }
                if cabs(total) <= floor || total.im.abs() * horizon > 1. {
                    continue;
                }
                let weight = (fraction * total.re * horizon).exp();
                for (s, entry) in bound.iter_mut().enumerate().take(readout.rows.min(n)) {
                    *entry += cabs(cmul(readout.at(s, k), c)) * weight;
                }
            }
        }
        if !(trace.is_finite() && trace > 0.5) || bound.iter().any(|x| !x.is_finite()) {
            return Err(format!("The slow generator reads the initial state with trace {trace:e}; the tolerance was not relaxed"));
        }
        Ok(bound.into_iter().map(|x| x / trace).collect())
    }
    pub fn belongs_to(&self, prepared: &OwnedPrepared) -> bool { Arc::ptr_eq(&self.owner, &prepared.primitives) }
    pub fn dimension(&self) -> usize { self.generator.rows }
    pub(crate) fn ground_entries(&self) -> usize { self.g * self.g }
    pub(crate) fn slot_of(&self, charge: &[i32]) -> Option<usize> { self.slots.iter().position(|s| s == charge) }
    /// Whether a charge at envelope frequency `nu` carries a slow ground
    /// coherence (the slot criterion; slots are the slow charges two
    /// significant couplings reach).
    pub(crate) fn is_slow(&self, nu: f64) -> bool { slow_ground_coherence(nu, &self.ground_energy, self.diagnostics.slow_frequency_cutoff) }

    /// Non-ground inverse of a slow charge for a field with a slow envelope,
    /// `(i nu_c - L_0)_{QQ}^{-1}` (cached).
    fn slow_charge_solver(&self, slot: usize) -> Result<Arc<BlockSolver>, String> {
        let elimination = self.elimination.as_ref().ok_or("Slow dual has no elimination structure")?;
        let mut cache = elimination.slow_cache.lock().map_err(|_| "Slow dual solver cache poisoned".to_string())?;
        if let Some(solver) = cache.get(&slot) {
            return Ok(Arc::clone(solver));
        }
        let z = complex(0., self.slot_frequency[slot]);
        let solver = Arc::new(BlockSolver::NonGround(NonGroundSolver::new(z, Arc::clone(&elimination.blocks), &self.lapack)?));
        cache.insert(slot, Arc::clone(&solver));
        Ok(solver)
    }

    /// Exact inverse `(i nu_k - L_0)^{-1}` of a fast charge block (cached).
    fn fast_charge_solver(&self, k: &Charge) -> Result<Arc<BlockSolver>, String> {
        let elimination = self.elimination.as_ref().ok_or("Slow dual has no elimination structure")?;
        let mut cache = elimination.cache.lock().map_err(|_| "Slow dual solver cache poisoned".to_string())?;
        if let Some(solver) = cache.get(k) {
            return Ok(Arc::clone(solver));
        }
        let nu = self.owner.hilbert.source.phases.frequency(k).map(|f| f.hi + f.lo)?;
        let solver = FullBlockSolver::new(complex(0., nu), Arc::clone(&elimination.sylvester), &self.frame, &self.lapack)?;
        let solver = Arc::new(BlockSolver::Full(solver));
        cache.insert(k.clone(), Arc::clone(&solver));
        Ok(solver)
    }

    /// Markov slaving of a field at a fast (non-slow) charge `k`: every
    /// non-ground entry is eliminated with the exact charge-block inverse,
    /// `x = (i nu_k - L_0)^{-1} y`, and returned one Liouvillian hop into the
    /// slow slots, `s_c = P L_{c-k} x`. The ground block of `y` is ignored here
    /// (the caller feeds it to the central slot with its physical phase).
    /// `field` holds frame entries (row a*n+b) in columns; the result lists
    /// `(slot, g^2 x columns)` blocks.
    pub(crate) fn slaved_returns(&self, k: &[i32], field: &Dense) -> Result<Vec<(usize, Dense)>, String> {
        self.slaved_returns_all(&[(k.to_vec(), field)], field.cols)
    }

    /// `slaved_returns` for several fast charges at once: the eliminations of
    /// one charge run as one batch, and the non-ground parts that reach a slow
    /// charge are summed over all charges before their (second-hop)
    /// elimination there, which is linear and shared.
    pub(crate) fn slaved_returns_all(&self, fields: &[(Charge, &Dense)], columns: usize) -> Result<Vec<(usize, Dense)>, String> {
        let (n, g) = (self.n, self.g);
        let g2 = g * g;
        let lap = &self.lapack;
        let mut out: BTreeMap<usize, Dense> = BTreeMap::new();
        let mut second: BTreeMap<usize, Vec<Option<Dense>>> = BTreeMap::new();
        for (k, field) in fields {
            if field.cols != columns || self.slot_of(k).is_some() {
                continue;
            }
            let landing: Vec<(&ChargeTerm, usize, Charge)> = self
                .frame
                .terms
                .iter()
                .filter_map(|term| {
                    let c = add_charge(k, &term.charge, false).ok()?;
                    self.slot_of(&c).map(|slot| (term, slot, c))
                })
                .collect();
            if landing.is_empty() {
                continue;
            }
            let mut ys: Vec<Dense> = Vec::new();
            let mut index: Vec<usize> = Vec::new();
            for column in 0..columns {
                let mut y = Dense::zeros(n, n);
                let mut any = false;
                for entry in 0..n * n {
                    let (a, b) = (entry / n, entry % n);
                    if a < g && b < g {
                        continue;
                    }
                    let v = field.at(entry, column);
                    if v.re != 0. || v.im != 0. {
                        any = true;
                        y.set(a, b, v);
                    }
                }
                if any {
                    ys.push(y);
                    index.push(column);
                }
            }
            if ys.is_empty() {
                continue;
            }
            let refs: Vec<&Dense> = ys.iter().collect();
            let xs = match self.fast_charge_solver(k)?.as_ref() {
                BlockSolver::Full(solver) => solver.solve_all(&refs, &self.frame, lap)?,
                other => refs.iter().map(|y| other.solve(y, &self.frame, lap)).collect::<Result<_, _>>()?,
            };
            drop(ys);
            let x_refs: Vec<&Dense> = xs.iter().collect();
            for (term, slot, _) in &landing {
                let images = self.frame.apply_all(term, &x_refs, lap)?;
                let block = out.entry(*slot).or_insert_with(|| Dense::zeros(g2, columns));
                let pending = second.entry(*slot).or_insert_with(|| vec![None; columns]);
                for (image, &column) in images.iter().zip(&index) {
                    let w = pending[column].get_or_insert_with(|| Dense::zeros(n, n));
                    for b in 0..n {
                        for a in 0..n {
                            let v = image.at(a, b);
                            if a < g && b < g {
                                block.add_at(a * g + b, column, v);
                            } else {
                                w.add_at(a, b, v);
                            }
                        }
                    }
                }
            }
        }
        // Second hop: the non-ground part arriving at a slow charge relaxes
        // there (static elimination of a slow envelope) and returns its
        // ground block, as in J_eff.
        for (slot, pending) in second {
            let c = &self.slots[slot];
            let solver = self.slow_charge_solver(slot)?;
            for (column, w) in pending.into_iter().enumerate() {
                let Some(w) = w else { continue };
                if w.frobenius() == 0. {
                    continue;
                }
                let x2 = solver.solve(&w, &self.frame, lap)?;
                for term in &self.frame.terms {
                    let c2 = add_charge(c, &term.charge, false)?;
                    if let Some(slot2) = self.slot_of(&c2) {
                        let ground = self.frame.apply_ground_output(term, &x2, &self.lapack)?;
                        let block = out.entry(slot2).or_insert_with(|| Dense::zeros(g2, columns));
                        for (row, v) in ground.iter().enumerate() {
                            block.add_at(row, column, *v);
                        }
                    }
                }
            }
        }
        Ok(out.into_iter().collect())
    }

    /// Transform a slot-block vector (g^2 entries placed in `slot`) into
    /// eigen-coordinates of `J_eff`: `W[:, block] v`.
    pub(crate) fn to_modal(&self, slot: usize, block: &Dense) -> Result<Dense, String> {
        let g2 = self.ground_entries();
        if block.rows != g2 {
            return Err("Slot source block must have g^2 rows".into());
        }
        // W[:, slot block] read in place (column-major, leading dimension dim).
        let dim = self.dimension();
        let mut out = Dense::zeros(dim, block.cols);
        self.lapack.complex_gemm().multiply(dim, block.cols, g2, ONE, &self.inverse_vectors.data[slot * g2 * dim..], dim, &block.data, g2, ZERO, &mut out.data, dim)?;
        Ok(out)
    }
}

impl DressedFrame {
    #[allow(clippy::too_many_arguments)]
    fn new(
        left: &Dense,
        right: &Dense,
        population_readout: &[(usize, usize, usize, C)],
        recycling0: &[(usize, usize, C)],
        g: usize,
        slow_inverse_vectors: &Dense,
        slots: usize,
        lap: &Lapack,
    ) -> Result<Self, String> {
        let n = left.rows;
        let e = n - g;
        let eigen = |a: &Dense| -> Result<(Vec<C>, Dense, Dense, f64), String> {
            let e = robust_eigen(a, None, lap)?;
            Ok((e.values, e.vectors, e.inverse, e.condition))
        };
        let (left_values, left_vectors, left_inverse, left_condition) = eigen(left)?;
        let (right_values, right, right_inverse, right_condition) = eigen(right)?;
        let condition = left_condition.max(right_condition);
        if !(condition.is_finite() && condition < 1e8) {
            return Err(format!("Dressed no-jump eigenbasis is ill-conditioned ({condition:e})"));
        }
        let n2 = n * n;
        // rho[s, i*n+k] = sum_ab w_sab V_L[a,i] V_R^-1[k,b]
        let states = population_readout.iter().map(|r| r.0 + 1).max().unwrap_or(0);
        let mut readout = Dense::zeros(states, n2);
        for &(s, a, b, w) in population_readout {
            for i in 0..n {
                let li = cmul(w, left_vectors.at(a, i));
                for k in 0..n {
                    readout.add_at(s, i * n + k, cmul(li, right_inverse.at(k, b)));
                }
            }
        }
        // recycling[ab, i*n+k] = sum_{i'j'} R0[ab, i'j'] V_L[g+i', i] V_R^-1[k, g+j']
        //                     = (V_L,E^T R0_ab V_R^-1_E^T)[i, k] with the e x e
        // block R0_ab: two products for all ab, whatever the map's sparsity.
        let g2 = g * g;
        let mut left_e = Dense::zeros(n, e);
        let mut right_e = Dense::zeros(e, n);
        for j in 0..e {
            for i in 0..n {
                left_e.set(i, j, left_vectors.at(g + j, i));
                right_e.set(j, i, right_inverse.at(i, g + j));
            }
        }
        let mut blocks = Dense::zeros(e, g2 * e);
        for &(ab, ij, v) in recycling0 {
            blocks.set(ij / e, ab * e + ij % e, v);
        }
        let product = left_e.multiply(&blocks, lap)?;
        let mut stack = Dense::zeros(g2 * n, e);
        for ab in 0..g2 {
            for j in 0..e {
                for i in 0..n {
                    stack.set(ab * n + i, j, product.at(i, ab * e + j));
                }
            }
        }
        let sandwiched = stack.multiply(&right_e, lap)?;
        let mut recycling = Dense::zeros(g2, n2);
        for ab in 0..g2 {
            for i in 0..n {
                for k in 0..n {
                    recycling.set(ab, i * n + k, sandwiched.at(ab * n + i, k));
                }
            }
        }
        let dim = slow_inverse_vectors.rows;
        let mut modal_recycling = Vec::with_capacity(slots);
        for s in 0..slots {
            let block = slow_inverse_vectors.block(0..dim, s * g2..(s + 1) * g2);
            modal_recycling.push(block.multiply(&recycling, lap)?);
        }
        Ok(Self { left_values, right_values, left: left_vectors, left_inverse, right, right_inverse, readout, modal_recycling, condition })
    }

    /// `V_L^-1 X V_R` for an n x n frame matrix.
    pub fn to_eigen(&self, x: &Dense, lap: &Lapack) -> Result<Dense, String> { self.left_inverse.multiply(x, lap)?.multiply(&self.right, lap) }
}

/// Output times evaluated together in the family products (bounds memory).
const TIME_BLOCK: usize = 256;

// ---------------------------------------------------------------------------
// Exponential series of the population error
// ---------------------------------------------------------------------------

/// Frequency key (1e-9 rad/us resolution): equal frequencies share a family.
pub(crate) fn frequency_key(f: f64) -> i64 { (f * 1e9).round() as i64 }

/// Output times; a uniform grid `t_i = i h` (most requests) is evaluated by
/// the recurrence `exp(mu t_{i+1}) = exp(mu t_i) exp(mu h)`.
pub(crate) struct TimeGrid<'a> {
    pub times: &'a [f64],
    step: Option<f64>,
    pub maximum: f64,
}

impl<'a> TimeGrid<'a> {
    pub fn new(times: &'a [f64]) -> Self {
        let maximum = times.iter().copied().fold(0_f64, f64::max);
        let count = times.len();
        let step = if count >= 2 && times[0] == 0. {
            let h = times[count - 1] / (count - 1) as f64;
            let tolerance = 1e-12 * maximum.max(1.);
            (h > 0. && times.iter().enumerate().all(|(i, &t)| (t - h * i as f64).abs() <= tolerance)).then_some(h)
        } else {
            None
        };
        Self { times, step, maximum }
    }

    /// Calls `visit(i, exp(mu t_i))` for every time where `exp(Re mu t)` is
    /// not negligible (decaying exponents stop early on a uniform grid).
    fn for_each(&self, mu: C, mut visit: impl FnMut(usize, C)) {
        match self.step {
            Some(h) => {
                let ratio = cexp(cscale(mu, h));
                let mut e = ONE;
                for i in 0..self.times.len() {
                    if mu.re < 0. && mu.re * self.times[i] < NEGLIGIBLE_EXPONENT {
                        break;
                    }
                    visit(i, e);
                    e = cmul(e, ratio);
                }
            }
            None => {
                for (i, &t) in self.times.iter().enumerate() {
                    if mu.re * t < NEGLIGIBLE_EXPONENT {
                        continue;
                    }
                    visit(i, cexp(cscale(mu, t)));
                }
            }
        }
    }
}

/// `out[t, s] += sum_c W[s, c] exp(mu_c t)`: each exponent touches only the
/// samples where it is not negligible, so fast dressed modes cost a few early
/// samples and the cost hardly depends on the number of output times.
pub(crate) fn add_natural(out: &mut [C], states: usize, weights: &Dense, exponents: &[C], grid: &TimeGrid) {
    for (c, &mu) in exponents.iter().enumerate() {
        let column = &weights.data[c * weights.rows..c * weights.rows + states];
        if column.iter().all(|z| z.re == 0. && z.im == 0.) {
            continue;
        }
        grid.for_each(mu, |i, e| {
            for (o, w) in out[i * states..(i + 1) * states].iter_mut().zip(column) {
                *o = cadd(*o, cmul(*w, e));
            }
        });
    }
}

/// The signed population error as exponential families: sources of the slow
/// error (propagated exactly through `J_eff`) and direct populations of fast
/// responses, assembled per charge and evaluated once on the output grid.
pub(crate) struct Series {
    /// Candidate exponents `beta_j`, shared by every family.
    pub(crate) beta: Vec<C>,
    pub(crate) t_max: f64,
    /// Slow sources `v_j exp((beta_j + i f) t)` per frequency `f` (dim x m).
    pub(crate) slow: BTreeMap<i64, (f64, Dense)>,
    /// Dressed recycling sources into slot `s`:
    /// `h_ik M_s[:, ik] exp((a_i + b_k + i o) t)` per offset `o`.
    pub(crate) dressed_slow: BTreeMap<(usize, i64), (f64, Vec<C>)>,
    /// Single exponential sources `v exp(mu t)` (split resonant pairs),
    /// summed per exponent (`exponent_key`).
    pub(crate) slow_points: BTreeMap<(u64, u64), (C, Vec<C>)>,
    /// Clustered pair sources `v e_t[mu1, mu2]`.
    pub(crate) slow_pairs: Vec<(C, C, Vec<C>)>,
    /// Initial slow error (J eigen-coordinates).
    pub(crate) initial: Option<Vec<C>>,
    /// Direct populations `F[:, j] exp((beta_j + i f) t)` per `f` (n x m).
    pub(crate) direct: BTreeMap<i64, (f64, Dense)>,
    /// Direct dressed amplitudes `h_ik` of `exp((a_i + b_k) t)` (the physical
    /// phase of a charge cancels its envelope shift, so all charges merge).
    pub(crate) dressed_direct: Option<Vec<C>>,
    /// Direct near-resonant populations `p e_t[x, y]`.
    pub(crate) direct_near: Vec<(Vec<C>, C, C)>,
    /// Direct single exponentials `p exp(mu t)` (split near-resonant
    /// populations), summed per exponent.
    pub(crate) direct_points: BTreeMap<(u64, u64), (C, Vec<C>)>,
}

/// Exact key of an exponent: split pair nodes computed from the same inputs
/// share it, so their terms merge into one exponential.
fn exponent_key(mu: C) -> (u64, u64) { (mu.re.to_bits(), mu.im.to_bits()) }

/// `points[mu] += scale * v`.
fn add_point(points: &mut BTreeMap<(u64, u64), (C, Vec<C>)>, mu: C, scale: C, v: &[C]) {
    let entry = points.entry(exponent_key(mu)).or_insert_with(|| (mu, vec![ZERO; v.len()]));
    for (a, b) in entry.1.iter_mut().zip(v) {
        *a = cadd(*a, cmul(scale, *b));
    }
}

fn accumulate(target: &mut Dense, values: &Dense) {
    for (a, b) in target.data.iter_mut().zip(&values.data) {
        *a = cadd(*a, *b);
    }
}

/// `target[:, start..start + values.cols] += values` (same row count).
fn accumulate_columns(target: &mut Dense, start: usize, values: &Dense) {
    let rows = values.rows;
    let at = start * rows;
    for (a, b) in target.data[at..at + values.data.len()].iter_mut().zip(&values.data) {
        *a = cadd(*a, *b);
    }
}

impl Series {
    pub(crate) fn new(beta: &[C], t_max: f64) -> Self {
        Self {
            beta: beta.to_vec(),
            t_max,
            slow: BTreeMap::new(),
            dressed_slow: BTreeMap::new(),
            slow_points: BTreeMap::new(),
            slow_pairs: Vec::new(),
            initial: None,
            direct: BTreeMap::new(),
            dressed_direct: None,
            direct_near: Vec::new(),
            direct_points: BTreeMap::new(),
        }
    }

    /// Slow source columns for candidate exponents `start..start + cols`.
    pub(crate) fn add_slow(&mut self, frequency: f64, start: usize, columns: &Dense) {
        let m = self.beta.len();
        let entry = self.slow.entry(frequency_key(frequency)).or_insert_with(|| (frequency, Dense::zeros(columns.rows, m)));
        accumulate_columns(&mut entry.1, start, columns);
    }

    /// Direct population columns for candidate exponents `start..start + cols`.
    pub(crate) fn add_direct(&mut self, frequency: f64, start: usize, columns: &Dense) {
        let m = self.beta.len();
        let entry = self.direct.entry(frequency_key(frequency)).or_insert_with(|| (frequency, Dense::zeros(columns.rows, m)));
        accumulate_columns(&mut entry.1, start, columns);
    }

    pub(crate) fn add_dressed_slow(&mut self, slot: usize, offset: f64, amplitudes: &[C]) {
        let entry = self.dressed_slow.entry((slot, frequency_key(offset))).or_insert_with(|| (offset, vec![ZERO; amplitudes.len()]));
        for (a, b) in entry.1.iter_mut().zip(amplitudes) {
            *a = cadd(*a, *b);
        }
    }

    pub(crate) fn add_dressed_direct(&mut self, amplitudes: &[C]) {
        let target = self.dressed_direct.get_or_insert_with(|| vec![ZERO; amplitudes.len()]);
        for (a, b) in target.iter_mut().zip(amplitudes) {
            *a = cadd(*a, *b);
        }
    }

    /// Moves the dressed amplitudes of `other` (they do not depend on its
    /// exponents) into this series.
    pub(crate) fn take_dressed(&mut self, other: &mut Series) {
        if let Some(amplitudes) = other.dressed_direct.take() {
            self.add_dressed_direct(&amplitudes);
        }
        for ((slot, _), (offset, amplitudes)) in std::mem::take(&mut other.dressed_slow) {
            self.add_dressed_slow(slot, offset, &amplitudes);
        }
    }

    pub(crate) fn add_initial(&mut self, modal: &[C]) {
        let target = self.initial.get_or_insert_with(|| vec![ZERO; modal.len()]);
        for (a, b) in target.iter_mut().zip(modal) {
            *a = cadd(*a, *b);
        }
    }

    /// `v e_t[mu1, mu2]`: two exponentials when the nodes separate on the
    /// horizon, the divided-difference form otherwise.
    pub(crate) fn add_pair(&mut self, first: C, second: C, v: Vec<C>) {
        let delta = csub(first, second);
        if cabs(delta) * self.t_max > 1. {
            let scale = cdiv(ONE, delta);
            add_point(&mut self.slow_points, first, scale, &v);
            add_point(&mut self.slow_points, second, csub(ZERO, scale), &v);
        } else {
            self.slow_pairs.push((first, second, v));
        }
    }

    /// Single exponential slow source `v exp(mu t)` (merged per exponent).
    pub(crate) fn add_slow_point(&mut self, mu: C, v: &[C]) {
        add_point(&mut self.slow_points, mu, ONE, v);
    }

    /// Direct populations `p exp(mu t)` (merged per exponent).
    pub(crate) fn add_direct_point(&mut self, mu: C, p: &[C]) {
        add_point(&mut self.direct_points, mu, ONE, p);
    }

    /// Direct populations `p e_t[first, second]`, split like `add_pair`: a
    /// slow halo mode has one such term per candidate column and resonant
    /// coherence, and the split ones merge into one exponential per column
    /// and per coherence.
    pub(crate) fn add_direct_pair(&mut self, first: C, second: C, p: Vec<C>) {
        let delta = csub(first, second);
        if cabs(delta) * self.t_max > 1. {
            let scale = cdiv(ONE, delta);
            add_point(&mut self.direct_points, first, scale, &p);
            add_point(&mut self.direct_points, second, csub(ZERO, scale), &p);
        } else {
            self.direct_near.push((p, first, second));
        }
    }
}

impl SlowDual {
    /// Evaluate a series on the output grid: `[time, state]` complex
    /// populations (the real part is the estimate).
    pub(crate) fn evaluate(&self, series: &Series, grid: &TimeGrid) -> Result<Vec<C>, String> {
        self.evaluate_parts(series, grid, None)
    }

    /// `evaluate`, optionally returning the homogeneous slow amplitudes
    /// (added to `homogeneous_out`) instead of evaluating them, so partial
    /// series can be summed and their common `exp(J t)` part read out once.
    pub(crate) fn evaluate_parts(&self, series: &Series, grid: &TimeGrid, homogeneous_out: Option<&mut Vec<C>>) -> Result<Vec<C>, String> {
        self.evaluate_with(series, grid, &self.readout, true, homogeneous_out)
    }

    /// `evaluate_parts` through other readout groups of the slow state
    /// (`rows x dim` each, at the slots' phases; `rows` columns per time in
    /// the output), with or without the direct populations of the fast
    /// responses (which only the population readout has).
    pub(crate) fn evaluate_with(
        &self,
        series: &Series,
        grid: &TimeGrid,
        readout_groups: &[(f64, Dense)],
        direct: bool,
        homogeneous_out: Option<&mut Vec<C>>,
    ) -> Result<Vec<C>, String> {
        let n = self.n;
        let rows = readout_groups.first().map_or(n, |(_, r)| r.rows);
        if direct && rows != n {
            return Err("Direct populations need the population readout".into());
        }
        let dim = self.dimension();
        let nt = grid.times.len();
        let t_max = grid.maximum;
        let alpha = &self.eigenvalues;
        let lap = &self.lapack;
        let m = series.beta.len();
        let mut out = vec![ZERO; nt * rows];
        let mut homogeneous = series.initial.clone().unwrap_or_else(|| vec![ZERO; dim]);
        // Amplitude-level terms v e_t[mu, alpha_k] at slow mode k.
        let mut near: Vec<(usize, C, C)> = Vec::new();
        // Particular solution of x' = diag(alpha) x + v exp(mu t):
        // v / (mu - alpha); the homogeneous part keeps x(0) unchanged.
        let particular = |v: C, mu: C, k: usize, homogeneous: &mut Vec<C>, near: &mut Vec<(usize, C, C)>| -> C {
            let delta = csub(mu, alpha[k]);
            if cabs(delta) * t_max <= 1. {
                near.push((k, v, mu));
                ZERO
            } else {
                let y = cdiv(v, delta);
                homogeneous[k] = csub(homogeneous[k], y);
                y
            }
        };
        // 1. Slow families -> readout families (n x m per frequency).
        let mut families: BTreeMap<i64, (f64, Dense)> = if direct { series.direct.clone() } else { BTreeMap::new() };
        for (f, v) in series.slow.values() {
            let mut c = Dense::zeros(dim, m);
            for j in 0..m {
                let mu = cadd(series.beta[j], complex(0., *f));
                for k in 0..dim {
                    let x = v.at(k, j);
                    if x.re != 0. || x.im != 0. {
                        let y = particular(x, mu, k, &mut homogeneous, &mut near);
                        c.set(k, j, y);
                    }
                }
            }
            for (omega, readout) in readout_groups {
                let values = readout.multiply(&c, lap)?;
                let key = frequency_key(f + omega);
                let entry = families.entry(key).or_insert_with(|| (f + omega, Dense::zeros(rows, m)));
                accumulate(&mut entry.1, &values);
            }
        }
        // 2. Dressed families (n x n^2 weights per exponent offset).
        let mut naturals: BTreeMap<i64, (f64, Dense)> = BTreeMap::new();
        if let Some(dressed) = &self.dressed {
            let n2 = n * n;
            let lambda: Vec<C> = (0..n2).map(|ik| cadd(dressed.left_values[ik / n], dressed.right_values[ik % n])).collect();
            if let Some(h) = series.dressed_direct.as_ref().filter(|_| direct) {
                let mut w = dressed.readout.clone();
                for (ik, &x) in h.iter().enumerate() {
                    for s in 0..w.rows {
                        let value = cmul(w.at(s, ik), x);
                        w.set(s, ik, value);
                    }
                }
                naturals.insert(0, (0., w));
            }
            for ((slot, _), (o, h)) in &series.dressed_slow {
                let recycling = &dressed.modal_recycling[*slot];
                let mut c = Dense::zeros(dim, n2);
                for (ik, &x) in h.iter().enumerate() {
                    if x.re == 0. && x.im == 0. {
                        continue;
                    }
                    let mu = cadd(lambda[ik], complex(0., *o));
                    for k in 0..dim {
                        let v = cmul(recycling.at(k, ik), x);
                        if v.re != 0. || v.im != 0. {
                            let y = particular(v, mu, k, &mut homogeneous, &mut near);
                            c.set(k, ik, y);
                        }
                    }
                }
                for (omega, readout) in readout_groups {
                    let values = readout.multiply(&c, lap)?;
                    let key = frequency_key(o + omega);
                    let entry = naturals.entry(key).or_insert_with(|| (o + omega, Dense::zeros(rows, n2)));
                    accumulate(&mut entry.1, &values);
                }
            }
            for (o, w) in naturals.values() {
                let exponents: Vec<C> = lambda.iter().map(|l| cadd(*l, complex(0., *o))).collect();
                add_natural(&mut out, rows, w, &exponents, grid);
            }
        } else if series.dressed_direct.is_some() || !series.dressed_slow.is_empty() {
            return Err("Dressed sources without a dressed frame".into());
        }
        // 3. Single exponential sources: particular amplitudes per readout group.
        for (mu, v) in series.slow_points.values() {
            let mut c = Dense::zeros(dim, 1);
            for k in 0..dim {
                if v[k].re != 0. || v[k].im != 0. {
                    let y = particular(v[k], *mu, k, &mut homogeneous, &mut near);
                    c.set(k, 0, y);
                }
            }
            for (omega, readout) in readout_groups {
                let values = readout.multiply(&c, lap)?;
                add_natural(&mut out, rows, &values, &[cadd(*mu, complex(0., *omega))], grid);
            }
        }
        // 4. Families: one product per block of times, phases by recurrence.
        if !families.is_empty() && m > 0 {
            let list: Vec<&(f64, Dense)> = families.values().collect();
            let count = list.len();
            let mut stacked = Dense::zeros(rows * count, m);
            for (q, (_, f)) in list.iter().enumerate() {
                for j in 0..m {
                    for s in 0..rows {
                        stacked.set(q * rows + s, j, f.at(s, j));
                    }
                }
            }
            let mut start = 0;
            while start < nt {
                let width = TIME_BLOCK.min(nt - start);
                let mut phases = Dense::zeros(m, width);
                for j in 0..m {
                    let mu = series.beta[j];
                    for i in 0..width {
                        phases.set(j, i, cexp(cscale(mu, grid.times[start + i])));
                    }
                }
                let values = stacked.multiply(&phases, lap)?;
                for (q, (f, _)) in list.iter().enumerate() {
                    for i in 0..width {
                        let t = grid.times[start + i];
                        let phase = if *f == 0. { ONE } else { cexp(complex(0., f * t)) };
                        let row = &mut out[(start + i) * rows..(start + i + 1) * rows];
                        for s in 0..rows {
                            row[s] = cadd(row[s], cmul(phase, values.at(q * rows + s, i)));
                        }
                    }
                }
                start += width;
            }
        }
        // 5. Clustered pair sources v e_t[mu1, mu2]. For a slow mode k away
        // from both nodes, e_t[mu1, mu2, alpha] = (e_t[mu1, mu2] - e_t[mu2,
        // alpha]) / (mu1 - alpha): a populations term in e_t[mu1, mu2], an
        // exponential exp(mu2 t) and a homogeneous part. Modes close to the
        // nodes keep the full divided difference.
        let mut amplitudes: Option<Dense> = None;
        let mut pair_terms: Vec<(Vec<C>, C, C)> = Vec::new();
        for (mu1, mu2, v) in &series.slow_pairs {
            let mut via_pair = Dense::zeros(dim, 1);
            let mut point = Dense::zeros(dim, 1);
            let mut close: Vec<usize> = Vec::new();
            for k in 0..dim {
                if v[k].re == 0. && v[k].im == 0. {
                    continue;
                }
                let to_mu1 = csub(*mu1, alpha[k]);
                let to_mu2 = csub(*mu2, alpha[k]);
                if cabs(to_mu1) * t_max > 1. && cabs(to_mu2) * t_max > 1. {
                    let a = cdiv(v[k], to_mu1);
                    let c = csub(ZERO, cdiv(a, to_mu2));
                    via_pair.set(k, 0, a);
                    point.set(k, 0, c);
                    homogeneous[k] = csub(homogeneous[k], c);
                } else {
                    close.push(k);
                }
            }
            for (omega, readout) in readout_groups {
                let shift = complex(0., *omega);
                let p = readout.multiply(&via_pair, lap)?;
                pair_terms.push((p.data, cadd(*mu1, shift), cadd(*mu2, shift)));
                let values = readout.multiply(&point, lap)?;
                add_natural(&mut out, rows, &values, &[cadd(*mu2, shift)], grid);
            }
            if !close.is_empty() {
                let a = amplitudes.get_or_insert_with(|| Dense::zeros(dim, nt));
                for (it, &t) in grid.times.iter().enumerate() {
                    for &k in &close {
                        a.add_at(k, it, cmul(v[k], exp_divided_3(*mu1, *mu2, alpha[k], t)));
                    }
                }
            }
        }
        // 6. Near-resonant slow terms: amplitude v e_t[mu, alpha_k].
        if !near.is_empty() {
            let a = amplitudes.get_or_insert_with(|| Dense::zeros(dim, nt));
            for (it, &t) in grid.times.iter().enumerate() {
                for &(k, v, mu) in &near {
                    a.add_at(k, it, cmul(v, exp_divided_2(mu, alpha[k], t)));
                }
            }
        }
        if let Some(a) = &amplitudes {
            for (omega, readout) in readout_groups {
                let values = readout.multiply(a, lap)?;
                for (it, &t) in grid.times.iter().enumerate() {
                    let phase = if *omega == 0. { ONE } else { cexp(complex(0., omega * t)) };
                    for s in 0..rows {
                        out[it * rows + s] = cadd(out[it * rows + s], cmul(phase, values.at(s, it)));
                    }
                }
            }
        }
        for (p, x, y) in &pair_terms {
            for (it, &t) in grid.times.iter().enumerate() {
                let value = exp_divided_2(*x, *y, t);
                for s in 0..rows {
                    out[it * rows + s] = cadd(out[it * rows + s], cmul(p[s], value));
                }
            }
        }
        // 7. Homogeneous slow solution exp(alpha t) x0 through every group.
        if let Some(target) = homogeneous_out {
            for (t, h) in target.iter_mut().zip(&homogeneous) {
                *t = cadd(*t, *h);
            }
        } else {
            self.add_homogeneous(&mut out, &homogeneous, grid, readout_groups);
        }
        // 8. Direct near-resonant populations.
        let (direct_points, direct_near) = if direct { (&series.direct_points, &series.direct_near[..]) } else { (&BTreeMap::new(), &[][..]) };
        for (mu, p) in direct_points.values() {
            add_natural(&mut out, n, &Dense { rows: n, cols: 1, data: p.clone() }, &[*mu], grid);
        }
        for (p, x, y) in direct_near {
            for (it, &t) in grid.times.iter().enumerate() {
                let value = exp_divided_2(*x, *y, t);
                for s in 0..n {
                    out[it * n + s] = cadd(out[it * n + s], cmul(p[s], value));
                }
            }
        }
        if out.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) {
            return Err("Nonfinite population error series".into());
        }
        Ok(out)
    }
}

impl SlowDual {
    /// `out[t, s] += sum_groups exp(i omega t) (R V diag(h) exp(alpha t))_s`.
    pub(crate) fn add_homogeneous(&self, out: &mut [C], homogeneous: &[C], grid: &TimeGrid, readout_groups: &[(f64, Dense)]) {
        let alpha = &self.eigenvalues;
        for (omega, readout) in readout_groups {
            let rows = readout.rows;
            let mut w = readout.clone();
            for (k, &h) in homogeneous.iter().enumerate() {
                for s in 0..rows {
                    let value = cmul(w.at(s, k), h);
                    w.set(s, k, value);
                }
            }
            let exponents: Vec<C> = alpha.iter().map(|a| cadd(*a, complex(0., *omega))).collect();
            add_natural(out, rows, &w, &exponents, grid);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lap() -> Lapack { Lapack::load().unwrap() }

    fn random_dense(rows: usize, cols: usize, seed: u64) -> Dense {
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        let mut m = Dense::zeros(rows, cols);
        for k in 0..rows * cols {
            m.data[k] = complex(next(), next());
        }
        m
    }

    /// Independent dense frame Liouvillian charge `p` (row-major vec index
    /// `i*n+j`), built from the gauged/rotated H and the ORIGINAL collapse
    /// channels (not the compiled Gamma/recycling used by the estimator).
    fn dense_charge(p: &PrimitivePreparation, charge: &[i32]) -> Dense {
        let source = &p.hilbert.source;
        let n = source.n;
        let d = n * n;
        let mut out = Dense::zeros(d, d);
        let mat = |m: &[C], i: usize, j: usize| m[i * n + j];
        if let Some(h) = source.hamiltonian.get(charge) {
            for i in 0..n {
                for j in 0..n {
                    for k in 0..n {
                        // -i H X: (i,j) <- H[i,k] X[k,j]
                        out.add_at(i * n + j, k * n + j, minus_i(mat(h, i, k)));
                        // +i X H: (i,j) <- X[i,k] H[k,j]
                        out.add_at(i * n + j, i * n + k, csub(ZERO, minus_i(mat(h, k, j))));
                    }
                }
            }
        }
        for channel in &source.collapse {
            for (qa, ca) in channel {
                for (qb, cb) in channel {
                    let q: Vec<i32> = qa.iter().zip(qb).map(|(a, b)| a - b).collect();
                    if q != charge {
                        continue;
                    }
                    // gamma = Cb^H Ca
                    let mut gamma = vec![ZERO; d];
                    for i in 0..n {
                        for j in 0..n {
                            for k in 0..n {
                                gamma[i * n + j] = cadd(gamma[i * n + j], cmul(cconj(mat(cb, k, i)), mat(ca, k, j)));
                            }
                        }
                    }
                    for i in 0..n {
                        for j in 0..n {
                            for k in 0..n {
                                for l in 0..n {
                                    // Ca X Cb^H: (i,j) <- Ca[i,k] X[k,l] conj(Cb[j,l])
                                    out.add_at(i * n + j, k * n + l, cmul(mat(ca, i, k), cconj(mat(cb, j, l))));
                                }
                                out.add_at(i * n + j, k * n + j, cscale(gamma[i * n + k], -0.5));
                                out.add_at(i * n + j, i * n + k, cscale(gamma[k * n + j], -0.5));
                            }
                        }
                    }
                }
            }
        }
        out
    }

    fn matrix_to_vec(x: &Dense) -> Vec<C> {
        let n = x.rows;
        (0..n * n).map(|k| x.at(k / n, k % n)).collect()
    }
    fn vec_to_matrix(v: &[C], n: usize) -> Dense {
        let mut x = Dense::zeros(n, n);
        for k in 0..n * n {
            x.set(k / n, k % n, v[k]);
        }
        x
    }
    fn dense_apply(op: &Dense, v: &[C]) -> Vec<C> {
        (0..op.rows).map(|i| (0..op.cols).fold(ZERO, |s, j| cadd(s, cmul(op.at(i, j), v[j])))).collect()
    }

    #[test]
    fn slow_dual_unrepresented_readout_reads_the_coupling_norm() {
        // ||R xhat|| = ||K V xhat|| for couplings into two charges whose
        // stacked rows outnumber the slow dimension.
        let lap = lap();
        let dim = 5;
        let mut coupling = BTreeMap::new();
        coupling.insert(vec![1], random_dense(4, dim, 21));
        coupling.insert(vec![2], random_dense(4, dim, 22));
        let vectors = random_dense(dim, dim, 23);
        let readout = unrepresented_norm_readout(&coupling, &vectors, dim, &lap).unwrap();
        let rows = &readout[0].1;
        assert!(rows.rows <= dim);
        for seed in [24, 25] {
            let x = random_dense(dim, 1, seed);
            let mut expected = 0_f64;
            for k in coupling.values() {
                expected += k.multiply(&vectors, &lap).unwrap().multiply(&x, &lap).unwrap().frobenius().powi(2);
            }
            let actual = rows.multiply(&x, &lap).unwrap().frobenius();
            assert!((actual - expected.sqrt()).abs() < 1e-10 * expected.sqrt(), "{actual} vs {}", expected.sqrt());
        }
        assert!(unrepresented_norm_readout(&BTreeMap::new(), &vectors, dim, &lap).unwrap().is_empty());
    }

    #[test]
    fn slow_dual_frame_liouvillian_matches_original_channels() {
        let owner = super::super::tests::fixture();
        let lap = lap();
        let frame = FrameLiouvillian::from_primitives(owner.primitives()).unwrap();
        let n = frame.n;
        let x = random_dense(n, n, 11);
        for term in &frame.terms {
            let dense = dense_charge(owner.primitives(), &term.charge);
            let expected = dense_apply(&dense, &matrix_to_vec(&x));
            let actual = matrix_to_vec(&frame.apply(term, &x, &lap).unwrap());
            for k in 0..n * n {
                assert!(cabs(csub(actual[k], expected[k])) < 1e-12, "charge {:?} entry {k}", term.charge);
            }
            let ground = frame.apply_ground_output(term, &x, &lap).unwrap();
            // The dense recycling representation (perturbed bases) agrees
            // with the entry list.
            if !term.recycling.is_empty() {
                let (g, e) = (frame.g, frame.e);
                let mut matrix = Dense::zeros(g * g, e * e);
                for &(ab, ij, v) in &term.recycling {
                    matrix.set(ab, ij, v);
                }
                let only_recycling = |recycling_matrix: Option<Dense>| ChargeTerm {
                    charge: term.charge.clone(),
                    hamiltonian: None,
                    gamma: None,
                    recycling: term.recycling.clone(),
                    recycling_matrix,
                    hamiltonian_entries: None,
                    gamma_entries: None,
                    significant: true,
                };
                let ys = [random_dense(n, n, 12), random_dense(n, n, 13)];
                let refs: Vec<&Dense> = ys.iter().collect();
                let listed = frame.recycled(&only_recycling(None), &refs, &lap).unwrap();
                let product = frame.recycled(&only_recycling(Some(matrix)), &refs, &lap).unwrap();
                for (a, b) in listed.data.iter().zip(&product.data) {
                    assert!(cabs(csub(*a, *b)) < 1e-12);
                }
            }
            for a in 0..frame.g {
                for b in 0..frame.g {
                    assert!(cabs(csub(ground[a * frame.g + b], expected[a * n + b])) < 1e-12);
                }
            }
            for a in 0..frame.g {
                for b in 0..frame.g {
                    let mut unit = Dense::zeros(n, n);
                    unit.set(a, b, ONE);
                    let image = matrix_to_vec(&frame.apply_to_ground_unit(term, a, b));
                    let reference = dense_apply(&dense, &matrix_to_vec(&unit));
                    for k in 0..n * n {
                        assert!(cabs(csub(image[k], reference[k])) < 1e-13);
                    }
                }
            }
        }
    }

    #[test]
    fn slow_dual_block_solvers_invert_the_charge_block() {
        let owner = super::super::tests::fixture();
        let lap = lap();
        let frame = FrameLiouvillian::from_primitives(owner.primitives()).unwrap();
        let (n, g, e) = (frame.n, frame.g, frame.e);
        let d = n * n;
        let dense0 = dense_charge(owner.primitives(), &frame.zero_charge());
        let (left, right) = frame.no_jump();
        let blocks = Arc::new(NoJumpBlocks::new(&left, &right, g, e, &lap).unwrap());
        let sylvester = Arc::new(SylvesterFactor { left: Schur::new(&left, &lap).unwrap(), right: Schur::new(&right, &lap).unwrap() });
        let y = random_dense(n, n, 5);
        for z in [complex(0., 7.), complex(0., -14.), complex(0.3, 2.)] {
            // Full block: (z - L_0) x = y.
            let full = FullBlockSolver::new(z, Arc::clone(&sylvester), &frame, &lap).unwrap();
            let x = full.solve(&y, &frame, &lap).unwrap();
            let lx = dense_apply(&dense0, &matrix_to_vec(&x));
            let xv = matrix_to_vec(&x);
            let yv = matrix_to_vec(&y);
            for k in 0..d {
                let residual = csub(csub(cmul(z, xv[k]), lx[k]), yv[k]);
                assert!(cabs(residual) < 1e-10, "full z={z:?} k={k} {}", cabs(residual));
            }
        }
        // Non-ground block at z = 0: Q rows satisfied, ground block of x zero.
        let z = complex(0., 0.);
        let solver = NonGroundSolver::new(z, Arc::clone(&blocks), &lap).unwrap();
        let x = solver.solve(&y, &lap).unwrap();
        let lx = dense_apply(&dense0, &matrix_to_vec(&x));
        let xv = matrix_to_vec(&x);
        let yv = matrix_to_vec(&y);
        for i in 0..n {
            for j in 0..n {
                let k = i * n + j;
                if i < g && j < g {
                    assert_eq!(xv[k].re, 0.);
                    assert_eq!(xv[k].im, 0.);
                } else {
                    let residual = csub(csub(cmul(z, xv[k]), lx[k]), yv[k]);
                    assert!(cabs(residual) < 1e-10, "nonground k={k} {}", cabs(residual));
                }
            }
        }
        let _ = vec_to_matrix(&yv, n);
    }

    /// Dense, independent evaluation of the one-hop Schur complement.
    fn dense_generator(owner: &OwnedPrepared, dual: &SlowDual) -> Dense {
        let p = owner.primitives();
        let frame = &dual.frame;
        let (n, g) = (frame.n, frame.g);
        let d = n * n;
        let g2 = g * g;
        let phases = &p.hilbert.source.phases;
        let nu = |c: &[i32]| phases.frequency(c).map(|f| f.hi + f.lo).unwrap();
        let ground = |k: usize| k / n < g && k % n < g;
        let dense0 = dense_charge(p, &frame.zero_charge());
        let mut out = Dense::zeros(dual.dimension(), dual.dimension());
        for (s, c) in dual.slots.iter().enumerate() {
            for ab in 0..g2 {
                let column = s * g2 + ab;
                out.add_at(column, column, complex(0., -nu(c)));
                let mut unit = vec![ZERO; d];
                unit[(ab / g) * n + ab % g] = ONE;
                let mut second_hop: BTreeMap<usize, Vec<C>> = BTreeMap::new();
                for term in &frame.terms {
                    let c2: Vec<i32> = c.iter().zip(&term.charge).map(|(a, b)| a + b).collect();
                    let mut y = dense_apply(&dense_charge(p, &term.charge), &unit);
                    let slot2 = dual.slots.iter().position(|x| *x == c2);
                    if let Some(s2) = slot2 {
                        for k in 0..d {
                            if ground(k) {
                                out.add_at(s2 * g2 + (k / n) * g + k % n, column, y[k]);
                                y[k] = ZERO;
                            }
                        }
                    }
                    // Dense (i nu - L_0) restricted to the eliminated indices.
                    let eliminated: Vec<usize> = (0..d).filter(|&k| slot2.is_none() || !ground(k)).collect();
                    let size = eliminated.len();
                    let mut m = Dense::zeros(size, size);
                    for (ii, &i) in eliminated.iter().enumerate() {
                        for (jj, &j) in eliminated.iter().enumerate() {
                            let mut v = csub(ZERO, dense0.at(i, j));
                            if i == j {
                                v = cadd(v, complex(0., nu(&c2) - dual.diagnostics.slot_rotation[s]));
                            }
                            m.set(ii, jj, v);
                        }
                    }
                    let rhs: Vec<C> = eliminated.iter().map(|&k| y[k]).collect();
                    let solved = LuFactor::new(&m, &dual.lapack).unwrap().solve(&rhs, &dual.lapack).unwrap();
                    let mut x = vec![ZERO; d];
                    for (ii, &k) in eliminated.iter().enumerate() {
                        x[k] = solved[ii];
                    }
                    for term2 in &frame.terms {
                        let c3: Vec<i32> = c2.iter().zip(&term2.charge).map(|(a, b)| a + b).collect();
                        if let Some(s3) = dual.slots.iter().position(|x| *x == c3) {
                            let image = dense_apply(&dense_charge(p, &term2.charge), &x);
                            for k in 0..d {
                                if ground(k) {
                                    out.add_at(s3 * g2 + (k / n) * g + k % n, column, image[k]);
                                } else if slot2.is_none() {
                                    let w = second_hop.entry(s3).or_insert_with(|| vec![ZERO; d]);
                                    w[k] = cadd(w[k], image[k]);
                                }
                            }
                        }
                    }
                }
                // Two-hop path: fast charge -> slow charge, eliminated there.
                for (s3, w) in second_hop {
                    let c3 = dual.slots[s3].clone();
                    let eliminated: Vec<usize> = (0..d).filter(|&k| !ground(k)).collect();
                    let size = eliminated.len();
                    let mut m = Dense::zeros(size, size);
                    for (ii, &i) in eliminated.iter().enumerate() {
                        for (jj, &j) in eliminated.iter().enumerate() {
                            let mut v = csub(ZERO, dense0.at(i, j));
                            if i == j {
                                v = cadd(v, complex(0., nu(&c3) - dual.diagnostics.slot_rotation[s]));
                            }
                            m.set(ii, jj, v);
                        }
                    }
                    let rhs: Vec<C> = eliminated.iter().map(|&k| w[k]).collect();
                    let solved = LuFactor::new(&m, &dual.lapack).unwrap().solve(&rhs, &dual.lapack).unwrap();
                    let mut x2 = vec![ZERO; d];
                    for (ii, &k) in eliminated.iter().enumerate() {
                        x2[k] = solved[ii];
                    }
                    for term3 in &frame.terms {
                        let c4: Vec<i32> = c3.iter().zip(&term3.charge).map(|(a, b)| a + b).collect();
                        if let Some(s4) = dual.slots.iter().position(|x| *x == c4) {
                            let image = dense_apply(&dense_charge(p, &term3.charge), &x2);
                            for k in 0..d {
                                if ground(k) {
                                    out.add_at(s4 * g2 + (k / n) * g + k % n, column, image[k]);
                                }
                            }
                        }
                    }
                }
            }
        }
        out
    }

    /// Reduced generator (16x16, column-major) and initial coefficients of a
    /// twoNear 20 us candidate on which the static OpenBLAS zgeev did not
    /// converge (info = 16).
    const TWO_NEAR_GENERATOR: [f64; 256] = [
        0.013389733806765471, 6.2446865262876, -24.29865897709495, 29.3467480158981, -9.325925599681618, 7.908471050681866,
        -3.4271722666627746, 2.478999701632979, -1.6003858464375926, 1.0930809922930456, -1.294937411825693, 1.3926883683181914,
        -2.7657990542291597, 5.14161883389553, -7.1816567847628034, 13.61918171987518, -4.985521428958255, -18.72413201014758,
        150.67921830892342, -181.9830899159343, 57.83130640004982, -49.04148202005681, 21.252353205122475, -15.372609431336183,
        9.924211137543603, -6.7783483192168275, 8.030087542643942, -8.636254078170264, 17.151099009615734, -31.88387707088485,
        44.53442835078859, -84.45439316706253, 22.279956450569564, -145.18446007764726, -9.602925358025663, 35.75298505145798,
        -11.361724565330658, 9.634847293285794, -4.175304411105398, 3.0201506472287267, -1.9497386846105695, 1.3316913953259424,
        -1.5776129058422343, 1.6967031378653128, -3.369557823270193, 6.26400876763722, -8.749377368306726, 16.59218314599894,
        -28.59666251603138, 185.38947327452948, -20.002014157672143, -10.289963360766336, 9.625663847325482, -8.16265161365984,
        3.537321298848041, -2.558673652908122, 1.651818784452211, -1.12820797145012, 1.336553257331971, -1.437446941925895,
        2.8546913750367047, -5.306873881245496, 7.412480699428328, -14.056913591326412, 12.226219803121724, -58.38411005060801,
        10.642109782287251, -1.5182535605350431, -2.133320188358361, 18.769259739544662, -8.133744769753072, 5.883436529484436,
        -3.7982126767875806, 2.5942209246068373, -3.073289703027707, 3.305283402961018, -6.564103481600909, 12.202664638085537,
        -17.044310686757047, 32.32256452878271, -9.492314188878535, 51.22770649262087, -8.34145981884343, 2.046759213637414,
        -15.037934873252524, -2.119142893207906, 9.585432017324532, -6.933495280115308, 4.476106806953453, -3.0572299648661243,
        3.62180167585625, -3.8952009663250546, 7.735645594487002, -14.380560768788364, 20.086329807241743, -38.091402068104614,
        8.41292582623988, -22.160929356936087, 10.513678821641157, 1.2684924801411386, 3.920638716878097, -4.914277308529898,
        -4.7131492322934765, 17.875929572112728, -11.540295130991103, 7.88214941713726, -9.337726679940854, 10.042604106549438,
        -19.944033026062105, 37.07594320106341, -51.786549011165704, 98.20720221746765, -5.333754784886255, 16.23087414208622,
        -6.522658726226324, 0.009330555362361409, -3.016349696082772, 3.890686264049446, -11.882497631615847, -2.8980687938909426,
        14.782473947645418, -10.096592889111731, 11.9611063087051, -12.864015117780326, 25.547193530175466, -47.49221549115547,
        66.33568117187214, -125.7979492632412, 8.01449567702867, -10.292060792022987, 12.488670438851527, 1.7034470744712151,
        -1.798612500769612, 1.571466527661922, -1.0332303648810546, -4.866514602230568, -10.682903892724427, 20.956761963466548,
        -24.826796546905, 26.70089854470372, -53.02644806870271, 98.57613142969417, -137.68814037527855, 261.1096379129938,
        -3.567974449681649, 6.239945469541092, -5.2108923794106605, 0.7341660464045606, -0.5555916231488176, 0.5111161892922792,
        -2.195490312934039, 3.7855484273635303, -8.592236610187712, -6.529261860243103, 31.4283578505938, -33.80079307460134,
        67.12643328920933, -124.78799806349369, 174.30007861017876, -330.53994638396586, -2.025954560180662, -7.757198376371546,
        -2.8325127494257862, -3.5541962222468233, 5.037531138239357, -5.171278655704313, 12.230563816704436, -11.114621898527565,
        25.11818341903272, -23.326432194028044, -6.406847816129385, 28.400225396461977, -56.40122204947061, 104.84984320862462,
        -146.4510716183584, 277.7275259553257, -0.539412109137554, 7.978454421181472, -0.8903323052488238, 2.250539592119526,
        -3.7226917666442456, 4.444619194024183, -10.25075593398157, 11.672553682938574, -23.998912086634157, 27.671919015553552,
        -20.99879091883935, -3.369358917187592, 46.410213467433564, -86.27656763589236, 120.50848604412504, -228.53041254336537,
        -6.591344839251049, -16.65228243641197, -10.5584955055227, -7.788468786230066, 12.550689644681981, -14.09001310147175,
        33.191728868403416, -32.42676721375799, 68.61893463989259, -65.90629235787186, 40.833194714093054, -37.89996940446756,
        -16.654802940130285, 68.14127835088742, -95.17766809206005, 180.49344790315453, -3.143922508439295, 31.60354455566023,
        -3.5800288768934516, 5.587385434739409, -14.150680206299631, 15.844385261020069, -40.19530408683001, 50.888561175724114,
        -103.12215722683847, 125.94766828755796, -101.82304635312873, 82.28227390106821, -58.824604520115294, -5.02361814240608,
        34.95223527009092, -66.28288143004127, 4.873542466799331, -44.62632505220751, 5.403747484005436, -7.9553369510432175,
        19.02122039507459, -21.901908341508722, 55.10999986257832, -69.30846349763199, 142.76516545543325, -176.2453718748021,
        143.4177709511742, -116.86670900830777, 86.05793551027838, -25.450333820735693, -4.809607610487583, 47.048563777993294,
        -17.910424824964096, 84.68263191721552, -23.238604450514032, 11.067014559887037, -30.9531948143412, 35.29486621646599,
        -92.26584559192138, 123.58359690959018, -253.97911995839382, 331.8717425004693, -283.3722293995622, 227.5803611944636,
        -190.6563725419299, 57.19583956911605, -36.46280682381189, -16.549709749858767,
    ];
    const TWO_NEAR_INITIAL: [f64; 16] = [
        0.9999999999999996, -1.2054382705493797e-20, -3.318092855588688e-23, 4.894072255306808e-21, -1.0939592847984502e-21, -3.2754090435112446e-22,
        3.365293849292495e-21, -2.7388402446028236e-21, 3.1922192308128452e-21, -1.3633138491701512e-20, -4.601302017251321e-21, 1.6021454489386455e-22,
        -1.0514861452832665e-20, 8.728823413965825e-22, -4.650099211043414e-21, 2.6787694837904278e-21,
    ];

    /// Real test matrix with the structure of reduced generators built on
    /// gauged optical coherences: a slow block (rates 1e-3 .. 1/us) plus very
    /// fast, damped oscillating pairs (frequencies up to 4.5e7 rad/us), weakly
    /// coupled, mixed by an orthogonal change of basis.
    fn fast_slow_generator(n_slow: usize, frequencies: &[f64]) -> Vec<f64> {
        let n = n_slow + 2 * frequencies.len();
        let mut a = vec![0.; n * n];
        let mut state: u64 = 12345;
        let mut next = || {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        for i in 0..n_slow {
            a[i + i * n] = -(1e-3_f64).powf(1. - i as f64 / (n_slow - 1).max(1) as f64);
            for j in 0..n_slow {
                if i != j {
                    a[i + j * n] += 0.05 * next();
                }
            }
        }
        for (k, &w) in frequencies.iter().enumerate() {
            let (p, q) = (n_slow + 2 * k, n_slow + 2 * k + 1);
            a[p + p * n] = -18.;
            a[q + q * n] = -18.;
            a[p + q * n] = w;
            a[q + p * n] = -w;
            for i in 0..n_slow {
                a[p + i * n] += 1e-3 * next();
                a[i + q * n] += 1e-3 * next();
            }
        }
        // Mix: A <- H A H with one Householder reflector H.
        let v: Vec<f64> = (0..n).map(|_| next()).collect();
        let vv: f64 = v.iter().map(|x| x * x).sum();
        let h: Vec<f64> = (0..n * n).map(|k| (if k % (n + 1) == 0 { 1. } else { 0. }) - 2. * v[k % n] * v[k / n] / vv).collect();
        let product = |x: &[f64], y: &[f64]| -> Vec<f64> {
            let mut out = vec![0.; n * n];
            for j in 0..n {
                for k in 0..n {
                    let b = y[k + j * n];
                    for i in 0..n {
                        out[i + j * n] += x[i + k * n] * b;
                    }
                }
            }
            out
        };
        product(&product(&h, &a), &h)
    }

    #[test]
    fn slow_dual_robust_eigen_resolves_reduced_generators_on_every_path() {
        let lap = lap();
        let fast = fast_slow_generator(10, &[4.48e7, 4.4802e7, 4.4795e7, 2.1e3]);
        let cases: [(&str, Vec<f64>); 2] = [("twoNear", TWO_NEAR_GENERATOR.to_vec()), ("fast-slow", fast)];
        for (name, real) in &cases {
            let n = (real.len() as f64).sqrt() as usize;
            let a = Dense { rows: n, cols: n, data: real.iter().map(|&x| complex(x, 0.)).collect() };
            // Every attempt (forced by skipping the earlier ones) must resolve
            // the matrix on its own.
            for first in 0..5 {
                let eigen = robust_eigen_from(&a, Some(real), &lap, first).unwrap_or_else(|e| panic!("{name} from attempt {first}: {e}"));
                assert!(eigen.residual < 1e-9, "{name} from attempt {first} ({}): residual {:e}", eigen.method, eigen.residual);
            }
        }
    }

    #[test]
    fn slow_dual_reduced_source_reproduces_the_candidate_trajectory() {
        // a(t) = V exp(beta t) V^-1 a0 against a scaling-and-squaring Taylor
        // exponential of the failing twoNear generator.
        let lap = lap();
        let n = 16;
        let a = Dense { rows: n, cols: n, data: TWO_NEAR_GENERATOR.iter().map(|&x| complex(x, 0.)).collect() };
        let eigen = robust_eigen(&a, Some(&TWO_NEAR_GENERATOR), &lap).unwrap();
        let weights = LuFactor::new(&eigen.vectors, &lap)
            .unwrap()
            .solve(&TWO_NEAR_INITIAL.iter().map(|&x| complex(x, 0.)).collect::<Vec<_>>(), &lap)
            .unwrap();
        for &t in &[0.003, 0.05, 0.4] {
            // Reference: exp(A t) by 2^s squaring of a 30-term Taylor series.
            let norm = a.frobenius() * t;
            let s = norm.log2().ceil().max(0.) as i32 + 4;
            let h = t / 2f64.powi(s);
            let mut term = Dense::identity(n);
            let mut sum = Dense::identity(n);
            for k in 1..30 {
                term = term.multiply(&a, &lap).unwrap();
                for z in term.data.iter_mut() {
                    *z = cscale(*z, h / k as f64);
                }
                for (x, y) in sum.data.iter_mut().zip(&term.data) {
                    *x = cadd(*x, *y);
                }
            }
            for _ in 0..s {
                sum = sum.multiply(&sum, &lap).unwrap();
            }
            let x0 = Dense { rows: n, cols: 1, data: TWO_NEAR_INITIAL.iter().map(|&x| complex(x, 0.)).collect() };
            let reference = sum.multiply(&x0, &lap).unwrap();
            let modal: Vec<C> = (0..n).map(|j| cmul(weights[j], cexp(cscale(eigen.values[j], t)))).collect();
            let value = eigen.vectors.multiply(&Dense { rows: n, cols: 1, data: modal }, &lap).unwrap();
            let difference = value.data.iter().zip(&reference.data).map(|(x, y)| cabs(csub(*x, *y))).fold(0_f64, f64::max);
            assert!(difference < 1e-9, "t = {t}: trajectory mismatch {difference:e}");
        }
    }

    #[test]
    fn slow_dual_batched_products_match_single_ones() {
        let owner = super::super::tests::fixture();
        let dual = SlowDual::prepare(&owner, SlowDualOptions::default()).unwrap();
        let (n, lap) = (dual.n, &dual.lapack);
        let blocks: Vec<Dense> = (0..3)
            .map(|j| {
                let mut x = Dense::zeros(n, n);
                for k in 0..n * n {
                    x.data[k] = complex(((k * 7 + j * 3) % 11) as f64 - 5., ((k * 5 + j) % 7) as f64 - 3.);
                }
                x
            })
            .collect();
        let refs: Vec<&Dense> = blocks.iter().collect();
        for term in &dual.frame.terms {
            let batched = dual.frame.apply_all(term, &refs, lap).unwrap();
            for (x, y) in blocks.iter().zip(&batched) {
                let single = dual.frame.apply(term, x, lap).unwrap();
                let difference = single.data.iter().zip(&y.data).map(|(a, b)| cabs(csub(*a, *b))).fold(0_f64, f64::max);
                assert!(difference < 1e-12 * single.frobenius().max(1.), "apply_all mismatch {difference:e}");
            }
        }
        let dressed = dual.dressed.as_ref().expect("fixture has a dressed frame");
        let batched = sandwich_all(&dressed.left_inverse, &dressed.right, &refs, lap).unwrap();
        for (x, y) in blocks.iter().zip(&batched) {
            let single = dressed.to_eigen(x, lap).unwrap();
            let difference = single.data.iter().zip(&y.data).map(|(a, b)| cabs(csub(*a, *b))).fold(0_f64, f64::max);
            assert!(difference < 1e-12 * single.frobenius().max(1.), "sandwich_all mismatch {difference:e}");
        }
    }

    #[test]
    fn slow_dual_generator_conserves_the_central_trace() {
        // With one slow charge the eliminated-space approximation keeps the
        // left trace vector exact: every column of J_eff has zero population
        // trace in the central block (no leak at the drive-charge pumping rate).
        let owner = super::super::tests::fixture();
        // A tiny cutoff keeps only the central charge slow.
        let dual = SlowDual::prepare(&owner, SlowDualOptions { slow_frequency_cutoff: Some(1e-6) }).unwrap();
        assert_eq!(dual.slots.len(), 1, "fixture must have a single slow charge");
        assert!(dual.frame.terms.len() > 1, "fixture must have drive charges");
        let g = dual.g;
        let scale = dual.generator.frobenius();
        for column in 0..dual.dimension() {
            let mut trace = ZERO;
            for a in 0..g {
                trace = cadd(trace, dual.generator.at(a * g + a, column));
            }
            assert!(cabs(trace) < 1e-12 * scale, "column {column} leaks trace {:e} (scale {scale:e})", cabs(trace));
        }
        assert!(dual.diagnostics.generator_spectral_abscissa.abs() < 1e-9 * scale, "abscissa {:e}", dual.diagnostics.generator_spectral_abscissa);
    }

    #[test]
    fn slow_dual_generator_matches_dense_one_hop_schur_complement() {
        let owner = super::super::tests::fixture();
        for cutoff in [None, Some(1e4)] {
            // A huge cutoff forces extra slow charges (multi-slot path).
            let dual = SlowDual::prepare(&owner, SlowDualOptions { slow_frequency_cutoff: cutoff }).unwrap();
            if cutoff.is_some() {
                assert!(dual.slots.len() > 1, "large cutoff must create Raman slots");
            }
            let expected = dense_generator(&owner, &dual);
            let scale = expected.frobenius();
            let mut difference = 0_f64;
            for k in 0..expected.data.len() {
                difference = difference.hypot(cabs(csub(dual.generator.data[k], expected.data[k])));
            }
            assert!(difference < 1e-10 * scale.max(1.), "generator mismatch {difference:e} (scale {scale:e})");
        }
    }

    #[test]
    fn slow_dual_divided_differences_match_definitions() {
        let t = 2.5;
        // Separated nodes: explicit formulas.
        let (x, y, z) = (complex(-0.3, 1.7), complex(-1.1, -0.4), complex(0.2, 0.9));
        let d2 = exp_divided_2(x, y, t);
        let explicit = cdiv(csub(cexp(cscale(x, t)), cexp(cscale(y, t))), csub(x, y));
        assert!(cabs(csub(d2, explicit)) < 1e-13);
        let d3 = exp_divided_3(x, y, z, t);
        let explicit3 = cdiv(csub(exp_divided_2(x, y, t), exp_divided_2(y, z, t)), csub(x, z));
        assert!(cabs(csub(d3, explicit3)) < 1e-12);
        // Clustered nodes: limits t exp(x t) and t^2/2 exp(x t).
        let w = complex(-0.7, 0.3);
        let limit2 = cscale(cexp(cscale(w, t)), t);
        assert!(cabs(csub(exp_divided_2(w, w, t), limit2)) < 1e-13);
        let limit3 = cscale(cexp(cscale(w, t)), t * t / 2.);
        assert!(cabs(csub(exp_divided_3(w, w, w, t), limit3)) < 1e-12);
        // Nearly clustered agrees with a finite-difference of the 2-node form.
        let h = complex(1e-6, -2e-7);
        let fd = cdiv(csub(exp_divided_2(cadd(w, h), w, t), exp_divided_2(w, w, t)), h);
        assert!(cabs(csub(exp_divided_3(cadd(w, h), w, w, t), fd)) < 1e-5);
    }

    #[test]
    fn slow_dual_triangular_sylvester_solves_general_left_right() {
        let lap = lap();
        let (p, r) = (5, 7);
        let a_left = random_dense(p, p, 1);
        let a_right = random_dense(r, r, 2);
        let y = random_dense(p, r, 3);
        let factor = SylvesterFactor { left: Schur::new(&a_left, &lap).unwrap(), right: Schur::new(&a_right, &lap).unwrap() };
        let z = complex(0.3, 2.1);
        let x = factor.solve(z, &y, &lap).unwrap();
        let back = NoJumpBlocks::forward(z, &a_left, &a_right, &x, &lap).unwrap();
        for k in 0..p * r {
            assert!(cabs(csub(back.data[k], y.data[k])) < 1e-11, "{k}");
        }
    }

    #[test]
    fn slow_dual_series_matches_quadrature_for_separated_and_resonant_sources() {
        // A 1-slot fake dual with a diagonal generator is enough to test the
        // modal integration formulas: compare against fine trapezoid quadrature.
        let alpha = vec![complex(0., 0.), complex(-2.0, 3.0), complex(-0.5, -1.0)];
        let dim = alpha.len();
        let lap = Arc::new(lap());
        let identity = Dense::identity(dim);
        let dual = SlowDual {
            owner: super::super::tests::fixture().primitives,
            lapack: Arc::clone(&lap),
            frame: FrameLiouvillian { n: 0, g: 0, e: 0, terms: vec![], axes: 0 },
            n: dim,
            g: 0,
            slots: vec![vec![]],
            slot_frequency: vec![0.],
            generator: Dense::zeros(dim, dim),
            eigenvalues: alpha.clone(),
            inverse_vectors: identity.clone(),
            readout: vec![(0., identity.clone())],
            diagonal_reference: vec![],
            population_readout: vec![],
            dressed: None,
            slaved_central: vec![],
            elimination: None,
            diagnostics: SlowDualDiagnostics {
                slow_charges: vec![],
                slow_dimension: dim,
                operator_charges: 0,
                weak_operator_charges: 0,
                unrepresented_slow_charges: 0,
                unrepresented_rate: 0.,
                eliminated_charge_blocks: 0,
                slot_rotation: vec![0.],
                optical_decay_floor: 1.,
                slow_frequency_cutoff: 1.,
                minimum_eliminated_denominator: 1.,
                maximum_woodbury_condition: 1.,
                maximum_optical_schur_condition: 1.,
                generator_eigenvector_condition: 1.,
                generator_reconstruction_relative: 0.,
                generator_eigen_method: "test",
                generator_spectral_abscissa: 0.,
                dressed_condition: None,
                preparation_ms: 0.,
            },
            ground_energy: vec![],
            unrepresented_readout: vec![],
        };
        // Families with beta = {0, -0.1}: resonant with alpha[0] (secular
        // growth), separated, a fast-phase family, a clustered and a separable
        // pair, an initial value, direct populations and a direct near term.
        let beta = [complex(0., 0.), complex(-0.1, 0.)];
        let mut slow0 = Dense::zeros(dim, 2);
        for (k, v) in [complex(0.3, 0.), complex(0.1, -0.2), ZERO].iter().enumerate() {
            slow0.set(k, 0, *v);
        }
        for (k, v) in [complex(1., 0.5), complex(0.2, 0.), complex(-0.4, 0.1)].iter().enumerate() {
            slow0.set(k, 1, *v);
        }
        let mut slow40 = Dense::zeros(dim, 2);
        for (k, v) in [complex(0.05, 0.), complex(-0.3, 0.2), complex(0.1, 0.1)].iter().enumerate() {
            slow40.set(k, 1, *v);
        }
        let mut direct = Dense::zeros(dim, 2);
        direct.set(0, 0, complex(0.02, 0.01));
        direct.set(2, 1, complex(-0.03, 0.));
        let clustered = (complex(-0.5, -1.0), complex(-0.5 + 1e-4, -1.0), vec![ZERO, complex(0.3, 0.1), complex(0.7, 0.)]);
        let separable = (complex(-1.0, 2.0), complex(0.3, -0.5), vec![complex(0.1, 0.), ZERO, complex(0.05, -0.02)]);
        let initial = [complex(0.2, 0.), complex(0., 0.1), complex(-0.1, 0.)];
        let near = (vec![complex(0.01, 0.), complex(0., 0.02), ZERO], complex(-0.2, 1.), complex(-0.2001, 1.));
        // Separated direct pairs sharing their second node (split and merged).
        let split = [
            (vec![complex(0.02, 0.), complex(0., -0.01), complex(0.03, 0.01)], complex(-0.3, 2.), complex(-1., -1.5)),
            (vec![complex(-0.01, 0.02), ZERO, complex(0.01, 0.)], complex(0.05, -0.7), complex(-1., -1.5)),
        ];
        let uniform: Vec<f64> = (0..9).map(|i| 0.5 * i as f64).collect();
        for times in [vec![0., 0.5, 1.7, 4.0], uniform] {
            let grid = TimeGrid::new(&times);
            let mut series = Series::new(&beta, grid.maximum);
            series.add_slow(0., 0, &slow0);
            series.add_slow(40., 0, &slow40);
            series.add_pair(clustered.0, clustered.1, clustered.2.clone());
            series.add_pair(separable.0, separable.1, separable.2.clone());
            series.add_initial(&initial);
            series.add_direct(2., 0, &direct);
            series.direct_near.push(near.clone());
            for (p, first, second) in &split {
                series.add_direct_pair(*first, *second, p.clone());
            }
            assert_eq!(series.direct_points.len(), 3, "the shared node merges");
            let values = dual.evaluate(&series, &grid).unwrap();
            // Reference: integrate x' = alpha x + s(t) with RK4 on a fine grid.
            let source = |s: f64| -> Vec<C> {
                let mut out = vec![ZERO; dim];
                for k in 0..dim {
                    for j in 0..2 {
                        out[k] = cadd(out[k], cmul(slow0.at(k, j), cexp(cscale(beta[j], s))));
                        out[k] = cadd(out[k], cmul(slow40.at(k, j), cexp(cscale(cadd(beta[j], complex(0., 40.)), s))));
                    }
                    out[k] = cadd(out[k], cmul(clustered.2[k], exp_divided_2(clustered.0, clustered.1, s)));
                    out[k] = cadd(out[k], cmul(separable.2[k], exp_divided_2(separable.0, separable.1, s)));
                }
                out
            };
            for (it, &t_end) in times.iter().enumerate() {
                let steps = 40000;
                let h = t_end / steps as f64;
                let mut x: Vec<C> = initial.to_vec();
                let derivative = |s: f64, x: &[C]| -> Vec<C> {
                    let src = source(s);
                    (0..dim).map(|k| cadd(cmul(alpha[k], x[k]), src[k])).collect()
                };
                for step in 0..steps {
                    let s = step as f64 * h;
                    let k1 = derivative(s, &x);
                    let x2: Vec<C> = (0..dim).map(|k| cadd(x[k], cscale(k1[k], h / 2.))).collect();
                    let k2 = derivative(s + h / 2., &x2);
                    let x3: Vec<C> = (0..dim).map(|k| cadd(x[k], cscale(k2[k], h / 2.))).collect();
                    let k3 = derivative(s + h / 2., &x3);
                    let x4: Vec<C> = (0..dim).map(|k| cadd(x[k], cscale(k3[k], h))).collect();
                    let k4 = derivative(s + h, &x4);
                    for k in 0..dim {
                        let inc = cadd(cadd(k1[k], cscale(k2[k], 2.)), cadd(cscale(k3[k], 2.), k4[k]));
                        x[k] = cadd(x[k], cscale(inc, h / 6.));
                    }
                }
                for k in 0..dim {
                    let mut expected = x[k];
                    for j in 0..2 {
                        expected = cadd(expected, cmul(direct.at(k, j), cexp(cscale(cadd(beta[j], complex(0., 2.)), t_end))));
                    }
                    expected = cadd(expected, cmul(near.0[k], exp_divided_2(near.1, near.2, t_end)));
                    for (p, first, second) in &split {
                        expected = cadd(expected, cmul(p[k], exp_divided_2(*first, *second, t_end)));
                    }
                    let error = cabs(csub(values[it * dim + k], expected));
                    assert!(error < 1e-8, "time {t_end} mode {k}: {error:e}");
                }
            }
        }
    }
}