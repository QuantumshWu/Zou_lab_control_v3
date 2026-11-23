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

/// A complex number: its real part and then its imaginary part (`repr(C)`: a slice of them is the reals the
/// interleaved kernels below take two at a time).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
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
    /// |z|, without overflow or underflow of the square (`hypot`).
    pub fn abs(self) -> f64 {
        hypot(self.re, self.im)
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

// The kernels below (`combine`, the products) are written once, for 256-bit vectors of reals with fused
// multiply-add, and the whole build takes those instructions (apps/atom-workbench/.cargo/config.toml; a build that sets
// RUSTFLAGS itself names them there): no second, scalar form and no choice at run time. Without the features a fused
// multiply-add is a library call per number -- so such a build is refused here instead of running many times slower.
#[cfg(not(all(target_feature = "avx2", target_feature = "fma")))]
compile_error!("atom-motion's kernels are written for AVX2 with FMA: build with -C target-feature=+avx2,+fma (apps/atom-workbench/.cargo/config.toml)");

/// The reals of one vector of the split kernel's sums (`combine`): their vectors' lengths are multiples of it.
pub const LANES: usize = 4;

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

// The kernels on complex numbers as they lie, real and imaginary part side by side (`C64` slices): two numbers a vector
// of four reals, the products' four real products and their two sums made as the scalar forms make them -- each
// number's result is the scalar form's bit for bit (each operation rounds the same two numbers), so a caller's sums
// are the same whichever it calls. A row's last odd number takes a vector of two reals.
use std::arch::x86_64::{
    __m128d, __m256d, _mm256_add_pd, _mm256_addsub_pd, _mm256_castpd256_pd128, _mm256_fmadd_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_permute_pd, _mm256_set1_pd, _mm256_setr_pd,
    _mm256_setzero_pd, _mm256_storeu_pd, _mm256_sub_pd, _mm_add_pd, _mm_addsub_pd, _mm_fmadd_pd, _mm_loadu_pd, _mm_mul_pd, _mm_permute_pd, _mm_setzero_pd, _mm_storeu_pd, _mm_sub_pd,
};

/// f y of two numbers in a vector, as `C64`'s product makes it: (f.re y.re − f.im y.im, f.re y.im + f.im y.re).
#[inline(always)]
unsafe fn times(re: __m256d, im: __m256d, y: __m256d) -> __m256d {
    _mm256_addsub_pd(_mm256_mul_pd(re, y), _mm256_mul_pd(im, _mm256_permute_pd::<0b0101>(y)))
}
/// The same of one number.
#[inline(always)]
unsafe fn times1(re: __m128d, im: __m128d, y: __m128d) -> __m128d {
    _mm_addsub_pd(_mm_mul_pd(re, y), _mm_mul_pd(im, _mm_permute_pd::<0b01>(y)))
}

/// z_j ← z_j + f y_j (`subtract`: z_j − f y_j), each as `C64`'s operators make it: the product rounded, then the sum.
#[inline]
pub fn mul_add(subtract: bool, z: &mut [C64], f: C64, y: &[C64]) {
    let n = z.len();
    assert!(y.len() >= n);
    // SAFETY: n numbers of each (above); C64 is two f64 (`repr(C)`); the build has AVX2 and FMA (above).
    unsafe {
        let (zp, yp) = (z.as_mut_ptr() as *mut f64, y.as_ptr() as *const f64);
        let (re, im) = (_mm256_set1_pd(f.re), _mm256_set1_pd(f.im));
        let mut j = 0;
        while j + 2 <= n {
            let (at, by) = (zp.add(2 * j), times(re, im, _mm256_loadu_pd(yp.add(2 * j))));
            let x = _mm256_loadu_pd(at);
            _mm256_storeu_pd(at, if subtract { _mm256_sub_pd(x, by) } else { _mm256_add_pd(x, by) });
            j += 2;
        }
        if j < n {
            let (at, by) = (zp.add(2 * j), times1(_mm256_castpd256_pd128(re), _mm256_castpd256_pd128(im), _mm_loadu_pd(yp.add(2 * j))));
            let x = _mm_loadu_pd(at);
            _mm_storeu_pd(at, if subtract { _mm_sub_pd(x, by) } else { _mm_add_pd(x, by) });
        }
    }
}

/// out_j = 1/((c_j − w) − κ) for every j, each as the scalar form makes it -- the two differences, |z|² = re² + im², one
/// division, the conjugate scaled -- but four numbers' squared moduli to a vector and one vector division for the four.
pub fn reciprocals(c: &[C64], w: f64, kappa: C64, out: &mut [C64]) {
    let n = c.len();
    assert!(out.len() >= n);
    // SAFETY: n numbers of each (above); C64 is two f64 (`repr(C)`); the build has AVX2 (above).
    unsafe {
        use std::arch::x86_64::{_mm256_div_pd, _mm256_hadd_pd, _mm256_unpackhi_pd, _mm256_unpacklo_pd};
        let (cp, op) = (c.as_ptr() as *const f64, out.as_mut_ptr() as *mut f64);
        let (wv, kv) = (_mm256_setr_pd(w, 0.0, w, 0.0), _mm256_setr_pd(kappa.re, kappa.im, kappa.re, kappa.im));
        let (sign, one) = (_mm256_setr_pd(1.0, -1.0, 1.0, -1.0), _mm256_set1_pd(1.0));
        let mut j = 0;
        while j + 4 <= n {
            let a = _mm256_sub_pd(_mm256_sub_pd(_mm256_loadu_pd(cp.add(2 * j)), wv), kv);
            let b = _mm256_sub_pd(_mm256_sub_pd(_mm256_loadu_pd(cp.add(2 * j + 4)), wv), kv);
            // [|z0|², |z2|², |z1|², |z3|²], their reciprocals, laid back by pairs.
            let r = _mm256_div_pd(one, _mm256_hadd_pd(_mm256_mul_pd(a, a), _mm256_mul_pd(b, b)));
            _mm256_storeu_pd(op.add(2 * j), _mm256_mul_pd(_mm256_mul_pd(a, sign), _mm256_unpacklo_pd(r, r)));
            _mm256_storeu_pd(op.add(2 * j + 4), _mm256_mul_pd(_mm256_mul_pd(b, sign), _mm256_unpackhi_pd(r, r)));
            j += 4;
        }
        while j < n {
            let z = C64 { re: c[j].re - w, im: c[j].im - 0.0 } - kappa;
            let d = 1.0 / z.norm_sqr();
            out[j] = C64 { re: z.re * d, im: -z.im * d };
            j += 1;
        }
    }
}

/// x_j ← x_j e^{w z_j} for every j, four at a time (`exp4`): each argument w z_j as `C64`'s product makes it and the
/// product with x_j as `C64`'s -- a state's turns at a time (spectral.rs `Picture::at`). The numbers go to the vectors
/// and back by shuffles, not through memory a lane at a time (a vector read of four numbers just written one by one waits
/// for the writes, some twenty cycles a vector: the kernel's own work); the numbers past the whole vectors through a
/// vector padded with zeros.
pub fn times_exp(x: &mut [C64], w: C64, z: &[C64]) {
    use std::arch::x86_64::{_mm256_movedup_pd, _mm256_permute2f128_pd, _mm256_permute4x64_pd, _mm256_unpackhi_pd, _mm256_unpacklo_pd};
    let n = x.len();
    assert!(z.len() >= n);
    // SAFETY: whole vectors of four numbers within x and z (`whole`), the rest through padded copies; C64 is two f64
    // (`repr(C)`); the build has AVX2 and FMA (above).
    unsafe {
        let (wr, wi) = (_mm256_set1_pd(w.re), _mm256_set1_pd(w.im));
        let four = |x: *mut f64, z: *const f64| {
            let (v0, v1) = (_mm256_loadu_pd(z), _mm256_loadu_pd(z.add(4)));
            // [z0 z1 z2 z3]'s real parts and imaginary parts.
            let zr = _mm256_permute4x64_pd::<0xD8>(_mm256_unpacklo_pd(v0, v1));
            let zi = _mm256_permute4x64_pd::<0xD8>(_mm256_unpackhi_pd(v0, v1));
            let a = _mm256_sub_pd(_mm256_mul_pd(wr, zr), _mm256_mul_pd(wi, zi));
            let b = _mm256_add_pd(_mm256_mul_pd(wr, zi), _mm256_mul_pd(wi, zr));
            let (mut er, mut ei, inside) = exp4(a, b);
            if inside != 0b1111 {
                let (mut ra, mut rb, mut re, mut im) = ([0.0; LANES], [0.0; LANES], [0.0; LANES], [0.0; LANES]);
                for (v, to) in [(a, &mut ra), (b, &mut rb), (er, &mut re), (ei, &mut im)] {
                    _mm256_storeu_pd(to.as_mut_ptr(), v);
                }
                for l in (0..LANES).filter(|&l| inside & (1 << l) == 0) {
                    let e = C64::new(ra[l], rb[l]).exp();
                    (re[l], im[l]) = (e.re, e.im);
                }
                (er, ei) = (_mm256_loadu_pd(re.as_ptr()), _mm256_loadu_pd(im.as_ptr()));
            }
            // Back to numbers side by side, [e0 e1] and [e2 e3], and x e as `C64`'s product.
            let (low, high) = (_mm256_unpacklo_pd(er, ei), _mm256_unpackhi_pd(er, ei));
            for (at, e) in [(0, _mm256_permute2f128_pd::<0x20>(low, high)), (4, _mm256_permute2f128_pd::<0x31>(low, high))] {
                let y = _mm256_loadu_pd(x.add(at));
                let product = _mm256_addsub_pd(_mm256_mul_pd(y, _mm256_movedup_pd(e)), _mm256_mul_pd(_mm256_permute_pd::<0b0101>(y), _mm256_permute_pd::<0b1111>(e)));
                _mm256_storeu_pd(x.add(at), product);
            }
        };
        let whole = n / LANES * LANES;
        let (xp, zp) = (x.as_mut_ptr() as *mut f64, z.as_ptr() as *const f64);
        for j in (0..whole).step_by(LANES) {
            four(xp.add(2 * j), zp.add(2 * j));
        }
        if whole < n {
            let (mut xs, mut zs) = ([C64::ZERO; LANES], [C64::ZERO; LANES]);
            xs[..n - whole].copy_from_slice(&x[whole..]);
            zs[..n - whole].copy_from_slice(&z[whole..n]);
            four(xs.as_mut_ptr() as *mut f64, zs.as_ptr() as *const f64);
            x[whole..].copy_from_slice(&xs[..n - whole]);
        }
    }
}

/// e^{a_l} (cos b_l + i sin b_l) of four lanes (`exp4`), a lane outside its range the library's own: the kernel lane by
/// lane, for the tests.
#[cfg(test)]
pub(crate) fn exps(a: [f64; LANES], b: [f64; LANES]) -> [C64; LANES] {
    let mut out = [C64::ZERO; LANES];
    // SAFETY: four lanes in and out; the build has AVX2 and FMA (above).
    let inside = unsafe {
        let (re, im, inside) = exp4(_mm256_loadu_pd(a.as_ptr()), _mm256_loadu_pd(b.as_ptr()));
        let (mut res, mut ims) = ([0.0; LANES], [0.0; LANES]);
        _mm256_storeu_pd(res.as_mut_ptr(), re);
        _mm256_storeu_pd(ims.as_mut_ptr(), im);
        for l in 0..LANES {
            out[l] = C64::new(res[l], ims[l]);
        }
        inside
    };
    for l in (0..LANES).filter(|&l| inside & (1 << l) == 0) {
        out[l] = C64::new(a[l], b[l]).exp();
    }
    out
}

/// e^{a_l} (cos b_l + i sin b_l) of four lanes, a vector of the real parts and one of the imaginary parts, each within a
/// few units of its round-off of the modulus e^{a_l} (tests, `four_exponentials_are_the_library_s_within_their_round_off`),
/// and the lanes in range (a mask, lane l its bit l):
/// - e^a: a = k ln 2 + r, k = round(a/ln 2), r by Cody & Waite's two parts of ln 2 (fdlibm's: the first's trailing bits
///   zero, each product rounded once in its fused step), |r| ≤ ln 2/2; e^r = 1 + r + r²/2 + … to r¹³/13! (the next term
///   r¹⁴/14! ≤ 4·10⁻¹⁸ of one), and 2^k from k's bits;
/// - cos b, sin b: b = k π/2 + r, k = round(2b/π), r by three parts of π/2 of 33 bits each (fdlibm's `rem_pio2`): every
///   k π/2's part exact for |k| < 2^20, the first difference exact by Sterbenz's lemma, so r is within an ulp of its own
///   and k 10⁻³¹ besides; on |r| ≤ π/4 fdlibm's kernels (`__kernel_sin`, `__kernel_cos`: their minimax polynomials, each
///   within an ulp), the quadrant by k's last two bits.
/// A lane outside the range these hold on -- a outside [−708, 709] (e^a beyond the normal numbers' range or its 2^k),
/// |b| beyond 2^19 π/2, a number that is not finite (a comparison with NaN is false) -- is out of range: its caller takes
/// the library's (`C64::exp`), NaN in, NaN out. Inlined where it is called: each lane is a chain of some forty dependent
/// fused steps, and a caller's successive vectors overlap.
#[inline(always)]
unsafe fn exp4(av: __m256d, bv: __m256d) -> (__m256d, __m256d, i32) {
    use std::arch::x86_64::{_mm256_add_epi64, _mm256_and_pd, _mm256_and_si256, _mm256_andnot_pd, _mm256_blendv_pd, _mm256_castpd_si256, _mm256_castsi256_pd, _mm256_cmp_pd, _mm256_cmpeq_epi64, _mm256_movemask_pd, _mm256_round_pd, _mm256_set1_epi64x, _mm256_slli_epi64, _mm256_xor_pd, _CMP_GE_OQ, _CMP_LE_OQ, _MM_FROUND_NO_EXC, _MM_FROUND_TO_NEAREST_INT};
    // fdlibm's constants (e_exp.c, e_rem_pio2.c, k_sin.c, k_cos.c).
    const LN2_HI: f64 = 6.931_471_803_691_238_164_90e-1;
    const LN2_LO: f64 = 1.908_214_929_270_587_700_02e-10;
    const PIO2: [f64; 3] = [1.570_796_326_734_125_614_17, 6.077_100_506_303_965_976_60e-11, 2.022_266_248_711_166_455_80e-21];
    const SIN: [f64; 6] = [-1.666_666_666_666_663_243_48e-1, 8.333_333_333_322_489_461_24e-3, -1.984_126_982_985_794_931_34e-4, 2.755_731_370_707_006_767_89e-6, -2.505_076_025_340_686_341_95e-8, 1.589_690_995_211_550_102_21e-10];
    const COS: [f64; 6] = [4.166_666_666_666_660_190_37e-2, -1.388_888_888_887_410_957_49e-3, 2.480_158_728_947_672_941_78e-5, -2.755_731_435_139_066_330_35e-7, 2.087_572_321_298_174_827_90e-9, -1.135_964_755_778_819_482_65e-11];
    // 2^52 + 2^51: a whole number of magnitude below 2^51 added to it lies in the sum's last bits.
    const MAGIC: f64 = 6_755_399_441_055_744.0;
    const NEAREST: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
    // SAFETY: the caller's (the build has AVX2 and FMA).
    unsafe {
        let fma = |x: __m256d, y: __m256d, z: __m256d| _mm256_fmadd_pd(x, y, z);
        let set = _mm256_set1_pd;
        // e^a.
        let k = _mm256_round_pd::<NEAREST>(_mm256_mul_pd(av, set(std::f64::consts::LOG2_E)));
        let minus = _mm256_sub_pd(_mm256_setzero_pd(), k);
        let r = fma(minus, set(LN2_LO), fma(minus, set(LN2_HI), av));
        let mut q = set(1.0 / 6_227_020_800.0);
        for f in [479_001_600.0, 39_916_800.0, 3_628_800.0, 362_880.0, 40_320.0, 5_040.0, 720.0, 120.0, 24.0, 6.0, 2.0] {
            q = fma(q, r, set(1.0 / f));
        }
        let p = _mm256_add_pd(set(1.0), fma(_mm256_mul_pd(r, r), q, r));
        let power = _mm256_slli_epi64::<52>(_mm256_add_epi64(_mm256_castpd_si256(_mm256_add_pd(k, set(MAGIC))), _mm256_set1_epi64x(1023 - 0x4338_0000_0000_0000)));
        let modulus = _mm256_mul_pd(p, _mm256_castsi256_pd(power));
        // cos b, sin b.
        let k = _mm256_round_pd::<NEAREST>(_mm256_mul_pd(bv, set(std::f64::consts::FRAC_2_PI)));
        let minus = _mm256_sub_pd(_mm256_setzero_pd(), k);
        let r = fma(minus, set(PIO2[2]), fma(minus, set(PIO2[1]), fma(minus, set(PIO2[0]), bv)));
        let z = _mm256_mul_pd(r, r);
        let mut s = set(SIN[5]);
        for c in SIN[1..5].iter().rev() {
            s = fma(s, z, set(*c));
        }
        let sine = fma(_mm256_mul_pd(z, r), fma(z, s, set(SIN[0])), r);
        let mut c = set(COS[5]);
        for x in COS[..5].iter().rev() {
            c = fma(c, z, set(*x));
        }
        let (hz, zc) = (_mm256_mul_pd(set(0.5), z), _mm256_mul_pd(_mm256_mul_pd(z, z), c));
        let w = _mm256_sub_pd(set(1.0), hz);
        let cosine = _mm256_add_pd(w, _mm256_add_pd(_mm256_sub_pd(_mm256_sub_pd(set(1.0), w), hz), zc));
        // The quadrant k mod 4 from k's last bits: odd swaps the two, k ≡ 1, 2 turns the cosine's sign, k ≡ 2, 3 the sine's.
        let bits = _mm256_castpd_si256(_mm256_add_pd(k, set(MAGIC)));
        let (one, two) = (_mm256_set1_epi64x(1), _mm256_set1_epi64x(2));
        let swap = _mm256_castsi256_pd(_mm256_cmpeq_epi64(_mm256_and_si256(bits, one), one));
        let (cos, sin) = (_mm256_blendv_pd(cosine, sine, swap), _mm256_blendv_pd(sine, cosine, swap));
        let cos = _mm256_xor_pd(cos, _mm256_castsi256_pd(_mm256_slli_epi64::<62>(_mm256_and_si256(_mm256_add_epi64(bits, one), two))));
        let sin = _mm256_xor_pd(sin, _mm256_castsi256_pd(_mm256_slli_epi64::<62>(_mm256_and_si256(bits, two))));
        let inside = _mm256_and_pd(_mm256_and_pd(_mm256_cmp_pd::<_CMP_GE_OQ>(av, set(-708.0)), _mm256_cmp_pd::<_CMP_LE_OQ>(av, set(709.0))), _mm256_cmp_pd::<_CMP_LE_OQ>(_mm256_andnot_pd(set(-0.0), bv), set(823_549.6)));
        (_mm256_mul_pd(cos, modulus), _mm256_mul_pd(sin, modulus), _mm256_movemask_pd(inside))
    }
}

/// out (r × c) = a (r × k) b (k × c), all row by row: each entry from zero by k fused steps, b's rows in order,
/// o.re ← x.re y.re + (−x.im y.im + o.re) and o.im ← x.re y.im + (x.im y.re + o.im) (x = a's entry, y = b's) -- two
/// fused multiply-adds a real. Up to eight numbers of a row of out are held in registers through the k steps.
pub fn fused_product(a: &[C64], b: &[C64], dims: (usize, usize, usize), out: &mut [C64]) {
    product::<false, false>(a, b, dims, out);
}

/// out (r × c) += a (r × k) b (k × c): `fused_product` with each entry's chain from out's own value.
pub fn fused_product_onto(a: &[C64], b: &[C64], dims: (usize, usize, usize), out: &mut [C64]) {
    product::<false, true>(a, b, dims, out);
}

/// out (r × c) = a† b, a (k × r): `fused_product` on a's columns, each entry conjugated (no adjoint made).
pub fn adjoint_product(a: &[C64], b: &[C64], dims: (usize, usize, usize), out: &mut [C64]) {
    product::<true, false>(a, b, dims, out);
}

/// The products' one kernel: out (r × c) = op(a) b, op(a) = a (r × k) or with `ADJOINT` a† of a (k × r); with `ONTO` each
/// entry's chain starts at out's value instead of zero.
#[inline(always)]
fn product<const ADJOINT: bool, const ONTO: bool>(a: &[C64], b: &[C64], (r, k, c): (usize, usize, usize), out: &mut [C64]) {
    assert!(a.len() >= r * k && b.len() >= k * c && out.len() >= r * c);
    // SAFETY: the lengths above; C64 is two f64 (`repr(C)`); the build has AVX2 and FMA (above).
    unsafe {
        let (ap, bp, op) = (a.as_ptr(), b.as_ptr() as *const f64, out.as_mut_ptr() as *mut f64);
        for i in 0..r {
            let mut j = 0;
            while j < c {
                let w = (c - j).min(8);
                let pairs = w / 2;
                let to = op.add(2 * (i * c + j));
                let (mut held, mut tail) = ([_mm256_setzero_pd(); 4], _mm_setzero_pd());
                if ONTO {
                    for (v, sum) in held.iter_mut().enumerate() {
                        if v < pairs {
                            *sum = _mm256_loadu_pd(to.add(4 * v));
                        }
                    }
                    if w % 2 == 1 {
                        tail = _mm_loadu_pd(to.add(4 * pairs));
                    }
                }
                for l in 0..k {
                    let x = if ADJOINT { (*ap.add(l * r + i)).conj() } else { *ap.add(i * k + l) };
                    let at = bp.add(2 * (l * c + j));
                    let (re, im) = (_mm256_set1_pd(x.re), _mm256_setr_pd(-x.im, x.im, -x.im, x.im));
                    for (v, sum) in held.iter_mut().enumerate() {
                        if v < pairs {
                            let y = _mm256_loadu_pd(at.add(4 * v));
                            *sum = _mm256_fmadd_pd(re, y, _mm256_fmadd_pd(im, _mm256_permute_pd::<0b0101>(y), *sum));
                        }
                    }
                    if w % 2 == 1 {
                        let y = _mm_loadu_pd(at.add(4 * pairs));
                        tail = _mm_fmadd_pd(_mm256_castpd256_pd128(re), y, _mm_fmadd_pd(_mm256_castpd256_pd128(im), _mm_permute_pd::<0b01>(y), tail));
                    }
                }
                for (v, sum) in held.iter().enumerate() {
                    if v < pairs {
                        _mm256_storeu_pd(to.add(4 * v), *sum);
                    }
                }
                if w % 2 == 1 {
                    _mm_storeu_pd(to.add(4 * pairs), tail);
                }
                j += w;
            }
        }
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

/// The larger of a and b, and NaN where either is NaN. `f64::max` returns the other number where one is NaN, so that a
/// maximum folded with it reads a NaN as nothing -- a block of NaN as the fold's start, 0 where it starts at 0 -- and
/// the NaN never reaches what would refuse it; this one carries it to whatever reads the maximum. ±∞ are numbers like
/// any other. Where neither is NaN it is `f64::max` itself, to the bit.
pub fn larger(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

/// The smaller of a and b, and NaN where either is NaN (`larger`); where neither is, `f64::min` itself.
pub fn smaller(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.min(b)
    }
}

/// The largest of `from` and the values, NaN where any of them is NaN (`larger`); `from` where there are none.
pub fn largest(from: f64, values: impl IntoIterator<Item = f64>) -> f64 {
    values.into_iter().fold(from, larger)
}

/// √(a² + b²) without overflow or underflow of the squares. Where neither square leaves the normal range -- both parts
/// within 10^±150, or one of them zero -- it is the root of their sum, which is the C library's hypot there to the last
/// bit (the library's own form: measured on 3·10⁸ pairs over 2^±450, every ratio of scales, equal parts and zeros, none
/// apart), inlined and without the library's call; beyond, the library's.
#[inline]
pub fn hypot(a: f64, b: f64) -> f64 {
    let (x, y) = (a.abs(), b.abs());
    let (big, small) = if x >= y { (x, y) } else { (y, x) };
    if big < 1e150 && big > 1e-150 && (small > 1e-150 || small == 0.0) {
        (a * a + b * b).sqrt()
    } else {
        a.hypot(b)
    }
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
        for (row, x) in out.data.chunks_exact_mut(b.cols.max(1)).zip(self.data.chunks_exact(self.cols.max(1))) {
            for (x, b) in x.iter().zip(b.data.chunks_exact(b.cols.max(1))) {
                mul_add(false, row, *x, b);
            }
        }
        out
    }
    /// A† B, without forming A†.
    pub fn adjoint_product(&self, b: &Mat) -> Mat {
        assert_eq!(self.rows, b.rows, "adjoint product of {}x{} and {}x{}", self.rows, self.cols, b.rows, b.cols);
        let mut out = Mat::zeros(self.cols, b.cols);
        for k in 0..self.rows {
            for (i, row) in out.data.chunks_exact_mut(b.cols.max(1)).enumerate().take(self.cols) {
                mul_add(false, row, self[(k, i)].conj(), &b.data[k * b.cols..(k + 1) * b.cols]);
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
        (0..self.cols).map(|j| (0..self.rows).map(|i| self[(i, j)].abs()).sum::<f64>()).fold(0.0, larger)
    }
    /// The Frobenius norm √Σ|a_ij|².
    pub fn frobenius(&self) -> f64 {
        norm_sqr(&self.data).sqrt()
    }
    /// The largest entry's modulus; NaN where an entry is (`largest`), 0 of no entries.
    pub fn max_abs(&self) -> f64 {
        largest(0.0, self.data.iter().map(|z| z.abs()))
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
            mul_add(true, &mut row[k + 1..], l, &pivot_row[k + 1..]);
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
            mul_add(true, &mut row[..columns], *l, other);
        }
    }
    for i in (0..n).rev() {
        let (rest, below) = b.split_at_mut((i + 1) * columns);
        let row = &mut rest[i * columns..];
        for (upper, other) in f[i * n + i + 1..(i + 1) * n].iter().zip(below.chunks_exact(columns)) {
            mul_add(true, &mut row[..columns], *upper, other);
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
    let norm = hypot(ax, ay);
    (ax / norm, (x / ax) * y.conj() / norm)
}

/// (u, w) ← (u c + w s, w c − u s*) on two vectors, number by number, as `C64`'s operators make it (the interleaved
/// kernels' products).
fn rotate(first: &mut [C64], second: &mut [C64], c: f64, s: C64) {
    let n = first.len().min(second.len());
    // SAFETY: n numbers of each; C64 is two f64 (`repr(C)`); the build has AVX2 and FMA.
    unsafe {
        let (up, wp) = (first.as_mut_ptr() as *mut f64, second.as_mut_ptr() as *mut f64);
        let (c4, re, im, conj) = (_mm256_set1_pd(c), _mm256_set1_pd(s.re), _mm256_set1_pd(s.im), _mm256_set1_pd(-s.im));
        let mut j = 0;
        while j + 2 <= n {
            let (u, w) = (_mm256_loadu_pd(up.add(2 * j)), _mm256_loadu_pd(wp.add(2 * j)));
            _mm256_storeu_pd(up.add(2 * j), _mm256_add_pd(_mm256_mul_pd(u, c4), times(re, im, w)));
            _mm256_storeu_pd(wp.add(2 * j), _mm256_sub_pd(_mm256_mul_pd(w, c4), times(re, conj, u)));
            j += 2;
        }
        if j < n {
            let (c2, re, im, conj) = (_mm256_castpd256_pd128(c4), _mm256_castpd256_pd128(re), _mm256_castpd256_pd128(im), _mm256_castpd256_pd128(conj));
            let (u, w) = (_mm_loadu_pd(up.add(2 * j)), _mm_loadu_pd(wp.add(2 * j)));
            _mm_storeu_pd(up.add(2 * j), _mm_add_pd(_mm_mul_pd(u, c2), times1(re, im, w)));
            _mm_storeu_pd(wp.add(2 * j), _mm_sub_pd(_mm_mul_pd(w, c2), times1(re, conj, u)));
        }
    }
}

/// (u, w) ← (u c + w s*, w c − u s) on the columns k and k + 1 of the first `rows` rows of the n × n matrix h (row by
/// row), as `C64`'s operators make it: a row's two numbers in one vector.
fn turn(h: &mut [C64], n: usize, k: usize, rows: usize, c: f64, s: C64) {
    use std::arch::x86_64::{_mm256_blend_pd, _mm256_permute2f128_pd};
    assert!(k + 1 < n && rows <= n && h.len() >= n * n);
    // SAFETY: the two numbers at k of each of the rows lie in h (above); C64 is two f64 (`repr(C)`); AVX2 and FMA.
    unsafe {
        let (c4, re, im) = (_mm256_set1_pd(c), _mm256_set1_pd(s.re), _mm256_setr_pd(-s.im, -s.im, s.im, s.im));
        for row in 0..rows {
            let at = (h.as_mut_ptr() as *mut f64).add(2 * (row * n + k));
            let pair = _mm256_loadu_pd(at);
            let (scaled, crossed) = (_mm256_mul_pd(pair, c4), times(re, im, _mm256_permute2f128_pd::<1>(pair, pair)));
            _mm256_storeu_pd(at, _mm256_blend_pd::<0b1100>(_mm256_add_pd(scaled, crossed), _mm256_sub_pd(scaled, crossed)));
        }
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
        let Some((mut work, scale)) = schur(h, q, v, sums, n) else {
            return false;
        };
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
                mul_add(false, r, *y, q);
            }
            let norm = r.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
            if !(norm > 0.0 && norm.is_finite()) {
                return false;
            }
            for z in r.iter_mut() {
                *z = *z / norm;
            }
            one = larger(one, r.iter().map(|z| z.abs()).sum::<f64>());
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

    /// The eigendecomposition of a block-diagonal A (its entries between two of `blocks` zero), block by block: each
    /// block decomposed alone (`decompose`, each on its own place in `scratch`: none made after the first of a size), its eigencomponents in its own states' places -- R and
    /// R⁻¹ block-diagonal, so that no eigenvector of one block is mixed with another's, not even where two blocks share an
    /// eigenvalue --, `part` each component's block. The condition of the whole R (its columns' largest 1-norm times
    /// ‖R⁻¹‖₁), the blocks' work summed. False where a block's decomposition is (`decompose`).
    pub fn decompose_blocks(&mut self, a: &Mat, blocks: &[Vec<usize>], scratch: &mut Vec<(Mat, Eigen)>, part: &mut Vec<usize>) -> bool {
        let n = a.rows;
        assert_eq!(n, a.cols, "the eigendecomposition of a {}x{} matrix", a.rows, a.cols);
        if self.values.len() != n {
            self.values.resize(n, C64::ZERO);
            for m in [&mut self.vectors, &mut self.inverse] {
                (m.rows, m.cols) = (n, n);
                m.data.resize(n * n, C64::ZERO);
            }
            self.columns[2].resize(n * n, C64::ZERO);
        }
        self.vectors.data.fill(C64::ZERO);
        self.inverse.data.fill(C64::ZERO);
        part.clear();
        part.resize(n, usize::MAX);
        let mut work = 0.0;
        if scratch.len() != blocks.len() {
            scratch.resize_with(blocks.len(), Default::default);
        }
        for (b, states) in blocks.iter().enumerate() {
            let m = states.len();
            let (block, scratch) = &mut scratch[b];
            if block.rows != m {
                *block = Mat::zeros(m, m);
            }
            for (i, &s) in states.iter().enumerate() {
                for (j, &t) in states.iter().enumerate() {
                    block.data[i * m + j] = a[(s, t)];
                }
            }
            if !scratch.decompose(block) {
                return false;
            }
            work += scratch.work;
            for (j, &c) in states.iter().enumerate() {
                self.values[c] = scratch.values[j];
                part[c] = b;
                for (i, &s) in states.iter().enumerate() {
                    self.vectors.data[s * n + c] = scratch.vectors.data[i * m + j];
                    self.inverse.data[c * n + s] = scratch.inverse.data[j * m + i];
                }
            }
        }
        assert!(part.iter().all(|&b| b != usize::MAX), "the blocks are a partition of the states");
        let mut one = 0.0_f64;
        for k in 0..n {
            let column = &mut self.columns[2][k * n..(k + 1) * n];
            for (i, z) in column.iter_mut().enumerate() {
                *z = self.vectors.data[i * n + k];
            }
            one = larger(one, column.iter().map(|z| z.abs()).sum::<f64>());
        }
        self.condition = one * self.inverse.norm1();
        self.work = work;
        self.condition.is_finite()
    }
}

/// The complex Schur form of the n × n matrix in `h` (row by row), in its place: Householder reduction to Hessenberg
/// form and the single-shift implicit QR iteration (`Eigen::decompose`), T = Q†AQ upper triangular in `h` and Q in `q`
/// column by column (a column's numbers side by side); `v` and `sums` are n numbers of scratch. Its complex multiply-adds
/// and the Hessenberg form's largest entry (the deflation's scale); None where the QR iteration does not converge within
/// LAPACK zlahqr's 30·max(10, n) iterations per eigenvalue.
pub(crate) fn schur(h: &mut [C64], q: &mut [C64], v: &mut [C64], sums: &mut [C64], n: usize) -> Option<(usize, f64)> {
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
            mul_add(false, sums, v.conj(), row);
        }
        for s in sums.iter_mut() {
            *s = *s * beta;
        }
        for (v, row) in v[k + 1..].iter().zip(h[(k + 1) * n..].chunks_exact_mut(n)) {
            mul_add(true, row, *v, sums);
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
            mul_add(false, sums, *v, column);
        }
        for s in sums.iter_mut() {
            *s = *s * beta;
        }
        for (v, column) in v[k + 1..].iter().zip(q[(k + 1) * n..].chunks_exact_mut(n)) {
            mul_add(true, column, v.conj(), sums);
        }
        for i in k + 2..n {
            h[i * n + k] = C64::ZERO;
        }
    }
    // The QR iteration on the active block [lo, hi), the rotations accumulated into Q.
    let scale = larger(h.iter().map(|z| z.abs1()).fold(0.0, larger), f64::MIN_POSITIVE);
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
            return None;
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
            turn(h, n, k, (k + 3).min(hi), c, sn);
            let (first, second) = q[k * n..(k + 2) * n].split_at_mut(n);
            rotate(first, second, c, sn.conj());
            if k > lo {
                h[(k + 1) * n + k - 1] = C64::ZERO;
            }
        }
    }
    Some((work, scale))
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
        let sigma = larger(rho, gamma);
        assert!(sigma.is_finite() && rho >= 0.0 && gamma >= 0.0, "the exponential of a matrix of bounds {rho}, {gamma}");
        if sigma == 0.0 {
            return Chebyshev { parts: 1, terms: 0 };
        }
        let corner = C64::new(rho / sigma, gamma / sigma);
        let r = (corner + (corner * corner - C64::ONE).sqrt()).abs();
        let parts = (larger((sigma * gamma).sqrt(), 0.5 * sigma * r / f64::MAX.ln()).ceil() as usize).max(1);
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
        lo = C64::new(smaller(lo.re, d.re - r.re), smaller(lo.im, d.im - r.im));
        hi = C64::new(larger(hi.re, d.re + r.re), larger(hi.im, d.im + r.im));
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
        let sigma = larger(rho, gamma);
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

    /// The four lanes' exponentials (`exps`) against the library's, `C64::exp` -- e^a by the platform's `exp`, cos b and
    /// sin b by its `sin_cos`, each within an ulp --: over 4·10⁵ lanes of a from −720 to 720 (and densely from −50 to 5,
    /// the decays a step meets) and b of every size from 10⁻²⁰ to 2·10⁶ of either sign, beside multiples of π/2 up to
    /// 5·10⁵ of them to within 10⁻¹² to 10⁻³, and the numbers no lane takes (±0, ±∞, NaN, a beyond the normal range, b
    /// beyond 2^19 π/2): within 4u of the modulus |e^{a + ib}| where it is finite, and the library's own -- the same
    /// infinities and NaNs -- where it is not. `times_exp` on lengths of every remainder against the scalar product with
    /// the library's exponential: within 6u of |x| |e^{wz}|.
    #[test]
    fn four_exponentials_are_the_library_s_within_their_round_off() {
        let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
        let mut uniform = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut lanes: Vec<(f64, f64)> = Vec::new();
        for k in 0..400_000 {
            let a = if k % 2 == 0 { -720.0 + 1440.0 * uniform() } else { -50.0 + 55.0 * uniform() };
            let size = 10f64.powf(-20.0 + 26.3 * uniform());
            let b = match k % 3 {
                0 => size,
                1 => -size,
                _ => {
                    let quarter = (uniform() * 5e5).round();
                    quarter * std::f64::consts::FRAC_PI_2 + if uniform() < 0.5 { 1.0 } else { -1.0 } * 10f64.powf(-12.0 + 9.0 * uniform())
                }
            };
            lanes.push((a, b));
        }
        let (nan, inf) = (f64::NAN, f64::INFINITY);
        for a in [0.0, -0.0, 708.0, -708.0, 709.0, 709.5, 710.0, -708.5, -745.0, -746.0, -1e3, inf, -inf, nan] {
            for b in [0.0, -0.0, 1.0, std::f64::consts::FRAC_PI_2, 823_549.0, 823_550.0, 1e7, 1e300, inf, -inf, nan] {
                lanes.push((a, b));
            }
        }
        while lanes.len() % LANES != 0 {
            lanes.push((0.0, 0.0));
        }
        let mut worst = 0.0_f64;
        for four in lanes.chunks_exact(LANES) {
            let found = exps(std::array::from_fn(|l| four[l].0), std::array::from_fn(|l| four[l].1));
            for (&(a, b), found) in four.iter().zip(found) {
                let library = C64::new(a, b).exp();
                if library.re.is_finite() && library.im.is_finite() {
                    let ratio = (found - library).abs() / (UNIT_ROUNDOFF * a.exp());
                    assert!(ratio <= 4.0 || (found - library).abs() == 0.0, "e^({a} + i{b}): {found:?} against {library:?}, {ratio:.2} u of the modulus");
                    worst = larger(worst, if ratio.is_finite() { ratio } else { 0.0 });
                } else {
                    let same = |x: f64, y: f64| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan());
                    assert!(same(found.re, library.re) && same(found.im, library.im), "e^({a} + i{b}): {found:?} against {library:?}");
                }
            }
        }
        eprintln!("{} lanes: within {worst:.2} u of the modulus", lanes.len());
        for n in 1..=9 {
            let x: Vec<C64> = (0..n).map(|_| C64::new(uniform() - 0.5, uniform() - 0.5)).collect();
            // Eigenvalues: frequencies of either sign, decays (their imaginary parts negative).
            let z: Vec<C64> = (0..n).map(|_| C64::new(2e4 * (uniform() - 0.5), -300.0 * uniform())).collect();
            let w = C64::new(0.0, -0.37 * uniform());
            let mut found = x.clone();
            times_exp(&mut found, w, &z);
            for ((found, x), z) in found.iter().zip(&x).zip(&z) {
                let wanted = *x * (w * *z).exp();
                assert!((*found - wanted).abs() <= 6.0 * UNIT_ROUNDOFF * x.abs() * (w * *z).re.exp(), "n {n}: {found:?} against {wanted:?}");
            }
        }
    }

    /// The NaN-carrying maximum and minimum (`larger`, `smaller`, `largest`, `Mat::max_abs`): a NaN anywhere -- the start,
    /// the first, the middle, the last, beside ±∞ -- is the result; ±∞ are numbers like any other; no values leave the
    /// start; and where no NaN is, the result is `f64::max`'s and `f64::min`'s to the bit, signed zeros included.
    #[test]
    fn the_largest_carries_a_nan_and_is_f64_max_elsewhere() {
        let (nan, inf) = (f64::NAN, f64::INFINITY);
        assert_eq!(largest(0.0, []), 0.0);
        assert_eq!(largest(-inf, []), -inf);
        assert!(largest(nan, []).is_nan());
        assert_eq!(largest(0.0, [1.0, -3.0, 2.5]), 2.5);
        assert_eq!(largest(0.0, [-1.0, -3.0]), 0.0);
        assert_eq!(largest(-inf, [-1.0, -3.0]), -1.0);
        assert_eq!(largest(0.0, [1.0, inf, 2.0]), inf);
        assert_eq!(largest(0.0, [-inf]), 0.0);
        for values in [[nan, 1.0, 2.0], [1.0, nan, 2.0], [1.0, 2.0, nan], [inf, nan, -inf], [nan, nan, nan]] {
            assert!(largest(0.0, values).is_nan() && largest(-inf, values).is_nan(), "{values:?}");
            assert!(values.into_iter().fold(inf, smaller).is_nan(), "{values:?}");
        }
        assert!(largest(nan, [1.0, 2.0]).is_nan() && larger(nan, -inf).is_nan() && smaller(inf, nan).is_nan());
        assert_eq!((smaller(-inf, 3.0), smaller(2.0, 3.0), [4.0, -inf, 1.0].into_iter().fold(inf, smaller)), (-inf, 2.0, -inf));
        let numbers = [0.0, -0.0, 1.5, -2.25, 1e-310, -1e300, inf, -inf, f64::MIN_POSITIVE];
        for a in numbers {
            for b in numbers {
                assert_eq!((larger(a, b).to_bits(), smaller(a, b).to_bits()), (a.max(b).to_bits(), a.min(b).to_bits()), "{a:e}, {b:e}");
            }
        }
        let mut m = Mat::from_fn(3, 3, |i, j| C64::new(i as f64 - 1.0, j as f64));
        assert_eq!(m.max_abs(), m.data.iter().map(|z| z.abs()).fold(0.0, f64::max));
        m.data[4] = C64::new(nan, 0.0);
        assert!(m.max_abs().is_nan());
        m.data.iter_mut().for_each(|z| *z = C64::new(nan, nan));
        assert!(m.max_abs().is_nan(), "a block of NaN is NaN, not 0");
        m.data[7] = C64::new(-inf, 0.0);
        assert!(m.max_abs().is_nan());
        m.data.iter_mut().for_each(|z| *z = C64::ZERO);
        m.data[2] = C64::new(0.0, -inf);
        assert_eq!(m.max_abs(), inf);
        assert_eq!(Mat::from_fn(0, 0, |_, _| C64::ONE).max_abs(), 0.0);
    }
    use crate::random::{Purpose, Stream};
    use crate::units::rounding;

    /// A⁻¹ B by a factorization.
    fn solve_columns(lu: &Lu, b: &Mat) -> Mat {
        let mut out = b.clone();
        solve(&lu.factors.data, &lu.pivots, lu.factors.rows, &mut out.data, b.cols);
        out
    }

    /// The split kernel (`combine`) is the complex sums it stands for, to the rounding of a sum of its length; and its
    /// pace (printed: ns a complex multiply-add on five columns of 16 rows, a five-state cluster's jumps on its state).
    #[test]
    fn the_split_kernels_are_their_complex_sums() {
        let mut s = Stream::new(11, Purpose::Initial, 0, 0);
        let mut draw = |count: usize| -> Vec<f64> { (0..count).map(|_| s.uniform() - 0.5).collect() };
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
            combine(false, std::hint::black_box(&weights[..count]), count, std::hint::black_box(&columns), &mut sums, 1, 0);
        }
        std::hint::black_box(sums);
        eprintln!("combine {:.3} ns a complex multiply-add", started.elapsed().as_secs_f64() * 1e9 / (rounds * rows * count) as f64);
    }

    /// The interleaved kernels (`mul_add`, `fused_product`, `rotate`, `turn`) and `hypot` are their scalar forms bit for
    /// bit, at every length from 0 to 17 (whole vectors and the odd number after them), on
    /// numbers of many scales and signs, zeros among them.
    #[test]
    fn the_vector_kernels_are_their_scalar_forms_bit_for_bit() {
        let mut s = Stream::new(29, Purpose::Initial, 0, 0);
        let mut number = || {
            let scale = 2f64.powi((s.uniform() * 60.0) as i32 - 30);
            let mut part = || if s.uniform() < 0.05 { 0.0 } else { (s.uniform() - 0.5) * scale };
            C64::new(part(), part())
        };
        let bits = |z: &[C64]| z.iter().map(|z| (z.re.to_bits(), z.im.to_bits())).collect::<Vec<_>>();
        // `hypot` against the library's, over 2^±600 and every ratio of the parts' scales, equal parts and zeros.
        let mut t = Stream::new(37, Purpose::Initial, 0, 0);
        for k in 0..2_000_000 {
            let first = (t.uniform() * 1200.0) as i32 - 600;
            let second = if k % 2 == 0 { first + (t.uniform() * 120.0) as i32 - 60 } else { (t.uniform() * 1200.0) as i32 - 600 };
            let (a, b) = ((t.uniform() - 0.5) * 2f64.powi(first), (t.uniform() - 0.5) * 2f64.powi(second));
            let b = [b, 0.0, a, -a, b][k % 5];
            assert_eq!(hypot(a, b).to_bits(), a.hypot(b).to_bits(), "hypot({a:e}, {b:e})");
        }
        for n in 0..18 {
            let (y, z, f): (Vec<C64>, Vec<C64>, C64) = ((0..n).map(|_| number()).collect(), (0..n).map(|_| number()).collect(), number());
            for subtract in [false, true] {
                let mut kernel = z.clone();
                mul_add(subtract, &mut kernel, f, &y);
                let scalar: Vec<C64> = z.iter().zip(&y).map(|(z, y)| if subtract { *z - f * *y } else { *z + f * *y }).collect();
                assert_eq!(bits(&kernel), bits(&scalar), "mul_add of {n}, subtract {subtract}");
            }
            let c = number().re;
            let (mut u, mut w) = (z.clone(), y.clone());
            rotate(&mut u, &mut w, c, f);
            let rotated: Vec<(C64, C64)> = z.iter().zip(&y).map(|(u, w)| (*u * c + *w * f, *w * c - *u * f.conj())).collect();
            assert_eq!((bits(&u), bits(&w)), (bits(&rotated.iter().map(|p| p.0).collect::<Vec<_>>()), bits(&rotated.iter().map(|p| p.1).collect::<Vec<_>>())), "rotate of {n}");
            if n >= 2 {
                let h: Vec<C64> = (0..n * n).map(|_| number()).collect();
                for k in 0..n - 1 {
                    let mut kernel = h.clone();
                    turn(&mut kernel, n, k, n, c, f);
                    let mut scalar = h.clone();
                    for row in scalar.chunks_exact_mut(n) {
                        let (u, w) = (row[k], row[k + 1]);
                        row[k] = u * c + w * f.conj();
                        row[k + 1] = w * c - u * f;
                    }
                    assert_eq!(bits(&kernel), bits(&scalar), "turn of {n} at {k}");
                }
            }
            for (r, k) in [(1, n), (3, n), (n, 5), (n, 8), (2, 0)] {
                let (a, b): (Vec<C64>, Vec<C64>) = ((0..r * k).map(|_| number()).collect(), (0..k * n).map(|_| number()).collect());
                let mut kernel = vec![C64::new(7.0, 7.0); r * n];
                fused_product(&a, &b, (r, k, n), &mut kernel);
                let mut scalar = vec![C64::ZERO; r * n];
                for (o, a) in scalar.chunks_exact_mut(n.max(1)).zip(a.chunks_exact(k.max(1))).take(if k == 0 { 0 } else { r }) {
                    for (x, b) in a.iter().zip(b.chunks_exact(n.max(1))) {
                        for (o, y) in o.iter_mut().zip(b) {
                            o.re = x.re.mul_add(y.re, (-x.im).mul_add(y.im, o.re));
                            o.im = x.re.mul_add(y.im, x.im.mul_add(y.re, o.im));
                        }
                    }
                }
                assert_eq!(bits(&kernel), bits(&scalar), "fused product {r} × {k} × {n}");
            }
        }
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
        a.data.iter().zip(&b.data).fold(0.0, |m, (x, y)| larger(m, (*x - *y).abs()))
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
            let worst = e.values.iter().zip(&d).fold(0.0_f64, |m, (x, y)| larger(m, (x - y).abs()));
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
        let worst = found.iter().zip(&lambda).fold(0.0_f64, |m, (x, y)| larger(m, (*x - *y).abs()));
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

    /// The decomposition block by block (`Eigen::decompose_blocks`) of a block-diagonal generator of the same shape, its
    /// blocks interleaved among the states (of 1 to 9 states, two of them sharing an eigenvalue): R Λ R⁻¹ = A to
    /// γ κ(R) ‖A‖, R and R⁻¹ zero between two blocks -- every eigenvector of one block, even where two blocks share an
    /// eigenvalue --, each component's block its states', a second call on the same scratch the same to the bit.
    #[test]
    fn the_block_eigendecomposition_gives_the_matrix_back() {
        let blocks: Vec<Vec<usize>> = vec![vec![0, 3, 7, 12], vec![1, 2], vec![4], vec![5, 6, 8, 9, 10, 11, 13, 14, 15]];
        let n = 16;
        let block_of: Vec<usize> = (0..n).map(|s| blocks.iter().position(|b| b.contains(&s)).unwrap()).collect();
        let mut s = Stream::new(5, Purpose::Latin, n as u64, 2);
        let full = sample(n, n, 4242).hermitian_part().scaled(C64::real(50.0));
        let mut h = Mat::from_fn(n, n, |i, j| if block_of[i] == block_of[j] { full[(i, j)] } else { C64::ZERO });
        for i in 0..n {
            h[(i, i)] += C64::new(400.0 * (s.uniform() - 0.5), -19.0 * (s.uniform() + 0.5));
        }
        // The lone state's eigenvalue is its diagonal: set to another block's first eigenvalue, the two blocks share it.
        let other = Mat::from_fn(2, 2, |i, j| h[(blocks[1][i], blocks[1][j])]);
        h[(4, 4)] = eigen(&other).unwrap().values[0];
        let (mut e, mut scratch, mut part) = (Eigen::default(), Vec::new(), Vec::new());
        assert!(e.decompose_blocks(&h, &blocks, &mut scratch, &mut part), "converges");
        let back = e.vectors.product(&Mat::diagonal(&e.values)).product(&e.inverse);
        let bound = rounding(4 * n * n) * e.condition * h.max_abs();
        let error = difference(&back, &h);
        assert!(error <= bound, "{error:e} > {bound:e}");
        for a in 0..n {
            assert_eq!(part[a], block_of[a], "component {a}'s block");
            for c in (0..n).filter(|&c| block_of[c] != block_of[a]) {
                assert!(e.vectors[(a, c)] == C64::ZERO && e.inverse[(c, a)] == C64::ZERO, "({a}, {c}) between two blocks");
            }
        }
        let first = (e.values.clone(), e.vectors.clone(), e.inverse.clone(), e.condition);
        assert!(e.decompose_blocks(&h, &blocks, &mut scratch, &mut part));
        assert!(first == (e.values.clone(), e.vectors.clone(), e.inverse.clone(), e.condition), "a second call the same");
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
            let most = hermitian_eigen(part).values.iter().map(|l| (l - scalar).abs()).fold(0.0, larger);
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
            let error = w.iter().zip(&want).fold(0.0_f64, |m, (x, y)| larger(m, (*x - *y).abs()));
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
                let error = x.iter().zip(&want).fold(0.0_f64, |m, (p, q)| larger(m, (*p - *q).abs()));
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
