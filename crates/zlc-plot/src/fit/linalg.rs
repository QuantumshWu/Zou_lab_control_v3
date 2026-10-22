//! The LAPACK routines numba's `np.linalg.eigh`, `np.linalg.svd` and
//! `_GEQRF` reach in the compiled lane, as reference LAPACK 3.12 writes
//! them (`dsyevd -> dsytd2/dsteqr/dorm2r`, `dgeqrf -> dgeqr2`,
//! `dgesdd -> dgebd2/dbdsdc/dbdsqr/dormbr`), over the OpenBLAS kernels of
//! [`super::blas`].  Bit-exact with the oracle on every probed matrix.

use super::blas::{self, Mat};
use super::scratch::Scratch;

const SAFMIN: f64 = f64::MIN_POSITIVE; // DLAMCH('S') = 2^-1022
const EPS_E: f64 = f64::EPSILON / 2.0; // DLAMCH('E') = 2^-53
const EPS_P: f64 = f64::EPSILON; // DLAMCH('P') = 2^-52
const HUGE: f64 = f64::MAX;

/// What numba raises from a LAPACK wrapper.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinAlgError {
    NonFinite,
    NoConvergence(&'static str),
    Factorization(&'static str),
}

impl std::fmt::Display for LinAlgError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LinAlgError::NonFinite => f.write_str("Array must not contain infs or NaNs."),
            LinAlgError::NoConvergence(text) | LinAlgError::Factorization(text) => f.write_str(text),
        }
    }
}

#[inline(always)]
fn sign(a: f64, b: f64) -> f64 { a.abs().copysign(b) }

/// Fortran `MAX(a, b)` as gfortran lowers it for finite operands.
#[inline(always)]
fn fmax(a: f64, b: f64) -> f64 { if a >= b { a } else { b } }
#[inline(always)]
fn fmin(a: f64, b: f64) -> f64 { if a <= b { a } else { b } }

fn dlapy2(x: f64, y: f64) -> f64 {
    let x_nan = x.is_nan();
    let y_nan = y.is_nan();
    let mut value = 0.0;
    if x_nan {
        value = x;
    }
    if y_nan {
        value = y;
    }
    if !(x_nan || y_nan) {
        let xa = x.abs();
        let ya = y.abs();
        let w = fmax(xa, ya);
        let z = fmin(xa, ya);
        value = if z == 0.0 || w > HUGE {
            w
        } else {
            let q = z / w;
            w * (1.0 + q * q).sqrt()
        };
    }
    value
}

/// `dlartg` (LAPACK 3.10+ `la_xlartg`): `(c, s, r)`.
pub fn dlartg(f: f64, g: f64) -> (f64, f64, f64) {
    let rtmin = SAFMIN.sqrt();
    let rtmax = ((1.0 / SAFMIN) / 2.0).sqrt();
    let f1 = f.abs();
    let g1 = g.abs();
    if g == 0.0 {
        return (1.0, 0.0, f);
    }
    if f == 0.0 {
        return (0.0, sign(1.0, g), g1);
    }
    if f1 > rtmin && f1 < rtmax && g1 > rtmin && g1 < rtmax {
        let d = (f * f + g * g).sqrt();
        let c = f1 / d;
        let r = sign(d, f);
        let s = g / r;
        return (c, s, r);
    }
    let u = fmin(1.0 / SAFMIN, fmax(fmax(SAFMIN, f1), g1));
    let fs = f / u;
    let gs = g / u;
    let d = (fs * fs + gs * gs).sqrt();
    let c = fs.abs() / d;
    let r = sign(d, f);
    let s = gs / r;
    (c, s, r * u)
}

/// `dlaev2(a, b, c)`: `(rt1, rt2, cs1, sn1)`.
fn dlaev2(a: f64, b: f64, c: f64) -> (f64, f64, f64, f64) {
    let sm = a + c;
    let df = a - c;
    let adf = df.abs();
    let tb = b + b;
    let ab = tb.abs();
    let (acmx, acmn) = if a.abs() > c.abs() { (a, c) } else { (c, a) };
    let rt = if adf > ab {
        let q = ab / adf;
        adf * (1.0 + q * q).sqrt()
    } else if adf < ab {
        let q = adf / ab;
        ab * (1.0 + q * q).sqrt()
    } else {
        ab * 2.0_f64.sqrt()
    };
    let (rt1, rt2, sgn1);
    if sm < 0.0 {
        rt1 = 0.5 * (sm - rt);
        sgn1 = -1;
        rt2 = (acmx / rt1) * acmn - (b / rt1) * b;
    } else if sm > 0.0 {
        rt1 = 0.5 * (sm + rt);
        sgn1 = 1;
        rt2 = (acmx / rt1) * acmn - (b / rt1) * b;
    } else {
        rt1 = 0.5 * rt;
        rt2 = -0.5 * rt;
        sgn1 = 1;
    }
    let (cs, sgn2) = if df >= 0.0 { (df + rt, 1) } else { (df - rt, -1) };
    let acs = cs.abs();
    let (mut cs1, mut sn1);
    if acs > ab {
        let ct = -tb / cs;
        sn1 = 1.0 / (1.0 + ct * ct).sqrt();
        cs1 = ct * sn1;
    } else if ab == 0.0 {
        cs1 = 1.0;
        sn1 = 0.0;
    } else {
        let tn = -cs / tb;
        cs1 = 1.0 / (1.0 + tn * tn).sqrt();
        sn1 = tn * cs1;
    }
    if sgn1 == sgn2 {
        let tn = cs1;
        cs1 = -sn1;
        sn1 = tn;
    }
    (rt1, rt2, cs1, sn1)
}

/// `dlarfg(n, alpha, x)`: returns `(beta, tau)`, scaling `x` in place.
fn dlarfg(n: usize, alpha: f64, x: &mut [f64]) -> (f64, f64) {
    if n <= 1 {
        return (alpha, 0.0);
    }
    let mut alpha = alpha;
    let mut xnorm = blas::dnrm2(x);
    if xnorm == 0.0 {
        return (alpha, 0.0);
    }
    let mut beta = -sign(dlapy2(alpha, xnorm), alpha);
    let safmin = SAFMIN / EPS_E;
    let mut knt = 0;
    if beta.abs() < safmin {
        let rsafmn = 1.0 / safmin;
        loop {
            knt += 1;
            for v in x.iter_mut() {
                *v *= rsafmn;
            }
            beta *= rsafmn;
            alpha *= rsafmn;
            if !(beta.abs() < safmin && knt < 20) {
                break;
            }
        }
        xnorm = blas::dnrm2(x);
        beta = -sign(dlapy2(alpha, xnorm), alpha);
    }
    let tau = (beta - alpha) / beta;
    let scal = 1.0 / (alpha - beta);
    for v in x.iter_mut() {
        *v *= scal;
    }
    for _ in 0..knt {
        beta *= safmin;
    }
    (beta, tau)
}

/// `dlascl('G', ..., cfrom, cto, ...)` on a list of values.
fn dlascl(cfrom: f64, cto: f64, values: &mut [f64]) {
    let smlnum = SAFMIN;
    let bignum = 1.0 / smlnum;
    let mut cfromc = cfrom;
    let mut ctoc = cto;
    loop {
        let cfrom1 = cfromc * smlnum;
        let (mul, done);
        if cfrom1 == cfromc {
            mul = ctoc / cfromc;
            done = true;
        } else {
            let cto1 = ctoc / bignum;
            if cto1 == ctoc {
                mul = ctoc;
                done = true;
                cfromc = 1.0;
            } else if cfrom1.abs() > ctoc.abs() && ctoc != 0.0 {
                mul = smlnum;
                done = false;
                cfromc = cfrom1;
            } else if cto1.abs() > cfromc.abs() {
                mul = bignum;
                done = false;
                ctoc = cto1;
            } else {
                mul = ctoc / cfromc;
                done = true;
                if mul == 1.0 {
                    return;
                }
            }
        }
        for v in values.iter_mut() {
            *v *= mul;
        }
        if done {
            return;
        }
    }
}

/// `dlarf('L', m, n, v, 1, tau, C(r0, c0), ldc, work)`.
fn dlarf_left(m: usize, n: usize, v: &[f64], tau: f64, c: &mut Mat, r0: usize, c0: usize) {
    if tau == 0.0 {
        return;
    }
    let mut lastv = m;
    while lastv > 0 && v[lastv - 1] == 0.0 {
        lastv -= 1;
    }
    if lastv == 0 {
        return;
    }
    let lastc = if n == 0 {
        0
    } else if c.get(r0, c0 + n - 1) != 0.0 || c.get(r0 + lastv - 1, c0 + n - 1) != 0.0 {
        n
    } else {
        let mut found = 0;
        for j in (1..=n).rev() {
            if (0..lastv).any(|i| c.get(r0 + i, c0 + j - 1) != 0.0) {
                found = j;
                break;
            }
        }
        found
    };
    let w = blas::dgemv_t(lastv, lastc, 1.0, c, r0, c0, &v[..lastv]);
    blas::dger(lastv, lastc, -tau, &v[..lastv], &w, c, r0, c0);
}

/// `dlarf('R', m, n, v, incv, tau, C(r0, c0), ldc, work)`.
fn dlarf_right(m: usize, n: usize, v: &[f64], tau: f64, c: &mut Mat, r0: usize, c0: usize) {
    if tau == 0.0 {
        return;
    }
    let mut lastv = n;
    while lastv > 0 && v[lastv - 1] == 0.0 {
        lastv -= 1;
    }
    if lastv == 0 {
        return;
    }
    let lastc = if m == 0 {
        0
    } else if c.get(r0 + m - 1, c0) != 0.0 || c.get(r0 + m - 1, c0 + lastv - 1) != 0.0 {
        m
    } else {
        let mut found = 0;
        for j in 0..lastv {
            let mut i = m;
            while i >= 1 && c.get(r0 + i - 1, c0 + j) == 0.0 {
                i -= 1;
            }
            found = found.max(i);
        }
        found
    };
    let w = blas::dgemv_n_strided(lastc, lastv, 1.0, c, r0, c0, &v[..lastv]);
    // dger(lastc, lastv, -tau, w, 1, v, incv, C)
    let mut column = vec![0.0; lastc];
    for j in 0..lastv {
        let t = -tau * v[j];
        for (i, value) in column.iter_mut().enumerate() {
            *value = c.get(r0 + i, c0 + j);
        }
        blas::daxpy(t, &w, &mut column);
        for (i, &value) in column.iter().enumerate() {
            c.set(r0 + i, c0 + j, value);
        }
    }
}

fn dsytd2_lower(n: usize, a: &mut Mat) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut d = vec![0.0; n];
    let mut e = vec![0.0; n.saturating_sub(1)];
    let mut tau = vec![0.0; n.saturating_sub(1)];
    for i in 0..n.saturating_sub(1) {
        let mut x: Vec<f64> = ((i + 2)..n).map(|r| a.get(r, i)).collect();
        let (alpha, taui) = dlarfg(n - i - 1, a.get(i + 1, i), &mut x);
        for (k, r) in ((i + 2)..n).enumerate() {
            a.set(r, i, x[k]);
        }
        a.set(i + 1, i, alpha);
        e[i] = alpha;
        if taui != 0.0 {
            a.set(i + 1, i, 1.0);
            let m = n - i - 1;
            let v: Vec<f64> = (0..m).map(|k| a.get(i + 1 + k, i)).collect();
            let mut y = vec![0.0; m];
            {
                let view = |r: usize, c: usize| a.get(i + 1 + r, i + 1 + c);
                blas::dsymv_lower(m, taui, &view, &v, &mut y);
            }
            let alpha2 = -0.5 * taui * blas::ddot(&y, &v);
            blas::daxpy(alpha2, &v, &mut y);
            blas::dsyr2_lower(m, -1.0, &v, &y, a, i + 1, i + 1);
            a.set(i + 1, i, e[i]);
        }
        d[i] = a.get(i, i);
        tau[i] = taui;
    }
    if n > 0 {
        d[n - 1] = a.get(n - 1, n - 1);
    }
    (d, e, tau)
}

fn dlasr_right_forward(rows: usize, count: usize, c: &[f64], s: &[f64], z: &mut Mat, col0: usize) {
    for j in 0..count.saturating_sub(1) {
        let (ct, st) = (c[j], s[j]);
        if ct != 1.0 || st != 0.0 {
            for i in 0..rows {
                let temp = z.get(i, col0 + j + 1);
                let here = z.get(i, col0 + j);
                z.set(i, col0 + j + 1, ct * temp - st * here);
                z.set(i, col0 + j, st * temp + ct * here);
            }
        }
    }
}

fn dlasr_right_backward(rows: usize, count: usize, c: &[f64], s: &[f64], z: &mut Mat, col0: usize) {
    for j in (0..count.saturating_sub(1)).rev() {
        let (ct, st) = (c[j], s[j]);
        if ct != 1.0 || st != 0.0 {
            for i in 0..rows {
                let temp = z.get(i, col0 + j + 1);
                let here = z.get(i, col0 + j);
                z.set(i, col0 + j + 1, ct * temp - st * here);
                z.set(i, col0 + j, st * temp + ct * here);
            }
        }
    }
}

fn dlasr_left_forward(count: usize, cols: usize, c: &[f64], s: &[f64], a: &mut Mat, row0: usize) {
    for j in 0..count.saturating_sub(1) {
        let (ct, st) = (c[j], s[j]);
        if ct != 1.0 || st != 0.0 {
            for i in 0..cols {
                let temp = a.get(row0 + j + 1, i);
                let here = a.get(row0 + j, i);
                a.set(row0 + j + 1, i, ct * temp - st * here);
                a.set(row0 + j, i, st * temp + ct * here);
            }
        }
    }
}

fn dlasr_left_backward(count: usize, cols: usize, c: &[f64], s: &[f64], a: &mut Mat, row0: usize) {
    for j in (0..count.saturating_sub(1)).rev() {
        let (ct, st) = (c[j], s[j]);
        if ct != 1.0 || st != 0.0 {
            for i in 0..cols {
                let temp = a.get(row0 + j + 1, i);
                let here = a.get(row0 + j, i);
                a.set(row0 + j + 1, i, ct * temp - st * here);
                a.set(row0 + j, i, st * temp + ct * here);
            }
        }
    }
}

fn dlanst_max(d: &[f64], e: &[f64]) -> f64 {
    let n = d.len();
    if n == 0 {
        return 0.0;
    }
    let mut anorm = d[n - 1].abs();
    for i in 0..n - 1 {
        let s = d[i].abs();
        if anorm < s || s.is_nan() {
            anorm = s;
        }
        let s = e[i].abs();
        if anorm < s || s.is_nan() {
            anorm = s;
        }
    }
    anorm
}

/// `dsteqr('I', ...)`: eigenvalues ascending in `d`, vectors in `z`.
fn dsteqr_identity(n: usize, d: &mut [f64], e: &mut [f64]) -> Result<Mat, LinAlgError> {
    let mut z = Mat::identity(n);
    if n <= 1 {
        return Ok(z);
    }
    let eps = EPS_E;
    let eps2 = eps * eps;
    let safmin = SAFMIN;
    let safmax = 1.0 / safmin;
    let ssfmax = safmax.sqrt() / 3.0;
    let ssfmin = safmin.sqrt() / eps2;
    let nmaxit = n * 30;
    let mut jtot = 0;
    // 1-based indices below, as the Fortran.
    let mut l1 = 1usize;
    let nm1 = n - 1;
    let dd = |k: usize| k - 1;
    loop {
        if l1 > n {
            break;
        }
        if l1 > 1 {
            e[l1 - 2] = 0.0;
        }
        let mut m = n;
        if l1 <= nm1 {
            for mm in l1..=nm1 {
                let tst = e[mm - 1].abs();
                if tst == 0.0 {
                    m = mm;
                    break;
                }
                if tst <= (d[dd(mm)].abs().sqrt() * d[mm].abs().sqrt()) * eps {
                    e[mm - 1] = 0.0;
                    m = mm;
                    break;
                }
            }
        }
        let mut l = l1;
        let lsv = l;
        let mut lend = m;
        let lendsv = lend;
        l1 = m + 1;
        if lend == l {
            continue;
        }
        let anorm = dlanst_max(&d[l - 1..lend], &e[l - 1..lend - 1]);
        let mut iscale = 0;
        if anorm == 0.0 {
            continue;
        }
        if anorm > ssfmax {
            iscale = 1;
            dlascl(anorm, ssfmax, &mut d[l - 1..lend]);
            dlascl(anorm, ssfmax, &mut e[l - 1..lend - 1]);
        } else if anorm < ssfmin {
            iscale = 2;
            dlascl(anorm, ssfmin, &mut d[l - 1..lend]);
            dlascl(anorm, ssfmin, &mut e[l - 1..lend - 1]);
        }
        if d[lend - 1].abs() < d[l - 1].abs() {
            lend = lsv;
            l = lendsv;
        }
        if lend > l {
            // QL iteration
            loop {
                let mut m = lend;
                if l != lend {
                    for mm in l..lend {
                        let tst = e[mm - 1].abs() * e[mm - 1].abs();
                        if tst <= (eps2 * d[mm - 1].abs()) * d[mm].abs() + safmin {
                            m = mm;
                            break;
                        }
                    }
                }
                if m < lend {
                    e[m - 1] = 0.0;
                }
                let mut p = d[l - 1];
                if m == l {
                    d[l - 1] = p;
                    l += 1;
                    if l <= lend {
                        continue;
                    }
                    break;
                }
                if m == l + 1 {
                    let (rt1, rt2, c, s) = dlaev2(d[l - 1], e[l - 1], d[l]);
                    dlasr_right_backward(n, 2, &[c], &[s], &mut z, l - 1);
                    d[l - 1] = rt1;
                    d[l] = rt2;
                    e[l - 1] = 0.0;
                    l += 2;
                    if l <= lend {
                        continue;
                    }
                    break;
                }
                if jtot == nmaxit {
                    break;
                }
                jtot += 1;
                let mut g = (d[l] - p) / (2.0 * e[l - 1]);
                let mut r = dlapy2(g, 1.0);
                g = d[m - 1] - p + (e[l - 1] / (g + sign(r, g)));
                let mut s = 1.0;
                let mut c = 1.0;
                p = 0.0;
                let count = m - l;
                let mut cs = vec![0.0; count];
                let mut ss = vec![0.0; count];
                let mut i = m - 1;
                while i >= l {
                    let f = s * e[i - 1];
                    let b = c * e[i - 1];
                    let rot = dlartg(g, f);
                    c = rot.0;
                    s = rot.1;
                    r = rot.2;
                    if i != m - 1 {
                        e[i] = r;
                    }
                    g = d[i] - p;
                    r = (d[i - 1] - g) * s + 2.0 * c * b;
                    p = s * r;
                    d[i] = g + p;
                    g = c * r - b;
                    cs[i - l] = c;
                    ss[i - l] = -s;
                    if i == 0 {
                        break;
                    }
                    i -= 1;
                }
                dlasr_right_backward(n, m - l + 1, &cs, &ss, &mut z, l - 1);
                d[l - 1] -= p;
                e[l - 1] = g;
            }
        } else {
            // QR iteration
            loop {
                let mut m = lend;
                if l != lend {
                    let mut mm = l;
                    while mm > lend {
                        let tst = e[mm - 2].abs() * e[mm - 2].abs();
                        if tst <= (eps2 * d[mm - 1].abs()) * d[mm - 2].abs() + safmin {
                            m = mm;
                            break;
                        }
                        mm -= 1;
                    }
                }
                if m > lend {
                    e[m - 2] = 0.0;
                }
                let mut p = d[l - 1];
                if m == l {
                    d[l - 1] = p;
                    if l == 0 {
                        break;
                    }
                    l -= 1;
                    if l >= lend {
                        continue;
                    }
                    break;
                }
                if m + 1 == l {
                    let (rt1, rt2, c, s) = dlaev2(d[l - 2], e[l - 2], d[l - 1]);
                    dlasr_right_forward(n, 2, &[c], &[s], &mut z, l - 2);
                    d[l - 2] = rt1;
                    d[l - 1] = rt2;
                    e[l - 2] = 0.0;
                    if l < 2 + lend {
                        break;
                    }
                    l -= 2;
                    if l >= lend {
                        continue;
                    }
                    break;
                }
                if jtot == nmaxit {
                    break;
                }
                jtot += 1;
                let mut g = (d[l - 2] - p) / (2.0 * e[l - 2]);
                let mut r = dlapy2(g, 1.0);
                g = d[m - 1] - p + (e[l - 2] / (g + sign(r, g)));
                let mut s = 1.0;
                let mut c = 1.0;
                p = 0.0;
                let count = l - m;
                let mut cs = vec![0.0; count];
                let mut ss = vec![0.0; count];
                for i in m..l {
                    let f = s * e[i - 1];
                    let b = c * e[i - 1];
                    let rot = dlartg(g, f);
                    c = rot.0;
                    s = rot.1;
                    r = rot.2;
                    if i != m {
                        e[i - 2] = r;
                    }
                    g = d[i - 1] - p;
                    r = (d[i] - g) * s + 2.0 * c * b;
                    p = s * r;
                    d[i - 1] = g + p;
                    g = c * r - b;
                    cs[i - m] = c;
                    ss[i - m] = s;
                }
                dlasr_right_forward(n, l - m + 1, &cs, &ss, &mut z, m - 1);
                d[l - 1] -= p;
                e[l - 2] = g;
            }
        }
        if iscale == 1 {
            dlascl(ssfmax, anorm, &mut d[lsv - 1..lendsv]);
            dlascl(ssfmax, anorm, &mut e[lsv - 1..lendsv - 1]);
        } else if iscale == 2 {
            dlascl(ssfmin, anorm, &mut d[lsv - 1..lendsv]);
            dlascl(ssfmin, anorm, &mut e[lsv - 1..lendsv - 1]);
        }
        if jtot >= nmaxit {
            if e.iter().any(|&v| v != 0.0) {
                return Err(LinAlgError::NoConvergence("Internal algorithm failed to converge."));
            }
            return Ok(z);
        }
    }
    // Selection sort, ascending.
    for ii in 2..=n {
        let i = ii - 1;
        let mut k = i;
        let mut p = d[i - 1];
        for j in ii..=n {
            if d[j - 1] < p {
                k = j;
                p = d[j - 1];
            }
        }
        if k != i {
            d[k - 1] = d[i - 1];
            d[i - 1] = p;
            for r in 0..n {
                let a = z.get(r, i - 1);
                let b = z.get(r, k - 1);
                z.set(r, i - 1, b);
                z.set(r, k - 1, a);
            }
        }
    }
    Ok(z)
}

/// `dorm2r('L', 'N', m, n, k, A(ar0, ac0), lda, tau, C(cr0, cc0), ldc)`.
fn dorm2r_left_notrans(m: usize, n: usize, k: usize, a: &mut Mat, ar0: usize, ac0: usize, tau: &[f64], c: &mut Mat, cr0: usize, cc0: usize) {
    for i in (1..=k).rev() {
        let mi = m - i + 1;
        let aii = a.get(ar0 + i - 1, ac0 + i - 1);
        a.set(ar0 + i - 1, ac0 + i - 1, 1.0);
        let v: Vec<f64> = (0..mi).map(|r| a.get(ar0 + i - 1 + r, ac0 + i - 1)).collect();
        dlarf_left(mi, n, &v, tau[i - 1], c, cr0 + i - 1, cc0);
        a.set(ar0 + i - 1, ac0 + i - 1, aii);
    }
}

/// numba's `np.linalg.eigh(a)` (LAPACK `dsyevd`, `JOBZ='V'`, `UPLO='L'`)
/// of a row-major `n x n` matrix: eigenvalues ascending, eigenvectors
/// row-major (`vectors[i * n + j]` = component `i` of vector `j`).
pub fn eigh_lower(n: usize, matrix: &[f64]) -> Result<(Vec<f64>, Vec<f64>), LinAlgError> {
    if matrix.iter().take(n * n).any(|v| !v.is_finite()) {
        return Err(LinAlgError::NonFinite);
    }
    if n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let mut a = Mat::from_row_major(n, n, matrix);
    if n == 1 {
        return Ok((vec![a.get(0, 0)], vec![1.0]));
    }
    let smlnum = SAFMIN / EPS_P;
    let bignum = 1.0 / smlnum;
    let rmin = smlnum.sqrt();
    let rmax = bignum.sqrt();
    let mut anrm = 0.0;
    for j in 0..n {
        for i in j..n {
            let s = a.get(i, j).abs();
            if anrm < s || s.is_nan() {
                anrm = s;
            }
        }
    }
    let mut sigma = 1.0;
    let mut iscale = false;
    if anrm > 0.0 && anrm < rmin {
        iscale = true;
        sigma = rmin / anrm;
    } else if anrm > rmax {
        iscale = true;
        sigma = rmax / anrm;
    }
    if iscale {
        let mut lower: Vec<f64> = Vec::new();
        for j in 0..n {
            for i in j..n {
                lower.push(a.get(i, j));
            }
        }
        dlascl(1.0, sigma, &mut lower);
        let mut k = 0;
        for j in 0..n {
            for i in j..n {
                a.set(i, j, lower[k]);
                k += 1;
            }
        }
    }
    let (mut d, mut e, tau) = dsytd2_lower(n, &mut a);
    let mut z = dsteqr_identity(n, &mut d, &mut e)?;
    dorm2r_left_notrans(n - 1, n, n - 1, &mut a, 1, 0, &tau, &mut z, 1, 0);
    if iscale {
        let inverse = 1.0 / sigma;
        for v in d.iter_mut() {
            *v *= inverse;
        }
    }
    Ok((d, z.to_row_major()))
}

/// numba's `_GEQRF` of the row-major `m x n` matrix (`m > n`): the upper
/// triangle `R`, row-major `n x n`.
pub fn qr_r(m: usize, n: usize, matrix: &[f64]) -> Result<Vec<f64>, LinAlgError> {
    // A tall Jacobian's copy comes from the thread's pool (see scratch.rs).
    let mut a = Mat { rows: m, cols: n, data: Scratch::filled(m * n, 0.0).into_vec() };
    for (i, row) in matrix.chunks_exact(n.max(1)).take(m).enumerate() {
        for (j, &value) in row.iter().enumerate() {
            a.data[j * m + i] = value;
        }
    }
    let k = m.min(n);
    for i in 0..k {
        // Column i below the diagonal is contiguous: reflect it in place.
        let column = &mut a.data[i * m + i..(i + 1) * m];
        let (head, tail) = column.split_at_mut(1);
        let (beta, tau) = dlarfg(m - i, head[0], tail);
        head[0] = beta;
        if i + 1 < n {
            let mut v = column.to_vec();
            v[0] = 1.0;
            dlarf_left(m - i, n - i - 1, &v, tau, &mut a, i, i + 1);
        }
    }
    let mut r = vec![0.0; n * n];
    for row in 0..n {
        for column in row..n {
            r[row * n + column] = a.get(row, column);
        }
    }
    drop(Scratch::adopt(a.data));
    Ok(r)
}

fn dgebd2(m: usize, n: usize, a: &mut Mat) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut d = vec![0.0; n];
    let mut e = vec![0.0; n.saturating_sub(1)];
    let mut tauq = vec![0.0; n];
    let mut taup = vec![0.0; n];
    for i in 0..n {
        let mut x: Vec<f64> = ((i + 1)..m).map(|r| a.get(r, i)).collect();
        let (beta, tq) = dlarfg(m - i, a.get(i, i), &mut x);
        for (k, r) in ((i + 1)..m).enumerate() {
            a.set(r, i, x[k]);
        }
        tauq[i] = tq;
        d[i] = beta;
        a.set(i, i, 1.0);
        if i + 1 < n {
            let v: Vec<f64> = (i..m).map(|r| a.get(r, i)).collect();
            dlarf_left(m - i, n - i - 1, &v, tq, a, i, i + 1);
        }
        a.set(i, i, d[i]);
        if i + 1 < n {
            let mut x: Vec<f64> = ((i + 2)..n).map(|c| a.get(i, c)).collect();
            let (beta, tp) = dlarfg(n - i - 1, a.get(i, i + 1), &mut x);
            for (k, c) in ((i + 2)..n).enumerate() {
                a.set(i, c, x[k]);
            }
            taup[i] = tp;
            e[i] = beta;
            a.set(i, i + 1, 1.0);
            let v: Vec<f64> = ((i + 1)..n).map(|c| a.get(i, c)).collect();
            dlarf_right(m - i - 1, n - i - 1, &v, tp, a, i + 1, i + 1);
            a.set(i, i + 1, e[i]);
        } else {
            taup[i] = 0.0;
        }
    }
    (d, e, tauq, taup)
}

fn dlas2(f: f64, g: f64, h: f64) -> (f64, f64) {
    let fa = f.abs();
    let ga = g.abs();
    let ha = h.abs();
    let fhmn = fmin(fa, ha);
    let fhmx = fmax(fa, ha);
    if fhmn == 0.0 {
        let ssmax = if fhmx == 0.0 {
            ga
        } else {
            let q = fmin(fhmx, ga) / fmax(fhmx, ga);
            fmax(fhmx, ga) * (1.0 + q * q).sqrt()
        };
        return (0.0, ssmax);
    }
    if ga < fhmx {
        let as_ = 1.0 + fhmn / fhmx;
        let at = (fhmx - fhmn) / fhmx;
        let q = ga / fhmx;
        let au = q * q;
        let c = 2.0 / ((as_ * as_ + au).sqrt() + (at * at + au).sqrt());
        (fhmn * c, fhmx / c)
    } else {
        let au = fhmx / ga;
        if au == 0.0 {
            ((fhmn * fhmx) / ga, ga)
        } else {
            let as_ = 1.0 + fhmn / fhmx;
            let at = (fhmx - fhmn) / fhmx;
            let p = as_ * au;
            let q = at * au;
            let c = 1.0 / ((1.0 + p * p).sqrt() + (1.0 + q * q).sqrt());
            let mut ssmin = (fhmn * c) * au;
            ssmin += ssmin;
            (ssmin, ga / (c + c))
        }
    }
}

/// `dlasv2(f, g, h)`: `(ssmin, ssmax, snr, csr, snl, csl)`.
fn dlasv2(f: f64, g: f64, h: f64) -> (f64, f64, f64, f64, f64, f64) {
    let mut ft = f;
    let mut fa = ft.abs();
    let mut ht = h;
    let mut ha = h.abs();
    let mut pmax = 1;
    let swap = ha > fa;
    if swap {
        pmax = 3;
        std::mem::swap(&mut ft, &mut ht);
        std::mem::swap(&mut fa, &mut ha);
    }
    let gt = g;
    let ga = gt.abs();
    let (mut ssmin, mut ssmax, clt, crt, slt, srt);
    if ga == 0.0 {
        ssmin = ha;
        ssmax = fa;
        clt = 1.0;
        crt = 1.0;
        slt = 0.0;
        srt = 0.0;
    } else {
        let mut gasmal = true;
        let mut values = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        if ga > fa {
            pmax = 2;
            if fa / ga < EPS_E {
                gasmal = false;
                let smax = ga;
                let smin = if ha > 1.0 { fa / (ga / ha) } else { (fa / ga) * ha };
                values = (smin, smax, 1.0, ft / gt, ht / gt, 1.0);
            }
        }
        if gasmal {
            let dd = fa - ha;
            let mut l = if dd == fa { 1.0 } else { dd / fa };
            let m = gt / ft;
            let mut t = 2.0 - l;
            let mm = m * m;
            let tt = t * t;
            let s = (tt + mm).sqrt();
            let r = if l == 0.0 { m.abs() } else { (l * l + mm).sqrt() };
            let a = 0.5 * (s + r);
            let smin = ha / a;
            let smax = fa * a;
            if mm == 0.0 {
                t = if l == 0.0 { sign(2.0, ft) * sign(1.0, gt) } else { gt / sign(dd, ft) + m / t };
            } else {
                t = (m / (s + t) + m / (r + l)) * (1.0 + a);
            }
            l = (t * t + 4.0).sqrt();
            let crt_ = 2.0 / l;
            let srt_ = t / l;
            let clt_ = (crt_ + srt_ * m) / a;
            let slt_ = (ht / ft) * srt_ / a;
            values = (smin, smax, clt_, crt_, slt_, srt_);
        }
        ssmin = values.0;
        ssmax = values.1;
        clt = values.2;
        crt = values.3;
        slt = values.4;
        srt = values.5;
    }
    let (csl, snl, csr, snr) = if swap { (srt, crt, slt, clt) } else { (clt, slt, crt, srt) };
    let tsign = match pmax {
        1 => sign(1.0, csr) * sign(1.0, csl) * sign(1.0, f),
        2 => sign(1.0, snr) * sign(1.0, csl) * sign(1.0, g),
        _ => sign(1.0, snr) * sign(1.0, snl) * sign(1.0, h),
    };
    ssmax = sign(ssmax, tsign);
    ssmin = sign(ssmin, tsign * sign(1.0, f) * sign(1.0, h));
    (ssmin, ssmax, snr, csr, snl, csl)
}

fn rotate_rows(vt: &mut Mat, r1: usize, r2: usize, cols: usize, c: f64, s: f64) {
    for j in 0..cols {
        let (x, y) = blas::drot_pair(c, s, vt.get(r1, j), vt.get(r2, j));
        vt.set(r1, j, x);
        vt.set(r2, j, y);
    }
}

fn rotate_cols(u: &mut Mat, c1: usize, c2: usize, rows: usize, c: f64, s: f64) {
    for i in 0..rows {
        let (x, y) = blas::drot_pair(c, s, u.get(i, c1), u.get(i, c2));
        u.set(i, c1, x);
        u.set(i, c2, y);
    }
}

fn swap_rows(a: &mut Mat, r1: usize, r2: usize) {
    for j in 0..a.cols {
        let x = a.get(r1, j);
        let y = a.get(r2, j);
        a.set(r1, j, y);
        a.set(r2, j, x);
    }
}

fn swap_cols(a: &mut Mat, c1: usize, c2: usize) {
    for i in 0..a.rows {
        let x = a.get(i, c1);
        let y = a.get(i, c2);
        a.set(i, c1, y);
        a.set(i, c2, x);
    }
}

/// `dbdsqr('U', n, n, n, 0, ...)` on a bidiagonal `(d, e)`.
fn dbdsqr_upper(n: usize, d: &mut [f64], e: &mut [f64], vt: &mut Mat, u: &mut Mat) -> Result<(), LinAlgError> {
    if n > 1 {
        let eps = EPS_E;
        let unfl = SAFMIN;
        let tolmul = fmax(10.0, fmin(100.0, super::num::pow(eps, -0.125)));
        let tol = tolmul * eps;
        let mut smax = 0.0;
        for &v in d.iter() {
            smax = fmax(smax, v.abs());
        }
        for &v in e.iter().take(n - 1) {
            smax = fmax(smax, v.abs());
        }
        let mut sminoa = d[0].abs();
        if sminoa != 0.0 {
            let mut mu = sminoa;
            for i in 2..=n {
                mu = d[i - 1].abs() * (mu / (mu + e[i - 2].abs()));
                sminoa = fmin(sminoa, mu);
                if sminoa == 0.0 {
                    break;
                }
            }
        }
        sminoa /= (n as f64).sqrt();
        let nf = n as f64;
        let thresh = fmax(tol * sminoa, 6.0 * (nf * (nf * unfl)));
        let maxitdivn = 6 * n;
        let mut iterdivn = 0;
        let mut iter: isize = -1;
        let mut oldll: isize = -1;
        let mut oldm: isize = -1;
        let mut idir = 0;
        let mut m = n;
        let mut w1 = vec![0.0; n];
        let mut w2 = vec![0.0; n];
        let mut w3 = vec![0.0; n];
        let mut w4 = vec![0.0; n];
        'main: loop {
            if m <= 1 {
                break;
            }
            if iter >= n as isize {
                iter -= n as isize;
                iterdivn += 1;
                if iterdivn >= maxitdivn {
                    return Err(LinAlgError::NoConvergence("SVD did not converge"));
                }
            }
            smax = d[m - 1].abs();
            let mut split: Option<usize> = None;
            for lll in 1..m {
                let llc = m - lll;
                let abss = d[llc - 1].abs();
                let abse = e[llc - 1].abs();
                if abse <= thresh {
                    split = Some(llc);
                    break;
                }
                smax = fmax(fmax(smax, abss), abse);
            }
            let mut ll = match split {
                Some(llc) => {
                    e[llc - 1] = 0.0;
                    if llc == m - 1 {
                        m -= 1;
                        continue 'main;
                    }
                    llc
                }
                None => 0,
            };
            ll += 1;
            if ll == m - 1 {
                let (sigmn, sigmx, sinr, cosr, sinl, cosl) = dlasv2(d[m - 2], e[m - 2], d[m - 1]);
                d[m - 2] = sigmx;
                e[m - 2] = 0.0;
                d[m - 1] = sigmn;
                rotate_rows(vt, m - 2, m - 1, n, cosr, sinr);
                rotate_cols(u, m - 2, m - 1, n, cosl, sinl);
                m -= 2;
                continue;
            }
            if ll as isize > oldm || (m as isize) < oldll {
                idir = if d[ll - 1].abs() >= d[m - 1].abs() { 1 } else { 2 };
            }
            let mut smin: f64;
            if idir == 1 {
                if e[m - 2].abs() <= tol.abs() * d[m - 1].abs() {
                    e[m - 2] = 0.0;
                    continue;
                }
                let mut mu = d[ll - 1].abs();
                smin = mu;
                for lll in ll..m {
                    if e[lll - 1].abs() <= tol * mu {
                        e[lll - 1] = 0.0;
                        continue 'main;
                    }
                    mu = d[lll].abs() * (mu / (mu + e[lll - 1].abs()));
                    smin = fmin(smin, mu);
                }
            } else {
                if e[ll - 1].abs() <= tol.abs() * d[ll - 1].abs() {
                    e[ll - 1] = 0.0;
                    continue;
                }
                let mut mu = d[m - 1].abs();
                smin = mu;
                let mut lll = m - 1;
                while lll >= ll {
                    if e[lll - 1].abs() <= tol * mu {
                        e[lll - 1] = 0.0;
                        continue 'main;
                    }
                    mu = d[lll - 1].abs() * (mu / (mu + e[lll - 1].abs()));
                    smin = fmin(smin, mu);
                    lll -= 1;
                }
            }
            oldll = ll as isize;
            oldm = m as isize;
            let mut shift;
            if nf * tol * (smin / smax) <= fmax(eps, 0.01 * tol) {
                shift = 0.0;
            } else {
                let sll;
                if idir == 1 {
                    sll = d[ll - 1].abs();
                    shift = dlas2(d[m - 2], e[m - 2], d[m - 1]).0;
                } else {
                    sll = d[m - 1].abs();
                    shift = dlas2(d[ll - 1], e[ll - 1], d[ll]).0;
                }
                if sll > 0.0 {
                    let q = shift / sll;
                    if q * q < eps {
                        shift = 0.0;
                    }
                }
            }
            iter += (m - ll) as isize;
            let count = m - ll + 1;
            if shift == 0.0 {
                if idir == 1 {
                    let mut cs = 1.0;
                    let mut oldcs = 1.0;
                    let mut oldsn = 0.0;
                    for i in ll..m {
                        let (c, s, r) = dlartg(d[i - 1] * cs, e[i - 1]);
                        cs = c;
                        let sn = s;
                        if i > ll {
                            e[i - 2] = oldsn * r;
                        }
                        let (c2, s2, r2) = dlartg(oldcs * r, d[i] * sn);
                        oldcs = c2;
                        oldsn = s2;
                        d[i - 1] = r2;
                        w1[i - ll] = cs;
                        w2[i - ll] = sn;
                        w3[i - ll] = oldcs;
                        w4[i - ll] = oldsn;
                    }
                    let h = d[m - 1] * cs;
                    d[m - 1] = h * oldcs;
                    e[m - 2] = h * oldsn;
                    dlasr_left_forward(count, n, &w1, &w2, vt, ll - 1);
                    dlasr_right_forward(n, count, &w3, &w4, u, ll - 1);
                    if e[m - 2].abs() <= thresh {
                        e[m - 2] = 0.0;
                    }
                } else {
                    let mut cs = 1.0;
                    let mut oldcs = 1.0;
                    let mut oldsn = 0.0;
                    let mut i = m;
                    while i > ll {
                        let (c, s, r) = dlartg(d[i - 1] * cs, e[i - 2]);
                        cs = c;
                        let sn = s;
                        if i < m {
                            e[i - 1] = oldsn * r;
                        }
                        let (c2, s2, r2) = dlartg(oldcs * r, d[i - 2] * sn);
                        oldcs = c2;
                        oldsn = s2;
                        d[i - 1] = r2;
                        w1[i - ll - 1] = cs;
                        w2[i - ll - 1] = -sn;
                        w3[i - ll - 1] = oldcs;
                        w4[i - ll - 1] = -oldsn;
                        i -= 1;
                    }
                    let h = d[ll - 1] * cs;
                    d[ll - 1] = h * oldcs;
                    e[ll - 1] = h * oldsn;
                    dlasr_left_backward(count, n, &w3, &w4, vt, ll - 1);
                    dlasr_right_backward(n, count, &w1, &w2, u, ll - 1);
                    if e[ll - 1].abs() <= thresh {
                        e[ll - 1] = 0.0;
                    }
                }
            } else if idir == 1 {
                let mut f = (d[ll - 1].abs() - shift) * (sign(1.0, d[ll - 1]) + shift / d[ll - 1]);
                let mut g = e[ll - 1];
                for i in ll..m {
                    let (cosr, sinr, r) = dlartg(f, g);
                    if i > ll {
                        e[i - 2] = r;
                    }
                    f = cosr * d[i - 1] + sinr * e[i - 1];
                    e[i - 1] = cosr * e[i - 1] - sinr * d[i - 1];
                    g = sinr * d[i];
                    d[i] *= cosr;
                    let (cosl, sinl, r) = dlartg(f, g);
                    d[i - 1] = r;
                    f = cosl * e[i - 1] + sinl * d[i];
                    d[i] = cosl * d[i] - sinl * e[i - 1];
                    if i < m - 1 {
                        g = sinl * e[i];
                        e[i] *= cosl;
                    }
                    w1[i - ll] = cosr;
                    w2[i - ll] = sinr;
                    w3[i - ll] = cosl;
                    w4[i - ll] = sinl;
                }
                e[m - 2] = f;
                dlasr_left_forward(count, n, &w1, &w2, vt, ll - 1);
                dlasr_right_forward(n, count, &w3, &w4, u, ll - 1);
                if e[m - 2].abs() <= thresh {
                    e[m - 2] = 0.0;
                }
            } else {
                let mut f = (d[m - 1].abs() - shift) * (sign(1.0, d[m - 1]) + shift / d[m - 1]);
                let mut g = e[m - 2];
                let mut i = m;
                while i > ll {
                    let (cosr, sinr, r) = dlartg(f, g);
                    if i < m {
                        e[i - 1] = r;
                    }
                    f = cosr * d[i - 1] + sinr * e[i - 2];
                    e[i - 2] = cosr * e[i - 2] - sinr * d[i - 1];
                    g = sinr * d[i - 2];
                    d[i - 2] *= cosr;
                    let (cosl, sinl, r) = dlartg(f, g);
                    d[i - 1] = r;
                    f = cosl * e[i - 2] + sinl * d[i - 2];
                    d[i - 2] = cosl * d[i - 2] - sinl * e[i - 2];
                    if i > ll + 1 {
                        g = sinl * e[i - 3];
                        e[i - 3] *= cosl;
                    }
                    w1[i - ll - 1] = cosr;
                    w2[i - ll - 1] = -sinr;
                    w3[i - ll - 1] = cosl;
                    w4[i - ll - 1] = -sinl;
                    i -= 1;
                }
                e[ll - 1] = f;
                if e[ll - 1].abs() <= thresh {
                    e[ll - 1] = 0.0;
                }
                dlasr_left_backward(count, n, &w3, &w4, vt, ll - 1);
                dlasr_right_backward(n, count, &w1, &w2, u, ll - 1);
            }
        }
    }
    for i in 0..n {
        if d[i] < 0.0 {
            d[i] = -d[i];
            for j in 0..n {
                vt.set(i, j, vt.get(i, j) * -1.0);
            }
        }
    }
    for i in 1..n {
        let mut isub = 1;
        let mut smin = d[0];
        for j in 2..=(n + 1 - i) {
            if d[j - 1] <= smin {
                isub = j;
                smin = d[j - 1];
            }
        }
        if isub != n + 1 - i {
            d[isub - 1] = d[n - i];
            d[n - i] = smin;
            swap_rows(vt, isub - 1, n - i);
            swap_cols(u, isub - 1, n - i);
        }
    }
    Ok(())
}

fn dlasdq_upper(n: usize, d: &mut [f64], e: &mut [f64], vt: &mut Mat, u: &mut Mat) -> Result<(), LinAlgError> {
    dbdsqr_upper(n, d, e, vt, u)?;
    for i in 1..=n {
        let mut isub = i;
        let mut smin = d[i - 1];
        for j in (i + 1)..=n {
            if d[j - 1] < smin {
                isub = j;
                smin = d[j - 1];
            }
        }
        if isub != i {
            d[isub - 1] = d[i - 1];
            d[i - 1] = smin;
            swap_rows(vt, isub - 1, i - 1);
            swap_cols(u, isub - 1, i - 1);
        }
    }
    Ok(())
}

/// `dormbr('P', 'R', 'T', ...)` for a square bidiagonalisation:
/// `dorml2('R', 'N', n, n-1, n-1, A(1,2), lda, taup, VT(1,2), ldvt)`.
fn dorml2_right_notrans(n: usize, a: &mut Mat, taup: &[f64], vt: &mut Mat) {
    let rows = n;
    let cols = n - 1;
    let k = n - 1;
    for i in (1..=k).rev() {
        let ni = cols - i + 1;
        let aii = a.get(i - 1, i);
        a.set(i - 1, i, 1.0);
        let v: Vec<f64> = (0..ni).map(|c| a.get(i - 1, i + c)).collect();
        dlarf_right(rows, ni, &v, taup[i - 1], vt, 0, i);
        a.set(i - 1, i, aii);
    }
}

/// numba's `np.linalg.svd(a, full_matrices=False)` of a row-major square
/// `n x n` matrix (`dgesdd`, `JOBZ='S'`): singular values descending and
/// `VT` row-major.
pub fn svd_small(n: usize, matrix: &[f64]) -> Result<(Vec<f64>, Vec<f64>), LinAlgError> {
    if matrix.iter().take(n * n).any(|v| !v.is_finite()) {
        return Err(LinAlgError::NonFinite);
    }
    let mut a = Mat::from_row_major(n, n, matrix);
    let eps = EPS_P;
    let smlnum = SAFMIN.sqrt() / eps;
    let bignum = 1.0 / smlnum;
    let mut anrm = 0.0;
    for j in 0..n {
        for i in 0..n {
            let t = a.get(i, j).abs();
            if anrm < t || t.is_nan() {
                anrm = t;
            }
        }
    }
    let mut iscl = false;
    if anrm > 0.0 && anrm < smlnum {
        iscl = true;
        dlascl(anrm, smlnum, &mut a.data);
    } else if anrm > bignum {
        iscl = true;
        dlascl(anrm, bignum, &mut a.data);
    }
    let mut s;
    let vt_rows;
    if n == 1 {
        s = vec![a.get(0, 0).abs()];
        vt_rows = vec![1.0];
    } else {
        let (mut d, mut e, tauq, taup) = dgebd2(n, n, &mut a);
        let mut u = Mat::identity(n);
        let mut vt = Mat::identity(n);
        dlasdq_upper(n, &mut d, &mut e, &mut vt, &mut u)?;
        for ii in 2..=n {
            let i = ii - 1;
            let mut kk = i;
            let mut p = d[i - 1];
            for j in ii..=n {
                if d[j - 1] > p {
                    kk = j;
                    p = d[j - 1];
                }
            }
            if kk != i {
                d[kk - 1] = d[i - 1];
                d[i - 1] = p;
                swap_cols(&mut u, i - 1, kk - 1);
                swap_rows(&mut vt, i - 1, kk - 1);
            }
        }
        dorm2r_left_notrans(n, n, n, &mut a, 0, 0, &tauq, &mut u, 0, 0);
        dorml2_right_notrans(n, &mut a, &taup, &mut vt);
        s = d;
        vt_rows = vt.to_row_major();
    }
    if iscl {
        if anrm > bignum {
            dlascl(bignum, anrm, &mut s);
        }
        if anrm < smlnum {
            dlascl(smlnum, anrm, &mut s);
        }
    }
    Ok((s, vt_rows))
}
