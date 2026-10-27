//! Reduced model of a trial space.
//!
//! Galerkin projection of `E x' = F x` on the orthonormal basis `Q`, with the
//! equation along the physical-trace direction replaced by exact trace
//! conservation. With the trace functional `t` of the columns, the reflector
//! `R = I - 2 v v^T` maps `t` to a multiple of `e_0`; in rotated coordinates
//! `c = R a` the first row becomes `c_0' = 0`. The returned generator is
//! unrotated: `a' = G a`, `a(0) = Q^T x0`.
//!
//! # Growing Ritz modes
//!
//! The Lindblad semigroup is a contraction in the trace norm, not in the
//! Euclidean metric of the (gauged) trial coordinates, so a Galerkin
//! projection can have eigenvalues with a positive real part. Over a short
//! horizon they are harmless; over a long one `exp(Re(beta) T)` makes the
//! candidate, and its error estimate, explode, and more columns need not
//! remove them. Every reduced eigenvalue whose growth over the horizon is
//! not negligible (`Re(beta) T > GROWTH`) is reflected to `-Re(beta)`. The
//! candidate is a different trajectory then, still `a' = G a` on the same
//! space, and its error is estimated exactly like any other (the residual
//! `F Q - E Q G` uses the returned `G`): nothing is assumed, a poor choice
//! only shows up as a larger estimate.

use super::*;
use super::trial::TrialSpace;

pub(crate) struct ReducedModel {
    pub m: usize,
    pub generator: Vec<f64>,
    pub initial: Vec<f64>,
    /// Reduced eigenvalues reflected into the left half plane.
    pub reflected: usize,
}

/// Growth `Re(beta) T` over the horizon below which a reduced eigenvalue is
/// left as it is (a relative growth of the candidate below 1e-9).
const GROWTH: f64 = 1e-9;

fn norm(values: &[f64]) -> f64 {
    values.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Reflects the eigenvalues of the real `m x m` generator with `Re(beta) T >
/// GROWTH` to `-Re(beta)`: with `G V = V L` in real block form (2 x 2 blocks
/// `[a b; -b a]` for complex pairs), `G' = G - 2 V_u diag(a_u) (V^-1)_u` over
/// the growing columns `u`. The update is kept only when `G' V = V L'` holds
/// to roundoff (a defective or ill-conditioned eigenbasis leaves `G`).
/// Returns the number of reflected eigenvalues.
fn reflect_growing(lapack: &Lapack, generator: &mut [f64], m: usize, horizon: f64) -> Result<usize, String> {
    let (real, imaginary, vectors) = lapack.eigen(generator, m)?;
    let growing: Vec<usize> = (0..m).filter(|&j| real[j] * horizon > GROWTH).collect();
    if growing.is_empty() {
        return Ok(0);
    }
    if vectors.iter().any(|x| !x.is_finite()) {
        return Ok(0);
    }
    let (lu, pivots) = lapack.factor(&vectors, m)?;
    let mut identity = vec![0.; m * m];
    for j in 0..m {
        identity[j + j * m] = 1.;
    }
    let inverse = lapack.solve_matrix_factored(&lu, &pivots, &identity, m)?;
    let mut update = vec![0.; m * m];
    for &u in &growing {
        // Column u of V times row u of V^-1, weighted by -2 a_u. A complex
        // pair occupies columns (j, j + 1) with equal real parts, so each of
        // its two real columns is reflected with the same a.
        let a = real[u];
        for c in 0..m {
            let row = inverse[u + c * m];
            if row == 0. {
                continue;
            }
            for r in 0..m {
                update[r + c * m] -= 2. * a * vectors[r + u * m] * row;
            }
        }
    }
    let mut reflected: Vec<f64> = generator.iter().zip(&update).map(|(g, d)| g + d).collect();
    if reflected.iter().any(|x| !x.is_finite()) {
        return Ok(0);
    }
    // Check G' V = V L' with L' the reflected real block form.
    let scale = generator.iter().fold(0_f64, |s, x| s.max(x.abs())).max(f64::MIN_POSITIVE);
    let mut product = vec![0.; m * m];
    lapack.gemm(false, false, m, m, m, 1., &reflected, m, &vectors, m, 0., &mut product, m)?;
    let mut j = 0;
    let mut defect = 0_f64;
    let column_norm = |c: usize| vectors[c * m..(c + 1) * m].iter().fold(0_f64, |s, x| s.hypot(*x)).max(f64::MIN_POSITIVE);
    while j < m {
        let a = if real[j] * horizon > GROWTH { -real[j] } else { real[j] };
        if imaginary[j] != 0. && j + 1 < m {
            let b = imaginary[j];
            // G v_r = a v_r - b v_i, G v_i = b v_r + a v_i for v = v_r + i v_i.
            for r in 0..m {
                let (vr, vi) = (vectors[r + j * m], vectors[r + (j + 1) * m]);
                defect = defect.max((product[r + j * m] - (a * vr - b * vi)).abs() / column_norm(j).hypot(column_norm(j + 1)));
                defect = defect.max((product[r + (j + 1) * m] - (b * vr + a * vi)).abs() / column_norm(j).hypot(column_norm(j + 1)));
            }
            j += 2;
        } else {
            for r in 0..m {
                defect = defect.max((product[r + j * m] - a * vectors[r + j * m]).abs() / column_norm(j));
            }
            j += 1;
        }
    }
    if defect > 1e-8 * scale {
        return Ok(0);
    }
    generator.swap_with_slice(&mut reflected);
    Ok(growing.len())
}

impl ReducedModel {
    /// The reduced model of `space` on the horizon `horizon` (us); zero leaves
    /// growing Ritz modes as they are.
    pub fn build(space: &TrialSpace, horizon: f64) -> Result<Self, String> {
        let lapack = space.lapack();
        let m = space.columns();
        let (trace_q, trace_e, trace_f) = space.traces();
        let length = norm(trace_q);
        if !(length > 1e-8) {
            return Err("The trial space has no resolved physical-trace direction".into());
        }
        let mut v: Vec<f64> = trace_q.iter().map(|x| x / length).collect();
        v[0] += if v[0] >= 0. { 1. } else { -1. };
        let v_length = norm(&v);
        for x in &mut v {
            *x /= v_length;
        }
        let mut reflector = vec![0.; m * m];
        for j in 0..m {
            for i in 0..m {
                reflector[i + j * m] = -2. * v[i] * v[j];
            }
            reflector[j + j * m] += 1.;
        }
        let rotate = |matrix: &[f64]| -> Result<Vec<f64>, String> {
            let mut half = vec![0.; m * m];
            let mut out = vec![0.; m * m];
            lapack.gemm(false, false, m, m, m, 1., &reflector, m, matrix, m, 0., &mut half, m)?;
            lapack.gemm(false, false, m, m, m, 1., &half, m, &reflector, m, 0., &mut out, m)?;
            Ok(out)
        };
        let mut mass = rotate(space.mass())?;
        let mut pencil = rotate(space.pencil())?;
        // E is trace preserving and F maps into traceless fields, so in
        // rotated coordinates t_E R = gamma e_0 and t_F R = 0 up to roundoff.
        let gamma: f64 = trace_q.iter().zip(&reflector[..m]).map(|(a, b)| a * b).sum();
        let scale = space.mass().iter().chain(space.pencil()).map(|x| x.abs()).fold(1_f64, f64::max);
        for j in 0..m {
            let column = &reflector[j * m..(j + 1) * m];
            let e: f64 = trace_e.iter().zip(column).map(|(a, b)| a * b).sum::<f64>() / gamma;
            let f: f64 = trace_f.iter().zip(column).map(|(a, b)| a * b).sum::<f64>() / gamma;
            let expected = if j == 0 { 1. } else { 0. };
            if (e - expected).abs() > 1e-9 || f.abs() > 1e-9 * scale {
                return Err("Trial images violate the exact trace identities".into());
            }
        }
        for j in 0..m {
            mass[j * m] = if j == 0 { 1. } else { 0. };
            pencil[j * m] = 0.;
        }
        let (lu, pivots) = lapack.factor(&mass, m)?;
        let rotated = lapack.solve_matrix_factored(&lu, &pivots, &pencil, m)?;
        let mut generator = rotate(&rotated)?;
        if generator.iter().any(|x| !x.is_finite()) {
            return Err("Nonfinite reduced generator".into());
        }
        let reflected = if horizon > 0. { reflect_growing(lapack, &mut generator, m, horizon)? } else { 0 };
        Ok(Self { m, generator, initial: space.initial_coefficients().to_vec(), reflected })
    }

    /// Reduced coefficients at uniformly spaced `times` starting at zero,
    /// column-major `[m, times]`.
    pub fn trajectory(&self, lapack: &Lapack, times: &[f64]) -> Result<Vec<f64>, String> {
        let m = self.m;
        let count = times.len();
        if count == 0 || times[0] != 0. {
            return Err("Output times must start at zero".into());
        }
        let mut out = vec![0.; m * count];
        out[..m].copy_from_slice(&self.initial);
        if count == 1 {
            return Ok(out);
        }
        let step = times[count - 1] / (count - 1) as f64;
        let tolerance = 1e-12 * times[count - 1].max(1.);
        if times.iter().enumerate().any(|(i, &t)| (t - step * i as f64).abs() > tolerance) {
            return Err("Output times must be uniformly spaced".into());
        }
        let propagator = crate::dense_expm::exponential(lapack, &self.generator, m, step)?;
        for i in 1..count {
            let (done, rest) = out.split_at_mut(i * m);
            lapack.gemm(false, false, m, 1, m, 1., &propagator, m, &done[(i - 1) * m..], m, 0., &mut rest[..m], m)?;
        }
        if out.iter().any(|x| !x.is_finite()) {
            return Err("Nonfinite reduced trajectory".into());
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn growing_eigenvalues_are_reflected_and_the_rest_kept() {
        let lapack = Lapack::load().unwrap();
        // V diag-block(0.5, [-1 2; -2 -1]) V^-1 with a non-orthogonal V.
        let v = [1., 0.2, -0.3, 0.4, 1., 0.1, 0.25, -0.5, 1.];
        let block = [0.5, 0., 0., 0., -1., -2., 0., 2., -1.];
        let mut product = vec![0.; 9];
        lapack.gemm(false, false, 3, 3, 3, 1., &v, 3, &block, 3, 0., &mut product, 3).unwrap();
        let (lu, pivots) = lapack.factor(&v, 3).unwrap();
        let identity = [1., 0., 0., 0., 1., 0., 0., 0., 1.];
        let inverse = lapack.solve_matrix_factored(&lu, &pivots, &identity, 3).unwrap();
        let mut generator = vec![0.; 9];
        lapack.gemm(false, false, 3, 3, 3, 1., &product, 3, &inverse, 3, 0., &mut generator, 3).unwrap();
        let original = generator.clone();
        // Negligible growth over a tiny horizon: unchanged.
        assert_eq!(reflect_growing(&lapack, &mut generator, 3, 1e-12).unwrap(), 0);
        assert_eq!(generator, original);
        assert_eq!(reflect_growing(&lapack, &mut generator, 3, 10.).unwrap(), 1);
        let (real, imaginary, _) = lapack.eigen(&generator, 3).unwrap();
        let mut values: Vec<(f64, f64)> = real.into_iter().zip(imaginary).collect();
        values.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
        let expected = [(-1., -2.), (-1., 2.), (-0.5, 0.)];
        for (got, want) in values.iter().zip(expected) {
            assert!((got.0 - want.0).abs() < 1e-12 && (got.1 - want.1).abs() < 1e-12, "{values:?}");
        }
    }
}
