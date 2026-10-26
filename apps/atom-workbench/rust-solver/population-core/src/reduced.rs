//! Reduced model of a trial space.
//!
//! Galerkin projection of `E x' = F x` on the orthonormal basis `Q`, with the
//! equation along the physical-trace direction replaced by exact trace
//! conservation. With the trace functional `t` of the columns, the reflector
//! `R = I - 2 v v^T` maps `t` to a multiple of `e_0`; in rotated coordinates
//! `c = R a` the first row becomes `c_0' = 0`. The returned generator is
//! unrotated: `a' = G a`, `a(0) = Q^T x0`.

use super::*;
use super::trial::TrialSpace;

pub(crate) struct ReducedModel {
    pub m: usize,
    pub generator: Vec<f64>,
    pub initial: Vec<f64>,
}

fn norm(values: &[f64]) -> f64 {
    values.iter().map(|x| x * x).sum::<f64>().sqrt()
}

impl ReducedModel {
    pub fn build(space: &TrialSpace) -> Result<Self, String> {
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
        let generator = rotate(&rotated)?;
        if generator.iter().any(|x| !x.is_finite()) {
            return Err("Nonfinite reduced generator".into());
        }
        Ok(Self { m, generator, initial: space.initial_coefficients().to_vec() })
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
        let (propagator, _) = crate::dense_expm::exponential(lapack, &self.generator, m, step)?;
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
