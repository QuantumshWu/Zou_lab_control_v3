//! Which parameters a fit's data determine, and their covariance: the
//! information matrix's pseudo-inverse on the identified subspace, read
//! from the matrix equilibrated by its diagonal.
//!
//! A rank test on the raw matrix compares eigen- or singular values across
//! parameters of different units.  A parameter the data constrain only
//! weakly in the units it is written in (a decay time far beyond the
//! window, in microseconds) has a tiny column there; the test calls the
//! matrix singular and every parameter's error is lost, the well-determined
//! ones included.  Equilibrated -- `D A D` with `D = diag(A)^-1/2`, or for a
//! factor `F` with `A = F^T F` its columns scaled to unit norm -- the test is
//! exactly invariant under a change of any parameter's unit, and a value
//! below the tolerance is a genuine null direction: a combination of
//! parameters the data do not determine at all.
//!
//! A parameter whose unit vector has a component in that null space is
//! undetermined: its standard error and its covariance row and column are
//! NaN.  Every other parameter's covariance is the pseudo-inverse's -- the
//! variance of a linear functional the data do determine.  A parameter with
//! no influence (a zero or non-finite diagonal) is itself a null direction.

use super::blas;
use super::linalg::{self, LinAlgError};
use super::num::{EPSILON, TINY};

/// An information matrix's identified part.
#[derive(Debug, Clone)]
pub struct Identified {
    /// The equilibrated pseudo-inverse, row-major `n x n`.
    scaled: Vec<f64>,
    /// The equilibration: each parameter's root diagonal (1 for one
    /// without influence, whose row and column are zero).
    norms: Vec<f64>,
    /// Per parameter: the data determine it.
    pub defined: Vec<bool>,
}

impl Identified {
    /// The pseudo-inverse in the parameters' own units, times `factor`,
    /// symmetrised.  Finite everywhere: an undetermined parameter's row
    /// still carries the identified functionals' weights on it (what a
    /// sandwich needs); [`Identified::mask`] removes it.
    pub fn pseudo_inverse(&self, factor: f64) -> Vec<f64> {
        let n = self.norms.len();
        let mut computed = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                computed[i * n + j] = self.scaled[i * n + j] * (factor / (self.norms[i] * self.norms[j]));
            }
        }
        let mut result = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                result[i * n + j] = (computed[i * n + j] + computed[j * n + i]) / 2.0;
            }
        }
        result
    }

    /// NaN in every undetermined parameter's row and column.
    pub fn mask(&self, matrix: &mut [f64]) {
        let n = self.norms.len();
        for i in (0..n).filter(|&i| !self.defined[i]) {
            for k in 0..n {
                matrix[i * n + k] = f64::NAN;
                matrix[k * n + i] = f64::NAN;
            }
        }
    }

    /// The covariance: the masked pseudo-inverse times `factor`.
    pub fn covariance(&self, factor: f64) -> Vec<f64> {
        let mut covariance = self.pseudo_inverse(factor);
        self.mask(&mut covariance);
        covariance
    }
}

/// The identified part of the symmetric information matrix `information`
/// (row-major `n x n`): eigenvalues of the equilibrated matrix within
/// `EPSILON * size * largest` of zero are its null space.  `None` where
/// nothing is identified, or the matrix is indefinite beyond that
/// tolerance (not a maximum of the likelihood).
pub fn from_information(n: usize, information: &[f64], size: usize) -> Result<Option<Identified>, LinAlgError> {
    let mut norms = vec![1.0; n];
    let mut live = vec![false; n];
    for i in 0..n {
        let diagonal = information[i * n + i];
        if diagonal < 0.0 {
            return Ok(None);
        }
        let root = diagonal.sqrt();
        if root.is_finite() && root > TINY {
            norms[i] = root;
            live[i] = true;
        }
    }
    let mut normalized = vec![0.0; n * n];
    for i in (0..n).filter(|&i| live[i]) {
        for j in (0..n).filter(|&j| live[j]) {
            normalized[i * n + j] = information[i * n + j] / (norms[i] * norms[j]);
        }
    }
    let mut symmetric = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            symmetric[i * n + j] = (normalized[i * n + j] + normalized[j * n + i]) / 2.0;
        }
    }
    let (values, vectors) = linalg::eigh_lower(n, &symmetric)?;
    let largest = values.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    if !(largest > 0.0) {
        return Ok(None);
    }
    let tolerance = EPSILON * size as f64 * largest;
    if values.iter().any(|&v| v < -tolerance) {
        return Ok(None);
    }
    Ok(identify(norms, &values, &values, |i, k| vectors[i * n + k], tolerance))
}

/// The identified part of `F^T F` for the row-major `n x n` factor `F`
/// (the triangle of a Jacobian's QR): singular values of `F` with unit-norm
/// columns within `EPSILON * size * largest` of zero are its null space.
/// `None` where nothing is identified.
pub fn from_factor(n: usize, factor: &[f64], size: usize) -> Result<Option<Identified>, LinAlgError> {
    let mut norms = vec![1.0; n];
    let mut scaled = vec![0.0; n * n];
    for column in 0..n {
        let values: Vec<f64> = (0..n).map(|row| factor[row * n + column]).collect();
        let norm = blas::dnrm2(&values);
        if norm.is_finite() && norm > TINY {
            norms[column] = norm;
            for row in 0..n {
                scaled[row * n + column] = values[row] / norm;
            }
        }
    }
    let (singular_values, right) = linalg::svd_small(n, &scaled)?;
    let largest = singular_values.first().copied().unwrap_or(0.0);
    if !(largest > 0.0) {
        return Ok(None);
    }
    let tolerance = EPSILON * size as f64 * largest;
    let weights: Vec<f64> = singular_values.iter().map(|s| s * s).collect();
    Ok(identify(norms, &singular_values, &weights, |i, k| right[k * n + i], tolerance))
}

/// The spectral pseudo-inverse of the equilibrated matrix and which
/// parameters it determines.  `magnitudes[k]` is the `k`-th eigen- or
/// singular value the tolerance is read against, `weights[k]` its
/// eigenvalue of the information, `component(i, k)` parameter `i`'s
/// component of its vector.
fn identify(norms: Vec<f64>, magnitudes: &[f64], weights: &[f64], component: impl Fn(usize, usize) -> f64, tolerance: f64) -> Option<Identified> {
    let n = norms.len();
    let (null, range): (Vec<usize>, Vec<usize>) = (0..magnitudes.len()).partition(|&k| magnitudes[k].abs() <= tolerance);
    if range.is_empty() {
        return None;
    }
    // A computed vector is resolved to about `tolerance / gap` (the
    // eigenvector perturbation bound, the gap the smallest identified
    // value): a parameter's null component beyond that is real -- it moves
    // along a direction the data do not see.
    let gap = range.iter().map(|&k| magnitudes[k].abs()).fold(f64::INFINITY, f64::min);
    let resolution = tolerance / gap;
    let defined: Vec<bool> = (0..n).map(|i| null.iter().map(|&k| component(i, k) * component(i, k)).sum::<f64>().sqrt() <= resolution).collect();
    if !defined.iter().any(|&d| d) {
        return None;
    }
    let mut scaled = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut value = 0.0;
            for &k in &range {
                value = (component(i, k) / weights[k]).mul_add(component(j, k), value);
            }
            scaled[i * n + j] = value;
        }
    }
    Some(Identified { scaled, norms, defined })
}
