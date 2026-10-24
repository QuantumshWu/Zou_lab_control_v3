//! Compensated defect of a COMPLETE modal representation of a real generator.
//!
//! D = W (A R - R Lambda) is formed only after the large terms in A R and
//! R Lambda have cancelled in double-double arithmetic. W is the supplied
//! numerical inverse; its residual W R - I is reported separately. No mode,
//! matrix entry, frequency, decay rate, or small defect is discarded.
//!
//! Norms here refer to the caller's coordinate system. They are not physical
//! trace norms unless the caller performs the appropriate Hermitian metric
//! conversion. Roundoff estimates assume IEEE round-to-nearest arithmetic
//! and the BLAS floating-point model; they are not interval certificates.
use super::lapack::{Complex64 as Z, Lapack};
use std::time::Instant;

pub type Sparse = Vec<Vec<(usize, f64)>>;
const U: f64 = f64::EPSILON * 0.5;
fn gamma(n: usize) -> f64 {
    let x = n as f64 * U;
    x / (1. - x)
}
fn abs1(z: Z) -> f64 {
    (z.re.abs() + z.im.abs()) * (1. + 4. * U)
}
fn modulus(z: Z) -> f64 {
    z.re.hypot(z.im) * (1. + 8. * U)
}
fn real_one(entries: &[f64], n: usize) -> f64 {
    entries
        .chunks(n)
        .map(|column| column.iter().sum::<f64>() / (1. - gamma(n + 2)))
        .fold(0., f64::max)
}
fn complex_one(entries: &[Z], n: usize) -> f64 {
    entries
        .chunks(n)
        .map(|column| column.iter().map(|&z| modulus(z)).sum::<f64>() / (1. - gamma(n + 2)))
        .fold(0., f64::max)
}
fn finite(values: &[Z], label: &str) -> Result<(), String> {
    if values
        .iter()
        .any(|z| !z.re.is_finite() || !z.im.is_finite())
    {
        Err(format!("Non-finite {label} in modal defect"))
    } else {
        Ok(())
    }
}

/// Complete right-eigenvector basis without invariant-cluster refinement.
/// The caller MUST retain/verify compute(...).matrix; this basis alone does
/// not certify evolution, and no small eigenvalue is rounded to zero.
pub fn raw_modal_basis(
    lap: &Lapack,
    a: &Sparse,
) -> Result<super::cluster_action::ModalBasis, String> {
    use super::cluster_action::{ModalBasis, ModeOrigin};
    let n = a.len();
    let size = n.checked_mul(n).ok_or("Raw modal dimensions overflow")?;
    if n == 0 {
        return Err("Empty raw modal generator".into());
    }
    let mut dense = vec![0.; size];
    for (j, column) in a.iter().enumerate() {
        for &(i, value) in column {
            if i >= n || !value.is_finite() {
                return Err("Invalid raw modal generator entry".into());
            }
            dense[i + j * n] += value;
        }
    }
    let (wr, wi, real_vectors) = lap.eigen(&dense, n)?;
    if wr
        .iter()
        .chain(&wi)
        .chain(&real_vectors)
        .any(|x| !x.is_finite())
    {
        return Err("Non-finite raw eigensystem".into());
    }
    let real_norm = real_vectors
        .chunks(n)
        .map(|col| col.iter().map(|x| x.abs()).sum::<f64>())
        .fold(0., f64::max);
    let (lu, pivots) = lap.factor(&real_vectors, n)?;
    let real_condition = lap.condition(&lu, n, real_norm)?;
    if !real_condition.is_finite() || real_condition > 1e10 {
        return Err(format!("Raw modal basis is ill-conditioned ({real_condition:e}); retain the refined/general solver"));
    }
    let mut identity = vec![0.; size];
    for i in 0..n {
        identity[i + i * n] = 1.;
    }
    let real_inverse = lap.solve_matrix_factored(&lu, &pivots, &identity, n)?;
    let mut vectors = vec![Z::default(); size];
    let mut inverse = vec![Z::default(); size];
    let mut eigenvalues = vec![Z::default(); n];
    let mut j = 0;
    while j < n {
        if wi[j] > 0. {
            if j + 1 >= n || wi[j + 1] != -wi[j] || wr[j + 1] != wr[j] {
                return Err("Raw eigensystem has an incomplete conjugate pair".into());
            }
            eigenvalues[j] = Z {
                re: wr[j],
                im: wi[j],
            };
            eigenvalues[j + 1] = Z {
                re: wr[j + 1],
                im: wi[j + 1],
            };
            for i in 0..n {
                vectors[i + j * n] = Z {
                    re: real_vectors[i + j * n],
                    im: real_vectors[i + (j + 1) * n],
                };
                vectors[i + (j + 1) * n] = Z {
                    re: real_vectors[i + j * n],
                    im: -real_vectors[i + (j + 1) * n],
                };
                inverse[j + i * n] = Z {
                    re: 0.5 * real_inverse[j + i * n],
                    im: -0.5 * real_inverse[j + 1 + i * n],
                };
                inverse[j + 1 + i * n] = Z {
                    re: 0.5 * real_inverse[j + i * n],
                    im: 0.5 * real_inverse[j + 1 + i * n],
                };
            }
            j += 2;
        } else {
            if wi[j] < 0. {
                return Err("Raw eigensystem has an unpaired negative imaginary mode".into());
            }
            eigenvalues[j] = Z { re: wr[j], im: 0. };
            for i in 0..n {
                vectors[i + j * n].re = real_vectors[i + j * n];
                inverse[j + i * n].re = real_inverse[j + i * n];
            }
            j += 1;
        }
    }
    finite(&vectors, "raw complex basis")?;
    finite(&inverse, "raw complex inverse")?;
    let condition = complex_one(&vectors, n) * complex_one(&inverse, n);
    if !condition.is_finite() || condition > 1e10 {
        return Err(format!(
            "Raw complex modal basis is ill-conditioned ({condition:e})"
        ));
    }
    // OriginalFast denotes unrefined original eigenmodes in the existing
    // representation; all n modes are present regardless of their frequency.
    Ok(ModalBasis {
        dimension: n,
        eigenvalues,
        vectors,
        inverse,
        condition,
        origin: vec![ModeOrigin::OriginalFast; n],
    })
}

#[derive(Clone, Copy, Default)]
struct Double {
    hi: f64,
    lo: f64,
}
impl Double {
    #[inline(always)]
    fn add(&mut self, hi: f64, lo: f64) {
        let sum = self.hi + hi;
        let z = sum - self.hi;
        let error = (self.hi - (sum - z)) + (hi - z);
        let tail = self.lo + lo + error;
        let value = sum + tail;
        self.lo = tail - (value - sum);
        self.hi = value;
    }
    #[inline(always)]
    fn product(&mut self, a: f64, b: f64) {
        let p = a * b;
        self.add(p, a.mul_add(b, -p));
    }
    #[inline(always)]
    fn value(self) -> f64 {
        self.hi + self.lo
    }
}
#[derive(Clone, Copy, Default)]
struct Wide {
    re: Double,
    im: Double,
}
impl Wide {
    #[inline(always)]
    fn product(&mut self, a: Z, b: Z) {
        self.re.product(a.re, b.re);
        self.re.product(-a.im, b.im);
        self.im.product(a.re, b.im);
        self.im.product(a.im, b.re);
    }
    #[inline(always)]
    fn value(self) -> Z {
        Z {
            re: self.re.value(),
            im: self.im.value(),
        }
    }
}

#[inline(always)]
fn residual_kernel(a: &Sparse, vectors: &[Z], eigenvalues: &[Z]) -> (Vec<Z>, Vec<f64>) {
    let n = a.len();
    let mut residual = vec![Wide::default(); n * n];
    let mut magnitude = vec![0.; n * n];
    let mut counts = vec![0; n];
    for column in a {
        for &(i, _) in column {
            counts[i] += 1;
        }
    }
    for j in 0..n {
        for (k, entries) in a.iter().enumerate() {
            let v = vectors[k + j * n];
            for &(i, value) in entries {
                residual[i + j * n].re.product(value, v.re);
                residual[i + j * n].im.product(value, v.im);
                magnitude[i + j * n] += value.abs() * abs1(v);
            }
        }
        let lambda = eigenvalues[j];
        for i in 0..n {
            let v = vectors[i + j * n];
            residual[i + j * n].product(
                Z {
                    re: -lambda.re,
                    im: -lambda.im,
                },
                v,
            );
            magnitude[i + j * n] += abs1(lambda) * abs1(v);
        }
    }
    let mut result = Vec::with_capacity(n * n);
    let mut error = Vec::with_capacity(n * n);
    for j in 0..n {
        for i in 0..n {
            let k = i + j * n;
            let value = residual[k].value();
            result.push(value);
            let count = counts[i] + 8;
            let underflow = 32. * count as f64 * f64::from_bits(1);
            error.push(
                8. * U * abs1(value)
                    + 64. * count as f64 * U * U * magnitude[k] / (1. - gamma(4 * count + 32))
                    + underflow,
            );
        }
    }
    (result, error)
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "fma")]
unsafe fn residual_fma(a: &Sparse, vectors: &[Z], eigenvalues: &[Z]) -> (Vec<Z>, Vec<f64>) {
    residual_kernel(a, vectors, eigenvalues)
}
#[derive(Clone, Copy, Default)]
struct ResidualCell {
    value: Wide,
    magnitude: f64,
}
#[inline(always)]
fn residual_column(a: &Sparse, vector: &[Z], lambda: Z, column: &mut [ResidualCell]) {
    for (k, entries) in a.iter().enumerate() {
        let v = vector[k];
        for &(i, value) in entries {
            column[i].value.re.product(value, v.re);
            column[i].value.im.product(value, v.im);
            column[i].magnitude += value.abs() * abs1(v);
        }
    }
    for (i, &v) in vector.iter().enumerate() {
        column[i].value.product(
            Z {
                re: -lambda.re,
                im: -lambda.im,
            },
            v,
        );
        column[i].magnitude += abs1(lambda) * abs1(v);
    }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "fma")]
unsafe fn residual_column_fma(a: &Sparse, vector: &[Z], lambda: Z, column: &mut [ResidualCell]) {
    residual_column(a, vector, lambda, column)
}
fn parallel_residual(
    a: &Sparse,
    vectors: &[Z],
    eigenvalues: &[Z],
    hardware_fma: bool,
    work: usize,
) -> (Vec<Z>, Vec<f64>) {
    let n = a.len();
    let mut cells = vec![ResidualCell::default(); n * n];
    let mut counts = vec![0; n];
    for col in a {
        for &(i, _) in col {
            counts[i] += 1;
        }
    }
    // Each independent column retains EXACTLY the original increasing-k
    // accumulation sequence. There is no cross-thread floating reduction.
    super::cluster_action::parallel_columns(&mut cells, n, work, |j, column| {
        let vector = &vectors[j * n..(j + 1) * n];
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if hardware_fma {
            unsafe { residual_column_fma(a, vector, eigenvalues[j], column) };
            return;
        }
        residual_column(a, vector, eigenvalues[j], column);
    });
    let mut result = Vec::with_capacity(n * n);
    let mut error = Vec::with_capacity(n * n);
    for j in 0..n {
        for i in 0..n {
            let cell = cells[i + j * n];
            let value = cell.value.value();
            result.push(value);
            let count = counts[i] + 8;
            let underflow = 32. * count as f64 * f64::from_bits(1);
            error.push(
                8. * U * abs1(value)
                    + 64. * count as f64 * U * U * cell.magnitude / (1. - gamma(4 * count + 32))
                    + underflow,
            );
        }
    }
    (result, error)
}
fn residual(a: &Sparse, vectors: &[Z], eigenvalues: &[Z]) -> (Vec<Z>, Vec<f64>, bool) {
    let hardware_fma = {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            std::is_x86_feature_detected!("fma")
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            false
        }
    };
    let work = a
        .iter()
        .map(Vec::len)
        .sum::<usize>()
        .saturating_mul(a.len());
    if work >= 200_000 {
        let (values, errors) = parallel_residual(a, vectors, eigenvalues, hardware_fma, work);
        return (values, errors, hardware_fma);
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::is_x86_feature_detected!("fma") {
        let (values, errors) = unsafe { residual_fma(a, vectors, eigenvalues) };
        return (values, errors, true);
    }
    let (values, errors) = residual_kernel(a, vectors, eigenvalues);
    (values, errors, false)
}

#[derive(Clone, Debug)]
pub struct Diagnostics {
    pub dimension: usize,
    pub original_residual_max: f64,
    pub original_residual_one: f64,
    pub original_residual_frobenius: f64,
    pub original_residual_one_upper: f64,
    pub modal_defect_one: f64,
    pub modal_roundoff_one: f64,
    pub inverse_residual_one_bound: f64,
    pub used_hardware_fma: bool,
    pub elapsed_ms: f64,
}
pub struct Defect {
    /// Column-major W (A R - R Lambda), retaining every entry.
    pub matrix: Vec<Z>,
    /// Column-major defect in the ORIGINAL generator coordinates.
    pub original_residual: Vec<Z>,
    /// Absolute complex-modulus rounding estimates, per original entry.
    pub original_residual_error: Vec<f64>,
    /// Absolute complex-modulus rounding estimates, per modal entry.
    pub matrix_error: Vec<f64>,
    pub diagnostics: Diagnostics,
}

pub fn compute(
    lap: &Lapack,
    a: &Sparse,
    vectors: &[Z],
    inverse: &[Z],
    eigenvalues: &[Z],
) -> Result<Defect, String> {
    let started = Instant::now();
    let n = a.len();
    let size = n.checked_mul(n).ok_or("Modal defect dimension overflow")?;
    if n == 0 || vectors.len() != size || inverse.len() != size || eigenvalues.len() != n {
        return Err("Invalid complete modal defect dimensions".into());
    }
    for entries in a {
        if entries.iter().any(|&(i, x)| i >= n || !x.is_finite()) {
            return Err("Invalid sparse generator in modal defect".into());
        }
    }
    finite(vectors, "basis")?;
    finite(inverse, "inverse")?;
    finite(eigenvalues, "spectrum")?;
    let (original_residual, original_residual_error, used_hardware_fma) =
        residual(a, vectors, eigenvalues);
    finite(&original_residual, "compensated original residual")?;
    if original_residual_error.iter().any(|x| !x.is_finite()) {
        return Err("Original residual roundoff estimate overflowed".into());
    }
    let inverse_abs: Vec<_> = inverse.iter().map(|&z| abs1(z)).collect();
    let residual_abs: Vec<_> = original_residual.iter().map(|&z| abs1(z)).collect();
    let vectors_abs: Vec<_> = vectors.iter().map(|&z| abs1(z)).collect();
    let matrix = lap.multiply_complex(inverse, &original_residual, n, n, n)?;
    let magnitude = lap.multiply(&inverse_abs, &residual_abs, n, n, n);
    let propagated = lap.multiply(&inverse_abs, &original_residual_error, n, n, n);
    let matrix_error: Vec<_> = magnitude
        .iter()
        .zip(&propagated)
        .map(|(&m, &p)| (gamma(8 * n + 32) * m + p) / (1. - gamma(n + 4)))
        .collect();
    let mut inverse_residual = lap.multiply_complex(inverse, vectors, n, n, n)?;
    for i in 0..n {
        inverse_residual[i + i * n].re -= 1.;
    }
    let inverse_magnitude = lap.multiply(&inverse_abs, &vectors_abs, n, n, n);
    let inverse_error: Vec<_> = inverse_residual
        .iter()
        .zip(&inverse_magnitude)
        .map(|(&e, &m)| modulus(e) + gamma(8 * n + 32) * m / (1. - gamma(n + 4)) + 4. * U)
        .collect();
    let original_upper: Vec<_> = original_residual
        .iter()
        .zip(&original_residual_error)
        .map(|(&r, &e)| modulus(r) + e)
        .collect();
    let diagnostics = Diagnostics {
        dimension: n,
        original_residual_max: original_residual
            .iter()
            .map(|&z| modulus(z))
            .fold(0., f64::max),
        original_residual_one: complex_one(&original_residual, n),
        original_residual_frobenius: original_residual
            .iter()
            .map(|&z| modulus(z))
            .fold(0_f64, f64::hypot),
        original_residual_one_upper: real_one(&original_upper, n),
        modal_defect_one: complex_one(&matrix, n),
        modal_roundoff_one: real_one(&matrix_error, n),
        inverse_residual_one_bound: real_one(&inverse_error, n),
        used_hardware_fma,
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.,
    };
    finite(&matrix, "modal defect")?;
    if matrix_error
        .iter()
        .chain(&inverse_error)
        .any(|x| !x.is_finite())
    {
        return Err("Modal product or inverse roundoff estimate overflowed".into());
    }
    if !diagnostics.inverse_residual_one_bound.is_finite()
        || diagnostics.inverse_residual_one_bound >= 1.
    {
        return Err(format!(
            "Modal inverse is not verified: ||W R-I|| bound {}",
            diagnostics.inverse_residual_one_bound
        ));
    }
    if [
        diagnostics.original_residual_max,
        diagnostics.original_residual_one_upper,
        diagnostics.original_residual_frobenius,
        diagnostics.modal_defect_one,
        diagnostics.modal_roundoff_one,
    ]
    .iter()
    .any(|x| !x.is_finite())
    {
        return Err("Modal defect norm estimate overflowed".into());
    }
    Ok(Defect {
        matrix,
        original_residual,
        original_residual_error,
        matrix_error,
        diagnostics,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn z(x: f64) -> Z {
        Z { re: x, im: 0. }
    }
    #[test]
    fn compensated_residual_retains_cancelled_small_decay() {
        let lap = Lapack::load().unwrap();
        let a = vec![
            vec![(0, 1e16), (1, 0.)],
            vec![(0, 1.), (1, -2.)],
            vec![(0, -1e16), (2, -3.)],
        ];
        let v = vec![
            z(1.),
            z(1.),
            z(1.),
            z(0.),
            z(1.),
            z(0.),
            z(0.),
            z(0.),
            z(1.),
        ];
        let w = vec![
            z(1.),
            z(-1.),
            z(-1.),
            z(0.),
            z(1.),
            z(0.),
            z(0.),
            z(0.),
            z(1.),
        ];
        let result = compute(&lap, &a, &v, &w, &[z(0.), z(-2.), z(-3.)]).unwrap();
        assert_eq!(result.original_residual[0].re, 1.);
        assert_eq!(result.matrix[0].re, 1.);
        assert!(
            result.diagnostics.original_residual_one_upper
                >= result.diagnostics.original_residual_one
        );
    }
    #[test]
    fn complete_complex_basis_and_small_nonzero_rates_are_retained() {
        let lap = Lapack::load().unwrap();
        let a = vec![vec![(0, -1e-12), (1, -1e8)], vec![(0, 1e8), (1, -1e-12)]];
        let v = vec![z(1.), Z { re: 0., im: 1. }, z(1.), Z { re: 0., im: -1. }];
        let w = vec![
            z(0.5),
            z(0.5),
            Z { re: 0., im: -0.5 },
            Z { re: 0., im: 0.5 },
        ];
        // Intentionally omit the decay from the supplied eigenvalues.
        let result = compute(
            &lap,
            &a,
            &v,
            &w,
            &[Z { re: 0., im: 1e8 }, Z { re: 0., im: -1e8 }],
        )
        .unwrap();
        assert!((result.matrix[0].re + 1e-12).abs() < 1e-27);
        assert!((result.matrix[3].re + 1e-12).abs() < 1e-27);
        assert!(result.matrix[1].re.abs() < 1e-27 && result.matrix[2].re.abs() < 1e-27);
        assert!(result.diagnostics.inverse_residual_one_bound < 1e-12);
    }
    #[test]
    fn fma_dispatch_matches_portable_compensated_kernel() {
        let n = 17;
        let a: Sparse = (0..n)
            .map(|j| {
                (0..7)
                    .map(|k| ((j + 3 * k) % n, ((j * 19 + k * 5) % 71) as f64 / 13. - 2.))
                    .collect()
            })
            .collect();
        let v: Vec<_> = (0..n * n)
            .map(|i| Z {
                re: ((i * 17) % 53) as f64 / 11.,
                im: ((i * 13) % 47) as f64 / 7.,
            })
            .collect();
        let values: Vec<_> = (0..n)
            .map(|i| Z {
                re: -(i as f64) / 17.,
                im: i as f64 * 1e8,
            })
            .collect();
        let portable = residual_kernel(&a, &v, &values);
        let selected = residual(&a, &v, &values);
        for (a, b) in portable.0.iter().zip(&selected.0) {
            assert_eq!(a.re.to_bits(), b.re.to_bits());
            assert_eq!(a.im.to_bits(), b.im.to_bits());
        }
        assert_eq!(portable.1, selected.1);
    }
    #[test]
    fn parallel_columns_match_original_serial_residual_bit_for_bit() {
        let n = 211;
        let a: Sparse = (0..n)
            .map(|j| {
                (0..13)
                    .map(|k| ((j + 17 * k) % n, ((j * 19 + k * 5) % 71) as f64 / 13. - 2.))
                    .collect()
            })
            .collect();
        let v: Vec<_> = (0..n * n)
            .map(|i| Z {
                re: ((i * 17) % 53) as f64 / 11.,
                im: ((i * 13) % 47) as f64 / 7.,
            })
            .collect();
        let values: Vec<_> = (0..n)
            .map(|i| Z {
                re: -(i as f64) / 17.,
                im: i as f64 * 1e8,
            })
            .collect();
        let reference = residual_kernel(&a, &v, &values);
        let parallel = residual(&a, &v, &values);
        for (a, b) in reference.0.iter().zip(&parallel.0) {
            assert_eq!(a.re.to_bits(), b.re.to_bits());
            assert_eq!(a.im.to_bits(), b.im.to_bits());
        }
        assert_eq!(reference.1, parallel.1);
    }
    #[test]
    fn invalid_dimensions_and_nonfinite_inputs_fail_explicitly() {
        let lap = Lapack::load().unwrap();
        assert!(compute(&lap, &vec![], &[], &[], &[]).is_err());
        assert!(compute(
            &lap,
            &vec![vec![(0, f64::NAN)]],
            &[z(1.)],
            &[z(1.)],
            &[z(0.)]
        )
        .is_err());
        assert!(compute(&lap, &vec![vec![(1, 1.)]], &[z(1.)], &[z(1.)], &[z(0.)]).is_err());
        assert!(compute(&lap, &vec![vec![(0, 1.)]], &[z(1.)], &[z(0.)], &[z(0.)]).is_err());
    }
    #[test]
    fn complete_raw_basis_preserves_conjugate_phase_and_rejects_defective_basis() {
        let lap = Lapack::load().unwrap();
        let a = vec![vec![(0, -0.3), (1, 1.7)], vec![(0, -1.7), (1, -0.3)]];
        let basis = raw_modal_basis(&lap, &a).unwrap();
        assert_eq!(basis.dimension, 2);
        let defect = compute(&lap, &a, &basis.vectors, &basis.inverse, &basis.eigenvalues).unwrap();
        assert!(defect.diagnostics.modal_defect_one < 1e-14);
        let t = 0.37;
        let mut out = [Z::default(); 2];
        for j in 0..2 {
            let c = basis.inverse[j];
            let lambda = basis.eigenvalues[j];
            let (s, co) = (lambda.im * t).sin_cos();
            let e = Z {
                re: (lambda.re * t).exp() * co,
                im: (lambda.re * t).exp() * s,
            };
            let coefficient = Z {
                re: c.re * e.re - c.im * e.im,
                im: c.re * e.im + c.im * e.re,
            };
            for i in 0..2 {
                let v = basis.vectors[i + j * 2];
                out[i].re += v.re * coefficient.re - v.im * coefficient.im;
                out[i].im += v.re * coefficient.im + v.im * coefficient.re;
            }
        }
        assert!((out[0].re - (-0.3 * t).exp() * (1.7 * t).cos()).abs() < 1e-14);
        assert!((out[1].re - (-0.3 * t).exp() * (1.7 * t).sin()).abs() < 1e-14);
        assert!(out.iter().all(|z| z.im.abs() < 1e-14));
        assert!(raw_modal_basis(&lap, &vec![vec![(0, -1.)], vec![(0, 1.), (1, -1.)]]).is_err());
    }
}
