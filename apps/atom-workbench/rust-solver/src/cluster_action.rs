//! Numerically refine a complete invariant spectral cluster before propagation.
//!
//! A large imaginary spectral range can corrupt tiny decay rates and the
//! eigenvectors within a near-degenerate slow cluster. Correcting individual
//! eigenvalues is insufficient. The full left/right subspaces are retained and
//! B=(U^T V)^-1 U^T A V is recomputed with compensated sparse products and dots.
//! Every unrefined fast mode is still propagated. The explicit imaginary
//! partition limit is a numerical choice, never a physical/model cutoff.
//!
//! The real LAPACK encoding retains complete conjugate pairs. Preparation is
//! reusable for arbitrary initial vectors and sample times. Defective full
//! eigenbases fail explicitly so callers can use dense_expm's Pade fallback.
use super::{
    dense_expm,
    lapack::{Complex64, Lapack},
};
use std::time::Instant;

pub type Sparse = Vec<Vec<(usize, f64)>>;

/// Independent columns can run concurrently without changing any element's
/// floating-point accumulation order. BLAS calls remain outside these workers.
pub fn parallel_columns<T: Send, F: Fn(usize, &mut [T]) + Sync>(
    values: &mut [T],
    rows: usize,
    estimated_work: usize,
    body: F,
) {
    let columns = values.len() / rows;
    let configured = std::env::var("ATOM_NUMERIC_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map_or(1, |v| v.get())
                .min(4)
        });
    let workers = if estimated_work < 200_000 {
        1
    } else {
        configured.clamp(1, 8).min(columns.max(1))
    };
    columns_with_workers(values, rows, workers, body);
}
fn columns_with_workers<T: Send, F: Fn(usize, &mut [T]) + Sync>(
    values: &mut [T],
    rows: usize,
    workers: usize,
    body: F,
) {
    let columns = values.len() / rows;
    if workers <= 1 || columns <= 1 {
        for (j, column) in values.chunks_mut(rows).enumerate() {
            body(j, column);
        }
        return;
    }
    let width = columns.div_ceil(workers);
    std::thread::scope(|scope| {
        for (chunk_index, chunk) in values.chunks_mut(rows * width).enumerate() {
            let body = &body;
            scope.spawn(move || {
                for (local, column) in chunk.chunks_mut(rows).enumerate() {
                    body(chunk_index * width + local, column);
                }
            });
        }
    });
}

#[derive(Clone, Debug)]
pub struct Diagnostics {
    pub dimension: usize,
    pub cluster_dimension: usize,
    pub imaginary_partition_limit: f64,
    pub full_basis_condition: f64,
    pub overlap_condition: f64,
    pub generator_norm: f64,
    pub cluster_norm: f64,
    /// A V - V B in original coordinate units; includes the measured BLAS
    /// residual used for the verification product, not a forward-error claim.
    pub subspace_residual_max: f64,
    pub subspace_residual_frobenius: f64,
    pub imaginary_cluster_gap: Option<f64>,
    pub preparation_ms: f64,
}

#[derive(Debug)]
pub struct ActionResult {
    pub states: Vec<Vec<f64>>,
    /// Actual full-basis reconstruction error before overriding t=0 exactly.
    pub initial_error: f64,
    pub elapsed_ms: f64,
    pub cluster_used_pade: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModeOrigin {
    RefinedCluster,
    OriginalFast,
    PhysicalCoordinate,
}

/// Complete complex modal representation, column-major matrices. Refined
/// cluster modes precede all original fast modes. A conjugate pair occupies
/// adjacent columns, positive imaginary eigenvalue first. No mode is removed.
pub struct ModalBasis {
    pub dimension: usize,
    pub eigenvalues: Vec<Complex64>,
    pub vectors: Vec<Complex64>,
    pub inverse: Vec<Complex64>,
    pub condition: f64,
    pub origin: Vec<ModeOrigin>,
}

#[derive(Clone, Copy, Default)]
struct Double {
    hi: f64,
    lo: f64,
}
impl Double {
    #[inline]
    fn add(&mut self, high: f64, low: f64) {
        let sum = self.hi + high;
        let z = sum - self.hi;
        let error = (self.hi - (sum - z)) + (high - z);
        let correction = self.lo + low + error;
        let result = sum + correction;
        self.lo = correction - (result - sum);
        self.hi = result;
    }
    #[inline]
    fn product(&mut self, a: f64, b: f64) {
        let p = a * b;
        self.add(p, a.mul_add(b, -p));
    }
    #[inline]
    fn value(self) -> f64 {
        self.hi + self.lo
    }
}

fn finite(values: &[f64], label: &str) -> Result<(), String> {
    if values.iter().any(|v| !v.is_finite()) {
        Err(format!(
            "Non-finite {label} in invariant-cluster propagation"
        ))
    } else {
        Ok(())
    }
}
fn norm_one(matrix: &[f64], n: usize) -> f64 {
    matrix
        .chunks(n)
        .map(|col| col.iter().map(|x| x.abs()).sum::<f64>())
        .fold(0., f64::max)
}
fn max_abs(values: &[f64]) -> f64 {
    values.iter().map(|v| v.abs()).fold(0., f64::max)
}
fn pick_columns(matrix: &[f64], n: usize, indices: &[usize]) -> Vec<f64> {
    indices
        .iter()
        .flat_map(|&index| matrix[index * n..(index + 1) * n].iter().copied())
        .collect()
}
fn modal_values(
    wr: &[f64],
    wi: &[f64],
    coefficients: &[f64],
    times: &[f64],
) -> Result<Vec<f64>, String> {
    let n = wr.len();
    let mut modes = vec![0.; n * times.len()];
    for (ti, &t) in times.iter().enumerate() {
        let mut j = 0;
        while j < n {
            if wi[j] > 0. {
                if j + 1 >= n {
                    return Err("Incomplete conjugate eigenvector pair".into());
                }
                if coefficients[j] == 0. && coefficients[j + 1] == 0. {
                    j += 2;
                    continue;
                }
                let phase = wi[j] * t;
                let (s, c) = phase.sin_cos();
                let decay = (wr[j] * t).exp();
                modes[j + ti * n] = decay * (c * coefficients[j] + s * coefficients[j + 1]);
                modes[j + 1 + ti * n] = decay * (-s * coefficients[j] + c * coefficients[j + 1]);
                j += 2;
            } else {
                if coefficients[j] != 0. {
                    modes[j + ti * n] = coefficients[j] * (wr[j] * t).exp();
                }
                j += 1;
            }
        }
    }
    finite(&modes, "spectral exponential")?;
    Ok(modes)
}

struct SmallSpectrum {
    wr: Vec<f64>,
    wi: Vec<f64>,
    vectors: Vec<f64>,
    lu: Vec<f64>,
    pivots: Vec<i32>,
}
enum Central {
    Empty,
    Spectrum(SmallSpectrum),
    Pade(Vec<f64>),
}

pub struct Prepared {
    pub diagnostics: Diagnostics,
    n: usize,
    wr: Vec<f64>,
    wi: Vec<f64>,
    vectors: Vec<f64>,
    lu: Vec<f64>,
    pivots: Vec<i32>,
    central_indices: Vec<usize>,
    central_vectors: Vec<f64>,
    central: Central,
}

/// Dense wrapper for the constant-generator solver, preserving every entry.
pub fn prepare_dense(
    lap: &Lapack,
    a: &[f64],
    n: usize,
    imaginary_partition_limit: f64,
) -> Result<Prepared, String> {
    if n == 0 || n.checked_mul(n) != Some(a.len()) {
        return Err("Invalid dense invariant-cluster dimensions".into());
    }
    finite(a, "input matrix")?;
    let sparse: Sparse = (0..n)
        .map(|col| {
            (0..n)
                .filter_map(|row| {
                    let value = a[row + col * n];
                    (value != 0.).then_some((row, value))
                })
                .collect()
        })
        .collect();
    prepare(lap, &sparse, imaginary_partition_limit)
}

pub fn prepare(
    lap: &Lapack,
    a: &Sparse,
    imaginary_partition_limit: f64,
) -> Result<Prepared, String> {
    let started = Instant::now();
    let n = a.len();
    if n == 0
        || n > i32::MAX as usize
        || !imaginary_partition_limit.is_finite()
        || imaginary_partition_limit < 0.
    {
        return Err("Invalid invariant-cluster dimensions or numerical partition".into());
    }
    let size = n
        .checked_mul(n)
        .ok_or("Dense invariant-cluster allocation overflow")?;
    let mut dense = vec![0.; size];
    for (col, entries) in a.iter().enumerate() {
        for &(row, value) in entries {
            if row >= n || !value.is_finite() {
                return Err("Invalid sparse generator entry".into());
            }
            dense[row + col * n] += value;
        }
    }
    finite(&dense, "assembled input matrix")?;
    let generator_norm = norm_one(&dense, n);
    let (wr, wi, left, vectors) = lap.eigen_both(&dense, n)?;
    finite(&wr, "eigenvalue real parts")?;
    finite(&wi, "eigenvalue imaginary parts")?;
    finite(&left, "left eigenvectors")?;
    finite(&vectors, "right eigenvectors")?;
    let (lu, pivots) = lap.factor(&vectors, n)?;
    let condition = lap.condition(&lu, n, norm_one(&vectors, n))?;
    if !condition.is_finite() || condition > 1e10 {
        return Err(format!("Full eigenbasis is defective or ill-conditioned ({condition:e}); use the Pade exponential fallback"));
    }
    let central_indices: Vec<_> = (0..n)
        .filter(|&i| wi[i].abs() <= imaginary_partition_limit)
        .collect();
    let m = central_indices.len();
    let central_vectors = pick_columns(&vectors, n, &central_indices);
    let mut overlap_condition = 1.;
    let mut block_norm = 0.;
    let mut residual_max = 0_f64;
    let mut residual_frobenius = 0_f64;
    let central = if m == 0 {
        Central::Empty
    } else {
        let u = pick_columns(&left, n, &central_indices);
        // Retain high+low sparse products through U^T A V, instead of first
        // rounding a cancellation-dominated A V back to a single double.
        let mut av = vec![Double::default(); n * m];
        let products = a.iter().map(Vec::len).sum::<usize>() * m;
        parallel_columns(&mut av, n, products, |k, column| {
            for (col, entries) in a.iter().enumerate() {
                let v = central_vectors[col + k * n];
                if v == 0. {
                    continue;
                }
                for &(row, value) in entries {
                    column[row].product(value, v);
                }
            }
        });
        if av.iter().any(|v| !v.hi.is_finite() || !v.lo.is_finite()) {
            return Err("Compensated sparse product overflowed".into());
        }
        let mut projected = vec![(0., 0.); m * m];
        parallel_columns(&mut projected, m, n * m * m, |col, column| {
            for row in 0..m {
                let mut g = Double::default();
                let mut value = Double::default();
                for k in 0..n {
                    let uk = u[k + row * n];
                    g.product(uk, central_vectors[k + col * n]);
                    value.product(uk, av[k + col * n].hi);
                    value.product(uk, av[k + col * n].lo);
                }
                column[row] = (g.value(), value.value());
            }
        });
        let gram: Vec<_> = projected.iter().map(|v| v.0).collect();
        let numerator: Vec<_> = projected.iter().map(|v| v.1).collect();
        drop(projected);
        finite(&gram, "cluster overlap")?;
        finite(&numerator, "projected generator")?;
        let (glu, gpiv) = lap.factor(&gram, m)?;
        overlap_condition = lap.condition(&glu, m, norm_one(&gram, m))?;
        if !overlap_condition.is_finite() || overlap_condition > 1e10 {
            return Err(format!(
                "Invariant-cluster left/right overlap is ill-conditioned ({overlap_condition:e})"
            ));
        }
        let block = lap.solve_matrix_factored(&glu, &gpiv, &numerator, m)?;
        finite(&block, "refined cluster generator")?;
        block_norm = norm_one(&block, m);
        let vb = lap.multiply(&central_vectors, &block, n, m, m);
        for (value, &predicted) in av.iter().zip(&vb) {
            let residual = (value.hi - predicted) + value.lo;
            residual_max = residual_max.max(residual.abs());
            residual_frobenius = residual_frobenius.hypot(residual);
        }
        match lap.eigen(&block, m).and_then(|(cw, ci, cv)| {
            let (clu, cp) = lap.factor(&cv, m)?;
            let condition = lap.condition(&clu, m, norm_one(&cv, m))?;
            if !condition.is_finite() || condition > 1e10 {
                return Err("Refined block eigenbasis needs Pade propagation".into());
            }
            Ok(SmallSpectrum {
                wr: cw,
                wi: ci,
                vectors: cv,
                lu: clu,
                pivots: cp,
            })
        }) {
            Ok(spectrum) => Central::Spectrum(spectrum),
            Err(_) => Central::Pade(block),
        }
    };
    let gap = if m == 0 || m == n {
        None
    } else {
        let mut result = f64::INFINITY;
        for &i in &central_indices {
            for j in 0..n {
                if wi[j].abs() > imaginary_partition_limit {
                    result = result.min((wi[i] - wi[j]).abs());
                }
            }
        }
        Some(result)
    };
    Ok(Prepared {
        diagnostics: Diagnostics {
            dimension: n,
            cluster_dimension: m,
            imaginary_partition_limit,
            full_basis_condition: condition,
            overlap_condition,
            generator_norm,
            cluster_norm: block_norm,
            subspace_residual_max: residual_max,
            subspace_residual_frobenius: residual_frobenius,
            imaginary_cluster_gap: gap,
            preparation_ms: started.elapsed().as_secs_f64() * 1000.,
        },
        n,
        wr,
        wi,
        vectors,
        lu,
        pivots,
        central_indices,
        central_vectors,
        central,
    })
}

impl Prepared {
    /// Cache this exported basis when repeated Fourier transformations are
    /// needed. apply() and its existing cache remain entirely unchanged.
    pub fn modal_basis(&self, lap: &Lapack) -> Result<ModalBasis, String> {
        let n = self.n;
        let m = self.central_indices.len();
        let mut wr = vec![];
        let mut wi = vec![];
        let mut real_vectors = Vec::with_capacity(n * n);
        let mut origin = vec![];
        match &self.central {
            Central::Empty=>{},
            Central::Pade(_)=>return Err("The refined cluster has no well-conditioned complete eigenbasis; retain the Pade/general propagation path".into()),
            Central::Spectrum(spectrum)=>{
                real_vectors.extend(lap.multiply(&self.central_vectors,&spectrum.vectors,n,m,m));
                wr.extend_from_slice(&spectrum.wr);wi.extend_from_slice(&spectrum.wi);
                origin.extend(vec![ModeOrigin::RefinedCluster;m]);
            }
        }
        let mut is_central = vec![false; n];
        for &index in &self.central_indices {
            is_central[index] = true;
        }
        for i in 0..n {
            if !is_central[i] {
                real_vectors.extend_from_slice(&self.vectors[i * n..(i + 1) * n]);
                wr.push(self.wr[i]);
                wi.push(self.wi[i]);
                origin.push(ModeOrigin::OriginalFast);
            }
        }
        finite(&real_vectors, "exported modal basis")?;
        let (lu, pivots) = lap.factor(&real_vectors, n)?;
        let mut identity = vec![0.; n * n];
        for i in 0..n {
            identity[i + i * n] = 1.;
        }
        let real_inverse = lap.solve_matrix_factored(&lu, &pivots, &identity, n)?;
        finite(&real_inverse, "exported modal inverse")?;
        let mut vectors = vec![Complex64::default(); n * n];
        let mut inverse = vec![Complex64::default(); n * n];
        let mut eigenvalues = vec![Complex64::default(); n];
        let mut j = 0;
        while j < n {
            if wi[j] > 0. {
                if j + 1 >= n || wi[j + 1] >= 0. {
                    return Err("Exported modal conjugate pair is incomplete".into());
                }
                eigenvalues[j] = Complex64 {
                    re: wr[j],
                    im: wi[j],
                };
                eigenvalues[j + 1] = Complex64 {
                    re: wr[j + 1],
                    im: wi[j + 1],
                };
                for i in 0..n {
                    vectors[i + j * n] = Complex64 {
                        re: real_vectors[i + j * n],
                        im: real_vectors[i + (j + 1) * n],
                    };
                    vectors[i + (j + 1) * n] = Complex64 {
                        re: real_vectors[i + j * n],
                        im: -real_vectors[i + (j + 1) * n],
                    };
                    // x=a*c_re+b*c_im = (a+i b)*(c_re-i c_im)/2 + c.c.
                    inverse[j + i * n] = Complex64 {
                        re: 0.5 * real_inverse[j + i * n],
                        im: -0.5 * real_inverse[j + 1 + i * n],
                    };
                    inverse[j + 1 + i * n] = Complex64 {
                        re: 0.5 * real_inverse[j + i * n],
                        im: 0.5 * real_inverse[j + 1 + i * n],
                    };
                }
                j += 2;
            } else {
                if wi[j] < 0. {
                    return Err("Unpaired negative-imaginary modal eigenvalue".into());
                }
                eigenvalues[j] = Complex64 { re: wr[j], im: 0. };
                for i in 0..n {
                    vectors[i + j * n].re = real_vectors[i + j * n];
                    inverse[j + i * n].re = real_inverse[j + i * n];
                }
                j += 1;
            }
        }
        let norm = |matrix: &[Complex64]| {
            matrix
                .chunks(n)
                .map(|col| col.iter().map(|z| z.re.hypot(z.im)).sum::<f64>())
                .fold(0., f64::max)
        };
        let condition = norm(&vectors) * norm(&inverse);
        if !condition.is_finite() || condition > 1e10 {
            return Err(format!(
                "Exported complex modal basis is ill-conditioned ({condition:e})"
            ));
        }
        Ok(ModalBasis {
            dimension: n,
            eigenvalues,
            vectors,
            inverse,
            condition,
            origin,
        })
    }
    pub fn apply(
        &self,
        lap: &Lapack,
        initial: &[f64],
        times: &[f64],
    ) -> Result<ActionResult, String> {
        let started = Instant::now();
        let n = self.n;
        let m = self.central_indices.len();
        if initial.len() != n {
            return Err("Initial vector has the wrong invariant-cluster dimension".into());
        }
        finite(initial, "initial vector")?;
        finite(times, "times")?;
        let coefficients = lap.solve_factored(&self.lu, &self.pivots, initial, n)?;
        let rebuilt = lap.multiply(&self.vectors, &coefficients, n, 1, n);
        let difference: Vec<_> = rebuilt.iter().zip(initial).map(|(a, b)| a - b).collect();
        let initial_error = max_abs(&difference);
        let central_initial: Vec<_> = self
            .central_indices
            .iter()
            .map(|&i| coefficients[i])
            .collect();
        let mut fast = coefficients.clone();
        for &i in &self.central_indices {
            fast[i] = 0.;
        }
        let modes = modal_values(&self.wr, &self.wi, &fast, times)?;
        let mut all = lap.multiply(&self.vectors, &modes, n, times.len(), n);
        let cluster_used_pade = matches!(self.central, Central::Pade(_));
        let evolved = match &self.central {
            Central::Empty => vec![],
            Central::Spectrum(spectrum) => {
                let c = lap.solve_factored(&spectrum.lu, &spectrum.pivots, &central_initial, m)?;
                let modes = modal_values(&spectrum.wr, &spectrum.wi, &c, times)?;
                lap.multiply(&spectrum.vectors, &modes, m, times.len(), m)
            }
            Central::Pade(block) => dense_expm::action(lap, block, &central_initial, times)?
                .into_iter()
                .flatten()
                .collect(),
        };
        if m > 0 {
            let contribution = lap.multiply(&self.central_vectors, &evolved, n, times.len(), m);
            for (x, y) in all.iter_mut().zip(contribution) {
                *x += y;
            }
        }
        finite(&all, "reconstructed trajectory")?;
        let states = times
            .iter()
            .enumerate()
            .map(|(i, &t)| {
                if t == 0. {
                    initial.to_vec()
                } else {
                    all[i * n..(i + 1) * n].to_vec()
                }
            })
            .collect();
        Ok(ActionResult {
            states,
            initial_error,
            elapsed_ms: started.elapsed().as_secs_f64() * 1000.,
            cluster_used_pade,
        })
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn independent_column_threads_preserve_compensated_accumulation_bit_for_bit() {
        let rows = 37;
        let columns = 53;
        let mut serial = vec![0.; rows * columns];
        let mut parallel = serial.clone();
        let calculate = |column: usize, values: &mut [f64]| {
            for (row, value) in values.iter_mut().enumerate() {
                let mut sum = Double::default();
                for k in 0..129 {
                    let a = ((row + 3 * k + column) as f64).sin();
                    let b = if k % 3 == 0 {
                        1e16
                    } else if k % 3 == 1 {
                        1.
                    } else {
                        -1e16
                    };
                    sum.product(a, b);
                }
                *value = sum.value();
            }
        };
        columns_with_workers(&mut serial, rows, 1, calculate);
        columns_with_workers(&mut parallel, rows, 4, calculate);
        assert!(serial
            .iter()
            .zip(&parallel)
            .all(|(a, b)| a.to_bits() == b.to_bits()));
    }
    fn near(a: f64, b: f64, tolerance: f64) {
        assert!(
            (a - b).abs() <= tolerance,
            "{a} != {b}; tolerance {tolerance}"
        );
    }
    fn stiff_example() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let g = 2_f64.powi(-20);
        let o = vec![
            0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, -0.5, -0.5, 0.5,
        ];
        let b = [
            -g, g, 0., 0., g, -g, 0., 0., 0., 0., -0.5, 1e8, 0., 0., -1e8, -0.5,
        ];
        let mut a = vec![0.; 16];
        for i in 0..4 {
            for j in 0..4 {
                let mut value = Double::default();
                for k in 0..4 {
                    for l in 0..4 {
                        value.product(o[i + k * 4] * b[k + l * 4], o[j + l * 4]);
                    }
                }
                a[i + j * 4] = value.value();
            }
        }
        let initial = o[..4].to_vec();
        (a, initial, o)
    }
    #[test]
    fn compensated_product_retains_cancelled_low_part() {
        let mut value = Double::default();
        value.product(1e16, 1.);
        value.product(1., 1.);
        value.product(-1e16, 1.);
        assert_eq!(value.value(), 1.);
    }
    #[test]
    fn whole_slow_cluster_is_accurate_across_a_hundred_million_frequency_range() {
        let lap = Lapack::load().unwrap();
        let (a, initial, o) = stiff_example();
        let times = [0., 0.125, 1., 20.];
        let prepared = prepare_dense(&lap, &a, 4, 1.).unwrap();
        assert_eq!(prepared.diagnostics.cluster_dimension, 2);
        let result = prepared.apply(&lap, &initial, &times).unwrap();
        assert_eq!(result.states[0], initial);
        let g = 2_f64.powi(-20);
        for (x, t) in result.states.iter().zip(times) {
            let p0 = 0.5 * (1. + (-2. * g * t).exp());
            let p1 = 1. - p0;
            for i in 0..4 {
                near(x[i], o[i] * p0 + o[i + 4] * p1, 2e-12);
            }
        }
        assert!(prepared.diagnostics.cluster_norm < 1e-4);
        assert!(prepared.diagnostics.imaginary_cluster_gap.unwrap() > 9e7);
    }
    #[test]
    fn signed_permutation_gauge_preserves_every_coordinate() {
        let lap = Lapack::load().unwrap();
        let (a, initial, _) = stiff_example();
        let p = [2, 0, 3, 1];
        let signs = [1., -1., 1., -1.];
        let mut gauge = vec![0.; 16];
        let mut gauge_initial = vec![0.; 4];
        for i in 0..4 {
            gauge_initial[i] = signs[i] * initial[p[i]];
            for j in 0..4 {
                gauge[i + j * 4] = signs[i] * signs[j] * a[p[i] + p[j] * 4];
            }
        }
        let times = [0., 0.125, 1., 20.];
        let base = prepare_dense(&lap, &a, 4, 1.)
            .unwrap()
            .apply(&lap, &initial, &times)
            .unwrap();
        let other = prepare_dense(&lap, &gauge, 4, 1.)
            .unwrap()
            .apply(&lap, &gauge_initial, &times)
            .unwrap();
        for (x, y) in base.states.iter().zip(&other.states) {
            for i in 0..4 {
                near(x[p[i]], signs[i] * y[i], 3e-12);
            }
        }
    }
    #[test]
    fn unrefined_fast_conjugate_pair_is_kept_completely() {
        let lap = Lapack::load().unwrap();
        let a = [-0.2, 7., -7., -0.2];
        let times = [0., 0.3, 2.];
        let prepared = prepare_dense(&lap, &a, 2, 0.).unwrap();
        assert_eq!(prepared.diagnostics.cluster_dimension, 0);
        let result = prepared.apply(&lap, &[1., 0.], &times).unwrap();
        for (x, t) in result.states.iter().zip(times) {
            near(x[0], (-0.2 * t).exp() * (7. * t).cos(), 1e-13);
            near(x[1], (-0.2 * t).exp() * (7. * t).sin(), 1e-13);
        }
    }
    #[test]
    fn cached_preparation_supports_new_initial_vectors_and_times() {
        let lap = Lapack::load().unwrap();
        let a = [-0.2, 7., -7., -0.2];
        let prepared = prepare_dense(&lap, &a, 2, 20.).unwrap();
        for initial in [[1., 0.], [0.3, -0.7]] {
            let times = [0., 0.1, 0.9];
            let expected = dense_expm::action(&lap, &a, &initial, &times).unwrap();
            let result = prepared.apply(&lap, &initial, &times).unwrap();
            for (x, y) in result.states.iter().zip(expected) {
                for (a, b) in x.iter().zip(y) {
                    near(*a, b, 2e-13);
                }
            }
        }
    }
    #[test]
    fn defective_full_basis_requests_the_existing_pade_fallback() {
        let lap = Lapack::load().unwrap();
        let a = [-1., 0., 1., -1.];
        assert!(prepare_dense(&lap, &a, 2, 20.).is_err());
        let result = dense_expm::action(&lap, &a, &[0., 1.], &[1.]).unwrap();
        near(result[0][0], (-1_f64).exp(), 1e-14);
    }
    #[test]
    fn exactly_unoccupied_modes_do_not_create_zero_times_infinity() {
        let lap = Lapack::load().unwrap();
        let a = [-1., 0., 0., 1000.];
        let result = prepare_dense(&lap, &a, 2, 1.)
            .unwrap()
            .apply(&lap, &[1., 0.], &[1.])
            .unwrap();
        near(result.states[0][0], (-1_f64).exp(), 1e-14);
        assert_eq!(result.states[0][1], 0.);
    }
    #[test]
    fn exported_complex_modes_preserve_inverse_phase_and_all_fast_components() {
        fn mul(a: Complex64, b: Complex64) -> Complex64 {
            Complex64 {
                re: a.re * b.re - a.im * b.im,
                im: a.re * b.im + a.im * b.re,
            }
        }
        let lap = Lapack::load().unwrap();
        let (a, mut initial, o) = stiff_example();
        for i in 0..4 {
            initial[i] += 0.001 * o[i + 8];
        }
        let prepared = prepare_dense(&lap, &a, 4, 1.).unwrap();
        let modal = prepared.modal_basis(&lap).unwrap();
        assert_eq!(modal.dimension, 4);
        assert_eq!(
            modal
                .origin
                .iter()
                .filter(|&&value| value == ModeOrigin::RefinedCluster)
                .count(),
            2
        );
        for row in 0..4 {
            for col in 0..4 {
                let mut sum = Complex64::default();
                for k in 0..4 {
                    let v = mul(modal.vectors[row + k * 4], modal.inverse[k + col * 4]);
                    sum.re += v.re;
                    sum.im += v.im;
                }
                near(sum.re, if row == col { 1. } else { 0. }, 2e-13);
                near(sum.im, 0., 2e-13);
            }
        }
        let times = [0., 0.125, 1.];
        let expected = prepared.apply(&lap, &initial, &times).unwrap();
        for (ti, &t) in times.iter().enumerate() {
            let mut state = [Complex64::default(); 4];
            for j in 0..4 {
                let mut coefficient = Complex64::default();
                for i in 0..4 {
                    coefficient.re += modal.inverse[j + i * 4].re * initial[i];
                    coefficient.im += modal.inverse[j + i * 4].im * initial[i];
                }
                let eigen = modal.eigenvalues[j];
                let (s, c) = (eigen.im * t).sin_cos();
                let exponential = Complex64 {
                    re: (eigen.re * t).exp() * c,
                    im: (eigen.re * t).exp() * s,
                };
                let value = mul(coefficient, exponential);
                for i in 0..4 {
                    let z = mul(modal.vectors[i + j * 4], value);
                    state[i].re += z.re;
                    state[i].im += z.im;
                }
            }
            for i in 0..4 {
                near(state[i].re, expected.states[ti][i], 2e-13);
                near(state[i].im, 0., 2e-13);
            }
        }
    }
}
