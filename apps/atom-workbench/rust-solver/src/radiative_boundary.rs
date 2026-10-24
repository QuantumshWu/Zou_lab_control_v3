//! Complete matrix-free Fourier boundary of the radiative Lindblad generator.
//!
//! Finite Fourier polynomials are evaluated on an EXACTLY zero-padded tensor
//! grid, all C rho C^dagger and anticommutator products are evaluated there,
//! and a multidimensional DFT recovers every output coefficient without alias.
//! No collapse term, state, coherence or small Fourier coefficient is removed.
//! H_F is constant and cannot create an outside density harmonic.
use crate::{
    generator_mean::BoundaryEvaluation,
    harmonics::{self, FourierCoefficient},
    lapack::Complex64 as Z,
    multitone_frame::Frame,
    Lapack, C,
};
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

type Fourier = BTreeMap<Vec<i32>, Vec<Z>>;
fn sign(index: &[i32]) -> i32 {
    index
        .iter()
        .find(|&&i| i != 0)
        .map_or(1, |&i| if i > 0 { 1 } else { -1 })
}
fn abs(z: Z) -> f64 {
    z.re.hypot(z.im)
}
fn frobenius(x: &[Z]) -> f64 {
    x.iter().fold(0_f64, |v, &z| v.hypot(abs(z)))
}
fn mul(a: Z, b: Z) -> Z {
    Z {
        re: a.re * b.re - a.im * b.im,
        im: a.re * b.im + a.im * b.re,
    }
}
fn gamma(operations: usize) -> f64 {
    let x = operations as f64 * f64::EPSILON;
    x / (1. - x)
}
fn remap(index: &[i32], mapping: &[Vec<i32>], axes: usize) -> Result<Vec<i32>, String> {
    if index.len() != mapping.len() || mapping.iter().any(|q| q.len() != axes) {
        return Err("Radiative boundary frequency-map dimensions differ".into());
    }
    let mut result = vec![0_i32; axes];
    for (&q, axis) in index.iter().zip(mapping) {
        for (i, &a) in axis.iter().enumerate() {
            result[i] = q
                .checked_mul(a)
                .and_then(|v| result[i].checked_add(v))
                .ok_or("Radiative boundary frequency-index overflow")?;
        }
    }
    Ok(result)
}
fn mapped_channels(
    frame: &Frame,
    mapping: &[Vec<i32>],
    axes: usize,
) -> Result<Vec<(Fourier, f64, usize)>, String> {
    let n = frame.dimension;
    let mut channels = Vec::new();
    for channel in &frame.collapse {
        let mut result: Fourier = BTreeMap::new();
        let mut source_norm = 0.;
        for (index, matrix) in channel {
            if matrix.len() != n * n
                || matrix
                    .iter()
                    .any(|z| !z.re.is_finite() || !z.im.is_finite())
            {
                return Err("Invalid complete collapse Fourier coefficient".into());
            }
            let mapped = remap(index, mapping, axes)?;
            let target = result
                .entry(mapped)
                .or_insert_with(|| vec![Z::default(); n * n]);
            let mut norm = 0_f64;
            for row in 0..n {
                for col in 0..n {
                    let z = matrix[row * n + col];
                    target[row + col * n].re += z.re;
                    target[row + col * n].im += z.im;
                    norm = norm.hypot(z.abs());
                }
            }
            source_norm += norm;
        }
        result.retain(|_, value| value.iter().any(|z| z.re != 0. || z.im != 0.));
        if !result.is_empty() {
            channels.push((result, source_norm, channel.len()));
        }
    }
    Ok(channels)
}
fn strides(dimensions: &[usize]) -> Vec<usize> {
    let mut result = vec![1; dimensions.len()];
    for i in (0..dimensions.len().saturating_sub(1)).rev() {
        result[i] = result[i + 1] * dimensions[i + 1];
    }
    result
}
fn location(index: &[i32], dimensions: &[usize], strides: &[usize]) -> usize {
    index
        .iter()
        .zip(dimensions)
        .zip(strides)
        .map(|((&q, &n), &s)| q.rem_euclid(n as i32) as usize * s)
        .sum()
}
fn index_at(point: usize, dimensions: &[usize], strides: &[usize]) -> Vec<i32> {
    dimensions
        .iter()
        .zip(strides)
        .map(|(&n, &s)| {
            let k = (point / s) % n;
            if n == 1 || k < n / 2 {
                k as i32
            } else {
                k as i32 - n as i32
            }
        })
        .collect()
}

/// Unnormalized separable radix-two transform. Inverse has exp(+i k.theta),
/// so a coefficient array directly becomes polynomial values with no scale.
fn fft(values: &mut [Z], dimensions: &[usize], width: usize, inverse: bool) {
    let total = values.len() / width;
    let strides = strides(dimensions);
    for (axis, &length) in dimensions.iter().enumerate() {
        if length == 1 {
            continue;
        }
        let stride = strides[axis];
        for block in (0..total).step_by(length * stride) {
            for offset in 0..stride {
                let mut j = 0;
                for i in 1..length {
                    let mut bit = length >> 1;
                    while j & bit != 0 {
                        j ^= bit;
                        bit >>= 1;
                    }
                    j ^= bit;
                    if i < j {
                        let a = (block + i * stride + offset) * width;
                        let b = (block + j * stride + offset) * width;
                        for k in 0..width {
                            values.swap(a + k, b + k);
                        }
                    }
                }
            }
        }
        let mut span = 2;
        while span <= length {
            let twiddles: Vec<_> = (0..span / 2)
                .map(|j| {
                    let (s, c) = if j == 0 {
                        (0., 1.)
                    } else if 4 * j == span {
                        (1., 0.)
                    } else {
                        (2. * std::f64::consts::PI * j as f64 / span as f64).sin_cos()
                    };
                    Z {
                        re: c,
                        im: if inverse { s } else { -s },
                    }
                })
                .collect();
            for block in (0..total).step_by(length * stride) {
                for offset in 0..stride {
                    for start in (0..length).step_by(span) {
                        for (j, &twiddle) in twiddles.iter().enumerate() {
                            let a = (block + (start + j) * stride + offset) * width;
                            let b = (block + (start + j + span / 2) * stride + offset) * width;
                            for k in 0..width {
                                let x = values[a + k];
                                let y = mul(values[b + k], twiddle);
                                values[a + k] = Z {
                                    re: x.re + y.re,
                                    im: x.im + y.im,
                                };
                                values[b + k] = Z {
                                    re: x.re - y.re,
                                    im: x.im - y.im,
                                };
                            }
                        }
                    }
                }
            }
            span *= 2;
        }
    }
}
fn adjoint(a: &[Z], n: usize) -> Vec<Z> {
    (0..n * n)
        .map(|k| {
            let z = a[k / n + (k % n) * n];
            Z {
                re: z.re,
                im: -z.im,
            }
        })
        .collect()
}

pub fn evaluate(
    lap: &Lapack,
    frame: &Frame,
    coefficients: &[FourierCoefficient],
    retained_signed: &[Vec<i32>],
    input_axis_indices: &[Vec<i32>],
) -> Result<BoundaryEvaluation, String> {
    let started = Instant::now();
    let n = frame.dimension;
    let d = n
        .checked_mul(n)
        .ok_or("Boundary density dimension overflow")?;
    let axes = coefficients
        .first()
        .ok_or("Radiative boundary needs density coefficients")?
        .index
        .len();
    if n == 0 || n > 64 || axes > 3 || input_axis_indices.len() != frame.frequencies.len() {
        return Err("Invalid radiative boundary dimensions/frequency axes".into());
    }
    harmonics::uniform_difference(coefficients, &[])?;
    let retained: BTreeSet<_> = retained_signed.iter().cloned().collect();
    if retained.iter().any(|q| q.len() != axes)
        || coefficients
            .iter()
            .any(|c| c.real.len() != d || c.imag.len() != d || !retained.contains(&c.index))
    {
        return Err("Radiative density coefficient is outside its retained lattice".into());
    }
    let channels = mapped_channels(frame, input_axis_indices, axes)?;
    let mut rho_extent = vec![0_i32; axes];
    let mut rho_max = 0.;
    let mut rho_squared = 0.;
    for c in coefficients {
        let multiplicity = if c.index.iter().all(|&q| q == 0) {
            1.
        } else {
            2.
        };
        let mut norm_squared = 0.;
        for i in 0..d {
            let weight = if i < n { 1. } else { 2. };
            norm_squared += weight * (c.real[i] * c.real[i] + c.imag[i] * c.imag[i]);
        }
        rho_max += multiplicity * norm_squared.sqrt();
        rho_squared += multiplicity * norm_squared;
        for (i, &q) in c.index.iter().enumerate() {
            rho_extent[i] = rho_extent[i].max(
                q.checked_abs()
                    .ok_or("Boundary density harmonic overflow")?,
            );
        }
    }
    let rho_rms = rho_squared.sqrt();
    let mut extent = rho_extent.clone();
    // A channel's common Fourier offset cancels in C rho C^dagger. Its
    // difference range, not its absolute carrier index, determines padding.
    for (channel, _, _) in &channels {
        for axis in 0..axes {
            let first = channel.keys().map(|q| q[axis]).min().unwrap();
            let last = channel.keys().map(|q| q[axis]).max().unwrap();
            extent[axis] = extent[axis].max(
                last.checked_sub(first)
                    .and_then(|w| w.checked_add(rho_extent[axis]))
                    .ok_or("Boundary convolution extent overflow")?,
            );
        }
    }
    let dimensions: Vec<_> = extent
        .iter()
        .map(|&q| {
            (q as usize)
                .checked_mul(2)
                .and_then(|x| x.checked_add(1))
                .and_then(|x| x.checked_next_power_of_two())
                .ok_or("Boundary FFT shape overflow")
        })
        .collect::<Result<_, _>>()?;
    let points = dimensions.iter().try_fold(1_usize, |a, &b| {
        a.checked_mul(b).ok_or("Boundary FFT point-count overflow")
    })?;
    let elements = points
        .checked_mul(d)
        .ok_or("Boundary FFT allocation overflow")?;
    if elements > 16_777_216 {
        return Err(format!("Complete boundary convolution needs {points} phase points and {elements} matrix elements, beyond the explicit workspace budget"));
    }
    let strides = strides(&dimensions);
    let mut density = vec![Z::default(); elements];
    for c in coefficients {
        let point = location(&c.index, &dimensions, &strides);
        for i in 0..d {
            density[point * d + i] = Z {
                re: c.real[i],
                im: c.imag[i],
            };
        }
        if c.index.iter().any(|&q| q != 0) {
            let negative: Vec<_> = c.index.iter().map(|q| -q).collect();
            let point = location(&negative, &dimensions, &strides);
            for i in 0..d {
                density[point * d + i] = Z {
                    re: c.real[i],
                    im: -c.imag[i],
                };
            }
        }
    }
    fft(&mut density, &dimensions, d, true);
    let mut rho = vec![Z::default(); elements];
    let mut imaginary_squared = 0.;
    let mut imaginary_max = 0_f64;
    for point in 0..points {
        let x = &density[point * d..(point + 1) * d];
        let r = &mut rho[point * d..(point + 1) * d];
        for i in 0..d {
            imaginary_squared += if i < n { 1. } else { 2. } * x[i].im * x[i].im;
            imaginary_max = imaginary_max.max(x[i].im.abs());
        }
        for i in 0..n {
            r[i + i * n].re = x[i].re;
        }
        let mut k = n;
        for i in 0..n {
            for j in i + 1..n {
                r[i + j * n] = Z {
                    re: x[k].re,
                    im: x[k + 1].re,
                };
                r[j + i * n] = Z {
                    re: x[k].re,
                    im: -x[k + 1].re,
                };
                k += 2;
            }
        }
    }
    drop(density);
    let fft_steps = dimensions
        .iter()
        .map(|n| n.trailing_zeros() as usize)
        .sum::<usize>();
    let fft_eta = gamma(32 * fft_steps + 16);
    let matrix_gamma = gamma(8 * n);
    let rho_error = fft_eta * rho_rms + (imaginary_squared / points as f64).sqrt();
    let mut phase_error = 0.;
    let mut forcing_rms = 0.;
    let mut forcing = vec![Z::default(); elements];
    let mut hermiticity = 0_f64;
    let mut gemms = 0;
    for (channel, source_norm, source_count) in &channels {
        let coefficient_rms = channel
            .values()
            .map(|c| frobenius(c).powi(2))
            .sum::<f64>()
            .sqrt();
        let merge_eta = gamma(*source_count + 1);
        let collapse_error = fft_eta * coefficient_rms + merge_eta * source_norm;
        let collapse_max_error = (fft_eta + merge_eta) * source_norm;
        let cmax = source_norm + collapse_max_error;
        phase_error += (4. * source_norm + 2. * collapse_max_error) * rho_max * collapse_error
            + 2. * cmax * cmax * rho_error
            + (4. * matrix_gamma + 2. * matrix_gamma * matrix_gamma + 8. * f64::EPSILON)
                * cmax
                * cmax
                * (rho_rms + rho_error);
        forcing_rms += 2. * cmax * cmax * (rho_rms + rho_error);
        let mut values = vec![Z::default(); elements];
        for (index, c) in channel {
            let point = location(index, &dimensions, &strides);
            for i in 0..d {
                values[point * d + i].re += c[i].re;
                values[point * d + i].im += c[i].im;
            }
        }
        fft(&mut values, &dimensions, d, true);
        for point in 0..points {
            let c = &values[point * d..(point + 1) * d];
            let r = &rho[point * d..(point + 1) * d];
            let cd = adjoint(c, n);
            let cr = lap.multiply_complex(c, r, n, n, n)?;
            let jump = lap.multiply_complex(&cr, &cd, n, n, n)?;
            let gram = lap.multiply_complex(&cd, c, n, n, n)?;
            let gr = lap.multiply_complex(&gram, r, n, n, n)?;
            gemms += 4;
            let output = &mut forcing[point * d..(point + 1) * d];
            for i in 0..n {
                output[i].re += jump[i + i * n].re - gr[i + i * n].re;
                hermiticity = hermiticity.max(jump[i + i * n].im.abs());
            }
            let mut k = n;
            for i in 0..n {
                for j in i + 1..n {
                    let a = i + j * n;
                    let b = j + i * n;
                    output[k].re += jump[a].re - 0.5 * (gr[a].re + gr[b].re);
                    output[k + 1].re += jump[a].im - 0.5 * (gr[a].im - gr[b].im);
                    k += 2;
                    hermiticity =
                        hermiticity.max((jump[a].re - jump[b].re).hypot(jump[a].im + jump[b].im));
                }
            }
        }
    }
    if forcing.iter().any(|z| !z.re.is_finite())
        || !phase_error.is_finite()
        || !forcing_rms.is_finite()
    {
        return Err("Radiative boundary arithmetic overflow".into());
    }
    fft(&mut forcing, &dimensions, d, false);
    let mut outside = Vec::new();
    let mut outside_signed = 0;
    for point in 0..points {
        let index = index_at(point, &dimensions, &strides);
        if index
            .iter()
            .zip(&extent)
            .any(|(&q, &bound)| q.abs() > bound)
            || retained.contains(&index)
        {
            continue;
        }
        outside_signed += 1;
        if sign(&index) < 0 {
            continue;
        }
        let value = &forcing[point * d..(point + 1) * d];
        outside.push(FourierCoefficient {
            index,
            real: value.iter().map(|z| z.re / points as f64).collect(),
            imag: value.iter().map(|z| z.im / points as f64).collect(),
        });
    }
    // Parseval: coefficient l2 error equals RMS phase error. Cauchy-Schwarz
    // contributes sqrt(number of signed outside modes), never N copies of a
    // uniform per-coefficient estimate. FFT and complex-product rounding are
    // included under the standard floating-point gamma_k arithmetic model.
    let roundoff = (outside_signed as f64).sqrt() * (phase_error + fft_eta * forcing_rms);
    if !roundoff.is_finite() {
        return Err("Boundary roundoff allowance overflow".into());
    }
    Ok(BoundaryEvaluation {
        coefficients: outside,
        roundoff_bound: roundoff,
        diagnostics: json!({"method":"complete zero-padded tensor FFT radiative convolution","fftDimensions":dimensions,"phasePoints":points,"outsideSignedCount":outside_signed,"matrixMultiplications":gemms,"mappedCollapseCoefficientCounts":channels.iter().map(|c|c.0.len()).collect::<Vec<_>>(),"retainedAllCollapseTerms":true,"densityImaginaryRoundoffMax":imaginary_max,"jumpHermiticityRoundoffMax":hermiticity,"fftRoundoffFactor":fft_eta,"complexProductRoundoffFactor":matrix_gamma,"phaseRmsRoundoffAllowance":phase_error,"outputFftRmsRoundoffAllowance":fft_eta*forcing_rms,"uniformRoundoffAllowance":roundoff,"roundoffModel":"gamma_k floating-point model; complex FFT twiddles use platform sin/cos; Parseval l2 to Fourier l1 conversion","elapsedMs":started.elapsed().as_secs_f64()*1000.}),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn frame(axes: usize) -> Frame {
        let n = 2;
        let a = vec![
            C::new(0.13, 0.02),
            C::new(0.7, 0.1),
            C::new(-0.2, 0.03),
            C::new(-0.1, 0.02),
        ];
        let b = vec![
            C::new(0.1, -0.07),
            C::new(-0.04, 0.02),
            C::new(0.3, 0.08),
            C::new(0.01, -0.04),
        ];
        let c = vec![
            C::new(-0.07, 0.02),
            C::new(0.1, -0.11),
            C::new(0.01, 0.06),
            C::new(0.15, 0.),
        ];
        Frame {
            dimension: n,
            frequencies: if axes == 2 {
                vec![1., 2_f64.sqrt()]
            } else {
                vec![1.]
            },
            gauges: vec![vec![0; axes]; n],
            order: 1,
            modes: BTreeMap::from([(
                vec![0; axes],
                vec![C::new(1., 0.), C::default(), C::default(), C::new(1., 0.)],
            )]),
            hamiltonian: vec![C::default(); 4],
            collapse: vec![BTreeMap::from([
                (vec![0; axes], a),
                (if axes == 2 { vec![1, -1] } else { vec![1] }, b),
                (if axes == 2 { vec![-1, 0] } else { vec![-1] }, c),
            ])],
            unitarity_bound: 0.,
            schrodinger_residual: 0.,
            schrodinger_residual_sum: 0.,
            minimum_central_weight: 1.,
            sambe_dimension: 0,
            preparation_ms: 0.,
            radiation_diagnostics: None,
        }
    }
    fn coefficient(q: Vec<i32>, scale: f64) -> FourierCoefficient {
        FourierCoefficient {
            index: q,
            real: vec![0.6, -0.6, 0.1, -0.07]
                .into_iter()
                .map(|x| x * scale)
                .collect(),
            imag: vec![0.04, -0.04, -0.03, 0.05]
                .into_iter()
                .map(|x| x * scale)
                .collect(),
        }
    }
    fn compare(
        frame: &Frame,
        coefficients: &[FourierCoefficient],
        retained: &[Vec<i32>],
        mapping: &[Vec<i32>],
    ) {
        let lap = Lapack::load().unwrap();
        let axes = coefficients[0].index.len();
        let channels = mapped_channels(frame, mapping, axes).unwrap();
        let mut reduced = crate::multitone_frame::Frame {
            dimension: 2,
            frequencies: vec![1.; axes],
            gauges: vec![vec![0; axes]; 2],
            order: 1,
            modes: BTreeMap::new(),
            hamiltonian: vec![C::default(); 4],
            collapse: vec![],
            unitarity_bound: 0.,
            schrodinger_residual: 0.,
            schrodinger_residual_sum: 0.,
            minimum_central_weight: 1.,
            sambe_dimension: 0,
            preparation_ms: 0.,
            radiation_diagnostics: None,
        };
        reduced.collapse = channels
            .into_iter()
            .map(|(channel, _, _)| {
                channel
                    .into_iter()
                    .map(|(q, a)| {
                        (
                            q,
                            (0..4)
                                .map(|i| {
                                    let z = a[i / 2 + (i % 2) * 2];
                                    C::new(z.re, z.im)
                                })
                                .collect(),
                        )
                    })
                    .collect()
            })
            .collect();
        let drives: Vec<_> = reduced
            .generator_indices()
            .into_iter()
            .map(|q| {
                let h = reduced.generator_harmonic(&lap, &q).unwrap();
                harmonics::FourierDrive {
                    index: q,
                    cos: h.cos,
                    sin: h.sin,
                }
            })
            .collect();
        let direct = harmonics::boundary_forcing(&drives, coefficients, retained).unwrap();
        let fast = evaluate(&lap, frame, coefficients, retained, mapping).unwrap();
        let difference = harmonics::uniform_difference(&direct.coefficients, &fast.coefficients)
            .unwrap()
            .max_coordinate;
        assert!(
            difference < 1e-12,
            "coefficient difference {difference:e}; {}",
            fast.diagnostics
        );
        assert!(
            difference <= fast.roundoff_bound,
            "roundoff allowance failed {difference:e} > {:e}",
            fast.roundoff_bound
        );
        assert!(
            direct.bound.max_coordinate > 1e-4,
            "fixture must exercise an actual outside harmonic"
        );
    }
    fn initial(axes: usize) -> FourierCoefficient {
        FourierCoefficient {
            index: vec![0; axes],
            real: vec![0.6, 0.4, 0.03, -0.02],
            imag: vec![0.; 4],
        }
    }
    #[test]
    fn two_axis_complete_complex_convolution_matches_all_kronecker_coefficients() {
        let f = frame(2);
        let c = vec![
            initial(2),
            coefficient(vec![1, 0], 0.02),
            coefficient(vec![0, 1], 0.015),
        ];
        let r = vec![vec![0, 0], vec![1, 0], vec![-1, 0], vec![0, 1], vec![0, -1]];
        compare(&f, &c, &r, &[vec![1, 0], vec![0, 1]]);
    }
    #[test]
    fn exact_commensurate_mapping_coalesces_before_dissipator() {
        let mut f = frame(2);
        f.collapse[0].insert(vec![2, -1], vec![C::new(0.1, 0.2); 4]);
        let c = vec![initial(1), coefficient(vec![1], 0.03)];
        compare(&f, &c, &[vec![0], vec![1], vec![-1]], &[vec![1], vec![2]]);
    }
    #[test]
    fn high_mixed_harmonics_are_zero_padded_not_aliased() {
        let mut f = frame(2);
        let a = f.collapse[0].remove(&vec![1, -1]).unwrap();
        let b = f.collapse[0].remove(&vec![-1, 0]).unwrap();
        f.collapse[0].insert(vec![3, -2], a);
        f.collapse[0].insert(vec![-3, 2], b);
        let c = vec![initial(2), coefficient(vec![1, 0], 0.03)];
        compare(
            &f,
            &c,
            &[vec![0, 0], vec![1, 0], vec![-1, 0]],
            &[vec![1, 0], vec![0, 1]],
        );
    }
    #[test]
    fn stationary_callback_preserves_the_full_mean_solution() {
        let lap = Lapack::load().unwrap();
        let mut f = frame(2);
        f.frequencies = vec![100., 2_f64.sqrt() * 100.];
        let model = crate::Model { frequency_coordinates:None,
            dimension: 2,
            states: vec![],
            h_real: vec![0.; 4],
            h_imag: vec![0.; 4],
            collapse: vec![],
            initial_index: 0,
            initial_real: None,
            initial_imag: None,
            time_dependent: false,
            drives: vec![],
        };
        let rho = vec![C::new(1., 0.), C::default(), C::default(), C::default()];
        let g = crate::build_generator(&f.base_model(&model, &rho)).unwrap();
        let input = crate::generator_mean::Input {
            n: 2,
            base: g.matrix,
            drives: vec![],
            frequencies: f.frequencies.clone(),
            initial: g.initial,
            hilbert_support: vec![0, 1],
            bounded_generator_tail: 0.,
        };
        let catalog = f.generator_indices();
        let provider = |index: &[i32]| {
            let h = f.generator_harmonic(&lap, index)?;
            Ok(harmonics::FourierDrive {
                index: h.index,
                cos: h.cos,
                sin: h.sin,
            })
        };
        let options = crate::generator_mean::Options {
            max_order: 12,
            ..Default::default()
        };
        let direct = crate::generator_mean::solve_streamed_with_options(
            &lap, &input, &catalog, provider, options,
        )
        .unwrap();
        let mut callback =
            |c: &[FourierCoefficient], r: &[Vec<i32>], m: &[Vec<i32>]| evaluate(&lap, &f, c, r, m);
        let fast = crate::generator_mean::solve_streamed_with_boundary(
            &lap,
            &input,
            &catalog,
            provider,
            &mut callback,
            options,
        )
        .unwrap();
        let error = harmonics::uniform_difference(&direct.coefficients, &fast.coefficients)
            .unwrap()
            .max_coordinate;
        assert!(
            error < 1e-10,
            "independently refined meshes disagree: {error:e}"
        );
        assert!(
            direct.diagnostics["boundaryForcingUniformMax"]
                .as_f64()
                .unwrap()
                < 1e-7
        );
        assert!(
            fast.diagnostics["boundaryForcingUniformMax"]
                .as_f64()
                .unwrap()
                < 1e-7
        );
    }
}
