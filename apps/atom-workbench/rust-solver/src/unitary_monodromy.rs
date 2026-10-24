//! Periodic Hilbert frame from a directly integrated unitary monodromy.
//! Intended for an EXACT invariant support of at most four internal states.
//! No Floquet replica is selected by central weight. A common energy origin
//! is subtracted before integration; it cancels from photon-energy differences.
//! All sampled P modes and every P^dagger D P product are retained. Products
//! use a zero-padded 2N transform, so their harmonics are not Nyquist-aliased
//! before the emitted-frequency spectral density is applied.
use crate::lapack::Complex64;
use crate::{
    adaptive_periodic,
    cluster_action::Sparse,
    complex_input,
    multitone_frame::Frame,
    periodic_action::Harmonic,
    radiative_bath::{self, Radiation},
    Lapack, Model, C,
};
use std::collections::BTreeMap;
use std::time::Instant;

fn adjoint(a: &[C], n: usize) -> Vec<C> {
    (0..n * n).map(|k| a[(k % n) * n + k / n].conj()).collect()
}
fn multiply(a: &[C], b: &[C], n: usize) -> Vec<C> {
    let mut c = vec![C::default(); n * n];
    for i in 0..n {
        for k in 0..n {
            for j in 0..n {
                c[i * n + j] += a[i * n + k] * b[k * n + j];
            }
        }
    }
    c
}
fn frobenius(a: &[C]) -> f64 {
    a.iter().fold(0., |x, z| x.hypot(z.abs()))
}
fn identity(n: usize) -> Vec<C> {
    let mut a = vec![C::default(); n * n];
    for i in 0..n {
        a[i * n + i] = C::new(1., 0.);
    }
    a
}
fn hermitian(a: &[C], n: usize) -> Result<(), String> {
    let scale = a.iter().map(|z| z.abs()).fold(1_f64, f64::max);
    for i in 0..n {
        for j in 0..n {
            if (a[i * n + j] - a[j * n + i].conj()).abs() > 1e-12 * scale {
                return Err("Monodromy Hamiltonian is not Hermitian".into());
            }
        }
    }
    Ok(())
}

/// Real form of U'= -i H U. Every complex matrix entry is retained, ordered
/// [Re U00, Im U00, Re U01, Im U01, ...], independently of density coordinates.
fn unitary_generator(h: &[C], n: usize) -> Sparse {
    let mut a = vec![vec![]; 2 * n * n];
    for i in 0..n {
        for k in 0..n {
            let z = h[i * n + k];
            for j in 0..n {
                let input = 2 * (k * n + j);
                let output = 2 * (i * n + j);
                if z.im != 0. {
                    a[input].push((output, z.im));
                    a[input + 1].push((output + 1, z.im));
                }
                if z.re != 0. {
                    a[input].push((output + 1, -z.re));
                    a[input + 1].push((output, z.re));
                }
            }
        }
    }
    a
}
fn unpack(x: &[f64]) -> Vec<C> {
    x.chunks_exact(2).map(|v| C::new(v[0], v[1])).collect()
}
fn column_major(a: &[C], n: usize) -> Vec<Complex64> {
    (0..n * n)
        .map(|k| {
            let z = a[(k % n) * n + k / n];
            Complex64 { re: z.re, im: z.im }
        })
        .collect()
}
fn row_major(a: &[Complex64], n: usize) -> Vec<C> {
    (0..n * n)
        .map(|k| {
            let z = a[k / n + (k % n) * n];
            C::new(z.re, z.im)
        })
        .collect()
}

/// Radix-two DFT. Equal constant samples cancel exactly in the butterflies;
/// no coefficient-magnitude threshold is used to manufacture sparse modes.
fn fft(values: &mut [Vec<C>], inverse: bool) {
    let n = values.len();
    let width = values[0].len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            values.swap(i, j);
        }
    }
    let mut span = 2;
    while span <= n {
        for start in (0..n).step_by(span) {
            for j in 0..span / 2 {
                let angle = 2. * std::f64::consts::PI * j as f64 / span as f64;
                let (s, c) = if j == 0 {
                    (0., 1.)
                } else if 4 * j == span {
                    (1., 0.)
                } else {
                    angle.sin_cos()
                };
                let twiddle = C::new(c, if inverse { s } else { -s });
                let (left, right) = values.split_at_mut(start + span / 2);
                let a = &mut left[start + j];
                let b = &mut right[j];
                for k in 0..width {
                    let x = a[k];
                    let y = b[k] * twiddle;
                    a[k] = x + y;
                    b[k] = x - y;
                }
            }
        }
        span *= 2;
    }
    if inverse {
        for row in values {
            for z in row {
                *z = *z * (1. / n as f64);
            }
        }
    }
}
fn modes(samples: Vec<Vec<C>>) -> BTreeMap<Vec<i32>, Vec<C>> {
    let mut values = samples;
    let n = values.len();
    fft(&mut values, false);
    let mut result = BTreeMap::new();
    for (k, mut value) in values.into_iter().enumerate() {
        for z in &mut value {
            *z = *z * (1. / n as f64);
        }
        if value.iter().any(|z| z.re != 0. || z.im != 0.) {
            let q = if k < n / 2 {
                k as i32
            } else {
                k as i32 - n as i32
            };
            result.insert(vec![q], value);
        }
    }
    result
}
fn padded_samples(modes: &BTreeMap<Vec<i32>, Vec<C>>, count: usize, width: usize) -> Vec<Vec<C>> {
    let mut values = vec![vec![C::default(); width]; count];
    for (q, coefficient) in modes {
        let index = q[0].rem_euclid(count as i32) as usize;
        for (i, &value) in coefficient.iter().enumerate() {
            values[index][i] += value * count as f64;
        }
    }
    fft(&mut values, true);
    values
}

fn diagonalize_unitary(lap: &Lapack, u: &[C], n: usize) -> Result<(Vec<f64>, Vec<C>, f64), String> {
    let ud = adjoint(u, n);
    let mut last = f64::INFINITY;
    // Re U and Im U commute for unitary U. A generic real combination avoids
    // the cos(theta)=cos(-theta) degeneracy of Re U alone. Validate the actual
    // transformed U, so an accidental degeneracy never silently selects HF.
    for alpha in [2_f64.sqrt(), std::f64::consts::PI, 1. / 3_f64.sqrt()] {
        let h: Vec<_> = u
            .iter()
            .zip(&ud)
            .map(|(&a, &b)| {
                let real = (a + b) * 0.5;
                let delta = a - b;
                real + C::new(delta.im, -delta.re) * (0.5 * alpha)
            })
            .collect();
        let (_, v) = lap.hermitian_eigen(&column_major(&h, n), n)?;
        let v = row_major(&v, n);
        let eigen = multiply(&multiply(&adjoint(&v, n), u, n), &v, n);
        let mut error = 0_f64;
        let mut phases = vec![0.; n];
        for i in 0..n {
            phases[i] = eigen[i * n + i].im.atan2(eigen[i * n + i].re);
            for j in 0..n {
                error = error.max(if i == j {
                    (eigen[i * n + j].abs() - 1.).abs()
                } else {
                    eigen[i * n + j].abs()
                });
            }
        }
        last = error;
        if error < 2e-10 {
            return Ok((phases, v, error));
        }
    }
    Err(format!(
        "Unitary monodromy could not be diagonalized consistently: {last:e}"
    ))
}

pub fn prepare(
    lap: &Lapack,
    model: &Model,
    radiation: &Radiation,
    grid: usize,
) -> Result<Frame, String> {
    let started = Instant::now();
    let n = model.dimension;
    if n == 0
        || n > 4
        || model.drives.len() != 1
        || grid < 16
        || grid > 4096
        || !grid.is_power_of_two()
    {
        return Err("Monodromy frame requires 1–4 exact support states, one beat, and a power-of-two grid in 16..4096".into());
    }
    let omega = model.drives[0].omega;
    if !omega.is_finite() || omega <= 0. {
        return Err("Invalid monodromy beat frequency".into());
    }
    let period = 2. * std::f64::consts::PI / omega;
    let mut h = complex_input(&model.h_real, &model.h_imag, n * n, "monodromy H0")?;
    let origin = (0..n).map(|i| h[i * n + i].re).sum::<f64>() / n as f64;
    for i in 0..n {
        h[i * n + i].re -= origin;
    }
    let cosine = complex_input(
        &model.drives[0].cos.real,
        &model.drives[0].cos.imag,
        n * n,
        "monodromy Hcos",
    )?;
    let sine = complex_input(
        &model.drives[0].sin.real,
        &model.drives[0].sin.imag,
        n * n,
        "monodromy Hsin",
    )?;
    for a in [&h, &cosine, &sine] {
        hermitian(a, n)?;
    }
    let base = unitary_generator(&h, n);
    let drive = Harmonic {
        index: 1,
        cos: unitary_generator(&cosine, n),
        sin: unitary_generator(&sine, n),
    };
    let mut initial = vec![0.; 2 * n * n];
    for i in 0..n {
        initial[2 * (i * n + i)] = 1.;
    }
    let times: Vec<_> = (0..=grid)
        .map(|i| period * i as f64 / grid as f64)
        .collect();
    let integrated = adaptive_periodic::integrate_verified(
        &base,
        &[drive],
        omega,
        &initial,
        &times,
        1e-12,
        2e-14,
        2e-10,
    )?;
    let final_u = unpack(integrated.action.states.last().unwrap());
    let mut unitary = multiply(&adjoint(&final_u, n), &final_u, n);
    for i in 0..n {
        unitary[i * n + i].re -= 1.;
    }
    let monodromy_error = frobenius(&unitary);
    if monodromy_error > 2e-10 {
        return Err(format!(
            "Integrated monodromy is not unitary to tolerance: {monodromy_error:e}"
        ));
    }
    let (phases, vectors, _) = diagonalize_unitary(lap, &final_u, n)?;
    let energies: Vec<_> = phases.iter().map(|p| -p / period).collect();
    let mut hf = vec![C::default(); n * n];
    for i in 0..n {
        hf[i * n + i].re = energies[i];
    }
    let mut samples = Vec::with_capacity(grid);
    for (i, x) in integrated.action.states[..grid].iter().enumerate() {
        let mut p = multiply(&unpack(x), &vectors, n);
        for column in 0..n {
            let (s, c) = (energies[column] * times[i]).sin_cos();
            let phase = C::new(c, s);
            for row in 0..n {
                p[row * n + column] = p[row * n + column] * phase;
            }
        }
        samples.push(p);
    }
    let p_modes = modes(samples);
    let padded = padded_samples(&p_modes, 2 * grid, n * n);
    let gram_samples: Vec<_> = padded
        .iter()
        .map(|p| {
            let mut a = multiply(&adjoint(p, n), p, n);
            for i in 0..n {
                a[i * n + i].re -= 1.;
            }
            a
        })
        .collect();
    let gram = modes(gram_samples);
    let unitarity_bound = gram.values().map(|a| frobenius(a)).sum::<f64>();
    if unitarity_bound > 1e-8 {
        return Err(format!("Monodromy periodic interpolation is unresolved at grid {grid}: unitarity {unitarity_bound:e}"));
    }
    let positive: Vec<_> = cosine
        .iter()
        .zip(&sine)
        .map(|(&a, &b)| (a + C::new(b.im, -b.re)) * 0.5)
        .collect();
    let negative = adjoint(&positive, n);
    let mut residual: BTreeMap<i32, Vec<C>> = BTreeMap::new();
    for (index, p) in &p_modes {
        let q = index[0];
        let hp = multiply(&h, p, n);
        let ph = multiply(p, &hf, n);
        let center = residual
            .entry(q)
            .or_insert_with(|| vec![C::default(); n * n]);
        for k in 0..n * n {
            center[k] += hp[k] + p[k] * (q as f64 * omega) - ph[k];
        }
        for (shift, a) in [(1, &positive), (-1, &negative)] {
            let value = multiply(a, p, n);
            let target = residual
                .entry(q + shift)
                .or_insert_with(|| vec![C::default(); n * n]);
            for k in 0..n * n {
                target[k] += value[k];
            }
        }
    }
    let residual_sum = residual.values().map(|a| frobenius(a)).sum::<f64>();
    let residual_l2 = residual
        .values()
        .map(|a| frobenius(a).powi(2))
        .sum::<f64>()
        .sqrt();
    if !residual_sum.is_finite() || residual_sum * period > 1e-8 {
        return Err(format!(
            "Monodromy Fourier frame fails the original equation at grid {grid}: residual*T={:e}",
            residual_sum * period
        ));
    }
    let mut channels = Vec::new();
    for raw in radiation.normalized_dipoles(n)? {
        let sampled = padded
            .iter()
            .map(|p| multiply(&multiply(&adjoint(p, n), &raw, n), p, n))
            .collect();
        channels.push(modes(sampled));
    }
    let weighted = radiative_bath::weight_fourier(lap, radiation, &hf, &[omega], &channels)?;
    let minimum_central_weight = p_modes
        .get(&vec![0])
        .map(|p| {
            (0..n)
                .map(|j| (0..n).map(|i| p[i * n + j].abs().powi(2)).sum::<f64>())
                .fold(1., f64::min)
        })
        .unwrap_or(0.);
    Ok(Frame {
        dimension: n,
        frequencies: vec![omega],
        gauges: vec![vec![0]; n],
        order: (grid / 2) as i32,
        modes: p_modes,
        hamiltonian: hf,
        collapse: weighted.channels,
        unitarity_bound,
        schrodinger_residual: residual_l2,
        schrodinger_residual_sum: residual_sum,
        minimum_central_weight,
        sambe_dimension: 0,
        preparation_ms: started.elapsed().as_secs_f64() * 1000.,
        radiation_diagnostics: Some(weighted.diagnostics),
    })
}

/// Complete already-weighted jumps in the original physical rotating frame.
/// Zero padding covers P C_F P^dagger's ENTIRE Fourier convolution. A caller
/// may subsequently bound small numerical tails, but this helper drops none.
pub fn physical_collapse(frame: &Frame) -> Result<Vec<radiative_bath::Fourier>, String> {
    let n = frame.dimension;
    if n == 0 || n > 4 || frame.frequencies.len() != 1 {
        return Err("Physical jump reconstruction requires a small single-frequency frame".into());
    }
    collapse_in_physical(&frame.physical_modes(), &frame.collapse, n)
}
fn collapse_in_physical(
    p: &BTreeMap<Vec<i32>, Vec<C>>,
    channels: &[radiative_bath::Fourier],
    n: usize,
) -> Result<Vec<radiative_bath::Fourier>, String> {
    let minimum = p
        .keys()
        .map(|q| q[0] as i64)
        .min()
        .ok_or("Empty periodic frame")?;
    let maximum = p.keys().map(|q| q[0] as i64).max().unwrap();
    let mut result = Vec::with_capacity(channels.len());
    for channel in channels {
        if channel.is_empty() {
            result.push(BTreeMap::new());
            continue;
        }
        let first = channel.keys().map(|q| q[0] as i64).min().unwrap() + minimum - maximum;
        let last = channel.keys().map(|q| q[0] as i64).max().unwrap() + maximum - minimum;
        let bound = first.unsigned_abs().max(last.unsigned_abs());
        if bound > 32767 {
            return Err(
                "Physical jump Fourier convolution exceeds the reconstruction budget".into(),
            );
        }
        let count = (2 * (bound as usize + 1)).next_power_of_two().max(16);
        let ps = padded_samples(p, count, n * n);
        let cs = padded_samples(channel, count, n * n);
        let values = ps
            .iter()
            .zip(cs)
            .map(|(p, c)| multiply(&multiply(p, &c, n), &adjoint(p, n), n))
            .collect();
        result.push(modes(values));
    }
    Ok(result)
}

/// Numerically stable evaluation of the EXACT unitary identity
/// C_phys = c0 D + P [C_F - c0 (P^dagger D P)] P^dagger.
/// The constant vacuum spectrum is transformed analytically before numerical
/// approximation; no normalization of P or rho is performed. The bath helper
/// evaluates the small spectral difference directly, without subtracting two
/// optical-size numbers or two nearly equal weighted matrices. All Fourier
/// coefficients of the remaining spectral correction are transformed.
pub fn physical_collapse_stable(
    lap: &Lapack,
    frame: &Frame,
    radiation: &Radiation,
) -> Result<Vec<radiative_bath::Fourier>, String> {
    let n = frame.dimension;
    if n == 0 || n > 4 || frame.frequencies.len() != 1 {
        return Err(
            "Stable physical jump reconstruction requires a small single-frequency frame".into(),
        );
    }
    let p = frame.physical_modes();
    let first = p
        .keys()
        .map(|q| q[0] as i64)
        .min()
        .ok_or("Empty periodic frame")?;
    let last = p.keys().map(|q| q[0] as i64).max().unwrap();
    let width = last - first;
    if width > 32767 {
        return Err("Stable physical jump convolution exceeds the reconstruction budget".into());
    }
    let count = (2 * (width as usize + 1)).next_power_of_two().max(16);
    let values = padded_samples(&p, count, n * n);
    let raw = radiation.normalized_dipoles(n)?;
    let channels: Vec<_> = raw
        .iter()
        .map(|d| {
            modes(
                values
                    .iter()
                    .map(|p| multiply(&multiply(&adjoint(p, n), d, n), p, n))
                    .collect(),
            )
        })
        .collect();
    let ratio = radiation.carrier_angular_frequency / radiation.reference_angular_frequency;
    let baseline = ratio * ratio.sqrt();
    let difference = radiative_bath::weight_fourier_difference(
        lap,
        radiation,
        &frame.hamiltonian,
        &frame.frequencies,
        &channels,
        baseline,
    )?;
    let mut physical = collapse_in_physical(&p, &difference.channels, n)?;
    for (channel, d) in physical.iter_mut().zip(raw) {
        if d.iter().any(|z| z.re != 0. || z.im != 0.) {
            let dc = channel
                .entry(vec![0])
                .or_insert_with(|| vec![C::default(); n * n]);
            for (a, b) in dc.iter_mut().zip(d) {
                *a += b * baseline;
            }
        }
    }
    Ok(physical)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Operator;
    fn model(a: f64, b: f64, omega: f64) -> (Model, Radiation) {
        let zero = Operator {
            real: vec![0.; 4],
            imag: vec![0.; 4],
        };
        let h = vec![1e6, a, a, 1e6];
        let m = Model { frequency_coordinates:None,
            dimension: 2,
            states: vec![],
            h_real: h,
            h_imag: vec![0.; 4],
            collapse: vec![],
            initial_index: 0,
            initial_real: None,
            initial_imag: None,
            time_dependent: true,
            drives: vec![crate::Drive {
                omega,
                cos: Operator {
                    real: vec![0., b, b, 0.],
                    imag: vec![0.; 4],
                },
                sin: zero.clone(),
            }],
        };
        let r = Radiation {
            reference_angular_frequency: 1e9,
            carrier_angular_frequency: 1e9,
            dipoles: vec![zero.clone(), zero.clone(), zero],
        };
        (m, r)
    }
    fn check(m: &Model, r: &Radiation, grid: usize) {
        let lap = Lapack::load().unwrap();
        let f = prepare(&lap, m, r, grid).unwrap();
        let initial = vec![C::new(1., 0.), C::default(), C::default(), C::default()];
        let sigma = f.into_frame(&initial);
        for t in [0., 0.13, 0.7, 1.3, 4.7, 20.] {
            let p = f.physical_frame(t);
            let e0 = f.hamiltonian[0].re;
            let e1 = f.hamiltonian[3].re;
            let (si, co) = ((e1 - e0) * t).sin_cos();
            let mut s = sigma.clone();
            s[1] = s[1] * C::new(co, si);
            s[2] = s[2] * C::new(co, -si);
            let rho = multiply(&multiply(&p, &s, 2), &adjoint(&p, 2), 2);
            let theta = m.h_real[1] * t
                + m.drives[0].cos.real[1] * (m.drives[0].omega * t).sin() / m.drives[0].omega;
            assert!(
                (rho[0].re - theta.cos().powi(2)).abs() < 2e-9,
                "t={t}, error={:e}",
                rho[0].re - theta.cos().powi(2)
            );
            assert!((rho[1].im - theta.sin() * theta.cos()).abs() < 2e-9);
        }
    }
    #[test]
    fn common_energy_and_multiple_quasienergy_branches() {
        let (m, r) = model(3.7, 0., 0.6);
        check(&m, &r, 128);
    }
    #[test]
    fn strong_slow_beat_matches_exact_commuting_unitary() {
        let (m, r) = model(1.3, 4., 0.6);
        check(&m, &r, 256);
    }
    #[test]
    fn undersampled_pure_phase_is_caught_by_original_equation() {
        let (m, r) = model(20., 0., 0.6);
        let lap = Lapack::load().unwrap();
        assert!(prepare(&lap, &m, &r, 16)
            .err()
            .unwrap()
            .contains("original equation"));
    }
    #[test]
    fn fft_constant_samples_have_exact_zero_nonzero_modes() {
        let a = vec![C::new(0.3, 0.2), C::new(-0.7, 0.4)];
        let m = modes(vec![a.clone(); 128]);
        assert_eq!(m.len(), 1);
        assert_eq!(m[&vec![0]][0].re, a[0].re);
    }
    #[test]
    fn physical_jump_reconstruction_keeps_all_fourier_cross_terms() {
        let (m, mut r) = model(1.3, 4., 0.6);
        r.dipoles[0].real[1] = 1.;
        let lap = Lapack::load().unwrap();
        let frame = prepare(&lap, &m, &r, 256).unwrap();
        let physical = physical_collapse(&frame).unwrap();
        let eval = |m: &radiative_bath::Fourier, t: f64| {
            let mut value = vec![C::default(); 4];
            for (q, c) in m {
                let (s, co) = (q[0] as f64 * 0.6 * t).sin_cos();
                for (i, &z) in c.iter().enumerate() {
                    value[i] += z * C::new(co, s);
                }
            }
            value
        };
        for j in 0..17 {
            let t = j as f64 * 0.137;
            let p = frame.physical_frame(t);
            let direct = multiply(
                &multiply(&p, &eval(&frame.collapse[0], t), 2),
                &adjoint(&p, 2),
                2,
            );
            let transformed = eval(&physical[0], t);
            assert!(direct
                .iter()
                .zip(transformed)
                .all(|(&a, b)| (a - b).abs() < 1e-11));
        }
    }
    #[test]
    fn stable_physical_jumps_agree_without_constant_spectrum_ghost_tail() {
        let (m, mut r) = model(1.3, 4., 0.6);
        r.dipoles[0].real[1] = 1.;
        let lap = Lapack::load().unwrap();
        let frame = prepare(&lap, &m, &r, 256).unwrap();
        let direct = physical_collapse(&frame).unwrap();
        let stable = physical_collapse_stable(&lap, &frame, &r).unwrap();
        let mut difference = 0.;
        for (q, c) in &direct[0] {
            for i in 0..4 {
                difference += (c[i] - stable[0].get(q).map(|x| x[i]).unwrap_or_default()).abs();
            }
        }
        for (q, c) in &stable[0] {
            if !direct[0].contains_key(q) {
                difference += c.iter().map(|z| z.abs()).sum::<f64>();
            }
        }
        assert!(
            difference < 1e-9,
            "equivalent unitary formulations disagree: {difference:e}"
        );
        let tail = |m: &radiative_bath::Fourier| {
            m.iter()
                .filter(|(q, _)| q[0].abs() > 2)
                .flat_map(|(_, x)| x.iter())
                .map(|z| z.abs())
                .sum::<f64>()
        };
        assert!(
            tail(&stable[0]) < tail(&direct[0]) * 0.01,
            "stable spectral difference did not suppress constant-spectrum roundoff"
        );
    }
}
