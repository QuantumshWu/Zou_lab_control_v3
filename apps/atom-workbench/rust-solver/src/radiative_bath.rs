//! Common-dipole, nonsecular radiative bath in a complete Floquet basis.
//!
//! Nathan & Rudner, Phys. Rev. B 102, 115109 (2020), Sec. IV / Eq. (35):
//! https://arxiv.org/abs/2004.01469 . With our exp(+i k.Omega t) convention,
//! emission has omega=carrier+epsilon_b-epsilon_a-k.Omega. The zero-temperature
//! free-space spectral density is proportional to omega^3 for omega>0.
//!
//! Each polarization is ONE bath channel shared by D1 and D2. Its amplitudes
//! are summed before D[C]; virtual intermediate families are not reservoirs.
//! All signed Fourier coefficients, matrix entries and cross terms are kept.
//! Optical RWA and the atomic Hilbert-space truncation belong to the supplied
//! Hamiltonian. Static measured energies contain their static radiative shift;
//! a drive-dependent Lamb-shift correction is not implemented here.
use super::{
    complex_input,
    lapack::{Complex64, Lapack},
    Model, Operator, C,
};
use serde::Deserialize;
use std::collections::{BTreeMap, VecDeque};
use std::time::Instant;

#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct Radiation {
    /// Angular frequencies in rad/us; dipoles are normalized to reference.
    pub reference_angular_frequency: f64,
    pub carrier_angular_frequency: f64,
    /// Three reference-normalized lowering matrices, row-major, 1/sqrt(us).
    pub dipoles: Vec<Operator>,
}
pub(super) type Fourier = BTreeMap<Vec<i32>, Vec<C>>;
#[derive(Clone, Debug)]
pub(super) struct Diagnostics {
    pub dimension: usize,
    pub polarization_channels: usize,
    pub fourier_coefficients: usize,
    pub minimum_candidate_emission_frequency: f64,
    pub maximum_candidate_emission_frequency: f64,
    pub nonpositive_frequency_entries: usize,
    pub largest_spectral_amplitude_weight: f64,
    pub hamiltonian_components: Vec<usize>,
    pub retained_component_blocks: usize,
    pub preparation_ms: f64,
}
pub(super) struct WeightedBath {
    pub channels: Vec<Fourier>,
    pub diagnostics: Diagnostics,
}

#[derive(Clone, Copy, Default)]
struct DD {
    hi: f64,
    lo: f64,
}
impl DD {
    fn add(&mut self, hi: f64, lo: f64) {
        let s = self.hi + hi;
        let z = s - self.hi;
        let e = (self.hi - (s - z)) + (hi - z);
        let tail = self.lo + lo + e;
        let value = s + tail;
        self.lo = tail - (value - s);
        self.hi = value;
    }
    fn product(&mut self, a: f64, b: f64) {
        let p = a * b;
        self.add(p, a.mul_add(b, -p));
    }
    fn value(self) -> f64 {
        self.hi + self.lo
    }
}
fn column_major(a: &[C], n: usize) -> Vec<Complex64> {
    (0..n * n)
        .map(|k| {
            let z = a[(k % n) * n + k / n];
            Complex64 { re: z.re, im: z.im }
        })
        .collect()
}
fn adjoint_column(a: &[Complex64], n: usize) -> Vec<Complex64> {
    (0..n * n)
        .map(|k| {
            let z = a[k / n + (k % n) * n];
            Complex64 {
                re: z.re,
                im: -z.im,
            }
        })
        .collect()
}
struct Component {
    indices: Vec<usize>,
    energies: Vec<f64>,
    vectors: Vec<Complex64>,
    inverse: Vec<Complex64>,
}
fn exact_components(lap: &Lapack, h: &[C], n: usize) -> Result<Vec<Component>, String> {
    let mut seen = vec![false; n];
    let mut result = vec![];
    for start in 0..n {
        if seen[start] {
            continue;
        }
        seen[start] = true;
        let mut queue = VecDeque::from([start]);
        let mut indices = vec![];
        while let Some(i) = queue.pop_front() {
            indices.push(i);
            for j in 0..n {
                let value = h[i * n + j];
                if !seen[j] && (value.re != 0. || value.im != 0.) {
                    seen[j] = true;
                    queue.push_back(j);
                }
            }
        }
        indices.sort_unstable();
        let size = indices.len();
        let matrix: Vec<_> = indices
            .iter()
            .flat_map(|&i| indices.iter().map(move |&j| h[i * n + j]))
            .collect();
        let (energies, vectors) = lap.hermitian_eigen(&column_major(&matrix, size), size)?;
        let inverse = adjoint_column(&vectors, size);
        result.push(Component {
            indices,
            energies,
            vectors,
            inverse,
        });
    }
    Ok(result)
}

impl Radiation {
    pub fn normalized_dipoles(&self, n: usize) -> Result<Vec<Vec<C>>, String> {
        if n == 0
            || !self.reference_angular_frequency.is_finite()
            || self.reference_angular_frequency <= 0.
            || !self.carrier_angular_frequency.is_finite()
            || self.carrier_angular_frequency <= 0.
            || self.dipoles.len() != 3
        {
            return Err("Radiative bath requires positive optical frequencies and exactly three common dipole channels".into());
        }
        self.dipoles
            .iter()
            .map(|d| complex_input(&d.real, &d.imag, n * n, "reference-normalized dipole"))
            .collect()
    }
}

/// channels must already use the PHYSICAL Floquet frame, including any
/// integer state-frequency gauge. hamiltonian and all matrices are row-major.
pub(super) fn weight_fourier(
    lap: &Lapack,
    radiation: &Radiation,
    hamiltonian: &[C],
    frequencies: &[f64],
    channels: &[Fourier],
) -> Result<WeightedBath, String> {
    weight_internal(lap, radiation, hamiltonian, frequencies, channels, None)
}

/// Spectral correction relative to a known constant amplitude. This evaluates
/// f(carrier+shift)-f(carrier) through log1p/expm1, preserving corrections far
/// below one optical-frequency ULP. Together with the exact constant dipole
/// this is the SAME common-channel bath, not a low-order spectral expansion.
pub(super) fn weight_fourier_difference(
    lap: &Lapack,
    radiation: &Radiation,
    hamiltonian: &[C],
    frequencies: &[f64],
    channels: &[Fourier],
    baseline_scale: f64,
) -> Result<WeightedBath, String> {
    if !baseline_scale.is_finite() {
        return Err("Non-finite radiative baseline amplitude".into());
    }
    weight_internal(
        lap,
        radiation,
        hamiltonian,
        frequencies,
        channels,
        Some(baseline_scale),
    )
}

fn weight_internal(
    lap: &Lapack,
    radiation: &Radiation,
    hamiltonian: &[C],
    frequencies: &[f64],
    channels: &[Fourier],
    baseline: Option<f64>,
) -> Result<WeightedBath, String> {
    let started = Instant::now();
    let n = (hamiltonian.len() as f64).sqrt() as usize;
    if n == 0
        || n * n != hamiltonian.len()
        || channels.len() != 3
        || frequencies.iter().any(|w| !w.is_finite() || *w <= 0.)
    {
        return Err("Invalid complete Floquet radiative-bath dimensions/frequencies".into());
    }
    radiation.normalized_dipoles(n)?;
    for i in 0..n {
        for j in 0..n {
            let a = hamiltonian[i * n + j];
            if !a.re.is_finite()
                || !a.im.is_finite()
                || (a - hamiltonian[j * n + i].conj()).abs() > 1e-11 * (1. + a.abs())
            {
                return Err("Floquet bath Hamiltonian must be finite and Hermitian".into());
            }
        }
    }
    let components = exact_components(lap, hamiltonian, n)?;
    let mut output = Vec::with_capacity(3);
    let mut minimum = f64::INFINITY;
    let mut maximum = f64::NEG_INFINITY;
    let mut nonpositive = 0;
    let mut largest = 0_f64;
    let mut count = 0;
    let mut retained_blocks = 0;
    for channel in channels {
        let mut weighted = Fourier::new();
        for (index, matrix) in channel {
            if index.len() != frequencies.len()
                || matrix.len() != n * n
                || matrix
                    .iter()
                    .any(|z| !z.re.is_finite() || !z.im.is_finite())
            {
                return Err("Invalid signed Fourier dipole coefficient".into());
            }
            let mut result = vec![C::default(); n * n];
            for row in &components {
                for col in &components {
                    // An absent source->target component block is an exact selection
                    // rule. Never create it through a global degenerate eigenspace.
                    if !row.indices.iter().any(|&i| {
                        col.indices.iter().any(|&j| {
                            let z = matrix[i * n + j];
                            z.re != 0. || z.im != 0.
                        })
                    }) {
                        continue;
                    }
                    retained_blocks += 1;
                    let rows = row.indices.len();
                    let columns = col.indices.len();
                    let block: Vec<_> = col
                        .indices
                        .iter()
                        .flat_map(|&j| {
                            row.indices.iter().map(move |&i| {
                                let z = matrix[i * n + j];
                                Complex64 { re: z.re, im: z.im }
                            })
                        })
                        .collect();
                    let right =
                        lap.multiply_complex(&block, &col.vectors, rows, columns, columns)?;
                    let mut eigen =
                        lap.multiply_complex(&row.inverse, &right, rows, columns, rows)?;
                    for b in 0..columns {
                        for a in 0..rows {
                            let mut shift = DD::default();
                            shift.add(col.energies[b], 0.);
                            shift.add(-row.energies[a], 0.);
                            for (&k, &omega) in index.iter().zip(frequencies) {
                                shift.product(-(k as f64), omega);
                            }
                            let mut photon = DD::default();
                            photon.add(radiation.carrier_angular_frequency, 0.);
                            photon.add(shift.hi, shift.lo);
                            let emission = photon.value();
                            if !emission.is_finite() {
                                return Err("Emitted-photon frequency overflowed".into());
                            }
                            minimum = minimum.min(emission);
                            maximum = maximum.max(emission);
                            let weight = if emission <= 0. {
                                nonpositive += 1;
                                -baseline.unwrap_or(0.)
                            } else if let Some(base) = baseline {
                                let ratio = radiation.carrier_angular_frequency
                                    / radiation.reference_angular_frequency;
                                let carrier_scale = ratio * ratio.sqrt();
                                carrier_scale
                                    * (1.5
                                        * (shift.value() / radiation.carrier_angular_frequency)
                                            .ln_1p())
                                    .exp_m1()
                                    + (carrier_scale - base)
                            } else {
                                let ratio = emission / radiation.reference_angular_frequency;
                                ratio * ratio.sqrt()
                            };
                            if !weight.is_finite() {
                                return Err("Vacuum spectral-density weight overflowed".into());
                            }
                            largest = largest.max(weight.abs());
                            let k = a + b * rows;
                            eigen[k].re *= weight;
                            eigen[k].im *= weight;
                        }
                    }
                    let left = lap.multiply_complex(&row.vectors, &eigen, rows, columns, rows)?;
                    let value =
                        lap.multiply_complex(&left, &col.inverse, rows, columns, columns)?;
                    if value.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) {
                        return Err("Non-finite dressed radiative jump".into());
                    }
                    for b in 0..columns {
                        for a in 0..rows {
                            let z = value[a + b * rows];
                            result[row.indices[a] * n + col.indices[b]] = C::new(z.re, z.im);
                        }
                    }
                }
            }
            weighted.insert(index.clone(), result);
            count += 1;
        }
        output.push(weighted);
    }
    if retained_blocks == 0 {
        minimum = 0.;
        maximum = 0.;
    }
    Ok(WeightedBath {
        channels: output,
        diagnostics: Diagnostics {
            dimension: n,
            polarization_channels: 3,
            fourier_coefficients: count,
            minimum_candidate_emission_frequency: minimum,
            maximum_candidate_emission_frequency: maximum,
            nonpositive_frequency_entries: nonpositive,
            largest_spectral_amplitude_weight: largest,
            hamiltonian_components: components.iter().map(|c| c.indices.len()).collect(),
            retained_component_blocks: retained_blocks,
            preparation_ms: started.elapsed().as_secs_f64() * 1000.,
        },
    })
}

/// Constant rotating-frame H is itself a valid dressed representation of the
/// monochromatically driven lab Hamiltonian. This includes laser-off models.
pub(super) fn constant_dressed(
    lap: &Lapack,
    model: &Model,
    radiation: &Radiation,
) -> Result<(Model, Diagnostics), String> {
    if !model.drives.is_empty() || model.time_dependent {
        return Err(
            "A time-dependent optical model requires the complete Floquet radiative bath".into(),
        );
    }
    let h = complex_input(
        &model.h_real,
        &model.h_imag,
        model.dimension * model.dimension,
        "radiative Hamiltonian",
    )?;
    let raw = radiation.normalized_dipoles(model.dimension)?;
    let channels: Vec<_> = raw
        .into_iter()
        .map(|d| BTreeMap::from([(Vec::new(), d)]))
        .collect();
    let weighted = weight_fourier(lap, radiation, &h, &[], &channels)?;
    let mut result = model.clone();
    result.collapse = weighted
        .channels
        .iter()
        .map(|c| {
            let value = &c[&Vec::<i32>::new()];
            Operator {
                real: value.iter().map(|z| z.re).collect(),
                imag: value.iter().map(|z| z.im).collect(),
            }
        })
        .collect();
    Ok((result, weighted.diagnostics))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn spectral_difference_retains_sub_ulp_emission_correction() {
        let lap = Lapack::load().unwrap();
        let zero = Operator {
            real: vec![0.; 4],
            imag: vec![0.; 4],
        };
        let mut dipole = zero.clone();
        dipole.real[1] = 1.;
        let radiation = Radiation {
            reference_angular_frequency: 1e9,
            carrier_angular_frequency: 1e9,
            dipoles: vec![dipole, zero.clone(), zero],
        };
        let raw = radiation.normalized_dipoles(2).unwrap();
        let channels: Vec<_> = raw
            .into_iter()
            .map(|v| BTreeMap::from([(vec![], v)]))
            .collect();
        let h = vec![C::default(), C::default(), C::default(), C::new(1e-9, 0.)];
        let ordinary = weight_fourier(&lap, &radiation, &h, &[], &channels).unwrap();
        assert_eq!(ordinary.channels[0][&vec![]][1].re, 1.);
        let correction =
            weight_fourier_difference(&lap, &radiation, &h, &[], &channels, 1.).unwrap();
        assert!((correction.channels[0][&vec![]][1].re - 1.5e-18).abs() < 1e-32);
        let h = vec![C::default(), C::default(), C::default(), C::new(-2e9, 0.)];
        let correction =
            weight_fourier_difference(&lap, &radiation, &h, &[], &channels, 1.).unwrap();
        assert_eq!(correction.channels[0][&vec![]][1].re, -1.);
    }
    fn op(matrix: &[C]) -> Operator {
        Operator {
            real: matrix.iter().map(|c| c.re).collect(),
            imag: matrix.iter().map(|c| c.im).collect(),
        }
    }
    fn radiation(n: usize, carrier: f64, reference: f64, d: Vec<C>) -> Radiation {
        Radiation {
            reference_angular_frequency: reference,
            carrier_angular_frequency: carrier,
            dipoles: vec![
                op(&d),
                op(&vec![C::default(); n * n]),
                op(&vec![C::default(); n * n]),
            ],
        }
    }
    #[test]
    fn free_decay_uses_bare_emitted_frequency_not_laser_frequency() {
        let lap = Lapack::load().unwrap();
        let mut d = vec![C::default(); 4];
        d[1] = C::new(2., 0.);
        let bath = radiation(2, 900., 1000., d.clone());
        let h = vec![C::default(), C::default(), C::default(), C::new(100., 0.)];
        let channels = vec![
            BTreeMap::from([(vec![], d)]),
            BTreeMap::from([(vec![], vec![C::default(); 4])]),
            BTreeMap::from([(vec![], vec![C::default(); 4])]),
        ];
        let result = weight_fourier(&lap, &bath, &h, &[], &channels).unwrap();
        assert!((result.channels[0][&vec![]][1].re - 2.).abs() < 1e-14);
        assert_eq!(result.diagnostics.polarization_channels, 3);
    }
    #[test]
    fn fourier_sign_and_integer_energy_gauge_preserve_photon_frequency() {
        let lap = Lapack::load().unwrap();
        let mut d = vec![C::default(); 4];
        d[1] = C::new(2., 0.);
        let bath = radiation(2, 900., 1000., d.clone());
        let channels = vec![
            BTreeMap::from([(vec![1], d)]),
            BTreeMap::new(),
            BTreeMap::new(),
        ];
        let h = vec![C::default(), C::default(), C::default(), C::new(150., 0.)];
        let result = weight_fourier(&lap, &bath, &h, &[50.], &channels).unwrap();
        assert!((result.channels[0][&vec![1]][1].re - 2.).abs() < 1e-14);
    }
    #[test]
    fn negative_frequency_has_no_zero_temperature_vacuum_emission() {
        let lap = Lapack::load().unwrap();
        let mut d = vec![C::default(); 4];
        d[1] = C::new(1., 0.);
        let bath = radiation(2, 2., 3., d.clone());
        let channels = vec![
            BTreeMap::from([(vec![2], d)]),
            BTreeMap::new(),
            BTreeMap::new(),
        ];
        let h = vec![C::default(), C::default(), C::default(), C::new(1., 0.)];
        let result = weight_fourier(&lap, &bath, &h, &[2.], &channels).unwrap();
        assert!(result.channels[0][&vec![2]]
            .iter()
            .all(|z| z.re == 0. && z.im == 0.));
        assert!(result.diagnostics.nonpositive_frequency_entries > 0);
    }
    #[test]
    fn common_virtual_paths_recover_weak_drive_raman_interference() {
        let lap = Lapack::load().unwrap();
        let n = 4;
        let carrier = 1e9;
        let mut d = vec![C::default(); n * n];
        for (i, j, v) in [(0, 2, 1.), (0, 3, 1.), (1, 2, 1.), (1, 3, -1.)] {
            d[i * n + j] = C::new(v, 0.);
        }
        let bath = radiation(n, carrier, carrier, d.clone());
        let mut h = vec![C::default(); n * n];
        h[1 * n + 1] = C::new(-1., 0.);
        h[2 * n + 2] = C::new(1e5, 0.);
        h[3 * n + 3] = C::new(2e5, 0.);
        let drive = 1.;
        for e in [2, 3] {
            h[e] = C::new(drive, 0.);
            h[e * n] = C::new(drive, 0.);
        }
        let channels = vec![
            BTreeMap::from([(vec![], d)]),
            BTreeMap::new(),
            BTreeMap::new(),
        ];
        let result = weight_fourier(&lap, &bath, &h, &[], &channels).unwrap();
        let (_, u) = lap.hermitian_eigen(&column_major(&h, n), n).unwrap();
        let ug = (0..n)
            .max_by(|&a, &b| {
                let aa = u[a * n].re.powi(2) + u[a * n].im.powi(2);
                let bb = u[b * n].re.powi(2) + u[b * n].im.powi(2);
                aa.total_cmp(&bb)
            })
            .unwrap();
        let uf = (0..n)
            .max_by(|&a, &b| {
                let aa = u[1 + a * n].re.powi(2) + u[1 + a * n].im.powi(2);
                let bb = u[1 + b * n].re.powi(2) + u[1 + b * n].im.powi(2);
                aa.total_cmp(&bb)
            })
            .unwrap();
        let inverse = adjoint_column(&u, n);
        let right = lap
            .multiply_complex(&column_major(&result.channels[0][&vec![]], n), &u, n, n, n)
            .unwrap();
        let eigen = lap.multiply_complex(&inverse, &right, n, n, n).unwrap();
        let amplitude = eigen[uf + ug * n];
        let actual = amplitude.re.powi(2) + amplitude.im.powi(2);
        let expected = (drive / 1e5 - drive / 2e5).powi(2) * ((carrier + 1.) / carrier).powi(3);
        assert!((actual / expected - 1.).abs() < 1e-7);
    }
    #[test]
    fn complex_atomic_phase_gauge_is_covariant() {
        let lap = Lapack::load().unwrap();
        let phase = C::new(0.731_f64.cos(), 0.731_f64.sin());
        let u = [C::new(1., 0.), phase];
        let transform =
            |a: &[C]| -> Vec<C> { (0..4).map(|k| u[k / 2] * a[k] * u[k % 2].conj()).collect() };
        let h = vec![
            C::default(),
            C::new(0.31, 0.14),
            C::new(0.31, -0.14),
            C::new(7., 0.),
        ];
        let d = vec![C::default(), C::new(1., 0.2), C::default(), C::default()];
        let bath = radiation(2, 1000., 1000., d.clone());
        let original = weight_fourier(
            &lap,
            &bath,
            &h,
            &[],
            &vec![
                BTreeMap::from([(vec![], d.clone())]),
                BTreeMap::new(),
                BTreeMap::new(),
            ],
        )
        .unwrap();
        let gauged = weight_fourier(
            &lap,
            &bath,
            &transform(&h),
            &[],
            &vec![
                BTreeMap::from([(vec![], transform(&d))]),
                BTreeMap::new(),
                BTreeMap::new(),
            ],
        )
        .unwrap();
        let expected = transform(&original.channels[0][&vec![]]);
        for (a, b) in gauged.channels[0][&vec![]].iter().zip(expected) {
            assert!((*a - b).abs() < 2e-14);
        }
    }
    #[test]
    fn exact_disconnected_dipole_blocks_remain_zero_even_with_degeneracy() {
        let lap = Lapack::load().unwrap();
        let n = 4;
        let mut h = vec![C::default(); 16];
        for (i, j) in [(0, 2), (1, 3)] {
            h[i * n + j] = C::new(0.2, 0.1);
            h[j * n + i] = C::new(0.2, -0.1);
            h[j * n + j] = C::new(7., 0.);
        }
        let mut d = vec![C::default(); 16];
        d[2] = C::new(1., 0.);
        let bath = radiation(n, 1000., 1000., d.clone());
        let channels = vec![
            BTreeMap::from([(vec![], d)]),
            BTreeMap::new(),
            BTreeMap::new(),
        ];
        let result = weight_fourier(&lap, &bath, &h, &[], &channels).unwrap();
        assert_eq!(result.diagnostics.hamiltonian_components, vec![2, 2]);
        for i in 0..n {
            for j in 0..n {
                if [1, 3].contains(&i) || [1, 3].contains(&j) {
                    let value = result.channels[0][&vec![]][i * n + j];
                    assert_eq!(value.re, 0.);
                    assert_eq!(value.im, 0.);
                }
            }
        }
        h[3] = C::new(1e-24, 0.);
        h[12] = C::new(1e-24, 0.);
        let connected = weight_fourier(&lap, &bath, &h, &[], &channels).unwrap();
        assert_eq!(connected.diagnostics.hamiltonian_components, vec![4]);
    }
}
