//! Covariance validity is read per parameter from the information
//! equilibrated by its diagonal: independent of the parameters' units, and
//! an undetermined parameter loses its own error only.
use std::f64::consts::PI;
use std::sync::Arc;

use zlc_plot::fit::identify;
use zlc_plot::fit::*;

/// A deterministic normal sequence (xorshift + Box-Muller).
struct Noise(u64);

impl Noise {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 { (-2.0 * self.uniform().ln()).sqrt() * (2.0 * PI * self.uniform()).cos() }
}

const AMPLITUDE: f64 = 0.08;
const OFFSET: f64 = 0.1;
const FREQUENCY: f64 = 0.0265; // per microsecond
const PHASE: f64 = -1.7;

/// A Rabi scan of one site: 50 durations over 0.1-150 us, each point the
/// mean of 11-99 repeats with its standard error (dof n - 1), and a decay
/// far beyond the window -- none visible.  `unit` is the x unit in us.
fn rabi_cell(seed: u64, unit: f64) -> FitCell {
    let mut rng = Noise(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let x: Vec<f64> = (0..50).map(|i| 0.1 + 3.06 * i as f64).collect();
    let repeats: Vec<f64> = (0..50).map(|_| (11.0 + 88.0 * rng.uniform()).floor()).collect();
    let truth = |t: f64| AMPLITUDE * (2.0 * PI * FREQUENCY * (t - 0.1) + PHASE).sin() + OFFSET;
    let sem: Vec<f64> = x.iter().zip(&repeats).map(|(&t, &n)| (truth(t) * (1.0 - truth(t)) / n).sqrt() * (0.6 + 0.8 * rng.uniform())).collect();
    let y: Vec<f64> = x.iter().zip(&sem).map(|(&t, &s)| truth(t) + s * rng.normal()).collect();
    let mut cell = FitCell::points(vec![x.iter().map(|t| t / unit).collect()], y);
    cell.sigma = Some(Arc::from(sem));
    cell.sigma_dof = Some(Arc::from(repeats.iter().map(|n| n - 1.0).collect::<Vec<f64>>()));
    cell
}

fn fit(cell: &FitCell) -> FitResult {
    let engine = FitEngine::default();
    let model = engine.registry.get("damped_sine").unwrap();
    engine.fit(&model, cell, &FitRequest { initial: None, bounds: None, options: None, cancelled: None }).unwrap()
}

/// The Student-t sandwich at `p`, built here independently: analytic
/// Jacobian, `psi`/`psi'` weights, the bread inverted through its
/// correlation form by Gauss-Jordan.  Standard errors, `[A, B, f, tau, phi]`,
/// and whether the raw-unit rank test (`EPSILON * n * largest` on the raw
/// bread's eigenvalues) calls the bread singular.
fn sandwich_errors(cell: &FitCell, p: &[f64]) -> (Vec<f64>, bool) {
    let CellData::Points { coordinates, observations } = &cell.data else { unreachable!() };
    let (x, y) = (&coordinates[0], observations.as_ref().unwrap());
    let (sigma, dof) = (cell.sigma.as_ref().unwrap(), cell.sigma_dof.as_ref().unwrap());
    let origin = x.iter().copied().fold(f64::INFINITY, f64::min);
    let (n, k) = (x.len(), 5);
    let (mut bread, mut meat) = (vec![0.0; k * k], vec![0.0; k * k]);
    for i in 0..n {
        let s = x[i] - origin;
        let (decay, argument) = ((-s / p[3]).exp(), 2.0 * PI * p[2] * s + p[4]);
        let row = [argument.sin() * decay, 1.0, p[0] * argument.cos() * decay * 2.0 * PI * s, p[0] * argument.sin() * decay * s / (p[3] * p[3]), p[0] * argument.cos() * decay];
        let t = (y[i] - (p[0] * argument.sin() * decay + p[1])) / sigma[i];
        let nu = dof[i];
        let (score, slope) = (t * (nu + 1.0) / (nu + t * t), (nu + 1.0) * (nu - t * t) / ((nu + t * t) * (nu + t * t)));
        for a in 0..k {
            for b in 0..k {
                let jj = row[a] * row[b] / (sigma[i] * sigma[i]);
                bread[a * k + b] += slope * jj;
                meat[a * k + b] += score * score * jj;
            }
        }
    }
    let (values, _) = zlc_plot::fit::linalg::eigh_lower(k, &bread).unwrap();
    let largest = values.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let raw_singular = values.iter().any(|&v| v <= f64::EPSILON * n as f64 * largest);
    let d: Vec<f64> = (0..k).map(|a| bread[a * k + a].sqrt()).collect();
    let mut m: Vec<f64> = (0..k * k).map(|i| bread[i] / (d[i / k] * d[i % k])).collect();
    let mut inverse: Vec<f64> = (0..k * k).map(|i| if i / k == i % k { 1.0 } else { 0.0 }).collect();
    for c in 0..k {
        let pivot = m[c * k + c];
        for j in 0..k {
            m[c * k + j] /= pivot;
            inverse[c * k + j] /= pivot;
        }
        for r in (0..k).filter(|&r| r != c) {
            let factor = m[r * k + c];
            for j in 0..k {
                m[r * k + j] -= factor * m[c * k + j];
                inverse[r * k + j] -= factor * inverse[c * k + j];
            }
        }
    }
    let a_inv: Vec<f64> = (0..k * k).map(|i| inverse[i] / (d[i / k] * d[i % k])).collect();
    let correction = n as f64 / (n - k) as f64;
    let errors = (0..k)
        .map(|a| {
            let mut value = 0.0;
            for c in 0..k {
                for e in 0..k {
                    value += a_inv[a * k + c] * meat[c * k + e] * a_inv[e * k + a];
                }
            }
            (value * correction).sqrt()
        })
        .collect();
    (errors, raw_singular)
}

#[test]
fn an_invisible_decay_keeps_every_error_finite() {
    let mut raw_singular = 0;
    for seed in 1..=6 {
        let cell = rabi_cell(seed, 1.0);
        let result = fit(&cell);
        assert!(result.success, "{}", result.message);
        assert!(result.covariance_valid, "seed {seed}: {:?}", result.parameter_values);
        let tau = result.parameter("decay_time").unwrap();
        let (want, singular) = sandwich_errors(&cell, &result.parameter_values);
        raw_singular += singular as usize;
        for (i, name) in result.parameter_names().iter().enumerate() {
            let got = result.standard_errors[i];
            assert!(got.is_finite() && got > 0.0, "seed {seed} {name}: {got}");
            if *name != "decay_time" {
                assert!((got - want[i]).abs() <= 1e-6 * want[i], "seed {seed} {name}: {got} vs {}", want[i]);
            }
        }
        // tau, weakly determined: an error large against its value, finite.
        let tau_error = result.error("decay_time").unwrap();
        assert!(tau_error > 0.1 * tau, "seed {seed}: tau {tau} +- {tau_error}");
        assert!((result.error("baseband_frequency").unwrap() / FREQUENCY) < 0.05);
        assert!(result.parameter_error_validity().iter().all(|(_, valid)| *valid));
    }
    // The cases the raw-unit test lost every error on are among them.
    assert!(raw_singular > 0, "no seed is singular in raw units");
}

/// A series that never changed (a site that survives every shot): nothing
/// measures the shape's parameters, and no error may claim they are known.
#[test]
fn a_series_that_never_changed_claims_no_certainty() {
    let engine = FitEngine::default();
    let x: Vec<f64> = (0..50).map(|i| 0.1 + 3.06 * i as f64).collect();
    for (model, level, shape) in [("damped_sine", 1.0, vec!["baseband_frequency", "decay_time", "phase"]), ("damped_sine", 0.0, vec!["baseband_frequency", "decay_time", "phase"]), ("gaussian_offset", -5.0, vec!["center", "sigma"])] {
        let spec = engine.registry.get(model).unwrap();
        let cell = FitCell::points(vec![x.clone()], vec![level; x.len()]);
        let got = engine.fit(&spec, &cell, &FitRequest { initial: None, bounds: None, options: None, cancelled: None }).unwrap();
        eprintln!("{model} at {level}: {:?} +- {:?}", got.parameter_values, got.standard_errors);
        for name in shape {
            let index = spec.parameter_index(name).unwrap();
            let (value, error) = (got.parameter_values[index], got.standard_errors[index]);
            assert!(error.is_nan() || error > 0.1 * value.abs(), "{model} at {level}: {name} {value} +- {error}");
        }
    }
}

#[test]
fn the_x_unit_changes_no_validity() {
    for seed in 1..=6 {
        let micro = fit(&rabi_cell(seed, 1.0));
        // The same scan in seconds: f in 1/s, tau in s.
        let second = fit(&rabi_cell(seed, 1.0e6));
        assert_eq!(micro.covariance_valid, second.covariance_valid, "seed {seed}");
        for name in ["amplitude", "offset", "baseband_frequency", "decay_time", "phase"] {
            let relative = |r: &FitResult| r.error(name).unwrap() / r.parameter(name).unwrap().abs();
            let (a, b) = (relative(&micro), relative(&second));
            assert_eq!(a.is_finite(), b.is_finite(), "seed {seed} {name}");
            if name != "decay_time" {
                assert!((a - b).abs() <= 1e-4 * a, "seed {seed} {name}: {a} vs {b}");
            }
        }
    }
}

/// `J^T J` of a Jacobian whose columns are `columns`, row-major.
fn information(columns: &[Vec<f64>]) -> Vec<f64> {
    let k = columns.len();
    (0..k * k).map(|i| columns[i / k].iter().zip(&columns[i % k]).map(|(a, b)| a * b).sum()).collect()
}

fn columns(rows: usize) -> Vec<Vec<f64>> {
    let x: Vec<f64> = (0..rows).map(|i| i as f64 / rows as f64).collect();
    vec![x.iter().map(|_| 1.0).collect(), x.clone(), x.iter().map(|t| (3.0 * t).sin()).collect(), x.iter().map(|t| t * t).collect()]
}

#[test]
fn an_unidentifiable_parameter_loses_its_own_error_only() {
    let rows = 40;
    // A parameter with no influence at all: a zero column.
    let mut with_zero = columns(rows);
    with_zero[2] = vec![0.0; rows];
    // Two parameters entering only as their sum: equal columns.
    let mut with_pair = columns(rows);
    with_pair[3] = with_pair[1].clone();
    for (matrix, undefined) in [(information(&with_zero), vec![2]), (information(&with_pair), vec![1, 3])] {
        let identified = identify::from_information(4, &matrix, rows).unwrap().unwrap();
        let covariance = identified.covariance(1.0);
        for i in 0..4 {
            assert_eq!(identified.defined[i], !undefined.contains(&i), "{i}");
            assert_eq!(covariance[i * 4 + i].is_finite(), identified.defined[i]);
        }
        // The determined ones: the inverse of the information with the
        // undetermined directions taken out (here: the zero column dropped,
        // or the pair merged into one column).
        let kept: Vec<Vec<f64>> = if undefined.len() == 1 { [0, 1, 3].iter().map(|&c| with_zero[c].clone()).collect() } else { vec![with_pair[0].clone(), with_pair[1].clone(), with_pair[2].clone()] };
        let reduced = identify::from_information(3, &information(&kept), rows).unwrap().unwrap();
        assert!(reduced.defined.iter().all(|&d| d));
        let small = reduced.covariance(1.0);
        let map: Vec<usize> = if undefined.len() == 1 { vec![0, 1, 3] } else { vec![0, usize::MAX, 2] };
        for (a, &i) in map.iter().enumerate().filter(|&(_, &i)| i != usize::MAX) {
            for (b, &j) in map.iter().enumerate().filter(|&(_, &j)| j != usize::MAX) {
                let (got, want) = (covariance[i * 4 + j], small[a * 3 + b]);
                assert!((got - want).abs() <= 1e-8 * (small[a * 3 + a] * small[b * 3 + b]).sqrt(), "[{i},{j}] {got} vs {want}");
            }
        }
    }
    // Nothing determined at all.
    assert!(identify::from_information(2, &[0.0; 4], rows).unwrap().is_none());
}

#[test]
fn rescaling_parameters_changes_no_validity() {
    let rows = 40;
    let mut cases = vec![columns(rows)];
    let mut pair = columns(rows);
    pair[3] = pair[1].clone();
    cases.push(pair);
    for base in cases {
        let reference = identify::from_information(4, &information(&base), rows).unwrap().unwrap();
        let reference_covariance = reference.covariance(1.0);
        for scales in [[1.0e-9, 1.0, 1.0e6, 1.0e-3], [1.0e12, 1.0e-12, 1.0, 1.0e8]] {
            let scaled: Vec<Vec<f64>> = base.iter().zip(scales).map(|(c, s)| c.iter().map(|v| v * s).collect()).collect();
            let identified = identify::from_information(4, &information(&scaled), rows).unwrap().unwrap();
            assert_eq!(identified.defined, reference.defined, "{scales:?}");
            let covariance = identified.covariance(1.0);
            for i in (0..4).filter(|&i| reference.defined[i]) {
                for j in (0..4).filter(|&j| reference.defined[j]) {
                    // A column scaled by s scales its parameter by 1/s.
                    let want = reference_covariance[i * 4 + j] / (scales[i] * scales[j]);
                    let size = (covariance[i * 4 + i] * covariance[j * 4 + j]).sqrt();
                    assert!((covariance[i * 4 + j] - want).abs() <= 1e-9 * size, "{scales:?} [{i},{j}]");
                }
            }
            // The factor route (a QR triangle's columns) reads the same.
            let k = 4;
            let flat: Vec<f64> = (0..rows * k).map(|i| scaled[i % k][i / k]).collect();
            let triangle = zlc_plot::fit::linalg::qr_r(rows, k, &flat).unwrap();
            let factor = identify::from_factor(k, &triangle, rows).unwrap().unwrap();
            assert_eq!(factor.defined, reference.defined, "factor {scales:?}");
        }
    }
}
