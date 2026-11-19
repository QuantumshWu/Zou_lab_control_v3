//! Proportions (the MEAN of booleans, observations with Bernoulli trials)
//! are fitted by the binomial likelihood, histograms by the Poisson one,
//! and for both the standard errors are the inverse Fisher information:
//! the model fixes the noise, the deviance is not a scale (docs/UNCERTAINTY.md).
use std::sync::Arc;

use zlc_plot::fit::engine::evaluate_model;
use zlc_plot::fit::*;

/// A deterministic random sequence (xorshift + Box-Muller).
struct Noise(u64);

impl Noise {
    fn new(seed: u64) -> Self { Noise(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1) }
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 { (-2.0 * self.uniform().ln()).sqrt() * (2.0 * std::f64::consts::PI * self.uniform()).cos() }
    /// Successes in `n` Bernoulli trials of probability `p`.
    fn binomial(&mut self, n: f64, p: f64) -> f64 { (0..n as usize).filter(|_| self.uniform() < p).count() as f64 }
    /// A Poisson count of mean `rate` (Knuth, in pieces of mean at most 10).
    fn poisson(&mut self, rate: f64) -> f64 {
        let pieces = (rate / 10.0).ceil().max(1.0) as usize;
        let limit = (-rate / pieces as f64).exp();
        let mut count = 0.0;
        for _ in 0..pieces {
            let mut product = self.uniform();
            while product > limit {
                product *= self.uniform();
                count += 1.0;
            }
        }
        count
    }
}

const FLOOR: f64 = 1e-9;

/// The model on the registry's compiled lane, and the same model outside the
/// registry: the fallback lane.
fn lanes(engine: &FitEngine, model_id: &str) -> [(&'static str, Arc<FitModelSpec>); 2] {
    let compiled = engine.registry.get(model_id).unwrap();
    let fallback = Arc::new((*compiled).clone());
    [("compiled", compiled), ("fallback", fallback)]
}

fn request(bounds: Option<Bounds>) -> FitRequest<'static> { FitRequest { initial: None, bounds, options: None, cancelled: None } }

fn offset_fixed_at_zero() -> Option<Bounds> { Some(Bounds(vec![("offset".to_string(), (Some(0.0), Some(0.0)))])) }

fn proportion_cell(x: &[f64], k: &[f64], n: &[f64]) -> FitCell {
    let mut cell = FitCell::points(vec![x.to_vec()], k.iter().zip(n).map(|(k, n)| k / n).collect());
    cell.trials = Some(n.to_vec().into());
    cell
}

/// The binomial (with `trials`) or Poisson Fisher weight of a point at model value `f`.
fn fisher_weight(f: f64, trials: Option<f64>) -> f64 {
    match trials {
        Some(n) => {
            let f = f.clamp(FLOOR, 1.0 - FLOOR);
            n / (f * (1.0 - f))
        }
        None => 1.0 / f.max(FLOOR),
    }
}

/// `sqrt(diag)` of the inverse Fisher information at `result`'s parameters,
/// computed here independently: central-difference Jacobian of the model's
/// values, `J^T W J` over the free parameters, Gauss-Jordan on its
/// correlation form.  A fixed parameter's error is 0.
fn fisher_errors(result: &FitResult, x: &[f64], trials: Option<&[f64]>) -> Vec<f64> {
    let p = &result.parameter_values;
    let free: Vec<usize> = (0..p.len()).filter(|&i| !result.fixed_parameter_names.contains(&result.model.parameters[i].name)).collect();
    let coords = vec![x.to_vec()];
    let f = evaluate_model(&result.model, &coords, p).unwrap();
    let columns: Vec<Vec<f64>> = free
        .iter()
        .map(|&j| {
            let h = 1e-6 * p[j].abs().max(1e-3);
            let (mut up, mut down) = (p.clone(), p.clone());
            up[j] += h;
            down[j] -= h;
            let (a, b) = (evaluate_model(&result.model, &coords, &up).unwrap(), evaluate_model(&result.model, &coords, &down).unwrap());
            a.iter().zip(&b).map(|(a, b)| (a - b) / (2.0 * h)).collect()
        })
        .collect();
    let k = free.len();
    let information: Vec<f64> = (0..k * k)
        .map(|i| (0..x.len()).map(|point| columns[i / k][point] * columns[i % k][point] * fisher_weight(f[point], trials.map(|t| t[point]))).sum())
        .collect();
    let inverse = invert(k, &information);
    let mut errors = vec![0.0; p.len()];
    for (a, &j) in free.iter().enumerate() {
        errors[j] = inverse[a * k + a].sqrt();
    }
    errors
}

/// The inverse of a symmetric positive-definite matrix (Gauss-Jordan on its
/// correlation form).
fn invert(k: usize, matrix: &[f64]) -> Vec<f64> {
    let d: Vec<f64> = (0..k).map(|a| matrix[a * k + a].sqrt()).collect();
    let mut m: Vec<f64> = (0..k * k).map(|i| matrix[i] / (d[i / k] * d[i % k])).collect();
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
    (0..k * k).map(|i| inverse[i] / (d[i / k] * d[i % k])).collect()
}

/// The binomial deviance, summed, of proportions `y` of `n` trials at `f`.
fn binomial_deviance(y: &[f64], n: &[f64], f: &[f64]) -> f64 {
    let term = |a: f64, b: f64| if a > 0.0 { a * (a / b).ln() } else { 0.0 };
    (0..y.len()).map(|i| {
        let f = f[i].clamp(FLOOR, 1.0 - FLOOR);
        2.0 * n[i] * (term(y[i], f) + term(1.0 - y[i], 1.0 - f))
    }).sum()
}

fn relative(got: f64, want: f64) -> f64 { (got - want).abs() / want.abs() }

const REFERENCE_X: [f64; 8] = [0.0, 25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0];
const REFERENCE_K: [f64; 8] = [18.0, 13.0, 11.0, 8.0, 4.0, 4.0, 2.0, 0.0];

/// A survival decay, 8 points of 20 trials, `A exp(-x/tau)`: the binomial
/// maximum-likelihood answer and its Fisher covariance of a checked
/// prototype (SciPy), on both lanes.
#[test]
fn the_binomial_reference_is_reproduced_on_both_lanes() {
    let engine = FitEngine::default();
    let trials = [20.0; 8];
    let cell = proportion_cell(&REFERENCE_X, &REFERENCE_K, &trials);
    for (lane, model) in lanes(&engine, "exponential_decay") {
        let result = engine.fit(&model, &cell, &request(offset_fixed_at_zero())).unwrap();
        assert!(result.success && result.covariance_valid, "{lane}: {}", result.message);
        // The fallback lane solves seed by seed and keeps no solver row.
        assert_eq!(result.diagnostics.is_none(), lane == "fallback");
        let (amplitude, tau) = (result.parameter("amplitude").unwrap(), result.parameter("decay_time").unwrap());
        eprintln!("{lane}: A = {amplitude} ± {}, tau = {tau} ± {}", result.error("amplitude").unwrap(), result.error("decay_time").unwrap());
        assert!(relative(amplitude, 0.90218585) < 1e-5, "{lane}: A {amplitude}");
        assert!(relative(tau, 84.522032) < 1e-5, "{lane}: tau {tau}");
        let c = |a: usize, b: usize| result.covariance[a * 3 + b];
        for (got, want) in [(c(0, 0), 0.0037442270), (c(0, 2), -0.33345775), (c(2, 0), -0.33345775), (c(2, 2), 150.80504)] {
            assert!(relative(got, want) < 1e-4, "{lane}: covariance {got} vs {want}");
        }
        assert!(relative(result.error("amplitude").unwrap(), 0.0611901) < 1e-4);
        assert!(relative(result.error("decay_time").unwrap(), 12.28027) < 1e-4);
        assert_eq!(result.error("offset").unwrap(), 0.0);
        assert_eq!(result.error_basis, ErrorBasis::Likelihood(Likelihood::Binomial));
        // The reduced chi-square is the deviance per degree of freedom.
        let y: Vec<f64> = REFERENCE_K.iter().map(|k| k / 20.0).collect();
        let deviance = binomial_deviance(&y, &trials, &result.fitted_values());
        assert!(relative(result.reduced_chi_square, deviance / 6.0) < 1e-9, "{lane}: {} vs {}", result.reduced_chi_square, deviance / 6.0);
        assert_eq!(result.error_basis_text(), r"± 1$\sigma$: binomial likelihood");
    }
}

/// The class: for proportions and counts, on both lanes and over data sets
/// whose deviance per degree of freedom ranges widely, every reported error
/// is the inverse Fisher information at the fitted parameters -- whatever
/// the deviance.
#[test]
fn counted_and_proportion_errors_are_the_inverse_fisher_information() {
    let engine = FitEngine::default();
    let mut reduced = Vec::new();
    // Proportions: a survival decay, 8 to 15 points of 10, 20 or 50 trials.
    for seed in 1..=8u64 {
        let mut rng = Noise::new(seed);
        let points = 8 + (seed as usize % 8);
        let x: Vec<f64> = (0..points).map(|i| 20.0 * i as f64).collect();
        let n: Vec<f64> = (0..points).map(|_| [10.0, 20.0, 50.0][(rng.uniform() * 3.0) as usize]).collect();
        let k: Vec<f64> = x.iter().zip(&n).map(|(&x, &n)| rng.binomial(n, 0.8 * (-x / 90.0).exp())).collect();
        let cell = proportion_cell(&x, &k, &n);
        for (lane, model) in lanes(&engine, "exponential_decay") {
            let result = engine.fit(&model, &cell, &request(offset_fixed_at_zero())).unwrap();
            assert!(result.success && result.covariance_valid, "seed {seed} {lane}: {}", result.message);
            let want = fisher_errors(&result, &x, Some(&n));
            for (i, (&got, &want)) in result.standard_errors.iter().zip(&want).enumerate() {
                assert!((got - want).abs() <= 1e-6 * want, "binomial seed {seed} {lane} [{i}]: {got} vs Fisher {want} (deviance/dof {})", result.reduced_chi_square);
            }
            reduced.push(result.reduced_chi_square);
        }
    }
    // Counts: a Gaussian histogram of 40 to 400 shots in 21 bins, many sparse.
    for seed in 1..=8u64 {
        let mut rng = Noise::new(100 + seed);
        let shots = 40.0 * seed as f64 + 40.0 * rng.uniform();
        let step = 0.5;
        let x: Vec<f64> = (0..21).map(|i| -5.0 + step * i as f64).collect();
        let density = |v: f64| (-0.5 * (v / 1.3) * (v / 1.3)).exp() / (1.3 * (2.0 * std::f64::consts::PI).sqrt());
        let counts: Vec<f64> = x.iter().map(|&v| rng.poisson(shots * step * density(v))).collect();
        let cell = FitCell::points(vec![x.clone()], counts);
        for (lane, model) in lanes(&engine, "histogram_gaussian") {
            let result = engine.fit(&model, &cell, &request(None)).unwrap();
            assert!(result.success && result.covariance_valid, "seed {seed} {lane}: {}", result.message);
            assert_eq!(result.error_basis, ErrorBasis::Likelihood(Likelihood::Poisson));
            let want = fisher_errors(&result, &x, None);
            for (i, (&got, &want)) in result.standard_errors.iter().zip(&want).enumerate() {
                assert!((got - want).abs() <= 1e-6 * want, "Poisson seed {seed} {lane} [{i}]: {got} vs Fisher {want} (deviance/dof {})", result.reduced_chi_square);
            }
            reduced.push(result.reduced_chi_square);
        }
    }
    let (low, high) = reduced.iter().fold((f64::INFINITY, 0.0_f64), |(l, h), &r| (l.min(r), h.max(r)));
    eprintln!("deviance per degree of freedom over the sets: {low:.3} .. {high:.3}");
    // Data sets whose deviance/dof is far from 1: a scaled covariance would
    // be off by sqrt of that.
    assert!(low < 0.7 && high > 1.3, "{low} .. {high}");
}

/// An all-successes point (`y = 1`) is a proportion like any other: the fit
/// is finite, its errors the Fisher ones, and the point does not pull the
/// curve to itself (the binomial MLE of these data, SciPy: tau = 76.52498,
/// its error 9.23799; amplitude on the boundary f(0) = 1).
#[test]
fn an_all_successes_point_neither_vanishes_nor_dominates() {
    let engine = FitEngine::default();
    let trials = [20.0; 8];
    let mut k = REFERENCE_K;
    k[0] = 20.0;
    let cell = proportion_cell(&REFERENCE_X, &k, &trials);
    for (lane, model) in lanes(&engine, "exponential_decay") {
        let result = engine.fit(&model, &cell, &request(offset_fixed_at_zero())).unwrap();
        assert!(result.success && result.covariance_valid, "{lane}: {}", result.message);
        assert!(result.parameter_values.iter().chain(&result.standard_errors).all(|v| v.is_finite()));
        let (amplitude, tau, tau_error) = (result.parameter("amplitude").unwrap(), result.parameter("decay_time").unwrap(), result.error("decay_time").unwrap());
        eprintln!("{lane}: A = {amplitude} ± {:e}, tau = {tau} ± {tau_error}", result.error("amplitude").unwrap());
        assert!(relative(tau, 76.52498) < 1e-4, "{lane}: tau {tau}");
        assert!(relative(tau_error, 9.23799) < 1e-3, "{lane}: tau error {tau_error}");
        // f(0) at the clamp: the amplitude sits on the boundary f(0) = 1.
        assert!((amplitude - 1.0).abs() < 1e-6, "{lane}: A {amplitude}");
        let want = fisher_errors(&result, &REFERENCE_X, Some(&trials));
        for (i, (&got, &want)) in result.standard_errors.iter().zip(&want).enumerate() {
            assert!((got - want).abs() <= 1e-3 * want, "{lane} [{i}]: {got} vs Fisher {want}");
        }
    }
}

/// The audit's survival decay (A = 0.9, tau = 80, 8 points of 20 trials),
/// 400 times: the binomial fit's tau is unbiased and its ±1σ covers the
/// truth about 68 % of the time.  Printed beside it: the Gaussian fit with
/// the proportions' own standard errors `sqrt(p (1 - p) / (n - 1))`.
#[test]
fn binomial_errors_cover_their_truth() {
    let engine = FitEngine::default();
    let model = engine.registry.get("exponential_decay").unwrap();
    let (amplitude, tau, n) = (0.9, 80.0, 20.0);
    let trials = [n; 8];
    let mut rng = Noise::new(7);
    let (mut covered, mut sum, mut count) = (0, 0.0, 0);
    let (mut gaussian_covered, mut gaussian_sum, mut gaussian_count) = (0, 0.0, 0);
    for _ in 0..400 {
        let k: Vec<f64> = REFERENCE_X.iter().map(|&x| rng.binomial(n, amplitude * (-x / tau).exp())).collect();
        let binomial = engine.fit(&model, &proportion_cell(&REFERENCE_X, &k, &trials), &request(offset_fixed_at_zero()));
        if let Ok(result) = binomial.as_ref().map_err(|_| ()).and_then(|r| if r.success && r.covariance_valid { Ok(r) } else { Err(()) }) {
            let (value, error) = (result.parameter("decay_time").unwrap(), result.error("decay_time").unwrap());
            covered += ((value - tau).abs() <= error) as usize;
            sum += value;
            count += 1;
        }
        let y: Vec<f64> = k.iter().map(|k| k / n).collect();
        let mut cell = FitCell::points(vec![REFERENCE_X.to_vec()], y.clone());
        cell.sigma = Some(y.iter().map(|p| (p * (1.0 - p) / (n - 1.0)).sqrt()).collect::<Vec<f64>>().into());
        if let Ok(result) = engine.fit(&model, &cell, &request(offset_fixed_at_zero())) {
            if result.success && result.covariance_valid {
                let (value, error) = (result.parameter("decay_time").unwrap(), result.error("decay_time").unwrap());
                gaussian_covered += ((value - tau).abs() <= error) as usize;
                gaussian_sum += value;
                gaussian_count += 1;
            }
        }
    }
    let coverage = covered as f64 / count as f64;
    let bias = sum / count as f64 / tau - 1.0;
    eprintln!(
        "binomial: {count} fits, tau bias {:+.2} %, ±1σ coverage {:.1} %; Gaussian with the proportions' sem: {gaussian_count} fits, bias {:+.2} %, coverage {:.1} %",
        100.0 * bias,
        100.0 * coverage,
        100.0 * (gaussian_sum / gaussian_count as f64 / tau - 1.0),
        100.0 * gaussian_covered as f64 / gaussian_count as f64
    );
    assert!(count >= 395, "{count} of 400 fitted");
    assert!((0.63..=0.73).contains(&coverage), "coverage {coverage}");
    assert!(bias.abs() < 0.02, "bias {bias}");
}

/// Sparse photon-count histograms (`histogram_poisson_gaussian`, 120 shots
/// of rate 3 read with sigma 0.35, 0.25-wide bins): the rate's ±1σ covers
/// the truth about 68 % of the time.  The error the deviance/dof scale gave
/// (`error x sqrt(reduced chi-square)`) is printed beside it.
#[test]
fn sparse_histograms_cover_their_rate() {
    let engine = FitEngine::default();
    let model = engine.registry.get("histogram_poisson_gaussian").unwrap();
    let (shots, rate, sigma, step) = (120, 3.0, 0.35, 0.25);
    let x: Vec<f64> = (0..61).map(|i| -1.0 + step * i as f64).collect();
    let mut rng = Noise::new(11);
    let (mut covered, mut scaled_covered, mut count) = (0, 0, 0);
    let (mut smallest, mut largest, mut ratio_sum) = (f64::INFINITY, 0.0_f64, 0.0);
    for _ in 0..300 {
        let mut counts = vec![0.0; x.len()];
        for _ in 0..shots {
            let value = rng.poisson(rate) + sigma * rng.normal();
            let bin = ((value - x[0]) / step + 0.5).floor();
            if bin >= 0.0 && (bin as usize) < x.len() {
                counts[bin as usize] += 1.0;
            }
        }
        let Ok(result) = engine.fit(&model, &FitCell::points(vec![x.clone()], counts), &request(None)) else { continue };
        if !(result.success && result.covariance_valid) {
            continue;
        }
        let (value, error) = (result.parameter("rate").unwrap(), result.error("rate").unwrap());
        let scaled = error * result.reduced_chi_square.sqrt();
        covered += ((value - rate).abs() <= error) as usize;
        scaled_covered += ((value - rate).abs() <= scaled) as usize;
        count += 1;
        let ratio = scaled / error;
        smallest = smallest.min(ratio);
        largest = largest.max(ratio);
        ratio_sum += ratio;
    }
    let coverage = covered as f64 / count as f64;
    eprintln!(
        "rate: {count} fits, Fisher ±1σ coverage {:.1} %; deviance-scaled coverage {:.1} %, scaled/Fisher error {:.3} .. {:.3} (mean {:.3})",
        100.0 * coverage,
        100.0 * scaled_covered as f64 / count as f64,
        smallest,
        largest,
        ratio_sum / count as f64
    );
    assert!(count >= 290, "{count} of 300 fitted");
    assert!((0.62..=0.74).contains(&coverage), "coverage {coverage}");
}

/// A binomial series weighed against its nested model: the evidence is the
/// difference of the binomial deviances less `k ln N`, `N` the trials.
#[test]
fn binomial_evidence_is_the_deviance_difference() {
    let engine = FitEngine::default();
    let wide = engine.registry.get("lorentzian_doublet").unwrap();
    let narrow = engine.registry.get("lorentzian").unwrap();
    let x: Vec<f64> = (0..61).map(|i| -15.0 + 0.5 * i as f64).collect();
    let line = |x: f64, center: f64| 0.25 * 4.0 / ((x - center) * (x - center) + 0.25 * 4.0);
    let mut rng = Noise::new(3);
    let n = vec![50.0; x.len()];
    let k: Vec<f64> = x.iter().map(|&x| rng.binomial(50.0, 0.85 - 0.5 * line(x, -4.0) - 0.4 * line(x, 4.0))).collect();
    let cell = proportion_cell(&x, &k, &n);
    let two = engine.fit(&wide, &cell, &request(None)).unwrap();
    let one = engine.fit(&narrow, &cell, &request(None)).unwrap();
    assert!(two.success && one.success && !two.reduced, "{} / {}", two.message, one.message);
    assert_eq!(two.error_basis, ErrorBasis::Likelihood(Likelihood::Binomial));
    let y: Vec<f64> = k.iter().map(|k| k / 50.0).collect();
    let want = binomial_deviance(&y, &n, &one.fitted_values()) - binomial_deviance(&y, &n, &two.fitted_values()) - 3.0 * (50.0 * x.len() as f64).ln();
    eprintln!("evidence {} (want {want})", two.evidence);
    assert!((two.evidence - want).abs() <= 1e-6 * want.abs(), "{} vs {want}", two.evidence);
}

/// Trials are checked against their observations; a dropped point drops its trials.
#[test]
fn trials_are_checked_and_follow_their_points() {
    let engine = FitEngine::default();
    let n = [20.0; 8];
    for (lane, model) in lanes(&engine, "exponential_decay") {
        let refuse = |cell: &FitCell, message: &str| {
            let error = engine.fit(&model, cell, &request(offset_fixed_at_zero())).unwrap_err().to_string();
            assert!(error.contains(message), "{lane}: {error:?} lacks {message:?}");
        };
        let mut with_sigma = proportion_cell(&REFERENCE_X, &REFERENCE_K, &n);
        with_sigma.sigma = Some(vec![0.1; 8].into());
        refuse(&with_sigma, "an external sigma has no meaning there");
        let mut short = proportion_cell(&REFERENCE_X, &REFERENCE_K, &n);
        short.trials = Some(vec![20.0; 7].into());
        refuse(&short, "trials must match observations");
        let mut zero = proportion_cell(&REFERENCE_X, &REFERENCE_K, &n);
        zero.trials = Some([20.0, 20.0, 20.0, 0.5, 20.0, 20.0, 20.0, 20.0].to_vec().into());
        refuse(&zero, "trials must be finite and at least 1");
        let mut above = proportion_cell(&REFERENCE_X, &REFERENCE_K, &n);
        if let CellData::Points { observations, .. } = &mut above.data {
            let mut values = observations.as_ref().unwrap().to_vec();
            values[2] = 1.5;
            *observations = Some(values.into());
        }
        refuse(&above, "must lie in [0, 1]");
        // A NaN observation is dropped with its trials: the fit is that of the other points.
        let mut holed_k = REFERENCE_K;
        holed_k[4] = f64::NAN;
        let mut holed_n = n;
        holed_n[3] = 50.0;
        let holed = engine.fit(&model, &proportion_cell(&REFERENCE_X, &holed_k, &holed_n), &request(offset_fixed_at_zero())).unwrap();
        let keep: Vec<usize> = (0..8).filter(|&i| i != 4).collect();
        let pick = |values: &[f64]| keep.iter().map(|&i| values[i]).collect::<Vec<f64>>();
        let direct = engine.fit(&model, &proportion_cell(&pick(&REFERENCE_X), &pick(&holed_k), &pick(&holed_n)), &request(offset_fixed_at_zero())).unwrap();
        assert_eq!(holed.parameter_values, direct.parameter_values, "{lane}");
        assert_eq!(holed.standard_errors, direct.standard_errors, "{lane}");
    }
    // A histogram's counts are Poisson: trials are refused there.
    let model = engine.registry.get("histogram_gaussian").unwrap();
    let mut cell = FitCell::points(vec![REFERENCE_X.to_vec()], REFERENCE_K.to_vec());
    cell.trials = Some(n.to_vec().into());
    let error = engine.fit(&model, &cell, &request(None)).unwrap_err().to_string();
    assert!(error.contains("Bernoulli trials have no meaning there"), "{error}");
}
