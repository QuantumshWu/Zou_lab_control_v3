//! Proportions (the MEAN of booleans, observations with Bernoulli trials)
//! are fitted by the binomial likelihood, histograms by the Poisson one;
//! the model fixes the noise, the deviance is not a scale, and the errors
//! are the profile-likelihood intervals' (docs/UNCERTAINTY.md).
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

/// The inverse Fisher information at `result`'s parameters, computed here
/// independently: central-difference Jacobian of the model's values,
/// `J^T W J` over the free parameters, Gauss-Jordan on its correlation
/// form.  Row-major over all parameters, 0 at fixed ones.
fn fisher_covariance(result: &FitResult, x: &[f64], trials: Option<&[f64]>) -> Vec<f64> {
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
    let n = p.len();
    let mut covariance = vec![0.0; n * n];
    for (a, &i) in free.iter().enumerate() {
        for (b, &j) in free.iter().enumerate() {
            covariance[i * n + j] = inverse[a * k + b];
        }
    }
    covariance
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
    (0..y.len()).map(|i| 2.0 * n[i] * (term(y[i], f[i]) + term(1.0 - y[i], 1.0 - f[i]))).sum()
}

/// The Poisson deviance, summed, of counts `y` at `f`.
fn poisson_deviance(y: &[f64], f: &[f64]) -> f64 {
    (0..y.len()).map(|i| {
        let f = f[i].max(FLOOR);
        2.0 * (f - y[i] + if y[i] > 0.0 { y[i] * (y[i] / f).ln() } else { 0.0 }).max(0.0)
    }).sum()
}

fn relative(got: f64, want: f64) -> f64 { (got - want).abs() / want.abs() }

/// Every parameter's error is its interval's half-width and the covariance
/// is `D R D`: half-widths on the Fisher correlation `R`.
fn assert_interval_errors(label: &str, result: &FitResult, fisher: &[f64]) {
    let intervals = result.intervals.as_ref().unwrap_or_else(|| panic!("{label}: no intervals"));
    let n = result.parameter_values.len();
    for i in 0..n {
        let [low, high] = intervals[i];
        assert!(low <= result.parameter_values[i] && result.parameter_values[i] <= high, "{label} [{i}]: {low} .. {high} misses {}", result.parameter_values[i]);
        assert!((result.standard_errors[i] - 0.5 * (high - low)).abs() <= 1e-12 * result.standard_errors[i], "{label} [{i}]");
        for j in 0..n {
            let scale = (fisher[i * n + i] * fisher[j * n + j]).sqrt();
            if scale > 0.0 {
                let correlation = result.covariance[i * n + j] / (result.standard_errors[i] * result.standard_errors[j]);
                assert!((correlation - fisher[i * n + j] / scale).abs() <= 1e-6, "{label} [{i},{j}]: correlation {correlation} vs Fisher {}", fisher[i * n + j] / scale);
            }
        }
    }
}

/// An interval's ends against a reference profile: each within 2e-3 of its side's length.
fn assert_ends(label: &str, result: &FitResult, name: &str, want: [f64; 2]) {
    let index = result.model.parameter_index(name).unwrap();
    let (value, [low, high]) = (result.parameter_values[index], result.intervals.as_ref().unwrap()[index]);
    eprintln!("{label} {name}: {value} in [{low}, {high}] (reference [{}, {}])", want[0], want[1]);
    for (got, want) in [(low, want[0]), (high, want[1])] {
        let side = (want - value).abs();
        assert!((got - want).abs() <= 2e-3 * side + 1e-8 * value.abs(), "{label} {name}: end {got} vs {want} (side {side})");
    }
}

const REFERENCE_X: [f64; 8] = [0.0, 25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0];
const REFERENCE_K: [f64; 8] = [18.0, 13.0, 11.0, 8.0, 4.0, 4.0, 2.0, 0.0];

/// A survival decay, 8 points of 20 trials, `A exp(-x/tau)`: the binomial
/// maximum-likelihood answer of a checked prototype (SciPy), and its
/// profile-likelihood intervals (SciPy: Nelder-Mead profile, brentq on a
/// rise of 1, `prof.py` in the work notes), on both lanes.
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
        assert!(relative(amplitude, 0.90218585) < 1e-5, "{lane}: A {amplitude}");
        assert!(relative(tau, 84.522032) < 1e-5, "{lane}: tau {tau}");
        assert_ends(lane, &result, "amplitude", [0.8332957110133942, 0.9526838224621427]);
        assert_ends(lane, &result, "decay_time", [73.49190165036055, 98.47508008892879]);
        // The Fisher covariance of the prototype: its correlation stays.
        let mut fisher = vec![0.0; 9];
        for (index, value) in [(0, 0.0037442270), (2, -0.33345775), (6, -0.33345775), (8, 150.80504)] {
            fisher[index] = value;
        }
        assert_interval_errors(lane, &result, &fisher);
        assert_eq!(result.error("offset").unwrap(), 0.0);
        assert_eq!(result.intervals.as_ref().unwrap()[1], [0.0, 0.0]);
        assert_eq!(result.error_basis, ErrorBasis::Likelihood(Likelihood::Binomial));
        // The reduced chi-square is the deviance per degree of freedom.
        let y: Vec<f64> = REFERENCE_K.iter().map(|k| k / 20.0).collect();
        let deviance = binomial_deviance(&y, &trials, &result.fitted_values());
        assert!(relative(result.reduced_chi_square, deviance / 6.0) < 1e-9, "{lane}: {} vs {}", result.reduced_chi_square, deviance / 6.0);
        assert_eq!(result.error_basis_text(), "68 % likelihood interval (binomial)");
    }
}

/// An all-successes point (`y = 1`): the fit puts f(0) = A on the edge of
/// the probabilities, where the log-likelihood is linear in A and no
/// curvature describes it.  The profile does: A's interval is one-sided,
/// its upper side 0 (A cannot pass 1), its lower side the likelihood-ratio
/// distance; tau's is two-sided.  SciPy profile on the support (`prof.py`):
/// A in [0.9712432308837338, 1], tau in [67.9301940955145, 86.39872214385115],
/// MLE A = 1, tau = 76.52498052650844.
#[test]
fn an_all_successes_point_gets_a_one_sided_interval() {
    let engine = FitEngine::default();
    let trials = [20.0; 8];
    let mut k = REFERENCE_K;
    k[0] = 20.0;
    let cell = proportion_cell(&REFERENCE_X, &k, &trials);
    for (lane, model) in lanes(&engine, "exponential_decay") {
        let result = engine.fit(&model, &cell, &request(offset_fixed_at_zero())).unwrap();
        assert!(result.success && result.covariance_valid, "{lane}: {}", result.message);
        assert!(result.parameter_values.iter().chain(&result.standard_errors).all(|v| v.is_finite()));
        let (amplitude, tau) = (result.parameter("amplitude").unwrap(), result.parameter("decay_time").unwrap());
        assert!(relative(tau, 76.52498052650844) < 1e-4, "{lane}: tau {tau}");
        // f(0) at the clamp: the amplitude on the boundary f(0) = 1.
        assert!((amplitude - 1.0).abs() < 1e-6, "{lane}: A {amplitude}");
        let [low, high] = result.intervals.as_ref().unwrap()[0];
        eprintln!("{lane}: A = {amplitude} in [{low}, {high}], error {}", result.error("amplitude").unwrap());
        // The upper side is 0: it ends where f(0) leaves [0, 1] (to the solve's resolution).
        assert!(high >= amplitude && (high - 1.0).abs() <= 1e-7, "{lane}: A's upper end {high}");
        assert!((low - 0.9712432308837338).abs() <= 2e-3 * (1.0 - 0.9712432308837338), "{lane}: A's lower end {low}");
        assert_ends(lane, &result, "decay_time", [67.9301940955145, 86.39872214385115]);
        assert!(result.error("amplitude").unwrap() > 0.01, "{lane}: A's error {}", result.error("amplitude").unwrap());
        let fisher = fisher_covariance(&result, &REFERENCE_X, Some(&trials));
        assert_interval_errors(lane, &result, &fisher);
    }
}

/// A Poisson histogram (71 counts in 21 bins of 0.5), `histogram_gaussian`:
/// the MLE and its profile intervals against SciPy (`profp.py`, Nelder-Mead
/// profile, brentq on a rise of 1), on both lanes.
#[test]
fn a_histogram_gets_its_profile_intervals() {
    let engine = FitEngine::default();
    let x: Vec<f64> = (0..21).map(|i| -5.0 + 0.5 * i as f64).collect();
    let counts = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 7.0, 8.0, 12.0, 10.0, 11.0, 10.0, 6.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    let cell = FitCell::points(vec![x.clone()], counts.to_vec());
    for (lane, model) in lanes(&engine, "histogram_gaussian") {
        let result = engine.fit(&model, &cell, &request(None)).unwrap();
        assert!(result.success && result.covariance_valid, "{lane}: {}", result.message);
        for (name, want) in [("amplitude", 35.50009656), ("center", 0.16902326), ("sigma", 1.1163152)] {
            assert!(relative(result.parameter(name).unwrap(), want) < 1e-5, "{lane} {name}: {}", result.parameter(name).unwrap());
        }
        assert_ends(lane, &result, "amplitude", [31.452002789028146, 39.88147251183366]);
        assert_ends(lane, &result, "center", [0.036069889315810066, 0.30198057361140485]);
        assert_ends(lane, &result, "sigma", [1.0287871074213146, 1.2170422597989148]);
        assert_interval_errors(lane, &result, &fisher_covariance(&result, &x, None));
        assert_eq!(result.error_basis_text(), "68 % likelihood interval (Poisson)");
    }
}

/// The class: for proportions and counts, on both lanes, over data sets
/// whose deviance per degree of freedom ranges widely, every interval holds
/// its MLE, the profile deviance -- refitted here by the engine with the
/// parameter fixed, the deviance computed here -- rises by 1 at each end
/// (or less, at the edge of the likelihood's support), the errors are the
/// half-widths on the Fisher correlation, and nothing scales by the deviance.
#[test]
fn likelihood_intervals_are_the_profile() {
    let engine = FitEngine::default();
    let mut reduced = Vec::new();
    let (mut ends, mut saturated) = (0, 0);
    let mut check = |label: &str, model: &Arc<FitModelSpec>, cell: &FitCell, base: Vec<(String, (Option<f64>, Option<f64>))>, deviance: &dyn Fn(&[f64]) -> f64, x: &[f64], trials: Option<&[f64]>| {
        let result = engine.fit(model, cell, &request((!base.is_empty()).then(|| Bounds(base.clone())))).unwrap();
        assert!(result.success && result.covariance_valid, "{label}: {}", result.message);
        assert_interval_errors(label, &result, &fisher_covariance(&result, x, trials));
        let minimum = deviance(&result.fitted_values());
        for (index, parameter) in result.model.parameters.iter().enumerate() {
            if result.fixed_parameter_names.contains(&parameter.name) {
                continue;
            }
            let value = result.parameter_values[index];
            for end in result.intervals.as_ref().unwrap()[index] {
                if end == value {
                    saturated += 1;
                    continue;
                }
                let mut bounds = base.clone();
                bounds.push((parameter.name.clone(), (Some(end), Some(end))));
                let held = engine.fit(model, cell, &request(Some(Bounds(bounds)))).unwrap();
                let fitted = held.fitted_values();
                let rise = deviance(&fitted) - minimum;
                let at_edge = fitted.iter().any(|&f| trials.is_some() && f >= 1.0 - 1e-6);
                ends += 1;
                if at_edge {
                    saturated += 1;
                    assert!(rise <= 1.01, "{label} {}: rise {rise} at the support's edge {end}", parameter.name);
                } else {
                    assert!((rise - 1.0).abs() <= 0.01, "{label} {}: rise {rise} at {end} (value {value})", parameter.name);
                }
            }
        }
        result.reduced_chi_square
    };
    // Proportions: a survival decay, 8 to 15 points of 10, 20 or 50 trials.
    for seed in 1..=8u64 {
        let mut rng = Noise::new(seed);
        let points = 8 + (seed as usize % 8);
        let x: Vec<f64> = (0..points).map(|i| 20.0 * i as f64).collect();
        let n: Vec<f64> = (0..points).map(|_| [10.0, 20.0, 50.0][(rng.uniform() * 3.0) as usize]).collect();
        let k: Vec<f64> = x.iter().zip(&n).map(|(&x, &n)| rng.binomial(n, 0.8 * (-x / 90.0).exp())).collect();
        let y: Vec<f64> = k.iter().zip(&n).map(|(k, n)| k / n).collect();
        let cell = proportion_cell(&x, &k, &n);
        let deviance = |f: &[f64]| binomial_deviance(&y, &n, f);
        for (lane, model) in lanes(&engine, "exponential_decay") {
            let base = vec![("offset".to_string(), (Some(0.0), Some(0.0)))];
            reduced.push(check(&format!("binomial seed {seed} {lane}"), &model, &cell, base, &deviance, &x, Some(&n)));
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
        let cell = FitCell::points(vec![x.clone()], counts.clone());
        let deviance = |f: &[f64]| poisson_deviance(&counts, f);
        for (lane, model) in lanes(&engine, "histogram_gaussian") {
            reduced.push(check(&format!("Poisson seed {seed} {lane}"), &model, &cell, Vec::new(), &deviance, &x, None));
        }
    }
    let (low, high) = reduced.iter().fold((f64::INFINITY, 0.0_f64), |(l, h), &r| (l.min(r), h.max(r)));
    eprintln!("deviance per degree of freedom over the sets: {low:.3} .. {high:.3}; {ends} ends refitted, {saturated} at the support's edge");
    // Data sets whose deviance/dof is far from 1: a scaled covariance would
    // be off by sqrt of that.
    assert!(low < 0.7 && high > 1.3, "{low} .. {high}");
}

/// Survival decays of 8 points x 20 trials, 400 draws each: one well inside
/// (A = 0.9, the audit's) and one boundary-prone (A = 0.97: f(0) comes out
/// at 1 in about half the draws).  The likelihood intervals cover the truth
/// about 68 % of the time for tau in both and for A inside; on the
/// boundary A's interval covers about half the time, as the profile
/// likelihood itself does there (SciPy, the same profile on 200 draws of
/// A = 0.97: A 49.5 %, tau 67 %; `cov.py` in the work notes) -- a coverage
/// property of likelihood intervals on a boundary, not of this solver.
/// Printed beside them, the Fisher (curvature) ±1σ and the Gaussian fit
/// with the proportions' own standard errors `sqrt(p (1 - p) / (n - 1))`.
#[test]
fn likelihood_intervals_cover_their_truth() {
    let engine = FitEngine::default();
    let model = engine.registry.get("exponential_decay").unwrap();
    let (tau, n, draws) = (80.0, 20.0, 400);
    let trials = [n; 8];
    // (A, seed, A's coverage band): a binomial share of 400 draws within
    // three of its standard deviations (2.3 %) of 68 %, or of SciPy's 49.5 %.
    for (amplitude, seed, band) in [(0.9, 7, 0.61..=0.75), (0.97, 8, 0.42..=0.57)] {
        let mut rng = Noise::new(seed);
        let (mut count, mut on_edge, mut sum) = (0, 0, 0.0);
        let (mut covered, mut fisher_covered) = ([0; 2], [0; 2]);
        let (mut gaussian_covered, mut gaussian_count) = (0, 0);
        let truth = [amplitude, tau];
        for _ in 0..draws {
            let k: Vec<f64> = REFERENCE_X.iter().map(|&x| rng.binomial(n, amplitude * (-x / tau).exp())).collect();
            let Ok(result) = engine.fit(&model, &proportion_cell(&REFERENCE_X, &k, &trials), &request(offset_fixed_at_zero())) else { continue };
            if !(result.success && result.covariance_valid) {
                continue;
            }
            count += 1;
            on_edge += (k[0] == n) as usize;
            sum += result.parameter_values[2];
            let fisher = fisher_covariance(&result, &REFERENCE_X, Some(&trials));
            for (slot, index) in [(0, 0), (1, 2)] {
                let [low, high] = result.intervals.as_ref().unwrap()[index];
                covered[slot] += (low <= truth[slot] && truth[slot] <= high) as usize;
                fisher_covered[slot] += ((result.parameter_values[index] - truth[slot]).abs() <= fisher[index * 3 + index].sqrt()) as usize;
            }
            let y: Vec<f64> = k.iter().map(|k| k / n).collect();
            let mut cell = FitCell::points(vec![REFERENCE_X.to_vec()], y.clone());
            cell.sigma = Some(y.iter().map(|p| (p * (1.0 - p) / (n - 1.0)).sqrt()).collect::<Vec<f64>>().into());
            if let Ok(gaussian) = engine.fit(&model, &cell, &request(offset_fixed_at_zero())) {
                if gaussian.success && gaussian.covariance_valid {
                    gaussian_covered += ((gaussian.parameter_values[2] - tau).abs() <= gaussian.standard_errors[2]) as usize;
                    gaussian_count += 1;
                }
            }
        }
        let share = |c: usize| c as f64 / count as f64;
        eprintln!(
            "A = {amplitude}: {count} fits ({on_edge} with k(0) = n), tau bias {:+.2} %; likelihood interval coverage A {:.1} %, tau {:.1} %; Fisher ±1σ A {:.1} %, tau {:.1} %; Gaussian with the proportions' sem: tau {:.1} %",
            100.0 * (sum / count as f64 / tau - 1.0),
            100.0 * share(covered[0]),
            100.0 * share(covered[1]),
            100.0 * share(fisher_covered[0]),
            100.0 * share(fisher_covered[1]),
            100.0 * gaussian_covered as f64 / gaussian_count as f64
        );
        assert!(count >= draws - 5, "{count} of {draws} fitted");
        assert!(band.contains(&share(covered[0])), "A = {amplitude}: A's coverage {}", share(covered[0]));
        assert!((0.61..=0.75).contains(&share(covered[1])), "A = {amplitude}: tau's coverage {}", share(covered[1]));
    }
}

/// Sparse photon-count histograms (`histogram_poisson_gaussian`, 120 shots
/// of rate 3 read with sigma 0.35, 0.25-wide bins): the rate's likelihood
/// interval covers the truth about 68 % of the time.  Printed beside it:
/// the Fisher ±1σ and the error the deviance/dof scale gave
/// (`Fisher x sqrt(reduced chi-square)`).
#[test]
fn sparse_histograms_cover_their_rate() {
    let engine = FitEngine::default();
    let model = engine.registry.get("histogram_poisson_gaussian").unwrap();
    let (shots, rate, sigma, step) = (120, 3.0, 0.35, 0.25);
    let x: Vec<f64> = (0..61).map(|i| -1.0 + step * i as f64).collect();
    let mut rng = Noise::new(11);
    let (mut covered, mut fisher_covered, mut scaled_covered, mut count) = (0, 0, 0, 0);
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
        let index = model.parameter_index("rate").unwrap();
        let value = result.parameter_values[index];
        let [low, high] = result.intervals.as_ref().unwrap()[index];
        let fisher = fisher_covariance(&result, &x, None)[index * 3 + index].sqrt();
        covered += (low <= rate && rate <= high) as usize;
        fisher_covered += ((value - rate).abs() <= fisher) as usize;
        scaled_covered += ((value - rate).abs() <= fisher * result.reduced_chi_square.sqrt()) as usize;
        count += 1;
    }
    let share = |c: usize| c as f64 / count as f64;
    eprintln!(
        "rate: {count} fits, likelihood interval coverage {:.1} %, Fisher ±1σ {:.1} %, deviance-scaled ±1σ {:.1} %",
        100.0 * share(covered),
        100.0 * share(fisher_covered),
        100.0 * share(scaled_covered)
    );
    assert!(count >= 290, "{count} of 300 fitted");
    assert!((0.62..=0.74).contains(&share(covered)), "coverage {}", share(covered));
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

/// Wall time of the reference fit and of a 35-cell batch of survival decays
/// (8 points x 20 trials): `cargo test --test fit_binomial binomial_fit_cost -- --ignored --nocapture`.
#[test]
#[ignore = "a timing, read by hand"]
fn binomial_fit_cost() {
    let engine = FitEngine::default();
    let model = engine.registry.get("exponential_decay").unwrap();
    let trials = [20.0; 8];
    let reference = proportion_cell(&REFERENCE_X, &REFERENCE_K, &trials);
    let mut rng = Noise::new(5);
    let batch: Vec<FitCell> = (0..35)
        .map(|_| {
            let k: Vec<f64> = REFERENCE_X.iter().map(|&x| rng.binomial(20.0, 0.9 * (-x / 80.0).exp())).collect();
            proportion_cell(&REFERENCE_X, &k, &trials)
        })
        .collect();
    let rounds = 200;
    let started = std::time::Instant::now();
    for _ in 0..rounds {
        engine.fit(&model, &reference, &request(offset_fixed_at_zero())).unwrap();
    }
    let single = started.elapsed().as_secs_f64() / rounds as f64;
    let started = std::time::Instant::now();
    for _ in 0..rounds / 10 {
        engine.fit_batch(&model, &batch, &request(offset_fixed_at_zero())).unwrap();
    }
    let batched = started.elapsed().as_secs_f64() / (rounds / 10) as f64;
    eprintln!("reference fit {:.3} ms; 35-cell batch {:.3} ms", 1e3 * single, 1e3 * batched);
}
