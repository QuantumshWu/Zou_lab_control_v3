//! The Lorentzian doublet: two lines of independent center, width and
//! height on one offset, settled against the single Lorentzian by BIC.
use std::sync::Arc;

use zlc_plot::fit::classifier::evaluate_component;
use zlc_plot::fit::*;

type Line = (f64, f64, f64);

fn lorentzian(x: f64, (center, width, height): Line) -> f64 {
    let half_squared = 0.25 * width * width;
    height * half_squared / ((x - center) * (x - center) + half_squared)
}

/// A deterministic normal sequence (xorshift + Box-Muller).
struct Noise(u64);

impl Noise {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 { (-2.0 * self.uniform().ln()).sqrt() * (2.0 * std::f64::consts::PI * self.uniform()).cos() }
}

fn data(lines: &[Line], offset: f64, noise: f64, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = Noise(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let x: Vec<f64> = (0..201).map(|i| -20.0 + 0.2 * i as f64).collect();
    let y = x.iter().map(|&v| offset + lines.iter().map(|&l| lorentzian(v, l)).sum::<f64>() + noise * rng.normal()).collect();
    (x, y)
}

fn model() -> (FitEngine, Arc<FitModelSpec>) {
    let engine = FitEngine::default();
    let model = engine.registry.get("lorentzian_doublet").unwrap();
    (engine, model)
}

fn fit_with(lines: &[Line], offset: f64, noise: f64, seed: u64, options: FitOptions) -> FitResult {
    let (engine, model) = model();
    let (x, y) = data(lines, offset, noise, seed);
    let request = FitRequest { initial: None, bounds: None, options: Some(options), cancelled: None };
    engine.fit(&model, &FitCell::points(vec![x], y), &request).unwrap()
}

fn fit(lines: &[Line], offset: f64, noise: f64, seed: u64) -> FitResult { fit_with(lines, offset, noise, seed, FitOptions::default()) }

fn get(result: &FitResult, name: &str) -> f64 { result.parameter(name).unwrap() }

/// Both lines recovered within `tolerance` (absolute, in data units),
/// listed by center: line B at `x_0 + splitting`, the splitting >= 0.
fn assert_lines(result: &FitResult, want: [Line; 2], offset: f64, tolerance: f64) {
    assert!(result.success, "{}", result.message);
    assert!(!result.reduced, "settled on one line: {}", result.message);
    assert!(get(result, "splitting") >= 0.0);
    let center_b = get(result, "center") + get(result, "splitting");
    let got = [(get(result, "center"), get(result, "fwhm"), get(result, "amplitude")), (center_b, get(result, "fwhm_B"), get(result, "amplitude_B"))];
    for (g, w) in got.iter().zip(want) {
        assert!((g.0 - w.0).abs() < tolerance && (g.1 - w.1).abs() < tolerance && (g.2 - w.2).abs() < tolerance, "{got:?} vs {want:?}");
    }
    assert!((get(result, "offset") - offset).abs() < tolerance, "offset {}", get(result, "offset"));
}

#[test]
fn the_model_is_two_independent_lines_reducing_to_one() {
    let (_, model) = model();
    assert_eq!(model.parameter_names(), ["center", "fwhm", "amplitude", "offset", "splitting", "fwhm_B", "amplitude_B"]);
    assert_eq!(model.symbols(), ["x_0", "FWHM", "H", "B", "delta", "FWHM_B", "H_B"]);
    assert_eq!(model.headline, "splitting");
    assert_eq!(model.parameters[4].domain, ParameterDomain::Nonnegative);
    assert_eq!(model.parameters[4].unit_relation, UnitRelation::Axis0);
    assert_eq!(model.targets, [FitTarget::Series]);
    let reduction = model.reduction.as_ref().unwrap();
    assert_eq!(reduction.nested_model_id, "lorentzian");
    assert_eq!(model.presentation.components.iter().map(|c| c.component_id.as_str()).collect::<Vec<_>>(), ["A", "B"]);
    // The previous symmetric doublet is gone, not aliased: a saved
    // configuration naming it is refused by name.
    let engine = FitEngine::default();
    assert_eq!(engine.registry.get("symmetric_lorentzian_doublet").unwrap_err().to_string(), "unknown fit model: 'symmetric_lorentzian_doublet'");
    assert!(builtin_fit_models().iter().all(|m| m.model_id != "symmetric_lorentzian_doublet"));
}

#[test]
fn separated_lines_are_recovered_exactly_without_noise() {
    let want = [(-6.0, 2.0, 5.0), (4.0, 3.0, 2.0)];
    // Listed either way round, the result is ordered by center.
    for lines in [want, [want[1], want[0]]] {
        let result = fit(&lines, 1.0, 0.0, 1);
        assert_lines(&result, want, 1.0, 1e-6);
    }
}

#[test]
fn separated_lines_are_recovered_through_noise() {
    let want = [(-6.0, 2.0, 5.0), (4.0, 3.0, 2.0)];
    for seed in 1..=20 {
        let result = fit(&want, 1.0, 0.2, seed);
        assert_lines(&result, want, 1.0, 0.35);
        assert!(result.evidence > 10.0, "ΔBIC {}", result.evidence);
    }
}

#[test]
fn overlapping_lines_are_resolved() {
    // Separated by less than their widths: one bump with a shoulder.
    let want = [(-1.0, 3.0, 4.0), (1.2, 2.5, 3.0)];
    for seed in 1..=10 {
        let result = fit(&want, 0.5, 0.05, seed);
        assert_lines(&result, want, 0.5, 0.3);
    }
}

#[test]
fn unequal_lines_are_resolved() {
    // Twenty times weaker and three times wider, beside and under a tail.
    let strong = (-3.0, 1.5, 10.0);
    let weak = (5.0, 4.5, 0.5);
    for seed in 1..=10 {
        let result = fit(&[strong, weak], 0.0, 0.05, seed);
        assert_lines(&result, [strong, weak], 0.0, 0.6);
    }
    // A weak shoulder close to a strong line.
    let shoulder = (0.5, 2.0, 1.5);
    let main = (-1.5, 2.0, 8.0);
    for seed in 1..=10 {
        let result = fit(&[main, shoulder], 0.0, 0.05, seed);
        assert_lines(&result, [main, shoulder], 0.0, 0.3);
    }
}

#[test]
fn dips_and_a_peak_beside_a_dip_are_resolved() {
    let dips = [(-5.0, 2.0, -3.0), (5.0, 2.0, -2.0)];
    assert_lines(&fit(&dips, 10.0, 0.05, 3), dips, 10.0, 0.2);
    let mixed = [(-4.0, 2.5, 3.0), (3.0, 2.0, -2.0)];
    assert_lines(&fit(&mixed, 0.0, 0.05, 4), mixed, 0.0, 0.2);
}

#[test]
fn one_line_settles_on_the_single_lorentzian() {
    let line = (1.0, 3.0, 5.0);
    let (engine, _) = model();
    let single = engine.registry.get("lorentzian").unwrap();
    for seed in 1..=20 {
        let result = fit(&[line], 2.0, 0.2, seed);
        assert!(result.success, "{}", result.message);
        assert!(result.reduced, "seed {seed}: two lines, ΔBIC {}", result.evidence);
        assert!(result.evidence <= 10.0 && result.evidence.is_finite());
        assert!(result.message.starts_with("one population: BIC gain"), "{}", result.message);
        // The reduced answer is the single Lorentzian's, the second line
        // pinned on the first with no height, so nothing is unidentified.
        let (x, y) = data(&[line], 2.0, 0.2, seed);
        let request = FitRequest { initial: None, bounds: None, options: None, cancelled: None };
        let nested = engine.fit(&single, &FitCell::points(vec![x.clone()], y), &request).unwrap();
        for name in ["center", "fwhm", "amplitude", "offset"] {
            assert_eq!(get(&result, name), get(&nested, name), "{name}");
            assert_eq!(result.error(name), nested.error(name), "{name}");
        }
        assert_eq!(get(&result, "splitting"), 0.0);
        assert_eq!(get(&result, "fwhm_B"), get(&result, "fwhm"));
        assert_eq!(get(&result, "amplitude_B"), 0.0);
        for name in ["splitting", "fwhm_B", "amplitude_B"] {
            assert!(result.fixed_parameter_names.iter().any(|f| f == name));
            assert_eq!(result.error(name), Some(0.0));
        }
        assert!(result.standard_errors.iter().all(|e| e.is_finite()));
        let b = evaluate_component(&result.model, "B", &[x], &result.parameter_values).unwrap();
        assert!(b.iter().all(|&v| v == get(&result, "offset")));
    }
}

#[test]
fn the_bic_threshold_decides() {
    // A weak second line: the evidence for it is moderate.
    let lines = [(-3.0, 2.0, 3.0), (3.0, 2.0, 0.35)];
    let open = fit_with(&lines, 0.0, 0.2, 7, FitOptions { min_bic_gain: Some(f64::NEG_INFINITY), ..FitOptions::default() });
    assert!(!open.reduced && open.evidence.is_finite());
    let evidence = open.evidence;
    assert!(evidence > 0.0 && evidence < 200.0, "ΔBIC {evidence}");
    let below = fit_with(&lines, 0.0, 0.2, 7, FitOptions { min_bic_gain: Some(evidence - 0.5), ..FitOptions::default() });
    assert!(!below.reduced && below.evidence == evidence);
    let above = fit_with(&lines, 0.0, 0.2, 7, FitOptions { min_bic_gain: Some(evidence + 0.5), ..FitOptions::default() });
    assert!(above.reduced && above.evidence == evidence, "{}", above.message);
    // No threshold: no comparison, the doublet as solved.
    let none = fit_with(&lines, 0.0, 0.2, 7, FitOptions { min_bic_gain: None, ..FitOptions::default() });
    assert!(!none.reduced && none.evidence.is_nan());
    assert_eq!(none.parameter_values, open.parameter_values);
}

#[test]
fn the_evidence_is_the_gaussian_bic_difference() {
    let lines = [(-6.0, 2.0, 5.0), (4.0, 3.0, 2.0)];
    let (engine, model) = model();
    let (x, y) = data(&lines, 1.0, 0.2, 11);
    let request = FitRequest { initial: None, bounds: None, options: None, cancelled: None };
    let cell = FitCell::points(vec![x], y.clone());
    let wide = engine.fit(&model, &cell, &request).unwrap();
    let narrow = engine.fit(&engine.registry.get("lorentzian").unwrap(), &cell, &request).unwrap();
    let rss = |r: &FitResult| r.residuals().iter().map(|v| v * v).sum::<f64>();
    let n = y.len() as f64;
    let want = n * (rss(&narrow) / rss(&wide)).ln() - 3.0 * n.ln();
    assert!((wide.evidence - want).abs() < 1e-6 * want.abs(), "{} vs {want}", wide.evidence);
}

#[test]
fn components_draw_each_line_on_the_offset() {
    let lines = [(-6.0, 2.0, 5.0), (4.0, 3.0, 2.0)];
    let result = fit(&lines, 1.0, 0.0, 1);
    let x = vec![vec![-6.0, 4.0, 40.0]];
    let a = evaluate_component(&result.model, "A", &x, &result.parameter_values).unwrap();
    let b = evaluate_component(&result.model, "B", &x, &result.parameter_values).unwrap();
    assert!((a[0] - 1.0 - 5.0).abs() < 1e-6);
    assert!((b[1] - 1.0 - 2.0).abs() < 1e-6);
    assert!((a[2] - 1.0).abs() < 0.02 && (b[2] - 1.0).abs() < 0.02);
}

#[test]
fn the_analytic_jacobian_matches_finite_differences() {
    let (_, model) = model();
    let kernel = model.kernel().unwrap();
    let x: Vec<f64> = (0..41).map(|i| -10.0 + 0.5 * i as f64).collect();
    let p = [-1.5, 2.0, 3.0, 0.5, 2.5, 3.5, -1.2];
    let axes = models::Axes::one(&x);
    let (_, jacobian) = kernel.value_jacobian(&axes, &p, true).unwrap();
    for k in 0..7 {
        let step = 1e-6 * p[k].abs().max(1.0);
        let (mut up, mut down) = (p, p);
        up[k] += step;
        down[k] -= step;
        let (hi, _) = kernel.value_jacobian(&axes, &up, false).unwrap();
        let (lo, _) = kernel.value_jacobian(&axes, &down, false).unwrap();
        for i in 0..x.len() {
            let numeric = (hi[i] - lo[i]) / (2.0 * step);
            assert!((numeric - jacobian[i * 7 + k]).abs() < 1e-6, "parameter {k} point {i}: {numeric} vs {}", jacobian[i * 7 + k]);
        }
    }
}

// ---------------------------------------------------------------- Student-t

/// A curve as the panel builds it: `repeats` noisy repeats per point, the
/// mean, its standard error estimated from those repeats (`n - 1` degrees of
/// freedom).
fn repeated(lines: &[Line], offset: f64, noise: f64, repeats: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut rng = Noise(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let x: Vec<f64> = (0..121).map(|i| -15.0 + 0.25 * i as f64).collect();
    let (mut mean, mut sem) = (Vec::new(), Vec::new());
    for &v in &x {
        let truth = offset + lines.iter().map(|&l| lorentzian(v, l)).sum::<f64>();
        let samples: Vec<f64> = (0..repeats).map(|_| truth + noise * rng.normal()).collect();
        let m = samples.iter().sum::<f64>() / repeats as f64;
        let variance = samples.iter().map(|s| (s - m) * (s - m)).sum::<f64>() / (repeats - 1) as f64;
        mean.push(m);
        sem.push((variance / repeats as f64).sqrt());
    }
    (x, mean, sem)
}

fn repeat_cell(x: &[f64], mean: &[f64], sem: &[f64], dof: Option<f64>) -> FitCell {
    let mut cell = FitCell::points(vec![x.to_vec()], mean.to_vec());
    cell.sigma = Some(sem.to_vec().into());
    cell.sigma_dof = dof.map(|d| vec![d; x.len()].into());
    cell
}

fn solve(model_id: &str, cell: &FitCell) -> FitResult {
    let engine = FitEngine::default();
    let model = engine.registry.get(model_id).unwrap();
    engine.fit(&model, cell, &FitRequest { initial: None, bounds: None, options: None, cancelled: None }).unwrap()
}

#[test]
fn the_student_likelihood_is_the_gaussian_one_for_many_repeats() {
    // Infinite degrees of freedom is exactly the Gaussian weighting's
    // solution; many repeats come within the estimate's own noise of it.
    let line = (1.0, 3.0, 5.0);
    let (x, mean, sem) = repeated(&[line], 2.0, 0.25, 400, 5);
    let gaussian = solve("lorentzian", &repeat_cell(&x, &mean, &sem, None));
    let infinite = solve("lorentzian", &repeat_cell(&x, &mean, &sem, Some(f64::INFINITY)));
    assert_eq!(infinite.parameter_values, gaussian.parameter_values);
    let student = solve("lorentzian", &repeat_cell(&x, &mean, &sem, Some(399.0)));
    for name in ["center", "fwhm", "amplitude", "offset"] {
        let (g, t, e) = (get(&gaussian, name), get(&student, name), gaussian.error(name).unwrap());
        assert!((g - t).abs() < 0.1 * e, "{name}: {g} vs {t} (error {e})");
        let (ge, te) = (e, student.error(name).unwrap());
        assert!((te / ge - 1.0).abs() < 0.1, "{name} error: {ge} vs {te}");
    }
}

#[test]
fn few_repeats_no_longer_make_one_line_two() {
    // Four repeats: estimated standard errors scatter by a factor of a few,
    // and the Gaussian weighting believed the smallest of them. One line,
    // over many noise draws, is settled as one line by the Student-t BIC.
    let line = (1.0, 3.0, 5.0);
    let mut gaussian_two = 0;
    let mut student_two = 0;
    for seed in 1..=30 {
        let (x, mean, sem) = repeated(&[line], 2.0, 0.25, 4, seed);
        gaussian_two += usize::from(!solve("lorentzian_doublet", &repeat_cell(&x, &mean, &sem, None)).reduced);
        let student = solve("lorentzian_doublet", &repeat_cell(&x, &mean, &sem, Some(3.0)));
        assert!(student.success);
        student_two += usize::from(!student.reduced);
    }
    assert!(student_two <= 1, "Student-t: two lines in {student_two} of 30 (Gaussian weighting: {gaussian_two})");
    assert!(gaussian_two > student_two, "Gaussian weighting: {gaussian_two}, Student-t: {student_two}");
}

#[test]
fn few_repeats_keep_two_lines_and_sensible_errors() {
    // A real doublet with four repeats is still two lines; the errors the
    // likelihood reports cover the truth about as often as they should.
    let want = [(-4.0, 2.0, 6.0), (3.5, 3.0, 3.0)];
    let (mut inside, mut total) = (0, 0);
    for seed in 1..=20 {
        let (x, mean, sem) = repeated(&want, 2.0, 0.25, 4, seed);
        let result = solve("lorentzian_doublet", &repeat_cell(&x, &mean, &sem, Some(3.0)));
        assert_lines(&result, want, 2.0, 0.3);
        assert!(result.covariance_valid);
        let truth = [("center", -4.0), ("fwhm", 2.0), ("amplitude", 6.0), ("offset", 2.0), ("splitting", 7.5), ("fwhm_B", 3.0), ("amplitude_B", 3.0)];
        for (name, value) in truth {
            let error = result.error(name).unwrap();
            assert!(error.is_finite() && error > 0.0 && error < 0.5, "{name} error {error}");
            inside += usize::from((get(&result, name) - value).abs() <= 2.0 * error);
            total += 1;
        }
    }
    // Two standard errors hold about 95%; allow the binomial spread of 140.
    assert!(inside as f64 >= 0.85 * total as f64, "{inside} of {total} within two errors");
}

#[test]
fn the_student_evidence_is_twice_the_likelihood_gain() {
    let lines = [(-4.0, 2.0, 6.0), (3.5, 3.0, 3.0)];
    let (x, mean, sem) = repeated(&lines, 2.0, 0.25, 4, 9);
    let cell = repeat_cell(&x, &mean, &sem, Some(3.0));
    let wide = solve("lorentzian_doublet", &cell);
    let narrow = solve("lorentzian", &cell);
    let minus_two_log_likelihood = |r: &FitResult| -> f64 {
        r.residuals().iter().zip(&sem).map(|(res, s)| { let t = res / s; 4.0 * (t * t / 3.0).ln_1p() }).sum()
    };
    let n = x.len() as f64;
    let want = minus_two_log_likelihood(&narrow) - minus_two_log_likelihood(&wide) - 3.0 * n.ln();
    assert!((wide.evidence - want).abs() < 1e-6 * want.abs(), "{} vs {want}", wide.evidence);
}

#[test]
fn one_repeat_points_keep_their_stated_or_fallback_scale() {
    // Points of one repeat carry no scatter: a stated sigma is used as known
    // (infinite dof); none at all (NaN) falls back to the largest sigma, as
    // the Gaussian weighting does. Mixed with repeat-estimated points, the
    // fit still recovers the line.
    let line = (1.0, 3.0, 5.0);
    let (x, mean, mut sem) = repeated(&[line], 2.0, 0.25, 4, 3);
    let mut dof = vec![3.0; x.len()];
    for i in (0..x.len()).step_by(7) {
        dof[i] = f64::INFINITY;
        if i % 2 == 0 { sem[i] = f64::NAN; }
    }
    let mut cell = FitCell::points(vec![x.clone()], mean.clone());
    cell.sigma = Some(sem.into());
    cell.sigma_dof = Some(dof.into());
    let result = solve("lorentzian", &cell);
    assert!(result.success && (get(&result, "center") - 1.0).abs() < 0.1 && (get(&result, "fwhm") - 3.0).abs() < 0.3, "{:?}", result.parameter_values);
}

#[test]
fn counts_and_robust_losses_are_not_student() {
    // The Student-t likelihood is for means of repeats on the linear loss:
    // a robust loss the caller chose stays that loss.
    let line = (1.0, 3.0, 5.0);
    let (x, mean, sem) = repeated(&[line], 2.0, 0.25, 4, 3);
    let engine = FitEngine::default();
    let model = engine.registry.get("lorentzian").unwrap();
    let options = FitOptions { loss: "soft_l1".into(), ..FitOptions::default() };
    let request = FitRequest { initial: None, bounds: None, options: Some(options), cancelled: None };
    let with = engine.fit(&model, &repeat_cell(&x, &mean, &sem, Some(3.0)), &request).unwrap();
    let without = engine.fit(&model, &repeat_cell(&x, &mean, &sem, None), &request).unwrap();
    assert_eq!(with.parameter_values, without.parameter_values);
}

// ------------------------------------------------------------ unknown models

#[test]
fn a_target_naming_an_unknown_model_resolves_with_the_fit_inactive() {
    use zlc_plot::configuration::value::{Value, ValueMap};
    use zlc_plot::configuration::{FitLayer, FitTargetQuery, FitUnits};
    let layer = zlc_plot::session::EngineFitLayer::new(Arc::new(FitEngine::default()));
    let units = FitUnits { registry: None, value: None, axis_0: None, axis_1: None, histogram: false, rolling_primary: false };
    // Any model the catalogue lacks, the retired doublet among them: no
    // error, no request, and a message naming the model.
    for model in ["symmetric_lorentzian_doublet", "some_future_model"] {
        let target = ValueMap::from_pairs([("model", Value::text(model)), ("expression", Value::text("x_0=1"))]);
        let query = FitTargetQuery { target: &target, facet_grid: false, units: &units, live: false, previous_request: None, previous_expression_failure: None };
        let resolution = layer.resolve_target(&query).unwrap();
        assert!(resolution.request.is_none() && resolution.expression_failure.is_none());
        let message = resolution.unavailable.unwrap();
        assert!(message.contains(model) && message.contains("not applied"), "{message}");
    }
}
