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
/// listed by center.
fn assert_lines(result: &FitResult, want: [Line; 2], offset: f64, tolerance: f64) {
    assert!(result.success, "{}", result.message);
    assert!(!result.reduced, "settled on one line: {}", result.message);
    let got = [(get(result, "center"), get(result, "fwhm"), get(result, "amplitude")), (get(result, "center_B"), get(result, "fwhm_B"), get(result, "amplitude_B"))];
    assert!(got[0].0 <= got[1].0, "lines out of order: {got:?}");
    for (g, w) in got.iter().zip(want) {
        assert!((g.0 - w.0).abs() < tolerance && (g.1 - w.1).abs() < tolerance && (g.2 - w.2).abs() < tolerance, "{got:?} vs {want:?}");
    }
    assert!((get(result, "offset") - offset).abs() < tolerance, "offset {}", get(result, "offset"));
}

#[test]
fn the_model_is_two_independent_lines_reducing_to_one() {
    let (_, model) = model();
    assert_eq!(model.parameter_names(), ["center", "fwhm", "amplitude", "offset", "center_B", "fwhm_B", "amplitude_B"]);
    assert_eq!(model.symbols(), ["x_0", "FWHM", "H", "B", "x_B", "FWHM_B", "H_B"]);
    assert_eq!(model.headline, "center");
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
        assert_eq!(get(&result, "center_B"), get(&result, "center"));
        assert_eq!(get(&result, "fwhm_B"), get(&result, "fwhm"));
        assert_eq!(get(&result, "amplitude_B"), 0.0);
        for name in ["center_B", "fwhm_B", "amplitude_B"] {
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
fn fixing_one_line_keeps_the_callers_labels() {
    // A caller who fixes line B's width has named the lines: no reordering.
    let lines = [(-6.0, 2.0, 5.0), (4.0, 3.0, 2.0)];
    let (engine, model) = model();
    let (x, y) = data(&lines, 1.0, 0.0, 1);
    let bounds = Bounds(vec![("fwhm".into(), (Some(3.0), Some(3.0)))]);
    let request = FitRequest { initial: None, bounds: Some(bounds), options: None, cancelled: None };
    let result = engine.fit(&model, &FitCell::points(vec![x], y), &request).unwrap();
    assert!(result.success && !result.reduced, "{}", result.message);
    assert!((get(&result, "center") - 4.0).abs() < 1e-6 && (get(&result, "center_B") + 6.0).abs() < 1e-6, "{:?}", result.parameter_values);
}
