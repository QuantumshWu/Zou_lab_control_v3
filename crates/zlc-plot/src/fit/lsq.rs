//! The fallback lane: a model with no compiled descriptor (a caller's own
//! model, or a built-in replaced outside the registry).  The oracle solves
//! it with SciPy's `least_squares`; this port keeps its semantics -- the
//! candidate set, clipping and de-duplication of seeds, the invalid-residual
//! sentinel, ranking of candidates by cost with the relative tie, the warm
//! shortcut, the SVD rank test of the covariance -- and solves each seed
//! with the compiled lane's trust-region solver (see `TOLERANCES.md`).

use std::sync::Arc;

use super::engine::{evaluate_model, Check, FitCell, FitEngine, FitRequest, CellData};
use super::models::{point_terms, Axes};
use super::num::{nextafter, pmax, pmin, Zr, EPSILON};
use super::result::{Bounds, FitError, FitOptions, FitResult};
use super::shared::{
    binned_curve_statistics, expand_fixed_covariance, fixed_parameter_partition, histogram_bounds, observation_count, poisson_deviance_residual, sigma_weights, solver_bounds,
    too_few_points, FIT_RSS_TIE_RELATIVE,
};
use super::spec::{Evaluator, FitModelSpec, DOMAIN_ANCHORED};
use super::trf::{self, CellObjective};

struct ResidualObjective<'a> {
    model: &'a FitModelSpec,
    coords: &'a [Vec<f64>],
    values: &'a [f64],
    weights: Option<&'a [f64]>,
    counted: bool,
    loss: i64,
    lower: &'a [f64],
    upper: &'a [f64],
    free: &'a [usize],
}

impl ResidualObjective<'_> {
    fn predict(&self, params: &[f64]) -> Option<Vec<f64>> {
        let predicted = evaluate_model(self.model, self.coords, params).ok()?;
        if predicted.len() != self.values.len() || predicted.iter().any(|v| !v.is_finite()) {
            return None;
        }
        Some(predicted)
    }

    fn jacobian(&self, params: &[f64]) -> Option<Vec<f64>> {
        let n = self.values.len();
        let p = params.len();
        if let Evaluator::Custom(custom) = &self.model.evaluator {
            let first = self.model.anchored_axis(&self.coords[0]);
            let mut axes: Vec<&[f64]> = vec![&first];
            axes.extend(self.coords[1..].iter().map(|c| c.as_slice()));
            if let Some(j) = custom.jacobian(&axes, params) {
                return Some(j);
            }
        } else if let Evaluator::Builtin(kernel) = &self.model.evaluator {
            let first = self.model.anchored_axis(&self.coords[0]);
            let axes = if self.coords.len() == 1 { Axes::one(&first) } else { Axes::two(&first, &self.coords[1]) };
            return kernel.value_jacobian(&axes, params, true).ok().map(|(_, j)| j);
        }
        // SciPy's 2-point difference, stepping away from a bound it would cross.
        let base = self.predict(params)?;
        let mut jac = vec![0.0; n * p];
        for &j in self.free {
            let x = params[j];
            let mut h = EPSILON.sqrt() * pmax(1.0, x.abs()) * if x < 0.0 { -1.0 } else { 1.0 };
            let free_index = self.free.iter().position(|&f| f == j).expect("free");
            let _ = free_index;
            if x + h > self.upper[j] || x + h < self.lower[j] {
                h = -h;
            }
            let mut shifted = params.to_vec();
            shifted[j] = x + h;
            let dx = shifted[j] - x;
            let moved = self.predict(&shifted)?;
            for i in 0..n {
                jac[i * p + j] = (moved[i] - base[i]) / dx;
            }
        }
        Some(jac)
    }
}

impl CellObjective for ResidualObjective<'_> {
    fn evaluate(&self, params: &[f64], free: &[usize], gradient: &mut [f64], information: &mut [f64], jacobian_row: &mut [f64], derivatives: bool) -> Zr<(f64, f64, bool)> {
        let Some(predicted) = self.predict(params) else { return Ok((f64::INFINITY, f64::INFINITY, false)) };
        let jacobian = if derivatives {
            match self.jacobian(params) {
                Some(j) if j.iter().all(|v| v.is_finite()) => Some(j),
                _ => return Ok((f64::INFINITY, f64::INFINITY, false)),
            }
        } else {
            None
        };
        if derivatives {
            super::models::reset_accumulators(gradient, information);
        }
        let p = params.len();
        let mut cost = 0.0;
        let mut rss = 0.0;
        for i in 0..predicted.len() {
            let weight = self.weights.map_or(1.0, |w| w[i]);
            let (_raw, squared, point_cost, gf, inf, finite) = point_terms(predicted[i], self.values[i], self.counted, weight, self.weights.is_some(), self.loss)?;
            if !finite {
                return Ok((f64::INFINITY, f64::INFINITY, false));
            }
            if let Some(j) = &jacobian {
                for (k, &f) in free.iter().enumerate() {
                    jacobian_row[k] = j[i * p + f];
                }
                super::models::accumulate(&jacobian_row[..free.len()], gradient, information, gf, inf);
            }
            cost += point_cost;
            rss += squared;
        }
        if derivatives {
            super::models::finish_information(information, free.len());
        }
        Ok((cost, rss, true))
    }
    fn selected(&self) -> usize { self.values.len() }
}

/// `fit()` for a model without a compiled descriptor.
pub fn fit_fallback(engine: &FitEngine, spec_in: &Arc<FitModelSpec>, cell: &FitCell, request: &FitRequest, opts: &FitOptions) -> Result<FitResult, FitError> {
    let CellData::Points { coordinates, observations: Some(observations) } = &cell.data else {
        return Err(FitError::type_error("observations are required for coordinate-array fitting"));
    };
    let mut spec = spec_in.clone();
    if coordinates.len() != spec.independent_arity {
        return Err(FitError::value("coordinate arity does not match fit model"));
    }
    if coordinates.iter().any(|a| a.len() != observations.len()) {
        return Err(FitError::value("coordinates and observations must have equal flattened shape"));
    }
    let counted = spec.counted();
    if let Some(sigma) = &cell.sigma {
        if sigma.len() != observations.len() {
            return Err(FitError::value("observation_sigma must match observations"));
        }
        if counted {
            return Err(FitError::value("histogram targets weight themselves by counts; an external sigma has no meaning there"));
        }
    }
    if let Some(indices) = &cell.selected_indices {
        if indices.len() != observations.len() {
            return Err(FitError::value("selected_indices must match observations"));
        }
    }
    let finite: Vec<bool> = (0..observations.len()).map(|i| observations[i].is_finite() && coordinates.iter().all(|a| a[i].is_finite())).collect();
    let keep = |s: &[f64]| -> Vec<f64> { s.iter().zip(&finite).filter(|(_, f)| **f).map(|(&v, _)| v).collect() };
    let coords: Vec<Vec<f64>> = coordinates.iter().map(|a| keep(a)).collect();
    let values = keep(observations);
    let indices: Vec<i64> = match &cell.selected_indices {
        Some(i) => i.iter().zip(&finite).filter(|(_, f)| **f).map(|(&v, _)| v).collect(),
        None => (0..observations.len() as i64).zip(&finite).filter(|(_, f)| **f).map(|(v, _)| v).collect(),
    };
    let sigma = cell.sigma.as_ref().map(|s| keep(s));
    let bounds = histogram_bounds(&spec, &coords[0], request.bounds.as_ref())?;
    let (fixed_names, free_indices) = fixed_parameter_partition(&spec, bounds.as_ref());
    let check = Check::new(opts.deadline_seconds, request.cancelled);
    check.check()?;
    solver_bounds(&spec, None, bounds.as_ref())?;
    let points = observation_count(&values, counted);
    if points <= free_indices.len() as i64 {
        return Err(too_few_points(&spec, free_indices.len(), points));
    }
    if spec.capabilities.contains(DOMAIN_ANCHORED) {
        let (low, _) = super::shared::min_max(&coords[0]);
        spec = spec.anchored_at(low)?;
    }
    let (mut solver_coords, mut solver_values, mut weights, mut binned) = (coords.clone(), values.clone(), None, false);
    if let Some(sigma) = &sigma {
        weights = sigma_weights(sigma);
    } else if !free_indices.is_empty() && !counted && spec.independent_arity == 1 && opts.loss == "linear" {
        if let Some(max_exact) = opts.max_exact_points {
            if values.len() as i64 > 2 * max_exact {
                if let Some((x, y, w)) = binned_curve_statistics(&coords[0], &values, max_exact) {
                    solver_coords = vec![x];
                    solver_values = y;
                    weights = Some(w);
                    binned = true;
                }
            }
        }
    }
    let default_bounds: Option<Bounds> = match &spec.evaluator {
        Evaluator::Builtin(_) if spec.compiled_bounds => Some(super::pylane::compiled_bounds(&spec, &solver_coords, &solver_values)?),
        Evaluator::Custom(custom) => {
            let first = spec.anchored_axis(&solver_coords[0]);
            let mut axes: Vec<&[f64]> = vec![&first];
            axes.extend(solver_coords[1..].iter().map(|c| c.as_slice()));
            custom.bounds(&axes, &solver_values).map(Bounds)
        }
        _ => None,
    };
    let (lower, upper) = solver_bounds(&spec, default_bounds.as_ref(), bounds.as_ref())?;
    let count = spec.parameters.len();
    if free_indices.is_empty() {
        check.check()?;
        let fitted = evaluate_model(&spec, &coords, &lower)?;
        if fitted.len() != values.len() || fitted.iter().any(|v| !v.is_finite()) {
            return Err(FitError::runtime("fixed fit evaluation is non-finite"));
        }
        let residuals: Vec<f64> = values.iter().zip(&fitted).map(|(v, f)| v - f).collect();
        let quality: Vec<f64> = if counted {
            poisson_deviance_residual(&fitted, &values)
        } else {
            match &weights {
                None => residuals.clone(),
                Some(w) => residuals.iter().zip(w).map(|(r, w)| r * w).collect(),
            }
        };
        let reduced = super::blas::ddot(&quality, &quality) / values.len() as f64;
        return FitResult::new(spec, lower, vec![0.0; count], vec![0.0; count * count], FitResult::eager(fitted, residuals, indices), cell.data_revision as u64, true, "all parameters fixed", reduced, true, fixed_names);
    }
    let complete_seeds = super::pylane::initial_candidates(&spec, &solver_coords, &solver_values, request.initial.as_ref(), cell.warm_start.as_deref())?;
    let mut seeds: Vec<Vec<f64>> = Vec::new();
    for complete in complete_seeds {
        let free_seed: Vec<f64> = free_indices
            .iter()
            .map(|&i| {
                let low_inside = nextafter(lower[i], upper[i]);
                let high_inside = nextafter(upper[i], lower[i]);
                pmin(pmax(complete[i], low_inside), high_inside)
            })
            .collect();
        if !seeds.iter().any(|s| s.iter().zip(&free_seed).all(|(a, b)| a.to_bits() == b.to_bits())) {
            seeds.push(free_seed);
        }
    }
    let loss = super::models::loss_code(&opts.loss).expect("validated loss");
    let objective = ResidualObjective { model: &spec, coords: &solver_coords, values: &solver_values, weights: weights.as_deref(), counted, loss, lower: &lower, upper: &upper, free: &free_indices };
    let free_lower: Vec<f64> = free_indices.iter().map(|&i| lower[i]).collect();
    let free_upper: Vec<f64> = free_indices.iter().map(|&i| upper[i]).collect();
    let mut successful: Vec<(f64, f64, Vec<f64>, i32)> = Vec::new();
    let mut unsuccessful: Vec<(f64, f64, Vec<f64>, i32)> = Vec::new();
    let budget = opts.evaluation_budget(values.len());
    for (seed_index, seed) in seeds.iter().enumerate() {
        check.check()?;
        let outcome = match trf::solve_seed(&objective, &lower, &free_indices, seed, &free_lower, &free_upper, budget, 1e-8, 1e-8, 1e-8) {
            Ok(o) => o,
            Err(_) => continue,
        };
        if outcome.status == trf::STATUS_INVALID || !outcome.cost.is_finite() || !outcome.raw_rss.is_finite() {
            continue;
        }
        let cost = 2.0 * outcome.cost;
        let mut full = lower.clone();
        for (k, &i) in free_indices.iter().enumerate() {
            full[i] = outcome.values[k];
        }
        let success = outcome.status > trf::STATUS_MAX_NFEV;
        let entry = (cost, outcome.raw_rss, full, outcome.status);
        if success { successful.push(entry) } else { unsuccessful.push(entry) }
        if cell.warm_start.is_some() && seed_index == 0 && success && outcome.raw_rss <= FIT_RSS_TIE_RELATIVE {
            break;
        }
    }
    let candidates = if successful.is_empty() { unsuccessful } else { successful };
    if candidates.is_empty() {
        return Err(FitError::runtime("least-squares failed for every initializer"));
    }
    let mut best = 0;
    for index in 1..candidates.len() {
        let scale = pmax(1.0, candidates[best].0.abs());
        if candidates[index].0 < candidates[best].0 - FIT_RSS_TIE_RELATIVE * scale {
            best = index;
        }
    }
    let (_cost, mut rss, solved, status) = candidates[best].clone();
    let fitted = evaluate_model(&spec, &coords, &solved)?;
    if fitted.len() != values.len() || fitted.iter().any(|v| !v.is_finite()) {
        return Err(FitError::runtime("winning fit evaluation is non-finite"));
    }
    let residuals: Vec<f64> = values.iter().zip(&fitted).map(|(v, f)| v - f).collect();
    let degrees = (values.len() as i64 - free_indices.len() as i64).max(1) as f64;
    if binned {
        rss = super::blas::ddot(&residuals, &residuals);
    } else if let Some(w) = &weights {
        let weighted: Vec<f64> = residuals.iter().zip(w).map(|(r, w)| r * w).collect();
        rss = super::blas::ddot(&weighted, &weighted);
    }
    let reduced = rss / degrees;
    // The modified Jacobian at the solution, and the SVD rank test.
    let (free_covariance, covariance_valid) = covariance_at(&objective, &solved, &free_indices, reduced);
    let (covariance, errors) = expand_fixed_covariance(count, &free_indices, &free_covariance, covariance_valid);
    let result = FitResult::new(
        spec.clone(),
        solved,
        errors,
        covariance,
        FitResult::eager(fitted, residuals, indices),
        cell.data_revision as u64,
        status > trf::STATUS_MAX_NFEV,
        &trf::termination_message(status),
        reduced,
        covariance_valid,
        fixed_names,
    )?;
    let nested_cell = FitCell { warm_start: None, ..cell.clone() };
    let nested = engine.nested_population_fit(spec_in, std::slice::from_ref(&nested_cell), request.bounds.as_ref(), opts, request.cancelled)?;
    engine.settle_population(&spec, result, nested.into_iter().next().flatten().as_ref(), opts)
}

/// `_covariance(jac, reduced)`: the rank-tested inverse information of the
/// loss-scaled Jacobian at the solution.
fn covariance_at(objective: &ResidualObjective, solved: &[f64], free: &[usize], reduced: f64) -> (Vec<f64>, bool) {
    let n = free.len();
    let invalid = (vec![f64::NAN; n * n], false);
    let Some(jac) = objective.jacobian(solved) else { return invalid };
    let Some(predicted) = objective.predict(solved) else { return invalid };
    let points = objective.values.len();
    let p = solved.len();
    if points <= n || !reduced.is_finite() || reduced < 0.0 {
        return invalid;
    }
    let mut scaled = vec![0.0; points * n];
    for i in 0..points {
        let weight = objective.weights.map_or(1.0, |w| w[i]);
        let Ok((raw, _sq, _c, _g, _inf, finite)) = point_terms(predicted[i], objective.values[i], objective.counted, weight, objective.weights.is_some(), objective.loss) else { return invalid };
        if !finite {
            return invalid;
        }
        let Ok((_r0, rho1, rho2)) = super::models::rho(raw * raw, objective.loss) else { return invalid };
        let robust = pmax(rho1 + 2.0 * rho2 * raw * raw, EPSILON).sqrt();
        let scale = if objective.counted {
            let expected = pmax(predicted[i], super::models::COUNT_FLOOR);
            let absolute = raw.abs();
            if absolute > super::models::COUNT_FLOOR { (expected - objective.values[i]).abs() / (expected * pmax(absolute, super::models::COUNT_FLOOR)) } else { 1.0 / expected.sqrt() }
        } else {
            weight
        };
        for (k, &f) in free.iter().enumerate() {
            scaled[i * n + k] = jac[i * p + f] * scale * robust;
        }
    }
    if scaled.iter().any(|v| !v.is_finite()) {
        return invalid;
    }
    let Ok(r) = super::linalg::qr_r(points, n, &scaled) else { return invalid };
    let Ok((s, vt)) = super::linalg::svd_small(n, &r) else { return invalid };
    let tolerance = EPSILON * points.max(n) as f64 * s[0];
    if !s.iter().all(|v| v.is_finite()) || s[n - 1] <= tolerance {
        return invalid;
    }
    let mut covariance = vec![0.0; n * n];
    for a in 0..n {
        for b in 0..n {
            let mut value = 0.0;
            for k in 0..n {
                value += vt[k * n + a] * vt[k * n + b] / (s[k] * s[k]);
            }
            covariance[a * n + b] = value * reduced;
        }
    }
    for a in 0..n {
        for b in 0..a {
            let v = (covariance[a * n + b] + covariance[b * n + a]) / 2.0;
            covariance[a * n + b] = v;
            covariance[b * n + a] = v;
        }
    }
    if covariance.iter().any(|v| !v.is_finite()) || (0..n).any(|i| covariance[i * n + i] < 0.0) {
        return invalid;
    }
    (covariance, true)
}
