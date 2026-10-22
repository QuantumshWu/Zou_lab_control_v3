//! The two-state classifier read from a bimodal fit
//! (`_classifier_threshold`, `_bimodal_classifier_metrics`), the authored
//! classifier result (`_authored_classifier_result`), and the presented
//! component curves (`evaluate_component`).

use std::sync::Arc;

use super::engine::evaluate_model;
use super::models::{Axes, Kernel};
use super::num::{erf, numpy_sum, pmax, pmin, pow, EPSILON};
use super::result::{FitError, FitResult};
use super::spec::{ComponentKind, Evaluator, FitModelSpec};

/// `_CLASSIFIER_MINIMUM_COMPONENT_SHOTS`.
pub const MINIMUM_COMPONENT_SHOTS: f64 = 4.0;

/// Gaussian CDF as the oracle writes it.
fn cdf(value: f64, mean: f64, sigma: f64) -> f64 { 0.5 * (1.0 + erf((value - mean) / (sigma * 2.0_f64.sqrt()))) }

/// Python float division: raises on zero.
fn pydiv(a: f64, b: f64) -> Result<f64, FitError> { super::num::div(a, b).map_err(|_| FitError::ZeroDivision) }

/// `_classifier_threshold(error, left, right, lower, upper)`.
pub fn classifier_threshold(error: &dyn Fn(f64) -> f64, left: (f64, f64, f64), right: (f64, f64, f64), lower: f64, upper: f64) -> Result<f64, FitError> {
    let (left_mean, left_sigma, left_weight) = left;
    let (right_mean, right_sigma, right_weight) = right;
    let rs2 = pow(right_sigma, 2.0);
    let ls2 = pow(left_sigma, 2.0);
    let quadratic = pydiv(0.5, rs2)? - pydiv(0.5, ls2)?;
    let linear = pydiv(left_mean, ls2)? - pydiv(right_mean, rs2)?;
    let ratio = pydiv(left_weight * right_sigma, right_weight * left_sigma)?;
    if ratio <= 0.0 {
        return Err(FitError::value("math domain error"));
    }
    let constant = pydiv(0.5 * pow(right_mean, 2.0), rs2)? - pydiv(0.5 * pow(left_mean, 2.0), ls2)? + ratio.ln();
    let mut candidates = vec![lower, upper];
    if quadratic == 0.0 {
        if linear != 0.0 {
            candidates.push(pydiv(-constant, linear)?);
        }
    } else {
        let discriminant = linear * linear - 4.0 * quadratic * constant;
        if discriminant >= 0.0 {
            let root = discriminant.sqrt();
            let shifted = -0.5 * (linear + root.copysign(linear));
            candidates.push(pydiv(shifted, quadratic)?);
            if shifted != 0.0 {
                candidates.push(pydiv(constant, shifted)?);
            }
        }
    }
    let mut best: Option<(f64, f64)> = None;
    for value in candidates {
        if !(lower <= value && value <= upper) {
            continue;
        }
        let key = error(value);
        match best {
            None => best = Some((value, key)),
            Some((_, best_key)) if key < best_key => best = Some((value, key)),
            _ => {}
        }
    }
    best.map(|(value, _)| value).ok_or_else(|| FitError::value("min() iterable argument is empty"))
}

/// `_bimodal_classifier_metrics(result, threshold)`: threshold (None for no
/// classifier), left and right fractions, fidelity.
pub fn bimodal_classifier_metrics(result: &FitResult, threshold: Option<f64>) -> Result<(Option<f64>, f64, f64, f64), FitError> {
    let none = (None, f64::NAN, f64::NAN, f64::NAN);
    if result.model.model_id != "bimodal_gaussian" || !result.success {
        return Err(FitError::value("threshold classification requires a successful bimodal fit"));
    }
    if result.reduced && threshold.is_none() {
        return Ok(none);
    }
    let get = |name: &str| result.parameter(name).expect("bimodal parameter");
    let left_mean = get("center");
    let right_mean = left_mean + get("delta_center");
    let left_sigma = pmax(get("sigma").abs(), EPSILON);
    let right_sigma = pmax(get("sigma_B").abs(), EPSILON);
    let right_weight = get("ratio");
    let shots = numpy_sum(&result.fitted_values());
    if !(right_weight.is_finite() && (0.0..=1.0).contains(&right_weight)) || !(shots.is_finite() && shots > 0.0) {
        return Ok(none);
    }
    let left_weight = 1.0 - right_weight;
    let error = |value: f64| left_weight * (1.0 - cdf(value, left_mean, left_sigma)) + right_weight * cdf(value, right_mean, right_sigma);
    let threshold = match threshold {
        Some(t) => t,
        None => {
            if pmin(left_weight, right_weight) * shots < MINIMUM_COMPONENT_SHOTS {
                return Ok(none);
            }
            let (lower, upper) = if right_mean < left_mean { (right_mean, left_mean) } else { (left_mean, right_mean) };
            if upper <= lower {
                lower
            } else {
                classifier_threshold(&error, (left_mean, left_sigma, left_weight), (right_mean, right_sigma, right_weight), lower, upper)?
            }
        }
    };
    let left_correct = cdf(threshold, left_mean, left_sigma);
    let right_correct = 1.0 - cdf(threshold, right_mean, right_sigma);
    let left_fraction = left_weight * cdf(threshold, left_mean, left_sigma) + right_weight * cdf(threshold, right_mean, right_sigma);
    let fidelity = if result.reduced { f64::NAN } else { left_weight * left_correct + right_weight * right_correct };
    Ok((Some(threshold), left_fraction, 1.0 - left_fraction, fidelity))
}

/// `evaluate_component(component_id, coordinates, values)`.
pub fn evaluate_component(model: &FitModelSpec, component_id: &str, coords: &[Vec<f64>], params: &[f64]) -> Result<Vec<f64>, FitError> {
    let requested = crate::pystr::strip(component_id);
    let component = model.presentation.components.iter().find(|c| c.component_id == requested).ok_or_else(|| FitError::value(format!("unknown fit component: {}", super::num::str_repr(component_id))))?;
    if params.len() != model.parameters.len() {
        return Err(FitError::value("fit parameter count does not match model"));
    }
    let Evaluator::Builtin(kernel) = &model.evaluator else { return Err(FitError::runtime("custom components are evaluated by the caller")) };
    let (single, p) = match kernel {
        Kernel::Bimodal => (Kernel::Histogram, params),
        Kernel::PoissonBimodal => (Kernel::Poisson, params),
        _ => return Err(FitError::value(format!("unknown fit component: {}", super::num::str_repr(component_id)))),
    };
    let (amplitude, location, sigma, delta, sigma_b, ratio) = (p[0], p[1], p[2], p[3], p[4], p[5]);
    let component_parameters = match component.kind {
        ComponentKind::LowerPopulation => [amplitude * (1.0 - ratio), location, sigma],
        ComponentKind::UpperPopulation => [amplitude * ratio, location + delta, sigma_b],
    };
    let first = model.anchored_axis(&coords[0]);
    Ok(single.value_jacobian(&Axes::one(&first), &component_parameters, false)?.0)
}

/// `_authored_classifier_result`: the caller's Gaussian pair, scaled to the
/// histogram by least squares, unsolved.
pub fn authored_classifier_result(
    model: &Arc<FitModelSpec>,
    coordinates: &[Vec<f64>],
    observed: &[f64],
    selected_indices: &[i64],
    data_revision: u64,
    components: &[(String, f64)],
    units: &[(String, String)],
) -> Result<FitResult, FitError> {
    let get = |name: &str| components.iter().find(|(n, _)| n == name).map(|(_, v)| *v).ok_or_else(|| FitError::value(format!("'{name}'")));
    let values = [("amplitude", 1.0), ("center", get("center")?), ("sigma", get("sigma")?), ("delta_center", get("delta_center")?), ("sigma_B", get("sigma_B")?), ("ratio", get("ratio")?)];
    let mut parameters: Vec<f64> = model.parameter_names().iter().map(|n| values.iter().find(|(k, _)| k == n).map(|(_, v)| *v).unwrap_or(f64::NAN)).collect();
    let shape = evaluate_model(model, coordinates, &parameters)?;
    let denominator = super::blas::ddot(&shape, &shape);
    let amplitude = if denominator > 0.0 { pmax(0.0, super::blas::ddot(&shape, observed) / denominator) } else { 0.0 };
    let index = model.parameter_index("amplitude")?;
    parameters[index] = amplitude;
    let fitted: Vec<f64> = shape.iter().map(|s| s * amplitude).collect();
    let residuals: Vec<f64> = observed.iter().zip(&fitted).map(|(o, f)| o - f).collect();
    let count = parameters.len();
    let reduced = super::blas::ddot(&residuals, &residuals) / (observed.len() as i64 - count as i64).max(1) as f64;
    let result = FitResult::new(
        model.clone(),
        parameters,
        vec![f64::NAN; count],
        vec![f64::NAN; count * count],
        FitResult::eager(fitted, residuals, selected_indices.to_vec()),
        data_revision,
        true,
        "authored Gaussian components",
        reduced,
        false,
        Vec::new(),
    )?;
    result.with_parameter_units(units)
}
