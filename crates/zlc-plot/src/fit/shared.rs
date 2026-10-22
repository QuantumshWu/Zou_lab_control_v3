//! The Python-side preparation every lane shares (`fit.py`'s private
//! helpers): bounds, histogram confinement, fixed partitions, weights,
//! binned statistics, seeds and the Poisson deviance -- NumPy semantics
//! reproduced (pairwise sums, `np.unique`, `np.median`).

use super::num::{numpy_median, numpy_sum, pmax, pmin, str_list_repr, EPSILON};
use super::result::{Bound, Bounds, FitError, ParamValues};
use super::spec::{FitModelSpec, ParameterDomain, UnitRelation};

pub const FIT_RSS_TIE_RELATIVE: f64 = 1e-10;
pub const COUNT_FLOOR: f64 = 1e-9;

/// `_span(x)`.
pub fn span(x: &[f64]) -> f64 {
    let (low, high) = min_max(x);
    pmax(high - low, EPSILON)
}

/// `np.min`, `np.max` of finite values.
pub fn min_max(x: &[f64]) -> (f64, f64) {
    let mut low = f64::INFINITY;
    let mut high = f64::NEG_INFINITY;
    for &v in x {
        if v < low || v.is_nan() {
            low = v;
        }
        if v > high || v.is_nan() {
            high = v;
        }
    }
    (low, high)
}

/// `np.unique(x)` (sorted, distinct).
pub fn unique_sorted(x: &[f64]) -> Vec<f64> {
    let mut values = x.to_vec();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    values.dedup_by(|a, b| *a == *b);
    values
}

/// `_histogram_step(x)`: the median distance between distinct bins.
pub fn histogram_step(x: &[f64]) -> f64 {
    let unique = unique_sorted(x);
    let width = span(x);
    let step = if unique.len() > 1 {
        let differences: Vec<f64> = unique.windows(2).map(|w| (w[1] - w[0]).abs()).collect();
        numpy_median(&differences)
    } else {
        pmax(width, EPSILON)
    };
    pmax(step, EPSILON)
}

/// `_confines_to_histogram(model)`.
pub fn confines_to_histogram(model: &FitModelSpec) -> bool { model.counted() }

/// `_histogram_bounds(model, coordinates, bounds)`.
pub fn histogram_bounds(model: &FitModelSpec, x: &[f64], bounds: Option<&Bounds>) -> Result<Option<Bounds>, FitError> {
    if !confines_to_histogram(model) {
        return Ok(bounds.cloned());
    }
    if x.is_empty() {
        return Ok(bounds.cloned());
    }
    let step = histogram_step(x);
    let (low_x, high_x) = min_max(x);
    let low_edge = low_x - 0.5 * step;
    let high_edge = high_x + 0.5 * step;
    let breadth = high_edge - low_edge;
    let mut confined = bounds.cloned().unwrap_or_default();
    for item in &model.parameters {
        if item.unit_relation != UnitRelation::Axis0 {
            continue;
        }
        let g = crate::pyfmt::format_g;
        let (limit, role) = if item.affine_point {
            ((low_edge, high_edge), format!("a location on the histogram's axis, which runs from {} to {}", g(low_edge, 6), g(high_edge, 6)))
        } else if item.domain == ParameterDomain::Positive {
            ((0.5 * step, breadth), format!("a width the histogram resolves, from half a bin ({}) to its breadth ({})", g(0.5 * step, 6), g(breadth, 6)))
        } else {
            ((-breadth, breadth), format!("a length on the histogram's axis, within its breadth ({})", g(breadth, 6)))
        };
        let (domain_low, domain_high) = item.bounds();
        let mut low = pmax(limit.0, domain_low);
        let mut high = pmin(limit.1, domain_high);
        let asked = confined.get(&item.name).unwrap_or((None, None));
        if let Some(a) = asked.0 {
            low = pmax(low, a);
        }
        if let Some(b) = asked.1 {
            high = pmin(high, b);
        }
        if low > high {
            return Err(FitError::value(format!("{} is {role}; the bound {} has nothing inside it", super::num::str_repr(&item.name), bound_repr(asked))));
        }
        confined.set(&item.name, (Some(low), Some(high)));
    }
    Ok(Some(confined))
}

fn optional_repr(value: Option<f64>) -> String {
    match value {
        None => "None".into(),
        Some(v) => crate::pyfmt::repr(v),
    }
}

/// Python's repr of a `(low, high)` tuple.
pub fn bound_repr(bound: Bound) -> String { format!("({}, {})", optional_repr(bound.0), optional_repr(bound.1)) }

/// `_solver_bounds(model, defaults, requested)`.
pub fn solver_bounds(model: &FitModelSpec, defaults: Option<&Bounds>, requested: Option<&Bounds>) -> Result<(Vec<f64>, Vec<f64>), FitError> {
    let names = model.parameter_names();
    for (label, source) in [("default bounds", defaults), ("bounds", requested)] {
        if let Some(source) = source {
            let mut unknown: Vec<&str> = source.names().filter(|n| !names.contains(n)).collect();
            unknown.sort();
            unknown.dedup();
            if !unknown.is_empty() {
                return Err(FitError::value(format!("{label} name unknown parameters: {}", str_list_repr(&unknown))));
            }
        }
    }
    let mut lower = Vec::with_capacity(names.len());
    let mut upper = Vec::with_capacity(names.len());
    for parameter in &model.parameters {
        let (mut low, mut high) = parameter.bounds();
        let mut exact = false;
        let from_requested = requested.is_some_and(|r| r.contains(&parameter.name));
        let source = if from_requested { requested } else { defaults };
        if let Some(pair) = source.and_then(|s| s.get(&parameter.name)) {
            exact = from_requested && pair.0.is_some() && pair.1.is_some() && pair.0 == pair.1;
            if let Some(a) = pair.0 {
                low = pmax(low, a);
            }
            if let Some(b) = pair.1 {
                high = pmin(high, b);
            }
        }
        if low > high || (low == high && !exact) {
            return Err(FitError::value(format!("empty bounds for parameter {}", super::num::str_repr(&parameter.name))));
        }
        lower.push(low);
        upper.push(high);
    }
    Ok((lower, upper))
}

/// `_fixed_parameter_partition(model, bounds)`: fixed names in model
/// order, free indices.
pub fn fixed_parameter_partition(model: &FitModelSpec, bounds: Option<&Bounds>) -> (Vec<String>, Vec<usize>) {
    let fixed: Vec<&str> = bounds.map(|b| b.0.iter().filter(|(_, (lo, hi))| lo.is_some() && hi.is_some() && lo == hi).map(|(n, _)| n.as_str()).collect()).unwrap_or_default();
    let names = model.parameter_names();
    (
        names.iter().filter(|n| fixed.contains(n)).map(|n| n.to_string()).collect(),
        names.iter().enumerate().filter(|(_, n)| !fixed.contains(n)).map(|(i, _)| i).collect(),
    )
}

/// `_observation_count(values, counted)`.
pub fn observation_count(values: &[f64], counted: bool) -> i64 {
    if counted {
        let total = numpy_sum(values);
        let shots = if total.is_finite() { total.trunc() as i64 } else { i64::MAX };
        (values.len() as i64).min(shots)
    } else {
        values.len() as i64
    }
}

/// `_too_few_points(model, free_count, points)`.
pub fn too_few_points(model: &FitModelSpec, free_count: usize, points: i64) -> FitError {
    FitError::value(format!("{} needs more points than its {free_count} free parameters: {points} finite here", model.model_id))
}

/// A warm start, which the product always gives whole (a previous
/// solution): one finite value per parameter, as `_initial_values` checks it.
pub fn warm_seed(model: &FitModelSpec, warm: &[f64]) -> Result<Vec<f64>, FitError> {
    if warm.len() != model.parameters.len() || warm.iter().any(|v| !v.is_finite()) {
        return Err(FitError::value("fit initializer returned invalid parameter values"));
    }
    Ok(warm.to_vec())
}

/// `_sigma_weights(sigma)`.
pub fn sigma_weights(sigma: &[f64]) -> Option<Vec<f64>> {
    let usable: Vec<f64> = sigma.iter().copied().filter(|s| s.is_finite() && *s > 0.0).collect();
    if usable.is_empty() {
        return None;
    }
    let (smallest, largest) = min_max(&usable);
    Some(sigma.iter().map(|&s| 1.0 / if s.is_finite() { pmax_np(s, smallest) } else { largest }).collect())
}

/// `np.maximum(a, b)` for finite operands (NaN propagates).
#[inline]
fn pmax_np(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    if a >= b { a } else { b }
}

/// `_binned_curve_statistics(x, values, bins)`.
pub fn binned_curve_statistics(x: &[f64], values: &[f64], bins: i64) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let (low, high) = min_max(x);
    if !low.is_finite() || !high.is_finite() || low == high {
        return None;
    }
    let scale = (bins - 1) as f64 / (high - low);
    let bins = bins as usize;
    let codes: Vec<usize> = x
        .iter()
        .map(|&v| {
            let code = ((v - low) * scale) as i64; // astype(int64) truncates
            code.clamp(0, bins as i64 - 1) as usize
        })
        .collect();
    let mut counts = vec![0i64; bins];
    for &c in &codes {
        counts[c] += 1;
    }
    let used: Vec<usize> = (0..bins).filter(|&b| counts[b] != 0).collect();
    if used.len() < 8 || used.len() * 2 > x.len() {
        return None;
    }
    let mut x_sums = vec![0.0; bins];
    let mut y_sums = vec![0.0; bins];
    for (i, &c) in codes.iter().enumerate() {
        x_sums[c] += x[i];
        y_sums[c] += values[i];
    }
    let weights: Vec<f64> = used.iter().map(|&b| counts[b] as f64).collect();
    let x_means: Vec<f64> = used.iter().zip(&weights).map(|(&b, &w)| x_sums[b] / w).collect();
    let y_means: Vec<f64> = used.iter().zip(&weights).map(|(&b, &w)| y_sums[b] / w).collect();
    Some((x_means, y_means, weights.iter().map(|w| w.sqrt()).collect()))
}

/// `_poisson_deviance_residual(fitted, observed)`.
pub fn poisson_deviance_residual(fitted: &[f64], observed: &[f64]) -> Vec<f64> {
    fitted
        .iter()
        .zip(observed)
        .map(|(&f, &n)| {
            let expected = pmax_np(f, COUNT_FLOOR);
            let logarithm = if n > 0.0 { n * (n / expected).ln() } else { 0.0 };
            let deviance = 2.0 * pmax_np(expected - n + logarithm, 0.0);
            deviance.sqrt().copysign(expected - n)
        })
        .collect()
}

/// `_authored_seed(model, initial)`: NaN where the guess names nothing.
pub fn authored_seed(model: &FitModelSpec, initial: &ParamValues) -> Result<Vec<f64>, FitError> {
    let names = model.parameter_names();
    let (seed, named): (Vec<f64>, Vec<f64>) = match initial {
        ParamValues::Named(items) => {
            let mut unknown: Vec<&str> = items.iter().map(|(n, _)| n.as_str()).filter(|n| !names.contains(n)).collect();
            if !unknown.is_empty() {
                unknown.sort();
                unknown.dedup();
                return Err(FitError::value(format!("initial values name unknown parameters: {}", str_list_repr(&unknown))));
            }
            (names.iter().map(|n| initial.get(n).unwrap_or(f64::NAN)).collect(), items.iter().map(|(_, v)| *v).collect())
        }
        ParamValues::Positional(values) => (values.clone(), values.clone()),
    };
    if seed.len() != names.len() || !named.iter().all(|v| v.is_finite()) {
        return Err(FitError::value("fit initializer returned invalid parameter values"));
    }
    Ok(seed)
}

/// `_expand_fixed_covariance`.
pub fn expand_fixed_covariance(count: usize, free_indices: &[usize], free_covariance: &[f64], valid: bool) -> (Vec<f64>, Vec<f64>) {
    let mut covariance = vec![0.0; count * count];
    if !valid {
        return (covariance, vec![f64::NAN; count]);
    }
    let free = free_indices.len();
    for (a, &i) in free_indices.iter().enumerate() {
        for (b, &j) in free_indices.iter().enumerate() {
            covariance[i * count + j] = free_covariance[a * free + b];
        }
    }
    let errors = (0..count).map(|i| pmax(covariance[i * count + i], 0.0).sqrt()).collect();
    (covariance, errors)
}
