//! The compiled lane's model-neutral trust-region-reflective solver
//! (`_fit_compiled._solve_seed`, `_solve_cell`, `_prepare_one`,
//! `_finalize_one`), line for line.

use rayon::prelude::*;
use super::linalg::{self, LinAlgError};
use super::models::{point_terms, rho, COUNT_FLOOR};
use super::num::{div, nextafter, pmax, pmin, KernelError, Zr, EPSILON};

pub const RSS_TIE_RELATIVE: f64 = 1.0e-10;
pub const TRUST_BOUNDARY_FRACTION: f64 = 0.95;

pub const STATUS_INVALID: i32 = -2;
pub const STATUS_NO_CANDIDATE: i32 = -1;
pub const STATUS_MAX_NFEV: i32 = 0;
pub const STATUS_GTOL: i32 = 1;
pub const STATUS_FTOL: i32 = 2;
pub const STATUS_XTOL: i32 = 3;
pub const STATUS_FTOL_XTOL: i32 = 4;

/// `termination_message(status)`.
pub fn termination_message(status: i32) -> String {
    match status {
        STATUS_INVALID => "compiled fit encountered non-finite numerical state".into(),
        STATUS_NO_CANDIDATE => "compiled fit has no finite initializer".into(),
        STATUS_MAX_NFEV => "The maximum number of function evaluations is exceeded.".into(),
        STATUS_GTOL => "`gtol` termination condition is satisfied.".into(),
        STATUS_FTOL => "`ftol` termination condition is satisfied.".into(),
        STATUS_XTOL => "`xtol` termination condition is satisfied.".into(),
        STATUS_FTOL_XTOL => "Both `ftol` and `xtol` termination conditions are satisfied.".into(),
        other => format!("compiled fit terminated with status {other}"),
    }
}

/// What a numba kernel can raise inside `_solve_seed`: both are caught by
/// `_solve_cell`'s `except Exception`.
#[derive(Debug, Clone, PartialEq)]
pub enum SolveError {
    Kernel(KernelError),
    LinAlg(LinAlgError),
}

impl From<KernelError> for SolveError {
    fn from(error: KernelError) -> Self { SolveError::Kernel(error) }
}

impl From<LinAlgError> for SolveError {
    fn from(error: LinAlgError) -> Self { SolveError::LinAlg(error) }
}

impl std::fmt::Display for SolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SolveError::Kernel(error) => write!(f, "{error}"),
            SolveError::LinAlg(error) => write!(f, "{error}"),
        }
    }
}

/// One cell's objective as the descriptor's callback sees it.
pub trait CellObjective: Sync {
    /// `(cost, raw_rss, finite)`; fills the free gradient and the row-major
    /// free information when `derivatives`.
    fn evaluate(&self, params: &[f64], free: &[usize], gradient: &mut [f64], information: &mut [f64], jacobian_row: &mut [f64], derivatives: bool) -> Zr<(f64, f64, bool)>;
    /// How many observations are selected (valid).
    fn selected(&self) -> usize;
    /// A solve still short of its tolerances after [`STALL_ITERATIONS`]: the
    /// objective may finish it its own way from the free `values` (`params`
    /// holds them in place; `cost` and `raw_rss` are theirs) within `budget`
    /// more evaluations. `None`: the solve goes on as it was.
    #[allow(clippy::too_many_arguments)]
    fn finish_stalled(&self, _params: &[f64], _free: &[usize], _values: &[f64], _cost: f64, _raw_rss: f64, _lower: &[f64], _upper: &[f64],
                      _budget: i64, _tolerances: (f64, f64, f64)) -> Option<Result<SeedOutcome, SolveError>> { None }
}

/// The iterations after which a solve that has not met its tolerances is
/// offered to its objective's [`CellObjective::finish_stalled`]. Every solve
/// that settles sooner is untouched.
pub const STALL_ITERATIONS: i32 = 100;

// --------------------------------------------------------------------------
// Small dense helpers of the step selection
// --------------------------------------------------------------------------

#[inline]
fn vector_norm(values: &[f64]) -> f64 {
    let mut total = 0.0;
    for &v in values {
        total += v * v;
    }
    pmax(total, 0.0).sqrt()
}

#[inline]
fn quadratic(information: &[f64], gradient: &[f64], step: &[f64]) -> f64 {
    let n = step.len();
    let mut quad = 0.0;
    let mut linear = 0.0;
    for row in 0..n {
        linear += gradient[row] * step[row];
        let mut product = 0.0;
        for column in 0..n {
            product += information[row * n + column] * step[column];
        }
        quad += step[row] * product;
    }
    0.5 * quad + linear
}

#[inline]
fn quadratic_line(information: &[f64], gradient: &[f64], direction: &[f64], origin: &[f64], has_origin: bool) -> (f64, f64, f64) {
    let n = direction.len();
    let mut product = [0.0_f64; 8];
    let mut second = 0.0;
    let mut first = 0.0;
    for row in 0..n {
        let mut value = 0.0;
        for column in 0..n {
            value += information[row * n + column] * direction[column];
        }
        product[row] = value;
        second += direction[row] * value;
        first += gradient[row] * direction[row];
    }
    let mut constant = 0.0;
    if has_origin {
        for index in 0..n {
            first += origin[index] * product[index];
        }
        constant = quadratic(information, gradient, origin);
    }
    (0.5 * second, first, constant)
}

#[inline]
fn minimize_quadratic(second: f64, first: f64, lower: f64, upper: f64, constant: f64) -> Zr<(f64, f64)> {
    let mut best_step = lower;
    let mut best_value = lower * (second * lower + first) + constant;
    let value = upper * (second * upper + first) + constant;
    if value < best_value {
        best_step = upper;
        best_value = value;
    }
    if second != 0.0 {
        let candidate = div(-0.5 * first, second)?;
        if lower < candidate && candidate < upper {
            let value = candidate * (second * candidate + first) + constant;
            if value < best_value {
                best_step = candidate;
                best_value = value;
            }
        }
    }
    Ok((best_step, best_value))
}

#[inline]
fn step_to_bound(values: &[f64], step: &[f64], lower: &[f64], upper: &[f64], hits: &mut [i32]) -> f64 {
    let mut minimum = f64::INFINITY;
    for index in 0..values.len() {
        hits[index] = 0;
        if step[index] == 0.0 {
            continue;
        }
        let low = (lower[index] - values[index]) / step[index];
        let high = (upper[index] - values[index]) / step[index];
        let forward = pmax(low, high);
        if forward < minimum {
            minimum = forward;
        }
    }
    for index in 0..values.len() {
        if step[index] == 0.0 {
            continue;
        }
        let low = (lower[index] - values[index]) / step[index];
        let high = (upper[index] - values[index]) / step[index];
        if pmax(low, high) == minimum {
            hits[index] = if step[index] > 0.0 { 1 } else { -1 };
        }
    }
    minimum
}

#[inline]
fn positive_intersection(origin: &[f64], direction: &[f64], radius: f64) -> Zr<f64> {
    let mut quad = 0.0;
    let mut linear = 0.0;
    let mut constant = -radius * radius;
    for index in 0..origin.len() {
        quad += direction[index] * direction[index];
        linear += origin[index] * direction[index];
        constant += origin[index] * origin[index];
    }
    let discriminant = pmax(linear * linear - quad * constant, 0.0);
    let root = discriminant.sqrt();
    let qvalue = -(linear + root.copysign(linear));
    if qvalue == 0.0 {
        return div(-linear, quad);
    }
    let first = div(qvalue, quad)?;
    let second = div(constant, qvalue)?;
    Ok(pmax(first, second))
}

#[inline]
fn inside_bounds(values: &[f64], step: &[f64], lower: &[f64], upper: &[f64]) -> bool {
    for index in 0..values.len() {
        let candidate = values[index] + step[index];
        if candidate < lower[index] || candidate > upper[index] {
            return false;
        }
    }
    true
}

#[allow(clippy::too_many_arguments)]
fn select_reflective_step(
    values: &[f64],
    information: &[f64],
    gradient: &[f64],
    unconstrained: &mut [f64],
    scaling: &[f64],
    radius: f64,
    lower: &[f64],
    upper: &[f64],
    theta: f64,
    step: &mut [f64],
    scaled_step: &mut [f64],
) -> Zr<f64> {
    let count = values.len();
    let mut direct = [0.0_f64; 8];
    for index in 0..count {
        direct[index] = scaling[index] * unconstrained[index];
    }
    if inside_bounds(values, &direct[..count], lower, upper) {
        for index in 0..count {
            step[index] = direct[index];
            scaled_step[index] = unconstrained[index];
        }
        return Ok(-quadratic(information, gradient, unconstrained));
    }
    let mut hits = [0_i32; 8];
    let direct_stride = step_to_bound(values, &direct[..count], lower, upper, &mut hits[..count]);
    let mut reflected_scaled = [0.0_f64; 8];
    let mut reflected = [0.0_f64; 8];
    let mut at_bound = [0.0_f64; 8];
    for index in 0..count {
        reflected_scaled[index] = if hits[index] != 0 { -unconstrained[index] } else { unconstrained[index] };
        direct[index] *= direct_stride;
        unconstrained[index] *= direct_stride;
        at_bound[index] = values[index] + direct[index];
        reflected[index] = scaling[index] * reflected_scaled[index];
    }
    let to_radius = positive_intersection(unconstrained, &reflected_scaled[..count], radius)?;
    let mut dummy_hits = [0_i32; 8];
    let to_bound = step_to_bound(&at_bound[..count], &reflected[..count], lower, upper, &mut dummy_hits[..count]);
    let reflected_stride = pmin(to_bound, to_radius);
    let (reflected_lower, reflected_upper) = if reflected_stride > 0.0 {
        (div((1.0 - theta) * direct_stride, reflected_stride)?, if reflected_stride == to_bound { theta * to_bound } else { to_radius })
    } else {
        (0.0, -1.0)
    };
    let reflected_value;
    if reflected_lower <= reflected_upper {
        let (second, first, constant) = quadratic_line(information, gradient, &reflected_scaled[..count], unconstrained, true);
        let (stride, value) = minimize_quadratic(second, first, reflected_lower, reflected_upper, constant)?;
        reflected_value = value;
        for index in 0..count {
            reflected_scaled[index] = unconstrained[index] + stride * reflected_scaled[index];
            reflected[index] = scaling[index] * reflected_scaled[index];
        }
    } else {
        reflected_value = f64::INFINITY;
    }
    for index in 0..count {
        direct[index] *= theta;
        unconstrained[index] *= theta;
    }
    let direct_value = quadratic(information, gradient, unconstrained);

    let mut anti_gradient_scaled = [0.0_f64; 8];
    let mut anti_gradient = [0.0_f64; 8];
    let mut norm_squared = 0.0;
    for index in 0..count {
        anti_gradient_scaled[index] = -gradient[index];
        anti_gradient[index] = scaling[index] * anti_gradient_scaled[index];
        norm_squared += anti_gradient_scaled[index] * anti_gradient_scaled[index];
    }
    let (anti_gradient_stride, anti_gradient_value);
    if norm_squared == 0.0 {
        anti_gradient_stride = 0.0;
        anti_gradient_value = 0.0;
    } else {
        let to_radius = div(radius, norm_squared.sqrt())?;
        let to_bound = step_to_bound(values, &anti_gradient[..count], lower, upper, &mut dummy_hits[..count]);
        let maximum = if to_bound < to_radius { theta * to_bound } else { to_radius };
        let (second, first, _constant) = quadratic_line(information, gradient, &anti_gradient_scaled[..count], &anti_gradient_scaled[..count], false);
        let (stride, value) = minimize_quadratic(second, first, 0.0, maximum, 0.0)?;
        anti_gradient_stride = stride;
        anti_gradient_value = value;
    }
    for index in 0..count {
        anti_gradient_scaled[index] *= anti_gradient_stride;
        anti_gradient[index] *= anti_gradient_stride;
    }
    if direct_value < reflected_value && direct_value < anti_gradient_value {
        for index in 0..count {
            step[index] = direct[index];
            scaled_step[index] = unconstrained[index];
        }
        return Ok(-direct_value);
    }
    if reflected_value < direct_value && reflected_value < anti_gradient_value {
        for index in 0..count {
            step[index] = reflected[index];
            scaled_step[index] = reflected_scaled[index];
        }
        return Ok(-reflected_value);
    }
    for index in 0..count {
        step[index] = anti_gradient[index];
        scaled_step[index] = anti_gradient_scaled[index];
    }
    Ok(-anti_gradient_value)
}

/// `_make_feasible`.
pub fn make_feasible(values: &mut [f64], lower: &[f64], upper: &[f64], relative_step: f64) {
    for index in 0..values.len() {
        let lower_distance = values[index] - lower[index];
        let upper_distance = upper[index] - values[index];
        let lower_active = lower[index].is_finite() && lower_distance <= pmin(upper_distance, relative_step * pmax(1.0, lower[index].abs()));
        let upper_active = upper[index].is_finite() && upper_distance <= pmin(lower_distance, relative_step * pmax(1.0, upper[index].abs()));
        if lower_active {
            values[index] = if relative_step == 0.0 {
                nextafter(lower[index], upper[index])
            } else {
                lower[index] + relative_step * pmax(1.0, lower[index].abs())
            };
        } else if upper_active {
            values[index] = if relative_step == 0.0 {
                nextafter(upper[index], lower[index])
            } else {
                upper[index] - relative_step * pmax(1.0, upper[index].abs())
            };
        }
        if values[index] < lower[index] || values[index] > upper[index] {
            values[index] = 0.5 * (lower[index] + upper[index]);
        }
    }
}

/// `_cholesky_inside`: the Newton step when the matrix is safely positive
/// definite AND the step lies inside the radius.
fn cholesky_inside(information: &[f64], gradient: &[f64], radius: f64, lower_factor: &mut [f64], intermediate: &mut [f64], result: &mut [f64]) -> Zr<bool> {
    let count = gradient.len();
    let mut maximum_diagonal = 0.0;
    for index in 0..count {
        maximum_diagonal = pmax(maximum_diagonal, information[index * count + index].abs());
    }
    let threshold = EPSILON * count as f64 * pmax(1.0, maximum_diagonal);
    for row in 0..count {
        for column in 0..=row {
            let mut value = information[row * count + column];
            for prior in 0..column {
                value -= lower_factor[row * count + prior] * lower_factor[column * count + prior];
            }
            if row == column {
                if !value.is_finite() || value <= threshold {
                    return Ok(false);
                }
                lower_factor[row * count + column] = value.sqrt();
            } else {
                lower_factor[row * count + column] = div(value, lower_factor[column * count + column])?;
            }
        }
        for column in (row + 1)..count {
            lower_factor[row * count + column] = 0.0;
        }
    }
    for row in 0..count {
        let mut value = -gradient[row];
        for prior in 0..row {
            value -= lower_factor[row * count + prior] * intermediate[prior];
        }
        intermediate[row] = div(value, lower_factor[row * count + row])?;
    }
    let mut norm_squared = 0.0;
    for reverse in 0..count {
        let row = count - 1 - reverse;
        let mut value = intermediate[row];
        for following in (row + 1)..count {
            value -= lower_factor[following * count + row] * result[following];
        }
        result[row] = div(value, lower_factor[row * count + row])?;
        norm_squared += result[row] * result[row];
    }
    Ok(pmax(norm_squared, 0.0).sqrt() <= radius)
}

/// `_trust_eigh`: the trust-region step from the eigen-decomposition.
fn trust_eigh(observations: usize, eigenvalues: &[f64], eigenvectors: &[f64], projected_gradient: &[f64], radius: f64, mut alpha: f64, output: &mut [f64]) -> Zr<f64> {
    let count = eigenvalues.len();
    let ev = |row: usize, column: usize| eigenvectors[row * count + column];
    let mut full_rank = false;
    if observations >= count && eigenvalues[count - 1] > 0.0 {
        let threshold = EPSILON * observations as f64 * eigenvalues[count - 1].sqrt();
        full_rank = pmax(eigenvalues[0], 0.0).sqrt() > threshold;
    }
    if full_rank {
        let mut norm_squared = 0.0;
        for row in 0..count {
            let mut value = 0.0;
            for column in 0..count {
                value += ev(row, column) * div(projected_gradient[column], eigenvalues[column])?;
            }
            output[row] = -value;
            norm_squared += value * value;
        }
        if norm_squared.sqrt() <= radius {
            return Ok(0.0);
        }
    }
    let mut gradient_squared = 0.0;
    for &value in projected_gradient {
        gradient_squared += value * value;
    }
    let mut gradient_norm = gradient_squared.sqrt();
    if !gradient_norm.is_finite() || gradient_norm == 0.0 {
        gradient_norm = 0.0;
        for &value in projected_gradient {
            gradient_norm = gradient_norm.hypot(value);
        }
        if gradient_norm == 0.0 {
            for value in output.iter_mut() {
                *value = 0.0;
            }
            return Ok(0.0);
        }
    }
    let mut alpha_upper = div(gradient_norm, radius)?;
    let mut alpha_lower = 0.0;
    if full_rank {
        let mut norm_squared = 0.0;
        let mut derivative_numerator = 0.0;
        for index in 0..count {
            let value = div(projected_gradient[index], eigenvalues[index])?;
            norm_squared += value * value;
            let e = eigenvalues[index];
            derivative_numerator += div(projected_gradient[index] * projected_gradient[index], e * (e * e))?;
        }
        let norm = norm_squared.sqrt();
        let phi = norm - radius;
        let derivative = div(-derivative_numerator, norm)?;
        alpha_lower = div(-phi, derivative)?;
    }
    if (!full_rank && alpha == 0.0) || alpha < alpha_lower || alpha > alpha_upper {
        alpha = pmax(0.001 * alpha_upper, pmax(alpha_lower * alpha_upper, 0.0).sqrt());
    }
    for _iteration in 0..10 {
        if (!full_rank && alpha == 0.0) || alpha < alpha_lower || alpha > alpha_upper {
            alpha = pmax(0.001 * alpha_upper, pmax(alpha_lower * alpha_upper, 0.0).sqrt());
        }
        let mut norm_squared = 0.0;
        let mut derivative_numerator = 0.0;
        let mut stable_derivative = false;
        for index in 0..count {
            let denominator = eigenvalues[index] + alpha;
            let value = div(projected_gradient[index], denominator)?;
            norm_squared += value * value;
            let cube = denominator * denominator * denominator;
            if cube == 0.0 {
                stable_derivative = true;
            } else {
                derivative_numerator += div(projected_gradient[index] * projected_gradient[index], cube)?;
            }
        }
        let mut norm = norm_squared.sqrt();
        let mut phi = norm - radius;
        let ratio;
        if stable_derivative || !derivative_numerator.is_finite() {
            norm = 0.0;
            for index in 0..count {
                norm = norm.hypot(div(projected_gradient[index], eigenvalues[index] + alpha)?);
            }
            let mut derivative = 0.0;
            for index in 0..count {
                let denominator = eigenvalues[index] + alpha;
                let relative = div(div(projected_gradient[index], denominator)?, norm)?;
                derivative += div(relative * relative, denominator)?;
            }
            phi = norm - radius;
            ratio = div(-(1.0 - div(radius, norm)?), derivative)?;
        } else {
            let derivative = div(-derivative_numerator, norm)?;
            ratio = div(phi, derivative)?;
        }
        if phi < 0.0 {
            alpha_upper = alpha;
        }
        alpha_lower = pmax(alpha_lower, alpha - ratio);
        alpha -= div((phi + radius) * ratio, radius)?;
        if phi.abs() < 0.01 * radius {
            break;
        }
    }
    let mut norm_squared = 0.0;
    for row in 0..count {
        let mut value = 0.0;
        for column in 0..count {
            value += ev(row, column) * div(projected_gradient[column], eigenvalues[column] + alpha)?;
        }
        output[row] = -value;
        norm_squared += value * value;
    }
    if norm_squared > 0.0 {
        let scale = div(radius, norm_squared.sqrt())?;
        for value in output.iter_mut() {
            *value *= scale;
        }
    }
    Ok(alpha)
}

#[inline]
fn update_radius(radius: f64, actual: f64, predicted: f64, step_norm: f64) -> Zr<(f64, f64)> {
    let ratio = if predicted > 0.0 {
        div(actual, predicted)?
    } else if predicted == 0.0 && actual == 0.0 {
        1.0
    } else {
        0.0
    };
    if ratio < 0.25 {
        return Ok((0.25 * step_norm, ratio));
    }
    if ratio > 0.75 && step_norm > TRUST_BOUNDARY_FRACTION * radius {
        return Ok((2.0 * radius, ratio));
    }
    Ok((radius, ratio))
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn termination(actual: f64, cost: f64, step: &[f64], values: &[f64], ratio: f64, ftol: f64, xtol: f64, trust_limited: bool) -> i32 {
    let function_done = !trust_limited && actual < ftol * cost && ratio > 0.25;
    let mut parameter_done = true;
    for index in 0..values.len() {
        if step[index].abs() >= xtol * (xtol + values[index].abs()) {
            parameter_done = false;
            break;
        }
    }
    if function_done && parameter_done {
        return STATUS_FTOL_XTOL;
    }
    if function_done {
        return STATUS_FTOL;
    }
    if parameter_done {
        return STATUS_XTOL;
    }
    STATUS_MAX_NFEV
}

/// The eigen-decomposition `_solve_seed` takes when the Cholesky step is
/// refused: eigenvalues ascending, eigenvectors row-major (columns are the
/// vectors), small negative eigenvalues flushed to zero, and the gradient
/// projected onto them.
fn trust_eigen(trust_information: &[f64], scaled_gradient: &[f64], eigenvalues: &mut Vec<f64>, eigenvectors: &mut Vec<f64>, projected_gradient: &mut [f64]) -> Result<(), SolveError> {
    let count = scaled_gradient.len();
    let (values, vectors) = linalg::eigh_lower(count, trust_information)?;
    *eigenvalues = values;
    *eigenvectors = vectors;
    let top = eigenvalues[count - 1];
    for index in 0..count {
        if eigenvalues[index] < 0.0 && eigenvalues[index] > -1.0e-12 * pmax(1.0, top) {
            eigenvalues[index] = 0.0;
        }
    }
    for column in 0..count {
        let mut value = 0.0;
        for row in 0..count {
            value += eigenvectors[row * count + column] * scaled_gradient[row];
        }
        projected_gradient[column] = value;
    }
    Ok(())
}

/// The outcome of one seed's solve.
#[derive(Debug, Clone)]
pub struct SeedOutcome {
    pub values: Vec<f64>,
    pub cost: f64,
    pub raw_rss: f64,
    pub status: i32,
    pub iterations: i32,
    pub nfev: i32,
    pub njev: i32,
}

/// `_solve_seed`.
#[allow(clippy::too_many_arguments)]
pub fn solve_seed(
    objective: &dyn CellObjective,
    full_template: &[f64],
    free_indices: &[usize],
    seed: &[f64],
    lower: &[f64],
    upper: &[f64],
    max_nfev: i64,
    ftol: f64,
    xtol: f64,
    gtol: f64,
) -> Result<SeedOutcome, SolveError> {
    let free_count = seed.len();
    let mut values = seed.to_vec();
    for index in 0..free_count {
        if !values[index].is_finite() {
            return Ok(SeedOutcome { values, cost: f64::INFINITY, raw_rss: f64::INFINITY, status: STATUS_INVALID, iterations: 0, nfev: 0, njev: 0 });
        }
        let low_inside = if lower[index].is_finite() { nextafter(lower[index], upper[index]) } else { lower[index] };
        let high_inside = if upper[index].is_finite() { nextafter(upper[index], lower[index]) } else { upper[index] };
        values[index] = pmin(pmax(values[index], low_inside), high_inside);
    }
    make_feasible(&mut values, lower, upper, 1.0e-10);

    let selected = objective.selected();
    if selected <= free_count {
        return Ok(SeedOutcome { values, cost: f64::INFINITY, raw_rss: f64::INFINITY, status: STATUS_INVALID, iterations: 0, nfev: 0, njev: 0 });
    }

    let n = free_count;
    let mut full_parameters = full_template.to_vec();
    let mut gradient = vec![0.0; n];
    let mut information = vec![0.0; n * n];
    let mut jacobian_row = vec![0.0; n];
    let mut scale_inverse = vec![0.0; n];
    let mut coleman_li = vec![0.0; n];
    let mut coleman_li_derivative = vec![0.0; n];
    let mut scaling = vec![0.0; n];
    let mut scaled_gradient = vec![0.0; n];
    let mut trust_information = vec![0.0; n * n];
    let mut diagonal = vec![0.0; n];
    let mut unconstrained = vec![0.0; n];
    let mut step = vec![0.0; n];
    let mut scaled_step = vec![0.0; n];
    let mut trial = vec![0.0; n];
    let mut projected_gradient = vec![0.0; n];
    let mut lower_factor = vec![0.0; n * n];
    let mut intermediate = vec![0.0; n];
    let mut eigenvalues: Vec<f64> = vec![0.0; n];
    let mut eigenvectors: Vec<f64> = vec![0.0; n * n];

    let mut initialized = false;
    let mut radius = 1.0;
    let mut alpha = 0.0;
    let mut nfev: i32 = 0;
    let mut njev: i32 = 0;
    let mut iteration: i32 = 0;
    let status;
    let mut cost;
    let mut raw_rss;

    loop {
        for index in 0..n {
            full_parameters[free_indices[index]] = values[index];
        }
        let (c, r, finite) = objective.evaluate(&full_parameters, free_indices, &mut gradient, &mut information, &mut jacobian_row, true)?;
        cost = c;
        raw_rss = r;
        if iteration == 0 {
            nfev = 1;
        }
        njev += 1;
        if !finite || !cost.is_finite() || !raw_rss.is_finite() {
            status = STATUS_INVALID;
            break;
        }

        let mut gradient_norm = 0.0;
        for index in 0..n {
            let norm = pmax(information[index * n + index], 0.0).sqrt();
            scale_inverse[index] = if initialized { pmax(norm, scale_inverse[index]) } else if norm == 0.0 { 1.0 } else { norm };
            let value = gradient[index];
            if value < 0.0 && upper[index].is_finite() {
                coleman_li[index] = upper[index] - values[index];
                coleman_li_derivative[index] = -1.0;
            } else if value > 0.0 && lower[index].is_finite() {
                coleman_li[index] = values[index] - lower[index];
                coleman_li_derivative[index] = 1.0;
            } else {
                coleman_li[index] = 1.0;
                coleman_li_derivative[index] = 0.0;
            }
            gradient_norm = pmax(gradient_norm, (value * coleman_li[index]).abs());
        }
        initialized = true;
        if gradient_norm < gtol {
            status = STATUS_GTOL;
            break;
        }
        if nfev as i64 >= max_nfev {
            status = STATUS_MAX_NFEV;
            break;
        }
        if iteration == STALL_ITERATIONS {
            if let Some(finished) = objective.finish_stalled(&full_parameters, free_indices, &values, cost, raw_rss, lower, upper, max_nfev - nfev as i64, (ftol, xtol, gtol)) {
                let finished = finished?;
                if finished.status != STATUS_INVALID && finished.cost <= cost {
                    return Ok(SeedOutcome { iterations: iteration + finished.iterations, nfev: nfev + finished.nfev, njev: njev + finished.njev, ..finished });
                }
            }
        }

        for index in 0..n {
            if coleman_li_derivative[index] != 0.0 {
                coleman_li[index] *= scale_inverse[index];
            }
            let inverse = div(1.0, scale_inverse[index])?;
            scaling[index] = coleman_li[index].sqrt() * inverse;
            scaled_gradient[index] = scaling[index] * gradient[index];
            diagonal[index] = pmax(gradient[index] * coleman_li_derivative[index] * inverse, 0.0);
        }
        for row in 0..n {
            for column in 0..n {
                trust_information[row * n + column] = scaling[row] * information[row * n + column] * scaling[column];
            }
            trust_information[row * n + row] += diagonal[row];
        }
        if iteration == 0 {
            let mut norm_squared = 0.0;
            for index in 0..n {
                let value = div(values[index] * scale_inverse[index], coleman_li[index].sqrt())?;
                norm_squared += value * value;
            }
            radius = norm_squared.sqrt();
            if radius == 0.0 {
                radius = 1.0;
            }
        }

        let cholesky_ready = cholesky_inside(&trust_information, &scaled_gradient, radius, &mut lower_factor, &mut intermediate, &mut unconstrained)?;
        let mut eigen_ready = false;
        if !cholesky_ready {
            trust_eigen(&trust_information, &scaled_gradient, &mut eigenvalues, &mut eigenvectors, &mut projected_gradient)?;
            eigen_ready = true;
        } else {
            alpha = 0.0;
        }

        let theta = pmax(0.995, 1.0 - gradient_norm);
        let mut accepted = false;
        let mut terminal = STATUS_MAX_NFEV;
        let mut first = true;
        while (nfev as i64) < max_nfev {
            if !(first && cholesky_ready) {
                if !eigen_ready {
                    trust_eigen(&trust_information, &scaled_gradient, &mut eigenvalues, &mut eigenvectors, &mut projected_gradient)?;
                    eigen_ready = true;
                }
                alpha = trust_eigh(selected, &eigenvalues, &eigenvectors, &projected_gradient, radius, alpha, &mut unconstrained)?;
            }
            first = false;
            let predicted = select_reflective_step(&values, &trust_information, &scaled_gradient, &mut unconstrained, &scaling, radius, lower, upper, theta, &mut step, &mut scaled_step)?;
            for index in 0..n {
                trial[index] = values[index] + step[index];
            }
            make_feasible(&mut trial, lower, upper, 0.0);
            for index in 0..n {
                full_parameters[free_indices[index]] = trial[index];
            }
            let (trial_cost, trial_rss, trial_finite) = objective.evaluate(&full_parameters, free_indices, &mut gradient, &mut information, &mut jacobian_row, false)?;
            nfev += 1;
            let scaled_norm = vector_norm(&scaled_step);
            if !trial_finite || !trial_cost.is_finite() {
                radius = 0.25 * scaled_norm;
                continue;
            }
            let actual = cost - trial_cost;
            let old_radius = radius;
            let (new_radius, ratio) = update_radius(old_radius, actual, predicted, scaled_norm)?;
            radius = new_radius;
            terminal = termination(actual, cost, &step, &values, ratio, ftol, xtol, scaled_norm > TRUST_BOUNDARY_FRACTION * old_radius);
            if radius != 0.0 {
                alpha *= div(old_radius, radius)?;
            }
            if actual > 0.0 {
                values.copy_from_slice(&trial);
                cost = trial_cost;
                raw_rss = trial_rss;
                accepted = true;
                break;
            }
            if terminal != STATUS_MAX_NFEV {
                break;
            }
        }

        iteration += 1;
        if terminal != STATUS_MAX_NFEV {
            status = terminal;
            break;
        }
        if nfev as i64 >= max_nfev {
            status = STATUS_MAX_NFEV;
            break;
        }
        if !accepted {
            status = STATUS_INVALID;
            break;
        }
    }
    Ok(SeedOutcome { values, cost, raw_rss, status, iterations: iteration, nfev, njev })
}

/// What `_solve_cell` returns.
#[derive(Debug, Clone)]
pub struct CellSolution {
    pub parameters: Vec<f64>,
    pub cost: f64,
    pub raw_rss: f64,
    pub status: i32,
    pub iterations: i32,
    pub nfev: i32,
    pub njev: i32,
    pub winner_seed: i32,
}

/// Below this many selected observations a cell's seeds solve one after
/// another on the calling thread.
const PARALLEL_SEED_POINTS: usize = 4096;

/// `_solve_cell`: every prepared seed competes; a raising seed is skipped.
#[allow(clippy::too_many_arguments)]
pub fn solve_cell(
    objective: &dyn CellObjective,
    full_seeds: &[Vec<f64>],
    full_lower: &[f64],
    full_upper: &[f64],
    free_indices: &[usize],
    max_nfev: i64,
    ftol: f64,
    xtol: f64,
    gtol: f64,
    warm_first: bool,
    prepare_status: i32,
) -> CellSolution {
    let full_count = full_lower.len();
    let free_count = free_indices.len();
    let mut best_full = vec![f64::NAN; full_count];
    if prepare_status < STATUS_MAX_NFEV || full_seeds.is_empty() {
        return CellSolution { parameters: best_full, cost: f64::INFINITY, raw_rss: f64::INFINITY, status: prepare_status, iterations: 0, nfev: 0, njev: 0, winner_seed: -1 };
    }
    let full_template = full_lower.to_vec();
    let free_lower: Vec<f64> = free_indices.iter().map(|&p| full_lower[p]).collect();
    let free_upper: Vec<f64> = free_indices.iter().map(|&p| full_upper[p]).collect();
    let solve = |seed: &Vec<f64>| {
        let free_seed: Vec<f64> = free_indices.iter().map(|&p| seed[p]).collect();
        solve_seed(objective, &full_template, free_indices, &free_seed, &free_lower, &free_upper, max_nfev, ftol, xtol, gtol)
    };
    // Every seed's solve is its own: on a large objective they run at once
    // and are weighed below in seed order, as one after another would be
    // (a warm first seed that settles it is solved, and weighed, first).
    let mut outcomes: Vec<Option<Result<SeedOutcome, SolveError>>> = (0..full_seeds.len()).map(|_| None).collect();
    if full_seeds.len() > 1 && objective.selected() >= PARALLEL_SEED_POINTS {
        let rest = usize::from(warm_first);
        let mut settled = false;
        if warm_first {
            let first = solve(&full_seeds[0]);
            settled = matches!(&first, Ok(o) if o.status > STATUS_MAX_NFEV && o.raw_rss <= RSS_TIE_RELATIVE);
            outcomes[0] = Some(first);
        }
        if !settled {
            let solved: Vec<Result<SeedOutcome, SolveError>> = full_seeds[rest..].par_iter().map(solve).collect();
            for (slot, outcome) in outcomes[rest..].iter_mut().zip(solved) {
                *slot = Some(outcome);
            }
        }
    }

    let mut best_cost = f64::INFINITY;
    let mut best_rss = f64::INFINITY;
    let mut best_status = STATUS_NO_CANDIDATE;
    let mut best_iterations = 0;
    let mut best_nfev = 0;
    let mut best_njev = 0;
    let mut best_seed: i32 = -1;
    let mut have = false;
    let mut best_success = false;

    for (seed_index, seed) in full_seeds.iter().enumerate() {
        let outcome = match outcomes[seed_index].take().unwrap_or_else(|| solve(seed)) {
            Ok(outcome) => outcome,
            Err(_) => {
                best_status = if !have { STATUS_INVALID } else { best_status };
                continue;
            }
        };
        let successful = outcome.status > STATUS_MAX_NFEV;
        let cost = outcome.cost;
        let choose = if !have {
            true
        } else if successful && !best_success {
            true
        } else {
            successful == best_success
                && if best_cost.is_finite() { cost < best_cost - RSS_TIE_RELATIVE * pmax(1.0, best_cost.abs()) } else { cost.is_finite() }
        };
        if choose {
            best_full.copy_from_slice(&full_template);
            for index in 0..free_count {
                best_full[free_indices[index]] = outcome.values[index];
            }
            best_cost = cost;
            best_rss = outcome.raw_rss;
            best_status = outcome.status;
            best_iterations = outcome.iterations;
            best_nfev = outcome.nfev;
            best_njev = outcome.njev;
            best_seed = seed_index as i32;
            best_success = successful;
            have = true;
        }
        if seed_index == 0 && warm_first && successful && outcome.raw_rss <= RSS_TIE_RELATIVE {
            break;
        }
    }
    if !have {
        return CellSolution { parameters: best_full, cost: f64::INFINITY, raw_rss: f64::INFINITY, status: best_status, iterations: 0, nfev: 0, njev: 0, winner_seed: -1 };
    }
    CellSolution { parameters: best_full, cost: best_cost, raw_rss: best_rss, status: best_status, iterations: best_iterations, nfev: best_nfev, njev: best_njev, winner_seed: best_seed }
}

// --------------------------------------------------------------------------
// Preparation of one cell's seeds and bounds
// --------------------------------------------------------------------------

fn seed_finite(seed: &[f64]) -> bool { seed.iter().all(|v| v.is_finite()) }

fn clip_full_seed(seed: &mut [f64], lower: &[f64], upper: &[f64], free_mask: &[bool]) -> bool {
    for index in 0..seed.len() {
        if !seed[index].is_finite() {
            return false;
        }
        if free_mask[index] {
            let low_inside = if lower[index].is_finite() { nextafter(lower[index], upper[index]) } else { lower[index] };
            let high_inside = if upper[index].is_finite() { nextafter(upper[index], lower[index]) } else { upper[index] };
            seed[index] = pmin(pmax(seed[index], low_inside), high_inside);
        } else {
            seed[index] = lower[index];
        }
    }
    seed_finite(seed)
}

fn append_seed(source: &[f64], seeds: &mut Vec<Vec<f64>>, capacity: usize, lower: &[f64], upper: &[f64], free_mask: &[bool]) {
    let mut candidate = source.to_vec();
    if !clip_full_seed(&mut candidate, lower, upper, free_mask) {
        return;
    }
    for prior in seeds.iter() {
        if candidate.iter().zip(prior).all(|(a, b)| a == b) {
            return;
        }
    }
    if seeds.len() >= capacity {
        return;
    }
    seeds.push(candidate);
}

/// What the preparation of one cell produced.
#[derive(Debug, Clone)]
pub struct PreparedCell {
    pub seeds: Vec<Vec<f64>>,
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
    pub status: i32,
}

/// `_prepare_one`, with the model's prepare callback abstracted.
///
/// `prepare(seeds_block, rows, lower, upper)` fills cold seeds into a
/// `rows x parameters` NaN block and tightens the bounds; it returns the
/// count written.
#[allow(clippy::too_many_arguments)]
pub fn prepare_one(
    prepare: &mut dyn FnMut(&mut [f64], usize, &mut [f64], &mut [f64]) -> Zr<i64>,
    base_lower: &[f64],
    base_upper: &[f64],
    requested_lower: &[f64],
    requested_upper: &[f64],
    requested_mask: &[bool],
    warm_seed: Option<&[f64]>,
    authored_seeds: Option<&[Vec<f64>]>,
    free_mask: &[bool],
    max_cold_candidates: usize,
    seed_capacity: usize,
) -> Zr<PreparedCell> {
    let parameter_count = base_lower.len();
    let mut lower = base_lower.to_vec();
    let mut upper = base_upper.to_vec();
    let use_authored = authored_seeds.is_some();
    let mut needs_cold = !use_authored;
    if let Some(authored) = authored_seeds {
        if authored.iter().any(|seed| !seed_finite(seed)) {
            needs_cold = true;
        }
    }
    let rows = if needs_cold { max_cold_candidates } else { 0 };
    let mut cold = vec![f64::NAN; rows * parameter_count];
    let cold_count = prepare(&mut cold, rows, &mut lower, &mut upper)?;
    let invalid = |lower: Vec<f64>, upper: Vec<f64>| PreparedCell { seeds: Vec::new(), lower, upper, status: STATUS_INVALID };
    if cold_count < 0 || cold_count as usize > rows {
        return Ok(invalid(lower, upper));
    }
    let cold_count = cold_count as usize;
    for index in 0..parameter_count {
        if requested_mask[index] {
            lower[index] = requested_lower[index];
            upper[index] = requested_upper[index];
        }
        if lower[index].is_nan() || upper[index].is_nan() {
            return Ok(invalid(lower, upper));
        }
        if free_mask[index] {
            if !(lower[index] < upper[index]) {
                return Ok(invalid(lower, upper));
            }
        } else if lower[index] != upper[index] {
            return Ok(invalid(lower, upper));
        }
    }
    let mut seeds: Vec<Vec<f64>> = Vec::new();
    if let Some(warm) = warm_seed {
        append_seed(warm, &mut seeds, seed_capacity, &lower, &upper, free_mask);
        if seeds.is_empty() {
            return Ok(invalid(lower, upper));
        }
    }
    let cold_row = |index: usize| &cold[index * parameter_count..(index + 1) * parameter_count];
    if let Some(authored) = authored_seeds {
        for seed in authored {
            if seed_finite(seed) {
                append_seed(seed, &mut seeds, seed_capacity, &lower, &upper, free_mask);
                continue;
            }
            for cold_index in 0..cold_count {
                let mut merged = cold_row(cold_index).to_vec();
                for parameter in 0..parameter_count {
                    if seed[parameter].is_finite() {
                        merged[parameter] = seed[parameter];
                    }
                }
                append_seed(&merged, &mut seeds, seed_capacity, &lower, &upper, free_mask);
            }
        }
    } else {
        for cold_index in 0..cold_count {
            append_seed(cold_row(cold_index), &mut seeds, seed_capacity, &lower, &upper, free_mask);
        }
    }
    if seeds.is_empty() {
        return Ok(PreparedCell { seeds, lower, upper, status: STATUS_NO_CANDIDATE });
    }
    Ok(PreparedCell { seeds, lower, upper, status: STATUS_MAX_NFEV })
}

// --------------------------------------------------------------------------
// Finalisation: fitted values, residuals, covariance
// --------------------------------------------------------------------------

/// What `_finalize_one` returns.
#[derive(Debug, Clone)]
pub struct Finalized {
    pub fitted: Vec<f64>,
    pub residuals: Vec<f64>,
    /// Row-major `parameters x parameters`.
    pub covariance: Vec<f64>,
    pub errors: Vec<f64>,
    pub reduced: f64,
    pub covariance_valid: bool,
}

/// What can escape `_finalize_one` (no `try` there): it fails the bucket.
#[derive(Debug, Clone, PartialEq)]
pub enum FinalizeError {
    Kernel(KernelError),
    LinAlg(LinAlgError),
}

impl std::fmt::Display for FinalizeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FinalizeError::Kernel(error) => write!(f, "{error}"),
            FinalizeError::LinAlg(error) => write!(f, "{error}"),
        }
    }
}

impl From<KernelError> for FinalizeError {
    fn from(error: KernelError) -> Self { FinalizeError::Kernel(error) }
}

impl From<LinAlgError> for FinalizeError {
    fn from(error: LinAlgError) -> Self { FinalizeError::LinAlg(error) }
}

/// `_finalize_one` for a point-layout cell.  `value_jacobian(selected,
/// with_jacobian)` evaluates the model at the selected (valid) points:
/// values and a row-major `selected x parameters` Jacobian.
#[allow(clippy::too_many_arguments)]
pub fn finalize_one(
    value_jacobian: &dyn Fn(bool) -> Zr<(Vec<f64>, Vec<f64>)>,
    observations: &[f64],
    valid: &[bool],
    parameters: &[f64],
    free_indices: &[usize],
    weights: &[f64],
    use_weights: bool,
    poisson: bool,
    loss_code: i64,
) -> Result<Finalized, FinalizeError> {
    let point_count = observations.len();
    let parameter_count = parameters.len();
    let free_count = free_indices.len();
    let mut residuals = vec![f64::NAN; point_count];
    let mut covariance = vec![0.0; parameter_count * parameter_count];
    let mut errors = vec![0.0; parameter_count];
    let all_nan = |residuals: Vec<f64>| Finalized {
        fitted: vec![f64::NAN; point_count],
        residuals,
        covariance: vec![f64::NAN; parameter_count * parameter_count],
        errors: vec![f64::NAN; parameter_count],
        reduced: f64::INFINITY,
        covariance_valid: false,
    };
    if !seed_finite(parameters) {
        return Ok(all_nan(residuals));
    }
    let selected_count = valid.iter().take(point_count).filter(|&&v| v).count();
    let (selected_fitted, full_jacobian) = value_jacobian(free_count != 0)?;
    if selected_fitted.len() != selected_count {
        return Ok(all_nan(residuals));
    }
    let mut fitted = if selected_count == point_count { selected_fitted.clone() } else { vec![f64::NAN; point_count] };
    let mut selected = 0usize;
    let mut raw_rss = 0.0;
    let mut finite = true;
    let mut scaled_jacobian = vec![0.0; selected_count * free_count];
    for point in 0..point_count {
        if !valid[point] {
            continue;
        }
        fitted[point] = selected_fitted[selected];
        residuals[point] = observations[point] - fitted[point];
        if !fitted[point].is_finite() || !residuals[point].is_finite() {
            finite = false;
        }
        selected += 1;
        let weight = if use_weights { weights[point] } else { 1.0 };
        let (raw, squared, _cost, _gf, _if, point_finite) = point_terms(fitted[point], observations[point], poisson, weight, use_weights, loss_code)?;
        raw_rss += squared;
        let (_rho0, rho1, rho2) = rho(raw * raw, loss_code)?;
        let robust_scale = pmax(rho1 + 2.0 * rho2 * raw * raw, EPSILON).sqrt();
        let residual_scale = if poisson {
            let expected = pmax(fitted[point], COUNT_FLOOR);
            let absolute = raw.abs();
            if absolute > COUNT_FLOOR {
                div((expected - observations[point]).abs(), expected * pmax(absolute, COUNT_FLOOR))?
            } else {
                div(1.0, expected.sqrt())?
            }
        } else if use_weights {
            weights[point]
        } else {
            1.0
        };
        if !point_finite {
            finite = false;
        }
        let row = selected - 1;
        for free in 0..free_count {
            let value = full_jacobian[row * parameter_count + free_indices[free]] * residual_scale * robust_scale;
            scaled_jacobian[row * free_count + free] = value;
            if !value.is_finite() {
                finite = false;
            }
        }
    }
    let reduced = div(raw_rss, (selected as i64 - free_count as i64).max(1) as f64)?;
    if free_count == 0 {
        return Ok(Finalized { fitted, residuals, covariance, errors, reduced, covariance_valid: finite });
    }
    let mut covariance_valid = finite && selected > free_count && reduced.is_finite() && reduced >= 0.0;
    let mut free_covariance = vec![0.0; free_count * free_count];
    if covariance_valid {
        let triangular = linalg::qr_r(selected, free_count, &scaled_jacobian)?;
        let (singular_values, right) = linalg::svd_small(free_count, &triangular)?;
        covariance_valid = singular_values.len() == free_count && singular_values[0].is_finite() && singular_values[free_count - 1].is_finite();
        let tolerance = if covariance_valid { EPSILON * selected.max(free_count) as f64 * singular_values[0] } else { f64::INFINITY };
        covariance_valid = covariance_valid && singular_values[free_count - 1] > tolerance;
        if covariance_valid {
            for row in 0..free_count {
                for column in 0..free_count {
                    let mut value = 0.0;
                    for singular in 0..free_count {
                        value += div(right[singular * free_count + row] * right[singular * free_count + column], singular_values[singular] * singular_values[singular])?;
                    }
                    free_covariance[row * free_count + column] = value * reduced;
                }
            }
            for row in 0..free_count {
                for column in 0..row {
                    let value = 0.5 * (free_covariance[row * free_count + column] + free_covariance[column * free_count + row]);
                    free_covariance[row * free_count + column] = value;
                    free_covariance[column * free_count + row] = value;
                }
                let diagonal = free_covariance[row * free_count + row];
                if !diagonal.is_finite() || diagonal < 0.0 {
                    covariance_valid = false;
                }
            }
        }
    }
    let mut free_mask = vec![false; parameter_count];
    for &free in free_indices {
        free_mask[free] = true;
    }
    if covariance_valid {
        for row in 0..free_count {
            let full_row = free_indices[row];
            errors[full_row] = pmax(free_covariance[row * free_count + row], 0.0).sqrt();
            for column in 0..free_count {
                covariance[full_row * parameter_count + free_indices[column]] = free_covariance[row * free_count + column];
            }
        }
    } else {
        for row in 0..parameter_count {
            if free_mask[row] {
                errors[row] = f64::NAN;
            }
            for column in 0..parameter_count {
                if free_mask[row] && free_mask[column] {
                    covariance[row * parameter_count + column] = f64::NAN;
                }
            }
        }
    }
    Ok(Finalized { fitted, residuals, covariance, errors, reduced, covariance_valid })
}
