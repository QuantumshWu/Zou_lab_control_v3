//! Regular-image fits (`_fit_radial`): the separable Gaussian image models
//! seeded on a bounded proxy, refined on the full frame through the same
//! trust-region solver with the closed-form linear objective, with the
//! axis-Gram information and the deferred per-pixel arrays.

use std::sync::Arc;

use rayon::prelude::*;

use super::blas;
use super::engine::{Check, CellOutcome, FitRequest};
use super::models::{point_terms, Kernel, LOSS_LINEAR};
use super::num::{div, fsum, numpy_sum, pmax, Zr, EPSILON, TINY};
pub use super::plane::Samples;
use super::plane::{centred_context, plane_products, residual_rss};
use super::prepare::Seeds;
use super::result::{DeferredArrays, FitError, FitOptions, FitResult, ObservationArrays, SolveDiagnostics};
use super::scratch::Scratch;
use super::shared::{authored_seed, fixed_parameter_partition, solver_bounds, too_few_points, warm_seed};
use super::spec::{FitModelSpec, REGULAR_IMAGE_RADIAL, REGULAR_IMAGE_SEPARABLE};
use super::trf::{self, CellObjective, STATUS_MAX_NFEV};

pub const STRIPE_ROWS: usize = 64;
pub const SAMPLE_LIMIT: usize = 129;
pub const IMAGE_FTOL: f64 = 1e-10;
pub const IMAGE_GTOL: f64 = 1e-8;
pub const PROXY_TOL: f64 = 1e-5;

/// `RegularImageFitInput`: a Cartesian image without coordinate grids,
/// its samples in the camera's own storage (u8/u16/i16/f32/f64).
#[derive(Debug, Clone)]
pub struct RegularImageFitInput {
    pub x: Arc<[f64]>,
    pub y: Arc<[f64]>,
    /// Row-major `(len(y), len(x))`.
    pub observations: Samples,
    pub valid_mask: Option<Arc<[bool]>>,
    pub selected_indices: Option<Arc<[i64]>>,
}

impl RegularImageFitInput {
    pub fn new(x: Vec<f64>, y: Vec<f64>, observations: impl Into<Samples>, valid_mask: Option<Vec<bool>>, selected_indices: Option<Vec<i64>>) -> Result<Self, FitError> {
        let observations = observations.into();
        if x.is_empty() || y.is_empty() {
            return Err(FitError::value("regular image coordinates cannot be empty"));
        }
        if x.iter().chain(y.iter()).any(|v| !v.is_finite()) {
            return Err(FitError::value("regular image coordinates must be finite"));
        }
        for (name, values) in [("x", &x), ("y", &y)] {
            if values.len() > 1 {
                let increasing = values.windows(2).all(|w| w[1] - w[0] > 0.0);
                let decreasing = values.windows(2).all(|w| w[1] - w[0] < 0.0);
                if !(increasing || decreasing) {
                    return Err(FitError::value(format!("regular image {name} coordinates must be strictly monotonic")));
                }
            }
        }
        if observations.len() != x.len() * y.len() {
            return Err(FitError::value("regular image observations must have shape (len(y), len(x))"));
        }
        if let Some(mask) = &valid_mask {
            if mask.len() != observations.len() {
                return Err(FitError::value("regular image valid_mask must match observations"));
            }
        }
        if let Some(indices) = &selected_indices {
            if indices.len() != observations.len() {
                return Err(FitError::value("regular image selected_indices must identify every image cell"));
            }
        }
        Ok(RegularImageFitInput {
            x: x.into(),
            y: y.into(),
            observations,
            valid_mask: valid_mask.map(Into::into),
            selected_indices: selected_indices.map(Into::into),
        })
    }
    pub fn width(&self) -> usize { self.x.len() }
    pub fn height(&self) -> usize { self.y.len() }
}

/// `_SeparableKernel`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SeparableKernel {
    radial: bool,
    parameter_count: usize,
    x_radius_index: usize,
    y_radius_index: usize,
}

const RADIAL: SeparableKernel = SeparableKernel { radial: true, parameter_count: 5, x_radius_index: 2, y_radius_index: 2 };
const ANISOTROPIC: SeparableKernel = SeparableKernel { radial: false, parameter_count: 6, x_radius_index: 2, y_radius_index: 3 };

fn kernel_for(model: &FitModelSpec) -> Result<SeparableKernel, FitError> {
    for (capability, kernel) in [(REGULAR_IMAGE_RADIAL, RADIAL), (REGULAR_IMAGE_SEPARABLE, ANISOTROPIC)] {
        if model.capabilities.contains(capability) {
            if model.parameters.len() != kernel.parameter_count {
                return Err(FitError::value(format!("capability {} requires {} parameters", super::num::str_repr(capability), kernel.parameter_count)));
            }
            return Ok(kernel);
        }
    }
    Err(FitError::value("this model does not declare a regular-image capability"))
}

impl SeparableKernel {
    fn point_kernel(self) -> Kernel { if self.radial { Kernel::Radial } else { Kernel::Anisotropic } }
}

// --------------------------------------------------------------------------
// numba kernels
// --------------------------------------------------------------------------

/// `_compiled_axis_terms`.
#[inline(always)]
fn axis_terms(coordinate: f64, center: f64, radius: f64) -> Zr<(f64, f64, f64)> {
    let delta = coordinate - center;
    let basis = div(-(delta * delta), radius * radius)?.exp();
    Ok((basis, basis * div(2.0 * delta * delta, radius * radius * radius)?, basis * div(2.0 * delta, radius * radius)?))
}

/// `_compiled_regular_linear_objective`.
#[allow(clippy::too_many_arguments)]
fn linear_objective(
    plane: Option<(&Plane, &Memo)>,
    parameters: &[f64],
    free_indices: &[usize],
    gradient: &mut [f64],
    information: &mut [f64],
    x_vectors: &[Vec<f64>; 3],
    y_vectors: &[Vec<f64>; 3],
    derivatives: bool,
    radial: bool,
    context: Option<&[f64; 4]>,
    observed_total: &dyn Fn() -> f64,
    information_only: bool,
) -> (f64, f64, bool) {
    let height = y_vectors[0].len();
    let width = x_vectors[0].len();
    let amplitude = parameters[0];
    let offset = parameters[1] - context.map_or(0.0, |c| c[0]);
    let mut projected = [vec![0.0; height], vec![0.0; height], vec![0.0; height]];
    let mut raw_rss = 0.0;
    let points = plane.map_or(0, |(p, _)| p.samples.len());
    if let (false, Some((plane, memo))) = (information_only, plane) {
        let key: Vec<u64> = parameters.iter().map(|v| v.to_bits()).collect();
        let (known_product, known_rss) = memo.recall(&key);
        if derivatives || context.is_some() {
            // One read of the plane for every vector the call needs; the
            // trial a derivative call repeats already gave `image @ x0`.
            let count = if derivatives { 3 } else { 1 };
            let first = usize::from(known_product.is_some());
            if let Some(product) = known_product {
                projected[0] = product;
            }
            let vectors: Vec<&[f64]> = x_vectors[first..count].iter().map(Vec::as_slice).collect();
            if !vectors.is_empty() {
                for (row, product) in projected[first..].iter_mut().zip(plane_products(plane.samples, plane.reference, width, height, &vectors)) {
                    *row = product;
                }
            }
        }
        if let Some(context) = context {
            let x_sum = super::num::sequential_sum(&x_vectors[0]);
            let y_sum = super::num::sequential_sum(&y_vectors[0]);
            let terms = [
                amplitude * amplitude * blas::ddot(&x_vectors[0], &x_vectors[0]) * blas::ddot(&y_vectors[0], &y_vectors[0]),
                2.0 * amplitude * offset * x_sum * y_sum,
                offset * offset * points as f64,
                -2.0 * amplitude * blas::ddot(&y_vectors[0], &projected[0]),
                -2.0 * offset * context[1],
                context[2],
            ];
            let mut correction = 0.0;
            let mut magnitude = 0.0;
            for &term in &terms {
                let combined = raw_rss + term;
                correction += if f64::abs(raw_rss) >= term.abs() { (raw_rss - combined) + term } else { (term - combined) + raw_rss };
                raw_rss = combined;
                magnitude += term.abs();
            }
            raw_rss += correction;
            let operations = (points + width + height + 16) as f64;
            let mut roundoff = operations * EPSILON;
            roundoff /= 1.0 - roundoff;
            if raw_rss <= roundoff * magnitude {
                raw_rss = residual_rss(plane.samples, plane.reference, amplitude, offset, &x_vectors[0], &y_vectors[0]);
            }
        } else {
            raw_rss = known_rss.unwrap_or_else(|| residual_rss(plane.samples, plane.reference, amplitude, offset, &x_vectors[0], &y_vectors[0]));
        }
        memo.remember(key, (derivatives || context.is_some()).then(|| projected[0].clone()), context.is_none().then_some(raw_rss));
    }
    if !derivatives {
        return (0.5 * raw_rss, raw_rss, raw_rss.is_finite());
    }
    let observed_sum = match context {
        Some(c) => c[1],
        None => observed_total(),
    };
    let (x_ones, y_ones) = (vec![1.0; width], vec![1.0; height]);
    let x_full: [&[f64]; 4] = [&x_vectors[0], &x_vectors[1], &x_vectors[2], &x_ones];
    let y_full: [&[f64]; 4] = [&y_vectors[0], &y_vectors[1], &y_vectors[2], &y_ones];
    let mut x_sums = [0.0; 4];
    let mut y_sums = [0.0; 4];
    let mut x_inner = [[0.0; 4]; 4];
    let mut y_inner = [[0.0; 4]; 4];
    let mut data_inner = [[0.0; 4]; 4];
    for left in 0..4 {
        let mut x_sum = 0.0;
        for &v in x_full[left] {
            x_sum += v;
        }
        x_sums[left] = x_sum;
        let mut y_sum = 0.0;
        for &v in y_full[left] {
            y_sum += v;
        }
        y_sums[left] = y_sum;
        for right in 0..4 {
            let mut x_dot = 0.0;
            for column in 0..width {
                x_dot += x_full[left][column] * x_full[right][column];
            }
            x_inner[left][right] = x_dot;
            let mut y_dot = 0.0;
            let mut data_dot = 0.0;
            for row in 0..height {
                y_dot += y_full[left][row] * y_full[right][row];
                if right < 3 {
                    data_dot += y_full[left][row] * projected[right][row];
                }
            }
            y_inner[left][right] = y_dot;
            data_inner[left][right] = if right == 3 { if left == 3 { observed_sum } else { 0.0 } } else { data_dot };
        }
    }
    let parameter_count = parameters.len();
    let mut term_count = vec![1usize; parameter_count];
    let mut term_y = vec![[0usize; 2]; parameter_count];
    let mut term_x = vec![[0usize; 2]; parameter_count];
    let mut term_scale = vec![[1.0_f64; 2]; parameter_count];
    term_y[1][0] = 3;
    term_x[1][0] = 3;
    if radial {
        term_count[2] = 2;
        term_y[2] = [0, 1];
        term_x[2] = [1, 0];
        term_scale[2] = [amplitude, amplitude];
        term_y[3][0] = 0;
        term_x[3][0] = 2;
        term_scale[3][0] = amplitude;
        term_y[4][0] = 2;
        term_x[4][0] = 0;
        term_scale[4][0] = amplitude;
    } else {
        term_y[2][0] = 0;
        term_x[2][0] = 1;
        term_scale[2][0] = amplitude;
        term_y[3][0] = 1;
        term_x[3][0] = 0;
        term_scale[3][0] = amplitude;
        term_y[4][0] = 0;
        term_x[4][0] = 2;
        term_scale[4][0] = amplitude;
        term_y[5][0] = 2;
        term_x[5][0] = 0;
        term_scale[5][0] = amplitude;
    }
    let mut full_gradient = vec![0.0; parameter_count];
    let mut full_information = vec![0.0; parameter_count * parameter_count];
    for parameter in 0..parameter_count {
        let mut value = 0.0;
        for term in 0..term_count[parameter] {
            let yi = term_y[parameter][term];
            let xi = term_x[parameter][term];
            let scale = term_scale[parameter][term];
            let model_dot = amplitude * y_inner[0][yi] * x_inner[0][xi] + offset * y_sums[yi] * x_sums[xi];
            value += scale * (model_dot - data_inner[yi][xi]);
        }
        full_gradient[parameter] = value;
        for other in 0..=parameter {
            let mut value = 0.0;
            for left in 0..term_count[parameter] {
                for right in 0..term_count[other] {
                    value += term_scale[parameter][left] * term_scale[other][right] * y_inner[term_y[parameter][left]][term_y[other][right]] * x_inner[term_x[parameter][left]][term_x[other][right]];
                }
            }
            full_information[parameter * parameter_count + other] = value;
            full_information[other * parameter_count + parameter] = value;
        }
    }
    let n = free_indices.len();
    for row in 0..n {
        gradient[row] = full_gradient[free_indices[row]];
        for column in 0..n {
            information[row * n + column] = full_information[free_indices[row] * parameter_count + free_indices[column]];
        }
    }
    let finite = raw_rss.is_finite() && gradient[..n].iter().all(|v| v.is_finite()) && information[..n * n].iter().all(|v| v.is_finite());
    (0.5 * raw_rss, raw_rss, finite)
}

/// The last evaluation's plane-wide results, by its parameters (bit for
/// bit): a solve's derivative call repeats the trial it accepted, whose
/// `image @ x0` (and, on the proxy, residual RSS) are the same numbers.
#[derive(Default)]
struct Memo(std::sync::Mutex<Option<(Vec<u64>, Option<Vec<f64>>, Option<f64>)>>);

impl Memo {
    fn recall(&self, key: &[u64]) -> (Option<Vec<f64>>, Option<f64>) {
        match &*self.0.lock().expect("memo") {
            Some((known, product, rss)) if known == key => (product.clone(), *rss),
            _ => (None, None),
        }
    }
    fn remember(&self, key: Vec<u64>, product: Option<Vec<f64>>, rss: Option<f64>) { *self.0.lock().expect("memo") = Some((key, product, rss)); }
}

/// A plane as an objective reads it: each sample less `reference` (the
/// refinement's centring; zero on the proxy).
struct Plane<'a> {
    samples: &'a Samples,
    reference: f64,
}

/// The grid-layout objective of one cell (`_compiled_regular_image_objective`).
struct GridObjective<'a> {
    x: &'a [f64],
    y: &'a [f64],
    plane: Plane<'a>,
    /// `None`: every sample valid.
    valid: Option<&'a [bool]>,
    context: Option<[f64; 4]>,
    loss_code: i64,
    radial: bool,
    selected: usize,
    observed_sum: std::sync::OnceLock<f64>,
    memo: Memo,
    /// The bounded proxy stage (it only seeds the refinement).
    proxy: bool,
}

impl CellObjective for GridObjective<'_> {
    fn evaluate(&self, parameters: &[f64], free_indices: &[usize], gradient: &mut [f64], information: &mut [f64], jacobian_row: &mut [f64], derivatives: bool) -> Zr<(f64, f64, bool)> {
        let point_count = self.plane.samples.len();
        let width = self.x.len();
        let height = self.y.len();
        if point_count == 0 || width * height != point_count {
            return Ok((f64::INFINITY, f64::INFINITY, false));
        }
        let Some((x_vectors, y_vectors)) = self.axis_vectors(parameters)? else {
            return Ok((f64::INFINITY, f64::INFINITY, false));
        };
        if self.closed_form() {
            let observed_total = || self.observed_total();
            return Ok(linear_objective(Some((&self.plane, &self.memo)), parameters, free_indices, gradient, information, &x_vectors, &y_vectors, derivatives, self.radial, self.context.as_ref(), &observed_total, false));
        }
        let amplitude = parameters[0];
        let offset = parameters[1] - self.context.map_or(0.0, |c| c[0]);
        let pass = GeneralPass { reference: self.plane.reference, valid: self.valid, x: &x_vectors, y: &y_vectors, amplitude, offset, radial: self.radial, loss_code: self.loss_code, free: free_indices, derivatives };
        let identity = free_indices.len() == parameters.len() && free_indices.iter().enumerate().all(|(i, &f)| i == f);
        let plain = self.loss_code == LOSS_LINEAR;
        // Compiled for the storage, the parameter count when every
        // parameter is free, and plain least squares: the same operations
        // in the same order in every instance.
        macro_rules! run {
            ($s:expr) => {
                match (parameters.len(), identity, plain) {
                    (5, true, true) => pass.run::<5, true, _>($s, gradient, information, jacobian_row),
                    (5, true, false) => pass.run::<5, false, _>($s, gradient, information, jacobian_row),
                    (6, true, true) => pass.run::<6, true, _>($s, gradient, information, jacobian_row),
                    (6, true, false) => pass.run::<6, false, _>($s, gradient, information, jacobian_row),
                    (_, _, true) => pass.run::<0, true, _>($s, gradient, information, jacobian_row),
                    (_, _, false) => pass.run::<0, false, _>($s, gradient, information, jacobian_row),
                }
            };
        }
        match self.plane.samples {
            Samples::U8(s) => run!(s),
            Samples::U16(s) => run!(s),
            Samples::I16(s) => run!(s),
            Samples::F32(s) => run!(s),
            Samples::F64(s) => run!(s),
        }
    }
    fn selected(&self) -> usize { self.selected }
    /// A stalled solve on a frame of many small spots (a single Gaussian
    /// drawn onto one atom of an array). On the bounded proxy the spot is
    /// narrower than the proxy's sample spacing, so the solve creeps toward
    /// the radius floor for thousands of iterations without moving its
    /// centre: the proxy only seeds the refinement, which takes the point it
    /// has reached. The full-resolution refinement crawls along the valley
    /// where amplitude and radius trade off (often to the evaluation budget):
    /// it goes on with the amplitude and offset solved out ([`Projected`]),
    /// which has no such valley and reaches the same minimum.
    fn finish_stalled(&self, params: &[f64], free: &[usize], values: &[f64], cost: f64, raw_rss: f64, lower: &[f64], upper: &[f64],
                      budget: i64, (ftol, xtol, gtol): (f64, f64, f64)) -> Option<Result<trf::SeedOutcome, trf::SolveError>> {
        if self.proxy {
            return Some(Ok(trf::SeedOutcome { values: values.to_vec(), cost, raw_rss, status: trf::STATUS_FTOL, iterations: 0, nfev: 0, njev: 0 }));
        }
        let count = params.len();
        if !self.closed_form() || free.len() != count || free.iter().enumerate().any(|(index, &parameter)| index != parameter) {
            return None;
        }
        let projected = Projected { grid: self, lower: [lower[0], lower[1]], upper: [upper[0], upper[1]], product: Default::default() };
        let shape: Vec<usize> = (2..count).collect();
        let outcome = match trf::solve_seed(&projected, params, &shape, &values[2..], &lower[2..], &upper[2..], budget, ftol, xtol, gtol) {
            Ok(outcome) => outcome,
            Err(error) => return Some(Err(error)),
        };
        if outcome.status == trf::STATUS_INVALID {
            return None;
        }
        let mut answer = params.to_vec();
        answer[2..].copy_from_slice(&outcome.values);
        let Ok(Some(linear)) = projected.linear(&answer) else { return None };
        answer[0] = linear.amplitude;
        answer[1] = linear.offset;
        Some(Ok(trf::SeedOutcome { values: answer, ..outcome }))
    }
}

impl GridObjective<'_> {
    /// The axis vectors (basis, radius term, centre term) of `parameters`'s
    /// shape; `None` for a radius that is not positive.
    fn axis_vectors(&self, parameters: &[f64]) -> Zr<Option<([Vec<f64>; 3], [Vec<f64>; 3])>> {
        let (width, height) = (self.x.len(), self.y.len());
        let radius_x = parameters[2];
        let radius_y = if self.radial { parameters[2] } else { parameters[3] };
        let center_x = parameters[parameters.len() - 2];
        let center_y = parameters[parameters.len() - 1];
        if radius_x <= 0.0 || radius_y <= 0.0 {
            return Ok(None);
        }
        let mut x_vectors = [vec![0.0; width], vec![0.0; width], vec![0.0; width]];
        let mut y_vectors = [vec![0.0; height], vec![0.0; height], vec![0.0; height]];
        for column in 0..width {
            let (a, b, c) = axis_terms(self.x[column], center_x, radius_x)?;
            x_vectors[0][column] = a;
            x_vectors[1][column] = b;
            x_vectors[2][column] = c;
        }
        for row in 0..height {
            let (a, b, c) = axis_terms(self.y[row], center_y, radius_y)?;
            y_vectors[0][row] = a;
            y_vectors[1][row] = b;
            y_vectors[2][row] = c;
        }
        Ok(Some((x_vectors, y_vectors)))
    }

    /// Every sample valid under plain least squares: the closed-form objective.
    fn closed_form(&self) -> bool {
        let all_valid = match &self.context {
            Some(c) => c[3] == self.plane.samples.len() as f64,
            None => self.valid.is_none_or(|valid| valid.iter().all(|&v| v)),
        };
        all_valid && self.loss_code == LOSS_LINEAR
    }

    /// The observed (reference-less) sum, the same every call: summed once.
    fn observed_total(&self) -> f64 {
        *self.observed_sum.get_or_init(|| {
            let mut total = 0.0;
            for p in 0..self.plane.samples.len() {
                total += self.plane.samples.get(p) - self.plane.reference;
            }
            total
        })
    }
}

/// A shape's exact least-squares amplitude and offset, with what gave them.
struct Linear {
    amplitude: f64,
    offset: f64,
    x_vectors: [Vec<f64>; 3],
    y_vectors: [Vec<f64>; 3],
    product: Vec<f64>,
}

/// The closed-form image objective with its linear parameters solved out
/// (variable projection): for each shape (radii, centre) the amplitude and
/// offset are the least-squares ones, the cost is the full objective's
/// there, and the solver walks the shape alone on the Schur complement of the
/// information on the linear pair (Kaufman's step). Its minima are the full
/// objective's.
struct Projected<'a, 'b> {
    grid: &'a GridObjective<'b>,
    lower: [f64; 2],
    upper: [f64; 2],
    /// The last shape's `image @ x basis` (a derivative call repeats the trial it accepted).
    product: std::sync::Mutex<Option<(Vec<u64>, Vec<f64>)>>,
}

impl Projected<'_, '_> {
    /// `None` where the normal equations are singular or the answer is out of bounds.
    fn linear(&self, parameters: &[f64]) -> Zr<Option<Linear>> {
        let grid = self.grid;
        let Some((x_vectors, y_vectors)) = grid.axis_vectors(parameters)? else { return Ok(None) };
        let key: Vec<u64> = parameters[2..].iter().map(|value| value.to_bits()).collect();
        let known = match &*self.product.lock().expect("product") {
            Some((shape, product)) if *shape == key => Some(product.clone()),
            _ => None,
        };
        let product = match known {
            Some(product) => product,
            None => {
                let product = plane_products(grid.plane.samples, grid.plane.reference, grid.x.len(), grid.y.len(), &[&x_vectors[0]]).remove(0);
                *self.product.lock().expect("product") = Some((key, product.clone()));
                product
            }
        };
        let points = grid.plane.samples.len() as f64;
        let (x_sum, y_sum) = (super::num::sequential_sum(&x_vectors[0]), super::num::sequential_sum(&y_vectors[0]));
        let basis_square = blas::ddot(&x_vectors[0], &x_vectors[0]) * blas::ddot(&y_vectors[0], &y_vectors[0]);
        let basis_sum = x_sum * y_sum;
        let data_basis = blas::ddot(&y_vectors[0], &product);
        let data_sum = grid.context.map_or_else(|| grid.observed_total(), |c| c[1]);
        let determinant = basis_square * points - basis_sum * basis_sum;
        if !(determinant > 0.0 && determinant.is_finite()) {
            return Ok(None);
        }
        let amplitude = (points * data_basis - basis_sum * data_sum) / determinant;
        let offset = (basis_square * data_sum - basis_sum * data_basis) / determinant + grid.context.map_or(0.0, |c| c[0]);
        let inside = |value: f64, index: usize| value.is_finite() && value >= self.lower[index] && value <= self.upper[index];
        if !inside(amplitude, 0) || !inside(offset, 1) {
            return Ok(None);
        }
        Ok(Some(Linear { amplitude, offset, x_vectors, y_vectors, product }))
    }
}

impl CellObjective for Projected<'_, '_> {
    fn evaluate(&self, parameters: &[f64], free_indices: &[usize], gradient: &mut [f64], information: &mut [f64], _jacobian_row: &mut [f64], derivatives: bool) -> Zr<(f64, f64, bool)> {
        let Some(linear) = self.linear(parameters)? else { return Ok((f64::INFINITY, f64::INFINITY, false)) };
        let grid = self.grid;
        let mut full = parameters.to_vec();
        full[0] = linear.amplitude;
        full[1] = linear.offset;
        let count = full.len();
        grid.memo.remember(full.iter().map(|value| value.to_bits()).collect(), Some(linear.product), None);
        let every: Vec<usize> = (0..count).collect();
        let mut full_gradient = vec![0.0; count];
        let mut full_information = vec![0.0; count * count];
        let observed_total = || grid.observed_total();
        let (cost, raw_rss, finite) = linear_objective(Some((&grid.plane, &grid.memo)), &full, &every, &mut full_gradient, &mut full_information,
                                                       &linear.x_vectors, &linear.y_vectors, derivatives, grid.radial, grid.context.as_ref(), &observed_total, false);
        if !derivatives || !finite {
            return Ok((cost, raw_rss, finite));
        }
        // The shape's information less what the linear pair explains.
        let entry = |row: usize, column: usize| full_information[row * count + column];
        let (aa, ab, bb) = (entry(0, 0), entry(0, 1), entry(1, 1));
        let determinant = aa * bb - ab * ab;
        if !(determinant > 0.0 && determinant.is_finite()) {
            return Ok((cost, raw_rss, false));
        }
        let n = free_indices.len();
        for (row, &p) in free_indices.iter().enumerate() {
            gradient[row] = full_gradient[p];
            for (column, &q) in free_indices.iter().enumerate() {
                let explained = (entry(p, 0) * (bb * entry(0, q) - ab * entry(1, q)) + entry(p, 1) * (aa * entry(1, q) - ab * entry(0, q))) / determinant;
                information[row * n + column] = entry(p, q) - explained;
            }
        }
        let finite = gradient[..n].iter().chain(&information[..n * n]).all(|value| value.is_finite());
        Ok((cost, raw_rss, finite))
    }
    fn selected(&self) -> usize { self.grid.selected }
}

/// The point-by-point objective (`_compiled_regular_image_objective` off the
/// linear path: masks, non-finite samples, robust losses): every valid
/// pixel in row-major order into one cost, one RSS and, with derivatives,
/// the gradient and lower information, held in registers.
struct GeneralPass<'a> {
    reference: f64,
    valid: Option<&'a [bool]>,
    x: &'a [Vec<f64>; 3],
    y: &'a [Vec<f64>; 3],
    amplitude: f64,
    offset: f64,
    radial: bool,
    loss_code: i64,
    free: &'a [usize],
    derivatives: bool,
}

impl GeneralPass<'_> {
    /// `N > 0`: all `N` parameters free in order; `PLAIN`: linear loss.
    fn run<const N: usize, const PLAIN: bool, T: super::plane::Sample>(&self, s: &[T], gradient: &mut [f64], information: &mut [f64], jacobian_row: &mut [f64]) -> Zr<(f64, f64, bool)> {
        let (width, height) = (self.x[0].len(), self.y[0].len());
        let nfree = if N > 0 { N } else { self.free.len() };
        let (amplitude, offset) = (self.amplitude, self.offset);
        let mut g = [0.0_f64; 6];
        let mut info = [0.0_f64; 36];
        let mut projected = [0.0_f64; 6];
        let mut full = [0.0_f64; 6];
        let mut cost = 0.0;
        let mut raw_rss = 0.0;
        for row in 0..height {
            let (y_basis, y_radius, y_center) = (self.y[0][row], self.y[1][row], self.y[2][row]);
            for column in 0..width {
                let point = row * width + column;
                if self.valid.is_some_and(|valid| !valid[point]) {
                    continue;
                }
                let x_basis = self.x[0][column];
                let phi = y_basis * x_basis;
                let predicted = amplitude * phi + offset;
                let observed = s[point].value() - self.reference;
                let (_raw, squared, local_cost, gradient_factor, information_factor, finite) =
                    if PLAIN { point_terms(predicted, observed, false, 1.0, false, LOSS_LINEAR)? } else { point_terms(predicted, observed, false, 1.0, false, self.loss_code)? };
                if !finite || !predicted.is_finite() {
                    return Ok((f64::INFINITY, f64::INFINITY, false));
                }
                cost += local_cost;
                raw_rss += squared;
                if !self.derivatives {
                    continue;
                }
                full[0] = phi;
                full[1] = 1.0;
                if self.radial {
                    full[2] = amplitude * (y_basis * self.x[1][column] + y_radius * x_basis);
                    full[3] = amplitude * y_basis * self.x[2][column];
                    full[4] = amplitude * y_center * x_basis;
                } else {
                    full[2] = amplitude * y_basis * self.x[1][column];
                    full[3] = amplitude * y_radius * x_basis;
                    full[4] = amplitude * y_basis * self.x[2][column];
                    full[5] = amplitude * y_center * x_basis;
                }
                if N > 0 {
                    for value in &full[..N] {
                        if !value.is_finite() {
                            return Ok((f64::INFINITY, f64::INFINITY, false));
                        }
                    }
                    projected = full;
                } else {
                    for (k, &f) in self.free.iter().enumerate() {
                        let value = full[f];
                        if !value.is_finite() {
                            return Ok((f64::INFINITY, f64::INFINITY, false));
                        }
                        projected[k] = value;
                    }
                }
                for r in 0..nfree {
                    let value = projected[r];
                    g[r] += value * gradient_factor;
                    for c in 0..=r {
                        info[r * 6 + c] += value * projected[c] * information_factor;
                    }
                }
            }
        }
        if self.derivatives {
            jacobian_row[..nfree].copy_from_slice(&projected[..nfree]);
            for r in 0..nfree {
                gradient[r] = g[r];
                for c in 0..=r {
                    information[r * nfree + c] = info[r * 6 + c];
                    information[c * nfree + r] = info[r * 6 + c];
                }
            }
        }
        Ok((cost, raw_rss, cost.is_finite() && raw_rss.is_finite()))
    }
}

/// `_compiled_regular_information_batch` for one cell: the axis-Gram
/// information and the direct RSS (centred parameters and plane).
fn information_and_rss(x: &[f64], y: &[f64], parameters: &[f64], radial: bool, plane: &Plane) -> (Vec<f64>, f64) {
    let count = parameters.len();
    let nan = (vec![f64::NAN; count * count], f64::NAN);
    let radius_x = parameters[2];
    let radius_y = if radial { parameters[2] } else { parameters[3] };
    if parameters.iter().any(|v| !v.is_finite()) || radius_x * radius_x == 0.0 || radius_y * radius_y == 0.0 {
        return nan;
    }
    let mut x_vectors = [vec![0.0; x.len()], vec![0.0; x.len()], vec![0.0; x.len()]];
    let mut y_vectors = [vec![0.0; y.len()], vec![0.0; y.len()], vec![0.0; y.len()]];
    for (column, &xv) in x.iter().enumerate() {
        let Ok((a, b, c)) = axis_terms(xv, parameters[count - 2], radius_x) else { return nan };
        x_vectors[0][column] = a;
        x_vectors[1][column] = b;
        x_vectors[2][column] = c;
    }
    for (row, &yv) in y.iter().enumerate() {
        let Ok((a, b, c)) = axis_terms(yv, parameters[count - 1], radius_y) else { return nan };
        y_vectors[0][row] = a;
        y_vectors[1][row] = b;
        y_vectors[2][row] = c;
    }
    let free: Vec<usize> = (0..count).collect();
    let mut gradient = vec![0.0; count];
    let mut information = vec![0.0; count * count];
    linear_objective(None, parameters, &free, &mut gradient, &mut information, &x_vectors, &y_vectors, true, radial, None, &|| 0.0, true);
    // Stripe by stripe, each from a total of zero: one residual pass.
    (information, residual_rss(plane.samples, plane.reference, parameters[0], parameters[1], &x_vectors[0], &y_vectors[0]))
}

// --------------------------------------------------------------------------
// Python-side preparation
// --------------------------------------------------------------------------

struct CroppedImage {
    data: RegularImageFitInput,
    origin: Option<(usize, usize, usize)>,
}

fn row_has(mask: &[bool], width: usize, row: usize) -> bool { mask[row * width..(row + 1) * width].iter().any(|&v| v) }

/// `np.any(mask, axis=0)`: the rows OR-ed together in one row-major pass.
fn columns_have(mask: &[bool], width: usize) -> Vec<bool> {
    let mut any = vec![false; width];
    for row in mask.chunks_exact(width.max(1)) {
        for (flag, &value) in any.iter_mut().zip(row) {
            *flag |= value;
        }
    }
    any
}

/// `_crop_to_valid_bounds`.
fn crop_to_valid_bounds(data: &RegularImageFitInput, check: &Check) -> Result<CroppedImage, FitError> {
    let Some(mask) = &data.valid_mask else { return Ok(CroppedImage { data: data.clone(), origin: None }) };
    check.check()?;
    let (height, width) = (data.height(), data.width());
    let rows: Vec<usize> = (0..height).filter(|&r| row_has(mask, width, r)).collect();
    let columns: Vec<usize> = columns_have(mask, width).iter().enumerate().filter(|(_, any)| **any).map(|(c, _)| c).collect();
    if rows.is_empty() || columns.is_empty() {
        return Err(FitError::value("regular image has no finite valid observations"));
    }
    let (y0, y1) = (rows[0], rows[rows.len() - 1] + 1);
    let (x0, x1) = (columns[0], columns[columns.len() - 1] + 1);
    let cropped_mask: Vec<bool> = (y0..y1).flat_map(|r| (x0..x1).map(move |c| (r, c))).map(|(r, c)| mask[r * width + c]).collect();
    let all_valid = cropped_mask.iter().all(|&v| v);
    if (y1 - y0, x1 - x0) == (height, width) {
        if !all_valid {
            return Ok(CroppedImage { data: data.clone(), origin: None });
        }
        let mut full = data.clone();
        full.valid_mask = None;
        return Ok(CroppedImage { data: full, origin: None });
    }
    let observations = data.observations.gather((y0..y1).flat_map(|r| (x0..x1).map(move |c| r * width + c)));
    let selected: Option<Vec<i64>> = data.selected_indices.as_ref().map(|s| (y0..y1).flat_map(|r| (x0..x1).map(move |c| (r, c))).map(|(r, c)| s[r * width + c]).collect());
    let origin = if selected.is_some() { None } else { Some((y0, x0, width)) };
    let cropped = RegularImageFitInput::new(data.x[x0..x1].to_vec(), data.y[y0..y1].to_vec(), observations, if all_valid { None } else { Some(cropped_mask) }, selected)?;
    Ok(CroppedImage { data: cropped, origin })
}

/// `np.linspace(0, n - 1, count, dtype=np.int64)` positions.
fn linspace_indices(n: usize, count: usize) -> Vec<usize> {
    if count == 0 {
        return Vec::new();
    }
    if count == 1 {
        return vec![0];
    }
    let div = (count - 1) as f64;
    let stop = (n - 1) as f64;
    let step = stop / div;
    let mut out: Vec<usize> = (0..count)
        .map(|i| {
            let value = if step == 0.0 { (i as f64 / div) * stop } else { i as f64 * step };
            value.floor() as usize
        })
        .collect();
    out[count - 1] = n - 1;
    out
}

fn bounded_indices(valid: &[bool], limit: usize) -> Vec<usize> {
    let indices: Vec<usize> = (0..valid.len()).filter(|&i| valid[i]).collect();
    let count = indices.len().min(limit);
    if indices.is_empty() {
        return Vec::new();
    }
    linspace_indices(indices.len(), count).into_iter().map(|p| indices[p]).collect()
}

/// `_regular_image_subsample`: the bounded proxy.
fn subsample(data: &RegularImageFitInput, limit: usize, check: &Check) -> Result<RegularImageFitInput, FitError> {
    let (height, width) = (data.height(), data.width());
    if height <= limit && width <= limit {
        return Ok(data.clone());
    }
    check.check()?;
    let mut mask: Option<Scratch<bool>> = data.valid_mask.as_ref().map(|m| Scratch::copied(m));
    if data.observations.is_float() {
        let finite = Scratch::collected(data.observations.len(), (0..data.observations.len()).map(|p| data.observations.get(p).is_finite()));
        mask = Some(match mask {
            None => finite,
            Some(m) => Scratch::collected(m.len(), m.iter().zip(finite.iter()).map(|(a, b)| *a && *b)),
        });
    }
    let (y_index, x_index) = match &mask {
        None => (bounded_indices(&vec![true; height], limit), bounded_indices(&vec![true; width], limit)),
        Some(m) => (
            bounded_indices(&(0..height).map(|r| row_has(m, width, r)).collect::<Vec<bool>>(), limit),
            bounded_indices(&columns_have(m, width), limit),
        ),
    };
    if y_index.is_empty() || x_index.is_empty() {
        return Err(FitError::value("regular image has no finite valid observations"));
    }
    let observations: Vec<f64> = y_index.iter().flat_map(|&r| x_index.iter().map(move |&c| (r, c))).map(|(r, c)| data.observations.get(r * width + c)).collect();
    let mut valid: Vec<bool> = observations.iter().map(|v| v.is_finite()).collect();
    if let Some(m) = &data.valid_mask {
        for (k, (r, c)) in y_index.iter().flat_map(|&r| x_index.iter().map(move |&c| (r, c))).enumerate() {
            valid[k] = valid[k] && m[r * width + c];
        }
    }
    let all = valid.iter().all(|&v| v);
    RegularImageFitInput::new(x_index.iter().map(|&c| data.x[c]).collect(), y_index.iter().map(|&r| data.y[r]).collect(), observations, if all { None } else { Some(valid) }, None)
}

/// `_axis_terms` (NumPy): basis, radius term, centre term.
fn numpy_axis_terms(coordinates: &[f64], center: f64, radius: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let r2 = super::num::pow(radius, 2.0);
    let r3 = super::num::pow(radius, 3.0);
    let mut basis = Vec::with_capacity(coordinates.len());
    let mut radius_term = Vec::with_capacity(coordinates.len());
    let mut center_term = Vec::with_capacity(coordinates.len());
    for &c in coordinates {
        let delta = c - center;
        let b = (-(delta * delta) / r2).exp();
        basis.push(b);
        radius_term.push(b * (2.0 * (delta * delta) / r3));
        center_term.push(b * (2.0 * delta / r2));
    }
    (basis, radius_term, center_term)
}

/// `_regular_image_result_arrays`: the deferred per-pixel arrays.
fn result_arrays(kernel: SeparableKernel, data: &RegularImageFitInput, parameters: &[f64], count: usize, origin: Option<(usize, usize, usize)>) -> (Vec<f64>, Vec<f64>, Vec<i64>) {
    let (amplitude, offset) = (parameters[0], parameters[1]);
    let p = parameters.len();
    let x_basis = numpy_axis_terms(&data.x, parameters[p - 2], parameters[kernel.x_radius_index]).0;
    let mut fitted = Vec::with_capacity(count);
    let mut residuals = Vec::with_capacity(count);
    let mut indices = Vec::with_capacity(count);
    let width = data.width();
    let finite_everywhere = !data.observations.is_float() || (0..data.observations.len()).all(|p| data.observations.get(p).is_finite());
    let mut start = 0;
    while start < data.height() {
        let stop = data.height().min(start + STRIPE_ROWS);
        let y_basis = numpy_axis_terms(&data.y[start..stop], parameters[p - 1], parameters[kernel.y_radius_index]).0;
        for (local, row) in (start..stop).enumerate() {
            for column in 0..width {
                let point = row * width + column;
                let masked_out = data.valid_mask.as_ref().is_some_and(|m| !m[point]) || (!finite_everywhere && !data.observations.get(point).is_finite());
                if masked_out {
                    continue;
                }
                let predicted = amplitude * y_basis[local] * x_basis[column] + offset;
                fitted.push(predicted);
                residuals.push(data.observations.get(point) - predicted);
                indices.push(match (&data.selected_indices, origin) {
                    (Some(s), _) => s[point],
                    (None, None) => point as i64,
                    (None, Some((row_origin, column_origin, full_width))) => ((row + row_origin) * full_width + column_origin + column) as i64,
                });
            }
        }
        start = stop;
    }
    (fitted, residuals, indices)
}

/// One stripe's share of the striped objective.
struct StripeShare {
    cost: f64,
    square_sum: f64,
    gradient: Vec<f64>,
    core: Vec<f64>,
    offset_column: Vec<f64>,
    offset_diagonal: f64,
    count: usize,
}

/// `_regular_image_striped_objective` (masked or robust final pass):
/// `(cost, rss, information, observation count)`.  Each stripe's share is
/// its own (stripes in parallel, each checked first as the oracle checks
/// it), added in stripe order.
fn striped_objective(kernel: SeparableKernel, data: &RegularImageFitInput, parameters: &[f64], loss: &str, check: &Check) -> Result<(f64, f64, Vec<f64>, usize), FitError> {
    let p = kernel.parameter_count;
    let m = p - 1;
    let (amplitude, offset) = (parameters[0], parameters[1]);
    let x_terms = numpy_axis_terms(&data.x, parameters[p - 2], parameters[kernel.x_radius_index]);
    let x_vectors = [x_terms.0, x_terms.1, x_terms.2];
    let width = data.width();
    let samples = &data.observations;
    let finite_everywhere = !samples.is_float() || (0..samples.len()).all(|p| samples.get(p).is_finite());
    let stripe = |start: usize| -> Result<StripeShare, FitError> {
        check.check()?;
        let stop = data.height().min(start + STRIPE_ROWS);
        let y_terms = numpy_axis_terms(&data.y[start..stop], parameters[p - 1], parameters[kernel.y_radius_index]);
        let y = [y_terms.0, y_terms.1, y_terms.2];
        let selected: Vec<usize> = (start * width..stop * width)
            .filter(|&point| !(data.valid_mask.as_ref().is_some_and(|mask| !mask[point]) || (!finite_everywhere && !samples.get(point).is_finite())))
            .collect();
        let rows = selected.len();
        let mut residual = Vec::with_capacity(rows);
        // The planes: y0 x0, then amplitude times each derivative plane
        // (its terms added in the oracle's order).
        let mut planes: Vec<Vec<f64>> = (0..m).map(|_| Vec::with_capacity(rows)).collect();
        for &point in &selected {
            let (local, column) = (point / width - start, point % width);
            let term = |yi: usize, xi: usize| y[yi][local] * x_vectors[xi][column];
            let radial_value = term(0, 0);
            residual.push((amplitude * radial_value + offset - samples.get(point)) / 1.0);
            planes[0].push(radial_value);
            if kernel.radial {
                planes[1].push(amplitude * (term(0, 1) + term(1, 0)));
                planes[2].push(amplitude * term(0, 2));
                planes[3].push(amplitude * term(2, 0));
            } else {
                planes[1].push(amplitude * term(0, 1));
                planes[2].push(amplitude * term(1, 0));
                planes[3].push(amplitude * term(0, 2));
                planes[4].push(amplitude * term(2, 0));
            }
        }
        let (rho, first, weight) = loss_terms(&residual, loss);
        let weighted: Vec<f64> = first.iter().zip(&residual).map(|(f, r)| f * r).collect();
        let mut gradient = vec![0.0; p];
        gradient[0] = blas::ddot(&planes[0], &weighted) / 1.0;
        gradient[1] = numpy_sum(&weighted) / 1.0;
        for (k, plane) in planes.iter().enumerate().skip(1) {
            gradient[1 + k] = blas::ddot(plane, &weighted) / 1.0;
        }
        let mut jacobian = vec![0.0; rows * m];
        let mut weighted_jacobian = vec![0.0; rows * m];
        for i in 0..rows {
            for a in 0..m {
                jacobian[i * m + a] = planes[a][i] / 1.0;
                weighted_jacobian[i * m + a] = weight[i] * jacobian[i * m + a];
            }
        }
        let core = blas::gemm_at_b(rows, m, m, &jacobian, &weighted_jacobian);
        let offset_column: Vec<f64> = (0..m)
            .map(|a| {
                let mut column_sum = 0.0;
                for i in 0..rows {
                    column_sum += weighted_jacobian[i * m + a];
                }
                column_sum / 1.0
            })
            .collect();
        Ok(StripeShare {
            cost: 0.5 * numpy_sum(&rho),
            square_sum: blas::ddot(&residual, &residual),
            gradient,
            core,
            offset_column,
            offset_diagonal: numpy_sum(&weight) / 1.0,
            count: rows,
        })
    };
    let starts: Vec<usize> = (0..data.height()).step_by(STRIPE_ROWS).collect();
    let shares: Vec<StripeShare> = starts.par_iter().map(|&start| stripe(start)).collect::<Result<_, _>>()?;
    let mut gradient = vec![0.0; p];
    let mut core = vec![0.0; m * m];
    let mut offset_column = vec![0.0; m];
    let mut offset_diagonal = 0.0;
    let mut observation_count = 0usize;
    for share in &shares {
        observation_count += share.count;
        for (total, value) in gradient.iter_mut().zip(&share.gradient) {
            *total += value;
        }
        for (total, value) in core.iter_mut().zip(&share.core) {
            *total += value;
        }
        for (total, value) in offset_column.iter_mut().zip(&share.offset_column) {
            *total += value;
        }
        offset_diagonal += share.offset_diagonal;
    }
    let cost = fsum(&shares.iter().map(|share| share.cost).collect::<Vec<f64>>());
    let rss = fsum(&shares.iter().map(|share| share.square_sum).collect::<Vec<f64>>());
    let dense: Vec<usize> = std::iter::once(0).chain(2..p).collect();
    let mut information = vec![0.0; p * p];
    for (a, &i) in dense.iter().enumerate() {
        for (b, &j) in dense.iter().enumerate() {
            information[i * p + j] = core[a * m + b];
        }
        information[p + i] = offset_column[a];
        information[i * p + 1] = offset_column[a];
    }
    information[p + 1] = offset_diagonal;
    let finite = cost.is_finite() && rss.is_finite() && gradient.iter().all(|v| v.is_finite()) && information.iter().all(|v| v.is_finite());
    if !finite {
        return Err(FitError::FloatingPoint("regular-image objective is non-finite".into()));
    }
    Ok((cost, rss, information, observation_count))
}

/// `_regular_image_loss_terms`.
fn loss_terms(residual: &[f64], loss: &str) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let squared: Vec<f64> = residual.iter().map(|r| r * r).collect();
    if loss == "linear" {
        return (squared.clone(), vec![1.0; residual.len()], vec![1.0; residual.len()]);
    }
    let mut rho = Vec::with_capacity(residual.len());
    let mut first = Vec::with_capacity(residual.len());
    let mut second = Vec::with_capacity(residual.len());
    for &s in &squared {
        match loss {
            "soft_l1" => {
                let root = (1.0 + s).sqrt();
                rho.push(2.0 * s / (root + 1.0));
                first.push(1.0 / root);
                second.push(-0.5 / super::num::pow(root, 3.0));
            }
            "huber" => {
                let outside = s > 1.0;
                let root = s.sqrt();
                let inverse = if root != 0.0 { 1.0 / root } else { 1.0 };
                rho.push(if outside { 2.0 * root - 1.0 } else { s });
                first.push(if outside { inverse } else { 1.0 });
                second.push(if outside { -0.5 * super::num::pow(inverse, 3.0) } else { 0.0 });
            }
            "cauchy" => {
                let f = 1.0 / (1.0 + s);
                rho.push(s.ln_1p());
                first.push(f);
                second.push(-(f * f));
            }
            _ => {
                let denominator = 1.0 + s * s;
                rho.push(s.atan());
                first.push(1.0 / denominator);
                second.push(-2.0 * s / (denominator * denominator));
            }
        }
    }
    let weight: Vec<f64> = (0..squared.len()).map(|i| pmax_np(first[i] + 2.0 * second[i] * squared[i], EPSILON)).collect();
    (rho, first, weight)
}

fn pmax_np(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a >= b {
        a
    } else {
        b
    }
}

/// `_covariance_from_information` for one matrix (NumPy `eigh`).
fn covariance_from_information(matrix: &[f64], count: usize, reduced: f64, observations: usize) -> (Vec<f64>, bool) {
    let invalid = (vec![f64::NAN; count * count], false);
    if count == 0 || observations <= count || matrix.iter().any(|v| !v.is_finite()) || !reduced.is_finite() || reduced < 0.0 {
        return invalid;
    }
    let norms: Vec<f64> = (0..count).map(|i| pmax_np(matrix[i * count + i], 0.0).sqrt()).collect();
    if !norms.iter().all(|&n| n > TINY) {
        return invalid;
    }
    let mut normalized = vec![0.0; count * count];
    let mut divisor = vec![0.0; count * count];
    for i in 0..count {
        for j in 0..count {
            divisor[i * count + j] = norms[i] * norms[j];
            normalized[i * count + j] = matrix[i * count + j] / divisor[i * count + j];
        }
    }
    let mut symmetric = vec![0.0; count * count];
    for i in 0..count {
        for j in 0..count {
            symmetric[i * count + j] = (normalized[i * count + j] + normalized[j * count + i]) / 2.0;
        }
    }
    let Ok((values, vectors)) = super::linalg::eigh_lower(count, &symmetric) else { return invalid };
    let top = values.iter().copied().fold(f64::NEG_INFINITY, |a, b| if b > a { b } else { a });
    let tolerance = EPSILON * count as f64 * top;
    if !(values[0] > tolerance) {
        return invalid;
    }
    let mut computed = vec![0.0; count * count];
    for i in 0..count {
        for j in 0..count {
            let mut value = 0.0;
            for k in 0..count {
                // NumPy matmul: OpenBLAS dgemm, one FMA accumulator per element.
                value = (vectors[i * count + k] / values[k]).mul_add(vectors[j * count + k], value);
            }
            computed[i * count + j] = value;
        }
    }
    for i in 0..count * count {
        computed[i] *= reduced / divisor[i];
    }
    let mut result = vec![0.0; count * count];
    for i in 0..count {
        for j in 0..count {
            result[i * count + j] = (computed[i * count + j] + computed[j * count + i]) / 2.0;
        }
    }
    if result.iter().any(|v| !v.is_finite()) || (0..count).any(|i| result[i * count + i] < 0.0) {
        return invalid;
    }
    (result, true)
}

// --------------------------------------------------------------------------
// The batch owner
// --------------------------------------------------------------------------

struct StageItem<'a> {
    data: &'a RegularImageFitInput,
}

struct StageSolution {
    parameters: Vec<f64>,
    raw_rss: f64,
    status: i32,
    success: bool,
    information: Option<Option<Vec<f64>>>,
    diagnostics: SolveDiagnostics,
}

#[allow(clippy::too_many_arguments)]
fn solve_stage(
    kernel: SeparableKernel,
    model: &FitModelSpec,
    items: &[(usize, StageItem)],
    seeds: Option<&[(usize, Vec<f64>)]>,
    refinement: bool,
    base_lower: &[f64],
    base_upper: &[f64],
    requested_lower: &[f64],
    requested_upper: &[f64],
    requested_mask: &[bool],
    authored_initial: Option<&[f64]>,
    warm: &[Option<Vec<f64>>],
    options: &FitOptions,
    check: &Check,
) -> Result<Vec<(usize, StageSolution)>, FitError> {
    let loss = super::models::loss_code(&options.loss).expect("validated loss");
    let point_kernel = kernel.point_kernel();
    let coarse = !refinement && options.loss == "linear";
    let (ftol, xtol, gtol) = if coarse { (PROXY_TOL, PROXY_TOL, PROXY_TOL) } else { (IMAGE_FTOL, IMAGE_FTOL, IMAGE_GTOL) };
    // Cells sharing axes and shape form one group, solved group after
    // group with the oracle's two checks before each; a group's cells
    // solve in parallel.
    let mut groups: Vec<Vec<&(usize, StageItem)>> = Vec::new();
    for entry in items {
        let same = |other: &&(usize, StageItem)| {
            let (a, b) = (other.1.data, entry.1.data);
            a.height() == b.height() && a.width() == b.width() && a.x.iter().zip(b.x.iter()).all(|(p, q)| p.to_bits() == q.to_bits()) && a.y.iter().zip(b.y.iter()).all(|(p, q)| p.to_bits() == q.to_bits())
        };
        match groups.iter_mut().find(|group| same(&group[0])) {
            Some(group) => group.push(entry),
            None => groups.push(vec![entry]),
        }
    }
    let mut out = Vec::with_capacity(items.len());
    for group in groups {
    check.check()?;
    check.check()?;
    // `direct_seed`: the proxy's warm starts, checked group by group.
    let group_warm: Vec<Option<Vec<f64>>> = group
        .iter()
        .map(|entry| match (&warm[entry.0], refinement) {
            (Some(start), false) => warm_seed(model, start).map(Some),
            _ => Ok(None),
        })
        .collect::<Result<_, _>>()?;
    let solutions: Vec<(usize, Result<StageSolution, FitError>)> = group
        .par_iter()
        .zip(group_warm.par_iter())
        .map(|(entry, warm_seed)| {
            let (cell, item) = (&entry.0, &entry.1);
            let data = item.data;
            let (height, width) = (data.height(), data.width());
            let points = height * width;
            // The refinement reads its plane centred on the first usable
            // sample; the proxy reads its own float copy as it is.
            let context = refinement.then(|| centred_context(&data.observations, data.valid_mask.as_deref(), width));
            let plane = Plane { samples: &data.observations, reference: context.map_or(0.0, |c| c[0]) };
            let all_finite = data.valid_mask.is_none() && (!data.observations.is_float() || (refinement && context.is_some_and(|c| c[3] == points as f64)));
            // `None` when every sample is valid, which a camera frame is.
            let valid: Option<Scratch<bool>> = if all_finite {
                None
            } else {
                Some(Scratch::collected(points, (0..points).map(|p| data.valid_mask.as_ref().is_none_or(|m| m[p]) && (plane.samples.get(p) - plane.reference).is_finite())))
            };
            let (mut lower, mut upper) = (base_lower.to_vec(), base_upper.to_vec());
            if refinement {
                // refinement_bounds
                let mut floors: Vec<(usize, f64)> = Vec::new();
                for (index, coordinates) in [(kernel.x_radius_index, &data.x), (kernel.y_radius_index, &data.y)] {
                    let differences: Vec<f64> = coordinates.windows(2).map(|w| (w[1] - w[0]).abs()).collect();
                    let resolution = if differences.is_empty() { EPSILON } else { super::shared::min_max(&differences).0 };
                    match floors.iter_mut().find(|(i, _)| *i == index) {
                        Some(entry) => entry.1 = if resolution < entry.1 { resolution } else { entry.1 },
                        None => floors.push((index, if resolution < f64::INFINITY { resolution } else { f64::INFINITY })),
                    }
                }
                for (index, resolution) in floors {
                    lower[index] = pmax(0.5 * resolution, EPSILON);
                }
                let p = lower.len();
                let (xl, xh) = super::shared::min_max(&data.x);
                let (yl, yh) = super::shared::min_max(&data.y);
                lower[p - 2] = xl;
                upper[p - 2] = xh;
                lower[p - 1] = yl;
                upper[p - 1] = yh;
            }
            let free_mask: Vec<bool> = (0..lower.len()).map(|i| !(requested_mask[i] && requested_lower[i] == requested_upper[i])).collect();
            let free_indices: Vec<usize> = (0..lower.len()).filter(|&i| free_mask[i]).collect();
            let authored: Option<Vec<Vec<f64>>> = match seeds {
                Some(list) => list.iter().find(|(c, _)| c == cell).map(|(_, s)| vec![s.clone()]),
                None => authored_initial.map(|row| vec![row.to_vec()]),
            };
            // The proxy's point-wise preparation reads expanded coordinates
            // and its own float values; the refinement prepares nothing.
            let (expanded_x, expanded_y) = if refinement {
                (Scratch::filled(1, 0.0), Scratch::filled(1, 0.0))
            } else {
                (Scratch::collected(points, (0..points).map(|p| data.x[p % width])), Scratch::collected(points, (0..points).map(|p| data.y[p / width])))
            };
            let axes = super::models::Axes::two(&expanded_x, &expanded_y);
            let mut prepare = |block: &mut [f64], rows: usize, lower: &mut [f64], upper: &mut [f64]| -> Zr<i64> {
                if refinement {
                    return Ok(0);
                }
                let mut seeds = Seeds { data: block, rows, cols: lower.len() };
                // `np.asarray(source, dtype=np.float64)`: a frame small enough
                // to be its own proxy is still read as floats here.
                let converted: Vec<f64>;
                let values: &[f64] = match &data.observations {
                    Samples::F64(values) => values,
                    other => {
                        converted = other.to_f64();
                        &converted
                    }
                };
                let everywhere: Vec<bool>;
                let valid: &[bool] = match &valid {
                    Some(valid) => valid,
                    None => {
                        everywhere = vec![true; points];
                        &everywhere
                    }
                };
                let count = point_kernel.prepare(&axes, values, valid, &mut seeds, lower, upper)?;
                regular_resolution_floor(&expanded_x, &expanded_y, 2, lower);
                if !kernel.radial {
                    regular_resolution_floor(&expanded_x, &expanded_y, 3, lower);
                }
                let _ = upper;
                Ok(count)
            };
            let max_candidates = point_kernel.max_candidates();
            let prepared = match trf::prepare_one(&mut prepare, &lower, &upper, requested_lower, requested_upper, requested_mask, warm_seed.as_deref(), authored.as_deref(), &free_mask, max_candidates, 1 + max_candidates.max(1)) {
                Ok(p) => p,
                Err(_) => return (*cell, Err(FitError::ZeroDivision)),
            };
            let selected = valid.as_ref().map_or(points, |valid| valid.iter().filter(|&&v| v).count());
            let objective = GridObjective { x: &data.x, y: &data.y, plane, valid: valid.as_deref().map(|v| &v[..]), context, loss_code: loss, radial: kernel.radial, selected, observed_sum: std::sync::OnceLock::new(), memo: Memo::default(), proxy: !refinement };
            let budget = options.evaluation_budget(points);
            let solution = trf::solve_cell(&objective, &prepared.seeds, &prepared.lower, &prepared.upper, &free_indices, budget, ftol, xtol, gtol, warm_seed.is_some(), prepared.status);
            let mut raw_rss = solution.raw_rss;
            let mut information = None;
            if refinement && options.loss == "linear" {
                if let Some(c) = context {
                    if c[3] == points as f64 {
                        let mut centred = solution.parameters.clone();
                        centred[1] -= c[0];
                        let (matrix, rss) = information_and_rss(&data.x, &data.y, &centred, kernel.radial, &objective.plane);
                        raw_rss = rss;
                        information = Some(if matrix.iter().all(|v| v.is_finite()) { Some(matrix) } else { None });
                    }
                }
            }
            let _ = model;
            let diagnostics = SolveDiagnostics { status: solution.status, nfev: solution.nfev, njev: solution.njev, iterations: solution.iterations, winner_seed: solution.winner_seed, cost: solution.cost, raw_rss: solution.raw_rss };
            (*cell, Ok(StageSolution { success: solution.status > STATUS_MAX_NFEV, status: solution.status, parameters: solution.parameters, raw_rss, information, diagnostics }))
        })
        .collect();
    for (cell, solution) in solutions {
        out.push((cell, solution?));
    }
    }
    out.sort_by_key(|(cell, _)| *cell);
    Ok(out)
}

/// `_regular_resolution_floor(coordinates, radius_index, lower)`.
fn regular_resolution_floor(x: &[f64], y: &[f64], radius_index: usize, lower: &mut [f64]) {
    let points = x.len();
    let mut width = 1;
    let first_y = y[0];
    while width < points && y[width] == first_y {
        width += 1;
    }
    let mut resolution = f64::INFINITY;
    for column in 1..width {
        resolution = super::num::pmin(resolution, (x[column] - x[column - 1]).abs());
    }
    let height = points / width;
    for row in 1..height {
        resolution = super::num::pmin(resolution, (y[row * width] - y[(row - 1) * width]).abs());
    }
    if !resolution.is_finite() || resolution <= 0.0 {
        resolution = EPSILON;
    }
    lower[radius_index] = pmax(0.5 * resolution, EPSILON);
}

/// How many threads a regular-image fit takes. Its parallel work is a few
/// 64-row stripes per evaluation and two seeds; on live 372 x 460 camera
/// frames 3 threads fit as fast as the whole 16-thread pool on ~1.7 cores
/// instead of ~5.7, and the shared pool stays free
/// for the raster and the other panels. The stripes and seeds are fixed, so
/// the answers do not depend on the count.
const IMAGE_FIT_THREADS: usize = 3;

fn on_image_fit_threads<T: Send>(work: impl FnOnce() -> T + Send) -> T {
    static POOL: std::sync::OnceLock<Option<rayon::ThreadPool>> = std::sync::OnceLock::new();
    let pool = POOL.get_or_init(|| rayon::ThreadPoolBuilder::new().num_threads(IMAGE_FIT_THREADS)
        .thread_name(|index| format!("zlc-image-fit-{index}")).build().ok());
    match pool { Some(pool) => pool.install(work), None => work() }
}

/// `fit_regular_separable_images`: every regular image through one
/// proxy-to-full route, on [`IMAGE_FIT_THREADS`] threads.
pub fn fit_regular_separable_images(
    model: &Arc<FitModelSpec>,
    items: &[Arc<RegularImageFitInput>],
    data_revisions: &[i64],
    request: &FitRequest,
    warm_starts: &[Option<Vec<f64>>],
    options: &FitOptions,
) -> Result<Vec<CellOutcome>, FitError> {
    on_image_fit_threads(|| fit_images(model, items, data_revisions, request, warm_starts, options))
}

fn fit_images(
    model: &Arc<FitModelSpec>,
    items: &[Arc<RegularImageFitInput>],
    data_revisions: &[i64],
    request: &FitRequest,
    warm_starts: &[Option<Vec<f64>>],
    options: &FitOptions,
) -> Result<Vec<CellOutcome>, FitError> {
    if items.is_empty() {
        return Ok(Vec::new());
    }
    if data_revisions.len() != items.len() || warm_starts.len() != items.len() {
        return Err(FitError::value("regular-image batch metadata must match its cells"));
    }
    let kernel = kernel_for(model)?;
    let check = Check::new(options.deadline_seconds, request.cancelled);
    let mut outcomes: Vec<Option<CellOutcome>> = vec![None; items.len()];
    let mut prepared: Vec<Option<(CroppedImage, RegularImageFitInput)>> = Vec::with_capacity(items.len());
    for item in items {
        check.check()?;
        let attempt = crop_to_valid_bounds(item, &check).and_then(|cropped| subsample(&cropped.data, SAMPLE_LIMIT, &check).map(|proxy| (cropped, proxy)));
        match attempt {
            Ok(pair) => prepared.push(Some(pair)),
            Err(error @ (FitError::Cancelled | FitError::Deadline)) => return Err(error),
            Err(error) => {
                outcomes[prepared.len()] = Some(Err(error.failure_text()));
                prepared.push(None);
            }
        }
    }
    let (base_lower, base_upper) = solver_bounds(model, None, None)?;
    let (requested_lower, requested_upper) = solver_bounds(model, None, request.bounds.as_ref())?;
    let requested_mask: Vec<bool> = model.parameters.iter().map(|p| request.bounds.as_ref().is_some_and(|b| b.contains(&p.name))).collect();
    let (fixed_names, free_indices) = fixed_parameter_partition(model, request.bounds.as_ref());
    let authored_initial = match &request.initial {
        Some(initial) => Some(authored_seed(model, initial)?),
        None => None,
    };
    let active: Vec<usize> = (0..items.len()).filter(|&c| prepared[c].is_some()).collect();
    let proxy_items: Vec<(usize, StageItem)> = active.iter().map(|&c| (c, StageItem { data: &prepared[c].as_ref().expect("active").1 })).collect();
    let proxy = solve_stage(kernel, model, &proxy_items, None, false, &base_lower, &base_upper, &requested_lower, &requested_upper, &requested_mask, authored_initial.as_deref(), warm_starts, options, &check)?;
    let seeds: Vec<(usize, Vec<f64>)> = proxy.iter().map(|(c, s)| (*c, s.parameters.clone())).collect();
    let full_items: Vec<(usize, StageItem)> = active.iter().map(|&c| (c, StageItem { data: &prepared[c].as_ref().expect("active").0.data })).collect();
    let full = solve_stage(kernel, model, &full_items, Some(&seeds), true, &base_lower, &base_upper, &requested_lower, &requested_upper, &requested_mask, authored_initial.as_deref(), warm_starts, options, &check)?;
    check.check()?;
    let parameter_count = model.parameters.len();
    let mut finished: Vec<(usize, Vec<f64>, bool, String, usize, f64, Vec<f64>, SolveDiagnostics)> = Vec::new();
    for (cell, solution) in full {
        let diagnostics = solution.diagnostics;
        let (cropped, _) = prepared[cell].as_ref().expect("active");
        let attempt = (|| -> Result<(Vec<f64>, bool, String, usize, f64, Vec<f64>), FitError> {
            let message = trf::termination_message(solution.status);
            let (raw_rss, information, observation_count) = match solution.information {
                Some(info) => {
                    let info = info.ok_or_else(|| FitError::FloatingPoint("regular-image information is non-finite".into()))?;
                    (solution.raw_rss, info, cropped.data.observations.len())
                }
                None => {
                    let (_cost, rss, info, count) = striped_objective(kernel, &cropped.data, &solution.parameters, &options.loss, &check)?;
                    (rss, info, count)
                }
            };
            if observation_count <= free_indices.len() {
                return Err(too_few_points(model, free_indices.len(), observation_count as i64));
            }
            let degrees = (observation_count - free_indices.len()).max(1) as f64;
            Ok((solution.parameters.clone(), solution.success, message, observation_count, raw_rss / degrees, information))
        })();
        match attempt {
            Ok((p, s, m, c, r, i)) => finished.push((cell, p, s, m, c, r, i, diagnostics)),
            Err(error @ (FitError::Cancelled | FitError::Deadline)) => return Err(error),
            Err(error) => outcomes[cell] = Some(Err(error.failure_text())),
        }
    }
    check.check()?;
    // The oracle checks again between the covariances and the results;
    // nothing between can fail.
    check.check()?;
    for (cell, parameters, success, message, observation_count, reduced, information, diagnostics) in finished {
        let mut covariance = vec![0.0; parameter_count * parameter_count];
        let mut covariance_valid = true;
        if !free_indices.is_empty() {
            let n = free_indices.len();
            let mut free_matrix = vec![0.0; n * n];
            for (a, &i) in free_indices.iter().enumerate() {
                for (b, &j) in free_indices.iter().enumerate() {
                    free_matrix[a * n + b] = information[i * parameter_count + j];
                }
            }
            let (free_covariance, valid) = covariance_from_information(&free_matrix, n, reduced, observation_count);
            covariance_valid = valid;
            for (a, &i) in free_indices.iter().enumerate() {
                for (b, &j) in free_indices.iter().enumerate() {
                    covariance[i * parameter_count + j] = free_covariance[a * n + b];
                }
            }
        }
        let errors: Vec<f64> = (0..parameter_count).map(|i| pmax_np(covariance[i * parameter_count + i], 0.0).sqrt()).collect();
        let (cropped, _) = prepared[cell].as_ref().expect("active");
        let data = cropped.data.clone();
        let origin = cropped.origin;
        let result_parameters = parameters.clone();
        let deferred = DeferredArrays::new(Box::new(move || result_arrays(kernel, &data, &result_parameters, observation_count, origin)));
        let outcome = FitResult::new(model.clone(), parameters, errors, covariance, ObservationArrays::Deferred(Arc::new(deferred)), data_revisions[cell] as u64, success, &message, reduced, covariance_valid, fixed_names.clone())
            .map(|result| FitResult { diagnostics: Some(diagnostics), ..result });
        outcomes[cell] = Some(outcome.map_err(|e| e.failure_text()));
    }
    Ok(outcomes.into_iter().map(|o| o.unwrap_or_else(|| Err("regular-image fit failed".into()))).collect())
}

