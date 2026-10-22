//! The thirteen built-in model kernels of `_fit_compiled`: one point value
//! and analytic Jacobian row per model, the objective accumulation every
//! descriptor shares, and the value/Jacobian evaluation of the finaliser.
//!
//! Operation order follows the numba source line for line (see
//! `TOLERANCES.md`); every division is numba's raising one.

use super::num::{div, pmax, pmin, KernelError, Zr, ALLOCATION_LIMIT, EPSILON, PI};

pub const COUNT_FLOOR: f64 = 1.0e-9;
pub const SQRT_TWO_PI: f64 = 2.5066282746310002;
pub const MAD_TO_SIGMA: f64 = 1.4826;

pub const LOSS_LINEAR: i64 = 0;
pub const LOSS_SOFT_L1: i64 = 1;
pub const LOSS_HUBER: i64 = 2;
pub const LOSS_CAUCHY: i64 = 3;
pub const LOSS_ARCTAN: i64 = 4;

/// `LOSS_CODES[name]`.
pub fn loss_code(name: &str) -> Option<i64> {
    Some(match name {
        "linear" => LOSS_LINEAR,
        "soft_l1" => LOSS_SOFT_L1,
        "huber" => LOSS_HUBER,
        "cauchy" => LOSS_CAUCHY,
        "arctan" => LOSS_ARCTAN,
        _ => return None,
    })
}

/// Which compiled point kernel a built-in model evaluates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kernel {
    Lorentzian,
    Gaussian,
    Histogram,
    Bimodal,
    Poisson,
    PoissonBimodal,
    Doublet,
    Damped,
    Exponential,
    Saturation,
    ReleaseRecapture,
    Radial,
    Anisotropic,
}

impl Kernel {
    pub fn parameter_count(self) -> usize {
        match self {
            Kernel::Histogram | Kernel::Poisson | Kernel::Exponential | Kernel::Saturation => 3,
            Kernel::Lorentzian | Kernel::Gaussian | Kernel::ReleaseRecapture => 4,
            Kernel::Doublet | Kernel::Damped | Kernel::Radial => 5,
            Kernel::Bimodal | Kernel::PoissonBimodal | Kernel::Anisotropic => 6,
        }
    }

    pub fn arity(self) -> usize {
        match self {
            Kernel::Radial | Kernel::Anisotropic => 2,
            _ => 1,
        }
    }

    /// `descriptor.max_candidates`.
    pub fn max_candidates(self) -> usize {
        match self {
            Kernel::Gaussian | Kernel::Histogram | Kernel::Bimodal | Kernel::Poisson | Kernel::PoissonBimodal => 1,
            Kernel::Lorentzian | Kernel::Exponential | Kernel::Radial | Kernel::Anisotropic => 2,
            Kernel::Damped | Kernel::Saturation => 3,
            Kernel::ReleaseRecapture => 4,
            Kernel::Doublet => 8,
        }
    }
}

/// Coordinates of a point-layout cell: one slice per independent axis.
#[derive(Clone, Copy)]
pub struct Axes<'a> {
    pub axes: [&'a [f64]; 2],
    pub arity: usize,
}

impl<'a> Axes<'a> {
    pub fn one(x: &'a [f64]) -> Self { Axes { axes: [x, &[]], arity: 1 } }
    pub fn two(x: &'a [f64], y: &'a [f64]) -> Self { Axes { axes: [x, y], arity: 2 } }
    pub fn from_slices(axes: &[&'a [f64]]) -> Self {
        match axes.len() {
            1 => Axes::one(axes[0]),
            _ => Axes::two(axes[0], axes[1]),
        }
    }
    #[inline(always)]
    pub fn at(&self, axis: usize, point: usize) -> f64 { self.axes[axis][point] }
    pub fn len(&self) -> usize { self.axes[0].len() }
    pub fn is_empty(&self) -> bool { self.axes[0].is_empty() }
}

// --------------------------------------------------------------------------
// Poisson-Gaussian quadrature
// --------------------------------------------------------------------------

pub const POISSON_GAUSSIAN_WINDOW: f64 = 8.0;
pub const POISSON_NODES_PER_SCALE: f64 = 12.0;
pub const POISSON_MAX_NODES_PER_UNIT: i64 = 4096;
pub const POISSON_TAIL: f64 = 50.0;
pub const POISSON_ZERO_RATE: f64 = 1e-150;
pub const EULER_GAMMA: f64 = 0.5772156649015329;
pub const ZETA_2: f64 = 1.6449340668482264;
pub const ZETA_3: f64 = 1.2020569031595943;

/// `_poisson_grid`: `(table, first, n, mass, mean)`.
#[derive(Debug, Clone, Default)]
pub struct PoissonGrid {
    pub table: Vec<f64>,
    pub first: i64,
    pub n: i64,
    pub mass: f64,
    pub mean: f64,
}

fn poisson_support(rate: f64) -> Zr<(f64, f64)> {
    let root = rate.sqrt();
    let mut high = rate + 12.0 * root + 20.0;
    if rate < 1.0 {
        high = pmin(high, div(POISSON_TAIL, -rate.ln())? + 1.0);
    }
    Ok((pmax(rate - 10.0 * root - 10.0, 0.0), high))
}

fn poisson_nodes_per_unit(rate: f64, sigma: f64) -> Zr<i64> {
    let scale = if rate >= 1.0 { rate.sqrt() } else { div(1.0, 1.0 + rate.ln().abs())? };
    let nodes = div(POISSON_NODES_PER_SCALE, pmin(sigma, scale))?;
    let ceiled = super::num::numba_int(nodes.ceil());
    Ok(ceiled.max(1).min(POISSON_MAX_NODES_PER_UNIT))
}

#[inline]
fn py_max_i(a: i64, b: i64) -> i64 { if b > a { b } else { a } }
#[inline]
fn py_min_i(a: i64, b: i64) -> i64 { if b < a { b } else { a } }

pub fn poisson_grid(rate: f64, sigma: f64) -> Zr<PoissonGrid> {
    if !(rate >= POISSON_ZERO_RATE) || !(sigma > 0.0) {
        return Ok(PoissonGrid { table: Vec::new(), first: 0, n: 1, mass: 1.0, mean: 0.0 });
    }
    let n = poisson_nodes_per_unit(rate, sigma)?;
    let (low, high) = poisson_support(rate)?;
    let nf = n as f64;
    if !((high * nf).ceil() - (low * nf).floor() + 1.0 <= ALLOCATION_LIMIT) { return Err(KernelError::Allocation); }
    let first = super::num::numba_int((low * nf).floor());
    let last = super::num::numba_int((high * nf).ceil());
    let size = (last - first + 1).max(0) as usize;
    let mut table = vec![0.0_f64; size];
    let h = div(1.0, nf)?;
    let log_rate = rate.ln();
    let start = py_max_i(py_min_i(super::num::numba_int((rate * nf + 0.5).floor()), last - n + 1), first);
    let stop = py_min_i(start + n - 1, last);
    let at = |m: i64| (m - first) as usize;
    for m in start..=stop {
        let u = m as f64 * h;
        table[at(m)] = (u * log_rate - rate - super::num::lgamma(u + 1.0)).exp();
    }
    for m in (stop + 1)..=last {
        table[at(m)] = div(table[at(m - n)] * rate, m as f64 * h)?;
    }
    let mut m = start - 1;
    while m > first - 1 {
        table[at(m)] = div(table[at(m + n)] * (m as f64 * h + 1.0), rate)?;
        m -= 1;
    }
    let mut mass = 0.0;
    let mut moment = 0.0;
    for m in first..=last {
        let mut weight = table[at(m)];
        if m == 0 {
            weight *= 0.5;
        }
        mass += weight;
        moment += weight * (m as f64 * h);
    }
    mass *= h;
    moment *= h;
    if first == 0 {
        let a = log_rate + EULER_GAMMA;
        let p2 = a * a - ZETA_2;
        let p3 = a * a * a - 3.0 * a * ZETA_2 + 2.0 * ZETA_3;
        let c2 = h * h / 12.0 * table[0];
        let c4 = h * h * h * h / 720.0 * table[0];
        mass += c2 * a - c4 * p3;
        moment += c2 - 3.0 * c4 * p2;
    }
    let mean = div(moment, mass)?;
    Ok(PoissonGrid { table, first, n, mass, mean })
}

fn poisson_convolution(x: f64, rate: f64, sigma: f64, grid: &PoissonGrid, derivatives: bool) -> Zr<(f64, f64, f64)> {
    let table = &grid.table;
    if table.is_empty() {
        return Ok((0.0, 0.0, 0.0));
    }
    let (first, n) = (grid.first, grid.n);
    let nf = n as f64;
    let h = div(1.0, nf)?;
    let reach = POISSON_GAUSSIAN_WINDOW * sigma;
    let m_a = py_max_i(first, super::num::numba_int(((x - reach) * nf).ceil()));
    let m_b = py_min_i(first + table.len() as i64 - 1, super::num::numba_int(((x + reach) * nf).floor()));
    if m_b < m_a {
        return Ok((0.0, 0.0, 0.0));
    }
    let inverse = div(1.0, sigma * sigma)?;
    let mut delta = x - m_a as f64 * h;
    let mut shape = (-0.5 * delta * delta * inverse).exp();
    let mut ratio = ((delta - 0.5 * h) * h * inverse).exp();
    let decay = (-h * h * inverse).exp();
    let mut s0 = 0.0;
    let mut s1 = 0.0;
    let mut s2 = 0.0;
    for m in m_a..=m_b {
        if ((m - m_a) & 63) == 63 {
            shape = (-0.5 * delta * delta * inverse).exp();
            ratio = ((delta - 0.5 * h) * h * inverse).exp();
        }
        let mut weight = table[(m - first) as usize] * shape;
        if m == 0 {
            weight *= 0.5;
        }
        s0 += weight;
        if derivatives {
            s1 += weight * (m as f64 * h);
            s2 += weight * delta * delta;
        }
        delta -= h;
        shape *= ratio;
        ratio *= decay;
    }
    let mut i0 = h * s0;
    let mut i1 = h * s1;
    let mut i2 = h * s2;
    if m_a == 0 {
        let a = rate.ln() + EULER_GAMMA;
        let p2 = a * a - ZETA_2;
        let p3 = a * a * a - 3.0 * a * ZETA_2 + 2.0 * ZETA_3;
        let t = div(x, sigma)?;
        let g1 = div(t, sigma)?;
        let g2 = (t * t - 1.0) * inverse;
        let g3 = div((t * t * t - 3.0 * t) * inverse, sigma)?;
        let d1 = a + g1;
        let d2 = p2 + 2.0 * a * g1 + g2;
        let d3 = p3 + 3.0 * p2 * g1 + 3.0 * a * g2 + g3;
        let f0 = table[0] * (-0.5 * t * t).exp();
        let c2 = h * h / 12.0 * f0;
        let c4 = h * h * h * h / 720.0 * f0;
        i0 += c2 * d1 - c4 * d3;
        if derivatives {
            i1 += c2 - 3.0 * c4 * d2;
            i2 += c2 * (d1 * x * x - 2.0 * x) - c4 * (d3 * x * x - 6.0 * x * d2 + 6.0 * d1);
        }
    }
    Ok((i0, i1, i2))
}

fn poisson_component(x: f64, amplitude: f64, rate: f64, sigma: f64, grid: &PoissonGrid, derivatives: bool) -> Zr<(f64, f64, f64)> {
    if !(sigma > 0.0) {
        return Ok((0.0, 0.0, 0.0));
    }
    let mut norm = div(1.0, sigma * SQRT_TWO_PI)?;
    if !(rate >= POISSON_ZERO_RATE) {
        let t = div(x, sigma)?;
        let shape = norm * (-0.5 * t * t).exp();
        let sigma_d = if derivatives { div(amplitude * shape * (t * t - 1.0), sigma)? } else { 0.0 };
        return Ok((shape, 0.0, sigma_d));
    }
    let (i0, i1, i2) = poisson_convolution(x, rate, sigma, grid, derivatives)?;
    norm = div(norm, grid.mass)?;
    let shape = i0 * norm;
    if !derivatives {
        return Ok((shape, 0.0, 0.0));
    }
    let rate_derivative = div(amplitude * norm * (i1 - i0 * grid.mean), rate)?;
    let sigma_derivative = div(amplitude * norm * (div(i2, sigma * sigma)? - i0), sigma)?;
    Ok((shape, rate_derivative, sigma_derivative))
}

// --------------------------------------------------------------------------
// Release-recapture Lambert W
// --------------------------------------------------------------------------

fn release_recapture_w(frequency: f64, time: f64) -> Zr<f64> {
    let magnitude = (frequency * time).abs() * (2.0 * PI);
    if magnitude < 1.0e-4 {
        let z = magnitude * magnitude;
        return Ok(z * (1.0 - z * (1.0 - 1.5 * z)));
    }
    if magnitude > 1.0e150 {
        let log_z = 2.0 * ((2.0 * PI).ln() + frequency.ln() + time.abs().ln());
        let mut w = log_z - log_z.ln();
        for _ in 0..8 {
            let step = div((w + w.ln() - log_z) * w, 1.0 + w)?;
            w -= step;
            if step.abs() <= 2.0 * EPSILON * w {
                break;
            }
        }
        return Ok(w);
    }
    let z = magnitude * magnitude;
    let mut w = z.ln_1p();
    if z > 3.0 {
        let log_z = z.ln();
        w = log_z - log_z.ln();
    }
    for _ in 0..12 {
        let residual = w - z * (-w).exp();
        let step = div(residual, 1.0 + w - div(0.5 * (w + 2.0) * residual, w + 1.0)?)?;
        w -= step;
        if step.abs() <= 2.0 * EPSILON * (1.0 + w) {
            break;
        }
    }
    Ok(w)
}

// --------------------------------------------------------------------------
// Point kernels
// --------------------------------------------------------------------------

/// Per-evaluation precomputation (the Poisson grids), built once per
/// objective / value call from the parameters, as the numba callbacks do.
#[derive(Debug, Clone, Default)]
pub struct Pre {
    pub left: PoissonGrid,
    pub right: PoissonGrid,
}

impl Kernel {
    pub fn pre(self, params: &[f64]) -> Zr<Pre> {
        Ok(match self {
            Kernel::Poisson => Pre { left: poisson_grid(params[1], params[2])?, right: PoissonGrid::default() },
            Kernel::PoissonBimodal => Pre {
                left: poisson_grid(params[1], params[2])?,
                right: poisson_grid(params[1] + params[3], params[4])?,
            },
            _ => Pre::default(),
        })
    }

    /// `_point_<model>(coords, point, parameters, row)`; `row` is empty for
    /// a value-only call (numba's `row.size == 0`).
    #[inline(always)]
    pub fn point(self, pre: &Pre, coords: &Axes, point: usize, p: &[f64], row: &mut [f64]) -> Zr<f64> {
        let want = !row.is_empty();
        match self {
            Kernel::Lorentzian => {
                let (center, width, amplitude, offset) = (p[0], p[1], p[2], p[3]);
                let delta = coords.at(0, point) - center;
                let half = 0.5 * width;
                let half_squared = half * half;
                let denominator = delta * delta + half_squared;
                let shape = div(half_squared, denominator)?;
                if want {
                    let denominator_squared = denominator * denominator;
                    row[0] = div(amplitude * half_squared * 2.0 * delta, denominator_squared)?;
                    row[1] = div(amplitude * half * delta * delta, denominator_squared)?;
                    row[2] = shape;
                    row[3] = 1.0;
                }
                Ok(amplitude * shape + offset)
            }
            Kernel::Gaussian => {
                let (amplitude, offset, sigma, center) = (p[0], p[1], p[2], p[3]);
                let delta = coords.at(0, point) - center;
                let sigma2 = sigma * sigma;
                let shape = div(-0.5 * delta * delta, sigma2)?.exp();
                if want {
                    row[0] = shape;
                    row[1] = 1.0;
                    row[2] = div(amplitude * shape * delta * delta, sigma2 * sigma)?;
                    row[3] = div(amplitude * shape * delta, sigma2)?;
                }
                Ok(amplitude * shape + offset)
            }
            Kernel::Histogram => {
                let (amplitude, center, sigma) = (p[0], p[1], p[2]);
                let delta = coords.at(0, point) - center;
                let sigma2 = sigma * sigma;
                let density = div(div(-0.5 * delta * delta, sigma2)?.exp(), sigma * SQRT_TWO_PI)?;
                let value = amplitude * density;
                if want {
                    row[0] = density;
                    row[1] = div(value * delta, sigma2)?;
                    row[2] = value * (div(delta * delta, sigma2 * sigma)? - div(1.0, sigma)?);
                }
                Ok(value)
            }
            Kernel::Bimodal => {
                let (amplitude, center, sigma, delta_center, sigma_b, ratio) = (p[0], p[1], p[2], p[3], p[4], p[5]);
                let x = coords.at(0, point);
                let delta_a = x - center;
                let delta_b = x - center - delta_center;
                let sigma2 = sigma * sigma;
                let sigma_b2 = sigma_b * sigma_b;
                let density_a = div(div(-0.5 * delta_a * delta_a, sigma2)?.exp(), sigma * SQRT_TWO_PI)?;
                let density_b = div(div(-0.5 * delta_b * delta_b, sigma_b2)?.exp(), sigma_b * SQRT_TWO_PI)?;
                let value_a = amplitude * (1.0 - ratio) * density_a;
                let value_b = amplitude * ratio * density_b;
                if want {
                    row[0] = (1.0 - ratio) * density_a + ratio * density_b;
                    row[1] = div(value_a * delta_a, sigma2)? + div(value_b * delta_b, sigma_b2)?;
                    row[2] = value_a * (div(delta_a * delta_a, sigma2 * sigma)? - div(1.0, sigma)?);
                    row[3] = div(value_b * delta_b, sigma_b2)?;
                    row[4] = value_b * (div(delta_b * delta_b, sigma_b2 * sigma_b)? - div(1.0, sigma_b)?);
                    row[5] = amplitude * (density_b - density_a);
                }
                Ok(value_a + value_b)
            }
            Kernel::Poisson => {
                let (amplitude, rate, sigma) = (p[0], p[1], p[2]);
                let (shape, rate_d, sigma_d) = poisson_component(coords.at(0, point), amplitude, rate, sigma, &pre.left, want)?;
                if want {
                    row[0] = shape;
                    row[1] = rate_d;
                    row[2] = sigma_d;
                }
                Ok(amplitude * shape)
            }
            Kernel::PoissonBimodal => {
                let (amplitude, rate, sigma, delta_rate, sigma_b, ratio) = (p[0], p[1], p[2], p[3], p[4], p[5]);
                let x = coords.at(0, point);
                let amplitude_a = amplitude * (1.0 - ratio);
                let amplitude_b = amplitude * ratio;
                let (shape_a, rate_d_a, sigma_d_a) = poisson_component(x, amplitude_a, rate, sigma, &pre.left, want)?;
                let (shape_b, rate_d_b, sigma_d_b) = poisson_component(x, amplitude_b, rate + delta_rate, sigma_b, &pre.right, want)?;
                if want {
                    row[0] = (1.0 - ratio) * shape_a + ratio * shape_b;
                    row[1] = rate_d_a + rate_d_b;
                    row[2] = sigma_d_a;
                    row[3] = rate_d_b;
                    row[4] = sigma_d_b;
                    row[5] = amplitude * (shape_b - shape_a);
                }
                Ok(amplitude_a * shape_a + amplitude_b * shape_b)
            }
            Kernel::Doublet => {
                let (center, width, amplitude, offset, splitting) = (p[0], p[1], p[2], p[3], p[4]);
                let x = coords.at(0, point);
                let half = 0.5 * width;
                let half_squared = half * half;
                let left_delta = x - (center - 0.5 * splitting);
                let right_delta = x - (center + 0.5 * splitting);
                let left_den = left_delta * left_delta + half_squared;
                let right_den = right_delta * right_delta + half_squared;
                let left_shape = div(half_squared, left_den)?;
                let right_shape = div(half_squared, right_den)?;
                if want {
                    let left_den2 = left_den * left_den;
                    let right_den2 = right_den * right_den;
                    let left_center_d = div(amplitude * half_squared * 2.0 * left_delta, left_den2)?;
                    let right_center_d = div(amplitude * half_squared * 2.0 * right_delta, right_den2)?;
                    row[0] = left_center_d + right_center_d;
                    row[1] = amplitude * half * (div(left_delta * left_delta, left_den2)? + div(right_delta * right_delta, right_den2)?);
                    row[2] = left_shape + right_shape;
                    row[3] = 1.0;
                    row[4] = -0.5 * left_center_d + 0.5 * right_center_d;
                }
                Ok(amplitude * (left_shape + right_shape) + offset)
            }
            Kernel::Damped => {
                let (amplitude, offset, frequency, decay, phase) = (p[0], p[1], p[2], p[3], p[4]);
                let x = coords.at(0, point);
                let exponential = div(-x, decay)?.exp();
                let argument = 2.0 * PI * frequency * x + phase;
                let sine = argument.sin();
                let exp_sine = exponential * sine;
                if want {
                    let exp_cosine = exponential * argument.cos();
                    row[0] = exp_sine;
                    row[1] = 1.0;
                    row[2] = amplitude * exp_cosine * 2.0 * PI * x;
                    row[3] = div(amplitude * exp_sine * x, decay * decay)?;
                    row[4] = amplitude * exp_cosine;
                }
                Ok(offset + amplitude * exp_sine)
            }
            Kernel::Exponential => {
                let (amplitude, offset, decay) = (p[0], p[1], p[2]);
                let x = coords.at(0, point);
                let exponential = div(-x, decay)?.exp();
                if want {
                    row[0] = exponential;
                    row[1] = 1.0;
                    row[2] = div(amplitude * exponential * x, decay * decay)?;
                }
                Ok(offset + amplitude * exponential)
            }
            Kernel::Saturation => {
                let (asymptote, numerator, shift) = (p[0], p[1], p[2]);
                let x = coords.at(0, point);
                let denominator = x + shift;
                if !(denominator > 0.0) {
                    for value in row.iter_mut() {
                        *value = f64::NAN;
                    }
                    return Ok(f64::NAN);
                }
                let inverse = div(1.0, denominator)?;
                let predicted = (asymptote * x + numerator) * inverse;
                if want {
                    row[0] = x * inverse;
                    row[1] = inverse;
                    row[2] = -predicted * inverse;
                }
                Ok(predicted)
            }
            Kernel::ReleaseRecapture => {
                let (amplitude, offset, eta, frequency) = (p[0], p[1], p[2], p[3]);
                let time = coords.at(0, point);
                if time == 0.0 {
                    if want {
                        row[0] = 1.0;
                        row[1] = 1.0;
                        row[2] = 0.0;
                        row[3] = 0.0;
                    }
                    return Ok(amplitude + offset);
                }
                let w = release_recapture_w(frequency, time)?;
                let q = (-w).exp();
                let denominator = -(-eta).exp_m1();
                let scaled = eta * q;
                let survival = div(-(-scaled).exp_m1(), denominator)?;
                if !want {
                    return Ok(amplitude * survival + offset);
                }
                let complement = -(-w).exp_m1();
                let exponential = (-scaled).exp();
                let derivative_eta = if eta < 0.01 {
                    let q2 = q * q;
                    let eta3 = eta * (eta * eta);
                    let eta5 = eta * ((eta * eta) * (eta * eta));
                    survival
                        * complement
                        * (0.5 - eta * (q + 1.0) / 12.0 + eta3 * (q + 1.0) * (q2 + 1.0) / 720.0
                            - eta5 * (1.0 + q + q2 + q2 * q + q2 * q2 + q2 * q2 * q) / 30240.0)
                } else if q > 0.5 {
                    div(exponential * (-(-eta * complement).exp_m1() - complement * denominator), denominator * denominator)?
                } else {
                    div(q * exponential - survival * (-eta).exp(), denominator)?
                };
                let derivative_frequency =
                    div(div(-2.0 * (scaled * exponential), denominator)? * div(w, 1.0 + w)?, frequency)?;
                row[0] = survival;
                row[1] = 1.0;
                row[2] = amplitude * derivative_eta;
                row[3] = amplitude * derivative_frequency;
                Ok(amplitude * survival + offset)
            }
            Kernel::Radial => {
                let (amplitude, offset, radius, center_x, center_y) = (p[0], p[1], p[2], p[3], p[4]);
                let delta_x = coords.at(0, point) - center_x;
                let delta_y = coords.at(1, point) - center_y;
                let squared = delta_x * delta_x + delta_y * delta_y;
                let radius2 = radius * radius;
                let shape = div(-squared, radius2)?.exp();
                if want {
                    row[0] = shape;
                    row[1] = 1.0;
                    row[2] = div(amplitude * shape * 2.0 * squared, radius2 * radius)?;
                    row[3] = div(amplitude * shape * 2.0 * delta_x, radius2)?;
                    row[4] = div(amplitude * shape * 2.0 * delta_y, radius2)?;
                }
                Ok(offset + amplitude * shape)
            }
            Kernel::Anisotropic => {
                let (amplitude, offset, radius_x, radius_y, center_x, center_y) = (p[0], p[1], p[2], p[3], p[4], p[5]);
                let delta_x = coords.at(0, point) - center_x;
                let delta_y = coords.at(1, point) - center_y;
                let delta_x2 = delta_x * delta_x;
                let delta_y2 = delta_y * delta_y;
                let radius_x2 = radius_x * radius_x;
                let radius_y2 = radius_y * radius_y;
                let shape = (-(div(delta_x2, radius_x2)? + div(delta_y2, radius_y2)?)).exp();
                if want {
                    row[0] = shape;
                    row[1] = 1.0;
                    row[2] = div(amplitude * shape * 2.0 * delta_x2, radius_x2 * radius_x)?;
                    row[3] = div(amplitude * shape * 2.0 * delta_y2, radius_y2 * radius_y)?;
                    row[4] = div(amplitude * shape * 2.0 * delta_x, radius_x2)?;
                    row[5] = div(amplitude * shape * 2.0 * delta_y, radius_y2)?;
                }
                Ok(offset + amplitude * shape)
            }
        }
    }

    /// `_value_jacobian_<model>(coords, parameters, with_jacobian)`:
    /// values, and a row-major `points x parameters` Jacobian (empty when
    /// not asked).
    pub fn value_jacobian(self, coords: &Axes, params: &[f64], with_jacobian: bool) -> Zr<(Vec<f64>, Vec<f64>)> {
        let points = coords.len();
        let count = params.len();
        let pre = self.pre(params)?;
        let mut output = vec![0.0; points];
        let mut jacobian = vec![0.0; if with_jacobian { points * count } else { 0 }];
        let mut empty: [f64; 0] = [];
        for point in 0..points {
            let row: &mut [f64] = if with_jacobian { &mut jacobian[point * count..(point + 1) * count] } else { &mut empty };
            output[point] = self.point(&pre, coords, point, params, row)?;
        }
        Ok((output, jacobian))
    }
}

// --------------------------------------------------------------------------
// Objective accumulation
// --------------------------------------------------------------------------

/// `_rho(z, loss_code)`.
#[inline(always)]
pub fn rho(z: f64, loss_code: i64) -> Zr<(f64, f64, f64)> {
    Ok(match loss_code {
        LOSS_LINEAR => (z, 1.0, 0.0),
        LOSS_SOFT_L1 => {
            let total = 1.0 + z;
            let root = total.sqrt();
            (2.0 * (root - 1.0), div(1.0, root)?, div(-0.5, total * root)?)
        }
        LOSS_HUBER => {
            if z <= 1.0 {
                (z, 1.0, 0.0)
            } else {
                let root = z.sqrt();
                (2.0 * root - 1.0, div(1.0, root)?, div(-0.5, z * root)?)
            }
        }
        LOSS_CAUCHY => {
            let denominator = 1.0 + z;
            (z.ln_1p(), div(1.0, denominator)?, div(-1.0, denominator * denominator)?)
        }
        _ => {
            let denominator = 1.0 + z * z;
            (z.atan(), div(1.0, denominator)?, div(-2.0 * z, denominator * denominator)?)
        }
    })
}

/// `compiled_point_terms`: `(raw, squared, cost, gradient_factor,
/// information_factor, finite)`.
#[inline(always)]
pub fn point_terms(predicted: f64, observed: f64, poisson: bool, weight: f64, use_weights: bool, loss_code: i64) -> Zr<(f64, f64, f64, f64, f64, bool)> {
    let mut raw;
    let residual_jacobian_scale;
    if poisson {
        let expected = pmax(predicted, COUNT_FLOOR);
        let logarithm = if observed > 0.0 { observed * div(observed, expected)?.ln() } else { 0.0 };
        let deviance = 2.0 * pmax(expected - observed + logarithm, 0.0);
        raw = deviance.sqrt();
        if expected < observed {
            raw = -raw;
        }
        let absolute = raw.abs();
        residual_jacobian_scale = if absolute > COUNT_FLOOR {
            div((expected - observed).abs(), expected * pmax(absolute, COUNT_FLOOR))?
        } else {
            div(1.0, expected.sqrt())?
        };
    } else {
        raw = predicted - observed;
        residual_jacobian_scale = if use_weights { weight } else { 1.0 };
        if use_weights {
            raw *= weight;
        }
    }
    if !raw.is_finite() {
        return Ok((raw, f64::INFINITY, f64::INFINITY, 0.0, 0.0, false));
    }
    let squared = raw * raw;
    let (rho0, rho1, rho2) = rho(squared, loss_code)?;
    let gradient_factor = raw * rho1 * residual_jacobian_scale;
    let information_factor = residual_jacobian_scale * residual_jacobian_scale * pmax(rho1 + 2.0 * rho2 * squared, EPSILON);
    Ok((raw, squared, 0.5 * rho0, gradient_factor, information_factor, true))
}

/// `compiled_accumulate`: gradient and the lower triangle of the
/// information, column by column.
#[inline(always)]
pub fn accumulate(jacobian_row: &[f64], gradient: &mut [f64], information: &mut [f64], gradient_factor: f64, information_factor: f64) {
    let count = jacobian_row.len();
    for row in 0..count {
        let value = jacobian_row[row];
        gradient[row] += value * gradient_factor;
        for column in 0..=row {
            information[row * count + column] += value * jacobian_row[column] * information_factor;
        }
    }
}

#[inline(always)]
pub fn reset_accumulators(gradient: &mut [f64], information: &mut [f64]) {
    for value in gradient.iter_mut() {
        *value = 0.0;
    }
    for value in information.iter_mut() {
        *value = 0.0;
    }
}

#[inline(always)]
pub fn finish_information(information: &mut [f64], count: usize) {
    for row in 0..count {
        for column in 0..row {
            information[column * count + row] = information[row * count + column];
        }
    }
}

/// Everything an objective call needs beside the parameters.
pub struct ObjectiveData<'a> {
    pub observations: &'a [f64],
    pub valid: &'a [bool],
    pub weights: &'a [f64],
    pub use_weights: bool,
    pub poisson: bool,
    pub loss_code: i64,
}

impl Kernel {
    /// `_objective_<model>`: `(cost, raw_rss, finite)`, filling the free
    /// gradient and information when `derivatives`.  The kernel is chosen
    /// once and its point function inlined into the loop.
    #[allow(clippy::too_many_arguments)]
    pub fn objective(
        self,
        coords: &Axes,
        data: &ObjectiveData,
        params: &[f64],
        free: &[usize],
        gradient: &mut [f64],
        information: &mut [f64],
        jacobian_row: &mut [f64],
        derivatives: bool,
    ) -> Zr<(f64, f64, bool)> {
        let pre = self.pre(params)?;
        // Specialized for the common case (every parameter free, plain
        // least squares) so the loop bounds and branches are constants;
        // the operations and their order are the same in every instance.
        let plain = !data.poisson && !data.use_weights && data.loss_code == LOSS_LINEAR;
        let identity = free.len() == params.len() && free.iter().enumerate().all(|(i, &p)| i == p);
        macro_rules! run {
            ($kernel:expr, $n:literal) => {{
                let point_fn = |point: usize, row: &mut [f64]| $kernel.point(&pre, coords, point, params, row);
                match (identity && params.len() == $n, plain) {
                    (true, true) => objective_loop::<$n, true, _>(data, params.len(), free, gradient, information, jacobian_row, derivatives, point_fn),
                    (true, false) => objective_loop::<$n, false, _>(data, params.len(), free, gradient, information, jacobian_row, derivatives, point_fn),
                    (false, true) => objective_loop::<0, true, _>(data, params.len(), free, gradient, information, jacobian_row, derivatives, point_fn),
                    (false, false) => objective_loop::<0, false, _>(data, params.len(), free, gradient, information, jacobian_row, derivatives, point_fn),
                }
            }};
        }
        match self {
            Kernel::Lorentzian => run!(Kernel::Lorentzian, 4),
            Kernel::Gaussian => run!(Kernel::Gaussian, 4),
            Kernel::Histogram => run!(Kernel::Histogram, 3),
            Kernel::Bimodal => run!(Kernel::Bimodal, 6),
            Kernel::Poisson => run!(Kernel::Poisson, 3),
            Kernel::PoissonBimodal => run!(Kernel::PoissonBimodal, 6),
            Kernel::Doublet => run!(Kernel::Doublet, 5),
            Kernel::Damped => run!(Kernel::Damped, 5),
            Kernel::Exponential => run!(Kernel::Exponential, 3),
            Kernel::Saturation => run!(Kernel::Saturation, 3),
            Kernel::ReleaseRecapture => run!(Kernel::ReleaseRecapture, 4),
            Kernel::Radial => run!(Kernel::Radial, 5),
            Kernel::Anisotropic => run!(Kernel::Anisotropic, 6),
        }
    }
}

/// The shared body of every `_objective_<model>`: the accumulators live in
/// fixed arrays (same operations, same order) and are written out once.
/// `N > 0`: all `N` parameters free in order; `PLAIN`: linear loss, no
/// weights, no Poisson deviance.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn objective_loop<const N: usize, const PLAIN: bool, F: FnMut(usize, &mut [f64]) -> Zr<f64>>(
    data: &ObjectiveData,
    count: usize,
    free: &[usize],
    gradient: &mut [f64],
    information: &mut [f64],
    jacobian_row: &mut [f64],
    derivatives: bool,
    mut point_fn: F,
) -> Zr<(f64, f64, bool)> {
    let nfree = if N > 0 { N } else { free.len() };
    let count = if N > 0 { N } else { count };
    let mut g = [0.0_f64; 6];
    let mut info = [0.0_f64; 36];
    let mut projected = [0.0_f64; 6];
    let mut cost = 0.0;
    let mut rss = 0.0;
    let mut full = [0.0_f64; 6];
    let observations = data.observations;
    let valid = &data.valid[..observations.len()];
    for point in 0..observations.len() {
        if !valid[point] {
            continue;
        }
        let row: &mut [f64] = if derivatives { &mut full[..count] } else { &mut full[..0] };
        let predicted = point_fn(point, row)?;
        let (_raw, squared, point_cost, gradient_factor, information_factor, finite) = if PLAIN {
            point_terms(predicted, observations[point], false, 1.0, false, LOSS_LINEAR)?
        } else {
            let weight = if data.use_weights { data.weights[point] } else { 1.0 };
            point_terms(predicted, observations[point], data.poisson, weight, data.use_weights, data.loss_code)?
        };
        if !finite || !predicted.is_finite() {
            return Ok((f64::INFINITY, f64::INFINITY, false));
        }
        if derivatives {
            if N > 0 {
                for index in 0..N {
                    if !full[index].is_finite() {
                        return Ok((f64::INFINITY, f64::INFINITY, false));
                    }
                }
                projected = full;
            } else {
                for (index, &parameter) in free.iter().enumerate() {
                    let value = full[parameter];
                    projected[index] = value;
                    if !value.is_finite() {
                        return Ok((f64::INFINITY, f64::INFINITY, false));
                    }
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
        cost += point_cost;
        rss += squared;
    }
    if derivatives {
        jacobian_row[..nfree].copy_from_slice(&projected[..nfree]);
        for r in 0..nfree {
            gradient[r] = g[r];
            for c in 0..=r {
                information[r * nfree + c] = info[r * 6 + c];
                information[c * nfree + r] = info[r * 6 + c];
            }
        }
    }
    Ok((cost, rss, true))
}
