//! The built-ins' Python-lane initializers (`model.initializer`,
//! `candidate_initializer`, `bounds_initializer`) with NumPy semantics, and
//! `_initial_values` / `_initial_candidates`.
//!
//! The compiled lane reaches these only for a partial warm start (a
//! mapping that names some parameters); the fallback lane always does.
//! `np.fft.rfft` (the damped sine's frequency seed) is a direct DFT here:
//! only the index of the largest bin is used, which agrees except where two
//! bins' magnitudes tie to rounding.

use super::models::{Axes, Kernel, MAD_TO_SIGMA};
use super::num::{nextafter, numpy_median, numpy_sum, pmax, pmin, str_list_repr, EPSILON, PI, TINY};
use super::prepare::Seeds;
use super::result::{Bounds, FitError, ParamValues};
use super::shared::{histogram_step, min_max, span, unique_sorted};
use super::spec::{Evaluator, FitModelSpec};

fn argmax(values: &[f64]) -> usize {
    let mut best = 0;
    for i in 1..values.len() {
        if values[i] > values[best] || (values[i].is_nan() && !values[best].is_nan()) {
            best = i;
        }
    }
    best
}

fn argmin(values: &[f64]) -> usize {
    let mut best = 0;
    for i in 1..values.len() {
        if values[i] < values[best] || (values[i].is_nan() && !values[best].is_nan()) {
            best = i;
        }
    }
    best
}

/// `_data_interval(values)`.
fn data_interval(values: &[f64]) -> (f64, f64) {
    let (low, high) = min_max(values);
    if low < high {
        return (low, high);
    }
    let padding = EPSILON.sqrt() * pmax(low.abs(), 1.0);
    (low - padding, high + padding)
}

fn value_range(values: &[f64]) -> f64 {
    let (low, high) = data_interval(values);
    high - low
}

fn ptp(values: &[f64]) -> f64 {
    let (low, high) = min_max(values);
    high - low
}

/// `np.argsort(x, kind="stable")`.
fn stable_order(x: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..x.len()).collect();
    order.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap_or_else(|| x[a].is_nan().cmp(&x[b].is_nan())));
    order
}

fn sorted_histogram(x: &[f64], y: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let order = stable_order(x);
    (order.iter().map(|&i| x[i]).collect(), order.iter().map(|&i| if y[i] < 0.0 { 0.0 } else { y[i] }).collect())
}

/// `np.searchsorted(a, v)` (side left).
fn searchsorted(a: &[f64], v: f64) -> usize { a.partition_point(|&e| e < v) }

/// `_histogram_cuts(x, counts)`.
fn histogram_cuts(x: &[f64], counts: &[f64]) -> Vec<f64> {
    let total = numpy_sum(counts);
    let mut cumulative = Vec::with_capacity(counts.len());
    let mut running = 0.0;
    for &c in counts {
        running += c / total;
        cumulative.push(running);
    }
    let mut splits = Vec::new();
    for fraction in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] {
        let index = searchsorted(&cumulative, fraction);
        if index < x.len() {
            splits.push(x[index]);
        }
    }
    let n = counts.len();
    if n >= 1 {
        let mut mass = Vec::with_capacity(n);
        let mut moment = Vec::with_capacity(n);
        let (mut m, mut f) = (0.0, 0.0);
        for i in 0..n {
            m += counts[i];
            f += counts[i] * x[i];
            mass.push(m);
            moment.push(f);
        }
        let last_moment = moment[n - 1];
        let mut between = Vec::with_capacity(n.saturating_sub(1));
        for i in 0..n.saturating_sub(1) {
            let left_mass = if mass[i] >= TINY || mass[i].is_nan() { mass[i] } else { TINY };
            let right = total - mass[i];
            let right_mass = if right >= TINY || right.is_nan() { right } else { TINY };
            let left_mean = moment[i] / left_mass;
            let right_mean = (last_moment - moment[i]) / right_mass;
            let gap = right_mean - left_mean;
            between.push(left_mass * right_mass * (gap * gap));
        }
        if !between.is_empty() && between.iter().any(|v| v.is_finite()) {
            let cleaned: Vec<f64> = between
                .iter()
                .map(|&v| if v.is_nan() { f64::NEG_INFINITY } else if v == f64::INFINITY { f64::MAX } else if v == f64::NEG_INFINITY { -f64::MAX } else { v })
                .collect();
            splits.push(x[argmax(&cleaned)]);
        }
    }
    let mut unique: Vec<f64> = Vec::new();
    for s in splits {
        if !unique.contains(&s) {
            unique.push(s);
        }
    }
    unique
}

fn masked_sum(values: &[f64], mask: &[bool]) -> f64 {
    let selected: Vec<f64> = values.iter().zip(mask).filter(|(_, m)| **m).map(|(&v, _)| v).collect();
    numpy_sum(&selected)
}

fn product_sum(a: &[f64], b: &[f64], mask: &[bool]) -> f64 {
    let selected: Vec<f64> = a.iter().zip(b).zip(mask).filter(|(_, m)| **m).map(|((&p, &q), _)| p * q).collect();
    numpy_sum(&selected)
}

fn bimodal_candidates(x_in: &[f64], y: &[f64]) -> Result<Vec<Vec<f64>>, FitError> {
    let (x, counts) = sorted_histogram(x_in, y);
    let width = span(&x);
    let step = histogram_step(&x);
    let total = numpy_sum(&counts);
    let fallback = || {
        let midpoint = if x.is_empty() {
            0.0
        } else {
            let (low, high) = min_max(&x);
            (low + high) / 2.0
        };
        vec![vec![total * step, midpoint - width / 4.0, width / 10.0, width / 2.0, width / 10.0, 0.5]]
    };
    if x.len() < 3 || total <= 0.0 {
        return Ok(fallback());
    }
    let mut seeds: Vec<Vec<f64>> = Vec::new();
    for split in histogram_cuts(&x, &counts) {
        let left: Vec<bool> = x.iter().map(|&v| v <= split).collect();
        let right: Vec<bool> = left.iter().map(|&l| !l).collect();
        let left_weight = masked_sum(&counts, &left);
        let right_weight = masked_sum(&counts, &right);
        if left_weight <= 0.0 || right_weight <= 0.0 {
            continue;
        }
        let left_center = product_sum(&x, &counts, &left) / left_weight;
        let right_center = product_sum(&x, &counts, &right) / right_weight;
        if !(right_center > left_center) {
            continue;
        }
        let spread = |mask: &[bool], center: f64, weight: f64| {
            let terms: Vec<f64> = x.iter().zip(&counts).zip(mask).filter(|(_, m)| **m).map(|((&xv, &c), _)| c * ((xv - center) * (xv - center))).collect();
            pmax((numpy_sum(&terms) / weight).sqrt(), step)
        };
        seeds.push(vec![total * step, left_center, spread(&left, left_center, left_weight), right_center - left_center, spread(&right, right_center, right_weight), right_weight / total]);
    }
    if seeds.is_empty() {
        return Ok(fallback());
    }
    let axes = Axes::one(&x);
    let mut keyed: Vec<(f64, Vec<f64>)> = Vec::new();
    for seed in seeds {
        let predicted = Kernel::Bimodal.value_jacobian(&axes, &seed, false)?.0;
        let squares: Vec<f64> = predicted.iter().zip(&counts).map(|(p, c)| (p - c) * (p - c)).collect();
        keyed.push((numpy_sum(&squares), seed));
    }
    keyed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    Ok(keyed.into_iter().map(|(_, s)| s).collect())
}

/// Python `_poisson_moments(x, counts, step)`.
fn poisson_moments(x: &[f64], counts: &[f64], step: f64) -> (f64, f64, f64) {
    let total = numpy_sum(counts);
    if total <= 0.0 {
        let mean = numpy_sum(x) / x.len() as f64;
        return (0.0, pmax(mean, 0.25 * step), pmax(span(x) / 6.0, step));
    }
    let mut cumulative = Vec::with_capacity(counts.len());
    let mut running = 0.0;
    for &c in counts {
        running += c;
        cumulative.push(running);
    }
    let quantile = |fraction: f64| x[searchsorted(&cumulative, fraction * total).min(x.len() - 1)];
    let lower = quantile(0.25);
    let upper = quantile(0.75);
    let core: Vec<bool> = x.iter().map(|&v| v >= lower && v <= upper).collect();
    let rate = pmax(product_sum(x, counts, &core) / masked_sum(counts, &core), 0.25 * step);
    let width = (upper - lower) / 1.349;
    let sigma = pmax(pmax(width * width - rate, 0.0).sqrt(), step);
    (total, rate, sigma)
}

fn poisson_seed_distance(x: &[f64], counts: &[f64], seed: &[f64]) -> f64 {
    let (amplitude, rate, sigma, delta_rate, sigma_b, ratio) = (seed[0], seed[1], seed[2], seed[3], seed[4], seed[5]);
    let mut predicted = vec![0.0; counts.len()];
    for (area, component_rate, component_sigma) in [(amplitude * (1.0 - ratio), rate, sigma), (amplitude * ratio, rate + delta_rate, sigma_b)] {
        let variance = component_rate + component_sigma * component_sigma;
        let height = area / ((2.0 * PI).sqrt() * variance.sqrt());
        for (p, &xv) in predicted.iter_mut().zip(x) {
            let d = xv - component_rate;
            *p += height * (-0.5 * (d * d) / variance).exp();
        }
    }
    let squares: Vec<f64> = predicted.iter().zip(counts).map(|(p, c)| (p - c) * (p - c)).collect();
    numpy_sum(&squares)
}

fn init_poisson_bimodal(x_in: &[f64], y: &[f64]) -> Vec<f64> {
    let (x, counts) = sorted_histogram(x_in, y);
    let width = span(&x);
    let step = histogram_step(&x);
    let total = numpy_sum(&counts);
    let fallback = || {
        let midpoint = if x.is_empty() {
            0.0
        } else {
            let (low, high) = min_max(&x);
            (low + high) / 2.0
        };
        vec![total * step, pmax(midpoint - width / 4.0, 0.25 * step), pmax(width / 10.0, step), width / 2.0, pmax(width / 10.0, step), 0.5]
    };
    if x.len() < 3 || total <= 0.0 {
        return fallback();
    }
    let mut best: Option<(f64, Vec<f64>)> = None;
    for split in histogram_cuts(&x, &counts) {
        let left: Vec<bool> = x.iter().map(|&v| v <= split).collect();
        if masked_sum(&counts, &left) <= 0.0 || masked_sum(&counts, &left.iter().map(|&l| !l).collect::<Vec<bool>>()) <= 0.0 {
            continue;
        }
        let pick = |want: bool| -> (Vec<f64>, Vec<f64>) {
            let xs: Vec<f64> = x.iter().zip(&left).filter(|(_, l)| **l == want).map(|(&v, _)| v).collect();
            let cs: Vec<f64> = counts.iter().zip(&left).filter(|(_, l)| **l == want).map(|(&v, _)| v).collect();
            (xs, cs)
        };
        let (lx, lc) = pick(true);
        let (rx, rc) = pick(false);
        let (left_total, left_rate, left_sigma) = poisson_moments(&lx, &lc, step);
        let (right_total, right_rate, right_sigma) = poisson_moments(&rx, &rc, step);
        let _ = left_total;
        if !(right_rate > left_rate) {
            continue;
        }
        let seed = vec![total * step, left_rate, left_sigma, right_rate - left_rate, right_sigma, right_total / total];
        let distance = poisson_seed_distance(&x, &counts, &seed);
        if best.as_ref().is_none_or(|(d, _)| distance < *d) {
            best = Some((distance, seed));
        }
    }
    best.map(|(_, s)| s).unwrap_or_else(fallback)
}

/// `_find_peaks(signal, prominence=, width=)`.
fn find_peaks(values: &[f64], prominence: f64, width: f64) -> (Vec<usize>, Vec<f64>) {
    let steps: Vec<f64> = values.windows(2).map(|w| w[1] - w[0]).collect();
    let changes: Vec<usize> = (0..steps.len()).filter(|&i| steps[i] != 0.0).collect();
    if changes.len() < 2 {
        return (Vec::new(), Vec::new());
    }
    let signs: Vec<bool> = changes.iter().map(|&c| steps[c] > 0.0).collect();
    let mut peaks = Vec::new();
    for r in 0..signs.len() - 1 {
        if signs[r] && !signs[r + 1] {
            peaks.push((changes[r] + 1 + changes[r + 1]) / 2);
        }
    }
    let last = values.len() as isize - 1;
    let mut kept = Vec::new();
    let mut widths = Vec::new();
    for peak in peaks {
        let top = values[peak];
        let (mut left_base, mut index) = (peak as isize, peak as isize);
        let mut left_min = top;
        while index >= 0 && values[index as usize] <= top {
            if values[index as usize] < left_min {
                left_min = values[index as usize];
                left_base = index;
            }
            index -= 1;
        }
        let mut right_base = peak as isize;
        index = peak as isize;
        let mut right_min = top;
        while index <= last && values[index as usize] <= top {
            if values[index as usize] < right_min {
                right_min = values[index as usize];
                right_base = index;
            }
            index += 1;
        }
        let rise = top - pmax(left_min, right_min);
        if rise < prominence {
            continue;
        }
        let level = top - 0.5 * rise;
        let mut i = peak as isize;
        while left_base < i && level < values[i as usize] {
            i -= 1;
        }
        let mut left_crossing = i as f64;
        if values[i as usize] < level {
            left_crossing += (level - values[i as usize]) / (values[i as usize + 1] - values[i as usize]);
        }
        i = peak as isize;
        while i < right_base && level < values[i as usize] {
            i += 1;
        }
        let mut right_crossing = i as f64;
        if values[i as usize] < level {
            right_crossing -= (level - values[i as usize]) / (values[i as usize - 1] - values[i as usize]);
        }
        let extent = right_crossing - left_crossing;
        if extent < width {
            continue;
        }
        kept.push(peak);
        widths.push(extent);
    }
    (kept, widths)
}

fn doublet_candidates(x_in: &[f64], y_in: &[f64]) -> Vec<Vec<f64>> {
    let order = stable_order(x_in);
    let x: Vec<f64> = order.iter().map(|&i| x_in[i]).collect();
    let ordered_y: Vec<f64> = order.iter().map(|&i| y_in[i]).collect();
    let x_span = span(&x);
    let y_range = value_range(&ordered_y);
    let unique = unique_sorted(&x);
    let mut step = if unique.len() > 1 { numpy_median(&unique.windows(2).map(|w| (w[1] - w[0]).abs()).collect::<Vec<f64>>()) } else { x_span };
    step = pmax(step, EPSILON);
    let (ylow, yhigh) = min_max(&ordered_y);
    let mut seeds = Vec::new();
    for sign in [1.0, -1.0] {
        let signed: Vec<f64> = ordered_y.iter().map(|v| sign * v).collect();
        let (peaks, widths) = find_peaks(&signed, y_range / 8.0, 1.0);
        if peaks.is_empty() {
            continue;
        }
        // np.argsort(signed[peaks])[::-1][:4] (quicksort; distinct heights).
        let mut by_height: Vec<usize> = (0..peaks.len()).collect();
        by_height.sort_by(|&a, &b| signed[peaks[a]].partial_cmp(&signed[peaks[b]]).unwrap_or(std::cmp::Ordering::Equal));
        by_height.reverse();
        let strongest: Vec<usize> = by_height.iter().take(4).map(|&i| peaks[i]).collect();
        let width_of = |peak: usize| {
            let index = peaks.iter().position(|&p| p == peak).expect("peak");
            pmax(widths[index] * step, step)
        };
        let first = strongest[0];
        for &second in &strongest {
            seeds.push(vec![(x[first] + x[second]) / 2.0, width_of(first), sign * y_range, if sign > 0.0 { ylow } else { yhigh }, (x[second] - x[first]).abs()]);
        }
    }
    if !seeds.is_empty() {
        return seeds;
    }
    let width = x_span / 8.0;
    vec![
        vec![x[argmax(&ordered_y)], width, y_range, ylow, width * 2.0],
        vec![x[argmin(&ordered_y)], width, -y_range, yhigh, width * 2.0],
    ]
}

fn init_damped_sine(x: &[f64], y: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..x.len()).collect();
    order.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap_or(std::cmp::Ordering::Equal));
    let sx: Vec<f64> = order.iter().map(|&i| x[i]).collect();
    let sy: Vec<f64> = order.iter().map(|&i| y[i]).collect();
    let offset = numpy_sum(&sy) / sy.len() as f64;
    let amplitude = pmax(ptp(&sy) / 2.0, EPSILON);
    let spacing = if sx.len() > 1 { numpy_median(&sx.windows(2).map(|w| w[1] - w[0]).collect::<Vec<f64>>()) } else { 1.0 };
    let n = sy.len();
    let bins = n / 2 + 1;
    let frequency = if bins > 1 {
        let centered: Vec<f64> = sy.iter().map(|v| v - offset).collect();
        let magnitude = |k: usize| {
            let (mut re, mut im) = (0.0, 0.0);
            for (t, &v) in centered.iter().enumerate() {
                let angle = -2.0 * PI * (k * t % n) as f64 / n as f64;
                re += v * angle.cos();
                im += v * angle.sin();
            }
            re.hypot(im)
        };
        let spectrum: Vec<f64> = (1..bins).map(magnitude).collect();
        let k = 1 + argmax(&spectrum);
        let d = pmax(spacing.abs(), EPSILON);
        k as f64 * (1.0 / (n as f64 * d))
    } else {
        1.0 / span(x)
    };
    let frequency = pmax(frequency, EPSILON);
    vec![amplitude, offset, frequency, span(x), 0.0]
}

fn init_exponential(x: &[f64], y: &[f64]) -> Vec<f64> {
    let order = stable_order(x);
    let tail = 1.max(y.len() / 10);
    let selected: Vec<f64> = order[order.len() - tail..].iter().map(|&i| y[i]).collect();
    let offset = numpy_median(&selected);
    let decay_time = span(x) / 3.0;
    let mut amplitude = y[argmin(x)] - offset;
    if amplitude == 0.0 {
        let p = ptp(y);
        amplitude = if p != 0.0 { p } else { 1.0 };
    }
    vec![amplitude, offset, decay_time]
}

fn signal_weights(values: &[f64], offset: f64, sign: f64) -> Vec<f64> {
    let deviation: Vec<f64> = values.iter().map(|v| sign * (v - offset)).collect();
    let absolute: Vec<f64> = values.iter().map(|v| (v - offset).abs()).collect();
    let scale = numpy_median(&absolute) * MAD_TO_SIGMA;
    let floor = scale * (2.0 * (values.len().max(2) as f64).ln()).sqrt();
    let weights: Vec<f64> = deviation.iter().map(|d| pmax_np(d - floor, 0.0)).collect();
    if numpy_sum(&weights) > 0.0 {
        return weights;
    }
    deviation.iter().map(|&d| pmax_np(d, 0.0)).collect()
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

fn sample_step(values: &[f64]) -> f64 {
    let ordered = unique_sorted(values);
    if ordered.len() < 2 {
        return EPSILON;
    }
    numpy_median(&ordered.windows(2).map(|w| w[1] - w[0]).collect::<Vec<f64>>())
}

fn weighted_sum(a: &[f64], w: &[f64]) -> f64 { numpy_sum(&a.iter().zip(w).map(|(x, w)| x * w).collect::<Vec<f64>>()) }

fn radial_seed(x: &[f64], y: &[f64], values: &[f64], sign: f64) -> Vec<f64> {
    let offset = numpy_median(values);
    let weights = signal_weights(values, offset, sign);
    let total = numpy_sum(&weights);
    let (cx, cy, radius);
    if total <= 0.0 {
        cx = numpy_sum(x) / x.len() as f64;
        cy = numpy_sum(y) / y.len() as f64;
        radius = pmax(span(x), span(y)) / 4.0;
    } else {
        cx = weighted_sum(x, &weights) / total;
        cy = weighted_sum(y, &weights) / total;
        let terms: Vec<f64> = (0..x.len()).map(|i| weights[i] * ((x[i] - cx) * (x[i] - cx) + (y[i] - cy) * (y[i] - cy))).collect();
        radius = pmax((numpy_sum(&terms) / total).sqrt(), 0.5 * pmin(sample_step(x), sample_step(y)));
    }
    let (low, high) = min_max(values);
    let mut amplitude = if sign > 0.0 { high - offset } else { low - offset };
    if amplitude == 0.0 {
        amplitude = sign * value_range(values);
    }
    vec![amplitude, offset, radius, cx, cy]
}

fn anisotropic_seed(x: &[f64], y: &[f64], values: &[f64], sign: f64) -> Vec<f64> {
    let offset = numpy_median(values);
    let weights = signal_weights(values, offset, sign);
    let total = numpy_sum(&weights);
    let (cx, cy, mut rx, mut ry);
    if total <= 0.0 {
        cx = numpy_sum(x) / x.len() as f64;
        cy = numpy_sum(y) / y.len() as f64;
        rx = span(x) / 4.0;
        ry = span(y) / 4.0;
    } else {
        cx = weighted_sum(x, &weights) / total;
        cy = weighted_sum(y, &weights) / total;
        let tx: Vec<f64> = (0..x.len()).map(|i| weights[i] * ((x[i] - cx) * (x[i] - cx))).collect();
        let ty: Vec<f64> = (0..y.len()).map(|i| weights[i] * ((y[i] - cy) * (y[i] - cy))).collect();
        rx = (numpy_sum(&tx) / total).sqrt();
        ry = (numpy_sum(&ty) / total).sqrt();
    }
    rx = pmax(rx, 0.5 * sample_step(x));
    ry = pmax(ry, 0.5 * sample_step(y));
    let (low, high) = min_max(values);
    let mut amplitude = if sign > 0.0 { high - offset } else { low - offset };
    if amplitude == 0.0 {
        amplitude = sign * value_range(values);
    }
    vec![amplitude, offset, rx, ry, cx, cy]
}

/// The compiled preparation's seeds (`_saturation_candidates`,
/// `_release_recapture_candidates`) and bounds (`_compiled_bounds`).
fn compiled_prepare(kernel: Kernel, coords: &[Vec<f64>], values: &[f64], rows: usize, lower: &mut [f64], upper: &mut [f64]) -> Result<(Vec<Vec<f64>>, i64), FitError> {
    let p = kernel.parameter_count();
    let mut block = vec![f64::NAN; rows * p];
    let valid = vec![true; values.len()];
    let axes = if coords.len() == 1 { Axes::one(&coords[0]) } else { Axes::two(&coords[0], &coords[1]) };
    let count = {
        let mut seeds = Seeds { data: &mut block, rows, cols: p };
        kernel.prepare(&axes, values, &valid, &mut seeds, lower, upper)?
    };
    Ok(((0..count.max(0) as usize).map(|r| block[r * p..(r + 1) * p].to_vec()).collect(), count))
}

/// `model.initializer(coordinates, observations)` for a built-in, with the
/// model's anchoring applied to the first axis.
pub fn builtin_initializer(model: &FitModelSpec, coords: &[Vec<f64>], values: &[f64]) -> Result<Vec<f64>, FitError> {
    Ok(builtin_candidates(model, coords, values, true)?.remove(0))
}

/// `candidate_initializer` (or the initializer alone when `first_only`).
pub fn builtin_candidates(model: &FitModelSpec, coords: &[Vec<f64>], values: &[f64], first_only: bool) -> Result<Vec<Vec<f64>>, FitError> {
    let Evaluator::Builtin(kernel) = &model.evaluator else { return Err(FitError::runtime("not a built-in model")) };
    let first = model.anchored_axis(&coords[0]);
    let x = &first;
    let candidates = match kernel {
        Kernel::Lorentzian => {
            let width = span(x) / 4.0;
            let range = value_range(values);
            let (low, high) = min_max(values);
            vec![vec![x[argmax(values)], width, range, low], vec![x[argmin(values)], width, -range, high]]
        }
        Kernel::Gaussian => {
            let count = 1.max((values.len() / 10).min(20));
            let mut edge: Vec<f64> = values[..count].to_vec();
            edge.extend_from_slice(&values[values.len() - count..]);
            let offset = numpy_median(&edge);
            let delta: Vec<f64> = values.iter().map(|v| v - offset).collect();
            let index = argmax(&delta.iter().map(|d| d.abs()).collect::<Vec<f64>>());
            let mut amplitude = delta[index];
            if amplitude == 0.0 {
                let p = ptp(values);
                amplitude = if p != 0.0 { p } else { 1.0 };
            }
            vec![vec![amplitude, offset, span(x) / 6.0, x[index]]]
        }
        Kernel::Histogram => {
            let weights: Vec<f64> = values.iter().map(|&v| pmax_np(v, 0.0)).collect();
            let total = numpy_sum(&weights);
            let (center, sigma);
            if total <= 0.0 {
                center = numpy_sum(x) / x.len() as f64;
                sigma = span(x) / 6.0;
            } else {
                center = weighted_sum(x, &weights) / total;
                let terms: Vec<f64> = x.iter().zip(&weights).map(|(&xv, &w)| w * ((xv - center) * (xv - center))).collect();
                sigma = pmax((numpy_sum(&terms) / total).sqrt(), span(x) / 1000.0);
            }
            vec![vec![total * histogram_step(x), center, sigma]]
        }
        Kernel::Bimodal => vec![bimodal_candidates(x, values)?.remove(0)],
        Kernel::Poisson => {
            let (sx, counts) = sorted_histogram(x, values);
            let step = histogram_step(&sx);
            let (total, rate, sigma) = poisson_moments(&sx, &counts, step);
            vec![vec![total * step, rate, sigma]]
        }
        Kernel::PoissonBimodal => vec![init_poisson_bimodal(x, values)],
        Kernel::Doublet => doublet_candidates(x, values),
        Kernel::Damped => {
            let base = init_damped_sine(x, values);
            if first_only {
                vec![base]
            } else {
                [-PI / 2.0, 0.0, PI / 2.0]
                    .iter()
                    .map(|shift| {
                        let mut seed = base.clone();
                        seed[4] = (base[4] + shift + PI).rem_euclid(2.0 * PI) - PI;
                        seed
                    })
                    .collect()
            }
        }
        Kernel::Exponential => {
            let base = init_exponential(x, values);
            if first_only {
                vec![base]
            } else {
                vec![base.clone(), vec![-base[0], base[1], base[2]]]
            }
        }
        Kernel::Saturation => {
            let mut lower = vec![f64::NEG_INFINITY; 3];
            let mut upper = vec![f64::INFINITY; 3];
            let (seeds, count) = compiled_prepare(*kernel, &[x.clone()], values, kernel.max_candidates(), &mut lower, &mut upper)?;
            if count == 0 {
                return Err(FitError::value("saturation fit requires distinct finite coordinates"));
            }
            seeds
        }
        Kernel::ReleaseRecapture => {
            let tiny = nextafter(0.0, 1.0);
            let mut lower = vec![0.0, f64::NEG_INFINITY, tiny, tiny];
            let mut upper = vec![f64::INFINITY; 4];
            let (seeds, count) = compiled_prepare(*kernel, &[x.clone()], values, kernel.max_candidates(), &mut lower, &mut upper)?;
            if count == 0 {
                return Err(FitError::value("release-recapture fit requires finite time and survival data"));
            }
            seeds
        }
        Kernel::Radial => {
            let plus = radial_seed(x, &coords[1], values, 1.0);
            if first_only { vec![plus] } else { vec![plus, radial_seed(x, &coords[1], values, -1.0)] }
        }
        Kernel::Anisotropic => {
            let plus = anisotropic_seed(x, &coords[1], values, 1.0);
            if first_only { vec![plus] } else { vec![plus, anisotropic_seed(x, &coords[1], values, -1.0)] }
        }
    };
    Ok(if first_only { candidates.into_iter().take(1).collect() } else { candidates })
}

/// Whether a built-in declares a `candidate_initializer`.
pub fn has_candidates(model: &FitModelSpec) -> bool {
    match &model.evaluator {
        Evaluator::Builtin(kernel) => matches!(
            kernel,
            Kernel::Lorentzian | Kernel::Doublet | Kernel::Damped | Kernel::Exponential | Kernel::Saturation | Kernel::ReleaseRecapture | Kernel::Radial | Kernel::Anisotropic
        ),
        Evaluator::Custom(custom) => custom.candidates(&[], &[]).is_some(),
    }
}

/// `_compiled_bounds`: a built-in's data-derived bounds.
pub fn compiled_bounds(model: &FitModelSpec, coords: &[Vec<f64>], values: &[f64]) -> Result<Bounds, FitError> {
    let Evaluator::Builtin(kernel) = &model.evaluator else { return Ok(Bounds::default()) };
    let p = model.parameters.len();
    let mut lower = vec![f64::NEG_INFINITY; p];
    let mut upper = vec![f64::INFINITY; p];
    let mut first = vec![model.anchored_axis(&coords[0])];
    first.extend(coords[1..].iter().cloned());
    compiled_prepare(*kernel, &first, values, 0, &mut lower, &mut upper)?;
    Ok(Bounds(
        model
            .parameters
            .iter()
            .enumerate()
            .filter(|(i, _)| lower[*i].is_finite() || upper[*i].is_finite())
            .map(|(i, p)| (p.name.clone(), (if lower[i].is_infinite() { None } else { Some(lower[i]) }, if upper[i].is_infinite() { None } else { Some(upper[i]) })))
            .collect(),
    ))
}

fn model_initializer(model: &FitModelSpec, coords: &[Vec<f64>], values: &[f64]) -> Result<Vec<f64>, FitError> {
    match &model.evaluator {
        Evaluator::Builtin(_) => builtin_initializer(model, coords, values),
        Evaluator::Custom(custom) => {
            let first = model.anchored_axis(&coords[0]);
            let mut axes: Vec<&[f64]> = vec![&first];
            axes.extend(coords[1..].iter().map(|c| c.as_slice()));
            Ok(custom.initializer(&axes, values))
        }
    }
}

/// `_initial_values(model, coordinates, values, initial)`.
pub fn initial_values(model: &FitModelSpec, coords: &[Vec<f64>], values: &[f64], initial: Option<&ParamValues>) -> Result<Vec<f64>, FitError> {
    let names = model.parameter_names();
    let seed = match initial {
        None => model_initializer(model, coords, values)?,
        Some(ParamValues::Named(items)) => {
            let mut unknown: Vec<&str> = items.iter().map(|(n, _)| n.as_str()).filter(|n| !names.contains(n)).collect();
            if !unknown.is_empty() {
                unknown.sort();
                unknown.dedup();
                return Err(FitError::value(format!("initial values name unknown parameters: {}", str_list_repr(&unknown))));
            }
            let mut defaults: Vec<Option<f64>> = if items.len() == names.len() { vec![None; names.len()] } else { model_initializer(model, coords, values)?.into_iter().map(Some).collect() };
            for (name, value) in items {
                if let Some(i) = names.iter().position(|n| n == name) {
                    if i < defaults.len() {
                        defaults[i] = Some(*value);
                    }
                }
            }
            defaults.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect()
        }
        Some(ParamValues::Positional(values)) => values.clone(),
    };
    if seed.len() != names.len() || seed.iter().any(|v| !v.is_finite()) {
        return Err(FitError::value("fit initializer returned invalid parameter values"));
    }
    Ok(seed)
}

/// `_initial_candidates(model, coordinates, values, initial, warm_start)`.
pub fn initial_candidates(model: &FitModelSpec, coords: &[Vec<f64>], values: &[f64], initial: Option<&ParamValues>, warm: Option<&[f64]>) -> Result<Vec<Vec<f64>>, FitError> {
    let dedupe = |seeds: Vec<Vec<f64>>| {
        let mut unique: Vec<Vec<f64>> = Vec::new();
        for seed in seeds {
            if !unique.iter().any(|u| u.iter().zip(&seed).all(|(a, b)| a.to_bits() == b.to_bits())) {
                unique.push(seed);
            }
        }
        unique
    };
    if let Some(warm) = warm {
        let mut seeds = vec![super::shared::warm_seed(model, warm)?];
        seeds.extend(initial_candidates(model, coords, values, initial, None)?);
        return Ok(dedupe(seeds));
    }
    let guess = match initial {
        Some(ParamValues::Named(items)) if items.len() != model.parameters.len() => Some(items.clone()),
        _ => None,
    };
    let candidates_available = match &model.evaluator {
        Evaluator::Builtin(_) => has_candidates(model),
        Evaluator::Custom(custom) => custom.candidates(&[], &[]).is_some() || true,
    };
    if !candidates_available || (initial.is_some() && guess.is_none()) {
        return Ok(vec![initial_values(model, coords, values, initial)?]);
    }
    let candidates = match &model.evaluator {
        Evaluator::Builtin(_) => builtin_candidates(model, coords, values, false)?,
        Evaluator::Custom(custom) => {
            let first = model.anchored_axis(&coords[0]);
            let mut axes: Vec<&[f64]> = vec![&first];
            axes.extend(coords[1..].iter().map(|c| c.as_slice()));
            match custom.candidates(&axes, values) {
                Some(c) => c,
                None => return Ok(vec![initial_values(model, coords, values, initial)?]),
            }
        }
    };
    if candidates.is_empty() {
        return Err(FitError::value("fit initializer returned no candidates"));
    }
    let names = model.parameter_names();
    let mut seeds = Vec::new();
    for candidate in candidates {
        let mut seed = initial_values(model, coords, values, Some(&ParamValues::Positional(candidate)))?;
        if let Some(guess) = &guess {
            let mut merged: Vec<(String, f64)> = names.iter().map(|n| n.to_string()).zip(seed.iter().copied()).collect();
            for (name, value) in guess {
                if let Some(entry) = merged.iter_mut().find(|(n, _)| n == name) {
                    entry.1 = *value;
                } else {
                    merged.push((name.clone(), *value));
                }
            }
            seed = initial_values(model, coords, values, Some(&ParamValues::Named(merged)))?;
        }
        seeds.push(seed);
    }
    Ok(dedupe(seeds))
}
