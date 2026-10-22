//! The built-in models' `_prepare_<model>` callbacks: cold seeds and
//! model-derived bounds for one cell, exactly as `_fit_compiled` writes them.

use super::models::{Axes, Kernel, MAD_TO_SIGMA, SQRT_TWO_PI};
use super::num::{div, nextafter, numba_argsort, numba_mean, numba_median, pmax, pmin, Zr, EPSILON, PI, TINY};
use super::scratch::Scratch;

/// A `rows x parameters` seed block (row-major); zero rows asks for bounds only.
pub struct Seeds<'a> {
    pub data: &'a mut [f64],
    pub rows: usize,
    pub cols: usize,
}

impl Seeds<'_> {
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: f64) { self.data[row * self.cols + col] = value; }
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f64 { self.data[row * self.cols + col] }
}

/// `_compact_observations`: the valid points' coordinates and values.
pub fn compact_observations(coords: &Axes, observations: &[f64], valid: &[bool]) -> (Vec<Scratch<f64>>, Scratch<f64>) {
    let selected = valid.iter().take(observations.len()).filter(|&&v| v).count();
    let mut compact: Vec<Scratch<f64>> = (0..coords.arity).map(|_| Scratch::with_capacity(selected)).collect();
    let mut values = Scratch::with_capacity(selected);
    for point in 0..observations.len() {
        if !valid[point] {
            continue;
        }
        values.push(observations[point]);
        for (axis, column) in compact.iter_mut().enumerate() {
            column.push(coords.at(axis, point));
        }
    }
    (compact, values)
}

/// `_minimum_maximum`.
pub fn minimum_maximum(values: &[f64]) -> (f64, f64) {
    let mut low = values[0];
    let mut high = values[0];
    for &v in &values[1..] {
        low = pmin(low, v);
        high = pmax(high, v);
    }
    if low < high {
        return (low, high);
    }
    let padding = EPSILON.sqrt() * pmax(low.abs(), 1.0);
    (low - padding, high + padding)
}

/// `_array_span`.
pub fn array_span(values: &[f64]) -> f64 {
    let (low, high) = minimum_maximum(values);
    pmax(high - low, EPSILON)
}

pub fn array_argmax(values: &[f64]) -> usize {
    let mut result = 0;
    for index in 1..values.len() {
        if values[index] > values[result] {
            result = index;
        }
    }
    result
}

pub fn array_argmin(values: &[f64]) -> usize {
    let mut result = 0;
    for index in 1..values.len() {
        if values[index] < values[result] {
            result = index;
        }
    }
    result
}

/// `_unique_step`: the median distinct spacing of the sorted values.
///
/// The positive spacings of the sorted values are exactly those between
/// consecutive distinct values (the two zeros are one value; a NaN, sorted
/// last, only ever makes a NaN spacing, which is dropped; an infinity
/// repeated makes a NaN spacing too), so the distinct values are gathered
/// first: a sorted input needs no sort, and a coordinate grid has few.
pub fn unique_step(values: &[f64]) -> f64 {
    let distinct = distinct_ascending(values);
    let differences: Vec<f64> = distinct.windows(2).map(|w| w[1] - w[0]).collect();
    if differences.is_empty() {
        let mut ordered = values.to_vec();
        ordered.sort_unstable_by(super::num::numba_order);
        return array_span(&ordered);
    }
    pmax(numba_median(&differences), EPSILON)
}

/// The distinct non-NaN values, ascending (`-0.0` read as `0.0`).
fn distinct_ascending(values: &[f64]) -> Vec<f64> {
    let normal = |v: f64| if v == 0.0 { 0.0 } else { v };
    if values.windows(2).all(|w| w[0] <= w[1]) {
        let mut out: Vec<f64> = Vec::new();
        for &v in values {
            if out.last() != Some(&v) {
                out.push(normal(v));
            }
        }
        return out;
    }
    // A grid's expanded x: one strictly increasing row, repeated.
    let period = 1 + values.windows(2).take_while(|w| w[0] < w[1]).count();
    if values.len() % period == 0 && values.chunks_exact(period).all(|row| row == &values[..period]) {
        return values[..period].iter().map(|&v| normal(v)).collect();
    }
    let mut seen = std::collections::HashSet::with_hasher(BitsHash);
    let mut out: Vec<f64> = Vec::new();
    for &v in values {
        if !v.is_nan() && seen.insert(normal(v).to_bits()) {
            out.push(normal(v));
        }
    }
    out.sort_unstable_by(|a, b| a.partial_cmp(b).expect("no NaN"));
    out
}

/// A multiplicative hash for float bit patterns.
#[derive(Clone, Copy, Default)]
struct BitsHash;

struct BitsHasher(u64);

impl std::hash::BuildHasher for BitsHash {
    type Hasher = BitsHasher;
    fn build_hasher(&self) -> BitsHasher { BitsHasher(0) }
}

impl std::hash::Hasher for BitsHasher {
    fn finish(&self) -> u64 { self.0 }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0.rotate_left(8) ^ b as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        }
    }
    fn write_u64(&mut self, value: u64) { self.0 = (value ^ (value >> 29)).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (value >> 32); }
}

fn sorted_compact(coords: &Axes, observations: &[f64], valid: &[bool]) -> (Vec<f64>, Vec<f64>) {
    let (compact, raw) = compact_observations(coords, observations, valid);
    let order = numba_argsort(&compact[0]);
    let x: Vec<f64> = order.iter().map(|&i| compact[0][i]).collect();
    let values: Vec<f64> = order.iter().map(|&i| raw[i]).collect();
    (x, values)
}

impl Kernel {
    /// The model's `_prepare_<model>(coords, observations, valid, seeds,
    /// lower, upper, context)`; returns the number of cold seeds written.
    pub fn prepare(self, coords: &Axes, observations: &[f64], valid: &[bool], seeds: &mut Seeds, lower: &mut [f64], upper: &mut [f64]) -> Zr<i64> {
        match self {
            Kernel::Lorentzian => prepare_lorentzian(coords, observations, valid, seeds, lower, upper),
            Kernel::Gaussian => prepare_gaussian(coords, observations, valid, seeds),
            Kernel::Histogram => prepare_histogram(coords, observations, valid, seeds),
            Kernel::Bimodal => prepare_bimodal(coords, observations, valid, seeds),
            Kernel::Poisson => prepare_poisson_histogram(coords, observations, valid, seeds),
            Kernel::PoissonBimodal => prepare_poisson_bimodal(coords, observations, valid, seeds),
            Kernel::Doublet => prepare_doublet(coords, observations, valid, seeds, lower, upper),
            Kernel::Damped => prepare_damped(coords, observations, valid, seeds),
            Kernel::Exponential => prepare_exponential(coords, observations, valid, seeds),
            Kernel::Saturation => prepare_saturation(coords, observations, valid, seeds, lower, upper),
            Kernel::ReleaseRecapture => prepare_release_recapture(coords, observations, valid, seeds),
            Kernel::Radial => prepare_radial(coords, observations, valid, seeds, lower, upper),
            Kernel::Anisotropic => prepare_anisotropic(coords, observations, valid, seeds, lower, upper),
        }
    }
}

fn prepare_lorentzian(coords: &Axes, observations: &[f64], valid: &[bool], seeds: &mut Seeds, lower: &mut [f64], upper: &mut [f64]) -> Zr<i64> {
    let (compact, values) = compact_observations(coords, observations, valid);
    if values.is_empty() || (0 < seeds.rows && seeds.rows < 2) {
        return Ok(0);
    }
    let x = &compact[0];
    let (xlow, xhigh) = minimum_maximum(x);
    let (ylow, yhigh) = minimum_maximum(&values);
    let xspan = pmax(xhigh - xlow, EPSILON);
    let value_range = yhigh - ylow;
    let width = xspan / 4.0;
    lower[0] = pmax(lower[0], xlow);
    upper[0] = pmin(upper[0], xhigh);
    lower[1] = pmax(lower[1], width / 10.0);
    upper[1] = pmin(upper[1], width * 10.0);
    lower[2] = pmax(lower[2], -10.0 * value_range);
    upper[2] = pmin(upper[2], 10.0 * value_range);
    lower[3] = pmax(lower[3], ylow - 10.0 * value_range);
    upper[3] = pmin(upper[3], yhigh + 10.0 * value_range);
    if seeds.rows == 0 {
        return Ok(0);
    }
    seeds.set(0, 0, x[array_argmax(&values)]);
    seeds.set(0, 1, width);
    seeds.set(0, 2, value_range);
    seeds.set(0, 3, ylow);
    seeds.set(1, 0, x[array_argmin(&values)]);
    seeds.set(1, 1, width);
    seeds.set(1, 2, -value_range);
    seeds.set(1, 3, yhigh);
    Ok(2)
}

fn prepare_gaussian(coords: &Axes, observations: &[f64], valid: &[bool], seeds: &mut Seeds) -> Zr<i64> {
    if seeds.rows == 0 {
        return Ok(0);
    }
    let (compact, values) = compact_observations(coords, observations, valid);
    if values.is_empty() {
        return Ok(0);
    }
    let x = &compact[0];
    let count = 1.max((values.len() / 10).min(20));
    let mut edge = vec![0.0; 2 * count];
    for index in 0..count {
        edge[index] = values[index];
        edge[count + index] = values[values.len() - count + index];
    }
    let offset = numba_median(&edge);
    let mut peak = 0;
    let mut largest = (values[0] - offset).abs();
    for (index, &value) in values.iter().enumerate().skip(1) {
        let candidate = (value - offset).abs();
        if candidate > largest {
            largest = candidate;
            peak = index;
        }
    }
    let mut amplitude = values[peak] - offset;
    if amplitude == 0.0 {
        let (low, high) = minimum_maximum(&values);
        amplitude = high - low;
        if amplitude == 0.0 {
            amplitude = 1.0;
        }
    }
    seeds.set(0, 0, amplitude);
    seeds.set(0, 1, offset);
    seeds.set(0, 2, array_span(x) / 6.0);
    seeds.set(0, 3, x[peak]);
    Ok(1)
}

fn prepare_histogram(coords: &Axes, observations: &[f64], valid: &[bool], seeds: &mut Seeds) -> Zr<i64> {
    if seeds.rows == 0 {
        return Ok(0);
    }
    let (compact, values) = compact_observations(coords, observations, valid);
    if values.is_empty() {
        return Ok(0);
    }
    let x = &compact[0];
    let mut total = 0.0;
    let mut first_moment = 0.0;
    for index in 0..values.len() {
        let weight = pmax(values[index], 0.0);
        total += weight;
        first_moment += x[index] * weight;
    }
    let span = array_span(x);
    let (center, sigma);
    if total <= 0.0 {
        center = numba_mean(x);
        sigma = span / 6.0;
    } else {
        center = div(first_moment, total)?;
        let mut variance = 0.0;
        for index in 0..values.len() {
            let weight = pmax(values[index], 0.0);
            let difference = x[index] - center;
            variance += weight * difference * difference;
        }
        sigma = pmax(div(variance, total)?.sqrt(), span / 1000.0);
    }
    let step = unique_step(x);
    seeds.set(0, 0, total * step);
    seeds.set(0, 1, center);
    seeds.set(0, 2, sigma);
    Ok(1)
}

fn bimodal_score(x: &[f64], counts: &[f64], seed: &[f64; 6]) -> Zr<f64> {
    let amplitude = seed[0];
    let center_a = seed[1];
    let sigma_a = seed[2];
    let center_b = seed[1] + seed[3];
    let sigma_b = seed[4];
    let ratio = seed[5];
    let height_a = div(amplitude * (1.0 - ratio), sigma_a * SQRT_TWO_PI)?;
    let height_b = div(amplitude * ratio, sigma_b * SQRT_TWO_PI)?;
    let mut distance = 0.0;
    for index in 0..x.len() {
        let a = div(x[index] - center_a, sigma_a)?;
        let b = div(x[index] - center_b, sigma_b)?;
        let predicted = height_a * (-0.5 * a * a).exp() + height_b * (-0.5 * b * b).exp();
        let difference = predicted - counts[index];
        distance += difference * difference;
    }
    Ok(distance)
}

fn try_bimodal_split(x: &[f64], counts: &[f64], split_value: f64, step: f64, output: &mut [f64; 6]) -> Zr<f64> {
    let mut left_mass = 0.0;
    let mut right_mass = 0.0;
    let mut left_moment = 0.0;
    let mut right_moment = 0.0;
    for index in 0..x.len() {
        let count = pmax(counts[index], 0.0);
        if x[index] <= split_value {
            left_mass += count;
            left_moment += x[index] * count;
        } else {
            right_mass += count;
            right_moment += x[index] * count;
        }
    }
    if left_mass <= 0.0 || right_mass <= 0.0 {
        return Ok(f64::INFINITY);
    }
    let left_center = div(left_moment, left_mass)?;
    let right_center = div(right_moment, right_mass)?;
    if !(right_center > left_center) {
        return Ok(f64::INFINITY);
    }
    let mut left_variance = 0.0;
    let mut right_variance = 0.0;
    for index in 0..x.len() {
        let count = pmax(counts[index], 0.0);
        if x[index] <= split_value {
            let difference = x[index] - left_center;
            left_variance += count * difference * difference;
        } else {
            let difference = x[index] - right_center;
            right_variance += count * difference * difference;
        }
    }
    let total = left_mass + right_mass;
    output[0] = total * step;
    output[1] = left_center;
    output[2] = pmax(div(left_variance, left_mass)?.sqrt(), step);
    output[3] = right_center - left_center;
    output[4] = pmax(div(right_variance, right_mass)?.sqrt(), step);
    output[5] = div(right_mass, total)?;
    bimodal_score(x, counts, output)
}

/// `_two_state_cuts` on a sorted histogram; at most ten cuts.
fn two_state_cuts(x: &[f64], values: &[f64], total: f64, split_values: &mut [f64; 10]) -> Zr<usize> {
    let count = x.len();
    let mut split_count = 0usize;
    let mut cumulative = 0.0;
    let mut target: i64 = 1;
    for index in 0..count {
        cumulative += div(pmax(values[index], 0.0), total)?;
        while target <= 9 && cumulative >= 0.1 * target as f64 {
            let candidate = x[index];
            if !split_values[..split_count].contains(&candidate) {
                split_values[split_count] = candidate;
                split_count += 1;
            }
            target += 1;
        }
    }
    let mut left_mass = 0.0;
    let mut left_moment = 0.0;
    let mut total_moment = 0.0;
    for index in 0..count {
        total_moment += pmax(values[index], 0.0) * x[index];
    }
    let mut best_between = f64::NEG_INFINITY;
    let mut best_index = 0usize;
    for index in 0..count.saturating_sub(1) {
        let value = pmax(values[index], 0.0);
        left_mass += value;
        left_moment += value * x[index];
        let right_mass = total - left_mass;
        let safe_left = pmax(left_mass, TINY);
        let safe_right = pmax(right_mass, TINY);
        let left_mean = div(left_moment, safe_left)?;
        let right_mean = div(total_moment - left_moment, safe_right)?;
        let gap = right_mean - left_mean;
        let between = safe_left * safe_right * (gap * gap);
        if between.is_finite() && between > best_between {
            best_between = between;
            best_index = index;
        }
    }
    if best_between > f64::NEG_INFINITY && split_count < 10 {
        let candidate = x[best_index];
        if !split_values[..split_count].contains(&candidate) {
            split_values[split_count] = candidate;
            split_count += 1;
        }
    }
    Ok(split_count)
}

fn prepare_bimodal(coords: &Axes, observations: &[f64], valid: &[bool], seeds: &mut Seeds) -> Zr<i64> {
    if seeds.rows == 0 {
        return Ok(0);
    }
    let (x, values) = sorted_compact(coords, observations, valid);
    let count = values.len();
    if count == 0 {
        return Ok(0);
    }
    let span = array_span(&x);
    let step = unique_step(&x);
    let mut total = 0.0;
    for &value in &values {
        total += pmax(value, 0.0);
    }
    let mut best = [0.0_f64; 6];
    let mut found = false;
    if count >= 3 && total > 0.0 {
        let mut split_values = [0.0_f64; 10];
        let split_count = two_state_cuts(&x, &values, total, &mut split_values)?;
        let mut trial = [0.0_f64; 6];
        let mut best_score = f64::INFINITY;
        for &split in &split_values[..split_count] {
            let score = try_bimodal_split(&x, &values, split, step, &mut trial)?;
            if score < best_score {
                best_score = score;
                best = trial;
                found = true;
            }
        }
    }
    if !found {
        let midpoint = 0.5 * (x[0] + x[count - 1]);
        best = [total * step, midpoint - span / 4.0, span / 10.0, span / 2.0, span / 10.0, 0.5];
    }
    for (parameter, &value) in best.iter().enumerate() {
        seeds.set(0, parameter, value);
    }
    Ok(1)
}

/// `_poisson_moments(x, values, split, side, step)` on a sorted histogram:
/// `(amplitude, rate, sigma, mass)`.
fn poisson_moments(x: &[f64], values: &[f64], split: f64, side: i64, step: f64) -> Zr<(f64, f64, f64, f64)> {
    let skip = |value: f64| (side < 0 && value > split) || (side > 0 && value <= split);
    let mut total = 0.0;
    for index in 0..x.len() {
        if skip(x[index]) {
            continue;
        }
        total += pmax(values[index], 0.0);
    }
    if total <= 0.0 {
        return Ok((0.0, 0.0, 0.0, 0.0));
    }
    let mut cumulative = 0.0;
    let mut lower = x[0];
    let mut upper = x[0];
    let mut quartile: i64 = 0;
    for index in 0..x.len() {
        if skip(x[index]) {
            continue;
        }
        cumulative += pmax(values[index], 0.0);
        while quartile < 3 && cumulative >= 0.25 * (quartile + 1) as f64 * total {
            if quartile == 0 {
                lower = x[index];
            } else if quartile == 2 {
                upper = x[index];
            }
            quartile += 1;
        }
    }
    let mut core_mass = 0.0;
    let mut core_moment = 0.0;
    for index in 0..x.len() {
        if skip(x[index]) {
            continue;
        }
        if x[index] < lower || x[index] > upper {
            continue;
        }
        let value = pmax(values[index], 0.0);
        core_mass += value;
        core_moment += x[index] * value;
    }
    let rate = pmax(div(core_moment, core_mass)?, 0.25 * step);
    let width = (upper - lower) / 1.349;
    let sigma = pmax(pmax(width * width - rate, 0.0).sqrt(), step);
    Ok((total * step, rate, sigma, total))
}

fn prepare_poisson_histogram(coords: &Axes, observations: &[f64], valid: &[bool], seeds: &mut Seeds) -> Zr<i64> {
    if seeds.rows == 0 {
        return Ok(0);
    }
    let (x, values) = sorted_compact(coords, observations, valid);
    if values.is_empty() {
        return Ok(0);
    }
    let step = unique_step(&x);
    let (_amplitude, mut rate, mut sigma, total) = poisson_moments(&x, &values, 0.0, 0, step)?;
    if total <= 0.0 {
        rate = pmax(numba_mean(&x), 0.25 * step);
        sigma = pmax(array_span(&x) / 6.0, step);
    }
    seeds.set(0, 0, total * step);
    seeds.set(0, 1, rate);
    seeds.set(0, 2, sigma);
    Ok(1)
}

fn poisson_bimodal_score(x: &[f64], counts: &[f64], seed: &[f64; 6]) -> Zr<f64> {
    let amplitude = seed[0];
    let rate_a = seed[1];
    let rate_b = seed[1] + seed[3];
    let ratio = seed[5];
    let variance_a = rate_a + seed[2] * seed[2];
    let variance_b = rate_b + seed[4] * seed[4];
    let height_a = div(amplitude * (1.0 - ratio), SQRT_TWO_PI * variance_a.sqrt())?;
    let height_b = div(amplitude * ratio, SQRT_TWO_PI * variance_b.sqrt())?;
    let mut distance = 0.0;
    for index in 0..x.len() {
        let delta_a = x[index] - rate_a;
        let delta_b = x[index] - rate_b;
        let predicted = height_a * div(-0.5 * delta_a * delta_a, variance_a)?.exp() + height_b * div(-0.5 * delta_b * delta_b, variance_b)?.exp();
        let difference = predicted - counts[index];
        distance += difference * difference;
    }
    Ok(distance)
}

fn try_poisson_split(x: &[f64], counts: &[f64], split_value: f64, step: f64, output: &mut [f64; 6]) -> Zr<f64> {
    let (_la, left_rate, left_sigma, left_mass) = poisson_moments(x, counts, split_value, -1, step)?;
    let (_ra, right_rate, right_sigma, right_mass) = poisson_moments(x, counts, split_value, 1, step)?;
    if left_mass <= 0.0 || right_mass <= 0.0 || !(right_rate > left_rate) {
        return Ok(f64::INFINITY);
    }
    let total = left_mass + right_mass;
    output[0] = total * step;
    output[1] = left_rate;
    output[2] = left_sigma;
    output[3] = right_rate - left_rate;
    output[4] = right_sigma;
    output[5] = div(right_mass, total)?;
    poisson_bimodal_score(x, counts, output)
}

fn prepare_poisson_bimodal(coords: &Axes, observations: &[f64], valid: &[bool], seeds: &mut Seeds) -> Zr<i64> {
    if seeds.rows == 0 {
        return Ok(0);
    }
    let (x, values) = sorted_compact(coords, observations, valid);
    let count = values.len();
    if count == 0 {
        return Ok(0);
    }
    let span = array_span(&x);
    let step = unique_step(&x);
    let mut total = 0.0;
    for &value in &values {
        total += pmax(value, 0.0);
    }
    let mut trial = [0.0_f64; 6];
    let mut best = [0.0_f64; 6];
    let mut found = false;
    if count >= 3 && total > 0.0 {
        let mut split_values = [0.0_f64; 10];
        let split_count = two_state_cuts(&x, &values, total, &mut split_values)?;
        let mut best_score = f64::INFINITY;
        for &split in &split_values[..split_count] {
            let score = try_poisson_split(&x, &values, split, step, &mut trial)?;
            if score < best_score {
                best_score = score;
                best = trial;
                found = true;
            }
        }
    }
    if !found {
        let midpoint = 0.5 * (x[0] + x[count - 1]);
        best = [
            total * step,
            pmax(midpoint - span / 4.0, 0.25 * step),
            pmax(span / 10.0, step),
            span / 2.0,
            pmax(span / 10.0, step),
            0.5,
        ];
    }
    for (parameter, &value) in best.iter().enumerate() {
        seeds.set(0, parameter, value);
    }
    Ok(1)
}

/// `_peak_properties`: peaks at least `threshold` prominent and a sample
/// wide at half prominence; returns the count written.
fn peak_properties(values: &[f64], threshold: f64, peaks: &mut Vec<usize>, widths: &mut Vec<f64>) -> Zr<usize> {
    let size = values.len();
    let mut index = 1usize;
    while index + 1 < size {
        if values[index] > values[index - 1] {
            let mut end = index;
            while end + 1 < size && values[end + 1] == values[index] {
                end += 1;
            }
            if end < size - 1 && values[end] > values[end + 1] {
                let peak = (index + end) / 2;
                let height = values[peak];
                let mut left_minimum = height;
                let mut scan = peak;
                while scan > 0 {
                    scan -= 1;
                    left_minimum = pmin(left_minimum, values[scan]);
                    if values[scan] > height {
                        break;
                    }
                }
                let mut right_minimum = height;
                scan = peak;
                while scan < size - 1 {
                    scan += 1;
                    right_minimum = pmin(right_minimum, values[scan]);
                    if values[scan] > height {
                        break;
                    }
                }
                let prominence = height - pmax(left_minimum, right_minimum);
                if prominence >= threshold {
                    let level = height - 0.5 * prominence;
                    let mut left = peak;
                    while left > 0 && values[left] > level {
                        left -= 1;
                    }
                    let left_position = if values[left + 1] == values[left] {
                        left as f64
                    } else {
                        left as f64 + div(level - values[left], values[left + 1] - values[left])?
                    };
                    let mut right = peak;
                    while right < size - 1 && values[right] > level {
                        right += 1;
                    }
                    let right_position = if values[right - 1] == values[right] {
                        right as f64
                    } else {
                        (right - 1) as f64 + div(level - values[right - 1], values[right] - values[right - 1])?
                    };
                    let width = right_position - left_position;
                    if width >= 1.0 {
                        peaks.push(peak);
                        widths.push(width);
                    }
                }
            }
            index = end + 1;
        } else {
            index += 1;
        }
    }
    Ok(peaks.len())
}

fn append_doublet_sign(x: &[f64], values: &[f64], sign: f64, value_range: f64, step: f64, seeds: &mut Seeds, start: usize) -> Zr<usize> {
    let signed: Vec<f64> = values.iter().map(|&v| sign * v).collect();
    let mut peaks = Vec::new();
    let mut widths = Vec::new();
    let count = peak_properties(&signed, value_range / 8.0, &mut peaks, &mut widths)?;
    if count == 0 {
        return Ok(start);
    }
    let chosen_count = 4.min(count);
    let mut chosen = vec![0usize; chosen_count];
    let mut used = vec![false; count];
    for slot in chosen.iter_mut() {
        let mut best: isize = -1;
        let mut best_value = f64::NEG_INFINITY;
        for candidate in 0..count {
            let value = signed[peaks[candidate]];
            if !used[candidate] && (value > best_value || (value == best_value && candidate as isize > best)) {
                best = candidate as isize;
                best_value = value;
            }
        }
        let best = if best < 0 { count - 1 } else { best as usize };
        used[best] = true;
        *slot = best;
    }
    let first = peaks[chosen[0]];
    let width = pmax(widths[chosen[0]] * step, step);
    let (low, high) = minimum_maximum(values);
    let offset = if sign > 0.0 { low } else { high };
    for (output, &which) in chosen.iter().enumerate() {
        let second = peaks[which];
        let row = start + output;
        if row >= seeds.rows {
            break;
        }
        seeds.set(row, 0, 0.5 * (x[first] + x[second]));
        seeds.set(row, 1, width);
        seeds.set(row, 2, sign * value_range);
        seeds.set(row, 3, offset);
        seeds.set(row, 4, (x[second] - x[first]).abs());
    }
    Ok((start + chosen_count).min(seeds.rows))
}

fn prepare_doublet(coords: &Axes, observations: &[f64], valid: &[bool], seeds: &mut Seeds, lower: &mut [f64], upper: &mut [f64]) -> Zr<i64> {
    let (x, values) = sorted_compact(coords, observations, valid);
    if values.is_empty() {
        return Ok(0);
    }
    let (xlow, xhigh) = minimum_maximum(&x);
    let (ylow, yhigh) = minimum_maximum(&values);
    let xspan = pmax(xhigh - xlow, EPSILON);
    let value_range = yhigh - ylow;
    let step = unique_step(&x);
    let mut scratch = [0.0_f64; 5];
    let requested = seeds.rows;
    let (count, width) = {
        let mut candidates = if requested > 0 {
            Seeds { data: &mut *seeds.data, rows: requested, cols: 5 }
        } else {
            Seeds { data: &mut scratch, rows: 1, cols: 5 }
        };
        let mut count = append_doublet_sign(&x, &values, 1.0, value_range, step, &mut candidates, 0)?;
        if count < candidates.rows {
            count = append_doublet_sign(&x, &values, -1.0, value_range, step, &mut candidates, count)?;
        }
        if count == 0 {
            (0usize, xspan / 8.0)
        } else {
            (count, candidates.get(0, 1))
        }
    };
    let mut count = count as i64;
    if count == 0 && requested > 0 {
        if requested < 2 {
            return Ok(0);
        }
        let high_index = array_argmax(&values);
        let low_index = array_argmin(&values);
        let row0 = [x[high_index], width, value_range, ylow, 2.0 * width];
        let row1 = [x[low_index], width, -value_range, yhigh, 2.0 * width];
        for column in 0..5 {
            seeds.set(0, column, row0[column]);
            seeds.set(1, column, row1[column]);
        }
        count = 2;
    }
    lower[0] = pmax(lower[0], xlow);
    upper[0] = pmin(upper[0], xhigh);
    lower[1] = pmax(lower[1], width / 10.0);
    upper[1] = pmin(upper[1], width * 10.0);
    lower[2] = pmax(lower[2], -10.0 * value_range);
    upper[2] = pmin(upper[2], 10.0 * value_range);
    lower[3] = pmax(lower[3], ylow - 10.0 * value_range);
    upper[3] = pmin(upper[3], yhigh + 10.0 * value_range);
    lower[4] = pmax(lower[4], 0.0);
    upper[4] = pmin(upper[4], 2.0 * xspan);
    Ok(if requested > 0 { count } else { 0 })
}

fn prepare_damped(coords: &Axes, observations: &[f64], valid: &[bool], seeds: &mut Seeds) -> Zr<i64> {
    if seeds.rows < 3 {
        return Ok(0);
    }
    let (x, values) = sorted_compact(coords, observations, valid);
    if values.is_empty() {
        return Ok(0);
    }
    let count = values.len();
    let offset = numba_mean(&values);
    let (low, high) = minimum_maximum(&values);
    let amplitude = pmax(0.5 * (high - low), EPSILON);
    let spacing = if count > 1 { unique_step(&x) } else { 1.0 };
    let mut best_harmonic: i64 = 1;
    let mut best_power = -1.0;
    for harmonic in 1..=(count / 2) as i64 {
        let angle = 2.0 * PI * harmonic as f64 / count as f64;
        let coefficient = 2.0 * angle.cos();
        let mut previous = 0.0;
        let mut before = 0.0;
        for &value in &values {
            let current = (value - offset) + coefficient * previous - before;
            before = previous;
            previous = current;
        }
        let power = previous * previous + before * before - coefficient * previous * before;
        if power > best_power {
            best_power = power;
            best_harmonic = harmonic;
        }
    }
    let mut frequency = if count / 2 >= 1 { div(best_harmonic as f64, count as f64 * spacing)? } else { div(1.0, array_span(&x))? };
    frequency = pmax(frequency, EPSILON);
    let decay = array_span(&x);
    let phases = [-0.5 * PI, 0.0, 0.5 * PI];
    for (seed, &phase) in phases.iter().enumerate() {
        seeds.set(seed, 0, amplitude);
        seeds.set(seed, 1, offset);
        seeds.set(seed, 2, frequency);
        seeds.set(seed, 3, decay);
        seeds.set(seed, 4, phase);
    }
    Ok(3)
}

fn prepare_exponential(coords: &Axes, observations: &[f64], valid: &[bool], seeds: &mut Seeds) -> Zr<i64> {
    if seeds.rows < 2 {
        return Ok(0);
    }
    let (x, values) = sorted_compact(coords, observations, valid);
    if values.is_empty() {
        return Ok(0);
    }
    let tail_count = 1.max(values.len() / 10);
    let offset = numba_median(&values[values.len() - tail_count..]);
    let decay = array_span(&x) / 3.0;
    let mut amplitude = values[0] - offset;
    if amplitude == 0.0 {
        let (low, high) = minimum_maximum(&values);
        amplitude = high - low;
        if amplitude == 0.0 {
            amplitude = 1.0;
        }
    }
    seeds.set(0, 0, amplitude);
    seeds.set(0, 1, offset);
    seeds.set(0, 2, decay);
    seeds.set(1, 0, -amplitude);
    seeds.set(1, 1, offset);
    seeds.set(1, 2, decay);
    Ok(2)
}

/// `_noise_floor`.
fn noise_floor(values: &[f64], offset: f64) -> f64 {
    let deviations = Scratch::collected(values.len(), values.iter().map(|&v| (v - offset).abs()));
    let scale = numba_median(&deviations) * MAD_TO_SIGMA;
    scale * (2.0 * pmax(values.len() as f64, 2.0).ln()).sqrt()
}

/// The median offset and its noise floor, the same for either sign.
fn offset_and_floor(values: &[f64]) -> (f64, f64) {
    let offset = numba_median(values);
    (offset, noise_floor(values, offset))
}

fn radial_seed(compact: &[Scratch<f64>], values: &[f64], (offset, mut floor): (f64, f64), sign: f64, output: &mut [f64]) -> Zr<()> {
    let (x, y) = (&compact[0], &compact[1]);
    let mut total = 0.0;
    let mut x_moment = 0.0;
    let mut y_moment = 0.0;
    for _attempt in 0..2 {
        total = 0.0;
        x_moment = 0.0;
        y_moment = 0.0;
        for index in 0..values.len() {
            let weight = pmax(sign * (values[index] - offset) - floor, 0.0);
            total += weight;
            x_moment += x[index] * weight;
            y_moment += y[index] * weight;
        }
        if total > 0.0 {
            break;
        }
        floor = 0.0;
    }
    let (center_x, center_y, radius);
    if total <= 0.0 {
        center_x = numba_mean(x);
        center_y = numba_mean(y);
        radius = pmax(array_span(x), array_span(y)) / 4.0;
    } else {
        center_x = div(x_moment, total)?;
        center_y = div(y_moment, total)?;
        let mut moment = 0.0;
        for index in 0..values.len() {
            let weight = pmax(sign * (values[index] - offset) - floor, 0.0);
            let delta_x = x[index] - center_x;
            let delta_y = y[index] - center_y;
            moment += weight * (delta_x * delta_x + delta_y * delta_y);
        }
        radius = pmax(div(moment, total)?.sqrt(), EPSILON);
    }
    let (low, high) = minimum_maximum(values);
    let mut amplitude = if sign > 0.0 { high - offset } else { low - offset };
    if amplitude == 0.0 {
        amplitude = sign * (high - low);
    }
    output[0] = amplitude;
    output[1] = offset;
    output[2] = radius;
    output[3] = center_x;
    output[4] = center_y;
    Ok(())
}

#[inline]
fn resolved_width(width: f64, step: f64) -> f64 { pmax(width, 0.5 * step) }

fn prepare_radial(coords: &Axes, observations: &[f64], valid: &[bool], seeds: &mut Seeds, lower: &mut [f64], upper: &mut [f64]) -> Zr<i64> {
    let (compact, values) = compact_observations(coords, observations, valid);
    if values.is_empty() || (0 < seeds.rows && seeds.rows < 2) {
        return Ok(0);
    }
    let mut rows = [[0.0_f64; 5]; 2];
    let centre = offset_and_floor(&values);
    radial_seed(&compact, &values, centre, 1.0, &mut rows[0])?;
    radial_seed(&compact, &values, centre, -1.0, &mut rows[1])?;
    let step = pmin(unique_step(&compact[0]), unique_step(&compact[1]));
    rows[0][2] = resolved_width(rows[0][2], step);
    rows[1][2] = resolved_width(rows[1][2], step);
    let (xlow, xhigh) = minimum_maximum(&compact[0]);
    let (ylow, yhigh) = minimum_maximum(&compact[1]);
    let (low, high) = minimum_maximum(&values);
    let value_range = high - low;
    let radius_low = pmax(pmin(rows[0][2], rows[1][2]) / 10.0, EPSILON);
    let radius_high = pmax(rows[0][2], rows[1][2]) * 10.0;
    lower[0] = pmax(lower[0], -4.0 * value_range);
    upper[0] = pmin(upper[0], 4.0 * value_range);
    lower[1] = pmax(lower[1], low - value_range);
    upper[1] = pmin(upper[1], high + value_range);
    lower[2] = pmax(lower[2], radius_low);
    upper[2] = pmin(upper[2], radius_high);
    lower[3] = pmax(lower[3], xlow);
    upper[3] = pmin(upper[3], xhigh);
    lower[4] = pmax(lower[4], ylow);
    upper[4] = pmin(upper[4], yhigh);
    if seeds.rows == 0 {
        return Ok(0);
    }
    for (row, values) in rows.iter().enumerate() {
        for (column, &value) in values.iter().enumerate() {
            seeds.set(row, column, value);
        }
    }
    Ok(2)
}

fn anisotropic_seed(compact: &[Scratch<f64>], values: &[f64], (offset, mut floor): (f64, f64), sign: f64, output: &mut [f64]) -> Zr<()> {
    let (x, y) = (&compact[0], &compact[1]);
    let mut total = 0.0;
    let mut x_moment = 0.0;
    let mut y_moment = 0.0;
    for _attempt in 0..2 {
        total = 0.0;
        x_moment = 0.0;
        y_moment = 0.0;
        for index in 0..values.len() {
            let weight = pmax(sign * (values[index] - offset) - floor, 0.0);
            total += weight;
            x_moment += x[index] * weight;
            y_moment += y[index] * weight;
        }
        if total > 0.0 {
            break;
        }
        floor = 0.0;
    }
    let (center_x, center_y, mut radius_x, mut radius_y);
    if total <= 0.0 {
        center_x = numba_mean(x);
        center_y = numba_mean(y);
        radius_x = array_span(x) / 4.0;
        radius_y = array_span(y) / 4.0;
    } else {
        center_x = div(x_moment, total)?;
        center_y = div(y_moment, total)?;
        let mut variance_x = 0.0;
        let mut variance_y = 0.0;
        for index in 0..values.len() {
            let weight = pmax(sign * (values[index] - offset) - floor, 0.0);
            let dx = x[index] - center_x;
            let dy = y[index] - center_y;
            variance_x += weight * (dx * dx);
            variance_y += weight * (dy * dy);
        }
        radius_x = div(variance_x, total)?.sqrt();
        radius_y = div(variance_y, total)?.sqrt();
    }
    radius_x = pmax(radius_x, EPSILON);
    radius_y = pmax(radius_y, EPSILON);
    let (low, high) = minimum_maximum(values);
    let mut amplitude = if sign > 0.0 { high - offset } else { low - offset };
    if amplitude == 0.0 {
        amplitude = sign * (high - low);
    }
    output[0] = amplitude;
    output[1] = offset;
    output[2] = radius_x;
    output[3] = radius_y;
    output[4] = center_x;
    output[5] = center_y;
    Ok(())
}

fn prepare_anisotropic(coords: &Axes, observations: &[f64], valid: &[bool], seeds: &mut Seeds, lower: &mut [f64], upper: &mut [f64]) -> Zr<i64> {
    let (compact, values) = compact_observations(coords, observations, valid);
    if values.is_empty() || (0 < seeds.rows && seeds.rows < 2) {
        return Ok(0);
    }
    let mut rows = [[0.0_f64; 6]; 2];
    let centre = offset_and_floor(&values);
    anisotropic_seed(&compact, &values, centre, 1.0, &mut rows[0])?;
    anisotropic_seed(&compact, &values, centre, -1.0, &mut rows[1])?;
    let (step_x, step_y) = (unique_step(&compact[0]), unique_step(&compact[1]));
    for row in rows.iter_mut() {
        row[2] = resolved_width(row[2], step_x);
        row[3] = resolved_width(row[3], step_y);
    }
    let (xlow, xhigh) = minimum_maximum(&compact[0]);
    let (ylow, yhigh) = minimum_maximum(&compact[1]);
    let (low, high) = minimum_maximum(&values);
    let value_range = high - low;
    let radius_x_low = pmin(rows[0][2], rows[1][2]);
    let radius_x_high = pmax(rows[0][2], rows[1][2]);
    let radius_y_low = pmin(rows[0][3], rows[1][3]);
    let radius_y_high = pmax(rows[0][3], rows[1][3]);
    lower[0] = pmax(lower[0], -4.0 * value_range);
    upper[0] = pmin(upper[0], 4.0 * value_range);
    lower[1] = pmax(lower[1], low - value_range);
    upper[1] = pmin(upper[1], high + value_range);
    lower[2] = pmax(lower[2], pmax(radius_x_low / 10.0, EPSILON));
    upper[2] = pmin(upper[2], radius_x_high * 10.0);
    lower[3] = pmax(lower[3], pmax(radius_y_low / 10.0, EPSILON));
    upper[3] = pmin(upper[3], radius_y_high * 10.0);
    lower[4] = pmax(lower[4], xlow);
    upper[4] = pmin(upper[4], xhigh);
    lower[5] = pmax(lower[5], ylow);
    upper[5] = pmin(upper[5], yhigh);
    if seeds.rows == 0 {
        return Ok(0);
    }
    for (row, values) in rows.iter().enumerate() {
        for (column, &value) in values.iter().enumerate() {
            seeds.set(row, column, value);
        }
    }
    Ok(2)
}

fn prepare_release_recapture(coords: &Axes, observations: &[f64], valid: &[bool], seeds: &mut Seeds) -> Zr<i64> {
    if seeds.rows == 0 {
        return Ok(0);
    }
    let mut low = f64::INFINITY;
    let mut high = f64::NEG_INFINITY;
    for point in 0..observations.len() {
        if valid[point] {
            low = pmin(low, observations[point]);
            high = pmax(high, observations[point]);
        }
    }
    if !low.is_finite() {
        return Ok(0);
    }
    let amplitude = pmax(high - low, EPSILON);
    let mut best: isize = -1;
    let mut distance = f64::INFINITY;
    for point in 0..observations.len() {
        if valid[point] && coords.at(0, point) != 0.0 {
            let normalized = div(observations[point] - low, amplitude)?;
            if (normalized - 0.5).abs() < distance {
                distance = (normalized - 0.5).abs();
                best = point as isize;
            }
        }
    }
    if best < 0 {
        return Ok(0);
    }
    let best = best as usize;
    let survival = pmin(pmax(div(observations[best] - low, amplitude)?, 1.0e-6), 1.0 - 1.0e-6);
    let time = coords.at(0, best).abs();
    for (index, &eta) in [0.3_f64, 1.0, 5.0, 20.0].iter().enumerate() {
        let q = div(-(survival * (-eta).exp_m1()).ln_1p(), eta)?;
        seeds.set(index, 0, amplitude);
        seeds.set(index, 1, low);
        seeds.set(index, 2, eta);
        seeds.set(index, 3, div(div(-q.ln(), q)?.sqrt(), 2.0 * PI * time)?);
    }
    Ok(4)
}

fn prepare_saturation(coords: &Axes, observations: &[f64], valid: &[bool], seeds: &mut Seeds, lower: &mut [f64], upper: &mut [f64]) -> Zr<i64> {
    let mut low = f64::INFINITY;
    let mut high = f64::NEG_INFINITY;
    let mut count: i64 = 0;
    let mut mean_y = 0.0;
    for point in 0..observations.len() {
        if valid[point] {
            let x = coords.at(0, point);
            low = pmin(low, x);
            high = pmax(high, x);
            count += 1;
            if seeds.rows > 0 {
                mean_y += div(observations[point] - mean_y, count as f64)?;
            }
        }
    }
    if count < 2 || !(high > low) {
        return Ok(0);
    }
    lower[2] = pmax(lower[2], nextafter(-low, f64::INFINITY));
    if seeds.rows == 0 {
        return Ok(0);
    }
    let span = high - low;
    for (index, &factor) in [0.1, 1.0, 10.0].iter().enumerate() {
        let shift = pmax(lower[2], pmin(upper[2], span * factor - low));
        if !(low + shift > 0.0) {
            return Ok(0);
        }
        let mut mean_fraction = 0.0;
        for point in 0..observations.len() {
            if valid[point] {
                mean_fraction += div(div(span, coords.at(0, point) + shift)?, count as f64)?;
            }
        }
        let mut variance = 0.0;
        let mut covariance = 0.0;
        for point in 0..observations.len() {
            if valid[point] {
                let centered = div(span, coords.at(0, point) + shift)? - mean_fraction;
                variance += centered * centered;
                covariance += centered * (observations[point] - mean_y);
            }
        }
        let slope = if variance > 0.0 { div(covariance, variance)? } else { 0.0 };
        let asymptote = mean_y - slope * mean_fraction;
        seeds.set(index, 0, asymptote);
        seeds.set(index, 1, asymptote * shift + slope * span);
        seeds.set(index, 2, shift);
    }
    Ok(3)
}
