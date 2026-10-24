//! Complete Hilbert-space Fourier frame for several optical carriers.
//!
//! Integer state gauges only change coordinates. Every original Hamiltonian
//! and collapse entry is transformed, including counter-resonant couplings
//! and decay interference. The finite Sambe span supplies uniform unitarity
//! and original-equation residual diagnostics for the caller's refinement and
//! finite-time error checks. No hyperfine manifold is eliminated.
use super::{
    complex_input, coordinate_map, coordinates,
    lapack::{Complex64 as Z, Lapack},
    Model, Operator, C,
};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::time::Instant;

pub type Index = Vec<i32>;
pub type Sparse = Vec<Vec<(usize, f64)>>;
type Fourier = BTreeMap<Index, Vec<C>>;
pub struct Harmonic {
    pub index: Index,
    pub cos: Sparse,
    pub sin: Sparse,
}
pub struct DensityFrame {
    pub model: Model,
    pub harmonics: Vec<Harmonic>,
    pub bounded_generator_tail: f64,
    pub bounded_indices: Vec<Index>,
}
pub struct Frame {
    pub dimension: usize,
    pub frequencies: Vec<f64>,
    pub gauges: Vec<Index>,
    pub order: i32,
    pub modes: Fourier,
    pub hamiltonian: Vec<C>,
    pub collapse: Vec<Fourier>,
    pub unitarity_bound: f64,
    pub schrodinger_residual: f64,
    pub schrodinger_residual_sum: f64,
    pub minimum_central_weight: f64,
    pub sambe_dimension: usize,
    pub preparation_ms: f64,
    pub radiation_diagnostics: Option<crate::radiative_bath::Diagnostics>,
}
#[derive(Clone, Copy, Default)]
struct DD {
    hi: f64,
    lo: f64,
}
impl DD {
    fn add(&mut self, hi: f64, lo: f64) {
        let s = self.hi + hi;
        let z = s - self.hi;
        let e = (self.hi - (s - z)) + (hi - z);
        let low = self.lo + lo + e;
        let value = s + low;
        self.lo = low - (value - s);
        self.hi = value;
    }
    fn product(&mut self, a: f64, b: f64) {
        let p = a * b;
        self.add(p, a.mul_add(b, -p));
    }
    fn value(self) -> f64 {
        self.hi + self.lo
    }
}
#[derive(Clone, Copy, Default)]
struct DDC {
    re: DD,
    im: DD,
}
impl DDC {
    fn product(&mut self, a: C, b: C) {
        self.re.product(a.re, b.re);
        self.re.product(-a.im, b.im);
        self.im.product(a.re, b.im);
        self.im.product(a.im, b.re);
    }
    fn value(self) -> C {
        C::new(self.re.value(), self.im.value())
    }
}
fn plus(a: &[i32], b: &[i32]) -> Index {
    a.iter().zip(b).map(|(a, b)| a + b).collect()
}
fn minus(a: &[i32], b: &[i32]) -> Index {
    a.iter().zip(b).map(|(a, b)| a - b).collect()
}
fn negative(a: &[i32]) -> Index {
    a.iter().map(|a| -a).collect()
}
fn canonical(a: &[i32]) -> bool {
    a.iter().find(|&&v| v != 0).is_some_and(|&v| v > 0)
}
fn dot(index: &[i32], frequencies: &[f64]) -> f64 {
    let mut value = DD::default();
    for (&q, &omega) in index.iter().zip(frequencies) {
        value.product(q as f64, omega);
    }
    value.value()
}
fn adjoint(a: &[C], n: usize) -> Vec<C> {
    (0..n * n).map(|i| a[(i % n) * n + i / n].conj()).collect()
}
fn multiply(a: &[C], b: &[C], n: usize) -> Vec<C> {
    let mut result = vec![C::default(); n * n];
    for i in 0..n {
        for k in 0..n {
            let v = a[i * n + k];
            if v.re == 0. && v.im == 0. {
                continue;
            }
            for j in 0..n {
                result[i * n + j] += v * b[k * n + j];
            }
        }
    }
    result
}
fn norm_one(a: &[C], n: usize) -> f64 {
    (0..n)
        .map(|j| (0..n).map(|i| a[i * n + j].abs()).sum::<f64>())
        .fold(0., f64::max)
}
fn norm_inf(a: &[C], n: usize) -> f64 {
    a.chunks(n)
        .map(|row| row.iter().map(|z| z.abs()).sum::<f64>())
        .fold(0., f64::max)
}
fn frobenius(a: &[C]) -> f64 {
    a.iter().fold(0_f64, |norm, z| norm.hypot(z.abs()))
}
fn add_matrix(target: &mut Fourier, index: Index, matrix: &[C], n: usize) {
    let row = target
        .entry(index)
        .or_insert_with(|| vec![C::default(); n * n]);
    for (a, &b) in row.iter_mut().zip(matrix) {
        *a += b;
    }
}
fn remove_exact_zeros(values: &mut Fourier) {
    values.retain(|_, matrix| matrix.iter().any(|z| z.re != 0. || z.im != 0.));
}
fn phase(index: &[i32], frequencies: &[f64], time: f64) -> C {
    let mut angle = DD::default();
    for (&q, &w) in index.iter().zip(frequencies) {
        let high = w * time;
        let low = w.mul_add(time, -high);
        angle.product(high, q as f64);
        angle.product(low, q as f64);
    }
    let (s, c) = angle.hi.sin_cos();
    let (ds, dc) = angle.lo.sin_cos();
    C::new(c * dc - s * ds, s * dc + c * ds)
}
fn original_hamiltonian(model: &Model) -> Result<Fourier, String> {
    let axes = model.drives.len();
    let n = model.dimension;
    let mut output = Fourier::new();
    output.insert(
        vec![0; axes],
        complex_input(&model.h_real, &model.h_imag, n * n, "multitone H0")?,
    );
    for (axis, drive) in model.drives.iter().enumerate() {
        if !drive.omega.is_finite() || drive.omega <= 0. {
            return Err("Invalid multitone frequency".into());
        }
        let cosine = complex_input(&drive.cos.real, &drive.cos.imag, n * n, "multitone cosine")?;
        let sine = complex_input(&drive.sin.real, &drive.sin.imag, n * n, "multitone sine")?;
        let mut index = vec![0; axes];
        index[axis] = 1;
        output.insert(
            index.clone(),
            cosine
                .iter()
                .zip(&sine)
                .map(|(&a, &b)| C::new((a.re + b.im) * 0.5, (a.im - b.re) * 0.5))
                .collect(),
        );
        output.insert(
            negative(&index),
            cosine
                .iter()
                .zip(&sine)
                .map(|(&a, &b)| C::new((a.re - b.im) * 0.5, (a.im + b.re) * 0.5))
                .collect(),
        );
    }
    remove_exact_zeros(&mut output);
    Ok(output)
}

/// A numerical rotating-coordinate choice from resonant edges of H(t).
/// Edges not chosen here remain in the transformed Hamiltonian unchanged.
/// Conflicting cycles are harmless: their fields retain explicit harmonics.
pub fn resonant_gauges(model: &Model) -> Result<Vec<Index>, String> {
    let n = model.dimension;
    let axes = model.drives.len();
    let frequencies: Vec<_> = model.drives.iter().map(|d| d.omega).collect();
    let original = original_hamiltonian(model)?;
    let zero = vec![0; axes];
    let h0 = original.get(&zero).ok_or("Missing static Hamiltonian")?;
    let mut edges = vec![];
    // A fine-line carrier changes the coordinates of an ENTIRE fine manifold,
    // including hyperfine states that the current polarization does not drive
    // resonantly. Selecting only near-resonant F' edges can leave another F'
    // one complete carrier away and make the numerical normal form singular.
    // This is a gauge choice only: gauged_operator and the raw dipole transform
    // below still retain every Hamiltonian and time-dependent decay entry.
    if axes == 1 && model.states.len() == n {
        let family: Vec<Option<usize>> = model
            .states
            .iter()
            .map(|s| {
                if s.starts_with("ground-") {
                    Some(0)
                } else if s.starts_with("D1-") {
                    Some(1)
                } else if s.starts_with("D2-") {
                    Some(2)
                } else {
                    None
                }
            })
            .collect();
        if family.iter().all(Option::is_some) && (0..3).all(|f| family.contains(&Some(f))) {
            let mut total = [0.; 3];
            let mut count = [0_usize; 3];
            for (i, f) in family.iter().enumerate() {
                let f = f.unwrap();
                total[f] += h0[i * n + i].re;
                count[f] += 1;
            }
            let centers = [
                total[0] / count[0] as f64,
                total[1] / count[1] as f64,
                total[2] / count[2] as f64,
            ];
            let omega = frequencies[0];
            let shifts = [
                0,
                ((centers[0] - centers[1]) / omega).round() as i32,
                ((centers[0] - centers[2]) / omega).round() as i32,
            ];
            let before = centers.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                - centers.iter().copied().fold(f64::INFINITY, f64::min);
            let after: Vec<_> = centers
                .iter()
                .zip(shifts)
                .map(|(&e, q)| e + q as f64 * omega)
                .collect();
            let after = after.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                - after.iter().copied().fold(f64::INFINITY, f64::min);
            if shifts.iter().all(|&q| q.unsigned_abs() <= 1)
                && shifts.iter().any(|&q| q != 0)
                && after < 0.5 * before
            {
                return Ok(family.iter().map(|f| vec![shifts[f.unwrap()]]).collect());
            }
        }
    }
    for (index, h) in &original {
        let frequency = dot(index, &frequencies);
        for i in 0..n {
            for j in i + 1..n {
                let amplitude = h[i * n + j].abs();
                if amplitude == 0. {
                    continue;
                }
                let detuning = (h0[i * n + i].re - h0[j * n + j].re + frequency).abs();
                // This threshold only chooses a gauge, never removes a model entry.
                if detuning <= 1e3_f64.max(10. * amplitude) {
                    edges.push((amplitude / (1. + detuning), i, j, index.clone()));
                }
            }
        }
    }
    edges.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut groups: Vec<_> = (0..n).collect();
    let mut gauges = vec![zero.clone(); n];
    for (_, i, j, index) in edges {
        let left = groups[i];
        let right = groups[j];
        if left == right {
            continue;
        }
        let shift = minus(&minus(&gauges[i], &gauges[j]), &index);
        for k in 0..n {
            if groups[k] == right {
                gauges[k] = plus(&gauges[k], &shift);
                groups[k] = left;
            }
        }
    }
    // Keep a ground F=2 coordinate fixed when its semantic ID is available;
    // otherwise use the most populated input basis state as the anchor.
    for group in groups.iter().copied().collect::<BTreeSet<_>>() {
        let members: Vec<_> = (0..n).filter(|&i| groups[i] == group).collect();
        let anchor = members
            .iter()
            .copied()
            .find(|&i| {
                model
                    .states
                    .get(i)
                    .is_some_and(|s| s.starts_with("g-2-") || s.starts_with("ground-2-"))
            })
            .or_else(|| members.iter().copied().find(|&i| i == model.initial_index))
            .unwrap_or(members[0]);
        let shift = gauges[anchor].clone();
        for i in members {
            gauges[i] = minus(&gauges[i], &shift);
        }
    }
    Ok(gauges)
}

/// Exact one-beat coordinate change when every collapse operator acquires
/// only a channel-wide phase. D[exp(i q omega t) C] = D[C] identically. A
/// repump gauge that phases different decay branches differently is rejected;
/// its radiative coherence transfer must keep explicit Fourier terms.
pub fn single_beat_static_decay_gauge(model: &Model) -> Result<(Model, Vec<i32>), String> {
    if model.drives.len() != 1 {
        return Err("An exact single-beat gauge needs one frequency".into());
    }
    let n = model.dimension;
    let mut gauges = resonant_gauges(model)?;
    // In the atomic protocol, phase a complete fine manifold together,
    // including states that a particular polarization does not drive. This
    // keeps a spherical radiative channel's phase common to every branch.
    // The integer is a frequency-coordinate choice; all H entries survive.
    if model.states.len() == n {
        let groups: Vec<Option<usize>> = model
            .states
            .iter()
            .map(|s| {
                if s.starts_with("ground-") {
                    Some(0)
                } else if s.starts_with("D1-") {
                    Some(1)
                } else if s.starts_with("D2-") {
                    Some(2)
                } else {
                    None
                }
            })
            .collect();
        if groups.iter().all(Option::is_some) && (0..3).all(|group| groups.contains(&Some(group))) {
            let mut total = [0.; 3];
            let mut count = [0_usize; 3];
            for (i, group) in groups.iter().enumerate() {
                let group = group.unwrap();
                total[group] += model.h_real[i * n + i];
                count[group] += 1;
            }
            let centers = [
                total[0] / count[0] as f64,
                total[1] / count[1] as f64,
                total[2] / count[2] as f64,
            ];
            let omega = model.drives[0].omega;
            let shifts = [
                0,
                ((centers[0] - centers[1]) / omega).round() as i32,
                ((centers[0] - centers[2]) / omega).round() as i32,
            ];
            if shifts.iter().all(|&q| q.unsigned_abs() <= 1) {
                gauges = groups.iter().map(|g| vec![shifts[g.unwrap()]]).collect();
            }
        }
    }
    if gauges.iter().all(|q| q[0] == 0) {
        return Err("No useful resonant state gauge was found".into());
    }
    for operator in &model.collapse {
        let c = complex_input(&operator.real, &operator.imag, n * n, "gauged collapse")?;
        let mut channel_phase = None;
        for i in 0..n {
            for j in 0..n {
                let value = c[i * n + j];
                if value.re == 0. && value.im == 0. {
                    continue;
                }
                let q = gauges[j][0] - gauges[i][0];
                if channel_phase.is_some_and(|old| old != q) {
                    return Err("This gauge creates branch-dependent decay phases; retain the full time-dependent dissipator".into());
                }
                channel_phase = Some(q);
            }
        }
    }
    let original = original_hamiltonian(model)?;
    let mut h = gauged_operator(&original, &gauges, n);
    let omega = model.drives[0].omega;
    if h.keys().any(|q| q[0].abs() > 1) {
        return Err("The state gauge requires higher explicit Hamiltonian harmonics".into());
    }
    let mut transformed = model.clone();
    let mut zero = h
        .remove(&vec![0])
        .unwrap_or_else(|| vec![C::default(); n * n]);
    for i in 0..n {
        zero[i * n + i].re += gauges[i][0] as f64 * omega;
    }
    let plus = h
        .remove(&vec![1])
        .unwrap_or_else(|| vec![C::default(); n * n]);
    let minus = h
        .remove(&vec![-1])
        .unwrap_or_else(|| vec![C::default(); n * n]);
    let cosine: Vec<_> = plus.iter().zip(&minus).map(|(&a, &b)| a + b).collect();
    let sine: Vec<_> = plus
        .iter()
        .zip(&minus)
        .map(|(&a, &b)| {
            let difference = a - b;
            C::new(-difference.im, difference.re)
        })
        .collect();
    transformed.h_real = zero.iter().map(|z| z.re).collect();
    transformed.h_imag = zero.iter().map(|z| z.im).collect();
    transformed.drives[0].cos = Operator {
        real: cosine.iter().map(|z| z.re).collect(),
        imag: cosine.iter().map(|z| z.im).collect(),
    };
    transformed.drives[0].sin = Operator {
        real: sine.iter().map(|z| z.re).collect(),
        imag: sine.iter().map(|z| z.im).collect(),
    };
    Ok((transformed, gauges.iter().map(|q| q[0]).collect()))
}

pub fn restore_single_beat_density(rho: &mut [C], n: usize, gauges: &[i32], omega: f64, time: f64) {
    let phases: Vec<_> = gauges
        .iter()
        .map(|&q| phase(&[q], &[omega], time))
        .collect();
    for i in 0..n {
        for j in 0..n {
            if i != j {
                rho[i * n + j] = rho[i * n + j] * phases[i] * phases[j].conj();
            }
        }
    }
}

fn gauged_operator(original: &Fourier, gauges: &[Index], n: usize) -> Fourier {
    let mut result = Fourier::new();
    for (index, matrix) in original {
        for i in 0..n {
            for j in 0..n {
                let value = matrix[i * n + j];
                if value.re == 0. && value.im == 0. {
                    continue;
                }
                let q = plus(index, &minus(&gauges[j], &gauges[i]));
                result.entry(q).or_insert_with(|| vec![C::default(); n * n])[i * n + j] += value;
            }
        }
    }
    result
}

pub fn prepare(lap: &Lapack, model: &Model, order: i32, gauges: &[Index]) -> Result<Frame, String> {
    prepare_with_radiation(lap, model, order, gauges, None)
}
pub fn prepare_with_radiation(
    lap: &Lapack,
    model: &Model,
    order: i32,
    gauges: &[Index],
    radiation: Option<&crate::radiative_bath::Radiation>,
) -> Result<Frame, String> {
    let started = Instant::now();
    let n = model.dimension;
    let axes = model.drives.len();
    let maximum_order = if axes == 1 { 12 } else { 4 };
    if n == 0
        || axes < 1
        || axes > 2
        || order < 1
        || order > maximum_order
        || gauges.len() != n
        || gauges.iter().any(|q| q.len() != axes)
    {
        return Err("Multitone frame requires valid state gauges, one beat at order1..12 or two beats at order1..4".into());
    }
    let frequencies: Vec<_> = model.drives.iter().map(|d| d.omega).collect();
    let zero = vec![0; axes];
    let original = original_hamiltonian(model)?;
    let mut h = gauged_operator(&original, gauges, n);
    let h0 = h
        .entry(zero.clone())
        .or_insert_with(|| vec![C::default(); n * n]);
    for i in 0..n {
        h0[i * n + i].re += dot(&gauges[i], &frequencies);
    }
    let indices: Vec<_> = if axes == 1 {
        (-order..=order).map(|i| vec![i]).collect()
    } else {
        (-order..=order)
            .flat_map(|i| (-order..=order).map(move |j| vec![i, j]))
            .collect()
    };
    let positions: BTreeMap<_, _> = indices
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, q)| (q, i))
        .collect();
    let size = indices.len() * n;
    if size > 3000 {
        return Err("Multitone Hilbert basis exceeds the3000-coordinate preparation budget".into());
    }
    let mut sambe = vec![C::default(); size * size];
    for (row, index) in indices.iter().enumerate() {
        for i in 0..n {
            sambe[row * n + i + (row * n + i) * size].re = dot(index, &frequencies);
        }
        for (q, matrix) in &h {
            if let Some(&col) = positions.get(&minus(index, q)) {
                for i in 0..n {
                    for j in 0..n {
                        sambe[row * n + i + (col * n + j) * size] += matrix[i * n + j];
                    }
                }
            }
        }
    }
    let mut components = vec![];
    let mut seen = vec![false; size];
    for start in 0..size {
        if seen[start] {
            continue;
        }
        seen[start] = true;
        let mut queue = VecDeque::from([start]);
        let mut members = vec![];
        while let Some(col) = queue.pop_front() {
            members.push(col);
            for row in 0..size {
                let value = sambe[row + col * size];
                if !seen[row] && (value.re != 0. || value.im != 0.) {
                    seen[row] = true;
                    queue.push_back(row);
                }
            }
        }
        components.push(members);
    }
    let center = positions[&zero];
    let mut candidates = vec![];
    for members in components {
        let r = members.len();
        let matrix: Vec<_> = (0..r * r)
            .map(|k| {
                let z = sambe[members[k % r] + members[k / r] * size];
                Z { re: z.re, im: z.im }
            })
            .collect();
        let (values, vectors) = lap.hermitian_eigen(&matrix, r)?;
        for (j, &value) in values.iter().enumerate() {
            let weight = members
                .iter()
                .enumerate()
                .filter(|(_, row)| **row >= center * n && **row < (center + 1) * n)
                .map(|(i, _)| {
                    let z = vectors[i + j * r];
                    z.re * z.re + z.im * z.im
                })
                .sum::<f64>();
            if weight == 0. {
                continue;
            }
            let mut v = vec![C::default(); size];
            for (i, &row) in members.iter().enumerate() {
                let z = vectors[i + j * r];
                v[row] = C::new(z.re, z.im);
            }
            candidates.push((weight, value, v));
        }
    }
    candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
    if candidates.len() < n {
        return Err("Not enough independent central Floquet modes".into());
    }
    candidates.truncate(n);
    let minimum_central_weight = candidates.iter().map(|v| v.0).fold(f64::INFINITY, f64::min);
    candidates.sort_by(|a, b| a.1.total_cmp(&b.1));
    let mut modes = Fourier::new();
    for (k, index) in indices.iter().enumerate() {
        let mut matrix = vec![C::default(); n * n];
        for (j, (_, _, v)) in candidates.iter().enumerate() {
            for i in 0..n {
                matrix[i * n + j] = v[k * n + i];
            }
        }
        modes.insert(index.clone(), matrix);
    }
    remove_exact_zeros(&mut modes);
    let mut av = vec![DDC::default(); size * n];
    for (j, (_, _, v)) in candidates.iter().enumerate() {
        for col in 0..size {
            let vc = v[col];
            if vc.re == 0. && vc.im == 0. {
                continue;
            }
            for row in 0..size {
                let a = sambe[row + col * size];
                if a.re != 0. || a.im != 0. {
                    av[row + j * size].product(a, vc);
                }
            }
        }
    }
    let mut hf = vec![C::default(); n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = DDC::default();
            for row in 0..size {
                let u = candidates[i].2[row].conj();
                let v = av[row + j * size];
                sum.product(u, C::new(v.re.hi, v.im.hi));
                sum.product(u, C::new(v.re.lo, v.im.lo));
            }
            hf[i * n + j] = sum.value();
        }
    }
    for i in 0..n {
        hf[i * n + i].im = 0.;
        for j in i + 1..n {
            let value = (hf[i * n + j] + hf[j * n + i].conj()) * 0.5;
            hf[i * n + j] = value;
            hf[j * n + i] = value.conj();
        }
    }
    let mut unitary = Fourier::new();
    for (a, pa) in &modes {
        let left = adjoint(pa, n);
        for (b, pb) in &modes {
            add_matrix(&mut unitary, minus(b, a), &multiply(&left, pb, n), n);
        }
    }
    for i in 0..n {
        unitary.get_mut(&zero).unwrap()[i * n + i].re -= 1.;
    }
    let unitarity_bound = unitary.values().map(|a| frobenius(a)).sum::<f64>();
    if !unitarity_bound.is_finite() || unitarity_bound > 1e-8 {
        return Err(format!("Multitone central span fails uniform unitarity: {unitarity_bound:e}; increase its Fourier order"));
    }
    let mut residual = Fourier::new();
    for (q, p) in &modes {
        let mut value = multiply(p, &hf, n);
        for v in &mut value {
            *v = *v * -1.;
        }
        let frequency = dot(q, &frequencies);
        for (i, &v) in p.iter().enumerate() {
            value[i] += v * frequency;
        }
        add_matrix(&mut residual, q.clone(), &value, n);
    }
    for (q, hq) in &h {
        for (k, p) in &modes {
            add_matrix(&mut residual, plus(q, k), &multiply(hq, p, n), n);
        }
    }
    let schrodinger_residual = residual
        .values()
        .flat_map(|m| m.iter().map(|z| z.abs()))
        .fold(0., f64::max);
    let schrodinger_residual_sum = residual.values().map(|m| frobenius(m)).sum::<f64>();
    let raw_operators = if let Some(radiation) = radiation {
        radiation.normalized_dipoles(n)?
    } else {
        model
            .collapse
            .iter()
            .map(|op| complex_input(&op.real, &op.imag, n * n, "multitone collapse"))
            .collect::<Result<Vec<_>, String>>()?
    };
    let mut collapse = vec![];
    for operator in raw_operators {
        let mut original_c = Fourier::new();
        original_c.insert(zero.clone(), operator);
        let cg = gauged_operator(&original_c, gauges, n);
        let mut transformed = Fourier::new();
        for (a, pa) in &modes {
            let adj = adjoint(pa, n);
            for (q, cq) in &cg {
                let left = multiply(&adj, cq, n);
                for (b, pb) in &modes {
                    add_matrix(
                        &mut transformed,
                        plus(&minus(b, a), q),
                        &multiply(&left, pb, n),
                        n,
                    );
                }
            }
        }
        remove_exact_zeros(&mut transformed);
        collapse.push(transformed);
    }
    let radiation_diagnostics = if let Some(radiation) = radiation {
        let weighted =
            crate::radiative_bath::weight_fourier(lap, radiation, &hf, &frequencies, &collapse)?;
        collapse = weighted.channels;
        Some(weighted.diagnostics)
    } else {
        None
    };
    Ok(Frame {
        dimension: n,
        frequencies,
        gauges: gauges.to_vec(),
        order,
        modes,
        hamiltonian: hf,
        collapse,
        unitarity_bound,
        schrodinger_residual,
        schrodinger_residual_sum,
        minimum_central_weight,
        sambe_dimension: size,
        preparation_ms: started.elapsed().as_secs_f64() * 1000.,
        radiation_diagnostics,
    })
}

impl Frame {
    pub fn at_time(&self, time: f64) -> Vec<C> {
        let mut output = vec![C::default(); self.dimension * self.dimension];
        for (q, p) in &self.modes {
            let factor = phase(q, &self.frequencies, time);
            for (a, &b) in output.iter_mut().zip(p) {
                *a += b * factor;
            }
        }
        output
    }
    pub fn physical_frame(&self, time: f64) -> Vec<C> {
        let mut p = self.at_time(time);
        for (row, g) in self.gauges.iter().enumerate() {
            let factor = phase(g, &self.frequencies, time);
            for v in &mut p[row * self.dimension..(row + 1) * self.dimension] {
                *v = *v * factor;
            }
        }
        p
    }
    pub fn into_frame(&self, rho: &[C]) -> Vec<C> {
        let p = self.at_time(0.);
        multiply(
            &multiply(&adjoint(&p, self.dimension), rho, self.dimension),
            &p,
            self.dimension,
        )
    }
    pub fn physical_density(&self, rho: &[C], time: f64) -> Vec<C> {
        let p = self.physical_frame(time);
        multiply(
            &multiply(&p, rho, self.dimension),
            &adjoint(&p, self.dimension),
            self.dimension,
        )
    }
    pub fn physical_modes(&self) -> Fourier {
        let n = self.dimension;
        let mut result = Fourier::new();
        for (q, p) in &self.modes {
            for row in 0..n {
                let index = plus(q, &self.gauges[row]);
                let target = result
                    .entry(index)
                    .or_insert_with(|| vec![C::default(); n * n]);
                for col in 0..n {
                    target[row * n + col] += p[row * n + col];
                }
            }
        }
        remove_exact_zeros(&mut result);
        result
    }
    pub fn scattering_rate(&self, rho: &[C], time: f64) -> f64 {
        let n = self.dimension;
        let mut result = 0.;
        for channel in &self.collapse {
            let mut c = vec![C::default(); n * n];
            for (q, coefficient) in channel {
                let factor = phase(q, &self.frequencies, time);
                for (a, &b) in c.iter_mut().zip(coefficient) {
                    *a += b * factor;
                }
            }
            let gram = multiply(&adjoint(&c, n), &c, n);
            result += super::scatter_rate(&gram, rho, n);
        }
        result
    }
    pub fn invariant_hilbert_support(&self, initial: &[C]) -> Vec<usize> {
        let n = self.dimension;
        let mut edges = vec![vec![]; n];
        for col in 0..n {
            for row in 0..n {
                let h = self.hamiltonian[row * n + col];
                if h.re != 0. || h.im != 0. {
                    edges[col].push(row);
                }
            }
        }
        for channel in &self.collapse {
            let mut nonzero = vec![false; n * n];
            for c in channel.values() {
                for (i, z) in c.iter().enumerate() {
                    nonzero[i] |= z.re != 0. || z.im != 0.;
                }
            }
            for col in 0..n {
                for row in 0..n {
                    if nonzero[row * n + col] {
                        edges[col].push(row);
                    }
                    if (0..n).any(|k| nonzero[k * n + row] && nonzero[k * n + col]) {
                        edges[col].push(row);
                    }
                }
            }
        }
        let mut support = vec![false; n];
        for i in 0..n {
            for j in 0..n {
                let z = initial[i * n + j];
                if z.re != 0. || z.im != 0. {
                    support[i] = true;
                    support[j] = true;
                }
            }
        }
        let mut queue: VecDeque<_> = (0..n).filter(|&i| support[i]).collect();
        while let Some(i) = queue.pop_front() {
            for &j in &edges[i] {
                if !support[j] {
                    support[j] = true;
                    queue.push_back(j);
                }
            }
        }
        (0..n).filter(|&i| support[i]).collect()
    }
    pub fn generator_indices(&self) -> Vec<Index> {
        let mut indices = BTreeSet::new();
        for channel in &self.collapse {
            for a in channel.keys() {
                for b in channel.keys() {
                    let q = minus(a, b);
                    if canonical(&q) {
                        indices.insert(q);
                    }
                }
            }
        }
        indices.into_iter().collect()
    }
    /// Generate one complete dissipator coefficient on demand. Long-time
    /// solvers can stream even tiny boundary terms without allocating every
    /// dense1024² Fourier matrix at once.
    pub fn generator_harmonic(&self, lap: &Lapack, index: &[i32]) -> Result<Harmonic, String> {
        let n = self.dimension;
        let d = n * n;
        let coords = coordinates(n);
        let map = coordinate_map(&coords, n);
        let pairs: Vec<_> = self
            .collapse
            .iter()
            .flat_map(|channel| {
                channel.iter().filter_map(|(q, a)| {
                    channel
                        .get(&minus(q, index))
                        .map(|b| (a.as_slice(), b.as_slice()))
                })
            })
            .collect();
        let j = super::hilbert_floquet::choi(lap, &pairs, n)?;
        let gram = super::hilbert_floquet::partial_trace(&j, n);
        let mut cosine = vec![0.; d * d];
        let mut sine = vec![0.; d * d];
        let mut imcos = vec![0.; n * d];
        let mut imsin = vec![0.; n * d];
        for b in 0..n {
            for k in 0..n {
                for a in 0..n {
                    for i in 0..n {
                        let v = j[a * n + i + (b * n + k) * d];
                        if v.re != 0. || v.im != 0. {
                            super::hilbert_floquet::cross_term(
                                &mut cosine,
                                &mut sine,
                                &mut imcos,
                                &mut imsin,
                                &map,
                                a * n + b,
                                i * n + k,
                                C::new(v.re, v.im),
                                n,
                            );
                        }
                    }
                }
            }
        }
        for i in 0..n {
            for k in 0..n {
                let value = gram[i * n + k] * -0.5;
                if value.re == 0. && value.im == 0. {
                    continue;
                }
                for col in 0..n {
                    super::hilbert_floquet::cross_term(
                        &mut cosine,
                        &mut sine,
                        &mut imcos,
                        &mut imsin,
                        &map,
                        i * n + col,
                        k * n + col,
                        value,
                        n,
                    );
                    super::hilbert_floquet::cross_term(
                        &mut cosine,
                        &mut sine,
                        &mut imcos,
                        &mut imsin,
                        &map,
                        col * n + k,
                        col * n + i,
                        value,
                        n,
                    );
                }
            }
        }
        let scale = cosine
            .iter()
            .chain(&sine)
            .map(|v| v.abs())
            .fold(1., f64::max);
        if imcos
            .iter()
            .chain(&imsin)
            .any(|v| !v.is_finite() || v.abs() > 1e-10 * scale)
        {
            return Err("Multitone collapse Fourier coefficient breaks Hermiticity".into());
        }
        for matrix in [&cosine, &sine] {
            for col in 0..d {
                let trace: f64 = (0..n).map(|row| matrix[row + col * d]).sum();
                if !trace.is_finite() || trace.abs() > 1e-10 * scale {
                    return Err(
                        "Multitone collapse Fourier coefficient breaks trace preservation".into(),
                    );
                }
            }
        }
        let sparse = |matrix: Vec<f64>| {
            matrix
                .chunks(d)
                .map(|column| {
                    column
                        .iter()
                        .enumerate()
                        .filter_map(|(row, &v)| (v != 0.).then_some((row, v)))
                        .collect()
                })
                .collect()
        };
        Ok(Harmonic {
            index: index.to_vec(),
            cos: sparse(cosine),
            sin: sparse(sine),
        })
    }
    pub fn base_model(&self, original: &Model, initial: &[C]) -> Model {
        let mut model = original.clone();
        let rho = self.into_frame(initial);
        model.h_real = self.hamiltonian.iter().map(|z| z.re).collect();
        model.h_imag = self.hamiltonian.iter().map(|z| z.im).collect();
        model.collapse = self
            .collapse
            .iter()
            .flat_map(|c| c.values())
            .map(|c| Operator {
                real: c.iter().map(|z| z.re).collect(),
                imag: c.iter().map(|z| z.im).collect(),
            })
            .collect();
        model.initial_real = Some(rho.iter().map(|z| z.re).collect());
        model.initial_imag = Some(rho.iter().map(|z| z.im).collect());
        model.drives.clear();
        model.time_dependent = false;
        model
    }
    pub fn density_model(
        &self,
        lap: &Lapack,
        original: &Model,
        initial: &[C],
        tail_budget: f64,
    ) -> Result<DensityFrame, String> {
        if !tail_budget.is_finite() || tail_budget < 0. {
            return Err("Invalid Fourier generator tail budget".into());
        }
        let n = self.dimension;
        let zero = vec![0; self.frequencies.len()];
        let model = self.base_model(original, initial);
        let mut bounds = BTreeMap::<Index, f64>::new();
        for channel in &self.collapse {
            let norms: Vec<_> = channel
                .iter()
                .map(|(q, c)| (q, norm_one(c, n), norm_inf(c, n)))
                .collect();
            for (a, a1, ai) in &norms {
                for (b, b1, bi) in &norms {
                    *bounds.entry(minus(a, b)).or_default() += a1 * b1 + 0.5 * (bi * a1 + ai * b1);
                }
            }
        }
        // ||T||_1<=2 and ||T^-1||_1<=1 convert vectorized complex density
        // coefficients to Hermitian coordinates. Both signed harmonics enter
        // this uniform-in-phase tail. No original collapse coefficient is lost.
        let mut candidates: Vec<_> = bounds
            .iter()
            .filter(|(q, _)| canonical(q))
            .map(|(q, &b)| {
                (
                    q.clone(),
                    2. * (b + bounds.get(&negative(q)).copied().unwrap_or(0.)),
                )
            })
            .collect();
        candidates.sort_by(|a, b| a.1.total_cmp(&b.1));
        let mut bounded_generator_tail = 0.;
        let mut bounded_indices = vec![];
        let mut omitted = BTreeSet::new();
        for (q, bound) in candidates {
            if bounded_generator_tail + bound <= tail_budget {
                bounded_generator_tail += bound;
                bounded_indices.push(q.clone());
                omitted.insert(q);
            }
        }
        let mut harmonics = vec![];
        for index in bounds.keys() {
            if index == &zero || !canonical(index) || omitted.contains(index) {
                continue;
            }
            harmonics.push(self.generator_harmonic(lap, index)?);
        }
        Ok(DensityFrame {
            model,
            harmonics,
            bounded_generator_tail,
            bounded_indices,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::Drive;
    use super::*;
    fn example() -> Model {
        Model { frequency_coordinates:None,
            dimension: 3,
            states: vec![],
            h_real: vec![-7., 0., 0., 0., 0., 0.2, 0., 0.2, 0.4],
            h_imag: vec![0.; 9],
            collapse: vec![Operator {
                real: vec![0., 0., 0.7, 0., 0., 1., 0., 0., 0.],
                imag: vec![0.; 9],
            }],
            initial_index: 1,
            initial_real: None,
            initial_imag: None,
            time_dependent: true,
            drives: vec![
                Drive {
                    omega: 101.,
                    cos: Operator {
                        real: vec![0., 0., 0.03, 0., 0., 0.02, 0.03, 0.02, 0.],
                        imag: vec![0.; 9],
                    },
                    sin: Operator {
                        real: vec![0.; 9],
                        imag: vec![0., 0., 0.03, 0., 0., 0.02, -0.03, -0.02, 0.],
                    },
                },
                Drive {
                    omega: 7.,
                    cos: Operator {
                        real: vec![0., 0., 0.17, 0., 0., 0.1, 0.17, 0.1, 0.],
                        imag: vec![0.; 9],
                    },
                    sin: Operator {
                        real: vec![0.; 9],
                        imag: vec![0., 0., 0.17, 0., 0., 0.1, -0.17, -0.1, 0.],
                    },
                },
            ],
        }
    }
    #[test]
    fn integer_gauge_transforms_every_entry_and_decay_phase() {
        let model = example();
        let gauges = vec![vec![0, 1], vec![0, 0], vec![0, 0]];
        let original = original_hamiltonian(&model).unwrap();
        let transformed = gauged_operator(&original, &gauges, 3);
        for time in [0., 0.03, 0.71] {
            let mut a = vec![C::default(); 9];
            let mut b = vec![C::default(); 9];
            for (q, h) in &original {
                let f = phase(q, &[101., 7.], time);
                for (i, &v) in h.iter().enumerate() {
                    a[i] += v * f;
                }
            }
            for (q, h) in &transformed {
                let f = phase(q, &[101., 7.], time);
                for (i, &v) in h.iter().enumerate() {
                    b[i] += v * f;
                }
            }
            for i in 0..3 {
                for j in 0..3 {
                    let f = phase(&minus(&gauges[j], &gauges[i]), &[101., 7.], time);
                    assert!((a[i * 3 + j] * f - b[i * 3 + j]).abs() < 1e-14);
                }
            }
        }
    }
    #[test]
    fn separated_fine_manifold_gauge_is_exact_and_preserves_cross_line_drives() {
        let omega = 1e5;
        let model = Model { frequency_coordinates:None,
            dimension: 3,
            states: vec!["ground-2-0".into(), "D2-2-0".into(), "D1-2-0".into()],
            h_real: vec![0., 0.4, 0.1, 0.4, 0.2, 0., 0.1, 0., -omega + 0.3],
            h_imag: vec![0.; 9],
            collapse: vec![
                Operator {
                    real: vec![0., 1., 0., 0., 0., 0., 0., 0., 0.],
                    imag: vec![0.; 9],
                },
                Operator {
                    real: vec![0., 0., 0.8, 0., 0., 0., 0., 0., 0.],
                    imag: vec![0.; 9],
                },
            ],
            initial_index: 0,
            initial_real: None,
            initial_imag: None,
            time_dependent: true,
            drives: vec![Drive {
                omega,
                cos: Operator {
                    real: vec![0., 0.05, 0.3, 0.05, 0., 0., 0.3, 0., 0.],
                    imag: vec![0.; 9],
                },
                sin: Operator {
                    real: vec![0.; 9],
                    imag: vec![0., -0.05, -0.3, 0.05, 0., 0., 0.3, 0., 0.],
                },
            }],
        };
        let (transformed, gauges) = single_beat_static_decay_gauge(&model).unwrap();
        assert_eq!(gauges, vec![0, 0, 1]);
        assert!(
            transformed.drives[0].cos.real.iter().any(|v| *v != 0.),
            "Off-resonant cross-line drive remains explicit"
        );
        let before = original_hamiltonian(&model).unwrap();
        let after = original_hamiltonian(&transformed).unwrap();
        for time in [0., 0.000001, 0.13, 0.713] {
            let evaluate = |values: &Fourier| {
                let mut output = vec![C::default(); 9];
                for (q, m) in values {
                    let p = phase(q, &[omega], time);
                    for (i, &v) in m.iter().enumerate() {
                        output[i] += v * p;
                    }
                }
                output
            };
            let original = evaluate(&before);
            let mut expected = original.clone();
            for i in 0..3 {
                for j in 0..3 {
                    expected[i * 3 + j] =
                        original[i * 3 + j] * phase(&[gauges[j] - gauges[i]], &[omega], time);
                }
                expected[i * 3 + i].re += gauges[i] as f64 * omega;
            }
            assert!(expected
                .iter()
                .zip(evaluate(&after))
                .all(|(&a, b)| (a - b).abs() < 2e-12));
        }
        for (a, b) in transformed.collapse.iter().zip(&model.collapse) {
            assert_eq!(a.real, b.real);
            assert_eq!(a.imag, b.imag);
        }
        let lap = Lapack::load().unwrap();
        let frame = super::super::hilbert_floquet::prepare(&lap, &transformed, 2).unwrap();
        assert!(frame.unitarity_bound < 1e-10);
    }
    #[test]
    fn branch_dependent_repump_decay_must_not_use_static_decay_shortcut() {
        let mut model = example();
        model.drives.remove(0);
        assert!(single_beat_static_decay_gauge(&model)
            .err()
            .unwrap()
            .contains("branch-dependent decay"));
    }
    #[test]
    fn fine_carrier_gauge_includes_off_resonant_hyperfine_states_and_common_decay() {
        let n = 4;
        let omega = 1e5;
        let mut h = vec![0.; n * n];
        h[5] = 0.2;
        h[10] = -omega + 0.3;
        h[15] = -omega - 4000.;
        h[1] = 0.4;
        h[4] = 0.4;
        let mut co = vec![0.; n * n];
        co[2] = 0.3;
        co[8] = 0.3;
        co[3] = 0.1;
        co[12] = 0.1;
        let mut si = vec![0.; n * n];
        si[2] = -0.3;
        si[8] = 0.3;
        si[3] = -0.1;
        si[12] = 0.1;
        let mut c = vec![0.; n * n];
        c[1] = 1.;
        c[2] = 0.8;
        c[3] = 0.7;
        let model = Model { frequency_coordinates:None,
            dimension: n,
            states: vec![
                "ground-2-0".into(),
                "D2-2-0".into(),
                "D1-2-0".into(),
                "D1-1-0".into(),
            ],
            h_real: h,
            h_imag: vec![0.; n * n],
            collapse: vec![Operator {
                real: c,
                imag: vec![0.; n * n],
            }],
            initial_index: 0,
            initial_real: None,
            initial_imag: None,
            time_dependent: true,
            drives: vec![Drive {
                omega,
                cos: Operator {
                    real: co,
                    imag: vec![0.; n * n],
                },
                sin: Operator {
                    real: vec![0.; n * n],
                    imag: si,
                },
            }],
        };
        let gauges = resonant_gauges(&model).unwrap();
        assert_eq!(gauges, vec![vec![0], vec![0], vec![1], vec![1]]);
        assert!(
            single_beat_static_decay_gauge(&model).is_err(),
            "Common D1/D2 jump must keep its relative carrier phase"
        );
        let raw = BTreeMap::from([(
            vec![0],
            complex_input(
                &model.collapse[0].real,
                &model.collapse[0].imag,
                n * n,
                "test",
            )
            .unwrap(),
        )]);
        let transformed = gauged_operator(&raw, &gauges, n);
        assert!(transformed.contains_key(&vec![0]));
        assert!(transformed.contains_key(&vec![1]));
        for time in [0., 0.0137, 0.271] {
            let mut actual = vec![C::default(); n * n];
            for (q, v) in &transformed {
                for (a, &b) in actual.iter_mut().zip(v) {
                    *a += b * phase(q, &[omega], time);
                }
            }
            for i in 0..n {
                for j in 0..n {
                    let expected = raw[&vec![0]][i * n + j]
                        * phase(&minus(&gauges[j], &gauges[i]), &[omega], time);
                    assert!((actual[i * n + j] - expected).abs() < 1e-14);
                }
            }
        }
    }
    #[test]
    fn two_frequency_frame_preserves_full_time_dependent_dissipator() {
        let lap = Lapack::load().unwrap();
        let model = example();
        let gauges = vec![vec![0, 1], vec![0, 0], vec![0, 0]];
        let frame = prepare(&lap, &model, 4, &gauges).unwrap();
        assert!(frame.unitarity_bound < 1e-8);
        let original =
            complex_input(&model.collapse[0].real, &model.collapse[0].imag, 9, "test").unwrap();
        let mut initial = vec![C::default(); 9];
        initial[4] = C::new(1., 0.);
        let parts = frame.density_model(&lap, &model, &initial, 0.).unwrap();
        let base = super::super::build_generator(&parts.model).unwrap();
        let x = [0.3, 0.5, 0.2, 0.01, -0.02, 0.03, 0.04, -0.05, 0.06];
        let apply = |a: &Sparse| {
            let mut y = vec![0.; 9];
            for (col, entries) in a.iter().enumerate() {
                for &(row, v) in entries {
                    y[row] += v * x[col];
                }
            }
            y
        };
        for time in [0., 0.03, 0.271, 0.71] {
            let p = frame.physical_frame(time);
            let expected = multiply(&multiply(&adjoint(&p, 3), &original, 3), &p, 3);
            let mut actual = vec![C::default(); 9];
            for (q, c) in &frame.collapse[0] {
                let f = phase(q, &frame.frequencies, time);
                for (i, &v) in c.iter().enumerate() {
                    actual[i] += v * f;
                }
            }
            assert!(actual
                .iter()
                .zip(&expected)
                .all(|(&a, &b)| (a - b).abs() < 1e-12));
            let mut at_time = parts.model.clone();
            at_time.collapse = vec![Operator {
                real: expected.iter().map(|z| z.re).collect(),
                imag: expected.iter().map(|z| z.im).collect(),
            }];
            let derivative = apply(&super::super::build_generator(&at_time).unwrap().matrix);
            let mut reconstructed = apply(&base.matrix);
            for harmonic in &parts.harmonics {
                let f = phase(&harmonic.index, &frame.frequencies, time);
                let c = apply(&harmonic.cos);
                let s = apply(&harmonic.sin);
                for i in 0..9 {
                    reconstructed[i] += f.re * c[i] + f.im * s[i];
                }
            }
            assert!(reconstructed
                .iter()
                .zip(derivative)
                .all(|(a, b)| (a - b).abs() < 1e-11));
        }
    }
    #[test]
    #[ignore = "Generated three-beam full32-state frame and density assembly audit"]
    fn rb_three_beam_frame_audit() {
        let mut raw: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../artifacts/v4/three-beam-master.json"
            ))
            .unwrap(),
        )
        .unwrap();
        raw["states"] = serde_json::json!(raw["states"]
            .as_array()
            .unwrap()
            .iter()
            .map(|state| state
                .as_str()
                .map(str::to_string)
                .unwrap_or_else(|| state["id"].as_str().unwrap().to_string()))
            .collect::<Vec<_>>());
        let model: Model = serde_json::from_value(raw).unwrap();
        let lap = Lapack::load().unwrap();
        let gauges = resonant_gauges(&model).unwrap();
        println!("gauges={gauges:?}");
        let g = super::super::build_generator(&model).unwrap();
        let initial = super::super::expand_density(&g.coords, &g.initial, 32);
        for order in [2, 3] {
            let frame = prepare(&lap, &model, order, &gauges).unwrap();
            let start = Instant::now();
            let parts = frame.density_model(&lap, &model, &initial, 1e-13).unwrap();
            println!(
                "{}",
                serde_json::json!({"order":order,"dimension":frame.sambe_dimension,"frameMs":frame.preparation_ms,"densityMs":start.elapsed().as_secs_f64()*1000.,"unitarity":frame.unitarity_bound,"residual":frame.schrodinger_residual,"residualSum":frame.schrodinger_residual_sum,"centralWeight":frame.minimum_central_weight,"harmonics":parts.harmonics.len(),"tail":parts.bounded_generator_tail})
            );
        }
    }
    #[test]
    #[ignore = "Full32-state two-beat normal-form experiment; not a production acceptance claim"]
    fn rb_three_beam_full_density_experiment() {
        use super::super::{multitone_action as action, Coordinate};
        let started = Instant::now();
        let mut raw: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../artifacts/v4/three-beam-master.json"
            ))
            .unwrap(),
        )
        .unwrap();
        raw["states"] = serde_json::json!(raw["states"]
            .as_array()
            .unwrap()
            .iter()
            .map(|state| state
                .as_str()
                .map(str::to_string)
                .unwrap_or_else(|| state["id"].as_str().unwrap().to_string()))
            .collect::<Vec<_>>());
        let model: Model = serde_json::from_value(raw).unwrap();
        let lap = Lapack::load().unwrap();
        let order = std::env::var("ATOM_MULTITONE_ORDER")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2);
        let g = super::super::build_generator(&model).unwrap();
        let rho0 = super::super::expand_density(&g.coords, &g.initial, 32);
        let gauges = resonant_gauges(&model).unwrap();
        let frame = prepare(&lap, &model, order, &gauges).unwrap();
        let parts = frame.density_model(&lap, &model, &rho0, 1e-13).unwrap();
        let base =
            super::super::hilbert_floquet::build_density_generator(&lap, &parts.model).unwrap();
        let input = action::Input {
            base: base.matrix,
            harmonics: parts
                .harmonics
                .into_iter()
                .map(|h| action::Harmonic {
                    index: h.index,
                    cos: h.cos,
                    sin: h.sin,
                })
                .collect(),
            frequencies: frame.frequencies.clone(),
            bounded_generator_tail: parts.bounded_generator_tail,
        };
        println!(
            "native multitone input: order={order}, frameMs={}, tail={:e}, harmonics={}",
            frame.preparation_ms,
            input.bounded_generator_tail,
            input.harmonics.len()
        );
        let times: Vec<_> = (0..161).map(|i| 20. * i as f64 / 160.).collect();
        if std::env::var_os("ATOM_MULTITONE_EXPORT").is_some() {
            use std::io::Write;
            let folder =
                std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../artifacts/v4"));
            let mut file = std::io::BufWriter::new(
                std::fs::File::create(folder.join(format!("multitone-input-k{order}.bin")))
                    .unwrap(),
            );
            file.write_all(b"ATOMMT01").unwrap();
            for value in [
                input.base.len(),
                times.len(),
                input.frequencies.len(),
                input.harmonics.len(),
            ] {
                file.write_all(&(value as u64).to_le_bytes()).unwrap();
            }
            file.write_all(&input.bounded_generator_tail.to_le_bytes())
                .unwrap();
            for value in &input.frequencies {
                file.write_all(&value.to_le_bytes()).unwrap();
            }
            fn sparse(file: &mut impl Write, matrix: &Sparse) {
                for column in matrix {
                    file.write_all(&(column.len() as u64).to_le_bytes())
                        .unwrap();
                    for &(row, value) in column {
                        file.write_all(&(row as u64).to_le_bytes()).unwrap();
                        file.write_all(&value.to_le_bytes()).unwrap();
                    }
                }
            }
            sparse(&mut file, &input.base);
            for harmonic in &input.harmonics {
                for index in &harmonic.index {
                    file.write_all(&index.to_le_bytes()).unwrap();
                }
                sparse(&mut file, &harmonic.cos);
                sparse(&mut file, &harmonic.sin);
            }
            for value in base.initial.iter().chain(&times) {
                file.write_all(&value.to_le_bytes()).unwrap();
            }
            file.flush().unwrap();
            let modes:Vec<_>=frame.modes.iter().map(|(index,matrix)|serde_json::json!({"index":index,"real":matrix.iter().map(|z|z.re).collect::<Vec<_>>(),"imag":matrix.iter().map(|z|z.im).collect::<Vec<_>>()})).collect();
            std::fs::write(folder.join(format!("multitone-frame-k{order}.json")),serde_json::json!({"dimension":32,"frequencies":frame.frequencies,"gauges":frame.gauges,"modes":modes,"unitarity":frame.unitarity_bound,"schrodingerResidual":frame.schrodinger_residual}).to_string()).unwrap();
            if std::env::var_os("ATOM_MULTITONE_EXPORT_ONLY").is_some() {
                return;
            }
        }
        let prepared = if std::env::var_os("ATOM_MULTITONE_INITIAL_CERT").is_some() {
            action::prepare_for_initial(&lap, &input, &base.initial, 20., 1e-8)
        } else {
            action::prepare(&lap, &input, 20., 1e-8)
        }
        .unwrap();
        println!("native multitone prepared: {:?}", prepared.diagnostics);
        let evolved = prepared.apply(&lap, &base.initial, &times).unwrap();
        let mut output = vec![];
        let mut trace = 0_f64;
        let mut positive = 0_f64;
        let mut initial_error = 0_f64;
        for (t, x) in times.iter().zip(&evolved.states) {
            let sigma = super::super::expand_density(&g.coords, x, 32);
            let rho = frame.physical_density(&sigma, *t);
            let coords: Vec<_> = g
                .coords
                .iter()
                .map(|&coord| match coord {
                    Coordinate::Population(i) => rho[i * 32 + i].re,
                    Coordinate::Real(i, j) => rho[i * 32 + j].re,
                    Coordinate::Imag(i, j) => rho[i * 32 + j].im,
                })
                .collect();
            trace = trace.max((coords[..32].iter().sum::<f64>() - 1.).abs());
            positive = positive.max(super::super::positivity(&lap, &rho, 32).unwrap());
            if *t == 0. {
                initial_error = coords
                    .iter()
                    .zip(&g.initial)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0., f64::max);
            }
            output.push(coords);
        }
        let report = serde_json::json!({"order":order,"timesUs":times,"coordinates":output,"elapsedMs":started.elapsed().as_secs_f64()*1000.,"frameMs":frame.preparation_ms,"frameUniformUnitarity":frame.unitarity_bound,"frameResidual":frame.schrodinger_residual,"frameResidualSum":frame.schrodinger_residual_sum,"prepareMs":prepared.diagnostics.preparation_ms,"applyMs":evolved.elapsed_ms,"fourierErrorBound":evolved.fourier_error_bound,"traceError":trace,"positivityError":positive,"initialError":initial_error,"status":"experimental; frame/modal/propagation floating-point checks are separate from Fourier truncation"});
        let path = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../artifacts/v4"))
            .join(format!("native-three-beam-k{order}.json"));
        std::fs::write(path, report.to_string()).unwrap();
        println!("trace={trace:e}, positivity={positive:e}, initial={initial_error:e}, FourierBound={:e}, totalMs={}",evolved.fourier_error_bound,started.elapsed().as_secs_f64()*1000.);
        assert!(trace.max(positive).max(initial_error) < 1e-8);
    }
}
