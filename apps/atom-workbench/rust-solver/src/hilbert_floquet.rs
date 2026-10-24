//! Periodic, nearly unitary change of basis for a rapidly driven Hamiltonian.
//!
//! This module does NOT average or discard the transformed dissipator. Every
//! Fourier coefficient of P(t)^dagger C P(t) is returned to the density solver.
//! The isolated central Sambe subspace is projected as a whole: retaining its
//! complete small Hermitian Hamiltonian avoids inaccurate individual slow
//! quasienergies in a spectrum containing optical beat frequencies.
use super::{
    complex_input, coordinate_map, coordinates,
    lapack::{Complex64, Lapack},
    periodic_action::ObservableHarmonic,
    CoordMap, Coordinate, Model, Operator, C,
};
use std::collections::VecDeque;
use std::time::Instant;

#[derive(Clone)]
pub(super) struct CollapseFourier {
    pub index: i32,
    pub matrix: Vec<C>,
}
pub(super) struct DissipativeHarmonic {
    pub index: i32,
    pub cos: Vec<Vec<(usize, f64)>>,
    pub sin: Vec<Vec<(usize, f64)>>,
}
pub(super) struct DensityFrame {
    pub model: Model,
    pub harmonics: Vec<DissipativeHarmonic>,
}
pub(super) struct Frame {
    pub dimension: usize,
    pub omega: f64,
    pub order: i32,
    /// Row-major P_k, in increasing harmonic index from -order to +order.
    pub modes: Vec<Vec<C>>,
    pub hamiltonian: Vec<C>,
    pub collapse: Vec<Vec<CollapseFourier>>,
    pub unitarity_bound: f64,
    pub schrodinger_residual: f64,
    pub spectral_gap: f64,
    pub preparation_ms: f64,
    pub radiation_diagnostics: Option<crate::radiative_bath::Diagnostics>,
}
#[derive(Default, Clone, Copy)]
struct DD {
    hi: f64,
    lo: f64,
}
impl DD {
    fn add(&mut self, hi: f64, lo: f64) {
        let s = self.hi + hi;
        let z = s - self.hi;
        let e = (self.hi - (s - z)) + (hi - z);
        let tail = self.lo + lo + e;
        let out = s + tail;
        self.lo = tail - (out - s);
        self.hi = out;
    }
    fn product(&mut self, a: f64, b: f64) {
        let p = a * b;
        self.add(p, a.mul_add(b, -p));
    }
    fn value(self) -> f64 {
        self.hi + self.lo
    }
}
#[derive(Default, Clone, Copy)]
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
fn multiply(a: &[C], b: &[C], n: usize) -> Vec<C> {
    let mut out = vec![C::default(); n * n];
    for i in 0..n {
        for k in 0..n {
            let v = a[i * n + k];
            if v.re == 0. && v.im == 0. {
                continue;
            }
            for j in 0..n {
                out[i * n + j] += v * b[k * n + j];
            }
        }
    }
    out
}
fn adjoint(a: &[C], n: usize) -> Vec<C> {
    (0..n * n)
        .map(|index| a[(index % n) * n + index / n].conj())
        .collect()
}
fn max_matrix(a: &[C]) -> f64 {
    a.iter().map(|z| z.abs()).fold(0., f64::max)
}
fn time_phase(omega: f64, time: f64, index: i32) -> C {
    let main = omega * time;
    let low = omega.mul_add(time, -main);
    let q = index as f64;
    let angle = main * q;
    let correction = main.mul_add(q, -angle) + low * q;
    let (s, c) = angle.sin_cos();
    let (ds, dc) = correction.sin_cos();
    C::new(c * dc - s * ds, s * dc + c * ds)
}

impl Frame {
    pub fn at_phase(&self, phase: f64) -> Vec<C> {
        let n = self.dimension;
        let mut out = vec![C::default(); n * n];
        for (index, mode) in self.modes.iter().enumerate() {
            let harmonic = index as i32 - self.order;
            let (s, c) = (harmonic as f64 * phase).sin_cos();
            let ph = C::new(c, s);
            for (i, &v) in mode.iter().enumerate() {
                out[i] += v * ph;
            }
        }
        out
    }
    pub fn into_frame(&self, rho: &[C]) -> Vec<C> {
        let p = self.at_phase(0.);
        multiply(
            &multiply(&adjoint(&p, self.dimension), rho, self.dimension),
            &p,
            self.dimension,
        )
    }
    pub fn at_time(&self, t: f64) -> Vec<C> {
        let mut p = vec![C::default(); self.dimension * self.dimension];
        for (index, mode) in self.modes.iter().enumerate() {
            let phase = time_phase(self.omega, t, index as i32 - self.order);
            for (i, &v) in mode.iter().enumerate() {
                p[i] += v * phase;
            }
        }
        p
    }
    pub fn physical_density(&self, rho: &[C], t: f64) -> Vec<C> {
        let p = self.at_time(t);
        multiply(
            &multiply(&p, rho, self.dimension),
            &adjoint(&p, self.dimension),
            self.dimension,
        )
    }
    /// Linear projections of the complete transformed density onto physical
    /// populations and total photon scattering. This changes only the output
    /// contraction: all density coherences remain in the propagated state.
    pub fn observable_fourier(&self, scattering: &[C]) -> Result<Vec<ObservableHarmonic>, String> {
        let n = self.dimension;
        let rows = n + 1;
        let d = n * n;
        let blocks = self.modes.len();
        let coords = coordinates(n);
        if scattering.len() != d {
            return Err("Scattering observable has the wrong dimensions".into());
        }
        let mut result: Vec<_> = (0..2 * blocks - 1)
            .map(|k| ObservableHarmonic {
                index: k as i32 - (blocks - 1) as i32,
                rows,
                matrix: vec![Complex64::default(); rows * d],
            })
            .collect();
        for (a, pa) in self.modes.iter().enumerate() {
            let scattering_pa = multiply(scattering, pa, n);
            for (b, pb) in self.modes.iter().enumerate() {
                let target = &mut result[a + blocks - 1 - b].matrix;
                let scattering_frame = multiply(&adjoint(pb, n), &scattering_pa, n);
                for row in 0..rows {
                    let entry = |i: usize, j: usize| {
                        if row < n {
                            pb[row * n + i].conj() * pa[row * n + j]
                        } else {
                            scattering_frame[i * n + j]
                        }
                    };
                    for (col, &coord) in coords.iter().enumerate() {
                        let value = match coord {
                            Coordinate::Population(i) => entry(i, i),
                            Coordinate::Real(i, j) => entry(j, i) + entry(i, j),
                            Coordinate::Imag(i, j) => {
                                let difference = entry(j, i) - entry(i, j);
                                C::new(-difference.im, difference.re)
                            }
                        };
                        target[row + col * rows].re += value.re;
                        target[row + col * rows].im += value.im;
                    }
                }
            }
        }
        Ok(result)
    }
    pub fn scattering_fourier(&self) -> std::collections::BTreeMap<i32, Vec<C>> {
        let n = self.dimension;
        let mut result = std::collections::BTreeMap::<i32, Vec<C>>::new();
        for channel in &self.collapse {
            for a in channel {
                let adj = adjoint(&a.matrix, n);
                for b in channel {
                    let product = multiply(&adj, &b.matrix, n);
                    let target = result
                        .entry(b.index - a.index)
                        .or_insert_with(|| vec![C::default(); n * n]);
                    for (v, &term) in target.iter_mut().zip(&product) {
                        *v += term;
                    }
                }
            }
        }
        result
    }
    /// Photon counting uses the ACTUAL dressed C(t), including Fourier cross
    /// terms. A bare-atom constant Gamma operator is not valid for this bath.
    pub fn observable_fourier_with_scattering(&self) -> Result<Vec<ObservableHarmonic>, String> {
        let n = self.dimension;
        let rows = n + 1;
        let mut result: std::collections::BTreeMap<_, _> = self
            .observable_fourier(&vec![C::default(); n * n])?
            .into_iter()
            .map(|o| (o.index, o))
            .collect();
        let coords = coordinates(n);
        for (index, operator) in self.scattering_fourier() {
            let target = result.entry(index).or_insert_with(|| ObservableHarmonic {
                index,
                rows,
                matrix: vec![Complex64::default(); rows * n * n],
            });
            for (col, &coordinate) in coords.iter().enumerate() {
                let value = match coordinate {
                    Coordinate::Population(i) => operator[i * n + i],
                    Coordinate::Real(i, j) => operator[j * n + i] + operator[i * n + j],
                    Coordinate::Imag(i, j) => {
                        let z = operator[j * n + i] - operator[i * n + j];
                        C::new(-z.im, z.re)
                    }
                };
                target.matrix[n + col * rows].re += value.re;
                target.matrix[n + col * rows].im += value.im;
            }
        }
        Ok(result.into_values().collect())
    }
    pub fn scattering_rate(&self, rho: &[C], time: f64) -> f64 {
        let n = self.dimension;
        let mut rate = 0.;
        for channel in &self.collapse {
            let mut operator = vec![C::default(); n * n];
            for coefficient in channel {
                let phase = time_phase(self.omega, time, coefficient.index);
                for (a, &b) in operator.iter_mut().zip(&coefficient.matrix) {
                    *a += b * phase;
                }
            }
            let gram = multiply(&adjoint(&operator, n), &operator, n);
            rate += super::scatter_rate(&gram, rho, n);
        }
        rate
    }
    pub fn density_model(
        &self,
        lap: &Lapack,
        original: &Model,
        initial: &[C],
    ) -> Result<DensityFrame, String> {
        let n = self.dimension;
        let mut model = original.clone();
        let rho = self.into_frame(initial);
        model.h_real = self.hamiltonian.iter().map(|z| z.re).collect();
        model.h_imag = self.hamiltonian.iter().map(|z| z.im).collect();
        model.collapse = self
            .collapse
            .iter()
            .flatten()
            .map(|c| Operator {
                real: c.matrix.iter().map(|z| z.re).collect(),
                imag: c.matrix.iter().map(|z| z.im).collect(),
            })
            .collect();
        model.initial_real = Some(rho.iter().map(|z| z.re).collect());
        model.initial_imag = Some(rho.iter().map(|z| z.im).collect());
        model.drives.clear();
        model.time_dependent = false;
        let harmonics = dissipative_harmonics(lap, &self.collapse, n)?;
        Ok(DensityFrame { model, harmonics })
    }
}

pub(super) fn choi(
    lap: &Lapack,
    pairs: &[(&[C], &[C])],
    n: usize,
) -> Result<Vec<Complex64>, String> {
    let d = n * n;
    let count = pairs.len();
    if count == 0 {
        return Ok(vec![Complex64 { re: 0., im: 0. }; d * d]);
    }
    let mut a = vec![Complex64 { re: 0., im: 0. }; d * count];
    let mut b = vec![Complex64 { re: 0., im: 0. }; d * count];
    for (p, &(first, second)) in pairs.iter().enumerate() {
        for i in 0..d {
            a[i + p * d] = Complex64 {
                re: first[i].re,
                im: first[i].im,
            };
            b[p + i * count] = Complex64 {
                re: second[i].re,
                im: -second[i].im,
            };
        }
    }
    // J = sum vec(A) vec(B)^dagger. BLAS accumulates the complete cross
    // convolution before reshuffling J into a density superoperator.
    lap.multiply_complex(&a, &b, d, d, count)
}
pub(super) fn partial_trace(j: &[Complex64], n: usize) -> Vec<C> {
    let d = n * n;
    let mut gram = vec![C::default(); d];
    for i in 0..n {
        for k in 0..n {
            for row in 0..n {
                let v = j[row * n + k + (row * n + i) * d];
                gram[i * n + k] += C::new(v.re, v.im);
            }
        }
    }
    gram
}

/// Dense accumulation for transformed collapse matrices avoids sorting millions
/// of duplicate sparse products. The resulting generator is exactly the same
/// full Lindblad superoperator, with algebraic zeros preserved.
pub(super) fn build_density_generator(
    lap: &Lapack,
    model: &Model,
) -> Result<super::Generator, String> {
    let n = model.dimension;
    let d = n * n;
    let mut h_model = model.clone();
    h_model.collapse.clear();
    let mut g = super::build_generator(&h_model)?;
    let map = coordinate_map(&g.coords, n);
    let mut matrix = vec![0.; d * d];
    for (col, entries) in g.matrix.iter().enumerate() {
        for &(row, v) in entries {
            matrix[row + col * d] = v;
        }
    }
    let operators: Vec<_> = model
        .collapse
        .iter()
        .map(|op| complex_input(&op.real, &op.imag, d, "transformed collapse"))
        .collect::<Result<_, String>>()?;
    let pairs: Vec<_> = operators
        .iter()
        .map(|c| (c.as_slice(), c.as_slice()))
        .collect();
    let j = choi(lap, &pairs, n)?;
    let gram = partial_trace(&j, n);
    let mut imaginary = vec![0.; n * d];
    for b in 0..n {
        for k in 0..n {
            for a in 0..n {
                for i in 0..n {
                    let v = j[a * n + i + (b * n + k) * d];
                    if v.re != 0. || v.im != 0. {
                        dense_term(
                            &mut matrix,
                            &mut imaginary,
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
            for column in 0..n {
                dense_term(
                    &mut matrix,
                    &mut imaginary,
                    &map,
                    i * n + column,
                    k * n + column,
                    value,
                    n,
                );
                dense_term(
                    &mut matrix,
                    &mut imaginary,
                    &map,
                    column * n + k,
                    column * n + i,
                    value,
                    n,
                );
            }
        }
    }
    if imaginary.iter().any(|v| !v.is_finite() || v.abs() > 1e-9) {
        return Err("Dense transformed generator violates Hermiticity preservation".into());
    }
    g.matrix = matrix
        .chunks(d)
        .map(|col| {
            col.iter()
                .enumerate()
                .filter_map(|(row, &v)| (v != 0.).then_some((row, v)))
                .collect()
        })
        .collect();
    g.scattering = gram;
    Ok(g)
}

#[inline]
fn dense_term(
    matrix: &mut [f64],
    imaginary_diagonal: &mut [f64],
    map: &[CoordMap],
    row: usize,
    col: usize,
    value: C,
    n: usize,
) {
    let d = n * n;
    let out = map[row];
    if out.lower {
        return;
    }
    let input = map[col];
    matrix[out.real + input.real * d] += value.re;
    if let Some(i) = input.imag {
        matrix[out.real + i * d] -= input.sign * value.im;
    }
    if let Some(o) = out.imag {
        matrix[o + input.real * d] += value.im;
        if let Some(i) = input.imag {
            matrix[o + i * d] += input.sign * value.re;
        }
    } else {
        imaginary_diagonal[out.real + input.real * n] += value.im;
        if let Some(i) = input.imag {
            imaginary_diagonal[out.real + i * n] += input.sign * value.re;
        }
    }
}
#[inline]
pub(super) fn cross_term(
    cos: &mut [f64],
    sin: &mut [f64],
    imcos: &mut [f64],
    imsin: &mut [f64],
    map: &[CoordMap],
    row: usize,
    col: usize,
    value: C,
    n: usize,
) {
    dense_term(cos, imcos, map, row, col, value, n);
    dense_term(sin, imsin, map, row, col, C::new(-value.im, value.re), n);
    let rt = (row % n) * n + row / n;
    let ct = (col % n) * n + col / n;
    dense_term(cos, imcos, map, rt, ct, value.conj(), n);
    dense_term(sin, imsin, map, rt, ct, C::new(-value.im, -value.re), n);
}

/// Exact Fourier convolution of D[C(t)], including C_a rho C_b^dagger and
/// every associated anticommutator. Only exactly zero entries are omitted.
pub(super) fn dissipative_harmonics(
    lap: &Lapack,
    channels: &[Vec<CollapseFourier>],
    n: usize,
) -> Result<Vec<DissipativeHarmonic>, String> {
    let d = n * n;
    let coords = coordinates(n);
    let map = coordinate_map(&coords, n);
    let max_index = channels
        .iter()
        .flat_map(|c| c.iter())
        .map(|c| c.index.abs())
        .max()
        .unwrap_or(0)
        * 2;
    let mut output = vec![];
    for harmonic in 1..=max_index {
        let mut cos = vec![0.; d * d];
        let mut sin = vec![0.; d * d];
        let mut imcos = vec![0.; n * d];
        let mut imsin = vec![0.; n * d];
        let pairs: Vec<_> = channels
            .iter()
            .flat_map(|channel| {
                channel.iter().flat_map(move |a| {
                    channel
                        .iter()
                        .filter(move |b| a.index - b.index == harmonic)
                        .map(move |b| (a.matrix.as_slice(), b.matrix.as_slice()))
                })
            })
            .collect();
        let j = choi(lap, &pairs, n)?;
        let gram = partial_trace(&j, n);
        for b in 0..n {
            for k in 0..n {
                for a in 0..n {
                    for i in 0..n {
                        let v = j[a * n + i + (b * n + k) * d];
                        if v.re != 0. || v.im != 0. {
                            cross_term(
                                &mut cos,
                                &mut sin,
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
                for j in 0..n {
                    cross_term(
                        &mut cos,
                        &mut sin,
                        &mut imcos,
                        &mut imsin,
                        &map,
                        i * n + j,
                        k * n + j,
                        value,
                        n,
                    );
                    cross_term(
                        &mut cos,
                        &mut sin,
                        &mut imcos,
                        &mut imsin,
                        &map,
                        j * n + k,
                        j * n + i,
                        value,
                        n,
                    );
                }
            }
        }
        let max = cos.iter().chain(&sin).map(|v| v.abs()).fold(0., f64::max);
        if imcos
            .iter()
            .chain(&imsin)
            .any(|v| !v.is_finite() || v.abs() > 1e-10 * (1. + max))
        {
            return Err("Transformed dissipator harmonic violates Hermiticity preservation".into());
        }
        for matrix in [&cos, &sin] {
            for col in 0..d {
                let trace: f64 = (0..n).map(|row| matrix[row + col * d]).sum();
                if !trace.is_finite() || trace.abs() > 1e-10 * (1. + max) {
                    return Err(
                        "Transformed dissipator harmonic violates trace preservation".into(),
                    );
                }
            }
        }
        let sparse = |matrix: Vec<f64>| -> Vec<Vec<(usize, f64)>> {
            matrix
                .chunks(d)
                .map(|col| {
                    col.iter()
                        .enumerate()
                        .filter_map(|(row, &v)| (v != 0.).then_some((row, v)))
                        .collect()
                })
                .collect()
        };
        if max > 0. {
            output.push(DissipativeHarmonic {
                index: harmonic,
                cos: sparse(cos),
                sin: sparse(sin),
            });
        }
    }
    Ok(output)
}

pub(super) fn prepare(lap: &Lapack, model: &Model, order: i32) -> Result<Frame, String> {
    prepare_with_radiation(lap, model, order, None)
}
pub(super) fn prepare_with_radiation(
    lap: &Lapack,
    model: &Model,
    order: i32,
    radiation: Option<&crate::radiative_bath::Radiation>,
) -> Result<Frame, String> {
    let started = Instant::now();
    let n = model.dimension;
    if model.drives.len() != 1 || order < 1 || order > 12 {
        return Err(
            "The isolated Hilbert Floquet fast path requires one beat and harmonic order 1..12"
                .into(),
        );
    }
    let drive = &model.drives[0];
    let omega = drive.omega;
    if !omega.is_finite() || omega <= 0. {
        return Err("Invalid Hamiltonian beat frequency".into());
    }
    let h0 = complex_input(&model.h_real, &model.h_imag, n * n, "Hilbert H0")?;
    let hc = complex_input(&drive.cos.real, &drive.cos.imag, n * n, "Hilbert cosine")?;
    let hs = complex_input(&drive.sin.real, &drive.sin.imag, n * n, "Hilbert sine")?;
    let plus: Vec<_> = hc
        .iter()
        .zip(&hs)
        .map(|(&c, &s)| C::new((c.re + s.im) * 0.5, (c.im - s.re) * 0.5))
        .collect();
    let minus: Vec<_> = hc
        .iter()
        .zip(&hs)
        .map(|(&c, &s)| C::new((c.re - s.im) * 0.5, (c.im + s.re) * 0.5))
        .collect();
    let lo = (0..n)
        .map(|i| h0[i * n + i].re)
        .fold(f64::INFINITY, f64::min);
    let hi = (0..n)
        .map(|i| h0[i * n + i].re)
        .fold(f64::NEG_INFINITY, f64::max);
    let radius = (0..n)
        .map(|i| {
            (0..n)
                .filter(|&j| j != i)
                .map(|j| h0[i * n + j].abs())
                .sum::<f64>()
        })
        .fold(0., f64::max);
    if omega <= 2. * (hi - lo + 2. * radius) {
        return Err("Hamiltonian Floquet zones are not isolated enough for this fast path; use the general full Fourier solver".into());
    }
    let blocks = (2 * order + 1) as usize;
    let m = blocks * n;
    let mut sambe = vec![C::default(); m * m];
    for block in 0..blocks {
        let harmonic = block as i32 - order;
        for i in 0..n {
            for j in 0..n {
                let row = block * n + i;
                let col = block * n + j;
                sambe[row + col * m] = h0[i * n + j];
                if i == j {
                    sambe[row + col * m].re += harmonic as f64 * omega;
                }
                if block > 0 {
                    sambe[row + ((block - 1) * n + j) * m] = plus[i * n + j];
                }
                if block + 1 < blocks {
                    sambe[row + ((block + 1) * n + j) * m] = minus[i * n + j];
                }
            }
        }
    }
    // Work within exact Hamiltonian graph components, preserving algebraic
    // selection rules even when the eigenspectrum is degenerate.
    let mut components = vec![];
    let mut seen = vec![false; m];
    for start in 0..m {
        if seen[start] {
            continue;
        }
        seen[start] = true;
        let mut queue = VecDeque::from([start]);
        let mut members = vec![];
        while let Some(col) = queue.pop_front() {
            members.push(col);
            for row in 0..m {
                let v = sambe[row + col * m];
                if !seen[row] && (v.re != 0. || v.im != 0.) {
                    seen[row] = true;
                    queue.push_back(row);
                }
            }
        }
        components.push(members);
    }
    let center = 0.5 * (lo + hi);
    let mut selected: Vec<(f64, Vec<C>)> = vec![];
    let mut other_values = vec![];
    for members in components {
        let r = members.len();
        let mut a = vec![Complex64 { re: 0., im: 0. }; r * r];
        for (j, &col) in members.iter().enumerate() {
            for (i, &row) in members.iter().enumerate() {
                let v = sambe[row + col * m];
                a[i + j * r] = Complex64 { re: v.re, im: v.im };
            }
        }
        let (values, vectors) = lap.hermitian_eigen(&a, r)?;
        for (j, &value) in values.iter().enumerate() {
            if (value - center).abs() < 0.5 * omega {
                let mut v = vec![C::default(); m];
                for (i, &row) in members.iter().enumerate() {
                    let z = vectors[i + j * r];
                    v[row] = C::new(z.re, z.im);
                }
                selected.push((value, v));
            } else {
                other_values.push(value);
            }
        }
    }
    if selected.len() != n {
        return Err(format!("Central Floquet zone contains {} modes rather than {n}; increase the harmonic basis or use the general solver",selected.len()));
    }
    selected.sort_by(|a, b| a.0.total_cmp(&b.0));
    let gap = selected
        .iter()
        .flat_map(|(value, _)| other_values.iter().map(move |other| (value - other).abs()))
        .fold(f64::INFINITY, f64::min);
    let mut modes = vec![vec![C::default(); n * n]; blocks];
    for (j, (_, v)) in selected.iter().enumerate() {
        for k in 0..blocks {
            for i in 0..n {
                modes[k][i * n + j] = v[k * n + i];
            }
        }
    }
    let mut fv = vec![DDC::default(); m * n];
    for (j, (_, v)) in selected.iter().enumerate() {
        for col in 0..m {
            let vc = v[col];
            if vc.re == 0. && vc.im == 0. {
                continue;
            }
            for row in 0..m {
                let value = sambe[row + col * m];
                if value.re != 0. || value.im != 0. {
                    fv[row + j * m].product(value, vc);
                }
            }
        }
    }
    let mut effective = vec![C::default(); n * n];
    for i in 0..n {
        for j in 0..n {
            let mut value = DDC::default();
            for row in 0..m {
                let u = selected[i].1[row].conj();
                let v = fv[row + j * m];
                value.product(u, C::new(v.re.hi, v.im.hi));
                value.product(u, C::new(v.re.lo, v.im.lo));
            }
            effective[i * n + j] = value.value();
        }
    }
    // Hermiticity is a mathematical invariant of V^dagger F V. Average only
    // its roundoff-level antisymmetric residue, never populations or rates.
    for i in 0..n {
        effective[i * n + i].im = 0.;
        for j in i + 1..n {
            let value = (effective[i * n + j] + effective[j * n + i].conj()) * 0.5;
            effective[i * n + j] = value;
            effective[j * n + i] = value.conj();
        }
    }
    let mut unitary_coeff = vec![vec![C::default(); n * n]; 2 * blocks - 1];
    for a in 0..blocks {
        let adj = adjoint(&modes[a], n);
        for b in 0..blocks {
            let product = multiply(&adj, &modes[b], n);
            let k = b + blocks - 1 - a;
            for (i, &v) in product.iter().enumerate() {
                unitary_coeff[k][i] += v;
            }
        }
    }
    for i in 0..n {
        unitary_coeff[blocks - 1][i * n + i].re -= 1.;
    }
    let unitarity = (0..n * n)
        .map(|i| unitary_coeff.iter().map(|c| c[i].abs()).sum::<f64>())
        .fold(0., f64::max);
    if !unitarity.is_finite() || unitarity > 1e-8 {
        return Err(format!(
            "Periodic Floquet basis is not unitary to tolerance: {unitarity:e}"
        ));
    }
    let mut defect = 0_f64;
    for block in -1_i32..blocks as i32 + 1 {
        let mut residual = vec![C::default(); n * n];
        if block >= 0 && (block as usize) < blocks {
            let k = block as usize;
            residual = multiply(&h0, &modes[k], n);
            let rhs = multiply(&modes[k], &effective, n);
            for i in 0..n * n {
                residual[i] += modes[k][i] * ((block - order) as f64 * omega) - rhs[i];
            }
        }
        if block > 0 && (block as usize) <= blocks {
            let product = multiply(&plus, &modes[(block - 1) as usize], n);
            for (i, &v) in product.iter().enumerate() {
                residual[i] += v;
            }
        }
        if block >= -1 && ((block + 1) as usize) < blocks {
            let product = multiply(&minus, &modes[(block + 1) as usize], n);
            for (i, &v) in product.iter().enumerate() {
                residual[i] += v;
            }
        }
        defect = defect.max(max_matrix(&residual));
    }
    let original_operators = if let Some(radiation) = radiation {
        radiation.normalized_dipoles(n)?
    } else {
        model
            .collapse
            .iter()
            .map(|op| complex_input(&op.real, &op.imag, n * n, "Floquet collapse"))
            .collect::<Result<Vec<_>, String>>()?
    };
    let mut transformed: Vec<Vec<CollapseFourier>> = vec![];
    for c in original_operators {
        let mut coefficients = vec![vec![C::default(); n * n]; 2 * blocks - 1];
        for a in 0..blocks {
            let left = multiply(&adjoint(&modes[a], n), &c, n);
            for b in 0..blocks {
                let product = multiply(&left, &modes[b], n);
                let k = b + blocks - 1 - a;
                for (i, &v) in product.iter().enumerate() {
                    coefficients[k][i] += v;
                }
            }
        }
        transformed.push(
            coefficients
                .into_iter()
                .enumerate()
                .map(|(k, matrix)| CollapseFourier {
                    index: k as i32 - (blocks - 1) as i32,
                    matrix,
                })
                .collect(),
        );
    }
    let radiation_diagnostics = if let Some(radiation) = radiation {
        let coefficients: Vec<crate::radiative_bath::Fourier> = transformed
            .iter()
            .map(|channel| {
                channel
                    .iter()
                    .map(|c| (vec![c.index], c.matrix.clone()))
                    .collect()
            })
            .collect();
        let weighted = crate::radiative_bath::weight_fourier(
            lap,
            radiation,
            &effective,
            &[omega],
            &coefficients,
        )?;
        transformed = weighted
            .channels
            .into_iter()
            .map(|channel| {
                channel
                    .into_iter()
                    .map(|(index, matrix)| CollapseFourier {
                        index: index[0],
                        matrix,
                    })
                    .collect()
            })
            .collect();
        Some(weighted.diagnostics)
    } else {
        None
    };
    Ok(Frame {
        dimension: n,
        omega,
        order,
        modes,
        hamiltonian: effective,
        collapse: transformed,
        unitarity_bound: unitarity,
        schrodinger_residual: defect,
        spectral_gap: gap,
        preparation_ms: started.elapsed().as_secs_f64() * 1000.,
        radiation_diagnostics,
    })
}

#[cfg(test)]
mod tests {
    use super::super::Drive;
    use super::*;
    #[test]
    fn periodic_frame_is_unitary_and_retains_complete_collapse_fourier() {
        let model = Model { frequency_coordinates:None,
            dimension: 2,
            states: vec![],
            h_real: vec![0., 0.3, 0.3, 1.],
            h_imag: vec![0.; 4],
            collapse: vec![Operator {
                real: vec![0., 1., 0., 0.],
                imag: vec![0.; 4],
            }],
            initial_index: 0,
            initial_real: None,
            initial_imag: None,
            time_dependent: true,
            drives: vec![Drive {
                omega: 80.,
                cos: Operator {
                    real: vec![0., 0.7, 0.7, 0.],
                    imag: vec![0.; 4],
                },
                sin: Operator {
                    real: vec![0.; 4],
                    imag: vec![0., 0.7, -0.7, 0.],
                },
            }],
        };
        let frame = prepare(&Lapack::load().unwrap(), &model, 4).unwrap();
        assert!(frame.unitarity_bound < 1e-11);
        assert!(frame.schrodinger_residual < 1e-8);
        let original =
            complex_input(&model.collapse[0].real, &model.collapse[0].imag, 4, "test").unwrap();
        for phase in [0., 0.13, 1.2, 3.5, 5.8] {
            let p = frame.at_phase(phase);
            let expected = multiply(&multiply(&adjoint(&p, 2), &original, 2), &p, 2);
            let mut actual = vec![C::default(); 4];
            for c in &frame.collapse[0] {
                let (s, co) = (c.index as f64 * phase).sin_cos();
                for (i, &v) in c.matrix.iter().enumerate() {
                    actual[i] += v * C::new(co, s);
                }
            }
            assert!(
                max_matrix(
                    &actual
                        .iter()
                        .zip(expected)
                        .map(|(&a, b)| a - b)
                        .collect::<Vec<_>>()
                ) < 1e-12
            );
        }
        assert!(frame.collapse[0]
            .iter()
            .any(|c| c.index != 0 && max_matrix(&c.matrix) > 1e-4));
        let lap = Lapack::load().unwrap();
        let density = frame
            .density_model(
                &lap,
                &model,
                &[C::new(1., 0.), C::default(), C::default(), C::default()],
            )
            .unwrap();
        let base = super::super::build_generator(&density.model).unwrap();
        let x = [0.4, 0.6, 0.1, -0.07];
        let lab = super::super::build_generator(&model).unwrap();
        let observable = frame.observable_fourier(&lab.scattering).unwrap();
        for phase in [0., 0.13, 1.2, 3.5, 5.8] {
            let sigma = super::super::expand_density(&coordinates(2), &x, 2);
            let p = frame.at_phase(phase);
            let rho = multiply(&multiply(&p, &sigma, 2), &adjoint(&p, 2), 2);
            let expected = [
                rho[0].re,
                rho[3].re,
                super::super::scatter_rate(&lab.scattering, &rho, 2),
            ];
            let mut actual = [C::default(); 3];
            for harmonic in &observable {
                let (s, c) = (harmonic.index as f64 * phase).sin_cos();
                let factor = C::new(c, s);
                for (row, value) in actual.iter_mut().enumerate() {
                    for (col, &v) in x.iter().enumerate() {
                        let coefficient = harmonic.matrix[row + col * 3];
                        *value += C::new(coefficient.re, coefficient.im) * factor * v;
                    }
                }
            }
            for (a, b) in actual.iter().zip(expected) {
                assert!((a.re - b).abs() < 1e-12);
                assert!(a.im.abs() < 1e-12);
            }
        }
        let optimized = build_density_generator(&lap, &density.model).unwrap();
        for (col, entries) in base.matrix.iter().enumerate() {
            for &(row, v) in entries {
                let actual = optimized.matrix[col]
                    .iter()
                    .find(|&&(i, _)| i == row)
                    .map(|&(_, v)| v)
                    .unwrap_or(0.);
                assert!((actual - v).abs() < 1e-12);
            }
        }
        for phase in [0.1, 0.7, 2.1, 4.2] {
            let apply = |matrix: &Vec<Vec<(usize, f64)>>| {
                let mut y = vec![0.; 4];
                for (col, entries) in matrix.iter().enumerate() {
                    for &(row, v) in entries {
                        y[row] += v * x[col];
                    }
                }
                y
            };
            let mut actual = apply(&base.matrix);
            for harmonic in &density.harmonics {
                let (s, c) = (harmonic.index as f64 * phase).sin_cos();
                let a = apply(&harmonic.cos);
                let b = apply(&harmonic.sin);
                for i in 0..4 {
                    actual[i] += c * a[i] + s * b[i];
                }
            }
            let p = frame.at_phase(phase);
            let c = multiply(&multiply(&adjoint(&p, 2), &original, 2), &p, 2);
            let mut at_phase = density.model.clone();
            at_phase.collapse = vec![Operator {
                real: c.iter().map(|z| z.re).collect(),
                imag: c.iter().map(|z| z.im).collect(),
            }];
            let expected = apply(&super::super::build_generator(&at_phase).unwrap().matrix);
            for (a, b) in actual.iter().zip(expected) {
                assert!((a - b).abs() < 1e-12, "{a} vs {b}");
            }
        }
    }
    #[test]
    #[ignore = "Full32-state Floquet-frame preparation and operator audit"]
    fn default_rb32_frame_audit() {
        let source = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../artifacts/v4/default-master.json"
        ))
        .unwrap();
        let model: Model = serde_json::from_str(&source).unwrap();
        let lap = Lapack::load().unwrap();
        let original = super::super::build_generator(&model).unwrap();
        let initial = super::super::expand_density(&original.coords, &original.initial, 32);
        for order in [1, 2] {
            let frame = prepare(&lap, &model, order).unwrap();
            let started = Instant::now();
            let density = frame.density_model(&lap, &model, &initial).unwrap();
            let density_ms = started.elapsed().as_secs_f64() * 1000.;
            let start = Instant::now();
            let g = build_density_generator(&lap, &density.model).unwrap();
            let harmonics:Vec<_>=density.harmonics.iter().map(|h|serde_json::json!({"index":h.index,"cosNnz":h.cos.iter().map(Vec::len).sum::<usize>(),"sinNnz":h.sin.iter().map(Vec::len).sum::<usize>(),"cosNorm1":h.cos.iter().map(|c|c.iter().map(|x|x.1.abs()).sum::<f64>()).fold(0.,f64::max),"sinNorm1":h.sin.iter().map(|c|c.iter().map(|x|x.1.abs()).sum::<f64>()).fold(0.,f64::max)})).collect();
            println!(
                "{}",
                serde_json::json!({"order":order,"frameMs":frame.preparation_ms,"densityPartsMs":density_ms,"baseGeneratorMs":start.elapsed().as_secs_f64()*1000.,"unitarity":frame.unitarity_bound,"schrodingerResidual":frame.schrodinger_residual,"gap":frame.spectral_gap,"baseNnz":g.matrix.iter().map(Vec::len).sum::<usize>(),"harmonics":harmonics})
            );
            assert!(frame.unitarity_bound < 1e-8);
            assert!(density.harmonics.iter().any(|h| h.index == 1));
        }
    }
}
