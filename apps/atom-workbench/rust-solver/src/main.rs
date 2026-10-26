//! Full-density Lindblad worker. Matrices in the public
//! JSON protocol are row-major, frequencies are rad/us, collapse amplitudes
//! are sqrt(1/us). LAPACK internally uses column-major LP64 arrays.
#![recursion_limit = "256"]
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::io::{self, BufRead, Write};
use std::time::Instant;

mod harmonics;
// Shared with population-core, which uses the eigen and Sylvester kernels
// this worker no longer needs.
#[allow(dead_code)]
mod lapack;
#[cfg(feature="population-core")]
mod population_core_adapter;
#[cfg(all(test,feature="population-core"))]
mod kernel_bench;
use lapack::Lapack;
const NATIVE_PROTOCOL: &str = "atom-lindblad-jsonl/2";
const FULL_MODEL_SCHEMA: &str = "rb87-full/2";
const GENERIC_MODEL_SCHEMA: &str = "generic-lindblad/1";

#[derive(Clone, Copy, Default, Debug)]
struct C {
    re: f64,
    im: f64,
}
impl C {
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }
    fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
}
impl std::ops::Add for C {
    type Output = Self;
    fn add(self, b: Self) -> Self {
        Self::new(self.re + b.re, self.im + b.im)
    }
}
impl std::ops::Sub for C {
    type Output = Self;
    fn sub(self, b: Self) -> Self {
        Self::new(self.re - b.re, self.im - b.im)
    }
}
impl std::ops::Mul for C {
    type Output = Self;
    fn mul(self, b: Self) -> Self {
        Self::new(
            self.re * b.re - self.im * b.im,
            self.re * b.im + self.im * b.re,
        )
    }
}
impl std::ops::Mul<f64> for C {
    type Output = Self;
    fn mul(self, b: f64) -> Self {
        Self::new(self.re * b, self.im * b)
    }
}
impl std::ops::AddAssign for C {
    fn add_assign(&mut self, b: Self) {
        self.re += b.re;
        self.im += b.im;
    }
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Operator {
    real: Vec<f64>,
    imag: Vec<f64>,
}
#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Drive {
    omega: f64,
    cos: Operator,
    sin: Operator,
}
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
struct FrequencyCoordinates {
    frequencies:Vec<f64>,
    #[serde(default,rename="frequenciesMHz")]
    frequencies_mhz:Option<Vec<f64>>,
    drive_indices:Vec<Vec<i32>>,
    frame_indices:Vec<Vec<i32>>,
    frame_sectors:Vec<i32>,
}
#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Model {
    dimension: usize,
    #[serde(default)]
    states: Vec<String>,
    h_real: Vec<f64>,
    h_imag: Vec<f64>,
    collapse: Vec<Operator>,
    #[serde(default)]
    initial_index: usize,
    initial_real: Option<Vec<f64>>,
    initial_imag: Option<Vec<f64>>,
    #[serde(default)]
    time_dependent: bool,
    #[serde(default)]
    drives: Vec<Drive>,
    #[serde(default)]
    frequency_coordinates:Option<FrequencyCoordinates>,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Options {
    #[serde(default = "default_duration")]
    duration_us: f64,
    #[serde(default = "default_samples")]
    samples: usize,
    /// Absolute population error on the probability (0..1) scale.
    #[serde(default = "default_population_tolerance")]
    population_tolerance: f64,
    #[serde(default = "default_true")]
    include_asymptotic: bool,
    #[serde(default)]
    steady_only: bool,
}
fn default_duration() -> f64 {
    5.
}
fn default_samples() -> usize {
    200
}
fn default_population_tolerance() -> f64 {
    0.005
}
fn default_true() -> bool {
    true
}
impl Default for Options {
    fn default() -> Self {
        Self {
            duration_us: 5.,
            samples: default_samples(),
            population_tolerance: default_population_tolerance(),
            include_asymptotic: true,
            steady_only: false,
        }
    }
}
impl Options {
    fn validate_population_tolerance(&self) -> Result<(), String> {
        if !self.population_tolerance.is_finite()
            || self.population_tolerance <= 0.
            || self.population_tolerance > 0.02
        {
            return Err("populationTolerance must be finite and in (0,0.02] on the probability scale".into());
        }
        Ok(())
    }
}
#[cfg(test)]
mod population_option_tests {
    use super::*;
    #[test]
    fn population_options_keep_legacy_default_and_requested_probability_scale() {
        let legacy:Options=serde_json::from_value(json!({})).unwrap();
        assert_eq!(legacy.population_tolerance,0.005);
        for value in [0.001,0.005,0.02,0.003] {
            let parsed:Options=serde_json::from_value(json!({"populationTolerance":value})).unwrap();
            assert_eq!(parsed.population_tolerance,value);
            assert!(parsed.validate_population_tolerance().is_ok());
        }
        for value in [0.,-0.001,0.020001,f64::INFINITY,f64::NAN] {
            assert!(Options{population_tolerance:value,..Default::default()}.validate_population_tolerance().is_err());
        }
    }
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Request {
    #[serde(default)]
    id: Value,
    #[serde(default)]
    op: Option<String>,
    #[serde(default)]
    model: Option<Model>,
    #[serde(default)]
    model_schema: Option<String>,
    #[serde(default)]
    options: Options,
    #[serde(default)]
    progress: bool,
}

thread_local! {static PROGRESS_REQUEST:std::cell::RefCell<Option<(Value,String,Option<Vec<Vec<i32>>>)>>=const{std::cell::RefCell::new(None)};}
fn stamp_progress_coordinates(result:&mut Value,schema:&str,mapping:Option<&Vec<Vec<i32>>>)->bool{
    if schema!=FULL_MODEL_SCHEMA{return true;}
    let Some(mapping)=mapping else{return false;};
    result["diagnostics"]["carrierSourceToBasis"]=json!(mapping);
    result["diagnostics"]["frequencyCoordinateSource"]=json!("declared-carrier-coordinates");true
}
fn publish_mean(mut result:Value){
    PROGRESS_REQUEST.with(|context|{if let Some((id,schema,mapping))=context.borrow().as_ref(){
        if !stamp_progress_coordinates(&mut result,schema,mapping.as_ref()){return;}
        result["protocol"]=json!(NATIVE_PROTOCOL);result["modelSchema"]=json!(schema);
        let mut out=io::BufWriter::new(io::stdout().lock());
        let message=json!({"id":id,"protocol":NATIVE_PROTOCOL,"progress":"asymptotic","result":result});
        if serde_json::to_writer(&mut out,&message).is_ok(){let _=writeln!(out);let _=out.flush();}
    }});
}

#[derive(Clone, Copy)]
enum Coordinate {
    Population(usize),
    Real(usize, usize),
    Imag(usize, usize),
}
fn coordinates(n: usize) -> Vec<Coordinate> {
    let mut c: Vec<_> = (0..n).map(Coordinate::Population).collect();
    for i in 0..n {
        for j in i + 1..n {
            c.push(Coordinate::Real(i, j));
            c.push(Coordinate::Imag(i, j));
        }
    }
    c
}
#[derive(Clone, Copy, Default)]
struct CoordMap {
    real: usize,
    imag: Option<usize>,
    sign: f64,
    lower: bool,
}
fn coordinate_map(coords: &[Coordinate], n: usize) -> Vec<CoordMap> {
    let mut map = vec![CoordMap::default(); n * n];
    for (k, &coord) in coords.iter().enumerate() {
        match coord {
            Coordinate::Population(i) => {
                map[i * n + i] = CoordMap {
                    real: k,
                    imag: None,
                    sign: 0.,
                    lower: false,
                }
            }
            Coordinate::Real(i, j) => {
                map[i * n + j].real = k;
                map[j * n + i].real = k;
            }
            Coordinate::Imag(i, j) => {
                map[i * n + j].imag = Some(k);
                map[i * n + j].sign = 1.;
                map[j * n + i].imag = Some(k);
                map[j * n + i].sign = -1.;
                map[j * n + i].lower = true;
            }
        }
    }
    map
}
#[inline]
fn add_entry(matrix: &mut [Vec<(usize, f64)>], row: usize, col: usize, v: f64) {
    if v != 0. {
        matrix[col].push((row, v));
    }
}
#[inline]
fn add_superterm(
    matrix: &mut [Vec<(usize, f64)>],
    diagonal_imag: &mut [f64],
    map: &[CoordMap],
    row: usize,
    col: usize,
    v: C,
    n: usize,
) {
    let out = map[row];
    if out.lower {
        return;
    }
    let input = map[col];
    add_entry(matrix, out.real, input.real, v.re);
    if let Some(ii) = input.imag {
        add_entry(matrix, out.real, ii, -input.sign * v.im);
    }
    if let Some(oi) = out.imag {
        add_entry(matrix, oi, input.real, v.im);
        if let Some(ii) = input.imag {
            add_entry(matrix, oi, ii, input.sign * v.re);
        }
    } else {
        diagonal_imag[out.real + input.real * n] += v.im;
        if let Some(ii) = input.imag {
            diagonal_imag[out.real + ii * n] += input.sign * v.re;
        }
    }
}
fn complex_input(real: &[f64], imag: &[f64], size: usize, name: &str) -> Result<Vec<C>, String> {
    if real.len() != size || imag.len() != size {
        return Err(format!(
            "{name}: expected {size} row-major real and imaginary entries"
        ));
    }
    if real.iter().chain(imag).any(|x| !x.is_finite()) {
        return Err(format!("{name}: non-finite matrix entry"));
    }
    Ok(real.iter().zip(imag).map(|(&r, &i)| C::new(r, i)).collect())
}
fn norm_one(a: &[f64], n: usize) -> f64 {
    a.chunks(n)
        .map(|c| c.iter().map(|x| x.abs()).sum::<f64>())
        .fold(0., f64::max)
}
fn max_abs(a: &[f64]) -> f64 {
    a.iter().map(|x| x.abs()).fold(0., f64::max)
}

struct Generator {
    n: usize,
    coords: Vec<Coordinate>,
    matrix: Vec<Vec<(usize, f64)>>,
    initial: Vec<f64>,
    scattering: Vec<C>,
}
fn build_generator(model: &Model) -> Result<Generator, String> {
    let n = model.dimension;
    if n == 0 || n > 64 {
        return Err("dimension must be in [1,64]".into());
    }
    let d = n * n;
    let h = complex_input(&model.h_real, &model.h_imag, d, "Hamiltonian")?;
    for a in 0..n {
        for b in 0..n {
            if (h[a * n + b] - h[b * n + a].conj()).abs() > 1e-10 * (1. + h[a * n + b].abs()) {
                return Err("Hamiltonian is not Hermitian".into());
            }
        }
    }
    // Accumulate each exact complex superoperator term directly into real
    // Hermitian coordinates. This is an invertible coordinate transform,
    // not a population equation or a discarded-coherence approximation.
    // Sparse entry products here are algebraic zeros, never detuning cutoffs.
    let coords = coordinates(n);
    let map = coordinate_map(&coords, n);
    let mut matrix = vec![Vec::new(); d];
    let mut diagonal_imag = vec![0.; n * d];
    for a in 0..n {
        for c in 0..n {
            let v = h[a * n + c];
            if v.re == 0. && v.im == 0. {
                continue;
            }
            for b in 0..n {
                add_superterm(
                    &mut matrix,
                    &mut diagonal_imag,
                    &map,
                    a * n + b,
                    c * n + b,
                    C::new(v.im, -v.re),
                    n,
                );
                add_superterm(
                    &mut matrix,
                    &mut diagonal_imag,
                    &map,
                    b * n + c,
                    b * n + a,
                    C::new(-v.im, v.re),
                    n,
                );
            }
        }
    }
    let mut scatter = vec![C::default(); d];
    for (operator_index, op) in model.collapse.iter().enumerate() {
        let c = complex_input(
            &op.real,
            &op.imag,
            d,
            &format!("collapse[{operator_index}]"),
        )?;
        let nonzero: Vec<_> = c
            .iter()
            .enumerate()
            .filter(|(_, z)| z.re != 0. || z.im != 0.)
            .map(|(k, &z)| (k / n, k % n, z))
            .collect();
        for &(a, i, v) in &nonzero {
            for &(b, j, w) in &nonzero {
                add_superterm(
                    &mut matrix,
                    &mut diagonal_imag,
                    &map,
                    a * n + b,
                    i * n + j,
                    v * w.conj(),
                    n,
                );
            }
        }
        let mut a = vec![C::default(); d];
        for &(row, i, v) in &nonzero {
            for &(row2, j, w) in &nonzero {
                if row == row2 {
                    a[i * n + j] += v.conj() * w;
                }
            }
        }
        for i in 0..d {
            scatter[i] += a[i];
        }
        for i in 0..n {
            for k in 0..n {
                let v = a[i * n + k] * (-0.5);
                if v.re == 0. && v.im == 0. {
                    continue;
                }
                for j in 0..n {
                    add_superterm(
                        &mut matrix,
                        &mut diagonal_imag,
                        &map,
                        i * n + j,
                        k * n + j,
                        v,
                        n,
                    );
                    add_superterm(
                        &mut matrix,
                        &mut diagonal_imag,
                        &map,
                        j * n + k,
                        j * n + i,
                        v,
                        n,
                    );
                }
            }
        }
    }
    if max_abs(&diagonal_imag) > 1e-9 {
        return Err("Lindblad generator fails Hermiticity invariance".into());
    }
    // Stable sorting retains accumulation order. Only exactly zero entries
    // are removed, including exact cancellations of Hamiltonian diagonals.
    for column in &mut matrix {
        column.sort_by_key(|&(row, _)| row);
        let mut reduced: Vec<(usize, f64)> = Vec::with_capacity(column.len());
        for &(row, value) in column.iter() {
            if let Some(last) = reduced.last_mut() {
                if last.0 == row {
                    last.1 += value;
                    continue;
                }
            }
            reduced.push((row, value));
        }
        reduced.retain(|&(_, value)| value != 0.);
        *column = reduced;
    }
    let mut initial = vec![0.; d];
    match (&model.initial_real, &model.initial_imag) {
        (Some(re), Some(im)) => {
            let rho = complex_input(re, im, d, "initial density")?;
            for a in 0..n {
                for b in 0..n {
                    if (rho[a * n + b] - rho[b * n + a].conj()).abs() > 1e-10 {
                        return Err("initial density is not Hermitian".into());
                    }
                }
            }
            for (index, &coord) in coords.iter().enumerate() {
                initial[index] = match coord {
                    Coordinate::Population(i) => rho[i * n + i].re,
                    Coordinate::Real(i, j) => rho[i * n + j].re,
                    Coordinate::Imag(i, j) => rho[i * n + j].im,
                };
            }
        }
        (None, None) => {
            if model.initial_index >= n {
                return Err("initialIndex is outside state space".into());
            }
            initial[model.initial_index] = 1.;
        }
        _ => return Err("initialReal and initialImag must be provided together".into()),
    }
    if (initial[..n].iter().sum::<f64>() - 1.).abs() > 1e-10 {
        return Err("initial density must have trace 1".into());
    }
    // Trace preservation of each independent Hermitian coordinate.
    for column in &matrix {
        let tr: f64 = column
            .iter()
            .filter(|&&(row, _)| row < n)
            .map(|&(_, v)| v)
            .sum();
        let scale = column.iter().map(|&(_, v)| v.abs()).fold(0., f64::max);
        if tr.abs() > 1e-9 * (1. + scale) {
            return Err("Lindblad generator fails trace preservation".into());
        }
    }
    Ok(Generator {
        n,
        coords,
        matrix,
        initial,
        scattering: scatter,
    })
}

/// A physical invariant Hilbert support, not the smaller coordinate graph of
/// one trajectory. It is closed under H, every harmonic Hamiltonian, jumps C,
/// and the no-jump backaction C†C. Only algebraic zeros are omitted.
fn hilbert_support(model: &Model) -> Result<Vec<usize>, String> {
    let n = model.dimension;
    if n == 0 || n > 64 {
        return Err("dimension must be in [1,64]".into());
    }
    let mut edges = vec![Vec::new(); n];
    let mut connect_hamiltonian = |re: &[f64], im: &[f64], name: &str| -> Result<(), String> {
        let h = complex_input(re, im, n * n, name)?;
        for i in 0..n {
            for j in 0..n {
                if h[i * n + j].re != 0. || h[i * n + j].im != 0. {
                    edges[i].push(j);
                    edges[j].push(i);
                }
            }
        }
        Ok(())
    };
    connect_hamiltonian(&model.h_real, &model.h_imag, "Hamiltonian support")?;
    for drive in &model.drives {
        connect_hamiltonian(
            &drive.cos.real,
            &drive.cos.imag,
            "cosine Hamiltonian support",
        )?;
        connect_hamiltonian(&drive.sin.real, &drive.sin.imag, "sine Hamiltonian support")?;
    }
    for op in &model.collapse {
        let c = complex_input(&op.real, &op.imag, n * n, "collapse support")?;
        let nonzero: Vec<_> = c
            .iter()
            .enumerate()
            .filter(|(_, z)| z.re != 0. || z.im != 0.)
            .map(|(k, &z)| (k / n, k % n, z))
            .collect();
        let mut k = vec![C::default(); n * n];
        for &(row, col, _) in &nonzero {
            edges[col].push(row);
        }
        for &(row, i, v) in &nonzero {
            for &(row2, j, w) in &nonzero {
                if row == row2 {
                    k[i * n + j] += v.conj() * w;
                }
            }
        }
        for i in 0..n {
            for j in 0..n {
                if k[i * n + j].re != 0. || k[i * n + j].im != 0. {
                    edges[i].push(j);
                    edges[j].push(i);
                }
            }
        }
    }
    let mut seen = vec![false; n];
    let mut queue = VecDeque::new();
    match (&model.initial_real, &model.initial_imag) {
        (Some(re), Some(im)) => {
            let rho = complex_input(re, im, n * n, "initial support")?;
            for i in 0..n {
                if (0..n).any(|j| {
                    rho[i * n + j].re != 0.
                        || rho[i * n + j].im != 0.
                        || rho[j * n + i].re != 0.
                        || rho[j * n + i].im != 0.
                }) {
                    seen[i] = true;
                    queue.push_back(i);
                }
            }
        }
        (None, None) => {
            if model.initial_index >= n {
                return Err("initialIndex is outside state space".into());
            }
            seen[model.initial_index] = true;
            queue.push_back(model.initial_index);
        }
        _ => return Err("initialReal and initialImag must be provided together".into()),
    }
    while let Some(i) = queue.pop_front() {
        for &j in &edges[i] {
            if !seen[j] {
                seen[j] = true;
                queue.push_back(j);
            }
        }
    }
    Ok((0..n).filter(|&i| seen[i]).collect())
}
/// Generic dense static stationary solver. It serves only generic-lindblad/1
/// steadyOnly requests (the worker start-up check); every rb87-full/2
/// request is solved by the population core.
struct Solver {
    lap: Lapack,
}

fn expand_density(coords: &[Coordinate], x: &[f64], n: usize) -> Vec<C> {
    let mut rho = vec![C::default(); n * n];
    for (&coord, &v) in coords.iter().zip(x) {
        match coord {
            Coordinate::Population(i) => rho[i * n + i].re = v,
            Coordinate::Real(i, j) => {
                rho[i * n + j].re = v;
                rho[j * n + i].re = v;
            }
            Coordinate::Imag(i, j) => {
                rho[i * n + j].im = v;
                rho[j * n + i].im = -v;
            }
        }
    }
    rho
}
fn positivity(lap: &Lapack, rho: &[C], n: usize) -> Result<f64, String> {
    // Hermitian complex matrix has the same spectrum (twice) in this real
    // symmetric embedding, so LAPACK dsyev supplies a full PSD diagnostic.
    let m = 2 * n;
    let mut embedded = vec![0.; m * m];
    for i in 0..n {
        for j in 0..n {
            let v = rho[i * n + j];
            embedded[i + j * m] = v.re;
            embedded[i + (j + n) * m] = -v.im;
            embedded[i + n + j * m] = v.im;
            embedded[i + n + (j + n) * m] = v.re;
        }
    }
    Ok((-lap.symmetric_eigenvalues(&embedded, m)?[0]).max(0.))
}
fn scatter_rate(operator: &[C], rho: &[C], n: usize) -> f64 {
    let mut value = C::default();
    for i in 0..n {
        for j in 0..n {
            value += operator[i * n + j] * rho[j * n + i];
        }
    }
    value.re
}
impl Solver {
    fn new() -> Result<Self, String> {
        Ok(Self { lap: Lapack::load()? })
    }
    /// generic-lindblad/1: the unique stationary density of a static
    /// generator (steadyOnly). Finite-time and driven generic requests are not
    /// served.
    fn solve(&mut self, model: &Model, options: &Options) -> Result<Value, String> {
        let started = Instant::now();
        options.validate_population_tolerance()?;
        if model.time_dependent {
            return Err("Time-dependent input needs explicit harmonic drives; no unsupported term may be silently omitted.".into());
        }
        if !model.drives.is_empty() {
            return Err("Driven generators are solved through modelSchema rb87-full/2; generic-lindblad/1 accepts static generators only".into());
        }
        let g = build_generator(model)?;
        if positivity(&self.lap, &expand_density(&g.coords, &g.initial, g.n), g.n)? > 1e-10 {
            return Err("initial density is not positive semidefinite".into());
        }
        if !options.steady_only {
            return Err("generic-lindblad/1 serves only steadyOnly stationary states; trajectories use rb87-full/2".into());
        }
        let construction_ms = started.elapsed().as_secs_f64() * 1000.;
        self.stationary_only(model, &g, started, construction_ms)?
            .ok_or_else(|| "No unique, well-conditioned stationary density exists for this generic generator".into())
    }
    /// Trace-constrained dense LU on the full Hermitian operator algebra of
    /// the invariant Hilbert support. A nonsingular, well-conditioned system
    /// proves a unique stationary density.
    fn stationary_only(
        &mut self,
        model: &Model,
        g: &Generator,
        started: Instant,
        construction_ms: f64,
    ) -> Result<Option<Value>, String> {
        // Unique in one initial-coordinate trajectory is insufficient: pure
        // Rabi has a unique trace-constrained solution in {p_g,p_e,Im rho_ge}
        // yet does not relax. Include the full operator algebra of the closed
        // Hilbert support, including its unseeded conserved coherences.
        let support = hilbert_support(model)?;
        let mut present = vec![false; g.n];
        for &i in &support {
            present[i] = true;
        }
        let inds: Vec<_> = g
            .coords
            .iter()
            .enumerate()
            .filter_map(|(index, coord)| match *coord {
                Coordinate::Population(i) => present[i].then_some(index),
                Coordinate::Real(i, j) | Coordinate::Imag(i, j) => {
                    (present[i] && present[j]).then_some(index)
                }
            })
            .collect();
        let r = inds.len();
        let trace_row = inds
            .iter()
            .position(|&i| i < g.n)
            .ok_or("No population coordinate")?;
        let mut a = vec![0.; r * r];
        let mut positions = vec![usize::MAX; g.n * g.n];
        for (i, &global) in inds.iter().enumerate() {
            positions[global] = i;
        }
        for (col, &global) in inds.iter().enumerate() {
            for &(row, value) in &g.matrix[global] {
                let local = positions[row];
                if local == usize::MAX {
                    return Err(
                        "Physical Hilbert support failed generator-invariance verification".into(),
                    );
                }
                a[local + col * r] = value;
            }
        }
        let mut constrained = a.clone();
        let mut rhs = vec![0.; r];
        rhs[trace_row] = 1.;
        for (col, &index) in inds.iter().enumerate() {
            constrained[trace_row + col * r] = if index < g.n { 1. } else { 0. };
        }
        // Row equilibration changes neither the equation nor its solution.
        for row in 0..r {
            let scale = (0..r)
                .map(|col| constrained[row + col * r].abs())
                .fold(0., f64::max);
            if scale == 0. {
                return Ok(None);
            }
            for col in 0..r {
                constrained[row + col * r] /= scale;
            }
            rhs[row] /= scale;
        }
        let (lu, pivots) = match self.lap.factor(&constrained, r) {
            Ok(v) => v,
            Err(_) => return Ok(None),
        };
        let condition = self.lap.condition(&lu, r, norm_one(&constrained, r))?;
        if !condition.is_finite() || condition > 1e12 {
            return Ok(None);
        }
        let steady = self.lap.solve_factored(&lu, &pivots, &rhs, r)?;
        let remainder = self.lap.multiply(&a, &steady, r, 1, r);
        let residual = remainder.iter().map(|x| x * x).sum::<f64>().sqrt();
        let population_residual = inds
            .iter()
            .enumerate()
            .filter(|(_, global)| **global < g.n)
            .map(|(i, _)| remainder[i].abs())
            .fold(0., f64::max);
        let mut x = vec![0.; g.n * g.n];
        for (i, &global) in inds.iter().enumerate() {
            x[global] = steady[i];
        }
        let rho = expand_density(&g.coords, &x, g.n);
        let trace_error = (x[..g.n].iter().sum::<f64>() - 1.).abs();
        let positivity_error = positivity(&self.lap, &rho, g.n)?;
        if residual > 1e-6 || trace_error.max(positivity_error) > 2e-6 {
            return Ok(None);
        }
        let initial_rho = expand_density(&g.coords, &g.initial, g.n);
        let states: Vec<String> = if model.states.len() == g.n {
            model.states.clone()
        } else {
            (0..g.n).map(|i| i.to_string()).collect()
        };
        Ok(Some(json!({"timesUs":[0.],"states":states,"populations":[g.initial[..g.n].to_vec()],
            "scatteringRatePerUs":[scatter_rate(&g.scattering,&initial_rho,g.n)],"asymptoticPopulations":x[..g.n].to_vec(),
            "asymptoticDensityReal":rho.iter().map(|z|z.re).collect::<Vec<_>>(),"asymptoticDensityImag":rho.iter().map(|z|z.im).collect::<Vec<_>>(),
            "asymptoticScatteringRatePerUs":scatter_rate(&g.scattering,&rho,g.n),"asymptoticStatus":"stationary",
            "diagnostics":{"method":"Rust / LAPACK full-density trace-constrained Lindblad stationary solve","engine":"rust-lapack",
                "liouvilleDimension":g.n*g.n,"stationaryPhysicalHilbertDimension":support.len(),"stationaryOperatorDimension":r,"stationaryOnly":true,"elapsedMs":started.elapsed().as_secs_f64()*1000.,
                "constructionMs":construction_ms,"traceError":trace_error,"positivityError":positivity_error,"hermiticityError":0.,
                "stationarySystemCondition":condition,"residual":residual,"populationResidual":population_residual,
                "asymptoticStatus":"stationary","zeroModeCount":1,"densityLimitExists":true,
                "uniquenessScope":"full Hermitian operator algebra on the invariant physical Hilbert support",
                "asymptoticReason":"Unique full-density Lindblad stationary solution, verified by matrix residual and positivity."}})))
    }
}

#[cfg(test)]
mod progress_tests{
    use super::*;
    #[test]
    fn declared_static_progress_carries_mapping_and_legacy_progress_omits_it(){
        let mut declared=json!({"diagnostics":{},"asymptoticPopulations":[1.,0.]});
        assert!(!stamp_progress_coordinates(&mut declared,"rb87-full/2",None));
        assert!(declared["diagnostics"].get("carrierSourceToBasis").is_none());
        let mapping=vec![vec![1],vec![3]];
        assert!(stamp_progress_coordinates(&mut declared,"rb87-full/2",Some(&mapping)));
        assert_eq!(declared["diagnostics"]["carrierSourceToBasis"],json!(mapping));
        let mut legacy=json!({"diagnostics":{}});assert!(stamp_progress_coordinates(&mut legacy,GENERIC_MODEL_SCHEMA,None));
        assert!(legacy["diagnostics"].get("carrierSourceToBasis").is_none());
        let mut static_zero=json!({"diagnostics":{}});assert!(stamp_progress_coordinates(&mut static_zero,"rb87-full/2",Some(&vec![])));
        assert_eq!(static_zero["diagnostics"]["carrierSourceToBasis"],json!([]));
    }
}

/// rb87-full/2: every finite-time trajectory and every long-time mean
/// (static or driven) is solved by the population core.
fn full_model_solve(solver:&mut Solver,model:&Model,options:&Options)->Result<Value,String>{
    options.validate_population_tolerance()?;
    if model.frequency_coordinates.is_none(){return Err("rb87-full/2 requires authoritative frequencyCoordinates metadata".into());}
    let axes=harmonics::model_phase_axes(model)?;
    // Request-local phase metadata must precede EVERY possible progress path.
    PROGRESS_REQUEST.with(|context|{
        if let Some((_,schema,mapping))=context.borrow_mut().as_mut(){if schema==FULL_MODEL_SCHEMA{*mapping=Some(axes.source_to_basis.clone());}}
    });
    #[cfg(feature="population-core")]
    if population_core_adapter::supports(model,options){
        return population_core_adapter::solve(solver,model,options);
    }
    let _=solver;
    Err("rb87-full/2 requires the complete 32-state Rb87 model and the population core".into())
}

#[cfg(feature="static-numerics")]
pub fn embedded_source_manifest()->&'static str{include_str!(env!("ATOM_NUMERIC_MANIFEST"))}

/// Shared JSONL entry used by the ordinary solver executable and the native
/// desktop executable's isolated --worker process. Numerical code is shared.
pub fn main() {
    let mut solver = match Solver::new() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("atom-lindblad initialization failed: {e}");
            std::process::exit(1)
        }
    };
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("stdin: {e}");
                break;
            }
        };
        if line.trim().is_empty() {
            continue;
        }
        let output = match serde_json::from_str::<Request>(&line) {
            Ok(request) => {
                PROGRESS_REQUEST.with(|p|*p.borrow_mut()=if request.progress{Some((request.id.clone(),request.model_schema.clone().unwrap_or(GENERIC_MODEL_SCHEMA.into()),None))}else{None});
                if request.op.as_deref() == Some("capabilities") {
                    json!({"id":request.id,"protocol":NATIVE_PROTOCOL,"capabilities":[FULL_MODEL_SCHEMA,GENERIC_MODEL_SCHEMA]})
                } else {
                    let result = if request.op.is_some() {
                        Err("Unknown native operation".into())
                    } else if let Err(error) = request.options.validate_population_tolerance() {
                        Err(error)
                    } else if let Some(model) = request.model.as_ref() {
                        match request.model_schema.as_deref(){
                            Some(FULL_MODEL_SCHEMA)=>full_model_solve(&mut solver,model,&request.options),
                            None|Some(GENERIC_MODEL_SCHEMA)=>solver.solve(model,&request.options),
                            _=>Err("Unsupported native model schema".into()),
                        }
                    } else {
                        Err("A solve request requires a model".into())
                    };
                    match result {
                        Ok(mut result) => {
                            result["protocol"] = json!(NATIVE_PROTOCOL);
                            result["modelSchema"] = json!(request
                                .model_schema
                                .as_deref()
                                .unwrap_or(GENERIC_MODEL_SCHEMA));
                            json!({"id":request.id,"protocol":NATIVE_PROTOCOL,"result":result})
                        }
                        Err(error) => {
                            json!({"id":request.id,"protocol":NATIVE_PROTOCOL,"error":error})
                        }
                    }
                }
            }
            Err(error) => {
                json!({"id":null,"protocol":NATIVE_PROTOCOL,"error":format!("Invalid request: {error}")})
            }
        };
        PROGRESS_REQUEST.with(|p|*p.borrow_mut()=None);
        let mut stdout=io::BufWriter::new(io::stdout().lock());
        if serde_json::to_writer(&mut stdout, &output).is_err() {
            break;
        }
        if writeln!(stdout).is_err() || stdout.flush().is_err() {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn model(n: usize, h: Vec<f64>, c: Vec<Operator>, initial: usize) -> Model {
        Model { frequency_coordinates:None,
            dimension: n,
            states: vec![],
            h_real: h,
            h_imag: vec![0.; n * n],
            collapse: c,
            initial_index: initial,
            initial_real: None,
            initial_imag: None,
            time_dependent: false,
            drives: vec![],
        }
    }
    fn steady() -> Options {
        Options { steady_only: true, ..Options::default() }
    }
    #[test]
    fn stationary_only_solves_full_density_without_spectrum() {
        let mut s = Solver::new().unwrap();
        let c = Operator { real: vec![0., 1., 0., 0.], imag: vec![0.; 4] };
        let out = s.solve(&model(2, vec![0., 1., 1., 0.], vec![c], 0), &steady()).unwrap();
        assert_eq!(out["diagnostics"]["stationaryOnly"], true);
        assert!((out["asymptoticPopulations"][1].as_f64().unwrap() - 4. / 9.).abs() < 1e-10);
        assert_eq!(out["timesUs"], json!([0.]));
    }
    #[test]
    fn startup_check_model_is_its_own_stationary_state() {
        let mut s = Solver::new().unwrap();
        let mut m = model(1, vec![0.], vec![], 0);
        m.states = vec!["startup".into()];
        let out = s.solve(&m, &steady()).unwrap();
        assert_eq!(out["populations"], json!([[1.]]));
        assert_eq!(out["asymptoticStatus"], "stationary");
    }
    #[test]
    fn generic_requests_are_static_and_stationary_only() {
        let mut s = Solver::new().unwrap();
        let mut m = model(2, vec![0.; 4], vec![], 0);
        m.time_dependent = true;
        assert!(s.solve(&m, &steady()).unwrap_err().contains("explicit harmonic drives"));
        let c = Operator { real: vec![0., 1., 0., 0.], imag: vec![0.; 4] };
        assert!(s.solve(&model(2, vec![0., 1., 1., 0.], vec![c], 0), &Options::default()).unwrap_err().contains("steadyOnly"));
        // A closed two-level system has no unique stationary density.
        assert!(s.solve(&model(2, vec![0., 1., 1., 0.], vec![], 0), &steady()).is_err());
    }
    #[test]
    fn hilbert_support_includes_no_jump_backaction_and_harmonic_connections() {
        // C maps |0> only to itself, but C+C couples |0> and |1>.
        let mut m = model(
            3,
            vec![0.; 9],
            vec![Operator {
                real: vec![1., 1., 0., 0., 0., 0., 0., 0., 0.],
                imag: vec![0.; 9],
            }],
            0,
        );
        assert_eq!(hilbert_support(&m).unwrap(), vec![0, 1]);
        let mut h = vec![0.; 9];
        h[5] = 1.;
        h[7] = 1.;
        m.drives.push(Drive {
            omega: 10.,
            cos: Operator { real: h, imag: vec![0.; 9] },
            sin: Operator { real: vec![0.; 9], imag: vec![0.; 9] },
        });
        assert_eq!(hilbert_support(&m).unwrap(), vec![0, 1, 2]);
    }
    #[test]
    fn invalid_initial_density_is_rejected_instead_of_repaired() {
        let mut solver = Solver::new().unwrap();
        for (real, imag) in [
            (vec![1.1, 0., 0., -0.1], vec![0.; 4]),
            (vec![0.5, 0.2, 0., 0.5], vec![0.; 4]),
            (vec![0.4, 0., 0., 0.5], vec![0.; 4]),
            (vec![0.5, 0., 0., 0.5], vec![0.01, 0., 0., 0.]),
        ] {
            let c = Operator { real: vec![0., 1., 0., 0.], imag: vec![0.; 4] };
            let mut m = model(2, vec![0.; 4], vec![c], 0);
            m.initial_real = Some(real);
            m.initial_imag = Some(imag);
            assert!(solver.solve(&m, &steady()).is_err());
        }
    }
}
