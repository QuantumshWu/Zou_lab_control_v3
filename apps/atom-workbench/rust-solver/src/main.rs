//! Full-density Lindblad worker. Matrices in the public
//! JSON protocol are row-major, frequencies are rad/us, collapse amplitudes
//! are sqrt(1/us). LAPACK internally uses column-major LP64 arrays.
#![recursion_limit = "256"]
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::io::{self, BufRead, Write};
use std::time::Instant;

mod bath_frame;
mod block_solve;
mod cluster_action;
mod cptp_period;
mod defect_bound;
mod dense_expm;
mod full_period;
mod floquet;
mod harmonics;
mod lapack;
mod matrix_probe;
mod native_superlu;
mod stationary_structure;
mod structured_lindblad;
mod structured_floquet;
#[cfg(feature="population-core")]
mod population_core_adapter;
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
    #[serde(default)]
    return_density: bool,
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
            return_density: false,
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

fn reachable(matrix: &[Vec<(usize, f64)>], initial: &[f64]) -> Vec<usize> {
    let d = initial.len();
    let mut seen = vec![false; d];
    let mut queue = VecDeque::new();
    for (i, &x) in initial.iter().enumerate() {
        if x != 0. {
            seen[i] = true;
            queue.push_back(i);
        }
    }
    while let Some(col) = queue.pop_front() {
        for &(row, _) in &matrix[col] {
            if !seen[row] {
                seen[row] = true;
                queue.push_back(row);
            }
        }
    }
    (0..d).filter(|&i| seen[i]).collect()
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
struct Spectrum {
    wr: Vec<f64>,
    wi: Vec<f64>,
    vectors: Vec<f64>,
    lu: Vec<f64>,
    pivots: Vec<i32>,
    condition: f64,
    residual: f64,
    norm: f64,
}
fn spectral(lap: &Lapack, a: &[f64], n: usize) -> Result<Spectrum, String> {
    let (wr, wi, vectors) = lap.eigen(a, n)?;
    let mut product = lap.multiply(a, &vectors, n, n, n);
    for j in 0..n {
        for row in 0..n {
            let mut expected = vectors[row + j * n] * wr[j];
            if wi[j] > 0. {
                expected -= vectors[row + (j + 1) * n] * wi[j];
            } else if wi[j] < 0. {
                expected -= vectors[row + (j - 1) * n] * wi[j];
            }
            product[row + j * n] -= expected;
        }
    }
    let residual = max_abs(&product);
    let norm = norm_one(a, n).max(1.);
    let (lu, pivots) = lap.factor(&vectors, n)?;
    let condition = lap.condition(&lu, n, norm_one(&vectors, n))?;
    if !condition.is_finite() || condition > 1e10 || residual > 1e-8 * norm {
        return Err(format!("Lindblad eigenbasis is ill-conditioned (condition {condition:e}, residual {residual:e}); no inaccurate trajectory returned"));
    }
    Ok(Spectrum {
        wr,
        wi,
        vectors,
        lu,
        pivots,
        condition,
        residual,
        norm,
    })
}
struct SpectralCertificate {
    residual_norm: Vec<f64>,
    vector_norm: Vec<f64>,
    gamma: f64,
}
struct Cached {
    matrix: Vec<f64>,
    dimension: usize,
    hilbert_dimension: usize,
    physical_support_dimension: usize,
    coordinate_indices: Vec<usize>,
    spectrum: Spectrum,
    certificate: SpectralCertificate,
    refined: Option<cluster_action::Prepared>,
    modal_certificate: Option<defect_bound::Prepared>,
    modal_certificate_failure: Option<String>,
}
struct CachedStationary {
    matrix: Vec<f64>,
    indices: Vec<usize>,
    hilbert_dimension: usize,
    result: Value,
    density: Vec<C>,
}
struct Solver {
    lap: Lapack,
    cache: VecDeque<Cached>,
    stationary_cache: VecDeque<CachedStationary>,
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
fn initialize_numeric_parallelism() {
    static INITIALIZED: std::sync::Once = std::sync::Once::new();
    INITIALIZED.call_once(|| {
        // Enabling faer/rayon otherwise defaults to every available core.
        // Bound this worker's Rust tasks explicitly; BLAS remains separate.
        let workers = std::env::var("ATOM_NUMERIC_THREADS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map_or(1, |value| value.get())
                    .min(4)
            })
            .clamp(1, 8);
        faer::set_global_parallelism(if workers == 1 {
            faer::Par::Seq
        } else {
            faer::Par::rayon(workers)
        });
    });
}
impl Solver {
    fn new() -> Result<Self, String> {
        initialize_numeric_parallelism();
        Ok(Self {
            lap: Lapack::load()?,
            cache: VecDeque::new(),
            stationary_cache: VecDeque::new(),
        })
    }
    // A posteriori defect of the *actual initial-state trajectory*, rather
    // than a global ||L||-based eigenvalue threshold. For Hermitian inputs a
    // CPTP semigroup is trace-norm contractive. Integrating ||x'-Lx||_1 bounds
    // its propagation error. ||rho||_1 <= sqrt(n)||rho||_F converts the real
    // Hermitian-coordinate residual without discarding any coherence.
    // The matrix-product roundoff term is componentwise; phase evaluation is
    // a floating-point risk estimate, not an interval-arithmetic certificate.
    fn spectral_certificate(
        &self,
        a: &[f64],
        spec: &Spectrum,
        inds: &[usize],
        hilbert: usize,
    ) -> SpectralCertificate {
        let r = inds.len();
        let product = self.lap.multiply(a, &spec.vectors, r, r, r);
        let absolute_a: Vec<_> = a.iter().map(|x| x.abs()).collect();
        let absolute_v: Vec<_> = spec.vectors.iter().map(|x| x.abs()).collect();
        let product_magnitude = self.lap.multiply(&absolute_a, &absolute_v, r, r, r);
        let unit = f64::EPSILON * 0.5;
        let gamma = (r as f64 + 8.) * unit / (1. - (r as f64 + 8.) * unit);
        let mut residual_norm = vec![0.; r];
        let mut vector_norm = vec![0.; r];
        for j in 0..r {
            for i in 0..r {
                let v = spec.vectors[i + j * r];
                let mut expected = v * spec.wr[j];
                let mut magnitude = (v * spec.wr[j]).abs();
                if spec.wi[j] > 0. {
                    let z = spec.vectors[i + (j + 1) * r] * spec.wi[j];
                    expected -= z;
                    magnitude += z.abs();
                } else if spec.wi[j] < 0. {
                    let z = spec.vectors[i + (j - 1) * r] * spec.wi[j];
                    expected -= z;
                    magnitude += z.abs();
                }
                let residual = (product[i + j * r] - expected).abs()
                    + gamma * (product_magnitude[i + j * r] + magnitude);
                let weight = if inds[i] < hilbert { 1. } else { 2. };
                residual_norm[j] += weight * residual * residual;
                vector_norm[j] += weight * v * v;
            }
        }
        for value in residual_norm.iter_mut().chain(&mut vector_norm) {
            *value = value.sqrt();
        }
        SpectralCertificate {
            residual_norm,
            vector_norm,
            gamma,
        }
    }
    fn spectral_defect_bound(
        &self,
        spec: &Spectrum,
        coefficients: &[f64],
        certificate: &SpectralCertificate,
        hilbert: usize,
        time: f64,
    ) -> (f64, f64) {
        let r = coefficients.len();
        let unit = f64::EPSILON * 0.5;
        let gamma = certificate.gamma;
        let residual_norm = &certificate.residual_norm;
        let vector_norm = &certificate.vector_norm;
        let mut defect = 0.;
        let mut floating = 0.;
        let mut j = 0;
        while j < r {
            let complex = spec.wi[j] > 0.;
            let coefficient = if complex {
                coefficients[j].hypot(coefficients[j + 1])
            } else {
                coefficients[j].abs()
            };
            if coefficient == 0. {
                j += if complex { 2 } else { 1 };
                continue;
            }
            let real = spec.wr[j];
            let integrated = if real == 0. {
                time
            } else {
                (real * time).exp_m1() / real
            };
            let growth = (real * time).max(0.).exp();
            let phase_peak_time = if real < 0. {
                time.min(-1. / real)
            } else {
                time
            };
            let phase_growth = phase_peak_time * (real * phase_peak_time).exp();
            let residual = if complex {
                residual_norm[j].hypot(residual_norm[j + 1])
            } else {
                residual_norm[j]
            };
            let vector = if complex {
                vector_norm[j].hypot(vector_norm[j + 1])
            } else {
                vector_norm[j]
            };
            defect += residual * coefficient * integrated;
            floating += vector
                * coefficient
                * ((gamma + 64. * unit) * growth
                    + 2. * unit * phase_growth * (real.abs() + spec.wi[j].abs()));
            j += if complex { 2 } else { 1 };
        }
        let norm_factor = (hilbert as f64).sqrt();
        (norm_factor * defect, norm_factor * floating)
    }
    fn pade_constant(
        &self,
        model: &Model,
        g: &Generator,
        a: &[f64],
        inds: &[usize],
        initial: &[f64],
        options: &Options,
        stationary: &Option<Value>,
        started: Instant,
        construction_ms: f64,
        reason: &str,
    ) -> Result<Value, String> {
        if inds.len() > 64 {
            return Err(format!("{reason}; dense Pade fallback for {} reachable coordinates is outside the interactive fallback budget; no unverified trajectory returned",inds.len()));
        }
        let times: Vec<_> = if options.steady_only {
            vec![0.]
        } else {
            (0..options.samples)
                .map(|i| options.duration_us * i as f64 / (options.samples - 1) as f64)
                .collect()
        };
        let report = dense_expm::action_with_diagnostics(&self.lap, a, initial, &times)?;
        let risk = report
            .steps
            .iter()
            .map(|s| s.action_roundoff_risk_estimate.unwrap_or(f64::INFINITY))
            .fold(0_f64, f64::max);
        let backward = report
            .steps
            .iter()
            .map(|s| s.linear_solve_backward_error)
            .fold(0_f64, f64::max);
        if !risk.is_finite() || risk > 5e-6 || backward > 1e-10 {
            return Err(format!("Dense Pade fallback cannot verify the requested trajectory: roundoff-risk estimate {risk:e}, linear-solve backward error {backward:e}"));
        }
        let support = hilbert_support(model)?;
        let mut populations = Vec::new();
        let mut scattering = Vec::new();
        let mut density_real = Vec::new();
        let mut density_imag = Vec::new();
        let mut trace = 0_f64;
        let mut psd = 0_f64;
        let mut initial_error = 0_f64;
        for (time, state) in report.states.iter().enumerate() {
            let mut x = vec![0.; g.n * g.n];
            for (i, &global) in inds.iter().enumerate() {
                x[global] = state[i];
            }
            if time == 0 {
                initial_error = max_abs(
                    &x.iter()
                        .zip(&g.initial)
                        .map(|(a, b)| a - b)
                        .collect::<Vec<_>>(),
                );
            }
            trace = trace.max((x[..g.n].iter().sum::<f64>() - 1.).abs());
            let rho = expand_density(&g.coords, &x, g.n);
            let mut block = Vec::new();
            for &i in &support {
                for &j in &support {
                    block.push(rho[i * g.n + j]);
                }
            }
            psd = psd.max(positivity(&self.lap, &block, support.len())?);
            scattering.push(scatter_rate(&g.scattering, &rho, g.n));
            populations.push(x[..g.n].to_vec());
            if options.return_density {
                density_real.push(rho.iter().map(|z| z.re).collect::<Vec<_>>());
                density_imag.push(rho.iter().map(|z| z.im).collect::<Vec<_>>());
            }
        }
        if trace.max(psd).max(initial_error) > 5e-6 {
            return Err(format!("Dense Pade density verification failed: trace {trace:e}, positivity {psd:e}, initial {initial_error:e}"));
        }
        let status = if !options.include_asymptotic {
            "not-requested"
        } else if stationary.is_some() {
            "stationary"
        } else {
            "not-converged"
        };
        let states: Vec<_> = if model.states.len() == g.n {
            model.states.clone()
        } else {
            (0..g.n).map(|i| i.to_string()).collect()
        };
        Ok(
            json!({"states":states,"timesUs":times,"populations":populations,"scatteringRatePerUs":scattering,
            "densityReal":if options.return_density{Some(density_real)}else{None},"densityImag":if options.return_density{Some(density_imag)}else{None},
            "asymptoticPopulations":stationary.as_ref().map(|v|v["asymptoticPopulations"].clone()),"asymptoticScatteringRatePerUs":stationary.as_ref().map(|v|v["asymptoticScatteringRatePerUs"].clone()),"asymptoticStatus":status,
            "asymptoticDensityReal":stationary.as_ref().map(|v|v["asymptoticDensityReal"].clone()),"asymptoticDensityImag":stationary.as_ref().map(|v|v["asymptoticDensityImag"].clone()),
            "diagnostics":{"engine":"rust-lapack","method":"Full-density scaling-and-squaring Pade [13/13] exponential","fallbackReason":reason,
                "liouvilleDimension":g.n*g.n,"reachableDimension":inds.len(),"physicalHilbertDimension":support.len(),"elapsedMs":started.elapsed().as_secs_f64()*1000.,"constructionMs":construction_ms,
                "traceError":trace,"positivityError":psd,"hermiticityError":0.,"initialStateError":initial_error,"trajectoryErrorEstimate":risk,"linearSolveBackwardError":backward,
                "asymptoticStatus":status,"stationaryProofCached":stationary.as_ref().map(|v|v["diagnostics"]["stationaryProofCached"].clone()),"residual":stationary.as_ref().map(|v|v["diagnostics"]["residual"].clone()),"populationResidual":stationary.as_ref().map(|v|v["diagnostics"]["populationResidual"].clone()),
                "steadyMethod":if stationary.is_some(){"Verified trace-constrained LU on the full invariant Hilbert support"}else{"No stationary projection asserted for this defective spectrum"}}}),
        )
    }
    fn solve(&mut self, model: &Model, options: &Options) -> Result<Value, String> {
        let started = Instant::now();
        options.validate_population_tolerance()?;
        if !options.duration_us.is_finite()
            || options.duration_us <= 0.
        {
            return Err("durationUs must be positive and finite".into());
        }
        if options.samples < 2 || options.samples > 2001 {
            return Err("samples must be in [2,2001]".into());
        }
        if !model.drives.is_empty() {
            return Err("Driven generators are solved through modelSchema rb87-full/2; generic-lindblad/1 accepts static generators only".into());
        }
        if model.time_dependent {
            return Err("Time-dependent input needs explicit harmonic drives; no unsupported term may be silently omitted.".into());
        }
        let g = build_generator(model)?;
        let initial_rho = expand_density(&g.coords, &g.initial, g.n);
        if positivity(&self.lap, &initial_rho, g.n)? > 1e-10 {
            return Err("initial density is not positive semidefinite".into());
        }
        let physical_support = hilbert_support(model)?;
        let d = g.n * g.n;
        let inds = reachable(&g.matrix, &g.initial);
        let r = inds.len();
        let mut a = vec![0.; r * r];
        let mut initial = vec![0.; r];
        let mut positions = vec![usize::MAX; d];
        for (i, &global) in inds.iter().enumerate() {
            positions[global] = i;
        }
        for (j, &col) in inds.iter().enumerate() {
            initial[j] = g.initial[col];
            for &(row, value) in &g.matrix[col] {
                let i = positions[row];
                if i != usize::MAX {
                    a[i + j * r] = value;
                }
            }
        }
        let construction_ms = started.elapsed().as_secs_f64() * 1000.;
        // The full trace-constrained stationary equation is a much cheaper
        // path for level colouring. A nonsingular, well-conditioned system
        // proves a unique stationary density in the reachable subspace.
        // Singular/nonunique cases continue to initial-state mode projection.
        let unique_stationary = if options.include_asymptotic {
            self.stationary_only(model, &g, started, construction_ms)?
        } else {
            None
        };
        if options.steady_only {
            if let Some(result) = &unique_stationary {
                let mut result = result.clone();
                if options.return_density {
                    result["densityReal"] =
                        json!([initial_rho.iter().map(|z| z.re).collect::<Vec<_>>()]);
                    result["densityImag"] =
                        json!([initial_rho.iter().map(|z| z.im).collect::<Vec<_>>()]);
                }
                return Ok(result);
            }
        }
        let key = self.cache.iter().position(|c| {
            c.dimension == r
                && c.hilbert_dimension == g.n
                && c.physical_support_dimension == physical_support.len()
                && c.coordinate_indices == inds
                && c.matrix == a
        });
        let cached = key.is_some();
        let entry = if let Some(k) = key {
            self.cache.remove(k).unwrap()
        } else {
            let spectrum = match spectral(&self.lap, &a, r) {
                Ok(value) => value,
                Err(error) => {
                    return self.pade_constant(
                        model,
                        &g,
                        &a,
                        &inds,
                        &initial,
                        options,
                        &unique_stationary,
                        started,
                        construction_ms,
                        &error,
                    )
                }
            };
            if r <= 64 && spectrum.condition > 1e7 {
                return self.pade_constant(model,&g,&a,&inds,&initial,options,&unique_stationary,started,construction_ms,"Nearly defective small-system eigenbasis; using the matrix exponential directly");
            }
            let certificate = self.spectral_certificate(&a, &spectrum, &inds, g.n);
            // Numerical partition only: the complete slow invariant block is
            // refined, and every fast mode remains in Prepared::apply.
            let refined = if spectrum.norm > 1e5 || r <= 64 {
                match cluster_action::prepare_dense(&self.lap, &a, r, 1e5) {
                    Ok(value) => Some(value),
                    Err(error) => {
                        return self.pade_constant(
                            model,
                            &g,
                            &a,
                            &inds,
                            &initial,
                            options,
                            &unique_stationary,
                            started,
                            construction_ms,
                            &error,
                        )
                    }
                }
            } else {
                None
            };
            let mut modal_certificate = None;
            let mut modal_certificate_failure = None;
            if let Some(prepared) = &refined {
                let metric: Vec<_> = inds
                    .iter()
                    .map(|&i| if i < g.n { 1. } else { 2. })
                    .collect();
                match prepared.modal_basis(&self.lap).and_then(|modal| {
                    defect_bound::prepare(
                        &self.lap,
                        &a,
                        modal,
                        &metric,
                        (physical_support.len() as f64).sqrt(),
                    )
                }) {
                    Ok(value) => modal_certificate = Some(value),
                    Err(error) => modal_certificate_failure = Some(error),
                }
            }
            Cached {
                matrix: a.clone(),
                dimension: r,
                hilbert_dimension: g.n,
                physical_support_dimension: physical_support.len(),
                coordinate_indices: inds.clone(),
                spectrum,
                certificate,
                refined,
                modal_certificate,
                modal_certificate_failure,
            }
        };
        self.cache.push_back(entry);
        while self.cache.len() > 3 {
            self.cache.pop_front();
        }
        let spec = &self.cache.back().unwrap().spectrum;
        let coefficients = self
            .lap
            .solve_factored(&spec.lu, &spec.pivots, &initial, r)?;
        let eig_error = spec.residual.max(f64::EPSILON * spec.norm * 2.);
        let zero_tol = (8. * eig_error).max(2e-12);
        let zero: Vec<_> = spec
            .wr
            .iter()
            .zip(&spec.wi)
            .map(|(&re, &im)| re.hypot(im) <= zero_tol)
            .collect();
        if spec.wr.iter().any(|&x| x > 1e-8_f64.max(16. * eig_error)) {
            return Err("Positive Lindblad eigenvalue exceeds numerical error".into());
        }
        let times: Vec<_> = if options.steady_only {
            vec![0.]
        } else {
            (0..options.samples)
                .map(|i| options.duration_us * i as f64 / (options.samples - 1) as f64)
                .collect()
        };
        let mut modes = vec![0.; r * times.len()];
        for (tidx, &t) in times.iter().enumerate() {
            let mut j = 0;
            while j < r {
                if spec.wi[j] > 0. {
                    if coefficients[j] == 0. && coefficients[j + 1] == 0. {
                        j += 2;
                        continue;
                    }
                    // A spectral resolution threshold is a reliability test,
                    // never permission to turn a real slow decay into zero.
                    let decay = (spec.wr[j] * t).exp();
                    let (s, c) = (spec.wi[j] * t).sin_cos();
                    modes[j + tidx * r] = decay * (c * coefficients[j] + s * coefficients[j + 1]);
                    modes[j + 1 + tidx * r] =
                        decay * (-s * coefficients[j] + c * coefficients[j + 1]);
                    j += 2;
                } else {
                    if coefficients[j] != 0. {
                        modes[j + tidx * r] = coefficients[j] * (spec.wr[j] * t).exp();
                    }
                    j += 1;
                }
            }
        }
        let raw_evolved = self.lap.multiply(&spec.vectors, &modes, r, times.len(), r);
        let (spectral_defect_bound, spectral_roundoff_estimate) = self.spectral_defect_bound(
            spec,
            &coefficients,
            &self.cache.back().unwrap().certificate,
            g.n,
            *times.last().unwrap(),
        );
        let raw_initial_trace_norm_bound = (g.n as f64
            * inds
                .iter()
                .enumerate()
                .map(|(i, &global)| {
                    let error = raw_evolved[i] - initial[i];
                    if global < g.n {
                        error * error
                    } else {
                        2. * error * error
                    }
                })
                .sum::<f64>())
        .sqrt();
        let refined = self.cache.back().unwrap().refined.as_ref();
        let modal_certificate = self.cache.back().unwrap().modal_certificate.as_ref();
        let modal_action = if let Some(prepared) = modal_certificate {
            Some(prepared.apply(&self.lap, &initial, &times)?)
        } else {
            None
        };
        let refined_action = if modal_action.is_none() {
            if let Some(prepared) = refined {
                Some(prepared.apply(&self.lap, &initial, &times)?)
            } else {
                None
            }
        } else {
            None
        };
        let mut refined_difference_bound = 0_f64;
        let corrected_states = modal_action
            .as_ref()
            .map(|action| &action.states)
            .or_else(|| refined_action.as_ref().map(|action| &action.states));
        let evolved = if let Some(states) = corrected_states {
            for (t, state) in states.iter().enumerate() {
                let squared = inds
                    .iter()
                    .enumerate()
                    .map(|(i, &global)| {
                        let delta = state[i] - raw_evolved[i + t * r];
                        if global < g.n {
                            delta * delta
                        } else {
                            2. * delta * delta
                        }
                    })
                    .sum::<f64>();
                refined_difference_bound = if squared.is_finite() {
                    refined_difference_bound.max((g.n as f64 * squared).sqrt())
                } else {
                    f64::INFINITY
                };
            }
            states.iter().flatten().copied().collect::<Vec<_>>()
        } else {
            raw_evolved
        };
        let spectral_ms = started.elapsed().as_secs_f64() * 1000. - construction_ms;
        let mut populations = Vec::with_capacity(times.len());
        let mut scattering = Vec::with_capacity(times.len());
        let mut density_real = Vec::new();
        let mut density_imag = Vec::new();
        let mut trace_error: f64 = 0.;
        let mut positivity_error: f64 = 0.;
        let mut initial_error: f64 = 0.;
        let mut initial_trace_norm_bound: f64 = 0.;
        for t in 0..times.len() {
            let mut x = vec![0.; d];
            for (i, &index) in inds.iter().enumerate() {
                x[index] = evolved[i + t * r];
            }
            if t == 0 {
                initial_error = max_abs(
                    &x.iter()
                        .zip(&g.initial)
                        .map(|(a, b)| a - b)
                        .collect::<Vec<_>>(),
                );
                initial_trace_norm_bound = (g.n as f64
                    * x.iter()
                        .zip(&g.initial)
                        .enumerate()
                        .map(|(i, (a, b))| {
                            if i < g.n {
                                (a - b).powi(2)
                            } else {
                                2. * (a - b).powi(2)
                            }
                        })
                        .sum::<f64>())
                .sqrt();
            }
            trace_error = trace_error.max((x[..g.n].iter().sum::<f64>() - 1.).abs());
            let rho = expand_density(&g.coords, &x, g.n);
            // Outside the structurally invariant Hilbert support, rho is
            // identically zero. Test the exact nonzero block rather than
            // diagonalising dozens of padded zeros at every sample.
            let psd = if physical_support.len() < g.n {
                let mut block = Vec::with_capacity(physical_support.len().pow(2));
                for &i in &physical_support {
                    for &j in &physical_support {
                        block.push(rho[i * g.n + j]);
                    }
                }
                positivity(&self.lap, &block, physical_support.len())?
            } else {
                positivity(&self.lap, &rho, g.n)?
            };
            positivity_error = positivity_error.max(psd);
            if options.return_density {
                density_real.push(rho.iter().map(|z| z.re).collect::<Vec<_>>());
                density_imag.push(rho.iter().map(|z| z.im).collect::<Vec<_>>());
            }
            scattering.push(scatter_rate(&g.scattering, &rho, g.n));
            populations.push(x[..g.n].to_vec());
        }
        if trace_error.max(positivity_error).max(initial_error) > 5e-6 {
            return Err(format!("Density verification failed: trace {trace_error:e}, positivity {positivity_error:e}, initial {initial_error:e}"));
        }
        // The raw spectral defect bounds its own trajectory, not the refined
        // one. Retain the conservative triangle certificate at every returned
        // sample; independent cluster/gauge checks assess the tighter actual
        // numerical accuracy without relabelling that as a universal bound.
        let raw_triangle_bound = raw_initial_trace_norm_bound
            + spectral_defect_bound
            + spectral_roundoff_estimate
            + refined_difference_bound;
        let modal_bound = modal_action
            .as_ref()
            .map(|action| action.bound.total)
            .filter(|x| x.is_finite());
        let trajectory_error_bound = modal_bound.map_or(raw_triangle_bound, |bound| {
            if raw_triangle_bound.is_finite() {
                bound.min(raw_triangle_bound)
            } else {
                bound
            }
        });
        if !trajectory_error_bound.is_finite() || trajectory_error_bound > 5e-6 {
            let reason=format!("Spectral trajectory error is unresolved: selected envelope {trajectory_error_bound:e}, frequency-aware modal envelope {modal_bound:?}, raw integrated density defect {spectral_defect_bound:e}, floating estimate {spectral_roundoff_estimate:e}, initial bound {raw_initial_trace_norm_bound:e}, refined comparison {refined_difference_bound:e}; no eigenvalues were replaced by zero");
            return self.pade_constant(
                model,
                &g,
                &a,
                &inds,
                &initial,
                options,
                &unique_stationary,
                started,
                construction_ms,
                &reason,
            );
        }
        let nullity = zero.iter().filter(|&&x| x).count();
        let weak = (0..r).any(|i| !zero[i] && spec.wr[i].hypot(spec.wi[i]) <= 10. * zero_tol);
        let certified_zero_modes = stationary_structure::certified_zero_modes(&g, &inds);
        let uncertain_nullity = nullity > 1 && certified_zero_modes < nullity;
        let active: Vec<_> = (0..r)
            .map(|i| {
                if spec.wi[i] > 0. {
                    coefficients[i].hypot(coefficients[i + 1]) > 1e-8
                } else if spec.wi[i] < 0. {
                    coefficients[i].hypot(coefficients[i - 1]) > 1e-8
                } else {
                    coefficients[i].abs() > 1e-8
                }
            })
            .collect();
        let persistent: Vec<_> = (0..r)
            .map(|i| !zero[i] && spec.wr[i] >= -zero_tol && active[i])
            .collect();
        let any_persistent = persistent.iter().any(|&b| b);
        let mut persistent_population = false;
        for j in 0..r {
            if persistent[j] {
                for (i, &global) in inds.iter().enumerate() {
                    if global < g.n && (spec.vectors[i + j * r] * coefficients[j]).abs() > 1e-8 {
                        persistent_population = true;
                    }
                }
            }
        }
        let gap = (0..r)
            .filter(|&i| spec.wr[i] < -zero_tol && active[i])
            .map(|i| -spec.wr[i])
            .reduce(f64::min);
        let mut asym: Option<Vec<f64>> = None;
        let mut asym_density_real: Option<Vec<f64>> = None;
        let mut asym_density_imag: Option<Vec<f64>> = None;
        let mut asym_scatter = None;
        let mut residual = None;
        let mut population_residual = None;
        let mut status = "not-requested";
        if options.include_asymptotic {
            if let Some(unique) = &unique_stationary {
                // The same independently verified full-density stationary solve
                // is used whether or not finite-time samples were requested.
                // THz coherences must not change the nullity classification of
                // a trace-constrained LU system whose uniqueness was verified.
                status = "stationary";
                asym = Some(
                    unique["asymptoticPopulations"]
                        .as_array()
                        .ok_or("Invalid internal stationary result")?
                        .iter()
                        .map(|v| v.as_f64().unwrap())
                        .collect(),
                );
                asym_scatter = unique["asymptoticScatteringRatePerUs"].as_f64();
                asym_density_real = unique["asymptoticDensityReal"]
                    .as_array()
                    .map(|row| row.iter().map(|v| v.as_f64().unwrap()).collect());
                asym_density_imag = unique["asymptoticDensityImag"]
                    .as_array()
                    .map(|row| row.iter().map(|v| v.as_f64().unwrap()).collect());
                residual = unique["diagnostics"]["residual"].as_f64();
                population_residual = unique["diagnostics"]["populationResidual"].as_f64();
            } else {
                status = if weak || uncertain_nullity {
                    "unresolved-slow-modes"
                } else if persistent_population {
                    "nonstationary-limit"
                } else if nullity == 0 {
                    "not-converged"
                } else {
                    "stationary"
                };
                if status == "stationary" {
                    let mut steady = vec![0.; r];
                    for j in 0..r {
                        if zero[j] {
                            for i in 0..r {
                                steady[i] += spec.vectors[i + j * r] * coefficients[j];
                            }
                        }
                    }
                    if nullity == 1 {
                        let trace_row = inds
                            .iter()
                            .position(|&i| i < g.n)
                            .ok_or("No population coordinate in reachable space")?;
                        let mut constrained = a.clone();
                        let mut rhs = vec![0.; r];
                        rhs[trace_row] = 1.;
                        for (col, &index) in inds.iter().enumerate() {
                            constrained[trace_row + col * r] = if index < g.n { 1. } else { 0. };
                        }
                        steady = self.lap.solve(&constrained, &rhs, r)?;
                    }
                    let remainder = self.lap.multiply(&a, &steady, r, 1, r);
                    let steady_residual = remainder.iter().map(|v| v * v).sum::<f64>().sqrt();
                    let pop_res = inds
                        .iter()
                        .enumerate()
                        .filter(|(_, i)| **i < g.n)
                        .map(|(i, _)| remainder[i].abs())
                        .fold(0., f64::max);
                    residual = Some(steady_residual);
                    population_residual = Some(pop_res);
                    let mut x = vec![0.; d];
                    for (i, &index) in inds.iter().enumerate() {
                        x[index] = steady[i];
                    }
                    let rho = expand_density(&g.coords, &x, g.n);
                    let tr = (x[..g.n].iter().sum::<f64>() - 1.).abs();
                    let psd = positivity(&self.lap, &rho, g.n)?;
                    if tr.max(psd) > 2e-6 || steady_residual > 1e-6 {
                        status = "not-converged";
                    } else {
                        status = if any_persistent {
                            "population-stationary"
                        } else if nullity == 1 {
                            "stationary"
                        } else {
                            "nonunique-initial-dependent"
                        };
                        asym = Some(x[..g.n].to_vec());
                        asym_density_real = Some(rho.iter().map(|z| z.re).collect());
                        asym_density_imag = Some(rho.iter().map(|z| z.im).collect());
                        asym_scatter = Some(scatter_rate(&g.scattering, &rho, g.n));
                    }
                }
            }
        }
        let elapsed = started.elapsed().as_secs_f64() * 1000.;
        let states: Vec<String> = if model.states.len() == g.n {
            model.states.clone()
        } else {
            (0..g.n).map(|i| i.to_string()).collect()
        };
        let mut diagnostics = json!({"method":if refined.is_some(){"Rust / LAPACK full-density Lindblad invariant-cluster propagation"}else{"Rust / LAPACK full-density Lindblad spectral propagation"}, "liouvilleDimension":d,"reachableDimension":r,"physicalHilbertDimension":physical_support.len(),
                "engine":"rust-lapack","eigensystemCached":cached,"elapsedMs":elapsed,"constructionMs":construction_ms,"spectralMs":spectral_ms,
                "traceError":trace_error,"positivityError":positivity_error,"hermiticityError":0.,"initialStateError":initial_error,
                "eigensystemResidual":spec.residual,"eigenvectorCondition":spec.condition,"eigenvalueResolutionPerUs":zero_tol,
                "integratedDensityDefectBound":spectral_defect_bound,"spectralRoundoffEstimate":spectral_roundoff_estimate,"initialTraceNormBound":initial_trace_norm_bound,"trajectoryErrorEstimate":trajectory_error_bound});
        diagnostics["invariantCluster"] = if let Some(prepared) = refined {
            json!({"dimension":prepared.diagnostics.cluster_dimension,"partitionRadPerUs":prepared.diagnostics.imaginary_partition_limit,"gapRadPerUs":prepared.diagnostics.imaginary_cluster_gap,"subspaceResidual":prepared.diagnostics.subspace_residual_max,"preparationMs":prepared.diagnostics.preparation_ms,"cached":cached,"applicationMs":modal_action.as_ref().map(|v|v.elapsed_ms).or_else(||refined_action.as_ref().map(|v|v.elapsed_ms)),"initialReconstructionError":if modal_action.is_some(){Some(initial_error)}else{refined_action.as_ref().map(|v|v.initial_error)},"usedPade":refined_action.as_ref().map(|v|v.cluster_used_pade),"rawComparisonTraceNormBound":refined_difference_bound,"rawInitialTraceNormBound":raw_initial_trace_norm_bound,"rawTriangleBound":raw_triangle_bound.is_finite().then_some(raw_triangle_bound),"rawTriangleAvailable":raw_triangle_bound.is_finite()})
        } else {
            Value::Null
        };
        diagnostics["modalDefect"] = if let (Some(prepared), Some(action)) =
            (modal_certificate, &modal_action)
        {
            let b = &action.bound;
            json!({"defectNormUpper":prepared.diagnostics.defect_norm_upper,"inverseResidualBound":prepared.diagnostics.inverse_residual_bound,"preparationMs":prepared.diagnostics.preparation_ms,"cached":cached,"initialReconstruction":b.initial_reconstruction,"firstOrder":b.first_order,"higherOrder":b.higher_order,"globalHigherOrder":b.global_higher_order,"comparisonThreshold":b.comparison_threshold,"comparisonRoundoff":b.comparison_roundoff,"inverseAndProductUncertainty":b.inverse_and_product_uncertainty,"floatingEvaluation":b.floating_evaluation,"total":b.total,"selected":b.total<=raw_triangle_bound})
        } else {
            json!({"unavailable":self.cache.back().unwrap().modal_certificate_failure})
        };
        let stationary_diagnostics = json!({
                "asymptoticStatus":status,"residual":if status=="population-stationary" {None}else {residual},"populationResidual":population_residual,
                "stationaryProjectionResidual":residual,"zeroModeCount":if unique_stationary.is_some(){1}else{nullity},
                "spectralNearZeroCount":nullity,"structuralZeroModeCount":certified_zero_modes,"stationarySystemCondition":unique_stationary.as_ref().map(|v|v["diagnostics"]["stationarySystemCondition"].clone()),
                "stationaryPhysicalHilbertDimension":unique_stationary.as_ref().map(|v|v["diagnostics"]["stationaryPhysicalHilbertDimension"].clone()),"stationaryOperatorDimension":unique_stationary.as_ref().map(|v|v["diagnostics"]["stationaryOperatorDimension"].clone()),
                "stationaryProofCached":unique_stationary.as_ref().map(|v|v["diagnostics"]["stationaryProofCached"].clone()),
                "steadyMethod":if unique_stationary.is_some(){"Independent full-density trace-constrained LU, identical to steadyOnly"}else{"Initial-state spectral stationary projection"},
                "densityLimitExists":if unique_stationary.is_some(){Some(true)}else if status=="unresolved-slow-modes"||status=="not-converged"{None}else{Some(!any_persistent)},
                "uniquenessScope":if unique_stationary.is_some(){"full Hermitian operator algebra on the invariant physical Hilbert support"}else{"initial-state spectral projection on reachable Hermitian coordinates"},"smallestRelaxationRatePerUs":if weak||uncertain_nullity{None}else{gap},"estimatedSlowTimeUs":if weak||uncertain_nullity{None}else{gap.map(|x|1./x)},
                "asymptoticReason":if unique_stationary.is_some(){"Unique full-density Lindblad stationary solution, independently verified by trace-constrained LU, residual and positivity."}else{match status {"population-stationary"=>"Populations converge, but undamped coherences prevent a stationary full density limit.","stationary"|"nonunique-initial-dependent"=>"Full Lindblad stationary projection from the specified initial density.",_=>"No trustworthy stationary population limit is asserted."}}});
        diagnostics
            .as_object_mut()
            .unwrap()
            .extend(stationary_diagnostics.as_object().unwrap().clone());
        Ok(
            json!({"timesUs":times,"states":states,"populations":populations,"scatteringRatePerUs":scattering,
            "densityReal":if options.return_density{Some(density_real)}else{None},"densityImag":if options.return_density{Some(density_imag)}else{None},
            "asymptoticPopulations":asym,"asymptoticScatteringRatePerUs":asym_scatter,"asymptoticStatus":status,
            "asymptoticDensityReal":asym_density_real,"asymptoticDensityImag":asym_density_imag,"diagnostics":diagnostics}),
        )
    }
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
        if let Some(index) = self.stationary_cache.iter().position(|entry| {
            entry.hilbert_dimension == g.n && entry.indices == inds && entry.matrix == a
        }) {
            let mut entry = self.stationary_cache.remove(index).unwrap();
            let initial = expand_density(&g.coords, &g.initial, g.n);
            entry.result["populations"] = json!([g.initial[..g.n].to_vec()]);
            entry.result["scatteringRatePerUs"] =
                json!([scatter_rate(&g.scattering, &initial, g.n)]);
            // Identical Lindblad generators can have different jump counting
            // observables (e.g. C -> {C, alpha I}); recompute both scattering
            // readouts rather than caching them along with the stationary rho.
            entry.result["asymptoticScatteringRatePerUs"] =
                json!(scatter_rate(&g.scattering, &entry.density, g.n));
            entry.result["states"] = json!(if model.states.len() == g.n {
                model.states.clone()
            } else {
                (0..g.n).map(|i| i.to_string()).collect()
            });
            entry.result["diagnostics"]["stationaryProofCached"] = json!(true);
            entry.result["diagnostics"]["constructionMs"] = json!(construction_ms);
            entry.result["diagnostics"]["elapsedMs"] =
                json!(started.elapsed().as_secs_f64() * 1000.);
            let result = entry.result.clone();
            self.stationary_cache.push_back(entry);
            return Ok(Some(result));
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
        let result = json!({"timesUs":[0.],"states":states,"populations":[g.initial[..g.n].to_vec()],
            "scatteringRatePerUs":[scatter_rate(&g.scattering,&initial_rho,g.n)],"asymptoticPopulations":x[..g.n].to_vec(),
            "asymptoticDensityReal":rho.iter().map(|z|z.re).collect::<Vec<_>>(),"asymptoticDensityImag":rho.iter().map(|z|z.im).collect::<Vec<_>>(),
            "asymptoticScatteringRatePerUs":scatter_rate(&g.scattering,&rho,g.n),"asymptoticStatus":"stationary",
            "diagnostics":{"method":"Rust / LAPACK full-density trace-constrained Lindblad stationary solve","engine":"rust-lapack",
                "liouvilleDimension":g.n*g.n,"reachableDimension":r,"stationaryPhysicalHilbertDimension":support.len(),"stationaryOperatorDimension":r,"stationaryOnly":true,"elapsedMs":started.elapsed().as_secs_f64()*1000.,
                "constructionMs":construction_ms,"traceError":trace_error,"positivityError":positivity_error,"hermiticityError":0.,
                "stationarySystemCondition":condition,"stationaryProofCached":false,"residual":residual,"populationResidual":population_residual,"stationaryProjectionResidual":residual,
                "asymptoticStatus":"stationary","zeroModeCount":1,"densityLimitExists":true,
                "uniquenessScope":"full Hermitian operator algebra on the invariant physical Hilbert support",
                "asymptoticReason":"Unique full-density Lindblad stationary solution, verified by matrix residual and positivity."}}
        );
        self.stationary_cache.push_back(CachedStationary {
            matrix: a,
            indices: inds,
            hilbert_dimension: g.n,
            result: result.clone(),
            density: rho,
        });
        while self.stationary_cache.len() > 3 {
            self.stationary_cache.pop_front();
        }
        Ok(Some(result))
    }
}

/// Numerical method selection only: both routes retain the same complete
/// Hamiltonian and collapse operators and enforce their own convergence gates.
/// Very long trajectories avoid repeatedly refining a period discretization
/// over millions of cycles. No laser or density coordinate is discarded.
fn prefer_full_period(dimension:usize,frequencies:&[f64],duration_us:f64,steady_only:bool)->bool{
    if steady_only||dimension!=32||frequencies.len()!=1||!frequencies[0].is_finite()||frequencies[0]<=0.
        ||!duration_us.is_finite()||duration_us<=0.{return false;}
    let period=std::f64::consts::TAU/frequencies[0];
    // A GHz repump can execute many periods during a short physical interval.
    // Its period count alone is not a reason to reject this numerical route.
    period<=0.02&&duration_us<=100.
}
fn full_period_requested(model:&Model,options:&Options)->bool{
    match std::env::var("ATOM_FULL_PERIOD").as_deref(){
        Ok("1")=>crate::full_period::supports(model),
        Ok("auto")=>harmonics::model_frequency_basis(model,16).is_ok_and(|basis|basis.exact&&basis.rounded_relations==0&&prefer_full_period(model.dimension,&basis.frequencies,options.duration_us,options.steady_only)),
        _=>false,
    }
}
#[cfg(test)]
mod period_routing_tests{
    use super::*;
    #[test]
    fn method_selection_retains_long_lowbeat_and_multifrequency_fallbacks(){
        let beat=std::f64::consts::TAU*100.;
        assert!(prefer_full_period(32,&[beat],20.,false));
        assert!(prefer_full_period(32,&[std::f64::consts::TAU*6568.],20.,false));
        assert!(!prefer_full_period(32,&[beat],100.01,false));
        assert!(!prefer_full_period(32,&[beat],100000.,false));
        assert!(!prefer_full_period(32,&[std::f64::consts::TAU*1.25],20.,false));
        assert!(!prefer_full_period(32,&[beat,beat*1.7],20.,false));
        assert!(!prefer_full_period(32,&[beat],20.,true));
        assert!(!prefer_full_period(32,&[f64::NAN],20.,false));
    }
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
fn full_period_result(lap:&Lapack,model:&Model,options:&Options)->Result<Value,String>{
    let started=Instant::now();
    let mut result=if options.include_asymptotic{
        let value=floquet::solve_full(lap,model,&Options{steady_only:true,..*options})?;
        publish_mean(value.clone());value
    }else{json!({"asymptoticPopulations":null,"asymptoticStatus":"not-requested"})};
    let mean_ms=started.elapsed().as_secs_f64()*1000.;
    let report=full_period::solve(lap,model,options)?;
    let g=build_generator(model)?;
    let densities:Vec<_>=report.states.iter().map(|state|expand_density(&g.coords,state,model.dimension)).collect();
    result["timesUs"]=json!(report.times);result["states"]=json!(model.states);
    result["populations"]=json!(report.states.iter().map(|x|x[..model.dimension].to_vec()).collect::<Vec<_>>());
    result["scatteringRatePerUs"]=json!(densities.iter().map(|rho|scatter_rate(&g.scattering,rho,model.dimension)).collect::<Vec<_>>());
    if options.return_density{result["densityReal"]=json!(densities.iter().map(|rho|rho.iter().map(|c|c.re).collect::<Vec<_>>()).collect::<Vec<_>>());result["densityImag"]=json!(densities.iter().map(|rho|rho.iter().map(|c|c.im).collect::<Vec<_>>()).collect::<Vec<_>>());}
    let previous=result["diagnostics"].clone();result["diagnostics"]=report.diagnostics;result["diagnostics"]["stationaryDiagnostics"]=previous;
    result["diagnostics"]["stationarySolveMs"]=json!(mean_ms);result["diagnostics"]["stationaryOnly"]=json!(false);
    result["diagnostics"]["elapsedMs"]=json!(started.elapsed().as_secs_f64()*1000.);Ok(result)
}

/// Static rb87-full/2 mean: the verified trace-constrained full-density
/// stationary solve. Initial-dependent cases fall back to Solver::solve.
fn static_stationary(solver:&mut Solver,model:&Model,options:&Options)->Result<Value,String>{
    if model.time_dependent||!model.drives.is_empty(){return Err("Nonstatic effective Hamiltonian".into());}
    if !options.steady_only{return Err("Static finite-time trajectories are solved by the population core".into());}
    let started=Instant::now();
    let g=build_generator(model)?;
    let construction=started.elapsed().as_secs_f64()*1000.;
    solver.stationary_only(model,&g,started,construction)?.ok_or_else(||"Initial-dependent steady state needs spectral projection".into())
}

fn full_model_solve(solver:&mut Solver,model:&Model,options:&Options)->Result<Value,String>{
    options.validate_population_tolerance()?;
    if model.frequency_coordinates.is_none(){return Err("rb87-full/2 requires authoritative frequencyCoordinates metadata".into());}
    let basis=harmonics::model_frequency_basis(model,16)?;
    // Request-local phase metadata must precede EVERY possible progress path,
    // including the static mean without Fourier coefficients.
    PROGRESS_REQUEST.with(|context|{
        if let Some((_,schema,mapping))=context.borrow_mut().as_mut(){if schema==FULL_MODEL_SCHEMA{*mapping=basis.source_to_basis.clone();}}
    });
    #[cfg(feature="population-core")]
    if population_core_adapter::supports(model,options)&&!full_period_requested(model,options){
        return population_core_adapter::solve(solver,model,options);
    }
    let mut result=if !model.drives.is_empty(){
        if options.steady_only{floquet::solve_full(&solver.lap,model,options)}
        else if full_period_requested(model,options){full_period_result(&solver.lap,model,options)}
        else{Err("Finite-time driven trajectories are solved by the population core or the complete-period lane".into())}
    }else{
        let start=Instant::now();match static_stationary(solver,model,options){Ok(v)=>Ok(v),Err(reason)=>solver.solve(model,options).map(|mut v|{
            v["diagnostics"]["fastPathFallback"]=json!(reason);v["diagnostics"]["elapsedMs"]=json!(start.elapsed().as_secs_f64()*1000.);v
        })}
    }?;
    result["diagnostics"]["frequencyCoordinateSource"]=json!(basis.source());
    result["diagnostics"]["carrierSourceToBasis"]=json!(basis.source_to_basis);
    result["diagnostics"]["frequencyReconstructionErrors"]=json!(basis.reconstruction_errors);
    Ok(result)
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
                        // Each protocol request performs a fresh numerical calculation.
                        solver.cache.clear();
                        solver.stationary_cache.clear();
                        floquet::clear_request_results();
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
    fn modal_defect_case(
        lap: &Lapack,
        m: &Model,
    ) -> (Generator, Vec<usize>, Vec<f64>, defect_bound::Prepared) {
        let g = build_generator(m).unwrap();
        let inds = reachable(&g.matrix, &g.initial);
        let r = inds.len();
        let mut positions = vec![usize::MAX; g.n * g.n];
        for (i, &global) in inds.iter().enumerate() {
            positions[global] = i;
        }
        let mut a = vec![0.; r * r];
        for (j, &global) in inds.iter().enumerate() {
            for &(row, value) in &g.matrix[global] {
                if positions[row] != usize::MAX {
                    a[positions[row] + j * r] = value;
                }
            }
        }
        let initial = inds.iter().map(|&i| g.initial[i]).collect();
        let metric: Vec<_> = inds
            .iter()
            .map(|&i| if i < g.n { 1. } else { 2. })
            .collect();
        let refined = cluster_action::prepare_dense(lap, &a, r, 1e5).unwrap();
        let modal = refined.modal_basis(lap).unwrap();
        let rank = hilbert_support(m).unwrap().len();
        let certificate =
            defect_bound::prepare(lap, &a, modal, &metric, (rank as f64).sqrt()).unwrap();
        (g, inds, initial, certificate)
    }
    fn difference_trace_norm(lap: &Lapack, a: &[C], b: &[C], n: usize) -> f64 {
        let dim = 2 * n;
        let mut matrix = vec![0.; dim * dim];
        for i in 0..n {
            for j in 0..n {
                let z = a[i * n + j] - b[i * n + j];
                matrix[i + j * dim] = z.re;
                matrix[i + (j + n) * dim] = -z.im;
                matrix[i + n + j * dim] = z.im;
                matrix[i + n + (j + n) * dim] = z.re;
            }
        }
        lap.symmetric_eigenvalues(&matrix, dim)
            .unwrap()
            .iter()
            .map(|x| x.abs())
            .sum::<f64>()
            * 0.5
    }
    fn bound_json(b: &defect_bound::Bound) -> Value {
        json!({"timeUs":b.time,"total":b.total,"accepted":b.total.is_finite()&&b.total<=5e-6,"initial":b.initial_reconstruction,"firstOrder":b.first_order,"higherOrder":b.higher_order,"globalHigherOrder":b.global_higher_order,"comparisonThreshold":b.comparison_threshold,"comparisonRoundoff":b.comparison_roundoff,"inverseAndProductUncertainty":b.inverse_and_product_uncertainty,"floating":b.floating_evaluation,"coefficientNorm":b.coefficient_weighted_norm})
    }
    #[test]
    fn coherent_rabi_and_no_stationary_claim() {
        let mut s = Solver::new().unwrap();
        let m = model(2, vec![0., 1., 1., 0.], vec![], 0);
        let out = s
            .solve(
                &m,
                &Options {
                    duration_us: 2.,
                    samples: 41,
                    include_asymptotic: true,
                    ..Options::default()
                },
            )
            .unwrap();
        for i in 0..41 {
            let t = i as f64 / 20.;
            let p = out["populations"][i][1].as_f64().unwrap();
            assert!((p - t.sin().powi(2)).abs() < 1e-10);
        }
        assert_eq!(out["asymptoticStatus"], "nonstationary-limit");
        let only = s
            .solve(
                &m,
                &Options {
                    steady_only: true,
                    ..Options::default()
                },
            )
            .unwrap();
        assert_eq!(only["asymptoticStatus"], "nonstationary-limit");
        assert!(only["asymptoticPopulations"].is_null());
    }
    #[test]
    fn spontaneous_decay_analytic() {
        let mut s = Solver::new().unwrap();
        let c = Operator {
            real: vec![0., 2f64.sqrt(), 0., 0.],
            imag: vec![0.; 4],
        };
        let m = model(2, vec![0.; 4], vec![c], 1);
        let out = s
            .solve(
                &m,
                &Options {
                    duration_us: 3.,
                    samples: 31,
                    include_asymptotic: true,
                    ..Options::default()
                },
            )
            .unwrap();
        for i in 0..31 {
            let t = i as f64 / 10.;
            assert!((out["populations"][i][1].as_f64().unwrap() - (-2. * t).exp()).abs() < 1e-10);
        }
        assert!((out["asymptoticPopulations"][0].as_f64().unwrap() - 1.).abs() < 1e-10);
    }
    #[test]
    fn open_driven_three_level_pumps_to_dark_sink() {
        let mut s = Solver::new().unwrap();
        let mut h = vec![0.; 9];
        h[2] = 2.;
        h[6] = 2.;
        let mut c0 = vec![0.; 9];
        c0[2] = 1.;
        let mut c1 = vec![0.; 9];
        c1[5] = 1.;
        let m = model(
            3,
            h,
            vec![
                Operator {
                    real: c0,
                    imag: vec![0.; 9],
                },
                Operator {
                    real: c1,
                    imag: vec![0.; 9],
                },
            ],
            0,
        );
        let out = s
            .solve(
                &m,
                &Options {
                    duration_us: 40.,
                    samples: 41,
                    include_asymptotic: true,
                    ..Options::default()
                },
            )
            .unwrap();
        assert!(out["populations"][40][1].as_f64().unwrap() > 0.999999);
        assert!((out["asymptoticPopulations"][1].as_f64().unwrap() - 1.).abs() < 1e-10);
    }
    #[test]
    fn thermal_two_level_stationary_population() {
        let mut s = Solver::new().unwrap();
        let decay = Operator {
            real: vec![0., 2f64.sqrt(), 0., 0.],
            imag: vec![0.; 4],
        };
        let pump = Operator {
            real: vec![0., 0., 1., 0.],
            imag: vec![0.; 4],
        };
        let out = s
            .solve(
                &model(2, vec![0.; 4], vec![decay, pump], 0),
                &Options::default(),
            )
            .unwrap();
        assert!((out["asymptoticPopulations"][1].as_f64().unwrap() - 1. / 3.).abs() < 1e-10);
    }
    #[test]
    fn complex_hamiltonian_phase_and_density_input() {
        let mut s = Solver::new().unwrap();
        let mut m = model(2, vec![0.; 4], vec![], 0);
        m.h_imag = vec![0., -1., 1., 0.];
        let out = s
            .solve(
                &m,
                &Options {
                    duration_us: 1.,
                    samples: 11,
                    include_asymptotic: false,
                    ..Options::default()
                },
            )
            .unwrap();
        assert!((out["populations"][10][1].as_f64().unwrap() - 1f64.sin().powi(2)).abs() < 1e-10);
    }
    #[test]
    fn dark_sink_and_driven_cycle_keep_initial_dependent_stationary_weights() {
        let mut solver = Solver::new().unwrap();
        let mut h = vec![0.; 16]; h[6] = 1.; h[9] = 1.;
        let jump = |row: usize, col: usize, rate: f64| {
            let mut real = vec![0.; 16]; real[row * 4 + col] = rate.sqrt();
            Operator { real, imag: vec![0.; 16] }
        };
        let mut m = model(4, h, vec![jump(0,3,1.), jump(1,3,1.), jump(1,2,1.)], 3);
        for (initial_dark, expected_dark) in [(0., 0.5), (0.2, 0.6)] {
            let mut initial = vec![0.; 16]; initial[0] = initial_dark; initial[15] = 1.-initial_dark;
            m.initial_real = Some(initial); m.initial_imag = Some(vec![0.; 16]);
            let out = solver.solve(&m, &Options { steady_only:true, ..Options::default() }).unwrap();
            assert_eq!(out["asymptoticStatus"], "nonunique-initial-dependent");
            assert_eq!(out["diagnostics"]["structuralZeroModeCount"], 2);
            let p = out["asymptoticPopulations"].as_array().unwrap();
            assert!((p[0].as_f64().unwrap()-expected_dark).abs()<1e-10);
            assert!((p[2].as_f64().unwrap()-(1.-expected_dark)*4./9.).abs()<1e-10);
        }
        // Even a very small nonzero escape destroys the closed cycling block.
        // It must not be counted as an exact stationary mode by a tolerance.
        m.collapse.push(jump(0,2,1e-14));
        let g=build_generator(&m).unwrap(); let indices=reachable(&g.matrix,&g.initial);
        assert_eq!(stationary_structure::certified_zero_modes(&g,&indices),1);
    }
    #[test]
    fn nonunique_stationary_space_preserves_initial_mixture() {
        let mut s = Solver::new().unwrap();
        let mut m = model(2, vec![0.; 4], vec![], 0);
        m.initial_real = Some(vec![0.3, 0., 0., 0.7]);
        m.initial_imag = Some(vec![0.; 4]);
        let out = s
            .solve(
                &m,
                &Options {
                    steady_only: true,
                    ..Options::default()
                },
            )
            .unwrap();
        assert_eq!(out["asymptoticStatus"], "nonunique-initial-dependent");
        assert_eq!(out["asymptoticPopulations"], json!([0.3, 0.7]));
    }
    #[test]
    fn stationary_only_solves_full_density_without_spectrum() {
        let mut s = Solver::new().unwrap();
        let c = Operator {
            real: vec![0., 1., 0., 0.],
            imag: vec![0.; 4],
        };
        let out = s
            .solve(
                &model(2, vec![0., 1., 1., 0.], vec![c], 0),
                &Options {
                    steady_only: true,
                    ..Options::default()
                },
            )
            .unwrap();
        assert_eq!(out["diagnostics"]["stationaryOnly"], true);
        assert!((out["asymptoticPopulations"][1].as_f64().unwrap() - 4. / 9.).abs() < 1e-10);
        assert_eq!(out["timesUs"], json!([0.]));
    }
    #[test]
    fn time_dependent_input_is_rejected() {
        let mut s = Solver::new().unwrap();
        let mut m = model(2, vec![0.; 4], vec![], 0);
        m.time_dependent = true;
        assert!(s
            .solve(&m, &Options::default())
            .unwrap_err()
            .contains("explicit harmonic drives"));
    }
    #[test]
    fn hilbert_support_includes_no_jump_backaction_and_harmonic_connections() {
        // C maps |0> only to itself, but C†C couples |0> and |1>.
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
            cos: Operator {
                real: h,
                imag: vec![0.; 9],
            },
            sin: Operator {
                real: vec![0.; 9],
                imag: vec![0.; 9],
            },
        });
        assert_eq!(hilbert_support(&m).unwrap(), vec![0, 1, 2]);
    }
    #[test]
    fn thz_coherences_do_not_freeze_a_resolved_slow_population_decay() {
        // Tensor product of a 1e-8/us amplitude-damping qubit and a fast,
        // 4e7 rad/us qubit with 1/us decay. The population closed form is
        // independent of the optical phase, while the full density includes it.
        let gamma = 1e-8_f64;
        let mut h = vec![0.; 16];
        h[5] = 4e7;
        h[15] = 4e7;
        let mut slow = vec![0.; 16];
        slow[2] = gamma.sqrt();
        slow[7] = gamma.sqrt();
        let mut fast = vec![0.; 16];
        fast[1] = 1.;
        fast[11] = 1.;
        let mut m = model(
            4,
            h,
            vec![
                Operator {
                    real: slow,
                    imag: vec![0.; 16],
                },
                Operator {
                    real: fast,
                    imag: vec![0.; 16],
                },
            ],
            2,
        );
        let mut initial = vec![0.; 16];
        initial[10] = 0.5;
        initial[11] = 0.5;
        initial[14] = 0.5;
        initial[15] = 0.5;
        m.initial_real = Some(initial);
        m.initial_imag = Some(vec![0.; 16]);
        let mut solver = Solver::new().unwrap();
        let options = Options {
            duration_us: 1e6,
            samples: 21,
            ..Options::default()
        };
        let full = solver.solve(&m, &options).unwrap();
        let only = solver
            .solve(
                &m,
                &Options {
                    steady_only: true,
                    ..Options::default()
                },
            )
            .unwrap();
        assert_eq!(full["asymptoticStatus"], only["asymptoticStatus"]);
        assert_eq!(full["asymptoticPopulations"], only["asymptoticPopulations"]);
        for i in 0..21 {
            let t = options.duration_us * i as f64 / 20.;
            let s = (-gamma * t).exp();
            let f = 0.5 * (-t).exp();
            let expected = [(1. - s) * (1. - f), (1. - s) * f, s * (1. - f), s * f];
            for j in 0..4 {
                assert!((full["populations"][i][j].as_f64().unwrap() - expected[j]).abs() < 1e-10);
            }
        }
        assert!(
            full["diagnostics"]["trajectoryErrorEstimate"]
                .as_f64()
                .unwrap()
                < 5e-6
        );
    }
    #[test]
    fn constant_kernel_returns_full_density_with_correct_coherence_phase() {
        let mut solver = Solver::new().unwrap();
        let m = model(2, vec![0., 1., 1., 0.], vec![], 0);
        let out = solver
            .solve(
                &m,
                &Options {
                    duration_us: 1.,
                    samples: 21,
                    return_density: true,
                    ..Options::default()
                },
            )
            .unwrap();
        for i in 0..21 {
            let t = i as f64 / 20.;
            assert!((out["densityReal"][i][0].as_f64().unwrap() - t.cos().powi(2)).abs() < 1e-11);
            assert!(
                (out["densityImag"][i][1].as_f64().unwrap() - 0.5 * (2. * t).sin()).abs() < 1e-11
            );
            assert!(
                (out["densityImag"][i][2].as_f64().unwrap() + 0.5 * (2. * t).sin()).abs() < 1e-11
            );
        }
    }
    #[test]
    fn stationary_cache_preserves_current_initial_and_scattering_observable() {
        let mut solver = Solver::new().unwrap();
        let mut m = model(
            2,
            vec![0., 1., 1., 0.],
            vec![Operator {
                real: vec![0., 1., 0., 0.],
                imag: vec![0.; 4],
            }],
            0,
        );
        let options = Options {
            steady_only: true,
            return_density: true,
            ..Options::default()
        };
        let first = solver.solve(&m, &options).unwrap();
        m.initial_index = 1;
        m.states = vec!["g".into(), "e".into()];
        m.collapse.push(Operator {
            real: vec![2., 0., 0., 2.],
            imag: vec![0.; 4],
        });
        let second = solver.solve(&m, &options).unwrap();
        assert_eq!(second["diagnostics"]["stationaryProofCached"], true);
        assert_eq!(second["states"], json!(["g", "e"]));
        assert_eq!(second["populations"], json!([[0., 1.]]));
        assert_eq!(second["densityReal"], json!([[0., 0., 0., 1.]]));
        assert_eq!(
            first["asymptoticPopulations"],
            second["asymptoticPopulations"]
        );
        assert!(
            (second["asymptoticScatteringRatePerUs"].as_f64().unwrap()
                - first["asymptoticScatteringRatePerUs"].as_f64().unwrap()
                - 4.)
                .abs()
                < 1e-12
        );
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
            let mut m = model(2, vec![0.; 4], vec![], 0);
            m.initial_real = Some(real);
            m.initial_imag = Some(imag);
            assert!(solver.solve(&m, &Options::default()).is_err());
        }
    }
    #[test]
    fn critically_damped_full_density_uses_pade_without_deleting_jordan_dynamics() {
        let mut solver = Solver::new().unwrap();
        let m = model(
            2,
            vec![0., 0.5, 0.5, 0.],
            vec![Operator {
                real: vec![0., 2., 0., 0.],
                imag: vec![0.; 4],
            }],
            0,
        );
        let out = solver
            .solve(
                &m,
                &Options {
                    duration_us: 3.,
                    samples: 61,
                    return_density: true,
                    ..Options::default()
                },
            )
            .unwrap();
        assert!(out["diagnostics"]["method"]
            .as_str()
            .unwrap()
            .contains("Pade"));
        for i in 0..61 {
            let t = i as f64 / 20.;
            let excited = 1. / 18. - (-3. * t).exp() * (1. / 18. + t / 6.);
            let imaginary = 2. / 9. - (-3. * t).exp() * (2. / 9. + t / 6.);
            assert!((out["populations"][i][1].as_f64().unwrap() - excited).abs() < 2e-12);
            assert!((out["densityImag"][i][1].as_f64().unwrap() - imaginary).abs() < 2e-12);
        }
        assert!((out["asymptoticPopulations"][1].as_f64().unwrap() - 1. / 18.).abs() < 1e-12);
    }
    #[test]
    fn frequency_aware_bound_is_conservative_for_thz_and_slow_decay_true_density() {
        let lap = Lapack::load().unwrap();
        let gamma = 1e-8_f64;
        let omega = 4e7_f64;
        let mut h = vec![0.; 16];
        h[5] = omega;
        h[15] = omega;
        let mut slow = vec![0.; 16];
        slow[2] = gamma.sqrt();
        slow[7] = gamma.sqrt();
        let mut fast = vec![0.; 16];
        fast[1] = 1.;
        fast[11] = 1.;
        let mut m = model(
            4,
            h,
            vec![
                Operator {
                    real: slow,
                    imag: vec![0.; 16],
                },
                Operator {
                    real: fast,
                    imag: vec![0.; 16],
                },
            ],
            2,
        );
        let mut initial = vec![0.; 16];
        initial[10] = 0.5;
        initial[11] = 0.5;
        initial[14] = 0.5;
        initial[15] = 0.5;
        m.initial_real = Some(initial);
        m.initial_imag = Some(vec![0.; 16]);
        let (g, inds, initial, certificate) = modal_defect_case(&lap, &m);
        let mut records = Vec::new();
        for end in [20., 200., 2000., 1e6] {
            let mut times = vec![0., 0.013, 0.5, 1.07, 5., 20.];
            times.push(end);
            let result = certificate.apply(&lap, &initial, &times).unwrap();
            let mut worst = 0_f64;
            for (state, &t) in result.states.iter().zip(&times) {
                let mut x = vec![0.; 16];
                for (i, &global) in inds.iter().enumerate() {
                    x[global] = state[i];
                }
                let actual = expand_density(&g.coords, &x, 4);
                let s = (-gamma * t).exp();
                let excited = 0.5 * (-t).exp();
                let amp = 0.5 * (-0.5 * t).exp();
                let phase = omega * t;
                let tail = omega.mul_add(t, -phase);
                let (si, co) = phase.sin_cos();
                let (sl, cl) = tail.sin_cos();
                let off = C::new(amp * (co * cl - si * sl), amp * (si * cl + co * sl));
                let mut expected = vec![C::default(); 16];
                for (a, weight) in [1. - s, s].into_iter().enumerate() {
                    let i = 2 * a;
                    expected[i * 4 + i].re = weight * (1. - excited);
                    expected[(i + 1) * 4 + i + 1].re = weight * excited;
                    expected[i * 4 + i + 1] = off * weight;
                    expected[(i + 1) * 4 + i] = off.conj() * weight;
                }
                worst = worst.max(difference_trace_norm(&lap, &actual, &expected, 4));
            }
            assert!(
                worst <= result.bound.total,
                "T={end}, true error {worst:e}, envelope {}",
                result.bound.total
            );
            assert!(
                result.bound.total < 5e-6,
                "T={end}: conservative envelope {} is not yet useful",
                result.bound.total
            );
            records.push(json!({"bound":bound_json(&result.bound),"trueTraceNormError":worst}));
        }
        let folder = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../artifacts/v4/native-constant");
        std::fs::create_dir_all(&folder).unwrap();
        std::fs::write(folder.join("analytic-defect-bound.json"),serde_json::to_string_pretty(&json!({"cases":records,"defectNorm":certificate.diagnostics.defect_norm_upper,"inverseResidualBound":certificate.diagnostics.inverse_residual_bound})).unwrap()).unwrap();
    }
    #[test]
    #[ignore = "Generated full Rb87 constant-H fixture and performance audit"]
    fn constant_rb32_stationary_and_transient_consistency_audit() {
        let folder = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../artifacts/v4/native-constant");
        let cases: Value =
            serde_json::from_str(&std::fs::read_to_string(folder.join("cases.json")).unwrap())
                .unwrap();
        let mut report = Vec::new();
        let mut failures = Vec::new();
        let mut solver = Solver::new().unwrap();
        for case in cases.as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            let m: Model = serde_json::from_value(case["model"].clone()).unwrap();
            let duration = case["durationUs"].as_f64().unwrap();
            let only = solver.solve(
                &m,
                &Options {
                    steady_only: true,
                    ..Options::default()
                },
            );
            let full = solver.solve(
                &m,
                &Options {
                    duration_us: duration,
                    samples: 161,
                    ..Options::default()
                },
            );
            let mut delta = 0_f64;
            if let (Ok(a), Ok(b)) = (&only, &full) {
                if a["asymptoticStatus"] != b["asymptoticStatus"] {
                    failures.push(format!("{name}: steady status differs"));
                }
                if let (Some(x), Some(y)) = (
                    a["asymptoticPopulations"].as_array(),
                    b["asymptoticPopulations"].as_array(),
                ) {
                    for (x, y) in x.iter().zip(y) {
                        delta = delta.max((x.as_f64().unwrap() - y.as_f64().unwrap()).abs());
                    }
                    if delta > 1e-10 {
                        failures.push(format!("{name}: stationary delta {delta:e}"));
                    }
                }
                if name == "cycling-F2m2-F3m3" {
                    for row in b["populations"].as_array().unwrap() {
                        for (i, p) in row.as_array().unwrap().iter().enumerate() {
                            if !matches!(m.states[i].as_str(), "ground-2-2" | "D2-3-3")
                                && p.as_f64().unwrap().abs() > 1e-11
                            {
                                failures.push(format!(
                                    "{name}: cycling support leaked into {}",
                                    m.states[i]
                                ));
                            }
                        }
                    }
                }
                if name == "lights-off-D2-stretched" {
                    let e = m.initial_index;
                    let gamma = m
                        .collapse
                        .iter()
                        .map(|op| {
                            (0..m.dimension)
                                .map(|row| {
                                    op.real[row * m.dimension + e].powi(2)
                                        + op.imag[row * m.dimension + e].powi(2)
                                })
                                .sum::<f64>()
                        })
                        .sum::<f64>();
                    for (i, t) in b["timesUs"].as_array().unwrap().iter().enumerate() {
                        if (b["populations"][i][e].as_f64().unwrap()
                            - (-gamma * t.as_f64().unwrap()).exp())
                        .abs()
                            > 1e-11
                        {
                            failures.push(format!("{name}: analytic decay differs"));
                        }
                    }
                }
                if matches!(name, "open-F2-F2" | "F2-F3-linear") {
                    let f1 = m
                        .states
                        .iter()
                        .enumerate()
                        .filter(|(_, id)| id.starts_with("ground-1-"))
                        .map(|(i, _)| b["asymptoticPopulations"][i].as_f64().unwrap())
                        .sum::<f64>();
                    if f1 < 0.99 {
                        failures.push(format!("{name}: expected predominant F1 pumping, got {f1}"));
                    }
                }
            } else {
                failures.push(format!(
                    "{name}: {}",
                    full.as_ref().err().or_else(|| only.as_ref().err()).unwrap()
                ));
            }
            let warm = solver.solve(
                &m,
                &Options {
                    duration_us: duration,
                    samples: 161,
                    ..Options::default()
                },
            );
            if let (Ok(cold), Ok(warm)) = (&full, &warm) {
                assert_eq!(cold["populations"], warm["populations"]);
            }
            report.push(json!({"name":name,"steadyOnly":only.unwrap_or_else(|e|json!({"error":e})),"trajectory":full.unwrap_or_else(|e|json!({"error":e})),"warmDiagnostics":warm.map(|v|v["diagnostics"].clone()).unwrap_or_else(|e|json!({"error":e})),"stationaryMaxDifference":delta}));
        }
        std::fs::write(
            folder.join("audit.json"),
            serde_json::to_string_pretty(&json!({"cases":report,"failures":failures})).unwrap(),
        )
        .unwrap();
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }
    #[test]
    #[ignore = "Independent basis-phase audit of every full-density component on stiff Rb87 generators"]
    fn constant_rb32_phase_gauge_density_audit() {
        let folder = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../artifacts/v4/native-constant");
        let cases: Value =
            serde_json::from_str(&std::fs::read_to_string(folder.join("cases.json")).unwrap())
                .unwrap();
        let mut records = Vec::new();
        let mut failures = Vec::new();
        let mut solver = Solver::new().unwrap();
        for case in cases.as_array().unwrap().iter().filter(|c| {
            matches!(
                c["name"].as_str(),
                Some("open-F2-F2" | "tilted-B-general-polarization")
            )
        }) {
            let m: Model = serde_json::from_value(case["model"].clone()).unwrap();
            let n = m.dimension;
            let angles: Vec<_> = (0..n)
                .map(|i| ((i * i + 3 * i) % 17) as f64 * 0.13 + ((i + 5) % 4) as f64 * 0.07)
                .collect();
            let rotate = |re: &[f64], im: &[f64]| -> (Vec<f64>, Vec<f64>) {
                let mut real = Vec::new();
                let mut imag = Vec::new();
                for i in 0..n {
                    for j in 0..n {
                        let (s, c) = (angles[i] - angles[j]).sin_cos();
                        let z = C::new(re[i * n + j], im[i * n + j]) * C::new(c, s);
                        real.push(z.re);
                        imag.push(z.im);
                    }
                }
                (real, imag)
            };
            let mut gauged = m.clone();
            (gauged.h_real, gauged.h_imag) = rotate(&m.h_real, &m.h_imag);
            for (op, original) in gauged.collapse.iter_mut().zip(&m.collapse) {
                (op.real, op.imag) = rotate(&original.real, &original.imag);
            }
            let options = Options {
                duration_us: 1.,
                samples: 21,
                return_density: true,
                ..Options::default()
            };
            let original = solver.solve(&m, &options).unwrap();
            let changed = solver.solve(&gauged, &options).unwrap();
            let mut density_delta = 0_f64;
            let mut population_delta = 0_f64;
            for t in 0..21 {
                for i in 0..n {
                    for j in 0..n {
                        let (s, c) = (angles[i] - angles[j]).sin_cos();
                        let expected = C::new(
                            original["densityReal"][t][i * n + j].as_f64().unwrap(),
                            original["densityImag"][t][i * n + j].as_f64().unwrap(),
                        ) * C::new(c, s);
                        let actual = C::new(
                            changed["densityReal"][t][i * n + j].as_f64().unwrap(),
                            changed["densityImag"][t][i * n + j].as_f64().unwrap(),
                        );
                        density_delta = density_delta.max((expected - actual).abs());
                        if i == j {
                            population_delta =
                                population_delta.max((expected.re - actual.re).abs());
                        }
                    }
                }
            }
            let mut stationary_delta = 0_f64;
            if let (Some(a), Some(b)) = (
                original["asymptoticPopulations"].as_array(),
                changed["asymptoticPopulations"].as_array(),
            ) {
                for (x, y) in a.iter().zip(b) {
                    stationary_delta =
                        stationary_delta.max((x.as_f64().unwrap() - y.as_f64().unwrap()).abs());
                }
            }
            if density_delta > 1e-8 || stationary_delta > 1e-8 {
                failures.push(format!(
                    "{}: full density {density_delta:e}, stationary {stationary_delta:e}",
                    case["name"]
                ));
            }
            records.push(json!({"name":case["name"],"durationUs":1.,"samples":21,"maxDensityDifference":density_delta,"maxPopulationDifference":population_delta,"stationaryDifference":stationary_delta,"originalDiagnostics":original["diagnostics"],"gaugeDiagnostics":changed["diagnostics"]}));
        }
        std::fs::write(
            folder.join("phase-gauge-audit.json"),
            serde_json::to_string_pretty(&json!({"cases":records,"failures":failures})).unwrap(),
        )
        .unwrap();
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }
    #[test]
    #[ignore = "Frequency-aware defect envelope across 20, 200, 2000 and one million microseconds for seven full Rb87 cases"]
    fn constant_rb32_frequency_aware_defect_audit() {
        let folder = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../artifacts/v4/native-constant");
        let cases: Value =
            serde_json::from_str(&std::fs::read_to_string(folder.join("cases.json")).unwrap())
                .unwrap();
        let lap = Lapack::load().unwrap();
        let mut records = Vec::new();
        for case in cases.as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            let model: Model = serde_json::from_value(case["model"].clone()).unwrap();
            let (g, inds, initial, certificate) = modal_defect_case(&lap, &model);
            let mut rows = Vec::new();
            for end in [20., 200., 2000., 1e6] {
                let result = certificate
                    .apply(&lap, &initial, &[0., end * 0.37, end])
                    .unwrap();
                let mut trace = 0_f64;
                let mut psd = 0_f64;
                let mut analytic_error: Option<f64> = None;
                for (state, &t) in result.states.iter().zip(&[0., end * 0.37, end]) {
                    let mut x = vec![0.; g.n * g.n];
                    for (i, &global) in inds.iter().enumerate() {
                        x[global] = state[i];
                    }
                    let rho = expand_density(&g.coords, &x, g.n);
                    trace = trace.max((x[..g.n].iter().sum::<f64>() - 1.).abs());
                    psd = psd.max(positivity(&lap, &rho, g.n).unwrap());
                    if name == "no-light-ground-state" {
                        let expected = expand_density(&g.coords, &g.initial, g.n);
                        analytic_error = Some(
                            analytic_error
                                .unwrap_or(0.)
                                .max(difference_trace_norm(&lap, &rho, &expected, g.n)),
                        );
                    }
                    if name == "lights-off-D2-stretched" {
                        let e = model.initial_index;
                        let ground = model.states.iter().position(|s| s == "ground-2-2").unwrap();
                        let gamma = model
                            .collapse
                            .iter()
                            .map(|c| {
                                (0..g.n)
                                    .map(|i| {
                                        c.real[i * g.n + e].powi(2) + c.imag[i * g.n + e].powi(2)
                                    })
                                    .sum::<f64>()
                            })
                            .sum::<f64>();
                        let p = (-gamma * t).exp();
                        let mut expected = vec![C::default(); g.n * g.n];
                        expected[e * g.n + e].re = p;
                        expected[ground * g.n + ground].re = 1. - p;
                        analytic_error = Some(
                            analytic_error
                                .unwrap_or(0.)
                                .max(difference_trace_norm(&lap, &rho, &expected, g.n)),
                        );
                    }
                    if name == "cycling-F2m2-F3m3" && t >= 1. {
                        let ground = model.states.iter().position(|s| s == "ground-2-2").unwrap();
                        let excited = model.states.iter().position(|s| s == "D2-3-3").unwrap();
                        let h = C::new(
                            model.h_real[ground * g.n + excited],
                            model.h_imag[ground * g.n + excited],
                        );
                        let delta = model.h_real[excited * g.n + excited]
                            - model.h_real[ground * g.n + ground];
                        let gamma = model
                            .collapse
                            .iter()
                            .map(|c| {
                                (0..g.n)
                                    .map(|i| {
                                        c.real[i * g.n + excited].powi(2)
                                            + c.imag[i * g.n + excited].powi(2)
                                    })
                                    .sum::<f64>()
                            })
                            .sum::<f64>();
                        let omega2 = 4. * h.abs().powi(2);
                        let p = omega2 / (gamma * gamma + 2. * omega2 + 4. * delta * delta);
                        let z = 2. * p - 1.;
                        let off = C::new(-h.im * z, h.re * z)
                            * C::new(-gamma * 0.5, -delta)
                            * (1. / (gamma * gamma * 0.25 + delta * delta));
                        let mut expected = vec![C::default(); g.n * g.n];
                        expected[ground * g.n + ground].re = 1. - p;
                        expected[excited * g.n + excited].re = p;
                        expected[ground * g.n + excited] = off;
                        expected[excited * g.n + ground] = off.conj();
                        analytic_error = Some(analytic_error.unwrap_or(0.).max(
                            difference_trace_norm(&lap, &rho, &expected, g.n)
                                + 2. * (-gamma * t * 0.5).exp(),
                        ));
                    }
                }
                if let Some(error) = analytic_error {
                    assert!(
                        error <= result.bound.total,
                        "{name} T={end}: analytic error {error:e} > envelope {}",
                        result.bound.total
                    );
                }
                rows.push(json!({"bound":bound_json(&result.bound),"analyticErrorUpperBound":analytic_error,"traceError":trace,"positivityError":psd,"applicationMs":result.elapsed_ms}));
            }
            records.push(json!({"name":name,"defectNorm":certificate.diagnostics.defect_norm_upper,"defectProductRoundoff":certificate.diagnostics.defect_product_roundoff,"inverseResidualBound":certificate.diagnostics.inverse_residual_bound,"preparationMs":certificate.diagnostics.preparation_ms,"windows":rows}));
        }
        std::fs::write(folder.join("frequency-aware-defect-audit.json"),serde_json::to_string_pretty(&json!({"cases":records,"scope":"Analytic truth is checked only for cycling (Bloch stationary solution plus rigorous exponential tail), free decay, and no-light invariance. Other Rb87 entries report an error envelope and numerical diagnostics, not an independently known true error."})).unwrap()).unwrap();
    }
    #[test]
    #[ignore = "Production acceptance and cached timing across four duration windows for seven full Rb87 generators"]
    fn constant_rb32_production_duration_audit() {
        let folder = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../artifacts/v4/native-constant");
        let cases: Value =
            serde_json::from_str(&std::fs::read_to_string(folder.join("cases.json")).unwrap())
                .unwrap();
        let mut solver = Solver::new().unwrap();
        let mut records = Vec::new();
        let mut failures = Vec::new();
        for case in cases.as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            let model: Model = serde_json::from_value(case["model"].clone()).unwrap();
            let mut rows = Vec::new();
            for duration in [20., 200., 2000., 1e6] {
                match solver.solve(
                    &model,
                    &Options {
                        duration_us: duration,
                        samples: 161,
                        ..Options::default()
                    },
                ) {
                    Ok(value) => {
                        let bound = value["diagnostics"]["trajectoryErrorEstimate"]
                            .as_f64()
                            .unwrap();
                        assert!(bound.is_finite() && bound <= 5e-6);
                        rows.push(json!({"durationUs":duration,"accepted":true,"diagnostics":value["diagnostics"],"finalPopulations":value["populations"].as_array().unwrap().last(),"asymptoticStatus":value["asymptoticStatus"]}));
                    }
                    Err(error) => {
                        if duration <= 2000. {
                            failures.push(format!("{name} at {duration} us: {error}"));
                        }
                        rows.push(json!({"durationUs":duration,"accepted":false,"reason":error}));
                    }
                }
            }
            records.push(json!({"name":name,"windows":rows}));
        }
        std::fs::write(folder.join("production-duration-audit.json"),serde_json::to_string_pretty(&json!({"cases":records,"failures":failures,"scope":"Unchanged production acceptance threshold of 5e-6; the million-microsecond window may explicitly reject if no conservative certificate is available."})).unwrap()).unwrap();
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }
}
