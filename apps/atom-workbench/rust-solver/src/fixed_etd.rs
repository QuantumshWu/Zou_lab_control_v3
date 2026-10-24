//! Experimental complete-density short-time exponential integrator.
//!
//! A0 contains diagonal H, diagonal C^H C damping, and EVERY recycling term.
//! Its recycling edges must go between disjoint coordinate sets. Thus R D^k R
//! vanishes for every k and f(A0) has an exact first-divided-difference form.
//! All remaining Hamiltonian/damping terms and all drives stay in B(t).
//! Cox--Matthews ETDRK4: JCP 176 (2002), 430--455,
//! https://doi.org/10.1006/jcph.2002.6995 . No state/coherence/physical term is
//! discarded. Step doubling and an independent mesh are observed numerical
//! checks, NOT a certified continuous-time forward-error bound.
use super::{add_superterm, build_generator, complex_input, coordinate_map, coordinates, expand_density, positivity, Coordinate, Lapack, Model, Options, C};
use serde_json::{json, Value};
use std::{collections::{BTreeMap, BTreeSet, VecDeque}, rc::Rc, time::Instant};

type Sparse = Vec<Vec<(usize, C)>>;
const MAX_ATTEMPTS: usize = 100_000;
const MAX_PRODUCTS: usize = 32_000_000_000;
const LOCAL_TOLERANCE: f64 = 2e-7;
const POPULATION_COMPARISON: f64 = 2e-4;

#[derive(Clone)]
struct Edge { row: usize, col: usize, value: C }
struct Drive { omega: f64, omega_low:f64, cos: Sparse, sin: Sparse }
struct Prepared {
    n: usize,
    diagonal: Vec<C>,
    recycling: Vec<Edge>,
    base: Sparse,
    drives: Vec<Drive>,
    phase_limit: f64,
    rhs_nnz: usize,
    initial: Vec<C>,
}
struct Functions {
    // E, E_half, Q, f1, f2, f3 of the entire A0, not just its diagonal.
    diagonal: Vec<[C; 6]>,
    recycling: Vec<[C; 6]>,
}
struct Stepper<'a> {
    model: &'a Prepared,
    functions: VecDeque<(u64, Rc<Functions>)>,
    rhs_evaluations: usize,
    products: usize,
    function_builds: usize,
    function_ms: f64,
}
struct Run {
    states: Vec<Vec<C>>,
    accepted: usize,
    rejected: usize,
    rhs_evaluations: usize,
    products: usize,
    function_builds: usize,
    function_ms: f64,
    max_step: f64,
    min_step: f64,
    maximum_local_estimate: f64,
    sum_local_estimates: f64,
    elapsed_ms: f64,
}
pub struct Report {
    pub times: Vec<f64>,
    /// Original real Hermitian density coordinates, including all coherences.
    pub states: Vec<Vec<f64>>,
    pub diagnostics: Value,
    pub trace_error: f64,
    pub positivity_error: f64,
}

fn finite(z: C) -> bool { z.re.is_finite() && z.im.is_finite() }
fn divide(a: C, b: C) -> C {
    // Smith's complex division avoids squaring optical frequencies.
    if b.re.abs() >= b.im.abs() {
        let ratio = b.im/b.re; let denominator = b.re+b.im*ratio;
        C::new((a.re+a.im*ratio)/denominator, (a.im-a.re*ratio)/denominator)
    } else {
        let ratio = b.re/b.im; let denominator = b.im+b.re*ratio;
        C::new((a.re*ratio+a.im)/denominator, (a.im*ratio-a.re)/denominator)
    }
}
fn exponential(z: C) -> C {
    let (sine, cosine) = z.im.sin_cos(); let factor = z.re.exp();
    C::new(factor*cosine, factor*sine)
}
/// phi_k(z)=sum_j z^j/(j+k)!, with phi_0=exp. Taylor near zero
/// avoids cancellation in the recurrence used for large imaginary arguments.
fn phi(z: C) -> [C; 6] {
    let mut result = [C::default(); 6];
    if z.abs() < 1. {
        for k in 0..6 {
            let factorial = (1..=k).product::<usize>().max(1) as f64;
            let mut term = C::new(1./factorial, 0.); let mut sum = term;
            for j in 1..80 {
                term = term*z*(1./(j+k) as f64); sum += term;
                if term.abs() <= 0.25*f64::EPSILON*sum.abs() { break; }
            }
            result[k] = sum;
        }
    } else {
        result[0] = exponential(z);
        let constants = [1., 1., 0.5, 1./6., 1./24.];
        for k in 1..6 { result[k] = divide(result[k-1]-C::new(constants[k-1], 0.), z); }
    }
    result
}
fn values_and_derivatives(lambda: C, h: f64) -> ([C; 6], [C; 6]) {
    let p = phi(lambda*h); let q = phi(lambda*(0.5*h));
    let derivative = [p[0], p[1]-p[2], p[2]-p[3]*2., p[3]-p[4]*3.];
    let half_derivative = q[1]-q[2];
    let values = [p[0], q[0], q[1]*(0.5*h),
        (p[1]-p[2]*3.+p[3]*4.)*h, (p[2]-p[3]*2.)*h, (p[3]*4.-p[2])*h];
    let derivatives = [p[0]*h, q[0]*(0.5*h), half_derivative*(0.25*h*h),
        (derivative[1]-derivative[2]*3.+derivative[3]*4.)*(h*h),
        (derivative[2]-derivative[3]*2.)*(h*h),
        (derivative[3]*4.-derivative[2])*(h*h)];
    (values, derivatives)
}
fn compress(matrix: &mut Sparse) {
    for column in matrix {
        column.sort_by_key(|&(row, _)| row);
        let mut result: Vec<(usize, C)> = Vec::new();
        for &(row, value) in column.iter() {
            if let Some(last) = result.last_mut().filter(|last| last.0 == row) { last.1 += value; }
            else { result.push((row, value)); }
        }
        result.retain(|&(_, value)| value.re != 0. || value.im != 0.);
        *column = result;
    }
}
fn add_hamiltonian(matrix: &mut Sparse, h: &[C], n: usize) {
    for a in 0..n { for c in 0..n {
        let value = h[a*n+c]; if value.re == 0. && value.im == 0. { continue; }
        for b in 0..n {
            matrix[c*n+b].push((a*n+b, C::new(value.im, -value.re)));
            matrix[b*n+a].push((b*n+c, C::new(-value.im, value.re)));
        }
    } }
}
fn hamiltonian(h: &[C], n: usize) -> Sparse {
    let mut result = vec![Vec::new(); n*n]; add_hamiltonian(&mut result, h, n); compress(&mut result); result
}
fn checked_drive(real: &[f64], imag: &[f64], n: usize, name: &str) -> Result<Vec<C>, String> {
    let values = complex_input(real, imag, n*n, name)?;
    for i in 0..n { for j in 0..n {
        if (values[i*n+j]-values[j*n+i].conj()).abs() > 1e-10*(1.+values[i*n+j].abs()) {
            return Err(format!("{name} is not Hermitian"));
        }
    } }
    Ok(values)
}
impl Prepared {
    fn new(model: &Model) -> Result<Self, String> {
        let generator = build_generator(model)?; let n = model.dimension; let d = n*n;
        let initial = expand_density(&generator.coords, &generator.initial, n);
        let mut h = complex_input(&model.h_real, &model.h_imag, d, "ETD Hamiltonian")?;
        let mut gram = vec![C::default(); d]; let mut recycling = vec![Vec::new(); d];
        for operator in &model.collapse {
            let c = complex_input(&operator.real, &operator.imag, d, "ETD collapse")?;
            let nonzero: Vec<_> = c.iter().enumerate().filter(|(_, v)| v.re != 0. || v.im != 0.)
                .map(|(k, &v)| (k/n, k%n, v)).collect();
            for &(a, i, v) in &nonzero { for &(b, j, w) in &nonzero {
                recycling[i*n+j].push((a*n+b, v*w.conj()));
                if a == b { gram[i*n+j] += v.conj()*w; }
            } }
        }
        compress(&mut recycling);
        let mut input = BTreeSet::new(); let mut output = BTreeSet::new(); let mut edges = Vec::new();
        for (col, entries) in recycling.into_iter().enumerate() { for (row, value) in entries {
            input.insert(col); output.insert(row); edges.push(Edge { row, col, value });
        } }
        if !input.is_disjoint(&output) {
            return Err("Fixed ETD requires algebraically disjoint recycling input/output sectors".into());
        }
        let diagonal: Vec<_> = (0..d).map(|k| {
            let (i, j) = (k/n, k%n);
            C::new(-0.5*(gram[i*n+i].re+gram[j*n+j].re), -(h[i*n+i].re-h[j*n+j].re))
        }).collect();
        for i in 0..n { h[i*n+i] = C::default(); }
        let mut base = hamiltonian(&h, n);
        // Retain every off-diagonal damping term. It need not be zero for
        // dressed-state collapse operators or interfering emission channels.
        for i in 0..n { for k in 0..n { if i != k {
            let value = gram[i*n+k]*(-0.5); if value.re == 0. && value.im == 0. { continue; }
            for j in 0..n {
                base[k*n+j].push((i*n+j, value));
                base[j*n+i].push((j*n+k, value));
            }
        } } }
        compress(&mut base);
        let mut drives = Vec::new(); let mut maximum_frequency = 0_f64;
        let frequency_parts=crate::harmonics::model_drive_frequency_parts(model)?;
        for (drive,(omega,omega_low)) in model.drives.iter().zip(frequency_parts) {
            if !drive.omega.is_finite() || drive.omega <= 0. { return Err("Invalid ETD drive frequency".into()); }
            maximum_frequency = maximum_frequency.max(omega+omega_low);
            drives.push(Drive { omega,omega_low,
                cos: hamiltonian(&checked_drive(&drive.cos.real, &drive.cos.imag, n, "ETD cosine drive")?, n),
                sin: hamiltonian(&checked_drive(&drive.sin.real, &drive.sin.imag, n, "ETD sine drive")?, n) });
        }
        let rhs_nnz = base.iter().map(Vec::len).sum::<usize>() + drives.iter()
            .map(|drive| drive.cos.iter().chain(&drive.sin).map(Vec::len).sum::<usize>()).sum::<usize>();
        Ok(Self { n, diagonal, recycling: edges, base, drives,
            phase_limit: if maximum_frequency == 0. { f64::INFINITY } else { std::f64::consts::TAU/(8.*maximum_frequency) },
            rhs_nnz, initial })
    }
    fn functions(&self, h: f64) -> Result<Functions, String> {
        let diagonal: Vec<_> = self.diagonal.iter().map(|&lambda| values_and_derivatives(lambda, h).0).collect();
        let mut recycling = Vec::with_capacity(self.recycling.len());
        for edge in &self.recycling {
            let difference = self.diagonal[edge.row]-self.diagonal[edge.col];
            let mut values = [C::default(); 6];
            if (difference*h).abs() < 1e-5 {
                let midpoint = (self.diagonal[edge.row]+self.diagonal[edge.col])*0.5;
                values = values_and_derivatives(midpoint, h).1;
            } else {
                for k in 0..6 { values[k] = divide(diagonal[edge.row][k]-diagonal[edge.col][k], difference); }
            }
            for value in &mut values { *value = *value*edge.value; }
            recycling.push(values);
        }
        if diagonal.iter().chain(&recycling).flatten().any(|&v| !finite(v)) {
            return Err("Non-finite ETD matrix-function coefficient".into());
        }
        Ok(Functions { diagonal, recycling })
    }
    fn combine(&self, functions: &Functions, terms: &[(&[C], usize, f64)]) -> Vec<C> {
        let mut result = vec![C::default(); self.diagonal.len()];
        for &(input, function, factor) in terms {
            for (i, value) in result.iter_mut().enumerate() { *value += functions.diagonal[i][function]*input[i]*factor; }
            for (edge, values) in self.recycling.iter().zip(&functions.recycling) {
                result[edge.row] += values[function]*input[edge.col]*factor;
            }
        }
        result
    }
}
impl<'a> Stepper<'a> {
    fn new(model: &'a Prepared) -> Self {
        Self { model, functions: VecDeque::new(), rhs_evaluations: 0, products: 0, function_builds: 0, function_ms: 0. }
    }
    fn functions(&mut self, h: f64) -> Result<Rc<Functions>, String> {
        let key = h.to_bits();
        if let Some((_, value)) = self.functions.iter().find(|(old, _)| *old == key) { return Ok(value.clone()); }
        let before = Instant::now(); let result = Rc::new(self.model.functions(h)?);
        self.function_ms += before.elapsed().as_secs_f64()*1000.; self.function_builds += 1;
        self.functions.push_back((key, result.clone()));
        if self.functions.len() > 24 { self.functions.pop_front(); }
        Ok(result)
    }
    fn rhs(&mut self, t: f64, x: &[C]) -> Result<Vec<C>, String> {
        self.rhs_evaluations += 1; self.products += self.model.rhs_nnz;
        if self.products > MAX_PRODUCTS { return Err("Fixed ETD exceeded its short-time multiplication budget".into()); }
        let mut result = vec![C::default(); x.len()];
        let mut apply = |matrix: &Sparse, factor: f64| {
            for (col, entries) in matrix.iter().enumerate() {
                let input = x[col]*factor;
                for &(row, value) in entries { result[row] += value*input; }
            }
        };
        apply(&self.model.base, 1.);
        for drive in &self.model.drives {
            let high = drive.omega*t; let low = drive.omega_low.mul_add(t,drive.omega.mul_add(t, -high));
            let (s, c) = high.sin_cos(); let (ds, dc) = low.sin_cos();
            apply(&drive.cos, c*dc-s*ds); apply(&drive.sin, s*dc+c*ds);
        }
        Ok(result)
    }
    fn step(&mut self, t: f64, h: f64, x: &[C]) -> Result<Vec<C>, String> {
        let functions = self.functions(h)?; let model = self.model;
        let n0 = self.rhs(t, x)?;
        let a = model.combine(&functions, &[(x, 1, 1.), (&n0, 2, 1.)]);
        let na = self.rhs(t+0.5*h, &a)?;
        let b = model.combine(&functions, &[(x, 1, 1.), (&na, 2, 1.)]);
        let nb = self.rhs(t+0.5*h, &b)?;
        let c = model.combine(&functions, &[(&a, 1, 1.), (&nb, 2, 2.), (&n0, 2, -1.)]);
        let nc = self.rhs(t+h, &c)?;
        let next = model.combine(&functions, &[(x, 0, 1.), (&n0, 3, 1.), (&na, 4, 2.), (&nb, 4, 2.), (&nc, 5, 1.)]);
        if next.iter().any(|&value| !finite(value)) { return Err("Non-finite ETDRK4 candidate".into()); }
        Ok(next)
    }
}
fn integrate(model: &Prepared, times: &[f64], tolerance: f64, maximum_step: f64) -> Result<Run, String> {
    let started = Instant::now(); let mut stepper = Stepper::new(model);
    let mut state = model.initial.clone(); let mut states = Vec::with_capacity(times.len());
    let mut time = 0.; let mut h = maximum_step.min(times[1]);
    let mut accepted = 0; let mut rejected = 0; let mut max_step = 0_f64; let mut min_step = f64::INFINITY;
    let mut maximum_local_estimate = 0_f64; let mut sum_local_estimates = 0.;
    for &target in times {
        while time < target {
            if accepted+rejected >= MAX_ATTEMPTS { return Err(format!("Fixed ETD exceeded its short-time step budget at {time:e}us after {accepted} accepted / {rejected} rejected steps")); }
            let remaining = target-time; let step = h.min(maximum_step).min(remaining);
            if step <= 0. || time+step == time { return Err("Fixed ETD time step underflow".into()); }
            let coarse = stepper.step(time, step, &state)?;
            let half = stepper.step(time, 0.5*step, &state)?;
            let fine = stepper.step(time+0.5*step, 0.5*step, &half)?;
            let estimate = fine.iter().zip(&coarse).fold(0_f64, |norm, (&a, &b)| norm.hypot((a-b).abs()))/15.;
            if !estimate.is_finite() { return Err("Non-finite ETD step comparison".into()); }
            if estimate <= tolerance {
                state = fine; time = if step == remaining { target } else { time+step };
                accepted += 1; max_step = max_step.max(step); min_step = min_step.min(step);
                maximum_local_estimate = maximum_local_estimate.max(estimate); sum_local_estimates += estimate;
                if accepted%1024 == 0 && std::env::var_os("ATOM_SOLVER_DEBUG").is_some() {
                    eprintln!("Fixed ETD t={time:.6}us accepted={accepted} rejected={rejected} h={step:e} local={estimate:e} rhs={} functions={} functionMs={:.2} elapsedMs={:.2}",
                        stepper.rhs_evaluations, stepper.function_builds, stepper.function_ms, started.elapsed().as_secs_f64()*1000.);
                }
                // Dyadic changes permit reuse of THIS request's matrix-function
                // coefficients; no trajectory or previous parameter result is reused.
                if estimate < tolerance/64. && step == h { h = (2.*h).min(maximum_step); }
            } else { rejected += 1; h = 0.5*step; }
        }
        states.push(state.clone());
    }
    Ok(Run { states, accepted, rejected, rhs_evaluations: stepper.rhs_evaluations,
        products: stepper.products, function_builds: stepper.function_builds, function_ms: stepper.function_ms,
        max_step, min_step: if min_step.is_finite() { min_step } else { 0. }, maximum_local_estimate,
        sum_local_estimates, elapsed_ms: started.elapsed().as_secs_f64()*1000. })
}
fn run_diagnostics(run: &Run) -> Value {
    json!({"acceptedSteps":run.accepted,"rejectedSteps":run.rejected,"rhsEvaluations":run.rhs_evaluations,
        "sparseEntryProducts":run.products,"matrixFunctionBuilds":run.function_builds,"matrixFunctionMs":run.function_ms,
        "maximumStepUs":run.max_step,"minimumStepUs":run.min_step,"maximumLocalEstimate":run.maximum_local_estimate,
        "sumLocalEstimates":run.sum_local_estimates,"elapsedMs":run.elapsed_ms})
}
pub fn solve(lap: &Lapack, model: &Model, options: &Options) -> Result<Report, String> {
    if options.steady_only || !options.duration_us.is_finite() || options.duration_us <= 0.
        || options.duration_us > 1000. || options.samples < 2 || options.samples > 2001 {
        return Err("Fixed ETD is a finite-time experiment for durations up to1000us and2..2001samples".into());
    }
    let started = Instant::now(); let prepared = Prepared::new(model)?; let n = prepared.n;
    if positivity(lap, &prepared.initial, n)? > 1e-10 { return Err("Invalid initial density for ETD".into()); }
    let times: Vec<_> = (0..options.samples).map(|i| options.duration_us*i as f64/(options.samples-1) as f64).collect();
    let initial_step = prepared.phase_limit.min(options.duration_us/16.).min(times[1]);
    if options.duration_us/initial_step > MAX_ATTEMPTS as f64 { return Err("Fixed ETD resolved phases exceed the short-time budget".into()); }
    let coarse = integrate(&prepared, &times, LOCAL_TOLERANCE, initial_step)?;
    // A thirds mesh is independent of every accepted step's internal halves.
    // The second run must actually refine the largest observed accepted step.
    let refinement_step = (coarse.max_step/3.).min(initial_step/3.);
    let fine = integrate(&prepared, &times, LOCAL_TOLERANCE/4., refinement_step)?;
    let mut groups = BTreeMap::<String, Vec<usize>>::new();
    for (i, state) in model.states.iter().take(n).enumerate() {
        let parts: Vec<_> = state.split('-').collect();
        if parts.len() >= 3 && matches!(parts[0], "ground"|"D1"|"D2") {
            groups.entry(format!("{}-{}", parts[0], parts[1])).or_default().push(i);
        }
    }
    let mut population_error = 0_f64; let mut group_error = 0_f64;
    let mut trace_error = 0_f64; let mut positivity_error = 0_f64; let mut hermiticity_error = 0_f64;
    for (a, b) in coarse.states.iter().zip(&fine.states) {
        for i in 0..n { population_error = population_error.max((a[i*n+i].re-b[i*n+i].re).abs()); }
        for indices in groups.values() {
            group_error = group_error.max(indices.iter().map(|&i| a[i*n+i].re-b[i*n+i].re).sum::<f64>().abs());
        }
        let trace = (0..n).fold(C::default(), |sum, i| sum+b[i*n+i]);
        trace_error = trace_error.max((trace-C::new(1., 0.)).abs());
        for i in 0..n { for j in 0..n { hermiticity_error = hermiticity_error.max((b[i*n+j]-b[j*n+i].conj()).abs()); } }
        positivity_error = positivity_error.max(positivity(lap, b, n)?);
    }
    if population_error.max(group_error) > POPULATION_COMPARISON || trace_error > 1e-7
        || positivity_error > 1e-5 || hermiticity_error > 1e-8 {
        return Err(format!("Fixed ETD numerical verification failed: population={population_error:e}, Fgroup={group_error:e}, trace={trace_error:e}, positivity={positivity_error:e}, Hermiticity={hermiticity_error:e}"));
    }
    let coords = coordinates(n);
    let states = fine.states.iter().map(|rho| coords.iter().map(|&coordinate| match coordinate {
        Coordinate::Population(i) => rho[i*n+i].re,
        Coordinate::Real(i,j) => rho[i*n+j].re,
        Coordinate::Imag(i,j) => rho[i*n+j].im,
    }).collect()).collect();
    let diagnostics = json!({"engine":"rust-fixed-etd-experiment","method":"Complete density / nilpotent-recycling ETDRK4",
        "elapsedMs":started.elapsed().as_secs_f64()*1000.,"physicalHilbertDimension":n,"liouvilleDimension":n*n,
        "recyclingEntries":prepared.recycling.len(),"rhsEntries":prepared.rhs_nnz,"phaseStepLimitUs":prepared.phase_limit,
        "populationRefinementError":population_error,"groupPopulationRefinementError":group_error,
        "traceError":trace_error,"positivityError":positivity_error,"hermiticityError":hermiticity_error,
        "coarse":run_diagnostics(&coarse),"independentThirdsRefinement":run_diagnostics(&fine),
        "errorScope":"Observed independent-mesh population comparison plus density validity checks; no certified continuous-time error bound or infinite-time result."});
    Ok(Report { times, states, diagnostics, trace_error, positivity_error })
}

// The multistep experiment below uses the same exact physical split, but
// performs all long-run state/history/operator products in the original
// n^2 REAL Hermitian coordinates. Only coefficient preparation is complex.
type RealSparse = Vec<Vec<(usize, f64)>>;
struct RealDrive { omega: f64, omega_low:f64, cos: RealSparse, sin: RealSparse }
struct RealPrepared {
    base: RealSparse,
    drives: Vec<RealDrive>,
    coords: Vec<Coordinate>,
    initial: Vec<f64>,
    rhs_nnz: usize,
}
struct AbFunctions { actions: [RealSparse; 5] }
struct AbRun {
    states: Vec<Vec<f64>>,
    subdivisions: usize,
    steps: usize,
    rhs_evaluations: usize,
    products: usize,
    function_ms: f64,
    elapsed_ms: f64,
}
fn compress_real(matrix: &mut RealSparse) {
    for column in matrix {
        column.sort_by_key(|&(row, _)| row);
        let mut result: Vec<(usize, f64)> = Vec::new();
        for &(row, value) in column.iter() {
            if let Some(last) = result.last_mut().filter(|last| last.0 == row) { last.1 += value; }
            else { result.push((row, value)); }
        }
        result.retain(|&(_, value)| value != 0.); *column = result;
    }
}
fn physical_terms_to_real(n: usize, terms: impl IntoIterator<Item=(usize,usize,C)>) -> Result<RealSparse, String> {
    let d = n*n; let map = coordinate_map(&coordinates(n), n);
    let mut result = vec![Vec::new(); d]; let mut imaginary = vec![0.; n*d];
    for (row, col, value) in terms { add_superterm(&mut result, &mut imaginary, &map, row, col, value, n); }
    if imaginary.iter().any(|v| v.abs() > 1e-9) { return Err("ETD real-coordinate map fails Hermiticity invariance".into()); }
    compress_real(&mut result); Ok(result)
}
fn density_coordinates(coords: &[Coordinate], rho: &[C], n: usize) -> Vec<f64> {
    coords.iter().map(|&coordinate| match coordinate {
        Coordinate::Population(i) => rho[i*n+i].re,
        Coordinate::Real(i,j) => rho[i*n+j].re,
        Coordinate::Imag(i,j) => rho[i*n+j].im,
    }).collect()
}
impl RealPrepared {
    fn new(prepared: &Prepared) -> Result<Self, String> {
        let convert = |a: &Sparse| physical_terms_to_real(prepared.n,
            a.iter().enumerate().flat_map(|(col, values)| values.iter().map(move |&(row, value)| (row,col,value))));
        let base = convert(&prepared.base)?;
        let drives: Vec<_> = prepared.drives.iter().map(|drive| Ok(RealDrive { omega: drive.omega,omega_low:drive.omega_low,
            cos: convert(&drive.cos)?, sin: convert(&drive.sin)? })).collect::<Result<_,String>>()?;
        let coords = coordinates(prepared.n);
        let initial = density_coordinates(&coords, &prepared.initial, prepared.n);
        let rhs_nnz = base.iter().map(Vec::len).sum::<usize>() + drives.iter()
            .map(|drive| drive.cos.iter().chain(&drive.sin).map(Vec::len).sum::<usize>()).sum::<usize>();
        Ok(Self { base, drives, coords, initial, rhs_nnz })
    }
    fn rhs(&self, t: f64, x: &[f64]) -> Vec<f64> {
        let mut result = vec![0.; x.len()];
        let mut apply = |matrix: &RealSparse, factor: f64| {
            for (j, col) in matrix.iter().enumerate() {
                let input = x[j]*factor;
                for &(i, value) in col { result[i] += value*input; }
            }
        };
        apply(&self.base, 1.);
        for drive in &self.drives {
            let high = drive.omega*t; let low = drive.omega_low.mul_add(t,drive.omega.mul_add(t, -high));
            let (s,c) = high.sin_cos(); let (ds,dc) = low.sin_cos();
            apply(&drive.cos,c*dc-s*ds); apply(&drive.sin,s*dc+c*ds);
        }
        result
    }
}
fn ab_values_and_derivatives(lambda: C, h: f64) -> ([C; 5], [C; 5]) {
    let p = phi(lambda*h);
    let derivative = [p[0], p[1]-p[2], p[2]-p[3]*2., p[3]-p[4]*3., p[4]-p[5]*4.];
    let combinations = |p: &[C]| [
        p[1]+p[2]*(11./6.)+p[3]*2.+p[4],
        p[2]*(-3.)+p[3]*(-5.)+p[4]*(-3.),
        p[2]*1.5+p[3]*4.+p[4]*3.,
        p[2]*(-1./3.)-p[3]-p[4],
    ];
    let beta = combinations(&p); let dbeta = combinations(&derivative);
    ([p[0],beta[0]*h,beta[1]*h,beta[2]*h,beta[3]*h],
        [p[0]*h,dbeta[0]*(h*h),dbeta[1]*(h*h),dbeta[2]*(h*h),dbeta[3]*(h*h)])
}
impl AbFunctions {
    fn new(prepared: &Prepared, h: f64) -> Result<Self, String> {
        let diagonal: Vec<_> = prepared.diagonal.iter().map(|&lambda| ab_values_and_derivatives(lambda,h).0).collect();
        let mut recycling = Vec::with_capacity(prepared.recycling.len());
        for edge in &prepared.recycling {
            let difference = prepared.diagonal[edge.row]-prepared.diagonal[edge.col];
            let mut values = [C::default();5];
            if (difference*h).abs() < 1e-5 {
                values = ab_values_and_derivatives((prepared.diagonal[edge.row]+prepared.diagonal[edge.col])*0.5,h).1;
            } else { for k in 0..5 { values[k] = divide(diagonal[edge.row][k]-diagonal[edge.col][k],difference); } }
            for value in &mut values { *value = *value*edge.value; } recycling.push(values);
        }
        if diagonal.iter().chain(&recycling).flatten().any(|&z| !finite(z)) { return Err("Non-finite exponential AB4 coefficient".into()); }
        let mut actions: [RealSparse;5] = std::array::from_fn(|_|Vec::new());
        for k in 0..5 {
            let diagonal_terms = diagonal.iter().enumerate().map(|(i,values)|(i,i,values[k]));
            let recycling_terms = prepared.recycling.iter().zip(&recycling).map(|(edge,values)|(edge.row,edge.col,values[k]));
            actions[k] = physical_terms_to_real(prepared.n,diagonal_terms.chain(recycling_terms))?;
        }
        Ok(Self { actions })
    }
    fn apply(&self, state: &[f64], current: &[f64], history: &VecDeque<Vec<f64>>) -> Vec<f64> {
        let inputs = [state,current,history[0].as_slice(),history[1].as_slice(),history[2].as_slice()];
        let mut result = vec![0.;state.len()];
        for (matrix,input) in self.actions.iter().zip(inputs) {
            for (j,col) in matrix.iter().enumerate() { for &(i,value) in col { result[i] += value*input[j]; } }
        }
        result
    }
}
fn integrate_ab4(prepared: &Prepared, real: &RealPrepared, times: &[f64], subdivisions: usize) -> Result<AbRun,String> {
    let started = Instant::now(); let steps = subdivisions.checked_mul(times.len()-1).ok_or("AB4 step count overflow")?;
    if subdivisions == 0 || steps > 500_000 || steps.saturating_mul(real.rhs_nnz) > MAX_PRODUCTS {
        return Err("Exponential AB4 exceeds the short-time step/product budget".into());
    }
    let h = times[1]/subdivisions as f64; let before = Instant::now();
    let functions = AbFunctions::new(prepared,h)?; let function_ms = before.elapsed().as_secs_f64()*1000.;
    let mut startup = Stepper::new(prepared); let mut history = VecDeque::new();
    let mut state = real.initial.clone(); let mut states = vec![state.clone()];
    for step in 0..steps {
        let time = step as f64*h; let current = real.rhs(time,&state);
        let next = if step < 3 {
            let rho = expand_density(&real.coords,&state,prepared.n);
            let rho = startup.step(time,h,&rho)?;
            density_coordinates(&real.coords,&rho,prepared.n)
        } else { functions.apply(&state,&current,&history) };
        if next.iter().any(|x|!x.is_finite()||x.abs()>2.) {
            return Err(format!("Exponential AB4 unstable grid at t={time:e}us, subdivisions={subdivisions}"));
        }
        history.push_front(current); if history.len()>3 { history.pop_back(); }
        state = next;
        if (step+1)%subdivisions == 0 { states.push(state.clone()); }
        if (step+1)%4096 == 0 && std::env::var_os("ATOM_SOLVER_DEBUG").is_some() {
            eprintln!("Fixed real AB4 t={:.6}us steps={} h={h:e} elapsedMs={:.2}",(step+1)as f64*h,step+1,started.elapsed().as_secs_f64()*1000.);
        }
    }
    Ok(AbRun { states,subdivisions,steps,rhs_evaluations:steps+startup.rhs_evaluations,
        products:steps*real.rhs_nnz+startup.products,function_ms:function_ms+startup.function_ms,
        elapsed_ms:started.elapsed().as_secs_f64()*1000. })
}
fn ab_run_diagnostics(run: &AbRun) -> Value {
    json!({"subdivisionsPerSample":run.subdivisions,"steps":run.steps,"rhsEvaluations":run.rhs_evaluations,
        "entryProducts":run.products,"matrixFunctionMs":run.function_ms,"elapsedMs":run.elapsed_ms})
}
pub fn solve_ab4(lap: &Lapack, model: &Model, options: &Options) -> Result<Report,String> {
    if options.steady_only || !options.duration_us.is_finite() || options.duration_us<=0.
        || options.duration_us>1000. || options.samples<2 || options.samples>2001 {
        return Err("Exponential AB4 is a finite-time experiment for durations up to1000us and2..2001samples".into());
    }
    let started=Instant::now(); let prepared=Prepared::new(model)?; let real=RealPrepared::new(&prepared)?;
    if positivity(lap,&prepared.initial,prepared.n)?>1e-10 { return Err("Invalid initial density for exponential AB4".into()); }
    let times:Vec<_>=(0..options.samples).map(|i|options.duration_us*i as f64/(options.samples-1)as f64).collect();
    let maximum_step=prepared.phase_limit.min(options.duration_us/16.);
    let mut subdivisions=(times[1]/maximum_step).ceil().max(1.) as usize;
    let mut groups=BTreeMap::<String,Vec<usize>>::new();
    for(i,state)in model.states.iter().take(prepared.n).enumerate(){let parts:Vec<_>=state.split('-').collect();
        if parts.len()>=3&&matches!(parts[0],"ground"|"D1"|"D2"){groups.entry(format!("{}-{}",parts[0],parts[1])).or_default().push(i);}}
    let mut attempts=Vec::new(); let mut last_reason=String::new();
    for _ in 0..4 {
        let fine_subdivisions=subdivisions.checked_mul(3).and_then(|m|m.checked_add(1)).ok_or("AB4 refinement overflow")?;
        let coarse=match integrate_ab4(&prepared,&real,&times,subdivisions){Ok(run)=>run,Err(reason)=>{
            if !reason.contains("unstable grid"){return Err(reason);}last_reason=reason.clone();attempts.push(json!({"subdivisions":subdivisions,"rejected":reason}));subdivisions*=2;continue;}};
        // 3M+1 shares no integer step mesh with M. This is independent of
        // ETDRK4 startup and avoids simply reproducing a dyadic phase alias.
        let fine=match integrate_ab4(&prepared,&real,&times,fine_subdivisions){Ok(run)=>run,Err(reason)=>{
            if !reason.contains("unstable grid"){return Err(reason);}last_reason=reason.clone();attempts.push(json!({"subdivisions":subdivisions,"rejected":reason}));subdivisions*=2;continue;}};
        let mut population_error=0_f64;let mut group_error=0_f64;let mut trace_error=0_f64;let mut positivity_error=0_f64;
        for(a,b)in coarse.states.iter().zip(&fine.states){
            for i in 0..prepared.n{population_error=population_error.max((a[i]-b[i]).abs());}
            for indices in groups.values(){group_error=group_error.max(indices.iter().map(|&i|a[i]-b[i]).sum::<f64>().abs());}
            trace_error=trace_error.max((b[..prepared.n].iter().sum::<f64>()-1.).abs());
            positivity_error=positivity_error.max(positivity(lap,&expand_density(&real.coords,b,prepared.n),prepared.n)?);
        }
        attempts.push(json!({"coarse":ab_run_diagnostics(&coarse),"independentRefinement":ab_run_diagnostics(&fine),
            "populationDifference":population_error,"groupDifference":group_error,"traceError":trace_error,"positivityError":positivity_error}));
        if population_error.max(group_error)<=POPULATION_COMPARISON&&trace_error<=1e-7&&positivity_error<=1e-5 {
            let diagnostics=json!({"engine":"rust-fixed-etd-ab4-experiment","method":"Complete real density / nilpotent-recycling exponential AB4",
                "elapsedMs":started.elapsed().as_secs_f64()*1000.,"physicalHilbertDimension":prepared.n,"liouvilleDimension":prepared.n*prepared.n,
                "recyclingEntries":prepared.recycling.len(),"rhsEntries":real.rhs_nnz,"phaseStepLimitUs":prepared.phase_limit,
                "populationRefinementError":population_error,"groupPopulationRefinementError":group_error,"traceError":trace_error,"positivityError":positivity_error,
                "hermiticityError":0.,"attempts":attempts,
                "errorScope":"Observed M versus3M+1 population/F-group comparison and trace/PSD checks; no certified continuous-time error bound or infinite-time result."});
            return Ok(Report{times,states:fine.states,diagnostics,trace_error,positivity_error});
        }
        last_reason=format!("population={population_error:e}, Fgroup={group_error:e}, trace={trace_error:e}, positivity={positivity_error:e}");
        subdivisions*=2;
    }
    Err(format!("Exponential AB4 independent-grid verification did not converge: {last_reason}"))
}

/// Fixed, linear one-period action for a matrix-free monodromy solver.
/// The caller establishes periodicity and compares actual trajectories/means
/// under independent grid refinement. Basis vectors need not be physical
/// density matrices: no trace, positivity, magnitude, or normalization gate
/// is applied to them here.
pub struct PeriodAction {
    prepared: Prepared,
    real: RealPrepared,
    period: f64,
    substeps: usize,
    functions: AbFunctions,
    startup_functions: Rc<Functions>,
}

impl PeriodAction {
    pub fn new(model: &Model, period: f64, substeps: usize) -> Result<Self,String> {
        if !period.is_finite() || period <= 0. || substeps == 0 || substeps > 500_000 {
            return Err("Period action requires a positive finite period and1..500000fixed substeps".into());
        }
        let prepared=Prepared::new(model)?;let real=RealPrepared::new(&prepared)?;
        let h=period/substeps as f64;
        if h==0. { return Err("Period action step underflow".into()); }
        let functions=AbFunctions::new(&prepared,h)?;
        let startup_functions=Rc::new(prepared.functions(h)?);
        Ok(Self{prepared,real,period,substeps,functions,startup_functions})
    }
    pub fn dimension(&self)->usize { self.prepared.n*self.prepared.n }
    pub fn apply(&self,initial:&[f64])->Result<Vec<f64>,String> {
        self.advance(initial,self.period,self.substeps,&self.functions,self.startup_functions.clone())
    }
    /// `phase` is a duration in microseconds in [0,period], not a fraction.
    /// Evolution starts at drive phase t=0. This is the map after any integer
    /// number of periods only when the caller has established periodicity.
    pub fn partial(&self,initial:&[f64],phase:f64)->Result<Vec<f64>,String> {
        self.validate(initial)?;
        if !phase.is_finite()||phase<0.||phase>self.period { return Err("Partial period duration lies outside[0,period]".into()); }
        if phase==0. { return Ok(initial.to_vec()); }
        if phase==self.period { return self.apply(initial); }
        let substeps=(self.substeps as f64*phase/self.period).ceil().max(1.) as usize;
        let h=phase/substeps as f64;
        let functions=AbFunctions::new(&self.prepared,h)?;
        let startup=Rc::new(self.prepared.functions(h)?);
        self.advance(initial,phase,substeps,&functions,startup)
    }
    fn validate(&self,initial:&[f64])->Result<(),String> {
        if initial.len()!=self.dimension()||initial.iter().any(|x|!x.is_finite()) {
            return Err("Invalid full Hermitian-coordinate period-action vector".into());
        }
        Ok(())
    }
    fn advance(&self,initial:&[f64],duration:f64,substeps:usize,functions:&AbFunctions,startup_functions:Rc<Functions>)->Result<Vec<f64>,String> {
        self.validate(initial)?;let h=duration/substeps as f64;
        let mut startup=Stepper::new(&self.prepared);
        startup.functions.push_back((h.to_bits(),startup_functions));
        let mut state=initial.to_vec();let mut history=VecDeque::new();
        for step in 0..substeps {
            let time=step as f64*h;let current=self.real.rhs(time,&state);
            let next=if step<3 {
                let rho=expand_density(&self.real.coords,&state,self.prepared.n);
                let next=startup.step(time,h,&rho)?;
                density_coordinates(&self.real.coords,&next,self.prepared.n)
            }else{functions.apply(&state,&current,&history)};
            if next.iter().any(|x|!x.is_finite()) { return Err("Non-finite linear period-action result".into()); }
            history.push_front(current);if history.len()>3 {history.pop_back();}state=next;
        }
        Ok(state)
    }
    /// Sparse numerical preconditioner I-E(A0*T)-T*phi1(A0*T)*B_static.
    /// This is NOT I minus the physical period map. Every original drive and
    /// damping term remains in apply(); the caller must use that full action
    /// for all outer residuals and must impose its own normalization row.
    pub fn static_period_preconditioner(&self)->Result<Vec<Vec<(usize,f64)>>,String> {
        // In ETDRK coefficients at h=2T, E_half=exp(A0*T) and Q=T*phi1(A0*T).
        // Form these directly instead of canceling four AB history weights.
        let coefficients=self.prepared.functions(2.*self.period)?;
        let action=|which:usize| physical_terms_to_real(self.prepared.n,
            coefficients.diagonal.iter().enumerate().map(|(i,values)|(i,i,values[which]))
                .chain(self.prepared.recycling.iter().zip(&coefficients.recycling)
                    .map(|(edge,values)|(edge.row,edge.col,values[which]))));
        let exponential=action(1)?;let phi1=action(2)?;let d=self.dimension();
        let mut result:RealSparse=(0..d).map(|i|vec![(i,1.)]).collect();
        for j in 0..d {
            for &(i,value)in &exponential[j] {result[j].push((i,-value));}
            for &(middle,b)in &self.real.base[j] {
                for &(i,g)in &phi1[middle] {result[j].push((i,-g*b));}
            }
        }
        compress_real(&mut result);Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Drive as ModelDrive, Operator};
    fn two_level(omega: f64, coupling: f64) -> Model {
        Model { frequency_coordinates:None, dimension: 2, states: vec!["ground-0-0".into(),"D2-0-0".into()],
            h_real: vec![0., coupling, coupling, 1.3], h_imag: vec![0.;4],
            collapse: vec![Operator { real: vec![0.,1.,0.,0.], imag: vec![0.;4] }],
            initial_index: 0, initial_real: None, initial_imag: None, time_dependent: omega != 0.,
            drives: if omega == 0. { vec![] } else { vec![ModelDrive { omega,
                cos: Operator { real: vec![0.,0.2,0.2,0.], imag:vec![0.;4] },
                sin: Operator { real:vec![0.;4], imag:vec![0.,0.1,-0.1,0.] } }] } }
    }
    #[test]
    fn phi_values_and_nearly_coincident_derivatives_are_stable() {
        let zero = phi(C::default());
        for (a,b) in zero.iter().zip([1.,1.,0.5,1./6.,1./24.,1./120.]) { assert!((a.re-b).abs() < 1e-15 && a.im == 0.); }
        for lambda in [C::default(),C::new(-0.4,2.),C::new(-3.,-1e5)] {
            let (values, derivative) = values_and_derivatives(lambda,0.01);
            assert!(values.iter().chain(&derivative).all(|&z| finite(z)));
            let epsilon = 1e-5;
            let left = values_and_derivatives(lambda-C::new(epsilon,0.),0.01).0;
            let right = values_and_derivatives(lambda+C::new(epsilon,0.),0.01).0;
            for k in 0..6 { assert!(((right[k]-left[k])*(0.5/epsilon)-derivative[k]).abs() < 2e-9); }
        }
    }
    #[test]
    fn nilpotent_a0_exponential_matches_complete_dense_generator() {
        let lap = Lapack::load().unwrap(); let model = two_level(0.,0.);
        let prepared = Prepared::new(&model).unwrap(); let g = build_generator(&model).unwrap();
        let mut dense = vec![0.;16]; for (j,col) in g.matrix.iter().enumerate() { for &(i,a) in col { dense[i+j*4] += a; } }
        let initial = vec![0.3,0.7,0.1,-0.2]; let rho = expand_density(&g.coords,&initial,2);
        for t in [1e-6,0.03,2.] {
            let functions = prepared.functions(t).unwrap();
            let actual = prepared.combine(&functions,&[(&rho,0,1.)]);
            let expected = crate::dense_expm::action(&lap,&dense,&initial,&[t]).unwrap();
            let expected = expand_density(&g.coords,&expected[0],2);
            assert!(actual.iter().zip(expected).all(|(&a,b)|(a-b).abs()<1e-12));
        }
    }
    #[test]
    fn etd_keeps_offdiagonal_damping_and_matches_full_static_evolution() {
        let lap = Lapack::load().unwrap(); let mut model = two_level(0.,0.7);
        // A third state allows interfering excited columns and therefore
        // off-diagonal C^H C; that term must remain in B(t).
        model.dimension=3; model.states=vec![]; model.h_real=vec![0.,0.7,0.2,0.7,1.3,0.,0.2,0.,2.1];model.h_imag=vec![0.;9];
        model.collapse=vec![Operator{real:vec![0.,1.,0.4,0.,0.,0.,0.,0.,0.],imag:vec![0.;9]}];
        let options=Options{duration_us:2.,samples:21,include_asymptotic:false,..Options::default()};
        let actual=solve(&lap,&model,&options).unwrap();let g=build_generator(&model).unwrap();
        let mut dense=vec![0.;81];for(j,col)in g.matrix.iter().enumerate(){for &(i,a)in col{dense[i+j*9]+=a;}}
        let expected=crate::dense_expm::action(&lap,&dense,&g.initial,&actual.times).unwrap();
        let error=actual.states.iter().zip(expected).flat_map(|(a,b)|a.iter().zip(b).map(|(a,b)|(a-b).abs())).fold(0.,f64::max);
        assert!(error<2e-6,"Complete ETD static density error {error:e}");
    }
    #[test]
    fn etd_time_dependent_two_level_matches_independent_full_fourier() {
        let lap=Lapack::load().unwrap();let model=two_level(3.7,0.35);
        let options=Options{duration_us:2.,samples:21,include_asymptotic:false,..Options::default()};
        let actual=solve(&lap,&model,&options).unwrap();
        let expected=crate::floquet::solve(&lap,&model,&options).unwrap();
        let error=actual.states.iter().zip(expected["populations"].as_array().unwrap()).flat_map(|(a,b)|{
            a[..2].iter().zip(b.as_array().unwrap()).map(|(a,b)|(a-b.as_f64().unwrap()).abs())
        }).fold(0.,f64::max);
        assert!(error<2e-6,"Complete time-dependent ETD population error {error:e}");
    }
    #[test]
    fn exponential_ab4_zero_generator_has_classical_history_coefficients() {
        let h=0.3;let(values,_)=ab_values_and_derivatives(C::default(),h);
        assert_eq!(values[0].re,1.);
        for(value,expected)in values[1..].iter().zip([55./24.,-59./24.,37./24.,-9./24.]){
            assert!((value.re/h-expected).abs()<2e-15&&value.im==0.);
        }
    }
    #[test]
    fn exponential_ab4_real_products_match_complete_complex_operators() {
        let model=two_level(3.7,0.35);let prepared=Prepared::new(&model).unwrap();let real=RealPrepared::new(&prepared).unwrap();
        let state=vec![0.3,0.7,0.1,-0.2];let rho=expand_density(&real.coords,&state,2);
        let mut reference=Stepper::new(&prepared);
        let complex=reference.rhs(0.137,&rho).unwrap();let expected=density_coordinates(&real.coords,&complex,2);
        let actual=real.rhs(0.137,&state);
        assert!(actual.iter().zip(expected).all(|(a,b)|(a-b).abs()<1e-13));
        let functions=AbFunctions::new(&prepared,0.03).unwrap();let mut actual=vec![0.;4];
        for(j,col)in functions.actions[0].iter().enumerate(){for &(i,a)in col{actual[i]+=a*state[j];}}
        let complex=prepared.combine(&prepared.functions(0.03).unwrap(),&[(&rho,0,1.)]);
        let expected=density_coordinates(&real.coords,&complex,2);
        assert!(actual.iter().zip(expected).all(|(a,b)|(a-b).abs()<1e-13));
    }
    #[test]
    fn exponential_ab4_time_dependent_matches_verified_full_rk4() {
        let lap=Lapack::load().unwrap();let model=two_level(3.7,0.35);
        let options=Options{duration_us:2.,samples:21,include_asymptotic:false,..Options::default()};
        let actual=solve_ab4(&lap,&model,&options).unwrap();let expected=solve(&lap,&model,&options).unwrap();
        let error=actual.states.iter().zip(expected.states).flat_map(|(a,b)|a.iter().zip(b).map(|(a,b)|(a-b).abs())).fold(0.,f64::max);
        assert!(error<2e-6,"Real exponential AB4 full-density discrepancy {error:e}");
    }
    #[test]
    fn exponential_ab4_keeps_offdiagonal_gram_against_full_dense_evolution() {
        let lap=Lapack::load().unwrap();let mut model=two_level(0.,0.7);
        model.dimension=3;model.states=vec![];model.h_real=vec![0.,0.7,0.2,0.7,1.3,0.,0.2,0.,2.1];model.h_imag=vec![0.;9];
        model.collapse=vec![Operator{real:vec![0.,1.,0.4,0.,0.,0.,0.,0.,0.],imag:vec![0.;9]}];
        let options=Options{duration_us:2.,samples:21,include_asymptotic:false,..Options::default()};
        let actual=solve_ab4(&lap,&model,&options).unwrap();let g=build_generator(&model).unwrap();
        let mut dense=vec![0.;81];for(j,col)in g.matrix.iter().enumerate(){for &(i,a)in col{dense[i+j*9]+=a;}}
        let expected=crate::dense_expm::action(&lap,&dense,&g.initial,&actual.times).unwrap();
        let error=actual.states.iter().zip(expected).flat_map(|(a,b)|a.iter().zip(b).map(|(a,b)|(a-b).abs())).fold(0.,f64::max);
        assert!(error<2e-6,"Real exponential AB4 complete damping error {error:e}");
    }
    #[test]
    fn period_action_is_linear_and_partial_uses_a_duration() {
        let model=two_level(3.7,0.35);let period=std::f64::consts::TAU/3.7;
        let action=PeriodAction::new(&model,period,64).unwrap();assert_eq!(action.dimension(),4);
        let x=vec![0.4,-0.2,0.13,0.01];let y=vec![-0.1,0.7,-0.08,0.2];
        let sum:Vec<_>=x.iter().zip(&y).map(|(a,b)|2.*a-0.3*b).collect();
        let a=action.apply(&x).unwrap();let b=action.apply(&y).unwrap();let actual=action.apply(&sum).unwrap();
        assert!(actual.iter().zip(a.iter().zip(b)).all(|(v,(a,b))|(v-(2.*a-0.3*b)).abs()<2e-13));
        assert_eq!(action.partial(&x,0.).unwrap(),x);
        assert_eq!(action.partial(&x,period).unwrap(),a);
        let half=action.partial(&x,0.5*period).unwrap();
        let expected=PeriodAction::new(&model,0.5*period,32).unwrap().apply(&x).unwrap();
        assert_eq!(half,expected);
        assert!(action.partial(&x,-0.1).is_err());
    }
    #[test]
    fn static_period_preconditioner_equals_identity_minus_action_when_b_is_zero() {
        let model=two_level(0.,0.);let action=PeriodAction::new(&model,0.2,16).unwrap();
        let matrix=action.static_period_preconditioner().unwrap();let x=vec![0.3,0.7,0.1,-0.2];
        let px=action.apply(&x).unwrap();let mut actual=vec![0.;4];
        for(j,col)in matrix.iter().enumerate(){for &(i,a)in col{actual[i]+=a*x[j];}}
        assert!(actual.iter().zip(x.iter().zip(px)).all(|(a,(x,p))|(a-(x-p)).abs()<2e-13));
    }
}
