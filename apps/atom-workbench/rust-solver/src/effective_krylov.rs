//! Rational Krylov exponential action, retaining the original full Lindblad
//! operator. Two nested projection spaces are compared at every output sample.
use super::*;
fn trajectory(
    lap: &Lapack,
    a: &[f64],
    initial: &[f64],
    steady: &[f64],
    trace_vector: &[f64],
    dt: f64,
    samples: usize,
    tolerance: f64,
) -> Result<(Vec<Vec<f64>>, f64, usize), String> {
    let n = initial.len();
    let mut shifted = a.to_vec();
    let shift = 10.;
    for i in 0..n {
        shifted[i + i * n] -= shift;
    }
    let (lu, piv) = lap.factor(&shifted, n)?;
    let norm = initial.iter().map(|v| v * v).sum::<f64>().sqrt();
    let mut columns = vec![initial.iter().map(|v| v / norm).collect::<Vec<_>>()];
    let mut previous: Option<Vec<Vec<f64>>> = None;
    let mut last_difference = f64::INFINITY;
    for m in 1..=128.min(n) {
        if m > 1 {
            let mut v = lap.solve_factored(&lu, &piv, &columns[m - 2], n)?;
            let trace = v.iter().zip(trace_vector).map(|(a, b)| a * b).sum::<f64>();
            for (value, s) in v.iter_mut().zip(steady) {
                *value -= trace * s;
            }
            for _ in 0..2 {
                for q in &columns {
                    let d = q.iter().zip(&v).map(|(a, b)| a * b).sum::<f64>();
                    for (v, q) in v.iter_mut().zip(q) {
                        *v -= d * q;
                    }
                }
            }
            let length = v.iter().map(|v| v * v).sum::<f64>().sqrt();
            if length < 1e-14 {
                break;
            }
            for v in &mut v {
                *v /= length;
            }
            columns.push(v);
        }
        if m < 24 || m % 16 != 0 {
            continue;
        }
        let v: Vec<f64> = columns.iter().flatten().copied().collect();
        let av = lap.multiply(a, &v, n, m, n);
        let mut projected = vec![0.; m * m];
        for i in 0..m {
            for j in 0..m {
                projected[i + j * m] = (0..n).map(|k| v[k + i * n] * av[k + j * n]).sum();
            }
        }
        let (step, _) = dense_expm::exponential(lap, &projected, m, dt)?;
        let mut state = vec![0.; m];
        state[0] = norm;
        let mut rows = Vec::new();
        for sample in 0..samples {
            if sample > 0 {
                state = lap.multiply(&step, &state, m, 1, m);
            }
            rows.push(lap.multiply(&v, &state, n, 1, m));
        }
        if let Some(old) = previous {
            let difference = rows
                .iter()
                .zip(&old)
                .flat_map(|(a, b)| {
                    a.iter()
                        .zip(b)
                        .map(|(a, b)| (a - b).abs())
                        .collect::<Vec<_>>()
                })
                .fold(0., f64::max);
            let population_difference = rows
                .iter()
                .zip(&old)
                .flat_map(|(a, b)| {
                    a.iter()
                        .zip(b)
                        .zip(trace_vector)
                        .filter(|(_, weight)| **weight != 0.)
                        .map(|((a, b), _)| (a - b).abs())
                        .collect::<Vec<_>>()
                })
                .fold(0., f64::max);
            last_difference = difference;
            if population_difference < 5e-5 && difference < tolerance {
                return Ok((rows, population_difference, m));
            }
        }
        previous = Some(rows);
    }
    Err(format!(
        "Rational Krylov subspace refinement did not converge; difference={last_difference:e}"
    ))
}
pub fn solve(solver: &mut Solver, model: &Model, options: &Options) -> Result<Value, String> {
    if model.time_dependent || !model.drives.is_empty() {
        return Err("Nonstatic effective Hamiltonian".into());
    }
    let started = Instant::now();
    let g = build_generator(model)?;
    let inds = reachable(&g.matrix, &g.initial);
    let r = inds.len();
    let mut positions = vec![usize::MAX; g.n * g.n];
    for (i, &k) in inds.iter().enumerate() {
        positions[k] = i;
    }
    let mut a = vec![0.; r * r];
    for (j, &k) in inds.iter().enumerate() {
        for &(row, v) in &g.matrix[k] {
            let i = positions[row];
            if i != usize::MAX {
                a[i + j * r] = v;
            }
        }
    }
    let construction = started.elapsed().as_secs_f64() * 1000.;
    let stationary = solver.stationary_only(model, &g, &a, &inds, started, construction)?;
    if stationary.is_none() {
        return Err("Initial-dependent steady state needs spectral projection".into());
    }
    if options.steady_only {
        return stationary.ok_or("No stationary state requested".into());
    }
    if options.include_asymptotic {publish_mean(stationary.as_ref().unwrap().clone());}
    if !(options.duration_us > 0. && options.duration_us.is_finite())
        || options.samples < 2
        || options.samples > 2001
    {
        return Err("Invalid trajectory interval".into());
    }
    let dt = options.duration_us / (options.samples - 1) as f64;
    let stationary_value = stationary.as_ref().unwrap();
    let stationary_real = stationary_value["asymptoticDensityReal"]
        .as_array()
        .ok_or("Missing stationary real density")?;
    let stationary_imag = stationary_value["asymptoticDensityImag"]
        .as_array()
        .ok_or("Missing stationary imaginary density")?;
    let steady: Vec<f64> = inds
        .iter()
        .map(|&i| match g.coords[i] {
            Coordinate::Population(a) => stationary_real[a * g.n + a].as_f64().unwrap(),
            Coordinate::Real(a, b) => stationary_real[a * g.n + b].as_f64().unwrap(),
            Coordinate::Imag(a, b) => stationary_imag[a * g.n + b].as_f64().unwrap(),
        })
        .collect();
    let initial: Vec<_> = inds
        .iter()
        .enumerate()
        .map(|(i, &k)| g.initial[k] - steady[i])
        .collect();
    let trace_vector: Vec<f64> = inds
        .iter()
        .map(|&i| if i < g.n { 1. } else { 0. })
        .collect();
    let (rows, error, dimension) = trajectory(
        &solver.lap,
        &a,
        &initial,
        &steady,
        &trace_vector,
        dt,
        options.samples,
        1e-3,
    )?;
    // Finite-trajectory numerical positivity tolerance is 1e-4, one tenth
    // of the requested 1e-3 absolute population accuracy. No eigenvalue clipping
    // or density renormalization is performed; actual defect is returned.
    let mut pops = Vec::new();
    let mut scatter = Vec::new();
    let mut real = Vec::new();
    let mut imag = Vec::new();
    let mut trace = 0_f64;
    let mut psd = 0_f64;
    for row in rows {
        let mut x = vec![0.; g.n * g.n];
        for (i, &j) in inds.iter().enumerate() {
            x[j] = row[i] + steady[i];
        }
        let rho = expand_density(&g.coords, &x, g.n);
        trace = trace.max((x[..g.n].iter().sum::<f64>() - 1.).abs());
        psd = psd.max(positivity(&solver.lap, &rho, g.n)?);
        pops.push(x[..g.n].to_vec());
        scatter.push(scatter_rate(&g.scattering, &rho, g.n));
        if options.return_density {
            real.push(rho.iter().map(|z| z.re).collect::<Vec<_>>());
            imag.push(rho.iter().map(|z| z.im).collect::<Vec<_>>());
        }
    }
    if !error.is_finite() || trace > 5e-6 || psd > 1e-4 {
        return Err(format!("Krylov trace/positivity failed: {trace:e}/{psd:e}"));
    }
    let mut result=if options.include_asymptotic{stationary}else{None}.unwrap_or_else(||json!({"asymptoticPopulations":null,"asymptoticScatteringRatePerUs":null,"asymptoticStatus":"not-requested","diagnostics":{}}));
    result["states"] = json!(model.states);
    result["timesUs"] = json!((0..options.samples)
        .map(|i| i as f64 * dt)
        .collect::<Vec<_>>());
    result["populations"] = json!(pops);
    result["scatteringRatePerUs"] = json!(scatter);
    if options.return_density {
        result["densityReal"] = json!(real);
        result["densityImag"] = json!(imag);
    }
    let d = result["diagnostics"].as_object_mut().unwrap();
    d.insert(
        "method".into(),
        json!("Rust / full-density rational Krylov exponential action"),
    );
    d.insert("engine".into(), json!("rust-krylov"));
    d.insert("stationaryOnly".into(), json!(false));
    d.insert(
        "elapsedMs".into(),
        json!(started.elapsed().as_secs_f64() * 1000.),
    );
    d.insert("constructionMs".into(), json!(construction));
    d.insert("traceError".into(), json!(trace));
    d.insert("positivityError".into(), json!(psd));
    d.insert("hermiticityError".into(), json!(0.));
    d.insert("integrationRefinementError".into(), json!(error));
    d.insert("liouvilleDimension".into(), json!(g.n * g.n));
    d.insert("reachableDimension".into(), json!(r));
    d.insert("krylovDimension".into(), json!(dimension));
    Ok(result)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn complete_three_beam_density_matches_high_accuracy_spectral_populations() {
        let request: Request =
            serde_json::from_str(include_str!("../tests/fixtures/effective-three-beam.json"))
                .unwrap();
        let model = request.model.unwrap();
        let mut fast_solver = Solver::new().unwrap();
        let mut reference_solver = Solver::new().unwrap();
        let fast = solve(&mut fast_solver, &model, &request.options).unwrap();
        let reference = reference_solver.solve(&model, &request.options).unwrap();
        let mut error = 0_f64;
        for (a, b) in fast["populations"]
            .as_array()
            .unwrap()
            .iter()
            .zip(reference["populations"].as_array().unwrap())
        {
            for (a, b) in a.as_array().unwrap().iter().zip(b.as_array().unwrap()) {
                error = error.max((a.as_f64().unwrap() - b.as_f64().unwrap()).abs());
            }
        }
        assert!(error < 1e-3, "population error {error:e}");
        assert!(fast["diagnostics"]["traceError"].as_f64().unwrap() < 1e-6);
        assert!(fast["diagnostics"]["positivityError"].as_f64().unwrap() < 1e-4);
    }
}
