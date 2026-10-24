//! Atomic radiation entry point. Generic numerical fixtures deliberately keep
//! the independent C-only API; an atomic request never falls back to it before
//! the common-dipole emitted-frequency bath has actually been constructed.
use super::*;
use crate::radiative_bath::{self, Radiation};
use std::cell::RefCell;
use std::collections::BTreeMap;

pub(super) fn same_radiation(a: Option<&Radiation>, b: Option<&Radiation>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            a.reference_angular_frequency == b.reference_angular_frequency
                && a.carrier_angular_frequency == b.carrier_angular_frequency
                && a.dipoles.len() == b.dipoles.len()
                && a.dipoles
                    .iter()
                    .zip(&b.dipoles)
                    .all(|(a, b)| a.real == b.real && a.imag == b.imag)
        }
        _ => false,
    }
}
struct MeanCache {
    model: Model,
    radiation: Radiation,
    value: Value,
}
thread_local! {static MEANS:RefCell<VecDeque<MeanCache>>=const{RefCell::new(VecDeque::new())};}
fn same_initial(a: &Model, b: &Model) -> bool {
    a.initial_index == b.initial_index
        && a.initial_real == b.initial_real
        && a.initial_imag == b.initial_imag
        && a.states == b.states
}
fn adjoint(a: &[C], n: usize) -> Vec<C> {
    (0..n * n).map(|i| a[(i % n) * n + i / n].conj()).collect()
}
fn multiply(a: &[C], b: &[C], n: usize) -> Vec<C> {
    let mut result = vec![C::default(); n * n];
    for i in 0..n {
        for k in 0..n {
            let value = a[i * n + k];
            if value.re == 0. && value.im == 0. {
                continue;
            }
            for j in 0..n {
                result[i * n + j] += value * b[k * n + j];
            }
        }
    }
    result
}
fn mapped(index: &[i32], mapping: &[Vec<i32>], axes: usize) -> Vec<i32> {
    let mut result = vec![0; axes];
    for (&value, axis) in index.iter().zip(mapping) {
        for (i, &a) in axis.iter().enumerate() {
            result[i] += value * a;
        }
    }
    result
}
fn difference(a: &[i32], b: &[i32]) -> Vec<i32> {
    a.iter().zip(b).map(|(a, b)| a - b).collect()
}
fn add_fourier(target: &mut BTreeMap<Vec<i32>, Vec<C>>, index: Vec<i32>, value: &[C]) {
    let matrix = target
        .entry(index)
        .or_insert_with(|| vec![C::default(); value.len()]);
    for (a, &b) in matrix.iter_mut().zip(value) {
        *a += b;
    }
}
fn complex_density(real: &[f64], imag: &[f64], coords: &[Coordinate], n: usize) -> Vec<C> {
    let mut rho = vec![C::default(); n * n];
    for (k, &coordinate) in coords.iter().enumerate() {
        let v = C::new(real[k], imag[k]);
        match coordinate {
            Coordinate::Population(i) => rho[i * n + i] += v,
            Coordinate::Real(i, j) => {
                rho[i * n + j] += v;
                rho[j * n + i] += v;
            }
            Coordinate::Imag(i, j) => {
                rho[i * n + j] += C::new(-v.im, v.re);
                rho[j * n + i] += C::new(v.im, -v.re);
            }
        }
    }
    rho
}
fn physical_mean(frame: &multitone_frame::Frame, output: &generator_mean::Output) -> (Vec<C>, f64) {
    let n = frame.dimension;
    let coords = coordinates(n);
    let axes = output.frequencies.len();
    let mut densities = BTreeMap::new();
    for coefficient in &output.coefficients {
        let rho = complex_density(&coefficient.real, &coefficient.imag, &coords, n);
        densities.insert(coefficient.index.clone(), rho.clone());
        if coefficient.index.iter().any(|&i| i != 0) {
            densities.insert(
                coefficient.index.iter().map(|v| -v).collect(),
                adjoint(&rho, n),
            );
        }
    }
    let mut physical_modes = BTreeMap::new();
    for (index, p) in frame.physical_modes() {
        add_fourier(
            &mut physical_modes,
            mapped(&index, &output.input_axis_indices, axes),
            &p,
        );
    }
    let mut mean = vec![C::default(); n * n];
    for (a, pa) in &physical_modes {
        for (b, pb) in &physical_modes {
            if let Some(rho) = densities.get(&difference(b, a)) {
                let term = multiply(&multiply(pa, rho, n), &adjoint(pb, n), n);
                for (v, &term) in mean.iter_mut().zip(&term) {
                    *v += term;
                }
            }
        }
    }
    let mut scattering = C::default();
    for channel in &frame.collapse {
        let mut coefficients = BTreeMap::new();
        for (index, c) in channel {
            add_fourier(
                &mut coefficients,
                mapped(index, &output.input_axis_indices, axes),
                c,
            );
        }
        for (a, ca) in &coefficients {
            let left = adjoint(ca, n);
            for (b, cb) in &coefficients {
                if let Some(rho) = densities.get(&difference(a, b)) {
                    let gram = multiply(&left, cb, n);
                    for i in 0..n {
                        for j in 0..n {
                            scattering += gram[i * n + j] * rho[j * n + i];
                        }
                    }
                }
            }
        }
    }
    (mean, scattering.re)
}

fn periodic_mean(lap: &Lapack, model: &Model, radiation: &Radiation) -> Result<Value, String> {
    if let Some(projected) = small_periodic::project(model, radiation)? {
        let mut result = periodic_mean(lap, &projected.model, &projected.radiation)?;
        for key in ["asymptoticPopulations"] {
            if let Some(row) = result[key].as_array() {
                let values: Vec<_> = row.iter().map(|v| v.as_f64().unwrap()).collect();
                result[key] = json!(small_periodic::lift_populations(
                    &values,
                    &projected.support,
                    model.dimension
                ));
            }
        }
        for key in ["asymptoticDensityReal", "asymptoticDensityImag"] {
            if let Some(row) = result[key].as_array() {
                let mut lifted = vec![0.; model.dimension * model.dimension];
                let n = projected.support.len();
                for (i, &a) in projected.support.iter().enumerate() {
                    for (j, &b) in projected.support.iter().enumerate() {
                        lifted[a * model.dimension + b] =
                            row[i * n + j].as_f64().ok_or("Invalid mean density")?;
                    }
                }
                result[key] = json!(lifted);
            }
        }
        if let Some(rows) = result["populations"].as_array() {
            let values: Vec<Vec<f64>> = rows
                .iter()
                .map(|row| {
                    small_periodic::lift_populations(
                        &row.as_array()
                            .unwrap()
                            .iter()
                            .map(|v| v.as_f64().unwrap())
                            .collect::<Vec<_>>(),
                        &projected.support,
                        model.dimension,
                    )
                })
                .collect();
            result["populations"] = json!(values);
        }
        result["states"] = json!(model.states);
        result["diagnostics"]["stationaryPhysicalHilbertDimension"] =
            json!(projected.support.len());
        result["diagnostics"]["stationaryInvariantSupport"] = json!(projected.support);
        result["diagnostics"]["globalUniquenessEstablished"] = json!(false);
        result["diagnostics"]["initialDependenceScope"] =
            json!("within the stated invariant Hilbert support");
        return Ok(result);
    }
    let cached = MEANS.with(|cache| {
        let mut cache = cache.borrow_mut();
        let i = cache.iter().position(|entry| {
            fast_periodic::same_operator(&entry.model, model)
                && same_initial(&entry.model, model)
                && same_radiation(Some(&entry.radiation), Some(radiation))
        });
        i.map(|i| cache.remove(i).unwrap())
    });
    if let Some(entry) = cached {
        let mut result = entry.value.clone();
        result["diagnostics"]["stationaryCached"] = json!(true);
        result["diagnostics"]["elapsedMs"] = json!(0.);
        MEANS.with(|cache| cache.borrow_mut().push_back(entry));
        return Ok(result);
    }
    let started = Instant::now();
    let original = build_generator(model)?;
    let initial = expand_density(&original.coords, &original.initial, model.dimension);
    let gauges = multitone_frame::resonant_gauges(model)?;
    let mut previous: Option<Vec<C>> = None;
    let mut last_error = String::new();
    let maximum_order = if model.dimension <= 4 && model.drives.len() == 1 {
        12
    } else {
        4
    };
    for order in 1..=maximum_order {
        let frame = match multitone_frame::prepare_with_radiation(
            lap,
            model,
            order,
            &gauges,
            Some(radiation),
        ) {
            Ok(frame) => frame,
            Err(error) => {
                last_error = error;
                continue;
            }
        };
        let base_model = frame.base_model(model, &initial);
        let base = match hilbert_floquet::build_density_generator(lap, &base_model) {
            Ok(base) => base,
            Err(error) if error.contains("initial density") => {
                last_error = error;
                continue;
            }
            Err(error) => return Err(error),
        };
        let sigma0 = frame.into_frame(&initial);
        let input = generator_mean::Input {
            n: model.dimension,
            base: base.matrix,
            initial: base.initial,
            drives: vec![],
            frequencies: frame.frequencies.clone(),
            hilbert_support: frame.invariant_hilbert_support(&sigma0),
            bounded_generator_tail: 0.,
        };
        let catalog = frame.generator_indices();
        let mean_options = generator_mean::Options {
            max_order: maximum_order,
            max_entries: if model.dimension == 32 && frame.frequencies.len() > 1 {
                160_000_000
            } else {
                32_000_000
            },
            max_coordinates: if model.dimension == 32 && frame.frequencies.len() > 1 {
                18_432
            } else {
                32_768
            },
        };
        let mut boundary = |coefficients: &[harmonics::FourierCoefficient],
                            retained: &[Vec<i32>],
                            mapping: &[Vec<i32>]| {
            radiative_boundary::evaluate(lap, &frame, coefficients, retained, mapping)
        };
        let output = match generator_mean::solve_streamed_with_boundary(
            lap,
            &input,
            &catalog,
            |index| {
                let h = frame.generator_harmonic(lap, index)?;
                Ok(harmonics::FourierDrive {
                    index: h.index,
                    cos: h.cos,
                    sin: h.sin,
                })
            },
            &mut boundary,
            mean_options,
        ) {
            Ok(output) => output,
            Err(error) => {
                last_error = error;
                continue;
            }
        };
        let (mean, scattering) = physical_mean(&frame, &output);
        let trace = (0..model.dimension)
            .map(|i| mean[i * model.dimension + i].re)
            .sum::<f64>();
        let psd = positivity(lap, &mean, model.dimension)?;
        let hermiticity = mean
            .iter()
            .zip(adjoint(&mean, model.dimension))
            .map(|(&a, b)| (a - b).abs())
            .fold(0., f64::max);
        if mean.iter().any(|z| !z.re.is_finite() || !z.im.is_finite())
            || !scattering.is_finite()
            || scattering < -1e-10
            || (trace - 1.).abs() > 1e-8
            || psd > 1e-8
            || hermiticity > 1e-8
        {
            return Err(
                "Common-vacuum physical mean failed normalization/positivity/Hermiticity checks"
                    .into(),
            );
        }
        // Opt-in scientific audit: the complete solution and its actual
        // numerical coordinate frame must be exported together.
        if let Some(folder) = std::env::var_os("ATOM_RADIATIVE_AUDIT_DIR") {
            let folder = std::path::PathBuf::from(folder);
            std::fs::create_dir_all(&folder).map_err(|error| error.to_string())?;
            let encode = |matrix: &[C]| json!({"real":matrix.iter().map(|z|z.re).collect::<Vec<_>>(),"imag":matrix.iter().map(|z|z.im).collect::<Vec<_>>()});
            let modes: Vec<_> = frame
                .physical_modes()
                .iter()
                .map(|(index, matrix)| json!({"index":index,"matrix":encode(matrix)}))
                .collect();
            let collapse: Vec<Vec<_>> = frame
                .collapse
                .iter()
                .map(|channel| {
                    channel
                        .iter()
                        .map(|(index, matrix)| json!({"index":index,"matrix":encode(matrix)}))
                        .collect()
                })
                .collect();
            let coefficients:Vec<_>=output.coefficients.iter().map(|coefficient|json!({"index":coefficient.index,"real":coefficient.real,"imag":coefficient.imag})).collect();
            let record = json!({"modelSchema":ATOMIC_MODEL_SCHEMA,"dimension":model.dimension,"matrixOrder":"row-major","frameHarmonicOrder":order,"frequencies":frame.frequencies,"gauges":frame.gauges,"hamiltonian":encode(&frame.hamiltonian),"physicalModes":modes,"collapse":collapse,"initialFrameDensity":encode(&sigma0),"nativeMean":{"coefficients":coefficients,"frequencies":output.frequencies,"inputAxisIndices":output.input_axis_indices,"diagnostics":output.diagnostics},"physicalMean":encode(&mean),"physicalMeanScatteringPerUs":scattering});
            std::fs::write(
                folder.join(format!("frame-{order}-mean.json")),
                serde_json::to_vec(&record).map_err(|error| error.to_string())?,
            )
            .map_err(|error| error.to_string())?;
        }
        let difference = previous
            .as_ref()
            .map(|old| {
                mean.iter()
                    .zip(old)
                    .map(|(&a, &b)| (a - b).abs())
                    .fold(0., f64::max)
            })
            .unwrap_or(f64::INFINITY);
        if difference < 1e-8 {
            let result = json!({"states":model.states,"timesUs":[0.],"populations":[original.initial[..model.dimension].to_vec()],"scatteringRatePerUs":[frame.scattering_rate(&sigma0,0.)],
                "asymptoticPopulations":(0..model.dimension).map(|i|mean[i*model.dimension+i].re).collect::<Vec<_>>(),"asymptoticScatteringRatePerUs":scattering,"asymptoticStatus":"periodic-mean",
                "asymptoticDensityReal":mean.iter().map(|z|z.re).collect::<Vec<_>>(),"asymptoticDensityImag":mean.iter().map(|z|z.im).collect::<Vec<_>>(),
                "diagnostics":{"method":"Rust / full-density common-vacuum Fourier mean","steadyMethod":"Full Hermitian operator Fourier equation with emitted-frequency common dipoles","engine":"rust-floquet","radiationModel":ATOMIC_MODEL_SCHEMA,"radiationChannels":3,
                    "elapsedMs":started.elapsed().as_secs_f64()*1000.,"stationaryOnly":true,"stationaryCached":false,"initialDependent":false,"densityLimitExists":false,"asymptoticStatus":"periodic-mean",
                    "asymptoticReason":"Long-time phase average from the complete trace-constrained Fourier density equation; not inferred from a finite-time plateau or finite-time tail bound.",
                    "traceError":(trace-1.).abs(),"positivityError":psd,"hermiticityError":hermiticity,"frameHarmonicOrder":order,"frameConvergenceError":difference,"frameUnitarityBound":frame.unitarity_bound,
                    "frameSchrodingerResidual":frame.schrodinger_residual,"harmonicOrder":output.diagnostics["harmonicOrder"],"harmonicConvergenceError":output.diagnostics["uniformFourierDifference"],"residual":output.diagnostics["residual"],"boundaryForcingMax":output.diagnostics["boundaryForcingUniformMax"],
                    "stationaryOperatorDimension":output.diagnostics["operatorDimension"],"stationarySystemConditionEstimate":output.diagnostics["conditionEstimate"],
                    "independentBeats":output.frequencies.len(),"frequencyBasis":output.frequencies,"frequencyReconstructionErrors":output.diagnostics["frequencyReconstructionErrors"],"meanFourierDiagnostics":output.diagnostics,
                    "uniquenessScope":"full Hermitian algebra on the invariant Hilbert support for every retained Fourier harmonic"}});
            MEANS.with(|cache| {
                let mut cache = cache.borrow_mut();
                cache.push_back(MeanCache {
                    model: model.clone(),
                    radiation: radiation.clone(),
                    value: result.clone(),
                });
                while cache.len() > 8 {
                    cache.pop_front();
                }
            });
            return Ok(result);
        }
        last_error = format!("Radiative-frame mean has not converged: {difference:e}");
        previous = Some(mean);
    }
    if model.dimension <= 4 && model.drives.len() == 1 {
        let result = small_periodic::periodic_mean(lap, model, radiation)?;
        MEANS.with(|cache| {
            let mut cache = cache.borrow_mut();
            cache.push_back(MeanCache {
                model: model.clone(),
                radiation: radiation.clone(),
                value: result.clone(),
            });
            while cache.len() > 8 {
                cache.pop_front();
            }
        });
        return Ok(result);
    }
    Err(format!(
        "No verified common-vacuum long-time mean is available: {last_error}"
    ))
}

pub(super) fn solve(
    solver: &mut Solver,
    model: &Model,
    radiation: &Radiation,
    options: &Options,
) -> Result<Value, String> {
    if model.dimension != 32 {
        return Err("rb87-dipole/2 requires all32 internal states".into());
    }
    radiation.normalized_dipoles(model.dimension)?;
    if model.drives.is_empty() && !model.time_dependent {
        let started = Instant::now();
        let (corrected, bath) = radiative_bath::constant_dressed(&solver.lap, model, radiation)?;
        let mut result = solver.solve(&corrected, options)?;
        result["diagnostics"]["radiationModel"] = json!(ATOMIC_MODEL_SCHEMA);
        result["diagnostics"]["radiationChannels"] = json!(bath.polarization_channels);
        result["diagnostics"]["radiationPreparationMs"] = json!(bath.preparation_ms);
        result["diagnostics"]["elapsedMs"] = json!(started.elapsed().as_secs_f64() * 1000.);
        return Ok(result);
    }
    if options.steady_only {
        return periodic_mean(&solver.lap, model, radiation);
    }
    if !options.duration_us.is_finite()
        || options.duration_us <= 0.
        || options.samples < 2
        || options.samples > 2001
    {
        return Err("Invalid atomic trajectory interval or sample count".into());
    }
    let started = Instant::now();
    let projected = small_periodic::project(model, radiation)?;
    let fast = || {
        if model.drives.len() == 1 {
            fast_periodic::trajectory_with_radiation(&solver.lap, model, Some(radiation), options)
                .or_else(|_| {
                    fast_multitone::trajectory_with_radiation(
                        &solver.lap,
                        model,
                        Some(radiation),
                        options,
                    )
                })
        } else {
            fast_multitone::trajectory_with_radiation(&solver.lap, model, Some(radiation), options)
        }
    };
    let trajectory = if let Some(projection) = projected {
        small_periodic::trajectory(&solver.lap, model, &projection, options).or_else(|direct| {
            fast().map_err(|fallback| {
                format!(
                    "Direct invariant-support solver: {direct}; full-space fallback: {fallback}"
                )
            })
        })
    } else {
        fast()
    }?;
    let mut result = if options.include_asymptotic {
        match periodic_mean(&solver.lap, model, radiation) {
            Ok(value) => value,
            Err(error) => {
                json!({"asymptoticPopulations":null,"asymptoticScatteringRatePerUs":null,"asymptoticStatus":"not-converged","diagnostics":{"asymptoticStatus":"not-converged","asymptoticReason":error}})
            }
        }
    } else {
        json!({"asymptoticPopulations":null,"asymptoticScatteringRatePerUs":null,"asymptoticStatus":"not-requested","diagnostics":{}})
    };
    result["states"] = json!(model.states);
    result["timesUs"] = json!(trajectory.times);
    result["populations"] = json!(trajectory.populations);
    result["scatteringRatePerUs"] = json!(trajectory.scattering);
    if options.return_density {
        let coords = coordinates(model.dimension);
        let rows: Vec<_> = trajectory
            .states
            .iter()
            .map(|x| expand_density(&coords, x, model.dimension))
            .collect();
        result["densityReal"] = json!(rows
            .iter()
            .map(|r| r.iter().map(|z| z.re).collect::<Vec<_>>())
            .collect::<Vec<_>>());
        result["densityImag"] = json!(rows
            .iter()
            .map(|r| r.iter().map(|z| z.im).collect::<Vec<_>>())
            .collect::<Vec<_>>());
    }
    let diagnostics = result["diagnostics"]
        .as_object_mut()
        .ok_or("Missing radiative diagnostics")?;
    let stationary_ms = diagnostics.get("elapsedMs").cloned().unwrap_or(json!(0.));
    let mean_trace = diagnostics
        .get("traceError")
        .and_then(Value::as_f64)
        .unwrap_or(0.);
    let mean_psd = diagnostics
        .get("positivityError")
        .and_then(Value::as_f64)
        .unwrap_or(0.);
    for key in [
        "frameHarmonicOrder",
        "frameConvergenceError",
        "frameUnitarityBound",
        "frameSchrodingerResidual",
        "harmonicOrder",
        "harmonicConvergenceError",
        "boundaryForcingMax",
        "residual",
        "traceError",
        "positivityError",
        "hermiticityError",
        "independentBeats",
        "frequencyBasis",
        "frequencyReconstructionErrors",
    ] {
        if let Some(value) = diagnostics.get(key).cloned() {
            let name = format!("asymptotic{}{}", key[..1].to_ascii_uppercase(), &key[1..]);
            diagnostics.insert(name, value);
        }
    }
    for (key, value) in trajectory.diagnostics.as_object().unwrap() {
        diagnostics.insert(key.clone(), value.clone());
    }
    diagnostics.insert("stationarySolveMs".into(), stationary_ms);
    diagnostics.insert("stationaryOnly".into(), json!(false));
    diagnostics.insert(
        "elapsedMs".into(),
        json!(started.elapsed().as_secs_f64() * 1000.),
    );
    diagnostics.insert(
        "traceError".into(),
        json!(trajectory.trace_error.max(mean_trace)),
    );
    diagnostics.insert(
        "positivityError".into(),
        json!(trajectory.positivity_error.max(mean_psd)),
    );
    diagnostics.insert("initialStateError".into(), json!(trajectory.initial_error));
    Ok(result)
}
