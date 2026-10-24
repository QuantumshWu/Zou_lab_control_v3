//! Complete periodic Lindblad integration on an EXACT invariant Hilbert support.
//! No weak-field, detuning, population-only or secular approximation is used.
//! This complements the high-frequency normal-form method at low beat frequency.
use super::*;
use crate::radiative_bath::Radiation;

pub(super) struct Projection {
    pub model: Model,
    pub radiation: Radiation,
    pub support: Vec<usize>,
}

/// Structural closure under H(t), each raw dipole and all possible no-jump
/// shared-output pairs. The latter is deliberately structural: spectral bath
/// weighting need not preserve an accidental cancellation in raw D†D.
pub(super) fn project(model: &Model, radiation: &Radiation) -> Result<Option<Projection>, String> {
    let n = model.dimension;
    if model.drives.len() != 1 {
        return Ok(None);
    }
    let raw = radiation.normalized_dipoles(n)?;
    let mut structural = model.clone();
    structural.collapse = radiation.dipoles.clone();
    let mut support = hilbert_support(&structural)?;
    loop {
        let mut seen = vec![false; n];
        for &i in &support {
            seen[i] = true;
        }
        for c in &raw {
            for row in 0..n {
                let sources: Vec<_> = (0..n)
                    .filter(|&col| {
                        let z = c[row * n + col];
                        z.re != 0. || z.im != 0.
                    })
                    .collect();
                if sources.iter().any(|&i| seen[i]) {
                    for col in sources {
                        seen[col] = true;
                    }
                }
            }
        }
        let extra: Vec<_> = (0..n).filter(|&i| seen[i]).collect();
        if extra == support {
            break;
        }
        // Re-close newly added sources under H and directed dipoles.
        let mut rho = vec![0.; n * n];
        for &i in &extra {
            rho[i * n + i] = 1. / extra.len() as f64;
        }
        structural.initial_real = Some(rho);
        structural.initial_imag = Some(vec![0.; n * n]);
        support = hilbert_support(&structural)?;
    }
    if support.is_empty() || support.len() > 4 || support.len() == n {
        return Ok(None);
    }
    let m = support.len();
    let matrix = |a: &[f64]| {
        support
            .iter()
            .flat_map(|&i| support.iter().map(move |&j| a[i * n + j]))
            .collect::<Vec<_>>()
    };
    let operator = |a: &Operator| Operator {
        real: matrix(&a.real),
        imag: matrix(&a.imag),
    };
    let mut reduced = model.clone();
    reduced.dimension = m;
    reduced.states = support
        .iter()
        .map(|&i| {
            model
                .states
                .get(i)
                .cloned()
                .unwrap_or_else(|| i.to_string())
        })
        .collect();
    reduced.h_real = matrix(&model.h_real);
    reduced.h_imag = matrix(&model.h_imag);
    // A common energy origin does not affect either the commutator or emitted
    // photon energy differences, and avoids needless GHz cancellation here.
    let origin = reduced.h_real[0];
    for i in 0..m {
        reduced.h_real[i * m + i] -= origin;
    }
    reduced.collapse = model.collapse.iter().map(operator).collect();
    reduced.drives = model
        .drives
        .iter()
        .map(|d| Drive {
            omega: d.omega,
            cos: operator(&d.cos),
            sin: operator(&d.sin),
        })
        .collect();
    reduced.initial_index = support
        .iter()
        .position(|&i| i == model.initial_index)
        .unwrap_or(0);
    reduced.initial_real = model.initial_real.as_ref().map(|a| matrix(a));
    reduced.initial_imag = model.initial_imag.as_ref().map(|a| matrix(a));
    let radiation = Radiation {
        reference_angular_frequency: radiation.reference_angular_frequency,
        carrier_angular_frequency: radiation.carrier_angular_frequency,
        dipoles: radiation.dipoles.iter().map(operator).collect(),
    };
    Ok(Some(Projection {
        model: reduced,
        radiation,
        support,
    }))
}

pub(super) fn lift_populations(row: &[f64], support: &[usize], n: usize) -> Vec<f64> {
    let mut full = vec![0.; n];
    for (&i, &p) in support.iter().zip(row) {
        full[i] = p;
    }
    full
}
fn coords(rho: &[C], n: usize) -> Vec<f64> {
    coordinates(n)
        .iter()
        .map(|c| match *c {
            Coordinate::Population(i) => rho[i * n + i].re,
            Coordinate::Real(i, j) => rho[i * n + j].re,
            Coordinate::Imag(i, j) => rho[i * n + j].im,
        })
        .collect()
}

/// Return to the original physical density coordinates after constructing the
/// frequency-weighted bath. This is a complete Fourier unitary transform,
/// not an averaged dissipator. The original H(t) is added explicitly below.
fn laboratory_frame(
    lap: &Lapack,
    mut frame: multitone_frame::Frame,
    model: &Model,
    radiation: &Radiation,
) -> Result<multitone_frame::Frame, String> {
    let collapse = crate::unitary_monodromy::physical_collapse_stable(lap, &frame, radiation)?;
    let n = model.dimension;
    let mut identity = vec![C::default(); n * n];
    for i in 0..n {
        identity[i * n + i] = C::new(1., 0.);
    }
    frame.collapse = collapse;
    frame.modes = std::collections::BTreeMap::from([(vec![0], identity)]);
    frame.gauges = vec![vec![0]; n];
    frame.hamiltonian = complex_input(
        &model.h_real,
        &model.h_imag,
        n * n,
        "physical periodic Hamiltonian",
    )?;
    Ok(frame)
}

pub(super) fn density_parts(
    lap: &Lapack,
    frame: &multitone_frame::Frame,
    model: &Model,
    initial: &[C],
    tail: f64,
    laboratory: bool,
) -> Result<multitone_frame::DensityFrame, String> {
    let mut parts = frame.density_model(lap, model, initial, tail)?;
    if laboratory {
        for drive in &model.drives {
            let generator = |op: &Operator| -> Result<_, String> {
                let mut m = model.clone();
                m.h_real = op.real.clone();
                m.h_imag = op.imag.clone();
                m.collapse.clear();
                m.drives.clear();
                m.time_dependent = false;
                Ok(build_generator(&m)?.matrix)
            };
            let cos = generator(&drive.cos)?;
            let sin = generator(&drive.sin)?;
            if let Some(h) = parts.harmonics.iter_mut().find(|h| h.index == vec![1]) {
                for (a, b) in h.cos.iter_mut().zip(cos) {
                    a.extend(b);
                }
                for (a, b) in h.sin.iter_mut().zip(sin) {
                    a.extend(b);
                }
            } else {
                parts.harmonics.push(multitone_frame::Harmonic {
                    index: vec![1],
                    cos,
                    sin,
                });
            }
        }
    }
    Ok(parts)
}

pub(super) fn trajectory(
    lap: &Lapack,
    original: &Model,
    projection: &Projection,
    options: &Options,
) -> Result<fast_periodic::Report, String> {
    let started = Instant::now();
    let model = &projection.model;
    let radiation = &projection.radiation;
    let n = model.dimension;
    let base0 = build_generator(model)?;
    let initial = expand_density(&base0.coords, &base0.initial, n);
    let times: Vec<_> = (0..options.samples)
        .map(|i| options.duration_us * i as f64 / (options.samples - 1) as f64)
        .collect();
    let mut observed_times = times.clone();
    let period = 2. * std::f64::consts::PI / model.drives[0].omega;
    for fraction in [0., 0.125, 0.375, 0.625, 0.875] {
        for phase in [0.137, 0.25, 0.5, 0.75] {
            let time = options.duration_us * fraction + phase * period;
            if time <= options.duration_us {
                observed_times.push(time);
            }
        }
    }
    observed_times.sort_by(f64::total_cmp);
    observed_times.dedup();
    let requested: Vec<_> = times
        .iter()
        .map(|t| observed_times.binary_search_by(|v| v.total_cmp(t)).unwrap())
        .collect();
    let gauges = multitone_frame::resonant_gauges(model)?;
    let mut previous: Option<Vec<Vec<f64>>> = None;
    let mut previous_scatter: Option<Vec<f64>> = None;
    let mut last_error = String::new();
    let resolutions: Vec<_> = (1..=12)
        .map(|order| (false, order))
        .chain(
            [32, 64, 128, 256, 512, 1024]
                .into_iter()
                .map(|grid| (true, grid)),
        )
        .collect();
    for (monodromy, resolution) in resolutions {
        let prepared = if monodromy {
            crate::unitary_monodromy::prepare(lap, model, radiation, resolution)
        } else {
            multitone_frame::prepare_with_radiation(
                lap,
                model,
                resolution as i32,
                &gauges,
                Some(radiation),
            )
        };
        let frame = match prepared {
            Ok(frame) => frame,
            Err(e) => {
                last_error = e;
                continue;
            }
        };
        let frame = if monodromy {
            laboratory_frame(lap, frame, model, radiation)?
        } else {
            frame
        };
        let order = frame.order;
        // Zero tail budget means every nonzero dissipator Fourier coefficient
        // is explicitly evaluated, including all signed cross convolutions.
        let parts = density_parts(
            lap,
            &frame,
            model,
            &initial,
            if monodromy {
                1e-12 / (1. + options.duration_us)
            } else {
                0.
            },
            monodromy,
        )?;
        let generator_tail = parts.bounded_generator_tail;
        let finite_tail =
            (n as f64 * (2. * n as f64).sqrt() * generator_tail * options.duration_us).exp_m1();
        if finite_tail > 1e-10 {
            return Err(format!(
                "Physical periodic generator-tail budget failed: {finite_tail:e}"
            ));
        }
        let base = match build_generator(&parts.model) {
            Ok(base) => base,
            Err(e) if e.contains("initial density") => {
                last_error = e;
                continue;
            }
            Err(e) => return Err(e),
        };
        let harmonics: Vec<_> = parts
            .harmonics
            .into_iter()
            .map(|h| periodic_action::Harmonic {
                index: h.index[0],
                cos: h.cos,
                sin: h.sin,
            })
            .collect();
        let phase_steps = options.duration_us
            * frame.frequencies[0].abs()
            * harmonics
                .iter()
                .map(|h| h.index.abs())
                .max()
                .unwrap_or(1)
                .max(1) as f64
            * 8.
            / (2. * std::f64::consts::PI);
        if phase_steps > 50_000. {
            return Err(format!("Direct periodic phase resolution requires at least {phase_steps:.0} steps; use the full Fourier/normal-form solver for this interval"));
        }
        let integrated = crate::adaptive_periodic::integrate_verified(
            &base.matrix,
            &harmonics,
            frame.frequencies[0],
            &base.initial,
            &observed_times,
            1e-10,
            2e-12,
            2e-9,
        )?;
        let mut rows = Vec::with_capacity(observed_times.len());
        let mut scattering = Vec::with_capacity(observed_times.len());
        let mut trace: f64 = 0.;
        let mut psd: f64 = 0.;
        let mut initial_error: f64 = 0.;
        for (i, (&time, x)) in observed_times
            .iter()
            .zip(&integrated.action.states)
            .enumerate()
        {
            let sigma = expand_density(&base.coords, x, n);
            let rho = frame.physical_density(&sigma, time);
            trace = trace.max(((0..n).map(|j| rho[j * n + j].re).sum::<f64>() - 1.).abs());
            psd = psd.max(positivity(lap, &rho, n)?);
            if i == 0 {
                initial_error = rho
                    .iter()
                    .zip(&initial)
                    .map(|(&a, &b)| (a - b).abs())
                    .fold(0., f64::max);
            }
            scattering.push(frame.scattering_rate(&sigma, time));
            rows.push(coords(&rho, n));
        }
        if trace > 1e-8
            || psd > 1e-8
            || initial_error > 1e-8
            || scattering.iter().any(|x| !x.is_finite() || *x < -1e-10)
        {
            last_error=format!("Direct full-density physical checks: trace {trace:e}, PSD {psd:e}, initial {initial_error:e}");
            continue;
        }
        let difference = previous
            .as_ref()
            .map(|old| {
                rows.iter()
                    .flatten()
                    .zip(old.iter().flatten())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0., f64::max)
            })
            .unwrap_or(f64::INFINITY);
        let scatter_difference = previous_scatter
            .as_ref()
            .map(|old| {
                scattering
                    .iter()
                    .zip(old)
                    .map(|(a, b)| (a - b).abs() / (1. + a.abs().max(b.abs())))
                    .fold(0., f64::max)
            })
            .unwrap_or(f64::INFINITY);
        if difference < 1e-8 && scatter_difference < 1e-8 {
            let populations = requested
                .iter()
                .map(|&i| lift_populations(&rows[i][..n], &projection.support, original.dimension))
                .collect();
            let states = requested
                .iter()
                .map(|&index| {
                    let rho = expand_density(&base.coords, &rows[index], n);
                    let mut full = vec![C::default(); original.dimension * original.dimension];
                    for (i, &a) in projection.support.iter().enumerate() {
                        for (j, &b) in projection.support.iter().enumerate() {
                            full[a * original.dimension + b] = rho[i * n + j];
                        }
                    }
                    coords(&full, original.dimension)
                })
                .collect();
            let scattering = requested.iter().map(|&i| scattering[i]).collect();
            return Ok(fast_periodic::Report {
                times,
                states,
                populations,
                scattering,
                trace_error: trace,
                positivity_error: psd,
                initial_error,
                diagnostics: json!({
                "method":"Rust / full-density adaptive periodic Lindblad","engine":"rust-dopri5","timeDependent":true,"densityDimension":original.dimension,"invariantHilbertDimension":n,"invariantSupport":projection.support,
                "elapsedMs":started.elapsed().as_secs_f64()*1000.,"frameConstruction":if monodromy{"complete-unitary-monodromy; physical density coordinates"}else{"Hilbert-Sambe"},"frameHarmonicOrder":order,"frameConvergenceError":difference,"frameScatteringConvergenceError":scatter_difference,"frameUnitarityBound":frame.unitarity_bound,"frameSchrodingerResidual":frame.schrodinger_residual,"boundedGeneratorTail":generator_tail,"finiteGeneratorTailTraceBound":finite_tail,
                "integrationComparisonError":integrated.comparison_error,"integrationComparisonStepLimit":integrated.comparison_step_limit,"integrationLocalErrorMax":integrated.action.max_error,"integrationAcceptedSteps":integrated.action.accepted,"integrationRejectedSteps":integrated.action.rejected,"integrationRhsEvaluations":integrated.total_rhs_evaluations,
                "verificationSamples":observed_times.len(),"validationScope":"Independent integration tolerances and adjacent Floquet-frame orders compared on every requested full density matrix plus dispersed beat phases; embedded step estimate is local, not an absolute global theorem.","radiationModel":ATOMIC_MODEL_SCHEMA,"radiationChannels":3}),
            });
        }
        last_error=format!("Direct periodic full-density frame convergence {difference:e}, scattering {scatter_difference:e}");
        previous = Some(rows);
        previous_scatter = Some(scattering);
    }
    Err(format!(
        "Small invariant-support periodic integration did not converge: {last_error}"
    ))
}

pub(super) fn periodic_mean(
    lap: &Lapack,
    model: &Model,
    radiation: &Radiation,
) -> Result<Value, String> {
    let started = Instant::now();
    let original = build_generator(model)?;
    let n = model.dimension;
    let initial = expand_density(&original.coords, &original.initial, n);
    let mut previous: Option<(Vec<C>, f64)> = None;
    let mut last_error = String::new();
    for grid in [32, 64, 128, 256, 512, 1024] {
        let frame = match crate::unitary_monodromy::prepare(lap, model, radiation, grid) {
            Ok(frame) => frame,
            Err(error) => {
                last_error = error;
                continue;
            }
        };
        let frame = laboratory_frame(lap, frame, model, radiation)?;
        let mean = match crate::small_periodic_mean::solve(lap, &frame, model, &initial, true) {
            Ok(mean) => mean,
            Err(error) => {
                last_error = error;
                continue;
            }
        };
        let difference = previous
            .as_ref()
            .map(|(old, _)| {
                mean.density
                    .iter()
                    .zip(old)
                    .map(|(&a, &b)| (a - b).abs())
                    .fold(0., f64::max)
            })
            .unwrap_or(f64::INFINITY);
        let scattering_difference = previous
            .as_ref()
            .map(|(_, old)| {
                (mean.scattering - old).abs() / (1. + mean.scattering.abs().max(old.abs()))
            })
            .unwrap_or(f64::INFINITY);
        if difference < 1e-8 && scattering_difference < 1e-8 {
            return Ok(
                json!({"states":model.states,"timesUs":[0.],"populations":[original.initial[..n].to_vec()],"scatteringRatePerUs":[frame.scattering_rate(&frame.into_frame(&initial),0.)],
            "asymptoticPopulations":(0..n).map(|i|mean.density[i*n+i].re).collect::<Vec<_>>(),"asymptoticScatteringRatePerUs":mean.scattering,"asymptoticStatus":"periodic-mean",
            "asymptoticDensityReal":mean.density.iter().map(|z|z.re).collect::<Vec<_>>(),"asymptoticDensityImag":mean.density.iter().map(|z|z.im).collect::<Vec<_>>(),
            "diagnostics":{"method":"Rust / complete periodic density-map fixed point","steadyMethod":"Complete one-period Hermitian density map with trace-constrained fixed point and emitted-frequency common bath","engine":"rust-dopri5","radiationModel":ATOMIC_MODEL_SCHEMA,"radiationChannels":3,
                "elapsedMs":started.elapsed().as_secs_f64()*1000.,"stationaryOnly":true,"stationaryCached":false,"initialDependent":false,"initialDependenceScope":"within the stated invariant Hilbert support","densityLimitExists":false,"asymptoticStatus":"periodic-mean",
                "asymptoticReason":"Unique trace-constrained fixed point of the complete one-period density propagator; the physical periodic waveform is integrated over one period, never inferred from a finite-time plateau.",
                "traceError":mean.trace_error,"positivityError":mean.positivity_error,"hermiticityError":mean.hermiticity_error,"frameConstruction":"complete-unitary-monodromy","frameHarmonicOrder":frame.order,"frameConvergenceError":difference,"frameScatteringConvergenceError":scattering_difference,"frameUnitarityBound":frame.unitarity_bound,"frameSchrodingerResidual":frame.schrodinger_residual,
                "residual":mean.diagnostics["residual"],"stationaryPhysicalHilbertDimension":n,"stationaryOperatorDimension":n*n,"stationarySystemConditionEstimate":mean.diagnostics["conditionEstimate"],"meanPeriodMapDiagnostics":mean.diagnostics,
                "uniquenessScope":"full Hermitian algebra on the stated invariant Hilbert support"}}),
            );
        }
        last_error=format!("Periodic-map frame convergence: density {difference:e}, scattering {scattering_difference:e}");
        previous = Some((mean.density, mean.scattering));
    }
    Err(format!(
        "No verified small-support periodic-map mean: {last_error}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> (Model, Radiation) {
        let n = 5;
        let mut h = vec![0.; n * n];
        h[1] = 0.5;
        h[n] = 0.5;
        let mut d = vec![0.; n * n];
        d[1] = 1.;
        let zero = Operator {
            real: vec![0.; n * n],
            imag: vec![0.; n * n],
        };
        let op = Operator {
            real: d,
            imag: vec![0.; n * n],
        };
        (
            Model { frequency_coordinates:None,
                dimension: n,
                states: (0..n).map(|i| i.to_string()).collect(),
                h_real: h.clone(),
                h_imag: vec![0.; n * n],
                collapse: vec![],
                initial_index: 0,
                initial_real: None,
                initial_imag: None,
                time_dependent: true,
                drives: vec![Drive {
                    omega: 7.,
                    cos: Operator {
                        real: h,
                        imag: vec![0.; n * n],
                    },
                    sin: zero.clone(),
                }],
            },
            Radiation {
                reference_angular_frequency: 1000.,
                carrier_angular_frequency: 1000.,
                dipoles: vec![op, zero.clone(), zero],
            },
        )
    }
    #[test]
    fn exact_support_keeps_every_nonzero_edge() {
        let (mut model, bath) = fixture();
        let projected = project(&model, &bath).unwrap().unwrap();
        assert_eq!(projected.support, vec![0, 1]);
        for i in 1..4 {
            model.h_real[i * 5 + i + 1] = 1e-30;
            model.h_real[(i + 1) * 5 + i] = 1e-30;
        }
        assert!(
            project(&model, &bath).unwrap().is_none(),
            "A tiny physical coupling is never a support cutoff"
        );
    }
    #[test]
    fn raw_shared_output_closure_survives_weighting() {
        let (model, mut bath) = fixture();
        bath.dipoles[0].real[2] = 1.;
        let projected = project(&model, &bath).unwrap().unwrap();
        assert_eq!(projected.support, vec![0, 1, 2]);
    }
    #[test]
    #[ignore]
    fn low_beat_physical_fourier_profile() {
        let app = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap();
        let source = std::fs::read_to_string(
            app.join("artifacts/v4/radiation-protocol/low-beat-0p1-request.json"),
        )
        .unwrap();
        let request: Request = serde_json::from_str(&source).unwrap();
        let projection = project(
            request.model.as_ref().unwrap(),
            request.radiation.as_ref().unwrap(),
        )
        .unwrap()
        .unwrap();
        let model = &projection.model;
        let lap = Lapack::load().unwrap();
        let base = build_generator(model).unwrap();
        let initial = expand_density(&base.coords, &base.initial, 2);
        for grid in [256, 512] {
            let frame = crate::unitary_monodromy::prepare(&lap, model, &projection.radiation, grid)
                .unwrap();
            let unitary = frame.unitarity_bound;
            let frame = laboratory_frame(&lap, frame, model, &projection.radiation).unwrap();
            let mut norms = std::collections::BTreeMap::<i32, f64>::new();
            for c in &frame.collapse {
                for (q, m) in c {
                    *norms.entry(q[0]).or_default() += m.iter().map(|z| z.abs()).sum::<f64>();
                }
            }
            let parts = density_parts(&lap, &frame, model, &initial, 1e-11 / 11., true).unwrap();
            let count: usize = parts
                .harmonics
                .iter()
                .map(|h| h.cos.iter().chain(&h.sin).map(Vec::len).sum::<usize>())
                .sum();
            let maxindex = parts
                .harmonics
                .iter()
                .map(|h| h.index[0].abs())
                .max()
                .unwrap_or(0);
            println!(
                "{}",
                json!({"grid":grid,"unitarity":unitary,"jumpNorms":norms,"retainedHarmonics":parts.harmonics.len(),"maximumIndex":maxindex,"nonzero":count,"tail":parts.bounded_generator_tail})
            );
        }
    }
}
