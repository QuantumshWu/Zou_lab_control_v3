//! Periodic fixed point of the COMPLETE small-system density propagator.
//! A finite-time plateau is never used as an infinite-time result. The full
//! one-period map acts on every Hermitian operator, with a trace-constrained
//! inverse establishing uniqueness on the stated invariant Hilbert support.
use super::*;

pub(super) struct Mean {
    pub density: Vec<C>,
    pub scattering: f64,
    pub diagnostics: Value,
    pub trace_error: f64,
    pub positivity_error: f64,
    pub hermiticity_error: f64,
}

pub(super) fn solve(
    lap: &Lapack,
    frame: &multitone_frame::Frame,
    model: &Model,
    initial: &[C],
    laboratory: bool,
) -> Result<Mean, String> {
    let started = Instant::now();
    let n = model.dimension;
    let d = n * n;
    if n == 0 || n > 4 || frame.frequencies.len() != 1 {
        return Err("Periodic-map mean requires a small exact one-beat invariant support".into());
    }
    let period = 2. * std::f64::consts::PI / frame.frequencies[0];
    // Every source C_k remains in Frame. Only generator products covered by
    // this explicit uniform numerical bound may avoid evaluation; the map
    // inverse below converts it into an INFINITE-time stationary error bound.
    let parts = crate::small_periodic::density_parts(
        lap,
        frame,
        model,
        initial,
        1e-11 / (1. + period),
        laboratory,
    )?;
    let base = build_generator(&parts.model)?;
    let harmonics: Vec<_> = parts
        .harmonics
        .into_iter()
        .map(|h| periodic_action::Harmonic {
            index: h.index[0],
            cos: h.cos,
            sin: h.sin,
        })
        .collect();
    let mut map = vec![0.; d * d];
    let mut map_refinement: f64 = 0.;
    let mut evaluations = 0;
    for col in 0..d {
        let mut basis = vec![0.; d];
        basis[col] = 1.;
        let integrated = adaptive_periodic::integrate_verified(
            &base.matrix,
            &harmonics,
            frame.frequencies[0],
            &basis,
            &[0., period],
            1e-10,
            2e-12,
            1e-9,
        )?;
        map_refinement = map_refinement.max(integrated.comparison_error);
        evaluations += integrated.total_rhs_evaluations;
        map[col * d..(col + 1) * d].copy_from_slice(integrated.action.states.last().unwrap());
    }
    let map_trace = (0..d)
        .map(|col| {
            ((0..n).map(|row| map[row + col * d]).sum::<f64>() - if col < n { 1. } else { 0. })
                .abs()
        })
        .fold(0., f64::max);
    if !map_trace.is_finite() || map_trace > 1e-9 {
        return Err(format!(
            "Complete one-period density map fails trace preservation: {map_trace:e}"
        ));
    }
    let columns: Vec<_> = map
        .chunks(d)
        .map(|x| expand_density(&base.coords, x, n))
        .collect();
    let mut choi = vec![C::default(); d * d];
    for i in 0..n {
        for a in 0..n {
            for b in 0..n {
                choi[(i * n + a) * d + i * n + b] = columns[i][a * n + b];
            }
        }
    }
    let mut col = n;
    for i in 0..n {
        for j in i + 1..n {
            for a in 0..n {
                for b in 0..n {
                    let r = columns[col][a * n + b];
                    let v = columns[col + 1][a * n + b];
                    let iv = C::new(-v.im, v.re);
                    choi[(i * n + a) * d + j * n + b] = (r - iv) * 0.5;
                    choi[(j * n + a) * d + i * n + b] = (r + iv) * 0.5;
                }
            }
            col += 2;
        }
    }
    let choi_error = positivity(lap, &choi, d)?;
    if choi_error > 1e-8 {
        return Err(format!(
            "Complete one-period map fails its Choi positivity check: {choi_error:e}"
        ));
    }
    let mut equation: Vec<_> = map.iter().map(|v| -v).collect();
    for i in 0..d {
        equation[i + i * d] += 1.;
    }
    for col in 0..d {
        equation[col * d] = if col < n { 1. } else { 0. };
    }
    let (lu, pivots) = lap.factor(&equation, d)?;
    let condition = lap.condition(&lu, d, norm_one(&equation, d))?;
    if !condition.is_finite() || condition > 1e10 {
        return Err(format!("Periodic fixed point is nonunique or numerically unresolved: trace-constrained condition {condition:e}"));
    }
    let mut rhs = vec![0.; d];
    rhs[0] = 1.;
    let fixed = lap.solve_factored(&lu, &pivots, &rhs, d)?;
    let mut inverse = vec![0.; d * d];
    for col in 0..d {
        let mut b = vec![0.; d];
        b[col] = 1.;
        inverse[col * d..(col + 1) * d].copy_from_slice(&lap.solve_factored(&lu, &pivots, &b, d)?);
    }
    let inverse_norm = norm_one(&inverse, d);
    let residual = (0..d)
        .map(|row| {
            ((0..d)
                .map(|col| map[row + col * d] * fixed[col])
                .sum::<f64>()
                - fixed[row])
                .abs()
        })
        .fold(0., f64::max);
    let sigma0 = expand_density(&base.coords, &fixed, n);
    let fixed_psd = positivity(lap, &sigma0, n)?;
    if residual > 1e-9 || fixed_psd > 1e-9 || (fixed[..n].iter().sum::<f64>() - 1.).abs() > 1e-9 {
        return Err("Periodic fixed-point density failed residual/trace/positivity checks".into());
    }
    // ||x||1<=n||rho||tr and ||rho||tr<=sqrt(2n)||x||1. True complete
    // C(t) evolution is CPTP. Duhamel/Gronwall bounds the omitted generator's
    // one-period map error even if the numerical generator is not itself CP.
    let to_trace = (2. * n as f64).sqrt();
    let delta = n as f64 * to_trace * parts.bounded_generator_tail;
    let map_tail = (delta * period).exp_m1();
    let neumann = inverse_norm * n as f64 * to_trace * map_tail;
    // E_num (x_num-x_true)=(E_true-E_num)x_true and the TRUE CPTP fixed
    // state's trace norm is one. This uses the actual numerical inverse,
    // so no 1/(1-neumann) factor belongs in this bound. The Neumann test
    // separately establishes uniqueness for the untruncated true map.
    let stationary_tail = to_trace * inverse_norm * n as f64 * map_tail;
    let waveform_tail =
        (stationary_tail + map_tail) * (delta * period).exp() * (1. + frame.unitarity_bound);
    // Numerical ODE refinement is observed, separate from the analytic bound
    // on omitted Fourier products; it is not mislabeled an absolute theorem.
    let observed_sensitivity = to_trace * inverse_norm * d as f64 * map_refinement;
    if !neumann.is_finite()
        || neumann >= 1.
        || !waveform_tail.is_finite()
        || waveform_tail > 1e-9
        || !observed_sensitivity.is_finite()
        || observed_sensitivity > 1e-8
    {
        return Err(format!("Periodic-map stationary error/sensitivity gate failed: Neumann {neumann:e}, tail {waveform_tail:e}, numerical refinement {observed_sensitivity:e}"));
    }
    let mut previous: Option<(Vec<C>, f64)> = None;
    let mut last_error = String::new();
    // Coprime quadrature meshes help avoid a shared alias at the input beat.
    for intervals in [127_usize, 257, 521, 1049] {
        let times: Vec<_> = (0..=intervals)
            .map(|i| period * i as f64 / intervals as f64)
            .collect();
        let integrated = adaptive_periodic::integrate_verified(
            &base.matrix,
            &harmonics,
            frame.frequencies[0],
            &fixed,
            &times,
            1e-10,
            2e-12,
            1e-9,
        )?;
        evaluations += integrated.total_rhs_evaluations;
        let mut mean = vec![C::default(); d];
        let mut scattering = 0.;
        let mut trace: f64 = 0.;
        let mut psd: f64 = 0.;
        let mut hermiticity: f64 = 0.;
        let mut closure: f64 = 0.;
        for (i, (&time, x)) in times.iter().zip(&integrated.action.states).enumerate() {
            let sigma = expand_density(&base.coords, x, n);
            let rho = frame.physical_density(&sigma, time);
            let weight = if i == 0 || i == intervals { 0.5 } else { 1. };
            for (a, &b) in mean.iter_mut().zip(&rho) {
                *a += b * (weight / intervals as f64);
            }
            scattering += weight * frame.scattering_rate(&sigma, time) / intervals as f64;
            trace = trace.max(((0..n).map(|j| rho[j * n + j].re).sum::<f64>() - 1.).abs());
            psd = psd.max(positivity(lap, &rho, n)?);
            if i == intervals {
                closure = x
                    .iter()
                    .zip(&fixed)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0., f64::max);
            }
            for a in 0..n {
                for b in 0..n {
                    hermiticity = hermiticity.max((rho[a * n + b] - rho[b * n + a].conj()).abs());
                }
            }
        }
        if trace > 1e-8
            || psd > 1e-8
            || hermiticity > 1e-8
            || closure > 1e-8
            || !scattering.is_finite()
            || scattering < -1e-10
        {
            return Err(format!("Periodic density waveform failed physical checks: trace {trace:e}, PSD {psd:e}, closure {closure:e}"));
        }
        let difference = previous
            .as_ref()
            .map(|(old, _)| {
                mean.iter()
                    .zip(old)
                    .map(|(&a, &b)| (a - b).abs())
                    .fold(0., f64::max)
            })
            .unwrap_or(f64::INFINITY);
        let scattering_difference = previous
            .as_ref()
            .map(|(_, old)| (scattering - old).abs() / (1. + scattering.abs().max(old.abs())))
            .unwrap_or(f64::INFINITY);
        if difference < 1e-9 && scattering_difference < 1e-9 {
            let mean_real: Vec<_> = mean.iter().map(|z| z.re).collect();
            let mean_imag: Vec<_> = mean.iter().map(|z| z.im).collect();
            return Ok(Mean {
                density: mean,
                scattering,
                trace_error: trace,
                positivity_error: psd,
                hermiticity_error: hermiticity,
                diagnostics: json!({
            "scope":"Complete Hermitian one-period map and trace-constrained fixed point; uniqueness only on the stated invariant Hilbert support. Analytic bounds describe omitted Fourier products, while ODE and frame errors are independently refined numerical checks.",
            "periodUs":period,"operatorDimension":d,"conditionEstimate":condition,"traceConstrainedInverseOneNorm":inverse_norm,"residual":residual,"mapTraceError":map_trace,"mapChoiPositivityError":choi_error,"mapIntegrationRefinement":map_refinement,"observedStationarySensitivity":observed_sensitivity,
            "physicalMeanReal":mean_real,"physicalMeanImag":mean_imag,"fixedPointCoordinates":fixed,
            "boundedGeneratorTail":parts.bounded_generator_tail,"boundedGeneratorIndices":parts.bounded_indices,"mapTailTraceBound":map_tail,"stationaryTailTraceBound":stationary_tail,"waveformTailTraceBound":waveform_tail,"inversePerturbationNeumannBound":neumann,
            "quadratureIntervals":intervals,"meanQuadratureConvergence":difference,"scatteringQuadratureConvergence":scattering_difference,"waveformClosure":closure,"waveformTraceError":trace,"waveformPositivityError":psd,"rhsEvaluations":evaluations,"elapsedMs":started.elapsed().as_secs_f64()*1000.}),
            });
        }
        last_error=format!("Periodic mean quadrature has not converged: density {difference:e}, scattering {scattering_difference:e}");
        previous = Some((mean, scattering));
    }
    Err(last_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture(thermal: bool) -> (Model, multitone_frame::Frame, Vec<C>) {
        let zero = Operator {
            real: vec![0.; 4],
            imag: vec![0.; 4],
        };
        let model = Model { frequency_coordinates:None,
            dimension: 2,
            states: vec![],
            h_real: vec![0.; 4],
            h_imag: vec![0.; 4],
            collapse: vec![],
            initial_index: 0,
            initial_real: None,
            initial_imag: None,
            time_dependent: true,
            drives: vec![Drive {
                omega: 2.,
                cos: zero.clone(),
                sin: zero,
            }],
        };
        let channels = if thermal {
            vec![
                vec![
                    C::default(),
                    C::new(2_f64.sqrt(), 0.),
                    C::default(),
                    C::default(),
                ],
                vec![C::default(), C::default(), C::new(1., 0.), C::default()],
            ]
        } else {
            vec![vec![
                C::new(1., 0.),
                C::default(),
                C::default(),
                C::new(-1., 0.),
            ]]
        };
        let frame = multitone_frame::Frame {
            dimension: 2,
            frequencies: vec![2.],
            gauges: vec![vec![0]; 2],
            order: 1,
            modes: std::collections::BTreeMap::from([(
                vec![0],
                vec![C::new(1., 0.), C::default(), C::default(), C::new(1., 0.)],
            )]),
            hamiltonian: vec![C::default(); 4],
            collapse: channels
                .into_iter()
                .map(|c| std::collections::BTreeMap::from([(vec![0], c)]))
                .collect(),
            unitarity_bound: 0.,
            schrodinger_residual: 0.,
            schrodinger_residual_sum: 0.,
            minimum_central_weight: 1.,
            sambe_dimension: 0,
            preparation_ms: 0.,
            radiation_diagnostics: None,
        };
        (
            model,
            frame,
            vec![C::new(1., 0.), C::default(), C::default(), C::default()],
        )
    }
    #[test]
    fn full_density_period_map_has_analytic_thermal_fixed_point() {
        let lap = Lapack::load().unwrap();
        let (m, f, x) = fixture(true);
        let mean = solve(&lap, &f, &m, &x, false).unwrap();
        assert!((mean.density[0].re - 2. / 3.).abs() < 1e-11);
        assert!((mean.density[3].re - 1. / 3.).abs() < 1e-11);
        assert_eq!(mean.diagnostics["operatorDimension"], 4);
        assert!(mean.diagnostics["mapChoiPositivityError"].as_f64().unwrap() < 1e-12);
    }
    #[test]
    fn nonunique_population_constants_are_not_given_an_arbitrary_mean() {
        let lap = Lapack::load().unwrap();
        let (m, f, x) = fixture(false);
        assert!(solve(&lap, &f, &m, &x, false).is_err());
    }
}
