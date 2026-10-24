//! Full-density multi-frequency integration adapter.
//!
//! Atomic dispatch uses this path with the common radiative bath and explicit
//! frame/full-density error gates. It never supplies an infinite-time result
//! from a finite-time Fourier residual estimate.
use super::*;
use crate::{multitone_action as action, multitone_frame as frame};
use std::cell::RefCell;

struct Entry {
    key: Model,
    radiation: Option<crate::radiative_bath::Radiation>,
    order: i32,
    maximum_time: f64,
    frame: frame::Frame,
    prepared: action::Prepared,
    profile: Value,
}
thread_local! {static CACHE:RefCell<VecDeque<Entry>>=const{RefCell::new(VecDeque::new())};}
fn coordinates_from_density(coords: &[Coordinate], rho: &[C], n: usize) -> Vec<f64> {
    coords
        .iter()
        .map(|&c| match c {
            Coordinate::Population(i) => rho[i * n + i].re,
            Coordinate::Real(i, j) => rho[i * n + j].re,
            Coordinate::Imag(i, j) => rho[i * n + j].im,
        })
        .collect()
}
fn save(entry: Entry) {
    CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        cache.push_back(entry);
        while cache.len() > 2 {
            cache.pop_front();
        }
    });
}

pub(super) fn trajectory(
    lap: &Lapack,
    model: &Model,
    options: &Options,
) -> Result<crate::fast_periodic::Report, String> {
    trajectory_with_radiation(lap, model, None, options)
}
pub(super) fn trajectory_with_radiation(
    lap: &Lapack,
    model: &Model,
    radiation: Option<&crate::radiative_bath::Radiation>,
    options: &Options,
) -> Result<crate::fast_periodic::Report, String> {
    if model.dimension != 32 || model.drives.is_empty() || model.drives.len() > 2 {
        return Err("This full-density multitone preparation supports one or two explicit independent beats".into());
    }
    let started = Instant::now();
    let original = build_generator(model)?;
    let n = model.dimension;
    let initial = expand_density(&original.coords, &original.initial, n);
    let gauges = frame::resonant_gauges(model)?;
    let times: Vec<_> = (0..options.samples)
        .map(|i| options.duration_us * i as f64 / (options.samples - 1) as f64)
        .collect();
    let mut verification = times.clone();
    for fraction in [0., 0.25, 0.5, 0.75] {
        for drive in &model.drives {
            for phase in [0.137, 0.359, 0.75] {
                let t = options.duration_us * fraction
                    + phase * 2. * std::f64::consts::PI / drive.omega;
                if t <= options.duration_us {
                    verification.push(t);
                }
            }
        }
    }
    verification.sort_by(f64::total_cmp);
    verification.dedup();
    let requested: Vec<_> = times
        .iter()
        .map(|t| verification.binary_search_by(|v| v.total_cmp(t)).unwrap())
        .collect();
    let mut previous: Option<Vec<Vec<f64>>> = None;
    let mut all_cached = true;
    let mut build_ms = 0.;
    let mut apply_ms = 0.;
    let mut last_frame_error = String::new();
    let mut profiles = vec![];
    let mut interval_extensions = 0;
    for order in 2..=4 {
        let cached = CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            let index = cache.iter().position(|entry| {
                entry.order == order
                    && crate::fast_periodic::same_operator(&entry.key, model)
                    && crate::radiative_solve::same_radiation(entry.radiation.as_ref(), radiation)
            });
            index.map(|i| cache.remove(i).unwrap())
        });
        let entry = if let Some(entry) = cached {
            entry
        } else {
            all_cached = false;
            let before = Instant::now();
            let optical = match frame::prepare_with_radiation(lap, model, order, &gauges, radiation)
            {
                Ok(value) => value,
                Err(error) => {
                    if error.contains("unitarity") {
                        last_frame_error = error;
                        continue;
                    }
                    return Err(error);
                }
            };
            let frame_ms = before.elapsed().as_secs_f64() * 1000.;
            let stage = Instant::now();
            let parts = optical.density_model(lap, model, &initial, 1e-13)?;
            let collapse_fourier_ms = stage.elapsed().as_secs_f64() * 1000.;
            let stage = Instant::now();
            let base = hilbert_floquet::build_density_generator(lap, &parts.model)?;
            let base_generator_ms = stage.elapsed().as_secs_f64() * 1000.;
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
                frequencies: optical.frequencies.clone(),
                bounded_generator_tail: parts.bounded_generator_tail,
            };
            let prepared = action::prepare_lindblad_for_initial(
                lap,
                &input,
                &base.initial,
                options.duration_us,
                1e-8,
            )?;
            let d = &prepared.diagnostics;
            let profile = json!({"frameOrder":order,"frameMs":frame_ms,"collapseFourierMs":collapse_fourier_ms,"baseGeneratorMs":base_generator_ms,"normalFormPreparationMs":d.preparation_ms,"modalPreparationMs":d.modal_preparation_ms,"harmonicProjectionMs":d.harmonic_projection_ms,"normalFormMs":d.normal_form_ms,"physicalTransformMs":d.physical_transform_ms,"constantPreparationMs":d.constant_preparation_ms,
                "referenceBasis":if d.physical_reference{"physical Hermitian coordinates; complete L0 retained"}else{"complete modal basis"},"modalCondition":d.modal_condition,"modalDefect":d.modal_defect_norm,"modalRepresentationError":d.modal_representation_error,"iterations":d.iterations.iter().map(|i|json!({"coefficients":i.coefficients,"products":i.products,"elapsedMs":i.elapsed_ms,"normMs":i.norm_ms,"productMs":i.product_ms,"usedAlgebraicBound":i.used_algebraic_bound,"totalResidual":i.total_residual,"generatorTail":i.generator_tail,"roundingBound":i.rounding_bound,"productBound":i.product_bound,"evaluatedResidual":i.evaluated_residual})).collect::<Vec<_>>()});
            if std::env::var_os("ATOM_SOLVER_DEBUG").is_some() {
                eprintln!("multitone preparation {profile}");
            }
            build_ms += before.elapsed().as_secs_f64() * 1000.;
            Entry {
                key: model.clone(),
                radiation: radiation.cloned(),
                order,
                maximum_time: options.duration_us,
                frame: optical,
                prepared,
                profile,
            }
        };
        profiles.push(entry.profile.clone());
        if options.duration_us > entry.maximum_time {
            interval_extensions += 1;
        }
        let transformed = entry.frame.into_frame(&initial);
        let x0 = coordinates_from_density(&original.coords, &transformed, n);
        let evolved = match entry
            .prepared
            .apply_checked_interval(lap, &x0, &verification)
        {
            Ok(value) => value,
            Err(error) => {
                save(entry);
                return Err(error);
            }
        };
        apply_ms += evolved.elapsed_ms;
        let mut physical = Vec::with_capacity(verification.len());
        let mut scattering = Vec::with_capacity(verification.len());
        let mut trace_error = 0_f64;
        let mut positivity_error = 0_f64;
        let mut initial_error = 0_f64;
        let mut hermiticity_error = 0_f64;
        for (x, &time) in evolved.states.iter().zip(&verification) {
            let sigma = expand_density(&original.coords, x, n);
            let rho = entry.frame.physical_density(&sigma, time);
            if rho.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) {
                return Err("Non-finite reconstructed multitone density".into());
            }
            for i in 0..n {
                for j in 0..n {
                    hermiticity_error =
                        hermiticity_error.max((rho[i * n + j] - rho[j * n + i].conj()).abs());
                }
            }
            let values = coordinates_from_density(&original.coords, &rho, n);
            trace_error = trace_error.max((values[..n].iter().sum::<f64>() - 1.).abs());
            positivity_error = positivity_error.max(positivity(lap, &rho, n)?);
            if time == 0. {
                initial_error = max_abs(
                    &values
                        .iter()
                        .zip(&original.initial)
                        .map(|(a, b)| a - b)
                        .collect::<Vec<_>>(),
                );
            }
            scattering.push(entry.frame.scattering_rate(&sigma, time));
            physical.push(values);
        }
        if trace_error
            .max(positivity_error)
            .max(initial_error)
            .max(hermiticity_error)
            > 1e-8
        {
            return Err(format!("Multitone full-density check failed: trace={trace_error:e}, positivity={positivity_error:e}, initial={initial_error:e}, Hermiticity={hermiticity_error:e}"));
        }
        let difference = previous
            .as_ref()
            .map(|old| {
                physical
                    .iter()
                    .zip(old)
                    .flat_map(|(a, b)| a.iter().zip(b).map(|(a, b)| (a - b).abs()))
                    .fold(0., f64::max)
            })
            .unwrap_or(f64::INFINITY);
        let iterations:Vec<_>=entry.prepared.diagnostics.iterations.iter().map(|iteration|json!({"coefficients":iteration.coefficients,"evaluatedResidual":iteration.evaluated_residual,"productBound":iteration.product_bound,"roundingBound":iteration.rounding_bound,"generatorTail":iteration.generator_tail,"totalResidual":iteration.total_residual,"boundaryResidual":iteration.boundary_residual,"usedAlgebraicBound":iteration.used_algebraic_bound})).collect();
        let diagnostics = json!({"method":"Rust / full-density multitone Lindblad propagation","engine":"rust-floquet","frameHarmonicOrder":order,"frameConvergenceError":difference,
            "frameUnitarityBound":entry.frame.unitarity_bound,"frameSchrodingerResidual":entry.frame.schrodinger_residual,"frameSchrodingerResidualSum":entry.frame.schrodinger_residual_sum,
            "exactRotatingStateGauges":entry.frame.gauges,"finiteTimeNumericalTailBound":evolved.fourier_error_bound,"initialTransformError":evolved.initial_mismatch,"constantInitialReconstructionBound":evolved.constant_initial_reconstruction,
            "completeGeneratorTailTraceBound":evolved.cptp_generator_tail_bound,
            "radiationModel":if radiation.is_some(){ATOMIC_MODEL_SCHEMA}else{"generic-lindblad/1"},
            "modalCondition":entry.prepared.diagnostics.modal_condition,"baseModalDefect":entry.prepared.diagnostics.modal_defect_norm,"modalRepresentationError":entry.prepared.diagnostics.modal_representation_error,
            "referenceBasis":if entry.prepared.diagnostics.physical_reference{"physical Hermitian coordinates; complete L0 retained"}else{"complete modal basis"},
            "constantCorrectionNorm":entry.prepared.diagnostics.constant_correction_norm,"constantComponentDimensions":entry.prepared.diagnostics.component_dimensions,"normalFormIterations":iterations,
            "buildMs":build_ms,"propagateMs":apply_ms,"elapsedMs":started.elapsed().as_secs_f64()*1000.,"cached":all_cached,"verificationSamples":verification.len(),"hermiticityError":hermiticity_error,
            "originalPreparationProfiles":profiles,
            "intervalExtensions":interval_extensions,"certifiedDurationUs":options.duration_us,
            "numericalTailScope":"Physical density trace-norm bound for finite-time Fourier residual, boundary and bounded algebraic products, plus a separate CPTP bound on unbuilt generator products; frame/modal/static propagation floating-point checks are separately assessed."});
        save(entry);
        if difference < 1e-8 {
            return Ok(crate::fast_periodic::Report {
                times,
                populations: requested
                    .iter()
                    .map(|&i| physical[i][..n].to_vec())
                    .collect(),
                scattering: requested.iter().map(|&i| scattering[i]).collect(),
                states: if options.return_density {
                    requested.iter().map(|&i| physical[i].clone()).collect()
                } else {
                    vec![]
                },
                diagnostics,
                trace_error,
                positivity_error,
                initial_error,
            });
        }
        previous = Some(physical);
    }
    Err(format!("The complete multitone density did not converge under Fourier refinement. {last_frame_error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "End-to-end complete two-beat density adapter with independent frame-order convergence"]
    fn rb_three_beam_adapter_audit() {
        let mut raw: Value = serde_json::from_str(
            &std::fs::read_to_string(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../artifacts/v4/three-beam-master.json"
            ))
            .unwrap(),
        )
        .unwrap();
        raw["states"] = json!(raw["states"]
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
        let options = Options {
            duration_us: 20.,
            samples: 161,
            return_density: true,
            ..Options::default()
        };
        let result = trajectory(&lap, &model, &options).unwrap();
        let warm = trajectory(
            &lap,
            &model,
            &Options {
                return_density: false,
                ..options
            },
        )
        .unwrap();
        assert_eq!(result.populations, warm.populations);
        println!("cold={} warm={}", result.diagnostics, warm.diagnostics);
        std::fs::write(concat!(env!("CARGO_MANIFEST_DIR"),"/../artifacts/v4/three-beam-adapter-audit.json"),json!({"cold":result.diagnostics,"warm":warm.diagnostics,"timesUs":result.times,"coordinates":result.states,"trace":result.trace_error,"positivity":result.positivity_error}).to_string()).unwrap();
    }
}
