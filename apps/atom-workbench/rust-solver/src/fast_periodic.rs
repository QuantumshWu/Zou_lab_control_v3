//! Fast, error-gated finite-time path for an isolated optical beat.
//! The original 32-state density and every transformed collapse harmonic are
//! retained. Long-time values use a separate complete Fourier density solve.
use super::*;
use crate::{hilbert_floquet, periodic_action};
use std::cell::RefCell;

struct Entry {
    key: Model,
    frame_model: Model,
    radiation: Option<crate::radiative_bath::Radiation>,
    order: i32,
    maximum_time: f64,
    frame: hilbert_floquet::Frame,
    gauges: Vec<i32>,
    prepared: periodic_action::Prepared,
    observables: periodic_action::ObservablePrepared,
}
thread_local! {static CACHE:RefCell<VecDeque<Entry>>=const{RefCell::new(VecDeque::new())};}

pub(super) fn same_operator(a: &Model, b: &Model) -> bool {
    let matrix = |a: &Operator, b: &Operator| a.real == b.real && a.imag == b.imag;
    a.dimension == b.dimension
        && a.h_real == b.h_real
        && a.h_imag == b.h_imag
        && a.collapse.len() == b.collapse.len()
        && a.collapse
            .iter()
            .zip(&b.collapse)
            .all(|(a, b)| matrix(a, b))
        && a.drives.len() == b.drives.len()
        && a.drives
            .iter()
            .zip(&b.drives)
            .all(|(a, b)| a.omega == b.omega && matrix(&a.cos, &b.cos) && matrix(&a.sin, &b.sin))
}
fn frame_coordinates(coords: &[Coordinate], rho: &[C], n: usize) -> Vec<f64> {
    coords
        .iter()
        .map(|&coord| match coord {
            Coordinate::Population(i) => rho[i * n + i].re,
            Coordinate::Real(i, j) => rho[i * n + j].re,
            Coordinate::Imag(i, j) => rho[i * n + j].im,
        })
        .collect()
}
// This changes only observable divided-difference caching. The propagator's
// numerical tail is still checked at every actual requested time and initial
// density. Small time-window edits need not repeat identical projection GEMMs.
fn observable_horizon(time: f64) -> f64 {
    (2. * time).min(1e6)
}

pub(super) struct Report {
    pub times: Vec<f64>,
    pub states: Vec<Vec<f64>>,
    pub populations: Vec<Vec<f64>>,
    pub scattering: Vec<f64>,
    pub diagnostics: Value,
    pub trace_error: f64,
    pub positivity_error: f64,
    pub initial_error: f64,
}
pub(super) fn trajectory(lap: &Lapack, model: &Model, options: &Options) -> Result<Report, String> {
    trajectory_with_radiation(lap, model, None, options)
}
pub(super) fn trajectory_with_radiation(
    lap: &Lapack,
    model: &Model,
    radiation: Option<&crate::radiative_bath::Radiation>,
    options: &Options,
) -> Result<Report, String> {
    if model.dimension != 32 || model.drives.len() != 1 {
        return Err("The isolated-beat fast path is not applicable to this model".into());
    }
    let started = Instant::now();
    let original = build_generator(model)?;
    let n = model.dimension;
    let initial = expand_density(&original.coords, &original.initial, n);
    let times: Vec<_> = (0..options.samples)
        .map(|i| options.duration_us * i as f64 / (options.samples - 1) as f64)
        .collect();
    // Extra physical times sample distinct optical phases even if the requested
    // plot grid happens to be commensurate with the beat. They are verification
    // points, not invented interpolated chart samples.
    let mut verification = if options.return_density {
        times.clone()
    } else {
        vec![0., options.duration_us]
    };
    let period = 2. * std::f64::consts::PI / model.drives[0].omega;
    for fraction in [0., 0.125, 0.375, 0.625, 0.875] {
        let anchor = options.duration_us * fraction;
        for phase in [0.137, 0.25, 0.5, 0.75] {
            let t = anchor + phase * period;
            if t <= options.duration_us {
                verification.push(t);
            }
        }
    }
    verification.sort_by(f64::total_cmp);
    verification.dedup();
    let mut observed_times = times.clone();
    observed_times.extend(&verification);
    observed_times.sort_by(f64::total_cmp);
    observed_times.dedup();
    let requested: Vec<_> = times
        .iter()
        .map(|t| observed_times.binary_search_by(|v| v.total_cmp(t)).unwrap())
        .collect();
    let mut previous: Option<Vec<Vec<f64>>> = None;
    let mut previous_observed: Option<Vec<Vec<f64>>> = None;
    let mut all_cached = true;
    let mut build_ms = 0.;
    let mut apply_ms = 0.;
    let mut interval_extensions = 0;
    let scattering_scale = original
        .scattering
        .chunks(n)
        .map(|row| row.iter().map(|z| z.abs()).sum::<f64>())
        .fold(1., f64::max);
    for order in 1..=3 {
        // The optical frame and complete modal basis do not depend on the
        // plotting interval. A longer request must pass fresh actual-time
        // bounds below; it need not repeat their eigensystem construction.
        let cached = CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            let index = cache.iter().position(|entry| {
                entry.order == order
                    && same_operator(&entry.key, model)
                    && crate::radiative_solve::same_radiation(entry.radiation.as_ref(), radiation)
            });
            index.map(|i| cache.remove(i).unwrap())
        });
        let reused_preparation = cached.is_some();
        let mut entry = if let Some(entry) = cached {
            entry
        } else {
            all_cached = false;
            let build = Instant::now();
            let (frame, frame_model, gauges) = match hilbert_floquet::prepare_with_radiation(
                lap, model, order, radiation,
            ) {
                Ok(frame) => (frame, model.clone(), vec![0; n]),
                Err(original_reason) => {
                    if radiation.is_some() {
                        return Err(format!("{original_reason}; common-vacuum radiation requires the complete state-gauged Floquet frame"));
                    }
                    let (transformed,gauges)=crate::multitone_frame::single_beat_static_decay_gauge(model).map_err(|reason|format!("{original_reason}; exact rotating-coordinate fallback: {reason}"))?;
                    (
                        hilbert_floquet::prepare(lap, &transformed, order)?,
                        transformed,
                        gauges,
                    )
                }
            };
            let parts = frame.density_model(lap, &frame_model, &initial)?;
            let base = hilbert_floquet::build_density_generator(lap, &parts.model)?;
            let harmonics: Vec<_> = parts
                .harmonics
                .into_iter()
                .map(|h| periodic_action::Harmonic {
                    index: h.index,
                    cos: h.cos,
                    sin: h.sin,
                })
                .collect();
            // density_model includes every C(t) cross harmonic, establishing
            // the complete CPTP generator required by its contractivity bound.
            let prepared = periodic_action::prepare_lindblad_for_initial(
                lap,
                &base.matrix,
                &harmonics,
                frame.omega,
                &base.initial,
                options.duration_us,
                5e-9,
                n,
            )?;
            let operators = frame.observable_fourier_with_scattering()?;
            let maximum_time = observable_horizon(options.duration_us);
            let observables = prepared.prepare_observables(lap, &operators, maximum_time)?;
            build_ms += build.elapsed().as_secs_f64() * 1000.;
            Entry {
                key: model.clone(),
                frame_model,
                radiation: radiation.cloned(),
                order,
                maximum_time,
                frame,
                gauges,
                prepared,
                observables,
            }
        };
        if options.duration_us > entry.maximum_time {
            let before = Instant::now();
            entry.maximum_time = observable_horizon(options.duration_us);
            entry.observables = entry.prepared.prepare_observables(
                lap,
                &entry.frame.observable_fourier_with_scattering()?,
                entry.maximum_time,
            )?;
            interval_extensions += 1;
            build_ms += before.elapsed().as_secs_f64() * 1000.;
        }
        let frame_initial = entry.frame.into_frame(&initial);
        let frame_vector = frame_coordinates(&original.coords, &frame_initial, n);
        let apply = |entry: &Entry| -> Result<_, String> {
            Ok((
                entry
                    .prepared
                    .apply_with_tolerance(lap, &frame_vector, &verification, 5e-9)?,
                entry.prepared.apply_observables(
                    lap,
                    &entry.observables,
                    &frame_vector,
                    &observed_times,
                    5e-9,
                )?,
            ))
        };
        let action_result = apply(&entry);
        let (action, observed) = if action_result.is_err()
            && (reused_preparation || entry.prepared.diagnostics.raw_modal_components > 0)
        {
            // Bounded Fourier terms chosen for an older interval/initial
            // density may fail the new gate. Rebuild their preparation using
            // the same complete optical frame, never relax the error budget.
            let before = Instant::now();
            let parts = entry
                .frame
                .density_model(lap, &entry.frame_model, &initial)?;
            let base = hilbert_floquet::build_density_generator(lap, &parts.model)?;
            let harmonics: Vec<_> = parts
                .harmonics
                .into_iter()
                .map(|h| periodic_action::Harmonic {
                    index: h.index,
                    cos: h.cos,
                    sin: h.sin,
                })
                .collect();
            entry.prepared = entry.prepared.reprepare_for_interval(
                lap,
                &base.matrix,
                &harmonics,
                options.duration_us,
                5e-9,
            )?;
            entry.maximum_time = observable_horizon(options.duration_us);
            entry.observables = entry.prepared.prepare_observables(
                lap,
                &entry.frame.observable_fourier_with_scattering()?,
                entry.maximum_time,
            )?;
            all_cached = false;
            build_ms += before.elapsed().as_secs_f64() * 1000.;
            apply(&entry)?
        } else {
            action_result?
        };
        apply_ms += action.elapsed_ms + observed.elapsed_ms;
        let mut physical = Vec::with_capacity(verification.len());
        let mut trace_error = 0_f64;
        let mut positivity_error = 0_f64;
        let mut initial_error = 0_f64;
        let mut hermiticity_error = 0_f64;
        let mut projection_error = 0_f64;
        for row in &observed.values {
            if row.len() != n + 1 || row.iter().any(|v| !v.is_finite()) {
                return Err("Non-finite physical observable projection".into());
            }
            trace_error = trace_error.max((row[..n].iter().sum::<f64>() - 1.).abs());
            positivity_error =
                positivity_error.max(row[..n].iter().map(|v| (-v).max(0.)).fold(0., f64::max));
            if row[n] < -1e-8 * scattering_scale {
                return Err("Negative photon scattering in observable projection".into());
            }
        }
        for (index, (x, &t)) in action.states.iter().zip(&verification).enumerate() {
            let sigma = expand_density(&original.coords, x, n);
            let mut rho = entry.frame.physical_density(&sigma, t);
            crate::multitone_frame::restore_single_beat_density(
                &mut rho,
                n,
                &entry.gauges,
                entry.frame.omega,
                t,
            );
            if rho.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) {
                return Err("Floquet-frame reconstruction produced a non-finite density".into());
            }
            for i in 0..n {
                for j in 0..n {
                    hermiticity_error =
                        hermiticity_error.max((rho[i * n + j] - rho[j * n + i].conj()).abs());
                }
            }
            let coordinates = frame_coordinates(&original.coords, &rho, n);
            trace_error = trace_error.max((coordinates[..n].iter().sum::<f64>() - 1.).abs());
            positivity_error = positivity_error.max(positivity(lap, &rho, n)?);
            let projected = &observed.values[observed_times
                .binary_search_by(|v| v.total_cmp(&t))
                .unwrap()];
            projection_error = projection_error.max(
                coordinates[..n]
                    .iter()
                    .zip(&projected[..n])
                    .map(|(a, b)| (a - b).abs())
                    .fold(0., f64::max),
            );
            projection_error = projection_error.max(
                (entry.frame.scattering_rate(&sigma, t) - projected[n]).abs() / scattering_scale,
            );
            if index == 0 {
                initial_error = max_abs(
                    &coordinates
                        .iter()
                        .zip(&original.initial)
                        .map(|(a, b)| a - b)
                        .collect::<Vec<_>>(),
                );
            }
            physical.push(coordinates);
        }
        if trace_error
            .max(positivity_error)
            .max(initial_error)
            .max(hermiticity_error)
            .max(projection_error)
            > 1e-8
        {
            return Err(format!("Floquet-frame density verification failed: trace={trace_error:e}, positivity={positivity_error:e}, initial={initial_error:e}, Hermiticity={hermiticity_error:e}, projection={projection_error:e}"));
        }
        let density_delta = previous
            .as_ref()
            .map(|old| {
                physical
                    .iter()
                    .zip(old)
                    .flat_map(|(a, b)| a.iter().zip(b).map(|(a, b)| (a - b).abs()))
                    .fold(0., f64::max)
            })
            .unwrap_or(f64::INFINITY);
        let observable_delta = previous_observed
            .as_ref()
            .map(|old| {
                observed
                    .values
                    .iter()
                    .zip(old)
                    .flat_map(|(a, b)| {
                        a.iter().zip(b).enumerate().map(|(i, (a, b))| {
                            (a - b).abs() / if i == n { scattering_scale } else { 1. }
                        })
                    })
                    .fold(0., f64::max)
            })
            .unwrap_or(f64::INFINITY);
        let delta = density_delta.max(observable_delta);
        let diagnostics = json!({"method":"Rust / full-density Lindblad Floquet-frame propagation","engine":"rust-floquet",
            "frameHarmonicOrder":order,"frameConvergenceError":delta,"frameUnitarityBound":entry.frame.unitarity_bound,
            "frameSchrodingerResidual":entry.frame.schrodinger_residual,"frameIsolatedGap":entry.frame.spectral_gap,
            "exactRotatingStateGauge":entry.gauges,
            "radiationModel":if radiation.is_some(){ATOMIC_MODEL_SCHEMA}else{"generic-lindblad/1"},"rawModalComponents":entry.prepared.diagnostics.raw_modal_components,"modalBasisReused":entry.prepared.diagnostics.modal_basis_reused,
            "finiteTimeNumericalTailBound":action.bounds.total_coordinate_l1,"inverseInitialTailBound":action.bounds.inverse_initial,
            "inverseGeneratorTailBound":action.bounds.inverse_generator,"oscillatingConvolutionTailBound":action.bounds.oscillatory_convolution,
            "dysonRemainderBound":action.bounds.dyson_remainder,"modalCondition":entry.prepared.diagnostics.modal_condition,
            "boundedFourierTail":action.bounds.bounded_fourier_tail,"observableErrorBounds":observed.row_error_bounds,"observableProjectionError":projection_error,
            "constantComponentDimensions":entry.prepared.diagnostics.component_dimensions,"meanCorrectionNorm":entry.prepared.diagnostics.mean_correction_norm,
            "transformNorm":entry.prepared.diagnostics.summed_transform_norm,"minimumSylvesterGap":entry.prepared.diagnostics.minimum_sylvester_gap,
            "buildMs":build_ms,"propagateMs":apply_ms,"elapsedMs":started.elapsed().as_secs_f64()*1000.,"cached":all_cached,"observableIntervalExtensions":interval_extensions,
            "verificationSamples":verification.len(),"observableSamples":observed_times.len(),"hermiticityError":hermiticity_error,
            "numericalTailScope":"Neumann/Dyson finite-time truncation only; frame and modal floating-point error are checked separately."});
        CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            cache.push_back(entry);
            while cache.len() > 4 {
                cache.pop_front();
            }
        });
        if delta < 1e-8 {
            let states = if options.return_density {
                times
                    .iter()
                    .map(|t| {
                        physical[verification.binary_search_by(|v| v.total_cmp(t)).unwrap()].clone()
                    })
                    .collect()
            } else {
                vec![]
            };
            return Ok(Report {
                times,
                states,
                populations: requested
                    .iter()
                    .map(|&i| observed.values[i][..n].to_vec())
                    .collect(),
                scattering: requested.iter().map(|&i| observed.values[i][n]).collect(),
                diagnostics,
                trace_error,
                positivity_error,
                initial_error,
            });
        }
        previous = Some(physical);
        previous_observed = Some(observed.values);
    }
    Err("The full-density Floquet-frame harmonic refinement did not converge; use the original Fourier method".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "Corrected32-state duration edit must reuse preparation and agree with a fresh solve"]
    fn corrected_bath_duration_cache_audit() {
        let request: Value = serde_json::from_str(
            &std::fs::read_to_string(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../artifacts/v4/corrected-bath/default-finite-request.json"
            ))
            .unwrap(),
        )
        .unwrap();
        let model: Model = serde_json::from_value(request["model"].clone()).unwrap();
        let radiation: crate::radiative_bath::Radiation =
            serde_json::from_value(request["radiation"].clone()).unwrap();
        let lap = Lapack::load().unwrap();
        CACHE.with(|cache| cache.borrow_mut().clear());
        let first = trajectory_with_radiation(
            &lap,
            &model,
            Some(&radiation),
            &Options {
                duration_us: 20.,
                samples: 161,
                ..Options::default()
            },
        )
        .unwrap();
        let options = Options {
            duration_us: 30.,
            samples: 161,
            return_density: true,
            ..Options::default()
        };
        let extended = trajectory_with_radiation(&lap, &model, Some(&radiation), &options).unwrap();
        assert_eq!(extended.diagnostics["cached"], true);
        assert_eq!(extended.diagnostics["buildMs"], 0.);
        CACHE.with(|cache| cache.borrow_mut().clear());
        let fresh = trajectory_with_radiation(&lap, &model, Some(&radiation), &options).unwrap();
        let difference = extended
            .states
            .iter()
            .zip(&fresh.states)
            .flat_map(|(a, b)| a.iter().zip(b).map(|(a, b)| (a - b).abs()))
            .fold(0., f64::max);
        assert!(
            difference < 1e-8,
            "Duration extension changed full-density result by {difference:e}"
        );
        std::fs::write(concat!(env!("CARGO_MANIFEST_DIR"),"/../artifacts/v4/corrected-bath/duration-cache-audit.json"),json!({"first":first.diagnostics,"extended":extended.diagnostics,"fresh":fresh.diagnostics,"fullDensityDifference":difference}).to_string()).unwrap();
        println!(
            "duration-cache full-density difference={difference:e}; extended={}",
            extended.diagnostics
        );
    }
    #[test]
    #[ignore = "Real simultaneous-D1/D2 example with exact manifold rotating gauge"]
    fn both_fine_lines_gauged_full_density_audit() {
        let model: Model = serde_json::from_str(
            &std::fs::read_to_string(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../artifacts/v4/both-lines-master.json"
            ))
            .unwrap(),
        )
        .unwrap();
        let lap = Lapack::load().unwrap();
        let options = Options {
            duration_us: 0.5,
            samples: 161,
            return_density: true,
            ..Options::default()
        };
        let result = trajectory(&lap, &model, &options).unwrap();
        let gauges = result.diagnostics["exactRotatingStateGauge"]
            .as_array()
            .unwrap();
        for (i, state) in model.states.iter().enumerate() {
            assert_eq!(
                gauges[i].as_i64().unwrap(),
                if state.starts_with("D1-") { 1 } else { 0 }
            );
        }
        assert_eq!(result.states.len(), 161);
        assert!(
            result
                .trace_error
                .max(result.positivity_error)
                .max(result.initial_error)
                < 1e-8
        );
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
        std::fs::write(concat!(env!("CARGO_MANIFEST_DIR"),"/../artifacts/v4/both-lines-gauge-audit.json"),json!({"cold":result.diagnostics,"warm":warm.diagnostics,"timesUs":result.times,"coordinates":result.states,"populations":result.populations,"scattering":result.scattering,"trace":result.trace_error,"positivity":result.positivity_error,"initialError":result.initial_error}).to_string()).unwrap();
        println!("cold={} warm={}", result.diagnostics, warm.diagnostics);
    }
    #[test]
    #[ignore = "Full32-state fast periodic path against the independent original-Fourier K3 density"]
    fn default_rb32_fast_trajectory_audit() {
        let model: Model = serde_json::from_str(
            &std::fs::read_to_string(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../artifacts/v4/default-master.json"
            ))
            .unwrap(),
        )
        .unwrap();
        let lap = Lapack::load().unwrap();
        let options = Options {
            duration_us: 20.,
            samples: 161,
            return_density: true,
            ..Options::default()
        };
        let result = trajectory(&lap, &model, &options).unwrap();
        println!("{}", result.diagnostics);
        let reference: Value = serde_json::from_str(
            &std::fs::read_to_string(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../artifacts/v4/native-dense-k3.json"
            ))
            .unwrap(),
        )
        .unwrap();
        let mut error = 0_f64;
        for (a, b) in result
            .states
            .iter()
            .zip(reference["coordinates"].as_array().unwrap())
        {
            for (x, y) in a.iter().zip(b.as_array().unwrap()) {
                error = error.max((x - y.as_f64().unwrap()).abs());
            }
        }
        assert!(
            error < 1e-8,
            "independent original-Fourier density difference {error:e}"
        );
        println!("Independent K3 full-density difference: {error:e}");
        let repeated = trajectory(
            &lap,
            &model,
            &Options {
                return_density: false,
                ..options
            },
        )
        .unwrap();
        println!("Warm observable: {}", repeated.diagnostics);
        assert!(repeated.states.is_empty());
        assert_eq!(result.populations, repeated.populations);
        assert_eq!(result.scattering, repeated.scattering);
        std::fs::write(concat!(env!("CARGO_MANIFEST_DIR"),"/../artifacts/v4/native-fast-periodic-audit.json"),json!({"errorVersusOriginalK3":error,"cold":result.diagnostics,"warm":repeated.diagnostics}).to_string()).unwrap();
    }
}
