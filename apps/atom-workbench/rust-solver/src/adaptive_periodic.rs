//! Direct integration of the COMPLETE real Hermitian-coordinate equation.
//!
//! Dormand & Prince, J. Comput. Appl. Math. 6 (1980), 19–26, DOI
//! 10.1016/0771-050X(80)90013-3. Coefficients are the published DP5(4)
//! tableau, also documented by SciPy's RK45 implementation:
//! https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.RK45.html
//! No populations/coherences are removed, normalized, clipped or repaired.
//! The embedded estimate controls LOCAL error. `integrate_verified` repeats
//! the complete trajectory at a tighter tolerance and checks GLOBAL sample
//! agreement; this observed comparison is not an a-priori uniform error bound.
use crate::{cluster_action::Sparse, periodic_action::Harmonic};
use std::time::Instant;

const MAX_ATTEMPTS: usize = 250_000;
const MAX_MULTIPLICATIONS: usize = 250_000_000;
const C: [f64; 7] = [0., 1. / 5., 3. / 10., 4. / 5., 8. / 9., 1., 1.];
const A: [[f64; 7]; 7] = [
    [0.; 7],
    [1. / 5., 0., 0., 0., 0., 0., 0.],
    [3. / 40., 9. / 40., 0., 0., 0., 0., 0.],
    [44. / 45., -56. / 15., 32. / 9., 0., 0., 0., 0.],
    [
        19372. / 6561.,
        -25360. / 2187.,
        64448. / 6561.,
        -212. / 729.,
        0.,
        0.,
        0.,
    ],
    [
        9017. / 3168.,
        -355. / 33.,
        46732. / 5247.,
        49. / 176.,
        -5103. / 18656.,
        0.,
        0.,
    ],
    [
        35. / 384.,
        0.,
        500. / 1113.,
        125. / 192.,
        -2187. / 6784.,
        11. / 84.,
        0.,
    ],
];
// Embedded fourth-order minus fifth-order formula (including the FSAL stage).
const E: [f64; 7] = [
    -71. / 57600.,
    0.,
    71. / 16695.,
    -71. / 1920.,
    17253. / 339200.,
    -22. / 525.,
    1. / 40.,
];

#[derive(Debug)]
pub struct Action {
    pub states: Vec<Vec<f64>>,
    pub accepted: usize,
    pub rejected: usize,
    pub rhs_evaluations: usize,
    /// Maximum accepted-step absolute embedded LOCAL error, not global error.
    pub max_error: f64,
    pub max_error_ratio: f64,
    pub sum_local_error: f64,
    pub min_step: f64,
    pub max_step: f64,
    pub phase_step_limit: f64,
    pub elapsed_ms: f64,
}
#[derive(Debug)]
pub struct Verified {
    pub action: Action,
    pub comparison_error: f64,
    pub coarse_tolerance: f64,
    pub fine_tolerance: f64,
    pub coarse_accepted: usize,
    pub coarse_rejected: usize,
    pub total_rhs_evaluations: usize,
    pub comparison_step_limit: f64,
}

fn validate(matrix: &Sparse, n: usize) -> Result<(usize, f64), String> {
    if matrix.len() != n {
        return Err("Periodic operator dimension mismatch".into());
    }
    let mut count = 0;
    let mut norm = 0_f64;
    for column in matrix {
        let mut sum = 0.;
        for &(row, value) in column {
            if row >= n || !value.is_finite() {
                return Err("Invalid periodic operator entry".into());
            }
            sum += value.abs();
            count += 1;
        }
        norm = norm.max(sum);
    }
    if !norm.is_finite() {
        return Err("Periodic operator norm overflow".into());
    }
    Ok((count, norm))
}
fn add_product(matrix: &Sparse, x: &[f64], factor: f64, y: &mut [f64]) {
    for (j, column) in matrix.iter().enumerate() {
        let value = factor * x[j];
        if value == 0. {
            continue;
        }
        for &(i, a) in column {
            y[i] += a * value;
        }
    }
}
fn angle(index: i32, omega: f64, time: f64) -> Result<(f64, f64), String> {
    let high = omega * time;
    let low = omega.mul_add(time, -high);
    let value = high * index as f64;
    let tail = high.mul_add(index as f64, -value) + low * index as f64;
    if !value.is_finite() || !tail.is_finite() {
        return Err("Periodic phase overflow".into());
    }
    let (s, c) = value.sin_cos();
    let (ds, dc) = tail.sin_cos();
    Ok((s * dc + c * ds, c * dc - s * ds))
}
fn rhs(
    base: &Sparse,
    harmonics: &[Harmonic],
    omega: f64,
    time: f64,
    x: &[f64],
    y: &mut [f64],
) -> Result<(), String> {
    y.fill(0.);
    add_product(base, x, 1., y);
    for h in harmonics {
        let (s, c) = angle(h.index, omega, time)?;
        add_product(&h.cos, x, c, y);
        add_product(&h.sin, x, s, y);
    }
    if y.iter().any(|x| !x.is_finite()) {
        return Err("Periodic derivative is non-finite".into());
    }
    Ok(())
}

pub fn integrate(
    base: &Sparse,
    harmonics: &[Harmonic],
    omega: f64,
    initial: &[f64],
    times: &[f64],
    tolerance: f64,
) -> Result<Action, String> {
    integrate_limited(
        base,
        harmonics,
        omega,
        initial,
        times,
        tolerance,
        f64::INFINITY,
    )
}
fn integrate_limited(
    base: &Sparse,
    harmonics: &[Harmonic],
    omega: f64,
    initial: &[f64],
    times: &[f64],
    tolerance: f64,
    extra_step_limit: f64,
) -> Result<Action, String> {
    let started = Instant::now();
    let n = initial.len();
    if n == 0
        || initial.iter().any(|x| !x.is_finite())
        || times.is_empty()
        || times.iter().any(|t| !t.is_finite() || *t < 0.)
        || times.windows(2).any(|w| w[1] <= w[0])
    {
        return Err("Invalid periodic initial vector or increasing sample times".into());
    }
    if !tolerance.is_finite() || tolerance < 32. * f64::EPSILON || tolerance > 0.01 {
        return Err(
            "Periodic integration tolerance is outside the supported floating-point range".into(),
        );
    }
    if !omega.is_finite() || omega < 0. || (!harmonics.is_empty() && omega == 0.) {
        return Err("Periodic frequency must be finite and positive".into());
    }
    let (mut nonzero, mut norm) = validate(base, n)?;
    let mut max_index = 0_u32;
    for h in harmonics {
        if h.index == 0 {
            return Err("Zero Fourier harmonic belongs in the base operator".into());
        }
        max_index = max_index.max(h.index.unsigned_abs());
        for matrix in [&h.cos, &h.sin] {
            let (count, size) = validate(matrix, n)?;
            nonzero = nonzero
                .checked_add(count)
                .ok_or("Periodic operator size overflow")?;
            norm += size;
        }
    }
    if !norm.is_finite() {
        return Err("Complete periodic operator norm overflow".into());
    }
    let end = *times.last().unwrap();
    let phase_limit = if max_index == 0 {
        f64::INFINITY
    } else {
        2. * std::f64::consts::PI / (8. * omega * max_index as f64)
    };
    let step_limit = phase_limit.min(extra_step_limit);
    if step_limit <= 0. || end / step_limit > MAX_ATTEMPTS as f64 {
        return Err(
            "Resolved periodic phase/refinement sampling exceeds the integration step budget"
                .into(),
        );
    }
    let minimum_steps = (end / step_limit).ceil() as usize;
    if minimum_steps.saturating_mul(6).saturating_mul(nonzero) > MAX_MULTIPLICATIONS {
        return Err("Resolved periodic equation exceeds the sparse multiplication budget".into());
    }
    let mut action = Action {
        states: Vec::with_capacity(times.len()),
        accepted: 0,
        rejected: 0,
        rhs_evaluations: 0,
        max_error: 0.,
        max_error_ratio: 0.,
        sum_local_error: 0.,
        min_step: f64::INFINITY,
        max_step: 0.,
        phase_step_limit: phase_limit,
        elapsed_ms: 0.,
    };
    let mut y = initial.to_vec();
    let mut stages = vec![vec![0.; n]; 7];
    let mut candidate = vec![0.; n];
    let mut temporary = vec![0.; n];
    let mut time = 0.;
    let mut step = if end == 0. {
        0.
    } else {
        (end / 100.)
            .min(if norm > 0. { 0.1 / norm } else { end })
            .min(step_limit)
    };
    rhs(base, harmonics, omega, 0., &y, &mut stages[0])?;
    action.rhs_evaluations = 1;
    let mut recently_rejected = false;
    for &target in times {
        while time < target {
            if action.accepted + action.rejected >= MAX_ATTEMPTS {
                return Err("Adaptive periodic integration exhausted its step budget".into());
            }
            if action
                .rhs_evaluations
                .saturating_add(6)
                .saturating_mul(nonzero)
                > MAX_MULTIPLICATIONS
            {
                return Err(
                    "Adaptive periodic integration exhausted its sparse multiplication budget"
                        .into(),
                );
            }
            let remaining = target - time;
            let requested = step.min(step_limit).min(remaining);
            let mut next = if requested == remaining {
                target
            } else {
                time + requested
            };
            if next - time > step_limit {
                next = next.next_down();
            }
            let h = next - time;
            if !h.is_finite() || h <= 0. || next <= time {
                return Err(
                    "Adaptive periodic step underflow before reaching the requested sample".into(),
                );
            }
            for stage in 1..7 {
                for i in 0..n {
                    let mut increment = 0.;
                    for j in 0..stage {
                        increment += A[stage][j] * stages[j][i];
                    }
                    temporary[i] = y[i] + h * increment;
                }
                if temporary.iter().any(|v| !v.is_finite()) {
                    return Err("Adaptive periodic stage is non-finite".into());
                }
                let stage_time = if stage >= 5 {
                    next
                } else {
                    time + C[stage] * h
                };
                rhs(
                    base,
                    harmonics,
                    omega,
                    stage_time,
                    &temporary,
                    &mut stages[stage],
                )?;
                action.rhs_evaluations += 1;
                if stage == 6 {
                    candidate.copy_from_slice(&temporary);
                }
            }
            let mut ratio = 0_f64;
            let mut absolute = 0_f64;
            for i in 0..n {
                let error = h * E.iter().zip(&stages).map(|(&a, k)| a * k[i]).sum::<f64>();
                let scale = tolerance * (1. + y[i].abs().max(candidate[i].abs()));
                absolute = absolute.max(error.abs());
                ratio = ratio.max(error.abs() / scale);
            }
            if !ratio.is_finite() || !absolute.is_finite() {
                return Err("Adaptive periodic error estimate is non-finite".into());
            }
            if ratio <= 1. {
                y.copy_from_slice(&candidate);
                time = next;
                action.accepted += 1;
                action.max_error = action.max_error.max(absolute);
                action.max_error_ratio = action.max_error_ratio.max(ratio);
                action.sum_local_error += absolute;
                action.min_step = action.min_step.min(h);
                action.max_step = action.max_step.max(h);
                stages.swap(0, 6); // FSAL: exact final-stage derivative, no interpolation.
                let factor = if ratio == 0. {
                    10.
                } else {
                    (0.9 * ratio.powf(-0.2)).clamp(0.2, 10.)
                };
                step = h * if recently_rejected {
                    factor.min(1.)
                } else {
                    factor
                };
                recently_rejected = false;
            } else {
                action.rejected += 1;
                step = h * (0.9 * ratio.powf(-0.2)).clamp(0.2, 1.);
                recently_rejected = true;
            }
        }
        action.states.push(y.clone());
    }
    if action.min_step == f64::INFINITY {
        action.min_step = 0.;
    }
    action.elapsed_ms = started.elapsed().as_secs_f64() * 1000.;
    Ok(action)
}

pub fn integrate_verified(
    base: &Sparse,
    harmonics: &[Harmonic],
    omega: f64,
    initial: &[f64],
    times: &[f64],
    coarse_tolerance: f64,
    fine_tolerance: f64,
    agreement_tolerance: f64,
) -> Result<Verified, String> {
    if fine_tolerance >= coarse_tolerance
        || !agreement_tolerance.is_finite()
        || agreement_tolerance <= 0.
    {
        return Err(
            "Periodic verification requires tighter recomputation and a finite agreement tolerance"
                .into(),
        );
    }
    let coarse = integrate(base, harmonics, omega, initial, times, coarse_tolerance)?;
    // Samples or the phase cap can dominate both tolerance runs, accidentally
    // giving identical meshes and a meaningless zero comparison. Halving the
    // observed coarse maximum step makes the second integration a real mesh
    // refinement too, while retaining precise requested sample times.
    let comparison_step_limit = if coarse.max_step > 0. {
        coarse.max_step / 2.
    } else {
        f64::INFINITY
    };
    let fine = integrate_limited(
        base,
        harmonics,
        omega,
        initial,
        times,
        fine_tolerance,
        comparison_step_limit,
    )?;
    let error = coarse
        .states
        .iter()
        .zip(&fine.states)
        .flat_map(|(a, b)| a.iter().zip(b).map(|(a, b)| (a - b).abs()))
        .fold(0_f64, f64::max);
    if !error.is_finite() || error > agreement_tolerance {
        return Err(format!("Periodic full-trajectory tolerance refinement failed: {error:e} > {agreement_tolerance:e}"));
    }
    Ok(Verified {
        comparison_error: error,
        coarse_tolerance,
        fine_tolerance,
        coarse_accepted: coarse.accepted,
        coarse_rejected: coarse.rejected,
        total_rhs_evaluations: coarse.rhs_evaluations + fine.rhs_evaluations,
        comparison_step_limit,
        action: fine,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn times(end: f64) -> Vec<f64> {
        (0..81).map(|i| end * i as f64 / 80.).collect()
    }
    #[test]
    fn constant_decay_preserves_complete_density() {
        let a = vec![
            vec![],
            vec![(0, 2.), (1, -2.)],
            vec![(2, -1.)],
            vec![(3, -1.)],
        ];
        let t = times(3.);
        let out =
            integrate_verified(&a, &[], 0., &[0., 1., 0., 0.], &t, 1e-10, 2e-12, 1e-8).unwrap();
        for (x, &t) in out.action.states.iter().zip(&t) {
            assert!((x[1] - (-2. * t).exp()).abs() < 1e-11);
            assert!((x[0] + x[1] - 1.).abs() < 2e-15);
        }
    }
    #[test]
    fn coherent_rabi_keeps_imaginary_coherence() {
        let a = vec![
            vec![(3, 2.)],
            vec![(3, -2.)],
            vec![],
            vec![(0, -4.), (1, 4.)],
        ];
        let t = times(4.);
        let out =
            integrate_verified(&a, &[], 0., &[1., 0., 0., 0.], &t, 1e-10, 2e-12, 1e-8).unwrap();
        for (x, &t) in out.action.states.iter().zip(&t) {
            assert!((x[0] - (2. * t).cos().powi(2)).abs() < 2e-11);
            assert!((x[3] - (2. * t).sin() * (2. * t).cos()).abs() < 2e-11);
            assert_eq!(x[2], 0.);
        }
    }
    #[test]
    fn periodic_scalar_matches_analytic_integral() {
        let base = vec![vec![(0, -0.7)]];
        let h = Harmonic {
            index: 1,
            cos: vec![vec![(0, 0.2)]],
            sin: vec![vec![(0, -0.1)]],
        };
        let t = times(2.);
        let out = integrate(&base, &[h], 4., &[1.], &t, 2e-12).unwrap();
        for (x, &t) in out.states.iter().zip(&t) {
            let exact =
                (-0.7 * t + 0.2 * (4. * t).sin() / 4. - 0.1 * (1. - (4. * t).cos()) / 4.).exp();
            assert!((x[0] - exact).abs() < 5e-12);
        }
        assert!(out.max_step <= out.phase_step_limit * (1. + f64::EPSILON));
    }
    #[test]
    fn exact_samples_and_zero_initial_are_not_repaired() {
        let t = [0., 1e-15, 0.1, 0.30000000000000004];
        let out = integrate(&vec![vec![(0, 1.)]], &[], 0., &[0.], &t, 1e-10).unwrap();
        assert_eq!(out.states, vec![vec![0.]; 4]);
        assert!(integrate(&vec![vec![]], &[], 0., &[1.], &[0., 0.], 1e-10).is_err());
    }
    #[test]
    fn excessive_resolved_frequency_fails_explicitly() {
        let h = Harmonic {
            index: 1,
            cos: vec![vec![(0, 0.1)]],
            sin: vec![vec![]],
        };
        assert!(integrate(&vec![vec![]], &[h], 1e9, &[1.], &[0., 1.], 1e-10)
            .unwrap_err()
            .contains("step budget"));
    }
    #[test]
    fn nonfinite_growth_and_failed_global_refinement_are_rejected() {
        assert!(integrate(&vec![vec![(0, 1e308)]], &[], 0., &[1e308], &[0., 1.], 1e-10).is_err());
        assert!(integrate_verified(
            &vec![vec![(0, -2.)]],
            &[],
            0.,
            &[1.],
            &times(2.),
            1e-5,
            1e-8,
            1e-30
        )
        .unwrap_err()
        .contains("tolerance refinement failed"));
    }
    #[test]
    fn complex_time_dependent_collapse_matches_independent_dop853() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/generator-mean-periodic-collapse.json"
        ))
        .unwrap();
        let matrix = |value: &serde_json::Value| -> Sparse {
            let x = value.as_array().unwrap();
            (0..4)
                .map(|j| {
                    (0..4)
                        .filter_map(|i| {
                            let value = x[i * 4 + j].as_f64().unwrap();
                            (value != 0.).then_some((i, value))
                        })
                        .collect()
                })
                .collect()
        };
        let base = matrix(&fixture["base"]);
        let harmonics: Vec<_> = fixture["drives"]
            .as_array()
            .unwrap()
            .iter()
            .map(|h| Harmonic {
                index: h["index"][0].as_i64().unwrap() as i32,
                cos: matrix(&h["cos"]),
                sin: matrix(&h["sin"]),
            })
            .collect();
        let omega = fixture["frequency"].as_f64().unwrap();
        let t: Vec<_> = fixture["oracle"]["phases"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() / omega)
            .collect();
        let initial: Vec<_> = fixture["oracle"]["initial"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let result =
            integrate_verified(&base, &harmonics, omega, &initial, &t, 1e-10, 2e-12, 1e-9).unwrap();
        let mut error = 0_f64;
        for (value, expected) in result
            .action
            .states
            .iter()
            .zip(fixture["oracle"]["densityCoordinates"].as_array().unwrap())
        {
            for i in 0..4 {
                error = error.max((value[i] - expected[i].as_f64().unwrap()).abs());
            }
        }
        assert!(error < 1e-11, "DOP853 disagreement {error:e}");
    }
}
