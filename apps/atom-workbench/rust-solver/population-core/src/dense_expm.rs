//! Dense matrix-exponential action without an eigenvector decomposition.
//!
//! Fixed [13/13] Pade, scaling and squaring: Higham (2005), Algorithm 2.3,
//! SIAM J. Matrix Anal. Appl. 26(4), 1179-1193, doi:10.1137/04061101X.
//! https://eprints.maths.manchester.ac.uk/634/
//! Coefficients are the closed-form diagonal-Pade coefficients, normalized
//! by b0. theta13=5.371920351148152 is the double-precision backward-error
//! threshold in that paper. This is the conservative norm-based 2005 method,
//! not the adaptive-power-norm 2009 algorithm used by modern SciPy.
//!
//! Matrices are column-major. No positivity repair, trace normalization,
//! eigenmode deletion, or rate-equation replacement occurs here. Jordan and
//! defective generators are valid inputs. A large imaginary spectrum can
//! amplify rounding through many squarings; the diagnostics are deliberately
//! not a universal forward-accuracy promise.
use super::lapack::Lapack;

const THETA13: f64 = 5.371_920_351_148_152;
const UNIT_ROUNDOFF: f64 = f64::EPSILON * 0.5;
const MAX_SCALINGS: u32 = 64;
const B: [f64; 14] = [
    64_764_752_532_480_000.,
    32_382_376_266_240_000.,
    7_771_770_303_897_600.,
    1_187_353_796_428_800.,
    129_060_195_264_000.,
    10_559_470_521_600.,
    670_442_572_800.,
    33_522_128_640.,
    1_323_241_920.,
    40_840_800.,
    960_960.,
    16_380.,
    182.,
    1.,
];

#[derive(Clone, Debug)]
pub struct StepDiagnostics {
    pub time: f64,
    pub scaled_matrix_norm: f64,
    pub scalings: u32,
    pub pade_denominator_condition: f64,
    /// ||Q R-P||_1 / (||Q||_1 ||R||_1 + ||P||_1).
    pub linear_solve_backward_error: f64,
    /// Exact-arithmetic Pade truncation target, relative to the scaled input.
    pub pade_backward_error_target: f64,
    /// Initial floating-error *estimate*, propagated conservatively through
    /// E -> E^2 using e' <= 2||E||e+e^2+gamma_n||E||^2.
    /// This is not a certified total forward-error bound: the initial Pade
    /// rounding estimate is condition-based, not interval arithmetic. None
    /// indicates that even this risk estimate overflowed.
    pub matrix_roundoff_risk_estimate: Option<f64>,
    pub action_roundoff_risk_estimate: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct ActionDiagnostics {
    pub states: Vec<Vec<f64>>,
    pub steps: Vec<StepDiagnostics>,
}

fn finite(values: &[f64], stage: &str) -> Result<(), String> {
    if values.iter().any(|x| !x.is_finite()) {
        Err(format!(
            "Dense Pade exponential produced a non-finite value in {stage}"
        ))
    } else {
        Ok(())
    }
}
fn one_norm(a: &[f64], n: usize) -> f64 {
    a.chunks(n)
        .map(|col| col.iter().map(|x| x.abs()).sum::<f64>())
        .fold(0., f64::max)
}
fn identity(n: usize) -> Vec<f64> {
    let mut a = vec![0.; n * n];
    for i in 0..n {
        a[i + i * n] = 1.;
    }
    a
}
fn gamma(operations: usize) -> f64 {
    let value = operations as f64 * UNIT_ROUNDOFF;
    if value < 1. {
        value / (1. - value)
    } else {
        f64::INFINITY
    }
}
fn product(lap: &Lapack, a: &[f64], b: &[f64], n: usize, stage: &str) -> Result<Vec<f64>, String> {
    let value = lap.multiply(a, b, n, n, n);
    finite(&value, stage)?;
    Ok(value)
}
fn polynomial(n: usize, terms: &[(&[f64], usize)], diagonal: usize) -> Vec<f64> {
    let mut value = vec![0.; n * n];
    for &(a, k) in terms {
        let coefficient = B[k] / B[0];
        for (out, &x) in value.iter_mut().zip(a) {
            *out += coefficient * x;
        }
    }
    if diagonal < 14 {
        let coefficient = B[diagonal] / B[0];
        for i in 0..n {
            value[i + i * n] += coefficient;
        }
    }
    value
}

pub(crate) fn exponential(
    lap: &Lapack,
    a: &[f64],
    n: usize,
    t: f64,
) -> Result<(Vec<f64>, StepDiagnostics), String> {
    let mut at: Vec<_> = a.iter().map(|&x| x * t).collect();
    finite(&at, "A*t")?;
    let norm = one_norm(&at, n);
    if !norm.is_finite() {
        return Err("Dense Pade exponential input 1-norm overflowed".into());
    }
    if norm == 0. {
        return Ok((
            identity(n),
            StepDiagnostics {
                time: t,
                scaled_matrix_norm: 0.,
                scalings: 0,
                pade_denominator_condition: 1.,
                linear_solve_backward_error: 0.,
                pade_backward_error_target: 0.,
                matrix_roundoff_risk_estimate: Some(0.),
                action_roundoff_risk_estimate: Some(0.),
            },
        ));
    }
    let scaling_float = (norm / THETA13).log2().ceil().max(0.);
    if !scaling_float.is_finite() || scaling_float > MAX_SCALINGS as f64 {
        return Err(format!("Dense Pade exponential requires {scaling_float} squarings (limit {MAX_SCALINGS}); double-precision phase accuracy cannot be assessed for this input"));
    }
    let scalings = scaling_float as u32;
    let factor = 2_f64.powi(-(scalings as i32));
    for value in &mut at {
        *value *= factor;
    }
    let scaled_norm = one_norm(&at, n);
    let a2 = product(lap, &at, &at, n, "A2")?;
    let a4 = product(lap, &a2, &a2, n, "A4")?;
    let a6 = product(lap, &a4, &a2, n, "A6")?;
    // U=A[A6(b13 A6+b11 A4+b9 A2)+b7 A6+b5 A4+b3 A2+b1 I].
    let u_high = polynomial(n, &[(&a6, 13), (&a4, 11), (&a2, 9)], 14);
    let mut u_inner = product(lap, &a6, &u_high, n, "odd Pade polynomial")?;
    let u_low = polynomial(n, &[(&a6, 7), (&a4, 5), (&a2, 3)], 1);
    for (x, y) in u_inner.iter_mut().zip(u_low) {
        *x += y;
    }
    let u = product(lap, &at, &u_inner, n, "odd Pade numerator")?;
    // V=A6(b12 A6+b10 A4+b8 A2)+b6 A6+b4 A4+b2 A2+b0 I.
    let v_high = polynomial(n, &[(&a6, 12), (&a4, 10), (&a2, 8)], 14);
    let mut v = product(lap, &a6, &v_high, n, "even Pade polynomial")?;
    let v_low = polynomial(n, &[(&a6, 6), (&a4, 4), (&a2, 2)], 0);
    for (x, y) in v.iter_mut().zip(v_low) {
        *x += y;
    }
    let q: Vec<_> = v.iter().zip(&u).map(|(a, b)| a - b).collect();
    let p: Vec<_> = v.iter().zip(&u).map(|(a, b)| a + b).collect();
    finite(&q, "Pade denominator")?;
    finite(&p, "Pade numerator")?;
    let (lu, pivots) = lap.factor(&q, n)?;
    let q_norm = one_norm(&q, n);
    let condition = lap.condition(&lu, n, q_norm)?;
    if !condition.is_finite() {
        return Err("Dense Pade denominator condition is not finite".into());
    }
    // Existing LP64 wrapper accepts one RHS. Factor once, then solve every
    // column; passing n*n entries to a one-RHS dgetrs would be incorrect.
    let mut result = vec![0.; n * n];
    for column in 0..n {
        let solved = lap.solve_factored(&lu, &pivots, &p[column * n..(column + 1) * n], n)?;
        result[column * n..(column + 1) * n].copy_from_slice(&solved);
    }
    finite(&result, "Pade linear solve")?;
    let mut residual = product(lap, &q, &result, n, "Pade residual")?;
    for (x, &rhs) in residual.iter_mut().zip(&p) {
        *x -= rhs;
    }
    let result_norm = one_norm(&result, n);
    let denominator = q_norm * result_norm + one_norm(&p, n);
    let solve_error = one_norm(&residual, n) / denominator;
    if !solve_error.is_finite() {
        return Err("Dense Pade solve residual is not finite".into());
    }
    let local_rounding = condition * (solve_error + gamma(6 * n + 40)) * result_norm;
    let mut risk = local_rounding + UNIT_ROUNDOFF * scaled_norm * result_norm;
    for _ in 0..scalings {
        let previous_norm = one_norm(&result, n);
        result = product(lap, &result, &result, n, "squaring")?;
        risk = 2. * previous_norm * risk + risk * risk + gamma(n) * previous_norm * previous_norm;
    }
    Ok((
        result,
        StepDiagnostics {
            time: t,
            scaled_matrix_norm: scaled_norm,
            scalings,
            pade_denominator_condition: condition,
            linear_solve_backward_error: solve_error,
            pade_backward_error_target: UNIT_ROUNDOFF,
            matrix_roundoff_risk_estimate: risk.is_finite().then_some(risk),
            action_roundoff_risk_estimate: None,
        },
    ))
}

/// Evaluate exp(t A) x independently at every requested finite time. In
/// particular, no nonnormal eigenvector inversion is attempted. Prefer the
/// diagnostic variant when accepting results against a physical tolerance.
pub fn action(
    lap: &Lapack,
    a: &[f64],
    initial: &[f64],
    times: &[f64],
) -> Result<Vec<Vec<f64>>, String> {
    Ok(action_with_diagnostics(lap, a, initial, times)?.states)
}

pub fn action_with_diagnostics(
    lap: &Lapack,
    a: &[f64],
    initial: &[f64],
    times: &[f64],
) -> Result<ActionDiagnostics, String> {
    let n = initial.len();
    if n == 0 || n > i32::MAX as usize || n.checked_mul(n) != Some(a.len()) {
        return Err("Dense exponential expects an n*n column-major matrix and n-vector".into());
    }
    finite(a, "input matrix")?;
    finite(initial, "initial vector")?;
    finite(times, "sample times")?;
    let initial_norm = initial.iter().map(|v| v.abs()).sum::<f64>();
    if !initial_norm.is_finite() {
        return Err("Dense exponential initial vector norm overflowed".into());
    }
    let mut states = Vec::with_capacity(times.len());
    let mut steps = Vec::with_capacity(times.len());
    for &time in times {
        let (matrix, mut diagnostics) = exponential(lap, a, n, time)?;
        let state = if time == 0. {
            initial.to_vec()
        } else {
            lap.multiply(&matrix, initial, n, 1, n)
        };
        finite(&state, "exponential action")?;
        let multiplication_risk = gamma(n) * one_norm(&matrix, n) * initial_norm;
        diagnostics.action_roundoff_risk_estimate = diagnostics
            .matrix_roundoff_risk_estimate
            .map(|risk| risk * initial_norm + multiplication_risk)
            .filter(|risk| risk.is_finite());
        states.push(state);
        steps.push(diagnostics);
    }
    Ok(ActionDiagnostics { states, steps })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn near(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol, "{a} != {b}, tolerance {tol}");
    }
    #[test]
    fn zero_matrix_preserves_every_coordinate() {
        let lap = Lapack::load().unwrap();
        let initial = vec![0.2, -0.7, 1.3];
        for state in action(&lap, &[0.; 9], &initial, &[0., 0.3, 100.]).unwrap() {
            assert_eq!(state, initial);
        }
    }
    #[test]
    fn defective_jordan_exponential_is_polynomial_times_decay() {
        let lap = Lapack::load().unwrap();
        let times = [0., 0.03, 0.5, 2., 10.];
        let result = action(&lap, &[-0.7, 0., 1., -0.7], &[0., 1.], &times).unwrap();
        for (x, t) in result.iter().zip(times) {
            near(x[0], t * (-0.7 * t).exp(), 3e-14);
            near(x[1], (-0.7 * t).exp(), 3e-14);
        }
    }
    #[test]
    fn damped_rotation_preserves_phase_without_an_eigenbasis() {
        let lap = Lapack::load().unwrap();
        let times = [0., 0.01, 0.3, 2.];
        for omega in [1., 7., 1000.] {
            let report =
                action_with_diagnostics(&lap, &[-0.2, omega, -omega, -0.2], &[1., 0.], &times)
                    .unwrap();
            for (x, t) in report.states.iter().zip(times) {
                near(x[0], (-0.2 * t).exp() * (omega * t).cos(), 2e-12);
                near(x[1], (-0.2 * t).exp() * (omega * t).sin(), 2e-12);
            }
            assert!(report
                .steps
                .iter()
                .all(|s| s.linear_solve_backward_error < 1e-15));
        }
    }
    #[test]
    fn critically_damped_optical_bloch_jordan_block_preserves_trace() {
        // Coordinates [trace,y,z], gamma=4, Omega=1=gamma/4. The yz
        // block is defective: A_yz=-3 I+N, N^2=0. Start in |ground>.
        let lap = Lapack::load().unwrap();
        let a = [0., 0., -4., 0., -2., 1., 0., -1., -4.];
        let times = [0., 0.01, 0.2, 1., 10., 100.];
        let report = action_with_diagnostics(&lap, &a, &[1., 0., -1.], &times).unwrap();
        for (x, t) in report.states.iter().zip(times) {
            near(x[0], 1., 4e-13);
            near(x[1], 4. / 9. + (-3. * t).exp() * (-4. / 9. - t / 3.), 4e-13);
            near(
                x[2],
                -8. / 9. + (-3. * t).exp() * (-1. / 9. - t / 3.),
                4e-13,
            );
            assert!(x[1] * x[1] + x[2] * x[2] <= 1. + 1e-12);
        }
    }
    #[test]
    fn high_frequency_reports_rounding_risk_instead_of_uniform_accuracy_claim() {
        let lap = Lapack::load().unwrap();
        let omega = 1e8;
        let t = 0.1;
        let report =
            action_with_diagnostics(&lap, &[0., omega, -omega, 0.], &[1., 0.], &[t]).unwrap();
        let error = (report.states[0][0] - (omega * t).cos())
            .hypot(report.states[0][1] - (omega * t).sin());
        assert!(error < 1e-7, "rotation forward error {error:e}");
        assert!(report.steps[0].scalings >= 20);
        assert!(report.steps[0]
            .action_roundoff_risk_estimate
            .map_or(true, |risk| risk > 1e-10));
    }
    #[test]
    fn nonfinite_overflow_and_excessive_scaling_fail_explicitly() {
        let lap = Lapack::load().unwrap();
        assert!(action(&lap, &[f64::NAN], &[1.], &[1.]).is_err());
        assert!(action(&lap, &[1.], &[1.], &[f64::INFINITY]).is_err());
        assert!(action(&lap, &[1e300], &[1.], &[1.])
            .unwrap_err()
            .contains("squarings"));
        assert!(action(&lap, &[1000.], &[1.], &[1.])
            .unwrap_err()
            .contains("non-finite"));
        assert!(action(&lap, &[1e308], &[1.], &[2.])
            .unwrap_err()
            .contains("non-finite"));
    }
}
