//! Frequency-aware a posteriori error envelope for a complete modal basis.
//!
//! In exact arithmetic, M=V^-1 A V=Lambda+D. The first Duhamel term has
//! entries D_ij (exp(lambda_j t)-exp(lambda_i t))/(lambda_j-lambda_i).
//! All entries of D are included. For 0<=t<=T its magnitude is bounded by
//! min(max t exp(max(Re lambda_i,Re lambda_j)t),
//!     (sup exp(Re lambda_i t)+sup exp(Re lambda_j t))/|lambda_j-lambda_i|).
//! The remaining Dyson terms are bounded by
//! exp(max(0,alpha)T) (exp(||D|| T)-1-||D||T) ||c||.
//!
//! Modal coordinates use weights ||V_j|| in the caller's physical Frobenius
//! metric. The supplied embedding factor converts that metric to trace norm.
//! The initial reconstruction term uses contractivity of the PHYSICAL
//! Lindblad semigroup. Do not use that term for an arbitrary truncated
//! extended generator without separately establishing contractivity.
//!
//! IEEE round-to-nearest/FMA error envelopes account for compensated AV-VL,
//! BLAS products, approximate inverse, and modal evaluation. They are
//! conservative numerical risk estimates, not an interval-arithmetic proof
//! of a platform's libm accuracy. The exact-arithmetic Duhamel/Dyson bound is
//! distinct from these floating estimates and remains in the diagnostics.
use super::{
    cluster_action::ModalBasis,
    dense_expm,
    lapack::{Complex64 as Z, Lapack},
};
use std::time::Instant;

const U: f64 = f64::EPSILON * 0.5;
fn gamma(n: usize) -> f64 {
    let x = n as f64 * U;
    x / (1. - x)
}
fn abs1(z: Z) -> f64 {
    (z.re.abs() + z.im.abs()) * (1. + 4. * U)
}
fn modulus(z: Z) -> f64 {
    z.re.hypot(z.im) * (1. + 8. * U)
}
fn finite(values: &[f64], name: &str) -> Result<(), String> {
    if values.iter().any(|x| !x.is_finite()) {
        Err(format!("Non-finite {name} in modal defect bound"))
    } else {
        Ok(())
    }
}
fn zfinite(values: &[Z], name: &str) -> Result<(), String> {
    if values
        .iter()
        .any(|x| !x.re.is_finite() || !x.im.is_finite())
    {
        Err(format!("Non-finite {name} in modal defect bound"))
    } else {
        Ok(())
    }
}
#[derive(Clone, Copy, Default)]
struct Double {
    hi: f64,
    lo: f64,
}
impl Double {
    fn add(&mut self, hi: f64, lo: f64) {
        let sum = self.hi + hi;
        let z = sum - self.hi;
        let error = (self.hi - (sum - z)) + (hi - z);
        let tail = self.lo + lo + error;
        let value = sum + tail;
        self.lo = tail - (value - sum);
        self.hi = value;
    }
    fn product(&mut self, a: f64, b: f64) {
        let p = a * b;
        self.add(p, a.mul_add(b, -p));
    }
    fn value(self) -> f64 {
        self.hi + self.lo
    }
}
#[derive(Clone, Copy, Default)]
struct Wide {
    re: Double,
    im: Double,
}
impl Wide {
    fn product(&mut self, a: Z, b: Z) {
        self.re.product(a.re, b.re);
        self.re.product(-a.im, b.im);
        self.im.product(a.re, b.im);
        self.im.product(a.im, b.re);
    }
    fn value(self) -> Z {
        Z {
            re: self.re.value(),
            im: self.im.value(),
        }
    }
}
fn complex_product(lap: &Lapack, a: &[Z], b: &[Z], n: usize) -> (Vec<Z>, Vec<f64>) {
    let ar: Vec<_> = a.iter().map(|z| z.re).collect();
    let ai: Vec<_> = a.iter().map(|z| z.im).collect();
    let br: Vec<_> = b.iter().map(|z| z.re).collect();
    let bi: Vec<_> = b.iter().map(|z| z.im).collect();
    let rr = lap.multiply(&ar, &br, n, n, n);
    let ii = lap.multiply(&ai, &bi, n, n, n);
    let ri = lap.multiply(&ar, &bi, n, n, n);
    let ir = lap.multiply(&ai, &br, n, n, n);
    let value = (0..n * n)
        .map(|i| Z {
            re: rr[i] - ii[i],
            im: ri[i] + ir[i],
        })
        .collect();
    let aa: Vec<_> = a.iter().map(|&z| abs1(z)).collect();
    let ba: Vec<_> = b.iter().map(|&z| abs1(z)).collect();
    let mag = lap
        .multiply(&aa, &ba, n, n, n)
        .into_iter()
        .map(|x| x / (1. - gamma(n)))
        .collect();
    (value, mag)
}
fn weighted_one(entries: &[f64], weights: &[f64], n: usize) -> f64 {
    (0..n)
        .map(|j| {
            (0..n)
                .map(|i| entries[i + j * n] * weights[i] / weights[j])
                .sum::<f64>()
                / (1. - gamma(4 * n + 4))
        })
        .fold(0., f64::max)
}
fn exponential(x: f64) -> f64 {
    let e = x.exp();
    if e == 0. {
        f64::MIN_POSITIVE
    } else {
        e * (1. + 32. * U)
    }
}
fn remainder(x: f64) -> f64 {
    if x == 0. {
        return 0.;
    }
    if x < 0.5 {
        let mut term = x * x * 0.5;
        let mut sum = term;
        for k in 3..100 {
            term *= x / k as f64;
            sum += term;
            if term <= sum * U {
                break;
            }
        }
        sum / (1. - gamma(300))
    } else {
        (x.exp_m1() - x) * (1. + 64. * U)
    }
}

#[derive(Clone, Debug)]
pub struct Diagnostics {
    pub dimension: usize,
    pub defect_norm_upper: f64,
    pub defect_product_roundoff: f64,
    pub inverse_residual_bound: f64,
    pub inverse_defect_uncertainty: f64,
    pub preparation_ms: f64,
}
#[derive(Clone, Debug)]
pub struct Bound {
    pub time: f64,
    pub initial_reconstruction: f64,
    pub first_order: f64,
    pub higher_order: f64,
    pub inverse_and_product_uncertainty: f64,
    pub floating_evaluation: f64,
    pub total: f64,
    pub coefficient_weighted_norm: f64,
    pub global_higher_order: f64,
    pub comparison_threshold: Option<f64>,
    pub comparison_roundoff: f64,
}
pub struct Action {
    pub states: Vec<Vec<f64>>,
    pub bound: Bound,
    pub elapsed_ms: f64,
}
struct Comparison {
    threshold: f64,
    groups: Vec<usize>,
    generator: Vec<f64>,
}
pub struct Prepared {
    pub diagnostics: Diagnostics,
    n: usize,
    modal: ModalBasis,
    metric: Vec<f64>,
    embedding: f64,
    weights: Vec<f64>,
    weighted_defect: Vec<f64>,
    inverse_gap: Vec<f64>,
    real_vectors: Vec<f64>,
    comparisons: Vec<Comparison>,
}

pub fn prepare(
    lap: &Lapack,
    a: &[f64],
    modal: ModalBasis,
    metric: &[f64],
    embedding: f64,
) -> Result<Prepared, String> {
    let started = Instant::now();
    let n = modal.dimension;
    if n == 0
        || a.len() != n * n
        || modal.vectors.len() != n * n
        || modal.inverse.len() != n * n
        || modal.eigenvalues.len() != n
        || metric.len() != n
        || metric.iter().any(|x| !x.is_finite() || *x <= 0.)
        || !embedding.is_finite()
        || embedding <= 0.
    {
        return Err("Invalid physical metric or basis dimensions for modal defect bound".into());
    }
    finite(a, "generator")?;
    zfinite(&modal.vectors, "basis")?;
    zfinite(&modal.inverse, "inverse")?;
    zfinite(&modal.eigenvalues, "eigenvalues")?;
    let mut weights = vec![0.; n];
    for j in 0..n {
        weights[j] = ((0..n)
            .map(|i| {
                metric[i]
                    * (modal.vectors[i + j * n].re.powi(2) + modal.vectors[i + j * n].im.powi(2))
            })
            .sum::<f64>()
            / (1. - gamma(8 * n + 8)))
        .sqrt()
            * (1. + 4. * U);
        if weights[j] == 0. {
            return Err("Zero modal basis column".into());
        }
    }
    let mut residual = vec![Wide::default(); n * n];
    let mut magnitude = vec![0.; n * n];
    let mut counts = vec![0; n];
    let columns: Vec<Vec<(usize, f64)>> = (0..n)
        .map(|k| {
            (0..n)
                .filter_map(|i| {
                    let value = a[i + k * n];
                    if value == 0. {
                        None
                    } else {
                        counts[i] += 1;
                        Some((i, value))
                    }
                })
                .collect()
        })
        .collect();
    // Column-major traversal keeps one compensated residual column hot. The
    // accumulation order in every entry is still the same increasing k.
    for j in 0..n {
        for (k, entries) in columns.iter().enumerate() {
            let v = modal.vectors[k + j * n];
            for &(i, value) in entries {
                residual[i + j * n].re.product(value, v.re);
                residual[i + j * n].im.product(value, v.im);
                magnitude[i + j * n] += value.abs() * abs1(v);
            }
        }
    }
    for j in 0..n {
        let lambda = modal.eigenvalues[j];
        for i in 0..n {
            let v = modal.vectors[i + j * n];
            residual[i + j * n].product(
                Z {
                    re: -lambda.re,
                    im: -lambda.im,
                },
                v,
            );
            magnitude[i + j * n] += abs1(lambda) * abs1(v);
        }
    }
    let mut r = Vec::with_capacity(n * n);
    let mut r_error = Vec::with_capacity(n * n);
    for j in 0..n {
        for i in 0..n {
            let k = i + j * n;
            let value = residual[k].value();
            r.push(value);
            let compensated = 64. * (counts[i] + 8) as f64 * U * U * magnitude[k]
                / (1. - gamma(4 * counts[i] + 32));
            r_error.push(8. * U * abs1(value) + compensated);
        }
    }
    drop(residual);
    drop(magnitude);
    zfinite(&r, "compensated residual")?;
    finite(&r_error, "residual roundoff envelope")?;
    let (defect, defect_magnitude) = complex_product(lap, &modal.inverse, &r, n);
    let inverse_abs: Vec<_> = modal.inverse.iter().map(|&z| abs1(z)).collect();
    let propagated = lap.multiply(&inverse_abs, &r_error, n, n, n);
    let defect_error: Vec<_> = propagated
        .iter()
        .zip(&defect_magnitude)
        .map(|(p, m)| p / (1. - gamma(n)) + 4. * gamma(n + 4) * m)
        .collect();
    let defect_abs: Vec<_> = defect.iter().map(|&z| modulus(z)).collect();
    let product_error = weighted_one(&defect_error, &weights, n);
    let defect_norm = weighted_one(&defect_abs, &weights, n);
    let (mut inverse_residual, inverse_magnitude) =
        complex_product(lap, &modal.inverse, &modal.vectors, n);
    for i in 0..n {
        inverse_residual[i + i * n].re -= 1.;
    }
    let inverse_error: Vec<_> = inverse_residual
        .iter()
        .zip(&inverse_magnitude)
        .map(|(&e, &m)| modulus(e) + 4. * gamma(n + 4) * m + 4. * U)
        .collect();
    let inverse_bound = weighted_one(&inverse_error, &weights, n);
    if !inverse_bound.is_finite() || inverse_bound >= 1. {
        return Err(format!(
            "Modal inverse cannot be verified: ||WV-I|| bound {inverse_bound:e}"
        ));
    }
    let uncertainty = (product_error + inverse_bound * defect_norm) / (1. - inverse_bound);
    let norm_upper = (defect_norm + product_error) / (1. - inverse_bound);
    let mut weighted_defect = vec![0.; n * n];
    let mut inverse_gap = vec![0.; n * n];
    for j in 0..n {
        for i in 0..n {
            let k = i + j * n;
            weighted_defect[k] = weights[i] * defect_abs[k] * (1. + 4. * U);
            let li = modal.eigenvalues[i];
            let lj = modal.eigenvalues[j];
            let gap = (li.re - lj.re).hypot(li.im - lj.im);
            let lower = (gap - 8. * U * (abs1(li) + abs1(lj))).max(0.);
            inverse_gap[k] = if lower > 0. {
                (1. + 8. * U) / lower
            } else {
                f64::INFINITY
            };
        }
    }
    let mut real_vectors = vec![0.; n * n];
    let mut j = 0;
    while j < n {
        let lambda = modal.eigenvalues[j];
        if lambda.im > 0. {
            if j + 1 >= n
                || modal.eigenvalues[j + 1].re != lambda.re
                || modal.eigenvalues[j + 1].im != -lambda.im
            {
                return Err("Incomplete conjugate modal pair".into());
            }
            for i in 0..n {
                let v = modal.vectors[i + j * n];
                let other = modal.vectors[i + (j + 1) * n];
                let w = modal.inverse[j + i * n];
                let w_other = modal.inverse[j + 1 + i * n];
                if v.re != other.re
                    || v.im != -other.im
                    || w.re != w_other.re
                    || w.im != -w_other.im
                {
                    return Err("Modal basis does not preserve exact real conjugate pairing".into());
                }
                real_vectors[i + j * n] = v.re;
                real_vectors[i + (j + 1) * n] = v.im;
            }
            j += 2;
        } else if lambda.im < 0. {
            return Err("Unpaired negative imaginary modal value".into());
        } else {
            for i in 0..n {
                if modal.vectors[i + j * n].im != 0. || modal.inverse[j + i * n].im != 0. {
                    return Err("Real mode has a complex coordinate vector".into());
                }
                real_vectors[i + j * n] = modal.vectors[i + j * n].re;
            }
            j += 1;
        }
    }
    // The four largest gaps in decay rates define numerical comparison
    // partitions, independent of requested T or a desired answer. Each one
    // covers every mode and every D entry; no model term is removed.
    let mut rates: Vec<_> = modal
        .eigenvalues
        .iter()
        .filter(|z| z.re < 0.)
        .map(|z| -z.re)
        .collect();
    rates.sort_by(f64::total_cmp);
    rates.dedup();
    let mut splits: Vec<_> = rates
        .windows(2)
        .filter_map(|w| {
            let ratio = w[1] / w[0];
            (ratio > 1. + 1e-12).then_some((ratio.ln(), (w[0].ln() * 0.5 + w[1].ln() * 0.5).exp()))
        })
        .collect();
    if modal.eigenvalues.iter().any(|z| z.re >= 0.) {
        if let Some(&minimum) = rates.first() {
            splits.push((f64::INFINITY, minimum * 0.5));
        }
    }
    splits.sort_by(|a, b| b.0.total_cmp(&a.0));
    splits.truncate(4);
    let mut comparisons = Vec::new();
    for (_, threshold) in splits {
        let groups: Vec<_> = modal
            .eigenvalues
            .iter()
            .map(|z| usize::from(z.re < -threshold))
            .collect();
        if !groups.contains(&0) || !groups.contains(&1) {
            continue;
        }
        let mut alpha = [f64::NEG_INFINITY; 2];
        for (j, z) in modal.eigenvalues.iter().enumerate() {
            alpha[groups[j]] = alpha[groups[j]].max(z.re);
        }
        let mut r = [0_f64; 4];
        for j in 0..n {
            let mut columns = [0.; 2];
            for i in 0..n {
                columns[groups[i]] += weighted_defect[i + j * n] / weights[j];
            }
            for i in 0..2 {
                r[i + groups[j] * 2] =
                    r[i + groups[j] * 2].max(columns[i] / (1. - gamma(4 * n + 8)));
            }
        }
        for value in &mut r {
            *value = (*value + uncertainty) * (1. + 8. * U);
        }
        // z0'=A0 z0; z1'=A0 z1+R z0; z2'=(A0+R)z2+R z1.
        // q'=max(0,alpha) q+1^T R(z1+z2) is monotone and bounds
        // ||z2(t)||_1 for EVERY t<=T, even after fast z2 modes have decayed.
        let mut generator = vec![0.; 49];
        for i in 0..2 {
            for block in 0..3 {
                generator[2 * block + i + (2 * block + i) * 7] = alpha[i];
            }
        }
        generator[6 + 6 * 7] = alpha[0].max(alpha[1]).max(0.);
        let add_upper = |target: &mut f64, value: f64| {
            let old = *target;
            *target = old + value + 4. * U * (old.abs() + value.abs());
        };
        for j in 0..2 {
            for i in 0..2 {
                let value = r[i + j * 2];
                add_upper(&mut generator[2 + i + j * 7], value);
                add_upper(&mut generator[4 + i + (2 + j) * 7], value);
                add_upper(&mut generator[4 + i + (4 + j) * 7], value);
                add_upper(&mut generator[6 + (2 + j) * 7], value);
                add_upper(&mut generator[6 + (4 + j) * 7], value);
            }
        }
        comparisons.push(Comparison {
            threshold,
            groups,
            generator,
        });
    }
    finite(&weighted_defect, "weighted modal defect")?;
    Ok(Prepared {
        diagnostics: Diagnostics {
            dimension: n,
            defect_norm_upper: norm_upper,
            defect_product_roundoff: product_error,
            inverse_residual_bound: inverse_bound,
            inverse_defect_uncertainty: uncertainty,
            preparation_ms: started.elapsed().as_secs_f64() * 1000.,
        },
        n,
        modal,
        metric: metric.to_vec(),
        embedding,
        weights,
        weighted_defect,
        inverse_gap,
        real_vectors,
        comparisons,
    })
}

impl Prepared {
    fn coefficients(&self, initial: &[f64]) -> Result<Vec<Z>, String> {
        if initial.len() != self.n {
            return Err("Wrong initial dimension for modal defect bound".into());
        }
        finite(initial, "initial state")?;
        let mut out = vec![Wide::default(); self.n];
        for (j, &x) in initial.iter().enumerate() {
            if x == 0. {
                continue;
            }
            for i in 0..self.n {
                let w = self.modal.inverse[i + j * self.n];
                out[i].re.product(w.re, x);
                out[i].im.product(w.im, x);
            }
        }
        let values: Vec<_> = out.into_iter().map(Wide::value).collect();
        zfinite(&values, "initial modal coefficients")?;
        Ok(values)
    }
    fn bound_with_coefficients(
        &self,
        lap: &Lapack,
        initial: &[f64],
        c: &[Z],
        time: f64,
    ) -> Result<Bound, String> {
        if !time.is_finite() || time < 0. {
            return Err("Invalid time for modal defect bound".into());
        }
        let n = self.n;
        let mut initial_squared = 0.;
        for i in 0..n {
            let mut residual = Wide::default();
            let mut magnitude = initial[i].abs();
            for j in 0..n {
                let v = self.modal.vectors[i + j * n];
                residual.product(v, c[j]);
                magnitude += abs1(v) * abs1(c[j]);
            }
            residual.re.add(-initial[i], 0.);
            let value = residual.value();
            let error = modulus(value)
                + 8. * U * abs1(value)
                + 64. * (n + 8) as f64 * U * U * magnitude / (1. - gamma(8 * n + 32));
            initial_squared += self.metric[i] * error * error;
        }
        let initial_bound = self.embedding * (initial_squared / (1. - gamma(4 * n + 8))).sqrt();
        let mut growth = Vec::with_capacity(n);
        let mut time_growth = Vec::with_capacity(n);
        let mut coefficient_norm = 0.;
        let mut floating = 0.;
        for j in 0..n {
            let lambda = self.modal.eigenvalues[j];
            let g = exponential(lambda.re.max(0.) * time);
            let peak = if lambda.re < 0. {
                time.min(-1. / lambda.re)
            } else {
                time
            };
            let tg = peak * exponential(lambda.re * peak);
            growth.push(g);
            time_growth.push(tg);
            let amount = self.weights[j] * modulus(c[j]);
            coefficient_norm += amount;
            floating += amount
                * ((gamma(n + 8) + 64. * U) * g
                    + 2. * U * (lambda.re.abs() + lambda.im.abs()) * tg);
        }
        coefficient_norm /= 1. - gamma(4 * n + 8);
        let mut first = 0.;
        for j in 0..n {
            let coefficient = modulus(c[j]);
            if coefficient == 0. {
                continue;
            }
            let mut column = 0.;
            for i in 0..n {
                let entry = self.weighted_defect[i + j * n];
                if entry == 0. {
                    continue;
                }
                let kernel = time_growth[i]
                    .max(time_growth[j])
                    .min((growth[i] + growth[j]) * self.inverse_gap[i + j * n]);
                column += entry * kernel;
            }
            first += coefficient * column / (1. - gamma(4 * n + 8));
        }
        first *= self.embedding / (1. - gamma(4 * n + 8));
        let alpha = self
            .modal
            .eigenvalues
            .iter()
            .map(|v| v.re)
            .fold(f64::NEG_INFINITY, f64::max);
        let envelope = exponential(alpha.max(0.) * time);
        let uncertain = self.embedding
            * coefficient_norm
            * envelope
            * time
            * self.diagnostics.inverse_defect_uncertainty;
        let global_higher = self.embedding
            * coefficient_norm
            * envelope
            * remainder(self.diagnostics.defect_norm_upper * time);
        let mut higher = global_higher;
        let mut comparison_threshold = None;
        let mut comparison_roundoff = 0.;
        for comparison in &self.comparisons {
            let mut initial = vec![0.; 7];
            for j in 0..n {
                initial[comparison.groups[j]] += self.weights[j] * modulus(c[j]);
            }
            for value in &mut initial {
                *value /= 1. - gamma(4 * n + 8);
            }
            if let Ok(action) =
                dense_expm::action_with_diagnostics(lap, &comparison.generator, &initial, &[time])
            {
                if let Some(risk) = action.steps[0].action_roundoff_risk_estimate {
                    if risk.is_finite() && action.steps[0].linear_solve_backward_error < 1e-10 {
                        let candidate =
                            self.embedding * (action.states[0][6].max(0.) + risk) * (1. + 32. * U);
                        if candidate < higher {
                            higher = candidate;
                            comparison_threshold = Some(comparison.threshold);
                            comparison_roundoff = self.embedding * risk;
                        }
                    }
                }
            }
        }
        let floating = self.embedding * floating / (1. - gamma(4 * n + 8));
        let total = (initial_bound + first + uncertain + higher + floating) * (1. + 32. * U);
        Ok(Bound {
            time,
            initial_reconstruction: initial_bound,
            first_order: first,
            higher_order: higher,
            inverse_and_product_uncertainty: uncertain,
            floating_evaluation: floating,
            total,
            coefficient_weighted_norm: coefficient_norm,
            global_higher_order: global_higher,
            comparison_threshold,
            comparison_roundoff,
        })
    }
    pub fn bound(&self, lap: &Lapack, initial: &[f64], time: f64) -> Result<Bound, String> {
        let c = self.coefficients(initial)?;
        self.bound_with_coefficients(lap, initial, &c, time)
    }
    pub fn apply(&self, lap: &Lapack, initial: &[f64], times: &[f64]) -> Result<Action, String> {
        let started = Instant::now();
        finite(times, "times")?;
        if times.iter().any(|t| *t < 0.) {
            return Err("Negative modal propagation time".into());
        }
        let c = self.coefficients(initial)?;
        let bound = self.bound_with_coefficients(
            lap,
            initial,
            &c,
            times.iter().copied().fold(0., f64::max),
        )?;
        let n = self.n;
        let mut modes = vec![0.; n * times.len()];
        for (ti, &t) in times.iter().enumerate() {
            let mut j = 0;
            while j < n {
                let lambda = self.modal.eigenvalues[j];
                if lambda.im > 0. {
                    if c[j].re != 0. || c[j].im != 0. {
                        let (s, cs) = (lambda.im * t).sin_cos();
                        let e = (lambda.re * t).exp();
                        modes[j + ti * n] = 2. * e * (c[j].re * cs - c[j].im * s);
                        modes[j + 1 + ti * n] = -2. * e * (c[j].re * s + c[j].im * cs);
                    }
                    j += 2;
                } else {
                    if c[j].re != 0. {
                        modes[j + ti * n] = c[j].re * (lambda.re * t).exp();
                    }
                    j += 1;
                }
            }
        }
        finite(&modes, "modal exponential")?;
        let all = lap.multiply(&self.real_vectors, &modes, n, times.len(), n);
        finite(&all, "propagated state")?;
        // Keep t=0's numerical reconstruction visible; do not repair it.
        let states = all.chunks(n).map(|x| x.to_vec()).collect();
        Ok(Action {
            states,
            bound,
            elapsed_ms: started.elapsed().as_secs_f64() * 1000.,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn missing_a_slow_decay_is_detected_by_the_full_defect() {
        let lap = Lapack::load().unwrap();
        let g = 1e-8;
        let a = vec![-g, g, 0., 0.];
        let z = Z::default();
        let one = Z { re: 1., im: 0. };
        let modal = ModalBasis {
            dimension: 2,
            eigenvalues: vec![z; 2],
            vectors: vec![one, z, z, one],
            inverse: vec![one, z, z, one],
            condition: 1.,
            origin: vec![super::super::cluster_action::ModeOrigin::RefinedCluster; 2],
        };
        let prepared = prepare(&lap, &a, modal, &[1., 1.], 2_f64.sqrt()).unwrap();
        let report = prepared.apply(&lap, &[1., 0.], &[0., 1e6]).unwrap();
        let true_error = 2. * (1. - (-g * 1e6).exp());
        assert!(report.bound.total >= true_error);
        assert!(report.bound.total > 5e-6);
    }
    #[test]
    fn defective_remainder_series_retains_tiny_positive_terms() {
        for x in [1e-12, 1e-6, 0.1, 0.5, 1.] {
            assert!(remainder(x) >= x * x * 0.5);
            assert!(remainder(x).is_finite());
        }
    }
    #[test]
    fn comparison_remainder_covers_early_error_after_endpoint_has_relaxed() {
        let lap = Lapack::load().unwrap();
        let z = Z::default();
        let one = Z { re: 1., im: 0. };
        let minus = Z { re: -1., im: 0. };
        // True gamma=.1, intentionally bad modal gamma=1. By T=100 both
        // approach the same ground state, but their earlier error is large.
        let modal = ModalBasis {
            dimension: 2,
            eigenvalues: vec![minus, z],
            vectors: vec![one, minus, z, one],
            inverse: vec![one, one, z, one],
            condition: 4.,
            origin: vec![super::super::cluster_action::ModeOrigin::RefinedCluster; 2],
        };
        let prepared = prepare(&lap, &[-0.1, 0.1, 0., 0.], modal, &[1., 1.], 2_f64.sqrt()).unwrap();
        let result = prepared
            .apply(&lap, &[1., 0.], &[0., 1., 2., 3., 10., 100.])
            .unwrap();
        let peak = 10_f64.ln() / 0.9;
        let true_peak = 2. * ((-0.1 * peak).exp() - (-peak).exp());
        assert!(result.bound.comparison_threshold.is_some());
        assert!(result.bound.total >= true_peak);
        assert!(result.bound.total < 100.);
        assert!(result.bound.global_higher_order > 1e30);
    }
}
