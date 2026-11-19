//! Profile-likelihood intervals of a fit whose likelihood fixes the noise
//! (Poisson, binomial; MINUIT's MINOS).
//!
//! For a free parameter `j` the 68.27 % interval is the set of values whose
//! profile deviance -- the deviance minimised over the other free
//! parameters with `theta_j` held -- rises at most 1 above the minimum.  A
//! curvature (Fisher) error describes it only where the profile is a
//! parabola; on the edge of the likelihood's domain (a proportion fitted at
//! 0 or 1) the profile is linear and only the profile itself has the
//! interval.  A side whose profile has not risen by 1 where the solver bound
//! or the likelihood's support ends (a model value leaves `[0, 1]` for a
//! proportion, goes negative for a count: `Likelihood::supports`, where the
//! deviance is infinite) ends there: it is one-sided.
//!
//! Each side is found by constrained refits (the cell's own trust-region
//! solve with `theta_j` held), each warm from the nearest one inside:
//! stepping out from the MLE by the Fisher error (the parameter's own scale
//! where that is not finite) until the rise reaches 1, then a safeguarded
//! secant on `sqrt(rise)` (linear in `theta_j` for a parabolic profile) to
//! `SIDE_TOLERANCE` of the side.

use super::num::{pmax, pmin, EPSILON};
use super::trf::{self, CellObjective, STATUS_INVALID};

/// The deviance rise bounding a one-parameter 68.27 % interval (the 1-sigma
/// quantile of chi-square with one degree of freedom).
pub const PROFILE_RISE: f64 = 1.0;

/// A side is settled to this fraction of its length (an approximate
/// convergence: the interval is an error bar).
pub const SIDE_TOLERANCE: f64 = 1e-3;

/// Steps out at least double, so this many of them leave the float's
/// resolution of the first step behind: a profile that has not risen by
/// then is unbounded.
const MAXIMUM_STEPS: u32 = f64::MANTISSA_DIGITS;

/// One fitted cell, profiled.
pub struct Profile<'a> {
    /// The cell's objective: its cost is half the deviance (linear loss).
    pub objective: &'a dyn CellObjective,
    /// The maximum-likelihood parameters (all of them).
    pub parameters: &'a [f64],
    pub free: &'a [usize],
    /// The solver bounds, all parameters.
    pub lower: &'a [f64],
    pub upper: &'a [f64],
    /// The evaluation budget of one refit.
    pub budget: i64,
}

/// A profiled point: its distance from the MLE, its deviance rise (`None`:
/// outside the likelihood's support or not solvable), and the other free
/// parameters' values there.
struct Point {
    distance: f64,
    rise: Option<f64>,
    others: Vec<f64>,
}

impl Profile<'_> {
    /// Twice the objective's cost at `params`.
    fn deviance(&self, params: &[f64]) -> Option<f64> {
        let (cost, _rss, finite) = self.objective.evaluate(params, &[], &mut [], &mut [], &mut [], false).ok()?;
        (finite && cost.is_finite()).then_some(2.0 * cost)
    }

    /// The profile deviance at `theta_j = value`, the others solved from
    /// `seed`; `None` where no model in the likelihood's support holds it.
    fn profiled(&self, j: usize, value: f64, others: &[usize], seed: &[f64]) -> Option<(f64, Vec<f64>)> {
        let mut template = self.parameters.to_vec();
        template[j] = value;
        if others.is_empty() {
            return self.deviance(&template).map(|d| (d, Vec::new()));
        }
        let lower: Vec<f64> = others.iter().map(|&i| self.lower[i]).collect();
        let upper: Vec<f64> = others.iter().map(|&i| self.upper[i]).collect();
        let outcome = trf::solve_seed(self.objective, &template, others, seed, &lower, &upper, self.budget, trf::TOLERANCE, trf::TOLERANCE, trf::TOLERANCE).ok()?;
        (outcome.status != STATUS_INVALID && outcome.cost.is_finite()).then_some((2.0 * outcome.cost, outcome.values))
    }

    /// The 68.27 % interval `[low, high]` of every parameter: a fixed one's
    /// is its value; one whose `error` (Fisher) is NaN -- undetermined --, or
    /// whose profile never rises by 1 inside an unbounded range, is NaN.
    pub fn intervals(&self, errors: &[f64]) -> Vec<[f64; 2]> {
        let mut intervals: Vec<[f64; 2]> = self.parameters.iter().map(|&v| [v, v]).collect();
        let Some(minimum) = self.deviance(self.parameters) else {
            for &j in self.free {
                intervals[j] = [f64::NAN; 2];
            }
            return intervals;
        };
        for &j in self.free {
            if errors[j].is_nan() {
                intervals[j] = [f64::NAN; 2];
                continue;
            }
            let others: Vec<usize> = self.free.iter().copied().filter(|&i| i != j).collect();
            let seed: Vec<f64> = others.iter().map(|&i| self.parameters[i]).collect();
            let low = self.side(j, -1.0, &others, &seed, minimum, errors[j]);
            let high = self.side(j, 1.0, &others, &seed, minimum, errors[j]);
            intervals[j] = if low.is_finite() && high.is_finite() { [low, high] } else { [f64::NAN; 2] };
        }
        intervals
    }

    /// The end of parameter `j`'s interval in `direction` (+1 or -1).
    fn side(&self, j: usize, direction: f64, others: &[usize], seed: &[f64], minimum: f64, error: f64) -> f64 {
        let value = self.parameters[j];
        let bound = if direction > 0.0 { self.upper[j] } else { self.lower[j] };
        let room = (bound - value).abs();
        if !(room > 0.0) {
            return value;
        }
        let at = |distance: f64| value + direction * distance;
        let evaluate = |distance: f64, seed: &[f64]| -> Point {
            match self.profiled(j, at(distance), others, seed) {
                Some((deviance, others)) => Point { distance, rise: Some(deviance - minimum), others },
                None => Point { distance, rise: None, others: Vec::new() },
            }
        };
        let first = if error.is_finite() && error > 0.0 { error } else if value != 0.0 { value.abs() } else { 1.0 };
        // Step out until the rise reaches PROFILE_RISE, the support ends or the bound is met.
        let mut inner = Point { distance: 0.0, rise: Some(0.0), others: seed.to_vec() };
        let mut step = pmin(first, room);
        let mut outer = None;
        for _ in 0..MAXIMUM_STEPS {
            let point = evaluate(step, &inner.others);
            match point.rise {
                Some(rise) if rise < PROFILE_RISE => {
                    if step >= room {
                        return bound;
                    }
                    // A parabola would reach the rise at step sqrt(R / rise); at least double.
                    let growth = if rise > 0.0 { (PROFILE_RISE / rise).sqrt() } else { f64::INFINITY };
                    inner = point;
                    step = pmin(step * pmax(growth, 2.0), room);
                }
                _ => {
                    outer = Some(point);
                    break;
                }
            }
        }
        let Some(mut outer) = outer else { return direction * f64::INFINITY };
        // Narrow [inner, outer]: secant on sqrt(rise), every other step a bisection.
        let tolerance = SIDE_TOLERANCE * pmax(outer.distance, first) + EPSILON * value.abs();
        let estimate = |inner: &Point, outer: &Point| -> f64 {
            match (inner.rise, outer.rise) {
                (Some(a), Some(b)) => {
                    let (a, b) = (pmax(a, 0.0).sqrt(), b.sqrt());
                    inner.distance + (PROFILE_RISE.sqrt() - a) * (outer.distance - inner.distance) / (b - a)
                }
                _ => inner.distance,
            }
        };
        let mut bisect = false;
        for _ in 0..2 * MAXIMUM_STEPS {
            if outer.distance - inner.distance <= tolerance {
                break;
            }
            let secant = estimate(&inner, &outer);
            let middle = 0.5 * (inner.distance + outer.distance);
            let distance = if bisect || outer.rise.is_none() || !(secant > inner.distance && secant < outer.distance) { middle } else { secant };
            bisect = !bisect;
            let point = evaluate(distance, &inner.others);
            match point.rise {
                Some(rise) if rise < PROFILE_RISE => inner = point,
                _ => outer = point,
            }
        }
        at(estimate(&inner, &outer))
    }
}

/// The covariance and errors of a likelihood fit from its intervals: each
/// error the half-width `(high - low) / 2` (the symmetric standard
/// uncertainty of a 68 % interval), the covariance `D R D` with `R` the
/// Fisher correlation and `D` those half-widths, so its diagonal is the
/// errors squared.  A parameter without an interval (NaN) is undetermined:
/// NaN error, NaN row and column.  `count`: parameters; row-major.
pub fn interval_covariance(covariance: &mut [f64], errors: &mut [f64], intervals: &[[f64; 2]], free: &[usize]) {
    let count = errors.len();
    let half: Vec<f64> = intervals.iter().map(|[low, high]| 0.5 * (high - low)).collect();
    let fisher = covariance.to_vec();
    for &a in free {
        for &b in free {
            let scale = (fisher[a * count + a] * fisher[b * count + b]).sqrt();
            covariance[a * count + b] = if half[a].is_finite() && half[b].is_finite() { fisher[a * count + b] / scale * half[a] * half[b] } else { f64::NAN };
            if a == b {
                covariance[a * count + a] = half[a] * half[a];
            }
        }
        errors[a] = half[a];
    }
}
