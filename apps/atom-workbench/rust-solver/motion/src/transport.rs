//! The transport of the step's local energy errors to an output (SPEC §2.2, # The read).
//!
//! # What is transported
//! A read trajectory (adapt.rs) accumulates per lab axis the local errors its steps make of that axis's energy, and the
//! energy its jumps put in. A local error made at the time t_j is an error of T at a later output t_K only as far as
//! the motion remembers it: a trap without scattering keeps an atom's energy, and what a step took from it stays
//! taken; a cooled motion forgets an atom's energy at its cooling rate, and with it the error.
//!
//! # The regression
//! To first order in the local errors ℓ_n, the step's error of an ensemble mean E[h(X_{t_K})] is the telescoping sum
//! Σ_n E[∇u(t_n, X_n)·ℓ_n], u(t, x) = E[h(X_{t_K}) | X_t = x] the exact dynamics' expectation from the state x. The
//! motion's slow variable is the atom's energy E (the loss energy E_s: defined everywhere, kept exactly by a trap
//! without scattering). With u linear in it over the ensemble's spread, its slope is the ensemble's own least-squares
//! regression of the energy at t_K on that at t_j,
//!   b(K, j) = Cov(E_j, E_K) / Var(E_j),
//! and an axis's sum is Σ_j b(K, j) δ_j, δ_j the ensemble's mean local error of that axis made over the output interval
//! ending at t_j. This is Onsager's regression as the ensemble measures it: a disturbance of the mean relaxes as the
//! atoms' own fluctuations regress. An energy that is kept regresses on itself exactly (b = 1, with no statistical
//! error): the local errors add up as they are made, the plain sum, in which a bounded trap error's alternating parts
//! cancel as they do along a trajectory. An energy that is forgotten (b → 0) lets them go. Where the ensemble's
//! energies do not differ the regression measures nothing: the error is carried unchanged, the safe side.
//!
//! One kernel carries the three axes' errors. Axes that exchange energy (a round trap's two radial ones) then keep
//! their own sums where the atoms share them; the largest axis's read is no smaller for it, and that is the one the
//! step is held to. (A regression of the three axes' energies on each other would follow the exchange, but its nine
//! entries' noise multiplies every interval's error and undoes the cancellation above: measured on the reference traps
//! R1, an interval of 5 % at a step whose plain sum reads 0.5 %.)
//!
//! # The local error of an interval
//! δ_j is the mean of the errors accumulated over the interval -- the doubling's differences and the kick the carried
//! internal state changes (adapt.rs) -- and of the energy the jumps put in over it times the hazard's relative error
//! there, (Σ its accumulated error)/(Σ its integral): a jump's recoil is not in the jump-free doubling, and a hazard that
//! errs by a share makes that share too many or too few of them.
//!
//! # The cut
//! The lags K − j are added in pairs from the output backwards while a pair's regressions add up positive: Geyer's
//! initial positive sequence (Statist. Sci. 7 (1992) 473) -- beyond the first pair that does not, an estimated
//! regression is its noise, and adding it would only add that noise times all the errors made before.
//!
//! # The error of the read
//! The delete-one-group jackknife of the whole estimate, as every other estimate's (estimate.rs): the regressions, the
//! local errors, the hazard's ratio and the cut made anew without each group (the caller divides each by T's scale of
//! the same trajectories). With the cut each replicate's own, the jackknife carries the cut's chance too.

use crate::linalg::V3;
use crate::units::UNIT_ROUNDOFF;

/// What the transport takes of one read trajectory at one output (adapt.rs): its energy (the loss energy E_s,
/// rad/µs); per lab axis the accumulated local errors of the axis's energy and the accumulated energy its jumps put in;
/// the hazard's integral ∫γ dt along it and that integral's accumulated local error.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mark {
    pub level: f64,
    pub error: V3,
    pub input: V3,
    pub hazard: f64,
    pub hazard_error: f64,
}

impl Mark {
    pub const ZERO: Mark = Mark { level: 0.0, error: [0.0; 3], input: [0.0; 3], hazard: 0.0, hazard_error: 0.0 };
}

/// One lag's sums over some trajectories: their number; the energies at the lag's time and at the output, each less a
/// fixed centre, and their products; the interval's local errors, the jumps' input, the hazard and its error.
#[derive(Clone, Copy, Default)]
struct Sums {
    n: f64,
    from: f64,
    to: f64,
    within: f64,
    across: f64,
    error: V3,
    input: V3,
    hazard: f64,
    hazard_error: f64,
}

impl Sums {
    fn add(&mut self, from: f64, to: f64, error: V3, input: V3, hazard: f64, hazard_error: f64) {
        self.n += 1.0;
        self.from += from;
        self.to += to;
        self.within += from * from;
        self.across += from * to;
        for x in 0..3 {
            self.error[x] += error[x];
            self.input[x] += input[x];
        }
        self.hazard += hazard;
        self.hazard_error += hazard_error;
    }

    /// These sums and `sign` times others.
    fn with(&self, o: &Sums, sign: f64) -> Sums {
        Sums {
            n: self.n + sign * o.n,
            from: self.from + sign * o.from,
            to: self.to + sign * o.to,
            within: self.within + sign * o.within,
            across: self.across + sign * o.across,
            error: std::array::from_fn(|x| self.error[x] + sign * o.error[x]),
            input: std::array::from_fn(|x| self.input[x] + sign * o.input[x]),
            hazard: self.hazard + sign * o.hazard,
            hazard_error: self.hazard_error + sign * o.hazard_error,
        }
    }

    /// The lag's regression b(K, j) and the interval's local error per axis δ_j (module documentation, # The
    /// regression, # The local error of an interval). `scale` is the energies' size at the lag's time (the centre's
    /// modulus): a variance below the round-off of the sums of that size is none, and the error is carried unchanged.
    fn regression(&self, scale: f64) -> (f64, V3) {
        let n = self.n;
        let (variance, covariance) = (self.within / n - (self.from / n).powi(2), self.across / n - self.from / n * self.to / n);
        // The n values' rounded mean leaves each deviation n u of the size, a variance of that squared.
        let b = if variance > (n * UNIT_ROUNDOFF * scale).powi(2) { covariance / variance } else { 1.0 };
        let share = if self.hazard > 0.0 { self.hazard_error / self.hazard } else { 0.0 };
        (b, std::array::from_fn(|x| (self.error[x] + share * self.input[x]) / n))
    }
}

/// The step's local errors of the three lab axes' energies carried to an output (rad/µs): of all the read trajectories
/// held there, and of those without each group in turn (a group that holds none leaves the estimate as it is).
#[derive(Clone, Debug, PartialEq)]
pub struct Transported {
    pub all: V3,
    pub without: Vec<V3>,
}

/// The transported errors at the output `output` from the read trajectories' marks (module documentation):
/// `series[i][k]` is trajectory i's mark at output k while it is held, the trajectories in groups of `group` in their
/// order. None where fewer than two groups hold an atom at the output or a group left out leaves fewer than two atoms:
/// no read.
pub fn transported<S: AsRef<[Option<Mark>]>>(series: &[S], group: usize, output: usize) -> Option<Transported> {
    let series: Vec<&[Option<Mark>]> = series.iter().map(AsRef::as_ref).collect();
    let held: Vec<usize> = (0..series.len()).filter(|&i| series[i][output].is_some()).collect();
    let groups = series.len().div_ceil(group);
    let mark = |i: usize, k: usize| series[i][k].expect("a trajectory held at an output is held at every earlier one");
    let members: Vec<f64> = (0..groups).map(|g| held.iter().filter(|&&i| i / group == g).count() as f64).collect();
    let all = held.len() as f64;
    if members.iter().filter(|&&m| m > 0.0).count() < 2 || members.iter().any(|&m| m > 0.0 && all - m < 2.0) {
        return None;
    }
    let centre = |k: usize| held.iter().map(|&i| mark(i, k).level).sum::<f64>() / all;
    let to = centre(output);
    // The lag's sums per group, the energies less the lag's and the output's centres.
    let lag = |j: usize| -> (Vec<Sums>, f64) {
        let from = centre(j);
        let mut sums = vec![Sums::default(); groups];
        for &i in &held {
            let (m, before) = (mark(i, j), mark(i, j - 1));
            sums[i / group].add(
                m.level - from,
                mark(i, output).level - to,
                std::array::from_fn(|a| m.error[a] - before.error[a]),
                std::array::from_fn(|a| m.input[a] - before.input[a]),
                m.hazard - before.hazard,
                m.hazard_error - before.hazard_error,
            );
        }
        (sums, from.abs())
    };
    // The estimate of all the trajectories (0) and of those without each group, lag pair by lag pair, each until its
    // own cut.
    let mut totals = vec![[0.0; 3]; 1 + groups];
    let mut open = vec![true; totals.len()];
    let mut j = output;
    while j >= 1 && open.iter().any(|&o| o) {
        let pair: Vec<usize> = [Some(j), j.checked_sub(1).filter(|&k| k >= 1)].into_iter().flatten().collect();
        let mut added = vec![([0.0; 3], 0.0); totals.len()];
        for &k in &pair {
            let (sums, scale) = lag(k);
            let whole = sums.iter().fold(Sums::default(), |a, s| a.with(s, 1.0));
            for (r, slot) in added.iter_mut().enumerate().filter(|(r, _)| open[*r]) {
                let (b, local) = if r == 0 { whole.regression(scale) } else { whole.with(&sums[r - 1], -1.0).regression(scale) };
                for a in 0..3 {
                    slot.0[a] += b * local[a];
                }
                slot.1 += b;
            }
        }
        for r in 0..totals.len() {
            open[r] = open[r] && added[r].1 > 0.0;
            if open[r] {
                for a in 0..3 {
                    totals[r][a] += added[r].0[a];
                }
            }
        }
        j = match j.checked_sub(pair.len()) {
            Some(next) => next,
            None => break,
        };
    }
    Some(Transported { all: totals[0], without: totals[1..].to_vec() })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimate::spread;
    use crate::random::{Purpose, Stream};

    const GROUP: usize = 8;

    /// Trajectories of marks from a rule: `next(stream, level) -> level` over each interval, the local error `error` an
    /// interval per axis, no jumps.
    fn series(trajectories: usize, outputs: usize, start: impl Fn(&mut Stream) -> f64, next: impl Fn(&mut Stream, f64) -> f64, error: V3) -> Vec<Vec<Option<Mark>>> {
        (0..trajectories)
            .map(|i| {
                let mut s = Stream::new(0, Purpose::Initial, i as u64, 0);
                let mut level = start(&mut s);
                (0..=outputs)
                    .map(|k| {
                        if k > 0 {
                            level = next(&mut s, level);
                        }
                        Some(Mark { level, error: error.map(|e| e * k as f64), input: [0.0; 3], hazard: 0.0, hazard_error: 0.0 })
                    })
                    .collect()
            })
            .collect()
    }

    /// An exponential deviate of mean 1: a thermal oscillator's energy over k_BT.
    fn thermal(s: &mut Stream) -> f64 {
        -(1.0 - s.uniform()).ln()
    }

    /// The jackknife's error of an axis's transported error.
    fn error(t: &Transported, a: usize) -> f64 {
        spread(t.without.iter().map(|w| w[a]), t.without.len() as f64).error
    }

    /// Energies that are kept: every interval's error arrives whole, the plain sum, with no statistical error -- the
    /// regression of a kept energy on itself is exactly 1.
    #[test]
    fn kept_energies_carry_every_error() {
        let (outputs, local) = (40, [1e-3, -2e-3, 5e-4]);
        let s = series(88, outputs, thermal, |_, level| level, local);
        let read = transported(&s, GROUP, outputs).unwrap();
        for a in 0..3 {
            let plain = local[a] * outputs as f64;
            assert!((read.all[a] - plain).abs() <= 1e-12 * plain.abs(), "axis {a}: {} against the plain sum {plain}", read.all[a]);
            assert!(error(&read, a) <= 1e-12 * plain.abs(), "axis {a}: a statistical error {} of a kept energy's sum", error(&read, a));
        }
    }

    /// Energies that relax: E ← ρE + (1 − ρ)·(a fresh thermal energy) over an interval regresses by ρ a lag, and a
    /// constant local error δ an interval arrives as δ(1 + ρ + ρ² + …) = δ/(1 − ρ) on every axis by its own δ -- not as
    /// the plain sum δK --, each within the family's interval of its jackknife (decide.rs). The cut drops the lags that
    /// only bring noise: uncut, the far lags' noise would add up as the run's length, the error as its square root.
    #[test]
    fn relaxing_energies_forget_their_errors() {
        let (rho, delta) = (0.6, [1e-3, 2e-3, -1e-3]);
        let relax = |s: &mut Stream, level: f64| rho * level + (1.0 - rho) * thermal(s);
        let mut errors = vec![];
        for outputs in [50, 400] {
            let s = series(704, outputs, thermal, relax, delta);
            let read = transported(&s, GROUP, outputs).unwrap();
            for a in 0..3 {
                let expected = delta[a] / (1.0 - rho);
                let interval = crate::estimate::Spread { error: error(&read, a), groups: read.without.len() as f64 }.interval_over(6.0);
                assert!((read.all[a] - expected).abs() <= interval, "axis {a} at {outputs} outputs: {} ± {interval} against {expected}", read.all[a]);
            }
            errors.push(error(&read, 0));
        }
        assert!(errors[1] < (400.0_f64 / 50.0).sqrt() * errors[0], "the error grew with the run as an uncut sum's: {errors:?}");
    }

    /// Atoms that all have the same, kept energy: nothing to regress on, the errors carried unchanged.
    #[test]
    fn energies_without_spread_carry_their_errors() {
        let (outputs, delta) = (20, 1e-3);
        let s = series(88, outputs, |_| 2.0, |_, level| level, [0.0, delta, 0.0]);
        let read = transported(&s, GROUP, outputs).unwrap();
        assert!((read.all[1] - delta * outputs as f64).abs() <= 1e-15 && read.all[0] == 0.0, "{:?}", read.all);
    }

    /// The jumps' input times the hazard's relative error is a local error like the doubling's.
    #[test]
    fn a_hazard_s_error_scales_the_jumps_input() {
        let outputs = 10;
        let mut s = series(88, outputs, thermal, |_, level| level, [0.0; 3]);
        for t in &mut s {
            for (k, m) in t.iter_mut().enumerate() {
                let m = m.as_mut().unwrap();
                (m.input, m.hazard, m.hazard_error) = ([0.02 * k as f64, 0.0, 0.0], 3.0 * k as f64, 0.15 * k as f64);
            }
        }
        let read = transported(&s, GROUP, outputs).unwrap();
        let expected = 0.05 * 0.02 * outputs as f64;
        assert!((read.all[0] - expected).abs() <= 1e-12 * expected && read.all[1] == 0.0, "{:?}", read.all);
    }

    /// Fewer than two groups holding an atom at the output read nothing; a group that holds none leaves the estimate
    /// as it is.
    #[test]
    fn one_group_reads_nothing() {
        let mut s = series(24, 5, thermal, |_, level| level, [1e-3; 3]);
        for t in s.iter_mut().skip(8) {
            t[5] = None;
        }
        assert!(transported(&s, GROUP, 5).is_none());
        for t in s.iter_mut().skip(8).take(8) {
            t[4] = None;
        }
        let read = transported(&s, GROUP, 4).unwrap();
        assert_eq!(read.without[1], read.all);
    }
}
