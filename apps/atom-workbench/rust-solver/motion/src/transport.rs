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
//! motion's slow variables are the three axes' energies E (adapt.rs `Oscillator`: the harmonic energies about the
//! trap the reference follows, a smooth function of the state everywhere). With u linear in them over the ensemble's
//! spread, its slope is the ensemble's own least-squares regression of the energies at t_K on those at t_j,
//!   β(K, j) = Var(E_j)⁻¹ Cov(E_j, E_K)   (3 × 3; row: the axis at t_j, column: the axis at t_K),
//! and the sum is Σ_j β(K, j)ᵀ δ_j, δ_j the ensemble's mean local error made over the output interval ending at t_j.
//! This is Onsager's regression as the ensemble measures it: a disturbance of the mean relaxes as the atoms' own
//! fluctuations regress. Energies that are kept (β = 1) add up as they are made, the plain sum; energies exchanged
//! between axes (a round trap's two radial ones) move by β's off-diagonal entries, their sum kept -- each axis's read
//! is then the two axes' mean, which is what its T becomes, and the two axes' opposite chance parts cancel in it;
//! forgotten ones (β → 0) leave. Along a direction in which the ensemble's energies do not differ the regression
//! measures nothing: an error is carried there unchanged, the safe side, and the exact answer where every atom's
//! energy is the same and kept.
//!
//! # The local error of an interval
//! δ_j is the mean of the errors accumulated over the interval -- the doubling's differences and the kick the carried
//! internal state changes (adapt.rs) -- and of the energy the jumps put in over it times the hazard's relative error
//! there, (Σ its accumulated error)/(Σ its integral): a jump's recoil is not in the jump-free doubling, and a hazard that
//! errs by a share makes that share too many or too few of them.
//!
//! # The cut
//! The lags K − j are added in pairs from the output backwards while a pair's regressions are positive in the mean of
//! their column sums (the share of an energy error still in the three energies): Geyer's initial positive sequence
//! (Statist. Sci. 7 (1992) 473) -- beyond the first pair that is not, an estimated regression is its noise, and adding
//! it would only add that noise times all the errors made before.
//!
//! # The error of the read
//! The delete-one-group jackknife of the whole estimate, as every other estimate's (estimate.rs): the regressions, the
//! local errors, the hazard's ratio and the cut made anew without each group (the caller divides each by T's scale of
//! the same trajectories). With the cut each replicate's own, the jackknife carries the cut's chance too: over 40 seeds
//! of a relaxing ensemble (regression 0.6 a lag, 50 and 400 outputs, 88 and 704 trajectories) the read's deviation over
//! its jackknife error has the standard deviation 0.84 to 0.93 and the read's mean is 1.02 to 1.09 of the exact sum (the
//! cut keeps the positive chance pairs); with one cut for all replicates that standard deviation was 1.2 at 88
//! trajectories.

use crate::linalg::{symmetric3_eigen, M3, V3};
use crate::units::UNIT_ROUNDOFF;

/// What the transport takes of one read trajectory at one output (adapt.rs): per lab axis its energy (rad/µs), the
/// accumulated local errors of it and the accumulated energy its jumps put in; the hazard's integral ∫γ dt along it
/// and that integral's accumulated local error.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mark {
    pub level: V3,
    pub error: V3,
    pub input: V3,
    pub hazard: f64,
    pub hazard_error: f64,
}

impl Mark {
    pub const ZERO: Mark = Mark { level: [0.0; 3], error: [0.0; 3], input: [0.0; 3], hazard: 0.0, hazard_error: 0.0 };
}

/// One lag's sums over some trajectories: their number; the energies at the lag's time and at the output, each less a
/// fixed centre, and their products; the interval's local errors, the jumps' input, the hazard and its error.
#[derive(Clone, Copy, Default)]
struct Sums {
    n: f64,
    from: V3,
    to: V3,
    within: M3,
    across: M3,
    error: V3,
    input: V3,
    hazard: f64,
    hazard_error: f64,
}

impl Sums {
    fn add(&mut self, from: V3, to: V3, error: V3, input: V3, hazard: f64, hazard_error: f64) {
        self.n += 1.0;
        for x in 0..3 {
            self.from[x] += from[x];
            self.to[x] += to[x];
            self.error[x] += error[x];
            self.input[x] += input[x];
            for y in 0..3 {
                self.within[x][y] += from[x] * from[y];
                self.across[x][y] += from[x] * to[y];
            }
        }
        self.hazard += hazard;
        self.hazard_error += hazard_error;
    }

    /// These sums and `sign` times others.
    fn with(&self, o: &Sums, sign: f64) -> Sums {
        Sums {
            n: self.n + sign * o.n,
            from: std::array::from_fn(|x| self.from[x] + sign * o.from[x]),
            to: std::array::from_fn(|x| self.to[x] + sign * o.to[x]),
            within: std::array::from_fn(|x| std::array::from_fn(|y| self.within[x][y] + sign * o.within[x][y])),
            across: std::array::from_fn(|x| std::array::from_fn(|y| self.across[x][y] + sign * o.across[x][y])),
            error: std::array::from_fn(|x| self.error[x] + sign * o.error[x]),
            input: std::array::from_fn(|x| self.input[x] + sign * o.input[x]),
            hazard: self.hazard + sign * o.hazard,
            hazard_error: self.hazard_error + sign * o.hazard_error,
        }
    }

    /// The lag's transported local error per axis at the output, β(K, j)ᵀ δ_j, and the mean of β's column sums (module
    /// documentation, # The regression, # The local error of an interval). `scale` is the energies' size at the lag's
    /// time (the centre's modulus): a variance below the round-off of the sums of that size is none.
    fn moved(&self, scale: f64) -> (V3, f64) {
        let n = self.n;
        let (mean_from, mean_to): (V3, V3) = (self.from.map(|x| x / n), self.to.map(|x| x / n));
        let variance: M3 = std::array::from_fn(|x| std::array::from_fn(|y| self.within[x][y] / n - mean_from[x] * mean_from[y]));
        let covariance: M3 = std::array::from_fn(|x| std::array::from_fn(|y| self.across[x][y] / n - mean_from[x] * mean_to[y]));
        let (values, vectors) = symmetric3_eigen(&variance);
        // The n values' rounded mean leaves each deviation n u of the size, a variance of that squared.
        let none = (n * UNIT_ROUNDOFF * scale).powi(2);
        let mut beta = [[0.0; 3]; 3];
        for m in 0..3 {
            let v = vectors[m];
            // Along a direction of variance: v (vᵀ C)/λ; along one of none: v vᵀ, the error carried unchanged.
            let row: V3 = if values[m] > none { std::array::from_fn(|y| (0..3).map(|x| v[x] * covariance[x][y]).sum::<f64>() / values[m]) } else { v };
            for x in 0..3 {
                for y in 0..3 {
                    beta[x][y] += v[x] * row[y];
                }
            }
        }
        let share = if self.hazard > 0.0 { self.hazard_error / self.hazard } else { 0.0 };
        let local: V3 = std::array::from_fn(|x| (self.error[x] + share * self.input[x]) / n);
        (std::array::from_fn(|y| (0..3).map(|x| beta[x][y] * local[x]).sum()), beta.iter().flatten().sum::<f64>() / 3.0)
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
    let centre = |k: usize| -> V3 { std::array::from_fn(|a| held.iter().map(|&i| mark(i, k).level[a]).sum::<f64>() / all) };
    let to = centre(output);
    // The lag's sums per group, the energies less the lag's and the output's centres.
    let lag = |j: usize| -> (Vec<Sums>, f64) {
        let from = centre(j);
        let mut sums = vec![Sums::default(); groups];
        for &i in &held {
            let (m, before, last) = (mark(i, j), mark(i, j - 1), mark(i, output));
            sums[i / group].add(
                std::array::from_fn(|a| m.level[a] - from[a]),
                std::array::from_fn(|a| last.level[a] - to[a]),
                std::array::from_fn(|a| m.error[a] - before.error[a]),
                std::array::from_fn(|a| m.input[a] - before.input[a]),
                m.hazard - before.hazard,
                m.hazard_error - before.hazard_error,
            );
        }
        (sums, from.iter().map(|x| x * x).sum::<f64>().sqrt())
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
                let (moved, weight) = if r == 0 { whole.moved(scale) } else { whole.with(&sums[r - 1], -1.0).moved(scale) };
                for a in 0..3 {
                    slot.0[a] += moved[a];
                }
                slot.1 += weight;
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
    use crate::estimate::{spread, Spread};
    use crate::random::{Purpose, Stream};

    const GROUP: usize = 8;

    /// Trajectories of marks from a rule: `next(stream, level) -> level` over each interval, the local error `error` an
    /// interval per axis, no jumps.
    fn series(trajectories: usize, outputs: usize, start: impl Fn(&mut Stream) -> V3, next: impl Fn(&mut Stream, V3) -> V3, error: V3) -> Vec<Vec<Option<Mark>>> {
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

    /// Three axes' thermal energies.
    fn thermals(s: &mut Stream) -> V3 {
        [thermal(s), thermal(s), thermal(s)]
    }

    /// The jackknife's spread of an axis's transported error.
    fn error(t: &Transported, a: usize) -> Spread {
        spread(t.without.iter().map(|w| w[a]), t.without.len() as f64)
    }

    /// Energies that are kept: every interval's error arrives whole, the plain sum, with no statistical error -- the
    /// regression of a kept energy on itself is exactly 1.
    #[test]
    fn kept_energies_carry_every_error() {
        let (outputs, local) = (40, [1e-3, -2e-3, 5e-4]);
        let s = series(88, outputs, thermals, |_, level| level, local);
        let read = transported(&s, GROUP, outputs).unwrap();
        for a in 0..3 {
            let plain = local[a] * outputs as f64;
            assert!((read.all[a] - plain).abs() <= 1e-12 * plain.abs(), "axis {a}: {} against the plain sum {plain}", read.all[a]);
            assert!(error(&read, a).error <= 1e-12 * plain.abs(), "axis {a}: a statistical error {} of a kept energy's sum", error(&read, a).error);
        }
    }

    /// Energies that relax: E ← ρE + (1 − ρ)·(a fresh thermal energy) over an interval regresses by ρ a lag, and a
    /// constant local error δ an interval arrives as δ(1 + ρ + ρ² + …) = δ/(1 − ρ) -- not as the plain sum δK --, each
    /// axis within the family's interval of its jackknife (six looks). The cut drops the lags that only bring noise:
    /// uncut, the far lags' noise would add up as the run's length, the error as its square root.
    #[test]
    fn relaxing_energies_forget_their_errors() {
        let (rho, delta) = (0.6, 1e-3);
        let relax = |s: &mut Stream, level: V3| -> V3 { std::array::from_fn(|a| rho * level[a] + (1.0 - rho) * thermal(s)) };
        let mut errors = vec![];
        for outputs in [50, 400] {
            let s = series(704, outputs, thermals, relax, [delta; 3]);
            let read = transported(&s, GROUP, outputs).unwrap();
            for a in 0..3 {
                let (expected, interval) = (delta / (1.0 - rho), error(&read, a).interval_over(6.0));
                assert!((read.all[a] - expected).abs() <= interval, "axis {a} at {outputs} outputs: {} ± {interval} against {expected}", read.all[a]);
            }
            errors.push(error(&read, 0).error);
        }
        assert!(errors[1] < (400.0_f64 / 50.0).sqrt() * errors[0], "the error grew with the run as an uncut sum's: {errors:?}");
    }

    /// Energies exchanged between two axes, their sum kept: an error made on one axis arrives on both, and whole in
    /// their sum.
    #[test]
    fn exchanged_energies_keep_their_sum() {
        let (outputs, delta) = (30, 1e-3);
        let exchange = |s: &mut Stream, level: V3| -> V3 {
            let share = s.uniform();
            let both = level[0] + level[1];
            [share * both, (1.0 - share) * both, level[2]]
        };
        let s = series(704, outputs, thermals, exchange, [delta, 0.0, 0.0]);
        let read = transported(&s, GROUP, outputs).unwrap();
        let both = Transported { all: [read.all[0] + read.all[1], 0.0, 0.0], without: read.without.iter().map(|w| [w[0] + w[1], 0.0, 0.0]).collect() };
        let interval = error(&both, 0).interval();
        assert!((both.all[0] - delta * outputs as f64).abs() <= interval, "{} ± {interval} against the sum {}", both.all[0], delta * outputs as f64);
        assert!(read.all[1] > 0.0 && read.all[2].abs() <= error(&read, 2).interval(), "{:?}", read.all);
    }

    /// An axis on which every atom has the same, kept energy: nothing to regress on, the error carried unchanged.
    #[test]
    fn an_axis_without_spread_carries_its_error() {
        let (outputs, delta) = (20, 1e-3);
        let s = series(88, outputs, |s| [thermal(s), 2.0, thermal(s)], |_, level| level, [0.0, delta, 0.0]);
        let read = transported(&s, GROUP, outputs).unwrap();
        assert!((read.all[1] - delta * outputs as f64).abs() <= 1e-12 * delta * outputs as f64 && read.all[0] == 0.0, "{:?}", read.all);
    }

    /// The jumps' input times the hazard's relative error is a local error like the doubling's.
    #[test]
    fn a_hazard_s_error_scales_the_jumps_input() {
        let outputs = 10;
        let mut s = series(88, outputs, thermals, |_, level| level, [0.0; 3]);
        for t in &mut s {
            for (k, m) in t.iter_mut().enumerate() {
                let m = m.as_mut().unwrap();
                (m.input, m.hazard, m.hazard_error) = ([0.02 * k as f64, 0.0, 0.0], 3.0 * k as f64, 0.15 * k as f64);
            }
        }
        let read = transported(&s, GROUP, outputs).unwrap();
        let expected = 0.05 * 0.02 * outputs as f64;
        assert!((read.all[0] - expected).abs() <= 1e-12 * expected && read.all[1].abs() <= 1e-12 * expected, "{:?}", read.all);
    }

    /// Fewer than two groups holding an atom at the output read nothing; a group that holds none leaves the estimate
    /// as it is.
    #[test]
    fn one_group_reads_nothing() {
        let mut s = series(24, 5, thermals, |_, level| level, [1e-3; 3]);
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
