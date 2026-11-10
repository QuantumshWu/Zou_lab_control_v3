//! The transport of the step's local energy errors to an output (SPEC §2.2, # The read).
//!
//! # What is transported
//! A read trajectory (adapt.rs) accumulates per lab axis the local errors its steps make of that axis's oscillator
//! energy, and the energy its jumps put in. A local error made at the time t_j is an error of T at a later output t_K
//! only as far as the motion remembers it: a trap without scattering keeps an atom's energy, and what a step took from it
//! stays taken; a cooled motion forgets an atom's energy at its cooling rate, and with it the error.
//!
//! # The regression
//! To first order in the local errors ℓ_n, the step's error of an ensemble mean E[h(X_{t_K})] is the telescoping sum
//! Σ_n E[∇u(t_n, X_n)·ℓ_n], u(t, x) = E[h(X_{t_K}) | X_t = x] the exact dynamics' expectation from the state x. The three
//! axes' energies E are the motion's slow variables; with u linear in them over the ensemble's spread, its slope is the
//! ensemble's own least-squares regression of the energies at t_K on those at t_j,
//!   β(K, j) = Var(E_j)⁻¹ Cov(E_j, E_K)   (3 × 3; row: the axis at t_j, column: the axis at t_K),
//! and the sum is Σ_j β(K, j)ᵀ δ_j, δ_j the ensemble's mean local error made over the output interval ending at t_j.
//! This is Onsager's regression as the ensemble measures it: a disturbance of the mean relaxes as the atoms' own
//! fluctuations regress. Energies that are kept (β = 1) add up as they are made -- the plain sum; energies exchanged
//! between axes move by β's off-diagonal entries, their sum kept; forgotten ones (β → 0) leave. Along a direction in
//! which the ensemble's energies do not differ the regression measures nothing: an error is carried there unchanged (the
//! safe side), which is also the exact answer where every atom's energy is the same and kept.
//!
//! # The local error of an interval
//! δ_j is the mean of the errors accumulated over the interval -- the doubling's differences and the kick the carried
//! internal state changes (adapt.rs) -- and of the energy the jumps put in over it times the hazard's relative error
//! there, (Σ its accumulated error)/(Σ its integral): a jump's recoil is not in the jump-free doubling, and a hazard that
//! errs by a share makes that share too many or too few of them.
//!
//! # The cut
//! The lags K − j are added in pairs from the output backwards while a pair's regressions are positive in the mean of
//! their entries' column sums (the share of an energy error still in the three energies): Geyer's initial positive
//! sequence (Statist. Sci. 7 (1992) 473) -- beyond the first pair that is not, an estimated regression is its noise, and
//! adding it would only add that noise times all the errors made before.
//!
//! # The error of the read
//! The delete-one-group jackknife of the whole estimate, as every other estimate's (estimate.rs): the regressions, the
//! local errors, the hazard's ratio, T's scale and the cut made anew without each group. With the cut each replicate's
//! own, the jackknife carries the cut's chance too: over 40 seeds of a relaxing ensemble (regression 0.6 a lag, 50 and
//! 400 outputs, 88 and 704 trajectories) the read's deviation over its jackknife error has the standard deviation 0.84 to
//! 0.93 and the read's mean is 1.02 to 1.09 of the exact sum (the cut keeps the positive chance pairs); with one cut for
//! all replicates that standard deviation was 1.2 at 88 trajectories.

use crate::decide::Read;
use crate::estimate::spread;
use crate::linalg::{symmetric3_eigen, M3, V3};
use crate::units::UNIT_ROUNDOFF;

/// What the transport takes of one read trajectory at one output (adapt.rs): per lab axis its oscillator energy
/// (rad/µs), the accumulated local errors of it and the accumulated energy its jumps put in; the hazard's integral ∫γ dt
/// along it and that integral's accumulated local error.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mark {
    pub level: V3,
    pub error: V3,
    pub input: V3,
    pub hazard: f64,
    pub hazard_error: f64,
}

/// One lag's sums over some trajectories: their number; the energies at the lag's time and at the output, each less a
/// fixed centre; their products; the interval's local errors, the jumps' input, the hazard and its error.
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
        let mut out = *self;
        out.n += sign * o.n;
        for x in 0..3 {
            out.from[x] += sign * o.from[x];
            out.to[x] += sign * o.to[x];
            out.error[x] += sign * o.error[x];
            out.input[x] += sign * o.input[x];
            for y in 0..3 {
                out.within[x][y] += sign * o.within[x][y];
                out.across[x][y] += sign * o.across[x][y];
            }
        }
        out.hazard += sign * o.hazard;
        out.hazard_error += sign * o.hazard_error;
        out
    }

    /// The lag's transported local error per axis at the output, β(K, j)ᵀ δ_j, and the mean of β's column sums (module
    /// documentation, # The regression, # The local error of an interval). `scale` is the energies' size at the lag's time
    /// (the centre's modulus): a variance below the round-off of the sums of that size is none.
    fn transported(&self, scale: f64) -> (V3, f64) {
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
        let moved: V3 = std::array::from_fn(|y| (0..3).map(|x| beta[x][y] * local[x]).sum());
        (moved, beta.iter().flatten().sum::<f64>() / 3.0)
    }
}

/// T's relative step error per lab axis at the output `output` from the read trajectories' marks (module
/// documentation): `series[i][k]` is trajectory i's mark at output k while it is held, the trajectories in groups of
/// `group` in their order. None where fewer than two groups hold an atom at the output, where a group left out leaves
/// fewer than two atoms, or where an axis's mean energy is not positive: no read.
pub fn transported(series: &[Vec<Option<Mark>>], group: usize, output: usize) -> Option<[Read; 3]> {
    let held: Vec<usize> = (0..series.len()).filter(|&i| series[i][output].is_some()).collect();
    let groups = series.len().div_ceil(group);
    let mark = |i: usize, k: usize| series[i][k].expect("a trajectory held at an output is held at every earlier one");
    let members: Vec<f64> = (0..groups).map(|g| held.iter().filter(|&&i| i / group == g).count() as f64).collect();
    let holding: Vec<usize> = (0..groups).filter(|&g| members[g] > 0.0).collect();
    let all = held.len() as f64;
    if holding.len() < 2 || holding.iter().any(|&g| all - members[g] < 2.0) {
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
    // The estimate of all the trajectories (0) and of those without each holding group, lag pair by lag pair, each
    // until its own cut.
    let mut totals = vec![[0.0; 3]; 1 + holding.len()];
    let mut open = vec![true; totals.len()];
    let mut j = output;
    while j >= 1 && open.iter().any(|&o| o) {
        let pair: Vec<usize> = [Some(j), j.checked_sub(1).filter(|&k| k >= 1)].into_iter().flatten().collect();
        let mut added = vec![([0.0; 3], 0.0); totals.len()];
        for &k in &pair {
            let (sums, scale) = lag(k);
            let whole = sums.iter().fold(Sums::default(), |a, s| a.with(s, 1.0));
            for (r, slot) in added.iter_mut().enumerate().filter(|(r, _)| open[*r]) {
                let (moved, share) = if r == 0 { whole.transported(scale) } else { whole.with(&sums[holding[r - 1]], -1.0).transported(scale) };
                for a in 0..3 {
                    slot.0[a] += moved[a];
                }
                slot.1 += share;
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
    // T's scale: the axis's mean energy at the output, of the same trajectories as the estimate.
    let level = |without: Option<usize>| -> V3 {
        let kept: Vec<usize> = held.iter().copied().filter(|&i| without != Some(i / group)).collect();
        std::array::from_fn(|a| kept.iter().map(|&i| mark(i, output).level[a]).sum::<f64>() / kept.len() as f64)
    };
    let scale = level(None);
    if scale.iter().any(|&s| !(s > 0.0)) {
        return None;
    }
    let scales: Vec<V3> = holding.iter().map(|&g| level(Some(g))).collect();
    Some(std::array::from_fn(|a| Read {
        value: totals[0][a] / scale[a],
        spread: spread((0..holding.len()).map(|r| totals[1 + r][a] / scales[r][a]), holding.len() as f64),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::{Purpose, Stream};
    use crate::units::CONFIDENCE;

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

    /// Energies that are kept: every interval's error arrives whole, the plain sum, with no statistical error -- the
    /// regression of a kept energy on itself is exactly 1.
    #[test]
    fn kept_energies_carry_every_error() {
        let (outputs, error) = (40, [1e-3, -2e-3, 5e-4]);
        let s = series(88, outputs, |s| [thermal(s), thermal(s), thermal(s)], |_, level| level, error);
        let read = transported(&s, GROUP, outputs).unwrap();
        for a in 0..3 {
            let mean = s.iter().map(|t| t[outputs].unwrap().level[a]).sum::<f64>() / 88.0;
            let plain = error[a] * outputs as f64 / mean;
            assert!((read[a].value - plain).abs() <= 1e-12 * plain.abs(), "axis {a}: {} against the plain sum {plain}", read[a].value);
        }
    }

    /// Energies that relax: E ← ρE + (1 − ρ)·(a fresh thermal energy) over an interval regresses by ρ a lag, and a
    /// constant local error δ an interval arrives as δ(1 + ρ + ρ² + …) = δ/(1 − ρ) -- not as the plain sum δK --, read
    /// within its own interval at the crate's confidence over the six looks made here. The cut drops the lags that only
    /// bring noise: uncut, the far lags' noise would add up as the run's length, the interval as its square root.
    #[test]
    fn relaxing_energies_forget_their_errors() {
        let (rho, delta) = (0.6, 1e-3);
        let relax = |s: &mut Stream, level: V3| -> V3 { std::array::from_fn(|a| rho * level[a] + (1.0 - rho) * thermal(s)) };
        let mut intervals = vec![];
        for outputs in [50, 400] {
            let s = series(704, outputs, |s| [thermal(s), thermal(s), thermal(s)], relax, [delta; 3]);
            let read = transported(&s, GROUP, outputs).unwrap();
            for a in 0..3 {
                let mean = s.iter().map(|t| t[outputs].unwrap().level[a]).sum::<f64>() / 704.0;
                let expected = delta / (1.0 - rho) / mean;
                let interval = read[a].spread.interval_over(6.0);
                assert!((read[a].value - expected).abs() <= interval, "axis {a} at {outputs} outputs: {} ± {interval} against {expected}", read[a].value);
            }
            intervals.push(CONFIDENCE * read[0].spread.error);
        }
        assert!(intervals[1] < (400.0_f64 / 50.0).sqrt() * intervals[0], "the interval grew with the run as an uncut sum's: {intervals:?}");
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
        let s = series(704, outputs, |s| [thermal(s), thermal(s), thermal(s)], exchange, [delta, 0.0, 0.0]);
        let read = transported(&s, GROUP, outputs).unwrap();
        let mean = |a: usize| s.iter().map(|t| t[outputs].unwrap().level[a]).sum::<f64>() / 704.0;
        let arrived = read[0].value * mean(0) + read[1].value * mean(1);
        let interval = CONFIDENCE * (read[0].spread.error * mean(0) + read[1].spread.error * mean(1));
        assert!((arrived - delta * outputs as f64).abs() <= interval, "{arrived} ± {interval} against the sum {}", delta * outputs as f64);
        assert!(read[1].value > 0.0 && read[2].value.abs() <= CONFIDENCE * read[2].spread.error + 1e-15);
    }

    /// An axis on which every atom has the same, kept energy: nothing to regress on, the error carried unchanged.
    #[test]
    fn an_axis_without_spread_carries_its_error() {
        let (outputs, delta) = (20, 1e-3);
        let s = series(88, outputs, |s| [thermal(s), 2.0, thermal(s)], |_, level| level, [0.0, delta, 0.0]);
        let read = transported(&s, GROUP, outputs).unwrap();
        assert!((read[1].value - delta * outputs as f64 / 2.0).abs() <= 1e-12);
        assert_eq!(read[0].value, 0.0);
    }

    /// The jumps' input times the hazard's relative error is a local error like the doubling's.
    #[test]
    fn a_hazard_s_error_scales_the_jumps_input() {
        let outputs = 10;
        let mut s = series(88, outputs, |s| [thermal(s), thermal(s), thermal(s)], |_, level| level, [0.0; 3]);
        for t in &mut s {
            for (k, m) in t.iter_mut().enumerate() {
                let m = m.as_mut().unwrap();
                (m.input, m.hazard, m.hazard_error) = ([0.02 * k as f64, 0.0, 0.0], 3.0 * k as f64, 0.15 * k as f64);
            }
        }
        let read = transported(&s, GROUP, outputs).unwrap();
        let mean = s.iter().map(|t| t[outputs].unwrap().level[0]).sum::<f64>() / 88.0;
        let expected = 0.05 * 0.02 * outputs as f64 / mean;
        assert!((read[0].value - expected).abs() <= 1e-12 * expected);
    }

    /// Fewer than two groups holding an atom at the output read nothing.
    #[test]
    fn one_group_reads_nothing() {
        let mut s = series(16, 5, |s| [thermal(s), thermal(s), thermal(s)], |_, level| level, [1e-3; 3]);
        for t in s.iter_mut().skip(8) {
            t[5] = None;
        }
        assert!(transported(&s, GROUP, 5).is_none());
        assert!(transported(&s, GROUP, 4).is_some());
    }
}
