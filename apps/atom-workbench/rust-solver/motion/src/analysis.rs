//! The analysis times: the times the solve's reads and decisions are to stand on, chosen by how fast what is read
//! changes -- not by the outputs shown (motion3d_status/rewrite/r_samples.status.md: the output grid entered the
//! decisions through the reads at every output). A pure function; the solve reads it (solve.rs, # The analysis times:
//! the readings, the candidates' first round, what stands on the times).
//!
//! # The rule
//! The candidates are the companion level's step ends, m·2τ (m = 0 … M, M·2τ the run's end): every time both levels of
//! the pilot have a state of their own, the finest grid a read of the step's error can stand on. The times start as
//! the run's two ends and the lights' switching times, the cuts: a reading may turn there at once, and a turn at a
//! known time is no read's to find. An interval [a, b] with candidates inside is read at its middle and its two
//! quarters (the candidates there, by index) -- and where it is longer than a span `window` the caller gives, at the
//! middles of its halves' halves on, as far as leaves its reads no farther apart than a quarter of that span (# What it
//! does not see) --; each reading's deviation from the line through the ends' readings,
//!   d(t) = y(t) − [y(a)(b − t) + y(b)(t − a)]/(b − a),
//! is compared with its own noise: d's delete-one-group jackknife (estimate.rs, # Errors) -- the ends and the point are
//! read on the same trajectories, so their correlation is in it --, its interval at the confidence z among the
//! interval's looks (the quantities times the points read, Šidák: estimate.rs, # Many looks), and the rounding of d's
//! own five operations, γ₅(|y(t)| + |y(a)| + |y(b)|) (units.rs `rounding`): a reading the same in every group has no
//! spread, and the line through its ends misses it by that rounding where it does not change (measured: a T read the
//! same in eleven identical groups split a 1000 µs run down to its candidates). Where some deviation lies
//! beyond its interval the interval is halved at its middle and each half judged alike; else the line through its ends
//! holds what is read inside it within the noise, and the interval stands. The analysis times are the ends of the
//! intervals that stand.
//!
//! It is the tables' rule (model/tabulate.rs, # Refinement; `halving_on`): halve at the middles while the coarser
//! grid's interpolant misses the new nodes beyond the share -- here the share is the read's own noise, below which no
//! read of the run can resolve anything -- down to the grid's floor, the candidates' spacing. The tables' second clause
//! (an axis whose read falls by a factor of 2 or less a halving on a grid that resolves its line's rate stops there: a
//! kink, a read of no order) rests on a scale known beforehand, the line's rate; a reading in time has none -- a
//! transient faster than its interval reads the same at every halving until the interval meets its width --, so the
//! halving runs to the noise or to the candidates. Nothing is counted against a number of times: how many there are is
//! what the readings ask, through their own noise.
//!
//! Why the quarters as well: the line's leading error on a smooth reading, (h²/8) y″ at the middle of an interval of
//! width h, is read there; a part odd about the middle (a cubic's, an edge off the middle) is zero there and not at the
//! quarters, which also lie within h/8 of every time of the interval's inner half.
//!
//! What it does not see: a feature narrower than the reads' spacing that none of them touches -- a peak of width w
//! anywhere in [0, end] is seen at the first reads where it reaches one of them above its noise (within h/8 of a
//! quarter or the middle, h the first interval). That is why the switching times, where a reading's sharp features
//! begin, are cuts from the outset, and why an interval longer than the caller's span is read at points a quarter of
//! the span apart at most (the solve's span is T's shortest window, the trap's fastest period: solve.rs `analysed`):
//! a feature of that width anywhere is then read. The reads are the nested middles of the interval's halves -- the
//! middle, the quarters, the eighths, ... --, so that each half's reads are among its parent's and the reads that split
//! an interval are its halves' own: a feature found is not lost a halving later. They are reads, not ends: an interval
//! whose reads all lie on its line stands whole, however long, and adds no time.
//!
//! Noise: an interval over which the reading is a line splits by chance with probability p = P(|N| > z) at most
//! (Šidák, and Student's t at the groups for the jackknife's own spread); read at its middle and quarters alone, its
//! halves are judged on reads of their own, so the splits of a flat reading are a branching process of mean 2p
//! children an interval: 1/(1 − 2p) intervals from each cut's (1.10 at z = 2), whatever the number of candidates. Read
//! at nested middles, a half's reads are its parent's, and the outlier that split the parent by chance splits the half
//! that holds it again, as a feature would be followed -- what makes a feature found stay found: a chance split costs
//! as many more as the halvings down to the candidates at the most, log₂ of their number.

use crate::estimate::{spread, Spread};
use crate::units::rounding;

/// A quantity read at an analysis time: its estimate over all the groups and its values with each group left out (the
/// delete-one-group jackknife's, estimate.rs # Errors), the same groups at every time.
#[derive(Clone, Debug, PartialEq)]
pub struct Reading {
    pub whole: f64,
    pub leave: Vec<f64>,
}

/// The analysis times (module documentation): of the candidates `times` (increasing; the companion's step ends), the
/// indices of the ends of the intervals that stand, from the cuts `cuts` (indices into `times`; the first candidate and
/// the last are cuts whatever is given), each interval read at points no farther apart than a quarter of `window`
/// (the same units as `times`; unbounded: its middle and quarters alone). `read(k)` reads the quantities at the
/// candidate k, the same quantities in the same order at every candidate; each candidate is read once.
pub fn analysis_times(times: &[f64], cuts: &[usize], window: f64, read: &mut dyn FnMut(usize) -> Vec<Reading>) -> Vec<usize> {
    let last = times.len() - 1;
    let mut ends: Vec<usize> = cuts.iter().copied().chain([0, last]).filter(|&k| k <= last).collect();
    ends.sort_unstable();
    ends.dedup();
    let mut readings: Vec<Option<Vec<Reading>>> = vec![None; times.len()];
    let mut reading = |k: usize| -> Vec<Reading> { readings[k].get_or_insert_with(|| read(k)).clone() };
    let mut open: Vec<(usize, usize)> = ends.windows(2).map(|w| (w[0], w[1])).collect();
    while let Some((a, b)) = open.pop() {
        let middle = (a + b) / 2;
        // The nested middles to the depth that leaves a quarter of the window at most between them: the middle and the
        // quarters at two, for an interval no longer than the window.
        let depth = 2 + ((times[b] - times[a]) / window).log2().ceil().max(0.) as u32;
        let mut points = vec![];
        middles(a, b, depth, &mut points);
        points.dedup();
        points.retain(|&k| a < k && k < b);
        if points.is_empty() {
            continue;
        }
        let (start, end) = (reading(a), reading(b));
        let looks = (points.len() * start.len()) as f64;
        let beyond = points.iter().any(|&k| {
            let s = (times[k] - times[a]) / (times[b] - times[a]);
            reading(k).iter().zip(start.iter().zip(&end)).any(|(y, (ya, yb))| {
                let line = |at: f64, from: f64, to: f64| at - ((1.0 - s) * from + s * to);
                let off = line(y.whole, ya.whole, yb.whole);
                let leave = y.leave.iter().zip(ya.leave.iter().zip(&yb.leave)).map(|(&y, (&ya, &yb))| line(y, ya, yb));
                let noise: Spread = spread(leave, y.leave.len() as f64);
                off.abs() > noise.interval_over(looks) + rounding(5) * (y.whole.abs() + ya.whole.abs() + yb.whole.abs())
            })
        });
        if beyond {
            ends.push(middle);
            open.extend([(a, middle), (middle, b)]);
        }
    }
    ends.sort_unstable();
    ends
}

/// The nested middles of the candidates' indices a < b to `depth` (`analysis_times`): the middle and, below it, its
/// halves' own, in order.
fn middles(a: usize, b: usize, depth: u32, out: &mut Vec<usize>) {
    if depth == 0 || b <= a + 1 {
        return;
    }
    let m = (a + b) / 2;
    middles(a, m, depth - 1, out);
    out.push(m);
    middles(m, b, depth - 1, out);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::{Purpose, Stream};

    /// The candidates of a run of 1000 µs at the companion's step 2τ = 1000/1024 µs.
    fn candidates() -> Vec<f64> {
        (0..=1024).map(|m| 1000.0 * m as f64 / 1024.0).collect()
    }

    /// Readings of the quantities `f` on `groups` groups, each group's value the quantity plus a normal noise of the
    /// deviation `sigma` (a fresh draw per group, time and quantity, from the case's stream): the mean over the groups
    /// and the means with each group left out.
    fn noisy<'a>(quantities: &'a [&'a dyn Fn(f64) -> f64], groups: usize, sigma: f64, case: u64) -> impl FnMut(usize, f64) -> Vec<Reading> + 'a {
        move |k, t| {
            quantities
                .iter()
                .enumerate()
                .map(|(q, f)| {
                    let mut s = Stream::new(29, Purpose::Initial, case, (k * quantities.len() + q) as u32);
                    let values: Vec<f64> = (0..groups).map(|_| f(t) + sigma * s.normal()).collect();
                    let sum: f64 = values.iter().sum();
                    Reading { whole: sum / groups as f64, leave: values.iter().map(|v| (sum - v) / (groups - 1) as f64).collect() }
                })
                .collect()
        }
    }

    /// The analysis times of the readings `at` over the candidates with the cuts `cuts`, and the candidates read.
    fn analysed(cuts: &[usize], at: &mut dyn FnMut(usize, f64) -> Vec<Reading>) -> (Vec<usize>, usize) {
        within(cuts, f64::INFINITY, at)
    }

    /// `analysed` with the reads of an interval no farther apart than a quarter of `window` (µs).
    fn within(cuts: &[usize], window: f64, at: &mut dyn FnMut(usize, f64) -> Vec<Reading>) -> (Vec<usize>, usize) {
        let times = candidates();
        let mut count = 0;
        let ends = analysis_times(&times, cuts, window, &mut |k| {
            count += 1;
            at(k, times[k])
        });
        (ends, count)
    }

    /// The largest distance of the line through the analysis times from a function, over every candidate.
    fn missed(ends: &[usize], f: &dyn Fn(f64) -> f64) -> f64 {
        let times = candidates();
        ends.windows(2)
            .flat_map(|w| (w[0]..=w[1]).map(move |k| (w[0], w[1], k)))
            .map(|(a, b, k)| {
                let s = (times[k] - times[a]) / (times[b] - times[a]);
                (f(times[k]) - ((1.0 - s) * f(times[a]) + s * f(times[b]))).abs()
            })
            .fold(0.0, f64::max)
    }

    /// A transient after a light's switch and an edge inside the run (module documentation, # The rule): a population
    /// that relaxes from 0.1 to 0.6 at the rate 1/(4 µs) from a switch at 250 µs (a cut), and a second quantity, a
    /// temperature-like ratio that steps by a fifth across 3 µs at 637 µs (no cut: the reads find it), each read on 16
    /// groups at a group's noise of 0.02 (the mean's 0.005). The halving resolves both: the line through the analysis
    /// times misses neither by more than six of the mean's noise at any candidate, the times are dense at the switch's
    /// right and at the edge and sparse elsewhere -- fewer than a tenth of the 1025 candidates, fewer than a third read.
    #[test]
    fn a_transient_and_an_edge_are_halved_to_their_width_and_no_further() {
        let (switch, edge) = (250.0, 637.0);
        let relax = |t: f64| if t < switch { 0.1 } else { 0.6 - 0.5 * (-(t - switch) / 4.0).exp() };
        let step = |t: f64| 1.0 + 0.2 / (1.0 + (-(t - edge) / 3.0).exp());
        let cut = candidates().iter().position(|&t| t == switch).expect("the switch on the companion's grid");
        let (ends, count) = analysed(&[cut], &mut noisy(&[&relax, &step], 16, 0.02, 1));
        let times = candidates();
        let near = |at: f64, width: f64| ends.iter().filter(|&&k| (times[k] - at).abs() <= width).count();
        let (relaxed, stepped) = (missed(&ends, &relax), missed(&ends, &step));
        eprintln!("{} analysis times of 1025 candidates ({count} read): {} within 20 µs after the switch, {} within 15 µs of the edge; the line misses the relaxation by {relaxed:.4}, the edge by {stepped:.4} (the mean's noise 0.005)", ends.len(), ends.iter().filter(|&&k| times[k] >= switch && times[k] <= switch + 20.0).count(), near(edge, 15.0));
        assert!(relaxed <= 6.0 * 0.005 && stepped <= 6.0 * 0.005, "the line misses the relaxation by {relaxed}, the edge by {stepped}");
        assert!(near(switch + 10.0, 10.0) >= 4 && near(edge, 15.0) >= 4, "dense at the features: {ends:?}");
        assert!(ends.len() < 103 && count < 342, "{} analysis times, {count} read", ends.len());
    }

    /// A narrow peak (module documentation, # The rule): a Gaussian population peak of height 0.3 and width 8 µs at
    /// 513 µs -- the first reads (at the quarters and the middle of [0, 1000]) meet its flank 13 µs off its centre --,
    /// read on 16 groups at a group's noise of 0.02. The halving follows it down to its width: the line through the
    /// analysis times misses it by no more than six of the mean's noise at any candidate, with times every few µs across
    /// it and few elsewhere.
    #[test]
    fn a_narrow_peak_is_halved_to_its_width() {
        let centre = 513.0;
        let peak = |t: f64| 0.2 + 0.3 * (-0.5 * ((t - centre) / 8.0).powi(2)).exp();
        let (ends, count) = analysed(&[], &mut noisy(&[&peak], 16, 0.02, 2));
        let times = candidates();
        let across = ends.iter().filter(|&&k| (times[k] - centre).abs() <= 24.0).count();
        let away = ends.iter().filter(|&&k| (times[k] - centre).abs() > 100.0).count();
        let off = missed(&ends, &peak);
        eprintln!("{} analysis times ({count} read): {across} within 3 widths of the peak, {away} more than 100 µs away; the line misses it by {off:.4} (the mean's noise 0.005)", ends.len());
        assert!(off <= 6.0 * 0.005, "the line misses the peak by {off}: {ends:?}");
        assert!(across >= 6 && away <= 8, "{across} times across the peak, {away} away from it: {ends:?}");
    }

    /// A peak narrower than the first reads' spacing (module documentation, # What it does not see): height 0.3 and width
    /// 3 µs at 313 µs, 63 µs from the first reads at the quarters of [0, 1000], read on 16 groups at a group's noise of
    /// 0.02 -- unseen there: no time comes within three widths of it, and the line misses it by its height. Read no farther
    /// apart than a quarter of a 40 µs window (the nested middles to 7.8 µs), it is found and halved to its width, the
    /// line within six of the mean's noise of it; and flat noise read so, on 100 cases, splits no more than the
    /// branching process's mean times the halvings down to the candidates (module documentation, # Noise: a chance
    /// outlier is followed down), no case beyond twice those halvings: the reads add no time where nothing changes but
    /// what chance gives.
    #[test]
    fn a_peak_between_the_first_reads_is_read_where_they_are_a_quarter_window_apart() {
        let peak = |t: f64| 0.2 + 0.3 * (-0.5 * ((t - 313.) / 3.).powi(2)).exp();
        let (blind, _) = within(&[], f64::INFINITY, &mut noisy(&[&peak], 16, 0.02, 3));
        let times = candidates();
        assert!(blind.iter().all(|&k| (times[k] - 313.).abs() > 9.) && missed(&blind, &peak) > 0.3 - 6. * 0.005, "{blind:?}");
        let (seen, count) = within(&[], 40., &mut noisy(&[&peak], 16, 0.02, 3));
        let off = missed(&seen, &peak);
        eprintln!("{} analysis times ({count} read); the line misses the peak by {off:.4}", seen.len());
        assert!(off <= 6. * 0.005, "the line misses the peak by {off}: {seen:?}");
        let flat: [&dyn Fn(f64) -> f64; 2] = [&|_| 0.3, &|_| 12.0];
        let splits: Vec<f64> = (0..100u64).map(|case| (within(&[], 40., &mut noisy(&flat, 8, 0.05, 700 + case)).0.len() - 2) as f64).collect();
        let n = splits.len() as f64;
        let mean = splits.iter().sum::<f64>() / n;
        let deviation = (splits.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (n - 1.0) / n).sqrt();
        let (p, levels) = (crate::estimate::tail_among(1.0), (candidates().len() as f64 - 1.).log2());
        assert!(mean <= levels * (1. / (1. - 2. * p) - 1.) + crate::units::CONFIDENCE * deviation && splits.iter().all(|&x| x <= 2. * levels), "{mean} ± {deviation} splits a case: {splits:?}");
    }

    /// Flat noise is not chased (module documentation, # Noise): three quantities constant in time, read on 8 groups at
    /// a group's noise of 0.05, over 400 cases. An interval splits by chance with probability p = 0.0455 at most, so the
    /// analysis times of a case are 2 plus a branching process's splits, 1/(1 − 2p) − 1 = 0.10 in the mean; the cases'
    /// mean lies within z of its standard error of that at most, and no case splits more than four times. The whole run
    /// is read at its two ends and three points.
    #[test]
    fn flat_noise_is_not_halved() {
        let flat: [&dyn Fn(f64) -> f64; 3] = [&|_| 0.3, &|_| 0.5, &|_| 12.0];
        let cases = 400;
        let mut splits = Vec::with_capacity(cases);
        let mut reads = 0;
        for case in 0..cases as u64 {
            let (ends, count) = analysed(&[], &mut noisy(&flat, 8, 0.05, 100 + case));
            splits.push((ends.len() - 2) as f64);
            reads += count;
        }
        let n = cases as f64;
        let mean = splits.iter().sum::<f64>() / n;
        let deviation = (splits.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (n - 1.0) / n).sqrt();
        let p = crate::estimate::tail_among(1.0);
        let expected = 1.0 / (1.0 - 2.0 * p) - 1.0;
        let most = splits.iter().copied().fold(0.0, f64::max);
        eprintln!("flat noise: {mean:.3} ± {deviation:.3} splits a case (at most {expected:.3} expected), {most} at the most; {:.2} candidates read a case", reads as f64 / n);
        assert!(mean <= expected + crate::units::CONFIDENCE * deviation && most <= 4.0, "{mean} ± {deviation} splits a case against {expected}; {most} at the most");
    }
}
