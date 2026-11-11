//! A read, its two ends, pays, counted work (SPEC §5).
//!
//! # A decision
//! Every decision is made of four things (SPEC §5.1): the current level, a more exact one, the output the two are
//! compared on, and the work. The two levels differ by d on the same trajectories or on the same sampled states, and
//! the current level's own error follows from it: where the finer level's parameter is the fraction r of the current
//! one's and the level's error goes as its parameter to the order q, e(h) − e(rh) = d gives e(h) = d/(1 − r^q)
//! (`richardson`). A read (`Read`) is an estimate of the error's size with its statistical error: a quantity x ≥ 0 that
//! grows with the level's parameter to some order -- the error's mean square over the outputs it is read at, of the
//! order 2q (estimate.rs `between`) -- estimated without bias, so that the read itself lies on either side of zero
//! where there is no error. A read has two ends at the confidence z:
//! - the far end max(x, 0) + t·SE (`Read::most`): what the read does not exclude. A level that is kept is booked at
//!   it, and what it is held to has to make room for it (the step: solve.rs `unexcluded`);
//! - the near end x − t·SE (`Read::least`): what the read shows clearly. A level is left for its next finer one on
//!   what its read shows, where no booking stands for the choice (the step's scout, solve.rs `clear`; an inherited
//!   read refuted, solve.rs `refuted`).
//! Between the two the read does not say, and a choice made there is made by the work each side is predicted to take
//! with the far end booked (the step's check, solve.rs `choose`): no bound that the read is passed or failed against,
//! so no read that the noise puts on one side of one.
//!
//! The far end is taken from zero where the read lies below it: a size is not negative, a read below zero is a read
//! of none, and a read of none excludes nothing smaller than its own interval -- however far below zero the noise has
//! put it.
//!
//! t·SE is the interval at the confidence z over the looks the decision takes (estimate.rs `Spread::interval_over`).
//!
//! # Many reads
//! A decision that looks at a family of m reads (SPEC §5.1; the step's check) and acts where any of them shows
//! something clearly takes the largest of m noisy reads: clear is the near end among the family's m looks (Šidák over
//! them, so that the largest noisy read acts no more often than one look would). What is not excluded is each read's
//! own statement, its far end at one look: every read's quantity is booked by itself, whatever the others are. A read
//! that is no number excludes nothing and shows nothing.
//!
//! # pays
//! A spend W for a predicted saving ΔC pays where ΔC − t·SE(ΔC) > W (`pays`; the left side is `assured`, by which savings
//! are ranked): only a saving the read tells from nothing moves the work, its error the jackknife of the difference of the
//! two levels' costs.
//!
//! # Work
//! Work is counted, not timed (SPEC §5.1): the real multiply-adds of the engine's algorithms (`Work`), each counted where
//! its algorithm is written; a complex multiply-add is four of them (`COMPLEX`). No unit cost is calibrated on a machine,
//! so a request takes the same decisions on any machine and at any load.
use crate::estimate::Spread;

/// Work in real multiply-adds (module documentation, # Work).
pub type Work = f64;
/// A complex multiply-add a + b c: (b_r c_r − b_i c_i) and (b_r c_i + b_i c_r) are four real multiplies, each added.
pub const COMPLEX: Work = 4.;

/// A read (module documentation): an estimate and its statistical error.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Read {
    pub value: f64,
    pub spread: Spread,
}

/// Richardson's factor 1/(1 − r^q) (module documentation): a level's own error over its difference with a finer level
/// whose parameter is `ratio` times its own, the error going as the parameter to `order`.
pub fn richardson(ratio: f64, order: f64) -> f64 {
    1. / (1. - ratio.powf(order))
}

impl Read {
    /// The read's far end among `looks` reads, max(x, 0) + t·SE: what it does not exclude (module documentation);
    /// unbounded for a read that is not a number or whose spread is unread.
    pub fn most(&self, looks: f64) -> f64 {
        let most = self.value.max(0.) + self.spread.interval_over(looks);
        if self.value.is_nan() || most.is_nan() { f64::INFINITY } else { most }
    }
    /// The read's near end among `looks` reads, x − t·SE: what it shows clearly; nothing (−∞) where its spread is
    /// unread, no number for a read that is none.
    pub fn least(&self, looks: f64) -> f64 {
        self.value - self.spread.interval_over(looks)
    }
}

/// What a predicted saving is at the least at the confidence z, saving − t·SE (module documentation, # pays): negative
/// for one that may be a loss, no number for a read that is none.
pub fn assured(saving: Read) -> f64 {
    saving.value - saving.spread.interval()
}

/// Whether spending `spend` for the predicted `saving` pays (module documentation, # pays): saving − t·SE > spend.
pub fn pays(spend: Work, saving: Read) -> bool {
    assured(saving) > spend
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::{Purpose, Stream};
    use crate::units::CONFIDENCE;

    fn read(value: f64, error: f64) -> Read {
        Read { value, spread: Spread { error, groups: f64::INFINITY } }
    }

    /// A read's two ends (one look, unlimited groups: t = z): the far end x + z·SE and the near end x − z·SE. A read
    /// below zero -- an unbiased read of a size that is none -- has its far end from zero, its own interval, however
    /// far below it lies, and shows nothing. A read that is not a number excludes nothing and shows nothing (no
    /// number is above any bound); one on a single group, whose spread is unread, likewise.
    #[test]
    fn a_read_has_two_ends() {
        assert!((read(-5., 0.1).most(1.) - CONFIDENCE * 0.1).abs() < 1e-15 && read(-5., 0.1).least(1.) < 0.);
        let (none, single) = (read(f64::NAN, 0.), Read { value: 5., spread: Spread { error: 0.1, groups: 1. } });
        assert!(none.most(1.) == f64::INFINITY && !(none.least(1.) > 0.) && single.most(1.) == f64::INFINITY && single.least(1.) == f64::NEG_INFINITY);
        assert!((read(-0.5, 0.25).most(1.) - CONFIDENCE * 0.25).abs() < 1e-15 && (read(0.5, 0.25).most(1.) - (0.5 + CONFIDENCE * 0.25)).abs() < 1e-15);
        assert!((read(0.5, 0.25).least(1.) - (0.5 - CONFIDENCE * 0.25)).abs() < 1e-15 && (read(-0.5, 0.25).least(1.) - (-0.5 - CONFIDENCE * 0.25)).abs() < 1e-15);
        assert_eq!((read(f64::NAN, 0.).most(1.), Read { value: 1., spread: Spread::UNREAD }.least(1.)), (f64::INFINITY, f64::NEG_INFINITY));
    }

    /// Richardson: a halving at order 2 reads 3/4 of the coarse error, a quartering at order 1 too, an exact
    /// finer level all of it -- a halving's factors at the orders 2 and 4 are 4/3 and 16/15 to the bit, 2^q/(2^q − 1).
    #[test]
    fn the_level_s_error_follows_its_order() {
        assert_eq!((richardson(0.5, 2.), richardson(0.5, 4.), richardson(0., 1.)), (4. / 3., 16. / 15., 1.));
    }

    /// A spend pays only for a saving beyond it and the saving's interval; a saving that is not a number pays nothing.
    #[test]
    fn a_spend_pays_beyond_the_saving_s_interval() {
        assert!(pays(2., read(3., 0.4)) && !pays(2., read(3., 0.6)));
        assert!(pays(0., read(1., 0.)) && !pays(0., read(0., 0.)) && !pays(1., read(f64::NAN, 0.)));
    }

    /// Over m looks of pure noise, some read's near end lies above zero by chance no more often than one look's does
    /// at z (Šidák). A near end is on one side, so each look shows something by chance with half its two-sided tail: of
    /// 4000 repetitions of 100 independent normal reads, the share with a read beyond the family's interval is
    /// 1 − (1 − q/2)¹⁰⁰ = 0.0230 at Šidák's tail q = 1 − (1 − p)^{1/100}, p = P(|N| > 2) = 0.0455 -- one look's own
    /// p/2 = 0.0228 --, where one look's interval at every read refutes in 1 − (1 − p/2)¹⁰⁰ = 90 % of them; each share
    /// a count's, with the standard error √(s(1 − s)/4000), within its interval at the confidence z over the two looks.
    #[test]
    fn many_looks_refute_by_chance_as_rarely_as_one() {
        let (repetitions, looks) = (4000, 100);
        let mut stream = Stream::new(11, Purpose::Initial, 0, 0);
        let (mut sidak, mut single) = (0, 0);
        for _ in 0..repetitions {
            let draws: Vec<Read> = (0..looks).map(|_| read(stream.normal(), 1.)).collect();
            sidak += usize::from(draws.iter().any(|r| r.least(looks as f64) > 0.));
            single += usize::from(draws.iter().any(|r| r.least(1.) > 0.));
        }
        let (p, n) = (0.04550026389635842_f64, repetitions as f64);
        let q = -((-p).ln_1p() / looks as f64).exp_m1();
        for (count, share) in [(sidak, 1. - (1. - 0.5 * q).powi(looks)), (single, 1. - (1. - 0.5 * p).powi(looks))] {
            let within = Spread { error: (share * (1. - share) / n).sqrt(), groups: f64::INFINITY }.interval_over(2.);
            assert!((count as f64 / n - share).abs() <= within, "{count} of {n} against {share} ± {within}");
        }
    }
}
