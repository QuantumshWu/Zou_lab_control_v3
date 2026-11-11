//! A read, its three outcomes, pays, counted work (SPEC §5).
//!
//! # A decision
//! Every decision is made of four things (SPEC §5.1): the current level, a more exact one, the output the two are
//! compared on, and the work. The two levels differ by d on the same trajectories or on the same sampled states, and
//! the current level's own error follows from it: where the finer level's parameter is the fraction r of the current
//! one's and the level's error goes as its parameter to the order q, e(h) − e(rh) = d gives e(h) = d/(1 − r^q)
//! (`richardson`). A read (`Read`) is an estimate of the error's size with its statistical error: a quantity x ≥ 0 that
//! grows with the level's parameter to some order -- the error's mean square over the outputs it is read at, of the
//! order 2q (estimate.rs `between`) -- estimated without bias, so that the read itself lies on either side of zero
//! where there is no error. Against the bound b that x may reach -- the square of the share σ of its target that the
//! level may take -- there are three outcomes and no fourth (`Read::outcome`):
//! - within, max(x, 0) + t·SE ≤ b: keep the level;
//! - beyond, x − t·SE > b: refine by the measured ratio at the read's order, h' = h (b/x)^{1/order} (`Read::refined`),
//!   the parameter at which the measured size meets the bound -- on a ladder of halvings, the first rung at or below it
//!   (`halved`);
//! - unreadable, neither: the far end (`Read::most`) is what the read does not exclude, the near end x − t·SE
//!   (`Read::least`) what it shows clearly.
//!
//! The far end is taken from zero where the read lies below it: a size is not negative, a read below zero is a read
//! of none, and a read of none excludes nothing smaller than its own interval -- so no read is within a bound that
//! its interval alone exceeds, however far below zero the noise has put it.
//!
//! t·SE is the interval at the confidence z over the looks the decision takes (estimate.rs `Spread::interval_over`).
//!
//! # Many reads
//! A decision that looks at a family of m reads, each against its own bound (`judge`; SPEC §5.1), refines where any of
//! them lies beyond, and so takes the largest of m noisy reads: beyond is judged with the family's interval (Šidák over
//! the m looks, so that the largest noisy read refines no more often than one look would), by the read that asks for the
//! finest level. Within is judged with each read's own interval, one look: every read has to lie within its bound by
//! itself, whatever the others are. Any other family is unreadable.
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

/// A read's outcome against a bound (module documentation).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Within,
    Beyond,
    Unreadable,
}

/// Richardson's factor 1/(1 − r^q) (module documentation): a level's own error over its difference with a finer level
/// whose parameter is `ratio` times its own, the error going as the parameter to `order`.
pub fn richardson(ratio: f64, order: f64) -> f64 {
    1. / (1. - ratio.powf(order))
}

/// The parameter a refinement by `factor` takes a level's `parameter` to on a ladder of halvings (module documentation):
/// the fewest whole halvings that bring it to the factor, one at least -- a refinement takes a finer level, and an
/// unreadable read (the factor 1) the next one.
pub fn halved(parameter: f64, factor: f64) -> f64 {
    parameter * 0.5_f64.powf((1. / factor).log2().ceil().max(1.))
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
    /// Within, beyond or unreadable against `bound` among `looks` reads (module documentation); a read that is not a
    /// number is unreadable.
    pub fn outcome(&self, bound: f64, looks: f64) -> Outcome {
        if self.most(looks) <= bound {
            Outcome::Within
        } else if self.least(looks) > bound {
            Outcome::Beyond
        } else {
            Outcome::Unreadable
        }
    }
    /// The factor on the level's parameter at which the measured size meets `bound` at the read's `order`,
    /// (b/x)^{1/order}.
    pub fn refined(&self, bound: f64, order: f64) -> f64 {
        (bound / self.value).powf(1. / order)
    }
}

/// A family's outcome (module documentation, # Many reads).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Judgement {
    /// Every read lies within its bound by itself.
    Within,
    /// Some read lies beyond its bound among them all: the least factor on the level's parameter at which a read beyond
    /// meets its bound (`Read::refined`).
    Beyond(f64),
    Unreadable,
}

/// The outcome of a family of reads, each with its bound, at the reads' `order` in the level's parameter (module
/// documentation, # Many reads).
pub fn judge(reads: &[(Read, f64)], order: f64) -> Judgement {
    let looks = reads.len() as f64;
    let beyond = reads.iter().filter(|(r, bound)| r.outcome(*bound, looks) == Outcome::Beyond).map(|(r, bound)| r.refined(*bound, order)).reduce(f64::min);
    match beyond {
        Some(factor) => Judgement::Beyond(factor),
        None if reads.iter().all(|(r, bound)| r.outcome(*bound, 1.) == Outcome::Within) => Judgement::Within,
        None => Judgement::Unreadable,
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

    /// The three outcomes at their edges (one look, unlimited groups: t = z), the unreadable one also for a read that is
    /// not a number and one on a single group. A read below zero -- an unbiased read of a size that is none -- is
    /// within where its interval is, and never beyond, however far below it lies.
    #[test]
    fn a_read_has_three_outcomes() {
        assert_eq!(read(0.5, 0.25).outcome(1., 1.), Outcome::Within);
        assert_eq!(read(0.5, 0.26).outcome(1., 1.), Outcome::Unreadable);
        assert_eq!(read(1.5, 0.24).outcome(1., 1.), Outcome::Beyond);
        assert_eq!(read(1.5, 0.25).outcome(1., 1.), Outcome::Unreadable);
        assert_eq!(read(f64::NAN, 0.).outcome(1., 1.), Outcome::Unreadable);
        assert_eq!(Read { value: 5., spread: Spread { error: 0.1, groups: 1. } }.outcome(1., 1.), Outcome::Unreadable);
        assert_eq!((read(-0.5, 0.5).outcome(1., 1.), read(-0.5, 0.51).outcome(1., 1.), read(-5., 0.1).outcome(1., 1.)), (Outcome::Within, Outcome::Unreadable, Outcome::Within));
        assert!((read(-0.5, 0.25).most(1.) - CONFIDENCE * 0.25).abs() < 1e-15 && (read(0.5, 0.25).most(1.) - (0.5 + CONFIDENCE * 0.25)).abs() < 1e-15);
        assert!((read(0.5, 0.25).least(1.) - (0.5 - CONFIDENCE * 0.25)).abs() < 1e-15 && (read(-0.5, 0.25).least(1.) - (-0.5 - CONFIDENCE * 0.25)).abs() < 1e-15);
        assert_eq!((read(f64::NAN, 0.).most(1.), Read { value: 1., spread: Spread::UNREAD }.least(1.)), (f64::INFINITY, f64::NEG_INFINITY));
    }

    /// Richardson: a halving at order 2 reads 3/4 of the coarse error, a quartering at order 1 too, an exact
    /// finer level all of it -- a halving's factors at the orders 2 and 4 are 4/3 and 16/15 to the bit, 2^q/(2^q − 1);
    /// refining by the measured ratio meets the bound: a size four times its bound takes half the parameter at order
    /// 2, a quarter at order 1 -- and an error's mean square sixteen times the share's square, of the order 4 at a
    /// scheme of the order 2, half the step. On a ladder of halvings a factor takes whole halvings, one at least.
    #[test]
    fn the_level_s_error_and_its_refinement_follow_its_order() {
        assert_eq!((richardson(0.5, 2.), richardson(0.5, 4.), richardson(0., 1.)), (4. / 3., 16. / 15., 1.));
        let e = read(0.2, 0.);
        assert!((e.refined(0.05, 2.) - 0.5).abs() < 1e-15 && (e.refined(0.05, 1.) - 0.25).abs() < 1e-15);
        assert!((read(0.04, 0.).refined(0.0025, 4.) - 0.5).abs() < 1e-15);
        assert_eq!((halved(1., 0.5), halved(1., 0.3), halved(1., 0.25), halved(1., 1.), halved(3., 2.)), (0.5, 0.25, 0.25, 0.5, 1.5));
    }

    /// A family's outcome (`judge`; one look's interval is z of the error on unlimited groups): within where every read
    /// lies within its own bound; a size four times its bound, clear among all the looks, is beyond with the factor ½
    /// at order 2, and of two reads beyond the finer level is taken; one read within and one whose interval straddles its
    /// bound are unreadable; a read that is clear at one look but not among the family's looks is unreadable too, not
    /// beyond; a read that is no number makes its family unreadable; no read at all is within.
    #[test]
    fn a_family_is_beyond_by_all_its_looks_and_within_by_each() {
        let quiet = (read(0., 0.), 1.);
        assert_eq!(judge(&[quiet, (read(0.5, 0.1), 1.)], 2.), Judgement::Within);
        assert_eq!(judge(&[quiet, (read(4., 0.01), 1.)], 2.), Judgement::Beyond(0.5));
        assert_eq!(judge(&[(read(4., 0.01), 1.), (read(0.5, 0.001), 0.03125)], 2.), Judgement::Beyond(0.25));
        assert_eq!(judge(&[quiet, (read(0.5, 1.), 1.)], 2.), Judgement::Unreadable);
        // 1.5 ± 0.24: beyond 1 at one look (z = 2), not at Šidák's interval over a thousand (about 4.08).
        let family: Vec<(Read, f64)> = std::iter::once((read(1.5, 0.24), 1.)).chain(std::iter::repeat_n(quiet, 999)).collect();
        assert_eq!((family[0].0.outcome(1., 1.), judge(&family, 2.)), (Outcome::Beyond, Judgement::Unreadable));
        assert_eq!(judge(&[quiet, (read(f64::NAN, 0.), 1.)], 2.), Judgement::Unreadable);
        assert_eq!(judge(&[], 2.), Judgement::Within);
    }

    /// A spend pays only for a saving beyond it and the saving's interval; a saving that is not a number pays nothing.
    #[test]
    fn a_spend_pays_beyond_the_saving_s_interval() {
        assert!(pays(2., read(3., 0.4)) && !pays(2., read(3., 0.6)));
        assert!(pays(0., read(1., 0.)) && !pays(0., read(0., 0.)) && !pays(1., read(f64::NAN, 0.)));
    }

    /// Over m looks of pure noise against a zero bound, some read lies beyond by chance no more often than one look
    /// does at z (Šidák). A read is beyond on one side, so each look refutes by chance with half its two-sided tail: of
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
            sidak += usize::from(draws.iter().any(|r| r.outcome(0., looks as f64) == Outcome::Beyond));
            single += usize::from(draws.iter().any(|r| r.outcome(0., 1.) == Outcome::Beyond));
        }
        let (p, n) = (0.04550026389635842_f64, repetitions as f64);
        let q = -((-p).ln_1p() / looks as f64).exp_m1();
        for (count, share) in [(sidak, 1. - (1. - 0.5 * q).powi(looks)), (single, 1. - (1. - 0.5 * p).powi(looks))] {
            let within = Spread { error: (share * (1. - share) / n).sqrt(), groups: f64::INFINITY }.interval_over(2.);
            assert!((count as f64 / n - share).abs() <= within, "{count} of {n} against {share} ± {within}");
        }
    }
}
