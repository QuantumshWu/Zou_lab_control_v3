//! A read, its three outcomes, pays, counted work (SPEC §5).
//!
//! # A decision
//! Every decision is made of four things (SPEC §5.1): the current level, a more exact one, the output the two are
//! compared on, and the work. A read is the difference d of the two levels' outputs on the same trajectories (common
//! random numbers, SPEC §2.4) or on the same sampled states, with its statistical error (`Read`; estimate.rs `paired`).
//! The current level's own error follows from it (`Read::level`): where the finer level's parameter is the fraction r
//! of the current one's and the level's error goes as its parameter to the order q, e(h) − e(rh) = d gives
//! e(h) = d/(1 − r^q) (Richardson); r = 0 where the finer level is the comparison's exact one. Against the share σ of
//! its target that the level may take there are three outcomes and no fourth (`Read::outcome`):
//! - within, |e| + t·SE ≤ σ: keep the level;
//! - beyond, |e| − t·SE > σ: refine by the measured ratio at the level's order, h' = h (σ/|e|)^{1/q} (`Read::refined`),
//!   the parameter at which the measured error meets the share;
//! - unreadable, neither: refine within the work bound; where it is spent, flag the output and book the readable bound
//!   |e| + t·SE (`Read::bound`) into its total.
//!
//! t·SE is the interval at the confidence z over the looks the decision takes (estimate.rs `Spread::interval_over`):
//! a decision that refines where any of m outputs reads beyond takes the largest of m noisy reads.
//!
//! A falsifier is a decision whose current level is already booked at a bound that the read can only refute (SPEC §5.2,
//! §5.3): beyond refines; within and unreadable leave the booked bound standing.
//!
//! # pays
//! A spend W for a predicted saving ΔC pays where ΔC − t·SE(ΔC) > W (`pays`): only a saving the read tells from nothing
//! moves the work, its error the jackknife of the difference of the two levels' costs.
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

/// A read (module documentation): a difference between two levels and its statistical error.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Read {
    pub value: f64,
    pub spread: Spread,
}

/// A read's outcome against a share (module documentation).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Within,
    Beyond,
    Unreadable,
}

impl Read {
    /// The current level's own error from its difference with a finer level whose parameter is `ratio` times its own,
    /// the error going as the parameter to `order` (module documentation): d/(1 − r^q), its error likewise.
    pub fn level(self, ratio: f64, order: f64) -> Read {
        let f = 1. / (1. - ratio.powf(order));
        Read { value: self.value * f, spread: Spread { error: self.spread.error * f, groups: self.spread.groups } }
    }
    /// The readable bound |e| + t·SE among `looks` reads.
    pub fn bound(&self, looks: f64) -> f64 {
        self.value.abs() + self.spread.interval_over(looks)
    }
    /// Within, beyond or unreadable against `share` among `looks` reads (module documentation); a read that is not a
    /// number is unreadable.
    pub fn outcome(&self, share: f64, looks: f64) -> Outcome {
        let t = self.spread.interval_over(looks);
        if self.value.abs() + t <= share {
            Outcome::Within
        } else if self.value.abs() - t > share {
            Outcome::Beyond
        } else {
            Outcome::Unreadable
        }
    }
    /// The factor on the level's parameter at which its measured error meets `share` at its `order`, (σ/|e|)^{1/q}.
    pub fn refined(&self, share: f64, order: f64) -> f64 {
        (share / self.value.abs()).powf(1. / order)
    }
}

/// Whether spending `spend` for the predicted `saving` pays (module documentation, # pays): saving − t·SE > spend.
pub fn pays(spend: Work, saving: Read) -> bool {
    saving.value - saving.spread.interval() > spend
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
    /// not a number and one on a single group.
    #[test]
    fn a_read_has_three_outcomes() {
        assert_eq!(read(0.5, 0.25).outcome(1., 1.), Outcome::Within);
        assert_eq!(read(-0.5, 0.26).outcome(1., 1.), Outcome::Unreadable);
        assert_eq!(read(1.5, 0.24).outcome(1., 1.), Outcome::Beyond);
        assert_eq!(read(1.5, 0.25).outcome(1., 1.), Outcome::Unreadable);
        assert_eq!(read(f64::NAN, 0.).outcome(1., 1.), Outcome::Unreadable);
        assert_eq!(Read { value: 5., spread: Spread { error: 0.1, groups: 1. } }.outcome(1., 1.), Outcome::Unreadable);
        assert!((read(-0.5, 0.25).bound(1.) - (0.5 + CONFIDENCE * 0.25)).abs() < 1e-15);
    }

    /// Richardson: a halving at order 2 reads 3/4 of the coarse error, a quarter tolerance at order 1 too, an exact
    /// finer level all of it; refining by the measured ratio meets the share: an error four times the share takes half
    /// the parameter at order 2, a quarter at order 1.
    #[test]
    fn the_level_s_error_and_its_refinement_follow_its_order() {
        let d = read(0.3, 0.03);
        for (ratio, order) in [(0.5, 2.), (0.25, 1.)] {
            let e = d.level(ratio, order);
            assert!((e.value - 0.4).abs() < 1e-15 && (e.spread.error - 0.04).abs() < 1e-15, "{e:?}");
        }
        assert_eq!(d.level(0., 2.), d);
        let e = read(-0.2, 0.);
        assert!((e.refined(0.05, 2.) - 0.5).abs() < 1e-15 && (e.refined(0.05, 1.) - 0.25).abs() < 1e-15);
    }

    /// A spend pays only for a saving beyond it and the saving's interval; a saving that is not a number pays nothing.
    #[test]
    fn a_spend_pays_beyond_the_saving_s_interval() {
        assert!(pays(2., read(3., 0.4)) && !pays(2., read(3., 0.6)));
        assert!(pays(0., read(1., 0.)) && !pays(0., read(0., 0.)) && !pays(1., read(f64::NAN, 0.)));
    }

    /// Over m looks of pure noise against a zero share, some read lies beyond by chance as often as one look does at z
    /// (Šidák): 4000 repetitions of 100 independent normal reads, the family-wise share within four of its standard
    /// errors of P(|N| > 2) = 0.0455, where one look's interval at every read would refute in 99 % of them.
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
        let (p, n) = (0.04550026389635842, repetitions as f64);
        let rate = sidak as f64 / n;
        assert!((rate - p).abs() <= 4. * (p * (1. - p) / n).sqrt(), "{rate}");
        assert!(single as f64 / n > 0.98, "{single}");
    }
}
