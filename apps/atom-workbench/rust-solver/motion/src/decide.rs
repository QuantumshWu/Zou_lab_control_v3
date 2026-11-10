//! A read, its three outcomes, pays, counted work (SPEC §5).
//!
//! # A decision
//! Every decision is made of four things (SPEC §5.1): the current level, a more exact one, the output the two are
//! compared on, and the work. A read is the difference d of the two levels' outputs on the same trajectories or on the
//! same sampled states, with its statistical error (`Read`). The current level's own error follows from it: where the
//! finer level's parameter is the fraction r of the current one's and the level's error goes as its parameter to the
//! order q, e(h) − e(rh) = d gives e(h) = d/(1 − r^q) (`richardson`). Against the share σ of
//! its target that the level may take there are three outcomes and no fourth (`Read::outcome`):
//! - within, |e| + t·SE ≤ σ: keep the level;
//! - beyond, |e| − t·SE > σ: refine by the measured ratio at the level's order, h' = h (σ/|e|)^{1/q} (`Read::refined`),
//!   the parameter at which the measured error meets the share -- on a ladder of halvings, the first rung at or below it
//!   (`halved`);
//! - unreadable, neither: refine within the work bound (on a ladder, the next rung); where it is spent, flag the output
//!   and book the readable bound |e| + t·SE (`Read::bound`) into its total.
//!
//! t·SE is the interval at the confidence z over the looks the decision takes (estimate.rs `Spread::interval_over`).
//!
//! # Many reads
//! A decision that looks at a family of m reads, each against its own share (`judge`; SPEC §5.1), refines where any of
//! them lies beyond, and so takes the largest of m noisy reads: beyond is judged with the family's interval (Šidák over
//! the m looks, so that the largest noisy read refines no more often than one look would), by the read that asks for the
//! finest level. Within is judged with each read's own interval, one look: every read has to lie within its share by
//! itself, whatever the others are. Any other family is unreadable.
//!
//! A falsifier is a decision whose current level is already booked at a bound that the read can only refute (SPEC §5.2,
//! §5.3): beyond refines; within and unreadable leave the booked bound standing. What it reads clearly of the level's
//! error is the near end of the read's interval, |e| − t·SE (`Read::clear`).
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
    /// The readable bound |e| + t·SE among `looks` reads; unbounded for a read that is not a number.
    pub fn bound(&self, looks: f64) -> f64 {
        let bound = self.value.abs() + self.spread.interval_over(looks);
        if bound.is_nan() { f64::INFINITY } else { bound }
    }
    /// What the read shows clearly of the error among `looks` reads, |e| − t·SE: nothing (−∞) where its spread is unread,
    /// no number for a read that is none.
    pub fn clear(&self, looks: f64) -> f64 {
        self.value.abs() - self.spread.interval_over(looks)
    }
    /// Within, beyond or unreadable against `share` among `looks` reads (module documentation); a read that is not a
    /// number is unreadable.
    pub fn outcome(&self, share: f64, looks: f64) -> Outcome {
        if self.bound(looks) <= share {
            Outcome::Within
        } else if self.clear(looks) > share {
            Outcome::Beyond
        } else {
            Outcome::Unreadable
        }
    }
    /// Whether a read taken on the share `part` of its trajectories could still lie within `share` on all of them: its
    /// near end (none below zero) with the interval the whole sample would leave -- a standard error goes as N^{−1/2},
    /// so the part's times √part. A read that is no number could not.
    pub fn in_reach(&self, share: f64, part: f64) -> bool {
        self.value.is_finite() && self.clear(1.).max(0.) + self.spread.interval() * part.sqrt() <= share
    }
    /// The factor on the level's parameter at which its measured error meets `share` at its `order`, (σ/|e|)^{1/q}.
    pub fn refined(&self, share: f64, order: f64) -> f64 {
        (share / self.value.abs()).powf(1. / order)
    }
}

/// A family's outcome (module documentation, # Many reads).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Judgement {
    /// Every read lies within its share by itself.
    Within,
    /// Some read lies beyond its share among them all: the least factor on the level's parameter at which a read beyond
    /// meets its share (`Read::refined`).
    Beyond(f64),
    Unreadable,
}

/// The outcome of a family of reads, each with its share, at the level's `order` (module documentation, # Many reads).
pub fn judge(reads: &[(Read, f64)], order: f64) -> Judgement {
    let looks = reads.len() as f64;
    let beyond = reads.iter().filter(|(r, share)| r.outcome(*share, looks) == Outcome::Beyond).map(|(r, share)| r.refined(*share, order)).reduce(f64::min);
    match beyond {
        Some(factor) => Judgement::Beyond(factor),
        None if reads.iter().all(|(r, share)| r.outcome(*share, 1.) == Outcome::Within) => Judgement::Within,
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

    /// Richardson: a halving at order 2 reads 3/4 of the coarse error, a quartering at order 1 too, an exact
    /// finer level all of it -- a halving's factors at the orders 2 and 4 are 4/3 and 16/15 to the bit, 2^q/(2^q − 1);
    /// refining by the measured ratio meets the share: an error four times the share takes half the parameter at order
    /// 2, a quarter at order 1. On a ladder of halvings a factor takes whole halvings, one at least.
    #[test]
    fn the_level_s_error_and_its_refinement_follow_its_order() {
        assert_eq!((richardson(0.5, 2.), richardson(0.5, 4.), richardson(0., 1.)), (4. / 3., 16. / 15., 1.));
        let e = read(-0.2, 0.);
        assert!((e.refined(0.05, 2.) - 0.5).abs() < 1e-15 && (e.refined(0.05, 1.) - 0.25).abs() < 1e-15);
        assert_eq!((halved(1., 0.5), halved(1., 0.3), halved(1., 0.25), halved(1., 1.), halved(3., 2.)), (0.5, 0.25, 0.25, 0.5, 1.5));
    }

    /// A family's outcome (`judge`; one look's interval is z of the error on unlimited groups): within where every read
    /// lies within its own share; an error four times its share, clear among all the looks, is beyond with the factor ½
    /// at order 2, and of two reads beyond the finer level is taken; one read within and one whose interval straddles its
    /// share are unreadable; a read that is clear at one look but not among the family's looks is unreadable too, not
    /// beyond; a read that is no number makes its family unreadable; no read at all is within.
    #[test]
    fn a_family_is_beyond_by_all_its_looks_and_within_by_each() {
        let quiet = (read(0., 0.), 1.);
        assert_eq!(judge(&[quiet, (read(0.5, 0.1), 1.)], 2.), Judgement::Within);
        assert_eq!(judge(&[quiet, (read(4., 0.01), 1.)], 2.), Judgement::Beyond(0.5));
        assert_eq!(judge(&[(read(4., 0.01), 1.), (read(-0.5, 0.001), 0.03125)], 2.), Judgement::Beyond(0.25));
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

    /// Over m looks of pure noise against a zero share, some read lies beyond by chance as often as one look does at z
    /// (Šidák): of 4000 repetitions of 100 independent normal reads, the share with a read beyond the family's interval
    /// is P(|N| > 2) = 0.0455, where one look's interval at every read refutes in 1 − (1 − 0.0455)¹⁰⁰ = 99 % of them --
    /// each share a count's, with the standard error √(p(1 − p)/4000), within its interval at the confidence z over the
    /// two looks.
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
        for (count, share) in [(sidak, p), (single, 1. - (1. - p).powi(looks))] {
            let within = Spread { error: (share * (1. - share) / n).sqrt(), groups: f64::INFINITY }.interval_over(2.);
            assert!((count as f64 / n - share).abs() <= within, "{count} of {n} against {share} ± {within}");
        }
    }
}
