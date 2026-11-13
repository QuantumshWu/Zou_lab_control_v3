//! Outputs, estimators, the jackknife, Student t, Wilson's unseen share (SPEC §1.7, §3).
//!
//! # The outputs are expectations
//! The model's dynamics is a piecewise-deterministic Markov process: the classical point's flow and kicks, ψ's
//! non-Hermitian evolution, a jump where ‖ψ‖² falls to its threshold, the absorbing loss. With A_k = 1{τ > t_k} (held at
//! t_k: not lost by then, SPEC §1.7) every displayed output at t_k is an expectation or a smooth function of expectations:
//! - the loss L_k = E[1 − A_k], never decreasing since A_k is not;
//! - each row as shown P_{s,k} = E[A_k ρ_s(t_k)] (ρ the readout; the app shows p(1 − L) = P), and the held atoms'
//!   populations p_{s,k} = P_{s,k}/(1 − L_k) (the contract's `populations`);
//! - per lab axis a the temperature T_a = (ħ²/m k_B) Var(κ_a), the variance over the held atoms and over the times of the
//!   window about t_k (`half_window`): E[A ⟨κ_a²⟩]/E[A] − (E[A ⟨κ_a⟩]/E[A])², ⟨·⟩ the atom's mean over the window (over
//!   its held part when it is lost within it). Over a whole period of a harmonic axis the window takes the oscillation's
//!   phase out of ⟨κ_a²⟩, so the spread between atoms is that of their energies. Over less it does not, whatever the
//!   outputs' spacing asks: an atom's κ_a² is E_a(1 + cos 2θ)/(ħ²/m) at its phase θ, the window's mean of it
//!   E_a(1 + sinc(ωW) cos 2θ̄)/(ħ²/m) over a width W, and N atoms' phases do not cancel -- the estimate breathes at 2ω
//!   about its mean by its statistics, an rms of |sinc ωW|/√N of T for a thermal axis (E_a exponential), while its
//!   variance is (1 + sinc²ωW)/N against the energies' 1/N. A noise that looks like a motion, and twice the trajectories
//!   for the same error at W → 0 (measured, R1's tweezer at 1 µs outputs, four seeds: 7.9–8.7 % rms at 2ω on 200
//!   atoms, 1.9–2.4 % on 3200, 0.3 % over whole periods; no part of it common to the seeds);
//! - n̄_a = (ħ/m ω_a) E[A ⟨κ_a²⟩]/E[A] − ½ (over a period of a harmonic axis the kinetic energy is half the axis's energy,
//!   E_a/ħω_a = n̄_a + ½) and the mean velocity (ħ/m) E[A ⟨κ_a⟩]/E[A].
//!
//! # Estimators
//! Every trajectory runs to the last output and through its window, so an output's estimate is a sample mean over all
//! the trajectories: unbiased, for a given number of them, for the loss and the rows as shown, and, given the number n
//! of atoms held, for the held atoms' means. T's plug-in variance ȳ − x̄² (y = ⟨κ²⟩, x = ⟨κ⟩ over the held atoms) is low
//! by Var(x)/n, since E[x̄²] = μ_x² + Var(x)/n; with x̄² taken as its U-statistic ((Σx)² − Σx²)/(n(n − 1)) = x̄² − s²_x/n
//! the estimate is unbiased for n ≥ 2, and no less than the held atoms' mean of ⟨κ²⟩ − ⟨κ⟩² ≥ 0.
//!
//! # What is counted
//! A trajectory counts at an output where it has an outcome there under the run's model: its atom is held, or was lost
//! by then. One that ended before the output without an outcome (`Fate::Failed`) counts at no output after its end:
//! - the scheme refused its step at every length, or the model gave a jump no photon (adapt.rs): a numerical failure;
//! - a step of it was not proven (step.rs, # The domain): the attempt's model is no model of its path, and the attempt
//!   ends with the reach grown to hold it (solve.rs, # The reach's atoms). Such trajectories are in no final result,
//!   only in the results sent while their attempt runs -- and there a whole hand-out can end so: the scout's eleven
//!   atoms of a scene that heats them out of the first reaches (S2Q at 79cb5075: every one not proven by 1.1 ms, and
//!   the checkpoint made of them held 79 of 101 outputs no trajectory counted at; measured).
//!
//! The estimates are over the trajectories counted. What the others would show is unknown, and a probability errs by
//! at most its share: with u the share of an output's trajectories not counted (`Estimate::uncounted`), a row as shown
//! or the loss over the counted ones, P_c, and the whole ensemble's, P = (1 − u)P_c + u P_u with P_u ∈ [0, 1] unknown,
//! differ by u|P_u − P_c| ≤ u -- booked whole on what is shown (solve.rs `accounts`). Of T nothing bounds the part the
//! held atoms of unknown fate make: an atom's kinetic energy after its last state is bounded by nothing the run read,
//! so T has no account where some trajectory is not counted. Where none is counted (u = 1) nothing is estimated: the
//! loss stands at the output before it, what is shown is booked whole, and the result says so (solve.rs `assemble`).
//!
//! # Errors
//! The trajectories come in groups, a Latin hypercube inside each and the groups independent (ensemble.rs): an
//! estimate's standard error is its delete-one-group jackknife, SE² = ((g − 1)/g) Σ_i (θ_(−i) − θ̄)², each quantity's
//! own -- the rows as shown jackknifed as P itself, not (1 − L) times the held atoms' error, which leaves out the loss's
//! variance and its covariance (Var(pq) ≠ q² Var(p)). Its interval at the confidence z is Student's t at one degree of
//! freedom fewer than the groups that read it, at the normal's two-sided tail beyond z, times the standard error
//! (`Spread::interval`; Tukey's jackknife t, exact for normal group means): every group reads the loss and the rows as
//! shown, the groups holding an atom read the held atoms' quantities.
//!
//! # The step's error
//! The scheme is of the order p (units.rs `ORDER`): an output's step error at the step τ is e(τ) = C τ^p + O(τ^{p+1}), so
//! the same output at twice the step differs by Q(2τ) − Q(τ) = (2^p − 1) e(τ) to the leading order, and
//! e(τ) = [Q(2τ) − Q(τ)]/(2^p − 1) (decide.rs `richardson` at the ratio ½, less the level's own share). `between` takes it
//! from two ensembles of the same trajectories -- the same numbers, so the same initial atoms and the same streams -- at
//! τ and at 2τ: the loss and each row as shown as they are, the total T relative to the finer level's and weighed by the
//! finer level's held share 1 − L -- as T's account weighs it (solve.rs `accounts`): T describes the held atoms, and its
//! error counts at an output by their share --, over T's allowance there (`allowance`, the finer level's: solve.rs,
//! # T's allowance), so that one read holds every output to its own target. Unweighed, an output where nearly every atom is lost reads T on the few
//! groups that still hold one, at a relative noise the target weighs by almost nothing, and that noise is the read's
//! (measured, S2Q at 0.44 µs: T's read on 5 to 9 of its groups at far ends of 2 to 21 times the bound, from 88 to 5632
//! pairs alike).
//!
//! What it reads of a quantity is the error's size over the run: its mean square over the K outputs after t = 0,
//! M = (1/K) Σ_k e_k² -- the square of the mean error plus the error's variance over the outputs, so an error that
//! changes its sign in time (an oscillating population's) counts by its size, where its signed mean reads nothing.
//! The square of an estimate holds the estimate's own variance, a floor that no error is needed for. The groups are
//! independent, and a product of two different groups' values holds none: with x_ik group i's own value of e_k,
//!   M̂ = (1/K) Σ_k [(Σ_i x_ik)² − Σ_i x_ik²]/(G(G − 1)),
//! the mean over the outputs of the mean product of two different groups (per output the estimate's square less its
//! squared standard error), is M in the mean whatever the noise, and lies on either side of zero where there is no
//! error. A group's own value is the jackknife's, G times the whole ensemble's e_k less G − 1 times the e_k without the
//! group: the group's own difference where the quantity is a mean over the trajectories (the loss, a row as shown),
//! and for T, a ratio, its linear part about the pooled levels. The error of M̂ is its delete-one-group jackknife --
//! without group i the same mean product over the other groups, [(S_k − x_ik)² − (Q_k − x_ik²)]/((G − 1)(G − 2)) from
//! the outputs' sums S_k and Q_k of the values and of their squares --, each group left out of both levels and of
//! every output at once, so that the outputs' correlations in time are in it: what the two levels share (the initial
//! atoms, as long as the motion remembers them) cancels, and where they have decorrelated it is the two levels' own
//! errors. Its variance has two parts, (4/G)⟨μ, Cμ⟩ + (2/(G(G − 1))) tr C² (μ the errors over the outputs, C their
//! noise's covariance over one group, both in the outputs' mean product): the first is there with an error and falls
//! as 1/G, the second is the floor's own scatter and falls as 1/G²; the jackknife counts the second twice
//! (2(G − 1)/(G − 2) times: Efron and Stein), so it is high where there is no error, and never low.
//!
//! Nothing is assumed of how an error is carried from a step to an output: the two ensembles are carried by the
//! dynamics itself. What the mean square does not single out: an error confined to a short stretch of the run, which
//! the run's length dilutes. Against the signed mean, for an error constant in time: where the noise has K' independent
//! stretches over the run and σ at an output, the mean resolves t σ/√K' and the mean square's rms √(t·SE) about
//! (2/K')^{1/4} √t σ -- the same at one stretch, K'^{1/4} more at many; the mean's noise is the noise's slowest mode,
//! which the mean square weighs as one mode among K'.
//!
//! # Many looks
//! A decision that acts where any of m reads lies beyond its interval (decide.rs) takes the largest of m noisy reads: at
//! one look's tail p = P(|N| > z) each, some read of m independent ones lies beyond by chance with probability
//! 1 − (1 − p)^m. Each read's interval is taken at Šidák's tail 1 − (1 − p)^{1/m} (`Spread::interval_over`), which
//! keeps that probability at p -- exactly for independent reads, and at most p for the correlated reads of outputs that
//! share their trajectories (Šidák's inequality for the centred multivariate normal).
//!
//! # What no trajectory saw
//! A share that n of N independent draws met is bounded at the confidence z by the far end of Wilson's score interval
//! (`wilson`); at a count of zero, z²/(N + z²) (`unseen`): one function for every such share -- an output's statistics,
//! whose jackknife reads nothing of a loss or a state no trajectory reached, and the reach's unsampled share.
use crate::analysis::Reading;
use crate::decide::{richardson, Read};
use crate::request::STATES;
use crate::special::upper;
use crate::units::{CONFIDENCE, KT_PER_UK, ORDER, SUB_DOPPLER_RECOILS, TEMPERATURE_TARGET};
use rayon::prelude::*;
use std::borrow::Borrow;
use std::f64::consts::{PI, TAU};

/// The far end of Wilson's score interval at confidence z for a count n in N independent draws,
/// (n + z²/2 + z √(n(N − n)/N + z²/4))/(N + z²): the largest share the draws leave plausible (module documentation).
pub fn wilson(count: f64, draws: f64) -> f64 {
    let z = CONFIDENCE;
    let spread = if draws > 0. { count * (draws - count) / draws } else { 0. };
    (count + 0.5 * z * z + z * (spread + 0.25 * z * z).sqrt()) / (draws + z * z)
}

/// Wilson's far end at a count of zero, z²/(N + z²): the largest share that N draws leave unseen.
pub fn unseen(draws: f64) -> f64 {
    wilson(0., draws)
}

/// The draws that leave `share` unseen: the N with z²/(N + z²) = share (`unseen`), z²(1/share − 1).
pub fn draws_for(share: f64) -> f64 {
    CONFIDENCE * CONFIDENCE * (1. / share - 1.)
}

/// The probability that `count` or more of `draws` independent draws meet an event of the probability `share`: the
/// binomial's upper tail, 1 − Σ_{k < n} C(N, k) p^k (1 − p)^{N − k}, its terms one from the one before
/// (× (N − k)/(k + 1) · p/(1 − p)) -- exact at any count, where a normal's or Wilson's interval is not at a count of
/// one or two. A claim that a share is at most p is refuted by a count whose tail is below the test's (`tail_among`).
pub fn at_least(count: f64, draws: f64, share: f64) -> f64 {
    if share >= 1. {
        return 1.;
    }
    let (mut term, mut below) = ((1. - share).powf(draws), 0.);
    for k in 0..count as u64 {
        below += term;
        term *= (draws - k as f64) / (k + 1) as f64 * share / (1. - share);
    }
    (1. - below).clamp(0., 1.)
}

/// The two-sided tail at the confidence z of one look among `looks` (module documentation, # Many looks): Šidák's
/// 1 − (1 − p)^{1/m}, p = P(|N| > z). A one-sided test takes half of it.
pub fn tail_among(looks: f64) -> f64 {
    -((-normal_tail(CONFIDENCE)).ln_1p() / looks).exp_m1()
}

/// The half width of T's window on a lab axis about an output at t (SPEC §1.7): half of one trap period of the axis,
/// 2π/ω at its frequency ω > 0, cut at t = 0 so that it stays centred there; whatever the outputs' spacing (module
/// documentation): outputs closer than a period have overlapping windows. A window reaches no further than twice its
/// output's time, so the run goes on past its last output by half a period, and by no more than its own length
/// (solve.rs `Plan`).
pub fn half_window(frequency: f64, t: f64) -> f64 {
    (0.5 * TAU / frequency).min(t)
}

/// n̄ of a lab axis of frequency ω > 0 (rad/µs; module documentation): (ħ/m)Σ⟨κ²⟩ (rad/µs) summed over `atoms` atoms, over
/// their number and ω, less ½.
fn occupation(kinetic: f64, atoms: f64, frequency: f64) -> f64 {
    kinetic / (atoms * frequency) - 0.5
}

/// A quantity's statistical error: its jackknife standard error over the groups behind its estimate and the number of
/// groups that read it (module documentation, # Errors).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spread {
    pub error: f64,
    pub groups: f64,
}
impl Spread {
    /// An error read on fewer than two groups: unknown.
    pub const UNREAD: Spread = Spread { error: f64::INFINITY, groups: 0. };
    /// Whether it is read: a variance over independent groups has one degree of freedom fewer than the groups.
    pub fn read(&self) -> bool {
        self.groups >= 2. && self.error.is_finite()
    }
    /// Its interval at the confidence z for one look (`interval_over`).
    pub fn interval(&self) -> f64 {
        self.interval_over(1.)
    }
    /// Its interval at the confidence z among `looks` reads, one at least (module documentation, # Many looks): the
    /// quantile at Šidák's tail of Student's t at the groups that read it less one, the normal's on unlimited groups, times
    /// its standard error; unbounded when unread.
    pub fn interval_over(&self, looks: f64) -> f64 {
        if !self.read() {
            return f64::INFINITY;
        }
        if self.error == 0. {
            return 0.;
        }
        let freedom = (self.groups < f64::INFINITY).then(|| (self.groups.floor() - 1.) as u64);
        quantile(tail_among(looks), freedom) * self.error
    }
}

/// The two-sided quantile at the tail p: the q with P(|t_ν| > q) = p for Student's t at ν ≥ 1 degrees of freedom (`None`:
/// the normal), by bisection on the tail to the last bit; kept per thread, a pure function of both.
pub(crate) fn quantile(tail: f64, freedom: Option<u64>) -> f64 {
    thread_local! {
        static KEPT: std::cell::RefCell<std::collections::HashMap<(u64, Option<u64>), f64>> = Default::default();
    }
    let beyond = |q: f64| match freedom {
        Some(nu) => student_tail(q, nu),
        None => normal_tail(q),
    };
    KEPT.with(|kept| {
        *kept.borrow_mut().entry((tail.to_bits(), freedom)).or_insert_with(|| {
            let (mut low, mut high) = (0., 1.);
            while beyond(high) > tail {
                low = high;
                high *= 2.;
            }
            loop {
                let m = 0.5 * (low + high);
                if !(m > low && m < high) {
                    break high;
                }
                if beyond(m) > tail {
                    low = m;
                } else {
                    high = m;
                }
            }
        })
    })
}

/// The normal's two-sided tail beyond z, P(|N| > z) = erfc(z/√2) = Q(½, z²/2), the regularized upper incomplete gamma
/// function at ½ (special.rs).
fn normal_tail(z: f64) -> f64 {
    upper(0.5, 0.5 * z * z)
}

/// Student's t at ν (whole) degrees of freedom: its two-sided tail beyond q, 1 − A(q|ν), A by the finite sums of
/// Abramowitz and Stegun 26.7.3–4 in θ = atan(q/√ν): (2/π)(θ + sin θ (cos θ + (2/3) cos³θ + … + (2·4…(ν−3))/(3·5…(ν−2))
/// cos^(ν−2) θ)) at odd ν (θ alone at one), sin θ (1 + (1/2) cos²θ + … + (1·3…(ν−3))/(2·4…(ν−2)) cos^(ν−2) θ) at even.
fn student_tail(q: f64, nu: u64) -> f64 {
    let theta = (q / (nu as f64).sqrt()).atan();
    let (s, c) = theta.sin_cos();
    let c2 = c * c;
    let a = if nu % 2 == 1 {
        let (mut term, mut sum, mut k) = (c, if nu > 1 { c } else { 0. }, 1);
        while 2 * k + 1 < nu {
            term *= c2 * (2 * k) as f64 / (2 * k + 1) as f64;
            sum += term;
            k += 1;
        }
        2. / PI * (theta + s * sum)
    } else {
        let (mut term, mut sum, mut k) = (1., 1., 1);
        while 2 * k < nu {
            term *= c2 * (2 * k - 1) as f64 / (2 * k) as f64;
            sum += term;
            k += 1;
        }
        s * sum
    };
    1. - a
}

/// An atom held at an output: its readout's populations (SPEC §1.7) and per lab axis its means of κ and κ² over the
/// output's window (1/µm, 1/µm²); and what it is against what the model was judged for (SPEC §1.8): `unresolved` -- a
/// step of its trajectory, at the output or before it, did not resolve its path at its speed (step.rs
/// `Trajectory::unresolved`) --, else whether at the output's instant it lies `outside` the reach; and whether it is
/// `leaving`: its prepared energy at the output's instant at the potential's far value or above -- held in the trap's
/// region, bound no more, no draw of the bound atoms' distribution a reach is made for (solve.rs, # The reach's atoms).
#[derive(Clone, Debug, PartialEq)]
pub struct Held {
    pub rows: [f64; STATES],
    pub kappa: [f64; 3],
    pub kappa2: [f64; 3],
    pub outside: bool,
    pub unresolved: bool,
    pub leaving: bool,
}

/// One trajectory at one output, what became of it by then: its atom is held; it was lost (SPEC §1.7); or it ended
/// before the output with no outcome -- the scheme could not advance its state (adapt.rs), or a step of it was not
/// proven (solve.rs `run`) --: the trajectory is not counted at the output, and its share is booked (module
/// documentation, # What is counted).
#[derive(Clone, Debug, PartialEq)]
pub enum Fate {
    Held(Held),
    Lost,
    Failed,
}

/// One output's sums over a group's trajectories: the number counted and of the others (`Fate::Failed`), the atoms held, and over the held
/// atoms Σρ_s, Σ⟨κ⟩, Σ⟨κ⟩² (for the U-statistic, module documentation) and Σ⟨κ²⟩ per lab axis; of the held atoms
/// those outside the reach and those unresolved (`Held`), each kind's number and its Σ_a ⟨κ_a²⟩ over the three axes; and
/// the leaving atoms' part of the held atoms' motion (`Leaving`).
#[derive(Clone, Debug, PartialEq)]
pub struct Sums {
    pub trajectories: f64,
    pub failed: f64,
    pub held: f64,
    pub rows: [f64; STATES],
    pub kappa: [f64; 3],
    pub kappa_squared: [f64; 3],
    pub kappa2: [f64; 3],
    pub outside: [f64; 2],
    pub unresolved: [f64; 2],
    pub leaving: Leaving,
}

/// Of a set of sums' held atoms the leaving ones (`Held::leaving`): their number, those of them outside the reach, and
/// their Σ⟨κ⟩, Σ⟨κ⟩² and Σ⟨κ²⟩ per lab axis -- the bound atoms' motion and their count outside the reach are the held
/// atoms' less this (`bound_motion`, `Sums::bound_outside`).
#[derive(Clone, Debug, PartialEq)]
pub struct Leaving {
    pub held: f64,
    pub outside: f64,
    pub kappa: [f64; 3],
    pub kappa_squared: [f64; 3],
    pub kappa2: [f64; 3],
}

impl Sums {
    pub const ZERO: Sums = Sums { trajectories: 0., failed: 0., held: 0., rows: [0.; STATES], kappa: [0.; 3], kappa_squared: [0.; 3], kappa2: [0.; 3], outside: [0.; 2], unresolved: [0.; 2], leaving: Leaving { held: 0., outside: 0., kappa: [0.; 3], kappa_squared: [0.; 3], kappa2: [0.; 3] } };
    /// One trajectory at the output.
    pub fn add(&mut self, fate: &Fate) {
        if *fate == Fate::Failed {
            self.failed += 1.;
            return;
        }
        self.trajectories += 1.;
        if let Fate::Held(h) = fate {
            self.held += 1.;
            for (s, r) in self.rows.iter_mut().zip(&h.rows) {
                *s += r;
            }
            for a in 0..3 {
                self.kappa[a] += h.kappa[a];
                self.kappa_squared[a] += h.kappa[a] * h.kappa[a];
                self.kappa2[a] += h.kappa2[a];
            }
            let kinetic: f64 = h.kappa2.iter().sum();
            for (sum, _) in [(&mut self.outside, h.outside), (&mut self.unresolved, h.unresolved)].into_iter().filter(|x| x.1) {
                sum[0] += 1.;
                sum[1] += kinetic;
            }
            if h.leaving {
                let l = &mut self.leaving;
                l.held += 1.;
                l.outside += f64::from(u8::from(h.outside));
                for a in 0..3 {
                    l.kappa[a] += h.kappa[a];
                    l.kappa_squared[a] += h.kappa[a] * h.kappa[a];
                    l.kappa2[a] += h.kappa2[a];
                }
            }
        }
    }
    /// Another's sums added in with the sign (−1: left out).
    fn combine(&mut self, o: &Sums, sign: f64) {
        self.trajectories += sign * o.trajectories;
        self.failed += sign * o.failed;
        self.held += sign * o.held;
        for (s, r) in self.rows.iter_mut().zip(&o.rows) {
            *s += sign * r;
        }
        for a in 0..3 {
            self.kappa[a] += sign * o.kappa[a];
            self.kappa_squared[a] += sign * o.kappa_squared[a];
            self.kappa2[a] += sign * o.kappa2[a];
        }
        for i in 0..2 {
            self.outside[i] += sign * o.outside[i];
            self.unresolved[i] += sign * o.unresolved[i];
        }
        let (l, m) = (&mut self.leaving, &o.leaving);
        l.held += sign * m.held;
        l.outside += sign * m.outside;
        for a in 0..3 {
            l.kappa[a] += sign * m.kappa[a];
            l.kappa_squared[a] += sign * m.kappa_squared[a];
            l.kappa2[a] += sign * m.kappa2[a];
        }
    }
    /// The bound atoms (`Held::leaving`: the held ones less the leaving ones) and how many of them lie outside the reach
    /// (`Held::outside`): what the reach's falsifier counts (solve.rs `outgrown`).
    pub fn bound_outside(&self) -> (f64, f64) {
        (self.outside[0] - self.leaving.outside, self.held - self.leaving.held)
    }
    /// The bound atoms' motion (`bound_motion`): the held atoms less the leaving ones, in the number held and in the
    /// motion's sums -- all T reads of a set of sums.
    fn bound(&self) -> Sums {
        let l = &self.leaving;
        Sums {
            held: self.held - l.held,
            kappa: std::array::from_fn(|a| self.kappa[a] - l.kappa[a]),
            kappa_squared: std::array::from_fn(|a| self.kappa_squared[a] - l.kappa_squared[a]),
            kappa2: std::array::from_fn(|a| self.kappa2[a] - l.kappa2[a]),
            leaving: Sums::ZERO.leaving,
            ..self.clone()
        }
    }
}

/// One group's sums at every output (module documentation, # Errors), and the photons its trajectories' jumps emitted
/// with their Σ k_φ² (step.rs `Trajectory::scattered`): T's semiclassical scale (`semiclassical`).
#[derive(Clone, Debug, PartialEq)]
pub struct Group {
    pub sums: Vec<Sums>,
    pub scattered: [f64; 2],
}
impl Group {
    pub fn new(outputs: usize) -> Group {
        Group { sums: vec![Sums::ZERO; outputs], scattered: [0.; 2] }
    }
    /// The group's further trajectories, after those it holds.
    pub fn absorb(&mut self, other: Group) {
        for (s, o) in self.sums.iter_mut().zip(&other.sums) {
            s.combine(o, 1.);
        }
        self.scatter(other.scattered);
    }
    /// A trajectory's photons (`Trajectory::scattered`).
    pub fn scatter(&mut self, scattered: [f64; 2]) {
        self.scattered = [self.scattered[0] + scattered[0], self.scattered[1] + scattered[1]];
    }
    /// The group's sums at its places `at`, in their order (solve.rs: a grid's analysis times).
    pub fn at(&self, at: &[usize]) -> Group {
        Group { sums: at.iter().map(|&k| self.sums[k].clone()).collect(), scattered: self.scattered }
    }
    /// One trajectory at every output, in order.
    pub fn add(&mut self, outputs: &[Fate]) {
        assert_eq!(outputs.len(), self.sums.len(), "a trajectory has every output");
        for (s, o) in self.sums.iter_mut().zip(outputs) {
            s.add(o);
        }
    }
}

/// What the estimates need of the model: ħ/m (µm²/µs) and the reported trap's frequency along each lab axis (rad/µs),
/// positive: the curvatures of a potential's minimum (model.rs `Trap::lab`).
#[derive(Clone, Copy, Debug)]
pub struct Scales {
    pub hbar_over_mass: f64,
    pub frequencies: [f64; 3],
}

/// T's semiclassical error scale T_sc (µK; solve.rs, # T's allowance) of the trajectories whose photons are `scattered`
/// (`Group::scattered`): max(`SUB_DOPPLER_RECOILS` T_rec, ħω̄/2k_B) -- k_BT_rec = (ħk)²/m at the photons' own mean k²,
/// the mean over the lab axes of the reported trap's zero point ħω_a/2k_B. No photon, no recoil part.
pub fn semiclassical(scattered: [f64; 2], scales: &Scales) -> f64 {
    let recoil = if scattered[0] > 0. { scales.hbar_over_mass * scattered[1] / scattered[0] / KT_PER_UK } else { 0. };
    let zero_point = scales.frequencies.iter().sum::<f64>() / (6. * KT_PER_UK);
    (SUB_DOPPLER_RECOILS * recoil).max(zero_point)
}

/// T's allowance at a total T (µK) of the semiclassical scale `scale` (µK), relative to T (solve.rs, # T's allowance):
/// δT/T = √(ε_T² + (T_sc/T)²); unbounded where T is none.
pub fn allowance(temperature: f64, scale: f64) -> f64 {
    if temperature > 0. { TEMPERATURE_TARGET.hypot(scale / temperature) } else { f64::INFINITY }
}

/// The groups' photons together (`Group::scattered`).
fn scattered<G: Borrow<Group>>(groups: &[G]) -> [f64; 2] {
    groups.iter().fold([0.; 2], |a, g| [a[0] + g.borrow().scattered[0], a[1] + g.borrow().scattered[1]])
}

/// The estimates at one output (module documentation): the loss, the held atoms' populations (a row as shown is its
/// population times 1 − L) and the rows' spreads as shown, the held atoms' motion where at least two are held; each
/// spread over the groups that read it, and the trajectories counted and held atoms behind them. Where no trajectory is
/// counted the loss is the output's before (module documentation, # What is counted), and nothing is read.
#[derive(Clone, Debug)]
pub struct Estimate {
    pub trajectories: f64,
    pub held: f64,
    pub loss: f64,
    pub loss_error: Spread,
    pub shown_errors: [Spread; STATES],
    pub populations: [f64; STATES],
    pub motion: Option<Motion>,
    /// The trajectories that ended before the output with no outcome (`Fate::Failed`): none of the above counts them.
    pub failed: f64,
    /// Their share of the output's trajectories, u (module documentation, # What is counted): what the estimates over
    /// the counted ones err by at most on what is shown; one where none is counted.
    pub uncounted: f64,
    /// T's allowance at the output, relative to its total T (`allowance`; solve.rs, # T's allowance): the target the
    /// total T is held to there -- `TEMPERATURE_TARGET` where no T is read.
    pub temperature_target: f64,
    /// Of the held atoms those outside the reach, and those unresolved (`Held`): each kind's share of the atoms and its
    /// share of their kinetic energy, Σ_a ⟨κ_a²⟩ of the kind over that of all -- what the kind carries of what is shown
    /// and of the total T (SPEC §1.8).
    pub outside: [f64; 2],
    pub unresolved: [f64; 2],
}
/// The held atoms' motion at an output where at least two are held: T per lab axis (µK), T_total = (T_x + T_y + T_z)/3
/// with its own jackknife (the axes' correlations in it), n̄ and the mean velocity (m/s).
#[derive(Clone, Copy, Debug)]
pub struct Motion {
    pub temperature: [f64; 3],
    pub temperature_errors: [Spread; 3],
    pub temperature_total: f64,
    pub temperature_total_error: Spread,
    pub occupation: [f64; 3],
    pub mean_velocity: [f64; 3],
}
impl Estimate {
    /// The jackknife interval of what is shown of the atom at the confidence z (module documentation): the largest of the
    /// rows as shown and of the loss.
    pub fn shown_interval(&self) -> f64 {
        self.shown_errors.iter().map(Spread::interval).fold(self.loss_error.interval(), f64::max)
    }
    /// The statistics' error of what is shown (ledger.rs `Account::statistics`): the jackknife interval, at least what the
    /// trajectories leave unseen (`unseen`) -- a loss or a state none of them reached reads no spread --, and at most
    /// one: what is shown is probabilities, none of which errs by more than its range (an interval unread, or wider,
    /// says no more than that).
    pub fn shown_statistics(&self) -> f64 {
        self.shown_interval().max(unseen(self.trajectories)).min(1.)
    }
    /// The interval of the total T (the mean over the lab axes) at the confidence z, relative to it; unbounded where T
    /// is not read, or is none (atoms that do not move: no error is relative to it).
    pub fn temperature_interval(&self) -> f64 {
        self.motion.as_ref().filter(|m| m.temperature_total > 0.).map_or(f64::INFINITY, |m| m.temperature_total_error.interval() / m.temperature_total)
    }
}

/// What the jackknife spreads, of one set of sums: the loss and the rows as shown, none where no trajectory is counted;
/// T per lab axis, none below two atoms held.
struct Moments {
    shown: Option<(f64, [f64; STATES])>,
    temperature: Option<[f64; 3]>,
}
/// T per lab axis (µK) of held-atom sums (module documentation, # Estimators); none below two atoms.
fn temperature(s: &Sums, hbar_over_mass: f64) -> Option<[f64; 3]> {
    let n = s.held;
    (n >= 2.).then(|| {
        std::array::from_fn(|a| {
            let square = (s.kappa[a] * s.kappa[a] - s.kappa_squared[a]) / (n * (n - 1.));
            (s.kappa2[a] / n - square) * hbar_over_mass / KT_PER_UK
        })
    })
}
fn moments(s: &Sums, hbar_over_mass: f64) -> Moments {
    let n = s.trajectories;
    Moments { shown: (n > 0.).then(|| (1. - s.held / n, s.rows.map(|r| r / n))), temperature: temperature(s, hbar_over_mass) }
}
/// The jackknife's standard error of the values with each group left out (module documentation, # Errors), read on
/// `groups`.
pub fn spread(leave: impl ExactSizeIterator<Item = f64> + Clone, groups: f64) -> Spread {
    let g = leave.len() as f64;
    let mean = leave.clone().sum::<f64>() / g;
    Spread { error: ((g - 1.) / g * leave.map(|x| (x - mean).powi(2)).sum::<f64>()).sqrt(), groups }
}

/// One output's jackknife over the groups of one or more levels run on the same trajectories (module documentation,
/// # Errors): each level's sums pooled over the groups in their order (so that nothing depends on the threads) and their
/// moments, and per group left out of every level the levels' moments; the groups, and those holding an atom at some
/// level, which read the held atoms' quantities.
struct Jackknife {
    all: Vec<Sums>,
    full: Vec<Moments>,
    leave: Vec<Vec<Moments>>,
    count: f64,
    holding: f64,
}
fn jackknife<G: Borrow<Group>>(levels: &[&[G]], k: usize, hm: f64) -> Jackknife {
    let all: Vec<Sums> = levels
        .iter()
        .map(|groups| {
            let mut s = Sums::ZERO;
            for g in *groups {
                s.combine(&g.borrow().sums[k], 1.);
            }
            s
        })
        .collect();
    let count = levels[0].len();
    let holding = (0..count).filter(|&i| levels.iter().any(|l| l[i].borrow().sums[k].held > 0.)).count() as f64;
    let leave = if count >= 2 {
        (0..count)
            .map(|i| {
                levels
                    .iter()
                    .zip(&all)
                    .map(|(l, s)| {
                        let mut rest = s.clone();
                        rest.combine(&l[i].borrow().sums[k], -1.);
                        moments(&rest, hm)
                    })
                    .collect()
            })
            .collect()
    } else {
        vec![]
    };
    Jackknife { full: all.iter().map(|s| moments(s, hm)).collect(), all, leave, count: count as f64, holding }
}
impl Jackknife {
    /// A quantity of the levels' moments spread over every group: unread where a group left out leaves some level no
    /// trajectory counted (`Moments::shown`).
    fn every(&self, f: impl Fn(&[Moments]) -> f64) -> Spread {
        if self.leave.is_empty() || !self.leave.iter().all(|m| counted(m)) {
            return Spread::UNREAD;
        }
        spread(self.leave.iter().map(|m| f(m)), self.count)
    }
    /// A quantity of the levels' temperatures spread over the groups holding an atom: unread where a group left out
    /// leaves some level fewer than two atoms.
    fn held(&self, f: impl Fn(&[[f64; 3]]) -> f64) -> Spread {
        let left: Option<Vec<Vec<[f64; 3]>>> = self.leave.iter().map(|m| m.iter().map(|x| x.temperature).collect()).collect();
        match left {
            Some(l) if !l.is_empty() => spread(l.iter().map(|t| f(t)), self.holding),
            _ => Spread::UNREAD,
        }
    }
}

/// Whether every level's moments count a trajectory.
fn counted(levels: &[Moments]) -> bool {
    levels.iter().all(|m| m.shown.is_some())
}
/// The loss of a level's moments that count a trajectory (`counted`).
fn loss(m: &Moments) -> f64 {
    m.shown.expect("moments that count a trajectory").0
}
/// A row as shown of a level's moments that count a trajectory (`counted`).
fn shown(m: &Moments, s: usize) -> f64 {
    m.shown.expect("moments that count a trajectory").1[s]
}

/// The held atoms' motion of one output's jackknife (`summarize`), where at least two are held.
fn motion(j: &Jackknife, scales: &Scales) -> Option<Motion> {
    let (all, hm) = (&j.all[0], scales.hbar_over_mass);
    j.full[0].temperature.map(|t| Motion {
        temperature: t,
        temperature_errors: std::array::from_fn(|a| j.held(|x| x[0][a])),
        temperature_total: t.iter().sum::<f64>() / 3.,
        temperature_total_error: j.held(|x| x[0].iter().sum::<f64>() / 3.),
        occupation: std::array::from_fn(|a| occupation(hm * all.kappa2[a], all.held, scales.frequencies[a])),
        mean_velocity: std::array::from_fn(|a| hm * all.kappa[a] / all.held),
    })
}

/// The bound atoms' motion at every output (`Held::leaving`; solve.rs, # The reach's atoms): as `summarize` reads the
/// held atoms', of the held atoms less the leaving ones -- what a reach's temperature is read against.
pub fn bound_motion<G: Borrow<Group> + Sync>(groups: &[G], scales: &Scales) -> Vec<Option<Motion>> {
    let bound: Vec<Group> = groups.iter().map(|g| Group { sums: g.borrow().sums.iter().map(Sums::bound).collect(), scattered: g.borrow().scattered }).collect();
    let outputs = bound.first().map_or(0, |g| g.sums.len());
    (0..outputs).into_par_iter().map(|k| motion(&jackknife(&[&bound[..]], k, scales.hbar_over_mass), scales)).collect()
}

/// The estimates at every output from the groups' sums (module documentation): where no trajectory is counted, the
/// loss of the output before (# What is counted) -- at t = 0 every trajectory is held.
pub fn summarize<G: Borrow<Group> + Sync>(groups: &[G], scales: &Scales) -> Vec<Estimate> {
    let outputs = groups.first().map_or(0, |g| g.borrow().sums.len());
    let hm = scales.hbar_over_mass;
    let scale = semiclassical(scattered(groups), scales);
    let made: Vec<(Option<f64>, Estimate)> = (0..outputs)
        .into_par_iter()
        .map(|k| {
            let j = jackknife(&[groups], k, hm);
            let (all, full) = (&j.all[0], &j.full[0]);
            let n = all.held;
            let motion = motion(&j, scales);
            let kinetic: f64 = all.kappa2.iter().sum();
            let shares = |kind: &[f64; 2]| [if n > 0. { kind[0] / n } else { 0. }, if kinetic > 0. { kind[1] / kinetic } else { 0. }];
            let estimate = Estimate {
                outside: shares(&all.outside),
                unresolved: shares(&all.unresolved),
                trajectories: all.trajectories,
                held: n,
                loss: 0.,
                loss_error: j.every(|m| loss(&m[0])),
                shown_errors: std::array::from_fn(|s| j.every(|m| shown(&m[0], s))),
                populations: if n > 0. { all.rows.map(|r| r / n) } else { [0.; STATES] },
                motion,
                failed: all.failed,
                uncounted: if all.trajectories > 0. { all.failed / (all.trajectories + all.failed) } else { 1. },
                temperature_target: motion.map_or(TEMPERATURE_TARGET, |m| allowance(m.temperature_total, scale)),
            };
            (full.shown.map(|(loss, _)| loss), estimate)
        })
        .collect();
    let mut before = 0.;
    made.into_iter()
        .map(|(loss, mut e)| {
            e.loss = loss.unwrap_or(before);
            before = e.loss;
            e
        })
        .collect()
}

/// The size of the step's error over the run as two ensembles measure it (module documentation, # The step's error):
/// per quantity the read of its mean square over the times after t = 0 -- the loss's and each row's as shown
/// (absolute, squared), and the total T's (the mean over the three lab axes, the T the target is on) relative to it and
/// weighed by the held share, as its account is, over T's allowance at the time (`allowance`: a share of the
/// allowance, the same at every time, is the step's share of T's target) -- none where no time reads it at both
/// levels. A read is of the mean square itself, unbiased, so it lies about zero, on either side, where there is no
/// error.
#[derive(Clone, Debug)]
pub struct Between {
    pub loss: Read,
    pub shown: [Read; STATES],
    pub temperature: Option<Read>,
}

/// The quantities whose step error `between` reads, in its order: the loss, each row as shown, the total T.
const STEPPED: usize = STATES + 2;

/// A quantity's step error at a time, of the finer and the coarser level's moments `m` (module documentation, # The
/// step's error): e = [Q(2τ) − Q(τ)]/(2^p − 1) of the loss (q = 0) and of each row as shown (q = 1 + s), and of the
/// total T (q = 1 + STATES) relative to the finer level's and weighed by its held share -- over T's allowance there
/// where `between` reads it. One definition for the step's read and for the analysis times' (`readings`).
fn step_error(m: &[Moments], q: usize) -> f64 {
    let share = richardson(0.5, ORDER as f64) - 1.;
    let total = |m: &Moments| m.temperature.map_or(f64::NAN, |t| t.iter().sum::<f64>());
    match q {
        0 => share * (loss(&m[1]) - loss(&m[0])),
        q if q <= STATES => share * (shown(&m[1], q - 1) - shown(&m[0], q - 1)),
        _ => share * (1. - loss(&m[0])) * (total(&m[1]) / total(&m[0]) - 1.),
    }
}

/// Whether a time's moments read a quantity of `step_error` at every level: the loss and the rows where every level
/// counts a trajectory, T where every level's total T is a positive number besides.
fn stepped(m: &[Moments], q: usize) -> bool {
    counted(m) && (q <= STATES || m.iter().all(|x| x.temperature.is_some_and(|t| t.iter().sum::<f64>() > 0.)))
}

/// Each time's share of the run in a mean over the times `times` (increasing, from t = 0): half the interval to each
/// neighbour, the last time's half its own -- the trapezoid's weights, the mean of a quantity the line through the
/// times holds (analysis.rs) over the run.
pub fn trapezoid(times: &[f64]) -> Vec<f64> {
    let last = times.len().saturating_sub(1);
    (0..times.len()).map(|k| 0.5 * (times[(k + 1).min(last)] - times[k.saturating_sub(1)])).collect()
}

/// The step's error of the level `fine`, from it and the level `coarse` at twice its step on the same trajectories,
/// group by group in the same order, both at the times `times` (module documentation, # The step's error; increasing,
/// from t = 0): each quantity's mean square over the times after t = 0 -- the loss's and the rows' over the times
/// where both levels, and both without any one group, count a trajectory (module documentation, # What is counted),
/// T's over those where they hold two atoms at least --, each time weighed by its share of the run (`trapezoid`: the
/// trapezoid's, so that the mean is the run's whatever the times' spacing). No number on a single group, or where no
/// time reads it; no spread on fewer than three groups (a mean square without a group needs two more).
pub fn between<G: Borrow<Group> + Sync>(fine: &[G], coarse: &[G], scales: &Scales, times: &[f64]) -> Between {
    assert_eq!(fine.len(), coarse.len(), "the two levels hold the same groups");
    let outputs = fine.first().map_or(0, |g| g.borrow().sums.len());
    let (hm, groups) = (scales.hbar_over_mass, fine.len());
    let g = groups as f64;
    let weights = trapezoid(times);
    let each: Vec<(f64, Jackknife)> = (1..outputs).into_par_iter().map(|k| (weights[k], jackknife(&[fine, coarse], k, hm))).collect();
    // A quantity's mean square over some times: at each, the groups' own values of it (g times the whole less g − 1
    // times the rest) and the mean of their products two by two, weighed by the time's share; the same without each
    // group, spread over the groups. Each time's values over its `over`: a quantity held to a target that differs
    // between the times, its own.
    let size = |q: usize, over: &dyn Fn(&Jackknife) -> f64| -> Option<Read> {
        let at: Vec<&(f64, Jackknife)> = each.iter().filter(|(_, j)| stepped(&j.full, q) && j.leave.iter().all(|l| stepped(l, q))).collect();
        if at.is_empty() {
            return None;
        }
        let (mut whole, mut without, mut own, mut weight) = (0., vec![0.; groups], vec![0.; groups], 0.);
        for (w, j) in &at {
            let unit = over(j);
            let full = step_error(&j.full, q) / unit;
            own.iter_mut().zip(&j.leave).for_each(|(x, rest)| *x = g * full - (g - 1.) * step_error(rest, q) / unit);
            let (sum, squares) = own.iter().fold((0., 0.), |(s, q), x| (s + x, q + x * x));
            whole += w * (sum * sum - squares);
            without.iter_mut().zip(&own).for_each(|(o, x)| *o += w * ((sum - x).powi(2) - (squares - x * x)));
            weight += w;
        }
        let read = if q <= STATES { g } else { at.iter().map(|(_, j)| j.holding).fold(f64::INFINITY, f64::min) };
        Some(Read {
            value: if groups >= 2 { whole / (weight * g * (g - 1.)) } else { f64::NAN },
            spread: if groups >= 3 { spread(without.iter().map(|w| w / (weight * (g - 1.) * (g - 2.))), read) } else { Spread::UNREAD },
        })
    };
    let unread = Read { value: f64::NAN, spread: Spread::UNREAD };
    let total = |m: &Moments| m.temperature.map_or(f64::NAN, |t| t.iter().sum::<f64>());
    // T's error is relative to the finer level's T and weighed by its held share, as T's account is, over T's allowance
    // at the time -- the finer level's (`allowance`, of the finer level's photons).
    let scale = semiclassical(scattered(fine), scales);
    Between {
        loss: size(0, &|_| 1.).unwrap_or(unread),
        shown: std::array::from_fn(|s| size(1 + s, &|_| 1.).unwrap_or(unread)),
        temperature: size(1 + STATES, &|j| allowance(total(&j.full[0]) / 3., scale)),
    }
}

/// What the analysis times are read on at the place k of the groups' sums (analysis.rs `Reading`): of the level
/// `fine`, its loss, each row as shown and T along each lab axis -- as at an output, the held atoms' over their windows
/// --; and where its companion `coarse` is given (the same groups in the same order, at twice the step: not empty),
/// each quantity's step error as `between` reads it (`step_error`). Each with its values without each group, the same
/// groups at every place; a quantity a place does not read (no trajectory counted, fewer than two atoms held, at some
/// level or without some group) is no number there, and splits nothing.
pub fn readings<G: Borrow<Group> + Sync>(fine: &[G], coarse: &[G], k: usize, scales: &Scales) -> Vec<Reading> {
    let groups = fine.len();
    let unread = || Reading { whole: f64::NAN, leave: vec![f64::NAN; groups] };
    let reading = |j: &Jackknife, read: &dyn Fn(&[Moments]) -> bool, f: &dyn Fn(&[Moments]) -> f64| {
        if read(&j.full) && j.leave.iter().all(|l| read(l)) { Reading { whole: f(&j.full), leave: j.leave.iter().map(|l| f(l)).collect() } } else { unread() }
    };
    let own = jackknife(&[fine], k, scales.hbar_over_mass);
    let mut out: Vec<Reading> = (0..1 + STATES).map(|q| reading(&own, &counted, &|m| if q == 0 { loss(&m[0]) } else { shown(&m[0], q - 1) })).collect();
    out.extend((0..3).map(|a| reading(&own, &|m| m[0].temperature.is_some(), &|m| m[0].temperature.map_or(f64::NAN, |t| t[a]))));
    if !coarse.is_empty() {
        let both = jackknife(&[fine, coarse], k, scales.hbar_over_mass);
        out.extend((0..STEPPED).map(|q| reading(&both, &|m| stepped(m, q), &|m| step_error(m, q))));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ensemble::GROUP;

    /// A uniform deviate in (0, 1) from a splitmix64 state (the tests' own stream).
    fn uniform(state: &mut u64) -> f64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (((z ^ (z >> 31)) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    /// A group of `GROUP` atoms at t = 0 and at the outputs after it: at output k its first `counts[k]` atoms show
    /// state 0 and the others state 1, every atom with ⟨κ²⟩ = `energy[k]`, ⟨κ⟩ = 0.
    fn showing(counts: &[usize], energy: &[f64]) -> Group {
        let mut g = Group::new(counts.len() + 1);
        for i in 0..GROUP as usize {
            let fates: Vec<Fate> = std::iter::once(Fate::Held(atom(0, 0., 1.))).chain(counts.iter().zip(energy).map(|(&c, &e)| Fate::Held(atom(usize::from(i >= c), 0., e)))).collect();
            g.add(&fates);
        }
        g
    }

    /// The step's size between two ensembles (`between`) on constructed levels. Three groups, two outputs after t = 0 at
    /// t = 1 and 2, of the run's shares 1 and ½ (the trapezoid's); a row shown by c of 8 atoms at the coarser level and
    /// by a at the finer has the group's own error (c − a)/8/3 (p = 2). With the groups' own errors x = (3, −1, 1)/24 at
    /// the first output and (2, 2, 2)/24 at the second, the mean square is the mean over the outputs, by their shares,
    /// of the mean of the products of two different groups, ((3·(−1) + 3·1 + (−1)·1)/3 + 4/2)/24²/1.5 -- at the first
    /// output below zero: the three errors are no more than their scatter --, and without a group the product of the
    /// other two; its spread is the jackknife of those. The other row is the first's mirror; the loss has none. T is 7
    /// at the finer level and 4 and 6 at the coarser on every atom: the groups agree, and its mean square is the mean of
    /// (4/7 − 1)²/9 and (6/7 − 1)²/9 by the same shares with no spread, over T's allowance at 7 µK squared.
    /// Two groups read a value and no spread; one reads none.
    #[test]
    fn the_step_s_size_is_the_mean_product_of_two_groups_errors() {
        let level = |counts: [[usize; 2]; 3], energy: [f64; 2]| -> Vec<Group> { counts.iter().map(|c| showing(c, &energy)).collect() };
        let (fine, coarse) = (level([[4, 4], [4, 4], [4, 4]], [7., 7.]), level([[7, 6], [3, 6], [5, 6]], [4., 6.]));
        let times = [0., 1., 2.];
        let read = between(&fine, &coarse, &UNIT, &times);
        let pairs = |x: [f64; 3]| [x[1] * x[2], x[0] * x[2], x[0] * x[1]];
        let (first, second) = (pairs([3., -1., 1.]), pairs([2., 2., 2.]));
        let without: Vec<f64> = (0..3).map(|i| (first[i] + 0.5 * second[i]) / 1.5 / 576.).collect();
        let whole = without.iter().sum::<f64>() / 3.;
        assert!(first.iter().sum::<f64>() < 0. && (read.shown[0].value - whole).abs() < 1e-17 && (read.shown[1].value - whole).abs() < 1e-17, "{:?} against {whole}", read.shown[0]);
        let expected = spread(without.iter().copied(), 3.);
        assert!((read.shown[0].spread.error - expected.error).abs() < 1e-17 && read.shown[0].spread.groups == 3., "{:?} against {expected:?}", read.shown[0].spread);
        assert!(read.loss.value == 0. && read.loss.spread.error == 0.);
        let t = read.temperature.expect("T is read at both levels");
        let allowed = allowance(7., semiclassical([0.; 2], &UNIT));
        assert!((t.value - ((4. / 7. - 1.0f64).powi(2) + 0.5 * (6. / 7. - 1.0f64).powi(2)) / 1.5 / 9. / (allowed * allowed)).abs() < 1e-15 && t.spread.error < 1e-15, "{t:?}");
        let two = between(&fine[..2], &coarse[..2], &UNIT, &times).shown[0];
        assert!((two.value - (-3. + 0.5 * 4.) / 1.5 / 576.).abs() < 1e-17 && !two.spread.read(), "{two:?}");
        assert!(between(&fine[..1], &coarse[..1], &UNIT, &times).shown[0].value.is_nan());
    }

    /// The step's size (`between`) against synthetic levels: 11 groups of 8 atoms, 8 outputs after t = 0; at each an
    /// atom shows state 0 with probability ½ at the finer level and ½ + 3 b_k at the coarser, drawn anew at each level,
    /// output and atom, so that the row's step error at p = 2 is b_k and its noise at an output is √(2·¼/88)/3 = 0.025,
    /// as large as the bias 0.03. Over 1000 repetitions:
    /// - a constant bias b: the read's mean meets b² within its interval at the confidence z;
    /// - a bias that alternates in sign from output to output: the same b², where the errors' signed mean over the
    ///   outputs (the mean of the two levels' rows' differences) reads zero within its interval;
    /// - no bias: the read's mean meets zero within its interval -- no floor: the differences' plain mean square is the
    ///   noise's own, 2·¼/88/9 = 6.3e-4, many intervals away --, a read lies above its own interval no more often than
    ///   the normal's one-sided tail, and the jackknife's variance is the read's own times 2(G − 1)/(G − 2) (the
    ///   jackknife counts twice the part of a product's variance that belongs to two groups together, the only part
    ///   where there is no error: Efron and Stein), within the relative error of a variance of 1000 draws at z, twice.
    /// With the bias the jackknife's variance lies between the reads' own and that multiple of it.
    #[test]
    fn the_step_s_size_reads_a_bias_of_either_sign_and_no_floor() {
        let (bias, groups, outputs, repetitions) = (0.03, 11usize, 8usize, 1000usize);
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let noise = 2. * 0.25 / (groups as f64 * GROUP as f64) / 9.;
        for (case, signs) in [[1.; 8], [1., -1., 1., -1., 1., -1., 1., -1.], [0.; 8]].into_iter().enumerate() {
            let (mut sum, mut squares, mut variances, mut signed, mut signed_squares, mut above) = (0., 0., 0., 0., 0., 0usize);
            for _ in 0..repetitions {
                let mut level = |shift: f64| -> Vec<Group> {
                    (0..groups)
                        .map(|_| {
                            let mut g = Group::new(outputs + 1);
                            for _ in 0..GROUP {
                                let fates: Vec<Fate> = std::iter::once(Fate::Held(atom(0, 0., 1.)))
                                    .chain(signs.iter().map(|sign| Fate::Held(atom(usize::from(uniform(&mut state) >= 0.5 + shift * sign), 0., 1.))))
                                    .collect();
                                g.add(&fates);
                            }
                            g
                        })
                        .collect()
                };
                let (fine, coarse) = (level(0.), level(3. * bias));
                let read = between(&fine, &coarse, &UNIT, &(0..=outputs).map(|k| k as f64).collect::<Vec<_>>()).shown[0];
                let (f, c) = (summarize(&fine, &UNIT), summarize(&coarse, &UNIT));
                let shown = |e: &Estimate| e.populations[0] * (1. - e.loss);
                let mean = (1..=outputs).map(|k| (shown(&c[k]) - shown(&f[k])) / 3.).sum::<f64>() / outputs as f64;
                sum += read.value;
                squares += read.value * read.value;
                variances += read.spread.error * read.spread.error;
                signed += mean;
                signed_squares += mean * mean;
                above += usize::from(read.least(1.) > 0.);
            }
            let n = repetitions as f64;
            let (mean, signed) = (sum / n, signed / n);
            let variance = squares / n - mean * mean;
            let within = Spread { error: (variance / n).sqrt(), groups: n }.interval();
            let truth = signs[0] * signs[0] * bias * bias;
            assert!((mean - truth).abs() <= within, "case {case}: the read's mean {mean} ± {within} against {truth}");
            let tolerance = 2. * CONFIDENCE * (2. / n).sqrt();
            let ratio = variances / n / variance;
            if case == 1 {
                let within = Spread { error: ((signed_squares / n - signed * signed) / n).sqrt(), groups: n }.interval();
                assert!(signed.abs() <= within && within < 0.1 * bias, "the signed mean {signed} ± {within} reads no bias");
            }
            if case == 2 {
                assert!((noise - mean).abs() > 10. * within, "the noise's own mean square {noise} is no floor of {mean} ± {within}");
                let tail = 0.5 * normal_tail(CONFIDENCE);
                let chance = Spread { error: (tail * (1. - tail) / n).sqrt(), groups: f64::INFINITY }.interval();
                assert!(above as f64 / n <= tail + chance, "{above} of {n} reads lie above their interval, against {tail} ± {chance}");
                let twice = 2. * (groups as f64 - 1.) / (groups as f64 - 2.);
                assert!((ratio / twice - 1.).abs() <= tolerance, "the jackknife's variance over the reads', {ratio}, against {twice}");
            } else {
                let twice = 2. * (groups as f64 - 1.) / (groups as f64 - 2.);
                assert!(ratio >= 1. - tolerance && ratio <= twice * (1. + tolerance), "case {case}: the jackknife's variance over the reads', {ratio}");
            }
        }
    }

    /// T's step size against synthetic levels of a known bias. Each atom has an exponential energy E (⟨κ²⟩ = E,
    /// ⟨κ⟩ = 0) at the finer level and (1 + 3b)(ρE + (1 − ρ)E') at the coarser, E' an independent one: the coarser T is
    /// (1 + 3b) times the finer in the mean, so the step's error at p = 2 is b, and the levels' correlation is set by
    /// ρ. T is a ratio, and the groups' own errors its linearization about the pooled levels (the jackknife's
    /// pseudo-values): over 1000 repetitions of 64 groups of 8 atoms the read's mean meets b² within its interval at
    /// the confidence z -- for independent levels (ρ = 0), where one read's noise √(2/512)/3 = 0.021 is the bias's
    /// size, and for levels that share their atoms (ρ = 0.95), where it is a twentieth of that. No trap and no photon: T's
    /// allowance is the target itself, and the read is b² over its square.
    #[test]
    fn the_temperature_s_step_size_is_the_paired_ratio_s() {
        let free = Scales { frequencies: [0.; 3], ..UNIT };
        let (bias, groups, repetitions) = (0.02, 64, 1000);
        let mut state = 0x2545_F491_4F6C_DD1D_u64;
        let mut exponential = || -uniform(&mut state).ln();
        for rho in [0., 0.95] {
            let (mut sum, mut squares) = (0., 0.);
            for _ in 0..repetitions {
                let (mut fine, mut coarse) = (vec![], vec![]);
                for _ in 0..groups {
                    let energies: Vec<(f64, f64)> = (0..GROUP).map(|_| (exponential(), exponential())).collect();
                    fine.push(group(energies.iter().map(|&(e, _)| Some(atom(0, 0., e))).collect()));
                    coarse.push(group(energies.iter().map(|&(e, other)| Some(atom(0, 0., (1. + 3. * bias) * (rho * e + (1. - rho) * other)))).collect()));
                }
                let read = between(&twice(&fine), &twice(&coarse), &free, &[0., 1.]).temperature.unwrap();
                sum += read.value;
                squares += read.value * read.value;
            }
            let n = repetitions as f64;
            let mean = sum / n;
            let within = Spread { error: ((squares / n - mean * mean) / n).sqrt(), groups: n }.interval();
            let expected = (bias / TEMPERATURE_TARGET).powi(2);
            assert!((mean - expected).abs() <= within, "ρ = {rho}: the read's mean {mean} ± {within} against {expected}");
        }
    }

    /// Groups of one output as groups of two: t = 0 and the output itself.
    fn twice(groups: &[Group]) -> Vec<Group> {
        groups.iter().map(|g| Group { sums: vec![g.sums[0].clone(), g.sums[0].clone()], scattered: g.scattered }).collect()
    }

    /// Student's t at ν degrees of freedom at the normal's two-sided tail beyond z.
    fn student(z: f64, nu: u64) -> f64 {
        quantile(normal_tail(z), Some(nu))
    }

    /// Student's t at the normal's tail: the tables' t at 97.5 % (z = Φ⁻¹(0.975)) for ν = 1, 10, 30, its tail there the
    /// normal's to round-off; at z = 2 the closed forms of one and two degrees of freedom, tan(π(1 − p)/2) and
    /// (1 − p)√(2/(1 − (1 − p)²)), p = P(|N| > 2); the normal's own z as ν grows, (z³ + z)/(4ν) apart (Cornish–Fisher).
    #[test]
    fn student_s_t_is_the_quantile_at_the_normal_s_tail() {
        let z = 1.959963984540054;
        assert!((normal_tail(z) - 0.05).abs() < 1e-16, "{}", normal_tail(z));
        for (nu, q) in [(1, 12.706205), (10, 2.228139), (30, 2.042272)] {
            let t = student(z, nu);
            assert!((t - q).abs() < 5e-7 && (student_tail(t, nu) - 0.05).abs() < 1e-15, "{nu}: {t} against {q}");
        }
        let p = normal_tail(2.);
        assert!((p - 0.04550026389635842).abs() < 1e-16, "{p}");
        assert!((student(2., 1) - (std::f64::consts::FRAC_PI_2 * (1. - p)).tan()).abs() < 1e-10);
        assert!((student(2., 2) - (1. - p) * (2. / (1. - (1. - p) * (1. - p))).sqrt()).abs() < 1e-13);
        let t = student(2., 10_000);
        assert!((t - 2. - 10. / 4e4).abs() < 1e-6, "{t}");
    }

    /// The quantile is the tail's inverse to the last bit (`quantile`), the normal's and Student's, from the centre to the
    /// far tail: the tail at q is no more than the tail asked, the tail at the number before q is more. The whole tail
    /// asks for 0 and gets the least positive number.
    #[test]
    fn the_quantile_is_the_tail_s_inverse_to_the_last_bit() {
        for freedom in [None, Some(1), Some(2), Some(7), Some(40)] {
            let beyond = |q: f64| freedom.map_or(normal_tail(q), |nu| student_tail(q, nu));
            for tail in [0.999, 0.5, 0.31731050786291415, 0.05, 0.04550026389635842, 2e-10] {
                let q = quantile(tail, freedom);
                assert!(beyond(q) <= tail && beyond(q.next_down()) > tail, "{freedom:?} at {tail}: {q}");
            }
        }
        let far = quantile(2e-300, None);
        assert!(normal_tail(far) <= 2e-300 && normal_tail(far.next_down()) > 2e-300, "{far}");
        assert_eq!(quantile(1., None), f64::from_bits(1));
    }

    /// The interval over many looks (Šidák): one look reads z itself on unlimited groups; m looks read the quantile q
    /// at the tail whose m-fold complement is one look's, (1 − P(|N| > q))^m = 1 − P(|N| > z), about 4.08 at m = 1000.
    #[test]
    fn many_looks_take_sidak_s_tail() {
        let s = Spread { error: 1., groups: f64::INFINITY };
        assert!((s.interval() - CONFIDENCE).abs() <= 4. * f64::EPSILON, "{}", s.interval());
        let q = s.interval_over(1000.);
        let p = normal_tail(CONFIDENCE);
        assert!(((1. - normal_tail(q)).powf(1000.) - (1. - p)).abs() < 1e-13 && (q - 4.08).abs() < 0.01, "{q}");
        assert!(Spread { error: 1., groups: 8. }.interval_over(1000.) > q, "Student's tail is the wider");
        assert_eq!(Spread { error: 0., groups: f64::INFINITY }.interval_over(1e6), 0.);
    }

    /// The binomial's upper tail (`at_least`): of 88 draws at the share 2.44e-4 one or more meet it with 1 − (1 − p)⁸⁸
    /// = 2.12 %, two or more with 2.25e-4 (that less 88 p (1 − p)⁸⁷); none or more always; at a share of one half, 88 or more with
    /// 2⁻⁸⁸, 45 or more with a half less half the middle term; a share of none is met by no draw, one of all by every.
    /// And Šidák's tail of one look among m (`tail_among`): p at one, 1 − (1 − p)^{1/21} at twenty-one.
    #[test]
    fn a_count_s_tail_is_the_binomial_s() {
        let p = unseen(16384.);
        assert!((p - 2.44e-4).abs() < 1e-6 && (at_least(1., 88., p) - (1. - (1. - p).powi(88))).abs() < 1e-15 && (at_least(1., 88., p) - 0.0212).abs() < 1e-4);
        assert!((at_least(2., 88., p) - (1. - (1. - p).powi(88) - 88. * p * (1. - p).powi(87))).abs() < 1e-15 && (at_least(2., 88., p) - 2.25e-4).abs() < 1e-6 && at_least(0., 88., p) == 1.);
        let middle = (1..=44).fold(1., |c, k| c * (88 - 44 + k) as f64 / k as f64) * 0.5f64.powi(88);
        assert!((at_least(88., 88., 0.5) - 0.5f64.powi(88)).abs() < 1e-15 && (at_least(45., 88., 0.5) - (0.5 - 0.5 * middle)).abs() < 1e-12);
        assert_eq!((at_least(1., 88., 0.), at_least(88., 88., 1.)), (0., 1.));
        let one = tail_among(1.);
        assert!((one - 0.045_500_263_896).abs() < 1e-9 && (tail_among(21.) - (1. - (1. - one).powf(1. / 21.))).abs() < 1e-15);
    }

    /// Wilson's far end at a count of zero: z²/(N + z²), at z = 2 four in N + 4, and the N it is a given share at; a draw of
    /// none leaves all unseen. At a count of five in a hundred, the score interval's far end
    /// (p̂ + z²/2N + z√(p̂(1 − p̂)/N + z²/4N²))/(1 + z²/N); at a count of all, one.
    #[test]
    fn the_unseen_share_is_wilson_s_far_end() {
        assert_eq!(unseen(0.), 1.);
        assert!((unseen(96.) - 0.04).abs() < 1e-16);
        assert_eq!((draws_for(unseen(96.)), draws_for(1.)), (96., 0.));
        assert!((unseen(16380.) - 4. / 16384.).abs() < 1e-18);
        let (p, n): (f64, f64) = (0.05, 100.);
        let far = (p + 2. / n + 2. * (p * (1. - p) / n + 1. / (n * n)).sqrt()) / (1. + 4. / n);
        assert!((wilson(5., 100.) - far).abs() < 1e-15, "{}", wilson(5., 100.));
        assert!((wilson(100., 100.) - 1.).abs() < 1e-15);
    }

    /// The window: half a period of the axis, cut at t = 0.
    #[test]
    fn the_window_is_a_period_cut_at_t_0() {
        let w = TAU / 4.;
        assert_eq!(half_window(w, 100.), 2.);
        assert_eq!(half_window(w, 0.75), 0.75);
        assert_eq!(half_window(w, 0.), 0.);
    }

    /// A held atom in `state` with ⟨κ⟩ = x, ⟨κ²⟩ = y on every axis.
    fn atom(state: usize, x: f64, y: f64) -> Held {
        let mut rows = [0.; STATES];
        rows[state] = 1.;
        Held { rows, kappa: [x; 3], kappa2: [y; 3], outside: false, unresolved: false, leaving: false }
    }
    fn group(atoms: Vec<Option<Held>>) -> Group {
        let mut g = Group::new(1);
        for held in atoms {
            g.add(&[held.map_or(Fate::Lost, Fate::Held)]);
        }
        g
    }
    /// ħ/m in KT_PER_UK's unit, so that T reads ⟨κ²⟩ − ⟨κ⟩² itself.
    const UNIT: Scales = Scales { hbar_over_mass: KT_PER_UK, frequencies: [1.; 3] };

    /// The bound atoms' motion (`bound_motion`) is the held atoms' less the leaving ones' (`Held::leaving`): five groups
    /// of four atoms, of ⟨κ⟩ and ⟨κ²⟩ set apart by their group and place, a leaving atom in three of them a hundred
    /// times hotter -- T and its jackknife per axis read as `summarize` reads the same groups without the leaving
    /// atoms, to the rounding of the sums they are taken out of, where the leaving atoms make the held atoms' T ten times
    /// higher; where every atom is bound, the held atoms' own to the bit.
    #[test]
    fn the_bound_atoms_motion_is_the_held_atoms_less_the_leaving_ones() {
        let atoms = |g: usize, leaving: bool| -> Vec<Option<Held>> {
            (0..4)
                .map(|m| {
                    let x = 0.1 * (g + 2 * m) as f64;
                    let hot = leaving && m == g % 4 && g % 2 == 0;
                    Some(Held { leaving: hot, ..atom(0, if hot { 10. * x } else { x }, if hot { 100. * (x * x + 1.) } else { x * x + 0.3 * (1 + g) as f64 }) })
                })
                .collect()
        };
        let with: Vec<Group> = (0..5).map(|g| group(atoms(g, true))).collect();
        let without: Vec<Group> = (0..5).map(|g| group(atoms(g, true).into_iter().filter(|a| !a.as_ref().is_some_and(|a| a.leaving)).collect())).collect();
        let (bound, own) = (bound_motion(&with, &UNIT)[0].unwrap(), summarize(&without, &UNIT)[0].motion.unwrap());
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-12 * b.abs().max(1.);
        assert!((0..3).all(|a| close(bound.temperature[a], own.temperature[a]) && close(bound.temperature_errors[a].error, own.temperature_errors[a].error)), "{bound:?} against {own:?}");
        let all = summarize(&with, &UNIT)[0].motion.unwrap();
        assert!(all.temperature[0] > 10. * own.temperature[0], "the leaving atoms make the held atoms' T {} against {}", all.temperature[0], own.temperature[0]);
        let cold: Vec<Group> = (0..5).map(|g| group(atoms(g, false))).collect();
        assert_eq!(bound_motion(&cold, &UNIT)[0].unwrap().temperature, summarize(&cold, &UNIT)[0].motion.unwrap().temperature);
    }

    /// The jackknife of a mean over groups of one is the standard error of the mean, √(Σ(x − x̄)²/(g(g − 1))): five
    /// groups showing state 0 as 0 0 0 0 1, 0.2; the loss and the rows as shown read on every group, the held atoms'
    /// quantities on the groups holding one.
    #[test]
    fn a_mean_s_jackknife_is_its_standard_error() {
        let mut groups: Vec<Group> = [1, 1, 1, 1, 0].into_iter().map(|s| group(vec![Some(atom(s, 0., 1.))])).collect();
        let e = &summarize(&groups, &UNIT)[0];
        assert!((e.populations[0] * (1. - e.loss) - 0.2).abs() < 1e-15 && (e.shown_errors[0].error - 0.2).abs() < 1e-15, "{:?}", e.shown_errors[0]);
        assert_eq!(e.shown_errors[0].interval(), student(CONFIDENCE, 4) * e.shown_errors[0].error);
        groups.push(group(vec![None]));
        let e = &summarize(&groups, &UNIT)[0];
        let m = e.motion.unwrap();
        assert_eq!((e.loss_error.groups, e.shown_errors[0].groups, m.temperature_errors[0].groups), (6., 6., 5.));
        assert!((e.loss - 1. / 6.).abs() < 1e-15 && (e.populations[1] - 0.8).abs() < 1e-15);
    }

    /// A row as shown carries the loss's variance: every held atom in one state shows that state at 1 − L, its error the
    /// loss's own, where the held atoms' populations have none (the held share times their error reads nothing of it).
    #[test]
    fn a_row_as_shown_carries_the_loss_s_variance() {
        let groups: Vec<Group> =
            [true, true, false, true, false, true].into_iter().map(|held| group(vec![held.then(|| atom(0, 0., 1.))])).collect();
        let e = &summarize(&groups, &UNIT)[0];
        assert!(e.loss_error.error > 0.1);
        assert!((e.shown_errors[0].error - e.loss_error.error).abs() < 1e-15, "{:?} {:?}", e.shown_errors[0], e.loss_error);
        assert_eq!(e.populations[0], 1.);
    }

    /// Repeated draws of `groups` groups of `members` atoms with ⟨κ⟩ = x ~ N(μ, σ²) and ⟨κ²⟩ = x² + c, so that the variance
    /// over atoms and windows is σ² + c: the mean of T over the repetitions, its standard error, and the share of
    /// repetitions whose interval covers σ² + c. Gaussian deviates by Box–Muller on splitmix64 (the test's own stream).
    fn repeated(groups: usize, members: usize, repetitions: usize) -> (f64, f64, f64) {
        let (mu, sigma, c) = (0.7, 1.3, 0.4);
        let truth = sigma * sigma + c;
        let mut state = 0x2545_F491_4F6C_DD1D_u64;
        let mut uniform = || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            (((z ^ (z >> 31)) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        let (mut sum, mut sum2, mut covered) = (0., 0., 0);
        for _ in 0..repetitions {
            let drawn: Vec<Group> = (0..groups)
                .map(|_| {
                    group(
                        (0..members)
                            .map(|_| {
                                let x = mu + sigma * (-2. * uniform().ln()).sqrt() * (TAU * uniform()).cos();
                                Some(atom(0, x, x * x + c))
                            })
                            .collect(),
                    )
                })
                .collect();
            let m = summarize(&drawn, &UNIT)[0].motion.unwrap();
            let t = m.temperature[0];
            sum += t;
            sum2 += t * t;
            covered += usize::from((t - truth).abs() <= m.temperature_errors[0].interval());
        }
        let n = repetitions as f64;
        let mean = sum / n;
        (mean - truth, ((sum2 / n - mean * mean) / (n - 1.)).sqrt(), covered as f64 / n)
    }

    /// T's estimate is unbiased: over 4000 repetitions of 6 groups of 3 atoms (n = 18) its mean meets σ² + c within its
    /// interval at the confidence z over the repetitions, where the plug-in variance, lower by its bias σ²/n = 0.094, lies
    /// beyond it.
    #[test]
    fn the_temperature_is_unbiased() {
        let repetitions = 4000;
        let (bias, se, _) = repeated(6, 3, repetitions);
        let within = Spread { error: se, groups: repetitions as f64 }.interval();
        assert!(bias.abs() <= within, "bias {bias} ± {within}");
        assert!((bias - 1.3 * 1.3 / 18.).abs() > within, "the plug-in's bias is resolved: ± {within}");
    }

    /// T's jackknife interval at z = 2 covers the truth as often as the normal's 95.45 %: 32 groups of 8 atoms, where the
    /// variance's skew √(8/(n − 1)) = 0.18 is small; the share of 2000 repetitions that cover it, a count's with the
    /// standard error √(p(1 − p)/2000) = 0.5 %, within its interval at the confidence z.
    #[test]
    fn the_temperature_s_interval_covers() {
        let repetitions = 2000;
        let (_, _, coverage) = repeated(32, 8, repetitions);
        let p = 1. - normal_tail(CONFIDENCE);
        let within = Spread { error: (p * (1. - p) / repetitions as f64).sqrt(), groups: f64::INFINITY }.interval();
        assert!((coverage - p).abs() <= within, "coverage {coverage} against {p} ± {within}");
    }

    /// n̄ and the mean velocity of the held atoms; no T error from one group; no motion below two atoms held.
    #[test]
    fn the_occupation_and_velocity_are_the_held_atoms_means() {
        let scales = Scales { hbar_over_mass: 2., frequencies: [4.; 3] };
        let groups = vec![group(vec![Some(atom(0, 1., 3.)), Some(atom(0, 3., 11.)), None])];
        let m = summarize(&groups, &scales)[0].motion.unwrap();
        assert!((m.occupation[0] - (2. * 7. / 4. - 0.5)).abs() < 1e-15 && (m.mean_velocity[1] - 4.).abs() < 1e-15);
        assert!(!m.temperature_errors[0].read() && m.temperature_errors[0].interval() == f64::INFINITY);
        assert!(summarize(&[group(vec![Some(atom(0, 1., 3.)), None])], &scales)[0].motion.is_none());
    }

    /// The atoms apart at an output (`Estimate::outside`, `Estimate::unresolved`): of three held atoms with
    /// Σ_a ⟨κ_a²⟩ = 9, 33 and 6, the second outside the reach and the third unresolved, each kind is a third of the atoms
    /// and holds 33/48 and 6/48 of the kinetic energy -- whatever the groups they come in, and a lost atom is none of
    /// them.
    #[test]
    fn the_atoms_apart_are_their_shares_of_the_held_atoms_and_of_their_kinetic_energy() {
        let scales = Scales { hbar_over_mass: 2., frequencies: [4.; 3] };
        let apart = |y: f64, unresolved: bool| Some(Held { outside: !unresolved, unresolved, ..atom(0, 1., y) });
        let one = summarize(&[group(vec![Some(atom(0, 1., 3.)), apart(11., false), apart(2., true), None])], &scales);
        let two = summarize(&[group(vec![Some(atom(0, 1., 3.)), None]), group(vec![apart(11., false), apart(2., true)])], &scales);
        for e in [&one[0], &two[0]] {
            assert!((e.outside[0] - 1. / 3.).abs() < 1e-15 && (e.outside[1] - 33. / 48.).abs() < 1e-15, "{:?}", e.outside);
            assert!((e.unresolved[0] - 1. / 3.).abs() < 1e-15 && (e.unresolved[1] - 6. / 48.).abs() < 1e-15, "{:?}", e.unresolved);
        }
        assert_eq!(summarize(&[group(vec![None, None])], &scales)[0].outside, [0.; 2]);
    }

    /// What is counted (module documentation, # What is counted). Four groups of two trajectories at t = 0 and three
    /// outputs, every atom held in state 0 with ⟨κ²⟩ = 1 + its group's number until it ends with no outcome: the first
    /// of each group after the first output, the second after the second. At the first output all eight count; at the
    /// second four of eight, u = ½, and the estimates are the four's -- nothing lost, state 0 whole, T read on the four
    /// groups --; at the third none: u = 1, the loss the second's, nothing read, no atom held, no T, and what is shown
    /// errs by its statistics at most one. The step's size between two such levels reads the outputs that count a
    /// trajectory at both levels with every group left out: none of the identical levels' differences, zero read with
    /// a spread; where no output counts one, no number. And atoms lost before the others end: the loss is the counted
    /// ones', the share of the others apart.
    #[test]
    fn an_output_no_trajectory_counts_at_estimates_nothing_and_its_share_is_known() {
        let made = || -> Vec<Group> {
            (0..4)
                .map(|g| {
                    let mut group = Group::new(4);
                    let held = Fate::Held(atom(0, 0., 1. + g as f64));
                    group.add(&[held.clone(), held.clone(), Fate::Failed, Fate::Failed]);
                    group.add(&[held.clone(), held.clone(), held, Fate::Failed]);
                    group
                })
                .collect()
        };
        let groups = made();
        let e = summarize(&groups, &UNIT);
        assert!(e.iter().all(|e| e.loss == 0.) && (e[1].uncounted, e[2].uncounted, e[3].uncounted) == (0., 0.5, 1.), "{e:?}");
        assert!((e[2].trajectories, e[2].held, e[2].failed) == (4., 4., 4.) && e[2].populations[0] == 1. && e[2].loss_error.read() && e[2].motion.is_some_and(|m| m.temperature_errors[0].read()));
        assert!((e[3].trajectories, e[3].held, e[3].failed) == (0., 0., 8.) && e[3].motion.is_none() && !e[3].loss_error.read() && e[3].shown_errors.iter().all(|s| !s.read()));
        assert_eq!((e[3].shown_interval(), e[3].shown_statistics()), (f64::INFINITY, 1.));
        let read = between(&groups, &made(), &UNIT, &[0., 1., 2., 3.]);
        assert!(read.loss.value == 0. && read.shown[0].value == 0. && read.loss.spread.read() && read.temperature.is_some_and(|t| t.value == 0.), "{read:?}");
        let late: Vec<Group> = groups.iter().map(|g| Group { sums: vec![g.sums[0].clone(), g.sums[3].clone()], scattered: g.scattered }).collect();
        let none = between(&late, &late, &UNIT, &[0., 3.]);
        assert!(none.loss.value.is_nan() && !none.loss.spread.read() && none.temperature.is_none(), "{none:?}");
        let lost: Vec<Group> = (0..4)
            .map(|_| {
                let mut g = Group::new(3);
                g.add(&[Fate::Held(atom(0, 0., 1.)), Fate::Lost, Fate::Lost]);
                g.add(&[Fate::Held(atom(0, 0., 1.)), Fate::Held(atom(0, 0., 1.)), Fate::Failed]);
                g
            })
            .collect();
        let e = summarize(&lost, &UNIT);
        assert!((e[1].loss, e[1].uncounted) == (0.5, 0.) && (e[2].loss, e[2].uncounted, e[2].held) == (1., 0.5, 0.), "{e:?}");
    }
}
