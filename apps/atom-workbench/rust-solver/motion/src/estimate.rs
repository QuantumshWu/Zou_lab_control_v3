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
//!   phase out of ⟨κ_a²⟩, so the spread between atoms is that of their energies;
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
//! τ and at 2τ: the loss and each row as shown as they are, the total T relative to the finer level's.
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
use crate::decide::{richardson, Read};
use crate::request::STATES;
use crate::special::upper;
use crate::units::{CONFIDENCE, KT_PER_UK, ORDER};
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

/// The half width of T's window on a lab axis about an output at t (SPEC §1.7): half of one trap period of the axis,
/// 2π/ω at its frequency ω > 0, the window no longer than an output interval, cut at t = 0 so that it stays centred there.
pub fn half_window(frequency: f64, interval: f64, t: f64) -> f64 {
    (0.5 * (TAU / frequency).min(interval)).min(t)
}

/// n̄ of a lab axis of frequency ω > 0 (rad/µs; module documentation): (ħ/m)Σ⟨κ²⟩ (rad/µs) summed over `atoms` atoms, over
/// their number and ω, less ½.
pub fn occupation(kinetic: f64, atoms: f64, frequency: f64) -> f64 {
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
        let tail = -((-normal_tail(CONFIDENCE)).ln_1p() / looks).exp_m1();
        let freedom = (self.groups < f64::INFINITY).then(|| (self.groups.floor() - 1.) as u64);
        quantile(tail, freedom) * self.error
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
/// output's window (1/µm, 1/µm²).
#[derive(Clone, Debug, PartialEq)]
pub struct Held {
    pub rows: [f64; STATES],
    pub kappa: [f64; 3],
    pub kappa2: [f64; 3],
}

/// One trajectory at one output, what became of it by then: its atom is held; it was lost (SPEC §1.7); or the scheme
/// could not advance its state (adapt.rs) -- no physical outcome: the trajectory is left out of the output's estimates
/// and counted.
#[derive(Clone, Debug, PartialEq)]
pub enum Fate {
    Held(Held),
    Lost,
    Failed,
}

/// One output's sums over a group's trajectories: their number (the failed ones apart), the atoms held, and over the held
/// atoms Σρ_s, Σ⟨κ⟩, Σ⟨κ⟩² (for the U-statistic, module documentation) and Σ⟨κ²⟩ per lab axis.
#[derive(Clone, Debug, PartialEq)]
pub struct Sums {
    pub trajectories: f64,
    pub failed: f64,
    pub held: f64,
    pub rows: [f64; STATES],
    pub kappa: [f64; 3],
    pub kappa_squared: [f64; 3],
    pub kappa2: [f64; 3],
}
impl Sums {
    pub const ZERO: Sums = Sums { trajectories: 0., failed: 0., held: 0., rows: [0.; STATES], kappa: [0.; 3], kappa_squared: [0.; 3], kappa2: [0.; 3] };
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
    }
}

/// One group's sums at every output (module documentation, # Errors).
#[derive(Clone, Debug, PartialEq)]
pub struct Group {
    pub sums: Vec<Sums>,
}
impl Group {
    pub fn new(outputs: usize) -> Group {
        Group { sums: vec![Sums::ZERO; outputs] }
    }
    /// The group's further trajectories, after those it holds.
    pub fn absorb(&mut self, other: Group) {
        for (s, o) in self.sums.iter_mut().zip(&other.sums) {
            s.combine(o, 1.);
        }
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

/// The estimates at one output (module documentation): the loss, the rows as shown and the held atoms' populations, the
/// held atoms' motion where at least two are held; each spread over the groups that read it, and the trajectories and
/// held atoms behind them.
#[derive(Clone, Debug)]
pub struct Estimate {
    pub trajectories: f64,
    pub held: f64,
    pub loss: f64,
    pub loss_error: Spread,
    pub shown: [f64; STATES],
    pub shown_errors: [Spread; STATES],
    pub populations: [f64; STATES],
    pub motion: Option<Motion>,
    /// The trajectories the scheme could not advance to the output (`Fate::Failed`): none of the above counts them.
    pub failed: f64,
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
    /// trajectories leave unseen (`unseen`) -- a loss or a state none of them reached reads no spread.
    pub fn shown_statistics(&self) -> f64 {
        self.shown_interval().max(unseen(self.trajectories))
    }
    /// The interval of the total T (the mean over the lab axes) at the confidence z, relative to it; unbounded where T
    /// is not read.
    pub fn temperature_interval(&self) -> f64 {
        self.motion.as_ref().map_or(f64::INFINITY, |m| m.temperature_total_error.interval() / m.temperature_total)
    }
}

/// What the jackknife spreads, of one set of sums.
struct Moments {
    loss: f64,
    shown: [f64; STATES],
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
    Moments { loss: 1. - s.held / s.trajectories, shown: s.rows.map(|r| r / s.trajectories), temperature: temperature(s, hbar_over_mass) }
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
    /// A quantity of the levels' moments spread over every group.
    fn every(&self, f: impl Fn(&[Moments]) -> f64) -> Spread {
        if self.leave.is_empty() {
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

/// The estimates at every output from the groups' sums (module documentation).
pub fn summarize<G: Borrow<Group> + Sync>(groups: &[G], scales: &Scales) -> Vec<Estimate> {
    let outputs = groups.first().map_or(0, |g| g.borrow().sums.len());
    let hm = scales.hbar_over_mass;
    (0..outputs)
        .into_par_iter()
        .map(|k| {
            let j = jackknife(&[groups], k, hm);
            let (all, full) = (&j.all[0], &j.full[0]);
            let n = all.held;
            let motion = full.temperature.map(|t| Motion {
                temperature: t,
                temperature_errors: std::array::from_fn(|a| j.held(|x| x[0][a])),
                temperature_total: t.iter().sum::<f64>() / 3.,
                temperature_total_error: j.held(|x| x[0].iter().sum::<f64>() / 3.),
                occupation: std::array::from_fn(|a| occupation(hm * all.kappa2[a], n, scales.frequencies[a])),
                mean_velocity: std::array::from_fn(|a| hm * all.kappa[a] / n),
            });
            Estimate {
                trajectories: all.trajectories,
                held: n,
                loss: full.loss,
                loss_error: j.every(|m| m[0].loss),
                shown: full.shown,
                shown_errors: std::array::from_fn(|s| j.every(|m| m[0].shown[s])),
                populations: if n > 0. { all.rows.map(|r| r / n) } else { [0.; STATES] },
                motion,
                failed: all.failed,
            }
        })
        .collect()
}

/// The size of the step's error over the run as two ensembles measure it (module documentation, # The step's error):
/// per quantity the read of its mean square over the outputs after t = 0 -- the loss's and each row's as shown
/// (absolute, squared), and the total T's (the mean over the three lab axes, the T the target is on) relative to it --
/// none where no output reads it at both levels. A read is of the mean square itself, unbiased, so it lies about zero,
/// on either side, where there is no error.
#[derive(Clone, Debug)]
pub struct Between {
    pub loss: Read,
    pub shown: [Read; STATES],
    pub temperature: Option<Read>,
}

/// The step's error of the level `fine`, from it and the level `coarse` at twice its step on the same trajectories,
/// group by group in the same order (module documentation, # The step's error): each quantity's mean square over the
/// outputs after t = 0 -- T's over the outputs where both levels, and both without any one group, hold two atoms at
/// least. No number on a single group; no spread on fewer than three (a mean square without a group needs two more).
pub fn between<G: Borrow<Group> + Sync>(fine: &[G], coarse: &[G], scales: &Scales) -> Between {
    assert_eq!(fine.len(), coarse.len(), "the two levels hold the same groups");
    let outputs = fine.first().map_or(0, |g| g.borrow().sums.len());
    let (hm, groups) = (scales.hbar_over_mass, fine.len());
    let g = groups as f64;
    // The finer level's share of the difference: 1/(2^p − 1).
    let share = richardson(0.5, ORDER as f64) - 1.;
    let each: Vec<Jackknife> = (1..outputs).into_par_iter().map(|k| jackknife(&[fine, coarse], k, hm)).collect();
    // A quantity's mean square over some outputs: at each, the groups' own values of it (g times the whole less g − 1
    // times the rest) and the mean of their products two by two; the same without each group, spread over the groups.
    let size = |at: &[&Jackknife], f: &dyn Fn(&[Moments]) -> f64, read: f64| -> Read {
        let (mut whole, mut without, mut own) = (0., vec![0.; groups], vec![0.; groups]);
        for j in at {
            let full = f(&j.full);
            own.iter_mut().zip(&j.leave).for_each(|(x, rest)| *x = g * full - (g - 1.) * f(rest));
            let (sum, squares) = own.iter().fold((0., 0.), |(s, q), x| (s + x, q + x * x));
            whole += sum * sum - squares;
            without.iter_mut().zip(&own).for_each(|(w, x)| *w += (sum - x).powi(2) - (squares - x * x));
        }
        let n = at.len() as f64;
        Read {
            value: if groups >= 2 { whole / (n * g * (g - 1.)) } else { f64::NAN },
            spread: if groups >= 3 { spread(without.iter().map(|w| w / (n * (g - 1.) * (g - 2.))), read) } else { Spread::UNREAD },
        }
    };
    let all: Vec<&Jackknife> = each.iter().collect();
    let temperature = {
        let read = |m: &[Moments]| m.iter().all(|x| x.temperature.is_some());
        let at: Vec<&Jackknife> = each.iter().filter(|j| read(&j.full) && j.leave.iter().all(|l| read(l))).collect();
        (!at.is_empty()).then(|| {
            let total = |m: &Moments| m.temperature.map_or(f64::NAN, |t| t.iter().sum::<f64>());
            let e = |m: &[Moments]| share * (total(&m[1]) / total(&m[0]) - 1.);
            size(&at, &e, at.iter().map(|j| j.holding).fold(f64::INFINITY, f64::min))
        })
    };
    Between {
        loss: size(&all, &|m| share * (m[1].loss - m[0].loss), g),
        shown: std::array::from_fn(|s| size(&all, &|m| share * (m[1].shown[s] - m[0].shown[s]), g)),
        temperature,
    }
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

    /// The step's size between two ensembles (`between`) on constructed levels. Three groups, two outputs after t = 0;
    /// a row shown by c of 8 atoms at the coarser level and by a at the finer has the group's own error (c − a)/8/3
    /// (p = 2). With the groups' own errors x = (3, −1, 1)/24 at the first output and (2, 2, 2)/24 at the second, the
    /// mean square is the mean over the outputs of the mean of the products of two different groups,
    /// ((3·(−1) + 3·1 + (−1)·1)/3 + 4)/24²/2 -- at the first output below zero: the three errors are no more than
    /// their scatter --, and without a group the product of the other two; its spread is the jackknife of those. The
    /// other row is the first's mirror; the loss has none. T is 7 at the finer level and 4 and 6 at the coarser on
    /// every atom: the groups agree, and its mean square is the mean of (4/7 − 1)²/9 and (6/7 − 1)²/9 with no spread.
    /// Two groups read a value and no spread; one reads none.
    #[test]
    fn the_step_s_size_is_the_mean_product_of_two_groups_errors() {
        let level = |counts: [[usize; 2]; 3], energy: [f64; 2]| -> Vec<Group> { counts.iter().map(|c| showing(c, &energy)).collect() };
        let (fine, coarse) = (level([[4, 4], [4, 4], [4, 4]], [7., 7.]), level([[7, 6], [3, 6], [5, 6]], [4., 6.]));
        let read = between(&fine, &coarse, &UNIT);
        let pairs = |x: [f64; 3]| [x[1] * x[2], x[0] * x[2], x[0] * x[1]];
        let (first, second) = (pairs([3., -1., 1.]), pairs([2., 2., 2.]));
        let without: Vec<f64> = (0..3).map(|i| 0.5 * (first[i] + second[i]) / 576.).collect();
        let whole = without.iter().sum::<f64>() / 3.;
        assert!(first.iter().sum::<f64>() < 0. && (read.shown[0].value - whole).abs() < 1e-17 && (read.shown[1].value - whole).abs() < 1e-17, "{:?} against {whole}", read.shown[0]);
        let expected = spread(without.iter().copied(), 3.);
        assert!((read.shown[0].spread.error - expected.error).abs() < 1e-17 && read.shown[0].spread.groups == 3., "{:?} against {expected:?}", read.shown[0].spread);
        assert!(read.loss.value == 0. && read.loss.spread.error == 0.);
        let t = read.temperature.expect("T is read at both levels");
        assert!((t.value - 0.5 * ((4. / 7. - 1.0f64).powi(2) + (6. / 7. - 1.0f64).powi(2)) / 9.).abs() < 1e-15 && t.spread.error < 1e-15, "{t:?}");
        let two = between(&fine[..2], &coarse[..2], &UNIT).shown[0];
        assert!((two.value - 0.5 * (-3. + 4.) / 576.).abs() < 1e-17 && !two.spread.read(), "{two:?}");
        assert!(between(&fine[..1], &coarse[..1], &UNIT).shown[0].value.is_nan());
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
                let read = between(&fine, &coarse, &UNIT).shown[0];
                let (f, c) = (summarize(&fine, &UNIT), summarize(&coarse, &UNIT));
                let mean = (1..=outputs).map(|k| (c[k].shown[0] - f[k].shown[0]) / 3.).sum::<f64>() / outputs as f64;
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
    /// size, and for levels that share their atoms (ρ = 0.95), where it is a twentieth of that.
    #[test]
    fn the_temperature_s_step_size_is_the_paired_ratio_s() {
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
                let read = between(&twice(&fine), &twice(&coarse), &UNIT).temperature.unwrap();
                sum += read.value;
                squares += read.value * read.value;
            }
            let n = repetitions as f64;
            let mean = sum / n;
            let within = Spread { error: ((squares / n - mean * mean) / n).sqrt(), groups: n }.interval();
            assert!((mean - bias * bias).abs() <= within, "ρ = {rho}: the read's mean {mean} ± {within} against {}", bias * bias);
        }
    }

    /// Groups of one output as groups of two: t = 0 and the output itself.
    fn twice(groups: &[Group]) -> Vec<Group> {
        groups.iter().map(|g| Group { sums: vec![g.sums[0].clone(), g.sums[0].clone()] }).collect()
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

    /// The window: half a period of the axis, at most half an output interval, cut at t = 0.
    #[test]
    fn the_window_is_a_period_at_most_an_interval() {
        let w = TAU / 4.;
        assert_eq!(half_window(w, 10., 100.), 2.);
        assert_eq!(half_window(w, 1., 100.), 0.5);
        assert_eq!(half_window(w, 10., 0.75), 0.75);
        assert_eq!(half_window(w, 10., 0.), 0.);
    }

    /// A held atom in `state` with ⟨κ⟩ = x, ⟨κ²⟩ = y on every axis.
    fn atom(state: usize, x: f64, y: f64) -> Held {
        let mut rows = [0.; STATES];
        rows[state] = 1.;
        Held { rows, kappa: [x; 3], kappa2: [y; 3] }
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

    /// The jackknife of a mean over groups of one is the standard error of the mean, √(Σ(x − x̄)²/(g(g − 1))): five
    /// groups showing state 0 as 0 0 0 0 1, 0.2; the loss and the rows as shown read on every group, the held atoms'
    /// quantities on the groups holding one.
    #[test]
    fn a_mean_s_jackknife_is_its_standard_error() {
        let mut groups: Vec<Group> = [1, 1, 1, 1, 0].into_iter().map(|s| group(vec![Some(atom(s, 0., 1.))])).collect();
        let e = &summarize(&groups, &UNIT)[0];
        assert!((e.shown[0] - 0.2).abs() < 1e-15 && (e.shown_errors[0].error - 0.2).abs() < 1e-15, "{:?}", e.shown_errors[0]);
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
}
