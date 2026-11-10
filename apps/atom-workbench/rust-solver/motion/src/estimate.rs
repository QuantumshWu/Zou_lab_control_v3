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
//! The trajectories that are read carry their accumulated signed step errors (adapt.rs `Drift`). An output's step error
//! is their mean (`stepped`): on T_a the axis's energy error over k_B T_a (relative), on each row as shown its error over
//! all the trajectories, each jackknifed like the estimate it belongs to; where a step of some atom there went unread
//! the reads are no numbers. Of the loss energy the mean and the spread over every read trajectory -- a lost one's error
//! as it stood at its loss, where its loss was decided --, the window within which a loss decision may differ (solve.rs).
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
//! whose jackknife reads nothing of a loss or a state no trajectory reached, the reach's unsampled share, and the
//! trajectories within the step's error of the escape (solve.rs).
use crate::adapt::Drift;
use crate::decide::Read;
use crate::request::STATES;
use crate::special::upper;
use crate::units::{CONFIDENCE, KT_PER_UK};
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

/// An atom held at an output: its readout's populations (SPEC §1.7), per lab axis its means of κ and κ² over the
/// output's window (1/µm, 1/µm²), and, where its trajectory is read, its accumulated step errors (adapt.rs `Drift`).
#[derive(Clone, Debug, PartialEq)]
pub struct Held {
    pub rows: [f64; STATES],
    pub kappa: [f64; 3],
    pub kappa2: [f64; 3],
    pub drift: Option<Drift>,
}

/// What became of a trajectory by an output: its atom is held; it was lost (SPEC §1.7); or the scheme could not advance
/// its state (adapt.rs) -- no physical outcome: the trajectory is left out of the output's estimates and counted.
#[derive(Clone, Debug, PartialEq)]
pub enum Fate {
    Held(Held),
    Lost,
    Failed,
}

/// One trajectory at one output: its fate; where its loss decision is judged against the step's error, whether its loss
/// energy lay within that of the escape (solve.rs); and, of a read trajectory, the loss energy's accumulated error where
/// that decision stood -- at the output while held, at its loss once lost.
#[derive(Clone, Debug, PartialEq)]
pub struct Output {
    pub fate: Fate,
    pub near: Option<bool>,
    pub energy: Option<f64>,
}

/// One output's sums over a group's trajectories: their number (the failed ones apart), the atoms held, and over the held atoms Σρ_s, Σ⟨κ⟩,
/// Σ⟨κ⟩² (for the U-statistic, module documentation) and Σ⟨κ²⟩ per lab axis; the held atoms that are read and the sums
/// of their accumulated step errors (`Drift`); the trajectories judged against the step's error and those of them near
/// the escape; the trajectories whose loss energy's error is read, its sum and its squares' (`Output`).
#[derive(Clone, Debug, PartialEq)]
pub struct Sums {
    pub trajectories: f64,
    pub failed: f64,
    pub held: f64,
    pub rows: [f64; STATES],
    pub kappa: [f64; 3],
    pub kappa_squared: [f64; 3],
    pub kappa2: [f64; 3],
    pub read: f64,
    pub drift: Drift,
    pub judged: f64,
    pub near: f64,
    pub energy: [f64; 3],
}
impl Sums {
    pub const ZERO: Sums = Sums {
        trajectories: 0.,
        failed: 0.,
        held: 0.,
        rows: [0.; STATES],
        kappa: [0.; 3],
        kappa_squared: [0.; 3],
        kappa2: [0.; 3],
        read: 0.,
        drift: Drift::ZERO,
        judged: 0.,
        near: 0.,
        energy: [0.; 3],
    };
    /// One trajectory at the output.
    pub fn add(&mut self, output: &Output) {
        if output.fate == Fate::Failed {
            self.failed += 1.;
            return;
        }
        self.trajectories += 1.;
        if let Some(near) = output.near {
            self.judged += 1.;
            self.near += f64::from(u8::from(near));
        }
        if let Some(e) = output.energy {
            self.energy = [self.energy[0] + 1., self.energy[1] + e, self.energy[2] + e * e];
        }
        if let Fate::Held(h) = &output.fate {
            self.held += 1.;
            for (s, r) in self.rows.iter_mut().zip(&h.rows) {
                *s += r;
            }
            for a in 0..3 {
                self.kappa[a] += h.kappa[a];
                self.kappa_squared[a] += h.kappa[a] * h.kappa[a];
                self.kappa2[a] += h.kappa2[a];
            }
            if let Some(d) = &h.drift {
                self.read += 1.;
                self.add_drift(d, 1.);
            }
        }
    }
    fn add_drift(&mut self, d: &Drift, sign: f64) {
        for a in 0..3 {
            self.drift.axes[a] += sign * d.axes[a];
        }
        self.drift.energy += sign * d.energy;
        for (s, r) in self.drift.rows.iter_mut().zip(&d.rows) {
            *s += sign * r;
        }
        self.drift.unread += sign * d.unread;
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
        self.read += sign * o.read;
        self.judged += sign * o.judged;
        self.near += sign * o.near;
        for (s, x) in self.energy.iter_mut().zip(&o.energy) {
            *s += sign * x;
        }
        self.add_drift(&o.drift, sign);
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
    /// One trajectory's outputs, in order.
    pub fn add(&mut self, outputs: &[Output]) {
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
    /// The trajectories judged against the step's error of the loss energy, and those of them within it of the escape
    /// (`Sums::judged`, `Sums::near`).
    pub judged: f64,
    pub near: f64,
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
    /// T's largest interval at the confidence z over the lab axes, each relative to its `scale` (µK); unbounded where T is
    /// not read.
    pub fn temperature_interval(&self, scale: [f64; 3]) -> f64 {
        match &self.motion {
            Some(m) => (0..3).map(|a| m.temperature_errors[a].interval() / scale[a]).fold(0., f64::max),
            None => f64::INFINITY,
        }
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
                judged: all.judged,
                near: all.near,
                failed: all.failed,
            }
        })
        .collect()
}

/// One output's reads between two levels run on the same trajectories (decide.rs): the coarse level's loss, rows as
/// shown and T per lab axis (µK) less the fine level's, each jackknifed with a group left out of both -- the
/// difference's own error, in which the levels' common noise cancels. T where both levels hold two atoms.
#[derive(Clone, Debug)]
pub struct Paired {
    pub loss: Read,
    pub shown: [Read; STATES],
    pub temperature: Option<[Read; 3]>,
}

/// The reads between a coarse and a fine level at every output, their groups the same trajectories in the same order.
pub fn paired<G: Borrow<Group> + Sync>(coarse: &[G], fine: &[G], scales: &Scales) -> Vec<Paired> {
    assert_eq!(coarse.len(), fine.len(), "paired levels run the same groups");
    let outputs = coarse.first().map_or(0, |g| g.borrow().sums.len());
    (0..outputs)
        .into_par_iter()
        .map(|k| {
            let j = jackknife(&[coarse, fine], k, scales.hbar_over_mass);
            let read = |f: &dyn Fn(&Moments) -> f64| Read { value: f(&j.full[0]) - f(&j.full[1]), spread: j.every(|m| f(&m[0]) - f(&m[1])) };
            let temperature = match (j.full[0].temperature, j.full[1].temperature) {
                (Some(a), Some(b)) => Some(std::array::from_fn(|x| Read { value: a[x] - b[x], spread: j.held(|t| t[0][x] - t[1][x]) })),
                _ => None,
            };
            Paired { loss: read(&|m| m.loss), shown: std::array::from_fn(|s| read(&|m| m.shown[s])), temperature }
        })
        .collect()
}

/// The step's error at one output, read on the trajectories that carry their drift (module documentation, # The step's
/// error): T's relative error per lab axis where two atoms are read, each row's as shown, and, where some trajectory's
/// is read, the loss energy's accumulated error -- its mean and its spread between trajectories (rad/µs).
#[derive(Clone, Debug)]
pub struct Stepped {
    pub temperature: Option<[Read; 3]>,
    pub shown: [Read; STATES],
    pub energy: Option<[f64; 2]>,
}

/// The step's error at every output from the groups' sums.
pub fn stepped<G: Borrow<Group> + Sync>(groups: &[G], scales: &Scales) -> Vec<Stepped> {
    let outputs = groups.first().map_or(0, |g| g.borrow().sums.len());
    let hm = scales.hbar_over_mass;
    (0..outputs)
        .into_par_iter()
        .map(|k| {
            let mut all = Sums::ZERO;
            for g in groups {
                all.combine(&g.borrow().sums[k], 1.);
            }
            let rest: Vec<Sums> = groups
                .iter()
                .map(|g| {
                    let mut r = all.clone();
                    r.combine(&g.borrow().sums[k], -1.);
                    r
                })
                .collect();
            let reading = groups.iter().filter(|g| Borrow::<Group>::borrow(*g).sums[k].read > 0.).count() as f64;
            // A read with an unread step behind it is no number (module documentation, # The step's error).
            let complete = all.drift.unread == 0.;
            let read = |f: &dyn Fn(&Sums) -> f64, groups: f64| Read {
                value: if complete { f(&all) } else { f64::NAN },
                spread: if complete && rest.len() >= 2 { spread(rest.iter().map(|s| f(s)), groups) } else { Spread::UNREAD },
            };
            let relative = |s: &Sums, a: usize| match temperature(s, hm) {
                Some(t) if s.read > 0. => s.drift.axes[a] / s.read / (KT_PER_UK * t[a]),
                _ => f64::NAN,
            };
            let [count, sum, squares] = all.energy;
            let energy = (count > 0.).then(|| {
                let mean = sum / count;
                [mean, (squares / count - mean * mean).max(0.).sqrt()]
            });
            Stepped {
                temperature: (all.read >= 2. && all.held >= 2.).then(|| std::array::from_fn(|a| read(&|s: &Sums| relative(s, a), reading))),
                shown: std::array::from_fn(|q| read(&|s: &Sums| s.drift.rows[q] / s.trajectories, rest.len() as f64)),
                energy,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The step's error (`stepped`): three groups of two atoms at rest in the mean, each with ⟨κ²⟩ = 4 on every axis
    /// (T_a = 4 ħ/m k_B), read with one drift -- the axes' energies off by 1, 2 and 3 % of k_B T_a, a row by 0.02, the loss
    /// energy by 0.5 and 1.5 in turn. T's read is those shares with no spread; the row's as shown is 0.02 on the held
    /// atoms over all the trajectories (each group's third trajectory is lost: two of three held);
    /// the loss energy's mean 1 and spread ½.
    #[test]
    fn the_step_s_error_is_the_drifts_mean() {
        let scales = Scales { hbar_over_mass: 1e-3, frequencies: [1.; 3] };
        let kt = 4. * scales.hbar_over_mass;
        let atom = |energy: f64| {
            let mut drift = Drift::ZERO;
            drift.axes = [0.01 * kt, 0.02 * kt, 0.03 * kt];
            drift.rows[2] = 0.02;
            drift.energy = energy;
            let mut rows = [0.; STATES];
            rows[2] = 1.;
            Output { fate: Fate::Held(Held { rows, kappa: [0.; 3], kappa2: [4.; 3], drift: Some(drift) }), near: None, energy: Some(energy) }
        };
        let groups: Vec<Group> = (0..3)
            .map(|_| {
                let mut g = Group::new(1);
                g.add(&[atom(0.5)]);
                g.add(&[atom(1.5)]);
                g.add(&[Output { fate: Fate::Lost, near: None, energy: None }]);
                g
            })
            .collect();
        let read = &stepped(&groups, &scales)[0];
        let t = read.temperature.as_ref().expect("two atoms are read");
        for (a, share) in [0.01, 0.02, 0.03].into_iter().enumerate() {
            assert!((t[a].value - share).abs() < 1e-15 && t[a].spread.error < 1e-15, "{:?}", t[a]);
        }
        assert!((read.shown[2].value - 0.02 * 2. / 3.).abs() < 1e-15 && read.shown[0].value == 0., "{:?}", read.shown[2]);
        let energy = read.energy.expect("the trajectories' loss energy errors are read");
        assert!((energy[0] - 1.).abs() < 1e-15 && (energy[1] - 0.5).abs() < 1e-15, "{energy:?}");
        // One unread step among them: the reads are no numbers (unreadable), whatever the others read.
        let mut unread = atom(0.5);
        if let Fate::Held(h) = &mut unread.fate {
            h.drift.as_mut().unwrap().unread = 1.;
        }
        let mut spoiled = Group::new(1);
        spoiled.add(&[unread]);
        let read = &stepped(&[groups[0].clone(), groups[1].clone(), spoiled], &scales)[0];
        assert!(read.shown[2].value.is_nan() && !read.shown[2].spread.read() && read.shown[2].bound(1.) == f64::INFINITY);
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
        Held { rows, kappa: [x; 3], kappa2: [y; 3], drift: None }
    }
    fn group(atoms: Vec<Option<Held>>) -> Group {
        let mut g = Group::new(1);
        for held in atoms {
            g.add(&[Output { fate: held.map_or(Fate::Lost, Fate::Held), near: None, energy: None }]);
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

    /// T's estimate is unbiased: over 4000 repetitions of 6 groups of 3 atoms (n = 18) its mean meets σ² + c within four
    /// of its standard errors, where the plug-in variance's bias σ²/n = 0.094 is ten of them.
    #[test]
    fn the_temperature_is_unbiased() {
        let (bias, se, _) = repeated(6, 3, 4000);
        assert!(bias.abs() <= 4. * se, "bias {bias} ± {se}");
        assert!(1.3 * 1.3 / 18. > 6. * se, "the plug-in's bias is resolved: {se}");
    }

    /// T's jackknife interval at z = 2 covers the truth about as often as the normal's 95.45 %: 32 groups of 8 atoms,
    /// 2000 repetitions (the share's own standard error 0.5 %), where the variance's skew √(8/(n − 1)) = 0.18 is small.
    #[test]
    fn the_temperature_s_interval_covers() {
        let (_, _, coverage) = repeated(32, 8, 2000);
        assert!((coverage - 0.9545).abs() < 0.02, "coverage {coverage}");
    }

    /// A paired read cancels the levels' common noise: the same atoms with every ⟨κ²⟩ larger by δ at the fine level read
    /// T's difference −δ with no error, the loss and the rows none at all; one atom lost at the fine level alone reads
    /// the loss's difference −1/N with an error of one group's.
    #[test]
    fn a_paired_read_cancels_the_common_noise() {
        let mut state = 0x853C_49E6_748F_EA9B_u64;
        let mut draw = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let atoms: Vec<Vec<(usize, f64)>> = (0..5).map(|_| (0..4).map(|_| ((4. * draw()) as usize, draw() - 0.5)).collect()).collect();
        let level = |shift: f64, lost: Option<(usize, usize)>| -> Vec<Group> {
            atoms
                .iter()
                .enumerate()
                .map(|(g, members)| {
                    group(members.iter().enumerate().map(|(i, &(s, x))| (lost != Some((g, i))).then(|| atom(s, x, x * x + 1. + shift))).collect())
                })
                .collect()
        };
        let delta = 0.25;
        let reads = paired(&level(0., None), &level(delta, None), &UNIT);
        let t = reads[0].temperature.unwrap();
        assert!((t[1].value + delta).abs() < 1e-14 && t[1].spread.error < 1e-14 && t[1].spread.groups == 5., "{:?}", t[1]);
        assert!(reads[0].loss.value == 0. && reads[0].loss.spread.error == 0. && reads[0].shown.iter().all(|r| r.value.abs() < 1e-15));
        let reads = paired(&level(0., None), &level(0., Some((2, 1))), &UNIT);
        let loss = reads[0].loss;
        // One group's difference −1/4 among five, the others none: the jackknife of a mean, √(Σ(d − d̄)²/(g(g − 1))).
        let d: [f64; 5] = [0., 0., -0.25, 0., 0.];
        let mean = d.iter().sum::<f64>() / 5.;
        let se = (d.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / 20.).sqrt();
        assert!((loss.value + 1. / 20.).abs() < 1e-15 && (loss.spread.error - se).abs() < 1e-15, "{loss:?}");
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
