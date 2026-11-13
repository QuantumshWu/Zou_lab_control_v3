//! The ledger: model, step and statistics errors per output, items, flags (SPEC §4).
//!
//! # The total error
//! An output's error against its target has three parts from independent sources, combined in quadrature (GUM):
//! E² = A² + b² + S² (`Account`):
//! - A, the model's: each unit's error measured along the trajectories (SPEC §5.2), the share of the thermal ensemble
//!   that the model's reach leaves out -- the unseen share at the most trajectories a solve runs (SPEC §1.8) --, and
//!   what the run's atoms that leave that reach carry, read where they are (solve.rs, # The reach's atoms);
//! - b, the discretization's, the part a finer one takes away: the step's, what the pilot's check between two steps
//!   reads clearly of its error at the output (SPEC §2.2: a check at the pilot's resolution, not a bound; the step's
//!   share until it is read); and at t = 0, where T is the
//!   initial ensemble's exact moment by quadrature, that quadrature's last two levels' difference (solve.rs
//!   `Fixed::start`);
//! - S, the statistics': the jackknife's interval at the confidence z, on what is shown no less than what the run's own
//!   trajectories leave unseen -- an event none of them showed has no spread to read (estimate.rs
//!   `Estimate::shown_statistics`). The same function as A's share, another error: that one is the model's, at the most
//!   trajectories; this one the sample's, at its own number.
//! The populations' are absolute against `POPULATION_TARGET`, every row as shown and the loss; T's relative, of the total
//! T (the mean over the three lab axes, the one T the target is on; no lab axis is held to it), against T's allowance
//! at each output where it has an account -- `TEMPERATURE_TARGET` where T is high, the semiclassical error scale where
//! it is not (solve.rs, # T's allowance) --, its three parts weighed by the held share 1 − L (SPEC §0, §4: the
//! temperature describes the held atoms; solve.rs `accounts`).
//!
//! # The contract's shape (contract §2.2)
//! `totals`: the populations' and T's a-priori part √(A² + b²) -- what the approximations leave, the statistics aside (the
//! result reports T's statistical error itself, src/solver-types.ts `LedgerTotal`) -- at the output where it is largest,
//! T's over every output where the total T has an account, with the target; `items`:
//! each approximation's label, its estimated error (on the populations absolute, on T relative; its largest over the
//! outputs, so that an output without atoms takes no part) and what the engine did to get it, in Chinese as the app shows
//! them; `flagged`: where
//! some output's total exceeds its target -- the span, the largest total and its three parts --, and every other reason
//! a span is less than the solve claims; `note`: what is true of the setup whatever the solve does, never an accuracy
//! conclusion.
use crate::units::{POPULATION_TARGET, TEMPERATURE_TARGET, UNIT_ROUNDOFF};

/// One approximation: what it is, its estimated error and what the engine did with it (Chinese, as the app shows them).
#[derive(Clone, Debug, PartialEq)]
pub struct Item {
    pub label: String,
    pub estimate: f64,
    pub action: String,
}

/// A target's total at the output where it takes most of its target, and that target.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Total {
    pub error: f64,
    pub target: f64,
}

/// The populations' total (absolute) and T's (relative).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Totals {
    pub population: Total,
    pub temperature: Total,
}

/// One output's error against its target in its three parts A, b, S (module documentation, # The total error).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Account {
    pub model: f64,
    pub step: f64,
    pub statistics: f64,
}
impl Account {
    /// √(A² + b² + S²).
    pub fn total(&self) -> f64 {
        (self.a_priori().powi(2) + self.statistics * self.statistics).sqrt()
    }
    /// √(A² + b²): the approximations' part (module documentation, # The contract's shape).
    pub fn a_priori(&self) -> f64 {
        (self.model * self.model + self.step * self.step).sqrt()
    }
}

/// A solve's ledger (module documentation, # The contract's shape).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Ledger {
    pub items: Vec<Item>,
    pub totals: Totals,
    pub flagged: Option<String>,
    pub note: Option<String>,
}

impl Ledger {
    /// An approximation with its estimated error, not negative; one below u·ε, which adds nothing to the population
    /// target at the format's resolution (ε(1 + u) rounds to ε), is no item. An estimate that is no finite number (the
    /// contract's items are) is flagged as unread instead. Returns the item's place in `items` until they are closed, none
    /// where it is no item.
    pub fn item(&mut self, label: &str, estimate: f64, action: &str) -> Option<usize> {
        if !estimate.is_finite() {
            self.flag(format!("{label}：误差无法估计"));
            return None;
        }
        (estimate >= UNIT_ROUNDOFF * POPULATION_TARGET).then(|| {
            self.items.push(Item { label: label.into(), estimate, action: action.into() });
            self.items.len() - 1
        })
    }
    /// A reason a span of the outputs is less than the solve claims, after those before it.
    pub fn flag(&mut self, reason: String) {
        self.flagged = Some(match self.flagged.take() {
            Some(before) => format!("{before}；{reason}"),
            None => reason,
        });
    }
    /// The items from the largest estimate down, the totals and the flags of the outputs at `times` (µs): the populations'
    /// accounts at every output and the total T's (the mean over the lab axes: what T's target is on, SPEC §4) where it
    /// has one, against T's allowance at each output, `allowed` (relative; solve.rs, # T's allowance).
    pub fn close(&mut self, times: &[f64], population: &[Account], temperature: &[Option<Account>], allowed: &[f64]) {
        self.items.sort_by(|a, b| b.estimate.total_cmp(&a.estimate));
        let populations: Vec<Option<Account>> = population.iter().map(|&a| Some(a)).collect();
        let fixed = vec![POPULATION_TARGET; times.len()];
        self.totals = Totals { population: largest(&populations, &fixed), temperature: largest(temperature, allowed) };
        for (accounts, targets, what) in [(&populations[..], &fixed[..], "布居与丢失"), (temperature, allowed, "总温度")] {
            // Not within the target: beyond it, or no number. The worst: of the most of its target.
            let beyond: Vec<usize> =
                (0..times.len()).filter(|&k| accounts[k].is_some_and(|a| !(a.total() <= targets[k]))).collect();
            if let Some(&worst) = beyond.iter().max_by(|&&i, &&j| (accounts[i].unwrap().total() / targets[i]).total_cmp(&(accounts[j].unwrap().total() / targets[j]))) {
                let a = accounts[worst].unwrap();
                self.flag(format!(
                    "{}：{what}的估计误差超出目标，最大 {}（目标 {}；模型近似 {}、时间步长 {}、统计 {}）",
                    span(&beyond, times),
                    percent(a.total()),
                    percent(targets[worst]),
                    percent(a.model),
                    percent(a.step),
                    percent(a.statistics)
                ));
            }
        }
    }
}

/// The optical sector's noise over a run's trajectories (step.rs, # The optical sector's noise): `sums` = [Σ ε_Q tr C, Σ tr C,
/// the scattered light's Σ tr(2D)|τ|, the pending covariances' −tr left at the trajectories' ends] (1/µm²), C a step's
/// covariance of the noise. Two items, each relative to the momentum diffusion's whole input Σ tr C + Σ tr(2D)|τ| -- T's
/// relative error where the diffusion sets T (hypothesis: the steady state's T is linear in the diffusion; in heating the
/// energy's input is): K_Q's locality beyond its first order along the path, ε_Q = (ν/γ_Q)² of the light pattern's fastest
/// beat on the path ν = max |(k_l − k_m)·v| (none where the cluster is still) and the optical rate γ_Q = Γ/2 (step.rs, #
/// The optical sector's noise: the second order's bound for normal modes; the two-level atom's twelve cases at kv = Γ/10
/// within 0.53 ε_Q; for multilevel clusters a hypothesis), and the noise's covariance the draws left pending (none drawn
/// that is not the model's, nothing clipped). And the note on what the trajectory keeps of the ground manifold. Returns T's booked
/// relative error, both items'; nothing where no explicit cluster drew noise.
pub fn optical(ledger: &mut Ledger, sums: [f64; 4]) -> f64 {
    let whole = sums[1] + sums[2];
    if !(sums[1] != 0.0 && whole > 0.0) {
        return 0.0;
    }
    let (locality, debt) = (sums[0] / whole, sums[3].max(0.0) / whole);
    ledger.item("显式激发态的光学扇区动量噪声：K_Q 沿路径一阶之外的局域近似（激发态相干的记忆 1/(Γ/2) 内运动原子所见光场图样的拍频）", locality, "K 沿路径的一阶已精确计入；剩下的二阶每步按 (光束对的最大 |Δk·v| / (Γ/2))² × 该步的噪声协方差计入（正规模式下的推导界），对动量扩散总输入的比例计入温度的模型误差（两能级行波/驻波 12 例 kv = Γ/10 实测误差在该尺度的 0.53 倍以内；多能级簇为假设）");
    ledger.item("显式激发态的光学扇区动量噪声：尚未抽取的负协方差", debt, "瞬时为负的部分不截断，留作每条轨迹的待抽协方差；运行结束时剩余的部分对动量扩散总输入的比例计入温度的模型误差");
    let note = "显式激发态的簇：基态流形叠加态的力按平均场（Ehrenfest）沿轨迹携带，其关联中经过基态流形的部分（P 扇区）没有推导的误差界，是模型的已知局限（静止时 lin⊥lin 约为扩散的 −10 %）";
    ledger.note = Some(match ledger.note.take() {
        Some(before) => format!("{before}；{note}"),
        None => note.into(),
    });
    locality + debt
}

/// The a-priori part among the accounts given at the output where it takes most of its target there, `targets`
/// (module documentation, # The contract's shape), and that target; none given, no error, and T's target where it is
/// high (its allowance's limit).
fn largest(accounts: &[Option<Account>], targets: &[f64]) -> Total {
    let at = (0..accounts.len()).filter(|&k| accounts[k].is_some()).max_by(|&i, &j| (accounts[i].unwrap().a_priori() / targets[i]).total_cmp(&(accounts[j].unwrap().a_priori() / targets[j])));
    at.map_or(Total { error: 0., target: TEMPERATURE_TARGET }, |k| Total { error: accounts[k].unwrap().a_priori(), target: targets[k] })
}

/// A share in percent to two significant digits, the digits the app writes an error to (src/accuracy.ts `errorPercent`:
/// `formatNumber(100 error, {digits: 2})`).
pub fn percent(x: f64) -> String {
    format!("{}%", significant(100. * x, 2))
}
/// A time (µs) to three significant digits, the digits the app writes a sample's time to (src/accuracy.ts
/// `partialSummary`), in μs below a millisecond and in ms from there on.
pub fn when(us: f64) -> String {
    if us >= 1e3 {
        format!("{} ms", significant(us / 1e3, READING))
    } else {
        format!("{} μs", significant(us, READING))
    }
}
/// The significant digits a physical quantity is read to in the ledger's words (a time, a frequency): a display
/// convention, no result depends on it.
pub(crate) const READING: i32 = 3;
/// x rounded to `digits` significant digits: the nearest double to the rounded decimal, so that it prints as one.
pub(crate) fn significant(x: f64, digits: i32) -> f64 {
    if x == 0. || !x.is_finite() {
        return x;
    }
    let e = digits - 1 - x.abs().log10().floor() as i32;
    if e >= 0 {
        let scale = 10f64.powi(e);
        (x * scale).round() / scale
    } else {
        let scale = 10f64.powi(-e);
        (x / scale).round() * scale
    }
}
/// The outputs `at` (indices in order, at least one) in the app's words: from the first on where they run through the
/// last output; from the first to the last where they run without a gap; else between the two, their count.
pub fn span(at: &[usize], times: &[f64]) -> String {
    let (first, last) = (at[0], at[at.len() - 1]);
    if at.len() == last - first + 1 {
        if last == times.len() - 1 && first > 0 {
            format!("{} 起", when(times[first]))
        } else if first == last {
            when(times[first])
        } else {
            format!("{} 至 {}", when(times[first]), when(times[last]))
        }
    } else {
        format!("{} 至 {} 之间 {} 个时刻", when(times[first]), when(times[last]), at.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn account(model: f64, step: f64, statistics: f64) -> Account {
        Account { model, step, statistics }
    }

    /// The total is the three parts in quadrature.
    #[test]
    fn the_total_is_its_parts_in_quadrature() {
        assert!((account(0.03, 0.04, 0.12).total() - 0.13).abs() < 1e-16);
        assert_eq!(Account::default().total(), 0.);
    }

    /// The totals take the a-priori part where it takes most of its target; the outputs beyond the target are flagged as
    /// one span, with the largest and its parts -- what is shown, and the total T where it has an account (an output
    /// without one is neither totalled nor flagged); nothing flagged within the targets, and no note.
    #[test]
    fn the_ledger_totals_and_flags_the_outputs_beyond_their_targets() {
        let times: Vec<f64> = (0..6).map(|k| 250. * k as f64).collect();
        let within = vec![account(0.01, 0.0224, 0.03); 6];
        let read = vec![Some(account(0.01, 0.0447, 0.05)); 6];
        let mut quiet = Ledger::default();
        quiet.close(&times, &within, &read, &[TEMPERATURE_TARGET; 6]);
        assert_eq!((quiet.flagged.as_ref(), quiet.note.as_ref()), (None, None));
        assert!((quiet.totals.population.error - within[0].a_priori()).abs() < 1e-16 && quiet.totals.population.target == POPULATION_TARGET);

        let mut population = within.clone();
        population[3] = account(0.01, 0.0224, 0.06);
        population[5] = account(0.01, 0.0224, 0.05);
        let mut temperature = read.clone();
        temperature[2] = Some(account(0.0, 0.0447, 0.2));
        temperature[4] = Some(account(0.06, 0.0447, 0.02));
        temperature[5] = None;
        let mut l = Ledger::default();
        l.close(&times, &population, &temperature, &[TEMPERATURE_TARGET; 6]);
        assert!((l.totals.population.error - population[3].a_priori()).abs() < 1e-16);
        assert_eq!(l.totals.temperature.error, temperature[4].unwrap().a_priori(), "the statistics take no part");
        let flagged = l.flagged.unwrap();
        assert!(flagged.starts_with("750 μs 至 1.25 ms 之间 2 个时刻：布居与丢失的估计误差超出目标，最大 6.5%（目标 5%；模型近似 1%、时间步长 2.2%、统计 6%）"), "{flagged}");
        assert!(flagged.ends_with("；500 μs：总温度的估计误差超出目标，最大 20%（目标 10%；模型近似 0%、时间步长 4.5%、统计 20%）"), "{flagged}");
    }

    /// An item below u·ε is none, the others have their places; the items come out largest first.
    #[test]
    fn items_are_the_approximations_that_count() {
        let mut l = Ledger::default();
        assert_eq!((l.item("small", 1e-3, ""), l.item("nothing", 1e-32, ""), l.item("step", 0.022, "")), (Some(0), None, Some(1)));
        l.close(&[0., 1.], &[Account::default(); 2], &[None; 2], &[TEMPERATURE_TARGET; 2]);
        assert_eq!(l.items.iter().map(|i| i.label.as_str()).collect::<Vec<_>>(), ["step", "small"]);
    }

    /// Spans and numbers in the app's words.
    #[test]
    fn spans_and_numbers_read_as_the_app_writes_them() {
        let times: Vec<f64> = (0..5).map(|k| 0.5 * k as f64 * 1e3).collect();
        assert_eq!(span(&[2, 3, 4], &times), "1 ms 起");
        assert_eq!(span(&[1, 2], &times), "500 μs 至 1 ms");
        assert_eq!(span(&[0, 4], &times), "0 μs 至 2 ms 之间 2 个时刻");
        assert_eq!(span(&[3], &times), "1.5 ms");
        assert_eq!(span(&[0, 1, 2, 3, 4], &times), "0 μs 至 2 ms");
        assert_eq!((when(0.125), when(12.34), when(999.4), when(12_345.)), ("0.125 μs".into(), "12.3 μs".into(), "999 μs".into(), "12.3 ms".into()));
        assert_eq!((percent(0.0723), percent(0.000_123), percent(0.)), ("7.2%".into(), "0.012%".into(), "0%".into()));
    }
}
