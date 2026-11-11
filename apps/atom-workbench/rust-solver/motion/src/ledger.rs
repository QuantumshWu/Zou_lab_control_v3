//! The ledger: model, step and statistics errors per output, items, flags (SPEC §4).
//!
//! # The total error
//! An output's error against its target has three parts from independent sources, combined in quadrature (GUM):
//! E² = A² + b² + S² (`Account`):
//! - A, the model's: each unit's error measured along the trajectories (SPEC §5.2), and the share of the thermal ensemble
//!   that the model's reach leaves out -- the unseen share at the most trajectories a solve runs (SPEC §1.8);
//! - b, the discretization's, the part a finer one takes away: the step's, what the pilot's check between two steps
//!   reads clearly of its error at the output (SPEC §2.2: a check at the pilot's resolution, not a bound; the step's
//!   share until it is read); and at t = 0, where T is the
//!   initial ensemble's exact moment by quadrature, that quadrature's last two levels' difference (solve.rs
//!   `Fixed::start`);
//! - S, the statistics': the jackknife's interval at the confidence z, on what is shown no less than what the run's own
//!   trajectories leave unseen -- an event none of them showed has no spread to read (estimate.rs
//!   `Estimate::shown_statistics`). The same function as A's share, another error: that one is the model's, at the most
//!   trajectories; this one the sample's, at its own number.
//! The populations' are absolute against `POPULATION_TARGET`, every row as shown and the loss; T's relative, per lab
//! axis, against `TEMPERATURE_TARGET` at the outputs that axis is held to (solve.rs).
//!
//! # The contract's shape (contract §2.2)
//! `totals`: the populations' and T's a-priori part √(A² + b²) -- what the approximations leave, the statistics aside (the
//! result reports T's statistical error itself, src/solver-types.ts `LedgerTotal`) -- at the output where it is largest,
//! T's over every output and lab axis where T has an account, with the target; `items`:
//! each approximation's label, its estimated error (its largest over the outputs on what is shown, so that an output
//! without atoms takes no part) and what the engine did to get it, in Chinese as the app shows them; `flagged`: where
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
            self.flag(format!("{label}：误差读不出"));
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
    /// accounts at every output; T's per lab axis where it has one (SPEC §4), `held` where that axis is held to its
    /// target. T's a-priori total is taken over every output and axis where T has an account, held to its target or not
    /// -- what the approximations leave of the T that is shown --; an axis is flagged against the target only at the
    /// outputs it is held to, each axis by its name.
    pub fn close(&mut self, times: &[f64], population: &[Account], temperature: &[[Option<Account>; 3]], held: &[[bool; 3]]) {
        self.items.sort_by(|a, b| b.estimate.total_cmp(&a.estimate));
        let populations: Vec<Option<Account>> = population.iter().map(|&a| Some(a)).collect();
        let axes: [Vec<Option<Account>>; 3] = std::array::from_fn(|a| temperature.iter().map(|t| t[a]).collect());
        let held: [Vec<bool>; 3] = std::array::from_fn(|a| held.iter().map(|h| h[a]).collect());
        let of_axes = Total { error: axes.iter().map(|accounts| largest(accounts, TEMPERATURE_TARGET).error).fold(0., f64::max), target: TEMPERATURE_TARGET };
        self.totals = Totals { population: largest(&populations, POPULATION_TARGET), temperature: of_axes };
        let everywhere = vec![true; times.len()];
        for (accounts, target, flaggable, what) in [
            (&populations[..], POPULATION_TARGET, &everywhere[..], "布居与损失"),
            (&axes[0][..], TEMPERATURE_TARGET, &held[0][..], "温度 T_x "),
            (&axes[1][..], TEMPERATURE_TARGET, &held[1][..], "温度 T_y "),
            (&axes[2][..], TEMPERATURE_TARGET, &held[2][..], "温度 T_z "),
        ] {
            // Not within the target: beyond it, or no number.
            let beyond: Vec<usize> =
                (0..times.len()).filter(|&k| flaggable[k] && accounts[k].is_some_and(|a| !(a.total() <= target))).collect();
            if let Some(&worst) = beyond.iter().max_by(|&&i, &&j| accounts[i].unwrap().total().total_cmp(&accounts[j].unwrap().total())) {
                let a = accounts[worst].unwrap();
                self.flag(format!(
                    "{}：{what}的估计误差超出目标，最大 {}（目标 {}；模型 {}、步长 {}、统计 {}）",
                    span(&beyond, times),
                    percent(a.total()),
                    percent(target),
                    percent(a.model),
                    percent(a.step),
                    percent(a.statistics)
                ));
            }
        }
    }
}

/// The largest a-priori part among the accounts given (module documentation, # The contract's shape), and the target; none
/// given, no error.
fn largest(accounts: &[Option<Account>], target: f64) -> Total {
    let error = accounts.iter().flatten().map(Account::a_priori).fold(0., f64::max);
    Total { error, target }
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
        format!("{} ms", significant(us / 1e3, 3))
    } else {
        format!("{} μs", significant(us, 3))
    }
}
/// x rounded to `digits` significant digits: the nearest double to the rounded decimal, so that it prints as one.
fn significant(x: f64, digits: i32) -> f64 {
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

    /// The totals take the a-priori part where it takes most of its target; the outputs beyond the target are flagged as one span,
    /// with the largest and its parts; T is totalled wherever an axis has an account -- an output the axis is not held to
    /// at, and all of them when it is held to none -- and flagged per axis, only where that axis is held to its target: an
    /// axis beyond the target at an output it is not held to is not flagged, whatever the other axes are held to there;
    /// nothing flagged within the targets, and no note.
    #[test]
    fn the_ledger_totals_and_flags_the_outputs_beyond_their_targets() {
        let times: Vec<f64> = (0..6).map(|k| 250. * k as f64).collect();
        let within = vec![account(0.01, 0.0224, 0.03); 6];
        let read = vec![[Some(account(0.01, 0.0447, 0.05)); 3]; 6];
        let mut quiet = Ledger::default();
        quiet.close(&times, &within, &read, &[[true; 3]; 6]);
        assert_eq!((quiet.flagged.as_ref(), quiet.note.as_ref()), (None, None));
        assert!((quiet.totals.population.error - within[0].a_priori()).abs() < 1e-16 && quiet.totals.population.target == POPULATION_TARGET);

        let mut population = within.clone();
        population[3] = account(0.01, 0.0224, 0.06);
        population[5] = account(0.01, 0.0224, 0.05);
        let mut temperature = read.clone();
        temperature[2][0] = Some(account(0.0, 0.0447, 0.2));
        temperature[4][1] = Some(account(0.06, 0.0447, 0.3));
        temperature[5][2] = None;
        let mut held = [[true; 3]; 6];
        held[4][1] = false;
        let mut l = Ledger::default();
        l.close(&times, &population, &temperature, &held);
        assert!((l.totals.population.error - population[3].a_priori()).abs() < 1e-16);
        assert_eq!(l.totals.temperature.error, temperature[4][1].unwrap().a_priori(), "the output the axis is not held to at; the statistics take no part");
        let flagged = l.flagged.unwrap();
        assert!(flagged.starts_with("750 μs 至 1.25 ms 之间 2 个时刻：布居与损失的估计误差超出目标，最大 6.5%（目标 5%；模型 1%、步长 2.2%、统计 6%）"), "{flagged}");
        assert!(flagged.ends_with("；500 μs：温度 T_x 的估计误差超出目标，最大 20%（目标 10%；模型 0%、步长 4.5%、统计 20%）"), "{flagged}");

        // The same axis held there: flagged, by its name, beside the other axis's.
        held[4][1] = true;
        let mut both = Ledger::default();
        both.close(&times, &population, &temperature, &held);
        assert!(both.flagged.unwrap().ends_with("；500 μs：温度 T_x 的估计误差超出目标，最大 20%（目标 10%；模型 0%、步长 4.5%、统计 20%）；1 ms：温度 T_y 的估计误差超出目标，最大 31%（目标 10%；模型 6%、步长 4.5%、统计 30%）"));

        let mut none_held = Ledger::default();
        none_held.close(&times, &within, &temperature, &[[false; 3]; 6]);
        assert_eq!((none_held.totals.temperature, none_held.flagged), (l.totals.temperature, None));
    }

    /// An item below u·ε is none, the others have their places; the items come out largest first.
    #[test]
    fn items_are_the_approximations_that_count() {
        let mut l = Ledger::default();
        assert_eq!((l.item("small", 1e-3, ""), l.item("nothing", 1e-32, ""), l.item("step", 0.022, "")), (Some(0), None, Some(1)));
        l.close(&[0., 1.], &[Account::default(); 2], &[[None; 3]; 2], &[[false; 3]; 2]);
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
