//! The ledger: model, step and statistics errors per output, items, flags (SPEC §4).
//!
//! # The total error
//! An output's error against its target has three parts from independent sources, combined in quadrature (GUM):
//! E² = A² + b² + S² (`Account`):
//! - A, the model's: each unit's error measured along the trajectories (SPEC §5.2) and the unseen share's bound;
//! - b, the step's: the bound its tolerance gives, s·ε (SPEC §2.2), standing unless a falsifier refutes it (SPEC §5.3);
//! - S, the statistics': the interval at the confidence z (estimate.rs).
//! The populations' are absolute against `POPULATION_TARGET`, every row as shown and the loss; T's relative against
//! `TEMPERATURE_TARGET` at the outputs it is held to (solve.rs).
//!
//! # The contract's shape (contract §2.2)
//! `totals`: the populations' and T's total at the output where it takes most of its target, with the target; `items`:
//! each approximation's label, its estimated error (its largest over the outputs on what is shown, so that an output
//! without atoms takes no part) and what the engine did with it, in Chinese as the app shows them; `flagged`: where
//! some output's total exceeds its target -- the span, the largest total and its three parts --, and every other reason
//! a span is less than the solve claims; `note`: what is true of the setup whatever the solve does, never an accuracy
//! conclusion.
use crate::units::{POPULATION_TARGET, TEMPERATURE_TARGET};

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

/// One output's error against its target in its three parts (module documentation, # The total error).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Account {
    pub model: f64,
    pub step: f64,
    pub statistics: f64,
}
impl Account {
    /// √(A² + b² + S²).
    pub fn total(&self) -> f64 {
        (self.model * self.model + self.step * self.step + self.statistics * self.statistics).sqrt()
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
    /// An approximation with its estimated error, finite and not negative (the contract's).
    pub fn item(&mut self, label: &str, estimate: f64, action: &str) {
        debug_assert!(estimate.is_finite() && estimate >= 0., "{label}: {estimate}");
        self.items.push(Item { label: label.into(), estimate, action: action.into() });
    }
    /// A reason a span of the outputs is less than the solve claims, after those before it.
    pub fn flag(&mut self, reason: String) {
        self.flagged = Some(match self.flagged.take() {
            Some(before) => format!("{before}；{reason}"),
            None => reason,
        });
    }
    /// The totals and the flags of the outputs at `times` (µs): the populations' accounts at every output; T's where it
    /// is read, `held` where it is held to its target. T's total is taken over the outputs it is held to, over every one
    /// where it is read when it is held to none (its error is then the solve's, not a target's), and only the outputs it
    /// is held to are flagged.
    pub fn close(&mut self, times: &[f64], population: &[Account], temperature: &[Option<Account>], held: &[bool]) {
        let held_to = held.iter().zip(temperature).any(|(&h, t)| h && t.is_some());
        let temperatures: Vec<Option<Account>> =
            temperature.iter().zip(held).map(|(t, &h)| t.filter(|_| h || !held_to)).collect();
        let populations: Vec<Option<Account>> = population.iter().map(|&a| Some(a)).collect();
        self.totals = Totals { population: largest(&populations, POPULATION_TARGET), temperature: largest(&temperatures, TEMPERATURE_TARGET) };
        let everywhere = vec![true; times.len()];
        for (accounts, target, flaggable, what) in
            [(&populations, POPULATION_TARGET, &everywhere[..], "布居与损失"), (&temperatures, TEMPERATURE_TARGET, held, "温度")]
        {
            let beyond: Vec<usize> =
                (0..times.len()).filter(|&k| flaggable[k] && accounts[k].is_some_and(|a| a.total() > target)).collect();
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

/// The total that takes most of the target among the accounts given, and the target; none given, no error.
fn largest(accounts: &[Option<Account>], target: f64) -> Total {
    let error = accounts.iter().flatten().map(Account::total).fold(0., f64::max);
    Total { error, target }
}

/// A share in percent to two significant digits, as the app writes its errors.
pub fn percent(x: f64) -> String {
    format!("{}%", significant(100. * x, 2))
}
/// A time (µs) in the app's words: µs below a millisecond, ms above, to three significant digits.
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

    /// The totals take the output where each takes most of its target; the outputs beyond it are flagged as one span,
    /// with the largest and its parts; T is totalled and flagged only where it is held to its target, and totalled
    /// where it is read when it is held to none; nothing flagged within the targets, and no note.
    #[test]
    fn the_ledger_totals_and_flags_the_outputs_beyond_their_targets() {
        let times: Vec<f64> = (0..6).map(|k| 250. * k as f64).collect();
        let within = vec![account(0.01, 0.0224, 0.03); 6];
        let read = vec![Some(account(0.01, 0.0447, 0.05)); 6];
        let mut quiet = Ledger::default();
        quiet.close(&times, &within, &read, &[true; 6]);
        assert_eq!((quiet.flagged.as_ref(), quiet.note.as_ref()), (None, None));
        assert!((quiet.totals.population.error - within[0].total()).abs() < 1e-16 && quiet.totals.population.target == POPULATION_TARGET);

        let mut population = within.clone();
        population[3] = account(0.01, 0.0224, 0.06);
        population[5] = account(0.01, 0.0224, 0.05);
        let mut temperature = read.clone();
        temperature[2] = Some(account(0.0, 0.0447, 0.2));
        temperature[4] = Some(account(0.0, 0.0447, 0.3));
        let held = [true, true, true, true, false, true];
        let mut l = Ledger::default();
        l.close(&times, &population, &temperature, &held);
        assert!((l.totals.population.error - population[3].total()).abs() < 1e-16);
        assert!((l.totals.temperature.error - temperature[2].unwrap().total()).abs() < 1e-16, "the unheld output takes no part");
        let flagged = l.flagged.unwrap();
        assert!(flagged.starts_with("750 μs 至 1.25 ms 之间 2 个时刻：布居与损失的估计误差超出目标，最大 6.5%（目标 5%；模型 1%、步长 2.2%、统计 6%）"), "{flagged}");
        assert!(flagged.ends_with("；500 μs：温度的估计误差超出目标，最大 20%（目标 10%；模型 0%、步长 4.5%、统计 20%）"), "{flagged}");

        let mut none_held = Ledger::default();
        none_held.close(&times, &within, &temperature, &[false; 6]);
        assert!((none_held.totals.temperature.error - temperature[4].unwrap().total()).abs() < 1e-16 && none_held.flagged.is_none());
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
