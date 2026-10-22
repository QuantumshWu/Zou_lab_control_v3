//! The live board's display cadence (v2 zlc_runtime.presentation HarmonicClock and
//! BoardScheduler): a panel's update interval caps how often it takes new data. Its
//! interval's crossings count on one board clock, so panels of one interval take
//! new data in the same turn and a slower harmonic interval's crossings are the
//! faster's: panels showing one shot group flip together (a panel's own phase, as
//! from the picture it last staged, kept a panel added later a turn apart from the
//! others for good). A panel is due once a crossing has passed since it last staged;
//! a panel never staged joins the board's next crossing (at once on a board that
//! has shown nothing). A late turn shows the newest data and never replays the shots
//! it missed. A panel inside its interval keeps the data it shows, so its fit and
//! readouts do not move.
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

#[derive(Default)]
pub struct Cadence {
    /// The board clock every interval's crossings count from.
    origin: Option<Instant>,
    /// When each panel last staged new data.
    staged: BTreeMap<String, Instant>,
    /// When the board's current job was dispatched (a panel it presents staged then).
    dispatched: Option<Instant>,
    /// When a panel held back may take its new data.
    next: Option<Instant>,
}

impl Cadence {
    /// The board panels (id -> interval ms) that may not take new data now, or
    /// `None` when none may (the board then waits for the first that may).
    pub fn plan(&mut self, intervals: &BTreeMap<String, u64>, now: Instant) -> Option<BTreeSet<String>> {
        self.staged.retain(|panel, _| intervals.contains_key(panel));
        let origin = *self.origin.get_or_insert(now);
        let board = self.staged.values().max().copied();
        // The first crossing of `interval` after `at`.
        let crossing = |at: Instant, interval: u64| {
            let interval = interval.max(1);
            origin + Duration::from_millis((at.saturating_duration_since(origin).as_millis() as u64 / interval + 1) * interval)
        };
        let due_at = |panel: &str, interval: u64| self.staged.get(panel).copied().or(board).map(|at| crossing(at, interval));
        let held: BTreeMap<&str, Instant> = intervals.iter()
            .filter_map(|(panel, interval)| due_at(panel, *interval).filter(|at| *at > now).map(|at| (panel.as_str(), at))).collect();
        self.next = held.values().min().copied();
        if !intervals.is_empty() && held.len() == intervals.len() { return None; }
        self.dispatched = Some(now);
        Some(held.into_keys().map(str::to_owned).collect())
    }

    /// A command of the board's own (an operator's edit) stages its panel now, as v2's owed presentation.
    pub fn dispatch(&mut self, now: Instant) { self.dispatched = Some(now); }

    /// The panels a completed job put new pictures of on screen staged at its dispatch.
    pub fn presented<'a>(&mut self, panels: impl IntoIterator<Item = &'a String>) {
        let Some(at) = self.dispatched else { return };
        for panel in panels { self.staged.insert(panel.clone(), at); }
    }

    /// Whether a held panel's interval has passed (once: the board then syncs).
    pub fn owed(&mut self, now: Instant) -> bool {
        if self.next.is_some_and(|at| at <= now) { self.next = None; return true; }
        false
    }

    /// When the board next needs a turn for a held panel.
    pub fn wake_at(&self) -> Option<Instant> { self.next }
}

/// The board's turn at [`Cadence::wake_at`], on time. The event loop's own timed
/// wait lands on the system timer's tick (~15.6 ms on Windows), which made a
/// 100 ms interval ~108 ms; this thread waits coarsely until shortly before the
/// deadline, sleeps the rest with `std::thread::sleep` (a high-resolution
/// waitable timer on Windows) and then wakes the loop.
pub struct Waker { deadline: Arc<(Mutex<Option<Instant>>, Condvar)> }

impl Waker {
    /// Longer than the coarse wait's worst lateness.
    const COARSE: Duration = Duration::from_millis(20);
    /// The final sleep's slice: a deadline moved earlier is seen within it.
    const FINE: Duration = Duration::from_millis(4);

    pub fn new(wake: impl Fn() + Send + 'static) -> std::io::Result<Waker> {
        let deadline = Arc::new((Mutex::new(None::<Instant>), Condvar::new()));
        let shared = deadline.clone();
        std::thread::Builder::new().name("zlc-cadence-wake".into()).spawn(move || {
            let (slot, changed) = &*shared;
            let mut due = slot.lock().unwrap_or_else(|error| error.into_inner());
            loop {
                let Some(at) = *due else {
                    due = changed.wait(due).unwrap_or_else(|error| error.into_inner());
                    continue;
                };
                let now = Instant::now();
                if now >= at {
                    *due = None;
                    drop(due);
                    wake();
                    due = slot.lock().unwrap_or_else(|error| error.into_inner());
                } else if at - now > Self::COARSE {
                    due = changed.wait_timeout(due, at - now - Self::COARSE).unwrap_or_else(|error| error.into_inner()).0;
                } else {
                    drop(due);
                    std::thread::sleep((at - now).min(Self::FINE));
                    due = slot.lock().unwrap_or_else(|error| error.into_inner());
                }
            }
        })?;
        Ok(Waker { deadline })
    }

    /// Wake the loop at `at` (None: not at all), replacing the deadline set before.
    pub fn arm(&self, at: Option<Instant>) {
        let (slot, changed) = &*self.deadline;
        let mut due = slot.lock().unwrap_or_else(|error| error.into_inner());
        if *due != at { *due = at; changed.notify_one(); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn board(entries: &[(&str, u64)]) -> BTreeMap<String, u64> { entries.iter().map(|(panel, ms)| ((*panel).to_owned(), *ms)).collect() }

    #[test]
    fn intervals_cross_on_one_board_clock() {
        let mut cadence = Cadence::default();
        let intervals = board(&[("panel-1", 100), ("panel-2", 800)]);
        let start = Instant::now();
        // A board that has shown nothing: both due at once.
        assert_eq!(cadence.plan(&intervals, start), Some(BTreeSet::new()));
        cadence.presented(&["panel-1".to_owned(), "panel-2".to_owned()]);
        // 150 ms on: panel-1 has crossed 100, panel-2 is held until 800.
        let held = cadence.plan(&intervals, start + Duration::from_millis(150)).unwrap();
        assert_eq!(held, ["panel-2".to_owned()].into());
        assert_eq!(cadence.wake_at(), Some(start + Duration::from_millis(800)));
        cadence.presented(&["panel-1".to_owned()]);
        // A late turn does not move the clock: panel-1, staged at 150, is due at the 200 crossing.
        assert_eq!(cadence.plan(&intervals, start + Duration::from_millis(190)), None);
        assert_eq!(cadence.wake_at(), Some(start + Duration::from_millis(200)));
        assert!(!cadence.owed(start + Duration::from_millis(199)));
        assert!(cadence.owed(start + Duration::from_millis(200)));
        // A panel added later joins the board's next crossing, in step with panel-1.
        let intervals = board(&[("panel-1", 100), ("panel-2", 800), ("panel-3", 100)]);
        let held = cadence.plan(&intervals, start + Duration::from_millis(200)).unwrap();
        assert_eq!(held, ["panel-2".to_owned()].into());
        cadence.presented(&["panel-1".to_owned(), "panel-3".to_owned()]);
        // The 800 crossing is every interval's: all three in one turn.
        assert_eq!(cadence.plan(&intervals, start + Duration::from_millis(800)), Some(BTreeSet::new()));
        // A due panel that showed nothing new keeps its old staging: due at once later.
        cadence.presented(&[]);
        assert_eq!(cadence.plan(&intervals, start + Duration::from_millis(810)), Some(BTreeSet::new()));
    }
}
