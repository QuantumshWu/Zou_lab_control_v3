//! Opt-in timing of the renderer's phases (`ZLC_RENDER_PROFILE` set): each
//! [`span`] adds its wall time to a process-wide table [`report`] reads.
use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("ZLC_RENDER_PROFILE").is_some())
}

fn table() -> &'static Mutex<BTreeMap<&'static str, (f64, usize)>> {
    static TABLE: OnceLock<Mutex<BTreeMap<&'static str, (f64, usize)>>> = OnceLock::new();
    TABLE.get_or_init(Default::default)
}

pub struct Span(&'static str, Instant);

impl Drop for Span {
    fn drop(&mut self) {
        let ms = self.1.elapsed().as_secs_f64() * 1e3;
        let mut table = table().lock().unwrap();
        let entry = table.entry(self.0).or_insert((0.0, 0));
        entry.0 += ms;
        entry.1 += 1;
    }
}

/// Time the enclosing scope under `name` when profiling is on.
pub fn span(name: &'static str) -> Option<Span> { enabled().then(|| Span(name, Instant::now())) }

/// (name, total ms, count) per phase, and clear the table.
pub fn report() -> Vec<(&'static str, f64, usize)> {
    let mut table = table().lock().unwrap();
    let out = table.iter().map(|(k, v)| (*k, v.0, v.1)).collect();
    table.clear();
    out
}
