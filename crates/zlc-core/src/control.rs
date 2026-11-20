//! Who sent a command, and what Stop all makes safe.
//!
//! The operator at the window and a remote (MCP/AI) client send the same
//! commands; whether a remote client may change anything is the endpoint's one
//! switch (v2 `allow_control`), not a property of the instrument. Outputs
//! register how they go safe; Stop all runs every one of them and is never
//! refused.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, Mutex};

/// Who issued a command.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Origin {
    /// The operator at the instrument's own window.
    Operator,
    /// An automated client (MCP/AI, a script), by its declared name.
    Remote(String),
}

impl Origin {
    pub fn is_operator(&self) -> bool { matches!(self, Origin::Operator) }
}

impl fmt::Display for Origin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self { Origin::Operator => f.write_str("operator"), Origin::Remote(name) => write!(f, "remote client '{name}'") }
    }
}

/// Something that can drive its outputs to their safe state immediately.
pub type SafeAction = Arc<dyn Fn() -> Result<(), String> + Send + Sync>;

/// The outputs Stop all drives safe, by name. Cheap to clone; clones share it.
#[derive(Clone, Default)]
pub struct SafeActions { actions: Arc<Mutex<BTreeMap<String, SafeAction>>> }

impl SafeActions {
    pub fn new() -> Self { Self::default() }

    fn lock(&self) -> std::sync::MutexGuard<'_, BTreeMap<String, SafeAction>> {
        self.actions.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub fn register(&self, output: &str, safe: SafeAction) { self.lock().insert(output.to_owned(), safe); }

    pub fn unregister(&self, output: &str) { self.lock().remove(output); }

    /// The registered outputs' names.
    pub fn outputs(&self) -> Vec<String> { self.lock().keys().cloned().collect() }

    /// Drive every registered output safe, all at once and outside the lock (a device may
    /// report back): every action runs even when another fails or is slow to answer, and
    /// none waits behind another. Returns the outputs that did not report safe, with the error.
    pub fn make_safe(&self) -> Vec<String> {
        let actions: Vec<(String, SafeAction)> = self.lock().iter().map(|(name, action)| (name.clone(), action.clone())).collect();
        let said = |name: &str, outcome: std::thread::Result<Result<(), String>>| match outcome {
            Ok(Ok(())) => None,
            Ok(Err(error)) => Some(format!("{name}: {error}")),
            Err(_) => Some(format!("{name}: its safe action failed unexpectedly")),
        };
        std::thread::scope(|scope| {
            let running: Vec<(&String, Result<std::thread::ScopedJoinHandle<'_, Result<(), String>>, SafeAction>)> = actions.iter().map(|(name, action)| {
                let asked = action.clone();
                // A thread that cannot start still has its output asked, here.
                (name, std::thread::Builder::new().name(format!("zlc-safe-{name}")).spawn_scoped(scope, move || asked()).map_err(|_| action.clone()))
            }).collect();
            running.into_iter().filter_map(|(name, running)| match running {
                Ok(handle) => said(name, handle.join()),
                Err(action) => said(name, Ok(action())),
            }).collect()
        })
    }

    /// [`SafeActions::make_safe`] off the caller's thread (the window's turn goes on meanwhile):
    /// `done` is told the outputs that did not report safe.
    pub fn make_safe_then(&self, done: impl FnOnce(Vec<String>) + Send + 'static) {
        fn tell<F: FnOnce(Vec<String>)>(slot: &Mutex<Option<F>>, failures: Vec<String>) {
            let done = slot.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).take();
            if let Some(done) = done { done(failures); }
        }
        let slot = Arc::new(Mutex::new(Some(done)));
        let (safe, told) = (self.clone(), slot.clone());
        let spawned = std::thread::Builder::new().name("zlc-stop-all".into()).spawn(move || tell(&told, safe.make_safe()));
        // No thread for it: every output is still asked, here.
        if spawned.is_err() { tell(&slot, self.make_safe()); }
    }
}

/// The MCP tool name of an open instance's command: `<instance>__<command>`, each part
/// as [`tool_word`] spells it. Only the tool name is restricted; the instance keeps the
/// id its apparatus gives it everywhere else.
pub fn tool_name(instance: &str, command: &str) -> String { format!("{}__{}", tool_word(instance), tool_word(command)) }

/// A logic node's plugin instance is named under this prefix, devices under their own ids:
/// a device and a node may share an id (v2 kept them apart), and an id never holds `/`.
pub const NODE_INSTANCE: &str = "node/";

/// The plugin instance of logic node `node`.
pub fn node_instance(node: &str) -> String { format!("{NODE_INSTANCE}{node}") }

/// The id an instance is known by outside the plugin host: a node's without its prefix.
pub fn public_instance(instance: &str) -> &str { instance.strip_prefix(NODE_INSTANCE).unwrap_or(instance) }

/// One part of a tool name. MCP clients take tool names of `[A-Za-z0-9_-]` and limit
/// their length, while an id is whatever its apparatus says (v2: any text without `/`).
/// A part that already is such a word -- letters, digits and single `_`, not ending in
/// `_`, at most 24 characters -- stands as it is (`n100_COM3`). Any other is its letters
/// and digits (runs of anything else as one `_`, at most 15), `-`, and eight hex digits
/// hashed from its exact text (`pump_laser_2-1f0c93ab`): the same in every session, never
/// equal to a word that stood as it is (those have no `-`), and the first `__` of a tool
/// name always separates its two parts.
pub fn tool_word(text: &str) -> String {
    const LONGEST: usize = 24;
    let word = text.bytes().all(|byte| byte.is_ascii_alphanumeric() || byte == b'_');
    if word && !text.is_empty() && text.len() <= LONGEST && !text.contains("__") && !text.ends_with('_') { return text.to_owned(); }
    let mut stem = String::new();
    for character in text.chars() {
        if character.is_ascii_alphanumeric() { stem.push(character); } else if !stem.is_empty() && !stem.ends_with('_') { stem.push('_'); }
    }
    stem.truncate(LONGEST - 9);
    let stem = match stem.trim_end_matches('_') { "" => "id", stem => stem };
    // FNV-1a over the exact bytes, folded to 32 bits.
    let hash = text.bytes().fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3));
    format!("{stem}-{:08x}", (hash ^ (hash >> 32)) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn tool_names_keep_plain_ids_and_spell_any_other_safely() {
        assert_eq!(tool_name("n100_COM3", "editor_open"), "n100_COM3__editor_open");
        assert_eq!(tool_name("camera", "set_exposure"), "camera__set_exposure");
        let ids = ["n100 COM3", "n100-COM3", "相机", "dg4000_TCPIP0__192_168_1_5__inst0__INSTR", "camera_", "a__b", "pump laser #2",
                   "a_very_long_device_name_of_thirty", "n100_COM3", "COM3", "com3"];
        let words: Vec<String> = ids.iter().map(|id| tool_word(id)).collect();
        for (id, word) in ids.iter().zip(&words) {
            assert!(!word.is_empty() && word.len() <= 24, "{id:?} -> {word:?}");
            assert!(word.bytes().all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-'), "{id:?} -> {word:?}");
            assert!(!word.contains("__") && !word.ends_with('_'), "{id:?} -> {word:?}");
            assert_eq!(&tool_word(id), word, "stable");
        }
        let distinct: BTreeMap<&String, &str> = words.iter().zip(ids).collect();
        assert_eq!(distinct.len(), ids.len(), "{words:?}");
        assert_eq!(tool_word("n100 COM3").split_once('-').unwrap().0, "n100_COM3");
        assert_eq!(tool_word("相机").split_once('-').unwrap().0, "id");
        // The first `__` separates the parts whatever the instance was called.
        let name = tool_name("a__b", "c");
        assert_eq!(name.split_once("__").unwrap(), (tool_word("a__b").as_str(), "c"));
        assert_ne!(name, tool_name("a", "b__c"));
    }

    #[test]
    fn stop_all_asks_every_output_at_once_and_reports_off_the_callers_thread() {
        use std::sync::mpsc;
        use std::time::{Duration, Instant};
        // Each output answers only once the other has been asked: one after the other, neither would.
        let safe = SafeActions::new();
        let (a_asked, a_heard) = mpsc::channel::<()>();
        let (b_asked, b_heard) = mpsc::channel::<()>();
        let (a_heard, b_heard) = (Mutex::new(a_heard), Mutex::new(b_heard));
        safe.register("coil-a", Arc::new(move || { a_asked.send(()).unwrap(); b_heard.lock().unwrap().recv_timeout(Duration::from_secs(5)).map_err(|_| "b was never asked".to_owned()) }));
        safe.register("coil-b", Arc::new(move || { b_asked.send(()).unwrap(); a_heard.lock().unwrap().recv_timeout(Duration::from_secs(5)).map_err(|_| "a was never asked".to_owned()) }));
        safe.register("board", Arc::new(|| Err("board not answering".into())));
        let (told, failures) = mpsc::channel();
        let asked = Instant::now();
        safe.make_safe_then(move |failed| told.send(failed).unwrap());
        assert!(asked.elapsed() < Duration::from_millis(500), "the caller does not wait for the outputs");
        assert_eq!(failures.recv_timeout(Duration::from_secs(10)).unwrap(), vec!["board: board not answering".to_owned()]);
    }

    #[test]
    fn stop_all_runs_every_safe_action_even_after_a_failure() {
        let safe = SafeActions::new();
        let count = Arc::new(AtomicUsize::new(0));
        for name in ["coil-x", "coil-y"] {
            let counter = count.clone();
            safe.register(name, Arc::new(move || { counter.fetch_add(1, Ordering::SeqCst); Ok(()) }));
        }
        safe.register("dac", Arc::new(|| Err("board not answering".into())));
        assert_eq!(safe.make_safe(), vec!["dac: board not answering".to_owned()]);
        assert_eq!(count.load(Ordering::SeqCst), 2);
        safe.unregister("dac");
        assert_eq!(safe.outputs(), vec!["coil-x".to_owned(), "coil-y".to_owned()]);
        assert!(safe.make_safe().is_empty());
    }
}
