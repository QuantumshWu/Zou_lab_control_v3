//! The acquisition's device state, requested after configuration and before data.
//! A task and its ready acquisition share one observation. Missing device answers
//! are recorded as unavailable; recording never takes another device's owner.
use std::collections::BTreeSet;

use serde_json::{Map, Value, json};

use crate::apparatus::LiveState;
use crate::{Ticket, Workbench, workspace};

/// How long a device's `snapshot` may take: `Plugins::call` answers it within this (one
/// busy in another run's long call is recorded as not answering, never marked failed).
pub const SNAPSHOT_WAIT: std::time::Duration = std::time::Duration::from_secs(2);

/// The pulse board's capability: its entry is the bench's `sequencer`, with who drives it.
const PULSE_BOARD: &str = "sequencer.streamer";

/// A bench being taken: the answers still out, the entries so far (instance -> entry).
#[derive(Clone, Debug, Default)]
pub struct Bench {
    taken_at: String, asked: BTreeSet<u64>, devices: Map<String, Value>, sequencer: Map<String, Value>,
    completed_at: String,
    pub(crate) waiting: Vec<(String, u64)>,
    /// Separate from device answers, which may themselves contain transport fields.
    device_transports: Map<String, Value>,
}

impl Bench {
    pub fn complete(&self) -> bool { !self.taken_at.is_empty() && self.asked.is_empty() }

    pub(crate) fn record(&self, context: &str) -> Value {
        let mut record = json!({"phase": "acquisition", "context": context, "taken_at": self.taken_at,
            "completed_at": self.completed_at, "devices": self.devices, "sequencer": self.sequencer});
        if !self.device_transports.is_empty() { record["device_transports"] = json!(self.device_transports); }
        record
    }

    fn entry(&mut self, device: &str) -> Option<&mut Value> { self.devices.get_mut(device).or_else(|| self.sequencer.get_mut(device)) }
}

impl Workbench {
    /// A measurement or task records the bench; a processor's sources' records reach it through lineage.
    pub(crate) fn takes_bench(&self, type_id: &str) -> bool {
        self.catalog.get(type_id).is_some_and(|descriptor| !descriptor.is_device() && descriptor.node().node_kind != "processor")
    }

    /// Every live device entered as it stands (instance -> `{type_id, role}`), each that reports
    /// its settings asked; the board's entry also says whether a run or an outside driver holds it.
    fn take_bench(&mut self, run: &str) -> Bench {
        let mut bench = Bench { taken_at: workspace::utc_now(), ..Bench::default() };
        let ids: Vec<String> = self.devices.keys().cloned().collect();
        for id in ids {
            let live = &self.devices[&id];
            let kind = self.live_type(live);
            let board = kind.is_some_and(|item| item.capabilities.iter().any(|token| token == PULSE_BOARD));
            let mut entry = json!({"type_id": kind.map_or(live.entry.type_id.as_str(), |item| item.type_id.as_str()), "role": live.entry.role});
            if let Some(remote) = live.working_point.get("remote").filter(|remote| remote.is_object()) {
                bench.device_transports.insert(id.clone(), json!({"kind": "fabric", "endpoint": remote}));
            }
            let unavailable = match live.state {
                LiveState::Opening => Some("still opening".to_owned()),
                LiveState::Closing => Some("closing".to_owned()),
                LiveState::Failed => Some(format!("failed: {}", live.error.as_deref().unwrap_or("unknown error"))),
                LiveState::Open if live.published => Some("published on the bench fabric: another computer uses it".to_owned()),
                LiveState::Open if live.working_point["snapshot"] != true => Some(live.working_point["snapshot_unavailable"].as_str()
                    .unwrap_or("does not report its settings").to_owned()),
                LiveState::Open => None,
            };
            if board {
                let using = self.device_use(&id);
                entry["driver"] = match (using.runs.first(), using.outside) {
                    (Some(claim), _) => json!({"run": claim.run, "node": claim.node_name}),
                    (None, Some(outside)) => json!({"outside": outside}),
                    (None, None) => Value::Null,
                };
            }
            match unavailable {
                Some(why) => entry["unavailable"] = json!(why),
                None => {
                    let ticket = self.ticket(Ticket::Snapshot { run: run.to_owned(), device: id.clone() });
                    bench.asked.insert(ticket);
                    self.plugins.call(&id, ticket, "snapshot", &json!({}));
                }
            }
            if board { bench.sequencer.insert(id, entry); } else { bench.devices.insert(id, entry); }
        }
        bench
    }

    /// The first configured caller records the acquisition context. A ready child
    /// and the task that triggers it share that observation in their saved sources.
    pub(crate) fn run_snapshot(&mut self, run: &str, ticket: u64, wait: bool) {
        let Some(active) = self.run(run).filter(|run| !run.terminal() && run.state != "stopping") else {
            self.plugins.acquired(run, ticket, Err("the recording run has ended".into())); return;
        };
        let context = active.snapshot_owner.clone();
        if !self.node(&active.node).is_some_and(|node| self.takes_bench(&node.type_id)) {
            self.plugins.acquired(run, ticket, Ok(Value::Null)); return;
        }
        let Some(owner) = self.run(&context).filter(|run| !run.terminal() && run.state != "stopping") else {
            self.plugins.acquired(run, ticket, Err("the acquisition context has ended".into())); return;
        };
        if let Some(bench) = owner.bench.as_ref().filter(|bench| bench.complete()) {
            self.plugins.recorded(run, ticket, Ok(bench.record(&context))); return;
        }
        if owner.bench.is_none() {
            // A consumer with no capture of its own keeps its source records;
            // publication never creates a retrospective or all-empty observation.
            if wait { self.plugins.acquired(run, ticket, Ok(Value::Null)); return; }
            let bench = self.take_bench(&context);
            self.run_mut(&context).unwrap().bench = Some(bench);
        }
        self.run_mut(&context).unwrap().bench.as_mut().unwrap().waiting.push((run.to_owned(), ticket));
        self.finish_bench(&context);
    }

    fn finish_bench(&mut self, context: &str) {
        let Some(bench) = self.run_mut(context).and_then(|run| run.bench.as_mut()).filter(|bench| bench.complete()) else { return };
        if bench.completed_at.is_empty() { bench.completed_at = workspace::utc_now(); }
        let record = bench.record(context);
        let waiting = std::mem::take(&mut bench.waiting);
        for (run, ticket) in waiting {
            let result = if self.run(&run).is_some_and(|run| !run.terminal() && run.state != "stopping") {
                Ok(record.clone())
            } else { Err("the recording run has ended".into()) };
            self.plugins.recorded(&run, ticket, result);
        }
    }

    pub(crate) fn cancel_bench(&mut self, context: &str) {
        let mut waiting = Vec::new();
        for owner in &mut self.runs {
            if let Some(bench) = &mut owner.bench {
                for (run, ticket) in std::mem::take(&mut bench.waiting) {
                    if owner.id == context || run == context { waiting.push((run, ticket)); }
                    else { bench.waiting.push((run, ticket)); }
                }
            }
        }
        for (run, ticket) in waiting { self.plugins.acquired(&run, ticket, Err("the acquisition context ended before recording device state".into())); }
    }

    /// One device's answer preserves the apparatus identity and its own fields;
    /// the last answer releases the acquisition's waiting callers.
    pub(crate) fn bench_answered(&mut self, run: &str, device: &str, ticket: u64, result: Result<Value, String>) {
        let Some(bench) = self.run_mut(run).and_then(|run| run.bench.as_mut()) else { return };
        if !bench.asked.remove(&ticket) { return; }
        if let Some(entry) = bench.entry(device) {
            match result {
                Ok(Value::Object(answer)) => {
                    // Keep the apparatus identity authoritative, but never discard
                    // a device-owned field merely because it uses the same name.
                    let mut conflicts = Map::new();
                    for (key, value) in answer {
                        if entry.get(&key).is_none_or(Value::is_null) { entry[key] = value; }
                        else if entry[&key] != value { conflicts.insert(key, value); }
                    }
                    if !conflicts.is_empty() {
                        if let Some(reported) = entry.as_object_mut().and_then(|entry| entry.remove("snapshot_fields")) {
                            conflicts.insert("snapshot_fields".into(), reported);
                        }
                        entry["snapshot_fields"] = Value::Object(conflicts);
                    }
                },
                Ok(Value::Null) => entry["unavailable"] = json!("does not report its settings"),
                Ok(other) => entry["settings"] = other,
                Err(error) => entry["unavailable"] = json!(error),
            }
        }
        self.finish_bench(run);
    }
}
