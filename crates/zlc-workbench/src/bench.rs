//! The bench at a Start (PLUGIN_PROTOCOL §5 `snapshot`): a measurement or task run
//! records every device of the apparatus and the pulse board as the core's fact,
//! whatever its node chose to write in its own record (v2 kept only that). Asked of
//! each open device that reports its settings once the start may launch, all at once;
//! the run launches when every answer is in. A device that cannot answer (busy past
//! the bound, failed, or handed to another computer) is entered as
//! unavailable with why: it never holds or fails the Start. State at Start only.
use std::collections::BTreeSet;

use serde_json::{Map, Value, json};

use crate::apparatus::LiveState;
use crate::logic::Claim;
use crate::{Host, Ticket, Workbench, workspace};

/// How long a device's `snapshot` may take: `Plugins::call` answers it within this (one
/// busy in another run's long call is recorded as not answering, never marked failed).
pub const SNAPSHOT_WAIT: std::time::Duration = std::time::Duration::from_secs(2);

/// The pulse board's capability: its entry is the bench's `sequencer`, with who drives it.
const PULSE_BOARD: &str = "sequencer.streamer";

/// A bench being taken: the answers still out, the entries so far (instance -> entry).
#[derive(Clone, Debug, Default)]
pub struct Bench {
    taken_at: String, asked: BTreeSet<u64>, devices: Map<String, Value>, sequencer: Map<String, Value>,
    /// Separate from device answers, which may themselves contain transport fields.
    device_transports: Map<String, Value>,
}

impl Bench {
    pub fn complete(&self) -> bool { self.asked.is_empty() }

    /// The record's `apparatus`; a board nobody drove at Start is driven by the run that claims it.
    pub(crate) fn record(&self, run: &str, node: &str, claims: &[Claim]) -> Value {
        let mut sequencer = self.sequencer.clone();
        for (device, entry) in &mut sequencer {
            if entry["driver"].is_null() && claims.iter().any(|claim| &claim.device == device) { entry["driver"] = json!({"run": run, "node": node}); }
        }
        let mut record = json!({"taken_at": self.taken_at, "devices": self.devices, "sequencer": sequencer});
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
    pub(crate) fn take_bench(&mut self, node: &str) -> Bench {
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
                    let ticket = self.ticket(Ticket::Snapshot { node: node.to_owned(), device: id.clone() });
                    bench.asked.insert(ticket);
                    self.plugins.call(&id, ticket, "snapshot", &json!({}));
                }
            }
            if board { bench.sequencer.insert(id, entry); } else { bench.devices.insert(id, entry); }
        }
        bench
    }

    /// One device's answer for the bench of `node`'s pending start: its settings beside its
    /// identity (which they never replace), or why it gave none. The last answer launches the start.
    pub(crate) fn bench_answered(&mut self, node: &str, device: &str, ticket: u64, result: Result<Value, String>, host: &mut dyn Host) {
        let Some(bench) = self.nodes.iter_mut().find(|item| item.id == node).and_then(|item| item.pending.as_mut())
            .and_then(|pending| pending.bench.as_mut()) else { return };
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
        if bench.complete() { self.launch_pending(host); }
    }
}
