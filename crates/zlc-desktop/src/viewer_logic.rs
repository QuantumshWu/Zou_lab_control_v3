//! A Figure Viewer's logic nodes: pure processors over the data it holds, in the plugin
//! processes the board's nodes run in, by the same workbench rules. Each Viewer that adds one
//! has a workbench of its own (`Workbench::document`, no apparatus) and its own run map
//! (`Builtin`), so its nodes, runs and answers never meet the board's. Node ids are
//! `figure-<viewer>-<type>`, so their outputs are `@logic/figure-<viewer>-<type>/<output>`:
//! `archive::viewer_owned` keeps them from the board, and `Scope` from every other Viewer.
use std::path::Path;
use std::sync::Arc;

use serde_json::{Value, json};
use zlc_core::control::{NODE_INSTANCE, Origin};
use zlc_core::runtime::Runtime;
use zlc_plugin_host::{HostEvent, PluginHost};
use zlc_workbench::{Event, Host, PanelContent, SignalInfo, Workbench};

use crate::remote::RemoteCall;
use crate::workbench_host::{self, Builtin};

/// The catalogue commands a Viewer's page and its nodes answer.
pub const COMMANDS: &[&str] = &["add_node", "rename_node", "edit_node", "check_node", "revert_node", "start_node", "stop_node", "remove_node"];

/// The Viewer a node's plugin instance belongs to, if it is a Viewer's.
fn viewer_of_instance(instance: &str) -> Option<u32> {
    instance.strip_prefix(NODE_INSTANCE).and_then(crate::panels::viewer_of)
}

/// The Viewer whose node a host fact is of. A process lost with several instances is told one instance at a time (`split_failed`).
pub fn viewer_of_event(event: &HostEvent) -> Option<u32> {
    match event {
        HostEvent::RunReady { instance, .. } | HostEvent::RunProgress { instance, .. } | HostEvent::RunFinished { instance, .. } | HostEvent::Request { instance, .. } => viewer_of_instance(instance),
        HostEvent::Failed { instances, .. } => instances.first().and_then(|instance| viewer_of_instance(instance)),
        _ => None,
    }
}

/// A lost process's instances, when they are not all one document's: each told apart.
pub fn split_failed(event: &HostEvent) -> Option<Vec<HostEvent>> {
    let HostEvent::Failed { instances, reason } = event else { return None };
    let first = instances.first().map(|instance| viewer_of_instance(instance))?;
    if instances.iter().all(|instance| viewer_of_instance(instance) == first) { return None; }
    Some(instances.iter().map(|instance| HostEvent::Failed { instances: vec![instance.clone()], reason: reason.clone() }).collect())
}

/// What a Viewer's workbench sees of the application: the signals that Viewer offers, and no panels
/// (its nodes open none; the operator binds their outputs through the Viewer's own signal tree).
pub struct Scope<'a> { pub runtime: &'a Runtime, pub offers: &'a dyn Fn(&str) -> bool }

impl Host for Scope<'_> {
    fn plot_kind(&self, kind: &str) -> Option<zlc_workbench::PlotKindFacts> { workbench_host::plot_kind(kind) }
    fn signals(&self) -> Vec<SignalInfo> { workbench_host::offered_signals(self.runtime, self.offers) }
    fn panels(&self) -> Vec<PanelContent> { Vec::new() }
    fn reserve_panel_ids(&mut self, _: usize) -> Vec<String> { Vec::new() }
    fn add_panel(&mut self, _: &str, _: &str, _: &str, _: u64, _: Value) -> Result<String, String> { Err("A Figure Viewer's node opens no panel of its own".into()) }
    fn close_panel(&mut self, _: &str) {}
    fn schemas(&self, signal: &str) -> Option<(Value, Value)> { workbench_host::schemas(self.runtime, signal) }
}

pub struct Logic { workbench: Workbench, builtin: Builtin, refused: Vec<(u64, Result<Value, String>)> }

impl Logic {
    pub fn new(runtime: Runtime, host: PluginHost, root: &Path, reply: Arc<dyn Fn(Event) + Send + Sync>,
               core: Arc<dyn Fn(RemoteCall) -> Result<(), String> + Send + Sync>) -> Result<Self, String> {
        let builtin = Builtin::new(runtime, Some(host), reply, core);
        builtin.catalog_ready();
        let workbench = Workbench::document(root, Box::new(builtin.clone()))?;
        Ok(Self { workbench, builtin, refused: Vec::new() })
    }

    /// A processor of `type_id` on `viewer`: refused unless its declaration says it can run on saved data.
    fn added(&self, viewer: u32, params: &Value) -> Result<Value, String> {
        let type_id = params["type_id"].as_str().ok_or("type_id is required")?;
        let kind = self.workbench.catalog().get(type_id).filter(|item| item.pure_processor())
            .ok_or_else(|| format!("{type_id} needs more than a dataset: a Figure Viewer runs only processors with no devices, files or resources"))?;
        let number = (1..).find(|number| !self.workbench.has_node(&node_id(viewer, type_id, *number))).expect("a free id");
        let name = if number == 1 { kind.label().to_owned() } else { format!("{} {number}", kind.label()) };
        Ok(json!({"type_id": type_id, "node_id": node_id(viewer, type_id, number), "name": name, "source_signal": params["source_signal"],
                  "values": params.get("values").cloned().unwrap_or_else(|| json!({})), "auto_preview": false}))
    }

    /// One catalogue command from the Viewer's page; its answer is in `take_replies`, now or later.
    pub fn command(&mut self, viewer: u32, name: &str, params: &Value, origin: &Origin, request: u64, scope: &mut Scope) {
        let params = if name == "add_node" {
            match self.added(viewer, params) { Ok(params) => params, Err(error) => { self.refused.push((request, Err(error))); return; } }
        } else { params.clone() };
        workbench_host::run(&mut self.workbench, &self.builtin, name, &params, origin, request, scope);
        self.settle(scope);
    }

    fn settle(&mut self, scope: &mut Scope) {
        workbench_host::deliver(&mut self.workbench, &self.builtin, scope);
        if self.workbench.on_runtime(scope) { workbench_host::deliver(&mut self.workbench, &self.builtin, scope); }
    }

    /// What its plugin calls answered (workbench_host's threads); facts held for a start's answer follow it.
    pub fn answered(&mut self, event: Event, scope: &mut Scope) {
        let held = self.builtin.replay();
        for event in std::iter::once(event).chain(held) { self.workbench.on_event(event, scope); }
        self.settle(scope);
    }

    /// A plugin host fact of one of its nodes.
    pub fn host_event(&mut self, event: HostEvent, scope: &mut Scope) {
        for event in self.builtin.host_event(event) { self.workbench.on_event(event, scope); }
        self.settle(scope);
    }

    /// The Runtime's signals changed.
    pub fn runtime_changed(&mut self, scope: &mut Scope) { self.settle(scope); }

    pub fn take_replies(&mut self) -> Vec<(u64, Result<Value, String>)> {
        let mut replies = std::mem::take(&mut self.refused);
        replies.extend(self.workbench.take_replies());
        replies
    }

    pub fn rows(&self, scope: &Scope) -> Value { self.workbench.logic_rows(scope) }

    /// The Viewer or its Figure goes: every run stops, every node's instance closes and its signals are withdrawn.
    pub fn close(mut self, scope: &mut Scope) { self.workbench.close_nodes(scope); }
}

fn node_id(viewer: u32, type_id: &str, number: u32) -> String {
    let prefix = crate::panels::viewer_prefix(viewer);
    if number == 1 { format!("{prefix}-{type_id}") } else { format!("{prefix}-{type_id}{number}") }
}

#[cfg(test)]
#[path = "viewer_logic_path.rs"]
mod path;
