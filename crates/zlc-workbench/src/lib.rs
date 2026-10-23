//! The application composition: workspace, apparatus and device instances,
//! logic rows and their runs, the board and its layout files
//! (docs/WORKBENCH_COMMANDS.md). One owner turn applies every command and
//! every plugin event; plugin work is asked of [`Plugins`] and completes as an
//! [`Event`]. Scientific data never passes through here.
pub mod fields;
pub mod types;
pub mod workspace;
mod apparatus;
mod choices;
mod layout;
mod logic;

use std::collections::{BTreeMap, VecDeque};
use std::path::{Path, PathBuf};

use serde_json::{Map, Value, json};
use zlc_core::control::Origin;

pub use choices::viewer_signal_tree;
pub use layout::INTERVALS;
pub use types::{TypeCatalog, TypeDescriptor};
pub use workspace::Workspace;

/// The plugin host as the workbench drives it (PLUGIN_PROTOCOL; line A).
/// Every call returns at once; completion arrives as an [`Event`].
pub trait Plugins {
    /// The describe result: device and node types, and modules that failed.
    fn catalog(&mut self) -> TypeCatalog;
    /// Open a device instance (`context.simulation` for shared-world types) or a node instance.
    fn open(&mut self, instance: &str, type_id: &str, parameters: &Value, context: &Value);
    fn close(&mut self, instance: &str);
    /// A pure draft query answered by the node's own process (`declare_outputs`, `admit_source`, ...).
    fn query(&mut self, instance: &str, ticket: u64, name: &str, args: &Value);
    /// A standard device op (`fields`, `values`, `tune`) on an open instance.
    fn call(&mut self, instance: &str, ticket: u64, op: &str, params: &Value);
    fn start(&mut self, run: &RunSpec);
    fn rename(&mut self, _instance: &str, _name: &str) {}
    /// A processor follows its source (None: no longer): the run started on each new
    /// generation of it reads that generation from its first commit.
    fn follow(&mut self, _instance: &str, _signal: Option<&str>) {}
    fn stop(&mut self, run: &str);
    /// The operator's answer to a run's request; `None` cancels it (the run stops).
    fn answer(&mut self, run: &str, request: &str, response: Option<&Value>);
    /// The reply to a run's `RunAcquire`: at the child's ready or terminal.
    fn acquired(&mut self, run: &str, ticket: u64, result: Result<Value, String>);
    /// A device type's hardware scan, no instance opened: `{found: [{instance_id, role, parameters}], notes}`.
    fn discover(&mut self, ticket: u64, type_id: &str);
    /// Device types whose scan missed its deadline and is still answering (v2 `_scan_pending`).
    fn scanning(&self) -> Vec<String> { Vec::new() }
    /// The plugin types are still being discovered (the catalogue is not in yet).
    fn discovering(&self) -> bool { false }
    /// Offer an open device to another bench computer (Remote on) or take it back.
    fn publish(&mut self, ticket: u64, instance: &str, role: &str, published: bool);
}

/// Facts the plugin host reports.
#[derive(Clone, Debug)]
pub enum Event {
    /// Lifecycle work is ordered by its host, but its reply can already be in
    /// the UI queue when a newer Open/Close supersedes it. Check at consumption.
    Lifecycle { current: std::sync::Arc<std::sync::atomic::AtomicU64>, revision: u64, event: Box<Event> },
    Opened { instance: String, working_point: Value },
    Failed { instance: String, error: String },
    Closed { instance: String, error: Option<String> },
    Answered { ticket: u64, result: Result<Value, String> },
    Log { instance: String, line: String },
    RunStarted { run: String },
    RunReady { run: String },
    RunProgress { run: String, message: String, current: Option<u64>, total: Option<u64> },
    RunRequest { run: String, request: Value },
    RunArtifact { run: String, artifact: Value },
    RunAcquire { run: String, ticket: u64, node: String, completion: String },
    /// `outcome`: completed, stopped or failed.
    RunFinished { run: String, outcome: String, error: Option<Value>, generation: Option<String> },
}

/// What a node run is started with.
#[derive(Clone, Debug)]
pub struct RunSpec {
    pub prepared: Option<String>,
    pub node_name: String,
    pub run: String,
    pub node_id: String,
    /// The node's plugin instance (`zlc_core::control::node_instance`).
    pub instance: String,
    pub type_id: String,
    /// Enabled authoring values only.
    pub values: Map<String, Value>,
    pub source_signal: String,
    /// Requirement argument -> device instance id.
    pub devices: BTreeMap<String, String>,
    pub device_labels: BTreeMap<String, String>,
    /// The arguments whose device the run shares (a scan's device axis: its fields protected, not the device).
    pub shared: std::collections::BTreeSet<String>,
    pub artifacts: Map<String, Value>,
    /// Workspace resources by build argument, resolved (a pulse: its description).
    pub resources: Map<String, Value>,
    /// Task run directory.
    pub directory: Option<PathBuf>,
    /// Output name -> the signal the run publishes it as.
    pub outputs: BTreeMap<String, String>,
    pub origin: String,
}

/// A Runtime signal as the board sees it.
#[derive(Clone, Debug)]
/// `shape`: the axis sizes of its three domains, as v2 titles show them (`(20) × (3 × 10) × (35)`).
/// `owner`: its generation's owner, the one atomic producer that publishes it (a panel's ROI and fit are two).
pub struct SignalInfo { pub name: String, pub generation: String, pub owner: String, pub published: bool, pub sealed: bool, pub shape: String }

/// One panel's content, from the application's accepted state.
#[derive(Clone, Debug)]
/// `outputs`: what it can publish, (name, label): its region's catalogue and its fit's parameters.
pub struct PanelContent { pub panel: String, pub signal: String, pub kind: String, pub window: u64, pub target: Value, pub outputs: Vec<(String, String)> }

/// What the application provides: signals and the panels it shows.
pub trait Host {
    fn signals(&self) -> Vec<SignalInfo>;
    fn panels(&self) -> Vec<PanelContent>;
    /// Panel ids the application will use for the next panels it adds.
    fn reserve_panel_ids(&mut self, count: usize) -> Vec<String>;
    /// Show a panel under a reserved (or freshly minted, when `panel` is empty) id; returns the id.
    fn add_panel(&mut self, panel: &str, signal: &str, kind: &str, window: u64, target: Value) -> Result<String, String>;
    fn close_panel(&mut self, panel: &str);
    /// A signal's dataset schema as `zlc_data.codec` trees: its run's, and its latest
    /// event's (null before one); None while the Runtime has no such signal.
    fn schemas(&self, _signal: &str) -> Option<(Value, Value)> { None }
}

/// What an answered ticket was for.
#[derive(Clone, Debug)]
pub(crate) enum Ticket {
    Prepare { node: String, frozen: logic::Frozen, origin: String, acquisition: Option<(String, u64, String)> },
    Discard,
    Query { node: String, revision: u64, name: String },
    Estimate { request: u64 },
    Fields { device: String, request: Option<u64> },
    Tune { device: String, request: u64 },
    Command { request: u64 },
    Discover { type_id: String },
    Publish { device: String, published: bool, request: u64 },
    /// A panel's region asked its producer what it sets (`selection_patch`).
    Selection { node: String, panel: String, selection: Value },
}

pub struct Workbench {
    pub(crate) plugins: Box<dyn Plugins>,
    pub(crate) catalog: TypeCatalog,
    pub(crate) workspace: Workspace,
    pub(crate) apparatus: apparatus::Apparatus,
    pub(crate) devices: BTreeMap<String, apparatus::Live>,
    pub(crate) nodes: Vec<logic::Node>,
    pub(crate) runs: Vec<logic::Run>,
    /// Run identities outlive node removal and board replacement.
    pub(crate) run_sequences: BTreeMap<String, u64>,
    pub(crate) board: layout::Board,
    pub(crate) notes: Vec<String>,
    /// The Device Manager's one line of what just happened (v2 DeviceManagerPresenter._report):
    /// text and its status word.
    pub(crate) report: (String, &'static str),
    /// An Init, Apply, Close or shut down whose devices are still opening or
    /// closing: its line is said when the last one settles.
    pub(crate) settling: Option<apparatus::Settling>,
    /// When the Apply or Close under way began (its wait for the runs it stops is bounded).
    pub(crate) settling_since: Option<std::time::Instant>,
    /// The Task Console's one line of what just happened (v2 ConsolePresenter._report):
    /// "selectors disabled", "added <node>", "<node> started", "board cleared", ...
    /// Said when it happens, never re-derived per state push.
    pub(crate) console_report: (String, &'static str),
    /// What the line last said of the notes and of the running Task (v2 `_shown_task_status`):
    /// each is said again only when it changes.
    pub(crate) notes_said: Vec<String>,
    pub(crate) task_said: Option<(String, String)>,
    pub(crate) tickets: BTreeMap<u64, Ticket>,
    pub(crate) next_ticket: u64,
    pub(crate) replies: Vec<(u64, Result<Value, String>)>,
    pub(crate) logs: BTreeMap<String, VecDeque<String>>,
    /// The simulation settings the live simulated instances were opened with.
    pub(crate) applied_simulation: Option<Map<String, Value>>,
    /// A hardware scan in flight.
    pub(crate) scan: Option<apparatus::Scan>,
    /// Each node's source bundle as its queries read it (`bundle`): node -> (what it was
    /// built from, the members).
    pub(crate) bundles: BTreeMap<String, (String, Value)>,
    /// v2 `selection_restore`: (node, draft field) -> the panel whose region set it, and
    /// what it held before.
    pub(crate) selection_restore: BTreeMap<(String, String), (String, Value)>,
    /// The region each panel last told its producer (a new read of the same region tells nothing).
    pub(crate) routed_selections: BTreeMap<String, Value>,
    /// Devices driven from outside the workbench, by whom (the Pulse Editor playing on the session's sequencer).
    pub(crate) device_users: DeviceUsers,
}

/// v2 DeviceUseCoordinator's other owners: each device driven from outside the workbench, by
/// whom. Its driver writes it as it takes and leaves the device; a start that claims one is refused.
pub type DeviceUsers = std::sync::Arc<std::sync::Mutex<BTreeMap<String, String>>>;

/// Every command the workbench answers (the catalogue's workbench part).
pub const COMMANDS: &[&str] = &[
    "list_plugins", "get_device_log", "estimate_node", "list_runs", "get_run", "read_layout", "list_recipes",
    "add_device", "remove_device", "edit_device", "edit_simulation", "use_template", "discover_devices", "add_discovered",
    "load_apparatus", "save_apparatus", "apply_devices", "close_device", "shut_down_devices",
    "read_device", "tune_device", "accept_device_risk", "publish_device",
    "add_node", "rename_node", "edit_node", "check_node", "revert_node", "start_node", "stop_node", "remove_node", "answer_request",
    "order_panels", "rename_panel", "set_panel_interval", "set_panel_overlay", "publish_panel_outputs", "set_board",
    "save_layout", "load_layout", "clear_board", "run_recipe",
];

/// Commands that leave the result to a later plugin event.
const DEFERRED: &[&str] = &["estimate_node", "read_device", "tune_device", "discover_devices", "publish_device"];

pub(crate) fn text<'a>(params: &'a Value, name: &str) -> Result<&'a str, String> {
    params[name].as_str().ok_or_else(|| format!("{name} is required"))
}

impl Workbench {
    pub fn new(root: &Path, mut plugins: Box<dyn Plugins>) -> Result<Self, String> {
        let workspace = Workspace::new(root)?;
        let catalog = plugins.catalog();
        let mut workbench = Self { plugins, catalog, apparatus: apparatus::Apparatus::empty(workspace.apparatus()), workspace,
            devices: BTreeMap::new(), nodes: Vec::new(), runs: Vec::new(), run_sequences: BTreeMap::new(), board: layout::Board::default(), notes: Vec::new(),
            report: (String::new(), "idle"), settling: None, settling_since: None,
            // v2 set_deriving(False) when the console is built.
            console_report: ("selectors disabled".into(), "task"), notes_said: Vec::new(), task_said: None,
            tickets: BTreeMap::new(), next_ticket: 1, replies: Vec::new(), logs: BTreeMap::new(), applied_simulation: None, scan: None, bundles: BTreeMap::new(),
            selection_restore: BTreeMap::new(), routed_selections: BTreeMap::new(), device_users: DeviceUsers::default() };
        let path = workbench.workspace.apparatus();
        if path.is_file() {
            if let Err(error) = workbench.load_apparatus(&path) { workbench.notes.push(error); }
        } else {
            // v2 DeviceManagerPresenter.load on a workspace without one.
            workbench.report = (format!("no apparatus at {} yet; add devices and save", apparatus::file_name(&path)), "task");
        }
        workbench.say_new_notes();
        Ok(workbench)
    }

    pub fn handles(name: &str) -> bool { COMMANDS.contains(&name) }

    /// The plugin host finished discovery: read its types, and the apparatus
    /// file again if nothing has been applied or edited since it was read;
    /// else the draft is checked against them. An apparatus read before
    /// discovery answered says nothing about which types are installed.
    pub fn refresh_catalog(&mut self) {
        self.catalog = self.plugins.catalog();
        let path = self.apparatus.path.clone();
        if self.devices.is_empty() && !self.apparatus.edited() && path.is_file() {
            self.notes.clear();
            if let Err(error) = self.load_apparatus(&path) { self.notes.push(error); }
        } else {
            for note in self.uninstalled_draft() { if !self.notes.contains(&note) { self.notes.push(note); } }
        }
        self.say_new_notes();
    }
    pub fn workspace(&self) -> &Workspace { &self.workspace }
    /// The file dialog a window command without a path asks (v2 ask_open_path / ask_save_path):
    /// its caption, its v2 filter, whether it saves, where it starts -- a folder, or the file a
    /// save proposes -- and its kind's own folder ([`workspace::dialog_start`]). Devices@ starts at
    /// the apparatus being edited (v2), the Task Console at its open layout, else in the
    /// workspace's layouts (v2's day folder was shared with screenshots and runs, and a new one each day).
    pub fn file_dialog(&self, action: &str) -> Option<(&'static str, &'static str, bool, PathBuf, PathBuf)> {
        const APPARATUS: &str = "Apparatus (*.json);;All files (*)";
        const LAYOUTS: &str = "Layouts (*.json)";
        let folder = |path: &Path| path.parent().map(Path::to_path_buf).unwrap_or_default();
        let (root, layouts) = (self.workspace.root.clone(), self.workspace.layouts());
        Some(match action {
            "load_apparatus" => ("Open apparatus", APPARATUS, false, folder(&self.apparatus.path), root),
            "save_apparatus" => ("Save apparatus as", APPARATUS, true, self.apparatus.path.clone(), root),
            "load_layout" => ("Open TaskConsole layout", LAYOUTS, false, self.board.path.as_deref().map_or_else(|| layouts.clone(), folder), layouts),
            "save_layout" => ("Save TaskConsole layout", LAYOUTS, true, self.board.path.clone().unwrap_or_else(|| layouts.join("layout.json")), layouts),
            _ => return None,
        })
    }
    /// Workspace resources of `contract_prefix...` resolve through `resolver` (the application knows the formats).
    pub fn add_resource_resolver(&mut self, contract_prefix: &str, resolver: workspace::Resolver) { self.workspace.add_resolver(contract_prefix, resolver); }

    pub(crate) fn ticket(&mut self, ticket: Ticket) -> u64 {
        let id = self.next_ticket;
        self.next_ticket += 1;
        self.tickets.insert(id, ticket);
        id
    }

    /// Apply one validated command. Its reply is queued for
    /// [`Workbench::take_replies`] under `request`, now or when the plugin answers.
    pub fn apply(&mut self, name: &str, params: &Value, origin: &Origin, request: u64, host: &mut dyn Host) {
        if self.handles_command(name) {
            if let Err(error) = self.run_device_command(name, params, request) { self.replies.push((request, Err(error))); }
            return;
        }
        let result = self.dispatch(name, params, origin, request, host);
        match result {
            Ok(None) if DEFERRED.contains(&name) => {}
            Ok(reply) => self.replies.push((request, Ok(reply.unwrap_or_else(|| json!({}))))),
            Err(error) => self.replies.push((request, Err(error))),
        }
        self.sync_console_report(host);
    }

    /// Replies ready to send: (request, result).
    pub fn take_replies(&mut self) -> Vec<(u64, Result<Value, String>)> { std::mem::take(&mut self.replies) }

    fn dispatch(&mut self, name: &str, params: &Value, origin: &Origin, request: u64, host: &mut dyn Host) -> Result<Option<Value>, String> {
        // What a successful console command says on the status strip, worded as v2's presenter did
        // (a start says its line when the run starts, a remove when the node is gone: logic.rs; a
        // board replaced when it is: layout.rs).
        let node_name = |workbench: &Self| params["node"].as_str().and_then(|id| workbench.node(id)).map(|node| node.name.clone());
        let renamed = (name == "rename_node").then(|| node_name(self)).flatten();
        let result = self.dispatch_command(name, params, origin, request, host);
        if let Ok(answer) = &result {
            let line = match name {
                "add_node" => answer.as_ref().and_then(|answer| answer["node"].as_str()).and_then(|id| self.node(id)).map(|node| (format!("added {}", node.name), "task")),
                "rename_node" => renamed.zip(node_name(self)).filter(|(before, after)| before != after).map(|(before, after)| (format!("renamed {before} to {after}"), "task")),
                "set_board" => params["selectors"].as_bool().map(|on| ((if on { "selectors enabled" } else { "selectors disabled" }).to_owned(), "task")),
                "save_layout" => params["path"].as_str().map(|path| (format!("layout saved to {}", self.workspace.resolve(path).display()), "task")),
                _ => None,
            };
            if let Some(line) = line { self.console_report = line; }
        }
        result
    }

    /// The desktop's own console commands (a blank panel added) say their line here too.
    pub fn console_note(&mut self, text: String) { self.console_report = (text, "task"); }
    pub fn console_warning(&mut self, text: String) { self.console_report = (text, "warning"); }

    fn dispatch_command(&mut self, name: &str, params: &Value, origin: &Origin, request: u64, host: &mut dyn Host) -> Result<Option<Value>, String> {
        let origin_text = match origin { Origin::Operator => "operator".to_owned(), Origin::Remote(client) => client.clone() };
        match name {
            "list_plugins" => Ok(Some(self.list_plugins(params["type_id"].as_str()))),
            "get_device_log" => {
                let device = text(params, "device")?;
                let lines = params["lines"].as_u64().unwrap_or(200).min(2000) as usize;
                let mut log = self.logs.get(device).map(|log| log.iter().skip(log.len().saturating_sub(lines)).cloned().collect::<Vec<_>>()).unwrap_or_default();
                // v2's words for a device whose code tells its log nothing.
                if self.devices.get(device).is_some_and(|live| live.working_point["narrates"] == false) {
                    log.insert(0, "(this device type does not narrate its own interactions yet; fabric lines naming it appear once published)".into());
                }
                Ok(Some(json!({"device": device, "lines": log})))
            }
            "estimate_node" => self.estimate_node(text(params, "node")?, request).map(|()| None),
            "list_runs" => Ok(Some(self.list_runs(params["node"].as_str(), params["limit"].as_u64().unwrap_or(50).min(200) as usize))),
            "get_run" => self.get_run(text(params, "run")?).map(Some),
            "read_layout" => self.read_layout(text(params, "path")?, host).map(Some),
            "list_recipes" => self.list_recipes(params["name"].as_str()).map(Some),
            "add_device" => self.add_device(params).map(Some),
            "remove_device" => self.remove_device(text(params, "device")?).map(Some),
            "edit_device" => self.edit_device(params).map(Some),
            "edit_simulation" => self.edit_simulation(&params["values"]).map(Some),
            "use_template" => self.use_template(text(params, "template")?).map(Some),
            "discover_devices" => self.discover_devices(request),
            "add_discovered" => self.add_discovered(text(params, "instance_id")?).map(Some),
            "load_apparatus" => { let path = self.workspace.resolve(text(params, "path")?); self.load_apparatus(&path).map(Some) }
            "save_apparatus" => {
                let path = self.workspace.resolve(text(params, "path")?);
                self.save_apparatus(&path, params["replace_unread"].as_bool().unwrap_or(false)).map(Some)
            }
            "apply_devices" => self.apply_devices(host).map(Some),
            "close_device" => self.close_device(text(params, "device")?, host).map(Some),
            "shut_down_devices" => self.shut_down_devices(host).map(Some),
            "read_device" => self.read_device(text(params, "device")?, params["units"].as_object(), Some(request)).map(|()| None),
            "tune_device" => self.tune_device(params, request).map(|()| None),
            "accept_device_risk" => self.accept_device_risk(text(params, "device")?, params["accepted"].as_bool().unwrap_or(false)).map(Some),
            "publish_device" => self.publish_device(text(params, "device")?, params["published"].as_bool().ok_or("published is required")?, request).map(|()| None),
            "add_node" => self.add_node(params, host).map(Some),
            "rename_node" => self.rename_node(text(params, "node")?, text(params, "name")?, &host.signals()).map(Some),
            "edit_node" => self.edit_node(params, host).map(Some),
            "check_node" => self.check_node(text(params, "node")?, host).map(Some),
            "revert_node" => self.revert_node(text(params, "node")?, host).map(Some),
            "start_node" => self.start_node(text(params, "node")?, params["expected_draft_revision"].as_u64(), &origin_text, None, host).map(Some),
            "stop_node" => self.stop_node(text(params, "node")?, host).map(Some),
            "remove_node" => self.remove_node(text(params, "node")?, host).map(Some),
            "answer_request" => self.answer_request(text(params, "node")?, text(params, "request")?, &params["response"], matches!(origin, Origin::Operator)).map(Some),
            "order_panels" => self.order_panels(params["order"].as_array().ok_or("order is required")?, host).map(Some),
            "rename_panel" => self.rename_panel(text(params, "panel")?, params["title"].as_str()).map(Some),
            "set_panel_interval" => self.set_panel_interval(text(params, "panel")?, params["interval_ms"].as_u64().ok_or("interval_ms is required")?).map(Some),
            "set_panel_overlay" => self.set_panel_overlay(text(params, "panel")?, params["signal"].as_str().unwrap_or(""), host).map(Some),
            "publish_panel_outputs" => self.publish_panel_outputs(text(params, "panel")?, &params["outputs"]).map(Some),
            "set_board" => {
                if let Some(paused) = params["paused"].as_bool() { self.board.paused = paused; }
                if let Some(selectors) = params["selectors"].as_bool() { self.board.selectors = selectors; }
                Ok(Some(json!({"paused": self.board.paused, "selectors": self.board.selectors})))
            }
            "save_layout" => { let path = self.workspace.resolve(text(params, "path")?); self.save_layout(&path, host).map(Some) }
            "load_layout" => {
                let path = self.workspace.resolve(text(params, "path")?);
                self.load_layout(&path, params["stop_running"].as_bool().unwrap_or(false), host).map(Some)
            }
            "clear_board" => self.clear_board(params["stop_running"].as_bool().unwrap_or(false), host).map(Some),
            "run_recipe" => self.run_recipe(text(params, "name")?, &params["parameters"], params["start"].as_bool().unwrap_or(true), &origin_text, host).map(Some),
            _ => Err(format!("{name} is not a workbench command")),
        }
    }

    /// One plugin fact, applied on the owner turn.
    pub fn on_event(&mut self, event: Event, host: &mut dyn Host) {
        match event {
            Event::Opened { instance, working_point } => self.instance_opened(&instance, working_point, host),
            Event::Failed { instance, error } => self.instance_failed(&instance, error, host),
            Event::Closed { instance, error } => self.instance_closed(&instance, error, host),
            Event::Answered { ticket, result } => self.answered(ticket, result, host),
            Event::Lifecycle { current, revision, event } => {
                if current.load(std::sync::atomic::Ordering::Acquire) == revision { self.on_event(*event, host); }
                return;
            }
            Event::Log { instance, line } => {
                let log = self.logs.entry(instance).or_default();
                if log.len() >= 2000 { log.pop_front(); }
                log.push_back(line);
            }
            Event::RunStarted { run } => self.run_started(&run),
            Event::RunReady { run } => self.run_ready(&run),
            Event::RunProgress { run, message, current, total } => {
                if let Some(record) = self.run_mut(&run) { record.progress = Some(json!({"message": message, "current": current, "total": total})); }
            }
            Event::RunRequest { run, request } => self.run_requested(&run, request),
            Event::RunArtifact { run, artifact } => { if let Some(record) = self.run_mut(&run) { record.artifacts.push(artifact); } }
            Event::RunAcquire { run, ticket, node, completion } => self.run_acquire(&run, ticket, &node, &completion, host),
            Event::RunFinished { run, outcome, error, generation } => self.run_finished(&run, &outcome, error, generation, host),
        }
        self.poll_recipe(host);
        self.sync_console_report(host);
    }

    fn answered(&mut self, ticket: u64, result: Result<Value, String>, host: &mut dyn Host) {
        let Some(purpose) = self.tickets.remove(&ticket) else { return };
        match purpose {
            Ticket::Prepare { node, frozen, origin, acquisition } => self.prepared(&node, ticket, frozen, origin, acquisition, result, host),
            Ticket::Discard => {},
            Ticket::Query { node, revision, name } => self.query_answered(&node, revision, &name, result, host),
            Ticket::Estimate { request } => self.replies.push((request, result.or_else(|error| Ok(json!({"status": "unknown", "reason": error}))))),
            Ticket::Fields { device, request } => {
                if self.devices.get(&device).is_none_or(|live| live.fields_ticket != Some(ticket)) {
                    if let Some(request) = request { self.replies.push((request, Err("This device reading was superseded; read the current device again.".into()))); }
                    return;
                }
                if let (Some(live), Ok(fields)) = (self.devices.get_mut(&device), &result) { live.fields = Some(fields.clone()); }
                // The reply is the rows the state shows (TunableField), not the plugin's raw read.
                let rows = self.devices.get(&device).and_then(|live| live.fields.as_ref().map(|read| self.tunable_rows(&device, live, read)));
                if let Some(request) = request { self.replies.push((request, result.map(|_| json!({"device": device, "fields": rows})))); }
            }
            Ticket::Command { request } => self.replies.push((request, result)),
            Ticket::Discover { type_id } => self.discovered(&type_id, result),
            Ticket::Publish { device, published, request } => self.published(&device, published, request, result),
            Ticket::Selection { node, panel, selection } => self.selection_patched(&node, &panel, &selection, result),
            Ticket::Tune { device, request } => {
                let reply = result.clone().map(|readback| json!({"device": device, "readback": readback}));
                self.replies.push((request, reply));
                if result.is_ok() { let _ = self.read_device(&device, None, None); }
            }
        }
    }

    /// The Runtime's signal set changed: following processors and previews act
    /// on it. Whether that started a run or opened a panel.
    pub fn on_runtime(&mut self, host: &mut dyn Host) -> bool {
        let before = (self.runs.len(), self.board.panels.len());
        self.refresh_bundles(host);
        self.follow_sources(host);
        self.open_previews(host);
        self.sync_console_report(host);
        before != (self.runs.len(), self.board.panels.len())
    }

    /// The first apparatus device of a type (draft order).
    pub fn instance_of(&self, type_id: &str) -> Option<String> {
        self.apparatus.draft.iter().find(|entry| entry.type_id == type_id).map(|entry| entry.instance_id.clone())
    }

    /// What a device is called (v2 session.device_labels): its role, else its instance id.
    pub fn device_label(&self, instance: &str) -> String {
        self.devices.get(instance).map(|live| live.entry.role.clone()).filter(|role| !role.is_empty()).unwrap_or_else(|| instance.to_owned())
    }

    /// The first logic node of a type (board order).
    pub fn node_of(&self, type_id: &str) -> Option<String> {
        self.nodes.iter().find(|node| node.type_id == type_id).map(|node| node.id.clone())
    }

    /// `stop_all`: every run is stopped (the desktop then drives the outputs safe).
    pub fn stop_all(&mut self, host: &mut dyn Host) {
        let nodes: Vec<String> = self.nodes.iter().map(|node| node.id.clone()).collect();
        for node in nodes { let _ = self.stop_node(&node, host); }
    }

    /// The sections every state message carries (WORKBENCH_COMMANDS.md §3).
    pub fn state(&self, host: &dyn Host) -> Map<String, Value> {
        let signals = host.signals();
        let mut sections = Map::new();
        sections.insert("workspace".into(), json!({
            "root": self.workspace.root, "data": self.workspace.data(), "today": self.workspace.today(), "pulses": self.workspace.pulses(), "layouts": self.workspace.layouts(),
            "layout": {"path": self.board.path, "dirty": self.board.dirty}, "notes": self.notes}));
        sections.insert("apparatus".into(), self.apparatus_state());
        sections.insert("node_types".into(), Value::Array(self.node_types()));
        sections.insert("devices".into(), Value::Array(self.device_rows()));
        sections.insert("logic".into(), Value::Array(self.nodes.iter().map(|node| self.node_row(node, &signals)).collect()));
        sections.insert("board".into(), self.board_state(&host.panels(), &signals));
        sections.insert("runs".into(), Value::Array(self.runs.iter().rev().take(50).map(logic::Run::row).collect()));
        sections
    }

    /// Put the sections into a state message, ordering `panels` as the board does.
    pub fn decorate(&self, state: &mut Value, host: &dyn Host) {
        if let Some(object) = state.as_object_mut() { object.extend(self.state(host)); }
        if let Some(rows) = state["panels"].as_array_mut() {
            let order = |row: &Value| row["id"].as_str().and_then(|id| self.board.position(id)).unwrap_or(usize::MAX);
            rows.sort_by_key(order);
            for row in rows.iter_mut() {
                if let Some(entry) = row["id"].as_str().and_then(|id| self.board.get(id)) {
                    if let Some(title) = &entry.title { row["title"] = json!(title); }
                }
            }
        }
    }

    /// Why a panel's science may not change now (a running Task shows it), if it may not.
    pub fn panel_frozen(&self, panel: &str, signal: &str) -> Option<String> {
        let task = self.active_task()?;
        self.task_protected_signals(task).contains(signal).then(|| {
            let _ = panel;
            format!("{} is running: this panel's source and fates are fixed until it stops", task.name)
        })
    }

    /// v2 Pause: the board's pictures hold still (derivations go on); Resume shows the newest.
    pub fn paused(&self) -> bool { self.board.paused }

    /// Which ROI/fit outputs each panel publishes: panel -> {output: false} (absent = published).
    pub fn published_outputs(&self) -> Map<String, Value> {
        self.board.panels.iter().filter(|entry| !entry.published_outputs.is_empty())
            .map(|entry| (entry.panel.clone(), Value::Object(entry.published_outputs.clone()))).collect()
    }

    /// Each board panel's update interval (ms).
    pub fn panel_intervals(&self) -> BTreeMap<String, u64> {
        self.board.panels.iter().map(|entry| (entry.panel.clone(), entry.interval_ms)).collect()
    }

    /// Each panel's Overlay: the judgement signal its image is drawn with.
    pub fn panel_overlays(&self) -> BTreeMap<String, String> {
        self.board.panels.iter().filter(|entry| !entry.overlay_signal.is_empty())
            .map(|entry| (entry.panel.clone(), entry.overlay_signal.clone())).collect()
    }

    /// The application added or closed a panel by its own command.
    pub fn panel_added(&mut self, panel: &str) { self.board.add(panel); }
    /// A closed panel's region gives back what it set (v2 `_remove_panel_now`).
    pub fn panel_closed(&mut self, panel: &str) {
        self.board.remove(panel);
        self.routed_selections.remove(panel);
        self.restore_producer_draft(panel);
    }

    /// The device instances an active run holds (a `core` command checks its asking run against them).
    pub fn run_claims(&self, run: &str) -> Vec<String> {
        self.run(run).filter(|record| !record.terminal()).map(|record| record.claims.iter().map(|claim| claim.device.clone()).collect()).unwrap_or_default()
    }

    /// The active run that holds a device instance, if one does (the pulse
    /// editor may not drive a sequencer a run holds).
    pub fn device_claimant(&self, device: &str) -> Option<String> { self.claims_on(device).into_iter().next().map(|claim| claim.node_name) }

    /// The record an outside driver keeps of the devices it holds (see [`DeviceUsers`]).
    pub fn device_users(&self) -> DeviceUsers { self.device_users.clone() }

    /// A node's run, when it has reached `until` (`ready`, `terminal`, `request`).
    pub fn wait_reached(&self, node: &str, until: &str) -> Option<Value> {
        let node = self.node(node)?;
        let run = node.run.as_deref().and_then(|run| self.run(run))?;
        let reached = match until {
            "ready" => run.ready || run.terminal(),
            "request" => run.request.is_some() || run.terminal(),
            _ => run.terminal(),
        };
        reached.then(|| run.row())
    }
}

#[cfg(test)]
pub(crate) mod testing;
