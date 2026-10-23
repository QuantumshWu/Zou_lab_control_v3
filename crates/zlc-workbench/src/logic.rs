//! Logic rows and their runs: drafts and their admission, the running
//! configuration, device claims, follow intent of processors, task run
//! directories, operator requests and acquisition children. One state machine
//! per run (`Run::state`); the row's `state`/`status` are projections of it.
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use serde_json::{Map, Value, json};

use zlc_core::control::{NODE_INSTANCE, node_instance};

use crate::{Host, RunSpec, SignalInfo, Ticket, TypeDescriptor, Workbench, fields, text, workspace};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Draft {
    pub values: Map<String, Value>,
    pub source_signal: String,
    pub device_keys: BTreeMap<String, String>,
    pub artifact_inputs: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Instance { Opening, Open, Failed(String) }

/// A start that waits for the runs holding its devices (or its own last run) to end.
#[derive(Clone, Debug)]
pub struct PendingStart { pub origin: String, pub acquisition: Option<(String, u64, String)> }

pub struct Node {
    pub id: String,
    pub name: String,
    pub type_id: String,
    pub draft: Draft,
    pub auto_preview: bool,
    pub revision: u64,
    /// The configuration the current run, follow or pending start uses.
    pub running: Option<Draft>,
    /// The answers that configuration was started with.
    pub frozen: Option<Frozen>,
    /// The current, else the last, run.
    pub run: Option<String>,
    pub starts: u64,
    /// A refused Start or a failed run; cleared by the next draft change.
    pub error: Option<String>,
    /// A processor started on a source that comes and goes keeps following it.
    pub follow: bool,
    pub follow_attempted: Option<String>,
    pub pending: Option<PendingStart>,
    pub removing: bool,
    pub instance: Instance,
    /// Plugin query answers, the newest of each: name -> (draft revision, answer). What the
    /// page draws and the issues it shows come from them, a newer draft's too until its own land.
    pub answers: BTreeMap<String, (u64, Result<Value, String>)>,
    /// The queries answered for this draft (a Start waits until each is).
    pub checked: BTreeSet<String>,
    /// The workspace resources the draft names, read at its last change: field -> resolved value.
    pub resources: Map<String, Value>,
    /// Why a named resource could not be read (Start is refused while any).
    pub resource_issues: Vec<String>,
    /// A Start that waits for the plugin to finish checking the draft: (origin, acquisition request).
    pub start_after_check: Option<(String, Option<(String, u64, String)>)>,
}

/// The node editor's model (`editor_facts`).
const EDITOR_FACTS: &str = "editor_facts";

impl Node {
    /// A node on the board, drawn at once from its type's answers for its default draft (the
    /// catalogue's) until its own process answers its draft.
    pub(crate) fn new(id: String, name: String, type_id: String, draft: Draft, auto_preview: bool, descriptor: Option<&TypeDescriptor>) -> Self {
        let answers = descriptor.map(|item| item.node().answers.iter().map(|(query, answer)| (query.clone(), (0, Ok(answer.clone())))).collect()).unwrap_or_default();
        Node { id, name, type_id, draft, auto_preview, revision: 1, running: None, frozen: None, run: None, starts: 0, error: None, follow: false, follow_attempted: None,
               pending: None, removing: false, instance: Instance::Opening, answers, checked: BTreeSet::new(), resources: Map::new(), resource_issues: Vec::new(),
               start_after_check: None }
    }

    /// A new draft, or the same one asked again: its queries are outstanding until answered anew.
    /// The page goes on drawing the newest answers meanwhile (v2 answered a draft at once), so a
    /// field the edit did not touch keeps its value, its choices and its words.
    fn recheck(&mut self) { self.checked.clear(); }
}

/// One device a run uses (v2 DeviceClaim). A declared requirement takes its device
/// exclusively; a device the draft names for some of its fields (a scan's device axis,
/// v2's runtime claim) shares it, its fields protected. Two runs conflict only when
/// both claims on one device are exclusive (v2 `_conflict`).
#[derive(Clone, Debug, PartialEq)]
pub struct Claim { pub argument: String, pub device: String, pub protected: Option<Vec<String>>, pub exclusive: bool }

/// What a Start froze of the draft it started (v2 LogicCandidate): its effective values
/// and the devices it names. A restart from that configuration (queued, follow,
/// acquisition) uses them after the draft has moved on.
#[derive(Clone, Debug, PartialEq)]
pub struct Frozen { pub draft: Draft, pub values: Map<String, Value>, pub named: Vec<Claim> }

pub struct Run {
    pub id: String,
    pub node: String,
    /// starting, running, stopping, completed, stopped, failed.
    pub state: String,
    pub ready: bool,
    pub started_at: String,
    pub ended_at: Option<String>,
    pub progress: Option<Value>,
    pub request: Option<Value>,
    pub directory: Option<PathBuf>,
    pub artifacts: Vec<Value>,
    pub generation: Option<String>,
    pub stop_reason: Option<String>,
    pub error: Option<Value>,
    pub claims: Vec<Claim>,
    pub start_record: Option<Value>,
    /// Preview signals already shown, and the generation each had at Start.
    pub previewed: BTreeSet<String>,
    pub preview_baseline: BTreeMap<String, Option<String>>,
    /// Panels this (task) run opened for its previews; closed at its end.
    pub preview_panels: Vec<String>,
    /// (parent run, ticket, completion) when a parent asked for this run.
    pub acquisition: Option<(String, u64, String)>,
    /// Acquisition children this run started (node ids).
    pub children: Vec<String>,
}

impl Run {
    pub fn terminal(&self) -> bool { matches!(self.state.as_str(), "completed" | "stopped" | "failed") }
    pub fn row(&self) -> Value {
        let mut row = json!({"run": self.id, "node_id": self.node, "state": self.state, "ready": self.ready, "started_at": self.started_at,
               "ended_at": self.ended_at, "directory": self.directory, "generation": self.generation, "stop_reason": self.stop_reason,
               "error": self.error});
        // Cloned, not serialized again by json! (a point review's request carries every candidate).
        row["progress"] = self.progress.clone().unwrap_or(Value::Null);
        row["request"] = self.request.clone().unwrap_or(Value::Null);
        row["artifacts"] = Value::Array(self.artifacts.clone());
        row
    }
}

/// A claim as Device Manager rows and Device Control show it.
pub struct ClaimView { pub node: String, pub node_name: String, pub run: String, pub protected: Option<Vec<String>>, pub exclusive: bool }

/// Who uses a device (v2 DeviceUseCoordinator): the runs that claim it, and the driver
/// outside the workbench that holds it (the Pulse Editor playing the sequencer).
pub struct DeviceUse { pub runs: Vec<ClaimView>, pub outside: Option<String> }

impl DeviceUse {
    /// Every user, by the name a sentence calls it: its nodes, then the outside driver.
    pub fn users(&self) -> Vec<String> {
        let mut users: Vec<String> = Vec::new();
        for user in self.runs.iter().map(|claim| claim.node_name.clone()).chain(self.outside.clone()) {
            if !users.contains(&user) { users.push(user); }
        }
        users
    }
}

/// v2 DeviceUseBusy's words.
pub(crate) fn in_use(users: &[String]) -> String { format!("device is in use by {}", users.join(", ")) }

/// v2 `logic_name_text`: words one space apart.
pub(crate) fn name_text(text: &str) -> String { text.split_whitespace().collect::<Vec<_>>().join(" ") }

fn name_key(text: &str) -> String { name_text(text).to_lowercase() }

/// v2 `logic_name_problem`: None when `text` may name `node` on a board of `others` (id, name).
pub(crate) fn name_problem<'a>(text: &str, node: &str, others: impl IntoIterator<Item = (&'a str, &'a str)>) -> Option<String> {
    if text.contains('/') || text.chars().any(|c| c.is_control() || c == '\u{2028}' || c == '\u{2029}') {
        return Some("A name is one line of text without “/”.".into());
    }
    if text.is_empty() { return Some("A node needs a name.".into()); }
    let key = name_key(text);
    if names_a_panel(&key) { return Some(format!("“{text}” is how a panel is named; choose another name.")); }
    for (id, name) in others.into_iter().filter(|(id, _)| *id != node) {
        if name_key(name) == key { return Some(format!("Another node is already called “{name}”.")); }
        if name_key(id) == key { return Some(format!("“{id}” is another node's internal ID; choose another name.")); }
    }
    None
}

fn names_a_panel(text: &str) -> bool { text.strip_prefix("panel-").is_some_and(|digits| !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit())) }

/// `@logic/<producer>/<output>`, split at the last `/` of the remainder.
pub fn split_signal(signal: &str) -> Option<(&str, &str)> {
    signal.strip_prefix("@logic/")?.rsplit_once('/').filter(|(producer, output)| !producer.is_empty() && !output.is_empty())
}

pub fn signal_key(producer: &str, output: &str) -> String { format!("@logic/{producer}/{output}") }

/// The draft queries asked whenever a draft changes (PLUGIN_PROTOCOL §2);
/// admission that needs data or files runs in the plugin at Start.
const DRAFT_QUERIES: [&str; 9] = ["declare_outputs", "declare_previews", "resolve_field_availability", "resolve_field_choices", "resolve_defaults",
                                  "resolve_device_claims", "editor_facts", ADMIT_SOURCE, ADMIT];
/// What the draft cannot compute from what its source published (v2 admit_source): asked
/// once every member of the source bundle has a schema; its answers are draft issues.
const ADMIT_SOURCE: &str = "admit_source";
/// What only the node's own code can say of the draft (v2 finalize_logic_draft): its values
/// projected by its schema, validator included; its resources (admit_resources); its artifact
/// files decoded. Every node answers it (the SDK); its answers are draft issues.
const ADMIT: &str = "admit";
/// The argument a device the draft names is claimed and passed as (`tunable:<key>`).
const TUNABLE: &str = "tunable:";
/// Asked per output; its answers feed the consumers' `source` fact.
const FRAMES_QUERY: &str = "describe_output_frames";
/// A run asking the operator to set a physical axis by hand (v2 manual_axis_setting).
const MANUAL_AXIS: &str = "manual-axis";
/// Admission issues that only mean "not yet": a Start waits them out.
pub(crate) const CHECKING: &str = "Checking the draft…";
pub(crate) const STARTING_PLUGIN: &str = " is starting its plugin";
/// A device the draft needs is still opening or reopening (Init/Apply in flight): a Start waits for it.
pub(crate) const DEVICE_OPENING: &str = " is still opening";

/// Whether an admission issue only means "not yet": Start waits it out (v2's start_camera
/// opened the camera, then started).
fn waiting(issue: &str) -> bool { issue == CHECKING || issue.ends_with(STARTING_PLUGIN) || issue.ends_with(DEVICE_OPENING) }

impl Workbench {
    pub(crate) fn node(&self, id: &str) -> Option<&Node> { self.nodes.iter().find(|node| node.id == id) }
    fn node_mut(&mut self, id: &str) -> Result<&mut Node, String> { self.nodes.iter_mut().find(|node| node.id == id).ok_or_else(|| format!("No logic node {id:?}")) }
    pub(crate) fn run(&self, id: &str) -> Option<&Run> { self.runs.iter().find(|run| run.id == id) }
    pub(crate) fn run_mut(&mut self, id: &str) -> Option<&mut Run> { self.runs.iter_mut().find(|run| run.id == id) }
    fn node_type(&self, type_id: &str) -> Result<TypeDescriptor, String> {
        self.catalog.get(type_id).filter(|item| !item.is_device()).cloned().ok_or_else(|| format!("No logic node type {type_id:?} is installed"))
    }

    /// The node's current run, if it has not ended.
    pub(crate) fn active_run(&self, node: &Node) -> Option<&Run> { node.run.as_deref().and_then(|id| self.run(id)).filter(|run| !run.terminal()) }

    /// The Task that has the board (v2 task takeover): running, or accepted and waiting to start.
    pub(crate) fn active_task(&self) -> Option<&Node> {
        self.nodes.iter().find(|node| (self.active_run(node).is_some() || node.pending.is_some() || node.start_after_check.is_some())
            && self.catalog.get(&node.type_id).is_some_and(|item| item.node().node_kind == "task"))
    }

    /// v2 `_task_acquisition_nodes`: the nodes a Task may start while it has the board -- the
    /// acquisition its draft names and the processors its source comes through, and theirs.
    pub(crate) fn task_acquisition_nodes(&self, task: &Node) -> BTreeSet<String> {
        let (mut allowed, mut pending) = (BTreeSet::new(), vec![task.id.clone()]);
        while let Some(id) = pending.pop() {
            let Some(node) = self.node(&id).filter(|_| allowed.insert(id.clone())) else { continue };
            let Some(descriptor) = self.catalog.get(&node.type_id) else { continue };
            let effective = fields::effective(&descriptor.parameters, &node.draft.values);
            let acquisition = descriptor.node().acquisition_input.as_ref().filter(|field| descriptor.parameters.iter()
                .find(|item| &item.name == *field).is_none_or(|item| fields::enabled(item, &effective)));
            if let Some(chosen) = acquisition.and_then(|field| node.draft.values.get(field)).and_then(Value::as_str).filter(|chosen| !chosen.trim().is_empty()) {
                pending.push(chosen.to_owned());
            }
            let producer = split_signal(&node.draft.source_signal).map(|(producer, _)| producer.split('/').next().unwrap_or(producer));
            if let Some(producer) = producer.filter(|producer| self.node(producer).and_then(|other| self.catalog.get(&other.type_id))
                .is_some_and(|other| other.node().node_kind == "processor")) {
                pending.push(producer.to_owned());
            }
        }
        allowed.remove(&task.id);
        allowed
    }

    /// What the strip is told outside a command (v2 ConsolePresenter._report): new notes,
    /// once; while a Task runs, "<task>: <status>" each time that status changes (v2
    /// _project_task_takeover), and once more when the Task ends (_finish_task_takeover).
    pub(crate) fn sync_console_report(&mut self, host: &dyn Host) {
        self.say_new_notes();
        let id = self.active_task().map(|task| task.id.clone()).or_else(|| self.task_said.as_ref().map(|(id, _)| id.clone()));
        let Some(node) = id.and_then(|id| self.node(&id)) else { self.task_said = None; return };
        let row = self.node_row(node, &host.signals());
        let said = (node.id.clone(), format!("{}: {}", node.name, row["status"].as_str().unwrap_or("")));
        let running = self.active_run(node).is_some();
        if self.task_said.as_ref() != Some(&said) { self.console_report = (said.1.clone(), if row["state"] == "error" { "error" } else { "task" }); }
        self.task_said = running.then_some(said);
    }

    pub(crate) fn say_new_notes(&mut self) {
        if self.notes == self.notes_said { return; }
        let said = std::mem::replace(&mut self.notes_said, self.notes.clone());
        if !self.notes.is_empty() { self.console_report = (self.notes.join("\n"), "warning"); }
        // Notes that are no longer true (a file read again) take their line with them.
        else if self.console_report == (said.join("\n"), "warning") { self.console_report = (String::new(), "idle"); }
    }

    fn task_gate(&self, verb: &str) -> Result<(), String> {
        match self.active_task() { Some(task) => Err(format!("{} is running; stop the task before {verb}", task.name)), None => Ok(()) }
    }

    /// The active task's published and preview signals: panels showing them keep their science.
    pub(crate) fn task_protected_signals(&self, task: &Node) -> BTreeSet<String> {
        let mut signals: BTreeSet<String> = self.node_outputs(task).into_iter().map(|(signal, _)| signal).collect();
        signals.extend(self.node_previews(task).iter().map(|preview| self.preview_signal(task, preview)));
        signals
    }

    fn preview_signal(&self, node: &Node, preview: &crate::types::Preview) -> String {
        if preview.producer.is_empty() { signal_key(&node.id, &preview.output) } else { signal_key(&format!("{}/{}", node.id, preview.producer), &preview.output) }
    }

    /// What the node previews: its draft's `declare_previews` answer (v2 NodePreviewSpec rows,
    /// whose output is a declaration), else the type's static previews.
    pub(crate) fn node_previews(&self, node: &Node) -> Vec<crate::types::Preview> {
        let declared = node.answers.get("declare_previews").and_then(|(_, answer)| answer.as_ref().ok()).and_then(Value::as_array).map(|items| items.iter()
            .filter_map(|item| {
                let mut item = item.clone();
                if let Some(name) = item["output"]["name"].as_str().map(str::to_owned) { item["output"] = json!(name); }
                serde_json::from_value::<crate::types::Preview>(item).ok()
            }).collect());
        declared.unwrap_or_else(|| self.catalog.get(&node.type_id).map(|item| item.node().previews.clone()).unwrap_or_default())
    }

    /// (signal, contract) the node publishes: the plugin's `declare_outputs` answer for this draft, else its static outputs.
    pub(crate) fn node_outputs(&self, node: &Node) -> Vec<(String, Option<String>)> {
        let declared = node.answers.get("declare_outputs").and_then(|(_, answer)| answer.as_ref().ok())
            .and_then(|answer| serde_json::from_value::<Vec<crate::types::OutputSpec>>(answer.clone()).ok());
        let outputs = declared.or_else(|| self.catalog.get(&node.type_id).map(|item| item.node().outputs.clone())).unwrap_or_default();
        outputs.into_iter().map(|output| (signal_key(&node.id, &output.name), output.contract)).collect()
    }

    /// v2 `logic_name_problem`: None when `text` may name `node` on this board.
    fn name_problem(&self, text: &str, node: &str) -> Option<String> {
        name_problem(text, node, self.nodes.iter().map(|other| (other.id.as_str(), other.name.as_str())))
    }

    fn free_node_id(&self, type_id: &str) -> String {
        (1..).map(|n: u32| if n == 1 { type_id.to_owned() } else { format!("{type_id}{n}") })
            .find(|id| self.node(id).is_none() && self.name_problem(id, id).is_none()).unwrap()
    }

    fn active_requirements<'a>(descriptor: &'a TypeDescriptor, values: &Map<String, Value>) -> Vec<&'a crate::types::DeviceRequirement> {
        let effective: Map<String, Value> = descriptor.parameters.iter()
            .map(|field| (field.name.clone(), values.get(&field.name).cloned().unwrap_or_else(|| field.default.clone()))).collect();
        descriptor.node().devices.iter().filter(|requirement| requirement.enabled_when.as_ref()
            .is_none_or(|(controller, offered)| offered.iter().any(|value| effective.get(controller).unwrap_or(&Value::Null) == value))).collect()
    }

    fn has_source(descriptor: &TypeDescriptor) -> bool { !descriptor.node().inputs.is_empty() }

    /// Ask the node's process every draft query its type answers, for this revision.
    fn send_queries(&mut self, id: &str) { self.ask_queries(id, None) }

    /// Ask the draft queries (`only`: just that one) for the node's current draft.
    fn ask_queries(&mut self, id: &str, only: Option<&str>) {
        let Some(node) = self.node(id) else { return };
        let (resources, resource_issues) = self.read_resources(&node.type_id, &node.draft.values);
        let node = self.node_mut(id).unwrap();
        (node.resources, node.resource_issues) = (resources, resource_issues);
        let node = self.node(id).unwrap();
        if node.instance != Instance::Open { return; }
        let Some(descriptor) = self.catalog.get(&node.type_id) else { return };
        let names: Vec<String> = descriptor.node().queries.iter().filter(|name| DRAFT_QUERIES.contains(&name.as_str())).cloned().collect();
        // The facts by name; the SDK passes them to the v2 callable in its own order (PLUGIN_PROTOCOL §5).
        let revision = node.revision;
        let args = json!({"values": Value::Object(node.draft.values.clone()), "devices": self.device_infos(&node.draft.device_keys),
            "resources": node.resources});
        // A node's editor offers every tunable device of the bench (v2 `tunable_devices`),
        // each under the argument a run claims it by.
        let mut bench = node.draft.device_keys.clone();
        bench.extend(self.devices.iter().filter(|(key, live)| self.device_open(key) && live.working_point["tunable"] == true)
            .map(|(key, _)| (format!("{TUNABLE}{key}"), key.clone())));
        let source = self.source_frames(node);
        let bundle = self.bundles.get(id).map_or_else(|| json!([]), |(_, bundle)| bundle.clone());
        let complete = self.bundle_complete(id);
        let editor = json!({"values": args["values"], "devices": self.device_infos(&bench), "resources": args["resources"], "source": source,
            "bundle": bundle, "artifacts": self.artifact_facts(node)});
        let admit = json!({"values": args["values"], "bundle": bundle});
        let admitted = json!({"values": Value::Object(self.draft_values(node)), "resources": args["resources"], "artifacts": self.artifact_facts(node)});
        // Each output's frames, from the frames of this node's own source (v2 describe_output_frames).
        let frames: Vec<String> = if descriptor.node().queries.iter().any(|name| name == FRAMES_QUERY) {
            self.node_outputs(node).into_iter().filter_map(|(signal, _)| split_signal(&signal).map(|(_, output)| output.to_owned())).collect()
        } else { Vec::new() };
        for name in names {
            if (name == ADMIT_SOURCE && !complete) || only.is_some_and(|only| only != name) { continue; }
            let ticket = self.ticket(Ticket::Query { node: id.to_owned(), revision, name: name.clone() });
            let facts = match name.as_str() { EDITOR_FACTS => &editor, ADMIT_SOURCE => &admit, ADMIT => &admitted, _ => &args };
            self.plugins.query(&node_instance(id), ticket, &name, facts);
        }
        for output in frames.into_iter().filter(|_| only.is_none()) {
            let ticket = self.ticket(Ticket::Query { node: id.to_owned(), revision, name: format!("{FRAMES_QUERY}:{output}") });
            self.plugins.query(&node_instance(id), ticket, FRAMES_QUERY, &json!({"values": args["values"], "output_name": output, "source": source}));
        }
    }

    /// The node's source frames `[axis_name, [labels...]]` (PLUGIN_PROTOCOL §2): what its
    /// source's producer answered for that output, else null.
    fn source_frames(&self, node: &Node) -> Value {
        let Some((producer, output)) = split_signal(&node.draft.source_signal) else { return Value::Null };
        self.node(producer).and_then(|producer| producer.answers.get(&format!("{FRAMES_QUERY}:{output}")))
            .and_then(|(_, answer)| answer.clone().ok()).unwrap_or(Value::Null)
    }

    /// The signals a node's source bundle holds, by member name: every output of the source's
    /// producer when its input takes the whole bundle (v2 `select_bundle`), else the selected one.
    fn bundle_signals(&self, node: &Node) -> Vec<(String, String)> {
        let Some((producer, _)) = split_signal(&node.draft.source_signal) else { return Vec::new() };
        let whole = self.catalog.get(&node.type_id).and_then(|item| item.node().inputs.first()).is_some_and(|input| input.select_bundle);
        let signals: Vec<String> = match self.node(producer).filter(|_| whole) {
            Some(producer) => self.node_outputs(producer).into_iter().map(|(signal, _)| signal).collect(),
            None => vec![node.draft.source_signal.clone()],
        };
        signals.into_iter().filter_map(|signal| split_signal(&signal).map(|(_, name)| name.to_owned()).map(|name| (name, signal))).collect()
    }

    /// The source bundles the nodes' queries read (an editor, an admission check), each member
    /// with its schemas (v2 `input_bundle`), rebuilt when a member is declared anew or first
    /// publishes; a node whose bundle changed asks again.
    pub(crate) fn refresh_bundles(&mut self, host: &mut dyn Host) {
        let signals = host.signals();
        let readers: Vec<String> = self.nodes.iter().filter(|node| self.catalog.get(&node.type_id)
            .is_some_and(|item| item.node().queries.iter().any(|name| name == EDITOR_FACTS || name == ADMIT_SOURCE))).map(|node| node.id.clone()).collect();
        for id in readers {
            let members = self.bundle_signals(self.node(&id).unwrap());
            let key = members.iter().map(|(name, signal)| {
                let seen = signals.iter().find(|info| info.name == *signal);
                format!("{name}={}", seen.map_or_else(String::new, |info| format!("{}:{}", info.generation, info.published)))
            }).collect::<Vec<_>>().join(";");
            if self.bundles.get(&id).is_some_and(|(known, _)| *known == key) { continue; }
            let bundle = Value::Array(members.iter().map(|(name, signal)| {
                let (schema, event) = host.schemas(signal).unwrap_or((Value::Null, Value::Null));
                json!({"name": name, "schema": schema, "event_schema": event})
            }).collect());
            self.bundles.insert(id.clone(), (key, bundle));
            self.send_queries(&id);
        }
    }

    /// The files the draft names (`artifacts`): its artifact inputs as written, and each
    /// path as the workspace resolves it, so an editor can read what a file is.
    fn artifact_facts(&self, node: &Node) -> Value {
        let mut resolved = Map::new();
        for value in node.draft.artifact_inputs.values() {
            let paths: Vec<&str> = match value {
                Value::String(path) => vec![path.as_str()],
                Value::Array(claims) => claims.iter().filter_map(|claim| claim["path"].as_str()).collect(),
                _ => Vec::new(),
            };
            for path in paths.into_iter().filter(|path| !path.is_empty()) {
                resolved.insert(path.to_owned(), json!(self.workspace.resolve(path).display().to_string()));
            }
        }
        json!({"inputs": node.draft.artifact_inputs, "resolved": resolved})
    }

    /// Every member of the node's source bundle has a schema.
    fn bundle_complete(&self, id: &str) -> bool {
        self.bundles.get(id).and_then(|(_, bundle)| bundle.as_array())
            .is_some_and(|members| !members.is_empty() && members.iter().all(|member| !member["schema"].is_null()))
    }

    /// The bench's devices changed: editors that offer them ask again.
    fn refresh_editors(&mut self) {
        let editors: Vec<String> = self.nodes.iter().filter(|node| self.catalog.get(&node.type_id)
            .is_some_and(|item| item.node().queries.iter().any(|name| name == "editor_facts"))).map(|node| node.id.clone()).collect();
        for id in editors { self.send_queries(&id); }
    }

    /// The devices the draft itself names (`resolve_device_claims`: a scan's device
    /// axes), each claimed for its named fields and passed as its own argument.
    fn named_claims(node: &Node) -> Vec<Claim> {
        let Some((_, Ok(Value::Object(named)))) = node.answers.get("resolve_device_claims") else { return Vec::new() };
        named.iter().filter_map(|(argument, claim)| claim["instance"].as_str().map(|device| Claim { argument: argument.clone(), device: device.to_owned(),
            protected: Some(claim["fields"].as_array().into_iter().flatten().filter_map(Value::as_str).map(str::to_owned).collect()), exclusive: false })).collect()
    }

    /// Each enabled workspace resource a draft names, read now (v2 reads files at a draft's
    /// explicit boundaries: a change, Refresh, Start): field -> resolved value, and the issues.
    fn read_resources(&self, type_id: &str, values: &Map<String, Value>) -> (Map<String, Value>, Vec<String>) {
        let (mut resolved, mut issues) = (Map::new(), Vec::new());
        let Some(descriptor) = self.catalog.get(type_id) else { return (resolved, issues) };
        let effective = fields::effective(&descriptor.parameters, values);
        for spec in &descriptor.node().workspace_resources {
            let Some(field) = descriptor.parameters.iter().find(|field| field.name == spec.field) else { continue };
            if !fields::enabled(field, &effective) { continue; }
            let label = if field.label.is_empty() { &field.name } else { &field.label };
            match self.workspace.resource(spec, label, values.get(&spec.field).and_then(Value::as_str).unwrap_or("")) {
                Ok(value) => { resolved.insert(spec.field.clone(), value); }
                Err(issue) => issues.push(issue),
            }
        }
        (resolved, issues)
    }

    /// The draft as it takes effect: the node's own defaults in its empty and derived fields
    /// (v2 `_defaults_from_resources`, the `resolve_defaults` answer for this draft) and the
    /// switches the chosen devices cannot do off.
    pub(crate) fn draft_values(&self, node: &Node) -> Map<String, Value> {
        let mut values = node.draft.values.clone();
        self.node_defaults(node, &mut values);
        self.unavailable_switches_off(node, &mut values);
        values
    }

    fn node_defaults(&self, node: &Node, values: &mut Map<String, Value>) {
        let Some((_, Ok(Value::Object(offered)))) = node.answers.get("resolve_defaults") else { return };
        let Some(descriptor) = self.catalog.get(&node.type_id) else { return };
        for (name, value) in offered {
            let Some(field) = descriptor.parameters.iter().find(|field| &field.name == name) else { continue };
            let empty = match values.get(name) { None | Some(Value::Null) => true, Some(Value::String(text)) => text.trim().is_empty(), _ => false };
            if field.derived || empty { values.insert(name.clone(), value.clone()); }
        }
    }

    /// v2: a switch the chosen device cannot do has one neutral effective value, off; the raw
    /// draft keeps the operator's choice, so a capable device brings it back.
    fn unavailable_switches_off(&self, node: &Node, values: &mut Map<String, Value>) {
        let Some((_, Ok(Value::Object(unavailable)))) = node.answers.get("resolve_field_availability") else { return };
        let Some(descriptor) = self.catalog.get(&node.type_id) else { return };
        for field in descriptor.parameters.iter().filter(|field| field.kind == "bool" && unavailable.contains_key(&field.name)) {
            values.insert(field.name.clone(), Value::Bool(false));
        }
    }

    /// Selected devices as the plugin's device proxies are described.
    fn device_infos(&self, keys: &BTreeMap<String, String>) -> Value {
        // `tunable`: the proxy reads and tunes its knobs (the SDK's tunable quartet);
        // `reads`: the read-only methods the device implements (its own unit conversions).
        Value::Object(keys.iter().filter_map(|(argument, key)| self.devices.get(key).map(|live| {
            let kind = self.live_type(live);
            (argument.clone(), json!({"instance": key, "role": self.device_role(key), "type_id": kind.map_or(live.entry.type_id.as_str(), |item| item.type_id.as_str()),
                "capabilities": kind.map(|item| item.capabilities.clone()).unwrap_or_default(),
                "tunable": live.working_point["tunable"] == true, "reads": live.working_point["reads"].clone()}))
        })).collect())
    }

    pub(crate) fn query_answered(&mut self, id: &str, revision: u64, name: &str, result: Result<Value, String>, host: &mut dyn Host) {
        let mut changed = None;
        let mut readmit = false;
        if let Ok(node) = self.node_mut(id) {
            if node.revision == revision {
                node.checked.insert(name.to_owned());
                let previous = node.answers.insert(name.to_owned(), (revision, result.clone()));
                let fresh = previous.is_none_or(|(before, answer)| before != revision || answer != result);
                changed = name.strip_prefix(FRAMES_QUERY).and_then(|rest| rest.strip_prefix(':')).filter(|_| fresh).map(|output| signal_key(id, output));
                // Defaults the node fills in change what its schema admits: ask that again.
                if name == "resolve_defaults" && fresh { node.checked.remove(ADMIT); readmit = true; }
            }
        }
        // New frames of an output: the nodes reading it ask again with their new source.
        if let Some(signal) = changed {
            let readers: Vec<String> = self.nodes.iter().filter(|node| node.draft.source_signal == signal).map(|node| node.id.clone()).collect();
            for reader in readers { self.send_queries(&reader); }
        }
        if readmit { self.ask_queries(id, Some(ADMIT)); }
        self.start_checked(id, host);
    }

    /// A Start that waited for the draft check: now it starts, or reports why not.
    fn start_checked(&mut self, id: &str, host: &mut dyn Host) {
        let Some(node) = self.node(id) else { return };
        let Some((origin, acquisition)) = node.start_after_check.clone() else { return };
        let signals = host.signals();
        if self.issues(node, &signals).iter().any(|issue| waiting(issue)) { return; }
        if let Ok(node) = self.node_mut(id) { node.start_after_check = None; }
        let parent = acquisition.clone();
        if let Err(error) = self.start_node(id, None, &origin, acquisition, host) {
            if let Some((parent, ticket, _)) = parent { self.plugins.acquired(&parent, ticket, Err(error)); }
        }
    }

    pub(crate) fn add_node(&mut self, params: &Value, host: &mut dyn Host) -> Result<Value, String> {
        self.task_gate("adding a node")?;
        let type_id = text(params, "type_id")?;
        let descriptor = self.node_type(type_id)?;
        let id = match params["node_id"].as_str() {
            Some(id) => {
                let id = id.trim().to_owned();
                if let Some(node) = self.node(&id) { return Err(format!("{id} is already used by node {}", node.name)); }
                if let Some(problem) = self.name_problem(&id, &id) { return Err(problem); }
                id
            }
            None => self.free_node_id(type_id),
        };
        let name = name_text(params["name"].as_str().unwrap_or(&id));
        if let Some(problem) = self.name_problem(&name, &id) { return Err(problem); }
        let supplied = params["values"].as_object().cloned().unwrap_or_default();
        let (values, _) = fields::project_values(&descriptor.parameters, &supplied, false)?;
        let mut device_keys: BTreeMap<String, String> = params["device_keys"].as_object().map(|keys| keys.iter()
            .map(|(argument, key)| (argument.clone(), key.as_str().unwrap_or("").to_owned())).collect()).unwrap_or_default();
        for requirement in Self::active_requirements(&descriptor, &values) {
            if !device_keys.contains_key(&requirement.argument) {
                let first = self.device_options(&requirement.capability).into_iter().next().map(|(key, _)| key).unwrap_or_default();
                device_keys.insert(requirement.argument.clone(), first);
            }
        }
        let draft = Draft { values, source_signal: params["source_signal"].as_str().unwrap_or("").trim().to_owned(), device_keys,
            artifact_inputs: params["artifact_inputs"].as_object().cloned().unwrap_or_default() };
        self.plugins.open(&node_instance(&id), type_id, &json!({}), &json!({}));
        self.nodes.push(Node::new(id.clone(), name, type_id.to_owned(), draft, params["auto_preview"].as_bool().unwrap_or(true), Some(&descriptor)));
        self.board.dirty = true;
        let _ = host;
        Ok(json!({"node": id}))
    }

    pub(crate) fn rename_node(&mut self, id: &str, name: &str) -> Result<Value, String> {
        let name = name_text(name);
        self.node(id).ok_or_else(|| format!("No logic node {id:?}"))?;
        if let Some(problem) = self.name_problem(&name, id) { return Err(problem); }
        self.node_mut(id)?.name = name.clone();
        self.board.dirty = true;
        Ok(json!({"node": id, "name": name}))
    }

    pub(crate) fn edit_node(&mut self, params: &Value, host: &mut dyn Host) -> Result<Value, String> {
        let _ = host;
        self.edit_draft(params)
    }

    fn edit_draft(&mut self, params: &Value) -> Result<Value, String> {
        let id = text(params, "node")?;
        self.task_gate("changing a logic draft")?;
        let node = self.node(id).ok_or_else(|| format!("No logic node {id:?}"))?;
        if let Some(expected) = params["expected_draft_revision"].as_u64().filter(|expected| *expected != node.revision) {
            return Err(format!("{} changed (draft revision {} is not {expected}); read it again", node.name, node.revision));
        }
        let descriptor = self.node_type(&node.type_id)?;
        let mut draft = node.draft.clone();
        if let Some(patch) = params["values"].as_object() {
            let mut merged = draft.values.clone();
            merged.extend(patch.clone());
            draft.values = fields::project_values(&descriptor.parameters, &merged, false)?.0;
        }
        if let Some(source) = params["source_signal"].as_str() {
            let source = source.trim();
            if !source.is_empty() && !Self::has_source(&descriptor) { return Err(format!("{} has no dataset input", descriptor.label())); }
            if self.node_outputs(node).iter().any(|(signal, _)| signal == source) { return Err("A node cannot read its own output".into()); }
            draft.source_signal = source.to_owned();
        }
        if let Some(keys) = params["device_keys"].as_object() {
            for (argument, key) in keys {
                if !descriptor.node().devices.iter().any(|requirement| &requirement.argument == argument) {
                    return Err(format!("{} has no device input {argument:?}", descriptor.label()));
                }
                draft.device_keys.insert(argument.clone(), key.as_str().ok_or("device keys are text")?.to_owned());
            }
        }
        if let Some(patch) = params["artifact_inputs"].as_object() {
            for (name, value) in patch {
                let Some(spec) = descriptor.node().artifact_inputs.iter().find(|spec| &spec.name == name) else {
                    return Err(format!("{} has no artifact input {name:?}", descriptor.label()));
                };
                let valid = if spec.frame_claims { value.as_array().is_some_and(|claims| claims.iter().all(|claim| claim["path"].is_string() && claim["frames"].is_array())) }
                    else { value.is_string() };
                if !valid { return Err(format!("{}: {}", spec.label, if spec.frame_claims { "a list of {path, frames}" } else { "a path" })); }
                draft.artifact_inputs.insert(name.clone(), value.clone());
            }
        }
        // A requirement the edit turned on takes the first compatible device, as at Add.
        for requirement in Self::active_requirements(&descriptor, &draft.values) {
            if draft.device_keys.get(&requirement.argument).is_none_or(String::is_empty) {
                if let Some((first, _)) = self.device_options(&requirement.capability).into_iter().next() {
                    draft.device_keys.insert(requirement.argument.clone(), first);
                }
            }
        }
        let auto_preview = params["auto_preview"].as_bool();
        let node = self.node_mut(id)?;
        let changed = draft != node.draft;
        if let Some(flag) = auto_preview { node.auto_preview = flag; }
        if changed {
            node.draft = draft;
            node.revision += 1;
            node.error = None;
            node.recheck();
        }
        let revision = node.revision;
        if changed { self.send_queries(id); }
        if changed || auto_preview.is_some() { self.board.dirty = true; }
        Ok(json!({"node": id, "draft_revision": revision}))
    }

    /// v2 `_route_exact_panel_selection`: a region committed on a panel showing a node's
    /// output asks that node what it sets in its draft (`selection_patch`, for the kinds of
    /// region its type maps); `None` gives back what this panel's region set (v2
    /// `_restore_producer_draft`). A running Task's panel keeps its science. Whether a
    /// draft changed now.
    pub fn panel_selection(&mut self, panel: &str, signal: &str, selection: Option<&Value>, context: &Value) -> bool {
        if self.routed_selections.get(panel) == selection { return false; }
        let Some(selection) = selection else {
            self.routed_selections.remove(panel);
            return self.restore_producer_draft(panel);
        };
        self.routed_selections.insert(panel.to_owned(), selection.clone());
        if self.panel_frozen(panel, signal).is_some() { return false; }
        let Some(node) = split_signal(signal).and_then(|(producer, _)| self.node(producer)).filter(|node| node.instance == Instance::Open) else { return false };
        let kinds = (selection["plot_kind"].as_str().unwrap_or(""), selection["selector_kind"].as_str().unwrap_or(""));
        let mapped = self.catalog.get(&node.type_id).is_some_and(|item| item.node().selection_mappings.iter()
            .any(|mapping| (mapping.plot_kind.as_str(), mapping.selector_kind.as_str()) == kinds));
        if !mapped { return false; }
        let (id, args) = (node.id.clone(), json!({"selection": selection, "draft": Value::Object(node.draft.values.clone()), "context": context}));
        let ticket = self.ticket(Ticket::Selection { node: id.clone(), panel: panel.to_owned(), selection: selection.clone() });
        self.plugins.query(&node_instance(&id), ticket, "selection_patch", &args);
        false
    }

    /// The producer's answer: the draft fields the region sets, each remembered with
    /// what it held before (once, whichever panel set it first). A region that has
    /// changed or gone since it asked sets nothing.
    pub(crate) fn selection_patched(&mut self, id: &str, panel: &str, selection: &Value, result: Result<Value, String>) {
        if self.routed_selections.get(panel) != Some(selection) { return; }
        let patch = match result {
            Ok(Value::Object(patch)) => patch,
            // The region names nothing this producer can be set from.
            Ok(_) => return,
            Err(error) => { self.console_report = (format!("{panel}: {error}"), "warning"); return; }
        };
        let Some(node) = self.node(id) else { return };
        let before: Vec<(String, Value)> = patch.keys().map(|name| (name.clone(), self.selection_restore.get(&(id.to_owned(), name.clone()))
            .map_or_else(|| node.draft.values.get(name).cloned().unwrap_or(Value::Null), |(_, value)| value.clone()))).collect();
        match self.edit_draft(&json!({"node": id, "values": patch})) {
            Ok(_) => for (name, value) in before { self.selection_restore.insert((id.to_owned(), name), (panel.to_owned(), value)); },
            Err(error) => self.console_report = (error, "warning"),
        }
    }

    /// Give back the draft fields still owned by this panel's region.
    pub(crate) fn restore_producer_draft(&mut self, panel: &str) -> bool {
        let mut owned: BTreeMap<String, Map<String, Value>> = BTreeMap::new();
        for ((node, field), (owner, value)) in &self.selection_restore {
            if owner == panel { owned.entry(node.clone()).or_default().insert(field.clone(), value.clone()); }
        }
        let mut changed = false;
        for (node, values) in owned {
            if self.edit_draft(&json!({"node": node, "values": values})).is_ok() {
                changed = true;
                for field in values.keys() { self.selection_restore.remove(&(node.clone(), field.clone())); }
            }
        }
        changed
    }

    pub(crate) fn check_node(&mut self, id: &str, host: &mut dyn Host) -> Result<Value, String> {
        self.reopen_failed(id);
        let node = self.node_mut(id)?;
        node.recheck();
        node.error = None;
        self.send_queries(id);
        let signals = host.signals();
        let node = self.node(id).unwrap();
        Ok(json!({"node": id, "issues": self.issues(node, &signals)}))
    }

    pub(crate) fn revert_node(&mut self, id: &str, _host: &mut dyn Host) -> Result<Value, String> {
        self.task_gate("changing a logic draft")?;
        let node = self.node_mut(id)?;
        let running = node.running.clone().ok_or_else(|| format!("{} is not running; there is no running configuration to return to", node.name))?;
        if node.draft != running {
            node.draft = running;
            node.revision += 1;
            node.recheck();
            node.error = None;
            self.send_queries(id);
        }
        Ok(json!({"node": id}))
    }

    /// Admission: every reason the draft cannot start now, in v2's order.
    pub(crate) fn issues(&self, node: &Node, signals: &[SignalInfo]) -> Vec<String> {
        let Some(descriptor) = self.catalog.get(&node.type_id) else { return vec![format!("{} is not installed", node.type_id)] };
        let spec = descriptor.node();
        let mut issues = Vec::new();
        match &node.instance {
            Instance::Failed(error) => issues.push(format!("{} could not start its plugin: {error}", node.name)),
            Instance::Opening => issues.push(format!("{}{STARTING_PLUGIN}", node.name)),
            Instance::Open => {}
        }
        issues.extend(node.resource_issues.iter().cloned());
        issues.extend(fields::completeness_issue(&descriptor.parameters, &self.draft_values(node)));
        let answer = |name: &str| node.answers.get(name).map(|(_, answer)| answer);
        let outstanding = spec.queries.iter().filter(|name| DRAFT_QUERIES.contains(&name.as_str()))
            .filter(|name| name.as_str() != ADMIT_SOURCE || self.bundle_complete(&node.id))
            .any(|name| !node.checked.contains(name));
        // What the variables cannot compute from what the source published (v2: draft issues).
        if let Some(Ok(Value::Array(found))) = answer(ADMIT_SOURCE).filter(|_| self.bundle_complete(&node.id)) {
            issues.extend(found.iter().filter_map(Value::as_str).map(str::to_owned));
        }
        if let Some(Ok(Value::Array(found))) = answer(ADMIT) { issues.extend(found.iter().filter_map(Value::as_str).map(str::to_owned)); }
        // A draft query its plugin could not answer is the draft's issue (v2 raised it where it finalized the draft).
        for name in DRAFT_QUERIES { if let Some(Err(error)) = answer(name) { issues.push(format!("{name}: {error}")); } }
        if outstanding && node.instance == Instance::Open { issues.push(CHECKING.into()); }
        // v2: an acquisition the draft names (its field enabled) must be a measurement that
        // reports what the node waits for; none named is the node running without one.
        let effective = fields::effective(&descriptor.parameters, &node.draft.values);
        let acquisition = spec.acquisition_input.as_ref().filter(|field| descriptor.parameters.iter()
            .find(|item| &item.name == *field).is_none_or(|item| fields::enabled(item, &effective)));
        if let Some(field) = acquisition.filter(|field| node.draft.values.get(*field).and_then(Value::as_str).is_some_and(|chosen| !chosen.trim().is_empty())) {
            let chosen = node.draft.values.get(field).and_then(Value::as_str).unwrap_or("");
            let wanted = spec.acquisition_completion.as_deref().unwrap_or("ready");
            let ok = self.node(chosen).filter(|other| other.id != node.id).and_then(|other| self.catalog.get(&other.type_id))
                .is_some_and(|other| other.node().node_kind == "measurement" && (wanted == "terminal" || other.node().reports_ready));
            if !ok {
                let label = descriptor.parameters.iter().find(|item| &item.name == field).map(|item| item.label.clone()).unwrap_or_else(|| field.clone());
                issues.push(format!("{label} must name a measurement node that {}", if wanted == "terminal" { "finishes" } else { "reports ready" }));
            }
        }
        for requirement in Self::active_requirements(descriptor, &node.draft.values) {
            let options = self.device_options(&requirement.capability);
            let key = node.draft.device_keys.get(&requirement.argument).map(String::as_str).unwrap_or("");
            // Opening, or closed by an Apply that reopens it with new settings.
            let opening = self.devices.iter().find(|(id, live)| (live.state == crate::apparatus::LiveState::Opening || live.reopen.is_some())
                && (key.is_empty() || key == id.as_str()) && self.catalog.get(&live.entry.type_id).is_some_and(|item| item.capabilities.contains(&requirement.capability)));
            if let Some((id, _)) = opening.filter(|_| !options.iter().any(|(open, _)| open == key)) {
                issues.push(format!("{}{DEVICE_OPENING}", self.device_role(id)));
            } else if options.is_empty() {
                issues.push(format!("{} needs a {} and this apparatus has none open", descriptor.label(), requirement.capability));
            } else if !options.iter().any(|(id, _)| id == key) {
                let roles: Vec<&str> = options.iter().map(|(_, role)| role.as_str()).collect();
                issues.push(if key.is_empty() { format!("{} must be chosen: one of {}", requirement.argument, roles.join(", ")) }
                    else { format!("{} is not an open {}; choose one of {}", self.device_role(key), requirement.capability, roles.join(", ")) });
            }
        }
        // v2: what the device cannot do refuses a value only when the draft asks for it; an
        // unavailable switch is simply off (draft_values), not an issue.
        if let Some(Ok(Value::Object(unavailable))) = answer("resolve_field_availability") {
            let effective = self.draft_values(node);
            for (name, reason) in unavailable {
                let value = effective.get(name).cloned().unwrap_or(Value::Null);
                if !matches!(value, Value::Null | Value::Bool(false)) && value != json!("") {
                    issues.push(reason.as_str().unwrap_or("not available with this device").to_owned());
                }
            }
        }
        for claim in Self::named_claims(node).iter().filter(|claim| !self.device_open(&claim.device)) {
            issues.push(format!("{} is not an open device", self.device_role(&claim.device)));
        }
        // v2: a published device is handed over; no Logic here uses it until Remote is off.
        let used: Vec<String> = Self::active_requirements(descriptor, &node.draft.values).iter()
            .filter_map(|requirement| node.draft.device_keys.get(&requirement.argument).cloned())
            .chain(Self::named_claims(node).iter().map(|claim| claim.device.clone())).collect();
        for device in used.iter().filter(|device| self.device_published(device)) {
            let issue = format!("{} is published on the bench fabric; withdraw Remote to use it here", self.device_role(device));
            if !issues.contains(&issue) { issues.push(issue); }
        }
        if let Some(input) = source_input(spec, &effective) {
            let label = source_words(input);
            let source = node.draft.source_signal.as_str();
            if source.is_empty() { issues.push(format!("{label} must be selected")); }
            else if let Some(wanted) = &input.contract {
                let declared = self.nodes.iter().flat_map(|other| self.node_outputs(other)).find(|(signal, _)| signal == source).and_then(|(_, contract)| contract);
                if declared.as_ref().is_some_and(|contract| contract != wanted) {
                    issues.push(format!("{source} is not declared as {wanted} Dataset"));
                }
            }
        }
        for spec in &spec.artifact_inputs {
            let value = node.draft.artifact_inputs.get(&spec.name).cloned().unwrap_or(Value::Null);
            let paths: Vec<String> = match &value {
                Value::String(path) if !path.is_empty() => vec![path.clone()],
                Value::Array(claims) => claims.iter().filter_map(|claim| claim["path"].as_str().map(str::to_owned)).collect(),
                _ => Vec::new(),
            };
            let label = if spec.label.is_empty() { &spec.name } else { &spec.label };
            if paths.is_empty() && spec.required.unwrap_or(true) { issues.push(format!("{label} must be selected")); }
            for path in paths.iter().filter(|path| path.is_empty() || !self.workspace.resolve(path).is_file()) {
                issues.push(if path.is_empty() { format!("{label} must be selected") } else { format!("{label}: {path} does not exist") });
            }
        }
        let _ = signals;
        issues
    }

    fn source_published<'a>(&self, signals: &'a [SignalInfo], source: &str) -> Option<&'a SignalInfo> {
        signals.iter().find(|signal| signal.name == source && signal.published)
    }

    pub(crate) fn claims_on(&self, device: &str) -> Vec<ClaimView> {
        self.nodes.iter().filter_map(|node| self.active_run(node).map(|run| (node, run)))
            .flat_map(|(node, run)| run.claims.iter().filter(|claim| claim.device == device)
                .map(|claim| ClaimView { node: node.id.clone(), node_name: node.name.clone(), run: run.id.clone(), protected: claim.protected.clone(),
                                         exclusive: claim.exclusive }))
            .collect()
    }

    pub(crate) fn claiming_nodes(&self, device: &str) -> Vec<String> {
        let mut nodes: Vec<String> = self.claims_on(device).into_iter().map(|claim| claim.node).collect();
        nodes.dedup();
        nodes
    }

    /// The nodes a start with `claims` must stop and wait for (v2 `_conflict`): those whose
    /// run claims exclusively a device this start also takes exclusively.
    fn conflicting_nodes(&self, claims: &[Claim]) -> BTreeSet<String> {
        claims.iter().filter(|claim| claim.exclusive).flat_map(|claim| self.claims_on(&claim.device))
            .filter(|held| held.exclusive).map(|held| held.node).collect()
    }

    /// Who uses a device now: the one question every admission that touches a device asks.
    pub(crate) fn device_use(&self, device: &str) -> DeviceUse {
        let outside = self.device_users.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).get(device).cloned();
        DeviceUse { runs: self.claims_on(device), outside }
    }

    /// The devices a start of `node` with `draft` would claim. A draft other than the node's
    /// own (a queued, follow or acquisition restart) names the devices it was started with.
    fn wanted_claims(&self, node: &Node, draft: &Draft) -> Vec<Claim> {
        let Some(descriptor) = self.catalog.get(&node.type_id) else { return Vec::new() };
        let mut claims: Vec<Claim> = Self::active_requirements(descriptor, &draft.values).into_iter().filter_map(|requirement| {
            draft.device_keys.get(&requirement.argument).filter(|key| !key.is_empty())
                .map(|key| Claim { argument: requirement.argument.clone(), device: key.clone(), protected: requirement.protected_fields.clone(), exclusive: true })
        }).collect();
        if *draft == node.draft { claims.extend(Self::named_claims(node)); }
        else if let Some(frozen) = node.frozen.as_ref().filter(|frozen| frozen.draft == *draft) { claims.extend(frozen.named.iter().cloned()); }
        claims
    }

    /// v2 DeviceUseBusy: a claimed device driven outside the workbench refuses the start.
    fn outside_use(&self, claims: &[Claim]) -> Option<String> {
        let mut users: Vec<String> = Vec::new();
        for user in claims.iter().filter_map(|claim| self.device_use(&claim.device).outside) { if !users.contains(&user) { users.push(user); } }
        (!users.is_empty()).then(|| in_use(&users))
    }

    /// Start, or Restart a running node (its run stops first).
    /// `acquisition`: a parent run asks for this node, which then starts with
    /// its running configuration (or its draft) and answers the parent.
    pub(crate) fn start_node(&mut self, id: &str, expected: Option<u64>, origin: &str, acquisition: Option<(String, u64, String)>,
                             host: &mut dyn Host) -> Result<Value, String> {
        let node = self.node(id).ok_or_else(|| format!("No logic node {id:?}"))?;
        if node.removing { return Err(format!("{} is being removed", node.name)); }
        if let Some(task) = self.active_task().filter(|task| task.id != id) {
            // v2 _task_acquisition_nodes: while a Task runs, only its own acquisition chain starts, and only for it.
            if acquisition.is_none() || !self.task_acquisition_nodes(task).contains(id) {
                return Err(format!("{} is running; stop the task before starting another node", task.name));
            }
        }
        if let Some(expected) = expected.filter(|expected| *expected != node.revision) {
            return Err(format!("{} changed (draft revision {} is not {expected}); read it again", node.name, node.revision));
        }
        let reopened = self.reopen_failed(id);
        let node = self.node(id).unwrap();
        let signals = host.signals();
        let running = acquisition.as_ref().and(node.running.clone());
        if running.is_none() || reopened {
            let issues = self.issues(node, &signals);
            // The plugin has not answered for this draft yet: Start once it has.
            if !issues.is_empty() && issues.iter().all(|issue| waiting(issue)) {
                self.node_mut(id)?.start_after_check = Some((origin.to_owned(), acquisition));
                return Ok(json!({"node": id, "phase": "checking"}));
            }
            if let Some(first) = issues.first() {
                let error = format!("Draft: {first}");
                self.node_mut(id)?.error = Some(first.clone());
                return Err(error);
            }
        }
        self.node_mut(id)?.start_after_check = None;
        let node = self.node(id).unwrap();
        let descriptor = self.node_type(&node.type_id)?;
        // A Start from the draft freezes what its answers say of it (v2 LogicCandidate); a
        // restart of the running configuration keeps what that one was started with.
        let frozen = running.is_none().then(|| Frozen { draft: node.draft.clone(), values: self.draft_values(node), named: Self::named_claims(node) });
        let draft = running.unwrap_or_else(|| node.draft.clone());
        if let Some(error) = self.outside_use(&self.wanted_claims(node, &draft)) {
            self.node_mut(id)?.error = Some(error.clone());
            return Err(error);
        }
        let processor = descriptor.node().node_kind == "processor";
        let source_absent = processor && self.source_published(&signals, &draft.source_signal).is_none();
        if processor { self.plugins.follow(&node_instance(id), Some(&draft.source_signal)); }
        let node = self.node_mut(id)?;
        node.error = None;
        node.running = Some(draft.clone());
        if frozen.is_some() { node.frozen = frozen; }
        node.follow = processor;
        if processor { node.follow_attempted = None; }
        if source_absent && acquisition.is_none() {
            // A processor waits for its source; it starts when the source publishes.
            let name = node.name.clone();
            self.console_report = (format!("{name} following {} (waiting for the signal)", self.signal_label(&draft.source_signal)), "task");
            return Ok(json!({"node": id, "phase": "following"}));
        }
        node.pending = Some(PendingStart { origin: origin.to_owned(), acquisition });
        let claims = self.wanted_claims(self.node(id).unwrap(), &draft);
        let mut holders = self.conflicting_nodes(&claims);
        if self.node(id).and_then(|node| self.active_run(node)).is_some() { holders.insert(id.to_owned()); }
        for holder in &holders { self.stop_run_of(holder, "another start needs its devices"); }
        self.launch_pending(host);
        let node = self.node(id).unwrap();
        let phase = if node.pending.is_some() { "waiting" } else { "starting" };
        // v2: a start behind other nodes' runs says so; one behind its own last run waits quietly.
        let mut blockers: Vec<&str> = holders.iter().filter(|holder| *holder != id).filter_map(|holder| self.node(holder)).map(|node| node.name.as_str()).collect();
        if node.pending.is_some() && !blockers.is_empty() {
            blockers.sort_unstable();
            self.console_report = (format!("{} queued while {} stops", node.name, blockers.join(", ")), "task");
        }
        Ok(json!({"node": id, "phase": phase, "waiting_for": holders}))
    }

    /// Start every pending node whose devices and own last run are free.
    pub(crate) fn launch_pending(&mut self, host: &mut dyn Host) {
        let waiting: Vec<String> = self.nodes.iter().filter(|node| node.pending.is_some()).map(|node| node.id.clone()).collect();
        for id in waiting {
            let node = self.node(id.as_str()).unwrap();
            if self.active_run(node).is_some() { continue; }
            let draft = node.running.clone().unwrap_or_else(|| node.draft.clone());
            let claims = self.wanted_claims(node, &draft);
            if let Some(error) = self.outside_use(&claims) {
                let pending = self.node_mut(&id).unwrap().pending.take();
                if let Some((parent, ticket, _)) = pending.and_then(|pending| pending.acquisition) { self.plugins.acquired(&parent, ticket, Err(error.clone())); }
                self.node_mut(&id).unwrap().error = Some(error);
                continue;
            }
            if !self.conflicting_nodes(&claims).is_empty() || claims.iter().any(|claim| !self.device_open(&claim.device)) {
                if claims.iter().any(|claim| !self.device_open(&claim.device) && self.claiming_nodes(&claim.device).is_empty()) {
                    let pending = self.node_mut(&id).unwrap().pending.take();
                    let missing: Vec<&str> = claims.iter().filter(|claim| !self.device_open(&claim.device)).map(|claim| claim.device.as_str()).collect();
                    let error = format!("{} is not open", missing.join(", "));
                    if let Some((parent, ticket, _)) = pending.and_then(|pending| pending.acquisition) { self.plugins.acquired(&parent, ticket, Err(error.clone())); }
                    self.node_mut(&id).unwrap().error = Some(error);
                }
                continue;
            }
            let pending = self.node_mut(&id).unwrap().pending.take().unwrap();
            let parent = pending.acquisition.clone();
            if let Err(error) = self.launch(&id, draft, claims, pending, host) {
                if let Some((parent, ticket, _)) = parent { self.plugins.acquired(&parent, ticket, Err(error.clone())); }
                self.node_mut(&id).unwrap().error = Some(error);
            }
        }
    }

    fn launch(&mut self, id: &str, draft: Draft, claims: Vec<Claim>, pending: PendingStart, host: &mut dyn Host) -> Result<(), String> {
        let node = self.node(id).unwrap();
        let descriptor = self.node_type(&node.type_id)?;
        let task = descriptor.node().node_kind == "task";
        let run_id = format!("{id}#{}", node.starts + 1);
        // The node's own defaults apply to the draft they were answered for.
        let authored = if draft == node.draft { self.draft_values(node) }
            else if let Some(frozen) = node.frozen.as_ref().filter(|frozen| frozen.draft == draft) { frozen.values.clone() }
            else { draft.values.clone() };
        let effective: Map<String, Value> = descriptor.parameters.iter()
            .map(|field| (field.name.clone(), authored.get(&field.name).cloned().unwrap_or_else(|| field.default.clone()))).collect();
        // Start is a boundary at which named files are read again.
        let (resolved, resource_issues) = self.read_resources(&node.type_id, &authored);
        if let Some(issue) = resource_issues.into_iter().next() { return Err(issue); }
        let resources: Map<String, Value> = descriptor.node().workspace_resources.iter().filter_map(|spec| resolved.get(&spec.field)
            .map(|value| (if spec.argument.is_empty() { spec.field.clone() } else { spec.argument.clone() }, value.clone()))).collect();
        // A folder named relatively is under the workspace (v2 projects it as root/<text>).
        let values: Map<String, Value> = descriptor.parameters.iter().filter(|field| fields::enabled(field, &effective))
            .map(|field| (field.name.clone(), match (field.kind.as_str(), &effective[&field.name]) {
                ("folder", Value::String(path)) if !path.trim().is_empty() => json!(self.workspace.resolve(path.trim())),
                (_, value) => value.clone(),
            })).collect();
        let devices: BTreeMap<String, String> = claims.iter().map(|claim| (claim.argument.clone(), claim.device.clone())).collect();
        let shared: BTreeSet<String> = claims.iter().filter(|claim| !claim.exclusive).map(|claim| claim.argument.clone()).collect();
        let artifacts: Map<String, Value> = descriptor.node().artifact_inputs.iter().filter_map(|spec| draft.artifact_inputs.get(&spec.name)
            .map(|value| (spec.name.clone(), match value {
                Value::String(path) => json!(self.workspace.resolve(path)),
                Value::Array(claims) => Value::Array(claims.iter().map(|claim| json!({"path": self.workspace.resolve(claim["path"].as_str().unwrap_or("")),
                    "frames": claim["frames"]})).collect()),
                other => other.clone(),
            }))).collect();
        let outputs: BTreeMap<String, String> = self.node_outputs(node).into_iter()
            .filter_map(|(signal, _)| split_signal(&signal).map(|(_, output)| (output.to_owned(), signal.clone()))).collect();
        let started_at = workspace::utc_now();
        let input = json!({"authored": values, "source_signal": draft.source_signal,
            "devices": claims.iter().map(|claim| (claim.argument.clone(), json!({"instance_id": claim.device,
                "capability": descriptor.node().devices.iter().find(|requirement| requirement.argument == claim.argument).map(|requirement| requirement.capability.clone())})))
                .collect::<Map<String, Value>>(),
            "artifacts": artifacts, "resources": resolved});
        let (directory, start_record) = if task {
            let folder = self.workspace.day_folder()?;
            let directory = workspace::unique_directory(&folder, &node.type_id)?;
            let record = json!({"schema": "zlc.task-start", "run_id": directory.file_name().map(|name| name.to_string_lossy().into_owned()),
                "task": {"api_name": node.type_id, "instance_id": id}, "input": input, "started_at": started_at});
            workspace::write_readable(&directory.join("start.json"), &record)?;
            (Some(directory), Some(record))
        } else { (None, None) };
        let signals = host.signals();
        let preview_baseline = self.node_previews(node).iter().map(|preview| {
            let signal = self.preview_signal(node, preview);
            let generation = signals.iter().find(|item| item.name == signal).map(|item| item.generation.clone());
            (signal, generation)
        }).collect();
        let spec = RunSpec { run: run_id.clone(), node_id: id.to_owned(), instance: node_instance(id), type_id: node.type_id.clone(), values, source_signal: draft.source_signal.clone(),
            device_labels: devices.values().map(|key| (key.clone(), self.device_role(key).to_owned())).collect(),
            devices, shared, artifacts, resources, directory: directory.clone(), outputs, origin: pending.origin.clone() };
        let follow_generation = self.source_published(&signals, &draft.source_signal).map(|signal| signal.generation.clone());
        self.runs.push(Run { id: run_id.clone(), node: id.to_owned(), state: "starting".into(), ready: false,
            started_at, ended_at: None, progress: None, request: None, directory, artifacts: Vec::new(), generation: None, stop_reason: None,
            error: None, claims, start_record, previewed: BTreeSet::new(), preview_baseline, preview_panels: Vec::new(),
            acquisition: pending.acquisition.clone(), children: Vec::new() });
        self.trim_runs();
        // v2 _activate_candidate: what a run starts with is its producer's own; a region that
        // wrote it no longer gives it back.
        self.selection_restore.retain(|(node, _), _| node != id);
        let node = self.node_mut(id)?;
        node.starts += 1;
        node.run = Some(run_id);
        node.running = Some(draft);
        if node.follow { node.follow_attempted = follow_generation; }
        // v2 _activate_candidate: said when a run starts, however it was asked for (Start, a
        // follow restart, a task's acquisition).
        self.console_report = (format!("{} started", node.name), "task");
        self.plugins.start(&spec);
        Ok(())
    }

    fn trim_runs(&mut self) {
        let finished: Vec<usize> = self.runs.iter().enumerate().filter(|(_, run)| run.terminal()).map(|(index, _)| index).collect();
        let excess = finished.len().saturating_sub(50);
        for index in finished.into_iter().take(excess).rev() {
            let id = self.runs[index].id.clone();
            if self.nodes.iter().all(|node| node.run.as_deref() != Some(id.as_str())) { self.runs.remove(index); }
        }
    }

    fn stop_run_of(&mut self, id: &str, reason: &str) {
        let Some(run) = self.node(id).and_then(|node| self.active_run(node)).map(|run| run.id.clone()) else { return };
        let run = self.run_mut(&run).unwrap();
        run.state = "stopping".into();
        run.stop_reason.get_or_insert_with(|| reason.to_owned());
        let id = run.id.clone();
        // The operator's request it waits on is cancelled with it (v2: Stop wakes the wait).
        let request = run.request.take().and_then(|request| request["request"].as_str().map(str::to_owned));
        // Every Stop is sent (v2 cancelled the host each time): one asked again is never a no-op.
        self.plugins.stop(&id);
        if let Some(request) = request { self.plugins.answer(&id, &request, None); }
    }

    /// A run asks the operator. One asked after its Stop is answered as stopped and never
    /// shown (v2: Stop wakes the wait, and a request made after it does not wait).
    pub(crate) fn run_requested(&mut self, id: &str, request: Value) {
        let Some(run) = self.run_mut(id) else { return };
        if run.state != "stopping" && !run.terminal() { run.request = Some(request); return; }
        if let Some(name) = request["request"].as_str() { self.plugins.answer(id, name, None); }
    }

    /// Stop: following, a waiting start and the run itself end. Never refused.
    pub(crate) fn stop_node(&mut self, id: &str, _host: &mut dyn Host) -> Result<Value, String> {
        let node = self.node_mut(id)?;
        if std::mem::take(&mut node.follow) { self.plugins.follow(&node_instance(id), None); }
        let node = self.node_mut(id)?;
        node.start_after_check = None;
        let pending = node.pending.take();
        if let Some((parent, ticket, _)) = pending.and_then(|pending| pending.acquisition) {
            self.plugins.acquired(&parent, ticket, Err("the operator pressed Stop".into()));
        }
        let children = self.node(id).and_then(|node| self.active_run(node)).map(|run| run.children.clone()).unwrap_or_default();
        self.stop_run_of(id, "the operator pressed Stop");
        for child in children { self.stop_run_of(&child, "its parent run stopped"); }
        let active = self.node(id).and_then(|node| self.active_run(node)).is_some();
        if !active { let node = self.node_mut(id)?; node.running = None; node.frozen = None; }
        Ok(json!({"node": id, "stopping": active}))
    }

    pub(crate) fn remove_node(&mut self, id: &str, host: &mut dyn Host) -> Result<Value, String> {
        self.task_gate("removing a node")?;
        self.stop_node(id, host)?;
        let active = self.node(id).and_then(|node| self.active_run(node)).is_some();
        if active {
            let node = self.node_mut(id)?;
            node.removing = true;
            self.console_report = (format!("{} is stopping", node.name), "task");
        } else { self.retire_node(id); }
        Ok(json!({"node": id, "removed": !active}))
    }

    fn retire_node(&mut self, id: &str) {
        if let Some(node) = self.node(id) { self.console_report = (format!("removed {}", node.name), "task"); }
        self.nodes.retain(|node| node.id != id);
        self.selection_restore.retain(|(node, _), _| node != id);
        self.plugins.close(&node_instance(id));
        self.board.dirty = true;
    }

    /// `operator`: the answer comes from the window (a client's and a run's do not).
    pub(crate) fn answer_request(&mut self, id: &str, request: &str, response: &Value, operator: bool) -> Result<Value, String> {
        let node = self.node(id).ok_or_else(|| format!("No logic node {id:?}"))?;
        let run = self.active_run(node).ok_or_else(|| format!("{} is not running", node.name))?;
        let current = run.request.as_ref().and_then(|current| current["request"].as_str()).unwrap_or("");
        if current != request { return Err(format!("{request} is not {}'s current request{}", node.name,
            if current.is_empty() { String::new() } else { format!(" ({current} is)") })); }
        let response = (!response.is_null()).then_some(response);
        // v2: a physical setting only a hand makes is confirmed by the operator; anyone may cancel it (the run's Stop).
        if response.is_some() && !operator && run.request.as_ref().is_some_and(|current| current["kind"] == MANUAL_AXIS) {
            return Err("Physical manual-axis requests cannot be answered here; the operator answers them at the window".into());
        }
        let run_id = run.id.clone();
        if response.is_some_and(|response| !response.is_object()) { return Err("A response is an object of the request's fields, or null to cancel".into()); }
        // A cancel stops the run first, as v2's Stop woke the wait with the stop already set.
        if response.is_none() { self.stop_run_of(id, "the operator cancelled the request"); }
        self.plugins.answer(&run_id, request, response);
        if let Some(run) = self.run_mut(&run_id) { run.request = None; }
        Ok(json!({"node": id, "request": request}))
    }

    pub(crate) fn estimate_node(&mut self, id: &str, request: u64) -> Result<(), String> {
        let node = self.node(id).ok_or_else(|| format!("No logic node {id:?}"))?;
        let answers = self.catalog.get(&node.type_id).is_some_and(|item| item.node().queries.iter().any(|name| name == "estimate_run"));
        if !answers || node.instance != Instance::Open {
            self.replies.push((request, Ok(json!({"status": "unknown", "reason": format!("{} gives no estimate", node.name)}))));
            return Ok(());
        }
        let args = json!({"values": self.draft_values(node), "devices": self.device_infos(&node.draft.device_keys), "resources": node.resources});
        let ticket = self.ticket(Ticket::Estimate { request });
        self.plugins.query(&node_instance(id), ticket, "estimate_run", &args);
        Ok(())
    }

    /// A plugin instance opened: a node's (named `node/<id>`) or a device's.
    pub(crate) fn instance_opened(&mut self, instance: &str, working_point: Value, host: &mut dyn Host) {
        if let Some(id) = instance.strip_prefix(NODE_INSTANCE) {
            if let Ok(node) = self.node_mut(id) {
                node.instance = Instance::Open;
                self.send_queries(id);
                self.start_checked(id, host);
            }
        } else if self.device_opened(instance, working_point) {
            self.default_device_keys(instance);
            self.refresh_editors();
            self.launch_pending(host);
            // Starts that waited for this device go now.
            let waiting: Vec<String> = self.nodes.iter().filter(|node| node.start_after_check.is_some()).map(|node| node.id.clone()).collect();
            for node in waiting { self.start_checked(&node, host); }
        }
    }

    /// A device just opened: drafts that have no device yet for a requirement
    /// it provides take it (as a draft added after it opened would).
    fn default_device_keys(&mut self, device: &str) {
        let mut changed = Vec::new();
        for index in 0..self.nodes.len() {
            let Some(descriptor) = self.catalog.get(&self.nodes[index].type_id).cloned() else { continue };
            for requirement in Self::active_requirements(&descriptor, &self.nodes[index].draft.values) {
                let empty = self.nodes[index].draft.device_keys.get(&requirement.argument).is_none_or(String::is_empty);
                if empty && self.capability_of(device, &requirement.capability) {
                    let node = &mut self.nodes[index];
                    node.draft.device_keys.insert(requirement.argument.clone(), device.to_owned());
                    node.revision += 1;
                    node.recheck();
                    changed.push(node.id.clone());
                }
            }
        }
        for id in changed { self.send_queries(&id); }
    }

    pub(crate) fn instance_failed(&mut self, instance: &str, error: String, host: &mut dyn Host) {
        let Some(id) = instance.strip_prefix(NODE_INSTANCE) else {
            if self.device_failed(instance, error) {
                for node in self.claiming_nodes(instance) { self.stop_run_of(&node, "its device failed"); }
                self.close_released();
                // A Start that waited for this device now says why it cannot.
                let waiting: Vec<String> = self.nodes.iter().filter(|node| node.start_after_check.is_some()).map(|node| node.id.clone()).collect();
                for node in waiting { self.start_checked(&node, host); }
            }
            return;
        };
        if let Ok(node) = self.node_mut(id) { node.instance = Instance::Failed(error.clone()); }
        // A lost node process ends its run with that reason.
        if let Some(run) = self.node(id).and_then(|node| self.active_run(node)).map(|run| run.id.clone()) {
            self.run_finished(&run, "failed", Some(json!({"type": "PluginLost", "message": error})), None, host);
        }
    }

    pub(crate) fn instance_closed(&mut self, instance: &str, error: Option<String>, host: &mut dyn Host) {
        match instance.strip_prefix(NODE_INSTANCE) {
            Some(id) => if let (Ok(node), Some(error)) = (self.node_mut(id), error) { node.instance = Instance::Failed(error); },
            None => if self.device_closed(instance, error) { self.refresh_editors(); self.launch_pending(host); },
        }
    }

    /// v2: a row can always be started again. A node whose plugin instance was lost (its process
    /// ended) or did not open opens again, as Apply opens a device that failed.
    fn reopen_failed(&mut self, id: &str) -> bool {
        let Some(node) = self.nodes.iter_mut().find(|node| node.id == id && matches!(node.instance, Instance::Failed(_))) else { return false };
        node.instance = Instance::Opening;
        let type_id = node.type_id.clone();
        self.plugins.open(&node_instance(id), &type_id, &json!({}), &json!({}));
        true
    }

    pub(crate) fn run_started(&mut self, id: &str) {
        if let Some(run) = self.run_mut(id).filter(|run| run.state == "starting") { run.state = "running".into(); }
    }

    pub(crate) fn run_ready(&mut self, id: &str) {
        let Some(run) = self.run_mut(id) else { return };
        run.ready = true;
        if run.state == "starting" { run.state = "running".into(); }
        if let Some((parent, ticket, completion)) = run.acquisition.clone().filter(|(_, _, completion)| completion == "ready") {
            run.acquisition = None;
            let row = run.row();
            let _ = completion;
            self.plugins.acquired(&parent, ticket, Ok(row));
        }
    }

    /// A parent run asks for its acquisition node (v2 `restart_logic`).
    pub(crate) fn run_acquire(&mut self, parent: &str, ticket: u64, child: &str, completion: &str, host: &mut dyn Host) {
        let refuse = |workbench: &mut Self, error: String| workbench.plugins.acquired(parent, ticket, Err(error));
        let Some(parent_node) = self.run(parent).filter(|run| !run.terminal()).map(|run| run.node.clone()) else { return refuse(self, "the asking run has ended".into()) };
        // v2 _restart_acquisition: a run asked to stop starts nothing more.
        if self.run(parent).is_some_and(|run| run.state == "stopping") { return refuse(self, "stopped before acquisition restart".into()); }
        let Some(node) = self.node(child) else { return refuse(self, format!("No logic node {child:?}")) };
        if node.id == parent_node { return refuse(self, "A run cannot acquire itself".into()); }
        let request = Some((parent.to_owned(), ticket, completion.to_owned()));
        match self.start_node(child, None, parent, request, host) {
            Ok(_) => { if let Some(run) = self.run_mut(parent) { run.children.push(child.to_owned()); } }
            Err(error) => refuse(self, error),
        }
    }

    pub(crate) fn run_finished(&mut self, id: &str, outcome: &str, error: Option<Value>, generation: Option<String>, host: &mut dyn Host) {
        let Some(run) = self.run_mut(id) else { return };
        if run.terminal() { return; }
        run.state = match outcome { "completed" | "stopped" | "failed" => outcome.to_owned(), _ => "failed".to_owned() };
        run.ended_at = Some(workspace::utc_now());
        run.request = None;
        if generation.is_some() { run.generation = generation; }
        run.error = error.clone();
        let (node_id, acquisition, children, preview_panels) = (run.node.clone(), run.acquisition.take(), run.children.clone(), std::mem::take(&mut run.preview_panels));
        let record = run.start_record.as_ref().map(|start| json!({"schema": "zlc.task-run", "run_id": start["run_id"], "task": start["task"],
            "input": start["input"], "status": {"state": run.state, "started_at": run.started_at, "ended_at": run.ended_at,
            "progress": run.progress, "stop_reason": run.stop_reason}, "artifacts": run.artifacts, "error": run.error}));
        let directory = run.directory.clone();
        let row = run.row();
        if let (Some(record), Some(directory)) = (record, directory) {
            if let Err(failure) = workspace::write_readable(&directory.join("run.json"), &record) {
                self.notes.push(format!("{id}: the run record could not be written: {failure}"));
            }
        }
        if let Some((parent, ticket, _)) = acquisition {
            let reply = if outcome == "completed" { Ok(row) } else { Err(format!("{node_id} ended {outcome} before it was ready")) };
            self.plugins.acquired(&parent, ticket, reply);
        }
        for child in children { self.stop_run_of(&child, "its parent run ended"); }
        for panel in preview_panels { host.close_panel(&panel); self.board.remove(&panel); }
        let message = error.as_ref().map(|error| error["message"].as_str().map(str::to_owned).unwrap_or_else(|| error.to_string()));
        if let Ok(node) = self.node_mut(&node_id) {
            if outcome == "failed" {
                let message = message.unwrap_or_else(|| "the run failed".into());
                node.error = Some(if node.follow { format!("failed; restarts with the source's next run: {message}") } else { message });
            }
            if !node.follow && node.pending.is_none() { node.running = None; node.frozen = None; }
            if node.removing { self.retire_node(&node_id); }
        }
        self.close_released();
        self.launch_pending(host);
        self.commit_pending_board(host);
    }

    /// Following processors start on each new publication generation of their source.
    pub(crate) fn follow_sources(&mut self, host: &mut dyn Host) {
        let signals = host.signals();
        // v2 _follow_processor_sources: while a Task has the board, only its own chain restarts.
        let allowed = self.active_task().map(|task| self.task_acquisition_nodes(task));
        let ready: Vec<(String, String)> = self.nodes.iter()
            .filter(|node| node.follow && node.pending.is_none() && self.active_run(node).is_none() && !node.removing)
            .filter(|node| allowed.as_ref().is_none_or(|allowed| allowed.contains(&node.id)))
            .filter_map(|node| {
                let source = node.running.as_ref().map(|draft| draft.source_signal.as_str()).unwrap_or(&node.draft.source_signal);
                self.source_published(&signals, source).filter(|signal| node.follow_attempted.as_deref() != Some(signal.generation.as_str()))
                    .map(|signal| (node.id.clone(), signal.generation.clone()))
            }).collect();
        for (id, generation) in ready {
            let node = self.node_mut(&id).unwrap();
            node.follow_attempted = Some(generation);
            node.pending = Some(PendingStart { origin: "operator".into(), acquisition: None });
        }
        self.launch_pending(host);
    }

    /// Declared previews get a panel once their signal first publishes in this run.
    pub(crate) fn open_previews(&mut self, host: &mut dyn Host) {
        let signals = host.signals();
        let shown: BTreeSet<String> = host.panels().into_iter().map(|panel| panel.signal).collect();
        let mut wanted = Vec::new();
        for node in self.nodes.iter().filter(|node| node.auto_preview) {
            let Some(run) = self.active_run(node) else { continue };
            let Some(descriptor) = self.catalog.get(&node.type_id) else { continue };
            for preview in &self.node_previews(node) {
                let signal = self.preview_signal(node, preview);
                if run.previewed.contains(&signal) { continue; }
                let Some(published) = self.source_published(&signals, &signal) else { continue };
                if run.preview_baseline.get(&signal).cloned().flatten().as_deref() == Some(published.generation.as_str()) { continue; }
                let task = descriptor.node().node_kind == "task";
                wanted.push((run.id.clone(), signal.clone(), preview.plot_kind.clone(), preview.semantic.clone(), task, shown.contains(&signal)));
            }
        }
        for (run, signal, kind, semantic, task, already) in wanted {
            if let Some(record) = self.run_mut(&run) { record.previewed.insert(signal.clone()); }
            if already { continue; }
            let target = if semantic.is_object() { json!({"semantic": semantic}) } else { json!({}) };
            match host.add_panel("", &signal, if kind.is_empty() { "image" } else { &kind }, 1, target) {
                Ok(panel) => {
                    self.board.add(&panel);
                    if task { if let Some(record) = self.run_mut(&run) { record.preview_panels.push(panel); } }
                }
                Err(error) => self.notes.push(format!("preview {signal}: {error}")),
            }
        }
    }

    /// Source choices for a node: published signals and other nodes' outputs.
    fn source_choices(&self, node: &Node, signals: &[SignalInfo]) -> Vec<String> {
        let own: BTreeSet<String> = self.node_outputs(node).into_iter().map(|(signal, _)| signal).collect();
        let wanted = self.catalog.get(&node.type_id).and_then(|item| item.node().inputs.first().cloned());
        let Some(input) = wanted else { return Vec::new() };
        let mut names: Vec<(String, Option<String>)> = self.nodes.iter().filter(|other| other.id != node.id).flat_map(|other| self.node_outputs(other)).collect();
        for signal in signals { if !names.iter().any(|(name, _)| name == &signal.name) { names.push((signal.name.clone(), None)); } }
        names.into_iter().filter(|(name, contract)| !own.contains(name)
                && input.contract.as_ref().is_none_or(|wanted| contract.as_ref().is_none_or(|contract| contract == wanted)))
            .map(|(name, _)| name).collect()
    }

    /// v2 _source_choices of a whole-bundle input: one entry per producer (once published, per
    /// owner: a panel's ROI and fit are two), keyed by the member the draft names, else its
    /// first, read as its members in the bundle's order: `Occupancy 1/counts, occupied, frame_judged`.
    fn bundle_choices(&self, node: &Node, sources: &[String], signals: &[SignalInfo]) -> Vec<(String, String)> {
        let mut bundles: Vec<((bool, String), Vec<&String>)> = Vec::new();
        for key in sources {
            let owner = match signals.iter().find(|info| &info.name == key) {
                Some(info) => (true, info.owner.clone()),
                None => (false, split_signal(key).map_or_else(|| key.clone(), |(producer, _)| producer.to_owned())),
            };
            match bundles.iter_mut().find(|(known, _)| *known == owner) { Some((_, members)) => members.push(key), None => bundles.push((owner, vec![key])) }
        }
        let output = |key: &str| split_signal(key).map_or_else(|| key.rsplit('/').next().unwrap_or(key).to_owned(), |(_, output)| output.to_owned());
        bundles.into_iter().map(|(_, mut members)| {
            let anchor = members.iter().find(|key| ***key == node.draft.source_signal).copied().unwrap_or(members[0]);
            // v2 _bundle_order: its publication's members, else its producer node's declared outputs.
            let generation = signals.iter().find(|info| &info.name == anchor).map(|info| info.generation.as_str());
            let order: Vec<String> = match generation {
                Some(generation) => signals.iter().filter(|info| info.generation == generation).map(|info| info.name.clone()).collect(),
                None => split_signal(anchor).and_then(|(producer, _)| self.node(producer)).map(|producer| self.node_outputs(producer).into_iter().map(|(signal, _)| signal).collect()).unwrap_or_default(),
            };
            members.sort_by_key(|key| order.iter().position(|known| known == *key).unwrap_or(order.len()));
            let text = std::iter::once(self.signal_label(members[0])).chain(members[1..].iter().map(|key| output(key))).collect::<Vec<_>>().join(", ");
            (anchor.clone(), text)
        }).collect()
    }

    /// A signal spelled with node names: `Camera measurement/frames`.
    pub fn signal_label(&self, signal: &str) -> String {
        match split_signal(signal) {
            Some((producer, output)) => format!("{}/{output}", self.producer_text(producer)),
            None => signal.to_owned(),
        }
    }

    pub(crate) fn node_row(&self, node: &Node, signals: &[SignalInfo]) -> Value {
        let descriptor = self.catalog.get(&node.type_id);
        // A draft its plugin is still starting for, or checking, shows what the newest answers it
        // has say (its type's for a new node); a Start waits for its own (v2 answered at once).
        let known = descriptor.is_some_and(|item| item.node().queries.iter()
            .filter(|name| DRAFT_QUERIES.contains(&name.as_str()) && ![ADMIT_SOURCE, ADMIT].contains(&name.as_str())).all(|name| node.answers.contains_key(name)));
        let issues: Vec<String> = self.issues(node, signals).into_iter().filter(|issue| !(known && (issue == CHECKING || issue.ends_with(STARTING_PLUGIN)))).collect();
        let run = node.run.as_deref().and_then(|id| self.run(id));
        let active = run.filter(|run| !run.terminal());
        let edited = node.running.as_ref().is_some_and(|running| *running != node.draft);
        let source = node.running.as_ref().map(|draft| draft.source_signal.clone()).unwrap_or_else(|| node.draft.source_signal.clone());
        let editor_facts = node.answers.get(EDITOR_FACTS).map(|(_, answer)| answer);
        let sources = self.source_choices(node, signals);
        let bundle = descriptor.and_then(|item| item.node().inputs.first()).is_some_and(|input| input.select_bundle);
        let (phase, state, mut status) = if node.removing {
            ("removing", "running", "removing: stopping first".to_owned())
        } else if let Some(run) = active {
            let text = match (&run.request, &run.progress, run.state.as_str()) {
                (Some(request), _, _) => format!("waiting for the operator: {}", request["title"].as_str().unwrap_or("input")),
                (_, Some(progress), _) => progress_text(progress),
                (_, _, "starting") => "starting".to_owned(),
                (_, _, "stopping") => "stopping".to_owned(),
                _ => if run.ready || descriptor.is_some_and(|item| !item.node().reports_ready) { "running".to_owned() } else { "preparing".to_owned() },
            };
            (if run.state == "stopping" { "stopping" } else if run.state == "starting" { "starting" } else { "running" }, "running", text)
        } else if node.start_after_check.is_some() {
            ("waiting", "running", "starting once the draft is checked".to_owned())
        } else if node.pending.is_some() {
            let holders: BTreeSet<String> = self.conflicting_nodes(&node.running.as_ref().map(|draft| self.wanted_claims(node, draft)).unwrap_or_default())
                .into_iter().filter(|holder| *holder != node.id).filter_map(|holder| self.node(&holder).map(|other| other.name.clone())).collect();
            ("waiting", "running", if holders.is_empty() { "restart queued".to_owned() } else { format!("waiting for {}", holders.into_iter().collect::<Vec<_>>().join(", ")) })
        } else if node.follow {
            let sealed = signals.iter().any(|signal| signal.name == source && signal.sealed);
            ("following", if node.error.is_some() { "warning" } else { "running" },
             node.error.clone().unwrap_or_else(|| format!("following {} ({})", self.signal_label(&source), if sealed { "source stopped" } else { "waiting for the signal" })))
        } else if let Some(error) = &node.error {
            ("idle", "error", error.clone())
        } else if let Some(first) = issues.first() {
            ("idle", "draft", format!("Draft: {first}"))
        } else {
            ("idle", "idle", run.map(|run| run.state.clone()).unwrap_or_else(|| "not started".to_owned()))
        };
        if edited { status.push_str(" · edited"); }
        let device_choices: Map<String, Value> = descriptor.map(|item| Self::active_requirements(item, &node.draft.values).into_iter()
            .map(|requirement| {
                let mut choices: Vec<Value> = self.device_options(&requirement.capability).into_iter().map(|(id, role)| json!([id, role])).collect();
                if let Some(held) = node.draft.device_keys.get(&requirement.argument).filter(|key| !key.is_empty()) {
                    if !choices.iter().any(|choice| choice[0] == held.as_str()) {
                        choices.push(json!([held, format!("{} (unavailable)", self.device_role(held))]));
                    }
                }
                (requirement.argument.clone(), Value::Array(choices))
            }).collect()).unwrap_or_default();
        let problems: Vec<(String, String)> = descriptor.map(|item| item.parameters.iter().filter_map(|field| {
            let value = node.draft.values.get(&field.name).cloned().unwrap_or(Value::Null);
            fields::project(field, &value, true).err().map(|error| (field.name.clone(), error))
        }).collect()).unwrap_or_default();
        let controls: Vec<Value> = descriptor.map(|item| {
            let mut controls = fields::controls(&item.parameters, &self.draft_values(node), "field", &problems);
            // Shown, not the operator's (v2 project_logic_schema): a derived field, with its
            // description as the reason, and one the chosen device cannot do, with the device's.
            let unavailable = node.answers.get("resolve_field_availability").and_then(|(_, answer)| answer.as_ref().ok()).and_then(Value::as_object);
            for (control, field) in controls.iter_mut().zip(&item.parameters) {
                let reason = unavailable.and_then(|reasons| reasons.get(&field.name)).and_then(Value::as_str).filter(|reason| !reason.trim().is_empty())
                    .map(str::to_owned).or_else(|| field.derived.then(|| field.description.clone().unwrap_or_default()));
                if let Some(reason) = reason { control["unavailable_reason"] = json!(reason); }
            }
            // Options the draft decides (an SLM rearrangement's imaging periods are its pulse's
            // periods): the field is a choice of them; a held value not among them stays, unavailable.
            let offered = node.answers.get("resolve_field_choices").and_then(|(_, answer)| answer.as_ref().ok()).and_then(Value::as_object);
            for (name, choices) in offered.into_iter().flatten() {
                let Some(control) = controls.iter_mut().find(|control| control["name"] == name.as_str()) else { continue };
                let mut choices = choices.as_array().cloned().unwrap_or_default();
                let held = control["value"].clone();
                if !held.is_null() && held != json!("") && !choices.iter().any(|choice| choice[0] == held) {
                    choices.push(json!([held.clone(), format!("{} (unavailable)", held.as_str().map(str::to_owned).unwrap_or_else(|| held.to_string()))]));
                }
                control["kind"] = json!("choice");
                control["choices"] = Value::Array(choices);
            }
            if let Some(control) = item.node().acquisition_input.as_ref()
                .and_then(|name| controls.iter_mut().find(|control| control["name"] == name.as_str())) {
                let terminal = item.node().acquisition_completion.as_deref() == Some("terminal");
                let mut choices = vec![json!(["", "None"])];
                choices.extend(self.nodes.iter().filter(|other| other.id != node.id
                    && self.catalog.get(&other.type_id).is_some_and(|descriptor| descriptor.node().node_kind == "measurement"
                        && (terminal || descriptor.node().reports_ready)))
                    .map(|other| json!([other.id, other.name])));
                let held = control["value"].as_str().unwrap_or("");
                if !held.is_empty() && !choices.iter().any(|choice| choice[0] == held) {
                    let label = self.node(held).map(|node| node.name.as_str()).unwrap_or(held);
                    choices.push(json!([held, format!("{label} (unavailable)")]));
                }
                control["kind"] = json!("choice");
                control["allow_none"] = json!(false); // None is the declared empty string, not a second null option.
                control["choices"] = Value::Array(choices);
            }
            // A path field (v2 FluentPathEdit) names a file of its resource's collection folder, offered
            // as "<label> (*.json)" with Refresh (v2 _project_resource_field), or a folder under the
            // workspace (_project_folder_field); a relative text names a path in `directory`.
            for control in controls.iter_mut().filter(|control| control["kind"] == "path") {
                let label = control["label"].as_str().unwrap_or_default().to_owned();
                match item.node().workspace_resources.iter().find(|spec| control["name"] == spec.field.as_str()) {
                    Some(spec) => {
                        let directory = self.workspace.root.join(&spec.directory);
                        let patterns: Vec<String> = spec.suffixes.iter().map(|suffix| format!("*{suffix}")).collect();
                        control["filter"] = json!(format!("{label} ({})", patterns.join(" ")));
                        control["description"] = json!(format!("Choose a file from {}", directory.display()));
                        control["refreshable"] = json!(true);
                        control["directory"] = json!(directory);
                    }
                    None => {
                        control["description"] = json!(format!("Choose a folder under {}", self.workspace.root.display()));
                        control["directory"] = json!(self.workspace.root);
                    }
                }
            }
            controls
        }).unwrap_or_default();
        let mut row = json!({
            "node_id": node.id, "name": node.name, "type_id": node.type_id,
            "node_kind": descriptor.map(|item| item.node().node_kind.clone()).unwrap_or_default(),
            "label": descriptor.map(TypeDescriptor::label).unwrap_or(&node.type_id),
            "description": descriptor.map(|item| item.description.clone()).unwrap_or_default(),
            "draft_revision": node.revision,
            "values": node.draft.values,
            "source_signal": node.draft.source_signal,
            // The source row's words; none where the draft uses no dataset input (v2 source_required).
            "source_label": descriptor.and_then(|item| source_input(item.node(), &fields::effective(&item.parameters, &node.draft.values))).map(source_words),
            "source_choices": if bundle { self.bundle_choices(node, &sources, signals).into_iter().map(|(key, text)| json!([key, text])).collect::<Vec<_>>() }
                else { sources.iter().map(|name| json!([name, self.signal_label(name)])).collect::<Vec<_>>() },
            // Its Signal picker (v2 _source_choices): the sources under their producers; a bundle is offered flat, one entry per producer.
            "source_tree": (!bundle).then(|| self.source_tree(&sources, &node.draft.source_signal)),
            "device_keys": node.draft.device_keys,
            "device_choices": device_choices,
            "editor_facts_error": editor_facts.and_then(|answer| answer.as_ref().err()),
            "artifact_inputs": node.draft.artifact_inputs,
            "artifact_specs": descriptor.map(|item| item.node().artifact_inputs.iter().map(|spec| json!({"name": spec.name, "label": spec.label,
                "file_filter": spec.file_filter, "required": spec.required.unwrap_or(true), "frame_claims": spec.frame_claims})).collect::<Vec<_>>()).unwrap_or_default(),
            "auto_preview": node.auto_preview,
            "device_labels": descriptor.map(|item| item.node().devices.iter().map(|requirement| (requirement.argument.clone(), json!(words(&requirement.argument))))
                .collect::<Map<String, Value>>()).unwrap_or_default(),
            "outputs": self.node_outputs(node).into_iter().map(|(signal, contract)| json!({"signal": signal, "contract": contract,
                "label": self.signal_label(&signal), "shape": signals.iter().find(|item| item.name == signal).map(|item| item.shape.clone())})).collect::<Vec<_>>(),
            "issues": issues, "can_start": issues.is_empty(), "edited": edited,
            "can_stop": phase != "idle", "preview_offered": !self.node_previews(node).is_empty(),
            "state": state, "status": status, "phase": phase,
            // v2 _producer_projection: the draft fields a selector of its output writes.
            "selector_fields": descriptor.map(|item| {
                let mut names: Vec<&str> = Vec::new();
                for name in item.node().selection_mappings.iter().flat_map(|mapping| &mapping.draft_fields) { if !names.contains(&name.as_str()) { names.push(name); } }
                names
            }).unwrap_or_default(),
        });
        // json! copies what it is given (it serializes a reference): the large parts -- the form's
        // controls, the plugin's editor model and the run -- go in by move and by clone.
        row["controls"] = Value::Array(controls);
        row["editor_facts"] = editor_facts.and_then(|answer| answer.as_ref().ok()).cloned().unwrap_or(Value::Null);
        row["run"] = run.map(Run::row).unwrap_or(Value::Null);
        row
    }

    /// Node types for the Add node chooser.
    pub(crate) fn node_types(&self) -> Vec<Value> {
        self.catalog.nodes().filter(|item| item.addable.unwrap_or(true)).map(|item| {
            let declared = item.node().queries.iter().any(|name| name == "declare_outputs");
            let publishes = if declared { "what its draft declares".to_owned() }
                else if item.node().outputs.is_empty() { "nothing".to_owned() }
                else { item.node().outputs.iter().map(|output| output.name.clone()).collect::<Vec<_>>().join(", ") };
            json!({"type_id": item.type_id, "label": item.label(), "node_kind": item.node().node_kind, "publishes": publishes,
                   "description": item.description})
        }).collect()
    }

    pub(crate) fn list_runs(&self, node: Option<&str>, limit: usize) -> Value {
        Value::Array(self.runs.iter().rev().filter(|run| node.is_none_or(|node| run.node == node)).take(limit).map(Run::row).collect())
    }

    pub(crate) fn get_run(&self, id: &str) -> Result<Value, String> {
        let run = self.run(id).ok_or_else(|| format!("No run {id:?} in this session"))?;
        let mut row = run.row();
        row["start"] = run.start_record.clone().unwrap_or(Value::Null);
        row["record"] = run.directory.as_ref().map(|directory| directory.join("run.json")).filter(|path| path.is_file())
            .and_then(|path| workspace::read_strict(&path, "run record").ok()).unwrap_or(Value::Null);
        Ok(row)
    }

    pub(crate) fn list_plugins(&self, type_id: Option<&str>) -> Value {
        let types: Vec<Value> = self.catalog.types.iter().filter(|item| type_id.is_none_or(|wanted| item.type_id == wanted))
            .map(|item| {
                let mut row = item.summary();
                row["parameters"] = Value::Array(fields::controls(&item.parameters, &fields::defaults(&item.parameters), "field", &[]));
                if let Some(node) = &item.node {
                    row["node"] = json!({"node_kind": node.node_kind, "outputs": node.outputs.iter().map(|output| json!({"name": output.name, "contract": output.contract})).collect::<Vec<_>>(),
                        "inputs": node.inputs.iter().map(|input| json!({"name": input.name, "label": input.label, "contract": input.contract, "delivery": input.delivery})).collect::<Vec<_>>(),
                        "devices": node.devices.iter().map(|requirement| json!({"capability": requirement.capability, "argument": requirement.argument})).collect::<Vec<_>>(),
                        "artifact_inputs": node.artifact_inputs.iter().map(|spec| json!({"name": spec.name, "label": spec.label, "frame_claims": spec.frame_claims})).collect::<Vec<_>>(),
                        "reports_ready": node.reports_ready, "acquisition_input": node.acquisition_input});
                }
                row["commands"] = Value::Array(item.commands.clone());
                row
            }).collect();
        json!({"types": types, "errors": self.catalog.errors.iter().map(|(module, reason)| json!({"module": module, "reason": reason})).collect::<Vec<_>>()})
    }
}

/// The dataset input a draft uses: the first whose enabled_when its values meet (v2 dataset_inputs).
fn source_input<'a>(spec: &'a crate::types::NodeSpec, effective: &Map<String, Value>) -> Option<&'a crate::types::InputSpec> {
    spec.inputs.iter().find(|input| input.enabled_when.as_ref()
        .is_none_or(|(controller, offered)| offered.iter().any(|value| effective.get(controller).unwrap_or(&Value::Null) == value)))
}

/// What the source row and its draft issues call that input (v2 source_label): a bundle input is
/// chosen as a producer's whole bundle; any other reads its name as words ("Frames", "Occupied").
fn source_words(input: &crate::types::InputSpec) -> String {
    if input.select_bundle { "Signal bundle".to_owned() } else { words(&input.name) }
}

/// An identifier as words, first letter capital (`camera_adapter` → `Camera adapter`).
/// v2 `name_words`: a snake_case name read out as words, sentence case, with the
/// lab's acronyms in capitals ("SLM feedback", "RF", "Fit params").
pub(crate) fn words(name: &str) -> String {
    const ACRONYMS: [&str; 9] = ["api", "dac", "daq", "id", "psf", "rf", "roi", "slm", "ttl"];
    let words: Vec<String> = name.replace(['_', '.'], " ").split_whitespace().enumerate().map(|(index, word)| {
        if ACRONYMS.contains(&word.to_lowercase().as_str()) { return word.to_uppercase(); }
        let mut chars = word.chars();
        match (index, chars.next()) { (0, Some(first)) => first.to_uppercase().chain(chars).collect(), _ => word.to_owned() }
    }).collect();
    words.join(" ")
}

fn progress_text(progress: &Value) -> String {
    let message = progress["message"].as_str().unwrap_or("");
    match (progress["current"].as_u64(), progress["total"].as_u64()) {
        (Some(current), Some(total)) => {
            let width = total.to_string().len();
            format!("{message} {current:0width$}/{total}").trim().to_owned()
        }
        (Some(current), None) => format!("{message} {current}").trim().to_owned(),
        _ => if message.is_empty() { "running".to_owned() } else { message.to_owned() },
    }
}
