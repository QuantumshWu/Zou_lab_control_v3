//! The workbench on the window's event loop (WORKBENCH_COMMANDS.md §11): the
//! application side of its `Host` trait, and its `Plugins` over the plugin
//! host (PLUGIN_PROTOCOL): devices and nodes run in plugin processes.
use std::cell::RefCell;
use std::collections::{BTreeMap, VecDeque};
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{Value, json};
use zlc_core::control::Origin;
use zlc_core::runtime::{Generation, GenerationId, Runtime};
use zlc_plugin_host::{HostEvent, Lent, PluginHost, RunRequest};
use zlc_workbench::{Event, Host, PanelContent, Plugins, RunSpec, SignalInfo, TypeCatalog, Workbench};

use crate::panels::Panels;

/// v2 topology.format_signal_shape: "(1) × (3) × (96 × 128)", an empty domain "(1)".
pub fn shape_text(schema: &zlc_core::schema::DatasetSchema) -> String {
    use zlc_core::schema::DomainRole;
    [DomainRole::Repeat, DomainRole::Point, DomainRole::Cell].iter().map(|role| {
        let sizes = schema.domain(*role).axes().iter().map(|axis| axis.len().to_string()).collect::<Vec<_>>();
        format!("({})", if sizes.is_empty() { "1".to_owned() } else { sizes.join(" × ") })
    }).collect::<Vec<_>>().join(" × ")
}

pub fn signals(runtime: &Runtime) -> Vec<SignalInfo> {
    // The Figure Viewer's data is not the apparatus's: no node reads it.
    runtime.signals().into_iter().filter(|source| !crate::archive::viewer_owned(&source.signal)).map(|source| {
        SignalInfo { name: source.signal.to_string(), generation: source.generation.id().as_uuid().to_string(), owner: source.generation.owner().to_owned(),
                     published: source.published, sealed: source.sealed, shape: shape_text(&source.schema) }
    }).collect()
}

/// Every bound panel with its authored target: the accepted row's, or the
/// target a just-added panel waits to be prepared with.
fn panel_contents(bindings: &Panels, accepted: &Value, targets: &BTreeMap<String, Value>) -> Vec<PanelContent> {
    bindings.describe().into_iter().map(|(panel, signal, window, kind)| {
        let row = accepted["panels"].as_array().and_then(|rows| rows.iter().find(|row| row["id"] == panel.as_str()));
        let target = targets.get(&panel).cloned().or_else(|| row.map(|row| row["target"].clone())).unwrap_or(Value::Null);
        let kind = row.and_then(|row| row["kind"].as_str()).map(str::to_owned).unwrap_or(kind);
        // v2 _panel_publisher_fields: its surface's region catalogue, then its accepted fit model's parameters (by symbol).
        let pair = |item: &Value| Some((item[0].as_str()?.to_owned(), item[1].as_str()?.to_owned()));
        let mut outputs: Vec<(String, String)> = row.and_then(|row| row["selection_outputs"].as_array()).into_iter().flatten().filter_map(pair).collect();
        if let Some(fit) = row.map(|row| &row["fit"]).filter(|fit| fit["selected"].as_str().is_some_and(|model| !model.is_empty())) {
            let model = fit["models"].as_array().into_iter().flatten().find(|model| model["key"] == fit["selected"]);
            outputs.extend(model.and_then(|model| model["parameters"].as_array()).into_iter().flatten().filter_map(pair)
                .map(|(name, symbol)| { let label = if symbol.is_empty() { name.clone() } else { symbol }; (name, label) }));
        }
        PanelContent { panel, signal, kind, window: window as u64, target, outputs }
    }).collect()
}

/// The application as the workbench changes it: panels are bound here and
/// prepared by the scientific worker on its next sync.
pub struct DesktopHost<'a> {
    pub runtime: &'a Runtime,
    pub bindings: &'a mut Panels,
    pub accepted: &'a Value,
    pub queue: &'a mut VecDeque<Value>,
    pub targets: &'a mut BTreeMap<String, Value>,
}

/// v2 `add_panel(cell_kind="")`: a new grid's cell is the data's choice ("automatic (from data)")
/// until its Setting names one; a target that names one keeps it.
pub fn automatic_cell(kind: &str, mut target: Value) -> Value {
    if kind != "facet_grid" || target["semantic"].get("cell_kind").is_some() { return target; }
    if !target.is_object() { target = json!({}); }
    if !target["semantic"].is_object() { target["semantic"] = json!({"kind": kind}); }
    target["semantic"]["cell_kind"] = json!("");
    target
}

impl Host for DesktopHost<'_> {
    fn signals(&self) -> Vec<SignalInfo> { signals(self.runtime) }
    fn panels(&self) -> Vec<PanelContent> { panel_contents(self.bindings, self.accepted, self.targets) }
    fn reserve_panel_ids(&mut self, count: usize) -> Vec<String> { self.bindings.reserve_ids(count).unwrap_or_default() }
    fn add_panel(&mut self, panel: &str, signal: &str, kind: &str, window: u64, target: Value) -> Result<String, String> {
        let panel = if panel.is_empty() { self.bindings.add(self.runtime, signal, kind)? } else { self.bindings.add_as(self.runtime, panel, signal, kind)?; panel.to_owned() };
        if window > 1 { self.bindings.configure(self.runtime, &panel, &json!({"window": window}))?; }
        let target = automatic_cell(kind, target);
        if target.is_object() { self.targets.insert(panel.clone(), target); }
        if !self.queue.iter().any(|command| command["action"] == "sync_panels") { self.queue.push_back(json!({"id": 0, "action": "sync_panels"})); }
        Ok(panel)
    }
    fn close_panel(&mut self, panel: &str) {
        // A panel's Edit goes with it.
        for panel in [panel.to_owned(), crate::panels::edit_of(panel)] {
            if self.bindings.remove(&panel) { self.queue.push_back(json!({"id": 0, "action": "close_panel", "panel": panel})); }
            self.targets.remove(&panel);
        }
    }
    fn schemas(&self, signal: &str) -> Option<(Value, Value)> {
        let source = self.runtime.signals().into_iter().find(|source| &*source.signal == signal)?;
        let event = self.runtime.read_latest(&source.generation, signal).ok()
            .and_then(|publication| publication.data().map(|block| zlc_plugin_host::wire::schema_tree(block.schema())));
        Some((zlc_plugin_host::wire::schema_tree(&source.schema), event.unwrap_or(Value::Null)))
    }
}

/// A read-only view for decorating a state message.
struct View { signals: Vec<SignalInfo>, panels: Vec<PanelContent> }

impl Host for View {
    fn signals(&self) -> Vec<SignalInfo> { self.signals.clone() }
    fn panels(&self) -> Vec<PanelContent> { self.panels.clone() }
    fn reserve_panel_ids(&mut self, _: usize) -> Vec<String> { Vec::new() }
    fn add_panel(&mut self, _: &str, _: &str, _: &str, _: u64, _: Value) -> Result<String, String> { Err("read-only view".into()) }
    fn close_panel(&mut self, _: &str) {}
}

/// Put the workbench sections into a state message, and publish the open
/// devices' commands as MCP tools.
pub fn decorate(workbench: &Workbench, runtime: &Runtime, bindings: &Panels, targets: &BTreeMap<String, Value>,
                tools: &crate::remote::Tools, state: &mut Value) {
    let _span = crate::trace::span("workbench.decorate", || json!({}));
    let view = View { signals: signals(runtime), panels: panel_contents(bindings, state, targets) };
    workbench.decorate(state, &view);
    tools.publish(workbench.plugin_tools());
}

/// What the workbench asks of plugins, carried out in the plugin host's
/// processes (line A): every call runs off the window's turn and its answer
/// comes back as a workbench event.
pub struct Stand {
    runtime: Runtime,
    host: Option<PluginHost>,
    /// Why there is no plugin host (no science environment): every plugin family says it.
    missing: Option<String>,
    catalog_ready: bool,
    reply: Arc<dyn Fn(Event) + Send + Sync>,
    /// Instances opened in the host: id -> type.
    instances: BTreeMap<String, String>,
    /// The one map between the workbench's runs and the plugin host's.
    runs: Arc<Mutex<Runs>>,
    /// Host events of a run whose start reply has not arrived yet.
    unmapped: Vec<HostEvent>,
    /// Run directory per workbench run (artifacts must live inside it).
    directories: BTreeMap<String, std::path::PathBuf>,
    /// Operator requests: request id -> host ticket.
    requests: BTreeMap<String, u64>,
    /// A run's `core` request goes on the bus as a remote call does.
    core: Arc<dyn Fn(crate::remote::RemoteCall) -> Result<(), String> + Send + Sync>,
    events: Vec<Event>,
    /// The latest Camera measurement run, the signal it publishes, and its state.
    camera: Option<(String, Arc<str>)>,
    acquisition: Value,
    /// Device types the core implements itself (the pulse board): their
    /// descriptors and where their open/close/calls go.
    native: Option<(Vec<Value>, Rc<dyn Fn(NativeRequest)>)>,
    native_instances: BTreeMap<String, String>,
    /// This computer on the bench fabric: devices it publishes, `remote.device` rows it opens.
    bench: Option<crate::bench::Bench>,
    /// Device types whose scan missed its deadline and is still answering (v2 _scan_pending).
    scanning: Arc<Mutex<std::collections::BTreeSet<String>>>,
    /// Each instance's opens and closes, in the order asked: the next turn to hand out, and the gate.
    lifecycles: BTreeMap<String, (u64, Arc<Gate>)>,
}

/// Workbench runs and plugin-host runs. A run's host id is known only when its start
/// answers; a Stop asked before that is kept here and sent the moment it is known.
#[derive(Default)]
struct Runs {
    /// Host run id -> (host instance, workbench run id).
    mapped: BTreeMap<String, (String, String)>,
    /// Workbench runs whose start has not answered yet: whether a Stop waits for it.
    starting: BTreeMap<String, bool>,
}

impl Runs {
    /// (host run id, host instance) of a workbench run.
    fn hosted(&self, ours: &str) -> Option<(String, String)> {
        self.mapped.iter().find(|(_, (_, run))| run == ours).map(|(hosted, (instance, _))| (hosted.clone(), instance.clone()))
    }
    fn ours(&self, ours: &str) -> bool { self.starting.contains_key(ours) || self.mapped.values().any(|(_, run)| run == ours) }
}

/// One instance's open/close turns: a board replaced by one with the same node closes it and
/// opens it again at once, and the open must not reach the host before the close has.
#[derive(Default)]
struct Gate { done: Mutex<u64>, ready: std::sync::Condvar, current: Arc<std::sync::atomic::AtomicU64> }

impl Gate {
    fn event(&self, turn: u64, event: Event) -> Event {
        Event::Lifecycle { current: self.current.clone(), revision: turn + 1, event: Box::new(event) }
    }
    /// Run `work` once every earlier turn has run (an earlier turn that failed still counts as run).
    fn run<T>(&self, turn: u64, work: impl FnOnce() -> T) -> T {
        struct Next<'a>(&'a Gate, u64);
        impl Drop for Next<'_> {
            fn drop(&mut self) { *self.0.done.lock().unwrap_or_else(|error| error.into_inner()) = self.1 + 1; self.0.ready.notify_all(); }
        }
        drop(self.ready.wait_while(self.done.lock().unwrap_or_else(|error| error.into_inner()), |done| *done < turn).unwrap_or_else(|error| error.into_inner()));
        let _next = Next(self, turn);
        work()
    }
}

/// A request for a native device instance (`pulse_host`), answered later as a workbench event.
pub enum NativeRequest {
    Open { instance: String, type_id: String, parameters: Value, context: Value },
    Close { instance: String },
    Call { ticket: u64, op: String },
    /// Remote on/off for the pulse board (its own server admits the other computers).
    Publish { ticket: u64, instance: String, role: String, published: bool },
    /// What a native row leaves blank, found before it opens (the local board's UART), off the window's turn.
    Resolve { ticket: u64, type_id: String, parameters: Value, excluded: Vec<String> },
}

#[derive(Clone)]
pub struct Builtin(Rc<RefCell<Stand>>);

const CAMERA_MEASUREMENT: &str = "camera_measurement";
const CALL: Duration = Duration::from_secs(120);

/// A plugin host type as the workbench reads descriptors.
pub(crate) fn descriptor(item: &zlc_plugin_host::TypeDescriptor) -> Value {
    let mut node = item.node.clone().unwrap_or(Value::Null);
    if node.is_object() { node["queries"] = json!(item.queries); }
    json!({"type_id": item.type_id, "kind": item.kind, "family": item.family, "label": item.label, "description": item.description,
        "module": item.module, "simulation": item.simulation, "capabilities": item.capabilities, "parameters": item.parameters,
        "physical_outputs": item.physical_outputs.iter().map(|output| json!({"name": output.name})).collect::<Vec<_>>(),
        "commands": item.commands.iter().map(|command| json!({"name": command.name, "summary": command.summary, "read_only": command.read_only,
            "drives": command.drives, "input_schema": command.input_schema, "params": command.params, "file": command.file})).collect::<Vec<_>>(),
        "addable": item.addable, "node": node, "queries": if item.node.is_none() { json!(item.queries) } else { json!([]) },
        "control_window": item.control_window, "identity": item.identity, "resources": item.resources})
}

fn origin(text: &str) -> Origin { if text == "operator" || text.is_empty() { Origin::Operator } else { Origin::Remote(text.to_owned()) } }

fn refused(message: String) -> zlc_plugin_host::CallError { zlc_plugin_host::CallError::new(zlc_plugin_host::ErrorKind::Refused, message) }

/// `run.figure` (PLUGIN_PROTOCOL §5): the run's figure archive, and its image
/// when asked, written inside its run directory; `{archive, image}`.
/// A path a run names, inside its run directory (relative ones are taken from it).
fn run_path(directory: Option<&std::path::PathBuf>, path: &Value) -> Result<std::path::PathBuf, String> {
    let directory = directory.ok_or("This run has no run directory")?;
    let path = std::path::Path::new(path.as_str().ok_or("A path must be text")?);
    let path = if path.is_absolute() { path.to_path_buf() } else { directory.join(path) };
    if !path.starts_with(directory) || path.components().any(|part| matches!(part, std::path::Component::ParentDir)) {
        return Err(format!("{} is outside the run directory", path.display()));
    }
    Ok(path)
}

/// `run.animation`: the GIF of the run's frame images, written inside its run directory; `{path}`.
fn save_animation(params: &Value, directory: Option<std::path::PathBuf>) -> Result<Value, String> {
    let path = run_path(directory.as_ref(), &params["path"])?;
    let images = params["images"].as_array().ok_or("An animation needs its images")?.iter()
        .map(|image| run_path(directory.as_ref(), image)).collect::<Result<Vec<_>, _>>()?;
    let frame_ms = params["frame_ms"].as_u64().unwrap_or(200).min(u64::from(u32::MAX)) as u32;
    let bytes = crate::animation::gif(&images.iter().map(|image| image.as_path()).collect::<Vec<_>>(), frame_ms)?;
    let written = zlc_archive::durable::atomic_write_file(&path, |mut file| {
        std::io::Write::write_all(&mut file, &bytes).map_err(|error| zlc_archive::ArchiveError::io(&error, &path.to_string_lossy()))?;
        Ok(file)
    }).map_err(|error| error.message)?;
    Ok(json!({"path": written.display().to_string()}))
}

fn save_figure(params: &Value, lent: &Lent, directory: Option<std::path::PathBuf>) -> Result<Value, String> {
    use zlc_plot::projection::{Source, SourceData};
    use zlc_plot::session::{real_values, ImagePointOverlay, OverlayArray, PointStatus};
    let path = run_path(directory.as_ref(), &params["path"])?;
    let source = |dataset: &Value| -> Result<Source, String> {
        let dataset = lent.dataset(dataset)?;
        Ok(Source { repeat_counts: None, schema: dataset.block.schema().clone(), data: SourceData::Dense(dataset.block), block_id: dataset.block_id.into(),
                    generation: dataset.generation.into(), revision: dataset.revision, window: None })
    };
    let array = |reference: &Value| -> Result<Option<OverlayArray>, String> {
        if reference.is_null() { return Ok(None); }
        let array = lent.array(reference)?;
        Ok(Some(OverlayArray { values: real_values(&array.values), shape: array.shape }))
    };
    let texts = |value: &Value| value.as_array().map(|items| items.iter().map(|item| item.as_str().map(str::to_owned)).collect::<Vec<_>>());
    let overlay = &params["overlay"];
    let overlay = if overlay.is_null() { None } else {
        let statuses = texts(&overlay["static_statuses"])
            .map(|items| items.into_iter().map(|status| PointStatus::parse(status.as_deref().unwrap_or(""))).collect::<Result<Vec<_>, _>>())
            .transpose().map_err(|error| error.message)?;
        let status = if overlay["status"].is_null() { None } else { Some(Arc::new(source(&overlay["status"])?)) };
        Some(ImagePointOverlay::new(overlay["revision"].as_i64().ok_or("An overlay needs its revision")?.into(),
            array(&overlay["coordinates"])?.ok_or("An overlay needs its coordinates")?,
            texts(&overlay["point_ids"]).map(|ids| ids.into_iter().map(Option::unwrap_or_default).collect()),
            texts(&overlay["labels"]), statuses, status, array(&overlay["paths_xy"])?).map_err(|error| error.message)?)
    };
    let figure = zlc_plot::figure::RunFigure { path, data: source(&params["data"])?, overlay, recipe: params["recipe"].clone(),
        source: params["source"].clone(), lineage: params["lineage"].clone(), image: params["image"].as_bool().unwrap_or(true) };
    let (image, archive) = zlc_plot::figure::save_run_figure(figure, &crate::pulse_host::services()).map_err(|error| error.message)?;
    Ok(json!({"archive": archive.display().to_string(), "image": image.map(|image| image.display().to_string())}))
}

impl Builtin {
    pub fn new(runtime: Runtime, host: Option<PluginHost>, reply: Arc<dyn Fn(Event) + Send + Sync>,
               core: Arc<dyn Fn(crate::remote::RemoteCall) -> Result<(), String> + Send + Sync>) -> Self {
        Self(Rc::new(RefCell::new(Stand { runtime, host, missing: None, catalog_ready: false, reply, instances: BTreeMap::new(), runs: Arc::default(),
            unmapped: Vec::new(), directories: BTreeMap::new(), requests: BTreeMap::new(), core, events: Vec::new(), camera: None,
            acquisition: json!({"state": "idle", "produced": 0}), native: None, native_instances: BTreeMap::new(), bench: None, scanning: Arc::default(), lifecycles: BTreeMap::new() })))
    }
    /// Native device types (their descriptors) and the handler their instances' requests go to.
    pub fn set_native(&self, types: Vec<Value>, handler: Rc<dyn Fn(NativeRequest)>) { self.0.borrow_mut().native = Some((types, handler)); }
    /// This computer on the bench fabric (`remote.device` rows, the Remote switch).
    pub fn set_bench(&self, bench: crate::bench::Bench) { self.0.borrow_mut().bench = Some(bench); }
    pub fn bench(&self) -> Option<crate::bench::Bench> { self.0.borrow().bench.clone() }
    fn native_handler(&self, type_id: Option<&str>, instance: &str) -> Option<Rc<dyn Fn(NativeRequest)>> {
        let stand = self.0.borrow();
        let (types, handler) = stand.native.as_ref()?;
        let native = match type_id {
            Some(type_id) => types.iter().any(|item| item["type_id"] == type_id),
            None => stand.native_instances.contains_key(instance),
        };
        native.then(|| handler.clone())
    }
    /// There is no plugin host, for this reason: the catalogue lists every plugin family as unavailable with it.
    pub fn host_missing(&self, reason: String) { self.0.borrow_mut().missing = Some(reason); }
    /// Discovery finished: the host's catalogue no longer blocks.
    pub fn catalog_ready(&self) { self.0.borrow_mut().catalog_ready = true; }
    pub fn take_events(&self) -> Vec<Event> { std::mem::take(&mut self.0.borrow_mut().events) }

    /// Whether the latest Camera measurement run is still running.
    pub fn active(&self) -> bool {
        let stand = self.0.borrow();
        stand.camera.as_ref().is_some_and(|(run, _)| stand.runs.lock().unwrap().ours(run))
    }

    /// The latest camera run's generation, from the signal it publishes.
    fn camera_generation(stand: &Stand) -> Option<Generation> {
        stand.camera.as_ref().and_then(|(_, signal)| stand.runtime.signals().into_iter().find(|item| item.signal == *signal).map(|item| item.generation))
    }
    pub fn generation(&self) -> Option<GenerationId> { Self::camera_generation(&self.0.borrow()).map(|token| token.id()) }

    /// The camera summary the window's camera bar and `wait {produced}` read:
    /// the latest Camera measurement run, its count read from the Runtime.
    pub fn acquisition(&self) -> Value {
        let stand = self.0.borrow();
        let mut summary = stand.acquisition.clone();
        if let (Some((_, signal)), Some(generation)) = (&stand.camera, Self::camera_generation(&stand)) {
            if let Ok(latest) = stand.runtime.read_latest(&generation, signal) {
                let count = latest.key().ordinal.map(|ordinal| ordinal + 1)
                    .or_else(|| latest.canonical().map(|snapshot| snapshot.segment_count() as u64));
                if let Some(count) = count { summary["produced"] = json!(count); }
            }
        }
        summary
    }

    fn camera_fact(&self, run: &str, state: &str, error: Option<&Value>) {
        let mut stand = self.0.borrow_mut();
        if stand.camera.as_ref().is_none_or(|(camera, _)| camera != run) { return; }
        if stand.acquisition["state"] != "stopping" || state != "running" { stand.acquisition["state"] = json!(state); }
        if let Some(error) = error { stand.acquisition["error"] = error["message"].clone(); }
    }

    /// Ask every plugin run to stop, one still starting as soon as it has (closing the window).
    pub fn stop_all(&self) {
        let runs: Vec<(String, String)> = {
            let stand = self.0.borrow();
            let mut runs = stand.runs.lock().unwrap();
            for asked in runs.starting.values_mut() { *asked = true; }
            runs.mapped.iter().map(|(hosted, (instance, _))| (hosted.clone(), instance.clone())).collect()
        };
        for (hosted, instance) in runs { self.stop_hosted(hosted, instance); }
    }

    fn stop_hosted(&self, hosted: String, instance: String) {
        let _ = self.spawn("stop", move |host| Event::Log { line: match host.stop_run(&instance, &hosted) {
            Ok(()) => format!("stop requested for {hosted}"), Err(error) => format!("stop of {hosted}: {error}") }, instance });
    }

    /// A plugin host fact, as workbench events. A run's fact that arrives
    /// before its start reply waits for it (`replay`).
    pub fn host_event(&self, event: HostEvent) -> Vec<Event> {
        // A draft query's catalogue read (an estimate's pulse_estimate): no run to map.
        let event = match event {
            HostEvent::Request { run, ticket, op, params, .. } if run.is_empty() && op == "core" => { self.draft_core(ticket, &params); return Vec::new(); }
            event => event,
        };
        let hosted = match &event {
            HostEvent::RunReady { run, .. } | HostEvent::RunProgress { run, .. } | HostEvent::RunFinished { run, .. } | HostEvent::Request { run, .. } => Some(run.clone()),
            _ => None,
        };
        let ours = hosted.as_ref().map(|run| self.0.borrow().runs.lock().unwrap().mapped.get(run).map(|(_, ours)| ours.clone()));
        let Some(ours) = ours.map_or(Some(String::new()), |ours| ours) else {
            let mut stand = self.0.borrow_mut();
            if stand.unmapped.len() >= 1024 { stand.unmapped.remove(0); }
            stand.unmapped.push(event);
            return Vec::new();
        };
        match event {
            HostEvent::Failed { instances, reason } => instances.into_iter().map(|instance| Event::Failed { instance, error: reason.clone() }).collect(),
            HostEvent::RunReady { .. } => { self.camera_fact(&ours, "running", None); vec![Event::RunReady { run: ours }] }
            HostEvent::RunProgress { message, current, total, .. } => vec![Event::RunProgress { run: ours, message, current, total }],
            HostEvent::RunFinished { run, outcome, error, generation, .. } => {
                self.camera_fact(&ours, if outcome == "failed" { "error" } else { "stopped" }, error.as_ref());
                let mut stand = self.0.borrow_mut();
                stand.runs.lock().unwrap().mapped.remove(&run);
                stand.directories.remove(&ours);
                stand.requests.retain(|request, _| !request.starts_with(&format!("{ours}:")));
                vec![Event::RunFinished { run: ours, outcome, error, generation }]
            }
            HostEvent::Request { ticket, op, params, origin, lent, .. } => self.request(ours, ticket, &op, params, origin, lent),
            // A device's own facts are its window's (main.rs gives them to slm_host).
            HostEvent::CatalogChanged | HostEvent::Notice { .. } => Vec::new(),
            HostEvent::DeviceLog { instance, line } => vec![Event::Log { instance, line }],
        }
    }

    /// Facts held back until their run's start reply: now mappable.
    pub fn replay(&self) -> Vec<Event> {
        let held = std::mem::take(&mut self.0.borrow_mut().unmapped);
        held.into_iter().flat_map(|event| self.host_event(event)).collect()
    }

    fn respond(&self, ticket: u64, result: Result<Value, String>) {
        if let Some(host) = &self.0.borrow().host { host.respond(ticket, result.map_err(refused)); }
    }

    /// A draft query asks one catalogue command: answered only when it is a read
    /// (a read-only command), as any client's would be; nothing a query asks can act.
    fn draft_core(&self, ticket: u64, params: &Value) {
        let name = params["command"].as_str().unwrap_or("").to_owned();
        if zlc_bus::catalog::find(&name).is_none_or(|spec| !spec.read_only) {
            return self.respond(ticket, Err(format!("a draft query may only read; {name} is not a read")));
        }
        let (reply, answer) = std::sync::mpsc::channel();
        let arguments = params.get("params").cloned().filter(Value::is_object).unwrap_or_else(|| json!({}));
        let call = crate::remote::RemoteCall { origin: Origin::Operator, name, params: arguments, reply };
        let host = self.0.borrow().host.clone();
        let sent = (self.0.borrow().core)(call);
        match (sent, host) {
            (Ok(()), Some(host)) => {
                let _ = std::thread::Builder::new().name("zlc-plugin-draft-core".into()).spawn(move || {
                    let result = answer.recv_timeout(Duration::from_secs(120)).unwrap_or_else(|_| Err("the application did not answer".into()));
                    host.respond(ticket, result.map(|output| output.json).map_err(refused));
                });
            }
            (Err(error), _) => self.respond(ticket, Err(error)),
            (Ok(()), None) => {}
        }
    }

    /// What a run asks of the application (PLUGIN_PROTOCOL §5).
    fn request(&self, run: String, ticket: u64, op: &str, params: Value, origin: Origin, lent: Lent) -> Vec<Event> {
        match op {
            "run.figure" | "run.animation" => {
                // Drawn off the window's turn; the run's thread waits for the paths.
                let (directory, host) = { let stand = self.0.borrow(); (stand.directories.get(&run).cloned(), stand.host.clone()) };
                let Some(host) = host else { return Vec::new() };
                let figure = op == "run.figure";
                let spawned = std::thread::Builder::new().name("zlc-plugin-figure".into()).spawn(move || {
                    let written = if figure { save_figure(&params, &lent, directory) } else { save_animation(&params, directory) };
                    host.respond(ticket, written.map_err(refused));
                });
                if let Err(error) = spawned { self.respond(ticket, Err(error.to_string())); }
                Vec::new()
            }
            "run.operator_input" => {
                let request = format!("{run}:{ticket}");
                self.0.borrow_mut().requests.insert(request.clone(), ticket);
                vec![Event::RunRequest { run, request: json!({"request": request, "kind": params["kind"], "title": params["title"],
                    "message": params["message"], "payload": params["payload"]}) }]
            }
            "run.artifact" => {
                // Registered only once it exists, complete, inside the run's directory.
                let directory = self.0.borrow().directories.get(&run).cloned();
                let registered = (|| -> Result<Value, String> {
                    let directory = directory.ok_or("This run has no run directory")?;
                    let path = std::path::Path::new(params["path"].as_str().ok_or("An artifact needs a path")?);
                    let path = if path.is_absolute() { path.to_path_buf() } else { directory.join(path) };
                    let relative = path.strip_prefix(&directory).map_err(|_| format!("{} is outside the run directory", path.display()))?;
                    if matches!(relative.to_str(), Some("start.json" | "run.json")) { return Err("start.json and run.json are the run's own records".into()); }
                    let size = std::fs::metadata(&path).map_err(|error| format!("{}: {error}", path.display()))?.len();
                    Ok(json!({"name": params["name"], "path": relative.to_string_lossy().replace('\\', "/"), "role": params["role"],
                              "contract_id": params["contract_id"], "size_bytes": size}))
                })();
                self.respond(ticket, registered.clone());
                registered.map(|artifact| vec![Event::RunArtifact { run, artifact }]).unwrap_or_default()
            }
            "run.acquire" => vec![Event::RunAcquire { run, ticket, node: params["node"].as_str().unwrap_or("").to_owned(),
                completion: params["completion"].as_str().unwrap_or("ready").to_owned() }],
            "core" => {
                // The same command a client would send, under the run's origin.
                let (reply, answer) = std::sync::mpsc::channel();
                // `_run` names the asking run, so a command can hold it to that run's claims.
                let mut arguments = params.get("params").cloned().filter(Value::is_object).unwrap_or_else(|| json!({}));
                arguments["_run"] = json!(run);
                let call = crate::remote::RemoteCall { origin, name: params["command"].as_str().unwrap_or("").to_owned(), params: arguments, reply };
                let host = self.0.borrow().host.clone();
                let sent = (self.0.borrow().core)(call);
                match (sent, host) {
                    (Ok(()), Some(host)) => {
                        let _ = std::thread::Builder::new().name("zlc-plugin-core".into()).spawn(move || {
                            let result = answer.recv_timeout(Duration::from_secs(900)).unwrap_or_else(|_| Err("the application did not answer".into()));
                            host.respond(ticket, result.map(|output| output.json).map_err(refused));
                        });
                    }
                    (Err(error), _) => self.respond(ticket, Err(error)),
                    (Ok(()), None) => {}
                }
                Vec::new()
            }
            other => { self.respond(ticket, Err(format!("the application has no request {other}"))); Vec::new() }
        }
    }

    /// Do `work` with the host off the window's turn; its answer is an event.
    /// The next open/close turn of an instance.
    /// A client of a type another computer announces (its own server: the pulse board, the SLM
    /// head) is watched on the bench fabric: the link that decides when a remote.device's
    /// computer is gone decides it for this one too, and it is lost with that reason -- a plugin
    /// device as a remote.device is (`PluginHost::lost`), the native board through the workbench.
    fn watch(&self, instance: &str, type_id: &str, parameters: &Value) {
        let (bench, host, reply, announced) = {
            let stand = self.0.borrow();
            let native = stand.native.as_ref().is_some_and(|(types, _)| types.iter().any(|item| item["announces"] == type_id));
            let plugin = || stand.host.as_ref().and_then(|host| host.catalog().ok()).is_some_and(|catalog| catalog.types.iter().any(|item| item.announces == type_id));
            (stand.bench.clone(), stand.host.clone(), stand.reply.clone(), native || plugin())
        };
        let Some(bench) = bench.filter(|_| announced) else { return };
        let (name, kind, parameters) = (instance.to_owned(), type_id.to_owned(), parameters.clone());
        let registry = host.as_ref().map(|host| host.registry().clone()).unwrap_or_else(|| zlc_plugin_host::Registry::standalone("fabric-watch"));
        let _ = std::thread::Builder::new().name("zlc-fabric-watch".into()).spawn(move || {
            let lost = name.clone();
            bench.watch(registry, &name, &kind, &parameters, move |reason| match host.as_ref().filter(|host| host.descriptor(&lost).is_some()) {
                Some(host) => host.lost(&lost, reason),
                None => reply(Event::Failed { instance: lost.clone(), error: reason.to_owned() }),
            });
        });
    }

    fn turn(&self, instance: &str) -> (u64, Arc<Gate>) {
        let mut stand = self.0.borrow_mut();
        let entry = stand.lifecycles.entry(instance.to_owned()).or_insert_with(|| (0, Arc::new(Gate::default())));
        entry.0 += 1;
        entry.1.current.store(entry.0, std::sync::atomic::Ordering::Release);
        (entry.0 - 1, entry.1.clone())
    }

    fn spawn(&self, name: &str, work: impl FnOnce(&PluginHost) -> Event + Send + 'static) -> Result<(), String> {
        let stand = self.0.borrow();
        let host = stand.host.clone().ok_or_else(|| stand.missing.clone().unwrap_or_else(|| zlc_plugin_host::NOT_INSTALLED.to_owned()))?;
        let reply = stand.reply.clone();
        std::thread::Builder::new().name(format!("zlc-plugin-{name}")).spawn(move || reply(work(&host))).map(|_| ()).map_err(|error| error.to_string())
    }
}

/// The device families zlc_atom ships (its `devices/<family>` packages), read from the checkout
/// without an interpreter.
fn plugin_families() -> Vec<String> {
    let mut families: Vec<String> = std::fs::read_dir(zlc_core::checkout::path("science/src/zlc_atom/devices")).into_iter().flatten().flatten()
        .filter(|entry| entry.path().join("__init__.py").is_file()).filter_map(|entry| entry.file_name().into_string().ok()).collect();
    families.sort();
    families
}

impl Plugins for Builtin {
    fn catalog(&mut self) -> TypeCatalog {
        let stand = self.0.borrow();
        let (mut types, mut errors, mut templates) = (Vec::new(), Vec::new(), serde_json::Map::new());
        if let Some((native, _)) = &stand.native { types.extend(native.iter().cloned()); }
        if stand.bench.is_some() { types.push(zlc_fabric::remote_type()); }
        if let Some(host) = stand.host.as_ref().filter(|_| stand.catalog_ready) {
            match host.catalog() {
                Ok(catalog) => {
                    types.extend(catalog.types.iter().map(descriptor));
                    errors.extend(catalog.errors.iter().map(|error| json!({"module": error.module, "error": error.error})));
                    templates = catalog.templates.clone();
                }
                Err(error) => errors.push(json!({"module": "zlc_plugin describe", "error": error})),
            }
        }
        // No interpreter to ask: each plugin family (a folder under devices/) is unavailable for that reason.
        if let (None, Some(reason)) = (&stand.host, &stand.missing) {
            let families = plugin_families();
            if families.is_empty() { errors.push(json!({"module": "zlc_atom", "error": reason})); }
            errors.extend(families.into_iter().map(|family| json!({"module": format!("zlc_atom.devices.{family}"), "error": reason})));
        }
        TypeCatalog::from_describe(&json!({"protocol": [2, 0], "types": types, "errors": errors, "templates": templates})).unwrap_or_else(|error| {
            eprintln!("{}", json!({"type": "plugin-error", "error": error}));
            TypeCatalog::default()
        })
    }
    fn open(&mut self, instance: &str, type_id: &str, parameters: &Value, context: &Value) {
        self.watch(instance, type_id, parameters);
        if let Some(handler) = self.native_handler(Some(type_id), instance) {
            self.0.borrow_mut().native_instances.insert(instance.to_owned(), type_id.to_owned());
            handler(NativeRequest::Open { instance: instance.to_owned(), type_id: type_id.to_owned(), parameters: parameters.clone(), context: context.clone() });
            return;
        }
        self.0.borrow_mut().instances.insert(instance.to_owned(), type_id.to_owned());
        let (name, kind, parameters, group) = (instance.to_owned(), type_id.to_owned(), parameters.clone(), context["simulation"].clone());
        let group = if group.is_object() { group } else { json!({}) };
        let bench = self.bench().filter(|_| type_id == zlc_fabric::REMOTE_TYPE);
        let (turn, gate) = self.turn(instance);
        let started = self.spawn("open", move |host| gate.event(turn, gate.run(turn, || match match &bench { Some(bench) => bench.open(host, &name, &parameters), None => host.open(&name, &kind, parameters, group) } {
            Ok(point) => Event::Opened { instance: name, working_point: point },
            Err(error) => Event::Failed { instance: name, error: error.to_string() },
        })));
        if let Err(error) = started { self.0.borrow_mut().events.push(Event::Failed { instance: instance.into(), error }); }
    }
    fn close(&mut self, instance: &str) {
        if let Some(bench) = self.bench() { bench.unwatch(instance); }
        if let Some(handler) = self.native_handler(None, instance) {
            // The board this computer announces is taken back first, as a plugin device is (the computers watching it lose it).
            if let Some(bench) = self.bench() { bench.withdraw_if_published(instance); }
            self.0.borrow_mut().native_instances.remove(instance);
            handler(NativeRequest::Close { instance: instance.to_owned() });
            return;
        }
        // A previous Close may have failed after it left the requested set.
        // Ask the actual host again; its Close is idempotent when already gone.
        self.0.borrow_mut().instances.remove(instance);
        if self.0.borrow().host.is_none() {
            self.0.borrow_mut().events.push(Event::Closed { instance: instance.into(), error: None });
            return;
        }
        let (name, bench) = (instance.to_owned(), self.bench());
        let (turn, gate) = self.turn(instance);
        let started = self.spawn("close", move |host| gate.event(turn, gate.run(turn, || {
            // A published device is taken back from the bench first (Shut down withdraws, as v2's did).
            if let Some(bench) = &bench { bench.withdraw_if_published(&name); }
            Event::Closed { error: host.close(&name).err().map(|error| error.to_string()), instance: name }
        })));
        if let Err(error) = started { self.0.borrow_mut().events.push(Event::Closed { instance: instance.into(), error: Some(error) }); }
    }
    fn query(&mut self, instance: &str, ticket: u64, name: &str, args: &Value) {
        let mut params = args.clone();
        params["name"] = json!(name);
        // A native device (the pulse board) is no plugin's to read: the query sees only plugin devices.
        if let Some(devices) = params["devices"].as_object_mut() {
            let stand = self.0.borrow();
            devices.retain(|_, info| !info["instance"].as_str().is_some_and(|id| stand.native_instances.contains_key(id)));
        }
        let instance = instance.to_owned();
        let started = self.spawn("query", move |host| Event::Answered { ticket,
            result: host.call(&instance, "query", params, CALL).map(|reply| reply.value).map_err(|error| error.to_string()) });
        if let Err(error) = started { self.0.borrow_mut().events.push(Event::Answered { ticket, result: Err(error) }); }
    }
    fn call(&mut self, instance: &str, ticket: u64, op: &str, params: &Value) {
        if let Some(handler) = self.native_handler(None, instance) {
            handler(NativeRequest::Call { ticket, op: op.to_owned() });
            return;
        }
        let (instance, op, params) = (instance.to_owned(), op.to_owned(), params.clone());
        // The bench a Start records waits for no device long: one busy in another run's call misses
        // the bound without being taken for unresponsive.
        let started = self.spawn("call", move |host| Event::Answered { ticket,
            result: if op == "snapshot" { host.snapshot(&instance, zlc_workbench::SNAPSHOT_WAIT) } else { host.call(&instance, &op, params, CALL).map(|reply| reply.value) }
                .map_err(|error| error.to_string()) });
        if let Err(error) = started { self.0.borrow_mut().events.push(Event::Answered { ticket, result: Err(error) }); }
    }
    fn start(&mut self, run: &RunSpec) {
        self.rename(&run.instance, &run.node_name);
        let runs = self.0.borrow().runs.clone();
        runs.lock().unwrap().starting.insert(run.run.clone(), false);
        {
            let mut stand = self.0.borrow_mut();
            if let Some(directory) = &run.directory { stand.directories.insert(run.run.clone(), directory.clone()); }
            if let (CAMERA_MEASUREMENT, Some(signal)) = (run.type_id.as_str(), run.outputs.get("frames")) {
                stand.camera = Some((run.run.clone(), signal.as_str().into()));
                stand.acquisition = json!({"state": "starting", "produced": 0});
            }
        }
        let (node, ours) = (run.instance.clone(), run.run.clone());
        // A native device (the pulse board) stays claimed here, in the workbench; the
        // plugin reaches it only through catalogue commands (pulse_run, ...), so the
        // plugin host is handed only the plugin devices -- and of a native one only its
        // key, `<argument>_key` as every claimed device's (a run record names the instance).
        let (devices, native): (Vec<_>, Vec<_>) = { let stand = self.0.borrow(); run.devices.iter()
            .map(|(argument, instance)| (argument.clone(), instance.clone())).partition(|(_, instance)| !stand.native_instances.contains_key(instance)) };
        let keys = native.into_iter().map(|(argument, instance)| (format!("{argument}_key"), json!(instance)));
        let request = RunRequest { prepared: run.prepared.clone(), values: Value::Object(run.values.clone()), devices: devices.into_iter().collect(), device_labels: run.device_labels.clone(), shared: run.shared.clone(), origin: origin(&run.origin),
            // Build arguments beside the values: chosen artifacts and resolved workspace resources.
            resources: Value::Object(run.artifacts.iter().chain(&run.resources).map(|(key, value)| (key.clone(), value.clone())).chain(keys).collect()), source_signal: (!run.source_signal.is_empty()).then(|| run.source_signal.clone()),
            run_directory: run.directory.clone(), outputs: run.outputs.clone(), apparatus: run.apparatus.clone() };
        let prepared = run.prepared.clone();
        let started = self.spawn("start", move |host| match host.start_run(&node, request) {
            Ok(hosted) => {
                // Mapped, and a Stop asked while it was starting goes now: no Stop is lost.
                let stop = { let mut runs = runs.lock().unwrap(); runs.mapped.insert(hosted.clone(), (node.clone(), ours.clone())); runs.starting.remove(&ours).unwrap_or(false) };
                if stop { let _ = host.stop_run(&node, &hosted); }
                Event::RunStarted { run: ours }
            }
            Err(error) => {
                if let Some(token) = prepared { let _ = host.call(&node, "run.discard", json!({"prepared": token}), CALL); }
                runs.lock().unwrap().starting.remove(&ours);
                Event::RunFinished { run: ours, outcome: "failed".into(), error: Some(json!({"type": error.kind.name(), "message": error.message})), generation: None }
            }
        });
        if let Err(error) = started {
            self.0.borrow().runs.lock().unwrap().starting.remove(&run.run);
            self.0.borrow_mut().events.push(Event::RunFinished { run: run.run.clone(), outcome: "failed".into(),
                error: Some(json!({"type": "StartError", "message": error})), generation: None });
        }
    }
    fn rename(&mut self, instance: &str, name: &str) {
        if let Some(host) = &self.0.borrow().host { host.set_node_name(instance, name); }
    }
    fn follow(&mut self, instance: &str, signal: Option<&str>) {
        // At once, not on a lane: it must be in place before the source's next run.begin.
        if let Some(host) = &self.0.borrow().host { host.follow(instance, signal); }
    }
    fn stop(&mut self, run: &str) {
        if self.0.borrow().camera.as_ref().is_some_and(|(camera, _)| camera == run) { self.0.borrow_mut().acquisition["state"] = json!("stopping"); }
        let hosted = {
            let stand = self.0.borrow();
            let mut runs = stand.runs.lock().unwrap();
            let hosted = runs.hosted(run);
            // Still starting: its start sends the Stop once the host has named it.
            if hosted.is_none() { if let Some(asked) = runs.starting.get_mut(run) { *asked = true; } }
            hosted
        };
        if let Some((hosted, instance)) = hosted { self.stop_hosted(hosted, instance); }
    }
    fn answer(&mut self, _run: &str, request: &str, response: Option<&Value>) {
        let ticket = self.0.borrow_mut().requests.remove(request);
        // A cancel is the run's Stop (v2: operator input was cancelled): the plugin ends stopped.
        let result = response.cloned().ok_or_else(|| zlc_plugin_host::CallError::new(zlc_plugin_host::ErrorKind::Cancelled, "operator input was cancelled"));
        if let (Some(ticket), Some(host)) = (ticket, &self.0.borrow().host) { host.respond(ticket, result); }
    }
    fn acquired(&mut self, _run: &str, ticket: u64, result: Result<Value, String>) { self.respond(ticket, result); }
    fn discover(&mut self, ticket: u64, type_id: &str, excluded: &[String]) {
        let (type_id, excluded) = (type_id.to_owned(), excluded.to_vec());
        // The bench part of the scan: what other computers publish (v2's remote family).
        let bench = self.bench().filter(|_| type_id == zlc_fabric::REMOTE_TYPE);
        // v2 _scan_families: one deadline from the scan's start, the process's start inside it. A family
        // that misses it is reported so, its answer when it comes is dropped, and until then it is still
        // scanning (Init, Apply and another scan wait for it).
        let until = std::time::Instant::now() + SCAN;
        let scanning = self.0.borrow().scanning.clone();
        let started = self.spawn("discover", move |host| {
            let (sender, answer) = std::sync::mpsc::channel();
            let (host, family, ended) = (host.clone(), type_id.clone(), scanning.clone());
            let asked = std::thread::Builder::new().name(format!("zlc-scan-{type_id}")).spawn(move || {
                let _ = sender.send(match &bench { Some(bench) => Ok(bench.discover(&host)), None => host.discover(&family, &excluded, LATE_SCAN).map_err(|error| error.message) });
                ended.lock().unwrap().remove(&family);
            });
            if let Err(error) = asked { return Event::Answered { ticket, result: Err(error.to_string()) }; }
            let result = match answer.recv_timeout(until.saturating_duration_since(std::time::Instant::now())) {
                Ok(result) => result,
                Err(_) => {
                    let mut late = scanning.lock().unwrap();
                    late.insert(type_id.clone());
                    // An answer that came while the deadline passed is as late as one after it.
                    if answer.try_recv().is_ok() { late.remove(&type_id); }
                    Err(format!("no answer within {}s", SCAN.as_secs()))
                }
            };
            Event::Answered { ticket, result }
        });
        if let Err(error) = started { self.0.borrow_mut().events.push(Event::Answered { ticket, result: Err(error) }); }
    }
    fn resolve(&mut self, ticket: u64, instance: &str, type_id: &str, parameters: &Value, excluded: &[String]) {
        if let Some(handler) = self.native_handler(Some(type_id), instance) {
            handler(NativeRequest::Resolve { ticket, type_id: type_id.to_owned(), parameters: parameters.clone(), excluded: excluded.to_vec() });
            return;
        }
        let (type_id, parameters, excluded) = (type_id.to_owned(), parameters.clone(), excluded.to_vec());
        // A resolve probes as a scan does, port by port with a bounded wait each, so it is held to a
        // family's scan deadline (the process's start inside it); Init waits for nothing longer.
        let started = self.spawn("resolve", move |host| Event::Answered { ticket,
            result: host.resolve(&type_id, parameters, &excluded, SCAN).map_err(|error| error.to_string()) });
        if let Err(error) = started { self.0.borrow_mut().events.push(Event::Answered { ticket, result: Err(error) }); }
    }
    fn scanning(&self) -> Vec<String> { self.0.borrow().scanning.lock().unwrap().iter().cloned().collect() }
    fn discovering(&self) -> bool { let stand = self.0.borrow(); stand.host.is_some() && !stand.catalog_ready }
    fn publish(&mut self, ticket: u64, instance: &str, role: &str, published: bool) {
        let (instance, role) = (instance.to_owned(), role.to_owned());
        let Some(bench) = self.bench() else {
            self.0.borrow_mut().events.push(Event::Answered { ticket, result: Err("This computer is not on the bench fabric".into()) });
            return;
        };
        if let Some(handler) = self.native_handler(None, &instance) {
            // The pulse board has a server of its own: the fabric only announces it.
            handler(NativeRequest::Publish { ticket, instance, role, published });
            return;
        }
        let started = self.spawn("publish", move |host| Event::Answered { ticket, result: bench.publish(host, &instance, &role, published) });
        if let Err(error) = started { self.0.borrow_mut().events.push(Event::Answered { ticket, result: Err(error) }); }
    }
}

/// How long one device family's hardware scan may take (v2 _FAMILY_SCAN_DEADLINE_SECONDS).
const SCAN: Duration = Duration::from_secs(20);
/// How long a family past the deadline is waited for (v2 waited on its thread for ever).
const LATE_SCAN: Duration = Duration::from_secs(600);

/// Apply a workbench command and deliver the answers it produced at once.
pub fn run(workbench: &mut Workbench, builtin: &Builtin, name: &str, params: &Value, origin: &Origin, request: u64, host: &mut dyn Host) {
    workbench.apply(name, params, origin, request, host);
    deliver(workbench, builtin, host);
}

pub fn deliver(workbench: &mut Workbench, builtin: &Builtin, host: &mut dyn Host) {
    loop {
        let events = builtin.take_events();
        if events.is_empty() { break; }
        for event in events { workbench.on_event(event, host); }
    }
}

/// `start_camera`, kept for the window's camera bar and MCP recipes: the
/// Virtual MOT camera as an apparatus device and a Camera measurement node.
pub fn start_camera(workbench: &mut Workbench, builtin: &Builtin, command: &Value, origin: &Origin, host: &mut dyn Host) -> Result<(), String> {
    let mut step = |workbench: &mut Workbench, name: &str, params: Value| -> Result<Value, String> {
        run(workbench, builtin, name, &params, origin, 0, host);
        workbench.take_replies().into_iter().map(|(_, result)| result).last().unwrap_or(Ok(Value::Null))
    };
    let roi = command["roi_xywh"].as_array();
    let mut settings = json!({"exposure_seconds": command["exposure_seconds"]});
    if let Some(shape) = command.get("frame_shape_yx").filter(|shape| !shape.is_null()) { settings["frame_shape_yx"] = shape.clone(); }
    for (index, name) in ["roi_x", "roi_y", "roi_width", "roi_height"].into_iter().enumerate() {
        settings[name] = roi.and_then(|roi| roi.get(index)).cloned().unwrap_or(Value::Null);
    }
    let device = workbench.instance_of("camera.virtual_mot");
    let device = match device {
        Some(device) => { step(workbench, "edit_device", json!({"device": device, "parameters": settings}))?; device }
        None => step(workbench, "add_device", json!({"type_id": "camera.virtual_mot", "instance_id": "mot_camera", "parameters": settings}))?["device"]
            .as_str().unwrap_or("mot_camera").to_owned(),
    };
    // The simulated world's seed makes the frames reproducible.
    if let Some(seed) = command["seed"].as_u64() { step(workbench, "edit_simulation", json!({"values": {"seed": seed}}))?; }
    step(workbench, "apply_devices", json!({}))?;
    let values = json!({"repeat": command["frames"].as_u64().unwrap_or(0), "exposure_seconds": null});
    let node = match workbench.node_of("camera_measurement") {
        Some(node) => { step(workbench, "edit_node", json!({"node": node, "values": values, "device_keys": {"camera": device}}))?; node }
        None => step(workbench, "add_node", json!({"type_id": "camera_measurement", "values": values, "device_keys": {"camera": device}}))?["node"]
            .as_str().unwrap_or("camera_measurement").to_owned(),
    };
    step(workbench, "start_node", json!({"node": node})).map(|_| ())
}

pub fn stop_camera(workbench: &mut Workbench, builtin: &Builtin, origin: &Origin, host: &mut dyn Host) -> Result<(), String> {
    let node = workbench.node_of("camera_measurement").ok_or("Camera acquisition is not active")?;
    run(workbench, builtin, "stop_node", &json!({"node": node}), origin, 0, host);
    workbench.take_replies().into_iter().map(|(_, result)| result).last().unwrap_or(Ok(Value::Null)).map(|_| ())
}
