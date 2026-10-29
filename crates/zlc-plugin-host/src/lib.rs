//! The Rust side of the plugin protocol (docs/PLUGIN_PROTOCOL.md).
//!
//! `PluginHost` runs every plugin instance -- devices and logic nodes alike,
//! each on its own lane -- in one `python -m zlc_plugin host` process, as v2
//! ran them in one; a type that declares why it cannot share it (`isolated`)
//! gets one of its own. It routes every request between the core, devices and
//! nodes, owns the block registry, publishes node runs into the Runtime and
//! registers physical outputs for Stop all. The core holds no interpreter: an
//! instance whose call hangs is a failed instance, a process that stops
//! answering is failed instances, and each can be opened again.
pub mod blocks;
mod cache;
pub mod descriptor;
mod files;
mod frame;
mod process;
pub mod schema;
pub mod wire;

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;

use serde_json::{Value, json};
use zlc_core::control::{Origin, SafeActions, public_instance};
use zlc_core::runtime::{Bundle, Declaration, DeliveryKind, DeliveryPolicy, Generation, GenerationId, HistoryLease, Input, Read, ReadKind,
    ReadMode, ReadRequest, RunRecord, Runtime, RuntimeError, Start, Storage, Subscription};
use zlc_core::schema::{DatasetSchema, DomainRole};

pub use blocks::{Lend, Registry};
pub use descriptor::{Catalog, CommandDescriptor, FileParam, TypeDescriptor};
use process::{Incoming, Job, PluginProcess};

pub const PROTOCOL: [u32; 2] = [2, 0];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorKind { Refused, Invalid, Failed, Timeout, Lost, Cancelled, StaleRun, UnknownOp, NotClaimed }

impl ErrorKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Refused => "refused", Self::Invalid => "invalid", Self::Failed => "failed", Self::Timeout => "timeout",
            Self::Lost => "lost", Self::Cancelled => "cancelled", Self::StaleRun => "stale_run", Self::UnknownOp => "unknown_op",
            Self::NotClaimed => "not_claimed",
        }
    }
    pub fn parse(text: &str) -> Self {
        match text {
            "refused" => Self::Refused, "invalid" => Self::Invalid, "timeout" => Self::Timeout, "lost" => Self::Lost,
            "cancelled" => Self::Cancelled, "stale_run" => Self::StaleRun, "unknown_op" => Self::UnknownOp,
            "not_claimed" => Self::NotClaimed, _ => Self::Failed,
        }
    }
}

/// A failed call: its kind says what is known about the device (§7).
#[derive(Clone, Debug)]
pub struct CallError { pub kind: ErrorKind, pub message: String, pub remote_type: String, pub traceback: String }

impl CallError {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self { kind, message: message.into(), remote_type: String::new(), traceback: String::new() }
    }
    pub fn lost(message: impl Into<String>) -> Self { Self::new(ErrorKind::Lost, message) }
    pub fn from_wire(error: &Value) -> Self {
        Self { kind: ErrorKind::parse(error["kind"].as_str().unwrap_or("failed")),
               message: error["message"].as_str().unwrap_or("").to_owned(),
               remote_type: error["type"].as_str().unwrap_or("").to_owned(),
               traceback: error["traceback"].as_str().unwrap_or("").to_owned() }
    }
    pub fn wire(&self) -> Value {
        json!({"kind": self.kind.name(), "type": self.remote_type, "message": self.message, "traceback": self.traceback})
    }
    /// The plugin raised this Python exception type (e.g. "builtins:TimeoutError").
    pub fn is(&self, remote_type: &str) -> bool { self.remote_type == remote_type }
}

impl std::fmt::Display for CallError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = self.remote_type.rsplit(':').next().filter(|name| !name.is_empty()).unwrap_or(self.kind.name());
        write!(f, "{kind}: {}", self.message)
    }
}

impl From<String> for CallError { fn from(message: String) -> Self { Self::new(ErrorKind::Failed, message) } }

/// Where the interpreter is and what it imports.
pub struct HostConfig {
    pub python: PathBuf,
    /// `PYTHONPATH` of every plugin process (the science/plugin sources).
    pub python_path: Vec<PathBuf>,
    /// Packages whose `devices`/`nodes` folders are discovered.
    pub packages: Vec<String>,
    /// Memory cap of all plugin processes together, if any.
    pub memory_limit_mib: Option<u64>,
}

impl HostConfig {
    /// The interpreter [`plugin_python`] finds and the given source root.
    pub fn from_env(science_src: PathBuf) -> Result<Self, String> {
        let python = plugin_python()?;
        let memory_limit_mib = std::env::var("ZLC_PLUGIN_MEMORY_MIB").ok().and_then(|text| text.parse().ok());
        Ok(Self { python, python_path: vec![science_src], packages: vec!["zlc_atom".into()], memory_limit_mib })
    }
}

/// Why no plugin process can start on this machine.
pub const NOT_INSTALLED: &str = "The plugin Python environment is not installed: run installation.exe";

/// The interpreter plugin processes (and every other Python the core starts) run:
/// `ZLC_PLUGIN_PYTHON` (or `PYO3_PYTHON`) when set, else the answer of the one
/// resolver installation.exe's status, its install step and the GPU warmup use
/// (tools\tools_root.ps1 Get-ZlcSciencePython, asked once through
/// `install_science.ps1 -PluginPython`): the checkout's science environment
/// when one was made, else the base Python v2 runs with, as it is. The
/// executable started from Explorer needs no launcher.
pub fn plugin_python() -> Result<PathBuf, String> {
    static FOUND: std::sync::OnceLock<Result<PathBuf, String>> = std::sync::OnceLock::new();
    FOUND.get_or_init(|| {
        let named = |name: &str| std::env::var_os(name).filter(|value| !value.is_empty()).map(PathBuf::from);
        if let Some(python) = named("ZLC_PLUGIN_PYTHON").or_else(|| named("PYO3_PYTHON")) { return Ok(python); }
        use std::os::windows::process::CommandExt;
        let script = zlc_core::checkout::path("tools").join("install_science.ps1");
        let output = Command::new("powershell.exe").args(["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File"]).arg(&script).arg("-PluginPython")
            .stdin(Stdio::null()).creation_flags(0x0800_0000).output().map_err(|error| format!("{NOT_INSTALLED} ({}: {error})", script.display()))?; // CREATE_NO_WINDOW
        let answer = String::from_utf8_lossy(&output.stdout);
        let base = answer.lines().map(str::trim).rfind(|line| !line.is_empty()).map(PathBuf::from).filter(|path| path.is_file());
        base.ok_or_else(|| format!("{NOT_INSTALLED} ({})", answer.trim().trim_start_matches("ERROR:").trim()))
    }).clone()
}

/// What a plugin process needs in its environment beyond the interpreter, so no
/// launcher has to set it: the base installation a venv was made from (its
/// `pyvenv.cfg` `home`) first on `PATH` with its DLL folders -- for a conda base
/// the ones activation adds, as v2's launcher did, since numpy and ssl load DLLs
/// from `Library\bin`, and told to search them -- and v2's thread policy (zou_lab_control bootstrap):
/// OpenBLAS on a team of min(4, logical CPUs), Numba on OpenMP, OpenMP threads
/// that sleep when idle. An operator's own setting of each is kept.
pub fn plugin_environment(python: &std::path::Path) -> Vec<(&'static str, std::ffi::OsString)> {
    let mut environment = Vec::new();
    let home = base_home(python);
    let mut folders = vec![home.clone()];
    let conda = home.join("conda-meta").is_dir();
    if conda {
        folders.extend(["Library\\mingw-w64\\bin", "Library\\usr\\bin", "Library\\bin", "Scripts", "bin", "condabin"].map(|part| home.join(part)));
        // A conda Python (3.8+) searches PATH for an extension's DLLs only when asked to.
        if std::env::var_os("CONDA_DLL_SEARCH_MODIFICATION_ENABLE").is_none() { environment.push(("CONDA_DLL_SEARCH_MODIFICATION_ENABLE", "1".into())); }
    }
    folders.push(home.join("DLLs"));
    folders.retain(|folder| folder.is_dir());
    if !folders.is_empty() {
        if let Some(existing) = std::env::var_os("PATH") { folders.extend(std::env::split_paths(&existing)); }
        if let Ok(path) = std::env::join_paths(folders) { environment.push(("PATH", path)); }
    }
    let team = std::thread::available_parallelism().map_or(1, |count| count.get()).min(4).to_string();
    for (name, value) in [("OPENBLAS_NUM_THREADS", team.as_str()), ("NUMBA_THREADING_LAYER", "omp"), ("OMP_WAIT_POLICY", "PASSIVE")] {
        if std::env::var_os(name).is_none() { environment.push((name, value.into())); }
    }
    environment
}

/// A venv's base installation (`home` in the `pyvenv.cfg` beside `Scripts\`), else the interpreter's own folder.
fn base_home(python: &std::path::Path) -> PathBuf {
    let folder = python.parent().map(std::path::Path::to_path_buf).unwrap_or_default();
    let configs = [folder.parent().map(|venv| venv.join("pyvenv.cfg")), Some(folder.join("pyvenv.cfg"))];
    configs.into_iter().flatten().filter_map(|config| std::fs::read_to_string(config).ok()).find_map(|text| text.lines().find_map(|line| {
        let (key, value) = line.split_once('=')?;
        (key.trim().eq_ignore_ascii_case("home") && !value.trim().is_empty()).then(|| PathBuf::from(value.trim()))
    })).unwrap_or(folder)
}

/// What the host reports to the application.
#[derive(Debug)]
pub enum HostEvent {
    /// Instances lost with their process (crash, hang, protocol error).
    Failed { instances: Vec<String>, reason: String },
    RunReady { instance: String, run: String },
    RunProgress { instance: String, run: String, message: String, current: Option<u64>, total: Option<u64> },
    /// `generation`: the Runtime generation the run published under, if it began.
    RunFinished { instance: String, run: String, outcome: String, result: Value, error: Option<Value>, generation: Option<String> },
    /// A run asks the application: `run.operator_input`, `run.artifact`, `run.figure`, `run.animation`,
    /// `run.acquire` or `core` (a catalogue command, under the run's origin). Answer with
    /// `respond(ticket, ..)`; `lent` holds the blocks the request's arrays are on.
    Request { instance: String, run: String, ticket: u64, op: String, params: Value, origin: Origin, lent: Lent },
    /// Plugin commands appeared or disappeared (MCP `tools/list_changed`).
    CatalogChanged,
    /// An open device told the core a fact of its own (`device.notice`, outside any
    /// request: an SLM Editor's draft solved); `lent` holds its arrays' blocks.
    Notice { instance: String, name: String, value: Value, lent: Lent },
    /// A line of a device's own log (its code's narration, `device.log`).
    DeviceLog { instance: String, line: String },
}

/// A plugin's answer and the holds on the blocks it lent.
pub struct Reply { pub value: Value, lends: Vec<Arc<Lend>> }

impl Reply {
    /// An answer whose `$array`s are on these blocks (one hold per listing).
    pub fn new(value: Value, lends: Vec<Arc<Lend>>) -> Reply { Reply { value, lends } }
    /// Decode one `$array`/`$inline` of this reply; lent arrays are borrowed, not copied.
    pub fn array(&self, reference: &Value) -> Result<wire::Array, String> { wire::array(reference, &self.lends) }
    /// The blocks the answer lent, with multiplicity.
    pub fn lends(&self) -> &[Arc<Lend>] { &self.lends }
}

/// An instance served somewhere other than this host's plugin processes: a
/// device another core publishes (zlc-fabric). It takes the plugin protocol's
/// instance ops (`spi`, `fields`, `tune`, `values`, `command`, `make_safe`);
/// `lends` are blocks this core holds whose arrays the request carries.
pub trait Endpoint: Send + Sync {
    fn request(&self, op: &str, params: Value, lends: Vec<Arc<Lend>>, deadline: Duration) -> Result<Reply, CallError>;
    /// The instance leaves this core; the far side keeps its device.
    fn close(&self);
    /// What the endpoint has to say about its link, newest last.
    fn tail(&self, lines: usize) -> Vec<String>;
}

/// Where an instance's requests go.
#[derive(Clone)]
enum Serving { Process(Arc<PluginProcess>), Remote(Arc<dyn Endpoint>) }

impl Serving {
    fn process(&self) -> Option<&Arc<PluginProcess>> { match self { Serving::Process(process) => Some(process), Serving::Remote(_) => None } }
    /// The pid blocks are held for (0: a remote instance holds nothing here).
    fn pid(&self) -> u32 { self.process().map_or(0, |process| process.pid()) }
    fn tail(&self, lines: usize) -> Vec<String> {
        match self { Serving::Process(process) => process.tail(lines), Serving::Remote(endpoint) => endpoint.tail(lines) }
    }
    /// One request; the lends' blocks become the instance's holds (a remote one sends their bytes).
    fn request(&self, registry: &Registry, op: &str, params: Value, lends: Vec<Arc<Lend>>, deadline: Duration) -> Result<Reply, CallError> {
        match self {
            Serving::Process(process) => {
                let blocks: Vec<String> = lends.iter().map(|lend| lend.name().to_owned()).collect();
                for name in &blocks { registry.hold_for(process.pid(), name)?; }
                drop(lends);
                let incoming = process.request_with(op, params, blocks, deadline)?;
                Ok(Reply { value: incoming.message["ok"].clone(), lends: incoming.lends })
            }
            Serving::Remote(endpoint) => endpoint.request(op, params, lends, deadline),
        }
    }
}

/// The holds on the blocks a plugin's request lent.
#[derive(Default)]
pub struct Lent(Vec<Arc<Lend>>);

impl std::fmt::Debug for Lent {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_list().entries(self.0.iter().map(|lend| lend.name())).finish()
    }
}

impl Lent {
    /// One `$array`/`$inline` of the request, borrowed from its block.
    pub fn array(&self, reference: &Value) -> Result<wire::Array, String> { wire::array(reference, &self.0) }
    /// One complete dataset of the request (`zlc_plugin.figures.dataset_to_wire`).
    pub fn dataset(&self, dataset: &Value) -> Result<wire::Dataset, String> { wire::decode_dataset(dataset, &self.0) }
}

/// A plugin command as a catalogue entry (§8).
#[derive(Clone, Debug)]
pub struct PluginCommand {
    /// Its tool name, `<instance>__<command>` (`zlc_core::control::tool_name`).
    pub name: String,
    pub instance: String,
    pub command: String,
    pub summary: String,
    /// Only reads (offered to a read-only MCP client).
    pub read_only: bool,
    pub input_schema: Value,
}

/// What a node run is given at Start.
pub struct RunRequest {
    pub prepared: Option<String>,
    pub values: Value,
    /// Device instance per requirement argument.
    pub devices: BTreeMap<String, String>,
    pub device_labels: BTreeMap<String, String>,
    /// The arguments whose device the run shares (a scan's device axis); it takes every other one exclusively.
    pub shared: std::collections::BTreeSet<String>,
    pub origin: Origin,
    pub resources: Value,
    /// A processor's input: the signal its deliveries come from (§5).
    pub source_signal: Option<String>,
    pub run_directory: Option<PathBuf>,
    /// Runtime signal name per output; an output not named here is `<instance>/<output>`.
    pub outputs: BTreeMap<String, String>,
    /// The bench at Start (each open device's settings, the pulse board), the core's fact:
    /// run.begin writes it into the run record as `apparatus`. None for a processor (its
    /// sources' records reach it through lineage).
    pub apparatus: Option<Value>,
}

struct OutputState { signal: Arc<str>, schema: Arc<DatasetSchema>, finite: bool, chunk: Option<Arc<DatasetSchema>> }

struct RunState {
    id: String,
    origin: Origin,
    devices: BTreeMap<String, String>,
    signals: BTreeMap<String, String>,
    generation: Option<Generation>,
    /// The signal its commits derive from: one run of it (a processor's subscription),
    /// or the signal across its runs (a run following it), declared at run.begin.
    route: Option<(Arc<str>, Option<GenerationId>)>,
    outputs: HashMap<String, OutputState>,
    histories: Vec<HistoryLease>,
    apparatus: Option<Value>,
}

struct Instance {
    display_name: String,
    descriptor: Arc<TypeDescriptor>,
    serving: Serving,
    tunable: bool,
    snapshot: bool,
    /// The read-only methods the opened device implements (its open reply): what a draft query may call.
    reads: Vec<String>,
    /// The runs that claim this device, each exclusively or not (v2 DeviceClaim): two
    /// exclusive claims never hold one device; a shared one (a scan's axis) sits beside either.
    claims: Vec<(String, bool)>,
    /// Handed to another core (`lend_out`): nothing here may use it until taken
    /// back, and what it says for itself goes to that core.
    published: Option<Forward>,
    local_requests: Arc<std::sync::atomic::AtomicUsize>,
    run: Option<RunState>,
}

/// Registered under the same state lock that hands the device to the fabric,
/// covering the gap between local admission and sending the plugin request.
struct LocalRequest(Arc<std::sync::atomic::AtomicUsize>);
impl Drop for LocalRequest {
    fn drop(&mut self) { self.0.fetch_sub(1, Ordering::AcqRel); }
}

/// Where a device lent to another core says what it says for itself
/// (`device.notice`, `device.log`): the event's op, params and the blocks its arrays are on.
pub type Forward = Arc<dyn Fn(&str, &Value, Vec<Arc<Lend>>) + Send + Sync>;

fn published_refusal(instance: &str) -> CallError {
    CallError::new(ErrorKind::Refused, format!("{instance}: published on the bench fabric -- withdraw Remote to use it here"))
}

#[derive(Default)]
struct State {
    instances: BTreeMap<String, Instance>,
    leases: HashMap<u64, (u32, Read)>,
    next_lease: u64,
    /// Plugin requests the application has yet to answer: ticket -> (process, request id).
    requests: HashMap<u64, (Arc<PluginProcess>, Value)>,
    next_ticket: u64,
    closed: bool,
    /// Non-processor runs reading a source signal by name (a scan over the camera it
    /// restarts): each new generation of that signal is subscribed at its run.begin.
    followers: Vec<Follower>,
    /// Processors waiting for their source's next generation (a following processor):
    /// instance -> (source signal, its input subscribed at that generation's run.begin).
    awaiting: HashMap<String, (String, Option<Source>)>,
}

struct Follower { instance: String, run: String, signal: String, input: Value, sources: std::sync::mpsc::Sender<Source> }

/// A processor's subscribed input (§5).
struct Source { feed: Feed, bundle: bool, input: String, generation: GenerationId }

/// A live source's deliveries, or the whole datasets of a finite run that has
/// sealed (v2's frozen processor evaluated the dataset as it finished).
enum Feed { Live(Subscription), Sealed(Vec<Read>) }

struct Inner {
    config: HostConfig,
    /// What every plugin process gets in its environment ([`plugin_environment`]).
    environment: Vec<(&'static str, std::ffi::OsString)>,
    session: String,
    runtime: Runtime,
    safe: SafeActions,
    registry: Arc<Registry>,
    job: Job,
    catalog: Mutex<Option<Arc<Catalog>>>,
    state: Mutex<State>,
    /// Live plugin processes by pid, apart from `state`: a lent block is released
    /// to its owner from whichever thread drops it, which may hold `state` (the
    /// runtime drops a node's previous generation inside `run.begin`). Taken
    /// after `state` when both are needed, never before it.
    processes: Mutex<HashMap<u32, Arc<PluginProcess>>>,
    events: Box<dyn Fn(HostEvent) + Send + Sync>,
    /// One gate per process key: finding or starting its process, and letting one that
    /// nothing uses go, are each one step, so instances opened together (Init opens them
    /// in parallel) share one process, and a closing one never ends a process another is joining.
    gates: Mutex<HashMap<String, Arc<Mutex<()>>>>,
}

/// The plugin host. Cheap to clone; clones share it.
#[derive(Clone)]
pub struct PluginHost { inner: Arc<Inner> }

const OPEN_DEADLINE: Duration = Duration::from_secs(180);
const CALL_DEADLINE: Duration = Duration::from_secs(120);
/// How long a lost device's client may take to let its dead server go.
const LOST_CLOSE: Duration = Duration::from_secs(10);
/// How long an instance may take to close when the application ends.
const SHUTDOWN_CLOSE: Duration = Duration::from_secs(3);
const SAFE_DEADLINE: Duration = Duration::from_secs(2);
/// An unanswered request this long past its deadline makes the instance unresponsive.
const OVERDUE_GRACE: Duration = Duration::from_secs(5);
/// The key of the one plugin process every type shares unless it is `isolated`.
const SHARED: &str = "plugins";

/// An instance is named as its apparatus or board names it; only its commands' tool names are spelled for MCP.
fn named(instance: &str) -> Result<(), CallError> {
    if instance.is_empty() { Err(CallError::new(ErrorKind::Invalid, "An instance needs a name")) } else { Ok(()) }
}

impl PluginHost {
    pub fn new(config: HostConfig, runtime: Runtime, safe: SafeActions,
               events: impl Fn(HostEvent) + Send + Sync + 'static) -> Result<Self, String> {
        let session = uuid::Uuid::new_v4().simple().to_string()[..12].to_owned();
        let job = Job::new(config.memory_limit_mib)?;
        let inner = Arc::new_cyclic(|weak: &Weak<Inner>| {
            let owner = weak.clone();
            let registry = Registry::new(session.clone(), Box::new(move |pid, names| {
                if let Some(inner) = owner.upgrade() { inner.release_to(pid, names); }
            }));
            Inner { environment: plugin_environment(&config.python), config, session, runtime, safe, registry, job, catalog: Mutex::new(None),
                    state: Mutex::default(), processes: Mutex::default(), events: Box::new(events), gates: Mutex::default() }
        });
        let watched = Arc::downgrade(&inner);
        std::thread::Builder::new().name("zlc-plugin-watch".into()).spawn(move || watch(watched)).map_err(|error| error.to_string())?;
        Ok(Self { inner })
    }

    pub fn session(&self) -> &str { &self.inner.session }
    pub fn registry(&self) -> &Arc<Registry> { &self.inner.registry }

    /// Every discoverable plugin type: discovery runs once per host, and only when
    /// the catalogue kept for these sources and this environment is not current (cache.rs).
    pub fn catalog(&self) -> Result<Arc<Catalog>, String> {
        let mut cached = self.inner.catalog.lock().unwrap();
        if let Some(catalog) = cached.as_ref() { return Ok(catalog.clone()); }
        let key = cache::key(&self.inner.config, &self.inner.environment);
        let kept = cache::read(&key).and_then(|describe| serde_json::from_slice::<Catalog>(&describe).ok());
        let catalog = match kept {
            Some(catalog) => catalog,
            None => {
                // The plugin process answers it, as it answers its instances: one interpreter,
                // one environment; a process that nothing else uses goes once it has.
                let process = self.inner.acquire(SHARED, None)?;
                let answer = process.request_untracked("describe", json!({"packages": self.inner.config.packages}), OPEN_DEADLINE);
                self.inner.done(&process);
                let describe = answer.map_err(|error| format!("Plugin discovery failed: {error}"))?.message["ok"].take();
                let catalog: Catalog = serde_json::from_value(describe.clone()).map_err(|error| format!("Plugin discovery answered: {error}"))?;
                cache::write(&key, describe.to_string().as_bytes());
                catalog
            }
        };
        if catalog.protocol[0] != PROTOCOL[0] {
            return Err(format!("Plugins speak protocol {}, the core {}", catalog.protocol[0], PROTOCOL[0]));
        }
        for error in &catalog.errors { eprintln!("{}", json!({"type": "plugin-discovery-error", "module": error.module, "error": error.error})); }
        let catalog = Arc::new(catalog);
        *cached = Some(catalog.clone());
        Ok(catalog)
    }

    /// The process a type's instances open in: the one plugin process, or the type's
    /// own when it declares why it cannot share it (`isolated`).
    pub fn key_of(descriptor: &TypeDescriptor) -> String {
        if descriptor.isolated.is_empty() { SHARED.to_owned() } else { format!("type:{}", descriptor.type_id) }
    }

    /// Open one instance; `simulation` is the installation's simulation parameters, for a
    /// type standing in the simulated world.
    pub fn open(&self, instance: &str, type_id: &str, parameters: Value, simulation: Value) -> Result<Value, CallError> {
        let inner = &self.inner;
        let catalog = self.catalog()?;
        let descriptor = Arc::new(catalog.find(type_id).cloned()
            .ok_or_else(|| CallError::new(ErrorKind::Invalid, format!("No plugin type {type_id:?}")))?);
        named(instance)?;
        {
            let state = inner.state.lock().unwrap();
            if state.closed { return Err(CallError::lost("the plugin host is closed")); }
            if state.instances.contains_key(instance) { return Err(CallError::new(ErrorKind::Refused, format!("{instance} is already open"))); }
        }
        let process = inner.acquire(&Self::key_of(&descriptor), Some(instance))?;
        let opened = process.request("open", json!({"instance": instance, "type_id": type_id, "module": descriptor.module,
            "parameters": parameters, "simulation": simulation, "plain": true}), OPEN_DEADLINE);
        let reply = match opened {
            Ok(reply) => reply,
            Err(error) => {
                process.instances.lock().unwrap().remove(instance);
                inner.release_process(&process);
                return Err(error);
            }
        };
        let tunable = reply.message["ok"]["tunable"].as_bool().unwrap_or(false);
        let snapshot = reply.message["ok"]["snapshot"] == true;
        let reads = reply.message["ok"]["reads"].as_array().into_iter().flatten().filter_map(Value::as_str).map(str::to_owned).collect();
        inner.admit(instance, descriptor, Serving::Process(process), tunable, snapshot, reads)?;
        Ok(reply.message["ok"].clone())
    }

    /// Admit a device instance served by `endpoint` (another core's published
    /// device) under `descriptor`, the type it has there: its outputs join this
    /// core's Stop all (made safe through the endpoint), its commands the catalogue.
    pub fn attach(&self, instance: &str, descriptor: TypeDescriptor, endpoint: Arc<dyn Endpoint>, tunable: bool, snapshot: bool, reads: Vec<String>) -> Result<(), CallError> {
        if !descriptor.is_device() { return Err(CallError::new(ErrorKind::Invalid, format!("{} is not a device type", descriptor.type_id))); }
        named(instance)?;
        {
            let state = self.inner.state.lock().unwrap();
            if state.closed { return Err(CallError::lost("the plugin host is closed")); }
            if state.instances.contains_key(instance) { return Err(CallError::new(ErrorKind::Refused, format!("{instance} is already open"))); }
        }
        self.inner.admit(instance, Arc::new(descriptor), Serving::Remote(endpoint), tunable, snapshot, reads)
    }

    /// The type an open instance has (a remote device's is its publisher's).
    pub fn descriptor(&self, instance: &str) -> Option<Arc<TypeDescriptor>> {
        self.inner.state.lock().unwrap().instances.get(instance).map(|entry| entry.descriptor.clone())
    }

    /// Whether an open device speaks the tunable quartet.
    pub fn tunable(&self, instance: &str) -> bool { self.inner.state.lock().unwrap().instances.get(instance).is_some_and(|entry| entry.tunable) }
    pub fn reports_settings(&self, instance: &str) -> bool { self.inner.state.lock().unwrap().instances.get(instance).is_some_and(|entry| entry.snapshot) }
    pub fn reads(&self, instance: &str) -> Vec<String> {
        self.inner.state.lock().unwrap().instances.get(instance).map(|entry| entry.reads.clone()).unwrap_or_default()
    }

    /// A device whose other computer is gone for good (zlc-fabric decides: it withdrew
    /// the device, or fell silent): whether the device is served through the fabric or is
    /// a client of that computer's own server, it leaves this core as a crashed plugin's
    /// does (v2's error state; Apply opens it again).
    pub fn lost(&self, instance: &str, reason: &str) {
        if !self.inner.state.lock().unwrap().instances.get(instance).is_some_and(|entry| entry.descriptor.is_device()) { return; }
        let _ = self.close_within(instance, LOST_CLOSE);
        (self.inner.events)(HostEvent::Failed { instances: vec![instance.to_owned()], reason: reason.to_owned() });
    }

    // ------------------------------------------------------------ publishing (zlc-fabric)
    /// Hand an open device to another core: until `take_back`, no run, command,
    /// tune or call of this core reaches it; `serve` does, and its notices and
    /// log lines go to `forward` (they are that core's to hear).
    pub fn lend_out(&self, instance: &str, forward: Forward) -> Result<(), CallError> {
        let mut state = self.inner.state.lock().unwrap();
        let entry = state.instances.get_mut(instance).filter(|entry| entry.descriptor.is_device())
            .ok_or_else(|| CallError::new(ErrorKind::Invalid, format!("{instance} is not an open device")))?;
        if matches!(entry.serving, Serving::Remote(_)) {
            return Err(CallError::new(ErrorKind::Refused, format!("{instance} already comes from the bench fabric -- publish it on the machine that owns it")));
        }
        if entry.published.is_some() { return Err(published_refusal(instance)); }
        if let Some((run, _)) = entry.claims.first() { return Err(CallError::new(ErrorKind::Refused, format!("{instance} is claimed by run {run}"))); }
        entry.published = Some(forward);
        Ok(())
    }

    /// A device another core publishes said something for itself (zlc-fabric
    /// carries its `device.notice` and `device.log`): told here as if it were
    /// this core's own device.
    pub fn remote_event(&self, instance: &str, op: &str, params: &Value, lends: Vec<Arc<Lend>>) {
        let remote = self.inner.state.lock().unwrap().instances.get(instance).is_some_and(|entry| matches!(entry.serving, Serving::Remote(_)));
        if !remote { return; }
        let text = |name: &str| params[name].as_str().unwrap_or("").to_owned();
        match op {
            "device.notice" => (self.inner.events)(HostEvent::Notice { instance: instance.to_owned(), name: text("name"), value: params["value"].clone(), lent: Lent(lends) }),
            "device.log" => (self.inner.events)(HostEvent::DeviceLog { instance: instance.to_owned(), line: text("line") }),
            _ => {}
        }
    }

    pub fn take_back(&self, instance: &str) {
        if let Some(entry) = self.inner.state.lock().unwrap().instances.get_mut(instance) { entry.published = None; }
    }

    /// Actual plugin replies outstanding, including calls whose caller timed
    /// out. Fabric retirement uses this before making a device safe and reusable.
    pub fn requests_pending(&self, instance: &str) -> bool {
        self.inner.state.lock().unwrap().instances.get(instance)
            .is_some_and(|entry| entry.local_requests.load(Ordering::Acquire) > 0
                || entry.serving.process().is_some_and(|process| process.requests_pending(instance)))
    }

    /// One request of the core a device is lent to: `lends` (blocks it sent,
    /// copied into this core's own) become the device's holds.
    pub fn serve(&self, instance: &str, op: &str, mut params: Value, lends: Vec<Arc<Lend>>, deadline: Duration) -> Result<Reply, CallError> {
        let serving = {
            let state = self.inner.state.lock().unwrap();
            let entry = state.instances.get(instance).ok_or_else(|| CallError::lost(format!("{instance} is no longer open")))?;
            if entry.published.is_none() { return Err(CallError::new(ErrorKind::NotClaimed, format!("{instance} is not published"))); }
            entry.serving.clone()
        };
        params["instance"] = json!(instance);
        if op == "snapshot" {
            // A busy snapshot is unavailable metadata, never a failed device.
            // Use the same untracked request as a local Start, after the
            // fabric has checked that this connection owns the instance.
            params["plain"] = json!(true);
            let Serving::Process(process) = serving else { return Err(CallError::new(ErrorKind::Refused, "a published snapshot must come from its owning process")); };
            return process.snapshot(instance, deadline)
                .map(|incoming| Reply::new(incoming.message["ok"].clone(), incoming.lends));
        }
        serving.request(&self.inner.registry, op, params, lends, deadline)
    }

    /// A line in an instance's log (what the bench fabric did with it).
    pub fn note(&self, instance: &str, line: &str) {
        let serving = self.inner.state.lock().unwrap().instances.get(instance).map(|entry| entry.serving.clone());
        if let Some(Serving::Process(process)) = serving { process.note(line); }
    }

    /// A device type's hardware scan (v2 Scan hardware), with no instance opened:
    /// in the process its instances open in (a vendor runtime that is a process
    /// singleton is the one its open devices use), answering `{found, notes}` or
    /// the scan's refusal. `excluded`: the resources bound now, which it touches none of.
    pub fn discover(&self, type_id: &str, excluded: &[String], deadline: Duration) -> Result<Value, CallError> {
        self.typed(type_id, "discover", json!({"excluded": excluded}), deadline)
    }

    /// What a row leaves blank, found before it opens (`resolve`), with no instance opened, in
    /// the process its instance will open in: `{parameters, note}` or why it found nothing.
    pub fn resolve(&self, type_id: &str, parameters: Value, excluded: &[String], deadline: Duration) -> Result<Value, CallError> {
        self.typed(type_id, "resolve", json!({"parameters": parameters, "excluded": excluded}), deadline)
    }

    /// One op of a type rather than of an instance (`discover`, `resolve`), on its process's own thread.
    fn typed(&self, type_id: &str, op: &str, mut params: Value, deadline: Duration) -> Result<Value, CallError> {
        let inner = &self.inner;
        let catalog = self.catalog()?;
        let descriptor = catalog.find(type_id).ok_or_else(|| CallError::new(ErrorKind::Invalid, format!("No plugin type {type_id:?}")))?;
        if inner.state.lock().unwrap().closed { return Err(CallError::lost("the plugin host is closed")); }
        let process = inner.acquire(&Self::key_of(descriptor), None)?;
        for (key, value) in [("type_id", json!(type_id)), ("module", json!(descriptor.module)), ("plain", json!(true))] { params[key] = value; }
        let answer = process.request_untracked(op, params, deadline);
        inner.done(&process);
        answer.map(|reply| reply.message["ok"].clone())
    }

    /// Close one instance; a process it leaves with nothing to serve ends.
    pub fn close(&self, instance: &str) -> Result<(), CallError> { self.close_within(instance, CALL_DEADLINE) }

    fn close_within(&self, instance: &str, deadline: Duration) -> Result<(), CallError> {
        let inner = &self.inner;
        let serving = {
            let state = inner.state.lock().unwrap();
            state.instances.get(instance).map(|entry| (entry.serving.clone(), entry.descriptor.clone()))
        };
        let Some((serving, descriptor)) = serving else {
            // Init can fail after acquiring a handle but before admission. Its
            // SDK lane owns that cleanup even though no live host entry exists.
            let processes: Vec<_> = inner.processes.lock().unwrap().values()
                .filter(|process| process.alive.load(Ordering::Acquire) && !process.retiring.load(Ordering::Acquire)).cloned().collect();
            for process in processes { process.request("close", json!({"instance": instance}), deadline)?; }
            return Ok(());
        };
        let process = match serving {
            Serving::Process(process) => process,
            Serving::Remote(endpoint) => { inner.forget(instance, &descriptor); endpoint.close(); return Ok(()); }
        };
        // A refused/failed close still owns its handle. Keep the host entry so
        // another Init cannot open over it; a later Close retries that owner.
        process.request("close", json!({"instance": instance}), deadline)?;
        inner.forget(instance, &descriptor);
        // A node closed (removed, or its layout replaced) takes its signals with it (v2 retire).
        if !descriptor.is_device() { let _ = inner.runtime.withdraw(public_instance(instance)); }
        process.instances.lock().unwrap().remove(instance);
        inner.release_process(&process);
        Ok(())
    }

    /// Refused while a run claims `instance` or another core holds it.
    fn admit_local(&self, instance: &str, unclaimed: bool) -> Result<LocalRequest, CallError> {
        let state = self.inner.state.lock().unwrap();
        let entry = state.instances.get(instance).ok_or_else(|| CallError::new(ErrorKind::Invalid, format!("{instance} is not open")))?;
        if entry.published.is_some() { return Err(published_refusal(instance)); }
        if unclaimed { if let Some((run, _)) = entry.claims.first() { return Err(CallError::new(ErrorKind::NotClaimed, format!("{instance} is claimed by run {run}"))); } }
        entry.local_requests.fetch_add(1, Ordering::AcqRel);
        Ok(LocalRequest(entry.local_requests.clone()))
    }

    /// Call the open device's adapter (the core's own use: plain JSON reply).
    pub fn spi(&self, instance: &str, method: &str, args: Value, kwargs: Value, deadline: Duration) -> Result<Reply, CallError> {
        let _request = self.admit_local(instance, true)?;
        self.inner.call(instance, "spi", json!({"method": method, "args": args, "kwargs": kwargs}), deadline)
    }

    /// Any instance operation (`fields`, `tune`, `values`, `query`, `configure`).
    pub fn call(&self, instance: &str, op: &str, params: Value, deadline: Duration) -> Result<Reply, CallError> {
        let _request = self.admit_local(instance, false)?;
        self.inner.call(instance, op, params, deadline)
    }

    /// The bench at a Start (`snapshot`, §5): an open device's settings as plain JSON, null when it
    /// does not report them. A device busy in another run's call may miss `deadline`, which says
    /// nothing of its health (untracked: the late answer is dropped). A remote
    /// device uses its owner's identical snapshot operation over the fabric.
    pub fn snapshot(&self, instance: &str, deadline: Duration) -> Result<Value, CallError> {
        let _request = self.admit_local(instance, false)?;
        let serving = {
            let state = self.inner.state.lock().unwrap();
            let entry = state.instances.get(instance).filter(|entry| entry.descriptor.is_device())
                .ok_or_else(|| CallError::new(ErrorKind::Invalid, format!("{instance} is not an open device")))?;
            if entry.published.is_some() { return Err(published_refusal(instance)); }
            entry.serving.clone()
        };
        let params = json!({"instance": instance, "plain": true});
        match serving {
            Serving::Process(process) => process.snapshot(instance, deadline).map(|reply| reply.message["ok"].clone()),
            Serving::Remote(endpoint) => endpoint.request("snapshot", params, Vec::new(), deadline).map(|reply| reply.value),
        }
    }

    /// The catalogue entries of every open instance's commands.
    pub fn commands(&self) -> Vec<PluginCommand> {
        let state = self.inner.state.lock().unwrap();
        state.instances.iter().flat_map(|(instance, entry)| entry.descriptor.commands.iter().map(move |command| PluginCommand {
            name: zlc_core::control::tool_name(instance, &command.name), instance: instance.clone(), command: command.name.clone(),
            summary: command.summary.clone(), read_only: command.read_only,
            input_schema: command.input_schema.clone() })).collect()
    }

    /// Run one plugin command: its schema first.
    pub fn command(&self, name: &str, params: &Value) -> Result<Reply, CallError> {
        let command = self.commands().into_iter().find(|command| command.name == name)
            .ok_or_else(|| CallError::new(ErrorKind::UnknownOp, format!("No plugin command {name}")))?;
        schema::validate(params, &command.input_schema).map_err(|error| CallError::new(ErrorKind::Invalid, error))?;
        let _request = self.admit_local(&command.instance, false)?;
        self.inner.call(&command.instance, "command", json!({"name": command.command, "params": params}), CALL_DEADLINE)
    }

    /// A processor follows `signal` (None: no longer): each generation of it that begins is
    /// subscribed at its run.begin, before its first commit, for the run started on it.
    pub fn follow(&self, instance: &str, signal: Option<&str>) {
        let mut state = self.inner.state.lock().unwrap();
        match signal {
            Some(signal) => { state.awaiting.insert(instance.to_owned(), (signal.to_owned(), None)); }
            None => { state.awaiting.remove(instance); }
        }
    }

    /// The current operator name; run.begin freezes it into that run's record.
    pub fn set_node_name(&self, instance: &str, name: &str) {
        if let Some(entry) = self.inner.state.lock().unwrap().instances.get_mut(instance) { entry.display_name = name.to_owned(); }
    }

    /// Start a run of a node instance, claiming its devices; returns the run id.
    pub fn start_run(&self, instance: &str, request: RunRequest) -> Result<String, CallError> {
        let inner = &self.inner;
        let run = uuid::Uuid::new_v4().to_string();
        let (process, devices, source, following) = {
            let mut state = inner.state.lock().unwrap();
            let entry = state.instances.get(instance).ok_or_else(|| CallError::new(ErrorKind::Invalid, format!("{instance} is not open")))?;
            if entry.descriptor.is_device() { return Err(CallError::new(ErrorKind::Invalid, format!("{instance} is a device, not a node"))); }
            if entry.run.is_some() { return Err(CallError::new(ErrorKind::Refused, format!("{instance} is already running"))); }
            let process = entry.serving.process().cloned().ok_or_else(|| CallError::new(ErrorKind::Invalid, format!("{instance} is not a node of this core")))?;
            // A processor's input is subscribed before it starts: nothing published
            // in between is lost, a live finite source replays what it has committed
            // (v2) and a sealed one gives its whole dataset. Another node started on a
            // source signal over an exact dataset input (a scan that fires the pulse
            // driving that source) takes the same deliveries through RunContext.next_input;
            // over a 'latest' input it reads its source by name (v2: no source route).
            let mut following = None;
            // A following processor reads the generation it waited for from its first commit.
            let parked = state.awaiting.get_mut(instance).filter(|(signal, _)| Some(signal.as_str()) == request.source_signal.as_deref())
                .and_then(|(_, parked)| parked.take());
            let entry = state.instances.get(instance).expect("checked");
            let input = entry.descriptor.active_input(&request.values);
            let source = match &entry.descriptor.node {
                Some(node) if node["node_kind"] == "processor" => Some(match parked {
                    Some(source) => source,
                    None => inner.subscribe_source(input, request.source_signal.as_deref(), Start::Replay)?,
                }),
                _ => {
                    if let Some(input) = input.filter(|input| request.source_signal.is_some() && input["delivery"] == "exact") {
                        // It follows the signal by name: the generation publishing now, if
                        // any, then each one that begins while it runs (the acquisition it
                        // restarts), subscribed before that generation's first commit.
                        let (sources, received) = std::sync::mpsc::channel();
                        if let Ok(source) = inner.subscribe_source(Some(input), request.source_signal.as_deref(), Start::Current) { let _ = sources.send(source); }
                        following = Some((Follower { instance: instance.to_owned(), run: run.clone(), signal: request.source_signal.clone().unwrap_or_default(),
                            input: input.clone(), sources }, received));
                    }
                    None
                }
            };
            let route = request.source_signal.as_deref().filter(|_| source.is_some() || following.is_some())
                .map(|signal| (Arc::from(signal), source.as_ref().map(|source| source.generation)));
            let mut devices = serde_json::Map::new();
            for (argument, device) in &request.devices {
                let target = state.instances.get(device).filter(|target| target.descriptor.is_device())
                    .ok_or_else(|| CallError::new(ErrorKind::Invalid, format!("{device} is not an open device")))?;
                let exclusive = !request.shared.contains(argument);
                if let Some((owner, _)) = target.claims.iter().find(|(_, held)| exclusive && *held) {
                    return Err(CallError::new(ErrorKind::Refused, format!("{device} is claimed by run {owner}")));
                }
                if target.published.is_some() { return Err(published_refusal(device)); }
                devices.insert(argument.clone(), json!({"instance": device, "role": request.device_labels.get(device).unwrap_or(device), "type_id": target.descriptor.type_id,
                    "capabilities": target.descriptor.capabilities, "tunable": target.tunable, "reads": target.reads}));
            }
            for (argument, device) in &request.devices {
                state.instances.get_mut(device).unwrap().claims.push((run.clone(), !request.shared.contains(argument)));
            }
            state.instances.get_mut(instance).unwrap().run = Some(RunState { id: run.clone(), origin: request.origin.clone(),
                devices: request.devices.clone(), signals: request.outputs.clone(), generation: None, route, outputs: HashMap::new(),
                histories: Vec::new(), apparatus: request.apparatus.clone() });
            let following = following.map(|(follower, received)| { state.followers.push(follower); received });
            (process, devices, source, following)
        };
        let started = process.request("run.start", json!({"instance": instance, "run": run, "values": request.values,
            "devices": devices, "resources": request.resources, "source_signal": request.source_signal, "prepared": request.prepared,
            "run_directory": request.run_directory, "origin": request.origin.to_string()}), CALL_DEADLINE);
        if let Err(error) = started {
            inner.end_run(instance, &run, "failed", &json!({"error": error.message}));
            return Err(error);
        }
        if let Some(source) = source {
            let (weak, name, id, target) = (Arc::downgrade(inner), instance.to_owned(), run.clone(), process.clone());
            std::thread::Builder::new().name(format!("zlc-processor-{instance}"))
                .spawn(move || deliver(weak, name, id, target, source)).map_err(|error| CallError::from(error.to_string()))?;
        }
        if let Some(sources) = following {
            let (weak, name, id, target) = (Arc::downgrade(inner), instance.to_owned(), run.clone(), process.clone());
            std::thread::Builder::new().name(format!("zlc-follow-{instance}"))
                .spawn(move || follow(weak, name, id, target, sources)).map_err(|error| CallError::from(error.to_string()))?;
        }
        Ok(run)
    }

    /// Answer a plugin's `HostEvent::Request` (operator input, artifact, acquisition, core command).
    pub fn respond(&self, ticket: u64, result: Result<Value, CallError>) {
        let waiting = self.inner.state.lock().unwrap().requests.remove(&ticket);
        if let Some((process, id)) = waiting { self.inner.answer(&process, &id, result.map(|value| (value, Vec::new()))); }
    }

    /// Ask a run to stop; it ends with `RunFinished`.
    pub fn stop_run(&self, instance: &str, run: &str) -> Result<(), CallError> {
        let process = {
            let state = self.inner.state.lock().unwrap();
            let entry = state.instances.get(instance).ok_or_else(|| CallError::new(ErrorKind::Invalid, format!("{instance} is not open")))?;
            if entry.run.as_ref().is_none_or(|current| current.id != run) {
                return Err(CallError::new(ErrorKind::StaleRun, format!("{run} is not {instance}'s current run")));
            }
            entry.serving.process().cloned().ok_or_else(|| CallError::new(ErrorKind::Invalid, format!("{instance} is not a node of this core")))?
        };
        process.event("run.stop", json!({"instance": instance, "run": run}))
    }

    /// Open instances: (id, type, pid; 0 for one another core serves).
    pub fn instances(&self) -> Vec<(String, String, u32)> {
        let state = self.inner.state.lock().unwrap();
        state.instances.iter().map(|(name, entry)| (name.clone(), entry.descriptor.type_id.clone(), entry.serving.pid())).collect()
    }

    /// The last lines a plugin's process wrote to its log (a remote device: its link's).
    pub fn log(&self, instance: &str, lines: usize) -> Vec<String> {
        let state = self.inner.state.lock().unwrap();
        state.instances.get(instance).map(|entry| entry.serving.tail(lines)).unwrap_or_default()
    }

    /// Close everything and end every plugin process.
    pub fn shutdown(&self) {
        let names: Vec<String> = { let mut state = self.inner.state.lock().unwrap(); state.closed = true; state.instances.keys().cloned().collect() };
        // Shutdown alone may end the owner. A full pipe can hold even close/exit
        // behind a frame already being written, so the existing close + exit
        // budget needs an independent timer that does not acquire the writer.
        let owned: Vec<_> = self.inner.processes.lock().unwrap().values().cloned().collect();
        let closing = self.inner.clone();
        let (finished, finish) = std::sync::mpsc::channel::<()>();
        let watchdog = std::thread::spawn(move || {
            if matches!(finish.recv_timeout(SHUTDOWN_CLOSE + Duration::from_secs(10)), Err(std::sync::mpsc::RecvTimeoutError::Timeout)) {
                for process in &owned { process.retiring.store(true, Ordering::Release); }
                if let Err(error) = closing.job.terminate() {
                    eprintln!("Plugin shutdown could not terminate its job: {error}");
                    for process in owned { if !process.wait_exit(Duration::ZERO) { process.kill(); } }
                }
            }
        });
        // Every instance closes at once and within a bound: a run that does not stop (a scan
        // waiting for an input that will not come) cannot hold the application open.
        let closers: Vec<_> = names.into_iter().map(|name| { let host = self.clone();
            std::thread::spawn(move || { if let Err(error) = host.close_within(&name, SHUTDOWN_CLOSE) { eprintln!("Plugin {name} close: {error}"); } }) }).collect();
        for closer in closers { let _ = closer.join(); }
        let processes: Vec<_> = self.inner.processes.lock().unwrap().drain().map(|(_, process)| process).collect();
        let retirers: Vec<_> = processes.into_iter().map(|process| { let inner = self.inner.clone();
            std::thread::spawn(move || inner.retire(&process)) }).collect();
        for retirer in retirers { let _ = retirer.join(); }
        let _ = finished.send(());
        let _ = watchdog.join();
    }
}

impl Inner {
    fn gate(&self, key: &str) -> Arc<Mutex<()>> { self.gates.lock().unwrap().entry(key.to_owned()).or_default().clone() }

    /// The live process of `key`, started if there is none, now in use by `instance`
    /// (or by a request that is no instance's, until [`Inner::done`]).
    fn acquire(self: &Arc<Self>, key: &str, instance: Option<&str>) -> Result<Arc<PluginProcess>, String> {
        let gate = self.gate(key);
        let _held = gate.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let process = match self.live(key) { Some(process) => process, None => self.spawn(key)? };
        match instance {
            Some(instance) => { process.instances.lock().unwrap().insert(instance.to_owned()); }
            None => { process.users.fetch_add(1, Ordering::AcqRel); }
        }
        Ok(process)
    }

    /// A request that was no instance's has been answered.
    fn done(&self, process: &Arc<PluginProcess>) {
        process.users.fetch_sub(1, Ordering::AcqRel);
        self.release_process(process);
    }

    /// A live, non-retiring process of `key`.
    fn live(&self, key: &str) -> Option<Arc<PluginProcess>> {
        self.processes.lock().unwrap().values()
            .find(|process| process.key == key && process.alive.load(Ordering::Acquire) && !process.retiring.load(Ordering::Acquire)).cloned()
    }

    fn command(&self) -> Command {
        let mut command = Command::new(&self.config.python);
        let mut paths = self.config.python_path.clone();
        if let Some(existing) = std::env::var_os("PYTHONPATH") { paths.extend(std::env::split_paths(&existing)); }
        command.env_remove("PYTHONHOME").env("PYTHONUNBUFFERED", "1").env("PYTHONIOENCODING", "utf-8")
            .env("PYTHONPATH", std::env::join_paths(paths).unwrap_or_default());
        // The plugins' caches (startup solves, SLM GPU choices, CuPy's compiled kernels) stay in
        // the checkout's runtime folder, as everything a running program writes.
        command.env("ZLC_V3_LOCAL", zlc_core::checkout::local_root());
        if std::env::var_os("CUPY_CACHE_DIR").is_none() { command.env("CUPY_CACHE_DIR", zlc_core::checkout::local("cupy")); }
        // The base installation's DLL folders and v2's thread policy (past four, OpenBLAS
        // threads on a solver's matrices wait on each other): the plugin processes do the
        // science (WGS solves, fits).
        command.envs(self.environment.iter().map(|(name, value)| (name, value)));
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
        command
    }

    fn spawn(self: &Arc<Self>, key: &str) -> Result<Arc<PluginProcess>, String> {
        let child = self.command().args(["-m", "zlc_plugin", "host"])
            .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn()
            .map_err(|error| format!("Cannot start a plugin process with {}: {error}", self.config.python.display()))?;
        let (process, stdout, stderr) = PluginProcess::new(key.to_owned(), child);
        if let Err(error) = self.job.assign(process.raw_handle()) { process.kill(); return Err(error); }
        let label = key.to_owned();
        process.read_log(stderr, move |line| eprintln!("[plugin {label}] {line}"));
        let (weak, reading) = (Arc::downgrade(self), process.clone());
        std::thread::Builder::new().name(format!("zlc-plugin-read-{key}"))
            .spawn(move || read_loop(weak, reading, stdout)).map_err(|error| error.to_string())?;
        let hello = match process.request("hello", json!({"protocol": PROTOCOL, "session": self.session}), OPEN_DEADLINE) {
            Ok(hello) => hello,
            Err(error) => { process.kill(); return Err(format!("Plugin process handshake: {error}; {}", process.tail(8).join(" | "))); }
        };
        if hello.message["ok"]["protocol"][0].as_u64() != Some(u64::from(PROTOCOL[0])) {
            self.retire(&process);
            return Err(format!("Plugin process speaks protocol {}", hello.message["ok"]["protocol"]));
        }
        // A venv's python.exe is a launcher: the interpreter, which names and
        // owns the blocks, is its child, and reports its own pid.
        let pid = hello.message["ok"]["pid"].as_u64().and_then(|pid| u32::try_from(pid).ok())
            .ok_or("Plugin process handshake has no pid")?;
        process.set_pid(pid);
        self.processes.lock().unwrap().insert(pid, process.clone());
        Ok(process)
    }

    fn call(&self, instance: &str, op: &str, mut params: Value, deadline: Duration) -> Result<Reply, CallError> {
        let (serving, file) = {
            let state = self.state.lock().unwrap();
            let entry = state.instances.get(instance).ok_or_else(|| CallError::new(ErrorKind::Invalid, format!("{instance} is not open")))?;
            // A command's file is this computer's: read here, or written here from what the device gives (files.rs).
            let file = if op == "command" { entry.descriptor.commands.iter().find(|command| params["name"] == command.name.as_str()).and_then(|command| command.file.clone()) } else { None };
            (entry.serving.clone(), file)
        };
        params["instance"] = json!(instance);
        params["plain"] = json!(true);
        let Some(file) = file else { return serving.request(&self.registry, op, params, Vec::new(), deadline) };
        let path = files::path(&params, &file)?;
        let lends = files::hand(&mut params, &file, &path, &self.registry)?;
        let reply = serving.request(&self.registry, op, params, lends, deadline)?;
        if file.direction == "write" { files::write(reply, &path) } else { Ok(reply) }
    }

    /// An opened instance joins the host: its outputs Stop all, its commands the catalogue.
    fn admit(self: &Arc<Self>, instance: &str, descriptor: Arc<TypeDescriptor>, serving: Serving, tunable: bool, snapshot: bool, reads: Vec<String>) -> Result<(), CallError> {
        self.state.lock().unwrap().instances.insert(instance.to_owned(),
            Instance { display_name: public_instance(instance).to_owned(), descriptor: descriptor.clone(), serving, tunable, snapshot, reads, claims: Vec::new(), published: None,
                local_requests: Arc::default(), run: None });
        for output in &descriptor.physical_outputs {
            let (weak, name, port) = (Arc::downgrade(self), instance.to_owned(), output.name.clone());
            self.safe.register(&format!("{instance}/{}", output.name), Arc::new(move || {
                let inner = weak.upgrade().ok_or("the plugin host has closed")?;
                inner.call(&name, "make_safe", json!({"output": port}), SAFE_DEADLINE).map(|_| ()).map_err(|error| error.to_string())
            }));
        }
        if !descriptor.commands.is_empty() { (self.events)(HostEvent::CatalogChanged); }
        Ok(())
    }

    fn release_to(&self, pid: u32, names: Vec<String>) {
        let process = self.processes.lock().unwrap().get(&pid).cloned();
        if let Some(process) = process { let _ = process.event("release", json!({"blocks": names})); }
    }

    /// An instance left: its outputs leave Stop all and its run ends.
    fn forget(&self, instance: &str, descriptor: &TypeDescriptor) {
        for output in &descriptor.physical_outputs { self.safe.unregister(&format!("{instance}/{}", output.name)); }
        let run = self.state.lock().unwrap().instances.get(instance).and_then(|entry| entry.run.as_ref().map(|run| run.id.clone()));
        if let Some(run) = run { self.end_run(instance, &run, "failed", &json!({"error": "the node instance closed"})); }
        let mut state = self.state.lock().unwrap();
        state.instances.remove(instance);
        state.awaiting.remove(instance);
        if !descriptor.commands.is_empty() { drop(state); (self.events)(HostEvent::CatalogChanged); }
    }

    /// A process nothing uses any more ends: none waits warm for the next open (no idle
    /// interpreter, no pre-warming). Marked retiring under its gate, so an open from now
    /// on starts a new one rather than joining it.
    fn release_process(&self, process: &Arc<PluginProcess>) {
        {
            let gate = self.gate(&process.key);
            let _held = gate.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            if !process.instances.lock().unwrap().is_empty() || process.users.load(Ordering::Acquire) > 0
                || process.retiring.swap(true, Ordering::AcqRel) { return; }
        }
        self.retire(process);
    }

    /// An instance whose call overran its deadline leaves its process, which carries on
    /// with the others, and fails; it can be opened again.
    fn unresponsive(&self, process: &Arc<PluginProcess>, instance: &str) {
        let held = self.state.lock().unwrap().instances.get(instance).filter(|entry| entry.serving.pid() == process.pid())
            .map(|entry| (entry.descriptor.clone(), entry.run.as_ref().map(|run| run.id.clone())));
        let Some((descriptor, run)) = held else { return };
        let reason = format!("{instance} did not answer within its deadline; it was taken out of the plugin process");
        eprintln!("Plugin {}:{}: {reason}", process.key, process.pid());
        let _ = process.event("abandon", json!({"instance": instance}));
        if let Some(run) = run { self.end_run(instance, &run, "failed", &json!({"error": reason})); }
        self.forget(instance, &descriptor);
        process.instances.lock().unwrap().remove(instance);
        (self.events)(HostEvent::Failed { instances: vec![instance.to_owned()], reason });
        self.release_process(process);
    }

    fn retire(&self, process: &Arc<PluginProcess>) {
        process.retiring.store(true, Ordering::Release);
        let _ = process.request("exit", json!({}), Duration::from_secs(5));
        if !process.wait_exit(Duration::from_secs(5)) { process.kill(); }
        self.processes.lock().unwrap().remove(&process.pid());
    }

    /// End the instance's current run `run`: its devices released, its generation sealed (whose id it answers).
    /// A processor that did not finish with its source is withdrawn, its signals with it (v2).
    fn end_run(&self, instance: &str, run: &str, outcome: &str, detail: &Value) -> Option<String> {
        let (ended, processor) = {
            let mut state = self.state.lock().unwrap();
            let entry = state.instances.get_mut(instance)?;
            if entry.run.as_ref().is_none_or(|current| current.id != run) { return None; }
            let processor = entry.descriptor.node.as_ref().is_some_and(|node| node["node_kind"] == "processor");
            let ended = entry.run.take().unwrap();
            state.followers.retain(|follower| !(follower.instance == instance && follower.run == run));
            for device in ended.devices.values() {
                if let Some(target) = state.instances.get_mut(device) {
                    target.claims.retain(|(owner, _)| owner != run);
                }
            }
            (ended, processor)
        };
        let generation = ended.generation.as_ref()?;
        let mut sealed = detail.clone();
        sealed["status"] = json!(outcome);
        sealed["run"] = json!(ended.id);
        sealed["origin"] = json!(ended.origin.to_string());
        let result = if processor && outcome != "completed" { self.runtime.retire(generation) } else { self.runtime.seal(generation, sealed) };
        if let Err(error) = result { eprintln!("Plugin run {run} end: {error}"); }
        Some(generation.id().as_uuid().to_string())
    }

    fn process_exited(&self, process: &Arc<PluginProcess>, reason: String) {
        process.ended();
        self.registry.holder_gone(process.pid());
        let retiring = process.retiring.load(Ordering::Acquire);
        let names: Vec<String> = process.instances.lock().unwrap().iter().cloned().collect();
        // The channel closes when the interpreter ends; its launcher follows with its code.
        process.wait_exit(Duration::from_secs(2));
        let code = process.exit_code().map(|code| format!(" (exit code {code})")).unwrap_or_default();
        let tail = process.tail(6);
        let detail = if tail.is_empty() { format!("{reason}{code}") } else { format!("{reason}{code}; last log: {}", tail.join(" | ")) };
        for name in &names {
            let (descriptor, run) = {
                let state = self.state.lock().unwrap();
                let Some(entry) = state.instances.get(name) else { continue };
                (entry.descriptor.clone(), entry.run.as_ref().map(|run| run.id.clone()))
            };
            if let Some(run) = run { self.end_run(name, &run, "failed", &json!({"error": detail})); }
            self.forget(name, &descriptor);
        }
        let released: Vec<_> = {
            self.processes.lock().unwrap().remove(&process.pid());
            let mut state = self.state.lock().unwrap();
            state.requests.retain(|_, (asking, _)| !Arc::ptr_eq(asking, process));
            let held: Vec<u64> = state.leases.iter().filter(|(_, (holder, _))| *holder == process.pid()).map(|(lease, _)| *lease).collect();
            held.into_iter().filter_map(|lease| state.leases.remove(&lease)).collect()
        };
        drop(released);
        if !names.is_empty() && !retiring { (self.events)(HostEvent::Failed { instances: names, reason: detail }); }
    }

    // ------------------------------------------------------------ plugin -> core

    fn plugin_event(self: &Arc<Self>, process: &Arc<PluginProcess>, message: Value, lends: Vec<Arc<Lend>>) {
        let params = &message["p"];
        let text = |name: &str| params[name].as_str().unwrap_or("").to_owned();
        match message["op"].as_str().unwrap_or("") {
            // Only an open device of this process speaks for itself.
            // A device lent to another core speaks to that core (zlc-fabric).
            "device.notice" => {
                let instance = text("instance");
                let served = self.state.lock().unwrap().instances.get(&instance)
                    .filter(|entry| entry.descriptor.is_device() && entry.serving.pid() == process.pid()).map(|entry| entry.published.clone());
                match served {
                    Some(Some(forward)) => forward("device.notice", params, lends),
                    Some(None) => (self.events)(HostEvent::Notice { instance, name: text("name"), value: params["value"].clone(), lent: Lent(lends) }),
                    None => {}
                }
            }
            "release" => {
                let names: Vec<String> = params["blocks"].as_array().into_iter().flatten().filter_map(|name| name.as_str().map(str::to_owned)).collect();
                self.registry.released_by(process.pid(), &names);
                // The reads (and the blocks they free) are dropped after letting the lock go.
                let released: Vec<_> = {
                    let mut state = self.state.lock().unwrap();
                    params["leases"].as_array().into_iter().flatten().filter_map(Value::as_u64)
                        .filter(|lease| state.leases.get(lease).is_some_and(|(holder, _)| *holder == process.pid()))
                        .collect::<Vec<_>>().into_iter().filter_map(|lease| state.leases.remove(&lease)).collect()
                };
                drop(released);
            }
            "run.ready" => if self.current_run(&text("instance"), &text("run")) {
                (self.events)(HostEvent::RunReady { instance: text("instance"), run: text("run") });
            },
            "run.progress" => if self.current_run(&text("instance"), &text("run")) {
                (self.events)(HostEvent::RunProgress { instance: text("instance"), run: text("run"), message: text("message"),
                    current: params["current"].as_u64(), total: params["total"].as_u64() });
            },
            "run.finished" => {
                let (instance, run, outcome) = (text("instance"), text("run"), text("outcome"));
                let generation = self.end_run(&instance, &run, &outcome, &json!({"result": params["result"], "error": params["error"]}));
                (self.events)(HostEvent::RunFinished { instance, run, outcome, result: params["result"].clone(),
                    error: params.get("error").filter(|error| !error.is_null()).cloned(), generation });
            }
            // Its log stays this core's too (the publishing machine's row narrates it).
            "device.log" => if process.instances.lock().unwrap().contains(&text("instance")) {
                let forward = self.state.lock().unwrap().instances.get(&text("instance")).and_then(|entry| entry.published.clone());
                if let Some(forward) = forward { forward("device.log", params, Vec::new()); }
                (self.events)(HostEvent::DeviceLog { instance: text("instance"), line: text("line") });
            },
            other => eprintln!("Plugin {} sent unknown event {other}", process.key),
        }
    }

    fn current_run(&self, instance: &str, run: &str) -> bool {
        self.state.lock().unwrap().instances.get(instance).and_then(|entry| entry.run.as_ref()).is_some_and(|current| current.id == run)
    }

    fn plugin_request(self: &Arc<Self>, process: &Arc<PluginProcess>, message: Value, lends: Vec<Arc<Lend>>) {
        let id = message["id"].clone();
        let op = message["op"].as_str().unwrap_or("").to_owned();
        if op == "device" {
            // Waiting for another plugin must not stop this one's channel.
            let (inner, process) = (self.clone(), process.clone());
            let _ = std::thread::Builder::new().name("zlc-plugin-device-call".into()).spawn(move || {
                let answer = inner.route_device(&process, &message, lends);
                inner.answer(&process, &id, answer);
            });
            return;
        }
        let answer = match op.as_str() {
            "run.begin" => self.run_begin(&message["p"]).map(|value| (value, Vec::new())),
            "run.commit" => self.run_commit(&message["p"], &lends).map(|value| (value, Vec::new())),
            "runtime.read" => self.runtime_read(process, &message["p"]),
            "run.operator_input" | "run.artifact" | "run.figure" | "run.animation" | "run.acquire" | "core" => {
                // The application answers these (respond); the run's thread waits.
                let params = &message["p"];
                let (instance, run) = (params["instance"].as_str().unwrap_or("").to_owned(), params["run"].as_str().unwrap_or("").to_owned());
                let mut state = self.state.lock().unwrap();
                // A draft query's `core` (zlc_plugin.core) has no run: the request carries
                // run "", and the application answers reads only.
                let drafted = op == "core" && params["run"].is_null() && state.instances.contains_key(&instance);
                let origin = if drafted { Ok(Origin::Operator) } else { Self::run_of(&mut state, params).map(|current| current.origin.clone()) };
                match origin {
                    Ok(origin) => {
                        state.next_ticket += 1;
                        let ticket = state.next_ticket;
                        state.requests.insert(ticket, (process.clone(), id));
                        drop(state);
                        (self.events)(HostEvent::Request { instance, run, ticket, op: op.clone(), params: params.clone(), origin, lent: Lent(lends) });
                        return;
                    }
                    Err(error) => Err(error),
                }
            }
            other => Err(CallError::new(ErrorKind::UnknownOp, format!("the core has no operation {other:?}"))),
        };
        self.answer(process, &id, answer);
    }

    fn answer(&self, process: &PluginProcess, id: &Value, answer: Result<(Value, Vec<String>), CallError>) {
        let message = match answer {
            Ok((value, blocks)) => json!({"t": "res", "id": id, "ok": value, "blocks": blocks}),
            Err(error) => json!({"t": "res", "id": id, "error": error.wire()}),
        };
        if let Err(error) = process.send(&message) { eprintln!("Plugin {} answer: {error}", process.key); }
    }

    fn run_of<'a>(state: &'a mut State, params: &Value) -> Result<&'a mut RunState, CallError> {
        let instance = params["instance"].as_str().unwrap_or("");
        let run = params["run"].as_str().unwrap_or("");
        state.instances.get_mut(instance).and_then(|entry| entry.run.as_mut()).filter(|current| current.id == run)
            .ok_or_else(|| CallError::new(ErrorKind::StaleRun, format!("{run} is not {instance}'s current run")))
    }

    fn run_begin(&self, params: &Value) -> Result<Value, CallError> {
        let instance = params["instance"].as_str().unwrap_or("").to_owned();
        let mut declarations = Vec::new();
        let mut outputs = HashMap::new();
        for output in params["outputs"].as_array().ok_or_else(|| CallError::new(ErrorKind::Invalid, "run.begin needs outputs"))? {
            let name = output["name"].as_str().ok_or_else(|| CallError::new(ErrorKind::Invalid, "an output needs a name"))?;
            let schema = wire::schema_from_tree(&output["schema"]).map_err(|error| CallError::new(ErrorKind::Invalid, format!("{name}: {error}")))?;
            let finite = match output["storage"].as_str() {
                Some("latest") => false, Some("finite") => true,
                other => return Err(CallError::new(ErrorKind::Invalid, format!("{name}: unknown storage {other:?}"))),
            };
            declarations.push((name.to_owned(), schema, finite));
        }
        let mut state = self.state.lock().unwrap();
        let display_name = state.instances.get(&instance).map(|entry| entry.display_name.clone()).unwrap_or_else(|| public_instance(&instance).to_owned());
        let run = Self::run_of(&mut state, params)?;
        if run.generation.is_some() { return Err(CallError::new(ErrorKind::Refused, "this run already declared its outputs")); }
        let declarations = declarations.into_iter().map(|(name, schema, finite)| {
            let signal: Arc<str> = run.signals.get(&name).cloned().unwrap_or_else(|| format!("{}/{name}", public_instance(&instance))).into();
            outputs.insert(name, OutputState { signal: signal.clone(), schema: schema.clone(), finite, chunk: None });
            Declaration { signal, schema, storage: if finite { Storage::Finite } else { Storage::Latest } }
        }).collect();
        // Scientific facts belong to the plugin; identity and the name at declaration belong to the host,
        // and the bench at Start to the core: a plugin's own `apparatus` is dropped, never shown as the core's.
        let mut record = if params["record"].is_object() { params["record"].clone() } else { json!({}) };
        record["node"] = json!(public_instance(&instance));
        record["node_name"] = json!(display_name);
        match &run.apparatus {
            Some(bench) => record["apparatus"] = bench.clone(),
            None => if let Some(fields) = record.as_object_mut() { fields.remove("apparatus"); },
        }
        // A run consuming a source commits each result with the event it is of: the
        // Runtime selects a picture and its judgement as one shot through this route.
        let generation = match run.route.clone() {
            Some((source, of)) => self.runtime.begin_consuming(public_instance(&instance), declarations, record, source, of),
            None => self.runtime.begin(public_instance(&instance), declarations, record),
        }.map_err(|error| CallError::from(error.to_string()))?;
        let id = generation.id().as_uuid().to_string();
        run.generation = Some(generation);
        let declared: Vec<String> = outputs.values().map(|output| output.signal.to_string()).collect();
        run.outputs = outputs;
        // Runs following one of these signals read this generation from its first commit.
        for follower in state.followers.iter().filter(|follower| declared.contains(&follower.signal)) {
            if let Ok(source) = self.subscribe_source(Some(&follower.input), Some(&follower.signal), Start::Current) { let _ = follower.sources.send(source); }
        }
        // So do the processors waiting for one of them: the run the application starts on it takes this.
        let State { awaiting, instances, .. } = &mut *state;
        for (instance, (signal, parked)) in awaiting.iter_mut().filter(|(_, (signal, _))| declared.contains(signal)) {
            if let Some(entry) = instances.get(instance) {
                *parked = self.subscribe_source(entry.descriptor.active_input(&Value::Null), Some(signal), Start::Replay).ok();
            }
        }
        Ok(json!({"generation": id}))
    }

    fn run_commit(&self, params: &Value, lends: &[Arc<Lend>]) -> Result<Value, CallError> {
        let invalid = |message: String| CallError::new(ErrorKind::Invalid, message);
        let mut names = HashMap::new();
        let (generation, inputs, parents) = {
            let mut state = self.state.lock().unwrap();
            let parents = params["parents"].as_array().into_iter().flatten().filter_map(Value::as_u64)
                .map(|lease| state.leases.get(&lease).map(|(_, read)| read.anchor().clone())
                    .ok_or_else(|| invalid(format!("parent lease {lease} is not held"))))
                .collect::<Result<Vec<_>, _>>()?;
            let run = Self::run_of(&mut state, params)?;
            let generation = run.generation.clone().ok_or_else(|| invalid("run.commit before run.begin".into()))?;
            let mut inputs = Vec::new();
            for entry in params["inputs"].as_array().ok_or_else(|| invalid("run.commit needs inputs".into()))? {
                let name = entry["output"].as_str().unwrap_or("");
                let output = run.outputs.get_mut(name).ok_or_else(|| invalid(format!("{name} was not declared at run.begin")))?;
                if let Some(tree) = entry.get("schema").filter(|tree| !tree.is_null()) {
                    output.chunk = Some(wire::schema_from_tree(tree).map_err(|error| invalid(format!("{name}: {error}")))?);
                }
                let schema = if output.finite {
                    output.chunk.clone().ok_or_else(|| invalid(format!("{name}: a finite event needs its chunk schema")))?
                } else { output.chunk.clone().unwrap_or_else(|| output.schema.clone()) };
                let data = wire::decode_block(&entry["data"], schema, lends).map_err(|error| invalid(format!("{name}: {error}")))?;
                let ordinal = entry["ordinal"].as_u64().ok_or_else(|| invalid(format!("{name}: an event needs its ordinal")))?;
                let origin = match &entry["origin"] {
                    Value::Null => None,
                    value => Some(serde_json::from_value::<[usize; 2]>(value.clone()).map_err(|error| invalid(format!("{name} origin: {error}")))?),
                };
                names.insert(output.signal.clone(), name.to_owned());
                inputs.push(Input::Event { signal: output.signal.clone(), ordinal, data, origin });
            }
            (generation, inputs, parents)
        };
        let metadata = if params["metadata"].is_object() { params["metadata"].clone() } else { json!({}) };
        let delivery = self.runtime.commit(&generation, &Bundle { inputs, metadata, parents })
            .map_err(|error| CallError::from(error.to_string()))?;
        let keys: serde_json::Map<String, Value> = delivery.publications.iter().map(|publication| {
            let key = publication.key();
            (names[&key.signal].clone(), json!({"generation": key.generation.as_uuid().to_string(), "signal": key.signal.as_ref(),
                          "revision": key.revision, "ordinal": key.ordinal}))
        }).collect();
        Ok(Value::Object(keys))
    }

    fn runtime_read(&self, process: &Arc<PluginProcess>, params: &Value) -> Result<(Value, Vec<String>), CallError> {
        let invalid = |message: String| CallError::new(ErrorKind::Invalid, message);
        let asked = params["signal"].as_str().ok_or_else(|| invalid("runtime.read needs a signal".into()))?;
        // A run reads an output of its own by `<instance>/<output>`: the signal it publishes it as.
        let own = {
            let mut state = self.state.lock().unwrap();
            let instance = params["instance"].as_str().unwrap_or("");
            asked.strip_prefix(&format!("{}/", public_instance(instance))).and_then(|output| Self::run_of(&mut state, params).ok()
                .and_then(|run| run.outputs.get(output).map(|declared| declared.signal.to_string())))
        };
        let signal = own.as_deref().unwrap_or(asked);
        let description = self.runtime.signals().into_iter().rev().find(|item| item.signal.as_ref() == signal)
            .ok_or_else(|| invalid(format!("{signal} is not a Runtime signal")))?;
        let generation = description.generation.clone();
        let read = match params["mode"].as_str().unwrap_or("event") {
            // A dataset publication (a Fit or ROI over a finite run) is its own event.
            "event" => self.runtime.read_latest(&generation, signal).and_then(|event|
                if event.data().is_none() && event.canonical().is_some() { Ok(Read::from_canonical(event)) } else { Read::from_event(event) }),
            "dataset" => self.runtime.read_views(&[(generation, signal.into(), None)]).remove(0),
            "window" => {
                let width = params["width"].as_u64().filter(|width| *width > 0).ok_or_else(|| invalid("a window read needs a width".into()))? as usize;
                // The run keeps the history it reads; retention starts with this lease.
                let lease = self.runtime.retain_history(&generation, signal, width).map_err(|error| CallError::from(error.to_string()))?;
                let mut state = self.state.lock().unwrap();
                if let Ok(run) = Self::run_of(&mut state, params) { run.histories.push(lease); }
                drop(state);
                self.runtime.read_window(&generation, signal, width).map(Read::from_window)
            }
            other => return Err(invalid(format!("unknown read mode {other:?}"))),
        }.map_err(|error| CallError::from(error.to_string()))?;
        Ok((self.encode_read(process.pid(), read, None)?, Vec::new()))
    }

    /// One read for plugin `holder`, leased until the plugin releases it (§6).
    /// A finite event also states its run's canonical schema and its placement.
    /// With `refs` (a run's deliveries) each schema and the source run's record are named by
    /// number, and the lineage is left out (the SDK makes no `Read` of a delivery).
    fn encode_read(&self, holder: u32, read: Read, mut refs: Option<&mut RunRefs>) -> Result<Value, CallError> {
        let registry = &self.registry;
        let encode = |origin: [usize; 2], block: &zlc_core::dataset::DataBlock| -> Result<Value, String> {
            Ok(json!({"origin": origin, "extent": [block.schema().domain(DomainRole::Repeat).size(), block.schema().domain(DomainRole::Point).size()],
                      "data": wire::encode_block(block, registry, holder)?}))
        };
        // `field` whole, or `<field>_ref` into the run's schemas.
        let mut name_schema = |target: &mut Value, field: &str, schema: &Arc<DatasetSchema>| match refs.as_deref_mut() {
            Some(refs) => target[format!("{field}_ref")] = json!(refs.reference(schema)),
            None => target[field] = wire::schema_tree(schema),
        };
        let mut placement = Value::Null;
        let (segments, window) = match read.kind() {
            ReadKind::Event(event) => {
                let data = event.data().expect("event read has data");
                if let Some(canonical) = event.canonical() {
                    let origin = canonical.segments().find(|segment| Arc::ptr_eq(segment.data(), data)).map(|segment| segment.origin());
                    placement = json!({"cell_origin": origin});
                    name_schema(&mut placement, "canonical_schema", canonical.schema());
                }
                (vec![encode([0, 0], data)?], Value::Null)
            }
            ReadKind::Canonical(event) => {
                let snapshot = event.canonical().expect("canonical read");
                // The whole dataset is its own canonical placement.
                placement = json!({"cell_origin": [0, 0]});
                name_schema(&mut placement, "canonical_schema", snapshot.schema());
                (snapshot.segments().map(|segment| encode(segment.origin(), segment.data())).collect::<Result<Vec<_>, _>>()?,
                 snapshot.window().map_or(Value::Null, |(start, latest)| json!([start, latest])))
            }
            ReadKind::Window(window) => {
                let segments = window.entries().iter().map(|entry| {
                    let origin = window.placement(entry).ok_or("Publication is outside its window")?;
                    encode(origin, entry.data().ok_or("History entry has no payload")?)
                }).collect::<Result<Vec<_>, String>>()?;
                let ordinal = read.key().ordinal.unwrap_or(0) as i128;
                (segments, json!([ordinal + 1 - window.entries().len().max(1) as i128, ordinal]))
            }
        };
        let key_json = |key: &zlc_core::runtime::PublicationKey| json!({"generation": key.generation.as_uuid().to_string(),
            "signal": key.signal.as_ref(), "revision": key.revision, "ordinal": key.ordinal});
        let anchor = read.anchor();
        let mut answer = json!({"key": key_json(read.key()), "window": window,
            "segments": segments, "metadata": anchor.event().metadata, "placement": placement});
        name_schema(&mut answer, "schema", read.schema());
        match refs {
            Some(refs) => answer["run_record_ref"] = json!(refs.record(anchor.run())),
            None => {
                // Its ancestry, nearest first, each publication once (v2 walked direct_parent_publications).
                let (mut lineage, mut seen) = (Vec::new(), std::collections::HashSet::new());
                let mut pending: std::collections::VecDeque<_> = anchor.event().parents.iter().collect();
                while let Some(parent) = pending.pop_front() {
                    if !seen.insert(&parent.key) { continue; }
                    lineage.push(json!({"key": key_json(&parent.key), "run_record": parent.run.metadata, "event_record": parent.event.metadata}));
                    pending.extend(parent.event.parents.iter());
                }
                answer["run_record"] = anchor.run().metadata.clone();
                answer["lineage"] = json!(lineage);
            }
        }
        let mut state = self.state.lock().unwrap();
        state.next_lease += 1;
        let lease = state.next_lease;
        state.leases.insert(lease, (holder, read));
        answer["lease"] = json!(lease);
        Ok(answer)
    }

    /// A node calls a device it claimed; the reply's blocks pass to the node.
    fn route_device(&self, node: &Arc<PluginProcess>, message: &Value, lends: Vec<Arc<Lend>>) -> Result<(Value, Vec<String>), CallError> {
        let params = &message["p"];
        let argument = params["argument"].as_str().unwrap_or("");
        let method = params["method"].as_str().unwrap_or("").to_owned();
        let (device, serving) = {
            let mut state = self.state.lock().unwrap();
            // A draft query (no run) reads the devices its draft names, as v2's
            // declare_outputs read `source.outputs`: reads only, nothing claimed.
            let drafted = params["run"].is_null();
            let device = if drafted {
                params["device"].as_str().filter(|device| state.instances.get(*device).is_some_and(|entry| entry.descriptor.is_device()))
                    .ok_or_else(|| CallError::new(ErrorKind::Invalid, format!("a draft query names no open device as {argument:?}")))?.to_owned()
            } else {
                let run = Self::run_of(&mut state, params)?;
                run.devices.get(argument).cloned()
                    .ok_or_else(|| CallError::new(ErrorKind::NotClaimed, format!("this run has no device {argument:?}")))?
            };
            let entry = state.instances.get(&device).ok_or_else(|| CallError::lost(format!("{device} is no longer open")))?;
            if drafted && !descriptor::method_reads(&method, &entry.reads) {
                return Err(CallError::new(ErrorKind::Refused, format!("a draft query only reads {device}; {method} is not a read")));
            }
            (device, entry.serving.clone())
        };
        // The node's lends in its arguments pass to the device as the device's holds
        // (to a remote device, as the bytes of its blocks).
        let deadline = message["deadline_ms"].as_u64().map_or(CALL_DEADLINE, Duration::from_millis);
        let reply = serving.request(&self.registry, "spi", json!({"instance": device, "method": method,
            "args": params["args"], "kwargs": params["kwargs"]}), lends, deadline)?;
        let returned: Vec<String> = reply.lends.iter().map(|lend| lend.name().to_owned()).collect();
        for name in &returned { self.registry.hold_for(node.pid(), name)?; }
        Ok((reply.value, returned))
    }

    /// Subscribe a run's dataset input: one signal, or with `select_bundle` every
    /// output of that signal's producer, delivered together as one bundle; an exact
    /// input from `start` on, as v2's follow queue.
    fn subscribe_source(&self, input: Option<&Value>, signal: Option<&str>, start: Start) -> Result<Source, CallError> {
        let invalid = |message: String| CallError::new(ErrorKind::Invalid, message);
        let input = input.ok_or_else(|| invalid("the node has no enabled dataset input".into()))?;
        let signal = signal.ok_or_else(|| invalid("a processor run needs its source signal".into()))?;
        let signals = self.runtime.signals();
        let anchor = signals.iter().rev().find(|item| item.signal.as_ref() == signal)
            .ok_or_else(|| invalid(format!("{signal} is not a Runtime signal")))?;
        let bundle = input["select_bundle"] == true;
        let members = signals.iter().filter(|item| item.generation.id() == anchor.generation.id()
            && (bundle || item.signal == anchor.signal));
        let name = input["name"].as_str().unwrap_or("source").to_owned();
        if anchor.sealed && anchor.finite {
            let views: Vec<_> = members.filter(|item| item.published).map(|item| (item.generation.clone(), item.signal.clone(), None)).collect();
            let reads = self.runtime.read_views(&views).into_iter().collect::<Result<Vec<_>, _>>()
                .map_err(|error| CallError::from(format!("{signal}: {error}")))?;
            return Ok(Source { feed: Feed::Sealed(reads), bundle, input: name, generation: anchor.generation.id() });
        }
        let requests = members.map(|item| ReadRequest { signal: item.signal.clone(), mode: ReadMode::Event }).collect();
        let (policy, start) = if input["delivery"] == "latest" { (DeliveryPolicy::Latest, Start::Current) } else { (DeliveryPolicy::Follow, start) };
        let subscription = self.runtime.subscribe_reads(&anchor.generation, requests, policy, start)
            .map_err(|error| CallError::from(error.to_string()))?;
        Ok(Source { feed: Feed::Live(subscription), bundle, input: name, generation: anchor.generation.id() })
    }
}

/// A delivery waits as long as its run: answered when the run takes it, refused
/// (`stale_run`) when the run ends first, lost with the process. How long a run
/// leaves its input waiting says nothing about the process's health.
const DELIVERY_WAIT: Duration = Duration::from_secs(7 * 24 * 3600);

/// A processor run's deliveries, in order, one at a time: the next waits for the
/// plugin to have evaluated and committed the previous (v2's follow processor).
/// The source's seal or retirement ends the run; an exact queue overflow fails it.
fn deliver(weak: Weak<Inner>, instance: String, run: String, process: Arc<PluginProcess>, source: Source) {
    let send = |params: Value| process.request_untracked("run.deliver", params, DELIVERY_WAIT).is_ok();
    let base = json!({"instance": instance, "run": run});
    let with = |kind: &str, extra: Value| { let mut params = base.clone(); params["kind"] = json!(kind);
        if let (Some(target), Some(source)) = (params.as_object_mut(), extra.as_object()) { target.extend(source.clone()); } params };
    // One delivery of these reads, encoded; the host is not held while the run evaluates it.
    // False once the run is gone or the delivery failed.
    let refs = std::cell::RefCell::new(RunRefs::default());
    let data = |inner: Arc<Inner>, reads: &[Read]| match data_params(&inner, &process, &source, reads, &mut refs.borrow_mut()) {
        Ok(extra) => { drop(inner); send(with("data", extra)) }
        Err(error) => { send(with("failed", json!({"error": error}))); false }
    };
    let subscription = match &source.feed {
        Feed::Live(subscription) => subscription,
        Feed::Sealed(reads) => {
            let Some(inner) = weak.upgrade().filter(|inner| inner.current_run(&instance, &run)) else { return };
            if data(inner, reads) { send(with("sealed", json!({}))); }
            return;
        }
    };
    loop {
        let Some(inner) = weak.upgrade() else { return };
        if !inner.current_run(&instance, &run) || !process.alive.load(Ordering::Acquire) { return; }
        // A subscription bound to the source's generation ends with its seal or
        // retirement as its terminal, after every queued delivery.
        let delivery = match subscription.next(Duration::from_millis(250)) {
            Ok(None) => continue,
            Ok(Some(delivery)) => delivery,
            Err(RuntimeError::Sealed) => { send(with("sealed", json!({}))); return; }
            Err(RuntimeError::UnknownGeneration) => { send(with("retired", json!({}))); return; }
            Err(error) => { send(with("failed", json!({"error": format!("processor input: {error}")}))); return; }
        };
        match &delivery.kind {
            DeliveryKind::Data => { if !data(inner, &delivery.reads) { return; } }
            DeliveryKind::Sealed { .. } => { send(with("sealed", json!({}))); return; }
            DeliveryKind::Retired => { send(with("retired", json!({}))); return; }
        }
    }
}

/// A following run's deliveries: the subscription in hand until a newer generation's
/// arrives; a source that ends (sealed, retired) leaves it waiting for the next one,
/// and the run's own timeouts decide (v2 read the signal by name across restarts).
fn follow(weak: Weak<Inner>, instance: String, run: String, process: Arc<PluginProcess>, sources: std::sync::mpsc::Receiver<Source>) {
    let send = |params: Value| process.request_untracked("run.deliver", params, DELIVERY_WAIT).is_ok();
    let base = json!({"instance": instance, "run": run});
    let with = |kind: &str, extra: Value| { let mut params = base.clone(); params["kind"] = json!(kind);
        if let (Some(target), Some(source)) = (params.as_object_mut(), extra.as_object()) { target.extend(source.clone()); } params };
    let mut current: Option<Source> = None;
    let mut refs = RunRefs::default();
    loop {
        let Some(inner) = weak.upgrade() else { return };
        if !inner.current_run(&instance, &run) || !process.alive.load(Ordering::Acquire) { return; }
        // A new source is a new set of schemas.
        while let Ok(next) = sources.try_recv() { current = Some(next); refs.reset(); }
        let Some(source) = &current else {
            drop(inner);
            if let Ok(next) = sources.recv_timeout(Duration::from_millis(250)) { current = Some(next); refs.reset(); }
            continue;
        };
        // A finite source that had already sealed is the world before this run: wait for the next.
        let Feed::Live(subscription) = &source.feed else { current = None; continue; };
        let delivery = match subscription.next(Duration::from_millis(250)) {
            Ok(None) => continue,
            Ok(Some(delivery)) => delivery,
            Err(RuntimeError::Sealed | RuntimeError::UnknownGeneration) => { current = None; continue; }
            Err(error) => { send(with("failed", json!({"error": format!("input: {error}")}))); return; }
        };
        match &delivery.kind {
            DeliveryKind::Data => match data_params(&inner, &process, source, &delivery.reads, &mut refs) {
                Ok(extra) => { drop(inner); if !send(with("data", extra)) { return; } }
                Err(error) => { send(with("failed", json!({"error": error}))); return; }
            },
            DeliveryKind::Sealed { .. } | DeliveryKind::Retired => current = None,
        }
    }
}

/// One data delivery as `run.deliver` carries it: `{inputs: {member: read}, anchor}`, the
/// reads naming their schemas and source run records by number, and `schemas` / `run_records`
/// defining those the run has not been sent yet (`refs_reset` first after a new source).
fn data_params(inner: &Inner, process: &PluginProcess, source: &Source, reads: &[Read], refs: &mut RunRefs) -> Result<Value, String> {
    let mut inputs = serde_json::Map::new();
    let mut anchor = None;
    for read in reads {
        let signal = read.key().signal.to_string();
        let member = if source.bundle { signal.rsplit('/').next().unwrap_or(&signal).to_owned() } else { source.input.clone() };
        let mut value = inner.encode_read(process.pid(), read.clone(), Some(&mut *refs)).map_err(|error| error.message)?;
        value["signal"] = json!(signal);
        anchor.get_or_insert(member.clone());
        inputs.insert(member, value);
    }
    let mut params = json!({"inputs": inputs, "anchor": anchor});
    refs.define(&mut params);
    Ok(params)
}

/// What one run's deliveries have sent (PLUGIN_PROTOCOL.md §5): each schema and each source
/// run's record whole once, numbered, and named by its number afterwards, so a delivery
/// carries its values, not every coordinate of its axes or every site of a run record again.
/// A run's deliveries go out one at a time and the run takes them in order, so a number is
/// always defined before a read names it.
#[derive(Default)]
struct RunRefs {
    /// Every schema sent this run: its Arc, its number and the tree it went as.
    schemas: Vec<(Arc<DatasetSchema>, u64, Value)>,
    /// Every source run record sent this run, with its number.
    records: Vec<(Arc<RunRecord>, u64)>,
    /// The definitions the next delivery carries.
    fresh_schemas: serde_json::Map<String, Value>,
    fresh_records: serde_json::Map<String, Value>,
    reset: bool,
    next: u64,
}

impl RunRefs {
    /// The number of `schema`, defined in the next delivery when it is new. Another Arc of
    /// an equal schema is that number when its tree is the very one sent.
    fn reference(&mut self, schema: &Arc<DatasetSchema>) -> u64 {
        if let Some((_, number, _)) = self.schemas.iter().find(|(known, ..)| Arc::ptr_eq(known, schema)) { return *number; }
        let tree = wire::schema_tree(schema);
        if let Some(entry) = self.schemas.iter_mut().find(|(known, _, sent)| **known == **schema && *sent == tree) {
            entry.0 = schema.clone();
            return entry.1;
        }
        self.next += 1;
        self.fresh_schemas.insert(self.next.to_string(), tree.clone());
        self.schemas.push((schema.clone(), self.next, tree));
        self.next
    }

    /// The number of a source run's record, defined in the next delivery when it is new.
    fn record(&mut self, run: &Arc<RunRecord>) -> u64 {
        if let Some(entry) = self.records.iter_mut().find(|(known, _)| Arc::ptr_eq(known, run) || known.metadata == run.metadata) {
            entry.0 = run.clone();
            return entry.1;
        }
        self.next += 1;
        self.fresh_records.insert(self.next.to_string(), run.metadata.clone());
        self.records.push((run.clone(), self.next));
        self.next
    }

    /// A new source: the plugin forgets the table with the next delivery.
    fn reset(&mut self) {
        *self = Self { reset: true, next: self.next, ..Self::default() };
    }

    /// The pending definitions, into a delivery's params.
    fn define(&mut self, params: &mut Value) {
        if std::mem::take(&mut self.reset) { params["refs_reset"] = json!(true); }
        if !self.fresh_schemas.is_empty() { params["schemas"] = Value::Object(std::mem::take(&mut self.fresh_schemas)); }
        if !self.fresh_records.is_empty() { params["run_records"] = Value::Object(std::mem::take(&mut self.fresh_records)); }
    }
}

fn read_loop(weak: Weak<Inner>, process: Arc<PluginProcess>, mut stdout: std::process::ChildStdout) {
    let reason = loop {
        let message = match frame::read(&mut stdout) {
            Ok(Some(message)) => message,
            Ok(None) => break "the plugin process exited".to_owned(),
            Err(error) => break format!("the plugin channel failed: {error}"),
        };
        let Some(inner) = weak.upgrade() else { return };
        let mut lends = Vec::new();
        for name in message["blocks"].as_array().into_iter().flatten().filter_map(Value::as_str) {
            match inner.registry.lend(name, process.pid()) {
                Ok(lend) => lends.push(Arc::new(lend)),
                Err(error) => eprintln!("Plugin {} lent an unusable block: {error}", process.key),
            }
        }
        match message["t"].as_str() {
            Some("res") => {
                if let Some(id) = message["id"].as_u64() { process.resolve(id, Incoming { message, lends }); }
            }
            Some("evt") => inner.plugin_event(&process, message, lends),
            Some("req") => inner.plugin_request(&process, message, lends),
            _ => eprintln!("Plugin {} sent a message without a kind", process.key),
        }
    };
    process.missed_pings.store(0, Ordering::Relaxed);
    if let Some(inner) = weak.upgrade() { inner.process_exited(&process, reason); }
}

/// Health, per process: a ping every second, answered by its main thread whatever
/// its lanes are doing. `MISSED_PINGS` unanswered in a row -- the interpreter itself
/// is stuck -- end the process, and every instance in it fails (each can be opened
/// again). A request overdue past its grace fails only the instance it was for: that
/// instance leaves the process, the others carry on. The ping allowance is long
/// because a Numba compilation holds the GIL for seconds.
const MISSED_PINGS: u64 = 60;

fn watch(weak: Weak<Inner>) {
    loop {
        std::thread::sleep(Duration::from_secs(1));
        let Some(inner) = weak.upgrade() else { return };
        let processes: Vec<_> = inner.processes.lock().unwrap().values().cloned().collect();
        drop(inner);
        for process in processes {
            if process.retiring.load(Ordering::Acquire) || !process.alive.load(Ordering::Acquire) { continue; }
            let overdue = process.overdue(OVERDUE_GRACE);
            // An overdue request of no instance's (the handshake) is the process's own.
            let missed = process.missed_pings.load(Ordering::Relaxed);
            if overdue.iter().any(Option::is_none) || missed >= MISSED_PINGS {
                eprintln!("Plugin {}:{} is unresponsive ({}); ending it", process.key, process.pid(),
                          if missed >= MISSED_PINGS { format!("{missed} pings unanswered") } else { "a request is overdue".to_owned() });
                process.kill();
                continue;
            }
            if let Some(inner) = weak.upgrade() { for instance in overdue.into_iter().flatten() { inner.unresponsive(&process, &instance); } }
            if process.ping() { process.missed_pings.store(0, Ordering::Relaxed); }
            else { process.missed_pings.fetch_add(1, Ordering::Relaxed); }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_admission_is_pending_before_the_request_reaches_the_plugin() {
        let Some(python) = std::env::var_os("ZLC_PLUGIN_PYTHON") else { return };
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let host = PluginHost::new(HostConfig { python: python.into(),
            python_path: vec![root.join("../../science/src"), root.join("tests/plugins")],
            packages: vec!["zlc_test_plugins".into()], memory_limit_mib: Some(3072),
        }, Runtime::new(), SafeActions::new(), |_| {}).unwrap();
        host.open("rf", "test.fabric", json!({}), json!({})).unwrap();
        let request = host.admit_local("rf", true).unwrap();
        host.lend_out("rf", Arc::new(|_, _, _| {})).unwrap();
        assert!(host.requests_pending("rf"), "the admitted call has not sent its request yet");
        assert!(host.admit_local("rf", false).is_err());
        assert!(host.lend_out("rf", Arc::new(|_, _, _| {})).is_err(), "duplicate lend cannot replace its owner");
        drop(request);
        assert!(!host.requests_pending("rf"));
        host.take_back("rf");
        host.shutdown();
    }

    #[test]
    fn a_run_is_sent_each_schema_once_then_names_it() {
        let tree: Value = serde_json::from_str(r#"{"schema": "zlc_data.DatasetSchema", "repeat_domain": {"schema": "zlc_data.DomainSpec", "shape": [1], "axes": [{"schema": "zlc_data.AxisSpec", "axis_id": "p.s.repeat", "name": "repeat", "role": "repeat", "size": 1, "coordinates": null, "unit": null, "coordinate_frame": null, "index_origin": 0, "coordinate_labels": null}], "axis_codes": [[0]]}, "point_domain": {"schema": "zlc_data.DomainSpec", "shape": [1], "axes": [], "axis_codes": []}, "cell_domain": {"schema": "zlc_data.DomainSpec", "shape": [3], "axes": [{"schema": "zlc_data.AxisSpec", "axis_id": "site", "name": "Site", "role": "site", "size": 3, "coordinates": [1, 2, 3], "unit": null, "coordinate_frame": null, "index_origin": 0, "coordinate_labels": null}], "axis_codes": null}, "value_schema": {"schema": "zlc_data.ValueSchema", "validity_contract": {"mode": "VALUE", "component_axis_ids": []}, "dtype": "<f8", "value_unit": null, "name": "s"}}"#).unwrap();
        let schema = wire::schema_from_tree(&tree).unwrap();
        let mut refs = RunRefs::default();
        let first = refs.reference(&schema);
        let mut params = json!({});
        refs.define(&mut params);
        assert_eq!(params["schemas"][first.to_string()], wire::schema_tree(&schema));
        // Named again, as the same Arc or an equal schema with the same tree: nothing is sent.
        let equal = wire::schema_from_tree(&tree).unwrap();
        assert_eq!((refs.reference(&schema), refs.reference(&equal)), (first, first));
        let mut params = json!({});
        refs.define(&mut params);
        assert_eq!(params, json!({}));
        // A new source: the plugin forgets the table and the schema is sent whole again.
        refs.reset();
        let again = refs.reference(&schema);
        let mut params = json!({});
        refs.define(&mut params);
        assert_eq!((&params["refs_reset"], &params["schemas"][again.to_string()]), (&json!(true), &wire::schema_tree(&schema)));
    }
}
