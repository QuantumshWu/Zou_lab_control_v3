//! Standalone Figure slice. Rust owns commands, accepted panel/front identity,
//! the scientific document and transport. Python runs only device plugins.
// A window program: Windows gives it no console (its output: diagnostics.rs).
#![windows_subsystem = "windows"]
mod diagnostics;
mod dialog;
mod delta;
mod pixels;
mod windows;
mod science;
mod overlay;
mod panels;
mod derivation;
mod trace;
mod archive;
mod manual;
mod remote;
mod record;
mod chrome;
mod frame;
mod workbench_host;
mod animation;
mod pulse_host;
mod job;
mod bench;
mod share;
mod slm_host;
mod readers;
mod cadence;
mod installer;

use std::{collections::{BTreeMap,BTreeSet,VecDeque}, path::PathBuf, sync::{Arc,mpsc,atomic::Ordering}};
use serde_json::{Value, json};
use tao::{event::{Event, WindowEvent}, event_loop::{ControlFlow, EventLoopBuilder}, platform::windows::WindowExtWindows};
use wry::WebContext;
use zlc_core::runtime::Runtime;

/// `window`: the page a command came from (None: a remote client or the host itself).
enum AppEvent { Command { window: Option<windows::Kind>, command: Value }, Completed(science::Completion), Plugin(zlc_plugin_host::HostEvent),
    Workbench(zlc_workbench::Event), CatalogReady, RuntimeChanged, RebaseNeeded, WorkerError(String), InvalidMessage(String), DerivationError(String),
    Remote(remote::RemoteCall), WaitExpired(u64), Title { window: windows::Kind, title: String },
    /// The page-drawn title bar's buttons: hover, press, maximized (frame.rs).
    Caption { window: windows::Kind, message: Value }, Pulse(pulse_host::PulseEvent), Native(workbench_host::NativeRequest), PulsePoll,
    /// A held panel's update interval has passed (cadence::Waker).
    Beat,
    /// A window's ×, or close_window {window}: what closing it means is its kind's (v2's rules).
    CloseWindow(windows::Kind),
    /// Save image… finished: the window's request and where the picture went.
    ImageSaved { request: u64, result: Result<Value, String> },
    /// A browser asked to watch, or left (the Browser sharing card changes).
    ShareChanged,
    /// A window shown last turn becomes the active one (windows.rs `show`).
    Activate(windows::Kind),
    /// An SLM Editor's pictures drawn (slm_host's lane).
    Slm(slm_host::Drawn),
    /// A live panel's region changed: its producer is told (science.rs SelectionNote).
    Selection(science::SelectionNote),
    /// installation.exe's checks and steps (installer.rs).
    Installer(installer::Report) }

/// Write the window's picture to `path` (a .png); its path and size.
fn write_image(path: &std::path::Path, png: &[u8]) -> Result<Value, String> {
    if let Some(folder) = path.parent().filter(|folder| !folder.as_os_str().is_empty()) {
        std::fs::create_dir_all(folder).map_err(|error| format!("Cannot create {}: {error}", folder.display()))?;
    }
    std::fs::write(path, png).map_err(|error| format!("Cannot write {}: {error}", path.display()))?;
    let (width, height) = pixels::png_size(png).unwrap_or_default();
    Ok(json!({"path": path, "width": width, "height": height}))
}

/// Where a remote client's Save image goes when it names no file: today's data folder, `screenshot.png` or the next free `screenshot-<n>.png`.
fn fresh_image_path(today: &std::path::Path) -> PathBuf {
    (1..).map(|n: u32| today.join(if n == 1 { "screenshot.png".to_owned() } else { format!("screenshot-{n}.png") }))
        .find(|path| !path.exists()).expect("a free name")
}

/// What the scientific worker (science.rs, the Figure document) answers.
const SCIENCE_ACTIONS: &[&str] = &["ready", "sync_panels", "close", "describe", "get_fit", "close_panel", "save", "open", "drop_document",
    "present_archive", "pointer", "configure", "scope_step", "focus", "reset_view", "fit"];

struct PendingArchive {data:Arc<archive::MountedArchive>,bindings:panels::Panels,request:u64,document:u32}

/// One Figure Viewer's document (v2: one window per opened Figure), numbered
/// as its window is: its panels (`figure-<n>-<k>`), its accepted state and its archive.
struct Viewer {
    bindings: panels::Panels, state: Value, error: Option<String>, pending: bool,
    archive: Option<Arc<archive::MountedArchive>>,
    /// Its window is gone; its panels close, then the document is dropped.
    closing: bool,
    /// Its data working copies (Edit data / New data) and their published Applies.
    drafts: manual::Drafts,
    /// The open asked while its new window's page was not up yet: it runs once the page has given its device ratio.
    waiting: Option<Value>,
    /// A preview panel's first target (the archive recipe it starts from), until its document holds it.
    targets: BTreeMap<String, Value>,
    /// The generation each applied copy's signal publishes in.
    generations: BTreeMap<String, zlc_core::runtime::Generation>,
    /// Its panels' own outputs (an ROI of an archived image) as last seen: a new one re-syncs the document.
    derived: Vec<String>,
    /// Its board (v2 FigureViewer's ConsoleBoardView, panel_only): the panels in the order shown, each one's
    /// name, update interval and Overlay -- what Setting's Panel rows edit and a carried title reorders.
    board: Vec<ViewerPanel>,
}

/// One panel of a viewer's board: its title (v2 PanelState.title: a dataset's label, a blank panel's
/// kind, the operator's name), its update interval and its Overlay ("": off).
struct ViewerPanel { panel: String, title: Option<String>, interval_ms: u64, overlay: String }

/// Where a Figure Viewer's file dialogs start without a Figure: today's folder, made as v2's viewer makes it.
fn viewer_folder(workspace: &zlc_workbench::Workspace) -> PathBuf { workspace.day_folder().unwrap_or_else(|_| workspace.data()) }

/// A viewer's board follows its document's panels: a new one joins at the end (named as v2
/// names it), a closed one leaves; the panels are listed in the board's order.
fn settle_board(board: &mut Vec<ViewerPanel>, state: &mut Value) {
    let rows: Vec<Value> = state["panels"].as_array().cloned().unwrap_or_default();
    board.retain(|entry| rows.iter().any(|row| row["id"] == entry.panel.as_str()));
    for row in &rows {
        let Some(id) = row["id"].as_str().filter(|id| !id.starts_with(panels::EDIT_PREFIX)) else { continue };
        if board.iter().any(|entry| entry.panel == id) { continue; }
        let signal = row["binding"]["signal"].as_str().unwrap_or("");
        let title = if signal.is_empty() {
            // v2 add_blank_panel: the kind, then "kind 2", ... among the board's titles.
            let kind = row["kind"].as_str().unwrap_or("panel");
            let mut name = kind.to_owned();
            let mut suffix = 2;
            while board.iter().any(|entry| entry.title.as_deref() == Some(name.as_str())) { name = format!("{kind} {suffix}"); suffix += 1; }
            Some(name)
        } else if signal.starts_with("@figure/") {
            // A saved dataset's panel is titled with its label (v2 initial_recipe's title).
            state["signals"].as_array().into_iter().flatten().find(|item| item["id"] == signal).and_then(|item| item["label"].as_str()).map(str::to_owned)
        } else { None };
        board.push(ViewerPanel { panel: id.to_owned(), title, interval_ms: zlc_workbench::INTERVALS[0], overlay: String::new() });
    }
    // Setting's Signal picker: v2's tree of what the viewer holds (zlc_workbench::viewer_signal_tree).
    let text = |value: &Value| value.as_str().unwrap_or_default().to_owned();
    let signals: Vec<(String, String, String)> = state["signals"].as_array().into_iter().flatten()
        .map(|item| (text(&item["id"]), text(&item["owner"]), text(&item["shape"]))).collect();
    let shown: Vec<(String, String)> = rows.iter().map(|row| (text(&row["id"]), text(&row["binding"]["signal"]))).collect();
    let tree = |panel: &str| zlc_workbench::viewer_signal_tree(panel, shown.iter().find(|(id, _)| id == panel).map_or("", |(_, signal)| signal.as_str()), &signals, &shown);
    let mut ordered = rows.clone();
    ordered.sort_by_key(|row| board.iter().position(|entry| row["id"] == entry.panel.as_str()).unwrap_or(usize::MAX));
    state["panels"] = Value::Array(ordered);
    state["board"] = json!({"panels": board.iter().map(|entry| json!({"panel": entry.panel, "title": entry.title,
            "name": entry.title.clone().or_else(|| panels::viewer_caption(&entry.panel)).unwrap_or_default(), "interval_ms": entry.interval_ms,
            "overlay_signal": entry.overlay, "overlay_choices": [], "published_outputs": {}, "outputs": [], "signal_tree": tree(&entry.panel)})).collect::<Vec<_>>(),
        "paused": false, "selectors": true, "active_task": null, "frozen": {}});
}

impl Viewer {
    fn new(number: u32) -> Self {
        Self { bindings: panels::Panels::with_prefix(panels::viewer_prefix(number)), error: None, pending: false, archive: None, closing: false,
            drafts: manual::Drafts::default(), waiting: None, targets: BTreeMap::new(), generations: BTreeMap::new(), derived: Vec::new(), board: Vec::new(),
            state: json!({"type":"state", "request":0, "document":{"path":"", "title":"Figure Viewer"}, "panels":[], "signals":[], "error":null}) }
    }

    fn settle_board(&mut self) { settle_board(&mut self.board, &mut self.state); }

    /// Setting's Panel rows and a carried title, on this viewer's board (v2 panel_only ConsolePresenter).
    fn board_command(&mut self, action: &str, command: &Value) -> Result<Value, String> {
        if action == "order_panels" {
            let wanted: Vec<String> = command["order"].as_array().ok_or("order is required")?.iter().filter_map(Value::as_str).map(str::to_owned).collect();
            let mut rest = std::mem::take(&mut self.board);
            let mut ordered = Vec::with_capacity(rest.len());
            for id in &wanted { if let Some(at) = rest.iter().position(|entry| &entry.panel == id) { ordered.push(rest.remove(at)); } }
            ordered.extend(rest);
            self.board = ordered;
            self.settle_board();
            return Ok(json!({"order": self.board.iter().map(|entry| &entry.panel).collect::<Vec<_>>()}));
        }
        let panel = command["panel"].as_str().ok_or("Name the panel")?;
        let entry = self.board.iter_mut().find(|entry| entry.panel == panel).ok_or_else(|| format!("No panel {panel:?} on the board"))?;
        let answer = match action {
            "rename_panel" => {
                let title = command["title"].as_str().map(|title| title.split_whitespace().collect::<Vec<_>>().join(" ")).filter(|title| !title.is_empty());
                entry.title = title.clone();
                json!({"panel": panel, "title": title})
            }
            "set_panel_interval" => {
                let interval = command["interval_ms"].as_u64().ok_or("interval_ms is required")?;
                if !zlc_workbench::INTERVALS.contains(&interval) { return Err("The update interval is 100, 200, 400 or 800 ms".into()); }
                entry.interval_ms = interval;
                json!({"panel": panel, "interval_ms": interval})
            }
            _ => {
                // A saved Figure's image keeps the rings it was saved with; no judgement of it runs here.
                let signal = command["signal"].as_str().unwrap_or("");
                if !signal.is_empty() { return Err(format!("{signal} judges no sites of this panel's image")); }
                entry.overlay.clear();
                json!({"panel": panel, "overlay_signal": ""})
            }
        };
        self.settle_board();
        Ok(answer)
    }
}

/// Which entry this run is: experiment.exe (Devices@ first, v2's experiment.bat), figure_viewer.exe
/// (one FigureViewer@ alone, an archive to open) or pulse_editor.exe (PulseGUI@ alone, a pulse to open)
/// (installation.exe, --installer, is installer.rs's window alone.)
#[derive(Clone, PartialEq)]
enum Mode { Experiment, FigureViewer(Option<String>), PulseEditor(Option<String>) }

/// The Pulse Editor's drafts a close would throw away, in v2's order and words (PulseGUI's close guard).
fn pulse_unsaved(state: &Value) -> Vec<&'static str> {
    [("Pulse", &state["dirty"]), ("Config", &state["config"]["dirty"]), ("Subpulse", &state["subpulse"]["dirty"])]
        .into_iter().filter(|(_, dirty)| **dirty == true).map(|(what, _)| what).collect()
}

/// Whether any document shows `panel`.
/// A run taking or leaving the session's sequencer is the Pulse Editor's news on this turn,
/// not at its next poll (its On Pulse is enabled the moment the board is free).
fn tell_editor_claims(pulse: &mut pulse_host::PulseHost, workbench: &zlc_workbench::Workbench) {
    let claimant = pulse.apparatus_device().and_then(|device| workbench.device_claimant(&device));
    if pulse.editor.set_claimed_by(claimant) { pulse.editor.poll(); pulse.dispatch(); }
}

/// A pointer resting or leaving, no button held: it only says where the pointer is.
fn hovering(command: &Value) -> bool {
    command["action"] == "pointer" && (command["event"] == "leave" || (command["event"] == "move" && command["button"].is_null()
        && command["buttons"].as_array().is_none_or(|buttons| buttons.is_empty())))
}

fn shows_panel(accepted: &Value, viewers: &BTreeMap<u32, Viewer>, panel: &Value) -> bool {
    std::iter::once(accepted).chain(viewers.values().map(|viewer| &viewer.state))
        .any(|state| state["panels"].as_array().is_some_and(|rows| rows.iter().any(|row| row["id"] == *panel)))
}

/// The Figure Viewer's accepted document as the page and remote clients get it:
/// its own section, beside the live board's panels. The document goes into the
/// message and back, never copied: it carries the opened Figure's whole record.
fn viewer_message<T>(viewer: &mut Value, deliver: impl FnOnce(&Value) -> T) -> T {
    let mut message = json!({"type":"state", "request":viewer["request"], "error":viewer["error"], "result":viewer["result"]});
    message["viewer"] = std::mem::take(viewer);
    let delivered = deliver(&message);
    *viewer = message["viewer"].take();
    delivered
}

/// Whether `kind` is a window the operator has: a Figure Viewer until it closes.
fn listed(kind: &windows::Kind, viewers: &BTreeMap<u32, Viewer>) -> bool {
    match kind { windows::Kind::FigureViewer(number) => viewers.get(number).is_some_and(|viewer| !viewer.closing), _ => true }
}

/// get_state windows: the windows the operator has, in the order they were made.
fn listed_windows(windows: &windows::Windows, viewers: &BTreeMap<u32, Viewer>) -> Value {
    let mut rows = windows.describe();
    if let Some(rows) = rows.as_array_mut() { rows.retain(|row| row["window"].as_str().and_then(windows::Kind::parse).is_some_and(|kind| listed(&kind, viewers))); }
    rows
}

/// Whether `device` is open in the apparatus (its control and log windows live while it is).
fn live_device(state: &Value, device: &str) -> bool {
    state["devices"].as_array().is_some_and(|rows| rows.iter().any(|row| row["instance_id"] == device && row["state"] == "open"))
}

/// The Figure Viewer a new Figure opens in: a new one, its window made as it is shown (v2: one per Figure).
fn take_viewer(viewers: &mut BTreeMap<u32, Viewer>, next: &mut u32) -> u32 {
    let number = *next;
    *next += 1;
    viewers.insert(number, Viewer::new(number));
    number
}

/// v2 `_apply_data_draft`, the viewer's part: publish the copy under its own
/// producer (descending from the dataset or Apply it edits) and show it in its
/// preview panel -- the one it had while that kind can still show the shape,
/// else a new one started from the archive's recipe or the kind the data fits.
/// Returns the panel a replaced preview leaves, to be closed.
fn apply_data(runtime: &Runtime, viewer: &mut Viewer, apply: manual::Apply) -> Result<Option<String>, String> {
    use zlc_core::runtime::{Bundle, Declaration, Input, Storage};
    let schema = apply.snapshot.schema().clone();
    let plot = zlc_plot::configuration::PlotSchema::new(schema.clone());
    let fits = |kind: &str| zlc_plot::kinds::PlotKind::parse(kind)
        .is_some_and(|kind| zlc_plot::configuration::fitting_spec(Some(&plot), Some(kind), None).ok().flatten().is_some());
    let shown = viewer.state["panels"].as_array().and_then(|rows| apply.panel.as_ref().and_then(|panel| rows.iter().find(|row| row["id"] == panel.as_str())))
        .and_then(|row| row["kind"].as_str()).map(str::to_owned);
    let keep = apply.panel.clone().filter(|panel| viewer.bindings.has(panel) && shown.as_deref().is_none_or(fits));
    let kind = match (&keep, &apply.recipe) {
        (Some(_), _) => String::new(),
        (None, Some(recipe)) => recipe["semantic"]["kind"].as_str().unwrap_or("image").to_owned(),
        (None, None) => zlc_plot::configuration::fitting_spec(Some(&plot), None, None).map_err(|error| error.to_string())?
            .ok_or("no plot kind can show data of this shape")?.kind().value().to_owned(),
    };
    let parents: Vec<_> = apply.parent.as_ref().and_then(|signal| viewer.drafts.published.get(signal).cloned()
        .or_else(|| viewer.archive.as_ref().and_then(|archive| archive.items().iter().find(|item| *item.signal == **signal).and_then(|item| item.publication.clone()))))
        .into_iter().collect();
    let signal: Arc<str> = apply.signal.as_str().into();
    let operation = apply.metadata["operation"].clone();
    let generation = runtime.begin(apply.owner.clone(), vec![Declaration { signal: signal.clone(), schema, storage: Storage::Snapshot }], apply.metadata)
        .map_err(|error| error.to_string())?;
    let published = runtime.commit(&generation, &Bundle { inputs: vec![Input::Dataset { signal: signal.clone(), ordinal: None, snapshot: apply.snapshot.clone() }],
        metadata: json!({"operation": operation}), parents }).map_err(|error| error.to_string())?;
    runtime.seal(&generation, json!({"reason": "manual data"})).map_err(|error| error.to_string())?;
    if let Some(publication) = published.publications.iter().find(|publication| publication.key().signal == signal) {
        viewer.drafts.published.insert(apply.signal.clone(), publication.clone());
    }
    viewer.generations.insert(apply.signal.clone(), generation);
    let (panel, replaced) = match keep {
        Some(panel) => { viewer.bindings.sync_generations(runtime); (panel, None) }
        None => {
            let panel = viewer.bindings.add(runtime, &apply.signal, &kind)?;
            if let Some(recipe) = &apply.recipe { viewer.targets.insert(panel.clone(), recipe.clone()); }
            let replaced = apply.panel.filter(|old| viewer.bindings.remove(old));
            (panel, replaced)
        }
    };
    viewer.bindings.set_layer(&panel, apply.overlay.clone());
    let epoch = viewer.bindings.epoch(&panel).unwrap_or(1);
    viewer.drafts.epochs.insert(apply.editor.clone(), epoch);
    if let Some(draft) = viewer.drafts.get_mut(&apply.editor) { draft.applied_as(apply.snapshot, panel); }
    viewer.pending = true;
    Ok(replaced)
}

/// v2 `_discard_applied_data`, the viewer's part: the preview goes and the copy's producer retires.
fn discard_data(runtime: &Runtime, viewer: &mut Viewer, panel: Option<String>, signal: &str) -> Option<String> {
    if let Some(generation) = viewer.generations.remove(signal) { let _ = runtime.retire(&generation); }
    viewer.drafts.published.remove(signal);
    viewer.pending = true;
    panel.filter(|panel| viewer.bindings.remove(panel))
}

fn panel_ids(state: &Value) -> Vec<String> {
    state["panels"].as_array().into_iter().flatten().filter_map(|row| row["id"].as_str().map(str::to_owned)).collect()
}

/// What every window of a run is made with: the WebView2 profile in the checkout's .local, the
/// page, the browser's arguments and, for automated runs, where the windows are parked.
fn window_setup() -> Result<(WebContext, windows::Setup), Box<dyn std::error::Error>> {
    let profile = zlc_core::checkout::local("webview2");
    std::fs::create_dir_all(&profile)?;
    let context = WebContext::new(Some(profile));
    // Automated runs park the shown windows off every screen (ZLC_V3_WINDOW_AT=x,y);
    // a hidden window would be throttled, so occlusion is not computed then.
    let parked = std::env::var("ZLC_V3_WINDOW_AT").ok().and_then(|text| {
        let (x, y) = text.split_once(',')?;
        Some(tao::dpi::PhysicalPosition::new(x.trim().parse::<i32>().ok()?, y.trim().parse::<i32>().ok()?))
    });
    // The page: a release's own copy beside it (bin\engine\ui, built with it), else the checkout's.
    let beside = std::env::current_exe()?.with_file_name("ui");
    let ui_root = if beside.join("index.html").is_file() { beside } else { zlc_core::checkout::path("apps/desktop/ui") }.canonicalize()?;
    let debug_port = std::env::var("ZLC_V3_DEBUG_PORT").ok();
    // wry's own defaults, plus an sRGB output: the page's colours (the zlc_ui
    // tokens and every finished plot pixel) reach the screen as their values,
    // as Qt's do, instead of being converted to the monitor profile.
    // DeferRendererTasksAfterInput held the core's answer to a click until the
    // frame after the click was presented: every clicked command showed its
    // result a frame later than it had arrived (~25 ms instead of ~10).
    let mut browser_args = String::from("--disable-features=msWebOOUI,msPdfOOUI,msSmartScreenProtection,DeferRendererTasksAfterInput");
    if parked.is_some() { browser_args.push_str(",CalculateNativeWinOcclusion"); }
    browser_args.push_str(" --force-color-profile=srgb");
    if let Some(port) = &debug_port {
        let port: u16 = port.parse()?;
        browser_args.push_str(&format!(" --remote-debugging-port={port} --remote-debugging-address=127.0.0.1"));
    }
    // An instrument window, not a browser: F5/Ctrl+R/Ctrl+F/Ctrl+P/F12 would
    // reload or decorate the page (editing and clipboard keys still work).
    Ok((context, windows::Setup { ui_root, browser_args, devtools: debug_port.is_some(), parked }))
}

fn main() {
    trace::point("startup.main", || json!({"created_ns": trace::process_created_ns()}));
    let routed = diagnostics::route(if std::env::args().any(|arg| arg == "--installer") { "installation" } else { "zlc-desktop" });
    // `run` returns only when the application cannot start; its event loop ends the process.
    let log = routed.log.as_ref().map(|path| path.display().to_string());
    let error = match std::panic::catch_unwind(move || run(log)) {
        Ok(Ok(())) => return,
        Ok(Err(error)) => error.to_string(),
        Err(panic) => panic.downcast_ref::<String>().cloned().or_else(|| panic.downcast_ref::<&str>().map(|text| text.to_string())).unwrap_or_default(),
    };
    routed.fatal(&error);
    std::process::exit(1);
}

fn run(log: Option<String>) -> Result<(), Box<dyn std::error::Error>> {
    // Latency-sensitive scientific work must not depend on window-focus power
    // heuristics. This changes neither priority, affinity nor system power policy.
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, SetProcessInformation,
        ProcessPowerThrottling, PROCESS_POWER_THROTTLING_STATE,
        PROCESS_POWER_THROTTLING_CURRENT_VERSION, PROCESS_POWER_THROTTLING_EXECUTION_SPEED};
    let power=PROCESS_POWER_THROTTLING_STATE {Version:PROCESS_POWER_THROTTLING_CURRENT_VERSION,
        ControlMask:PROCESS_POWER_THROTTLING_EXECUTION_SPEED,StateMask:0};
    if unsafe {SetProcessInformation(GetCurrentProcess(),ProcessPowerThrottling,
        (&power as *const PROCESS_POWER_THROTTLING_STATE).cast(),std::mem::size_of_val(&power) as u32)}==0 {
        eprintln!("Scientific HighQoS unavailable: {}",std::io::Error::last_os_error());
    }
    // One bounded pool for every panel, fit, projection and raster band:
    // panels run side by side and none waits on another's interpreter.
    rayon::ThreadPoolBuilder::new().thread_name(|index| format!("zlc-pool-{index}")).build_global()?;
    let usage = "Usage: zlc-desktop [--open Figure.npz | --figure-viewer [Figure.npz] | --pulse-editor [pulse.json] | --installer] [--workspace directory] [--mcp-read-only]";
    let (mut initial_path, mut workspace_root) = (None, std::env::var_os("ZLC_WORKSPACE").map(PathBuf::from));
    // v2's MCP switch: AI clients may drive the instrument as the operator does,
    // unless it starts with --mcp-read-only (then they only read).
    let mut mcp_control = true;
    let mut mode = Mode::Experiment;
    let mut installing = false;
    let mut args = std::env::args().skip(1).peekable();
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--open" => initial_path = Some(args.next().ok_or(usage)?),
            // v2's standalone apps: no apparatus, no Devices@; the file to open is optional.
            "--figure-viewer" => mode = Mode::FigureViewer(args.next_if(|arg| !arg.starts_with("--"))),
            "--pulse-editor" => mode = Mode::PulseEditor(args.next_if(|arg| !arg.starts_with("--"))),
            "--installer" => installing = true,
            "--workspace" => workspace_root = Some(PathBuf::from(args.next().ok_or(usage)?)),
            "--mcp-read-only" => mcp_control = false,
            _ => return Err(usage.into()),
        }
    }
    let event_loop = EventLoopBuilder::<AppEvent>::with_user_event().build();
    // installation.exe: its window alone, no instrument behind it.
    if installing { let (context, setup) = window_setup()?; return installer::run(event_loop, context, setup, log); }
    // The plugins' interpreter (installation.exe's resolver answers) is asked while the first window is made.
    std::thread::spawn(|| { let _ = zlc_plugin_host::plugin_python(); });
    let proxy = event_loop.create_proxy();
    let worker_proxy = proxy.clone();
    let runtime = Runtime::new();
    let scientific_runtime = runtime.clone();
    let derivation_proxy = proxy.clone();
    let prepared_proxy=proxy.clone();
    let selection_proxy=proxy.clone();
    let (sender, receiver) = mpsc::sync_channel::<science::Job>(1);
    let worker = std::thread::Builder::new().name("zlc-science".into()).spawn(move || {
        let mut science = science::Science::new(scientific_runtime,Arc::new(move |error| {
            let _ = derivation_proxy.send_event(AppEvent::DerivationError(error));
        }),Arc::new(move || {let _=prepared_proxy.send_event(AppEvent::RebaseNeeded);}),
            Arc::new(move |note| {let _=selection_proxy.send_event(AppEvent::Selection(note));}));
        for job in receiver {
            let closing = job.command["action"] == "close";
            // A defect inside the document leaves its state unknown: report it and stop.
            let mut result = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| science.execute(&job))) {
                Ok(result) => result,
                Err(panic) => {
                    let reason = panic.downcast_ref::<String>().cloned().or_else(|| panic.downcast_ref::<&str>().map(|text| text.to_string()));
                    let _ = worker_proxy.send_event(AppEvent::WorkerError(format!("Scientific worker failed: {}", reason.unwrap_or_default())));
                    break;
                }
            };
            let (feedback,decision)=mpsc::channel();
            result.decision=Some(feedback);
            let delivered=worker_proxy.send_event(AppEvent::Completed(result)).is_ok();
            let decision=if delivered {decision.recv().unwrap_or_default()}else{science::PresentationDecision::default()};
            science.finish_presentation(decision);
            if !delivered || closing {break;}
        }
    })?;
    let mut worker = Some(worker);
    let (context, setup) = window_setup()?;
    let debug_port = std::env::var("ZLC_V3_DEBUG_PORT").ok();
    // Browser sharing serves the same page, read-only, to the lab network (off until Start sharing).
    let share_proxy = std::sync::Mutex::new(event_loop.create_proxy());
    let share = share::Share::new(setup.ui_root.clone(), move || { if let Ok(proxy) = share_proxy.lock() { let _ = proxy.send_event(AppEvent::ShareChanged); } });
    // v2's experiment entry opens Devices@Zou lab; the other windows follow it.
    let mut windows = windows::Windows::new(context, setup, proxy.clone());
    windows.set_share(share.clone());
    // A standalone app's file, or experiment.exe's --open Figure, is opened once the first window's page is up.
    let mut opened = false;
    let mut initialized = false;
    let watch = runtime.watch_changes()?;
    let pending = watch.pending.clone();
    let watch_proxy = event_loop.create_proxy();
    let mut watch_listener = Some(std::thread::Builder::new().name("zlc-runtime-watch".into()).spawn(move || {
        while watch.wait().is_ok() {
            if watch_proxy.send_event(AppEvent::RuntimeChanged).is_err() { break; }
        }
    })?);
    let mut accepted = json!({"type":"state", "request":0, "document":{"path":"", "title":"Figure Viewer"}, "panels":[], "error":null,
        "acquisition":{"state":"idle","produced":0}});
    // The live document's panel rows as its last accepted job answered them, before they were decorated.
    let mut answered_rows = Value::Null;
    let mut queue: VecDeque<Value> = VecDeque::new();
    let mut active = false;
    let mut closing = false;
    let mut worker_failed = false;
    let mut sequence = 0u64;
    let mut ticket = 0u64;
    let mut bindings = panels::Panels::new();
    // The live board's display cadence: each panel's update interval.
    let mut cadence = cadence::Cadence::default();
    // Whether the worker's last job was the live board's data: a waiting pointer goes next.
    let mut board_last = false;
    let beat = { let proxy = event_loop.create_proxy(); cadence::Waker::new(move || { let _ = proxy.send_event(AppEvent::Beat); })? };
    // The Figure Viewer's own document (v2's FigureViewer window): an opened
    // archive's panels and their accepted state; the live board runs beside it.
    let mut viewers: BTreeMap<u32, Viewer> = BTreeMap::new();
    let mut next_viewer: u32 = 1;
    // The first window: Devices@ (the others follow it), or a standalone app's one window.
    let standalone = mode != Mode::Experiment;
    match &mode {
        Mode::Experiment => windows.make(&windows::Kind::Devices, &event_loop, true)?,
        Mode::FigureViewer(_) => {
            viewers.insert(next_viewer, Viewer::new(next_viewer));
            windows.make(&windows::Kind::FigureViewer(next_viewer), &event_loop, true)?;
            next_viewer += 1;
        }
        Mode::PulseEditor(_) => windows.make(&windows::Kind::PulseEditor, &event_loop, true)?,
    }
    // The operator agreed to discard the Pulse Editor's drafts: its close goes ahead.
    let mut discard_pulse = false;
    let mut reset_document = false;
    let mut pending_archive:Option<PendingArchive>=None;
    let mut user_error: Option<String> = None;
    // What Stop all drives safe: the pulse board and every open device's outputs.
    let safe_actions = zlc_core::control::SafeActions::new();
    accepted["share"] = share.state();
    // Device plugins run in their own processes; the core holds no interpreter.
    // Their Python is the checkout's science environment (ZLC_PLUGIN_PYTHON overrides),
    // its environment set by the host: started from Explorer, nothing else is needed.
    // Devices@ lists the plugin types: discovery runs beside its window (from the
    // catalogue kept for these sources and this environment, else Python). Nothing
    // else is started ahead: a plugin process starts when its device opens.
    let science_src = zlc_core::checkout::path("science/src");
    let plugin_proxy = std::sync::Mutex::new(event_loop.create_proxy());
    let plugins = zlc_plugin_host::HostConfig::from_env(science_src).map(|mut config| {
        // Further plugin packages beside zlc_atom (a site's own): "package=source root;...".
        for entry in std::env::var("ZLC_PLUGIN_PACKAGES").unwrap_or_default().split(';').filter(|entry| !entry.is_empty()) {
            if let Some((package, root)) = entry.split_once('=') { config.packages.push(package.into()); config.python_path.push(root.into()); }
        }
        config
    }).and_then(|config| zlc_plugin_host::PluginHost::new(
        config, runtime.clone(), safe_actions.clone(), move |event| {
            if let Ok(proxy) = plugin_proxy.lock() { let _ = proxy.send_event(AppEvent::Plugin(event)); }
        }));
    match &plugins {
        Ok(host) if !standalone => {
            let (warming, ready) = (host.clone(), std::sync::Mutex::new(event_loop.create_proxy()));
            std::thread::Builder::new().name("zlc-plugin-warm".into()).spawn(move || {
                let discovery = trace::span("startup.discovery", || json!({}));
                if let Err(error) = warming.catalog() { eprintln!("{}", json!({"type":"plugin-error","error":error})); }
                drop(discovery);
                if let Ok(proxy) = ready.lock() { let _ = proxy.send_event(AppEvent::CatalogReady); }
            })?;
        }
        Ok(_) => {}
        Err(error) => eprintln!("{}", json!({"type":"plugin-error","error":error})),
    }
    // The application composition (devices, logic, runs, board), driving the
    // plugin host through workbench_host::Builtin.
    let (reply_proxy, core_proxy) = (std::sync::Mutex::new(event_loop.create_proxy()), std::sync::Mutex::new(event_loop.create_proxy()));
    let builtin = workbench_host::Builtin::new(runtime.clone(), plugins.as_ref().ok().cloned(), Arc::new(move |event| {
        if let Ok(proxy) = reply_proxy.lock() { let _ = proxy.send_event(AppEvent::Workbench(event)); }
    }), Arc::new(move |call| core_proxy.lock().map_err(|_| "instrument closing".to_owned())?
        .send_event(AppEvent::Remote(call)).map_err(|_| "the instrument window has closed".to_owned())));
    // No science environment: the Device Manager says so for every plugin family.
    if let Err(error) = &plugins { builtin.host_missing(error.clone()); }
    // Each open SLM Editor's pictures, drawn off the window's turn.
    let slm_proxy = std::sync::Mutex::new(event_loop.create_proxy());
    let mut slm = slm_host::SlmHost::new(Arc::new(move |drawn| { if let Ok(proxy) = slm_proxy.lock() { let _ = proxy.send_event(AppEvent::Slm(drawn)); } }));
    let native_proxy = event_loop.create_proxy();
    builtin.set_native(pulse_host::device_descriptors(), std::rc::Rc::new(move |request| { let _ = native_proxy.send_event(AppEvent::Native(request)); }));
    // This computer on the bench fabric: what the fabric does with a device goes in its log.
    let bench_proxy = std::sync::Mutex::new(event_loop.create_proxy());
    builtin.set_bench(bench::Bench::new(Arc::new(move |instance, line| if let Ok(proxy) = bench_proxy.lock() {
        let _ = proxy.send_event(AppEvent::Workbench(zlc_workbench::Event::Log { instance: instance.to_owned(), line: line.to_owned() }));
    })));
    let poll_proxy = std::sync::Mutex::new(event_loop.create_proxy());
    std::thread::Builder::new().name("zlc-pulse-poll".into()).spawn(move || loop {
        std::thread::sleep(std::time::Duration::from_millis(500));
        if poll_proxy.lock().map_err(|_| ()).and_then(|proxy| proxy.send_event(AppEvent::PulsePoll).map_err(|_| ())).is_err() { break; }
    })?;
    // v2 Workspace.default: --workspace or ZLC_WORKSPACE, else the checkout's own untracked
    // <checkout>/workspace (.gitignore), made on demand with each kind's folder.
    let mut workbench = zlc_workbench::Workbench::new(&workspace_root.unwrap_or_else(|| zlc_core::checkout::path("workspace")), Box::new(builtin.clone()))?;
    // The pulse board and the Pulse Editor; the sequencer device types are native.
    let pulse_proxy = std::sync::Mutex::new(event_loop.create_proxy());
    let mut pulse = pulse_host::PulseHost::new(Arc::new(move |event| {
        if let Ok(proxy) = pulse_proxy.lock() { let _ = proxy.send_event(AppEvent::Pulse(event)); }
    }), safe_actions.clone(), plugins.as_ref().ok().cloned(), &workbench.workspace().pulses());
    pulse.users = workbench.device_users();
    // A pulse document chosen as a node's workspace resource resolves to its
    // description (PULSE_V3 `pulse_describe`), read by the one pulse codec.
    workbench.add_resource_resolver("zlc.pulse/", Arc::new(|path| zlc_pulse::run::describe_document(path).map_err(|error| error.to_string())
        .and_then(|description| serde_json::from_str(&description.compact_text()).map_err(|error| error.to_string()))));
    if let Some(missing) = zlc_workbench::COMMANDS.iter().find(|name| zlc_bus::catalog::find(name).is_none()) {
        return Err(format!("Workbench command {missing} is not in the catalogue").into());
    }
    let mut pending_targets: std::collections::BTreeMap<String, Value> = Default::default();
    let tools: remote::Tools = Default::default();
    workbench_host::decorate(&workbench, &runtime, &bindings, &pending_targets, &tools, &mut accepted);
    let remote_proxy = std::sync::Mutex::new(event_loop.create_proxy());
    let root = workbench.workspace().root.clone();
    let (endpoint, endpoint_record, connection_record) = remote::start_endpoint(move |call| remote_proxy.lock().map_err(|_| "instrument closing".to_owned())?
        .send_event(AppEvent::Remote(call)).map_err(|_| "the instrument window has closed".to_owned()), mcp_control, tools.clone(), root.clone(), runtime.clone())?;
    let self_proxy = event_loop.create_proxy();
    let mut remote_pending: std::collections::BTreeMap<u64, remote::Pending> = Default::default();
    let mut waiters: Vec<remote::Waiter> = Vec::new();
    let mut latest_fronts: std::collections::BTreeMap<String, science::Front> = Default::default();
    let mut next_remote = remote::FIRST_REMOTE_REQUEST;
    let first = &windows.get(&match &mode { Mode::Experiment => windows::Kind::Devices, Mode::FigureViewer(_) => windows::Kind::FigureViewer(1), Mode::PulseEditor(_) => windows::Kind::PulseEditor })
        .expect("the first window").window;
    println!("{}", json!({"type":"native-window", "hwnd":first.hwnd(), "dpr":first.scale_factor(),
        "inner_size":[first.inner_size().width,first.inner_size().height], "debug_port":debug_port,
        "mcp":{"url":format!("http://{}/mcp", endpoint.address), "record":endpoint_record}}));
    event_loop.run(move |event, target, flow| {
        let _keep_alive = &endpoint;
        *flow = ControlFlow::Wait;
        let error_request = match &event {
            Event::UserEvent(AppEvent::Command { command, .. }) => command.get("id").cloned(),
            _ => None,
        };
        // Opt-in timing (ZLC_V3_TRACE): one span per application event turn.
        let _turn = trace::span("ui.turn", || json!({"event": match &event {
            Event::UserEvent(AppEvent::Command { command, .. }) => command["action"].clone(),
            Event::UserEvent(AppEvent::Completed(_)) => json!("completed"),
            Event::UserEvent(AppEvent::Pulse(_)) => json!("pulse-event"),
            Event::UserEvent(AppEvent::Plugin(_) | AppEvent::Workbench(_)) => json!("workbench-event"),
            Event::UserEvent(AppEvent::RuntimeChanged) => json!("runtime"),
            Event::UserEvent(_) => json!("other"),
            _ => json!("window"),
        }}));
        let result = (|| -> Result<(), String> {
            match event {
                Event::WindowEvent { window_id, event: WindowEvent::CloseRequested, .. } => {
                    if let Some(kind) = windows.kind_of(window_id) { let _ = self_proxy.send_event(AppEvent::CloseWindow(kind)); }
                }
                Event::UserEvent(AppEvent::CloseWindow(kind)) => {
                    match Some(kind) {
                        // A viewer's window takes its document with it (v2: one Figure, one window).
                        Some(kind @ windows::Kind::FigureViewer(_)) => {
                            windows.close(&kind);
                            let windows::Kind::FigureViewer(number) = kind else { unreachable!() };
                            if let Some(viewer) = viewers.get_mut(&number) {
                                viewer.closing = true;
                                for panel in panel_ids(&viewer.state) { queue.push_back(json!({"id":0,"action":"close_panel","panel":panel,"document":number})); }
                                queue.push_back(json!({"id":0,"action":"drop_document","document":number}));
                            }
                            // A standalone viewer's application ends with its last window, as v2's QApplication did.
                            if !standalone || windows.any_visible() { return Ok(()); }
                        }
                        // The Pulse Editor is a device's control: closing hides it (v2 hide_on_close).
                        Some(windows::Kind::PulseEditor) if !standalone => { windows.hide(&windows::Kind::PulseEditor); return Ok(()); }
                        // Standalone (v2 apps/pulse_editor.py): its close asks about each unsaved draft first.
                        Some(windows::Kind::PulseEditor) => {
                            let unsaved = pulse_unsaved(&pulse.state());
                            if !unsaved.is_empty() && !discard_pulse {
                                return windows.send_to(&windows::Kind::PulseEditor, &json!({"type": "close_requested", "unsaved": unsaved}));
                            }
                            discard_pulse = false;
                            windows.hide(&windows::Kind::PulseEditor);
                            // The application ends with its last window, as v2's QApplication did.
                            if windows.any_visible() { return Ok(()); }
                        }
                        // A device's control hides (v2 hide_on_close); its log is over (v2 retires it).
                        Some(kind @ windows::Kind::Control(_)) => { windows.hide(&kind); return Ok(()); }
                        Some(kind @ windows::Kind::Log(_)) => { windows.close(&kind); return Ok(()); }
                        // Closing an SLM Editor ends it (v2: its window closed for good, the draft with it).
                        Some(windows::Kind::SlmEditor(device)) => {
                            windows.close(&windows::Kind::SlmEditor(device.clone()));
                            slm.forget(&device);
                            if workbench.handles_command(&format!("{device}__editor_close")) {
                                let _ = self_proxy.send_event(AppEvent::Command { window: None, command: json!({"id": 0, "action": format!("{device}__editor_close")}) });
                            }
                            return Ok(());
                        }
                        // Closing a review rejects it, as v2's dialog did: the run is told Stop, then it goes.
                        Some(windows::Kind::Review(node)) => {
                            let request = accepted["logic"].as_array().and_then(|rows| rows.iter().find(|row| row["node_id"] == node.as_str()))
                                .map(|row| row["run"]["request"]["request"].clone()).unwrap_or(Value::Null);
                            if !request.is_null() {
                                let _ = self_proxy.send_event(AppEvent::Command { window: None, command: json!({"id": 0, "action": "answer_request", "node": node, "request": request, "response": null}) });
                            }
                            let _ = self_proxy.send_event(AppEvent::Command { window: None, command: json!({"id": 0, "action": "close_point_review", "node": node}) });
                            return Ok(());
                        }
                        // Devices and TaskConsole close the experiment (v2 close guards).
                        _ => {}
                    }
                    if worker_failed { *flow = ControlFlow::Exit; return Ok(()); }
                    if !closing {
                        closing = true;
                        tools.set_closing();
                        builtin.stop_all();
                        pulse.shutdown();
                        queue.clear();
                        queue.push_back(json!({"id":0,"action":"close"}));
                    }
                }
                Event::UserEvent(AppEvent::Command { window: from, mut command }) if !closing => {
                    let action = command["action"].as_str().ok_or("Command has no action")?.to_owned();
                    if matches!(action.as_str(), "map_ready"|"frame_consumed"|"mappings_released") {
                        // A page's own mappings: they are its window's.
                        return match from.as_ref().and_then(|kind| windows.get_mut(kind)) { Some(window) => window.pixels.acknowledge(&command), None => Ok(()) };
                    }
                    if action == "frontend_error" {
                        let error = format!("Frontend: {}", command["error"].as_str().unwrap_or("unknown presentation error"));
                        eprintln!("{}",json!({"type":"frontend-error","error":error}));
                        accepted["error"] = json!(error);
                        // Do not echo the state which failed to reconcile: that
                        // would turn one view error into an IPC feedback loop.
                        return match &from { Some(kind) => windows.send_to(kind, &json!({"type":"error","request":command["id"],"error":error})), None => Ok(()) };
                    }
                    if action == "pointer" && command["panel"] == pulse_host::PREVIEW_PANEL {
                        // The Pulse Editor's preview is the pulse host's plot, not a board panel.
                        pulse.pointer(command);
                        return Ok(());
                    }
                    if action == "pointer" && command["panel"].as_str().and_then(slm_host::device_of).is_some() {
                        // An SLM Editor's plots are its own document's.
                        slm.pointer(command);
                        return Ok(());
                    }
                    if action == "pointer" && !shows_panel(&accepted, &viewers, &command["panel"]) {
                        // The page can cancel a captured pointer while removing
                        // its retired panel. That cancellation has no new owner.
                        return Ok(());
                    }
                    // Every catalogue action is validated the same way whoever sent it
                    // (a read-only endpoint already refused a remote change).
                    let origin = remote::take_origin(&mut command);
                    if let Some(spec) = zlc_bus::catalog::find(&action) {
                        spec.validate(&remote::parameters(&command)).map_err(|error| format!("Invalid {action}: {error}"))?;
                    } else if workbench.handles_command(&action) {
                    } else if !origin.is_operator() {
                        return Err(format!("{action} is not a command remote clients can send"));
                    }
                    // An SLM Editor's window speaks of its picture and its file dialogs;
                    // the device hears its own editor_* command (what an MCP client sends).
                    let (action, mut command) = match action.as_str() {
                        "slm_paint" => { crate::trace::point("slm.paint", || json!({"id": command["id"]})); let paint = slm.paint(&command)?; (paint["action"].as_str().unwrap_or("").to_owned(), paint) }
                        "slm_file" => {
                            let device = command["device"].as_str().ok_or("Name the SLM")?.to_owned();
                            let file = command["file"].as_str().ok_or("Name the file action")?.to_owned();
                            let spec = slm.file(&device, &file).ok_or_else(|| format!("{device}'s SLM Editor has no file action {file}"))?;
                            // v2 SlmEditor._choose: its caption and filter, offered in the workspace's data.
                            let data = workbench.workspace().data();
                            let chosen = dialog::file(windows.hwnd(from.as_ref()), spec["title"].as_str().unwrap_or(""), spec["filter"].as_str().unwrap_or(""),
                                                      spec["save"] == true, &data.join(spec["name"].as_str().unwrap_or("")), &data)?;
                            let Some(path) = chosen else {
                                return windows.send(&json!({"type": "state", "request": command["id"], "error": null, "result": null}));
                            };
                            let named = format!("{device}__editor_{file}");
                            (named.clone(), json!({"id": command["id"], "action": named, "path": path}))
                        }
                        _ => (action, command),
                    };
                    if action == "close_window" && command.get("window").is_some_and(|window| !window.is_null()) {
                        // One window, as its × closes it (MCP addresses it by name).
                        let kind = command["window"].as_str().and_then(windows::Kind::parse).filter(|kind| windows.get(kind).is_some() && listed(kind, &viewers))
                            .ok_or_else(|| format!("No open window {}", command["window"]))?;
                        self_proxy.send_event(AppEvent::CloseWindow(kind)).map_err(|_| "event loop closed".to_owned())?;
                        let state = json!({"type":"state", "request": command["id"], "error": null, "result": {"closed": command["window"]}});
                        remote::settle(&state, &mut remote_pending);
                        return windows.send(&state);
                    }
                    if action == "close_pulse_editor" && origin.is_operator() {
                        // The page asked about each unsaved draft and the operator chose Discard.
                        discard_pulse = true;
                        return self_proxy.send_event(AppEvent::CloseWindow(windows::Kind::PulseEditor)).map_err(|_| "event loop closed".to_owned());
                    }
                    if action == "close_window" {
                        if !origin.is_operator() { return Err("Name the window to close".into()); }
                        closing = true;
                        builtin.stop_all();
                        pulse.shutdown();
                        queue.clear(); command = json!({"id":0,"action":"close"});
                    }
                    if action == "stop_all" {
                        let mut host = workbench_host::DesktopHost { runtime: &runtime, bindings: &mut bindings, accepted: &accepted,
                            queue: &mut queue, targets: &mut pending_targets };
                        workbench.stop_all(&mut host);
                        workbench_host::deliver(&mut workbench, &builtin, &mut host);
                        accepted["acquisition"] = builtin.acquisition();
                        workbench_host::decorate(&workbench, &runtime, &bindings, &pending_targets, &tools, &mut accepted);
                        let failures = safe_actions.make_safe();
                        accepted["request"] = command["id"].clone();
                        accepted["error"] = if failures.is_empty() { Value::Null } else { json!(format!("Not every output reported safe: {}", failures.join("; "))) };
                        return windows.send(&accepted);
                    }
                    if matches!(action.as_str(), "share_start" | "share_stop" | "share_decide") {
                        // Browser sharing: the window's state and pictures to approved browsers.
                        let outcome = match action.as_str() {
                            "share_start" => {
                                let mut state = accepted.clone();
                                state["pulse"] = pulse.state();
                                let fronts = latest_fronts.iter().filter_map(|(panel, front)| accepted["panels"].as_array()
                                    .and_then(|rows| rows.iter().find(|row| row["id"] == panel.as_str()))
                                    .and_then(|row| row["frame"]["sequence"].as_u64()).map(|sequence| (panel.clone(), sequence, front.clone()))).collect();
                                share.start(&state, fronts)
                            }
                            "share_stop" => { share.stop(); Ok(()) }
                            _ => share.decide(command["client"].as_str().unwrap_or(""), command["allow"].as_bool().unwrap_or(false)),
                        };
                        accepted["share"] = share.state();
                        accepted["request"] = command["id"].clone();
                        accepted["error"] = outcome.err().map_or(Value::Null, Value::String);
                        return windows.send(&accepted);
                    }
                    if action == "show_window" {
                        let kind = command["window"].as_str().and_then(windows::Kind::parse)
                            .ok_or("Name a window: devices, task-console, pulse-editor, figure-viewer-<n>, control:<device>, log:<device>, slm-editor:<device> or review:<node>")?;
                        match &kind {
                            // A review window is its run's request's: it opens with it, never on its own.
                            windows::Kind::Review(node) if windows.get(&kind).is_none() => return Err(format!("{node} has no open point review")),
                            // A control opens on its device's reading (v2): made hidden, its page shows it once its fields are read.
                            windows::Kind::Control(device) | windows::Kind::Log(device) | windows::Kind::SlmEditor(device) if !live_device(&accepted, device) => {
                                return Err(format!("No open device {device:?}"));
                            }
                            windows::Kind::FigureViewer(_) if !listed(&kind, &viewers) => return Err(format!("No Figure Viewer {} (open a Figure to make one)", kind.key())),
                            windows::Kind::Control(_) if windows.get(&kind).is_none() => windows.make(&kind, target, false)?,
                            _ => windows.show(&kind, target)?,
                        }
                        // TaskConsole's Monitor or Logic page (v2 view.monitor / view.logic): the page switches its tab.
                        if let Some(page) = command["page"].as_str() {
                            if kind != windows::Kind::TaskConsole || !matches!(page, "monitor" | "logic") { return Err("page is monitor or logic, of task-console".into()); }
                            windows.send_to(&kind, &json!({"type": "page", "page": page}))?;
                        }
                        let mut state = json!({"type":"state", "request": command["id"], "result": {"windows": listed_windows(&windows, &viewers)}});
                        state["error"] = Value::Null;
                        remote::settle(&state, &mut remote_pending);
                        return windows.send(&state);
                    }
                    if matches!(action.as_str(), "choose_file" | "choose_folder") {
                        // A path field's Browse… (v2 FluentPathEdit): this window's own dialog; MCP names paths itself.
                        // `folder` is the field's kind's folder (a resource's collection, the workspace for a folder
                        // field): a relative start names a path in it, the dialog opens there when the start is gone,
                        // and a path picked inside it comes back relative to it, so the draft moves with the workspace.
                        // Without one a relative start is from the workspace, and the dialog's own folder is data
                        // for a node's file (an artifact input, a calibration), today's for Panel Edit's folder.
                        let workspace = workbench.workspace();
                        let file = action == "choose_file";
                        let folder = command["folder"].as_str().filter(|folder| !folder.is_empty()).map(|folder| workspace.resolve(folder));
                        let home = folder.clone().unwrap_or_else(|| if file { workspace.data() } else { workspace.today() });
                        let start = command["start"].as_str().map(str::trim).filter(|path| !path.is_empty())
                            .map_or_else(PathBuf::new, |path| folder.as_ref().map_or_else(|| workspace.resolve(path), |folder| folder.join(path)));
                        let (hwnd, title) = (windows.hwnd(from.as_ref()), command["title"].as_str().unwrap_or(""));
                        let chosen = if file { dialog::file(hwnd, title, command["filter"].as_str().unwrap_or(""), false, &start, &home)? }
                                     else { dialog::folder(hwnd, title, &start, &home)? };
                        let chosen = chosen.map(|path| folder.as_ref().and_then(|folder| std::path::Path::new(&path).strip_prefix(folder).ok())
                            .filter(|inside| !inside.as_os_str().is_empty()).map_or_else(|| path.clone(), |inside| inside.display().to_string()));
                        // Only the asking window is told: the reply carries no section of anyone's.
                        let state = json!({"type": "state", "request": command["id"], "error": null, "result": chosen.map_or(Value::Null, |path| json!({"path": path}))});
                        return windows.send(&state);
                    }
                    if action == "save_image" {
                        // Save image… (v2 save_screenshot): the page as it is on screen, in the file the operator picks.
                        let request = command["id"].as_u64().unwrap_or(0);
                        let shown = command["window"].as_str().and_then(windows::Kind::parse).or(from.clone()).unwrap_or(windows::Kind::TaskConsole);
                        let path = match command["path"].as_str() {
                            Some(path) => PathBuf::from(path),
                            None => match dialog::file(windows.hwnd(from.as_ref()), "Save screenshot", "PNG images (*.png)", true,
                                                       &workbench.workspace().today().join("screenshot.png"), &workbench.workspace().data())? {
                                // A name's own dots stay ("scan_1.5us"): a name not ending in .png gets it added.
                                Some(path) if path.to_ascii_lowercase().ends_with(".png") => PathBuf::from(path),
                                Some(path) => PathBuf::from(format!("{path}.png")),
                                None => { let mut state = accepted.clone(); state["request"] = command["id"].clone(); state["result"] = Value::Null; return windows.send(&state); }
                            },
                        };
                        let saved = self_proxy.clone();
                        return windows.get(&shown).ok_or_else(|| format!("The {} window is not open", shown.title()))?.pixels.capture(move |png| {
                            let result = png.and_then(|png| write_image(&path, &png));
                            let _ = saved.send_event(AppEvent::ImageSaved { request, result });
                        });
                    }
                    if action == "open_recorded_pulse" {
                        let key = command["key"].as_str().unwrap_or("");
                        let asking = match &from { Some(windows::Kind::FigureViewer(number)) => viewers.get(number), _ => None };
                        let played = asking.into_iter().chain(viewers.values()).filter_map(|viewer| viewer.archive.as_ref())
                            .find_map(|archive| archive.record.pulses.iter().find(|pulse| pulse.key == key))
                            .ok_or_else(|| format!("This Figure played no pulse {key:?}"))?;
                        let answer = pulse.editor.open_recorded(&played.tree.to_string(), command["discard_changes"] == true);
                        pulse.dispatch();
                        let mut state = json!({"type": "state", "request": command["id"]});
                        state["pulse"] = pulse.state();
                        match answer { Ok(result) => state["result"] = result, Err(error) => state["error"] = json!(error) }
                        remote::settle(&state, &mut remote_pending);
                        let sent = windows.send(&state);
                        pulse.recycle(&mut state["pulse"]);
                        return sent;
                    }
                    if action.starts_with("pulse_") {
                        // The Pulse Editor and the board: answered on this turn or by a lane later.
                        let request = command["id"].as_u64().unwrap_or(0);
                        let claimant = pulse.apparatus_device().and_then(|device| workbench.device_claimant(&device));
                        let hwnd = windows.hwnd(from.as_ref());
                        let handled = pulse.command(&action, &command, origin.is_operator(), request, |run| workbench.run_claims(run), claimant,
                                                    |title, filter, save, start, home| dialog::file(hwnd, title, filter, save, start, home));
                        let mut state = json!({"type": "state", "request": request});
                        state["pulse"] = pulse.state();
                        match handled {
                            pulse_host::Handled::Now(Ok(result)) => { state["result"] = result; }
                            pulse_host::Handled::Now(Err(error)) => { state["error"] = json!(error); }
                            pulse_host::Handled::Later => { state.as_object_mut().unwrap().remove("request"); }
                        }
                        remote::settle(&state, &mut remote_pending);
                        let sent = windows.send(&state);
                        pulse.recycle(&mut state["pulse"]);
                        return sent;
                    }
                    if worker_failed { return Err("Scientific worker stopped; restart after the displayed error".into()); }
                    // A Figure Viewer's board is its own (v2 panel_only): its order, panel names,
                    // intervals and Overlay are kept with its document, not the live board's.
                    if matches!(action.as_str(), "order_panels" | "rename_panel" | "set_panel_interval" | "set_panel_overlay") {
                        let viewer_number = match &from {
                            Some(windows::Kind::FigureViewer(number)) => Some(*number),
                            _ => command["panel"].as_str().or_else(|| command["order"][0].as_str()).and_then(panels::viewer_of),
                        };
                        if let Some(number) = viewer_number {
                            let viewer = viewers.get_mut(&number).filter(|viewer| !viewer.closing).ok_or_else(|| format!("FigureViewer {number} is not open"))?;
                            let result = viewer.board_command(&action, &command)?;
                            viewer.state["request"] = command["id"].clone();
                            viewer.state["result"] = result;
                            viewer.state["error"] = Value::Null;
                            return viewer_message(&mut viewer.state, |message| {
                                remote::settle(message, &mut remote_pending);
                                windows.send_to(&windows::Kind::FigureViewer(number), message)
                            });
                        }
                    }
                    if matches!(action.as_str(), "start_camera" | "stop_camera") || zlc_workbench::Workbench::handles(&action)
                        || workbench.handles_command(&action) {
                        if let Some((title, filter, save, start, home)) = workbench.file_dialog(&action).filter(|_| command["path"].is_null()) {
                            match dialog::file(windows.hwnd(from.as_ref()), title, filter, save, &start, &home)? {
                                Some(path) => command["path"] = json!(path),
                                None => { let mut state = accepted.clone(); state["request"] = command["id"].clone(); return windows.send(&state); }
                            }
                        }
                        let request = command["id"].as_u64().unwrap_or(0);
                        if action == "start_camera" && builtin.active() { return Err("Camera acquisition is already active".into()); }
                        if action == "start_camera" { reset_document = bindings.is_empty(); user_error = None; }
                        let mut host = workbench_host::DesktopHost { runtime: &runtime, bindings: &mut bindings, accepted: &accepted,
                            queue: &mut queue, targets: &mut pending_targets };
                        let alias = match action.as_str() {
                            "start_camera" => Some(workbench_host::start_camera(&mut workbench, &builtin, &command, &origin, &mut host)),
                            "stop_camera" => Some(workbench_host::stop_camera(&mut workbench, &builtin, &origin, &mut host)),
                            _ => { workbench_host::run(&mut workbench, &builtin, &action, &remote::parameters(&command), &origin, request, &mut host); None }
                        };
                        if workbench.on_runtime(&mut host) { workbench_host::deliver(&mut workbench, &builtin, &mut host); }
                        tell_editor_claims(&mut pulse, &workbench);
                        // A changed Overlay redraws the picture it is of, data or not.
                        if action == "set_panel_overlay" { pending.store(true, Ordering::Release); }
                        accepted["acquisition"] = builtin.acquisition();
                        if let Some(result) = alias {
                            result?;
                            accepted["request"] = command["id"].clone(); accepted["error"] = Value::Null;
                        }
                        let mut replied = false;
                        for (request, result) in workbench.take_replies() {
                            replied = true;
                            accepted["request"] = json!(request);
                            match result {
                                Ok(value) => { accepted["result"] = value; accepted["error"] = Value::Null; }
                                Err(error) => { accepted["result"] = Value::Null; accepted["error"] = json!(error); }
                            }
                            remote::settle(&accepted, &mut remote_pending);
                        }
                        // A device command handed to its device changes nothing here until the device
                        // answers (an SLM Editor's paint stroke is one): no state goes out for it now.
                        if !replied && workbench.handles_command(&action) { return Ok(()); }
                        workbench_host::decorate(&workbench, &runtime, &bindings, &pending_targets, &tools, &mut accepted);
                        return windows.send(&accepted);
                    }
                    // A command of an opened Figure goes to its Figure Viewer's document: one
                    // from that window, of one of its panels, of its data, or `open {window}`.
                    // A Figure opened anywhere else opens in a Figure Viewer of its own (v2).
                    let document: u32 = match &from {
                        Some(windows::Kind::FigureViewer(number)) => *number,
                        _ => command["panel"].as_str().and_then(panels::viewer_of)
                            .or_else(|| command["signal"].as_str().filter(|_| action == "add_panel").and_then(|signal| viewers.iter()
                                .find(|(number, viewer)| viewer.archive.as_ref().is_some_and(|archive| archive.offers(signal, **number))).map(|(number, _)| *number)))
                            .or_else(|| command["window"].as_str().filter(|_| action == "open").and_then(windows::Kind::parse)
                                .and_then(|kind| if let windows::Kind::FigureViewer(number) = kind { Some(number) } else { None }))
                            .unwrap_or(0),
                    };
                    if document != 0 && !viewers.get(&document).is_some_and(|viewer| !viewer.closing) {
                        return Err(format!("FigureViewer {document} is not open"));
                    }
                    if action == "open" { if let Some(object) = command.as_object_mut() { object.remove("window"); } }
                    let in_viewer = document != 0;
                    if in_viewer { command["document"] = json!(document); }
                    if matches!(action.as_str(), "new_data" | "edit_data" | "data_editor" | "close_data_editor") {
                        // v2 FigureViewer's Edit data / New data: working copies of a Figure's
                        // datasets, edited in its own tabs, applied to preview panels, saved as Figures.
                        let viewer = viewers.get_mut(&document).filter(|_| in_viewer).ok_or("Data is edited in a Figure Viewer")?;
                        let mut effect = manual::Effect::None;
                        let mut focus = None;
                        match action.as_str() {
                            "new_data" => focus = Some(viewer.drafts.new_data()?.id.clone()),
                            "edit_data" => {
                                let signal = command["signal"].as_str().ok_or("Choose the data to edit")?.to_owned();
                                let archive = viewer.archive.clone();
                                let draft = viewer.drafts.edit(&signal, || {
                                    let archive = archive.as_ref().ok_or("This Figure Viewer has no Figure open")?;
                                    let item = archive.items().iter().find(|item| *item.signal == *signal).ok_or_else(|| format!("unknown editable Dataset {signal:?}"))?;
                                    Ok(manual::EditSource { snapshot: item.plane.clone(), name: archive.label(&signal).unwrap_or(&item.key).to_owned(),
                                        path: archive.path().to_owned(), dataset: item.key.clone(), recipe: Some(item.target.clone()), signal: signal.clone(),
                                        overlay: item.overlay.clone() })
                                })?;
                                draft.open = true;
                                focus = Some(draft.id.clone());
                            }
                            "data_editor" => {
                                let editor = command["editor"].as_str().ok_or("Name the data editor")?;
                                let draft = viewer.drafts.get_mut(editor).ok_or_else(|| format!("unknown data editor {editor:?}"))?;
                                effect = draft.intent(&command["intent"]);
                            }
                            _ => {
                                let editor = command["editor"].as_str().ok_or("Name the data editor")?.to_owned();
                                let discard = command["discard"].as_bool().unwrap_or(false);
                                let draft = viewer.drafts.get_mut(&editor).ok_or_else(|| format!("unknown data editor {editor:?}"))?;
                                // v2 close_data_editor: unsaved edits go only when the operator agreed.
                                if draft.dirty() && !discard { draft.warn("Save or discard this data working copy before closing"); }
                                else { effect = draft.close(); }
                            }
                        }
                        let mut close = Vec::new();
                        match effect {
                            manual::Effect::None => {}
                            manual::Effect::Apply(apply) => {
                                let editor = apply.editor.clone();
                                match apply_data(&runtime, viewer, *apply) {
                                    Ok(replaced) => close.extend(replaced),
                                    Err(error) => if let Some(draft) = viewer.drafts.get_mut(&editor) { draft.refused(error); },
                                }
                            }
                            manual::Effect::Discarded { panel, signal } => close.extend(discard_data(&runtime, viewer, panel, &signal)),
                            manual::Effect::Save { editor, panel, path, suggested } => {
                                let path = match path { Some(path) => Some(path), None => {
                                    // Beside its source Figure, else in today's folder (as the viewer opens).
                                    let home = viewer_folder(workbench.workspace());
                                    let start = if std::path::Path::new(&suggested).parent().is_some_and(|folder| !folder.as_os_str().is_empty()) { PathBuf::from(&suggested) }
                                                else { home.join(&suggested) };
                                    dialog::file(windows.hwnd(from.as_ref()), "Save edited figure", "Saved figure archives (*.npz)", true, &start, &home)?
                                } };
                                if let Some(path) = path {
                                    // A Save as… of an archive name writes its picture beside it (v2 `with_suffix(".png")`);
                                    // a name's own dots ("scan_1.5us") stay in it.
                                    let image = match path.strip_suffix(".npz").or_else(|| path.strip_suffix(".NPZ")) { Some(stem) => format!("{stem}.png"), None => format!("{path}.png") };
                                    let applies = viewer.drafts.get_mut(&editor).map_or(0, |draft| draft.applies);
                                    viewer.drafts.saving = Some((command["id"].as_u64().unwrap_or(0), editor, applies));
                                    queue.push_back(json!({"id": command["id"], "action": "save", "panel": panel, "path": image, "document": document}));
                                    viewer.state["data"] = viewer.drafts.state();
                                    return Ok(());
                                }
                            }
                        }
                        for panel in close { queue.push_back(json!({"id": 0, "action": "close_panel", "panel": panel, "document": document})); }
                        viewer.drafts.list.retain(|draft| draft.open || draft.applied());
                        viewer.state["data"] = viewer.drafts.state();
                        viewer.state["data_focus"] = json!(focus);
                        viewer.state["request"] = command["id"].clone();
                        viewer.state["error"] = Value::Null;
                        viewer_message(&mut viewer.state, |message| windows.send_to(&windows::Kind::FigureViewer(document), message))?;
                        viewer.state["data_focus"] = Value::Null;
                        return Ok(());
                    }
                    if action == "add_panel" {
                        // v2 adds a blank panel; its signal is chosen in its Setting.
                        let signal = command["signal"].as_str().unwrap_or("");
                        let kind = command["kind"].as_str().ok_or("Choose a plot kind")?;
                        let panel = if in_viewer { viewers.get_mut(&document).expect("open").bindings.add(&runtime,signal,kind)? }
                            else {
                                let panel = bindings.add(&runtime,signal,kind)?;
                                workbench.panel_added(&panel);
                                if signal.is_empty() { workbench.console_note(format!("added {panel}; choose a signal in Setting")); }
                                // v2 update_panel_state: a panel given a signal says what it is doing.
                                else if runtime.signals().iter().any(|item| item.signal.as_ref() == signal && item.published) {
                                    workbench.console_note(format!("{panel} is preparing {}", workbench.signal_label(signal)));
                                } else { workbench.console_warning(format!("{} has not published yet; {panel} remains ready", workbench.signal_label(signal))); }
                                panel
                            };
                        // A new grid starts with its cell the data's choice (v2 cell_kind "").
                        let start = workbench_host::automatic_cell(kind, Value::Null);
                        if start.is_object() {
                            match viewers.get_mut(&document).filter(|_| in_viewer) { Some(viewer) => { viewer.targets.insert(panel.clone(), start); } None => { pending_targets.insert(panel.clone(), start); } }
                        }
                        command["panel"] = json!(panel);
                        command["action"] = json!("sync_panels");
                    }
                    if action == "configure" {
                        let panel = command["panel"].as_str().ok_or("Configuration has no panel")?.to_owned();
                        if !in_viewer && (command.get("binding").is_some() || command.get("semantic").is_some()) {
                            // An Edit's settings are its panel's: the panel's protection holds for both.
                            let subject = panel.strip_prefix(panels::EDIT_PREFIX).unwrap_or(&panel);
                            let signal = bindings.describe().into_iter().find(|(id, ..)| id == subject).map(|(_, signal, ..)| signal).unwrap_or_default();
                            if let Some(reason) = workbench.panel_frozen(&panel, &signal) { return Err(reason); }
                        }
                        let (set, shown) = match viewers.get_mut(&document) { Some(viewer) => (&mut viewer.bindings, &viewer.state), None => (&mut bindings, &accepted) };
                        if let Some(patch) = command.get("binding") {
                            // Each document shows its own data: an archive's, or the apparatus's.
                            if patch["signal"].as_str().is_some_and(|signal| !signal.is_empty() && archive::viewer_owned(signal) != in_viewer) {
                                return Err(if in_viewer { "A Figure Viewer panel shows the opened Figure's data" } else { "An opened Figure's data is shown in the Figure Viewer" }.into());
                            }
                            let before = set.describe().into_iter().find(|(id, ..)| *id == panel).map(|(_, signal, ..)| signal).unwrap_or_default();
                            set.configure(&runtime,&panel,patch)?;
                            // v2 update_panel_state: a panel given another signal says what it is doing.
                            if let Some(signal) = patch["signal"].as_str().filter(|signal| !in_viewer && !signal.is_empty() && *signal != before) {
                                let words = workbench.signal_label(signal);
                                if runtime.signals().iter().any(|item| item.signal.as_ref() == signal && item.published) { workbench.console_note(format!("{panel} is preparing {words}")); }
                                else { workbench.console_warning(format!("{words} has not published yet; {panel} remains ready")); }
                            }
                        }
                        if let Some(epoch) = set.epoch(&panel) {
                            if command["parameters"].get("window").is_some() { return Err("Use the panel input Window; it owns both retention and the scientific window".into()); }
                            let previous = shown["panels"].as_array().and_then(|panels|panels.iter().find(|item|item["id"]==panel.as_str()))
                                .and_then(|item|item["data_binding_epoch"].as_u64());
                            if command.get("binding").is_some() || previous != Some(epoch) || command["semantic"].get("kind").is_some() { command["action"] = json!("sync_panels"); }
                        }
                    }
                    if action == "close_panel" {
                        if let Some(panel) = command["panel"].as_str() {
                            if let Some(viewer) = viewers.get_mut(&document) {
                                viewer.bindings.remove(panel);
                                viewer.targets.remove(panel);
                                // A panel's Edit goes with it.
                                let edit = panels::edit_of(panel);
                                if viewer.bindings.remove(&edit) { queue.push_back(json!({"id": 0, "action": "close_panel", "panel": edit, "document": document})); }
                            }
                            else {
                                bindings.remove(panel); workbench.panel_closed(panel); pending_targets.remove(panel);
                                // A panel's Edit goes with it.
                                let edit = panels::edit_of(panel);
                                if bindings.remove(&edit) { queue.push_back(json!({"id":0,"action":"close_panel","panel":edit})); }
                            }
                        }
                    }
                    if matches!(action.as_str(), "edit_panel" | "refresh_panel_edit") {
                        // v2 Panel Edit: the panel held still on one read; Refresh snapshot takes its newest.
                        let named = command["panel"].as_str().ok_or("Name the panel")?;
                        let panel = named.strip_prefix(panels::EDIT_PREFIX).unwrap_or(named).to_owned();
                        if in_viewer {
                            // v2 FigureViewer: a saved Figure's panel has the board's Edit page too (its own tab there).
                            let viewer = viewers.get_mut(&document).expect("open");
                            if !viewer.bindings.has(&panel) { return Err(format!("No panel {panel} on the board")); }
                            let edit = panels::edit_of(&panel);
                            let open = viewer.bindings.has(&edit);
                            if action == "refresh_panel_edit" && !open { return Err(format!("{panel} has no open Edit")); }
                            if !open || action == "refresh_panel_edit" {
                                viewer.bindings.freeze(&runtime, &panel, &viewer.state, &Default::default())?;
                                if let Some(target) = viewer.state["panels"].as_array().and_then(|rows| rows.iter().find(|row| row["id"] == panel.as_str())).map(|row| row["target"].clone()) {
                                    if !open { viewer.targets.insert(edit.clone(), target); }
                                }
                            }
                            command = json!({"id": command["id"], "action": "sync_panels", "document": document});
                        }
                        else {
                            if !bindings.has(&panel) { return Err(format!("No panel {panel} on the board")); }
                            let edit = panels::edit_of(&panel);
                            let open = bindings.has(&edit);
                            if action == "refresh_panel_edit" && !open { return Err(format!("{panel} has no open Edit")); }
                            if !open || action == "refresh_panel_edit" {
                                bindings.freeze(&runtime, &panel, &accepted, &workbench.panel_overlays())?;
                                // It starts as its panel is set; from then on the two share every setting.
                                if let Some(target) = accepted["panels"].as_array().and_then(|rows| rows.iter().find(|row| row["id"] == panel.as_str())).map(|row| row["target"].clone()) {
                                    if !open { pending_targets.insert(edit.clone(), target); }
                                }
                            }
                            command = json!({"id": command["id"], "action": "sync_panels"});
                        }
                    }
                    if action == "open_point_review" {
                        // v2 PointReviewView's plot: the image a run's point-selection request
                        // names, held on its read, every candidate drawn (review-<node>, the window's).
                        let node = command["node"].as_str().ok_or("Name the node")?;
                        let request = accepted["logic"].as_array().and_then(|rows| rows.iter().find(|row| row["node_id"] == node))
                            .map(|row| row["run"]["request"].clone()).filter(|request| request["kind"] == "point-selection" && request["request"] == command["request"])
                            .ok_or_else(|| format!("{node} asks no such point review"))?;
                        // A request naming its image and its points is drawn; one without is reviewed by its list.
                        if let (Some(output), Some(points)) = (request["payload"]["output_name"].as_str().filter(|name| !name.is_empty()), request["payload"]["points"].as_array()) {
                            let points = points.iter().map(|point| Some((point[0].as_str()?.to_owned(), point[1].as_str()?.to_owned(), point[2].as_f64()?, point[3].as_f64()?)))
                                .collect::<Option<Vec<_>>>().ok_or("A review point is [id, label, x, y]")?;
                            let panel = bindings.review(&runtime, node, &format!("@logic/{node}/{output}"), points)?;
                            // v2's review plot: an Image panel at the 4x4 preset.
                            pending_targets.insert(panel, json!({"semantic": {"kind": "image"}, "size": "4x4"}));
                        }
                        // v2's review window: a screen fit titled with the request's words; the others wait for it.
                        let kind = windows::Kind::Review(node.to_owned());
                        windows.show(&kind, target)?;
                        if let Some(window) = windows.get(&kind) { window.window.set_title(request["title"].as_str().filter(|title| !title.is_empty()).unwrap_or("Review")); }
                        windows.set_modal(&kind, true);
                        command = json!({"id": command["id"], "action": "sync_panels"});
                    }
                    if action == "close_point_review" {
                        // The review answered, or its request gone: its window and its panel go.
                        let node = command["node"].as_str().ok_or("Name the node")?.to_owned();
                        let kind = windows::Kind::Review(node.clone());
                        windows.set_modal(&kind, false);
                        windows.close(&kind);
                        let panel = panels::review_of(&node);
                        if !bindings.remove(&panel) { return Ok(()); }
                        pending_targets.remove(&panel);
                        command = json!({"id": command["id"], "action": "close_panel", "panel": panel});
                    }
                    if action == "point_review_state" {
                        // What the review marks -- excluded, selected -- drawn on its candidates.
                        let ids = |key: &str| command[key].as_array().into_iter().flatten().filter_map(Value::as_str).map(str::to_owned).collect();
                        let panel = command["panel"].as_str().ok_or("Name the review")?.to_owned();
                        bindings.mark_review(&panel, &ids("excluded"), &ids("selected"))?;
                        // Only the review's picture changes: its panel alone is drawn again.
                        command = json!({"id": command["id"], "action": "configure", "panel": panel});
                    }
                    if action == "open" && command.get("path").is_none() {
                        // v2 FigureViewer's path field: Browse… starts at the open Figure, else in today's folder.
                        let shown = match &from { Some(windows::Kind::FigureViewer(number)) => viewers.get(number).and_then(|viewer| viewer.state["document"]["path"].as_str()), _ => None };
                        let home = viewer_folder(workbench.workspace());
                        let start = shown.filter(|path| !path.is_empty()).map_or_else(|| home.clone(), PathBuf::from);
                        match dialog::file(windows.hwnd(from.as_ref()), "Open a saved figure archive (.npz)", "Saved figure archives (*.npz)", false, &start, &home)? {
                            Some(path) => command["path"] = json!(path),
                            None => { let mut state = accepted.clone(); state["request"] = command["id"].clone(); return windows.send(&state); }
                        }
                    }
                    // Coalesce only unsent pointer movement from the same panel.
                    // Press/release/configure/file operations keep exact order.
                    if action == "pointer" && command["event"] == "move" {
                        if let Some(last) = queue.back_mut().filter(|last| last["action"] == "pointer"
                            && last["event"] == "move" && last["panel"] == command["panel"]) {
                            *last = command;
                            return Ok(());
                        }
                    }
                    // A hover still waiting is stale once the pointer rests elsewhere: the newer takes its place.
                    if hovering(&command) {
                        if let Some(last) = queue.iter_mut().rev().find(|queued| queued["action"] == "pointer" && queued["panel"] == command["panel"])
                            .filter(|last| hovering(last)) {
                            *last = command;
                            return Ok(());
                        }
                    }
                    if action == "fit_window" {
                        if let (Some(kind), Some(height)) = (&from, command["height"].as_f64().filter(|height| *height > 0.0)) { windows.fit(kind, command["width"].as_f64().unwrap_or(0.0), height); }
                        return Ok(());
                    }
                    if action == "ready" {
                        // A page is up: it is given everything it shows, its document draws at its device ratio.
                        let kind = from.clone().ok_or("ready comes from a window")?;
                        trace::point("window.ready", || json!({"window": kind.key()}));
                        let ratio = command["dpr"].as_f64().filter(|dpr| *dpr > 0.0);
                        if let Some(window) = windows.get_mut(&kind) { window.ready = true; }
                        match &kind {
                            windows::Kind::TaskConsole => { share.set_dpr(ratio.unwrap_or(1.0)); queue.push_back(json!({"id":0,"action":"ready","dpr":ratio})); }
                            windows::Kind::FigureViewer(number) => {
                                queue.push_back(json!({"id":0,"action":"ready","dpr":ratio,"document":number}));
                                if let Some(open) = viewers.get_mut(number).and_then(|viewer| viewer.waiting.take()) { queue.push_back(open); }
                            }
                            windows::Kind::PulseEditor => { pulse.dpr = ratio.unwrap_or(1.0); pulse.dispatch(); }
                            // A review draws the live document's panel at the Task Console's ratio.
                            windows::Kind::Devices | windows::Kind::Control(_) | windows::Kind::Log(_) | windows::Kind::Review(_) | windows::Kind::Installer => {}
                            // An SLM Editor draws its own document at its page's ratio.
                            windows::Kind::SlmEditor(device) => slm.ready(device, ratio.unwrap_or(1.0)),
                        }
                        windows.page_loaded(&kind);
                        windows.send_state_to(&kind, &accepted)?;
                        // Its title bar: the window's width and whether it is maximized.
                        if let Some(caption) = windows.get(&kind).and_then(|shown| frame::caption(&shown.window)) { windows.send_to(&kind, &caption)?; }
                        if let windows::Kind::SlmEditor(device) = &kind {
                            windows.send_to(&kind, &slm.message(device))?;
                            for (panel, sequence, front) in slm.fronts(device) { windows.submit(panel, sequence, front)?; }
                        }
                        if let windows::Kind::FigureViewer(number) = &kind { if let Some(viewer) = viewers.get_mut(number) { viewer_message(&mut viewer.state, |message| windows.send_to(&kind, message))?; } }
                        if kind == windows::Kind::PulseEditor {
                            let mut state = json!({"type": "state"});
                            state["pulse"] = pulse.state();
                            windows.send_to(&kind, &state)?;
                            pulse.recycle(&mut state["pulse"]);
                        }
                        // Its panels' last pictures, so a window shown again is whole at once.
                        for (panel, front) in &latest_fronts {
                            if windows::owner(panel) != kind { continue; }
                            let shown = match &kind { windows::Kind::FigureViewer(number) => viewers.get(number).map_or(&accepted, |viewer| &viewer.state), _ => &accepted };
                            if let Some(sequence) = shown["panels"].as_array().and_then(|rows| rows.iter().find(|row| row["id"] == panel.as_str())).and_then(|row| row["frame"]["sequence"].as_u64()) {
                                windows.submit(panel.clone(), sequence, front.clone())?;
                            }
                        }
                        if standalone && !opened {
                            // A standalone app's file, once its one window is up.
                            opened = true;
                            match &mode {
                                Mode::FigureViewer(Some(path)) => queue.push_back(json!({"id":0,"action":"open","path":path,"document":1})),
                                Mode::PulseEditor(Some(path)) => { let _ = self_proxy.send_event(AppEvent::Command { window: Some(windows::Kind::PulseEditor), command: json!({"id": 0, "action": "pulse_open", "path": path}) }); }
                                _ => {}
                            }
                        }
                        // Every other window is made when it is first shown, as v2 made its windows.
                        if kind == windows::Kind::Devices && !opened {
                            opened = true;
                            if let Some(path) = initial_path.take() {
                                let number = take_viewer(&mut viewers, &mut next_viewer);
                                windows.show(&windows::Kind::FigureViewer(number), target)?;
                                if let Some(viewer) = viewers.get_mut(&number) { viewer.waiting = Some(json!({"id":0,"action":"open","path":path,"document":number})); }
                            }
                        }
                        return Ok(());
                    }
                    // An opened Figure is read in its own window: its viewer's, or a new one.
                    if action == "open" {
                        let number = if in_viewer { document } else { take_viewer(&mut viewers, &mut next_viewer) };
                        command["document"] = json!(number);
                        let kind = windows::Kind::FigureViewer(number);
                        windows.show(&kind, target)?;
                        // A new window draws its Figure at its page's device ratio: the open waits for the page.
                        if !windows.get(&kind).is_some_and(|window| window.ready) {
                            let request = command["id"].clone();
                            if let Some(viewer) = viewers.get_mut(&number) { viewer.waiting = Some(command); }
                            return windows.send(&json!({"type":"busy", "request":request}));
                        }
                    }
                    // The scientific worker answers its own actions only: any other is refused here, by name,
                    // instead of failing inside the worker (a missing panel read as a KeyError).
                    let queued = command["action"].as_str().unwrap_or("");
                    if !SCIENCE_ACTIONS.contains(&queued) { return Err(format!("Unknown command {queued:?}")); }
                    if queue.len() >= 64 { return Err("Scientific command queue is full; no command was silently dropped".into()); }
                    let request = command["id"].clone();
                    // An operator's edit of one panel goes before the hovers waiting for the worker.
                    let edit = matches!(command["action"].as_str(), Some("configure"|"scope_step"|"focus"|"reset_view"|"fit"));
                    let at = queue.len() - if edit { queue.iter().rev().take_while(|queued| hovering(queued)).count() } else { 0 };
                    queue.insert(at, command);
                    if action != "pointer" { windows.send(&json!({"type":"busy", "request":request}))?; }
                }
                Event::UserEvent(AppEvent::Completed(mut completion)) => {
                    active = false;
                    let document = completion.document;
                    let in_viewer = document != 0;
                    if completion.action == "drop_document" {
                        // A closed Figure Viewer's last job: its document and its archive go.
                        if let Some(feedback)=completion.decision.take() {let _=feedback.send(science::PresentationDecision::default());}
                        viewers.remove(&document);
                    } else if let (Some(decoded), Some(viewer)) = (completion.imported.take(), viewers.get_mut(&document)) {
                        // Parsing has not touched the viewer's document. Only
                        // a fully rendered candidate can replace its panels.
                        if let Some(feedback)=completion.decision.take() {let _=feedback.send(science::PresentationDecision::default());}
                        viewer.state["request"]=json!(completion.request);
                        let ids=viewer.bindings.reserve_ids(decoded.items.len())?;
                        let data=Arc::new(archive::MountedArchive::mount(runtime.clone(),decoded,ids)?);
                        let candidate=panels::Panels::from_inputs(&runtime,&data.inputs(),&panels::viewer_prefix(document))?;
                        pending_archive=Some(PendingArchive{data,bindings:candidate,request:completion.request,document});
                        queue.push_front(json!({"id":completion.request,"action":"present_archive","document":document}));
                    } else if in_viewer && !viewers.contains_key(&document) {
                        if let Some(feedback)=completion.decision.take() {let _=feedback.send(science::PresentationDecision::default());}
                    } else {
                    if completion.action=="present_archive" {
                        let candidate=pending_archive.take().ok_or("Figure candidate is no longer available")?;
                        let failure=match &completion.result {
                            Err(error)=>Some(error.clone()),
                            Ok((state,fronts))=>candidate.data.items().iter().find_map(|item|{
                                let row=state["panels"].as_array().and_then(|rows|rows.iter().find(|row|row["id"]==item.panel));
                                row.and_then(|row|row["error"].as_str()).map(str::to_owned)
                                    .or_else(||(!fronts.contains_key(&item.panel)).then(||format!("{} has no completed Figure front",item.key)))
                            }),
                        };
                        if let Some(error)=failure {completion.result=Err(error);}
                        else {
                            // The opened Figure replaces its viewer's panels only; the live board runs on.
                            if let Some(viewer) = viewers.get_mut(&document) {
                                let mut next=candidate.bindings;next.continue_ids_from(&viewer.bindings);
                                viewer.bindings=next;
                                drop(viewer.archive.replace(candidate.data));
                                viewer.error=None;
                            }
                        }
                    }
                    // The other documents' panels keep their mappings and pictures.
                    let other: Vec<String> = std::iter::once((0, &accepted)).chain(viewers.iter().map(|(number, viewer)| (*number, &viewer.state)))
                        .filter(|(number, _)| *number != document).flat_map(|(_, state)| panel_ids(state)).collect();
                    let archive = viewers.get(&document).and_then(|viewer| viewer.archive.clone());
                    // The data working copies this viewer applied are its data too (v2 `_project_data_choices`).
                    let manual_data = viewers.get(&document).map(|viewer| viewer.drafts.offered()).unwrap_or_default();
                    let (bindings, accepted, user_error, viewer_pending, mut viewer_data) = match viewers.get_mut(&document) {
                        Some(viewer) => (&mut viewer.bindings, &mut viewer.state, &mut viewer.error, Some(&mut viewer.pending), Some((&mut viewer.drafts, &mut viewer.targets, &mut viewer.board))),
                        None => (&mut bindings, &mut accepted, &mut user_error, None, None),
                    };
                    let feedback=completion.decision;
                    let mut decision=science::PresentationDecision::default();
                    bindings.sync_generations(&runtime);
                    let current_authorities = bindings.authorities();
                    let mut stale: BTreeSet<_> = completion.authorities.iter().filter_map(|(panel,authority)|
                        (!*authority.lock().unwrap_or_else(|poisoned|poisoned.into_inner())
                            && !completion.publications.iter().any(|(id,key)|id==panel && bindings.accepts_source(panel,key)))
                            .then(||panel.clone())).collect();
                    let stale_targets:BTreeSet<_>=completion.intents.iter().filter_map(|(panel,authority)|
                        (!*authority.lock().unwrap_or_else(|poisoned|poisoned.into_inner())).then(||panel.clone())).collect();
                    let superseded = completion.target.as_ref().is_some_and(|panel|stale_targets.contains(panel));
                    if !stale.is_empty() {
                        // A source/lease transition can invalidate a downstream
                        // panel without invalidating the command which caused it.
                        match viewer_pending { Some(flag) => *flag = true, None => pending.store(true, Ordering::Release) }
                        trace::point("job.panels_superseded",||json!({"job":completion.ticket,"request":completion.request,
                            "panels":stale,"target":completion.target,"target_superseded":superseded}));
                    }
                    let superseded_error = "This panel's signal binding changed before the operation completed; the operation was superseded.";
                    match completion.result {
                        // A hover that drew nothing and changed no row only answers its request: every
                        // window already shows what it would be told (a pointer resting costs no decorating).
                        Ok((state, fronts)) if !in_viewer && completion.action == "pointer" && fronts.is_empty() && stale.is_empty()
                            && stale_targets.is_empty() && state["error"].is_null() && state["panels"] == answered_rows
                            && current_authorities.keys().eq(completion.authorities.keys()) => {
                            *user_error = None;
                            decision.targets.extend(panel_ids(&state));
                            accepted["request"] = json!(completion.request);
                            accepted["error"] = Value::Null;
                            windows.send(accepted)?;
                        }
                        Ok((mut state, mut fronts)) => {
                            if !in_viewer { answered_rows = state["panels"].clone(); }
                            // Same-shot presentation is all-or-nothing. A failed
                            // or superseded member cannot leave its siblings a
                            // shot ahead; unrelated components remain independent.
                            for cohort in &completion.cohorts {
                                if cohort.iter().any(|id|stale.contains(id) || stale_targets.contains(id) || !fronts.contains_key(id)) {
                                    stale.extend(cohort.iter().cloned());
                                }
                            }
                            if completion.sync {
                                if state["error"].is_null() { state["error"] = json!(*user_error); }
                            }
                            else { *user_error = None; }
                            if superseded && completion.request != 0 { state["error"] = json!(superseded_error); }
                            if !in_viewer {
                                state["acquisition"] = accepted["acquisition"].clone();
                                state["share"] = share.state();
                                for (_,read) in &completion.publications {
                                    if builtin.generation()==Some(read.generation) {
                                        if let Some(ordinal)=read.ordinal {
                                            state["acquisition"]["produced"] = json!((ordinal+1).max(state["acquisition"]["produced"].as_u64().unwrap_or(0)));
                                        }
                                    }
                                }
                            }
                            // Each document offers its own data: the opened Figure's, or the apparatus's.
                            bindings.decorate(&mut state,&runtime,|signal| if in_viewer { archive.as_ref().is_some_and(|archive| archive.offers(signal, document))
                                    || manual_data.iter().any(|(applied, _)| applied == signal) }
                                else { !archive::viewer_owned(signal) });
                            let previous = accepted["panels"].as_array().ok_or("Accepted state has no panels")?;
                            let rows = state["panels"].as_array_mut().ok_or("Scientific response has no panels")?;
                            let mut retained = Vec::with_capacity(rows.len());
                            for mut panel in rows.drain(..) {
                                let id = panel["id"].as_str().ok_or("Panel has no identity")?;
                                let target_invalid=stale_targets.contains(id)
                                    || (current_authorities.contains_key(id) && !completion.authorities.contains_key(id));
                                if target_invalid {
                                    fronts.remove(id);
                                    if current_authorities.contains_key(id) {
                                        if let Some(old) = previous.iter().find(|old|old["id"]==id) { retained.push(old.clone()); }
                                    }
                                } else {
                                    decision.targets.insert(id.to_owned());
                                    if stale.contains(id) {
                                        fronts.remove(id);
                                        let old=previous.iter().find(|old|old["id"]==id);
                                        // Keep authored controls, but every visible
                                        // fact continues to describe the old front.
                                        for field in ["axes","structure","source_signal","focus","frame","data_binding_epoch","kind"] {
                                            if let Some(old)=old {panel[field]=old[field].clone();}
                                            else if field=="axes" {panel[field]=json!([]);}
                                            else if field!="kind" {panel[field]=Value::Null;}
                                        }
                                        panel["fit"]["computed"]=old.map(|row|row["fit"]["computed"].clone()).unwrap_or_else(||json!({}));
                                    }
                                    retained.push(panel);
                                }
                            }
                            // A returned full state may omit a panel whose binding
                            // changed while this job ran. Keep only its last whole
                            // accepted row, never new axes/metadata over old pixels.
                            for old in previous {
                                let Some(id) = old["id"].as_str() else { continue; };
                                if current_authorities.contains_key(id)
                                    && (stale.contains(id) || !completion.authorities.contains_key(id))
                                    && !retained.iter().any(|panel|panel["id"]==id) {
                                    retained.push(old.clone());
                                }
                            }
                            *rows = retained;
                            fronts.retain(|id,_|rows.iter().any(|panel|panel["id"]==id.as_str()) && !stale.contains(id)
                                && (!current_authorities.contains_key(id) || completion.authorities.contains_key(id)));
                            if matches!(completion.action.as_str(),"present_archive"|"close") {
                                decision.targets.extend(previous.iter().filter_map(|row|row["id"].as_str().map(str::to_owned)));
                            } else if completion.action=="close_panel" && !superseded {
                                decision.targets.extend(completion.target.iter().cloned());
                            }
                            decision.fronts.extend(fronts.keys().cloned());
                            // A pointer's picture shows the data already on screen: the panel's interval runs on.
                            if !in_viewer && completion.action != "pointer" { cadence.presented(fronts.keys()); }
                            sequence += 1;
                            trace::point("job.accept",||json!({"job":completion.ticket,"sequence":sequence,
                                "fronts":fronts.keys().collect::<Vec<_>>(),"request":completion.request}));
                            for panel in state["panels"].as_array_mut().ok_or("Scientific response has no panels")? {
                                science::spell_numbers(panel);
                                if let Some(caption) = panel["id"].as_str().and_then(panels::viewer_caption) { panel["caption_id"] = json!(caption); }
                                let id = panel["id"].as_str().ok_or("Panel has no identity")?;
                                if let Some(front) = fronts.get(id) {
                                    panel["frame"] = json!({"width":front.width,"height":front.height,"sequence":sequence});
                                } else if let Some(old) = accepted["panels"].as_array().and_then(|panels| panels.iter().find(|p| p["id"] == id)) {
                                    panel["frame"] = old["frame"].clone();
                                }
                            }
                            let ids: Vec<_> = state["panels"].as_array().unwrap().iter().filter_map(|p| p["id"].as_str().map(str::to_owned)).collect();
                            // A panel and its Edit share one configuration (v2's one panel state):
                            // what an operator changed on either is given to the other.
                            if let Some(changed) = completion.target.as_deref().filter(|_| !in_viewer && completion.request != 0) {
                                let twin = changed.strip_prefix(panels::EDIT_PREFIX).map(str::to_owned).unwrap_or_else(|| panels::edit_of(changed));
                                let target = |id: &str| state["panels"].as_array().and_then(|rows| rows.iter().find(|row| row["id"] == id)).map(|row| row["target"].clone());
                                if let (true, Some(source), Some(other)) = (bindings.has(&twin), target(changed), target(&twin)) {
                                    let mut patch = json!({"id":0, "action":"configure", "panel":twin});
                                    for key in ["semantic", "parameters", "fit", "size", "selectors", "viewport", "facet_focus", "classifier_thresholds"] {
                                        if source[key] != other[key] { patch[key] = source[key].clone(); }
                                    }
                                    if patch.as_object().is_some_and(|fields| fields.len() > 3) { queue.push_back(patch); }
                                }
                            }
                            if !in_viewer {
                                // A prepared panel's document now holds the target it was added with.
                                pending_targets.retain(|panel, _| !ids.contains(panel));
                                workbench_host::decorate(&workbench, &runtime, bindings, &pending_targets, &tools, &mut state);
                            }
                            // The opened Figure's record: the info pane's tabs and its Flow.
                            if let Some(archive) = archive.as_ref().filter(|archive| state["document"]["path"] == archive.path()) {
                                // The record the viewer already holds moves on, not copied again (it is the Figure's whole provenance).
                                let held = (completion.action != "present_archive" && accepted["document"]["path"] == archive.path() && !accepted["document"]["info"].is_null())
                                    .then(|| (accepted["document"]["info"].take(), accepted["document"]["flow"].take()));
                                let (info, flow) = held.unwrap_or_else(|| (archive.record.tabs.clone(), archive.record.flow.clone()));
                                state["document"]["info"] = info;
                                state["document"]["flow"] = flow;
                                for signal in state["signals"].as_array_mut().into_iter().flatten() {
                                    if let Some(label) = signal["id"].as_str().and_then(|id| archive.label(id)) { signal["label"] = json!(label); }
                                }
                            }
                            if let Some((drafts, targets, board)) = viewer_data.as_mut() {
                                for signal in state["signals"].as_array_mut().into_iter().flatten() {
                                    if let Some((_, name)) = manual_data.iter().find(|(applied, _)| signal["id"] == applied.as_str()) { signal["label"] = json!(name); }
                                }
                                // A preview's document now holds the recipe it started from.
                                targets.retain(|panel, _| !ids.contains(panel));
                                if let Some((request, editor, applies)) = drafts.saving.clone().filter(|(request, ..)| *request == completion.request) {
                                    drafts.saving = None;
                                    let archive_name = state["saved"].as_array().into_iter().flatten().filter_map(Value::as_str).find(|path| path.to_ascii_lowercase().ends_with(".npz"))
                                        .map(|path| std::path::Path::new(path).file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default());
                                    if let (Some(draft), Some(name)) = (drafts.get_mut(&editor), archive_name) { let current = draft.applies == applies; draft.saved(&name, current); }
                                    let _ = request;
                                }
                                drafts.settle(&state["panels"]);
                                state["data"] = drafts.state();
                                settle_board(board, &mut state);
                            }
                            *accepted = state;
                            if in_viewer { viewer_message(accepted, |message| windows.send_to(&windows::Kind::FigureViewer(document), message))?; } else { windows.send(accepted)?; }
                            let mut kept = ids.clone();
                            kept.extend(other.iter().cloned());
                            kept.push(pulse_host::PREVIEW_PANEL.to_owned());
                            kept.extend(slm.panels());
                            windows.retain_panels(&kept)?;
                            latest_fronts.retain(|id, _| ids.contains(id) || other.contains(id));
                            for (id, front) in fronts { latest_fronts.insert(id.clone(), front.clone()); windows.submit(id, sequence, front)?; }
                        }
                        Err(error) => {
                            let error = if superseded { superseded_error.to_owned() } else { error };
                            if let Some((drafts, _, _)) = viewer_data.as_mut() {
                                if let Some((_, editor, _)) = drafts.saving.clone().filter(|(request, ..)| *request == completion.request) {
                                    drafts.saving = None;
                                    if let Some(draft) = drafts.get_mut(&editor) { let name = draft.name().to_owned(); draft.refused(format!("cannot save {name}: {error}")); }
                                    accepted["data"] = drafts.state();
                                }
                            }
                            if completion.request != 0 { *user_error = Some(error.clone()); }
                            accepted["request"] = json!(completion.request);
                            accepted["error"] = json!(error);
                            if in_viewer { viewer_message(accepted, |message| windows.send_to(&windows::Kind::FigureViewer(document), message))?; } else { windows.send(accepted)?; }
                            eprintln!("{}", json!({"type":"scientific-error","request":completion.request,"error":error}));
                        }
                    }
                    // The live board's answers settle at the end of the turn; the viewer's here.
                    if in_viewer { remote::settle(accepted, &mut remote_pending); remote::check_waiters(accepted, &mut waiters, None); }
                    if let Some(feedback)=feedback {let _=feedback.send(decision);}
                    if closing && queue.is_empty() {
                        if let Some(worker) = worker.take() { let _ = worker.join(); }
                        *flow = ControlFlow::Exit;
                    }
                    }
                }
                Event::UserEvent(AppEvent::WorkerError(error)) => {
                    worker_failed = true; active = false; queue.clear();
                    if closing { *flow = ControlFlow::Exit; }
                    return Err(error);
                }
                Event::UserEvent(AppEvent::InvalidMessage(error)) => return Err(format!("Invalid UI command: {error}")),
                Event::UserEvent(AppEvent::DerivationError(error)) => {
                    user_error = Some(error.clone());
                    return Err(format!("Scientific derivation: {error}"));
                }
                Event::UserEvent(AppEvent::Remote(call)) => {
                    let answer = |result: Result<zlc_bus::mcp::ToolOutput, String>| { let _ = call.reply.send(result); };
                    if closing { answer(Err("The instrument is closing".into())); return Ok(()); }
                    match call.name.as_str() {
                        "get_state" => {
                            let mut state = accepted.clone();
                            // The pulse's view is made only for a caller who asks for it (it is the largest section).
                            if call.params["sections"].as_array().is_none_or(|sections| sections.iter().any(|section| section == "pulse")) { state["pulse"] = pulse.state(); }
                            state["share"] = share.state();
                            state["windows"] = listed_windows(&windows, &viewers);
                            // The Figure Viewer's document: the opened archive, its panels and its Data.
                            state["viewers"] = Value::Array(viewers.iter().filter(|(_, viewer)| !viewer.closing).map(|(number, viewer)| {
                                let mut shown = viewer.state.clone();
                                if let Some(object) = shown.as_object_mut() { for key in ["type", "request", "error", "result"] { object.remove(key); } }
                                shown["window"] = json!(windows::Kind::FigureViewer(*number).key());
                                shown
                            }).collect());
                            if let Some(panel) = call.params["panel"].as_str() {
                                if let Some(rows) = state["panels"].as_array_mut() { rows.retain(|row| row["id"] == panel); }
                                for shown in state["viewers"].as_array_mut().into_iter().flatten() {
                                    if let Some(rows) = shown["panels"].as_array_mut() { rows.retain(|row| row["id"] == panel); }
                                }
                            }
                            if let Some(node) = call.params["node"].as_str() {
                                if let Some(rows) = state["logic"].as_array_mut() { rows.retain(|row| row["node_id"] == node); }
                            }
                            if let (Some(sections), Some(object)) = (call.params["sections"].as_array(), state.as_object_mut()) {
                                object.retain(|key, _| key == "type" || key == "request" || sections.iter().any(|section| section == key.as_str()));
                            }
                            answer(Ok(zlc_bus::mcp::ToolOutput { json: state, png: None }));
                        }
                        "list_signals" => answer(Ok(zlc_bus::mcp::ToolOutput { json: json!({"signals": remote::signals(&runtime)}), png: None })),
                        "save_image" | "get_window_image" => {
                            // The same picture the window's Save image… writes; the reply carries it too
                            // (get_window_image: only the reply, v2 lab_screenshot).
                            let path = (call.name == "save_image").then(|| call.params["path"].as_str().map(|path| root.join(path))
                                .unwrap_or_else(|| fresh_image_path(&workbench.workspace().today())));
                            // The named window's picture (MCP addresses windows by name); the Task Console's by default.
                            let Some(shown) = call.params["window"].as_str().map_or(Some(windows::Kind::TaskConsole), windows::Kind::parse) else {
                                answer(Err("Name a window: devices, task-console, pulse-editor, figure-viewer-<n>, control:<device> or log:<device>".into())); return Ok(());
                            };
                            let reply = call.reply.clone();
                            let captured = match windows.get(&shown) {
                                Some(window) => window.pixels.capture(move |png| {
                                    let _ = reply.send(png.and_then(|png| Ok(zlc_bus::mcp::ToolOutput { json: match &path {
                                        Some(path) => write_image(path, &png)?,
                                        None => { let (width, height) = pixels::png_size(&png).unwrap_or_default(); json!({"width": width, "height": height}) }
                                    }, png: Some(png) })));
                                }),
                                None => Err(format!("The {} window is not open", shown.title())),
                            };
                            if let Err(error) = captured { answer(Err(error)); }
                        }
                        "close_experiment" => {
                            // v2 lab_close(experiment): unsaved drafts first say so; then as the window's close button.
                            let shown = pulse.state();
                            let unsaved: Vec<&str> = [("dirty", "the pulse"), ("config", "its Config values"), ("subpulse", "a subpulse")].into_iter()
                                .filter(|(key, _)| if *key == "dirty" { shown["dirty"] == true } else { shown[*key]["dirty"] == true }).map(|(_, what)| what).collect();
                            if !unsaved.is_empty() && call.params["discard_changes"] != true {
                                answer(Err(format!("The Pulse Editor has unsaved changes to {}: save them (pulse_save, pulse_config, pulse_subpulse) or close with discard_changes true",
                                                   unsaved.join(", "))));
                                return Ok(());
                            }
                            answer(Ok(zlc_bus::mcp::ToolOutput { json: json!({"closing": true}), png: None }));
                            if worker_failed { *flow = ControlFlow::Exit; return Ok(()); }
                            closing = true;
                            tools.set_closing();
                            builtin.stop_all();
                            pulse.shutdown();
                            queue.clear();
                            queue.push_back(json!({"id":0,"action":"close"}));
                        }
                        "get_panel_image" => {
                            let panel = call.params["panel"].as_str().unwrap_or("");
                            answer(match latest_fronts.get(panel) {
                                Some(front) => remote::png(front).map(|png| zlc_bus::mcp::ToolOutput { png: Some(png), json: json!({"panel": panel,
                                    "width": front.width, "height": front.height, "sequence": (std::iter::once(&accepted).chain(viewers.values().map(|viewer| &viewer.state)).find_map(|state| state["panels"].as_array()
                                        .and_then(|rows| rows.iter().find(|row| row["id"] == panel)).map(|row| row["frame"]["sequence"].clone())))}) }),
                                None => Err(format!("{panel} has no accepted picture yet")),
                            });
                        }
                        "wait" => {
                            next_remote += 1;
                            match remote::waiter(next_remote, &call, &accepted) {
                                Ok(waiter) => {
                                    let (id, seconds) = (waiter.id, call.params["timeout_seconds"].as_f64().unwrap_or(30.0).clamp(0.0, remote::WAIT_SECONDS));
                                    waiters.push(waiter);
                                    let timer = self_proxy.clone();
                                    std::thread::Builder::new().name("zlc-mcp-wait".into()).spawn(move || {
                                        std::thread::sleep(std::time::Duration::from_secs_f64(seconds));
                                        let _ = timer.send_event(AppEvent::WaitExpired(id));
                                    }).map_err(|error| error.to_string())?;
                                }
                                Err(error) => answer(Err(error)),
                            }
                        }
                        _ => {
                            next_remote += 1;
                            match remote::command_for(&call, next_remote, &accepted, &root) {
                                Ok(command) => {
                                    remote_pending.insert(next_remote, remote::pending(&call, &accepted));
                                    self_proxy.send_event(AppEvent::Command { window: None, command }).map_err(|_| "event loop closed".to_owned())?;
                                }
                                Err(error) => answer(Err(error)),
                            }
                        }
                    }
                }
                Event::UserEvent(AppEvent::WaitExpired(id)) => remote::check_waiters(&accepted, &mut waiters, Some(id)),
                Event::UserEvent(AppEvent::ImageSaved { request, result }) => {
                    let mut state = accepted.clone();
                    state["request"] = json!(request);
                    match result {
                        Ok(saved) => { state["result"] = saved; state["error"] = Value::Null; }
                        Err(error) => { state["result"] = Value::Null; state["error"] = json!(error); }
                    }
                    if !closing { windows.send(&state)?; }
                }
                Event::UserEvent(AppEvent::Title { window, title }) => if let Some(shown) = windows.get(&window) { shown.window.set_title(&title); },
                Event::UserEvent(AppEvent::Caption { window, message }) => { windows.send_to(&window, &message)?; }
                Event::UserEvent(AppEvent::Activate(window)) => windows.activate(&window),
                Event::UserEvent(AppEvent::ShareChanged) => {
                    accepted["share"] = share.state();
                    if !closing { windows.send(&accepted)?; }
                }
                Event::UserEvent(AppEvent::RuntimeChanged) => {
                    // A viewer's panel derived a signal (v2: a saved Figure's board keeps its
                    // selector outputs): its document lists it and its consumers redraw.
                    if viewers.values().any(|viewer| !viewer.closing && !viewer.bindings.is_empty()) {
                        let sources = runtime.signals();
                        for (number, viewer) in viewers.iter_mut().filter(|(_, viewer)| !viewer.closing && !viewer.bindings.is_empty()) {
                            let prefix = format!("@logic/{}-", panels::viewer_prefix(*number));
                            let owned: Vec<String> = sources.iter().filter(|source| source.signal.starts_with(&prefix))
                                .map(|source| format!("{}#{:?}#{}", source.signal, source.generation.id(), source.published)).collect();
                            if owned != viewer.derived { viewer.derived = owned; viewer.pending = true; }
                        }
                    }
                    // Following processors and previews start on new publications.
                    let mut host = workbench_host::DesktopHost { runtime: &runtime, bindings: &mut bindings, accepted: &accepted,
                        queue: &mut queue, targets: &mut pending_targets };
                    if workbench.on_runtime(&mut host) {
                        workbench_host::deliver(&mut workbench, &builtin, &mut host);
                        workbench_host::decorate(&workbench, &runtime, &bindings, &pending_targets, &tools, &mut accepted);
                        if !closing { windows.send(&accepted)?; }
                    }
                }
                // An SLM Editor's draft, told by its device: its window's state and pictures.
                Event::UserEvent(AppEvent::Plugin(zlc_plugin_host::HostEvent::Notice { instance, name, value, lent })) => {
                    if name != "slm_editor" { return Ok(()); }
                    crate::trace::point("slm.notice", || json!({"device": instance}));
                    let message = slm.notice(&instance, &value, &lent)?;
                    if !closing { windows.send_to(&windows::Kind::SlmEditor(instance), &message)?; }
                }
                Event::UserEvent(AppEvent::Slm(drawn)) => {
                    if let Some((device, message, fronts)) = slm.drawn(drawn) {
                        let kind = windows::Kind::SlmEditor(device);
                        if !closing {
                            windows.send_to(&kind, &message)?;
                            remote::settle(&message, &mut remote_pending);
                            for (panel, sequence, front) in fronts { windows.submit(panel, sequence, front)?; }
                        }
                    }
                }
                // Plugin facts (host events, answers of calls made off this turn) reach the workbench here.
                Event::UserEvent(event @ (AppEvent::Plugin(_) | AppEvent::Workbench(_) | AppEvent::CatalogReady)) => {
                    let events = match event {
                        AppEvent::Plugin(event) => builtin.host_event(event),
                        AppEvent::Workbench(event) => {
                            // A sequencer lost with its computer lets its board go: the device is held failed.
                            if let zlc_workbench::Event::Failed { instance, .. } = &event { pulse.lose_device(instance); }
                            std::iter::once(event).chain(builtin.replay()).collect()
                        }
                        _ => { trace::point("startup.catalog", || json!({})); builtin.catalog_ready(); workbench.refresh_catalog(); Vec::new() }
                    };
                    let mut host = workbench_host::DesktopHost { runtime: &runtime, bindings: &mut bindings, accepted: &accepted,
                        queue: &mut queue, targets: &mut pending_targets };
                    for event in events { workbench.on_event(event, &mut host); }
                    workbench_host::deliver(&mut workbench, &builtin, &mut host);
                    workbench.on_runtime(&mut host);
                    accepted["acquisition"] = builtin.acquisition();
                    for (request, result) in workbench.take_replies() {
                        accepted["request"] = json!(request);
                        match result {
                            Ok(value) => { accepted["result"] = value; accepted["error"] = Value::Null; }
                            Err(error) => { accepted["result"] = Value::Null; accepted["error"] = json!(error); }
                        }
                        remote::settle(&accepted, &mut remote_pending);
                    }
                    workbench_host::decorate(&workbench, &runtime, &bindings, &pending_targets, &tools, &mut accepted);
                    if !closing { windows.send(&accepted)?; }
                    tell_editor_claims(&mut pulse, &workbench);
                }
                // v2 _route_exact_panel_selection: an Edit's region is its panel's.
                Event::UserEvent(AppEvent::Selection(note)) => {
                    let panel = note.panel.strip_prefix(panels::EDIT_PREFIX).unwrap_or(&note.panel).to_owned();
                    if workbench.panel_selection(&panel, &note.signal, note.selection.as_ref(), &note.context) && !closing {
                        workbench_host::decorate(&workbench, &runtime, &bindings, &pending_targets, &tools, &mut accepted);
                        windows.send(&accepted)?;
                    }
                }
                Event::UserEvent(AppEvent::RebaseNeeded) => {pending.store(true,Ordering::Release); for viewer in viewers.values_mut() { viewer.pending = true; }}
                Event::UserEvent(AppEvent::Pulse(event)) => {
                    let mut state = json!({"type": "state"});
                    match event {
                        pulse_host::PulseEvent::Outcome(outcome) => { pulse.editor.deliver(outcome); pulse.dispatch(); }
                        pulse_host::PulseEvent::Reply { request, result, touched } => {
                            state["request"] = json!(request);
                            match result { Ok(value) => state["result"] = value, Err(error) => state["error"] = json!(error) }
                            pulse.replied(touched);
                        }
                        pulse_host::PulseEvent::Opened { instance, result } => {
                            let event = match pulse.device_opened(&instance, &workbench.device_label(&instance), result) {
                                Ok(point) => zlc_workbench::Event::Opened { instance, working_point: point },
                                Err(error) => zlc_workbench::Event::Failed { instance, error },
                            };
                            let _ = self_proxy.send_event(AppEvent::Workbench(event));
                        }
                        pulse_host::PulseEvent::Closed { instance, error } => {
                            let _ = self_proxy.send_event(AppEvent::Workbench(zlc_workbench::Event::Closed { instance, error }));
                        }
                        pulse_host::PulseEvent::Preview { result, key } => {
                            if let Some((sequence, front)) = pulse.preview_drawn(result, key) {
                                windows.submit(pulse_host::PREVIEW_PANEL.to_owned(), sequence, front)?;
                            }
                        }
                        pulse_host::PulseEvent::Gesture(result) => {
                            if let Some((sequence, front)) = pulse.gestured(result) {
                                windows.submit(pulse_host::PREVIEW_PANEL.to_owned(), sequence, front)?;
                            }
                        }
                        pulse_host::PulseEvent::PreviewSaved(result) => pulse.preview_saved(result),
                        pulse_host::PulseEvent::Log { instance, line } => {
                            let _ = self_proxy.send_event(AppEvent::Workbench(zlc_workbench::Event::Log { instance, line }));
                        }
                    }
                    state["pulse"] = pulse.state();
                    // A page is told only what changed (delta.rs): an unchanged pulse sends nothing.
                    remote::settle(&state, &mut remote_pending);
                    if !closing { windows.send(&state)?; }
                    pulse.recycle(&mut state["pulse"]);
                }
                Event::UserEvent(AppEvent::Native(request)) => match request {
                    workbench_host::NativeRequest::Open { instance, type_id, parameters, context } => pulse.open_device(&instance, &type_id, &parameters, &context),
                    workbench_host::NativeRequest::Close { instance } => pulse.close_device(&instance),
                    workbench_host::NativeRequest::Call { ticket, op } => {
                        let answer = if op == "fields" { json!([]) } else { json!({}) };
                        let _ = self_proxy.send_event(AppEvent::Workbench(zlc_workbench::Event::Answered { ticket, result: Ok(answer) }));
                    }
                    // Remote for the pulse board on this computer: its own server admits the other
                    // computers, and the fabric announces it as a sequencer.hardware at that server.
                    workbench_host::NativeRequest::Publish { ticket, instance, role, published } => {
                        let result = (|| -> Result<Value, String> {
                            let bench = builtin.bench().ok_or("This computer is not on the bench fabric")?;
                            let host = plugins.as_ref().map_err(|error| error.clone())?;
                            if !published {
                                bench.publish(host, &instance, &role, false)?;
                                pulse.admit_peers(&instance, false)?;
                                return Ok(json!({}));
                            }
                            let port = pulse.admit_peers(&instance, true)?;
                            bench.publish_own(host, &instance, &role, pulse_host::ANNOUNCED_AS, "Pulse board", json!({"port": port}))
                                .inspect_err(|_| { let _ = pulse.admit_peers(&instance, false); })
                        })();
                        let _ = self_proxy.send_event(AppEvent::Workbench(zlc_workbench::Event::Answered { ticket, result }));
                    }
                },
                Event::UserEvent(AppEvent::PulsePoll) => {
                    if !closing {
                        tell_editor_claims(&mut pulse, &workbench);
                        pulse.editor.poll();
                        pulse.dispatch();
                    }
                }
                _ => {}
            }
            // A panel held inside its update interval may now take its new data.
            if cadence.owed(std::time::Instant::now()) { pending.store(true, Ordering::Release); }
            if !active && !worker_failed {
                let mut inputs = Vec::new();
                // v2 Pause freezes only the picture: the live board's new data waits (pending stays set) until Resume.
                let live = !closing && !workbench.paused() && pending.load(Ordering::Acquire);
                // The board's due data goes before a waiting pointer, and a waiting pointer before the
                // board's next data: a gesture never holds the picture back, nor the picture a gesture.
                let board_turn = live && queue.front().is_none_or(|next| next["action"] == "pointer" && !board_last);
                let board = if !board_turn { None } else if bindings.is_empty() {
                    pending.store(false, Ordering::Release);
                    bindings.decorate(&mut accepted, &runtime, |signal| !archive::viewer_owned(signal));
                    windows.send(&accepted)?;
                    None
                } else {
                    // Only the panels past their interval take the new data (v2 BoardScheduler);
                    // while none may, the data waits for the first that may.
                    cadence.plan(&workbench.panel_intervals(), std::time::Instant::now())
                        .map(|held| json!({"id":0,"action":"sync_panels","held":held}))
                };
                let synced = board.is_some();
                let command = if board.is_some() { board } else if let Some(command) = queue.pop_front() { Some(command) } else if live { None }
                else if let Some(number) = viewers.iter_mut().filter(|_| !closing).find(|(_, viewer)| viewer.pending && !viewer.closing && !viewer.bindings.is_empty())
                    .map(|(number, viewer)| { viewer.pending = false; *number }) {
                    // A viewer panel's own output (an ROI of an archived image) moved on.
                    Some(json!({"id":0,"action":"sync_panels","document":number}))
                } else { None };
                if let Some(mut command) = command {
                    board_last = synced;
                    let document = command.as_object_mut().and_then(|object| object.remove("document")).and_then(|document| document.as_u64()).unwrap_or(0) as u32;
                    let in_viewer = document != 0;
                    let (shown, set) = match viewers.get(&document) { Some(viewer) => (&viewer.state, &viewer.bindings), None => (&accepted, &bindings) };
                    let importing=if command["action"]=="present_archive" {
                        Some(pending_archive.as_ref().filter(|candidate|candidate.request==command["id"].as_u64().unwrap_or(0))
                            .ok_or("Figure candidate belongs to another request")?)
                    }else{None};
                    // Authoring belongs to the native accepted panel, not to a
                    // solver cache. A missing fit RESULT cannot erase its request.
                    let mut targets = serde_json::Map::new();
                    let mut target_requests=serde_json::Map::new();
                    if let Some(panels) = shown["panels"].as_array() {
                        for panel in panels {
                            if let Some(id) = panel["id"].as_str() {
                                targets.insert(id.to_owned(), panel["target"].clone());
                                target_requests.insert(id.to_owned(),panel["target_request"].clone());
                            }
                        }
                    }
                    // A panel the workbench added (a layout, a preview) is prepared with its target.
                    for (panel, target) in pending_targets.iter().filter(|_| !in_viewer) {
                        if targets.get(panel).is_none_or(Value::is_null) { targets.insert(panel.clone(), target.clone()); }
                    }
                    // A data preview starts from the archive recipe its copy kept.
                    for (panel, target) in viewers.get(&document).into_iter().flat_map(|viewer| viewer.targets.iter()) {
                        if targets.get(panel).is_none_or(Value::is_null) { targets.insert(panel.clone(), target.clone()); }
                    }
                    if let Some(target) = command["panel"].as_str().and_then(|id| targets.get_mut(id)).filter(|target|target.is_object()) {
                        for key in ["semantic", "parameters"] {
                            if let Some(patch) = command[key].as_object() {
                                if let Some(values) = target[key].as_object_mut() { values.extend(patch.clone()); }
                            }
                        }
                        // Interaction state a client sets directly (selectors, viewport, focus,
                        // thresholds) replaces the target field, exactly as a gesture would.
                        for key in ["fit", "size", "selectors", "viewport", "facet_focus", "classifier_thresholds"] {
                            if let Some(value) = command.get(key) { target[key] = value.clone(); }
                        }
                    }
                    command["targets"] = Value::Object(targets);
                    if let Some(candidate)=importing {command["targets"]=Value::Object(candidate.data.targets());}
                    if !matches!(command["action"].as_str(),Some("describe"|"save"|"ready"|"get_fit")) && command["id"].as_u64().unwrap_or(0)!=0 {
                        if let Some(panel)=command["panel"].as_str() {target_requests.insert(panel.to_owned(),command["id"].clone());}
                    }
                    command["target_requests"]=Value::Object(target_requests);
                    command["published_outputs"]=Value::Object(workbench.published_outputs());
                    if let Some(candidate)=importing {
                        inputs=candidate.data.inputs();
                    } else if command["action"] == "sync_panels" && in_viewer {
                        if let Some(viewer) = viewers.get_mut(&document) {
                            viewer.bindings.sync_generations(&runtime);
                            viewer.bindings.follow_runs(&runtime, &viewer.state, &Default::default());
                            inputs = viewer.bindings.capture(&runtime,&command["targets"],&viewer.state,&Default::default());
                        }
                    } else if command["action"] == "sync_panels" {
                        bindings.sync_generations(&runtime);
                        // Clear before the atomic read: a commit during/after it
                        // leaves pending set for the next turn, not a lost tail.
                        pending.store(false, Ordering::Release);
                        bindings.follow_runs(&runtime, &accepted, &workbench.panel_overlays());
                        inputs = bindings.capture(&runtime,&command["targets"],&accepted,&workbench.panel_overlays());
                        command["reset_document"] = json!(std::mem::take(&mut reset_document));
                    } else if !matches!(command["action"].as_str(),Some("describe"|"save"|"open"|"close"|"close_panel"|"ready"|"get_fit")) {
                        if let Some(panel)=command["panel"].as_str() {
                            let overlays = if in_viewer { Default::default() } else { workbench.panel_overlays() };
                            inputs=set.capture(&runtime,&command["targets"],shown,&overlays).into_iter().filter(|input|input.panel==panel).collect();
                        }
                    }
                    ticket += 1;
                    trace::point("job.dispatch",||json!({"job":ticket,"action":command["action"],"queue":queue.len(),
                        "inputs":inputs.iter().map(|input|json!({"panel":input.panel,"window":input.window,"epoch":input.epoch,
                            "source":input.read.as_ref().map(|read|trace::key(read.key()))})).collect::<Vec<_>>()}));
                    if !in_viewer && importing.is_none() { cadence.dispatch(std::time::Instant::now()); }
                    let selected=importing.map(|candidate|&candidate.bindings).unwrap_or(viewers.get(&document).map_or(&bindings, |viewer| &viewer.bindings));
                    let archive=importing.map(|candidate|candidate.data.clone());
                    if let Err(error) = sender.send(science::Job{ticket,command,inputs,authorities:selected.authorities(),
                        intents:selected.intent_authorities(),archive,document:importing.map_or(document, |candidate| candidate.document)}) {
                        worker_failed = true;
                        if closing { *flow = ControlFlow::Exit; }
                        return Err(error.to_string());
                    }
                    active = true;
                }
            }
            Ok(())
        })();
        if let Err(error) = result {
            eprintln!("{}", json!({"type":"desktop-error","error":error}));
            accepted["error"] = json!(error);
            if let Some(request) = error_request { accepted["request"] = request; }
            if let Err(error) = windows.send(&accepted) { eprintln!("Cannot report UI error: {error}"); }
        }
        // v2's opening rules: Init opens TaskConsole in front of Devices; Shut down puts Devices back.
        let now = matches!(accepted["apparatus"]["action"].as_str(), Some("apply" | "shut_down"));
        if now != initialized && !closing {
            initialized = now;
            let shown = if now { windows.show(&windows::Kind::TaskConsole, target) } else { windows.hide(&windows::Kind::TaskConsole); windows.show(&windows::Kind::Devices, target) };
            if let Err(error) = shown { eprintln!("{}", json!({"type":"desktop-error","error":error})); }
        }
        for kind in windows.kinds().cloned().collect::<Vec<_>>() {
            if let windows::Kind::Control(device) | windows::Kind::Log(device) | windows::Kind::SlmEditor(device) = &kind {
                if !live_device(&accepted, device) { windows.close(&kind); slm.forget(device); }
            }
        }
        remote::settle(&accepted, &mut remote_pending);
        remote::check_waiters(&accepted, &mut waiters, None);
        // The turn at which a held panel may take its new data.
        beat.arm(cadence.wake_at());
        if *flow == ControlFlow::Exit {
            // No client can reach this instrument any more.
            let _ = std::fs::remove_file(&connection_record);
            share.stop();
            if let Ok(host) = &plugins { host.shutdown(); }
            runtime.close();
            drop(pending_archive.take());viewers.clear();
            if let Some(listener) = watch_listener.take() {
                if listener.join().is_err() { eprintln!("Runtime change listener panicked"); }
            }
            trace::flush();
        }
    });
}
