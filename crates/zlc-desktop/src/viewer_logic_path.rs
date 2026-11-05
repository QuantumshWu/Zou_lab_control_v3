//! A saved Figure in a Figure Viewer, a real plugin process running the real Derive on it: the same
//! `Logic` the window drives, the document the window draws with. ZLC_PLUGIN_PYTHON names the
//! interpreter, ZLC_VIEWER_FILE the saved `repeat` x `rabi.duration` x `Site` scan, ZLC_VIEWER_OUT
//! a folder for what the page's own checks read (the rows at each step, the panel's picture).
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use zlc_bus::mcp::ToolOutput;
use zlc_core::control::{Origin, SafeActions, node_instance};
use zlc_core::runtime::Runtime;
use zlc_plugin_host::{HostConfig, HostEvent, PluginHost};
use zlc_workbench::Event;

use super::*;
use crate::archive::MountedArchive;
use crate::panels::Panels;
use crate::science::{Front, Job, PresentationDecision, Science, SelectionNote};

const NODE: &str = "figure-1-derive";
const FREQUENCY: &str = "@logic/figure-1-derive/rabi_baseband_frequency";
/// The longest the window's turn may be held by anything of the Viewer's logic.
const TURN: Duration = Duration::from_millis(250);

fn variables(every: usize) -> Value {
    json!([{"name": "a", "publish": false, "terms": [{"of": "data"}], "statistic": "mean", "over": ["repeat", "Site"], "by": "repeat", "every": every},
           {"name": "rabi", "terms": [{"of": "a"}], "statistic": "fit", "over": ["rabi.duration"], "model": "damped_sine"}])
}

struct Stage {
    host: PluginHost, runtime: Runtime, archive: Arc<MountedArchive>, logic: Option<Logic>, science: Science, out: Option<PathBuf>,
    events: mpsc::Receiver<HostEvent>, replies: mpsc::Receiver<Event>, next_request: u64, answers: BTreeMap<u64, Result<Value, String>>, longest: Duration,
}

impl Stage {
    fn offered(archive: &MountedArchive, signal: &str) -> bool { crate::viewer_offers(1, Some(archive), &[], signal) }

    /// One piece of the window's work on its logic, as a turn takes it.
    fn turn(&mut self, work: impl FnOnce(&mut Logic, &mut Scope)) {
        let archive = self.archive.clone();
        let offers = move |signal: &str| Self::offered(&archive, signal);
        let mut scope = Scope { runtime: &self.runtime, offers: &offers };
        let started = Instant::now();
        let logic = self.logic.as_mut().expect("open");
        work(&mut *logic, &mut scope);
        self.answers.extend(logic.take_replies());
        self.longest = self.longest.max(started.elapsed());
    }

    fn rows(&self) -> Value {
        let archive = self.archive.clone();
        let offers = move |signal: &str| Self::offered(&archive, signal);
        self.logic.as_ref().expect("open").rows(&Scope { runtime: &self.runtime, offers: &offers })
    }

    fn command(&mut self, name: &str, params: Value) -> u64 {
        self.next_request += 1;
        let request = self.next_request;
        self.turn(|logic, scope| logic.command(1, name, &params, &Origin::Operator, request, scope));
        request
    }

    /// Everything the window would be told, until `reached` holds of its node's row.
    fn until(&mut self, what: &str, limit: Duration, reached: impl Fn(&Value) -> bool) -> Value {
        let deadline = Instant::now() + limit;
        loop {
            while let Ok(event) = self.events.try_recv() { self.turn(|logic, scope| logic.host_event(event, scope)); }
            while let Ok(event) = self.replies.try_recv() { self.turn(|logic, scope| logic.answered(event, scope)); }
            self.turn(|logic, scope| logic.runtime_changed(scope));
            let rows = self.rows();
            if let Some(row) = rows.as_array().and_then(|rows| rows.first()).filter(|row| reached(row)) { return row.clone(); }
            assert!(Instant::now() < deadline, "{what}: phase {} state {} status {} issues {} outputs {} run {}", rows[0]["phase"], rows[0]["state"], rows[0]["status"], rows[0]["issues"], rows[0]["outputs"], rows[0]["run"]);
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    fn answer(&mut self, request: u64, what: &str) -> Result<Value, String> {
        let deadline = Instant::now() + Duration::from_secs(60);
        while !self.answers.contains_key(&request) {
            while let Ok(event) = self.events.try_recv() { self.turn(|logic, scope| logic.host_event(event, scope)); }
            while let Ok(event) = self.replies.try_recv() { self.turn(|logic, scope| logic.answered(event, scope)); }
            assert!(Instant::now() < deadline, "{what} was not answered");
            std::thread::sleep(Duration::from_millis(2));
        }
        self.answers.remove(&request).unwrap()
    }

    fn dump(&self, name: &str, value: &Value) {
        if let Some(out) = &self.out { std::fs::write(out.join(format!("{name}.json")), serde_json::to_vec_pretty(value).unwrap()).unwrap(); }
    }

    fn finished(&mut self, what: &str, every: usize) -> Value {
        let groups = 1000 / every;
        let row = self.until(what, Duration::from_secs(120), |row| {
            row["phase"] == "idle" && row["run"]["ended_at"].is_string()
                && row["outputs"].as_array().is_some_and(|outputs| outputs.iter().any(|output| output["signal"] == FREQUENCY && output["shape"].as_str().is_some_and(|shape| shape.starts_with(&format!("({groups} ")))))
        });
        assert_eq!(row["run"]["state"], "completed", "{what}: {}", row["run"]);
        row
    }

    /// What the document draws of `panel`, through the worker's own job; the state with what the page reads.
    fn draw(&mut self, bindings: &mut Panels, ticket: u64, mut command: Value, shown: &Value, target: &Value) -> (Value, Option<Front>) {
        bindings.sync_generations(&self.runtime);
        let panels: Vec<String> = bindings.authorities().keys().cloned().collect();
        let targets: serde_json::Map<String, Value> = panels.iter().map(|panel| (panel.clone(), target.clone())).collect();
        command["targets"] = Value::Object(targets);
        command["target_requests"] = json!({});
        command["published_outputs"] = json!({});
        // As the window: a save and a page's ready read nothing anew.
        let inputs = if command["action"] == "sync_panels" { bindings.capture(&self.runtime, &command["targets"], shown, &Default::default()) } else { Vec::new() };
        let job = Job { ticket, command, inputs, authorities: bindings.authorities(), intents: bindings.intent_authorities(), archive: None, document: 1 };
        let mut completion = self.science.execute(&job);
        let (mut state, fronts) = std::mem::replace(&mut completion.result, Err(String::new())).expect("the document drew");
        let targets: BTreeSet<String> = panels.into_iter().collect();
        let drawn: BTreeSet<String> = fronts.keys().cloned().collect();
        self.science.finish_presentation(PresentationDecision { targets, fronts: drawn });
        let archive = self.archive.clone();
        bindings.decorate(&mut state, &self.runtime, |signal| Self::offered(&archive, signal));
        (state, fronts.into_values().next())
    }
}

fn parameter(runtime: &Runtime, signal: &str) -> (String, Vec<f64>, Vec<f64>) {
    let source = runtime.signals().into_iter().rev().find(|item| &*item.signal == signal && item.published).unwrap_or_else(|| panic!("{signal}"));
    let latest = runtime.read_latest(&source.generation, signal).unwrap();
    let dataset = latest.canonical().expect("a finite output");
    let mut rows = Vec::new();
    for placement in dataset.segments() {
        let zlc_core::AnyBuffer::F64(values) = placement.data().values() else { panic!("dtype") };
        let sigma = placement.data().sigma().expect("a standard error");
        for (k, (value, error)) in values.as_slice().iter().zip(sigma.as_slice()).enumerate() { rows.push((placement.origin()[0] + k, *value, *error)); }
    }
    rows.sort_by_key(|row| row.0);
    (dataset.schema().value().unit().unwrap_or("").to_owned(), rows.iter().map(|row| row.1).collect(), rows.iter().map(|row| row.2).collect())
}

fn stage(python: std::ffi::OsString, file: &Path, out: Option<PathBuf>) -> Stage {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let config = HostConfig { python: python.into(), python_path: vec![root.join("../../science/src")], packages: vec!["zlc_atom".into()], memory_limit_mib: Some(8192) };
    let runtime = Runtime::new();
    let (sender, events) = mpsc::channel();
    let sender = Mutex::new(sender);
    let host = PluginHost::new(config, runtime.clone(), SafeActions::new(), move |event| { let _ = sender.lock().unwrap().send(event); }).unwrap();
    let science = Science::new(runtime.clone(), Arc::new(|error: String| panic!("{error}")), Arc::new(|| {}), Arc::new(|_: SelectionNote| {}));
    let decoded = science.open(file.to_str().unwrap()).unwrap();
    let archive = Arc::new(MountedArchive::mount(runtime.clone(), decoded, vec!["figure-1-1".to_owned()]).unwrap());
    let (sender, replies) = mpsc::channel();
    let sender = Mutex::new(sender);
    // What main.rs's `fit_models` arm answers; any other command is not this path's.
    let core: Arc<dyn Fn(crate::remote::RemoteCall) -> Result<(), String> + Send + Sync> = Arc::new(|call| {
        assert_eq!(call.name, "fit_models");
        call.reply.send(Ok(ToolOutput { json: crate::run_fit::fit_models(), png: None })).map_err(|error| error.to_string())
    });
    host.catalog().unwrap();
    let workspace = std::env::temp_dir().join(format!("zlc-viewer-logic-{}", std::process::id()));
    let logic = Logic::new(runtime.clone(), host.clone(), &workspace, Arc::new(move |event: Event| { let _ = sender.lock().unwrap().send(event); }), core).unwrap();
    Stage { host, runtime, archive, logic: Some(logic), science, out, events, replies, next_request: 0, answers: BTreeMap::new(), longest: Duration::ZERO }
}

/// A host fact goes to the document whose node it is of; a device named like a Viewer's node is no Viewer's.
#[test]
fn a_hosts_facts_go_to_the_viewer_of_their_node() {
    let ready = |instance: &str| HostEvent::RunReady { instance: instance.to_owned(), run: "r".to_owned() };
    assert_eq!(viewer_of_event(&ready("node/figure-2-derive")), Some(2));
    assert_eq!(viewer_of_event(&ready("node/derive")), None);
    assert_eq!(viewer_of_event(&ready("figure-2-derive")), None);
    let failed = |instances: &[&str]| HostEvent::Failed { instances: instances.iter().map(|name| (*name).to_owned()).collect(), reason: "lost".to_owned() };
    assert!(split_failed(&failed(&["node/figure-2-a", "node/figure-2-b"])).is_none());
    let split = split_failed(&failed(&["node/figure-2-a", "node/occupancy"])).expect("told apart");
    assert_eq!(split.iter().map(viewer_of_event).collect::<Vec<_>>(), [Some(2), None]);
}

/// ZLC_VIEWER_FILE: from the Viewer's Add to the saved result reopened. The whole path, in the order the operator goes it.
#[test]
fn a_viewers_processor_fits_the_saved_figure() {
    let (Some(python), Some(file)) = (std::env::var_os("ZLC_PLUGIN_PYTHON"), std::env::var_os("ZLC_VIEWER_FILE")) else {
        eprintln!("ZLC_PLUGIN_PYTHON and ZLC_VIEWER_FILE are not set: skipping");
        return;
    };
    let out = std::env::var_os("ZLC_VIEWER_OUT").map(PathBuf::from);
    if let Some(out) = &out { std::fs::create_dir_all(out).unwrap(); }
    let scratch = std::env::temp_dir().join(format!("zlc-viewer-figure-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).unwrap();
    let mut stage = stage(python, Path::new(&file), out);
    let dataset = stage.archive.items()[0].signal.to_string();
    let saved_read = stage.archive.inputs().remove(0).read.expect("the opened data").key().clone();

    // What may be added is read from the declarations.
    let catalog = stage.host.catalog().unwrap();
    let added = stage.logic.as_ref().unwrap().workbench.pure_processors();
    let offered: Vec<&str> = added.as_array().unwrap().iter().filter_map(|item| item["type_id"].as_str()).collect();
    println!("offered: {offered:?}; of {} types", catalog.types.len());
    assert!(offered.contains(&"derive"), "{offered:?}");
    for refused in ["scan", "camera_measurement", "calibration"] { assert!(!offered.contains(&refused), "{refused}"); }
    let request = stage.command("add_node", json!({"type_id": "scan", "source_signal": dataset}));
    assert!(stage.answer(request, "a measurement's add").is_err(), "a measurement cannot run on saved data");

    // Added: the same row the main window's card draws, its Inputs the dataset.
    let request = stage.command("add_node", json!({"type_id": "derive", "source_signal": dataset}));
    assert_eq!(stage.answer(request, "the add").unwrap()["node"], NODE);
    let row = stage.until("its process answers the draft, the dataset in its Inputs", Duration::from_secs(60), |row| row["editor_facts"]["inputs"][0]["name"] == "data");
    assert!(row["editor_facts"]["inputs"][0]["axes"].to_string().contains("rabi.duration"), "{}", row["editor_facts"]["inputs"]);
    assert_eq!(row["source_signal"], dataset.as_str());
    stage.dump("rows_added", &row);

    // Edited: Derive's cards, its fit's models asked by `fit_models` from the draft query.
    let request = stage.command("edit_node", json!({"node": NODE, "values": {"variables": variables(50)}}));
    stage.answer(request, "the edit").unwrap();
    let row = stage.until("the variables are typed", Duration::from_secs(60), |row| row["can_start"] == true && row["editor_facts"]["cards"].as_array().is_some_and(|cards| cards.len() == 2)
        && row["editor_facts"]["cards"][1]["statistic"]["publishes"].as_array().is_some_and(|publishes| !publishes.is_empty()));
    assert!(row["editor_facts"]["cards"][1]["statistic"]["models"].as_array().unwrap().iter().any(|model| model[0] == "damped_sine"), "{}", row["editor_facts"]["cards"][1]["statistic"]);
    stage.dump("rows_edited", &row);

    // The first evaluation, over the saved data, off the window's turn.
    let started = Instant::now();
    let request = stage.command("start_node", json!({"node": NODE}));
    stage.answer(request, "Start").unwrap();
    let running = stage.until("it starts", Duration::from_secs(60), |row| row["phase"] != "idle");
    stage.dump("rows_running", &running);
    let row = stage.finished("the first evaluation", 50);
    println!("first evaluation over the saved data: {:.2} s", started.elapsed().as_secs_f64());
    stage.dump("rows_done", &row);
    let (unit, values, sigmas) = parameter(&stage.runtime, FREQUENCY);
    assert_eq!(unit, "MHz");
    assert_eq!(values.len(), 20);
    println!("frequency, kHz: {:?}", values.iter().map(|value| (value * 1e4).round() / 10.0).collect::<Vec<_>>());
    assert!((values[0] * 1e3 - 26.1).abs() < 0.15 && (values[19] * 1e3 - 35.2).abs() < 0.15, "{values:?}");
    assert!(sigmas.iter().all(|sigma| sigma.is_finite() && *sigma < 2e-3), "{sigmas:?}");

    // Isolated by name: this Viewer's, nobody else's.
    assert!(!crate::workbench_host::signals(&stage.runtime).iter().any(|signal| signal.name.contains("figure-1-") || signal.name.starts_with("@figure/")), "the board sees none of it");
    assert!(!crate::viewer_offers(2, None, &[], FREQUENCY), "another Viewer offers none of it");
    assert!(crate::archive::viewer_owned(FREQUENCY));

    // A panel of this Viewer bound to it, drawn by the real document.
    let mut bindings = Panels::with_prefix(crate::panels::viewer_prefix(1));
    let panel = bindings.add(&stage.runtime, FREQUENCY, "curve").unwrap();
    let mut shown = json!({"panels": [], "plot_kinds": []});
    let (state, _) = stage.draw(&mut bindings, 1, json!({"id": 0, "action": "ready", "dpr": 1.5}), &shown, &Value::Null);
    shown = state;
    let (mut state, front) = stage.draw(&mut bindings, 2, json!({"id": 0, "action": "sync_panels"}), &shown, &Value::Null);
    assert!(state["panels"][0]["error"].is_null(), "{}", state["panels"][0]["error"]);
    let front = front.expect("the panel is drawn");
    if let Some(out) = &stage.out { std::fs::write(out.join("panel.png"), crate::remote::png(&front).unwrap()).unwrap(); }
    let mut board = Vec::new();
    crate::settle_board(&mut board, &mut state);
    assert!(state["board"]["panels"][0]["signal_tree"].to_string().contains(FREQUENCY), "the signal tree offers it: {}", state["board"]["panels"][0]["signal_tree"]);
    state["logic"] = json!([row]);
    state["document"]["path"] = json!(stage.archive.path());
    state["processor_types"] = added.clone();
    // As the window tells the page of a panel with its picture (main.rs).
    state["panels"][0]["frame"] = json!({"width": front.width, "height": front.height, "sequence": 1});
    state["panels"][0]["caption_id"] = json!(crate::panels::viewer_caption(&panel));
    stage.dump("viewer_state", &state);
    let record = stage.archive.record();
    stage.dump("record", &json!({"info": record.tabs, "flow": record.flow}));

    // Saved from the panel: its provenance holds the dataset it read and the node's parameters.
    let saved = scratch.join("rabi.png");
    let target = state["panels"][0]["target"].clone();
    let (answer, _) = stage.draw(&mut bindings, 3, json!({"id": 3, "action": "save", "panel": panel, "path": saved}), &state, &target);
    let written: Vec<String> = answer["saved"].as_array().unwrap_or_else(|| panic!("{}", answer["saved"])).iter().filter_map(Value::as_str).map(str::to_owned).collect();
    let archive_path = written.iter().find(|path| path.to_ascii_lowercase().ends_with(".npz")).expect("the archive").clone();
    let reopened = stage.science.open(&archive_path).unwrap();
    assert_eq!(reopened.items.len(), 1);
    let lineage = reopened.lineage.read().unwrap().expect("the saved history").to_serde();
    let words = lineage.to_string();
    assert!(words.contains(&dataset) || words.contains("figure-import"), "the source dataset: {words:.800}");
    assert!(words.contains("damped_sine") && words.contains("rabi.duration"), "the node's parameters: {words:.800}");
    assert!(words.contains(NODE), "the node: {words:.800}");
    let mut info = reopened.info.clone();
    info["sections"]["lineage"] = lineage.clone();
    let record = crate::record::describe(&info);
    stage.dump("reopened_record", &json!({"tabs": record.tabs, "flow": record.flow}));
    // Shown as saved: the reopened dataset is the 20 groups, no node runs over it.
    let plane = &reopened.items[0].plane;
    assert_eq!(plane.schema().domain(zlc_core::schema::DomainRole::Repeat).size(), 20);
    assert_eq!(stage.host.instances().len(), 1, "nothing was started to show it");

    // Edited and applied: it runs again, what it replaces goes first.
    let request = stage.command("edit_node", json!({"node": NODE, "values": {"variables": variables(100)}}));
    stage.answer(request, "the second edit").unwrap();
    let edited = stage.rows()[0].clone();
    assert!(edited["edited"] == true && edited["status"] == "completed · edited", "{}", edited["status"]);
    stage.dump("rows_edited_after", &edited);
    let request = stage.command("start_node", json!({"node": NODE}));
    stage.answer(request, "the second Start").unwrap();
    stage.finished("the second evaluation", 100);
    assert_eq!(parameter(&stage.runtime, FREQUENCY).1.len(), 10);

    // A run in progress is replaced: the replacing run is the one that shows.
    let request = stage.command("edit_node", json!({"node": NODE, "values": {"variables": variables(50)}}));
    stage.answer(request, "the third edit").unwrap();
    let request = stage.command("start_node", json!({"node": NODE}));
    stage.answer(request, "the run to replace").unwrap();
    stage.until("it runs", Duration::from_secs(60), |row| row["run"]["ended_at"].is_null() && (row["run"]["state"] == "running" || row["run"]["state"] == "starting"));
    let request = stage.command("edit_node", json!({"node": NODE, "values": {"variables": variables(200)}}));
    stage.answer(request, "the edit during the run").unwrap();
    let edited = stage.rows()[0].clone();
    assert!(edited["edited"] == true && edited["status"].as_str().unwrap().contains("edited"), "{}", edited["status"]);
    stage.dump("rows_edited_running", &edited);
    let request = stage.command("start_node", json!({"node": NODE}));
    stage.answer(request, "the replacing Start").unwrap();
    stage.finished("the replacing run", 200);
    assert_eq!(parameter(&stage.runtime, FREQUENCY).1.len(), 5);
    let request = stage.command("list_runs", json!({"node": NODE}));
    let runs = stage.answer(request, "the runs").unwrap();
    let states: Vec<&str> = runs.as_array().unwrap().iter().filter_map(|run| run["state"].as_str()).collect();
    assert_eq!(states[..2], ["completed", "stopped"], "the replaced run stopped, newest first");

    // A failure is its own: the row says it, the opened data is as it was.
    let request = stage.command("edit_node", json!({"node": NODE, "values": {"variables": [{"name": "broken", "terms": [{"of": "nosuch"}]}]}}));
    stage.answer(request, "the broken edit").unwrap();
    let broken = stage.until("the issue", Duration::from_secs(60), |row| row["can_start"] == false && row["issues"].to_string().contains("nosuch"));
    stage.dump("rows_broken", &broken);
    let request = stage.command("start_node", json!({"node": NODE}));
    assert!(stage.answer(request, "the refused Start").is_err());
    let refused = stage.rows()[0].clone();
    assert_eq!(refused["state"], "error", "{}", refused["status"]);
    assert!(refused["status"].as_str().unwrap().contains("nosuch"), "{}", refused["status"]);
    stage.dump("rows_refused", &refused);
    // Its process killed under it: the real fact, told to its row alone.
    let request = stage.command("edit_node", json!({"node": NODE, "values": {"variables": variables(50)}}));
    stage.answer(request, "the repaired edit").unwrap();
    let (_, _, pid) = stage.host.instances().into_iter().find(|(name, ..)| *name == node_instance(NODE)).expect("open");
    assert!(std::process::Command::new("taskkill").args(["/F", "/T", "/PID", &pid.to_string()]).output().unwrap().status.success());
    let failed = stage.until("the loss is said", Duration::from_secs(30), |row| row["status"].as_str().is_some_and(|status| status.contains("could not start its plugin")));
    stage.dump("rows_failed", &failed);
    assert_eq!(stage.archive.inputs().remove(0).read.expect("the opened data").key(), &saved_read);

    assert!(stage.longest < TURN, "a turn of the Viewer's logic took {:?}", stage.longest);
    println!("the longest turn: {:?}", stage.longest);

    // Closed: its runs stop, its instances close, its signals are gone; the Figure is not touched by that.
    let logic = stage.logic.take().unwrap();
    let archive = stage.archive.clone();
    let offers = move |signal: &str| Stage::offered(&archive, signal);
    logic.close(&mut Scope { runtime: &stage.runtime, offers: &offers });
    let deadline = Instant::now() + Duration::from_secs(30);
    while !stage.host.instances().is_empty() || stage.runtime.signals().iter().any(|signal| signal.signal.starts_with("@logic/figure-1-derive")) {
        assert!(Instant::now() < deadline, "closed: {:?}", stage.host.instances());
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(stage.archive.inputs().remove(0).read.is_some());
    stage.host.shutdown();
    let _ = std::fs::remove_dir_all(&scratch);
}
