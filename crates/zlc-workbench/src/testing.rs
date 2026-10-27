//! A recording plugin host and application for the workbench's own tests.
use std::cell::RefCell;
use std::rc::Rc;

use serde_json::{Value, json};

use crate::{Host, PanelContent, Plugins, RunSpec, SignalInfo, TypeCatalog};

#[derive(Default)]
pub struct Calls(pub Vec<String>, pub Vec<RunSpec>, pub Vec<(String, u64, Value)>);

pub struct FakePlugins { pub calls: Rc<RefCell<Calls>>, pub catalog: Value }

/// A plugin host whose discovery answers when `ready` is set: until then it has no types.
pub struct LatePlugins { pub plugins: FakePlugins, pub ready: Rc<std::cell::Cell<bool>> }

impl Plugins for LatePlugins {
    fn catalog(&mut self) -> TypeCatalog {
        if self.ready.get() { self.plugins.catalog() } else { TypeCatalog::from_describe(&json!({"protocol": [1, 0], "types": [], "errors": []})).unwrap() }
    }
    fn discovering(&self) -> bool { !self.ready.get() }
    fn open(&mut self, instance: &str, type_id: &str, parameters: &Value, context: &Value) { self.plugins.open(instance, type_id, parameters, context) }
    fn close(&mut self, instance: &str) { self.plugins.close(instance) }
    fn query(&mut self, instance: &str, ticket: u64, name: &str, args: &Value) { self.plugins.query(instance, ticket, name, args) }
    fn call(&mut self, instance: &str, ticket: u64, op: &str, params: &Value) { self.plugins.call(instance, ticket, op, params) }
    fn start(&mut self, run: &RunSpec) { self.plugins.start(run) }
    fn stop(&mut self, run: &str) { self.plugins.stop(run) }
    fn answer(&mut self, run: &str, request: &str, response: Option<&Value>) { self.plugins.answer(run, request, response) }
    fn acquired(&mut self, run: &str, ticket: u64, result: Result<Value, String>) { self.plugins.acquired(run, ticket, result) }
    fn discover(&mut self, ticket: u64, type_id: &str, excluded: &[String]) { self.plugins.discover(ticket, type_id, excluded) }
    fn resolve(&mut self, ticket: u64, instance: &str, type_id: &str, parameters: &Value, excluded: &[String]) { self.plugins.resolve(ticket, instance, type_id, parameters, excluded) }
    fn publish(&mut self, ticket: u64, instance: &str, role: &str, published: bool) { self.plugins.publish(ticket, instance, role, published) }
}

impl Plugins for FakePlugins {
    fn catalog(&mut self) -> TypeCatalog { TypeCatalog::from_describe(&self.catalog).unwrap() }
    fn open(&mut self, instance: &str, type_id: &str, _: &Value, _: &Value) { self.calls.borrow_mut().0.push(format!("open {instance} {type_id}")); }
    fn close(&mut self, instance: &str) { self.calls.borrow_mut().0.push(format!("close {instance}")); }
    fn query(&mut self, instance: &str, ticket: u64, name: &str, args: &Value) {
        let mut calls = self.calls.borrow_mut();
        calls.0.push(format!("query {instance} {ticket} {name}"));
        calls.2.push((name.to_owned(), ticket, args.clone()));
    }
    fn call(&mut self, instance: &str, ticket: u64, op: &str, params: &Value) {
        let mut calls = self.calls.borrow_mut();
        calls.0.push(format!("call {instance} {ticket} {op}"));
        calls.2.push((op.to_owned(), ticket, params.clone()));
    }
    fn start(&mut self, run: &RunSpec) { let mut calls = self.calls.borrow_mut(); calls.0.push(format!("start {}", run.run)); calls.1.push(run.clone()); }
    fn stop(&mut self, run: &str) { self.calls.borrow_mut().0.push(format!("stop {run}")); }
    fn answer(&mut self, run: &str, request: &str, response: Option<&Value>) { self.calls.borrow_mut().0.push(format!("answer {run} {request} {}", response.is_some())); }
    fn acquired(&mut self, run: &str, ticket: u64, result: Result<Value, String>) { self.calls.borrow_mut().0.push(format!("acquired {run} {ticket} {}", result.is_ok())); }
    fn discover(&mut self, ticket: u64, type_id: &str, excluded: &[String]) {
        let mut calls = self.calls.borrow_mut();
        calls.0.push(format!("discover {ticket} {type_id}"));
        calls.2.push(("discover".into(), ticket, json!({"type_id": type_id, "excluded": excluded})));
    }
    fn resolve(&mut self, ticket: u64, instance: &str, type_id: &str, parameters: &Value, excluded: &[String]) {
        let mut calls = self.calls.borrow_mut();
        calls.0.push(format!("resolve {ticket} {instance} {type_id}"));
        calls.2.push(("resolve".into(), ticket, json!({"instance": instance, "parameters": parameters, "excluded": excluded})));
    }
    fn publish(&mut self, ticket: u64, instance: &str, _: &str, published: bool) { self.calls.borrow_mut().0.push(format!("publish {ticket} {instance} {published}")); }
}

#[derive(Default)]
pub struct FakeHost { pub signals: Vec<SignalInfo>, pub panels: Vec<PanelContent>, pub next: u64,
                      pub schemas: std::collections::BTreeMap<String, (Value, Value)> }

impl Host for FakeHost {
    fn signals(&self) -> Vec<SignalInfo> { self.signals.clone() }
    fn panels(&self) -> Vec<PanelContent> { self.panels.clone() }
    fn reserve_panel_ids(&mut self, count: usize) -> Vec<String> {
        (0..count).map(|_| { self.next += 1; format!("panel-{}", self.next) }).collect()
    }
    fn add_panel(&mut self, panel: &str, signal: &str, kind: &str, window: u64, target: Value) -> Result<String, String> {
        let panel = if panel.is_empty() { self.reserve_panel_ids(1).remove(0) } else { panel.to_owned() };
        self.panels.push(PanelContent { panel: panel.clone(), signal: signal.into(), kind: kind.into(), window, target, outputs: Vec::new() });
        Ok(panel)
    }
    fn close_panel(&mut self, panel: &str) { self.panels.retain(|content| content.panel != panel); }
    fn schemas(&self, signal: &str) -> Option<(Value, Value)> { self.schemas.get(signal).cloned() }
}

/// A camera, a sequencer with a coil output, a camera measurement, an
/// occupancy processor and a scan task that acquires through the measurement.
pub fn catalog() -> Value {
    json!({"protocol": [2, 0], "errors": [{"module": "zlc_atom.devices.camera.pylon.device_types", "error": "No module named pypylon"}], "types": [
        {"type_id": "camera.virtual_mot", "kind": "device", "family": "camera", "label": "Virtual MOT camera", "simulation": true,
         "capabilities": ["camera.adapter"], "parameters": [{"name": "exposure_seconds", "type": "float", "label": "Exposure", "default": 0.1, "minimum": 0.0}],
         "commands": [{"name": "editor_send", "summary": "Send the draft", "drives": true, "params": [], "input_schema": {"type": "object"}},
                      {"name": "editor_save", "summary": "Save the draft", "params": [{"name": "path", "type": "str", "label": "File", "required": true}],
                       "input_schema": {"type": "object"}, "file": {"param": "path", "direction": "write"}}]},
        {"type_id": "sequencer.virtual", "kind": "device", "family": "sequencer", "label": "Virtual sequencer", "simulation": true,
         "capabilities": ["sequencer.streamer"], "physical_outputs": [{"name": "mot_coils"}], "parameters": []},
        {"type_id": "camera_measurement", "kind": "node", "label": "Camera measurement", "parameters": [
            {"name": "repeat", "type": "int", "label": "Repeat", "default": 0},
            {"name": "frames_per_cycle", "type": "int", "label": "Frames per cycle", "default": 1, "minimum": 1}],
         "node": {"node_kind": "measurement", "outputs": [{"name": "frames", "contract": "zlc.camera.frames"}], "reports_ready": true,
                  "devices": [{"capability": "camera.adapter", "argument": "camera", "protected_fields": ["exposure_seconds"]}],
                  "previews": [{"output": "frames", "plot_kind": "facet_grid"}]}},
        {"type_id": "occupancy", "kind": "node", "label": "Occupancy", "parameters": [{"name": "model_kind", "type": "str", "label": "Model", "default": "default"},
            {"name": "post_selection", "type": "bool", "label": "Post-selection", "default": false}],
         "node": {"node_kind": "processor", "inputs": [{"name": "frames", "contract": null, "delivery": "exact"}], "replaces": {"post_selected_occupancy": {"post_selection": true}},
                  "outputs": [{"name": "occupied", "contract": "zlc_plot.image-point-overlay-status"}],
                  "artifact_inputs": [{"name": "calibrations", "label": "Calibrations", "frame_claims": true}]}},
        {"type_id": "seamless_scan", "kind": "node", "label": "Seamless scan", "parameters": [
            {"name": "acquisition_logic", "type": "str", "label": "Acquisition", "default": ""},
            {"name": "repeats", "type": "int", "label": "Repeats", "default": 20}],
         "node": {"node_kind": "task", "inputs": [{"name": "signal", "delivery": "exact"}], "outputs": [{"name": "scan"}],
                  "devices": [{"capability": "sequencer.streamer", "argument": "sequencer", "protected_fields": ["program"]}],
                  "acquisition_input": "acquisition_logic", "acquisition_completion": "ready"}}
    ]})
}

mod tests {
    use super::*;
    use crate::{Event, Workbench, workspace};
    use zlc_core::control::node_instance;
    use zlc_core::control::Origin;

    fn scratch(name: &str) -> std::path::PathBuf {
        let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let path = std::env::temp_dir().join(format!("zlc-workbench-{name}-{}-{stamp}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    fn workbench(root: &std::path::Path) -> (Workbench, Rc<RefCell<Calls>>) {
        let calls = Rc::new(RefCell::new(Calls::default()));
        let plugins = FakePlugins { calls: calls.clone(), catalog: catalog() };
        (Workbench::new(root, Box::new(plugins)).unwrap(), calls)
    }

    fn apply(workbench: &mut Workbench, host: &mut FakeHost, name: &str, params: Value) -> Result<Value, String> {
        workbench.apply(name, &params, &Origin::Operator, 1, host);
        let result = workbench.take_replies().pop().unwrap().1;
        settle_preparations(workbench, host);
        result
    }

    fn settle_preparations(workbench: &mut Workbench, host: &mut FakeHost) {
        loop {
            let tickets: Vec<u64> = workbench.tickets.iter().filter_map(|(ticket, value)| matches!(value, crate::Ticket::Prepare { .. }).then_some(*ticket)).collect();
            if tickets.is_empty() { break; }
            for ticket in tickets { workbench.on_event(Event::Answered { ticket, result: Ok(json!({"prepared": format!("test-{ticket}")})) }, host); }
        }
    }

    fn event(workbench: &mut Workbench, event: Event, host: &mut FakeHost) {
        workbench.on_event(event, host);
        settle_preparations(workbench, host);
    }

    #[test]
    fn an_apparatus_read_before_discovery_answers_says_only_what_is_true() {
        let root = scratch("late");
        std::fs::write(root.join("apparatus.json"), r#"{"format": "zlc.installation",
            "simulation": {"image_shape_yx": [96, 128], "grid_shape_yx": [5, 7], "seed": 0, "world_profile": ""},
            "devices": [{"instance_id": "camera", "role": "camera", "type_id": "camera.virtual_mot", "parameters": {"exposure_seconds": 0.02}},
                        {"instance_id": "old", "role": "old", "type_id": "camera.gone", "parameters": {}}]}"#).unwrap();
        let ready = Rc::new(std::cell::Cell::new(false));
        let plugins = LatePlugins { plugins: FakePlugins { calls: Rc::default(), catalog: catalog() }, ready: ready.clone() };
        let mut workbench = Workbench::new(&root, Box::new(plugins)).unwrap();
        let mut host = FakeHost::default();
        assert!(workbench.notes.is_empty() && workbench.console_report.1 != "warning", "nothing is known yet: {:?}", workbench.notes);
        assert!(apply(&mut workbench, &mut host, "apply_devices", json!({})).unwrap_err().contains("still being discovered"));
        ready.set(true);
        workbench.refresh_catalog();
        assert!(workbench.notes.len() == 1 && workbench.notes[0].starts_with("old: device type camera.gone is not installed"), "only what discovery lacks: {:?}", workbench.notes);
        assert!(!workbench.console_report.0.contains("virtual_mot"));
        let applied = apply(&mut workbench, &mut host, "apply_devices", json!({})).unwrap();
        assert_eq!(applied["opening"], json!(["camera", "old"]));
        assert_eq!(workbench.device_rows()[0]["state"], "opening");
    }

    #[test]
    fn acquisition_controls_offer_names_and_keep_unavailable_references() {
        let root = scratch("acquisition-form");
        let mut catalog = catalog();
        let types = catalog["types"].as_array_mut().unwrap();
        let mut finite = types[2].clone();
        finite["type_id"] = json!("finite");
        finite["node"]["reports_ready"] = json!(false);
        types.push(finite);
        let mut feedback = types[4].clone();
        feedback["type_id"] = json!("feedback");
        feedback["node"]["acquisition_completion"] = json!("terminal");
        feedback["parameters"][0]["required"] = json!(true);
        feedback["parameters"][0]["enabled_when"] = json!(["mode", ["external"]]);
        feedback["parameters"].as_array_mut().unwrap().push(json!({"name": "mode", "type": "str", "default": "external"}));
        types.push(feedback);
        let mut workbench = Workbench::new(&root, Box::new(FakePlugins { calls: Rc::default(), catalog })).unwrap();
        let mut host = FakeHost::default();
        for type_id in ["camera_measurement", "finite", "occupancy", "seamless_scan", "feedback"] {
            apply(&mut workbench, &mut host, "add_node", json!({"type_id": type_id})).unwrap();
        }
        let control = |workbench: &Workbench, id: &str| {
            workbench.node_row(workbench.node(id).unwrap(), &[])["controls"][0].clone()
        };
        let scan = control(&workbench, "seamless_scan");
        assert_eq!(scan["kind"], "choice");
        assert_eq!(scan["allow_none"], false);
        assert_eq!(scan["choices"], json!([["", "None"], ["camera_measurement", "camera_measurement"]]));
        assert_eq!(control(&workbench, "feedback")["choices"].as_array().unwrap().len(), 3);
        assert_eq!(control(&workbench, "feedback")["required"], true);
        apply(&mut workbench, &mut host, "edit_node", json!({"node": "seamless_scan", "values": {"acquisition_logic": "camera_measurement"}})).unwrap();
        workbench.rename_node("camera_measurement", "Imaging camera", &[]).unwrap();
        workbench.rename_node("camera_measurement", "  ", &[]).unwrap();
        assert_eq!(workbench.node("camera_measurement").unwrap().name, "camera_measurement");
        assert!(workbench.rename_node("camera_measurement", "Signals", &[]).is_err());
        let external = SignalInfo { name: "@logic/notebook/value".into(), owner: "notebook".into(), generation: "g".into(), published: true, sealed: false, shape: "".into() };
        assert!(workbench.rename_node("camera_measurement", "Notebook", &[external]).is_err());
        workbench.rename_node("camera_measurement", "Imaging camera", &[]).unwrap();
        let scan = control(&workbench, "seamless_scan");
        assert_eq!(scan["value"], "camera_measurement");
        assert_eq!(scan["choices"][1], json!(["camera_measurement", "Imaging camera"]));
        for selected in ["finite", "occupancy", "seamless_scan", "removed_camera"] {
            apply(&mut workbench, &mut host, "edit_node", json!({"node": "seamless_scan", "values": {"acquisition_logic": selected}})).unwrap();
            let scan = control(&workbench, "seamless_scan");
            assert_eq!(scan["value"], selected);
            assert_eq!(scan["choices"][2], json!([selected, format!("{selected} (unavailable)")]));
            assert!(workbench.node_row(workbench.node("seamless_scan").unwrap(), &[])["issues"].as_array().unwrap()
                .iter().any(|issue| issue.as_str().unwrap().contains("reports ready")));
        }
        apply(&mut workbench, &mut host, "edit_node", json!({"node": "feedback", "values": {"mode": "camera", "acquisition_logic": "removed_camera"}})).unwrap();
        assert_eq!(control(&workbench, "feedback")["enabled"], false);
        assert_eq!(control(&workbench, "feedback")["value"], "removed_camera");
    }

    #[test]
    fn an_apparatus_runs_a_task_through_its_measurement() {
        let root = scratch("run");
        std::fs::write(root.join("apparatus.json"), r#"{"format": "zlc.installation",
            "simulation": {"image_shape_yx": [96, 128], "grid_shape_yx": [5, 7], "seed": 0, "world_profile": ""},
            "devices": [{"instance_id": "mot_camera", "role": "mot_camera", "type_id": "camera.virtual_mot", "parameters": {"exposure_seconds": 0.02, "gone": 1}},
                        {"instance_id": "sequencer", "role": "sequencer", "type_id": "sequencer.virtual", "parameters": {}},
                        {"instance_id": "basler", "role": "basler", "type_id": "camera.pylon", "parameters": {"serial": "x"}}]}"#).unwrap();
        let (mut workbench, calls) = workbench(&root);
        let mut host = FakeHost::default();
        assert!(workbench.notes.iter().any(|note| note.contains("gone is no longer")), "{:?}", workbench.notes);
        let applied = apply(&mut workbench, &mut host, "apply_devices", json!({})).unwrap();
        assert_eq!(applied["opening"], json!(["mot_camera", "sequencer", "basler"]));
        assert_eq!(workbench.device_rows()[2]["state"], "failed", "an unknown type fails with its reason");
        for device in ["mot_camera", "sequencer"] {
            event(&mut workbench,Event::Opened { instance: device.into(), working_point: json!({}) }, &mut host);
        }
        apply(&mut workbench, &mut host, "add_node", json!({"type_id": "camera_measurement"})).unwrap();
        apply(&mut workbench, &mut host, "add_node", json!({"type_id": "seamless_scan", "values": {"acquisition_logic": "camera_measurement"},
            "source_signal": "@logic/camera_measurement/frames"})).unwrap();
        for node in ["camera_measurement", "seamless_scan"] {
            event(&mut workbench,Event::Opened { instance: node_instance(node), working_point: Value::Null }, &mut host);
        }
        assert_eq!(workbench.node("camera_measurement").unwrap().draft.device_keys["camera"], "mot_camera");
        apply(&mut workbench, &mut host, "start_node", json!({"node": "seamless_scan"})).unwrap();
        let spec = calls.borrow().1.last().cloned().unwrap();
        assert_eq!((spec.run.as_str(), spec.devices["sequencer"].as_str()), ("seamless_scan#1", "sequencer"));
        assert_eq!(spec.outputs["scan"], "@logic/seamless_scan/scan");
        let directory = spec.directory.clone().unwrap();
        assert!(directory.starts_with(workbench.workspace.today()) && directory.ends_with("seamless_scan"));
        let start = workspace::read_strict(&directory.join("start.json"), "start").unwrap();
        assert_eq!(start["schema"], "zlc.task-start");
        assert_eq!(start["input"]["devices"]["sequencer"], json!({"instance_id": "sequencer", "capability": "sequencer.streamer"}));
        // While the task runs, the board is frozen and only its acquisition starts.
        assert!(apply(&mut workbench, &mut host, "add_node", json!({"type_id": "occupancy"})).unwrap_err().contains("stop the task"));
        assert!(apply(&mut workbench, &mut host, "start_node", json!({"node": "camera_measurement"})).unwrap_err().contains("stop the task"));
        event(&mut workbench,Event::RunStarted { run: "seamless_scan#1".into() }, &mut host);
        event(&mut workbench,Event::RunAcquire { run: "seamless_scan#1".into(), ticket: 7, node: "camera_measurement".into(), completion: "ready".into() }, &mut host);
        assert_eq!(calls.borrow().1.last().unwrap().run, "camera_measurement#1");
        let refused = apply(&mut workbench, &mut host, "tune_device", json!({"device": "mot_camera", "field": "exposure_seconds", "value": 0.2}));
        assert!(refused.unwrap_err().contains("exposure_seconds: Protected by camera_measurement"));
        // A command that drives a claimed device is refused (v2 DeviceUseBusy); one that edits its draft is not, its file the workspace's.
        let busy = apply(&mut workbench, &mut host, "mot_camera__editor_send", json!({}));
        assert_eq!(busy.unwrap_err(), "device is in use by camera_measurement");
        workbench.apply("mot_camera__editor_save", &json!({"path": "data/draft.json"}), &Origin::Operator, 2, &mut host);
        let (op, _, params) = calls.borrow().2.last().cloned().unwrap();
        assert_eq!((op.as_str(), params["name"].as_str()), ("command", Some("editor_save")));
        assert_eq!(params["params"]["path"], json!(root.join("data/draft.json").display().to_string()));
        event(&mut workbench,Event::RunReady { run: "camera_measurement#1".into() }, &mut host);
        assert!(calls.borrow().0.contains(&"acquired seamless_scan#1 7 true".to_owned()));
        // The console's line (v2 _report): a run says "started" when it starts; the Task's status
        // is said when it changes, never again for the same status.
        assert_eq!(workbench.console_report.0, "camera_measurement started");
        let progress = |current| Event::RunProgress { run: "seamless_scan#1".into(), message: "point".into(), current: Some(current), total: Some(3) };
        event(&mut workbench,progress(1), &mut host);
        assert_eq!(workbench.console_report.0, "seamless_scan: point 1/3");
        workbench.console_note("another line".into());
        event(&mut workbench,progress(1), &mut host);
        assert_eq!(workbench.console_report.0, "another line");
        event(&mut workbench,progress(2), &mut host);
        assert_eq!(workbench.console_report.0, "seamless_scan: point 2/3");
        event(&mut workbench,Event::RunRequest { run: "seamless_scan#1".into(), request: json!({"request": "seamless_scan#1:1", "kind": "manual-axis", "title": "Move the knob"}) }, &mut host);
        assert_eq!(workbench.wait_reached("seamless_scan", "request").unwrap()["request"]["kind"], "manual-axis");
        assert!(apply(&mut workbench, &mut host, "answer_request", json!({"node": "seamless_scan", "request": "old", "response": {}})).is_err());
        apply(&mut workbench, &mut host, "answer_request", json!({"node": "seamless_scan", "request": "seamless_scan#1:1", "response": {}})).unwrap();
        apply(&mut workbench, &mut host, "stop_node", json!({"node": "seamless_scan"})).unwrap();
        assert!(calls.borrow().0.ends_with(&["stop seamless_scan#1".to_owned(), "stop camera_measurement#1".to_owned()]));
        apply(&mut workbench, &mut host, "close_device", json!({"device": "sequencer"})).unwrap();
        assert!(!calls.borrow().0.contains(&"close sequencer".to_owned()), "a claimed device closes after its run");
        event(&mut workbench,Event::RunFinished { run: "camera_measurement#1".into(), outcome: "stopped".into(), error: None, generation: None }, &mut host);
        event(&mut workbench,Event::RunFinished { run: "seamless_scan#1".into(), outcome: "stopped".into(), error: None, generation: Some("g".into()) }, &mut host);
        assert!(calls.borrow().0.contains(&"close sequencer".to_owned()));
        let record = workspace::read_strict(&directory.join("run.json"), "run").unwrap();
        assert_eq!((record["schema"].as_str(), record["status"]["state"].as_str()), (Some("zlc.task-run"), Some("stopped")));
        assert_eq!(record["status"]["stop_reason"], "the operator pressed Stop");
        let state = workbench.state(&host);
        assert_eq!(state["logic"][1]["status"], "Draft: Seamless scan needs a sequencer.streamer and this apparatus has none open");
        assert_eq!((state["runs"][0]["run"].as_str(), state["runs"][1]["state"].as_str()), (Some("camera_measurement#1"), Some("stopped")));
    }

    /// A scan's device axes: the editor is offered the bench's tunable devices; a Start
    /// claims the named fields and passes the device as `tunable:<key>`.
    #[test]
    fn a_draft_names_the_devices_it_tunes() {
        let root = scratch("tunes");
        let mut catalog = catalog();
        catalog["types"].as_array_mut().unwrap().extend([
            json!({"type_id": "rf.virtual", "kind": "device", "family": "rf", "label": "Virtual RF", "capabilities": ["rf.source"],
                   "physical_outputs": [{"name": "rf_out"}], "parameters": []}),
            json!({"type_id": "device_scan", "kind": "node", "label": "Device scan", "parameters": [{"name": "plan", "type": "text", "label": "Plan", "default": ""}],
                   "node": {"node_kind": "task", "outputs": [], "queries": ["resolve_device_claims", "editor_facts"]}})]);
        let calls = Rc::new(RefCell::new(Calls::default()));
        let mut workbench = Workbench::new(&root, Box::new(FakePlugins { calls: calls.clone(), catalog })).unwrap();
        let mut host = FakeHost::default();
        apply(&mut workbench, &mut host, "add_device", json!({"type_id": "rf.virtual", "instance_id": "rf"})).unwrap();
        apply(&mut workbench, &mut host, "edit_device", json!({"device": "rf", "role": "Trap RF"})).unwrap();
        apply(&mut workbench, &mut host, "apply_devices", json!({})).unwrap();
        event(&mut workbench,Event::Opened { instance: "rf".into(), working_point: json!({"state": "open", "tunable": true}) }, &mut host);
        apply(&mut workbench, &mut host, "add_node", json!({"type_id": "device_scan", "values": {"plan": "device:rf:frequency"}})).unwrap();
        event(&mut workbench,Event::Opened { instance: node_instance("device_scan"), working_point: Value::Null }, &mut host);
        let asked = |name: &str| calls.borrow().2.iter().rev().find(|(query, ..)| query == name).cloned().unwrap();
        let (_, ticket, args) = asked("editor_facts");
        assert_eq!(args["devices"]["tunable:rf"]["role"], "Trap RF");
        assert_eq!(args["devices"]["tunable:rf"]["instance"], "rf", "the editor is offered every tunable device");
        assert!(asked("resolve_device_claims").2["devices"].get("tunable:rf").is_none());
        event(&mut workbench,Event::Answered { ticket, result: Ok(json!({"ports": [{"port": "device:rf:frequency"}]})) }, &mut host);
        let state = workbench.state(&host);
        assert_eq!(state["logic"][0]["status"], "Draft: Checking the draft…");
        let (_, ticket, _) = asked("resolve_device_claims");
        event(&mut workbench,Event::Answered { ticket, result: Ok(json!({"tunable:rf": {"instance": "rf", "fields": ["frequency"]}})) }, &mut host);
        let state = workbench.state(&host);
        assert_eq!((&state["logic"][0]["editor_facts"]["ports"][0]["port"], &state["logic"][0]["can_start"]), (&json!("device:rf:frequency"), &json!(true)));
        apply(&mut workbench, &mut host, "start_node", json!({"node": "device_scan"})).unwrap();
        assert_eq!(calls.borrow().1.last().unwrap().devices["tunable:rf"], "rf");
        let refused = apply(&mut workbench, &mut host, "tune_device", json!({"device": "rf", "field": "frequency", "value": 1.0}));
        assert_eq!(refused.unwrap_err(), "frequency: Protected by device_scan");
        assert_eq!(calls.borrow().1.last().unwrap().device_labels["rf"], "Trap RF");
        let refused = apply(&mut workbench, &mut host, "tune_device", json!({"device": "rf", "field": "amplitude", "value": 1.0}));
        assert!(refused.unwrap_err().starts_with("amplitude: Stop the active Logic"), "an unnamed field is claimed, not protected");
    }

    /// A node is drawn at once from its type's answers for its default draft (the catalogue's),
    /// before its plugin has started; an edit of one field leaves the others as the newest
    /// answers drew them while the plugin checks it (v2 answered at once): the field the plugin
    /// fills keeps its value and choices, no problem of a half-checked draft shows, and a Start
    /// waits for the check.
    #[test]
    fn a_node_is_drawn_at_once_and_an_edit_keeps_the_other_fields() {
        let root = scratch("recheck");
        let mut catalog = catalog();
        let choices = json!({"window": [["short", "Short"], ["long", "Long"]]});
        catalog["types"].as_array_mut().unwrap().push(json!({"type_id": "windows", "kind": "node", "label": "Windows", "parameters": [
            {"name": "samples", "type": "int", "label": "Samples", "default": 200},
            {"name": "window", "type": "text", "label": "Window", "default": "", "required": true}],
            "node": {"node_kind": "task", "outputs": [], "queries": ["resolve_defaults", "resolve_field_choices"],
                     "answers": {"resolve_defaults": {"window": "short"}, "resolve_field_choices": choices}}}));
        let calls = Rc::new(RefCell::new(Calls::default()));
        let mut workbench = Workbench::new(&root, Box::new(FakePlugins { calls: calls.clone(), catalog })).unwrap();
        let mut host = FakeHost::default();
        let drawn = |workbench: &Workbench, host: &FakeHost| {
            let state = workbench.state(host);
            let row = &state["logic"][0];
            (row["controls"][1].clone(), row["status"].clone(), row["can_start"].clone())
        };
        let answer = |mut workbench: &mut Workbench, host: &mut FakeHost| for (name, result) in [("resolve_defaults", json!({"window": "short"})),
            ("resolve_field_choices", choices.clone())] {
            let ticket = calls.borrow().2.iter().rev().find(|(query, ..)| query == name).unwrap().1;
            event(&mut workbench,Event::Answered { ticket, result: Ok(result) }, host);
        };
        apply(&mut workbench, &mut host, "add_node", json!({"type_id": "windows"})).unwrap();
        let first = drawn(&workbench, &host);
        assert_eq!((&first.0["kind"], &first.0["value"], &first.1), (&json!("choice"), &json!("short"), &json!("not started")));
        event(&mut workbench,Event::Opened { instance: node_instance("windows"), working_point: Value::Null }, &mut host);
        answer(&mut workbench, &mut host);
        assert_eq!(drawn(&workbench, &host), first, "its own answers change nothing its type's had drawn");
        apply(&mut workbench, &mut host, "edit_node", json!({"node": "windows", "values": {"samples": 250}})).unwrap();
        assert_eq!(drawn(&workbench, &host), first, "the window field, the status and Start stay while the plugin checks");
        assert_eq!(apply(&mut workbench, &mut host, "start_node", json!({"node": "windows"})).unwrap()["phase"], "checking");
        answer(&mut workbench, &mut host);
        assert_eq!(calls.borrow().1.len(), 1, "the Start went once the draft was checked");
    }

    /// A reader's `source` fact is its producer's describe_output_frames answer for
    /// that output; a new answer asks the reader again.
    #[test]
    fn a_reader_is_told_its_source_frames() {
        let root = scratch("frames");
        let mut catalog = catalog();
        let types = catalog["types"].as_array_mut().unwrap();
        types[2]["node"]["queries"] = json!(["describe_output_frames"]);
        types.push(json!({"type_id": "survival", "kind": "node", "label": "Survival", "parameters": [],
            "node": {"node_kind": "processor", "inputs": [{"name": "occupied", "delivery": "exact"}], "outputs": [{"name": "survival"}],
                     "queries": ["editor_facts"]}}));
        let calls = Rc::new(RefCell::new(Calls::default()));
        let mut workbench = Workbench::new(&root, Box::new(FakePlugins { calls: calls.clone(), catalog })).unwrap();
        let mut host = FakeHost::default();
        apply(&mut workbench, &mut host, "add_node", json!({"type_id": "camera_measurement"})).unwrap();
        apply(&mut workbench, &mut host, "add_node", json!({"type_id": "survival", "source_signal": "@logic/camera_measurement/frames"})).unwrap();
        for node in ["camera_measurement", "survival"] { event(&mut workbench,Event::Opened { instance: node_instance(node), working_point: Value::Null }, &mut host); }
        let asked = |name: &str| calls.borrow().2.iter().rev().find(|(query, ..)| query == name).cloned().unwrap();
        assert_eq!(asked("editor_facts").2["source"], Value::Null, "nothing answered yet");
        let (_, ticket, args) = asked("describe_output_frames");
        assert_eq!((&args["output_name"], &args["source"]), (&json!("frames"), &Value::Null));
        event(&mut workbench,Event::Answered { ticket, result: Ok(json!(["frame", ["0", "1", "2"]])) }, &mut host);
        assert_eq!(asked("editor_facts").2["source"], json!(["frame", ["0", "1", "2"]]));
    }

    /// A reader of a whole bundle is told its members' schemas once they publish (v2
    /// `input_bundle`); then its admission check runs, and what it refuses is a draft issue.
    #[test]
    fn a_bundle_reader_is_told_its_members_schemas_and_admits_them() {
        let root = scratch("bundle");
        let mut catalog = catalog();
        catalog["types"].as_array_mut().unwrap().push(json!({"type_id": "derive", "kind": "node", "label": "Derive", "parameters": [],
            "node": {"node_kind": "processor", "inputs": [{"name": "bundle", "delivery": "exact", "select_bundle": true}], "outputs": [],
                     "queries": ["editor_facts", "admit_source"]}}));
        let calls = Rc::new(RefCell::new(Calls::default()));
        let mut workbench = Workbench::new(&root, Box::new(FakePlugins { calls: calls.clone(), catalog })).unwrap();
        let mut host = FakeHost::default();
        apply(&mut workbench, &mut host, "add_node", json!({"type_id": "camera_measurement"})).unwrap();
        apply(&mut workbench, &mut host, "add_node", json!({"type_id": "derive", "source_signal": "@logic/camera_measurement/frames"})).unwrap();
        workbench.on_runtime(&mut host);
        event(&mut workbench,Event::Opened { instance: node_instance("derive"), working_point: Value::Null }, &mut host);
        let asked = |name: &str| calls.borrow().2.iter().rev().find(|(query, ..)| query == name).cloned();
        assert_eq!(asked("editor_facts").unwrap().2["bundle"], json!([{"name": "frames", "schema": null, "event_schema": null}]));
        assert!(asked("admit_source").is_none(), "nothing to admit before the members publish");
        let issues = |workbench: &Workbench, host: &FakeHost| workbench.state(host)["logic"][1]["issues"].clone();
        let (_, ticket, _) = asked("editor_facts").unwrap();
        event(&mut workbench,Event::Answered { ticket, result: Ok(json!({"editor": "derive_variables", "manages": ["variables"]})) }, &mut host);
        assert_eq!(issues(&workbench, &host), json!([]), "the unasked check holds nothing");
        host.signals.push(SignalInfo { name: "@logic/camera_measurement/frames".into(), generation: "g1".into(), owner: "camera_measurement".into(), published: true, sealed: false, shape: String::new() });
        host.schemas.insert("@logic/camera_measurement/frames".into(), (json!({"tree": "run"}), json!({"tree": "event"})));
        workbench.on_runtime(&mut host);
        let expected = json!([{"name": "frames", "schema": {"tree": "run"}, "event_schema": {"tree": "event"}}]);
        assert_eq!(asked("editor_facts").unwrap().2["bundle"], expected);
        let (_, ticket, args) = asked("admit_source").unwrap();
        assert_eq!(args["bundle"], expected);
        event(&mut workbench,Event::Answered { ticket, result: Ok(json!(["bad · term 2: and needs booleans"])) }, &mut host);
        assert!(issues(&workbench, &host).as_array().unwrap().iter().any(|issue| issue == "bad · term 2: and needs booleans"));
    }

    /// A whole-bundle input is offered one entry per producer, as v2's `_source_choices`:
    /// a node's outputs as one, a panel's ROI and its fit (two owners) as two.
    #[test]
    fn a_bundle_reader_is_offered_one_entry_per_producer() {
        let root = scratch("bundle_choices");
        let mut catalog = catalog();
        let types = catalog["types"].as_array_mut().unwrap();
        types.push(json!({"type_id": "derive", "kind": "node", "label": "Derive", "parameters": [],
            "node": {"node_kind": "processor", "inputs": [{"name": "bundle", "delivery": "exact", "select_bundle": true}], "outputs": []}}));
        types.push(json!({"type_id": "survey", "kind": "node", "label": "Survey", "parameters": [],
            "node": {"node_kind": "processor", "inputs": [{"name": "frames", "delivery": "exact"}], "outputs": [{"name": "counts"}, {"name": "occupied"}]}}));
        let calls = Rc::new(RefCell::new(Calls::default()));
        let mut workbench = Workbench::new(&root, Box::new(FakePlugins { calls, catalog })).unwrap();
        let mut host = FakeHost::default();
        apply(&mut workbench, &mut host, "add_node", json!({"type_id": "camera_measurement"})).unwrap();
        apply(&mut workbench, &mut host, "add_node", json!({"type_id": "survey", "source_signal": "@logic/camera_measurement/frames"})).unwrap();
        apply(&mut workbench, &mut host, "add_node", json!({"type_id": "derive", "source_signal": "@logic/survey/occupied"})).unwrap();
        let published = |name: &str, generation: &str, owner: &str| SignalInfo { name: name.into(), generation: generation.into(), owner: owner.into(),
            published: true, sealed: false, shape: String::new() };
        host.signals = vec![published("@logic/panel-1/roi_frame", "g2", "panel-1/roi"), published("@logic/panel-1/roi_sum", "g2", "panel-1/roi"),
                            published("@logic/panel-1/center", "g3", "panel-1/fit")];
        workbench.on_runtime(&mut host);
        let state = workbench.state(&host);
        let derive = state["logic"].as_array().unwrap().iter().find(|row| row["type_id"] == "derive").unwrap();
        assert_eq!(derive["source_choices"], json!([
            ["@logic/camera_measurement/frames", "camera_measurement/frames"],
            ["@logic/survey/occupied", "survey/counts, occupied"],
            ["@logic/panel-1/roi_frame", "panel-1/roi_frame, roi_sum"],
            ["@logic/panel-1/center", "panel-1/center"]]));
        assert_eq!(derive["source_tree"], Value::Null, "a bundle is offered flat");
    }

    #[test]
    fn a_v2_layout_loads_respelled_and_saves_back() {
        let root = scratch("layout");
        let (mut workbench, _calls) = workbench(&root);
        let mut host = FakeHost { next: 10, ..Default::default() };
        apply(&mut workbench, &mut host, "add_device", json!({"type_id": "camera.virtual_mot", "instance_id": "mot_camera"})).unwrap();
        apply(&mut workbench, &mut host, "apply_devices", json!({})).unwrap();
        event(&mut workbench,Event::Opened { instance: "mot_camera".into(), working_point: json!({}) }, &mut host);
        let layout = json!({"format": "zlc.console-board",
            "panels": [
                {"panel_id": "panel-1", "signal": "@logic/camera_measurement/frames", "title": "@logic/camera_measurement/frames", "kind": "facet_grid",
                 "cell_kind": "", "size": "2x2", "interval_ms": 100, "semantic": {}, "display": {"colormap": "viridis"}, "interaction": {"series_lock": null},
                 "fit": {}, "overlay_signal": "@logic/occupancy/occupied", "published_outputs": {},
                 "selector": {"plot_kind": "image", "selector_kind": "area", "drawn": null, "ranges": [
                     {"axis": "camera_measurement.frames.1.spatial-x", "lower": 919.5, "upper": 1007.8, "coordinate_frame": "sensor_pixel_xy", "domain": "cell_data"},
                     {"axis": "camera_measurement.frames.0.spatial-y", "lower": 565.0, "upper": 634.4, "coordinate_frame": "sensor_pixel_xy", "domain": "cell_data"}],
                     "facets": [{"axis": "camera_measurement.frames.frame", "value": 0.0, "domain": "point"}]},
                 "crosshair": {}, "classifier_thresholds": [], "focused_cell": 0},
                {"panel_id": "panel-3", "signal": "@logic/panel-1/roi_frame", "title": "ROI", "kind": "rolling", "size": "2x2", "interval_ms": 200,
                 "semantic": {"reduction": "mean"}, "display": {"window": 40}, "fit": {"model": "gaussian_offset"}, "published_outputs": {"center": true}},
                {"panel_id": "panel-4", "signal": "@logic/panel-9/center", "title": "", "kind": "curve"}],
            "logic": [
                {"node_id": "camera_measurement", "api_name": "camera_measurement", "values": {"repeat": 0, "frames_per_cycle": 2, "photoelectrons": true},
                 "source_signal": "", "device_keys": {"camera": "mot_camera"}, "artifact_inputs": {}, "auto_preview": true},
                {"node_id": "occupancy", "name": "Occupancy", "api_name": "occupancy", "values": {"model_kind": 3},
                 "source_signal": "@logic/camera_measurement/frames", "device_keys": {}, "artifact_inputs": {"calibrations": [{"path": "cal.npz", "frames": ["rest"]}]}}]});
        let path = root.join("layout.json");
        workspace::write_readable(&path, &layout).unwrap();
        let loaded = apply(&mut workbench, &mut host, "load_layout", json!({"path": path})).unwrap();
        assert_eq!(loaded["replaced"], true);
        let notes = workbench.notes.join("\n");
        assert!(notes.contains("Model must be text; the row is kept with that field at its default"), "{notes}");
        assert!(notes.contains("read @logic/panel-9/center, a panel this board does not carry; the connection was dropped"), "{notes}");
        let panels: Vec<(String, String)> = host.panels.iter().map(|panel| (panel.panel.clone(), panel.signal.clone())).collect();
        assert_eq!(panels, [("panel-11".into(), "@logic/camera_measurement/frames".into()), ("panel-12".into(), "@logic/panel-11/roi_frame".into()),
                            ("panel-13".into(), String::new())]);
        assert_eq!(host.panels[0].target["selectors"], json!([{"kind": "area", "value": {"x": [919.5, 1007.8], "y": [565.0, 634.4]}, "facet_index": 0, "revision": 1}]));
        assert_eq!((host.panels[1].window, &host.panels[1].target["parameters"]), (40, &json!({})));
        assert_eq!(workbench.node("camera_measurement").unwrap().draft.values, *json!({"repeat": 0, "frames_per_cycle": 2}).as_object().unwrap());
        assert_eq!(workbench.node("occupancy").unwrap().draft.values["model_kind"], "default");
        assert_eq!(workbench.board.get("panel-12").unwrap().title.as_deref(), Some("ROI"));
        // The Overlay offers the judgements of the image's measurement, and only those.
        let board = &workbench.state(&host)["board"];
        assert_eq!((&board["panels"][0]["overlay_signal"], &board["panels"][0]["overlay_choices"]),
                   (&json!("@logic/occupancy/occupied"), &json!([["@logic/occupancy/occupied", "Occupancy/occupied"]])));
        assert!(apply(&mut workbench, &mut host, "set_panel_overlay", json!({"panel": "panel-11", "signal": "@logic/camera_measurement/frames"}))
            .unwrap_err().contains("judges no sites"));
        apply(&mut workbench, &mut host, "set_panel_overlay", json!({"panel": "panel-11", "signal": null})).unwrap();
        assert!(workbench.panel_overlays().is_empty());
        apply(&mut workbench, &mut host, "set_panel_overlay", json!({"panel": "panel-11", "signal": "@logic/occupancy/occupied"})).unwrap();
        assert_eq!(workbench.panel_overlays()["panel-11"], "@logic/occupancy/occupied");
        let saved = root.join("saved.json");
        apply(&mut workbench, &mut host, "save_layout", json!({"path": saved})).unwrap();
        let document = workspace::read_strict(&saved, "saved").unwrap();
        assert_eq!(document["panels"][0]["selector"], layout["panels"][0]["selector"], "an unchanged v2 selector is written back as read");
        assert_eq!((document["panels"][1]["panel_id"].as_str(), document["panels"][1]["display"]["window"].as_u64()), (Some("panel-12"), Some(40)));
        assert_eq!(document["panels"][1]["interval_ms"], 200);
        assert_eq!((document["panels"][0]["overlay_signal"].as_str(), document["panels"][1]["overlay_signal"].as_str()), (Some("@logic/occupancy/occupied"), Some("")));
        assert_eq!(document["logic"][1]["artifact_inputs"], layout["logic"][1]["artifact_inputs"]);
    }

    #[test]
    fn a_hardware_scan_asks_every_family_and_says_what_it_could_not_ask() {
        let root = scratch("scan");
        let calls = Rc::new(RefCell::new(Calls::default()));
        let mut catalog = catalog();
        for index in [0, 1] { catalog["types"][index]["queries"] = json!(["discover"]); }
        let mut workbench = Workbench::new(&root, Box::new(FakePlugins { calls: calls.clone(), catalog })).unwrap();
        let mut host = FakeHost::default();
        workbench.apply("discover_devices", &json!({}), &Origin::Operator, 7, &mut host);
        assert!(workbench.take_replies().is_empty(), "the reply waits for every family");
        let asked: Vec<(u64, String)> = calls.borrow().0.iter().filter_map(|call| {
            let mut parts = call.split(' ');
            (parts.next()? == "discover").then_some(())?;
            Some((parts.next()?.parse().ok()?, parts.next()?.to_owned()))
        }).collect();
        assert_eq!(asked.iter().map(|(_, type_id)| type_id.as_str()).collect::<Vec<_>>(), ["camera.virtual_mot", "sequencer.virtual"]);
        assert_eq!(workbench.apparatus_state()["busy"], "scanning hardware");
        assert!(apply(&mut workbench, &mut host, "apply_devices", json!({})).unwrap_err().starts_with("hardware scan still running"));
        event(&mut workbench,Event::Answered { ticket: asked[0].0, result: Ok(json!({"found": [{"instance_id": "cam_1", "role": "cam_1",
            "parameters": {"exposure_seconds": 0.05}}], "notes": ["VISA lists nothing"]})) }, &mut host);
        event(&mut workbench,Event::Answered { ticket: asked[1].0, result: Err("VISA lists nothing".into()) }, &mut host);
        let (request, reply) = workbench.take_replies().pop().unwrap();
        assert_eq!(request, 7);
        assert_eq!(reply.unwrap()["message"], "discovered 1 device; camera.virtual_mot, sequencer.virtual: VISA lists nothing; camera: No module named pypylon");
        let found = &workbench.apparatus_state()["discovered"][0];
        assert_eq!((found["type_id"].as_str(), found["parameters"]["exposure_seconds"].as_f64()), (Some("camera.virtual_mot"), Some(0.05)));
        apply(&mut workbench, &mut host, "add_discovered", json!({"instance_id": "cam_1"})).unwrap();
        assert_eq!(workbench.device_rows()[0]["pending"], "add");
        // Loaded, then left out by another apparatus: its Remove is no refusal, and the scan's Add brings it back as itself.
        apply(&mut workbench, &mut host, "apply_devices", json!({})).unwrap();
        event(&mut workbench,Event::Opened { instance: "cam_1".into(), working_point: json!({}) }, &mut host);
        std::fs::write(root.join("other.json"), r#"{"format": "zlc.installation", "devices": []}"#).unwrap();
        apply(&mut workbench, &mut host, "load_apparatus", json!({"path": "other.json"})).unwrap();
        assert_eq!(workbench.device_rows()[0]["pending"], "remove");
        apply(&mut workbench, &mut host, "remove_device", json!({"device": "cam_1"})).unwrap();
        apply(&mut workbench, &mut host, "add_discovered", json!({"instance_id": "cam_1"})).unwrap();
        // (other.json's simulation is the defaults: the simulated camera is reopened, not opened twice.)
        assert_eq!(apply(&mut workbench, &mut host, "apply_devices", json!({})).unwrap()["opening"], json!([]));
        event(&mut workbench,Event::Closed { instance: "cam_1".into(), error: None }, &mut host);
        event(&mut workbench,Event::Opened { instance: "cam_1".into(), working_point: json!({}) }, &mut host);
        // Lost while open (the computer that publishes it went away): said at once.
        event(&mut workbench,Event::Failed { instance: "cam_1".into(), error: "pc2 withdrew it".into() }, &mut host);
        assert_eq!(workbench.apparatus_state()["report"]["text"], "cam_1 lost: pc2 withdrew it");
    }

    #[test]
    fn a_published_device_is_handed_over_and_a_remote_row_is_its_publishers_type() {
        let root = scratch("bench");
        let calls = Rc::new(RefCell::new(Calls::default()));
        let mut catalog = catalog();
        catalog["types"].as_array_mut().unwrap().push(json!({"type_id": "remote.device", "kind": "device", "family": "remote", "addable": false,
            "parameters": [{"name": "host", "type": "str", "label": "Fabric host", "default": ""}, {"name": "instance_id", "type": "str", "label": "Published", "default": ""}]}));
        let mut workbench = Workbench::new(&root, Box::new(FakePlugins { calls: calls.clone(), catalog })).unwrap();
        let mut host = FakeHost::default();
        apply(&mut workbench, &mut host, "add_device", json!({"type_id": "camera.virtual_mot", "instance_id": "mot_camera"})).unwrap();
        apply(&mut workbench, &mut host, "add_device", json!({"type_id": "remote.device", "instance_id": "remote_cam", "parameters": {"host": "pc2", "instance_id": "cam"}})).unwrap();
        apply(&mut workbench, &mut host, "apply_devices", json!({})).unwrap();
        event(&mut workbench,Event::Opened { instance: "mot_camera".into(), working_point: json!({}) }, &mut host);
        // Open, a remote row is the type its publisher has: a camera the nodes here can choose.
        event(&mut workbench,Event::Opened { instance: "remote_cam".into(), working_point: json!({"remote": {"publisher": "pc2"},
            "descriptor": {"type_id": "camera.dcam", "kind": "device", "family": "camera", "label": "Hamamatsu camera", "capabilities": ["camera.adapter"]}}) }, &mut host);
        let rows = workbench.device_rows();
        assert_eq!((rows[1]["family"].as_str(), rows[1]["label"].as_str(), rows[1]["publishable"].as_bool()), (Some("camera"), Some("Hamamatsu camera"), Some(false)));
        assert_eq!(rows[0]["publishable"], true);
        assert_eq!(workbench.device_options("camera.adapter").len(), 2);
        assert!(apply(&mut workbench, &mut host, "publish_device", json!({"device": "remote_cam", "published": true})).unwrap_err().contains("already comes from the bench fabric"));
        // Remote on: answered by the fabric, then nothing here uses it.
        workbench.apply("publish_device", &json!({"device": "mot_camera", "published": true}), &Origin::Operator, 9, &mut host);
        let ticket: u64 = calls.borrow().0.iter().rev().find_map(|call| call.strip_prefix("publish ")?.split(' ').next()?.parse().ok()).unwrap();
        event(&mut workbench,Event::Answered { ticket, result: Ok(json!({"port": 18859})) }, &mut host);
        assert_eq!(workbench.take_replies().pop().unwrap(), (9, Ok(json!({"device": "mot_camera", "published": true, "fabric": {"port": 18859}}))));
        assert_eq!(workbench.device_rows()[0]["published"], true);
        for (command, params) in [("tune_device", json!({"device": "mot_camera", "field": "exposure_seconds", "value": 0.2})),
                                  ("close_device", json!({"device": "mot_camera"})), ("read_device", json!({"device": "mot_camera"}))] {
            assert!(apply(&mut workbench, &mut host, command, params).unwrap_err().contains("published on the bench fabric"), "{command}");
        }
        apply(&mut workbench, &mut host, "add_node", json!({"type_id": "camera_measurement", "device_keys": {"camera": "mot_camera"}})).unwrap();
        let issues = workbench.node_row(workbench.node("camera_measurement").unwrap(), &[])["issues"].clone();
        assert!(issues.as_array().unwrap().iter().any(|issue| issue.as_str().unwrap().contains("published on the bench fabric")), "{issues}");
        // Remote off: back here.
        workbench.apply("publish_device", &json!({"device": "mot_camera", "published": false}), &Origin::Operator, 10, &mut host);
        let ticket: u64 = calls.borrow().0.iter().rev().find_map(|call| call.strip_prefix("publish ")?.split(' ').next()?.parse().ok()).unwrap();
        event(&mut workbench,Event::Answered { ticket, result: Ok(json!({})) }, &mut host);
        assert_eq!(workbench.device_rows()[0]["published"], false);
        apply(&mut workbench, &mut host, "close_device", json!({"device": "mot_camera"})).unwrap();
    }

    #[test]
    fn the_bundled_recipe_is_offered_and_asks_for_what_it_cannot_choose() {
        let root = scratch("recipes");
        let (mut workbench, _calls) = workbench(&root);
        let mut host = FakeHost::default();
        let recipe = apply(&mut workbench, &mut host, "list_recipes", json!({"name": "trap_lifetime"}));
        assert!(recipe.is_ok(), "{recipe:?}");
        let listed = apply(&mut workbench, &mut host, "list_recipes", json!({})).unwrap();
        assert_eq!(listed[0]["name"], "trap_lifetime", "{listed}");
        let recipe = apply(&mut workbench, &mut host, "list_recipes", json!({"name": "trap_lifetime"})).unwrap();
        assert_eq!(recipe["defaults"]["repeats"], 20);
        assert_eq!(recipe["defaults"]["panel_semantic"]["fate:cell_data:calibration.site"], "facet");
        let refused = apply(&mut workbench, &mut host, "run_recipe", json!({"name": "trap_lifetime", "parameters": {"repeats": 2}})).unwrap_err();
        assert_eq!(refused, "Recipe requires explicit parameters: camera, sequencer, pulse_path, calibrations, scan_plan, panel_semantic");
        // A workspace recipe of the same name replaces the bundled one.
        std::fs::create_dir_all(root.join("recipes")).unwrap();
        let bundled = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("recipes");
        let text = std::fs::read_to_string(bundled.join("trap_lifetime.recipe.json")).unwrap().replace("Per-site trap lifetime", "Lifetime here");
        std::fs::write(root.join("recipes/trap_lifetime.recipe.json"), text).unwrap();
        std::fs::copy(bundled.join("trap_lifetime.layout.json"), root.join("recipes/trap_lifetime.layout.json")).unwrap();
        assert_eq!(apply(&mut workbench, &mut host, "list_recipes", json!({})).unwrap()[0]["title"], "Lifetime here");
    }

    #[test]
    fn a_v2_apparatus_loads_entry_by_entry_and_keeps_its_ids() {
        let root = scratch("ids");
        std::fs::write(root.join("apparatus.json"), r#"{"format": "zlc.installation", "simulation": {"seed": 3, "grid_shape_yx": "wide"},
            "devices": [
                {"instance_id": "n100_COM3", "role": "n100_COM3", "type_id": "camera.virtual_mot", "parameters": {"exposure_seconds": -1}},
                {"instance_id": "pump laser #2", "role": "pump laser", "type_id": "camera.virtual_mot", "parameters": {}},
                {"instance_id": "broken", "type_id": "sequencer.virtual", "parameters": {}},
                {"instance_id": "n100_COM3", "role": "again", "type_id": "sequencer.virtual", "parameters": {}},
                {"instance_id": "seq/1", "role": "seq", "type_id": "sequencer.virtual", "parameters": {}},
                {"instance_id": "sequencer", "role": "sequencer", "type_id": "sequencer.virtual", "parameters": {}, "comment": "x"}]}"#).unwrap();
        let (mut workbench, calls) = workbench(&root);
        let mut host = FakeHost::default();
        let ids: Vec<&str> = workbench.apparatus.draft.iter().map(|entry| entry.instance_id.as_str()).collect();
        assert_eq!(ids, ["n100_COM3", "pump laser #2", "sequencer"], "the good entries load, ids as written");
        let (line, severity) = workbench.report.clone();
        assert_eq!(severity, "warning");
        assert!(line.starts_with("3 devices from apparatus.json · 3 not loaded -- device 3: "), "{line}");
        for reason in ["this one has no role", "device 4: n100_COM3 is already used by device n100_COM3", "device 5: device instance_id cannot contain '/'",
                       "n100_COM3: Exposure must be at least 0", "sequencer: comment is not part of a device entry", "simulation: "] {
            assert!(line.contains(reason), "{reason:?} in {line}");
        }
        assert_eq!(workbench.apparatus.draft[0].parameters["exposure_seconds"], json!(0.1), "a refused value is at its default");
        assert_eq!(workbench.apparatus.simulation["seed"], 3);
        // Open, each command has an MCP tool name; the windows' `<id>__<command>` with the id as written reaches it too.
        apply(&mut workbench, &mut host, "apply_devices", json!({})).unwrap();
        for device in ["n100_COM3", "pump laser #2", "sequencer"] { event(&mut workbench,Event::Opened { instance: device.into(), working_point: json!({}) }, &mut host); }
        let pump = zlc_core::control::tool_word("pump laser #2");
        assert!(pump.starts_with("pump_laser_2-"), "{pump}");
        let tools: Vec<String> = workbench.plugin_tools().iter().map(|tool| tool["name"].as_str().unwrap().to_owned()).collect();
        assert_eq!(tools, ["n100_COM3__editor_send".to_owned(), "n100_COM3__editor_save".into(), format!("{pump}__editor_send"), format!("{pump}__editor_save")]);
        assert_eq!(workbench.device_rows()[1]["commands"], json!(tools[2..]));
        for name in [format!("{pump}__editor_send"), "pump laser #2__editor_send".into(), "n100_COM3__editor_send".into()] {
            workbench.apply(&name, &json!({}), &Origin::Operator, 9, &mut host);
            assert!(workbench.take_replies().is_empty(), "{name} goes to its device");
        }
        let sent: Vec<String> = calls.borrow().0.iter().filter(|call| call.ends_with(" command")).map(|call| call.rsplit_once(' ').unwrap().0.rsplit_once(' ').unwrap().0.to_owned()).collect();
        assert_eq!(sent, ["call pump laser #2", "call pump laser #2", "call n100_COM3"]);
        // An id v2 accepts is accepted when added; a file that is not an apparatus is said on the line.
        apply(&mut workbench, &mut host, "add_device", json!({"type_id": "sequencer.virtual", "instance_id": "n100_COM4"})).unwrap();
        assert!(apply(&mut workbench, &mut host, "add_device", json!({"type_id": "sequencer.virtual", "instance_id": " x"})).unwrap_err().contains("surrounding space"));
        std::fs::write(root.join("other.json"), r#"{"format": "zlc.console-board"}"#).unwrap();
        assert!(apply(&mut workbench, &mut host, "load_apparatus", json!({"path": "other.json"})).is_err());
        assert!(workbench.report.0.starts_with("cannot read other.json: "), "{:?}", workbench.report);
    }

    #[test]
    fn a_layout_loads_entry_by_entry() {
        let root = scratch("layout-entries");
        let (mut workbench, _calls) = workbench(&root);
        let mut host = FakeHost::default();
        // A device called as a node of the layout is: both keep their ids (nodes are named apart, as in v2).
        apply(&mut workbench, &mut host, "add_device", json!({"type_id": "camera.virtual_mot", "instance_id": "camera_measurement"})).unwrap();
        let layout = json!({"format": "zlc.console-board",
            "panels": [{"panel_id": "panel-1", "kind": "image", "signal": "@logic/camera_measurement/frames", "title": ""},
                       {"panel_id": "panel-2", "kind": "sparkline", "signal": "x"}, {"kind": "curve"}],
            "logic": [{"node_id": "camera_measurement", "api_name": "camera_measurement", "values": {"repeat": "many"}},
                      {"node_id": "mystery", "api_name": "not_a_node"},
                      {"node_id": "occupancy", "api_name": "occupancy", "source_signal": "@logic/camera_measurement/frames", "auto_preview": "yes"},
                      {"node_id": "occupancy2", "api_name": "post_selected_occupancy", "name": "Occupancy"}]});
        workspace::write_readable(&root.join("board.json"), &layout).unwrap();
        let read = apply(&mut workbench, &mut host, "read_layout", json!({"path": "board.json"})).unwrap();
        let nodes: Vec<(&str, &str, &str)> = read["nodes"].as_array().unwrap().iter()
            .map(|node| (node["node_id"].as_str().unwrap(), node["name"].as_str().unwrap(), node["source_signal"].as_str().unwrap())).collect();
        assert_eq!(nodes, [("camera_measurement", "camera_measurement", ""), ("occupancy", "occupancy", "@logic/camera_measurement/frames"),
                           ("occupancy2", "occupancy2", "")]);
        // A type another took over loads as that one, with the values that make it the same node.
        assert_eq!(read["nodes"].as_array().unwrap()[1..].iter().map(|node| (node["type_id"].as_str().unwrap(), node["values"]["post_selection"].as_bool().unwrap()))
            .collect::<Vec<_>>(), [("occupancy", false), ("occupancy", true)]);
        assert_eq!(read["panels"].as_array().unwrap().iter().map(|panel| panel["signal"].as_str().unwrap()).collect::<Vec<_>>(), ["@logic/camera_measurement/frames"]);
        let notes = read["notes"].as_array().unwrap().iter().map(|note| note.as_str().unwrap()).collect::<Vec<_>>().join("\n");
        for said in ["panel entry 2: \"sparkline\" is not a Task Console plot kind; it is left out", "panel entry 3: it requires panel_id; it is left out",
                     "logic entry 2: no logic node named \"not_a_node\"; it is left out",
                     "logic camera_measurement: Repeat", "logic occupancy: its auto_preview \"yes\" cannot be read",
                     "logic occupancy2 is called by its internal ID: Another node is already called “occupancy”."] {
            assert!(notes.contains(said), "{said:?} in\n{notes}");
        }
        // A device and a node may share an id (their plugin instances are named apart); only one
        // of each kind holds it. An apparatus read with the board's ids drops nothing.
        apply(&mut workbench, &mut host, "load_layout", json!({"path": "board.json"})).unwrap();
        apply(&mut workbench, &mut host, "add_device", json!({"type_id": "sequencer.virtual", "instance_id": "occupancy"})).unwrap();
        assert_eq!(apply(&mut workbench, &mut host, "add_node", json!({"type_id": "occupancy", "node_id": "camera_measurement"})).unwrap_err(),
                   "camera_measurement is already used by node camera_measurement");
        std::fs::write(root.join("bench.json"), r#"{"format": "zlc.installation", "simulation": {}, "devices": [
            {"instance_id": "occupancy2", "role": "occupancy2", "type_id": "sequencer.virtual", "parameters": {}},
            {"instance_id": "sequencer", "role": "sequencer", "type_id": "sequencer.virtual", "parameters": {}}]}"#).unwrap();
        let loaded = apply(&mut workbench, &mut host, "load_apparatus", json!({"path": "bench.json"})).unwrap();
        assert_eq!(loaded["skipped"], json!([]));
        assert_eq!(workbench.apparatus.draft.iter().map(|entry| entry.instance_id.as_str()).collect::<Vec<_>>(), ["occupancy2", "sequencer"]);
    }

    /// Where a dialog opens, spelled as the shell is handed it (a `/` path is refused there).
    fn opened((folder, name): (std::path::PathBuf, String)) -> (String, String) { (folder.display().to_string(), name) }

    /// A start opens in its folder, else in its kind's folder (only a save keeps its name then),
    /// in native spelling.
    #[test]
    fn a_dialog_opens_in_its_folder_else_in_its_kinds_folder() {
        let root = workspace::Workspace::new(&scratch("dialog-start")).unwrap().root;
        let data = root.join(workspace::DATA);
        let at = |folder: &std::path::Path, name: &str| (folder.display().to_string(), name.to_owned());
        std::fs::create_dir_all(root.join("figs")).unwrap();
        std::fs::write(root.join("figs").join("a.npz"), b"").unwrap();
        let slashed = format!("{}/figs/a.npz", root.display()).replace('\\', "/");
        assert_eq!(opened(workspace::dialog_start(std::path::Path::new(&slashed), &data, false)), at(&root.join("figs"), "a.npz"));
        assert_eq!(opened(workspace::dialog_start(&root.join("figs"), &data, false)), at(&root.join("figs"), ""));
        // A file whose folder is gone (a layout made on another computer), or a day not made yet.
        assert_eq!(opened(workspace::dialog_start(&root.join("gone").join("2026_01_01").join("context.npz"), &data, false)), at(&data, ""));
        assert_eq!(opened(workspace::dialog_start(&data.join("1999_01_01").join("screenshot.png"), &data, true)), at(&data, "screenshot.png"));
        // No start; a kind's folder not made yet gives way to the nearest above it.
        assert_eq!(opened(workspace::dialog_start(std::path::Path::new(""), &data.join("1999_01_01"), false)), at(&data, ""));
    }

    /// Each kind's dialog opens at its own open document, else in its own folder; opening or
    /// saving one kind never moves another's.
    #[test]
    fn each_kind_of_file_dialog_opens_at_its_own_document() {
        let root = scratch("dialogs");
        let (mut workbench, _calls) = workbench(&root);
        let mut host = FakeHost::default();
        let root = workbench.workspace().root.clone();
        let layouts = workbench.workspace().layouts();
        let opens = |workbench: &Workbench, action: &str| {
            let (_, _, save, start, home) = workbench.file_dialog(action).unwrap();
            opened(workspace::dialog_start(&start, &home, save))
        };
        let both = |workbench: &Workbench, kind: &str| (opens(workbench, &format!("load_{kind}")), opens(workbench, &format!("save_{kind}")));
        let at = |folder: &std::path::Path, name: &str| (folder.display().to_string(), name.to_owned());
        let fresh = [(at(&root, ""), at(&root, workspace::APPARATUS)), (at(&layouts, ""), at(&layouts, "layout.json"))];
        assert_eq!([both(&workbench, "apparatus"), both(&workbench, "layout")], fresh);
        // A layout saved elsewhere: its dialogs follow it, the apparatus's stay.
        let boards = root.join("boards");
        std::fs::create_dir_all(&boards).unwrap();
        apply(&mut workbench, &mut host, "save_layout", json!({"path": "boards/mine.json"})).unwrap();
        assert_eq!(both(&workbench, "layout"), (at(&boards, ""), at(&boards, "mine.json")));
        assert_eq!(both(&workbench, "apparatus"), fresh[0]);
        // An apparatus opened from another folder, named with `/` as an MCP client names it: its
        // dialogs follow it, the layout's stay.
        let benches = root.join("benches");
        std::fs::create_dir_all(&benches).unwrap();
        std::fs::write(benches.join("bench.json"), r#"{"format": "zlc.installation", "simulation": {}, "devices": []}"#).unwrap();
        let slashed = format!("{}/bench.json", benches.display()).replace('\\', "/");
        apply(&mut workbench, &mut host, "load_apparatus", json!({"path": slashed})).unwrap();
        assert_eq!(both(&workbench, "apparatus"), (at(&benches, ""), at(&benches, "bench.json")));
        assert_eq!(opens(&workbench, "load_layout"), at(&boards, ""));
        // A cleared board has no layout file: back to the layouts folder.
        apply(&mut workbench, &mut host, "add_node", json!({"type_id": "occupancy"})).unwrap();
        apply(&mut workbench, &mut host, "clear_board", json!({})).unwrap();
        assert_eq!(both(&workbench, "layout"), fresh[1]);
        assert!(workbench.file_dialog("save_image").is_none());
    }

    /// v2 DeviceUseBusy: while a device is driven from outside (the Pulse Editor's own pulse on the
    /// sequencer), a start that claims it is refused and stops nothing; once it is free, it starts.
    #[test]
    fn a_start_is_refused_while_an_outside_driver_holds_its_device() {
        let root = scratch("outside_use");
        let (mut workbench, calls) = workbench(&root);
        let mut host = FakeHost::default();
        apply(&mut workbench, &mut host, "add_device", json!({"type_id": "camera.virtual_mot", "instance_id": "mot_camera"})).unwrap();
        apply(&mut workbench, &mut host, "apply_devices", json!({})).unwrap();
        event(&mut workbench,Event::Opened { instance: "mot_camera".into(), working_point: json!({}) }, &mut host);
        apply(&mut workbench, &mut host, "add_node", json!({"type_id": "camera_measurement"})).unwrap();
        event(&mut workbench,Event::Opened { instance: node_instance("camera_measurement"), working_point: Value::Null }, &mut host);
        workbench.device_users().lock().unwrap().insert("mot_camera".into(), "PulseGUI".into());
        assert_eq!(apply(&mut workbench, &mut host, "start_node", json!({"node": "camera_measurement"})).unwrap_err(), "device is in use by PulseGUI");
        assert!(calls.borrow().1.is_empty(), "nothing was launched");
        workbench.device_users().lock().unwrap().clear();
        apply(&mut workbench, &mut host, "start_node", json!({"node": "camera_measurement"})).unwrap();
        assert_eq!(calls.borrow().1.len(), 1, "launched once the device is free");
        event(&mut workbench,Event::RunStarted { run: "camera_measurement#1".into() }, &mut host);
        apply(&mut workbench, &mut host, "edit_node", json!({"node": "camera_measurement", "device_keys": {"camera": "gone"}})).unwrap();
        assert!(apply(&mut workbench, &mut host, "start_node", json!({"node": "camera_measurement"})).is_err());
        let row = workbench.node_row(workbench.node("camera_measurement").unwrap(), &[]);
        assert_eq!(row["state"], "warning");
        assert!(row["status"].as_str().unwrap().contains("Restart refused:"));
        assert_eq!(row["run"]["state"], "running", "the refused restart leaves the old run running");
        assert_eq!(calls.borrow().1.len(), 1);
        apply(&mut workbench, &mut host, "revert_node", json!({"node": "camera_measurement"})).unwrap();
        let row = workbench.node_row(workbench.node("camera_measurement").unwrap(), &[]);
        assert_eq!(row["state"], "running");
        assert!(!row["status"].as_str().unwrap().contains("Restart refused:"));
    }

    /// v2 _replace_board: Clear or Open while a node runs stops it, says so, and replaces the
    /// board once it has stopped (whatever its outcome); a newer request takes the waiting one's place.
    #[test]
    fn a_board_replaced_during_a_run_waits_for_it_to_stop() {
        let root = scratch("replace");
        let (mut workbench, calls) = workbench(&root);
        let mut host = FakeHost::default();
        apply(&mut workbench, &mut host, "add_device", json!({"type_id": "camera.virtual_mot", "instance_id": "mot_camera"})).unwrap();
        apply(&mut workbench, &mut host, "apply_devices", json!({})).unwrap();
        event(&mut workbench,Event::Opened { instance: "mot_camera".into(), working_point: json!({}) }, &mut host);
        apply(&mut workbench, &mut host, "add_node", json!({"type_id": "camera_measurement"})).unwrap();
        event(&mut workbench,Event::Opened { instance: node_instance("camera_measurement"), working_point: Value::Null }, &mut host);
        apply(&mut workbench, &mut host, "start_node", json!({"node": "camera_measurement"})).unwrap();
        workspace::write_readable(&root.join("board.json"), &json!({"format": "zlc.console-board", "panels": [],
            "logic": [{"node_id": "occupancy", "api_name": "occupancy", "auto_preview": "yes"}]})).unwrap();
        let refused = apply(&mut workbench, &mut host, "load_layout", json!({"path": "board.json"})).unwrap_err();
        assert_eq!(refused, "Loading this layout needs stop_running=true while Logic is active");
        assert_eq!(workbench.console_report, (refused, "warning"));
        let cleared = apply(&mut workbench, &mut host, "clear_board", json!({"stop_running": true})).unwrap();
        assert_eq!((&cleared["replaced"], &cleared["waiting_for"]), (&json!(false), &json!(["camera_measurement"])));
        assert!(calls.borrow().0.contains(&"stop camera_measurement#1".to_owned()));
        assert_eq!(workbench.console_report.0, "stopping camera_measurement; the board is replaced when they have stopped");
        // What the newer layout could not keep is said at once; the replacement then says only that it is done.
        assert_eq!(apply(&mut workbench, &mut host, "load_layout", json!({"path": "board.json", "stop_running": true})).unwrap()["replaced"], false);
        assert!(workbench.console_report.0.contains("auto_preview \"yes\" cannot be read"), "{:?}", workbench.console_report);
        assert!(workbench.node("camera_measurement").is_some(), "the board waits for the run");
        event(&mut workbench,Event::RunFinished { run: "camera_measurement#1".into(), outcome: "failed".into(), error: None, generation: None }, &mut host);
        assert_eq!(workbench.console_report, ("layout loaded".into(), "task"));
        assert_eq!(workbench.nodes.iter().map(|node| node.id.as_str()).collect::<Vec<_>>(), ["occupancy"]);
    }

    // ------------------------------------------------- guards of the v2 audit (AUDIT_out/workbench.md)

    const RIG: &str = r#"{"format": "zlc.installation", "simulation": {},
      "devices": [{"instance_id": "mot_camera", "role": "mot_camera", "type_id": "camera.virtual_mot", "parameters": {"exposure_seconds": 0.02}},
                  {"instance_id": "sequencer", "role": "sequencer", "type_id": "sequencer.virtual", "parameters": {}}]}"#;

    /// The rig above, Initialized and open.
    fn rig(name: &str, catalog: Value) -> (std::path::PathBuf, Workbench, Rc<RefCell<Calls>>, FakeHost) {
        let root = scratch(name);
        std::fs::write(root.join("apparatus.json"), RIG).unwrap();
        let calls = Rc::new(RefCell::new(Calls::default()));
        let mut workbench = Workbench::new(&root, Box::new(FakePlugins { calls: calls.clone(), catalog })).unwrap();
        let mut host = FakeHost::default();
        apply(&mut workbench, &mut host, "apply_devices", json!({})).unwrap();
        for device in ["mot_camera", "sequencer"] {
            event(&mut workbench,Event::Opened { instance: device.into(), working_point: json!({"tunable": true}) }, &mut host);
        }
        (root, workbench, calls, host)
    }

    fn add_open_node(mut workbench: &mut Workbench, host: &mut FakeHost, params: Value) {
        let node = apply(workbench, host, "add_node", params).unwrap()["node"].as_str().unwrap().to_owned();
        event(&mut workbench,Event::Opened { instance: node_instance(&node), working_point: Value::Null }, host);
    }

    fn count(calls: &Rc<RefCell<Calls>>, line: &str) -> usize { calls.borrow().0.iter().filter(|call| call.as_str() == line).count() }

    fn asked(calls: &Rc<RefCell<Calls>>, name: &str) -> u64 { calls.borrow().2.iter().rev().find(|(query, ..)| query == name).unwrap().1 }

    /// v2 _active_differs: a failed device and changed simulation settings are an Apply, not a Shut down.
    #[test]
    fn a_failed_device_and_changed_simulation_are_applied() {
        let (_root, mut workbench, calls, mut host) = rig("audit-apply", catalog());
        event(&mut workbench,Event::Failed { instance: "mot_camera".into(), error: "the plugin process exited".into() }, &mut host);
        assert_eq!(workbench.apparatus_state()["action"], "apply");
        apply(&mut workbench, &mut host, "apply_devices", json!({})).unwrap();
        assert_eq!(count(&calls, "open mot_camera camera.virtual_mot"), 2, "Apply opens it again");
        event(&mut workbench,Event::Opened { instance: "mot_camera".into(), working_point: json!({}) }, &mut host);
        assert_eq!(workbench.apparatus_state()["action"], "shut_down");
        apply(&mut workbench, &mut host, "edit_simulation", json!({"values": {"seed": 5}})).unwrap();
        assert_eq!(workbench.apparatus_state()["action"], "apply");
    }

    /// v2 _path_unread: a file not read in full is never replaced by Save unless asked; the draft is not that file.
    #[test]
    fn an_apparatus_not_read_in_full_is_not_written_back_smaller() {
        let root = scratch("audit-unread");
        let path = root.join("apparatus.json");
        std::fs::write(&path, r#"{"format": "zlc.installation", "simulation": {},
          "devices": [{"instance_id": "a", "role": "cam", "type_id": "camera.virtual_mot", "parameters": {"exposure_seconds": 0.02, "gone": 1}},
                      {"instance_id": "b", "role": "cam", "type_id": "camera.virtual_mot", "parameters": {}}]}"#).unwrap();
        let (mut bench, _calls) = workbench(&root);
        let mut host = FakeHost::default();
        let state = bench.apparatus_state();
        assert_eq!((state["dirty"].as_bool(), state["unread"].as_str()), (Some(true), Some("2 of its devices or settings were left out")));
        let refused = apply(&mut bench, &mut host, "save_apparatus", json!({"path": path.display().to_string()})).unwrap_err();
        assert!(refused.starts_with("apparatus.json was not read in full"), "{refused}");
        assert!(std::fs::read_to_string(&path).unwrap().contains("\"b\""), "the file keeps what the draft left out");
        apply(&mut bench, &mut host, "save_apparatus", json!({"path": path.display().to_string(), "replace_unread": true})).unwrap();
        assert_eq!(bench.apparatus_state()["dirty"], false);
        // A file that cannot be read at all: the draft is empty and not that file.
        std::fs::write(&path, "{ \"format\": ").unwrap();
        let (mut unread, _calls) = workbench(&root);
        assert!(unread.apparatus_state()["unread"].as_str().unwrap().starts_with("it could not be read"));
        apply(&mut unread, &mut host, "add_device", json!({"type_id": "camera.virtual_mot"})).unwrap();
        assert!(apply(&mut unread, &mut host, "save_apparatus", json!({"path": path.display().to_string()})).is_err());
        apply(&mut unread, &mut host, "save_apparatus", json!({"path": root.join("other.json").display().to_string()})).unwrap();
    }

    /// v2 DeviceUseCoordinator: one record of who uses a device; Apply, Close, Remote and a driving
    /// command are refused while the Pulse Editor drives it, in v2's words.
    #[test]
    fn a_device_an_outside_driver_holds_is_not_changed_published_or_driven() {
        let (_root, mut workbench, calls, mut host) = rig("audit-outside", catalog());
        workbench.device_users().lock().unwrap().insert("sequencer".into(), "PulseGUI".into());
        workbench.device_users().lock().unwrap().insert("mot_camera".into(), "PulseGUI".into());
        assert_eq!(apply(&mut workbench, &mut host, "publish_device", json!({"device": "sequencer", "published": true})).unwrap_err(),
                   "sequencer: not published -- device is in use by PulseGUI");
        apply(&mut workbench, &mut host, "edit_device", json!({"device": "mot_camera", "parameters": {"exposure_seconds": 0.07}})).unwrap();
        assert_eq!(apply(&mut workbench, &mut host, "apply_devices", json!({})).unwrap_err(), "device changes did not apply: device is in use by PulseGUI");
        assert_eq!(apply(&mut workbench, &mut host, "close_device", json!({"device": "sequencer"})).unwrap_err(),
                   "device changes did not apply: device is in use by PulseGUI");
        assert_eq!(apply(&mut workbench, &mut host, "mot_camera__editor_send", json!({})).unwrap_err(), "device is in use by PulseGUI");
        assert!(!calls.borrow().0.iter().any(|call| call.starts_with("close") || call.starts_with("publish")), "{:?}", calls.borrow().0);
    }

    /// v2 broker: one physical instrument binds one row; the later row naming it is refused before it opens.
    #[test]
    fn one_instrument_binds_one_row() {
        let mut catalog = catalog();
        catalog["types"].as_array_mut().unwrap().push(json!({"type_id": "rf.visa", "kind": "device", "family": "rf", "label": "VISA RF",
            "capabilities": ["rf.source"], "identity": "visa:{resource}", "parameters": [{"name": "resource", "type": "str", "label": "Resource", "default": ""}]}));
        let root = scratch("audit-identity");
        let calls = Rc::new(RefCell::new(Calls::default()));
        let mut workbench = Workbench::new(&root, Box::new(FakePlugins { calls: calls.clone(), catalog })).unwrap();
        let mut host = FakeHost::default();
        for (id, resource) in [("rf", "TCPIP::10.0.0.5::INSTR"), ("rf2", "tcpip::10.0.0.5::instr"), ("rf3", "TCPIP::10.0.0.6::INSTR")] {
            apply(&mut workbench, &mut host, "add_device", json!({"type_id": "rf.visa", "instance_id": id, "parameters": {"resource": resource}})).unwrap();
        }
        apply(&mut workbench, &mut host, "apply_devices", json!({})).unwrap();
        let rows = workbench.device_rows();
        assert_eq!(rows.iter().map(|row| row["state"].as_str().unwrap()).collect::<Vec<_>>(), ["opening", "failed", "opening"]);
        assert_eq!(rows[1]["error"], "RuntimeError: physical device 'visa:tcpip::10.0.0.5::instr' is already bound (to rf)");
        assert_eq!(count(&calls, "open rf2 rf.visa"), 0, "refused before it is opened");
    }

    /// An IMU on the serial port its row names (it scans for one), and a board whose UART Init
    /// finds when its row leaves it blank: both bind their port.
    fn ports_catalog() -> Value {
        let mut catalog = catalog();
        let types = catalog["types"].as_array_mut().unwrap();
        types.push(json!({"type_id": "waveform.n100", "kind": "device", "family": "waveform", "label": "IMU", "resources": ["serial:{port}"], "queries": ["discover"],
            "parameters": [{"name": "port", "type": "str", "label": "Serial port", "default": "", "blank_label": "found on Init"}]}));
        types.push(json!({"type_id": "sequencer.local", "kind": "device", "family": "sequencer", "label": "Board", "resources": ["serial:{uart_port}"], "queries": ["resolve"],
            "parameters": [{"name": "backend", "type": "str", "label": "Board transport", "default": "auto"},
                           {"name": "uart_port", "type": "str", "label": "UART port", "default": "", "blank_label": "found on Init"}]}));
        catalog
    }

    /// A draft of these rows (id, type, settings), not applied.
    fn ports_rig(name: &str, rows: &[(&str, &str, Value)]) -> (Workbench, Rc<RefCell<Calls>>, FakeHost) {
        let calls = Rc::new(RefCell::new(Calls::default()));
        let mut workbench = Workbench::new(&scratch(name), Box::new(FakePlugins { calls: calls.clone(), catalog: ports_catalog() })).unwrap();
        let mut host = FakeHost::default();
        for (id, type_id, parameters) in rows {
            apply(&mut workbench, &mut host, "add_device", json!({"type_id": type_id, "instance_id": id, "parameters": parameters})).unwrap();
        }
        (workbench, calls, host)
    }

    /// Every resolve asked so far: (ticket, {instance, parameters, excluded}).
    fn resolves(calls: &Rc<RefCell<Calls>>) -> Vec<(u64, Value)> {
        calls.borrow().2.iter().filter(|(op, ..)| op == "resolve").map(|(_, ticket, params)| (*ticket, params.clone())).collect()
    }

    /// Init finds what a row leaves blank before the row opens, while the other rows open at once: a
    /// board's UART among the ports nothing binds, pinned into the apparatus, which then reads unsaved.
    #[test]
    fn init_finds_a_blank_port_on_the_free_ports_and_pins_it() {
        let (mut wb, calls, mut host) = ports_rig("resolve", &[("sequencer", "sequencer.local", json!({})), ("n100", "waveform.n100", json!({"port": "com3"}))]);
        apply(&mut wb, &mut host, "save_apparatus", json!({"path": "apparatus.json"})).unwrap();
        apply(&mut wb, &mut host, "apply_devices", json!({})).unwrap();
        assert_eq!(count(&calls, "open n100 waveform.n100"), 1, "the IMU opens at once");
        assert_eq!(count(&calls, "open sequencer sequencer.local"), 0, "the board waits to be found");
        let asked = resolves(&calls);
        assert_eq!((asked.len(), &asked[0].1["instance"], &asked[0].1["excluded"]), (1, &json!("sequencer"), &json!(["serial:COM3"])));
        let state = wb.apparatus_state();
        assert_eq!((state["busy"].as_str(), state["dirty"].as_bool()), (Some("initializing devices · finding sequencer's UART port"), Some(false)));
        assert_eq!((wb.device_rows()[0]["state"].as_str(), wb.device_rows()[0]["resolving"].as_bool()), (Some("opening"), Some(true)));
        event(&mut wb, Event::Answered { ticket: asked[0].0, result: Ok(json!({"parameters": {"uart_port": "COM5"},
            "note": "board UART found on COM5 (passed over COM3: bound)"})) }, &mut host);
        assert_eq!(count(&calls, "open sequencer sequencer.local"), 1, "opened with what was found");
        assert_eq!(wb.apparatus.draft[0].parameters["uart_port"], "COM5");
        assert_eq!(wb.apparatus_state()["dirty"], true, "pinned into the apparatus, not saved");
        assert_eq!(wb.device_rows()[0]["pending"], Value::Null, "what opened is what the apparatus says");
        for device in ["n100", "sequencer"] { event(&mut wb, Event::Opened { instance: device.into(), working_point: json!({}) }, &mut host); }
        assert_eq!(wb.apparatus_state()["report"]["text"], "2 devices initialized: sequencer, n100 · sequencer: board UART found on COM5 \
                   (passed over COM3: bound), written into the apparatus -- Save to keep it");
    }

    /// One port, one row: of rows binding one resource the later is refused before it opens, however it spells it.
    #[test]
    fn two_rows_binding_one_port_open_only_the_first() {
        let (mut wb, calls, mut host) = ports_rig("bound", &[("n100", "waveform.n100", json!({"port": "COM3"})),
            ("n100b", "waveform.n100", json!({"port": r"\\.\com3"})), ("sequencer", "sequencer.local", json!({"uart_port": " COM3"}))]);
        apply(&mut wb, &mut host, "apply_devices", json!({})).unwrap();
        let rows = wb.device_rows();
        assert_eq!(rows.iter().map(|row| row["state"].as_str().unwrap()).collect::<Vec<_>>(), ["opening", "failed", "failed"]);
        for row in &rows[1..] { assert_eq!(row["error"], "RuntimeError: serial:COM3 is already bound (to n100)"); }
        assert_eq!((count(&calls, "open n100b waveform.n100"), count(&calls, "open sequencer sequencer.local"), resolves(&calls).len()), (0, 0, 0));
    }

    /// A scan is handed what the apparatus binds, touches none of it, and says what it passed over.
    #[test]
    fn a_scan_passes_over_the_bound_ports() {
        let (mut wb, calls, mut host) = ports_rig("scan-bound", &[("n100", "waveform.n100", json!({"port": "COM3"}))]);
        wb.apply("discover_devices", &json!({}), &Origin::Operator, 7, &mut host);
        let (_, ticket, asked) = calls.borrow().2.iter().find(|(op, ..)| op == "discover").cloned().unwrap();
        assert_eq!(asked, json!({"type_id": "waveform.n100", "excluded": ["serial:COM3"]}));
        event(&mut wb, Event::Answered { ticket, result: Ok(json!({"found": [], "notes": []})) }, &mut host);
        assert_eq!(wb.take_replies().pop().unwrap().1.unwrap()["message"], "discovered 0 devices; camera: No module named pypylon · passed over COM3 (n100)");
    }

    /// A row whose blank port is not found fails with what its resolve tried; the others open as ever,
    /// and Apply looks for it again.
    #[test]
    fn a_resolve_that_finds_nothing_fails_only_its_row() {
        let (mut wb, calls, mut host) = ports_rig("resolve-error", &[("sequencer", "sequencer.local", json!({"backend": "uart"})), ("n100", "waveform.n100", json!({"port": "COM3"}))]);
        apply(&mut wb, &mut host, "apply_devices", json!({})).unwrap();
        let why = "no board UART on the free ports (tried COM4: timeout; passed over COM3: bound)";
        event(&mut wb, Event::Answered { ticket: resolves(&calls)[0].0, result: Err(why.into()) }, &mut host);
        event(&mut wb, Event::Opened { instance: "n100".into(), working_point: json!({}) }, &mut host);
        let rows = wb.device_rows();
        assert_eq!((rows[0]["state"].as_str(), rows[0]["error"].as_str(), rows[1]["state"].as_str()), (Some("failed"), Some(why), Some("open")));
        assert_eq!(count(&calls, "open sequencer sequencer.local"), 0);
        let state = wb.apparatus_state();
        assert_eq!(state["report"]["text"], format!("1 device initialized: n100 · 1 failed -- sequencer: {why}"));
        assert_eq!((state["action"].as_str(), wb.apparatus.draft[0].parameters["uart_port"].as_str()), (Some("apply"), Some("")));
        apply(&mut wb, &mut host, "apply_devices", json!({})).unwrap();
        assert_eq!(resolves(&calls).len(), 2, "Apply looks again");
    }

    /// Two rows to be found are found one after the other, the second passing over what the first found.
    #[test]
    fn rows_to_be_found_are_found_one_at_a_time() {
        let (mut wb, calls, mut host) = ports_rig("resolve-order", &[("board", "sequencer.local", json!({})), ("board2", "sequencer.local", json!({}))]);
        apply(&mut wb, &mut host, "apply_devices", json!({})).unwrap();
        let asked = resolves(&calls);
        assert_eq!((asked.len(), &asked[0].1["instance"], &asked[0].1["excluded"]), (1, &json!("board"), &json!([])));
        event(&mut wb, Event::Answered { ticket: asked[0].0, result: Ok(json!({"parameters": {"uart_port": "COM5"}, "note": ""})) }, &mut host);
        let asked = resolves(&calls);
        assert_eq!((asked.len(), &asked[1].1["instance"], &asked[1].1["excluded"]), (2, &json!("board2"), &json!(["serial:COM5"])));
        event(&mut wb, Event::Answered { ticket: asked[1].0, result: Ok(json!({"parameters": {"uart_port": "COM6"}, "note": ""})) }, &mut host);
        assert_eq!((count(&calls, "open board sequencer.local"), count(&calls, "open board2 sequencer.local")), (1, 1));
        let ports: Vec<&Value> = wb.apparatus.draft.iter().map(|entry| &entry.parameters["uart_port"]).collect();
        assert_eq!(ports, [&json!("COM5"), &json!("COM6")]);
    }

    /// A row stops waiting when it is shut down or taken out by Apply: its resolve's late answer opens
    /// and pins nothing, and the next row is asked only once that answer is in (two never probe at once).
    #[test]
    fn a_row_taken_out_while_it_is_found_opens_nothing() {
        let (mut wb, calls, mut host) = ports_rig("resolve-cancel", &[("board", "sequencer.local", json!({})), ("n100", "waveform.n100", json!({"port": "COM3"}))]);
        let found = || Ok(json!({"parameters": {"uart_port": "COM5"}, "note": "board UART found on COM5"}));
        apply(&mut wb, &mut host, "apply_devices", json!({})).unwrap();
        let first = resolves(&calls)[0].0;
        apply(&mut wb, &mut host, "shut_down_devices", json!({})).unwrap();
        assert_eq!(count(&calls, "close board"), 0, "nothing was opened to close");
        event(&mut wb, Event::Closed { instance: "n100".into(), error: None }, &mut host);
        assert_eq!(wb.apparatus_state()["report"]["text"], "devices shut down");
        event(&mut wb, Event::Answered { ticket: first, result: found() }, &mut host);
        assert_eq!((count(&calls, "open board sequencer.local"), wb.apparatus.draft[0].parameters["uart_port"].as_str()), (0, Some("")));
        // Apply takes it out while it is found; a row added meanwhile waits for that answer.
        apply(&mut wb, &mut host, "apply_devices", json!({})).unwrap();
        event(&mut wb, Event::Opened { instance: "n100".into(), working_point: json!({}) }, &mut host);
        let second = resolves(&calls)[1].0;
        apply(&mut wb, &mut host, "remove_device", json!({"device": "board"})).unwrap();
        apply(&mut wb, &mut host, "apply_devices", json!({})).unwrap();
        assert!(!wb.devices.contains_key("board") && count(&calls, "close board") == 0);
        apply(&mut wb, &mut host, "add_device", json!({"type_id": "sequencer.local", "instance_id": "board2"})).unwrap();
        apply(&mut wb, &mut host, "apply_devices", json!({})).unwrap();
        assert_eq!(resolves(&calls).len(), 2, "the next waits for the answer still out");
        event(&mut wb, Event::Answered { ticket: second, result: found() }, &mut host);
        let asked = resolves(&calls);
        assert_eq!((asked.len(), &asked[2].1["instance"]), (3, &json!("board2")));
        assert_eq!(count(&calls, "open board sequencer.local"), 0);
    }

    /// v2: a row can always be started again; a node whose plugin process was lost opens again on check or Start.
    #[test]
    fn a_node_whose_plugin_was_lost_opens_again() {
        let (_root, mut workbench, calls, mut host) = rig("audit-reopen", catalog());
        add_open_node(&mut workbench, &mut host, json!({"type_id": "camera_measurement"}));
        event(&mut workbench,Event::Failed { instance: node_instance("camera_measurement"), error: "the plugin process exited".into() }, &mut host);
        let started = apply(&mut workbench, &mut host, "start_node", json!({"node": "camera_measurement"})).unwrap();
        assert_eq!((started["phase"].as_str(), count(&calls, "open node/camera_measurement camera_measurement")), (Some("checking"), 2));
        event(&mut workbench,Event::Opened { instance: node_instance("camera_measurement"), working_point: Value::Null }, &mut host);
        assert_eq!(calls.borrow().1.last().unwrap().run, "camera_measurement#1", "the Start goes once it is open");
    }

    /// The draft queries: `admit` is asked, its answers and any failed query are the draft's issues.
    #[test]
    fn what_only_the_plugin_can_admit_is_a_draft_issue() {
        let mut catalog = catalog();
        for item in catalog["types"].as_array_mut().unwrap().iter_mut().filter(|item| item["type_id"] == "camera_measurement") {
            item["node"]["queries"] = json!(["declare_outputs", "admit"]);
        }
        let (_root, mut workbench, calls, mut host) = rig("audit-admit", catalog);
        add_open_node(&mut workbench, &mut host, json!({"type_id": "camera_measurement"}));
        assert_eq!(apply(&mut workbench, &mut host, "start_node", json!({"node": "camera_measurement"})).unwrap()["phase"], "checking");
        event(&mut workbench,Event::Answered { ticket: asked(&calls, "admit"), result: Ok(json!(["Frames per cycle exceeds the pulse's frames"])) }, &mut host);
        event(&mut workbench,Event::Answered { ticket: asked(&calls, "declare_outputs"), result: Err("boom".into()) }, &mut host);
        let issues = workbench.state(&host)["logic"][0]["issues"].clone();
        assert_eq!(issues, json!(["Frames per cycle exceeds the pulse's frames", "declare_outputs: boom"]));
        assert!(calls.borrow().1.is_empty(), "the Start that waited is refused, not run");
    }

    /// v2 LogicCandidate / finalize(draft=running): a start queued behind another run, the draft
    /// edited meanwhile, still claims and passes what its own configuration named.
    #[test]
    fn a_queued_start_keeps_what_its_configuration_named() {
        let mut catalog = catalog();
        catalog["types"].as_array_mut().unwrap().extend([
            json!({"type_id": "rf.virtual", "kind": "device", "family": "rf", "label": "Virtual RF", "capabilities": ["rf.source"], "parameters": []}),
            json!({"type_id": "device_scan", "kind": "node", "label": "Device scan", "parameters": [{"name": "plan", "type": "text", "label": "Plan", "default": ""}],
                   "node": {"node_kind": "measurement", "outputs": [], "queries": ["resolve_device_claims"]}})]);
        let root = scratch("audit-frozen");
        let calls = Rc::new(RefCell::new(Calls::default()));
        let mut workbench = Workbench::new(&root, Box::new(FakePlugins { calls: calls.clone(), catalog })).unwrap();
        let mut host = FakeHost::default();
        apply(&mut workbench, &mut host, "add_device", json!({"type_id": "rf.virtual", "instance_id": "rf"})).unwrap();
        apply(&mut workbench, &mut host, "apply_devices", json!({})).unwrap();
        event(&mut workbench,Event::Opened { instance: "rf".into(), working_point: json!({"tunable": true}) }, &mut host);
        add_open_node(&mut workbench, &mut host, json!({"type_id": "device_scan", "values": {"plan": "device:rf:frequency"}}));
        let named = |field: &str| json!({"tunable:rf": {"instance": "rf", "fields": [field]}});
        event(&mut workbench,Event::Answered { ticket: asked(&calls, "resolve_device_claims"), result: Ok(named("frequency")) }, &mut host);
        apply(&mut workbench, &mut host, "start_node", json!({"node": "device_scan"})).unwrap();
        // A Restart waits for its own run: queued, edited meanwhile, then started with what it was started with.
        assert_eq!(apply(&mut workbench, &mut host, "start_node", json!({"node": "device_scan"})).unwrap()["phase"], "preparing");
        apply(&mut workbench, &mut host, "edit_node", json!({"node": "device_scan", "values": {"plan": "device:rf:amplitude"}})).unwrap();
        event(&mut workbench,Event::Answered { ticket: asked(&calls, "resolve_device_claims"), result: Ok(named("amplitude")) }, &mut host);
        event(&mut workbench,Event::RunFinished { run: "device_scan#1".into(), outcome: "stopped".into(), error: None, generation: None }, &mut host);
        let spec = calls.borrow().1.last().cloned().unwrap();
        assert_eq!((spec.run.as_str(), spec.values["plan"].as_str(), spec.devices.get("tunable:rf").map(String::as_str)),
                   ("device_scan#2", Some("device:rf:frequency"), Some("rf")));
        assert!(spec.shared.contains("tunable:rf"), "a device the draft names is shared, its fields protected");
        assert_eq!(workbench.run("device_scan#2").unwrap().claims[0].protected, Some(vec!["frequency".to_owned()]));
    }

    /// v2 runtime claims are not exclusive: a scan of the camera its acquisition uses keeps running.
    #[test]
    fn a_scan_of_its_acquisition_camera_runs() {
        let mut catalog = catalog();
        for item in catalog["types"].as_array_mut().unwrap().iter_mut().filter(|item| item["type_id"] == "seamless_scan") {
            item["node"]["queries"] = json!(["resolve_device_claims"]);
        }
        let (_root, mut workbench, calls, mut host) = rig("audit-shared", catalog);
        add_open_node(&mut workbench, &mut host, json!({"type_id": "camera_measurement"}));
        add_open_node(&mut workbench, &mut host, json!({"type_id": "occupancy"}));
        add_open_node(&mut workbench, &mut host, json!({"type_id": "seamless_scan", "values": {"acquisition_logic": "camera_measurement"},
            "source_signal": "@logic/camera_measurement/frames"}));
        event(&mut workbench,Event::Answered { ticket: asked(&calls, "resolve_device_claims"),
            result: Ok(json!({"tunable:mot_camera": {"instance": "mot_camera", "fields": ["exposure_seconds"]}})) }, &mut host);
        apply(&mut workbench, &mut host, "start_node", json!({"node": "seamless_scan"})).unwrap();
        event(&mut workbench,Event::RunStarted { run: "seamless_scan#1".into() }, &mut host);
        event(&mut workbench,Event::RunAcquire { run: "seamless_scan#1".into(), ticket: 7, node: "camera_measurement".into(), completion: "ready".into() }, &mut host);
        assert_eq!(workbench.run("seamless_scan#1").unwrap().state, "running");
        assert_eq!(calls.borrow().1.last().unwrap().run, "camera_measurement#1");
        // While the Task runs, only its own acquisition chain starts.
        event(&mut workbench,Event::RunAcquire { run: "seamless_scan#1".into(), ticket: 8, node: "occupancy".into(), completion: "ready".into() }, &mut host);
        assert_eq!(count(&calls, "acquired seamless_scan#1 8 false"), 1);
        // A run asked to stop starts nothing more.
        apply(&mut workbench, &mut host, "stop_node", json!({"node": "seamless_scan"})).unwrap();
        event(&mut workbench,Event::RunAcquire { run: "seamless_scan#1".into(), ticket: 9, node: "camera_measurement".into(), completion: "ready".into() }, &mut host);
        assert_eq!(count(&calls, "acquired seamless_scan#1 9 false"), 1);
    }

    /// v2: Stop always reaches the run, and a request made after it is the Stop, never shown.
    #[test]
    fn every_stop_is_sent_and_a_request_after_it_is_answered_as_stopped() {
        let (_root, mut workbench, calls, mut host) = rig("audit-stop", catalog());
        add_open_node(&mut workbench, &mut host, json!({"type_id": "camera_measurement"}));
        apply(&mut workbench, &mut host, "start_node", json!({"node": "camera_measurement"})).unwrap();
        apply(&mut workbench, &mut host, "stop_node", json!({"node": "camera_measurement"})).unwrap();
        apply(&mut workbench, &mut host, "stop_node", json!({"node": "camera_measurement"})).unwrap();
        assert_eq!(count(&calls, "stop camera_measurement#1"), 2);
        event(&mut workbench,Event::RunRequest { run: "camera_measurement#1".into(), request: json!({"request": "camera_measurement#1:4", "kind": "point-selection"}) }, &mut host);
        assert_eq!(count(&calls, "answer camera_measurement#1 camera_measurement#1:4 false"), 1);
        assert!(workbench.run("camera_measurement#1").unwrap().request.is_none());
    }

    /// v2: a physical axis set by hand is confirmed by the operator only; anyone may cancel it.
    #[test]
    fn a_manual_axis_is_answered_by_the_operator() {
        let (_root, mut workbench, _calls, mut host) = rig("audit-manual", catalog());
        add_open_node(&mut workbench, &mut host, json!({"type_id": "camera_measurement"}));
        apply(&mut workbench, &mut host, "start_node", json!({"node": "camera_measurement"})).unwrap();
        event(&mut workbench,Event::RunRequest { run: "camera_measurement#1".into(), request: json!({"request": "r1", "kind": "manual-axis"}) }, &mut host);
        let answer = json!({"node": "camera_measurement", "request": "r1", "response": {}});
        workbench.apply("answer_request", &answer, &Origin::Remote("ai".into()), 1, &mut host);
        assert!(workbench.take_replies().pop().unwrap().1.unwrap_err().starts_with("Physical manual-axis requests cannot be answered here"));
        assert!(apply(&mut workbench, &mut host, "answer_request", answer).is_ok());
    }

    /// v2 field_policy: a claim protects the fields its device couples to the ones it names.
    #[test]
    fn a_protected_field_protects_the_fields_coupled_to_it() {
        let (_root, mut workbench, calls, mut host) = rig("audit-coupled", catalog());
        add_open_node(&mut workbench, &mut host, json!({"type_id": "camera_measurement"}));
        apply(&mut workbench, &mut host, "start_node", json!({"node": "camera_measurement"})).unwrap();
        workbench.apply("read_device", &json!({"device": "mot_camera"}), &Origin::Operator, 2, &mut host);
        let row = |name: &str, group: [&str; 2]| json!({"field": {"name": name, "type": "float", "label": name, "default": 0.0}, "current": 1.0,
            "live_write": true, "dependency_group": group});
        event(&mut workbench,Event::Answered { ticket: asked(&calls, "fields"),
            result: Ok(json!([row("exposure_seconds", ["exposure_seconds", "gain"]), row("gain", ["gain", "exposure_seconds"])])) }, &mut host);
        assert_eq!(apply(&mut workbench, &mut host, "tune_device", json!({"device": "mot_camera", "field": "gain", "value": 2.0})).unwrap_err(),
                   "gain: Protected by camera_measurement");
    }

    /// v2 InstallationConfig, v2 discover: two devices of one name do not open; a scan waits for a device change.
    #[test]
    fn a_draft_with_two_devices_of_one_name_does_not_open_and_a_scan_waits_for_init() {
        let mut catalog = catalog();
        for item in catalog["types"].as_array_mut().unwrap().iter_mut().filter(|item| item["type_id"] == "camera.virtual_mot") { item["queries"] = json!(["discover"]); }
        let root = scratch("audit-names");
        let calls = Rc::new(RefCell::new(Calls::default()));
        let mut workbench = Workbench::new(&root, Box::new(FakePlugins { calls: calls.clone(), catalog })).unwrap();
        let mut host = FakeHost::default();
        for id in ["a", "b"] { apply(&mut workbench, &mut host, "add_device", json!({"type_id": "camera.virtual_mot", "instance_id": id})).unwrap(); }
        apply(&mut workbench, &mut host, "edit_device", json!({"device": "b", "role": "a"})).unwrap();
        assert_eq!(apply(&mut workbench, &mut host, "apply_devices", json!({})).unwrap_err(), "this apparatus cannot be initialized: duplicate device role: ['a']");
        apply(&mut workbench, &mut host, "edit_device", json!({"device": "b", "role": "b"})).unwrap();
        apply(&mut workbench, &mut host, "apply_devices", json!({})).unwrap();
        assert_eq!(apply(&mut workbench, &mut host, "discover_devices", json!({})).unwrap_err(), "finish the current device change before scanning hardware");
    }

    /// v2 device maintenance waits 30 s for the runs it stops, then leaves the device as it is and says who it waited for.
    #[test]
    fn an_apply_waits_thirty_seconds_for_the_runs_it_stops() {
        let (_root, mut workbench, _calls, mut host) = rig("audit-wait", catalog());
        add_open_node(&mut workbench, &mut host, json!({"type_id": "camera_measurement"}));
        apply(&mut workbench, &mut host, "start_node", json!({"node": "camera_measurement"})).unwrap();
        apply(&mut workbench, &mut host, "edit_device", json!({"device": "mot_camera", "parameters": {"exposure_seconds": 0.07}})).unwrap();
        apply(&mut workbench, &mut host, "apply_devices", json!({})).unwrap();
        workbench.poll(std::time::Instant::now());
        assert_eq!(workbench.apparatus_state()["busy"], "applying device changes");
        workbench.poll(std::time::Instant::now() + std::time::Duration::from_secs(31));
        let state = workbench.apparatus_state();
        assert_eq!((state["busy"].clone(), state["report"]["text"].clone(), state["action"].clone()),
                   (Value::Null, json!("device changes did not apply: device maintenance still waits for camera_measurement"), json!("apply")));
    }

    /// v2: a region's draft fields are the run's own once it starts; closing the region keeps them.
    #[test]
    fn a_started_run_keeps_what_a_region_set() {
        let mut catalog = catalog();
        for item in catalog["types"].as_array_mut().unwrap().iter_mut().filter(|item| item["type_id"] == "camera_measurement") {
            item["node"]["selection_mappings"] = json!([{"plot_kind": "image", "selector_kind": "area", "draft_fields": ["frames_per_cycle"]}]);
        }
        let (_root, mut workbench, calls, mut host) = rig("audit-region", catalog);
        add_open_node(&mut workbench, &mut host, json!({"type_id": "camera_measurement"}));
        let panel = host.add_panel("", "@logic/camera_measurement/frames", "image", 1, json!({})).unwrap();
        workbench.panel_added(&panel);
        workbench.panel_selection(&panel, "@logic/camera_measurement/frames", Some(&json!({"plot_kind": "image", "selector_kind": "area"})), &json!({}));
        event(&mut workbench,Event::Answered { ticket: asked(&calls, "selection_patch"), result: Ok(json!({"frames_per_cycle": 3})) }, &mut host);
        apply(&mut workbench, &mut host, "start_node", json!({"node": "camera_measurement"})).unwrap();
        workbench.panel_closed(&panel);
        assert_eq!(workbench.node("camera_measurement").unwrap().draft.values["frames_per_cycle"], 3);
    }

    /// A region narrows the draft it was drawn over: moved, its producer is asked about
    /// the field as it held before the region set it, and removed, it gives that back.
    #[test]
    fn a_moved_region_is_asked_about_the_draft_it_was_drawn_over() {
        let mut catalog = catalog();
        catalog["types"][2]["node"]["selection_mappings"] = json!([{"plot_kind": "image", "selector_kind": "area", "draft_fields": ["frames_per_cycle"]}]);
        let (_root, mut wb, calls, mut host) = rig("region-base", catalog);
        add_open_node(&mut wb, &mut host, json!({"type_id": "camera_measurement"}));
        let select = |wb: &mut Workbench, coordinate| { wb.panel_selection("p", "@logic/camera_measurement/frames",
            Some(&json!({"plot_kind": "image", "selector_kind": "area", "coordinate": coordinate})), &json!({})); };
        let asked_draft = |calls: &Rc<RefCell<Calls>>| calls.borrow().2.iter().rev().find(|(query, ..)| query == "selection_patch").unwrap().2["draft"].clone();
        select(&mut wb, 1);
        wb.on_event(Event::Answered { ticket: asked(&calls, "selection_patch"), result: Ok(json!({"frames_per_cycle": 3})) }, &mut host);
        assert_eq!(wb.node("camera_measurement").unwrap().draft.values["frames_per_cycle"], 3);
        select(&mut wb, 2);
        assert_eq!(asked_draft(&calls)["frames_per_cycle"], 1, "asked about its own last write");
        wb.on_event(Event::Answered { ticket: asked(&calls, "selection_patch"), result: Ok(json!({"frames_per_cycle": 4})) }, &mut host);
        assert_eq!(wb.node("camera_measurement").unwrap().draft.values["frames_per_cycle"], 4);
        wb.panel_selection("p", "@logic/camera_measurement/frames", None, &json!({}));
        assert_eq!(wb.node("camera_measurement").unwrap().draft.values["frames_per_cycle"], 1);
    }

    /// v2 `_commit_logic_draft`: an edit of a field a region set makes it the operator's --
    /// the region, moved, is asked about the edit, and removed, leaves it.
    #[test]
    fn an_edit_takes_what_a_region_set() {
        let mut catalog = catalog();
        catalog["types"][2]["node"]["selection_mappings"] = json!([{"plot_kind": "image", "selector_kind": "area", "draft_fields": ["frames_per_cycle"]}]);
        let (_root, mut wb, calls, mut host) = rig("region-edit", catalog);
        add_open_node(&mut wb, &mut host, json!({"type_id": "camera_measurement"}));
        let select = |wb: &mut Workbench, coordinate| { wb.panel_selection("p", "@logic/camera_measurement/frames",
            Some(&json!({"plot_kind": "image", "selector_kind": "area", "coordinate": coordinate})), &json!({})); };
        select(&mut wb, 1);
        wb.on_event(Event::Answered { ticket: asked(&calls, "selection_patch"), result: Ok(json!({"frames_per_cycle": 3})) }, &mut host);
        apply(&mut wb, &mut host, "edit_node", json!({"node": "camera_measurement", "values": {"frames_per_cycle": 5}})).unwrap();
        select(&mut wb, 2);
        assert_eq!(calls.borrow().2.iter().rev().find(|(query, ..)| query == "selection_patch").unwrap().2["draft"]["frames_per_cycle"], 5);
        wb.panel_selection("p", "@logic/camera_measurement/frames", None, &json!({}));
        assert_eq!(wb.node("camera_measurement").unwrap().draft.values["frames_per_cycle"], 5);
    }

    /// v2 lab_run_recipe: followers are standing processors, each once, apart from the entry node.
    #[test]
    fn a_recipe_follows_with_processors_only() {
        let root = scratch("audit-recipe");
        let (mut workbench, _calls) = workbench(&root);
        let mut host = FakeHost::default();
        std::fs::create_dir_all(root.join("recipes")).unwrap();
        let bundled = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("recipes");
        let text = std::fs::read_to_string(bundled.join("trap_lifetime.recipe.json")).unwrap()
            .replace(r#""followers": ["occupancy", "frame_survival"]"#, r#""followers": ["camera_measurement"]"#);
        std::fs::write(root.join("recipes/trap_lifetime.recipe.json"), text).unwrap();
        std::fs::copy(bundled.join("trap_lifetime.layout.json"), root.join("recipes/trap_lifetime.layout.json")).unwrap();
        let parameters = json!({"camera": "c", "sequencer": "s", "pulse_path": "p.json", "calibrations": [], "scan_plan": "{}", "panel_semantic": {}});
        assert_eq!(apply(&mut workbench, &mut host, "run_recipe", json!({"name": "trap_lifetime", "parameters": parameters})).unwrap_err(),
                   "Recipe followers must be standing processors, not device executions");
    }
    fn acquisition_pair(checking: bool) -> (Workbench, Rc<RefCell<Calls>>, FakeHost) {
        let mut catalog = catalog();
        if checking { catalog["types"][2]["node"]["queries"] = json!(["admit"]); }
        let (_root, mut wb, calls, mut host) = rig("audit-isolated", catalog);
        add_open_node(&mut wb, &mut host, json!({"type_id":"camera_measurement"}));
        add_open_node(&mut wb, &mut host, json!({"type_id":"seamless_scan", "values":{"acquisition_logic":"camera_measurement"},"source_signal":"@logic/camera_measurement/frames"}));
        apply(&mut wb,&mut host,"start_node",json!({"node":"seamless_scan"})).unwrap();
        event(&mut wb,Event::RunStarted{run:"seamless_scan#1".into()},&mut host);
        event(&mut wb,Event::RunAcquire{run:"seamless_scan#1".into(),ticket:55,node:"camera_measurement".into(),completion:"ready".into()},&mut host);
        (wb,calls,host)
    }
    #[test]
    fn parent_stop_and_checking_acquisition() {
        let (mut wb,calls,mut host)=acquisition_pair(true);
        assert!(wb.node("camera_measurement").unwrap().start_after_check.is_some());
        apply(&mut wb,&mut host,"stop_node",json!({"node":"seamless_scan"})).unwrap();
        event(&mut wb,Event::RunFinished{run:"seamless_scan#1".into(),outcome:"stopped".into(),error:None,generation:None},&mut host);
        let ticket=asked(&calls,"admit");
        event(&mut wb,Event::Answered{ticket,result:Ok(json!([]))},&mut host);
        assert!(!calls.borrow().1.iter().any(|run|run.node_id=="camera_measurement"));
    }
    #[test]
    fn child_stop_during_checking() {
        let (mut wb,calls,mut host)=acquisition_pair(true);
        apply(&mut wb,&mut host,"stop_node",json!({"node":"camera_measurement"})).unwrap();
        assert_eq!(count(&calls,"acquired seamless_scan#1 55 false"),1);
    }
    #[test]
    fn ready_is_not_terminal() {
        let (mut wb,calls,mut host)=acquisition_pair(false);
        event(&mut wb,Event::RunFinished{run:"camera_measurement#1".into(),outcome:"completed".into(),error:None,generation:None},&mut host);
        assert_eq!(count(&calls,"acquired seamless_scan#1 55 true"),0); assert_eq!(count(&calls,"acquired seamless_scan#1 55 false"),1);
    }
    #[test]
    fn acquisition_completion_contract() {
        let (mut wb,calls,mut host)=acquisition_pair(false);
        event(&mut wb,Event::RunAcquire{run:"seamless_scan#1".into(),ticket:56,node:"camera_measurement".into(),completion:"banana".into()},&mut host);
        assert_eq!(count(&calls,"acquired seamless_scan#1 56 false"),1); assert!(wb.node("camera_measurement").unwrap().pending.is_none());
    }
    #[test]
    fn unicode_names() {
        assert_eq!(crate::logic::name_key("Ა"), crate::logic::name_key("ა"));
        assert_eq!(crate::logic::name_key("Straße"), crate::logic::name_key("STRASSE"));
        assert!(crate::logic::name_problem("Ｃamera","new",[("old","Camera")]).is_some()); assert!(crate::logic::name_problem("Cam\u{200b}era","new",[]).is_some()); assert!(crate::logic::name_problem("Ｓｉｇｎａｌｓ","new",[]).is_some());
    }

    #[test]
    fn role_apply_refreshes_editor() {
        let mut catalog=catalog();
        catalog["types"][4]["node"]["queries"]=json!(["editor_facts"]);
        let (_root,mut wb,calls,mut host)=rig("audit-role",catalog);
        add_open_node(&mut wb,&mut host,json!({"type_id":"seamless_scan"}));
        let before=calls.borrow().2.iter().filter(|(name, ..)|name=="editor_facts").count();
        apply(&mut wb,&mut host,"edit_device",json!({"device":"mot_camera","role":"Imaging camera"})).unwrap();
        apply(&mut wb,&mut host,"apply_devices",json!({})).unwrap();
        let after=calls.borrow().2.iter().filter(|(name, ..)|name=="editor_facts").count();
        assert_eq!(wb.device_role("mot_camera"),"Imaging camera"); assert!(after>before);
    }
    #[test]
    fn frozen_outputs() {
        let mut catalog=catalog();
        catalog["types"].as_array_mut().unwrap().push(json!({"type_id":"dynamic","kind":"node","parameters":[{"name":"output","type":"str","default":"old"}],
            "node":{"node_kind":"processor","inputs":[{"name":"signal","delivery":"exact"}],"outputs":[],"queries":["declare_outputs"]}}));
        let (_root,mut wb,calls,mut host)=rig("audit-frozen-outputs",catalog);
        add_open_node(&mut wb,&mut host,json!({"type_id":"camera_measurement"}));
        host.signals.push(SignalInfo{name:"@logic/camera_measurement/frames".into(),owner:"camera_measurement".into(),generation:"g1".into(),published:true,sealed:false,shape:"".into()});
        add_open_node(&mut wb,&mut host,json!({"type_id":"dynamic","source_signal":"@logic/camera_measurement/frames"}));
        event(&mut wb,Event::Answered{ticket:asked(&calls,"declare_outputs"),result:Ok(json!([{"name":"old"}]))},&mut host);
        apply(&mut wb,&mut host,"start_node",json!({"node":"dynamic"})).unwrap();
        event(&mut wb,Event::RunStarted{run:"dynamic#1".into()},&mut host);
        apply(&mut wb,&mut host,"edit_node",json!({"node":"dynamic","values":{"output":"new"}})).unwrap();
        event(&mut wb,Event::Answered{ticket:asked(&calls,"declare_outputs"),result:Ok(json!([{"name":"new"}]))},&mut host);
        event(&mut wb,Event::RunFinished{run:"dynamic#1".into(),outcome:"completed".into(),error:None,generation:None},&mut host);
        host.signals[0].generation="g2".into(); wb.follow_sources(&mut host); settle_preparations(&mut wb, &mut host);
        let spec=calls.borrow().1.last().unwrap().clone();
        assert_eq!(spec.values["output"],"old"); assert_eq!(spec.outputs.keys().collect::<Vec<_>>(),["old"]);
    }

    #[test]
    fn late_ready_after_stop() {
        let (mut wb,calls,mut host)=acquisition_pair(false);
        apply(&mut wb,&mut host,"stop_node",json!({"node":"camera_measurement"})).unwrap();
        event(&mut wb,Event::RunReady{run:"camera_measurement#1".into()},&mut host);
        assert_eq!(wb.run("camera_measurement#1").unwrap().state,"stopping"); assert_eq!(count(&calls,"acquired seamless_scan#1 55 true"),0);
    }
    #[test]
    fn parent_stops_only_its_child_run() {
        let mut catalog=catalog();catalog["types"][4]["node"]["node_kind"]=json!("measurement");
        let (_root,mut wb,_calls,mut host)=rig("audit-owned-child",catalog);
        add_open_node(&mut wb,&mut host,json!({"type_id":"camera_measurement"}));
        add_open_node(&mut wb,&mut host,json!({"type_id":"seamless_scan","values":{"acquisition_logic":"camera_measurement"},"source_signal":"@logic/camera_measurement/frames"}));
        apply(&mut wb,&mut host,"start_node",json!({"node":"seamless_scan"})).unwrap();
        event(&mut wb,Event::RunStarted{run:"seamless_scan#1".into()},&mut host);
        event(&mut wb,Event::RunAcquire{run:"seamless_scan#1".into(),ticket:77,node:"camera_measurement".into(),completion:"ready".into()},&mut host);
        event(&mut wb,Event::RunReady{run:"camera_measurement#1".into()},&mut host);
        apply(&mut wb,&mut host,"start_node",json!({"node":"camera_measurement"})).unwrap();
        event(&mut wb,Event::RunFinished{run:"camera_measurement#1".into(),outcome:"stopped".into(),error:None,generation:None},&mut host);
        event(&mut wb,Event::RunStarted{run:"camera_measurement#2".into()},&mut host);
        event(&mut wb,Event::RunFinished{run:"seamless_scan#1".into(),outcome:"completed".into(),error:None,generation:None},&mut host);
        assert_eq!(wb.run("camera_measurement#2").unwrap().state,"running");
    }

    #[test]
    fn old_reply_cannot_replace_new_refresh() {
        let mut catalog=catalog();catalog["types"][4]["node"]["queries"]=json!(["editor_facts"]);
        let (_root,mut wb,calls,mut host)=rig("audit-refresh-reply",catalog);
        add_open_node(&mut wb,&mut host,json!({"type_id":"seamless_scan"}));
        let old=asked(&calls,"editor_facts");
        apply(&mut wb,&mut host,"check_node",json!({"node":"seamless_scan"})).unwrap();
        let fresh=asked(&calls,"editor_facts");assert_ne!(old,fresh);
        event(&mut wb,Event::Answered{ticket:fresh,result:Ok(json!({"marker":"fresh"}))},&mut host);
        event(&mut wb,Event::Answered{ticket:old,result:Ok(json!({"marker":"old"}))},&mut host);
        assert_eq!(wb.node_row(wb.node("seamless_scan").unwrap(),&[])["editor_facts"]["marker"],"fresh");
    }
    #[test]
    fn old_reply_cannot_touch_recreated_node() {
        let mut catalog=catalog();catalog["types"][4]["node"]["queries"]=json!(["editor_facts"]);
        let (_root,mut wb,calls,mut host)=rig("audit-recreate-reply",catalog);
        add_open_node(&mut wb,&mut host,json!({"type_id":"seamless_scan"}));
        let old=asked(&calls,"editor_facts");
        apply(&mut wb,&mut host,"remove_node",json!({"node":"seamless_scan"})).unwrap();
        add_open_node(&mut wb,&mut host,json!({"type_id":"seamless_scan"}));
        let fresh=asked(&calls,"editor_facts");
        event(&mut wb,Event::Answered{ticket:fresh,result:Ok(json!({"marker":"fresh"}))},&mut host);
        event(&mut wb,Event::Answered{ticket:old,result:Ok(json!({"marker":"old"}))},&mut host);
        assert_eq!(wb.node_row(wb.node("seamless_scan").unwrap(),&[])["editor_facts"]["marker"],"fresh");
    }

    #[test]
    fn completed_run_still_opens_first_preview() {
        let (_root,mut wb,_calls,mut host)=rig("audit-short-preview",catalog());
        add_open_node(&mut wb,&mut host,json!({"type_id":"camera_measurement"}));
        apply(&mut wb,&mut host,"start_node",json!({"node":"camera_measurement"})).unwrap();
        host.signals.push(SignalInfo{name:"@logic/camera_measurement/frames".into(),owner:"camera_measurement".into(),generation:"g1".into(),published:true,sealed:true,shape:"(1,1,1)".into()});
        event(&mut wb,Event::RunFinished{run:"camera_measurement#1".into(),outcome:"completed".into(),error:None,generation:Some("g1".into())},&mut host);
        wb.open_previews(&mut host);
        assert_eq!(host.panels.len(),1);
    }
    #[test]
    fn a_failed_preparation_preserves_the_active_run() {
        let (_root,mut wb,calls,mut host)=rig("prepare-failure",catalog());
        add_open_node(&mut wb,&mut host,json!({"type_id":"camera_measurement"}));
        apply(&mut wb,&mut host,"start_node",json!({"node":"camera_measurement"})).unwrap();
        event(&mut wb,Event::RunStarted{run:"camera_measurement#1".into()},&mut host);
        wb.apply("start_node",&json!({"node":"camera_measurement"}),&Origin::Operator,2,&mut host);
        let ticket=wb.node("camera_measurement").unwrap().preparing.unwrap();
        assert_eq!(wb.run("camera_measurement#1").unwrap().state,"running");
        wb.on_event(Event::Answered{ticket,result:Err("build failed".into())},&mut host);
        assert_eq!(wb.run("camera_measurement#1").unwrap().state,"running");
        assert_eq!(count(&calls,"stop camera_measurement#1"),0);
        assert_eq!(wb.node_row(wb.node("camera_measurement").unwrap(),&[])["state"],"warning");
    }
    #[test]
    fn restart_to_an_unpublished_source_stops_the_previous_run() {
        let mut catalog=catalog(); catalog["types"][3]["node"]["artifact_inputs"]=json!([]);
        let (_root,mut wb,calls,mut host)=rig("restart-waiting",catalog);
        add_open_node(&mut wb,&mut host,json!({"type_id":"camera_measurement"}));
        add_open_node(&mut wb,&mut host,json!({"type_id":"camera_measurement", "node_id":"next_camera"}));
        host.signals.push(SignalInfo{name:"@logic/camera_measurement/frames".into(),owner:"camera_measurement".into(),generation:"g1".into(),published:true,sealed:false,shape:"".into()});
        add_open_node(&mut wb,&mut host,json!({"type_id":"occupancy","source_signal":"@logic/camera_measurement/frames"}));
        apply(&mut wb,&mut host,"start_node",json!({"node":"occupancy"})).unwrap();
        event(&mut wb,Event::RunStarted{run:"occupancy#1".into()},&mut host);
        apply(&mut wb,&mut host,"edit_node",json!({"node":"occupancy","source_signal":"@logic/next_camera/frames"})).unwrap();
        apply(&mut wb,&mut host,"start_node",json!({"node":"occupancy"})).unwrap();
        assert_eq!(wb.run("occupancy#1").unwrap().state,"stopping");
        event(&mut wb,Event::RunFinished{run:"occupancy#1".into(),outcome:"stopped".into(),error:None,generation:None},&mut host);
        assert_eq!(calls.borrow().1.iter().filter(|run|run.node_id=="occupancy").count(),1);
        host.signals.push(SignalInfo{name:"@logic/next_camera/frames".into(),owner:"next_camera".into(),generation:"g2".into(),published:true,sealed:false,shape:"".into()});
        wb.follow_sources(&mut host); settle_preparations(&mut wb,&mut host);
        assert_eq!(calls.borrow().1.last().unwrap().source_signal,"@logic/next_camera/frames");
    }

    #[test]
    fn stopped_or_superseded_preparations_are_discarded() {
        let (_root,mut wb,calls,mut host)=rig("prepare-cancel",catalog());
        add_open_node(&mut wb,&mut host,json!({"type_id":"camera_measurement"}));
        wb.apply("start_node",&json!({"node":"camera_measurement"}),&Origin::Operator,2,&mut host);
        let first=wb.node("camera_measurement").unwrap().preparing.unwrap();
        wb.apply("start_node",&json!({"node":"camera_measurement"}),&Origin::Operator,3,&mut host);
        let second=wb.node("camera_measurement").unwrap().preparing.unwrap();
        wb.on_event(Event::Answered{ticket:first,result:Ok(json!({"prepared":"old"}))},&mut host);
        wb.apply("stop_node",&json!({"node":"camera_measurement"}),&Origin::Operator,4,&mut host);
        wb.on_event(Event::Answered{ticket:second,result:Ok(json!({"prepared":"stopped"}))},&mut host);
        assert!(calls.borrow().1.is_empty());
        let disposed:Vec<Value>=calls.borrow().2.iter().filter(|(op,..)|op=="run.discard").map(|(_,_,params)|params["prepared"].clone()).collect();
        assert_eq!(disposed,vec![json!("old"),json!("stopped")]);
    }

    #[test]
    fn replacing_a_device_cancels_its_unstarted_candidate() {
        let (_root,mut wb,calls,mut host)=rig("device-prepare-cancel",catalog());
        add_open_node(&mut wb,&mut host,json!({"type_id":"camera_measurement"}));
        wb.apply("start_node",&json!({"node":"camera_measurement"}),&Origin::Operator,2,&mut host);
        let ticket=wb.node("camera_measurement").unwrap().preparing.unwrap();
        wb.close_device("mot_camera",&mut host).unwrap();
        wb.on_event(Event::Answered{ticket,result:Ok(json!({"prepared":"closed-device"}))},&mut host);
        assert!(calls.borrow().1.is_empty());
        assert!(calls.borrow().2.iter().any(|(op,_,params)|op=="run.discard" && params["prepared"]=="closed-device"));
    }

    #[test]
    fn only_the_latest_device_read_can_replace_its_fields() {
        let (_root,mut wb,calls,mut host)=rig("device-read-order",catalog());
        wb.read_device("mot_camera",None,None).unwrap(); let old=asked(&calls,"fields");
        wb.read_device("mot_camera",None,None).unwrap(); let fresh=asked(&calls,"fields");
        wb.on_event(Event::Answered{ticket:fresh,result:Ok(json!({"marker":"fresh"}))},&mut host);
        wb.on_event(Event::Answered{ticket:old,result:Ok(json!({"marker":"old"}))},&mut host);
        assert_eq!(wb.devices["mot_camera"].fields.as_ref().unwrap()["marker"],"fresh");
    }

    #[test]
    fn a_selection_changed_away_and_back_still_rejects_its_old_reply() {
        let mut catalog=catalog();
        catalog["types"][2]["node"]["selection_mappings"]=json!([{"plot_kind":"image","selector_kind":"area","draft_fields":["frames_per_cycle"]}]);
        let (_root,mut wb,calls,mut host)=rig("selection-order",catalog);
        add_open_node(&mut wb,&mut host,json!({"type_id":"camera_measurement"}));
        let select=|wb:&mut Workbench, coordinate|{wb.panel_selection("p","@logic/camera_measurement/frames",Some(&json!({"plot_kind":"image","selector_kind":"area","coordinate":coordinate})),&json!({}));};
        select(&mut wb,1);let old=asked(&calls,"selection_patch");
        select(&mut wb,2);select(&mut wb,1);let fresh=asked(&calls,"selection_patch");
        wb.on_event(Event::Answered{ticket:old,result:Ok(json!({"frames_per_cycle":2}))},&mut host);
        assert_eq!(wb.node("camera_measurement").unwrap().draft.values["frames_per_cycle"],1);
        wb.on_event(Event::Answered{ticket:fresh,result:Ok(json!({"frames_per_cycle":3}))},&mut host);
        assert_eq!(wb.node("camera_measurement").unwrap().draft.values["frames_per_cycle"],3);
    }

    #[test]
    fn restarting_a_parent_cancels_its_child_before_the_parent_finishes() {
        let (mut wb,calls,mut host)=acquisition_pair(false);
        apply(&mut wb,&mut host,"start_node",json!({"node":"seamless_scan"})).unwrap();
        assert_eq!(wb.run("seamless_scan#1").unwrap().state,"stopping");
        assert_eq!(wb.run("camera_measurement#1").unwrap().state,"stopping");
        assert_eq!(count(&calls,"acquired seamless_scan#1 55 false"),1);
    }

    #[test]
    fn recreating_a_node_never_reuses_a_previous_run_identity() {
        let (_root,mut wb,calls,mut host)=rig("run-identity",catalog());
        add_open_node(&mut wb,&mut host,json!({"type_id":"camera_measurement"}));
        apply(&mut wb,&mut host,"start_node",json!({"node":"camera_measurement"})).unwrap();
        event(&mut wb,Event::RunFinished{run:"camera_measurement#1".into(),outcome:"completed".into(),error:None,generation:None},&mut host);
        apply(&mut wb,&mut host,"remove_node",json!({"node":"camera_measurement"})).unwrap();
        add_open_node(&mut wb,&mut host,json!({"type_id":"camera_measurement"}));
        apply(&mut wb,&mut host,"start_node",json!({"node":"camera_measurement"})).unwrap();
        assert_eq!(calls.borrow().1.last().unwrap().run,"camera_measurement#2");
        event(&mut wb,Event::RunStarted{run:"camera_measurement#2".into()},&mut host);
        event(&mut wb,Event::RunReady{run:"camera_measurement#1".into()},&mut host);
        assert_eq!(wb.run("camera_measurement#2").unwrap().state,"running");
        assert!(!wb.run("camera_measurement#2").unwrap().ready);
    }

    #[test]
    fn queued_lifecycle_replies_cannot_open_or_fail_a_replacement() {
        let (_root,mut wb,_calls,mut host)=rig("lifecycle-reply",catalog());
        apply(&mut wb,&mut host,"add_node",json!({"type_id":"camera_measurement"})).unwrap();
        let current=std::sync::Arc::new(std::sync::atomic::AtomicU64::new(2));
        let opened=||Event::Opened{instance:node_instance("camera_measurement"),working_point:Value::Null};
        wb.on_event(Event::Lifecycle{current:current.clone(),revision:1,event:Box::new(opened())},&mut host);
        assert_eq!(wb.node("camera_measurement").unwrap().instance,crate::logic::Instance::Opening);
        wb.on_event(Event::Lifecycle{current:current.clone(),revision:2,event:Box::new(opened())},&mut host);
        wb.on_event(Event::Lifecycle{current,revision:1,event:Box::new(Event::Failed{instance:node_instance("camera_measurement"),error:"old open failed".into()})},&mut host);
        assert_eq!(wb.node("camera_measurement").unwrap().instance,crate::logic::Instance::Open);
    }

    #[test]
    fn a_failed_device_close_is_visible_and_apply_retries_close_before_open() {
        let (_root,mut wb,calls,mut host)=rig("failed-close",catalog());
        wb.close_device("mot_camera",&mut host).unwrap();
        event(&mut wb,Event::Closed{instance:"mot_camera".into(),error:Some("handle still held".into())},&mut host);
        assert!(wb.devices["mot_camera"].close_failed);
        assert_eq!(wb.device_rows().iter().find(|row| row["instance_id"]=="mot_camera").unwrap()["close_failed"],true);
        assert!(wb.devices["mot_camera"].error.as_ref().unwrap().contains("handle still held"));
        let opened=count(&calls,"open mot_camera camera.virtual_mot");
        apply(&mut wb,&mut host,"apply_devices",json!({})).unwrap();
        assert_eq!(count(&calls,"open mot_camera camera.virtual_mot"),opened);
        assert_eq!(count(&calls,"close mot_camera"),2);
        event(&mut wb,Event::Closed{instance:"mot_camera".into(),error:None},&mut host);
        assert_eq!(count(&calls,"open mot_camera camera.virtual_mot"),opened+1);
        assert!(!wb.devices["mot_camera"].close_failed);
    }

    #[test]
    fn shutdown_never_reports_success_while_a_device_close_failed() {
        let (_root,mut wb,_calls,mut host)=rig("shutdown-refusal",catalog());
        apply(&mut wb,&mut host,"shut_down_devices",json!({})).unwrap();
        event(&mut wb,Event::Closed{instance:"mot_camera".into(),error:Some("handle still held".into())},&mut host);
        event(&mut wb,Event::Closed{instance:"sequencer".into(),error:None},&mut host);
        assert!(wb.apparatus_state()["report"]["text"].as_str().unwrap().contains("did not shut down"));
        assert!(wb.devices["mot_camera"].close_failed);
    }

    /// The bench at a Start: a measurement asks every open device that reports its settings,
    /// all at once, and launches once each has answered; a device that answers late, has failed
    /// or reports nothing is entered with why. The pulse board's entry says who drives it -- the
    /// Pulse Editor's On Pulse, or the run that claims it. A processor takes no bench.
    #[test]
    fn a_start_records_the_bench_it_starts_on() {
        let root = scratch("bench");
        let mut catalog = catalog();
        catalog["types"][3]["node"]["artifact_inputs"] = json!([]);
        catalog["types"].as_array_mut().unwrap().push(json!({"type_id": "rf.virtual", "kind": "device", "family": "rf", "label": "Virtual RF",
            "capabilities": ["rf.source"], "parameters": []}));
        std::fs::write(root.join("apparatus.json"), r#"{"format": "zlc.installation",
            "simulation": {"image_shape_yx": [96, 128], "grid_shape_yx": [5, 7], "seed": 0, "world_profile": ""},
            "devices": [{"instance_id": "mot_camera", "role": "mot_camera", "type_id": "camera.virtual_mot", "parameters": {}},
                        {"instance_id": "rf", "role": "Trap RF", "type_id": "rf.virtual", "parameters": {}},
                        {"instance_id": "spare", "role": "spare", "type_id": "camera.virtual_mot", "parameters": {}},
                        {"instance_id": "sequencer", "role": "sequencer", "type_id": "sequencer.virtual", "parameters": {}},
                        {"instance_id": "basler", "role": "basler", "type_id": "camera.pylon", "parameters": {}}]}"#).unwrap();
        let calls = Rc::new(RefCell::new(Calls::default()));
        let mut workbench = Workbench::new(&root, Box::new(FakePlugins { calls: calls.clone(), catalog })).unwrap();
        let mut host = FakeHost::default();
        apply(&mut workbench, &mut host, "apply_devices", json!({})).unwrap();
        for (device, reports) in [("mot_camera", true), ("rf", true), ("spare", false), ("sequencer", true)] {
            event(&mut workbench, Event::Opened { instance: device.into(), working_point: json!({"state": "open", "snapshot": reports}) }, &mut host);
        }
        add_open_node(&mut workbench, &mut host, json!({"type_id": "camera_measurement"}));
        add_open_node(&mut workbench, &mut host, json!({"type_id": "occupancy", "source_signal": "@logic/camera_measurement/frames"}));
        add_open_node(&mut workbench, &mut host, json!({"type_id": "seamless_scan", "values": {"acquisition_logic": "camera_measurement"},
            "source_signal": "@logic/camera_measurement/frames"}));
        // The Pulse Editor plays its own pulse (On Pulse) while the camera measurement starts.
        workbench.device_users().lock().unwrap().insert("sequencer".into(), "PulseGUI".into());
        let snapshots = |calls: &Rc<RefCell<Calls>>| -> Vec<(String, u64)> { calls.borrow().0.iter().filter_map(|call| {
            let words: Vec<&str> = call.split(' ').collect();
            (words.len() == 4 && words[0] == "call" && words[3] == "snapshot").then(|| (words[1].to_owned(), words[2].parse().unwrap()))
        }).collect() };
        apply(&mut workbench, &mut host, "start_node", json!({"node": "camera_measurement"})).unwrap();
        let asked: Vec<String> = snapshots(&calls).into_iter().map(|(device, _)| device).collect();
        assert_eq!(asked, ["mot_camera", "rf", "sequencer"], "each device that reports, at once");
        assert!(calls.borrow().1.is_empty(), "the run waits for the bench");
        assert_eq!(workbench.state(&host)["logic"][0]["status"], "reading the bench's settings");
        let answer = |workbench: &mut Workbench, host: &mut FakeHost, device: &str, result: Result<Value, String>| {
            let ticket = snapshots(&calls).iter().rev().find(|(asked, _)| asked == device).unwrap().1;
            event(workbench, Event::Answered { ticket, result }, host);
        };
        answer(&mut workbench, &mut host, "mot_camera", Ok(json!({"values": {"exposure": 0.02}, "provenance": {"settings_epoch": 3}, "role": "not the core's"})));
        answer(&mut workbench, &mut host, "rf", Err("timeout: snapshot did not answer within 2.0 s".into()));
        assert!(calls.borrow().1.is_empty());
        let board = json!({"loaded": true, "firing": true, "digest": "5f0c", "snapshot": {"program": {"digest": "5f0c"}, "pulse": {"name": "imaging"}}});
        answer(&mut workbench, &mut host, "sequencer", Ok(board.clone()));
        let spec = calls.borrow().1.last().cloned().expect("launched once every answer is in");
        assert_eq!(spec.run, "camera_measurement#1");
        let bench = spec.apparatus.unwrap();
        assert!(bench["taken_at"].as_str().unwrap().ends_with("+00:00"));
        assert_eq!(bench["devices"], json!({
            "basler": {"type_id": "camera.pylon", "role": "basler", "unavailable": format!("failed: {}", workbench.devices["basler"].error.clone().unwrap())},
            "mot_camera": {"type_id": "camera.virtual_mot", "role": "mot_camera", "values": {"exposure": 0.02}, "provenance": {"settings_epoch": 3}},
            "rf": {"type_id": "rf.virtual", "role": "Trap RF", "unavailable": "timeout: snapshot did not answer within 2.0 s"},
            "spare": {"type_id": "camera.virtual_mot", "role": "spare", "unavailable": "does not report its settings"}}));
        let mut sequencer = board.clone();
        sequencer["type_id"] = json!("sequencer.virtual");
        sequencer["role"] = json!("sequencer");
        sequencer["driver"] = json!({"outside": "PulseGUI"});
        assert_eq!(bench["sequencer"], json!({"sequencer": sequencer}));
        event(&mut workbench, Event::RunStarted { run: "camera_measurement#1".into() }, &mut host);

        // A late answer to a start that has launched changes nothing; a processor takes no bench.
        answer(&mut workbench, &mut host, "rf", Ok(json!({"values": {}})));
        let before = snapshots(&calls).len();
        host.signals.push(SignalInfo { name: "@logic/camera_measurement/frames".into(), generation: "g1".into(), owner: "camera_measurement".into(),
            published: true, sealed: false, shape: String::new() });
        apply(&mut workbench, &mut host, "start_node", json!({"node": "occupancy"})).unwrap();
        assert_eq!(calls.borrow().1.last().unwrap().run, "occupancy#1");
        assert!(calls.borrow().1.last().unwrap().apparatus.is_none());
        assert_eq!(snapshots(&calls).len(), before);

        // The board a run claims, driven by nobody at its Start, is driven by that run.
        for run in ["camera_measurement#1", "occupancy#1"] {
            event(&mut workbench, Event::RunFinished { run: run.into(), outcome: "stopped".into(), error: None, generation: None }, &mut host);
        }
        apply(&mut workbench, &mut host, "stop_node", json!({"node": "occupancy"})).unwrap();
        workbench.device_users().lock().unwrap().clear();
        apply(&mut workbench, &mut host, "start_node", json!({"node": "seamless_scan"})).unwrap();
        for (device, _) in snapshots(&calls).into_iter().skip(before) { answer(&mut workbench, &mut host, &device, Ok(json!({}))); }
        let spec = calls.borrow().1.last().cloned().unwrap();
        assert_eq!(spec.run, "seamless_scan#1");
        assert_eq!(spec.apparatus.unwrap()["sequencer"]["sequencer"]["driver"], json!({"run": "seamless_scan#1", "node": "seamless_scan"}));
    }

}
