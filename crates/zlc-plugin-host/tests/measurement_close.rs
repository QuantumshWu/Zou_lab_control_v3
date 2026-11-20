//! Closing a running Camera Measurement, alone or with the application, through the real
//! host: its run ends as stopped, within the close, and it ends its own capture -- the
//! camera's device finds it idle when it closes. The camera is one no trigger reaches
//! (zlc_atom_fixtures.fixture_camera): every read waits out its slice, as a qCMOS whose
//! board plays nothing. A Waveform Measurement waiting for its trigger is closed with the
//! application too. Needs ZLC_PLUGIN_PYTHON.
mod support;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use zlc_core::control::{Origin, SafeActions};
use zlc_core::runtime::Runtime;
use zlc_plugin_host::{HostConfig, HostEvent, PluginHost, RunRequest};

fn host() -> Option<(PluginHost, mpsc::Receiver<HostEvent>)> {
    let Some(python) = std::env::var_os("ZLC_PLUGIN_PYTHON") else {
        eprintln!("ZLC_PLUGIN_PYTHON is not set: skipping the plugin process test");
        return None;
    };
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let config = HostConfig { python: python.into(), python_path: vec![root.join("../../science/src"), root.join("tests/plugins")],
        packages: vec!["zlc_atom".into(), "zlc_atom_fixtures".into()], memory_limit_mib: Some(4096) };
    let (sender, events) = mpsc::channel();
    let sender = std::sync::Mutex::new(sender);
    let host = PluginHost::new(config, Runtime::new(), SafeActions::new(), move |event| { let _ = sender.lock().unwrap().send(event); }).unwrap();
    support::install(&host).unwrap();
    Some((host, events))
}

/// A measurement aligned to lane ch11 of the camera `camera`.
fn aligned(camera: &str) -> RunRequest {
    RunRequest { prepared: None, device_labels: Default::default(), device_units: Default::default(),
        values: json!({"repeat": 0, "trigger_by_channel": true, "trigger_channel": "ch11"}),
        devices: [("camera".to_owned(), camera.to_owned())].into(), shared: Default::default(), reads: Default::default(),
        origin: Origin::Operator, resources: json!({}), source_signal: None, run_directory: None,
        outputs: [("frames".to_owned(), format!("@logic/{camera}/frames"))].into(), record_devices: false }
}

/// The board's answer while it holds no pulse: frames belong to no shot, and none comes anyway.
fn idle_board() -> Value {
    json!({"revision": 0, "state": "none", "lane": "ch11", "command_id": null, "run_repeats": 0, "scan_repeats": 1, "sent_us": 0, "points": []})
}

fn said(log: &Path) -> Vec<String> {
    std::fs::read_to_string(log).unwrap_or_default().lines().map(str::to_owned).collect()
}

#[test]
fn closed_measurements_end_stopped_within_the_close_and_end_their_own_captures() {
    let Some((host, events)) = host() else { return };
    let folder = std::env::temp_dir().join(format!("zlc-measurement-close-{}", std::process::id()));
    std::fs::create_dir_all(&folder).unwrap();
    let log = |camera: &str| folder.join(format!("{camera}.log"));
    // Three cameras, three measurements: `polling` and `closed` read their camera, `asking`
    // still waits for the application to answer its watch of the board.
    let mut runs = HashMap::new();
    for (camera, node) in [("cam_polling", "polling"), ("cam_asking", "asking"), ("cam_closed", "closed")] {
        host.open(camera, "fixture.untriggered_camera", json!({"log": log(camera)}), json!({})).unwrap();
        host.open(node, "camera_measurement", json!({}), json!({})).unwrap();
        runs.insert(host.start_run(node, aligned(camera)).unwrap(), node);
    }
    // A waveform source whose trigger level is never crossed: the measurement reads and waits.
    host.open("source", "fixture.waveform", json!({"rate_hz": 400.0, "record_samples": 1, "outputs": 1, "channels": 3,
        "frequency_hz": 23.7, "noise": 0.03}), json!({})).unwrap();
    host.open("wave", "waveform_measurement", json!({}), json!({})).unwrap();
    runs.insert(host.start_run("wave", RunRequest { prepared: None, device_labels: Default::default(), device_units: Default::default(),
        values: json!({"repeat": 0, "trigger_slope": "rising", "trigger_channel": "q0/c0", "trigger_level": 5.0, "trigger_samples": 20}),
        devices: [("sampler".to_owned(), "source".to_owned())].into(), shared: Default::default(), reads: Default::default(),
        origin: Origin::Operator, resources: json!({}), source_signal: None, run_directory: None,
        outputs: [("q0".to_owned(), "@logic/wave/q0".to_owned())].into(), record_devices: false }).unwrap(), "wave");
    let (mut ready, mut asked) = (Vec::new(), Vec::new());
    let mut outcomes: HashMap<&str, (String, Option<Value>)> = HashMap::new();
    let take = |event: HostEvent, ready: &mut Vec<&'static str>, asked: &mut Vec<&'static str>, outcomes: &mut HashMap<&str, (String, Option<Value>)>| match event {
        HostEvent::Request { run, ticket, op, params, .. } => {
            assert_eq!((op.as_str(), &params["command"]), ("core", &json!("pulse_schedule")), "{params}");
            let node = runs[&run];
            asked.push(node);
            // The application answers two watches; the third is still unanswered when the measurement closes.
            if node != "asking" { host.respond(ticket, Ok(idle_board())); }
        }
        HostEvent::RunReady { run, .. } => ready.push(runs[&run]),
        HostEvent::RunFinished { run, outcome, error, .. } => { outcomes.insert(runs[&run], (outcome, error)); }
        HostEvent::Failed { reason, .. } => panic!("plugin failed: {reason}"),
        _ => {}
    };
    let until = Instant::now() + Duration::from_secs(60);
    while !(ready.contains(&"polling") && ready.contains(&"closed") && ready.contains(&"wave") && asked.contains(&"asking")) {
        let event = events.recv_timeout(until.saturating_duration_since(Instant::now())).expect("the measurements arm and ask the board");
        take(event, &mut ready, &mut asked, &mut outcomes);
    }
    // Every camera is armed and read: a read waits out its slice, as no trigger comes.
    std::thread::sleep(Duration::from_millis(300));

    // Closing one measurement stops its run; it ends its capture before its device is touched.
    let began = Instant::now();
    host.close("closed").unwrap();
    let took = began.elapsed();
    while let Ok(event) = events.try_recv() { take(event, &mut ready, &mut asked, &mut outcomes); }
    assert_eq!(outcomes.get("closed").map(|(outcome, _)| outcome.as_str()), Some("stopped"), "{outcomes:?}");
    assert!(took < Duration::from_secs(1), "a close waits for one read and the capture's end, not {took:?}");

    // Closing the application with one camera measurement reading, one waiting for the application
    // and a waveform measurement waiting for its trigger: each ends stopped within the close, and
    // each camera's device closes after its run let it go.
    let began = Instant::now();
    host.shutdown();
    let took = began.elapsed();
    while let Ok(event) = events.try_recv() { take(event, &mut ready, &mut asked, &mut outcomes); }
    for node in ["polling", "asking", "wave"] {
        assert_eq!(outcomes.get(node).map(|(outcome, _)| outcome.as_str()), Some("stopped"), "{node}: {outcomes:?}");
    }
    assert!(took < Duration::from_secs(2), "closing the application took {took:?}");
    assert_eq!(said(&log("cam_polling")), ["finish", "close idle"], "the measurement ended its own capture");
    assert_eq!(said(&log("cam_closed")), ["finish", "close idle"]);
    // Its watch refused, the waiting measurement disarmed its camera on its way out.
    assert_eq!(said(&log("cam_asking")), ["finish", "close idle"]);
    let _ = std::fs::remove_dir_all(&folder);
}
