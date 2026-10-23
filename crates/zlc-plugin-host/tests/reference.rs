//! A real v2 device and logic node through the host: the Virtual MOT camera and
//! camera_measurement, finite and monitor runs. Needs ZLC_PLUGIN_PYTHON; the
//! camera_measurement port is line D's (claude/v3-atom): ZLC_PLUGIN_PATH names
//! the science trees to import (os path list), else this checkout's science/src.
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::Duration;

use serde_json::json;
use zlc_core::AnyBuffer;
use zlc_core::control::{Origin, SafeActions};
use zlc_core::runtime::Runtime;
use zlc_core::schema::DomainRole;
use zlc_plugin_host::{HostConfig, HostEvent, PluginHost, RunRequest};

#[test]
fn the_virtual_mot_camera_and_camera_measurement_run_through_the_host() {
    let Some(python) = std::env::var_os("ZLC_PLUGIN_PYTHON") else { eprintln!("ZLC_PLUGIN_PYTHON is not set: skipped"); return };
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let python_path = std::env::var_os("ZLC_PLUGIN_PATH").map(|paths| std::env::split_paths(&paths).collect())
        .unwrap_or_else(|| vec![root.join("../../science/src")]);
    let (runtime, safe) = (Runtime::new(), SafeActions::new());
    let (sender, events) = mpsc::channel();
    let sender = std::sync::Mutex::new(sender);
    let host = PluginHost::new(HostConfig { python: python.into(), python_path, packages: vec!["zlc_atom".into()], memory_limit_mib: Some(4096) },
        runtime.clone(), safe, move |event| { let _ = sender.lock().unwrap().send(event); }).unwrap();
    let catalog = host.catalog().unwrap();
    if catalog.find("camera_measurement").is_none() { eprintln!("this science tree has no camera_measurement: skipped"); return; }
    let opened = host.open("mot", "camera.virtual_mot", json!({"exposure_seconds": 0.02}), json!({"seed": 0})).unwrap();
    host.open("camera_measurement1", "camera_measurement", json!({}), json!({})).unwrap();
    let start = |values: serde_json::Value| {
        let token = host.call("camera_measurement1", "query", json!({"name":"prepare_run","values":values,"resources":{},
            "devices":{"camera":{"instance":"mot","role":"mot","type_id":"camera.virtual_mot","capabilities":["camera.adapter"],"tunable":true,"reads":opened["reads"]}}}), Duration::from_secs(30)).unwrap().value["prepared"].as_str().unwrap().to_owned();
        host.start_run("camera_measurement1", RunRequest { prepared: Some(token), device_labels: Default::default(), values,
        devices: [("camera".to_owned(), "mot".to_owned())].into(), shared: Default::default(), origin: Origin::Operator, resources: json!({}),
        source_signal: None, run_directory: None,
        outputs: [("frames".to_owned(), "@logic/camera_measurement1/frames".to_owned())].into() }).unwrap() };
    let wait = |run: &str, stop_after: Option<u64>| loop {
        match events.recv_timeout(Duration::from_secs(120)).expect("run events") {
            HostEvent::RunProgress { run: id, current, .. } if id == run && current.is_some() => {}
            HostEvent::RunReady { run: id, .. } if id == run => {
                if let Some(seconds) = stop_after {
                    std::thread::sleep(Duration::from_secs(seconds));
                    host.stop_run("camera_measurement1", run).unwrap();
                }
            }
            HostEvent::RunFinished { run: id, outcome, error, .. } if id == run => break (outcome, error),
            HostEvent::Failed { reason, .. } => panic!("plugin failed: {reason}"),
            _ => {}
        }
    };
    let signal = |name: &str| runtime.signals().into_iter().rev().find(|item| item.signal.as_ref() == name).expect(name);

    // Finite: two complete cycles placed in the authored run geometry.
    let run = start(json!({"exposure_seconds": 0.02, "repeat": 2, "frames_per_cycle": 1}));
    let (outcome, error) = wait(&run, None);
    assert_eq!(outcome, "completed", "{error:?}");
    let frames = signal("@logic/camera_measurement1/frames");
    assert!(frames.finite && frames.sealed);
    let latest = runtime.read_latest(&frames.generation, "@logic/camera_measurement1/frames").unwrap();
    let canonical = latest.canonical().expect("finite dataset");
    assert_eq!((canonical.written_cells(), canonical.total_cells()), (2, 2));
    assert_eq!(canonical.schema().domain(DomainRole::Cell).physical_shape(), [1200, 1920]);
    let event = latest.data().unwrap();
    let AnyBuffer::U8(pixels) = event.values() else { panic!("Mono8 frames") };
    assert!(pixels.as_slice().iter().any(|value| *value > 0));
    assert!(latest.run().metadata.get("device_snapshots").is_some() || latest.run().metadata.as_object().is_some_and(|record| !record.is_empty()),
            "the measurement's run record: {}", latest.run().metadata);

    // Monitor: runs until stopped; Stop drains and ends the run as stopped.
    let run = start(json!({"exposure_seconds": 0.02, "repeat": 0, "frames_per_cycle": 1}));
    let (outcome, error) = wait(&run, Some(2));
    assert_eq!(outcome, "stopped", "{error:?}");
    let monitor = signal("@logic/camera_measurement1/frames");
    assert!(!monitor.finite && monitor.sealed);
    let latest = runtime.read_latest(&monitor.generation, "@logic/camera_measurement1/frames").unwrap();
    assert!(latest.key().ordinal.unwrap() >= 5, "about 2 s of 20 ms frames: {:?}", latest.key().ordinal);

    // A processor follows the live monitor (derive over the frames bundle): every
    // event in order, each committed with its source event as parent, until the
    // source's Stop seals it.
    if catalog.find("derive").is_none() { host.shutdown(); return; }
    let camera = start(json!({"exposure_seconds": 0.02, "repeat": 0, "frames_per_cycle": 1}));
    let frames = "@logic/camera_measurement1/frames";
    let live = loop {
        std::thread::sleep(Duration::from_millis(50));
        if let Some(item) = runtime.signals().into_iter().rev().find(|item| item.signal.as_ref() == frames && !item.sealed && item.published) { break item; }
    };
    host.open("derive1", "derive", json!({}), json!({})).unwrap();
    let variables = json!([{"name": "doubled", "terms": [{"of": "frames"}, {"op": "*", "number": 2}]}]);
    let token = host.call("derive1", "query", json!({"name":"prepare_run","values":{"variables":variables},"resources":{},"devices":{},"source_signal":frames}),
        Duration::from_secs(30)).unwrap().value["prepared"].as_str().unwrap().to_owned();
    let derive = host.start_run("derive1", RunRequest { prepared: Some(token), device_labels: Default::default(), values: json!({"variables": variables}), devices: Default::default(),
        shared: Default::default(), origin: Origin::Operator, resources: json!({}), source_signal: Some(frames.to_owned()), run_directory: None,
        outputs: [("doubled".to_owned(), "@logic/derive1/doubled".to_owned())].into() }).unwrap();
    std::thread::sleep(Duration::from_secs(2));
    host.stop_run("camera_measurement1", &camera).unwrap();
    let (mut camera_done, mut derive_done) = (None, None);
    while camera_done.is_none() || derive_done.is_none() {
        match events.recv_timeout(Duration::from_secs(120)).expect("run events") {
            HostEvent::RunFinished { run, outcome, error, .. } if run == camera => camera_done = Some((outcome, error)),
            HostEvent::RunFinished { run, outcome, error, .. } if run == derive => derive_done = Some((outcome, error)),
            HostEvent::Failed { reason, .. } => panic!("plugin failed: {reason}"),
            _ => {}
        }
    }
    assert_eq!(camera_done.unwrap().0, "stopped");
    let (outcome, error) = derive_done.unwrap();
    assert_eq!(outcome, "completed", "{error:?}");
    let source = runtime.read_latest(&live.generation, frames).unwrap();
    let doubled = signal("@logic/derive1/doubled");
    let derived = runtime.read_latest(&doubled.generation, "@logic/derive1/doubled").unwrap();
    assert_eq!(derived.key().ordinal, source.key().ordinal, "the processor reached the source's last event");
    assert_eq!(derived.event().parents[0].key, *source.key());
    let AnyBuffer::U8(pixels) = source.data().unwrap().values() else { panic!("Mono8") };
    let doubled_values: Vec<f64> = match derived.data().unwrap().values() {
        AnyBuffer::F64(values) => values.as_slice().to_vec(),
        AnyBuffer::I64(values) => values.as_slice().iter().map(|value| *value as f64).collect(),
        other => panic!("derive result dtype {:?}", other.dtype()),
    };
    assert!(pixels.as_slice().iter().zip(&doubled_values).all(|(pixel, value)| f64::from(*pixel) * 2.0 == *value));
    host.shutdown();
}
