//! Real plugin processes (needs ZLC_PLUGIN_PYTHON = the science env's python.exe).
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::Duration;

use serde_json::json;
use zlc_core::AnyBuffer;
use zlc_core::control::{Origin, SafeActions};
use zlc_core::runtime::Runtime;
use zlc_plugin_host::{ErrorKind, HostConfig, HostEvent, Lend, PluginHost, RunRequest};

fn host() -> Option<(PluginHost, Runtime, SafeActions, mpsc::Receiver<HostEvent>)> {
    let Some(python) = std::env::var_os("ZLC_PLUGIN_PYTHON") else {
        eprintln!("ZLC_PLUGIN_PYTHON is not set: skipping the plugin process test");
        return None;
    };
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let config = HostConfig { python: python.into(), python_path: vec![root.join("../../science/src"), root.join("tests/plugins")],
        packages: vec!["zlc_test_plugins".into()], memory_limit_mib: Some(3072) };
    let (runtime, safe) = (Runtime::new(), SafeActions::new());
    let (sender, events) = mpsc::channel();
    let sender = std::sync::Mutex::new(sender);
    let host = PluginHost::new(config, runtime.clone(), safe.clone(), move |event| { let _ = sender.lock().unwrap().send(event); }).unwrap();
    Some((host, runtime, safe, events))
}

fn block_of(values: &AnyBuffer) -> String {
    let (owner, _) = values.foreign().expect("borrowed from a block, not copied");
    owner.as_any().downcast_ref::<Lend>().expect("a plugin block").name().to_owned()
}

#[test]
fn devices_lend_frames_nodes_publish_through_claims_and_a_crash_loses_one_instance() {
    let Some((host, runtime, safe, events)) = host() else { return };
    let catalog = host.catalog().unwrap();
    assert!(catalog.errors.is_empty(), "{:?}", catalog.errors);
    assert!(catalog.find("test.echo").is_some() && catalog.find("counter").is_some());

    // A device lends a frame; the core borrows it, and releasing lets the plugin reuse the block.
    let opened = host.open("echo", "test.echo", json!({"gain": 2.0}), json!({})).unwrap();
    assert_eq!(opened["reads"], json!(["convert_tunable_value"]), "the reads it implements");
    let pid = host.instances().into_iter().find(|(name, ..)| name == "echo").unwrap().2;
    let reply = host.spi("echo", "frame", json!([256, 7]), json!({}), Duration::from_secs(10)).unwrap();
    let frame = reply.array(&reply.value).unwrap();
    assert_eq!(frame.shape, vec![256, 256]);
    let AnyBuffer::U16(values) = &frame.values else { panic!("dtype") };
    assert!(values.as_slice().iter().all(|value| *value == 7));
    let first = block_of(&frame.values);
    assert_eq!(host.registry().owner_of(&first).unwrap(), pid);
    drop((frame, reply));
    assert_eq!(host.registry().holds(&first), 0);
    std::thread::sleep(Duration::from_millis(200));
    let again = host.spi("echo", "frame", json!([256, 8]), json!({}), Duration::from_secs(10)).unwrap();
    assert_eq!(block_of(&again.array(&again.value).unwrap().values), first, "a released block is recycled");
    drop(again);
    assert_eq!(host.spi("echo", "get:opened_with", json!([]), json!({}), Duration::from_secs(5)).unwrap().value, json!({"gain": 2.0}));

    // Plugin commands are catalogue entries: validated, answered by the lane.
    let commands = host.commands();
    assert_eq!(commands.iter().map(|command| command.name.as_str()).collect::<Vec<_>>(), ["echo__frame_total", "echo__tell", "echo__keep", "echo__take"]);
    // A command's file is this computer's: the host writes what the device gave, and hands it what it reads.
    let file = std::env::temp_dir().join(format!("zlc-host-files-{}", std::process::id())).join("kept.txt");
    assert_eq!(host.command("echo__keep", &json!({"path": file, "text": "kept here"})).unwrap().value, json!(9));
    assert_eq!(std::fs::read_to_string(&file).unwrap(), "kept here");
    assert_eq!(host.command("echo__take", &json!({"path": file})).unwrap().value, json!("kept here"));
    let _ = std::fs::remove_dir_all(file.parent().unwrap());
    assert_eq!(host.command("echo__frame_total", &json!({"size": 4})).unwrap().value, json!(32));
    assert_eq!(host.command("echo__frame_total", &json!({"size": 0})).err().unwrap().kind, ErrorKind::Invalid);
    assert_eq!(host.command("echo__frame_total", &json!({"size": 4, "colour": 1})).err().unwrap().kind, ErrorKind::Invalid);

    // A draft query reads the device its draft names -- a read it lists -- and changes nothing.
    host.open("counter", "counter", json!({}), json!({})).unwrap();
    let echo = json!({"instance": "echo", "type_id": "test.echo", "capabilities": ["test.echo"], "tunable": false, "reads": opened["reads"]});
    let facts = host.call("counter", "query", json!({"name": "editor_facts", "values": {}, "resources": {}, "devices": {"echo": echo},
        "source": null, "bundle": [], "artifacts": null}), Duration::from_secs(20)).unwrap().value;
    assert_eq!(facts["converted"], json!([2.0, 4.0]), "{facts}");
    assert!(facts["refused"].as_str().unwrap().contains("fire is not a read"), "{facts}");

    // A node run claims its device, calls it through a proxy and publishes into Runtime
    // under the signal name its request gives; it asks the application through `core`.
    let frames_signal = "@logic/counter/frames";
    let run = host.start_run("counter", RunRequest { values: json!({"count": 3, "size": 128}),
        devices: [("echo".to_owned(), "echo".to_owned())].into(), origin: Origin::Operator, resources: json!({}),
        source_signal: None, run_directory: None, outputs: [("frames".to_owned(), frames_signal.to_owned())].into() }).unwrap();
    assert_eq!(host.spi("echo", "frame", json!([2, 1]), json!({}), Duration::from_secs(5)).err().unwrap().kind, ErrorKind::NotClaimed);
    let finished = |run: &str| {
        let (mut ready, mut progress) = (false, 0);
        loop {
            match events.recv_timeout(Duration::from_secs(60)).expect("run events") {
                HostEvent::RunReady { run: id, .. } if id == run => ready = true,
                HostEvent::RunProgress { run: id, .. } if id == run => progress += 1,
                HostEvent::Request { run: id, ticket, op, params, origin, .. } if id == run => {
                    assert_eq!((op.as_str(), &params["command"], &origin), ("core", &json!("probe"), &Origin::Operator));
                    host.respond(ticket, Ok(json!({"pong": params["params"]["value"]})));
                }
                HostEvent::RunFinished { run: id, outcome, result, error, .. } if id == run => break (outcome, result, error, ready, progress),
                HostEvent::Failed { reason, .. } => panic!("plugin failed: {reason}"),
                _ => {}
            }
        }
    };
    let (outcome, result, error, ready, progress) = finished(&run);
    assert_eq!(outcome, "completed", "{error:?}");
    assert_eq!(result, json!({"published": 3, "device_total": 3 * 128 * 128, "core": {"pong": 3}}));
    assert!(ready && progress == 3);
    let signal = runtime.signals().into_iter().find(|item| item.signal.as_ref() == frames_signal).expect("published signal");
    assert!(signal.sealed);
    let latest = runtime.read_latest(&signal.generation, frames_signal).unwrap();
    assert_eq!(latest.key().ordinal, Some(2));
    assert_eq!(latest.event().metadata, json!({"records": {"frames": {"index": 2}}}));
    let data = latest.data().unwrap();
    assert_eq!(data.schema().physical_shape(), [1, 1, 128, 128]);
    let AnyBuffer::U16(published) = data.values() else { panic!("dtype") };
    assert!(published.as_slice().iter().all(|value| *value == 3));
    // The device's frame went device -> node -> Runtime without a copy.
    assert_eq!(host.registry().owner_of(&block_of(data.values())).unwrap(), pid);
    assert_eq!(runtime.outcome(&signal.generation).unwrap().unwrap()["status"], "completed");
    assert_eq!(latest.run().metadata["count"], 3, "the run record is the node's own");
    host.spi("echo", "frame", json!([2, 1]), json!({}), Duration::from_secs(5)).expect("claim released at the end of the run");

    // A processor follows a source: on a sealed one it evaluates the current event,
    // commits it with that event as parent and the same ordinal, then completes.
    host.open("doubler", "doubler", json!({}), json!({})).unwrap();
    let doubled_signal = "@logic/doubler/doubled";
    let run = host.start_run("doubler", RunRequest { values: json!({}), devices: Default::default(), origin: Origin::Operator,
        resources: json!({}), source_signal: Some(frames_signal.to_owned()), run_directory: None,
        outputs: [("doubled".to_owned(), doubled_signal.to_owned())].into() }).unwrap();
    let (outcome, result, error, ..) = finished(&run);
    assert_eq!((outcome.as_str(), &result), ("completed", &json!({"evaluated": 1})), "{error:?}");
    let doubled = runtime.signals().into_iter().find(|item| item.signal.as_ref() == doubled_signal).expect("processor output");
    let derived = runtime.read_latest(&doubled.generation, doubled_signal).unwrap();
    assert_eq!(derived.key().ordinal, Some(2));
    assert_eq!(derived.event().parents.iter().map(|parent| parent.key.clone()).collect::<Vec<_>>(), [latest.key().clone()]);
    assert_eq!(derived.run().metadata, json!({"source": frames_signal, "source_count": 3}));
    assert_eq!(derived.event().metadata, json!({"records": {"doubled": {"from": {"index": 2}}}}));
    let AnyBuffer::U32(values) = derived.data().unwrap().values() else { panic!("dtype") };
    assert!(values.as_slice().iter().all(|value| *value == 6));

    // The next run of a node drops its previous generation inside run.begin; when Runtime
    // holds the last of a lent frame, its release reaches the owner from there. The
    // doubler's result keeps the frame it judged (its same-shot picture) as long as it is
    // the doubler's latest: the camera's next run leaves it, the doubler's next run drops it.
    let lent = block_of(data.values());
    drop((latest, derived));
    // The doubler's lease of it is released by its plugin, asynchronously.
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while host.registry().holds(&lent) > 1 && std::time::Instant::now() < deadline { std::thread::sleep(Duration::from_millis(20)); }
    assert_eq!(host.registry().holds(&lent), 1, "only Runtime holds the last frame");
    let run = host.start_run("counter", RunRequest { values: json!({"count": 1, "size": 128}),
        devices: [("echo".to_owned(), "echo".to_owned())].into(), origin: Origin::Operator, resources: json!({}),
        source_signal: None, run_directory: None, outputs: [("frames".to_owned(), frames_signal.to_owned())].into() }).unwrap();
    let (outcome, _, error, ..) = finished(&run);
    assert_eq!(outcome, "completed", "{error:?}");
    assert_eq!(host.registry().holds(&lent), 1);

    // A following processor reads its source's next generation from its first commit,
    // however late its own run starts: here, after that source has sealed.
    host.follow("doubler", Some(frames_signal));
    let run = host.start_run("counter", RunRequest { values: json!({"count": 3, "size": 128}),
        devices: [("echo".to_owned(), "echo".to_owned())].into(), origin: Origin::Operator, resources: json!({}),
        source_signal: None, run_directory: None, outputs: [("frames".to_owned(), frames_signal.to_owned())].into() }).unwrap();
    assert_eq!(finished(&run).0, "completed");
    let run = host.start_run("doubler", RunRequest { values: json!({}), devices: Default::default(), origin: Origin::Operator,
        resources: json!({}), source_signal: Some(frames_signal.to_owned()), run_directory: None,
        outputs: [("doubled".to_owned(), doubled_signal.to_owned())].into() }).unwrap();
    let (outcome, result, error, ..) = finished(&run);
    assert_eq!((outcome.as_str(), &result), ("completed", &json!({"evaluated": 3})), "{error:?}");
    assert_eq!(host.registry().holds(&lent), 0);
    host.follow("doubler", None);

    // A crash loses that instance only (v2: the device's error state); nothing else is touched.
    host.open("boom", "test.echo", json!({}), json!({})).unwrap();
    let crashed = host.spi("boom", "crash", json!([]), json!({}), Duration::from_secs(10)).err().unwrap();
    assert_eq!(crashed.kind, ErrorKind::Lost, "{crashed}");
    let failed = loop {
        if let HostEvent::Failed { instances, reason } = events.recv_timeout(Duration::from_secs(10)).expect("failure event") { break (instances, reason); }
    };
    assert_eq!(failed.0, ["boom"]);
    assert!(failed.1.contains("(exit code 3)"), "{}", failed.1);
    assert!(host.instances().iter().all(|(name, ..)| name != "boom"));
    // The other device is untouched.
    host.spi("echo", "frame", json!([2, 1]), json!({}), Duration::from_secs(5)).unwrap();

    assert_eq!(host.spi("echo", "made_safe", json!([]), json!({}), Duration::from_secs(5)).unwrap().value, json!([]));
    // Stop all drives the open device's output safe through its plugin.
    assert_eq!(safe.outputs(), ["echo/beam"]);
    assert!(safe.make_safe().is_empty());
    assert_eq!(host.spi("echo", "made_safe", json!([]), json!({}), Duration::from_secs(5)).unwrap().value, json!(["beam"]));
    host.shutdown();
    assert!(host.instances().is_empty());
}
