//! Real plugin processes (needs ZLC_PLUGIN_PYTHON = the science env's python.exe).
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::Duration;

use serde_json::json;
use zlc_core::AnyBuffer;
use zlc_core::control::{Origin, SafeActions};
use zlc_core::runtime::{Front, Runtime};
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
fn devices_lend_frames_nodes_publish_through_claims_and_share_one_process() {
    let Some((host, runtime, safe, events)) = host() else { return };
    let catalog = host.catalog().unwrap();
    assert!(catalog.errors.is_empty(), "{:?}", catalog.errors);
    assert!(catalog.find("test.echo").is_some() && catalog.find("counter").is_some());

    // A device lends a frame; the core borrows it, and releasing lets the plugin reuse the block.
    let opened = host.open("echo", "test.echo", json!({"gain": 2.0}), json!({})).unwrap();
    assert_eq!(opened["reads"], json!(["tunable_fields", "tunable_values", "settings_provenance", "convert_tunable_value"]), "the reads it implements");
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
    let prepare = |size| json!({"name":"prepare_run", "values":{"count":3,"size":size}, "resources":{},
        "devices":{"echo":{"instance":"echo","type_id":"test.echo","capabilities":["test.echo"],"tunable":true,"reads":opened["reads"]}}});
    let prepared = host.call("counter","query",prepare(128),Duration::from_secs(20)).unwrap().value["prepared"].as_str().unwrap().to_owned();
    host.set_node_name("counter", "Imaging camera");
    let run = host.start_run("counter", RunRequest { prepared: Some(prepared), device_labels: Default::default(), values: json!({"count": 3, "size": 128}),
        devices: [("echo".to_owned(), "echo".to_owned())].into(), shared: Default::default(), origin: Origin::Operator, resources: json!({}),
        source_signal: None, run_directory: None, outputs: [("frames".to_owned(), frames_signal.to_owned())].into(), record_devices: false }).unwrap();
    assert!(host.call("counter","query",prepare(13),Duration::from_secs(20)).err().unwrap().message.contains("unsupported test frame shape"));
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
    let frame = std::sync::Arc::downgrade(data);
    assert_eq!(data.schema().physical_shape(), [1, 1, 128, 128]);
    let AnyBuffer::U16(published) = data.values() else { panic!("dtype") };
    assert!(published.as_slice().iter().all(|value| *value == 3));
    // The device's frame went device -> node -> Runtime without a copy.
    assert_eq!(host.registry().owner_of(&block_of(data.values())).unwrap(), pid);
    assert_eq!(runtime.outcome(&signal.generation).unwrap().unwrap()["status"], "completed");
    assert_eq!(latest.run().metadata["count"], 3, "the run record is the node's own");
    assert_eq!(latest.run().metadata["node"], "counter");
    assert_eq!(latest.run().metadata["node_name"], "Imaging camera");
    host.set_node_name("counter", "Next camera name");
    assert_eq!(latest.run().metadata["node_name"], "Imaging camera", "an archived record keeps the name at declaration");
    assert_eq!(latest.run().metadata["converted"], json!([2.0, 4.0]));
    assert!((latest.run().metadata["tuned_dbm"].as_f64().unwrap() - 0.969100130080564).abs() < 1e-10);
    assert_eq!(latest.run().metadata["restored_dbm"], 0.0);
    host.spi("echo", "frame", json!([2, 1]), json!({}), Duration::from_secs(5)).expect("claim released at the end of the run");

    // A processor follows a source: on a sealed one it evaluates the current event,
    // commits it with that event as parent and the same ordinal, then completes.
    host.open("doubler", "doubler", json!({}), json!({})).unwrap();
    let doubled_signal = "@logic/doubler/doubled";
    let run = host.start_run("doubler", RunRequest { prepared: None, device_labels: Default::default(), values: json!({}), devices: Default::default(), shared: Default::default(), origin: Origin::Operator,
        resources: json!({}), source_signal: Some(frames_signal.to_owned()), run_directory: None,
        outputs: [("doubled".to_owned(), doubled_signal.to_owned())].into(), record_devices: false }).unwrap();
    let (outcome, result, error, ..) = finished(&run);
    assert_eq!((outcome.as_str(), &result), ("completed", &json!({"evaluated": 1})), "{error:?}");
    let doubled = runtime.signals().into_iter().find(|item| item.signal.as_ref() == doubled_signal).expect("processor output");
    let derived = runtime.read_latest(&doubled.generation, doubled_signal).unwrap();
    assert_eq!(derived.key().ordinal, Some(2));
    assert_eq!(derived.event().parents.iter().map(|parent| parent.key.clone()).collect::<Vec<_>>(), [latest.key().clone()]);
    assert_eq!(derived.run().metadata, json!({"source": frames_signal, "source_count": 3, "node": "doubler", "node_name": "doubler"}));
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
    let run = host.start_run("counter", RunRequest { prepared: None, device_labels: Default::default(), values: json!({"count": 1, "size": 128}),
        devices: [("echo".to_owned(), "echo".to_owned())].into(), shared: Default::default(), origin: Origin::Operator, resources: json!({}),
        source_signal: None, run_directory: None, outputs: [("frames".to_owned(), frames_signal.to_owned())].into(), record_devices: false }).unwrap();
    let (outcome, _, error, ..) = finished(&run);
    assert_eq!(outcome, "completed", "{error:?}");
    assert_eq!(host.registry().holds(&lent), 1);

    // A following processor reads its source's next generation from its first commit,
    // however late its own run starts: here, after that source has sealed.
    host.follow("doubler", Some(frames_signal));
    let run = host.start_run("counter", RunRequest { prepared: None, device_labels: Default::default(), values: json!({"count": 3, "size": 128}),
        devices: [("echo".to_owned(), "echo".to_owned())].into(), shared: Default::default(), origin: Origin::Operator, resources: json!({}),
        source_signal: None, run_directory: None, outputs: [("frames".to_owned(), frames_signal.to_owned())].into(), record_devices: false }).unwrap();
    assert_eq!(finished(&run).0, "completed");
    let run = host.start_run("doubler", RunRequest { prepared: None, device_labels: Default::default(), values: json!({}), devices: Default::default(), shared: Default::default(), origin: Origin::Operator,
        resources: json!({}), source_signal: Some(frames_signal.to_owned()), run_directory: None,
        outputs: [("doubled".to_owned(), doubled_signal.to_owned())].into(), record_devices: false }).unwrap();
    let (outcome, result, error, ..) = finished(&run);
    assert_eq!((outcome.as_str(), &result), ("completed", &json!({"evaluated": 3})), "{error:?}");
    // Its block went back to its owner; in the one plugin process the doubler's next
    // output may already be in it.
    assert!(frame.upgrade().is_none());
    host.follow("doubler", None);

    // Every instance, devices and nodes, is in the one plugin process.
    let pids: std::collections::BTreeSet<u32> = host.instances().into_iter().map(|(.., pid)| pid).collect();
    assert_eq!(pids.into_iter().collect::<Vec<_>>(), [pid]);
    let failure = || loop {
        if let HostEvent::Failed { instances, reason } = events.recv_timeout(Duration::from_secs(15)).expect("failure event") { break (instances, reason); }
    };
    // A call that overruns its deadline fails its instance only: it leaves the process,
    // the others carry on, and it opens again (its stuck call ends on its own).
    host.open("slow", "test.echo", json!({}), json!({})).unwrap();
    let overran = host.spi("slow", "sleep", json!([10]), json!({}), Duration::from_millis(200)).err().unwrap();
    assert_eq!(overran.kind, ErrorKind::Timeout, "{overran}");
    assert_eq!(failure().0, ["slow"]);
    host.spi("echo", "frame", json!([2, 1]), json!({}), Duration::from_secs(5)).unwrap();
    let until = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        match host.open("slow", "test.echo", json!({}), json!({})) {
            Ok(_) => break,
            Err(error) => { assert!(error.message.contains("still releasing"), "{error}"); assert!(std::time::Instant::now() < until); std::thread::sleep(Duration::from_millis(50)); }
        }
    }
    host.spi("slow", "frame", json!([2, 1]), json!({}), Duration::from_secs(5)).unwrap();
    // A crash ends the process: every instance in it fails (v2: the device's error
    // state), and each opens again in a new one.
    host.open("boom", "test.echo", json!({}), json!({})).unwrap();
    let crashed = host.spi("boom", "crash", json!([]), json!({}), Duration::from_secs(10)).err().unwrap();
    assert_eq!(crashed.kind, ErrorKind::Lost, "{crashed}");
    let failed = failure();
    assert!(["boom", "echo", "slow"].iter().all(|name| failed.0.iter().any(|lost| lost == name)), "{:?}", failed.0);
    assert!(failed.1.contains("(exit code 3)"), "{}", failed.1);
    assert!(host.instances().is_empty());
    host.open("echo", "test.echo", json!({"gain": 2.0}), json!({})).unwrap();
    assert_ne!(host.instances()[0].2, pid);

    assert_eq!(host.spi("echo", "made_safe", json!([]), json!({}), Duration::from_secs(5)).unwrap().value, json!([]));
    // Stop all drives the open device's output safe through its plugin.
    assert_eq!(safe.outputs(), ["echo/beam"]);
    assert!(safe.make_safe().is_empty());
    assert_eq!(host.spi("echo", "made_safe", json!([]), json!({}), Duration::from_secs(5)).unwrap().value, json!(["beam"]));
    host.shutdown();
    assert!(host.instances().is_empty());
}

#[test]
fn a_run_follows_only_an_exact_input_and_what_ends_early_leaves_with_its_signals() {
    let Some((host, runtime, _safe, events)) = host() else { return };
    let finished = |run: &str| loop {
        match events.recv_timeout(Duration::from_secs(60)).expect("run events") {
            HostEvent::RunFinished { run: id, outcome, error, .. } if id == run => break (outcome, error),
            HostEvent::Failed { reason, .. } => panic!("plugin failed: {reason}"),
            _ => {}
        }
    };
    let request = |values, source: Option<&str>, output: (&str, &str)| RunRequest { prepared: None, device_labels: Default::default(), values, devices: Default::default(), shared: Default::default(), origin: Origin::Operator,
        resources: json!({}), source_signal: source.map(str::to_owned), run_directory: None, outputs: [(output.0.to_owned(), output.1.to_owned())].into(), record_devices: false };
    let published = |signal: &str| {
        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        loop {
            if let Some(item) = runtime.signals().into_iter().find(|item| &*item.signal == signal && item.published) { break item; }
            assert!(std::time::Instant::now() < deadline, "{signal} was not published");
            std::thread::sleep(Duration::from_millis(20));
        }
    };
    let present = |signal: &str| runtime.signals().iter().any(|item| &*item.signal == signal);
    let holds_become = |block: &str, holds: usize| {
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while host.registry().holds(block) != holds && std::time::Instant::now() < deadline { std::thread::sleep(Duration::from_millis(10)); }
        host.registry().holds(block) == holds
    };

    // A source that published one frame and runs on.
    let frames = "@logic/holder/frames";
    host.open("holder", "holder", json!({}), json!({})).unwrap();
    let holder = host.start_run("holder", request(json!({}), None, ("frames", frames))).unwrap();
    let source = published(frames);
    let lent = block_of(runtime.read_latest(&source.generation, frames).unwrap().data().unwrap().values());
    let held = host.registry().holds(&lent);

    // Over the 'latest' input its values enable, a run reads its source by name (v2:
    // no source route): nothing is delivered to it, and its reading, which names no
    // parent, is no member of the source's shot.
    host.open("reader", "reader", json!({}), json!({})).unwrap();
    let reading = "@logic/reader/reading";
    let run = host.start_run("reader", request(json!({"mode": "latest"}), Some(frames), ("reading", reading))).unwrap();
    let own = published(reading);
    let shown = runtime.read_coherent_views(&[(source.generation.clone(), frames.into(), None), (own.generation.clone(), reading.into(), None)],
        &mut Front::default());
    assert!(shown.iter().all(Result::is_ok), "{:?}", shown.iter().map(|read| read.as_ref().err()).collect::<Vec<_>>());
    assert_eq!(host.registry().holds(&lent), held, "no delivery of the frame");
    assert_eq!(finished(&run).0, "completed");

    // Over its 'exact' input it follows the source; the delivery it never takes is
    // answered at its end (stale_run) and lets the frame go: nothing waits on a run
    // that has ended, and how long a run leaves its input says nothing of its process.
    let run = host.start_run("reader", request(json!({"mode": "exact"}), Some(frames), ("reading", reading))).unwrap();
    assert!(holds_become(&lent, held + 1), "the frame is delivered");
    assert_eq!(finished(&run).0, "completed");
    assert!(holds_become(&lent, held), "the untaken delivery let the frame go");

    // A closed node (removed, or its layout replaced) takes its signals with it (v2 retire).
    host.close("reader").unwrap();
    assert!(!present(reading));

    // A processor stopped while its source runs is withdrawn (v2): its signals go.
    host.open("doubler", "doubler", json!({}), json!({})).unwrap();
    let doubled = "@logic/doubler/doubled";
    let run = host.start_run("doubler", request(json!({}), Some(frames), ("doubled", doubled))).unwrap();
    published(doubled);
    host.stop_run("doubler", &run).unwrap();
    assert_eq!(finished(&run).0, "stopped");
    assert!(!present(doubled) && present(frames));
    host.stop_run("holder", &holder).unwrap();
    assert_eq!(finished(&holder).0, "stopped");
    host.shutdown();
}


#[test]
fn sdk_lifecycle_acknowledgements_and_cleanup_keep_exclusive_ownership() {
    let Some(python) = std::env::var_os("ZLC_PLUGIN_PYTHON") else { return };
    let tests = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests");
    let output = std::process::Command::new(python).args(["-m", "unittest", "discover", "-s"])
        .arg(tests).args(["-p", "python*py", "-v"]).output().unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
}


#[test]
fn a_failed_device_close_retains_ownership_until_cleanup_succeeds() {
    let Some((host, _runtime, _safe, _events)) = host() else { return };
    host.open("keeper", "test.echo", json!({}), json!({})).unwrap();
    for name in ["n100_com3", "pylon_camera"] {
        host.open(name, "test.echo", json!({}), json!({})).unwrap();
        host.spi(name, "fail_close_once", json!([]), json!({}), Duration::from_secs(5)).unwrap();
        assert!(host.close(name).unwrap_err().message.contains("still held"));
        assert!(host.instances().iter().any(|entry| entry.0 == name));
        assert_eq!(host.open(name, "test.echo", json!({}), json!({})).unwrap_err().kind, ErrorKind::Refused);
        host.close(name).unwrap();
        for _ in 0..20 {
            host.open(name, "test.echo", json!({}), json!({})).unwrap();
            host.close(name).unwrap();
        }
    }
    host.shutdown();
}


#[test]
fn close_releases_a_handle_left_by_a_failed_init_before_host_admission() {
    let Some((host, _runtime, _safe, _events)) = host() else { return };
    host.open("keeper", "test.echo", json!({}), json!({})).unwrap();
    let marker = std::env::temp_dir().join(format!("zlc-init-cleanup-{}.txt",uuid::Uuid::new_v4()));
    let error=host.open("bad", "test.failed_init", json!({"path":marker}), json!({})).unwrap_err();
    assert!(error.message.contains("verification failed"));
    assert!(!marker.exists());
    assert!(!host.instances().iter().any(|entry|entry.0=="bad"));
    host.close("bad").unwrap();
    assert_eq!(std::fs::read_to_string(&marker).unwrap(),"released");
    std::fs::remove_file(marker).unwrap();
    host.open("bad", "test.echo", json!({}), json!({})).unwrap();
    host.shutdown();
}


#[test]
fn a_start_records_the_bench_its_devices_report_and_a_busy_one_is_not_lost() {
    let Some((host, runtime, _safe, events)) = host() else { return };
    let snapshot = |instance: &str, wait_ms: u64| host.snapshot(instance, Duration::from_millis(wait_ms));
    // A tunable device answers its quartet's reads; one with a snapshot of its own answers that;
    // any other does not report its settings. Its open answer says which.
    for (instance, type_id, reports) in [("echo", "test.echo", true), ("pattern", "test.pattern", true), ("plain", "test.plain", false)] {
        assert_eq!(host.open(instance, type_id, json!({}), json!({})).unwrap()["snapshot"], reports, "{instance}");
    }
    assert_eq!(snapshot("echo", 5000).unwrap(), json!({"values": {"amplitude": 0.0}, "units": {"amplitude": "dBm"}, "provenance": {"load": 100}}));
    assert_eq!(snapshot("pattern", 5000).unwrap(), json!({"pattern": "flat", "amplitude": 0.0}));
    assert_eq!(snapshot("plain", 5000).unwrap(), json!(null));
    // A device busy in another call misses the bound, which says nothing of its health: it stays
    // open past the grace a call's missed deadline has (5 s), and answers once its lane is free.
    host.open("slow", "test.echo", json!({}), json!({})).unwrap();
    let sleeping = { let host = host.clone(); std::thread::spawn(move || host.spi("slow", "sleep", json!([8]), json!({}), Duration::from_secs(30))) };
    std::thread::sleep(Duration::from_millis(300));
    assert_eq!(snapshot("slow", 200).unwrap_err().kind, ErrorKind::Timeout);
    sleeping.join().unwrap().unwrap();
    assert!(events.try_iter().all(|event| !matches!(event, HostEvent::Failed { .. })), "a missed snapshot never fails its device");
    assert_eq!(snapshot("slow", 5000).unwrap()["values"]["amplitude"], 0.0);

    // A configured observation is installed before publication; a plugin's own
    // key of that name is never shown as the core's, recorded or not.
    host.open("counter", "counter", json!({}), json!({})).unwrap();
    let frames = "@logic/counter/frames";
    let bench = json!({"taken_at": "2026-10-25T12:00:00.000+00:00", "devices": {"echo": {"type_id": "test.echo", "role": "echo", "values": {"amplitude": 0.0}}},
                       "sequencer": {}});
    let start = |record_devices: bool| host.start_run("counter", RunRequest { prepared: None, device_labels: Default::default(),
        values: json!({"count": 1, "size": 128}), devices: [("echo".to_owned(), "echo".to_owned())].into(), shared: Default::default(), origin: Origin::Operator,
        resources: json!({}), source_signal: None, run_directory: None, outputs: [("frames".to_owned(), frames.to_owned())].into(), record_devices }).unwrap();
    let recorded = |run: String| {
        loop {
            match events.recv_timeout(Duration::from_secs(60)).expect("run events") {
                HostEvent::Request { instance, run, ticket, op, params, .. } if op == "run.snapshot" => {
                    assert_eq!(params["wait"], false, "recorded at the explicit pre-acquisition boundary");
                    host.record_devices(&instance, &run, bench.clone()).unwrap();
                    host.respond(ticket, Ok(bench.clone()));
                }
                HostEvent::Request { ticket, .. } => host.respond(ticket, Ok(json!({}))),
                HostEvent::RunFinished { run: id, outcome, error, .. } if id == run => { assert_eq!(outcome, "completed", "{error:?}"); break; }
                HostEvent::Failed { reason, .. } => panic!("plugin failed: {reason}"),
                _ => {}
            }
        }
        let signal = runtime.signals().into_iter().rev().find(|item| item.signal.as_ref() == frames).unwrap();
        runtime.read_latest(&signal.generation, frames).unwrap().run().metadata.clone()
    };
    assert_eq!(recorded(start(true))["apparatus"], bench);
    assert!(recorded(start(false)).get("apparatus").is_none());
    host.shutdown();
}

/// A type names the resources its rows bind (and that it finds a blank one: `resolve`); its scan is
/// handed the bound ones and touches none, a scan written before them still answers, and a row left
/// without its port is found on a free one before it opens -- or says why not.
#[test]
fn scans_and_resolves_are_handed_the_bound_resources() {
    let Some((host, _runtime, _safe, _events)) = host() else { return };
    let catalog = host.catalog().unwrap();
    let wired = catalog.find("test.wired").unwrap();
    assert_eq!((wired.resources.clone(), wired.queries.clone()), (vec!["serial:{port}".to_owned()], vec!["discover".to_owned(), "resolve".to_owned()]));
    assert!(catalog.find("test.echo").unwrap().resources.is_empty() && catalog.find("test.scanned").unwrap().queries == ["discover"]);
    let wait = Duration::from_secs(20);
    let scanned = host.discover("test.wired", &["serial:COM3".to_owned()], wait).unwrap();
    assert_eq!(scanned["notes"], json!(["excluded serial:COM3"]));
    let ports: Vec<&serde_json::Value> = scanned["found"].as_array().unwrap().iter().map(|found| &found["parameters"]["port"]).collect();
    assert_eq!(ports, [&json!("COM4"), &json!("COM5")]);
    let old = host.discover("test.scanned", &["serial:COM3".to_owned()], wait).unwrap();
    assert_eq!((old["found"][0]["instance_id"].as_str(), &old["notes"]), (Some("scanned"), &json!(["scanned as before"])));
    let pinned = host.resolve("test.wired", json!({"port": ""}), &["serial:COM3".to_owned(), "serial:COM4".to_owned()], wait).unwrap();
    assert_eq!(pinned, json!({"parameters": {"port": "COM5"}, "note": "found on COM5 (given '')"}));
    let all: Vec<String> = ["COM3", "COM4", "COM5"].iter().map(|port| format!("serial:{port}")).collect();
    let refused = host.resolve("test.wired", json!({}), &all, wait).unwrap_err();
    assert_eq!(refused.to_string(), "RuntimeError: no free wire (tried none; passed over COM3, COM4, COM5: bound)");
    assert_eq!(host.resolve("test.echo", json!({}), &[], wait).unwrap_err().kind, ErrorKind::UnknownOp);
    host.shutdown();
}
