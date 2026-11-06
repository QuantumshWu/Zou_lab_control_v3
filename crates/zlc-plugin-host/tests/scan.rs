//! Native scan service through a real plugin process; source events stay in Runtime.
//! Needs ZLC_PLUGIN_PYTHON, as the other plugin-process integration tests do.
use std::path::PathBuf;
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use zlc_core::{DType, OwnedBuffer};
use zlc_core::control::{Origin, SafeActions};
use zlc_core::dataset::{DataBlock, Validity};
use zlc_core::runtime::{Bundle, Declaration, Generation, Input, Publication, Read, Runtime, Storage};
use zlc_core::schema::{Axis, Coordinate, Coordinates, DatasetSchema, Domain, DomainRole, ValidityContract, ValueSchema};
use zlc_plugin_host::{HostConfig, HostEvent, PluginHost, RunRequest};

const SOURCE: &str = "@test/camera/frames";
const OUTPUT: &str = "@test/native_scan/scan";

struct Harness { host: PluginHost, runtime: Runtime, events: mpsc::Receiver<HostEvent> }

impl Drop for Harness { fn drop(&mut self) { self.host.shutdown(); } }

fn harness() -> Option<Harness> {
    let Some(python) = std::env::var_os("ZLC_PLUGIN_PYTHON") else {
        eprintln!("ZLC_PLUGIN_PYTHON is not set: skipping native scan process tests");
        return None;
    };
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let runtime = Runtime::new();
    let (sender, events) = mpsc::channel();
    let sender = Mutex::new(sender);
    let host = PluginHost::new(HostConfig {
        python: python.into(), python_path: vec![root.join("../../science/src"), root.join("tests/plugins")],
        packages: vec!["zlc_test_plugins".into()], memory_limit_mib: Some(3072),
    }, runtime.clone(), SafeActions::new(), move |event| { let _ = sender.lock().unwrap().send(event); }).unwrap();
    host.open("native", "native_scan", json!({}), json!({})).unwrap();
    Some(Harness { host, runtime, events })
}

fn sample_schema(points: usize, origin: i128) -> Arc<DatasetSchema> {
    let singleton = Domain::new(vec![1], vec![], vec![]).unwrap();
    let axis = |id: &str, len, origin| Axis::new(id.into(), id.into(), "coordinate".into(),
        Coordinates::Index { origin, len }, None, None, None, None).unwrap();
    let point = Domain::dense(vec![points], vec![axis("source.sample", points, origin)]).unwrap();
    let cell = Domain::dense(vec![2], vec![axis("source.site", 2, 0)]).unwrap();
    DatasetSchema::new(singleton, point, cell,
        ValueSchema::new(DType::U16, Some("count".into()), None, ValidityContract::Value).unwrap()).unwrap()
}

fn source(runtime: &Runtime, schema: Arc<DatasetSchema>, storage: Storage) -> Generation {
    // A direct Runtime generation must exist before the plugin subscribes to it.
    runtime.begin("camera", vec![Declaration { signal: SOURCE.into(), schema, storage }], json!({"fixture": "source"})).unwrap()
}

fn publish(runtime: &Runtime, source: &Generation, schema: Arc<DatasetSchema>, ordinal: u64, origin: Option<[usize; 2]>)
    -> Arc<Publication> {
    let count = schema.value_len();
    let data = DataBlock::new(schema, OwnedBuffer::from_vec(vec![ordinal as u16; count]).into(),
        Validity::Uniform(ordinal % 2 == 0), Some(OwnedBuffer::from_vec(vec![0.25; count]))).unwrap();
    runtime.commit(source, &Bundle {
        inputs: vec![Input::Event { signal: SOURCE.into(), ordinal, data, origin, shot_time: Some(ordinal as f64 * 0.01) }],
        metadata: json!({"records": {"frames": {"shot": ordinal, "actual_devices": {"camera": {"exposure": 0.01}}}}}),
        parents: vec![],
    }).unwrap().publications[0].clone()
}

fn spec() -> Value {
    json!({"axes": [{"port": "pulse:param:dwell", "identity": "dwell", "label": "Dwell", "unit": "us",
        "values": [1.0, 2.0, 3.0], "coordinate_of": null, "host_advanced": false}], "scan_repeats": 2, "run_repeats": 2})
}

fn start(h: &Harness, mode: &str, configure_first: bool) -> String {
    start_plan(h, spec(), mode, configure_first, "[[1.25],[2.25],[3.25]]")
}

fn start_plan(h: &Harness, spec: Value, mode: &str, configure_first: bool, effective_inner: &str) -> String {
    h.host.start_run("native", RunRequest {
        prepared: None, device_labels: Default::default(), devices: Default::default(), shared: Default::default(),
        values: json!({"spec": spec.to_string(), "mode": mode, "configure_first": configure_first,
            "effective_inner": effective_inner}),
        origin: Origin::Operator, resources: json!({}), source_signal: Some(SOURCE.into()), run_directory: None,
        outputs: [("scan".into(), OUTPUT.into())].into(), record_devices: false,
    }).unwrap()
}

fn ready(h: &Harness, run: &str) {
    loop {
        match h.events.recv_timeout(Duration::from_secs(20)).expect("scan readiness") {
            HostEvent::RunReady { run: id, .. } if id == run => return,
            HostEvent::RunFinished { run: id, outcome, error, .. } if id == run => panic!("scan ended before ready: {outcome}: {error:?}"),
            HostEvent::Failed { reason, .. } => panic!("plugin failed: {reason}"),
            _ => {},
        }
    }
}

fn finished(h: &Harness, run: &str) -> (String, Value, String) {
    loop {
        match h.events.recv_timeout(Duration::from_secs(20)).expect("scan completion") {
            HostEvent::RunFinished { run: id, outcome, result, error, .. } if id == run =>
                return (outcome, result, error.map(|error| error.to_string()).unwrap_or_default()),
            HostEvent::Failed { reason, .. } => panic!("plugin failed: {reason}"),
            _ => {},
        }
    }
}

#[test]
fn native_scan_places_repeats_and_retains_original_data_and_lineage_without_python_deliveries() {
    for configure_first in [true, false] {
        let Some(h) = harness() else { return };
        let schema = sample_schema(2, 0);
        let source = source(&h.runtime, schema.clone(), Storage::Latest);
        let run = start(&h, "collect", configure_first);
        ready(&h, &run);
        let parents: Vec<_> = (40..52).map(|ordinal| publish(&h.runtime, &source, schema.clone(), ordinal, None)).collect();
        let (outcome, result, error) = finished(&h, &run);
        assert_eq!(outcome, "completed", "{error}");
        assert_eq!(result["written"], 12);
        assert_eq!(result["rows"]["inner"], json!([[1.0], [2.0], [3.0]]));
        assert_eq!(result["rows"]["outer"], json!([[]]));
        assert_eq!(result["batches"].as_array().unwrap().iter().map(|n| n.as_u64().unwrap()).sum::<u64>(), 12);
        let signal = h.runtime.signals().into_iter().find(|signal| &*signal.signal == OUTPUT).unwrap();
        assert!(signal.finite && signal.sealed);
        let latest = h.runtime.read_latest(&signal.generation, OUTPUT).unwrap();
        let stored = latest.canonical().unwrap();
        assert_eq!((stored.written_cells(), stored.total_cells()), (24, 24));
        assert_eq!(stored.schema().domain(DomainRole::Repeat).size(), 4);
        assert_eq!(stored.schema().domain(DomainRole::Point).size(), 6);
        assert!(stored.schema().domain(DomainRole::Cell) == schema.domain(DomainRole::Cell));
        let axis = stored.schema().domain(DomainRole::Point).axis("scan.dwell").unwrap();
        assert_eq!(axis.coordinate(0), Some(Coordinate::Real(1.25)));
        for (index, (segment, parent)) in stored.segments().zip(&parents).enumerate() {
            let sweep = index / 6;
            let row = index % 6 / 2;
            let shot = index % 2;
            assert_eq!(segment.origin(), [sweep * 2 + shot, row * 2]);
            assert_eq!(segment.extent(), [1, 2]);
            assert!(Arc::ptr_eq(segment.data(), parent.data().unwrap()), "shot {index} was copied or wrapped");
        }
        assert_eq!(latest.key().ordinal, Some(51));
        assert_eq!(latest.event().parents[0].key, *parents[11].key());
        assert_eq!(latest.event().metadata["records"]["scan"]["shot"], 51);
        assert_eq!(latest.event().metadata["records"]["scan"]["actual_devices"]["rf"]["values"]["frequency"], 80.25);
        assert_eq!(latest.event().metadata["records"]["scan"]["actual_devices"]["camera"]["exposure"], 0.01);
        assert_eq!(latest.run().metadata["fixture"], "native-scan");
        assert_eq!(latest.run().metadata["plan"]["axes"][0]["values"], json!([1.25, 2.25, 3.25]));
        let read = Read::from_canonical(latest);
        let provenance = read.provenance();
        let acquisition = zlc_core::acquisition::expand_acquisition(&provenance["acquisition"]).unwrap();
        let events = acquisition["events"].as_array().unwrap();
        let ranges = acquisition["ranges"].as_array().unwrap();
        assert_eq!(ranges.len(), 12);
        for (index, range) in ranges.iter().enumerate() {
            let event = &events[range["event"].as_u64().unwrap() as usize];
            let parent = &events[event["parents"][0].as_u64().unwrap() as usize];
            assert_eq!(event["record"]["records"]["scan"]["shot"], json!(index + 40));
            assert_eq!(parent["record"]["records"]["frames"]["shot"], json!(index + 40));
        }
    }
}

#[test]
fn a_singleton_host_axis_still_arms_one_segment_per_sweep() {
    let Some(h) = harness() else { return };
    let schema = sample_schema(2, 0);
    let source = source(&h.runtime, schema.clone(), Storage::Latest);
    let spec = json!({"axes": [{"port": "device:rf:frequency", "identity": "frequency", "label": "RF frequency", "unit": "MHz",
        "values": [80.25], "coordinate_of": null, "host_advanced": true}], "scan_repeats": 2, "run_repeats": 1});
    let run = start_plan(&h, spec, "collect", false, "");
    ready(&h, &run);
    publish(&h.runtime, &source, schema.clone(), 100, None);
    loop {
        match h.events.recv_timeout(Duration::from_secs(20)).expect("second segment armed") {
            HostEvent::RunProgress { run: id, message, .. } if id == run && message == "armed:1" => break,
            HostEvent::RunFinished { run: id, outcome, error, .. } if id == run => panic!("scan ended before its second sweep: {outcome}: {error:?}"),
            HostEvent::Failed { reason, .. } => panic!("plugin failed: {reason}"),
            _ => {},
        }
    }
    publish(&h.runtime, &source, schema, 101, None);
    let (outcome, result, error) = finished(&h, &run);
    assert_eq!(outcome, "completed", "{error}");
    assert_eq!(result["written"], 2);
    assert_eq!(result["rows"]["inner"], json!([[]]));
    let signal = h.runtime.signals().into_iter().find(|signal| &*signal.signal == OUTPUT).unwrap();
    let latest = h.runtime.read_latest(&signal.generation, OUTPUT).unwrap();
    let origins: Vec<_> = latest.canonical().unwrap().segments().map(|segment| segment.origin()).collect();
    assert_eq!(origins, [[0, 0], [1, 0]]);
}

#[test]
fn native_scan_rejects_a_missing_source_ordinal_instead_of_reassigning_the_next_shot() {
    for mode in ["collect", "latch"] {
        let Some(h) = harness() else { return };
        let schema = sample_schema(2, 0);
        let source = source(&h.runtime, schema.clone(), Storage::Latest);
        let run = start(&h, mode, true);
        ready(&h, &run);
        publish(&h.runtime, &source, schema.clone(), 10, None);
        publish(&h.runtime, &source, schema, 12, None);
        let (outcome, _, error) = finished(&h, &run);
        assert_eq!(outcome, "failed");
        assert!(error.contains("ordinal jumped"), "{error}");
    }
}

#[test]
fn native_scan_reports_source_end_and_cancels_a_quiet_collection() {
    for cancel in [false, true] {
        let Some(h) = harness() else { return };
        let source = source(&h.runtime, sample_schema(2, 0), Storage::Latest);
        let run = start(&h, "collect", false);
        ready(&h, &run);
        let started = Instant::now();
        if cancel { h.host.stop_run("native", &run).unwrap(); }
        else { h.runtime.seal(&source, json!({"status": "completed"})).unwrap(); }
        let (outcome, _, error) = finished(&h, &run);
        assert!(started.elapsed() < Duration::from_secs(3), "a quiet input prevented completion");
        if cancel { assert_eq!(outcome, "stopped", "{error}"); }
        else {
            assert_eq!(outcome, "failed");
            assert!(error.contains("source signal") && error.contains("ended"), "{error}");
        }
    }
}

#[test]
fn native_scan_refuses_changed_source_event_schema() {
    let Some(h) = harness() else { return };
    let source = source(&h.runtime, sample_schema(3, 0), Storage::Finite);
    let run = start(&h, "collect", true);
    ready(&h, &run);
    // Both chunks are legal in the source's finite dataset, but one scan shot
    // cannot silently change from one source point to two source points.
    publish(&h.runtime, &source, sample_schema(1, 0), 0, Some([0, 0]));
    publish(&h.runtime, &source, sample_schema(2, 1), 1, Some([0, 1]));
    let (outcome, _, error) = finished(&h, &run);
    assert_eq!(outcome, "failed");
    assert!(error.contains("source dataset schema changed"), "{error}");
}

#[test]
fn a_new_run_cannot_use_the_previous_runs_scan_handle() {
    let Some(h) = harness() else { return };
    let _source = source(&h.runtime, sample_schema(2, 0), Storage::Latest);
    let previous = start(&h, "remember", true);
    let (outcome, remembered, error) = finished(&h, &previous);
    assert_eq!(outcome, "completed", "{error}");
    let current = start(&h, "stale", true);
    let (outcome, refused, error) = finished(&h, &current);
    assert_eq!(outcome, "completed", "{error}");
    assert_ne!(remembered["handle"], refused["new_handle"]);
    assert!(refused["stale_refused"].as_str().unwrap().contains("not current"), "{refused}");
}

#[test]
fn production_seamless_scan_runs_through_real_rpc_and_native_collection() {
    let Some(h) = harness() else { return };
    let schema = sample_schema(2, 0);
    let source = source(&h.runtime, schema.clone(), Storage::Latest);
    let run = start(&h, "seamless", true);
    let mut stages = Vec::new();
    let mut stops = 0;
    let mut waits = 0;
    let (outcome, result, error) = loop {
        match h.events.recv_timeout(Duration::from_secs(20)).expect("production scan event") {
            HostEvent::Request { run: id, ticket, op, params, .. } if id == run => {
                assert_eq!(op, "core");
                let args = &params["params"];
                match params["command"].as_str().unwrap() {
                    "pulse_stop" => { stops += 1; h.host.respond(ticket, Ok(json!({}))); },
                    "pulse_run" => {
                        let stage = args["stage"].as_str().unwrap();
                        stages.push(stage.to_owned());
                        let handle = args["scan_plan"]["handle"].as_str().expect("native scan handle");
                        assert!(args.get("scan").is_none(),"production scan materialized its table in Python");
                        let plan = h.host.scan_plan(&run,handle).unwrap();
                        assert_eq!(plan.inner_count(),3);
                        if stage == "load" {
                            assert_eq!(plan.axes()[0].values,[1.0,2.0,3.0]);
                        } else {
                            assert_eq!(stage, "fire");
                            assert_eq!(args["run_repeats"], 2);
                            assert_eq!(args["scan_repeats"], 2);
                        }
                        h.host.accept_scan_pulse(&run,handle,Some(vec![vec![1.25],vec![2.25],vec![3.25]]),
                            json!({"config":{},"program":{"rows":[[100],[200],[300]],"native_snapshot":true}})).unwrap();
                        h.host.respond(ticket, Ok(json!({"command_id": 77, "applied": {"config": {}},
                            "scan": {"handle":handle,"fields": ["dwell"], "row_count":3}})));
                        if stage == "fire" {
                            for ordinal in 0..12 { publish(&h.runtime, &source, schema.clone(), ordinal, None); }
                        }
                    },
                    "pulse_wait_done" => {
                        waits += 1;
                        assert_eq!(args["command_id"], 77);
                        h.host.respond(ticket, Ok(json!({"done": true, "status": "done", "command_id": 77, "elapsed_seconds": 0.002})));
                    },
                    other => panic!("unexpected production scan command {other}"),
                }
            },
            HostEvent::RunFinished { run: id, outcome, result, error, .. } if id == run =>
                break (outcome, result, error),
            HostEvent::Failed { reason, .. } => panic!("plugin failed: {reason}"),
            _ => {},
        }
    };
    assert_eq!(outcome, "completed", "{error:?}");
    assert_eq!(result, json!({"points": 3, "shots": 12}));
    assert_eq!(stages, ["load", "fire"]);
    assert_eq!((stops, waits), (1, 1));
    let signal = h.runtime.signals().into_iter().find(|signal| &*signal.signal == OUTPUT).unwrap();
    let latest = h.runtime.read_latest(&signal.generation, OUTPUT).unwrap();
    assert_eq!(latest.canonical().unwrap().written_cells(), 24);
    assert_eq!(latest.run().metadata["plan"]["axes"][0]["values"], json!([1.25, 2.25, 3.25]));
    assert_eq!(latest.run().metadata["device_snapshots"]["sequencer"]["program"]["native_snapshot"],true);
}

#[test]
#[ignore = "manual native collection throughput; requires ZLC_PLUGIN_PYTHON"]
fn native_collection_ten_thousand_events_benchmark() {
    let Some(h) = harness() else { return };
    let schema = sample_schema(2, 0);
    let source = source(&h.runtime, schema.clone(), Storage::Latest);
    let spec = json!({"axes": [{"port": "pulse:param:dwell", "identity": "dwell", "label": "Dwell", "unit": "us",
        "values": [1.0], "coordinate_of": null, "host_advanced": false}], "scan_repeats": 1, "run_repeats": 10_000});
    let run = start_plan(&h, spec, "bench", true, "");
    ready(&h, &run);
    let started = Instant::now();
    let mut consumed = 0;
    for first in (0..10_000).step_by(64) {
        let end = (first + 64).min(10_000);
        for ordinal in first..end { publish(&h.runtime, &source, schema.clone(), ordinal, None); }
        while consumed < end {
            match h.events.recv_timeout(Duration::from_secs(20)).expect("bench collection progress") {
                HostEvent::RunProgress { run: id, message, .. } if id == run && message.starts_with("collected:") => {
                    consumed = message["collected:".len()..].parse::<u64>().unwrap();
                },
                HostEvent::RunFinished { run: id, outcome, error, .. } if id == run => panic!("benchmark ended before progress: {outcome}: {error:?}"),
                HostEvent::Failed { reason, .. } => panic!("plugin failed: {reason}"),
                _ => {},
            }
        }
    }
    let (outcome, result, error) = finished(&h, &run);
    assert_eq!(outcome, "completed", "{error}");
    assert_eq!(result["written"], 10_000);
    let elapsed = started.elapsed();
    eprintln!("native scan: 10000 2-point/2-cell events, batches <=64, {:.3} s, {:.0} events/s (includes producer, Runtime commits and progress pacing)",
        elapsed.as_secs_f64(), 10_000.0 / elapsed.as_secs_f64());
}
