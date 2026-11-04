//! Derive's groups through the real host and Runtime: what reaches Runtime is taken.
//! Needs ZLC_PLUGIN_PYTHON. The numbers against raw arrays are python_derive_groups.py's.
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use zlc_core::AnyBuffer;
use zlc_core::control::{Origin, SafeActions};
use zlc_core::runtime::{Publication, Runtime, SignalDescription};
use zlc_plugin_host::{HostConfig, HostEvent, PluginHost, RunRequest};

type Host = (PluginHost, Runtime, mpsc::Receiver<HostEvent>);

fn host() -> Option<Host> {
    let Some(python) = std::env::var_os("ZLC_PLUGIN_PYTHON") else {
        eprintln!("ZLC_PLUGIN_PYTHON is not set: skipping the plugin process test");
        return None;
    };
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let config = HostConfig { python: python.into(), python_path: vec![root.join("../../science/src"), root.join("tests/plugins")],
        packages: vec!["zlc_atom".into(), "zlc_test_plugins".into()], memory_limit_mib: Some(4096) };
    let (runtime, safe) = (Runtime::new(), SafeActions::new());
    let (sender, events) = mpsc::channel();
    let sender = std::sync::Mutex::new(sender);
    let host = PluginHost::new(config, runtime.clone(), safe, move |event| { let _ = sender.lock().unwrap().send(event); }).unwrap();
    Some((host, runtime, events))
}

fn request(values: Value, source: Option<&str>, outputs: &[(&str, &str)]) -> RunRequest {
    RunRequest { prepared: None, device_labels: Default::default(), values, devices: Default::default(), shared: Default::default(), origin: Origin::Operator,
        resources: json!({}), source_signal: source.map(str::to_owned), run_directory: None,
        outputs: outputs.iter().map(|(name, signal)| ((*name).to_owned(), (*signal).to_owned())).collect(), record_devices: false }
}

fn published(runtime: &Runtime, signal: &str) -> SignalDescription {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if let Some(item) = runtime.signals().into_iter().rev().find(|item| &*item.signal == signal && item.published) { return item; }
        assert!(Instant::now() < deadline, "{signal} was not published");
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn await_end(events: &mpsc::Receiver<HostEvent>, wanted: &[&str]) {
    let mut left: Vec<String> = wanted.iter().map(|run| (*run).to_owned()).collect();
    while !left.is_empty() {
        match events.recv_timeout(Duration::from_secs(60)).expect("run events") {
            HostEvent::RunFinished { run, outcome, error, .. } if left.contains(&run) => {
                assert_eq!(outcome, "completed", "{error:?}");
                left.retain(|item| *item != run);
            }
            HostEvent::Failed { reason, .. } => panic!("plugin failed: {reason}"),
            _ => {}
        }
    }
}

fn floats(values: &AnyBuffer) -> Vec<f64> {
    let AnyBuffer::F64(values) = values else { panic!("dtype") };
    values.as_slice().to_vec()
}

/// (segments, each segment's Repeat origin with its values, written cells, a sigma stated).
fn dataset(latest: &Publication) -> (usize, Vec<([usize; 2], Vec<f64>)>, usize, bool) {
    let dataset = latest.canonical().expect("finite dataset");
    let mut segments: Vec<_> = dataset.segments().map(|segment| (segment.origin(), floats(segment.data().values()))).collect();
    segments.sort_by_key(|(origin, _)| *origin);
    let sigma = dataset.segments().all(|segment| segment.data().sigma().is_some());
    (dataset.segment_count(), segments, dataset.written_cells(), sigma)
}

#[test]
fn groups_reach_runtime_as_write_once_placements_beside_a_per_event_latest() {
    let Some((host, runtime, events)) = host() else { return };
    let variables = json!([
        {"name": "a", "terms": [{"of": "counts"}], "statistic": "mean", "over": ["repeat", "Site"], "by": "repeat", "every": 2},
        {"name": "b", "terms": [{"of": "counts"}], "statistic": "mean", "over": ["Site"]},
    ]);
    let source = "@logic/repeater/counts";
    host.open("repeater", "repeater", json!({}), json!({})).unwrap();
    let repeater = host.start_run("repeater", request(json!({"repeats": 5, "points": 2, "pause": 0.1}), None, &[("counts", source)])).unwrap();
    published(&runtime, source);
    // Live: a follows the running source, one cell an event.
    let outputs = [("a", "@logic/live/a"), ("b", "@logic/live/b")];
    host.open("live", "derive", json!({}), json!({})).unwrap();
    let live = host.start_run("live", request(json!({"variables": variables}), Some(source), &outputs)).unwrap();
    await_end(&events, &[&repeater, &live]);

    let a = published(&runtime, "@logic/live/a");
    assert!(a.finite && a.sealed);
    let latest = runtime.read_latest(&a.generation, "@logic/live/a").unwrap();
    let (segments, placed, written, sigma) = dataset(&latest);
    // Three groups each closed in a commit of its own, and b was in every other commit alone.
    assert_eq!((segments, written), (3, 6));
    assert!(sigma, "a mean states its standard error");
    let expected = [[5.1, 6.1], [25.1, 26.1], [40.1, 41.1]];
    for (row, (origin, values)) in placed.iter().enumerate() {
        assert_eq!(*origin, [row, 0]);
        for (point, want) in expected[row].iter().enumerate() {
            assert!((values[point] - want).abs() < 1e-12, "{values:?}");
        }
    }
    // The per-event output stays Latest, an event per delivery.
    let b = published(&runtime, "@logic/live/b");
    assert!(!b.finite);
    let last = runtime.read_latest(&b.generation, "@logic/live/b").unwrap();
    assert_eq!(last.key().ordinal, Some(9), "b follows its source's ordinals");
    assert!(last.data().unwrap().sigma().is_none());

    // Sealed: the whole dataset is one delivery and one commit, the same numbers.
    let only = json!([variables[0]]);
    let outputs = [("a", "@logic/sealed/a")];
    host.open("sealed", "derive", json!({}), json!({})).unwrap();
    let sealed = host.start_run("sealed", request(json!({"variables": only}), Some(source), &outputs)).unwrap();
    await_end(&events, &[&sealed]);
    let a = published(&runtime, "@logic/sealed/a");
    let latest = runtime.read_latest(&a.generation, "@logic/sealed/a").unwrap();
    let (segments, placed, written, sigma) = dataset(&latest);
    assert_eq!((segments, written), (1, 6));
    assert!(sigma);
    let flat: Vec<f64> = expected.iter().flatten().copied().collect();
    for (got, want) in placed[0].1.iter().zip(&flat) {
        assert!((got - want).abs() < 1e-12, "{:?}", placed[0].1);
    }
    host.shutdown();
}

#[test]
fn a_monitor_gives_one_latest_event_per_group_and_its_seal_the_partial_one() {
    let Some((host, runtime, events)) = host() else { return };
    let source = "@logic/repeater/counts";
    host.open("repeater", "repeater", json!({}), json!({})).unwrap();
    let repeater = host.start_run("repeater", request(json!({"repeats": 7, "monitor": true, "pause": 0.1}), None, &[("counts", source)])).unwrap();
    published(&runtime, source);
    let variables = json!([{"name": "a", "terms": [{"of": "counts"}], "statistic": "mean", "over": ["repeat"], "by": "repeat", "every": 3}]);
    host.open("groups", "derive", json!({}), json!({})).unwrap();
    let groups = host.start_run("groups", request(json!({"variables": variables}), Some(source), &[("a", "@logic/groups/a")])).unwrap();
    await_end(&events, &[&repeater, &groups]);
    let a = published(&runtime, "@logic/groups/a");
    assert!(!a.finite && a.sealed);
    let latest = runtime.read_latest(&a.generation, "@logic/groups/a").unwrap();
    // Groups 0, 1 by ordinal (shots 0-2, 3-5), then shot 6 alone at the seal.
    assert_eq!(latest.key().ordinal, Some(2));
    let values = floats(latest.data().unwrap().values());
    assert!((values[0] - 60.0).abs() < 1e-12, "{values:?}");
    host.shutdown();
}
