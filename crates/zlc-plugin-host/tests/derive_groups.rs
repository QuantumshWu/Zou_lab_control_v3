//! Derive's groups through the real host and Runtime: what reaches Runtime is taken.
//! Needs ZLC_PLUGIN_PYTHON. The numbers against raw arrays are python_derive_groups.py's.
mod support;

use std::path::PathBuf;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use zlc_core::AnyBuffer;
use zlc_core::control::{Origin, SafeActions};
use zlc_core::runtime::{Front, Publication, Runtime, SignalDescription};
use zlc_core::schema::DomainRole;
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
        support::install(&host).unwrap();
    Some((host, runtime, events))
}

fn request(values: Value, source: Option<&str>, outputs: &[(&str, &str)]) -> RunRequest {
    RunRequest { prepared: None, device_labels: Default::default(), device_units: Default::default(), values, devices: Default::default(), shared: Default::default(), origin: Origin::Operator,
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

/// What a finite output holds, by (Repeat row, Point row) of its segments: rows x points x sites
/// values, and how many segments were written.
fn plane(runtime: &Runtime, signal: &str) -> (Vec<f64>, Vec<f64>, usize, usize) {
    let item = published(runtime, signal);
    assert!(item.finite && item.sealed, "{signal}");
    let latest = runtime.read_latest(&item.generation, signal).unwrap();
    let dataset = latest.canonical().expect("finite dataset");
    let schema = dataset.schema();
    let [rows, points, cells] = [schema.domain(DomainRole::Repeat), schema.domain(DomainRole::Point), schema.domain(DomainRole::Cell)].map(|domain| domain.physical_shape().iter().product::<usize>());
    let (mut values, mut sigmas) = (vec![f64::NAN; rows * points * cells], vec![f64::NAN; rows * points * cells]);
    for segment in dataset.segments() {
        let [row, point] = segment.origin();
        let block = floats(segment.data().values());
        let (high, wide) = (segment.data().schema().domain(DomainRole::Repeat).physical_shape()[0], segment.data().schema().domain(DomainRole::Point).physical_shape()[0]);
        assert_eq!(block.len(), high * wide * cells);
        for r in 0..high {
            for p in 0..wide {
                for c in 0..cells {
                    let (from, to) = ((r * wide + p) * cells + c, ((row + r) * points + point + p) * cells + c);
                    assert!(values[to].is_nan(), "{signal} wrote a cell twice");
                    values[to] = block[from];
                    if let Some(sigma) = segment.data().sigma() { sigmas[to] = sigma.as_slice()[from]; }
                }
            }
        }
    }
    (values, sigmas, dataset.segment_count(), rows)
}

fn close(got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len());
    for (got, want) in got.iter().zip(want) { assert!((got - want).abs() < 1e-9, "{got} against {want}"); }
}

/// The repeater's source, sealed; each `(node, variables, outputs)` running while it publishes.
/// Then the same variables over what it sealed, as `<node>_sealed`.
fn both_routes(host: &PluginHost, events: &mpsc::Receiver<HostEvent>, runtime: &Runtime, values: Value, node: &str, variables: &Value, outputs: &[&str]) {
    // Its own name: a node run again leaves its last generation standing until it publishes.
    let source = &format!("@logic/repeater_{node}/counts");
    let source = source.as_str();
    let named = |node: &str| outputs.iter().map(|name| ((*name).to_owned(), format!("@logic/{node}/{name}"))).collect::<Vec<_>>();
    let start = |node: &str| {
        let names = named(node);
        let names: Vec<(&str, &str)> = names.iter().map(|(name, signal)| (name.as_str(), signal.as_str())).collect();
        host.start_run(node, request(json!({"variables": variables}), Some(source), &names)).unwrap()
    };
    // Open before the source runs: a source sealed before its reader starts is one delivery, not the live route.
    let sealed_node = format!("{node}_sealed");
    host.open(node, "derive", json!({}), json!({})).unwrap();
    host.open(&sealed_node, "derive", json!({}), json!({})).unwrap();
    if host.instances().iter().all(|(name, _, _)| name != "repeater") { host.open("repeater", "repeater", json!({}), json!({})).unwrap(); }
    let repeater = host.start_run("repeater", request(values, None, &[("counts", source)])).unwrap();
    published(runtime, source);
    let live = start(node);
    await_end(events, &[&repeater, &live]);
    let sealed = start(&sealed_node);
    await_end(events, &[&sealed]);
}

#[test]
fn a_point_axis_group_closes_per_kept_repeat_on_both_routes() {
    let Some((host, runtime, events)) = host() else { return };
    let variables = json!([{"name": "a", "terms": [{"of": "counts"}], "statistic": "mean", "over": ["duration"], "by": "duration", "every": 2}]);
    both_routes(&host, &events, &runtime, json!({"repeats": 2, "points": 5, "pause": 0.01}), "points", &variables, &["a"]);
    let (values, sigmas, segments, rows) = plane(&runtime, "@logic/points/a");
    // Each kept repeat's groups (p 0-1, 2-3, 4), site by site; each closed by itself, the last one full at one point.
    assert_eq!((segments, rows), (6, 2));
    let want: Vec<f64> = [0.0, 10.0].iter().flat_map(|r| [0.5, 2.5, 4.0].map(|p| r + p)).flat_map(|v| [0.0, 0.1, 0.2].map(|c| v + c)).collect();
    close(&values, &want);
    let half = (0.5f64 / 2.0).sqrt();
    assert!((sigmas[0] - half).abs() < 1e-9 && sigmas[6].is_nan(), "two points state their standard error; one has none: {sigmas:?}");
    let (sealed, _sigmas, segments, _rows) = plane(&runtime, "@logic/points_sealed/a");
    assert_eq!(segments, 1, "a sealed source is one delivery and one commit");
    close(&sealed, &want);
    host.shutdown();
}

#[test]
fn one_delivery_that_closes_two_rectangles_makes_two_commits() {
    let Some((host, runtime, events)) = host() else { return };
    let variables = json!([{"name": "a", "terms": [{"of": "counts"}], "statistic": "sum", "over": ["duration"], "by": "duration", "every": 2}]);
    // The fourth publication closes (0, 0), (0, 1) and (1, 0): rows of different lengths; the last one (1, 1).
    let places = "[[0,0,1,1],[0,3,1,1],[1,0,1,1],[0,1,2,2],[1,3,1,1]]";
    both_routes(&host, &events, &runtime, json!({"repeats": 2, "points": 4, "pause": 0.1, "places": places}), "many", &variables, &["a"]);
    let (values, _sigmas, segments, rows) = plane(&runtime, "@logic/many/a");
    assert_eq!((segments, rows), (3, 2), "the rectangles, then the last group");
    let want: Vec<f64> = [0.0, 10.0].iter().flat_map(|r| [2.0 * r + 1.0, 2.0 * r + 5.0]).flat_map(|v| [0.0, 0.2, 0.4].map(|c| v + c)).collect();
    close(&values, &want);
    close(&plane(&runtime, "@logic/many_sealed/a").0, &want);
    host.shutdown();
}

#[test]
fn a_group_of_groups_closes_with_what_it_reads() {
    let Some((host, runtime, events)) = host() else { return };
    let variables = json!([
        {"name": "a", "publish": false, "terms": [{"of": "counts"}], "statistic": "mean", "over": ["repeat"], "by": "repeat", "every": 2},
        {"name": "b", "terms": [{"of": "a"}], "statistic": "mean", "over": ["repeat", "Site"], "by": "repeat", "every": 2},
    ]);
    both_routes(&host, &events, &runtime, json!({"repeats": 7, "points": 2, "pause": 0.01}), "second", &variables, &["b"]);
    let (values, sigmas, segments, rows) = plane(&runtime, "@logic/second/b");
    // a: repeats 0-1, 2-3, 4-5, 6; b: a's 0-1 (repeats 0-3), a's 2-3: a mean of two means, 45 and 60; site mean 0.1.
    assert_eq!((segments, rows), (2, 2));
    let want = [15.1, 16.1, 52.6, 53.6];
    close(&values, &want);
    assert!(sigmas.iter().all(|sigma| sigma.is_finite() && *sigma > 0.0), "{sigmas:?}");
    let (sealed, sealed_sigmas, _segments, _rows) = plane(&runtime, "@logic/second_sealed/b");
    close(&sealed, &want);
    close(&sealed_sigmas, &sigmas);
    host.shutdown();
}

#[test]
fn a_repeat_group_beside_a_kept_repeat_axis_closes_by_its_own_axis() {
    let Some((host, runtime, events)) = host() else { return };
    let variables = json!([{"name": "a", "terms": [{"of": "counts"}], "statistic": "mean", "over": ["repeat", "Site"], "by": "repeat", "every": 2}]);
    for (node, slow) in [("fast", false), ("slow", true)] {
        both_routes(&host, &events, &runtime, json!({"repeats": 5, "points": 2, "shots": 2, "shots_slow": slow, "pause": 0.005}), node, &variables, &["a"]);
        let (values, _sigmas, _segments, rows) = plane(&runtime, &format!("@logic/{node}/a"));
        assert_eq!(rows, 6, "three groups of repeat beside two shots");
        // Group means of r (0-1, 2-3, 4) x 10 + shot x 100 + p + site mean 0.1, each in its group whole.
        let order: Vec<(f64, f64)> = if slow { [0.0, 100.0].iter().flat_map(|s| [5.0, 25.0, 40.0].map(|r| (r, *s))).collect() }
                                     else { [5.0, 25.0, 40.0].iter().flat_map(|r| [0.0, 100.0].map(|s| (*r, s))).collect() };
        let want: Vec<f64> = order.iter().flat_map(|(r, s)| [0.1, 1.1].map(|p| r + s + p)).collect();
        close(&values, &want);
        close(&plane(&runtime, &format!("@logic/{node}_sealed/a")).0, &want);
    }
    host.shutdown();
}

/// A slow clock on a running source: the signal is known from the first delivery, the source
/// stays live on a board beside it, and each closed group shows.
#[test]
fn a_group_on_a_running_source_holds_no_board_back() {
    let Some((host, runtime, events)) = host() else { return };
    let source = "@logic/repeater/counts";
    host.open("repeater", "repeater", json!({}), json!({})).unwrap();
    let repeater = host.start_run("repeater", request(json!({"repeats": 8, "points": 2, "pause": 0.1}), None, &[("counts", source)])).unwrap();
    let scan = published(&runtime, source);
    let variables = json!([{"name": "a", "terms": [{"of": "counts"}], "statistic": "mean", "over": ["repeat", "Site"], "by": "repeat", "every": 3}]);
    host.open("groups", "derive", json!({}), json!({})).unwrap();
    let groups = host.start_run("groups", request(json!({"variables": variables}), Some(source), &[("a", "@logic/groups/a")])).unwrap();
    // Declared before its first group closes, six events on.
    let until = Instant::now() + Duration::from_secs(5);
    let signal = loop {
        if let Some(found) = runtime.signals().into_iter().find(|item| &*item.signal == "@logic/groups/a") { break found; }
        assert!(Instant::now() < until, "not declared");
        std::thread::sleep(Duration::from_millis(5));
    };
    let requests = [(scan.generation.clone(), std::sync::Arc::<str>::from(source), None), (signal.generation.clone(), std::sync::Arc::<str>::from("@logic/groups/a"), None)];
    let mut front = Front::default();
    let (mut behind, mut reads, mut groups_seen) = (0, 0, std::collections::BTreeSet::new());
    while runtime.signals().into_iter().rev().find(|item| &*item.signal == source).is_some_and(|item| !item.sealed) {
        let answer = runtime.read_coherent_views(&requests, &mut front);
        let newest = runtime.read_latest(&scan.generation, source).unwrap().key().ordinal.unwrap();
        let shown = answer[0].as_ref().map(|read| read.key().ordinal.unwrap()).unwrap_or(0);
        if newest > shown + 1 { behind += 1; }
        reads += 1;
        if let Ok(read) = &answer[1] { groups_seen.insert(read.key().ordinal.unwrap()); }
        std::thread::sleep(Duration::from_millis(10));
    }
    await_end(&events, &[&repeater, &groups]);
    assert!(reads > 50);
    assert!(behind * 20 < reads, "the source is live: behind by more than an event in {behind} of {reads} reads");
    assert!(groups_seen.len() >= 2, "each closed group shows: {groups_seen:?}");
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
