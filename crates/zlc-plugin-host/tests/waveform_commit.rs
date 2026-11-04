//! Waveform Measurement through the real host and Runtime, with a virtual source
//! in the one plugin process. Needs ZLC_PLUGIN_PYTHON. ZLC_WAVEFORM_BENCH=1 also
//! runs the measurement of what a free-running source costs.
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use zlc_core::AnyBuffer;
use zlc_core::control::{Origin, SafeActions};
use zlc_core::runtime::{DeliveryPolicy, Front, Read, ReadKind, Runtime, RuntimeError, Start, Subscription};
use zlc_plugin_host::{HostConfig, HostEvent, PluginHost, RunRequest};

type Host = (PluginHost, Runtime, mpsc::Receiver<HostEvent>);

fn host() -> Option<Host> {
    let Some(python) = std::env::var_os("ZLC_PLUGIN_PYTHON") else {
        eprintln!("ZLC_PLUGIN_PYTHON is not set: skipping the plugin process test");
        return None;
    };
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let config = HostConfig { python: python.into(), python_path: vec![root.join("../../science/src"), root.join("tests/plugins")],
        packages: vec!["zlc_atom".into(), "zlc_atom_fixtures".into()], memory_limit_mib: Some(4096) };
    let (runtime, safe) = (Runtime::new(), SafeActions::new());
    let (sender, events) = mpsc::channel();
    let sender = std::sync::Mutex::new(sender);
    let host = PluginHost::new(config, runtime.clone(), safe, move |event| { let _ = sender.lock().unwrap().send(event); }).unwrap();
    Some((host, runtime, events))
}

fn request(values: Value, outputs: &[String]) -> RunRequest {
    RunRequest { prepared: None, device_labels: Default::default(), values, devices: [("sampler".to_owned(), "source".to_owned())].into(),
        shared: Default::default(), origin: Origin::Operator, resources: json!({}), source_signal: None, run_directory: None,
        outputs: outputs.iter().map(|name| (name.clone(), format!("@logic/wave/{name}"))).collect(), record_devices: false }
}

fn published(runtime: &Runtime, signal: &str) -> zlc_core::runtime::SignalDescription {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if let Some(item) = runtime.signals().into_iter().rev().find(|item| &*item.signal == signal && item.published) { return item; }
        assert!(Instant::now() < deadline, "{signal} was not published");
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// Kernel + user seconds of a process.
fn cpu_seconds(pid: u32) -> f64 {
    use windows_sys::Win32::Foundation::{CloseHandle, FILETIME};
    use windows_sys::Win32::System::Threading::{GetProcessTimes, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION};
    let zero = || FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
    let (mut created, mut exited, mut kernel, mut user) = (zero(), zero(), zero(), zero());
    let seconds = |time: &FILETIME| (((time.dwHighDateTime as u64) << 32) | time.dwLowDateTime as u64) as f64 * 1e-7;
    unsafe {
        let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        assert!(!handle.is_null(), "cannot open process {pid}");
        assert!(GetProcessTimes(handle, &mut created, &mut exited, &mut kernel, &mut user) != 0);
        CloseHandle(handle);
    }
    seconds(&kernel) + seconds(&user)
}

fn await_end(events: &mpsc::Receiver<HostEvent>, run: &str) -> (String, Value, Option<String>) {
    loop {
        match events.recv_timeout(Duration::from_secs(60)).expect("run events") {
            HostEvent::RunFinished { run: id, outcome, result, error, .. } if id == run => break (outcome, result, error.map(|error| error.to_string())),
            HostEvent::Failed { reason, .. } => panic!("plugin failed: {reason}"),
            _ => {}
        }
    }
}

/// Each event of `signal` a run published, in order, as a follower takes it:
/// (ordinal, first value, shot time, the source time its record says).
fn drain(subscription: &Subscription, into: &mut Vec<(u64, f32, f64, f64)>) {
    while let Ok(Some(delivery)) = subscription.next(Duration::ZERO) {
        for publication in delivery.publications.iter() {
            let ordinal = publication.key().ordinal.unwrap();
            let AnyBuffer::F32(values) = publication.data().unwrap().values() else { panic!("dtype") };
            let timing = &publication.event().metadata["records"]["q0"]["record_timing"]["wave"][ordinal.to_string()];
            assert!((publication.event().metadata["shot_time_seconds"]["q0"].as_f64().unwrap() - timing["record_time_seconds"].as_f64().unwrap()).abs() < 1e-12, "{ordinal}");
            into.push((ordinal, values.as_slice()[0], publication.shot_time().unwrap(), timing["source_time_seconds"].as_f64().unwrap()));
        }
    }
}

fn waveform(rate: f64, samples: u64, outputs: u64, channels: u64) -> Value {
    json!({"rate_hz": rate, "record_samples": samples, "outputs": outputs, "channels": channels, "frequency_hz": 23.7, "noise": 0.03})
}

fn triggered(repeat: u64) -> Value {
    json!({"repeat": repeat, "trigger_slope": "rising", "trigger_channel": "q0/c0", "trigger_level": 0.0})
}

fn start(host: &PluginHost, device: Value, values: Value) -> (String, Vec<String>) {
    let outputs: Vec<String> = (0..device["outputs"].as_u64().unwrap()).map(|index| format!("q{index}")).collect();
    host.open("source", "fixture.waveform", device, json!({})).unwrap();
    host.open("wave", "waveform_measurement", json!({}), json!({})).unwrap();
    (host.start_run("wave", request(values, &outputs)).unwrap(), outputs)
}

/// A value of the shown window's last event and the one before, per output `q0`.
fn tail(read: &Read) -> (u64, f32, f32) {
    let ReadKind::Window(window) = read.kind() else { panic!("a window") };
    let entries = window.entries();
    let value = |entry: &std::sync::Arc<zlc_core::runtime::Publication>| {
        let AnyBuffer::F32(values) = entry.data().unwrap().values() else { panic!("dtype") };
        values.as_slice()[0]
    };
    (entries[entries.len() - 1].key().ordinal.unwrap(), value(&entries[entries.len() - 2]), value(&entries[entries.len() - 1]))
}

/// Single-sample and multi-sample records, several outputs: every sample one event,
/// in order, followed whole; a board's window always ends on a rising edge, and the
/// phase at its right edge stays put from one refresh to the next.
#[test]
fn a_triggered_run_publishes_every_sample_and_a_board_locks_on_its_edges() {
    for (samples, outputs, channels) in [(1, 4, 10), (20, 2, 4)] {
        let Some((host, runtime, events)) = host() else { return };
        let (run, names) = start(&host, waveform(1000.0, samples, outputs, channels), triggered(0));
        let description = published(&runtime, "@logic/wave/q0");
        let lease = runtime.retain_history(&description.generation, "@logic/wave/q0", 300).unwrap();
        let follower = runtime.subscribe(&description.generation, vec!["@logic/wave/q0".into()], DeliveryPolicy::Follow, Start::Next).unwrap();
        let request = [(description.generation.clone(), std::sync::Arc::<str>::from("@logic/wave/q0"), Some(300))];
        let mut front = Front::default();
        let mut taken = Vec::new();
        let mut phases = Vec::new();
        let mut edges = std::collections::BTreeSet::new();
        let until = Instant::now() + Duration::from_secs(3);
        let mut behind = 0usize;
        while Instant::now() < until {
            std::thread::sleep(Duration::from_millis(30));
            drain(&follower, &mut taken);
            let Ok(shown) = runtime.read_coherent_views(&request, &mut front).remove(0) else { continue };
            // The lease may have come after the first edges.
            let ReadKind::Window(window) = shown.kind() else { panic!("a window") };
            if window.entries().len() < 2 { continue; }
            let (ordinal, before, at) = tail(&shown);
            assert!(before < 0.0 && at >= 0.0, "the view ends on an edge: {before} then {at}");
            edges.insert(ordinal);
            phases.push(at);
            let newest = runtime.read_latest(&description.generation, "@logic/wave/q0").unwrap().key().ordinal.unwrap();
            if newest > ordinal { behind += 1; }
        }
        assert!(edges.len() > 20, "{} distinct edges", edges.len());
        assert!(behind > 0, "other readers run ahead of a board");
        let spread = phases.iter().fold(f32::MIN, |a, b| a.max(*b)) - phases.iter().fold(f32::MAX, |a, b| a.min(*b));
        let per_sample = 2.0 * std::f32::consts::PI * 23.7 / 1000.0;
        assert!(spread < 2.0 * per_sample + 6.0 * 0.03, "the phase at the right edge moved {spread}");
        host.stop_run("wave", &run).unwrap();
        let (outcome, result, error) = await_end(&events, &run);
        assert_eq!(outcome, "stopped", "{error:?}");
        drain(&follower, &mut taken);
        drop(lease);
        let last = runtime.read_latest(&description.generation, "@logic/wave/q0").unwrap().key().ordinal.unwrap();
        assert_eq!(result["shots"].as_u64(), Some(last + 1), "Stop leaves nothing unpublished");
        assert_eq!(taken.last().unwrap().0, last);
        for pair in taken.windows(2) {
            assert_eq!(pair[1].0, pair[0].0 + 1, "every event in order");
            assert!(pair[1].2 > pair[0].2, "shot times strictly increase");
        }
        let origin = taken[0].3 - taken[0].2;
        for (ordinal, _, shot, source) in &taken {
            assert!((source - shot - origin).abs() < 1e-9);
            assert!((shot - *ordinal as f64 * 1e-3).abs() < 1e-6, "a sample's own time, every sample its own event: {ordinal} at {shot}");
        }
        for name in &names {
            let signal = runtime.signals().into_iter().rev().find(|item| *item.signal == *format!("@logic/wave/{name}")).unwrap();
            assert_eq!(signal.schema.domain(zlc_core::schema::DomainRole::Point).size(), 1, "one sample an event");
            assert_eq!(runtime.read_latest(&signal.generation, &signal.signal).unwrap().key().ordinal, Some(last), "one bundle");
        }
        host.shutdown();
    }
}

/// The same data through the batched commit as one commit per event gave: every
/// record its event, its ordinal, shot time and record; a board reads the newest.
#[test]
fn a_free_running_run_commits_its_events_in_batches_unchanged() {
    for (samples, expected_points) in [(1, 1), (50, 50)] {
        let Some((host, runtime, events)) = host() else { return };
        let (run, _) = start(&host, waveform(1000.0, samples, 1, 8), json!({"repeat": 0}));
        let description = published(&runtime, "@logic/wave/q0");
        assert_eq!(description.schema.domain(zlc_core::schema::DomainRole::Point).size(), expected_points);
        let follower = runtime.subscribe(&description.generation, vec!["@logic/wave/q0".into()], DeliveryPolicy::Follow, Start::Next).unwrap();
        let mut taken = Vec::new();
        for _ in 0..30 { std::thread::sleep(Duration::from_millis(50)); drain(&follower, &mut taken); }
        host.stop_run("wave", &run).unwrap();
        assert_eq!(await_end(&events, &run).0, "stopped");
        drain(&follower, &mut taken);
        assert!(taken.len() > 5);
        let period = samples as f64 * 1e-3;
        for pair in taken.windows(2) {
            assert_eq!(pair[1].0, pair[0].0 + 1);
            assert!((pair[1].2 - pair[0].2 - period).abs() < 1e-9);
        }
        let latest = runtime.read_latest(&description.generation, "@logic/wave/q0").unwrap();
        let request = [(description.generation.clone(), std::sync::Arc::<str>::from("@logic/wave/q0"), None)];
        assert_eq!(runtime.read_coherent_views(&request, &mut Front::default()).remove(0).unwrap().key(), latest.key());
        host.shutdown();
    }
}

/// `repeat` with a trigger: N consecutive samples from the first edge, nothing before it;
/// without one, a finite run of records as it always was.
#[test]
fn a_finite_triggered_run_keeps_n_samples_from_the_first_edge() {
    let Some((host, runtime, events)) = host() else { return };
    let (run, _) = start(&host, waveform(1000.0, 10, 2, 4), triggered(250));
    let (outcome, result, error) = await_end(&events, &run);
    assert_eq!((outcome.as_str(), &result), ("completed", &json!({"shots": 250})), "{error:?}");
    let signal = runtime.signals().into_iter().rev().find(|item| &*item.signal == "@logic/wave/q0").unwrap();
    assert!(signal.finite && signal.sealed);
    let latest = runtime.read_latest(&signal.generation, "@logic/wave/q0").unwrap();
    let dataset = latest.canonical().expect("finite dataset");
    assert_eq!((dataset.written_cells(), dataset.total_cells()), (250, 250));
    let values: Vec<f32> = dataset.segments().map(|segment| {
        let AnyBuffer::F32(values) = segment.data().values() else { panic!("dtype") };
        values.as_slice()[0]
    }).collect();
    assert!(values[0] >= 0.0 && values[0] < 0.6, "starts on the edge: {}", values[0]);
    assert!(values[1] > values[0] - 0.2, "rising");
    assert_eq!(latest.key().ordinal, Some(249));
    host.close("wave").unwrap();
    host.close("source").unwrap();

    let (run, _) = start(&host, waveform(1000.0, 10, 1, 2), json!({"repeat": 40}));
    let (outcome, result, _) = await_end(&events, &run);
    assert_eq!((outcome.as_str(), &result), ("completed", &json!({"shots": 40})));
    host.shutdown();
}

/// No edge: the run waits and says what it sees, the data goes on arriving at every
/// reader but a board's, and Stop leaves a run a board shows as it stands. A source
/// that stops delivering is a fault whatever the trigger is doing.
#[test]
fn a_quiet_trigger_is_a_status_and_a_dead_source_a_fault() {
    let Some((host, runtime, events)) = host() else { return };
    let quiet = json!({"repeat": 0, "trigger_slope": "rising", "trigger_channel": "q0/c0", "trigger_level": 5.0});
    let (run, _) = start(&host, waveform(400.0, 1, 1, 3), quiet);
    let description = published(&runtime, "@logic/wave/q0");
    let request = [(description.generation.clone(), std::sync::Arc::<str>::from("@logic/wave/q0"), None)];
    let until = Instant::now() + Duration::from_secs(10);
    let status = loop {
        assert!(Instant::now() < until, "no waiting status");
        if let Ok(HostEvent::RunProgress { message, .. }) = events.recv_timeout(Duration::from_secs(5)) {
            if message.starts_with("Waiting for trigger") { break message; }
        }
    };
    assert!(status.contains("q0 c0 ranged") && status.contains(" V, level 5"), "{status}");
    assert!(matches!(runtime.read_coherent_views(&request, &mut Front::default()).remove(0), Err(RuntimeError::NoData)));
    assert!(runtime.read_latest(&description.generation, "@logic/wave/q0").unwrap().key().ordinal.unwrap() > 100);
    host.stop_run("wave", &run).unwrap();
    assert_eq!(await_end(&events, &run).0, "stopped");
    assert!(runtime.read_coherent_views(&request, &mut Front::default()).remove(0).is_ok());
    host.shutdown();
}

/// Bench one layout: CPU of the plugin process and the core, round trips, over
/// `seconds`, with a board reading its 2000-event window of every output on each change.
fn bench_row(label: &str, device: Value, values: Value, seconds: f64) {
    let Some((host, runtime, events)) = host() else { return };
    let outputs: Vec<String> = (0..device["outputs"].as_u64().unwrap()).map(|index| format!("q{index}")).collect();
    host.open("source", "fixture.waveform", device, json!({})).unwrap();
    host.open("meter", "fixture.meter", json!({}), json!({})).unwrap();
    host.open("wave", "waveform_measurement", json!({}), json!({})).unwrap();
    let call = |method: &str| host.spi("meter", method, json!([]), json!({}), Duration::from_secs(5)).unwrap().value;
    call("start");
    let run = host.start_run("wave", request(values, &outputs)).unwrap();
    let pid = host.instances().into_iter().find(|(name, ..)| name == "wave").unwrap().2;
    let core = std::process::id();
    let first = published(&runtime, "@logic/wave/q0");
    let _leases: Vec<_> = outputs.iter().map(|name| runtime.retain_history(&first.generation, &format!("@logic/wave/{name}"), 2000).unwrap()).collect();
    let requests: Vec<_> = outputs.iter().map(|name| (first.generation.clone(), std::sync::Arc::<str>::from(format!("@logic/wave/{name}")), Some(2000))).collect();
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let board = {
        let (runtime, stop) = (runtime.clone(), stop.clone());
        std::thread::spawn(move || {
            let mut front = Front::default();
            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(50));
                let _ = runtime.read_coherent_views(&requests, &mut front);
            }
        })
    };
    std::thread::sleep(Duration::from_secs(1));
    call("reset");
    let (started, plugin, ours) = (Instant::now(), cpu_seconds(pid), cpu_seconds(core));
    std::thread::sleep(Duration::from_secs_f64(seconds));
    let elapsed = started.elapsed().as_secs_f64();
    let (plugin, ours) = (cpu_seconds(pid) - plugin, cpu_seconds(core) - ours);
    let meters = call("read");
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    board.join().unwrap();
    host.stop_run("wave", &run).unwrap();
    let (outcome, result, error) = await_end(&events, &run);
    println!("{label}: plugin CPU {:.1}%  core CPU {:.1}%  outcome {outcome} {result} {error:?}", 100.0 * plugin / elapsed, 100.0 * ours / elapsed);
    for (name, item) in meters.as_object().unwrap() {
        let (count, total) = (item[0].as_f64().unwrap(), item[1].as_f64().unwrap());
        println!("  {name:<20} {:>9.1}/s  {:>8.1} us each  {:>5.1}% of wall", count / elapsed, 1e6 * total / count, 100.0 * total / elapsed);
    }
    host.shutdown();
}

#[test]
fn bench_what_a_free_running_source_costs() {
    if std::env::var_os("ZLC_WAVEFORM_BENCH").is_none() { return; }
    let imu = || waveform(400.0, 1, 4, 10);
    let off = || json!({"repeat": 0});
    bench_row("400 Hz, 1 sample/record, 4 outputs, off", imu(), off(), 5.0);
    bench_row("400 Hz, 1 sample/record, 4 outputs, on", imu(), triggered(0), 5.0);
    bench_row("1 kHz, 1 sample/record, 8 channels, off", waveform(1000.0, 1, 1, 8), off(), 5.0);
    bench_row("1 kHz, 1 sample/record, 8 channels, on", waveform(1000.0, 1, 1, 8), triggered(0), 5.0);
    bench_row("1 kHz, 100 samples/record, 8 channels, off", waveform(1000.0, 100, 1, 8), off(), 5.0);
    bench_row("1 kHz, 100 samples/record, 8 channels, on", waveform(1000.0, 100, 1, 8), triggered(0), 5.0);
}
