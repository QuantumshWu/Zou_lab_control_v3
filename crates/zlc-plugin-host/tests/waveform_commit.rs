//! Waveform Measurement through the real host and Runtime, with a virtual source
//! in the one plugin process. Needs ZLC_PLUGIN_PYTHON. ZLC_WAVEFORM_BENCH=1 also
//! runs the measurement of what a free-running source costs.
mod support;

use std::path::PathBuf;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use zlc_core::AnyBuffer;
use zlc_core::control::{Origin, SafeActions};
use zlc_core::runtime::{DeliveryPolicy, Front, Runtime, Start, Subscription};
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
        support::install(&host).unwrap();
    Some((host, runtime, events))
}

fn request(values: Value, outputs: &[String]) -> RunRequest {
    RunRequest { prepared: None, device_labels: Default::default(), device_units: Default::default(), values, devices: [("sampler".to_owned(), "source".to_owned())].into(),
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
    json!({"repeat": repeat, "trigger_slope": "rising", "trigger_channel": "q0/c0", "trigger_level": 0.0, "trigger_samples": 60})
}

fn start(host: &PluginHost, device: Value, values: Value) -> (String, Vec<String>) {
    let outputs: Vec<String> = (0..device["outputs"].as_u64().unwrap()).map(|index| format!("q{index}")).collect();
    host.open("source", "fixture.waveform", device, json!({})).unwrap();
    host.open("wave", "waveform_measurement", json!({}), json!({})).unwrap();
    (host.start_run("wave", request(values, &outputs)).unwrap(), outputs)
}

/// Single-sample and multi-sample records, several outputs: each edge one event holding
/// its trace, in order, followed whole, with the level at the middle of every trace.
#[test]
fn a_triggered_run_publishes_one_centred_trace_for_each_edge() {
    for (samples, outputs, channels) in [(1, 4, 10), (20, 2, 4)] {
        let Some((host, runtime, events)) = host() else { return };
        let (run, names) = start(&host, waveform(1000.0, samples, outputs, channels), triggered(0));
        let description = published(&runtime, "@logic/wave/q0");
        let follower = runtime.subscribe(&description.generation, vec!["@logic/wave/q0".into()], DeliveryPolicy::Follow, Start::Next).unwrap();
        let mut taken = Vec::new();
        for _ in 0..60 { std::thread::sleep(Duration::from_millis(50)); drain(&follower, &mut taken); }
        host.stop_run("wave", &run).unwrap();
        let (outcome, result, error) = await_end(&events, &run);
        assert_eq!(outcome, "stopped", "{error:?}");
        drain(&follower, &mut taken);
        assert!(taken.len() > 40, "{} traces in 3 s of 23.7 Hz", taken.len());
        let last = runtime.read_latest(&description.generation, "@logic/wave/q0").unwrap().key().ordinal.unwrap();
        assert_eq!(result["shots"].as_u64(), Some(last + 1));
        for pair in taken.windows(2) {
            assert_eq!(pair[1].0, pair[0].0 + 1, "every event in order");
            assert!(pair[1].2 > pair[0].2, "crossing times strictly increase");
        }
        let origin = taken[0].3 - taken[0].2;
        for (_, _, shot, source) in &taken { assert!((source - shot - origin).abs() < 1e-9); }
        let latest = runtime.read_latest(&description.generation, "@logic/wave/q0").unwrap();
        let schema = latest.data().unwrap().schema().clone();
        assert_eq!(schema.domain(zlc_core::schema::DomainRole::Cell).axes()[1].name(), "record time");
        assert_eq!(schema.domain(zlc_core::schema::DomainRole::Cell).axes()[1].len(), 60);
        assert_eq!(schema.domain(zlc_core::schema::DomainRole::Point).size(), 1);
        let AnyBuffer::F32(values) = latest.data().unwrap().values() else { panic!("dtype") };
        assert!(values.as_slice()[30].abs() < 1e-4, "the level at the middle of the trace: {}", values.as_slice()[30]);
        for name in &names {
            let signal = runtime.signals().into_iter().rev().find(|item| *item.signal == *format!("@logic/wave/{name}")).unwrap();
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

/// `repeat` with a trigger: one trace for each of N edges, a complete dataset;
/// without one, a finite run of records as it always was.
#[test]
fn a_finite_triggered_run_keeps_one_trace_for_each_edge() {
    let Some((host, runtime, events)) = host() else { return };
    let (run, _) = start(&host, waveform(1000.0, 10, 2, 4), triggered(5));
    let (outcome, result, error) = await_end(&events, &run);
    assert_eq!((outcome.as_str(), &result), ("completed", &json!({"shots": 5})), "{error:?}");
    let signal = runtime.signals().into_iter().rev().find(|item| &*item.signal == "@logic/wave/q0").unwrap();
    assert!(signal.finite && signal.sealed);
    assert_eq!(signal.schema.domain(zlc_core::schema::DomainRole::Repeat).size(), 5);
    assert_eq!(signal.schema.domain(zlc_core::schema::DomainRole::Cell).axes()[1].len(), 60);
    let latest = runtime.read_latest(&signal.generation, "@logic/wave/q0").unwrap();
    let dataset = latest.canonical().expect("finite dataset");
    assert_eq!((dataset.written_cells(), dataset.total_cells()), (5, 5));
    assert_eq!(latest.key().ordinal, Some(4));
    host.close("wave").unwrap();
    host.close("source").unwrap();

    let (run, _) = start(&host, waveform(1000.0, 10, 1, 2), json!({"repeat": 40}));
    let (outcome, result, _) = await_end(&events, &run);
    assert_eq!((outcome.as_str(), &result), ("completed", &json!({"shots": 40})));
    host.shutdown();
}

/// No edge: the run waits and says what it sees, and Stop ends it. A source that
/// stops delivering is a fault whatever the trigger is doing.
#[test]
fn a_quiet_trigger_is_a_status() {
    let Some((host, _runtime, events)) = host() else { return };
    let quiet = json!({"repeat": 0, "trigger_slope": "rising", "trigger_channel": "q0/c0", "trigger_level": 5.0, "trigger_samples": 20});
    let (run, _) = start(&host, waveform(400.0, 1, 1, 3), quiet);
    let until = Instant::now() + Duration::from_secs(10);
    let status = loop {
        assert!(Instant::now() < until, "no waiting status");
        if let Ok(HostEvent::RunProgress { message, .. }) = events.recv_timeout(Duration::from_secs(5)) {
            if message.starts_with("Waiting for trigger") { break message; }
        }
    };
    assert!(status.contains("q0 c0 ranged") && status.contains(" V, level 5"), "{status}");
    host.stop_run("wave", &run).unwrap();
    assert_eq!(await_end(&events, &run).0, "stopped");
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
