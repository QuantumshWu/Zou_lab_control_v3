//! Line D's readout chain through the host: a recorded finite site-frame run
//! (zlc_atom_fixtures.fixture_frames, the v2 known-truth fixture), Occupancy
//! following it with a saved Calibration, Frame Survival following Occupancy.
//! Needs ZLC_PLUGIN_PYTHON, ZLC_ATOM_FIXTURE (the fixture NPZ) and
//! ZLC_ATOM_CALIBRATION (its calibration JSON); ZLC_ATOM_OUT, when set, receives
//! every committed segment for comparison with v2 (runs/v2port/D_tools).
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::Duration;

use serde_json::json;
use zlc_core::dataset::Validity;
use zlc_core::control::{Origin, SafeActions};
use zlc_core::runtime::Runtime;
use zlc_plugin_host::{HostConfig, HostEvent, PluginHost, RunRequest};

#[test]
fn occupancy_and_frame_survival_follow_a_finite_site_frame_run() {
    let (Some(python), Some(fixture), Some(calibration)) = (std::env::var_os("ZLC_PLUGIN_PYTHON"),
        std::env::var("ZLC_ATOM_FIXTURE").ok(), std::env::var("ZLC_ATOM_CALIBRATION").ok()) else {
        eprintln!("ZLC_PLUGIN_PYTHON, ZLC_ATOM_FIXTURE or ZLC_ATOM_CALIBRATION is not set: skipped"); return;
    };
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let python_path = vec![root.join("../../science/src"), root.join("tests/plugins")];
    let (runtime, safe) = (Runtime::new(), SafeActions::new());
    let (sender, events) = mpsc::channel();
    let sender = std::sync::Mutex::new(sender);
    let host = PluginHost::new(HostConfig { python: python.into(), python_path,
        packages: vec!["zlc_atom".into(), "zlc_atom_fixtures".into()], memory_limit_mib: Some(4096) },
        runtime.clone(), safe, move |event| { let _ = sender.lock().unwrap().send(event); }).unwrap();
    let request = |values: serde_json::Value, resources: serde_json::Value, source: Option<&str>, outputs: &[(&str, &str)]| {
        RunRequest { prepared: None, device_labels: Default::default(), values, devices: Default::default(), shared: Default::default(), origin: Origin::Operator, resources,
            source_signal: source.map(str::to_owned), run_directory: None,
            outputs: outputs.iter().map(|(output, signal)| (output.to_string(), signal.to_string())).collect() }
    };
    let published = |name: &str| loop {
        if let Some(item) = runtime.signals().into_iter().rev().find(|item| item.signal.as_ref() == name && item.published) {
            break item;
        }
        std::thread::sleep(Duration::from_millis(20));
    };

    host.open("camera_measurement", "fixture_frames", json!({}), json!({})).unwrap();
    host.open("occupancy1", "occupancy", json!({}), json!({})).unwrap();
    host.open("frame_survival1", "frame_survival", json!({}), json!({})).unwrap();
    let frames = "@logic/camera_measurement/frames";
    let source = host.start_run("camera_measurement", request(json!({"path": fixture, "hold_seconds": 3.0}), json!({}), None,
        &[("frames", frames)])).unwrap();
    published(frames);
    let occupancy = host.start_run("occupancy1", request(json!({"model_kind": "default"}),
        json!({"calibrations": [{"path": calibration, "frames": ["rest"]}]}), Some(frames),
        &[("counts", "@logic/occupancy1/counts"), ("occupied", "@logic/occupancy1/occupied"),
          ("frame_judged", "@logic/occupancy1/frame_judged")])).unwrap();
    published("@logic/occupancy1/occupied");
    let survival = host.start_run("frame_survival1", request(json!({"selection": "all", "pairs": [], "custom_pairs": []}),
        json!({}), Some("@logic/occupancy1/occupied"), &[("survival", "@logic/frame_survival1/survival")])).unwrap();

    let mut outcomes = std::collections::BTreeMap::new();
    while outcomes.len() < 3 {
        match events.recv_timeout(Duration::from_secs(120)).expect("run events") {
            HostEvent::RunFinished { run, outcome, error, .. } => { outcomes.insert(run, (outcome, error)); }
            HostEvent::Failed { reason, .. } => panic!("plugin failed: {reason}"),
            _ => {}
        }
    }
    for run in [&source, &occupancy, &survival] {
        let (outcome, error) = &outcomes[run];
        assert_eq!(outcome, "completed", "{run}: {error:?}");
    }

    let out = std::env::var("ZLC_ATOM_OUT").ok().map(PathBuf::from);
    for (signal, cells) in [("@logic/occupancy1/counts", 180), ("@logic/occupancy1/occupied", 180),
                            ("@logic/frame_survival1/survival", 180)] {
        let item = published(signal);
        assert!(item.finite && item.sealed, "{signal} is a sealed finite dataset");
        let latest = runtime.read_latest(&item.generation, signal).unwrap();
        let dataset = latest.canonical().expect("finite dataset");
        assert_eq!((dataset.written_cells(), dataset.total_cells()), (cells, cells), "{signal}: every cycle placed");
        let Some(out) = &out else { continue };
        let name = signal.rsplit('/').next().unwrap();
        std::fs::create_dir_all(out).unwrap();
        let mut manifest = Vec::new();
        for (index, segment) in dataset.segments().enumerate() {
            let data = segment.data();
            std::fs::write(out.join(format!("{name}.{index}.values")), data.values().as_bytes()).unwrap();
            let validity = match data.validity() {
                Validity::Uniform(valid) => json!(valid),
                Validity::Mask(mask) => {
                    std::fs::write(out.join(format!("{name}.{index}.valid")), mask.as_bytes()).unwrap();
                    json!("mask")
                }
            };
            manifest.push(json!({"origin": segment.origin(), "extent": segment.extent(),
                                 "dtype": format!("{:?}", data.dtype()), "validity": validity}));
        }
        std::fs::write(out.join(format!("{name}.json")), serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();
    }

    // Started on the sealed source, Occupancy evaluates its whole dataset in one
    // delivery (v2's frozen processor): the counts it followed, cell for cell.
    let counts = "@logic/occupancy1/counts";
    let followed = runtime.read_latest(&published(counts).generation, counts).unwrap();
    let frozen = host.start_run("occupancy1", request(json!({"model_kind": "default"}),
        json!({"calibrations": [{"path": calibration, "frames": ["rest"]}]}), Some(frames),
        &[("counts", counts), ("occupied", "@logic/occupancy1/occupied"), ("frame_judged", "@logic/occupancy1/frame_judged")])).unwrap();
    let (outcome, error) = loop {
        match events.recv_timeout(Duration::from_secs(120)).expect("run events") {
            HostEvent::RunFinished { run, outcome, error, .. } if run == frozen => break (outcome, error),
            HostEvent::Failed { reason, .. } => panic!("plugin failed: {reason}"),
            _ => {}
        }
    };
    assert_eq!(outcome, "completed", "{error:?}");
    let again = runtime.signals().into_iter().rev().find(|item| item.signal.as_ref() == counts && item.generation.id() != followed.key().generation)
        .expect("the frozen run's counts");
    let again = runtime.read_latest(&again.generation, counts).unwrap();
    let cells = |dataset: &zlc_core::store::Snapshot| {
        let mut cells = std::collections::BTreeMap::new();
        for segment in dataset.segments() {
            let ([repeat, point], [repeats, points]) = (segment.origin(), segment.extent());
            let bytes = segment.data().values().as_bytes();
            let per = bytes.len() / (repeats * points);
            for r in 0..repeats { for p in 0..points { cells.insert((repeat + r, point + p), bytes[(r * points + p) * per..][..per].to_vec()); } }
        }
        cells
    };
    let (followed, again) = (cells(followed.canonical().unwrap()), cells(again.canonical().unwrap()));
    assert_eq!(followed.len(), 180);
    assert!(followed == again, "the frozen counts differ from the followed ones");
    host.shutdown();
}
