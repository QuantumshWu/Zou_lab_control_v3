//! Pulse-timeline sessions against the recorded pulse-ops oracle
//! (`tests/gen/render_capture.py --pulse`): every op of every scenario is
//! applied to a native `PlotSession::pulse_timeline` drawing through the
//! native renderer, and after each one the front (shape and sha256), the
//! axes it describes (`_axis_wire`), its viewport, display parameters and
//! size are compared with what the Python session answered.
//!
//! Env-gated: `ZLC_PULSE_ORACLE` = the capture output directory;
//! `ZLC_PULSE_BENCH` = passes per scenario for timing every op.
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use serde_json::{json, Value as Json};
use zlc_plot::configuration::resolve::Defaults;
use zlc_plot::configuration::{PlotLabels, Value, ValueMap};
use zlc_plot::document::{axis_from_wire, axis_wire};
use zlc_plot::fit::FitEngine;
use zlc_plot::interaction::gesture::InteractionDefaults;
use zlc_plot::pulse::PulseTimelineData;
use zlc_plot::render::panel::NativeRendererFactory;
use zlc_plot::session::{EngineFitLayer, PlotSession, PointerAction, PointerEvent, SessionServices};

fn services() -> SessionServices {
    SessionServices {
        renderers: Arc::new(NativeRendererFactory), fit: Arc::new(EngineFitLayer::new(Arc::new(FitEngine::default()))),
        defaults: Arc::new(Defaults::default()), registry: None, interaction: InteractionDefaults::default(),
    }
}

fn sha256(bytes: &[u8]) -> String {
    use sha2::Digest;
    sha2::Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

/// One op applied to the session (`pulse.PulseRunner.apply`).
fn apply(session: &mut Option<PlotSession>, op: &Json, data: &BTreeMap<String, Arc<PulseTimelineData>>, previous_axes: &[Json]) {
    let name = op["op"].as_str().unwrap();
    let values = |doc: &Json| -> Option<ValueMap> { doc.as_object().map(|map| map.iter().map(|(k, v)| (k.as_str(), Value::from_json(v))).collect()) };
    if name == "create" {
        if let Some(old) = session.as_mut() { old.close(); }
        let created = PlotSession::pulse_timeline(data[op["data"].as_str().unwrap()].clone(), PlotLabels::default(), op["size"].as_str(),
                                                  values(&op["parameters"]).as_ref(), op["dpr"].as_f64().unwrap_or(1.0), None, services());
        *session = Some(created.expect("create"));
        return;
    }
    let s = session.as_mut().expect("an open session");
    let outcome = match name {
        "set_parameters" => s.configure_parameters(values(&op["values"]).unwrap()),
        "set_size" => s.set_size(op["preset"].as_str().unwrap()),
        "set_device_pixel_ratio" => s.set_device_pixel_ratio(op["ratio"].as_f64().unwrap()),
        "update_data" => s.update_pulse(data[op["data"].as_str().unwrap()].clone(), op["revision"].as_u64()),
        "set_x_limits" => s.set_x_limits(op["low"].as_f64().unwrap(), op["high"].as_f64().unwrap()),
        "reset_viewport" => s.reset_viewport(),
        "pointer" => {
            let axes = op["axes_index"].as_u64().map(|index| axis_from_wire(&previous_axes[index as usize]).expect("axes wire"));
            let event = PointerEvent {
                action: PointerAction::parse(op["event"].as_str().unwrap()).unwrap(), x: op["x"].as_f64().unwrap(), y: op["y"].as_f64().unwrap(),
                button: op["button"].as_u64().map(|b| b as u8), double: op["double"].as_bool().unwrap_or(false),
                step: op["step"].as_f64().unwrap_or(0.0), key: op["key"].as_str().map(str::to_owned), axes, time: None,
            };
            s.raster_pointer_event(&event).map(|_| ())
        }
        "close" => { s.close(); *session = None; Ok(()) }
        other => panic!("unknown pulse op {other}"),
    };
    outcome.unwrap_or_else(|error| panic!("{name}: {error}"));
}

/// What the session answers after an op (`PulseRunner.apply`'s result).
fn answer(session: &mut PlotSession) -> (Json, Json) {
    let front = session.rgba().expect("front");
    let axes: Vec<Json> = session.raster_axes_snapshot().expect("axes").iter().map(axis_wire).collect();
    let viewport = session.viewport().map_or(Json::Null, |(x, y)| json!([x.map(|x| json!([x.low, x.high])), y.map(|y| json!([y.low, y.high]))]));
    let state = json!({"axes": axes, "viewport": viewport, "parameters": Value::Map(session.display_state().values.clone()).to_json(),
                       "size": session.size()});
    (json!([[front.height, front.width, 4], sha256(front.pixels.as_slice())]), state)
}

#[test]
fn pulse_sessions_match_the_recorded_oracle() {
    let Ok(root) = std::env::var("ZLC_PULSE_ORACLE") else { return };
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&root).unwrap().map(|e| e.unwrap().path()).filter(|p| p.join("capture.json").exists()).collect();
    dirs.sort();
    let (mut compared, mut fronts, mut states) = (0usize, 0usize, 0usize);
    let mut timings: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut phases: BTreeMap<String, BTreeMap<&'static str, Vec<f64>>> = BTreeMap::new();
    let passes: usize = std::env::var("ZLC_PULSE_BENCH").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    for dir in dirs {
        let document: Json = serde_json::from_slice(&std::fs::read(dir.join("capture.json")).unwrap()).unwrap();
        if document["format"] != "pulse-ops" { continue; }
        let name = dir.file_name().unwrap().to_string_lossy().to_string();
        let data: BTreeMap<String, Arc<PulseTimelineData>> = document["data"].as_object().unwrap().iter()
            .map(|(key, tree)| (key.clone(), PulseTimelineData::from_json(tree).expect("timeline"))).collect();
        let ops = document["ops"].as_array().unwrap();
        for pass in 0..passes {
            let mut session: Option<PlotSession> = None;
            let mut previous_axes: Vec<Json> = Vec::new();
            for (number, op) in ops.iter().enumerate() {
                // Timed: the op and the front it leaves (what a host waits for).
                let started = std::time::Instant::now();
                apply(&mut session, op, &data, &previous_axes);
                let Some(s) = session.as_mut() else { continue };
                std::hint::black_box(s.rgba().expect("front"));
                let elapsed = started.elapsed().as_secs_f64() * 1e3;
                // Each op's renderer phases (`ZLC_RENDER_PROFILE`): the median over the timed passes.
                for (phase, ms, _) in zlc_plot::render::profile::report() {
                    if pass > 0 { phases.entry(format!("{name} {:05} {}", number + 1, op["op"].as_str().unwrap())).or_default().entry(phase).or_default().push(ms); }
                }
                let (front, state) = answer(s);
                if pass > 0 { timings.entry(format!("{name} {:05} {}", number + 1, op["op"].as_str().unwrap())).or_default().push(elapsed); }
                previous_axes = state["axes"].as_array().unwrap().clone();
                if pass > 0 { continue; }
                let expected = &op["result"];
                compared += 1;
                if front == expected["front"] { fronts += 1; } else {
                    eprintln!("~~~ {name} {:05} {}: front {front} != {}", number + 1, op["op"], expected["front"]);
                }
                let same = ["axes", "viewport", "parameters", "size"].iter().all(|key| state[*key] == expected[*key]);
                if same { states += 1; } else {
                    for key in ["axes", "viewport", "parameters", "size"] {
                        if state[key] != expected[key] { eprintln!("--- {name} {:05} {} {key}: native {} python {}", number + 1, op["op"], state[key], expected[key]); }
                    }
                }
            }
        }
    }
    for (key, times) in &timings {
        let mut sorted = times.clone();
        sorted.sort_by(f64::total_cmp);
        eprintln!("TIME {key} median {:.3} min {:.3} ms", sorted[sorted.len() / 2], sorted[0]);
    }
    for (key, table) in &phases {
        let parts: Vec<String> = table.iter().map(|(phase, ms)| {
            let mut ms = ms.clone();
            ms.sort_by(f64::total_cmp);
            format!("{phase} {:.2}", ms[ms.len() / 2])
        }).collect();
        eprintln!("PHASES {key}: {}", parts.join(", "));
    }
    eprintln!("pulse ops compared {compared}: fronts identical {fronts}, session answers identical {states}");
    assert_eq!((fronts, states), (compared, compared), "pulse session mismatches");
}
