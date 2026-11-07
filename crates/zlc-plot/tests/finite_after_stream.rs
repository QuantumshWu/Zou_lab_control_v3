//! A source's next run may publish a finite dataset where the last streamed events:
//! a panel keeps what was chosen, and shows the run's records.
use std::collections::BTreeSet;
use std::sync::Arc;

use serde_json::{json, Value as Json};
use zlc_core::dataset::{DataBlock, Validity};
use zlc_core::runtime::{Bundle, Declaration, Input, Read as RuntimeRead, Runtime, Storage};
use zlc_core::schema::{Axis, Coordinates, DatasetSchema, Domain, ValidityContract, ValueSchema};
use zlc_core::{DType, OwnedBuffer};
use zlc_plot::configuration::resolve::Defaults;
use zlc_plot::document::{DocumentRead, FigureDocument};
use zlc_plot::fit::FitEngine;
use zlc_plot::interaction::gesture::InteractionDefaults;
use zlc_plot::interaction::transform::AxisTransform;
use zlc_plot::projection::Source;
use zlc_plot::render::panel::NativeRendererFactory;
use zlc_plot::session::{EngineFitLayer, RgbaFront, SessionServices};

#[derive(Clone)]
struct Read(Arc<Source>);

impl DocumentRead for Read {
    fn source(&self) -> &Arc<Source> { &self.0 }
    fn provenance_json(&self) -> String { "{}".into() }
}

const P: &str = "p";
const REPEAT: &str = "wf.voltage.repeat";
const CHANNEL: &str = "wf.voltage.0.component";
const RECORD_TIME: &str = "wf.voltage.1.readout-event";
const SAMPLES: usize = 40;
const INTERVAL: f64 = 1e-3;

fn axis(id: &str, name: &str, role: &str, coordinates: Coordinates, unit: Option<&str>) -> Arc<Axis> {
    Axis::new(id.into(), name.into(), role.into(), coordinates, unit.map(Arc::from), None, None, None).unwrap()
}

/// What Waveform Measurement declares: `repeats` records, each on two channels with
/// `samples` of them on a time axis zero at the crossing (0: a single sample).
fn schema(repeats: usize, samples: usize) -> Arc<DatasetSchema> {
    let channel = axis(CHANNEL, "channel", "component", Coordinates::Index { origin: 0, len: 2 }, None);
    let cell = if samples == 0 { Domain::dense(vec![2], vec![channel]) } else {
        let times = (0..samples).map(|index| (index as f64 - (samples / 2) as f64) * INTERVAL).collect();
        Domain::dense(vec![2, samples], vec![channel, axis(RECORD_TIME, "record time", "readout-event", Coordinates::Numeric(OwnedBuffer::from_vec(times).into()), Some("s"))])
    }.unwrap();
    let repeat = axis(REPEAT, "repeat", "repeat", Coordinates::Index { origin: 0, len: repeats }, None);
    DatasetSchema::new(Domain::dense(vec![repeats], vec![repeat]).unwrap(), Domain::new(vec![1], vec![], vec![]).unwrap(), cell,
        ValueSchema::new(DType::F32, Some("V".into()), None, ValidityContract::Value).unwrap()).unwrap()
}

/// One event: channel 0 a sine of this amplitude, channel 1 its negative, 100 times larger.
fn event(schema: &Arc<DatasetSchema>, samples: usize, amplitude: f32) -> Arc<DataBlock> {
    let wave = |index: usize| amplitude * (index as f32 * 0.3).sin();
    let span = samples.max(1);
    let values = (0..span).map(wave).chain((0..span).map(|index| -100.0 * wave(index))).collect();
    DataBlock::new(schema.clone(), OwnedBuffer::from_vec(values).into(), Validity::Uniform(true), None).unwrap()
}

/// A stream's newest event, read as a panel of this window reads it.
fn stream(runtime: &Runtime, samples: usize, window: Option<usize>) -> RuntimeRead {
    let one = schema(1, samples);
    let generation = runtime.begin("wf", vec![Declaration { signal: "voltage".into(), schema: one.clone(), storage: Storage::Latest }], json!({})).unwrap();
    let lease = window.map(|window| runtime.retain_history(&generation, "voltage", window).unwrap());
    for ordinal in 0..200 {
        let crossing_time = (samples > 0).then_some(ordinal as f64 * 0.1);
        runtime.commit(&generation, &Bundle { inputs: vec![Input::Event { signal: "voltage".into(), ordinal, data: event(&one, samples, 1.0), origin: None, shot_time: crossing_time }],
            metadata: json!({}), parents: vec![] }).unwrap();
    }
    let read = runtime.read_views(&[(generation.clone(), "voltage".into(), window)]).pop().unwrap().unwrap();
    drop(lease);
    runtime.seal(&generation, json!({})).unwrap();
    read
}

/// A finite run of five records, of amplitudes 1 to 5.
fn finite(runtime: &Runtime, samples: usize) -> RuntimeRead {
    let (one, whole) = (schema(1, samples), schema(5, samples));
    let generation = runtime.begin("wf", vec![Declaration { signal: "voltage".into(), schema: whole, storage: Storage::Finite }], json!({})).unwrap();
    for row in 0..5usize {
        runtime.commit(&generation, &Bundle { inputs: vec![Input::Event { signal: "voltage".into(), ordinal: row as u64, data: event(&one, samples, 1.0 + row as f32), origin: Some([row, 0]), shot_time: None }],
            metadata: json!({}), parents: vec![] }).unwrap();
    }
    runtime.seal(&generation, json!({})).unwrap();
    runtime.read_views(&[(generation, "voltage".into(), None)]).pop().unwrap().unwrap()
}

struct Panel { document: FigureDocument<Read>, next: i64 }

impl Panel {
    fn new() -> Self {
        let services = SessionServices { renderers: Arc::new(NativeRendererFactory), fit: Arc::new(EngineFitLayer::new(Arc::new(FitEngine::default()))),
            defaults: Arc::new(Defaults::default()), registry: None, interaction: InteractionDefaults::default() };
        let mut panel = Self { document: FigureDocument::new(services, None), next: 0 };
        panel.execute(json!({"action": "ready", "dpr": 1.0}), None);
        panel
    }

    fn execute(&mut self, mut command: Json, read: Option<&RuntimeRead>) -> (Json, Option<RgbaFront>) {
        self.next += 1;
        command["id"] = json!(self.next);
        let reads = read.map(|read| (P.to_owned(), Read(Arc::new(Source::from_read(read).unwrap())))).into_iter().collect();
        let mut execution = self.document.execute(&command.to_string(), reads, None, None).unwrap();
        let all: BTreeSet<String> = [P.to_owned()].into();
        self.document.finish_presentation(&all, &all);
        assert!(execution.state["error"].is_null(), "{}", execution.state["error"]);
        (execution.state["panels"][0].clone(), execution.fronts.remove(P))
    }

    fn show(&mut self, kind: &str, epoch: u64, target: Json, read: &RuntimeRead) -> (Json, Option<RgbaFront>) {
        let shown = self.execute(json!({"action": "sync_panels", "inputs": [{"panel": P, "kind": kind, "window": 100, "signal": "voltage", "epoch": epoch}], "targets": {P: target}}), Some(read));
        assert!(shown.0["error"].is_null(), "{}", shown.0["error"]);
        shown
    }

    /// The axes that hold the data: a rolling plot's history, any other's main.
    fn main_axes(&self) -> AxisTransform {
        let session = self.document.session(P).expect("a session");
        let axes = session.lock().unwrap().raster_axes_snapshot().unwrap();
        axes.into_iter().find(|item| item.role == "main" || item.role == "history").expect("main axes")
    }
}

fn rows(panel: &Json) -> Vec<(String, Json)> {
    panel["controls"].as_array().unwrap().iter().take_while(|field| field["name"] != "title")
        .map(|field| (field["name"].as_str().unwrap().to_owned(), field["value"].clone())).collect()
}

fn triggered_preview() -> Json {
    json!({"semantic": {"kind": "curve", format!("fate:cell_data:{RECORD_TIME}"): "x",
        format!("fate:cell_data:{CHANNEL}"): ["scope-value", 0], format!("fate:repeat:{REPEAT}"): "group"}, "size": "2x2"})
}

#[test]
fn triggered_curve_uses_time_within_a_record_even_when_the_window_has_crossing_times() {
    let read = stream(&Runtime::new(), SAMPLES, Some(4));
    let mut panel = Panel::new();
    let (state, _) = panel.show("curve", 1, triggered_preview(), &read);
    assert!(rows(&state).iter().any(|(name, value)| name == &format!("fate:cell_data:{RECORD_TIME}") && value == "x"));
    let (low, high) = panel.main_axes().x_limits;
    assert!((low + 0.020).abs() < 1e-9 && (high - 0.019).abs() < 1e-9, "{low}..{high}");
}

/// Repeat 0 to repeat 5 with the trigger on: the same rows with the same choices, the
/// time axis from the crossing, and five traces one on the other.
#[test]
fn a_triggered_panel_keeps_its_rows_and_draws_each_record() {
    let runtime = Runtime::new();
    let mut panel = Panel::new();
    let live = stream(&runtime, SAMPLES, None);
    let (state, _) = panel.show("curve", 1, triggered_preview(), &live);
    let streamed = rows(&state);
    assert!(streamed.iter().any(|(name, value)| name == &format!("fate:cell_data:{RECORD_TIME}") && value == "x"), "{streamed:?}");
    let first = panel.main_axes();
    assert!((first.x_limits.0 + 0.020).abs() < 1e-9 && first.y_limits.1 < 2.0, "{:?} {:?}", first.x_limits, first.y_limits);

    let whole = finite(&runtime, SAMPLES);
    let (state, _) = panel.show("curve", 2, state["target"].clone(), &whole);
    assert_eq!(rows(&state), streamed, "every row, with what was chosen on it");
    let axes = panel.main_axes();
    assert!((axes.x_limits.0 + 0.020).abs() < 1e-9, "{:?}", axes.x_limits);
    assert!(axes.y_limits.1 >= 4.9, "the record of amplitude 5 is drawn, not the mean of them: {:?}", axes.y_limits);
}

/// A finite run with fewer records than the window is wide fills the frame.
#[test]
fn a_rolling_panel_frames_a_finite_run_by_its_records() {
    let runtime = Runtime::new();
    let mut panel = Panel::new();
    let target = json!({"semantic": {"kind": "rolling", format!("fate:cell_data:{CHANNEL}"): ["scope-value", 0]}, "size": "2x2"});
    let live = stream(&runtime, 0, Some(100));
    let (state, _) = panel.show("rolling", 1, target, &live);
    assert_eq!(panel.main_axes().x_limits, (-99.0, 0.0), "a stream's frame is the window");
    let whole = finite(&runtime, 0);
    panel.show("rolling", 2, state["target"].clone(), &whole);
    assert_eq!(panel.main_axes().x_limits, (-4.0, 0.0));
}

#[test]
fn a_waiting_panels_portable_choices_apply_to_its_first_available_signal() {
    let mut panel = Panel::new();
    let old = json!([{"panel":P,"signal":"old","kind":"curve","window":1,"epoch":1,"waiting":true,"reason":"No data yet"}]);
    let target = json!({"semantic":{"kind":"curve", "fate:cell_data:old.time":"x",
        format!("fate:cell_data:{CHANNEL}"):"group"}, "size":"2x2"});
    panel.execute(json!({"action":"sync_panels","inputs":old,"targets":{P:target}}), None);
    panel.execute(json!({"action":"configure","panel":P,"inputs":old,"semantic":{"reduction":"max"}}), None);
    let read = finite(&Runtime::new(), SAMPLES);
    let (shown, _) = panel.show("curve", 2, target, &read);
    let controls = rows(&shown);
    for (name, value) in [("reduction".into(), "max"), (format!("fate:cell_data:{CHANNEL}"), "group"),
                          (format!("fate:cell_data:{RECORD_TIME}"), "x")] {
        assert!(controls.contains(&(name, json!(value))), "{controls:?}");
    }
    let handle = panel.document.session(P).unwrap();
    let session = handle.lock().unwrap();
    let zlc_plot::projection::Payload::Curve(curve) = session.projection().payload().as_ref() else { panic!("expected a curve") };
    assert_eq!(curve.series.len(), 2);
    let peak = curve.series[0].y.display.to_f64().into_iter().fold(f64::NEG_INFINITY, f64::max);
    assert!(peak > 4.9 && peak <= 5.0, "the first frame uses Max over the five records: {peak}");
}
