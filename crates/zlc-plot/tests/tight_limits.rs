//! Tight limits are the data's own, through the document path the app takes.
use std::collections::BTreeSet;
use std::sync::Arc;

use serde_json::{json, Value as Json};
use zlc_core::dataset::{DataBlock, Validity};
use zlc_core::runtime::{Bundle, Declaration, Input, Runtime, Storage};
use zlc_core::schema::{Axis, Coordinates, DatasetSchema, Domain, ValidityContract, ValueSchema};
use zlc_core::store::{DatasetStore, Placement};
use zlc_core::{DType, OwnedBuffer};
use zlc_plot::configuration::resolve::Defaults;
use zlc_plot::document::{DocumentRead, FigureDocument};
use zlc_plot::fit::FitEngine;
use zlc_plot::interaction::gesture::InteractionDefaults;
use zlc_plot::interaction::transform::AxisTransform;
use zlc_plot::projection::Source;
use zlc_plot::render::limits::{autoscaled_limits, tight_limits};
use zlc_plot::render::panel::NativeRendererFactory;
use zlc_plot::session::{EngineFitLayer, RgbaFront, SessionServices};

#[derive(Clone)]
struct Read(Arc<Source>);

impl DocumentRead for Read {
    fn source(&self) -> &Arc<Source> { &self.0 }
    fn provenance_json(&self) -> String { "{}".into() }
}

/// One shot whose cell is `cell` samples of `values`, on spatial axes.
fn shot(cell: &[usize], values: Vec<f64>) -> Read {
    let index = |id: &str, role: &str, len: usize| Axis::new(id.into(), id.into(), role.into(), Coordinates::Index { origin: 0, len }, None, None, None, None).unwrap();
    let axes = match cell {
        [x] => vec![index("camera.x", "spatial-x", *x)],
        [y, x] => vec![index("camera.y", "spatial-y", *y), index("camera.x", "spatial-x", *x)],
        _ => unreachable!(),
    };
    let schema = DatasetSchema::new(Domain::dense(vec![1], vec![index("camera.repeat", "repeat", 1)]).unwrap(), Domain::dense(vec![1], vec![index("camera.frame", "readout-event", 1)]).unwrap(),
        Domain::dense(cell.to_vec(), axes).unwrap(), ValueSchema::new(DType::F64, None, None, ValidityContract::Value).unwrap()).unwrap();
    let block = DataBlock::new(schema.clone(), OwnedBuffer::from_vec(values).into(), Validity::Uniform(true), None).unwrap();
    let snapshot = DatasetStore::new(schema.clone()).commit(vec![Placement::new([0, 0], [1, 1], block).unwrap()]).unwrap();
    let runtime = Runtime::new();
    let generation = runtime.begin("fixture", vec![Declaration { signal: "shot".into(), schema: snapshot.schema().clone(), storage: Storage::Snapshot }], json!({})).unwrap();
    runtime.commit(&generation, &Bundle { inputs: vec![Input::Dataset { signal: "shot".into(), ordinal: None, snapshot }], metadata: json!({}), parents: vec![] }).unwrap();
    let read = runtime.read_coherent_views(&[(generation, "shot".into(), None)], &mut Default::default()).pop().unwrap().unwrap();
    Read(Arc::new(Source::from_read(&read).unwrap()))
}

const P: &str = "p";

/// A panel drawn from `read` with the first frame's front and its axes.
fn draw(read: Read, kind: &str, parameters: Json) -> (RgbaFront, Vec<AxisTransform>) {
    let services = SessionServices { renderers: Arc::new(NativeRendererFactory), fit: Arc::new(EngineFitLayer::new(Arc::new(FitEngine::default()))),
        defaults: Arc::new(Defaults::default()), registry: None, interaction: InteractionDefaults::default() };
    let mut document = FigureDocument::new(services, None);
    let all: BTreeSet<String> = [P.to_owned()].into();
    let mut front = None;
    let target = json!({"semantic": {"kind": kind}, "size": "2x2", "parameters": parameters});
    for (id, mut command) in [json!({"action": "ready", "dpr": 1.0}),
                              json!({"action": "sync_panels", "inputs": [{"panel": P, "kind": kind, "window": 1, "signal": "shot", "epoch": 1}], "targets": {P: target}})].into_iter().enumerate() {
        command["id"] = json!(id + 1);
        let mut execution = document.execute(&command.to_string(), [(P.to_owned(), read.clone())].into(), None, None).unwrap();
        document.finish_presentation(&all, &all);
        assert!(execution.state["error"].is_null(), "{}", execution.state["error"]);
        assert!(execution.state["panels"][0]["error"].is_null(), "{}", execution.state["panels"][0]["error"]);
        front = execution.fronts.remove(P).or(front);
    }
    let handle = document.session(P).expect("a session");
    let axes = handle.lock().unwrap().raster_axes_snapshot().unwrap();
    (front.expect("a painted front"), axes)
}

fn main_axes<'a>(axes: &'a [AxisTransform], role: &str) -> &'a AxisTransform {
    axes.iter().find(|item| item.role == role).unwrap_or_else(|| panic!("no {role} axes among {:?}", axes.iter().map(|item| &item.role).collect::<Vec<_>>()))
}

/// The darkest and brightest grey in the middle of an axes' box, away from its ticks and frame.
fn extremes(front: &RgbaFront, axes: &AxisTransform) -> (u8, u8) {
    let (width, height) = (front.width as usize, front.height as usize);
    let (left, top, right, bottom) = axes.bounds;
    let (mut low, mut high) = (u8::MAX, u8::MIN);
    for y in ((3.0 * top + bottom) / 4.0 * height as f64) as usize..((top + 3.0 * bottom) / 4.0 * height as f64) as usize {
        for x in ((3.0 * left + right) / 4.0 * width as f64) as usize..((left + 3.0 * right) / 4.0 * width as f64) as usize {
            let red = front.pixels.as_slice()[(y * width + x) * 4];
            low = low.min(red);
            high = high.max(red);
        }
    }
    (low, high)
}

#[test]
fn the_limits_are_the_datas_and_a_single_value_still_has_a_span() {
    assert_eq!(tight_limits((3.0, 7.0), 0.1), (3.0, 7.0));
    assert_eq!(tight_limits((5.0, 5.0), 0.1), (4.5, 5.5));
    assert_eq!(tight_limits((0.0, 0.0), 0.1), (-0.1, 0.1));
    // Normal keeps its padding, zero base and retention.
    assert_eq!(autoscaled_limits((3.0, 7.0), None, 0.1, 0.35, true, true), (0.0, 7.0 * 1.2));
    assert_eq!(autoscaled_limits((-2.0, 2.0), None, 0.1, 0.35, true, true), (-2.4, 2.4));
    assert_eq!(autoscaled_limits((1.0, 2.0), Some((0.5, 2.5)), 0.1, 0.35, false, true), (0.5, 2.5));
}

#[test]
fn a_tight_image_uses_the_whole_colormap() {
    let checker = |low: f64, high: f64| (0..64 * 64).map(|i| if (i % 64 + i / 64) % 2 == 0 { low } else { high }).collect::<Vec<f64>>();
    let (front, axes) = draw(shot(&[64, 64], checker(100.0, 200.0)), "image", json!({"relim_mode": "tight", "colormap": "gray"}));
    assert_eq!(extremes(&front, main_axes(&axes, "image")), (0, 255), "the data's extremes are the colormap's ends");
    let (front, axes) = draw(shot(&[64, 64], checker(-200.0, -100.0)), "image", json!({"relim_mode": "normal", "colormap": "gray"}));
    let (low, high) = extremes(&front, main_axes(&axes, "image"));
    assert!(low > 0 && high < 255, "normal keeps its padding: {low} {high}");
}

#[test]
fn a_constant_tight_image_draws() {
    let (front, axes) = draw(shot(&[32, 32], vec![5.0; 32 * 32]), "image", json!({"relim_mode": "tight", "colormap": "gray"}));
    let (low, high) = extremes(&front, main_axes(&axes, "image"));
    assert_eq!(low, high, "one value is one colour");
}

#[test]
fn a_tight_curve_stands_on_its_extremes() {
    let values: Vec<f64> = (0..50).map(|i| 3.0 + (i as f64 * 0.3).sin()).collect();
    let (low, high) = values.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(low, high), v| (low.min(*v), high.max(*v)));
    let (_, axes) = draw(shot(&[50], values.clone()), "curve", json!({"relim_mode": "tight"}));
    assert_eq!(main_axes(&axes, "main").y_limits, (low, high));
    let (_, axes) = draw(shot(&[50], values), "curve", json!({"relim_mode": "normal"}));
    assert_eq!(main_axes(&axes, "main").y_limits.0, 0.0, "normal is zero based");
}

#[test]
fn a_tight_histogram_reaches_its_peak() {
    let values: Vec<f64> = (0..400).map(|i| (i % 20) as f64 * (i % 7) as f64).collect();
    let (_, axes) = draw(shot(&[400], values.clone()), "histogram", json!({"relim_mode": "tight", "x_relim_mode": "tight", "bin_count": 20}));
    let tight = main_axes(&axes, "main").y_limits;
    let (_, axes) = draw(shot(&[400], values), "histogram", json!({"relim_mode": "normal", "x_relim_mode": "tight", "bin_count": 20}));
    let normal = main_axes(&axes, "main").y_limits;
    assert_eq!(tight.0, 0.0);
    assert_eq!(normal.1, tight.1 * 1.08, "the same bins, one with headroom");
}
