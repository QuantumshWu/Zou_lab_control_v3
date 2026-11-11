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
fn shot(cell: &[usize], values: Vec<f64>, sigma: Option<f64>) -> Read {
    let index = |id: &str, role: &str, len: usize| Axis::new(id.into(), id.into(), role.into(), Coordinates::Index { origin: 0, len }, None, None, None, None).unwrap();
    let axes = match cell {
        [x] => vec![index("camera.x", "spatial-x", *x)],
        [y, x] => vec![index("camera.y", "spatial-y", *y), index("camera.x", "spatial-x", *x)],
        _ => unreachable!(),
    };
    let schema = DatasetSchema::new(Domain::dense(vec![1], vec![index("camera.repeat", "repeat", 1)]).unwrap(), Domain::dense(vec![1], vec![index("camera.frame", "readout-event", 1)]).unwrap(),
        Domain::dense(cell.to_vec(), axes).unwrap(), ValueSchema::new(DType::F64, None, None, ValidityContract::Value).unwrap()).unwrap();
    let errors = sigma.map(|sigma| OwnedBuffer::from_vec(vec![sigma; values.len()]));
    let block = DataBlock::new(schema.clone(), OwnedBuffer::from_vec(values).into(), Validity::Uniform(true), errors).unwrap();
    let snapshot = DatasetStore::new(schema.clone()).commit(vec![Placement::new([0, 0], [1, 1], block).unwrap()]).unwrap();
    let runtime = Runtime::new();
    let generation = runtime.begin("fixture", vec![Declaration { signal: "shot".into(), schema: snapshot.schema().clone(), storage: Storage::Snapshot }], json!({})).unwrap();
    runtime.commit(&generation, &Bundle { inputs: vec![Input::Dataset { signal: "shot".into(), ordinal: None, snapshot }], metadata: json!({}), parents: vec![] }).unwrap();
    let read = runtime.read_coherent_views(&[(generation, "shot".into(), None)], &mut Default::default()).pop().unwrap().unwrap();
    Read(Arc::new(Source::from_read(&read).unwrap()))
}

const P: &str = "p";

fn document() -> FigureDocument<Read> {
    let services = SessionServices { renderers: Arc::new(NativeRendererFactory), fit: Arc::new(EngineFitLayer::new(Arc::new(FitEngine::default()))),
        defaults: Arc::new(Defaults::default()), registry: None, interaction: InteractionDefaults::default() };
    FigureDocument::new(services, None)
}

/// A panel drawn from `read` with the first frame's front and its axes.
fn draw(read: Read, kind: &str, parameters: Json) -> (RgbaFront, Vec<AxisTransform>) {
    let mut document = document();
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
    let (front, axes) = draw(shot(&[64, 64], checker(100.0, 200.0), None), "image", json!({"relim_mode": "tight", "colormap": "gray"}));
    assert_eq!(extremes(&front, main_axes(&axes, "image")), (0, 255), "the data's extremes are the colormap's ends");
    let (front, axes) = draw(shot(&[64, 64], checker(-200.0, -100.0), None), "image", json!({"relim_mode": "normal", "colormap": "gray"}));
    let (low, high) = extremes(&front, main_axes(&axes, "image"));
    assert!(low > 0 && high < 255, "normal keeps its padding: {low} {high}");
}

#[test]
fn a_constant_tight_image_draws() {
    let (front, axes) = draw(shot(&[32, 32], vec![5.0; 32 * 32], None), "image", json!({"relim_mode": "tight", "colormap": "gray"}));
    let (low, high) = extremes(&front, main_axes(&axes, "image"));
    assert_eq!(low, high, "one value is one colour");
}

#[test]
fn curve_limits_follow_values_when_error_bars_change() {
    for (mode, expected) in [("tight", (3.0, 7.0)), ("normal", (0.0, 7.0 * 1.2))] {
        let render = |sigma| {
            let data = shot(&[5], vec![5.0, 3.0, 5.0, 7.0, 5.0], Some(sigma));
            draw(data, "curve", json!({"relim_mode": mode, "uncertainty": true}))
        };
        let (narrow, narrow_axes) = render(0.1);
        let (wide, wide_axes) = render(20.0);
        assert_eq!(main_axes(&narrow_axes, "main").y_limits, expected, "{mode}");
        assert_eq!(main_axes(&wide_axes, "main").y_limits, expected, "{mode} with bars crossing zero");
        assert!(narrow.pixels.as_slice() != wide.pixels.as_slice(), "{mode} should draw the wider bars");
    }
}

#[test]
fn a_tight_histogram_reaches_its_peak() {
    let values: Vec<f64> = (0..400).map(|i| (i % 20) as f64 * (i % 7) as f64).collect();
    let (_, axes) = draw(shot(&[400], values.clone(), None), "histogram", json!({"relim_mode": "tight", "x_relim_mode": "tight", "bin_count": 20}));
    let tight = main_axes(&axes, "main").y_limits;
    let (_, axes) = draw(shot(&[400], values, None), "histogram", json!({"relim_mode": "normal", "x_relim_mode": "tight", "bin_count": 20}));
    let normal = main_axes(&axes, "main").y_limits;
    assert_eq!(tight.0, 0.0);
    assert_eq!(normal.1, tight.1 * 1.08, "the same bins, one with headroom");
}

#[test]
fn switching_a_panels_signal_applies_its_auto_mode_on_the_first_frame() {
    for (kind, mode) in [("curve", "tight"), ("curve", "normal"), ("image", "tight")] {
        let mut document = document();
        let panels = BTreeSet::from([P.to_owned()]);
        document.execute(r#"{"id":1,"action":"ready","dpr":1}"#, Default::default(), None, None).unwrap();
        document.finish_presentation(&panels, &panels);
        // Another frame of A, a sibling signal, a restart of B, then a run
        // whose first valid values arrive after an entirely empty frame.
        let frames = [("A","run1",1.,10.),("A","run1",1.,9.),("B","run1",1.,9.),("B","run2",1.,8.),
                      ("B","run3",f64::NAN,f64::NAN),("B","run3",0.2,0.8)];
        for (step, (signal, generation, low, high)) in frames.into_iter().enumerate() {
            let epoch = if signal == "A" { 1 } else { 2 };
            let mut command = json!({"id":step+2,"action":"sync_panels","inputs":[{"panel":P,"kind":kind,"window":1,"signal":signal,"epoch":epoch}]});
            if step == 0 {
                let mut parameters = json!({"relim_mode":mode});
                if kind == "image" { parameters["presentation"] = json!("height_bars"); }
                command["targets"] = json!({P: {"semantic":{"kind":kind},"parameters":parameters}});
            }
            let mut read = if kind == "image" { shot(&[2,2],vec![low,high,low,high],None) } else { shot(&[2],vec![low,high],Some(20.)) };
            let source = Arc::get_mut(&mut read.0).unwrap();
            source.generation = generation.into();
            source.block_id = format!("canonical:{signal}").into();
            source.revision = if matches!(step, 1 | 5) { 2 } else { 1 };
            let frame = document.execute(&command.to_string(), [(P.to_owned(),read)].into(), None, None).unwrap();
            document.finish_presentation(&panels,&panels);
            assert!(frame.state["panels"][0]["error"].is_null(), "{}",frame.state);
            let handle = document.session(P).unwrap();
            let session = handle.lock().unwrap();
            let retained_high = if step == 1 { 10. } else { high };
            let (shown, expected) = if kind == "image" {
                let margin = 0.1*(retained_high-low);
                (session.renderer().resolved_color_limits().unwrap(), (low-margin, retained_high+margin))
            } else {
                let axes = session.raster_axes_snapshot().unwrap();
                (main_axes(&axes,"main").y_limits, if mode == "tight" { (low,high) } else { (0.,retained_high*1.2) })
            };
            assert_eq!(shown, if high.is_nan() { (0.,1.) } else { expected }, "{kind}/{mode} step {step}");
        }
    }
}

#[test]
fn a_new_slice_of_the_same_signal_gets_its_own_normal_range() {
    let mut document = document();
    let panels = BTreeSet::from([P.to_owned()]);
    let read = shot(&[2, 2], vec![1.,10.,1.,9.], None);
    let scope = "fate:cell_data:camera.y";
    let commands = [
        json!({"action":"ready","dpr":1}),
        json!({"action":"sync_panels","inputs":[{"panel":P,"kind":"curve","window":1,"signal":"shot","epoch":1}],
            "targets":{P:{"semantic":{"kind":"curve","fate:cell_data:camera.x":"x",scope:["scope-value",0]},"parameters":{"relim_mode":"normal"}}}}),
        json!({"action":"configure","panel":P,"semantic":{scope:["scope-value",1]}}),
    ];
    for (step, mut command) in commands.into_iter().enumerate() {
        command["id"] = json!(step+1);
        let frame = document.execute(&command.to_string(), [(P.to_owned(),read.clone())].into(), None, None).unwrap();
        document.finish_presentation(&panels, &panels);
        assert!(frame.state["error"].is_null() && frame.state["panels"][0]["error"].is_null(), "{}", frame.state);
        if step > 0 {
            let axes = document.session(P).unwrap().lock().unwrap().raster_axes_snapshot().unwrap();
            assert_eq!(main_axes(&axes,"main").y_limits, (0.,if step == 1 { 12. } else { 9.*1.2 }));
        }
    }
}
