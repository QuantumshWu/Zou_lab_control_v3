//! Histograms of camera frames read from the Runtime as the desktop host
//! reads them, against the Python projection of the same frames
//! (`FitProjection`, `HistogramPlot()`, 60 bins).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use serde_json::json;
use zlc_core::dataset::{DataBlock, Validity};
use zlc_core::runtime::{Bundle, Declaration, Generation, Input, Read, Runtime, Storage};
use zlc_core::schema::{Axis, Coordinates, DatasetSchema, Domain, ValidityContract, ValueSchema};
use zlc_core::{AnyBuffer, OwnedBuffer};
use zlc_plot::document::{DocumentRead, FigureDocument};
use zlc_plot::projection::output::Payload;
use zlc_plot::projection::plan::{Projection, ProjectionPlan};
use zlc_plot::projection::Source;

const SHAPE: [usize; 2] = [1200, 1920];
const N: i64 = 384000;

/// The host's camera schema (`zlc-desktop` `camera_schema`).
fn camera_schema(shape: [usize; 2], dtype: zlc_core::DType) -> Arc<DatasetSchema> {
    let axis = |id: &str, role: &str, coordinates, spatial: bool| Axis::new(id.into(), role.into(), role.into(), coordinates,
        spatial.then(|| "pixel".into()), spatial.then(|| "sensor_pixel_xy".into()), None, None).unwrap();
    let one = |id: &str, role: &str| Domain::dense(vec![1], vec![axis(id, role, Coordinates::Index { origin: 0, len: 1 }, false)]).unwrap();
    let pixels = ["spatial-y", "spatial-x"].into_iter().enumerate().map(|(index, role)| axis(&format!("camera.{role}"), role,
        Coordinates::Numeric(OwnedBuffer::from_vec((0..shape[index] as i64).collect::<Vec<_>>()).into()), true)).collect();
    DatasetSchema::new(one("camera.repeat", "repeat"), one("camera.frame", "readout-event"), Domain::dense(shape.to_vec(), pixels).unwrap(),
        ValueSchema::new(dtype, Some("count".into()), None, ValidityContract::Value).unwrap()).unwrap()
}

fn begin(runtime: &Runtime, schema: Arc<DatasetSchema>) -> Generation {
    runtime.begin("camera", vec![Declaration { signal: "frames".into(), schema, storage: Storage::Latest }], json!({})).unwrap()
}

/// Commit one frame as the camera's latest event and read it back.
fn publish(runtime: &Runtime, generation: &Generation, shape: [usize; 2], values: AnyBuffer) -> Read {
    let data = DataBlock::new(camera_schema(shape, values.dtype()), values, Validity::Uniform(true), None).unwrap();
    runtime.commit(generation, &Bundle { inputs: vec![Input::Event { signal: "frames".into(), ordinal: 0, data, origin: None }],
        metadata: json!({}), parents: vec![] }).unwrap();
    runtime.read_coherent_views(&[(generation.clone(), Arc::from("frames"), None)], &mut Default::default()).pop().unwrap().unwrap()
}

/// Six equal planes of values {0, 1, 3, 4, 6, 7} times `scale` plus `offset`.
fn pattern(scale: f64, offset: f64) -> Vec<f64> {
    (0..SHAPE[0] * SHAPE[1]).map(|i| ((i % SHAPE[1]) / 640 * 3 + (i / SHAPE[1]) % 2) as f64 * scale + offset).collect()
}

/// Nonzero bins as `(left, right, count)`.
fn bars(payload: &Payload) -> Vec<(f64, f64, i64)> {
    let Payload::Histogram(histogram) = payload else { panic!("not a histogram") };
    let edges = histogram.edges.canonical.to_f64();
    histogram.counts.slice::<i64>().iter().enumerate().filter(|(_, c)| **c != 0).map(|(i, c)| (edges[i], edges[i + 1], *c)).collect()
}

/// Every pixel of a frame of each dtype the camera delivers is counted once,
/// through both runtime read paths (the host's `Source::from_read` and the
/// carried `project_read`), into Python's bins.
#[test]
fn camera_frames_bin_like_python() {
    let u8s = |v: Vec<f64>| AnyBuffer::from(OwnedBuffer::from_vec(v.into_iter().map(|x| x as u8).collect::<Vec<_>>()));
    let u16s = |v: Vec<f64>| AnyBuffer::from(OwnedBuffer::from_vec(v.into_iter().map(|x| x as u16).collect::<Vec<_>>()));
    let i16s = |v: Vec<f64>| AnyBuffer::from(OwnedBuffer::from_vec(v.into_iter().map(|x| x as i16).collect::<Vec<_>>()));
    let f32s = |v: Vec<f64>| AnyBuffer::from(OwnedBuffer::from_vec(v.into_iter().map(|x| x as f32).collect::<Vec<_>>()));
    let small = |scale: f64, offset: f64| (0..24).map(|i| i as f64 * scale + offset).collect::<Vec<_>>();
    // The 6x4 frame (24 single-sample bins: the first and the last) and the
    // camera frame with Python's bars.
    let cases = [
        ("u8", u8s(small(10.0, 0.0)), [(-1.5, 3.5), (228.5, 233.5)], u8s(pattern(1.0, 0.0)),
         vec![(-0.5, 0.5, N), (0.5, 1.5, N), (2.5, 3.5, N), (3.5, 4.5, N), (5.5, 6.5, N), (6.5, 7.5, N)]),
        ("u16", u16s(small(1000.0, 7.0)), [(-299.5, 122.5), (22910.5, 23332.5)], u16s(pattern(1000.0, 7.0)),
         vec![(-85.5, 43.5, N), (946.5, 1075.5, N), (2881.5, 3010.5, N), (3913.5, 4042.5, N), (5977.5, 6106.5, N), (6880.5, 7009.5, N)]),
        ("i16", i16s(small(10.0, -100.0)), [(-101.5, -96.5), (128.5, 133.5)], i16s(pattern(1.0, -3.0)),
         vec![(-3.5, -2.5, N), (-2.5, -1.5, N), (-0.5, 0.5, N), (0.5, 1.5, N), (2.5, 3.5, N), (3.5, 4.5, N)]),
        ("f32", f32s(small(1.5, 0.25)), [(-0.20999999999999996, 0.4225000000000001), (34.5775, 35.21)], f32s(pattern(0.5, 0.25)),
         vec![(0.2033333333333333, 0.26749999999999996, N), (0.7166666666666666, 0.7808333333333333, N), (1.7433333333333332, 1.8074999999999999, N),
              (2.1925, 2.256666666666667, N), (3.2191666666666667, 3.283333333333333, N), (3.7325, 3.796666666666667, N)]),
    ];
    let plan: ProjectionPlan = serde_json::from_value(json!({"spec": {"kind": "histogram", "group": null, "reduced": [], "reduction": "mean"},
        "display": {"window": 1, "bin_count": 60}})).unwrap();
    for (name, small_frame, ends, frame, expected) in cases {
        for (shape, values) in [([6, 4], small_frame), (SHAPE, frame)] {
            let runtime = Runtime::new();
            let generation = begin(&runtime, camera_schema(shape, values.dtype()));
            let read = publish(&runtime, &generation, shape, values);
            let hosted = bars(&Projection::new().project(&Source::from_read(&read).unwrap(), &plan).unwrap().payload);
            let carried = bars(&Projection::new().project_read(&read, &plan).unwrap().payload);
            assert_eq!(hosted, carried, "{name} {shape:?}: the two runtime read paths disagree");
            assert_eq!(hosted.iter().map(|bar| bar.2).sum::<i64>(), (shape[0] * shape[1]) as i64, "{name} {shape:?}: pixels counted");
            if shape == SHAPE {
                assert_eq!(hosted, expected, "{name}: camera frame bins");
            } else {
                assert_eq!((hosted.len(), [(hosted[0].0, hosted[0].1), (hosted[23].0, hosted[23].1)]), (24, ends), "{name}: 6x4 bins");
            }
        }
    }
}

#[derive(Clone)]
struct HostRead { source: Arc<Source> }

impl DocumentRead for HostRead {
    fn source(&self) -> &Arc<Source> { &self.source }
    fn provenance_json(&self) -> String { "null".into() }
}

/// A Histogram panel of the native document keeps its bins across a camera
/// restart, as Python's does: the restarted frame's 255 pixel widens the
/// domain without re-anchoring the integer bins. (A domain anchored afresh on
/// that frame splits the planes at 1.5 / 6.5 into two bins of half the
/// pixels each: what a panel shows when its first front was never accepted.)
#[test]
fn panel_bins_carry_across_a_camera_restart() {
    use zlc_plot::configuration::resolve::Defaults;
    use zlc_plot::fit::FitEngine;
    use zlc_plot::interaction::gesture::InteractionDefaults;
    use zlc_plot::render::panel::NativeRendererFactory;
    use zlc_plot::session::{EngineFitLayer, SessionServices};
    let services = SessionServices { renderers: Arc::new(NativeRendererFactory), fit: Arc::new(EngineFitLayer::new(Arc::new(FitEngine::default()))),
        defaults: Arc::new(Defaults::default()), registry: None, interaction: InteractionDefaults::default() };
    let mut document = FigureDocument::<HostRead>::new(services, None);
    document.execute(&json!({"action": "ready", "dpr": 1.0, "id": 1, "target_requests": {}, "targets": {}}).to_string(), BTreeMap::new(), None, None).unwrap();
    document.finish_presentation(&BTreeSet::new(), &BTreeSet::new());
    let panels = BTreeSet::from(["panel-1".to_owned()]);
    let runtime = Runtime::new();
    let mut target = json!(null);
    // (binding epoch, the frame's peak pixel, Python's bars).
    let shown = [(1, 122, vec![(-0.5, 2.5, 2 * N - 1), (2.5, 5.5, 2 * N), (5.5, 8.5, 2 * N), (119.5, 122.5, 1)]),
                 (2, 255, vec![(-1.5, 3.5, 3 * N - 1), (3.5, 8.5, 3 * N), (253.5, 258.5, 1)])];
    for (epoch, peak, expected) in shown {
        let generation = begin(&runtime, camera_schema(SHAPE, zlc_core::DType::U8));
        let mut values: Vec<u8> = pattern(1.0, 0.0).into_iter().map(|x| x as u8).collect();
        values[0] = peak;
        let read = HostRead { source: Arc::new(Source::from_read(&publish(&runtime, &generation, SHAPE, OwnedBuffer::from_vec(values).into())).unwrap()) };
        let inputs = json!([{"epoch": epoch, "kind": "histogram", "panel": "panel-1", "reason": null, "signal": "frames", "waiting": false, "window": 1}]);
        let command = if target.is_null() {
            json!({"action": "sync_panels", "id": 2, "kind": "histogram", "panel": "panel-1", "signal": "frames", "reset_document": true,
                   "inputs": inputs, "target_requests": {"panel-1": 2}, "targets": {}})
        } else {
            json!({"action": "sync_panels", "id": 0, "reset_document": false, "inputs": inputs, "target_requests": {"panel-1": 2}, "targets": {"panel-1": target}})
        };
        let execution = document.execute(&command.to_string(), BTreeMap::from([("panel-1".to_owned(), read)]), None, None).unwrap();
        document.finish_presentation(&panels, &panels);
        target = execution.state["panels"][0]["target"].clone();
        let session = document.session("panel-1").unwrap();
        assert_eq!(bars(session.lock().unwrap().projection().payload()), expected, "epoch {epoch}");
        runtime.seal(&generation, json!({})).unwrap();
    }
}


#[test]
fn panel_header_tracks_the_accepted_finite_repeat_counts() {
    use zlc_core::schema::DomainRole;
    use zlc_plot::{configuration::resolve::Defaults, fit::FitEngine, interaction::gesture::InteractionDefaults,
        render::panel::NativeRendererFactory, session::{EngineFitLayer, SessionServices}};
    let services = SessionServices { renderers: Arc::new(NativeRendererFactory), fit: Arc::new(EngineFitLayer::new(Arc::new(FitEngine::default()))),
        defaults: Arc::new(Defaults::default()), registry: None, interaction: InteractionDefaults::default() };
    let mut document = FigureDocument::<HostRead>::new(services, None);
    document.execute(r#"{"action":"ready","dpr":1.0,"id":1,"targets":{},"target_requests":{}}"#, BTreeMap::new(), None, None).unwrap();
    document.finish_presentation(&BTreeSet::new(), &BTreeSet::new());
    let chunk = camera_schema([2,3], zlc_core::DType::U8);
    let repeat = Domain::dense(vec![3],vec![Axis::new("camera.repeat".into(),"repeat".into(),"repeat".into(),Coordinates::Index{origin:0,len:3},None,None,None,None).unwrap()]).unwrap();
    let full = DatasetSchema::new(repeat,chunk.domain(DomainRole::Point).clone(),chunk.domain(DomainRole::Cell).clone(),chunk.value().clone()).unwrap();
    let runtime=Runtime::new();
    let generation=runtime.begin("camera",vec![Declaration{signal:"frames".into(),schema:full,storage:Storage::Finite}],json!({})).unwrap();
    let mut reads=vec![];
    let mut target=json!(null);
    for index in 0..3 {
        let block=DataBlock::new(chunk.clone(),OwnedBuffer::from_vec(vec![index as u8;6]).into(),Validity::Uniform(false),None).unwrap();
        let delivery=runtime.commit(&generation,&Bundle{inputs:vec![Input::Event{signal:"frames".into(),ordinal:index as u64,data:block,origin:Some([index,0])}],metadata:json!({}),parents:vec![]}).unwrap();
        let read=HostRead{source:Arc::new(Source::from_read(&Read::from_canonical(delivery.publications[0].clone())).unwrap())};
        reads.push(read);
    }
    runtime.seal(&generation,json!({})).unwrap();
    // Present old snapshots after the producer ends. Their own counts must
    // survive newer publications; no independent latest lookup is allowed.
    for (step,index) in [0usize,1,2].into_iter().enumerate() {
        assert_eq!(reads[index].source.repeat_counts,Some(vec![index+1]));
        let command=json!({"action":"sync_panels","id":2,"panel":"p","kind":"facet_grid","signal":"frames","reset_document":step==0,
            "inputs":[{"panel":"p","signal":"frames","kind":"facet_grid","window":1,"epoch":1,"waiting":false,"reason":null}],
            "target_requests":{"p":2},"targets":if target.is_null(){json!({})}else{json!({"p":target})}});
        let result=document.execute(&command.to_string(),[("p".to_owned(),reads[index].clone())].into(),None,None).unwrap();
        document.finish_presentation(&BTreeSet::from(["p".to_owned()]),&BTreeSet::from(["p".to_owned()]));
        let panel=&result.state["panels"][0];
        assert_eq!(panel["structure"][0],json!([["repeat",index+1]]), "step {step}: {panel}");
        assert_eq!(panel["structure"][1],json!([["readout-event",1]]));
        assert_eq!(panel["structure"][2],json!([["spatial-y",2],["spatial-x",3]]));
        target=panel["target"].clone();
    }
}


#[test]
fn a_loaded_empty_panel_reserves_its_authored_size_before_start() {
    use zlc_plot::{configuration::resolve::Defaults, fit::FitEngine, interaction::gesture::InteractionDefaults,
        render::panel::NativeRendererFactory, session::{EngineFitLayer, SessionServices}};
    let services = SessionServices { renderers: Arc::new(NativeRendererFactory), fit: Arc::new(EngineFitLayer::new(Arc::new(FitEngine::default()))),
        defaults: Arc::new(Defaults::default()), registry: None, interaction: InteractionDefaults::default() };
    let mut document = FigureDocument::<HostRead>::new(services, None);
    document.execute(r#"{"action":"ready","dpr":1.0,"id":1,"targets":{},"target_requests":{}}"#, BTreeMap::new(), None, None).unwrap();
    document.finish_presentation(&BTreeSet::new(), &BTreeSet::new());
    let panels=BTreeSet::from(["p".to_owned()]);
    let binding=json!([{"panel":"p","signal":"frames","kind":"facet_grid","window":1,"epoch":1,"waiting":true,"reason":"No data yet"}]);
    for (index,size) in ["1x2","4x8"].into_iter().enumerate() {
        let command=if index==0 {json!({"action":"sync_panels","id":2,"reset_document":true,"inputs":binding,"targets":{"p":{"size":size}},"target_requests":{"p":2}})}
            else {json!({"action":"configure","id":3,"panel":"p","size":size,"inputs":binding,"targets":{"p":{"size":"1x2"}},"target_requests":{"p":3}})};
        let result=document.execute(&command.to_string(),BTreeMap::new(),None,None).unwrap();
        let row=&result.state["panels"][0];
        let (width,height)=zlc_plot::layout::LayoutConfig::default().logical_size(size,zlc_plot::layout::Kind::FacetGrid).unwrap();
        assert_eq!(row["size"],size);
        assert_eq!(row["target"]["size"],size, "saving before Start must keep the authored size");
        assert_eq!(row["display_size"],json!({"width":width,"height":height}));
        assert_eq!(row["size_choices"].as_array().unwrap().len(),9);
        assert!(result.fronts.is_empty());
        document.finish_presentation(&panels,&BTreeSet::new());
    }
    let runtime=Runtime::new();let generation=begin(&runtime,camera_schema([2,3],zlc_core::DType::U8));
    let read=HostRead{source:Arc::new(Source::from_read(&publish(&runtime,&generation,[2,3],OwnedBuffer::from_vec(vec![1u8;6]).into())).unwrap())};
    let mut binding=binding;binding[0]["waiting"]=json!(false);binding[0]["reason"]=json!(null);
    let result=document.execute(&json!({"action":"sync_panels","id":4,"inputs":binding,"targets":{"p":{"size":"1x2"}},"target_requests":{"p":3}}).to_string(),
        [("p".to_owned(),read)].into(),None,None).unwrap();
    assert_eq!(result.state["panels"][0]["size"],"4x8");
    let front=&result.fronts["p"];
    let expected=zlc_plot::layout::LayoutConfig::default().logical_size("4x8",zlc_plot::layout::Kind::FacetGrid).unwrap();
    assert_eq!((front.width,front.height),expected);
    document.finish_presentation(&panels,&panels);
}
