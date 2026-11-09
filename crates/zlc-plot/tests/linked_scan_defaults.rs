//! A scan with a linked parameter (axis 2 = x + 6.37, an alternative coordinate of
//! axis 1) opens every default panel: the two coordinates are one axis, never given
//! two roles ("alternative coordinates of one axis cannot have different fates").
use std::collections::BTreeSet;
use std::sync::Arc;

use serde_json::{json, Value as Json};
use zlc_core::dataset::{DataBlock, Validity};
use zlc_core::runtime::{Bundle, Declaration, Input, Read as RuntimeRead, Runtime, Storage};
use zlc_core::scan::{AxisSpec, Plan, PlanSpec};
use zlc_core::schema::{Axis, Coordinates, DatasetSchema, Domain, DomainRole, ValidityContract, ValueSchema};
use zlc_core::{DType, OwnedBuffer};
use zlc_plot::configuration::resolve::Defaults;
use zlc_plot::document::{DocumentRead, FigureDocument};
use zlc_plot::fit::FitEngine;
use zlc_plot::interaction::gesture::InteractionDefaults;
use zlc_plot::projection::Source;
use zlc_plot::render::panel::NativeRendererFactory;
use zlc_plot::session::{EngineFitLayer, SessionServices};

#[derive(Clone)]
struct Read(Arc<Source>);
impl DocumentRead for Read {
    fn source(&self) -> &Arc<Source> { &self.0 }
    fn provenance_json(&self) -> String { "{}".into() }
}

fn index(id: &str, len: usize) -> Arc<Axis> {
    Axis::new(id.into(), id.into(), "generic".into(), Coordinates::Index { origin: 0, len }, None, None, None, None).unwrap()
}

/// One shot of the source: a value, or a small image.
fn source(image: bool) -> Arc<DatasetSchema> {
    let cell = if image { Domain::dense(vec![2, 3], vec![index("y", 2), index("x", 3)]) } else { Domain::dense(vec![1], vec![index("channel", 1)]) }.unwrap();
    DatasetSchema::new(Domain::dense(vec![1], vec![index("source.repeat", 1)]).unwrap(), Domain::new(vec![1], vec![], vec![]).unwrap(), cell,
        ValueSchema::new(DType::F64, Some("V".into()), None, ValidityContract::Value).unwrap()).unwrap()
}

/// Axis 1 from 100 to 115 in 40 points, axis 2 following it as x + 6.37, two shots a point.
fn scan(image: bool) -> RuntimeRead {
    let values: Vec<f64> = (0..40).map(|i| 100.0 + 15.0 * i as f64 / 39.0).collect();
    let axis = |port: &str, values: Vec<f64>, of: Option<&str>| AxisSpec { port: port.into(), identity: port.into(), label: port.into(), unit: String::new(),
        values, coordinate_of: of.map(str::to_owned), host_advanced: false };
    let plan = Plan::new(PlanSpec { axes: vec![axis("pulse:param:a", values.clone(), None),
        axis("pulse:param:b", values.iter().map(|v| v + 6.37).collect(), Some("pulse:param:a"))], scan_repeats: 1, run_repeats: 2 }).unwrap();
    let one = source(image);
    let schema = plan.dataset_schema(&one).unwrap();
    assert!(schema.domain(DomainRole::Point).axes().iter().any(|axis| axis.coordinate_of().is_some()), "a linked coordinate");
    let runtime = Runtime::new();
    let generation = runtime.begin("scan", vec![Declaration { signal: "scan".into(), schema, storage: Storage::Finite }], json!({})).unwrap();
    let size = one.domain(DomainRole::Cell).size();
    for point in 0..40usize {
        for shot in 0..2usize {
            let data = DataBlock::new(one.clone(), OwnedBuffer::from_vec(vec![point as f64; size]).into(), Validity::Uniform(true), None).unwrap();
            runtime.commit(&generation, &Bundle { inputs: vec![Input::Event { signal: "scan".into(), ordinal: (2 * point + shot) as u64, data,
                origin: Some(plan.placement(point, 0, shot, 1).unwrap()), shot_time: None }], metadata: json!({}), parents: vec![] }).unwrap();
        }
    }
    runtime.seal(&generation, json!({})).unwrap();
    runtime.read_views(&[(generation, "scan".into(), None)]).pop().unwrap().unwrap()
}

/// A fresh panel of this kind with no settings of its own (an auto-preview), its error.
fn opened(read: &RuntimeRead, kind: &str) -> Json {
    let services = SessionServices { renderers: Arc::new(NativeRendererFactory), fit: Arc::new(EngineFitLayer::new(Arc::new(FitEngine::default()))),
        defaults: Arc::new(Defaults::default()), registry: None, interaction: InteractionDefaults::default() };
    let mut document = FigureDocument::<Read>::new(services, None);
    let mut state = Json::Null;
    for (id, command) in [json!({"action": "ready", "dpr": 1.0}),
        json!({"action": "sync_panels", "inputs": [{"panel": "p", "kind": kind, "window": 1, "signal": "scan", "epoch": 1}], "targets": {"p": {}}})].into_iter().enumerate() {
        let mut command = command;
        command["id"] = json!(id + 1);
        let reads = [("p".to_owned(), Read(Arc::new(Source::from_read(read).unwrap())))].into_iter().collect();
        state = document.execute(&command.to_string(), reads, None, None).unwrap().state;
        let all: BTreeSet<String> = ["p".to_owned()].into();
        document.finish_presentation(&all, &all);
    }
    assert!(state["error"].is_null(), "{kind}: {}", state["error"]);
    state["panels"][0]["error"].clone()
}

#[test]
fn a_linked_scan_opens_every_default_panel() {
    for image in [false, true] {
        let read = scan(image);
        for kind in ["facet_grid", "curve", "image"] {
            let error = opened(&read, kind);
            assert!(error.is_null() || !error.to_string().contains("alternative coordinates"), "image={image} {kind}: {error}");
        }
    }
}
