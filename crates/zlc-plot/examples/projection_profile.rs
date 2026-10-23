//! Stage timings of a 40-shot ROI facet (min over repetitions).
use std::sync::Arc;
use std::time::Instant;

use zlc_core::dataset::{DataBlock, Validity};
use zlc_core::schema::{Axis, CodeBase, CodeMapping, Coordinates, DatasetSchema, Domain, ValidityContract, ValueSchema};
use zlc_core::{DType, OwnedBuffer};
use zlc_plot::kinds::{AxisRef, Reduction};
use zlc_plot::projection::curve::CellSpec;
use zlc_plot::projection::plan::{Defaults, DisplayFacts, ProjectionPlan, SpecFacts};
use zlc_plot::projection::{DataView, Projection, SchemaCache, Snapshot, Source, SourceData};

fn axis(id: &str, role: &str, coordinates: Coordinates) -> Arc<Axis> {
    Axis::new(id.into(), id.into(), role.into(), coordinates, None, None, None, None).unwrap()
}

fn time(label: &str, mut f: impl FnMut()) {
    let mut best = f64::INFINITY;
    for _ in 0..30 { let t = Instant::now(); f(); best = best.min(t.elapsed().as_secs_f64() * 1e3); }
    println!("{label:28} {best:8.3} ms");
}

fn main() {
    let (shots, height, width) = (40usize, 147usize, 201usize);
    let cell = Domain::dense(vec![height, width], vec![axis("y", "spatial-y", Coordinates::Index { origin: 0, len: height }),
        axis("x", "spatial-x", Coordinates::Index { origin: 0, len: width })]).unwrap();
    let history = axis("h", "primary-index", Coordinates::Numeric(OwnedBuffer::from_vec((0..shots as i64).map(|i| i - shots as i64 + 1).collect::<Vec<i64>>()).into()));
    let frame = axis("frame", "readout-event", Coordinates::Index { origin: 0, len: 1 });
    let point = Domain::new(vec![shots], vec![history, frame], vec![
        CodeMapping::new(CodeBase::Range { start: 0, step: 1, len: shots }, 1, 1).unwrap(),
        CodeMapping::new(CodeBase::Range { start: 0, step: 1, len: 1 }, 1, shots).unwrap()]).unwrap();
    let repeat = Domain::dense(vec![1], vec![axis("r", "repeat", Coordinates::Index { origin: 0, len: 1 })]).unwrap();
    let value = ValueSchema::new(DType::U8, Some("count".into()), None, ValidityContract::Value).unwrap();
    let schema = DatasetSchema::new(repeat, point, cell.clone(), value.clone()).unwrap();
    let segment_schema = DatasetSchema::new(
        Domain::dense(vec![1], vec![axis("sr", "g", Coordinates::Index { origin: 0, len: 1 })]).unwrap(),
        Domain::dense(vec![1], vec![axis("sp", "g", Coordinates::Index { origin: 0, len: 1 })]).unwrap(), cell, value).unwrap();
    let segments = (0..shots).map(|s| ([0usize, s], DataBlock::new(segment_schema.clone(),
        OwnedBuffer::from_vec((0..height * width).map(|i| ((i * 7 + s * 13) % 251) as u8).collect::<Vec<u8>>()).into(), Validity::Uniform(true), None).unwrap())).collect();
    let source = Source { repeat_counts: None, schema, data: SourceData::Segments(segments), block_id: "w".into(), generation: "g".into(), revision: 1, window: Some((0, 39)) };
    let x = AxisRef::cell_data("x").unwrap();
    let facet = AxisRef::point("h").unwrap();
    let plan = ProjectionPlan {
        spec: SpecFacts::FacetGrid { facet: Some(facet.clone()), cell: Box::new(SpecFacts::Curve { x: x.clone(), group: None, reduction: Reduction::Mean }) },
        scope: vec![], display: DisplayFacts { window: Some(40), uncertainty: Some(true), ..Default::default() }, view_display: None, defaults: Defaults::default(),
    };
    let mut cache = SchemaCache::default();
    let snapshot = Snapshot::from_source(&source, &mut cache).unwrap();
    time("convert source", || { Snapshot::from_source(&source, &mut cache).unwrap(); });
    let parts: Vec<zlc_plot::projection::nd::Array> = (0..40).map(|s| zlc_plot::projection::nd::Array::from_vec((0..29547).map(|i| ((i + s) % 200) as u8).collect::<Vec<u8>>(), vec![29547])).collect();
    let refs: Vec<&zlc_plot::projection::nd::Array> = parts.iter().collect();
    time("concat 40 segments", || { zlc_plot::projection::nd::Array::concat(&refs, vec![40 * 29547]); });
    let mut reused: Vec<u8> = Vec::with_capacity(40 * 29547);
    time("concat into reused", || { reused.clear(); for p in &parts { reused.extend_from_slice(p.slice::<u8>()); } });
    time("DataView::new", || { DataView::new(snapshot.clone(), vec![], None, None).unwrap(); });
    time("view + pack", || { let v = DataView::new(snapshot.clone(), vec![], None, None).unwrap(); v.packed(true).unwrap(); });
    time("dense projection (unc)", || { let v = DataView::new(snapshot.clone(), vec![], None, None).unwrap(); v.dense_tensor_projection(&[facet.clone(), x.clone()], Reduction::Mean, true).unwrap(); });
    time("dense projection", || { let v = DataView::new(snapshot.clone(), vec![], None, None).unwrap(); v.dense_tensor_projection(&[facet.clone(), x.clone()], Reduction::Mean, false).unwrap(); });
    let cellspec = CellSpec::Curve { x: x.clone(), group: None, reduction: Reduction::Mean };
    time("facet", || { let v = DataView::new(snapshot.clone(), vec![], None, None).unwrap(); v.facet(Some(&facet), &cellspec, true).unwrap(); });
    time("project_snapshot (fresh)", || { Projection::new().project_snapshot(snapshot.clone(), &plan).unwrap(); });
    let mut live = Projection::new();
    time("project_snapshot (live owner)", || { live.project_snapshot(snapshot.clone(), &plan).unwrap(); });
}
