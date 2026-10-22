//! Time recorded golden cases one at a time (min over repetitions):
//! `ZLC_CASE_SUITE=SUITE.json ZLC_CASE_NAME=FRAGMENT projection_case`.
use std::time::Instant;

use zlc_plot::projection::golden::{Suite, source};
use zlc_plot::projection::plan::{Projection, ProjectionPlan};
use zlc_plot::projection::{DataView, SchemaCache, Snapshot};

fn main() {
    let suite = Suite::read(std::path::Path::new(&std::env::var("ZLC_CASE_SUITE").expect("ZLC_CASE_SUITE")));
    let fragment = &std::env::var("ZLC_CASE_NAME").expect("ZLC_CASE_NAME");
    let mut cache = SchemaCache::default();
    let snapshots = suite.document["snapshots"].as_object().unwrap();
    for case in suite.document["cases"].as_array().unwrap().iter().filter(|c| c["name"].as_str().unwrap().contains(fragment.as_str()) && c.get("error").is_none()
        && std::env::var("ZLC_CASE_KIND").ok().is_none_or(|k| c["plan"]["spec"]["kind"] == k.as_str())).take(40) {
        let plan: ProjectionPlan = serde_json::from_value(case["plan"].clone()).unwrap();
        let source = source(&suite, &snapshots[case["snapshot"].as_str().unwrap()]);
        let snapshot = Snapshot::from_source(&source, &mut cache).unwrap();
        let time = |label: &str, f: &mut dyn FnMut()| {
            let mut best = f64::INFINITY;
            for _ in 0..40 { let t = Instant::now(); f(); best = best.min(t.elapsed().as_secs_f64() * 1e3); }
            println!("  {label:24} {best:8.4} ms");
        };
        println!("{} {}", case["name"], serde_json::to_string(&plan.spec).unwrap());
        time("convert", &mut || { Snapshot::from_source(&source, &mut cache).unwrap(); });
        time("view + packed", &mut || { DataView::new(snapshot.clone(), vec![], None, None).unwrap().packed(true).unwrap(); });
        if let zlc_plot::projection::plan::SpecFacts::FacetGrid { facet: Some(facet), cell } = &plan.spec {
            if let zlc_plot::projection::plan::SpecFacts::Curve { x, .. } = cell.as_ref() {
                let view = DataView::new(snapshot.clone(), vec![], None, None).unwrap();
                let refs = [facet.clone(), x.clone()];
                let (domains, codes, dims) = view.axis_projection(&refs).unwrap();
                let sizes: Vec<usize> = domains.iter().map(|d| d.size()).collect();
                time("segmented mean", &mut || { view.segmented_axes(&codes, &dims, &sizes, zlc_plot::kinds::Reduction::Mean, false).unwrap(); });
                time("segmented mean+sem", &mut || { view.segmented_axes(&codes, &dims, &sizes, zlc_plot::kinds::Reduction::Mean, true).unwrap(); });
            }
        }
        if let zlc_plot::projection::plan::SpecFacts::Rolling { x, group, reduction } = &plan.spec {
            let view = DataView::new(snapshot.clone(), vec![], None, None).unwrap();
            time("rolling history", &mut || { view.rolling_history(x.as_ref(), group.as_ref(), *reduction, true).unwrap(); });
            let history = view.rolling_history(x.as_ref(), group.as_ref(), *reduction, true).unwrap();
            let trailing = plan.display.trailing.unwrap_or(1);
            time("rolling payload", &mut || { zlc_plot::projection::rolling::rolling_payload(&view, &history, x.as_ref(), group.as_ref(), trailing, true).unwrap(); });
        }
        if let zlc_plot::projection::plan::SpecFacts::Image { x, y, reduction } = &plan.spec {
            time("DataView::new", &mut || { DataView::new(snapshot.clone(), vec![], None, None).unwrap(); });
            time("view + image", &mut || { DataView::new(snapshot.clone(), vec![], None, None).unwrap().image(x, y, *reduction).unwrap(); });
            time("view + resolve x", &mut || { let v = DataView::new(snapshot.clone(), vec![], None, None).unwrap(); v.resolve(x).unwrap(); });
            time("view + resolve x,y", &mut || { let v = DataView::new(snapshot.clone(), vec![], None, None).unwrap(); v.domain(x).unwrap(); v.domain(y).unwrap(); });
            time("view + dense", &mut || { let v = DataView::new(snapshot.clone(), vec![], None, None).unwrap(); v.dense_tensor_projection(&[y.clone(), x.clone()], *reduction, false).unwrap(); });
        }
        time("project (fresh owner)", &mut || { Projection::new().project_snapshot(snapshot.clone(), &plan).unwrap(); });
        let mut live = Projection::new();
        let mut held = None;
        time("project (live owner)", &mut || { drop(held.replace(live.project_snapshot(snapshot.clone(), &plan).unwrap())); });
    }
}
