//! Four live panels per shot, end to end from the runtime's own reads: a
//! 147x201 u8 ROI curve (mean + SEM), a 1200x1920 u16 image mean, a facet of
//! 40 ROI frames and a 3000-shot rolling history (mean + SEM), each shot
//! published to a `zlc_core::runtime::Runtime` and projected from
//! `read_views` with one live `Projection` per panel.
//!
//! In the warm-up blocks every shot is also projected from a freshly
//! converted `Source` by a second owner per panel, and the payloads must agree
//! bit for bit: the carried segments may only save work. Prints the median and
//! minimum milliseconds per panel and per shot: `ZLC_LIVE4_SHOTS` (default 20),
//! `ZLC_LIVE4_REPS` (5), `ZLC_LIVE4_WARM` (2), `ZLC_LIVE4_OUT` (every timed shot as JSON).
use std::sync::Arc;
use std::time::Instant;

use serde_json::{Value, json};
use zlc_core::dataset::{DataBlock, Validity};
use zlc_core::runtime::{Bundle, Declaration, Generation, Input, Runtime, Storage};
use zlc_core::schema::{Axis, Coordinates, DatasetSchema, Domain, ValidityContract, ValueSchema};
use zlc_core::{AnyBuffer, DType, OwnedBuffer};
use zlc_plot::projection::output::GoldenArrays;
use zlc_plot::projection::plan::{Projection, ProjectionPlan};
use zlc_plot::projection::Source;

const WINDOW: usize = 3000;

fn axis(id: &str, name: &str, role: &str, len: usize) -> Arc<Axis> {
    Axis::new(id.into(), name.into(), role.into(), Coordinates::Index { origin: 0, len }, None, None, None, None).unwrap()
}

fn camera(frames: usize, ny: usize, nx: usize, dtype: DType) -> Arc<DatasetSchema> {
    DatasetSchema::new(
        Domain::dense(vec![1], vec![axis("camera.repeat", "repeat", "repeat", 1)]).unwrap(),
        Domain::dense(vec![frames], vec![axis("camera.frame", "frame", "readout-event", frames)]).unwrap(),
        Domain::dense(vec![ny, nx], vec![axis("camera.spatial-y", "y", "spatial-y", ny), axis("camera.spatial-x", "x", "spatial-x", nx)]).unwrap(),
        ValueSchema::new(dtype, Some("count".into()), None, ValidityContract::Value).unwrap()).unwrap()
}

fn stream() -> Arc<DatasetSchema> {
    DatasetSchema::new(
        Domain::dense(vec![1], vec![axis("hist.repeat", "repeat", "repeat", 1)]).unwrap(),
        Domain::new(vec![1], vec![], vec![]).unwrap(),
        Domain::dense(vec![1], vec![axis("zlc_data.scalar", "value", "scalar", 1)]).unwrap(),
        ValueSchema::new(DType::F64, Some("count".into()), Some("signal".into()), ValidityContract::Value).unwrap()).unwrap()
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn normal(&mut self) -> f64 { (0..12).map(|_| (self.next() >> 11) as f64 / (1u64 << 53) as f64).sum::<f64>() - 6.0 }
}

fn block(schema: &Arc<DatasetSchema>, values: AnyBuffer) -> Arc<DataBlock> {
    DataBlock::new(schema.clone(), values, Validity::Uniform(true), None).unwrap()
}

fn plan(text: Value) -> ProjectionPlan { serde_json::from_value(text).unwrap() }

fn env(name: &str, default: usize) -> usize { std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default) }

fn main() {
    let (shots, reps, warm) = (env("ZLC_LIVE4_SHOTS", 20), env("ZLC_LIVE4_REPS", 5), env("ZLC_LIVE4_WARM", 2));
    let x = json!({"domain": "cell_data", "axis_id": "camera.spatial-x"});
    let y = json!({"domain": "cell_data", "axis_id": "camera.spatial-y"});
    let defaults = json!({"histogram_domain_padding_fraction": {"f": "9a9999999999a93f"}});
    let panels: [(&str, ProjectionPlan, Option<usize>); 4] = [
        ("roi", plan(json!({"spec": {"kind": "curve", "x": x, "group": null, "reduction": "mean"}, "display": {"window": 1, "uncertainty": true}, "defaults": defaults})), None),
        ("cam", plan(json!({"spec": {"kind": "image", "x": x, "y": y, "reduction": "mean"}, "defaults": defaults})), None),
        ("facet", plan(json!({"spec": {"kind": "facet_grid", "facet": {"domain": "point", "axis_id": "camera.frame"},
            "cell": {"kind": "image", "x": x, "y": y, "reduction": "mean"}}, "defaults": defaults})), None),
        ("stream", plan(json!({"spec": {"kind": "rolling", "x": null, "group": null, "reduction": "mean"},
            "display": {"window": WINDOW, "uncertainty": true, "trailing": 1, "bin_count": 60}, "defaults": defaults})), Some(WINDOW)),
    ];
    let schemas = [camera(1, 147, 201, DType::U8), camera(1, 1200, 1920, DType::U16), camera(40, 147, 201, DType::U8), stream()];
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut frames = |schema: &Arc<DatasetSchema>| -> AnyBuffer {
        let n = schema.value_len();
        match schema.dtype() {
            DType::U8 => OwnedBuffer::from_vec((0..n).map(|_| rng.next() as u8).collect()).into(),
            DType::U16 => OwnedBuffer::from_vec((0..n).map(|_| rng.next() as u16).collect()).into(),
            _ => OwnedBuffer::from_vec((0..n).map(|_| rng.normal() * 3.0 + 100.0).collect::<Vec<f64>>()).into(),
        }
    };
    let history: Vec<Arc<DataBlock>> = (0..WINDOW).map(|_| block(&schemas[3], frames(&schemas[3]))).collect();

    let mut timings: Vec<Vec<[f64; 6]>> = Vec::new();
    for rep in 0..warm + reps {
        let runtime = Runtime::new();
        let declarations = panels.iter().zip(&schemas).map(|((name, ..), schema)|
            Declaration { signal: (*name).into(), schema: schema.clone(), storage: Storage::Latest }).collect();
        let generation: Generation = runtime.begin("camera", declarations, json!({})).unwrap();
        let _lease = runtime.retain_history(&generation, "stream", WINDOW).unwrap();
        for (ordinal, data) in history.iter().enumerate() {
            runtime.commit(&generation, &Bundle { inputs: vec![Input::Event { signal: "stream".into(), ordinal: ordinal as u64, data: data.clone(), origin: None }],
                metadata: json!({}), parents: vec![] }).unwrap();
        }
        let mut live: Vec<Projection> = (0..4).map(|_| Projection::new()).collect();
        let mut fresh: Vec<Projection> = (0..4).map(|_| Projection::new()).collect();
        let mut held: Vec<Option<zlc_plot::projection::plan::Projected>> = (0..4).map(|_| None).collect();
        let requests: Vec<_> = panels.iter().map(|(name, _, width)| (generation.clone(), Arc::<str>::from(*name), *width)).collect();
        let mut block_times = Vec::new();
        for shot in 0..shots {
            // Acquisition and publishing are not timed.
            let inputs = panels.iter().enumerate().map(|(k, (name, ..))| Input::Event { signal: (*name).into(),
                ordinal: if k == 3 { (WINDOW + shot) as u64 } else { shot as u64 }, data: block(&schemas[k], frames(&schemas[k])), origin: None }).collect();
            runtime.commit(&generation, &Bundle { inputs, metadata: json!({}), parents: vec![] }).unwrap();
            let started = Instant::now();
            let reads: Vec<_> = runtime.read_views(&requests).into_iter().map(Result::unwrap).collect();
            let mut times = [started.elapsed().as_secs_f64() * 1e3, 0.0, 0.0, 0.0, 0.0, 0.0];
            for (k, read) in reads.iter().enumerate() {
                let started = Instant::now();
                let projected = live[k].project_read(read, &panels[k].1).unwrap();
                times[k + 1] = started.elapsed().as_secs_f64() * 1e3;
                // The carry may only save work: in the warm blocks (untimed in
                // effect, they pollute the caches) the fresh conversion agrees bit for bit.
                if rep < warm {
                    let reference = fresh[k].project(&Source::from_read(read).unwrap(), &panels[k].1).unwrap();
                    let (mut a, mut b) = (GoldenArrays::default(), GoldenArrays::default());
                    let (ja, jb) = (projected.payload.golden(&mut a), reference.payload.golden(&mut b));
                    assert!(ja == jb && a.0.len() == b.0.len() && a.0.iter().zip(&b.0).all(|(p, q)| p.dtype() == q.dtype() && p.shape() == q.shape() && p.bytes() == q.bytes()),
                        "{}: shot {shot} disagrees with a fresh conversion", panels[k].0);
                }
                drop(held[k].replace(projected));
            }
            // Releasing the reads is the runtime's half: a window's entry list
            // drops one reference per shot it holds.
            let started = Instant::now();
            drop(reads);
            times[5] = started.elapsed().as_secs_f64() * 1e3;
            block_times.push(times);
        }
        if rep >= warm { timings.push(block_times); }
    }
    let all: Vec<[f64; 6]> = timings.iter().flatten().copied().collect();
    let stats = |k: usize| { let mut v: Vec<f64> = all.iter().map(|t| match k { 6 => t[1] + t[2] + t[3] + t[4], 7 => t.iter().sum(), _ => t[k] }).collect();
        v.sort_by(f64::total_cmp); (v[v.len() / 2], v[0]) };
    for (k, label) in [(0, "read_views (runtime)"), (1, "roi curve 147x201"), (2, "image 1200x1920"), (3, "facet 40x147x201"), (4, "rolling 3000"),
                       (5, "read release (runtime)"), (6, "4 projections per shot"), (7, "end to end per shot")] {
        let (median, min) = stats(k);
        println!("{label:24} median {median:7.3} ms  min {min:7.3} ms  ({} shots)", all.len());
    }
    if let Ok(path) = std::env::var("ZLC_LIVE4_OUT") { std::fs::write(path, serde_json::to_string(&timings).unwrap()).unwrap(); }
}
