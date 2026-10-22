//! Golden tests of the native projection against the Python science package.
//! Suites are written by `tests/gen/*.py` (see `goldens.py` for the format);
//! the agreement rules are `src/projection/AGREEMENT.md`.
//!
//! Committed suites live in `tests/data/projection`; the large recorded
//! suites run when `ZLC_PROJECTION_GOLDEN_DIR` names their directory.

use std::path::{Path, PathBuf};


use serde_json::Value;
use zlc_plot::projection::golden::{Suite, compare, histogram_json, histogram_state, source};
use zlc_plot::projection::output::GoldenArrays;
use zlc_plot::projection::fit_selection::{FitContext, FitRequestFacts, SelectorFacts, fit_selection};
use zlc_plot::projection::samples::selector_mask;
use zlc_plot::projection::plan::{Projection, ProjectionPlan};
use zlc_plot::projection::{Snapshot, SchemaCache, Source};

fn load(path: &Path) -> Suite {
    let read = |path: PathBuf| -> Vec<u8> {
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        if path.extension().is_some_and(|x| x == "gz") {
            let mut out = Vec::new();
            std::io::Read::read_to_end(&mut flate2::read::GzDecoder::new(&bytes[..]), &mut out).unwrap();
            out
        } else { bytes }
    };
    let text = path.to_string_lossy();
    let bin = if let Some(stem) = text.strip_suffix(".json.gz") { PathBuf::from(format!("{stem}.bin.gz")) } else { path.with_extension("bin") };
    Suite::from_bytes(&read(path.to_path_buf()), read(bin))
}

#[derive(Default)]
struct Tally { cases: usize, passed: usize, errors: usize, arrays: usize, bytes: usize, failures: Vec<String> }

/// A refusal agrees when it is the same Python exception class with the same text.
fn refusal(label: &str, error: &zlc_plot::projection::ProjectionError, expected: &Value, problems: &mut Vec<String>) {
    if error.kind.python_name() != expected["type"].as_str().unwrap() || error.message != expected["message"].as_str().unwrap() {
        problems.push(format!("{label}{}: {} vs {}: {}", error.kind.python_name(), error.message, expected["type"], expected["message"]));
    }
}

fn run_suite(path: &Path, tally: &mut Tally, filter: Option<&str>) {
    let suite = load(path);
    let mut cache = SchemaCache::default();
    let snapshots = suite.document["snapshots"].as_object().unwrap();
    for case in suite.document["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap_or("?").to_owned();
        if filter.is_some_and(|f| !name.contains(f)) { continue; }
        tally.cases += 1;
        let plan: ProjectionPlan = match serde_json::from_value(case["plan"].clone()) {
            Ok(plan) => plan,
            Err(error) => { tally.failures.push(format!("{name}: plan does not parse: {error}")); continue; }
        };
        let snapshot = Snapshot::from_source(&source(&suite, &snapshots[case["snapshot"].as_str().unwrap()]), &mut cache);
        let mut projection = Projection::new();
        projection.histogram = histogram_state(&case["histogram_previous"]);
        let result = snapshot.and_then(|snapshot| projection.project_snapshot(snapshot, &plan));
        let mut problems = Vec::new();
        match (result, case.get("error")) {
            (Err(error), Some(expected)) => {
                tally.errors += 1;
                refusal("error ", &error, expected, &mut problems);
                // The domain a refused revision leaves behind is the next one's start.
                if let Some(state) = case.get("histogram") { compare(&suite, &GoldenArrays::default(), state, &histogram_json(&projection.histogram), "histogram", &mut problems); }
            }
            (Err(error), None) => problems.push(format!("rust refused: {error}")),
            (Ok(_), Some(expected)) => problems.push(format!("python refused ({}: {}), rust did not", expected["type"], expected["message"])),
            (Ok(projected), None) if case.get("mask_request").is_some() => {
                let state: SelectorFacts = serde_json::from_value(case["mask_request"].clone()).unwrap();
                match (selector_mask(&projected.view, &plan, &projected.payload, &state), case.get("mask_error")) {
                    (Ok(mask), None) => {
                        let mut arrays = GoldenArrays::default();
                        let actual = arrays.array(&mask);
                        compare(&suite, &arrays, &case["mask"], &actual, "mask", &mut problems);
                    }
                    (Err(error), Some(expected)) => refusal("mask error ", &error, expected, &mut problems),
                    (Err(error), None) => problems.push(format!("rust refused the selector mask: {error}")),
                    (Ok(_), Some(expected)) => problems.push(format!("python refused the selector mask ({}: {}), rust did not", expected["type"], expected["message"])),
                }
            }
            (Ok(projected), None) if case.get("fit").is_some() => {
                let fit = &case["fit"];
                let request: FitRequestFacts = serde_json::from_value(fit["request"].clone()).unwrap();
                let context: FitContext = serde_json::from_value(fit["context"].clone()).unwrap();
                let index = |key: &str| fit[key].as_u64().map(|v| v as usize);
                let selected = fit_selection(&projected.view, &plan, &projected.payload, &request, &context, index("facet_index"), index("group_index"));
                match (selected, case.get("fit_error")) {
                    (Ok(selection), None) => {
                        let mut arrays = GoldenArrays::default();
                        let actual = selection.golden(&mut arrays);
                        compare(&suite, &arrays, &case["selection"], &actual, "selection", &mut problems);
                    }
                    (Err(error), Some(expected)) => refusal("fit error ", &error, expected, &mut problems),
                    (Err(error), None) => problems.push(format!("rust refused the fit selection: {error}")),
                    (Ok(_), Some(expected)) => problems.push(format!("python refused the fit selection ({}: {}), rust did not", expected["type"], expected["message"])),
                }
            }
            (Ok(projected), None) => {
                let mut arrays = GoldenArrays::default();
                let payload = projected.payload.golden(&mut arrays);
                compare(&suite, &arrays, &case["payload"], &payload, "payload", &mut problems);
                compare(&suite, &arrays, &case["histogram"], &histogram_json(&projection.histogram), "histogram", &mut problems);
                match (Projection::selection_subject(&projected.view, &plan, &projected.payload), case.get("subject"), case.get("subject_error")) {
                    (Ok(subject), Some(expected), _) => compare(&suite, &arrays, expected, &subject.golden(), "subject", &mut problems),
                    (Err(error), _, Some(expected)) => refusal("subject error ", &error, expected, &mut problems),
                    (Err(error), _, None) => problems.push(format!("subject refused: {error}")),
                    (Ok(_), None, Some(expected)) => problems.push(format!("python refused the subject ({}: {}), rust did not", expected["type"], expected["message"])),
                    (Ok(_), None, None) => {}
                }
            }
        }
        if problems.is_empty() { tally.passed += 1; } else { tally.failures.push(format!("{name}: {}", problems.join(" | "))); }
        let (count, size) = suite.compared.replace((0, 0));
        tally.arrays += count;
        tally.bytes += size;
    }
}

fn report(label: &str, tally: &Tally) {
    eprintln!("{label}: {}/{} cases agree ({} refusals, {} arrays, {:.1} MB compared bit for bit)", tally.passed, tally.cases, tally.errors, tally.arrays, tally.bytes as f64 / 1e6);
    for failure in tally.failures.iter().take(40) { eprintln!("  {failure}"); }
}

fn suites(directory: &Path) -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(directory).map(|entries| entries.filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| { let name = p.to_string_lossy(); name.ends_with(".json") || name.ends_with(".json.gz") }).collect()).unwrap_or_default();
    paths.sort();
    paths
}

#[test]
fn committed_goldens_agree() {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/projection");
    let filter = std::env::var("ZLC_PROJECTION_CASE").ok();
    let mut tally = Tally::default();
    for path in suites(&directory) { run_suite(&path, &mut tally, filter.as_deref()); }
    report("committed", &tally);
    assert!(tally.cases > 0, "no committed golden cases found");
    assert!(tally.failures.is_empty(), "{} of {} golden cases disagree", tally.failures.len(), tally.cases);
}

#[test]
fn recorded_goldens_agree() {
    let Ok(directory) = std::env::var("ZLC_PROJECTION_GOLDEN_DIR") else { return };
    let filter = std::env::var("ZLC_PROJECTION_CASE").ok();
    let mut tally = Tally::default();
    for path in suites(Path::new(&directory)) { run_suite(&path, &mut tally, filter.as_deref()); }
    report("recorded", &tally);
    assert!(tally.failures.is_empty(), "{} of {} golden cases disagree", tally.failures.len(), tally.cases);
}

/// Live sequences replayed through ONE projection owner, so retention and
/// every carry between revisions run as they would live: each revision must
/// still equal its recorded payload and histogram state.
#[test]
fn sequences_agree_statefully() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/projection/synthetic.json.gz");
    let suite = load(&path);
    let mut cache = SchemaCache::default();
    let snapshots = suite.document["snapshots"].as_object().unwrap();
    let mut owners: std::collections::HashMap<String, Projection> = Default::default();
    let (mut cases, mut failures) = (0usize, Vec::new());
    for case in suite.document["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let Some((scenario, _)) = name.split_once('/') else { continue };
        if !scenario.starts_with("sequence_") || case.get("error").is_some() { continue; }
        cases += 1;
        let plan: ProjectionPlan = serde_json::from_value(case["plan"].clone()).unwrap();
        let owner = owners.entry(scenario.to_owned()).or_default();
        if histogram_json(&owner.histogram) != case["histogram_previous"] {
            failures.push(format!("{name}: carried histogram state differs before projecting"));
        }
        let snapshot = Snapshot::from_source(&source(&suite, &snapshots[case["snapshot"].as_str().unwrap()]), &mut cache).unwrap();
        let projected = owner.project_snapshot(snapshot, &plan).unwrap();
        let mut arrays = GoldenArrays::default();
        let mut problems = Vec::new();
        let payload = projected.payload.golden(&mut arrays);
        compare(&suite, &arrays, &case["payload"], &payload, "payload", &mut problems);
        compare(&suite, &arrays, &case["histogram"], &histogram_json(&owner.histogram), "histogram", &mut problems);
        if !problems.is_empty() { failures.push(format!("{name}: {}", problems.join(" | "))); }
    }
    eprintln!("stateful sequences: {}/{cases} revisions agree", cases - failures.len());
    for failure in failures.iter().take(20) { eprintln!("  {failure}"); }
    assert!(cases > 0 && failures.is_empty());
}

/// Speed against the Python projection on recorded live cases written by
/// `tests/gen/bench_python.py` (ZLC_PROJECTION_BENCH = its directory).
#[test]
fn bench_against_python() {
    let Ok(directory) = std::env::var("ZLC_PROJECTION_BENCH") else { return };
    let suite = load(&Path::new(&directory).join("bench.json"));
    let mut cache = SchemaCache::default();
    let snapshots = suite.document["snapshots"].as_object().unwrap();
    let mut groups: std::collections::BTreeMap<String, Vec<(f64, f64)>> = Default::default();
    // Live order: each scenario's revisions through ONE owner (carries and
    // recycled scratch as the app has them), the whole sequence repeated and
    // the fastest of each revision kept.
    // Python times from `tests/gen/time_python.py` (same inputs, same load) when given.
    let python_times: Option<Value> = std::env::var("ZLC_PROJECTION_PYTHON_TIMES").ok().map(|p| serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap());
    let python_of = |case: &Value| -> Option<f64> { match &python_times { Some(t) => t[case["name"].as_str().unwrap()].as_f64(), None => case["python_ns"].as_f64() } };
    let cases: Vec<&Value> = suite.document["cases"].as_array().unwrap().iter()
        .filter(|c| c.get("error").is_none() && python_of(c).is_some()).collect();
    let scenario = |case: &Value| case["name"].as_str().unwrap().split('/').next().unwrap().to_owned();
    let sources: Vec<Source> = cases.iter().map(|case| source(&suite, &snapshots[case["snapshot"].as_str().unwrap()])).collect();
    let plans: Vec<ProjectionPlan> = cases.iter().map(|case| serde_json::from_value(case["plan"].clone()).unwrap()).collect();
    let mut best = vec![f64::INFINITY; cases.len()];
    for _ in 0..7 {
        let mut owners: std::collections::HashMap<String, Projection> = Default::default();
        // The previous frame is still on screen while the next is projected.
        let mut held = None;
        for (index, case) in cases.iter().enumerate() {
            let owner = owners.entry(format!("{}|{}", scenario(case), serde_json::to_string(&plans[index].spec).unwrap())).or_default();
            let started = std::time::Instant::now();
            let snapshot = Snapshot::from_source(&sources[index], &mut cache).unwrap();
            let projected = owner.project_snapshot(snapshot, &plans[index]).unwrap();
            best[index] = best[index].min(started.elapsed().as_secs_f64() * 1e9);
            drop(held.replace(projected));
        }
    }
    for (index, case) in cases.iter().enumerate() {
        let document = &snapshots[case["snapshot"].as_str().unwrap()];
        let shape: Vec<String> = ["repeat", "point", "cell"].iter().flat_map(|d| document["schema"][d]["shape"].as_array().unwrap().iter().map(|v| v.to_string())).collect();
        let label = format!("{:10} ({})", plans[index].spec.kind().value(), shape.join("x"));
        groups.entry(label).or_default().push((python_of(case).unwrap(), best[index]));
    }
    eprintln!("{:44} {:>6} {:>12} {:>12} {:>8}", "case (R x P x cell)", "count", "python ms", "rust ms", "speedup");
    for (label, times) in &groups {
        let median = |mut v: Vec<f64>| { v.sort_by(f64::total_cmp); v[v.len() / 2] };
        let python = median(times.iter().map(|t| t.0).collect()) / 1e6;
        let rust = median(times.iter().map(|t| t.1).collect()) / 1e6;
        eprintln!("{label:44} {:>6} {python:>12.3} {rust:>12.3} {:>7.1}x", times.len(), python / rust);
    }
}

/// Runtime reads projected with carried segments and carried rolling rows
/// agree bit for bit with the same reads converted afresh: a window that
/// grows, slides across page boundaries and skips ordinals, grouped and
/// binned; a finite run that appends repeat after repeat.
#[test]
fn runtime_reads_agree_with_fresh_conversion() {
    use std::sync::Arc;
    use serde_json::json;
    use zlc_core::dataset::{DataBlock, Validity};
    use zlc_core::runtime::{Bundle, Declaration, Input, Runtime, Storage};
    use zlc_core::schema::{Axis, Coordinates, DatasetSchema, Domain, ValidityContract, ValueSchema};
    use zlc_core::{DType, OwnedBuffer};

    let axis = |id: &str, role: &str, len: usize| Axis::new(id.into(), id.into(), role.into(), Coordinates::Index { origin: 0, len }, None, None, None, None).unwrap();
    let scalar = Domain::dense(vec![1], vec![axis("zlc_data.scalar", "scalar", 1)]).unwrap();
    let schema = |repeat: usize, points: usize, dtype: DType| DatasetSchema::new(
        Domain::dense(vec![repeat], vec![axis("repeat", "repeat", repeat)]).unwrap(),
        Domain::dense(vec![points], vec![axis("frame", "readout-event", points)]).unwrap(),
        scalar.clone(), ValueSchema::new(dtype, Some("count".into()), None, ValidityContract::Value).unwrap()).unwrap();
    let shots = schema(1, 2, DType::U16);
    let run_schema = schema(30, 2, DType::F64);
    let row = schema(1, 2, DType::F64);
    let runtime = Runtime::new();
    let generation = runtime.begin("camera", vec![
        Declaration { signal: "shots".into(), schema: shots.clone(), storage: Storage::Latest },
        Declaration { signal: "run".into(), schema: run_schema.clone(), storage: Storage::Finite },
    ], json!({})).unwrap();
    let width = 150;
    let _lease = runtime.retain_history(&generation, "shots", width).unwrap();
    let history = runtime.window_schema(&generation, "shots", width).unwrap();
    let index = json!({"domain": "point", "axis_id": history.domain(zlc_core::schema::DomainRole::Point).axes()[0].id().to_string()});
    let frame = json!({"domain": "point", "axis_id": "frame"});
    let plan = |spec: serde_json::Value, display: serde_json::Value| -> ProjectionPlan { serde_json::from_value(json!({"spec": spec, "display": display})).unwrap() };
    let window_plans = [
        plan(json!({"kind": "rolling", "x": null, "group": null, "reduction": "mean"}), json!({"window": width, "uncertainty": true, "trailing": 3})),
        plan(json!({"kind": "rolling", "x": null, "group": frame, "reduction": "max"}), json!({"window": 40})),
        plan(json!({"kind": "histogram", "group": null, "reduction": "mean"}), json!({"window": width, "bin_count": 20})),
        plan(json!({"kind": "curve", "x": index, "group": frame, "reduction": "mean"}), json!({"window": width, "uncertainty": true})),
    ];
    let run_plans = [
        plan(json!({"kind": "rolling", "x": null, "group": null, "reduction": "mean"}), json!({"window": 30, "uncertainty": true, "trailing": 2})),
        plan(json!({"kind": "curve", "x": frame, "group": null, "reduction": "mean"}), json!({"uncertainty": true})),
    ];
    let mut owners: Vec<(Projection, Projection)> = (0..window_plans.len() + run_plans.len()).map(|_| Default::default()).collect();
    let (mut checked, mut ordinal, mut seed) = (0usize, 0u64, 12345u64);
    let mut next = || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed };
    for shot in 0..230usize {
        ordinal += if shot % 17 == 5 { 3 } else { 1 };
        let values: Vec<u16> = (0..2).map(|_| (next() % 97) as u16).collect();
        let mut inputs = vec![Input::Event { signal: "shots".into(), ordinal,
            data: DataBlock::new(shots.clone(), OwnedBuffer::from_vec(values).into(), Validity::Uniform(true), None).unwrap(), origin: None }];
        if shot < 30 {
            let values: Vec<f64> = (0..2).map(|_| (next() % 1000) as f64 / 7.0).collect();
            inputs.push(Input::Event { signal: "run".into(), ordinal: shot as u64,
                data: DataBlock::new(row.clone(), OwnedBuffer::from_vec(values).into(), Validity::Uniform(true), None).unwrap(), origin: Some([shot, 0]) });
        }
        runtime.commit(&generation, &Bundle { inputs, metadata: json!({}), parents: vec![] }).unwrap();
        let reads = runtime.read_views(&[(generation.clone(), Arc::from("shots"), Some(width)), (generation.clone(), Arc::from("run"), None)]);
        let reads: Vec<_> = reads.into_iter().map(Result::unwrap).collect();
        let jobs = window_plans.iter().map(|p| (&reads[0], p)).chain(run_plans.iter().map(|p| (&reads[1], p)));
        for ((read, plan), (live, fresh)) in jobs.zip(owners.iter_mut()) {
            let carried = live.project_read(read, plan).unwrap();
            let reference = fresh.project(&Source::from_read(read).unwrap(), plan).unwrap();
            let (mut a, mut b) = (GoldenArrays::default(), GoldenArrays::default());
            let (ja, jb) = (carried.payload.golden(&mut a), reference.payload.golden(&mut b));
            let same = ja == jb && a.0.len() == b.0.len()
                && a.0.iter().zip(&b.0).all(|(x, y)| x.dtype() == y.dtype() && x.shape() == y.shape() && x.bytes() == y.bytes());
            assert!(same, "shot {shot}: {:?} disagrees with a fresh conversion", plan.spec.kind());
            assert_eq!(histogram_json(&live.histogram), histogram_json(&fresh.histogram), "shot {shot}: carried histogram domain");
            checked += 1;
        }
    }
    eprintln!("runtime reads: {checked} carried projections agree with fresh conversions");
}

/// A finite run drawn as it grows: each revision reduces only the shots it
/// brought, on from the previous revision's walk (the facet of image means
/// camera frames are drawn with, an image's sums and extremes, curves over a
/// cell axis). Over 300 shots every revision equals a fresh projection of
/// it bit for bit, invalid shots and non-finite samples included.
#[test]
fn growing_run_reduces_only_the_shots_it_brought() {
    use std::sync::Arc;
    use serde_json::json;
    use zlc_core::dataset::{DataBlock, Validity};
    use zlc_core::runtime::{Bundle, Declaration, Input, Runtime, Storage};
    use zlc_core::schema::{Axis, Coordinates, DatasetSchema, Domain, ValidityContract, ValueSchema};
    use zlc_core::{DType, OwnedBuffer};

    const SHOTS: usize = 300;
    let (frames, ny, nx) = (3usize, 6usize, 5usize);
    let axis = |id: &str, role: &str, len: usize| Axis::new(id.into(), id.into(), role.into(), Coordinates::Index { origin: 0, len }, None, None, None, None).unwrap();
    let schema = |repeat: usize| DatasetSchema::new(
        Domain::dense(vec![repeat], vec![axis("repeat", "repeat", repeat)]).unwrap(),
        Domain::dense(vec![frames], vec![axis("frame", "readout-event", frames)]).unwrap(),
        Domain::dense(vec![ny, nx], vec![axis("y", "spatial-y", ny), axis("x", "spatial-x", nx)]).unwrap(),
        ValueSchema::new(DType::F64, Some("count".into()), None, ValidityContract::Value).unwrap()).unwrap();
    let (run, shot) = (schema(SHOTS), schema(1));
    let runtime = Runtime::new();
    let generation = runtime.begin("camera", vec![Declaration { signal: "frames".into(), schema: run, storage: Storage::Finite }], json!({})).unwrap();
    let reference = |domain: &str, id: &str| json!({"domain": domain, "axis_id": id});
    let (frame, y, x) = (reference("point", "frame"), reference("cell_data", "y"), reference("cell_data", "x"));
    let image = |reduction: &str| json!({"kind": "image", "x": x, "y": y, "reduction": reduction});
    let specs = [
        json!({"kind": "facet_grid", "facet": frame, "cell": image("mean")}),
        image("sum"), image("min"), image("max"),
        json!({"kind": "curve", "x": x, "group": frame, "reduction": "mean"}),
        json!({"kind": "facet_grid", "facet": frame, "cell": {"kind": "curve", "x": x, "group": null, "reduction": "first"}}),
    ];
    let plans: Vec<ProjectionPlan> = specs.iter().map(|spec| serde_json::from_value(json!({"spec": spec, "display": {}})).unwrap()).collect();
    let mut owners: Vec<Projection> = plans.iter().map(|_| Projection::new()).collect();
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed };
    let mut checked = 0usize;
    for index in 0..SHOTS {
        // Full-mantissa samples of wide magnitude (sums depend on their order), some not finite.
        let values: Vec<f64> = (0..frames * ny * nx).map(|_| {
            let bits = next();
            if bits % 97 == 0 { f64::NAN } else { (bits >> 11) as f64 / (1u64 << 53) as f64 * 10f64.powi((bits % 7) as i32) - 3.0 }
        }).collect();
        // Every seventh shot holds an invalid frame.
        let validity = if index % 7 == 3 { Validity::Mask(OwnedBuffer::from_vec((0..shot.component_mask_len()).map(|k| k != 1).collect())) }
                       else { Validity::Uniform(true) };
        let data = DataBlock::new(shot.clone(), OwnedBuffer::from_vec(values).into(), validity, None).unwrap();
        runtime.commit(&generation, &Bundle { inputs: vec![Input::Event { signal: "frames".into(), ordinal: index as u64, data, origin: Some([index, 0]) }],
            metadata: json!({}), parents: vec![] }).unwrap();
        let read = runtime.read_views(&[(generation.clone(), Arc::from("frames"), None)]).pop().unwrap().unwrap();
        let source = Source::from_read(&read).unwrap();
        for (plan, live) in plans.iter().zip(owners.iter_mut()) {
            let carried = live.project(&source, plan).unwrap();
            let fresh = Projection::new().project(&source, plan).unwrap();
            let (mut a, mut b) = (GoldenArrays::default(), GoldenArrays::default());
            let (ja, jb) = (carried.payload.golden(&mut a), fresh.payload.golden(&mut b));
            let same = ja == jb && a.0.len() == b.0.len()
                && a.0.iter().zip(&b.0).all(|(p, q)| p.dtype() == q.dtype() && p.shape() == q.shape() && p.bytes() == q.bytes());
            assert!(same, "shot {index}: {} differs from the reduction of the whole run", plan.spec.kind().value());
            checked += 1;
        }
    }
    eprintln!("growing run: {checked} revisions reduced from the previous walk agree with fresh projections");
}

/// Histograms of a finite run as it grows: the whole pool and a window
/// counted from each shot's carried extremes and counts (every shot counted
/// again only when the edges move), a reduction walked on. Over 300 shots
/// every revision equals the same projection of the run read as ONE block,
/// with nothing carried but the retained edges -- bins, counts, retained
/// domain and refusals -- float with non-finite samples and invalid frames,
/// float without, and integers. A window is the last shots' histogram whatever the rest of the
/// run holds: a valid, finite run bins its window exactly as the same run
/// with one non-finite sample in a shot the window has left.
#[test]
fn growing_run_histograms_count_only_the_shots_they_brought() {
    use std::sync::Arc;
    use serde_json::json;
    use zlc_core::dataset::{DataBlock, Validity};
    use zlc_core::runtime::{Bundle, Declaration, Input, Runtime, Storage};
    use zlc_core::schema::{Axis, Coordinates, DatasetSchema, Domain, ValidityContract, ValueSchema};
    use zlc_core::{AnyBuffer, DType, OwnedBuffer};
    use zlc_plot::projection::SourceData;

    const SHOTS: usize = 300;
    let (frames, ny, nx) = (3usize, 6usize, 5usize);
    let axis = |id: &str, role: &str, len: usize| Axis::new(id.into(), id.into(), role.into(), Coordinates::Index { origin: 0, len }, None, None, None, None).unwrap();
    let schema = |repeat: usize, dtype: DType| DatasetSchema::new(
        Domain::dense(vec![repeat], vec![axis("repeat", "repeat", repeat)]).unwrap(),
        Domain::dense(vec![frames], vec![axis("frame", "readout-event", frames)]).unwrap(),
        Domain::dense(vec![ny, nx], vec![axis("y", "spatial-y", ny), axis("x", "spatial-x", nx)]).unwrap(),
        ValueSchema::new(dtype, Some("count".into()), None, ValidityContract::Value).unwrap()).unwrap();
    let signals = [("noisy", DType::F64), ("clean", DType::F64), ("early_nan", DType::F64), ("counts", DType::U16)];
    const WINDOW: usize = 40;
    let runtime = Runtime::new();
    let generation = runtime.begin("camera", signals.iter().map(|(name, dtype)| Declaration { signal: (*name).into(), schema: schema(SHOTS, *dtype),
        storage: Storage::Finite }).collect(), json!({})).unwrap();
    let reference = |domain: &str, id: &str| json!({"domain": domain, "axis_id": id});
    let (frame, y, x) = (reference("point", "frame"), reference("cell_data", "y"), reference("cell_data", "x"));
    let histogram = |group: serde_json::Value, reduced: serde_json::Value| json!({"kind": "histogram", "group": group, "reduced": reduced, "reduction": "mean"});
    let plan = |spec: serde_json::Value, display: serde_json::Value| -> ProjectionPlan { serde_json::from_value(json!({"spec": spec, "display": display})).unwrap() };
    let whole = json!({"window": 1, "bin_count": 20});
    let windowed = json!({"window": WINDOW, "bin_count": 20});
    let plans = [
        plan(histogram(json!(null), json!([])), whole.clone()),
        plan(histogram(frame.clone(), json!([])), whole.clone()),
        plan(histogram(json!(null), json!([])), json!({"window": 1, "bin_count": 20, "x_relim_mode": "tight"})),
        plan(histogram(json!(null), json!([])), windowed.clone()),
        plan(histogram(frame.clone(), json!([])), windowed.clone()),
        plan(histogram(json!(null), json!([y, x])), whole.clone()),
        plan(histogram(json!(null), json!([y, x])), windowed.clone()),
        plan(json!({"kind": "facet_grid", "facet": frame, "cell": histogram(json!(null), json!([]))}), json!({"bin_count": 20})),
        plan(json!({"kind": "facet_grid", "facet": frame, "cell": histogram(json!(null), json!([]))}), windowed.clone()),
    ];
    let window_plans = [3usize, 4, 6, 8];
    let mut owners: Vec<Vec<Projection>> = signals.iter().map(|_| plans.iter().map(|_| Projection::new()).collect()).collect();
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut next = || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed };
    let (mut checked, mut refused, mut witnessed) = (0usize, 0usize, 0usize);
    // Every shot so far, per signal: float values, integer values, component validity.
    let mut merged: Vec<(Vec<f64>, Vec<u16>, Vec<bool>)> = signals.iter().map(|_| Default::default()).collect();
    for index in 0..SHOTS {
        let mut clean_windows: Vec<String> = Vec::new();
        let shot = schema(1, DType::F64);
        let count = frames * ny * nx;
        // The range widens now and then (the edges move), samples of every magnitude.
        let scale = 10f64.powi((index / 60) as i32);
        let noisy: Vec<f64> = (0..count).map(|_| { let bits = next(); if bits % 89 == 0 { f64::NAN } else { ((bits >> 11) as f64 / (1u64 << 53) as f64 - 0.3) * scale } }).collect();
        let clean: Vec<f64> = (0..count).map(|_| ((next() >> 11) as f64 / (1u64 << 53) as f64) * scale).collect();
        // The clean run with one non-finite sample in its first shot.
        let early_nan: Vec<f64> = clean.iter().enumerate().map(|(k, v)| if index == 0 && k == 7 { f64::NAN } else { *v }).collect();
        let counts: Vec<u16> = (0..count).map(|_| (next() % (50 + index as u64 * 7)) as u16).collect();
        let marks = |every: usize| (0..shot.component_mask_len()).map(|k| index % every != 2 || k != 0).collect::<Vec<bool>>();
        let validity = |marks: &Vec<bool>| if marks.iter().all(|m| *m) { Validity::Uniform(true) } else { Validity::Mask(OwnedBuffer::from_vec(marks.clone())) };
        let shots = [(noisy, vec![], marks(7)), (clean, vec![], vec![true; shot.component_mask_len()]), (early_nan, vec![], vec![true; shot.component_mask_len()]), (vec![], counts, marks(11))];
        for ((floats, integers, valid), (f, u, m)) in shots.iter().zip(merged.iter_mut()) { f.extend(floats); u.extend(integers); m.extend(valid); }
        let block = |(floats, integers, _): &(Vec<f64>, Vec<u16>, Vec<bool>), dtype: DType| -> AnyBuffer {
            if dtype == DType::U16 { OwnedBuffer::from_vec(integers.clone()).into() } else { OwnedBuffer::from_vec(floats.clone()).into() } };
        let blocks: Vec<(AnyBuffer, Validity, DType)> = shots.iter().zip(&signals).map(|(shot, (_, dtype))| (block(shot, *dtype), validity(&shot.2), *dtype)).collect();
        let inputs = signals.iter().zip(blocks).map(|((name, _), (values, validity, dtype))| Input::Event { signal: (*name).into(), ordinal: index as u64,
            data: DataBlock::new(schema(1, dtype), values, validity, None).unwrap(), origin: Some([index, 0]) }).collect();
        runtime.commit(&generation, &Bundle { inputs, metadata: json!({}), parents: vec![] }).unwrap();
        for (((name, dtype), owners), history) in signals.iter().zip(owners.iter_mut()).zip(&merged) {
            let read = runtime.read_views(&[(generation.clone(), Arc::from(*name), None)]).pop().unwrap().unwrap();
            let source = Source::from_read(&read).unwrap();
            let one_block = DataBlock::new(schema(index + 1, *dtype), block(history, *dtype), validity(&history.2), None).unwrap();
            let whole = Source { schema: source.schema.clone(), data: SourceData::Segments(vec![([0, 0], one_block)]), block_id: source.block_id.clone(),
                generation: source.generation.clone(), revision: source.revision, window: source.window };
            for (position, (plan, live)) in plans.iter().zip(owners.iter_mut()).enumerate() {
                // The run as one block, with nothing carried but the retained edges.
                let mut whole_run = Projection::new();
                whole_run.histogram = live.histogram.clone();
                let (carried, fresh) = (live.project(&source, plan), whole_run.project(&whole, plan));
                let what = format!("{name} shot {index}: {}", serde_json::to_string(&plan.spec).unwrap());
                match (carried, fresh) {
                    (Ok(carried), Ok(fresh)) => {
                        let (mut a, mut b) = (GoldenArrays::default(), GoldenArrays::default());
                        let (ja, jb) = (carried.payload.golden(&mut a), fresh.payload.golden(&mut b));
                        let same = ja == jb && a.0.len() == b.0.len()
                            && a.0.iter().zip(&b.0).all(|(p, q)| p.dtype() == q.dtype() && p.shape() == q.shape() && p.bytes() == q.bytes());
                        assert!(same, "{what}: differs from the run read as one block");
                    }
                    (Err(carried), Err(fresh)) => {
                        assert_eq!(carried.message, fresh.message, "{what}");
                        assert!(!window_plans.contains(&position), "{what}: a window refused: {}", carried.message);
                        refused += 1;
                    }
                    (carried, fresh) => panic!("{what}: {:?} vs {:?}", carried.err().map(|e| e.message), fresh.err().map(|e| e.message)),
                }
                assert_eq!(histogram_json(&live.histogram), histogram_json(&whole_run.histogram), "{what}: retained edges");
                checked += 1;
                // Once the window has left the first shot, the clean run and the one with
                // a non-finite sample there hold the same window: the same histogram.
                if window_plans.contains(&position) && index >= WINDOW && (*name == "clean" || *name == "early_nan") {
                    let payload = Projection::new().project(&source, plan).unwrap().payload;
                    let mut arrays = GoldenArrays::default();
                    let written = format!("{} {:?}", payload.golden(&mut arrays), arrays.0.iter().map(|a| a.bytes().to_vec()).collect::<Vec<_>>());
                    if *name == "clean" { clean_windows.push(written); }
                    else { assert!(clean_windows.remove(0) == written, "{what}: the clean window differs from the same window beside a non-finite shot"); witnessed += 1; }
                }
            }
        }
    }
    eprintln!("growing run histograms: {checked} revisions agree with the run read as one block ({refused} refused alike); {witnessed} clean windows equal their witness");
    assert!(witnessed == window_plans.len() * (SHOTS - WINDOW));
}
