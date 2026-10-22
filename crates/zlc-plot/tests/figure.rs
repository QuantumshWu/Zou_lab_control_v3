//! The figure document against `zlc_plot.figure_artifact` and
//! `zlc_science.figure`: every provenance capture, every save byte for byte
//! (archive and image), and what `read_document` answers for every archive
//! (resolved path, metadata, planes, overlay geometry and status plane, wire
//! target) or the class and message it refuses with.
//!
//! Committed: `tests/data/figure/` (`gen/figure_cases.py DIR`).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::Value as Json;
use sha2::{Digest, Sha256};
use zlc_archive::durable::resolve;
use zlc_archive::json::{dumps, loads, Style};
use zlc_archive::provenance::capture_figure_provenance;
use zlc_archive::{read_archive, ReadDataset};
use zlc_archive::tree::schema_tree;
use zlc_core::dataset::Validity;
use zlc_core::restrict::Source as Planes;
use zlc_core::runtime::{Bundle, Declaration, Input, Runtime, Storage};
use zlc_core::store::Snapshot;
use zlc_plot::configuration::resolve::Defaults;
use zlc_plot::configuration::value::Value;
use zlc_plot::document::{DocumentRead, FigureDocument, FigureSaver, SaveRequest};
use zlc_plot::figure::{read_document, read_figure_plot, save_run_figure, ArchiveSaver, RunFigure};
use zlc_plot::fit::FitEngine;
use zlc_plot::interaction::gesture::InteractionDefaults;
use zlc_plot::projection::Source;
use zlc_plot::render::panel::NativeRendererFactory;
use zlc_plot::session::{EngineFitLayer, ImagePointOverlay, PointStatus, SessionServices};

fn data() -> PathBuf { Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/figure") }

fn golden() -> Json {
    let bytes = std::fs::read(data().join("figure.json.gz")).unwrap();
    let mut text = String::new();
    std::io::Read::read_to_string(&mut flate2::read::GzDecoder::new(&bytes[..]), &mut text).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn digest(bytes: &[u8]) -> String { format!("{:x}", Sha256::digest(bytes)) }

/// A path under `root`, as the golden writes it.
fn relative(path: &str, root: &Path) -> String {
    Path::new(path).strip_prefix(root).unwrap_or(Path::new(path)).to_string_lossy().replace('\\', "/")
}

fn refusal(kind: &str, message: &str) -> Json { serde_json::json!([kind, message]) }

#[derive(Clone)]
struct SavedRead { source: Arc<Source>, provenance: String }

impl DocumentRead for SavedRead {
    fn source(&self) -> &Arc<Source> { &self.source }
    fn provenance_json(&self) -> String { self.provenance.clone() }
}

#[test]
fn provenance_captures_like_python() {
    let golden = golden();
    let mut failures = Vec::new();
    let cases = golden["provenance"].as_array().unwrap();
    for case in cases {
        let tree = loads(case["tree"].as_str().unwrap()).unwrap();
        let label = case["name"].as_str().unwrap();
        match (capture_figure_provenance((!tree.is_null()).then_some(&tree)), case.get("ok")) {
            (Ok((lineage, source)), Some(expected)) => {
                if dumps(&lineage, Style::Figure).unwrap() != expected[0].as_str().unwrap() || dumps(&source, Style::Figure).unwrap() != expected[1].as_str().unwrap() {
                    failures.push(format!("{label}: capture differs"));
                }
            }
            (Err(error), None) if refusal(error.kind.python_name(), &error.message) == case["error"] => {}
            (outcome, _) => failures.push(format!("{label}: {:?} vs {}", outcome.map(|_| ()), case.get("error").unwrap_or(&Json::Null))),
        }
    }
    assert!(failures.is_empty(), "{} of {} captures differ:\n{}", failures.len(), cases.len(), failures.join("\n"));
    assert!(cases.len() >= 25);
}

/// The read a host hands a save of this data: an event publication of a
/// dense block, or a mounted complete dataset (its placements), read back
/// from the runtime under the archived identity.
fn runtime_source(dataset: &ReadDataset) -> Source {
    let runtime = Runtime::new();
    let storage = if dataset.dense { Storage::Latest } else { Storage::Snapshot };
    let generation = runtime.begin("figure", vec![Declaration { signal: "data".into(), schema: dataset.snapshot.schema().clone(), storage }],
        serde_json::json!({})).unwrap();
    let input = match dataset.source() {
        Planes::Dense(block) => Input::Event { signal: "data".into(), ordinal: 1, data: block, origin: None },
        Planes::Placed { .. } => Input::Dataset { signal: "data".into(), ordinal: Some(1), snapshot: dataset.snapshot.clone() },
    };
    runtime.commit(&generation, &Bundle { inputs: vec![input], metadata: serde_json::json!({}), parents: vec![] }).unwrap();
    let read = runtime.read_views(&[(generation, Arc::from("data"), None)]).pop().unwrap().unwrap();
    let source = Source::from_read(&read).unwrap();
    let reference = &dataset.reference;
    Source { block_id: reference.block_id.as_str().into(), generation: reference.stream_generation.as_str().into(), revision: reference.revision, ..source }
}

/// Every save writes Python's bytes, from the archived dataset as read back
/// and from the runtime read a host hands the same data.
#[test]
fn figures_save_like_python() {
    let golden = golden();
    let out = Path::new(env!("CARGO_TARGET_TMPDIR")).join("figure");
    std::fs::create_dir_all(&out).unwrap();
    let root = resolve(&out).unwrap();
    let saves = golden["saves"].as_array().unwrap();
    // A refused save never reaches its data; it is given the first saved one.
    let first = saves.iter().find_map(|case| case.get("ok")).unwrap()[1].as_str().unwrap().to_owned();
    let mut failures = Vec::new();
    for case in saves {
        let name = case["name"].as_str().unwrap();
        let written = case.get("ok").map_or(first.as_str(), |paths| paths[1].as_str().unwrap());
        let archive = read_archive(&data().join(written)).unwrap();
        let plot = read_figure_plot(&archive, "data").unwrap();
        let mut recipe = Value::from_json(&archive.info.get("sections").unwrap().get("plot").unwrap().get("data").unwrap().to_serde());
        let Value::Map(recipe) = &mut recipe else { unreachable!() };
        recipe.insert("overlay", Value::None);
        let image = hex(case["image"].as_str().unwrap());
        let overlay = plot.overlay.map(Arc::new);
        for (route, source) in [("archived", Source::of_archived(&plot.data)), ("runtime", runtime_source(&plot.data))] {
            let label = format!("{name} ({route})");
            let read = SavedRead { source: Arc::new(source), provenance: case["tree"].as_str().unwrap().to_owned() };
            let target = out.join(route).join(case["base"].as_str().unwrap()).to_string_lossy().into_owned();
            let request = SaveRequest { path: &target, read: &read, overlay: overlay.clone() };
            let result = FigureSaver::<SavedRead>::prepare(&ArchiveSaver, &request)
                .and_then(|write| write.write(recipe.clone(), &mut |_format| Ok(image.clone())));
            match (result, case.get("ok")) {
                (Ok(paths), Some(expected)) => {
                    let paths: Vec<String> = paths.iter().map(|path| relative(path, &root)).collect();
                    let expected: Vec<String> = expected.as_array().unwrap().iter().map(|path| path.as_str().unwrap().replacen("save/", &format!("{route}/"), 1)).collect();
                    if paths != expected { failures.push(format!("{label}: wrote {paths:?}, expected {expected:?}")); continue; }
                    if std::fs::read(root.join(&expected[0])).unwrap() != image { failures.push(format!("{label}: image bytes differ")); }
                    if digest(&std::fs::read(root.join(&expected[1])).unwrap()) != case["archive_sha256"].as_str().unwrap() {
                        failures.push(format!("{label}: archive bytes differ"));
                    }
                }
                (Err(error), None) if refusal(error.python_name(), &error.message) == case["error"] => {}
                (outcome, _) => failures.push(format!("{label}: {:?} vs {}", outcome, case.get("error").unwrap_or(&Json::Null))),
            }
        }
    }
    assert!(failures.is_empty(), "{} of {} saves differ:
{}", failures.len(), 2 * saves.len(), failures.join("
"));
    assert!(saves.len() >= 13);
}

/// A run's figure (`run.figure`): the saved recipe given back as a run gives
/// it (no overlay key) through a fresh export session writes the same
/// archive, and the image it draws.
#[test]
fn run_figures_save_their_recipe() {
    let golden = golden();
    let out = Path::new(env!("CARGO_TARGET_TMPDIR")).join("run_figure");
    let services = SessionServices { renderers: Arc::new(NativeRendererFactory), fit: Arc::new(EngineFitLayer::new(Arc::new(FitEngine::default()))),
                                     defaults: Arc::new(Defaults::default()), registry: None, interaction: InteractionDefaults::default() };
    let mut failures = Vec::new();
    let saves: Vec<&Json> = golden["saves"].as_array().unwrap().iter().filter(|case| case.get("ok").is_some()).collect();
    for case in &saves {
        let name = case["name"].as_str().unwrap();
        let archive = read_archive(&data().join(case["ok"][1].as_str().unwrap())).unwrap();
        let plot = read_figure_plot(&archive, "data").unwrap();
        let sections = archive.info.get("sections").unwrap().to_serde();
        let mut recipe = sections["plot"]["data"].clone();
        recipe.as_object_mut().unwrap().remove("overlay");
        let figure = RunFigure { path: out.join(case["base"].as_str().unwrap()), data: Source::of_archived(&plot.data), overlay: plot.overlay,
                                 recipe, source: sections["source"].clone(), lineage: sections["lineage"].clone(), image: true };
        match save_run_figure(figure, &services) {
            Ok((image, archive)) => {
                if digest(&std::fs::read(&archive).unwrap()) != case["archive_sha256"].as_str().unwrap() { failures.push(format!("{name}: archive bytes differ")); }
                // Each format's own file (the SVG and PDF writers' pictures are checked against v2's by eye).
                let bytes = image.as_ref().and_then(|image| std::fs::read(image).ok()).unwrap_or_default();
                let suffix = image.as_ref().and_then(|image| image.extension()).map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
                let magic: &[u8] = match suffix.as_str() { "png" => b"\x89PNG", "svg" => b"<?xml", "pdf" => b"%PDF", _ => b"?" };
                if !bytes.starts_with(magic) { failures.push(format!("{name}: no {suffix} image")); }
            }
            outcome => failures.push(format!("{name}: {outcome:?}")),
        }
    }
    assert!(failures.is_empty(), "{} of {} run figures differ:\n{}", failures.len(), saves.len(), failures.join("\n"));
}

fn hex(text: &str) -> Vec<u8> { (0..text.len()).step_by(2).map(|index| u8::from_str_radix(&text[index..index + 2], 16).unwrap()).collect() }

/// The planes `describe_dataset` recorded for one snapshot.
fn plane_differs(snapshot: &Snapshot, expected: &Json) -> Option<String> {
    let segments: Vec<_> = snapshot.segments().collect();
    let wanted = expected["segments"].as_array().unwrap();
    if segments.len() != wanted.len() { return Some(format!("{} segments, expected {}", segments.len(), wanted.len())); }
    if dumps(&schema_tree(snapshot.schema()), Style::Figure).unwrap() != expected["schema"].as_str().unwrap() { return Some("schema".into()); }
    for (index, (segment, want)) in segments.iter().zip(wanted).enumerate() {
        let pair = |value: &Json| [value[0].as_u64().unwrap() as usize, value[1].as_u64().unwrap() as usize];
        let data = segment.data();
        let validity = match (data.validity(), &want["validity"]) {
            (Validity::Uniform(flag), Json::Bool(expected)) => flag == expected,
            (Validity::Mask(mask), Json::Object(expected)) => digest(mask.as_bytes()) == expected["sha256"].as_str().unwrap(),
            _ => false,
        };
        let sigma = match (data.sigma(), want["sigma"].as_str()) { (None, None) => true, (Some(sigma), Some(expected)) => digest(sigma.as_bytes()) == expected, _ => false };
        if segment.origin() != pair(&want["origin"]) || segment.extent() != pair(&want["extent"])
            || dumps(&schema_tree(data.schema()), Style::Figure).unwrap() != want["schema"].as_str().unwrap()
            || digest(data.values().as_bytes()) != want["values"].as_str().unwrap() || !validity || !sigma {
            return Some(format!("segment {index}"));
        }
    }
    None
}

#[test]
fn figures_read_like_python() {
    let golden = golden();
    let root = resolve(&data()).unwrap();
    let fit = Arc::new(EngineFitLayer::new(Arc::new(FitEngine::default())));
    let defaults = Arc::new(Defaults::default());
    let reads = golden["reads"].as_array().unwrap();
    let mut failures = Vec::new();
    for case in reads {
        let label = case["file"].as_str().unwrap();
        let path = data().join(label);
        let opened = read_document(&path.to_string_lossy(), fit.clone(), defaults.clone(), None);
        let opened = match (opened, case.get("ok")) {
            (Ok(opened), Some(expected)) => (opened, expected),
            (Err(error), None) => {
                let message = case["error"][1].as_str().unwrap().replace("{root}", &root.display().to_string().replace('\\', "\\\\"));
                if refusal(error.python_name(), &error.message) != refusal(case["error"][0].as_str().unwrap(), &message) {
                    failures.push(format!("{label}: {error:?} vs {}", case["error"]));
                }
                continue;
            }
            (outcome, _) => { failures.push(format!("{label}: {:?} vs {}", outcome.err(), case.get("error").unwrap_or(&Json::Null))); continue; }
        };
        let (opened, expected) = opened;
        if relative(&opened.path, &root) != expected["path"].as_str().unwrap() { failures.push(format!("{label}: path {}", opened.path)); }
        if opened.info != serde_json::from_str::<Json>(expected["info"].as_str().unwrap()).unwrap() { failures.push(format!("{label}: info differs")); }
        let items = expected["items"].as_array().unwrap();
        if opened.items.len() != items.len() { failures.push(format!("{label}: {} items", opened.items.len())); continue; }
        for (item, want) in opened.items.iter().zip(items) {
            let key = &item.key;
            if key != want["key"].as_str().unwrap() { failures.push(format!("{label}: key {key}")); continue; }
            if item.target != serde_json::from_str::<Json>(want["target"].as_str().unwrap()).unwrap() {
                failures.push(format!("{label}[{key}]: target {} vs {}", item.target, want["target"].as_str().unwrap()));
            }
            if let Some(problem) = plane_differs(&item.plane, &want["plane"]) { failures.push(format!("{label}[{key}]: plane {problem}")); }
            match (&item.overlay_plane, &want["overlay_plane"]) {
                (None, Json::Null) => {}
                (Some(plane), expected) if !expected.is_null() => {
                    if let Some(problem) = plane_differs(plane, expected) { failures.push(format!("{label}[{key}]: status plane {problem}")); }
                }
                _ => failures.push(format!("{label}[{key}]: status plane presence")),
            }
            let overlay = item.overlay.as_ref().map(|overlay| serde_json::json!({
                "revision": overlay.revision, "coordinates": &overlay.coordinates[..], "shape": [overlay.count(), 2],
                "point_ids": overlay.point_ids, "labels": overlay.labels,
                "static_statuses": overlay.static_statuses.as_ref().map(|statuses| statuses.iter().map(|status| status.value()).collect::<Vec<_>>()),
                "paths_xy": overlay.paths_xy.as_ref().map(|(values, vertices)| serde_json::json!({"values": &values[..], "shape": [overlay.count(), vertices, 2]})),
                "status": overlay.status.is_some(),
            }));
            if overlay.unwrap_or(Json::Null) != want["overlay"] { failures.push(format!("{label}[{key}]: overlay differs")); }
        }
    }
    assert!(failures.is_empty(), "{} of {} reads differ:\n{}", failures.len(), reads.len(), failures.join("\n"));
    assert!(reads.len() >= 57);
}

/// An image read with the point layer drawn on it (a review's candidates as it marks them).
#[derive(Clone)]
struct LayeredRead { source: Arc<Source>, layer: Arc<ImagePointOverlay> }

impl DocumentRead for LayeredRead {
    fn source(&self) -> &Arc<Source> { &self.source }
    fn provenance_json(&self) -> String { "{}".into() }
    fn image_overlay(&self) -> Option<Option<Arc<ImagePointOverlay>>> { Some(Some(self.layer.clone())) }
}

/// Rings marked again and again over the same picture (the compose keeps the
/// canvas under them) draw what a fresh document draws of each marking.
#[test]
fn rings_over_the_same_picture_draw_as_a_fresh_document() {
    let services = SessionServices { renderers: Arc::new(NativeRendererFactory), fit: Arc::new(EngineFitLayer::new(Arc::new(FitEngine::default()))),
                                     defaults: Arc::new(Defaults::default()), registry: None, interaction: InteractionDefaults::default() };
    let archive = read_archive(&data().join("save/image-static/fig.npz")).unwrap();
    let plot = read_figure_plot(&archive, "data").unwrap();
    let base = plot.overlay.unwrap();
    let source = Arc::new(Source::of_archived(&plot.data));
    let count = base.count();
    assert!(count >= 3);
    // The markings: none, the first excluded, then the second selected too, then the first restored.
    let marks: Vec<Vec<PointStatus>> = vec![
        vec![PointStatus::Unknown; count],
        (0..count).map(|i| if i == 0 { PointStatus::Invalid } else { PointStatus::Unknown }).collect(),
        (0..count).map(|i| match i { 0 => PointStatus::Invalid, 1 => PointStatus::Occupied, _ => PointStatus::Unknown }).collect(),
        (0..count).map(|i| if i == 1 { PointStatus::Occupied } else { PointStatus::Unknown }).collect(),
    ];
    let read = |index: usize| LayeredRead { source: source.clone(),
        layer: Arc::new(ImagePointOverlay { revision: index as u64, static_statuses: Some(marks[index].clone()), ..base.clone() }) };
    let input = serde_json::json!([{"panel": "p", "kind": "image", "window": 1, "signal": "data", "epoch": 1}]);
    let targets = serde_json::json!({"p": {"semantic": {"kind": "image"}}});
    let all: std::collections::BTreeSet<String> = ["p".to_owned()].into();
    let run = |document: &mut FigureDocument<LayeredRead>, action: &str, index: usize| -> Vec<u8> {
        let command = serde_json::json!({"id": index + 1, "action": action, "panel": "p", "inputs": input, "targets": targets});
        let execution = document.execute(&command.to_string(), [("p".to_owned(), read(index))].into(), None, None).unwrap();
        document.finish_presentation(&all, &all);
        execution.fronts["p"].pixels.as_slice().to_vec()
    };
    let fresh = |index: usize| {
        let mut document = FigureDocument::new(services.clone(), None);
        document.execute(r#"{"id": 0, "action": "ready", "dpr": 2.0}"#, Default::default(), None, None).unwrap();
        document.finish_presentation(&all, &all);
        run(&mut document, "sync_panels", index)
    };
    let mut marked = FigureDocument::new(services.clone(), None);
    marked.execute(r#"{"id": 0, "action": "ready", "dpr": 2.0}"#, Default::default(), None, None).unwrap();
    marked.finish_presentation(&all, &all);
    let first = run(&mut marked, "sync_panels", 0);
    assert_eq!(first, fresh(0));
    for index in 1..marks.len() {
        let pixels = run(&mut marked, "configure", index);
        assert_ne!(pixels, first, "marking {index} drew no change");
        assert!(pixels == fresh(index), "marking {index} differs from a fresh draw of it");
    }
}
