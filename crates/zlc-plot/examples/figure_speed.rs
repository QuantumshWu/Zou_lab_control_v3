//! Native figure-archive and coordinate-ROI timings against
//! `tests/gen/figure_bench.py`'s Python ones, on the same files; every
//! archive the native side writes is checked byte for byte against Python's.
//!
//! `figure_speed OUTDIR` (after `figure_bench.py OUTDIR`)
use std::sync::Arc;
use std::time::Instant;

use zlc_archive::manifest::ArchiveDataset;
use zlc_archive::{read_archive, write_figure_file, Entry, Json};
use zlc_core::restrict::Source as Planes;
use zlc_plot::configuration::resolve::Defaults;
use zlc_plot::configuration::value::Value;
use zlc_plot::document::{DocumentRead, FigureSaver, SaveRequest};
use zlc_plot::figure::{read_document, read_figure_plot, ArchiveSaver};
use zlc_plot::fit::FitEngine;
use zlc_plot::projection::Source;
use zlc_plot::roi::plan_from_metadata;
use zlc_plot::session::EngineFitLayer;

fn best<T>(repeat: usize, mut run: impl FnMut() -> T) -> (f64, T) {
    let mut fastest = f64::INFINITY;
    let mut last = None;
    for _ in 0..repeat {
        let started = Instant::now();
        let value = run();
        fastest = fastest.min(started.elapsed().as_secs_f64());
        last = Some(value);
    }
    (fastest, last.expect("at least one run"))
}

#[derive(Clone)]
struct SavedRead { source: Arc<Source>, provenance: String }

impl DocumentRead for SavedRead {
    fn source(&self) -> &Arc<Source> { &self.source }
    fn provenance_json(&self) -> String { self.provenance.clone() }
}

fn main() {
    let out = std::path::PathBuf::from(std::env::args().nth(1).expect("OUTDIR"));
    let bench: serde_json::Value = serde_json::from_slice(&std::fs::read(out.join("bench.json")).unwrap()).unwrap();
    let python = |name: &str| bench["python"][name].as_f64().unwrap();
    let scratch = out.join("native");
    std::fs::create_dir_all(scratch.join("panel")).unwrap();
    let mut rows = Vec::new();
    let mut row = |name: &str, python: f64, native: f64, note: String| {
        rows.push(format!("{name:<16} python {:>8.1} ms   native {:>8.1} ms   {:>6.1}x   {note}", python * 1e3, native * 1e3, python / native));
    };

    // The document: four camera panels and a 500-shot history.
    let document = out.join("document.npz");
    let (read, archive) = best(3, || read_archive(&document).unwrap());
    row("document read", python("document_read"), read, "Python: OwnedSnapshots".into());
    row("document open", python("document_open"), read, "read + NativeDataset.from_science".into());
    // Datasets in the order Python wrote their members.
    let order: Vec<String> = zlc_archive::zip::ZipReader::open(&document).unwrap().names().into_iter().map(str::to_owned).collect();
    let mut entries: Vec<(String, Entry)> = archive.datasets.iter()
        .map(|(key, dataset)| (key.clone(), Entry::Dataset(ArchiveDataset { reference: dataset.reference.clone(), source: dataset.source() })))
        .collect();
    entries.sort_by_key(|(key, _)| order.iter().position(|name| *name == format!("{key}.npy")));
    let mut sections = archive.info.get("sections").unwrap().clone();
    sections.remove("dataset");
    let name = archive.info.get("name").and_then(Json::as_str).unwrap().to_owned();
    let rewritten = scratch.join("document.npz");
    let (write, _) = best(3, || write_figure_file(&rewritten, &name, &entries, &sections).unwrap());
    let same = std::fs::read(&rewritten).unwrap() == std::fs::read(&document).unwrap();
    row("document write", python("document_write"), write, format!("{} MB, byte-identical: {same}", std::fs::metadata(&document).unwrap().len() >> 20));

    // One image panel with 300 sites and their paths: save and open.
    let panel = out.join("panel").join("camera.npz");
    let fit = Arc::new(EngineFitLayer::new(Arc::new(FitEngine::default())));
    let defaults = Arc::new(Defaults::default());
    let (open, _) = best(3, || read_document(&panel.to_string_lossy(), fit.clone(), defaults.clone(), None).unwrap());
    row("panel open", python("panel_open"), open, "read_document (recipe resolved)".into());
    let archive = read_archive(&panel).unwrap();
    let plot = read_figure_plot(&archive, "data").unwrap();
    let read = SavedRead {
        source: Arc::new(Source::of_archived(&plot.data)),
        provenance: serde_json::json!({"publication": {"stream": "camera", "generation": "g", "signal": "frame", "revision": 1, "ordinal": 1},
                                       "run": {"operator": "bench"}, "event": {"shot": 1}, "parents": []}).to_string(),
    };
    let Value::Map(mut recipe) = Value::from_json(&archive.info.get("sections").unwrap().get("plot").unwrap().get("data").unwrap().to_serde()) else { unreachable!() };
    recipe.insert("overlay", Value::None);
    let image = std::fs::read(out.join("panel").join("camera.png")).unwrap();
    let target = scratch.join("panel").join("camera.png").to_string_lossy().into_owned();
    let overlay = plot.overlay.map(Arc::new);
    let (save, _) = best(3, || {
        let request = SaveRequest { path: &target, read: &read, overlay: overlay.clone() };
        FigureSaver::<SavedRead>::prepare(&ArchiveSaver, &request).unwrap().write(recipe.clone(), &mut |_format| Ok(image.clone())).unwrap()
    });
    let same = std::fs::read(scratch.join("panel").join("camera.npz")).unwrap() == std::fs::read(&panel).unwrap();
    row("panel save", python("panel_save"), save, format!("byte-identical: {same}"));

    // The ROI of an AREA over one 1200x1920 frame, per read.
    let frame = read_archive(&document).unwrap();
    let frame = &frame.dataset("panel-0").unwrap().snapshot;
    let planes = Planes::Dense(frame.segments().next().unwrap().data().clone());
    let plan = plan_from_metadata(frame.schema(), &bench["roi"].to_string()).unwrap();
    let (apply, cut) = best(7, || plan.apply_source(&planes).unwrap());
    let extent = cut.schema().physical_shape().to_vec();
    row("roi apply", python("roi_apply"), apply, format!("cut {extent:?}"));

    for line in rows { println!("{line}"); }
}
